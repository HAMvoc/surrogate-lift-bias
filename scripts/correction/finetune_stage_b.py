"""Decomposition and first corrections study, task T7: balanced fine-tuning of the E2A M2 checkpoints on k Estateback cars + stage B (region_const, 8 regions) with cross-fitting, plus the pieces needed for CV+ (K = 5).
One task = (variant, seed, k, draw); results are written per task to <out>/<variant>_s<seed>_k<k>_d<draw>.npz.
Variants: dec (decoder MLP only, lr 3e-4) and full (whole network, lr 3e-5). Number of steps = the largest step (checked every `--every` steps) at which the Rel L2 on source validation cars
(F/N cars, 16,384 fixed points each) is within +0.005 of the checkpoint; no Estateback label is used for that choice.
Usage: python scripts/correction/finetune_stage_b.py --variant dec --seeds 0 1 2 --draws 0 1 2 3 --ks 5 10 25 --out DIR --hours 4.5 [--device cuda]"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.pressure_coefficients import CLIP, GEO, pool_weights  # noqa: E402
from scripts.diagnosis.force_alignment import car_terms  # noqa: E402
from scripts.diagnosis.regions import labeler as labeler_v2  # noqa: E402
from src.data.dataset import CACHE, CarSet, load_split  # noqa: E402
from src.models.m2 import M2  # noqa: E402

NREG = 8
LR = {"dec": 3e-5, "full": 1e-5}  # Amendment 2 of PREREG (dec 3e-4 and full 3e-5 forgot the source cars within 20 steps)
VERBOSE = False


def find_ckpt(ckpt_dir, seed):
    cands = [Path(ckpt_dir) / f"E2A_M2_R2_s{seed}" / "best.pt", Path(ckpt_dir) / f"s{seed}" / "best.pt"]
    cands += [ROOT / "kaggle" / "_out" / ("voxel-sdf-e2-s0" if seed == 0 else "voxel-sdf-e2-s12") / "runs" / f"E2A_M2_R2_s{seed}" / "best.pt"]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(cands)


def load_model(path, dev):
    c = torch.load(path, map_location=dev, weights_only=False)
    m = c["cfg"]["model_cfg"]
    model = M2(chs=tuple(m["chs"]), in_ch=1, proj=m["proj"], glob=m["glob"], fourier=0).to(dev)
    model.load_state_dict(c["ema"])
    return model, float(c["mu"]), float(c["sd"])


def fwd(model, b, dev):
    with torch.autocast("cuda", dtype=torch.float16, enabled=str(dev).startswith("cuda")):
        return model(b["vol"].to(dev), b["g"].to(dev), b["ndc"].to(dev), b["nrm"].to(dev)).float()


def val_rel(model, val, pts, mu, sd, dev):
    model.eval()
    r = []
    with torch.no_grad():
        for j in range(len(val)):
            b = val.batch([j], pts=pts[j])
            p = fwd(model, b, dev)[0].cpu() * sd + mu
            cp = b["cp"][0]
            r.append(float((p - cp).norm() / cp.norm()))
    return float(np.mean(r))


def finetune(model0, mu, sd, tgt, tgt_idx, src, val, vpts, variant, seed, dev, steps, every, tag=""):
    """Returns (model at the selected step, selected step, validation Rel L2 at step 0)."""
    m = copy.deepcopy(model0)
    train_params = [p for n, p in m.named_parameters() if (variant == "full" or n.startswith("mlp"))]
    for p in m.parameters():
        p.requires_grad = False
    for p in train_params:
        p.requires_grad = True
    opt = torch.optim.AdamW(train_params, lr=LR[variant], weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=str(dev).startswith("cuda"))
    rng = np.random.default_rng([seed, abs(hash(tag)) % 100000])
    gen = torch.Generator().manual_seed(int(rng.integers(1 << 30)))
    base = val_rel(m, val, vpts, mu, sd, dev)
    sel, sel_state = 0, None
    keys = [n for n, _ in m.named_parameters() if (variant == "full" or n.startswith("mlp"))]
    for step in range(1, steps + 1):
        m.train()
        opt.zero_grad(set_to_none=True)
        micro = [(tgt, list(rng.choice(tgt_idx, 2))) for _ in range(2)] + [(src, list(rng.choice(len(src), 2))) for _ in range(2)]  # 4 target + 4 source cars
        for data, idx in micro:
            b = data.batch(idx, 32768, gen)
            out = fwd(m, b, dev)
            loss = F.huber_loss(out, ((b["cp"] - mu) / sd).to(dev), delta=1.0)
            scaler.scale(loss / len(micro)).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        scaler.step(opt)
        scaler.update()
        if step % every == 0:
            v = val_rel(m, val, vpts, mu, sd, dev)
            if VERBOSE:
                print(f"    {tag} step {step} loss {float(loss):.4f} val {v:.4f} (base {base:.4f})", flush=True)
            if v <= base + 0.005:
                sel = step
                sd_ = dict(m.named_parameters())
                sel_state = {k: sd_[k].detach().cpu().clone() for k in keys}
            elif v > base + 0.03:
                break
    m0 = copy.deepcopy(model0)
    if sel_state is not None:
        with torch.no_grad():
            pr = dict(m0.named_parameters())
            for k, v in sel_state.items():
                pr[k].copy_(v.to(dev))
    return m0, sel, base


def predict(model, data, idxs, mu, sd, dev):
    model.eval()
    out = {}
    with torch.no_grad():
        for j in idxs:
            out[j] = (fwd(model, data.batch([j]), dev)[0].cpu().numpy().astype(np.float64) * sd + mu)
    return out


class Cars:
    """Per-car arrays of a CarSet needed for stage B and the metrics (labels v2, IPW weights)."""

    def __init__(self, data):
        self.data = data
        self.xyz, self.nrm, self.cp, self.area = (data.xyz.numpy(), data.nrm.numpy(), data.cp_raw.numpy(), data.area.numpy())
        self.lab = [labeler_v2(sid, self.xyz[j], self.nrm[j]) for j, sid in enumerate(data.ids)]
        self.ref = np.clip(self.cp.astype(np.float64), *CLIP)
        self.w = [pool_weights(self.area[j].astype(np.float64), sid) for j, sid in enumerate(data.ids)]

    def region_sums(self, j, pred):
        e = pred - self.ref[j]
        lab, w = self.lab[j], self.w[j]
        bc = lambda v: np.bincount(lab, weights=v, minlength=NREG + 1)[:NREG]
        return bc(w * e), bc(w), w.sum()

    def correct(self, j, pred, c):
        return pred - c[self.lab[j]]

    def metrics(self, j, pred):
        t = car_terms(pred, self.cp[j], self.nrm[j], self.area[j], self.data.ids[j])
        return t["dL"], t["dD"], float(np.linalg.norm(pred - self.ref[j]) / np.linalg.norm(self.ref[j]))


def fit_region_const(cars, idxs, preds):
    A, W, Wt = zip(*[cars.region_sums(j, preds[j]) for j in idxs])
    A, W, Wt = np.array(A), np.array(W), np.array(Wt)[:, None]
    return (A / Wt).mean(0) / np.maximum((W / Wt).mean(0), 1e-12)


def pipeline(models, cars_k, k_idx, ev, evc, kdata, vdata, src, val, vpts, variant, seed, dev, steps, every, tag):
    """Cross-fitted fine-tune + stage B on the cars k_idx of `kdata`. Returns per-evaluation-car arrays (125) of the corrected field metrics and the fold-held-out predictions' stage."""
    model0, mu, sd = models
    k_idx = list(k_idx)
    h1, h2 = k_idx[0::2], k_idx[1::2]
    res = {}
    ms = []
    sels = []
    for h, name in ((h1, "h1"), (h2, "h2")):
        mm, sel, base = finetune(model0, mu, sd, kdata, h, src, val, vpts, variant, seed, dev, steps, every, tag + name)
        ms.append(mm)
        sels.append(sel)
    # stage B: model 1 residuals on half 2 and model 2 residuals on half 1
    p_h2 = predict(ms[0], kdata, h2, mu, sd, dev)
    p_h1 = predict(ms[1], kdata, h1, mu, sd, dev)
    c1, c2 = fit_region_const(cars_k, h2, p_h2), fit_region_const(cars_k, h1, p_h1)
    pe1, pe2 = predict(ms[0], ev, range(len(ev)), mu, sd, dev), predict(ms[1], ev, range(len(ev)), mu, sd, dev)
    return ms, (c1, c2), (pe1, pe2), sels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["dec", "full"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--draws", type=int, nargs="+", default=[0])
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 25])
    ap.add_argument("--cvplus-k", type=int, nargs="*", default=[10, 25], help="k values with the CV+ (K = 5) refits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default=str(ROOT / "ckpt"))
    ap.add_argument("--hours", type=float, default=4.5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lr", type=float, help="override the learning rate of the variant")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 2 steps, 2 evaluation cars")
    a = ap.parse_args()
    global VERBOSE
    VERBOSE = a.verbose
    if a.lr:
        LR[a.variant] = a.lr
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    s = load_split("E2A")
    perm = [str(i) for i in np.random.default_rng(0).permutation(sorted(s["test"]))]
    cand_ids, ev_ids = perm[:125], perm[125:]
    rng0 = np.random.default_rng(1)
    src_ids = [str(i) for i in rng0.choice(sorted(s["train"]), 100, replace=False)]
    val_ids = list(s["val"])[:20]
    if a.smoke:
        ev_ids = ev_ids[:2]
    print("loading cars ...", flush=True)
    kdata, ev, src, val = CarSet(cand_ids, "R2", "sdf"), CarSet(ev_ids, "R2", "sdf"), CarSet(src_ids, "R2", "sdf"), CarSet(val_ids, "R2", "sdf")
    cars_k, cars_ev = Cars(kdata), Cars(ev)
    vpts = [torch.from_numpy(np.random.default_rng(100 + j).permutation(100000)[:16384]) for j in range(len(val))]
    models = {sd_: load_model(find_ckpt(a.ckpt_dir, sd_), dev) for sd_ in a.seeds}
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    steps = 2 if a.smoke else a.steps
    every = 1 if a.smoke else a.every
    # frozen predictions on the evaluation cars and the candidate cars, per seed
    frozen_ev = {sd_: predict(models[sd_][0], ev, range(len(ev)), models[sd_][1], models[sd_][2], dev) for sd_ in a.seeds}
    frozen_k = {sd_: predict(models[sd_][0], kdata, range(len(kdata)), models[sd_][1], models[sd_][2], dev) for sd_ in a.seeds}
    for draw in a.draws:
        for k in a.ks:
            for seed in a.seeds:
                if (time.time() - t0) / 3600 > a.hours:
                    print("time budget reached, stopping", flush=True)
                    return
                f = out / f"{a.variant}_s{seed}_k{k}_d{draw}.npz"
                if f.exists():
                    continue
                tt = time.time()
                k_idx = list(range(k)) if draw == 0 else list(np.random.default_rng(2000 + 100 * draw + k).permutation(125)[:k])
                mdl = models[seed]
                tag = f"{a.variant}{seed}{k}{draw}"
                rec = {}
                # fine-tune on all k cars, no stage B
                m_all, sel_all, base = finetune(mdl[0], mdl[1], mdl[2], kdata, k_idx, src, val, vpts, a.variant, seed, dev, steps, every, tag + "all")
                pe = predict(m_all, ev, range(len(ev)), mdl[1], mdl[2], dev)
                rec["noB"] = np.array([cars_ev.metrics(j, pe[j]) for j in range(len(ev))])
                # cross-fitted fine-tune + stage B
                ms, (c1, c2), (pe1, pe2), sels = pipeline(mdl, cars_k, k_idx, ev, cars_ev, kdata, val, src, val, vpts, a.variant, seed, dev, steps, every, tag)
                rec["B"] = np.array([cars_ev.metrics(j, 0.5 * (cars_ev.correct(j, pe1[j], c1) + cars_ev.correct(j, pe2[j], c2))) for j in range(len(ev))])
                rec["A_cf"] = np.array([cars_ev.metrics(j, 0.5 * (pe1[j] + pe2[j])) for j in range(len(ev))])  # cross-fit models averaged, no correction
                # frozen + region_const fitted on the same k cars
                cf = fit_region_const(cars_k, k_idx, frozen_k[seed])
                rec["frozen_B"] = np.array([cars_ev.metrics(j, cars_ev.correct(j, frozen_ev[seed][j], cf)) for j in range(len(ev))])
                rec["frozen"] = np.array([cars_ev.metrics(j, frozen_ev[seed][j]) for j in range(len(ev))])
                extra = {"sel_all": sel_all, "sel_h": np.array(sels), "base_val": base, "k_idx": np.array(k_idx)}
                # CV+ pieces (K = 5)
                if k in a.cvplus_k:
                    folds = np.array_split(np.random.default_rng(3000 + 100 * draw + k).permutation(k), 5)
                    eps, Rr = [], []
                    for fi, fold in enumerate(folds):
                        keep = [k_idx[i] for i in range(k) if i not in set(fold)]
                        msj, (d1, d2), (q1, q2), _ = pipeline(mdl, cars_k, keep, ev, cars_ev, kdata, val, src, val, vpts, a.variant, seed, dev, steps, every, tag + f"cv{fi}")
                        eps.append([cars_ev.metrics(j, 0.5 * (cars_ev.correct(j, q1[j], d1) + cars_ev.correct(j, q2[j], d2)))[0] for j in range(len(ev))])
                        hold = [k_idx[i] for i in fold]
                        h1 = predict(msj[0], kdata, hold, mdl[1], mdl[2], dev)
                        h2 = predict(msj[1], kdata, hold, mdl[1], mdl[2], dev)
                        Rr.append([cars_k.metrics(j, 0.5 * (cars_k.correct(j, h1[j], d1) + cars_k.correct(j, h2[j], d2)))[0] for j in hold])
                    extra["cv_eps"] = np.array(eps)  # (5, n_eval) corrected signed error with fold j left out
                    extra["cv_R"] = np.array([r for r in Rr], dtype=object)
                    extra["cv_fold_of_car"] = np.concatenate([[fi] * len(fo) for fi, fo in enumerate(folds)])
                    extra["cv_fold_cars"] = np.concatenate(folds)
                np.savez(f, **{kk: v for kk, v in rec.items()}, **extra, allow_pickle=True)
                print(f"{f.name}: sel steps all {sel_all}, halves {sels}, base val {base:.4f}, {time.time() - tt:.0f} s (total {(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
