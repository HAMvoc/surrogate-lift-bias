"""Two-stage correction study, task T1: two-stage C2 with out-of-fold stage B.
One task = (variant, seed, draw, k): K = 5 fold fine-tunes (stage A on k minus one fold) -> out-of-fold predictions of all k cars -> stage B (region_const_intmatch, 8 regions, lambda by LOO inside the OOF residuals)
fitted on the OOF residuals; final stage A = fine-tune on all k cars; nested CV+ pieces (stage B refitted without fold j). Writes <out>/<variant>_s<seed>_k<k>_d<draw>.npz.
Run with PYTHONHASHSEED=0 (the fine-tune RNG tag uses hash(str)).
Usage: python scripts/correction/two_stage_oof.py --variant dec --seeds 0 --draws 0 1 2 3 4 --ks 5 10 25 --out DIR --ckpt-dir DIR [--smoke]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.correction import finetune_stage_b as p2  # noqa: E402
from scripts.correction.calibration import build_stats, corrected, fit_const  # noqa: E402
from scripts.diagnosis.regions import labeler as labeler_v2  # noqa: E402
from src.data.dataset import CarSet, load_split  # noqa: E402

LAMS = (0.0, 1.0, 10.0)
K = 5


def pick_lambda(z, idx):
    """lambda in LAMS by leave-one-out over the cars idx of the OOF stats z: minimise mean(dL'^2 + dD'^2) among lambdas whose LOO field Rel L2 is <= 1.02 x that of lambda = 0."""
    idx = np.asarray(idx)
    sc, rl = [], []
    for lam in LAMS:
        s, r = [], []
        for a in idx:
            c = fit_const(z, idx[idx != a], lam)
            dL, dD, rel = corrected(z, np.array([a]), ("off", c))
            s.append(float(dL[0, 0] ** 2 + dD[0, 0] ** 2))
            r.append(float(rel[0, 0]))
        sc.append(np.mean(s))
        rl.append(np.mean(r))
    ok = [i for i in range(len(LAMS)) if rl[i] <= 1.02 * rl[0]]
    return LAMS[min(ok, key=lambda i: sc[i])]


def fit_b(z, idx):
    lam = pick_lambda(z, idx)
    return lam, fit_const(z, np.asarray(idx), lam)[0]  # offsets (R,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["dec", "full"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--draws", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 25])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default=str(ROOT / "ckpt"))
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="2 steps, 3 evaluation cars")
    ap.add_argument("--swap", action="store_true", help="Swapped-halves study: swapped halves (pool = perm[125:], evaluation = perm[:125], draws default_rng(1100 + d)); default = the original halves")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    s = load_split("E2A")
    perm = [str(i) for i in np.random.default_rng(0).permutation(sorted(s["test"]))]
    cand_ids, ev_ids = (perm[125:], perm[:125]) if a.swap else (perm[:125], perm[125:])
    draw_seed = 1100 if a.swap else 100
    src_ids = [str(i) for i in np.random.default_rng(1).choice(sorted(s["train"]), 100, replace=False)]
    val_ids = list(s["val"])[:20]
    if a.smoke:
        ev_ids = ev_ids[:3]
    print("loading cars ...", flush=True)
    kdata, ev, src, val = CarSet(cand_ids, "R2", "sdf"), CarSet(ev_ids, "R2", "sdf"), CarSet(src_ids, "R2", "sdf"), CarSet(val_ids, "R2", "sdf")
    cars_k, cars_ev = p2.Cars(kdata), p2.Cars(ev)
    vpts = [torch.from_numpy(np.random.default_rng(100 + j).permutation(100000)[:16384]) for j in range(len(val))]
    models = {sd: p2.load_model(p2.find_ckpt(a.ckpt_dir, sd), dev) for sd in a.seeds}
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    steps, every = (2, 1) if a.smoke else (a.steps, a.every)
    nev = len(ev)
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
                model0, mu, sd = models[seed]
                k_idx = [int(i) for i in np.random.default_rng(draw_seed + draw).permutation(125)[:k]]
                folds = np.array_split(np.random.default_rng(3000 + 100 * draw + k).permutation(k), K)  # positions in k_idx
                tag = f"{a.variant}{seed}{k}{draw}"
                # stage A: fold models and the full model
                fold_models, sel_f, oof = [], [], {}
                for fi, fold in enumerate(folds):
                    hold = [k_idx[i] for i in fold]
                    tr = [j for j in k_idx if j not in hold]
                    m, sel, base = p2.finetune(model0, mu, sd, kdata, tr, src, val, vpts, a.variant, seed, dev, steps, every, f"{tag}f{fi}")
                    fold_models.append(m)
                    sel_f.append(sel)
                    oof.update(p2.predict(m, kdata, hold, mu, sd, dev))
                m_all, sel_all, _ = p2.finetune(model0, mu, sd, kdata, k_idx, src, val, vpts, a.variant, seed, dev, steps, every, tag + "all")
                # stage B on the OOF residuals (stats of the k cars, one 'seed' = the OOF prediction)
                z = build_stats([cand_ids[j] for j in k_idx], lambda _s, sid: oof[cand_ids.index(sid)], 1, labeler_v2, p2.NREG)
                pos_fold = np.empty(k, int)
                for fi, fold in enumerate(folds):
                    pos_fold[fold] = fi
                lam_all, c_all = fit_b(z, np.arange(k))
                fits = []
                for fi in range(K):
                    keep = np.where(pos_fold != fi)[0]
                    fits.append(fit_b(z, keep) if len(keep) >= 2 else (1.0, fit_const(z, keep, 1.0)[0]))
                # CV+ scores of the k cars: |corrected error| of the fold-j model with B_j
                R = np.zeros(k)
                for i in range(k):
                    R[i] = abs(corrected(z, np.array([i]), ("off", fits[pos_fold[i]][1][None]))[0][0, 0])
                # consistency: coefficient of the corrected field equals the one from the sufficient statistics
                cons = max(abs(cars_k.metrics(k_idx[i], cars_k.correct(k_idx[i], oof[k_idx[i]], c_all))[0] - corrected(z, np.array([i]), ("off", c_all[None]))[0][0, 0]) for i in range(k))
                # evaluation cars: all models, one car at a time
                rec = {n: np.zeros((nev, 3)) for n in ("noB_full", "B_full", "noB_avg", "B_avg")}
                eps = np.zeros((K, nev))
                for j in range(nev):
                    b = ev.batch([j])
                    with torch.no_grad():
                        p_full = p2.fwd(m_all.eval(), b, dev)[0].cpu().numpy().astype(np.float64) * sd + mu
                        p_f = [p2.fwd(m.eval(), b, dev)[0].cpu().numpy().astype(np.float64) * sd + mu for m in fold_models]
                    p_avg = np.mean(p_f, axis=0)
                    rec["noB_full"][j] = cars_ev.metrics(j, p_full)
                    rec["B_full"][j] = cars_ev.metrics(j, cars_ev.correct(j, p_full, c_all))
                    rec["noB_avg"][j] = cars_ev.metrics(j, p_avg)
                    rec["B_avg"][j] = cars_ev.metrics(j, cars_ev.correct(j, p_avg, c_all))
                    for fi in range(K):
                        eps[fi, j] = cars_ev.metrics(j, cars_ev.correct(j, p_f[fi], fits[fi][1]))[0]
                oof_t = np.array([[corrected(z, np.array([i]), ("none", None))[0][0, 0], corrected(z, np.array([i]), ("none", None))[1][0, 0]] for i in range(k)])
                np.savez(f, **rec, eps=eps, cv_R=R, cv_fold_of=pos_fold, oof=oof_t, lam_all=lam_all, lam_f=np.array([x[0] for x in fits]), c_all=c_all, sel_all=sel_all, sel_f=np.array(sel_f), base_val=base,
                         k_idx=np.array(k_idx), cons=cons)
                print(f"{f.name}: sel all {sel_all}, folds {sel_f}, lambda {lam_all}, cons {cons:.2e}, base val {base:.4f}, {time.time() - tt:.0f} s (total {(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
