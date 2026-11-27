"""Label-budget study, task T3: B-same versus B-OOF on the same draws. For each draw (k = 10, seed 0, variant full): fine-tune on all k cars (weights SAVED), then stage B
(region_const_intmatch, 8 regions v2, lambda by LOO) fitted on (i) the in-sample residuals of that model on the k cars (B-same) and (ii) the out-of-fold offsets of the two-stage correction study for the same draw and seed
(data/manifests/stage_b_oof_offsets.json, from results/two_stage/T1/raw/full_s0_k10_d*.npz `c_all`). Both are applied to the new final model on the 125 evaluation cars.
Also: fine-tune without stage B (reproducibility check against the two-stage correction study `noB_full`), jackknife+ coverage of B-same (stage B refitted without each car, residuals in-sample).
Run with PYTHONHASHSEED=0. Usage: python scripts/correction/oof_vs_in_sample_finetune.py --ckpt-dir DIR --out DIR [--draws 0 1 ...] [--smoke]"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.correction import finetune_stage_b as p2  # noqa: E402
from scripts.correction.calibration import build_stats, corrected, fit_const, metrics  # noqa: E402
from scripts.correction.two_stage_oof import fit_b, pick_lambda  # noqa: E402
from scripts.diagnosis.regions import labeler as labeler_v2  # noqa: E402
from src.data.dataset import CarSet, load_split  # noqa: E402

K = 10
ALPHA = 0.1


def jackknife_same(z_same, idx, z_eval, ev):
    """jackknife+ (90%): stage B refitted without each of the k cars on the in-sample residuals; R_i = |corrected C_L,p error of the left-out car| (in-sample)."""
    k = len(idx)
    eps, R = np.zeros((k, len(ev))), np.zeros(k)
    for a in range(k):
        sub = np.delete(idx, a)
        lam = pick_lambda(z_same, sub) if len(sub) >= 5 else 1.0
        c = fit_const(z_same, sub, lam)[0]
        eps[a] = corrected(z_eval, ev, ("off", c[None]))[0].mean(0)
        R[a] = abs(corrected(z_same, idx[a:a + 1], ("off", c[None]))[0][0, 0])
    lo_i, hi_i = max(1, math.floor(ALPHA * (k + 1))), min(k, math.ceil((1 - ALPHA) * (k + 1)))
    lo, hi = np.sort(eps - R[:, None], axis=0)[lo_i - 1], np.sort(eps + R[:, None], axis=0)[hi_i - 1]
    return float(np.mean((lo <= 0) & (hi >= 0))), float(np.mean((hi - lo) / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(ROOT / "ckpt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--offsets", default=str(ROOT / "data" / "manifests" / "stage_b_oof_offsets.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="2 steps, 3 evaluation cars")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    s = load_split("E2A")
    perm = [str(i) for i in np.random.default_rng(0).permutation(sorted(s["test"]))]
    cand_ids, ev_ids = perm[:125], perm[125:]
    src_ids = [str(i) for i in np.random.default_rng(1).choice(sorted(s["train"]), 100, replace=False)]
    val_ids = list(s["val"])[:20]
    if a.smoke:
        ev_ids = ev_ids[:3]
    offs = json.load(open(a.offsets))
    print("loading cars ...", flush=True)
    kdata, ev, src, val = CarSet(cand_ids, "R2", "sdf"), CarSet(ev_ids, "R2", "sdf"), CarSet(src_ids, "R2", "sdf"), CarSet(val_ids, "R2", "sdf")
    vpts = [torch.from_numpy(np.random.default_rng(100 + j).permutation(100000)[:16384]) for j in range(len(val))]
    model0, mu, sd = p2.load_model(p2.find_ckpt(a.ckpt_dir, a.seed), dev)
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    steps, every = (2, 1) if a.smoke else (a.steps, a.every)
    evi = np.arange(len(ev_ids))
    rows = []
    for d in a.draws:
        f = out / f"p5_s{a.seed}_k{K}_d{d}.json"
        if f.exists():
            continue
        tt = time.time()
        k_idx = [int(i) for i in np.random.default_rng(100 + d).permutation(125)[:K]]
        tag = f"full{a.seed}{K}{d}all"  # same RNG tag as the two-stage correction study (final model, no stage B)
        m_all, sel, base = p2.finetune(model0, mu, sd, kdata, k_idx, src, val, vpts, "full", a.seed, dev, steps, every, tag)
        torch.save({n: p.detach().cpu() for n, p in m_all.state_dict().items()}, out / f"ft_full_s{a.seed}_k{K}_d{d}.pt")
        pin = p2.predict(m_all, kdata, k_idx, mu, sd, dev)  # in-sample predictions on the k cars
        z_same = build_stats([cand_ids[j] for j in k_idx], lambda _s, sid: pin[cand_ids.index(sid)], 1, labeler_v2, p2.NREG)
        lam_same, c_same = fit_b(z_same, np.arange(K))
        pe = p2.predict(m_all, ev, range(len(ev_ids)), mu, sd, dev)
        z_eval = build_stats(ev_ids, lambda _s, sid: pe[ev_ids.index(sid)], 1, labeler_v2, p2.NREG)
        c_oof = np.array(offs[str(d)]["c_all"])
        res = {"draw": d, "k_idx": k_idx, "sel_steps": sel, "base_val": base, "lam_same": lam_same, "lam_oof": offs[str(d)]["lam_all"], "c_same": c_same.tolist(), "c_oof": c_oof.tolist()}
        for name, fit in (("noB", ("none", None)), ("B_same", ("off", c_same[None])), ("B_OOF", ("off", c_oof[None]))):
            m = metrics(z_eval, evi, fit)
            res[name] = {kk: float(m[kk]) for kk in ("bias_L", "abs_bias_L", "abs_bias_D", "rmse_L", "rel_l2", "spearman")}
        # in-sample residual summary (what stage B sees): SD and mean of the in-sample C_L,p error on the k cars
        eL_in = z_same["dL"][0]
        res["in_sample_dL_mean"], res["in_sample_dL_sd"] = float(eL_in.mean()), float(eL_in.std(ddof=1))
        res["cov_same"], res["half_same"] = jackknife_same(z_same, np.arange(K), z_eval, evi)
        json.dump(res, open(f, "w"), indent=1)
        print(f"d{d}: sel {sel}, base val {base:.4f}, noB |bias| {res['noB']['abs_bias_L']:.1f}, B_same {res['B_same']['abs_bias_L']:.1f}, B_OOF {res['B_OOF']['abs_bias_L']:.1f}, cov_same {res['cov_same']:.2f}, {time.time() - tt:.0f} s (total {(time.time() - t0) / 60:.0f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
