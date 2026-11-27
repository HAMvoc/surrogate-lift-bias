"""Two-stage correction study, task T2 (frozen part) and T3 (frozen curve): frozen M2 (3 seeds) + none / scalar_shift / region_const / region_const_intmatch (lambda by LOO) on the shared draws.
Draw d, k: cand[default_rng(100 + d).permutation(125)[:k]]; evaluation = perm[125:] (125 cars). d = 0..49 (the first 10 are the shared T1/T2 draws), k in {1,2,3,5,10,25}.
Output: results/two_stage/T2/frozen_draws.csv (one row per method, k, draw) and frozen_eL.npz (seed-mean per-car lift error per method, k, draw). Usage: python scripts/correction/frozen_draws.py"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.correction.calibration import build_stats, corrected, fit_const, take  # noqa: E402
from scripts.diagnosis.regions import labeler as labeler_v2  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

OUT = ROOT / "results" / "two_stage" / "T2"
LAMS = (0.0, 1.0, 10.0)
KS = (1, 2, 3, 5, 10, 25)
ALPHA = 0.1


def seed_z(z, s):
    return {k: (v[s:s + 1] if k in ("Aw", "Au", "Qu", "dL", "dD") else v) for k, v in z.items()}


def pick_lambda(zs, idx):
    """PREREG: k < 5 -> lambda = 1; otherwise LOO: min mean(dL'^2 + dD'^2) among lambdas whose LOO Rel L2 <= 1.02 x that of lambda = 0."""
    idx = np.asarray(idx)
    if len(idx) < 5:
        return 1.0
    sc, rl = [], []
    for lam in LAMS:
        s, r = [], []
        for a in idx:
            dL, dD, rel = corrected(zs, np.array([a]), ("off", fit_const(zs, idx[idx != a], lam)))
            s.append(dL[0, 0] ** 2 + dD[0, 0] ** 2); r.append(rel[0, 0])
        sc.append(np.mean(s)); rl.append(np.mean(r))
    ok = [i for i in range(3) if rl[i] <= 1.02 * rl[0]]
    return LAMS[min(ok, key=lambda i: sc[i])]


def fit_method(z, idx, method):
    """-> ('none'|'shift'|'off', payload) with per-seed offsets (S, R) or shifts."""
    S = z["dL"].shape[0]
    if method == "none":
        return ("none", None)
    if method == "scalar_shift":
        return ("shift", (z["dL"][:, idx].mean(1), z["dD"][:, idx].mean(1)))
    if method == "region_const":
        return ("off", fit_const(z, idx, 0.0))
    cs = []
    for s in range(S):
        zs = seed_z(z, s)
        cs.append(fit_const(zs, idx, pick_lambda(zs, idx))[0])
    return ("off", np.array(cs))


def jackknife(z, idx, ev, method):
    k = len(idx)
    eps, Rr = np.zeros((k, len(ev))), np.zeros(k)
    for a in range(k):
        fit = fit_method(z, np.delete(idx, a), method)
        eps[a] = corrected(z, ev, fit)[0].mean(0)
        Rr[a] = abs(corrected(z, idx[a:a + 1], fit)[0].mean(0)[0])
    lo_i, hi_i = max(1, math.floor(ALPHA * (k + 1))), min(k, math.ceil((1 - ALPHA) * (k + 1)))
    lo, hi = np.sort(eps - Rr[:, None], axis=0)[lo_i - 1], np.sort(eps + Rr[:, None], axis=0)[hi_i - 1]
    return float(np.mean((lo <= 0) & (hi >= 0))), float(np.mean((hi - lo) / 2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    perm = [str(i) for i in np.random.default_rng(0).permutation(sorted(load_split("E2A")["test"]))]
    preds = [np.load(ROOT / "results" / f"E2A_M2_R2_s{s}" / "test_pred_fp16.npy", mmap_mode="r") for s in range(3)]
    pos = [{i: j for j, i in enumerate(pd.read_csv(ROOT / "results" / f"E2A_M2_R2_s{s}" / "test_per_car.csv").id)} for s in range(3)]
    z = build_stats(perm, lambda s, sid: preds[s][pos[s][sid]], 3, labeler_v2, 8)
    ev = np.arange(125, 250)
    gate = z["dL"][:, :].mean(0).mean()
    assert abs(gate - 183.9) <= 0.5, gate
    cfd = z["CL"][ev]
    rows, eLs = [], {}
    for d in range(50):
        for k in KS:
            idx = np.random.default_rng(100 + d).permutation(125)[:k]
            for m in ("none", "scalar_shift", "region_const", "intmatch_lam"):
                if m == "none" and (k != KS[0] or d):
                    continue
                fit = fit_method(z, idx, m)
                dL, dD, rel = corrected(z, ev, fit)
                eL, eD = dL.mean(0), dD.mean(0)
                row = {"method": m, "k": k, "draw": d, "bias_L": eL.mean(), "abs_bias_L": abs(eL.mean()), "bias_D": eD.mean(), "abs_bias_D": abs(eD.mean()), "rmse_L": np.sqrt(((eL - eL.mean()) ** 2).mean()),
                       "rel_l2": rel.mean(), "spearman": spearmanr(cfd + eL, cfd)[0], "slope": np.polyfit(cfd, cfd + eL, 1)[0]}
                if m != "none" and k >= 2:
                    row["coverage"], row["half"] = jackknife(z, idx, ev, m)
                rows.append(row)
                eLs[f"{m}|{k}|{d}"] = eL
    R = pd.DataFrame(rows)
    R.round(5).to_csv(OUT / "frozen_draws.csv", index=False)
    np.savez_compressed(OUT / "frozen_eL.npz", **eLs)
    print(R[R.draw < 10].groupby(["method", "k"])[["abs_bias_L", "rel_l2", "rmse_L", "abs_bias_D", "coverage"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
