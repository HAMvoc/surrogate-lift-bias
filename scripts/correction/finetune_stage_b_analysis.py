"""Decomposition and first corrections study, task T7 analysis of the fine-tune runs (results of scripts/correction/finetune_stage_b.py, downloaded to results/decomposition/T7/raw/{dec,full}/).
Per (variant, k, draw): seed-mean per-car error -> |bias| C_L,p, RMSE after bias removal, Rel L2, Spearman; CV+ coverage and half-width per seed (full pipeline refit per fold), averaged.
Controls on the same draws: fine-tune without stage B, frozen M2 + region_const; retrain only for draw 0 (existing runs).
Usage: python scripts/correction/finetune_stage_b_analysis.py"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.correction.calibration import build_stats, metrics  # noqa: E402
from scripts.diagnosis.regions import labeler as labeler_v2  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

RAW, OUT, RES = ROOT / "results" / "decomposition" / "T7" / "raw", ROOT / "results" / "decomposition" / "T7", ROOT / "results"
ALPHA = 0.1
CFGS = {"ft_noB": "noB", "ft_B": "B", "ft_cf_noB": "A_cf", "frozen_B": "frozen_B", "frozen": "frozen"}


def spread(e):
    return float(np.sqrt(((e - e.mean()) ** 2).mean()))


def cvplus(eps, R, fold_of_car, k):
    """eps (5, n_eval) corrected signed errors with fold j left out; R (k,) |held-out error| of each of the k cars; fold_of_car (k,). Returns coverage and mean half-width over evaluation cars."""
    lo_i, hi_i = max(1, math.floor(ALPHA * (k + 1))), min(k, math.ceil((1 - ALPHA) * (k + 1)))
    e_i = eps[fold_of_car]  # (k, n_eval)
    lo = np.sort(e_i - R[:, None], axis=0)[lo_i - 1]
    hi = np.sort(e_i + R[:, None], axis=0)[hi_i - 1]
    return float(np.mean((lo <= 0) & (hi >= 0))), float(np.mean((hi - lo) / 2))


def main():
    perm = [str(i) for i in np.random.default_rng(0).permutation(sorted(load_split("E2A")["test"]))]
    ev_ids = perm[125:]
    cfd = pd.read_csv(ROOT / "results" / "force_alignment" / "cfd_coefficients.csv").set_index("id").loc[ev_ids, "CL"].values
    files = {v: sorted((RAW / v).glob("*.npz")) if (RAW / v).exists() else [] for v in ("dec", "full")}
    rows, cvrows = [], []
    for variant, fl in files.items():
        tasks = {}
        for f in fl:
            v, s, k, d = f.stem.split("_")
            tasks[(int(k[1:]), int(d[1:]), int(s[1:]))] = np.load(f, allow_pickle=True)
        for k in sorted({t[0] for t in tasks}):
            for d in sorted({t[1] for t in tasks if t[0] == k}):
                seeds = sorted(t[2] for t in tasks if t[0] == k and t[1] == d)
                for cfg, key in CFGS.items():
                    arr = np.stack([tasks[(k, d, s)][key] for s in seeds])  # (S, n_eval, 3): dL, dD, rel
                    eL = arr[..., 0].mean(0)
                    rows.append({"variant": variant, "k": k, "draw": d, "seeds": len(seeds), "config": cfg, "bias_L": eL.mean(), "abs_bias_L": abs(eL.mean()), "rmse_L": spread(eL), "rel_l2": arr[..., 2].mean(),
                                 "abs_bias_D": abs(arr[..., 1].mean(0).mean()), "spearman": spearmanr(cfd + eL, cfd)[0]})
                # CV+ for the cross-fit pipeline (per seed)
                for s in seeds:
                    z = tasks[(k, d, s)]
                    if "cv_eps" in z.files:
                        fold_of = np.empty(k, int)
                        cars = z["cv_fold_cars"]
                        fold_of[np.argsort(cars)] = z["cv_fold_of_car"][np.argsort(cars)] if False else 0
                        # fold of each car in the order of k_idx: cv_fold_cars lists positions (0..k-1) of k_idx by fold
                        pos_fold = {int(c): int(f_) for c, f_ in zip(z["cv_fold_cars"], z["cv_fold_of_car"])}
                        fold_of = np.array([pos_fold[i] for i in range(k)])
                        Rflat = np.zeros(k)
                        for fi, rr in enumerate(z["cv_R"]):
                            members = [i for i in range(k) if fold_of[i] == fi]
                            Rflat[members] = np.abs(np.array(rr, float))
                        cov, hw = cvplus(z["cv_eps"], Rflat, fold_of, k)
                        cvrows.append({"variant": variant, "k": k, "draw": d, "seed": s, "coverage": cov, "half": hw})
    R = pd.DataFrame(rows)
    C = pd.DataFrame(cvrows)
    R.round(5).to_csv(OUT / "T7_per_draw.csv", index=False)
    C.round(5).to_csv(OUT / "T7_cvplus.csv", index=False)
    # retrain control (draw 0 only) from the saved retrain runs, same evaluation cars
    ret = []
    for k in (5, 10, 25):
        get = lambda s, sid, k=k: np.load(RES / f"E2A_M2_R2_s{s}_cal{k}" / "test_pred_fp16.npy", mmap_mode="r")[{i: j for j, i in enumerate(pd.read_csv(RES / f"E2A_M2_R2_s{s}_cal{k}" / "test_per_car.csv").id)}[sid]]
        z = build_stats(ev_ids, get, 3, labeler_v2, 8)
        m = metrics(z, np.arange(len(ev_ids)), ("none", None))
        ret.append({"k": k, "abs_bias_L": m["abs_bias_L"], "rmse_L": m["rmse_L"], "rel_l2": m["rel_l2"]})
    L = ["# Decomposition and first corrections study, task T7: balanced fine-tuning + stage B (reduced design)", "",
         "Draws, seeds and variants actually completed are in the 'n' columns; the design (PREREG Amendment 2) asked for dec: 3 seeds x draws 0-3, full: seed 0 x draws 0-2. Evaluation cars: 125 Estateback cars (permutation[125:]), none used for fitting or selection. "
         "Metrics on the seed-mean per-car error; coverage and half-width from CV+ (K = 5, full refit of fine-tune and stage B per fold; theoretical guarantee 1 - 2 alpha = 80%), computed per seed and averaged. "
         "'ft_B' = cross-fitted fine-tune + region_const (8 regions) averaged over the two cross-fit models; 'ft_noB' = one fine-tune on all k cars, no stage B; 'frozen_B' = frozen M2 + region_const on the same k cars; "
         "'retrain' = the existing run `E2A_M2_R2_s*_cal<k>` (draw 0 only, first k cars).", ""]
    if R.empty:
        (OUT / "T7_report.md").write_text("\n".join(L + ["No result files found."]), encoding="utf-8")
        return
    for variant in R.variant.unique():
        for k in (5, 10, 25):
            sub = R[(R.variant == variant) & (R.k == k)]
            if sub.empty:
                continue
            tab = []
            for cfg in CFGS:
                g = sub[sub.config == cfg]
                if g.empty:
                    continue
                fmt = lambda c, d=1: f"{g[c].mean():.{d}f} [{g[c].quantile(0.05):.{d}f}; {g[c].quantile(0.95):.{d}f}]" if len(g) > 1 else f"{g[c].mean():.{d}f}"
                row = {"config": cfg, "n draws (seeds)": f"{len(g)} ({int(g.seeds.max())})", "|bias| C_L,p": fmt("abs_bias_L"), "Rel L2": fmt("rel_l2", 4), "RMSE after bias removal": fmt("rmse_L"), "|bias| C_D,p": fmt("abs_bias_D"), "Spearman": fmt("spearman", 2)}
                if cfg == "ft_B" and not C.empty:
                    c = C[(C.variant == variant) & (C.k == k)]
                    row["CV+ coverage"], row["CV+ half-width"] = (f"{c.coverage.mean():.3f}", f"{c.half.mean():.0f}") if len(c) else ("-", "-")
                tab.append(row)
            if k in [r["k"] for r in ret]:
                r = [x for x in ret if x["k"] == k][0]
                tab.append({"config": "retrain (draw 0)", "n draws (seeds)": "1 (3)", "|bias| C_L,p": f"{r['abs_bias_L']:.1f}", "Rel L2": f"{r['rel_l2']:.4f}", "RMSE after bias removal": f"{r['rmse_L']:.1f}"})
            keys = ["config", "n draws (seeds)", "|bias| C_L,p", "Rel L2", "RMSE after bias removal", "|bias| C_D,p", "Spearman", "CV+ coverage", "CV+ half-width"]
            L += [f"## variant {variant}, k = {k} (mean [5%; 95%] over draws)", "", md([{kk: r.get(kk, "-") for kk in keys} for r in tab]), ""]
        # thresholds at k = 10
        sub = R[(R.variant == variant) & (R.k == 10)]
        if len(sub):
            g = sub[sub.config == "ft_B"]
            gf = sub[sub.config == "frozen_B"]
            c = C[(C.variant == variant) & (C.k == 10)] if not C.empty else C
            checks = {"|bias| <= 28": g.abs_bias_L.mean() <= 28, "Rel L2 <= 0.175": g.rel_l2.mean() <= 0.175}
            if len(c):
                checks["CV+ coverage 87-93%"] = 0.87 <= c.coverage.mean() <= 0.93
                checks["half-width < 140"] = c.half.mean() < 140
            checks["RMSE not worse than frozen + region_const"] = g.rmse_L.mean() <= gf.rmse_L.mean() + 1e-9
            L += [f"## Registered thresholds, variant {variant}, k = 10 (ft_B, mean over draws and seeds)", "", md([{"check": kk, "result": "pass" if v else "FAIL"} for kk, v in checks.items()]), "",
                  f"values: |bias| {g.abs_bias_L.mean():.1f}, Rel L2 {g.rel_l2.mean():.4f}, RMSE {g.rmse_L.mean():.1f} (frozen + region_const {gf.rmse_L.mean():.1f})" + (f", CV+ coverage {c.coverage.mean():.3f}, half-width {c.half.mean():.0f}" if len(c) else ""), ""]
    (OUT / "T7_report.md").write_text("\n".join(L), encoding="utf-8")
    print("report written")


if __name__ == "__main__":
    main()
