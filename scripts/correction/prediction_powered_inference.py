"""Prediction-powered inference study: prediction-powered inference (PPI) interval for the mean C_L,p of the new body type against the classical interval of the k labelled cars.
Cells, cars, draws and predictions are those of the reference-correction study (orig / swap / indep). y = CFD C_L,p (counts), p = seed-mean prediction = y + mean_seed dL, r = y - p.
Estimand mu = mean y of the evaluation cars. Classical: mean_lab(y) +- t * s / sqrt(k). PPI: mean_eval(p) + mean_lab(r) +- t * sqrt(var_eval(p)/N + var_lab(r)/k). t = t_{0.95, k-1}, 90% intervals.
Self-check: |PPI point - mu| equals the stored |bias| C_L,p of scalar_shift (results/reference_correction/shift_draws.csv). Output: results/ppi/{ppi_draws.csv, ppi_cells.csv, verdict.md}. Usage: python scripts/correction/prediction_powered_inference.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as tdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.correction.swapped_halves import PAIRS, stats  # noqa: E402
from scripts.training.infer_cars import FAMS  # noqa: E402
from scripts.correction.reference_vs_stage_b import indep_stats  # noqa: E402

OUT = ROOT / "results" / "ppi"
KS = (5, 10, 25)
NEED = 16  # 2/3 of 23 cells, PREREG
COV_MIN, R_MAX = 0.80, 0.90


def run_cell(setting, fold, fam, z, pool, ev, seed0, stored, rows):
    y = np.asarray(z["CL"], float)
    p = y + z["dL"].mean(0)
    mu, pe, N = y[ev].mean(), p[ev], len(ev)
    for d in range(50):
        perm = np.random.default_rng(seed0 + d).permutation(len(pool))
        for k in KS:
            idx = pool[perm[:k]]
            yl, rl = y[idx], y[idx] - p[idx]
            q = tdist.ppf(0.95, k - 1)
            c, hc = yl.mean(), q * yl.std(ddof=1) / np.sqrt(k)
            th, hp = pe.mean() + rl.mean(), q * np.sqrt(pe.var(ddof=1) / N + rl.var(ddof=1) / k)
            ref = stored.loc[(setting, fold, fam, k, d), "abs_bias_L"]
            assert abs(abs(th - mu) - ref) < 1e-3, (setting, fold, fam, k, d, abs(th - mu), ref)  # PPI point estimate = scalar_shift
            rows.append({"setting": setting, "fold": fold, "family": fam, "k": k, "draw": d, "mu": mu, "ppi": th, "half_ppi": hp, "cls": c, "half_cls": hc,
                         "cover_ppi": int(abs(th - mu) <= hp), "cover_cls": int(abs(c - mu) <= hc), "ratio": hp / hc})
    r = y[ev] - pe
    rho = float(np.corrcoef(y[ev], pe)[0, 1])
    u = float(np.sqrt(r.var(ddof=1) / y[ev].var(ddof=1)))
    print(setting, fold, fam, "done", flush=True)
    return {"setting": setting, "fold": fold, "family": fam, "rho": rho, "u": u, "sd_y": float(y[ev].std(ddof=1)), "rmse_r": float(np.sqrt((r ** 2).mean()))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(ROOT / "results" / "reference_correction" / "shift_draws.csv")
    stored = S[S.k.isin(KS)].set_index(["setting", "fold", "family", "k", "draw"]).sort_index()
    rows, info = [], []
    for fold, fam in PAIRS:
        _, z = stats(fold, fam)
        if (fold, fam) == ("E2A", "M2"):
            assert abs(z["dL"].mean(0).mean() - 183.9) <= 0.5  # gate
        info.append(run_cell("orig", fold, fam, z, np.arange(125), np.arange(125, 250), 100, stored, rows))
        info.append(run_cell("swap", fold, fam, z, np.arange(125, 250), np.arange(125), 1100, stored, rows))
    for fam in FAMS:
        n_old, n_new, z = indep_stats(fam)
        info.append(run_cell("indep", "E2A", fam, z, np.arange(n_old), np.arange(n_old, n_old + n_new), 2100, stored, rows))
    D = pd.DataFrame(rows)
    D.round(5).to_csv(OUT / "ppi_draws.csv", index=False)
    I = pd.DataFrame(info)
    g = D.groupby(["setting", "fold", "family", "k"], sort=False)
    C = g.agg(cov_ppi=("cover_ppi", "mean"), cov_cls=("cover_cls", "mean"), R_med=("ratio", "median"), half_ppi=("half_ppi", "median"), half_cls=("half_cls", "median"), n_draws=("draw", "size")).reset_index()
    C = C.merge(I, on=["setting", "fold", "family"])
    C.round(5).to_csv(OUT / "ppi_cells.csv", index=False)
    verdict(C)


def verdict(C):
    c10 = C[C.k == 10]
    n = len(c10)
    valid = int((c10.cov_ppi >= COV_MIN).sum())
    narrow = int((c10.R_med < R_MAX).sum())
    notnarrow = n - narrow
    q1 = "valid" if valid >= NEED else "not valid"
    q2 = "narrower" if narrow >= NEED else "not narrower" if notnarrow >= NEED else "mixed"
    rs = spearmanr(c10.R_med, c10.u)
    overall = "PPI adds value" if (q1 == "valid" and q2 == "narrower") else "PPI does not add value beyond the reference correction"
    f = lambda d, cols: md(d[cols].round(3).to_dict("records"))
    L = ["# Prediction-powered inference study: PPI interval for the mean C_L,p of the new body type (criteria: `PREREG.md`)", "",
         f"Cells at k = 10: {n}; threshold {NEED}. Self-check passed (|PPI point - mu| = stored scalar_shift |bias| C_L,p, all cells, draws, k).", "",
         f"**Q1 validity (coverage >= {COV_MIN}): {valid} of {n} cells valid -> {q1}.**", "",
         f"**Q2 efficiency (median R < {R_MAX}): {narrow} of {n} cells narrower, {notnarrow} not -> {q2}.**", "",
         f"**Overall: {overall}.**", "",
         f"Q3: Spearman between median R (k = 10) and u across the {n} cells: rho = {rs.statistic:.3f} (p = {rs.pvalue:.3g}).", "",
         "## Q4: by setting and k (mean over cells of coverage; median over cells of R)", "",
         md(C.groupby(["setting", "k"]).agg(cells=("family", "size"), cov_ppi=("cov_ppi", "mean"), cov_cls=("cov_cls", "mean"), R_med=("R_med", "median")).round(3).reset_index().to_dict("records")), "",
         "## Cells at k = 10", "", f(c10, ["setting", "fold", "family", "cov_ppi", "cov_cls", "R_med", "half_ppi", "half_cls", "rho", "u", "sd_y", "rmse_r"]), ""]
    for k in (5, 25):
        L += [f"## k = {k}", "", f(C[C.k == k], ["setting", "fold", "family", "cov_ppi", "cov_cls", "R_med", "half_ppi", "half_cls"]), ""]
    (OUT / "verdict.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
