"""Number-source study, task T3 (EXPLORATORY; does not change the registered 'partial' verdict of the label-budget study T1): cluster bootstrap of the slope and R^2 of the statistical-floor regression
(obs = slope * pred, no intercept; cells of results/label_budget/T1/T1_cells.csv). Clusters = (source, method, family, fold): 7 frozen (family, fold) pairs, 2 fine-tune variants, 1 retrain = 10 clusters; B = 2000.
Equivalent efficiency 1/slope^2: stage B behaves like a mean estimate from k/slope^2 cars. k needed with the slope: k = (slope * sqrt(2/pi) * sigma / target)^2.
Output: results/number_sources/T3/T3_boot.csv, T3_kneeded_slope.csv, T3_report.md. Usage: python scripts/correction/label_budget_bootstrap.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402

OUT = ROOT / "results" / "number_sources" / "T3"
B = 2000
C = float(np.sqrt(2 / np.pi))


def fit(x, y):
    s = (x * y).sum() / (x * x).sum()
    return s, 1 - ((y - s * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.read_csv(ROOT / "results" / "label_budget" / "T1" / "T1_cells.csv")
    R["cluster"] = R.source + "|" + R.method + "|" + R.family + "|" + R.fold
    cl = {c: g[["pred_pop", "obs"]].values for c, g in R.groupby("cluster")}
    names = list(cl)
    x0, y0 = R.pred_pop.values, R.obs.values
    s0, r0 = fit(x0, y0)
    rng = np.random.default_rng(0)
    sl, r2 = [], []
    for _ in range(B):
        pick = rng.integers(0, len(names), len(names))
        a = np.vstack([cl[names[i]] for i in pick])
        s, r = fit(a[:, 0], a[:, 1])
        sl.append(s); r2.append(r)
    sl, r2 = np.array(sl), np.array(r2)
    q = lambda v: np.percentile(v, [2.5, 97.5])
    ci_s, ci_r, eff = q(sl), q(r2), q(1 / sl ** 2)
    # leave-one-cluster-out as a robustness check
    loo = [fit(*np.vstack([cl[n] for n in names if n != m]).T) for m in names]
    stats = [("slope", s0, ci_s[0], ci_s[1]), ("r2", r0, ci_r[0], ci_r[1]), ("efficiency", 1 / s0 ** 2, eff[0], eff[1]), ("n_clusters", len(names), np.nan, np.nan), ("n_cells", len(R), np.nan, np.nan),
             ("slope_loo_min", min(s for s, _ in loo), np.nan, np.nan), ("slope_loo_max", max(s for s, _ in loo), np.nan, np.nan), ("slope_boot_mean", sl.mean(), np.nan, np.nan), ("slope_one_inside", float(ci_s[0] <= 1 <= ci_s[1]), np.nan, np.nan)]
    S = pd.DataFrame(stats, columns=["statistic", "value", "lo", "hi"])
    S.round(6).to_csv(OUT / "T3_boot.csv", index=False)
    F = pd.read_csv(ROOT / "results" / "label_budget" / "T1" / "T1_kneeded.csv")
    rows = []
    for _, r in F.iterrows():
        tag = "frozen" if r.source.startswith("M2") else "ftnob"
        for t in (10, 20, 28):
            k = lambda s: (s * C * r.sigma / t) ** 2
            rows.append({"source": tag, "sigma": r.sigma, "target": t, "k_formula": k(1.0), "k_slope": k(s0), "k_slope_lo": k(ci_s[0]), "k_slope_hi": k(ci_s[1])})
    K = pd.DataFrame(rows)
    K.round(4).to_csv(OUT / "T3_kneeded_slope.csv", index=False)
    L = ["# Number-source study, task T3 (EXPLORATORY): cluster bootstrap of the statistical-floor slope", "",
         "Exploratory; the registered verdict of the label-budget study T1 ('partial', slope 1.302 against the range [0.8; 1.25]) is not changed. Regression through the origin of observed on predicted |bias| over the 50 cells of the label-budget study T1; clusters (source, method, family, fold): "
         f"{len(names)} clusters of 2-6 cells (7 frozen pairs, 2 fine-tune variants, 1 retrain); B = {B} resamples of the clusters. The cells inside a cluster share the model and the evaluation cars, so the cell-level R^2 is optimistic and the cluster bootstrap is the appropriate interval.", "",
         md([{"statistic": "slope", "value": f"{s0:.3f}", "95% CI": f"[{ci_s[0]:.3f}; {ci_s[1]:.3f}]"}, {"statistic": "R^2 (centred)", "value": f"{r0:.3f}", "95% CI": f"[{ci_r[0]:.3f}; {ci_r[1]:.3f}]"},
             {"statistic": "equivalent efficiency 1/slope^2", "value": f"{1 / s0 ** 2:.3f}", "95% CI": f"[{eff[0]:.3f}; {eff[1]:.3f}]"}]), "",
         f"Interpretation of the efficiency: stage B behaves like a plain mean estimate from about k/slope^2 = {1 / s0 ** 2:.2f} k cars (CI {eff[0]:.2f} to {eff[1]:.2f} k). The interval for the slope {'contains' if ci_s[0] <= 1 <= ci_s[1] else 'does not contain'} 1. "
         f"Leave-one-cluster-out slopes: {min(s for s, _ in loo):.3f} to {max(s for s, _ in loo):.3f}.", "",
         "## k needed with the slope, k = (slope * sqrt(2/pi) * sigma / target)^2", "",
         md([{"sigma source": r.source, "sigma": f"{r.sigma:.1f}", "target |bias|": int(r.target), "k (formula, slope 1)": f"{r.k_formula:.1f}", "k (with slope)": f"{r.k_slope:.1f}", "k, CI of the slope": f"[{r.k_slope_lo:.1f}; {r.k_slope_hi:.1f}]"} for _, r in K.iterrows()]), ""]
    (OUT / "T3_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[4:14]))


if __name__ == "__main__":
    main()
