"""Classical-baseline and regional-gap study / W2: regional class-mean gap G_r = mean_train(c_r) - mean_test(c_r) of the lift coefficient, from the per-car CFD contributions of results/classical_baselines/cars.csv.
Bootstrap percentile 95% intervals (2000 resamples of the cars within the training group and within the test group, seed 0); Spearman correlation with the regional contribution of the M2 bias on E2A.
Output: results/classical_baselines/{w2_regions.csv, w2_verdict.md}. Usage: python scripts/diagnosis/regional_gap.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.regions import REGIONS2  # noqa: E402

OUT = ROOT / "results" / "classical_baselines"


def main():
    D = pd.read_csv(OUT / "cars.csv")
    tr, te = D[D.role == "train"], D[D.role == "test"]
    cols = [f"cl_{r}" for r in REGIONS2]
    G = tr[cols].mean().to_numpy() - te[cols].mean().to_numpy()
    g_ref = pd.read_csv(ROOT / "results" / "decomposition" / "T4" / "T4_overshoot.csv").query("q == 'C_L,p' and fold == 'E2A' and model == 'M2'").G.iloc[0]
    assert abs(G.sum() - g_ref) < 0.01, (G.sum(), g_ref)  # self-check, PREREG
    rng = np.random.default_rng(0)
    a, b = tr[cols].to_numpy(), te[cols].to_numpy()
    B = np.empty((2000, len(cols)))
    for i in range(2000):
        B[i] = a[rng.integers(0, len(a), len(a))].mean(0) - b[rng.integers(0, len(b), len(b))].mean(0)
    lo, hi = np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0)
    m2 = pd.read_csv(ROOT / "results" / "decomposition" / "T1" / "T1_bias_by_region_v2.csv").query("fold == 'E2A' and model == 'M2'").set_index("region")
    m2c = np.array([m2.loc[r, "contribution C_L,p"] for r in REGIONS2])
    R = pd.DataFrame({"region": REGIONS2, "G_r": G, "lo": lo, "hi": hi, "share": G / G.sum(), "excludes_zero": (lo > 0) | (hi < 0),
                      "mean_train": tr[cols].mean().to_numpy(), "mean_test": te[cols].mean().to_numpy(), "m2_bias_contribution": m2c, "m2_overshoot_r": m2c - G})
    R.round(4).to_csv(OUT / "w2_regions.csv", index=False)
    sig = R[R.excludes_zero]
    share = sig.G_r.sum() / G.sum()
    verdict = "concentrated" if len(sig) <= 3 and share >= 2 / 3 else "spread" if len(sig) > 3 else "mixed"
    rho_g, p_g = spearmanr(G, m2c)
    rho_o, p_o = spearmanr(G, m2c - G)
    top = R.reindex(R.G_r.abs().sort_values(ascending=False).index)
    L = ["# Classical-baseline and regional-gap study / W2: regional class-mean gap of the lift coefficient (criteria: `PREREG.md`)", "",
         f"Self-check passed: sum of G_r = {G.sum():.3f} counts = G of `T4_overshoot.csv` ({g_ref:.3f}); training cars {len(tr)}, test cars {len(te)}.", "",
         f"**Q2 verdict: {verdict}** - {len(sig)} region(s) with an interval excluding zero ({', '.join(sig.region)}), carrying {100 * share:.1f}% of G. Prediction: concentrated, roof_rear largest, underbody second.", "",
         f"Largest |G_r|: {top.region.iloc[0]} ({top.G_r.iloc[0]:.1f}), {top.region.iloc[1]} ({top.G_r.iloc[1]:.1f}), {top.region.iloc[2]} ({top.G_r.iloc[2]:.1f}).", "",
         f"Q3: Spearman across the {len(REGIONS2)} regions between G_r and the regional contribution of the M2 bias: rho = {rho_g:.3f} (p = {p_g:.3g}); with the regional overshoot: rho = {rho_o:.3f} (p = {p_o:.3g}).", "",
         "## Q1: G_r by region (counts; interval = bootstrap 95%)", "", md(R.round(2).to_dict("records")), ""]
    (OUT / "w2_verdict.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:10]))


if __name__ == "__main__":
    main()
