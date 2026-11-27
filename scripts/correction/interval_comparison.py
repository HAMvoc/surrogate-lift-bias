"""Interval-comparison study: coverage and half-width of the jackknife+ interval of the reference correction (stored, the reference-correction study) against an unconditional-spread interval (computed here), original halves, seven E2A models,
k in {5, 10, 25}, 50 draws. Output: results/intervals/{interval_cells.csv, verdict.md}. Usage: python scripts/correction/interval_comparison.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.correction.swapped_halves import PAIRS, stats  # noqa: E402

OUT = ROOT / "results" / "intervals"
KS = (5, 10, 25)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(ROOT / "results" / "reference_correction" / "shift_draws.csv")
    S = S[(S.setting == "orig") & (S.fold == "E2A") & S.k.isin(KS)].set_index(["family", "k", "draw"])
    pool, ev = np.arange(125), np.arange(125, 250)
    rows = []
    for fold, fam in PAIRS:
        if fold != "E2A":
            continue
        _, z = stats(fold, fam)
        e = z["dL"].mean(0)
        if fam == "M2":
            idx = pool[np.random.default_rng(100).permutation(125)[:10]]
            assert abs(abs(e[ev].mean() - e[idx].mean()) - S.loc[(fam, 10, 0), "abs_bias_L"]) < 1e-3  # self-check, PREREG
        for k in KS:
            cj, hj, cu, hu = [], [], [], []
            for d in range(50):
                idx = pool[np.random.default_rng(100 + d).permutation(125)[:k]]
                corr = e[ev] - e[idx].mean()
                h = tdist.ppf(0.95, k - 1) * e[idx].std(ddof=1) * np.sqrt(1 + 1 / k)
                cu.append(float((np.abs(corr) <= h).mean())); hu.append(float(h))
                r = S.loc[(fam, k, d)]
                cj.append(float(r.coverage)); hj.append(float(r.half))
            q = lambda a: (np.mean(a), np.percentile(a, 5), np.percentile(a, 95))
            rows.append({"family": fam, "k": k, "jk_cov": q(cj)[0], "jk_cov_p5": q(cj)[1], "jk_cov_p95": q(cj)[2], "jk_half": q(hj)[0], "jk_half_p5": q(hj)[1], "jk_half_p95": q(hj)[2],
                         "unc_cov": q(cu)[0], "unc_cov_p5": q(cu)[1], "unc_cov_p95": q(cu)[2], "unc_half": q(hu)[0], "unc_half_p5": q(hu)[1], "unc_half_p95": q(hu)[2]})
    C = pd.DataFrame(rows)
    C.round(4).to_csv(OUT / "interval_cells.csv", index=False)
    c10 = C[C.k == 10]
    adeq = int((c10.jk_cov >= 0.85).sum())
    narrow = int((c10.jk_half <= 1.10 * c10.unc_half).sum())
    q1 = "adequate" if adeq >= 5 else "inadequate"
    q2 = "comparable or narrower" if narrow >= 5 else "wider"
    L = ["# Interval-comparison study: interval of the reference correction against an unconditional-spread interval (criteria: `PREREG.md`)", "",
         "Self-check passed (M2, draw 0, k = 10 reproduces the stored |bias|).", "",
         f"**Q1 verdict: {q1}** - {adeq} of 7 models have mean jackknife+ coverage >= 0.85 at k = 10 (threshold 5). Prediction: adequate.", "",
         f"**Q2 verdict: {q2}** - {narrow} of 7 models have a jackknife+ half-width <= 1.10 x the unconditional one at k = 10 (threshold 5). Prediction: comparable or narrower.", "",
         "## Cells (mean over 50 draws; p5 and p95 over draws)", "", md(C.round(3).to_dict("records")), ""]
    (OUT / "verdict.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:7]))


if __name__ == "__main__":
    main()
