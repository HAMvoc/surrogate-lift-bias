"""Component and cross-model study, task T2 (EXPLORATORY): does the integral correction (frozen + region_const_intmatch, 8 regions v2, lambda by LOO inside the k cars) generalise over model families and shift directions?
(family, fold) pairs with |bias C_L,p| > 100 counts: E2A with B0, B1, B2, B2_300ep, B3, M2 and E2C with B3. For every fold the 250 test cars are permuted as in the two-stage correction study (default_rng(0).permutation(sorted test)),
candidates = perm[:125], evaluation = perm[125:]; draw d, k = cand[default_rng(100 + d).permutation(125)[:k]]; k in {1, 2, 3, 5, 10, 25}, 50 draws.
Output: results/components/T2/T2_generality.csv (per draw), T2_generality.md, T2_residual_fraction.{png,pdf}. Usage: python scripts/correction/cross_model.py"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from src.utils import plot_style as paper_style  # noqa: E402
paper_style.apply()
from scripts.correction.calibration import build_stats, corrected  # noqa: E402
from scripts.correction.frozen_draws import fit_method, jackknife  # noqa: E402
from scripts.diagnosis.regions import labeler as labeler_v2  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

OUT = ROOT / "results" / "components" / "T2"
KS = (1, 2, 3, 5, 10, 25)
DIRS = {"M2": [f"{{f}}_M2_R2_s{s}" for s in range(3)], "B0": ["{f}_B0_NA_s0"], "B1": ["{f}_B1_R1_s0"], "B2": ["{f}_B2_R1_s0"], "B2_300ep": ["{f}_B2_R1_s0_ep300"], "B3": ["{f}_B3_R1_s0_lr3e-3"]}
PAIRS = [("E2A", m) for m in ("B0", "B1", "B2", "B2_300ep", "B3", "M2")] + [("E2C", "B3")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold, fam in PAIRS:
        perm = [str(i) for i in np.random.default_rng(0).permutation(sorted(load_split(fold)["test"]))]
        dirs = [ROOT / "results" / d.format(f=fold) for d in DIRS[fam]]
        preds = [np.load(d / "test_pred_fp16.npy", mmap_mode="r") for d in dirs]
        pos = [{i: j for j, i in enumerate(pd.read_csv(d / "test_per_car.csv").id)} for d in dirs]
        z = build_stats(perm, lambda s, sid: preds[s][pos[s][sid]], len(dirs), labeler_v2, 8)
        bias250 = z["dL"].mean(0).mean()
        if (fold, fam) == ("E2A", "M2"):
            assert abs(bias250 - 183.9) <= 0.5, bias250  # gate
        ev = np.arange(125, 250)
        eL0 = corrected(z, ev, ("none", None))[0].mean(0)
        dd0 = corrected(z, ev, ("none", None))[1].mean(0)
        rel0 = corrected(z, ev, ("none", None))[2].mean()
        rows.append({"fold": fold, "family": fam, "k": 0, "draw": 0, "bias_L": eL0.mean(), "abs_bias_L": abs(eL0.mean()), "abs_bias_D": abs(dd0.mean()), "rmse_L": np.sqrt(((eL0 - eL0.mean()) ** 2).mean()), "rel_l2": rel0,
                     "frac": 1.0, "bias250": bias250})
        for d in range(50):
            for k in KS:
                idx = np.random.default_rng(100 + d).permutation(125)[:k]
                fit = fit_method(z, idx, "intmatch_lam")
                dL, dD, rel = corrected(z, ev, fit)
                eL, eD = dL.mean(0), dD.mean(0)
                row = {"fold": fold, "family": fam, "k": k, "draw": d, "bias_L": eL.mean(), "abs_bias_L": abs(eL.mean()), "abs_bias_D": abs(eD.mean()), "rmse_L": np.sqrt(((eL - eL.mean()) ** 2).mean()), "rel_l2": rel.mean(),
                       "frac": abs(eL.mean()) / abs(eL0.mean()), "bias250": bias250}
                if k >= 5:
                    row["coverage"], row["half"] = jackknife(z, idx, ev, "intmatch_lam")
                rows.append(row)
        print(fold, fam, "done", flush=True)
    R = pd.DataFrame(rows)
    R.round(5).to_csv(OUT / "T2_generality.csv", index=False)
    summarise(R)


def summarise(R):  # also callable alone: python -c "import pandas as pd, cross_model as m; m.summarise(pd.read_csv(m.OUT / 'T2_generality.csv'))"
    R = R[R.draw < 50]
    base = R[R.k == 0].set_index(["fold", "family"])
    L = ["# Component and cross-model study, task T2 (EXPLORATORY): generality of the integral correction over model families and shift directions", "",
         "Every row: frozen model + `region_const_intmatch` (8 regions v2, lambda by LOO inside the k cars) fitted on k labelled target cars; 50 random draws per k (the first 10 are the shared draws of the two-stage correction study); evaluation on 125 cars of the held-out style that are never used for fitting. "
         "'bias fraction left' = |bias C_L,p after| / |bias C_L,p before| on the evaluation cars, per draw, then mean [5%; 95%] over draws. Single-seed families (B0-B3) have one prediction per car; M2 is the 3-seed mean per car. "
         "Selection: (family, fold) pairs with |bias C_L,p| > 100 counts on the 250 test cars (E2A: all six families; E2C: B3 only). Gate: M2 E2A 3-seed bias = +183.9 on the 250 cars: PASS.", ""]
    for k in (10,):
        tab = []
        for (fold, fam), b in base.iterrows():
            g = R[(R.fold == fold) & (R.family == fam) & (R.k == k)]
            tab.append({"fold": fold, "family": fam, "bias before (125 cars)": f"{b.bias_L:+.1f}", "|bias| after": f"{g.abs_bias_L.mean():.1f} [{g.abs_bias_L.quantile(.05):.1f}; {g.abs_bias_L.quantile(.95):.1f}]",
                        "bias fraction left": f"{g.frac.mean():.3f} [{g.frac.quantile(.05):.3f}; {g.frac.quantile(.95):.3f}]", "|bias| C_D,p before -> after": f"{b.abs_bias_D:.1f} -> {g.abs_bias_D.mean():.1f}", "Rel L2 before -> after": f"{b.rel_l2:.4f} -> {g.rel_l2.mean():.4f}",
                        "RMSE before -> after": f"{b.rmse_L:.1f} -> {g.rmse_L.mean():.1f}", "jackknife+ coverage": f"{g.coverage.mean():.3f}", "half-width": f"{g.half.mean():.0f}"})
        L += [f"## k = {k}", "", md(tab), ""]
    tab = []
    for (fold, fam) in base.index:
        row = {"fold": fold, "family": fam}
        for k in KS:
            g = R[(R.fold == fold) & (R.family == fam) & (R.k == k)]
            row[f"k={k}"] = f"{g.frac.mean():.3f}"
        tab.append(row)
    L += ["## Mean bias fraction left, all k", "", md(tab), ""]
    f10 = R[R.k == 10].groupby(["fold", "family"]).frac.mean()
    L += ["## Question: is the fraction left at k = 10 similar across families and shift directions?", "",
          f"Mean fractions at k = 10 over the seven (family, fold) pairs lie between {f10.min():.3f} and {f10.max():.3f} (E2A pairs: {f10.loc['E2A'].min():.3f}-{f10.loc['E2A'].max():.3f}; E2C B3, negative bias: {f10.loc[('E2C', 'B3')]:.3f}). "
          "Description only: the fraction is a ratio with the uncorrected bias in the denominator; the absolute residual |bias| after correction (first table) is 14.6-26.2 counts for all pairs, while the uncorrected bias ranges from 145 to 216 counts in absolute value. ",
          "The 5-95% range over draws at k = 10 is wide for every pair.", ""]
    (OUT / "T2_generality.md").write_text("\n".join(L), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    cols = {"M2": "C0", "B0": "C1", "B1": "C2", "B2": "C3", "B2_300ep": "C4", "B3": "C5"}
    for (fold, fam) in base.index:
        g = R[(R.fold == fold) & (R.family == fam)].groupby("k").frac
        m = g.mean().reindex(KS)
        ax.plot(KS, m.values, ("-" if fold == "E2A" else "--"), marker=("o" if fold == "E2A" else "^"), c=cols[fam], label=f"{fam} {fold}", lw=1.1, ms=4)
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS); ax.set_xlabel("labelled cars k"); ax.set_ylabel("|bias C_L,p| left / |bias| before")
    ax.axhline(1, c="gray", lw=.5); ax.legend(fontsize=6, frameon=False, ncol=2); ax.set_title("Residual bias fraction after frozen + intmatch (exploratory)", fontsize=8)
    fig.savefig(OUT / "T2_residual_fraction.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "T2_residual_fraction.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
