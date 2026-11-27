"""Component and cross-model study, task T1 (EXPLORATORY): component effects of the two-stage C2 from the two-stage correction study per-draw results.
Comparisons over the 10 shared draws (seed-mean per draw): effect of A (ft_noB vs frozen), effect of B (ft_B_OOF vs ft_noB), dec vs full (per draw and seed, same draw and seed). Paired two-sided Wilcoxon, Holm within each family.
Inputs: results/two_stage/T1/T1_per_draw.csv, results/two_stage/T1/raw/*.npz, results/two_stage/T2/frozen_draws.csv. Outputs: results/components/T1/T1_components.{csv,md,png,pdf}.
Usage: python scripts/correction/components.py"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from src.utils import plot_style as paper_style  # noqa: E402
paper_style.apply()

P3, OUT = ROOT / "results" / "two_stage", ROOT / "results" / "components" / "T1"
FIN, NOB = "ft_B_OOF (final A)", "ft noB (final A)"
MET = {"abs_bias_L": "|bias| C_L,p", "abs_bias_D": "|bias| C_D,p", "rel_l2": "Rel L2", "rmse_L": "RMSE after bias removal"}


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def wil(d):
    d = np.asarray(d, float)
    return 1.0 if np.all(d == 0) else float(wilcoxon(d, alternative="two-sided").pvalue)


def comp(family, rows):
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p)
    df.insert(0, "family", family)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(P3 / "T1" / "T1_per_draw.csv")
    Fz = pd.read_csv(P3 / "T2" / "frozen_draws.csv")
    fz = Fz[Fz.method == "none"].iloc[0]
    assert abs(fz.abs_bias_L - 182.03) < 0.1  # frozen M2 3-seed bias (gate +183.9 is for the 250-car set; here the 125 evaluation cars)
    g = lambda v, k, m: D[(D.variant == v) & (D.k == k) & (D.method == m)].sort_values("draw")
    fam = []
    # A: ft_noB vs frozen
    rows = []
    for v in ("dec", "full"):
        for k in (5, 10, 25):
            a = g(v, k, NOB)
            for m in ("rel_l2", "rmse_L", "abs_bias_L"):
                d = a[m].values - fz[m]
                rows.append({"variant": v, "k": k, "metric": MET[m], "mean diff": d.mean(), "p5": np.percentile(d, 5), "p95": np.percentile(d, 95), "n": len(d), "p": wil(d), "reference": f"frozen {fz[m]:.4g}"})
    fam.append(comp("A: ft_noB minus frozen", rows))
    # B: ft_B_OOF vs ft_noB
    rows = []
    for v in ("dec", "full"):
        for k in (5, 10, 25):
            a, b = g(v, k, FIN), g(v, k, NOB)
            for m in ("abs_bias_L", "abs_bias_D", "rel_l2", "rmse_L"):
                d = a[m].values - b[m].values
                rows.append({"variant": v, "k": k, "metric": MET[m], "mean diff": d.mean(), "p5": np.percentile(d, 5), "p95": np.percentile(d, 95), "n": len(d), "p": wil(d), "reference": f"ft_noB {b[m].mean():.4g}"})
    fam.append(comp("B: ft_B_OOF minus ft_noB", rows))
    # dec vs full, same draw and seed (from the raw files)
    per = []
    for f in sorted((P3 / "T1" / "raw").glob("*.npz")):
        v, s, k, d = f.stem.split("_")
        z = np.load(f)
        for meth, key in ((FIN, "B_full"), (NOB, "noB_full")):
            a = z[key]
            per.append({"variant": v, "seed": int(s[1:]), "k": int(k[1:]), "draw": int(d[1:]), "method": meth, "abs_bias_L": abs(a[:, 0].mean()), "abs_bias_D": abs(a[:, 1].mean()),
                        "rmse_L": float(np.sqrt(((a[:, 0] - a[:, 0].mean()) ** 2).mean())), "rel_l2": a[:, 2].mean()})
    S = pd.DataFrame(per)
    S.round(5).to_csv(OUT / "T1_per_seed.csv", index=False)
    rows = []
    for meth in (FIN, NOB):
        for k in (5, 10, 25):
            a = S[(S.variant == "dec") & (S.k == k) & (S.method == meth)].set_index(["draw", "seed"]).sort_index()
            b = S[(S.variant == "full") & (S.k == k) & (S.method == meth)].set_index(["draw", "seed"]).sort_index()
            for m in ("abs_bias_L", "abs_bias_D", "rel_l2", "rmse_L"):
                d = a[m].values - b[m].values
                rows.append({"variant": "dec minus full", "k": k, "metric": f"{MET[m]} ({meth})", "mean diff": d.mean(), "p5": np.percentile(d, 5), "p95": np.percentile(d, 95), "n": len(d), "p": wil(d), "reference": f"full {b[m].mean():.4g}"})
    fam.append(comp("dec vs full (same draw and seed; pairs share the draw, so the pairs are not independent)", rows))
    C = pd.concat(fam, ignore_index=True)
    C.round(5).to_csv(OUT / "T1_components.csv", index=False)
    # OOF vs in-sample: not recomputable
    note = ("**OOF versus in-sample stage B (B-same): not possible on the two-stage correction study draws.** B-same needs the predictions of the final stage-A model on the k cars it was fine-tuned on; the two-stage correction study files keep only the "
            "out-of-fold residuals (`oof`), the evaluation-car errors and the stage-B offsets, and the fine-tuned weights were not saved. Re-running the fine-tune would need the GPU, which the component and cross-model study reserves for T3. "
            "Reference with a different design (Decomposition and first corrections study T5, `results/decomposition/T5/T5_report.md`; retrain run of the first k cars of the pool, `region_const`, 215 test cars, one draw): B-same |bias| C_L,p 88.3 / 77.7 / 20.9 counts and jackknife+ coverage 0.047 / 0.042 / 0.153 at k = 5 / 10 / 25, "
            "against |bias| 22.1-22.6 and CV+ median coverage 0.93 for the OOF pipeline at k = 10 in the two-stage correction study (`dec`, `full`). The two sets of numbers differ in cars, retrain draw, test set and interval method, so they are not a paired comparison. "
            "Within the retrain + OOF experiment of the component and cross-model study T3 the in-sample variant is not computed either.")
    md_rows = lambda df: md([{c: (f"{r[c]:.4g}" if isinstance(r[c], float) else r[c]) for c in ("variant", "k", "metric", "mean diff", "p5", "p95", "n", "p", "p_holm", "reference")} for _, r in df.iterrows()])
    L = ["# Component and cross-model study, task T1 (EXPLORATORY): component effects", "",
         "All analyses of this file are exploratory. Source: the 10 shared draws of the two-stage correction study (k = 5, 10, 25; `dec` and `full`); per-draw metrics are seed means over 3 seeds; differences are over draws (mean, 5% and 95% percentiles, paired two-sided Wilcoxon, p; Holm adjustment inside each family). "
         "With n = 10 pairs the smallest two-sided Wilcoxon p is 0.002; with n = 30 pairs it is far smaller; Holm multiplies by the number of tests of the family. Frozen M2 has one value (3-seed mean on the 125 evaluation cars): the A comparison is a one-sample Wilcoxon of the differences to that constant.", ""]
    for name, df in C.groupby("family", sort=False):
        L += [f"## {name} ({len(df)} tests)", "", md_rows(df), ""]
    L += ["## OOF versus in-sample stage B", "", note, ""]
    (OUT / "T1_components.md").write_text("\n".join(L), encoding="utf-8")
    # figure
    fig, axs = plt.subplots(2, 3, figsize=(8.2, 4.6))
    for r_, (m, lab) in enumerate((("abs_bias_L", "|bias| C_L,p (counts)"), ("rel_l2", "Rel L2"))):
        for c_, k in enumerate((5, 10, 25)):
            ax = axs[r_, c_]
            ax.bar(0, fz[m], color="0.6", width=0.6)
            for j, (meth, nm) in enumerate(((NOB, "ft_noB"), (FIN, "ft_B_OOF"))):
                for i, (v, col) in enumerate((("dec", "C3"), ("full", "C2"))):
                    a = g(v, k, meth)[m].values
                    x = 1 + j + (i - 0.5) * 0.36
                    ax.bar(x, a.mean(), width=0.34, color=col, label=f"FT-{v}" if (r_ == 0 and c_ == 0 and j == 0) else None)
                    ax.errorbar(x, a.mean(), yerr=[[a.mean() - np.percentile(a, 5)], [np.percentile(a, 95) - a.mean()]], c="k", capsize=2, lw=0.8)
            ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["frozen", "FT only", "two-stage"], fontsize=7); ax.set_title(f"k = {k}", fontsize=8)
            if c_ == 0:
                ax.set_ylabel(lab)
            if m == "rel_l2":
                ax.set_ylim(0.15, 0.21)
    axs[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Component effects (exploratory): mean over 10 draws, bars 5-95% over draws", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "T1_components.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "T1_components.pdf", bbox_inches="tight")
    print(C.groupby("family").size())
    for fam_, df in C.groupby("family", sort=False):
        print(fam_[:30], "tests", len(df), "Holm p<0.05:", int((df.p_holm < 0.05).sum()))


if __name__ == "__main__":
    main()
