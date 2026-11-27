"""Transolver on all held-out types study, task T4 (EXPLORATORY, description only): observed over predicted |bias| (statistical-floor formula, sigma_pop) against k for every cell of the floor analysis:
the six original families and methods (results/label_budget/T1/T1_cells.csv, 'old') and the Transolver cells of the Transolver study (T1_cells_p7.csv, 'tsv'). No model is fitted, no threshold.
Output: results/transolver_types/T4/{T4_by_k.csv, T4_summary.csv, T4_lines.csv, T4_report.md} and the figure T4_ratio_vs_k.{png,pdf}. Usage: python scripts/correction/label_budget_cells.py"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import plot_style as paper_style  # noqa: E402

paper_style.apply()
OUT = ROOT / "results" / "transolver_types" / "T4"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.read_csv(ROOT / "results" / "transolver" / "T5" / "T1_cells_p7.csv")
    R["group"] = np.where(R.family == "TSV", "tsv", "old")
    R["line"] = R.family + " " + R.method + " " + R.fold
    R[["group", "line", "source", "method", "family", "fold", "k", "obs", "sigma_pop", "pred_pop", "ratio_pop"]].round(4).to_csv(OUT / "T4_by_k.csv", index=False)
    rows, lines = [], []
    for g, d in R.groupby("group"):
        rho, p = spearmanr(d.k, d.ratio_pop)
        kmin, kmax = d.k.min(), d.k.max()
        rows.append({"group": g, "ratio_first_k": d[d.k == kmin].ratio_pop.mean(), "ratio_last_k": d[d.k == kmax].ratio_pop.mean(), "spearman_k_ratio": rho, "spearman_p": p, "n_cells": len(d), "k_first": kmin, "k_last": kmax})
    S = pd.DataFrame(rows)
    S.round(4).to_csv(OUT / "T4_summary.csv", index=False)
    per_k = R.groupby(["group", "k"]).ratio_pop.agg(["mean", "min", "max", "count"]).round(3).reset_index()
    for ln, d in R.groupby("line"):
        d = d.sort_values("k")
        if len(d) >= 3:
            rho, _ = spearmanr(d.k, d.ratio_pop)
        else:
            rho = np.nan
        lines.append({"line": ln, "group": d.group.iloc[0], "n_k": len(d), "k_min": d.k.min(), "k_max": d.k.max(), "ratio_at_k_min": d.ratio_pop.iloc[0], "ratio_at_k_max": d.ratio_pop.iloc[-1], "spearman_k_ratio": rho})
    L = pd.DataFrame(lines)
    L.round(3).to_csv(OUT / "T4_lines.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for ln, d in R.groupby("line"):
        d = d.sort_values("k")
        tsv = d.group.iloc[0] == "tsv"
        ax.plot(d.k, d.ratio_pop, "-o" if len(d) > 2 else "o", c="C6" if tsv else "0.55", lw=1.8 if tsv else 0.8, ms=4 if tsv else 2.5, zorder=3 if tsv else 2, label="Transolver frozen + B (E2A)" if tsv else None)
    ax.plot([], [], "-o", c="0.55", lw=0.8, ms=2.5, label="six original families and methods")
    ax.axhline(1, c="k", ls="--", lw=0.6); ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10, 25]); ax.set_xticklabels([1, 2, 3, 5, 10, 25])
    ax.set_xlabel("labelled cars k"); ax.set_ylabel("observed / predicted |bias| C_L,p"); ax.legend(fontsize=6, frameon=False)
    fig.savefig(OUT / "T4_ratio_vs_k.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "T4_ratio_vs_k.pdf", bbox_inches="tight"); plt.close(fig)
    md = lambda d: "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.values)
    neg = L[(L.n_k >= 3)]
    txt = ["# Transolver on all held-out types study, task T4 (EXPLORATORY): observed over predicted |bias| against k", "",
           "Description only; no model fitted, no threshold. Cells: the six original families and methods of the floor analysis (group `old`: Two-stage correction study T1 ft_B_OOF, the component and cross-model study T2 frozen + B over seven (family, fold) pairs, the component and cross-model study T3 retrain + B OOF) and the six Transolver cells of the Transolver study (group `tsv`, E2A, frozen + B, 50 draws). "
           "ratio = obs / (sqrt(2/pi) sigma_pop / sqrt(k)) per cell (slope 1 formula; the 1.302 of the label-budget study is not used here).", "",
           "## Mean ratio at each k", "", md(per_k), "",
           "## Summary (pooled cells of the group)", "", md(S.round(3)), "",
           "`ratio_first_k` and `ratio_last_k` are the means over the cells at the smallest and the largest k of the group; `spearman_k_ratio` is the Spearman correlation between k and the ratio over all cells of the group (p-value in `spearman_p`; the cells share draws, so the p-value is indicative only).", "",
           f"## Lines with three or more values of k (n = {len(neg)})", "", md(neg.round(3)), "",
           f"Lines with a negative Spearman correlation between k and the ratio: {int((neg.spearman_k_ratio < 0).sum())} of {len(neg)}.", ""]
    (OUT / "T4_report.md").write_text("\n".join(txt), encoding="utf-8")
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
