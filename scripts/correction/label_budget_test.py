"""Label-budget study, task T1: test of the statistical-floor hypothesis E|bias| ~ sqrt(2/pi) sigma / sqrt(k).
Cells: Two-stage correction study T1 (dec, full ft_B_OOF; k = 5, 10, 25), the component and cross-model study T2 (frozen + region_const_intmatch, 7 (family, fold) pairs, k = 1..25), the component and cross-model study T3 (retrain + B OOF, k = 10 and 25).
Output: results/label_budget/T1/T1_cells.csv, T1_floor.md, F10_floor.{png,pdf}. Usage: python scripts/correction/label_budget_test.py"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from src.utils import plot_style as paper_style  # noqa: E402
from scripts.correction import statistics_cache as pc  # noqa: E402
from scripts.diagnosis.region_errors import md  # noqa: E402

paper_style.apply()
OUT = ROOT / "results" / "label_budget" / "T1"
C = float(np.sqrt(2 / np.pi))
KS = (1, 2, 3, 5, 10, 25)


def sd(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1)) if len(x) > 1 else np.nan


def cells():
    rows = []
    # Two-stage correction study T1: ft_B_OOF final A (obs) against ft noB final A (sigma_pop) per draw; sigma_k from the stored OOF residuals
    D = pd.read_csv(ROOT / "results" / "two_stage" / "T1" / "T1_per_draw.csv")
    for v in ("dec", "full"):
        for k in (5, 10, 25):
            obs, sp, sk = [], [], []
            for d in range(10):
                a = D[(D.variant == v) & (D.k == k) & (D.draw == d)]
                obs.append(a[a.method == "ft_B_OOF (final A)"].abs_bias_L.iloc[0])
                sp.append(a[a.method == "ft noB (final A)"].rmse_L.iloc[0])
                sk.append(np.mean([sd(np.load(ROOT / "results" / "two_stage" / "T1" / "raw" / f"{v}_s{s}_k{k}_d{d}.npz")["oof"][:, 0]) for s in range(3)]))
            rows.append({"source": "Two-stage correction study T1", "method": f"ft_B_OOF {v}", "family": "M2", "fold": "E2A", "k": k, "n_draws": 10, "obs": np.mean(obs), "sigma_pop": np.mean(sp), "sigma_k": np.mean(sk),
                         "pred_pop": np.mean([C * s / np.sqrt(k) for s in sp]), "pred_k": np.mean([C * s / np.sqrt(k) for s in sk])})
    # Component and cross-model study T2: frozen + region_const_intmatch
    G = pd.read_csv(ROOT / "results" / "components" / "T2" / "T2_generality.csv")
    for fold, fam in pc.PAIRS:
        z = pc.pair_stats(fold, fam)
        eL = z["dL"].mean(0)
        sp = G[(G.fold == fold) & (G.family == fam) & (G.k == 0)].rmse_L.iloc[0]
        for k in KS:
            obs, sk = [], []
            for d in range(50):
                idx = np.random.default_rng(100 + d).permutation(125)[:k]
                sk.append(sd(eL[idx]))
                obs.append(G[(G.fold == fold) & (G.family == fam) & (G.k == k) & (G.draw == d)].abs_bias_L.iloc[0])
            rows.append({"source": "Component and cross-model study T2", "method": "frozen + B", "family": fam, "fold": fold, "k": k, "n_draws": 50, "obs": np.mean(obs), "sigma_pop": sp, "sigma_k": np.nanmean(sk) if k > 1 else np.nan,
                         "pred_pop": C * sp / np.sqrt(k), "pred_k": np.nanmean([C * s / np.sqrt(k) for s in sk]) if k > 1 else np.nan})
    # Component and cross-model study T3: retrain + B OOF (obs), retrain without B (sigma_pop), OOF residuals (sigma_k)
    T3 = pd.read_csv(ROOT / "results" / "components" / "T3" / "T3_draws.csv")
    for k, ds in ((10, range(3)), (25, range(4))):
        obs, sp, sk = [], [], []
        for d in ds:
            obs.append(T3[(T3.k == k) & (T3.draw == d) & (T3.method == "retrain + B OOF")].abs_bias_L.iloc[0])
            sp.append(T3[(T3.k == k) & (T3.draw == d) & (T3.method == "retrain")].rmse_L.iloc[0])
            sk.append(sd(pc.t3_stats(k, d)[1]["dL"][0]))
        rows.append({"source": "Component and cross-model study T3", "method": "retrain + B OOF", "family": "M2", "fold": "E2A", "k": k, "n_draws": len(sp), "obs": np.mean(obs), "sigma_pop": np.mean(sp), "sigma_k": np.mean(sk),
                     "pred_pop": np.mean([C * s / np.sqrt(k) for s in sp]), "pred_k": np.mean([C * s / np.sqrt(k) for s in sk])})
    R = pd.DataFrame(rows)
    R["ratio_pop"] = R.obs / R.pred_pop
    R["ratio_k"] = R.obs / R.pred_k
    return R


def regress(x, y):
    x, y = np.asarray(x), np.asarray(y)
    slope = (x * y).sum() / (x * x).sum()
    r2 = 1 - ((y - slope * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r2_unc = 1 - ((y - slope * x) ** 2).sum() / (y ** 2).sum()
    return slope, r2, r2_unc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = cells()
    R.round(5).to_csv(OUT / "T1_cells.csv", index=False)
    s_pop, r2_pop, r2u_pop = regress(R.pred_pop, R.obs)
    ok_k = R.pred_k.notna()
    s_k, r2_k, r2u_k = regress(R.pred_k[ok_k], R.obs[ok_k])
    frac = float(((R.ratio_pop >= 0.6) & (R.ratio_pop <= 1.6)).mean())
    frac_k = float(((R.ratio_k[ok_k] >= 0.6) & (R.ratio_k[ok_k] <= 1.6)).mean())
    keep = s_pop >= 0.8 and s_pop <= 1.25 and r2_pop >= 0.7 and frac >= 0.8
    rej = s_pop < 0.6 or s_pop > 1.6 or r2_pop < 0.4
    verdict = "GIỮ" if keep else ("BÁC BỎ" if rej else "MỘT PHẦN")
    low = R[R.k <= 2]
    pred_low = bool(low.ratio_pop.mean() > 1)
    # k needed
    D = pd.read_csv(ROOT / "results" / "two_stage" / "T1" / "T1_per_draw.csv")
    F = pd.read_csv(ROOT / "results" / "two_stage" / "T2" / "frozen_draws.csv")
    sig = {"M2 frozen (125 evaluation cars)": float(F[F.method == "none"].rmse_L.iloc[0]),
           "`full` ft_noB (k = 10, mean over draws)": float(D[(D.variant == "full") & (D.k == 10) & (D.method == "ft noB (final A)")].rmse_L.mean())}
    need = [{"sigma_pop source": n, "sigma (counts)": f"{s:.1f}", **{f"k for E|bias| <= {t}": f"{(C * s / t) ** 2:.1f}" for t in (10, 20, 28)}} for n, s in sig.items()]
    # post-hoc diagnostic (NOT registered): add the sampling error of the 125-car evaluation mean in quadrature, sigma * sqrt(1/k + 1/125)
    R["pred_posthoc"] = [C * r.sigma_pop * np.sqrt(1 / r.k + 1 / 125) if r.source != "Component and cross-model study T3" or True else np.nan for _, r in R.iterrows()]
    s_ph, r2_ph, _ = regress(R.pred_posthoc, R.obs)
    frac_ph = float(((R.obs / R.pred_posthoc >= 0.6) & (R.obs / R.pred_posthoc <= 1.6)).mean())
    # figure F10
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    col = {"ft_B_OOF dec": "C3", "ft_B_OOF full": "C2", "frozen + B": "0.45", "retrain + B OOF": "C4"}
    mk = {1: "o", 2: "s", 3: "^", 5: "D", 10: "v", 25: "P"}
    for _, r in R.iterrows():
        ax.scatter(r.pred_pop, r.obs, c=col[r.method], marker=mk[int(r.k)], s=22, edgecolor="k", linewidth=0.3, alpha=0.9)
    lim = [R[["pred_pop", "obs"]].values.min() * 0.8, R[["pred_pop", "obs"]].values.max() * 1.25]
    ax.plot(lim, lim, "k--", lw=0.8, label="y = x"); ax.plot(lim, [s_pop * l for l in lim], "r-", lw=0.7, label=f"fit through origin, slope {s_pop:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"predicted $E|\mathrm{bias}| = \sqrt{2/\pi}\,\sigma_{\mathrm{pop}}/\sqrt{k}$ (counts)"); ax.set_ylabel("observed |bias| C_L,p (counts, mean over draws)")
    h = [plt.Line2D([], [], marker="o", ls="", c=c, label=m) for m, c in col.items()] + [plt.Line2D([], [], marker=m, ls="", c="w", markeredgecolor="k", label=f"k = {k}") for k, m in mk.items()]
    l1 = ax.legend(handles=h, fontsize=6, frameon=False, loc="upper left", ncol=2); ax.add_artist(l1); ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.savefig(OUT / "F10_floor.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "F10_floor.pdf", bbox_inches="tight")
    pd.DataFrame([{"statistic": "slope_pop", "value": s_pop}, {"statistic": "r2_pop", "value": r2_pop}, {"statistic": "r2_pop_uncentred", "value": r2u_pop}, {"statistic": "frac_in_range_pop", "value": frac},
                  {"statistic": "slope_k", "value": s_k}, {"statistic": "r2_k", "value": r2_k}, {"statistic": "frac_in_range_k", "value": frac_k}, {"statistic": "mean_ratio_k_le_2", "value": low.ratio_pop.mean()},
                  {"statistic": "slope_posthoc", "value": s_ph}, {"statistic": "r2_posthoc", "value": r2_ph}, {"statistic": "cells", "value": len(R)}]).round(6).to_csv(OUT / "T1_stats.csv", index=False)
    pd.DataFrame([{"source": n, "sigma": s, **{f"k_{t}": (C * s / t) ** 2 for t in (10, 20, 28)}} for n, s in sig.items()]).round(4).to_csv(OUT / "T1_kneeded.csv", index=False)
    # report
    tab = [{"source": r.source, "method": r.method, "family": r.family, "fold": r.fold, "k": int(r.k), "draws": int(r.n_draws), "obs": f"{r.obs:.1f}", "sigma_pop": f"{r.sigma_pop:.1f}", "pred_pop": f"{r.pred_pop:.1f}", "obs/pred (pop)": f"{r.ratio_pop:.2f}",
            "sigma_k": "-" if np.isnan(r.sigma_k) else f"{r.sigma_k:.1f}", "pred_k": "-" if np.isnan(r.pred_k) else f"{r.pred_k:.1f}", "obs/pred (k)": "-" if np.isnan(r.ratio_k) else f"{r.ratio_k:.2f}"} for _, r in R.iterrows()]
    by_k = R.groupby("k").ratio_pop.agg(["mean", "min", "max", "count"]).round(3)
    by_m = R.groupby("method").ratio_pop.agg(["mean", "min", "max", "count"]).round(3)
    L = ["# Label-budget study, task T1: statistical-floor hypothesis E|bias| = sqrt(2/pi) sigma / sqrt(k) (registered in PREREG)", "",
         f"Gate: M2 E2A 3-seed bias = +183.9 (assert in `scripts/correction/warm_statistics_cache.py`): PASS. {len(R)} cells ({(R.source == 'Two-stage correction study T1').sum()} the two-stage correction study T1, {(R.source == 'Component and cross-model study T2').sum()} the component and cross-model study T2, {(R.source == 'Component and cross-model study T3').sum()} the component and cross-model study T3); 'obs' = |bias C_L,p| mean over draws, 'pred' = mean over draws of sqrt(2/pi) sigma/sqrt(k).", "",
         "## Registered statistics (sigma_pop, all cells)", "",
         md([{"statistic": "slope through the origin", "value": f"{s_pop:.3f}", "registered range for 'kept'": "[0.8; 1.25]"}, {"statistic": "R^2 (centred total sum of squares)", "value": f"{r2_pop:.3f}", "registered range for 'kept'": ">= 0.7 (rejected if < 0.4)"},
             {"statistic": "R^2 uncentred (not the criterion)", "value": f"{r2u_pop:.3f}", "registered range for 'kept'": "-"}, {"statistic": "cells with obs/pred in [0.6; 1.6]", "value": f"{100 * frac:.0f}% ({int(((R.ratio_pop >= 0.6) & (R.ratio_pop <= 1.6)).sum())} of {len(R)})", "registered range for 'kept'": ">= 80%"}]), "",
         f"**Verdict by the registered rule: {verdict}** (kept needs all three; rejected if slope outside [0.6; 1.6] or R^2 < 0.4).", "",
         f"Secondary statistics with sigma_k (cells with k >= 2, n = {int(ok_k.sum())}): slope {s_k:.3f}, R^2 {r2_k:.3f} (uncentred {r2u_k:.3f}), cells with obs/pred in [0.6; 1.6]: {100 * frac_k:.0f}%.", "",
         f"Registered prediction 'kept; for k <= 2 obs/pred > 1': verdict {verdict}; mean obs/pred over the cells with k <= 2 = {low.ratio_pop.mean():.2f} (n = {len(low)}) -> {'TRUE' if pred_low else 'FALSE'} for the k <= 2 part.", "",
         f"Post-hoc diagnostic, **not registered** and not part of the verdict: adding the sampling error of the 125-car evaluation mean in quadrature (pred = sqrt(2/pi) sigma sqrt(1/k + 1/125)) gives slope {s_ph:.3f}, R^2 {r2_ph:.3f}, cells in [0.6; 1.6]: {100 * frac_ph:.0f}%.", "",
         "## obs/pred (sigma_pop) by k and by method", "", md([{"k": int(i), **r.to_dict()} for i, r in by_k.iterrows()]), "", md([{"method": i, **r.to_dict()} for i, r in by_m.iterrows()]), "",
         "## k needed for E|bias| <= 10, 20, 28 counts (formula k = (sqrt(2/pi) sigma / target)^2)", "", md(need), "",
         "## Cells", "", md(tab), "", "Known limit registered before the run: the observed |bias| is measured on 125 evaluation cars, so it also contains the sampling error of that mean (about sigma/sqrt(125), about 6 counts for sigma = 70), which is not in the formula.", "",
         "![F10](F10_floor.png)", ""]
    (OUT / "T1_floor.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
