"""Two-stage correction study, task T4: diagnostic tables and figures (CPU, from existing results). Output: results/two_stage/T4/ (tables CSV + markdown, figures PNG 300 dpi + PDF).
Usage: python scripts/diagnosis/diagnostic_figures.py"""
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
from scripts.diagnosis.force_alignment import load_pool  # noqa: E402

OUT = ROOT / "results" / "two_stage" / "T4"
RES = ROOT / "results"
REG = ("wheels", "base", "underbody", "roof_rear", "roof_front", "front", "sides", "other")
REGN = ("wheels", "base", "underbody", "rear roof", "front roof", "front", "sides", "other")  # Figure-label study: names of the paper on the axes
DIRS = {"M2": [f"{{f}}_M2_R2_s{s}" for s in range(3)], "B0": ["{f}_B0_NA_s0"], "B1": ["{f}_B1_R1_s0"], "B2": ["{f}_B2_R1_s0"], "B2_300ep": ["{f}_B2_R1_s0_ep300"], "B3": ["{f}_B3_R1_s0_lr3e-3"]}
B = 2000
rng = np.random.default_rng(0)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def ci(x):
    x = np.asarray(x)
    b = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return np.percentile(b, [2.5, 97.5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    T = pd.read_csv(RES / "decomposition" / "T1" / "per_car_terms_v2.csv.gz")
    C = pd.read_csv(RES / "force_alignment" / "cfd_coefficients.csv").set_index("id")
    O = pd.read_csv(RES / "decomposition" / "T4" / "T4_overshoot.csv")
    A = T.drop(columns=["seed", "role"]).groupby(["fold", "model", "id"]).mean(numeric_only=True).reset_index()
    assert abs(A[(A.fold == "E2A") & (A.model == "M2")].dL.mean() - 183.9) <= 0.5
    # 1. bias anatomy
    rows = []
    for f in ("E2A", "E2B", "E2C"):
        for m in DIRS:
            g = A[(A.fold == f) & (A.model == m)]
            o = O[(O.q == "C_L,p") & (O.fold == f) & (O.model == m)].iloc[0]
            od = O[(O.q == "C_D,p") & (O.fold == f) & (O.model == m)].iloc[0]
            rel = np.mean([pd.read_csv(RES / d.format(f=f) / "test_per_car.csv").rel_l2.mean() for d in DIRS[m]])
            same = float((np.sign(g.dL) == np.sign(g.dL.mean())).mean())
            row = {"fold": f, "model": m, "bias C_L,p": g.dL.mean(), "CI lo": ci(g.dL)[0], "CI hi": ci(g.dL)[1], "bias C_D,p": g.dD.mean(), "G (L)": o.G, "overshoot (L)": o.overshoot, "slope (L)": o.slope, "G (D)": od.G, "overshoot (D)": od.overshoot,
                   "mean ||e||_A": g.e_norm.mean(), "Rel L2 (mean over cars, seeds)": rel, "share of cars with the sign of the bias": same}
            tot = g.dL.mean()
            for r in REG:
                row[f"{r} (% of bias)"] = 100 * g[f"reg_dL_{r}"].mean() / tot if abs(tot) > 1 else np.nan
            rows.append(row)
    A1 = pd.DataFrame(rows)
    A1.round(4).to_csv(OUT / "T4_1_bias_anatomy.csv", index=False)
    pr = A1.copy()
    for c in pr.columns[3:]:
        pr[c] = pr[c].map(lambda v: f"{v:.4g}")
    # 2. invariance
    p1 = pd.read_csv(RES / "zero_shot_checks" / "T1" / "T1_results.csv")
    p1 = p1[(p1.fold == "E2A") & (p1.role == "test")]
    ab = pd.read_csv(RES / "zero_shot_checks" / "T1" / "E2A_training_ablations_bias.csv")
    loc = pd.read_csv(RES / "decomposition" / "T6" / "T6_summary.csv") if (RES / "decomposition" / "T6" / "T6_summary.csv").exists() else None
    inv = [{"intervention": r["model"], "kind": "training ablation", "bias C_L,p": r["bias C_L,p"], "CI": "-", "seeds": r["seeds"], "environment": "Kaggle"} for _, r in ab.iterrows()]
    inv.append({"intervention": "M2 + loc (6 local features)", "kind": "training ablation", "bias C_L,p": 190.9, "CI": "[+181.4; +200.0]", "seeds": 3, "environment": "Modal"})
    inv.append({"intervention": "M2 original rerun on Modal (seed 0)", "kind": "environment control", "bias C_L,p": np.nan, "CI": "see the two-stage correction study T6", "seeds": 1, "environment": "Modal"})
    for _, r in p1.iterrows():
        inv.append({"intervention": f"inference variant {r['variant']}", "kind": "inference intervention", "bias C_L,p": r.dL, "CI": f"[{r.dL_lo:+.1f}; {r.dL_hi:+.1f}]", "seeds": 3, "environment": "Kaggle checkpoints, local inference"})
    A2 = pd.DataFrame(inv)
    A2.round(2).to_csv(OUT / "T4_2_invariance.csv", index=False)
    # 3. references
    T3n = pd.read_csv(RES / "decomposition" / "T3" / "T3_label_noise.csv")
    e1 = np.mean([pd.read_csv(RES / f"E1_M2_R2_s{s}" / "test_per_car.csv").rel_l2.mean() for s in range(3)])
    e2 = np.mean([pd.read_csv(RES / f"E2A_M2_R2_s{s}" / "test_per_car.csv").rel_l2.mean() for s in range(3)])
    A3 = pd.DataFrame([{"reference": "E1 in-distribution, 25 Estateback test cars", "RMSE after bias removal C_L,p (counts)": 49.9, "bias C_L,p": 1.6, "Rel L2 (all 75 E1 test cars)": e1, "source": "Decomposition and first corrections study T3"},
                       {"reference": "E2A held-out style, 250 Estateback cars", "RMSE after bias removal C_L,p (counts)": 70.4, "bias C_L,p": 183.9, "Rel L2 (all 250 cars)": e2, "source": "Decomposition and first corrections study T3"},
                       {"reference": "label noise of C_L,p (full 100k pool, from 20 half/half splits)", "RMSE after bias removal C_L,p (counts)": 10.1, "bias C_L,p": np.nan, "Rel L2": np.nan, "source": "Decomposition and first corrections study T3"}])
    A3.round(4).to_csv(OUT / "T4_3_references.csv", index=False)
    # 4. bias vs G
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    cols = dict(zip(DIRS, ["C0", "C1", "C2", "C3", "C4", "C5"]))
    mk = {"E2A": "o", "E2B": "s", "E2C": "^"}
    for ax, q, u in zip(axs, ("C_L,p", "C_D,p"), ("lift", "drag")):
        d = O[O.q == q]
        for _, r in d.iterrows():
            ax.scatter(r.G, r.bias, c=cols[r.model], marker=mk[r.fold], s=28, edgecolor="k", linewidth=0.4)
        lim = np.array([d[["G", "bias"]].values.min(), d[["G", "bias"]].values.max()])
        ax.plot(lim, lim, "k--", lw=0.7)
        ax.axhline(0, c="gray", lw=0.5); ax.axvline(0, c="gray", lw=0.5)
        ax.set_xlabel(f"G (train mean - test mean of CFD {q}, counts)"); ax.set_ylabel(f"bias of predicted {q} (counts)"); ax.set_title(f"{u}: bias against G; dashed: overshoot = 0")
    h = [plt.Line2D([], [], marker="o", ls="", c=c, label=m) for m, c in cols.items()] + [plt.Line2D([], [], marker=v, ls="", c="w", markeredgecolor="k", label=k) for k, v in mk.items()]
    axs[1].legend(handles=h, fontsize=6, ncol=2, frameon=False)
    save(fig, "T4_4_bias_vs_G")
    # 5. scatter E1 / E2A
    T0 = pd.read_csv(RES / "force_alignment" / "per_car_terms.csv.gz")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, f, ttl in zip(axs, ("E1", "E2A"), ("E1 (in-distribution), Estateback test cars", "E2A (Estateback held out)")):
        g = T0[(T0.fold == f) & (T0.model == "M2")].groupby("id").dL.mean()
        if f == "E1":
            g = g[[i for i in g.index if i[0] == "E"]]
        ref = C.loc[g.index, "CL"].values
        pred = ref + g.values
        ax.scatter(ref, pred, s=8, alpha=0.7)
        lim = [min(ref.min(), pred.min()), max(ref.max(), pred.max())]
        ax.plot(lim, lim, "k--", lw=0.7, label="y = x")
        sl, ic = np.polyfit(ref, pred, 1)
        ax.plot(lim, np.polyval([sl, ic], lim), "r-", lw=0.9, label=f"fit slope {sl:.2f}")
        ax.set_xlabel("CFD C_L,p (counts)"); ax.set_ylabel("predicted C_L,p, M2 3-seed mean (counts)"); ax.set_title(f"{ttl}, n = {len(ref)}"); ax.legend(frameon=False, fontsize=7)
    save(fig, "T4_5_scatter_E1_E2A")
    # 6. surface error maps, 3 Estateback cars (median / largest / smallest dL), M2 3-seed mean error
    g = A[(A.fold == "E2A") & (A.model == "M2")].set_index("id").dL.sort_values()
    picks = {"median": g.index[len(g) // 2], "largest dC_L": g.index[-1], "smallest dC_L": g.index[0]}
    preds = [np.load(RES / f"E2A_M2_R2_s{s}" / "test_pred_fp16.npy", mmap_mode="r") for s in range(3)]
    pos = [{i: j for j, i in enumerate(pd.read_csv(RES / f"E2A_M2_R2_s{s}" / "test_per_car.csv").id)} for s in range(3)]
    fig, axs = plt.subplots(3, 3, figsize=(10, 7.2))
    for r, (lab, sid) in enumerate(picks.items()):
        xyz, nrm, cp, _ = load_pool(sid)
        e = np.mean([np.asarray(preds[s][pos[s][sid]], np.float64) for s in range(3)], axis=0) - np.clip(cp.astype(np.float64), -10, 1)
        x, y, z = (xyz.astype(np.float64) * 5.0).T
        views = (("side (y > 0)", y > 0, x, z), ("top (n_z > 0)", nrm[:, 2] > 0, x, y), ("bottom (n_z < 0)", nrm[:, 2] < 0, x, y))
        for c, (nm, m, u, v) in enumerate(views):
            d = np.argsort((y if c == 0 else z)[m] * (1 if c == 0 else -1 if c == 2 else 1))
            sc = axs[r, c].scatter(u[m][d], v[m][d], c=e[m][d], s=0.6, cmap="RdBu_r", vmin=-0.5, vmax=0.5, rasterized=True)
            axs[r, c].set_aspect("equal"); axs[r, c].axis("off")
            axs[r, c].set_title(f"{sid}\n{lab} (dC_L {g[sid]:+.0f}), {nm}", fontsize=7)
    fig.colorbar(sc, ax=axs, shrink=0.6, label="Cp_pred - Cp_true (M2 3-seed mean)")
    save(fig, "T4_6_error_maps")
    # 7. regional contribution of the bias and of the between-car variance
    g = A[(A.fold == "E2A") & (A.model == "M2")]
    tot = g.dL.values
    cb = [g[f"reg_dL_{r}"].mean() for r in REG]
    cl = np.array([ci(g[f"reg_dL_{r}"])[0] for r in REG]); ch = np.array([ci(g[f"reg_dL_{r}"])[1] for r in REG])
    vs = [np.cov(g[f"reg_dL_{r}"], tot)[0, 1] / tot.var(ddof=1) * 100 for r in REG]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axs[0].bar(REGN, cb, yerr=[np.array(cb) - cl, ch - np.array(cb)], color="C0", capsize=2); axs[0].set_ylabel("contribution to bias C_L,p (counts)"); axs[0].set_title("M2, E2A: bias by region")
    axs[1].bar(REGN, vs, color="C1"); axs[1].set_ylabel("share of Var(dC_L,p) between cars (%)"); axs[1].set_title("variance share by region")
    for a_ in axs:
        a_.tick_params(axis="x", rotation=45)
    save(fig, "T4_7_region_contributions")
    pd.DataFrame({"region": REG, "bias_counts": cb, "bias_lo": cl, "bias_hi": ch, "variance_share_pct": vs}).round(3).to_csv(OUT / "T4_7_region_contributions.csv", index=False)
    (OUT / "T4_tables.md").write_text("\n".join([
        "# Two-stage correction study, task T4 tables (generated by scripts/diagnosis/diagnostic_figures.py)", "", "## 1. Bias anatomy: 6 model families x 3 folds (C_L,p in counts; Rel L2 = mean over cars and seeds of the per-car field Rel L2; regions v2, % of the bias)", "", md(pr.to_dict("records")), "",
        "## 2. Invariance of the bias (E2A, M2 and variants)", "", md(A2.round(2).fillna("-").to_dict("records")), "",
        "Definitions of the inference variants (a, b, c, d, eb, ec, ed) are in the intervention script and `results/zero_shot_checks/T1/T1_summary.md`. The only change of environment in this table is the `loc` row (Modal).", "",
        "## 3. References", "", md(A3.round(4).fillna("-").to_dict("records")), ""]), encoding="utf-8")
    print("done", picks)


if __name__ == "__main__":
    main()
