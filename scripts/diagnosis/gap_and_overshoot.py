"""Decomposition and first corrections study, task T4: bias = G + overshoot for every model family and fold. G = mean CFD C_L,p (train cars) - (test cars); overshoot = mean predicted C_L,p on the test cars - mean CFD of the TRAIN cars.
Also the shrinkage slope. Output: results/decomposition/T4/. Usage: python scripts/diagnosis/gap_and_overshoot.py"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.force_alignment_terms import COL, MK, fm  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

OUT = ROOT / "results" / "decomposition" / "T4"
B = 2000
FOLDS = ("E2A", "E2B", "E2C")
MODELS = ("B0", "B1", "B2", "B2_300ep", "B3", "M2")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(ROOT / "results" / "force_alignment" / "per_car_terms.csv.gz")
    C = pd.read_csv(ROOT / "results" / "force_alignment" / "cfd_coefficients.csv").set_index("id")
    A = T[T.fold.isin(FOLDS)].drop(columns=["seed", "role"]).groupby(["fold", "model", "id"]).mean(numeric_only=True).reset_index()
    assert abs(A[(A.fold == "E2A") & (A.model == "M2")].dL.mean() - 183.9) <= 0.5
    rows, num = [], []
    rng = np.random.default_rng(0)
    for q, dcol, ref in (("C_L,p", "dL", "CL"), ("C_D,p", "dD", "CD")):
        for f in FOLDS:
            s = load_split(f)
            tr, te = C.loc[s["train"], ref], C.loc[s["test"], ref]
            G = tr.mean() - te.mean()
            gb = [tr.values[rng.integers(0, len(tr), len(tr))].mean() - te.values[rng.integers(0, len(te), len(te))].mean() for _ in range(B)]
            for m in MODELS:
                g = A[(A.fold == f) & (A.model == m)].set_index("id")
                cfd_te = C.loc[g.index, ref].values
                pred = cfd_te + g[dcol].values
                over = pred - tr.mean()
                bias = g[dcol].values
                idx = rng.integers(0, len(g), (B, len(g)))
                ob = over[idx].mean(1)
                sl = [np.polyfit(cfd_te[i], pred[i], 1)[0] for i in idx[:500]]
                num.append({"q": q, "fold": f, "model": m, "G": G, "G_lo": np.percentile(gb, 2.5), "G_hi": np.percentile(gb, 97.5), "overshoot": over.mean(), "o_lo": np.percentile(ob, 2.5), "o_hi": np.percentile(ob, 97.5),
                            "bias": bias.mean(), "slope": np.polyfit(cfd_te, pred, 1)[0], "s_lo": np.percentile(sl, 2.5), "s_hi": np.percentile(sl, 97.5)})
                rows.append({"quantity": q, "fold": f, "model": m, "G [CI]": f"{G:+.1f} [{np.percentile(gb, 2.5):+.1f}; {np.percentile(gb, 97.5):+.1f}]", "overshoot [CI]": f"{over.mean():+.1f} [{np.percentile(ob, 2.5):+.1f}; {np.percentile(ob, 97.5):+.1f}]",
                             "bias": f"{bias.mean():+.1f}", "G + overshoot": f"{G + over.mean():+.1f}", "slope [CI]": f"{num[-1]['slope']:.2f} [{num[-1]['s_lo']:.2f}; {num[-1]['s_hi']:.2f}]"})
    N = pd.DataFrame(num)
    N.round(3).to_csv(OUT / "T4_overshoot.csv", index=False)
    chk = (N.bias - (N.G + N.overshoot)).abs().max()
    # figure: overshoot by family
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, f in zip(axs, FOLDS):
        d = N[(N.q == "C_L,p") & (N.fold == f)].set_index("model")
        for j, m in enumerate(MODELS):
            ax.errorbar(j, d.loc[m, "overshoot"], yerr=[[d.loc[m, "overshoot"] - d.loc[m, "o_lo"]], [d.loc[m, "o_hi"] - d.loc[m, "overshoot"]]], fmt=MK[m], color=COL[m], capsize=3, ms=6)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{f}: overshoot = mean prediction - mean CFD of the train cars (C_L,p, counts)", fontsize=7)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("overshoot (counts)")
    fig.tight_layout()
    fig.savefig(OUT / "T4_overshoot_by_family.png", dpi=200)
    plt.close(fig)
    # B1 rule
    b1 = N[(N.q == "C_L,p") & (N.fold == "E2A") & (N.model == "B1")].iloc[0]
    m2 = N[(N.q == "C_L,p") & (N.fold == "E2A") & (N.model == "M2")].iloc[0]
    ratio = b1.overshoot / m2.overshoot
    verdict = "NO" if b1.o_lo <= 0 else ("YES, a substantial part (>= 50% of M2's overshoot) is already in the geometric neighbour relation of the data" if ratio >= 0.5 else "YES, but a small part (< 50% of M2's)")
    L = ["# Decomposition and first corrections study, task T4: bias = G + overshoot, all model families", "",
         f"Gate: M2 E2A 3-seed bias C_L,p = {A[(A.fold == 'E2A') & (A.model == 'M2')].dL.mean():+.1f} counts: PASS. G = mean CFD of the fold's training cars minus its test cars; overshoot = mean predicted coefficient on the test cars minus the mean CFD of the TRAIN cars; bias = G + overshoot (largest difference found: {chk:.1e} counts). "
         "Per-car predictions are seed-averaged (M2: 3 seeds, others 1); CIs over cars (G: over train and test cars). Slope = regression of predicted on CFD coefficient over the test cars.", "",
         md([r for r in rows if r["quantity"] == "C_L,p"]), "", "## C_D,p", "", md([r for r in rows if r["quantity"] == "C_D,p"]), "",
         f"## Question: does the kNN on SDF (B1, no learned decoder) have a positive overshoot on E2A?", "",
         f"B1 overshoot {b1.overshoot:+.1f} [{b1.o_lo:+.1f}; {b1.o_hi:+.1f}] counts; M2 {m2.overshoot:+.1f}; ratio B1/M2 = {ratio:.2f}. Registered rule: positive if the CI lower bound > 0; a substantial part if the ratio >= 0.5. **Answer: {verdict}.** "
         "(The B1 number was already known from the force-alignment study, see PREREG.) Caution: B1 transplants the Cp fields of training cars and integrates them with the test car's own normals, so a non-zero overshoot also reflects the change of geometry under the integral, not only a bias of a learned model; "
         "the data cannot separate these two within this analysis. B0 (mean template field) behaves like B1 on E2A.", "", "![overshoot](T4_overshoot_by_family.png)", ""]
    (OUT / "T4_report.md").write_text("\n".join(L), encoding="utf-8")
    print(verdict, ratio)


if __name__ == "__main__":
    main()
