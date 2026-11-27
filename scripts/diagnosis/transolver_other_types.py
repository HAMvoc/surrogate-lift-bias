"""Transolver on all held-out types study, task T3: Transolver (seed 0) on E2B and E2C, (E2B: |bias C_L,p| <= 50 counts; E2C: description, label 'second error in the opposite direction' if |bias| > 100).
Reads results/{fold}_TSV_R1_s0/{test,val}_per_car.csv and test_pred_fp16.npy (copied from the Kaggle kernel output); every integral by scripts/diagnosis/force_alignment.py.
Output: results/transolver_types/T3/{T3_numeric.csv, T3_table.csv, T3_regions_v2.csv, T3_report.md}, results/transolver_types/T3/bias_anatomy_p8.csv and T4_overshoot_p8.csv (the _p7 files plus the Transolver rows of E2B and E2C).
Usage: python scripts/diagnosis/transolver_other_types.py [--folds E2B E2C] [--run-template "results/{fold}_TSV_R1_s0"] [--out results/transolver_types/T3]   (stand-in test: --folds E2A --out <tmp> reproduces seed 0 of the Transolver study)"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis import force_alignment as fa  # noqa: E402
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.transolver_bias import CFD, B, ci, fm  # noqa: E402
from scripts.diagnosis.regions import REGIONS2, labeler as labeler_v2  # noqa: E402
from src.data.dataset import load_split  # noqa: E402


def one(fold, tmpl):
    run = ROOT / tmpl.format(fold=fold)
    T = fa.terms_table(fold, "test", [("TSV", 0, run)], regions=True, labeler=labeler_v2, region_names=REGIONS2)
    s = load_split(fold)
    ids = sorted(T.id.unique())
    g = T.set_index("id").loc[ids]
    cfd = CFD.loc[ids]
    trL, trD = CFD.loc[s["train"], "CL"].mean(), CFD.loc[s["train"], "CD"].mean()
    dL, dD = g.dL.values, g.dD.values
    pred, predD = cfd.CL.values + dL, cfd.CD.values + dD
    G, GD = trL - cfd.CL.mean(), trD - cfd.CD.mean()
    over, overD = pred - trL, predD - trD
    assert abs(dL.mean() - (G + over.mean())) < 1e-6 and abs(dD.mean() - (GD + overD.mean())) < 1e-6
    rel_t = pd.read_csv(run / "test_per_car.csv").rel_l2.mean()
    rel_v = pd.read_csv(run / "val_per_car.csv").rel_l2.mean()
    reg = {r: g[f"reg_dL_{r}"].mean() for r in REGIONS2}
    ru = reg["roof_front"] + reg["roof_rear"] + reg["underbody"]
    cL, cD, co = ci(dL), ci(dD), ci(over)
    share = float((np.sign(dL) == np.sign(dL.mean())).mean())
    slope, sp = float(np.polyfit(cfd.CL.values, pred, 1)[0]), float(spearmanr(cfd.CL.values, pred)[0])
    num = {"fold": fold, "bias_L": cL[0], "bias_L_lo": cL[1], "bias_L_hi": cL[2], "bias_D": cD[0], "bias_D_lo": cD[1], "bias_D_hi": cD[2], "same_sign_share": share, "rel_l2_test": rel_t, "rel_l2_val": rel_v, "e_norm": g.e_norm.mean(), "rho_L": g.rho_L.mean(),
           "G": G, "overshoot": co[0], "overshoot_lo": co[1], "overshoot_hi": co[2], "slope": slope, "spearman": sp, "rmse_after_bias_removal": dL.std(), **{f"reg_{r}": v for r, v in reg.items()}, "roof_underbody_share_pct": 100 * ru / dL.mean(), "G_D": GD, "overshoot_D": overD.mean()}
    tab = {"fold": fold, "bias C_L,p [CI]": fm(cL), "bias C_D,p [CI]": fm(cD), "same-sign share": f"{share:.3f}", "Rel L2 test": f"{rel_t:.4f}", "Rel L2 val": f"{rel_v:.4f}", "||e||_A": f"{g.e_norm.mean():.4f}", "rho_L": f"{g.rho_L.mean():.3f}",
           "G": f"{G:+.1f}", "overshoot [CI]": fm(co), "slope": f"{slope:.2f}", "Spearman": f"{sp:.3f}", "RMSE after bias removal": f"{dL.std():.1f}"}
    rg = {"fold": fold, **{r: f"{v:+.1f} ({100 * v / dL.mean():.0f}%)" for r, v in reg.items()}, "roof+underbody share": f"{100 * ru / dL.mean():.0f}%"}
    return num, tab, rg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["E2B", "E2C"])
    ap.add_argument("--run-template", default="results/{fold}_TSV_R1_s0")
    ap.add_argument("--out", default="results/transolver_types/T3")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    res = [one(f, a.run_template) for f in a.folds]
    N = pd.DataFrame([r[0] for r in res])
    N.to_csv(out / "T3_numeric.csv", index=False)
    pd.DataFrame([r[1] for r in res]).to_csv(out / "T3_table.csv", index=False)
    pd.DataFrame([r[2] for r in res]).to_csv(out / "T3_regions_v2.csv", index=False)
    # seven-family tables with the Transolver rows of E2B and E2C (definitions of the _p7 files)
    A = pd.read_csv(ROOT / "paper_assets" / "tables" / "bias_anatomy_p7.csv")
    O = pd.read_csv(ROOT / "results" / "transolver" / "T5" / "T4_overshoot_p7.csv")
    rowsA, rowsO = [], []
    for _, n in N.iterrows():
        rowsA.append({"fold": n.fold, "model": "TSV", "bias C_L,p": n.bias_L, "CI lo": n.bias_L_lo, "CI hi": n.bias_L_hi, "bias C_D,p": n.bias_D, "G (L)": n.G, "overshoot (L)": n.overshoot, "slope (L)": n.slope, "G (D)": n.G_D, "overshoot (D)": n.overshoot_D,
                      "mean ||e||_A": n.e_norm, "Rel L2 (mean over cars, seeds)": n.rel_l2_test, "share of cars with the sign of the bias": n.same_sign_share, **{f"{r} (% of bias)": 100 * n[f"reg_{r}"] / n.bias_L for r in REGIONS2}})
        rowsO += [{"q": "C_L,p", "fold": n.fold, "model": "TSV", "G": n.G, "overshoot": n.overshoot, "o_lo": n.overshoot_lo, "o_hi": n.overshoot_hi, "bias": n.bias_L, "slope": n.slope}, {"q": "C_D,p", "fold": n.fold, "model": "TSV", "G": n.G_D, "overshoot": n.overshoot_D, "bias": n.bias_D}]
    A8 = pd.concat([A[~((A.model == "TSV") & A.fold.isin(N.fold))], pd.DataFrame(rowsA)], ignore_index=True)
    O8 = pd.concat([O[~((O.model == "TSV") & O.fold.isin(N.fold))], pd.DataFrame(rowsO)], ignore_index=True)
    A8.to_csv(out / "bias_anatomy_p8.csv", index=False); O8.to_csv(out / "T4_overshoot_p8.csv", index=False)
    nb = N.set_index("fold")
    L = ["# Transolver on all held-out types study, task T3: Transolver on E2B and E2C (seed 0, one run per fold)", "",
         f"Integrals: `scripts/diagnosis/force_alignment.py`. Bias = mean over the 250 test cars of predicted minus CFD (counts); CI car bootstrap B = {B}, percentile; the same-sign share is that of the single run against the sign of its own bias. No interval across seeds (one seed).", "",
         md(pd.DataFrame([r[1] for r in res]).to_dict("records")), "", "## Decomposition over the eight v2 regions (counts of C_L,p, share of the total bias)", "", md([r[2] for r in res]), "", "## Against the pre-registration", ""]
    if "E2B" in nb.index:
        b = nb.loc["E2B"].bias_L
        L.append(f"- **E2B** (prediction |bias C_L,p| <= 50 counts): bias {b:+.1f} [{nb.loc['E2B'].bias_L_lo:+.1f}; {nb.loc['E2B'].bias_L_hi:+.1f}] -> **{'held' if abs(b) <= 50 else 'not held'}**.")
    if "E2C" in nb.index:
        b = nb.loc["E2C"].bias_L
        lab = "**second error in the opposite direction** (|bias| > 100; B3 on E2C: -170.9)" if abs(b) > 100 and b < 0 else "no label (|bias| <= 100)" if abs(b) <= 100 else "|bias| > 100 but positive: the label 'opposite direction' does not apply (a large positive bias)"
        L.append(f"- **E2C** (description; label if |bias| > 100): bias {b:+.1f} [{nb.loc['E2C'].bias_L_lo:+.1f}; {nb.loc['E2C'].bias_L_hi:+.1f}], sign {'positive' if b > 0 else 'negative'}, size {abs(b):.1f} counts -> {lab}.")
    L.append("")
    (out / "T3_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
