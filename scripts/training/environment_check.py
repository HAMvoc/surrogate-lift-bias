"""Two-stage correction study, task T6: M2 original on Modal (E2A, seed 0, 150 epochs) against the three Kaggle seeds. Output results/two_stage/T6/T6_report.md. Usage: python scripts/training/environment_check.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md
from scripts.diagnosis.force_alignment import terms_table

M = ROOT / "results" / "two_stage" / "T2" / "raw" / "results" / "E2A_M2_R2_s0_modal"
runs = [("kaggle", s, ROOT / "results" / f"E2A_M2_R2_s{s}") for s in range(3)] + [("modal", 0, M)]
T = terms_table("E2A", "test", runs)
rows = []
for env, s, d in runs:
    g = T[(T.model == env) & (T.seed == s)]
    rows.append({"env": env, "seed": s, "bias C_L,p": g.dL.mean(), "bias C_D,p": g.dD.mean(), "Rel L2 test (mean over 250 cars)": pd.read_csv(d / "test_per_car.csv").rel_l2.mean(), "Rel L2 validation (mean over 50 cars)": pd.read_csv(d / "val_per_car.csv").rel_l2.mean()})
R = pd.DataFrame(rows)
k = R[R.env == "kaggle"]
assert abs(k["bias C_L,p"].mean() - 183.9) <= 0.5
m = R[R.env == "modal"].iloc[0]; k0 = k[k.seed == 0].iloc[0]
L = ["# Two-stage correction study, task T6: environment difference Kaggle vs Modal", "",
     "M2 original (cfg_hash 26006c2f90cb), E2A, seed 0, 150 epochs on Modal (torch 2.10, Python 3.11, T4) against the Kaggle runs (Python 3.13.16, torch 2.11.0+cu130, T4). Gate: 3-seed Kaggle bias = +183.9: PASS.", "", md(R.round(4).to_dict("records")), ""]
tab = []
for c in R.columns[2:]:
    span = k[c].max() - k[c].min(); d = m[c] - k0[c]
    tab.append({"metric": c, "Kaggle seed 0": f"{k0[c]:.4f}", "Modal seed 0": f"{m[c]:.4f}", "difference": f"{d:+.4f}", "range of the 3 Kaggle seeds (max - min)": f"{span:.4f}", "inside the seed spread (PREREG rule |diff| <= range)": "inside" if abs(d) <= span else "OUTSIDE"})
L += [md(tab), ""]
(ROOT / "results" / "two_stage" / "T6" / "T6_report.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))
