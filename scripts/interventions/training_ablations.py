"""Zero-shot checks study, task T1 extra (existing runs, no new training): C_L,p bias on E2A of the M2 training-level ablations (no decoder xyz, no normals, occupancy, grid R3) from their saved predictions,
for comparison with the inference-time interventions. Writes results/zero_shot_checks/T1/E2A_training_ablations_bias.csv. Usage: python scripts/interventions/training_ablations.py"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.force_alignment import terms_table  # noqa: E402

R = ROOT / "results"
rows = []
for name, pat, seeds in (("M2 R2 (reference)", "E2A_M2_R2_s{s}", (0, 1, 2)), ("M2 no decoder xyz", "E2A_M2NOXYZ_R2_s{s}", (0, 1, 2)), ("M2 no normals", "E2A_M2NON_R2_s{s}", (0,)),
                         ("M2 occupancy", "E2A_M2OCC_R2_s{s}", (0, 1, 2)), ("M2 R3", "E2A_M2_R3_s{s}", (0, 1, 2))):
    d = terms_table("E2A", "test", [(name, s, R / pat.format(s=s)) for s in seeds])
    a = d.drop(columns=["seed", "role", "model", "fold"]).groupby("id").mean(numeric_only=True)
    rows.append((name, d.seed.nunique(), a.dL.mean(), a.rho_L.mean(), a.e_norm.mean(), a.reg_dL_roof_rear.mean(), a.reg_dL_underbody.mean()))
pd.DataFrame(rows, columns=["model", "seeds", "bias C_L,p", "rho_L", "||e||", "roof_rear contribution", "underbody contribution"]).round(3).to_csv(ROOT / "results" / "zero_shot_checks" / "T1" / "E2A_training_ablations_bias.csv", index=False)
