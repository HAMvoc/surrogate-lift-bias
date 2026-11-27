"""Per-car C_D,p and C_L,p (CFD labels and model predictions, counts) for every held-out-style model run with saved predictions -> reports/integral_errors.csv.
Input to scripts/correction/conformal_first_look.py (prereg addendum 5). Usage: python scripts/diagnosis/integral_errors.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.interventions.integral_loss_and_long_training import pool  # noqa: E402
from scripts.diagnosis.pressure_coefficients import cdp  # noqa: E402
from scripts.diagnosis.physics_plots import cl  # noqa: E402

RES = ROOT / "results"
RUNS = [("M2", s, f"{{f}}_M2_R2_s{s}") for s in (0, 1, 2)] + [("B1", 0, "{f}_B1_R1_s0"), ("B2", 0, "{f}_B2_R1_s0"), ("B2_300ep", 0, "{f}_B2_R1_s0_ep300"), ("B3", 0, "{f}_B3_R1_s0_lr3e-3")]


def main():
    rows = []
    for f in ("E2A", "E2B", "E2C"):
        for model, seed, pat in RUNS:
            d = RES / pat.format(f=f)
            if not (d / "test_pred_fp16.npy").exists():
                print("missing", d.name)
                continue
            ids = list(pd.read_csv(d / "test_per_car.csv").id)
            pr = np.load(d / "test_pred_fp16.npy", mmap_mode="r")
            for n, i in enumerate(ids):
                p, ref, rd, rl = pool(i)
                c = np.asarray(pr[n], np.float64)
                rows.append((f, model, seed, i, rl, 1000 * cl(c, p["normal"][:, 2], p["area"], i, np.ones(len(c))), rd, 1000 * cdp(c, p["normal"][:, 0], p["area"], i)))
            print(f, model, seed, len(ids), flush=True)
    pd.DataFrame(rows, columns=["fold", "model", "seed", "id", "cl_ref", "cl_pred", "cd_ref", "cd_pred"]).round(3).to_csv(ROOT / "reports" / "integral_errors.csv", index=False)


if __name__ == "__main__":
    main()
