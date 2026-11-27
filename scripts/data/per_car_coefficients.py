"""Classical-baseline and regional-gap study: per-car CFD C_L,p and C_D,p (counts), and the contribution of each v2 region to C_L,p, for the 750 cars of the subset (inputs of W1 and W2).
Same definitions as scripts/correction/calibration.py:build_stats (Cp clipped, IPW pool weights, S = total area / frontal area); regions: the stored labels results/decomposition/T1/labels_v2.npz.
Self-check: the sum of the regional contributions equals the per-car C_L,p to 1e-6, and the E2A test-car mean of C_L,p differs from the training-car mean by G = 112.755 (T4_overshoot.csv) to 0.01.
Output: results/classical_baselines/cars.csv. Usage: python scripts/data/per_car_coefficients.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.pressure_coefficients import CLIP, GEO, pool_weights  # noqa: E402
from scripts.diagnosis.force_alignment import load_pool  # noqa: E402
from scripts.diagnosis.regions import REGIONS2  # noqa: E402
from src.data.dataset import category_of, load_split  # noqa: E402

OUT = ROOT / "results" / "classical_baselines"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labs = np.load(ROOT / "results" / "decomposition" / "T1" / "labels_v2.npz")
    man = pd.read_csv(ROOT / "data" / "manifests" / "manifest.csv").query("status == 'ok'")
    sp = load_split("E2A")
    role = {i: r for r in ("train", "val", "test") for i in sp[r]}
    nreg = len(REGIONS2)
    rows = []
    for sid in man.sample_id:
        _, nrm, cp, area = load_pool(sid)
        ref = np.clip(cp.astype(np.float64), *CLIP)
        w = pool_weights(np.asarray(area, np.float64), sid)
        sw = w.sum()
        S = float(GEO.at[sid, "area_total_m2"] / GEO.at[sid, "frontal_area_m2"])
        lab = labs[sid].astype(np.int64)
        cz = -1000 * S * np.bincount(lab, weights=w * ref * nrm[:, 2], minlength=nreg + 1)[:nreg] / sw
        cx = -1000 * S * (w * ref * nrm[:, 0]).sum() / sw
        rows.append({"sample_id": sid, "category": category_of(sid), "role": role.get(sid, "none"), "CL": cz.sum(), "CD": cx, **{f"cl_{r}": cz[k] for k, r in enumerate(REGIONS2)}})
    D = pd.DataFrame(rows)
    assert np.allclose(D[[f"cl_{r}" for r in REGIONS2]].sum(1), D.CL, atol=1e-6)
    tr, te = D[D.role == "train"].CL.mean(), D[D.role == "test"].CL.mean()
    g = pd.read_csv(ROOT / "results" / "decomposition" / "T4" / "T4_overshoot.csv").query("q == 'C_L,p' and fold == 'E2A' and model == 'M2'").G.iloc[0]
    assert abs((tr - te) - g) < 0.01, (tr - te, g)
    assert (D.role == "none").sum() == 0 and D.role.value_counts().to_dict() == {"train": len(sp["train"]), "test": len(sp["test"]), "val": len(sp["val"])}
    D.round(6).to_csv(OUT / "cars.csv", index=False)
    print(len(D), "cars; G =", round(tr - te, 3))


if __name__ == "__main__":
    main()
