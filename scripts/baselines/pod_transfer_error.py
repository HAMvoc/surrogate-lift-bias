"""POD-baseline study B, exploratory diagnostic (not pre-specified, added after the POD numbers were seen): how much of the POD field error comes from the nearest-point
transfer through the 20000 common points alone. For each validation and test car: its own CFD snapshot on R transferred back to its pool (no POD), Rel L2 and
C_L,p error. Also the mean Rel L2 of B0 and B1 on the same test cars from their stored per-car files. Output: results/pod/pod_transfer.csv, pod_transfer.md.
Usage: python scripts/baselines/pod_transfer_error.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.force_alignment import load_pool  # noqa: E402
from scripts.baselines.pod_reconstruction import NR, Car  # noqa: E402

OUT = ROOT / "results" / "pod"


def main():
    D = pd.read_csv(ROOT / "results" / "classical_baselines" / "cars.csv").set_index("sample_id")
    tr = list(D[D.role == "train"].index)
    rng = np.random.default_rng(0)  # identical draw of R as scripts/baselines/pod_reconstruction.py
    pools = {sid: np.asarray(load_pool(sid)[0], np.float64) for sid in tr}
    ci = rng.integers(0, len(tr), NR)
    R = np.array([pools[tr[c]][rng.integers(0, len(pools[tr[c]]))] for c in ci])
    del pools
    rows = []
    for role in ("val", "test"):
        for sid in D[D.role == role].index:
            c = Car(sid, R)
            cp = c.s[c.near]
            rows.append({"sample_id": sid, "role": role, "rel": c.rel(cp), "dCL": c.integrate(cp)[0] - D.at[sid, "CL"]})
    T = pd.DataFrame(rows)
    T.round(5).to_csv(OUT / "pod_transfer.csv", index=False)
    b = {m: pd.read_csv(ROOT / "results" / f"E2A_{m}_{g}_s0" / "test_per_car.csv").rel_l2.mean() for m, g in (("B0", "NA"), ("B1", "R1"))}
    s = T.groupby("role").agg(rel=("rel", "mean"), dCL=("dCL", "mean"))
    L = ["# POD-baseline study B, exploratory: transfer floor of the POD baseline (not pre-specified)", "",
         f"Own CFD field through R and back (no POD): Rel L2 validation {s.at['val', 'rel']:.3f}, test {s.at['test', 'rel']:.3f}; mean C_L,p error validation {s.at['val', 'dCL']:+.1f}, test {s.at['test', 'dCL']:+.1f} counts.",
         f"Stored Rel L2 on the E2A test cars: B0 {b['B0']:.3f}, B1 {b['B1']:.3f}.", ""]
    (OUT / "pod_transfer.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
