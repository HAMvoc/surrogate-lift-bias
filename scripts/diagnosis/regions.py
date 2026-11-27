"""Regions v2 (Decomposition and first corrections study T1): the force-alignment study rules plus a `wheels` region found by fitting the tyre circle on each car's own pool (no part labels exist in the data).
Per axle (prior x_rel 0.173 front, 0.800 rear as in configs/regions.yaml): vote over (x_c, R) with x_c in prior +- 0.25 m and R in [0.28, 0.40] m (centre height z_min + R) for the
points with |y| > 0.45 m that lie within 6 mm of the circle in the x-z plane (the tyre tread); width per wheel from the 1st-99th percentile of |y| of the winning points; wheel region =
rho <= R + 0.02 m and |y| within the width +- 0.02 m. Priority: wheels > base > underbody > roof_rear > roof_front > front > sides > other.
Usage: python scripts/diagnosis/regions.py [--all]   (writes results/decomposition/T1/labels_v2.npz and wheel_fits.csv)"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.force_alignment import load_pool, region_labels  # noqa: E402

REGIONS2 = ("wheels", "base", "underbody", "roof_rear", "roof_front", "front", "sides", "other")
L_M = 5.0
PRIOR_U = (0.173, 0.800)
OUT = ROOT / "results" / "decomposition" / "T1"
_CACHE = {}


def fit_wheels(xyz):
    """xyz in cache units. Returns (mask of wheel points, list of per-axle fits)."""
    p = xyz.astype(np.float64) * L_M
    x0, L, zmin = p[:, 0].min(), np.ptp(p[:, 0]), p[:, 2].min()
    mask = np.zeros(len(p), bool)
    fits = []
    for ax, u in zip(("front", "rear"), PRIOR_U):
        xp = x0 + u * L
        sel = (np.abs(p[:, 1]) > 0.45) & (p[:, 2] < zmin + 0.85) & (np.abs(p[:, 0] - xp) < 0.75)
        q = p[sel]
        best = (-1, None, None)
        for xc in np.arange(xp - 0.25, xp + 0.2501, 0.005):
            dx = q[:, 0] - xc
            for R in np.arange(0.28, 0.4001, 0.005):
                rho = np.hypot(dx, q[:, 2] - (zmin + R))
                n = int((np.abs(rho - R) < 0.006).sum())
                if n > best[0]:
                    best = (n, xc, R)
        n, xc, R = best
        rho = np.hypot(q[:, 0] - xc, q[:, 2] - (zmin + R))
        inl = q[np.abs(rho - R) < 0.006]
        for side in (+1, -1):
            s = inl[np.sign(inl[:, 1]) == side]
            if len(s) < 20:
                fits.append({"axle": ax, "side": side, "x_c": xc, "R": R, "votes": n, "y_in": np.nan, "y_out": np.nan})
                continue
            y_in, y_out = np.percentile(np.abs(s[:, 1]), [1, 99])
            full = np.hypot(p[:, 0] - xc, p[:, 2] - (zmin + R))
            mask |= (full <= R + 0.02) & (np.sign(p[:, 1]) == side) & (np.abs(p[:, 1]) >= y_in - 0.02) & (np.abs(p[:, 1]) <= y_out + 0.02)
            fits.append({"axle": ax, "side": side, "x_c": xc, "R": R, "votes": n, "y_in": y_in, "y_out": y_out})
    return mask, fits


def labels_v2(xyz, nrm):
    mask, fits = fit_wheels(xyz)
    lab = region_labels(xyz, nrm) + 1  # old indices 0..6 -> 1..7
    lab[mask] = 0
    return lab, fits


def labeler(sid, xyz, nrm):
    """Cached labeller for terms_table (labels_v2.npz must exist)."""
    if "z" not in _CACHE:
        f = ROOT / "data" / "manifests" / "labels_v2.npz"  # copy shipped with the code dataset (Kaggle)
        _CACHE["z"] = np.load(f if f.exists() else OUT / "labels_v2.npz")
    return _CACHE["z"][sid].astype(np.int64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from src.data.dataset import load_split
    s = load_split("E1")
    ids = sorted(set(s["train"]) | set(s["val"]) | set(s["test"]))
    labs, rows, t0 = {}, [], time.time()
    for n, sid in enumerate(ids):
        xyz, nrm, _, _ = load_pool(sid)
        lab, fits = labels_v2(xyz, nrm)
        labs[sid] = lab.astype(np.uint8)
        for f in fits:
            rows.append({"id": sid, **f, "wheel_area_frac_points": float((lab == 0).mean())})
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(ids)} {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(OUT / "labels_v2.npz", **labs)
    pd.DataFrame(rows).to_csv(OUT / "wheel_fits.csv", index=False)
    print("done", len(ids), f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
