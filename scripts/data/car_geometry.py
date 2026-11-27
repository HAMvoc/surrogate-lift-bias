"""Per-car reference areas for the pressure-drag diagnostic (spec 10.5): total surface area of the CFD surface mesh (the mesh the Cp labels live on)
and projected frontal area A_ref. A_ref is taken from the cached R3 occupancy (cells of the y-z projection that contain any inside voxel centre) and
validated against a polygon rasterisation of the mesh (`scripts/data/check_cp_scale.py: frontal_area`) on a few cars.
Also stores the inclusion rate of the pool sampling (for inverse-probability weights). Writes data/manifests/car_geometry.csv. Usage: python scripts/data/car_geometry.py [--workers 4] [--validate 8]"""
import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from src.data.grid import load_cfg  # noqa: E402

CFG = load_cfg()
RAW = ROOT / "data" / "raw" / CFG["source"]["vtk_dir"]
OUT = ROOT / "data" / "manifests" / "car_geometry.csv"


def frontal_from_occupancy(sid):
    """Projected area (m^2) along x from sdf_R3: cell (z, y) counts if any voxel centre along x is inside (SDF < 0)."""
    v = np.load(ROOT / "data" / "cache" / sid / "sdf_R3.npy")  # (Z, Y, X), fp16
    box, (nx, ny, nz) = CFG["box"], CFG["grids"]["R3"]
    dy = (box["y"][1] - box["y"][0]) * CFG["frame"]["l_ref_m"] / ny
    dz = (box["z"][1] - box["z"][0]) * CFG["frame"]["l_ref_m"] / nz
    return float((v < 0).any(axis=2).sum() * dy * dz)


def inclusion_rate(va, n):
    """lambda with sum_v (1 - exp(-lambda * va_v)) = n: Poisson-style approximation of the inclusion probability of an area-proportional sample
    without replacement (numpy `choice(replace=False, p=...)` is not exactly proportional: large-area vertices are included less often than their share)."""
    from scipy.optimize import brentq
    return float(brentq(lambda lam: (1 - np.exp(-lam * va)).sum() - n, 1.0, 1e8))


def one(sid):
    import pyvista as pv
    m = pv.read(RAW / f"{sid}.vtk").triangulate()
    area = np.asarray(m.compute_cell_sizes(length=False, volume=False)["Area"])
    faces = m.faces.reshape(-1, 4)[:, 1:]
    va = np.bincount(faces.ravel(), weights=np.repeat(area / 3, 3), minlength=m.n_points)  # vertex area, as in preprocess
    return sid, float(area.sum()), frontal_from_occupancy(sid), inclusion_rate(va, min(CFG["pool"]["n_points"], m.n_points))


def validate(ids):
    """Compare the occupancy-based frontal area with a rasterisation of the mesh (5 mm pixels) on a few cars."""
    import pyvista as pv
    from scripts.data.check_cp_scale import frontal_area
    import json
    rows = []
    for sid in ids:
        m = pv.read(RAW / f"{sid}.vtk").triangulate()
        off = np.array(json.loads((ROOT / "data" / "cache" / sid / "meta.json").read_text())["offset_m"])
        pts = np.asarray(m.points) - off
        tri = pts[m.faces.reshape(-1, 4)[:, 1:]]
        # check_cp.frontal_area assumes the car is centred laterally at y = 0 (it adds 1.2 m); the cache frame is centred, so it applies as is
        rows.append((sid, frontal_area(tri), frontal_from_occupancy(sid)))
        print(f"  {sid}: raster {rows[-1][1]:.4f} m2, occupancy {rows[-1][2]:.4f} m2, diff {100 * (rows[-1][2] / rows[-1][1] - 1):+.2f}%", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--validate", type=int, default=8)
    a = ap.parse_args()
    ids = [r["sample_id"] for r in csv.DictReader(open(ROOT / "data" / "manifests" / "manifest.csv", encoding="utf-8")) if r["status"] == "ok"]
    if a.validate:
        print("validation (occupancy vs raster):")
        v = validate(ids[:: max(1, len(ids) // a.validate)][: a.validate])
        d = np.array([o / r - 1 for _, r, o in v])
        print(f"  mean diff {100 * d.mean():+.2f}%, max |diff| {100 * np.abs(d).max():.2f}%", flush=True)
    done = {}
    if OUT.exists():
        done = {r["sample_id"]: r for r in csv.DictReader(open(OUT, encoding="utf-8"))}
    todo = [i for i in ids if i not in done]
    print(f"{len(todo)} cars to process", flush=True)
    rows = list(done.values())
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, (sid, area, front, lam) in enumerate(pool.imap_unordered(one, todo), 1):
            rows.append({"sample_id": sid, "area_total_m2": f"{area:.5f}", "frontal_area_m2": f"{front:.5f}", "incl_rate_per_m2": f"{lam:.3f}"})
            if k % 50 == 0:
                print(f"[{k}/{len(todo)}]", flush=True)
    rows.sort(key=lambda r: r["sample_id"])
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "area_total_m2", "frontal_area_m2", "incl_rate_per_m2"])
        w.writeheader()
        w.writerows(rows)
    a_ = np.array([float(r["area_total_m2"]) for r in rows])
    f_ = np.array([float(r["frontal_area_m2"]) for r in rows])
    print(f"{len(rows)} cars; total area {a_.min():.1f}-{a_.max():.1f} m2 (mean {a_.mean():.1f}); frontal area {f_.min():.3f}-{f_.max():.3f} m2 (mean {f_.mean():.3f})")


if __name__ == "__main__":
    main()
