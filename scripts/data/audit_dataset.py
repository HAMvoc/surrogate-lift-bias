"""Audit the downloaded cars (spec 4.6) -> reports/audit.md. Usage: python scripts/data/audit_dataset.py [data/raw]"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.grid import load_cfg  # noqa: E402

cfg = load_cfg()
raw = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "raw")
params = pd.read_csv((ROOT / cfg["source"]["params_csv"]).resolve()).set_index("Experiment")
cd = pd.read_csv(raw / cfg["source"]["cd_csv"]).set_index("ID")["Drag_Value"]
Q = 0.5 * cfg["cp"]["u_inf"] ** 2
L_REF = cfg["frame"]["l_ref_m"]

rows, seen = [], {}
for f in sorted((raw / cfg["source"]["vtk_dir"]).glob("*.vtk")):
    sid = f.stem
    m = pv.read(f)
    s = trimesh.load(raw / cfg["source"]["stl_dir"] / f"{sid}.stl", force="mesh")
    pts, p = np.asarray(m.points), np.asarray(m.point_data["p"])
    vb = np.array(m.bounds).reshape(3, 2)
    sb = s.bounds.T
    refl = pts * [1, -1, 1]
    dist, idx = cKDTree(pts).query(refl)
    h = hashlib.md5(np.round(pts[:2000], 4).tobytes()).hexdigest()
    seen.setdefault(h, []).append(sid)
    rows.append({
        "id": sid, "cat": sid[0],
        "finite": bool(np.isfinite(pts).all() and np.isfinite(p).all()),
        "n_pts": len(pts), "n_cells": m.n_cells, "stl_faces": len(s.faces),
        "len_m": vb[0, 1] - vb[0, 0], "wid_m": vb[1, 1] - vb[1, 0], "hei_m": vb[2, 1] - vb[2, 0],
        "len/L_ref": (vb[0, 1] - vb[0, 0]) / L_REF,
        "stl_vtk_bbox_diff_mm": 1000 * np.abs(sb - vb).max(),  # same coordinate system if ~0
        "z_min": vb[2, 0], "y_center": vb[1].mean(),
        "watertight": s.is_watertight,
        "sym_med_mm": 1000 * np.median(dist), "sym_p99_mm": 1000 * np.percentile(dist, 99),
        "sym_dp_med": np.median(np.abs(p - p[idx])) / Q,  # Cp difference between a point and its reflected point
        "cp_max": p.max() / Q, "cp_min": p.min() / Q,
        "has_params": sid in params.index, "has_cd": sid in cd.index,
    })
df = pd.DataFrame(rows)
dups = [v for v in seen.values() if len(v) > 1]

lines = ["# Audit (spec 4.6)", "", f"Source: `{raw}`; {len(df)} cars; L_ref = {L_REF} m; Cp = p / {Q:g}.", "",
         "## By body style", ""]
g = df.groupby("cat").agg(n=("id", "size"), len_min=("len_m", "min"), len_max=("len_m", "max"),
                          wid_max=("wid_m", "max"), hei_max=("hei_m", "max"),
                          cp_min=("cp_min", "min"), cp_max=("cp_max", "max"),
                          pts_min=("n_pts", "min"), pts_max=("n_pts", "max"), watertight=("watertight", "sum"))
lines += [g.round(3).to_markdown(), "", "## Per car", "", df.round(3).to_markdown(index=False), "", "## Check results", ""]
checks = {
    "1. No NaN/inf": df.finite.all(),
    "2. Units are metres (length 3–6 m)": df.len_m.between(3, 6).all(),
    "2. Coordinate origin: car bottom z=0, y symmetric about 0 (<5 mm)": bool((df.z_min.abs() < 5e-3).all() and (df.y_center.abs() < 5e-3).all()),
    "2. STL and VTK share a coordinate system (bbox offset <5 mm)": (df.stl_vtk_bbox_diff_mm < 5).all(),
    "3/4. Stagnation point Cp max in [0.99; 1.02]": df.cp_max.between(0.99, 1.02).all(),
    "5. STL mesh watertight (number of non-watertight cars)": f"{int((~df.watertight).sum())}/{len(df)} not watertight: " + ", ".join(df.id[~df.watertight]),
    "7. Symmetry about y=0: p99 distance of reflected points < 5 mm": df.sym_p99_mm.max() < 5,
    "7. Symmetry of p: median Cp difference < 0.01": df.sym_dp_med.max() < 0.01,
    "8. Duplicate cars (point hash)": "none" if not dups else str(dups),
    "In parameter table": f"{int(df.has_params.sum())}/{len(df)}; missing: " + ", ".join(df.id[~df.has_params]),
    "Has Cd label": f"{int(df.has_cd.sum())}/{len(df)}",
    f"Bounding box: max length / L_ref ≤ {cfg['box']['x'][1]}": f"max {df['len/L_ref'].max():.3f}",
}
lines += [f"- {k}: {'OK' if v is True else 'FAIL' if v is False else v}" for k, v in checks.items()]
(ROOT / "reports" / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[-len(checks) - 2:]))
