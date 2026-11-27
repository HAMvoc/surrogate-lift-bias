"""Quick audit of the probe-downloaded cars: pressure variables, bbox, watertight mesh, p range. Usage: python -I scripts/data/inspect_probe.py data/raw"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

raw = Path(sys.argv[1])
for v in sorted((raw / "pressure" / "PressureVTK").glob("*.vtk")):
    m = pv.read(v)
    s = trimesh.load(raw / "meshes" / "3DMeshesSTL" / f"{v.stem}.stl")
    b = np.array(m.bounds).reshape(3, 2)
    print(v.stem, "| pts", m.n_points, "cells", m.n_cells,
          "| point_arrays", {k: (m.point_data[k].shape, float(m.point_data[k].min()), float(m.point_data[k].max())) for k in m.point_data},
          "| cell_arrays", list(m.cell_data),
          "| bbox_m", np.round(b[:, 1] - b[:, 0], 3), "min", np.round(b[:, 0], 3),
          "| stl faces", len(s.faces), "watertight", s.is_watertight)
