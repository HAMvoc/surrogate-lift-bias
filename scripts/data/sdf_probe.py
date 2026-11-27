"""Probe SDF R1-R3 on a few cars: timing, sign, level 0 at surface points, and how far to trust the flood fill versus the exact signed SDF.
Usage: python scripts/data/sdf_probe.py data/raw E_S_WWC_WM_292 F_D_WM_WW_0335 ..."""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.grid import load_cfg, to_ndc  # noqa: E402
from src.data.sdf import load_car, make_scene, sdf_volumes  # noqa: E402

cfg = load_cfg()
raw, ids = Path(sys.argv[1]), sys.argv[2:]
box, grids, L, tau = cfg["box"], cfg["grids"], cfg["frame"]["l_ref_m"], cfg["sdf"]["tau"]
vox_m = {k: (box["x"][1] - box["x"][0]) * L / g[0] for k, g in grids.items()}


def level0_frac(vol, pts_lref, name):
    """Fraction of surface points whose interpolated |SDF| < half a voxel (metres)."""
    g = torch.tensor(to_ndc(pts_lref, box), dtype=torch.float32).view(1, -1, 1, 1, 3)
    v = torch.nn.functional.grid_sample(torch.tensor(vol)[None, None], g, padding_mode="border", align_corners=False)
    return float((v.abs().flatten().numpy() * tau * L < 0.5 * vox_m[name]).mean())


for sid in ids:
    verts, faces, off, wt = load_car(raw / cfg["source"]["stl_dir"] / f"{sid}.stl")
    scene = make_scene(verts, faces)
    vol, t, info = sdf_volumes(scene, grids, box, L, tau, wt)
    surf = (np.asarray(pv.read(raw / cfg["source"]["vtk_dir"] / f"{sid}.vtk").points) - off) / L
    surf = surf[np.random.default_rng(0).choice(len(surf), 100_000, replace=False)]
    print(f"\n{sid} faces={len(faces)} watertight={wt} inside_frac_R2={float((vol['R2'] < 0).mean()):.4f} "
          f"corner={vol['R2'][0, 0, 0]:+.2f} min={vol['R2'].min():+.2f} t(s)={ {k: round(v, 1) for k, v in t.items()} }")
    print("  level-0 |sdf|<0.5 voxel:", {k: round(level0_frac(vol[k], surf, k), 4) for k in vol}, info)
    if wt:  # trust in the flood fill: run it on a watertight car and compare signs with the exact signed SDF
        v2, _, i2 = sdf_volumes(scene, grids, box, L, tau, False, names=("R2",))
        a, b = vol["R2"] < 0, v2["R2"] < 0
        print(f"  floodfill vs exact sign: IoU={(a & b).sum() / (a | b).sum():.4f} inside_frac {a.mean():.4f} vs {b.mean():.4f}")
