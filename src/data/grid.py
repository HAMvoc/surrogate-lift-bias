"""Anisotropic voxel grid and coordinate -> grid_sample mapping. Tensors are (N, C, D, H, W) with D=Z, H=Y, W=X."""
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_cfg(path=None):
    return yaml.safe_load(Path(path or ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))


def box_arrays(box):
    return np.array([box["x"][0], box["y"][0], box["z"][0]]), np.array([box["x"][1], box["y"][1], box["z"][1]])


def centers(shape_xyz, box):
    """Voxel centres (L_ref units) per axis; shape_xyz = (X, Y, Z)."""
    lo, hi = box_arrays(box)
    return [lo[i] + (np.arange(n) + 0.5) * (hi[i] - lo[i]) / n for i, n in enumerate(shape_xyz)]


def center_points(shape_xyz, box):
    """(D, H, W, 3) coordinates (x, y, z) of every voxel centre, in tensor order (Z, Y, X)."""
    xs, ys, zs = centers(shape_xyz, box)
    z, y, x = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.stack([x, y, z], axis=-1)


def to_ndc(points, box):
    """Coordinates (x, y, z) -> [-1, 1]; order (x, y, z) = (W, H, D) matches grid_sample with align_corners=False."""
    lo, hi = box_arrays(box)
    return 2 * (np.asarray(points) - lo) / (hi - lo) - 1
