"""SDF on a voxel grid from a surface mesh. Negative inside, positive outside, clipped at tau then divided by tau."""
import time

import numpy as np
import open3d as o3d
import trimesh
from scipy import ndimage

from src.data.grid import center_points


def load_car(stl_path):
    """Read STL and translate: car nose at x=0, y symmetric about 0, tyre bottom at z=0 (metres). Returns (verts, faces, offset)."""
    m = trimesh.load(stl_path, force="mesh")  # merges duplicate vertices so the watertight check works
    b = m.bounds
    off = np.array([b[0, 0], (b[0, 1] + b[1, 1]) / 2, b[0, 2]])
    return (m.vertices - off).astype(np.float32), m.faces.astype(np.uint32), off, m.is_watertight


def make_scene(verts, faces):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.core.Tensor(verts), o3d.core.Tensor(faces))
    return s


def _query(fn, pts_m, chunk=2_000_000):
    flat = pts_m.reshape(-1, 3).astype(np.float32)
    out = np.concatenate([fn(o3d.core.Tensor(flat[i:i + chunk])).numpy() for i in range(0, len(flat), chunk)])
    return out.reshape(pts_m.shape[:-1])


def inside_mask_floodfill(unsigned_d, voxel_m):
    """Non-watertight mesh: shell ~1 voxel thick, flood fill from outside; empty parts not connected to the outside are inside."""
    shell = unsigned_d < voxel_m * np.sqrt(3) / 2
    lab, _ = ndimage.label(~shell)  # 6-connectivity, the default
    return (lab != lab[0, 0, 0]) & ~shell  # lab[0,0,0] = the "outside" component; the box corner is not on the shell


def sdf_volumes(scene, grids, box, l_ref, tau, watertight, names=("R1", "R2", "R3")):
    """Returns ({name: normalised float32 SDF (D,H,W)}, {name: seconds}, info)."""
    out, t = {}, {}
    info = {}
    tau_m = tau * l_ref
    mask3 = None
    if not watertight:
        t0 = time.time()
        shape3 = tuple(grids["R3"])
        p3 = center_points(shape3, box) * l_ref
        d3 = _query(scene.compute_distance, p3)
        vox3 = (box["x"][1] - box["x"][0]) * l_ref / shape3[0]
        mask3 = inside_mask_floodfill(d3, vox3)
        info["inside_frac_R3"] = float(mask3.mean())
        t["floodfill_R3"] = time.time() - t0
    for name in names:
        t0 = time.time()
        shape = tuple(grids[name])
        pts = center_points(shape, box) * l_ref
        if watertight:
            d = _query(scene.compute_signed_distance, pts)
        else:
            d = _query(scene.compute_distance, pts)
            lo = np.array([box["x"][0], box["y"][0], box["z"][0]]) * l_ref
            span = np.array([box["x"][1] - box["x"][0], box["y"][1] - box["y"][0], box["z"][1] - box["z"][0]]) * l_ref
            idx = np.clip(((pts - lo) / span * np.array(grids["R3"])).astype(int), 0, np.array(grids["R3"]) - 1)
            inside = mask3[idx[..., 2], idx[..., 1], idx[..., 0]]  # mask3 is (Z, Y, X)
            d = np.where(inside, -d, d)
        out[name] = np.clip(d / tau_m, -1, 1).astype(np.float32)
        t[name] = time.time() - t0
    return out, t, info
