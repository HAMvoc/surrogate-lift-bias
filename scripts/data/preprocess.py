"""Preprocessing (spec 5): each car -> data/cache/<id>/{sdf_R1,sdf_R2,sdf_R3,occ_R2}.npy, pool.npz, meta.json.
Deterministic and rerunnable: skips cars that already have meta.json with the same config hash. Only processes cars whose files are fully downloaded.
Usage: python scripts/data/preprocess.py [--workers 4] [--limit N] [--out data/cache] [id ...]"""
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.grid import load_cfg, to_ndc  # noqa: E402

CFG = load_cfg()
SRC = CFG["source"]
RAW = ROOT / "data" / "raw"
KEYS = ("frame", "box", "grids", "sdf", "pool", "cp")  # parameters affecting the output -> hash
CFG_HASH = hashlib.md5(json.dumps({k: CFG[k] for k in KEYS}, sort_keys=True).encode()).hexdigest()[:12]
SCRIPT_VERSION = 1  # bump when the output-writing logic changes


def vertex_normals_and_areas(m):
    """Outward vertex normals (VTK makes winding consistent) and vertex area = 1/3 of the total area of adjacent triangles."""
    m = m.triangulate()
    faces = m.faces.reshape(-1, 4)[:, 1:]
    mn = m.compute_normals(cell_normals=False, point_normals=True, consistent_normals=True,
                           auto_orient_normals=True, split_vertices=False)
    area = np.asarray(m.compute_cell_sizes(length=False, volume=False)["Area"])
    va = np.bincount(faces.ravel(), weights=np.repeat(area / 3, 3), minlength=m.n_points)
    return np.asarray(mn.point_data["Normals"], dtype=np.float32), va.astype(np.float32)


def process(args):
    sid, out_root = args
    import pyvista as pv  # import inside the worker (spawn)
    import torch
    from src.data.sdf import load_car, make_scene, sdf_volumes

    out = Path(out_root) / sid
    meta_path = out / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("cfg_hash") == CFG_HASH and old.get("version") == SCRIPT_VERSION and old.get("status") == "ok":
            return sid, "skip", old
    t0 = time.time()
    box, grids, L, tau = CFG["box"], CFG["grids"], CFG["frame"]["l_ref_m"], CFG["sdf"]["tau"]
    q = 0.5 * CFG["cp"]["u_inf"] ** 2
    verts, faces, off, wt = load_car(RAW / SRC["stl_dir"] / f"{sid}.stl")
    scene = make_scene(verts, faces)
    vol, t, info = sdf_volumes(scene, grids, box, L, tau, wt)

    m = pv.read(RAW / SRC["vtk_dir"] / f"{sid}.vtk")
    pts = (np.asarray(m.points, dtype=np.float64) - off)  # metres, same coordinate system as the translated STL
    p = np.asarray(m.point_data["p"], dtype=np.float64)
    nrm, va = vertex_normals_and_areas(m)
    n_pool = min(CFG["pool"]["n_points"], len(pts))
    seed = int(hashlib.md5(f"{CFG['pool']['seed']}-{sid}".encode()).hexdigest()[:8], 16)
    sel = np.random.default_rng(seed).choice(len(pts), n_pool, replace=False, p=va / va.sum())
    xyz = (pts[sel] / L).astype(np.float32)
    cp = ((p[sel] - CFG["cp"]["p_inf"]) / q).astype(np.float32)

    checks = {"cp_finite": bool(np.isfinite(cp).all()), "cp_max": float(cp.max()), "cp_min": float(cp.min()),
              "in_box": float(((to_ndc(xyz, box) >= -1) & (to_ndc(xyz, box) <= 1)).all(1).mean())}
    g = torch.tensor(to_ndc(xyz, box), dtype=torch.float32).view(1, -1, 1, 1, 3)
    vox = (box["x"][1] - box["x"][0]) * L / grids["R2"][0]
    v = torch.nn.functional.grid_sample(torch.tensor(vol["R2"])[None, None], g, padding_mode="border", align_corners=False)
    checks["level0_frac_R2"] = float((v.abs().flatten().numpy() * tau * L < 0.5 * vox).mean())  # E0 no. 3: >= 0.999
    if wt:  # E0 no. 4: normals point outward. Vertices +-5 mm along the normal must have positive/negative SDF sign.
        import open3d as o3d
        pn = pts[sel].astype(np.float32)
        nn = nrm[sel]
        sd = lambda x: scene.compute_signed_distance(o3d.core.Tensor(x.astype(np.float32))).numpy()
        checks["normal_out_frac"] = float(((sd(pn + 0.005 * nn) > 0) & (sd(pn - 0.005 * nn) < 0)).mean())
    # save
    out.mkdir(parents=True, exist_ok=True)
    for k in ("R1", "R2", "R3"):
        np.save(out / f"sdf_{k}.npy", vol[k].astype(np.float16))
    np.save(out / "occ_R2.npy", (vol["R2"] < 0).astype(np.uint8))
    np.savez(out / "pool.npz", xyz=xyz, normal=nrm[sel], area=va[sel], cp=cp)
    ok = checks["cp_finite"] and checks["in_box"] == 1.0 and checks["level0_frac_R2"] >= 0.999 \
        and checks.get("normal_out_frac", 1.0) >= 0.99
    meta = {"id": sid, "status": "ok" if ok else "check_failed", "cfg_hash": CFG_HASH, "version": SCRIPT_VERSION,
            "watertight": wt, "offset_m": off.tolist(), "faces": int(len(faces)), "n_pool": int(n_pool),
            "libs": {k: sys.modules[k].__version__ for k in ("numpy", "torch", "pyvista", "trimesh", "open3d")},
            "checks": checks, "sdf_info": info, "seconds": {**{k: round(x, 1) for k, x in t.items()}, "total": round(time.time() - t0, 1)}}
    meta_path.write_text(json.dumps(meta, indent=1))
    return sid, meta["status"], meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(ROOT / "data" / "cache"))
    a = ap.parse_args()

    import csv
    man = [r["sample_id"] for r in csv.DictReader(open(ROOT / "data" / "manifests" / "manifest.csv", encoding="utf-8")) if r["status"] == "ok"]
    want = a.ids or man
    have = [s for s in want if (RAW / SRC["vtk_dir"] / f"{s}.vtk").exists() and (RAW / SRC["stl_dir"] / f"{s}.stl").exists()]
    # files still downloading may be incomplete: skip cars with a file modified in the last 60 seconds
    have = [s for s in have if time.time() - max((RAW / SRC[d] / f"{s}.{e}").stat().st_mtime for d, e in (("vtk_dir", "vtk"), ("stl_dir", "stl"))) > 60]
    if a.limit:
        have = have[:a.limit]
    print(f"cfg_hash {CFG_HASH}; {len(have)}/{len(want)} cars ready; {a.workers} processes", flush=True)
    bad = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, (sid, st, meta) in enumerate(pool.imap_unordered(process, [(s, a.out) for s in have]), 1):
            if st == "check_failed":
                bad.append(sid)
            print(f"[{i}/{len(have)}] {sid} {st} {meta['seconds'].get('total', '-')}s {meta['checks'] if st != 'skip' else ''}", flush=True)
    print("check_failed:", bad or "none")
