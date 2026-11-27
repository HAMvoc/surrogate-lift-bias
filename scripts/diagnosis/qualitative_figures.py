"""Qualitative figures (spec 10.8): for each held-out body style (E2A/E2B/E2C) the car with median per-car error and the worst one.
Rows: reference Cp, M2 prediction, absolute error. Columns: four fixed orthographic views (side, top, front, rear), same camera for every car.
Prediction = mean over seeds of M2 R2 (seeds 0-2). Values live on the 100k-point pool; the full mesh is coloured by nearest pool point (all three rows
use the same interpolation, so the error map is not biased by it). Colour scales are fixed: Cp symmetric around 0 in [-1, 1] (clipped), |error| in [0, 0.5].
Usage: python scripts/diagnosis/qualitative_figures.py [--out reports/figures] [--ids ID ...]"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.grid import load_cfg  # noqa: E402

CFG = load_cfg()
FOLDS = {"E2A": "Estateback", "E2B": "Notchback", "E2C": "Fastback"}
CP_LIM, ERR_LIM = 1.0, 0.5
L_REF = CFG["frame"]["l_ref_m"]
CENTER = (2.3, 0.0, 0.7)  # fixed focal point (m) for every car
# name, camera position, view up, window (w, h) px, parallel scale (half viewport height in m)
VIEWS = (("side (from -y)", (2.3, -12, 0.7), (0, 0, 1), (900, 420), 1.40),
         ("top", (2.3, 0, 12), (0, 1, 0), (900, 420), 1.40),
         ("front", (-12, 0, 0.7), (0, 0, 1), (420, 420), 1.25),
         ("rear", (14, 0, 0.7), (0, 0, 1), (420, 420), 1.25))


def pick_cars(root, fold):
    """Median-error and worst-error car of a fold by seed-mean rel L2 (the ids and order follow test_per_car.csv)."""
    runs = [pd.read_csv(root / f"{fold}_M2_R2_s{s}" / "test_per_car.csv").set_index("id").rel_l2 for s in (0, 1, 2)]
    err = pd.concat(runs, axis=1).mean(1)
    return {"median": err.index[(err - err.median()).abs().argmin()], "worst": err.idxmax()}, err


def car_values(root, fold, sid):
    """Reference Cp (clipped), seed-mean M2 prediction at the 100k pool points, pool coordinates in metres."""
    ids = list(pd.read_csv(root / f"{fold}_M2_R2_s0" / "test_per_car.csv").id)
    j = ids.index(sid)
    pool = np.load(ROOT / "data" / "cache" / sid / "pool.npz")
    ref = np.clip(pool["cp"].astype(np.float64), *CFG["target"]["cp_clip"])
    preds = [np.load(root / f"{fold}_M2_R2_s{s}" / "test_pred_fp16.npy", mmap_mode="r")[j].astype(np.float64) for s in (0, 1, 2)]
    return ref, np.mean(preds, axis=0), pool["xyz"] * L_REF, preds


def load_mesh(sid, pool_xyz):
    """Original surface mesh in the cache frame (VTK points minus the nose/centre offset stored by preprocess) plus nearest-pool-point index."""
    off = np.array(json.loads((ROOT / "data" / "cache" / sid / "meta.json").read_text())["offset_m"])
    mesh = pv.read(ROOT / "data" / "raw" / CFG["source"]["vtk_dir"] / f"{sid}.vtk").triangulate()
    mesh.clear_data()
    mesh.points = np.asarray(mesh.points) - off
    return mesh, cKDTree(pool_xyz).query(mesh.points, workers=-1)[1]


def render(mesh, values, cmap, lim, lo=None):
    imgs = []
    mesh["v"] = values
    for _, pos, up, size, scale in VIEWS:
        pl = pv.Plotter(off_screen=True, window_size=size)
        pl.set_background("white")
        pl.add_mesh(mesh, scalars="v", cmap=cmap, clim=(-lim if lo is None else lo, lim), show_scalar_bar=False, ambient=0.55, diffuse=0.55, specular=0.0)
        pl.camera.parallel_projection = True
        pl.camera_position = [pos, CENTER, up]
        pl.camera.parallel_scale = scale
        imgs.append(pl.screenshot())
        pl.close()
    return imgs


def make_figure(root, fold, sid, kind, out):
    ref, pred, pxyz, per_seed = car_values(root, fold, sid)
    mesh, nn = load_mesh(sid, pxyz)
    rel = [np.sqrt(((p - ref) ** 2).sum() / (ref ** 2).sum()) for p in per_seed]
    csv0 = pd.read_csv(root / f"{fold}_M2_R2_s0" / "test_per_car.csv").set_index("id").rel_l2[sid]
    assert abs(rel[0] - csv0) < 2e-3, f"{sid}: prediction/id misalignment ({rel[0]:.4f} vs {csv0:.4f})"  # same guard as analyze_regions.py
    rows = (("Reference Cp (CFD)", ref[nn], "RdBu_r", CP_LIM, None), ("M2 prediction", pred[nn], "RdBu_r", CP_LIM, None),
            ("|error|", np.abs(pred - ref)[nn], "magma", ERR_LIM, 0.0))
    ratios = [v[3][0] for v in VIEWS]
    fig = plt.figure(figsize=(16, 7.6))
    gs = fig.add_gridspec(3, 4, width_ratios=ratios, wspace=0.02, hspace=0.04, left=0.07, right=0.985, top=0.9, bottom=0.07)
    for r, (title, vals, cmap, lim, lo) in enumerate(rows):
        for c, img in enumerate(render(mesh, vals, cmap, lim, lo)):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(title, fontsize=11)
            if r == 0:
                ax.set_title(VIEWS[c][0], fontsize=10)
    for x0, cmap, lim, lo, lab in ((0.07, "RdBu_r", CP_LIM, -CP_LIM, "Cp (clipped to [-1, 1] for display)"), (0.62, "magma", ERR_LIM, 0.0, "|error| (clipped at 0.5)")):
        cax = fig.add_axes([x0, 0.025, 0.3, 0.016])
        fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(lo, lim), cmap), cax=cax, orientation="horizontal").set_label(lab, fontsize=9)
    fig.suptitle(f"{sid} — {fold} (held out {FOLDS[fold]}), {kind} error car; per-car rel L2 {np.mean(rel):.3f} (seed mean)", fontsize=12)
    path = out / f"qual_{fold}_{kind}_{sid}.png"
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path, float(np.mean(rel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "figures"))
    ap.add_argument("--ids", nargs="*", help="render only these car ids (must be test cars of some fold)")
    a = ap.parse_args()
    root, out = ROOT / "results", Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in FOLDS:
        picks, err = pick_cars(root, fold)
        for kind, sid in picks.items():
            if a.ids and sid not in a.ids:
                continue
            path, rel = make_figure(root, fold, sid, kind, out)
            rows.append((fold, kind, sid, rel, path.name))
            print(fold, kind, sid, f"rel L2 {rel:.4f}", path.name, flush=True)
    (out / "qualitative_index.md").write_text(
        "# Qualitative figures (auto-generated by `scripts/diagnosis/qualitative_figures.py`)\n\n| fold | car | id | rel L2 (seed mean) | file |\n|---|---|---|---|---|\n"
        + "\n".join(f"| {f} | {k} | {s} | {r:.4f} | {p} |" for f, k, s, r, p in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
