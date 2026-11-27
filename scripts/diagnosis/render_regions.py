"""Zero-shot checks study, task T5: render the T3 (Force-alignment study) regions on three cars (one per body style) from the side, the top and the bottom, for a visual check of the region rules.
Points of the 100k pool coloured by region (back-face culled by the stored normals so only the surface facing the camera is drawn). The definitions are not changed here.
Writes results/zero_shot_checks/T5/T5_regions_<car>.png. Usage: python scripts/diagnosis/render_regions.py"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.force_alignment import REGIONS, load_pool, region_labels  # noqa: E402

OUT = ROOT / "results" / "zero_shot_checks" / "T5"
L_REF = 5.0
CARS = ("E_S_WW_WM_143", "N_S_WW_WM_066", "F_S_WWS_WM_502")
COL = {"base": "#D55E00", "underbody": "#999999", "roof_rear": "#E69F00", "roof_front": "#F0E442", "front": "#009E73", "sides": "#56B4E9", "other": "#CC79A7"}
# name, camera position, view up, cull (axis, sign): keep points whose normal component along axis * sign >= -0.05
VIEWS = (("side (from -y)", (2.3, -12, 0.7), (0, 0, 1), (1, -1)), ("top", (2.3, 0, 12), (0, 1, 0), (2, 1)), ("bottom", (2.3, 0, -12), (0, -1, 0), (2, -1)))


def shot(xyz, rgb, cam, up):
    pl = pv.Plotter(off_screen=True, window_size=(1200, 520))
    pl.set_background("white")
    pl.add_points(pv.PolyData(xyz), scalars=rgb, rgb=True, point_size=3.2, render_points_as_spheres=False)
    pl.camera.parallel_projection = True
    pl.camera.position, pl.camera.focal_point, pl.camera.up = cam, (2.3, 0.0, 0.7), up
    pl.camera.parallel_scale = 1.35 if up == (0, 0, 1) else 1.0
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(ROOT / "results" / "force_alignment" / "per_car_terms.csv.gz")
    g = T[(T.fold == "E2A") & (T.model == "M2")].drop(columns=["seed", "role"]).groupby("id").mean(numeric_only=True)
    assert abs(g.dL.mean() - 183.9) <= 0.5 and (g[[f"reg_dL_{r}" for r in REGIONS]].sum(1) - g.dL).abs().max() < 1e-6, "gate failed"
    for sid in CARS:
        xyz, nrm, _, _ = load_pool(sid)
        lab = region_labels(xyz, nrm)
        pts = xyz.astype(np.float64) * L_REF
        rgb = np.array([[int(COL[REGIONS[k]][i:i + 2], 16) for i in (1, 3, 5)] for k in lab], dtype=np.uint8)
        fig, axs = plt.subplots(3, 1, figsize=(8, 8.2))
        for ax, (name, cam, up, (axis, sign)) in zip(axs, VIEWS):
            keep = nrm[:, axis] * sign >= -0.05
            ax.imshow(shot(pts[keep], rgb[keep], cam, up))
            ax.set_title(f"{sid}: {name}", fontsize=8)
            ax.axis("off")
        fig.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=COL[r], label=f"{r} ({100 * (lab == i).mean():.1f}% of points)") for i, r in enumerate(REGIONS)], loc="lower center", ncol=4, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(OUT / f"T5_regions_{sid}.png", dpi=130)
        plt.close(fig)
        print("rendered", sid, flush=True)


if __name__ == "__main__":
    main()
