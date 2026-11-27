"""Decomposition and first corrections study, task T1: render the v2 regions (with wheels) on the same three cars as the zero-shot checks study T5. Usage: python scripts/diagnosis/render_regions_with_wheels.py"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.force_alignment import load_pool  # noqa: E402
from scripts.diagnosis.render_regions import CARS, VIEWS, shot  # noqa: E402
from scripts.diagnosis.regions import REGIONS2, labeler  # noqa: E402

OUT = ROOT / "results" / "decomposition" / "T1"
COL = {"wheels": "#000000", "base": "#D55E00", "underbody": "#999999", "roof_rear": "#E69F00", "roof_front": "#F0E442", "front": "#009E73", "sides": "#56B4E9", "other": "#CC79A7"}


def main():
    for sid in CARS:
        xyz, nrm, _, _ = load_pool(sid)
        lab = labeler(sid, xyz, nrm)
        pts = xyz.astype(np.float64) * 5.0
        rgb = np.array([[int(COL[REGIONS2[k]][i:i + 2], 16) for i in (1, 3, 5)] for k in lab], dtype=np.uint8)
        fig, axs = plt.subplots(3, 1, figsize=(8, 8.2))
        for ax, (name, cam, up, (axis, sign)) in zip(axs, VIEWS):
            keep = nrm[:, axis] * sign >= -0.05
            ax.imshow(shot(pts[keep], rgb[keep], cam, up))
            ax.set_title(f"{sid}: {name} (regions v2)", fontsize=8)
            ax.axis("off")
        fig.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=COL[r], label=f"{r} ({100 * (lab == i).mean():.1f}% of points)") for i, r in enumerate(REGIONS2)], loc="lower center", ncol=4, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(OUT / f"T1_regions_v2_{sid}.png", dpi=130)
        plt.close(fig)
        print("rendered", sid, flush=True)


if __name__ == "__main__":
    main()
