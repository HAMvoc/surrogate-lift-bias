"""Plot surface regions on the pool of 3 cars per body style (spec 10.4: check visually before use). Writes reports/regions/<id>.png.
Usage: python scripts/diagnosis/plot_regions.py [id ...]"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.regions import assign_regions, load_regions  # noqa: E402

cfg, names = load_regions()
ids = sys.argv[1:] or [i for c in "FNE" for i in [j for j in json.loads((ROOT / "data/manifests/splits/E1.json").read_text())["test"] if j[0] == c][:3]]
PAL = ["#9e9e9e", "#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#000000", "#7f7f7f"]
out = ROOT / "reports" / "regions"
out.mkdir(parents=True, exist_ok=True)
for sid in ids:
    p = np.load(ROOT / "data" / "cache" / sid / "pool.npz")
    r = assign_regions(p["xyz"], p["normal"], cfg)  # L_ref coordinates, exactly as in the cache
    xyz, nrm = p["xyz"] * 5.0, p["normal"]  # only for plotting in metres
    sub = np.random.default_rng(0).choice(len(xyz), 60000, replace=False)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5), gridspec_kw={"width_ratios": [4.6, 4.6, 2.2]})
    # each view: drop the half far from the camera, draw points nearest the camera last
    views = ((0, 2, xyz[sub, 1] > 0, xyz[sub, 1]), (0, 1, xyz[sub, 2] > np.median(xyz[:, 2]), xyz[sub, 2]), (1, 2, xyz[sub, 0] < np.median(xyz[:, 0]), -xyz[sub, 0]))
    for a, (ix, iy, keep, depth), t in zip(ax, views, ("side (x-z, seen from +y)", "top (x-y, upper half)", "front (y-z, front half)")):
        o = np.argsort(depth[keep])
        q = sub[keep][o]
        a.scatter(xyz[q, ix], xyz[q, iy], c=[PAL[k % 12] for k in r[q]], s=1.5)
        a.set_aspect("equal"); a.set_title(f"{sid} — {t}", fontsize=9)
    h = [plt.Line2D([], [], marker="o", ls="", color=PAL[i % 12], label=f"{n} ({(r == i).mean() * 100:.1f}%)") for i, n in enumerate(names)]
    ax[2].legend(handles=h, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(out / f"{sid}.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(sid, {n: round(float((r == i).mean()), 3) for i, n in enumerate(names)})
