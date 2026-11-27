"""Two-stage correction study, task T4: try to fix the wheel region (rear wheel outer face, tyre top). Variant v3 = the v2 circle fit with a looser mask: rho <= R + dR and |y| in [y_in - din, y_out + dout].
Reports the wheel area share and renders the wheel neighbourhood (side view) of 3 cars for v2 and v3. Usage: python scripts/diagnosis/wheel_fit_variant.py"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis import regions as rv
from scripts.diagnosis.force_alignment import load_pool

OUT = ROOT / "results" / "two_stage" / "T4" / "wheelfit"
CARS = ["E_S_WW_WM_143", "N_S_WW_WM_066", "F_S_WWS_WM_502"]
P = dict(dR=0.05, din=0.05, dout=0.10)


def mask_v3(xyz, fits):
    p = xyz.astype(np.float64) * rv.L_M
    zmin = p[:, 2].min()
    m = np.zeros(len(p), bool)
    for f in fits:
        if np.isnan(f["y_in"]):
            continue
        rho = np.hypot(p[:, 0] - f["x_c"], p[:, 2] - (zmin + f["R"]))
        m |= (rho <= f["R"] + P["dR"]) & (np.sign(p[:, 1]) == f["side"]) & (np.abs(p[:, 1]) >= f["y_in"] - P["din"]) & (np.abs(p[:, 1]) <= f["y_out"] + P["dout"])
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(11, 4.6))
    for c, sid in enumerate(CARS):
        xyz, nrm, _, _ = load_pool(sid)
        m2, fits = rv.fit_wheels(xyz)
        m3 = mask_v3(xyz, fits)
        p = xyz * rv.L_M
        print(sid, "share v2 %.3f v3 %.3f" % (m2.mean(), m3.mean()), "extra points", int((m3 & ~m2).sum()))
        for r, (nm, m) in enumerate((("v2", m2), ("v3", m3))):
            s = p[:, 1] > 0
            axs[r, c].scatter(p[s, 0], p[s, 2], c=np.where(m[s], "tab:red", "lightgray"), s=0.3, rasterized=True)
            axs[r, c].set_aspect("equal"); axs[r, c].set_title(f"{sid} {nm} wheels {100 * m.mean():.1f}%", fontsize=7); axs[r, c].axis("off")
    fig.savefig(OUT / "wheelfit_v2_v3.png", dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main()
