"""Physics-style diagnostics from the saved per-point test predictions (exploratory):
(1) Cp along the symmetry plane (y ~ 0): upper and lower surface profiles of the six qualitative cars (CFD vs M2 vs B2);
(2) pressure lift C_L,p = -(1/A_ref) sum Cp A n_z (total and rear half), same estimator and statistics as C_D,p;
(3) region errors of M2 on E2A for grids R1/R2/R3 (same seed 0 and cars): is the wheel error limited by voxel resolution?
Centerline profiles (1) show seed 0 of M2 and B2 only (no seed ensembling); tables (2) include M2 (3 seeds, per-seed errors averaged), B1, B2 and B3.
Usage: python scripts/diagnosis/physics_plots.py [--out reports/physics_plots.md]"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import accumulate, md as region_md  # noqa: E402
from scripts.diagnosis.pressure_coefficients import CLIP, FOLDS, GEO, MODELS, RES, boot, pool_weights, run_dirs  # noqa: E402
from scripts.diagnosis.qualitative_figures import pick_cars  # noqa: E402
from src.data.grid import load_cfg  # noqa: E402
from src.evaluation.regions import load_regions  # noqa: E402

L_REF = load_cfg()["frame"]["l_ref_m"]
CACHE = ROOT / "data" / "cache"
BAND, STEP = 0.03, 0.05  # metres: |y| < BAND counts as the symmetry plane; x bins of 5 cm


def preds(fold, model, seeds, suf=""):
    return [np.load(r / "test_pred_fp16.npy", mmap_mode="r") for r in run_dirs(fold, model, seeds, suf)]


def profile(x, cp, mask):
    edges = np.arange(0, 5.2, STEP)
    k = np.digitize(x[mask], edges)
    out = np.full(len(edges) - 1, np.nan)
    for b in range(1, len(edges)):
        v = cp[mask][k == b]
        if len(v) >= 3:
            out[b - 1] = v.mean()
    return 0.5 * (edges[:-1] + edges[1:]), out


def centerline(out_png):
    fig, ax = plt.subplots(3, 4, figsize=(17, 8.5), sharey=True)
    for r, fold in enumerate(FOLDS):
        ids = list(pd.read_csv(RES / f"{fold}_M2_R2_s0" / "test_per_car.csv").id)
        pm, pb = preds(fold, "M2_R2", (0,)), preds(fold, "B2_R1", (0,))
        for k, (kind, sid) in enumerate(pick_cars(RES, fold)[0].items()):
            j = ids.index(sid)
            p = np.load(CACHE / sid / "pool.npz")
            x, y, nrm = p["xyz"][:, 0] * L_REF, p["xyz"][:, 1] * L_REF, p["normal"]
            ref = np.clip(p["cp"].astype(np.float64), *CLIP)
            series = (("CFD", ref, "k"), ("M2 (seed 0)", np.asarray(pm[0][j], np.float64), "tab:red"), ("B2", np.asarray(pb[0][j], np.float64), "tab:blue"))
            for c, (surf, m) in enumerate((("upper", (nrm[:, 2] > 0.3)), ("lower", (nrm[:, 2] < -0.3)))):
                a = ax[r, 2 * k + c]
                for name, v, col in series:
                    xs, ys = profile(x, v, (np.abs(y) < BAND) & (np.abs(nrm[:, 1]) < 0.4) & m)
                    a.plot(xs, ys, color=col, lw=1.6 if name == "CFD" else 1.1, label=name)
                a.set_title(f"{sid} ({fold}, {kind}) {surf}", fontsize=8)
                a.grid(alpha=0.3)
                a.set_xlabel("x from nose (m)", fontsize=8)
    ax[0, 0].legend(fontsize=7)
    for r in range(3):
        ax[r, 0].set_ylabel("Cp (clipped), mean in 5 cm bins")
    fig.suptitle(f"Cp on the symmetry plane (|y| < {int(BAND * 100)} cm, near-horizontal surface), CFD vs predictions", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def cl(cp, nz, area, sid, rear):
    """C_L,p (x1) with the IPW ratio estimator, optionally only for the rear half (u > 0.5, u = x / car length)."""
    w = pool_weights(np.asarray(area, np.float64), sid)
    return -(GEO.at[sid, "area_total_m2"] / GEO.at[sid, "frontal_area_m2"]) * float((w * cp * nz * rear).sum() / w.sum())


def lift_table():
    rows = []
    for fold, held in FOLDS.items():
        ids = list(pd.read_csv(RES / f"{fold}_M2_R2_s0" / "test_per_car.csv").id)
        pools = [np.load(CACHE / i / "pool.npz") for i in ids]
        nz = [p["normal"][:, 2] for p in pools]
        ar = [p["area"] for p in pools]
        rear = []
        for p in pools:
            x = p["xyz"][:, 0]
            rear.append(((x - x.min()) / (x.max() - x.min()) > 0.5).astype(float))
        refs = [np.clip(p["cp"].astype(np.float64), *CLIP) for p in pools]
        for part, use_rear in (("total", False), ("rear half", True)):
            f = (lambda j, c: cl(c, nz[j], ar[j], ids[j], rear[j] if use_rear else np.ones_like(rear[j])))
            ref = np.array([1000 * f(j, refs[j]) for j in range(len(ids))])
            for model, rdir, seeds, suf in MODELS:
                pr = preds(fold, rdir, seeds, suf)
                if not pr:
                    continue
                est = np.array([[1000 * f(j, np.asarray(a[j], np.float64)) for j in range(len(ids))] for a in pr])
                ae = np.abs(est - ref).mean(0)
                lo, hi = boot(ae, np.mean)
                rows.append({"fold": fold, "part": part, "model": model, "ref mean [min, max] (counts)": f"{ref.mean():.1f} [{ref.min():.1f}, {ref.max():.1f}]", "ref sd": f"{ref.std(ddof=1):.1f}",
                             "MAE (counts) [CI 95%]": f"{ae.mean():.2f} [{lo:.2f}; {hi:.2f}]", "signed bias": f"{(est - ref).mean():+.2f}", "Spearman rho": f"{np.mean([spearmanr(e, ref)[0] for e in est]):.3f}"})
    return rows


def grid_region_table():
    cfg, names = load_regions()
    fold = "E2A"
    runs = {g: RES / f"{fold}_M2_{g}_s0" for g in ("R1", "R2", "R3")}
    ids = list(pd.read_csv(runs["R2"] / "test_per_car.csv").id)
    clip, out = tuple(load_cfg()["target"]["cp_clip"]), {}
    for g, d in runs.items():
        cnt, sae, sse, ref2, _ = accumulate(ids, [d], cfg, len(names), clip)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[g] = (np.nanmean(np.where(cnt > 0, sae / cnt, np.nan), 0), (sse / sse.sum(1, keepdims=True)).mean(0), sse.sum())
    rows = [{"region": nm, "MAE R1": f"{out['R1'][0][k]:.4f}", "MAE R2": f"{out['R2'][0][k]:.4f}", "MAE R3": f"{out['R3'][0][k]:.4f}",
             "R3 / R2": f"{out['R3'][0][k] / out['R2'][0][k]:.2f}", "R1 / R2": f"{out['R1'][0][k] / out['R2'][0][k]:.2f}", "error share % (R2)": f"{100 * out['R2'][1][k]:.1f}"} for k, nm in enumerate(names)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "physics_plots.md"))
    a = ap.parse_args()
    centerline(ROOT / "reports" / "centerline_cp.png")
    lift, grid = lift_table(), grid_region_table()
    mk = lambda rows: region_md(rows)
    text = ["# Physics-style diagnostics (exploratory; auto-generated by `scripts/diagnosis/physics_plots.py`)", "",
            "## 1. Cp along the symmetry plane", "", "![centerline](centerline_cp.png)", "",
            "Six qualitative cars (median and worst error per held-out style). Profiles are bin means of the pool points within 3 cm of y = 0 on near-horizontal surfaces; B1 and B3 are in the tables below, not in this figure.", "",
            "## 2. Pressure lift C_L,p (counts; positive = upward force), same estimator as C_D,p", "",
            "Rear half = points behind 50% of the car length. Pressure-only diagnostic; no total lift is available to compare with.", "", mk(lift), "",
            "## 3. Region MAE of M2 on E2A (held out Estateback) by grid resolution, seed 0, same 250 cars", "",
            "R3 / R2 below 1 means the finer grid lowers the region's error.", "", mk(grid), ""]
    Path(a.out).write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
