"""Force-alignment decomposition of the integral pressure-coefficient error (Force-alignment study of the C1 investigation).

Per car, with pool points i, e_i = Cp_pred_i - Cp_ref_i (reference clipped to CLIP, prediction as saved), IPW weights w_i = area_i / (1 - exp(-lambda * area_i))
(the convention of scripts/diagnosis/pressure_coefficients.py that reproduces +183.9 counts), <a, b> = sum(w a b) / sum(w), ||a|| = sqrt(<a, a>), S = A_total / A_ref:

    Delta C_k = -S <e, n_k>  =  -S * rho_k * ||e|| * ||n_k||,   rho_k = <e, n_k> / (||e|| ||n_k||)      (k = z: lift, k = x: drag; counts = 1000 x coefficient)

Regions (rule order = priority, first match wins; x_rel = (x - x_min) / (x_max - x_min) of the car's own pool) partition the points, so the regional
contributions sum exactly to Delta C_k.

Library use: terms_table(fold, role, runs) -> long DataFrame, one row per (model, seed, car). Self-test (gate for the analysis): pytest tests/test_force_alignment.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.pressure_coefficients import CLIP, GEO, pool_weights  # noqa: E402

CACHE = ROOT / "data" / "cache"
REGIONS = ("base", "underbody", "roof_rear", "roof_front", "front", "sides", "other")  # priority order


def region_labels(xyz, nrm):
    """Integer label (index into REGIONS) per point. base: n_x > 0.7 and x_rel > 0.8; underbody: n_z < -0.3; roof_rear: n_z > 0.3, x_rel >= 0.6;
    roof_front: n_z > 0.3, 0.25 <= x_rel < 0.6; front: x_rel < 0.25 (what is left after the rules above, so not underbody); sides: |n_y| > 0.7; other."""
    x = np.asarray(xyz[:, 0], np.float64)
    xr = (x - x.min()) / (x.max() - x.min())
    nx, ny, nz = (np.asarray(nrm[:, k], np.float64) for k in range(3))
    conds = [(nx > 0.7) & (xr > 0.8), nz < -0.3, (nz > 0.3) & (xr >= 0.6), (nz > 0.3) & (xr >= 0.25) & (xr < 0.6), xr < 0.25, np.abs(ny) > 0.7]
    lab = np.full(len(x), len(conds), dtype=np.int64)
    for k in reversed(range(len(conds))):
        lab[conds[k]] = k
    return lab


def car_terms(pred, ref, nrm, area, sid, labels=None, clip=CLIP, ipw=True, regions=REGIONS):
    """All force-alignment terms of one car as a dict (counts for the Delta C columns). `labels` (region_labels) adds regional columns.
    clip / ipw are for sensitivity checks only (defaults = the convention that reproduces +183.9): ipw=False uses w = area."""
    ref = np.clip(np.asarray(ref, np.float64), *clip)
    pred = np.asarray(pred, np.float64)
    e = pred - ref
    w = pool_weights(np.asarray(area, np.float64), sid) if ipw else np.asarray(area, np.float64)
    sw = w.sum()
    S = float(GEO.at[sid, "area_total_m2"] / GEO.at[sid, "frontal_area_m2"])
    ee = (w * e * e).sum()
    out = {"S": S, "e_norm": float(np.sqrt(ee / sw))}
    nreg = len(regions)
    for k, ax in (("L", 2), ("D", 0)):
        n = np.asarray(nrm[:, ax], np.float64)
        wen, wnn = (w * e * n).sum(), (w * n * n).sum()
        out[f"d{k}"] = -1000 * S * float(wen / sw)
        out[f"rho_{k}"] = float(wen / np.sqrt(ee * wnn))
        out[f"n_{k}"] = float(np.sqrt(wnn / sw))
        if labels is not None:
            bc = lambda v: np.bincount(labels, weights=v, minlength=nreg + 1)[:nreg]
            a_, e_, n_ = bc(w * e * n), bc(w * e * e), bc(w * n * n)
            pf = bc(w * pred * n) if k == "L" else None  # regional integral of the predicted pressure force (uses no labels)
            for r, name in enumerate(regions):
                out[f"reg_d{k}_{name}"] = -1000 * S * float(a_[r] / sw)
                if k == "L":
                    out[f"reg_rhoL_{name}"] = float(a_[r] / np.sqrt(e_[r] * n_[r])) if e_[r] > 0 and n_[r] > 0 else np.nan
                    out[f"reg_predL_{name}"] = -1000 * S * float(pf[r] / sw)
    if labels is not None:
        wr, we = (np.bincount(labels, weights=v, minlength=nreg + 1)[:nreg] for v in (w, w * e))
        for r, name in enumerate(regions):
            out[f"reg_area_{name}"] = float(wr[r] / sw)
            out[f"reg_me_{name}"] = float(we[r] / wr[r]) if wr[r] > 0 else np.nan
    return out


def cfd_coefficients(ref, nrm, area, sid):
    """C_L,p and C_D,p (counts) of the CFD labels (clipped Cp) with the same IPW estimator."""
    ref = np.clip(np.asarray(ref, np.float64), *CLIP)
    w = pool_weights(np.asarray(area, np.float64), sid)
    S = float(GEO.at[sid, "area_total_m2"] / GEO.at[sid, "frontal_area_m2"])
    return {"CL": -1000 * S * float((w * ref * nrm[:, 2]).sum() / w.sum()), "CD": -1000 * S * float((w * ref * nrm[:, 0]).sum() / w.sum())}


def load_pool(sid):
    p = np.load(CACHE / sid / "pool.npz")
    return p["xyz"], p["normal"], p["cp"], p["area"]


def terms_table(fold, role, runs, regions=True, log=print, labeler=None, region_names=REGIONS):
    """runs: [(model, seed, run_dir)]. Reads <role>_pred_fp16.npy / <role>_per_car.csv of each run dir (role: test | val); runs without the file are reported and skipped.
    Returns a long DataFrame (fold, role, model, seed, id, terms...). All runs of a fold must cover the same cars."""
    found, ids_of = [], {}
    for m, s, d in runs:
        npy, csv = Path(d) / f"{role}_pred_fp16.npy", Path(d) / f"{role}_per_car.csv"
        if npy.exists() and csv.exists():
            found.append((m, s, np.load(npy, mmap_mode="r"), {i: n for n, i in enumerate(pd.read_csv(csv).id)}))
            ids_of[(m, s)] = set(found[-1][3])
        else:
            log(f"no file: {npy} (model {m}, seed {s}, fold {fold}, role {role})")
    if not found:
        return pd.DataFrame()
    first = next(iter(ids_of.values()))
    assert all(v == first for v in ids_of.values()), f"{fold}: runs cover different cars"
    rows = []
    for sid in sorted(first):
        xyz, nrm, cp, area = load_pool(sid)
        lab = (labeler(sid, xyz, nrm) if labeler else region_labels(xyz, nrm)) if regions else None
        for m, s, pr, pos in found:
            rows.append({"fold": fold, "role": role, "model": m, "seed": s, "id": sid, **car_terms(pr[pos[sid]], cp, nrm, area, sid, lab, regions=region_names)})
    return pd.DataFrame(rows)
