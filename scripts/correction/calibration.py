"""Shared calibration library for the decomposition and first corrections study (T1, T5, T7 stage B): per-region sufficient statistics of a set of predictions, post-hoc corrections that subtract a constant per region from the
predicted Cp (scalar_shift, region_const, region_const_intmatch), metrics and the 90% jackknife+ interval. Generalised from scripts/correction/first_calibration.py to any number of regions and seeds.
Stats dict (n cars, S seeds, R regions): W (n,R) IPW weight per region, Wt (n,), nr (n,R) point counts, Mz/Mx (n,R) = S_car * sum_r w n_{z|x} / sum w (coefficient units),
Rn (n,) sum ref^2, CL (n,) CFD C_L,p counts, Aw (S,n,R) sum_r w e, Au (S,n,R) sum_r e, Qu (S,n) sum e^2, dL/dD (S,n) coefficient errors in counts."""
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.pressure_coefficients import CLIP, GEO, pool_weights  # noqa: E402
from scripts.diagnosis.force_alignment import load_pool  # noqa: E402

ALPHA = 0.1


def build_stats(ids, get_pred, n_seeds, labeler, nreg):
    """get_pred(seed, sid) -> predicted Cp (100k,) in pool order. labeler(sid, xyz, nrm) -> int labels in 0..nreg-1."""
    n = len(ids)
    z = {k: np.zeros(sh) for k, sh in dict(W=(n, nreg), Wt=(n,), nr=(n, nreg), Mz=(n, nreg), Mx=(n, nreg), Rn=(n,), CL=(n,), Aw=(n_seeds, n, nreg), Au=(n_seeds, n, nreg), Qu=(n_seeds, n), dL=(n_seeds, n), dD=(n_seeds, n)).items()}
    for j, sid in enumerate(ids):
        xyz, nrm, cp, area = load_pool(sid)
        lab = labeler(sid, xyz, nrm).astype(np.int64)
        ref = np.clip(cp.astype(np.float64), *CLIP)
        w = pool_weights(np.asarray(area, np.float64), sid)
        sw = w.sum()
        S = float(GEO.at[sid, "area_total_m2"] / GEO.at[sid, "frontal_area_m2"])
        bc = lambda v: np.bincount(lab, weights=v, minlength=nreg + 1)[:nreg]
        z["W"][j], z["Wt"][j], z["nr"][j] = bc(w), sw, np.bincount(lab, minlength=nreg + 1)[:nreg]
        z["Mz"][j], z["Mx"][j] = S * bc(w * nrm[:, 2]) / sw, S * bc(w * nrm[:, 0]) / sw
        z["Rn"][j] = (ref ** 2).sum()
        z["CL"][j] = -1000 * S * (w * ref * nrm[:, 2]).sum() / sw
        for s in range(n_seeds):
            e = np.asarray(get_pred(s, sid), np.float64) - ref
            z["Aw"][s, j], z["Au"][s, j], z["Qu"][s, j] = bc(w * e), bc(e), (e ** 2).sum()
            z["dL"][s, j], z["dD"][s, j] = -1000 * S * (w * e * nrm[:, 2]).sum() / sw, -1000 * S * (w * e * nrm[:, 0]).sum() / sw
    return z


def take(z, idx):
    """Sub-dict of the stats for the cars idx (keeps the seed axis)."""
    return {k: (v[:, idx] if v.ndim >= 2 and k in ("Aw", "Au", "Qu", "dL", "dD") else v[idx]) for k, v in z.items()}


def fit_const(z, idx, lam, fit_z=None):
    """Region offsets c (S, R) minimising mean_i sum_r IPW-weighted mean squared error + lam * mean_i (dL^2 + dD^2) (coefficient units). The fit cars are `idx` of `fit_z` (default z)."""
    f = fit_z if fit_z is not None else z
    R = f["W"].shape[1]
    Wi = f["Wt"][idx][:, None]
    mz, mx = f["Mz"][idx], f["Mx"][idx]
    H = np.diag((f["W"][idx] / Wi).mean(0)) + lam * (mz.T @ mz + mx.T @ mx) / len(idx)
    rhs = (f["Aw"][:, idx] / Wi[None]).mean(1)
    if lam:
        rhs = rhs - lam * ((f["dL"][:, idx, None] / 1000) * mz[None] + (f["dD"][:, idx, None] / 1000) * mx[None]).mean(1)
    return np.linalg.solve(H + 1e-12 * np.eye(R), rhs.T).T


def make_fit(method, z, idx, fit_z=None):
    """Returns ('none'|'shift'|'off', payload). 'shift': (dL_mean, dD_mean) per seed; 'off': offsets c (S, R) broadcast over cars."""
    f = fit_z if fit_z is not None else z
    if method == "none":
        return ("none", None)
    if method == "scalar_shift":
        return ("shift", (f["dL"][:, idx].mean(1), f["dD"][:, idx].mean(1)))
    lam = {"region_const": 0.0, "intmatch_1": 1.0, "intmatch_10": 10.0}[method]
    return ("off", fit_const(z, idx, lam, fit_z))


def corrected(z, cars, fit):
    """Seed x car arrays after correction (dL', dD' in counts) and the field Rel L2 of each (seed, car); cars index the stats z."""
    kind, p = fit
    dL, dD = z["dL"][:, cars], z["dD"][:, cars]
    rel = np.sqrt(z["Qu"][:, cars]) / np.sqrt(z["Rn"][cars])[None]
    if kind == "none":
        return dL, dD, rel
    if kind == "shift":
        return dL - p[0][:, None], dD - p[1][:, None], rel
    offs = np.broadcast_to(p[:, None, :], (p.shape[0], len(cars), p.shape[1]))
    dL2 = dL + 1000 * np.einsum("nr,snr->sn", z["Mz"][cars], offs)
    dD2 = dD + 1000 * np.einsum("nr,snr->sn", z["Mx"][cars], offs)
    q = z["Qu"][:, cars] - 2 * np.einsum("snr,snr->sn", offs, z["Au"][:, cars]) + np.einsum("snr,nr->sn", offs ** 2, z["nr"][cars])
    return dL2, dD2, np.sqrt(np.maximum(q, 0)) / np.sqrt(z["Rn"][cars])[None]


def metrics(z, cars, fit):
    dL, dD, rel = corrected(z, cars, fit)
    eL, eD = dL.mean(0), dD.mean(0)
    return {"abs_bias_L": abs(eL.mean()), "abs_bias_D": abs(eD.mean()), "rmse_L": float(np.sqrt(((eL - eL.mean()) ** 2).mean())), "mae_L": float(np.abs(eL - eL.mean()).mean()),
            "rel_l2": float(rel.mean()), "spearman": float(spearmanr(z["CL"][cars] + eL, z["CL"][cars])[0]), "bias_L": float(eL.mean()), "eL": eL}


def jackknife_plus(z, idx, cars, method, fit_z=None):
    """90% jackknife+ for C_L,p, full refit leaving out each calibration car; idx are the calibration cars in fit_z (default z); `cars` the evaluation cars in z. Works on the seed-mean error."""
    f = fit_z if fit_z is not None else z
    k = len(idx)
    eps, Rr = np.zeros((k, len(cars))), np.zeros(k)
    for a in range(k):
        sub = np.delete(idx, a)
        fit = make_fit(method, z, sub, fit_z)
        eps[a] = corrected(z, cars, fit)[0].mean(0)
        Rr[a] = abs(corrected(f, idx[a:a + 1], fit)[0].mean(0)[0])
    lo_i, hi_i = max(1, math.floor(ALPHA * (k + 1))), min(k, math.ceil((1 - ALPHA) * (k + 1)))
    lo = np.sort(eps - Rr[:, None], axis=0)[lo_i - 1]
    hi = np.sort(eps + Rr[:, None], axis=0)[hi_i - 1]
    return float(np.mean((lo <= 0) & (hi >= 0))), float(np.mean((hi - lo) / 2)), math.floor(ALPHA * (k + 1)) >= 1
