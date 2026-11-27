"""Zero-shot checks study, task T3: C2 first round, calibration of the frozen M2 on E2A from k labelled Estateback cars (CPU only).
Methods: none, scalar_shift, region_const, region_const_intmatch (lambda_int 1, 10), region_ridge (theta chosen by T2). Evaluation on the other 250 - k cars; metrics are computed on the
3-seed mean per-car error (Rel L2: mean over seeds). Corrected coefficients follow from per-region sufficient statistics (linear in the region offsets); a check against explicit
corrected fields (force_alignment.car_terms) must agree to 1e-6 counts.
Usage: python scripts/correction/first_calibration.py [--recompute-stats]"""
import argparse
import math
import re
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
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.pressure_coefficients import CLIP, GEO, pool_weights  # noqa: E402
from scripts.diagnosis.force_alignment import REGIONS, car_terms, load_pool, region_labels  # noqa: E402

OUT = ROOT / "results" / "zero_shot_checks" / "T3"
RES = ROOT / "results"
NR, S_ = len(REGIONS), 3
KS = (1, 2, 3, 5, 10, 25)
DRAWS = 50
ALPHA = 0.1
LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
METHODS = ["none", "scalar_shift", "region_const", "intmatch_1", "intmatch_10", "region_ridge"]
THR = {"abs_bias_L": 28.0, "rel_l2": 0.175, "rmse_L": 50.0, "cov": (0.87, 0.93), "half": 140.0}


# ------------------------------------------------------------------ sufficient statistics
def suff_stats(recompute):
    f = OUT / "suffstats.npz"
    if f.exists() and not recompute:
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    ids = list(pd.read_csv(RES / "E2A_M2_R2_s0" / "test_per_car.csv").id)
    preds = [np.load(RES / f"E2A_M2_R2_s{s}" / "test_pred_fp16.npy", mmap_mode="r") for s in range(S_)]
    pos = [{i: n for n, i in enumerate(pd.read_csv(RES / f"E2A_M2_R2_s{s}" / "test_per_car.csv").id)} for s in range(S_)]
    n = len(ids)
    z = {k: np.zeros(sh) for k, sh in dict(W=(n, NR), Wt=(n,), nr=(n, NR), Mz=(n, NR), Mx=(n, NR), Rn=(n,), CL=(n,), CD=(n,), Aw=(S_, n, NR), Au=(S_, n, NR), Qu=(S_, n), dL=(S_, n), dD=(S_, n)).items()}
    for j, sid in enumerate(ids):
        xyz, nrm, cp, area = load_pool(sid)
        lab = region_labels(xyz, nrm)
        ref = np.clip(cp.astype(np.float64), *CLIP)
        w = pool_weights(np.asarray(area, np.float64), sid)
        sw = w.sum()
        S = float(GEO.at[sid, "area_total_m2"] / GEO.at[sid, "frontal_area_m2"])
        bc = lambda v: np.bincount(lab, weights=v, minlength=NR + 1)[:NR]
        z["W"][j], z["Wt"][j], z["nr"][j] = bc(w), sw, np.bincount(lab, minlength=NR + 1)[:NR]
        z["Mz"][j], z["Mx"][j] = S * bc(w * nrm[:, 2]) / sw, S * bc(w * nrm[:, 0]) / sw
        z["Rn"][j] = (ref ** 2).sum()
        z["CL"][j], z["CD"][j] = -1000 * S * (w * ref * nrm[:, 2]).sum() / sw, -1000 * S * (w * ref * nrm[:, 0]).sum() / sw
        for s in range(S_):
            e = np.asarray(preds[s][pos[s][sid]], np.float64) - ref
            z["Aw"][s, j], z["Au"][s, j], z["Qu"][s, j] = bc(w * e), bc(e), (e ** 2).sum()
            z["dL"][s, j], z["dD"][s, j] = -1000 * S * (w * e * nrm[:, 2]).sum() / sw, -1000 * S * (w * e * nrm[:, 0]).sum() / sw
    z["ids"] = np.array(ids)
    np.savez_compressed(f, **z)
    return z


# ------------------------------------------------------------------ fits
def fit_const(z, idx, lam):
    """Region offsets c (S, 7): minimise mean_i sum_r IPW-weighted mean squared error + lam * mean_i (dL^2 + dD^2) (coefficient units, counts/1000)."""
    Wi = z["Wt"][idx][:, None]
    H = np.diag((z["W"][idx] / Wi).mean(0))
    mz, mx = z["Mz"][idx], z["Mx"][idx]
    H = H + lam * (mz.T @ mz + mx.T @ mx) / len(idx)
    rhs = (z["Aw"][:, idx] / Wi[None]).mean(1)  # (S, 7)
    if lam:
        rhs = rhs - lam * ((z["dL"][:, idx, None] / 1000) * mz[None] + (z["dD"][:, idx, None] / 1000) * mx[None]).mean(1)
    return np.linalg.solve(H + 1e-12 * np.eye(NR), rhs.T).T


def ridge_fit(Zc, Yc, lam):
    """Ridge with unpenalised intercept; Yc (k, m). Returns a predictor Z -> (n, m)."""
    zm, ym = Zc.mean(0), Yc.mean(0)
    A = (Zc - zm).T @ (Zc - zm) + lam * np.eye(Zc.shape[1])
    beta = np.linalg.solve(A, (Zc - zm).T @ (Yc - ym))
    return lambda Z: ym + (Z - zm) @ beta


def fit_ridge(z, Z, idx):
    """Per-region ridge of the weighted mean error on theta (all seeds at once); lambda by LOO inside the k cars (k >= 3). Returns offsets for arbitrary cars: f(car_idx) -> (S, n, 7)."""
    Y = (z["Aw"][:, idx] / np.maximum(z["W"][idx][None], 1e-12)).transpose(1, 0, 2).reshape(len(idx), -1)  # (k, S*7)
    Zc = Z[idx]
    if len(idx) < 3:
        c = fit_const(z, idx, 0.0)
        return lambda cars: np.broadcast_to(c[:, None, :], (S_, len(cars), NR))
    best, bl = np.inf, LAMBDAS[0]
    for lam in LAMBDAS:
        err = 0.0
        for i in range(len(idx)):
            keep = np.arange(len(idx)) != i
            err += ((ridge_fit(Zc[keep], Y[keep], lam)(Zc[i:i + 1]) - Y[i]) ** 2).sum()
        if err < best:
            best, bl = err, lam
    pred = ridge_fit(Zc, Y, bl)
    return lambda cars: pred(Z[cars]).reshape(len(cars), S_, NR).transpose(1, 0, 2)


def make_fitter(method, z, Z):
    """fit(idx) -> offsets(cars) giving (S, n, 7) region offsets to subtract; scalar_shift and none return None-like handled in evaluate."""
    if method == "none":
        return lambda idx: ("none", None)
    if method == "scalar_shift":
        return lambda idx: ("shift", (z["dL"][:, idx].mean(1), z["dD"][:, idx].mean(1)))
    if method == "region_const":
        return lambda idx: ("off", (lambda c: (lambda cars: np.broadcast_to(c[:, None, :], (S_, len(cars), NR))))(fit_const(z, idx, 0.0)))
    if method.startswith("intmatch"):
        lam = float(method.split("_")[1])
        return lambda idx: ("off", (lambda c: (lambda cars: np.broadcast_to(c[:, None, :], (S_, len(cars), NR))))(fit_const(z, idx, lam)))
    if method == "region_ridge":
        return lambda idx: ("off", fit_ridge(z, Z, idx))
    raise ValueError(method)


def corrected(z, cars, fit):
    """Seed x car arrays after correction: dL', dD' (counts) and rel L2."""
    kind, p = fit
    dL, dD = z["dL"][:, cars], z["dD"][:, cars]
    rel = np.sqrt(z["Qu"][:, cars]) / np.sqrt(z["Rn"][cars])[None]
    if kind == "none":
        return dL, dD, rel
    if kind == "shift":
        return dL - p[0][:, None], dD - p[1][:, None], rel
    offs = p(cars)
    dL2 = dL + 1000 * np.einsum("nr,snr->sn", z["Mz"][cars], offs)
    dD2 = dD + 1000 * np.einsum("nr,snr->sn", z["Mx"][cars], offs)
    q = z["Qu"][:, cars] - 2 * np.einsum("snr,snr->sn", offs, z["Au"][:, cars]) + np.einsum("snr,nr->sn", offs ** 2, z["nr"][cars])
    return dL2, dD2, np.sqrt(np.maximum(q, 0)) / np.sqrt(z["Rn"][cars])[None]


def metrics(z, cars, fit):
    dL, dD, rel = corrected(z, cars, fit)
    eL, eD = dL.mean(0), dD.mean(0)
    return {"abs_bias_L": abs(eL.mean()), "abs_bias_D": abs(eD.mean()), "rmse_L": float(np.sqrt(((eL - eL.mean()) ** 2).mean())), "rel_l2": float(rel.mean()),
            "spearman": float(spearmanr(z["CL"][cars] + eL, z["CL"][cars])[0]), "eL": eL}


def jackknife_plus(z, idx, cars, fitter):
    """90% jackknife+ for C_L,p with full refits leaving out each calibration car; works on the seed-mean corrected error."""
    k = len(idx)
    eps, R = np.zeros((k, len(cars))), np.zeros(k)
    for a in range(k):
        sub = np.delete(idx, a)
        f = fitter(sub)
        eps[a] = corrected(z, cars, f)[0].mean(0)
        R[a] = abs(corrected(z, idx[a:a + 1], f)[0].mean(0)[0])
    lo_i, hi_i = max(1, math.floor(ALPHA * (k + 1))), min(k, math.ceil((1 - ALPHA) * (k + 1)))
    lo = np.sort(eps - R[:, None], axis=0)[lo_i - 1]
    hi = np.sort(eps + R[:, None], axis=0)[hi_i - 1]
    return float(np.mean((lo <= 0) & (hi >= 0))), float(np.mean((hi - lo) / 2)), math.floor(ALPHA * (k + 1)) >= 1


def kmedoids(Z, k):
    """Deterministic PAM (greedy BUILD, then SWAP) on standardised theta."""
    D = np.linalg.norm(Z[:, None] - Z[None], axis=2)
    med = [int(np.argmin(D.sum(1)))]
    while len(med) < k:
        cost = [np.minimum(D[:, med].min(1), D[:, c]).sum() if c not in med else np.inf for c in range(len(Z))]
        med.append(int(np.argmin(cost)))
    cur = D[:, med].min(1).sum()
    for _ in range(20):
        improved = False
        for mi in range(k):
            for c in range(len(Z)):
                if c in med:
                    continue
                trial = med.copy()
                trial[mi] = c
                t = D[:, trial].min(1).sum()
                if t < cur - 1e-12:
                    med, cur, improved = trial, t, True
        if not improved:
            break
    return np.array(sorted(med))


def consistency_check(z):
    """Explicit corrected fields vs the sufficient-statistics formulas on 5 cars (seed 0): max |difference| of Delta C (counts) and of rel L2."""
    ids = list(z["ids"])
    rng = np.random.default_rng(5)
    offs = rng.normal(0, 0.03, (5, NR))
    preds = np.load(RES / "E2A_M2_R2_s0" / "test_pred_fp16.npy", mmap_mode="r")
    pos = {i: n for n, i in enumerate(pd.read_csv(RES / "E2A_M2_R2_s0" / "test_per_car.csv").id)}
    cars = np.arange(5)
    kind, p = "off", (lambda c: np.broadcast_to(offs[None], (S_, len(c), NR)))
    dL, dD, rel = corrected(z, cars, (kind, p))
    mx, mr = 0.0, 0.0
    for a, c in enumerate(cars):
        sid = ids[c]
        xyz, nrm, cp, area = load_pool(sid)
        lab = region_labels(xyz, nrm)
        p0 = np.asarray(preds[pos[sid]], np.float64) - offs[a][np.minimum(lab, NR - 1)] * (lab < NR)
        t = car_terms(p0, cp, nrm, area, sid)
        ref = np.clip(cp.astype(np.float64), *CLIP)
        mx = max(mx, abs(t["dL"] - dL[0, a]), abs(t["dD"] - dD[0, a]))
        mr = max(mr, abs(np.linalg.norm(p0 - ref) / np.linalg.norm(ref) - rel[0, a]))
    return mx, mr


# ------------------------------------------------------------------ main
def reference_rows():
    """Existing retraining (reports/calibration.md) and post-hoc conformal (reports/conformal_calibration.md) numbers at k = 10 and 25 for E2A."""
    out = []
    txt = (ROOT / "reports" / "calibration.md").read_text(encoding="utf-8")
    for s in range(3):
        sec = re.search(rf"## E2A, seed {s}\n(.*?)(?=\n## |\Z)", txt, re.S).group(1)
        for ln in sec.splitlines():
            c = [x.strip() for x in ln.strip("|").split("|")]
            if ln.startswith("|") and c[0] in ("k=10", "k=25"):
                out.append({"source": f"retrain (reports/calibration.md), seed {s}", "k": int(c[0][2:]), "rel L2 (225 test cars)": c[2], "C_L,p bias": c[3].split(" [")[0], "C_L,p MAE": c[4]})
    cc = (ROOT / "reports" / "conformal_calibration.md").read_text(encoding="utf-8")
    for ln in cc.splitlines():
        c = [x.strip() for x in ln.strip("|").split("|")]
        if ln.startswith("| E2A | M2 s") and c[2] == "C_L,p" and c[3] in ("10", "25"):
            out.append({"source": f"post-hoc conformal (reports/conformal_calibration.md), {c[1]}", "k": int(c[3]), "coverage": c[7], "half width": f"{float(c[9]) / 2:.0f}", "MAE after correction": c[5]})
    return [r for r in out if r["source"].startswith("retrain")], [r for r in out if r["source"].startswith("post-hoc")]


def main(recompute):
    OUT.mkdir(parents=True, exist_ok=True)
    z = suff_stats(recompute)
    gate = float(z["dL"].mean(0).mean())
    assert abs(gate - 183.9) <= 0.5, f"gate failed {gate}"
    mx, mr = consistency_check(z)
    assert mx < 1e-6 and mr < 1e-9, f"field/coefficient inconsistency {mx} {mr}"
    theta_name = (ROOT / "results" / "zero_shot_checks" / "T2" / "T2_chosen_theta.txt").read_text(encoding="utf-8")
    D = pd.read_csv(ROOT / "results" / "zero_shot_checks" / "T2" / "T2_descriptors.csv", index_col=0).loc[list(z["ids"])]
    cols = {"theta_rear (4)": list(D.columns[:4]), "theta_under (6)": list(D.columns[4:10]), "theta_rear + theta_under (10)": list(D.columns[:10])}[theta_name]
    Z = ((D[cols] - D[cols].mean()) / D[cols].std()).values
    n = len(z["ids"])
    rows, curves = [], []
    for k in KS:
        meds = kmedoids(Z, k)
        rng = np.random.default_rng(2026 + k)
        draws = [("random", t, np.sort(rng.permutation(n)[:k])) for t in range(DRAWS)] + [("kmedoids", 0, meds)]
        for method in METHODS:
            fitter = make_fitter(method, z, Z)
            for sampler, t, idx in draws:
                cars = np.setdiff1d(np.arange(n), idx)
                m = metrics(z, cars, fitter(idx))
                row = {"method": method, "sampler": sampler, "k": k, "draw": t, **{kk: v for kk, v in m.items() if kk != "eL"}, "cover": np.nan, "half": np.nan, "guarantee": np.nan}
                if k >= 5 and method != "none":
                    row["cover"], row["half"], row["guarantee"] = jackknife_plus(z, idx, cars, fitter)
                rows.append(row)
        print("k", k, "done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "T3_draws.csv.gz", index=False, compression="gzip")
    g = R.groupby(["method", "sampler", "k"])
    S = g.agg(**{f"{c}_mean": (c, "mean") for c in ("abs_bias_L", "abs_bias_D", "rmse_L", "rel_l2", "spearman", "cover", "half")},
              **{f"{c}_p5": (c, lambda x: np.nanpercentile(x, 5) if x.notna().any() else np.nan) for c in ("abs_bias_L", "abs_bias_D", "rmse_L", "rel_l2", "spearman", "cover", "half")},
              **{f"{c}_p95": (c, lambda x: np.nanpercentile(x, 95) if x.notna().any() else np.nan) for c in ("abs_bias_L", "abs_bias_D", "rmse_L", "rel_l2", "spearman", "cover", "half")},
              n_draws=("draw", "count"), guarantee=("guarantee", "mean")).reset_index()
    S.round(4).to_csv(OUT / "T3_c2_results.csv", index=False)
    plot(S)
    summary(S, R, gate, mx, mr, theta_name, cols)


def plot(S):
    col = {"none": "#7f7f7f", "scalar_shift": "#CC79A7", "region_const": "#0072B2", "intmatch_1": "#56B4E9", "intmatch_10": "#009E73", "region_ridge": "#D55E00"}
    mk = {"none": "x", "scalar_shift": "s", "region_const": "o", "intmatch_1": "^", "intmatch_10": "v", "region_ridge": "D"}
    panels = [("abs_bias_L", "|bias| C_L,p (counts)"), ("rmse_L", "RMSE C_L,p after bias removal (counts)"), ("rel_l2", "Rel L2 (field)"), ("abs_bias_D", "|bias| C_D,p (counts)"),
              ("spearman", "Spearman (corrected C_L,p vs CFD)"), ("cover", "jackknife+ 90% coverage"), ("half", "jackknife+ half-width (counts)")]
    fig, axs = plt.subplots(2, 4, figsize=(14, 6.4))
    for ax, (c, lab) in zip(axs.ravel(), panels):
        for m in METHODS:
            r = S[(S.method == m) & (S.sampler == "random")].sort_values("k")
            ax.plot(r.k, r[f"{c}_mean"], marker=mk[m], color=col[m], ms=4, label=m)
            ax.fill_between(r.k, r[f"{c}_p5"], r[f"{c}_p95"], color=col[m], alpha=0.12, lw=0)
            km = S[(S.method == m) & (S.sampler == "kmedoids")].sort_values("k")
            ax.scatter(km.k, km[f"{c}_mean"], color=col[m], marker=mk[m], s=26, facecolors="none", linewidths=0.9)
        ax.set_xscale("log")
        ax.set_xticks(KS)
        ax.set_xticklabels(KS)
        ax.set_xlabel("k labelled cars")
        ax.set_title(lab, fontsize=8)
        if c in THR:
            ax.axhline(THR[c] if not isinstance(THR[c], tuple) else THR[c][0], color="k", lw=0.6, ls="--")
    axs[0, 0].legend(fontsize=6, frameon=False)
    axs[1, 3].axis("off")
    axs[1, 3].text(0, 0.9, "lines: mean over 50 random draws\nband: 5-95% over draws\nopen markers: k-medoids on theta (single draw)\ndashed: registered threshold at k = 10", fontsize=7, va="top")
    fig.tight_layout()
    fig.savefig(OUT / "T3_curves.png", dpi=200)
    plt.close(fig)


def summary(S, R, gate, mx, mr, theta_name, cols):
    fmt = lambda r, c, d=1: f"{r[f'{c}_mean']:.{d}f} [{r[f'{c}_p5']:.{d}f}; {r[f'{c}_p95']:.{d}f}]"
    lines = ["# T3: calibration of the frozen M2 on E2A (CPU)", "",
             f"Gate: 3-seed mean bias C_L,p of the data used = {gate:+.1f} counts (required 183.9 +- 0.5): PASS. Field-coefficient consistency (explicit corrected fields vs sufficient-statistics formulas, 5 cars): max |Delta C| difference {mx:.1e} counts, max Rel L2 difference {mr:.1e}: PASS (required < 1e-6 and < 1e-9).",
             f"theta (registered choice from T2): **{theta_name}**, features {', '.join(cols)}. {DRAWS} random draws per k plus one k-medoids draw; each draw evaluated on the other 250 - k cars; metrics on the 3-seed mean error per car; table entries are mean [5%; 95%] over draws.", "",
             "k-medoids selection is deterministic and breaks the exchangeability assumption of the jackknife+ interval: its coverage is not guaranteed. jackknife+ for k < 9 uses clamped quantiles (no guarantee).", ""]
    for k in (5, 10, 25):
        rows = []
        for m in METHODS:
            r = S[(S.method == m) & (S.sampler == "random") & (S.k == k)].iloc[0]
            rows.append({"method": m, "|bias| C_L,p": fmt(r, "abs_bias_L"), "RMSE C_L,p after bias removal": fmt(r, "rmse_L"), "Rel L2": fmt(r, "rel_l2", 4), "|bias| C_D,p": fmt(r, "abs_bias_D"),
                         "Spearman": fmt(r, "spearman", 2), "jackknife+ coverage": (fmt(r, "cover", 3) if m != "none" else "-"), "half-width": (fmt(r, "half", 0) if m != "none" else "-")})
        lines += [f"## k = {k}, random sampler", "", md(rows), ""]
        rows = []
        for m in METHODS:
            r = S[(S.method == m) & (S.sampler == "kmedoids") & (S.k == k)].iloc[0]
            rows.append({"method": m, "|bias| C_L,p": f"{r.abs_bias_L_mean:.1f}", "RMSE after bias removal": f"{r.rmse_L_mean:.1f}", "Rel L2": f"{r.rel_l2_mean:.4f}", "|bias| C_D,p": f"{r.abs_bias_D_mean:.1f}", "Spearman": f"{r.spearman_mean:.2f}",
                         "jackknife+ coverage": (f"{r.cover_mean:.3f}" if m != "none" else "-"), "half-width": (f"{r.half_mean:.0f}" if m != "none" else "-")})
        lines += [f"### k = {k}, k-medoids on theta (single draw)", "", md(rows), ""]
    # thresholds at k = 10
    rows = []
    for m in METHODS:
        r = S[(S.method == m) & (S.sampler == "random") & (S.k == 10)].iloc[0]
        ok = {"|bias| <= 28": r.abs_bias_L_mean <= THR["abs_bias_L"], "Rel L2 <= 0.175": r.rel_l2_mean <= THR["rel_l2"], "RMSE <= 50": r.rmse_L_mean <= THR["rmse_L"]}
        if m != "none":
            ok["coverage 87-93%"] = THR["cov"][0] <= r.cover_mean <= THR["cov"][1]
            ok["half-width < 140"] = r.half_mean < THR["half"]
        rows.append({"method": m, **{c: f"{'pass' if v else 'FAIL'}" for c, v in ok.items()}, **({} if m != "none" else {"coverage 87-93%": "-", "half-width < 140": "-"}), "values": f"{r.abs_bias_L_mean:.1f} / {r.rel_l2_mean:.4f} / {r.rmse_L_mean:.1f} / " + (f"{r.cover_mean:.3f} / {r.half_mean:.0f}" if m != "none" else "- / -")})
    lines += ["## Registered thresholds at k = 10, random sampler (mean over draws; values = |bias| / Rel L2 / RMSE / coverage / half-width)", "", md(rows), ""]
    # prediction
    pr = []
    for k in (10, 25):
        sh = S[(S.method == "scalar_shift") & (S.sampler == "random") & (S.k == k)].iloc[0].rmse_L_mean
        row = {"k": k, "RMSE scalar_shift": f"{sh:.1f}"}
        for m in ("region_const", "region_ridge"):
            v = S[(S.method == m) & (S.sampler == "random") & (S.k == k)].iloc[0]
            row[f"RMSE {m} (reduction vs scalar_shift)"] = f"{v.rmse_L_mean:.1f} ({100 * (1 - v.rmse_L_mean / sh):+.1f}%)"
            row[f"|bias| {m}"] = f"{v.abs_bias_L_mean:.1f}"
            row[f"Rel L2 {m}"] = f"{v.rel_l2_mean:.4f}"
        pr.append(row)
    base = S[(S.method == "none") & (S.sampler == "random") & (S.k == 10)].iloc[0]
    rc = {k: S[(S.method == "region_const") & (S.sampler == "random") & (S.k == k)].iloc[0] for k in (10, 25)}
    rr = {k: S[(S.method == "region_ridge") & (S.sampler == "random") & (S.k == k)].iloc[0] for k in (10, 25)}
    ss = {k: S[(S.method == "scalar_shift") & (S.sampler == "random") & (S.k == k)].iloc[0] for k in (10, 25)}
    fixes = all(rc[k].abs_bias_L_mean <= 0.25 * base.abs_bias_L_mean and rc[k].rel_l2_mean < base.rel_l2_mean for k in (10, 25))
    flat = all(1 - rc[k].rmse_L_mean / ss[k].rmse_L_mean < 0.10 for k in (10, 25))
    ridge_helps = all(1 - rr[k].rmse_L_mean / ss[k].rmse_L_mean >= 0.10 for k in (10, 25))
    th_ok = "under" in theta_name
    lines += ["## Registered prediction", "", md(pr), "",
              f"- 'region_const fixes the bias and Rel L2' (operational check: |bias| at most a quarter of the uncorrected mean |bias| and Rel L2 below the uncorrected one, at k = 10 and 25): **{'TRUE' if fixes else 'FALSE'}**.",
              f"- 'region_const barely improves the RMSE after bias removal' (reduction vs scalar_shift below 10% at k = 10 and 25): **{'TRUE' if flat else 'FALSE'}**.",
              f"- 'only region_ridge with theta_under can reduce the RMSE' (reduction >= 10% at k = 10 and 25): {'**TRUE** for region_ridge' if ridge_helps else '**FALSE** for region_ridge'}" + ("" if th_ok else "; theta_under is not in the chosen theta, so this part is untestable") + ".",
              f"- Overall: the prediction is {'CORRECT' if (fixes and flat and (ridge_helps == th_ok or not th_ok)) else 'NOT (fully) correct'} as written; the three statements above carry the evidence.", "",
              "## Existing numbers for comparison (different test sets: the retrain rows use 225 fixed Estateback cars; this file uses the other 250 - k cars of each draw; M2 retrained with k extra cars versus frozen M2)", "", "Retraining:", "", md(reference_rows()[0]), "", "Post-hoc mean-centred conformal (full width / 2 = half-width):", "", md(reference_rows()[1]), ""]
    (OUT / "T3_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("summary written; prediction:", fixes, flat, ridge_helps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute-stats", action="store_true")
    main(ap.parse_args().recompute_stats)
