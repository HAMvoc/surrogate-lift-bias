"""Centerline and axle study: Reynolds number (A), centreline pressure profiles (B) and front-/rear-axle lift (C).
Definitions as scripts/data/per_car_coefficients.py and scripts/diagnosis/cumulative_lift.py (Cp clipped, IPW pool weights, S = total / frontal area, counts). Predictions: seed-mean E2A test predictions.
Self-checks: C_Lf + C_Lr = stored C_L,p (1e-6); a pure lift at the front axle goes to the front axle; G_f + G_r = G = 112.755 (0.01); M2 bias = 183.9 +- 0.5.
Output: results/centerline_and_axles/{reynolds.csv, centre_profiles.csv, centre_rear.csv, axle_cars.csv, axle_split.csv, axle_table_check.csv, verdict.md, centreline.png},
paper_assets/figures/centreline.pdf. Usage: python scripts/diagnosis/centerline_and_axles.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.pressure_coefficients import CLIP, GEO, pool_weights  # noqa: E402
from scripts.diagnosis.force_alignment import load_pool  # noqa: E402
from scripts.correction.cross_model import DIRS  # noqa: E402

OUT = ROOT / "results" / "centerline_and_axles"
PARAMS = (ROOT / ".." / "DrivAerNet" / "ParametricModels" / "DrivAerNet_ParametricData.csv").resolve()
L_M, U, NU = 5.0, 30.0, 1.562e-5
G0 = 112.755
NB, HALF, NZ, REAR = 50, 0.025, 0.1, 0.6
FAMS = ("B0", "B1", "B2", "B2_300ep", "B3", "M2", "TSV")
RUNS = {**DIRS, "TSV": [f"{{f}}_TSV_R1_s{s}" for s in range(3)]}
PROF = ("M2", "TSV")  # models whose centreline profile is computed
TYPES = ("Fastback", "Notchback", "Estateback")


def axle_front(xm, zm, cx, cz, x_r, wb):
    """Front-axle lift: moment of the force about the rear contact point (x_r, 0), divided by the wheelbase."""
    return float((zm * cx - (xm - x_r) * cz).sum() / wb)


def _selftest():
    # a pure lift of 100 at the front-axle position goes entirely to the front axle; a pure lift at the rear axle to the rear axle
    assert np.isclose(axle_front(np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([100.0]), 3.8, 2.8), 100.0)
    assert np.isclose(axle_front(np.array([3.8]), np.array([0.0]), np.array([0.0]), np.array([100.0]), 3.8, 2.8), 0.0)


def centre_bins(xr, y, nz, w, cp):
    """(2, NB) w-weighted mean Cp in the centreline strip per bin; row 0 upper, row 1 lower; NaN when fewer than 3 points."""
    o = np.full((2, NB), np.nan)
    b = np.minimum((xr * NB).astype(int), NB - 1)
    for k, sel in enumerate((nz > NZ, nz < -NZ)):
        s = sel & (np.abs(y) <= HALF)
        n = np.bincount(b[s], minlength=NB)
        sw = np.bincount(b[s], weights=w[s], minlength=NB)
        sv = np.bincount(b[s], weights=w[s] * cp[s], minlength=NB)
        ok = n >= 3
        o[k, ok] = sv[ok] / sw[ok]
    return o


def per_car():
    cars = pd.read_csv(ROOT / "results" / "classical_baselines" / "cars.csv")
    wf = pd.read_csv(ROOT / "results" / "decomposition" / "T1" / "wheel_fits.csv").groupby(["id", "axle"]).x_c.mean().unstack()
    preds, pos = {}, {}
    for f in FAMS:
        ds = [ROOT / "results" / d.format(f="E2A") for d in RUNS[f]]
        preds[f] = [np.load(d / "test_pred_fp16.npy", mmap_mode="r") for d in ds]
        pos[f] = [{i: j for j, i in enumerate(pd.read_csv(d / "test_per_car.csv").id)} for d in ds]
    rows, prof, mprof = [], np.full((len(cars), 2, NB), np.nan), {m: {} for m in PROF}
    for j, (sid, cl, role) in enumerate(zip(cars.sample_id, cars.CL, cars.role)):
        xyz, nrm, cp, area = load_pool(sid)
        p = xyz.astype(np.float64) * L_M
        ref = np.clip(cp.astype(np.float64), *CLIP)
        w = pool_weights(np.asarray(area, np.float64), sid)
        S = float(GEO.at[sid, "area_total_m2"] / GEO.at[sid, "frontal_area_m2"])
        kz, kx = -1000 * S * w * nrm[:, 2] / w.sum(), -1000 * S * w * nrm[:, 0] / w.sum()
        x, z = p[:, 0], p[:, 2]
        L = x.max() - x.min()
        xr = (x - x.min()) / L
        x_f, x_r = wf.at[sid, "front"], wf.at[sid, "rear"]
        wb = x_r - x_f
        cz, cx = kz * ref, kx * ref
        clf = axle_front(x, z, cx, cz, x_r, wb)
        assert abs(cz.sum() - cl) < 1e-3, (sid, cz.sum(), cl)
        r = {"sample_id": sid, "category": cars.category[j], "role": role, "L_m": L, "wb_m": wb, "x_f_rel": (x_f - x.min()) / L, "x_r_rel": (x_r - x.min()) / L,
             "CL": cz.sum(), "CLf": clf, "CLr": cz.sum() - clf}
        prof[j] = centre_bins(xr, p[:, 1], nrm[:, 2], w, ref)
        if role == "test":
            for f in FAMS:
                pr = np.mean([np.asarray(a[q[sid]], np.float64) for a, q in zip(preds[f], pos[f])], 0)
                pf = axle_front(x, z, kx * pr, kz * pr, x_r, wb)
                r[f"{f}_CL"], r[f"{f}_CLf"], r[f"{f}_CLr"] = (kz * pr).sum(), pf, (kz * pr).sum() - pf
                if f in PROF:
                    mprof[f][j] = centre_bins(xr, p[:, 1], nrm[:, 2], w, pr)
        rows.append(r)
    A = pd.DataFrame(rows)
    assert np.allclose(A.CLf + A.CLr, A.CL, atol=1e-6)
    return A, prof, mprof


def boot_diff(a, b, n=2000, seed=0):
    """95% percentile interval of mean(a) - mean(b), resampling within each group (rows = cars)."""
    rng = np.random.default_rng(seed)
    d = np.array([np.nanmean(a[rng.integers(0, len(a), len(a))], 0) - np.nanmean(b[rng.integers(0, len(b), len(b))], 0) for _ in range(n)])
    return np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0)


def boot_mean(a, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.array([np.nanmean(a[rng.integers(0, len(a), len(a))], 0) for _ in range(n)])
    return np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0)


def main():
    _selftest()
    OUT.mkdir(parents=True, exist_ok=True)
    A, prof, mprof = per_car()
    A.round(6).to_csv(OUT / "axle_cars.csv", index=False)
    tr, te = (A.role == "train").to_numpy(), (A.role == "test").to_numpy()
    T = A[te]

    # A. Reynolds number
    A["Re"] = U * A.L_m / NU
    re = [{"group": g, "n": len(d), "L_med": d.L_m.median(), "L_min": d.L_m.min(), "L_max": d.L_m.max(), "Re_med": d.Re.median(), "Re_min": d.Re.min(), "Re_max": d.Re.max(),
           "wb_med": d.wb_m.median()} for g, d in [("all", A)] + [(t, A[A.category == t]) for t in TYPES]]
    pd.DataFrame(re).to_csv(OUT / "reynolds.csv", index=False)

    # C. axle split
    G = A[tr].CL.mean() - T.CL.mean()
    assert abs(G - G0) < 0.01, G
    gl, gh = boot_diff(A[tr][["CLf", "CLr"]].to_numpy(), T[["CLf", "CLr"]].to_numpy())
    Gf, Gr = A[tr].CLf.mean() - T.CLf.mean(), A[tr].CLr.mean() - T.CLr.mean()
    sd_f, sd_r = T.CLf.std(ddof=0), T.CLr.std(ddof=0)
    rows = [{"model": "gap (CFD)", "B": G, "B_f": Gf, "B_f_lo": gl[0], "B_f_hi": gh[0], "B_r": Gr, "B_r_lo": gl[1], "B_r_hi": gh[1], "rear_share": Gr / G}]
    for f in FAMS:
        e = np.c_[T[f"{f}_CLf"] - T.CLf, T[f"{f}_CLr"] - T.CLr]
        lo, hi = boot_mean(e)
        b = e.mean(0)
        share = b[1] / b.sum()
        rows.append({"model": f, "B": b.sum(), "B_f": b[0], "B_f_lo": lo[0], "B_f_hi": hi[0], "B_r": b[1], "B_r_lo": lo[1], "B_r_hi": hi[1], "rear_share": share,
                     "O_f": b[0] - Gf, "O_r": b[1] - Gr, "B_f_over_sd": b[0] / sd_f, "B_r_over_sd": b[1] / sd_r,
                     "verdict": "rear-axle" if share >= 2 / 3 else "front-axle" if b[0] / b.sum() >= 2 / 3 else "both"})
    X = pd.DataFrame(rows)
    m2 = X.set_index("model").loc["M2"]
    assert abs(m2.B - 183.9) <= 0.5, m2.B
    X.to_csv(OUT / "axle_split.csv", index=False)

    tab = pd.read_csv(PARAMS).set_index("Experiment").loc[A.sample_id]
    chk = []
    for g, sel in [("all", np.ones(len(A), bool))] + [(t, (A.category == t).to_numpy()) for t in TYPES]:
        d = A[sel]
        tf, trr = 1000 * tab["Average Cl_f"].to_numpy()[sel], 1000 * tab["Average Cl_r"].to_numpy()[sel]
        chk.append({"group": g, "n": int(sel.sum()), "r_front": np.corrcoef(d.CLf, tf)[0, 1], "r_rear": np.corrcoef(d.CLr, trr)[0, 1],
                    "mean_diff_front": (d.CLf - tf).mean(), "mean_diff_rear": (d.CLr - trr).mean(), "r_total": np.corrcoef(d.CL, 1000 * tab["Average Cl"].to_numpy()[sel])[0, 1]})
    C = pd.DataFrame(chk)
    C.to_csv(OUT / "axle_table_check.csv", index=False)

    # B. centreline
    xc = (np.arange(NB) + 0.5) / NB
    P = {"x": xc}
    for k, s in enumerate(("upper", "lower")):
        for t in TYPES:
            P[f"{s}_{t.lower()}"] = np.nanmean(prof[(A.category == t).to_numpy(), k], 0)
        P[f"{s}_train"] = np.nanmean(prof[tr, k], 0)
        P[f"{s}_n_test"] = np.sum(~np.isnan(prof[te, k]), 0)
        for m in PROF:
            P[f"{s}_{m}"] = np.nanmean(np.array([mprof[m][j][k] for j in np.where(te)[0]]), 0)
    Pdf = pd.DataFrame(P)
    Pdf.round(5).to_csv(OUT / "centre_profiles.csv", index=False)
    rear = xc >= REAR
    rr = []
    for k, s in enumerate(("upper", "lower")):
        cfd = prof[te, k]
        dref = Pdf[f"{s}_train"].to_numpy() - Pdf[f"{s}_estateback"].to_numpy()
        for m in PROF:
            e = np.array([mprof[m][j][k] for j in np.where(te)[0]]) - cfd  # per car and bin
            er = np.nanmean(e[:, rear], 1)  # per car mean over rear bins
            lo, hi = boot_mean(er[:, None])
            eb = np.nanmean(e, 0)
            use = rear & (np.abs(dref) >= 0.02)
            rr.append({"surface": s, "model": m, "D_ref_rear": float(np.nanmean(dref[rear])), "E_rear": float(np.nanmean(er)), "E_lo": float(lo[0]), "E_hi": float(hi[0]),
                       "copy_frac": float(np.mean(eb[use] / dref[use])) if use.any() else np.nan, "n_bins_used": int(use.sum()),
                       "x_maxdiff": float(xc[np.nanargmax(np.abs(dref))])})
    R = pd.DataFrame(rr)
    R.to_csv(OUT / "centre_rear.csv", index=False)

    up = R[(R.surface == "upper") & (R.model == "M2")].iloc[0]
    lw = R[(R.surface == "lower") & (R.model == "M2")].iloc[0]
    nrear = int((X.verdict.iloc[1:] == "rear-axle").sum())
    re0 = re[0]
    L = ["# Centerline and axle study: Reynolds number, centreline profiles, axle lift (criteria: `PREREG.md`)", "",
         f"Self-checks passed: C_Lf + C_Lr = C_L,p for all {len(A)} cars; axle unit test; G_f + G_r = {Gf + Gr:.3f} = G; M2 bias {m2.B:.1f} (gate).", "",
         f"**A:** L median {re0['L_med']:.3f} m ({re0['L_min']:.3f} to {re0['L_max']:.3f}); Re_L median {re0['Re_med']:.3g} ({re0['Re_min']:.3g} to {re0['Re_max']:.3g}); wheelbase median {re0['wb_med']:.3f} m. Prediction: about 4.6 m, 8.9e6.", "",
         f"**Q-B1:** largest |Cp(train) - Cp(estateback)| on the upper centreline at x_rel = {up.x_maxdiff:.2f}: {'rear' if up.x_maxdiff >= REAR else 'front'}. Prediction: rear.",
         f"**Q-B2 (M2, x_rel >= {REAR}):** upper E = {up.E_rear:+.4f} [{up.E_lo:+.4f}; {up.E_hi:+.4f}], lower E = {lw.E_rear:+.4f} [{lw.E_lo:+.4f}; {lw.E_hi:+.4f}]. Prediction: upper < 0, lower > 0.",
         f"**Q-B3:** copy fraction of M2 on the upper surface f = {up.copy_frac:.2f} ({up.n_bins_used} bins). Prediction: > 0.5.", "",
         f"**Q-C1:** M2 rear share B_r / B = {m2.rear_share:.2f}, verdict {m2.verdict}; rear-axle for {nrear} of 7 models. Prediction: rear-axle, 0.65 to 0.85; at least 5 of 7.",
         f"**Q-C2:** gap rear share G_r / G = {Gr / G:.2f} (G_f {Gf:.1f}, G_r {Gr:.1f}). Prediction: >= 0.5.", "",
         "## Axle split (counts)", "", md(X.round(3).to_dict("records")), "", "## Table check (pool pressure axle lift against table total axle lift)", "", md(C.round(3).to_dict("records")), "",
         "## Centreline rear part", "", md(R.round(4).to_dict("records")), "", "## Reynolds number", "", md(pd.DataFrame(re).round(4).to_dict("records")), ""]
    (OUT / "verdict.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))
    fig(Pdf)


def fig(P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True, sharey=False)
    for k, (s, title) in enumerate((("upper", "upper surface (roof, hood)"), ("lower", "lower surface (underbody)"))):
        from src.utils.plot_style import OKABE, TYPE  # colour-vision-safe colours
        for t, (c, ls) in TYPE.items():
            ax[k].plot(P.x, P[f"{s}_{t}"], color=c, ls=ls, lw=1.1, label=f"{t} (CFD)")
        ax[k].plot(P.x, P[f"{s}_M2"], color=OKABE["blue"], ls="--", lw=1.1, label="M2 on estateback")
        ax[k].plot(P.x, P[f"{s}_TSV"], color="k", ls="-.", lw=1.1, label="Transolver on estateback")
        ax[k].invert_yaxis()
        ax[k].set(xlabel="relative position from the nose, $x/L$", ylabel="$C_p$ on the centerline", title=title)
        ax[k].title.set_fontsize(9)
    ax[0].legend(frameon=False, fontsize=7)
    p = ROOT / "paper_assets" / "figures"
    p.mkdir(parents=True, exist_ok=True)
    f.savefig(p / "centreline.pdf"); f.savefig(OUT / "centreline.png", dpi=150)


if __name__ == "__main__":
    main()
