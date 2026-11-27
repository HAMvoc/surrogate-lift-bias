"""Safety-margin and cumulative-lift study B: cumulative lift along the body, from the nose to the relative position x, for the training types and the estateback type, the class-mean gap curve with a bootstrap band,
and the same curve for the M2 lift error. Definitions as scripts/data/per_car_coefficients.py (Cp clipped, IPW pool weights, S = total / frontal area).
Self-checks: C(1) equals the stored per-car C_L,p (results/classical_baselines/cars.csv) to 1e-3; Delta(1) = G = 112.755 to 0.01; the M2 error curve ends at the M2 bias (gate 183.9 +- 0.5).
Output: results/margin_and_cumulative_lift/{cumlift_curves.csv, cumlift_tenths.csv, cumlift_verdict.md}, paper_assets/figures/cumlift.pdf. Usage: python scripts/diagnosis/cumulative_lift.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.pressure_coefficients import CLIP, GEO, pool_weights  # noqa: E402
from scripts.diagnosis.force_alignment import load_pool  # noqa: E402
from scripts.correction.cross_model import DIRS  # noqa: E402

OUT = ROOT / "results" / "margin_and_cumulative_lift"
X = np.linspace(0, 1, 101)
G0 = 112.755


def cum(xr, v):
    """Cumulative sum of v over points with x_rel <= X (grid)."""
    o = np.argsort(xr)
    cs = np.concatenate([[0.0], np.cumsum(v[o])])
    return cs[np.searchsorted(xr[o], X, side="right")]


def curves():
    cars = pd.read_csv(ROOT / "results" / "classical_baselines" / "cars.csv")
    dirs = [ROOT / "results" / d.format(f="E2A") for d in DIRS["M2"]]
    preds = [np.load(d / "test_pred_fp16.npy", mmap_mode="r") for d in dirs]
    pos = [{i: j for j, i in enumerate(pd.read_csv(d / "test_per_car.csv").id)} for d in dirs]
    C, B = np.zeros((len(cars), len(X))), {}
    for j, (sid, cl, role) in enumerate(zip(cars.sample_id, cars.CL, cars.role)):
        xyz, nrm, cp, area = load_pool(sid)
        ref = np.clip(cp.astype(np.float64), *CLIP)
        w = pool_weights(np.asarray(area, np.float64), sid)
        S = float(GEO.at[sid, "area_total_m2"] / GEO.at[sid, "frontal_area_m2"])
        x = xyz[:, 0].astype(np.float64)
        xr = (x - x.min()) / (x.max() - x.min())
        f = -1000 * S * w * nrm[:, 2] / w.sum()
        C[j] = cum(xr, f * ref)
        assert abs(C[j, -1] - cl) < 1e-3, (sid, C[j, -1], cl)
        if role == "test":
            pr = np.mean([np.asarray(p[q[sid]], np.float64) for p, q in zip(preds, pos)], 0)
            B[sid] = cum(xr, f * (pr - ref))
    return cars, C, B


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cars, C, B = curves()
    tr, te = (cars.role == "train").to_numpy(), (cars.role == "test").to_numpy()
    D = C[tr].mean(0) - C[te].mean(0)
    assert abs(D[-1] - G0) < 0.01, D[-1]
    Bm = np.mean([B[s] for s in cars.sample_id[te]], 0)
    assert abs(Bm[-1] - 183.9) <= 0.5, Bm[-1]  # gate (seed-mean prediction = mean over seeds of the bias)
    rng = np.random.default_rng(0)
    itr, ite = np.where(tr)[0], np.where(te)[0]
    boot = np.array([C[rng.choice(itr, len(itr))].mean(0) - C[rng.choice(ite, len(ite))].mean(0) for _ in range(2000)])
    lo, hi = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)
    cur = pd.DataFrame({"x": X, "gap": D, "gap_lo": lo, "gap_hi": hi, "m2_bias": Bm, "overshoot": Bm - D,
                        **{f"mean_{t.lower()}": C[(cars.category == t).to_numpy()].mean(0) for t in ("Fastback", "Notchback", "Estateback")},
                        "mean_train": C[tr].mean(0), "mean_test": C[te].mean(0)})
    cur.round(4).to_csv(OUT / "cumlift_curves.csv", index=False)
    ix = np.arange(0, 101, 10)
    inc = lambda a: a[..., ix[1:]] - a[..., ix[:-1]]
    bi = inc(boot)
    T = pd.DataFrame({"tenth": [f"{a / 100:.1f}-{b / 100:.1f}" for a, b in zip(ix[:-1], ix[1:])], "gap_inc": inc(D), "gap_inc_lo": np.percentile(bi, 2.5, 0), "gap_inc_hi": np.percentile(bi, 97.5, 0),
                      "share_of_G": inc(D) / D[-1], "m2_bias_inc": inc(Bm), "overshoot_inc": inc(Bm - D)})
    T.round(4).to_csv(OUT / "cumlift_tenths.csv", index=False)
    rear, front = T.gap_inc[7:].sum() / D[-1], T.gap_inc[:3].sum() / D[-1]
    verdict = "rear-concentrated" if rear >= 2 / 3 else "front-loaded" if front >= 2 / 3 else "spread"
    excl = np.where((lo > 0) | (hi < 0))[0]
    xs = float(X[excl[0]]) if len(excl) else float("nan")
    xh = float(X[np.argmax(D >= D[-1] / 2)])
    rho, p = spearmanr(T.gap_inc, T.m2_bias_inc)
    pd.DataFrame([("rear_share", rear), ("front_share", front), ("x_sep", xs), ("x_half", xh), ("spearman_tenths", rho), ("spearman_p", p), ("G", D[-1]), ("m2_bias", Bm[-1]),
                  ("gap_max", D.max()), ("x_gap_max", float(X[D.argmax()])), ("gap_min", D.min()), ("x_gap_min", float(X[D.argmin()]))], columns=["statistic", "value"]).to_csv(OUT / "cumlift_constants.csv", index=False)
    L = ["# Safety-margin and cumulative-lift study B: cumulative lift along the body (criteria: `PREREG.md`, part B)", "",
         f"Self-checks passed: C(1) = stored C_L,p for all {len(cars)} cars (1e-3); Delta(1) = {D[-1]:.3f} = G; M2 error curve ends at {Bm[-1]:.1f} counts (gate).", "",
         f"**Q-B1 verdict: {verdict}** - last three tenths carry {rear:.2f} of G, first three tenths {front:.2f}. Separation point x_s = {xs:.2f}; half of G reached at x = {xh:.2f}; "
         f"Delta ranges from {D.min():.1f} (x = {X[D.argmin()]:.2f}) to {D.max():.1f} (x = {X[D.argmax()]:.2f}) counts. Prediction: spread, rear third 30-50% of G, x_s < 0.3.", "",
         f"**Q-B2:** Spearman over the ten tenths between the increments of Delta and of the M2 error curve: {rho:.2f} (p {p:.3f}). Prediction: >= 0.6.", "",
         "## Increments per tenth of the length (counts; interval: 95% bootstrap over cars)", "", md(T.round(2).to_dict("records")), ""]
    (OUT / "cumlift_verdict.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:8]))
    fig(cur)


def fig(cur):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    from src.utils.plot_style import OKABE, TYPE  # colour-vision-safe colours
    for t, (c, ls) in TYPE.items():
        ax[0].plot(cur.x, cur[f"mean_{t}"], color=c, ls=ls, label=t)
    ax[0].set(xlabel="relative position from the nose, $x/L$", ylabel="cumulative $C_{L,p}$ (counts)"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].fill_between(cur.x, cur.gap_lo, cur.gap_hi, color="0.8", lw=0)
    ax[1].plot(cur.x, cur.gap, color="k", label="class-mean gap (CFD)")
    ax[1].plot(cur.x, cur.m2_bias, color=OKABE["blue"], ls="--", label="M2 lift error")
    ax[1].axhline(0, color="0.5", lw=0.6)
    ax[1].set(xlabel="relative position from the nose, $x/L$", ylabel="cumulative (counts)"); ax[1].legend(frameon=False, fontsize=8)
    p = ROOT / "paper_assets" / "figures"
    p.mkdir(parents=True, exist_ok=True)
    f.savefig(p / "cumlift.pdf"); f.savefig(OUT / "cumlift.png", dpi=150)


if __name__ == "__main__":
    main()
