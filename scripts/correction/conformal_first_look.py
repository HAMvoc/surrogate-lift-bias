"""Few-shot conformal calibration of integral coefficients under body-style shift (prereg addendum 5, exploratory).
Input: reports/integral_errors.csv (scripts/diagnosis/integral_errors.py). Output: reports/conformal_calibration.md.
Usage: python scripts/correction/conformal_first_look.py"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402

ALPHA, KS, DRAWS, GRID = 0.1, (10, 15, 25, 50), 2000, 801
RNG = np.random.default_rng(2026)


def full_conformal(e_cal, r):
    """Hull of z with rank(|z - ebar(z)|) among the k+1 scores <= r, ebar(z) = (sum e_cal + z) / (k + 1) (mean-centred full conformal)."""
    k, s = len(e_cal), e_cal.sum()
    m = s / k
    span = 6 * (np.abs(e_cal - m).max() + 1e-9)
    z = m + np.linspace(-span, span, GRID)
    ebar = (s + z) / (k + 1)
    s_new = np.abs(z - ebar)                                     # (GRID,)
    s_cal = np.abs(e_cal[:, None] - ebar[None, :])               # (k, GRID)
    ok = (s_cal < s_new[None, :]).sum(0) + 1 <= r                # rank of s_new among all k+1 scores
    return (z[ok].min(), z[ok].max()) if ok.any() else (-np.inf, np.inf)


def draw_stats(e, k, r):
    """One random calibration draw from per-car errors e; returns metrics on the remaining cars."""
    idx = RNG.permutation(len(e))
    cal, ev = e[idx[:k]], e[idx[k:]]
    m = cal.mean()
    lo, hi = full_conformal(cal, r)
    qU = np.sort(np.abs(cal))[r - 1]
    qN = np.sort(np.abs(cal - m))[r - 1]
    return {"covP": np.mean((ev >= lo) & (ev <= hi)), "wP": hi - lo, "covU": np.mean(np.abs(ev) <= qU), "wU": 2 * qU, "covN": np.mean(np.abs(ev - m) <= qN), "wN": 2 * qN,
            "maeRaw": np.abs(ev).mean(), "maeCor": np.abs(ev - m).mean(), "biasRaw": ev.mean(), "biasCor": (ev - m).mean()}


def cell(e, k):
    r = math.ceil((1 - ALPHA) * (k + 1))
    d = pd.DataFrame([draw_stats(e, k, r) for _ in range(DRAWS)])
    return d


def main():
    ie = pd.read_csv(ROOT / "reports" / "integral_errors.csv")
    ie["e_cl"], ie["e_cd"] = ie.cl_pred - ie.cl_ref, ie.cd_pred - ie.cd_ref
    models = [("M2", 0), ("M2", 1), ("M2", 2), ("B1", 0), ("B2", 0), ("B2_300ep", 0), ("B3", 0)]
    rows, low, summary = [], [], []
    for f in ("E2A", "E2B", "E2C"):
        for model, seed in models:
            g = ie[(ie.fold == f) & (ie.model == model) & (ie.seed == seed)]
            if len(g) != 250:
                continue
            for q, col in (("C_L,p", "e_cl"), ("C_D,p", "e_cd")):
                e = g[col].to_numpy()
                for k in KS:
                    d = cell(e, k)
                    covP = d.covP.mean()
                    rows.append({"fold": f, "model": f"{model} s{seed}", "quantity": q, "k": k, "raw MAE": f"{d.maeRaw.mean():.1f}", "corrected MAE": f"{d.maeCor.mean():.1f}", "raw bias": f"{d.biasRaw.mean():+.1f}",
                                 "coverage P": f"{covP:.3f}", "draws cov>=0.8": f"{(d.covP >= 0.8).mean():.2f}", "width P": f"{d.wP.mean():.1f}", "coverage U": f"{d.covU.mean():.3f}", "width U": f"{d.wU.mean():.1f}",
                                 "coverage N": f"{d.covN.mean():.3f}", "width N": f"{d.wN.mean():.1f}"})
                    if covP < 0.88:
                        low.append(f"{f} {model} s{seed} {q} k={k}: coverage {covP:.3f}")
                    if f == "E2A" and q == "C_L,p" and k == 25:
                        summary.append((model, seed, d.maeCor.mean() / d.maeRaw.mean()))
                print(f, model, seed, flush=True)
    # R2 per family (seed 0 of each; M2 uses all three seeds)
    r2 = {}
    for model, seed, ratio in summary:
        r2.setdefault(model, []).append(ratio)
    r2_rows = [{"family": m, "corrected / raw MAE at k=25 (C_L,p, E2A)": " / ".join(f"{x:.2f}" for x in v), "<= 0.5 for all seeds": all(x <= 0.5 for x in v)} for m, v in r2.items()]
    # R3: post-hoc correction with the addendum-1 pool (first k of the fixed permutation) on the 225 test cars, M2 seed 0, E2A
    g = ie[(ie.fold == "E2A") & (ie.model == "M2") & (ie.seed == 0)].set_index("id")
    order = [str(i) for i in np.random.default_rng(0).permutation(sorted(g.index))]
    pool, test = order[:25], order[25:]
    retr = {10: 89.3, 25: 51.2}
    r3_rows = []
    for k in (10, 25):
        m = g.loc[pool[:k], "cl_pred"].sub(g.loc[pool[:k], "cl_ref"]).mean()
        ev = g.loc[test, "cl_pred"] - g.loc[test, "cl_ref"]
        mae = float((ev - m).abs().mean())
        r3_rows.append({"k": k, "post-hoc MAE": f"{mae:.1f}", "retrained MAE (calibration.md)": retr[k], "ratio": f"{mae / retr[k]:.2f}", "post-hoc bias left": f"{(ev - m).mean():+.1f}"})
    ratios = [float(r["ratio"]) for r in r3_rows]
    r3 = "post-hoc equivalent (<= 1.1x at both k)" if max(ratios) <= 1.1 else "post-hoc better (< 0.9x at both k)" if max(ratios) < 0.9 else "post-hoc worse (> 1.1x at one k)"
    if max(ratios) < 0.9:
        r3 = "post-hoc better (< 0.9x at both k)"
    # R4: negative control, calibration residuals from another archetype's model (M2 seed 0), method U, C_L,p, evaluated on all 250 E2A cars
    ea = ie[(ie.fold == "E2A") & (ie.model == "M2") & (ie.seed == 0)].e_cl.to_numpy()
    r4_rows = []
    for other in ("E2B", "E2C"):
        eo = ie[(ie.fold == other) & (ie.model == "M2") & (ie.seed == 0)].e_cl.to_numpy()
        for k in (10, 25, 50):
            r = math.ceil((1 - ALPHA) * (k + 1))
            cov = np.mean([np.mean(np.abs(ea) <= np.sort(np.abs(RNG.choice(eo, k, replace=False)))[r - 1]) for _ in range(DRAWS)])
            r4_rows.append({"calibration from": other, "k": k, "coverage on E2A (C_L,p)": f"{cov:.3f}"})
    text = ["# Few-shot conformal calibration of integral coefficients (exploratory; auto-generated by `scripts/correction/conformal_first_look.py`)", "",
            f"Per-car error e = predicted minus CFD coefficient (counts) from the saved per-point predictions (pool IPW estimator, nominal coverage {1 - ALPHA:.0%}). For each (fold, model, k) {DRAWS} random calibration sets of k cars are drawn from the 250 held-out-style cars; "
            "the other 250 - k cars are evaluated. Methods: P = proposed mean-centred full conformal (point estimate = prediction minus the mean calibration error), U = split conformal without bias correction, N = naive centred (reuses the data, reference only). "
            "Width = interval width in counts; `corrected MAE` = mean |e - mean calibration error|. Differences below 10 counts are inside the pool-estimator noise floor.", "",
            "## R2: does the post-hoc bias correction work for every model family? (E2A, C_L,p, k = 25)", "", md(r2_rows), "",
            "## R3: post-hoc correction vs retraining (M2 seed 0, E2A, addendum-1 pool, 225 test cars)", "", md(r3_rows), "", f"Verdict: **{r3}**.", "",
            "## R1: cells where method P coverage < 0.88", "", ("none" if not low else "\n".join(f"- {x}" for x in low)), "",
            "## R4: negative control, calibration residuals from the other archetype's model (method U, M2 seed 0)", "", md(r4_rows), "",
            "## All cells", "", md(rows), ""]
    (ROOT / "reports" / "conformal_calibration.md").write_text(chr(10).join(text), encoding="utf-8")


if __name__ == "__main__":
    main()
