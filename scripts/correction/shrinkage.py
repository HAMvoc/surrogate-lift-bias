"""Label-budget study, task T2 (EXPLORATORY, not a main method): positive-part James-Stein shrinkage of the stage-B offsets, b_shrunk = b * max(0, 1 - SE^2 / b^2), SE = sigma_k / sqrt(k).
Variants: no shrinkage, per-region (SE_r from the SD over the k cars of the per-car mean error in region r), total (all offsets multiplied by max(0, 1 - SE^2/b^2), b = mean C_L,p error of the k cars, SE = SD/sqrt(k)).
Scope: frozen + region_const_intmatch (7 (family, fold) pairs, k in {2, 3, 5, 10, 25}, 50 draws, jackknife+ with the shrinkage inside every refit); retrain + B OOF (Component and cross-model study T3 draws; no coverage);
Two-stage correction study T1 ft_B_OOF (total shrinkage only, C_L,p and C_D,p only: the files hold neither fields nor regional statistics). Output: results/label_budget/T2/. Usage: python scripts/correction/shrinkage.py"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.correction import statistics_cache as pc  # noqa: E402
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.correction.calibration import corrected, fit_const, metrics  # noqa: E402
from scripts.correction.two_stage_oof import fit_b  # noqa: E402
from scripts.correction.frozen_draws import fit_method  # noqa: E402

OUT = ROOT / "results" / "label_budget" / "T2"
KS = (2, 3, 5, 10, 25)
ALPHA = 0.1
VARS = ("none", "region", "total")


def js(c, se):
    """positive-part James-Stein factor applied elementwise to offsets c with standard errors se."""
    c, se = np.asarray(c, float), np.asarray(se, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(c == 0, 0.0, np.maximum(0.0, 1 - se ** 2 / c ** 2))
    return f


def shrink(z, idx, c, kind):
    """c: offsets (S, R) fitted on the cars idx of z; returns shrunk offsets (S, R) and the mean retained fraction."""
    if kind == "none":
        return c, 1.0
    k = len(idx)
    if k < 2:  # SD of one car is undefined (happens inside jackknife+ refits at k = 2): leave the offsets unshrunk
        return c, 1.0
    out = np.empty_like(c)
    for s in range(c.shape[0]):
        if kind == "region":
            m = z["Aw"][s][idx] / np.maximum(z["W"][idx], 1e-12)  # per-car mean error in each region
            se = m.std(axis=0, ddof=1) / np.sqrt(k)
            f = js(c[s], se)
        else:
            b = z["dL"][s][idx].mean()
            se = z["dL"][s][idx].std(ddof=1) / np.sqrt(k)
            f = np.full(c.shape[1], 0.0 if b == 0 else max(0.0, 1 - se ** 2 / b ** 2))
        out[s] = c[s] * f
    return out, float(np.mean(out != 0) if False else 1.0)


def fit_kind(z, idx, kind):
    _, c = fit_method(z, idx, "intmatch_lam")
    return ("off", shrink(z, idx, c, kind)[0])


def jackknife(z, idx, ev, kind):
    k = len(idx)
    eps, R = np.zeros((k, len(ev))), np.zeros(k)
    for a in range(k):
        sub = np.delete(idx, a)
        fit = fit_kind(z, sub, kind)
        eps[a] = corrected(z, ev, fit)[0].mean(0)
        R[a] = abs(corrected(z, idx[a:a + 1], fit)[0].mean(0)[0])
    lo_i, hi_i = max(1, math.floor(ALPHA * (k + 1))), min(k, math.ceil((1 - ALPHA) * (k + 1)))
    lo, hi = np.sort(eps - R[:, None], axis=0)[lo_i - 1], np.sort(eps + R[:, None], axis=0)[hi_i - 1]
    return float(np.mean((lo <= 0) & (hi >= 0))), float(np.mean((hi - lo) / 2))


def frozen_part():
    rows = []
    ev = np.arange(125, 250)
    for fold, fam in pc.PAIRS:
        z = pc.pair_stats(fold, fam)
        for d in range(50):
            for k in KS:
                idx = np.random.default_rng(100 + d).permutation(125)[:k]
                for kind in VARS:
                    fit = fit_kind(z, idx, kind)
                    m = metrics(z, ev, fit)
                    cov, half = jackknife(z, idx, ev, kind)
                    rows.append({"part": "frozen + B", "fold": fold, "family": fam, "k": k, "draw": d, "variant": kind, "abs_bias_L": m["abs_bias_L"], "bias_L": m["bias_L"], "abs_bias_D": m["abs_bias_D"], "rel_l2": m["rel_l2"], "rmse_L": m["rmse_L"],
                                 "coverage": cov, "half": half})
        print(fold, fam, "frozen done", flush=True)
    return rows


def t3_part():
    rows = []
    for k, ds in ((10, range(3)), (25, range(4))):
        for d in ds:
            ze, zo = pc.t3_stats(k, d)
            evi = np.arange(len(ze["CL"]))
            idx = np.arange(k)
            lam, c = fit_b(zo, idx)
            for kind in VARS:
                cs, _ = shrink(zo, idx, c[None], kind)
                m = metrics(ze, evi, ("off", cs))
                rows.append({"part": "retrain + B OOF", "fold": "E2A", "family": "M2", "k": k, "draw": d, "variant": kind, "abs_bias_L": m["abs_bias_L"], "bias_L": m["bias_L"], "abs_bias_D": m["abs_bias_D"], "rel_l2": m["rel_l2"], "rmse_L": m["rmse_L"], "coverage": np.nan, "half": np.nan})
    return rows


def p3_part():
    rows = []
    for v in ("dec", "full"):
        for k in (5, 10, 25):
            for d in range(10):
                per = {"none": [], "total": []}
                for s in range(3):
                    z = np.load(ROOT / "results" / "two_stage" / "T1" / "raw" / f"{v}_s{s}_k{k}_d{d}.npz")
                    oof = z["oof"][:, 0]
                    b, se = oof.mean(), oof.std(ddof=1) / np.sqrt(k)
                    f = 0.0 if b == 0 else max(0.0, 1 - se ** 2 / b ** 2)
                    nb, bb = z["noB_full"], z["B_full"]
                    per["none"].append(bb)
                    per["total"].append(nb + f * (bb - nb))
                for kind in ("none", "total"):
                    a = np.mean(per[kind], axis=0)  # (125, 3) seed-mean per-car dL, dD, rel (rel is NOT valid after shrinkage)
                    rows.append({"part": "Two-stage correction study T1 ft_B_OOF", "fold": "E2A", "family": v, "k": k, "draw": d, "variant": kind, "abs_bias_L": abs(a[:, 0].mean()), "bias_L": a[:, 0].mean(), "abs_bias_D": abs(a[:, 1].mean()),
                                 "rel_l2": a[:, 2].mean() if kind == "none" else np.nan, "rmse_L": np.sqrt(((a[:, 0] - a[:, 0].mean()) ** 2).mean()), "coverage": np.nan, "half": np.nan})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(frozen_part() + t3_part() + p3_part())
    R.round(5).to_csv(OUT / "T2_shrinkage.csv", index=False)
    summarise(R)


def summarise(R):
    f = lambda g, c, d=1: f"{g[c].mean():.{d}f}"
    L = ["# Label-budget study, task T2 (EXPLORATORY): positive-part James-Stein shrinkage of stage B", "",
         "Exploratory, not used as a main method; for discussion only. Variants: `none` = stage B as in the two-stage correction study/4; `region` = each region offset shrunk with its own standard error; `total` = all offsets multiplied by one factor from the total C_L,p estimate. "
         "Mean over draws (frozen: 50 draws; retrain + B OOF: 7 draws; Two-stage correction study T1: 10 draws and 3 seeds). Coverage: jackknife+ with the shrinkage inside every refit (frozen only). Two-stage correction study T1 rows: `total` only, C_L,p and C_D,p only (Rel L2 not recomputable from the stored files).", ""]
    for k in (5, 10):
        tab = []
        for (fold, fam), _ in R[R.part == "frozen + B"].groupby(["fold", "family"]):
            for v in VARS:
                g = R[(R.part == "frozen + B") & (R.fold == fold) & (R.family == fam) & (R.k == k) & (R.variant == v)]
                tab.append({"fold": fold, "family": fam, "variant": v, "|bias| C_L,p": f(g, "abs_bias_L"), "|bias| C_D,p": f(g, "abs_bias_D"), "Rel L2": f(g, "rel_l2", 4), "RMSE": f(g, "rmse_L"), "coverage": f(g, "coverage", 3)})
        L += [f"## frozen + B, k = {k}", "", md(tab), ""]
    tab = []
    for k in KS:
        for v in VARS:
            g = R[(R.part == "frozen + B") & (R.family == "M2") & (R.k == k) & (R.variant == v)]
            tab.append({"k": k, "variant": v, "|bias| C_L,p": f(g, "abs_bias_L"), "|bias| C_D,p": f(g, "abs_bias_D"), "Rel L2": f(g, "rel_l2", 4), "coverage": f(g, "coverage", 3), "draws with |bias| larger than `none`": "-" if v == "none" else
                        f"{int((g.abs_bias_L.values > R[(R.part == 'frozen + B') & (R.family == 'M2') & (R.k == k) & (R.variant == 'none')].abs_bias_L.values).sum())} / {len(g)}"})
    L += ["## M2 frozen + B, all k", "", md(tab), ""]
    tab = []
    for (k, d), _ in R[R.part == "retrain + B OOF"].groupby(["k", "draw"]):
        row = {"k": k, "draw": d}
        for v in VARS:
            g = R[(R.part == "retrain + B OOF") & (R.k == k) & (R.draw == d) & (R.variant == v)].iloc[0]
            row[f"{v}: |bias| L / D / Rel L2"] = f"{g.abs_bias_L:.1f} / {g.abs_bias_D:.1f} / {g.rel_l2:.4f}"
        tab.append(row)
    L += ["## retrain + B OOF per draw (Component and cross-model study T3 draws; k = 10, d = 1 and 2 are the draws where stage B raised |bias|)", "", md(tab), ""]
    tab = []
    for k in (10, 25):
        row = {"k": k}
        for v in VARS:
            g = R[(R.part == "retrain + B OOF") & (R.k == k) & (R.variant == v)]
            row[f"{v}: mean |bias| L / D / Rel L2"] = f"{g.abs_bias_L.mean():.1f} / {g.abs_bias_D.mean():.1f} / {g.rel_l2.mean():.4f}"
        tab.append(row)
    L += ["## retrain + B OOF, mean over draws", "", md(tab), ""]
    tab = []
    for v in ("dec", "full"):
        for k in (5, 10, 25):
            row = {"variant": v, "k": k}
            for kind in ("none", "total"):
                g = R[(R.part == "Two-stage correction study T1 ft_B_OOF") & (R.family == v) & (R.k == k) & (R.variant == kind)]
                row[f"{kind}: |bias| L / D"] = f"{g.abs_bias_L.mean():.1f} / {g.abs_bias_D.mean():.1f}"
            tab.append(row)
    L += ["## Two-stage correction study T1 ft_B_OOF (total shrinkage only)", "", md(tab), ""]
    (OUT / "T2_shrinkage.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    if "--summary-only" in sys.argv:
        summarise(pd.read_csv(OUT / "T2_shrinkage.csv"))
    else:
        main()
