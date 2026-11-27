"""Force-alignment study of the C1 investigation (force alignment of the Cp error), tasks T1-T5. Reads saved per-point predictions only; writes to results/force_alignment/.
Per-car metrics are computed per seed and then averaged over the seeds of a model (predictions are NOT ensembled); CIs are bootstrap over cars (B = 2000, seed 0).
Usage: python scripts/diagnosis/force_alignment_terms.py [--role test|val] [--steps terms,t1,t2,t3,t4,t5] [--recompute]
--role val re-runs T1-T3 on the validation cars (50 per fold, in-distribution for the held-out-style folds) using predictions recomputed from the saved best.pt (results/force_alignment/ckpt_preds, scripts/training/predict_from_checkpoint.py); only M2 has them. T4/T5 need hold-out test cars and are skipped."""
import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.pressure_coefficients import pool_weights  # noqa: E402
from scripts.diagnosis.force_alignment import REGIONS, cfd_coefficients, load_pool, terms_table  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

OUT, RES = ROOT / "results" / "force_alignment", ROOT / "results"
CKPT = OUT / "ckpt_preds"  # predictions recomputed from best.pt by scripts/training/predict_from_checkpoint.py (validation sets; E1 test seeds 0 and 1, whose saved predictions were lost)
B = 2000
FOLDS = ("E2A", "E2B", "E2C")
MODELS = {"M2": ("{f}_M2_R2_s{s}", (0, 1, 2)), "B0": ("{f}_B0_NA_s{s}", (0,)), "B1": ("{f}_B1_R1_s{s}", (0,)), "B2": ("{f}_B2_R1_s{s}", (0,)),
          "B2_300ep": ("{f}_B2_R1_s{s}_ep300", (0,)), "B3": ("{f}_B3_R1_s{s}_lr3e-3", (0,))}
COL = {"M2": "#0072B2", "B0": "#7f7f7f", "B1": "#CC79A7", "B2": "#D55E00", "B2_300ep": "#E69F00", "B3": "#009E73"}
MK = {"M2": "o", "B0": "x", "B1": "s", "B2": "^", "B2_300ep": "v", "B3": "D"}
role, sfx = "test", ""


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "LOG.md", "a", encoding="utf-8") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = x[np.random.default_rng(0).integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ci_ratio(num, den):
    """Ratio of means (num/den, in %) with the same bootstrap cars for numerator and denominator."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.default_rng(0).integers(0, len(num), (B, len(num)))
    r = 100 * num[idx].mean(1) / den[idx].mean(1)
    return float(100 * num.mean() / den.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


fm = lambda t, d=3, s=True: f"{t[0]:{'+' if s else ''}.{d}f} [{t[1]:{'+' if s else ''}.{d}f}; {t[2]:{'+' if s else ''}.{d}f}]"


# ------------------------------------------------------------------ per-car terms
def build_terms(recompute):
    f_out = OUT / f"per_car_terms{sfx}.csv.gz"
    if f_out.exists() and not recompute:
        return pd.read_csv(f_out)
    parts = []
    for fold in ("E1",) + FOLDS if role == "test" else FOLDS:
        runs = []
        for m, (pat, seeds) in MODELS.items():
            for s in seeds:
                name = pat.format(f=fold, s=s)
                d = CKPT / name if role == "val" or (not (RES / name / "test_pred_fp16.npy").exists() and (CKPT / name / "test_pred_fp16.npy").exists()) else RES / name
                runs.append((m, s, d))
        t0 = time.time()
        d = terms_table(fold, role, runs, log=log)
        log(f"terms {fold} {role}: {len(d)} rows ({d.groupby(['model', 'seed']).ngroups if len(d) else 0} model-seed runs) in {time.time() - t0:.0f} s")
        parts.append(d)
    t = pd.concat([p for p in parts if len(p)], ignore_index=True)
    t["absrho_L"] = t.rho_L.abs()
    t.to_csv(f_out, index=False, compression="gzip")
    return t


def aggregate(t):
    """Mean over seeds per (fold, model, car)."""
    A = t.drop(columns=["seed", "role"]).groupby(["fold", "model", "id"]).mean(numeric_only=True).reset_index()
    A["n_seeds"] = t.groupby(["fold", "model", "id"]).seed.nunique().values
    return A


def subsets(fold, g):
    if fold == "E1":
        return [("E1 (all 75 cars)", g), ("E1 (Estateback cars only)", g[g.id.str[0] == "E"])]
    return [(fold, g)]


# ------------------------------------------------------------------ T1
def t1(T, A):
    rows, num = [], []
    for (f, m), g in A.groupby(["fold", "model"]):
        seeds = sorted(T[(T.fold == f) & (T.model == m)].seed.unique())
        for lab, gg in subsets(f, g):
            c = {k: ci(gg[k]) for k in ("rho_L", "absrho_L", "e_norm", "dL", "rho_D", "dD")}
            neg = float((gg.rho_L < 0).mean())
            rows.append({"fold": lab, "model": m, "n cars": len(gg), "seeds": ",".join(map(str, seeds)), "mean rho_L [CI]": fm(c["rho_L"]), "mean |rho_L| [CI]": fm(c["absrho_L"], s=False),
                         "share of cars rho_L<0": f"{neg:.2f}", "mean ||e||_A [CI]": fm(c["e_norm"], s=False), "bias C_L,p (counts) [CI]": fm(c["dL"], 1),
                         "mean rho_D [CI]": fm(c["rho_D"]), "bias C_D,p (counts) [CI]": fm(c["dD"], 1)})
            num.append({"fold": lab, "model": m, "n": len(gg), "seeds": ",".join(map(str, seeds)), "frac_rho_L_neg": neg,
                        **{f"{k}_{s}": v for k, c_ in c.items() for k, v, s in zip([k] * 3, c_, ("mean", "lo", "hi"))}})
    pd.DataFrame(num).to_csv(OUT / f"T1_family_alignment{sfx}.csv", index=False)
    txt = ["# T1: force alignment by model family and fold" + (" (validation cars)" if role == "val" else ""), "",
           "Per-car values are averaged over the seeds of the model (seeds column), then mean over n cars with bootstrap 95% CI (B = 2000, over cars). rho_L = <e, n_z>/(||e|| ||n_z||) (area/IPW weighted, e = predicted - CFD Cp), "
           "Delta C_L = -S rho_L ||e|| ||n_z|| (counts = 1000 x coefficient, S = A_total/A_ref). Negative rho_L = over-predicted lift. E1 rows are the held-in test set (in-distribution reference); M2 on E1 uses seeds 0, 1 (recomputed from best.pt, reproduces the saved per-car rel L2) and 2.", "", md(rows), ""]
    g = A[(A.fold == "E2C") & (A.model == "B3")]
    if len(g):
        pos = float((g.rho_L > 0).mean())
        txt += ["## Pre-stated check: RegDGCNN (B3) on E2C (Fastback held out, bias about -171 counts) should have rho_L > 0 for most cars", "",
                f"mean rho_L {fm(ci(g.rho_L))}; share of cars with rho_L > 0: {pos:.2f}; bias C_L,p {fm(ci(g.dL), 1)} counts. Prediction {'CONFIRMED' if ci(g.rho_L)[1] > 0 and pos > 0.5 else 'NOT confirmed'}.", ""]
    big = A[A.groupby(["fold", "model"]).dL.transform("mean").abs() >= 100]
    big = big[big.fold.isin(FOLDS)]
    if len(big):
        agree = big.groupby(["fold", "model"]).apply(lambda g: (np.sign(g.rho_L) == -np.sign(g.dL.mean())).mean(), include_groups=False)
        txt += ["## Sign check over all (fold, model) with |bias| >= 100 counts: share of cars whose rho_L has the sign that produces the bias (-sign of bias)", "", agree.round(2).to_frame("share").to_markdown(), ""]
    (OUT / f"T1_family_alignment{sfx}.md").write_text("\n".join(txt), encoding="utf-8")
    log("T1 written")


# ------------------------------------------------------------------ T2
def counterfactual(T, A, ref_fold):
    rows, shap = [], []
    for m in sorted(T.model.unique()):
        a, b = T[(T.fold == "E2A") & (T.model == m)], T[(T.fold == ref_fold) & (T.model == m)]
        seeds = sorted(set(a.seed) & set(b.seed))
        if not seeds:
            continue
        base, cfa, cfb, K, rA, eA, rB, eB = [], [], [], [], [], [], [], []
        for s in seeds:
            x, y = a[a.seed == s].set_index("id").sort_index(), b[b.seed == s]
            med_e, rho_m = y.e_norm.median(), y.rho_L.mean()
            base.append(x.dL.values)
            cfa.append((-1000 * x.S * x.rho_L * med_e * x.n_L).values)
            cfb.append((-1000 * x.S * rho_m * x.e_norm * x.n_L).values)
            K.append(float((-1000 * x.S * x.n_L).mean()))
            rA.append(x.rho_L.mean())
            eA.append(x.e_norm.median())
            rB.append(rho_m)
            eB.append(med_e)
        base, cfa, cfb = (np.mean(v, axis=0) for v in (base, cfa, cfb))
        K, rA, eA, rB, eB = (float(np.mean(v)) for v in (K, rA, eA, rB, eB))
        b0, ba, bb = ci(base), ci(cfa), ci(cfb)
        rows.append({"model": m, "seeds": len(seeds), "bias E2A (counts)": fm(b0, 1), f"(a) ||e|| -> median {ref_fold}": fm(ba, 1), "explained (a)": f"{100 * (1 - ba[0] / b0[0]):.0f}%",
                     f"(b) rho_L -> mean {ref_fold}": fm(bb, 1), "explained (b)": f"{100 * (1 - bb[0] / b0[0]):.0f}%"})
        bAA, bBB = K * rA * eA, K * rB * eB
        d = bAA - bBB
        phi_r, phi_e = 0.5 * K * (rA - rB) * (eA + eB), 0.5 * K * (eA - eB) * (rA + rB)
        shap.append({"model": m, "actual bias E2A": f"{b0[0]:+.1f}", "scalar model b(A,A)": f"{bAA:+.1f}", f"scalar model b({ref_fold},{ref_fold}) on E2A cars": f"{bBB:+.1f}", "difference to explain": f"{d:+.1f}",
                     "Shapley rho_L": f"{phi_r:+.1f} ({100 * phi_r / d:.0f}%)" if abs(d) > 1e-9 else "-", "Shapley ||e||": f"{phi_e:+.1f} ({100 * phi_e / d:.0f}%)" if abs(d) > 1e-9 else "-"})
    return rows, shap


def t2(T, A):
    fig, axs = plt.subplots(2, len(FOLDS), figsize=(7.1, 3.6), sharex=True)
    for r, m in enumerate(("M2", "B3")):
        for c, f in enumerate(FOLDS):
            g = A[(A.fold == f) & (A.model == m)]
            if not len(g):
                axs[r, c].text(0.5, 0.5, "no file", ha="center", transform=axs[r, c].transAxes)
                continue
            axs[r, c].hist(g.rho_L, bins=np.linspace(-0.7, 0.7, 29), color=COL[m], alpha=0.85)
            axs[r, c].axvline(0, color="k", lw=0.7)
            axs[r, c].axvline(g.rho_L.mean(), color="k", lw=0.9, ls="--")
            axs[r, c].set_title(f"{m} {f}: mean {g.rho_L.mean():+.3f}, {100 * (g.rho_L < 0).mean():.0f}% of cars < 0", fontsize=7)
    for ax in axs[1]:
        ax.set_xlabel("rho_L (per car, seed-mean)")
    fig.tight_layout()
    fig.savefig(OUT / f"T2_rho_hist{sfx}.png", dpi=200)
    plt.close(fig)
    txt = ["# T2: sign distribution, counterfactuals and C_D,p bias" + (" (validation cars)" if role == "val" else ""), ""]
    fr = A.groupby(["fold", "model"]).apply(lambda g: pd.Series({"n": len(g), "share rho_L<0": (g.rho_L < 0).mean(), "share rho_D<0": (g.rho_D < 0).mean()}), include_groups=False).round(2).reset_index()
    txt += ["## Share of cars with rho_L < 0 (seed-mean rho per car)", "", md(fr.to_dict("records")), ""]
    cd = []
    for (f, m), g in A.groupby(["fold", "model"]):
        for lab, gg in subsets(f, g):
            cd.append({"fold": lab, "model": m, "n": len(gg), "bias C_D,p (counts) [CI]": fm(ci(gg.dD), 1), "bias C_L,p (counts) [CI]": fm(ci(gg.dL), 1)})
    txt += ["## Bias of C_D,p and C_L,p by fold (all models with files)", "", md(cd), ""]
    if role == "test":
        for ref in ("E2B", "E2C"):
            rows, shap = counterfactual(T, A, ref)
            txt += [f"## Counterfactual on E2A cars with reference fold {ref}", "",
                    f"Per seed and car: (a) keep the car's rho_L, S and ||n_z||, replace ||e|| by the median ||e|| of the {ref} cars of the same model and seed; (b) keep ||e||, replace rho_L by the mean signed rho_L of the {ref} cars; "
                    "then mean over seeds per car, mean over the 250 cars, bootstrap CI over cars (the reference scalars are treated as fixed). Explained = 1 - bias after / bias before.", "", md(rows), "",
                    f"### Two-factor Shapley decomposition of b(A,A) - b({ref},{ref}) (scalar model b = K * rho * ||e||, K = mean over E2A cars of -1000 S n_L; rho = mean signed rho_L, ||e|| = median over cars; "
                    "the scalar model ignores the per-car covariance of rho and ||e||, compare 'actual bias' with 'scalar model')", "", md(shap), ""]
    (OUT / f"T2_counterfactual{sfx}.md").write_text("\n".join(txt), encoding="utf-8")
    log("T2 written")


# ------------------------------------------------------------------ T3
def t3(T, A):
    rows, dec = [], []
    for (f, m), g in A.groupby(["fold", "model"]):
        if f == "E1":
            continue
        bL, bD = g.dL.values, g.dD.values
        okL, okD = abs(bL.mean()) >= 20, abs(bD.mean()) >= 20
        for r in REGIONS:
            cL, cD = g[f"reg_dL_{r}"].values, g[f"reg_dD_{r}"].values
            a, pl, d_, pd_, rho = ci(cL), ci_ratio(cL, bL) if okL else (np.nan,) * 3, ci(cD), ci_ratio(cD, bD) if okD else (np.nan,) * 3, ci(g[f"reg_rhoL_{r}"])
            rows.append({"fold": f, "model": m, "n": len(g), "region": r, "area_share": g[f"reg_area_{r}"].mean(), "dCL_mean": a[0], "dCL_lo": a[1], "dCL_hi": a[2], "pct_bias_L": pl[0], "pct_lo": pl[1], "pct_hi": pl[2],
                         "dCD_mean": d_[0], "dCD_lo": d_[1], "dCD_hi": d_[2], "pct_bias_D": pd_[0], "rho_L_region_mean": rho[0], "rho_L_region_lo": rho[1], "rho_L_region_hi": rho[2]})
        if f == "E2A" and okL:
            grp = {"roof_rear": ["roof_rear"], "underbody + base": ["underbody", "base"], "roof_front": ["roof_front"], "front": ["front"], "sides": ["sides"], "other": ["other"]}
            row = {"model": m, "bias E2A (counts)": f"{bL.mean():+.1f}"}
            for k, rs in grp.items():
                row[k] = fm(ci_ratio(sum(g[f"reg_dL_{x}"].values for x in rs), bL), 0, False) + "%"
            dec.append(row)
    d = pd.DataFrame(rows)
    d.round(4).to_csv(OUT / f"T3_regional{sfx}.csv", index=False)
    fig, axs = plt.subplots(1, len(FOLDS), figsize=(7.1, 3.0), sharey=True)
    models = [m for m in ("M2", "B1", "B2_300ep", "B3") if m in set(d.model)]
    for ax, f in zip(axs, FOLDS):
        for j, m in enumerate(models):
            x = d[(d.fold == f) & (d.model == m)].set_index("region").reindex(REGIONS)
            ax.bar(np.arange(len(REGIONS)) + (j - (len(models) - 1) / 2) * 0.2, x.dCL_mean, 0.2, color=COL[m], label=m, yerr=[x.dCL_mean - x.dCL_lo, x.dCL_hi - x.dCL_mean], error_kw={"lw": 0.6})
        ax.set_xticks(range(len(REGIONS)))
        ax.set_xticklabels(REGIONS, rotation=55, ha="right", fontsize=6.5)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f, fontsize=8)
    axs[0].set_ylabel("contribution to C_L,p bias (counts)")
    axs[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"T3_regional_bar{sfx}.png", dpi=200)
    plt.close(fig)
    show = d[d.model.isin(models)][["fold", "model", "region", "area_share", "dCL_mean", "pct_bias_L", "dCD_mean", "rho_L_region_mean"]].copy()
    show["area_share"] = (100 * show.area_share).round(1)
    show = show.round({"dCL_mean": 1, "pct_bias_L": 0, "dCD_mean": 1, "rho_L_region_mean": 3})
    txt = ["# T3: regional decomposition of the C_L,p and C_D,p bias" + (" (validation cars)" if role == "val" else ""), "",
           "Regions partition the surface by rule (priority base > underbody > roof_rear > roof_front > front > sides > other; x_rel from each car's own pool; definitions in scripts/diagnosis/force_alignment.py); the regional contributions sum to the car's Delta C. "
           "pct_bias = contribution / total bias (percent; empty when |bias| < 20 counts). Full table with CIs: T3_regional.csv.", ""]
    if dec:
        txt += ["## Share of the E2A C_L,p bias by region group (percent of bias [bootstrap CI]); input to the decision tree (rear roof vs underbody + base)", "", md(dec), ""]
    txt += ["## Per region (counts; M2, B1, B2 300 ep, B3)", "", md(show.fillna("").to_dict("records")), ""]
    (OUT / f"T3_regional{sfx}.md").write_text("\n".join(txt), encoding="utf-8")
    log("T3 written")


# ------------------------------------------------------------------ T4
def cfd_table():
    f = OUT / "cfd_coefficients.csv"
    if f.exists():
        return pd.read_csv(f).set_index("id")
    ids = sorted(set().union(*(set(load_split("E1")[k]) for k in ("train", "val", "test"))))
    rows = []
    for sid in ids:
        _, nrm, cp, area = load_pool(sid)
        rows.append({"id": sid, **cfd_coefficients(cp, nrm, area, sid)})
    d = pd.DataFrame(rows)
    d.round(3).to_csv(f, index=False)
    return d.set_index("id")


def t4(A):
    C = cfd_table()
    rows, gro, pts = [], [], []
    for f in FOLDS:
        s = load_split(f)
        tr, te = C.loc[s["train"]], C.loc[s["test"]]
        rng = np.random.default_rng(0)
        G = {}
        for q in ("CL", "CD"):
            gb = [tr[q].values[rng.integers(0, len(tr), len(tr))].mean() - te[q].values[rng.integers(0, len(te), len(te))].mean() for _ in range(B)]
            G[q] = (tr[q].mean() - te[q].mean(), np.percentile(gb, 2.5), np.percentile(gb, 97.5))
        gro.append({"fold": f, "n train": len(tr), "n test": len(te), "train mean C_L,p": f"{tr.CL.mean():+.1f}", "test mean C_L,p": f"{te.CL.mean():+.1f}", "G_L = train - test [CI]": fm(G["CL"], 1),
                     "G_D = train - test [CI]": fm(G["CD"], 1)})
        for m in sorted(A.model.unique()):
            g = A[(A.fold == f) & (A.model == m)].set_index("id")
            if not len(g):
                continue
            ref = te.loc[g.index]
            out = {"fold": f, "model": m, "n test": len(g)}
            for q, dcol in (("CL", "dL"), ("CD", "dD")):
                pred = ref[q].values + g[dcol].values
                sl, ic = np.polyfit(ref[q].values, pred, 1)
                idx = rng.integers(0, len(g), (B, len(g)))
                sl_b = [np.polyfit(ref[q].values[i], pred[i], 1)[0] for i in idx[:500]]
                out[f"{q} slope [CI]"] = f"{sl:.2f} [{np.percentile(sl_b, 2.5):.2f}; {np.percentile(sl_b, 97.5):.2f}]"
                out[f"{q} intercept"] = f"{ic:+.1f}"
                dev = ci(pred - tr[q].mean())
                out[f"{q} mean(pred) - train mean [CI]"] = fm(dev, 1)
                out[f"{q} bias"] = f"{g[dcol].mean():+.1f}"
                pts.append({"fold": f, "model": m, "q": q, "G": G[q][0], "bias": g[dcol].mean(), "slope": sl})
            rows.append(out)
    pd.DataFrame(rows).to_csv(OUT / "T4_shrinkage.csv", index=False)
    P = pd.DataFrame(pts)
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.2))
    mk = {"E2A": "o", "E2B": "s", "E2C": "^"}
    for ax, q, lab in zip(axs, ("CL", "CD"), ("C_L,p", "C_D,p")):
        for _, r in P[P.q == q].iterrows():
            ax.scatter(r.G, r.bias, color=COL[r.model], marker=mk[r.fold], s=26, edgecolors="white", linewidths=0.4)
        lim = [min(P[P.q == q][["G", "bias"]].min()) - 10, max(P[P.q == q][["G", "bias"]].max()) + 10]
        ax.plot(lim, lim, "k--", lw=0.7)
        ax.set_xlabel(f"G = train mean - test mean of CFD {lab} (counts)", fontsize=7)
        ax.set_ylabel(f"bias of {lab} (counts)")
        ax.set_title(f"{lab}: bias = G if the model predicts the train-class mean", fontsize=7)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=MK[m], color="none", markerfacecolor=COL[m], ms=6, label=m) for m in sorted(A.model.unique())] + [Line2D([], [], marker=mk[f], color="none", markerfacecolor="#888", ms=6, label=f) for f in FOLDS]
    axs[0].legend(handles=h, fontsize=6, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "T4_bias_vs_G.png", dpi=200)
    plt.close(fig)
    txt = ["# T4: class-prior gap G and shrinkage", "",
           "G = mean CFD coefficient of the fold's training cars minus mean over its test cars (counts). If a model output the training-class mean, bias = G. Shrinkage: regression of the predicted coefficient on the CFD coefficient over the test cars of the fold "
           "(slope 1 = full per-car information, 0 = constant prediction; seed-mean prediction per car, slope CI from 500 bootstrap resamples). 'mean(pred) - train mean' = bias - G.", "", "## G per fold (CFD labels)", "", md(gro), "", "## Shrinkage and distance to the class-prior prediction", "", md(rows), ""]
    (OUT / "T4_shrinkage.md").write_text("\n".join(txt), encoding="utf-8")
    log("T4 written")


# ------------------------------------------------------------------ T5
def theta_rear(sid):
    xyz, nrm, _, area = load_pool(sid)
    w = pool_weights(np.asarray(area, np.float64), sid)
    x, y, z = (xyz[:, k].astype(np.float64) for k in range(3))
    L = x.max() - x.min()
    xr, zr = (x - x.min()) / L, (z - z.min()) / (z.max() - z.min())
    nx, nz = nrm[:, 0].astype(np.float64), nrm[:, 2].astype(np.float64)
    roof = (nz > 0.95) & (zr > 0.8)
    band = (xr > 0.93) & (xr < 0.97)
    up = (xr > 0.75) & (zr > 0.55)
    return {"base_area_frac": float(w[nx > 0.8].sum() / w.sum()),
            "flat_roof_len": float((np.percentile(x[roof], 98) - np.percentile(x[roof], 2)) / L) if roof.sum() > 50 else np.nan,
            "rear_upper_slope_deg": float(np.degrees(np.arccos(np.clip(nz[up], -1, 1))) @ w[up] / w[up].sum()) if up.sum() else np.nan,
            "rear_taper": float((y[band].max() - y[band].min()) / (y.max() - y.min())) if band.sum() > 10 else np.nan}


def cv_r2(X, y, reps=10):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    r = []
    for k in range(reps):
        pr = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))), X, y, cv=KFold(10, shuffle=True, random_state=k))
        r.append(1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(r)), float(np.min(r)), float(np.max(r))


def t5(A):
    g = A[(A.fold == "E2A") & (A.model == "M2")].set_index("id").sort_index()
    tot = g.dL.values
    txt = ["# T5: bridge to C2 (shared vs car-specific part of the M2 lift bias on E2A)", "",
           f"M2, E2A, n = {len(g)} Estateback cars, seed-mean per car (3 seeds). Delta C_L,p = sum over regions of the regional contributions.", ""]
    m2, v = tot.mean() ** 2, tot.var(ddof=1)
    txt += [f"Total: mean {tot.mean():+.1f} counts, SD across cars {np.sqrt(v):.1f}. Share of the second moment E[Delta C^2] carried by the common mean: {100 * m2 / (m2 + v):.1f}% (car-specific variance: {100 * v / (m2 + v):.1f}%).", ""]
    rows = []
    for r in REGIONS:
        c = g[f"reg_dL_{r}"].values
        me = g[f"reg_me_{r}"].values
        okm = ~np.isnan(me)
        rows.append({"region": r, "mean contribution (counts)": f"{c.mean():+.1f}", "SD across cars": f"{c.std(ddof=1):.1f}", "common-mean share of E[c^2]": f"{100 * c.mean() ** 2 / (c.mean() ** 2 + c.var(ddof=1)):.0f}%",
                     "share of Var(Delta C) = Cov(c_r, total)/Var(total)": f"{100 * np.cov(c, tot)[0, 1] / v:.0f}%", "mean weighted error in region (Cp)": f"{me[okm].mean():+.3f}", "SD of that error across cars": f"{me[okm].std(ddof=1):.3f}"})
    txt += ["## Per region", "", md(rows), ""]
    th = pd.DataFrame([theta_rear(i) for i in g.index], index=g.index)
    nn = int(th.isna().any(axis=1).sum())
    th = th.fillna(th.median())
    th.round(5).to_csv(OUT / "T5_theta_rear.csv")
    X0 = th.values
    Xp = g[[f"reg_predL_{r}" for r in REGIONS]].values
    res = [("theta_rear (4 geometric features)", cv_r2(X0, tot)), ("theta_rear + M2 predicted regional lift integrals (7)", cv_r2(np.hstack([X0, Xp]), tot)), ("M2 predicted regional lift integrals only (7)", cv_r2(Xp, tot))]
    cor = th.apply(lambda c: np.corrcoef(c, tot)[0, 1])
    txt += ["## theta_rear (unlabelled geometry of the car, from the cached surface pool) and ridge cross-validation", "",
            "Definitions (x_rel = (x - x_min)/L, z_rel = (z - z_min)/(z_max - z_min) per car; w = IPW weights): base_area_frac = weighted share of surface with n_x > 0.8; flat_roof_len = (98th - 2nd percentile of x over points with n_z > 0.95 and z_rel > 0.8) / L; "
            "rear_upper_slope_deg = weighted mean of arccos(n_z) over points with x_rel > 0.75 and z_rel > 0.55 (0 = flat roof, 90 = vertical); rear_taper = y-extent of points with 0.93 < x_rel < 0.97 divided by the car's y-extent. "
            f"Cars with a missing feature (median imputed): {nn}.", "",
            md([{"feature": k, "mean": f"{th[k].mean():.3f}", "SD": f"{th[k].std():.3f}", "corr with Delta C_L,p": f"{cor[k]:+.2f}"} for k in th.columns]), "",
            "Ridge (RidgeCV, alphas 1e-3..1e3, standardised features) cross-validated by car: 10-fold, repeated 10 times with different shuffles; R2 = 1 - SSE/SST of out-of-fold predictions of the car's Delta C_L,p (mean [min; max] over repeats). "
            "R2 is about the car-to-car variation of the bias, not about its mean. 'Predicted regional lift integrals' use the model's predictions only (no labels); regional integrals of the error would reproduce Delta C_L,p exactly and are not used.", "",
            md([{"features": k, "CV R2 [min; max]": f"{v_[0]:+.3f} [{v_[1]:+.3f}; {v_[2]:+.3f}]"} for k, v_ in res]), ""]
    (OUT / "T5_c2_bridge.md").write_text("\n".join(txt), encoding="utf-8")
    log("T5 written")


def main():
    global role, sfx
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="test", choices=["test", "val"])
    ap.add_argument("--steps", default="terms,t1,t2,t3,t4,t5")
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    role, sfx = a.role, "" if a.role == "test" else "_val"
    steps = a.steps.split(",")
    log(f"command: python scripts/diagnosis/force_alignment_terms.py --role {role} --steps {a.steps}" + (" --recompute" if a.recompute else ""))
    T = build_terms(a.recompute)
    A = aggregate(T)
    for name, fn, args in (("t1", t1, (T, A)), ("t2", t2, (T, A)), ("t3", t3, (T, A)), ("t4", t4, (A,)), ("t5", t5, (A,))):
        if name in steps and (role == "test" or name in ("t1", "t2", "t3")):
            t0 = time.time()
            fn(*args)
            log(f"{name} took {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
