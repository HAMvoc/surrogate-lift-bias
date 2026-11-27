"""Two-stage correction study analysis of T1 (two-stage C2, OOF stage B), T2 (baselines) and T3 (label-budget curve). Inputs: results/two_stage/T1/raw/*.npz (from the Modal volume), results/two_stage/T2/frozen_draws.csv,
retrain runs in results/two_stage/T2/raw/results/E2A_M2_R2_s0_cal<k>d<d>/ (optional). Output: results/two_stage/T1/T1_report.md, T1_per_draw.csv, T2/T2_report.md, T3/T3_curve.{png,pdf,csv}.
Usage: python scripts/correction/two_stage_analysis.py"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

P3 = ROOT / "results" / "two_stage"
ALPHA, B = 0.1, 2000
METHODS = {"ft_B_OOF (final A)": "B_full", "ft_B_OOF (fold-avg A)": "B_avg", "ft noB (final A)": "noB_full", "ft noB (fold-avg A)": "noB_avg"}
rng = np.random.default_rng(0)


def cvplus(eps, R, fold_of, k):
    lo_i, hi_i = max(1, math.floor(ALPHA * (k + 1))), min(k, math.ceil((1 - ALPHA) * (k + 1)))
    e_i = eps[fold_of]
    lo = np.sort(e_i - R[:, None], axis=0)[lo_i - 1]
    hi = np.sort(e_i + R[:, None], axis=0)[hi_i - 1]
    return float(np.mean((lo <= 0) & (hi >= 0))), float(np.mean((hi - lo) / 2))


def spread(e):
    return float(np.sqrt(((e - e.mean()) ** 2).mean()))


def agg(v):
    v = np.asarray(v, float)
    return f"{v.mean():.4g} [{np.percentile(v, 5):.4g}; {np.percentile(v, 95):.4g}] sd {v.std(ddof=1) if len(v) > 1 else float('nan'):.3g}"


def main():
    (P3 / "T1").mkdir(parents=True, exist_ok=True); (P3 / "T3").mkdir(exist_ok=True)
    perm = [str(i) for i in np.random.default_rng(0).permutation(sorted(load_split("E2A")["test"]))]
    ev_ids = perm[125:]
    cfd = pd.read_csv(ROOT / "results" / "force_alignment" / "cfd_coefficients.csv").set_index("id").loc[ev_ids, "CL"].values
    rows, cv, carerr = [], [], {}
    for f in sorted((P3 / "T1" / "raw").glob("*.npz")):
        v, s, k, d = f.stem.split("_")
        z = np.load(f)
        rec = {"variant": v, "seed": int(s[1:]), "k": int(k[1:]), "draw": int(d[1:])}
        for name, key in METHODS.items():
            rows.append({**rec, "method": name, "arr": z[key]})
        cov, hw = cvplus(z["eps"], z["cv_R"], z["cv_fold_of"], rec["k"])
        cv.append({**rec, "coverage": cov, "half": hw, "lam": float(z["lam_all"]), "sel_all": int(z["sel_all"]), "cons": float(z["cons"])})
    if not rows:
        print("no T1 files"); return
    R = pd.DataFrame(rows)
    C = pd.DataFrame(cv)
    out = []
    for (v, k, d, m), g in R.groupby(["variant", "k", "draw", "method"]):
        arr = np.stack(g.arr.values)  # (S, 125, 3)
        eL, eD = arr[..., 0].mean(0), arr[..., 1].mean(0)
        out.append({"variant": v, "k": k, "draw": d, "method": m, "seeds": len(g), "bias_L": eL.mean(), "abs_bias_L": abs(eL.mean()), "bias_D": eD.mean(), "abs_bias_D": abs(eD.mean()), "rmse_L": spread(eL), "rel_l2": arr[..., 2].mean(),
                    "spearman": spearmanr(cfd + eL, cfd)[0], "slope": np.polyfit(cfd, cfd + eL, 1)[0]})
        carerr[(v, k, d, m)] = (eL, arr[..., 2].mean(0))
    D = pd.DataFrame(out)
    cvd = C.groupby(["variant", "k", "draw"])[["coverage", "half", "lam", "sel_all", "cons"]].mean().reset_index()
    D = D.merge(cvd[["variant", "k", "draw", "coverage", "half"]], on=["variant", "k", "draw"], how="left")
    D.loc[D.method != "ft_B_OOF (final A)", ["coverage", "half"]] = np.nan
    D.round(5).to_csv(P3 / "T1" / "T1_per_draw.csv", index=False)
    cvd.round(5).to_csv(P3 / "T1" / "T1_cvplus_lambda.csv", index=False)
    # frozen and retrain baselines
    Fz = pd.read_csv(P3 / "T2" / "frozen_draws.csv")
    Fz = Fz[Fz.draw < 10].rename(columns={"method": "method0"})
    Fz["method"] = "frozen + " + Fz.method0
    retr = []
    rdir = P3 / "T2" / "raw" / "results"
    if rdir.exists():
        sys.path.insert(0, str(ROOT))
        from scripts.correction.calibration import build_stats, metrics
        from scripts.diagnosis.regions import labeler
        import re
        for r in sorted(rdir.glob("E2A_M2_R2_s0_cal*d*")):
            mm = re.fullmatch(r"E2A_M2_R2_s0_cal(\d+)d(\d+)", r.name)
            if not mm:
                continue
            k, d = int(mm.group(1)), int(mm.group(2))
            ids = list(pd.read_csv(r / "test_per_car.csv").id)
            pr = np.load(r / "test_pred_fp16.npy", mmap_mode="r")
            pos = {i: j for j, i in enumerate(ids)}
            z = build_stats(ev_ids, lambda _s, sid: pr[pos[sid]], 1, labeler, 8)
            m = metrics(z, np.arange(len(ev_ids)), ("none", None))
            retr.append({"method": "retrain (1 seed)", "k": k, "draw": d, "bias_L": m["bias_L"], "abs_bias_L": m["abs_bias_L"], "rmse_L": m["rmse_L"], "rel_l2": m["rel_l2"], "abs_bias_D": m["abs_bias_D"], "spearman": m["spearman"]})
    Rt = pd.DataFrame(retr)
    L = ["# Two-stage correction study, task T1 and T2: two-stage C2 with OOF stage B, baselines", "",
         "Evaluation: 125 Estateback cars (perm[125:]) not used for fitting or step selection. Per draw: seed-mean per-car error -> metrics; 'mean [5%; 95%] sd' are over draws (10 for T1 and frozen, 5 for retrain), the second CI tier (over cars) is in the last table. Seeds/draws completed per cell are in `T1_per_draw.csv` (column seeds). "
         "ft_B_OOF = stage A (fine-tune) + region_const_intmatch fitted on out-of-fold residuals; 'final A' = fine-tune on all k cars; 'fold-avg A' = mean of the five fold models. Environment: Modal T4 for all T1/T2 retrain runs; the E2A M2 checkpoints and the frozen predictions come from Kaggle runs (see T6 for the environment control).", ""]
    for k in (5, 10, 25):
        tab = []
        for v in ("dec", "full"):
            for m in METHODS:
                g = D[(D.variant == v) & (D.k == k) & (D.method == m)]
                if g.empty:
                    continue
                tab.append({"method": f"{v}: {m}", "draws": f"{len(g)} (seeds {int(g.seeds.min())}-{int(g.seeds.max())})", "|bias| C_L,p": agg(g.abs_bias_L), "bias_L signed": f"{g.bias_L.mean():+.1f}", "|bias| C_D,p": f"{g.abs_bias_D.mean():.1f}", "Rel L2": agg(g.rel_l2),
                            "RMSE": f"{g.rmse_L.mean():.1f}", "Spearman": f"{g.spearman.mean():.2f}", "slope": f"{g.slope.mean():.2f}"})
        for m, g in Fz[Fz.k == k].groupby("method"):
            tab.append({"method": m, "draws": f"{len(g)} (3 seeds)", "|bias| C_L,p": agg(g.abs_bias_L), "bias_L signed": f"{g.bias_L.mean():+.1f}", "|bias| C_D,p": f"{g.abs_bias_D.mean():.1f}", "Rel L2": agg(g.rel_l2), "RMSE": f"{g.rmse_L.mean():.1f}",
                        "Spearman": f"{g.spearman.mean():.2f}", "slope": f"{g.slope.mean():.2f}"})
        if len(Rt):
            g = Rt[Rt.k == k]
            if len(g):
                tab.append({"method": "retrain (150 epochs, 1 seed)", "draws": f"{len(g)} (1 seed)", "|bias| C_L,p": agg(g.abs_bias_L), "bias_L signed": f"{g.bias_L.mean():+.1f}", "|bias| C_D,p": f"{g.abs_bias_D.mean():.1f}", "Rel L2": agg(g.rel_l2),
                            "RMSE": f"{g.rmse_L.mean():.1f}", "Spearman": f"{g.spearman.mean():.2f}", "slope": "-"})
        L += [f"## k = {k}", "", md(tab), ""]
    # coverage distribution
    ct = []
    for v in ("dec", "full"):
        for k in (5, 10, 25):
            g = cvd[(cvd.variant == v) & (cvd.k == k)]
            if len(g):
                ct.append({"variant": v, "k": k, "draws": len(g), "coverage mean [5%; 95%]": agg(g.coverage), "p95 - p5": f"{np.percentile(g.coverage, 95) - np.percentile(g.coverage, 5):.3f}", "draws with coverage in [0.85, 0.95]": f"{int(((g.coverage >= .85) & (g.coverage <= .95)).sum())}/{len(g)}",
                           "median coverage": f"{g.coverage.median():.3f}", "median half-width": f"{g.half.median():.0f}", "lambda chosen (mean)": f"{g.lam.mean():.2f}", "max field-coefficient consistency gap": f"{g.cons.max():.1e}"})
    L += ["## CV+ coverage (nested stage B, K = 5; theoretical guarantee 1 - 2 alpha = 80%) per draw (mean of the seeds of the draw)", "", md(ct), ""]
    # thresholds
    L += ["## Registered criteria (PREREG), k = 10", ""]
    pre = []
    for v in ("dec", "full"):
        g = D[(D.variant == v) & (D.k == 10) & (D.method == "ft_B_OOF (final A)")].set_index("draw")
        if g.empty:
            continue
        c = cvd[(cvd.variant == v) & (cvd.k == 10)]
        res = {}
        for nm, comp in (("frozen + region_const", Fz[(Fz.k == 10) & (Fz.method0 == "region_const")]), ("retrain", Rt[Rt.k == 10] if len(Rt) else pd.DataFrame())):
            if len(comp) == 0:
                res[nm] = "no data yet"; continue
            comp = comp.set_index("draw")
            dd = sorted(set(g.index) & set(comp.index))
            if len(dd) < 3:
                res[nm] = f"only {len(dd)} common draws"; continue
            a_b, c_b, a_r, c_r = g.loc[dd, "abs_bias_L"].values, comp.loc[dd, "abs_bias_L"].values, g.loc[dd, "rel_l2"].values, comp.loc[dd, "rel_l2"].values
            ok_mean = a_b.mean() <= c_b.mean() and a_r.mean() <= c_r.mean()
            p_b = wilcoxon(a_b, c_b, alternative="less").pvalue if np.any(a_b != c_b) else 1.0
            p_r = wilcoxon(a_r, c_r, alternative="less").pvalue if np.any(a_r != c_r) else 1.0
            res[nm] = f"means {'<=' if ok_mean else 'NOT <='} comparator on both (|bias| {a_b.mean():.1f} vs {c_b.mean():.1f}; Rel L2 {a_r.mean():.4f} vs {c_r.mean():.4f}); Wilcoxon one-sided p: |bias| {p_b:.3f}, Rel L2 {p_r:.3f}, n = {len(dd)}; criterion {'MET' if ok_mean and min(p_b, p_r) < 0.05 else 'NOT met'}"
        pre.append({"variant": v, "primary: vs frozen + region_const": res["frozen + region_const"], "primary: vs retrain": res["retrain"]})
        sec = {"|bias| C_L,p <= 28": g.abs_bias_L.mean() <= 28, "|bias| C_D,p <= 19.0": g.abs_bias_D.mean() <= 19.0, "Rel L2 <= 0.175": g.rel_l2.mean() <= 0.175, "median coverage in [0.85, 0.95]": 0.85 <= c.coverage.median() <= 0.95, "median half-width < 140": c.half.median() < 140}
        pre.append({"variant": f"{v} secondary", "primary: vs frozen + region_const": "; ".join(f"{kk}: {'pass' if vv else 'FAIL'}" for kk, vv in sec.items()) + f" (|bias| {g.abs_bias_L.mean():.1f}, |bias_D| {g.abs_bias_D.mean():.1f}, Rel L2 {g.rel_l2.mean():.4f}, median coverage {c.coverage.median():.3f}, median half-width {c.half.median():.0f})", "primary: vs retrain": ""})
    L += [md(pre), ""]
    for v in ("dec", "full"):
        g = D[(D.variant == v) & (D.k == 10) & (D.method == "ft_B_OOF (final A)")]
        c = cvd[(cvd.variant == v) & (cvd.k == 10)]
        if len(g):
            L.append(f"Predictions ({v}): (1) Rel L2 < 0.1765 (Decomposition and first corrections study `dec` ft_B): mean {g.rel_l2.mean():.4f} -> {'TRUE' if g.rel_l2.mean() < 0.1765 else 'FALSE'}" + (" (the registered prediction concerns `dec`)" if v == "full" else "") + f"; (2) coverage spread p95 - p5 = {np.percentile(c.coverage, 95) - np.percentile(c.coverage, 5):.3f} > 0.15 -> {'TRUE' if np.percentile(c.coverage, 95) - np.percentile(c.coverage, 5) > 0.15 else 'FALSE'}.")
    L.append("")
    # car-level CI of the draw- and seed-averaged error
    ci = []
    for v in ("dec", "full"):
        for m in METHODS:
            for k in (5, 10, 25):
                es = [carerr[(v, k, d, m)] for d in range(10) if (v, k, d, m) in carerr]
                if not es:
                    continue
                eL = np.mean([e[0] for e in es], axis=0)
                rl = np.mean([e[1] for e in es], axis=0)
                bi = rng.integers(0, len(eL), (B, len(eL)))
                ci.append({"variant": v, "method": m, "k": k, "draws": len(es), "bias_L [CI over cars]": f"{eL.mean():+.1f} [{np.percentile(eL[bi].mean(1), 2.5):+.1f}; {np.percentile(eL[bi].mean(1), 97.5):+.1f}]",
                           "Rel L2 [CI over cars]": f"{rl.mean():.4f} [{np.percentile(rl[bi].mean(1), 2.5):.4f}; {np.percentile(rl[bi].mean(1), 97.5):.4f}]"})
    L += ["## CI over cars (bootstrap, B = 2000) of the per-car error averaged over draws and seeds", "", md(ci), ""]
    (P3 / "T1" / "T1_report.md").write_text("\n".join(L), encoding="utf-8")
    # T2: Pareto order and the retrain + OOF stage B case
    P = ["# Two-stage correction study, task T2: Pareto order of all methods (mean over draws of |bias| C_L,p and Rel L2; lower is better on both)", "",
         "A method is Pareto-optimal at a given k if no other method is at least as good on both and strictly better on one. Means over the draws available for each method (T1 and frozen: 10; retrain: 5, a different set of draws, so the retrain row is not paired with the others).", ""]
    for k in (5, 10, 25):
        cand = []
        for v in ("dec", "full"):
            for m in ("ft_B_OOF (final A)", "ft_B_OOF (fold-avg A)", "ft noB (final A)"):
                g = D[(D.variant == v) & (D.k == k) & (D.method == m)]
                if len(g):
                    cand.append((f"{v}: {m}", g.abs_bias_L.mean(), g.rel_l2.mean(), len(g)))
        for m, g in Fz[Fz.k == k].groupby("method"):
            cand.append((m, g.abs_bias_L.mean(), g.rel_l2.mean(), len(g)))
        if len(Rt) and len(Rt[Rt.k == k]):
            g = Rt[Rt.k == k]
            cand.append(("retrain (1 seed)", g.abs_bias_L.mean(), g.rel_l2.mean(), len(g)))
        tab = []
        for n, b, r, nd in sorted(cand, key=lambda x: x[2]):
            dom = any((b2 <= b and r2 <= r) and (b2 < b or r2 < r) for n2, b2, r2, _ in cand if n2 != n)
            tab.append({"method": n, "draws": nd, "|bias| C_L,p": f"{b:.1f}", "Rel L2": f"{r:.4f}", "Pareto-optimal": "no" if dom else "yes"})
        P += [f"## k = {k}", "", md(tab), ""]
    oofdirs = sorted(rdir.glob("E2A_M2_R2_s0_cal25d0f*")) if rdir.exists() else []
    if len(oofdirs) == 5 and len(Rt):
        from scripts.correction.calibration import build_stats as bs2, metrics as m2, corrected as co2
        from scripts.correction.two_stage_oof import fit_b
        oof_ids, preds = [], {}
        for r in oofdirs:
            ids = list(pd.read_csv(r / "test_per_car.csv").id)
            pr = np.load(r / "test_pred_fp16.npy", mmap_mode="r")
            for j, i in enumerate(ids):
                if i not in set(ev_ids):
                    oof_ids.append(i); preds[i] = pr[j]
        assert len(oof_ids) == 25, len(oof_ids)
        z = bs2(oof_ids, lambda _s, sid: preds[sid], 1, labeler, 8)
        lam, c = fit_b(z, np.arange(25))
        r0 = rdir / "E2A_M2_R2_s0_cal25d0"
        ids = list(pd.read_csv(r0 / "test_per_car.csv").id); pr = np.load(r0 / "test_pred_fp16.npy", mmap_mode="r"); pos = {i: j for j, i in enumerate(ids)}
        ze = bs2(ev_ids, lambda _s, sid: pr[pos[sid]], 1, labeler, 8)
        cases = {"retrain, no stage B": m2(ze, np.arange(125), ("none", None)), "retrain + stage B (OOF residuals of 5 fold runs)": m2(ze, np.arange(125), ("off", c[None]))}
        rows_ = [{"case (k = 25, draw 0)": n, "|bias| C_L,p": f"{v['abs_bias_L']:.1f}", "bias_L signed": f"{v['bias_L']:+.1f}", "Rel L2": f"{v['rel_l2']:.4f}", "RMSE": f"{v['rmse_L']:.1f}", "|bias| C_D,p": f"{v['abs_bias_D']:.1f}"} for n, v in cases.items()]
        for v in ("dec", "full"):
            g = D[(D.variant == v) & (D.k == 25) & (D.draw == 0) & (D.method == "ft_B_OOF (final A)")]
            if len(g):
                x = g.iloc[0]; rows_.append({"case (k = 25, draw 0)": f"T1 {v} ft_B_OOF (3 seeds)", "|bias| C_L,p": f"{x.abs_bias_L:.1f}", "bias_L signed": f"{x.bias_L:+.1f}", "Rel L2": f"{x.rel_l2:.4f}", "RMSE": f"{x.rmse_L:.1f}", "|bias| C_D,p": f"{x.abs_bias_D:.1f}"})
        P += ["## Retrain + stage B fitted on out-of-fold residuals (PREREG Amendment 1; one draw, one seed)", "", f"lambda chosen {lam}. Stage B offsets fitted on the 25 residuals of the five fold runs (each held-out fold predicted by the run that did not train on it), applied to the retrain run that used all 25 cars.", "", md(rows_), ""]
    (P3 / "T2" / "T2_report.md").write_text(chr(10).join(P), encoding="utf-8")
    # T3 curve
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.0))
    ks = (0, 1, 2, 3, 5, 10, 25)
    F50 = pd.read_csv(P3 / "T2" / "frozen_draws.csv")
    im = F50[F50.method == "intmatch_lam"].groupby("k")
    none = F50[F50.method == "none"].iloc[0]
    cols = ("abs_bias_L", "rel_l2", "rmse_L", "coverage")
    ttl = ("|bias| C_L,p (counts)", "Rel L2", "RMSE after bias removal (counts)", "jackknife+/CV+ coverage")
    for ax, c, t in zip(axs, cols, ttl):
        x = [k for k in ks if k in im.groups or k == 0]
        if c != "coverage":
            ym = [none[c]] + [im.get_group(k)[c].mean() for k in ks[1:]]
            lo = [none[c]] + [im.get_group(k)[c].quantile(.05) for k in ks[1:]]
            hi = [none[c]] + [im.get_group(k)[c].quantile(.95) for k in ks[1:]]
            ax.plot(ks, ym, "o-", c="C0", label="frozen + intmatch (50 draws)"); ax.fill_between(ks, lo, hi, color="C0", alpha=.15)
        else:
            ks2 = ks[2:]
            ax.plot(ks2, [im.get_group(k)[c].mean() for k in ks2], "o-", c="C0", label="frozen + intmatch (jackknife+)")
            ax.axhspan(0.85, 0.95, color="gray", alpha=.15); ax.axhline(0.8, c="gray", ls=":", lw=.7)
        for v, col in (("dec", "C3"), ("full", "C2")):
            g = D[(D.variant == v) & (D.method == "ft_B_OOF (final A)")].groupby("k")
            kk = [k for k in (5, 10, 25) if k in g.groups]
            if not kk:
                continue
            if c == "coverage":
                gg = cvd[cvd.variant == v].groupby("k")
                ax.plot(kk, [gg.get_group(k).coverage.mean() for k in kk], "s-", c=col, label=f"ft_B_OOF {v} (CV+)")
            else:
                ax.errorbar(kk, [g.get_group(k)[c].mean() for k in kk], yerr=[[g.get_group(k)[c].mean() - g.get_group(k)[c].quantile(.05) for k in kk], [g.get_group(k)[c].quantile(.95) - g.get_group(k)[c].mean() for k in kk]], fmt="s-", c=col, capsize=2, label=f"ft_B_OOF {v}")
        if c == "rel_l2":
            e1r = pd.read_csv(P3 / "T4" / "T4_3_references.csv").iloc[0, 3]
            ax.axhline(e1r, c="k", ls="--", lw=.8, label="E1 in-distribution")
        if c == "rmse_L":
            ax.axhline(49.9, c="k", ls="--", lw=.8, label="E1 in-distribution"); ax.axhline(10.1, c="gray", ls=":", lw=.8, label="label noise")
        ax.set_xscale("symlog", linthresh=1); ax.set_xticks(ks); ax.set_xticklabels(ks); ax.set_xlabel("labelled cars k"); ax.set_title(t, fontsize=8)
    axs[0].legend(fontsize=5, frameon=False)
    fig.savefig(P3 / "T3" / "T3_curve.png", dpi=300, bbox_inches="tight"); fig.savefig(P3 / "T3" / "T3_curve.pdf", bbox_inches="tight")
    F50[F50.method == "intmatch_lam"].groupby("k")[["abs_bias_L", "rel_l2", "rmse_L", "abs_bias_D", "coverage"]].agg(["mean", lambda s: s.quantile(.05), lambda s: s.quantile(.95)]).round(4).to_csv(P3 / "T3" / "T3_curve.csv")
    print("analysis written")


if __name__ == "__main__":
    main()
