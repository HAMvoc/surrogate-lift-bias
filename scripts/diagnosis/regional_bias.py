"""Decomposition and first corrections study, task T1: re-run the regional analyses with the v2 regions (wheels split off) and put old and new side by side.
(1) the force-alignment study T3 regional bias decomposition for every model family on E2A/E2B/E2C; (2) the force-alignment study T5 between-car variance decomposition (M2, E2A);
(3) the zero-shot checks study T3 region_const / scalar_shift with 8 regions on the same 50 random draws. Output: results/decomposition/T1/.
Usage: python scripts/diagnosis/regional_bias.py"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.force_alignment_terms import MODELS, ci, ci_ratio, fm  # noqa: E402
from scripts.diagnosis.force_alignment import REGIONS, region_labels, terms_table  # noqa: E402
from scripts.correction.calibration import build_stats, corrected, jackknife_plus, make_fit, metrics  # noqa: E402
from scripts.diagnosis.regions import REGIONS2, labeler as labeler_v2  # noqa: E402

OUT, RES = ROOT / "results" / "decomposition" / "T1", ROOT / "results"
FOLDS = ("E2A", "E2B", "E2C")


def v2_terms():
    f = OUT / "per_car_terms_v2.csv.gz"
    if f.exists():
        return pd.read_csv(f)
    parts = []
    for fold in FOLDS:
        runs = [(m, s, RES / pat.format(f=fold, s=s)) for m, (pat, seeds) in MODELS.items() for s in seeds]
        parts.append(terms_table(fold, "test", runs, labeler=labeler_v2, region_names=REGIONS2))
    t = pd.concat(parts, ignore_index=True)
    t.to_csv(f, index=False, compression="gzip")
    return t


def agg(t):
    return t.drop(columns=["seed", "role"]).groupby(["fold", "model", "id"]).mean(numeric_only=True).reset_index()


def main():
    lines = ["# Decomposition and first corrections study, task T1: regions v2 (wheels split off), old and new side by side", ""]
    # sanity on the wheel fit (amended bound 10-20%)
    z = np.load(OUT / "labels_v2.npz")
    fr = np.array([(z[k] == 0).mean() for k in z.files])
    w = pd.read_csv(OUT / "wheel_fits.csv")
    assert fr.min() >= 0.10 and fr.max() <= 0.20, "wheel share outside 10-20%: stop and report"
    assert 0.30 <= w.R.mean() <= 0.38
    lines += [f"Wheel fit (750 cars, `scripts/diagnosis/regions.py`): wheel share of the surface points per car {100 * fr.min():.1f}% to {100 * fr.max():.1f}% (median {100 * np.median(fr):.1f}%); fitted tyre radius {w.R.min():.3f}-{w.R.max():.3f} m (mean {w.R.mean():.3f}), "
              f"axle x_c front {w[w.axle == 'front'].x_c.mean():.2f} m and rear {w[w.axle == 'rear'].x_c.mean():.2f} m on average; minimum Hough votes {int(w.votes.min())}; width found for every wheel. Sanity bounds (PREREG with Amendment 1): 10-20% and mean R in [0.30, 0.38]: PASS.", ""]
    T2 = v2_terms()
    old = pd.read_csv(RES / "force_alignment" / "per_car_terms.csv.gz")
    gate = agg(T2)[(lambda a: (a.fold == "E2A") & (a.model == "M2"))(agg(T2))].dL.mean()
    assert abs(gate - 183.9) <= 0.5, gate
    lines += [f"Gate: M2 E2A 3-seed bias C_L,p with the v2 labels = {gate:+.1f} counts (required 183.9 +- 0.5): PASS.", ""]
    A_new, A_old = agg(T2), old[old.fold.isin(FOLDS)].drop(columns=["seed", "role"]).groupby(["fold", "model", "id"]).mean(numeric_only=True).reset_index()
    # (1) bias decomposition by region
    rows = []
    for (f, m), g in A_new.groupby(["fold", "model"]):
        go = A_old[(A_old.fold == f) & (A_old.model == m)]
        bL = g.dL.values
        for r in REGIONS2:
            cL = g[f"reg_dL_{r}"].values
            c = ci(cL)
            pct = ci_ratio(cL, bL) if abs(bL.mean()) >= 20 else (np.nan,) * 3
            rows.append({"fold": f, "model": m, "region": r, "area %": 100 * g[f"reg_area_{r}"].mean(), "contribution C_L,p": c[0], "lo": c[1], "hi": c[2], "pct of bias": pct[0], "pct lo": pct[1], "pct hi": pct[2]})
    D = pd.DataFrame(rows)
    D.round(3).to_csv(OUT / "T1_bias_by_region_v2.csv", index=False)
    old_rows = []
    for (f, m), g in A_old.groupby(["fold", "model"]):
        for r in REGIONS:
            cL = g[f"reg_dL_{r}"].values
            old_rows.append({"fold": f, "model": m, "region": r, "area %": 100 * g[f"reg_area_{r}"].mean(), "contribution C_L,p": cL.mean()})
    pd.DataFrame(old_rows).round(3).to_csv(OUT / "T1_bias_by_region_old.csv", index=False)
    lines += ["## 1. Bias C_L,p by region, E2A, all model families (counts; share of the bias in percent [CI over cars]; old = 7 regions of the force-alignment study, new = 8 regions with wheels)", ""]
    for m in ("M2", "B0", "B1", "B2_300ep", "B3"):
        a = D[(D.fold == "E2A") & (D.model == m)].set_index("region")
        o = pd.DataFrame(old_rows)
        o = o[(o.fold == "E2A") & (o.model == m)].set_index("region")
        tot = A_new[(A_new.fold == "E2A") & (A_new.model == m)].dL.mean()
        row_new = {r: f"{a.loc[r, 'contribution C_L,p']:+.1f} ({a.loc[r, 'pct of bias']:.0f}% [{a.loc[r, 'pct lo']:.0f}; {a.loc[r, 'pct hi']:.0f}])" for r in REGIONS2}
        row_old = {r: f"{o.loc[r, 'contribution C_L,p']:+.1f} ({100 * o.loc[r, 'contribution C_L,p'] / tot:.0f}%)" for r in REGIONS}
        lines += [f"**{m}** (bias {tot:+.1f} counts)", "", md([{"set": "new", **row_new}]), "", md([{"set": "old", **row_old}]), ""]
    area = [{"region": r, "area % (new)": f"{100 * A_new[(A_new.fold == 'E2A') & (A_new.model == 'M2')][f'reg_area_{r}'].mean():.1f}"} for r in REGIONS2]
    lines += ["Area shares of the new regions on the E2A test cars:", "", md(area), ""]
    # M2 on all three folds, new regions
    r3 = []
    for f in FOLDS:
        a = D[(D.fold == f) & (D.model == "M2")].set_index("region")
        tot = A_new[(A_new.fold == f) & (A_new.model == "M2")].dL.mean()
        r3.append({"fold": f, "bias": f"{tot:+.1f}", **{r: f"{a.loc[r, 'contribution C_L,p']:+.1f}" for r in REGIONS2}})
    lines += ["M2, contributions by region (counts) on the three folds, new regions:", "", md(r3), ""]
    # (2) variance decomposition, M2 E2A
    g = A_new[(A_new.fold == "E2A") & (A_new.model == "M2")]
    go = A_old[(A_old.fold == "E2A") & (A_old.model == "M2")]
    tot, v = g.dL.values, g.dL.var(ddof=1)
    rows = []
    for r in REGIONS2:
        c = g[f"reg_dL_{r}"].values
        rows.append({"region (new)": r, "mean (counts)": f"{c.mean():+.1f}", "SD across cars": f"{c.std(ddof=1):.1f}", "share of Var(Delta C) = Cov(c_r, total)/Var(total)": f"{100 * np.cov(c, tot)[0, 1] / v:.0f}%"})
    rows_o = []
    for r in REGIONS:
        c = go[f"reg_dL_{r}"].values
        rows_o.append({"region (old)": r, "mean (counts)": f"{c.mean():+.1f}", "SD across cars": f"{c.std(ddof=1):.1f}", "share of Var(Delta C)": f"{100 * np.cov(c, go.dL.values)[0, 1] / go.dL.var(ddof=1):.0f}%"})
    sh = {r: 100 * np.cov(g[f"reg_dL_{r}"].values, tot)[0, 1] / v for r in REGIONS2}
    ub = sh["underbody"]
    bias_sh = {r: 100 * g[f"reg_dL_{r}"].mean() / tot.mean() for r in REGIONS2}
    v_verdict = "KEPT" if ub >= 50 else ("REJECTED" if ub < 30 else "MIXED")
    b_verdict = "KEPT" if bias_sh["underbody"] >= 25 else ("REJECTED" if bias_sh["underbody"] < 15 else "MIXED")
    lines += ["## 2. Variance of Delta C_L,p between the 250 Estateback cars, M2 (3-seed mean): new and old regions", "", md(rows), "", md(rows_o), "",
              f"Registered hypotheses (PREREG T1): 'the underbody is the main source of the between-car variance' (kept if the new `underbody` carries >= 50% of the variance, rejected if < 30%): new underbody {ub:.0f}% (wheels {sh['wheels']:.0f}%) -> **{v_verdict}**. "
              f"'The underbody carries 33% of the bias' (kept if the new `underbody` has >= 25% of the bias, rejected if < 15%): new underbody {bias_sh['underbody']:.0f}% (wheels {bias_sh['wheels']:.0f}%) -> **{b_verdict}**.", ""]
    # (3) calibration with 8 regions vs 7
    ids = list(pd.read_csv(RES / "E2A_M2_R2_s0" / "test_per_car.csv").id)
    preds = [np.load(RES / f"E2A_M2_R2_s{s}" / "test_pred_fp16.npy", mmap_mode="r") for s in range(3)]
    pos = [{i: n for n, i in enumerate(pd.read_csv(RES / f"E2A_M2_R2_s{s}" / "test_per_car.csv").id)} for s in range(3)]
    get = lambda s, sid: preds[s][pos[s][sid]]
    t0 = time.time()
    st_new = build_stats(ids, get, 3, labeler_v2, 8)
    st_old = build_stats(ids, get, 3, lambda sid, xyz, nrm: region_labels(xyz, nrm), 7)
    assert abs(st_new["dL"].mean(0).mean() - 183.9) <= 0.5 and abs(st_old["dL"].mean(0).mean() - 183.9) <= 0.5
    n = len(ids)
    res = []
    for k in (5, 10, 25):
        rng = np.random.default_rng(2026 + k)
        draws = [np.sort(rng.permutation(n)[:k]) for _ in range(50)]
        for name, st in (("old (7 regions)", st_old), ("new (8 regions, wheels)", st_new)):
            for method in ("none", "scalar_shift", "region_const"):
                acc = []
                for idx in draws:
                    cars = np.setdiff1d(np.arange(n), idx)
                    m = metrics(st, cars, make_fit(method, st, idx))
                    cov = hw = np.nan
                    if method != "none":
                        cov, hw, _ = jackknife_plus(st, idx, cars, method)
                    acc.append({**{c: m[c] for c in ("abs_bias_L", "rmse_L", "rel_l2", "abs_bias_D")}, "cover": cov, "half": hw})
                a = pd.DataFrame(acc)
                res.append({"k": k, "regions": name, "method": method, **{c: f"{a[c].mean():.4g}" for c in a.columns}})
    lines += ["## 3. Zero-shot checks study T3 with 8 regions: `region_const` on the same 50 random draws per k (mean over draws; evaluation on the other 250 - k cars)", "",
              md([{**r, "abs_bias_L": f"{float(r['abs_bias_L']):.1f}", "rmse_L": f"{float(r['rmse_L']):.1f}", "rel_l2": f"{float(r['rel_l2']):.4f}", "abs_bias_D": f"{float(r['abs_bias_D']):.1f}", "cover": r["cover"] if r["cover"] != "nan" else "-", "half": (f"{float(r['half']):.0f}" if r["half"] != "nan" else "-")} for r in res]), ""]
    pd.DataFrame(res).to_csv(OUT / "T1_region_const_old_vs_new.csv", index=False)
    (OUT / "T1_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("done", f"{time.time() - t0:.0f} s calibration part; hypotheses:", v_verdict, b_verdict)


if __name__ == "__main__":
    main()
