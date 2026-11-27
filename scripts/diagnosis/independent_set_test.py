"""Independent-set study, task T2: independent test on the new Estateback cars. Nothing is printed before the whole analysis has run.
Inputs: the new cars (data/splits/fresh_estateback.csv, data/cache), their geometry rows (data/manifests/car_geometry_fresh.csv), v2 labels (results/independent_set/labels_v2_fresh.npz) and the predictions of the E2A models
on them (results/independent_set/preds/<run label>/, scripts/training/infer_cars.py); the predictions on the old test cars are the stored ones (results/<run label>/).
Output: results/independent_set/T2/ (Q1_bias.csv, Q2_draws.csv, Q2_summary.csv, Q3_cells.csv, Q4_coverage.csv, describe.csv, regions.csv, verdicts.md).
--standin: the evaluation set is the old test set itself (code check on old cars only, writes to results/independent_set/standin, prints no verdict). Usage: python scripts/diagnosis/independent_set_test.py [--standin]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis import pressure_coefficients as cda  # noqa: E402
from scripts.diagnosis import force_alignment as fa  # noqa: E402
from scripts.correction.calibration import build_stats, corrected  # noqa: E402
from scripts.correction.frozen_draws import fit_method, jackknife  # noqa: E402
from scripts.training.infer_cars import FAMS  # noqa: E402
from scripts.diagnosis.regions import REGIONS2  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

KS = (1, 2, 3, 5, 10, 25)
C = float(np.sqrt(2 / np.pi))
B = 2000


def boot_ci(x, seed=0):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standin", action="store_true")
    a = ap.parse_args()
    out = ROOT / "results" / "independent_set" / ("standin" if a.standin else "T2")
    out.mkdir(parents=True, exist_ok=True)
    old = [str(i) for i in np.random.default_rng(0).permutation(sorted(load_split("E2A")["test"]))]
    new = old if a.standin else sorted(pd.read_csv(ROOT / "data" / "splits" / "fresh_estateback.csv").sample_id)
    ids = old + new
    old_set = set(old)
    n_old, n_new = len(old), len(new)
    # geometry rows and labels of the new cars (in memory; the frozen files of earlier studies are not touched)
    if not a.standin:
        g = pd.read_csv(ROOT / "data" / "manifests" / "car_geometry_fresh.csv").set_index("sample_id")
        for sid in new:
            cda.GEO.loc[sid, ["area_total_m2", "frontal_area_m2", "incl_rate_per_m2"]] = g.loc[sid, ["area_total_m2", "frontal_area_m2", "incl_rate_per_m2"]].to_numpy(float)
    labs_old = np.load(ROOT / "results" / "decomposition" / "T1" / "labels_v2.npz")
    labs_new = labs_old if a.standin else np.load(ROOT / "results" / "independent_set" / "labels_v2_fresh.npz")
    labeler = lambda sid, xyz, nrm: (labs_new[sid] if sid in labs_new.files and sid not in labs_old.files else labs_old[sid]).astype(np.int64)
    pred_root = ROOT / "results" / "independent_set" / "preds"
    cfd = pd.read_csv(ROOT / "results" / "force_alignment" / "cfd_coefficients.csv").set_index("id")
    train = load_split("E2A")["train"]
    trL = cfd.loc[train, "CL"].mean()
    res = {}
    for fam, labels in FAMS.items():
        dirs_old = [ROOT / "results" / l for l in labels]
        dirs_new = dirs_old if a.standin else [pred_root / l for l in labels]
        if not all((d / "test_pred_fp16.npy").exists() for d in dirs_new):
            res[fam] = None
            continue
        arr = [(np.load(d / "test_pred_fp16.npy", mmap_mode="r"), {i: j for j, i in enumerate(pd.read_csv(d / "test_per_car.csv").id)}) for d in dirs_old]
        arr_n = [(np.load(d / "test_pred_fp16.npy", mmap_mode="r"), {i: j for j, i in enumerate(pd.read_csv(d / "test_per_car.csv").id)}) for d in dirs_new]

        # the same id can be an old car (stored predictions) or a new car (new predictions); in the stand-in both are the same old cars
        def get_pred(s, sid, arr=arr, arr_n=arr_n):
            src = arr if sid in old_set else arr_n
            return src[s][0][src[s][1][sid]]

        z = build_stats(ids, get_pred, len(labels), labeler, 8)
        res[fam] = {"z": z, "S": len(labels), "dirs_new": dirs_new}
    fams = [f for f in FAMS if res[f] is not None]
    skipped = [f for f in FAMS if res[f] is None]
    ev = np.arange(n_old, n_old + n_new)
    pool_idx = np.arange(n_old)
    rows_q2, q1, desc, regs = [], [], [], []
    for fam in fams:
        z = res[fam]["z"]
        dLs = z["dL"]  # (S, n)
        for where, sl in (("old", slice(0, n_old)), ("new", slice(n_old, n_old + n_new))):
            e = dLs[:, sl].mean(0)  # seed-mean error per car
            bias = e.mean()
            sign = np.sign(bias)
            cl = z["CL"][sl]
            pred = cl + e
            lo, hi = boot_ci(e)
            trG = trL - cl.mean()
            q1.append({"family": fam, "set": where, "bias_L": bias, "bias_lo": lo, "bias_hi": hi, "same_sign": float((np.sign(e) == sign).mean()), "bias_D": z["dD"][:, sl].mean(0).mean(), "G": trG, "overshoot": pred.mean() - trL,
                       "slope": float(np.polyfit(cl, pred, 1)[0]), "spearman": float(spearmanr(cl, pred)[0]), "rmse_after_bias": float(e.std()), "n_cars": len(e)})
        e0 = corrected(z, ev, ("none", None))
        eL0 = e0[0].mean(0)
        sigma = float(eL0.std())
        for d in range(50):
            for k in KS:
                idx = pool_idx[np.random.default_rng(2100 + d).permutation(n_old)[:k]]
                dL, dD, rel = corrected(z, ev, fit_method(z, idx, "intmatch_lam"))
                eL = dL.mean(0)
                r = {"family": fam, "k": k, "draw": d, "abs_bias_L": abs(eL.mean()), "abs_bias_D": abs(dD.mean(0).mean()), "rmse_L": float(eL.std()), "rel_l2": rel.mean(), "sigma_pop": sigma}
                if k >= 5:
                    r["coverage"], r["half"] = jackknife(z, idx, ev, "intmatch_lam")
                rows_q2.append(r)
        print(fam, "done", flush=True) if a.standin else None
    # regional decomposition (3-seed mean error of the new cars) with the existing terms table
    Q1 = pd.DataFrame(q1)
    Q1.to_csv(out / "Q1_bias.csv", index=False)
    D = pd.DataFrame(rows_q2)
    D.round(5).to_csv(out / "Q2_draws.csv", index=False)
    S = D.groupby(["family", "k"]).agg(abs_bias_L=("abs_bias_L", "mean"), abs_bias_D=("abs_bias_D", "mean"), rel_l2=("rel_l2", "mean"), rmse_L=("rmse_L", "mean"), coverage=("coverage", "median"), sigma_pop=("sigma_pop", "first")).reset_index()
    S.round(4).to_csv(out / "Q2_summary.csv", index=False)
    cells = S[S.k.isin([5, 10, 25])].copy()
    cells["pred_lo"] = 1.0 * C * cells.sigma_pop / np.sqrt(cells.k)
    cells["pred_hi"] = 1.3 * C * cells.sigma_pop / np.sqrt(cells.k)
    cells["fits"] = (cells.abs_bias_L >= 0.7 * cells.pred_lo) & (cells.abs_bias_L <= 1.4 * cells.pred_hi)
    cells.round(4).to_csv(out / "Q3_cells.csv", index=False)
    k10 = S[S.k == 10]
    cov = k10[["family", "coverage"]].assign(under=k10.coverage < 0.85)
    cov.round(4).to_csv(out / "Q4_coverage.csv", index=False)
    # regions
    for fam in fams:
        terms = fa.terms_table("FRESH" if not a.standin else "E2A", "test", [(fam, s, d) for s, d in enumerate(res[fam]["dirs_new"])], labeler=labeler, region_names=REGIONS2, log=lambda *x: None)
        t = terms.groupby("id").mean(numeric_only=True)
        row = {"family": fam, **{r: t[f"reg_dL_{r}"].mean() for r in REGIONS2}, "total": t.dL.mean()}
        row["roof_underbody_share_pct"] = 100 * (row["roof_rear"] + row["roof_front"] + row["underbody"]) / row["total"] if row["total"] else np.nan
        regs.append(row)
    pd.DataFrame(regs).to_csv(out / "regions.csv", index=False)
    # verdicts
    m = {f: Q1[(Q1.family == f) & (Q1.set == "new")].iloc[0] for f in fams}
    v1 = {}
    for f in ("M2", "TSV"):
        if f in m:
            b, s = m[f].bias_L, m[f].same_sign
            v1[f] = "repeats" if (120 <= b <= 260 and s >= 0.9) else "does not repeat" if b < 60 else "partly"
    q1_all = "repeats" if v1 and all(v == "repeats" for v in v1.values()) and len(v1) == 2 else "does not repeat" if v1 and all(v == "does not repeat" for v in v1.values()) and len(v1) == 2 else "partly"
    mean10 = float(k10.abs_bias_L.mean())
    q2 = "repeats" if 12 <= mean10 <= 32 else "does not repeat" if mean10 > 40 else "partly"
    fr = float(cells.fits.mean())
    q3 = "repeats" if fr >= 0.8 else "does not repeat" if fr < 0.5 else "partly"
    if a.standin:
        print("stand-in finished; written to", out)
        return
    L = ["# Independent-set study, task T2: independent test on the new Estateback cars", "", f"New cars: {n_new}; families run: {', '.join(fams)}; not run: {', '.join(skipped) or 'none'}.", "",
         "## Q1 (M2 and Transolver, 3-seed mean prediction)", ""] + [f"- {f}: bias {m[f].bias_L:+.1f} [{m[f].bias_lo:+.1f}; {m[f].bias_hi:+.1f}] counts, same sign on {100 * m[f].same_sign:.1f}% of the cars -> {v1[f]}" for f in v1] + [f"- **Q1 as a whole: {q1_all}** (repeats: bias in [+120; +260] and >= 90% same sign; does not repeat: bias below +60).", "",
         f"## Q2 (stage B, pool = old test cars, evaluation = new cars)", "", f"- mean over the {len(fams)} families of the mean |bias| C_L,p at k = 10: {mean10:.1f} counts (families: " + ", ".join(f"{r.family} {r.abs_bias_L:.1f}" for r in k10.itertuples()) + f") -> **{q2}** (repeats: [12; 32]; does not repeat: above 40).", "",
         f"## Q3 (floor law, sigma_pop on the new cars)", "", f"- cells that fit: {int(cells.fits.sum())} of {len(cells)} ({100 * fr:.0f}%) -> **{q3}** (repeats: >= 80%; does not repeat: < 50%).", "",
         "## Q4 (coverage, description)", "", "- median jackknife+ coverage at k = 10: " + ", ".join(f"{r.family} {r.coverage:.3f}" for r in k10.itertuples()) + f"; families below 0.85: {int((k10.coverage < 0.85).sum())} of {len(k10)}.", "",
         "Prediction: Q1, Q2 and Q3 repeat; coverage at least 0.85 for every family.", ""]
    (out / "verdicts.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
