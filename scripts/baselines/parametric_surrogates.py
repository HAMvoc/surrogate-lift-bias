"""Classical-baseline and regional-gap study / W1: classical parametric baselines (linear and quadratic response surface, Gaussian process) on the 23 design parameters, trained on the E2A training cars, against M2 on the estateback test cars
Inputs: results/classical_baselines/cars.csv (CFD C_L,p and C_D,p), the 23-parameter table of DrivAerNet, the stored predictions of M2 and of the Transolver.
Few-shot: halves, draws and reference correction of the reference-correction study (orig: seed 100 + d, swap: seed 1100 + d), compared with the stored M2 reference correction (results/reference_correction/shift_draws.csv).
Output: results/classical_baselines/{w1_models.csv, w1_fewshot_draws.csv, w1_cells.csv, w1_verdict.md}. Usage: python scripts/baselines/parametric_surrogates.py"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.correction.swapped_halves import stats  # noqa: E402

OUT = ROOT / "results" / "classical_baselines"
PARAMS = (ROOT / "configs" / "../../DrivAerNet/ParametricModels/DrivAerNet_ParametricData.csv").resolve()
ALPHAS = (0.01, 0.1, 1, 10, 100)
KS = (5, 10, 25)
TOL = 3.0  # counts, PREREG Q2 (as the reference-correction study)
SETTINGS = (("orig", np.arange(125), np.arange(125, 250), 100), ("swap", np.arange(125, 250), np.arange(125), 1100))


def standardise(Xtr, *others):
    mu, sd = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1.0)
    return [(x - mu) / sd for x in (Xtr, *others)]


def fit_predict(kind, Xtr, ytr, Xva, yva, Xte):
    """Returns (predict on val, predict on test, chosen setting)."""
    if kind == "G":
        k = ConstantKernel(1.0) * RBF(length_scale=np.full(Xtr.shape[1], 3.0), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=0).fit(Xtr, ytr)
        return gp.predict(Xva), gp.predict(Xte), "ard_rbf"
    if kind == "Q":
        P = PolynomialFeatures(2, include_bias=False)
        Ptr, Pva, Pte = standardise(P.fit_transform(Xtr), P.transform(Xva), P.transform(Xte))
        Xtr, Xva, Xte = Ptr, Pva, Pte
    best = min(((float(np.sqrt(np.mean((Ridge(alpha=a).fit(Xtr, ytr).predict(Xva) - yva) ** 2))), a) for a in ALPHAS))
    m = Ridge(alpha=best[1]).fit(Xtr, ytr)
    return m.predict(Xva), m.predict(Xte), f"alpha={best[1]}"


def metrics(pred, true):
    e = pred - true
    return {"bias": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean())), "rmse_nomean": float(e.std()), "rho": float(pearsonr(pred, true)[0])}


def label(d):
    pB = wilcoxon(d, alternative="less").pvalue if np.any(d) else 1.0  # baseline lower
    pM = wilcoxon(d, alternative="greater").pvalue if np.any(d) else 1.0  # M2 lower
    return ("baseline lower" if d.mean() < -TOL and pB < 0.05 else "M2 lower" if d.mean() > TOL and pM < 0.05 else "no clear difference" if abs(d.mean()) <= TOL else "unclear"), pB, pM


def main():
    D = pd.read_csv(OUT / "cars.csv").set_index("sample_id")
    pc = pd.read_csv(PARAMS).set_index("Experiment")
    cols = [c for c in pc.columns if not c.startswith(("Average", "Std"))]
    assert len(cols) == 23
    tr, va, te = (D[D.role == r].index for r in ("train", "val", "test"))
    Xtr, Xva, Xte = standardise(*(pc.loc[ix, cols].to_numpy(float) for ix in (tr, va, te)))
    # learned comparators on the same 250 test cars (stored predictions)
    perm, zM = stats("E2A", "M2")
    _, zT = stats("E2A", "TSV")
    assert abs(zM["dL"].mean(0).mean() - 183.9) <= 0.5  # gate
    stored = pd.read_csv(ROOT / "results" / "reference_correction" / "shift_draws.csv").query("fold == 'E2A' and family == 'M2'").set_index(["setting", "k", "draw"]).abs_bias_L
    eM = zM["dL"].mean(0)
    ev0 = np.arange(125, 250); id0 = np.arange(125)[np.random.default_rng(100).permutation(125)[:10]]
    assert abs(abs(eM[ev0].mean() - eM[id0].mean()) - stored[("orig", 10, 0)]) < 1e-3  # self-check against the stored M2 reference correction
    rows, E = [], {}
    for name, z in (("M2", zM), ("Transolver", zT)):
        p = z["CL"] + z["dL"].mean(0)
        rows.append({"model": name, "target": "CL", **{f"test_{k}": v for k, v in metrics(p, z["CL"]).items()}})
    perm_idx = pd.Index(perm)
    for tgt in ("CL", "CD"):
        for kind, nm in (("L", "linear response surface"), ("Q", "quadratic response surface"), ("G", "Gaussian process")):
            pva, pte, sel = fit_predict(kind, Xtr, D.loc[tr, tgt].to_numpy(), Xva, D.loc[va, tgt].to_numpy(), Xte)
            mt, mv = metrics(pte, D.loc[te, tgt].to_numpy()), metrics(pva, D.loc[va, tgt].to_numpy())
            rows.append({"model": nm, "target": tgt, "selected": sel, **{f"test_{k}": v for k, v in mt.items()}, **{f"val_{k}": v for k, v in mv.items()}})
            if tgt == "CL":
                ps = pd.Series(pte, index=te)
                E[nm] = ps.loc[perm_idx].to_numpy() - D.loc[perm_idx, "CL"].to_numpy()  # error per car in the order of the M2 stats
    M = pd.DataFrame(rows)
    M.round(4).to_csv(OUT / "w1_models.csv", index=False)
    # few-shot reference correction
    dr = []
    for st, pool, ev, seed0 in SETTINGS:
        for d in range(50):
            pm = np.random.default_rng(seed0 + d).permutation(len(pool))
            for k in KS:
                idx = pool[pm[:k]]
                base = {"M2": stored[(st, k, d)]}
                for nm, e in E.items():
                    base[nm] = abs(e[ev].mean() - e[idx].mean())
                dr.append({"setting": st, "k": k, "draw": d, **{f"abs_bias_{nm}": v for nm, v in base.items()}})
    R = pd.DataFrame(dr)
    R.round(5).to_csv(OUT / "w1_fewshot_draws.csv", index=False)
    cells = []
    for (st, k), g in R.groupby(["setting", "k"]):
        for nm in E:
            d = (g[f"abs_bias_{nm}"] - g["abs_bias_M2"]).to_numpy()
            lab, pB, pM = label(d)
            cells.append({"setting": st, "k": k, "baseline": nm, "baseline_abs_bias": g[f"abs_bias_{nm}"].mean(), "M2_abs_bias": g.abs_bias_M2.mean(), "diff": d.mean(), "p_baseline_lower": pB, "p_M2_lower": pM, "label": lab})
    C = pd.DataFrame(cells)
    C.round(4).to_csv(OUT / "w1_cells.csv", index=False)
    # verdicts
    thr = 0.5 * abs(zM["dL"].mean(0).mean())
    cl = M[(M.target == "CL") & M.selected.notna()]
    big = int((cl.test_bias.abs() >= thr).sum())
    q1 = "avoids the bias" if big == 0 else "does not avoid" if big >= 2 else "mixed"
    c10 = C[C.k == 10]
    cnt = c10.label.value_counts()
    q2 = next((l for l in ("baseline lower", "M2 lower", "no clear difference") if cnt.get(l, 0) >= 4), "mixed")
    fm = lambda d, cols_: md(d[cols_].round(3).to_dict("records"))
    L = ["# Classical-baseline and regional-gap study / W1: classical parametric baselines against M2 (criteria: `PREREG.md`)", "",
         "Self-checks passed: M2 gate (+183.9 counts) and the stored M2 reference correction (orig, k = 10, draw 0).", "",
         f"**Q1 verdict: {q1}** - |bias| of C_L,p on the 250 estateback cars at or above {thr:.1f} counts (half of M2): {big} of 3 baselines. Prediction: does not avoid.", "",
         f"**Q2 verdict: {q2}** - labels at k = 10 over the 6 cells: {dict(cnt)}. Prediction: no clear difference.", "",
         "## Q1, Q3, Q4: models on the E2A test and validation cars (counts; rho = Pearson correlation of prediction and CFD)", "",
         fm(M, [c for c in M.columns if c in ("model", "target", "selected", "test_bias", "test_rmse", "test_rmse_nomean", "test_rho", "val_rmse", "val_rho")]), "",
         "## Q2: reference correction, |bias| of C_L,p on the evaluation half (mean over 50 draws)", "", fm(C, ["setting", "k", "baseline", "baseline_abs_bias", "M2_abs_bias", "diff", "p_baseline_lower", "p_M2_lower", "label"]), ""]
    (OUT / "w1_verdict.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:8]))


if __name__ == "__main__":
    main()
