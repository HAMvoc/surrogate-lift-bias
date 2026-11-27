"""POD-baseline study B: POD field reconstruction baseline.
Snapshots: clipped Cp of each car at the pool point nearest to each of 20000 common points R (drawn from the pooled training points, as the nearest-point transfer of B0).
POD of the 450 E2A training cars; modal coefficients regressed on the 23 design parameters (POD-GP: one ARD GP per mode, kernel of W1; POD-ridge); r and the penalty chosen on the 50 validation cars.
Prediction transferred back to each car's own pool points by the nearest point of R, integrated with the car's normals and weights (definitions of scripts/data/per_car_coefficients.py).
Few-shot reference correction and labels as scripts/baselines/parametric_surrogates.py. Output: results/pod/{pod_models.csv, pod_select.csv, pod_percar.csv, pod_fewshot_draws.csv, pod_cells.csv, pod_verdict.md}.
Usage: python scripts/baselines/pod_reconstruction.py"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.pressure_coefficients import CLIP, GEO, pool_weights  # noqa: E402
from scripts.diagnosis.force_alignment import load_pool  # noqa: E402
from scripts.correction.swapped_halves import stats  # noqa: E402
from scripts.baselines.parametric_surrogates import KS, PARAMS, SETTINGS, label, metrics, standardise  # noqa: E402

OUT = ROOT / "results" / "pod"
NR, RS, ALPHAS = 20000, (5, 10, 20, 40), (0.01, 0.1, 1, 10, 100)
G0 = 112.755


class Car:
    __slots__ = ("sid", "ref", "kz", "kx", "near", "s")

    def __init__(self, sid, R):
        xyz, nrm, cp, area = load_pool(sid)
        p = np.asarray(xyz, np.float64)
        self.sid, self.ref = sid, np.clip(cp.astype(np.float64), *CLIP)
        w = pool_weights(np.asarray(area, np.float64), sid)
        S = float(GEO.at[sid, "area_total_m2"] / GEO.at[sid, "frontal_area_m2"])
        self.kz, self.kx = -1000 * S * w * nrm[:, 2] / w.sum(), -1000 * S * w * nrm[:, 0] / w.sum()
        self.s = self.ref[cKDTree(p).query(R, workers=-1)[1]]  # snapshot on R; the tree is not kept (memory)
        self.near = cKDTree(R).query(p, workers=-1)[1]  # nearest R point of every pool point (transfer back)

    def integrate(self, cp):
        return float((self.kz * cp).sum()), float((self.kx * cp).sum())

    def rel(self, cp):
        return float(np.linalg.norm(cp - self.ref) / np.linalg.norm(self.ref))


def selfcheck_rel():
    d = ROOT / "results" / "E2A_M2_R2_s0"
    pc = pd.read_csv(d / "test_per_car.csv")
    sid = pc.id[0]
    pred = np.asarray(np.load(d / "test_pred_fp16.npy", mmap_mode="r")[0], np.float64)
    ref = np.clip(load_pool(sid)[2].astype(np.float64), *CLIP)
    r = np.linalg.norm(pred - ref) / np.linalg.norm(ref)
    assert abs(r - pc.rel_l2[0]) < 1e-3, (r, pc.rel_l2[0])


def gp_fit(X, y):
    k = ConstantKernel(1.0) * RBF(length_scale=np.full(X.shape[1], 3.0), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=0).fit(X, y)


def main():
    selfcheck_rel()
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(ROOT / "results" / "classical_baselines" / "cars.csv").set_index("sample_id")
    pc = pd.read_csv(PARAMS).set_index("Experiment")
    cols = [c for c in pc.columns if not c.startswith(("Average", "Std"))]
    assert len(cols) == 23
    tr, va, te = (list(D[D.role == r].index) for r in ("train", "val", "test"))
    Xtr, Xva, Xte = standardise(*(pc.loc[ix, cols].to_numpy(float) for ix in (tr, va, te)))

    rng = np.random.default_rng(0)
    pools = {sid: np.asarray(load_pool(sid)[0], np.float64) for sid in tr}
    ci = rng.integers(0, len(tr), NR)
    R = np.array([pools[tr[c]][rng.integers(0, len(pools[tr[c]]))] for c in ci])
    del pools
    cars, Y = {}, {"tr": [], "va": [], "te": []}
    for g, ix in (("tr", tr), ("va", va), ("te", te)):
        for sid in ix:
            c = Car(sid, R)
            cl, cd = c.integrate(c.ref)  # self-check of the integration
            assert abs(cl - D.at[sid, "CL"]) < 1e-3 and abs(cd - D.at[sid, "CD"]) < 1e-3, sid
            Y[g].append(c.s)
            if g != "tr":  # only validation and test cars are needed again (memory)
                cars[sid] = c
    Y = {g: np.array(v) for g, v in Y.items()}
    mu = Y["tr"].mean(0)
    _, sv, Vt = np.linalg.svd(Y["tr"] - mu, full_matrices=False)
    energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    A = (Y["tr"] - mu) @ Vt[: max(RS)].T  # training coefficients

    def fields(coef, r):
        return mu + coef[:, :r] @ Vt[:r]

    def evaluate(ix, F):
        rows = []
        for sid, f in zip(ix, F):
            cp = f[cars[sid].near]
            cl, cd = cars[sid].integrate(cp)
            rows.append({"sample_id": sid, "CL_pred": cl, "CD_pred": cd, "rel": cars[sid].rel(cp)})
        return pd.DataFrame(rows).set_index("sample_id")

    # model selection on the validation cars
    gps = [gp_fit(Xtr, A[:, j]) for j in range(max(RS))]
    cgp_va = np.column_stack([g.predict(Xva) for g in gps])
    sel = []
    for r in RS:
        sel.append({"variant": "POD-GP", "r": r, "alpha": np.nan, "val_rel": evaluate(va, fields(cgp_va, r)).rel.mean(), "energy": energy[r - 1]})
        for a in ALPHAS:
            m = Ridge(alpha=a).fit(Xtr, A[:, :r])
            sel.append({"variant": "POD-ridge", "r": r, "alpha": a, "val_rel": evaluate(va, fields(m.predict(Xva), r)).rel.mean(), "energy": energy[r - 1]})
    S = pd.DataFrame(sel)
    S.round(5).to_csv(OUT / "pod_select.csv", index=False)
    best = {v: S[S.variant == v].sort_values("val_rel").iloc[0] for v in ("POD-GP", "POD-ridge")}

    preds, rows, percar = {}, [], []
    perm, zM = stats("E2A", "M2")
    assert abs(zM["dL"].mean(0).mean() - 183.9) <= 0.5
    for v, b in best.items():
        r = int(b.r)
        if v == "POD-GP":
            cte, cva = np.column_stack([g.predict(Xte) for g in gps]), cgp_va
        else:
            m = Ridge(alpha=b.alpha).fit(Xtr, A[:, :r])
            cte, cva = m.predict(Xte), m.predict(Xva)
        Pte, Pva = evaluate(te, fields(cte, r)), evaluate(va, fields(cva, r))
        for tgt, col in (("CL", "CL_pred"), ("CD", "CD_pred")):
            mt = metrics(Pte[col].to_numpy(), D.loc[te, tgt].to_numpy())
            mv = metrics(Pva[col].to_numpy(), D.loc[va, tgt].to_numpy())
            rows.append({"model": v, "target": tgt, "r": r, "alpha": b.alpha, **{f"test_{k}": x for k, x in mt.items()}, **{f"val_{k}": x for k, x in mv.items()},
                         "test_rel": Pte.rel.mean(), "val_rel": Pva.rel.mean()})
        preds[v] = Pte.CL_pred.loc[perm].to_numpy() - D.loc[perm, "CL"].to_numpy()
        percar.append(Pte.assign(model=v))
    # projection floor: CFD field of each test car projected on the r training modes of POD-GP
    r = int(best["POD-GP"].r)
    Ffl = fields((Y["te"] - mu) @ Vt[:r].T, r)
    Pfl = evaluate(te, Ffl)
    rows.append({"model": "POD projection floor", "target": "CL", "r": r, **{f"test_{k}": x for k, x in metrics(Pfl.CL_pred.to_numpy(), D.loc[te, "CL"].to_numpy()).items()}, "test_rel": Pfl.rel.mean()})
    M = pd.DataFrame(rows)
    gtr = D.loc[tr, "CL"].mean() - D.loc[te, "CL"].mean()
    assert abs(gtr - G0) < 0.01, gtr
    M["G"] = np.where(M.target == "CL", gtr, np.nan)
    M["overshoot"] = M.test_bias - M.G
    M.round(5).to_csv(OUT / "pod_models.csv", index=False)
    pd.concat(percar).round(5).to_csv(OUT / "pod_percar.csv")

    stored = pd.read_csv(ROOT / "results" / "reference_correction" / "shift_draws.csv").query("fold == 'E2A' and family == 'M2'").set_index(["setting", "k", "draw"]).abs_bias_L
    dr = []
    for st, pool, ev, seed0 in SETTINGS:
        for d in range(50):
            pm = np.random.default_rng(seed0 + d).permutation(len(pool))
            for k in KS:
                idx = pool[pm[:k]]
                dr.append({"setting": st, "k": k, "draw": d, "abs_bias_M2": stored[(st, k, d)], **{f"abs_bias_{v}": abs(e[ev].mean() - e[idx].mean()) for v, e in preds.items()}})
    Rr = pd.DataFrame(dr)
    Rr.round(5).to_csv(OUT / "pod_fewshot_draws.csv", index=False)
    cells = []
    for (st, k), g in Rr.groupby(["setting", "k"]):
        for v in preds:
            dd = (g[f"abs_bias_{v}"] - g.abs_bias_M2).to_numpy()
            lab, pB, pM = label(dd)
            cells.append({"setting": st, "k": k, "baseline": v, "baseline_abs_bias": g[f"abs_bias_{v}"].mean(), "M2_abs_bias": g.abs_bias_M2.mean(), "diff": dd.mean(), "p_baseline_lower": pB, "p_M2_lower": pM, "label": lab})
    C = pd.DataFrame(cells)
    C.round(4).to_csv(OUT / "pod_cells.csv", index=False)

    gp = M[(M.model == "POD-GP") & (M.target == "CL")].iloc[0]
    rd = M[(M.model == "POD-ridge") & (M.target == "CL")].iloc[0]
    q1 = lambda x: "avoids the bias" if abs(x.test_bias) < 92 else "does not avoid"
    c10 = C[C.k == 10].label.value_counts()
    q2 = next((l for l in ("baseline lower", "M2 lower", "no clear difference") if c10.get(l, 0) >= 3), "mixed")
    L = ["# POD-baseline study B: POD baseline (criteria: `PREREG.md`)", "",
         "Self-checks passed: Rel L2 formula reproduces the stored M2 value; integration reproduces cars.csv C_L,p and C_D,p for all 750 cars; M2 gate; G = 112.755.", "",
         f"Selected: POD-GP r = {int(best['POD-GP'].r)} (energy {best['POD-GP'].energy:.3f}); POD-ridge r = {int(best['POD-ridge'].r)}, alpha = {best['POD-ridge'].alpha}.", "",
         f"**Q1:** POD-GP bias C_L,p {gp.test_bias:+.1f} counts -> {q1(gp)}; POD-ridge {rd.test_bias:+.1f} -> {q1(rd)}. Prediction: does not avoid, 110-160.",
         f"**Q2:** labels at k = 10 over 4 cells: {dict(c10)} -> {q2}. Prediction: no clear difference.",
         f"**Q3:** Rel L2 estateback POD-GP {gp.test_rel:.3f}, POD-ridge {rd.test_rel:.3f}; validation {gp.val_rel:.3f}, {rd.val_rel:.3f}; rho estateback {gp.test_rho:.2f}, {rd.test_rho:.2f}. Prediction: 0.22-0.35, val < 0.22, rho < 0.53.",
         f"**Q4:** POD-GP overshoot {gp.overshoot:+.1f} counts (G {gtr:.1f}); projection floor bias {M[M.model == 'POD projection floor'].test_bias.iloc[0]:+.1f}, Rel L2 {Pfl.rel.mean():.3f}.", "",
         "## Models", "", md(M.round(4).to_dict("records")), "", "## Selection on the validation cars", "", md(S.round(4).to_dict("records")), "",
         "## Reference correction", "", md(C.round(3).to_dict("records")), ""]
    (OUT / "pod_verdict.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:10]))


if __name__ == "__main__":
    main()
