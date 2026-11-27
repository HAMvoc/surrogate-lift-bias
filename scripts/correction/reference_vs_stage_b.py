"""Reference-correction study: stage B (frozen + region_const_intmatch, stored numbers) against the minimal baseline scalar_shift on the same cars and draws.
Settings: orig (pool perm[:125], evaluation perm[125:], draws default_rng(100 + d)), swap (pool perm[125:], evaluation perm[:125], default_rng(1100 + d)), indep (pool = 250 old E2A test cars,
evaluation = the 250 new estateback cars of the independent-set study, default_rng(2100 + d)). Only scalar_shift is computed; stage B is read from the stored per-draw files, after a self-check that the same
code reproduces stage B on draw 0, k = 10 of every cell. Output: results/reference_correction/{shift_draws.csv, paired.csv, verdict.md}. Usage: python scripts/correction/reference_vs_stage_b.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis import pressure_coefficients as cda  # noqa: E402
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.correction.calibration import build_stats, corrected  # noqa: E402
from scripts.correction.frozen_draws import fit_method, jackknife  # noqa: E402
from scripts.correction.swapped_halves import PAIRS, stats  # noqa: E402
from scripts.training.infer_cars import FAMS  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

OUT = ROOT / "results" / "reference_correction"
KS = (1, 2, 3, 5, 10, 25)
TOL = 3.0  # counts, PREREG Q1
STORED = {"orig": ROOT / "results" / "transolver" / "T5" / "T2_generality_p7.csv", "swap": ROOT / "results" / "swapped_halves" / "T1" / "T1_swap_draws.csv",
          "indep": ROOT / "results" / "independent_set" / "T2" / "Q2_draws.csv"}


def one(z, idx, ev, method, k):
    dL, dD, rel = corrected(z, ev, fit_method(z, idx, method))
    r = {"abs_bias_L": abs(dL.mean(0).mean()), "abs_bias_D": abs(dD.mean(0).mean()), "rel_l2": rel.mean()}
    if k >= 5:
        r["coverage"], r["half"] = jackknife(z, idx, ev, method)
    return r


def run_cell(setting, fold, fam, z, pool, ev, seed0, stored, rows):
    ref = stored[(stored.k == 10) & (stored.draw == 0)]
    ref = ref[(ref.family == fam) & ((ref.fold == fold) if "fold" in ref else True)].abs_bias_L.iloc[0]
    got = one(z, pool[np.random.default_rng(seed0).permutation(len(pool))[:10]], ev, "intmatch_lam", 4)["abs_bias_L"]
    assert abs(got - ref) < 1e-3, (setting, fold, fam, got, ref)  # self-check against the frozen stage B number
    base = corrected(z, ev, ("none", None))
    rows.append({"setting": setting, "fold": fold, "family": fam, "k": 0, "draw": 0, "abs_bias_L": abs(base[0].mean(0).mean()), "abs_bias_D": abs(base[1].mean(0).mean()), "rel_l2": base[2].mean()})
    for d in range(50):
        perm = np.random.default_rng(seed0 + d).permutation(len(pool))
        for k in KS:
            rows.append({"setting": setting, "fold": fold, "family": fam, "k": k, "draw": d, **one(z, pool[perm[:k]], ev, "scalar_shift", k)})
    print(setting, fold, fam, "done", flush=True)


def indep_stats(fam):
    """Stats over 250 old E2A test cars (stored predictions) + 250 new cars (Independent-set study predictions), as in scripts/diagnosis/independent_set_test.py."""
    old = [str(i) for i in np.random.default_rng(0).permutation(sorted(load_split("E2A")["test"]))]
    new = sorted(pd.read_csv(ROOT / "data" / "splits" / "fresh_estateback.csv").sample_id)
    g = pd.read_csv(ROOT / "data" / "manifests" / "car_geometry_fresh.csv").set_index("sample_id")
    cols = ["area_total_m2", "frontal_area_m2", "incl_rate_per_m2"]
    for sid in new:
        cda.GEO.loc[sid, cols] = g.loc[sid, cols].to_numpy(float)
    lo, ln = np.load(ROOT / "results" / "decomposition" / "T1" / "labels_v2.npz"), np.load(ROOT / "results" / "independent_set" / "labels_v2_fresh.npz")
    old_set = set(old)
    lab = lambda sid, xyz, nrm: (lo[sid] if sid in old_set else ln[sid]).astype(np.int64)
    src = lambda root: [(np.load(root / l / "test_pred_fp16.npy", mmap_mode="r"), {i: j for j, i in enumerate(pd.read_csv(root / l / "test_per_car.csv").id)}) for l in FAMS[fam]]
    a_old, a_new = src(ROOT / "results"), src(ROOT / "results" / "independent_set" / "preds")

    def get_pred(s, sid):
        p, pos = (a_old if sid in old_set else a_new)[s]
        return p[pos[sid]]
    return len(old), len(new), build_stats(old + new, get_pred, len(FAMS[fam]), lab, 8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stored = {k: pd.read_csv(v) for k, v in STORED.items()}
    rows = []
    for fold, fam in PAIRS:
        _, z = stats(fold, fam)
        if (fold, fam) == ("E2A", "M2"):
            assert abs(z["dL"].mean(0).mean() - 183.9) <= 0.5  # gate
        run_cell("orig", fold, fam, z, np.arange(125), np.arange(125, 250), 100, stored["orig"], rows)
        run_cell("swap", fold, fam, z, np.arange(125, 250), np.arange(125), 1100, stored["swap"], rows)
    for fam in FAMS:
        n_old, n_new, z = indep_stats(fam)
        run_cell("indep", "E2A", fam, z, np.arange(n_old), np.arange(n_old, n_old + n_new), 2100, stored["indep"], rows)
    S = pd.DataFrame(rows)
    S.round(5).to_csv(OUT / "shift_draws.csv", index=False)
    summarise(S, stored)


def summarise(S, stored):
    B = pd.concat([v.assign(setting=k, fold=v["fold"] if "fold" in v else "E2A") for k, v in stored.items()])
    B = B[B.k > 0]
    m = S[S.k > 0].merge(B, on=["setting", "fold", "family", "k", "draw"], suffixes=("_shift", "_B"))
    assert len(m) == len(S[S.k > 0]), (len(m), len(S[S.k > 0]))
    rows = []
    for (st, fold, fam, k), g in m.groupby(["setting", "fold", "family", "k"], sort=False):
        r = {"setting": st, "fold": fold, "family": fam, "k": k, "n_draws": len(g)}
        for q in ("L", "D"):
            d = (g[f"abs_bias_{q}_B"] - g[f"abs_bias_{q}_shift"]).to_numpy()
            pB = wilcoxon(d, alternative="less").pvalue if np.any(d) else 1.0
            pS = wilcoxon(d, alternative="greater").pvalue if np.any(d) else 1.0
            lab = "B lower" if d.mean() < -TOL and pB < 0.05 else "shift lower" if d.mean() > TOL and pS < 0.05 else "no clear difference" if abs(d.mean()) <= TOL else "unclear"
            r.update({f"B_{q}": g[f"abs_bias_{q}_B"].mean(), f"shift_{q}": g[f"abs_bias_{q}_shift"].mean(), f"diff_{q}": d.mean(), f"p_B_lower_{q}": pB, f"p_shift_lower_{q}": pS, f"label_{q}": lab})
        r.update({"rel_B": g.rel_l2_B.mean(), "rel_shift": g.rel_l2_shift.mean(), "cov_B": g.coverage_B.median(), "cov_shift": g.coverage_shift.median(), "half_B": g.half_B.median(), "half_shift": g.half_shift.median()})
        rows.append(r)
    P = pd.DataFrame(rows)
    P.round(5).to_csv(OUT / "paired.csv", index=False)
    c10 = P[P.k == 10]
    counts = c10.label_L.value_counts()
    need = int(np.ceil(2 / 3 * len(c10)))
    verdict = next((l for l in ("B lower", "shift lower", "no clear difference") if counts.get(l, 0) >= need), "mixed")
    fmt = lambda d, cols: md(d[cols].round(3).to_dict("records"))
    L = ["# Reference-correction study: stage B against scalar_shift (criteria: `PREREG.md`)", "",
         f"Cells at k = 10: {len(c10)}; label counts for |bias| C_L,p: {dict(counts)}; threshold {need} of {len(c10)}.", "",
         f"**Q1 verdict: {verdict}** (prediction: no clear difference).", "",
         "## k = 10 (means over 50 draws; diff = stage B minus scalar_shift, counts; coverage and half-width: median over draws)", "",
         fmt(c10, ["setting", "fold", "family", "B_L", "shift_L", "diff_L", "p_B_lower_L", "label_L", "B_D", "shift_D", "diff_D", "label_D", "rel_B", "rel_shift", "cov_B", "cov_shift", "half_B", "half_shift"]), ""]
    for k in (5, 25):
        L += [f"## k = {k}", "", fmt(P[P.k == k], ["setting", "fold", "family", "B_L", "shift_L", "diff_L", "label_L", "B_D", "shift_D", "label_D", "rel_B", "cov_B", "cov_shift"]), ""]
    (OUT / "verdict.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:5]))


if __name__ == "__main__":
    main()
