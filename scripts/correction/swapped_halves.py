"""Swapped-halves study, task T1: stage B (frozen + region_const_intmatch, settings of the two-stage correction and component and cross-model studies unchanged) with the roles of the two halves of the 250 test cars swapped.
perm = default_rng(0).permutation(sorted(test)); original pool perm[:125], original evaluation perm[125:]; swapped: evaluation = perm[:125] (z indices 0..124), pool = perm[125:] (z indices 125..249);
draw d, k: pool[default_rng(1100 + d).permutation(125)[:k]] (same permutation for every k), 50 draws, k in {1, 2, 3, 5, 10, 25}.
Output: results/swapped_halves/T1/{T1_swap_draws.csv, T1_swap_summary.csv, T1_verdict.md}; split lists in data/splits/swap/. Self-check: the same code with the original halves reproduces the stored original numbers of one draw per family.
Usage: python scripts/correction/swapped_halves.py [--draws 50]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.correction.calibration import build_stats, corrected  # noqa: E402
from scripts.correction.frozen_draws import fit_method, jackknife  # noqa: E402
from scripts.correction.cross_model import DIRS, PAIRS  # noqa: E402
from scripts.diagnosis.regions import labeler as labeler_v2  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

KS = (1, 2, 3, 5, 10, 25)
DIRS = {**DIRS, "TSV": [f"{{f}}_TSV_R1_s{s}" for s in range(3)]}
PAIRS = PAIRS[:6] + [("E2A", "TSV")] + PAIRS[6:]  # the seven E2A families, then B3 on E2C
OUT = ROOT / "results" / "swapped_halves" / "T1"
SLOPE, C = 1.302, float(np.sqrt(2 / np.pi))
ORIG = pd.read_csv(ROOT / "results" / "transolver" / "T5" / "T2_generality_p7.csv")  # original setting: six families + Transolver (E2A), B3 on E2C


def stats(fold, fam):
    perm = [str(i) for i in np.random.default_rng(0).permutation(sorted(load_split(fold)["test"]))]
    dirs = [ROOT / "results" / d.format(f=fold) for d in DIRS[fam]]
    preds = [np.load(d / "test_pred_fp16.npy", mmap_mode="r") for d in dirs]
    pos = [{i: j for j, i in enumerate(pd.read_csv(d / "test_per_car.csv").id)} for d in dirs]
    return perm, build_stats(perm, lambda s, sid: preds[s][pos[s][sid]], len(dirs), labeler_v2, 8)


def row(z, idx, ev, eL0, k, d):
    dL, dD, rel = corrected(z, ev, fit_method(z, idx, "intmatch_lam"))
    eL = dL.mean(0)
    r = {"k": k, "draw": d, "bias_L": eL.mean(), "abs_bias_L": abs(eL.mean()), "abs_bias_D": abs(dD.mean(0).mean()), "rmse_L": float(eL.std()), "rel_l2": rel.mean(), "frac": abs(eL.mean()) / abs(eL0.mean())}
    if k >= 5:
        r["coverage"], r["half"] = jackknife(z, idx, ev, "intmatch_lam")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=50)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "splits" / "swap").mkdir(parents=True, exist_ok=True)
    rows = []
    for fold, fam in PAIRS:
        perm, z = stats(fold, fam)
        pool, ev = np.arange(125, 250), np.arange(125)
        if (fold, fam) == ("E2A", "M2"):
            assert abs(z["dL"].mean(0).mean() - 183.9) <= 0.5  # gate
        # self-check: original halves, one draw, k = 10 reproduce the stored original value
        ev_o = np.arange(125, 250)
        eL0_o = corrected(z, ev_o, ("none", None))[0].mean(0)
        r_o = row(z, np.random.default_rng(100).permutation(125)[:10], ev_o, eL0_o, 10, 0)
        ref = ORIG[(ORIG.fold == fold) & (ORIG.family == fam) & (ORIG.k == 10) & (ORIG.draw == 0)].abs_bias_L.iloc[0]
        assert abs(r_o["abs_bias_L"] - ref) < 1e-3, (fold, fam, r_o["abs_bias_L"], ref)
        e0 = corrected(z, ev, ("none", None))
        eL0 = e0[0].mean(0)
        rows.append({"fold": fold, "family": fam, "k": 0, "draw": 0, "bias_L": eL0.mean(), "abs_bias_L": abs(eL0.mean()), "abs_bias_D": abs(e0[1].mean(0).mean()), "rmse_L": float(eL0.std()), "rel_l2": e0[2].mean(), "frac": 1.0})
        for d in range(a.draws):
            for k in KS:
                idx = 125 + np.random.default_rng(1100 + d).permutation(125)[:k]
                rows.append({"fold": fold, "family": fam, **row(z, idx, ev, eL0, k, d)})
        print(fold, fam, "done", flush=True)
        if (fold, fam) == PAIRS[0]:  # split lists (identical for every family of a fold)
            pass
    R = pd.DataFrame(rows)
    R.round(5).to_csv(OUT / "T1_swap_draws.csv", index=False)
    for fold in ("E2A", "E2C"):
        perm = [str(i) for i in np.random.default_rng(0).permutation(sorted(load_split(fold)["test"]))]
        pd.DataFrame({"role": ["pool (new) = original evaluation"] * 125 + ["evaluation (new) = original candidates"] * 125, "id": perm[125:] + perm[:125]}).to_csv(ROOT / "data" / "splits" / "swap" / f"{fold}_swap_pool_evaluation.csv", index=False)
        dr = [{"draw": d, "k_order": " ".join(perm[125:][j] for j in np.random.default_rng(1100 + d).permutation(125)[:25])} for d in range(a.draws)]
        pd.DataFrame(dr).to_csv(ROOT / "data" / "splits" / "swap" / f"{fold}_swap_draws.csv", index=False)
    summarise(R)


def summarise(R):
    O = ORIG.copy()
    rows = []
    for fold, fam in PAIRS:
        s0 = R[(R.fold == fold) & (R.family == fam) & (R.k == 0)].rmse_L.iloc[0]
        o0 = O[(O.fold == fold) & (O.family == fam) & (O.k == 0)].rmse_L.iloc[0]
        for k in KS:
            s = R[(R.fold == fold) & (R.family == fam) & (R.k == k)]
            o = O[(O.fold == fold) & (O.family == fam) & (O.k == k)]
            rows.append({"fold": fold, "family": fam, "k": k, "sigma_orig": o0, "sigma_swap": s0, "bias_orig": o.abs_bias_L.mean(), "bias_swap": s.abs_bias_L.mean(), "ratio_orig": o.abs_bias_L.mean() / (SLOPE * C * o0 / np.sqrt(k)),
                         "ratio_swap": s.abs_bias_L.mean() / (SLOPE * C * s0 / np.sqrt(k)), "rel_orig": o.rel_l2.mean(), "rel_swap": s.rel_l2.mean(), "cov_orig": o.coverage.median() if k >= 5 else np.nan, "cov_swap": s.coverage.median() if k >= 5 else np.nan,
                         "dbias_orig": o.abs_bias_D.mean(), "dbias_swap": s.abs_bias_D.mean(), "rmse_orig": o.rmse_L.mean(), "rmse_swap": s.rmse_L.mean()})
    S = pd.DataFrame(rows)
    S.round(4).to_csv(OUT / "T1_swap_summary.csv", index=False)
    e = S[S.fold == "E2A"]
    q1 = e[e.k == 10].bias_swap.mean()
    v1 = "repeats" if 15 <= q1 <= 32 else "does not repeat" if q1 > 40 else "partly"
    cells = e[e.k.isin([5, 10, 25])]
    inr = ((cells.ratio_swap >= 0.7) & (cells.ratio_swap <= 1.4))
    f2 = float(inr.mean())
    v2 = "repeats" if f2 >= 0.8 else "does not repeat" if f2 < 0.5 else "partly"
    cv = e[e.k == 10].cov_swap
    n3 = int(((cv >= 0.85) & (cv <= 0.95)).sum())
    v3 = "repeats" if n3 >= 5 else "partly" if n3 >= 3 else "does not repeat"
    fmt = lambda d: md([{c: (f"{v:.3f}" if isinstance(v, float) else v) for c, v in r.items()} for r in d.to_dict("records")])
    show = S[S.k.isin([5, 10, 25])][["fold", "family", "k", "bias_orig", "bias_swap", "ratio_orig", "ratio_swap", "rel_orig", "rel_swap", "cov_orig", "cov_swap", "sigma_orig", "sigma_swap"]].copy()
    show[["bias_orig", "bias_swap", "sigma_orig", "sigma_swap"]] = show[["bias_orig", "bias_swap", "sigma_orig", "sigma_swap"]].round(1)
    L = ["# Swapped-halves study, task T1: swap of the candidate and evaluation halves (frozen + region_const_intmatch)", "",
         "Original: pool perm[:125], evaluation perm[125:], draws default_rng(100 + d). Swap: pool perm[125:], evaluation perm[:125], draws default_rng(1100 + d). 50 draws; |bias| C_L,p in counts (mean over draws); obs/pred = obs / (1.302 sqrt(2/pi) sigma_pop / sqrt(k)) with sigma_pop of the frozen model on the evaluation set of each setting; coverage = median over draws of jackknife+; Rel L2 mean over draws.", "",
         "## Original and swapped setting side by side (k = 5, 10, 25)", "", fmt(show), "",
         "## Criteria of PREREG.md", "",
         f"- **Q1** (mean over the seven E2A families of |bias| at k = 10; original {e[e.k == 10].bias_orig.mean():.1f}, range {e[e.k == 10].bias_orig.min():.1f} to {e[e.k == 10].bias_orig.max():.1f}): swapped {q1:.1f} (range {e[e.k == 10].bias_swap.min():.1f} to {e[e.k == 10].bias_swap.max():.1f}) -> **{v1}** ([15; 32] repeats, > 40 does not).",
         f"- **Q2** (cells with obs/pred in [0.7; 1.4], seven E2A families x k in {{5, 10, 25}}): swapped {int(inr.sum())} of {len(cells)} ({100 * f2:.0f}%; original {int((((cells.ratio_orig >= 0.7) & (cells.ratio_orig <= 1.4))).sum())} of {len(cells)}) -> **{v2}** (>= 80% repeats, < 50% does not).",
         f"- **Q3** (families with median coverage at k = 10 in [0.85; 0.95]): swapped {n3} of 7 (original {int(((e[e.k == 10].cov_orig >= 0.85) & (e[e.k == 10].cov_orig <= 0.95)).sum())} of 7) -> **{v3}** (>= 5 repeats; definition added: 3 or 4 partly, <= 2 does not).",
         "- Prediction: Q1, Q2 and Q3 all repeat.", ""]
    (OUT / "T1_verdict.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[-6:]))


if __name__ == "__main__":
    main()
