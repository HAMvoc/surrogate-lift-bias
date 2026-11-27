"""Transolver study, task T4: stage B (frozen + region_const_intmatch, 8 v2 regions, lambda by LOO) and the out-of-sample test of the floor law on Transolver (PREREG.md and addendum A).
The pipeline of scripts/correction/cross_model.py is re-used unchanged for a new family: E2A, 250 test cars permuted as in the two-stage correction study, candidates perm[:125], evaluation perm[125:], draw d = cand[default_rng(100 + d).permutation(125)[:k]], 50 draws, k in {1, 2, 3, 5, 10, 25}.
The model is the 3-seed average prediction per car. Predicted |bias| = SLOPE * sqrt(2/pi) * sigma / sqrt(k), sigma = RMSE after bias removal of the frozen model on the 125 evaluation cars, SLOPE = 1.302 (Number-source study, not re-estimated).
Verdict at k = 5, 10, 25: all ratios obs/pred in [0.7, 1.4] -> fits; any outside [0.5, 2.0] -> does not fit; else partly.
  python scripts/correction/transolver_stage_b.py [--runs "results/E2A_TSV_R1_s{s}"] [--out results/transolver/T4]    (--runs "results/E2A_M2_R2_s{s}" is the stand-in check on M2)"""
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
from scripts.diagnosis.regions import labeler as labeler_v2  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

KS = (1, 2, 3, 5, 10, 25)
SLOPE, C = 1.302, float(np.sqrt(2 / np.pi))


def run_family(dirs, perm):
    preds = [np.load(d / "test_pred_fp16.npy", mmap_mode="r") for d in dirs]
    pos = [{i: j for j, i in enumerate(pd.read_csv(d / "test_per_car.csv").id)} for d in dirs]
    z = build_stats(perm, lambda s, sid: preds[s][pos[s][sid]], len(dirs), labeler_v2, 8)
    ev = np.arange(125, 250)
    e0 = corrected(z, ev, ("none", None))
    eL0 = e0[0].mean(0)
    rows = [{"k": 0, "draw": 0, "bias_L": eL0.mean(), "abs_bias_L": abs(eL0.mean()), "abs_bias_D": abs(e0[1].mean(0).mean()), "rmse_L": float(eL0.std()), "rel_l2": e0[2].mean(), "frac": 1.0}]
    for d in range(50):
        for k in KS:
            idx = np.random.default_rng(100 + d).permutation(125)[:k]
            dL, dD, rel = corrected(z, ev, fit_method(z, idx, "intmatch_lam"))
            eL = dL.mean(0)
            row = {"k": k, "draw": d, "bias_L": eL.mean(), "abs_bias_L": abs(eL.mean()), "abs_bias_D": abs(dD.mean(0).mean()), "rmse_L": float(eL.std()), "rel_l2": rel.mean(), "frac": abs(eL.mean()) / abs(eL0.mean())}
            if k >= 5:
                row["coverage"], row["half"] = jackknife(z, idx, ev, "intmatch_lam")
            rows.append(row)
    return z, pd.DataFrame(rows)


def floor(R):
    s = R[R.k == 0].rmse_L.iloc[0]
    return {k: {"obs": R[R.k == k].abs_bias_L.mean(), "pred": SLOPE * C * s / np.sqrt(k), "sigma": s} for k in KS}


def verdict(F):
    r = [F[k]["obs"] / F[k]["pred"] for k in (5, 10, 25)]
    return r, ("fits" if all(0.7 <= x <= 1.4 for x in r) else "does not fit" if any(not 0.5 <= x <= 2.0 for x in r) else "partly")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/E2A_TSV_R1_s{s}")
    ap.add_argument("--out", default=str(ROOT / "results" / "transolver" / "T4"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    perm = [str(i) for i in np.random.default_rng(0).permutation(sorted(load_split("E2A")["test"]))]
    dirs = [ROOT / a.runs.format(s=s) for s in range(3)]
    z, R = run_family(dirs, perm)
    R.round(5).to_csv(out / "T4_draws.csv", index=False)
    b0 = R[R.k == 0].iloc[0]
    F = floor(R)
    ratios, v = verdict(F)
    ps = []
    for s in range(3):  # robustness: each seed alone (not in the verdict)
        _, Rs = run_family([dirs[s]], perm)
        Fs = floor(Rs)
        ps.append({"seed": s, **{f"ratio k={k}": f"{Fs[k]['obs'] / Fs[k]['pred']:.2f}" for k in (5, 10, 25)}, "verdict": verdict(Fs)[1], "sigma": f"{Fs[5]['sigma']:.1f}"})
    pd.DataFrame([{"k": k, "obs": F[k]["obs"], "sigma": F[k]["sigma"], "pred": F[k]["pred"], "ratio": F[k]["obs"] / F[k]["pred"]} for k in KS]).to_csv(out / "T4_floor.csv", index=False)
    pd.DataFrame([{"k": k, **{c: float(R[R.k == k][c].mean()) for c in ("abs_bias_L", "abs_bias_D", "rel_l2", "rmse_L", "frac") + (("coverage", "half") if k >= 5 else ())}} for k in (0,) + KS]).to_csv(out / "T4_summary.csv", index=False)
    t = []
    for k in KS:
        g = R[R.k == k]
        t.append({"k": k, "|bias| C_L,p": f"{g.abs_bias_L.mean():.1f} [{g.abs_bias_L.quantile(.05):.1f}; {g.abs_bias_L.quantile(.95):.1f}]", "|bias| C_D,p": f"{g.abs_bias_D.mean():.1f}", "Rel L2": f"{g.rel_l2.mean():.4f}", "RMSE": f"{g.rmse_L.mean():.1f}",
                  "bias fraction left": f"{g.frac.mean():.3f}", "jackknife+ coverage": f"{g.coverage.mean():.3f}" if k >= 5 else "-", "half-width": f"{g.half.mean():.0f}" if k >= 5 else "-"})
    f = [{"k": k, "obs |bias|": f"{F[k]['obs']:.1f}", "sigma": f"{F[k]['sigma']:.1f}", f"pred = {SLOPE} sqrt(2/pi) sigma/sqrt(k)": f"{F[k]['pred']:.1f}", "obs/pred": f"{F[k]['obs'] / F[k]['pred']:.2f}"} for k in KS]
    L = ["# Transolver study, task T4: stage B and the floor law on Transolver (frozen model, no stage A)", "",
         f"Runs: `{a.runs}` (3 seeds, seed-averaged per car). Frozen + `region_const_intmatch` (8 v2 regions, lambda by LOO in the k cars); 50 fixed draws; evaluation on 125 cars. Before stage B on those 125 cars: bias C_L,p {b0.bias_L:+.1f}, |bias| C_D,p {b0.abs_bias_D:.1f}, Rel L2 {b0.rel_l2:.4f}, RMSE after bias removal (sigma) {b0.rmse_L:.1f}.", "",
         "## Stage B (mean [5%; 95%] over draws)", "", md(t), "", "## Floor law, out of sample (slope 1.302 from the number-source study, not re-estimated)", "", md(f), "",
         f"Ratios at k = 5, 10, 25: {', '.join(f'{x:.2f}' for x in ratios)}. Criteria: all inside [0.7; 1.4] fits; any outside [0.5; 2.0] does not fit; otherwise partly. **Verdict: {v}.** Prediction was: fits.", "",
         "Robustness (each seed alone as the model; not in the verdict):", "", md(ps), ""]
    (out / "T4_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
