"""Label-budget study, task T3 analysis: B-same versus B-OOF on the same draws. Input: results/label_budget/T3/raw/p5_s0_k10_d*.json (Kaggle kernel output).
Output: results/label_budget/T3/T3_draws.csv, T3_stats.csv, T3_report.md. Usage: python scripts/correction/oof_vs_in_sample.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402

OUT = ROOT / "results" / "label_budget" / "T3"


def main():
    rows = []
    for d in range(10):
        f = OUT / "raw" / f"p5_s0_k10_d{d}.json"
        if not f.exists():
            print("missing draw", d)
            continue
        r = json.load(open(f))
        z = np.load(ROOT / "results" / "two_stage" / "T1" / "raw" / f"full_s0_k10_d{d}.npz")  # Two-stage correction study seed 0, same draw (Modal)
        old_nob, old_b = z["noB_full"][:, 0], z["B_full"][:, 0]
        row = {"draw": d, "sel_steps": r["sel_steps"], "lam_same": r["lam_same"], "lam_oof": r["lam_oof"], "in_sample_sd": r["in_sample_dL_sd"], "in_sample_mean": r["in_sample_dL_mean"], "cov_same": r["cov_same"], "half_same": r["half_same"],
               "p3_noB_abs_bias_L": abs(old_nob.mean()), "p3_B_abs_bias_L": abs(old_b.mean())}
        for key, pre in (("noB", "noB"), ("B_same", "same"), ("B_OOF", "oof")):
            for m in ("abs_bias_L", "bias_L", "abs_bias_D", "rel_l2", "rmse_L"):
                row[f"{pre}_{m}"] = r[key][m]
        rows.append(row)
    D = pd.DataFrame(rows)
    D.round(5).to_csv(OUT / "T3_draws.csv", index=False)
    n = len(D)
    diff = D.same_abs_bias_L - D.oof_abs_bias_L
    p = float(wilcoxon(diff, alternative="greater").pvalue) if n >= 5 and np.any(diff != 0) else float("nan")
    p_rel = float(wilcoxon(D.same_rel_l2 - D.oof_rel_l2, alternative="greater").pvalue) if n >= 5 else float("nan")
    st = {"n_draws": n, "median_diff_same_minus_oof": float(diff.median()), "mean_diff_same_minus_oof": float(diff.mean()), "wilcoxon_p_one_sided": p, "draws_oof_lower": int((diff > 0).sum()), "median_cov_same": float(D.cov_same.median()),
          "median_same_abs_bias_L": float(D.same_abs_bias_L.median()), "median_oof_abs_bias_L": float(D.oof_abs_bias_L.median()), "p_rel_l2_same_greater": p_rel,
          "repro_mean_abs_diff_noB": float((D.noB_abs_bias_L - D.p3_noB_abs_bias_L).abs().mean()), "repro_corr_noB": float(np.corrcoef(D.noB_abs_bias_L, D.p3_noB_abs_bias_L)[0, 1]) if n > 2 else float("nan")}
    pd.DataFrame([{"statistic": k, "value": v} for k, v in st.items()]).round(6).to_csv(OUT / "T3_stats.csv", index=False)
    c1, c2, c3 = (p < 0.05), (st["median_diff_same_minus_oof"] > 30), (st["median_cov_same"] < 0.5)
    L = ["# Label-budget study, task T3: B-same versus B-OOF on the same draws (k = 10, `full`, seed 0)", "",
         "Fine-tune `full` (lr 1e-5, step selection as the two-stage correction study) on all 10 cars of each shared draw, weights saved (Kaggle, `kaggle/notebook_oof_vs_in_sample.py`). B-same: stage B (`region_const_intmatch`, 8 regions v2, lambda by LOO) fitted on the in-sample residuals of that model; B-OOF: Two-stage correction study offsets of the same draw and seed (fitted on the out-of-fold residuals of five fold models, Modal). Both applied to the new final model, 125 evaluation cars. "
         "Environment: the new fine-tunes ran on Kaggle (Python 3.13, torch 2.11), the B-OOF offsets come from Modal runs (Two-stage correction study).", "",
         md([{"draw": int(r.draw), "steps": int(r.sel_steps), "noB |bias|": f"{r.noB_abs_bias_L:.1f}", "B-same |bias|": f"{r.same_abs_bias_L:.1f}", "B-OOF |bias|": f"{r.oof_abs_bias_L:.1f}", "B-same Rel L2": f"{r.same_rel_l2:.4f}", "B-OOF Rel L2": f"{r.oof_rel_l2:.4f}",
                 "B-same |bias| C_D": f"{r.same_abs_bias_D:.1f}", "B-OOF |bias| C_D": f"{r.oof_abs_bias_D:.1f}", "in-sample SD": f"{r.in_sample_sd:.1f}", "B-same coverage": f"{r.cov_same:.2f}", "lambda same / OOF": f"{r.lam_same:g} / {r.lam_oof:g}"} for _, r in D.iterrows()]), "",
         f"Draws completed: {n} of 10.", "",
         md([{"registered criterion": "B-OOF |bias| lower than B-same, Wilcoxon paired one-sided, p < 0.05", "result": f"p = {p:.4f}; OOF lower in {st['draws_oof_lower']} of {n} draws", "outcome": "YES" if c1 else "NO"},
             {"registered criterion": "prediction: median difference (B-same minus B-OOF) > 30 counts", "result": f"{st['median_diff_same_minus_oof']:.1f} counts (mean {st['mean_diff_same_minus_oof']:.1f})", "outcome": "TRUE" if c2 else "FALSE"},
             {"registered criterion": "prediction: median coverage of B-same < 0.5", "result": f"{st['median_cov_same']:.3f}", "outcome": "TRUE" if c3 else "FALSE"}]), "",
         f"Medians: |bias| of B-same {st['median_same_abs_bias_L']:.1f}, of B-OOF {st['median_oof_abs_bias_L']:.1f} counts; field Rel L2 of B-same minus B-OOF: one-sided Wilcoxon p = {p_rel:.3f} (descriptive). "
         f"Reproducibility of the fine-tune without stage B against the two-stage correction study (seed 0, same draws, Modal): mean absolute difference of |bias| {st['repro_mean_abs_diff_noB']:.1f} counts, correlation {st['repro_corr_noB']:.2f} (GPU runs are not bit-reproducible and the environment differs).", ""]
    (OUT / "T3_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[-12:]))


if __name__ == "__main__":
    main()
