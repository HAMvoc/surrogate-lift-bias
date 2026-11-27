"""Swapped-halves study, task T1b: M2 `full` ft_B_OOF (seed 0, k = 10) on the swapped halves, 10 draws.
Reads results/swapped_halves/T1b/raw/full_s0_k10_d{0..9}.npz (copied from the Kaggle kernel outputs), key `B_full` = per-car [error C_L,p, error C_D,p, Rel L2] of the final fine-tuned model plus stage B on the 125 new evaluation cars.
Criterion: mean |bias C_L,p| <= 28 counts and mean Rel L2 <= 0.1736 -> "repeats". Reference: the original (Two-stage correction study) seed-0 runs on the same k, draws 0..9 and the 3-seed result (22.6 counts, 0.1686).
Output: results/swapped_halves/T1b/{T1b_per_draw.csv, T1b_numeric.csv, T1b_report.md}. Usage: python scripts/correction/swapped_halves_two_stage.py"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "swapped_halves" / "T1b"


def metrics(z):
    a = z["B_full"]
    return abs(a[:, 0].mean()), abs(a[:, 1].mean()), a[:, 2].mean()


def main():
    rows = []
    for d in range(10):
        f = OUT / "raw" / f"full_s0_k10_d{d}.npz"
        assert f.exists(), f
        b, bd, r = metrics(np.load(f))
        o = np.load(ROOT / "results" / "two_stage" / "T1" / "raw" / f"full_s0_k10_d{d}.npz")
        ob, obd, orr = metrics(o)
        rows.append({"draw": d, "abs_bias_L": b, "abs_bias_D": bd, "rel_l2": r, "abs_bias_L_orig_s0": ob, "rel_l2_orig_s0": orr})
    T = pd.DataFrame(rows)
    T.round(5).to_csv(OUT / "T1b_per_draw.csv", index=False)
    D = pd.read_csv(ROOT / "results" / "two_stage" / "T1" / "T1_per_draw.csv")
    g = D[(D.variant == "full") & (D.k == 10) & (D.method == "ft_B_OOF (final A)")]
    bm, rm = T.abs_bias_L.mean(), T.rel_l2.mean()
    ok = bm <= 28 and rm <= 0.1736
    pd.DataFrame([("n_draws", len(T)), ("bias_mean", bm), ("rel_l2_mean", rm), ("rel_l2_orig", g.rel_l2.mean()), ("bias_orig", g.abs_bias_L.mean()), ("bias_orig_s0", T.abs_bias_L_orig_s0.mean()), ("rel_l2_orig_s0", T.rel_l2_orig_s0.mean())],
                 columns=["statistic", "value"]).to_csv(OUT / "T1b_numeric.csv", index=False)
    L = ["# Swapped-halves study, task T1b: swapped halves, M2 full ft_B_OOF (seed 0, k = 10, 10 draws)", "",
         f"Mean |bias| C_L,p {bm:.1f} counts (SD over draws {T.abs_bias_L.std(ddof=1):.1f}; range {T.abs_bias_L.min():.1f} to {T.abs_bias_L.max():.1f}); mean Rel L2 {rm:.4f}; mean |bias| C_D,p {T.abs_bias_D.mean():.1f}.",
         f"Original setting, 3 seeds (Two-stage correction study, draws 0..9): {g.abs_bias_L.mean():.1f} counts, Rel L2 {g.rel_l2.mean():.4f}; original seed 0 alone on the same draw indices: {T.abs_bias_L_orig_s0.mean():.1f} counts, Rel L2 {T.rel_l2_orig_s0.mean():.4f}.", "",
         f"**Criterion** (mean |bias| <= 28 and mean Rel L2 <= 0.1736): |bias| {'met' if bm <= 28 else 'not met'}, Rel L2 {'met' if rm <= 0.1736 else 'not met'} -> **{'repeats' if ok else 'does not repeat'}**.", ""]
    (OUT / "T1b_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
