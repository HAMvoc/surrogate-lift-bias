"""Decomposition and first corrections study, task T6: `loc` zero-shot on E2A (3 seeds, run on Modal) against the registered thresholds (PREREG T6). Output: results/decomposition/T6/T6_report.md
Usage: python scripts/interventions/local_descriptors.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.force_alignment import terms_table  # noqa: E402
from scripts.diagnosis.regions import REGIONS2, labeler  # noqa: E402

OUT, RAW = ROOT / "results" / "decomposition" / "T6", ROOT / "results" / "decomposition" / "T6" / "modal"
BASE_VAL = 0.1551  # PREREG: mean of the 3 M2 seeds
B = 2000


def main():
    loc = terms_table("E2A", "test", [("loc", s, RAW / "results" / f"E2A_M2_R2_s{s}_loc") for s in range(3)], labeler=labeler, region_names=REGIONS2)
    base = terms_table("E2A", "test", [("M2", s, ROOT / "results" / f"E2A_M2_R2_s{s}") for s in range(3)], labeler=labeler, region_names=REGIONS2)
    assert abs(base.groupby("id").dL.mean().mean() - 183.9) <= 0.5, "gate"
    rng = np.random.default_rng(0)
    exp = pd.concat([pd.read_csv(f) for f in sorted(RAW.glob("experiments_*.csv"))])
    vals = exp.set_index("run_id").best_val
    val = float(vals.mean())
    rows, per = [], {}
    for name, T in (("M2 (baseline)", base), ("M2 + loc", loc)):
        g = T.groupby("id").mean(numeric_only=True)
        e = g.dL.values
        idx = rng.integers(0, len(e), (B, len(e)))
        ci = np.percentile(e[idx].mean(1), [2.5, 97.5])
        per[name] = g
        rows.append({"model": name, "bias C_L,p [CI]": f"{e.mean():+.1f} [{ci[0]:+.1f}; {ci[1]:+.1f}]", "RMSE after bias removal": f"{np.sqrt(((e - e.mean()) ** 2).mean()):.1f}", "bias C_D,p": f"{g.dD.mean():+.1f}",
                     "by seed (bias L)": " / ".join(f"{T[T.seed == s].dL.mean():+.1f}" for s in range(3))})
    bias = per["M2 + loc"].dL.mean()
    drel = val - BASE_VAL
    if bias <= 95 and drel <= 0.005: cls = "SUCCESS"
    elif bias <= 95: cls = "PARTIAL (bias <= 95 but validation Rel L2 worsens by more than 0.005)"
    elif bias <= 138: cls = "PARTIAL"
    elif bias <= 155: cls = "UNCLASSIFIED (138-155)"
    else: cls = "NO EFFECT (> 155)"
    reg = pd.DataFrame({n: g[[f"reg_dL_{r}" for r in REGIONS2]].mean().values for n, g in per.items()}, index=REGIONS2).round(1) if all(f"reg_dL_{r}" in per["M2 + loc"].columns for r in REGIONS2) else None
    L = ["# Decomposition and first corrections study, task T6: `loc` zero-shot on E2A (3 seeds, Modal T4)", "",
         "Runs: `E2A_M2_R2_s{0,1,2}_loc` (`--local-feats`, same config as the original M2 runs otherwise), 150 epochs each, trained and tested on Modal (`modal_run/train_modal.py`) because the Kaggle weekly quota was used up. Torch 2.10 / Python 3.11 on Modal vs the Kaggle/laptop builds of the baseline: a software difference to keep in mind. Gate: baseline M2 3-seed bias = +183.9: PASS.", "",
         md(rows), "",
         f"Validation Rel L2 (50 F/N cars, best epoch of each run, from experiments csv): {', '.join(f'{v:.5f}' for v in vals.values)}; mean {val:.4f} vs registered baseline {BASE_VAL} -> change {drel:+.4f} (limit +0.005).", "",
         f"Registered thresholds: success if bias <= 95 and val change <= +0.005; partial 95-138 (or bias <= 95 with worse val); no effect > 155; 138-155 unclassified. Result: bias {bias:+.1f}, val change {drel:+.4f} -> **{cls}**.", ""]
    if reg is not None:
        L += ["Bias C_L,p by region v2 (counts):", "", md([{"region": r, **{n: reg.loc[r, n] for n in reg.columns}} for r in REGIONS2]), ""]
    G = 112.8
    over = bias - G
    L += [f"Overshoot decomposition (T4 convention): G = +{G} (fixed by the split); overshoot of `loc` = bias - G = {over:+.1f} (M2: +71.1).", ""]
    (OUT / "T6_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
