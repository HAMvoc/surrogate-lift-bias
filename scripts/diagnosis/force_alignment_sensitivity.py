"""Sensitivity of the M2 force-alignment numbers (rho_L, ||e||, C_L,p bias) to the Cp clip range of the reference and to the IPW weights (naive w = area).
Per-car values are averaged over the 3 seeds, then mean over cars with bootstrap CI (B = 2000). Output: results/force_alignment/S1_sensitivity.md
Usage: python scripts/diagnosis/force_alignment_sensitivity.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.force_alignment_terms import OUT, ci, fm, log  # noqa: E402
from scripts.diagnosis.force_alignment import car_terms, load_pool  # noqa: E402

RES = ROOT / "results"
VARIANTS = {"clip [-10, 1], IPW (used)": dict(clip=(-10, 1), ipw=True), "clip [-8, 1], IPW": dict(clip=(-8, 1), ipw=True), "clip [-15, 1], IPW": dict(clip=(-15, 1), ipw=True),
            "no clip, IPW": dict(clip=(-np.inf, np.inf), ipw=True), "clip [-10, 1], naive weights (w = area)": dict(clip=(-10, 1), ipw=False)}


def main():
    log("command: python scripts/diagnosis/force_alignment_sensitivity.py")
    rows = []
    for fold in ("E2A", "E2B", "E2C"):
        runs = [(np.load(RES / f"{fold}_M2_R2_s{s}" / "test_pred_fp16.npy", mmap_mode="r"), {i: n for n, i in enumerate(pd.read_csv(RES / f"{fold}_M2_R2_s{s}" / "test_per_car.csv").id)}) for s in (0, 1, 2)]
        ids = sorted(runs[0][1])
        acc = {v: [] for v in VARIANTS}
        for sid in ids:
            _, nrm, cp, area = load_pool(sid)
            for v, kw in VARIANTS.items():
                ts = [car_terms(pr[pos[sid]], cp, nrm, area, sid, **kw) for pr, pos in runs]
                acc[v].append({k: np.mean([t[k] for t in ts]) for k in ("rho_L", "e_norm", "dL")})
        for v, lst in acc.items():
            d = pd.DataFrame(lst)
            rows.append({"fold": fold, "variant": v, "mean rho_L [CI]": fm(ci(d.rho_L)), "mean ||e||_A [CI]": fm(ci(d.e_norm), 3, False), "bias C_L,p (counts) [CI]": fm(ci(d.dL), 1)})
        log(f"sensitivity {fold} done")
    txt = ["# S1: sensitivity of the M2 alignment numbers to the Cp clip range and the IPW weights", "",
           "M2, 3 seeds averaged per car (n = 250 cars per fold), bootstrap 95% CI over cars. The model predictions are unchanged; only the reference clip (applied to the CFD Cp) or the weights change. 'no clip' keeps the CFD spikes "
           "(37 cars of the 750 have Cp outside [-10, 1.02], reports/cp_outliers.csv). The predictions were trained on Cp clipped to [-10, 1].", "", md(rows), ""]
    (OUT / "S1_sensitivity.md").write_text("\n".join(txt), encoding="utf-8")


if __name__ == "__main__":
    main()
