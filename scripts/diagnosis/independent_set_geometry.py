"""Independent-set study, task T1: geometry of the new Estateback cars against the 250 old test cars: the descriptors theta_rear (4) and theta_under (6) of the force-alignment and zero-shot checks studies (computed from the pools) and the eight design parameters
of the zero-shot checks study T2 (DrivAerNet_ParametricData.csv). Description only (mean, SD, standardised difference of the means, two-sample Kolmogorov-Smirnov). No pressure label is used by these quantities.
Output: results/independent_set/T1/geometry_comparison.csv and geometry_report.md. Usage: python scripts/diagnosis/independent_set_geometry.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis import pressure_coefficients as cda  # noqa: E402
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.force_alignment_terms import theta_rear  # noqa: E402
from scripts.interventions.underbody_descriptors import PARAM_SEL, PARAMS, theta_under  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

OUT = ROOT / "results" / "independent_set" / "T1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = sorted(load_split("E2A")["test"])
    new = sorted(pd.read_csv(ROOT / "data" / "splits" / "fresh_estateback.csv").sample_id)
    g = pd.read_csv(ROOT / "data" / "manifests" / "car_geometry_fresh.csv").set_index("sample_id")
    for sid in new:
        cda.GEO.loc[sid, ["area_total_m2", "frontal_area_m2", "incl_rate_per_m2"]] = g.loc[sid, ["area_total_m2", "frontal_area_m2", "incl_rate_per_m2"]].to_numpy(float)
    feat = lambda ids: pd.DataFrame([{**theta_rear(i), **theta_under(i)} for i in ids], index=ids)
    fo, fn = feat(old), feat(new)
    par = pd.read_csv(PARAMS).set_index("Experiment")[PARAM_SEL]
    fo, fn = pd.concat([fo, par.loc[old].set_index(fo.index)], axis=1), pd.concat([fn, par.loc[new].set_index(fn.index)], axis=1)
    rows = []
    for c in fo.columns:
        a, b = fo[c].dropna(), fn[c].dropna()
        ks = ks_2samp(a, b)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append({"descriptor": c, "mean_old": a.mean(), "sd_old": a.std(ddof=1), "mean_new": b.mean(), "sd_new": b.std(ddof=1), "std_diff": (b.mean() - a.mean()) / pooled if pooled > 0 else np.nan, "ks_stat": ks.statistic, "ks_p": ks.pvalue})
    T = pd.DataFrame(rows)
    T.round(4).to_csv(OUT / "geometry_comparison.csv", index=False)
    wh = lambda ids: pd.Series([i.split("_")[2] for i in ids]).value_counts().to_dict()
    L = ["# Independent-set study, task T1: geometry of the new cars against the 250 old test cars", "", f"Old: {len(old)} cars (wheel codes {wh(old)}); new: {len(new)} cars (wheel codes {wh(new)}).", "",
         md([{k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()} for r in T.to_dict("records")]), "",
         f"Largest absolute standardised difference of the means: {T.std_diff.abs().max():.2f} ({T.loc[T.std_diff.abs().idxmax(), 'descriptor']}); descriptors with a Kolmogorov-Smirnov p-value below 0.05: {int((T.ks_p < 0.05).sum())} of {len(T)} (description; no multiple-testing correction).", ""]
    (OUT / "geometry_report.md").write_text("\n".join(L), encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main()
