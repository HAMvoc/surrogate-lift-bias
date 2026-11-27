"""Zero-shot checks study, task T2: underbody descriptors and extended oracle R2 (ridge, 10-fold CV by car, 10 repeats) for the M2 lift bias on E2A.
Targets: Delta C_L,p total and of the underbody region (M2, 3-seed mean per car). Writes results/zero_shot_checks/T2/{T2_descriptors.csv, T2_oracle_R2.md, T2_chosen_theta.txt}.
Usage: python scripts/interventions/underbody_descriptors.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.force_alignment_terms import cv_r2, theta_rear  # noqa: E402
from scripts.diagnosis.pressure_coefficients import pool_weights  # noqa: E402
from scripts.diagnosis.force_alignment import REGIONS, load_pool, region_labels  # noqa: E402

OUT = ROOT / "results" / "zero_shot_checks" / "T2"
L_M = 5.0  # cache coordinates are in units of L_ref = 5 m
PARAMS = ROOT.parent / "DrivAerNet" / "ParametricModels" / "DrivAerNet_ParametricData.csv"
PARAM_SEL = ["B_Ramp_Angle", "B_Diffusor_Angle", "B_Trunklid_Angle", "D_Rear_Window_Inclination", "G_Trunklid_Length", "G_Trunklid_Curvature", "A_Car_Roof_Height", "A_Car_Green_House_Angle"]


def theta_under(sid):
    xyz, nrm, _, area = load_pool(sid)
    lab = region_labels(xyz, nrm)
    w = pool_weights(np.asarray(area, np.float64), sid)
    x, y, z = (xyz[:, k].astype(np.float64) * L_M for k in range(3))  # metres
    L = x.max() - x.min()
    xr, yr = (x - x.min()) / L, np.abs(y) / np.abs(y).max()
    ub = (lab == REGIONS.index("underbody")) & (yr < 0.5)  # centre strip: keeps the wheels out of the profile
    # lowest-surface profile in 2%-of-length bins
    edges = np.arange(0.0, 1.0001, 0.02)
    cen, zmin = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = ub & (xr >= a) & (xr < b)
        if s.sum() >= 5:
            cen.append((a + b) / 2)
            zmin.append(np.percentile(z[s], 2))  # 2nd percentile of z: the 'lowest' point without single-point outliers
    cen, zmin = np.array(cen), np.array(zmin)
    prof = lambda lo, hi: zmin[(cen >= lo) & (cen <= hi)]
    front, rear = prof(0.15, 0.30), prof(0.70, 0.85)
    c_f, c_r = float(front.mean()), float(rear.mean())
    rake = float(np.degrees(np.arctan2(c_r - c_f, (0.775 - 0.225) * L)))
    dm = (cen >= 0.80) & (cen <= 0.95)
    diff = float(np.degrees(np.arctan(np.polyfit(cen[dm] * L, zmin[dm], 1)[0]))) if dm.sum() >= 3 else np.nan
    mid = (cen >= 0.20) & (cen <= 0.80)
    pf = np.polyfit(cen[mid] * L, zmin[mid], 1)
    flat = float(1000 * np.std(zmin[mid] - np.polyval(pf, cen[mid] * L)))
    return {"clear_front_m": c_f, "clear_rear_m": c_r, "rake_deg": rake, "diffuser_deg": diff, "flatness_mm": flat, "underbody_area_frac": float(w[lab == REGIONS.index("underbody")].sum() / w.sum())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(ROOT / "results" / "force_alignment" / "per_car_terms.csv.gz")
    g = T[(T.fold == "E2A") & (T.model == "M2")].drop(columns=["seed", "role"]).groupby("id").mean(numeric_only=True).sort_index()
    gate = float(g.dL.mean())
    assert abs(gate - 183.9) <= 0.5, f"gate failed: {gate}"
    ids = list(g.index)
    tr = pd.DataFrame([theta_rear(i) for i in ids], index=ids)
    tu = pd.DataFrame([theta_under(i) for i in ids], index=ids)
    n_missing = int(pd.concat([tr, tu], axis=1).isna().any(axis=1).sum())
    tr, tu = tr.fillna(tr.median()), tu.fillna(tu.median())
    P = pd.read_csv(PARAMS).set_index("Experiment")
    assert all(i in P.index for i in ids), "design parameters missing for some E2A cars"
    pcols = [c for c in P.columns if not (c.startswith("Average") or c.startswith("Std"))]
    par_all = P.loc[ids, pcols]
    pd.concat([tr, tu, par_all[PARAM_SEL].add_prefix("param_")], axis=1).round(5).to_csv(OUT / "T2_descriptors.csv")
    Xp = g[[f"reg_predL_{r}" for r in REGIONS]].values
    targets = {"Delta C_L,p total": g.dL.values, "Delta C_L,p underbody region": g.reg_dL_underbody.values}
    sets = {"theta_rear (4)": tr.values, "theta_under (6)": tu.values, "theta_rear + theta_under (10)": np.hstack([tr.values, tu.values]),
            "theta_param (8 design parameters)": par_all[PARAM_SEL].values, "theta_param_all (23 design parameters)": par_all.values}
    rows = []
    res = {}
    for tname, y in targets.items():
        for sname, X in sets.items():
            for var, XX in (("", X), (" + M2 predicted regional lift integrals (7)", np.hstack([X, Xp]))):
                r = cv_r2(XX, y)
                res[(tname, sname + var)] = r
                rows.append({"target": tname, "features": sname + var, "CV R2 mean": f"{r[0]:+.3f}", "min": f"{r[1]:+.3f}", "max": f"{r[2]:+.3f}"})
        r = cv_r2(Xp, y)
        res[(tname, "M2 predicted regional lift integrals only (7)")] = r
        rows.append({"target": tname, "features": "M2 predicted regional lift integrals only (7)", "CV R2 mean": f"{r[0]:+.3f}", "min": f"{r[1]:+.3f}", "max": f"{r[2]:+.3f}"})
    # registered selection among the three surface-derived sets on the total target
    cand = {"theta_rear (4)": 4, "theta_under (6)": 6, "theta_rear + theta_under (10)": 10}
    r2 = {k: res[("Delta C_L,p total", k)][0] for k in cand}
    best = max(r2.values())
    near = [k for k in cand if r2[k] >= best - 0.03]
    chosen = min(near, key=lambda k: (cand[k], -r2[k]))
    (OUT / "T2_chosen_theta.txt").write_text(chosen, encoding="utf-8")
    corr = pd.DataFrame({t: pd.concat([tr, tu], axis=1).apply(lambda c: np.corrcoef(c, y)[0, 1]) for t, y in targets.items()}).round(2)
    txt = ["# T2: underbody descriptors and oracle R2 (M2, E2A, 250 Estateback cars)", "",
           f"Gate: M2 3-seed mean bias C_L,p on E2A from the data used here = {gate:+.2f} counts (required 183.9 +- 0.5): PASS. Target SDs across cars: total {g.dL.std():.1f} counts, underbody region {g.reg_dL_underbody.std():.1f} counts.", "",
           "Design parameters: the 23 DrivAerNet parameters are not in `data/manifests/`; they are in `DrivAerNet/ParametricModels/DrivAerNet_ParametricData.csv` (configs/data.yaml `source.params_csv`; DrivAerNet v1 repository), which covers all 250 cars. "
           "It has no ride-height or rake parameter. They are used as reference sets only (not eligible for the T3 choice, PREREG).", "",
           "Note on PREREG: it says theta_under has 5 features but lists six items (front and rear clearance counted once); six features are used.", "",
           "## Definitions (all from the label-free surface pool; coordinates in metres = cache units x 5; x_rel = (x - x_min)/L; centre strip = |y|/max|y| < 0.5, which keeps the wheels out)", "",
           "- Underbody points: region `underbody` of `force_alignment.region_labels` (n_z < -0.3, not base) in the centre strip. Lowest-surface profile: 2nd percentile of z in 2%-of-length bins (bins with at least 5 points).",
           "- clear_front_m, clear_rear_m: mean of the profile for x_rel in [0.15, 0.30] and [0.70, 0.85].",
           "- rake_deg: atan2(clear_rear - clear_front, (0.775 - 0.225) L) in degrees (positive: rear higher).",
           "- diffuser_deg: slope angle of a straight-line fit to the profile for x_rel in [0.80, 0.95] (positive: rising to the rear).",
           "- flatness_mm: standard deviation (mm) of the profile residual about its straight-line fit for x_rel in [0.20, 0.80].",
           "- underbody_area_frac: IPW-weighted share of surface in the underbody region (any |y|).",
           "- theta_rear: as in the force-alignment study (base_area_frac, flat_roof_len, rear_upper_slope_deg, rear_taper).",
           f"- Cars with a missing surface descriptor (median-imputed): {n_missing}.", "",
           "Descriptor summary (mean, SD across the 250 cars) and correlation with the targets:", "",
           md([{"descriptor": c, "mean": f"{pd.concat([tr, tu], axis=1)[c].mean():.4g}", "SD": f"{pd.concat([tr, tu], axis=1)[c].std():.3g}", **{f"corr {t}": f"{corr.loc[c, t]:+.2f}" for t in targets}} for c in corr.index]), "",
           "## Oracle R2 (ridge, standardised features, RidgeCV alphas 1e-3..1e3, 10-fold CV by car, 10 repeats; mean [min; max] over repeats)", "",
           "R2 is about the car-to-car variation of the target, not about its mean. M2-predicted regional lift integrals use predictions only (no labels).", "",
           md(rows), "",
           f"## Registered choice of theta for T3 (total target, three surface-derived sets, differences below 0.03 not interpreted -> fewer features): **{chosen}**", "",
           md([{"set": k, "features": cand[k], "CV R2": f"{r2[k]:+.3f}"} for k in cand]), ""]
    (OUT / "T2_oracle_R2.md").write_text("\n".join(txt), encoding="utf-8")
    print("chosen:", chosen, {k: round(v, 3) for k, v in r2.items()})


if __name__ == "__main__":
    main()
