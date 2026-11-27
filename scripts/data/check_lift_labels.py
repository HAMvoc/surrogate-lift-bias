"""Safety-margin and cumulative-lift study C: cheap checks of the lift labels. C1: standard error of the group means and of G from the iteration spread Std Cl of the
original DrivAerNet table (an upper bound of per-car label noise). C2: per body type, the car whose 23 design parameters are closest to the zero offset (distance scaled by the parameter range).
Output: results/margin_and_cumulative_lift/{labels_c1.csv, labels_c2.csv, labels_verdict.md}. Usage: python scripts/data/check_lift_labels.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402

OUT = ROOT / "results" / "margin_and_cumulative_lift"
PARAMS = (ROOT / ".." / "DrivAerNet" / "ParametricModels" / "DrivAerNet_ParametricData.csv").resolve()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cars = pd.read_csv(ROOT / "results" / "classical_baselines" / "cars.csv").set_index("sample_id")
    tab = pd.read_csv(PARAMS).set_index("Experiment").loc[cars.index]
    assert len(tab) == 750
    pcols = list(tab.columns[:23])
    assert "Average Cd" not in pcols and pcols[-1] == "A_Car_Green_House_Angle"
    std = 1000 * tab["Std Cl"]
    tr, te = (cars.role == "train").to_numpy(), (cars.role == "test").to_numpy()
    G = cars.CL[tr].mean() - cars.CL[te].mean()
    assert abs(G - 112.755) < 0.01
    se = lambda m: float(np.sqrt((std[m] ** 2).mean() / m.sum()))
    se_g = float(np.sqrt(se(tr) ** 2 + se(te) ** 2))
    c1 = pd.DataFrame([("n_train", tr.sum()), ("n_test", te.sum()), ("rms_std_train", float(np.sqrt((std[tr] ** 2).mean()))), ("rms_std_test", float(np.sqrt((std[te] ** 2).mean()))),
                       ("se_mean_test", se(te)), ("se_mean_train", se(tr)), ("se_G", se_g), ("G", G), ("G_over_se", G / se_g)], columns=["statistic", "value"])
    c1.to_csv(OUT / "labels_c1.csv", index=False)
    P = tab[pcols]
    d = np.sqrt((((P - 0) / (P.max() - P.min())) ** 2).sum(1))
    rows = []
    for t in ("Fastback", "Notchback", "Estateback"):
        m = (cars.category == t).to_numpy()
        sid = d[m].idxmin()
        rows.append({"category": t, "sample_id": sid, "distance": d[sid], "median_distance_type": float(d[m].median()), "CL_pool": cars.CL[sid], "Cl_table": 1000 * tab.at[sid, "Average Cl"],
                     "Cl_f_table": 1000 * tab.at[sid, "Average Cl_f"], "Cl_r_table": 1000 * tab.at[sid, "Average Cl_r"], "Cd_table": 1000 * tab.at[sid, "Average Cd"], "CD_pool": cars.CD[sid],
                     "mean_CL_pool_type": cars.CL[m].mean(), "mean_Cl_table_type": 1000 * tab["Average Cl"][m].mean()})
    c2 = pd.DataFrame(rows)
    c2.round(4).to_csv(OUT / "labels_c2.csv", index=False)
    L = ["# Safety-margin and cumulative-lift study C: cheap checks of the lift labels (criteria: `PREREG.md`, part C)", "",
         f"**C1:** root-mean-square Std Cl {c1.value[3]:.1f} counts (estateback test cars), {c1.value[2]:.1f} (training cars); standard error of the mean of the {int(te.sum())} estateback cars "
         f"{c1.value[4]:.2f} counts; of G {se_g:.2f} counts; G / SE(G) = {G / se_g:.0f}. Prediction: about 1.6, about 2, above 40.", "",
         "**C2:** nearest-to-zero-offset car per body type (counts; pool = pressure part of this study, table = total lift of the original DrivAerNet campaign):", "", md(c2.round(3).to_dict("records")), ""]
    (OUT / "labels_verdict.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
