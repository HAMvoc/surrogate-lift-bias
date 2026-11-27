"""Decomposition and first corrections study, task T2: car configuration codes (wheel variant) and the between-car variation of the M2 lift bias on E2A.
Needs results/decomposition/T1/per_car_terms_v2.csv.gz (regional_bias.py). Output: results/decomposition/T2/.
Usage: python scripts/diagnosis/wheel_configuration.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.force_alignment_terms import ci, cv_r2, fm  # noqa: E402
from scripts.diagnosis.regions import REGIONS2  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

OUT = ROOT / "results" / "decomposition" / "T2"
PARAMS = ROOT.parent / "DrivAerNet" / "ParametricModels" / "DrivAerNet_ParametricData.csv"
STYLE = {"E": "Estateback", "F": "Fastback", "N": "Notchback"}
B = 2000


def parts(sid):
    a = sid.split("_")
    return a[0], a[1], a[2], a[3]  # style, underbody, wheels, mirrors


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = ["# Decomposition and first corrections study, task T2: configuration codes and the between-car variation (E2A)", ""]
    # 1. counts
    s1 = load_split("E1")
    allids = sorted(set(s1["train"]) | set(s1["val"]) | set(s1["test"]))
    cfg = pd.DataFrame([dict(zip(("style", "underbody", "wheels", "mirrors"), parts(i)), id=i) for i in allids])
    L += ["The code of a car id is `<style>_<underbody>_<wheels>_<mirrors>_<number>`. In the 750-car subset the underbody code is always S and the mirror code always WM; the wheel code has three values (WW, WWC, WWS). "
          "Earlier analyses (`scripts/diagnosis/wheel_and_bias.py`, the force-alignment study and the audit tables) distinguished only WW and WWS and counted WWC as WW; this analysis separates the three. The meaning of the codes was not found in the DrivAerNet++ sources "
          "I could read (a web search found only that the dataset has open and closed, smooth and detailed wheel variants), so only the codes are used.", "",
          f"Underbody codes: {sorted(cfg.underbody.unique())}; mirror codes: {sorted(cfg.mirrors.unique())}.", ""]
    rows = []
    for name, ids in (("E1 train", s1["train"]), ("E1 val", s1["val"]), ("E1 test", s1["test"])) + tuple((f"{f} {r}", load_split(f)[r]) for f in ("E2A", "E2B", "E2C") for r in ("train", "test")):
        c = pd.Series([f"{parts(i)[0]}_{parts(i)[2]}" for i in ids]).value_counts()
        rows.append({"split": name, "cars": len(ids), **{k: int(c.get(k, 0)) for k in sorted({f"{a}_{b}" for a, b in zip(cfg['style'], cfg['wheels'])})}})
    L += ["## Counts by (body style, wheel code) in each split", "", md(rows), ""]
    # 2. E2A analysis
    T = pd.read_csv(ROOT / "results" / "decomposition" / "T1" / "per_car_terms_v2.csv.gz")
    g = T[(T.fold == "E2A") & (T.model == "M2")].drop(columns=["seed", "role"]).groupby("id").mean(numeric_only=True).sort_index()
    assert abs(g.dL.mean() - 183.9) <= 0.5
    wc = pd.Series({i: parts(i)[2] for i in g.index})
    P = pd.read_csv(PARAMS).set_index("Experiment")
    pcols = [c for c in P.columns if not (c.startswith("Average") or c.startswith("Std"))]
    Xp = P.loc[list(g.index), pcols].values
    Xw = pd.get_dummies(wc).astype(float).values
    y = g.dL.values
    sets = {"wheel code (one-hot)": Xw, "23 design parameters": Xp, "wheel code + 23 design parameters": np.hstack([Xw, Xp])}
    targets = {"Delta C_L,p total": y, "wheels region": g.reg_dL_wheels.values, "underbody region (new)": g.reg_dL_underbody.values}
    r2 = {}
    rows = []
    for tn, yy in targets.items():
        for sn, X in sets.items():
            r = cv_r2(X, yy)
            r2[(tn, sn)] = r
            rows.append({"target": tn, "features": sn, "CV R2 mean [min; max]": f"{r[0]:+.3f} [{r[1]:+.3f}; {r[2]:+.3f}]"})
    L += [f"## E2A: wheel codes in the 250 Estateback test cars: {dict(wc.value_counts())}", "", "Oracle R2 (ridge, standardised features, 10-fold CV by car, 10 repeats) of the M2 (3-seed mean) lift bias:", "", md(rows), ""]
    # bias / RMSE by wheel code
    rows = []
    rng = np.random.default_rng(0)
    for code, gg in g.groupby(wc):
        e = gg.dL.values
        rows.append({"wheel code": code, "n": len(e), "bias C_L,p [CI]": fm(ci(e), 1), "SD (RMSE after bias removal)": f"{np.sqrt(((e - e.mean()) ** 2).mean()):.1f}", "MAE after bias removal": f"{np.abs(e - e.mean()).mean():.1f}",
                     "wheels region": f"{gg.reg_dL_wheels.mean():+.1f}", "underbody region": f"{gg.reg_dL_underbody.mean():+.1f}", "roof_rear region": f"{gg.reg_dL_roof_rear.mean():+.1f}"})
    codes = sorted(wc.unique())
    diff = None
    if len(codes) == 2:
        a, b = g.dL[wc == codes[0]].values, g.dL[wc == codes[1]].values
        d = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(B)]
        diff = (a.mean() - b.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5))
    L += ["Bias and spread by wheel code (M2, E2A):", "", md(rows), ""]
    if diff:
        L += [f"Difference of the mean bias {codes[0]} - {codes[1]}: {diff[0]:+.1f} counts [{diff[1]:+.1f}; {diff[2]:+.1f}] (bootstrap over cars).", ""]
    # variance shares from the new regions
    v = g.dL.var(ddof=1)
    sh = {r: 100 * np.cov(g[f"reg_dL_{r}"].values, y)[0, 1] / v for r in REGIONS2}
    rr = r2[("Delta C_L,p total", "wheel code (one-hot)")][0]
    wu = sh["wheels"] + sh["underbody"]
    if rr >= 0.30 and wu >= 70:
        verdict = "TRUE"
    elif rr < 0.10:
        verdict = "FALSE"
    else:
        verdict = "PARTLY TRUE"
    follows = diff is not None and abs(diff[0]) > 30 and (diff[1] > 0 or diff[2] < 0)
    L += ["## Registered hypothesis: 'most of the between-car variation lies in the wheels and the underbody, and follows the wheel configuration'", "",
          f"Between-car variance shares (new regions): wheels {sh['wheels']:.0f}%, underbody {sh['underbody']:.0f}% (together {wu:.0f}%). CV R2 of the wheel code alone for the total bias: {rr:+.3f}. Wheel-code bias difference > 30 counts with CI excluding 0: {'yes' if follows else 'no'}. "
          f"Rule: TRUE if R2 >= 0.30 and wheels + underbody >= 70%; FALSE if R2 < 0.10; otherwise PARTLY TRUE. **Verdict: {verdict}.**", ""]
    (OUT / "T2_report.md").write_text("\n".join(L), encoding="utf-8")
    cfg.to_csv(OUT / "T2_config_codes.csv", index=False)
    print("verdict", verdict, rr, wu)


if __name__ == "__main__":
    main()
