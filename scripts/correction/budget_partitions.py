"""Partition-level budget study: partition-level test of the label-budget relation (Q1), replay of the sequential procedure (Q2, Q3) and consistency of the pool-integrated lift with the dataset tables (B);
Per-car errors of the seven E2A models come from scripts/correction/swapped_halves.py:stats (seed-mean prediction minus CFD C_L,p, counts).
Output: results/budget_partitions/{budget_cells.csv, replay_cells.csv, dataset_check.csv, verdict.md}. Usage: python scripts/correction/budget_partitions.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.correction.swapped_halves import PAIRS, stats  # noqa: E402

OUT = ROOT / "results" / "budget_partitions"
KS = (2, 3, 5, 10, 25)
TAUS = (10, 20, 28)
P, K0, Z = 500, 5, 1.645
SQ = np.sqrt(2 / np.pi)
PARAMS = (ROOT / "configs" / "../../DrivAerNet/ParametricModels/DrivAerNet_ParametricData.csv").resolve()


def errors():
    E = {}
    for fold, fam in PAIRS:
        if fold != "E2A":
            continue
        perm, z = stats(fold, fam)
        E[fam] = z["dL"].mean(0)
        if fam == "M2":
            assert abs(E[fam].mean() - 183.9) <= 0.5  # gate
    assert len(E) == 7 and all(len(v) == 250 for v in E.values())
    return E


def partitions():
    return [np.random.default_rng(3700 + p).permutation(250) for p in range(P)]


def budget(E, parts):
    rows = []
    for fam, e in E.items():
        b = np.zeros((P, len(KS))); pred = np.zeros((P, len(KS)))
        for p, perm in enumerate(parts):
            pool, ev = perm[:125], perm[125:]
            sig = float(np.sqrt(np.mean((e[ev] - e[ev].mean()) ** 2)))
            for j, k in enumerate(KS):
                lab = np.random.default_rng(370000 + 1000 * p + k).choice(pool, k, replace=False)
                b[p, j] = abs(e[ev].mean() - e[lab].mean())
                pred[p, j] = SQ * sig / np.sqrt(k)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, P, (2000, P))
        for j, k in enumerate(KS):
            c = b[:, j].sum() / pred[:, j].sum()
            boot = b[idx, j].sum(1) / pred[idx, j].sum(1)
            rows.append({"family": fam, "k": k, "c_hat": c, "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5), "mean_b": b[:, j].mean(), "mean_pred": pred[:, j].mean()})
    return pd.DataFrame(rows)


def replay(E, parts):
    rows = []
    for fam, e in E.items():
        res = {t: ([], []) for t in TAUS}
        for p, perm in enumerate(parts):
            pool, ev = perm[:125], perm[125:]
            order = pool[np.random.default_rng(380000 + p).permutation(125)]
            sh = e[order[:K0]].std(ddof=1)
            for t in TAUS:
                ks = int(np.clip(np.ceil((Z * sh / t) ** 2), K0, 125))
                res[t][0].append(ks); res[t][1].append(abs(e[ev].mean() - e[order[:ks]].mean()))
        for t in TAUS:
            ks, bs = np.array(res[t][0]), np.array(res[t][1])
            rows.append({"family": fam, "tau": t, "coverage": float((bs <= t).mean()), "k_med": float(np.median(ks)), "k_p5": float(np.percentile(ks, 5)), "k_p95": float(np.percentile(ks, 95)), "b_med": float(np.median(bs))})
    return pd.DataFrame(rows)


def dataset():
    cars = pd.read_csv(ROOT / "results" / "classical_baselines" / "cars.csv").set_index("sample_id")
    pc = pd.read_csv(PARAMS).set_index("Experiment").loc[cars.index]
    d = cars.assign(cl_tab=1000 * pc["Average Cl"], cd_tab=1000 * pc["Average Cd"], std_cl=1000 * pc["Std Cl"])
    rows = [{"scope": "all 750 cars", "r_cl": np.corrcoef(d.CL, d.cl_tab)[0, 1], "r_cd": np.corrcoef(d.CD, d.cd_tab)[0, 1]}]
    for c in ("Fastback", "Notchback", "Estateback"):
        s = d[d.category == c]
        rows.append({"scope": c, "r_cl": np.corrcoef(s.CL, s.cl_tab)[0, 1], "r_cd": np.corrcoef(s.CD, s.cd_tab)[0, 1], "mean_cl_pool": s.CL.mean(), "mean_cl_table": s.cl_tab.mean(), "mean_std_cl": s.std_cl.mean()})
    tr, te = d[d.role == "train"], d[d.role == "test"]
    rows.append({"scope": "G (train minus estateback)", "g_pool": tr.CL.mean() - te.CL.mean(), "g_table": tr.cl_tab.mean() - te.cl_tab.mean()})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    E = errors(); parts = partitions()
    B, R, D = budget(E, parts), replay(E, parts), dataset()
    B.round(4).to_csv(OUT / "budget_cells.csv", index=False); R.round(4).to_csv(OUT / "replay_cells.csv", index=False); D.round(4).to_csv(OUT / "dataset_check.csv", index=False)
    inr = int(((B.c_hat >= 0.7) & (B.c_hat <= 1.4)).sum())
    q1 = "holds at partition level" if inr >= 24 else "does not hold"
    x, y = np.log(B.mean_pred), np.log(B.mean_b)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    meets = int((R.coverage >= 0.85).sum())
    q2 = "procedure meets the target" if meets >= 14 else "does not meet it"
    L = ["# Partition-level budget study: partition-level budget test, sequential replay, dataset check (criteria: `PREREG.md`)", "",
         "Self-check passed: M2 gate (+183.9 counts).", "",
         f"**Q1 verdict: {q1}** - {inr} of {len(B)} cells have c_hat in [0.7, 1.4] (threshold 24). Range of c_hat {B.c_hat.min():.2f} to {B.c_hat.max():.2f}; centred R^2 of the log-log fit over the cells {r2:.3f}; SD of log c_hat {np.log(B.c_hat).std(ddof=1):.3f}. Prediction: holds, c_hat 1.0 to 1.1.", "",
         f"**Q2 verdict: {q2}** - {meets} of {len(R)} cells reach coverage >= 0.85 (threshold 14). Prediction: meets at tau 20 and 28, fails at tau 10.", "",
         "## Q1 cells (c_hat with 95% bootstrap interval over partitions)", "", md(B.round(3).to_dict("records")), "",
         "## Q2 cells (coverage, median k*, 5th and 95th percentile of k*)", "", md(R.round(3).to_dict("records")), "",
         "## B: pool-integrated lift against the dataset tables (counts)", "", md(D.round(3).fillna("").to_dict("records")), ""]
    (OUT / "verdict.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:9]))


if __name__ == "__main__":
    main()
