"""Transolver study, task T2 and T3: capability gate on E1 and the Estateback bias of Transolver on E2A.
Reads results/<run>/{test,val}_per_car.csv and test_pred_fp16.npy (copied from the Kaggle kernel output) and uses scripts/diagnosis/force_alignment.py for every integral.
  python scripts/diagnosis/transolver_bias.py T2 [--run results/E1_TSV_R1_s0]
  python scripts/diagnosis/transolver_bias.py T3 [--runs "results/E2A_TSV_R1_s{s}"]     (3 seeds; the same code on "results/E2A_M2_R2_s{s}" reproduces the M2 gate +183.9)
Output: results/transolver/T2/ or T3/ (csv + md)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis import force_alignment as fa  # noqa: E402
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.regions import REGIONS2, labeler as labeler_v2  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

B = 2000
CFD = pd.read_csv(ROOT / "results" / "force_alignment" / "cfd_coefficients.csv").set_index("id")


def ci(x, seed=0):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


fm = lambda t, d=1, s=True: f"{t[0]:{'+' if s else ''}.{d}f} [{t[1]:{'+' if s else ''}.{d}f}; {t[2]:{'+' if s else ''}.{d}f}]"


def t2(run):
    run = ROOT / run
    T = fa.terms_table("E1", "test", [("TSV", 0, run)], regions=False)
    pc = pd.read_csv(run / "test_per_car.csv").set_index("id")
    val = pd.read_csv(run / "val_per_car.csv")
    m2 = pd.read_csv(ROOT / "results" / "force_alignment" / "per_car_terms.csv.gz", usecols=["fold", "role", "model", "seed", "id", "dL", "dD"])  # Force-alignment study table (E1 predictions of M2 were not kept as files)
    m2 = m2[(m2.fold == "E1") & (m2.role == "test") & (m2.model == "M2") & (m2.seed == 0)]
    m2pc = pd.read_csv(ROOT / "results" / "E1_M2_R2_s0" / "test_per_car.csv").set_index("id")
    rows = []
    for name, TT, P in (("Transolver E1 s0", T, pc), ("M2 E1 s0 (reference)", m2, m2pc)):
        for grp, ids in (("all 75", sorted(TT.id)), ("Estateback 25", [i for i in sorted(TT.id) if i[0] == "E"]), ("Fastback 25", [i for i in sorted(TT.id) if i[0] == "F"]), ("Notchback 25", [i for i in sorted(TT.id) if i[0] == "N"])):
            g = TT.set_index("id").loc[ids]
            rows.append({"model": name, "cars": grp, "Rel L2 test": P.loc[ids].rel_l2.mean(), "bias C_L,p": fm(ci(g.dL)), "bias C_D,p": fm(ci(g.dD)), "_L": g.dL.mean(), "_rel": P.loc[ids].rel_l2.mean()})
    R = pd.DataFrame(rows)
    est = R[(R.model == "Transolver E1 s0") & (R.cars == "Estateback 25")].iloc[0]
    allr = R[(R.model == "Transolver E1 s0") & (R.cars == "all 75")].iloc[0]
    ok_rel, ok_b = allr._rel <= 0.20, abs(est._L) <= 40
    out = ROOT / "results" / "transolver" / "T2"
    out.mkdir(parents=True, exist_ok=True)
    R.drop(columns=["_L", "_rel"]).to_csv(out / "T2_e1.csv", index=False)
    pd.DataFrame([{"statistic": "rel_l2_test_E1", "value": allr._rel}, {"statistic": "rel_l2_val_E1", "value": float(val.rel_l2.mean())}, {"statistic": "bias_L_estateback_E1", "value": est._L}, {"statistic": "best_epoch_E1", "value": float(pd.read_csv(run / "log.csv").dropna(subset=["val"]).sort_values("val").epoch.iloc[0])}, {"statistic": "epochs_run_E1", "value": float(pd.read_csv(run / "log.csv").epoch.max())}, {"statistic": "gpu_hours_E1", "value": float(pd.read_csv(run / "log.csv").seconds.sum() / 3600)}]).to_csv(out / "T2_numeric.csv", index=False)
    L = ["# Transolver study, task T2: capability gate on E1 (seed 0)", "",
         f"Gate (PREREG): Rel L2 on the E1 test set <= 0.20 and |bias C_L,p| <= 40 counts on the 25 Estateback test cars. Run: `{run.name}`; validation Rel L2 (75 cars) {val.rel_l2.mean():.4f}.", "",
         md([{k: (f"{v:.4f}" if k == "Rel L2 test" else v) for k, v in r.items() if not k.startswith("_")} for r in rows]), "",
         f"**Rel L2 test E1 = {allr._rel:.4f}: {'PASS' if ok_rel else 'FAIL'} (<= 0.20). |bias C_L,p| on the 25 Estateback cars = {abs(est._L):.1f}: {'PASS' if ok_b else 'FAIL'} (<= 40). Gate: {'PASSED, E2A may run' if ok_rel and ok_b else 'NOT PASSED: stop, do not run E2A'}.**", ""]
    (out / "T2_gate.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def t3(tmpl):
    runs = [("TSV", s, ROOT / tmpl.format(s=s)) for s in range(3)]
    T = fa.terms_table("E2A", "test", runs, regions=True, labeler=labeler_v2, region_names=REGIONS2)
    s = load_split("E2A")
    trL, trD = CFD.loc[s["train"], "CL"].mean(), CFD.loc[s["train"], "CD"].mean()
    ids = sorted(T.id.unique())
    P = {sd: T[T.seed == sd].set_index("id").loc[ids] for sd in range(3)}
    mean = lambda col: np.mean([P[sd][col].values for sd in range(3)], axis=0)
    dL3 = mean("dL")
    bias3 = dL3.mean()
    sgn = np.sign(bias3)
    rows, reg, num = [], [], []
    cfd_te = CFD.loc[ids]
    for sd in list(range(3)) + ["3-seed mean"]:
        if sd == "3-seed mean":
            dL, dD, rho, en = dL3, mean("dD"), mean("rho_L"), mean("e_norm")
            pcs = [pd.read_csv(ROOT / tmpl.format(s=k) / "test_per_car.csv").set_index("id").loc[ids].rel_l2.values for k in range(3)]
            vcs = [pd.read_csv(ROOT / tmpl.format(s=k) / "val_per_car.csv").rel_l2.mean() for k in range(3)]
            rel_t, rel_v = float(np.mean([p.mean() for p in pcs])), float(np.mean(vcs))
            rcols = {r: mean(f"reg_dL_{r}") for r in REGIONS2}
        else:
            g = P[sd]
            dL, dD, rho, en = g.dL.values, g.dD.values, g.rho_L.values, g.e_norm.values
            rel_t = pd.read_csv(ROOT / tmpl.format(s=sd) / "test_per_car.csv").rel_l2.mean()
            rel_v = pd.read_csv(ROOT / tmpl.format(s=sd) / "val_per_car.csv").rel_l2.mean()
            rcols = {r: g[f"reg_dL_{r}"].values for r in REGIONS2}
        pred = cfd_te.CL.values + dL
        over = pred - trL
        G = trL - cfd_te.CL.mean()
        share = float((np.sign(dL) == sgn).mean())
        lo, hi = ci(dL)[1:], None
        num.append({"seed": sd, "bias_L": dL.mean(), "bias_L_lo": lo[0], "bias_L_hi": lo[1], "bias_D": dD.mean(), "bias_D_lo": ci(dD)[1], "bias_D_hi": ci(dD)[2], "same_sign_share": share, "rel_l2_test": rel_t, "rel_l2_val": rel_v, "e_norm": en.mean(), "rho_L": rho.mean(),
                    "G": G, "overshoot": over.mean(), "overshoot_lo": ci(over)[1], "overshoot_hi": ci(over)[2], "slope": np.polyfit(cfd_te.CL.values, pred, 1)[0], "spearman": spearmanr(cfd_te.CL.values, pred)[0], "rmse_after_bias_removal": dL.std(),
                    **{f"reg_{r}": rcols[r].mean() for r in REGIONS2}, "roof_underbody_share_pct": 100 * (rcols["roof_front"].mean() + rcols["roof_rear"].mean() + rcols["underbody"].mean()) / dL.mean()})
        rows.append({"seed": sd, "bias C_L,p [CI]": fm(ci(dL)), "bias C_D,p [CI]": fm(ci(dD)), "same-sign share": f"{share:.3f}", "Rel L2 test": f"{rel_t:.4f}", "Rel L2 val": f"{rel_v:.4f}", "||e||_A": f"{en.mean():.4f}", "rho_L": f"{rho.mean():.3f}",
                     "G": f"{G:+.1f}", "overshoot [CI]": fm(ci(over)), "slope": f"{np.polyfit(cfd_te.CL.values, pred, 1)[0]:.2f}", "Spearman": f"{spearmanr(cfd_te.CL.values, pred)[0]:.3f}", "RMSE after bias removal": f"{dL.std():.1f}",
                     "_bias": dL.mean(), "_share": share, "_over": over.mean(), "_rel": rel_t})
        tot = dL.mean()
        reg.append({"seed": sd, **{r: f"{rcols[r].mean():+.1f} ({100 * rcols[r].mean() / tot:.0f}%)" for r in REGIONS2}, "roof+underbody share": f"{100 * (rcols['roof_front'].mean() + rcols['roof_rear'].mean() + rcols['underbody'].mean()) / tot:.0f}%",
                    "_ru": (rcols['roof_front'].mean() + rcols['roof_rear'].mean() + rcols['underbody'].mean()) / tot})
    R, G3 = pd.DataFrame(rows), pd.DataFrame(reg)
    per = R[R.seed != "3-seed mean"]
    b3, sh3 = float(per._bias.mean()), float(per._share.mean())
    q1 = "same error" if (b3 >= 100 and sh3 >= 0.90) else "does not" if abs(b3) <= 50 else "partly"
    p1 = (120 <= b3 <= 220) and bool(per._over.mean() > 0) and (G3[G3.seed == "3-seed mean"]._ru.iloc[0] >= 0.80)
    out = ROOT / "results" / "transolver" / "T3"
    out.mkdir(parents=True, exist_ok=True)
    R.drop(columns=[c for c in R if c.startswith("_")]).to_csv(out / "T3_table.csv", index=False)
    G3.drop(columns=["_ru"]).to_csv(out / "T3_regions_v2.csv", index=False)
    N = pd.DataFrame(num)
    N["seed"] = N.seed.astype(str)
    N.to_csv(out / "T3_numeric.csv", index=False)
    pd.DataFrame([{"statistic": "mean_seed_bias_L", "value": b3}, {"statistic": "mean_seed_same_sign_share", "value": sh3}, {"statistic": "mean_seed_rel_l2_test", "value": float(per._rel.mean())}, {"statistic": "mean_seed_overshoot", "value": float(per._over.mean())},
                  {"statistic": "bias_L_min_seed", "value": float(per._bias.min())}, {"statistic": "bias_L_max_seed", "value": float(per._bias.max())}]).to_csv(out / "T3_verdict_inputs.csv", index=False)
    L = ["# Transolver study, task T3: Transolver on E2A (3 seeds, 250 Estateback test cars)", "",
         f"Runs: `{tmpl}`. Integrals: `scripts/diagnosis/force_alignment.py` (IPW, clip, A_ref, 100k points). Bias = mean over the 250 cars of predicted minus CFD (counts); CI: car bootstrap B = {B}, percentile. Same-sign share: cars whose C_L,p error has the sign of the 3-seed mean bias ({bias3:+.1f}). "
         "G = mean CFD C_L,p of the 450 training cars minus the 250 test cars; overshoot = mean prediction on the test cars minus the mean CFD of the training cars (bias = G + overshoot). Slope and Spearman: prediction against CFD C_L,p across the 250 cars. M2 reference on E2A: bias +183.9, Rel L2 0.2024.", "",
         md([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]), "", "## Decomposition over the eight v2 regions (counts of C_L,p, share of the total bias)", "", md([{k: v for k, v in r.items() if not k.startswith("_")} for r in reg]), "",
         "## Verdicts (criteria of PREREG.md)", "",
         f"- **Question 1:** mean of the three seed biases {b3:+.1f} counts, mean same-sign share {sh3:.3f} -> **{q1}** (same error: bias >= +100 and share >= 90%; does not: |bias| <= 50; else partly).",
         f"- **Prediction 1** (bias in [+120; +220], positive overshoot, roof + underbody >= 80% of the bias): bias {b3:+.1f} ({'inside' if 120 <= b3 <= 220 else 'outside'}), mean overshoot {per._over.mean():+.1f} ({'positive' if per._over.mean() > 0 else 'not positive'}), roof + underbody share {100 * G3[G3.seed == '3-seed mean']._ru.iloc[0]:.0f}% -> **{'confirmed' if p1 else 'not confirmed'}**.",
         f"- **Question 2 (descriptive):** Rel L2 test, mean of the seeds {per._rel.mean():.4f} (M2 0.2024).", ""]
    (out / "T3_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["T2", "T3"])
    ap.add_argument("--run", default="results/E1_TSV_R1_s0")
    ap.add_argument("--runs", default="results/E2A_TSV_R1_s{s}")
    a = ap.parse_args()
    t2(a.run) if a.task == "T2" else t3(a.runs)
