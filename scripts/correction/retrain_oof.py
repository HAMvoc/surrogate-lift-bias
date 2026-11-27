"""Component and cross-model study, task T3: retrain + stage B fitted on out-of-fold residuals, several draws.
Per (k, draw): final run E2A_M2_R2_s0_cal{k}d{d} (all k cars) on the 125 evaluation cars; five fold runs E2A_M2_R2_s0_cal{k}d{d}f{0..4} give the out-of-fold residuals of the k cars (the held-out fold cars are appended to the test list of
each fold run); stage B = region_const_intmatch (8 regions v2, lambda by LOO) fitted on those residuals and applied to the final run. Inputs: results/two_stage/T2/raw/results (Two-stage correction study runs) and results/components/T3/raw/results (new fold runs).
Outputs: results/components/T3/T3_draws.csv, T3_report.md. Usage: python scripts/correction/retrain_oof.py"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.correction.calibration import build_stats, corrected, metrics  # noqa: E402
from scripts.correction.two_stage_oof import fit_b  # noqa: E402
from scripts.diagnosis.regions import labeler  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

OUT = ROOT / "results" / "components" / "T3"
RAWS = [ROOT / "results" / "two_stage" / "T2" / "raw" / "results", OUT / "raw" / "results"]
B = 2000


def run_dir(name):
    for r in RAWS:
        if (r / name / "test_pred_fp16.npy").exists():
            return r / name
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    perm = [str(i) for i in np.random.default_rng(0).permutation(sorted(load_split("E2A")["test"]))]
    ev_ids = perm[125:]
    rng = np.random.default_rng(0)
    rows = []
    case_err = {}
    for k, draws in ((25, (0, 1, 2, 3)), (10, (0, 1, 2))):
        for d in draws:
            fin = run_dir(f"E2A_M2_R2_s0_cal{k}d{d}")
            folds = [run_dir(f"E2A_M2_R2_s0_cal{k}d{d}f{f}") for f in range(5)]
            if fin is None or any(f is None for f in folds):
                print(f"k={k} d={d}: runs missing (final {fin is not None}, folds {[f is not None for f in folds]})")
                continue
            ids = list(pd.read_csv(fin / "test_per_car.csv").id)
            pr = np.load(fin / "test_pred_fp16.npy", mmap_mode="r")
            pos = {i: j for j, i in enumerate(ids)}
            ze = build_stats(ev_ids, lambda _s, sid: pr[pos[sid]], 1, labeler, 8)
            oof_ids, preds = [], {}
            for r in folds:
                fi = list(pd.read_csv(r / "test_per_car.csv").id)
                pf = np.load(r / "test_pred_fp16.npy", mmap_mode="r")
                for j, i in enumerate(fi):
                    if i not in set(ev_ids):
                        oof_ids.append(i); preds[i] = pf[j]
            assert len(oof_ids) == k and len(set(oof_ids)) == k, (k, d, len(oof_ids))
            zo = build_stats(oof_ids, lambda _s, sid: preds[sid], 1, labeler, 8)
            lam, c = fit_b(zo, np.arange(k))
            for name, fit in (("retrain", ("none", None)), ("retrain + B OOF", ("off", c[None]))):
                m = metrics(ze, np.arange(len(ev_ids)), fit)
                rows.append({"method": name, "k": k, "draw": d, "bias_L": m["bias_L"], "abs_bias_L": m["abs_bias_L"], "abs_bias_D": m["abs_bias_D"], "rmse_L": m["rmse_L"], "rel_l2": m["rel_l2"], "spearman": m["spearman"], "lam": lam if name != "retrain" else np.nan})
                case_err[(name, k, d)] = (m["eL"], corrected(ze, np.arange(len(ev_ids)), fit)[2].mean(0))
            print(f"k={k} d={d} done", flush=True)
    R = pd.DataFrame(rows)
    R.round(5).to_csv(OUT / "T3_draws.csv", index=False)
    # also the retrain (no B) draws 0..4 of the two-stage correction study for the main table
    rt = []
    for k in (10, 25):
        for d in range(5):
            fin = run_dir(f"E2A_M2_R2_s0_cal{k}d{d}")
            if fin is None:
                continue
            ids = list(pd.read_csv(fin / "test_per_car.csv").id)
            pr = np.load(fin / "test_pred_fp16.npy", mmap_mode="r")
            pos = {i: j for j, i in enumerate(ids)}
            ze = build_stats(ev_ids, lambda _s, sid: pr[pos[sid]], 1, labeler, 8)
            m = metrics(ze, np.arange(len(ev_ids)), ("none", None))
            rt.append({"method": "retrain", "k": k, "draw": d, "bias_L": m["bias_L"], "abs_bias_L": m["abs_bias_L"], "abs_bias_D": m["abs_bias_D"], "rmse_L": m["rmse_L"], "rel_l2": m["rel_l2"], "spearman": m["spearman"]})
    pd.DataFrame(rt).round(5).to_csv(OUT / "retrain_draws.csv", index=False)
    if R.empty:
        (OUT / "T3_report.md").write_text("# Component and cross-model study, task T3: no complete (final + 5 folds) draw available yet\n", encoding="utf-8"); return
    P1 = pd.read_csv(ROOT / "results" / "two_stage" / "T1" / "T1_per_draw.csv")
    L = ["# Component and cross-model study, task T3: retrain + stage B on out-of-fold residuals, several draws", "",
         "Final model: retrain run on all k cars (seed 0, 150 epochs); OOF residuals from five fold runs (each held-out fold predicted by the run that did not train on it); stage B = `region_const_intmatch`, lambda by LOO; evaluation on 125 cars not used anywhere. "
         "One seed per run, so the draw-level SD includes seed noise. Environment: Modal T4 (torch 2.10, Python 3.11) for every run in this table. Comparison column: `full` ft_B_OOF of the two-stage correction study (3-seed mean) on the same draw.", ""]
    tab = []
    for k in (10, 25):
        for d in sorted(R[R.k == k].draw.unique()):
            a = R[(R.k == k) & (R.draw == d) & (R.method == "retrain + B OOF")].iloc[0]
            b = R[(R.k == k) & (R.draw == d) & (R.method == "retrain")].iloc[0]
            f = P1[(P1.variant == "full") & (P1.k == k) & (P1.draw == d) & (P1.method == "ft_B_OOF (final A)")].iloc[0]
            tab.append({"k": k, "draw": d, "retrain |bias|": f"{b.abs_bias_L:.1f}", "retrain Rel L2": f"{b.rel_l2:.4f}", "retrain + B |bias|": f"{a.abs_bias_L:.1f}", "retrain + B bias (signed)": f"{a.bias_L:+.1f}", "retrain + B Rel L2": f"{a.rel_l2:.4f}", "RMSE": f"{a.rmse_L:.1f}",
                         "|bias| C_D,p": f"{a.abs_bias_D:.1f}", "lambda": f"{a.lam:g}", "full ft_B_OOF |bias|": f"{f.abs_bias_L:.1f}", "full ft_B_OOF Rel L2": f"{f.rel_l2:.4f}"})
    L += [md(tab), ""]
    sm = []
    verdict = {}
    for k in (10, 25):
        a = R[(R.k == k) & (R.method == "retrain + B OOF")]
        if a.empty:
            continue
        nb, rl = a.abs_bias_L.values, a.rel_l2.values
        eL = np.mean([case_err[("retrain + B OOF", k, d)][0] for d in a.draw], axis=0)
        rcar = np.mean([case_err[("retrain + B OOF", k, d)][1] for d in a.draw], axis=0)
        bi = rng.integers(0, len(eL), (B, len(eL)))
        ok = nb.mean() <= 28 and rl.mean() <= 0.170
        verdict[k] = (ok, nb.mean(), rl.mean(), nb.mean() <= 28, rl.mean() <= 0.170)
        sm.append({"k": k, "draws": len(a), "|bias| C_L,p mean [min; max over draws] (sd)": f"{nb.mean():.1f} [{nb.min():.1f}; {nb.max():.1f}] ({nb.std(ddof=1) if len(nb) > 1 else float('nan'):.1f})",
                   "bias_L signed [CI over cars]": f"{eL.mean():+.1f} [{np.percentile(eL[bi].mean(1), 2.5):+.1f}; {np.percentile(eL[bi].mean(1), 97.5):+.1f}]",
                   "Rel L2 mean [min; max] (sd)": f"{rl.mean():.4f} [{rl.min():.4f}; {rl.max():.4f}] ({rl.std(ddof=1) if len(rl) > 1 else float('nan'):.4f})", "Rel L2 [CI over cars]": f"[{np.percentile(rcar[bi].mean(1), 2.5):.4f}; {np.percentile(rcar[bi].mean(1), 97.5):.4f}]",
                   "RMSE": f"{a.rmse_L.mean():.1f}", "|bias| C_D,p": f"{a.abs_bias_D.mean():.1f}"})
    L += ["## Summary over the draws that were completed (PREREG question: |bias| <= 28 and Rel L2 <= 0.170, mean over draws)", "", md(sm), ""]
    pr_ = []
    for k, (ok, nb, rl, a1, a2) in verdict.items():
        pr_.append({"k": k, "|bias| <= 28": f"{'yes' if a1 else 'NO'} ({nb:.1f})", "Rel L2 <= 0.170": f"{'yes' if a2 else 'NO'} ({rl:.4f})", "both": "MET" if ok else "NOT met"})
    L += [md(pr_), ""]
    pred = []
    if 25 in verdict:
        pred.append(f"Registered prediction 'met at k = 25': {'TRUE' if verdict[25][0] else 'FALSE'}.")
    if 10 in verdict:
        v = verdict[10]
        pred.append(f"Registered prediction 'k = 10: |bias| <= 28 met and Rel L2 in [0.166, 0.170]': |bias| {'met' if v[3] else 'NOT met'} ({v[1]:.1f}), Rel L2 {v[2]:.4f} -> {'TRUE' if v[3] and 0.166 <= v[2] <= 0.170 else 'FALSE'}.")
    L += pred + [""]
    (OUT / "T3_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
