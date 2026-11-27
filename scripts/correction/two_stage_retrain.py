"""Decomposition and first corrections study, task T5: two-stage C2 on the existing retrain checkpoints. Stage A = M2 retrained with k extra Estateback cars (calibration pool = first k of the fixed permutation); stage B = a per-region
constant correction (scalar_shift, region_const with 8 regions, intmatch) fitted on labelled cars. Variants B-same / B-split / B-oracle-holdout, evaluation on permutation[35:] (215 cars).
Output: results/decomposition/T5/. Usage: python scripts/correction/two_stage_retrain.py"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.correction.calibration import build_stats, jackknife_plus, make_fit, metrics  # noqa: E402
from scripts.diagnosis.regions import labeler as labeler_v2  # noqa: E402
from src.data.dataset import CarSet, load_split  # noqa: E402
from src.models.m2 import M2  # noqa: E402

RES, OUT, K = ROOT / "results", ROOT / "results" / "decomposition" / "T5", ROOT / "kaggle" / "_out"
B = 2000
METHODS = ("scalar_shift", "region_const", "intmatch_1", "intmatch_10")
THR = {"abs_bias_L": 28.0, "rel_l2": 0.175, "cov": (0.87, 0.93), "half": 140.0}


def ckpt(seed, k):
    return K / ("calib" if seed == 0 else "calib2") / "runs" / f"E2A_M2_R2_s{seed}_cal{k}" / "best.pt"


def predict_ids(path, ids, dev):
    c = torch.load(path, map_location=dev, weights_only=False)
    m = c["cfg"]["model_cfg"]
    model = M2(chs=tuple(m["chs"]), in_ch=1, proj=m["proj"], glob=m["glob"], fourier=0).to(dev)
    model.load_state_dict(c["ema"])
    model.eval()
    data = CarSet(ids, "R2", "sdf")
    out = {}
    with torch.no_grad():
        for j, sid in enumerate(data.ids):
            b = {k: v.to(dev) for k, v in data.batch([j]).items()}
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                o = model(b["vol"], b["g"], b["ndc"], b["nrm"])
            out[sid] = (o[0].float().cpu().numpy() * c["sd"] + c["mu"]).astype(np.float16)
    return out


def boot_ci(x, f=np.mean):
    x = np.asarray(x, float)
    idx = np.random.default_rng(0).integers(0, len(x), (B, len(x)))
    v = np.array([f(x[i]) for i in idx])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    perm = [str(i) for i in np.random.default_rng(0).permutation(sorted(load_split("E2A")["test"]))]
    pool, oracle, evl = perm[:25], perm[25:35], perm[35:]
    n = len(perm)
    ev_idx = np.arange(35, n)
    # predictions of the frozen model and of every retrain run on all 250 cars
    def saved(run):
        d = RES / run
        ids = list(pd.read_csv(d / "test_per_car.csv").id)
        return np.load(d / "test_pred_fp16.npy", mmap_mode="r"), {i: j for j, i in enumerate(ids)}
    preds = {}
    for s in range(3):
        a, pos = saved(f"E2A_M2_R2_s{s}")
        preds[("frozen", s)] = {i: a[pos[i]] for i in perm}
        for k in (5, 10, 25):
            a, pos = saved(f"E2A_M2_R2_s{s}_cal{k}")
            d = {i: a[pos[i]] for i in perm if i in pos}
            miss = [i for i in perm if i not in pos]
            assert set(miss) == set(pool), "retrain runs must lack exactly the 25 pool cars"
            d.update(predict_ids(ckpt(s, k), miss, dev))  # the 25 pool cars (partly in the training set of the run)
            preds[(k, s)] = d
        print("predictions seed", s, f"{time.time() - t0:.0f} s", flush=True)
    stats = {}
    for key in ["frozen", 5, 10, 25]:
        stats[key] = build_stats(perm, lambda s, sid, key=key: preds[(key, s)][sid], 3, labeler_v2, 8)
    gate = stats["frozen"]["dL"].mean(0).mean()
    assert abs(gate - 183.9) <= 0.5, gate
    # consistency check of retrain predictions on the 225 saved cars against the saved per-car Rel L2
    for k in (5, 10, 25):
        csv = pd.read_csv(RES / f"E2A_M2_R2_s0_cal{k}" / "test_per_car.csv").set_index("id").rel_l2
        mine = pd.Series(np.sqrt(stats[k]["Qu"][0]) / np.sqrt(stats[k]["Rn"]), index=perm).loc[csv.index]
        assert np.abs(mine - csv).max() < 2e-3, ("rel L2 mismatch", k, np.abs(mine - csv).max())
    rows = []

    def add(group, k_a, k_b, label, z, fit_idx, method, fit_z=None):
        fit = make_fit(method, z, fit_idx, fit_z)
        m = metrics(z, ev_idx, fit)
        cov = hw = np.nan
        if method != "none" and len(fit_idx) >= 5:
            cov, hw, _ = jackknife_plus(z, fit_idx, ev_idx, method, fit_z)
        e = m["eL"]
        from scripts.correction.calibration import corrected
        rel_pc = corrected(z, ev_idx, fit)[2].mean(0)
        rows.append({"group": group, "stage A (retrain k)": k_a, "stage B cars": k_b, "variant": label, "method": method, "bias_L": m["bias_L"], "bias_L_lo": boot_ci(e)[0], "bias_L_hi": boot_ci(e)[1], "abs_bias_L": m["abs_bias_L"], "abs_bias_D": m["abs_bias_D"],
                     "rel_l2": m["rel_l2"], "rel_lo": boot_ci(rel_pc)[0], "rel_hi": boot_ci(rel_pc)[1], "rmse_L": m["rmse_L"], "mae_L": m["mae_L"], "spearman": m["spearman"], "cover": cov, "half": hw, "labels_total": (k_a if group == "B-same" else k_a + k_b)})
    zf = stats["frozen"]
    add("frozen", 0, 0, "frozen M2, no correction", zf, np.arange(5), "none")
    for k in (5, 10, 25):
        for method in METHODS:
            add("frozen", 0, k, "frozen M2 + stage B on first k", zf, np.arange(k), method)
        zk = stats[k]
        add("A only", k, 0, "retrain only", zk, np.arange(k), "none")
        for method in METHODS:
            add("B-same", k, k, "B fitted on the k retrain cars (in-sample for A)", zk, np.arange(k), method)
            add("B-oracle-holdout", k, 10, "B fitted on 10 other cars", zk, np.arange(25, 35), method)
    for k_total, k_a in ((10, 5), (25, 10)):
        for method in METHODS:
            add("B-split", k_a, k_total - k_a, f"A = retrain {k_a}, B on the next {k_total - k_a} cars of the budget {k_total}", stats[k_a], np.arange(k_a, k_total), method)
    R = pd.DataFrame(rows)
    R.round(5).to_csv(OUT / "T5_c2_two_stage.csv", index=False)
    e1 = pd.read_csv(ROOT / "results" / "decomposition" / "T3" / "T3_reference.md", sep="\n", header=None) if False else None
    L = ["# Decomposition and first corrections study, task T5: two-stage C2 on the retrain checkpoints", "",
         f"Gate: frozen M2, 3-seed bias C_L,p over the 250 cars = {gate:+.1f} counts: PASS. Saved per-car Rel L2 of the retrain runs reproduced from the predictions (max difference < 2e-3): PASS. Pool cars (permutation[0:25]) predicted by inference from `best.pt` of each run "
         "(`kaggle/_out/calib*/runs/E2A_M2_R2_s*_cal*`); the other cars from the saved test predictions. Evaluation set: permutation[35:] (215 Estateback cars, none used for any fit). CI = bootstrap over the 215 cars (bias, Rel L2). "
         "One card draw only (the pool is the first k cars of a fixed permutation), so the CIs do not contain draw-to-draw variation, which was large in the zero-shot checks study/addendum 3 (k = 10 random draws: 9% to 45% of the bias left). "
         "Metrics on the 3-seed mean error; jackknife+ 90% intervals with full refit of stage B (stage A is not refitted, so this is not a valid interval for the whole pipeline).", "",
         "Variants (PREREG): **B-same** = stage B fitted on the k cars that stage A was trained on (in-sample residuals); **B-split** = total budget k split between stage A (k_A cars) and stage B (the other cars of the budget); "
         "**B-oracle-holdout** = stage B fitted on 10 other cars (label budget k + 10, an upper reference); `frozen` = no retrain. Total labelled cars in the last column.", ""]
    for k in (5, 10, 25):
        sub = R[((R.group.isin(["A only", "B-same", "B-oracle-holdout"])) & (R["stage A (retrain k)"] == k)) | ((R.group == "frozen") & (R["stage B cars"].isin([0, k])))]
        sub = sub.copy()
        sub = sub[~((sub.group == "frozen") & (sub["stage B cars"] == 0) & (k != 5))]
        sub = pd.concat([sub, R[(R.group == "B-split") & (R["stage A (retrain k)"] * 0 + R.labels_total == k)]]) if k in (10, 25) else sub
        tab = [{"group": r.group, "stage A": r["stage A (retrain k)"], "stage B cars": r["stage B cars"], "method": r.method, "|bias| C_L,p": f"{r.abs_bias_L:.1f}", "Rel L2 [CI]": f"{r.rel_l2:.4f} [{r.rel_lo:.4f}; {r.rel_hi:.4f}]", "RMSE after bias removal": f"{r.rmse_L:.1f}",
                "|bias| C_D,p": f"{r.abs_bias_D:.1f}", "coverage": ("-" if np.isnan(r.cover) else f"{r.cover:.3f}"), "half-width": ("-" if np.isnan(r.half) else f"{r.half:.0f}"), "labels total": int(r.labels_total)} for _, r in sub.iterrows()]
        L += [f"## k = {k}" + (" (budget 10 for B-split)" if k == 10 else " (budget 25 for B-split)" if k == 25 else ""), "", md(tab), ""]
    # thresholds and prediction at k = 10 (region_const)
    def get(group, k, method, a=None):
        q = R[(R.group == group) & (R.method == method)]
        q = q[(q["stage A (retrain k)"] == (a if a is not None else k))]
        return q.iloc[0]
    rows = []
    for name, r in (("retrain only", get("A only", 10, "none")), ("B-same + region_const", get("B-same", 10, "region_const")), ("B-split + region_const (A = 5, B = 5)", get("B-split", 10, "region_const", a=5)),
                    ("B-oracle-holdout + region_const", get("B-oracle-holdout", 10, "region_const")), ("frozen + region_const (k = 10)", R[(R.group == "frozen") & (R["stage B cars"] == 10) & (R.method == "region_const")].iloc[0])):
        ok = {"|bias| <= 28": r.abs_bias_L <= 28, "Rel L2 <= 0.175": r.rel_l2 <= 0.175}
        if not np.isnan(r.cover):
            ok["coverage 87-93%"] = 0.87 <= r.cover <= 0.93
            ok["half-width < 140"] = r.half < 140
        rows.append({"configuration (k = 10)": name, **{c: ("pass" if v else "FAIL") for c, v in ok.items()}, "values": f"|bias| {r.abs_bias_L:.1f}, Rel L2 {r.rel_l2:.4f}, RMSE {r.rmse_L:.1f}" + ("" if np.isnan(r.cover) else f", coverage {r.cover:.3f}, half-width {r.half:.0f}")})
    allcols = ["|bias| <= 28", "Rel L2 <= 0.175", "coverage 87-93%", "half-width < 140"]
    rows = [{**{c: "-" for c in allcols}, **r} for r in rows]
    oh, same = get("B-oracle-holdout", 10, "region_const"), get("B-same", 10, "region_const")
    pred_a = oh.abs_bias_L <= 28 and oh.rel_l2 <= 0.175
    pred_b = same.abs_bias_L > 28
    L += ["## Registered thresholds at k = 10 and the prediction", "", md([{c: r[c] for c in ["configuration (k = 10)"] + allcols + ["values"]} for r in rows]), "",
          f"Prediction: 'retrain + region_const meets both |bias| <= 28 and Rel L2 <= 0.175 in B-oracle-holdout': |bias| {oh.abs_bias_L:.1f}, Rel L2 {oh.rel_l2:.4f} -> **{'TRUE' if pred_a else 'FALSE'}**. "
          f"'B-same keeps a bias above 28': |bias| {same.abs_bias_L:.1f} -> **{'TRUE' if pred_b else 'FALSE'}**. RMSE reference: E1 in-distribution 49.9 [29.7; 67.9] (T3).", ""]
    (OUT / "T5_report.md").write_text("\n".join(L), encoding="utf-8")
    print("prediction:", pred_a, pred_b, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
