"""Run B0 and B1 on the splits; writes results/<run_id>/{val,test}_per_car.csv (rel L2 on clipped Cp and on raw Cp).
B1 picks k in {1, 3, 5, 10} on E1 val, then locks it for all splits (spec 6.2 no. 4).
Usage: python scripts/training/run_baselines.py [--splits E1 E2A E2B E2C E2param] [--limit N]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.dataset import CarSet, category_of, load_split  # noqa: E402
from src.evaluation.metrics import car_stats, summarize  # noqa: E402
from src.models.baselines import TrainBank, predict_b0, predict_b1  # noqa: E402

KS = (1, 3, 5, 10)
N_B0 = 48  # B0 averages over 48 randomly chosen train cars (seed 0)


def per_car(ids, preds, cs):
    rows = []
    for j, (i, p) in enumerate(zip(ids, preds)):
        c, r = car_stats(p, cs.cp[j].numpy()), car_stats(p, cs.cp_raw[j].numpy())
        rows.append({"id": i, "category": category_of(i), **c, **{f"raw_{k}": v for k, v in r.items() if k in ("rel_l2", "mae", "rmse", "sse", "sum_c", "sum_c2")}})
    return pd.DataFrame(rows)


def run_split(name, k_lock, limit, dmat, pos):
    s = load_split(name)
    t0 = time.time()
    tr = CarSet(s["train"], "R1")  # only the pool is needed; R1 is the smallest grid
    bank = TrainBank(tr.xyz.numpy(), tr.cp.numpy())
    tri = np.array([pos[i] for i in s["train"]])
    b0_cars = np.random.default_rng(0).choice(len(tri), min(N_B0, len(tri)), replace=False)
    out, preds = {}, {}
    for role in ("val", "test") if name == "E1" else ("test",):  # val is only needed on E1 to lock k
        ids = s[role][:limit] if limit else s[role]
        cs = CarSet(ids, "R1")
        res = {"B0": [], **{f"B1_k{k}": [] for k in KS}}
        for j, i in enumerate(ids):
            pts, d = cs.xyz[j].numpy().astype(np.float64), dmat[pos[i], tri]
            res["B0"].append(predict_b0(bank, pts, b0_cars))
            for k, p in predict_b1(bank, pts, d, KS).items():
                res[f"B1_k{k}"].append(p)
        out[role] = {m: per_car(ids, pr, cs) for m, pr in res.items()}
        if role == "test":  # per-point predictions in pool order (fp16), same layout as the neural runs, for the physics/region analyses
            preds = {m: np.stack(pr).astype(np.float16) for m, pr in res.items()}
        print(f"  {name} {role}: {len(ids)} cars, {time.time() - t0:.0f}s", flush=True)
    return out, preds


def save(name, model, out, k=None, preds=None):
    run = f"{name}_{model.split('_')[0]}_{'NA' if model == 'B0' else 'R1'}_s0"
    d = ROOT / "results" / run
    d.mkdir(parents=True, exist_ok=True)
    for role in out:
        out[role][model].to_csv(d / f"{role}_per_car.csv", index=False)
    if preds is not None:
        np.save(d / "test_pred_fp16.npy", preds[model])
    if k:
        (d / "meta.json").write_text(json.dumps({"k": k}))
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["E1", "E2A", "E2B", "E2C", "E2param"])
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    z = np.load(ROOT / "data" / "manifests" / "dist_R1.npz", allow_pickle=True)
    pos = {i: k for k, i in enumerate(z["ids"])}
    k_lock = None
    for name in ["E1"] + [s for s in a.splits if s != "E1"]:
        if name not in a.splits and name != "E1":
            continue
        out, preds = run_split(name, k_lock, a.limit, z["full"], pos)
        if name == "E1":  # lock k by mean rel L2 on E1 val
            val = {k: out["val"][f"B1_k{k}"].rel_l2.mean() for k in KS}
            k_lock = min(val, key=val.get)
            print("E1 val rel L2 by k:", {k: round(v, 4) for k, v in val.items()}, "-> locked k =", k_lock)
        print(save(name, "B0", out, preds=preds), save(name, f"B1_k{k_lock}", out, k_lock, preds))
        for role in out:
            for m in ("B0", f"B1_k{k_lock}"):
                r = summarize(out[role][m])
                print(f"  {name} {role} {m}: rel L2 {r['rel_l2']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] median {r['median']:.4f} R2 {r['r2_pooled']:.3f}")


if __name__ == "__main__":
    main()
