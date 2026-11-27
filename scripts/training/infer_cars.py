"""Independent-set study, task T2: inference of the E2A models (as trained, no training) on a list of cars; writes <out>/<label>/{test_per_car.csv, test_pred_fp16.npy} in the layout of results/<run_id>/.
Families: M2 (3 seeds), TSV (3 seeds), B2, B2_300ep, B3 (neural, EMA weights of best.pt, autocast and chunking as in Trainer.evaluate) and B0, B1 (kNN baselines on the 450 E2A training cars).
Cars: --ids-file CSV with a column sample_id (the new cars), or --old N (the first N E2A test cars: smoke test against the stored predictions, with --compare).
Usage: python scripts/training/infer_cars.py --ids-file data/splits/fresh_estateback.csv --out results/independent_set/preds [--fams M2 TSV ...]
       python scripts/training/infer_cars.py --old 3 --compare --out results/independent_set/smoke"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from src.data.dataset import CarSet, load_split  # noqa: E402
from src.evaluation.metrics import car_stats  # noqa: E402
from src.models.baselines import TrainBank, predict_b0, predict_b1  # noqa: E402
from src.models.m2 import M2  # noqa: E402
from src.models.pointnet2 import PointNet2Lite  # noqa: E402
from src.models.regdgcnn import RegDGCNN  # noqa: E402
from src.models.transolver import TransolverPC  # noqa: E402

K = ROOT / "kaggle" / "_out"
CK = {  # label -> (stored results dir name, checkpoint)
    "E2A_M2_R2_s0": K / "voxel-sdf-e2-s0/runs/E2A_M2_R2_s0/best.pt", "E2A_M2_R2_s1": K / "voxel-sdf-e2-s12/runs/E2A_M2_R2_s1/best.pt", "E2A_M2_R2_s2": K / "voxel-sdf-e2-s12/runs/E2A_M2_R2_s2/best.pt",
    "E2A_TSV_R1_s0": K / "p7-e2a-s0/repo/runs/E2A_TSV_R1_s0/best.pt", "E2A_TSV_R1_s1": K / "p7-e2a-s1/repo/runs/E2A_TSV_R1_s1/best.pt", "E2A_TSV_R1_s2": K / "p7-e2a-s2/repo/runs/E2A_TSV_R1_s2/best.pt",
    "E2A_B2_R1_s0": K / "voxel-sdf-b2-final/runs/E2A_B2_R1_s0/best.pt", "E2A_B2_R1_s0_ep300": K / "calib3/runs/E2A_B2_R1_s0_ep300/best.pt", "E2A_B3_R1_s0_lr3e-3": K / "b3e1/runs/E2A_B3_R1_s0_lr3e-3/best.pt",
}
FAMS = {"M2": [f"E2A_M2_R2_s{s}" for s in range(3)], "TSV": [f"E2A_TSV_R1_s{s}" for s in range(3)], "B2": ["E2A_B2_R1_s0"], "B2_300ep": ["E2A_B2_R1_s0_ep300"], "B3": ["E2A_B3_R1_s0_lr3e-3"], "B0": ["E2A_B0_NA_s0"], "B1": ["E2A_B1_R1_s0"]}


def build(cfg):
    m, mc = cfg["model"], cfg["model_cfg"]
    if m == "tsv":
        return TransolverPC()
    if m == "b3":
        return RegDGCNN()
    if m == "b2":
        return PointNet2Lite()
    assert m == "m2", m
    return M2(chs=tuple(mc["chs"]), in_ch=1, proj=mc["proj"], glob=mc["glob"], fourier=0)


def per_car(ids, preds, cs):
    rows = []
    for j, (i, p) in enumerate(zip(ids, preds)):
        c, r = car_stats(p, cs.cp[j].numpy()), car_stats(p, cs.cp_raw[j].numpy())
        rows.append({"id": i, **c, **{f"raw_{k}": v for k, v in r.items() if k in ("rel_l2", "mae", "rmse", "sse", "sum_c", "sum_c2")}})
    return pd.DataFrame(rows)


@torch.no_grad()
def run_nn(label, ids, dev):
    torch.manual_seed(0)  # PointNet++ (B2) samples its centres at random; a fixed seed makes the new-car predictions reproducible
    ck = torch.load(CK[label], map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = build(cfg).to(dev)
    model.load_state_dict(ck["ema"])
    model.eval()
    mu, sd, chunk, amp = ck["mu"], ck["sd"], cfg.get("eval_chunk", 100000), cfg.get("amp", "none")
    cs = CarSet(ids, cfg["res"], "sdf", mult=16)
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    preds = []
    for j in range(len(ids)):
        b = {k: v.to(dev) for k, v in cs.batch([j]).items()}
        out = []
        for a in range(0, b["g"].shape[1], chunk):
            part = {k: (v[:, a:a + chunk] if k in ("g", "ndc", "nrm") else v) for k, v in b.items()}
            with torch.autocast(dev, dtype=dt, enabled=dt is not None and dev == "cuda"):
                o = model(part["vol"], part["g"], part["ndc"], part["nrm"])
            out.append(o.float()[0].cpu() * sd + mu)
        preds.append(torch.cat(out).numpy().astype(np.float16))
    return per_car(ids, preds, cs), np.stack(preds)


def run_b01(label, ids):
    s = load_split("E2A")
    tr = CarSet(s["train"], "R1")
    bank = TrainBank(tr.xyz.numpy(), tr.cp.numpy())
    cs = CarSet(ids, "R1")
    if label == "E2A_B0_NA_s0":
        b0_cars = np.random.default_rng(0).choice(len(s["train"]), min(48, len(s["train"])), replace=False)
        preds = [predict_b0(bank, cs.xyz[j].numpy().astype(np.float64), b0_cars) for j in range(len(ids))]
    else:
        k = json.load(open(ROOT / "results" / "E2A_B1_R1_s0" / "meta.json"))["k"]
        xt = torch.stack([torch.from_numpy(np.load(ROOT / "data" / "cache" / i / "sdf_R1.npy")).float().flatten() for i in s["train"]])
        preds = []
        for j, i in enumerate(ids):
            x = torch.from_numpy(np.load(ROOT / "data" / "cache" / i / "sdf_R1.npy")).float().flatten()
            sq = (x * x).sum()
            d = ((sq + (xt * xt).sum(1) - 2 * xt @ x).clamp_min(0).sqrt()).numpy()
            preds.append(predict_b1(bank, cs.xyz[j].numpy().astype(np.float64), d, (k,))[k])
    preds = [np.asarray(p) for p in preds]
    return per_car(ids, preds, cs), np.stack(preds).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file")
    ap.add_argument("--old", type=int)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--fams", nargs="*", default=list(FAMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    ids = sorted(pd.read_csv(a.ids_file).sample_id) if a.ids_file else sorted(load_split("E2A")["test"])[: a.old]
    for fam in a.fams:
        for label in FAMS[fam]:
            t0 = time.time()
            try:
                df, pr = run_b01(label, ids) if fam in ("B0", "B1") else run_nn(label, ids, a.device)
            except Exception as e:  # a family that cannot run is reported, not hidden
                print(f"FAILED {label}: {type(e).__name__}: {e}", flush=True)
                (Path(a.out) / "failed.txt").parent.mkdir(parents=True, exist_ok=True)
                open(Path(a.out) / "failed.txt", "a").write(f"{label}: {type(e).__name__}: {e}\n")
                continue
            d = Path(a.out) / label
            d.mkdir(parents=True, exist_ok=True)
            df.to_csv(d / "test_per_car.csv", index=False)
            np.save(d / "test_pred_fp16.npy", pr)
            msg = f"{label}: {len(ids)} cars, {time.time() - t0:.0f} s"
            if a.compare:
                old = ROOT / "results" / label
                pos = {i: n for n, i in enumerate(pd.read_csv(old / "test_per_car.csv").id)}
                st = np.load(old / "test_pred_fp16.npy", mmap_mode="r")
                diff = max(float(np.abs(pr[j].astype(np.float32) - st[pos[i]].astype(np.float32)).max()) for j, i in enumerate(ids))
                msg += f", max |dCp| against the stored predictions {diff:.4f}"
            print(msg, flush=True)


if __name__ == "__main__":
    main()
