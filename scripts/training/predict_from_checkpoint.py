"""Inference only (no training): per-point Cp predictions of a saved M2 run from its best.pt (EMA weights), computed like Trainer.evaluate
(fp16 autocast as in the run's cfg, all 100k pool points per car in chunks, output * sd + mu, saved as float16 in the pool order).
Writes <out>/<run_id>/{<role>_pred_fp16.npy, <role>_per_car.csv}; with --compare also prints the agreement with the saved predictions of the original run.
Usage: python scripts/training/predict_from_checkpoint.py --ckpt kaggle/_out/voxel-sdf-e2-s0/runs/E2A_M2_R2_s0/best.pt --split E2A --role val --out results/force_alignment/ckpt_preds [--compare results/E2A_M2_R2_s0]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.dataset import CarSet, load_split  # noqa: E402
from src.evaluation.metrics import car_stats  # noqa: E402
from src.models.m2 import M2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True, help="split whose <role> cars are predicted (e.g. E2A)")
    ap.add_argument("--role", default="val", choices=["val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", help="results/<run_id> of the original run (test role only): print agreement with its saved test predictions")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    run_id = Path(a.ckpt).parent.name
    c = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    cfg = c["cfg"]
    assert cfg["model"] == "m2" and cfg["split"] == a.split, f"checkpoint is {cfg['model']} on {cfg['split']}"
    mcfg = cfg["model_cfg"] if "model_cfg" in cfg else yaml.safe_load((ROOT / "configs" / "model_lite.yaml").read_text(encoding="utf-8"))
    model = M2(chs=tuple(mcfg["chs"]), in_ch=1, proj=mcfg["proj"], glob=mcfg["glob"], fourier=0).to(a.device)
    model.load_state_dict(c["ema"])
    model.eval()
    ids = load_split(a.split)[a.role]
    data = CarSet(ids, cfg["res"], "sdf")
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg["amp"])
    chunk = cfg.get("eval_chunk", 100000)
    preds, rows, t0 = [], [], time.time()
    with torch.no_grad():
        for j, sid in enumerate(data.ids):
            full = data.batch([j])
            full = {k: v.to(a.device) for k, v in full.items()}
            out = []
            for s in range(0, full["g"].shape[1], chunk):
                part = {k: (v[:, s:s + chunk] if v.dim() > 1 and k in ("g", "ndc", "nrm") else v) for k, v in full.items()}
                with torch.autocast(a.device.split(":")[0], dtype=dt, enabled=dt is not None and a.device != "cpu"):
                    o = model(part["vol"], part["g"], part["ndc"], part["nrm"])
                out.append(o[0].float().cpu() * c["sd"] + c["mu"])
            p = torch.cat(out).numpy()
            preds.append(p.astype(np.float16))
            rows.append({"id": sid, **car_stats(p, data.cp[j].numpy()), **{f"raw_{k}": v for k, v in car_stats(p, data.cp_raw[j].numpy()).items() if k in ("rel_l2", "mae", "rmse", "sse", "sum_c", "sum_c2")}})
    out_dir = Path(a.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{a.role}_pred_fp16.npy", np.stack(preds))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{a.role}_per_car.csv", index=False)
    print(f"{run_id} {a.role}: {len(ids)} cars in {time.time() - t0:.0f} s, mean rel L2 {df.rel_l2.mean():.4f}, checkpoint best val {c['best']:.5f} at epoch {c['epoch']}")
    if a.compare:
        old = Path(a.compare)
        do = pd.read_csv(old / f"{a.role}_per_car.csv")
        assert list(do.id) == list(df.id)
        msg = f"compare with {old.name}: mean rel L2 saved {do.rel_l2.mean():.4f} vs now {df.rel_l2.mean():.4f}; max |rel L2 difference| per car {np.abs(do.rel_l2.values - df.rel_l2.values).max():.2e}"
        if (old / f"{a.role}_pred_fp16.npy").exists():  # E1 seeds 0 and 1 lost their per-point predictions: only the per-car CSV can be compared
            po = np.load(old / f"{a.role}_pred_fp16.npy", mmap_mode="r")
            d = np.array([np.abs(np.asarray(po[i], np.float32) - preds[i].astype(np.float32)).max() for i in range(len(ids))])
            msg += f"; max |dCp| per car: median {np.median(d):.4f}, max {d.max():.4f}"
        print(msg)


if __name__ == "__main__":
    main()
