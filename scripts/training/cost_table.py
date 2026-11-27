"""Model cost table (spec 7.5, T2/F7): parameters, FLOPs, inference time and peak memory (measured here, on the local GPU),
training step time / peak VRAM / session hours (read from the Kaggle run logs). Writes reports/cost_table.md.
Platform note: inference is measured on the laptop GPU, training figures come from Kaggle T4 (two runs share one node, one GPU each); they are not comparable across platforms.
Usage: python scripts/training/cost_table.py [--out reports/cost_table.md]"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.grid import load_cfg  # noqa: E402
from src.models.m2 import M1, M2  # noqa: E402
from src.models.pointnet2 import PointNet2Lite  # noqa: E402

CFG = load_cfg()
N_PTS = CFG["pool"]["n_points"]
# label, factory, grid, training log (E1, 600 train cars), run id in experiments_kaggle_*.csv
ROWS = (("M1 (Cd)", lambda: M1(), "R1", "reports/t0/M1lr1e-3.log.csv", "E1_M1_R1_s0_M1lr1e-3"),
        ("M2", lambda: M2(), "R1", "reports/ablation/E1_M2_R1_s0.log.csv", "E1_M2_R1_s0"),
        ("M2", lambda: M2(), "R2", "reports/t0/E1_M2_R2_s0.log.csv", "E1_M2_R2_s0"),
        ("M2", lambda: M2(), "R3", "reports/ablation/E1_M2_R3_s0.log.csv", "E1_M2_R3_s0"),
        ("M2-occ", lambda: M2(), "R2", "reports/ablation/E1_M2OCC_R2_s0.log.csv", "E1_M2OCC_R2_s0"),
        ("M2-noXYZ", lambda: M2(use_xyz=False), "R2", "reports/ablation/E1_M2NOXYZ_R2_s0.log.csv", "E1_M2NOXYZ_R2_s0"),
        ("M2-noN", lambda: M2(use_nrm=False), "R2", "reports/ablation/E1_M2NON_R2_s0.log.csv", "E1_M2NON_R2_s0"),
        ("B2 (PointNet++ lite)", lambda: PointNet2Lite(), "-", "reports/b2/E1_B2_final.log.csv", "E1_B2_R1_s0"))


def inputs(grid, dev):
    x, y, z = CFG["grids"][grid] if grid != "-" else CFG["grids"]["R1"]
    dp = -(-z // 16) * 16
    g = torch.rand(1, N_PTS, 3, device=dev) * 2 - 1
    return torch.randn(1, 1, dp, y, x, device=dev), g, g.clone(), g.clone()


def forward(model, args, label):
    """One full-car inference: M2/M1 in one pass, B2 in chunks of 16,384 points (as in training/evaluation)."""
    vol, g, ndc, nrm = args
    if label.startswith("B2"):
        return [model(None, g[:, a:a + 16384], ndc[:, a:a + 16384], nrm[:, a:a + 16384]) for a in range(0, N_PTS, 16384)]
    return model(vol, g, ndc, nrm)


def measure(label, make, grid, dev):
    model = make().to(dev).eval()
    args = inputs(grid, dev)
    params = sum(p.numel() for p in model.parameters())
    with torch.no_grad(), FlopCounterMode(display=False) as fc_all:
        forward(model, args, label)
    enc = None
    if hasattr(model, "enc") and not label.startswith("B2"):
        with torch.no_grad(), FlopCounterMode(display=False) as fc_enc:
            model.enc(args[0])
        enc = fc_enc.get_total_flops()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(3):
            forward(model, args, label)
        torch.cuda.synchronize()
        ts = []
        for _ in range(10):
            t = time.perf_counter()
            forward(model, args, label)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
    return {"params": params, "flops": fc_all.get_total_flops(), "enc_flops": enc, "ms": 1000 * sorted(ts)[len(ts) // 2], "mem": torch.cuda.max_memory_allocated() / 2 ** 20}


def session_hours():
    out = {}
    for f in (ROOT / "reports").rglob("experiments_kaggle_*.csv"):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            out[r["run_id"]] = float(r["gpu_hours_this_call"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "cost_table.md"))
    a = ap.parse_args()
    dev = torch.device("cuda")
    hrs = session_hours()
    steps = math.ceil(600 / 8)  # E1: 600 train cars, effective batch 8
    rows = []
    for label, make, grid, log, rid in ROWS:
        m = measure(label, make, grid, dev)
        lg = pd.read_csv(ROOT / log)
        rows.append({"model": label, "grid": grid, "params (M)": f"{m['params'] / 1e6:.3f}",
                     "encoder GFLOPs / car": f"{m['enc_flops'] / 1e9:.1f}" if m["enc_flops"] else "-",
                     f"total GFLOPs / car ({N_PTS // 1000}k pts)": f"{m['flops'] / 1e9:.1f}",
                     "inference ms / car (laptop)": f"{m['ms']:.0f}", "inference peak MB (laptop)": f"{m['mem']:.0f}",
                     "train s / epoch (T4)": f"{lg.seconds.mean():.1f}", "ms / optimizer step (T4)": f"{1000 * lg.seconds.mean() / steps:.0f}",
                     "train peak VRAM MB (T4)": f"{lg.max_mem_mb.max():.0f}", "process hours / run (T4)": f"{hrs[rid]:.2f}" if rid in hrs else "-"})
        print(rows[-1], flush=True)
        torch.cuda.empty_cache()
    cols = list(rows[0])
    text = ["# Model cost table (auto-generated by `scripts/training/cost_table.py`)", "",
            f"Inference: batch 1, {N_PTS} query points, fp16 autocast, median of 10 runs after 3 warm-ups, on `{torch.cuda.get_device_name(0)}` (torch {torch.__version__}). "
            "Training: E1 (600 train cars, batch 2 x accum 4, fp16) from the Kaggle T4 run logs; two runs shared one node (one GPU each), so epoch times include some CPU contention. "
            "Hours are whole-process wall-clock (data loading + 150 epochs + evaluation). Laptop and Kaggle numbers must not be compared with each other.",
            "FLOPs are counted by `torch.utils.flop_counter` (convolutions and matmuls, multiply-add = 2); kNN distance computations and grid_sample are not counted. "
            "The M2 encoder runs once per car; the decoder cost scales with the number of query points. B2 runs in 16,384-point chunks.", "",
            "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    Path(a.out).write_text("\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
