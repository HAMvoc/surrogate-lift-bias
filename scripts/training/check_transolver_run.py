"""Transolver study, task T1: checks of a trained Transolver run (best.pt of runs/<run_id>) on a few cars of a split.
(1) inference cost: one forward over the whole 100k-point pool;
(2) dependence on the partition: Physics-Attention builds its slice tokens from the points of the input set, so the prediction of a point can depend on which other points are in the same call;
    the 100k single pass is compared with 2 x 50k and 10 x 10k random partitions (max |dCp| in Cp units and relative L2 of the difference);
(3) the integrals of scripts/diagnosis/force_alignment.py (IPW, clipped Cp, same A_ref, same 100k points) for the single pass, as a smoke check of the integration path.
Usage: python scripts/training/check_transolver_run.py --run-id E1_TSV_R1_s0 --split E1 --role val --n 5 --runs runs --out results/transolver/check_<run_id>.json"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from src.data.dataset import CarSet, load_split  # noqa: E402
from src.models.transolver import TransolverPC  # noqa: E402
from scripts.diagnosis import force_alignment as fa  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", default="E1")
    ap.add_argument("--role", default="val")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out")
    a = ap.parse_args()
    ck = torch.load(Path(a.runs) / a.run_id / "best.pt", map_location="cuda", weights_only=False)
    m = TransolverPC().cuda().eval()
    m.load_state_dict(ck["ema"])
    mu, sd = ck["mu"], ck["sd"]
    ids = load_split(a.split)[a.role][: a.n]
    data = CarSet(ids, "R1", "sdf")
    rows, gen = [], torch.Generator().manual_seed(0)
    for j, sid in enumerate(ids):
        b = data.batch([j])
        feats = (b["ndc"].cuda(), b["nrm"].cuda())
        P = feats[0].shape[1]

        @torch.no_grad()
        def run(idx):
            return m(None, None, feats[0][:, idx], feats[1][:, idx])[0].float().cpu() * sd + mu

        allidx = torch.arange(P, device="cuda")
        run(allidx)
        torch.cuda.synchronize()
        t = time.time()
        full = run(allidx)
        torch.cuda.synchronize()
        sec = time.time() - t
        row = {"id": sid, "sec_full_pass": sec, "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}
        for name, parts in (("2x50k", 2), ("10x10k", 10)):
            perm = torch.randperm(P, generator=gen)
            out = torch.empty(P)
            for c in perm.chunk(parts):
                out[c] = run(c.cuda())
            d = (out - full).abs()
            row[f"max_abs_diff_{name}"], row[f"rel_l2_diff_{name}"] = float(d.max()), float(torch.linalg.norm(out - full) / torch.linalg.norm(full))
        xyz, nrm, cp, area = fa.load_pool(sid)
        row.update({k: v for k, v in fa.car_terms(full.numpy(), cp, nrm, area, sid).items() if k in ("dL", "dD", "rho_L", "e_norm")})
        rows.append(row)
        print(json.dumps(row), flush=True)
    res = {"run_id": a.run_id, "n_cars": len(rows), "cars": rows, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "max_abs_diff_10x10k": max(r["max_abs_diff_10x10k"] for r in rows), "max_abs_diff_2x50k": max(r["max_abs_diff_2x50k"] for r in rows)}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    print("max |dCp| 2x50k", res["max_abs_diff_2x50k"], "10x10k", res["max_abs_diff_10x10k"])


if __name__ == "__main__":
    main()
