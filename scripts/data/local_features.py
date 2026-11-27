"""Compute src/data/local_features.compute for cars and write data/cache/<id>/localfeat.npy (float16, (100000, 6), pool order). Resumable (skips existing files).
Usage: python scripts/data/local_features.py [--ids ID ...] [--all] [--limit N]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.local_features import compute  # noqa: E402

CACHE = ROOT / "data" / "cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--all", action="store_true", help="the 750 cars of the subset (all splits)")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    ids = a.ids or []
    if a.all:
        sys.path.insert(0, str(ROOT))
        from src.data.dataset import load_split
        s = load_split("E1")
        ids = sorted(set(s["train"]) | set(s["val"]) | set(s["test"]))
    ids = ids[: a.limit] if a.limit else ids
    t0, done = time.time(), 0
    for n, sid in enumerate(ids):
        f = CACHE / sid / "localfeat.npy"
        if f.exists():
            continue
        p = np.load(CACHE / sid / "pool.npz")
        np.save(f, compute(p["xyz"], p["normal"]).astype(np.float16))
        done += 1
        if done % 25 == 0:
            print(f"{n + 1}/{len(ids)} cars, {done} written, {time.time() - t0:.0f} s", flush=True)
    print(f"done: {done} written of {len(ids)} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
