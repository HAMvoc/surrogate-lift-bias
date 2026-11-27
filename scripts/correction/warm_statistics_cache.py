"""Build the label-budget study statistics caches once (T1 and T2 share them). Usage: python scripts/correction/warm_statistics_cache.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.correction import statistics_cache as c
for fold, fam in c.PAIRS:
    z = c.pair_stats(fold, fam); print(fold, fam, z["dL"].shape, flush=True)
    if (fold, fam) == ("E2A", "M2"):
        assert abs(z["dL"].mean(0).mean() - 183.9) <= 0.5  # gate
for k, ds in ((25, range(4)), (10, range(3))):
    for d in ds:
        c.t3_stats(k, d); print("t3", k, d, flush=True)
