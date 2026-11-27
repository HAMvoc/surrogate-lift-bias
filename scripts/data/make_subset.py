"""Pick a seeded random subset: 250 cars per body style + 50 reserve, only from cars with a parameter table entry (excludes F_D_*).
Writes data/manifests/manifest.csv. Usage: python scripts/data/make_subset.py [--n 250] [--reserve 50] [--seed 0]"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.download_subset import list_stems  # noqa: E402
from src.data.grid import load_cfg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=250)
ap.add_argument("--reserve", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

cfg = load_cfg()["source"]
params = pd.read_csv((ROOT / cfg["params_csv"]).resolve()).set_index("Experiment")
pcols = [c for c in params.columns if not c.startswith(("Average", "Std"))]
api = HfApi()
avail = set(list_stems(api, cfg["hf_repo"], cfg["vtk_dir"], ".vtk")) & set(list_stems(api, cfg["hf_repo"], cfg["stl_dir"], ".stl"))
cand = params[params.index.isin(avail)].copy()
cand["category"] = [cfg["categories"][i[0]] for i in cand.index]
# group_id: the same (rounded) parameter vector means the same group, regardless of body style ("same front, different rear" leakage)
cand["group_id"] = ["g" + hashlib.md5(",".join(f"{v:.4f}" for v in r).encode()).hexdigest()[:10] for r in cand[pcols].to_numpy()]

rows = []
for cat, g in cand.groupby("category"):
    ids = g.sample(n=a.n + a.reserve, random_state=a.seed).index.tolist()  # shuffled order
    rows += [(i, cat, cand.at[i, "group_id"], "ok" if k < a.n else "reserve") for k, i in enumerate(ids)]
man = pd.DataFrame(rows, columns=["sample_id", "category", "group_id", "status"])
man.insert(0, "n", range(len(man)))
out = ROOT / "data" / "manifests" / "manifest.csv"
man.drop(columns="n").to_csv(out, index=False)
ok = man[man.status == "ok"]
print(f"candidates {len(cand)} (has parameters and files on HF); writing {out}")
print(ok.category.value_counts().to_string())
sh = ok.groupby("group_id").category.nunique()
print(f"group_id groups in subset: {ok.group_id.nunique()}; groups with >1 body style: {int((sh > 1).sum())}; cars in those groups: {int(ok.group_id.isin(sh[sh > 1].index).sum())}")
