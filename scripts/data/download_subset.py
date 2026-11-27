"""Selectively download DrivAerNet++ from the HF mirror: only STL + pressure VTK + the Cd file. Never touch lta/ (contains a private key)."""
import argparse
import csv
import random
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]


def list_stems(api, repo, folder, ext):
    return {
        Path(e.path).stem: e.size
        for e in api.list_repo_tree(repo, path_in_repo=folder, repo_type="dataset")
        if e.path.endswith(ext) and getattr(e, "size", None)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-category", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--manifest", help="CSV from make_subset.py; overrides --n-per-category")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))["source"]
    api, repo = HfApi(), cfg["hf_repo"]
    vtk = list_stems(api, repo, cfg["vtk_dir"], ".vtk")
    stl = list_stems(api, repo, cfg["stl_dir"], ".stl")
    both = sorted(set(vtk) & set(stl))

    if a.manifest:  # subset fixed by make_subset.py (status == ok)
        m = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
        picked = [r["sample_id"] for r in m if r["status"] == "ok"]
        assert set(picked) <= set(both), f"missing on HF: {sorted(set(picked) - set(both))[:5]}"
    else:
        rng = random.Random(a.seed)
        picked = []
        for prefix in cfg["categories"]:
            pool = [s for s in both if s.startswith(prefix + "_")]
            assert len(pool) >= a.n_per_category, f"{prefix}: only {len(pool)} cars"
            picked += rng.sample(pool, a.n_per_category)

    total = sum(vtk[s] + stl[s] for s in picked)
    print(f"{len(picked)} cars, {total / 1e9:.2f} GB")
    if a.dry_run:
        print(picked)
        return

    out = Path(a.out)
    hf_hub_download(repo, cfg["cd_csv"], repo_type="dataset", local_dir=out)
    for s in picked:
        hf_hub_download(repo, f"{cfg['stl_dir']}/{s}.stl", repo_type="dataset", local_dir=out)
        hf_hub_download(repo, f"{cfg['vtk_dir']}/{s}.vtk", repo_type="dataset", local_dir=out)

    if a.manifest:
        return
    with open(ROOT / "data" / "manifests" / "downloaded_probe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "category", "stl_bytes", "vtk_bytes"])
        for s in picked:
            w.writerow([s, cfg["categories"][s[0]], stl[s], vtk[s]])


if __name__ == "__main__":
    main()
