"""Prepare two private Kaggle Datasets (spec 11.6). Does NOT upload: prints the commands for you to run after `kaggle auth` (needs your account).
  1) code:  src, scripts, configs, manifest + splits, Cd CSV    -> kaggle/_stage/code
  2) cache: data/cache (5.2 GB)                                 -> writes dataset-metadata.json directly in data/cache
Usage: python kaggle/stage_upload.py --user <kaggle-username>"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def meta(path, user, slug, title):
    (path / "dataset-metadata.json").write_text(json.dumps({"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    a = ap.parse_args()
    code = ROOT / "kaggle" / "_stage" / "code"
    shutil.rmtree(code, ignore_errors=True)
    for d in ("src", "scripts", "configs"):
        shutil.copytree(ROOT / d, code / d, ignore=shutil.ignore_patterns("__pycache__"))
    (code / "data" / "manifests").mkdir(parents=True)
    shutil.copytree(ROOT / "data" / "manifests" / "splits", code / "data" / "manifests" / "splits")
    shutil.copy(ROOT / "data" / "manifests" / "manifest.csv", code / "data" / "manifests")
    shutil.copy(ROOT / "data" / "manifests" / "dist_R1.npz", code / "data" / "manifests")
    shutil.copy(ROOT / "data" / "manifests" / "car_geometry.csv", code / "data" / "manifests")  # needed by scripts/diagnosis/pressure_coefficients.py (IPW weights, areas)
    if (ROOT / "data" / "manifests" / "stage_b_oof_offsets.json").exists():  # Two-stage correction study stage-B offsets (OOF) for the label-budget study T3
        shutil.copy(ROOT / "data" / "manifests" / "stage_b_oof_offsets.json", code / "data" / "manifests")
    if (ROOT / "results" / "decomposition" / "T1" / "labels_v2.npz").exists():  # regions v2 labels (3 MB), used by scripts/diagnosis/regions.py on Kaggle
        shutil.copy(ROOT / "results" / "decomposition" / "T1" / "labels_v2.npz", code / "data" / "manifests")
    (code / "data" / "raw" / "drag").mkdir(parents=True)
    shutil.copy(ROOT / "data" / "raw" / "drag" / "DrivAerNetPlusPlus_Cd_8k_Updated.csv", code / "data" / "raw" / "drag")
    meta(code, a.user, "voxel-sdf-code", "voxel-sdf-code")
    meta(ROOT / "data" / "cache", a.user, "voxel-sdf-cache", "voxel-sdf-cache")
    print("Dataset PRIVATE (CC BY-NC data, do not republish). Run:\n"
          f"  kaggle datasets create -p {code} --dir-mode zip\n"
          f"  kaggle datasets create -p {ROOT / 'data' / 'cache'} --dir-mode zip\n"
          "Update the code after each change: kaggle datasets version -p <directory> -m <note> --dir-mode zip")


if __name__ == "__main__":
    main()
