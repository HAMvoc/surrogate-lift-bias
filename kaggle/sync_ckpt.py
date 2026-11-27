"""Download the kernel output to the laptop and push checkpoints (runs/*/{last,best}.pt, log.csv) as the private dataset voxel-sdf-ckpt, used as Input for the next session.
  python kaggle/sync_ckpt.py --kernel voxel-sdf-t0 [--create]      (--create the first time; later runs create a new version)
The output is also kept in kaggle/_out/<kernel> (log, runs, experiments.csv) for reading results."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "sen4505"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--create", action="store_true")
    a = ap.parse_args()
    out = ROOT / "kaggle" / "_out" / a.kernel
    out.mkdir(parents=True, exist_ok=True)
    print(subprocess.run(["kaggle", "kernels", "output", f"{USER}/{a.kernel}", "-p", str(out.resolve())], capture_output=True, text=True).stdout[-400:])
    stage = ROOT / "kaggle" / "_stage" / "ckpt"
    shutil.rmtree(stage, ignore_errors=True)
    n = 0
    for f in (out / "runs").glob("*/*"):
        if f.name in ("last.pt", "best.pt", "log.csv"):
            (stage / "runs" / f.parent.name).mkdir(parents=True, exist_ok=True)
            shutil.copy(f, stage / "runs" / f.parent.name / f.name)
            n += 1
    assert n, "no checkpoint in the output (kernel not finished or failed?)"
    (stage / "dataset-metadata.json").write_text(json.dumps({"title": "voxel-sdf-ckpt", "id": f"{USER}/voxel-sdf-ckpt", "licenses": [{"name": "other"}]}, indent=1))
    cmd = ["kaggle", "datasets", "create" if a.create else "version", "-p", str(stage.resolve()), "--dir-mode", "zip"] + ([] if a.create else ["-m", f"ckpt from {a.kernel}"])
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"{n} checkpoint/log files;", ((r.stdout or "") + (r.stderr or "")).strip()[-300:])


if __name__ == "__main__":
    main()
