"""Run scripts/training/train.py runs on Modal (replacement for kaggle/notebook.py when the Kaggle GPU quota is used up).
One run = one container with one GPU; runs go in parallel.
Setup (once):  modal volume create voxel-sdf-data;  modal volume put voxel-sdf-data modal_run/_stage/cache /cache
Run:           modal run modal_run/train_modal.py --runs "E2A:m2:R2:0:3e-3:loc,E2A:m2:R2:1:3e-3:loc" [--gpu T4] [--hours 6]
Fetch:         modal volume get voxel-sdf-out /<run_id> results/decomposition/T6/raw
Run spec = split:model:res:seed:lr:tag (same as kaggle/push_kernel.py; no optional epochs fields). Flags per tag as in kaggle/notebook.py FLAG_TAGS."""
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
FLAG_TAGS = {"nxg": ["--no-decoder-xyz", "--no-global-vector"], "loc": ["--local-feats"], "nxgloc": ["--no-decoder-xyz", "--no-global-vector", "--local-feats"]}

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libgl1", "libgomp1", "libglib2.0-0")
         .pip_install("torch==2.10.0", "numpy==2.2.6", "scipy==1.15.1", "pandas==2.2.3", "pyyaml==6.0.2", "trimesh==5.2.0", "open3d==0.19.0")
         .add_local_dir(ROOT / "kaggle" / "_stage" / "code", "/repo", copy=True, ignore=["dataset-metadata.json"]))
data = modal.Volume.from_name("voxel-sdf-data")
out = modal.Volume.from_name("voxel-sdf-out", create_if_missing=True)
app = modal.App("voxel-sdf", image=image)


@app.function(volumes={"/data": data}, timeout=3600)
def prepare():
    """Unpack the uploaded shards (tar files anywhere under /data/*shards*) into /data/cache and check the car count."""
    subprocess.run("cd /data && find . -path '*shards*' -name 'cache_*.tar' | sort | while read f; do tar xf \"$f\" -C /data || exit 1; done", shell=True, check=True)
    n = len(os.listdir("/data/cache"))
    assert n == 750, n
    data.commit()
    return n


@app.function(gpu="T4", volumes={"/data": data, "/out": out}, timeout=12 * 3600, max_containers=8)
def train(split, model, res, seed, lr, tag, hours, commit):
    rid = f"{split}_{model.upper()}_{res}_s{seed}" + (f"_{tag}" if tag else "")
    if not os.path.exists("/repo/data/cache"):
        os.symlink("/data/cache", "/repo/data/cache")
    os.makedirs("/out/runs", exist_ok=True)
    cmd = [sys.executable, "scripts/training/train.py", "--split", split, "--model", model, "--res", res, "--seed", str(seed), "--lr", lr, "--tag", tag,
           "--amp", "fp16", "--batch", "2", "--accum", "4", "--max-hours", str(hours), "--runs", "/out/runs"] + FLAG_TAGS.get(tag, []) + ["--eval-test", "--save-val-pred", "--save-ckpt"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", GIT_COMMIT=commit)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd="/repo", env=env).returncode
    subprocess.run(f"mkdir -p /out/results && cp -r /repo/results/{rid} /out/results/ 2>/dev/null; cp /repo/experiments.csv /out/experiments_{rid}.csv 2>/dev/null; true", shell=True)
    out.commit()
    print(f"{rid}: exit {rc}, {(time.time() - t0) / 60:.0f} min", flush=True)
    return rid, rc


@app.local_entrypoint()
def main(runs: str, hours: float = 6.0, gpu: str = "T4"):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    specs = [r.split(":") for r in runs.split(",")]
    assert all(len(s) == 6 for s in specs), "run = split:model:res:seed:lr:tag"
    fn = train.with_options(gpu=gpu) if gpu != "T4" else train
    calls = [fn.spawn(s[0], s[1], s[2], int(s[3]), s[4], s[5], hours, commit) for s in specs]  # independent of this process; results land in volume voxel-sdf-out
    print("spawned:", [c.object_id for c in calls])
