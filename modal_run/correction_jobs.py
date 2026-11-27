"""Two-stage correction study jobs on Modal (T4). Same image/volumes as modal_run/train_modal.py. Results land in volume voxel-sdf-out under /p3.
  modal run -d modal_run/correction_jobs.py::main --what t1 --variants dec,full --seeds 0,1,2 --chunks 0-4,5-9 --ks 5,10,25
  modal run -d modal_run/correction_jobs.py::main --what retrain --ks 10,25 --draws 0-4          (seed 0, k random draws)
  modal run -d modal_run/correction_jobs.py::main --what base                                   (T6: M2 original, E2A, seed 0)
  modal run -d modal_run/correction_jobs.py::main --what smoke                                  (T1 smoke, 2 steps)
Fetch: modal volume get --force voxel-sdf-out /p3 <dir>"""
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libgl1", "libgomp1", "libglib2.0-0")
         .pip_install("torch==2.10.0", "numpy==2.2.6", "scipy==1.15.1", "pandas==2.2.3", "pyyaml==6.0.2", "trimesh==5.2.0", "open3d==0.19.0")
         .add_local_dir(ROOT / "kaggle" / "_stage" / "code", "/repo", copy=True, ignore=["dataset-metadata.json"]))
data = modal.Volume.from_name("voxel-sdf-data")
out = modal.Volume.from_name("voxel-sdf-out", create_if_missing=True)
app = modal.App("voxel-sdf-p3", image=image)


def link_cache():
    if not os.path.exists("/repo/data/cache"):
        os.symlink("/data/cache", "/repo/data/cache")


@app.function(gpu="T4", volumes={"/data": data, "/out": out}, timeout=4 * 3600)
def t1(variant, seed, draws, ks, smoke=False):
    link_cache()
    d = "/out/p3/t1_smoke" if smoke else "/out/p3/t1"
    cmd = [sys.executable, "scripts/correction/two_stage_oof.py", "--variant", variant, "--seeds", str(seed), "--draws", *map(str, draws), "--ks", *map(str, ks), "--out", d, "--ckpt-dir", "/data/ckpt", "--hours", "3.6"] + (["--smoke"] if smoke else [])
    t0 = time.time()
    rc = subprocess.run(cmd, cwd="/repo", env=dict(os.environ, PYTHONHASHSEED="0", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")).returncode
    out.commit()
    print(f"t1 {variant} s{seed} draws {draws} ks {ks}: exit {rc}, {(time.time() - t0) / 3600:.2f} h", flush=True)
    return rc


@app.function(gpu="T4", volumes={"/data": data, "/out": out}, timeout=3 * 3600)
def retrain(seed, tag, extra, hours=2.5):
    """One train.py run of the original M2 E2A configuration (+ extra CLI args), eval on test, per-point predictions and EMA checkpoint saved."""
    link_cache()
    rid = f"E2A_M2_R2_s{seed}_{tag}"
    os.makedirs("/out/p3/runs", exist_ok=True)
    cmd = [sys.executable, "scripts/training/train.py", "--split", "E2A", "--model", "m2", "--res", "R2", "--seed", str(seed), "--lr", "3e-3", "--tag", tag, "--amp", "fp16", "--batch", "2", "--accum", "4",
           "--max-hours", str(hours), "--runs", "/out/p3/runs", "--eval-test", "--save-val-pred", "--save-ckpt"] + list(extra)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd="/repo", env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", GIT_COMMIT="two_stage")).returncode
    subprocess.run(f"mkdir -p /out/p3/results && cp -r /repo/results/{rid} /out/p3/results/ 2>/dev/null; cp /repo/experiments.csv /out/p3/experiments_{rid}.csv 2>/dev/null; true", shell=True)
    out.commit()
    print(f"{rid}: exit {rc}, {(time.time() - t0) / 3600:.2f} h", flush=True)
    return rid, rc


def rng(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


@app.local_entrypoint()
def main(what: str, variants: str = "dec,full", seeds: str = "0,1,2", chunks: str = "0-4,5-9", ks: str = "5,10,25", draws: str = "0-4", folds: str = ""):
    calls = []
    if what in ("t1", "smoke"):
        for v in variants.split(","):
            for sd in map(int, seeds.split(",")):
                for ch in ([chunks.split(",")[0].split("-")[0] + "-0"] if what == "smoke" else chunks.split(",")):
                    calls.append(t1.spawn(v, sd, rng(ch), [int(x) for x in ks.split(",")], what == "smoke"))
    elif what == "retrain":
        for k in map(int, ks.split(",")):
            for d in rng(draws):
                calls.append(retrain.spawn(0, f"cal{k}d{d}", ["--epochs", "150", "--n-points", "32768", "--val-every", "1", "--val-n", "0", "--train-n", "0", "--warmup-epochs", "5", "--calib-k", str(k), "--calib-rep", "1",
                                                           "--calib-select", "random", "--calib-draw", str(d)]))
    elif what == "oof":  # retrain with one fold of the chosen cars left out (k from --ks, draws from --draws, folds 0..4)
        for k in map(int, ks.split(",")):
            for d in rng(draws):
                for fo in range(5):
                    calls.append(retrain.spawn(0, f"cal{k}d{d}f{fo}", ["--epochs", "150", "--n-points", "32768", "--val-every", "1", "--val-n", "0", "--train-n", "0", "--warmup-epochs", "5", "--calib-k", str(k), "--calib-rep", "1",
                                                                  "--calib-select", "random", "--calib-draw", str(d), "--calib-fold", str(fo)]))
    elif what == "base":
        calls.append(retrain.spawn(0, "modal", []))
    print("spawned:", [c.object_id for c in calls])
