# Kaggle script for the decomposition and first corrections study T7 (balanced fine-tuning + stage B). Inputs: voxel-sdf-code, voxel-sdf-cache, voxel-sdf-m2ckpt. Two GPUs: `dec` on GPU 0, `full` on GPU 1, in parallel.
import glob, os, shutil, subprocess, sys, time

import torch

print("python", sys.version.split()[0], "torch", torch.__version__, "gpus", torch.cuda.device_count(), flush=True)


def find(slug):
    hits = [p for p in glob.glob(f"/kaggle/input/**/{slug}", recursive=True) if os.path.isdir(p)]
    assert hits, f"Input {slug} not found"
    return hits[0]


CODE, CACHE, CKPT, REPO = find("voxel-sdf-code"), find("voxel-sdf-cache"), find("voxel-sdf-m2ckpt"), "/kaggle/working/repo"
shutil.copytree(CODE, REPO, dirs_exist_ok=True)
if not os.path.exists(f"{REPO}/data/cache"):
    os.symlink(CACHE, f"{REPO}/data/cache")
COMMIT = ""
HOURS = 3.5  # no new task is started after this many hours (the script itself checks)
OUT = f"{REPO}/results/decomposition_t7"
os.makedirs(OUT, exist_ok=True)
base = [sys.executable, "scripts/correction/finetune_stage_b.py", "--ckpt-dir", CKPT, "--hours", str(HOURS)]
jobs = {"dec": base + ["--variant", "dec", "--seeds", "0", "1", "2", "--draws", "0", "1", "2", "3", "--ks", "5", "10", "25", "--cvplus-k", "10", "25", "--out", f"{OUT}/dec"],
        "full": base + ["--variant", "full", "--seeds", "0", "--draws", "0", "1", "2", "--ks", "5", "10", "25", "--cvplus-k", "10", "--out", f"{OUT}/full"]}
procs = []
for g, (name, cmd) in enumerate(jobs.items()):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", GIT_COMMIT=COMMIT)
    log = open(f"{OUT}/{name}.log", "w")
    procs.append((name, subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), log))
t0 = time.time()
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(120)
    for name, _, _ in procs:
        lines = open(f"{OUT}/{name}.log").read().strip().splitlines()
        print(f"[+{time.time() - t0:.0f}s] {name}: {lines[-1] if lines else '(no output yet)'}", flush=True)
for name, p, log in procs:
    log.close()
    print(name, "exit code", p.returncode, flush=True)
    print(open(f"{OUT}/{name}.log").read()[-3000:], flush=True)
assert all(p.returncode == 0 for _, p, _ in procs)
