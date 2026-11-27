# Kaggle script for the label-budget study T3 (B-same vs B-OOF, fine-tune full on k = 10, 10 draws, seed 0, ONE GPU). Inputs: voxel-sdf-code, voxel-sdf-cache, voxel-sdf-m2ckpt.
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
OUT = f"{REPO}/results/label_budget_t3"
os.makedirs(OUT, exist_ok=True)
cmd = [sys.executable, "scripts/correction/oof_vs_in_sample_finetune.py", "--ckpt-dir", CKPT, "--out", OUT]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0", GIT_COMMIT=COMMIT)
t0 = time.time()
with open(f"{OUT}/run.log", "w") as log:
    rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
print("exit code", rc, f"{(time.time() - t0) / 60:.1f} min", flush=True)
print(open(f"{OUT}/run.log").read()[-3000:], flush=True)
assert rc == 0
