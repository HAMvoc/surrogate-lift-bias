# Kaggle script for the swapped-halves study T1b (swap, M2 full ft_B_OOF; same runner as notebook_transolver.py, plus the dataset voxel-sdf-m2ckpt: {CKPT} in a command is replaced by its path).
# (copied from the Transolver study: Transolver) One kernel = one run (or the smoke test). Inputs: voxel-sdf-code, voxel-sdf-cache. ONE GPU (T4).
# CMDS is replaced by kaggle/push_kernel.py --cmds "<train.py args>;;<other script args>"; each command runs in sequence in the repo copy.
import glob, os, shlex, shutil, subprocess, sys, time

import torch

print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, flush=True)


def find(slug):
    hits = [p for p in glob.glob(f"/kaggle/input/**/{slug}", recursive=True) if os.path.isdir(p)]
    assert hits, f"Input {slug} not found"
    return hits[0]


CODE, CACHE, REPO = find("voxel-sdf-code"), find("voxel-sdf-cache"), "/kaggle/working/repo"
shutil.copytree(CODE, REPO, dirs_exist_ok=True)
if not os.path.exists(f"{REPO}/data/cache"):
    os.symlink(CACHE, f"{REPO}/data/cache")
assert glob.glob(f"{REPO}/data/cache/E_S_*"), "cache empty or wrong path"
CKPT = find("voxel-sdf-m2ckpt")
COMMIT = ""
CMDS = []  # CMDS_OVERRIDE
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0", GIT_COMMIT=COMMIT)
t0 = time.time()
for c in CMDS:
    tc = time.time()
    rc = subprocess.run([sys.executable] + shlex.split(c.replace("{CKPT}", CKPT)), cwd=REPO, env=env).returncode
    print(f"== exit code {rc} after {(time.time() - tc) / 60:.1f} min: {c}", flush=True)
    assert rc == 0
print(f"total {(time.time() - t0) / 3600:.3f} h", flush=True)
