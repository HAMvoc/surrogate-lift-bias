# %% [markdown]
# Kaggle notebook (spec 11.6). Attach Inputs: voxel-sdf-code, voxel-sdf-cache, and (in later sessions) the output of the previous session.
# Enable GPU (T4 x2). Each GPU runs one independent run, no DDP; if there are more runs than GPUs they run in batches.
# Not yet verified on real Kaggle: the first T0 run is also the G2 trial run (resume, speed, hour quota).
# %%
import glob, os, shutil, subprocess, sys, time

import torch

print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())


def find(slug):
    """Directory of a dataset attached as Input; Kaggle may place it at /kaggle/input/<slug> or deeper, so search by name."""
    hits = [p for p in glob.glob(f"/kaggle/input/**/{slug}", recursive=True) if os.path.isdir(p)]
    assert hits, f"Input {slug} not found; check the dataset is attached (ls /kaggle/input)"
    return hits[0]


CODE, CACHE, REPO = find("voxel-sdf-code"), find("voxel-sdf-cache"), "/kaggle/working/repo"
shutil.copytree(CODE, REPO, dirs_exist_ok=True)
LOCALFEAT = [p for p in glob.glob("/kaggle/input/**/voxel-sdf-localfeat", recursive=True) if os.path.isdir(p)]  # optional: per-car localfeat.npy (M2 --local-feats)
if LOCALFEAT:  # merged cache = symlinks to every file of the read-only cache plus the local features
    MERGED = "/kaggle/working/cache_merged"
    for d in os.listdir(CACHE):
        if not os.path.isdir(f"{CACHE}/{d}"):
            continue  # .gitkeep etc.
        os.makedirs(f"{MERGED}/{d}", exist_ok=True)
        for f in os.listdir(f"{CACHE}/{d}"):
            if not os.path.exists(f"{MERGED}/{d}/{f}"):
                os.symlink(f"{CACHE}/{d}/{f}", f"{MERGED}/{d}/{f}")
        if os.path.exists(f"{LOCALFEAT[0]}/{d}/localfeat.npy") and not os.path.exists(f"{MERGED}/{d}/localfeat.npy"):
            os.symlink(f"{LOCALFEAT[0]}/{d}/localfeat.npy", f"{MERGED}/{d}/localfeat.npy")
    CACHE = MERGED
if not os.path.exists(f"{REPO}/data/cache"):
    os.symlink(CACHE, f"{REPO}/data/cache")  # read-only cache, ~5.2 GB, not copied
assert glob.glob(f"{REPO}/data/cache/E_S_*"), "cache empty or wrong path"

# %% Run list: (split, model, res, seed, lr, tag). Edit the list and rerun the notebook to continue from last.pt.
COMMIT = ""  # filled in when the kernel is pushed (kaggle/push_kernel.py); written to experiments.csv
FLAG_TAGS = {"nxg": ["--no-decoder-xyz", "--no-global-vector"], "loc": ["--local-feats"], "nxgloc": ["--no-decoder-xyz", "--no-global-vector", "--local-feats"]}  # run tag -> extra train.py flags (M2 factorial, the zero-shot checks study T6; "loc" needs localfeat.npy in the cache dataset)
SAVE_VAL = False  # True: also keep per-point validation predictions and EMA checkpoints in repo/results/<run_id> (push_kernel.py --save-val-pred)
EVAL_TEST = False  # True: after finishing, evaluate val and test with best.pt (only once the config is locked; record in decisions.md)
SMOKE = False  # True: 2-epoch trial run (G2) to measure speed, VRAM, quota hours; do not use the results for T0
RUNS = [("E1", "m2", "R1", 0, lr, f"T0lr{lr}") for lr in ("1e-4", "3e-4", "1e-3")]  # T0: pick lr on R1, E1
if SMOKE:
    RUNS = [("E1", "m2", "R1", 0, lr, f"smoke{lr}") for lr in ("1e-4", "1e-3")]
# RUNS_OVERRIDE
MAX_HOURS = 10.5  # budget for the whole session (max session 12 hours); batches share it, stop cleanly and save last.pt (spec 8.4)
SESSION_START = time.time()
FAILED = []


def run_id(split, model, res, seed, tag):
    return f"{split}_{model.upper()}_{res}_s{seed}" + (f"_{tag}" if tag else "")


def restore(rid):
    """Load the latest last.pt/best.pt/log.csv from the previous session's output (attached as Input)."""
    for f in glob.glob(f"/kaggle/input/**/runs/{rid}/*", recursive=True):
        dst = f"/kaggle/working/runs/{rid}"
        os.makedirs(dst, exist_ok=True)
        if not os.path.exists(f"{dst}/{os.path.basename(f)}"):
            shutil.copy(f, dst)


n_gpu = max(1, torch.cuda.device_count())
for i in range(0, len(RUNS), n_gpu):  # one batch = n_gpu parallel runs, one GPU per run
    left_h = MAX_HOURS - (time.time() - SESSION_START) / 3600
    if left_h < 0.1:
        print("session hour budget exhausted, skipping the remaining runs:", RUNS[i:], flush=True)
        break
    procs = []
    for g, (split, model, res, seed, lr, tag, *extra) in enumerate(RUNS[i:i + n_gpu]):  # extra[0]: optional epochs override
        restore(run_id(split, model, res, seed, tag))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", GIT_COMMIT=COMMIT)
        cmd = [sys.executable, "scripts/training/train.py", "--split", split, "--model", model, "--res", res, "--seed", str(seed),
               "--lr", lr, "--tag", tag, "--amp", "fp16", "--batch", "2", "--accum", "4",
               "--max-hours", f"{left_h:.3f}", "--runs", "/kaggle/working/runs"] + (["--epochs", "2", "--warmup-epochs", "1"] if SMOKE else (["--epochs", str(extra[0])] if extra else []) + (["--n-points", str(extra[1])] if len(extra) > 1 else []) + (["--val-every", str(extra[2]), "--val-n", str(extra[3])] if len(extra) > 3 else []) + (["--train-n", str(extra[4]), "--warmup-epochs", str(extra[5])] if len(extra) > 5 else []) + (["--calib-k", str(extra[6]), "--calib-rep", str(extra[7])] if len(extra) > 7 and extra[6] > 0 else []) + (["--calib-select", extra[8], "--calib-draw", str(extra[9])] if len(extra) > 9 and extra[6] > 0 else []) + (["--lam-int", str(extra[10])] if len(extra) > 10 and extra[10] > 0 else [])) + FLAG_TAGS.get(tag, []) + (["--eval-test"] if EVAL_TEST else []) + (["--save-val-pred", "--save-ckpt"] if SAVE_VAL and EVAL_TEST else [])
        procs.append((run_id(split, model, res, seed, tag), subprocess.Popen(cmd, cwd=REPO, env=env)))
    t0 = time.time()
    while any(p.poll() is None for _, p in procs):  # progress heartbeat every 60 s (first attempt: no log seen for 26 minutes)
        time.sleep(60)
        rows = []
        for rid, _ in procs:
            try:  # the file may be mid-write; the progress heartbeat must never break a run
                lines = open(f"/kaggle/working/runs/{rid}/log.csv").read().strip().splitlines()
                ep = lines[-1].split(",")[0]
                vals = [float(l.split(",")[2]) for l in lines[1:] if l.split(",")[2]]  # empty cell = epoch without validation (--val-every)
                rows.append(f"{rid}: epoch {ep} val {vals[-1]:.4f} best {min(vals):.4f}" if vals else f"{rid}: epoch {ep} (no validation yet)")
            except Exception:
                rows.append(f"{rid}: loading data or no epoch yet")
        print(f"[batch {i // n_gpu + 1} +{time.time() - t0:.0f}s, session {(time.time() - SESSION_START) / 60:.0f} min] " + " | ".join(rows), flush=True)
    for rid, p in procs:
        print(f"{rid}: exit code {p.returncode}", flush=True)
        if p.returncode != 0:
            FAILED.append(rid)
# Output: /kaggle/working/runs/<run_id>/{last.pt,best.pt,log.csv} and /kaggle/working/repo/experiments.csv;
# pull to the laptop and upload as Input for the next session. Record the actual GPU hours to learn whether quota counts per session or per GPU.

# %% Summary: print log.csv of every run so it can be read from the kernel output
for f in sorted(glob.glob("/kaggle/working/runs/*/log.csv")):
    print("==", f)
    print(open(f).read())

assert not FAILED, f"runs failed: {FAILED}"  # so the kernel reports ERROR instead of COMPLETE
