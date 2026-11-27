"""Push the notebook to Kaggle and run it (T4 x2). Fills in the commit hash, sets SMOKE if needed.
  python kaggle/push_kernel.py --slug voxel-sdf-t0 [--smoke] [--with-ckpt] [--hours 11]
--with-ckpt: additionally attach the dataset voxel-sdf-ckpt (checkpoints from the previous session, made by kaggle/sync_ckpt.py) to resume."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "sen4505"


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs", help="split:model:res:seed:lr:tag[:epochs[:n_points[:val_every:val_n[:train_n:warmup[:calib_k:calib_rep[:calib_select:calib_draw:lam_int]]]]]],... replaces the notebook RUNS list")
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--save-val-pred", action="store_true", help="with --eval-test: keep val_pred_fp16.npy and ckpt_ema.pt of every run (C1 retrain)")
    ap.add_argument("--with-ckpt", action="store_true")
    ap.add_argument("--notebook", default="notebook.py", help="script in kaggle/ to push (default notebook.py)")
    ap.add_argument("--with-localfeat", action="store_true", help="also attach the dataset voxel-sdf-localfeat (needed for --local-feats runs)")
    ap.add_argument("--extra-sources", default="", help="comma-separated extra dataset slugs of the user to attach (e.g. voxel-sdf-m2ckpt)")
    ap.add_argument("--cmds", help="notebook_transolver.py only: commands (script + args) separated by ';;', run in order in the repo copy")
    ap.add_argument("--hours", type=float, default=11.0, help="kernel time limit (hours), max 12")
    a = ap.parse_args()
    commit = sh("git", "rev-parse", "--short", "HEAD") + ("-dirty" if sh("git", "status", "--porcelain", "--", "src", "scripts", "configs") else "")
    src = (ROOT / "kaggle" / a.notebook).read_text(encoding="utf-8")
    src = src.replace('COMMIT = ""', f'COMMIT = "{commit}"', 1)
    if a.runs:
        runs = [tuple(int(x) if k in (3, 6, 7, 8, 9, 10, 11, 12, 13, 15) else float(x) if k == 16 else x for k, x in enumerate(r.split(":"))) for r in a.runs.split(",")]
        assert all(len(r) in (6, 7, 8, 10, 12, 14, 17) for r in runs), "each run needs split:model:res:seed:lr:tag[:epochs[:n_points[:val_every:val_n[:train_n:warmup[:calib_k:calib_rep[:calib_select:calib_draw:lam_int]]]]]]"
        src = src.replace("# RUNS_OVERRIDE", f"RUNS = {runs!r}", 1)
    if a.cmds:
        src = src.replace("CMDS = []  # CMDS_OVERRIDE", f"CMDS = {a.cmds.split(';;')!r}", 1)
    if a.eval_test:
        src = src.replace("EVAL_TEST = False", "EVAL_TEST = True", 1)
    if a.save_val_pred:
        assert a.eval_test, "--save-val-pred needs --eval-test"
        src = src.replace("SAVE_VAL = False", "SAVE_VAL = True", 1)
    if a.smoke:
        src = src.replace("SMOKE = False", "SMOKE = True", 1)
    d = ROOT / "kaggle" / "_kernel" / a.slug
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.slug}.py").write_text(src, encoding="utf-8")
    sources = [f"{USER}/voxel-sdf-code", f"{USER}/voxel-sdf-cache"] + ([f"{USER}/voxel-sdf-ckpt"] if a.with_ckpt else []) + ([f"{USER}/voxel-sdf-localfeat"] if a.with_localfeat else []) + [f"{USER}/{s}" for s in a.extra_sources.split(",") if s]
    (d / "kernel-metadata.json").write_text(json.dumps({"id": f"{USER}/{a.slug}", "title": a.slug, "code_file": f"{a.slug}.py", "language": "python",
                                                        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_internet": "false",
                                                        "dataset_sources": sources, "competition_sources": [], "kernel_sources": []}, indent=1))
    print(f"commit {commit}; sources {sources}")
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d.resolve()), "--accelerator", "NvidiaTeslaT4", "-t", str(int(a.hours * 3600))], text=True, capture_output=True)
    print(r.stdout.strip(), r.stderr.strip())


if __name__ == "__main__":
    main()
