"""Train one run (spec 8). Stop/resume: rerun the same command; it loads runs/<run_id>/last.pt automatically.
Examples:
  python scripts/training/train.py --split E1 --model m2 --res R1 --seed 0
  python scripts/training/train.py --overfit 8 --res R1 --epochs 200 --lr 1e-3 --accum 2     # S2 check
The test set is only evaluated with --eval-test (after the config is locked, spec 6.2 no. 7)."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.dataset import CarSet, load_split  # noqa: E402
from src.data.local_features import N_LOCAL  # noqa: E402
from src.models.m2 import M1, M2  # noqa: E402
from src.models.pointnet2 import PointNet2Lite  # noqa: E402
from src.models.regdgcnn import RegDGCNN  # noqa: E402
from src.models.transolver import TransolverPC  # noqa: E402
from src.training.trainer import Trainer, cfg_hash, git_commit  # noqa: E402

MODELS = {"m2": {}, "m2occ": {}, "m2noxyz": {"use_xyz": False}, "m2non": {"use_nrm": False}, "m2s5": {}, "m1": {}, "b2": {}, "b3": {}, "tsv": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="E1")
    ap.add_argument("--model", default="m2", choices=MODELS)
    ap.add_argument("--res", default="R2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit", type=int, help="only the first N train cars, val = the same cars")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--accum", type=int)
    ap.add_argument("--ema", type=float)
    ap.add_argument("--warmup-epochs", type=int)
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"])
    ap.add_argument("--max-hours", type=float, help="stop cleanly after this long (spec 8.4)")
    ap.add_argument("--fourier", type=int, default=0, help="number of Fourier bands for query coordinates (0 = spec)")
    ap.add_argument("--train-n", type=int, help="use a category-stratified random subset of N train cars (seed 0); scale --epochs yourself to keep the number of optimizer steps")
    ap.add_argument("--calib-k", type=int, help="E2A/E2B/E2C only (prereg addenda 1-2): add the first K cars of the fixed 25-car calibration pool (from the held-out style test set) to train; the 25 calibration cars are always removed from test (225 test cars)")
    ap.add_argument("--calib-rep", type=int, default=1, help="repeat each calibration car this many times in the train list")
    ap.add_argument("--calib-select", default="prefix", choices=["prefix", "random", "kcenter", "novel"], help="prereg addendum 3: how the k calibration cars are chosen. prefix = first k of the 25-car pool (225 test cars, addenda 1-2); the others choose k of the 125 candidate cars (first 125 of the permutation) and test on the other 125: random (draw --calib-draw), kcenter (greedy k-center on R1 SDF), novel (greedy k-center seeded with the train cars)")
    ap.add_argument("--calib-draw", type=int, default=0, help="random draw index for --calib-select random")
    ap.add_argument("--calib-fold", type=int, help="Two-stage correction study T2 (out-of-fold retrain): leave out fold F (of 5, np.array_split(default_rng(3000 + 100 draw + k).permutation(k), 5)) of the chosen calibration cars from training and append them to the test cars, so their predictions are saved")
    ap.add_argument("--lam-int", type=float, help="prereg addendum 3: weight of the per-car integral loss (mean n_z*err and mean n_x*err, squared) added to the Huber loss")
    ap.add_argument("--n-points", type=int, help="B3 only: points per car per step and per evaluation chunk (default 10,000, the authors' value)")
    ap.add_argument("--val-every", type=int, help="validate every N epochs (and at the last); default every epoch")
    ap.add_argument("--val-n", type=int, help="validate on the first N val cars only (test is always the full set)")
    ap.add_argument("--patience", type=int, help="tsv: early stopping, epochs without validation improvement")
    ap.add_argument("--step-epochs", type=int, help="tsv: StepLR period in epochs")
    ap.add_argument("--tag", default="", help="run_id suffix, e.g. T0lr1e-4")
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--no-decoder-xyz", action="store_true", help="M2 family: do not give the query coordinates to the decoder (as M2noXYZ); default off")
    ap.add_argument("--no-global-vector", action="store_true", help="M2 family: do not give the global vector to the decoder; default off")
    ap.add_argument("--local-feats", action="store_true", help="M2 family: add the 6 local surface features per point (src/data/local_features.py; needs data/cache/<id>/localfeat.npy from scripts/data/local_features.py); default off")
    ap.add_argument("--print-cfg-hash", action="store_true", help="print run_id and the cfg_hash this run would record (M2 family) and exit")
    ap.add_argument("--save-val-pred", action="store_true", help="with --eval-test: also save val_pred_fp16.npy (per-point validation predictions, same order as val_per_car.csv); default off")
    ap.add_argument("--save-ckpt", action="store_true", help="with --eval-test: also save results/<run_id>/ckpt_ema.pt (EMA weights of best.pt, mu, sd, cfg; ~20 MB, git-ignored); default off")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    a = ap.parse_args()
    explicit = {k for k in ("epochs", "lr", "batch", "accum", "ema", "amp") if getattr(a, k) is not None} | ({"val_every"} if a.val_every else set()) | ({"warmup_epochs"} if a.warmup_epochs is not None else set())

    cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    mcfg = yaml.safe_load((ROOT / "configs" / "model_lite.yaml").read_text(encoding="utf-8"))
    for k in ("epochs", "lr", "batch", "accum", "ema", "amp"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.lam_int:
        cfg["lam_int"] = a.lam_int
    for flag in ("no_decoder_xyz", "no_global_vector", "local_feats"):  # recorded only when set, so the cfg_hash of every default run is unchanged
        if getattr(a, flag):
            cfg[flag] = True
    if a.val_every:
        cfg["val_every"] = a.val_every
    if a.warmup_epochs is not None:
        cfg["warmup_epochs"] = a.warmup_epochs
    run_id = f"{'OVERFIT' + str(a.overfit) if a.overfit else a.split}_{a.model.upper()}{'F' + str(a.fourier) if a.fourier else ''}_{a.res}_s{a.seed}{'_' + a.tag if a.tag else ''}"
    kind = "occ" if a.model == "m2occ" else "sdf"
    task = "cd" if a.model == "m1" else "cp"

    if a.print_cfg_hash:
        print(run_id, cfg_hash({**cfg, "model": a.model, "res": a.res, "split": a.split, "model_cfg": mcfg}))
        return
    s = load_split(a.split)
    if a.calib_k is not None:
        assert a.split in ("E2A", "E2B", "E2C"), "calibration pool is defined on the held-out style of E2A/E2B/E2C"
        order = [str(i) for i in np.random.default_rng(0).permutation(sorted(s["test"]))]
        k = a.calib_k
        if a.calib_select == "prefix":
            pool, test = order[:25], order[25:]
            assert len(test) == 225
            chosen = pool[:k]
        else:
            cand, test = order[:125], order[125:]
            assert len(test) == 125 and a.split == "E2A"
            if a.calib_select == "random":
                chosen = [cand[i] for i in np.random.default_rng(100 + a.calib_draw).permutation(125)[:k]]
            else:
                R1 = lambda ids: np.stack([np.load(ROOT / "data" / "cache" / i / "sdf_R1.npy").astype(np.float32).ravel() for i in ids])
                xc = R1(cand)
                dmin = np.full(125, np.inf)
                if a.calib_select == "novel":
                    for x in R1(s["train"]):
                        dmin = np.minimum(dmin, np.linalg.norm(xc - x, axis=1))
                    pick = []
                else:  # first centre: the candidate nearest to the candidate mean
                    pick = [int(np.argmin(np.linalg.norm(xc - xc.mean(0), axis=1)))]
                    dmin = np.linalg.norm(xc - xc[pick[0]], axis=1)
                while len(pick) < k:
                    pick.append(int(np.argmax(dmin)))
                    dmin = np.minimum(dmin, np.linalg.norm(xc - xc[pick[-1]], axis=1))
                chosen = [cand[i] for i in pick[:k]]
        hold = []
        if a.calib_fold is not None:
            folds = np.array_split(np.random.default_rng(3000 + 100 * a.calib_draw + k).permutation(k), 5)
            hold = [chosen[i] for i in folds[a.calib_fold]]
            chosen = [c for c in chosen if c not in hold]
            print("held-out fold cars (appended to the test cars):", hold, flush=True)
        print("calibration cars:", chosen, flush=True)
        s = {**s, "train": list(s["train"]) + chosen * a.calib_rep, "test": test + hold}
    tr_ids = s["train"][: a.overfit] if a.overfit else s["train"]
    if a.train_n:  # same N / n_categories cars from each present category, chosen with a fixed generator
        rng, cats = np.random.default_rng(0), sorted({i[0] for i in tr_ids})
        tr_ids = sorted(i for c in cats for i in rng.choice(sorted(x for x in tr_ids if x[0] == c), a.train_n // len(cats), replace=False))
    va_ids = tr_ids if a.overfit else (s["val"][: a.val_n] if a.val_n else s["val"])
    t0 = time.time()
    mult = 32 if a.model == "m2s5" else 16  # 5-stage encoder: grid sides must be divisible by 32
    train = CarSet(tr_ids, a.res, kind, mult=mult, local=a.local_feats)
    val = train if a.overfit else CarSet(va_ids, a.res, kind, mult=mult, local=a.local_feats)
    print(f"{run_id}: train {len(train)} val {len(val)} loaded in {time.time() - t0:.0f}s, device {a.device}", flush=True)

    if a.model == "b3":  # RegDGCNN: authors' 10,000 points; evaluation over the 100k pool in chunks of 10,000
        cfg.update(n_points=a.n_points or 10000, eval_chunk=a.n_points or 10000)
    if a.model == "b2":  # B2 takes 16,384 points per car, inference in blocks of 16,384 pool points
        cfg.update(n_points=16384, eval_chunk=16384)
    if a.model == "tsv":  # CarBench Transolver recipe; explicit CLI values (--lr, --epochs, --batch, --accum, --val-every) still win
        d = dict(lr=1e-3, weight_decay=0.0, ema=0.0, amp="none", warmup_epochs=0, epochs=1000, n_points=10000, eval_chunk=100000, step_epochs=150, gamma=0.1, patience=50, val_every=5, batch=2, accum=4)
        cfg.update({k: v for k, v in d.items() if k not in explicit})
        cfg["n_points"] = a.n_points or cfg["n_points"]
        cfg["patience"], cfg["step_epochs"] = a.patience if a.patience is not None else cfg["patience"], a.step_epochs or cfg["step_epochs"]
    model = TransolverPC() if a.model == "tsv" else RegDGCNN() if a.model == "b3" else PointNet2Lite() if a.model == "b2" else (M1 if task == "cd" else M2)(chs=tuple(mcfg["chs"]) + ((128,) if a.model == "m2s5" else ()), in_ch=1, **MODELS[a.model], **({"use_xyz": False} if a.no_decoder_xyz else {}), **({"use_glob": False} if a.no_global_vector else {}), **({"n_local": N_LOCAL} if a.local_feats else {}), **({} if task == "cd" else {"proj": mcfg["proj"], "glob": mcfg["glob"], "fourier": a.fourier}))
    print("parameters:", sum(p.numel() for p in model.parameters()), "encoder:", sum(p.numel() for p in getattr(model, "enc", model).parameters()), flush=True)
    tr = Trainer(model, train, val, {**cfg, "model": a.model, "res": a.res, "split": a.split, "model_cfg": mcfg}, Path(a.runs) / run_id, a.device, a.seed, task)
    resumed = tr.epoch
    done = tr.fit(max_seconds=a.max_hours * 3600 if a.max_hours else None)
    status = "done" if done else "paused"

    if done and a.eval_test and not a.overfit:
        best = torch.load(Path(a.runs) / run_id / "best.pt", map_location=a.device, weights_only=False)
        tr.ema.load_state_dict(best["ema"])
        out = ROOT / "results" / run_id
        out.mkdir(parents=True, exist_ok=True)
        for role, ids in (("val", va_ids), ("test", s["test"])):
            keep = role == "test" or a.save_val_pred
            df = tr.evaluate(val if role == "val" else CarSet(ids, a.res, kind, mult=mult, local=a.local_feats), keep=keep)
            df.to_csv(out / f"{role}_per_car.csv", index=False)
            if keep and task == "cp":  # in the CSV `id` order; same point order as pool.npz
                np.save(out / f"{role}_pred_fp16.npy", np.stack(tr.preds))
        if a.save_ckpt:
            torch.save({"ema": best["ema"], "mu": best["mu"], "sd": best["sd"], "cfg": best["cfg"], "cfg_hash": best["cfg_hash"]}, out / "ckpt_ema.pt")
    f = ROOT / "experiments.csv"
    new = not f.exists()
    with open(f, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run_id", "cfg_hash", "commit", "seed", "split", "res", "status", "epochs_done", "best_val", "gpu_hours_this_call", "device"])
        w.writerow([run_id, cfg_hash(tr.cfg), git_commit(), a.seed, a.split, a.res, status, tr.epoch, f"{tr.best:.5f}", f"{(time.time() - t0) / 3600:.3f}", a.device])
    print(json.dumps({"run_id": run_id, "status": status, "epochs_done": tr.epoch, "resumed_from": resumed, "best_val": tr.best}))


if __name__ == "__main__":
    main()
