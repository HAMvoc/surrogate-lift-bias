"""Label-budget study shared: sufficient statistics (calibration.build_stats) of the model predictions used by T1/T2, cached under results/label_budget/cache (git-ignored).
pair_stats(fold, family): the 250 test cars of the fold in the two-stage correction study/4 order (candidates = first 125, evaluation = last 125), seeds as in the component and cross-model study T2 script.
t3_stats(k, d): (stats of the 125 evaluation cars from the retrain run E2A_M2_R2_s0_cal{k}d{d}, stats of the k out-of-fold residual cars from the five fold runs, retrain-run stats without stage B)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.correction.calibration import build_stats  # noqa: E402
from scripts.correction.cross_model import DIRS  # noqa: E402
from scripts.diagnosis.regions import labeler as labeler_v2  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

CACHE = ROOT / "results" / "label_budget" / "cache"
RAWS = [ROOT / "results" / "two_stage" / "T2" / "raw" / "results", ROOT / "results" / "components" / "T3" / "raw" / "results"]
PAIRS = [("E2A", m) for m in ("B0", "B1", "B2", "B2_300ep", "B3", "M2")] + [("E2C", "B3")]


def _load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _save(z, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **z)


def perm_ids(fold):
    return [str(i) for i in np.random.default_rng(0).permutation(sorted(load_split(fold)["test"]))]


def pair_stats(fold, fam):
    f = CACHE / f"stats_{fold}_{fam}.npz"
    if f.exists():
        return _load(f)
    dirs = [ROOT / "results" / d.format(f=fold) for d in DIRS[fam]]
    preds = [np.load(d / "test_pred_fp16.npy", mmap_mode="r") for d in dirs]
    pos = [{i: j for j, i in enumerate(pd.read_csv(d / "test_per_car.csv").id)} for d in dirs]
    z = build_stats(perm_ids(fold), lambda s, sid: preds[s][pos[s][sid]], len(dirs), labeler_v2, 8)
    _save(z, f)
    return z


def run_dir(name):
    for r in RAWS:
        if (r / name / "test_pred_fp16.npy").exists():
            return r / name
    raise FileNotFoundError(name)


def t3_stats(k, d):
    f1, f2, f3 = (CACHE / f"t3_eval_k{k}_d{d}.npz", CACHE / f"t3_oof_k{k}_d{d}.npz", None)
    if f1.exists() and f2.exists():
        return _load(f1), _load(f2)
    ev_ids = perm_ids("E2A")[125:]
    fin = run_dir(f"E2A_M2_R2_s0_cal{k}d{d}")
    pr = np.load(fin / "test_pred_fp16.npy", mmap_mode="r")
    pos = {i: j for j, i in enumerate(pd.read_csv(fin / "test_per_car.csv").id)}
    ze = build_stats(ev_ids, lambda _s, sid: pr[pos[sid]], 1, labeler_v2, 8)
    oof_ids, preds = [], {}
    for f in range(5):
        r = run_dir(f"E2A_M2_R2_s0_cal{k}d{d}f{f}")
        ids = list(pd.read_csv(r / "test_per_car.csv").id)
        pf = np.load(r / "test_pred_fp16.npy", mmap_mode="r")
        for j, i in enumerate(ids):
            if i not in set(ev_ids):
                oof_ids.append(i); preds[i] = pf[j]
    zo = build_stats(oof_ids, lambda _s, sid: preds[sid], 1, labeler_v2, 8)
    _save(ze, f1); _save(zo, f2)
    return ze, zo
