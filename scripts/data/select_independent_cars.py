"""Independent-set study, task T1: selection of 250 new Estateback cars.
Candidates: Estateback cars (S underbody, WM mirror) that are not among the 750 cars of the subset; strata by wheel code (WW, WWC) with the quotas of the old test set; order per stratum
default_rng(11000 + j).permutation(sorted(ids)). A car is excluded if its group_id equals one of the 750 or if the front-half R1 SDF distance to one of the 750 is <= eps (0.6558281183242798).
Only the STL is downloaded here (shape only, no pressure); excluded STLs are deleted. Writes results/independent_set/screen_log.csv (resumable) and data/splits/fresh_estateback.csv.
Usage: python scripts/data/select_independent_cars.py [--workers 3]"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from src.data.grid import load_cfg  # noqa: E402

CFG = load_cfg()
SRC = CFG["source"]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "independent_set"
EPS = 0.6558281183242798
FRONT_X = 0.45
QUOTA = {"WW": 111, "WWC": 139}  # wheel-code mix of the 250 old Estateback test cars
SEEDS = {"WW": 11000, "WWC": 11001}
CAP = 39.5e9  # bytes: the 40 GB download cap of the independent set; the STL screening may use the cap minus the pressure files (VTK_BYTES each) of the cars kept so far
VTK_BYTES = 28e6  # a pressure file of an Estateback car is about 27 MB
MIN_FREE = 40e9  # stop when the drive D: has less than this free (the request: keep at least 30 GB)


def screen_one(sid):
    """R1 SDF of one car from its STL (code of scripts/data/preprocess.py), float16 as in the cache."""
    from src.data.sdf import load_car, make_scene, sdf_volumes
    box, grids, L, tau = CFG["box"], CFG["grids"], CFG["frame"]["l_ref_m"], CFG["sdf"]["tau"]
    verts, faces, off, wt = load_car(RAW / SRC["stl_dir"] / f"{sid}.stl")
    vol, _, _ = sdf_volumes(make_scene(verts, faces), grids, box, L, tau, wt, names=("R1",))
    return sid, vol["R1"].astype(np.float16)


def group_ids(ids):
    params = pd.read_csv((ROOT / SRC["params_csv"]).resolve()).set_index("Experiment")
    pcols = [c for c in params.columns if not c.startswith(("Average", "Std"))]
    return {i: "g" + hashlib.md5(",".join(f"{v:.4f}" for v in r).encode()).hexdigest()[:10] for i, r in zip(ids, params.loc[ids, pcols].to_numpy())}


def download(sid):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(SRC["hf_repo"], f"{SRC['stl_dir']}/{sid}.stl", repo_type="dataset", local_dir=RAW))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "data" / "manifests" / "manifest.csv")
    ok = man[man.status == "ok"]
    base_ids = list(ok.sample_id)
    assert len(base_ids) == 750
    base_grp = dict(zip(ok.sample_id, ok.group_id))
    assert group_ids(base_ids[:20]) == {i: base_grp[i] for i in base_ids[:20]}, "group_id rule does not reproduce the manifest"
    test_ids = set(json.load(open(ROOT / "data" / "manifests" / "splits" / "E2A.json"))["test"])
    allE = [i for i in pd.read_csv(RAW / "drag" / "DrivAerNetPlusPlus_Cd_8k_Updated.csv").ID if i[0] == "E"]
    cand = [i for i in allE if i not in test_ids and i not in set(base_ids)]
    # the 50 reserve Estateback cars are candidates (they are not in the 750 'ok' cars); the 250 old test cars are the 'ok' Estateback cars
    assert len(cand) == 1136, len(cand)
    cgrp = group_ids(cand)
    strata = {w: [i for i in sorted(cand) if i.split("_")[2] == w] for w in QUOTA}
    order = {w: [str(i) for i in np.random.default_rng(SEEDS[w]).permutation(strata[w])] for w in QUOTA}
    n_front = int((FRONT_X - CFG["box"]["x"][0]) / (CFG["box"]["x"][1] - CFG["box"]["x"][0]) * CFG["grids"]["R1"][0])
    bank = np.stack([np.load(ROOT / "data" / "cache" / i / "sdf_R1.npy")[..., :n_front].astype(np.float32).ravel() for i in base_ids])
    bgroups = set(base_grp.values())
    logf = OUT / "screen_log.csv"
    log = pd.read_csv(logf) if logf.exists() else pd.DataFrame(columns=["id", "wheel", "group_twin", "min_front_dist", "decision", "stl_bytes"])
    rows = log.to_dict("records")
    done = set(log.id)
    got = {w: sum((r["wheel"] == w and r["decision"] == "keep") for r in rows) for w in QUOTA}
    total_bytes = sum(r["stl_bytes"] for r in rows)
    print("resume:", got, len(rows), "screened", flush=True)
    # group twins are known from the design parameters alone: they are excluded without downloading the STL (same rule, fewer downloads)
    for w in QUOTA:
        for i in order[w]:
            if i not in done and cgrp[i] in bgroups:
                rows.append({"id": i, "wheel": w, "group_twin": True, "min_front_dist": float("nan"), "decision": "exclude", "stl_bytes": 0})
                done.add(i)
    pend = {w: [i for i in order[w] if i not in done] for w in QUOTA}
    step = {"WW": 9, "WWC": 11}  # batches in the ratio of the quotas, so that a stop for the download cap leaves the wheel-code mix close to that of the old test set
    stop = False
    while not stop and any(got[w] < QUOTA[w] and pend[w] for w in QUOTA):
        for w in QUOTA:
            if got[w] >= QUOTA[w] or not pend[w]:
                continue
            if total_bytes + VTK_BYTES * sum(got.values()) > CAP or shutil.disk_usage(str(ROOT)).free < MIN_FREE:
                print("STOP: download cap or free-space limit reached", total_bytes / 1e9, shutil.disk_usage(str(ROOT)).free / 1e9, flush=True)
                stop = True
                break
            nb = min(QUOTA[w] - got[w], step[w])
            batch, pend[w] = pend[w][:nb], pend[w][nb:]
            t0 = time.time()
            with cf.ThreadPoolExecutor(4) as ex:
                paths = list(ex.map(download, batch))
            with cf.ProcessPoolExecutor(a.workers) as ex:
                res = dict(ex.map(screen_one, batch))
            for sid, p in zip(batch, paths):
                v = res[sid][..., :n_front].astype(np.float32).ravel()
                d = np.sqrt(np.maximum(((bank - v) ** 2).sum(1), 0)).min()
                dec = "keep" if d > EPS else "exclude"
                sz = p.stat().st_size
                total_bytes += sz
                rows.append({"id": sid, "wheel": w, "group_twin": False, "min_front_dist": float(d), "decision": dec, "stl_bytes": sz})
                if dec == "keep":
                    got[w] += 1
                else:
                    p.unlink()
            pd.DataFrame(rows).to_csv(logf, index=False)
            print(f"{w}: kept {got[w]}/{QUOTA[w]}, screened {len(rows)}, {total_bytes / 1e9:.1f} GB, batch {time.time() - t0:.0f} s", flush=True)
    L = pd.DataFrame(rows)
    keep = L[L.decision == "keep"].sort_values("id")
    keep.assign(group_id=[cgrp[i] for i in keep.id])[["id", "wheel", "group_id", "min_front_dist"]].rename(columns={"id": "sample_id"}).to_csv(ROOT / "data" / "splits" / "fresh_estateback.csv", index=False)
    print("selected", len(keep), "excluded", int((L.decision == "exclude").sum()), "of which group twins", int(L.group_twin.sum()), flush=True)


if __name__ == "__main__":
    main()
