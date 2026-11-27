"""Independent-set study, task T1: pressure files, preprocessing and per-car geometry of the selected new Estateback cars (data/splits/fresh_estateback.csv).
1) download the VTK (pressure) of each selected car (the STL was downloaded by select_independent_cars.py); 2) scripts/data/preprocess.py (same configuration hash) -> data/cache/<id>/;
3) scripts/data/car_geometry.py `one` -> data/manifests/car_geometry_fresh.csv (area, frontal area, inclusion rate); 4) v2 region labels (wheel fit of scripts/diagnosis/regions.py) -> results/independent_set/labels_v2_fresh.npz;
5) the raw STL and VTK of the new cars are deleted. Resumable. Usage: python scripts/data/prepare_independent_cars.py [--workers 4]"""
import argparse
import concurrent.futures as cf
import csv
import subprocess
import sys
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
GEOF = ROOT / "data" / "manifests" / "car_geometry_fresh.csv"


def dl_vtk(sid):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(SRC["hf_repo"], f"{SRC['vtk_dir']}/{sid}.vtk", repo_type="dataset", local_dir=RAW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    ids = list(pd.read_csv(ROOT / "data" / "splits" / "fresh_estateback.csv").sample_id)
    todo = [i for i in ids if not (ROOT / "data" / "cache" / i / "meta.json").exists()]
    print(len(ids), "selected,", len(todo), "to preprocess", flush=True)
    missing = [i for i in todo if not (RAW / SRC["vtk_dir"] / f"{i}.vtk").exists()]
    with cf.ThreadPoolExecutor(4) as ex:
        list(ex.map(dl_vtk, missing))
    print("pressure files downloaded:", len(missing), flush=True)
    if todo:
        subprocess.run([sys.executable, "scripts/data/preprocess.py", "--workers", str(a.workers), *todo], cwd=ROOT, check=True)
    import json
    status = {i: json.loads((ROOT / "data" / "cache" / i / "meta.json").read_text())["status"] for i in ids}
    bad = [i for i, s in status.items() if s != "ok"]
    print("preprocess status not ok:", bad, flush=True)
    from scripts.data import car_geometry as cg
    done = {r["sample_id"] for r in csv.DictReader(open(GEOF, encoding="utf-8"))} if GEOF.exists() else set()
    new = not GEOF.exists()
    with open(GEOF, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["sample_id", "area_total_m2", "frontal_area_m2", "incl_rate_per_m2"])
        for i in ids:
            if i in done or i in bad:
                continue
            sid, area, front, lam = cg.one(i)
            w.writerow([sid, f"{area:.5f}", f"{front:.5f}", f"{lam:.3f}"])
            fh.flush()
    from scripts.diagnosis import regions as rv
    labs = {}
    for i in ids:
        if i in bad:
            continue
        p = np.load(ROOT / "data" / "cache" / i / "pool.npz")
        lab, _ = rv.labels_v2(p["xyz"], p["normal"])
        labs[i] = lab.astype(np.int8)
    np.savez_compressed(OUT / "labels_v2_fresh.npz", **labs)
    if not a.keep_raw:
        for i in ids:
            for d, e in (("stl_dir", "stl"), ("vtk_dir", "vtk")):
                f = RAW / SRC[d] / f"{i}.{e}"
                if f.exists():
                    f.unlink()
    print("done:", len(labs), "cars with geometry and labels; bad:", bad, flush=True)


if __name__ == "__main__":
    main()
