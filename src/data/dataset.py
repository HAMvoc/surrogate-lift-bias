"""Load the cache into RAM (spec 5.5, 8): SDF/occupancy Z-padded to a multiple of 16, point pool with raw and clipped Cp, Cd."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.grid import box_arrays, load_cfg

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
SPLITS = ROOT / "data" / "manifests" / "splits"


def load_split(name):
    return json.loads((SPLITS / f"{name}.json").read_text())


def category_of(sid):
    return {"F": "Fastback", "N": "Notchback", "E": "Estateback"}[sid[0]]


class CarSet:
    """Full tensors of a car set. vol (n, Dp, H, W) fp16; xyz/nrm/cp (n, P, ...) fp32.
    kind 'sdf' pads with +1 (far outside the car), 'occ' pads with 0 (R2 is divisible by 16 so it needs no padding)."""

    def __init__(self, ids, res="R2", kind="sdf", cfg=None, cache=CACHE, workers=32, mult=16, local=False):
        cfg = cfg or load_cfg()
        self.ids, self.res, self.kind = list(ids), res, kind
        lo, hi = box_arrays(cfg["box"])
        self.lo, self.hi = torch.tensor(lo, dtype=torch.float32), torch.tensor(hi, dtype=torch.float32)
        self.clip = tuple(cfg["target"]["cp_clip"])
        self.D0 = cfg["grids"][res][2]
        self.Dp = -(-self.D0 // mult) * mult  # mult = 2 ** n_stages (16 for the 4-stage encoder)
        def load_one(i):
            v = np.load(Path(cache) / i / (f"sdf_{res}.npy" if kind == "sdf" else "occ_R2.npy")).astype(np.float16)
            v = np.pad(v, ((0, self.Dp - self.D0), (0, 0), (0, 0)), constant_values=1.0 if kind == "sdf" else 0.0)
            p = np.load(Path(cache) / i / "pool.npz")
            lf = np.load(Path(cache) / i / "localfeat.npy") if local else None  # (P, 6) float16, scripts/data/local_features.py
            return v, (p["xyz"], p["normal"], p["cp"], p["area"]), lf

        # Multithreaded: Kaggle Input disk has very high per-file latency (measured: 1360 s for ~1350 files per process, sequential reads).
        with ThreadPoolExecutor(workers) as ex:
            loaded = list(ex.map(load_one, self.ids))
        vols, pools = [x[0] for x in loaded], [x[1] for x in loaded]
        self.loc = torch.from_numpy(np.stack([x[2] for x in loaded])) if local else None
        self.vol = torch.from_numpy(np.stack(vols))
        self.xyz, self.nrm, self.cp_raw, self.area = (torch.from_numpy(np.stack(x)) for x in zip(*pools))
        self.cp = self.cp_raw.clamp(*self.clip)
        cd = pd.read_csv(ROOT / "data" / "raw" / load_cfg()["source"]["cd_csv"]).set_index("ID")["Drag_Value"]
        self.cd = torch.tensor([cd[i] for i in self.ids], dtype=torch.float32)

    def __len__(self):
        return len(self.ids)

    def ndc(self, xyz):
        """Coordinates (L_ref) -> [-1, 1] on the original bounding box, order (x, y, z)."""
        return 2 * (xyz - self.lo) / (self.hi - self.lo) - 1

    def grid_coords(self, ndc):
        """NDC on the original box -> NDC of the Z-padded grid (same voxel size, extended toward high z)."""
        g = ndc.clone()
        g[..., 2] = (ndc[..., 2] + 1) * self.D0 / self.Dp - 1
        return g

    def batch(self, idx, n_pts=None, gen=None, pts=None):
        """idx: list of car indices. n_pts: random sample without replacement from the pool (already area-proportional); pts: fixed point indices."""
        idx = torch.as_tensor(idx)
        P = self.xyz.shape[1]
        if pts is None:
            pts = torch.randperm(P, generator=gen)[:n_pts] if n_pts else torch.arange(P)
        ndc = self.ndc(self.xyz[idx][:, pts])
        return {"vol": self.vol[idx].float().unsqueeze(1), "ndc": ndc, "g": self.grid_coords(ndc),
                "nrm": self.nrm[idx][:, pts], "cp": self.cp[idx][:, pts], "cp_raw": self.cp_raw[idx][:, pts], "cd": self.cd[idx],
                **({"loc": self.loc[idx][:, pts].float()} if self.loc is not None else {})}
