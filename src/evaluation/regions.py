"""Assign a surface region to each point per configs/regions.yaml (geometric rules on coordinates normalised by the car's own bounding box)."""
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_regions(path=None):
    cfg = yaml.safe_load(Path(path or ROOT / "configs" / "regions.yaml").read_text(encoding="utf-8"))
    names = [cfg["default"]] + list(dict.fromkeys(r["name"] for r in cfg["rules"]))
    return cfg, names


def assign_regions(xyz, nrm, cfg=None, l_ref=5.0):
    """(N,3) coordinates and (N,3) normals of one car -> (N,) region index, in the `names` order of load_regions (0 = default)."""
    cfg, names = load_regions() if cfg is None else (cfg, [cfg["default"]] + list(dict.fromkeys(r["name"] for r in cfg["rules"])))
    x, y, z = np.asarray(xyz, np.float64).T
    nrm = np.asarray(nrm, np.float64)
    var = {"u": (x - x.min()) / (x.max() - x.min()), "v": np.abs(y) / np.abs(y).max(), "w": (z - z.min()) / (z.max() - z.min()),
           "nx": nrm[:, 0], "ny": nrm[:, 1], "nz": nrm[:, 2], "any": np.abs(nrm[:, 1])}
    out = np.zeros(len(x), np.int8)
    free = np.ones(len(x), bool)
    for rule in cfg["rules"]:
        ok = free.copy()
        if "disc" in rule:
            d = rule["disc"]
            ok &= ((var["u"] - d["u"]) * (x.max() - x.min()) * l_ref) ** 2 + (z * l_ref - d["zm"]) ** 2 < d["r"] ** 2
        for k, (lo, hi) in ((k, v) for k, v in rule.items() if k not in ("name", "disc")):
            ok &= (var[k] >= lo) if lo is not None else True
            ok &= (var[k] <= hi) if hi is not None else True
        out[ok] = names.index(rule["name"])
        free &= ~ok
    return out
