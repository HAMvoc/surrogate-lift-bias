"""Zero-shot checks study, task T1: inference-time interventions on saved M2 R2 checkpoints (no training).
(a) global vector -> train mean; (b) roof_rear decoder xyz shifted by D metres; (c) roof_rear trilinear sampling coordinates shifted; (d) both;
(e_b/e_c/e_d) the same for roof_front (pseudo-control). Primary direction: toward the nose (x decreases); secondary (E2A test only): toward the tail.
Writes results/zero_shot_checks/T1/per_car.csv.gz (one row per set, seed, car, variant) and then summarises it (--analyze only re-summarises).
Usage: python scripts/interventions/inference_interventions.py [--analyze]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.force_alignment import REGIONS, car_terms, region_labels  # noqa: E402
from src.data.dataset import CACHE, CarSet, load_split  # noqa: E402
from src.models.m2 import M2, global_vec  # noqa: E402

OUT = ROOT / "results" / "zero_shot_checks" / "T1"
K = ROOT / "kaggle" / "_out"
DELTAS = (0.2, 0.4, 0.6)
L_REF = 5.0
RR, RF = REGIONS.index("roof_rear"), REGIONS.index("roof_front")
SETS = (("E2A", "test"), ("E2B", "test"), ("E2C", "test"), ("E2A", "val"))


def ckpt_path(fold, seed):
    return K / ("voxel-sdf-e2-s0" if seed == 0 else "voxel-sdf-e2-s12") / "runs" / f"{fold}_M2_R2_s{seed}" / "best.pt"


def load_model(fold, seed, dev):
    c = torch.load(ckpt_path(fold, seed), map_location=dev, weights_only=False)
    m = c["cfg"]["model_cfg"]
    model = M2(chs=tuple(m["chs"]), in_ch=1, proj=m["proj"], glob=m["glob"], fourier=0).to(dev)
    model.load_state_dict(c["ema"])
    return model.eval(), c["mu"], c["sd"]


def encode(model, vol):
    feats = model.enc(vol)
    return [r(x).float() for r, x in zip(model.reduce, feats)], model.glob(global_vec(feats[-1]))


def decode(model, rf, gv, g, ndc, nrm):
    """Same computation as M2.forward after the encoder (use_xyz, use_nrm on, no Fourier features)."""
    B, N, _ = g.shape
    grid = g.float().view(B, N, 1, 1, 3)
    f = [F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False).view(B, -1, N).transpose(1, 2) for x in rf]
    parts = f + [gv.unsqueeze(1).expand(B, N, -1).to(f[0].dtype), ndc.float().to(f[0].dtype), nrm.to(f[0].dtype)]
    return model.mlp(torch.cat(parts, -1)).squeeze(-1)


def train_mean_gv(fold, models, dev):
    """Mean of the global vector over the train cars of the fold, per model."""
    ids = load_split(fold)["train"]
    vols = []
    for i in ids:
        v = np.load(CACHE / i / "sdf_R2.npy").astype(np.float16)
        vols.append(torch.from_numpy(v))
    vols = torch.stack(vols)
    out = []
    for model, *_ in models:
        acc = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for a in range(0, len(vols), 16):
                _, gv = encode(model, vols[a:a + 16].to(dev).float().unsqueeze(1))
                acc.append(gv.float().cpu())
        out.append(torch.cat(acc).mean(0, keepdim=True).to(dev))
    return out


def variants(primary_only):
    v = [("none", None), ("a", None)]
    for sign, name in ((-1, "nose"), (+1, "tail")) if not primary_only else ((-1, "nose"),):
        for kind in ("b", "c", "d", "eb", "ec", "ed"):
            for d in DELTAS:
                v.append((f"{kind}_{d}_{name}", (kind, sign * d)))
    return v


def main_run(dev):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for fold, role in SETS:
        models = [load_model(fold, s, dev) for s in (0, 1, 2)]
        gvm = train_mean_gv(fold, models, dev)
        ids = load_split(fold)[role]
        data = CarSet(ids, "R2", "sdf")
        sd_, mu_ = None, None
        vs = variants(primary_only=not (fold == "E2A" and role == "test"))
        print(f"{fold} {role}: {len(ids)} cars, {len(vs)} variants, train-mean global vector done ({time.time() - t0:.0f} s)", flush=True)
        dndc = lambda d: 2 * (d / L_REF) / float(data.hi[0] - data.lo[0])
        for j, sid in enumerate(data.ids):
            b = data.batch([j])
            vol, g, ndc, nrm = (b[k].to(dev) for k in ("vol", "g", "ndc", "nrm"))
            lab = region_labels(data.xyz[j].numpy(), data.nrm[j].numpy())
            idx = {RR: torch.from_numpy(np.where(lab == RR)[0]).to(dev), RF: torch.from_numpy(np.where(lab == RF)[0]).to(dev)}
            cp, nr, ar = data.cp_raw[j].numpy(), data.nrm[j].numpy(), data.area[j].numpy()
            ref_c = np.clip(cp.astype(np.float64), -10, 1)
            for s, (model, mu, sd) in enumerate(models):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    rf, gv = encode(model, vol)
                    base = decode(model, rf, gv, g, ndc, nrm)
                    if j == 0 and s == 0:  # the split decode must reproduce the model's own forward
                        full = model(vol, g, ndc, nrm)
                        assert (full - base).abs().max().item() < 1e-3, "split forward differs from M2.forward"
                    to_np = lambda o: o[0].float().cpu().numpy().astype(np.float64) * sd + mu
                    base_np = to_np(base)
                    for name, spec in vs:
                        if name == "none":
                            p = base_np
                        elif name == "a":
                            p = to_np(decode(model, rf, gvm[s], g, ndc, nrm))
                        else:
                            kind, dx = spec
                            reg = RF if kind.startswith("e") else RR
                            ii = idx[reg]
                            k2 = kind[-1]  # b, c or d
                            ndc_s = ndc[:, ii].clone()
                            g_s = g[:, ii].clone()
                            if k2 in ("b", "d"):
                                ndc_s[..., 0] += dndc(dx)
                            if k2 in ("c", "d"):
                                g_s[..., 0] += dndc(dx)
                            sub = to_np(decode(model, rf, gv, g_s, ndc_s, nrm[:, ii]))
                            p = base_np.copy()
                            p[ii.cpu().numpy()] = sub
                        t = car_terms(p, cp, nr, ar, sid, lab)
                        rows.append({"fold": fold, "role": role, "seed": s, "id": sid, "variant": name, "rel_l2": float(np.linalg.norm(p - ref_c) / np.linalg.norm(ref_c)),
                                     **{k: v for k, v in t.items() if k in ("dL", "dD") or k.startswith(("reg_dL_", "reg_dD_"))}})
            if (j + 1) % 50 == 0:
                print(f"  {fold} {role} car {j + 1}/{len(ids)} ({time.time() - t0:.0f} s)", flush=True)
        del data
        torch.cuda.empty_cache()
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "per_car.csv.gz", index=False, compression="gzip")
    print("saved", len(d), "rows in", f"{time.time() - t0:.0f} s")


B = 2000


def ci(x):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(0).integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def analyze():
    d = pd.read_csv(OUT / "per_car.csv.gz")
    A = d.drop(columns=["seed"]).groupby(["fold", "role", "id", "variant"]).mean(numeric_only=True).reset_index()  # seed mean per car
    res = []
    for (fold, role, v), g in A.groupby(["fold", "role", "variant"]):
        r = {"fold": fold, "role": role, "variant": v, "n": len(g), "rel_l2": g.rel_l2.mean()}
        for k in ("dL", "dD", *[f"reg_dL_{x}" for x in REGIONS], *[f"reg_dD_{x}" for x in REGIONS]):
            m, lo, hi = ci(g[k])
            r[k], r[k + "_lo"], r[k + "_hi"] = m, lo, hi
        res.append(r)
    R = pd.DataFrame(res)
    R.to_csv(OUT / "T1_results.csv", index=False)
    return R, d, A


def summarise(R, d, A):
    g0 = R[(R.fold == "E2A") & (R.role == "test") & (R.variant == "none")].iloc[0]
    gate = abs(g0.dL - 183.9) <= 0.5
    lines = ["# T1: inference-time interventions on M2 checkpoints", "",
             f"Gate (E2A test, 3 seeds, no intervention, recomputed from the checkpoints): bias C_L,p = {g0.dL:+.1f} [{g0.dL_lo:+.1f}; {g0.dL_hi:+.1f}] counts (n = {int(g0.n)}) -> {'PASS' if gate else 'FAIL'} (required 183.9 +- 0.5).", ""]
    if not gate:
        (OUT / "T1_summary.md").write_text("\n".join(lines), encoding="utf-8")
        raise SystemExit("gate failed")
    E = R[(R.fold == "E2A") & (R.role == "test")].set_index("variant")
    b_rr, b_rf, b_tot = E.loc["none", "reg_dL_roof_rear"], E.loc["none", "reg_dL_roof_front"], E.loc["none", "dL"]
    rows = []
    for v in E.index:
        r = E.loc[v]
        rows.append({"variant": v, "bias C_L,p [CI]": f"{r.dL:+.1f} [{r.dL_lo:+.1f}; {r.dL_hi:+.1f}]", "roof_rear": f"{r.reg_dL_roof_rear:+.1f}", "roof_front": f"{r.reg_dL_roof_front:+.1f}",
                     "red. total": f"{100 * (1 - r.dL / b_tot):.0f}%", "red. roof_rear": f"{100 * (1 - r.reg_dL_roof_rear / b_rr):.0f}%", "red. roof_front": f"{100 * (1 - r.reg_dL_roof_front / b_rf):.0f}%",
                     "bias C_D,p": f"{r.dD:+.1f}", "rel L2 test": f"{r.rel_l2:.4f}", "rel L2 E2A val": (f"{R[(R.fold == 'E2A') & (R.role == 'val') & (R.variant == v)].rel_l2.iloc[0]:.4f}" if len(R[(R.fold == 'E2A') & (R.role == 'val') & (R.variant == v)]) else "-")})
    red = lambda v, reg, base: 1 - E.loc[v, f"reg_dL_{reg}"] / base
    # rule 1
    hits = []
    for kind in ("b", "c"):
        for dlt in DELTAS:
            r_rear = red(f"{kind}_{dlt}_nose", "roof_rear", b_rr)
            r_front = red(f"e{kind}_{dlt}_nose", "roof_front", b_rf)
            hits.append((kind, dlt, r_rear, r_front, r_rear >= 0.5 and abs(r_front) <= 0.2))
    rule1 = [h for h in hits if h[4]]
    nose = [v for v in E.index if v.endswith("_nose") and v[0] in "bcd" and v[:2] != "eb"]
    max_rear = {k: max(red(f"{k}_{dlt}_nose", "roof_rear", b_rr) for dlt in DELTAS) for k in ("b", "c", "d")}
    red_a = 1 - E.loc["a", "dL"] / b_tot
    if rule1:
        verdict = "RULE 1: coordinate shortcut at branch " + ", ".join(f"({h[0]}) {'decoder xyz' if h[0] == 'b' else 'trilinear sampling'} at D = {h[1]} m (roof_rear bias reduced {100 * h[2]:.0f}%, control roof_front {100 * h[3]:.0f}%)" for h in rule1)
    elif red_a >= 0.2 and all(v < 0.2 for v in max_rear.values()):
        verdict = f"RULE 2: the source is the global vector ((a) reduces the total bias by {100 * red_a:.0f}%; (b), (c), (d) reduce roof_rear by at most {100 * max(max_rear.values()):.0f}%)"
    elif red_a <= 0.2 and all(v <= 0.2 for v in max_rear.values()):
        verdict = f"RULE 3: shortcut hypothesis rejected at inference level ((a) reduces the total bias by {100 * red_a:.0f}%; best roof_rear reductions (b) {100 * max_rear['b']:.0f}%, (c) {100 * max_rear['c']:.0f}%, (d) {100 * max_rear['d']:.0f}%)"
    else:
        verdict = f"RULE 4: mixed / inconclusive ((a) total {100 * red_a:.0f}%; best roof_rear reductions (b) {100 * max_rear['b']:.0f}%, (c) {100 * max_rear['c']:.0f}%, (d) {100 * max_rear['d']:.0f}%)"
    lines += [f"**Verdict by the registered rules (primary direction, E2A test): {verdict}.**", "", "Pairs checked for rule 1 (kind, D, roof_rear reduction, roof_front reduction under the matching control, passes):", "",
              "\n".join(f"- ({h[0]}) D = {h[1]} m: roof_rear {100 * h[2]:.0f}%, control {100 * h[3]:.0f}%, {'pass' if h[4] else 'no'}" for h in hits), "",
              "## E2A test, all variants (seed-mean per car; CI of the total bias over 250 cars; regional columns are contributions to the C_L,p bias in counts; reduction = 1 - after/before)", ""]
    from scripts.diagnosis import region_errors  # noqa: E402
    lines += [region_errors.md(rows), ""]
    for fold, role in (("E2B", "test"), ("E2C", "test"), ("E2A", "val")):
        S = R[(R.fold == fold) & (R.role == role)].set_index("variant")
        base = S.loc["none"]
        rr = [{"variant": v, "n": int(S.loc[v, "n"]), "bias C_L,p": f"{S.loc[v, 'dL']:+.1f}", "roof_rear": f"{S.loc[v, 'reg_dL_roof_rear']:+.1f}", "roof_front": f"{S.loc[v, 'reg_dL_roof_front']:+.1f}", "bias C_D,p": f"{S.loc[v, 'dD']:+.1f}", "rel L2": f"{S.loc[v, 'rel_l2']:.4f}"} for v in S.index]
        lines += [f"## {fold} {role} (control; primary direction only; description only, not part of the rules)", "", region_errors.md(rr), ""]
    (OUT / "T1_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(verdict)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if not a.analyze:
        main_run("cuda" if torch.cuda.is_available() else "cpu")
    R, d, A = analyze()
    summarise(R, d, A)
