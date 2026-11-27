"""Result tables: E1, E2 (three folds), confirmatory M2 vs B1 comparison (Wilcoxon + Holm), D_X, S3/S4,
sensitivity to dropping near-duplicate cars, raw labels. Only reads results/<split>_<model>_<res>_s<seed>/test_per_car.csv; does not recompute models.
Only the seeds present are used; if seeds {0,1,2} are missing the report is marked PRELIMINARY and must not be used for conclusions.
Usage: python scripts/training/evaluate.py [--results results] [--out reports/results.md]"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import bootstrap_ci, degradation_D, holm, paired_compare, pooled_r2  # noqa: E402

FOLDS = {"E2A": "Estateback", "E2B": "Notchback", "E2C": "Fastback"}
CAT = {"F": "Fastback", "N": "Notchback", "E": "Estateback"}
FULL_SEEDS = {0, 1, 2}
D_LIMIT = 1.5  # prereg S4


def s0(runs):
    """Per-car frame of seed 0 only (matched-seed baseline for single-seed ablations)."""
    return combine({0: runs[0]})[0] if 0 in runs else None


def cat_of(i):
    return CAT[i[0]]


def load_runs(root, split, model, res, tag=""):
    """{seed: DataFrame(index=id)} of every run <split>_<model>_<res>_s<seed>[_<tag>] that has test_per_car.csv (exact name match: tagged runs are never mistaken for plain seeds)."""
    pat = re.compile(rf"{re.escape(split)}_{re.escape(model)}_{re.escape(res)}_s(\d+)" + (rf"_{re.escape(tag)}" if tag else "") + "$")
    out = {}
    for d in sorted(Path(root).iterdir()):
        m = pat.fullmatch(d.name)
        if m and (d / "test_per_car.csv").exists():
            out[int(m.group(1))] = pd.read_csv(d / "test_per_car.csv").set_index("id")
    return out


def combine(runs):
    """Mean over seeds per car (prereg); pooled R² is computed per seed then averaged."""
    dfs = list(runs.values())
    ids = dfs[0].index
    assert all(d.index.equals(ids) for d in dfs), "seeds must have the same cars in the same order"
    cols = [c for c in ("rel_l2", "raw_rel_l2", "mae", "rmse") if c in dfs[0]]
    per = pd.DataFrame({c: np.mean([d[c].to_numpy() for d in dfs], axis=0) for c in cols}, index=ids)
    return per, float(np.mean([pooled_r2(d) for d in dfs]))


def stat_row(name, per, r2, seeds=""):
    m, lo, hi = bootstrap_ci(per.rel_l2)
    return {"model": name, "seed": seeds, "n cars": len(per), "rel L2 [CI 95%]": f"{m:.4f} [{lo:.4f}; {hi:.4f}]", "median": f"{per.rel_l2.median():.4f}",
            "MAE": f"{per.mae.mean():.4f}", "RMSE": f"{per.rmse.mean():.4f}", "R² pooled": f"{r2:.3f}", "rel L2 raw labels": f"{per.raw_rel_l2.mean():.4f}"}


def md(rows):
    if not rows:
        return "_(no data)_\n"
    cols = list(rows[0])
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]) + "\n"


def paired_row(label, a, b, other="B1"):
    ids = a.index.intersection(b.index)
    d, lo, hi, p = paired_compare(a.loc[ids].rel_l2.to_numpy(), b.loc[ids].rel_l2.to_numpy())
    return {"comparison": label, "n cars": len(ids), f"mean diff (M2 − {other})": f"{d:+.4f}", "95% CI of diff": f"[{lo:+.4f}; {hi:+.4f}]", "Wilcoxon p": f"{p:.2e}"}, d, p, hi


def leak_ids(split_dir, fold):
    f = Path(split_dir) / f"{fold}.json"
    if not f.exists():
        return set()
    fl = json.loads(f.read_text())["leak_flags"]
    return set(fl["near_twin"]) | set(fl["twin_in_train"])


def b2_section(results, m2_by_split, model="B2", label="B2 (PointNet++ lite)", tag="", head="## B2 (exploratory, RQ3): point-based baseline", note=None):
    """Exploratory (RQ3): a baseline (default B2, point-based) per split and paired M2 - baseline on the same cars. B2 locked at lr 3e-3, 150 epochs."""
    b2 = {sp: load_runs(results, sp, model, "R1", tag) for sp in ("E1", *FOLDS)}
    if not any(b2.values()):
        return []
    rows, pairs = [], []
    e1b2 = combine(b2["E1"])[0] if b2["E1"] else None
    for sp, runs in b2.items():
        if not runs:
            continue
        per, r2 = combine(runs)
        row = {"split": sp, **stat_row(label, per, r2, ",".join(map(str, runs))), "D [CI 95%]": "-"}
        if sp in FOLDS and e1b2 is not None:  # same definition as M2's D, with the baseline's own E1 error as the base
            base = e1b2[[cat_of(i) == FOLDS[sp] for i in e1b2.index]].rel_l2
            D = degradation_D(per.rel_l2.to_numpy(), base.to_numpy())
            row["D [CI 95%]"] = f"{D[0]:.3f} [{D[1]:.3f}; {D[2]:.3f}]"
        rows.append(row)
        if m2_by_split.get(sp) is not None:
            pairs.append(paired_row(sp, m2_by_split[sp], per, model)[0])
    return [head, "",
            note or "B2 is not converged at 150 epochs and its lr grid has two values, so these numbers are an upper bound on B2 error.", "",
            md(rows), f"### Paired M2 vs {model} (same cars, exploratory; no multiplicity correction)", "", md(pairs)]


ABLATIONS = (("E4: occupancy input (M2-occ, R2)", "M2OCC", "R2"), ("E3: grid R1 (M2)", "M2", "R1"), ("E3: grid R3 (M2)", "M2", "R3"),
             ("E-Q: no query coordinates (M2-noXYZ, R2)", "M2NOXYZ", "R2"), ("E-Q: no normals (M2-noN, R2)", "M2NON", "R2"))


def ablation_section(results, base_s0):
    """Exploratory (RQ4, RQ5): single-seed ablations vs M2 R2 seed 0 on the same cars. base_s0: {split: per-car frame of M2 R2 seed 0}."""
    rows, pairs = [], []
    for label, model, res in ABLATIONS:
        e1 = load_runs(results, "E1", model, res)
        e1_per = combine(e1)[0] if e1 else None
        for sp in ("E1", *FOLDS):
            runs = load_runs(results, sp, model, res)
            if not runs or base_s0.get(sp) is None:
                continue
            per, r2 = combine(runs)
            row = {"split": sp, **stat_row(label, per, r2, ",".join(map(str, runs))), "D [CI 95%]": "-"}
            if sp in FOLDS and e1_per is not None:  # D with the ablation's own E1 error as the base
                base = e1_per[[cat_of(i) == FOLDS[sp] for i in e1_per.index]].rel_l2
                D = degradation_D(per.rel_l2.to_numpy(), base.to_numpy())
                row["D [CI 95%]"] = f"{D[0]:.3f} [{D[1]:.3f}; {D[2]:.3f}]"
            rows.append(row)
            pairs.append(paired_row(f"{label} / {sp}", base_s0[sp], per, "ablation")[0])
    if not rows:
        return []
    return ["## Ablations (exploratory, RQ4/RQ5): single seed", "",
            "Compared with M2 R2 seed 0 on the same cars. Seed-to-seed spread of M2 R2 is about 0.002-0.006 rel L2, so smaller differences are not interpretable. "
            "No multiplicity correction.", "", md(rows), "### Paired: M2 R2 seed 0 vs ablation (negative mean diff: M2 R2 better)", "", md(pairs)]


B2_VARIANTS = (("B2 lr 1e-2, 150 epochs", "lr1e-2"), ("B2 lr 3e-3, 300 epochs", "ep300"))


def b2_variants_section(results, m2_by_split):
    """Exploratory: is B2 under-trained? Variants of the locked B2 (lr 3e-3, 150 epochs) run after the lock, seed 0, same protocol otherwise."""
    rows, pairs = [], []
    locked = load_runs(results, "E1", "B2", "R1")
    for label, tag in B2_VARIANTS:
        e1v = load_runs(results, "E1", "B2", "R1", tag)
        e1_per = combine(e1v)[0] if e1v else None
        for sp in ("E1", *FOLDS):
            runs = load_runs(results, sp, "B2", "R1", tag)
            if not runs:
                continue
            per, r2 = combine(runs)
            row = {"split": sp, **stat_row(label, per, r2, ",".join(map(str, runs))), "D [CI 95%]": "-"}
            if sp in FOLDS and e1_per is not None:
                base = e1_per[[cat_of(i) == FOLDS[sp] for i in e1_per.index]].rel_l2
                D = degradation_D(per.rel_l2.to_numpy(), base.to_numpy())
                row["D [CI 95%]"] = f"{D[0]:.3f} [{D[1]:.3f}; {D[2]:.3f}]"
            rows.append(row)
            if m2_by_split.get(sp) is not None:
                pairs.append(paired_row(f"{label} / {sp}", m2_by_split[sp], per, "B2 variant")[0])
    if not rows:
        return []
    if locked:
        rows.insert(0, {"split": "E1", **stat_row("B2 locked (lr 3e-3, 150 epochs)", *combine(locked), ",".join(map(str, locked))), "D [CI 95%]": "-"})
    return ["## B2 sensitivity (exploratory): is B2 under-trained?", "",
            "Variants run after B2 was locked (lr 3e-3, 150 epochs). Single seed; the locked B2 row is repeated for reference.", "", md(rows),
            "### Paired: M2 vs B2 variant (negative mean diff: M2 better)", "", md(pairs)]


def build(results=ROOT / "results", split_dir=ROOT / "data" / "manifests" / "splits"):
    L = ["# Results (auto-generated by `scripts/training/evaluate.py`; do not edit by hand)", "",
         "Primary metric: per-car rel L2 on clipped Cp [-10, 1] (100k-point pool), mean over cars, per-car bootstrap CI (B = 2000). "
         "Evaluation tables following the pre-registered metrics.", ""]
    e1 = {m: load_runs(results, "E1", m, r) for m, r in (("M2", "R2"), ("B1", "R1"), ("B0", "NA"))}
    L += ["## E1 (interpolation, test 75 cars)", ""]
    rows, e1_m2 = [], None
    if e1["M2"]:
        e1_m2, r2 = combine(e1["M2"])
        rows.append(stat_row("M2 R2 (mean of seeds)", e1_m2, r2, ",".join(map(str, e1["M2"]))))
        rows += [stat_row(f"M2 R2 seed {s}", *combine({s: d}), str(s)) for s, d in e1["M2"].items() if len(e1["M2"]) > 1]
    for m, label in (("B1", "B1 (k=10)"), ("B0", "B0")):
        if e1[m]:
            rows.append(stat_row(label, *combine(e1[m]), "0"))
    L += [md(rows)]
    if e1_m2 is not None and e1["B1"]:
        b1_e1 = combine(e1["B1"])[0]
        r, d, p, hi = paired_row("S3: M2 vs B1 on E1", e1_m2, b1_e1)
        L += ["**S3** (median M2 < B1 and the CI of the difference excludes 0): " + ("**PASS**" if e1_m2.rel_l2.median() < b1_e1.rel_l2.median() and hi < 0 else "**FAIL**"), "", md([r])]

    folds = {f: {m: load_runs(results, f, m, r) for m, r in (("M2", "R2"), ("B1", "R1"), ("B0", "NA"))} for f in FOLDS}
    if not any(v["M2"] for v in folds.values()):
        return "\n".join(L + ["## E2", "", "_No M2 results on E2 yet._", ""] + b2_section(results, {"E1": e1_m2}) + b2_variants_section(results, {"E1": e1_m2}) + ablation_section(results, {"E1": s0(e1["M2"])}))
    seeds_present = {f: set(v["M2"]) for f, v in folds.items() if v["M2"]}
    prelim = not all(s >= FULL_SEEDS for s in seeds_present.values()) or len(seeds_present) < 3
    L += ["## E2 (one body style held out, test 250 cars per fold)", ""]
    if prelim:
        L += [f"> **PRELIMINARY — not for drawing conclusions.** Seeds present: { {f: sorted(s) for f, s in seeds_present.items()} }; prereg requires the mean of three seeds {sorted(FULL_SEEDS)} on all three folds.", ""]
    tab, conf, Ds, per_fold = [], [], {}, {}
    for f, held in FOLDS.items():
        v = folds[f]
        if not v["M2"]:
            continue
        m2, r2 = combine(v["M2"])
        b1 = combine(v["B1"])[0] if v["B1"] else None
        per_fold[f] = (m2, b1)
        base = e1_m2[[cat_of(i) == held for i in e1_m2.index]].rel_l2 if e1_m2 is not None else None
        D = degradation_D(m2.rel_l2.to_numpy(), base.to_numpy()) if base is not None and len(base) else None
        if D:
            Ds[f] = D
        r = stat_row(f"M2 R2 held out {held}", m2, r2, ",".join(map(str, v["M2"])))
        r["D [CI 95%]"] = f"{D[0]:.3f} [{D[1]:.3f}; {D[2]:.3f}]" if D else "-"
        tab.append({"fold": f, **r})
        for m, label in (("B1", "B1"), ("B0", "B0")):
            if v[m]:
                tab.append({"fold": f, **stat_row(label, *combine(v[m]), "0"), "D [CI 95%]": "-"})
        if b1 is not None:
            row, d, p, _ = paired_row(f"{f} ({held})", m2, b1)
            conf.append((row, d, p))
    L += [md(tab)]
    if conf:
        adj = holm([c[2] for c in conf])
        L += ["### Confirmatory comparison: M2 vs B1 (two-sided Wilcoxon, Holm over 3 hypotheses)", ""]
        rows = [{**c[0], "p Holm": f"{a:.2e}", "win": "yes" if a < 0.05 and c[1] < 0 else "no"} for c, a in zip(conf, adj)]
        L += [md(rows)]
        wins = len(conf) == 3 and all(a < 0.05 and c[1] < 0 for c, a in zip(conf, adj))
        if len(Ds) == 3:
            overall = float(np.mean([Ds[f][0] for f in FOLDS]))
            ok = wins and overall < D_LIMIT
            L += [f"**S4**: beats B1 on all three folds (Holm): {'yes' if wins else 'no'}; Overall D = {overall:.3f} (threshold < {D_LIMIT}). "
                  f"Conclusion: **{'PASS' if ok else 'FAIL'}**" + (" (PRELIMINARY)" if prelim else "") + ".", ""]
            if not ok:
                L += ["Do not use the word \"generalizes\" (spec 3.3).", ""]
    # sensitivity: drop cars with a near-duplicate train car or the same group_id (exploratory)
    sens = []
    for f, (m2, b1) in per_fold.items():
        drop = leak_ids(split_dir, f) & set(m2.index)
        keep = m2.index.difference(drop)
        if b1 is not None and len(keep):
            sens.append({"fold": f, "cars dropped": len(drop), "cars kept": len(keep), "M2 (kept)": f"{m2.loc[keep].rel_l2.mean():.4f}",
                         "B1 (kept)": f"{b1.loc[keep].rel_l2.mean():.4f}", "M2 (dropped)": f"{m2.loc[list(drop)].rel_l2.mean():.4f}" if drop else "-"})
    if sens:
        L += ["### Sensitivity (exploratory): drop test cars nearly identical to train cars (`leak_flags`)", "", md(sens)]
    ep = {m: load_runs(results, "E2param", m, r) for m, r in (("M2", "R2"), ("B1", "R1"), ("B0", "NA"))}
    if ep["M2"]:
        rows = [stat_row("M2 R2", *combine(ep["M2"]), ",".join(map(str, ep["M2"])))] + [stat_row(l, *combine(ep[m]), "0") for m, l in (("B1", "B1"), ("B0", "B0")) if ep[m]]
        L += ["## E2param (exploratory): parameter extrapolation within each body style", "", md(rows)]
    m2_all = {"E1": e1_m2, **{f: pair[0] for f, pair in per_fold.items()}}
    return "\n".join(L + b2_section(results, m2_all) + b2_section(results, m2_all, "B3", "B3 (RegDGCNN, 5,000 points)", "lr3e-3", "## B3 (exploratory): RegDGCNN, the DrivAerNet++ authors' baseline", "Single seed, lr 3e-3 chosen on a 25-car E1 validation subset. Deviations from the authors: 5,000 points per car instead of 10,000, sparse validation (every 10 epochs), our AdamW + cosine + EMA protocol. Not a reproduction of their reported numbers.") + b2_variants_section(results, m2_all)
                     + ablation_section(results, {"E1": s0(e1["M2"]), **{f: s0(v["M2"]) for f, v in folds.items()}}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "results.md"))
    a = ap.parse_args()
    text = build(Path(a.results))
    Path(a.out).write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console is cp1252 and cannot print the minus sign in the tables
    print(text)
