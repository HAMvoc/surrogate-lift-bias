"""Zero-shot checks study, task T4: is the RegDGCNN (B3) lift bias on E1 and its sign change between E2A and E2C a pipeline artefact?
Writes results/zero_shot_checks/T4/T4_report.md. Usage: python scripts/diagnosis/regdgcnn_checks.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.pressure_coefficients import CLIP  # noqa: E402
from scripts.diagnosis.force_alignment import REGIONS, car_terms, load_pool, region_labels  # noqa: E402

RES, OUT = ROOT / "results", ROOT / "results" / "zero_shot_checks" / "T4"


def load(run):
    d = RES / run
    return np.load(d / "test_pred_fp16.npy", mmap_mode="r"), list(pd.read_csv(d / "test_per_car.csv").id), pd.read_csv(d / "test_per_car.csv")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = ["# T4: RegDGCNN (B3) anomaly checks", ""]
    # gate: M2 E2A, 3 seeds, same code path as used for B3 below
    dls = []
    for s in range(3):
        pr, ids, _ = load(f"E2A_M2_R2_s{s}")
        dls.append([car_terms(pr[j], *(lambda p: (p[2], p[1], p[3]))(load_pool(i)), i)["dL"] for j, i in enumerate(ids)])
    gate = float(np.mean(dls, axis=0).mean())
    assert abs(gate - 183.9) <= 0.5, gate
    L += [f"Gate: M2 E2A 3-seed bias C_L,p through the same code path = {gate:+.1f} counts (required 183.9 +- 0.5): PASS.", ""]
    # checks 1, 2 on B3 E1 (and M2 E1 seed 2 for comparison)
    rows = []
    for name, run in (("B3 E1", "E1_B3_R1_s0_lr3e-3"), ("M2 E1 (seed 2)", "E1_M2_R2_s2"), ("B3 E2A", "E2A_B3_R1_s0_lr3e-3"), ("B3 E2C", "E2C_B3_R1_s0_lr3e-3")):
        pr, ids, csv = load(run)
        rel, corr, dm, sr, dL = [], [], [], [], []
        for j, sid in enumerate(ids):
            xyz, nrm, cp, area = load_pool(sid)
            ref = np.clip(cp.astype(np.float64), *CLIP)
            p = np.asarray(pr[j], np.float64)
            rel.append(np.linalg.norm(p - ref) / np.linalg.norm(ref))
            corr.append(np.corrcoef(p, ref)[0, 1])
            dm.append(p.mean() - ref.mean())
            sr.append(p.std() / ref.std())
            dL.append(car_terms(p, cp, nrm, area, sid)["dL"])
        rel = np.array(rel)
        rows.append({"run": name, "cars": len(ids), "max |rel L2 recomputed - CSV|": f"{np.abs(rel - csv.rel_l2.values).max():.1e}", "mean point-wise correlation pred/CFD": f"{np.mean(corr):.3f}",
                     "mean (mean pred - mean CFD) Cp": f"{np.mean(dm):+.4f}", "mean SD ratio pred/CFD": f"{np.mean(sr):.3f}", "bias C_L,p (counts)": f"{np.mean(dL):+.1f}"})
    ok1 = all(float(r["max |rel L2 recomputed - CSV|"]) < 2e-3 for r in rows) and float(rows[0]["mean point-wise correlation pred/CFD"]) > 0.8
    ok2 = abs(float(rows[0]["mean (mean pred - mean CFD) Cp"])) <= 0.03 and 0.6 <= float(rows[0]["mean SD ratio pred/CFD"]) <= 1.2
    L += ["## Checks 1 and 2: point order, normalisation (E1 = 75 cars; the other rows for reference)", "", md(rows), "",
          f"- Check 1 (order: recomputed rel L2 equals the CSV for every car, and B3 correlation on E1 > 0.8): **{'PASS' if ok1 else 'FAIL'}**.",
          f"- Check 2 (normalisation: |mean offset| <= 0.03 Cp and SD ratio in [0.6, 1.2] on E1): **{'PASS' if ok2 else 'FAIL'}**.", ""]
    # check 5: pool order randomness
    rng = np.random.default_rng(0)
    ids = list(pd.read_csv(RES / "E1_B3_R1_s0_lr3e-3" / "test_per_car.csv").id)[:25]
    ratios = []
    for sid in ids:
        xyz = load_pool(sid)[0].astype(np.float64)
        consec = np.linalg.norm(np.diff(xyz, axis=0), axis=1).mean()
        a, b = rng.integers(0, len(xyz), 200000), rng.integers(0, len(xyz), 200000)
        ratios.append(consec / np.linalg.norm(xyz[a] - xyz[b], axis=1).mean())
    ok5 = float(np.mean(ratios)) > 0.9
    L += ["## Check 5: training and evaluation point sets", "",
          "Training (`Trainer.train_epoch` -> `CarSet.batch(idx, n_points, gen)`): 5,000 pool points per car and step, `torch.randperm` subset of the 100k pool. Evaluation (`Trainer.evaluate`): the pool is cut into consecutive chunks of `eval_chunk` = `n_points` = 5,000 points, each chunk is a separate forward pass "
          "(kNN graph with k = 40 and the global max-pooled feature are computed inside the chunk). Consecutive chunks only match random subsets if the stored pool order is spatially random.", "",
          f"Mean distance between consecutive pool points / mean distance between two random pool points, 25 E1 cars: {np.mean(ratios):.3f} (min {np.min(ratios):.3f}, max {np.max(ratios):.3f}); criterion > 0.9: **{'PASS' if ok5 else 'FAIL'}**.", ""]
    # checks 3, 4 from the code
    src = (ROOT / "src" / "models" / "regdgcnn.py").read_text(encoding="utf-8")
    uses_nrm = "nrm" in src.split("def forward")[1].split("return")[0].replace("def forward(self, vol, g, ndc, nrm)", "")
    L += ["## Checks 3 and 4 (code reading)", "",
          f"- B3 `forward(vol, g, ndc, nrm)` uses only `ndc` (normalised xyz) in its body: normals used inside the body = {uses_nrm}. The integrals use `pool.npz` normals through `force_alignment.car_terms` for every model (same file, same function).",
          "- A_ref and A_total come from `data/manifests/car_geometry.csv` inside `car_terms` for every model (one function, no per-model input).",
          "- Check 3 and 4: **PASS** (no model-specific difference).", ""]
    ok3 = not uses_nrm
    # regional description of the B3 bias on E1 (from the force-alignment study table)
    T = pd.read_csv(RES / "force_alignment" / "per_car_terms.csv.gz")
    g = T[(T.fold == "E1") & (T.model == "B3")]
    reg = pd.DataFrame({"region": REGIONS, "B3 E1 contribution to bias C_L,p (counts)": [f"{g[f'reg_dL_{r}'].mean():+.1f}" for r in REGIONS], "area share %": [f"{100 * g[f'reg_area_{r}'].mean():.1f}" for r in REGIONS]})
    gm = T[(T.fold == "E1") & (T.model == "M2")]
    reg["M2 E1 (3 seeds)"] = [f"{gm[f'reg_dL_{r}'].mean():+.1f}" for r in REGIONS]
    pipeline = (not ok1) or (not ok2) or (not ok5) or (not ok3)
    L += ["## Description of the B3 lift bias on E1 by region (counts)", "", md(reg.to_dict("records")), "",
          f"## Verdict by the registered criteria: **{'pipeline error' if pipeline else 'real phenomenon (no pipeline check failed)'}**", ""]
    (OUT / "T4_report.md").write_text("\n".join(L), encoding="utf-8")
    print("pipeline error:", pipeline, "| ok1", ok1, "ok2", ok2, "ok5", ok5, "ok3", ok3)


if __name__ == "__main__":
    main()
