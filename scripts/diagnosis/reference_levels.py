"""Decomposition and first corrections study, task T3: reference levels for the between-car RMSE of the lift. (1) 56.5 vs 70.5: MAE vs RMSE of the same errors; (2) in-distribution reference: 25 Estateback cars of E1 (M2, 3 seeds);
(3) label noise of C_L,p from the 100k-point pool (20 random half/half splits per car). Output: results/decomposition/T3/.
Usage: python scripts/diagnosis/reference_levels.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.force_alignment_terms import ci, fm  # noqa: E402
from scripts.diagnosis.pressure_coefficients import CLIP, GEO, pool_weights  # noqa: E402
from scripts.diagnosis.force_alignment import load_pool  # noqa: E402

OUT = ROOT / "results" / "decomposition" / "T3"
B = 2000


def spread(e):
    e = np.asarray(e, float)
    return {"bias": e.mean(), "rmse": float(np.sqrt(((e - e.mean()) ** 2).mean())), "mae": float(np.abs(e - e.mean()).mean()), "sd_ddof1": float(e.std(ddof=1))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(ROOT / "results" / "force_alignment" / "per_car_terms.csv.gz")
    cfd = pd.read_csv(ROOT / "results" / "force_alignment" / "cfd_coefficients.csv").set_index("id")
    L = ["# Decomposition and first corrections study, task T3: reference levels for the lift RMSE", ""]
    # (1) E2A, 56 vs 70.5
    e2a = T[(T.fold == "E2A") & (T.model == "M2")]
    seedmean = e2a.groupby("id").dL.mean()
    assert abs(seedmean.mean() - 183.9) <= 0.5
    s = spread(seedmean.values)
    per_seed = [spread(e2a[e2a.seed == k].dL.values) for k in range(3)]
    L += ["## 1. Where 56 and 70.5 come from (M2, E2A, 250 cars)", "",
          f"Gate: 3-seed mean bias = {s['bias']:+.1f} counts: PASS. One definition from now on: **RMSE after bias removal** = sqrt(mean((e - mean e)^2)) of the seed-mean per-car error e (ddof = 0); **MAE after bias removal** = mean |e - mean e| of the same e.", "",
          md([{"quantity": "seed-mean error (the convention of the zero-shot checks study)", "RMSE after bias removal": f"{s['rmse']:.1f}", "SD with ddof = 1": f"{s['sd_ddof1']:.1f}", "MAE after bias removal": f"{s['mae']:.1f}", "MAE / RMSE": f"{s['mae'] / s['rmse']:.3f}"}] +
                 [{"quantity": f"seed {k} alone", "RMSE after bias removal": f"{p['rmse']:.1f}", "SD with ddof = 1": f"{p['sd_ddof1']:.1f}", "MAE after bias removal": f"{p['mae']:.1f}", "MAE / RMSE": f"{p['mae'] / p['rmse']:.3f}"} for k, p in enumerate(per_seed)]), "",
          f"Registered hypothesis (56.5 is the MAE and 70.5 the RMSE of the same seed-mean errors, MAE/RMSE about 0.8): MAE {s['mae']:.1f} and RMSE {s['rmse']:.1f}, ratio {s['mae'] / s['rmse']:.3f} -> **{'TRUE' if abs(s['mae'] - 56.5) < 0.6 and abs(s['rmse'] - 70.5) < 0.6 else 'FALSE'}**. "
          "The 56.5 is the 'C_L,p MAE after bias correction' of M2 in `reports/paper_assets/tables/T4_integral_audit.csv` (and the 'about 56 counts' of the investigation note); the 70.5 / 70.6 is the standard deviation of the same errors (Zero-shot checks study T3 and the force-alignment study T5). Seed averaging was applied before both. No weighting difference.", ""]
    # (2) E1 Estateback reference
    e1 = T[(T.fold == "E1") & (T.model == "M2") & (T.id.str[0] == "E")]
    assert set(e1.seed) == {0, 1, 2} and e1.id.nunique() == 25
    sm = e1.groupby("id").dL.mean()
    ref = cfd.loc[sm.index, "CL"].values
    pred = ref + sm.values
    sp = spread(sm.values)
    slope = np.polyfit(ref, pred, 1)[0]
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(B):
        i = rng.integers(0, 25, 25)
        bs.append((spread(sm.values[i])["rmse"], spearmanr(pred[i], ref[i])[0], np.polyfit(ref[i], pred[i], 1)[0]))
    bs = np.array(bs)
    pc = lambda j: f"[{np.percentile(bs[:, j], 2.5):.2f}; {np.percentile(bs[:, j], 97.5):.2f}]"
    e2 = spread(seedmean.values)
    cfd_e2a = cfd.loc[seedmean.index, "CL"].values
    sl2 = np.polyfit(cfd_e2a, cfd_e2a + seedmean.values, 1)[0]
    sp2 = spearmanr(cfd_e2a + seedmean.values, cfd_e2a)[0]
    L += ["## 2. In-distribution reference: 25 Estateback cars of E1 test (M2, seeds 0, 1, 2; the cars are in the training distribution)", "",
          md([{"set": "E1 Estateback (25 cars)", "bias C_L,p": f"{sp['bias']:+.1f}", "RMSE after bias removal [CI]": f"{sp['rmse']:.1f} [{np.percentile(bs[:, 0], 2.5):.1f}; {np.percentile(bs[:, 0], 97.5):.1f}]", "MAE after bias removal": f"{sp['mae']:.1f}",
                "Spearman [CI]": f"{spearmanr(pred, ref)[0]:.2f} {pc(1)}", "slope of predicted on CFD [CI]": f"{slope:.2f} {pc(2)}"},
               {"set": "E2A (250 cars, held-out style)", "bias C_L,p": f"{e2['bias']:+.1f}", "RMSE after bias removal [CI]": f"{e2['rmse']:.1f}", "MAE after bias removal": f"{e2['mae']:.1f}", "Spearman [CI]": f"{sp2:.2f}", "slope of predicted on CFD [CI]": f"{sl2:.2f}"}]), "",
          f"CFD C_L,p spread of the 25 E1 Estateback cars (SD {np.std(ref, ddof=1):.1f} counts) and of the 250 E2A cars (SD {np.std(cfd_e2a, ddof=1):.1f}).", ""]
    # (3) label noise
    ids = list(pd.read_csv(ROOT / "results" / "E2A_M2_R2_s0" / "test_per_car.csv").id)
    rng = np.random.default_rng(2026)
    noise = []
    for sid in ids:
        xyz, nrm, cp, area = load_pool(sid)
        r = np.clip(cp.astype(np.float64), *CLIP)
        w = pool_weights(np.asarray(area, np.float64), sid)
        S = float(GEO.at[sid, "area_total_m2"] / GEO.at[sid, "frontal_area_m2"])
        wr = w * r * nrm[:, 2].astype(np.float64)
        d = []
        for _ in range(20):
            perm = rng.permutation(len(r))
            a, b = perm[: len(r) // 2], perm[len(r) // 2:]
            ca = -1000 * S * wr[a].sum() / w[a].sum()
            cb = -1000 * S * wr[b].sum() / w[b].sum()
            d.append(ca - cb)
        noise.append(np.std(d, ddof=1) / np.sqrt(2))
    noise = np.array(noise)
    half = float(np.sqrt((noise ** 2).mean()))
    full = half / np.sqrt(2)
    mae56 = e2["mae"]
    L += ["## 3. Noise of the CFD C_L,p from the 100k-point pool (E2A test cars)", "",
          "Per car: 20 random splits of the 100k points into two disjoint halves, C_L,p of each half with the same IPW weights; half-pool noise = SD of the half difference / sqrt(2). The variance of the estimate scales like 1 / (number of points), so the full-pool noise is the half-pool noise / sqrt(2).", "",
          md([{"quantity": "half-pool noise, RMS over 250 cars", "counts": f"{half:.1f}"}, {"quantity": "half-pool noise, median / 5% / 95% over cars", "counts": f"{np.median(noise):.1f} / {np.percentile(noise, 5):.1f} / {np.percentile(noise, 95):.1f}"},
               {"quantity": "full-pool noise (half / sqrt 2), RMS", "counts": f"{full:.1f}"}]), "",
          "For comparison: the pool estimator against the exact mesh integral had a mean error of +1.2 counts and a maximum per-car error of 21 counts on 45 cars (`reports/lift_pool_validation.md`); this split-half number measures sampling noise of the pool only (the same 100k points are used for CFD and model).", ""]
    # conclusion
    e1r = sp["rmse"]
    realistic = e1r <= 50
    near_noise = full >= 25
    L += ["## Conclusion on the threshold 'RMSE after bias removal <= 50 counts'", "",
          f"Rule (PREREG): realistic if the E1 in-distribution reference is <= 50, unrealistic if it is above 50; noted as near the noise floor if the full-pool label noise is >= 25. E1 reference {e1r:.1f} counts [{np.percentile(bs[:, 0], 2.5):.1f}; {np.percentile(bs[:, 0], 97.5):.1f}] -> **the threshold is {'realistic' if realistic else 'NOT realistic (below the in-distribution level)'}**; "
          f"full-pool label noise {full:.1f} counts -> {'near the noise floor' if near_noise else 'well below the threshold, not the limiting factor'}. The E2A value is {e2['rmse']:.1f} ({100 * (e2['rmse'] / e1r - 1):+.0f}% against E1; the E1 CI is wide with 25 cars).", ""]
    (OUT / "T3_reference.md").write_text("\n".join(L), encoding="utf-8")
    pd.DataFrame({"id": ids, "half_pool_noise": noise}).round(3).to_csv(OUT / "T3_label_noise.csv", index=False)
    print("E1 rmse", e1r, "e2a rmse", e2["rmse"], "mae", e2["mae"], "noise half/full", half, full)


if __name__ == "__main__":
    main()
