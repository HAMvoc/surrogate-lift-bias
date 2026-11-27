"""Shift-indicator and axle study: label-free shift indicators (A), axle-wise reference correction (B), axle split and centreline on E2B/E2C (C)
Definitions as scripts/diagnosis/centerline_and_axles.py.
Output: results/shift_and_axles/{shift_cars.csv, shift_indicators.csv, axle_correction.csv, axle_folds.csv, centre_folds.csv, verdict.md, centreline_folds.png},
paper_assets/figures/centreline_folds.pdf. Usage: python scripts/diagnosis/shift_indicators_and_axles.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.diagnosis.pressure_coefficients import CLIP, GEO, pool_weights  # noqa: E402
from scripts.diagnosis.force_alignment import load_pool  # noqa: E402
from scripts.baselines.parametric_surrogates import KS, SETTINGS  # noqa: E402
from scripts.diagnosis.centerline_and_axles import FAMS, L_M, NB, RUNS, axle_front, boot_diff, boot_mean, centre_bins  # noqa: E402
from src.data.dataset import load_split  # noqa: E402

OUT = ROOT / "results" / "shift_and_axles"
CARS = pd.read_csv(ROOT / "results" / "classical_baselines" / "cars.csv").set_index("sample_id")
WF = pd.read_csv(ROOT / "results" / "decomposition" / "T1" / "wheel_fits.csv").groupby(["id", "axle"]).x_c.mean().unstack()


class Geo:
    """Per-car integration weights (counts) and axle geometry, as the centerline and axle study."""

    def __init__(self, sid):
        xyz, nrm, cp, area = load_pool(sid)
        p = xyz.astype(np.float64) * L_M
        w = pool_weights(np.asarray(area, np.float64), sid)
        S = float(GEO.at[sid, "area_total_m2"] / GEO.at[sid, "frontal_area_m2"])
        self.kz, self.kx = -1000 * S * w * nrm[:, 2] / w.sum(), -1000 * S * w * nrm[:, 0] / w.sum()
        self.x, self.y, self.z, self.nz, self.w = p[:, 0], p[:, 1], p[:, 2], nrm[:, 2], w
        self.xr = (self.x - self.x.min()) / (self.x.max() - self.x.min())
        self.xr_ax, self.wb = WF.at[sid, "rear"], WF.at[sid, "rear"] - WF.at[sid, "front"]
        self.ref = np.clip(cp.astype(np.float64), *CLIP)

    def lift(self, cp):
        cl = float((self.kz * cp).sum())
        f = axle_front(self.x, self.z, self.kx * cp, self.kz * cp, self.xr_ax, self.wb)
        return cl, f, cl - f

    def centre(self, cp):
        return centre_bins(self.xr, self.y, self.nz, self.w, cp)


def loader(dirs, role):
    preds = [np.load(Path(d) / f"{role}_pred_fp16.npy", mmap_mode="r") for d in dirs]
    pos = [{i: j for j, i in enumerate(pd.read_csv(Path(d) / f"{role}_per_car.csv").id)} for d in dirs]
    return lambda sid: [np.asarray(p[q[sid]], np.float64) for p, q in zip(preds, pos)]


# ---------------- A ----------------
def part_a():
    sp = load_split("E2A")
    va, te, tr = sp["val"], sp["test"], sp["train"]
    D = np.load(ROOT / "data" / "manifests" / "dist_R1.npz")
    ix = {s: i for i, s in enumerate(D["ids"])}
    full = D["full"]
    tri = [ix[s] for s in tr]
    m2 = {"val": loader([ROOT / "results" / "force_alignment" / "ckpt_preds" / f"E2A_M2_R2_s{s}" for s in range(3)], "val"),
          "test": loader([ROOT / "results" / f"E2A_M2_R2_s{s}" for s in range(3)], "test")}
    tsv = {r: loader([ROOT / "results" / f"E2A_TSV_R1_s{s}" for s in range(3)], r) for r in ("val", "test")}
    rows = []
    for role, ids in (("val", va), ("test", te)):
        for sid in ids:
            g = Geo(sid)
            cm = [g.lift(p)[0] for p in m2[role](sid)]
            ct = [g.lift(p)[0] for p in tsv[role](sid)]
            rows.append({"sample_id": sid, "role": role, "CL": CARS.at[sid, "CL"], "d_sdf": float(full[ix[sid], tri].min()),
                         "e_M2": np.mean(cm) - CARS.at[sid, "CL"], "s_M2": np.std(cm, ddof=1), "e_TSV": np.mean(ct) - CARS.at[sid, "CL"], "s_TSV": np.std(ct, ddof=1)})
    A = pd.DataFrame(rows)
    A.round(6).to_csv(OUT / "shift_cars.csv", index=False)
    T = A[A.role == "test"]
    y = (A.role == "test").astype(int)
    res = []
    for ind, model in (("d_sdf", "M2"), ("d_sdf", "TSV"), ("s_M2", "M2"), ("s_TSV", "TSV")):
        e = T[f"e_{model}"]
        r1, p1 = spearmanr(T[ind], e)
        r2, p2 = spearmanr(T[ind], (e - e.mean()).abs())
        r = {"indicator": ind, "model": model, "auroc": roc_auc_score(y, A[ind]), "bias_test": e.mean(), "mean_ind_test": T[ind].mean(),
             "mean_ind_val": A[A.role == "val"][ind].mean(), "mean_abs_e_val": A[A.role == "val"][f"e_{model}"].abs().mean(),
             "rho_e": r1, "p_e": p1, "rho_dev": r2, "p_dev": p2}
        r["size_ratio"] = r["mean_ind_test"] / abs(r["bias_test"]) if ind.startswith("s_") else np.nan
        r["detects"] = r["auroc"] >= 0.8
        r["sizes"] = bool(r["size_ratio"] >= 0.5) if ind.startswith("s_") else np.nan
        r["ranks"] = (abs(r1) >= 0.3 and p1 < 0.05) or (abs(r2) >= 0.3 and p2 < 0.05)
        res.append(r)
    R = pd.DataFrame(res)
    assert abs(R[(R.model == "M2")].bias_test.iloc[0] - 183.9) <= 0.5  # gate: seed-mean M2 bias on E2A
    R.round(5).to_csv(OUT / "shift_indicators.csv", index=False)
    return R


# ---------------- B ----------------
def part_b():
    ax = pd.read_csv(ROOT / "results" / "centerline_and_axles" / "axle_cars.csv").set_index("sample_id")
    perm = [str(i) for i in np.random.default_rng(0).permutation(sorted(load_split("E2A")["test"]))]
    T = ax.loc[perm]
    stored = pd.read_csv(ROOT / "results" / "reference_correction" / "shift_draws.csv").query("fold == 'E2A' and family == 'M2' and setting == 'orig' and k == 10 and draw == 0").abs_bias_L.iloc[0]
    eT = (T.M2_CL - T.CL).to_numpy()
    idx = np.arange(125)[np.random.default_rng(100).permutation(125)[:10]]
    assert abs(abs(eT[125:].mean() - eT[idx].mean()) - stored) < 0.01, (abs(eT[125:].mean() - eT[idx].mean()), stored)
    rows = []
    for fam in FAMS:
        E = {a: (T[f"{fam}_CL{a}"] - T[f"CL{a}"]).to_numpy() for a in ("f", "r")}
        for st, pool, ev, seed0 in SETTINGS:
            for k in KS:
                b = {a: [] for a in E}
                for d in range(50):
                    lab = pool[np.random.default_rng(seed0 + d).permutation(len(pool))[:k]]
                    for a, e in E.items():
                        b[a].append(abs(e[ev].mean() - e[lab].mean()))
                for a, e in E.items():
                    sig = e[ev].std(ddof=0)
                    rows.append({"family": fam, "setting": st, "k": k, "axle": a, "bias_before": e[ev].mean(), "abs_bias_after": np.mean(b[a]),
                                 "p95_after": np.percentile(b[a], 95), "frac_left": np.mean(b[a]) / abs(e[ev].mean()), "formula": np.sqrt(2 / np.pi) * sig / np.sqrt(k), "sigma": sig})
    B = pd.DataFrame(rows)
    B.round(4).to_csv(OUT / "axle_correction.csv", index=False)
    return B


# ---------------- C ----------------
def part_c():
    t4 = pd.read_csv(ROOT / "results" / "decomposition" / "T4" / "T4_overshoot.csv").query("q == 'C_L,p' and model == 'M2'").set_index("fold")
    rows, prof = [], {}
    for fold in ("E2B", "E2C"):
        sp = load_split(fold)
        tr, te = sp["train"], sp["test"]
        fams = [f for f in FAMS if all((ROOT / "results" / d.format(f=fold)).exists() for d in RUNS[f][:1])]
        get = {f: loader([ROOT / "results" / d.format(f=fold) for d in RUNS[f] if (ROOT / "results" / d.format(f=fold)).exists()], "test") for f in fams}
        cfd = {"train": [], "test": []}
        P = {"train": [], "test": [], "M2": []}
        per = []
        for role, ids in (("train", tr), ("test", te)):
            for sid in ids:
                g = Geo(sid)
                cl, f, r = g.lift(g.ref)
                assert abs(cl - CARS.at[sid, "CL"]) < 1e-3
                cfd[role].append((f, r))
                P[role].append(g.centre(g.ref))
                if role == "test":
                    row = {"sample_id": sid, "CLf": f, "CLr": r}
                    for fam in fams:
                        pr = np.mean(get[fam](sid), 0)
                        _, pf, prr = g.lift(pr)
                        row[f"{fam}_f"], row[f"{fam}_r"] = pf - f, prr - r
                        if fam == "M2":
                            P["M2"].append(g.centre(pr))
                    per.append(row)
        ctr, cte = np.array(cfd["train"]), np.array(cfd["test"])
        G = ctr.mean(0) - cte.mean(0)
        gl, gh = boot_diff(ctr, cte)
        assert abs(G.sum() - t4.at[fold, "G"]) < 0.5, (fold, G.sum(), t4.at[fold, "G"])
        Pc = pd.DataFrame(per)
        assert abs((Pc.M2_f + Pc.M2_r).mean() - t4.at[fold, "bias"]) < 0.5, (fold, (Pc.M2_f + Pc.M2_r).mean(), t4.at[fold, "bias"])
        rows.append({"fold": fold, "model": "gap (CFD)", "B": G.sum(), "B_f": G[0], "B_f_lo": gl[0], "B_f_hi": gh[0], "B_r": G[1], "B_r_lo": gl[1], "B_r_hi": gh[1]})
        for fam in fams:
            e = Pc[[f"{fam}_f", f"{fam}_r"]].to_numpy()
            lo, hi = boot_mean(e)
            b = e.mean(0)
            rows.append({"fold": fold, "model": fam, "B": b.sum(), "B_f": b[0], "B_f_lo": lo[0], "B_f_hi": hi[0], "B_r": b[1], "B_r_lo": lo[1], "B_r_hi": hi[1],
                         "O_f": b[0] - G[0], "O_r": b[1] - G[1]})
        prof[fold] = {k: np.nanmean(np.array(v), 0) for k, v in P.items()}
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "axle_folds.csv", index=False)
    xc = (np.arange(NB) + 0.5) / NB
    cp = {"x": xc}
    for fold, p in prof.items():
        for k, s in enumerate(("upper", "lower")):
            for g in ("train", "test", "M2"):
                cp[f"{fold}_{s}_{g}"] = p[g][k]
    pd.DataFrame(cp).round(5).to_csv(OUT / "centre_folds.csv", index=False)
    fig(pd.DataFrame(cp))
    return C


def fig(P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    names = {"E2B": "notchback held out (E2B)", "E2C": "fastback held out (E2C)"}
    for i, fold in enumerate(("E2B", "E2C")):
        for j, s in enumerate(("upper", "lower")):
            a = ax[i, j]
            a.plot(P.x, P[f"{fold}_{s}_train"], color="0.4", lw=1.1, label="training types (CFD)")
            a.plot(P.x, P[f"{fold}_{s}_test"], color="#D55E00", lw=1.1, label="held-out type (CFD)")  # Okabe-Ito vermillion
            a.plot(P.x, P[f"{fold}_{s}_M2"], color="#0072B2", ls="--", lw=1.1, label="M2 on held-out type")  # Okabe-Ito blue
            a.invert_yaxis()
            a.set(xlabel="$x/L$", ylabel="$C_p$ on the centerline", title=f"{names[fold]}, {s} surface")
            a.title.set_fontsize(8)
    ax[0, 0].legend(frameon=False, fontsize=7)
    p = ROOT / "paper_assets" / "figures"
    f.savefig(p / "centreline_folds.pdf"); f.savefig(OUT / "centreline_folds.png", dpi=150)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R, B, C = part_a(), part_b(), part_c()
    m10 = B[(B.family == "M2") & (B.setting == "orig") & (B.k == 10)].set_index("axle")
    r10 = B[(B.setting == "orig") & (B.k == 10) & (B.axle == "r")]
    qb1 = m10.at["r", "abs_bias_after"] <= 25 and m10.at["f", "abs_bias_after"] <= 15
    qb2 = int((r10.frac_left <= 0.15).sum())
    lines = ["# Shift-indicator and axle study (criteria: `PREREG.md`)", "",
             "Self-checks passed: M2 gate (183.9), stored M2 reference correction reproduced, CFD integration, G and M2 bias of E2B/E2C against T4_overshoot.csv.", ""]
    for _, r in R.iterrows():
        lines.append(f"**A {r.indicator}/{r.model}:** AUROC {r.auroc:.2f} (detects: {r.detects}); mean test {r.mean_ind_test:.3f} vs val {r.mean_ind_val:.3f}; size ratio {r.size_ratio:.3f} (sizes: {r.sizes}); "
                     f"rho(e) {r.rho_e:+.2f} (p {r.p_e:.3f}), rho(|dev|) {r.rho_dev:+.2f} (p {r.p_dev:.3f}) (ranks: {r.ranks}).")
    lines += ["", f"**Q-B1:** M2 k = 10 orig: rear |bias| {m10.at['r', 'abs_bias_after']:.1f}, front {m10.at['f', 'abs_bias_after']:.1f} -> {'met' if qb1 else 'not met'}. "
              f"**Q-B2:** families with rear fraction left <= 0.15 at k = 10: {qb2} of 7 -> {'met' if qb2 >= 5 else 'not met'}.", ""]
    for fold in ("E2B", "E2C"):
        c = C[C.fold == fold]
        g = c[c.model == "gap (CFD)"].iloc[0]
        m = c[c.model == "M2"].iloc[0]
        n = int((c.O_r > c.O_f).sum())
        nf = int(c.O_r.notna().sum())
        lines.append(f"**C {fold}:** G_f {g.B_f:+.1f}, G_r {g.B_r:+.1f} (Q-C2 |G_r| >= |G_f|: {abs(g.B_r) >= abs(g.B_f)}); M2 O_f {m.O_f:+.1f}, O_r {m.O_r:+.1f}; O_r > O_f for {n} of {nf} families "
                     f"(Q-C1: {'met' if (m.O_r > m.O_f and n >= min(5, nf)) else 'not met'}).")
    lines += ["", "## A", "", md(R.round(3).to_dict("records")), "", "## B (k = 10)", "", md(B[B.k == 10].round(2).to_dict("records")), "", "## C", "", md(C.round(2).to_dict("records")), ""]
    (OUT / "verdict.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
