"""Safety-margin and cumulative-lift study A: sequential label-budget procedure with a safety margin; selection on partitions of the 250 original estateback cars, test on the 250 new cars.
Errors: seed-mean predicted minus CFD C_L,p (counts) from scripts/correction/swapped_halves.py:stats (selection) and scripts/correction/reference_vs_stage_b.py:indep_stats (test).
Output: results/margin_and_cumulative_lift/{margin_select.csv, margin_ceiling.csv, margin_test.csv, margin_verdict.md}. Usage: python scripts/correction/sequential_margin.py"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT))
from scripts.diagnosis.region_errors import md  # noqa: E402
from scripts.correction.swapped_halves import PAIRS, stats  # noqa: E402
from scripts.correction.reference_vs_stage_b import indep_stats  # noqa: E402

OUT = ROOT / "results" / "margin_and_cumulative_lift"
TAUS, P, Z, KMAX = (10, 20, 28), 500, 1.645, 125
VARIANTS = list(product((5, 10), ("point", "chi2"), (1.0, 1.15, 1.3)))  # (k0, s, c); order = fewer adjustments first within each k0
PLAIN = (5, "point", 1.0)


def kstar(sh, k0, s, c, tau):
    sig = sh * np.sqrt((k0 - 1) / chi2.ppf(0.10, k0 - 1)) if s == "chi2" else sh
    return int(np.clip(np.ceil((c * Z * sig / tau) ** 2), k0, KMAX))


def run(e, cands, evals, orders, variants):
    """cands/evals/orders: per draw arrays of car indices; returns rows of coverage and k* per (variant, tau)."""
    out = {(v, t): ([], []) for v in variants for t in TAUS}
    for cand, ev, order in zip(cands, evals, orders):
        o = cand[order]
        me = e[ev].mean()
        for v in variants:
            sh = e[o[:v[0]]].std(ddof=1)
            for t in TAUS:
                k = kstar(sh, *v, t)
                out[(v, t)][0].append(k); out[(v, t)][1].append(abs(me - e[o[:k]].mean()))
    rows = []
    for (v, t), (ks, bs) in out.items():
        ks, bs = np.array(ks), np.array(bs)
        rows.append({"k0": v[0], "s": v[1], "c": v[2], "tau": t, "coverage": float((bs <= t).mean()), "k_med": float(np.median(ks)), "k_p5": float(np.percentile(ks, 5)), "k_p95": float(np.percentile(ks, 95))})
    return rows


def errors_orig():
    E = {}
    for fold, fam in PAIRS:
        if fold == "E2A":
            E[fam] = stats(fold, fam)[1]["dL"].mean(0)
    assert len(E) == 7 and abs(E["M2"].mean() - 183.9) <= 0.5  # gate
    return E


def select(E):
    parts = [np.random.default_rng(4000 + p).permutation(250) for p in range(P)]
    cands, evals = [q[:125] for q in parts], [q[125:] for q in parts]
    orders = [np.random.default_rng(410000 + p).permutation(125) for p in range(P)]
    rows, ceil = [], []
    for fam, e in E.items():
        rows += [{"family": fam, **r} for r in run(e, cands, evals, orders, VARIANTS)]
        b = np.array([abs(e[ev].mean() - e[ca].mean()) for ca, ev in zip(cands, evals)])
        ceil.append({"stage": "selection", "family": fam, **{f"cov_all_{t}": float((b <= t).mean()) for t in TAUS}})
    S = pd.DataFrame(rows)
    # self-check: the plain variant on the partition-level budget study partitions must reproduce results/budget_partitions/replay_cells.csv
    p37 = pd.read_csv(ROOT / "results" / "budget_partitions" / "replay_cells.csv")
    parts37 = [np.random.default_rng(3700 + p).permutation(250) for p in range(P)]
    o37 = [np.random.default_rng(380000 + p).permutation(125) for p in range(P)]
    for fam in ("M2", "B0"):
        chk = pd.DataFrame(run(E[fam], [q[:125] for q in parts37], [q[125:] for q in parts37], o37, [PLAIN]))
        ref = p37[p37.family == fam].set_index("tau")
        assert all(abs(chk.set_index("tau").coverage[t] - ref.coverage[t]) < 1e-9 and chk.set_index("tau").k_med[t] == ref.k_med[t] for t in TAUS), fam
    return S, ceil


def choose(S):
    g = []
    for v, d in S.groupby(["k0", "s", "c"], sort=False):
        ok = all((d[d.tau == t].coverage >= 0.85).sum() >= 5 for t in (20, 28))
        g.append({"k0": v[0], "s": v[1], "c": v[2], "admissible": ok, "k20": d[d.tau == 20].k_med.mean(), "k28": d[d.tau == 28].k_med.mean(),
                  "min_cov_20_28": d[d.tau.isin([20, 28])].coverage.min(), "n_ok_20": int((d[d.tau == 20].coverage >= 0.85).sum()), "n_ok_28": int((d[d.tau == 28].coverage >= 0.85).sum()),
                  "rank_simple": (v[1] == "chi2", v[2], v[0])})
    G = pd.DataFrame(g)
    A = G[G.admissible]
    if len(A):
        best = A.assign(r=A.rank_simple).sort_values(["k20", "k28", "r"]).iloc[0]
        failed = False
    else:
        best = G.sort_values("min_cov_20_28", ascending=False).iloc[0]
        failed = True
    return G.drop(columns="rank_simple"), (int(best.k0), best.s, float(best.c)), failed


def test(sel):
    E = {}
    for fam in ("B0", "B1", "B2", "B2_300ep", "B3", "M2", "TSV"):
        n_old, n_new, z = indep_stats(fam)
        assert (n_old, n_new) == (250, 250)
        E[fam] = z["dL"].mean(0)
    assert abs(E["M2"][:250].mean() - 183.9) <= 0.5  # gate on the original cars
    cand, ev = np.arange(250), np.arange(250, 500)
    orders = [np.random.default_rng(420000 + d).permutation(250) for d in range(P)]
    rows, ceil = [], []
    for fam, e in E.items():
        rows += [{"family": fam, **r} for r in run(e, [cand] * P, [ev] * P, orders, [sel, PLAIN])]
        b = np.array([abs(e[ev].mean() - e[np.random.default_rng(430000 + d).permutation(250)[:125]].mean()) for d in range(P)])
        ceil.append({"stage": "test", "family": fam, **{f"cov_all_{t}": float((b <= t).mean()) for t in TAUS}, "mean_err_new_minus_old": e[ev].mean() - e[cand].mean()})
    return pd.DataFrame(rows), ceil


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    E = errors_orig()
    S, ceil_s = select(E)
    G, sel, failed = choose(S)
    T, ceil_t = test(sel)
    C = pd.DataFrame(ceil_s + ceil_t)
    S.round(4).to_csv(OUT / "margin_select.csv", index=False); G.round(4).to_csv(OUT / "margin_variants.csv", index=False)
    T.round(4).to_csv(OUT / "margin_test.csv", index=False); C.round(4).to_csv(OUT / "margin_ceiling.csv", index=False)
    Ts = T[(T.k0 == sel[0]) & (T.s == sel[1]) & (T.c == sel[2])]
    meets = all((Ts[Ts.tau == t].coverage >= 0.85).sum() >= 5 for t in (20, 28))
    cs = C[C.stage == "selection"]
    unreach = int((cs.cov_all_10 < 0.85).sum()) >= 5
    L = ["# Safety-margin and cumulative-lift study A: sequential procedure with a safety margin (criteria: `PREREG.md`, part A)", "",
         "Self-checks passed: M2 gate (+183.9 counts) on both data sets; the plain variant reproduces `results/budget_partitions/replay_cells.csv` (M2, B0) on the partition-level budget study partitions.", "",
         f"**A1 selection:** {'no variant admissible (selection failed); carried to the test: ' if failed else 'selected variant: '}k0 = {sel[0]}, sigma = {sel[1]}, c = {sel[2]} ({int(G.admissible.sum())} of {len(G)} variants admissible).", "",
         f"**tau = 10 counts:** {'not reachable' if unreach else 'reachable'} with 125 candidate cars (ceiling coverage below 0.85 for {int((cs.cov_all_10 < 0.85).sum())} of 7 models).", "",
         f"**A2 test verdict: {'the procedure with margin meets the target' if meets else 'does not meet it'}** (coverage >= 0.85 at tau 20: {int((Ts[Ts.tau == 20].coverage >= 0.85).sum())} of 7 models; at tau 28: {int((Ts[Ts.tau == 28].coverage >= 0.85).sum())} of 7).", "",
         "## Variants (selection; k20, k28: mean over models of the median k*)", "", md(G.round(3).to_dict("records")), "",
         "## Test cells (selected and plain variant)", "", md(T.round(3).to_dict("records")), "",
         "## Ceiling (all 125 candidate cars labelled)", "", md(C.round(3).fillna("").to_dict("records")), ""]
    (OUT / "margin_verdict.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:10]))


if __name__ == "__main__":
    main()
