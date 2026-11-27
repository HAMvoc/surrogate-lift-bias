"""Metrics, spec 10. The pool is sampled proportionally to area, so the plain mean over points = the area-weighted mean."""
import numpy as np
from scipy import stats


def car_stats(pred, ref):
    pred, ref = np.asarray(pred, np.float64), np.asarray(ref, np.float64)
    e = pred - ref
    sse, sc2 = float((e ** 2).sum()), float((ref ** 2).sum())
    return {"n": len(ref), "rel_l2": float(np.sqrt(sse / sc2)), "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt(sse / len(ref))), "sse": sse, "sum_c": float(ref.sum()), "sum_c2": sc2}


def pooled_r2(df):
    """R² pooled over all points of all cars in df (columns n, sum_c, sum_c2, sse)."""
    n = df.n.sum()
    mean = df.sum_c.sum() / n
    return float(1 - df.sse.sum() / (df.sum_c2.sum() - n * mean ** 2))


def bootstrap_ci(x, B=2000, seed=0, level=0.95):
    """Per-car bootstrap CI of the mean."""
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    a = (1 - level) / 2
    return float(x.mean()), float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def paired_compare(a, b, B=2000, seed=0):
    """Paired per-car difference a - b: (mean, lo, hi, Wilcoxon p). Negative means a is better when lower metric is better."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    m, lo, hi = bootstrap_ci(d, B, seed)
    p = 1.0 if np.all(d == 0) else float(stats.wilcoxon(d).pvalue)
    return m, lo, hi, p


def holm(p):
    """Holm-Bonferroni correction: adjusted p, same order as the input."""
    p = np.asarray(p, np.float64)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, k in enumerate(o):
        run = max(run, (len(p) - r) * p[k])
        adj[k] = min(1.0, run)
    return adj


def degradation_D(held, base, B=2000, seed=0):
    """D = mean rel L2 when the body style is held out / when it is in train (E1), two-independent-sample bootstrap CI."""
    held, base = np.asarray(held, np.float64), np.asarray(base, np.float64)
    rng = np.random.default_rng(seed)
    r = held[rng.integers(0, len(held), (B, len(held)))].mean(1) / base[rng.integers(0, len(base), (B, len(base)))].mean(1)
    return float(held.mean() / base.mean()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def summarize(df, col="rel_l2", B=2000, seed=0):
    m, lo, hi = bootstrap_ci(df[col], B, seed)
    return {"n_cars": len(df), col: m, "ci_lo": lo, "ci_hi": hi, "median": float(df[col].median()),
            "mae": float(df.mae.mean()), "rmse": float(df.rmse.mean()), "r2_pooled": pooled_r2(df)}
