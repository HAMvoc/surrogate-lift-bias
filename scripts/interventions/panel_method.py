"""Decomposition and first corrections study, task T8: prototype of a constant-source panel method (non-lifting potential flow, triangular panels, mirror image through the ground plane z = 0, no wake).
Bodies: an icosphere (validation, no ground) and the decimated STL of a car. Flow U = (1, 0, 0). Cp = 1 - |V|^2 / U^2 at the panel centroids; mapped to pool points by the nearest panel centroid.
Rules (PREREG T8): sphere max error < 0.05 with <= 10,000 panels; one car per style; 20 exploratory Estateback cars; stop criterion Pearson(Cp_pot, CFD) < 0.5 on upward-facing points (n_z > 0.5) outside base and wheels.
Usage: python scripts/interventions/panel_method.py sphere | cars | explore [--panels 6000]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "decomposition" / "T8"
U = np.array([1.0, 0.0, 0.0])
FOUR_PI = 4 * np.pi


def tri_geometry(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    return (a + b + c) / 3, cr / (2 * area[:, None]), area, np.stack([a, b, c], 1)


def quad_sets(tri):
    """tri (nf,3,3) -> {1: (pts, w), 3: ..., 16: ...} with weights as fractions of the panel area."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    bary = lambda l: l[0] * a + l[1] * b + l[2] * c
    q1 = (bary((1 / 3, 1 / 3, 1 / 3))[:, None], np.ones((1,)))
    q3 = (np.stack([bary((2 / 3, 1 / 6, 1 / 6)), bary((1 / 6, 2 / 3, 1 / 6)), bary((1 / 6, 1 / 6, 2 / 3))], 1), np.full(3, 1 / 3))
    pts = []
    n = 4  # edge split into 4 -> 16 sub-triangles; their centroids
    for i in range(n):
        for j in range(n - i):
            l1, l2 = (i + 1 / 3) / n, (j + 1 / 3) / n
            pts.append(bary((1 - l1 - l2, l1, l2)))
            if i + j < n - 1:
                l1, l2 = (i + 2 / 3) / n, (j + 2 / 3) / n
                pts.append(bary((1 - l1 - l2, l1, l2)))
    q16 = (np.stack(pts, 1), np.full(len(pts), 1 / len(pts)))
    return {1: q1, 3: q3, 16: q16}


def vel_pairs(x, ii, jj, areas, qs, mode):
    """Velocity kernel (x - y)/r^3/4pi integrated over panel jj with the quadrature `mode`, at the points x[ii]. Returns (m,3)."""
    pts, w = qs[mode]
    d = x[ii][:, None, :] - pts[jj]
    r3 = np.maximum((d ** 2).sum(-1), 1e-30) ** 1.5
    return ((w[None, :, None] * areas[jj][:, None, None]) * d / r3[..., None]).sum(1) / FOUR_PI


class Panels:
    def __init__(self, V, F, ground):
        self.cen, self.nrm, self.area, tri = tri_geometry(V, F)
        self.N = len(F)
        self.qs = quad_sets(tri)
        self.h = np.sqrt(self.area)
        self.sets = [(self.cen, self.qs, self.area)]
        if ground:
            m = lambda p: p * np.array([1, 1, -1.0])
            tri_i = tri * np.array([1, 1, -1.0])
            qi = quad_sets(tri_i)
            self.sets.append((m(self.cen), qi, self.area))

    def kernel_block(self, rows, mat_only=True):
        """Velocity-kernel tensor K (R, N, 3) of all panels (body + image) on the collocation points `rows` (centroids), near/mid pairs with higher-order quadrature."""
        x = self.cen[rows]
        K = np.zeros((len(rows), self.N, 3))
        for cen, qs, area in self.sets:
            d = x[:, None, :] - cen[None]
            r2 = (d ** 2).sum(-1)
            K += (area[None] / np.maximum(r2, 1e-30) ** 1.5)[..., None] * d / FOUR_PI
        # replace the far-field estimate by better quadratures where the centroid is close (compared with the panel size)
        for si, (cen, qs, area) in enumerate(self.sets):
            d = x[:, None, :] - cen[None]
            r = np.sqrt((d ** 2).sum(-1))
            for mode, lo, hi in ((16, 0.0, 3.0), (3, 3.0, 8.0)):
                ii, jj = np.nonzero((r >= lo * self.h[None]) & (r < hi * self.h[None]))
                if si == 0:
                    keep = rows[ii] != jj  # no self pair
                    ii, jj = ii[keep], jj[keep]
                if len(ii) == 0:
                    continue
                for s in range(0, len(ii), 20000):
                    a, b = ii[s:s + 20000], jj[s:s + 20000]
                    far = (area[b][:, None] / np.maximum(r[a, b] ** 3, 1e-30)[:, None]) * d[a, b] / FOUR_PI
                    K[a, b] += vel_pairs(x, a, b, area, qs, mode) - far
        # self pair (body set, i == j): the principal-value velocity of a constant source at its own centroid is (1/2) n (normal part) and 0 tangential; set below
        K[np.arange(len(rows)), rows] = 0.0
        return K

    def influence(self, block=400):
        A = np.empty((self.N, self.N))
        for s in range(0, self.N, block):
            rows = np.arange(s, min(s + block, self.N))
            K = self.kernel_block(rows)
            A[rows] = np.einsum("rnk,rk->rn", K, self.nrm[rows])
        A[np.arange(self.N), np.arange(self.N)] += 0.5
        return A

    def velocity(self, sigma, block=400):
        V = np.empty((self.N, 3))
        for s in range(0, self.N, block):
            rows = np.arange(s, min(s + block, self.N))
            K = self.kernel_block(rows)
            V[rows] = U + np.einsum("rnk,n->rk", K, sigma) + 0.5 * sigma[rows, None] * self.nrm[rows]
        return V


def solve_panels(V, F, ground):
    t0 = time.time()
    P = Panels(V, F, ground)
    A = P.influence()
    t1 = time.time()
    sigma = solve(A, -(P.nrm @ U), assume_a="gen")
    t2 = time.time()
    Vel = P.velocity(sigma)
    cp = 1 - (Vel ** 2).sum(1)
    return P, sigma, cp, {"panels": P.N, "influence_s": t1 - t0, "solve_s": t2 - t1, "velocity_s": time.time() - t2, "net_source": float((sigma * P.area).sum()), "max_normal_velocity": float(np.abs((Vel * P.nrm).sum(1)).max())}


def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    V = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    F = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    for _ in range(level):
        cache, newF = {}, []
        V = list(V)

        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = (V[i] + V[j]) / 2
                V.append(m / np.linalg.norm(m))
                cache[key] = len(V) - 1
            return cache[key]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            newF += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        V, F = np.array(V), np.array(newF)
    return V, F


def run_sphere():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for level in (3, 4):
        V, F = icosphere(level)
        P, sigma, cp, info = solve_panels(V, F, ground=False)
        theta = np.arccos(np.clip(P.cen[:, 0] / np.linalg.norm(P.cen, axis=1), -1, 1))
        exact = 1 - 2.25 * np.sin(theta) ** 2
        err = np.abs(cp - exact)
        rows.append({"level": level, "panels": P.N, "max |Cp - exact|": f"{err.max():.4f}", "mean |Cp - exact|": f"{err.mean():.4f}", "net source": f"{info['net_source']:.1e}", "max normal velocity": f"{info['max_normal_velocity']:.1e}",
                     "time s": f"{info['influence_s'] + info['solve_s'] + info['velocity_s']:.0f}"})
        print(rows[-1], flush=True)
    from scripts.diagnosis.region_errors import md
    (OUT / "T8_sphere.md").write_text("# T8 sphere validation (no ground)\n\nCp = 1 - 2.25 sin^2(theta), theta from the flow axis; constant-source triangular panels, Cp at the panel centroids (the centroids of flat panels lie slightly inside the sphere).\n\n" + md(rows) + "\n", encoding="utf-8")
    return rows


def car_mesh(sid, panels):
    import open3d as o3d
    from scripts.diagnosis.force_alignment import load_pool
    m = o3d.io.read_triangle_mesh(str(ROOT / "data" / "raw" / "meshes" / "3DMeshesSTL" / f"{sid}.stl"))
    m.remove_duplicated_vertices()
    m.remove_degenerate_triangles()
    m = m.simplify_quadric_decimation(target_number_of_triangles=panels)
    V, F = np.asarray(m.vertices), np.asarray(m.triangles)
    xyz = load_pool(sid)[0].astype(np.float64) * 5.0
    if np.ptp(V[:, 0]) > 50:  # millimetres
        V = V / 1000.0
    # translation only: align the minimum x and z and the y-centre with the pool (the cache frame: nose at x = 0, ground at z = 0, symmetry plane y = 0)
    V = V - np.array([V[:, 0].min() - xyz[:, 0].min(), (V[:, 1].min() + V[:, 1].max()) / 2 - (xyz[:, 1].min() + xyz[:, 1].max()) / 2, V[:, 2].min() - xyz[:, 2].min()])
    # outward normals: positive signed volume
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        F = F[:, [0, 2, 1]]
    return V, F, xyz


def run_car(sid, panels):
    from scripts.diagnosis.pressure_coefficients import CLIP
    from scripts.diagnosis.force_alignment import load_pool
    from scripts.diagnosis.regions import labeler
    V, F, xyz = car_mesh(sid, panels)
    P, sigma, cp, info = solve_panels(V, F, ground=True)
    _, nrm, cp_cfd, _ = load_pool(sid)
    lab = labeler(sid, load_pool(sid)[0], nrm)
    idx = cKDTree(P.cen).query(xyz)[1]
    cp_pot = cp[idx]
    ref = np.clip(cp_cfd.astype(np.float64), *CLIP)
    sel = (nrm[:, 2] > 0.5) & (lab != 0) & (lab != 1)  # outside wheels (0) and base (1)
    r_sel = float(np.corrcoef(cp_pot[sel], ref[sel])[0, 1])
    r_all = float(np.corrcoef(cp_pot, ref)[0, 1])
    return {"id": sid, "panels": P.N, "pearson_upward": r_sel, "pearson_all": r_all, "n_upward": int(sel.sum()), "cp_pot_mean_upward": float(cp_pot[sel].mean()), "cfd_mean_upward": float(ref[sel].mean()), **info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sphere", "cars", "explore"])
    ap.add_argument("--panels", type=int, default=6000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    if a.mode == "sphere":
        run_sphere()
        return
    from src.data.dataset import load_split
    ids = ["E_S_WW_WM_143", "N_S_WW_WM_066", "F_S_WWS_WM_502"] if a.mode == "cars" else sorted(i for i in load_split("E1")["train"] if i[0] == "E")[:20]
    rows = []
    for sid in ids:
        t0 = time.time()
        r = run_car(sid, a.panels)
        r["total_s"] = time.time() - t0
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        pd.DataFrame(rows).round(4).to_csv(OUT / f"T8_{a.mode}.csv", index=False)


if __name__ == "__main__":
    main()
