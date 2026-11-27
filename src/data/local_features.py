"""Per-point local surface features of the 100k-point pool (used by M2 with --local-feats; 6 features per point, in the stored pool order).
Neighbours are found among the pool points of the same car (k = 64 nearest, the first 16 are the 16-neighbourhood).
  0: surface variation lambda3 / (lambda1 + lambda2 + lambda3) of the 16 neighbours (curvature proxy, fine scale)
  1: the same for the 64 neighbours (coarse scale)
  2, 3: lambda1 / sum and lambda2 / sum of the 16 neighbours (eigenvalues of the covariance of the neighbour coordinates, descending)
  4: standard deviation of the unit normals over the 16 neighbours, sqrt(sum over components of the variance)
  5: the same over the 64 neighbours
The normal itself is already an input of M2. Unit-free (ratios), so the L_ref scaling of the cache does not matter."""
import numpy as np
from scipy.spatial import cKDTree

N_LOCAL = 6


def _eig_ratios(P):
    """P (N, k, 3) neighbour coordinates -> normalised eigenvalues (N, 3), descending."""
    C = P - P.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", C, C) / P.shape[1]
    ev = np.linalg.eigvalsh(cov)[:, ::-1]  # descending
    return ev / np.maximum(ev.sum(1, keepdims=True), 1e-30)


def compute(xyz, nrm):
    xyz, nrm = np.asarray(xyz, np.float64), np.asarray(nrm, np.float64)
    idx = cKDTree(xyz).query(xyz, k=64, workers=-1)[1]
    e16, e64 = _eig_ratios(xyz[idx[:, :16]]), _eig_ratios(xyz[idx])
    s16 = np.sqrt(nrm[idx[:, :16]].var(1).sum(1))
    s64 = np.sqrt(nrm[idx].var(1).sum(1))
    return np.stack([e16[:, 2], e64[:, 2], e16[:, 0], e16[:, 1], s16, s64], 1).astype(np.float32)
