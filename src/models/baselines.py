"""B0 (position-wise mean Cp field) and B1 (geometric kNN on R1 SDF), spec 7.4. Runs on CPU."""
import numpy as np
from scipy.spatial import cKDTree


class TrainBank:
    """KD-tree (built on demand) over each train car's pool coordinates; look up the nearest point and take its (clipped) Cp."""

    def __init__(self, xyz, cp):
        self.xyz, self.cp, self.trees = np.asarray(xyz, np.float64), np.asarray(cp, np.float32), {}

    def cp_at(self, car, pts):
        if car not in self.trees:
            self.trees[car] = cKDTree(self.xyz[car])
        return self.cp[car][self.trees[car].query(pts, workers=-1)[1]]


def predict_b0(bank, pts, cars):
    """Mean Cp at the nearest point over the train cars `cars`.
    shortcut: only a subset of train cars (cost scales with cars x points), large enough that sampling noise is small; increase if needed."""
    return np.mean([bank.cp_at(int(i), pts) for i in cars], axis=0)


def predict_b1(bank, pts, dist_to_train, ks, eps=1e-6):
    """k nearest train cars by R1 SDF distance; 1/d-weighted mean of Cp at the nearest point on each car.
    Returns {k: prediction}; each neighbour car is looked up once for all k."""
    nn = np.argsort(dist_to_train)[: max(ks)]
    c = [bank.cp_at(int(i), pts) for i in nn]
    w = 1.0 / (dist_to_train[nn] + eps)
    return {k: sum(w[j] * c[j] for j in range(k)) / w[:k].sum() for k in ks}
