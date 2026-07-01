"""Clustering evaluation metrics: NMI, ARI and ACC.

NMI and ARI are the two metrics reported in scDMKC (Section 4.3); ACC is added
because the scMUG side reports it too, which is handy for the integration phase.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def nmi(y_true, y_pred) -> float:
    return float(normalized_mutual_info_score(y_true, y_pred))


def ari(y_true, y_pred) -> float:
    return float(adjusted_rand_score(y_true, y_pred))


def acc(y_true, y_pred) -> float:
    """Clustering accuracy via optimal (Hungarian) label matching."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Map labels to 0..K-1.
    _, y_true = np.unique(y_true, return_inverse=True)
    _, y_pred = np.unique(y_pred, return_inverse=True)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return float(w[row, col].sum()) / y_pred.size


def evaluate(y_true, y_pred) -> dict:
    return {"NMI": nmi(y_true, y_pred), "ARI": ari(y_true, y_pred),
            "ACC": acc(y_true, y_pred)}
