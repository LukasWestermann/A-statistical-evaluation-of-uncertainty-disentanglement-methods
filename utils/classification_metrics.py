"""
Classification metrics: accuracy, ECE, ROC-AUC.
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np


def accuracy_from_probs(probs: np.ndarray, y_true: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = np.argmax(probs, axis=1)
    return float(np.mean(y_pred == y_true))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute ECE with equal-width confidence bins.
    """
    y_true = y_true.reshape(-1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if np.any(mask):
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += np.abs(bin_conf - bin_acc) * (mask.mean())
    return float(ece)


def roc_auc_score_manual(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute ROC-AUC using rank statistics (no sklearn dependency).
    labels: 1 for positive, 0 for negative.
    """
    scores = scores.reshape(-1)
    labels = labels.reshape(-1).astype(int)
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")

    # Rank scores (average ranks for ties)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores)) + 1

    # Handle ties: average ranks for equal scores
    unique_scores, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for idx, cnt in enumerate(counts):
            if cnt > 1:
                tie_indices = np.where(inv == idx)[0]
                ranks[tie_indices] = ranks[tie_indices].mean()

    sum_ranks_pos = ranks[pos].sum()
    n_pos = pos.sum()
    n_neg = neg.sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)
