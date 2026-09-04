"""Metrics, and the prevalence correction that makes R2 possible at all.

The v1 plan asked for evaluation at 1:100 and 1:1000 prevalence by "upsampling negatives".
The corpus has 9,138 negatives. Against 9,138 positives, 1:100 needs 913,800 of them and
1:1000 needs 9,138,000 - and duplicating the ones we have changes nothing about PR-AUC
except to inflate the denominator with copies. You cannot upsample your way to a lower
positive rate. (docs/PLAN_REVIEW.md finding F3.)

The fix is importance weighting. Score the balanced test set once, then reweight the
negatives to what they would be at the target prevalence:

    w = ((1 - p) / p) * (n_pos / n_neg)

Precision computed with those weights is the precision you would see at prevalence `p`.
It is exact, it is standard, and it costs one multiplication instead of ten days of
fetching.

**Recall does not move.** It only involves positives, and their weights are unchanged.
Precision collapses. That asymmetry is the entire operational story of this problem and
it is why F1 on a balanced set is a misleading number to publish.
"""

from __future__ import annotations

import numpy as np


def prevalence_weights(y: np.ndarray, target_prevalence: float) -> np.ndarray:
    """Per-row weights that turn a balanced set into a `target_prevalence` set.

    Positives keep weight 1; each negative counts for as many negatives as the target
    prevalence implies. Nothing is duplicated and nothing is discarded.
    """
    if not 0.0 < target_prevalence < 1.0:
        raise ValueError("target_prevalence must be strictly between 0 and 1")
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both classes to reweight")

    w_neg = ((1.0 - target_prevalence) / target_prevalence) * (n_pos / n_neg)
    w = np.ones(len(y), dtype=float)
    w[y == 0] = w_neg
    return w


def weighted_confusion(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> dict:
    y, pred, w = np.asarray(y).astype(int), np.asarray(pred).astype(int), np.asarray(w, float)
    return {
        "tp": float(w[(y == 1) & (pred == 1)].sum()),
        "fp": float(w[(y == 0) & (pred == 1)].sum()),
        "fn": float(w[(y == 1) & (pred == 0)].sum()),
        "tn": float(w[(y == 0) & (pred == 0)].sum()),
    }


def precision_recall_curve(
    y: np.ndarray, scores: np.ndarray, w: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted PR curve. Returns (precision, recall, thresholds), recall ascending.

    Written out rather than delegated because scikit-learn's `sample_weight` path is easy
    to pass silently wrong, and this is the number the paper turns on.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)

    order = np.argsort(-s)
    y, s, w = y[order], s[order], w[order]

    tp = np.cumsum(w * (y == 1))
    fp = np.cumsum(w * (y == 0))
    total_pos = float((w * (y == 1)).sum())

    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(total_pos, 1e-12)
    return precision, recall, s


def average_precision(y: np.ndarray, scores: np.ndarray, w: np.ndarray | None = None) -> float:
    """Area under the weighted PR curve, by the step-wise (interpolation-free) rule."""
    precision, recall, _ = precision_recall_curve(y, scores, w)
    d_recall = np.diff(np.concatenate([[0.0], recall]))
    return float((precision * d_recall).sum())


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Of the k changesets we would actually send to a human, how many are damaging?

    The metric a review queue cares about. Unweighted on purpose: k is a real number of
    real items someone has to open, so reweighting it would describe a queue nobody runs.
    """
    y = np.asarray(y).astype(int)
    idx = np.argsort(-np.asarray(scores, float))[:k]
    return float(y[idx].mean()) if k else 0.0


def summary(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    prevalences: tuple[float, ...] = (0.5, 0.01, 0.001),
    ks: tuple[int, ...] = (10, 50, 100),
) -> dict:
    """Everything R1 and R2 need, in one dict, ready for results/*.json.

    `prevalences` includes 0.5 so the balanced number sits in the same table as the
    corrected ones - the collapse between them is the finding, and separating them across
    two tables is how a reader misses it.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, float)
    pred = (s >= threshold).astype(int)

    out: dict = {
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "observed_prevalence": round(float(y.mean()), 6),
        "threshold": threshold,
        "roc_auc": roc_auc(y, s),
        "precision_at_k": {str(k): round(precision_at_k(y, s, k), 4) for k in ks},
        "by_prevalence": {},
    }

    for p in prevalences:
        w = prevalence_weights(y, p)
        c = weighted_confusion(y, pred, w)
        prec = c["tp"] / max(c["tp"] + c["fp"], 1e-12)
        rec = c["tp"] / max(c["tp"] + c["fn"], 1e-12)
        out["by_prevalence"][str(p)] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-12), 4),
            "pr_auc": round(average_precision(y, s, w), 4),
        }
    return out


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC. Unaffected by prevalence, which is exactly why it is not enough."""
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks within ties, or identical scores bias the result.
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    metric=average_precision,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 20260904,
) -> tuple[float, float, float]:
    """(point estimate, lower, upper). A number without an interval does not go in the paper."""
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, float)
    rng = np.random.default_rng(seed)
    point = metric(y, s)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric(y[idx], s[idx]))
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
