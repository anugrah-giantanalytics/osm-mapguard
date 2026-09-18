"""Turning scores into the numbers a paper can report.

Built on `mapguard.evaluation.metrics`, which already does prevalence correction. What is
added here is the discipline around it:

- **The threshold is chosen on validation, applied to test.** Picking it on the test set
  and then reporting test F1 is optimistic by construction.
- **Every headline number carries a bootstrap interval.**
- **Calibration is measured, not assumed.** A model whose 0.9 means 60% is useless to a
  review queue that sorts by score, however good its ranking metrics look.
"""

from __future__ import annotations

import numpy as np

from mapguard.evaluation.metrics import average_precision, bootstrap_ci, roc_auc, summary


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 on the given (validation) set. Returns (threshold, f1)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99))):
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, float(best_f1)


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += (m.sum() / len(p)) * abs(conf - acc)
        table.append({"bin": b, "n": int(m.sum()), "mean_score": round(conf, 4),
                      "observed_rate": round(acc, 4)})
    return {
        "brier": round(float(np.mean((p - y) ** 2)), 5),
        "ece": round(float(ece), 5),
        "reliability": table,
    }


def report(y_val, p_val, y_test, p_test, *, n_boot: int = 1000, seed: int = 20260904) -> dict:
    """The full evaluation block written into results.json for any scored model."""
    y_val, p_val = np.asarray(y_val).astype(int), np.asarray(p_val, float).ravel()
    y_test, p_test = np.asarray(y_test).astype(int), np.asarray(p_test, float).ravel()

    threshold, val_f1 = best_f1_threshold(y_val, p_val)
    test = summary(y_test, p_test, threshold=threshold)
    pr_point, pr_lo, pr_hi = bootstrap_ci(y_test, p_test, metric=average_precision,
                                          n_boot=n_boot, seed=seed)
    roc_point, roc_lo, roc_hi = bootstrap_ci(y_test, p_test, metric=roc_auc,
                                             n_boot=n_boot, seed=seed)
    return {
        "threshold": {"value": round(threshold, 6), "chosen_on": "validation",
                      "validation_f1": round(val_f1, 4)},
        "validation": {"pr_auc": round(average_precision(y_val, p_val), 4),
                       "roc_auc": round(roc_auc(y_val, p_val), 4), "n": int(len(y_val))},
        "test": test,
        "test_intervals_95": {
            "pr_auc": [round(pr_point, 4), round(pr_lo, 4), round(pr_hi, 4)],
            "roc_auc": [round(roc_point, 4), round(roc_lo, 4), round(roc_hi, 4)],
        },
        "test_calibration": calibration(y_test, p_test),
        "ovid_table": ovid_table(y_test, p_test, threshold),
    }


def ovid_table(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Precision, recall, F1 and accuracy on the test set as it is, at the validation
    threshold - the four columns of Ovid's Table 1, so the numbers compare directly (Ovid's
    test set is balanced; check `positive_rate`)."""
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(1e-12, prec + rec), 4),
            "accuracy": round(float((pred == y).mean()), 4),
            "positive_rate": round(float(y.mean()), 4), "threshold": round(float(threshold), 6),
            "reference_ovid_2021": {"precision": 0.8035, "recall": 0.8302, "f1": 0.8166,
                                    "accuracy": 0.8137}}


def headline(rep: dict) -> dict:
    """The few numbers that go into index.jsonl and the paper's first table."""
    t = rep["test"]
    return {
        "test_pr_auc": rep["test_intervals_95"]["pr_auc"],
        "test_roc_auc": rep["test_intervals_95"]["roc_auc"],
        "test_pr_auc_at_1pct_prevalence": t["by_prevalence"]["0.01"]["pr_auc"],
        "test_f1_at_val_threshold": t["by_prevalence"]["0.5"]["f1"],
        "test_ovid_table": {k: rep["ovid_table"][k] for k in ("precision", "recall", "f1", "accuracy")},
        "test_ece": rep["test_calibration"]["ece"],
    }
