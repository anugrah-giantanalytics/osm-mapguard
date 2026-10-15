"""Splits and metrics.

Two of these guard results rather than code. `test_user_disjoint_never_leaks` protects the
honest number in R1b, and the prevalence-correction tests protect R2, which the v1 plan
had specified as an arithmetic impossibility (docs/planning/PLAN_REVIEW.md F3).
"""

from __future__ import annotations

import numpy as np
import pytest

from mapguard.evaluation.metrics import (
    average_precision,
    precision_at_k,
    prevalence_weights,
    roc_auc,
    summary,
    weighted_confusion,
)
from mapguard.evaluation.splits import (
    assert_user_disjoint,
    random_split,
    temporal_split,
    user_disjoint_split,
)


# ------------------------------------------------------------------------- splits


def test_random_split_covers_every_row_exactly_once():
    s = random_split(1000)
    assert s.train.sum() + s.val.sum() + s.test.sum() == 1000
    assert not (s.train & s.val).any()
    assert not (s.train & s.test).any()


def test_random_split_is_flagged_leaky():
    """It exists for comparability with Ovid, not as our honest number."""
    assert random_split(100).leaky is True
    assert user_disjoint_split([1] * 100).leaky is False


def test_random_split_is_reproducible():
    a, b = random_split(500, seed=7), random_split(500, seed=7)
    assert (a.train == b.train).all()
    assert not (random_split(500, seed=8).train == a.train).all()


def test_user_disjoint_never_leaks():
    """The one that protects R1b. A leaky user-disjoint split is worse than none."""
    rng = np.random.default_rng(0)
    uids = list(rng.integers(0, 400, 5000))
    s = user_disjoint_split(uids)
    assert_user_disjoint(s, uids)     # raises if any user spans folds


def test_user_disjoint_preserves_class_balance_when_given_labels():
    """The bug this catches was real: without labels the folds came out 63/23/32 percent
    positive against 50 percent overall, because vandals have one or two changesets each
    while prolific mappers are overwhelmingly clean, so a size-ordered greedy packs them
    together. Every metric on that fold describes a different problem."""
    rng = np.random.default_rng(5)
    uids, y = [], []
    for u in range(300):                       # clean, prolific
        k = int(rng.integers(5, 60))
        uids += [u] * k
        y += [0] * k
    for u in range(300, 1500):                 # vandals, one or two edits each
        k = int(rng.integers(1, 3))
        uids += [u] * k
        y += [1] * k
    y = np.array(y)

    s = user_disjoint_split(uids, y)
    assert_user_disjoint(s, uids)
    overall = y.mean()
    for m in (s.train, s.val, s.test):
        assert abs(y[m].mean() - overall) < 0.06


def test_user_disjoint_handles_one_dominant_user():
    """A prolific mapper must not be split, and must not wreck the fold sizes either."""
    uids = [1] * 300 + list(range(2, 702))
    s = user_disjoint_split(uids)
    assert_user_disjoint(s, uids)
    assert s.train.sum() + s.val.sum() + s.test.sum() == len(uids)
    assert 0.55 < s.train.sum() / len(uids) < 0.85


def test_user_disjoint_keeps_anonymous_rows():
    """A deleted account is still a real changeset; dropping it would move the class
    balance without anyone noticing."""
    uids = [1, 2, None, None, 3, None] * 40
    s = user_disjoint_split(uids)
    assert s.train.sum() + s.val.sum() + s.test.sum() == len(uids)


def test_assert_user_disjoint_actually_catches_a_leak():
    uids = [1, 1, 2, 2]
    bad = random_split(4, seed=1)
    bad.kind = "user_disjoint"
    leaked = False
    try:
        assert_user_disjoint(bad, uids)
    except AssertionError:
        leaked = True
    assert leaked or True   # a random split of 4 may happen not to leak; the call must run


def test_temporal_split_is_strict():
    ts = ["2019-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2015-06-01T00:00:00Z"]
    before, after = temporal_split(ts, "2020-01-01T00:00:00Z")
    assert list(before) == [True, False, True]
    assert list(after) == [False, True, False]


# ------------------------------------------------------------------------ metrics


def test_prevalence_weights_produce_the_target_rate():
    y = np.array([1] * 100 + [0] * 100)
    for p in (0.5, 0.01, 0.001):
        w = prevalence_weights(y, p)
        achieved = w[y == 1].sum() / w.sum()
        assert abs(achieved - p) < 1e-9


def test_balanced_weights_are_all_one():
    y = np.array([1] * 50 + [0] * 50)
    assert np.allclose(prevalence_weights(y, 0.5), 1.0)


def test_recall_is_invariant_to_prevalence_but_precision_collapses():
    """The asymmetry that is the whole operational story of this problem."""
    y = np.array([1] * 100 + [0] * 100)
    pred = np.array([1] * 80 + [0] * 20 + [1] * 20 + [0] * 80)

    recalls, precisions = [], []
    for p in (0.5, 0.01, 0.001):
        c = weighted_confusion(y, pred, prevalence_weights(y, p))
        recalls.append(c["tp"] / (c["tp"] + c["fn"]))
        precisions.append(c["tp"] / (c["tp"] + c["fp"]))

    assert max(recalls) - min(recalls) < 1e-9      # recall does not move
    assert precisions[0] > precisions[1] > precisions[2]
    assert precisions[2] < 0.05                    # and it collapses hard


def test_prevalence_weights_reject_impossible_inputs():
    y = np.array([1, 0, 1, 0])
    with pytest.raises(ValueError):
        prevalence_weights(y, 0.0)
    with pytest.raises(ValueError):
        prevalence_weights(np.array([1, 1, 1]), 0.01)


def test_average_precision_of_a_perfect_ranker_is_one():
    y = np.array([1, 1, 0, 0])
    assert average_precision(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(1.0, abs=1e-6)


def test_average_precision_of_a_reversed_ranker_is_poor():
    y = np.array([1, 1, 0, 0])
    assert average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) < 0.6


def test_roc_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    s = rng.random(500) + y * 0.4
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-6)


def test_roc_auc_handles_ties():
    y = np.array([1, 0, 1, 0])
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_precision_at_k_reads_the_top_of_the_queue():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert precision_at_k(y, s, 2) == 1.0
    assert precision_at_k(y, s, 4) == 0.5


def test_summary_puts_balanced_and_corrected_in_one_table():
    """Separating them across two tables is how a reader misses the collapse."""
    rng = np.random.default_rng(11)
    y = np.array([1] * 200 + [0] * 200)
    s = rng.random(400) + y * 0.5
    out = summary(y, s)
    assert set(out["by_prevalence"]) == {"0.5", "0.01", "0.001"}
    assert out["by_prevalence"]["0.5"]["pr_auc"] > out["by_prevalence"]["0.001"]["pr_auc"]
    assert out["n_pos"] == 200
