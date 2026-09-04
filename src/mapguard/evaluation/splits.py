"""Train/validation/test splits, and the reason there are two of them.

Ovid splits changesets at random. That puts the same vandal on both sides of the line, so
the model can learn the account rather than the edit, and the published F1 of 81.66 is
measured that way. We report against it, so we have to reproduce that protocol - but we
must not present a leaky number as our own honest result.

So there are two:

    R1a  random, seeded          comparability with published work. **Leaky. Say so.**
    R1b  user-disjoint           the honest number

Report both. **The gap between them is itself a finding** - it measures how much of the
published state of the art is account memorisation rather than edit understanding, which
is a question nobody in this literature has answered.

Splitting by user is not free: users are wildly unbalanced (a handful of prolific mappers
carry thousands of changesets), so a naive user split can miss the target proportions
badly. `user_disjoint_split` assigns users largest-first into whichever fold is furthest
below quota, which keeps the sizes close without ever splitting a user.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

DEFAULT_SEED = 20260904
DEFAULT_FRACTIONS = (0.7, 0.1, 0.2)   # train, val, test


@dataclass
class Split:
    """Boolean masks over the rows, plus enough provenance to reproduce it."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    kind: str            # "random" | "user_disjoint"
    seed: int
    leaky: bool          # true when the same user can appear on both sides

    def summary(self, y: np.ndarray | None = None) -> dict:
        out = {
            "kind": self.kind,
            "seed": self.seed,
            "leaky": self.leaky,
            "n_train": int(self.train.sum()),
            "n_val": int(self.val.sum()),
            "n_test": int(self.test.sum()),
        }
        if y is not None:
            for name, m in (("train", self.train), ("val", self.val), ("test", self.test)):
                out[f"pos_rate_{name}"] = round(float(y[m].mean()), 4) if m.sum() else None
        return out


def random_split(
    n: int, seed: int = DEFAULT_SEED, fractions: tuple[float, float, float] = DEFAULT_FRACTIONS
) -> Split:
    """R1a - Ovid's protocol.

    **This split is user-leaky.** Every table and figure built on it must say so; it
    exists for comparability with published numbers, not as our result.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)

    train = np.zeros(n, bool)
    val = np.zeros(n, bool)
    test = np.zeros(n, bool)
    train[order[:n_train]] = True
    val[order[n_train : n_train + n_val]] = True
    test[order[n_train + n_val :]] = True
    return Split(train, val, test, "random", seed, leaky=True)


def user_disjoint_split(
    uids,
    seed: int = DEFAULT_SEED,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
) -> Split:
    """R1b - no uid appears in more than one fold.

    Users are assigned largest-first to whichever fold is furthest below its quota. Doing
    it at random instead lets one prolific mapper blow a fold's size target by thousands
    of rows, and there is no way to fix that afterwards without splitting a user.

    Rows with a missing uid are treated as their own singleton users rather than being
    dropped: an anonymous or deleted account is still a real changeset, and dropping it
    would quietly change the class balance.
    """
    uids = list(uids)
    n = len(uids)

    groups: dict = defaultdict(list)
    for i, u in enumerate(uids):
        groups[u if u is not None else f"__anon_{i}"].append(i)

    rng = np.random.default_rng(seed)
    keys = sorted(groups, key=lambda k: (-len(groups[k]), str(k)))
    # Break ties randomly so the split is not an artefact of uid ordering.
    sizes = {k: len(groups[k]) for k in keys}
    keys.sort(key=lambda k: (-sizes[k], rng.random()))

    quotas = [f * n for f in fractions]
    filled = [0.0, 0.0, 0.0]
    assignment: list[list[int]] = [[], [], []]

    for k in keys:
        deficit = [quotas[i] - filled[i] for i in range(3)]
        target = int(np.argmax(deficit))
        assignment[target].extend(groups[k])
        filled[target] += sizes[k]

    masks = []
    for rows in assignment:
        m = np.zeros(n, bool)
        m[rows] = True
        masks.append(m)
    return Split(masks[0], masks[1], masks[2], "user_disjoint", seed, leaky=False)


def assert_user_disjoint(split: Split, uids) -> None:
    """Fail loudly. A silently leaky "user-disjoint" split is worse than no split."""
    uids = np.asarray(list(uids), dtype=object)
    folds = [set(uids[m]) - {None} for m in (split.train, split.val, split.test)]
    overlaps = {
        "train_val": folds[0] & folds[1],
        "train_test": folds[0] & folds[2],
        "val_test": folds[1] & folds[2],
    }
    bad = {k: len(v) for k, v in overlaps.items() if v}
    if bad:
        raise AssertionError(f"user-disjoint split leaks users across folds: {bad}")


def temporal_split(created_at, cutoff: str) -> tuple[np.ndarray, np.ndarray]:
    """R3b - everything before the cutoff trains, everything after is the test set.

    Used to guarantee that no Corpus D changeset predates the Corpus A training data.
    String comparison is safe here: OSM timestamps are ISO-8601 Zulu and sort lexically.
    """
    ts = np.array([t or "" for t in created_at])
    before = ts < cutoff
    return before, ~before
