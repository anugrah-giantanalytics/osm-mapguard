"""Frozen split files.

A split that is recomputed at training time is a split that silently changes whenever the
split code, the row order or the dataset changes - and then two "comparable" models were
trained on different rows. So a split is created once, written as lists of changeset ids,
and every later run on the same task reads that file.

    experiments/splits/<task>_<split>_seed<seed>.json

Reuse is guarded. The file records a fingerprint of the ids it was built from; if the
current data holds a different set of ids, the split is refused rather than quietly
re-aligned. Rebuild deliberately with `--set resplit=true`, and treat every result from
the old split as belonging to a different dataset.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mapguard.evaluation.splits import (
    assert_user_disjoint,
    random_split,
    temporal_split,
    user_disjoint_split,
)

from .run import REPO

SPLITS_DIR = REPO / "experiments" / "splits"


def id_fingerprint(cids) -> str:
    joined = ",".join(str(int(c)) for c in sorted(int(c) for c in cids))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def split_path(task: str, split: str, seed: int, cutoff: str | None = None) -> Path:
    name = f"{task}_{split}_seed{seed}"
    if split == "temporal" and cutoff:
        name += "_" + cutoff[:10]
    return SPLITS_DIR / f"{name}.json"


def _masks_from_ids(cids: np.ndarray, doc: dict) -> dict[str, np.ndarray]:
    index = {int(c): i for i, c in enumerate(cids)}
    out = {}
    for fold in ("train", "val", "test"):
        rows = [index[int(c)] for c in doc[fold] if int(c) in index]
        m = np.zeros(len(cids), bool)
        m[rows] = True
        out[fold] = m
    return out


def get_split(prep, *, task: str, split: str, seed: int, cutoff: str | None = None,
              val_fraction: float = 0.1, resplit: bool = False,
              exclude_test_users_from_train: bool = True) -> tuple[dict, Path, dict]:
    """Load the frozen split for (task, split, seed), creating it if it does not exist.

    Returns (masks, path, summary) where masks has boolean train/val/test over prep's rows.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = split_path(task, split, seed, cutoff)
    fp = id_fingerprint(prep.cids)

    if path.exists() and not resplit:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc["id_fingerprint"] != fp:
            raise SystemExit(
                f"frozen split {path.name} was built from a different set of changesets "
                f"(fingerprint {doc['id_fingerprint']} vs current {fp}).\n"
                "The data changed. Rebuild on purpose with --set resplit=true, and treat "
                "results on the old split as a different dataset."
            )
        masks = _masks_from_ids(prep.cids, doc)
        _check(masks, prep, split)
        return masks, path, doc["summary"]

    y, uids, n = prep.y, list(prep.uids), len(prep.y)
    if split == "R1a":
        s = random_split(n, seed=seed)
        masks, leaky = {"train": s.train, "val": s.val, "test": s.test}, True
    elif split == "R1b":
        s = user_disjoint_split(uids, y, seed=seed)
        masks, leaky = {"train": s.train, "val": s.val, "test": s.test}, False
    elif split == "temporal":
        if not cutoff:
            raise SystemExit("temporal split needs a cutoff")
        past, future = temporal_split(prep.created_at, cutoff)
        rng = np.random.default_rng(seed)
        past_rows = np.flatnonzero(past)
        rng.shuffle(past_rows)
        n_val = int(len(past_rows) * val_fraction)
        val = np.zeros(n, bool); val[past_rows[:n_val]] = True
        train = past & ~val
        test = future.copy()
        if exclude_test_users_from_train:
            # Without this, an account active on both sides of the cutoff lets the model
            # recognise the account instead of the edit, and the temporal result inherits
            # the same leak R1b exists to remove.
            test_users = {u for u in np.asarray(uids, dtype=object)[test] if u is not None}
            seen = np.array([u in test_users for u in uids])
            train &= ~seen
            val &= ~seen
        masks, leaky = {"train": train, "val": val, "test": test}, False
    else:
        raise SystemExit(f"unknown split {split!r}")

    _check(masks, prep, split)
    summary = {
        "kind": split, "leaky": leaky, "seed": seed, "cutoff": cutoff,
        **{f"n_{k}": int(v.sum()) for k, v in masks.items()},
        **{f"pos_rate_{k}": (round(float(y[v].mean()), 4) if v.any() else None)
           for k, v in masks.items()},
    }
    doc = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": task, "split": split, "seed": seed, "cutoff": cutoff,
        "id_fingerprint": fp, "n_total": n, "summary": summary,
        **{k: [int(c) for c in prep.cids[v]] for k, v in masks.items()},
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    print(f"created frozen split {path.name}: {summary}")
    return masks, path, summary


def _check(masks: dict, prep, split: str) -> None:
    t, v, s = masks["train"], masks["val"], masks["test"]
    if (t & v).any() or (t & s).any() or (v & s).any():
        raise SystemExit("split folds overlap")
    for name, m in masks.items():
        if not m.any():
            raise SystemExit(f"split fold '{name}' is empty")
        if len(np.unique(prep.y[m])) < 2:
            raise SystemExit(f"split fold '{name}' contains only one class")
    if split in ("R1b",):
        from mapguard.evaluation.splits import Split

        assert_user_disjoint(Split(t, v, s, split, 0, False), list(prep.uids))
