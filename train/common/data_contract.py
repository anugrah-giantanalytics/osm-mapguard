"""The data contract - the one place the training code meets the dataset.

The dataset is not finalised yet. Rather than guess its shape inside every training
script, this module states what the feature store **must** provide, validates it, and
refuses to run otherwise. When the data is built, it is built to this contract, and
nothing under train/ has to change.

Why refuse rather than fill: `mapguard.training.dataset.build_batch` fills a missing
tabular column with zeros so that partial stores can be explored. That is fine for
exploration and wrong for a reported result, because a model trained with a column
silently zeroed produces a real-looking number about a different feature set. Here a
missing required column is an error, and a missing optional column is recorded in the
run's results so it cannot go unnoticed.

Tables (Parquet, under the feature-store directory):

    changesets.parquet   one row per changeset
    edits.parquet        one row per sampled edit (K <= 256 per changeset)
    user_seq.parquet     one row per point-in-time history step

Task selection needs a `corpus` column on changesets:

    A  historical Ovid benchmark (revert-derived labels)
    M  modern OSMCha-reviewed changesets (label 1 = harmful, 0 = reviewed-and-cleared)
    N  random unreviewed changesets from the same window as M (presumed clean, label 0)

and the tasks are:

    T1   triage     corpus M only           harmful vs reviewed-and-cleared
    T2   detection  M positives + corpus N  harmful vs random unreviewed
    ovid benchmark  corpus A only           comparable with the published F1
    all             everything, no filtering - exploration only, not reportable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mapguard.training.dataset import EDIT_FEATURES, TABULAR_FEATURES

TABLES = {
    "changesets": "changesets.parquet",
    "edits": "edits.parquet",
    "user_seq": "user_seq.parquet",
}

REQUIRED = {
    "changesets": ("cid", "label", "uid", "created_at", "text_blob"),
    "edits": ("cid",) + tuple(EDIT_FEATURES),
    "user_seq": ("cid", "step", "num_changes", "gap_days", "comment_length", "bbox_diag_deg"),
}

# Present in the plan's feature set; their absence is recorded, not fatal.
OPTIONAL = {
    "changesets": tuple(TABULAR_FEATURES) + ("corpus", "editor", "comment_length"),
}

CORPORA = ("A", "M", "N")


class ContractError(SystemExit):
    pass


@dataclass
class FeatureStore:
    root: Path
    changesets: object          # pandas.DataFrame
    edits: object
    user_seq: object
    files: dict = field(default_factory=dict)
    missing_optional: list = field(default_factory=list)
    task: str = "all"

    def describe(self) -> dict:
        y = self.changesets["label"].to_numpy()
        out = {
            "root": str(self.root),
            "task": self.task,
            "n_changesets": int(len(self.changesets)),
            "n_positive": int((y == 1).sum()),
            "n_negative": int((y == 0).sum()),
            "n_edit_rows": int(len(self.edits)),
            "n_user_seq_rows": int(len(self.user_seq)),
            "missing_optional_columns": self.missing_optional,
        }
        if "corpus" in self.changesets.columns:
            out["by_corpus"] = {str(k): int(v) for k, v in
                                self.changesets["corpus"].value_counts().items()}
        return out


def default_root() -> Path:
    from mapguard.config import interim_dir

    return interim_dir()


def load(root: str | Path | None = None, task: str = "all") -> FeatureStore:
    import pandas as pd
    import pyarrow.parquet as pq

    root = Path(root) if root else default_root()
    files = {name: root / fname for name, fname in TABLES.items()}
    absent = [str(p) for p in files.values() if not p.exists()]
    if absent:
        raise ContractError(
            "feature store is incomplete - missing:\n  " + "\n  ".join(absent) +
            "\n\nThe dataset is configured separately (docs/DATA_FOUNDATION.md). Build the "
            "feature store first, or point data.feature_store at a directory that has it."
        )

    frames = {name: pq.read_table(path).to_pandas() for name, path in files.items()}

    problems = []
    for name, cols in REQUIRED.items():
        missing = [c for c in cols if c not in frames[name].columns]
        if missing:
            problems.append(f"{TABLES[name]}: missing required {missing}")
    if problems:
        raise ContractError("feature store violates the data contract:\n  " + "\n  ".join(problems))

    cs = frames["changesets"]
    missing_optional = [c for c in OPTIONAL["changesets"] if c not in cs.columns]

    labels = set(pd.unique(cs["label"].dropna()))
    if not labels <= {0, 1}:
        raise ContractError(f"changesets.label must be 0/1, found {sorted(labels)[:10]}")
    if cs["label"].isna().any():
        raise ContractError(f"{int(cs['label'].isna().sum())} changesets have no label; "
                            "unlabelled rows do not belong in a training store")
    if cs["cid"].duplicated().any():
        raise ContractError(f"{int(cs['cid'].duplicated().sum())} duplicate cids in changesets")

    cs = select_task(cs, task)
    keep = set(cs["cid"].tolist())
    edits = frames["edits"][frames["edits"]["cid"].isin(keep)]
    user_seq = frames["user_seq"][frames["user_seq"]["cid"].isin(keep)]

    return FeatureStore(
        root=root,
        changesets=cs.reset_index(drop=True),
        edits=edits.reset_index(drop=True),
        user_seq=user_seq.reset_index(drop=True),
        files=files,
        missing_optional=missing_optional,
        task=task,
    )


def select_task(cs, task: str):
    """Filter changesets to one task. See the module docstring for definitions."""
    if task == "all":
        return cs
    if "corpus" not in cs.columns:
        raise ContractError(
            f"task {task!r} needs a 'corpus' column on changesets (A/M/N) to tell the "
            "label sources apart. Without it the only valid task is 'all', and 'all' "
            "mixes labels with different meanings, so it is not reportable."
        )
    bad = sorted(set(cs["corpus"].dropna().unique()) - set(CORPORA))
    if bad:
        raise ContractError(f"unknown corpus values {bad}; expected {CORPORA}")

    corpus, label = cs["corpus"], cs["label"]
    if task == "T1":
        mask = corpus == "M"
    elif task == "T2":
        mask = ((corpus == "M") & (label == 1)) | (corpus == "N")
        n_pos_in_n = int(((corpus == "N") & (label == 1)).sum())
        if n_pos_in_n:
            raise ContractError(f"{n_pos_in_n} corpus-N rows carry label 1; corpus N is "
                                "random unreviewed changesets and must be labelled 0")
    elif task == "ovid":
        mask = corpus == "A"
    else:
        raise ContractError(f"unknown task {task!r}")

    out = cs[mask]
    y = out["label"].to_numpy()
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        raise ContractError(f"task {task!r} selects {len(out)} rows with only one class present")
    return out


def labels(store: FeatureStore) -> np.ndarray:
    return store.changesets["label"].to_numpy(dtype="float32")
