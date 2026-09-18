"""Feature store -> model-ready arrays: assembly, branch column selection, normalisation.

Three decisions live here because getting any of them wrong produces a plausible number
rather than an error.

**Heavy tails are log-compressed before scaling.** Geometry features span six orders of
magnitude - a nudged node moves 0.4 m, a vandalised one 400,000 m, and the largest Corpus A
changeset has 50,000 edits. Z-scoring raw values lets a handful of extreme rows set the
scale for everything, and the ordinary rows collapse to near-identical inputs. A signed
log1p keeps the ordering, keeps the sign of deltas, and brings the range to something a
dense layer can use.

**Statistics come from training rows only**, and from *real* edit positions only - padded
slots are zeros that would otherwise drag every mean toward zero. They are saved with the
run so evaluation applies exactly the same transform.

**Every row keeps at least one real edit.** A changeset with no edits gives an all-false
attention row, a softmax over all -inf, and NaN that poisons the whole batch. Corpus A
contains genuinely empty changesets (2 in a 500-row sample). They receive a single neutral
"null edit" at slot 0 - all-zero features, which after normalisation is exactly the
training mean - so the model sees "nothing here" rather than crashing.

Branch ablations remove *columns*, not just layers: switching off B1 must stop geometry
reaching the model through the edit trunk and the tabular features too, or the ablation
measures nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from mapguard.training.dataset import (
    EDIT_FEATURES,
    MAX_EDITS,
    MAX_USER_STEPS,
    TABULAR_FEATURES,
    TEXT_LEN,
    VOCAB_SIZE,
    build_batch,
)

GEOMETRY_EDIT_COLUMNS = (
    "centroid_shift_m", "hausdorff_m", "d_length_m", "d_area_m2",
    "length_ratio", "area_ratio", "d_points", "d_refs", "geom_route",
)
TEXT_TABULAR_COLUMNS = ("text_chars", "n_text_edits")
USER_FEATURES = ("num_changes", "gap_days", "comment_length", "bbox_diag_deg")


def is_geometry_tabular(col: str) -> bool:
    return col.startswith("geom_") or col in ("bbox_lat_span", "bbox_lon_span")


@dataclass
class Prepared:
    arrays: dict                      # edit_features, edit_mask, text, user_sequence, user_mask, tabular
    y: np.ndarray
    cids: np.ndarray
    uids: np.ndarray
    created_at: np.ndarray
    edit_columns: tuple
    tabular_columns: tuple
    n_null_edits_inserted: int = 0
    notes: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    def shapes(self) -> dict:
        a = self.arrays
        return {
            "max_edits": int(a["edit_features"].shape[1]),
            "edit_features": int(a["edit_features"].shape[2]),
            "text_len": int(a["text"].shape[1]),
            "vocab_size": int(VOCAB_SIZE),
            "user_steps": int(a["user_sequence"].shape[1]),
            "user_features": int(a["user_sequence"].shape[2]),
            "tabular_features": int(a["tabular"].shape[1]),
        }

    def subset(self, rows: np.ndarray) -> "Prepared":
        return replace(
            self,
            arrays={k: v[rows] for k, v in self.arrays.items()},
            y=self.y[rows], cids=self.cids[rows], uids=self.uids[rows],
            created_at=self.created_at[rows],
        )


def _clean_uids(raw) -> np.ndarray:
    """uid as int, or None for a deleted/anonymous account.

    Parquet turns a nullable integer column into floats with NaN, and NaN is not equal to
    itself - so a NaN uid silently fails every "same user" test, which is exactly the test
    the user-disjoint split and the pretraining exclusion depend on.
    """
    import pandas as pd

    out = np.empty(len(raw), dtype=object)
    for i, u in enumerate(raw):
        if u is None or pd.isna(u):         # covers None, float NaN and pandas.NA
            out[i] = None
        else:
            out[i] = int(u)
    return out


def assemble(store, *, max_edits: int = MAX_EDITS, user_steps: int = MAX_USER_STEPS) -> Prepared:
    """Raw padded arrays from the feature store. No normalisation yet."""
    b = build_batch(store.changesets, store.edits, store.user_seq,
                    max_edits=max_edits, max_user_steps=user_steps)
    cs = store.changesets
    return Prepared(
        arrays={
            "edit_features": b.edit_features,
            "edit_mask": b.edit_mask,
            "text": b.text,
            "user_sequence": b.user_sequence,
            "user_mask": b.user_mask,
            "tabular": b.tabular,
        },
        y=b.y.astype("float32"),
        cids=np.asarray(b.cids),
        uids=_clean_uids(cs["uid"].to_numpy()),
        created_at=np.asarray(cs["created_at"].astype(str).to_numpy()),
        edit_columns=tuple(EDIT_FEATURES),
        tabular_columns=tuple(TABULAR_FEATURES),
    )


def select_branches(prep: Prepared, branches) -> Prepared:
    """Drop the columns that belong to switched-off branches.

    B1 off  - geometry columns leave the edit trunk and the tabular features
    B2 off  - text-derived tabular counts leave too (the text input itself is simply absent)
    B3/B4/B5 are whole inputs; the model builder omits them, no columns to drop here
    """
    branches = set(branches)
    edit_keep = [c for c in prep.edit_columns
                 if "B1" in branches or c not in GEOMETRY_EDIT_COLUMNS]
    tab_keep = [c for c in prep.tabular_columns
                if ("B1" in branches or not is_geometry_tabular(c))
                and ("B2" in branches or c not in TEXT_TABULAR_COLUMNS)]
    if not edit_keep:
        raise ValueError("no edit columns left after branch selection")

    e_idx = [prep.edit_columns.index(c) for c in edit_keep]
    t_idx = [prep.tabular_columns.index(c) for c in tab_keep]
    arrays = dict(prep.arrays)
    arrays["edit_features"] = arrays["edit_features"][:, :, e_idx]
    arrays["tabular"] = arrays["tabular"][:, t_idx]
    return replace(prep, arrays=arrays, edit_columns=tuple(edit_keep),
                   tabular_columns=tuple(tab_keep))


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def ensure_nonempty(edit_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Give every fully-masked row one neutral null edit at slot 0. Returns (mask, count)."""
    mask = edit_mask.copy()
    empty = mask.sum(axis=1) == 0
    mask[empty, 0] = 1.0
    return mask, int(empty.sum())


class Normaliser:
    """signed log1p then z-score, fitted on training rows and real positions only."""

    GROUPS = ("edit", "user", "tabular")

    def __init__(self):
        self.stats: dict[str, dict] = {}

    # -- fitting ---------------------------------------------------------------
    def fit(self, prep: Prepared, train_rows: np.ndarray, groups=GROUPS) -> "Normaliser":
        a = prep.arrays
        if "edit" in groups:
            ef = signed_log1p(a["edit_features"][train_rows])
            real = a["edit_mask"][train_rows] > 0
            self.stats["edit"] = self._moments(ef[real], prep.edit_columns)
        if "user" in groups:
            us = signed_log1p(a["user_sequence"][train_rows])
            real = a["user_mask"][train_rows] > 0
            flat = us[real] if real.any() else np.zeros((1, us.shape[-1]), "float32")
            self.stats["user"] = self._moments(flat, USER_FEATURES)
        if "tabular" in groups:
            tb = signed_log1p(a["tabular"][train_rows])
            self.stats["tabular"] = self._moments(tb, prep.tabular_columns)
        return self

    @staticmethod
    def _moments(x: np.ndarray, columns) -> dict:
        x = np.nan_to_num(x.astype("float64"), nan=0.0, posinf=0.0, neginf=0.0)
        mu = x.mean(axis=0) if len(x) else np.zeros(x.shape[-1])
        sd = x.std(axis=0) if len(x) else np.ones(x.shape[-1])
        sd = np.where(sd < 1e-8, 1.0, sd)
        return {"columns": list(columns), "mean": mu.tolist(), "std": sd.tolist()}

    # -- applying ----------------------------------------------------------------
    def apply(self, prep: Prepared) -> Prepared:
        a = dict(prep.arrays)

        ef = self._z(signed_log1p(a["edit_features"]), "edit", prep.edit_columns)
        ef *= a["edit_mask"][:, :, None]                     # padding back to exactly 0
        mask, inserted = ensure_nonempty(a["edit_mask"])
        a["edit_features"], a["edit_mask"] = ef.astype("float32"), mask.astype("float32")

        us = self._z(signed_log1p(a["user_sequence"]), "user", USER_FEATURES)
        a["user_sequence"] = (us * a["user_mask"][:, :, None]).astype("float32")
        # Same guard for B4. A fully masked row is legal, but on a GPU it knocks the whole
        # batch off the cuDNN kernel onto the generic RNN loop, which is many times slower.
        # The null step is all zeros; user_n_prior and user_history_known carry the fact.
        umask, users_inserted = ensure_nonempty(a["user_mask"])
        a["user_mask"] = umask.astype("float32")

        a["tabular"] = self._z(signed_log1p(a["tabular"]), "tabular",
                               prep.tabular_columns).astype("float32")

        for k in ("edit_features", "user_sequence", "tabular"):
            if not np.isfinite(a[k]).all():
                raise ValueError(f"non-finite values in {k} after normalisation")

        notes = list(prep.notes)
        if inserted:
            notes.append(f"{inserted} changesets had no edits; a neutral null edit was inserted")
        if users_inserted:
            notes.append(f"{users_inserted} changesets had no user history; a null history step "
                         "was inserted")
        return replace(prep, arrays=a, n_null_edits_inserted=inserted, notes=notes)

    def _z(self, x: np.ndarray, group: str, columns) -> np.ndarray:
        s = self.stats.get(group)
        if s is None:
            raise ValueError(f"normaliser has no '{group}' statistics - fit or load them first")
        if list(columns) != s["columns"]:
            raise ValueError(
                f"{group} columns differ from the ones the normaliser was fitted on.\n"
                f"  fitted : {s['columns']}\n  now    : {list(columns)}\n"
                "A model cannot be evaluated or fine-tuned on a different feature layout."
            )
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return (x - np.asarray(s["mean"])) / np.asarray(s["std"])

    # -- persistence -------------------------------------------------------------
    def save(self, path: Path) -> Path:
        path.write_text(json.dumps(self.stats, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path, groups=GROUPS) -> "Normaliser":
        n = cls()
        stats = json.loads(Path(path).read_text(encoding="utf-8"))
        n.stats = {g: stats[g] for g in groups if g in stats}
        return n


# Columns that only the adiff route can fill. Corpus A has adiff on a subset; corpora M, N
# and R have none. Training on A and testing on R with these columns in would show the model
# a feature shift the training data never had, so `ovid_modern` drops them.
ADIFF_ONLY_COLUMNS = (
    "d_length_m", "d_area_m2", "length_ratio", "area_ratio", "d_points",
    "geom_frac_adiff", "geom_total_d_area_m2", "geom_max_abs_d_area_m2", "geom_max_area_ratio",
    "geom_total_d_points",
)


def exclude_columns(prep: Prepared, names) -> Prepared:
    """Remove named edit/tabular columns. Unknown names are ignored so one list serves every
    branch set."""
    names = set(names or ())
    if not names:
        return prep
    edit_keep = [c for c in prep.edit_columns if c not in names]
    tab_keep = [c for c in prep.tabular_columns if c not in names]
    arrays = dict(prep.arrays)
    arrays["edit_features"] = arrays["edit_features"][:, :, [prep.edit_columns.index(c) for c in edit_keep]]
    arrays["tabular"] = arrays["tabular"][:, [prep.tabular_columns.index(c) for c in tab_keep]]
    return replace(prep, arrays=arrays, edit_columns=tuple(edit_keep), tabular_columns=tuple(tab_keep))


def prepare(store, cfg: dict) -> Prepared:
    """assemble + branch selection + column exclusion. Normalisation comes later, once
    splits exist. `model.exclude_columns` takes a list of names, or "adiff_only"."""
    m = cfg.get("model", {})
    prep = assemble(store,
                    max_edits=int(m.get("max_edits", MAX_EDITS)),
                    user_steps=int(m.get("user_steps", MAX_USER_STEPS)))
    prep = select_branches(prep, m.get("branches", ("B1", "B2", "B3", "B4", "B5")))
    excl = m.get("exclude_columns")
    if excl == "adiff_only":
        excl = ADIFF_ONLY_COLUMNS
    return exclude_columns(prep, excl)


def text_len() -> int:
    return TEXT_LEN
