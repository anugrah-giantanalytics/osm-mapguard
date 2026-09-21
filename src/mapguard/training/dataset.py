"""Parquet tables -> padded arrays for the five branches.

This is where masking bugs are born. Every changeset holds a different number of edits and
every user a different length of history, so both get padded to a fixed width - and if the
mask that says which rows are real does not line up exactly with the padding, the model
attends to zeros. It trains, it scores, and it is wrong, and the failure presents as
"geometry did not help". Gate 1d checks the model honours the mask; this module makes sure
the mask is correct in the first place.

Outputs, keyed to the branches in PROJECT_PLAN section 5.3:

    edit_features  (N, K, F)   B1/B3  per-edit numeric, zero-padded
    edit_mask      (N, K)      -      1 = real edit, 0 = padding
    edit_tokens    (N, K, T)   B6     hashed edit tokens (what the edit did), 0 = padding
    edit_text      (N, K, L)   B6     characters of "old text>new text" per edit
    text           (N, C)      B2     character codes, zero-padded
    user_sequence  (N, S, 4)   B4     point-in-time history, zero-padded
    user_mask      (N, S)      -
    tabular        (N, T)      B5     changeset-level features

Ordering is the contract: rows come back in the order of the `changesets` table, and every
array shares that order. Nothing here sorts or reindexes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_EDITS = 256          # K, matches SAMPLE_K in features.build_store
MAX_USER_STEPS = 200     # S, matches SEQUENCE_LENGTH in features.user_history
TEXT_LEN = 256           # C, character sequence for the B2 char-CNN
EDIT_TEXT_LEN = 24       # L, per-edit "old>new" free text for B6
from mapguard.features.edit_tokens import TOKEN_BUCKETS, TOKENS_PER_EDIT, encode_tokens  # noqa: E402

# Printable ASCII plus a catch-all. Deliberately small: the B2 branch is meant to read
# shapes of text - repetition, case, gibberish, symbol runs - not to model language.
_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 .,-_/'()&:;!?#*+"
CHAR_TO_ID = {c: i + 2 for i, c in enumerate(_ALPHABET)}   # 0 = pad, 1 = out-of-alphabet
VOCAB_SIZE = len(CHAR_TO_ID) + 2

EDIT_FEATURES = (
    "centroid_shift_m", "hausdorff_m", "d_length_m", "d_area_m2",
    "length_ratio", "area_ratio", "d_points", "n_tags", "n_refs",
    "version", "is_new_object", "tags_added", "tags_removed", "tags_changed",
    "d_refs", "geom_route", "days_since_prev_version", "prev_same_author",
)

# account_age_days is deliberately NOT a model input. It comes from the account as it is
# *today*, and accounts are deleted after vandalism: in Corpus A, 97 deleted accounts own
# 599 changesets, 83% of them damaging (measured 2026-09-18). "Account unknown" is therefore
# a label leak from the future. user_days_active_span gives the same prior from
# point-in-time history, which a deletion cannot touch.
TABULAR_FEATURES = (
    "n_edits", "n_create", "n_modify", "n_delete", "frac_delete",
    "n_node", "n_way", "n_relation", "max_version", "mean_version", "n_v1",
    "total_tags", "mean_tags", "distinct_keys", "n_text_edits",
    "sample_ratio", "comment_length", "text_chars",
    "bbox_lat_span", "bbox_lon_span",
    "frac_edits_prev_version", "user_history_known", "geom_frac_adiff",
    "geom_n_geom_actions", "geom_max_centroid_shift_m", "geom_p95_centroid_shift_m",
    "geom_max_hausdorff_m", "geom_max_abs_d_area_m2", "geom_max_area_ratio",
    "geom_total_d_points", "geom_n_deleted", "geom_n_created",
    "user_n_prior", "user_days_since_prev", "user_days_active_span",
    "user_changesets_per_active_day", "user_mean_prior_changes",
    "user_max_prior_changes", "user_distinct_editors", "user_frac_prior_with_comment",
    "user_max_bbox_diag_deg", "user_history_truncated", "user_active_weeks",
    "n_prev_authors", "imagery_used",
)


COMMENT_CHARS = 120      # of TEXT_LEN; the rest is free-text tag values


def text_input(comment, text_blob) -> str:
    """B2 reads what the author typed: the changeset comment, then free-text tag values.

    Both are written at edit time, so neither carries future information. The comment comes
    first and is capped, so a long comment cannot crowd the tag text (`name=...`) out of the
    TEXT_LEN window. Checked 2026-09-19: reviewer or revert wording appears in <1% of
    comments in either class, so the comment does not echo the label.
    """
    c = "" if comment is None or comment != comment else str(comment)[:COMMENT_CHARS]
    b = "" if text_blob is None or text_blob != text_blob else str(text_blob)
    if c and b:
        return f"{c} | {b}"
    return c or b


def encode_text(s: str | None, length: int = TEXT_LEN) -> np.ndarray:
    """Characters to ids, lowercased, zero-padded.

    Case is folded because SHOUTING is captured better by a separate feature than by
    doubling the alphabet, and an out-of-alphabet character maps to 1 rather than being
    dropped - a name made entirely of unusual symbols should not arrive as an empty
    string, since that is exactly the kind of thing worth flagging.
    """
    out = np.zeros(length, dtype="int32")
    if not s:
        return out
    for i, ch in enumerate(str(s).lower()[:length]):
        out[i] = CHAR_TO_ID.get(ch, 1)
    return out


@dataclass
class Batch:
    """Model inputs plus the labels and ids, all sharing one row order."""

    cids: np.ndarray
    y: np.ndarray
    uids: np.ndarray
    edit_features: np.ndarray
    edit_mask: np.ndarray
    edit_tokens: np.ndarray
    edit_text: np.ndarray
    text: np.ndarray
    user_sequence: np.ndarray
    user_mask: np.ndarray
    tabular: np.ndarray

    def inputs(self) -> dict[str, np.ndarray]:
        return {
            "edit_features": self.edit_features,
            "edit_mask": self.edit_mask,
            "edit_tokens": self.edit_tokens,
            "edit_text": self.edit_text,
            "text": self.text,
            "user_sequence": self.user_sequence,
            "user_mask": self.user_mask,
            "tabular": self.tabular,
        }

    def __len__(self) -> int:
        return len(self.cids)

    def select(self, mask: np.ndarray) -> "Batch":
        """Subset by a boolean mask, keeping every array aligned."""
        return Batch(
            cids=self.cids[mask], y=self.y[mask], uids=self.uids[mask],
            edit_features=self.edit_features[mask], edit_mask=self.edit_mask[mask],
            edit_tokens=self.edit_tokens[mask], edit_text=self.edit_text[mask],
            text=self.text[mask], user_sequence=self.user_sequence[mask],
            user_mask=self.user_mask[mask], tabular=self.tabular[mask],
        )


def _clean(a: np.ndarray) -> np.ndarray:
    """NaN and inf to zero.

    Ratios divide by a previous value that can be zero, and `account_age_days` is None
    for a deleted account. A single NaN silently poisons every gradient in the batch, so
    it is dealt with here rather than discovered as a loss of `nan` at epoch 3.
    """
    return np.nan_to_num(a.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)


def build_batch(
    changesets,
    edits,
    user_seq,
    *,
    max_edits: int = MAX_EDITS,
    max_user_steps: int = MAX_USER_STEPS,
) -> Batch:
    """Assemble padded arrays from the three Parquet tables (as pandas frames)."""
    cids = changesets["cid"].to_numpy()
    index = {int(c): i for i, c in enumerate(cids)}
    n = len(cids)

    # --- B1/B3: per-edit -----------------------------------------------------
    ef = np.zeros((n, max_edits, len(EDIT_FEATURES)), dtype="float32")
    em = np.zeros((n, max_edits), dtype="float32")
    et = np.zeros((n, max_edits, TOKENS_PER_EDIT), dtype="int32")
    ex = np.zeros((n, max_edits, EDIT_TEXT_LEN), dtype="int32")

    if edits is not None and len(edits):
        cols = [edits[c].to_numpy() if c in edits.columns else np.zeros(len(edits))
                for c in EDIT_FEATURES]
        block = _clean(np.column_stack(cols))
        edit_cids = edits["cid"].to_numpy()
        toks = edits["diff_tokens"].tolist() if "diff_tokens" in edits.columns else None
        new_t = edits["text"].tolist() if "text" in edits.columns else None
        old_t = edits["old_text"].tolist() if "old_text" in edits.columns else None
        # Position within the changeset, not the row number in the table.
        slot: dict[int, int] = {}
        for r in range(len(edits)):
            cid = int(edit_cids[r])
            i = index.get(cid)
            if i is None:
                continue
            k = slot.get(cid, 0)
            if k >= max_edits:
                continue
            ef[i, k] = block[r]
            em[i, k] = 1.0
            if toks is not None:
                et[i, k] = encode_tokens(toks[r])
            if new_t is not None or old_t is not None:
                o = (old_t[r] if old_t is not None else "") or ""
                w = (new_t[r] if new_t is not None else "") or ""
                if o or w:
                    ex[i, k] = encode_text(f"{o}>{w}", EDIT_TEXT_LEN)
            slot[cid] = k + 1

    # --- B4: user history ----------------------------------------------------
    us = np.zeros((n, max_user_steps, 4), dtype="float32")
    um = np.zeros((n, max_user_steps), dtype="float32")

    if user_seq is not None and len(user_seq):
        seq_cids = user_seq["cid"].to_numpy()
        steps = user_seq["step"].to_numpy()
        vals = _clean(np.column_stack([
            user_seq["num_changes"].to_numpy(),
            user_seq["gap_days"].to_numpy(),
            user_seq["comment_length"].to_numpy(),
            user_seq["bbox_diag_deg"].to_numpy(),
        ]))
        for r in range(len(user_seq)):
            i = index.get(int(seq_cids[r]))
            k = int(steps[r])
            if i is None or k >= max_user_steps:
                continue
            us[i, k] = vals[r]
            um[i, k] = 1.0

    # --- B2: text ------------------------------------------------------------
    blobs = changesets["text_blob"].tolist() if "text_blob" in changesets.columns else [""] * n
    comments = (changesets["comment"].tolist() if "comment" in changesets.columns
                else [""] * n)
    text = np.stack([encode_text(text_input(c, b)) for c, b in zip(comments, blobs)])

    # --- B5: tabular ---------------------------------------------------------
    tab_cols = []
    for c in TABULAR_FEATURES:
        if c in changesets.columns:
            tab_cols.append(changesets[c].to_numpy(dtype="float64", na_value=0.0)
                            if hasattr(changesets[c], "to_numpy") else np.zeros(n))
        else:
            tab_cols.append(np.zeros(n))
    tabular = _clean(np.column_stack(tab_cols))

    y = changesets["label"].to_numpy(dtype="float32", na_value=0.0) \
        if "label" in changesets.columns else np.zeros(n, "float32")
    uids = changesets["uid"].to_numpy() if "uid" in changesets.columns else np.full(n, None)

    return Batch(
        cids=cids, y=y, uids=uids,
        edit_features=ef, edit_mask=em, edit_tokens=et, edit_text=ex, text=text,
        user_sequence=us, user_mask=um, tabular=tabular,
    )


def standardise(train: np.ndarray, *others: np.ndarray):
    """Z-score using the **training** statistics only.

    Fitting the scaler on the whole dataset leaks test-set distribution into training.
    It is a small leak and it is entirely avoidable, so it is avoided.
    """
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return ((train - mu) / sd, *[(o - mu) / sd for o in others])
