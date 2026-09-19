"""Padding and masking alignment.

If the mask does not line up with the padding, the model attends to zeros. It trains, it
scores, and it is wrong - and the failure looks like "geometry did not help" rather than
like a bug. Gate 1d checks the model honours a mask; these tests check the mask is right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mapguard.training.dataset import (
    CHAR_TO_ID,
    EDIT_FEATURES,
    TABULAR_FEATURES,
    VOCAB_SIZE,
    build_batch,
    encode_text,
    standardise,
)


def make_frames(edit_counts=(3, 0, 7), seq_counts=(2, 5, 0)):
    cids = [100, 200, 300][: len(edit_counts)]
    changesets = pd.DataFrame({
        "cid": cids,
        "label": [1, 0, 1][: len(cids)],
        "uid": [11, 22, None][: len(cids)],
        "text_blob": ["Church Street", "", "AAAA xxxx"][: len(cids)],
        **{c: [float(i + 1) for i in range(len(cids))] for c in TABULAR_FEATURES},
    })
    rows = []
    for cid, k in zip(cids, edit_counts):
        for j in range(k):
            rows.append({"cid": cid, **{c: float(j + 1) for c in EDIT_FEATURES}})
    edits = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["cid", *EDIT_FEATURES])

    srows = []
    for cid, k in zip(cids, seq_counts):
        for j in range(k):
            srows.append({"cid": cid, "step": j, "num_changes": float(j),
                          "gap_days": 1.0, "comment_length": 2.0, "bbox_diag_deg": 0.1})
    user_seq = pd.DataFrame(srows) if srows else pd.DataFrame(
        columns=["cid", "step", "num_changes", "gap_days", "comment_length", "bbox_diag_deg"])
    return changesets, edits, user_seq


def test_mask_marks_exactly_the_real_edits():
    b = build_batch(*make_frames(edit_counts=(3, 0, 7)))
    assert list(b.edit_mask.sum(axis=1)) == [3.0, 0.0, 7.0]


def test_padded_edit_rows_are_all_zero():
    """A padded slot must carry no signal at all - masking is the second line of defence,
    not the first."""
    b = build_batch(*make_frames(edit_counts=(3, 0, 7)))
    for i, k in enumerate([3, 0, 7]):
        assert np.all(b.edit_features[i, k:] == 0.0)
        if k:
            assert np.any(b.edit_features[i, :k] != 0.0)


def test_edits_are_slotted_per_changeset_not_by_table_row():
    """Edits arrive interleaved across changesets; each must start at slot 0 of its own
    row, or every sequence is offset by whatever came before it in the table."""
    changesets, edits, user_seq = make_frames(edit_counts=(2, 2, 2))
    shuffled = edits.sample(frac=1.0, random_state=0).reset_index(drop=True)
    b = build_batch(changesets, shuffled, user_seq)
    assert list(b.edit_mask.sum(axis=1)) == [2.0, 2.0, 2.0]
    assert np.all(b.edit_mask[:, :2] == 1.0)
    assert np.all(b.edit_mask[:, 2:] == 0.0)


def test_more_edits_than_the_cap_are_truncated_not_wrapped():
    changesets, _, user_seq = make_frames(edit_counts=(1,))
    changesets = changesets.iloc[:1]
    rows = [{"cid": 100, **{c: 1.0 for c in EDIT_FEATURES}} for _ in range(500)]
    b = build_batch(changesets, pd.DataFrame(rows), user_seq, max_edits=8)
    assert b.edit_mask.sum() == 8.0
    assert b.edit_features.shape[1] == 8


def test_user_sequence_uses_the_step_column():
    b = build_batch(*make_frames(seq_counts=(2, 5, 0)))
    assert list(b.user_mask.sum(axis=1)) == [2.0, 5.0, 0.0]


def test_row_order_matches_the_changesets_table():
    changesets, edits, user_seq = make_frames()
    b = build_batch(changesets, edits, user_seq)
    assert list(b.cids) == list(changesets["cid"])
    assert list(b.y) == [1.0, 0.0, 1.0]


def test_select_keeps_every_array_aligned():
    b = build_batch(*make_frames())
    sub = b.select(np.array([True, False, True]))
    assert list(sub.cids) == [100, 300]
    assert list(sub.edit_mask.sum(axis=1)) == [3.0, 7.0]
    assert len(sub) == 2


def test_nan_and_inf_never_reach_the_model():
    """One NaN poisons every gradient in the batch. Better caught here than as a loss of
    nan at epoch 3."""
    changesets, edits, user_seq = make_frames()
    changesets.loc[0, "account_age_days"] = np.nan
    edits.loc[0, "area_ratio"] = np.inf
    b = build_batch(changesets, edits, user_seq)
    assert np.isfinite(b.tabular).all()
    assert np.isfinite(b.edit_features).all()


def test_missing_optional_tables_still_produce_a_batch():
    """Geometry may not exist yet for every changeset; that is a coverage number, not a
    crash."""
    changesets, _, _ = make_frames()
    b = build_batch(changesets, None, None)
    assert b.edit_mask.sum() == 0.0
    assert b.user_mask.sum() == 0.0
    assert b.tabular.shape == (3, len(TABULAR_FEATURES))


def test_text_encoding_is_padded_and_bounded():
    v = encode_text("Church Street")
    assert v.shape == (256,)
    assert v[0] == CHAR_TO_ID["c"]
    assert v[13:].sum() == 0
    assert v.max() < VOCAB_SIZE


def test_unknown_characters_map_to_one_not_to_nothing():
    """A name of entirely unusual symbols must not arrive as an empty string - that is
    exactly the kind of thing worth flagging."""
    v = encode_text("你好世界")
    assert (v[:4] == 1).all()


def test_empty_text_is_all_padding():
    assert encode_text(None).sum() == 0
    assert encode_text("").sum() == 0


def test_standardise_uses_training_statistics_only():
    """Fitting the scaler on everything leaks the test distribution into training. Small
    leak, entirely avoidable."""
    train = np.array([[0.0], [2.0]])
    test = np.array([[10.0]])
    tr, te = standardise(train, test)
    assert tr.mean() == 0.0
    assert te[0, 0] == 9.0        # (10 - 1) / 1, using train mean 1 and train std 1


def test_standardise_survives_a_constant_column():
    train = np.array([[5.0, 1.0], [5.0, 3.0]])
    tr, = standardise(train)
    assert np.isfinite(tr).all()


def test_text_input_puts_the_comment_first_and_caps_it():
    from mapguard.training.dataset import COMMENT_CHARS, text_input

    assert text_input("fix road", "Main Street") == "fix road | Main Street"
    assert text_input(None, "Main Street") == "Main Street"
    assert text_input(float("nan"), None) == ""
    long = text_input("x" * 500, "Main Street")
    assert long.startswith("x" * COMMENT_CHARS + " | ") and long.endswith("Main Street")
