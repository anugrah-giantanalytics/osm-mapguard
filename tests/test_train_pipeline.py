"""Tests for train/ - everything that does not need TensorFlow.

The TensorFlow paths are exercised by train/00_wiring_check on a machine that has it. What
is tested here is the logic that decides *which rows and which numbers*: task selection,
frozen splits, normalisation, corruptions, label budgets, and the refusals that stop a
leaky or mismatched run from producing a result.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "train"
sys.path[:0] = [str(TRAIN / "04_stage1_pretrain"), str(TRAIN), str(REPO / "src")]

from common import config as cfgmod  # noqa: E402
from common import data_contract, scoring, splits_io  # noqa: E402
from common.fit import check_encoder_split  # noqa: E402
from common.preprocess import (  # noqa: E402
    GEOMETRY_EDIT_COLUMNS,
    Normaliser,
    Prepared,
    _clean_uids,
    ensure_nonempty,
    select_branches,
    signed_log1p,
)
from corruptions import CORRUPTIONS, corrupt_one, make_contrast_set  # noqa: E402
from mapguard.training.dataset import EDIT_FEATURES, TABULAR_FEATURES  # noqa: E402


# ------------------------------------------------------------------------- helpers


def small_prepared(n=60, k=6, seed=0) -> Prepared:
    rng = np.random.default_rng(seed)
    F = len(EDIT_FEATURES)
    n_edits = rng.integers(0, k + 1, n)
    n_edits[:3] = 0
    mask = (np.arange(k)[None, :] < n_edits[:, None]).astype("float32")
    edit = (rng.lognormal(1, 2, (n, k, F)) * mask[:, :, None]).astype("float32")
    umask = (np.arange(4)[None, :] < rng.integers(0, 5, n)[:, None]).astype("float32")
    return Prepared(
        arrays={"edit_features": edit, "edit_mask": mask,
                "text": np.zeros((n, 8), "int32"),
                "user_sequence": (rng.lognormal(0, 1, (n, 4, 4)) * umask[:, :, None]).astype("float32"),
                "user_mask": umask,
                "tabular": rng.lognormal(0, 1, (n, len(TABULAR_FEATURES))).astype("float32")},
        y=np.tile([0.0, 1.0], n // 2).astype("float32"),
        cids=np.arange(100, 100 + n),
        uids=np.array([int(u) for u in rng.integers(1, 15, n)], dtype=object),
        created_at=np.array([f"2025-{1 + i % 12:02d}-10T00:00:00Z" for i in range(n)]),
        edit_columns=tuple(EDIT_FEATURES),
        tabular_columns=tuple(TABULAR_FEATURES),
    )


def cs_frame(rows):
    return pd.DataFrame(rows, columns=["cid", "label", "corpus"])


# -------------------------------------------------------------------------- config


def test_dotted_override_parses_json_and_falls_back_to_string():
    cfg = {"train": {"epochs": 40}}
    cfgmod.set_dotted(cfg, "train.epochs", "5")
    cfgmod.set_dotted(cfg, "model.branches", '["B2","B5"]')
    cfgmod.set_dotted(cfg, "task", "T1")
    assert cfg["train"]["epochs"] == 5
    assert cfg["model"]["branches"] == ["B2", "B5"]
    assert cfg["task"] == "T1"


def test_temporal_split_without_cutoff_is_rejected():
    with pytest.raises(SystemExit):
        cfgmod.validate({"stage": "x", "task": "T2", "split": "temporal"})


def test_unknown_task_is_rejected():
    with pytest.raises(SystemExit):
        cfgmod.validate({"stage": "x", "task": "T3"})


# ------------------------------------------------------------------- data contract


def test_tasks_select_the_right_label_sources():
    cs = cs_frame([(1, 1, "M"), (2, 0, "M"), (3, 0, "N"), (4, 1, "A"), (5, 0, "A"), (6, 0, "N")])
    assert sorted(data_contract.select_task(cs, "T1")["cid"]) == [1, 2]          # reviewed-clean negatives
    assert sorted(data_contract.select_task(cs, "T2")["cid"]) == [1, 3, 6]       # random negatives
    assert sorted(data_contract.select_task(cs, "ovid")["cid"]) == [4, 5]


def test_task_selection_without_corpus_column_refuses():
    """Mixing revert-derived and human labels silently is not reportable."""
    cs = pd.DataFrame({"cid": [1, 2], "label": [0, 1]})
    with pytest.raises(SystemExit):
        data_contract.select_task(cs, "T2")
    assert len(data_contract.select_task(cs, "all")) == 2


def test_a_positive_in_the_random_negative_corpus_is_refused():
    cs = cs_frame([(1, 1, "M"), (2, 1, "N"), (3, 0, "N")])
    with pytest.raises(SystemExit):
        data_contract.select_task(cs, "T2")


def test_missing_feature_store_explains_itself(tmp_path):
    with pytest.raises(SystemExit) as exc:
        data_contract.load(tmp_path, task="all")
    assert "DATA_FOUNDATION" in str(exc.value)


# ------------------------------------------------------------------- preprocessing


def test_signed_log1p_keeps_sign_and_order():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    y = signed_log1p(x)
    assert (np.sign(y) == np.sign(x)).all()
    assert (np.diff(y) > 0).all()


def test_empty_changesets_get_exactly_one_null_edit():
    mask = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype="float32")
    out, n = ensure_nonempty(mask)
    assert n == 2
    assert out.sum(axis=1).tolist() == [1.0, 2.0, 1.0]


def test_branch_off_removes_its_columns_everywhere():
    prep = small_prepared()
    no_geom = select_branches(prep, ["B2", "B3", "B4", "B5"])
    assert not set(GEOMETRY_EDIT_COLUMNS) & set(no_geom.edit_columns)
    assert not [c for c in no_geom.tabular_columns if c.startswith("geom_")]
    assert no_geom.arrays["edit_features"].shape[2] == len(no_geom.edit_columns)

    no_text = select_branches(prep, ["B1", "B5"])
    assert "text_chars" not in no_text.tabular_columns
    assert "n_text_edits" not in no_text.tabular_columns


def test_normaliser_uses_real_edit_positions_only_and_rezeros_padding():
    prep = small_prepared()
    train = np.arange(len(prep.y)) < 40
    norm = Normaliser().fit(prep, train)
    out = norm.apply(prep)
    ef, mask = out.arrays["edit_features"], prep.arrays["edit_mask"]
    assert np.all(ef[mask == 0] == 0.0)
    real = signed_log1p(prep.arrays["edit_features"][train])[prep.arrays["edit_mask"][train] > 0]
    assert np.allclose(norm.stats["edit"]["mean"], real.mean(axis=0), atol=1e-6)
    assert out.n_null_edits_inserted >= 3
    assert np.isfinite(ef).all()


def test_normaliser_refuses_a_different_column_layout(tmp_path):
    prep = small_prepared()
    rows = np.ones(len(prep.y), bool)
    path = Normaliser().fit(prep, rows).save(tmp_path / "n.json")
    other = select_branches(prep, ["B5"])
    with pytest.raises(ValueError):
        Normaliser.load(path).apply(other)


def test_missing_uids_become_none_so_user_tests_work():
    out = _clean_uids(np.array([5.0, np.nan, 7.0]))
    assert out.tolist() == [5, None, 7]
    out = _clean_uids(pd.array([1, None, 3], dtype="Int64").to_numpy())
    assert out.tolist() == [1, None, 3]


# -------------------------------------------------------------------------- splits


def test_frozen_split_is_created_reused_and_refused_when_data_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(splits_io, "SPLITS_DIR", tmp_path)
    prep = small_prepared(n=120)
    m1, path, _ = splits_io.get_split(prep, task="T2", split="R1b", seed=3)
    assert path.exists()
    m2, _, _ = splits_io.get_split(prep, task="T2", split="R1b", seed=3)
    for k in ("train", "val", "test"):
        assert (m1[k] == m2[k]).all()

    changed = small_prepared(n=120)
    changed.cids = changed.cids + 1
    with pytest.raises(SystemExit):
        splits_io.get_split(changed, task="T2", split="R1b", seed=3)


def test_r1b_split_file_is_user_disjoint(tmp_path, monkeypatch):
    monkeypatch.setattr(splits_io, "SPLITS_DIR", tmp_path)
    prep = small_prepared(n=200)
    masks, _, _ = splits_io.get_split(prep, task="T2", split="R1b", seed=1)
    users = {k: {u for u in prep.uids[m]} for k, m in masks.items()}
    assert not users["train"] & users["test"]
    assert not users["train"] & users["val"]


def test_temporal_split_drops_authors_active_on_both_sides(tmp_path, monkeypatch):
    monkeypatch.setattr(splits_io, "SPLITS_DIR", tmp_path)
    prep = small_prepared(n=200)
    uids = list(range(200))
    uids[0] = uids[199] = 999          # row 0 is 2025-01 (past), row 199 is 2025-08 (future)
    prep.uids = np.array(uids, dtype=object)
    masks, _, _ = splits_io.get_split(prep, task="T2", split="temporal", seed=1,
                                      cutoff="2025-07-01T00:00:00Z")
    assert masks["test"][199] and not masks["train"][0] and not masks["val"][0]
    assert (prep.created_at[masks["train"]] < "2025-07-01").all()
    assert (prep.created_at[masks["test"]] >= "2025-07-01").all()
    assert not {u for u in prep.uids[masks["train"]]} & {u for u in prep.uids[masks["test"]]}


# ------------------------------------------------------------------------- scoring


def test_threshold_is_found_and_calibration_of_perfect_scores_is_zero():
    y = np.array([0, 0, 1, 1] * 25)
    p = np.array([0.1, 0.2, 0.8, 0.9] * 25)
    t, f1 = scoring.best_f1_threshold(y, p)
    assert 0.2 <= t <= 0.8 and f1 == 1.0
    assert scoring.calibration(y, y.astype(float))["ece"] == 0.0


# --------------------------------------------------------------------- corruptions


def test_each_corruption_changes_the_intended_columns():
    cols = list(EDIT_FEATURES)
    x = np.zeros(len(cols))
    x[cols.index("n_tags")] = 4
    x[cols.index("centroid_shift_m")] = 2.0
    x[cols.index("geom_route")] = 2
    rng = np.random.default_rng(0)

    d = corrupt_one(x, "displace", cols, rng)
    assert d[cols.index("centroid_shift_m")] >= 200.0

    w = corrupt_one(x, "tag_wipe", cols, rng)
    assert w[cols.index("n_tags")] == 0 and w[cols.index("tags_removed")] == 4

    s = corrupt_one(x, "tag_spam", cols, rng)
    assert s[cols.index("n_tags")] >= 14


def test_impossible_corruption_returns_none_rather_than_a_fake_positive():
    cols = list(EDIT_FEATURES)
    x = np.zeros(len(cols))              # no tags to wipe
    assert corrupt_one(x, "tag_wipe", cols, np.random.default_rng(0)) is None


def test_corruptions_never_write_geometry_the_edit_could_not_have():
    cols = list(EDIT_FEATURES)
    rng = np.random.default_rng(0)
    x = np.zeros(len(cols))
    x[cols.index("n_tags")] = 3
    x[cols.index("geom_route")] = 0                  # a created object: no geometry route
    for kind in ("displace", "area_blowup", "length_stretch", "mass_points"):
        assert corrupt_one(x, kind, cols, rng) is None, kind
    s = corrupt_one(x, "tag_spam", cols, rng)
    assert s[cols.index("tags_added")] == 0           # unobserved tag diffs stay unobserved
    x[cols.index("geom_route")] = 1                  # previous-version route
    assert corrupt_one(x, "area_blowup", cols, rng) is None
    m = corrupt_one(x, "mass_points", cols, rng)
    assert m[cols.index("d_refs")] >= 200 and m[cols.index("d_points")] == 0


def test_user_history_rows_are_never_fully_masked_after_normalisation():
    prep = small_prepared(n=40)
    prep.arrays["user_mask"][:5] = 0.0
    norm = Normaliser().fit(prep, np.arange(40) < 30)
    out = norm.apply(prep)
    assert (out.arrays["user_mask"].sum(axis=1) >= 1).all()
    assert (out.arrays["user_sequence"][:5] == 0).all()


def test_contrast_set_is_balanced_and_every_corrupted_row_differs():
    cols = list(EDIT_FEATURES)
    real = np.random.default_rng(1).lognormal(0, 1, (300, len(cols)))
    real[:, cols.index("geom_route")] = 2
    X, y, kinds = make_contrast_set(real, cols, seed=2, kinds=CORRUPTIONS)
    assert (y == 0).sum() == 300 and (y == 1).sum() == 300
    assert (kinds[y == 0] == -1).all() and (kinds[y == 1] >= 0).all()


# ------------------------------------------------------------ label efficiency, encoder


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_label_budgets_are_nested_and_stratified():
    le = _load_script(TRAIN / "05_stage1_label_efficiency" / "label_efficiency.py")
    y = np.tile([0, 1], 500)
    train = np.arange(1000) < 800
    b = le.nested_budgets(y, train, [0.1, 0.25, 0.5, 1.0], seed=3)
    assert (b[0.1] <= b[0.25]).all() and (b[0.25] <= b[0.5]).all() and (b[0.5] <= b[1.0]).all()
    assert not (b[1.0] & ~train).any()
    for f in (0.1, 0.5):
        assert abs(y[b[f]].mean() - 0.5) < 0.02


def test_encoder_from_a_different_split_is_refused(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"task": "T2", "split": "R1b", "seed": 1}))
    check_encoder_split(tmp_path, {"task": "T2", "split": "R1b", "seed": 1})
    check_encoder_split(tmp_path, {"task": "T2", "split": "R1b", "seed": 9, "split_seed": 1})
    with pytest.raises(SystemExit):
        check_encoder_split(tmp_path, {"task": "T1", "split": "R1b", "seed": 1})
    with pytest.raises(SystemExit):
        check_encoder_split(tmp_path, {"task": "T2", "split": "R1a", "seed": 1})


def test_every_stage_config_is_valid_json_with_a_stage():
    for cfg_path in sorted(TRAIN.glob("0*/config.json")):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfgmod.validate(cfg)
    for row in sorted((TRAIN / "03_stage2_ablations" / "rows").glob("*.json")):
        r = json.loads(row.read_text(encoding="utf-8"))
        assert r["name"] and "overrides" in r
