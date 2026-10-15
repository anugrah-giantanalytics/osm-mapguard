"""The leakage test.

If `test_future_changesets_never_reach_the_features` fails, stop everything. It means the
model can see what happened to an account after the changeset it is scoring, and every
number produced downstream is meaningless. See docs/planning/PLAN_REVIEW.md finding F6.
"""

from __future__ import annotations

from mapguard.features.user_history import (
    HISTORY_LIMIT,
    PriorChangeset,
    build,
    parse_changeset_list,
    point_in_time,
)

CHANGESET_LIST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <changeset id="300" created_at="2016-06-01T10:00:00Z" closed_at="2016-06-01T10:05:00Z"
             uid="42" user="mapper" changes_count="12"
             min_lat="51.5" min_lon="-0.11" max_lat="51.6" max_lon="-0.09">
    <tag k="comment" v="add shops"/><tag k="created_by" v="iD 2.0"/>
  </changeset>
  <changeset id="200" created_at="2015-01-01T09:00:00Z" closed_at="2015-01-01T09:01:00Z"
             uid="42" user="mapper" changes_count="4"
             min_lat="51.5" min_lon="-0.1" max_lat="51.5" max_lon="-0.1">
    <tag k="created_by" v="JOSM/1.5"/>
  </changeset>
</osm>"""


def cs(cid, created_at, changes=5, editor="iD", comment_len=0, diag=0.01):
    return PriorChangeset(cid, created_at, changes, editor, comment_len, diag)


HISTORY = [
    cs(1, "2014-01-01T00:00:00Z", changes=10),
    cs(2, "2014-06-01T00:00:00Z", changes=20, editor="JOSM"),
    cs(3, "2015-01-01T00:00:00Z", changes=30, comment_len=15),
    # everything below is AFTER the 2015-06-01 target used in the tests
    cs(4, "2016-01-01T00:00:00Z", changes=9999, editor="Potlatch"),
    cs(5, "2017-01-01T00:00:00Z", changes=8888, editor="Rapid"),
]
TARGET = "2015-06-01T00:00:00Z"


def test_future_changesets_never_reach_the_features():
    """The one that matters. A 2017 event must not touch a 2015 prediction."""
    f = build(42, HISTORY, TARGET)
    assert f.n_prior == 3
    assert f.max_prior_changes == 30           # not 9999, not 8888
    assert f.total_prior_changes == 60
    assert "Potlatch" not in {e for e in ("Potlatch",)} or f.distinct_editors == 2
    assert f.distinct_editors == 2             # iD and JOSM only; Rapid came later


def test_point_in_time_is_strict_and_ordered():
    prior = point_in_time(HISTORY, TARGET)
    assert [p.cid for p in prior] == [1, 2, 3]


def test_a_changeset_at_exactly_the_target_time_is_excluded():
    """Strictly before. An edit at the same instant is not prior information."""
    same = HISTORY + [cs(6, TARGET, changes=777)]
    assert 6 not in {p.cid for p in point_in_time(same, TARGET)}


def test_sequence_is_oldest_first_and_only_prior():
    f = build(42, HISTORY, TARGET)
    assert len(f.sequence) == 3
    assert [row[0] for row in f.sequence] == [10.0, 20.0, 30.0]


def test_first_ever_edit_is_a_signal_not_a_hole():
    """New mapper is the top harmful reason in the OSMCha labels; zero history is
    information, not missing data."""
    f = build(99, [], "2016-01-01T00:00:00Z")
    assert f.n_prior == 0
    assert f.days_since_prev is None
    assert f.sequence == []
    assert f.total_prior_changes == 0


def test_days_since_previous_is_measured_backwards_from_the_target():
    f = build(42, HISTORY, TARGET)
    assert 150 < f.days_since_prev < 152   # 2015-01-01 to 2015-06-01


def test_truncation_is_recorded_not_hidden():
    """The API caps history at 100. A truncated history is itself a fact about the
    account, so it is a feature rather than something to paper over."""
    long_history = [cs(i, "2014-01-01T00:00:00Z") for i in range(HISTORY_LIMIT)]
    f = build(42, long_history, TARGET)
    assert f.history_truncated is True
    assert build(42, HISTORY, TARGET).history_truncated is False


def test_parse_changeset_list_reads_bbox_and_editor():
    rows = parse_changeset_list(CHANGESET_LIST_XML)
    assert [r.cid for r in rows] == [300, 200]
    assert rows[0].num_changes == 12
    assert rows[0].editor == "iD 2.0"
    assert rows[0].comment_length == len("add shops")
    assert rows[0].bbox_diag_deg > 0
    assert rows[1].comment_length == 0


def test_malformed_list_returns_empty():
    assert parse_changeset_list("<osm><changeset") == []
