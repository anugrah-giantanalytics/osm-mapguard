"""adiff query construction, parsing, and the guard that keeps us off a shared service.

The fixture below is trimmed from a real 2026-09-04 response for way 202342236 and
changeset 52850166 - the two probes that overturned the v1 plan (docs/PLAN_REVIEW.md F1).
"""

from __future__ import annotations

from mapguard.data.osm_api import ChangesetMeta
from mapguard.data.overpass import (
    MAX_BBOX_DEGREES_SQ,
    build_query,
    parse_adiff,
    should_use_adiff,
)

ADIFF_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="Overpass API">
<action type="create">
  <node id="5162498181" lat="38.5707748" lon="-121.7145422" version="1"
        changeset="52850166" uid="5185313" user="higuy"/>
</action>
<action type="modify">
<old>
  <way id="202342236" version="3" changeset="19003120" uid="121415" user="edvac">
    <nd ref="2153022052" lat="10.3783181" lon="123.6337789"/>
    <nd ref="2123084535" lat="10.3787774" lon="123.6343200"/>
    <tag k="highway" v="residential"/>
  </way>
</old>
<new>
  <way id="202342236" version="4" changeset="52850166" uid="5185313" user="higuy">
    <nd ref="2153022052" lat="10.3783181" lon="123.6337789"/>
    <nd ref="2123084535" lat="10.3787774" lon="123.6343200"/>
    <nd ref="2123084539" lat="10.3789059" lon="123.6342760"/>
    <tag k="highway" v="residential"/>
    <tag k="name" v="Some Street"/>
  </way>
</new>
</action>
<action type="modify">
<old><node id="99" version="1" lat="1.0" lon="1.0" changeset="111"/></old>
<new><node id="99" version="2" lat="2.0" lon="2.0" changeset="111"/></new>
</action>
</osm>"""


def meta(**kw) -> ChangesetMeta:
    base = dict(
        cid=52850166,
        created_at="2017-10-12T06:55:16Z",
        closed_at="2017-10-12T06:55:17Z",
        uid=5185313,
        user="higuy",
        num_changes=5,
        min_lat=38.5706819,
        min_lon=-121.7148296,
        max_lat=38.5708774,
        max_lon=-121.7145422,
        comments_count=0,
        tags={"comment": "add building"},
    )
    base.update(kw)
    return ChangesetMeta(**base)


def test_query_puts_bbox_on_statements_not_in_the_setting_block():
    """The gotcha from probe P3: [bbox:...] alongside adiff fails to parse."""
    q = build_query(meta())
    assert q.startswith('[adiff:"')
    assert "[bbox:" not in q
    assert "node(38.5706819,-121.7148296,38.5708774,-121.7145422)" in q
    assert q.endswith("out geom meta;")


def test_query_window_is_padded():
    """An edit timestamped on the boundary must not fall outside the window."""
    q = build_query(meta())
    assert "2017-10-12T06:55:14Z" in q  # created_at - 2s
    assert "2017-10-12T06:55:19Z" in q  # closed_at  + 2s


def test_guard_rejects_missing_bbox():
    ok, why = should_use_adiff(meta(min_lat=None, min_lon=None, max_lat=None, max_lon=None))
    assert not ok and "bbox" in why


def test_guard_rejects_global_bbox():
    """A near-planet bbox must never be sent to a shared public instance."""
    ok, why = should_use_adiff(meta(min_lat=-80, min_lon=-170, max_lat=80, max_lon=170))
    assert not ok and "too large" in why


def test_guard_accepts_a_local_changeset():
    ok, why = should_use_adiff(meta())
    assert ok and why == "ok"
    assert MAX_BBOX_DEGREES_SQ > 0


def test_parse_filters_to_the_requested_changeset():
    """Overpass answers over an area and a window, so other people's edits come back
    too. Filtering on the changeset id is the caller's job and it must actually work."""
    actions = parse_adiff(ADIFF_XML, cid=52850166)
    assert len(actions) == 2
    assert {a.changeset for a in actions} == {52850166}


def test_parse_reads_both_sides_with_resolved_coordinates():
    """The whole point: a way comes back with lat/lon on its nodes, on both sides."""
    actions = parse_adiff(ADIFF_XML, cid=52850166)
    way = next(a for a in actions if a.otype == "way")
    assert way.old is not None and way.new is not None
    assert way.old.version == 3 and way.new.version == 4
    assert len(way.old.coords) == 2 and len(way.new.coords) == 3
    assert way.old.coords[0] == (10.3783181, 123.6337789)
    assert "name" not in way.old.tags and way.new.tags["name"] == "Some Street"


def test_parse_handles_create_actions_without_an_old_wrapper():
    actions = parse_adiff(ADIFF_XML, cid=52850166)
    created = next(a for a in actions if a.action == "create")
    assert created.old is None
    assert created.new is not None and created.new.coords == [(38.5707748, -121.7145422)]


def test_parse_without_a_changeset_filter_returns_everything():
    assert len(parse_adiff(ADIFF_XML)) == 3


def test_malformed_xml_returns_empty():
    assert parse_adiff("<osm><action") == []
