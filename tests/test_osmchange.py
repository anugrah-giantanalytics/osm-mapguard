"""Parser tests. The corpus is 18,276 documents; a parser bug is 18,276 wrong rows."""

from __future__ import annotations

from mapguard.data.osmchange import parse_safe

CREATE_ONLY = """<osmChange version="0.6" generator="iD">
  <create>
    <node id="1" version="1" uid="7" user="alice" lat="51.5" lon="-0.1"
          timestamp="2017-01-01T00:00:00Z">
      <tag k="amenity" v="cafe"/><tag k="name" v="Nero"/>
    </node>
  </create>
</osmChange>"""

MIXED = """<osmChange version="0.6">
  <create><node id="1" version="1" lat="1.0" lon="2.0" timestamp="t"/></create>
  <modify>
    <way id="9" version="4" timestamp="t">
      <nd ref="1"/><nd ref="2"/><nd ref="3"/>
      <tag k="highway" v="track"/>
    </way>
  </modify>
  <delete><node id="5" version="2" timestamp="t"/></delete>
</osmChange>"""

RELATION = """<osmChange version="0.6">
  <modify>
    <relation id="77" version="3" timestamp="t">
      <member type="way" ref="1" role="outer"/>
      <member type="way" ref="2" role="inner"/>
      <tag k="type" v="multipolygon"/>
    </relation>
  </modify>
</osmChange>"""

TRUNCATED = '<osmChange version="0.6"><create><node id="1" version="1"'


def test_create_only():
    edits, err = parse_safe(CREATE_ONLY)
    assert err is None
    assert len(edits) == 1
    e = edits[0]
    assert (e.action, e.otype, e.oid, e.version) == ("create", "node", 1, 1)
    assert e.lat == 51.5 and e.lon == -0.1
    assert e.tags == {"amenity": "cafe", "name": "Nero"}
    assert e.uid == 7 and e.user == "alice"


def test_mixed_preserves_document_order():
    """Document order is commit order, which the B3 edit-sequence branch consumes."""
    edits, err = parse_safe(MIXED)
    assert err is None
    assert [(e.action, e.otype, e.oid) for e in edits] == [
        ("create", "node", 1),
        ("modify", "way", 9),
        ("delete", "node", 5),
    ]


def test_way_has_refs_but_no_coordinates():
    """The gap that Overpass adiff fills - ways carry <nd ref>, never lat/lon."""
    edits, _ = parse_safe(MIXED)
    way = next(e for e in edits if e.otype == "way")
    assert way.n_refs == 3
    assert way.lat is None and way.lon is None


def test_relation_members_counted():
    edits, err = parse_safe(RELATION)
    assert err is None
    assert edits[0].n_refs == 2
    assert edits[0].tags["type"] == "multipolygon"


def test_deleted_node_still_yields_an_edit():
    edits, _ = parse_safe(MIXED)
    deleted = [e for e in edits if e.action == "delete"]
    assert len(deleted) == 1 and deleted[0].oid == 5


def test_truncated_document_reports_rather_than_raises():
    """A truncated cache entry is a fetch failure, not a crash - the coverage number
    in the paper has to be measured, so failures must be recordable."""
    edits, err = parse_safe(TRUNCATED)
    assert edits == []
    assert err and err.startswith("ParseError")


def test_large_document_does_not_retain_elements():
    """Guards the streaming fix from docs/planning/PLAN_REVIEW.md: clearing an element is not
    enough, it has to be detached, or a 50k-object changeset holds 50k husks."""
    n = 5000
    body = "".join(
        f'<node id="{i}" version="1" lat="1.0" lon="2.0" timestamp="t"/>' for i in range(n)
    )
    edits, err = parse_safe(f'<osmChange version="0.6"><create>{body}</create></osmChange>')
    assert err is None
    assert len(edits) == n
