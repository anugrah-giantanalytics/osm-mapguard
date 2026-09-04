"""Replication parsing, account age, rate limiting, and the two attribute-name traps.

None of these hit the network. The live checks live in the gate scripts, where their
results get committed to results/ as evidence.
"""

from __future__ import annotations

import time

from mapguard.data.http import TokenBucket
from mapguard.data.osm_api import parse_changeset_meta
from mapguard.data.replication import parse_changeset_file, seq_to_path
from mapguard.data.users import account_age_days, parse_user

# Trimmed from a real /replication/changesets/007/170/060.osm.gz, 2026-09-04.
REPLICATION_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="replicate_changesets.rb">
  <changeset id="188504682" created_at="2026-09-04T03:03:41Z" closed_at="2026-09-04T04:05:51Z"
             open="false" num_changes="10" user="Someone" uid="16804109"
             min_lat="45.4956345" max_lat="45.4959096" min_lon="9.2194345" max_lon="9.2197669"
             comments_count="0">
    <tag k="comment" v="Specify whether steps have a ramp"/>
    <tag k="created_by" v="StreetComplete 63.4"/>
  </changeset>
</osm>"""

# The API spells the same field differently. Trap noted in CHECKLIST 2.3.4.
API_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <changeset id="52850166" created_at="2017-10-12T06:55:16Z" closed_at="2017-10-12T06:55:17Z"
             open="false" user="higuy" uid="5185313" changes_count="5"
             min_lat="38.5706819" min_lon="-121.7148296" max_lat="38.5708774" max_lon="-121.7145422"
             comments_count="0">
    <tag k="comment" v="add building"/>
    <tag k="created_by" v="iD 2.4.3"/>
  </changeset>
</osm>"""

USER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <user id="5185313" display_name="higuy" account_created="2017-09-01T10:00:00Z">
    <changesets count="412"/>
  </user>
</osm>"""


def test_sequence_number_to_path():
    assert seq_to_path(7170060) == "007/170/060"
    assert seq_to_path(1) == "000/000/001"
    assert seq_to_path(1_234_567) == "001/234/567"


def test_replication_record_is_rich_enough_for_b5_and_revert_mining():
    rows = parse_changeset_file(REPLICATION_XML)
    assert len(rows) == 1
    r = rows[0]
    assert r.cid == 188504682
    assert r.num_changes == 10
    assert r.uid == 16804109
    assert r.editor == "StreetComplete 63.4"
    assert "ramp" in r.comment
    assert r.min_lat is not None and r.max_lon is not None


def test_api_changes_count_is_normalised_to_num_changes():
    """The API says changes_count, the replication stream says num_changes. Reading only
    one spelling makes the feature silently null for the other source."""
    m = parse_changeset_meta(API_XML)
    assert m is not None
    assert m.num_changes == 5
    assert m.editor == "iD 2.4.3"
    assert m.has_bbox
    assert m.bbox == (38.5706819, -121.7148296, 38.5708774, -121.7145422)


def test_bbox_order_is_south_west_north_east_for_overpass():
    m = parse_changeset_meta(API_XML)
    south, west, north, east = m.bbox
    assert south < north and west < east


def test_user_parse():
    u = parse_user(USER_XML)
    assert u["uid"] == 5185313
    assert u["account_created"].startswith("2017-09-01")
    assert u["changesets"] == 412


def test_account_age_at_the_moment_of_the_edit():
    days = account_age_days("2017-09-01T10:00:00Z", "2017-10-12T06:55:16Z")
    assert 40 < days < 42


def test_account_age_refuses_to_go_backwards():
    """A negative age means bad data. Returning None makes it show up as missing rather
    than as a suspiciously predictive feature."""
    assert account_age_days("2017-10-12T00:00:00Z", "2017-09-01T00:00:00Z") is None
    assert account_age_days(None, "2017-09-01T00:00:00Z") is None


def test_token_bucket_enforces_a_global_rate():
    """Eight threads each sleeping a second is eight requests a second at the server.
    One shared bucket is the only version that actually limits anything."""
    bucket = TokenBucket(rate=10.0, burst=1.0)
    bucket.take()
    started = time.monotonic()
    for _ in range(5):
        bucket.take()
    elapsed = time.monotonic() - started
    assert elapsed >= 0.4  # 5 tokens at 10/s, minus the initial one already held


def test_malformed_inputs_return_empty_rather_than_raising():
    assert parse_changeset_file("<osm><changeset") == []
    assert parse_changeset_meta("not xml") is None
    assert parse_user("<osm/>") is None
