"""The feature builder, end to end, against stores laid out exactly like the real ones.

The earlier builder passed its unit tests and crashed on the real Corpus A store, because
the tests never touched the table names the fetchers actually write. These tests build
tiny SQLite stores with the *real* schemas - `changeset(id, xml)` for Corpus A contents,
`contents(cid, xml)` for M/N - run `main()`, and read the Parquet back through the
training data contract.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "train"), str(REPO / "src")]

from mapguard.data import corpora  # noqa: E402
from mapguard.data.store import SCHEMA  # noqa: E402
from mapguard.features import build_store  # noqa: E402

# Node 10 moves ~111 m north; way 20 gains a node and a name; node 30 is created.
XML_A = """<osmChange version="0.6">
  <modify>
    <node id="10" version="3" lat="10.001" lon="20.0" timestamp="2016-05-01T00:00:00Z">
      <tag k="amenity" v="bench"/>
    </node>
    <way id="20" version="5" timestamp="2016-05-01T00:00:00Z">
      <nd ref="1"/><nd ref="2"/><nd ref="3"/>
      <tag k="highway" v="residential"/><tag k="name" v="Main Street"/>
    </way>
  </modify>
  <create>
    <node id="30" version="1" lat="1.0" lon="2.0" timestamp="2016-05-01T00:00:00Z"/>
  </create>
</osmChange>"""

XML_M = """<osmChange version="0.6">
  <modify><node id="40" version="2" lat="5.0" lon="5.0" timestamp="2025-02-01T00:00:00Z"/></modify>
</osmChange>"""

META_INSERT = ("INSERT INTO meta (cid,status,created_at,closed_at,uid,user,num_changes,min_lat,"
               "min_lon,max_lat,max_lon,comments_count,comment,editor) VALUES "
               "(?,200,?,?,?,'x',?,0,0,1,1,7,'c','iD')")


def _db(root: Path, rel: str) -> sqlite3.Connection:
    p = root.joinpath(*rel.split("/"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(p)


@pytest.fixture()
def stores(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("MAPGUARD_DATA", str(root))

    c = _db(root, "raw/corpus_a_meta.sqlite")
    c.execute(SCHEMA["meta"])
    c.execute(META_INSERT, (1001, "2016-05-01T00:00:00Z", "2016-05-01T01:00:00Z", 7, 3))
    c.execute(META_INSERT, (1002, "2016-06-01T00:00:00Z", "2016-06-01T01:00:00Z", 8, 3))
    c.commit()

    c = _db(root, "raw/corpus_a.sqlite")
    c.execute("CREATE TABLE changeset (id INTEGER PRIMARY KEY, vandalism INTEGER, status INTEGER,"
              " xml TEXT, n_bytes INTEGER, fetched_at TEXT)")
    c.execute("INSERT INTO changeset VALUES (1001, 1, 200, ?, 0, '')", (XML_A,))
    c.execute("INSERT INTO changeset VALUES (1002, 0, 200, ?, 0, '')", (XML_A,))
    c.commit()

    c = _db(root, "raw/prev_versions.sqlite")
    c.execute("CREATE TABLE versions (otype TEXT, oid INTEGER, version INTEGER, status INTEGER,"
              " visible INTEGER, lat REAL, lon REAL, n_refs INTEGER, tags_json TEXT)")
    c.execute("INSERT INTO versions VALUES ('node',10,2,200,1,10.0,20.0,0,?)",
              (json.dumps({"amenity": "bench"}),))
    c.execute("INSERT INTO versions VALUES ('way',20,4,200,1,NULL,NULL,2,?)",
              (json.dumps({"highway": "residential"}),))
    c.commit()

    c = _db(root, "raw/users.sqlite")
    c.execute("CREATE TABLE users (uid INTEGER PRIMARY KEY, status INTEGER, display_name TEXT,"
              " account_created TEXT, changesets INTEGER, raw_json TEXT, fetched TEXT)")
    c.execute("INSERT INTO users VALUES (7, 200, 'a', '2015-05-01T00:00:00Z', 1, NULL, NULL)")
    c.commit()

    prior = [{"cid": 900, "created_at": "2016-01-01T00:00:00Z", "num_changes": 4,
              "editor": "iD", "comment_length": 3, "bbox_diag_deg": 0.1},
             {"cid": 999, "created_at": "2016-05-20T00:00:00Z", "num_changes": 9,
              "editor": "iD", "comment_length": 0, "bbox_diag_deg": 0.2}]
    c = _db(root, "raw/user_histories_pit.sqlite")
    c.execute("CREATE TABLE histories (uid INTEGER, before TEXT, status INTEGER, n INTEGER,"
              " truncated INTEGER, json TEXT, fetched TEXT, PRIMARY KEY (uid, before))")
    # a complete history fetched for a later target of uid 7
    c.execute("INSERT INTO histories VALUES (7, '2016-07-01T00:00:00Z', 200, 2, 0, ?, NULL)",
              (json.dumps(prior),))
    c.commit()

    monkeypatch.setattr(build_store, "load_labels", lambda: {1001: 1, 1002: 0})
    return root


def _mn_store(root: Path):
    c = _db(root, corpora.CORPUS_MN_DB)
    c.executescript("CREATE TABLE selection (cid INTEGER PRIMARY KEY, corpus TEXT, label INTEGER,"
                    " quarter TEXT, rank TEXT);" + SCHEMA["meta"] + ";" + SCHEMA["contents"] + ";")
    c.execute("INSERT INTO selection VALUES (5001,'M',1,'2025Q1','a')")
    c.execute("INSERT INTO selection VALUES (5002,'N',0,'2025Q1','b')")
    c.execute("INSERT INTO selection VALUES (5003,'N',0,'2025Q1','c')")   # empty changeset
    c.execute(META_INSERT, (5001, "2025-02-01T00:00:00Z", "2025-02-01T01:00:00Z", 11, 1))
    c.execute(META_INSERT, (5002, "2025-02-02T00:00:00Z", "2025-02-02T01:00:00Z", 12, 1))
    c.execute(META_INSERT, (5003, "2025-02-03T00:00:00Z", "2025-02-03T01:00:00Z", 13, 0))
    for cid in (5001, 5002):
        c.execute("INSERT INTO contents (cid,status,xml) VALUES (?,200,?)", (cid, XML_M))
    c.commit()


def test_builder_reads_real_table_names_and_writes_the_contract(stores, tmp_path):
    from common import data_contract

    out = tmp_path / "fs"
    assert build_store.main(["--corpora", "A", "--out", str(out)]) == 0
    store = data_contract.load(out, task="ovid")
    cs = store.changesets.set_index("cid")
    assert set(cs["corpus"]) == {"A"}
    assert cs.loc[1001, "label"] == 1 and cs.loc[1002, "label"] == 0
    assert "comments_count" not in cs.columns        # read years later: it leaks the label
    assert cs.loc[1001, "account_age_known"] == 1
    assert cs.loc[1002, "account_age_known"] == 0     # uid 8 has no account row
    assert cs.loc[1001, "account_age_days"] == pytest.approx(366, abs=1)
    assert cs.loc[1002, "user_history_known"] == 0


def test_prev_version_route_gives_displacement_tag_diffs_and_ref_change(stores, tmp_path):
    out = tmp_path / "fs"
    build_store.main(["--corpora", "A", "--out", str(out)])
    e = pd.read_parquet(out / "edits.parquet")
    e = e[e.cid == 1001].set_index("oid")
    assert e.loc[10, "geom_route"] == 1
    assert e.loc[10, "centroid_shift_m"] == pytest.approx(111.2, abs=1.0)
    assert e.loc[20, "tags_added"] == 1                # name added
    assert e.loc[20, "d_refs"] == 1                    # 2 -> 3 nodes
    assert e.loc[30, "geom_route"] == 0                # created: no previous version
    cs = pd.read_parquet(out / "changesets.parquet").set_index("cid")
    assert cs.loc[1001, "frac_edits_prev_version"] == pytest.approx(2 / 3, abs=1e-3)


def test_point_in_time_history_uses_a_complete_later_window_but_only_the_past(stores, tmp_path):
    out = tmp_path / "fs"
    build_store.main(["--corpora", "A", "--out", str(out)])
    cs = pd.read_parquet(out / "changesets.parquet").set_index("cid")
    assert cs.loc[1001, "user_history_known"] == 1
    assert cs.loc[1001, "user_n_prior"] == 1           # the 2016-05-20 changeset is the future
    seq = pd.read_parquet(out / "user_seq.parquet")
    assert list(seq[seq.cid == 1001]["num_changes"]) == [4.0]


def test_truncated_later_window_is_never_used():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE histories (uid INTEGER, before TEXT, status INTEGER, n INTEGER,"
                 " truncated INTEGER, json TEXT)")
    conn.execute("INSERT INTO histories VALUES (1, '2020-01-01', 200, 100, 1, '[]')")
    assert corpora.history_for(conn, 1, "2019-01-01") == (None, "missing")
    assert corpora.history_for(conn, 1, "2020-01-01") == ("[]", "exact")
    conn.execute("INSERT INTO histories VALUES (2, '2020-01-01', 200, 5, 0, '[1]')")
    assert corpora.history_for(conn, 2, "2019-01-01") == ("[1]", "complete_later_window")
    assert corpora.history_for(conn, 2, "2021-01-01") == (None, "missing")


def test_modern_corpora_keep_m_and_only_usable_n(stores, tmp_path):
    from common import data_contract

    _mn_store(stores)
    out = tmp_path / "fs"
    build_store.main(["--corpora", "A,MN", "--out", str(out)])
    cs = pd.read_parquet(out / "changesets.parquet").set_index("cid")
    assert cs.loc[5001, "corpus"] == "M" and cs.loc[5001, "label"] == 1
    assert cs.loc[5002, "corpus"] == "N" and cs.loc[5002, "label"] == 0
    assert 5003 not in cs.index                        # empty changesets never count as N
    assert cs.loc[5001, "quarter"] == "2025Q1"
    t2 = data_contract.load(out, task="T2").changesets
    assert set(t2["cid"]) == {5001, 5002}
    report = json.loads((out / "build_report.json").read_text())
    assert report["coverage_by_corpus"]["A"]["label_1"] == 1


def test_empty_user_seq_still_writes_the_file(stores, tmp_path, monkeypatch):
    monkeypatch.setattr(build_store, "HISTORY_DB", "raw/none.sqlite")
    out = tmp_path / "fs"
    build_store.main(["--corpora", "A", "--out", str(out)])
    assert (out / "user_seq.parquet").exists()
