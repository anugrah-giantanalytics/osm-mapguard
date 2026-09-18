"""Corpus R: revert mining from the changeset dump, and the label rules applied to it.

The dump parser is checked against a synthetic *multistream* bzip2 file cut into byte
ranges at arbitrary points, the same way the real 8.7 GB dump is read. The expectation is
exact: however the file is chunked, every changeset is counted once.
"""

from __future__ import annotations

import bz2
import sqlite3

import pytest

from mapguard.data import changeset_dump as cd
from mapguard.data import corpora


def cs(cid, created, tags=None, uid=7):
    body = "".join(f'<tag k="{k}" v="{v}"/>' for k, v in (tags or {}).items())
    head = (f'<changeset id="{cid}" created_at="{created}" closed_at="{created}" open="false" '
            f'user="u{uid}" uid="{uid}" min_lat="1.0" min_lon="2.0" max_lat="1.1" max_lon="2.1" '
            f'comments_count="0" num_changes="3"')
    return head + (f">{body}</changeset>\n" if body else "/>\n")


# ------------------------------------------------------------------------ detection


def test_osm_revert_tag_gives_every_target():
    tags = {"revert:id": "https://www.openstreetmap.org/changeset/185800330;"
                         "https://www.openstreetmap.org/changeset/185800331",
            "created_by": "osm-revert git#c8b2ad4"}
    assert cd.revert_targets(185900000, tags) == ("osm-revert", [185800330, 185800331])


def test_forum_and_note_numbers_are_not_targets():
    tags = {"comment": "Revert all changesets by x since CS 183039302, see "
                       "https://c.osm.org/t/144453 and note/5328952"}
    assert cd.revert_targets(183900000, tags) == ("comment", [183039302])


def test_josm_reverter_localised_comment():
    tags = {"created_by": "reverter_plugin/36474;JOSM/1.5",
            "comment": "Wycofanie zestawu zmian nr 183825268 (powod: wandalizm)"}
    assert cd.revert_targets(183900000, tags) == ("josm-reverter", [183825268])


def test_no_revert_word_no_targets_and_future_ids_dropped():
    assert cd.revert_targets(100, {"comment": "added 183825268 buildings"}) is None
    assert cd.revert_targets(1_000_000, {"comment": "revert 2000000"}) is None   # not earlier


def test_sample_is_a_fixed_id_hash():
    picked = [c for c in range(1, 200_001) if cd.sampled(c)]
    assert 350 < len(picked) < 650                      # ~1 in 400
    assert picked == [c for c in range(1, 200_001) if cd.sampled(c)]


# ------------------------------------------------------------------------ chunking


def multistream(xml: str, piece: int) -> bytes:
    raw = xml.encode()
    return b"".join(bz2.compress(raw[i:i + piece], 9) for i in range(0, len(raw), piece))


@pytest.fixture()
def dump(monkeypatch):
    rows = []
    for i in range(3000):
        cid = 1_000_000 + i
        tags = {"comment": f"revert {cid - 5}"} if i % 97 == 0 and i else {"comment": f"edit {i}"}
        rows.append(cs(cid, f"2024-{1 + i % 12:02d}-10T00:00:00Z", tags))
    xml = '<?xml version="1.0"?>\n<osm version="0.6">\n' + "".join(rows) + "</osm>\n"
    blob = multistream(xml, 7_919)                     # element boundaries fall mid-stream
    monkeypatch.setattr(cd, "fetch_range", lambda url, a, b, retries=6: blob[a:b])
    return blob


def total(o):
    return sum(o["months"].values()), len(o["reverts"]), len(o["sample"])


def test_any_chunking_counts_every_changeset_exactly_once(dump):
    size = len(dump)
    whole = cd.scan_chunk("x", 0, size, size, "2000", "2100")
    assert whole["months"] and sum(whole["months"].values()) == 3000
    for n_chunks in (2, 3, 7):
        bounds = [size * k // n_chunks for k in range(n_chunks + 1)]
        parts = [cd.scan_chunk("x", a, b, size, "2000", "2100") for a, b in zip(bounds, bounds[1:])]
        got = tuple(sum(x) for x in zip(*(total(p) for p in parts)))
        assert got == total(whole), n_chunks


def test_date_window_filters_rows(dump):
    size = len(dump)
    o = cd.scan_chunk("x", 0, size, size, "2024-03-01", "2024-05-01")
    assert set(o["months"]) == {"2024-03", "2024-04"}


# ------------------------------------------------------------------------ label rules


def r_db(rows, targets, window=("2021-09-01", "2026-06-16")):
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE selection (cid INTEGER PRIMARY KEY, corpus TEXT, label INTEGER, quarter TEXT, rank TEXT);"
        "CREATE TABLE meta (cid INTEGER PRIMARY KEY, status INTEGER, created_at TEXT, closed_at TEXT,"
        " uid INTEGER, num_changes INTEGER);"
        "CREATE TABLE r_targets (cid INTEGER PRIMARY KEY, reverter_cid INTEGER, reverter_created TEXT,"
        " reverter_uid INTEGER, kind TEXT);"
        "CREATE TABLE r_window (k TEXT PRIMARY KEY, v TEXT);")
    conn.executemany("INSERT INTO r_window VALUES (?,?)", [("since", window[0]), ("censor", window[1])])
    for i, (cid, label, created, uid) in enumerate(rows):
        conn.execute("INSERT INTO selection VALUES (?,?,?,?,?)", (cid, "R", label, corpora.quarter_of(created), f"{i:04d}"))
        conn.execute("INSERT INTO meta VALUES (?,?,?,?,?,?)", (cid, 200, created, created, uid, 3))
    conn.executemany("INSERT INTO r_targets VALUES (?,?,?,?,?)", targets)
    return conn


def test_r_positives_need_another_author_a_prompt_revert_and_an_uncensored_date():
    rows = [(1, 1, "2024-02-01T00:00:00Z", 10),       # good
            (2, 1, "2024-02-01T00:00:00Z", 11),       # self-revert
            (3, 1, "2024-02-01T00:00:00Z", 12),       # reverted 200 days later
            (4, 1, "2026-07-01T00:00:00Z", 13),       # after the censor date
            (5, 0, "2024-02-15T00:00:00Z", 20),       # negative, same quarter
            (6, 0, "2024-03-01T00:00:00Z", 21)]       # second negative: quota is 1
    targets = [(1, 900, "2024-02-02T00:00:00Z", 99, "osm-revert"),
               (2, 901, "2024-02-02T00:00:00Z", 11, "osm-revert"),
               (3, 902, "2024-08-19T00:00:00Z", 99, "osm-revert"),
               (4, 903, "2026-07-02T00:00:00Z", 99, "osm-revert")]
    assert valid_r(r_db(rows, targets)) == {1: ("R", 1), 5: ("R", 0)}


def test_r_caps_changesets_per_author():
    rows = [(i, 1, "2024-02-01T00:00:00Z", 10) for i in range(1, 9)]
    targets = [(i, 900 + i, "2024-02-03T00:00:00Z", 99, "comment") for i in range(1, 9)]
    assert len(valid_r(r_db(rows, targets))) == corpora.R_PER_AUTHOR


def valid_r(conn):
    rows = conn.execute(
        "SELECT s.cid, s.corpus, s.label, s.quarter, m.status, m.closed_at, m.num_changes,"
        " m.created_at, m.uid FROM selection s JOIN meta m ON m.cid=s.cid ORDER BY s.rank").fetchall()
    return corpora._valid_r(conn, rows)
