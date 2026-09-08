"""Resumable SQLite stores for every fetch this project runs.

One database beats a directory of 18,276 small files: Windows is slow at that many
files and a Drive mount is slower, and a single database gives atomic writes, trivial
resume, and lets the featuriser query without a directory walk.

Failures are rows, not exceptions. A changeset that 404s or times out is stored with its
status so the coverage number in the paper is measured rather than assumed.

Never put a live database on a Drive or FUSE mount - SQLite locking there is unreliable.
Keep it on local disk and mirror a copy across periodically instead.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = {
    # osmChange documents for Corpus A
    "contents": """
        CREATE TABLE IF NOT EXISTS contents (
            cid      INTEGER PRIMARY KEY,
            status   INTEGER NOT NULL,
            xml      TEXT,
            bytes    INTEGER,
            seconds  REAL,
            error    TEXT,
            fetched  TEXT DEFAULT (datetime('now'))
        )
    """,
    # changeset metadata: who, when, bbox, comment, editor
    "meta": """
        CREATE TABLE IF NOT EXISTS meta (
            cid            INTEGER PRIMARY KEY,
            status         INTEGER NOT NULL,
            created_at     TEXT,
            closed_at      TEXT,
            uid            INTEGER,
            user           TEXT,
            num_changes    INTEGER,
            min_lat        REAL,
            min_lon        REAL,
            max_lat        REAL,
            max_lon        REAL,
            comments_count INTEGER,
            comment        TEXT,
            editor         TEXT,
            tags_json      TEXT,
            seconds        REAL,
            error          TEXT,
            fetched        TEXT DEFAULT (datetime('now'))
        )
    """,
    # human-reviewed labels from OSMCha, one row per reviewed changeset
    "labels": """
        CREATE TABLE IF NOT EXISTS labels (
            cid               INTEGER PRIMARY KEY,
            harmful           INTEGER,          -- 1 damaging, 0 reviewed-and-fine, NULL unknown
            checked           INTEGER NOT NULL,
            check_user        TEXT,
            check_date        TEXT,
            date              TEXT,
            uid               INTEGER,
            user              TEXT,
            editor            TEXT,
            comment           TEXT,
            n_create          INTEGER,
            n_modify          INTEGER,
            n_delete          INTEGER,
            is_suspect        INTEGER,
            reasons           TEXT,
            reviewed_features INTEGER,
            source            TEXT NOT NULL,    -- osmcha | revert_mining
            fetched           TEXT DEFAULT (datetime('now'))
        )
    """,
    # windows a harvest failed to retrieve, so coverage is measured rather than assumed.
    # A silently skipped window is a hole in the published dataset that nobody can see.
    "fetch_gaps": """
        CREATE TABLE IF NOT EXISTS fetch_gaps (
            source      TEXT NOT NULL,      -- osmcha | osm_api | overpass
            window_from TEXT NOT NULL,
            window_to   TEXT NOT NULL,
            klass       TEXT,               -- harmful | clean | NULL
            status      INTEGER,            -- 0 means the request never completed
            attempts    INTEGER DEFAULT 1,
            resolved    INTEGER DEFAULT 0,
            noted       TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (source, window_from, window_to, klass)
        )
    """,
    # before/after geometry from Overpass adiff, one row per changeset
    "geometry": """
        CREATE TABLE IF NOT EXISTS geometry (
            cid        INTEGER PRIMARY KEY,
            route      TEXT NOT NULL,       -- adiff | multifetch | skipped
            note       TEXT,
            n_actions  INTEGER,
            n_with_old INTEGER,
            n_way      INTEGER,
            json       TEXT,
            seconds    REAL,
            fetched    TEXT DEFAULT (datetime('now'))
        )
    """,
}


def connect(path: Path, tables: tuple[str, ...]) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60, isolation_level=None)
    # WAL keeps a reader (the progress query) from blocking the writer.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    for t in tables:
        conn.execute(SCHEMA[t])
    return conn


@contextmanager
def store(path: Path, *tables: str) -> Iterator[sqlite3.Connection]:
    conn = connect(path, tables or ("contents",))
    try:
        yield conn
    finally:
        conn.close()


def done_ids(conn: sqlite3.Connection, table: str, *, only_ok: bool = False) -> set[int]:
    """Which changesets are already fetched, so a restart skips them.

    `only_ok` retries previous failures; the default treats a recorded failure as done,
    which is what you want on a resume and not what you want on a repair pass.
    """
    where = " WHERE status = 200" if only_ok and table in ("contents", "meta") else ""
    return {row[0] for row in conn.execute(f"SELECT cid FROM {table}{where}")}


def counts(conn: sqlite3.Connection, table: str) -> dict:
    total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    out = {"total": total}
    if table in ("contents", "meta"):
        rows = conn.execute(
            f"SELECT status, COUNT(*) FROM {table} GROUP BY status ORDER BY 2 DESC"
        ).fetchall()
        out["by_status"] = {str(s): n for s, n in rows}
        out["ok"] = out["by_status"].get("200", 0)
    if table == "labels":
        rows = conn.execute(
            "SELECT harmful, COUNT(*) FROM labels GROUP BY harmful"
        ).fetchall()
        out["by_label"] = {("harmful" if h == 1 else "clean" if h == 0 else "unknown"): n
                           for h, n in rows}
    if table == "fetch_gaps":
        out["unresolved"] = conn.execute(
            "SELECT COUNT(*) FROM fetch_gaps WHERE resolved = 0").fetchone()[0]
        out["by_status"] = dict(conn.execute(
            "SELECT status, COUNT(*) FROM fetch_gaps GROUP BY status"))
    if table == "geometry":
        rows = conn.execute(
            "SELECT route, COUNT(*) FROM geometry GROUP BY route ORDER BY 2 DESC"
        ).fetchall()
        out["by_route"] = dict(rows)
    return out
