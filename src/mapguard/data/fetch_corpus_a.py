"""Corpus A fetcher - pull changeset contents for the 18,276 Ovid-labelled changesets.

Gate 0 killed the S3 shortcut for this era (zero coverage of 2014-2019), so contents
come one changeset at a time from the OSM API:

    GET /api/0.6/changeset/<id>/download  ->  osmChange XML

Design constraints, in priority order:

  1. Be a good citizen. One request per second by default, a real User-Agent, and
     exponential backoff on HTTP 509 "Bandwidth Limit Exceeded" - the signal OSM
     uses to say slow down. Never parallelise this against the live API.
  2. Be resumable. A five-hour run will be interrupted. State lives in SQLite, so
     a restart skips everything already fetched and picks up mid-stream.
  3. Be honest about failures. A changeset that 404s or times out is recorded with
     its status rather than silently dropped, so the coverage number in the paper
     is measured rather than assumed.

SQLite over a directory of files: 18k small files is slow on Windows, and a single
database gives atomic writes, trivial resume, and lets the featuriser query
directly without a directory walk.

Usage:
    python -m mapguard.data.fetch_corpus_a --limit 30        # smoke test
    python -m mapguard.data.fetch_corpus_a                   # full run, ~5 h
    python -m mapguard.data.fetch_corpus_a --stats           # progress only
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
LABELS = ROOT / "data" / "labels" / "ovid_labels.tsv"
DB = ROOT / "data" / "raw" / "corpus_a.sqlite"

OSM_API = "https://api.openstreetmap.org/api/0.6"
UA = {
    "User-Agent": (
        "MapGuard-research/0.1 (academic vandalism-detection study; "
        "https://github.com/ayushmgarg/osm_bad)"
    )
}
TIMEOUT = 60
MAX_RETRIES = 5


SCHEMA = """
CREATE TABLE IF NOT EXISTS changeset (
    id          INTEGER PRIMARY KEY,
    vandalism   INTEGER NOT NULL,
    status      INTEGER,
    xml         TEXT,
    n_bytes     INTEGER,
    fetched_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_status ON changeset(status);
"""


def connect() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    # durability matters less than not losing five hours to a corrupt write
    con.execute("PRAGMA journal_mode=WAL")
    return con


def seed(con: sqlite3.Connection) -> int:
    """Insert every label row once; existing rows are left untouched."""
    rows = []
    with LABELS.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            cid, label = line.rstrip("\n").split("\t")
            rows.append((int(cid), 1 if label.strip().lower() == "true" else 0))
    con.executemany(
        "INSERT OR IGNORE INTO changeset(id, vandalism) VALUES (?, ?)", rows
    )
    con.commit()
    return len(rows)


def pending(con: sqlite3.Connection, limit: int | None) -> list[tuple[int, int]]:
    """Anything never attempted, or attempted and failed transiently."""
    q = (
        "SELECT id, vandalism FROM changeset "
        "WHERE status IS NULL OR status >= 500 ORDER BY id"
    )
    if limit:
        q += f" LIMIT {limit}"
    return con.execute(q).fetchall()


def fetch_one(session: requests.Session, cid: int) -> tuple[int, str | None]:
    """One changeset, with backoff. Returns (status, xml-or-None)."""
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(
                f"{OSM_API}/changeset/{cid}/download", headers=UA, timeout=TIMEOUT
            )
            if r.status_code == 200:
                return 200, r.text
            if r.status_code in (509, 429, 502, 503, 504):
                # OSM telling us to slow down. Back off hard.
                wait = delay * (2 ** attempt)
                print(f"    {r.status_code} on {cid} - backing off {wait:.0f}s")
                time.sleep(wait)
                continue
            return r.status_code, None  # 404 / 410 are terminal, record and move on
        except requests.RequestException as exc:
            wait = delay * (2 ** attempt)
            print(f"    {type(exc).__name__} on {cid} - retry in {wait:.0f}s")
            time.sleep(wait)
    return 599, None  # exhausted retries


def stats(con: sqlite3.Connection) -> dict:
    total = con.execute("SELECT COUNT(*) FROM changeset").fetchone()[0]
    done = con.execute(
        "SELECT COUNT(*) FROM changeset WHERE status IS NOT NULL"
    ).fetchone()[0]
    ok = con.execute("SELECT COUNT(*) FROM changeset WHERE status = 200").fetchone()[0]
    by_status = dict(
        con.execute(
            "SELECT status, COUNT(*) FROM changeset "
            "WHERE status IS NOT NULL GROUP BY status"
        ).fetchall()
    )
    size = con.execute("SELECT COALESCE(SUM(n_bytes), 0) FROM changeset").fetchone()[0]
    return {
        "total": total, "attempted": done, "ok": ok,
        "by_status": by_status, "bytes": size,
    }


def print_stats(con: sqlite3.Connection) -> None:
    s = stats(con)
    pct = s["attempted"] / s["total"] * 100 if s["total"] else 0
    print(f"  labelled changesets : {s['total']:,}")
    print(f"  attempted           : {s['attempted']:,}  ({pct:.1f}%)")
    print(f"  fetched ok          : {s['ok']:,}")
    print(f"  status codes        : {s['by_status']}")
    print(f"  stored              : {s['bytes'] / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stop after N (smoke test)")
    ap.add_argument("--rate", type=float, default=1.0, help="requests per second")
    ap.add_argument("--stats", action="store_true", help="show progress and exit")
    args = ap.parse_args()

    con = connect()
    seed(con)

    if args.stats:
        print_stats(con)
        return

    todo = pending(con, args.limit)
    if not todo:
        print("nothing pending.")
        print_stats(con)
        return

    interval = 1.0 / args.rate
    eta = len(todo) * interval / 3600
    print(f"pending : {len(todo):,} changesets")
    print(f"rate    : {args.rate:g} req/s  ->  ETA ~{eta:.1f} h")
    print(f"db      : {DB.relative_to(ROOT)}\n")

    session = requests.Session()
    t_start = time.time()
    ok = fail = 0

    for i, (cid, _) in enumerate(todo, 1):
        t0 = time.time()
        status, xml = fetch_one(session, cid)
        con.execute(
            "UPDATE changeset SET status=?, xml=?, n_bytes=?, fetched_at=? WHERE id=?",
            (status, xml, len(xml) if xml else 0,
             time.strftime("%Y-%m-%dT%H:%M:%S"), cid),
        )
        if status == 200:
            ok += 1
        else:
            fail += 1

        if i % 25 == 0 or i == len(todo):
            con.commit()
            rate = i / (time.time() - t_start)
            remain = (len(todo) - i) / rate / 60 if rate else 0
            print(
                f"  {i:>6,}/{len(todo):,}  ok={ok:,} fail={fail:,}  "
                f"{rate:.2f}/s  ~{remain:.0f} min left",
                flush=True,
            )

        # pace to the requested rate, discounting time already spent in-flight
        sleep = interval - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)

    con.commit()
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min")
    print_stats(con)
    con.close()


if __name__ == "__main__":
    sys.exit(main())
