"""Corpus A fetcher - pull changeset contents for the 18,276 Ovid-labelled changesets.

Gate 0 killed the S3 shortcut for this era (zero coverage of 2014-2019), so contents
come one changeset at a time from the OSM API:

    GET /api/0.6/changeset/<id>/download  ->  osmChange XML

Design constraints, in priority order:

  1. Be a good citizen. A shared token bucket caps the *global* request rate no
     matter how many worker threads are running, a real User-Agent identifies us,
     and HTTP 509 "Bandwidth Limit Exceeded" triggers exponential backoff - that is
     the signal OSM uses to say slow down.
  2. Be resumable. A long run will be interrupted - Colab kills sessions at 12 h
     and after 90 min idle. State lives in SQLite, so a restart skips everything
     already fetched and picks up mid-stream.
  3. Be honest about failures. A changeset that 404s or times out is recorded with
     its status rather than silently dropped, so the coverage number in the paper
     is measured rather than assumed.

SQLite over a directory of files: 18k small files is slow on Windows and slower on
a Drive mount, and a single database gives atomic writes, trivial resume, and lets
the featuriser query directly without a directory walk.

Never put the live database on a Drive/FUSE mount - SQLite locking there is
unreliable. Keep it on local disk and pass --mirror to copy a checkpoint to Drive
every few hundred rows instead.

Usage:
    python -m mapguard.data.fetch_corpus_a --limit 30                 # smoke test
    python -m mapguard.data.fetch_corpus_a --rate 4 --workers 8       # full run
    python -m mapguard.data.fetch_corpus_a --stats                    # progress only
"""

from __future__ import annotations

import argparse
import queue
import shutil
import sqlite3
import sys
import threading
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


class RateLimiter:
    """Token bucket shared by every worker, so --rate is a global ceiling.

    Without this, N threads each sleeping 1/rate would issue N*rate req/s.
    """

    def __init__(self, rate: float) -> None:
        self.interval = 1.0 / rate if rate > 0 else 0.0
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def acquire(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.interval
        if wait > 0:
            time.sleep(wait)


def connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db, check_same_thread=False)
    con.executescript(SCHEMA)
    # durability matters less than not losing hours to a corrupt write
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


def pending(con: sqlite3.Connection, limit: int | None) -> list[int]:
    """Anything never attempted, or attempted and failed transiently."""
    q = (
        "SELECT id FROM changeset "
        "WHERE status IS NULL OR status >= 500 ORDER BY id"
    )
    if limit:
        q += f" LIMIT {limit}"
    return [r[0] for r in con.execute(q).fetchall()]


def fetch_one(
    session: requests.Session, cid: int, limiter: RateLimiter
) -> tuple[int, str | None]:
    """One changeset, with backoff. Returns (status, xml-or-None)."""
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        limiter.acquire()
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
    ok_pos = con.execute(
        "SELECT COUNT(*) FROM changeset WHERE status = 200 AND vandalism = 1"
    ).fetchone()[0]
    by_status = dict(
        con.execute(
            "SELECT status, COUNT(*) FROM changeset "
            "WHERE status IS NOT NULL GROUP BY status"
        ).fetchall()
    )
    size = con.execute("SELECT COALESCE(SUM(n_bytes), 0) FROM changeset").fetchone()[0]
    return {
        "total": total, "attempted": done, "ok": ok, "ok_vandalism": ok_pos,
        "by_status": by_status, "bytes": size,
    }


def print_stats(con: sqlite3.Connection) -> None:
    s = stats(con)
    pct = s["attempted"] / s["total"] * 100 if s["total"] else 0
    print(f"  labelled changesets : {s['total']:,}")
    print(f"  attempted           : {s['attempted']:,}  ({pct:.1f}%)")
    print(f"  fetched ok          : {s['ok']:,}  ({s['ok_vandalism']:,} vandalism)")
    print(f"  status codes        : {s['by_status']}")
    print(f"  stored              : {s['bytes'] / 1e6:.1f} MB")


def mirror_db(db: Path, dest: Path) -> None:
    """Checkpoint the database somewhere durable (Drive) without running it there.

    sqlite3.backup writes a consistent snapshot even while writers are active,
    which a plain file copy of a WAL database does not guarantee.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    src = sqlite3.connect(db)
    snap = sqlite3.connect(tmp)
    with snap:
        src.backup(snap)
    snap.close()
    src.close()
    shutil.move(str(tmp), str(dest))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stop after N (smoke test)")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="global requests/second ceiling across all workers")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent connections; hides latency, does not raise --rate")
    ap.add_argument("--db", type=Path, default=DB, help="sqlite path (keep off Drive)")
    ap.add_argument("--mirror", type=Path, default=None,
                    help="checkpoint the db here every --mirror-every rows (e.g. Drive)")
    ap.add_argument("--mirror-every", type=int, default=500)
    ap.add_argument("--stats", action="store_true", help="show progress and exit")
    args = ap.parse_args()

    con = connect(args.db)
    seed(con)

    if args.stats:
        print_stats(con)
        return

    todo = pending(con, args.limit)
    if not todo:
        print("nothing pending.")
        print_stats(con)
        return

    eta = len(todo) / args.rate / 3600
    print(f"pending : {len(todo):,} changesets")
    print(f"rate    : {args.rate:g} req/s global, {args.workers} workers "
          f"->  ETA ~{eta:.1f} h")
    print(f"db      : {args.db}")
    if args.mirror:
        print(f"mirror  : {args.mirror} every {args.mirror_every:,} rows")
    print()

    limiter = RateLimiter(args.rate)
    work: queue.Queue[int | None] = queue.Queue()
    done: queue.Queue[tuple[int, int, str | None]] = queue.Queue()
    for cid in todo:
        work.put(cid)

    def worker() -> None:
        session = requests.Session()
        while True:
            cid = work.get()
            if cid is None:
                return
            status, xml = fetch_one(session, cid, limiter)
            done.put((cid, status, xml))

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(max(1, args.workers))]
    for t in threads:
        t.start()

    t_start = time.time()
    ok = fail = 0
    since_mirror = 0

    for i in range(1, len(todo) + 1):
        cid, status, xml = done.get()
        con.execute(
            "UPDATE changeset SET status=?, xml=?, n_bytes=?, fetched_at=? WHERE id=?",
            (status, xml, len(xml) if xml else 0,
             time.strftime("%Y-%m-%dT%H:%M:%S"), cid),
        )
        ok, fail = (ok + 1, fail) if status == 200 else (ok, fail + 1)
        since_mirror += 1

        if i % 50 == 0 or i == len(todo):
            con.commit()
            rate = i / (time.time() - t_start)
            remain = (len(todo) - i) / rate / 60 if rate else 0
            print(
                f"  {i:>6,}/{len(todo):,}  ok={ok:,} fail={fail:,}  "
                f"{rate:.2f}/s  ~{remain:.0f} min left",
                flush=True,
            )

        if args.mirror and since_mirror >= args.mirror_every:
            con.commit()
            mirror_db(args.db, args.mirror)
            since_mirror = 0
            print(f"    checkpointed -> {args.mirror}", flush=True)

    for _ in threads:
        work.put(None)

    con.commit()
    if args.mirror:
        mirror_db(args.db, args.mirror)
        print(f"    final checkpoint -> {args.mirror}")
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min")
    print_stats(con)
    con.close()


if __name__ == "__main__":
    sys.exit(main())
