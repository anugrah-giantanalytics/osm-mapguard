"""Per-user changeset history for the B4 branch, fetched point-in-time.

One call per *user*, not per changeset - the corpus has roughly four thousand distinct
users against 18,276 changesets. For each user we ask for changesets closed before their
latest labelled target, newest first, capped at 100 by the API.

`GET /api/0.6/changesets?user=<uid>&time=2005-01-01T00:00:00Z,<before>&limit=100`

The API filters by time, and `mapguard.features.user_history.point_in_time` filters again
per target when features are built. Two filters for one rule is deliberate: a correctness
guarantee this important does not get to live on a remote service.

    python scripts/fetch_user_histories.py --limit 50    # smoke test
    python scripts/fetch_user_histories.py              # full run, resumable
    python scripts/fetch_user_histories.py --stats
"""

from __future__ import annotations

import argparse
import json
import queue
import sqlite3
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapguard.config import CORPUS_A_META_DB, DEFAULT_RATE, DEFAULT_WORKERS, data_path
from mapguard.data.http import PoliteSession
from mapguard.features.user_history import fetch_user_history

HISTORY_DB = "raw/user_histories.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS histories (
    uid       INTEGER PRIMARY KEY,
    status    INTEGER NOT NULL,
    before    TEXT,          -- the cutoff this history was fetched against
    n         INTEGER,
    truncated INTEGER,
    json      TEXT,
    fetched   TEXT DEFAULT (datetime('now'))
)
"""


def targets(meta_db: Path) -> dict[int, str]:
    """uid -> latest target timestamp, so one fetch covers all of that user's rows."""
    if not meta_db.exists():
        raise SystemExit(f"no metadata store at {meta_db} - run fetch_corpus_a_meta.py first")
    conn = sqlite3.connect(meta_db)
    rows = conn.execute(
        "SELECT uid, MAX(created_at) FROM meta"
        " WHERE status = 200 AND uid IS NOT NULL AND created_at IS NOT NULL"
        " GROUP BY uid"
    ).fetchall()
    conn.close()
    return {int(u): t for u, t in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    db = data_path(*HISTORY_DB.split("/"))
    conn = sqlite3.connect(db, timeout=60, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(SCHEMA)

    if args.stats:
        n, tot, trunc = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(n),0), COALESCE(SUM(truncated),0) FROM histories"
        ).fetchone()
        print(json.dumps(
            {"users": n, "prior_changesets": tot, "truncated_users": trunc,
             "mean_history": round(tot / max(1, n), 1)}, indent=2))
        return 0

    want = targets(data_path(*CORPUS_A_META_DB.split("/")))
    have = {r[0] for r in conn.execute("SELECT uid FROM histories")}
    todo = [(u, t) for u, t in want.items() if u not in have]
    if args.limit:
        todo = todo[: args.limit]

    print(f"database : {db}")
    print(f"users    : {len(want)} total, {len(todo)} to fetch")
    if not todo:
        return 0

    session = PoliteSession(args.rate)
    work: queue.Queue = queue.Queue()
    for item in todo:
        work.put(item)
    rows: queue.Queue = queue.Queue()

    def worker() -> None:
        while True:
            try:
                uid, before = work.get_nowait()
            except queue.Empty:
                return
            history, status = fetch_user_history(session, uid, before)
            rows.put((
                uid, status, before, len(history), 1 if len(history) >= 100 else 0,
                json.dumps([h.__dict__ for h in history], ensure_ascii=False),
            ))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()

    started, written = time.monotonic(), 0
    while any(t.is_alive() for t in threads) or not rows.empty():
        try:
            row = rows.get(timeout=1.0)
        except queue.Empty:
            continue
        conn.execute(
            "INSERT OR REPLACE INTO histories (uid,status,before,n,truncated,json)"
            " VALUES (?,?,?,?,?,?)", row)
        written += 1
        if written % 100 == 0:
            rate = written / max(1e-6, time.monotonic() - started)
            print(f"{written}/{len(todo)}  {rate:.1f}/s  eta {(len(todo)-written)/max(1e-6,rate)/60:.0f} min",
                  flush=True)

    n, tot, trunc = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(n),0), COALESCE(SUM(truncated),0) FROM histories"
    ).fetchone()
    print(json.dumps({"users": n, "prior_changesets": tot, "truncated_users": trunc}, indent=2))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
