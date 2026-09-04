"""Corpus A changeset metadata - who, when, bbox, comment, editor, for all 18,276.

Metadata is small, fast, and everything downstream needs it:

  - Gate 1c (the B0 leakage tripwire) runs on comment length, editor and account age
  - Overpass adiff needs the bbox and the [created_at, closed_at] window
  - the B5 tabular branch and every classical baseline use it directly

Separate from the contents fetch on purpose. Metadata is one small request per changeset
and finishes in about an hour; contents are larger and slower, and there is no reason to
block the tripwire on them.

    python scripts/fetch_corpus_a_meta.py --limit 50      # smoke test
    python scripts/fetch_corpus_a_meta.py                 # full run, resumable
    python scripts/fetch_corpus_a_meta.py --stats         # progress only
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapguard.config import (
    CORPUS_A_META_DB,
    DEFAULT_RATE,
    DEFAULT_WORKERS,
    LABELS,
    data_path,
)
from mapguard.data.http import PoliteSession
from mapguard.data.osm_api import fetch_changeset_meta
from mapguard.data.store import counts, done_ids, store

INSERT = """
INSERT OR REPLACE INTO meta
 (cid,status,created_at,closed_at,uid,user,num_changes,
  min_lat,min_lon,max_lat,max_lon,comments_count,comment,editor,tags_json,seconds,error)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


def label_ids() -> list[int]:
    ids = []
    with LABELS.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            ids.append(int(line.split("\t")[0]))
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    db = data_path(*CORPUS_A_META_DB.split("/"))

    with store(db, "meta") as conn:
        if args.stats:
            print(json.dumps(counts(conn, "meta"), indent=2))
            return 0

        todo = label_ids()
        already = done_ids(conn, "meta", only_ok=args.retry_failed)
        todo = [c for c in todo if c not in already]
        if args.limit:
            todo = todo[: args.limit]

        print(f"database : {db}")
        print(f"to fetch : {len(todo)} ({len(already)} already stored)")
        if not todo:
            print(json.dumps(counts(conn, "meta"), indent=2))
            return 0

        session = PoliteSession(args.rate)
        work: queue.Queue = queue.Queue()
        for cid in todo:
            work.put(cid)

        rows: queue.Queue = queue.Queue()
        stop = threading.Event()

        def worker() -> None:
            while not stop.is_set():
                try:
                    cid = work.get_nowait()
                except queue.Empty:
                    return
                meta, r = fetch_changeset_meta(session, cid)
                if meta is not None:
                    rows.put((
                        cid, r.status, meta.created_at, meta.closed_at, meta.uid, meta.user,
                        meta.num_changes, meta.min_lat, meta.min_lon, meta.max_lat, meta.max_lon,
                        meta.comments_count, meta.comment, meta.editor,
                        json.dumps(meta.tags, ensure_ascii=False), round(r.seconds, 2), r.error,
                    ))
                else:
                    rows.put((cid, r.status, None, None, None, None, None, None, None, None,
                              None, None, None, None, None, round(r.seconds, 2), r.error))

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
        for t in threads:
            t.start()

        started, written = time.monotonic(), 0
        try:
            while any(t.is_alive() for t in threads) or not rows.empty():
                try:
                    row = rows.get(timeout=1.0)
                except queue.Empty:
                    continue
                conn.execute(INSERT, row)
                written += 1
                if written % 100 == 0:
                    rate = written / max(1e-6, time.monotonic() - started)
                    left = (len(todo) - written) / max(1e-6, rate)
                    print(
                        f"{written}/{len(todo)}  {rate:.1f}/s  eta {left/60:.0f} min",
                        flush=True,
                    )
        except KeyboardInterrupt:
            stop.set()
            print("\ninterrupted - progress is stored, rerun to resume")

        print(json.dumps(counts(conn, "meta"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
