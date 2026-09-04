"""Corpus D - harvest live changeset metadata from the replication stream.

PROJECT_PLAN v1 said current OSM data was reachable "only through the OSM API or
Overpass, one changeset at a time", and so the temporal test could only ever be a demo.
The changeset replication stream is one gzipped file per minute, roughly 3 KB, live to
the current minute, free and account-free. A month of it is about 125 MB.

That makes R3b - train on 2014-2019, test on live 2026 - a real evaluation rather than a
demo, and it is the direct answer to "use the latest OSM data".

This script harvests a window and stores the metadata. Contents and geometry come later,
per changeset, for whatever sample the evaluation needs.

    python scripts/harvest_corpus_d.py --minutes 60          # last hour
    python scripts/harvest_corpus_d.py --minutes 10 --stats-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapguard.config import DEFAULT_RATE, RESULTS, data_path
from mapguard.data.http import PoliteSession
from mapguard.data.replication import changeset_state, fetch_changeset_sequence, minute_state
from mapguard.data.store import store

CORPUS_D_DB = "raw/corpus_d.sqlite"

INSERT = """
INSERT OR REPLACE INTO meta
 (cid,status,created_at,closed_at,uid,user,num_changes,
  min_lat,min_lon,max_lat,max_lon,comments_count,comment,editor,tags_json,seconds,error)
VALUES (?,200,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)
"""

# Patterns that mark a changeset as a revert. Revert mining is one of the two label
# routes for Corpus B and D (PROJECT_PLAN section 8.5) and needs no account, unlike
# OSMCha. This is a first pass, not the final rule - the real one gets validated
# against the OSMCha harmful set in CHECKLIST 6.2.3.
REVERT_PATTERNS = ("revert", "reverting", "undo changeset", "mass revert", "vandalism")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=30, help="how many minute-files to pull")
    ap.add_argument("--stats-only", action="store_true", help="report stream freshness and stop")
    args = ap.parse_args()

    session = PoliteSession(DEFAULT_RATE)

    cs_state = changeset_state(session)
    mi_state = minute_state(session)
    if cs_state is None:
        print("could not read the changeset replication state")
        return 2

    print(f"changeset stream : sequence {cs_state.sequence}  last_run {cs_state.timestamp}")
    if mi_state:
        print(f"minute stream    : sequence {mi_state.sequence}  timestamp {mi_state.timestamp}")
    if args.stats_only:
        return 0

    db = data_path(*CORPUS_D_DB.split("/"))
    started = time.monotonic()
    stored = 0
    reverts = 0
    editors: dict[str, int] = {}
    oldest = newest = None

    with store(db, "meta") as conn:
        for i in range(args.minutes):
            seq = cs_state.sequence - i
            for r in fetch_changeset_sequence(session, seq):
                conn.execute(
                    INSERT,
                    (
                        r.cid, r.created_at, r.closed_at, r.uid, r.user, r.num_changes,
                        r.min_lat, r.min_lon, r.max_lat, r.max_lon, r.comments_count,
                        r.comment, r.editor, json.dumps(r.tags, ensure_ascii=False),
                    ),
                )
                stored += 1
                low = r.comment.lower()
                if any(p in low for p in REVERT_PATTERNS):
                    reverts += 1
                editors[r.editor or "(none)"] = editors.get(r.editor or "(none)", 0) + 1
                if r.created_at:
                    oldest = min(oldest or r.created_at, r.created_at)
                    newest = max(newest or r.created_at, r.created_at)
            if (i + 1) % 10 == 0:
                print(f"{i+1}/{args.minutes} minutes, {stored} changesets", flush=True)

        total = conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0]

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "corpus": "D",
        "source": "planet.openstreetmap.org/replication/changesets",
        "head_sequence": cs_state.sequence,
        "head_last_run": cs_state.timestamp,
        "minute_stream_head": mi_state.timestamp if mi_state else None,
        "minutes_harvested": args.minutes,
        "changesets_stored_this_run": stored,
        "changesets_in_store": total,
        "window_oldest_created_at": oldest,
        "window_newest_created_at": newest,
        "revert_comment_matches": reverts,
        "revert_rate": round(reverts / max(1, stored), 5),
        "top_editors": dict(sorted(editors.items(), key=lambda kv: -kv[1])[:10]),
        "seconds": round(time.monotonic() - started, 1),
        "database": str(db),
    }

    out = RESULTS / "corpus_d_harvest.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n--- Corpus D harvest ---")
    print(json.dumps(summary, indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
