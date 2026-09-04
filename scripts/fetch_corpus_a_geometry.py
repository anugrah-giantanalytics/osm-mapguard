"""Corpus A geometry enrichment - the production version of Gate 1a.

Reads the metadata store, asks Overpass for before/after geometry per changeset, stores
the parsed actions and a changeset-level summary. This is what makes branch B1 run on
the labelled 2014-2019 benchmark instead of only on an unlabelled modern corpus.

Single-threaded on purpose. Overpass runs two slots per client and this is a shared
public service; the fetch is slow because being fast here would be antisocial. Gate 1b
measures whether the full corpus is affordable at this rate, and the answer may be
"subset it" - if so, subset by changeset id, never by size or user, or the subset
correlates with the label.

Resumable: every changeset is recorded, including the ones the guard skipped, so a
restart picks up where it stopped and the coverage number stays measured.

    python scripts/fetch_corpus_a_geometry.py --limit 50     # smoke test
    python scripts/fetch_corpus_a_geometry.py                # full run
    python scripts/fetch_corpus_a_geometry.py --stats        # progress only
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapguard.config import (
    CORPUS_A_GEOM_DB,
    CORPUS_A_META_DB,
    OVERPASS_RATE,
    data_path,
)
from mapguard.data.http import PoliteSession
from mapguard.data.osm_api import ChangesetMeta
from mapguard.data.overpass import fetch_geometry
from mapguard.data.store import counts, done_ids, store
from mapguard.features.geometry import summarise

INSERT = """
INSERT OR REPLACE INTO geometry
 (cid, route, note, n_actions, n_with_old, n_way, json, seconds)
VALUES (?,?,?,?,?,?,?,?)
"""


def meta_rows(path: Path, limit: int | None) -> list[ChangesetMeta]:
    if not path.exists():
        raise SystemExit(f"no metadata store at {path} - run scripts/fetch_corpus_a_meta.py first")
    conn = sqlite3.connect(path)
    q = (
        "SELECT cid, created_at, closed_at, uid, user, num_changes,"
        " min_lat, min_lon, max_lat, max_lon, comments_count, tags_json"
        " FROM meta WHERE status = 200 ORDER BY cid"
    )
    rows = conn.execute(q).fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append(
            ChangesetMeta(
                cid=r[0], created_at=r[1], closed_at=r[2], uid=r[3], user=r[4],
                num_changes=r[5], min_lat=r[6], min_lon=r[7], max_lat=r[8], max_lon=r[9],
                comments_count=r[10], tags=json.loads(r[11]) if r[11] else {},
            )
        )
    return out[:limit] if limit else out


def action_payload(actions) -> str:
    """Store the parsed actions, not the raw XML.

    The XML is large and mostly other people's edits; the parsed actions are what the
    featuriser reads, and keeping them means a schema change does not force a refetch
    of a service we should not be hammering twice.
    """
    return json.dumps(
        [
            {
                "action": a.action,
                "otype": a.otype,
                "oid": a.oid,
                "old": asdict(a.old) if a.old else None,
                "new": asdict(a.new) if a.new else None,
            }
            for a in actions
        ],
        ensure_ascii=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rate", type=float, default=OVERPASS_RATE)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    geom_db = data_path(*CORPUS_A_GEOM_DB.split("/"))

    with store(geom_db, "geometry") as conn:
        if args.stats:
            c = counts(conn, "geometry")
            summed = conn.execute(
                "SELECT SUM(n_actions), SUM(n_with_old), SUM(n_way) FROM geometry"
            ).fetchone()
            c["actions_total"], c["with_old_total"], c["ways_total"] = summed
            print(json.dumps(c, indent=2))
            return 0

        metas = meta_rows(data_path(*CORPUS_A_META_DB.split("/")), args.limit)
        already = done_ids(conn, "geometry")
        todo = [m for m in metas if m.cid not in already]

        print(f"database : {geom_db}")
        print(f"to fetch : {len(todo)} ({len(already)} already stored)")
        if not todo:
            print(json.dumps(counts(conn, "geometry"), indent=2))
            return 0

        ovp = PoliteSession(args.rate, max_retries=3)
        started = time.monotonic()

        for n, meta in enumerate(todo, 1):
            t0 = time.monotonic()
            actions, resp, note = fetch_geometry(ovp, meta)
            route = "adiff" if actions else ("skipped" if note.startswith("skipped") else "adiff")
            summary = summarise(actions)
            conn.execute(
                INSERT,
                (
                    meta.cid,
                    route,
                    note,
                    len(actions),
                    summary.get("n_with_old", 0),
                    summary.get("n_way", 0),
                    action_payload(actions),
                    round(time.monotonic() - t0, 2),
                ),
            )
            if n % 25 == 0 or n == len(todo):
                rate = n / max(1e-6, time.monotonic() - started)
                print(
                    f"{n}/{len(todo)}  {rate*60:.1f}/min  eta {(len(todo)-n)/max(1e-6,rate)/3600:.1f} h",
                    flush=True,
                )

        print(json.dumps(counts(conn, "geometry"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
