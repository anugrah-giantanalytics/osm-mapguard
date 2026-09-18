"""Mine five years of reverts and a random changeset sample from the OSM changeset dump.

    python scripts/mine_changesets.py --since 2021-09-01          # resumable
    python scripts/mine_changesets.py --stats

Why this exists. The project needs labelled, *recent* data spanning several years, and:

  * OSMCha has no reviewed changesets before 2023 (results/gate2a_osmcha_coverage.json:
    zero for 2018-2022), so human-reviewed labels cover 3.7 years at most;
  * Ovid's benchmark labels are revert-derived and stop in 2019.

Reverts are recorded in changeset metadata (tool tags and comments), and the changeset dump
holds that metadata for every changeset. Mining it gives revert-derived labels with the
*same definition as the Ovid benchmark* for 2021-09 to 2026-09, which is what lets the
temporal evaluation separate "time passed" from "the label source changed".

Only the byte range covering `--since` onwards is downloaded, in parallel range requests,
and parsed as it streams (see mapguard.data.changeset_dump). Nothing but the results is
stored:

    raw/changeset_dump.sqlite
        source    exact dump file, size, Last-Modified - the dataset's provenance
        chunks    byte ranges done, so a restart resumes
        reverts   every revert changeset: tool, comment, targets, partial flag
        sample    a fixed 1-in-400 hash sample of all changesets (the negative pool)
        months    changesets per month, for prevalence and sampling weights

Politeness: the dump is served from S3 (planet.openstreetmap.org redirects there), which is
built for range requests; four parallel ranges is the default.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapguard.config import RESULTS, data_path  # noqa: E402
from mapguard.data import changeset_dump as cd  # noqa: E402

DB = "raw/changeset_dump.sqlite"
CHUNK = 16_000_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS source (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS chunks (
    start INTEGER PRIMARY KEY, end INTEGER, n_changesets INTEGER, n_reverts INTEGER,
    n_sample INTEGER, first_cid INTEGER, last_cid INTEGER, seconds REAL,
    done TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS reverts (
    cid INTEGER PRIMARY KEY, created_at TEXT, uid INTEGER, kind TEXT, created_by TEXT,
    comment TEXT, targets TEXT, partial INTEGER
);
CREATE TABLE IF NOT EXISTS sample (
    cid INTEGER PRIMARY KEY, created_at TEXT, closed_at TEXT, uid INTEGER, num_changes INTEGER,
    min_lat REAL, min_lon REAL, max_lat REAL, max_lon REAL, comments_count INTEGER,
    editor TEXT, comment TEXT, tags_json TEXT
);
CREATE TABLE IF NOT EXISTS months (month TEXT PRIMARY KEY, n INTEGER);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(data_path(*DB.split("/")), timeout=120, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def _f(x):
    try:
        return float(x) if x not in (None, "") else None
    except ValueError:
        return None


def work(args):
    url, start, end, size, since, until = args
    t0 = time.monotonic()
    out = cd.scan_chunk(url, start, end, size, since, until)
    out["seconds"] = time.monotonic() - t0
    out["start"], out["end"] = start, end
    out["reverts"] = [r.__dict__ for r in out["reverts"]]
    return out


def stats(conn) -> dict:
    q = lambda s: conn.execute(s).fetchall()  # noqa: E731
    return {
        "source": dict(q("SELECT k, v FROM source")),
        "chunks_done": q("SELECT COUNT(*), MIN(first_cid), MAX(last_cid) FROM chunks")[0],
        "reverts": q("SELECT COUNT(*) FROM reverts")[0][0],
        "reverts_by_kind": q("SELECT kind, COUNT(*) FROM reverts GROUP BY 1 ORDER BY 2 DESC"),
        "reverts_by_year": q("SELECT substr(created_at,1,4), COUNT(*) FROM reverts GROUP BY 1"),
        "partial_reverts": q("SELECT COUNT(*) FROM reverts WHERE partial = 1")[0][0],
        "sample": q("SELECT COUNT(*) FROM sample")[0][0],
        "changesets_seen": q("SELECT COALESCE(SUM(n), 0) FROM months")[0][0],
        "months": q("SELECT month, n FROM months ORDER BY month"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default="2021-09-01")
    ap.add_argument("--until", default=None, help="exclusive; default: today")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-chunks", type=int, default=None, help="smoke test")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    conn = connect()
    if args.stats:
        print(json.dumps(stats(conn), indent=1))
        return 0

    until = args.until or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    src = dict(conn.execute("SELECT k, v FROM source"))
    if src:
        info = {"url": src["url"], "bytes": int(src["bytes"]), "file": src["file"],
                "last_modified": src["last_modified"]}
        start = int(src["start_offset"])
        if (src["since"], src["until"]) != (args.since, until) and not args.until:
            until = src["until"]          # resume with the window the run started with
        print(f"resuming {info['file']} from its recorded start offset {start:,}")
    else:
        info = cd.resolve()
        print(f"dump: {info['file']}  {info['bytes'] / 1e9:.2f} GB  modified {info['last_modified']}")
        print(f"finding the byte offset where {args.since} starts...")
        start = cd.offset_for_date(info["url"], info["bytes"], args.since)
        for k, v in {**info, "start_offset": start, "since": args.since, "until": until,
                     "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}.items():
            conn.execute("INSERT OR REPLACE INTO source VALUES (?, ?)", (k, str(v)))

    size = info["bytes"]
    done = {r[0] for r in conn.execute("SELECT start FROM chunks")}
    chunks = [(s, min(size, s + CHUNK)) for s in range(start, size, CHUNK)]
    todo = [c for c in chunks if c[0] not in done]
    if args.limit_chunks:
        todo = todo[: args.limit_chunks]
    total_mb = sum(e - s for s, e in todo) / 1e6
    print(f"{len(chunks)} chunks of {CHUNK / 1e6:.0f} MB from {start / 1e9:.2f} GB; "
          f"{len(todo)} to do ({total_mb:,.0f} MB)", flush=True)

    t0, got_mb = time.monotonic(), 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, (info["url"], s, e, size, args.since, until)): (s, e)
                   for s, e in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            s, e = futures[fut]
            try:
                out = fut.result()
            except Exception as exc:
                print(f"  chunk {s:,} failed: {exc!r} - rerun to retry it", flush=True)
                continue
            conn.execute("BEGIN")
            for r in out["reverts"]:
                conn.execute("INSERT OR REPLACE INTO reverts VALUES (?,?,?,?,?,?,?,?)",
                             (r["cid"], r["created_at"], r["uid"], r["kind"], r["created_by"],
                              r["comment"], json.dumps(r["targets"]), int(r["partial"])))
            for x in out["sample"]:
                conn.execute("INSERT OR REPLACE INTO sample VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                             (x["cid"], x["created_at"], x["closed_at"], x["uid"], x["num_changes"],
                              _f(x["min_lat"]), _f(x["min_lon"]), _f(x["max_lat"]), _f(x["max_lon"]),
                              x["comments_count"], x["editor"], x["comment"],
                              json.dumps(x["tags"], ensure_ascii=False)))
            for month, k in out["months"].items():
                conn.execute("INSERT INTO months VALUES (?, ?) ON CONFLICT(month) DO UPDATE SET n = n + ?",
                             (month, k, k))
            conn.execute("INSERT INTO chunks (start,end,n_changesets,n_reverts,n_sample,first_cid,"
                         "last_cid,seconds) VALUES (?,?,?,?,?,?,?,?)",
                         (s, e, sum(out["months"].values()), len(out["reverts"]), len(out["sample"]),
                          out["first_cid"], out["last_cid"], round(out["seconds"], 1)))
            conn.execute("COMMIT")
            got_mb += (e - s) / 1e6
            rate = got_mb / max(1e-6, time.monotonic() - t0)
            print(f"  {n}/{len(todo)} chunks  {rate:.2f} MB/s  eta {(total_mb - got_mb) / max(rate, 1e-6) / 60:.0f} min"
                  f"  last cid {out['last_cid']}", flush=True)

    summary = stats(conn)
    summary.pop("months")
    (RESULTS / "changeset_dump_mining.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
