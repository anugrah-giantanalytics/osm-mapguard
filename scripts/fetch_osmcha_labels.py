"""Human-reviewed labels for Corpus B and D, from OSMCha.

Closes the open item in PROJECT_PLAN section 8.5 and CHECKLIST 6.2.1. Two classes, both
of them reviewed by a person:

    harmful=True                 damaging
    checked=True, harmful=False  a reviewer looked and said it is fine

Fetching only the harmful class would leave us assuming unreviewed changesets are clean,
which quietly encodes the reviewers' attention patterns into the negative class. Reviewed
negatives are scarcer and much better.

Windows matter. An unbounded harmful query times out - the endpoint has to scan. Pass a
date range and page through it.

    python scripts/fetch_osmcha_labels.py --since 2026-08-01
    python scripts/fetch_osmcha_labels.py --since 2024-02-01 --until 2025-03-31  # Corpus B
    python scripts/fetch_osmcha_labels.py --stats
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapguard.config import RESULTS, data_path
from mapguard.data.osmcha import count_for, iter_labels_windowed, session
from mapguard.data.store import counts, store

LABELS_DB = "raw/osmcha_labels.sqlite"

INSERT = """
INSERT OR REPLACE INTO labels
 (cid,harmful,checked,check_user,check_date,date,uid,user,editor,comment,
  n_create,n_modify,n_delete,is_suspect,reasons,reviewed_features,source)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'osmcha')
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-08-01", help="date__gte, YYYY-MM-DD")
    ap.add_argument("--until", default=None, help="date__lte, YYYY-MM-DD")
    ap.add_argument("--window-days", type=int, default=7,
                    help="date slice width. Offset pagination on this endpoint dies "
                         "around page 30, so we walk dates instead")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--no-count", action="store_true",
                    help="skip the upfront availability counts. Over a multi-year window "
                         "those are full scans that time out and retry before any real "
                         "work starts; results/gate2a_osmcha_coverage.json already has them")
    args = ap.parse_args()

    db = data_path(*LABELS_DB.split("/"))

    with store(db, "labels", "fetch_gaps") as conn:
        if args.stats:
            c = counts(conn, "labels")
            c["distinct_reviewers"] = conn.execute(
                "SELECT COUNT(DISTINCT check_user) FROM labels WHERE check_user IS NOT NULL"
            ).fetchone()[0]
            c["date_range"] = conn.execute(
                "SELECT MIN(date), MAX(date) FROM labels"
            ).fetchone()
            print(json.dumps(c, indent=2))
            return 0

        sess = session()
        started = time.monotonic()
        until = args.until or datetime.now().strftime("%Y-%m-%d")

        totals = {}
        if args.no_count:
            totals = {"harmful": None, "clean": None,
                      "note": "skipped; see results/gate2a_osmcha_coverage.json"}
        else:
            for harmful in (True, False):
                totals["harmful" if harmful else "clean"] = count_for(
                    sess, harmful=harmful, date_gte=args.since, date_lte=until
                )
        print(f"window {args.since} .. {until}  (slices of {args.window_days} days)")
        print(f"available: {totals}")

        stored = {"harmful": 0, "clean": 0}
        for harmful in (True, False):
            key = "harmful" if harmful else "clean"
            print(f"\nfetching {key}...")
            def note_gap(w_from, w_to, klass, status):
                conn.execute(
                    "INSERT INTO fetch_gaps (source,window_from,window_to,klass,status)"
                    " VALUES ('osmcha',?,?,?,?)"
                    " ON CONFLICT(source,window_from,window_to,klass)"
                    " DO UPDATE SET attempts = attempts + 1, resolved = 0",
                    (w_from, w_to, klass, status))

            def note_done(w_from, w_to, klass):
                conn.execute(
                    "INSERT INTO fetch_gaps (source,window_from,window_to,klass,status,resolved)"
                    " VALUES ('osmcha',?,?,?,200,1)"
                    " ON CONFLICT(source,window_from,window_to,klass)"
                    " DO UPDATE SET resolved = 1, status = 200",
                    (w_from, w_to, klass))

            # Windows already fully covered by a previous run are skipped, so a restart
            # resumes rather than re-walking three years of slices.
            done_windows = {
                (a, b) for a, b in conn.execute(
                    "SELECT window_from, window_to FROM fetch_gaps"
                    " WHERE resolved = 1 AND klass = ?", (key,))
            }
            if done_windows:
                print(f"  resuming: {len(done_windows)} {key} windows already complete")

            for n, lab in enumerate(
                iter_labels_windowed(
                    sess,
                    harmful=harmful,
                    date_gte=args.since,
                    date_lte=until,
                    window_days=args.window_days,
                    on_gap=note_gap,
                    on_window_done=note_done,
                    skip_windows=done_windows,
                ),
                1,
            ):
                conn.execute(
                    INSERT,
                    (
                        lab.cid,
                        1 if lab.harmful else 0,
                        1 if lab.checked else 0,
                        lab.check_user, lab.check_date, lab.date, lab.uid, lab.user,
                        lab.editor, lab.comment, lab.n_create, lab.n_modify, lab.n_delete,
                        1 if lab.is_suspect else 0, lab.reasons, lab.reviewed_features,
                    ),
                )
                stored[key] = n
                if n % 200 == 0:
                    print(f"  {key}: {n}", flush=True)
            print(f"  {key}: {stored[key]} stored")

        gaps = counts(conn, "fetch_gaps")
        final = counts(conn, "labels")
        reviewers = conn.execute(
            "SELECT COUNT(DISTINCT check_user) FROM labels WHERE check_user IS NOT NULL"
        ).fetchone()[0]
        top_reasons = conn.execute(
            "SELECT reasons, COUNT(*) FROM labels WHERE harmful = 1 AND reasons != ''"
            " GROUP BY reasons ORDER BY 2 DESC LIMIT 10"
        ).fetchall()

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "osmcha.org/api/v1/changesets",
        "window": {"since": args.since, "until": until, "slice_days": args.window_days},
        "available_in_window": totals,
        "stored_this_run": stored,
        "store_totals": final,
        "fetch_gaps": gaps,
        "distinct_reviewers": reviewers,
        "top_harmful_reasons": dict(top_reasons),
        "seconds": round(time.monotonic() - started, 1),
        "database": str(db),
        "note": (
            "harmful=True is a human calling a changeset damaging. harmful=False is sent "
            "with checked=True so negatives are reviewed, not merely unexamined - "
            "otherwise the negative class encodes reviewer attention."
        ),
    }
    out = RESULTS / "osmcha_labels.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n--- OSMCha labels ---")
    print(json.dumps(summary, indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
