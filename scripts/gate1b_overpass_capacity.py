"""Gate 1b - will the public Overpass instance carry 18,276 adiff queries?

Gate 1a measures yield on a sample. This measures whether the sample rate survives being
sustained: Overpass runs two slots per client, and a study that quietly saturates a
volunteer-run service is both a technical and an ethical failure.

Run it on its own. Running it alongside Gate 1a measures the two jobs competing with
each other rather than the service.

Pass criterion (PROJECT_PLAN section 8.4): a written decision - full corpus, a mirror,
or a documented subset. The subset rule must not correlate with the label, so sample by
changeset id, never by size or user.

    python scripts/gate1b_overpass_capacity.py --minutes 30
    python scripts/gate1b_overpass_capacity.py --minutes 5    # quick look
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapguard.config import DEFAULT_RATE, LABELS, OVERPASS_RATE, RESULTS
from mapguard.data.http import PoliteSession
from mapguard.data.osm_api import fetch_changeset_meta
from mapguard.data.overpass import fetch_geometry, should_use_adiff

CORPUS_SIZE = 18_276


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--rate", type=float, default=OVERPASS_RATE)
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    ids = []
    with LABELS.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            ids.append(int(line.split("\t")[0]))
    random.Random(args.seed).shuffle(ids)

    api = PoliteSession(DEFAULT_RATE)
    ovp = PoliteSession(args.rate, max_retries=2)

    deadline = time.monotonic() + args.minutes * 60
    statuses: Counter = Counter()
    latencies: list[float] = []
    attempted = 0
    notes: Counter = Counter()

    for cid in ids:
        if time.monotonic() > deadline:
            break
        meta, _ = fetch_changeset_meta(api, cid)
        if meta is None:
            statuses["meta_fail"] += 1
            continue
        ok, why = should_use_adiff(meta)
        if not ok:
            notes[why.split("(")[0].strip()] += 1
            continue

        t0 = time.monotonic()
        actions, resp, note = fetch_geometry(ovp, meta)
        attempted += 1
        latencies.append(time.monotonic() - t0)
        statuses[str(resp.status) if resp else "none"] += 1
        notes[note] += 1
        if attempted % 10 == 0:
            elapsed = args.minutes * 60 - (deadline - time.monotonic())
            print(
                f"{attempted} queries in {elapsed/60:.1f} min "
                f"({attempted/max(1e-6, elapsed/60):.1f}/min) statuses={dict(statuses)}",
                flush=True,
            )

    elapsed_min = args.minutes - max(0.0, (deadline - time.monotonic()) / 60)
    per_min = attempted / max(1e-6, elapsed_min)
    hours_for_corpus = (CORPUS_SIZE / max(1e-6, per_min)) / 60

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gate": "1b",
        "question": "will the public Overpass instance carry 18,276 adiff queries?",
        "minutes_run": round(elapsed_min, 2),
        "queries_attempted": attempted,
        "queries_per_minute": round(per_min, 2),
        "statuses": dict(statuses),
        "notes": dict(notes),
        "latency_seconds": {
            "median": round(statistics.median(latencies), 2) if latencies else None,
            "p90": round(sorted(latencies)[int(0.9 * (len(latencies) - 1))], 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
        "projected_hours_for_full_corpus": round(hours_for_corpus, 1),
        "error_rate": round(
            sum(v for k, v in statuses.items() if k not in ("200",)) / max(1, attempted), 3
        ),
    }
    # A rate that finishes the corpus inside a working week, with a low error rate, is
    # enough. Anything slower needs a mirror or a documented subset.
    summary["verdict"] = (
        "PASS" if hours_for_corpus <= 40 and summary["error_rate"] < 0.1 else "SUBSET_OR_MIRROR"
    )
    summary["decision_required"] = (
        "full corpus is affordable"
        if summary["verdict"] == "PASS"
        else "pick a mirror instance, or subset by changeset id (never by size or user)"
    )

    out = RESULTS / "gate1b_overpass_capacity.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n--- Gate 1b ---")
    print(json.dumps(summary, indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
