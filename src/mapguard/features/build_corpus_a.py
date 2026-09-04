"""Corpus A featuriser - SQLite of osmChange XML -> two Parquet tables.

    changesets.parquet   one row per changeset: label, tabular features, text blob
    edits.parquet        one row per edit, in commit order

Why two tables. The B5 tabular branch and the classical baselines want a flat row
per changeset. The B3 edit-sequence BiLSTM and every Stage-1 pretext task want the
edits themselves. Writing both once, from a single pass, avoids re-parsing 18k XML
documents every time a branch changes.

The feature set is deliberately a superset of Ovid's (PROJECT_PLAN.md §3): the
same counts and version statistics, so the reproduction is honest, plus the
node-geometry and name-text signals Ovid discards - gaps G1 and G2. Whether they
help is the ablation's job, not this module's.

`uid` is carried through because the R1 split must be user-disjoint. A random split
leaks: the same vandal appears in train and test, and the model learns the account
rather than the edit.

Usage:
    python -m mapguard.features.build_corpus_a
    python -m mapguard.features.build_corpus_a --db /content/corpus_a.sqlite \
        --out /content/drive/MyDrive/mapguard/interim
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from mapguard.data.osmchange import Edit, parse_safe

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "data" / "raw" / "corpus_a.sqlite"
OUT = ROOT / "data" / "interim"

# Keys Ovid tracks as its "top-12", plus the ones vandalism actually lands on.
# Presence indicators, not counts - a changeset either touches highways or it does not.
WATCH_KEYS = (
    "highway", "building", "name", "amenity", "natural", "place", "landuse",
    "source", "waterway", "barrier", "surface", "addr:housenumber",
)

# Tag keys whose *values* are free text a human wrote. These feed the B2 char-CNN;
# `highway=residential` is a vocabulary item, `name=<slur>` is the thing we must read.
TEXT_KEYS = ("name", "description", "note", "comment", "operator", "brand", "inscription")


def _is_text_key(k: str) -> bool:
    return k in TEXT_KEYS or k.startswith("name:") or k.startswith("alt_name")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _parse_timestamp(ts: str) -> float | None:
    """OSM timestamps are ISO-8601 Zulu. datetime.fromisoformat handles them on 3.11+."""
    if not ts:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def changeset_features(cid: int, label: int, edits: list[Edit]) -> dict:
    n = len(edits)
    acts = Counter(e.action for e in edits)
    types = Counter(e.otype for e in edits)

    versions = [e.version for e in edits if e.version]
    n_tags = [len(e.tags) for e in edits]
    all_keys: Counter[str] = Counter()
    for e in edits:
        all_keys.update(e.tags.keys())

    # --- node geometry. Ways give <nd ref> only, so this is nodes-only by design.
    coords = [(e.lat, e.lon) for e in edits if e.lat is not None and e.lon is not None]
    if coords:
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        lat_c, lon_c = sum(lats) / len(lats), sum(lons) / len(lons)
        bbox = (min(lats), min(lons), max(lats), max(lons))
        bbox_diag_km = haversine_km(bbox[0], bbox[1], bbox[2], bbox[3])
        max_from_centroid = max(haversine_km(lat_c, lon_c, la, lo) for la, lo in coords)
        # A changeset whose nodes span thousands of km is either an import, a bot,
        # or a node dragged across the planet. All three are worth flagging.
        geom = {
            "n_coords": len(coords),
            "lat_centroid": lat_c,
            "lon_centroid": lon_c,
            "bbox_lat_span": bbox[2] - bbox[0],
            "bbox_lon_span": bbox[3] - bbox[1],
            "bbox_diag_km": bbox_diag_km,
            "max_dist_from_centroid_km": max_from_centroid,
        }
    else:
        geom = {
            "n_coords": 0, "lat_centroid": None, "lon_centroid": None,
            "bbox_lat_span": None, "bbox_lon_span": None,
            "bbox_diag_km": None, "max_dist_from_centroid_km": None,
        }

    # --- time
    stamps = [t for t in (_parse_timestamp(e.timestamp) for e in edits) if t is not None]
    duration_s = (max(stamps) - min(stamps)) if len(stamps) > 1 else 0.0

    # --- free text a human typed
    texts: list[str] = []
    for e in edits:
        for k, v in e.tags.items():
            if _is_text_key(k) and v:
                texts.append(v)
    # Newline as the separator: ASCII, printable on any console, and OSM tag values
    # cannot contain one - so the join stays unambiguously reversible.
    text_blob = "\n".join(texts)

    uids = [e.uid for e in edits if e.uid is not None]
    users = [e.user for e in edits if e.user]

    row = {
        "changeset": cid,
        "label": label,
        "uid": uids[0] if uids else None,
        "user": users[0] if users else None,
        "n_edits": n,
        "n_create": acts["create"],
        "n_modify": acts["modify"],
        "n_delete": acts["delete"],
        "n_node": types["node"],
        "n_way": types["way"],
        "n_relation": types["relation"],
        "frac_create": acts["create"] / n if n else 0.0,
        "frac_delete": acts["delete"] / n if n else 0.0,
        "version_max": max(versions) if versions else 0,
        "version_mean": sum(versions) / len(versions) if versions else 0.0,
        "frac_v1": sum(1 for v in versions if v == 1) / len(versions) if versions else 0.0,
        "n_tags_total": sum(n_tags),
        "n_tags_mean": sum(n_tags) / n if n else 0.0,
        "n_tags_max": max(n_tags) if n_tags else 0,
        "n_distinct_keys": len(all_keys),
        "n_refs_total": sum(e.n_refs for e in edits),
        "duration_s": duration_s,
        "edits_per_min": n / (duration_s / 60) if duration_s > 0 else float(n),
        "n_text_values": len(texts),
        "text_len_total": sum(len(t) for t in texts),
        "text_len_max": max((len(t) for t in texts), default=0),
        "text": text_blob,
        **geom,
    }
    for k in WATCH_KEYS:
        row[f"key_{k.replace(':', '_')}"] = int(all_keys[k] > 0)
    return row


def edit_records(cid: int, label: int, edits: list[Edit]) -> list[dict]:
    """Per-edit rows for the sequence branch and the Stage-1 pretext tasks."""
    out = []
    for i, e in enumerate(edits):
        out.append({
            "changeset": cid,
            "label": label,
            "seq": i,
            "action": e.action,
            "otype": e.otype,
            "oid": e.oid,
            "version": e.version,
            "timestamp": e.timestamp,
            "uid": e.uid,
            "lat": e.lat,
            "lon": e.lon,
            "n_tags": len(e.tags),
            "n_refs": e.n_refs,
            "tags": json.dumps(e.tags, ensure_ascii=False),
        })
    return out


def build_tables(db: Path, out: Path, max_edits_per_changeset: int) -> dict:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, vandalism, xml FROM changeset WHERE status = 200 ORDER BY id"
    )

    cs_rows: list[dict] = []
    ed_rows: list[dict] = []
    n_parse_fail = 0
    n_truncated = 0

    for r in rows:
        edits, err = parse_safe(r["xml"])
        if err:
            n_parse_fail += 1
            continue
        cs_rows.append(changeset_features(r["id"], r["vandalism"], edits))
        # Ovid zeroes the edit vector above a threshold (gap G4); we truncate the
        # *stored sequence* instead so the changeset-level features stay intact and
        # the BiLSTM still sees the head of a mass edit rather than nothing.
        keep = edits[:max_edits_per_changeset]
        if len(keep) < len(edits):
            n_truncated += 1
        ed_rows.extend(edit_records(r["id"], r["vandalism"], keep))

    con.close()

    if not cs_rows:
        raise SystemExit("no fetched changesets in the database yet - run the fetcher")

    # Imported here, not at module scope, so the feature functions above stay
    # stdlib-only and testable on the laptop's Python 3.14, which has no polars wheel.
    import polars as pl

    out.mkdir(parents=True, exist_ok=True)
    cs = pl.DataFrame(cs_rows)
    ed = pl.DataFrame(ed_rows)
    cs.write_parquet(out / "changesets.parquet")
    ed.write_parquet(out / "edits.parquet")

    summary = {
        "changesets": cs.height,
        "edits": ed.height,
        "vandalism": int(cs["label"].sum()),
        "benign": int((cs["label"] == 0).sum()),
        "distinct_users": int(cs["uid"].n_unique()),
        "parse_failures": n_parse_fail,
        "sequences_truncated": n_truncated,
        "median_edits": float(cs["n_edits"].median()),
        "max_edits": int(cs["n_edits"].max()),
        "with_node_coords": int((cs["n_coords"] > 0).sum()),
        "out_dir": str(out),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-edits", type=int, default=512,
                    help="cap on stored per-changeset edit sequence length")
    args = ap.parse_args()

    summary = build_tables(args.db, args.out, args.max_edits)
    print(json.dumps(summary, indent=2))

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "featurise_corpus_a.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
