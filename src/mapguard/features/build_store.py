"""Build the feature store: five SQLite sources in, three Parquet tables out.

    sources                                  tables
    -------                                  ------
    corpus_a_meta.sqlite     metadata        changesets.parquet   one row per changeset
    corpus_a.sqlite          osmChange XML   edits.parquet        one row per sampled edit
    corpus_a_geom.sqlite     adiff geometry  user_seq.parquet     one row per history step
    users.sqlite             account_created
    user_histories.sqlite    prior changesets

Two rules from PROJECT_PLAN section 5.4 are implemented here, and they are the reason
this module exists rather than a notebook cell.

**Rule 1 - nothing is zeroed.** Ovid discards the edit vector above a size threshold, so
mass vandalism and broken imports fall through (gap G4). Here the changeset-level summary
is computed over *every* edit - counts, op mix, tag-key histogram, geometry moments - and
only the *per-edit detail* is sampled, to K = 256, stratified so the extremes survive.
`sample_ratio` goes in as a feature so the model knows how much it is not seeing.

**Rule 2 - user features are point-in-time.** Every user aggregate and every sequence step
comes from `mapguard.features.user_history`, which filters to changesets strictly before
the target. See tests/test_user_history.py.

Memory: written with a ParquetWriter and flushed in row groups. The development laptop
has under 1 GB free, and the edits table is a few million rows.

    python -m mapguard.features.build_store
    python -m mapguard.features.build_store --limit 500 --out data/interim
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mapguard.config import (
    CORPUS_A_DB,
    CORPUS_A_GEOM_DB,
    CORPUS_A_META_DB,
    LABELS,
    data_path,
    interim_dir,
)
from mapguard.data.osmchange import Edit, parse_safe
from mapguard.data.overpass import GeometryAction, GeometryState
from mapguard.data.users import account_age_days
from mapguard.features.geometry import features_for, summarise
from mapguard.features.user_history import PriorChangeset, build as build_user

USERS_DB = "raw/users.sqlite"
HISTORY_DB = "raw/user_histories.sqlite"

# Ovid tracks a "top-12" key list; these are those plus the keys vandalism actually lands
# on. Presence indicators, not counts - a changeset either touches highways or it does not.
WATCH_KEYS = (
    "highway", "building", "name", "amenity", "natural", "place", "landuse",
    "source", "waterway", "barrier", "surface", "addr:housenumber",
)

# Tag keys whose values are free text a human wrote. These feed the B2 char-CNN:
# `highway=residential` is a vocabulary item, `name=<slur>` is the thing we must read.
TEXT_KEYS = ("name", "description", "note", "comment", "operator", "brand", "inscription")

SAMPLE_K = 256
BATCH_ROWS = 2000
SCHEMA_VERSION = 1


def is_text_key(k: str) -> bool:
    return k in TEXT_KEYS or k.startswith("name:") or k.startswith("alt_name")


# --------------------------------------------------------------------------- sources


def load_labels() -> dict[int, int]:
    out = {}
    with LABELS.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            cid, label = line.rstrip("\n").split("\t")
            out[int(cid)] = 1 if label.strip().lower() == "true" else 0
    return out


def open_ro(path: Path) -> sqlite3.Connection | None:
    return sqlite3.connect(path) if path.exists() else None


def geometry_actions(blob: str | None) -> list[GeometryAction]:
    """Rebuild GeometryAction objects from the stored JSON."""
    if not blob:
        return []
    try:
        raw = json.loads(blob)
    except ValueError:
        return []
    out = []
    for a in raw:
        def state(d):
            if not d:
                return None
            return GeometryState(
                otype=d["otype"], oid=d["oid"], version=d["version"],
                coords=[tuple(c) for c in d.get("coords", [])], tags=d.get("tags", {}),
            )
        out.append(
            GeometryAction(
                action=a["action"], otype=a["otype"], oid=a["oid"],
                changeset=None, old=state(a.get("old")), new=state(a.get("new")),
            )
        )
    return out


def prior_changesets(blob: str | None) -> list[PriorChangeset]:
    if not blob:
        return []
    try:
        raw = json.loads(blob)
    except ValueError:
        return []
    return [PriorChangeset(**r) for r in raw]


# ------------------------------------------------------------------- edit sampling


def stratified_sample(edits: list[Edit], geom_by_key: dict, k: int = SAMPLE_K) -> list[int]:
    """Indices of the edits to encode per-edit, keeping the informative extremes.

    Priority: largest geometric displacement, largest area change, every edit carrying
    free text, then random fill in commit order. Truncating by position instead would
    drop exactly the objects a reviewer would look at first.
    """
    n = len(edits)
    if n <= k:
        return list(range(n))

    shift, area, text = [], [], []
    for i, e in enumerate(edits):
        g = geom_by_key.get((e.otype, e.oid))
        if g is not None:
            shift.append((g.centroid_shift_m, i))
            area.append((abs(g.d_area_m2), i))
        if any(is_text_key(key) for key in e.tags):
            text.append(i)

    chosen: list[int] = []
    seen: set[int] = set()

    def add(indices):
        for i in indices:
            if i not in seen and len(chosen) < k:
                seen.add(i)
                chosen.append(i)

    add(i for _, i in sorted(shift, reverse=True)[: k // 4])
    add(i for _, i in sorted(area, reverse=True)[: k // 4])
    add(text[: k // 4])
    # Deterministic fill: evenly spaced through commit order, so the sample still
    # describes the shape of the changeset rather than only its extremes.
    if len(chosen) < k:
        step = max(1, n // max(1, (k - len(chosen))))
        add(range(0, n, step))
    add(range(n))
    return sorted(chosen)


# ------------------------------------------------------------------------ features


def changeset_row(
    cid: int,
    label: int | None,
    meta: dict,
    edits: list[Edit],
    actions: list[GeometryAction],
    account_created: str | None,
    history: list[PriorChangeset],
) -> dict:
    """One row of changesets.parquet. Summaries here never truncate (Rule 1)."""
    acts = Counter(e.action for e in edits)
    types = Counter(e.otype for e in edits)
    versions = [e.version for e in edits if e.version]
    keys: Counter[str] = Counter()
    for e in edits:
        keys.update(e.tags.keys())

    text_values = [
        v for e in edits for k, v in e.tags.items() if is_text_key(k) and v
    ]

    geom = summarise(actions)
    uf = build(meta.get("uid"), history, meta.get("created_at") or "")

    row = {
        "cid": cid,
        "label": label,
        "schema_version": SCHEMA_VERSION,
        # --- metadata
        "created_at": meta.get("created_at"),
        "uid": meta.get("uid"),
        "num_changes_meta": meta.get("num_changes"),
        "comment": (meta.get("comment") or "")[:500],
        "comment_length": len(meta.get("comment") or ""),
        "editor": meta.get("editor") or "",
        "comments_count": meta.get("comments_count"),
        "bbox_lat_span": (
            (meta["max_lat"] - meta["min_lat"])
            if meta.get("max_lat") is not None and meta.get("min_lat") is not None else None
        ),
        "bbox_lon_span": (
            (meta["max_lon"] - meta["min_lon"])
            if meta.get("max_lon") is not None and meta.get("min_lon") is not None else None
        ),
        "account_age_days": account_age_days(account_created, meta.get("created_at")),
        # --- exact edit statistics, over every edit
        "n_edits": len(edits),
        "n_create": acts.get("create", 0),
        "n_modify": acts.get("modify", 0),
        "n_delete": acts.get("delete", 0),
        "frac_delete": round(acts.get("delete", 0) / max(1, len(edits)), 4),
        "n_node": types.get("node", 0),
        "n_way": types.get("way", 0),
        "n_relation": types.get("relation", 0),
        "max_version": max(versions) if versions else 0,
        "mean_version": round(sum(versions) / len(versions), 3) if versions else 0.0,
        "n_v1": sum(1 for v in versions if v == 1),
        "total_tags": sum(len(e.tags) for e in edits),
        "mean_tags": round(sum(len(e.tags) for e in edits) / max(1, len(edits)), 3),
        "distinct_keys": len(keys),
        "n_text_edits": sum(1 for e in edits if any(is_text_key(k) for k in e.tags)),
        # --- sampling, so the model knows what it is not seeing
        "n_edits_seen": min(len(edits), SAMPLE_K),
        "sample_ratio": round(min(1.0, SAMPLE_K / max(1, len(edits))), 4),
        # --- text for B2
        "text_blob": " ".join(text_values)[:2000],
        "text_chars": sum(len(v) for v in text_values),
    }
    for key in WATCH_KEYS:
        row[f"key_{key.replace(':', '_')}"] = 1 if keys.get(key) else 0
    for k, v in geom.items():
        row[f"geom_{k}"] = v
    for k, v in uf.as_dict().items():
        row[f"user_{k}"] = v
    row["user_seq_len"] = len(uf.sequence)
    return row


def edit_rows(cid: int, edits: list[Edit], actions: list[GeometryAction]) -> list[dict]:
    geom_by_key = {}
    for a in actions:
        geom_by_key[(a.otype, a.oid)] = features_for(a)

    keep = stratified_sample(edits, geom_by_key)
    rows = []
    for pos in keep:
        e = edits[pos]
        g = geom_by_key.get((e.otype, e.oid))
        text = " ".join(v for k, v in e.tags.items() if is_text_key(k) and v)
        rows.append({
            "cid": cid,
            "position": pos,
            "action": e.action,
            "otype": e.otype,
            "oid": e.oid,
            "version": e.version,
            "is_new_object": 1 if e.version == 1 else 0,
            "n_tags": len(e.tags),
            "n_refs": e.n_refs,
            "lat": e.lat,
            "lon": e.lon,
            "text": text[:200],
            "has_geometry": 1 if g else 0,
            "centroid_shift_m": g.centroid_shift_m if g else None,
            "hausdorff_m": g.hausdorff_m if g else None,
            "d_length_m": g.d_length_m if g else None,
            "d_area_m2": g.d_area_m2 if g else None,
            "length_ratio": g.length_ratio if g else None,
            "area_ratio": g.area_ratio if g else None,
            "d_points": g.d_points if g else None,
            "tags_added": g.tags_added if g else None,
            "tags_removed": g.tags_removed if g else None,
            "tags_changed": g.tags_changed if g else None,
        })
    return rows


def user_seq_rows(cid: int, uid: int | None, history: list[PriorChangeset], created_at: str) -> list[dict]:
    uf = build_user(uid, history, created_at)
    return [
        {"cid": cid, "step": i, "num_changes": a, "gap_days": b,
         "comment_length": c, "bbox_diag_deg": d}
        for i, (a, b, c, d) in enumerate(uf.sequence)
    ]


# We alias the import so `build` inside changeset_row means the user-feature builder.
build = build_user


# --------------------------------------------------------------------------- writer


class TableWriter:
    """ParquetWriter that infers its schema from the first batch and flushes in groups."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []
        self.writer: pq.ParquetWriter | None = None
        self.n = 0

    def add(self, rows: list[dict] | dict) -> None:
        if isinstance(rows, dict):
            self.rows.append(rows)
        else:
            self.rows.extend(rows)
        if len(self.rows) >= BATCH_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        else:
            table = table.cast(self.writer.schema)
        self.writer.write_table(table)
        self.n += len(self.rows)
        self.rows = []

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else interim_dir()
    out.mkdir(parents=True, exist_ok=True)

    labels = load_labels()
    meta_conn = open_ro(data_path(*CORPUS_A_META_DB.split("/")))
    if meta_conn is None:
        raise SystemExit("no metadata store - run scripts/fetch_corpus_a_meta.py first")
    contents_conn = open_ro(data_path(*CORPUS_A_DB.split("/")))
    geom_conn = open_ro(data_path(*CORPUS_A_GEOM_DB.split("/")))
    users_conn = open_ro(data_path(*USERS_DB.split("/")))
    hist_conn = open_ro(data_path(*HISTORY_DB.split("/")))

    account_created = (
        dict(users_conn.execute("SELECT uid, account_created FROM users")) if users_conn else {}
    )
    histories = (
        {r[0]: r[1] for r in hist_conn.execute("SELECT uid, json FROM histories")}
        if hist_conn else {}
    )

    cw = TableWriter(out / "changesets.parquet")
    ew = TableWriter(out / "edits.parquet")
    sw = TableWriter(out / "user_seq.parquet")

    q = (
        "SELECT cid, created_at, closed_at, uid, user, num_changes, min_lat, min_lon,"
        " max_lat, max_lon, comments_count, comment, editor FROM meta"
        " WHERE status = 200 ORDER BY cid"
    )
    cols = ["cid", "created_at", "closed_at", "uid", "user", "num_changes", "min_lat",
            "min_lon", "max_lat", "max_lon", "comments_count", "comment", "editor"]

    seen = with_contents = with_geometry = with_history = 0
    for row in meta_conn.execute(q):
        meta = dict(zip(cols, row))
        cid = meta["cid"]
        if args.limit and seen >= args.limit:
            break
        seen += 1

        edits: list[Edit] = []
        if contents_conn:
            got = contents_conn.execute(
                "SELECT xml FROM contents WHERE cid = ? AND status = 200", (cid,)
            ).fetchone()
            if got and got[0]:
                edits, _ = parse_safe(got[0])
                with_contents += 1

        actions: list[GeometryAction] = []
        if geom_conn:
            got = geom_conn.execute("SELECT json FROM geometry WHERE cid = ?", (cid,)).fetchone()
            if got:
                actions = geometry_actions(got[0])
                if actions:
                    with_geometry += 1

        history = prior_changesets(histories.get(meta["uid"]))
        if history:
            with_history += 1

        cw.add(changeset_row(
            cid, labels.get(cid), meta, edits, actions,
            account_created.get(meta["uid"]), history,
        ))
        if edits:
            ew.add(edit_rows(cid, edits, actions))
        if history:
            sw.add(user_seq_rows(cid, meta["uid"], history, meta["created_at"] or ""))

        if seen % 1000 == 0:
            print(f"{seen} changesets  (contents {with_contents}, geometry {with_geometry})",
                  flush=True)

    for w in (cw, ew, sw):
        w.close()

    report = {
        "schema_version": SCHEMA_VERSION,
        "changesets": cw.n,
        "edits": ew.n,
        "user_seq_steps": sw.n,
        "coverage": {
            "with_contents": with_contents,
            "with_geometry": with_geometry,
            "with_user_history": with_history,
            "labelled": sum(1 for c in labels if c),
        },
        "sample_k": SAMPLE_K,
        "out": str(out),
    }
    print(json.dumps(report, indent=2))
    (out / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
