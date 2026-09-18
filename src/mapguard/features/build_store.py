"""Build the feature store: raw SQLite stores in, three Parquet tables out.

    sources (under MAPGUARD_DATA/raw)                      tables (under MAPGUARD_DATA/interim)
    ---------------------------------                      ------------------------------------
    corpus_a_meta.sqlite   A metadata                      changesets.parquet  one row per changeset
    corpus_a.sqlite        A osmChange (table changeset)   edits.parquet       one row per sampled edit
    corpus_a_geom.sqlite   A adiff geometry (subset)       user_seq.parquet    one row per history step
    corpus_mn.sqlite       M/N selection, metadata, osmChange
    prev_versions.sqlite   previous object versions        build_report.json   coverage, per corpus
    users.sqlite           account_created
    user_histories_pit.sqlite  point-in-time histories

Every source except the Corpus A metadata and contents is optional. A missing source is
reported in `build_report.json` and shows up in the data as an explicit *known* flag,
never as a silent zero. Unknown is not the same fact as zero: `user_n_prior = 0` with no
history fetched would tell the model "brand-new account", and if coverage differed by
class the model would learn the fetch, not the vandalism.

**Rule 1 - nothing is zeroed.** Changeset-level summaries run over *every* edit; only the
per-edit detail is sampled, to K = 256, stratified so the extremes survive. `sample_ratio`
tells the model how much it is not seeing.

**Rule 2 - user features are point-in-time.** Histories come from
`mapguard.data.corpora.history_for`, which only accepts a window that provably contains
everything before the target, and `user_history.point_in_time` filters again.

**Geometry has two routes, and each edit records which one it got** (`geom_route`):

    2  adiff         full before/after coordinates: length, area, shape, displacement
    1  prev_version  previous object version from the API: tag diffs for every object,
                     displacement for nodes, node-count change (`d_refs`) for ways
    0  none          a created object, or a version that could not be fetched

`account_age_days` / `account_age_known` are stored for analysis but are **not** model
inputs: the account is read as it is today, and deleted accounts are mostly vandals'.

`comments_count` is deliberately **not** a feature. It is read when the metadata is
fetched, years after the edit, and reviewers comment on exactly the changesets they
revert, so it would leak the label.

    python -m mapguard.features.build_store
    python -m mapguard.features.build_store --corpora A --limit 500 --out /tmp/fs
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
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
from mapguard.data.corpora import (
    CORPUS_MN_DB,
    r_row_info,
    HISTORY_DB,
    LEGACY_HISTORY_DB,
    USERS_DB,
    VERSIONS_DB,
    history_for,
    quarter_of,
    valid_selection,
)
from mapguard.data.osmchange import Edit, parse_safe
from mapguard.data.overpass import GeometryAction, GeometryState
from mapguard.data.users import account_age_days
from mapguard.features.geometry import features_for, summarise
from mapguard.features.user_history import PriorChangeset, build as build_user

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
SCHEMA_VERSION = 3

ROUTE_NONE, ROUTE_VERSION, ROUTE_ADIFF = 0, 1, 2

# Every key `geometry.summarise` can return. A changeset with no geometry gets zeros for
# all of them, so every row has the same columns (the Parquet schema is fixed by the first
# batch) and "no geometry" is carried by geom_frac_* instead.
GEOM_SUMMARY_KEYS = (
    "n_geom_actions", "n_way", "n_node", "n_with_old", "max_centroid_shift_m",
    "mean_centroid_shift_m", "p95_centroid_shift_m", "max_hausdorff_m", "total_d_area_m2",
    "max_abs_d_area_m2", "max_area_ratio", "total_d_points", "n_deleted", "n_created",
    "n_modified",
)

META_COLS = ["cid", "created_at", "closed_at", "uid", "user", "num_changes", "min_lat",
             "min_lon", "max_lat", "max_lon", "comments_count", "comment", "editor", "tags_json"]


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


def open_ro(rel: str) -> sqlite3.Connection | None:
    path = data_path(*rel.split("/"))
    if not path.exists():
        return None
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def geometry_actions(blob: str | None) -> list[GeometryAction]:
    """Rebuild adiff GeometryAction objects from the stored JSON."""
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


def _versions_has_meta(vconn: sqlite3.Connection | None) -> bool:
    if vconn is None:
        return False
    return "timestamp" in {r[1] for r in vconn.execute("PRAGMA table_info(versions)")}


def version_actions(
    edits: list[Edit], vconn: sqlite3.Connection | None, skip: set[tuple[str, int]],
    prev_meta: dict | None = None,
) -> tuple[list[GeometryAction], dict[tuple[str, int], int], int]:
    """GeometryActions from previous object versions, for edits adiff did not cover.

    Returns the actions, the previous `n_refs` per object (for `d_refs`), and how many
    edits were eligible (modify/delete of an existing version) so coverage is measurable.
    `prev_meta`, when given, is filled with (timestamp, uid) of every previous version found -
    including adiff-covered objects - for Ovid's "time to previous version" and "previous
    authors" features.
    """
    actions: list[GeometryAction] = []
    prev_refs: dict[tuple[str, int], int] = {}
    eligible = 0
    with_meta = _versions_has_meta(vconn)
    cols = "status, visible, lat, lon, n_refs, tags_json" + (", timestamp, uid" if with_meta else "")
    for e in edits:
        if e.action not in ("modify", "delete") or not e.version or e.version <= 1:
            continue
        eligible += 1
        key = (e.otype, e.oid)
        if vconn is None:
            continue
        row = vconn.execute(f"SELECT {cols} FROM versions WHERE otype=? AND oid=? AND version=?",
                            (e.otype, e.oid, e.version - 1)).fetchone()
        if not row or row[0] != 200:
            continue
        if prev_meta is not None and with_meta:
            prev_meta[key] = (row[6], row[7])
        if key in skip:
            continue
        _, visible, lat, lon, n_refs, tags_json = row[:6]
        is_node = e.otype == "node"
        old = GeometryState(
            otype=e.otype, oid=e.oid, version=e.version - 1,
            coords=[(lat, lon)] if is_node and visible and lat is not None else [],
            tags=json.loads(tags_json) if tags_json else {},
        )
        new = None if e.action == "delete" else GeometryState(
            otype=e.otype, oid=e.oid, version=e.version,
            coords=[(e.lat, e.lon)] if is_node and e.lat is not None else [],
            tags=dict(e.tags),
        )
        actions.append(GeometryAction(action=e.action, otype=e.otype, oid=e.oid,
                                      changeset=None, old=old, new=new))
        if n_refs is not None:
            prev_refs[key] = n_refs
    return actions, prev_refs, eligible


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
    free text, then an even spread through commit order. Truncating by position instead
    would drop exactly the objects a reviewer would look at first.
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
    if len(chosen) < k:
        step = max(1, n // max(1, (k - len(chosen))))
        add(range(0, n, step))
    add(range(n))
    return sorted(chosen)


# ------------------------------------------------------------------------ features


def changeset_row(
    cid: int,
    label: int,
    corpus: str,
    meta: dict,
    edits: list[Edit],
    actions: list[GeometryAction],
    routes: dict[tuple[str, int], int],
    account_created: str | None,
    history: list[PriorChangeset],
    history_known: bool,
    uf=None,
    prev_meta: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """One row of changesets.parquet. Summaries here never truncate (Rule 1)."""
    acts = Counter(e.action for e in edits)
    types = Counter(e.otype for e in edits)
    versions = [e.version for e in edits if e.version]
    keys: Counter[str] = Counter()
    for e in edits:
        keys.update(e.tags.keys())
    text_values = [v for e in edits for k, v in e.tags.items() if is_text_key(k) and v]

    geom = summarise(actions)
    uf = uf or build_user(meta.get("uid"), history, meta.get("created_at") or "")
    age = account_age_days(account_created, meta.get("created_at"))
    n = max(1, len(edits))
    route_counts = Counter(routes.get((e.otype, e.oid), ROUTE_NONE) for e in edits)

    row = {
        "cid": cid,
        "label": label,
        "corpus": corpus,
        "quarter": quarter_of(meta.get("created_at")),
        "schema_version": SCHEMA_VERSION,
        # --- metadata
        "created_at": meta.get("created_at"),
        "uid": meta.get("uid"),
        "num_changes_meta": meta.get("num_changes"),
        "comment": (meta.get("comment") or "")[:500],
        "comment_length": len(meta.get("comment") or ""),
        "editor": meta.get("editor") or "",
        "bbox_lat_span": (
            (meta["max_lat"] - meta["min_lat"])
            if meta.get("max_lat") is not None and meta.get("min_lat") is not None else None
        ),
        "bbox_lon_span": (
            (meta["max_lon"] - meta["min_lon"])
            if meta.get("max_lon") is not None and meta.get("min_lon") is not None else None
        ),
        "account_age_days": age,
        "account_age_known": 1 if age is not None else 0,
        # --- exact edit statistics, over every edit
        "n_edits": len(edits),
        "n_create": acts.get("create", 0),
        "n_modify": acts.get("modify", 0),
        "n_delete": acts.get("delete", 0),
        "frac_delete": round(acts.get("delete", 0) / n, 4),
        "n_node": types.get("node", 0),
        "n_way": types.get("way", 0),
        "n_relation": types.get("relation", 0),
        "max_version": max(versions) if versions else 0,
        "mean_version": round(sum(versions) / len(versions), 3) if versions else 0.0,
        "n_v1": sum(1 for v in versions if v == 1),
        "total_tags": sum(len(e.tags) for e in edits),
        "mean_tags": round(sum(len(e.tags) for e in edits) / n, 3),
        "distinct_keys": len(keys),
        "n_text_edits": sum(1 for e in edits if any(is_text_key(k) for k in e.tags)),
        # --- sampling, so the model knows what it is not seeing
        "n_edits_seen": min(len(edits), SAMPLE_K),
        "sample_ratio": round(min(1.0, SAMPLE_K / n), 4),
        # --- text for B2
        "text_blob": " ".join(text_values)[:2000],
        "text_chars": sum(len(v) for v in text_values),
        # --- geometry coverage, so absence of geometry is never read as "nothing moved"
        "geom_frac_adiff": round(route_counts.get(ROUTE_ADIFF, 0) / n, 4),
        "frac_edits_prev_version": round(route_counts.get(ROUTE_VERSION, 0) / n, 4),
        "user_history_known": 1 if history_known else 0,
        # --- Ovid: previous authors of the touched objects, and imagery use
        "n_prev_authors": len({u for _, u in (prev_meta or {}).values()
                               if u is not None and u != meta.get("uid")}),
        "imagery_used": 1 if "imagery_used" in _tags(meta) else 0,
        # not a model input: marks the rows B1 can be evaluated on with full geometry
        "has_adiff": 1 if ROUTE_ADIFF in routes.values() else 0,
    }
    for key in WATCH_KEYS:
        row[f"key_{key.replace(':', '_')}"] = 1 if keys.get(key) else 0
    for k in GEOM_SUMMARY_KEYS:
        row[f"geom_{k}"] = float(geom.get(k, 0.0))
    for k, v in uf.as_dict().items():
        row[f"user_{k}"] = v
    row["user_seq_len"] = len(uf.sequence)
    row.update(extra or {})
    return row


def _tags(meta: dict) -> dict:
    try:
        return json.loads(meta.get("tags_json") or "{}")
    except ValueError:
        return {}


def _days_between(a: str | None, b: str | None) -> float | None:
    from datetime import datetime

    if not a or not b:
        return None
    try:
        fa = datetime.fromisoformat(a.replace("Z", "+00:00"))
        fb = datetime.fromisoformat(b.replace("Z", "+00:00"))
    except ValueError:
        return None
    d = (fb - fa).total_seconds() / 86400.0
    return round(d, 4) if d >= 0 else None


def edit_rows(cid: int, edits: list[Edit], geom_by_key: dict, routes: dict,
              prev_refs: dict, prev_meta: dict | None = None) -> list[dict]:
    prev_meta = prev_meta or {}
    keep = stratified_sample(edits, geom_by_key)
    rows = []
    for pos in keep:
        e = edits[pos]
        key = (e.otype, e.oid)
        g = geom_by_key.get(key)
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
            "d_refs": (e.n_refs - prev_refs[key]) if key in prev_refs and e.action == "modify" else None,
            # Ovid: time to the previous version, and whether its author was someone else
            "days_since_prev_version": _days_between(prev_meta.get(key, (None, None))[0], e.timestamp),
            "prev_same_author": (None if prev_meta.get(key, (None, None))[1] is None or e.uid is None
                                 else int(prev_meta[key][1] == e.uid)),
            "lat": e.lat,
            "lon": e.lon,
            "text": text[:200],
            "has_geometry": 1 if g else 0,
            "geom_route": routes.get(key, ROUTE_NONE),
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


def user_seq_rows(cid: int, uf) -> list[dict]:
    return [
        {"cid": cid, "step": i, "num_changes": a, "gap_days": b,
         "comment_length": c, "bbox_diag_deg": d}
        for i, (a, b, c, d) in enumerate(uf.sequence)
    ]


# --------------------------------------------------------------------------- writer


class TableWriter:
    """ParquetWriter that infers its schema from the first batch and flushes in groups.

    Columns that are None in the whole first batch would be typed `null` and break the
    cast of every later batch, so known numeric columns are forced to float64/int64.
    """

    def __init__(self, path: Path, types: dict[str, pa.DataType] | None = None,
                 float_prefixes: tuple[str, ...] = ()):
        self.path = path
        self.float_prefixes = float_prefixes
        self.rows: list[dict] = []
        self.writer: pq.ParquetWriter | None = None
        self.types = types or {}
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
            fields = []
            for f in table.schema:
                t = self.types.get(f.name, f.type)
                if f.name.startswith(self.float_prefixes) and (
                        pa.types.is_integer(t) or pa.types.is_null(t) or pa.types.is_boolean(t)):
                    t = pa.float64()
                if pa.types.is_null(t):
                    t = pa.float64()
                fields.append(pa.field(f.name, t))
            schema = pa.schema(fields)
            table = table.cast(schema)
            self.writer = pq.ParquetWriter(self.path, schema, compression="zstd")
        else:
            table = table.select(self.writer.schema.names).cast(self.writer.schema)
        self.writer.write_table(table)
        self.n += len(self.rows)
        self.rows = []

    def close(self, empty_schema: pa.Schema | None = None) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()
        elif empty_schema is not None:
            # an empty table still has to exist: the training data contract requires the file
            pq.write_table(empty_schema.empty_table(), self.path)


FLOAT_EDIT = {c: pa.float64() for c in (
    "centroid_shift_m", "hausdorff_m", "d_length_m", "d_area_m2", "length_ratio",
    "area_ratio", "d_points", "tags_added", "tags_removed", "tags_changed", "d_refs",
    "lat", "lon", "days_since_prev_version", "prev_same_author")}
FLOAT_CS = {c: pa.float64() for c in (
    "account_age_days", "bbox_lat_span", "bbox_lon_span", "num_changes_meta", "uid")}
FLOAT_CS.update({"quarter": pa.string(), "created_at": pa.string(), "comment": pa.string(),
                 "editor": pa.string(), "text_blob": pa.string(), "r_stratum": pa.string(),
                 "revert_mentions_vandalism": pa.float64()})


# ----------------------------------------------------------------------------- main


def iter_corpus_a(labels: dict[int, int], limit: int | None):
    meta_conn = open_ro(CORPUS_A_META_DB)
    contents = open_ro(CORPUS_A_DB)
    if meta_conn is None or contents is None:
        raise SystemExit("Corpus A needs corpus_a_meta.sqlite and corpus_a.sqlite under "
                         "MAPGUARD_DATA/raw - see RUNBOOK.md")
    geom = open_ro(CORPUS_A_GEOM_DB)
    q = f"SELECT {', '.join(META_COLS)} FROM meta WHERE status = 200 ORDER BY cid"
    n = 0
    for row in meta_conn.execute(q):
        meta = dict(zip(META_COLS, row))
        cid = meta["cid"]
        if cid not in labels:
            continue
        got = contents.execute("SELECT xml FROM changeset WHERE id = ? AND status = 200",
                               (cid,)).fetchone()
        adiff = geom.execute("SELECT json FROM geometry WHERE cid = ? AND route = 'adiff'",
                             (cid,)).fetchone() if geom else None
        yield "A", labels[cid], meta, (got[0] if got else None), (adiff[0] if adiff else None)
        n += 1
        if limit and n >= limit:
            return


def iter_corpus_mn(limit: int | None):
    conn = open_ro(CORPUS_MN_DB)
    if conn is None:
        return
    try:
        valid = valid_selection(conn)
        info = r_row_info(conn)
    except sqlite3.OperationalError:
        return
    q = f"SELECT {', '.join(META_COLS)} FROM meta WHERE status = 200 ORDER BY cid"
    n = 0
    for row in conn.execute(q):
        meta = dict(zip(META_COLS, row))
        cid = meta["cid"]
        if cid not in valid:
            continue
        corpus, label = valid[cid]
        got = conn.execute("SELECT xml FROM contents WHERE cid = ? AND status = 200",
                           (cid,)).fetchone()
        meta["_extra"] = ({"r_stratum": info[cid][0],
                           "revert_mentions_vandalism": float(info[cid][1]) if label == 1 else None}
                          if corpus == "R" and cid in info else {})
        yield corpus, label, meta, (got[0] if got else None), None
        n += 1
        if limit and n >= limit:
            return


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="A,MN", help="comma list of A and MN")
    ap.add_argument("--limit", type=int, default=None, help="per corpus, for smoke tests")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else interim_dir()
    out.mkdir(parents=True, exist_ok=True)
    corpora = [c.strip().upper() for c in args.corpora.split(",") if c.strip()]

    users_conn = open_ro(USERS_DB)
    account_created = (
        dict(users_conn.execute("SELECT uid, account_created FROM users WHERE status = 200"))
        if users_conn else {}
    )
    hist_conn = open_ro(HISTORY_DB)
    legacy_conn = open_ro(LEGACY_HISTORY_DB)
    vconn = open_ro(VERSIONS_DB)

    cw = TableWriter(out / "changesets.parquet", FLOAT_CS, float_prefixes=("geom_", "user_"))
    ew = TableWriter(out / "edits.parquet", FLOAT_EDIT)
    sw = TableWriter(out / "user_seq.parquet")

    cov: dict[str, Counter] = {}

    def sources():
        if "A" in corpora:
            yield from iter_corpus_a(load_labels(), args.limit)
        if "MN" in corpora:
            yield from iter_corpus_mn(args.limit)

    seen = 0
    for corpus, label, meta, xml, adiff_blob in sources():
        c = cov.setdefault(corpus, Counter())
        c["candidates"] += 1
        if not xml:
            c["dropped_no_contents"] += 1
            continue
        edits, err = parse_safe(xml)
        if err:
            c["dropped_bad_xml"] += 1
            continue
        if not edits:
            # Nothing downloadable: never opened, or every edit hidden later. In Corpus A these
            # are 226 rows, 224 of them negatives - a free "clean" signal that measures how
            # the benchmark was sampled, not vandalism. Dropped and counted per label.
            c[f"dropped_empty_label_{label}"] += 1
            continue
        cid = meta["cid"]
        c[f"label_{label}"] += 1

        adiff = geometry_actions(adiff_blob)
        routes: dict[tuple[str, int], int] = {}
        geom_by_key = {}
        for a in adiff:
            routes[(a.otype, a.oid)] = ROUTE_ADIFF
            geom_by_key[(a.otype, a.oid)] = features_for(a)
        prev_meta: dict = {}
        vacts, prev_refs, eligible = version_actions(edits, vconn, set(routes), prev_meta)
        for a in vacts:
            routes[(a.otype, a.oid)] = ROUTE_VERSION
            geom_by_key[(a.otype, a.oid)] = features_for(a)
        c["edits"] += len(edits)
        c["edits_eligible_prev_version"] += eligible
        c["edits_prev_version"] += len(vacts)
        c["changesets_with_adiff"] += 1 if adiff else 0

        blob, rule = history_for(hist_conn, meta.get("uid"), meta.get("created_at"), legacy_conn)
        c[f"history_{rule}"] += 1
        c[f"history_{rule}_label_{label}"] += 1
        history = prior_changesets(blob)
        uf = build_user(meta.get("uid"), history, meta.get("created_at") or "")
        created = account_created.get(meta.get("uid"))
        c[f"account_age_known_label_{label}"] += 1 if created else 0

        extra = meta.pop("_extra", {})
        if corpus == "A":
            extra = {"r_stratum": None, "revert_mentions_vandalism": None}
        cw.add(changeset_row(cid, label, corpus, meta, edits, adiff + vacts, routes,
                             created, history, rule != "missing", uf, prev_meta,
                             {"r_stratum": extra.get("r_stratum"),
                              "revert_mentions_vandalism": extra.get("revert_mentions_vandalism")}))
        if edits:
            ew.add(edit_rows(cid, edits, geom_by_key, routes, prev_refs, prev_meta))
        if uf.sequence:
            sw.add(user_seq_rows(cid, uf))

        seen += 1
        if seen % 1000 == 0:
            print(f"{seen} changesets  {dict(c)}", flush=True)

    cw.close()
    ew.close()
    sw.close(pa.schema([("cid", pa.int64()), ("step", pa.int64()), ("num_changes", pa.float64()),
                        ("gap_days", pa.float64()), ("comment_length", pa.float64()),
                        ("bbox_diag_deg", pa.float64())]))

    report = {
        "schema_version": SCHEMA_VERSION,
        "changesets": cw.n,
        "edits": ew.n,
        "user_seq_steps": sw.n,
        "sample_k": SAMPLE_K,
        "sources_present": {
            "users": users_conn is not None, "histories_pit": hist_conn is not None,
            "histories_legacy": legacy_conn is not None, "prev_versions": vconn is not None,
            "corpus_mn": open_ro(CORPUS_MN_DB) is not None,
        },
        "coverage_by_corpus": {k: dict(sorted(v.items())) for k, v in cov.items()},
        "out": str(out),
        "note": ("coverage split by label on purpose: if history or account-age coverage "
                 "differs between classes, the corresponding features measure the fetch "
                 "and must not be reported as a signal"),
    }
    print(json.dumps(report, indent=2))
    (out / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
