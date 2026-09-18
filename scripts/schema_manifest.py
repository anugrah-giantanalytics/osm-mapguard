"""Write the feature-schema manifest: every column, its dtype, unit, meaning, missing policy and
measured null rate, from the feature store that was actually built.

    python scripts/schema_manifest.py                   # MAPGUARD_DATA/interim
    python scripts/schema_manifest.py --store D:/fs

Writes results/feature_schema.json. It is generated rather than hand-written so it cannot
drift from the Parquet files, and so a reviewer can check any feature the paper mentions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyarrow.parquet as pq  # noqa: E402

from mapguard.config import RESULTS, interim_dir  # noqa: E402
from mapguard.training.dataset import EDIT_FEATURES, TABULAR_FEATURES  # noqa: E402

# column -> (unit, meaning). Missing values become 0 at assembly (training/dataset._clean);
# where 0 would be ambiguous a *_known flag carries the fact instead.
DESCRIBE = {
    "cid": ("id", "changeset id"),
    "label": ("0/1", "1 = damaging under the corpus's label rule (see corpus)"),
    "corpus": ("A|M|N|R", "label source: A Ovid 2014-19, M OSMCha review, N random unreviewed, R revert-mined 2021-26"),
    "quarter": ("YYYYQn", "quarter of created_at"),
    "created_at": ("ISO-8601 UTC", "changeset creation time"),
    "schema_version": ("int", "feature-store schema version that built this row"),
    "num_changes_meta": ("count", "edit count reported by changeset metadata (cross-check of n_edits)"),
    "uid": ("id", "author account id; used for user-disjoint splits, never as a feature"),
    "comment": ("text", "changeset comment, truncated to 500 chars; not a model input"),
    "comment_length": ("chars", "length of the changeset comment"),
    "editor": ("text", "created_by tag; B0 tripwire only"),
    "bbox_lat_span": ("degrees", "changeset bbox height"),
    "bbox_lon_span": ("degrees", "changeset bbox width"),
    "account_age_days": ("days", "edit time minus account creation; missing when the account is unknown or deleted"),
    "account_age_known": ("0/1", "1 if account_age_days was measured"),
    "n_edits": ("count", "objects touched, over every edit"),
    "n_create": ("count", "created objects"), "n_modify": ("count", "modified objects"),
    "n_delete": ("count", "deleted objects"), "frac_delete": ("ratio", "n_delete / n_edits"),
    "n_node": ("count", "nodes touched"), "n_way": ("count", "ways touched"),
    "n_relation": ("count", "relations touched"),
    "max_version": ("version", "highest object version touched"),
    "mean_version": ("version", "mean object version touched"),
    "n_v1": ("count", "objects at version 1 (new)"),
    "total_tags": ("count", "tags on all touched objects after the edit"),
    "mean_tags": ("count", "tags per touched object"),
    "distinct_keys": ("count", "distinct tag keys touched"),
    "n_text_edits": ("count", "objects carrying free-text tags (name, description, ...)"),
    "n_edits_seen": ("count", "edits in the per-edit sample (<= 256)"),
    "sample_ratio": ("ratio", "n_edits_seen / n_edits"),
    "text_blob": ("text", "free-text tag values, B2 char-CNN input"),
    "text_chars": ("chars", "total free-text characters"),
    "geom_frac_adiff": ("ratio", "fraction of edits with full adiff geometry"),
    "frac_edits_prev_version": ("ratio", "fraction of edits with a fetched previous version"),
    "user_history_known": ("0/1", "1 if a point-in-time history was available"),
    "has_adiff": ("0/1", "changeset has adiff geometry; marks the B1 evaluation subset; not an input"),
    "user_seq_len": ("count", "history steps in user_seq"),
    # edits
    "position": ("index", "edit position in commit order"),
    "action": ("create|modify|delete", "osmChange action"),
    "otype": ("node|way|relation", "object type"),
    "oid": ("id", "object id"), "version": ("version", "object version after the edit"),
    "is_new_object": ("0/1", "version == 1"),
    "n_tags": ("count", "tags on the object after the edit"),
    "n_refs": ("count", "way nodes or relation members after the edit"),
    "d_refs": ("count", "n_refs change against the previous version (modify only)"),
    "lat": ("degrees", "node latitude after the edit"), "lon": ("degrees", "node longitude after the edit"),
    "text": ("text", "free-text tag values of this object"),
    "has_geometry": ("0/1", "any geometry route available"),
    "geom_route": ("0|1|2", "0 none, 1 previous version (nodes, tags), 2 adiff (full way geometry)"),
    "centroid_shift_m": ("metres", "how far the object's centroid moved"),
    "hausdorff_m": ("metres", "shape change, symmetric Hausdorff"),
    "d_length_m": ("metres", "way length change (adiff only)"),
    "d_area_m2": ("m^2", "closed-way area change (adiff only)"),
    "length_ratio": ("ratio", "new/old length, capped at 1e6 (adiff only)"),
    "area_ratio": ("ratio", "new/old area, capped at 1e6 (adiff only)"),
    "d_points": ("count", "vertex count change (adiff only)"),
    "tags_added": ("count", "tag keys added against the previous version"),
    "tags_removed": ("count", "tag keys removed against the previous version"),
    "tags_changed": ("count", "tag values changed against the previous version"),
    # user_seq
    "step": ("index", "position in the point-in-time history, oldest first"),
    "num_changes": ("count", "edits in that earlier changeset"),
    "gap_days": ("days", "time since the previous earlier changeset"),
    "bbox_diag_deg": ("degrees", "bbox diagonal of that earlier changeset"),
}
PREFIXES = {
    "key_": ("0/1", "changeset touches this tag key"),
    "geom_": ("mixed", "changeset-level geometry summary over every edit (see features/geometry.summarise)"),
    "user_": ("mixed", "point-in-time user aggregate, from changesets strictly before this one"),
}


def describe(col: str) -> tuple[str, str]:
    if col in DESCRIBE:
        return DESCRIBE[col]
    for p, d in PREFIXES.items():
        if col.startswith(p):
            return d
    return ("?", "undocumented - add it to scripts/schema_manifest.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", default=str(RESULTS / "feature_schema.json"))
    args = ap.parse_args()
    store = Path(args.store) if args.store else interim_dir()

    manifest = {"store": str(store), "tables": {},
                "model_inputs": {"edit_features": list(EDIT_FEATURES),
                                 "tabular": list(TABULAR_FEATURES)},
                "missing_policy": "null -> 0 at assembly; *_known flags and geom_route carry "
                                  "whether a value was measured, so 0 is never ambiguous"}
    report = store / "build_report.json"
    if report.exists():
        manifest["build_report"] = json.loads(report.read_text(encoding="utf-8"))
    undocumented = []
    for name in ("changesets", "edits", "user_seq"):
        pf = pq.ParquetFile(store / f"{name}.parquet")
        table = pf.read()
        cols = []
        for field in table.schema:
            arr = table.column(field.name)
            unit, meaning = describe(field.name)
            if unit == "?":
                undocumented.append(f"{name}.{field.name}")
            cols.append({"name": field.name, "dtype": str(field.type), "unit": unit,
                         "meaning": meaning,
                         "null_fraction": round(arr.null_count / max(1, len(arr)), 4)})
        manifest["tables"][name] = {"rows": table.num_rows, "columns": cols}
    missing_inputs = [c for c in TABULAR_FEATURES
                      if c not in {x["name"] for x in manifest["tables"]["changesets"]["columns"]}]
    manifest["model_inputs_missing_from_store"] = missing_inputs
    manifest["undocumented_columns"] = undocumented
    Path(args.out).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {sum(t['rows'] for t in manifest['tables'].values()):,} rows, "
          f"{len(undocumented)} undocumented columns, missing inputs: {missing_inputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
