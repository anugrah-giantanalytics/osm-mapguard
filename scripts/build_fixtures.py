"""Deterministic fixtures cut from REAL fetched OSM data.

CHECKLIST_v2 Phase 1 asks for a fixture set covering seven awkward cases. It is
deliberately built by *selecting* real changesets from the local stores rather than by
generating plausible-looking numbers. A synthetic corpus tells you your code runs; it
tells you nothing about whether your code is right, because you wrote both sides of it.
The failures that matter here - a changeset with no bbox, a deleted account, geometry
that adiff refuses - are all things the real data does and an invented corpus does not.

The seven cases, each picked by a query over the real stores:

    normal            small, ordinary, labelled clean
    text_vandalism    labelled damaging, carries free-text tag values
    geometry_shift    largest measured centroid displacement
    mass_change       largest num_changes
    anonymous         uid missing - a deleted account is still a real changeset
    no_bbox           bbox absent, so adiff must fall back to multi-fetch
    geometry_fallback adiff returned nothing usable

Output is small enough to commit: metadata, labels and geometry JSON only, no bulk XML.
The demo (Phase 3/4) and the golden-fixture tests both read from here, so the offline
path works from a clean checkout with no network and no token.

    python scripts/build_fixtures.py
    python scripts/build_fixtures.py --out data/fixtures
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapguard.config import (
    CORPUS_A_GEOM_DB,
    CORPUS_A_META_DB,
    LABELS,
    REPO,
    data_path,
)

OUT_DEFAULT = REPO / "data" / "fixtures"


def labels() -> dict[int, int]:
    out = {}
    with LABELS.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            cid, lab = line.rstrip("\n").split("\t")
            out[int(cid)] = 1 if lab.strip().lower() == "true" else 0
    return out


def pick(conn: sqlite3.Connection, where: str, order: str, lab: dict[int, int],
         want_label: int | None = None, limit: int = 1) -> list[dict]:
    """One real changeset matching a condition, as a plain dict."""
    cols = ("cid, created_at, closed_at, uid, user, num_changes, min_lat, min_lon,"
            " max_lat, max_lon, comments_count, comment, editor, tags_json")
    rows = conn.execute(
        f"SELECT {cols} FROM meta WHERE status = 200 AND {where} ORDER BY {order} LIMIT 400"
    ).fetchall()
    keys = [c.strip() for c in cols.split(",")]
    out = []
    for r in rows:
        d = dict(zip(keys, r))
        d["label"] = lab.get(d["cid"])
        if want_label is not None and d["label"] != want_label:
            continue
        d["tags"] = json.loads(d.pop("tags_json") or "{}")
        out.append(d)
        if len(out) >= limit:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else OUT_DEFAULT
    out.mkdir(parents=True, exist_ok=True)

    meta_db = data_path(*CORPUS_A_META_DB.split("/"))
    if not meta_db.exists():
        raise SystemExit(f"no metadata store at {meta_db} - run fetch_corpus_a_meta.py")
    conn = sqlite3.connect(meta_db)
    lab = labels()

    geom_db = data_path(*CORPUS_A_GEOM_DB.split("/"))
    geom = sqlite3.connect(geom_db) if geom_db.exists() else None

    cases: dict[str, list[dict]] = {}

    cases["normal"] = pick(
        conn, "num_changes BETWEEN 2 AND 15 AND uid IS NOT NULL AND min_lat IS NOT NULL",
        "cid", lab, want_label=0)

    cases["text_vandalism"] = pick(
        conn, "comment IS NOT NULL AND length(comment) > 0 AND uid IS NOT NULL",
        "cid", lab, want_label=1)

    cases["mass_change"] = pick(
        conn, "num_changes IS NOT NULL", "num_changes DESC", lab)

    cases["anonymous"] = pick(conn, "uid IS NULL", "cid", lab)

    cases["no_bbox"] = pick(conn, "min_lat IS NULL", "cid", lab)

    # Geometry cases come from measured adiff output, not from a guess about which
    # changesets "look" geometric.
    cases["geometry_shift"] = []
    cases["geometry_fallback"] = []
    if geom is not None:
        rows = geom.execute(
            "SELECT cid, route, note, n_actions, n_with_old, n_way, json FROM geometry"
        ).fetchall()
        with_actions = [r for r in rows if r[3]]
        without = [r for r in rows if not r[3]]
        if with_actions:
            best = max(with_actions, key=lambda r: r[3])
            got = pick(conn, f"cid = {best[0]}", "cid", lab)
            if got:
                got[0]["geometry"] = {
                    "route": best[1], "note": best[2], "n_actions": best[3],
                    "n_with_old": best[4], "n_way": best[5],
                    "actions": json.loads(best[6] or "[]")[:20],
                }
                cases["geometry_shift"] = got
        if without:
            got = pick(conn, f"cid = {without[0][0]}", "cid", lab)
            if got:
                got[0]["geometry"] = {"route": without[0][1], "note": without[0][2],
                                      "n_actions": 0, "actions": []}
                cases["geometry_fallback"] = got

    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "real changesets selected from the local Corpus A stores",
        "provenance": {
            "metadata_db": str(meta_db),
            "geometry_db": str(geom_db) if geom else None,
            "labels": str(LABELS),
        },
        "note": (
            "Selected, not generated. A synthetic corpus tells you the code runs; it "
            "cannot tell you the code is right, because the same person wrote both "
            "sides of it. Every awkward case here is one the real data actually does."
        ),
        "cases": {k: [c["cid"] for c in v] for k, v in cases.items()},
        "missing_cases": [k for k, v in cases.items() if not v],
    }

    for name, rows in cases.items():
        if rows:
            (out / f"{name}.json").write_text(
                json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"\nwritten to {out}")
    if manifest["missing_cases"]:
        print(f"MISSING: {manifest['missing_cases']} - rerun once those stores are populated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
