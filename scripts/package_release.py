"""Package the dataset for a GitHub release: pseudonymised, checksummed, documented.

Three things this does that a `zip -r` would not:

1. **Pseudonymises before anything leaves the machine.** `uid`, `user` and OSMCha's
   `check_user` are personal data, and this dataset labels people's edits as damaging.
   Published tables carry a salted SHA-256 of the uid and no usernames at all. The salt
   lives in `.env` and is never published, so the mapping cannot be reversed without it.
   Changeset ids stay, because they are already public and the data is reconstructible
   from them - but nothing we publish should make building a list of "vandals" easier
   than OSM already does.

2. **Generates the dataset card from the data**, not from intent. Row counts, date
   windows, geography and coverage gaps are measured at package time, so the card cannot
   drift from the archive it describes.

3. **Records provenance.** Git SHA, schema version, per-file SHA-256, and the exact
   fetch windows. Every number in the paper has to be traceable to the data that produced
   it, and a zip with no manifest breaks that chain.

Licence: OSM data is ODbL. A derived dataset is a Derivative Database and **must** be
released under ODbL with attribution. Shipping it under the repository's code licence
would be a licence violation, so `LICENSE-DATA` is written into every archive.

    python scripts/package_release.py --version v0.1.0
    python scripts/package_release.py --version v0.1.0 --publish
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapguard.config import REPO, data_path

OUT = REPO / "dist"
RAW = Path("C:/osm-data/raw")

ODBL = """This dataset is derived from OpenStreetMap.

  (c) OpenStreetMap contributors

OpenStreetMap data is licensed under the Open Data Commons Open Database License
(ODbL) v1.0: https://opendatacommons.org/licenses/odbl/1-0/

Because this dataset is derived from ODbL data, it is a Derivative Database and is
released under the same licence. Any further redistribution, with or without
modification, must:

  * attribute OpenStreetMap contributors,
  * keep the work under ODbL, and
  * make any modified version available under ODbL.

The OSMCha review labels are human review decisions recorded on osmcha.org, which
operates on OpenStreetMap data and is subject to the same terms.

PRIVACY: OpenStreetMap user identifiers have been replaced with a salted SHA-256 digest.
Usernames and reviewer names have been removed entirely. The salt is not published. Do
not attempt to re-identify contributors from this dataset, and do not use it to make
claims about individual accounts.

NOTE ON LABELS: a label of "harmful" reflects either a human reviewer's judgement on
osmcha.org or a revert-derived proxy, depending on the corpus. Neither is a statement
about a person's intent. See DATASET_CARD.md.
"""


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def salt() -> str:
    """Pseudonymisation salt. Local only; publishing it would undo the pseudonymisation."""
    from mapguard.data.osmcha import load_env

    load_env()
    s = os.environ.get("MAPGUARD_SALT", "").strip()
    if not s:
        raise SystemExit(
            "MAPGUARD_SALT is not set. Add a long random value to .env before packaging - "
            "without it the uid hashes would be trivially reversible by rainbow table."
        )
    return s


def pseudo(uid, s: str) -> str | None:
    if uid is None:
        return None
    return hashlib.sha256(f"{s}:{int(uid)}".encode()).hexdigest()[:16]


def scrub(obj, s: str):
    """Recursively replace uid with a pseudonym and drop every name field.

    Fixtures are selected from raw metadata, so they arrive carrying `uid` and `user`.
    Shipping those would put a named individual next to a vandalism label - the exact
    thing the privacy policy exists to prevent.
    """
    if isinstance(obj, list):
        return [scrub(o, s) for o in obj]
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        if k in ("user", "check_user", "display_name", "username"):
            continue
        if k == "uid":
            out["user_pseudonym"] = pseudo(v, s)
            continue
        out[k] = scrub(v, s)
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def export_corpus_a(out_dir: Path, s: str) -> dict:
    """Metadata + per-changeset edit summary, pseudonymised. No raw XML, no usernames."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from mapguard.data.osmchange import parse_safe

    meta = sqlite3.connect(f"file:{RAW/'corpus_a_meta.sqlite'}?mode=ro", uri=True)
    contents = sqlite3.connect(f"file:{RAW/'corpus_a.sqlite'}?mode=ro", uri=True)

    labels = {}
    with (REPO / "data" / "labels" / "ovid_labels.tsv").open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            a, b = line.rstrip("\n").split("\t")
            labels[int(a)] = 1 if b.strip().lower() == "true" else 0

    rows = []
    stats = {"n": 0, "with_contents": 0, "total_edits": 0}
    q = ("SELECT cid, created_at, closed_at, uid, num_changes, min_lat, min_lon,"
         " max_lat, max_lon, comments_count, comment, editor FROM meta WHERE status = 200")
    for (cid, created, closed, uid, nch, mnla, mnlo, mxla, mxlo, ncom, comment, editor) in meta.execute(q):
        got = contents.execute(
            "SELECT xml FROM changeset WHERE id = ? AND xml IS NOT NULL", (cid,)).fetchone()
        n_edits = n_node = n_way = n_rel = n_create = n_modify = n_delete = 0
        if got:
            edits, err = parse_safe(got[0])
            if not err:
                stats["with_contents"] += 1
                n_edits = len(edits)
                stats["total_edits"] += n_edits
                for e in edits:
                    n_node += e.otype == "node"
                    n_way += e.otype == "way"
                    n_rel += e.otype == "relation"
                    n_create += e.action == "create"
                    n_modify += e.action == "modify"
                    n_delete += e.action == "delete"
        rows.append({
            "changeset_id": cid,
            "label_harmful": labels.get(cid),
            "label_source": "ovid_revert_derived",
            "created_at": created,
            "closed_at": closed,
            "user_pseudonym": pseudo(uid, s),
            "num_changes": nch,
            "n_edits_parsed": n_edits,
            "n_node": n_node, "n_way": n_way, "n_relation": n_rel,
            "n_create": n_create, "n_modify": n_modify, "n_delete": n_delete,
            "min_lat": mnla, "min_lon": mnlo, "max_lat": mxla, "max_lon": mxlo,
            "comments_count": ncom,
            "comment_length": len(comment or ""),
            "editor": editor,
        })
        stats["n"] += 1

    meta.close(); contents.close()
    pq.write_table(pa.Table.from_pylist(rows), out_dir / "corpus_a_changesets.parquet",
                   compression="zstd")
    return stats


def export_labels(out_dir: Path, s: str) -> dict:
    """OSMCha review decisions. Reviewer names dropped, uid hashed."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    db = RAW / "osmcha_labels.sqlite"
    if not db.exists():
        return {"n": 0}
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = []
    for (cid, harmful, checked, check_date, date, uid, editor, comment,
         ncre, nmod, ndel, suspect, reasons) in c.execute(
            "SELECT cid,harmful,checked,check_date,date,uid,editor,comment,"
            "n_create,n_modify,n_delete,is_suspect,reasons FROM labels"):
        rows.append({
            "changeset_id": cid,
            "label_harmful": harmful,
            "label_source": "osmcha_human_review",
            "checked": checked,
            "review_date": check_date,
            "created_at": date,
            "user_pseudonym": pseudo(uid, s),
            "editor": editor,
            "comment_length": len(comment or ""),
            "n_create": ncre, "n_modify": nmod, "n_delete": ndel,
            "is_suspect_flagged": suspect,
            "flag_reasons": reasons,
        })
    gaps = [dict(zip(("source", "window_from", "window_to", "klass", "status",
                      "attempts", "resolved"), r))
            for r in c.execute("SELECT source,window_from,window_to,klass,status,"
                               "attempts,resolved FROM fetch_gaps")]
    c.close()
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), out_dir / "osmcha_labels.parquet",
                       compression="zstd")
    (out_dir / "fetch_coverage.json").write_text(json.dumps(gaps, indent=2), encoding="utf-8")
    return {"n": len(rows), "gaps": len(gaps),
            "unresolved": sum(1 for g in gaps if not g["resolved"])}


def geography(pq_path: Path) -> dict:
    import pyarrow.parquet as pq

    t = pq.read_table(pq_path, columns=["min_lat", "max_lat", "min_lon", "max_lon"]).to_pydict()
    def region(la, lo):
        if la is None or lo is None: return "unknown"
        if 35 <= la <= 72 and -25 <= lo <= 45: return "Europe"
        if 5 <= la <= 55 and 60 <= lo <= 150: return "Asia"
        if 15 <= la <= 72 and -170 <= lo <= -50: return "North America"
        if -56 <= la < 15 and -85 <= lo <= -33: return "South America"
        if -35 <= la <= 37 and -20 <= lo <= 52: return "Africa"
        if -50 <= la <= 0 and 110 <= lo <= 180: return "Oceania"
        return "other"
    from collections import Counter
    c = Counter()
    for a, b, x, y in zip(t["min_lat"], t["max_lat"], t["min_lon"], t["max_lon"]):
        c[region((a + b) / 2 if a is not None and b is not None else None,
                 (x + y) / 2 if x is not None and y is not None else None)] += 1
    total = sum(c.values())
    return {k: {"n": v, "share": round(v / total, 4)} for k, v in c.most_common()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="e.g. v0.1.0")
    ap.add_argument("--publish", action="store_true", help="create the GitHub release")
    ap.add_argument("--notes-only", action="store_true")
    args = ap.parse_args()

    s = salt()
    OUT.mkdir(parents=True, exist_ok=True)
    stage = OUT / f"mapguard-dataset-{args.version}"
    stage.mkdir(parents=True, exist_ok=True)

    print("exporting Corpus A (pseudonymised)...")
    a_stats = export_corpus_a(stage, s)
    print(f"  {a_stats['n']:,} changesets, {a_stats['with_contents']:,} with contents, "
          f"{a_stats['total_edits']:,} edits")

    print("exporting OSMCha labels (pseudonymised)...")
    l_stats = export_labels(stage, s)
    print(f"  {l_stats['n']:,} label rows")

    geo = geography(stage / "corpus_a_changesets.parquet")

    # results/ are gate measurements and carry no identity. Fixtures are real changesets
    # and DO carry uid and username - they were selected from raw metadata. Scrub them,
    # or the release names individuals next to a vandalism label.
    (stage / "results").mkdir(parents=True, exist_ok=True)
    for f in (REPO / "results").glob("*.json"):
        stage.joinpath("results", f.name).write_bytes(f.read_bytes())

    (stage / "fixtures").mkdir(parents=True, exist_ok=True)
    for f in (REPO / "data" / "fixtures").glob("*.json"):
        stage.joinpath("fixtures", f.name).write_text(
            json.dumps(scrub(json.loads(f.read_text(encoding="utf-8")), s), indent=2,
                       ensure_ascii=False), encoding="utf-8")

    (stage / "LICENSE-DATA").write_text(ODBL, encoding="utf-8")

    manifest = {
        "version": args.version,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "corpus_a": a_stats,
        "osmcha_labels": l_stats,
        "geography_corpus_a": geo,
        "pseudonymisation": "salted SHA-256 of uid, first 16 hex chars; usernames and "
                            "reviewer names removed; salt not published",
        "licence": "ODbL-1.0, (c) OpenStreetMap contributors",
    }
    (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    card = build_card(manifest, args.version)
    (stage / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    (REPO / "docs" / "DATASET_CARD.md").write_text(card, encoding="utf-8")

    zip_path = OUT / f"mapguard-dataset-{args.version}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                z.write(f, str(f.relative_to(stage)).replace("\\", "/"))

    sums = OUT / f"SHA256SUMS-{args.version}.txt"
    sums.write_text(f"{sha256_file(zip_path)}  {zip_path.name}\n", encoding="utf-8")

    print(f"\n{zip_path}  ({zip_path.stat().st_size/1e6:.1f} MB)")
    print(f"{sums}")
    print(f"sha256: {sha256_file(zip_path)[:32]}...")

    notes = OUT / f"RELEASE_NOTES-{args.version}.md"
    notes.write_text(card, encoding="utf-8")

    if args.publish:
        print("\ncreating GitHub release...")
        subprocess.run(
            ["gh", "release", "create", args.version, str(zip_path), str(sums),
             "--title", f"MapGuard dataset {args.version}",
             "--notes-file", str(notes)],
            cwd=REPO, check=True)
        print("published")
    else:
        print(f"\nnot published. To publish:\n"
              f"  gh release create {args.version} {zip_path} {sums} \\\n"
              f"    --title \"MapGuard dataset {args.version}\" --notes-file {notes}")
    return 0


def build_card(m: dict, version: str) -> str:
    a, lab, geo = m["corpus_a"], m["osmcha_labels"], m["geography_corpus_a"]
    geo_rows = "\n".join(
        f"| {k} | {v['n']:,} | {v['share']:.1%} |" for k, v in geo.items())
    return f"""# MapGuard dataset — {version}

Generated {m['generated_utc']} from commit `{m['git_sha'][:12]}`.

**This is a partial release.** Corpus A is complete and verified. The modern OSMCha
corpus is still being harvested and ships here only as far as it had progressed; see
*Coverage and gaps* below. It is versioned `v0.1.x` for that reason — `v1.0.0` is
reserved for the complete dataset.

## What this is

Changesets from OpenStreetMap labelled as damaging or not, for training and evaluating
vandalism-detection models. Two label sources with **different meanings** — see *Labels*.

## Contents

| File | Rows | Description |
|---|---|---|
| `corpus_a_changesets.parquet` | {a['n']:,} | Ovid-benchmark changesets, 2014–2019, with parsed edit counts |
| `osmcha_labels.parquet` | {lab['n']:,} | OSMCha human review decisions, 2023 onward |
| `fetch_coverage.json` | {lab.get('gaps', 0)} | windows the harvest failed to retrieve |
| `results/` | — | gate artefacts: every measurement behind the design decisions |
| `fixtures/` | — | real changesets selected as awkward cases |
| `MANIFEST.json` | — | provenance, counts, git SHA |

`corpus_a_changesets.parquet` covers **{a['n']:,}** changesets, of which
**{a['with_contents']:,}** had their full edit contents fetched and parsed, totalling
**{a['total_edits']:,} edits**.

## Labels — read this before using them

| Source | Meaning | Caveat |
|---|---|---|
| `ovid_revert_derived` | the changeset was later reverted | A revert is **not** a synonym for vandalism. Imports, accidents and mapping disputes get reverted too. Class balance is a constructed 50/50, not natural prevalence. |
| `osmcha_human_review` | a person reviewed it on osmcha.org | Reviewers only open changesets an automatic filter already flagged, so `harmful=false` here means **"flagged and cleared"**, not "clean". Both classes come from the suspicious tail. |

Real-world vandalism prevalence is well under 1%. Any metric computed on the balanced
set overstates operational performance substantially.

## Geography — Corpus A

| Region | Changesets | Share |
|---|---|---|
{geo_rows}

Europe-heavy, reflecting OpenStreetMap's real contributor distribution rather than a
sampling choice. Models evaluated on it are best evidenced on European edits.

## Coverage and gaps

`fetch_coverage.json` lists every date window the harvest failed to retrieve, with
status and retry count. Unresolved gaps at packaging time: **{lab.get('unresolved', 0)}**.
A dataset that cannot say what it failed to collect cannot support a coverage claim, so
the gaps ship with the data rather than being quietly dropped.

## Privacy

OpenStreetMap user ids are replaced with a **salted SHA-256 digest** (`user_pseudonym`,
16 hex chars). Usernames and OSMCha reviewer names are **removed entirely**. The salt is
not published.

Changeset ids are retained because they are already public and the data is
reconstructible from them through the OSM API — but do not attempt to re-identify
contributors, and do not use this dataset to make claims about individual accounts. A
label here is a judgement about an edit, not about a person.

## Licence

**ODbL-1.0. © OpenStreetMap contributors.**

This is a Derivative Database of OpenStreetMap data and must stay under ODbL. See
`LICENSE-DATA` in the archive. The repository's source code is licensed separately.

## Reproducing

Every changeset id is in the archive. Contents are re-fetchable from the OSM API:

    GET https://api.openstreetmap.org/api/0.6/changeset/<id>/download

Gate artefacts under `results/` record the measurements behind every design decision —
API latency, Overpass throughput, label coverage by year, and the leakage tripwire.
"""


if __name__ == "__main__":
    raise SystemExit(main())
