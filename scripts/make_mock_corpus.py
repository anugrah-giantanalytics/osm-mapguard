"""A tiny synthetic corpus in the real schema, so WS2/3/4 are not blocked on WS1.

CHECKLIST item 3.8. The data pipeline gates everyone else, and "wait two weeks for the
fetch" is how three of four workstreams lose a fortnight. This writes 200 changesets in
exactly the schema the real featuriser will emit, with plausible ranges and a planted
signal, so the evaluation harness, the branches and the training loop can all be built
and debugged against something today.

It is deliberately easy: geometry vandalism gets a large centroid shift, semantic
vandalism gets junk text. A model that cannot separate this is broken. A model that can
has proved nothing - that is what the real corpus is for.

    python scripts/make_mock_corpus.py
    python scripts/make_mock_corpus.py --n 1000 --out data/mock
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyarrow as pa
import pyarrow.parquet as pq

EDITORS = ["iD 2.20.2", "JOSM/1.5 (18463 en)", "StreetComplete 63.4", "Potlatch 2", "Rapid 2.1"]
CLEAN_COMMENTS = [
    "add missing footway", "fix building outline", "update opening hours",
    "survey: new bus stop", "align road to imagery", "add address",
]
JUNK_NAMES = ["aaaaaaa", "XXXXXXXX", "test123", "asdfgh", "hello world lol"]
REAL_NAMES = ["Church Street", "Bakers Lane", "Riverside Park", "Kings Road", "Mill Bridge"]


def build(n: int, seed: int):
    rng = random.Random(seed)
    cs_rows, edit_rows, geom_rows = [], [], []

    for i in range(n):
        cid = 20_000_000 + i * 137
        label = 1 if i % 2 == 0 else 0
        # Three vandalism flavours, so the per-type breakdown in section 11 has something
        # to exercise before the real taxonomy exists.
        vtype = ["geometric", "semantic", "destructive"][i % 3] if label else "none"
        uid = 1000 + (i % 60)
        n_edits = rng.randint(1, 40) if vtype != "destructive" else rng.randint(200, 2000)

        max_shift = rng.uniform(2_000, 400_000) if vtype == "geometric" else rng.uniform(0, 50)
        text_blob = " ".join(
            rng.choice(JUNK_NAMES if vtype == "semantic" else REAL_NAMES) for _ in range(3)
        )

        cs_rows.append({
            "cid": cid,
            "label": label,
            "vandalism_type": vtype,
            "uid": uid,
            "created_at": f"2017-{1 + i % 12:02d}-{1 + i % 28:02d}T12:00:00Z",
            "num_changes": n_edits,
            "n_edits_seen": min(n_edits, 256),
            "sample_ratio": round(min(1.0, 256 / n_edits), 4),
            "n_create": int(n_edits * 0.5),
            "n_modify": int(n_edits * 0.3),
            "n_delete": n_edits - int(n_edits * 0.5) - int(n_edits * 0.3),
            "n_node": int(n_edits * 0.7),
            "n_way": n_edits - int(n_edits * 0.7),
            "n_relation": 0,
            "comment": rng.choice(CLEAN_COMMENTS),
            "comment_length": rng.randint(4, 60),
            "editor": rng.choice(EDITORS),
            "account_age_days": rng.uniform(0, 3000),
            "max_centroid_shift_m": round(max_shift, 2),
            "p95_centroid_shift_m": round(max_shift * 0.6, 2),
            "max_abs_d_area_m2": round(rng.uniform(0, 1e6), 2),
            "max_area_ratio": round(rng.uniform(0.5, 200), 4),
            "total_d_points": rng.randint(-500, 500),
            "text_blob": text_blob,
        })

        for j in range(min(n_edits, 256)):
            edit_rows.append({
                "cid": cid,
                "position": j,
                "action": rng.choice(["create", "modify", "delete"]),
                "otype": rng.choice(["node", "way"]),
                "oid": rng.randint(1, 10**9),
                "version": rng.randint(1, 12),
                "n_tags": rng.randint(0, 8),
                "n_refs": rng.randint(0, 30),
                "centroid_shift_m": round(rng.uniform(0, max_shift), 2),
                "d_length_m": round(rng.uniform(-500, 500), 2),
                "d_area_m2": round(rng.uniform(-1e4, 1e4), 2),
                "text": rng.choice(JUNK_NAMES if vtype == "semantic" else REAL_NAMES) if j % 5 == 0 else "",
            })

        geom_rows.append({
            "cid": cid,
            "route": "adiff",
            "n_geom_actions": min(n_edits, 256),
            "n_with_old": int(min(n_edits, 256) * 0.6),
            "max_centroid_shift_m": round(max_shift, 2),
            "max_hausdorff_m": round(max_shift * 1.1, 2),
        })

    return cs_rows, edit_rows, geom_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "data" / "mock"
    out.mkdir(parents=True, exist_ok=True)

    cs, ed, gm = build(args.n, args.seed)
    for name, rows in (("changesets", cs), ("edits", ed), ("geometry", gm)):
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out / f"{name}.parquet", compression="zstd")
        print(f"{name:11} {len(rows):>7} rows -> {out / (name + '.parquet')}")

    print("\nSchema is the contract. If the real featuriser diverges from it, fix one or")
    print("the other deliberately and bump the schema version - do not let them drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
