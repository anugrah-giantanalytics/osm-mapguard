"""Build altair/_dist/mapguard_altair.zip - everything one Altair round needs.

    python altair/build_zip.py --round r3_v3_explore --plan altair/plans/r3_v3_explore.txt
    python altair/build_zip.py --round r3_x --plan ... --data dist/mapguard-dataset-v1.1.0/feature_store

Contents: the training bundle (src/, train/, data/interim, frozen splits, BUNDLE_MANIFEST.json -
see cluster/bundle.py) plus altair/: the job script, the WHERE job, README, ROUND.txt (the
round's folder name on the cluster), PLAN.txt (the steps), where_targets.parquet.
Side tables keyed by changeset id go into data/interim next to the store when they exist:
object_context.parquet (laptop-built) and where_features.parquet (cluster-built).

Privacy: the feature store in the zip is the pseudonymised one. where_targets.parquet holds
raw OSM user ids (public data, needed to find each author in the planet changeset dump) for
the 46,578 changesets; it exists only inside this git-ignored zip and the private cluster
folder, and the WHERE output it produces is keyed by changeset id only.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "altair" / "_dist"
SHIPPED = ("run_round2.sh", "where_job.py", "README.md")
SIDE_TABLES = {
    "object_context.parquet": Path("C:/osm-data/probes/object_context_A_MN.parquet"),
    "where_features.parquet": Path("C:/osm-data/where/where_features.parquet"),
}


def newest_store() -> Path:
    stores = sorted((REPO / "dist").glob("mapguard-dataset-v*/feature_store"),
                    key=lambda p: [int(x) for x in p.parent.name.split("-v")[-1].split(".")])
    if not stores:
        raise SystemExit("no dist/mapguard-dataset-v*/feature_store - run scripts/package_release.py")
    return stores[-1]


def where_targets() -> bytes:
    sys.path.insert(0, str(REPO / "src"))
    import pandas as pd

    from mapguard.config import interim_dir

    cs = pd.read_parquet(interim_dir() / "changesets.parquet", columns=["cid", "uid", "created_at"])
    cs = cs.dropna(subset=["uid"]).astype({"cid": "int64", "uid": "int64"})
    buf = io.BytesIO()
    cs.to_parquet(buf, index=False)
    print(f"where targets: {len(cs):,} changesets, {cs.uid.nunique():,} authors")
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--round", required=True, help="folder name of this round on the cluster")
    ap.add_argument("--plan", required=True, help="step plan, e.g. altair/plans/r3_v3_explore.txt")
    ap.add_argument("--data", default=None, help="feature store dir (default: newest packaged)")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.round):
        raise SystemExit("--round: letters, digits, _ . - only")
    data = Path(args.data) if args.data else newest_store()
    DIST.mkdir(parents=True, exist_ok=True)
    inner = DIST / "_bundle.zip"
    subprocess.check_call([sys.executable, str(REPO / "cluster" / "bundle.py"),
                           "--data", str(data), "--out", str(inner)])

    out = DIST / "mapguard_altair.zip"
    with zipfile.ZipFile(inner) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for info in src.infolist():
            z.writestr(info, src.read(info.filename),
                       compress_type=zipfile.ZIP_STORED if info.filename.endswith(".parquet")
                       else zipfile.ZIP_DEFLATED)

        def add(name: str, data: bytes, mode: int = 0o644) -> None:
            info = zipfile.ZipInfo(f"altair/{name}")
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)

        for name in SHIPPED:
            add(name, (REPO / "altair" / name).read_bytes().replace(b"\r\n", b"\n"),
                0o755 if name.endswith(".sh") else 0o644)
        add("ROUND.txt", (args.round + "\n").encode())
        add("PLAN.txt", Path(args.plan).read_bytes().replace(b"\r\n", b"\n"))
        for name, src_path in SIDE_TABLES.items():
            if src_path.exists():
                info = zipfile.ZipInfo(f"data/interim/{name}")
                info.compress_type = zipfile.ZIP_STORED
                z.writestr(info, src_path.read_bytes())
                print(f"side table: {name} ({src_path.stat().st_size / 1e6:.1f} MB)")
            else:
                print(f"side table: {name} not found at {src_path} - not shipped")
        add("where_targets.parquet", where_targets())
    inner.unlink()
    (DIST / "run_round2.sh").write_bytes((REPO / "altair" / "run_round2.sh").read_bytes()
                                         .replace(b"\r\n", b"\n"))
    for stale in ("smoke_test.sh",):
        (DIST / stale).unlink(missing_ok=True)
    print(f"\n{out}  ({out.stat().st_size / 1e6:.1f} MB)  round: {args.round}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
