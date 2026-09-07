"""Pack a cluster task into one uploadable zip.

The Altair portal takes files through a drag-and-drop box. Uploading a source tree file
by file is how versions drift apart, so a task goes up as a single zip with a manifest
recording exactly what was in it: git SHA, feature-schema version, row counts, and the
sha256 of every data file.

That manifest is not bookkeeping for its own sake. Every result the paper reports has to
name the commit and the data window it came from, and reconstructing that after the fact
from a folder of uploads is not possible.

    python cluster/bundle.py --task 00_env_check
    python cluster/bundle.py --task 02_stage2_train --data /c/osm-data/interim
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]
CLUSTER = REPO / "cluster"
OUT = CLUSTER / "_bundles"

# Which tasks need the feature store. 00_env_check deliberately needs nothing, so it can
# be run the moment an image is available.
NEEDS_DATA = {
    "00_env_check": (),
    "01_baselines": ("changesets.parquet",),
    "02_stage2_train": ("changesets.parquet", "edits.parquet", "user_seq.parquet"),
    "03_ablations": ("changesets.parquet", "edits.parquet", "user_seq.parquet"),
    "04_stage1_pretrain": ("changesets.parquet", "edits.parquet"),
}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip())
    except Exception:
        return True


def sha256(path: Path, limit: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        read = 0
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            read += len(chunk)
            if read >= limit:      # enough to detect a swap; full hash of GBs is wasteful
                h.update(b"TRUNCATED")
                break
    return h.hexdigest()[:16]


def parquet_rows(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(NEEDS_DATA))
    ap.add_argument("--data", default=None, help="feature store dir (default: MAPGUARD_DATA/interim)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    task_dir = CLUSTER / args.task
    if not task_dir.exists():
        raise SystemExit(f"no such task folder: {task_dir}")

    from mapguard.config import interim_dir

    data_dir = Path(args.data) if args.data else interim_dir()
    OUT.mkdir(parents=True, exist_ok=True)
    zip_path = Path(args.out) if args.out else OUT / f"{args.task}.zip"

    manifest = {
        "task": args.task,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "python_packed_with": sys.version.split()[0],
        "files": [],
        "data": {},
        "missing_data": [],
    }

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        # the notebook and its README
        for f in sorted(task_dir.iterdir()):
            if f.is_file():
                z.write(f, f"{args.task}/{f.name}")
                manifest["files"].append(f"{args.task}/{f.name}")

        # the library, so the notebook stays thin
        for f in sorted((REPO / "src").rglob("*.py")):
            rel = f.relative_to(REPO)
            z.write(f, str(rel).replace("\\", "/"))
            manifest["files"].append(str(rel).replace("\\", "/"))

        # small tracked artefacts the notebooks read
        for extra in ("requirements.txt", "data/labels/ovid_labels.tsv"):
            f = REPO / extra
            if f.exists():
                z.write(f, extra)
                manifest["files"].append(extra)
        fixtures = REPO / "data" / "fixtures"
        if fixtures.exists():
            for f in sorted(fixtures.glob("*.json")):
                z.write(f, f"data/fixtures/{f.name}")

        # the feature store
        for name in NEEDS_DATA[args.task]:
            src = data_dir / name
            if not src.exists():
                manifest["missing_data"].append(name)
                continue
            z.write(src, f"data/interim/{name}")
            manifest["data"][name] = {
                "bytes": src.stat().st_size,
                "rows": parquet_rows(src),
                "sha256_16": sha256(src),
            }

        report = data_dir / "build_report.json"
        if report.exists():
            z.write(report, "data/interim/build_report.json")
            manifest["feature_schema"] = json.loads(report.read_text()).get("schema_version")

        z.writestr(f"{args.task}/BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2))

    size_mb = zip_path.stat().st_size / 1e6
    print(json.dumps(manifest, indent=2)[:1200])
    print(f"\n{zip_path}  ({size_mb:.1f} MB)")
    if manifest["missing_data"]:
        print(f"MISSING: {manifest['missing_data']} - build the feature store first")
    if manifest["git_dirty"]:
        print("WARNING: working tree is dirty, so git_sha does not fully describe this bundle")
    print("\nUpload through the Altair portal: Files -> +Upload -> File")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
