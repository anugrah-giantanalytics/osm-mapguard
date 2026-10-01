"""Build altair/_dist/mapguard_altair.zip - everything one Altair job needs.

    python altair/build_zip.py
    python altair/build_zip.py --data dist/mapguard-dataset-v1.1.0/feature_store

Contents: the standard training bundle (src/, train/, data/interim, frozen splits,
BUNDLE_MANIFEST.json - see cluster/bundle.py) plus altair/ (job scripts and README).
The default data is the pseudonymised feature store of the newest packaged dataset, so the
zip never carries raw OSM user ids.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "altair" / "_dist"


def newest_store() -> Path:
    stores = sorted((REPO / "dist").glob("mapguard-dataset-v*/feature_store"),
                    key=lambda p: [int(x) for x in p.parent.name.split("-v")[-1].split(".")])
    if not stores:
        raise SystemExit("no dist/mapguard-dataset-v*/feature_store - run scripts/package_release.py")
    return stores[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=None, help="feature store dir (default: newest packaged)")
    args = ap.parse_args()
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
        for f in sorted((REPO / "altair").iterdir()):
            if f.is_file() and f.suffix in (".sh", ".md"):
                info = zipfile.ZipInfo(f"altair/{f.name}")
                info.external_attr = (0o755 if f.suffix == ".sh" else 0o644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, f.read_bytes().replace(b"\r\n", b"\n"))
    inner.unlink()
    for f in ("run_round2.sh", "smoke_test.sh"):
        (DIST / f).write_bytes((REPO / "altair" / f).read_bytes().replace(b"\r\n", b"\n"))
    print(f"\n{out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"job scripts: {DIST / 'run_round2.sh'}, {DIST / 'smoke_test.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
