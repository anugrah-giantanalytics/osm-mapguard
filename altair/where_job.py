"""WHERE features from the OSM planet changeset dump - runs on the cluster, CPU only.

    python where_job.py --targets where_targets.parquet --workdir <dir>

For every target changeset (cid, uid, created_at): where is it, relative to where the same
author mapped **before** it? Only changesets created strictly earlier count, so nothing from
the future leaks in.

Source: planet.openstreetmap.org/planet/changesets-latest.osm.bz2 - metadata of every
changeset since 2005 (id, uid, time, bbox). A single public file download, not API calls.

Steps (each resumable; rerun the same command after a crash):
  1. download     <workdir>/changesets-latest.osm.bz2  (HTTP Range resume)
  2. extract      <workdir>/author_points.npz  - (uid, t, lat, lon, cid) for target authors only
  3. features     <workdir>/where_features.parquet - one row per target cid, no uid
Progress: <workdir>/where.log and <workdir>/STATUS.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from array import array
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://planet.openstreetmap.org/planet/changesets-latest.osm.bz2"
UA = {"User-Agent": "MapGuard research (course project; one-off dump download)"}
ATTR = re.compile(r'(\w+)="([^"]*)"')
RECENT = 5000            # distance statistics use the author's most recent prior changesets
EARTH_KM = 6371.0


def log(workdir: Path, msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(workdir / "where.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    (workdir / "STATUS").write_text(line + "\n", encoding="utf-8")


def epoch(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()


# ------------------------------------------------------------------ 1. download
def download(workdir: Path) -> Path:
    dest = workdir / "changesets-latest.osm.bz2"
    done = workdir / "download.done"
    if done.exists() and dest.exists():
        log(workdir, f"download: already complete ({dest.stat().st_size / 1e9:.2f} GB)")
        return dest
    req = urllib.request.Request(URL, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=120) as r:
        final_url, total = r.geturl(), int(r.headers["Content-Length"])
        modified = r.headers.get("Last-Modified")
    (workdir / "source.json").write_text(json.dumps(
        {"url": final_url, "bytes": total, "last_modified": modified}, indent=1))
    log(workdir, f"download: {final_url} ({total / 1e9:.2f} GB, {modified})")
    for attempt in range(1, 31):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= total:
            break
        try:
            req = urllib.request.Request(final_url, headers={**UA, "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "ab") as fh:
                t0, got = time.time(), 0
                while chunk := r.read(8 << 20):
                    fh.write(chunk)
                    got += len(chunk)
                    if got % (512 << 20) < (8 << 20):
                        rate = got / max(time.time() - t0, 1) / 1e6
                        log(workdir, f"download: {(have + got) / 1e9:.2f}/{total / 1e9:.2f} GB "
                                     f"({rate:.1f} MB/s)")
        except Exception as exc:                        # network blip: resume from what we have
            log(workdir, f"download attempt {attempt} interrupted: {exc!r}; resuming in 30 s")
            time.sleep(30)
    if dest.stat().st_size != total:
        raise SystemExit(f"download incomplete: {dest.stat().st_size} of {total} bytes")
    done.write_text(str(total))
    log(workdir, "download: complete")
    return dest


# ------------------------------------------------------------------ 2. extract
def decompressor(path: Path):
    for tool, args in (("lbzip2", ["-dc", "-n", str(max(1, (os.cpu_count() or 2) - 1))]),
                       ("pbzip2", ["-dc"]), ("bzip2", ["-dc"])):
        if shutil.which(tool):
            return tool, subprocess.Popen([tool, *args, str(path)], stdout=subprocess.PIPE,
                                          bufsize=1 << 22)
    return "python-bz2", None


def extract(workdir: Path, dump: Path, targets: pd.DataFrame) -> Path:
    out = workdir / "author_points.npz"
    if out.exists():
        log(workdir, "extract: already done")
        return out
    uids = set(int(u) for u in targets["uid"])
    target_cids = set(int(c) for c in targets["cid"])
    U, T, LA, LO, C = array("q"), array("d"), array("f"), array("f"), array("q")
    tool, proc = decompressor(dump)
    log(workdir, f"extract: decompressing with {tool}; {len(uids):,} target authors")
    if proc is None:
        import bz2
        stream = bz2.open(dump, "rb")
    else:
        stream = proc.stdout
    n_cs = n_keep = 0
    t0 = time.time()
    for raw in stream:
        s = raw.lstrip()
        if not s.startswith(b"<changeset "):
            continue
        n_cs += 1
        if n_cs % 10_000_000 == 0:
            log(workdir, f"extract: {n_cs / 1e6:.0f}M changesets read, {n_keep:,} kept, "
                         f"{(time.time() - t0) / 60:.1f} min")
        i = s.find(b' uid="')
        if i < 0:
            continue
        j = s.find(b'"', i + 6)
        uid = int(s[i + 6:j])
        if uid not in uids:
            continue
        a = dict(ATTR.findall(s.decode("utf-8", "replace")))
        if "min_lat" not in a:
            continue                                   # empty changeset: no location
        lat = (float(a["min_lat"]) + float(a["max_lat"])) / 2
        lon = (float(a["min_lon"]) + float(a["max_lon"])) / 2
        U.append(uid); T.append(epoch(a["created_at"])); LA.append(lat); LO.append(lon)
        C.append(int(a["id"]))
        n_keep += 1
    if proc is not None and proc.wait() != 0:
        raise SystemExit(f"{tool} failed with exit code {proc.returncode}")
    np.savez(out, uid=np.frombuffer(U, dtype=np.int64), t=np.frombuffer(T, dtype=np.float64),
             lat=np.frombuffer(LA, dtype=np.float32), lon=np.frombuffer(LO, dtype=np.float32),
             cid=np.frombuffer(C, dtype=np.int64))
    found = len(target_cids & set(np.frombuffer(C, dtype=np.int64).tolist()))
    log(workdir, f"extract: done - {n_cs:,} changesets in dump, {n_keep:,} by target authors, "
                 f"{found:,}/{len(target_cids):,} target changesets have a bbox, "
                 f"{(time.time() - t0) / 60:.1f} min")
    return out


# ------------------------------------------------------------------ 3. features
def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def features(workdir: Path, points: Path, targets: pd.DataFrame) -> Path:
    out = workdir / "where_features.parquet"
    z = np.load(points)
    order = np.lexsort((z["t"], z["uid"]))
    uid, t, lat, lon, cid = (z[k][order] for k in ("uid", "t", "lat", "lon", "cid"))
    lat, lon = lat.astype(np.float64), lon.astype(np.float64)
    pos = {int(c): i for i, c in enumerate(cid)}
    starts = {}
    if len(uid):
        bounds = np.flatnonzero(np.diff(uid)) + 1
        for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(uid)]):
            starts[int(uid[s])] = (int(s), int(e))
    rows = []
    for c, u, ts in targets[["cid", "uid", "created_at"]].itertuples(index=False):
        c, u = int(c), int(u)
        r = {"cid": c, "where_cur_has_bbox": float(c in pos)}
        s, e = starts.get(u, (0, 0))
        tc = epoch(ts)
        k = s + int(np.searchsorted(t[s:e], tc, side="left"))   # rows s..k-1 are strictly earlier
        r["where_n_prior"] = float(k - s)
        if c in pos and k > s:
            lo = max(s, k - RECENT)
            plat, plon = lat[lo:k], lon[lo:k]
            clat, clon = lat[pos[c]], lon[pos[c]]
            home_lat, home_lon = np.median(plat), np.median(plon)
            d = haversine(clat, clon, plat, plon)
            spread = haversine(home_lat, home_lon, plat, plon)
            cells = set(zip(np.floor(plat).astype(int).tolist(), np.floor(plon).astype(int).tolist()))
            r.update({
                "where_dist_home_km": float(haversine(clat, clon, home_lat, home_lon)),
                "where_dist_nearest_km": float(d.min()),
                "where_dist_prev_km": float(d[-1]),
                "where_frac_within_25km": float((d <= 25).mean()),
                "where_spread_km": float(np.median(spread)),
                "where_cells_1deg": float(len(cells)),
                "where_new_cell": float((int(np.floor(clat)), int(np.floor(clon))) not in cells),
            })
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    known = df["where_dist_home_km"].notna().mean() if "where_dist_home_km" in df else 0.0
    log(workdir, f"features: wrote {out} - {len(df):,} rows, {known:.1%} with a location "
                 f"relative to prior history")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", required=True)
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_parquet(args.targets)
    t0 = time.time()
    dump = download(workdir)
    points = extract(workdir, dump, targets)
    features(workdir, points, targets)
    log(workdir, f"WHERE FINISHED in {(time.time() - t0) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
