"""Read the OSM changeset dump by byte range: five years of changeset metadata, no 8.7 GB file.

`planet.openstreetmap.org/planet/changesets-latest.osm.bz2` holds the metadata of every
changeset since 2005 - id, author, time, bbox, edit count and every changeset tag, including
the comment and `created_by`. It is the only bulk source of *revert* information: the API
cannot search comments, OSMCha has no reviews before 2023 (measured, see
results/gate2a_osmcha_coverage.json), and Overpass does not index changesets.

Two facts make it affordable, both measured on 2026-09-18:

  * The file is **multistream bzip2**: every ~900 KB of compressed data is an independent
    stream starting with a byte-aligned `BZh91AY&SY` header. Any byte range can be
    decompressed from its first header onwards, so only the part covering the wanted years
    is downloaded, in parallel ranges.
  * Changesets are **sorted by id**, and ids increase with time, so the byte offset where a
    date starts can be found by bisection with a few 2 MB probes.

Nothing is written to disk except what `scripts/mine_changesets.py` keeps: revert changesets
with their targets, a fixed hash sample of all changesets (the negative pool), and monthly
counts.

Revert detection uses what the revert tools actually write (inspected on real 2026 data):

    osm-revert (revert.monicz.dev)   tag  revert:id=https://www.openstreetmap.org/changeset/N[;...]
    JOSM reverter plugin             created_by=reverter_plugin/...  comment "Revert changeset N"
    simple_revert.py                 created_by=simple_revert.py     comment "Reverting N by user"
    by hand                          comment matching revert/reverted/reverting ... N
"""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass

import requests

DUMP_URL = "https://planet.openstreetmap.org/planet/changesets-latest.osm.bz2"
STREAM_HEADER = re.compile(rb"BZh[1-9]1AY&SY")
PROBE_BYTES = 2_500_000          # larger than one bzip2 -9 stream (~900 KB compressed)
SAMPLE_MOD = 400                 # keep 1 in 400 changesets as the negative pool

_ELEMENT = re.compile(r"<changeset\s([^>]*?)(/>|>(.*?)</changeset>)", re.S)
_ATTR = re.compile(r'(\w+)="([^"]*)"')
_TAG = re.compile(r'<tag k="([^"]*)" v="([^"]*)"\s*/>')
_HEAD = re.compile(r'<changeset id="(\d+)" created_at="([^"]+)"')
_REVERT_WORD = re.compile(r"revert", re.I)
_URL_ID = re.compile(r"changeset/(\d{4,})")
_BIG_NUMBER = re.compile(r"(?<![\d.])(\d{6,10})(?![\d.])")
# URLs and path-like references (forum topics c.osm.org/t/144453, note/5328952, ticket #...)
# carry numbers that are not changeset ids; only changeset/N survives from a URL.
_URLISH = re.compile(r"https?://\S+|\b[\w.]+/\d+")
_PARTIAL = re.compile(r"\bpartial|\bpart of|\brework|\bteilweise|\bparcial|\bpartiel", re.I)


def comment_ids(comment: str) -> list[int]:
    """Changeset ids named in a revert comment: changeset URLs, then bare 6-10 digit numbers
    outside any URL or path."""
    ids = [int(x) for x in _URL_ID.findall(comment)]
    ids += [int(x) for x in _BIG_NUMBER.findall(_URLISH.sub(" ", comment))]
    return ids


def user_agent() -> dict:
    from mapguard.config import HEADERS

    return dict(HEADERS)


# ------------------------------------------------------------------------- transport


def resolve(url: str = DUMP_URL) -> dict:
    """Follow the planet redirect to the dated file, so every run names its exact source."""
    r = requests.head(url, allow_redirects=True, timeout=60, headers=user_agent())
    r.raise_for_status()
    return {"url": r.url, "bytes": int(r.headers["Content-Length"]),
            "last_modified": r.headers.get("Last-Modified"),
            "file": r.url.rsplit("/", 1)[-1]}


def fetch_range(url: str, start: int, end: int, retries: int = 6) -> bytes:
    """Bytes [start, end) with retries. Returns fewer bytes only at end of file."""
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={**user_agent(), "Range": f"bytes={start}-{end - 1}"},
                             timeout=180)
            if r.status_code in (200, 206):
                return r.content
            last = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last = repr(exc)[:200]
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"range {start}-{end} failed: {last}")


# ------------------------------------------------------------------ decompression


def iter_streams(buf: bytes, base: int, stop_after: int | None = None):
    """Yield (stream_offset, text) for each complete bzip2 stream in `buf`.

    `base` is the file offset of buf[0]. Starts at the first stream header. Stops at the
    first stream whose start is >= `stop_after` *after* yielding it (that one is the overlap
    that completes an element cut at the chunk boundary), or at an incomplete stream.
    """
    import bz2

    m = STREAM_HEADER.search(buf)
    if not m:
        return
    pos = m.start()
    while pos < len(buf):
        d = bz2.BZ2Decompressor()
        try:
            out = d.decompress(buf[pos:])
        except (OSError, EOFError):
            return
        if not d.eof:
            return
        yield base + pos, out.decode("utf-8", "replace")
        if stop_after is not None and base + pos >= stop_after:
            return
        pos = len(buf) - len(d.unused_data)


def first_changeset_at(url: str, offset: int) -> tuple[int, str] | None:
    """(id, created_at) of the first changeset whose element starts after `offset`."""
    buf = fetch_range(url, offset, offset + PROBE_BYTES)
    for _, text in iter_streams(buf, offset):
        m = _HEAD.search(text)
        if m:
            return int(m.group(1)), m.group(2)
    return None


def offset_for_date(url: str, size: int, since: str, tolerance: int = 32_000_000,
                    log=print) -> int:
    """Largest probed offset whose first changeset is still before `since`.

    Starting a little early is harmless - rows before `since` are filtered by date - while
    starting late would silently drop data, so the search errs early.
    """
    lo, hi = 0, size
    while hi - lo > tolerance:
        mid = (lo + hi) // 2
        got = first_changeset_at(url, mid)
        log(f"  probe {mid / 1e9:6.3f} GB -> {got}")
        if got is None or got[1] >= since:
            hi = mid
        else:
            lo = mid
    return lo


# -------------------------------------------------------------------------- parsing


@dataclass
class Revert:
    cid: int
    created_at: str
    uid: int | None
    kind: str
    created_by: str
    comment: str
    targets: list[int]
    partial: bool = False


def revert_targets(cid: int, tags: dict[str, str]) -> tuple[str, list[int]] | None:
    """(kind, target ids) if this changeset reverts earlier ones, else None.

    Targets must be earlier changesets (smaller id). Numbers in a free-text comment are only
    taken when the comment talks about reverting, because comments also carry note ids,
    ticket numbers and phone numbers.
    """
    created_by = tags.get("created_by", "")
    comment = tags.get("comment", "")
    found: list[int] = []
    kind = None
    if "revert:id" in tags:
        kind = "osm-revert"
        found = [int(x) for x in re.findall(r"\d{4,}", tags["revert:id"])]
    elif "reverter_plugin" in created_by:
        kind = "josm-reverter"
        found = comment_ids(comment)
    elif "simple_revert" in created_by:
        kind = "simple_revert"
        found = comment_ids(comment)
    elif _REVERT_WORD.search(comment):
        kind = "comment"
        found = comment_ids(comment)
    if kind is None:
        return None
    targets = sorted({t for t in found if 1_000 <= t < cid})
    return (kind, targets) if targets else None


def parse_element(attrs_text: str, body: str | None) -> tuple[dict, dict]:
    attrs = {k: html.unescape(v) for k, v in _ATTR.findall(attrs_text)}
    tags = {html.unescape(k): html.unescape(v) for k, v in _TAG.findall(body or "")}
    return attrs, tags


def sampled(cid: int, mod: int = SAMPLE_MOD) -> bool:
    """Fixed 1-in-`mod` sample by a multiplicative hash of the id - never by content."""
    return ((cid * 2654435761) & 0xFFFFFFFF) % mod == 0


def scan_text(text: str, since: str, until: str, out: dict, first_only: bool = False) -> str:
    """Parse complete changeset elements in `text`; return the unparsed tail.

    `out` collects reverts, the hash sample, and monthly counts. The tail (an element cut by
    the stream boundary) is carried into the next stream by the caller. `first_only` parses
    just the element at the start of `text` - used to finish the one element a chunk
    started but the next chunk's first stream completes.
    """
    last_end = 0
    matches = _ELEMENT.finditer(text)
    if first_only:
        m0 = _ELEMENT.match(text)
        matches = [m0] if m0 else []
    for m in matches:
        last_end = m.end()
        attrs_text, body = m.group(1), m.group(3)
        head = _ATTR.findall(attrs_text[:120])
        cid = created = None
        for k, v in head:
            if k == "id":
                cid = int(v)
            elif k == "created_at":
                created = v
        if cid is None or not created or created < since or created >= until:
            continue
        month = created[:7]
        out["months"][month] = out["months"].get(month, 0) + 1
        is_sample = sampled(cid)
        maybe_revert = body is not None and ("evert" in body or "EVERT" in body)
        if not (is_sample or maybe_revert):
            continue
        attrs, tags = parse_element(attrs_text, body)
        uid = int(attrs["uid"]) if attrs.get("uid", "").isdigit() else None
        if maybe_revert:
            rv = revert_targets(cid, tags)
            if rv:
                comment = tags.get("comment", "")
                out["reverts"].append(Revert(cid, created, uid, rv[0], tags.get("created_by", "")[:120],
                                             comment[:500], rv[1], bool(_PARTIAL.search(comment))))
        if is_sample:
            out["sample"].append({
                "cid": cid, "created_at": created, "closed_at": attrs.get("closed_at"),
                "uid": uid, "num_changes": int(attrs.get("num_changes", 0) or 0),
                "min_lat": attrs.get("min_lat"), "min_lon": attrs.get("min_lon"),
                "max_lat": attrs.get("max_lat"), "max_lon": attrs.get("max_lon"),
                "comments_count": int(attrs.get("comments_count", 0) or 0),
                "editor": tags.get("created_by", "")[:120], "comment": tags.get("comment", "")[:500],
                "tags": tags,
            })
    # Keep everything after the last complete element: it may be the start of one that
    # finishes in the next stream - including a start cut inside "<changeset" itself.
    return text[last_end:]


def scan_chunk(url: str, start: int, end: int, size: int, since: str, until: str) -> dict:
    """Scan the streams that *start* in [start, end), finishing the last element with the
    first stream after `end`. Adjacent chunks therefore never double-count or drop an element.
    """
    stop = min(size, end + PROBE_BYTES)
    buf = fetch_range(url, start, stop)
    out = {"reverts": [], "sample": [], "months": {}, "first_cid": None, "last_cid": None,
           "streams": 0}
    carry = ""
    own_started = False
    for offset, text in iter_streams(buf, start, stop_after=end):
        if offset >= end:
            # overlap stream: only finish the element that began inside this chunk
            if carry:
                scan_text(carry.lstrip() + text, since, until, out, first_only=True)
            break
        if not own_started:
            # drop the partial element whose opening tag belongs to the previous chunk
            k = text.find("<changeset ")
            text = text[k:] if k >= 0 else ""
            own_started = True
        carry = scan_text(carry + text, since, until, out)
        out["streams"] += 1
    ids = [r.cid for r in out["reverts"]] + [s["cid"] for s in out["sample"]]
    if ids:
        out["first_cid"], out["last_cid"] = min(ids), max(ids)
    return out
