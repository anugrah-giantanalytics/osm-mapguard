"""The OSM changeset replication stream - the project's route to genuinely current data.

PROJECT_PLAN v1 said current OSM data was reachable "only through the OSM API or
Overpass, one changeset at a time", and concluded that the temporal test could only ever
be a demo. That was wrong. Two replication streams are live to the current minute, free,
and need no account:

    /replication/changesets/  changeset metadata, one gzipped .osm per minute (~3 KB)
    /replication/minute/      osmChange diffs, one per minute

Verified 2026-09-04: changeset sequence 7,170,065 at 04:12:51Z, minute sequence
7,271,812 at 04:12:47Z. See probes P4 and P5 in results/gate0d_review_probes.json.

This module reads the changeset stream, which is what Corpus D and the revert miner
need. A month of it is roughly 125 MB gzipped, so a window can be harvested on a laptop
without touching the 8.72 GB planet dump.

Sequence numbers map to paths as three zero-padded triples:
    7170060 -> 007/170/060.osm.gz
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator
from xml.etree import ElementTree as ET

from mapguard.config import REPLICATION_CHANGESETS, REPLICATION_MINUTE
from mapguard.data.http import PoliteSession


@dataclass
class ReplicationState:
    sequence: int
    timestamp: str


def seq_to_path(seq: int) -> str:
    return f"{seq // 1_000_000:03d}/{(seq // 1000) % 1000:03d}/{seq % 1000:03d}"


def changeset_state(session: PoliteSession) -> ReplicationState | None:
    """Current head of the changeset stream, from state.yaml."""
    r = session.get(f"{REPLICATION_CHANGESETS}/state.yaml")
    if not r.ok:
        return None
    seq, ts = None, ""
    for line in r.text.splitlines():
        if line.startswith("sequence:"):
            seq = int(line.split(":", 1)[1].strip())
        elif line.startswith("last_run:"):
            ts = line.split(":", 1)[1].strip()
    return ReplicationState(seq, ts) if seq is not None else None


def minute_state(session: PoliteSession) -> ReplicationState | None:
    """Current head of the minute diff stream, from state.txt.

    The file escapes colons in the timestamp (`2026-09-04T04\\:12\\:47Z`), which is
    valid YAML and surprising the first time you parse it by hand.
    """
    r = session.get(f"{REPLICATION_MINUTE}/state.txt")
    if not r.ok:
        return None
    seq, ts = None, ""
    for line in r.text.splitlines():
        if line.startswith("sequenceNumber="):
            seq = int(line.split("=", 1)[1].strip())
        elif line.startswith("timestamp="):
            ts = line.split("=", 1)[1].strip().replace("\\:", ":")
    return ReplicationState(seq, ts) if seq is not None else None


@dataclass
class ChangesetRecord:
    """One changeset from the replication stream.

    Note the attribute is `num_changes` here and `changes_count` on the API. Both are
    normalised to `num_changes` by this project - see mapguard.data.osm_api.
    """

    cid: int
    created_at: str | None
    closed_at: str | None
    open: bool
    uid: int | None
    user: str | None
    num_changes: int | None
    min_lat: float | None
    min_lon: float | None
    max_lat: float | None
    max_lon: float | None
    comments_count: int | None
    tags: dict[str, str]

    @property
    def comment(self) -> str:
        return self.tags.get("comment", "")

    @property
    def editor(self) -> str:
        return self.tags.get("created_by", "")


def _f(a: dict, k: str) -> float | None:
    v = a.get(k)
    return float(v) if v not in (None, "") else None


def _i(a: dict, k: str) -> int | None:
    v = a.get(k)
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def parse_changeset_file(xml: str) -> list[ChangesetRecord]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    out = []
    for el in root.findall("changeset"):
        a = el.attrib
        out.append(
            ChangesetRecord(
                cid=int(a["id"]),
                created_at=a.get("created_at"),
                closed_at=a.get("closed_at"),
                open=a.get("open", "false") == "true",
                uid=_i(a, "uid"),
                user=a.get("user"),
                num_changes=_i(a, "num_changes"),
                min_lat=_f(a, "min_lat"),
                min_lon=_f(a, "min_lon"),
                max_lat=_f(a, "max_lat"),
                max_lon=_f(a, "max_lon"),
                comments_count=_i(a, "comments_count"),
                tags={t.attrib["k"]: t.attrib["v"] for t in el.findall("tag")},
            )
        )
    return out


def fetch_changeset_sequence(session: PoliteSession, seq: int) -> list[ChangesetRecord]:
    """One minute of changeset metadata. Returns [] on any failure, by design -
    a gap in a stream is normal and must not stop a harvest."""
    r = session.get(f"{REPLICATION_CHANGESETS}/{seq_to_path(seq)}.osm.gz")
    if not r.ok:
        return []
    try:
        # The file is gzip *content*, not a gzip transfer-encoding, so requests does not
        # unwrap it. Work from raw bytes; decoding to str first is lossy and the archive
        # will not decompress afterwards.
        raw = r.content
        if raw.startswith(bytes([0x1F, 0x8B])):  # gzip magic number
            raw = gzip.decompress(raw)
        return parse_changeset_file(raw.decode("utf-8", "replace"))
    except (OSError, UnicodeDecodeError):
        return []


def walk_changesets(
    session: PoliteSession, start_seq: int, end_seq: int
) -> Iterator[ChangesetRecord]:
    """Every changeset between two sequence numbers, inclusive, oldest first."""
    for seq in range(start_seq, end_seq + 1):
        yield from fetch_changeset_sequence(session, seq)


def seq_for_datetime(head: ReplicationState, when: datetime) -> int:
    """Estimate the sequence number for a moment in time.

    The changeset stream is one file per minute, so the arithmetic is exact enough to
    seek with and should be confirmed by reading the file you land on.
    """
    head_dt = datetime.fromisoformat(head.timestamp.replace("Z", "+00:00").split(" +")[0])
    if head_dt.tzinfo is None:
        head_dt = head_dt.replace(tzinfo=timezone.utc)
    minutes = int((head_dt - when.astimezone(timezone.utc)).total_seconds() // 60)
    return max(0, head.sequence - minutes)
