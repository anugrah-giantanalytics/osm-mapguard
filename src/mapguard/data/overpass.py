"""Overpass `adiff` - before/after geometry for a historical changeset.

This module is the fix that the plan review turned up. PROJECT_PLAN v1 concluded that
way geometry for the 2014-2019 labels was unaffordable, and therefore that the geometry
branch B1 could only run on an unlabelled modern corpus - which would have left the
paper's headline contribution untestable against the published benchmark.

Overpass attic data goes back to 2012, and `adiff` returns both sides of an edit with
node coordinates already resolved onto the way:

    <action type="modify">
      <old><way id="..." version="3"><nd ref="..." lat="..." lon="..."/>...</way></old>
      <new><way id="..." version="4"><nd ref="..." lat="..." lon="..."/>...</way></new>
    </action>

Verified 2026-09-04 on way 202342236 and changeset 52850166 - see probes P2 and P3 in
results/gate0d_review_probes.json.

Two gotchas, both hit during verification and both encoded below:

  1. `[bbox:...]` in the setting block does NOT parse alongside `adiff`. The bbox has to
     go on each statement instead. The error is "bbox requires four arguments", which is
     misleading - the arguments were fine, the position was not.
  2. Overpass answers over an area and a time window, not over a changeset id. So the
     result contains every edit any user made inside that bbox in that window, and the
     caller must filter on the changeset attribute of the `<new>` element.

A changeset with no bbox (tag-only or empty) or a near-global bbox must not be sent
here - it would ask a shared public service for a slice of the planet. `should_use_adiff`
is the guard; the caller falls back to versioned multi-fetch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

from mapguard.config import OVERPASS
from mapguard.data.http import PoliteSession, Response
from mapguard.data.osm_api import ChangesetMeta

# A changeset touching more than this many square degrees is not a local edit; asking
# Overpass for it is antisocial and the answer would be mostly other people's work.
MAX_BBOX_DEGREES_SQ = 4.0
# Pad the window so an edit timestamped on the boundary is not missed.
WINDOW_PAD_SECONDS = 2


@dataclass
class GeometryState:
    """One side - old or new - of one edited object."""

    otype: str
    oid: int
    version: int
    coords: list[tuple[float, float]] = field(default_factory=list)  # (lat, lon)
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.coords)


@dataclass
class GeometryAction:
    """An adiff action, filtered down to the changeset we asked about."""

    action: str                 # create | modify | delete
    otype: str
    oid: int
    changeset: int | None
    old: GeometryState | None
    new: GeometryState | None


def _pad(ts: str, seconds: int) -> str:
    """Shift an ISO-8601 Zulu timestamp by N seconds. Returns the input on bad data."""
    from datetime import datetime, timedelta

    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) + timedelta(seconds=seconds)
    except (ValueError, AttributeError):
        return ts
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def bbox_area_sq_degrees(bbox: tuple[float, float, float, float]) -> float:
    south, west, north, east = bbox
    return abs(north - south) * abs(east - west)


def should_use_adiff(meta: ChangesetMeta) -> tuple[bool, str]:
    """Is this changeset a reasonable thing to ask a shared Overpass instance for?"""
    if not meta.has_bbox:
        return False, "no bbox"
    if not meta.created_at or not meta.closed_at:
        return False, "no time window"
    area = bbox_area_sq_degrees(meta.bbox)  # type: ignore[arg-type]
    if area > MAX_BBOX_DEGREES_SQ:
        return False, f"bbox too large ({area:.1f} sq deg)"
    return True, "ok"


def build_query(meta: ChangesetMeta) -> str:
    """The adiff query for one changeset. See gotcha 1 above for the bbox placement."""
    south, west, north, east = meta.bbox  # type: ignore[misc]
    b = f"{south},{west},{north},{east}"
    start = _pad(meta.created_at or "", -WINDOW_PAD_SECONDS)
    end = _pad(meta.closed_at or meta.created_at or "", WINDOW_PAD_SECONDS)
    return (
        f'[adiff:"{start}","{end}"];'
        f"(node({b});way({b}););"
        "out geom meta;"
    )


def _state(el: ET.Element | None) -> GeometryState | None:
    """Read one <old>/<new> wrapper, or a bare element for a create action."""
    if el is None:
        return None
    obj = None
    for t in ("node", "way", "relation"):
        obj = el.find(t)
        if obj is not None:
            break
    if obj is None:
        return None

    coords: list[tuple[float, float]] = []
    if obj.tag == "node":
        if "lat" in obj.attrib and "lon" in obj.attrib:
            coords.append((float(obj.attrib["lat"]), float(obj.attrib["lon"])))
    else:
        for nd in obj.findall("nd"):
            if "lat" in nd.attrib and "lon" in nd.attrib:
                coords.append((float(nd.attrib["lat"]), float(nd.attrib["lon"])))

    return GeometryState(
        otype=obj.tag,
        oid=int(obj.attrib["id"]),
        version=int(obj.attrib.get("version", 0)),
        coords=coords,
        tags={t.attrib["k"]: t.attrib["v"] for t in obj.findall("tag")},
    )


def _changeset_of(el: ET.Element | None) -> int | None:
    if el is None:
        return None
    for t in ("node", "way", "relation"):
        obj = el.find(t)
        if obj is not None and "changeset" in obj.attrib:
            return int(obj.attrib["changeset"])
    return None


def parse_adiff(xml: str, cid: int | None = None) -> list[GeometryAction]:
    """Parse an adiff response, keeping only actions belonging to `cid` when given.

    A `create` action has no `<old>` wrapper and its object sits directly under
    `<action>`; `modify` and `delete` carry `<old>` and `<new>`. Both shapes appear in
    real responses, so both are handled rather than assumed away.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []

    actions: list[GeometryAction] = []
    for act in root.findall("action"):
        kind = act.attrib.get("type", "")
        old_el, new_el = act.find("old"), act.find("new")

        if old_el is None and new_el is None:
            # create: the object is a direct child of <action>
            new = _state(act)
            old = None
            changeset = _changeset_of(act)
        else:
            old, new = _state(old_el), _state(new_el)
            changeset = _changeset_of(new_el) or _changeset_of(old_el)

        if cid is not None and changeset != cid:
            continue
        ref = new or old
        if ref is None:
            continue

        actions.append(
            GeometryAction(
                action=kind,
                otype=ref.otype,
                oid=ref.oid,
                changeset=changeset,
                old=old,
                new=new,
            )
        )
    return actions


def fetch_geometry(
    session: PoliteSession, meta: ChangesetMeta
) -> tuple[list[GeometryAction], Response | None, str]:
    """Before/after geometry for one changeset.

    Returns (actions, response, note). An empty list with note "skipped: ..." means the
    guard rejected the changeset and the caller should use versioned multi-fetch.
    """
    usable, why = should_use_adiff(meta)
    if not usable:
        return [], None, f"skipped: {why}"

    r = session.get(OVERPASS, params={"data": build_query(meta)})
    if not r.ok:
        return [], r, f"http {r.status}"
    if "<remark>" in r.text or "Error</strong>" in r.text:
        return [], r, "overpass error"
    return parse_adiff(r.text, meta.cid), r, "ok"
