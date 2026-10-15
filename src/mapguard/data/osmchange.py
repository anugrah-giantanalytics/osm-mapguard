"""osmChange XML -> flat per-edit records.

`GET /api/0.6/changeset/<id>/download` returns an osmChange document: three optional
blocks (create / modify / delete), each holding the *new* state of every object touched.
Structure:

    <osmChange version="0.6" generator="...">
      <create>
        <node id="1" version="1" timestamp="..." uid="7" user="x" lat="1.0" lon="2.0">
          <tag k="amenity" v="cafe"/>
        </node>
        <way id="9" version="3" ...><nd ref="1"/><tag k="highway" v="track"/></way>
      </create>
      <modify>...</modify>
      <delete>...</delete>
    </osmChange>

Two things the format does not give us:

  - the *previous* state of a modified object
  - coordinates for a way's nodes (only `<nd ref>` ids)

Both are supplied separately, and neither is expensive:

  - `mapguard.data.overpass` fetches before/after geometry with coordinates already
    resolved onto the way, via Overpass `adiff`, for any date from 2012 onward
  - `mapguard.data.osm_api.fetch_versioned` batches previous object states 725 at a time

PROJECT_PLAN v1 claimed the previous state cost one request per object and concluded
that Corpus A geometry was node-only. That was wrong, and it mattered - it would have
left the geometry branch testable only on an unlabelled corpus. See docs/planning/PLAN_REVIEW.md
finding F1.

Parsing streams. Some vandalism changesets hold tens of thousands of objects, and the
full 18k corpus has to pass through Colab's 12.7 GB without materialising every DOM.
`elem.clear()` alone is not enough for that: it empties an element's children but leaves
the stripped element attached to its parent, so a 50,000-object document still ends up
holding 50,000 husks. The parser below detaches each element from the root as it goes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator
from xml.etree import ElementTree as ET

ACTIONS = ("create", "modify", "delete")
OBJ_TYPES = ("node", "way", "relation")


@dataclass
class Edit:
    """One object touched by one changeset."""

    action: str           # create | modify | delete
    otype: str            # node | way | relation
    oid: int
    version: int
    timestamp: str
    uid: int | None
    user: str | None
    lat: float | None     # nodes only; ways carry <nd> refs without coordinates
    lon: float | None
    tags: dict[str, str] = field(default_factory=dict)
    n_refs: int = 0       # <nd> for ways, <member> for relations


def parse(xml: str) -> Iterator[Edit]:
    """Yield one Edit per object, in document order.

    Document order is commit order, which is what the B3 edit-sequence branch consumes,
    so it must not be sorted or deduplicated here.
    """
    from io import StringIO

    action_el: ET.Element | None = None
    action: str | None = None

    for event, elem in ET.iterparse(StringIO(xml), events=("start", "end")):
        if event == "start":
            if elem.tag in ACTIONS:
                action_el, action = elem, elem.tag
            continue

        if elem.tag in ACTIONS:
            # Block finished. Drop its remaining children so the document does not
            # accumulate; the objects themselves were already detached below.
            elem.clear()
            action_el, action = None, None
            continue

        if elem.tag not in OBJ_TYPES or action is None:
            continue

        a = elem.attrib
        yield Edit(
            action=action,
            otype=elem.tag,
            oid=int(a["id"]),
            version=int(a.get("version", 0)),
            timestamp=a.get("timestamp", ""),
            uid=int(a["uid"]) if "uid" in a else None,
            user=a.get("user"),
            lat=float(a["lat"]) if "lat" in a else None,
            lon=float(a["lon"]) if "lon" in a else None,
            tags={t.attrib["k"]: t.attrib["v"] for t in elem.findall("tag")},
            n_refs=len(elem.findall("nd")) + len(elem.findall("member")),
        )

        # Detach, not just clear. Clearing empties the children but leaves this element
        # in its parent's list, which is the difference between constant memory and
        # linear memory on a 50,000-object changeset.
        elem.clear()
        if action_el is not None:
            try:
                action_el.remove(elem)
            except ValueError:
                pass


def parse_safe(xml: str) -> tuple[list[Edit], str | None]:
    """parse(), but a malformed document returns an error string instead of raising.

    Truncated XML in the cache is a fetch failure, not a bug - record it and keep the
    corpus statistics honest rather than crashing an 18k-row loop.
    """
    try:
        return list(parse(xml)), None
    except ET.ParseError as exc:
        return [], f"ParseError: {exc}"
