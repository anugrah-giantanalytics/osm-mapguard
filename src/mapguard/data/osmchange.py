"""osmChange XML -> flat per-edit records.

`GET /api/0.6/changeset/<id>/download` returns an osmChange document: three
optional blocks (create / modify / delete), each holding the *new* state of every
object touched. Structure:

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

Two things the format does *not* give us, both noted in PROJECT_PLAN.md §8.2:

  - the *previous* state of a modified object (needs one extra API call per object)
  - coordinates for a way's nodes (only `<nd ref>` ids)

So Corpus A geometry is node-only, which is exactly why Corpus B exists. Nodes in
a `create` or `modify` block carry inline lat/lon and are usable immediately.

Parsing uses iterparse and clears elements as it goes: some vandalism changesets
hold tens of thousands of objects, and the full 18k corpus must stream through
Colab's 12.7 GB without materialising every DOM at once.
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
    lat: float | None     # nodes only
    lon: float | None
    tags: dict[str, str] = field(default_factory=dict)
    n_refs: int = 0       # <nd> for ways, <member> for relations


def parse(xml: str) -> Iterator[Edit]:
    """Yield one Edit per object, in document order.

    Document order is commit order, which is what the B3 edit-sequence branch
    consumes - so it must not be sorted or deduplicated here.
    """
    # iterparse wants a stream; the strings are small enough (KBs) to wrap in memory,
    # and this keeps the caller free to hold the XML in SQLite rather than on disk.
    from io import StringIO

    action = None
    for event, elem in ET.iterparse(StringIO(xml), events=("start", "end")):
        if event == "start":
            if elem.tag in ACTIONS:
                action = elem.tag
            continue

        if elem.tag in ACTIONS:
            action = None
            elem.clear()
            continue

        if elem.tag not in OBJ_TYPES or action is None:
            continue

        a = elem.attrib
        tags = {t.attrib["k"]: t.attrib["v"] for t in elem.findall("tag")}
        n_refs = len(elem.findall("nd")) + len(elem.findall("member"))

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
            tags=tags,
            n_refs=n_refs,
        )
        elem.clear()


def parse_safe(xml: str) -> tuple[list[Edit], str | None]:
    """parse(), but a malformed document returns an error string instead of raising.

    Truncated XML in the cache is a fetch failure, not a bug - record it and keep
    the corpus statistics honest rather than crashing an 18k-row loop.
    """
    try:
        return list(parse(xml)), None
    except ET.ParseError as exc:
        return [], f"ParseError: {exc}"
