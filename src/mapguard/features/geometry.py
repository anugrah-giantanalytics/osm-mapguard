"""Geometry features from an adiff before/after pair - the signal every prior model drops.

Gap G1 in PROJECT_PLAN section 3: Ovid sees a bounding box and nothing else. No shape,
no displacement, no area change. A node dragged 400 km and a node nudged 40 cm look
identical to it. These are the features that close that gap.

Everything is computed on WGS84 lat/lon with a local equirectangular approximation
rather than a projection. Over the extent of a single changeset the error is far below
the scale of the signals we care about, and it avoids a pyproj transform per object on a
laptop with 0.77 GB of free RAM. Distances come out in metres, areas in square metres.

The features are deliberately scale-aware rather than scale-free. "This way got 100x
longer" and "this way moved 400 km" are the vandalism modes; normalising them away would
throw out the thing we are trying to detect.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from mapguard.data.overpass import GeometryAction, GeometryState

EARTH_R = 6371008.8  # mean radius, metres


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(min(1.0, h)))


def _local_xy(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Metres east/north of the first point. Good enough inside one changeset."""
    if not coords:
        return []
    lat0, lon0 = coords[0]
    k = math.cos(math.radians(lat0))
    return [
        ((lon - lon0) * math.radians(1.0) * EARTH_R * k, (lat - lat0) * math.radians(1.0) * EARTH_R)
        for lat, lon in coords
    ]


def polyline_length_m(coords: list[tuple[float, float]]) -> float:
    return sum(haversine_m(coords[i], coords[i + 1]) for i in range(len(coords) - 1))


def polygon_area_m2(coords: list[tuple[float, float]]) -> float:
    """Shoelace area in the local plane. Zero for anything that is not a closed ring."""
    if len(coords) < 4 or coords[0] != coords[-1]:
        return 0.0
    xy = _local_xy(coords)
    s = sum(xy[i][0] * xy[i + 1][1] - xy[i + 1][0] * xy[i][1] for i in range(len(xy) - 1))
    return abs(s) / 2.0


def centroid(coords: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not coords:
        return None
    return (sum(c[0] for c in coords) / len(coords), sum(c[1] for c in coords) / len(coords))


def bbox_diagonal_m(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 2:
        return 0.0
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    return haversine_m((min(lats), min(lons)), (max(lats), max(lons)))


def hausdorff_m(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Symmetric Hausdorff between two point sets.

    Quadratic, so it is capped. A way with thousands of nodes is rare and the cap costs
    us a slightly loose bound on exactly the cases where any sane threshold has already
    fired.
    """
    if not a or not b:
        return 0.0
    if len(a) * len(b) > 250_000:
        a, b = a[:500], b[:500]
    fwd = max(min(haversine_m(p, q) for q in b) for p in a)
    bwd = max(min(haversine_m(q, p) for p in a) for q in b)
    return max(fwd, bwd)


@dataclass
class GeometryFeatures:
    """Per-edit geometry, both sides where they exist."""

    otype: str
    oid: int
    action: str
    has_old: bool
    has_new: bool

    n_points_old: int
    n_points_new: int
    d_points: int

    length_old_m: float
    length_new_m: float
    d_length_m: float
    length_ratio: float          # new/old, capped; 0 when old is 0

    area_old_m2: float
    area_new_m2: float
    d_area_m2: float
    area_ratio: float

    centroid_shift_m: float      # how far the object moved
    hausdorff_m: float           # how much its shape changed
    bbox_diag_old_m: float
    bbox_diag_new_m: float

    n_tags_old: int
    n_tags_new: int
    tags_added: int
    tags_removed: int
    tags_changed: int

    def as_dict(self) -> dict:
        return asdict(self)


def _ratio(new: float, old: float, cap: float = 1e6) -> float:
    if old <= 0:
        return 0.0
    return min(cap, new / old)


def _tag_deltas(old: GeometryState | None, new: GeometryState | None) -> tuple[int, int, int]:
    o = old.tags if old else {}
    n = new.tags if new else {}
    added = len(set(n) - set(o))
    removed = len(set(o) - set(n))
    changed = sum(1 for k in set(o) & set(n) if o[k] != n[k])
    return added, removed, changed


def features_for(action: GeometryAction) -> GeometryFeatures:
    old, new = action.old, action.new
    co = old.coords if old else []
    cn = new.coords if new else []

    len_o, len_n = polyline_length_m(co), polyline_length_m(cn)
    area_o, area_n = polygon_area_m2(co), polygon_area_m2(cn)
    cent_o, cent_n = centroid(co), centroid(cn)
    added, removed, changed = _tag_deltas(old, new)

    return GeometryFeatures(
        otype=action.otype,
        oid=action.oid,
        action=action.action,
        has_old=old is not None,
        has_new=new is not None,
        n_points_old=len(co),
        n_points_new=len(cn),
        d_points=len(cn) - len(co),
        length_old_m=round(len_o, 3),
        length_new_m=round(len_n, 3),
        d_length_m=round(len_n - len_o, 3),
        length_ratio=round(_ratio(len_n, len_o), 6),
        area_old_m2=round(area_o, 3),
        area_new_m2=round(area_n, 3),
        d_area_m2=round(area_n - area_o, 3),
        area_ratio=round(_ratio(area_n, area_o), 6),
        centroid_shift_m=round(haversine_m(cent_o, cent_n), 3) if (cent_o and cent_n) else 0.0,
        hausdorff_m=round(hausdorff_m(co, cn), 3),
        bbox_diag_old_m=round(bbox_diagonal_m(co), 3),
        bbox_diag_new_m=round(bbox_diagonal_m(cn), 3),
        n_tags_old=len(old.tags) if old else 0,
        n_tags_new=len(new.tags) if new else 0,
        tags_added=added,
        tags_removed=removed,
        tags_changed=changed,
    )


def summarise(actions: list[GeometryAction]) -> dict:
    """Changeset-level geometry summary, computed over ALL actions.

    This is Rule 1 from PROJECT_PLAN section 5.4: the per-edit encoder may sample, but
    the summary never truncates, so scale always reaches the classifier exactly. Ovid
    zeroes the edit vector above a size threshold and loses mass vandalism entirely
    (gap G4); this is the piece that makes not doing that possible.
    """
    if not actions:
        return {"n_geom_actions": 0}

    feats = [features_for(a) for a in actions]
    shifts = [f.centroid_shift_m for f in feats]
    hd = [f.hausdorff_m for f in feats]
    d_area = [f.d_area_m2 for f in feats]
    ratios = [f.area_ratio for f in feats if f.area_ratio > 0]

    def p95(xs: list[float]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        return s[min(len(s) - 1, int(0.95 * len(s)))]

    return {
        "n_geom_actions": len(feats),
        "n_way": sum(1 for f in feats if f.otype == "way"),
        "n_node": sum(1 for f in feats if f.otype == "node"),
        "n_with_old": sum(1 for f in feats if f.has_old),
        "max_centroid_shift_m": round(max(shifts), 3),
        "mean_centroid_shift_m": round(sum(shifts) / len(shifts), 3),
        "p95_centroid_shift_m": round(p95(shifts), 3),
        "max_hausdorff_m": round(max(hd), 3),
        "total_d_area_m2": round(sum(d_area), 3),
        "max_abs_d_area_m2": round(max((abs(x) for x in d_area), default=0.0), 3),
        "max_area_ratio": round(max(ratios), 6) if ratios else 0.0,
        "total_d_points": sum(f.d_points for f in feats),
        "n_deleted": sum(1 for f in feats if f.action == "delete"),
        "n_created": sum(1 for f in feats if f.action == "create"),
        "n_modified": sum(1 for f in feats if f.action == "modify"),
    }
