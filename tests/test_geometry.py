"""Geometry feature tests - the branch that closes gap G1.

Numbers here are checked against known distances rather than against whatever the code
currently returns, because a silently wrong displacement feature would look exactly like
a model that does not find geometry useful.
"""

from __future__ import annotations

import math

from mapguard.data.overpass import GeometryAction, GeometryState
from mapguard.features.geometry import (
    features_for,
    haversine_m,
    polygon_area_m2,
    polyline_length_m,
    summarise,
)


def node(oid, lat, lon, version=1, tags=None):
    return GeometryState("node", oid, version, [(lat, lon)], tags or {})


def way(oid, coords, version=1, tags=None):
    return GeometryState("way", oid, version, coords, tags or {})


def test_haversine_against_a_known_distance():
    """London to Paris is about 344 km."""
    d = haversine_m((51.5074, -0.1278), (48.8566, 2.3522))
    assert 340_000 < d < 348_000


def test_haversine_one_degree_of_latitude():
    d = haversine_m((0.0, 0.0), (1.0, 0.0))
    assert math.isclose(d, 111_195, rel_tol=0.001)


def test_polyline_length_sums_segments():
    coords = [(0.0, 0.0), (0.0, 0.01), (0.0, 0.02)]
    assert math.isclose(polyline_length_m(coords), 2 * haversine_m((0.0, 0.0), (0.0, 0.01)), rel_tol=1e-6)


def test_polygon_area_of_a_small_square():
    """A 0.001 degree square near the equator is roughly 111 m on a side."""
    d = 0.001
    ring = [(0.0, 0.0), (0.0, d), (d, d), (d, 0.0), (0.0, 0.0)]
    area = polygon_area_m2(ring)
    assert 12_000 < area < 12_600  # ~111.2 m squared


def test_open_way_has_no_area():
    assert polygon_area_m2([(0.0, 0.0), (0.0, 0.001), (0.001, 0.001)]) == 0.0


def test_node_dragged_across_a_country():
    """The textbook vandalism mode: one node, enormous displacement."""
    action = GeometryAction("modify", "node", 42, 1, node(42, 51.5, -0.1), node(42, 48.85, 2.35))
    f = features_for(action)
    assert f.centroid_shift_m > 300_000
    assert f.d_points == 0


def test_way_stretched_across_a_continent():
    before = way(9, [(51.50, -0.10), (51.51, -0.10)])
    after = way(9, [(51.50, -0.10), (40.00, -3.70)], version=2)
    f = features_for(GeometryAction("modify", "way", 9, 1, before, after))
    assert f.length_new_m > f.length_old_m * 100
    assert f.length_ratio > 100
    assert f.hausdorff_m > 1_000_000


def test_area_blown_up_by_a_hundred():
    d = 0.001
    small = [(0.0, 0.0), (0.0, d), (d, d), (d, 0.0), (0.0, 0.0)]
    big = [(0.0, 0.0), (0.0, 10 * d), (10 * d, 10 * d), (10 * d, 0.0), (0.0, 0.0)]
    f = features_for(GeometryAction("modify", "way", 1, 1, way(1, small), way(1, big, 2)))
    assert 95 < f.area_ratio < 105


def test_tag_deltas():
    old = way(1, [(0.0, 0.0)], tags={"highway": "residential", "name": "Old"})
    new = way(1, [(0.0, 0.0)], 2, tags={"highway": "motorway", "surface": "asphalt"})
    f = features_for(GeometryAction("modify", "way", 1, 1, old, new))
    assert (f.tags_added, f.tags_removed, f.tags_changed) == (1, 1, 1)


def test_create_has_no_old_side():
    f = features_for(GeometryAction("create", "node", 5, 1, None, node(5, 1.0, 1.0)))
    assert f.has_new and not f.has_old
    assert f.centroid_shift_m == 0.0
    assert f.n_points_old == 0


def test_summary_covers_every_action_not_a_sample():
    """Rule 1 of PROJECT_PLAN 5.4 - the summary never truncates, so scale always
    reaches the classifier exactly. This is what Ovid zeroes out (gap G4)."""
    actions = [
        GeometryAction("modify", "node", i, 1, node(i, 0.0, 0.0), node(i, 0.0, i * 0.001))
        for i in range(1, 501)
    ]
    s = summarise(actions)
    assert s["n_geom_actions"] == 500
    assert s["n_modified"] == 500
    assert s["max_centroid_shift_m"] > s["mean_centroid_shift_m"] > 0


def test_summary_of_nothing_is_not_an_error():
    assert summarise([])["n_geom_actions"] == 0
