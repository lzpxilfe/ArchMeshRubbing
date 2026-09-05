"""능선: the ridges between flake scars, read from the mesh.

The generated biface carries its scars as planes meeting at edges, and it
knows where those edges are (``dorsal_creases``), so the detector can be
held to them: every crease it reports on the dorsal face must lie on a
true ridge, and it must find most of the ridge length.  The edge reading
is held to the generator's own mesh, whose ridges are edges; the readings
at a scale, meant for scans, are held to the same face meshed as a scan
meshes it (``dorsal_sheet``).  What a real scan gave is in
docs/LITHIC_TRIAL.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.artifact_crease_lines import (
    CREST_RULE_TURNING_V1,
    DEFAULT_CREASE_DIHEDRAL_MIN_DEG,
    ArtifactCreaseError,
    CreaseChain,
    crease_summary,
    creases_seen_from,
    detect_convex_creases,
    link_chains,
)
from synthetic_lithic import BIFACE_SHAPE, dorsal_creases, dorsal_sheet, flaked_tool, plan_radius


def _sampled(polyline: np.ndarray, step_mm: float = 0.5) -> np.ndarray:
    points = []
    for start, end in zip(polyline[:-1], polyline[1:]):
        count = max(1, int(np.linalg.norm(end - start) / step_mm))
        points.extend(start + (end - start) * (k / count) for k in range(count))
    points.append(polyline[-1])
    return np.asarray(points)


def _distance_to_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


@pytest.fixture(scope="module")
def biface():
    vertices, faces = flaked_tool()
    return vertices, faces, detect_convex_creases(vertices, faces)


def test_every_dorsal_crease_found_is_a_ridge_and_most_ridge_length_is_found(biface) -> None:
    vertices, _faces, chains = biface
    truth = dorsal_creases()
    truth_length = sum(float(np.linalg.norm(q - p)) for p, q in truth)
    assert 150.0 < truth_length < 250.0

    up = np.array([0.0, 0.0, 1.0])
    dorsal = [
        chain
        for chain in chains
        if (chain.left_normals @ up > 0.0).all() and (chain.right_normals @ up > 0.0).all()
    ]
    assert dorsal
    for chain in dorsal:
        samples = _sampled(chain.points_mm)
        nearest = np.min([_distance_to_segment(samples, p, q) for p, q in truth], axis=0)
        # On a true ridge, end to end: no invented line.
        assert float((nearest < 0.3).mean()) >= 0.9, chain.points_mm[[0, -1]]
        assert chain.max_dihedral_deg >= DEFAULT_CREASE_DIHEDRAL_MIN_DEG

    truth_points = np.vstack([_sampled(np.vstack([p, q])) for p, q in truth])

    def recall(found_chains) -> float:
        found = np.vstack([_sampled(chain.points_mm) for chain in found_chains])
        gaps = np.min(
            np.linalg.norm(truth_points[:, None, :] - found[None, :, :], axis=2), axis=1
        )
        return float((gaps < 0.3).mean())

    # The default threshold keeps only the ridges that bend hard - about a
    # third of the ridge length on this tool, and none invented.  Lowering
    # it to 15 degrees finds seven tenths, still without inventing any; the
    # gentlest ridges, and the taper's flattening of every ridge toward the
    # margin, are what is left.  Measured, not chosen: docs/LITHIC_TRIAL.md.
    assert 0.25 <= recall(dorsal) < 0.5
    gentler = [
        chain
        for chain in detect_convex_creases(vertices, _faces, dihedral_min_deg=15.0)
        if (chain.left_normals @ up > 0.0).all() and (chain.right_normals @ up > 0.0).all()
    ]
    for chain in gentler:
        nearest = np.min(
            [_distance_to_segment(_sampled(chain.points_mm), p, q) for p, q in truth], axis=0
        )
        assert float((nearest < 0.3).mean()) >= 0.9
    assert recall(gentler) >= 0.6

    # The central ridge, where the two steepest scars meet along y = 0.  A
    # chain may run on past its end into a ridge that also qualifies, so
    # look at the points on the axis rather than for a chain that stops.
    on_axis = np.vstack([chain.points_mm for chain in dorsal])
    on_axis = on_axis[np.abs(on_axis[:, 1]) < 0.3]
    assert on_axis.shape[0] >= 3
    assert on_axis[:, 0].max() - on_axis[:, 0].min() > 0.6 * BIFACE_SHAPE.half_length_mm


def _sheet_scores(chains, *, tolerance_mm: float = 0.6) -> tuple[float, float, float]:
    """Precision and recall of a reading on the dorsal sheet, by sample
    point, ignoring the 5 mm bevel at the margin where the sheet's taper
    bends the surface without any scar; and the longest chain."""

    import math

    truth = dorsal_creases()
    truth_points = np.vstack([_sampled(np.vstack([p, q])) for p, q in truth])

    def inner(points: np.ndarray) -> np.ndarray:
        edge = np.array([plan_radius(BIFACE_SHAPE, math.atan2(y, x)) for x, y in points[:, :2]])
        return points[np.hypot(points[:, 0], points[:, 1]) < edge - 5.0]

    truth_points = inner(truth_points)
    if not chains:
        return 0.0, 0.0, 0.0
    found = inner(np.vstack([_sampled(chain.points_mm) for chain in chains]))
    if found.shape[0] == 0:
        return 0.0, 0.0, 0.0
    nearest = np.min([_distance_to_segment(found, p, q) for p, q in truth], axis=0)
    gaps = np.min(np.linalg.norm(truth_points[:, None, :] - found[None, :, :], axis=2), axis=1)
    longest = max(chain.length_mm for chain in chains)
    return float((nearest < tolerance_mm).mean()), float((gaps < tolerance_mm).mean()), longest


def test_on_a_scan_like_mesh_the_curvature_rule_draws_the_ridges_and_the_first_rule_fragments() -> None:
    """The dorsal face meshed as a scanner meshes it: a 0.5 mm grid with no
    vertex on any ridge.

    The edge reading sees a third of the ridge length (the grid's own
    edges rarely lie on a ridge).  The first crest rule, picking edges by
    their turning against every neighbour, lets the edges along one ridge
    suppress each other and leaves fragments no longer than the mesh's
    triangles strung a few together; a fifth of the ridge length, and most
    of what it reports is off any ridge.  The curvature rule draws nine
    tenths of the ridge length as a handful of chains, nine tenths of it
    on a ridge; the rest is the gentlest ridges and where the taper lays
    every ridge flat toward the margin.  docs/LITHIC_TRIAL.md has the table.
    """

    vertices, faces = dorsal_sheet(pitch_mm=0.5)
    _precision, edge_recall, _longest = _sheet_scores(detect_convex_creases(vertices, faces))
    assert 0.15 <= edge_recall <= 0.5

    precision, recall, longest = _sheet_scores(
        detect_convex_creases(
            vertices, faces, dihedral_min_deg=15.0, scale_mm=4.0, crest_rule=CREST_RULE_TURNING_V1
        )
    )
    assert precision < 0.5 and recall < 0.35 and longest < 35.0

    chains = detect_convex_creases(vertices, faces, dihedral_min_deg=15.0, scale_mm=4.0)
    precision, recall, longest = _sheet_scores(chains)
    assert precision >= 0.85 and recall >= 0.75 and longest >= 40.0
    assert len(chains) <= 12
    # Linking the ends closes the small gaps into fewer, longer chains.
    linked = detect_convex_creases(vertices, faces, dihedral_min_deg=15.0, scale_mm=4.0, link_mm=4.0)
    _precision, linked_recall, linked_longest = _sheet_scores(linked)
    assert linked_recall >= recall and linked_longest > longest and len(linked) < len(chains)


def test_a_rounded_ridge_is_partly_found_at_the_scale_and_a_broader_one_defeats_it() -> None:
    """Ridges rounded over 1.5 mm: the curvature rule draws about half the
    ridge length, and about six tenths of what it draws lies on a ridge -
    the rest runs on past a ridge's end over the smooth surface.  Rounded
    over 3 mm the ridges bend too little at any scale and what is drawn
    lies nowhere near them.  Numbers, not claims; docs/LITHIC_TRIAL.md."""

    from dataclasses import replace

    vertices, faces = dorsal_sheet(replace(BIFACE_SHAPE, rounding_mm=1.5), pitch_mm=0.5)
    assert detect_convex_creases(vertices, faces) == ()
    precision, recall, longest = _sheet_scores(
        detect_convex_creases(vertices, faces, dihedral_min_deg=15.0, scale_mm=4.0)
    )
    assert 0.5 <= precision <= 0.8 and 0.4 <= recall <= 0.7 and longest >= 60.0

    vertices, faces = dorsal_sheet(replace(BIFACE_SHAPE, rounding_mm=3.0), pitch_mm=0.5)
    _precision, recall, _longest = _sheet_scores(
        detect_convex_creases(vertices, faces, dihedral_min_deg=15.0, scale_mm=4.0)
    )
    assert recall < 0.1


def test_scanner_noise_floods_the_edge_reading_and_the_scale_reading_holds() -> None:
    """With 0.15 mm of noise on every vertex of the sheet - three tenths of
    its 0.5 mm pitch - the edge reading is nine tenths invented, the first
    crest rule finds nothing at all, and the curvature reading at a 4 mm
    scale is what it was on the clean sheet.  On a scan the scale reading
    is the one to use."""

    from scan_defects import roughen

    vertices, faces = dorsal_sheet(pitch_mm=0.5)
    noisy = roughen(vertices, faces, amplitude_mm=0.15)
    edge_precision, _recall, _longest = _sheet_scores(detect_convex_creases(noisy, faces))
    assert edge_precision < 0.3
    assert (
        detect_convex_creases(
            noisy, faces, dihedral_min_deg=15.0, scale_mm=4.0, crest_rule=CREST_RULE_TURNING_V1
        )
        == ()
    )
    precision, recall, longest = _sheet_scores(
        detect_convex_creases(noisy, faces, dihedral_min_deg=15.0, scale_mm=4.0)
    )
    assert precision >= 0.85 and recall >= 0.75 and longest >= 30.0


def test_the_margin_is_a_crease_but_not_an_inner_line(biface) -> None:
    """The edge of the tool bends hardest of all, and the plan already
    draws it as the outline; seen from above only the dorsal ridges show,
    and from below nothing, the ventral face being one smooth surface."""

    _vertices, _faces, chains = biface
    margin = [chain for chain in chains if np.abs(chain.points_mm[:, 2]).max() < 1e-6]
    assert margin
    assert max(chain.max_dihedral_deg for chain in margin) > 60.0

    top = creases_seen_from(chains, "top")
    assert top
    for polyline in top:
        # Inside the plan, or ending on its edge where a ridge runs out.
        radius = np.hypot(polyline[:, 0], polyline[:, 1])
        angles = np.arctan2(polyline[:, 1], polyline[:, 0])
        edge = np.array([plan_radius(BIFACE_SHAPE, float(a)) for a in angles])
        assert (radius <= edge + 0.05).all()
        assert (radius < edge - 1.0).any()
    assert creases_seen_from(chains, "bottom") == []


def _chain(*points, normal=(0.0, 0.0, 1.0)) -> CreaseChain:
    p = np.asarray(points, dtype=np.float64)
    n = np.tile(np.asarray(normal, dtype=np.float64), (len(p) - 1, 1))
    return CreaseChain(points_mm=p, dihedral_deg=np.full(len(p) - 1, 30.0), left_normals=n, right_normals=n)


def test_linking_joins_ends_that_point_at_each_other_and_leaves_the_rest() -> None:
    """A ridge broken by a gap or a fork is one line again; a corner and a
    parallel neighbour are not, and the branch at a fork stays its own."""

    straight_a = _chain((0, 0, 0), (2, 0, 0), (4, 0, 0))
    straight_b = _chain((5, 0.2, 0), (7, 0.2, 0), (9, 0.2, 0))  # 1 mm gap, nearly collinear
    corner = _chain((9, 0.2, 0), (9, 3, 0), (9, 6, 0))  # shares an end, turns 90 degrees
    parallel = _chain((0, 1.5, 0), (2, 1.5, 0), (4, 1.5, 0))  # beside straight_a, never meets it
    fork = _chain((4, 0, 0), (5, 1.5, 0), (6, 3, 0))  # leaves straight_a's end at 37 degrees

    linked = link_chains([straight_a, straight_b, corner, parallel, fork], gap_mm=2.0)
    lengths = sorted(round(chain.length_mm, 3) for chain in linked)
    # straight_a + bridge + straight_b: 4 + 1.02 + 4; the corner, the
    # parallel line and the fork's branch each stay as they were.
    assert lengths == [3.606, 4.0, 5.8, pytest.approx(9.02, abs=0.001)]
    joined = max(linked, key=lambda chain: chain.length_mm)
    assert joined.points_mm.shape == (6, 3)
    assert joined.dihedral_deg.shape == (5,)
    assert joined.left_normals.shape == (5, 3)
    # The bridge borrows the dihedral of the edge it continues.
    assert float(joined.dihedral_deg[2]) == 30.0

    # A shared end at a fork: the straighter continuation wins the join.
    straighter = _chain((4, 0, 0), (6, 0.3, 0), (8, 0.6, 0))
    linked = link_chains([straight_a, fork, straighter], gap_mm=1.0)
    assert sorted(round(chain.length_mm, 1) for chain in linked) == [3.6, 8.0]

    # Nothing within reach, nothing changes; and a chain never joins itself.
    ring = _chain((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0))
    assert [chain.points_mm.shape for chain in link_chains([ring, parallel], gap_mm=0.5)] == [
        (5, 3),
        (3, 3),
    ]
    with pytest.raises(ArtifactCreaseError, match="gap_mm"):
        link_chains([ring], gap_mm=-1.0)
    with pytest.raises(ArtifactCreaseError, match="angle_deg"):
        link_chains([ring], gap_mm=1.0, angle_deg=90.0)


def test_a_reading_is_the_same_twice_and_refuses_bad_input(biface) -> None:
    vertices, faces, chains = biface
    again = detect_convex_creases(vertices, faces)
    assert crease_summary(again) == crease_summary(chains)
    assert all(
        np.array_equal(first.points_mm, second.points_mm) for first, second in zip(chains, again)
    )
    with pytest.raises(ArtifactCreaseError, match="strictly between"):
        detect_convex_creases(vertices, faces, dihedral_min_deg=0.0)
    with pytest.raises(ArtifactCreaseError, match="degenerate"):
        detect_convex_creases(vertices, np.array([[0, 0, 1]]))
    with pytest.raises(ArtifactCreaseError, match="link_mm"):
        detect_convex_creases(vertices, faces, link_mm=-1.0)
    with pytest.raises(ArtifactCreaseError, match="crest_rule"):
        detect_convex_creases(vertices, faces, crest_rule="anything")
    assert detect_convex_creases(vertices, np.zeros((0, 3), dtype=np.int64)) == ()
    # The edge reading does not touch the crest rule.
    assert crease_summary(
        detect_convex_creases(vertices, faces, crest_rule=CREST_RULE_TURNING_V1)
    ) == crease_summary(chains)
