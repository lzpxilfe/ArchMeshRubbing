"""능선: the ridges between flake scars, read from the mesh.

The generated biface carries its scars as planes meeting at edges, and it
knows where those edges are (``dorsal_creases``), so the detector can be
held to them: every crease it reports on the dorsal face must lie on a
true ridge, and it must find most of the ridge length.  What it does on a
real scan, where a ridge is rounded and the mesh is noisy, is not known
yet - docs/LITHIC_TRIAL.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.artifact_crease_lines import (
    DEFAULT_CREASE_DIHEDRAL_MIN_DEG,
    ArtifactCreaseError,
    crease_summary,
    creases_seen_from,
    detect_convex_creases,
)
from synthetic_lithic import BIFACE_SHAPE, dorsal_creases, flaked_tool, plan_radius


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


def test_a_rounded_ridge_needs_the_scale_and_is_found_at_it_with_some_invention() -> None:
    """The ridge a scan has: rounded over a millimetre or two.

    No single edge of it bends much, so the edge-by-edge reading finds next
    to nothing.  Read at a scale of 4 mm - the bend between the surface
    4 mm to either side, the crest picked as the edge that turns most per
    millimetre among its neighbours - it finds most of the ridge length on
    a tool rounded over 1.5 mm, and also invents lines the sharp reading
    never did: a rounded surface bends at that scale in places that are
    not ridges.  A tool rounded over 3 mm defeats it.  These are the
    numbers, not a claim; docs/LITHIC_TRIAL.md.
    """

    from dataclasses import replace

    truth = dorsal_creases()
    truth_points = np.vstack([_sampled(np.vstack([p, q])) for p, q in truth])
    up = np.array([0.0, 0.0, 1.0])

    def dorsal_of(chains):
        return [
            chain
            for chain in chains
            if (chain.left_normals @ up > 0.0).all() and (chain.right_normals @ up > 0.0).all()
        ]

    def measure(chains, tolerance_mm: float = 0.6) -> tuple[float, float, float]:
        on, off = 0.0, 0.0
        for chain in chains:
            nearest = np.min(
                [_distance_to_segment(_sampled(chain.points_mm), p, q) for p, q in truth],
                axis=0,
            )
            if float((nearest < tolerance_mm).mean()) >= 0.9:
                on += chain.length_mm
            else:
                off += chain.length_mm
        if not chains:
            return on, off, 0.0
        found = np.vstack([_sampled(chain.points_mm) for chain in chains])
        gaps = np.min(np.linalg.norm(truth_points[:, None, :] - found[None, :, :], axis=2), axis=1)
        return on, off, float((gaps < tolerance_mm).mean())

    vertices, faces = flaked_tool(replace(BIFACE_SHAPE, rounding_mm=1.5))
    _on, _off, sharp_recall = measure(dorsal_of(detect_convex_creases(vertices, faces)))
    assert sharp_recall < 0.1

    on, off, recall = measure(
        dorsal_of(detect_convex_creases(vertices, faces, dihedral_min_deg=15.0, scale_mm=4.0))
    )
    assert recall >= 0.6
    assert on > off
    assert off > 0.0


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
    assert detect_convex_creases(vertices, np.zeros((0, 3), dtype=np.int64)) == ()
