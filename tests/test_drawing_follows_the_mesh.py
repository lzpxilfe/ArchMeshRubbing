"""The drawing is the mesh's, not a picture that happens to look like a pot.

Two things are asked here, because they are the two ways a drawing could be
independent of the geometry it claims to measure:

1. Where the outline is drawn is where the mesh's own silhouette is, to
   within the precision grid the recipe asked for.
2. Change the mesh and the drawing changes with it, in the place where the
   mesh changed and nowhere else.

Both are measured against `session.materialize()`, the source mesh with the
active Align applied - the same mesh the extractor is handed.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pytest

from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_vector_record import VectorGeometryPayload
from synthetic_vessel import positioned_vessel_session

SEGMENTS = 48
RINGS = 24
GRID_MM = 0.1
#: The boundary is snapped to the lattice, so half a cell is the floor.  Two
#: cells leaves room for that and for the union's own rounding without
#: leaving room for a drawing that has stopped following the mesh.
TOLERANCE_MM = 2.0 * GRID_MM


def _outline(relief: Callable[[float, float], float] | None = None) -> tuple[VectorGeometryPayload, np.ndarray]:
    """One front outline, and the canonical mesh it was measured from."""

    session, _vertices, _faces = positioned_vessel_session(
        segments=SEGMENTS, rings=RINGS, relief=relief
    )
    computation = compute_artifact_outline(session, "front", precision_grid_mm=GRID_MM)
    return computation.payload, session.materialize().mesh.vertices


def _crossings(payload: VectorGeometryPayload, height_mm: float) -> list[float]:
    """Where the drawn boundary crosses this height, in its own millimetres.

    A ring is a polygon, so its width at a height is where its edges cut that
    height - not the spread of whichever vertices happen to lie near it.
    """

    found: list[float] = []
    for path in payload.paths:
        points = np.asarray(path.points_mm, dtype=np.float64)
        closed = np.vstack([points, points[:1]]) if path.closed else points
        first, second = closed[:-1], closed[1:]
        rising = first[:, 1] != second[:, 1]
        low = np.minimum(first[:, 1], second[:, 1])
        high = np.maximum(first[:, 1], second[:, 1])
        cut = rising & (low <= height_mm) & (high >= height_mm)
        if not cut.any():
            continue
        a, b = first[cut], second[cut]
        t = (height_mm - a[:, 1]) / (b[:, 1] - a[:, 1])
        found.extend(a[:, 0] + t * (b[:, 0] - a[:, 0]))
    return found


def _outer_wall_profile(vertices: np.ndarray) -> list[tuple[float, float, float]]:
    """(height, widest u, narrowest u) for each ring of the outer wall.

    `hollow_vessel` lays the outer wall's rings down first, and every ring
    sits at one exact height, so at that height the widest vertex on the ring
    is the mesh's silhouette - no band of heights, and so no chance of
    catching the inner wall's rings instead.
    """

    outer = (RINGS + 1) * SEGMENTS
    u, v = vertices[:outer, 0], vertices[:outer, 2]
    profile: list[tuple[float, float, float]] = []
    for height in np.unique(np.round(v, 9)):
        on_ring = np.abs(v - height) < 1e-9
        if int(on_ring.sum()) < SEGMENTS:
            continue
        profile.append((float(height), float(u[on_ring].max()), float(u[on_ring].min())))
    return profile


def test_the_drawn_outline_is_the_meshs_own_silhouette() -> None:
    payload, vertices = _outline()
    profile = _outer_wall_profile(vertices)
    assert len(profile) == RINGS + 1

    gaps: list[float] = []
    for height, widest, narrowest in profile:
        crossings = _crossings(payload, height)
        assert len(crossings) >= 2, f"the drawing has no width at {height} mm"
        gaps.append(abs(max(crossings) - widest))
        gaps.append(abs(min(crossings) - narrowest))
    worst = max(gaps)
    assert worst <= TOLERANCE_MM, f"the drawn outline is {worst:.4f} mm off the mesh"
    # Not just bounded: actually snapped to the lattice, so most of the
    # boundary sits inside half a cell of the mesh.
    assert float(np.median(gaps)) <= GRID_MM


def test_a_change_in_the_mesh_moves_the_drawing_with_it() -> None:
    """A drawing that ignored the mesh would be the same drawing either way."""

    swell_mm = 2.0
    low, high = 30.0, 40.0

    def swollen(_angle_rad: float, z_mm: float) -> float:
        # A band of the wall pushed out, with cosine shoulders so the mesh
        # keeps a bounded slope and the change is the band's, not a cliff's.
        if not (low <= z_mm <= high):
            return 0.0
        edge = min(z_mm - low, high - z_mm, 1.0)
        return swell_mm * 0.5 * (1.0 - math.cos(math.pi * edge))

    plain, plain_mesh = _outline()
    swelled, swelled_mesh = _outline(swollen)
    assert plain.sha256 != swelled.sha256
    # The mesh really did swell, and only in the band.
    widened_mesh = [
        (before[0], after[1] - before[1])
        for before, after in zip(
            _outer_wall_profile(plain_mesh), _outer_wall_profile(swelled_mesh), strict=True
        )
    ]
    # A hair under the swell, because a ring is a polygon: its widest vertex
    # sits half a segment off the axis, so it carries cos(pi/segments) of a
    # radial change.  The drawing measures that same polygon.
    expected = swell_mm * math.cos(math.pi / SEGMENTS)
    assert max(value for _height, value in widened_mesh) == pytest.approx(expected, abs=1e-9)
    assert min(value for _height, value in widened_mesh) == pytest.approx(0.0, abs=1e-9)

    # The two meshes differ only in the band, and so must the two drawings.
    inside: list[float] = []
    outside: list[float] = []
    for height, _widest, _narrowest in _outer_wall_profile(plain_mesh):
        before = _crossings(plain, height)
        after = _crossings(swelled, height)
        if not before or not after:
            continue
        widened = max(after) - max(before)
        # The Align puts the floor's plane at the origin, so the band's own
        # heights are 10 mm lower in the record's frame.
        source_height = height + 10.0
        if low + 1.5 <= source_height <= high - 1.5:
            inside.append(widened)
        elif source_height < low - 1.0 or source_height > high + 1.0:
            outside.append(widened)

    assert inside and outside
    # Inside the band the drawing moved out by what the mesh moved out by.
    assert min(inside) == pytest.approx(swell_mm, abs=TOLERANCE_MM)
    assert max(inside) == pytest.approx(swell_mm, abs=TOLERANCE_MM)
    # Outside it the drawing did not move at all.
    assert max(abs(value) for value in outside) <= TOLERANCE_MM


def test_the_mesh_a_record_is_measured_from_is_the_aligned_one() -> None:
    """The gap measurements above are only meaningful in the right frame.

    An earlier attempt at them compared the record's coordinates against the
    source vertices and read the Align's own 10 mm as an error in the
    drawing.  This states which mesh the extractor is handed, so that the
    mistake cannot be made silently again.
    """

    session, source, _faces = positioned_vessel_session(segments=SEGMENTS, rings=RINGS)
    canonical: Any = session.materialize().mesh.vertices
    assert canonical.shape == source.shape
    # This vessel is stood on its own rotation axis, which moves it down its
    # own axis and leaves the cross-section where it was.
    assert np.allclose(canonical[:, :2], source[:, :2])
    assert not np.allclose(canonical[:, 2], source[:, 2])
    assert np.allclose(canonical[:, 2] - source[:, 2], canonical[0, 2] - source[0, 2])
