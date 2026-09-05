"""The synthetic roof tiles are tiles, and they are solids.

The fixture is only worth having if it carries the things a tile drawing and
a tile rubbing are about: two walls with cut ends between them, the paddle's
cord on the convex back, the cloth on the concave face, a 암키와 that tapers
and a 수키와 that ends in a 미구.
"""

from __future__ import annotations

from collections import Counter
import math

import numpy as np
import pytest

from synthetic_tile import (
    AMKIWA,
    AMKIWA_SHAPE,
    SUGKIWA,
    SUGKIWA_SHAPE,
    TileShape,
    cloth_relief,
    cord_relief,
    hollow_tile,
)

STEP_MM = 4.0


def _directed_edges(faces: np.ndarray) -> Counter:
    edges: Counter = Counter()
    for a, b, c in faces:
        edges[(int(a), int(b))] += 1
        edges[(int(b), int(c))] += 1
        edges[(int(c), int(a))] += 1
    return edges


@pytest.mark.parametrize("shape", [AMKIWA_SHAPE, SUGKIWA_SHAPE], ids=[AMKIWA, SUGKIWA])
def test_a_tile_is_a_closed_solid_wound_the_same_way_throughout(shape: TileShape) -> None:
    """Watertight, and every face out: the volume and the topology need both."""

    vertices, faces = hollow_tile(shape, axial_step_mm=STEP_MM, angular_step_mm=STEP_MM)
    edges = _directed_edges(faces)
    # No directed edge twice: two faces never share an edge the same way round.
    assert max(edges.values()) == 1
    # And every edge has its opposite: nothing is left open.
    assert all((b, a) in edges for a, b in edges)

    # The outward normals point away from the tile's own body.
    corners = vertices[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    centre = vertices.mean(axis=0)
    outward = corners.mean(axis=1) - centre
    # A closed solid wound outward has a positive signed volume.
    volume = float(
        np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum() / 6.0
    )
    assert volume > 0.0
    # Most faces face away from the body; the ones that do not are the
    # concave surface, which is a real part of a tile.
    assert float(np.mean(np.einsum("ij,ij->i", normals, outward) > 0.0)) > 0.5


def test_the_two_tiles_are_the_two_tiles() -> None:
    """A 암키와 is wide, shallow and tapered; a 수키와 is half-round."""

    wide, _faces = hollow_tile(AMKIWA_SHAPE, axial_step_mm=STEP_MM, angular_step_mm=STEP_MM)
    round_, _sfaces = hollow_tile(SUGKIWA_SHAPE, axial_step_mm=STEP_MM, angular_step_mm=STEP_MM)

    wide_extent = wide.max(axis=0) - wide.min(axis=0)
    round_extent = round_.max(axis=0) - round_.min(axis=0)
    # Across: a 암키와 is nearly twice a 수키와.  Up: the reverse, because a
    # half-round tile is as tall as its own radius.
    assert wide_extent[0] > 1.7 * round_extent[0]
    assert round_extent[2] > round_extent[0] / 2.2
    assert wide_extent[2] < wide_extent[0] / 3.0
    # Both are about a third of a metre long, as roof tiles are.
    assert 300.0 < wide_extent[1] < 360.0
    assert 300.0 < round_extent[1] < 360.0

    # The 암키와 narrows toward one end: courses have to lap.
    def width_at(vertices: np.ndarray, y_mm: float) -> float:
        band = np.abs(vertices[:, 1] - y_mm) < 6.0
        return float(vertices[band, 0].max() - vertices[band, 0].min())

    near = width_at(wide, -150.0)
    far = width_at(wide, 150.0)
    assert far < near
    assert 0.05 < (near - far) / near < 0.15

    # The 수키와 keeps its width and drops its radius for the 미구 instead.
    assert width_at(round_, -140.0) > width_at(round_, 140.0)
    body = round_[np.abs(round_[:, 1] + 100.0) < 5.0, 2].max()
    tongue = round_[np.abs(round_[:, 1] - 150.0) < 5.0, 2].max()
    assert body - tongue == pytest.approx(SUGKIWA_SHAPE.tongue_drop_mm, abs=1.5)


def test_the_surfaces_carry_the_marks_a_tile_carries() -> None:
    """타날문 on the back, 포목흔 on the face, and not the other way round."""

    along = np.linspace(0.0, 240.0, 1200)
    across = np.linspace(0.0, 160.0, 40)
    cord = np.array([[cord_relief(float(x), float(y)) for x in along] for y in across])
    cloth = np.array([[cloth_relief(float(x), float(y)) for x in along] for y in across])

    # The cord stands proud; the cloth is pressed in.
    assert cord.min() >= 0.0
    assert 0.25 < cord.max() <= 0.35
    assert cloth.max() <= 0.0
    # And it is far shallower: a rubbing reads the cloth as tone, not lines.
    assert abs(cloth.min()) < cord.max() / 2.0
    # The paddle covers the wall: nowhere is left unstruck.
    assert float(np.mean(cord.max(axis=0) > 0.05)) > 0.95

    # The cord's ridges are a few millimetres apart.
    line = cord[len(across) // 2]
    crossings = [
        float(x)
        for x, before, after in zip(along[1:], line[:-1], line[1:], strict=True)
        if before < line.mean() <= after
    ]
    pitches = np.diff(crossings)
    assert 2.0 < float(np.median(pitches)) < 4.0

    # The mesh gets them on the right walls: the convex surface is rougher.
    shape = TileShape(
        kind=SUGKIWA, length_mm=120.0, inner_radius_mm=60.0, thickness_mm=18.0, span_deg=178.0
    )
    plain, _faces = hollow_tile(shape, axial_step_mm=1.0, angular_step_mm=1.0, relief=False)
    marked, _mfaces = hollow_tile(shape, axial_step_mm=1.0, angular_step_mm=1.0, relief=True)
    assert plain.shape == marked.shape
    radius_plain = np.hypot(plain[:, 0], plain[:, 2])
    radius_marked = np.hypot(marked[:, 0], marked[:, 2])
    moved = np.abs(radius_marked - radius_plain)
    outer = radius_plain > (shape.inner_radius_mm + shape.thickness_mm / 2.0)
    assert moved[outer].max() > 2.0 * moved[~outer].max()
    # The cloth and the mould's facets together stay under a quarter of a
    # millimetre: a tile's inner face is smooth to the hand.
    assert moved[~outer].max() < 0.25
    assert moved[outer].max() < 0.6  # a cord, not a cordon


def test_a_corner_cut_and_a_split_side_are_what_a_real_tiles_edges_carry() -> None:
    """귀접이 and 분할흔, and what each does to the drawing.

    A real 암키와 was cut from the cylinder with a 와도 drawn part way through
    and snapped the rest, so its sides are half knife and half fracture; a
    Goguryeo tile also has the corner of its wide end trimmed on the slant.
    The generator's tiles had neither: their sides were planes and their
    corners square.  With the corner cut, the plan loses the chamfer's
    triangle, foreshortened by the arc's tilt at the side; with the split,
    a section across the tile ends in a kinked line rather than a straight
    one.  Both tiles stay closed solids.
    """

    from scan_defects import mesh_report
    from src.core.artifact_outline_extractor import extract_outline_geometry
    from src.core.artifact_vector_extractor import extract_cutline_geometry
    from src.core.artifact_vector_record import PlanarFrame

    base = dict(kind=AMKIWA, length_mm=120.0, inner_radius_mm=210.0, thickness_mm=20.0, span_deg=40.0)
    shapes = {
        "plain": TileShape(**base),
        "cornered": TileShape(**base, corner_cut_mm=30.0),
        "split": TileShape(**base, split_share=0.5),
    }
    meshes = {
        name: hollow_tile(shape, axial_step_mm=4.0, angular_step_mm=4.0, relief=False)
        for name, shape in shapes.items()
    }
    for vertices, faces in meshes.values():
        report = mesh_report(vertices, faces.astype(np.int64))
        assert report["boundary_edge_count"] == 0
        assert report["nonmanifold_edge_count"] == 0
        assert report["connected_piece_count"] == 1

    plan = {
        name: extract_outline_geometry(
            vertices, faces.astype(np.int64), "top", precision_grid_mm=0.2
        ).qc["outline_area_mm2"]
        for name, (vertices, faces) in meshes.items()
    }
    chamfer = 0.5 * 30.0 * 30.0 * math.cos(math.radians(20.0))
    assert plan["plain"] - plan["cornered"] == pytest.approx(chamfer, rel=0.05)
    assert plan["split"] == pytest.approx(plan["plain"], rel=0.01)

    across = PlanarFrame(
        origin_world_mm=(0.0, 0.37, 0.0),
        u_axis_world=(1.0, 0.0, 0.0),
        v_axis_world=(0.0, 0.0, 1.0),
        normal_world=(0.0, -1.0, 0.0),
    )

    def points_on_the_sides(vertices: np.ndarray, faces: np.ndarray) -> int:
        section = extract_cutline_geometry(vertices, faces.astype(np.int64), across)
        assert len(section.payload.paths) == 1
        points = np.asarray(section.payload.paths[0].points_mm)
        reach = float(np.abs(points[:, 0]).max())
        return int((np.abs(points[:, 0]) > reach - 21.0).sum())

    plain_sides = points_on_the_sides(*meshes["plain"])
    assert points_on_the_sides(*meshes["cornered"]) == plain_sides
    assert points_on_the_sides(*meshes["split"]) > plain_sides


def test_the_same_arguments_give_the_same_tile() -> None:
    first = hollow_tile(AMKIWA_SHAPE, axial_step_mm=STEP_MM, angular_step_mm=STEP_MM)
    again = hollow_tile(AMKIWA_SHAPE, axial_step_mm=STEP_MM, angular_step_mm=STEP_MM)
    assert np.array_equal(first[0], again[0])
    assert np.array_equal(first[1], again[1])


def test_a_shape_refuses_what_a_tile_cannot_be() -> None:
    with pytest.raises(ValueError, match="tile kind"):
        TileShape(kind="pot", length_mm=1.0, inner_radius_mm=1.0, thickness_mm=1.0, span_deg=1.0)
    with pytest.raises(ValueError, match="must be positive"):
        TileShape(kind=AMKIWA, length_mm=0.0, inner_radius_mm=1.0, thickness_mm=1.0, span_deg=1.0)
    with pytest.raises(ValueError, match="미구"):
        TileShape(
            kind=AMKIWA,
            length_mm=100.0,
            inner_radius_mm=10.0,
            thickness_mm=1.0,
            span_deg=30.0,
            tongue_mm=10.0,
        )
    with pytest.raises(ValueError, match="taper"):
        TileShape(
            kind=AMKIWA,
            length_mm=100.0,
            inner_radius_mm=10.0,
            thickness_mm=1.0,
            span_deg=30.0,
            taper=0.9,
        )


def test_the_arc_is_the_arc_the_shape_asked_for() -> None:
    """The span is a real angle at the cylinder's own centre."""

    vertices, _faces = hollow_tile(
        SUGKIWA_SHAPE, axial_step_mm=STEP_MM, angular_step_mm=STEP_MM, relief=False
    )
    # A 수키와 arcs above its axis, which sits at z = 0.
    outer = np.hypot(vertices[:, 0], vertices[:, 2]) > (
        SUGKIWA_SHAPE.inner_radius_mm + SUGKIWA_SHAPE.thickness_mm / 2.0
    )
    body = outer & (np.abs(vertices[:, 1] + 100.0) < 5.0)
    angles = np.degrees(np.arctan2(vertices[body, 0], vertices[body, 2]))
    assert float(angles.max() - angles.min()) == pytest.approx(
        SUGKIWA_SHAPE.span_deg, abs=1.0
    )
    radius = np.hypot(vertices[body, 0], vertices[body, 2])
    assert float(radius.mean()) == pytest.approx(SUGKIWA_SHAPE.outer_radius_mm, abs=0.5)
    assert math.isclose(
        SUGKIWA_SHAPE.outer_radius_mm,
        SUGKIWA_SHAPE.inner_radius_mm + SUGKIWA_SHAPE.thickness_mm,
    )
