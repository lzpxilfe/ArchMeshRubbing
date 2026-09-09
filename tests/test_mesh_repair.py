"""이음: sewing a piece onto the wall it stands on, and refusing to guess.

The fixture is 24ET0021 in miniature: a hollow drum for the pot and, standing
in its cavity, a piece with two skins whose edges stop a tenth of a
millimetre short of the inner wall - the ring of shadow a scanner's head
leaves where it cannot reach into a joint.  What the tests hold to is that
the piece is sewn to the *wall*, that no vertex is added or moved doing it,
and that a join naming edges this mesh does not have is refused rather than
applied to whatever sits at that index.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.artifact_mesh_bodies import (
    boundary_rings,
    diagnose_mesh_bodies,
    face_bodies,
)
from src.core.artifact_mesh_repair import (
    DEFAULT_REACH_UM,
    ArtifactMeshRepairError,
    RingPairJoin,
    apply_mesh_repair,
    body_sha256,
    ring_sha256,
    sew_ring_pair_to_body,
)


def _ring_points(radius: float, height: float, columns: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, columns, endpoint=False)
    return np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full(columns, height)]
    )


def _hollow_drum(
    *,
    outer: float = 10.0,
    inner: float = 9.5,
    bottom: float = 0.0,
    top: float = 20.0,
    rows: int = 40,
    columns: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """A wall with two skins and a rim at each end - a pot, in miniature.

    A single-skinned drum would not do: the piece stands in the cavity, and
    which skin it stands on is the whole question.  A solid drum has no
    cavity and no inner skin to stand on.
    """

    heights = np.linspace(bottom, top, rows + 1)
    outer_rings = [_ring_points(outer, z, columns) for z in heights]
    inner_rings = [_ring_points(inner, z, columns) for z in heights]
    vertices = np.concatenate(outer_rings + inner_rings)
    faces: list[list[int]] = []

    def quad(a: int, b: int, c: int, d: int) -> None:
        faces.append([a, b, c])
        faces.append([a, c, d])

    outer_base = 0
    inner_base = (rows + 1) * columns
    for row in range(rows):
        for i in range(columns):
            j = (i + 1) % columns
            # outer skin: normals away from the axis
            quad(
                outer_base + row * columns + i,
                outer_base + row * columns + j,
                outer_base + (row + 1) * columns + j,
                outer_base + (row + 1) * columns + i,
            )
            # inner skin: wound the other way, so its normals face the cavity
            quad(
                inner_base + row * columns + j,
                inner_base + row * columns + i,
                inner_base + (row + 1) * columns + i,
                inner_base + (row + 1) * columns + j,
            )
    for i in range(columns):
        j = (i + 1) % columns
        quad(  # bottom rim
            outer_base + j,
            outer_base + i,
            inner_base + i,
            inner_base + j,
        )
        quad(  # top rim
            outer_base + rows * columns + i,
            outer_base + rows * columns + j,
            inner_base + rows * columns + j,
            inner_base + rows * columns + i,
        )
    return vertices, np.asarray(faces, dtype=np.int64)


def _skin(
    *, radius: float, height: float, apex: float, columns: int = 24, upward: bool
) -> tuple[np.ndarray, np.ndarray]:
    """One skin of the piece: a fan from an apex out to one open ring."""

    rim = _ring_points(radius, height, columns)
    vertices = np.concatenate([rim, np.array([[0.0, 0.0, apex]])])
    centre = columns
    faces = [
        [centre, i, (i + 1) % columns] if upward else [centre, (i + 1) % columns, i]
        for i in range(columns)
    ]
    return vertices, np.asarray(faces, dtype=np.int64)


def _merge(*parts: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for part_vertices, part_faces in parts:
        vertices.append(part_vertices)
        faces.append(part_faces + offset)
        offset += part_vertices.shape[0]
    return np.concatenate(vertices), np.concatenate(faces)


def _artifact(
    *, ring_radius: float = 9.4
) -> tuple[np.ndarray, np.ndarray]:
    """A drum with a two-skinned piece standing in it, edges short of the wall."""

    return _merge(
        _hollow_drum(),
        _skin(radius=ring_radius, height=10.6, apex=13.0, upward=True),
        _skin(radius=ring_radius, height=9.4, apex=12.0, upward=False),
    )


def _join(vertices: np.ndarray, faces: np.ndarray) -> tuple[RingPairJoin, list[list[int]]]:
    rings = boundary_rings(faces)
    labels = face_bodies(faces)
    wall = int(np.argmax(np.bincount(labels)))
    return (
        RingPairJoin(
            first_ring_sha256=ring_sha256(rings[0]),
            second_ring_sha256=ring_sha256(rings[1]),
            body_sha256=body_sha256(faces, np.flatnonzero(labels == wall)),
        ),
        rings,
    )


def test_the_fixture_is_the_defect_this_repairs() -> None:
    vertices, faces = _artifact()
    report = diagnose_mesh_bodies(vertices, faces)

    assert len(report.bodies) == 3
    assert len(report.open_rings) == 2
    assert report.needs_decision
    # Each edge rests against the wall, and is also within reach of the other
    # edge - the ambiguity the diagnosis refuses to resolve.
    for ring in report.open_rings:
        assert len(ring.candidates) >= 2


def test_the_piece_is_sewn_to_the_wall_and_the_file_becomes_one_body() -> None:
    vertices, faces = _artifact()
    join, _ = _join(vertices, faces)
    after, receipt = apply_mesh_repair(vertices, faces, [join])

    assert receipt.body_count_before == 3
    assert receipt.body_count_after == 1
    assert receipt.open_ring_count_before == 2
    assert receipt.open_ring_count_after == 0
    assert receipt.non_manifold_edge_count_after == 0
    assert receipt.removed_face_count > 0
    assert receipt.added_face_count > 0

    report = diagnose_mesh_bodies(vertices, after)
    assert report.one_body
    assert not report.needs_decision


def test_no_vertex_is_added_or_moved() -> None:
    vertices, faces = _artifact()
    before = vertices.copy()
    join, _ = _join(vertices, faces)
    after, _ = apply_mesh_repair(vertices, faces, [join])

    assert np.array_equal(vertices, before)
    assert int(after.max()) < vertices.shape[0]
    # Every face still names a vertex the scan had; the repair writes
    # triangles, never points.
    assert set(np.unique(after).tolist()) <= set(range(vertices.shape[0]))


def test_the_wall_loses_the_band_that_the_joint_buries() -> None:
    vertices, faces = _artifact()
    join, rings = _join(vertices, faces)
    after, receipt = apply_mesh_repair(vertices, faces, [join])

    # The faces taken out are wall, not piece: the piece's own skins survive.
    piece_vertices = np.unique(np.concatenate([np.asarray(ring) for ring in rings]))
    kept = np.unique(after)
    assert np.isin(piece_vertices, kept).all()
    assert receipt.steps[0]["kind"] == "sew_ring_pair_to_body/v1"
    assert receipt.steps[0]["searched_face_count"] >= receipt.removed_face_count


def test_a_repair_replayed_on_the_same_mesh_gives_the_same_mesh() -> None:
    vertices, faces = _artifact()
    join, _ = _join(vertices, faces)
    first, _ = apply_mesh_repair(vertices, faces, [join])
    second, _ = apply_mesh_repair(vertices, faces, [RingPairJoin.from_dict(join.to_dict())])
    assert np.array_equal(first, second)


def test_a_ring_is_named_by_what_it_is_not_by_where_the_walk_began() -> None:
    ring = [7, 3, 11, 5]
    assert ring_sha256(ring) == ring_sha256(np.roll(ring, 2))
    # Direction is part of it: read the other way the loop bounds the other
    # side, and sewing to it would put the band on inside out.
    assert ring_sha256(ring) != ring_sha256(ring[::-1])


def test_a_join_written_for_another_mesh_is_refused() -> None:
    vertices, faces = _artifact()
    join, _ = _join(vertices, faces)
    stranger = RingPairJoin(
        first_ring_sha256="0" * 64,
        second_ring_sha256=join.second_ring_sha256,
        body_sha256=join.body_sha256,
    )
    with pytest.raises(ArtifactMeshRepairError, match="no first ring"):
        apply_mesh_repair(vertices, faces, [stranger])

    wrong_body = RingPairJoin(
        first_ring_sha256=join.first_ring_sha256,
        second_ring_sha256=join.second_ring_sha256,
        body_sha256="1" * 64,
    )
    with pytest.raises(ArtifactMeshRepairError, match="no body"):
        apply_mesh_repair(vertices, faces, [wrong_body])


def test_one_edge_twice_is_not_a_join() -> None:
    with pytest.raises(ArtifactMeshRepairError, match="two different rings"):
        RingPairJoin(
            first_ring_sha256="a" * 64,
            second_ring_sha256="a" * 64,
            body_sha256="b" * 64,
        )


@pytest.mark.parametrize("reach", [0, -1, 10_000_000, 1.5, True])
def test_an_unusable_reach_is_refused(reach: object) -> None:
    with pytest.raises(ArtifactMeshRepairError, match="reach_um"):
        RingPairJoin(
            first_ring_sha256="a" * 64,
            second_ring_sha256="b" * 64,
            body_sha256="c" * 64,
            reach_um=reach,  # type: ignore[arg-type]
        )


def test_a_join_written_with_the_wrong_keys_is_refused() -> None:
    with pytest.raises(ArtifactMeshRepairError, match="written with exactly"):
        RingPairJoin.from_dict({"first_ring_sha256": "a" * 64})


def test_edges_that_rest_on_nothing_are_refused_rather_than_reached_for() -> None:
    # The piece stands well clear of the wall, so there is no joint to repair.
    vertices, faces = _artifact(ring_radius=1.0)
    join, _ = _join(vertices, faces)
    with pytest.raises(ArtifactMeshRepairError, match="within"):
        apply_mesh_repair(vertices, faces, [join])


def test_a_repair_with_no_joins_is_refused() -> None:
    vertices, faces = _artifact()
    with pytest.raises(ArtifactMeshRepairError, match="no joins"):
        apply_mesh_repair(vertices, faces, [])


def test_sewing_takes_the_rings_and_the_body_as_given() -> None:
    # The low-level call does no looking up: what to sew is the caller's, and
    # here the caller is the archaeologist's decision.
    vertices, faces = _artifact()
    rings = boundary_rings(faces)
    labels = face_bodies(faces)
    wall = int(np.argmax(np.bincount(labels)))
    after, stats = sew_ring_pair_to_body(
        vertices,
        faces,
        first_ring=rings[0],
        second_ring=rings[1],
        body_faces=np.flatnonzero(labels == wall),
        reach_um=DEFAULT_REACH_UM,
    )
    assert stats["added_face_count"] > 0
    assert len(boundary_rings(after)) == 0
