"""갈라져 들어온 스캔: the count that no other check makes.

The case these tests are built around is a real one.  24ET0021 came out of
the scanner as three sheets - a pot, and the two skins of a boss standing in
it - and every gate the project had was happy: the sheets are individually
sound, the section merely came out as three closed loops instead of one, and
sewing the boss's two skins to each other left a mesh with no boundary edges
and no non-manifold edges that was still two objects standing apart.  So the
tests below check the two things that would have caught it: that separate
bodies are counted even when each one is perfectly closed, and that an open
ring which lies against more than one thing is reported as a choice rather
than resolved into whichever candidate happens to be nearest.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core import artifact_mesh_bodies as mesh_bodies

from src.core.artifact_mesh_bodies import (
    DEFAULT_SEAM_TOLERANCE_UM,
    JOIN_TO_BODY_SURFACE,
    JOIN_TO_OPEN_RING,
    RING_ONE_CANDIDATE,
    RING_SEVERAL_CANDIDATES,
    RING_UNATTACHED,
    ArtifactMeshBodiesError,
    boundary_rings,
    describe_mesh_bodies,
    diagnose_mesh_bodies,
    face_bodies,
)


def _closed_tube(
    *, radius: float, bottom: float, top: float, columns: int = 16, offset: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """A drum: a wall of one row of quads with a flat lid and floor."""

    angles = np.linspace(0.0, 2.0 * np.pi, columns, endpoint=False)
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    vertices = np.concatenate(
        [
            np.column_stack([ring, np.full(columns, bottom)]),
            np.column_stack([ring, np.full(columns, top)]),
            np.array([[0.0, 0.0, bottom], [0.0, 0.0, top]]),
        ]
    )
    vertices[:, 0] += offset
    low_centre, high_centre = 2 * columns, 2 * columns + 1
    faces = []
    for i in range(columns):
        j = (i + 1) % columns
        faces.append([i, j, columns + j])
        faces.append([i, columns + j, columns + i])
        faces.append([low_centre, j, i])
        faces.append([high_centre, columns + i, columns + j])
    return vertices, np.asarray(faces, dtype=np.int64)


def _open_disc(
    *, radius: float, height: float, columns: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """A fan of triangles from a centre point out to one open ring."""

    angles = np.linspace(0.0, 2.0 * np.pi, columns, endpoint=False)
    rim = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full(columns, height)]
    )
    vertices = np.concatenate([rim, np.array([[0.0, 0.0, height]])])
    centre = columns
    faces = [[centre, i, (i + 1) % columns] for i in range(columns)]
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


def test_one_closed_body_asks_for_no_decision() -> None:
    vertices, faces = _closed_tube(radius=10.0, bottom=0.0, top=20.0)
    report = diagnose_mesh_bodies(vertices, faces)

    assert len(report.bodies) == 1
    assert report.one_body
    assert report.bodies[0].closed
    assert report.open_rings == ()
    assert not report.needs_decision


def test_two_closed_bodies_are_counted_although_every_edge_check_passes() -> None:
    # This is the shape the first repair left 24ET0021 in: two sound, closed,
    # manifold objects in one file.  Boundary and non-manifold counts are both
    # zero here, so the body count is the only thing that can say so.
    vertices, faces = _merge(
        _closed_tube(radius=10.0, bottom=0.0, top=20.0),
        _closed_tube(radius=4.0, bottom=5.0, top=9.0, offset=40.0),
    )
    report = diagnose_mesh_bodies(vertices, faces)

    assert len(report.bodies) == 2
    assert [body.closed for body in report.bodies] == [True, True]
    assert boundary_rings(faces) == []
    assert report.needs_decision

    counted = report.qc_dict()
    assert counted["body_count"] == 2
    assert counted["open_ring_count"] == 0
    assert counted["needs_decision"] is True


def test_a_ring_against_a_wall_and_another_ring_is_left_as_a_choice() -> None:
    # A miniature of the artifact: a drum for the pot, and two open discs
    # standing in it whose rims stop 1 mm short of the wall and 2 mm short of
    # each other.  Both gaps are inside any workable tolerance.
    vertices, faces = _merge(
        _closed_tube(radius=10.0, bottom=0.0, top=20.0),
        _open_disc(radius=9.0, height=10.0),
        _open_disc(radius=9.0, height=12.0),
    )
    report = diagnose_mesh_bodies(vertices, faces)

    assert len(report.bodies) == 3
    assert len(report.open_rings) == 2
    assert report.needs_decision

    for ring in report.open_rings:
        assert ring.verdict == RING_SEVERAL_CANDIDATES
        kinds = [candidate.kind for candidate in ring.candidates]
        assert JOIN_TO_BODY_SURFACE in kinds
        assert JOIN_TO_OPEN_RING in kinds
        # Ranked nearest first, and the wall is nearer than the other disc, so
        # a reader who only looks at the top of the list still sees the wall.
        gaps = [candidate.median_gap_um for candidate in ring.candidates]
        assert gaps == sorted(gaps)
        assert ring.candidates[0].kind == JOIN_TO_BODY_SURFACE
        assert 900 <= ring.candidates[0].median_gap_um <= 1100


def test_a_ring_with_nothing_near_it_is_a_hole_not_a_joint() -> None:
    vertices, faces = _merge(
        _closed_tube(radius=10.0, bottom=0.0, top=20.0),
        _open_disc(radius=3.0, height=10.0),
    )
    report = diagnose_mesh_bodies(vertices, faces)

    (ring,) = report.open_rings
    assert ring.verdict == RING_UNATTACHED
    assert ring.candidates == ()
    # Two bodies still needs a decision - what the loose disc is - but the ring
    # itself has nothing to be sewn to.
    assert report.needs_decision


def test_one_candidate_is_reported_as_one() -> None:
    vertices, faces = _merge(
        _closed_tube(radius=10.0, bottom=0.0, top=20.0),
        _open_disc(radius=9.0, height=10.0),
    )
    report = diagnose_mesh_bodies(vertices, faces)

    (ring,) = report.open_rings
    assert ring.verdict == RING_ONE_CANDIDATE
    (candidate,) = ring.candidates
    assert candidate.kind == JOIN_TO_BODY_SURFACE
    assert candidate.ring_index is None
    assert candidate.body_index == 0


def test_the_tolerance_decides_what_counts_as_near() -> None:
    vertices, faces = _merge(
        _closed_tube(radius=10.0, bottom=0.0, top=20.0),
        _open_disc(radius=9.0, height=10.0),
    )
    far = diagnose_mesh_bodies(vertices, faces, seam_tolerance_um=500)
    (ring,) = far.open_rings
    assert ring.verdict == RING_UNATTACHED

    near = diagnose_mesh_bodies(vertices, faces, seam_tolerance_um=DEFAULT_SEAM_TOLERANCE_UM)
    assert near.open_rings[0].verdict == RING_ONE_CANDIDATE


def test_a_ring_follows_the_winding_of_the_faces_it_borders() -> None:
    vertices, faces = _open_disc(radius=5.0, height=0.0, columns=8)
    (ring,) = boundary_rings(faces)
    assert len(ring) == 8
    directed = {(int(a), int(b)) for a, b, c in faces} | {
        (int(b), int(c)) for a, b, c in faces
    } | {(int(c), int(a)) for a, b, c in faces}
    walked = {(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))}
    # The loop runs the way the faces wind it, so a repair sewing onto it uses
    # each edge the other way round and its new triangles agree.
    assert walked <= directed
    del vertices


def test_two_sheets_meeting_at_one_vertex_are_two_bodies() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    assert len(set(face_bodies(faces).tolist())) == 2
    # And the shared vertex starts two open edges, which is a boundary this
    # module refuses to cut for the drafter.
    with pytest.raises(ArtifactMeshBodiesError, match="pinches"):
        diagnose_mesh_bodies(vertices, faces)


@pytest.mark.parametrize(
    "vertices, faces, message",
    [
        (np.zeros((3, 3)), np.zeros((0, 3), dtype=np.int64), "no faces"),
        (np.zeros((3, 2)), np.array([[0, 1, 2]]), r"\(n, 3\) array of millimetres"),
        (np.zeros((3, 3)), np.array([[0, 1]]), r"\(m, 3\) array of triangles"),
        (np.zeros((3, 3)), np.array([[0, 1, 7]]), "vertex the mesh does not have"),
        (
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, np.nan, 0.0]]),
            np.array([[0, 1, 2]]),
            "finite",
        ),
    ],
)
def test_a_mesh_it_cannot_read_is_refused(
    vertices: np.ndarray, faces: np.ndarray, message: str
) -> None:
    with pytest.raises(ArtifactMeshBodiesError, match=message):
        diagnose_mesh_bodies(vertices, faces)


@pytest.mark.parametrize("tolerance", [0, -1, 2_000_000, 1.5, True])
def test_an_unusable_tolerance_is_refused(tolerance: object) -> None:
    vertices, faces = _closed_tube(radius=10.0, bottom=0.0, top=20.0)
    with pytest.raises(ArtifactMeshBodiesError, match="seam_tolerance_um"):
        diagnose_mesh_bodies(vertices, faces, seam_tolerance_um=tolerance)  # type: ignore[arg-type]


def test_the_read_out_names_the_bodies_and_ranks_the_candidates() -> None:
    vertices, faces = _merge(
        _closed_tube(radius=10.0, bottom=0.0, top=20.0),
        _open_disc(radius=9.0, height=10.0),
        _open_disc(radius=9.0, height=12.0),
    )
    lines = describe_mesh_bodies(diagnose_mesh_bodies(vertices, faces))
    text = "\n".join(lines)

    assert "몸 3개" in text
    # Bodies and rings are numbered from one for a reader, not from zero.
    assert "몸 1:" in text and "고리 1 " in text
    assert "어디에 이을지는 실측자가 정합니다" in text
    # The sentence that would have stopped the mistake: closed and manifold is
    # not the same as whole.
    assert "경계 모서리와 비다양체 수가 0이어도" in text


def test_the_read_out_says_so_when_the_file_is_one_body() -> None:
    vertices, faces = _closed_tube(radius=10.0, bottom=0.0, top=20.0)
    lines = describe_mesh_bodies(diagnose_mesh_bodies(vertices, faces))
    assert lines[0] == "이 파일은 몸 하나입니다."
    assert not any("실측자가 정합니다" in line for line in lines)


def test_a_shredded_file_is_refused_rather_than_ground_through(monkeypatch) -> None:
    # Every ring point is dropped onto every other body and every other ring,
    # so a file in thousands of scraps is a product, not a sum.  It comes back
    # as a refusal with the number in it, not as a wait.
    monkeypatch.setattr(mesh_bodies, "MAX_GAP_POINT_TESTS", 5_000)
    parts = [_open_disc(radius=1.0, height=float(k), columns=64) for k in range(8)]
    vertices, faces = _merge(*parts)
    with pytest.raises(ArtifactMeshBodiesError, match="shredded"):
        diagnose_mesh_bodies(vertices, faces, seam_tolerance_um=3_000)
