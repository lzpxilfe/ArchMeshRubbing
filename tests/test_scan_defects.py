"""What the program does when the mesh is a scan and not a solid.

Until now every test handed the program a closed, clean, axis-aligned body.
docs/REAL_DATA_TRIAL.md records what a real one is instead - the museum's
빗살무늬토기 is a body of 391,432 faces with a loose crumb of 24 beside it, 11
boundary edges, 28 non-manifold edges, and its height on Y - and these tests
put each of those onto a generated vessel and ask what happens.

They are not all assertions that the program is right.  Some of them pin
behaviour that is wrong and known to be wrong, so that fixing it has to
come here and say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from scan_defects import (
    add_loose_crumb,
    bite_the_rim,
    mesh_report,
    punch_hole,
    stand_it_wrong,
    warp,
)
from synthetic_vessel import HEIGHT_MM, hollow_vessel, outer_radius


def _vessel(segments: int = 48, rings: int = 16):
    vertices, faces, _rim, _floor = hollow_vessel(segments=segments, rings=rings)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def test_a_generated_vessel_is_the_solid_a_scan_never_is() -> None:
    """The baseline, so the defects below are measured against something."""

    report = mesh_report(*_vessel())

    assert report["connected_piece_count"] == 1
    assert report["boundary_edge_count"] == 0
    assert report["nonmanifold_edge_count"] == 0
    # Which is exactly what the museum scan is not: 2 pieces, 11 boundary
    # edges, 28 non-manifold edges.  A gate that has only ever seen this mesh
    # has never been asked the question a real one asks.


def test_a_loose_crumb_is_a_second_piece_and_the_mesh_report_says_so() -> None:
    """A scanner's fleck is a solid of its own, beside the artifact."""

    vertices, faces = _vessel()
    before = mesh_report(vertices, faces)
    crumbed = add_loose_crumb(vertices, faces, size_mm=1.5, gap_mm=3.0)
    after = mesh_report(*crumbed)

    assert after["connected_piece_count"] == 2
    assert after["largest_piece_faces"] == before["face_count"]
    assert after["smallest_piece_faces"] == 4
    # It is closed, so it adds no boundary edge: a crumb is not a hole.
    assert after["boundary_edge_count"] == before["boundary_edge_count"]
    # And it is outside the artifact, not inside it.
    points = crumbed[0]
    assert float(points[:, 0].max()) > float(vertices[:, 0].max())


def test_a_hole_and_a_broken_rim_leave_the_open_edges_a_scan_has() -> None:
    """A scan is open where the scanner could not see, and where the pot broke."""

    vertices, faces = _vessel()
    assert mesh_report(vertices, faces)["boundary_edge_count"] == 0

    holed = punch_hole(
        vertices,
        faces,
        centre_mm=(outer_radius(HEIGHT_MM * 0.5), 0.0, HEIGHT_MM * 0.5),
        radius_mm=6.0,
    )
    holed_report = mesh_report(*holed)
    assert holed_report["boundary_edge_count"] > 0
    assert holed_report["face_count"] < len(faces)
    # A hole takes faces away; it does not break the pot into pieces.
    assert holed_report["connected_piece_count"] == 1

    bitten = bite_the_rim(vertices, faces, from_angle_deg=20.0, to_angle_deg=95.0, depth_mm=12.0)
    bitten_report = mesh_report(*bitten)
    assert bitten_report["boundary_edge_count"] > 0
    assert bitten_report["face_count"] < len(faces)
    # The bite reaches the top, so the rim is no longer a closed ring.
    top = float(vertices[:, 2].max())
    kept = bitten[0][bitten[1]].mean(axis=1)
    at_the_top = kept[kept[:, 2] > top - 1.0]
    angles = np.degrees(np.arctan2(at_the_top[:, 1], at_the_top[:, 0]))
    assert not ((angles > 30.0) & (angles < 85.0)).any()


def test_standing_it_wrong_is_what_a_scan_file_actually_hands_over() -> None:
    """정치 has never been asked to do anything: the tests started upright."""

    vertices, faces = _vessel()
    standing = vertices.max(axis=0) - vertices.min(axis=0)
    upright = float(standing[2])

    # The museum file lies on its side, with the height on Y.
    on_its_side = stand_it_wrong(vertices, roll_deg=-90.0)
    extent = on_its_side.max(axis=0) - on_its_side.min(axis=0)
    # The height has moved out of Z and into Y, and the width has taken its
    # place - which is why the file's tallest extent says nothing about which
    # way up the pot is.
    assert float(extent[1]) == pytest.approx(upright, rel=1e-6)
    assert float(extent[2]) == pytest.approx(float(standing[1]), rel=1e-6)
    assert float(extent[2]) > upright

    # And turned about the vertical as well, with its centroid nowhere near
    # the origin - which is the state a published scan is actually in.
    as_scanned = stand_it_wrong(
        vertices, roll_deg=-90.0, yaw_deg=17.0, offset_mm=(-40.0, 220.0, -310.0)
    )
    assert float(np.linalg.norm(as_scanned.mean(axis=0))) > 100.0
    # Nothing about the artifact changed: it is the same solid, moved.
    assert mesh_report(as_scanned, faces) == mesh_report(vertices, faces)


def test_warping_takes_the_pot_off_being_a_surface_of_revolution() -> None:
    """A fired pot is out of true, and no axis fit makes that go away.

    This is the assumption most of the pipeline rests on - the rotation axis,
    the strip development, the mirrored 반입면·반단면 - and every test so far
    has handed it a body that really is a surface of revolution.
    """

    vertices, faces = _vessel(segments=96, rings=24)

    def out_of_round(points: np.ndarray) -> float:
        """Worst spread of radius around one ring, in millimetres."""

        worst = 0.0
        for height in (0.25, 0.5, 0.75):
            z = HEIGHT_MM * height
            here = points[np.abs(points[:, 2] - z) < 1.0]
            if here.size == 0:
                continue
            radius = np.hypot(here[:, 0], here[:, 1])
            outer = radius[radius > np.median(radius)]
            worst = max(worst, float(outer.max() - outer.min()))
        return worst

    assert out_of_round(vertices) < 0.5
    ovalled = warp(vertices, oval_mm=2.0)
    assert out_of_round(ovalled) > 3.0

    leaning = warp(vertices, lean_mm=6.0)
    def axis_at(points: np.ndarray, height: float) -> float:
        # Half a ring's spacing, so a band always holds a ring of the mesh.
        band = HEIGHT_MM / 24.0
        here = points[np.abs(points[:, 2] - HEIGHT_MM * height) < band]
        assert here.size, height
        return float((here[:, 0].min() + here[:, 0].max()) / 2.0)

    # A lean moves the centre of the section with height: one axis cannot
    # pass through both ends.
    assert abs(axis_at(vertices, 0.1) - axis_at(vertices, 0.9)) < 0.5
    assert abs(axis_at(leaning, 0.1) - axis_at(leaning, 0.9)) > 4.0

    # Warping is the artifact's shape, so it does not break the mesh.
    assert mesh_report(ovalled, faces)["connected_piece_count"] == 1
    assert mesh_report(leaning, faces)["boundary_edge_count"] == 0


def test_every_defect_is_the_same_mesh_twice() -> None:
    """A test that fails must fail the same way tomorrow.

    Nothing here draws on a random state: the crumb's height is hashed from
    its seed, and everything else is a function of the mesh it was given.
    """

    vertices, faces = _vessel(segments=24, rings=8)
    for make in (
        lambda: add_loose_crumb(vertices, faces, seed=3),
        lambda: punch_hole(
            vertices, faces, centre_mm=(30.0, 0.0, HEIGHT_MM / 2.0), radius_mm=5.0
        ),
        lambda: bite_the_rim(
            vertices, faces, from_angle_deg=0.0, to_angle_deg=60.0, depth_mm=8.0
        ),
    ):
        first_points, first_faces = make()
        again_points, again_faces = make()
        assert np.array_equal(first_points, again_points)
        assert np.array_equal(first_faces, again_faces)
    assert np.array_equal(
        stand_it_wrong(vertices, roll_deg=-90.0, yaw_deg=17.0),
        stand_it_wrong(vertices, roll_deg=-90.0, yaw_deg=17.0),
    )


def test_a_crumb_beside_the_pot_is_refused_and_named_rather_than_drawn() -> None:
    """Scanner noise does not reach the paper as if it were the artifact.

    A loose crumb beside the pot projects to its own closed outline.  Under
    outline algorithm 1.1.0 the extractor drew it - two exterior components
    where the artifact has one, an area 0.8 mm2 over the truth, no refusal,
    no warning - and a drafter who did not look closely published a speck of
    the turntable as part of a Neolithic pot.  1.2.0 refuses, and says what
    it saw in millimetres so the drafter can find the crumb.  A connected
    mesh has a connected silhouette, so there is nothing else two pieces can
    be.  Records written at 1.1.0 still recompute as they were written.
    """

    from src.core.artifact_outline_extractor import (
        OUTLINE_ALGORITHM_VERSION,
        OUTLINE_CLOSING_ALGORITHM_VERSION,
        extract_outline_geometry,
        outline_recipe,
    )
    from src.core.artifact_vector_extractor import ArtifactVectorExtractionError

    # A record made today is written under the gated version.
    assert outline_recipe("front", precision_grid_mm=0.2)["algorithm_version"] == (
        OUTLINE_ALGORITHM_VERSION
    )
    vertices, faces = _vessel(segments=64, rings=20)
    clean = extract_outline_geometry(vertices, faces, "front", precision_grid_mm=0.2)
    assert clean.qc["component_count"] == 1

    # Ten millimetres clear of the pot, so it is unmistakably not the pot.
    crumbed_vertices, crumbed_faces = add_loose_crumb(
        vertices, faces, size_mm=1.5, gap_mm=10.0
    )
    with pytest.raises(ArtifactVectorExtractionError) as refusal:
        extract_outline_geometry(
            crumbed_vertices, crumbed_faces, "front", precision_grid_mm=0.2
        )
    message = str(refusal.value)
    assert "more than one piece" in message
    assert "2 separate pieces" in message
    assert "loose fragment" in message

    # The version the record was written under is the version it recomputes
    # under: 1.1.0 still draws the crumb, exactly as it did.
    as_written = extract_outline_geometry(
        crumbed_vertices,
        crumbed_faces,
        "front",
        precision_grid_mm=0.2,
        algorithm_version=OUTLINE_CLOSING_ALGORITHM_VERSION,
    )
    assert as_written.qc["component_count"] == 2
    assert float(as_written.qc["outline_area_mm2"]) > float(clean.qc["outline_area_mm2"])

    # And an outline that is one piece is the same bytes at 1.2.0 as at
    # 1.1.0: the gate moves no vertex.
    closing = extract_outline_geometry(
        vertices,
        faces,
        "front",
        precision_grid_mm=0.2,
        algorithm_version=OUTLINE_CLOSING_ALGORITHM_VERSION,
    )
    assert closing.payload.sha256 == clean.payload.sha256


def test_a_hole_or_a_broken_rim_does_not_confuse_the_outline() -> None:
    """The two defects that are the artifact's own, and are handled.

    A hole in the wall is behind the silhouette, so the outline closes over
    it as it should; a broken rim is a real edge, and the outline follows it.
    Neither adds a component.  These pass today and are here so that a change
    made for the crumb does not quietly break them.
    """

    from src.core.artifact_outline_extractor import extract_outline_geometry

    vertices, faces = _vessel(segments=64, rings=20)
    holed = extract_outline_geometry(
        *punch_hole(
            vertices,
            faces,
            centre_mm=(outer_radius(HEIGHT_MM * 0.5), 0.0, HEIGHT_MM * 0.5),
            radius_mm=6.0,
        ),
        "front",
        precision_grid_mm=0.2,
    )
    assert holed.qc["component_count"] == 1
    assert sum(1 for p in holed.payload.paths if str(p.role) == "hole") == 0

    bitten = extract_outline_geometry(
        *bite_the_rim(
            vertices, faces, from_angle_deg=20.0, to_angle_deg=95.0, depth_mm=12.0
        ),
        "front",
        precision_grid_mm=0.2,
    )
    assert bitten.qc["component_count"] == 1
