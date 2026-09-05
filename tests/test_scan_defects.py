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

import re

import numpy as np
import pytest

from scan_defects import (
    add_loose_crumb,
    bite_the_rim,
    bridge_the_wall,
    dent_the_wall,
    fill_with_plaster,
    mesh_report,
    punch_hole,
    roughen,
    sharpen_the_base,
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


def test_a_join_meshed_across_the_wall_splits_the_section_and_not_the_outline() -> None:
    """The museum pot's broken section wall, reproduced.

    At one restoration join the scan meshes the inside of the wall, so a
    section through it closes into two loops - the wall looks cut through at
    that height - while the silhouette, which the join sits inside, does not
    change at all.  The generated join is the tidy version of the real one:
    two fracture faces and an open window between them, without the real
    scan's tangle of non-manifold edges.
    """

    from src.core.artifact_outline_extractor import extract_outline_geometry
    from src.core.artifact_vector_extractor import extract_cutline_geometry
    from src.core.artifact_vector_record import PlanarFrame

    vertices, faces = _vessel(segments=48, rings=16)
    assert mesh_report(vertices, faces)["boundary_edge_count"] == 0
    bridged = bridge_the_wall(
        vertices, faces, z_mm=HEIGHT_MM * 0.5, from_angle_deg=-25.0, to_angle_deg=25.0
    )
    report = mesh_report(*bridged)
    assert report["connected_piece_count"] == 1
    # The window's sides are open edges, as the real join has.
    assert report["boundary_edge_count"] > 0

    # A section through the join: one closed loop becomes two.
    plane = PlanarFrame(
        origin_world_mm=(0.0, 0.37, 0.0),
        u_axis_world=(1.0, 0.0, 0.0),
        v_axis_world=(0.0, 0.0, 1.0),
        normal_world=(0.0, -1.0, 0.0),
    )
    whole = extract_cutline_geometry(vertices, faces, plane)
    assert [path.closed for path in whole.payload.paths] == [True]
    split = extract_cutline_geometry(*bridged, plane)
    assert [path.closed for path in split.payload.paths] == [True, True]
    # One loop is the whole pot less the wall above the break; the other is
    # the piece of wall above it, which reaches down only to the break.
    lowest = sorted(
        min(point[1] for point in path.points_mm) for path in split.payload.paths
    )
    assert lowest[0] < 1.0
    assert abs(lowest[1] - HEIGHT_MM * 0.5) < 8.0

    # The join faces +X, so the right view looks straight at it and its
    # silhouette does not see it: the outline is the same bytes.
    plain = extract_outline_geometry(vertices, faces, "right", precision_grid_mm=0.2)
    joined = extract_outline_geometry(*bridged, "right", precision_grid_mm=0.2)
    assert joined.payload.sha256 == plain.payload.sha256


def test_a_tangled_join_stops_the_section_as_the_real_pots_does_off_centre() -> None:
    """The real join, one plane over: a branching junction, refused.

    The museum pot's section is two loops at y = 0 and a refusal at y = -6,
    where the plane meets the join's tangle of non-manifold edges.  The
    tangled join reproduces the refusal; the tidy one, the two loops.
    """

    from src.core.artifact_vector_extractor import (
        ArtifactVectorExtractionError,
        extract_cutline_geometry,
    )
    from src.core.artifact_vector_record import PlanarFrame

    vertices, faces = _vessel(segments=48, rings=16)
    tangled = bridge_the_wall(
        vertices,
        faces,
        z_mm=HEIGHT_MM * 0.5,
        from_angle_deg=-25.0,
        to_angle_deg=25.0,
        tangled=True,
    )
    report = mesh_report(*tangled)
    assert report["nonmanifold_edge_count"] > 0
    assert report["connected_piece_count"] == 1

    plane = PlanarFrame(
        origin_world_mm=(0.0, 0.37, 0.0),
        u_axis_world=(1.0, 0.0, 0.0),
        v_axis_world=(0.0, 0.0, 1.0),
        normal_world=(0.0, -1.0, 0.0),
    )
    with pytest.raises(ArtifactVectorExtractionError, match="non-manifold branching"):
        extract_cutline_geometry(*tangled, plane)


def test_a_pit_unrolls_but_an_undercut_folds_the_development_and_the_refusal_says_where() -> None:
    """The museum pot's strip development was refused for a UV overlap, and
    the overlap turned out to be one spot: steep faces 2 to 6 mm under the
    wall at one height, a chip whose fracture face the scanner meshed under
    the skin.  A pit alone unrolls - a cylindrical development keeps angle
    and station, and a dent changes neither.  A pit whose floor tucks under
    the lip above puts two sheets over one station, and then the refusal
    names the spot, so the window can be moved past it."""

    from src.core.artifact_surface_strip import select_positioned_surface_strip, strip_parameters
    from src.core.artifact_tile_unwrap_extractor import (
        SECTION_CENTER_CANONICAL_AXIS,
        STATION_MERIDIAN_ARC,
        ArtifactTileUnwrapError,
        compute_artifact_tile_unwrap,
    )
    from synthetic_vessel import positioned_vessel_session

    z = HEIGHT_MM * 0.5
    centre = (0.0, -outer_radius(z), z)

    def unwrap_through(undercut_mm: float):
        def dent(vertices, faces):
            return dent_the_wall(
                vertices, faces, centre_mm=centre, radius_mm=6.0, depth_mm=4.0, undercut_mm=undercut_mm
            )

        session, _v, _f = positioned_vessel_session(
            segments=96, rings=40, defect=dent, document_id=f"artifact:dent-{undercut_mm:g}"
        )
        strip = select_positioned_surface_strip(
            session,
            strip_parameters(
                reference_angle_microdegrees=-90_000_000,
                width_um=20_000,
                minimum_height_um=4_000,
                maximum_height_um=70_000,
            ),
        )
        return compute_artifact_tile_unwrap(
            session,
            longitudinal_axis="z",
            record_view="top",
            selected_face_indices=strip.face_indices,
            n_sections=12,
            section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
            station_policy=STATION_MERIDIAN_ARC,
        )

    plain = unwrap_through(0.0)
    assert plain.qc_dict()["uv_overlap_pair_count"] == 0

    # The vessel's rings are 2.25 mm apart; a 3 mm undercut tucks the floor
    # under the next ring up.
    with pytest.raises(ArtifactTileUnwrapError, match="folds over around canonical") as caught:
        unwrap_through(3.0)
    named = re.search(r"\(([-\d.]+), ([-\d.]+), ([-\d.]+)\) mm", str(caught.value))
    assert named is not None
    spot = np.array([float(named.group(k)) for k in (1, 2, 3)])
    assert float(np.linalg.norm(spot - np.asarray(centre))) < 10.0


def test_a_restoration_fill_is_blank_and_is_the_face_set_a_restored_record_marks() -> None:
    """Plaster across a gap: no relief on the scan, ``restored`` on the drawing.

    The fill takes the grain off a patch of wall and sets it a little under
    the surface; the mesh keeps its faces, so the patch is a face set, which
    is what a condition record carries.  Its projection in the view that
    looks straight at it is the patch's own area.
    """

    from src.core.artifact_condition_annotation import compute_condition_annotation
    from synthetic_vessel import grained_surface, outer_radius, positioned_vessel_session

    z = HEIGHT_MM * 0.6
    centre = (0.0, outer_radius(z), z)

    def fill(vertices, faces):
        filled_vertices, filled_faces, _filled = fill_with_plaster(
            vertices, faces, centre_mm=centre, radius_mm=8.0, wall_radius=outer_radius
        )
        return filled_vertices, filled_faces

    grained, vertices, faces = positioned_vessel_session(
        segments=96, rings=40, relief=grained_surface, document_id="artifact:grained"
    )
    session, filled_vertices, filled_faces = positioned_vessel_session(
        segments=96, rings=40, relief=grained_surface, defect=fill, document_id="artifact:filled"
    )
    _v, _f, filled = fill_with_plaster(
        vertices, faces, centre_mm=centre, radius_mm=8.0, wall_radius=outer_radius
    )
    assert len(filled) > 20
    assert mesh_report(filled_vertices, filled_faces) == mesh_report(vertices, faces)

    def roughness(points: np.ndarray) -> float:
        within = np.linalg.norm(points - np.asarray(centre), axis=1) <= 8.0
        within &= np.hypot(points[:, 0], points[:, 1]) > outer_radius(z) - 3.0
        radius = np.hypot(points[within, 0], points[within, 1])
        nominal = np.array([outer_radius(float(h)) for h in points[within, 2]])
        return float(np.std(radius - nominal))

    assert roughness(vertices) > 0.03
    assert roughness(filled_vertices) < 1e-9

    record = compute_condition_annotation(
        session, condition="restored", face_indices=filled, precision_grid_mm=0.05
    )
    front = next(view for view in record.payload.views if view.view == "front")
    triangles = filled_vertices[filled_faces[filled]]
    projected = 0.5 * np.abs(
        (triangles[:, 1, 0] - triangles[:, 0, 0]) * (triangles[:, 2, 2] - triangles[:, 0, 2])
        - (triangles[:, 1, 2] - triangles[:, 0, 2]) * (triangles[:, 2, 0] - triangles[:, 0, 0])
    ).sum()
    assert front.qc_summary()["area_mm2"] == pytest.approx(projected, rel=0.1)
    assert front.qc_summary()["component_count"] == 1
    del grained


def test_a_pointed_base_still_positions_and_is_drawn_to_a_point() -> None:
    """첨저: no foot to stand on, and the drawing shows the point.

    The rotation axis is taken from the rim and the inner floor, so a vessel
    with nothing flat underneath positions all the same; its front outline
    comes to a point a few millimetres wide where the flat one is fifty.
    """

    from src.core.artifact_outline_extractor import compute_artifact_outline
    from src.core.artifact_vector_extractor import compute_artifact_cutline
    from src.core.artifact_vector_record import PlanarFrame
    from synthetic_vessel import positioned_vessel_session

    def width_at_the_bottom(session) -> float:
        outline = compute_artifact_outline(session, "front", precision_grid_mm=0.2)
        points = np.asarray(
            next(path for path in outline.payload.paths if str(path.role) == "exterior").points_mm
        )
        lowest = float(points[:, 1].min())
        near = points[points[:, 1] < lowest + 3.0]
        return float(near[:, 0].max() - near[:, 0].min())

    flat, _v, _f = positioned_vessel_session(segments=48, rings=16, document_id="artifact:flat")
    pointed, _pv, _pf = positioned_vessel_session(
        segments=48, rings=16, defect=sharpen_the_base, document_id="artifact:pointed"
    )
    assert width_at_the_bottom(flat) > 40.0
    assert width_at_the_bottom(pointed) < 12.0

    section = compute_artifact_cutline(
        pointed,
        PlanarFrame(
            origin_world_mm=(0.0, 0.37, 0.0),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 0.0, 1.0),
            normal_world=(0.0, -1.0, 0.0),
        ),
    )
    points = np.vstack([np.asarray(path.points_mm) for path in section.payload.paths])
    apex = points[np.argmin(points[:, 1])]
    assert abs(float(apex[0])) < 4.0


def test_scanner_noise_moves_no_outline_and_no_section() -> None:
    """A few hundredths of a millimetre on every vertex, along its normal.

    The outline and the section shrug it off: at 0.1 mm of noise on a
    96 x 40 vessel the front outline's area moves by a few parts in ten
    thousand and the section stays one closed ring, at a coarse grid and a
    fine one.  The gates that watch for the grid's holes and severed pieces
    are not tripped by noise of this size.
    """

    from src.core.artifact_outline_extractor import extract_outline_geometry
    from src.core.artifact_vector_extractor import extract_cutline_geometry
    from src.core.artifact_vector_record import PlanarFrame
    from synthetic_vessel import grained_surface

    vertices, faces, _rim, _floor = hollow_vessel(segments=96, rings=40, relief=grained_surface)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    noisy = roughen(vertices, faces, amplitude_mm=0.1)
    moved = np.linalg.norm(noisy - vertices, axis=1)
    assert 0.09 < float(moved.max()) <= 0.1 + 1e-9
    assert np.array_equal(noisy, roughen(vertices, faces, amplitude_mm=0.1))
    assert mesh_report(noisy, faces) == mesh_report(vertices, faces)

    plane = PlanarFrame(
        origin_world_mm=(0.0, 0.37, 0.0),
        u_axis_world=(1.0, 0.0, 0.0),
        v_axis_world=(0.0, 0.0, 1.0),
        normal_world=(0.0, -1.0, 0.0),
    )
    for grid in (0.25, 0.05):
        clean = extract_outline_geometry(vertices, faces, "front", precision_grid_mm=grid)
        rough = extract_outline_geometry(noisy, faces, "front", precision_grid_mm=grid)
        assert rough.qc["hole_count"] == 0
        assert rough.qc["component_count"] == 1
        assert rough.qc["outline_area_mm2"] == pytest.approx(clean.qc["outline_area_mm2"], rel=3e-3)
    section = extract_cutline_geometry(noisy, faces, plane)
    assert [path.closed for path in section.payload.paths] == [True]


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


def _through_hole(vertices, faces, *, radius_mm: float = 6.0):
    """A hole through both walls at mid-height, where the front view looks."""

    from synthetic_vessel import WALL_MM

    z = HEIGHT_MM * 0.5
    for sign in (1.0, -1.0):
        for radius in (outer_radius(z), outer_radius(z) - WALL_MM):
            vertices, faces = punch_hole(
                vertices, faces, centre_mm=(0.0, sign * radius, z), radius_mm=radius_mm
            )
    return vertices, faces


def test_a_hole_through_the_artifact_is_drawn_and_measured() -> None:
    """A hole the artifact has stays a hole, at any grid.

    The front view looks along Y, so a hole through both walls on the Y axis
    is a hole in the silhouette.  The unsnapped union has it too, and covers
    only the snap error at its rim - well under the fraction the gate
    refuses at - and the outline records how much.
    """

    from src.core.artifact_outline_extractor import (
        OUTLINE_GRID_HOLE_COVER_FRACTION_MAX,
        OUTLINE_PIECE_GATE_ALGORITHM_VERSION,
        extract_outline_geometry,
    )

    vertices, faces = _through_hole(*_vessel(segments=64, rings=20))
    for grid in (0.2, 1.0):
        holed = extract_outline_geometry(vertices, faces, "front", precision_grid_mm=grid)
        assert holed.qc["hole_count"] == 1
        assert holed.qc["component_count"] == 1
        cover = holed.qc["grid_hole_unsnapped_cover_max"]
        assert 0.0 < cover < OUTLINE_GRID_HOLE_COVER_FRACTION_MAX / 4.0, (grid, cover)
        # The gate moves no vertex: 1.2.0 draws the same bytes, without the
        # measurement.
        as_1_2 = extract_outline_geometry(
            vertices,
            faces,
            "front",
            precision_grid_mm=grid,
            algorithm_version=OUTLINE_PIECE_GATE_ALGORITHM_VERSION,
        )
        assert as_1_2.payload.sha256 == holed.payload.sha256
        assert "grid_hole_unsnapped_cover_max" not in as_1_2.qc

    # No hole at all: the measurement is zero, and the key is still there.
    plain = extract_outline_geometry(*_vessel(), "front", precision_grid_mm=0.5)
    assert plain.qc["hole_count"] == 0
    assert plain.qc["grid_hole_unsnapped_cover_max"] == 0.0


def test_a_hole_the_grid_punched_is_refused_by_size_rather_than_drawn() -> None:
    """A grid coarser than the mesh opens holes the artifact does not have.

    On a vessel meshed at 200 x 60 a 0.5 mm grid collapses a fifth of the
    projected triangles, and four holes 3 to 4 mm wide open in the lattice
    union where the wall is seen edge-on - beside the one hole the artifact
    really has.  Outline 1.2.0 drew all five.  The unsnapped union covers the
    four entirely and the real one by under 3 %, so 1.3.0 refuses, says four
    of five are the grid's, and gives the size of the largest in
    millimetres.  The museum pot's top view at 1.0 mm is the same case with
    ten holes.
    """

    from src.core.artifact_outline_extractor import (
        OUTLINE_PIECE_GATE_ALGORITHM_VERSION,
        extract_outline_geometry,
    )
    from src.core.artifact_vector_extractor import ArtifactVectorExtractionError

    vertices, faces = _through_hole(*_vessel(segments=200, rings=60))
    with pytest.raises(ArtifactVectorExtractionError) as refusal:
        extract_outline_geometry(vertices, faces, "front", precision_grid_mm=0.5)
    message = str(refusal.value)
    assert "4 of 5 holes in the outline are the grid's" in message
    assert "covers 100% of a hole" in message
    assert "mm" in message and "finer precision_grid_mm" in message

    # As written under 1.2.0 the same outline passed, holes and all.
    drawn = extract_outline_geometry(
        vertices,
        faces,
        "front",
        precision_grid_mm=0.5,
        algorithm_version=OUTLINE_PIECE_GATE_ALGORITHM_VERSION,
    )
    assert drawn.qc["hole_count"] == 5
