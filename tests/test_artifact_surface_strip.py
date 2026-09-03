"""회전축 기준 외면 띠: the strip cut from three numbers, not from painting.

A rubber decides a strip by looking at the pot - this meridian, this wide,
from here to here - and the mesh should take the same three numbers.  What it
must not do is take the inner wall, so these tests spend most of their effort
on the two surfaces being told apart, and on the cut refusing rather than
guessing when they cannot be.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.artifact_surface_strip import (
    ArtifactSurfaceStripError,
    DEFAULT_STRIP_NORMAL_ANGLE_MICRODEGREES,
    STRIP_SELECTION_KIND,
    select_positioned_surface_strip,
    select_surface_strip,
    strip_parameters,
    validate_strip_parameters,
)
from src.core.artifact_tile_unwrap_extractor import (
    SECTION_CENTER_CANONICAL_AXIS,
    STATION_MERIDIAN_ARC,
    compute_artifact_tile_unwrap,
)
from synthetic_vessel import (
    HEIGHT_MM,
    hollow_vessel,
    meridional_strip_faces,
    outer_radius,
    outer_wall_faces,
    positioned_vessel_session,
)


QUARTER_TURN = 90_000_000


@pytest.fixture(scope="module")
def vessel() -> tuple[np.ndarray, np.ndarray]:
    vertices, faces, _rim, _floor = hollow_vessel(segments=96, rings=96)
    return vertices, faces


def _strip(vessel: tuple[np.ndarray, np.ndarray], **overrides):
    vertices, faces = vessel
    options = {
        "reference_angle_microdegrees": QUARTER_TURN,
        "width_um": 20_000,
        **overrides,
    }
    return select_surface_strip(vertices, faces, strip_parameters(**options))


def test_the_strip_covers_the_width_that_was_asked_for(
    vessel: tuple[np.ndarray, np.ndarray],
) -> None:
    """Every face holding any of the strip is taken, so the paper is covered.

    A triangle that holds a point of the strip must have a vertex inside it,
    so testing vertices takes every such face.  Testing face centres instead
    let the boundary fall wherever the facet spacing did - and that spacing is
    the arc r x dtheta, wider where the body swells - which quantised the
    width row by row.
    """

    vertices, faces = vessel
    by_centre = meridional_strip_faces(
        vertices, faces, center_angle_rad=math.pi / 2.0, width_mm=20.0
    )
    selection = _strip(vessel)
    chosen = set(selection.face_indices.tolist())

    # Every face whose centre is inside is still taken, plus the ring of
    # faces that straddle the edge.
    assert set(by_centre.tolist()) <= chosen
    assert selection.face_count > int(by_centre.size)
    assert selection.face_count < int(by_centre.size) * 1.3
    qc = selection.qc_dict()
    assert qc["component_count"] == 1
    assert qc["discarded_component_face_count"] == 0

    # Every selected face has a vertex inside the 20 mm of paper, and the
    # selection reaches past both edges so nothing of it is left uncovered.
    corners = vertices[faces[selection.face_indices]]
    radius = np.hypot(corners[:, :, 0], corners[:, :, 1])
    offset = np.abs(
        np.mod(
            np.arctan2(corners[:, :, 1], corners[:, :, 0]) - math.pi / 2.0 + math.pi,
            2.0 * math.pi,
        )
        - math.pi
    )
    arc = offset * radius
    assert bool((arc.min(axis=1) <= 10.0 + 1e-9).all())
    assert float(arc.max()) > 10.0


def test_the_width_is_measured_along_the_surface_not_in_degrees(
    vessel: tuple[np.ndarray, np.ndarray],
) -> None:
    vertices, faces = vessel
    selection = _strip(vessel)
    corners = vertices[faces[selection.face_indices]]
    radius = np.hypot(corners[:, :, 0], corners[:, :, 1]).reshape(-1)
    offset = np.abs(
        np.mod(
            np.arctan2(corners[:, :, 1], corners[:, :, 0]) - math.pi / 2.0 + math.pi,
            2.0 * math.pi,
        )
        - math.pi
    ).reshape(-1)

    # The angular half-width narrows as the radius grows, which is what keeps
    # the paper the same width at the neck and at the belly.
    narrow = radius < 30.0
    wide = radius > 45.0
    assert float(offset[narrow].max()) > float(offset[wide].max())


def test_the_height_range_clips_the_strip_at_both_ends(
    vessel: tuple[np.ndarray, np.ndarray],
) -> None:
    vertices, faces = vessel
    whole = _strip(vessel)
    band = _strip(vessel, minimum_height_um=20_000, maximum_height_um=60_000)

    assert band.face_count < whole.face_count
    assert set(band.face_indices.tolist()) <= set(whole.face_indices.tolist())
    # A face is taken when any vertex is in range, so the band overhangs the
    # cut by at most the one triangle that straddles it.
    corners = vertices[faces[band.face_indices]]
    assert float(corners[:, :, 2].min(axis=1).max()) <= 60.0 + 1e-9
    assert float(corners[:, :, 2].max(axis=1).min()) >= 20.0 - 1e-9
    ring = float(HEIGHT_MM) / 96.0
    assert band.qc["minimum_height_um"] >= int((20.0 - ring) * 1000)
    assert band.qc["maximum_height_um"] <= int((60.0 + ring) * 1000)


def test_the_full_revolution_is_the_outer_wall_and_nothing_else(
    vessel: tuple[np.ndarray, np.ndarray],
) -> None:
    vertices, faces = vessel
    selection = select_surface_strip(vertices, faces, strip_parameters())
    expected = np.flatnonzero(outer_wall_faces(vertices, faces))

    np.testing.assert_array_equal(selection.face_indices, expected)
    # The wall is two sheets of equal size plus rim and floor; taking the
    # outside must leave the inside behind.
    assert selection.face_count * 2 < int(faces.shape[0])
    assert selection.qc["inward_face_count"] > 0
    assert (
        selection.qc["outward_mean_radius_um"]
        > selection.qc["inward_mean_radius_um"]
    )
    # The floor and the rim annulus face along the axis, not away from it.
    centroids = vertices[faces[selection.face_indices]].mean(axis=1)
    assert float(centroids[:, 2].min()) > 0.0
    assert float(centroids[:, 2].max()) < HEIGHT_MM


def test_an_inside_out_mesh_is_refused_rather_than_giving_the_inner_wall(
    vessel: tuple[np.ndarray, np.ndarray],
) -> None:
    vertices, faces = vessel
    flipped = np.ascontiguousarray(faces[:, ::-1])

    with pytest.raises(ArtifactSurfaceStripError, match="wound inside out"):
        select_surface_strip(
            vertices,
            flipped,
            strip_parameters(
                reference_angle_microdegrees=QUARTER_TURN, width_um=20_000
            ),
        )


def test_a_duplicated_face_is_refused_before_the_unwrap_sees_it(
    vessel: tuple[np.ndarray, np.ndarray],
) -> None:
    vertices, faces = vessel
    inside = _strip(vessel).face_indices
    doubled = np.concatenate((faces, faces[inside[:1]]), axis=0)

    with pytest.raises(ArtifactSurfaceStripError, match="directed edges"):
        select_surface_strip(
            vertices,
            doubled,
            strip_parameters(
                reference_angle_microdegrees=QUARTER_TURN, width_um=20_000
            ),
        )


def test_a_strip_broken_in_two_is_refused_unless_the_break_is_accepted(
    vessel: tuple[np.ndarray, np.ndarray],
) -> None:
    vertices, faces = vessel
    selection = _strip(vessel)
    centroids = vertices[faces].mean(axis=1)
    # Punch a hole across the middle of the strip: the paper would now be two
    # pieces, and an unwrap needs one surface.
    hole = np.zeros((faces.shape[0],), dtype=bool)
    hole[selection.face_indices] = True
    hole &= (centroids[:, 2] > 44.0) & (centroids[:, 2] < 47.0)
    assert int(np.count_nonzero(hole)) > 0
    broken = faces[~hole]

    parameters = strip_parameters(
        reference_angle_microdegrees=QUARTER_TURN, width_um=20_000
    )
    with pytest.raises(ArtifactSurfaceStripError) as raised:
        select_surface_strip(vertices, broken, parameters)
    # The refusal names the piece sizes, because those are what tell a cut
    # artifact apart from a window that caught two real surfaces.
    message = str(raised.value)
    assert "2 separate pieces" in message
    kept = select_surface_strip(
        vertices, broken, parameters, largest_component=True
    )
    assert f"{kept.face_count}" in message
    assert kept.qc["component_count"] == 2
    assert kept.qc["discarded_component_face_count"] > 0
    assert (
        kept.face_count + kept.qc["discarded_component_face_count"]
        == kept.qc["outward_face_count"]
    )


def test_an_empty_window_says_which_number_to_change(
    vessel: tuple[np.ndarray, np.ndarray],
) -> None:
    vertices, faces = vessel
    with pytest.raises(ArtifactSurfaceStripError, match="no face lies in this strip"):
        select_surface_strip(
            vertices,
            faces,
            strip_parameters(
                width_um=20_000,
                minimum_height_um=int(HEIGHT_MM * 1000) + 10_000,
                maximum_height_um=int(HEIGHT_MM * 1000) + 20_000,
            ),
        )
    # A strip narrower than the vertex spacing holds no vertex at all, so no
    # face holds any of it.
    with pytest.raises(ArtifactSurfaceStripError, match="no face lies in this strip"):
        select_surface_strip(
            vertices,
            faces,
            strip_parameters(
                reference_angle_microdegrees=QUARTER_TURN, width_um=200
            ),
        )


def test_the_parameters_are_a_closed_integer_contract() -> None:
    parameters = strip_parameters(
        reference_angle_microdegrees=QUARTER_TURN, width_um=20_000
    )
    assert parameters["kind"] == STRIP_SELECTION_KIND
    assert (
        parameters["maximum_normal_angle_microdegrees"]
        == DEFAULT_STRIP_NORMAL_ANGLE_MICRODEGREES
    )
    assert validate_strip_parameters(parameters) == parameters

    with pytest.raises(ArtifactSurfaceStripError, match="must be an integer"):
        strip_parameters(width_um=20_000.5)  # type: ignore[arg-type]
    with pytest.raises(ArtifactSurfaceStripError, match="inclusive range"):
        strip_parameters(reference_angle_microdegrees=180_000_000)
    with pytest.raises(ArtifactSurfaceStripError, match="below maximum_height_um"):
        strip_parameters(minimum_height_um=50_000, maximum_height_um=50_000)
    with pytest.raises(ArtifactSurfaceStripError, match="x, y, or z"):
        strip_parameters(longitudinal_axis="w")
    with pytest.raises(ArtifactSurfaceStripError, match="keys are invalid"):
        validate_strip_parameters({**parameters, "extra": 1})


def test_cutting_a_strip_needs_an_artifact_stood_on_its_axis() -> None:
    session, _vertices, _faces = positioned_vessel_session(segments=48, rings=24)
    parameters = strip_parameters(
        reference_angle_microdegrees=QUARTER_TURN, width_um=20_000
    )

    selection = select_positioned_surface_strip(session, parameters)
    assert selection.face_count > 0

    dragged = session.activate_parent_align()
    with pytest.raises(ArtifactSurfaceStripError, match="measured rotation axis"):
        select_positioned_surface_strip(dragged, parameters)


def test_a_height_cut_across_a_band_leaves_one_piece() -> None:
    """Positioning moves the origin to the measured base, so a height is read
    in the canonical frame; the cut then runs through the middle of a
    triangulated band, and the window must not shed the faces it crosses.

    Testing the window at the face centre used to drop the triangles that
    straddle the cut and leave one of them hanging by a vertex, which the cut
    then refused as two pieces.  Testing it at the vertices keeps every
    straddling triangle, so the band comes off whole and the paper is covered
    down to the height that was asked for.
    """

    session, _vertices, _faces = positioned_vessel_session(segments=48, rings=48)
    parameters = strip_parameters(
        reference_angle_microdegrees=QUARTER_TURN,
        width_um=20_000,
        minimum_height_um=10_000,
    )

    selection = select_positioned_surface_strip(session, parameters)

    assert selection.qc["component_count"] == 1
    assert selection.qc["discarded_component_face_count"] == 0
    assert selection.face_count > 100
    # The cut is covered from both sides: the strip reaches below 10 mm, but
    # by less than the one triangle ring that straddles the cut.
    ring_um = int(HEIGHT_MM * 1000.0 / 48.0)
    assert int(selection.qc["minimum_height_um"]) < 10_000
    assert int(selection.qc["minimum_height_um"]) >= 10_000 - ring_um


def test_the_cut_strip_unrolls_the_way_the_painted_one_did() -> None:
    session, _vertices, _faces = positioned_vessel_session(segments=48, rings=48)
    selection = select_positioned_surface_strip(
        session,
        strip_parameters(
            reference_angle_microdegrees=QUARTER_TURN,
            width_um=20_000,
        ),
    )

    computation = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="z",
        record_view="top",
        selected_face_indices=selection.face_indices,
        n_sections=12,
        section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
        station_policy=STATION_MERIDIAN_ARC,
    )

    qc = computation.qc
    assert int(qc["selected_face_count"]) == selection.face_count
    # A 20 mm strip on this profile: the numbers docs/POTTERY_STRIP_UNWRAP.md
    # measured for the painted one.
    assert int(qc["distortion_max_millionths"]) < 120_000
    # The selection overhangs the requested width by the ring of triangles
    # that straddles each edge, which is what makes the 20 mm fully covered.
    # One facet spans the arc 2 pi r / segments, widest where the body swells.
    facet_arc_um = (
        2.0 * math.pi * float(selection.qc["maximum_radius_um"]) / 48.0
    )
    assert 20_000 < int(qc["width_um"]) < 20_000 + 2.0 * facet_arc_um
    profile = np.linspace(0.0, HEIGHT_MM, 2001)
    meridian = float(
        np.sum(
            np.hypot(
                np.diff(profile),
                np.diff(np.array([outer_radius(z) for z in profile])),
            )
        )
    )
    assert abs(int(qc["height_um"]) / 1000.0 - meridian) / meridian < 0.03
