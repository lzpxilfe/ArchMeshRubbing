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


def test_three_numbers_cut_the_strip_a_hand_would_paint(
    vessel: tuple[np.ndarray, np.ndarray],
) -> None:
    vertices, faces = vessel
    painted = meridional_strip_faces(
        vertices, faces, center_angle_rad=math.pi / 2.0, width_mm=20.0
    )
    selection = _strip(vessel)

    np.testing.assert_array_equal(selection.face_indices, painted)
    qc = selection.qc_dict()
    assert qc["component_count"] == 1
    assert qc["discarded_component_face_count"] == 0
    assert qc["selected_face_count"] == int(painted.size)


def test_the_strip_holds_its_width_where_the_body_swells(
    vessel: tuple[np.ndarray, np.ndarray],
) -> None:
    vertices, faces = vessel
    selection = _strip(vessel)
    centroids = vertices[faces[selection.face_indices]].mean(axis=1)
    radius = np.hypot(centroids[:, 0], centroids[:, 1])
    offset = np.abs(
        np.mod(
            np.arctan2(centroids[:, 1], centroids[:, 0]) - math.pi / 2.0 + math.pi,
            2.0 * math.pi,
        )
        - math.pi
    )
    arc = offset * radius

    # Every face centre is inside the 20 mm of paper, at the neck and at the
    # belly alike, and the strip really does reach both edges of it.
    assert float(arc.max()) <= 10.0 + 1e-9
    assert float(arc.max()) > 9.0
    # The angular half-width therefore narrows as the radius grows.
    narrow = radius < 30.0
    wide = radius > 45.0
    assert float(offset[narrow].max()) > float(offset[wide].max())


def test_the_height_range_clips_the_strip_at_both_ends(
    vessel: tuple[np.ndarray, np.ndarray],
) -> None:
    whole = _strip(vessel)
    band = _strip(vessel, minimum_height_um=20_000, maximum_height_um=60_000)

    assert band.face_count < whole.face_count
    assert band.qc["minimum_height_um"] >= 20_000
    assert band.qc["maximum_height_um"] <= 60_000
    assert set(band.face_indices.tolist()) <= set(whole.face_indices.tolist())


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
    with pytest.raises(ArtifactSurfaceStripError, match="too narrow"):
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


def test_a_height_cut_across_a_band_can_shed_one_face_and_says_so() -> None:
    """Positioning moves the origin to the measured base, so a height is read
    in the canonical frame; a cut there runs through the middle of a
    triangulated band and can leave a single face hanging by a vertex."""

    session, _vertices, _faces = positioned_vessel_session(segments=48, rings=48)
    parameters = strip_parameters(
        reference_angle_microdegrees=QUARTER_TURN,
        width_um=20_000,
        minimum_height_um=10_000,
    )

    with pytest.raises(ArtifactSurfaceStripError, match="separate pieces"):
        select_positioned_surface_strip(session, parameters)

    kept = select_positioned_surface_strip(
        session, parameters, largest_component=True
    )
    assert kept.qc["discarded_component_face_count"] == 1
    assert kept.face_count > 100


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
    assert 20_000 < int(qc["width_um"]) < 30_000
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
