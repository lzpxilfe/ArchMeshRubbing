"""토기 외면 띠 탁본: a strip of paper on a pot, done by the mesh.

A pot is doubly curved, so no wide area of it flattens without distortion -
paper wrinkles, and a mesh has to say how much.  Rubbers work around it with
a narrow strip down one meridian.  The mesh can do the same, provided the
strip is unrolled about the axis the pot was actually measured on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.application.artifact_measurements import (
    ArtifactMeasurementController,
    ArtifactMeasurementError,
)
from src.application.artifact_workbench import ArtifactWorkbench
from src.core.artifact_tile_unwrap_extractor import (
    SECTION_CENTER_CANONICAL_AXIS,
    SECTION_CENTER_FIT_PER_SECTION,
    STATION_CENTERLINE_ARC,
    STATION_MERIDIAN_ARC,
    ArtifactTileUnwrapError,
    compute_artifact_tile_unwrap,
)
from synthetic_vessel import (
    HEIGHT_MM,
    meridional_strip_faces,
    outer_radius,
    positioned_vessel_session,
)


def _strip(width_mm: float):
    session, vertices, faces = positioned_vessel_session(segments=48, rings=24)
    selected = meridional_strip_faces(
        vertices, faces, center_angle_rad=math.pi / 2.0, width_mm=width_mm
    )
    return session, selected


def _meridian_length_mm() -> float:
    zs = np.linspace(0.0, HEIGHT_MM, 4001)
    rs = np.array([outer_radius(z) for z in zs])
    return float(np.sum(np.hypot(np.diff(zs), np.diff(rs))))


def test_a_narrow_strip_unrolls_about_the_measured_axis_almost_exactly() -> None:
    session, selected = _strip(20.0)

    computation = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="z",
        record_view="top",
        selected_face_indices=selected,
        n_sections=12,
        section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
        station_policy=STATION_MERIDIAN_ARC,
    )

    qc = computation.qc
    # Measured on this profile: 9.2% at the strip's edge, 2.1% on average.
    assert qc["distortion_max_millionths"] < 120_000, qc
    assert qc["distortion_mean_millionths"] < 30_000, qc
    # Paper laid down the belly is as long as the profile, not as tall as the pot.
    assert qc["height_um"] == pytest.approx(_meridian_length_mm() * 1000.0, rel=0.02)
    assert qc["height_um"] > HEIGHT_MM * 1000.0
    assert computation.recipe["section_center_policy"] == SECTION_CENTER_CANONICAL_AXIS
    assert computation.recipe["station_policy"] == STATION_MERIDIAN_ARC


def test_fitting_a_circle_to_a_narrow_strip_is_what_collapses() -> None:
    """The tile default, applied to a pot strip, fails its own quality gate.

    A circle fitted to the short arc a 20 mm strip leaves in each section has
    a centre nowhere near the axis, and the strip is unrolled about it.
    """

    session, selected = _strip(20.0)

    with pytest.raises(ArtifactTileUnwrapError, match="section_distortion"):
        compute_artifact_tile_unwrap(
            session,
            longitudinal_axis="z",
            record_view="top",
            selected_face_indices=selected,
            n_sections=12,
            section_center_policy=SECTION_CENTER_FIT_PER_SECTION,
            station_policy=STATION_CENTERLINE_ARC,
        )


def test_the_axial_station_alone_shortens_the_belly() -> None:
    session, selected = _strip(20.0)

    computation = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="z",
        record_view="top",
        selected_face_indices=selected,
        n_sections=12,
        section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
        station_policy=STATION_CENTERLINE_ARC,
    )

    assert computation.qc["height_um"] == pytest.approx(HEIGHT_MM * 1000.0, rel=0.01)


def test_distortion_grows_with_the_strip_width_and_the_record_says_so() -> None:
    """The width is the rubber's trade-off; the QC makes it a number."""

    maxima = []
    for width in (10.0, 20.0, 40.0):
        session, selected = _strip(width)
        computation = compute_artifact_tile_unwrap(
            session,
            longitudinal_axis="z",
            record_view="top",
            selected_face_indices=selected,
            n_sections=12,
            section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
            station_policy=STATION_MERIDIAN_ARC,
        )
        maxima.append(int(computation.qc["distortion_max_millionths"]))

    assert maxima == sorted(maxima)
    # A 10 mm strip is within 5% everywhere on this profile.
    assert maxima[0] < 50_000


def test_unrolling_about_the_axis_needs_an_artifact_stood_on_one() -> None:
    session, _vertices, _faces = positioned_vessel_session()
    dragged = session.activate_parent_align()
    controller = ArtifactMeasurementController(ArtifactWorkbench(session=dragged))

    with pytest.raises(ArtifactMeasurementError, match="measured rotation axis"):
        controller.begin_tile_unwrap(
            longitudinal_axis="z",
            record_view="top",
            selected_face_indices=(0, 1, 2),
            section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
            station_policy=STATION_MERIDIAN_ARC,
        )
