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
from src.core.artifact_record_validation import validate_known_records
from src.core.artifact_tile_unwrap_export import (
    build_tile_unwrap_export,
    validate_tile_unwrap_export_bytes,
)
from src.core.artifact_tile_unwrap_extractor import (
    SECTION_CENTER_CANONICAL_AXIS,
    SECTION_CENTER_FIT_PER_SECTION,
    STATION_CENTERLINE_ARC,
    STATION_MERIDIAN_ARC,
    ArtifactTileUnwrapError,
    commit_artifact_tile_unwrap,
    compute_artifact_tile_unwrap,
)
from src.core.artifact_tile_unwrap_record import (
    ArtifactTileUnwrapRecordError,
    validate_tile_unwrap_qc,
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


def _grain(angle_rad: float, z_mm: float) -> float:
    """One grain of temper standing 0.8 mm proud of the wall, mid-strip.

    Exactly one vertex: the first ring segment past the strip's meridian, on
    the ring at 45 mm.
    """

    if 0.0 <= angle_rad - math.pi / 2.0 < 2.0 * math.pi / 48.0 and abs(z_mm - 45.0) < 0.5:
        return 0.8
    return 0.0


def test_one_steep_face_is_reported_on_the_measured_axis_and_refused_on_a_fit() -> None:
    """A rubbing records the wall's relief; the unwrap must not refuse it.

    On a 0.94 mm mesh a 0.8 mm grain makes faces that stand 30% longer than
    their unrolled shadow.  Under a fitted centre one such face would mean a
    centre in the wrong place, and the gate refuses; about the measured axis
    it is a grain, the record says how steep, and the strip still develops,
    commits, validates and exports - as a 1.4 sidecar, the first that can
    carry it.
    """

    session, vertices, faces = positioned_vessel_session(
        segments=48, rings=96, relief=_grain
    )
    selected = meridional_strip_faces(
        vertices, faces, center_angle_rad=math.pi / 2.0, width_mm=20.0
    )

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
    assert qc["distortion_max_millionths"] > 250_000, qc
    assert qc["distortion_p95_millionths"] <= 150_000, qc
    assert qc["distortion_mean_millionths"] <= 75_000, qc

    committed = commit_artifact_tile_unwrap(
        session,
        computation,
        record_id="record:unwrap:grain",
        created_at="2026-09-04T00:00:00Z",
        operator="pytest",
    )
    validate_known_records(committed.document)
    bundle = build_tile_unwrap_export(
        committed.document, "record:unwrap:grain", computation.unwrap
    )
    sidecar = validate_tile_unwrap_export_bytes(
        bundle.payload_bytes,
        bundle.obj_bytes,
        bundle.svg_bytes,
        bundle.sidecar_bytes,
        document=committed.document,
    )
    assert sidecar["schema_version"] == "1.5.0"
    assert sidecar["qc"]["record"]["distortion_max_millionths"] > 250_000

    # The same grain under the tile's fitted centre is a failed fit.  The
    # strip alone already collapses that fit, so read the gate directly on
    # the record's numbers.
    with pytest.raises(ArtifactTileUnwrapRecordError, match="max distortion"):
        validate_tile_unwrap_qc(
            sidecar["qc"]["record"],
            sidecar["geometry"],
            section_center_policy=SECTION_CENTER_FIT_PER_SECTION,
        )


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
