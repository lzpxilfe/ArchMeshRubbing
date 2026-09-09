"""The readings a drawing needs can be taken where the artifact is open.

The plate could always draw a corner, a groove, a ridge, a far silhouette
and a relief's shade; nothing in the application could make one.  These
tests hold the new layer to the two things that matter: a reading is the
core's reading, recorded under the archaeologist's name, and a refusal
says which reading and why.
"""

from __future__ import annotations

import pytest

from src.application.artifact_readings import (
    CREASE,
    FAR_SILHOUETTE,
    PROFILE_BREAK,
    PROFILE_GROOVE,
    READING_KINDS,
    READING_LABELS,
    RELIEF_SHADE,
    ArtifactReadingError,
    take_reading,
)
from src.core.artifact_document import RecordFreshness, RecordLifecycleStatus
from src.core.artifact_record_validation import validate_known_records
from synthetic_vessel import positioned_vessel_session

STAMP = "2026-09-09T00:00:00Z"
OPERATOR = "황진서"


def _wall(angle_rad: float, z_mm: float) -> float:
    """A wall with something for each reading: a shallow lotus petal to
    shade, and a groove cut right round below it."""

    import math  # noqa: PLC0415

    petal = (
        0.6 * max(0.0, math.cos(6.0 * angle_rad)) * math.sin(math.pi * (z_mm - 18.0) / 22.0)
        if 18.0 <= z_mm <= 40.0
        else 0.0
    )
    return petal + (-0.5 if 50.0 <= z_mm <= 53.0 else 0.0)


@pytest.fixture(scope="module")
def vessel():
    session, _vertices, _faces = positioned_vessel_session(
        segments=96, rings=60, relief=_wall, document_id="artifact:readings"
    )
    return session


@pytest.mark.parametrize(
    ("kind", "options"),
    [
        (PROFILE_BREAK, {"angle_min_deg": 8}),
        (PROFILE_GROOVE, {"minimum_depth_um": 100}),
        (FAR_SILHOUETTE, {"view": "front", "precision_grid_mm": 0.5}),
        (RELIEF_SHADE, {"view": "front"}),
        (CREASE, {"dihedral_min_deg": 25.0, "min_length_mm": 2.0}),
    ],
)
def test_a_reading_is_taken_and_recorded_under_the_archaeologist_s_name(vessel, kind, options) -> None:
    record_id = f"record:{kind}"
    outcome = take_reading(
        vessel, kind, record_id=record_id, created_at=STAMP, operator=OPERATOR, options=options
    )
    record = outcome.session.document.record_index[record_id]
    assert record.operator == OPERATOR, "the drawing prints who recorded it"
    assert record.created_at == STAMP
    assert record.lifecycle_status is RecordLifecycleStatus.READY
    freshness = outcome.session.document.record_freshnesses()[record_id]
    assert freshness is RecordFreshness.FRESH, "a reading is fresh for the Align it was taken under"
    validate_known_records(outcome.session.document)
    assert READING_LABELS[kind][0] in outcome.summary
    if kind == RELIEF_SHADE:
        assert outcome.raster is not None, "the shade's pixels travel beside its receipt"
    else:
        assert outcome.raster is None
    # The reading did not disturb the document it was taken from.
    assert record_id not in vessel.document.record_index


def test_every_reading_the_plate_can_draw_has_a_name_and_a_line_about_it() -> None:
    assert set(READING_KINDS) == set(READING_LABELS)
    assert all(len(READING_LABELS[kind]) == 2 and all(READING_LABELS[kind]) for kind in READING_KINDS)


def test_a_refusal_names_the_reading_and_the_reason(vessel) -> None:
    with pytest.raises(ArtifactReadingError, match="모르는 판독"):
        take_reading(vessel, "colour", record_id="record:x", created_at=STAMP, operator=OPERATOR)
    with pytest.raises(ArtifactReadingError, match="record id가 필요"):
        take_reading(vessel, PROFILE_BREAK, record_id="  ", created_at=STAMP, operator=OPERATOR)
    with pytest.raises(ArtifactReadingError, match="세션이 없습니다"):
        take_reading(object(), PROFILE_BREAK, record_id="record:x", created_at=STAMP, operator=OPERATOR)  # type: ignore[arg-type]
    with pytest.raises(ArtifactReadingError, match=f"{READING_LABELS[PROFILE_BREAK][0]}의 설정"):
        take_reading(
            vessel, PROFILE_BREAK, record_id="record:x", created_at=STAMP, operator=OPERATOR,
            options={"nonsense": 1},
        )
    taken = take_reading(
        vessel, PROFILE_BREAK, record_id="record:once", created_at=STAMP, operator=OPERATOR,
        options={"angle_min_deg": 8},
    )
    with pytest.raises(ArtifactReadingError, match="이미 이 문서에 있는"):
        take_reading(
            taken.session, PROFILE_BREAK, record_id="record:once", created_at=STAMP, operator=OPERATOR
        )


def test_a_reading_that_finds_nothing_is_refused_rather_than_recorded_empty(vessel) -> None:
    """An empty reading is not a finding: asked for ridges sharper than the
    wall has, the reading says so and records nothing."""

    with pytest.raises(ArtifactReadingError, match=READING_LABELS[CREASE][0]):
        take_reading(
            vessel, CREASE, record_id="record:none", created_at=STAMP, operator=OPERATOR,
            options={"dihedral_min_deg": 150.0, "min_length_mm": 80.0},
        )
    assert "record:none" not in vessel.document.record_index
