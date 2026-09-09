"""The readings a drawing needs can be taken where the artifact is open.

The plate could always draw a corner, a groove, a ridge, a far silhouette,
a relief's shade, a pattern's lines and the paint cut out of the colour;
nothing in the application could make one.  These
tests hold the new layer to the two things that matter: a reading is the
core's reading, recorded under the archaeologist's name, and a refusal
says which reading and why.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.application.artifact_readings import (
    CREASE,
    FAR_SILHOUETTE,
    PAINT_CUTOUT,
    PROFILE_BREAK,
    PROFILE_GROOVE,
    READING_KINDS,
    READING_LABELS,
    RELIEF_SHADE,
    TEXTURE_LINES,
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


@pytest.fixture(scope="module")
def textured(tmp_path_factory):
    """A vessel with grooves cut round it, plus the OBJ and normal map a
    pattern reading needs.  These two readings are the only ones that read
    the scanner's images rather than the mesh alone."""

    from test_texture_relief import _write_normal_map, _write_textured_obj  # noqa: PLC0415

    session, vertices, faces = positioned_vessel_session(
        segments=96, rings=40, document_id="artifact:readings-texture"
    )
    directory = tmp_path_factory.mktemp("texture")
    obj_path = directory / "vessel.obj"
    map_path = directory / "vessel_nor.png"
    _write_textured_obj(obj_path, vertices, np.asarray(faces, dtype=np.int64))
    _write_normal_map(map_path)
    return session, str(obj_path), str(map_path)


def test_a_pattern_can_be_read_where_the_artifact_is_open(textured) -> None:
    """Every plate that carries a pattern was drawn by a script, because the
    application could not make this record.  It can now."""

    session, obj_path, map_path = textured
    outcome = take_reading(
        session,
        TEXTURE_LINES,
        record_id="record:pattern",
        created_at=STAMP,
        operator=OPERATOR,
        options={"view": "front", "atlas_path": obj_path, "normal_map_path": map_path},
    )
    record = outcome.session.document.record_index["record:pattern"]
    assert record.type == "measurement.texture_lines.v1"
    assert record.operator == OPERATOR
    validate_known_records(outcome.session.document)
    assert int(outcome.qc["line_count"]) > 0
    assert READING_LABELS[TEXTURE_LINES][0] in outcome.summary
    # The recipe carries the files' own hashes, so the reading can be
    # re-run and checked against the images it was taken from.
    relief = record.recipe["texture_relief"]
    assert len(relief["atlas"]["sha256"]) == 64
    assert len(relief["normal_map"]["sha256"]) == 64


def test_the_files_a_texture_reading_needs_are_named_when_they_are_missing(textured) -> None:
    """"It failed" is no help when the answer is that a file was not given."""

    session, obj_path, map_path = textured
    with pytest.raises(ArtifactReadingError, match="OBJ 파일이 필요"):
        take_reading(
            session, TEXTURE_LINES, record_id="record:x", created_at=STAMP, operator=OPERATOR,
            options={"view": "front", "normal_map_path": map_path},
        )
    with pytest.raises(ArtifactReadingError, match="법선 지도나 색 지도"):
        take_reading(
            session, TEXTURE_LINES, record_id="record:x", created_at=STAMP, operator=OPERATOR,
            options={"view": "front", "atlas_path": obj_path},
        )
    with pytest.raises(ArtifactReadingError, match="색 지도가 필요"):
        take_reading(
            session, PAINT_CUTOUT, record_id="record:x", created_at=STAMP, operator=OPERATOR,
            options={"view": "front", "atlas_path": obj_path, "normal_map_path": map_path},
        )
    with pytest.raises(ArtifactReadingError, match="텍스처 좌표를 읽지 못했습니다"):
        take_reading(
            session, TEXTURE_LINES, record_id="record:x", created_at=STAMP, operator=OPERATOR,
            options={
                "view": "front",
                "atlas_path": f"{obj_path}.missing",
                "normal_map_path": map_path,
            },
        )
    with pytest.raises(ArtifactReadingError, match="지도 이미지를 읽지 못했습니다"):
        take_reading(
            session, TEXTURE_LINES, record_id="record:x", created_at=STAMP, operator=OPERATOR,
            options={
                "view": "front",
                "atlas_path": obj_path,
                "normal_map_path": f"{map_path}.missing",
            },
        )


def test_the_paint_can_be_cut_out_where_the_artifact_is_open(tmp_path) -> None:
    """The painted plates - the gold flower, the strawberry band - were all
    made by a script for want of this one call."""

    from PIL import Image  # noqa: PLC0415

    from test_texture_relief import MAP_SIDE, _write_textured_obj  # noqa: PLC0415
    from synthetic_vessel import HEIGHT_MM  # noqa: PLC0415

    session, vertices, faces = positioned_vessel_session(
        segments=96, rings=40, document_id="artifact:readings-paint"
    )
    obj_path = tmp_path / "vessel.obj"
    _write_textured_obj(obj_path, vertices, np.asarray(faces, dtype=np.int64))
    # A gold band round the wall, on an otherwise unpainted body.
    rgb = np.full((MAP_SIDE, MAP_SIDE, 3), 245, dtype=np.uint8)
    heights = HEIGHT_MM * (1.0 - (np.arange(MAP_SIDE) + 0.5) / MAP_SIDE)
    rgb[(heights >= 40.0) & (heights <= 44.0)] = np.array([200, 160, 60], dtype=np.uint8)
    map_path = tmp_path / "vessel_bc.png"
    Image.fromarray(rgb, mode="RGB").save(map_path)

    outcome = take_reading(
        session,
        PAINT_CUTOUT,
        record_id="record:cutout",
        created_at=STAMP,
        operator=OPERATOR,
        options={
            "view": "front",
            "atlas_path": str(obj_path),
            "colour_map_path": str(map_path),
        },
    )
    record = outcome.session.document.record_index["record:cutout"]
    assert record.type == "measurement.paint_cutout.v1"
    validate_known_records(outcome.session.document)
    assert int(outcome.qc["texture_paint_painted_pixel_count"]) > 0
    assert READING_LABELS[PAINT_CUTOUT][0] in outcome.summary
    # The pixels travel beside the receipt, the way the shade's do, so the
    # plate can paste them without reading the image again.
    assert outcome.raster is not None


def test_a_pattern_read_from_colour_traces_the_painted_lines(tmp_path) -> None:
    """One reading, two maps: the normal map gives the incised line, the
    colour map the painted one.  Which map decides what is read."""

    from PIL import Image  # noqa: PLC0415

    from src.core.artifact_texture_lines import (  # noqa: PLC0415
        TEXTURE_LINES_DOMAIN_AXIS,
        TEXTURE_LINES_RIDGE_RULE,
    )
    from test_texture_relief import MAP_SIDE, _write_textured_obj  # noqa: PLC0415
    from synthetic_vessel import HEIGHT_MM  # noqa: PLC0415

    session, vertices, faces = positioned_vessel_session(
        segments=96, rings=40, document_id="artifact:readings-painted-lines"
    )
    obj_path = tmp_path / "vessel.obj"
    _write_textured_obj(obj_path, vertices, np.asarray(faces, dtype=np.int64))
    rgb = np.full((MAP_SIDE, MAP_SIDE, 3), 245, dtype=np.uint8)
    heights = HEIGHT_MM * (1.0 - (np.arange(MAP_SIDE) + 0.5) / MAP_SIDE)
    rgb[(heights >= 41.0) & (heights <= 43.0)] = np.array([200, 160, 60], dtype=np.uint8)
    map_path = tmp_path / "vessel_bc.png"
    Image.fromarray(rgb, mode="RGB").save(map_path)

    outcome = take_reading(
        session,
        TEXTURE_LINES,
        record_id="record:painted-lines",
        created_at=STAMP,
        operator=OPERATOR,
        options={
            "view": "front",
            "atlas_path": str(obj_path),
            "colour_map_path": str(map_path),
            # Painted lines are read as the paper reads a stroke, unrolled
            # about the axis - the settings the core's own painted-line
            # reading uses.  This layer passes them through untouched.
            "rule": TEXTURE_LINES_RIDGE_RULE,
            "domain": TEXTURE_LINES_DOMAIN_AXIS,
            "depth_mm": 0.2,
            "orientation_mm": 0.0,
            "straightness_min_percent": 0,
            "min_length_mm": 5.0,
            "band_mm": 0.8,
        },
    )
    record = outcome.session.document.record_index["record:painted-lines"]
    assert record.type == "measurement.texture_lines.v1"
    assert int(outcome.qc["line_count"]) > 0
    # It was read from colour, and the recipe says so: a paint block stands
    # where the relief block would be, naming the image's own hash.
    assert "texture_relief" not in record.recipe
    assert len(record.recipe["texture_paint"]["colour_map"]["sha256"]) == 64
