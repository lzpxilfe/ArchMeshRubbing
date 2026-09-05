"""능선 record: the ridges read once, kept with the document, drawn as 내선."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from src.core.artifact_crease_record import (
    CREASE_PAYLOAD_EXTENSION_KEY,
    CREASE_RECORD_TYPE,
    CREASE_VIEWS,
    ArtifactCreaseRecordError,
    CreasePayload,
    commit_crease_reading,
    compute_crease_reading,
    crease_payload_from_record,
    crease_recipe,
    validate_crease_recipe,
)
from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_record_validation import validate_known_records
from src.core.artifact_vector_extractor import (
    commit_vector_computation,
    compute_artifact_cutline,
)
from src.core.artifact_vector_record import PlanarFrame
from src.core.drawing_sheet import (
    DrawingSheetOptions,
    SheetPage,
    TitleBlock,
    compose_drawing_sheet,
    validate_drawing_sheet_bytes,
)
from src.core.drawing_style import OUTLINE_HOLE
from src.core.project_file import load_artifact_project, save_artifact_project
from synthetic_lithic import lithic_session

CREASE_ID = "record:biface-ridges"
PLAN_ID = "record:biface-plan"
SIDE_ID = "record:biface-side"
SECTION_ID = "record:biface-section"


@pytest.fixture(scope="module")
def recorded():
    session, _vertices, _faces = lithic_session(document_id="artifact:biface-record")
    computation = compute_crease_reading(session)
    session = commit_crease_reading(
        session,
        computation,
        record_id=CREASE_ID,
        created_at="2026-09-05T00:00:01Z",
        operator="tester",
    )
    return session, computation


def test_the_recipe_carries_the_thresholds_and_rebuilds_from_them() -> None:
    recipe = crease_recipe(dihedral_min_deg=30.0, min_length_mm=1.5)
    assert recipe["detection_policy"]["dihedral_min_millideg"] == 30_000
    assert recipe["detection_policy"]["min_length_um"] == 1_500
    assert recipe["views"] == list(CREASE_VIEWS)
    assert validate_crease_recipe(recipe) == recipe
    with pytest.raises(ArtifactCreaseRecordError, match="does not match"):
        validate_crease_recipe({**recipe, "visibility": "anything"})
    with pytest.raises(ArtifactCreaseRecordError, match="inclusive range"):
        crease_recipe(dihedral_min_deg=0.0)


def test_a_reading_is_a_record_that_reopens_to_the_same_bytes(recorded) -> None:
    session, computation = recorded
    record = session.document.record_index[CREASE_ID]
    assert record.type == CREASE_RECORD_TYPE
    payload = crease_payload_from_record(record)
    assert payload == computation.payload
    assert payload.chain_count >= 5
    assert payload.lines_for_view("top").polylines
    assert payload.lines_for_view("bottom").polylines == ()
    assert record.qc["view_polyline_counts"]["top"] == len(payload.lines_for_view("top").polylines)
    validate_known_records(session.document)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "biface.amr"
        save_artifact_project(path, session.document)
        reopened = load_artifact_project(path)
    again = crease_payload_from_record(reopened.record_index[CREASE_ID])
    assert again.canonical_json_bytes() == payload.canonical_json_bytes()
    assert again.sha256 == payload.sha256

    # A record whose stored lines were touched does not read back.
    thawed = record.to_dict()
    thawed["extensions"][CREASE_PAYLOAD_EXTENSION_KEY]["payload"]["chain_count"] += 1
    with pytest.raises(ArtifactCreaseRecordError, match="SHA-256"):
        crease_payload_from_record(type(record).from_dict(thawed))


def test_a_payload_accounts_for_every_view_once() -> None:
    good = CreasePayload.from_dict(
        {
            "chain_count": 1,
            "max_dihedral_millideg": 45_000,
            "schema_version": "1.0.0",
            "total_length_um": 5_000,
            "views": [
                {"view": view, "polylines": [[[0, 0], [5_000, 0]]] if view == "top" else []}
                for view in CREASE_VIEWS
            ],
        }
    )
    assert good.lines_for_view("top").point_count == 2
    with pytest.raises(ArtifactCreaseRecordError, match="exactly once"):
        CreasePayload.from_dict({**good.to_dict(), "views": good.to_dict()["views"][:5]})
    with pytest.raises(ArtifactCreaseRecordError, match="repeats a point"):
        CreasePayload.from_dict(
            {
                **good.to_dict(),
                "views": [
                    {"view": view, "polylines": [[[0, 0], [0, 0]]] if view == "top" else []}
                    for view in CREASE_VIEWS
                ],
            }
        )


def test_the_ridges_are_drawn_on_the_plan_as_inner_lines_and_nowhere_else(recorded) -> None:
    session, computation = recorded
    plan = compute_artifact_outline(session, "top", precision_grid_mm=0.01)
    session = commit_vector_computation(
        session, plan, record_id=PLAN_ID, created_at="2026-09-05T00:00:02Z", operator="tester"
    )
    side = compute_artifact_outline(session, "front", precision_grid_mm=0.01)
    session = commit_vector_computation(
        session, side, record_id=SIDE_ID, created_at="2026-09-05T00:00:03Z", operator="tester"
    )
    section = compute_artifact_cutline(
        session,
        PlanarFrame(
            origin_world_mm=(0.0, 0.13, 0.0),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 0.0, 1.0),
            normal_world=(0.0, -1.0, 0.0),
        ),
    )
    session = commit_vector_computation(
        session,
        section,
        record_id=SECTION_ID,
        created_at="2026-09-05T00:00:04Z",
        operator="tester",
    )
    document = session.document

    def options(**overrides) -> DrawingSheetOptions:
        settings = {
            "title_block": TitleBlock(artifact_label="합성 뗀석기", rows=(("작성", "tester"),)),
            "page": SheetPage(size="A4", orientation="portrait"),
            "scale_denominator": 1.0,
        }
        settings.update(overrides)
        return DrawingSheetOptions(**settings)

    plain = compose_drawing_sheet(document, [PLAN_ID, SIDE_ID, SECTION_ID], options=options())
    explicit = compose_drawing_sheet(
        document, [PLAN_ID, SIDE_ID, SECTION_ID], options=options(crease_records=())
    )
    assert plain.svg_bytes == explicit.svg_bytes
    assert b"crease:" not in plain.svg_bytes
    assert "crease" not in json.loads(plain.sidecar_bytes)

    drawn = compose_drawing_sheet(
        document,
        [PLAN_ID, SIDE_ID, SECTION_ID],
        options=options(crease_records=(CREASE_ID,)),
    )
    validate_drawing_sheet_bytes(drawn.svg_bytes, drawn.sidecar_bytes)
    svg = drawn.svg_bytes.decode("utf-8")
    top_lines = computation.payload.lines_for_view("top").polylines
    front_lines = computation.payload.lines_for_view("front").polylines
    assert svg.count(f"crease:{CREASE_ID}:top:") == len(top_lines)
    assert svg.count(f"crease:{CREASE_ID}:front:") == len(front_lines)
    assert "crease:" + CREASE_ID + ":left" not in svg
    sidecar = json.loads(drawn.sidecar_bytes)
    block = sidecar["crease"]
    assert block["records"] == [
        {
            "chain_count": computation.payload.chain_count,
            "payload_sha256": computation.payload.sha256,
            "recipe_hash": document.record_index[CREASE_ID].recipe_hash,
            "record_id": CREASE_ID,
        }
    ]
    figures = {entry["figure_record_id"]: entry for entry in block["drawn"]}
    assert figures[PLAN_ID]["view"] == "top"
    assert figures[PLAN_ID]["line_kind"] == OUTLINE_HOLE
    assert int(figures[PLAN_ID]["polyline_count"]) == len(top_lines)
    assert figures[SIDE_ID]["view"] == "front"
    # A section is a cut, not a surface: it gets no ridge.
    assert SECTION_ID not in figures

    with pytest.raises(Exception, match="not a crease reading"):
        compose_drawing_sheet(document, [PLAN_ID], options=options(crease_records=(PLAN_ID,)))
