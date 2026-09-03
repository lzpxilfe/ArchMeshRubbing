from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import pytest

from src.core.artifact_condition_annotation import (
    ArtifactConditionAnnotationError,
    CONDITION_KINDS,
    CONDITION_PAYLOAD_EXTENSION_KEY,
    CONDITION_RECORD_TYPE,
    CONDITION_VIEWS,
    ConditionAnnotationPayload,
    ConditionViewBoundary,
    commit_condition_annotation,
    compute_condition_annotation,
    condition_payload_from_record,
    condition_selection,
    face_indices_from_ranges,
    face_ranges_from_indices,
    project_condition_region,
    validate_condition_selection,
)
from src.core.artifact_record_validation import (
    ArtifactKnownRecordError,
    validate_known_records,
)
from src.core.artifact_session import ArtifactSession
from src.core.canonical_json import canonical_json_bytes
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.project_file import load_artifact_project, save_artifact_project
from src.core.source_identity import SourceFingerprint


STAMP = "2026-09-03T00:00:00Z"
COMMITTED_AT = "2026-09-03T00:00:01Z"


def _session(half_extent_mm: float = 10.0) -> ArtifactSession:
    """A tetrahedron: four faces, none of them axis-parallel."""

    h = float(half_extent_mm)
    mesh = MeshData(
        vertices=np.array(
            [[h, h, h], [-h, -h, h], [-h, h, -h], [h, -h, -h]],
            dtype=np.float64,
        ),
        faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32),
        unit="mm",
        source_identity=SourceFingerprint(
            sha256="c" * 64,
            size_bytes=64,
            mtime_ns=1,
            original_name="condition.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    return ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/condition.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="condition-test",
        operator="tester",
        created_at=STAMP,
        document_id="artifact:condition-test",
        metadata_revision_id="metadata:condition-test",
        align_revision_id="align:condition-test",
    )


def _flat_plate_session() -> ArtifactSession:
    """Two triangles lying flat in the z = 0 plane.

    Seen from above the plate has area; seen from the front it is a line.  That
    is the honest empty projection a condition record has to survive.
    """

    mesh = MeshData(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [20.0, 12.0, 0.0],
                [0.0, 12.0, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        unit="mm",
        source_identity=SourceFingerprint(
            sha256="d" * 64,
            size_bytes=64,
            mtime_ns=1,
            original_name="plate.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    return ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/plate.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="condition-test",
        operator="tester",
        created_at=STAMP,
        document_id="artifact:plate",
        metadata_revision_id="metadata:plate",
        align_revision_id="align:plate",
    )


def _committed(condition: str = "missing", faces: object = (0, 2)) -> ArtifactSession:
    session = _session()
    computation = compute_condition_annotation(
        session,
        condition=condition,
        face_indices=faces,
    )
    return commit_condition_annotation(
        session,
        computation,
        record_id="record:condition-1",
        created_at=COMMITTED_AT,
        operator="tester",
    )


# --- the canonical face-set encoding -----------------------------------------


def test_face_ranges_round_trip_any_set_and_have_exactly_one_encoding() -> None:
    generator = np.random.default_rng(20260903)
    total = 4_096
    for _trial in range(64):
        count = int(generator.integers(1, 400))
        indices = np.unique(generator.integers(0, total, size=count))
        ranges = face_ranges_from_indices(indices, total_face_count=total)
        decoded = face_indices_from_ranges(ranges, total_face_count=total)
        assert np.array_equal(decoded, indices)
        # Order and duplicates are the caller's, never the encoding's.
        shuffled = generator.permutation(np.concatenate((indices, indices)))
        assert (
            face_ranges_from_indices(shuffled, total_face_count=total) == ranges
        )


def test_the_same_region_always_hashes_to_the_same_bytes() -> None:
    total = 1_000
    indices = [7, 8, 9, 40, 41, 900]
    first = condition_selection(
        total_face_count=total,
        face_ranges=face_ranges_from_indices(indices, total_face_count=total),
    )
    second = condition_selection(
        total_face_count=total,
        face_ranges=face_ranges_from_indices(
            list(reversed(indices)) + indices,
            total_face_count=total,
        ),
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["face_ranges"] == [[7, 10], [40, 42], [900, 901]]
    assert first["selected_face_count"] == 6


def test_adjacent_runs_are_merged_so_a_region_cannot_encode_two_ways() -> None:
    assert face_ranges_from_indices([0, 1, 2, 3], total_face_count=10) == ((0, 4),)
    with pytest.raises(ArtifactConditionAnnotationError, match="maximally merged"):
        condition_selection(total_face_count=10, face_ranges=[[0, 2], [2, 4]])


def test_a_selection_is_rejected_unless_it_is_canonical_and_in_bounds() -> None:
    with pytest.raises(ArtifactConditionAnnotationError, match="outside the geometry"):
        face_ranges_from_indices([0, 10], total_face_count=10)
    with pytest.raises(ArtifactConditionAnnotationError, match="at least one face"):
        face_ranges_from_indices([], total_face_count=10)
    with pytest.raises(ArtifactConditionAnnotationError, match="maximally merged"):
        condition_selection(total_face_count=10, face_ranges=[[4, 6], [0, 2]])
    with pytest.raises(ArtifactConditionAnnotationError, match="maximally merged"):
        condition_selection(total_face_count=10, face_ranges=[[0, 5], [3, 7]])
    with pytest.raises(ArtifactConditionAnnotationError, match="non-empty"):
        condition_selection(total_face_count=10, face_ranges=[[4, 4]])
    with pytest.raises(ArtifactConditionAnnotationError, match="at least one face"):
        condition_selection(total_face_count=10, face_ranges=[])


def test_a_selection_digest_covers_the_faces_not_the_encoding() -> None:
    selection = condition_selection(total_face_count=64, face_ranges=[[0, 4]])
    forged = dict(selection)
    forged["selection_sha256"] = "0" * 64
    with pytest.raises(ArtifactConditionAnnotationError, match="SHA-256"):
        validate_condition_selection(forged)
    miscounted = dict(selection)
    miscounted["selected_face_count"] = 3
    with pytest.raises(
        ArtifactConditionAnnotationError, match="selected_face_count"
    ):
        validate_condition_selection(miscounted)


# --- projection ---------------------------------------------------------------


def test_a_region_projects_into_every_view_that_can_see_it() -> None:
    computation = compute_condition_annotation(
        _session(),
        condition="missing",
        face_indices=[0, 2],
    )

    assert [view.view for view in computation.payload.views] == list(CONDITION_VIEWS)
    assert computation.qc["empty_views"] == []
    assert computation.qc["face_count"] == 2
    assert computation.qc["face_range_count"] == 2
    assert computation.qc["condition"] == "missing"
    for view in computation.qc["views"]:
        assert view["area_mm2"] > 0.0
        assert view["component_count"] == 1


def test_a_view_that_sees_the_region_edge_on_is_reported_not_raised() -> None:
    computation = compute_condition_annotation(
        _flat_plate_session(),
        condition="restored",
        face_indices=[0],
    )

    projected = [view.view for view in computation.payload.views]
    assert projected == ["bottom", "top"]
    assert computation.qc["empty_views"] == ["back", "front", "left", "right"]
    assert computation.qc["empty_view_reasons"] == [
        {"reason": "no_projected_area", "view": view}
        for view in ("back", "front", "left", "right")
    ]


def test_a_region_no_view_can_see_is_refused() -> None:
    # A single triangle standing in the x = 0 plane, on edge in four views and
    # a line in the other two: nothing to draw anywhere.
    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 5.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(ArtifactConditionAnnotationError, match="no usable boundary"):
        project_condition_region(
            vertices,
            faces,
            [0],
            precision_grid_mm=0.01,
        )


def test_projection_refuses_a_face_index_outside_the_geometry() -> None:
    session = _session()
    with pytest.raises(
        ArtifactConditionAnnotationError, match="outside the geometry"
    ):
        compute_condition_annotation(session, condition="worn", face_indices=[0, 4])


def test_only_the_closed_condition_vocabulary_is_accepted() -> None:
    assert CONDITION_KINDS == ("crack", "missing", "restored", "worn")
    with pytest.raises(ArtifactConditionAnnotationError, match="condition kind"):
        compute_condition_annotation(_session(), condition="chipped", face_indices=[0])


# --- the durable record -------------------------------------------------------


def test_a_committed_record_carries_the_region_and_validates() -> None:
    session = _committed()
    record = session.document.record_index["record:condition-1"]

    assert record.type == CONDITION_RECORD_TYPE
    assert record.selection_hash == record.recipe["selection"]["selection_sha256"]
    validate_known_records(session.document)

    payload = condition_payload_from_record(record)
    assert list(payload.face_indices()) == [0, 2]
    assert payload.condition == "missing"


def test_the_face_set_survives_saving_and_reopening() -> None:
    document = _committed().document

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "condition.amr"
        save_artifact_project(path, document)
        loaded = load_artifact_project(path)

    reopened = condition_payload_from_record(
        loaded.record_index["record:condition-1"]
    )
    assert list(reopened.face_indices()) == [0, 2]
    assert reopened == condition_payload_from_record(
        document.record_index["record:condition-1"]
    )
    assert loaded.canonical_sha256 == document.canonical_sha256


def test_a_tampered_region_is_refused_on_read_and_on_save() -> None:
    document = _committed().document
    record = document.record_index["record:condition-1"]
    descriptor = dict(record.extensions[CONDITION_PAYLOAD_EXTENSION_KEY])
    payload = dict(descriptor["payload"])
    selection = dict(payload["selection"])
    # Same face count, different faces: only the digest can catch this.
    selection["face_ranges"] = [[1, 2], [3, 4]]
    payload["selection"] = selection
    descriptor["payload"] = payload
    tampered = replace(
        record,
        extensions={CONDITION_PAYLOAD_EXTENSION_KEY: descriptor},
    )

    with pytest.raises(ArtifactConditionAnnotationError, match="SHA-256"):
        condition_payload_from_record(tampered)

    tampered_document = replace(document, records=(tampered,))
    with pytest.raises(ArtifactKnownRecordError, match="SHA-256"):
        validate_known_records(tampered_document)


def test_qc_cannot_claim_a_boundary_the_payload_does_not_hold() -> None:
    document = _committed().document
    record = document.record_index["record:condition-1"]
    qc = record.to_dict()["qc"]
    qc["face_count"] = 3
    forged = replace(record, qc=qc)

    with pytest.raises(ArtifactConditionAnnotationError, match="face_count"):
        condition_payload_from_record(forged)


def test_a_boundary_must_be_in_its_own_view_frame() -> None:
    payload = condition_payload_from_record(
        _committed().document.record_index["record:condition-1"]
    )
    front = payload.boundary_for_view("front")
    assert front is not None

    with pytest.raises(ArtifactConditionAnnotationError, match="canonical frame"):
        ConditionViewBoundary(view="left", outline=front.outline)


def test_a_payload_holds_at_most_one_boundary_per_view() -> None:
    payload = condition_payload_from_record(
        _committed().document.record_index["record:condition-1"]
    )
    front = payload.boundary_for_view("front")
    assert front is not None

    with pytest.raises(ArtifactConditionAnnotationError, match="one boundary per view"):
        ConditionAnnotationPayload(
            schema_version=payload.schema_version,
            condition=payload.condition,
            selection=payload.selection,
            views=(front, front),
        )


def test_every_view_is_accounted_for_with_a_boundary_or_a_reason() -> None:
    payload = condition_payload_from_record(
        _committed().document.record_index["record:condition-1"]
    )

    named = [view.view for view in payload.views] + [
        entry["view"] for entry in payload.skipped_views
    ]
    assert sorted(named) == list(CONDITION_VIEWS)

    with pytest.raises(ArtifactConditionAnnotationError, match="each of the six views"):
        ConditionAnnotationPayload(
            schema_version=payload.schema_version,
            condition=payload.condition,
            selection=payload.selection,
            views=payload.views[:2],
        )
    with pytest.raises(ArtifactConditionAnnotationError, match="skip reason"):
        ConditionAnnotationPayload(
            schema_version=payload.schema_version,
            condition=payload.condition,
            selection=payload.selection,
            views=payload.views[:5],
            skipped_views=({"reason": "because", "view": payload.views[5].view},),
        )


def test_one_unusable_view_does_not_refuse_the_whole_annotation() -> None:
    """The plate is edge-on in four views and still records two boundaries."""

    computation = compute_condition_annotation(
        _flat_plate_session(),
        condition="worn",
        face_indices=[0, 1],
    )

    assert len(computation.payload.views) == 2
    assert len(computation.payload.skipped_views) == 4


def test_condition_records_do_not_change_the_completion_gate() -> None:
    """Condition is information added to a survey, not a condition of one.

    An artifact drawn before this record type existed must not become
    incomplete the moment it exists.
    """

    from src.application.artifact_workflow_progress import (  # noqa: PLC0415
        derive_artifact_workflow_progress,
    )

    session = _session()
    computation = compute_condition_annotation(
        session,
        condition="crack",
        face_indices=[1],
    )
    annotated = commit_condition_annotation(
        session,
        computation,
        record_id="record:condition-1",
        created_at=COMMITTED_AT,
        operator="tester",
    )

    before = derive_artifact_workflow_progress(session, align_ready=True)
    after = derive_artifact_workflow_progress(annotated, align_ready=True)
    assert after == before
