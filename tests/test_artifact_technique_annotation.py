from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import pytest

from src.core.artifact_condition_annotation import CONDITION_RECORD_TYPE
from src.core.artifact_record_validation import (
    ArtifactKnownRecordError,
    validate_known_records,
)
from src.core.artifact_session import ArtifactSession
from src.core.artifact_technique_annotation import (
    ArtifactTechniqueAnnotationError,
    TECHNIQUE_KIND_LABELS_KO,
    TECHNIQUE_KINDS,
    TECHNIQUE_PAYLOAD_EXTENSION_KEY,
    TECHNIQUE_RECORD_TYPE,
    commit_technique_annotation,
    compute_technique_annotation,
    technique_payload_from_record,
    technique_recipe,
    technique_selection,
    validate_technique_recipe,
)
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.project_file import load_artifact_project, save_artifact_project
from src.core.source_identity import SourceFingerprint


STAMP = "2026-09-04T00:00:00Z"
COMMITTED_AT = "2026-09-04T00:00:01Z"
RECORD_ID = "record:technique-1"


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
            sha256="e" * 64,
            size_bytes=64,
            mtime_ns=1,
            original_name="technique.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    return ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/technique.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="technique-test",
        operator="tester",
        created_at=STAMP,
        document_id="artifact:technique-test",
        metadata_revision_id="metadata:technique-test",
        align_revision_id="align:technique-test",
    )


def _committed(technique: str = "finger_mark", faces: object = (0, 2)) -> ArtifactSession:
    session = _session()
    computation = compute_technique_annotation(
        session,
        technique=technique,
        face_indices=faces,
    )
    return commit_technique_annotation(
        session,
        computation,
        record_id=RECORD_ID,
        created_at=COMMITTED_AT,
        operator="tester",
    )


# --- the vocabulary -----------------------------------------------------------


def test_the_technique_vocabulary_is_closed_and_named() -> None:
    assert TECHNIQUE_KINDS == (
        "coil_joint",
        "finger_mark",
        "paddling",
        "water_smoothing",
        "wood_grain_smoothing",
    )
    assert set(TECHNIQUE_KIND_LABELS_KO) == set(TECHNIQUE_KINDS)
    assert TECHNIQUE_KIND_LABELS_KO["finger_mark"] == "지두흔"
    assert TECHNIQUE_KIND_LABELS_KO["coil_joint"] == "테쌓기흔"

    selection = technique_selection(total_face_count=4, face_indices=(0,))
    with pytest.raises(ArtifactTechniqueAnnotationError, match="technique must be one of"):
        technique_recipe(technique="burnishing", precision_grid_mm=0.01, selection=selection)
    # A condition kind is not a technique, whatever the two have in common.
    with pytest.raises(ArtifactTechniqueAnnotationError, match="technique must be one of"):
        technique_recipe(technique="missing", precision_grid_mm=0.01, selection=selection)


def test_the_recipe_is_the_whole_mark() -> None:
    selection = technique_selection(total_face_count=4, face_indices=(2, 0, 0))
    recipe = validate_technique_recipe(
        technique_recipe(technique="paddling", precision_grid_mm=0.05, selection=selection)
    )
    assert recipe["kind"] == "technique_annotation"
    assert recipe["technique"] == "paddling"
    assert recipe["selection"]["face_ranges"] == [[0, 1], [2, 3]]
    assert recipe["selection"]["selected_face_count"] == 2
    with pytest.raises(ArtifactTechniqueAnnotationError):
        technique_selection(total_face_count=4, face_indices=(0, 4))
    with pytest.raises(ArtifactTechniqueAnnotationError):
        technique_selection(total_face_count=4, face_indices=())


# --- the record -----------------------------------------------------------------


def test_a_committed_mark_carries_the_region_and_validates() -> None:
    session = _committed()
    record = session.document.record_index[RECORD_ID]

    assert record.type == TECHNIQUE_RECORD_TYPE
    assert record.type != CONDITION_RECORD_TYPE
    assert record.selection_hash == record.recipe["selection"]["selection_sha256"]
    validate_known_records(session.document)

    payload = technique_payload_from_record(record)
    assert list(payload.face_indices()) == [0, 2]
    assert payload.technique == "finger_mark"
    assert payload.face_count == 2
    # A tetrahedron face is seen from every direction it faces; the mark
    # projects into some views and accounts for the rest.
    assert payload.views
    qc = payload.qc_summary()
    assert qc["face_count"] == 2
    assert qc["projected_view_count"] == len(payload.views)


def test_the_face_set_survives_saving_and_reopening() -> None:
    document = _committed("coil_joint").document

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "technique.amr"
        save_artifact_project(path, document)
        loaded = load_artifact_project(path)

    reopened = technique_payload_from_record(loaded.record_index[RECORD_ID])
    assert list(reopened.face_indices()) == [0, 2]
    assert reopened.technique == "coil_joint"
    assert reopened == technique_payload_from_record(document.record_index[RECORD_ID])
    assert loaded.canonical_sha256 == document.canonical_sha256


def test_a_tampered_mark_is_refused_on_read_and_in_the_registry() -> None:
    document = _committed().document
    record = document.record_index[RECORD_ID]
    descriptor = dict(record.extensions[TECHNIQUE_PAYLOAD_EXTENSION_KEY])
    payload = dict(descriptor["payload"])
    selection = dict(payload["selection"])
    selection["face_ranges"] = [[1, 2], [3, 4]]
    payload["selection"] = selection
    descriptor["payload"] = payload
    tampered = replace(record, extensions={TECHNIQUE_PAYLOAD_EXTENSION_KEY: descriptor})

    with pytest.raises(ArtifactTechniqueAnnotationError, match="SHA-256"):
        technique_payload_from_record(tampered)

    tampered_document = replace(document, records=(tampered,))
    with pytest.raises(ArtifactKnownRecordError, match="SHA-256"):
        validate_known_records(tampered_document)


def test_a_relabelled_mark_is_refused() -> None:
    """The kind is inside the digest: a finger mark cannot be re-read as paddling."""

    document = _committed().document
    record = document.record_index[RECORD_ID]
    descriptor = dict(record.extensions[TECHNIQUE_PAYLOAD_EXTENSION_KEY])
    payload = dict(descriptor["payload"])
    payload["technique"] = "paddling"
    descriptor["payload"] = payload
    tampered = replace(record, extensions={TECHNIQUE_PAYLOAD_EXTENSION_KEY: descriptor})

    with pytest.raises(ArtifactTechniqueAnnotationError):
        technique_payload_from_record(tampered)


def test_technique_records_do_not_change_the_completion_gate() -> None:
    """A mark is information added to a survey, not a condition of one."""

    from src.application.artifact_workflow_progress import (  # noqa: PLC0415
        derive_artifact_workflow_progress,
    )

    session = _session()
    annotated = commit_technique_annotation(
        session,
        compute_technique_annotation(session, technique="paddling", face_indices=[1]),
        record_id=RECORD_ID,
        created_at=COMMITTED_AT,
        operator="tester",
    )
    before = derive_artifact_workflow_progress(session, align_ready=True)
    after = derive_artifact_workflow_progress(annotated, align_ready=True)
    assert after == before
