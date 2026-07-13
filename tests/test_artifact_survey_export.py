from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import src.application.artifact_survey_exports as survey_exports_app
import src.core.artifact_survey_export as survey_export
from src.application.artifact_exports import (
    ArtifactExportError,
    ArtifactExportState,
    ExportResourceLimitError,
    StaleExportOperationError,
)
from src.application.artifact_survey_exports import ArtifactSurveyExportController
from src.application.artifact_workbench import ArtifactWorkbench
from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_rubbing_extractor import (
    commit_artifact_rubbing,
    compute_artifact_rubbing,
)
from src.core.artifact_session import ArtifactSession
from src.core.artifact_survey_export import (
    ArtifactSurveyExportError,
    SURVEY_CUTLINE_VIEWS,
    SURVEY_EXPORT_MANIFEST_NAME,
    SURVEY_SIX_VIEWS,
    SurveyExportSelection,
    discard_prepared_survey_package,
    prepare_staged_survey_publication,
    publish_prepared_survey_package,
    stage_survey_export_package,
    validate_survey_export_package,
)
from src.core.artifact_vector_extractor import (
    commit_vector_computation,
    compute_artifact_cutline,
)
from src.core.artifact_verification import build_artifact_verification_report
from src.core.artifact_vector_record import PlanarFrame
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


STAMP = "2026-07-14T00:00:00Z"


@pytest.fixture(autouse=True)
def _confirmed_directory_fsync():
    with patch.object(survey_export, "fsync_export_directory", return_value=True):
        yield
    with survey_export._STAGING_LOCK:
        survey_export._STAGING_OWNERS.clear()
        survey_export._PREPARED_PUBLICATIONS.clear()


def _mesh() -> MeshData:
    return MeshData(
        vertices=np.asarray(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        ),
        faces=np.asarray(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=np.int32,
        ),
        unit="mm",
        filepath=Path("/source/survey.ply"),
        source_identity=SourceFingerprint(
            sha256="b" * 64,
            size_bytes=456,
            mtime_ns=1,
            original_name="survey.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )


def _cutline_frame(view: str) -> PlanarFrame:
    if view == "top":
        return PlanarFrame(
            origin_world_mm=(0.0, 0.0, 0.0),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 1.0, 0.0),
            normal_world=(0.0, 0.0, 1.0),
        )
    if view == "front":
        return PlanarFrame(
            origin_world_mm=(0.0, 0.0, 0.0),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 0.0, 1.0),
            normal_world=(0.0, -1.0, 0.0),
        )
    if view == "right":
        return PlanarFrame(
            origin_world_mm=(0.0, 0.0, 0.0),
            u_axis_world=(0.0, 1.0, 0.0),
            v_axis_world=(0.0, 0.0, 1.0),
            normal_world=(1.0, 0.0, 0.0),
        )
    raise AssertionError(view)


def _aligned_session() -> ArtifactSession:
    return ArtifactSession.create_from_source(
        _mesh(),
        resolved_source_path="/source/survey.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="survey-test",
        operator="pytest",
        created_at=STAMP,
        document_id="artifact:survey-export",
        metadata_revision_id="metadata:survey-export",
        align_revision_id="align:survey-initial",
    ).commit_preview(
        translation_mm=(0.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:survey-confirmed",
    )


def _completed_session() -> tuple[ArtifactSession, SurveyExportSelection]:
    session = _aligned_session()

    cutline_ids: list[str] = []
    for view in SURVEY_CUTLINE_VIEWS:
        record_id = f"record:cutline:{view}:survey"
        session = commit_vector_computation(
            session,
            compute_artifact_cutline(session, _cutline_frame(view)),
            record_id=record_id,
            created_at=STAMP,
            operator="pytest",
        )
        cutline_ids.append(record_id)

    outline_ids: list[str] = []
    for view in SURVEY_SIX_VIEWS:
        record_id = f"record:outline:{view}:survey"
        session = commit_vector_computation(
            session,
            compute_artifact_outline(session, view, precision_grid_mm=0.01),
            record_id=record_id,
            created_at=STAMP,
            operator="pytest",
            depends_on_record_ids=tuple(cutline_ids),
        )
        outline_ids.append(record_id)

    rubbing_ids: list[str] = []
    for view in SURVEY_SIX_VIEWS:
        record_id = f"record:rubbing:{view}:survey"
        computation = compute_artifact_rubbing(
            session,
            view,
            pixels_per_mm=2,
            margin_um=0,
            reference_radius_um=500,
            depth_quantization_um=10,
            black_point_um=100,
            ink_strength_percent=100,
            relief_polarity="bidirectional",
        )
        session = commit_artifact_rubbing(
            session,
            computation,
            record_id=record_id,
            created_at=STAMP,
            operator="pytest",
            depends_on_record_ids=tuple(outline_ids),
        )
        rubbing_ids.append(record_id)
    return session, SurveyExportSelection(
        cutline_record_ids=tuple(cutline_ids),
        outline_record_ids=tuple(outline_ids),
        rubbing_record_ids=tuple(rubbing_ids),
    )


def _publish(
    destination: Path,
    session: ArtifactSession,
    selection: SurveyExportSelection,
) -> Path:
    staging = stage_survey_export_package(destination, session, selection)
    prepared = prepare_staged_survey_publication(
        staging,
        destination,
        document=session.document,
    )
    return publish_prepared_survey_package(prepared)


def test_complete_survey_package_is_atomic_relocatable_and_deterministic(
    tmp_path: Path,
) -> None:
    session, selection = _completed_session()
    first = _publish(tmp_path / "first.amr-survey", session, selection)
    first_bundle = validate_survey_export_package(
        first,
        document=session.document,
    )
    assert first_bundle.artifact_count == 15
    assert (first_bundle.vector_count, first_bundle.rubbing_count) == (9, 6)
    assert len(first_bundle.manifest_sha256) == 64
    assert len(first_bundle.artifact_set_sha256) == 64
    jsonschema = importlib.import_module("jsonschema")
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "schemas/survey_export-1.0.0.schema.json")
        .read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    assert list(validator.iter_errors(first_bundle.manifest)) == []

    relocated = tmp_path / "한글 이동.amr-survey"
    first.rename(relocated)
    offline = validate_survey_export_package(relocated)
    assert offline.manifest_sha256 == first_bundle.manifest_sha256
    assert offline.artifact_set_sha256 == first_bundle.artifact_set_sha256
    report = build_artifact_verification_report(relocated)
    assert report["ok"] is True
    assert report["artifact_kind"] == "survey_export"
    assert report["evidence"]["artifact_count"] == 15
    assert report["evidence"]["artifact_set_sha256"] == (
        first_bundle.artifact_set_sha256
    )
    report_schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schemas/offline_verification_report-1.0.0.schema.json"
        ).read_text(encoding="utf-8")
    )
    report_validator = jsonschema.Draft202012Validator(report_schema)
    assert list(report_validator.iter_errors(report)) == []

    second = _publish(tmp_path / "second.amr-survey", session, selection)
    repeated = validate_survey_export_package(second)
    assert repeated.manifest_bytes == offline.manifest_bytes


def test_manifest_binds_each_child_and_rejects_tampering(tmp_path: Path) -> None:
    session, selection = _completed_session()
    destination = _publish(tmp_path / "survey.amr-survey", session, selection)
    manifest_path = destination / SURVEY_EXPORT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifacts"][0]["primary_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactSurveyExportError, match="does not exactly describe"):
        validate_survey_export_package(destination)


def test_child_bytes_are_revalidated_before_aggregate_hash(tmp_path: Path) -> None:
    session, selection = _completed_session()
    destination = _publish(tmp_path / "child-tamper.amr-survey", session, selection)
    svg = destination / "cutline-top.amr-vector" / "artifact.svg"
    svg.write_bytes(svg.read_bytes() + b" ")

    with pytest.raises(ArtifactSurveyExportError):
        validate_survey_export_package(destination)


def test_incomplete_or_wrong_view_selection_has_no_filesystem_effect(
    tmp_path: Path,
) -> None:
    session, selection = _completed_session()
    with pytest.raises(ArtifactSurveyExportError, match="exactly 3"):
        SurveyExportSelection(
            cutline_record_ids=selection.cutline_record_ids[:2],
            outline_record_ids=selection.outline_record_ids,
            rubbing_record_ids=selection.rubbing_record_ids,
        )

    swapped = SurveyExportSelection(
        cutline_record_ids=(
            selection.cutline_record_ids[1],
            selection.cutline_record_ids[0],
            selection.cutline_record_ids[2],
        ),
        outline_record_ids=selection.outline_record_ids,
        rubbing_record_ids=selection.rubbing_record_ids,
    )
    destination = tmp_path / "wrong.amr-survey"
    with pytest.raises(ArtifactSurveyExportError, match="canonical top axes"):
        stage_survey_export_package(destination, session, swapped)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_existing_destination_and_post_prepare_mutation_fail_closed(
    tmp_path: Path,
) -> None:
    session, selection = _completed_session()
    existing = tmp_path / "existing.amr-survey"
    existing.mkdir()
    marker = existing / "owner.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ArtifactSurveyExportError, match="already exists"):
        stage_survey_export_package(existing, session, selection)
    assert marker.read_text(encoding="utf-8") == "keep"

    destination = tmp_path / "prepared.amr-survey"
    staging = stage_survey_export_package(destination, session, selection)
    prepared = prepare_staged_survey_publication(
        staging,
        destination,
        document=session.document,
    )
    manifest = staging / SURVEY_EXPORT_MANIFEST_NAME
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ArtifactSurveyExportError, match="changed after preparation"):
        publish_prepared_survey_package(prepared)
    assert not destination.exists()
    assert discard_prepared_survey_package(prepared)


def test_destination_race_preserves_winner_and_discards_owned_tree(
    tmp_path: Path,
) -> None:
    session, selection = _completed_session()
    destination = tmp_path / "raced.amr-survey"
    staging = stage_survey_export_package(destination, session, selection)
    prepared = prepare_staged_survey_publication(
        staging,
        destination,
        document=session.document,
    )
    destination.mkdir()
    sentinel = destination / "winner.txt"
    sentinel.write_text("other process", encoding="utf-8")

    with pytest.raises(ArtifactSurveyExportError, match="already exists"):
        publish_prepared_survey_package(prepared)

    assert sentinel.read_text(encoding="utf-8") == "other process"
    assert discard_prepared_survey_package(prepared)
    assert not staging.exists()


def test_cancellation_discards_hidden_tree_and_never_publishes(tmp_path: Path) -> None:
    session, selection = _completed_session()
    destination = tmp_path / "cancelled.amr-survey"
    calls = 0

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 4

    with pytest.raises(RuntimeError, match="cancelled"):
        stage_survey_export_package(
            destination,
            session,
            selection,
            cancellation_probe=cancelled,
        )
    assert not destination.exists()
    assert not any(path.name.startswith(".amrs-") for path in tmp_path.iterdir())


def test_staging_root_replacement_is_preserved_as_foreign(tmp_path: Path) -> None:
    session, selection = _completed_session()
    destination = tmp_path / "identity.amr-survey"
    staging = stage_survey_export_package(destination, session, selection)
    moved = tmp_path / "owned-moved"
    staging.rename(moved)
    staging.mkdir()
    foreign = staging / "foreign.txt"
    foreign.write_text("keep", encoding="utf-8")
    with pytest.raises(ArtifactSurveyExportError, match="identity changed"):
        prepare_staged_survey_publication(
            staging,
            destination,
            document=session.document,
        )
    assert foreign.read_text(encoding="utf-8") == "keep"


def test_controller_stages_then_publishes_all_fifteen_under_final_authority(
    tmp_path: Path,
) -> None:
    session, selection = _completed_session()
    workbench = ArtifactWorkbench(session=session)
    controller = ArtifactSurveyExportController(
        workbench,
        id_factory=lambda prefix: f"{prefix}:test",
    )
    destination = tmp_path / "complete.amr-survey"

    item = controller.begin(destination)
    assert item.selection == selection
    assert item.expected_records == tuple(
        session.document.record_index[record_id]
        for record_id in selection.record_ids
    )

    result = controller.execute(item)
    assert not destination.exists()
    assert result.staging_directory.is_dir()
    assert controller.summary(item).state is ArtifactExportState.STAGED
    staged = validate_survey_export_package(
        result.staging_directory,
        document=session.document,
    )
    assert staged.artifact_count == 15

    publication = controller.publish_result(item, result)
    assert publication.destination == destination
    assert publication.record_ids == selection.record_ids
    assert publication.document_sha256 == session.document.canonical_sha256
    assert publication.durability_confirmed is True
    assert not result.staging_directory.exists()
    validate_survey_export_package(destination, document=session.document)
    assert controller.summary(item).state is ArtifactExportState.COMPLETED


def test_controller_begin_rejects_incomplete_workflow_without_reservation(
    tmp_path: Path,
) -> None:
    controller = ArtifactSurveyExportController(
        ArtifactWorkbench(session=_aligned_session())
    )
    destination = tmp_path / "incomplete.amr-survey"

    with pytest.raises(ArtifactExportError, match="requires dependency-valid 3/6/6"):
        controller.begin(destination)

    assert controller.active_summaries == ()
    assert not destination.exists()


def test_controller_requires_exact_result_capability(tmp_path: Path) -> None:
    session, _selection = _completed_session()
    controller = ArtifactSurveyExportController(ArtifactWorkbench(session=session))
    destination = tmp_path / "exact.amr-survey"
    item = controller.begin(destination)
    result = controller.execute(item)
    forged = replace(result)

    with pytest.raises(ArtifactExportError, match="exact survey result"):
        controller.publish_result(item, forged)
    with pytest.raises(ArtifactExportError, match="exact survey result"):
        controller.discard_result(item, forged)

    assert result.staging_directory.is_dir()
    assert not destination.exists()
    controller.publish_result(item, result)
    assert destination.is_dir()


def test_controller_align_change_revokes_and_cleans_staging(tmp_path: Path) -> None:
    session, _selection = _completed_session()
    workbench = ArtifactWorkbench(session=session)
    controller = ArtifactSurveyExportController(workbench)
    destination = tmp_path / "stale.amr-survey"
    item = controller.begin(destination)
    result = controller.execute(item)

    transition = workbench.prepare_align_commit(
        translation_mm=(1.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:survey-changed",
    )
    assert transition is not None
    activation = workbench.activate_projection(transition)
    workbench.finalize_projection(activation)

    with pytest.raises(StaleExportOperationError, match="projection"):
        controller.publish_result(item, result)

    assert not result.staging_directory.exists()
    assert not destination.exists()
    assert controller.summary(item).state is ArtifactExportState.STALE


def test_controller_budget_failure_and_staged_cancel_leave_no_output(
    tmp_path: Path,
) -> None:
    session, _selection = _completed_session()
    limited = ArtifactSurveyExportController(
        ArtifactWorkbench(session=session),
        rubbing_memory_budget_bytes=1,
    )
    rejected = tmp_path / "too-large.amr-survey"
    rejected_item = limited.begin(rejected)

    with pytest.raises(ExportResourceLimitError, match="exceeds"):
        limited.execute(rejected_item)

    assert not rejected.exists()
    assert list(tmp_path.iterdir()) == []
    assert limited.summary(rejected_item).state is ArtifactExportState.FAILED

    controller = ArtifactSurveyExportController(ArtifactWorkbench(session=session))
    cancelled = tmp_path / "cancelled-controller.amr-survey"
    cancelled_item = controller.begin(cancelled)
    result = controller.execute(cancelled_item)
    summary = controller.cancel(cancelled_item, reason="user cancelled")

    assert summary.state is ArtifactExportState.CANCELLED
    assert not result.staging_directory.exists()
    assert not cancelled.exists()
    assert list(tmp_path.iterdir()) == []


def test_controller_reports_post_rename_durability_uncertainty_as_completed(
    tmp_path: Path,
) -> None:
    session, _selection = _completed_session()
    controller = ArtifactSurveyExportController(ArtifactWorkbench(session=session))
    destination = tmp_path / "uncertain.amr-survey"
    item = controller.begin(destination)
    result = controller.execute(item)
    original_publish = survey_exports_app.publish_prepared_survey_package

    def publish_then_report_uncertain(prepared):
        original_publish(prepared)
        raise ArtifactSurveyExportError(
            "directory fsync failed after rename",
            committed=True,
        )

    with patch.object(
        survey_exports_app,
        "publish_prepared_survey_package",
        side_effect=publish_then_report_uncertain,
    ):
        publication = controller.publish_result(item, result)

    assert destination.is_dir()
    assert not result.staging_directory.exists()
    assert publication.durability_confirmed is False
    assert publication.warning_message is not None
    assert "fsync" in publication.warning_message
    assert controller.summary(item).state is ArtifactExportState.COMPLETED
