from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

import numpy as np
import pytest

import src.application.artifact_exports as artifact_exports
import src.core.artifact_rubbing_export as rubbing_export
import src.core.artifact_vector_export as vector_export
from src.application.artifact_exports import (
    ArtifactExportController,
    ArtifactExportError,
    ArtifactExportState,
    ExportCancelledError,
    ExportResourceLimitError,
    StaleExportOperationError,
)
from src.application.artifact_workbench import (
    ArtifactWorkbench,
    ConfirmedSourceMetadata,
    WorkflowBusyError,
)
from src.core.artifact_rubbing_export import validate_rubbing_export_package
from src.core.artifact_rubbing_extractor import (
    commit_artifact_rubbing,
    compute_artifact_rubbing,
)
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_export import (
    ArtifactVectorExportError,
    publish_prepared_vector_package,
    validate_vector_export_package,
)
from src.core.artifact_vector_extractor import (
    commit_vector_computation,
    compute_artifact_cutline,
)
from src.core.artifact_vector_record import PlanarFrame
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


STAMP = "2026-07-12T00:00:00Z"


@pytest.fixture(autouse=True)
def _confirmed_export_directory_fsync():
    with patch.object(
        vector_export,
        "_fsync_parent",
        return_value=True,
    ), patch.object(
        rubbing_export,
        "fsync_export_directory",
        return_value=True,
    ):
        yield


class SequentialIds:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, prefix: str) -> str:
        self.value += 1
        return f"{prefix}:test-{self.value}"


def _mesh(*, source_sha: str = "a" * 64, name: str = "artifact.ply") -> MeshData:
    vertices = np.asarray(
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
    )
    faces = np.asarray(
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
    )
    return MeshData(
        vertices=vertices,
        faces=faces,
        unit="mm",
        filepath=Path("/source") / name,
        source_identity=SourceFingerprint(
            sha256=source_sha,
            size_bytes=321,
            mtime_ns=1,
            original_name=name,
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )


def _aligned_session() -> ArtifactSession:
    session = ArtifactSession.create_from_source(
        _mesh(),
        resolved_source_path="/source/artifact.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="0.7-test",
        operator="pytest",
        created_at=STAMP,
        document_id="artifact:export-controller",
        metadata_revision_id="metadata:initial",
        align_revision_id="align:initial",
    )
    return session.commit_preview(
        translation_mm=(0.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:confirmed",
    )


def _frame(*, offset_z: float = 0.0) -> PlanarFrame:
    return PlanarFrame(
        origin_world_mm=(0.0, 0.0, offset_z),
        u_axis_world=(1.0, 0.0, 0.0),
        v_axis_world=(0.0, 1.0, 0.0),
        normal_world=(0.0, 0.0, 1.0),
    )


def _vector_session() -> ArtifactSession:
    session = _aligned_session()
    computation = compute_artifact_cutline(session, _frame())
    return commit_vector_computation(
        session,
        computation,
        record_id="record:vector:export",
        created_at=STAMP,
        operator="pytest",
    )


def _rubbing_session() -> ArtifactSession:
    session = _aligned_session()
    computation = compute_artifact_rubbing(
        session,
        "top",
        pixels_per_mm=1,
        margin_um=1_000,
        reference_radius_um=1_000,
        depth_quantization_um=10,
        black_point_um=250,
        ink_strength_percent=100,
        relief_polarity="bidirectional",
    )
    return commit_artifact_rubbing(
        session,
        computation,
        record_id="record:rubbing:export",
        created_at=STAMP,
        operator="pytest",
    )


def _append_unrelated_record(workbench: ArtifactWorkbench) -> ArtifactSession:
    session = workbench.snapshot.session
    assert session is not None
    computation = compute_artifact_cutline(session, _frame(offset_z=0.5))
    candidate = commit_vector_computation(
        session,
        computation,
        record_id="record:unrelated",
        created_at=STAMP,
        operator="pytest",
    )
    transition = workbench.prepare_record_commit(
        session,
        candidate,
        expected_new_record_ids=("record:unrelated",),
    )
    activation = workbench.activate_record_binding(transition)
    finalized = workbench.finalize_record_binding(activation)
    assert finalized.session is candidate
    return candidate


def _change_align(workbench: ArtifactWorkbench) -> None:
    transition = workbench.prepare_align_commit(
        translation_mm=(1.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:changed",
    )
    assert transition is not None
    activation = workbench.activate_projection(transition)
    workbench.finalize_projection(activation)


def test_vector_worker_stages_only_then_final_authority_publishes(tmp_path: Path) -> None:
    session = _vector_session()
    workbench = ArtifactWorkbench(session=session)
    controller = ArtifactExportController(workbench, id_factory=SequentialIds())
    destination = tmp_path / "section.amr-vector"

    item = controller.begin_vector(destination, "record:vector:export")
    result = controller.execute(item)

    assert not destination.exists()
    assert result.staging_directory.is_dir()
    validate_vector_export_package(
        result.staging_directory,
        document=session.document,
    )
    assert controller.summary(item).state is ArtifactExportState.STAGED

    publication = controller.publish_result(item, result)

    assert publication.destination == destination
    assert publication.durability_confirmed is True
    assert publication.warning_message is None
    assert destination.is_dir()
    assert not result.staging_directory.exists()
    validate_vector_export_package(destination, document=session.document)
    assert controller.summary(item).state is ArtifactExportState.COMPLETED


def test_begin_rejects_a_broken_destination_symlink_without_following_it(
    tmp_path: Path,
) -> None:
    session = _vector_session()
    controller = ArtifactExportController(ArtifactWorkbench(session=session))
    target = tmp_path / "missing-parent" / "target.amr-vector"
    destination = tmp_path / "linked.amr-vector"
    try:
        destination.symlink_to(target, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"directory symlink unavailable: {exc}")

    with pytest.raises(ArtifactExportError, match="already exists"):
        controller.begin_vector(destination, "record:vector:export")

    assert destination.is_symlink()
    assert not target.exists()


def test_same_align_unrelated_record_append_keeps_export_authority(tmp_path: Path) -> None:
    captured = _vector_session()
    workbench = ArtifactWorkbench(session=captured)
    controller = ArtifactExportController(workbench, id_factory=SequentialIds())
    destination = tmp_path / "same-align.amr-vector"
    item = controller.begin_vector(destination, "record:vector:export")
    result = controller.execute(item)

    current = _append_unrelated_record(workbench)
    assert current.document.canonical_sha256 != captured.document.canonical_sha256

    publication = controller.publish_result(item, result)

    assert publication.document_sha256 == captured.document.canonical_sha256
    assert destination.is_dir()
    assert set(current.document.record_index) == {
        "record:vector:export",
        "record:unrelated",
    }


def test_align_race_rejects_and_cleans_owned_staging(tmp_path: Path) -> None:
    session = _vector_session()
    workbench = ArtifactWorkbench(session=session)
    controller = ArtifactExportController(workbench, id_factory=SequentialIds())
    destination = tmp_path / "stale-align.amr-vector"
    item = controller.begin_vector(destination, "record:vector:export")
    result = controller.execute(item)

    _change_align(workbench)

    with pytest.raises(StaleExportOperationError, match="projection"):
        controller.publish_result(item, result)

    assert not destination.exists()
    assert not result.staging_directory.exists()
    assert controller.summary(item).state is ArtifactExportState.STALE


def test_pending_open_is_retryable_and_never_publishes_early(tmp_path: Path) -> None:
    session = _vector_session()
    workbench = ArtifactWorkbench(session=session)
    controller = ArtifactExportController(workbench, id_factory=SequentialIds())
    destination = tmp_path / "open-busy.amr-vector"
    item = controller.begin_vector(destination, "record:vector:export")
    result = controller.execute(item)
    ticket = workbench.begin_new_import(
        "/source/replacement.ply",
        ConfirmedSourceMetadata(
            unit="mm",
            source_x="+X",
            source_y="+Y",
            source_z="+Z",
            handedness="right",
        ),
        software_version="0.7-test",
        operator="pytest",
    )

    with pytest.raises(WorkflowBusyError, match="Open"):
        controller.publish_result(item, result)

    assert not destination.exists()
    assert result.staging_directory.is_dir()
    assert controller.summary(item).state is ArtifactExportState.STAGED

    workbench.cancel_load(ticket)
    controller.publish_result(item, result)
    assert destination.is_dir()


def test_completed_open_revokes_result_and_cleans_staging(tmp_path: Path) -> None:
    session = _vector_session()
    workbench = ArtifactWorkbench(session=session)
    controller = ArtifactExportController(workbench, id_factory=SequentialIds())
    destination = tmp_path / "open-stale.amr-vector"
    item = controller.begin_vector(destination, "record:vector:export")
    result = controller.execute(item)
    ticket = workbench.begin_new_import(
        "/source/replacement.ply",
        ConfirmedSourceMetadata(
            unit="mm",
            source_x="+X",
            source_y="+Y",
            source_z="+Z",
            handedness="right",
        ),
        software_version="0.7-test",
        operator="pytest",
        created_at=STAMP,
        document_id="artifact:replacement",
        metadata_revision_id="metadata:replacement",
        align_revision_id="align:replacement",
    )
    transition = workbench.prepare_loaded_source(
        ticket,
        _mesh(source_sha="b" * 64, name="replacement.ply"),
    )
    activation = workbench.activate_projection(transition)
    workbench.finalize_projection(activation)

    with pytest.raises(StaleExportOperationError):
        controller.publish_result(item, result)

    assert not destination.exists()
    assert not result.staging_directory.exists()
    assert controller.summary(item).state is ArtifactExportState.STALE


def test_forged_result_cannot_publish_or_delete_exact_result(tmp_path: Path) -> None:
    session = _vector_session()
    controller = ArtifactExportController(
        ArtifactWorkbench(session=session),
        id_factory=SequentialIds(),
    )
    destination = tmp_path / "exact-capability.amr-vector"
    item = controller.begin_vector(destination, "record:vector:export")
    result = controller.execute(item)
    forged = replace(result)

    with pytest.raises(ArtifactExportError, match="exact result capability"):
        controller.publish_result(item, forged)
    with pytest.raises(ArtifactExportError, match="exact result capability"):
        controller.discard_result(item, forged)

    assert result.staging_directory.is_dir()
    assert not destination.exists()
    controller.publish_result(item, result)
    assert destination.is_dir()


def test_destination_race_preserves_winner_and_discards_owned_stage(
    tmp_path: Path,
) -> None:
    session = _vector_session()
    controller = ArtifactExportController(
        ArtifactWorkbench(session=session),
        id_factory=SequentialIds(),
    )
    destination = tmp_path / "raced.amr-vector"
    item = controller.begin_vector(destination, "record:vector:export")
    result = controller.execute(item)
    destination.mkdir()
    sentinel = destination / "winner.txt"
    sentinel.write_text("other process", encoding="utf-8")

    with pytest.raises(ArtifactExportError, match="already exists"):
        controller.publish_result(item, result)

    assert sentinel.read_text(encoding="utf-8") == "other process"
    assert not result.staging_directory.exists()
    assert controller.summary(item).state is ArtifactExportState.FAILED


def test_replaced_staging_is_preserved_and_cleanup_is_not_claimed(
    tmp_path: Path,
) -> None:
    session = _vector_session()
    controller = ArtifactExportController(
        ArtifactWorkbench(session=session),
        id_factory=SequentialIds(),
    )
    destination = tmp_path / "foreign-stage.amr-vector"
    item = controller.begin_vector(destination, "record:vector:export")
    result = controller.execute(item)
    moved_owned = tmp_path / "moved-owned-stage"
    result.staging_directory.rename(moved_owned)
    result.staging_directory.mkdir()
    sentinel = result.staging_directory / "foreign.txt"
    sentinel.write_text("preserve me", encoding="utf-8")

    with pytest.raises(ArtifactExportError, match="cleanup was not proven"):
        controller.discard_result(item, result)

    assert sentinel.read_text(encoding="utf-8") == "preserve me"
    assert moved_owned.is_dir()
    assert not destination.exists()
    assert controller.summary(item).state is ArtifactExportState.FAILED


def test_cooperative_cancel_after_stage_creation_cleans_without_publish(
    tmp_path: Path,
) -> None:
    session = _vector_session()
    controller = ArtifactExportController(
        ArtifactWorkbench(session=session),
        id_factory=SequentialIds(),
    )
    destination = tmp_path / "cancelled.amr-vector"
    item = controller.begin_vector(destination, "record:vector:export")
    entered = Event()
    release = Event()
    original_stage = artifact_exports.stage_vector_package
    errors: list[BaseException] = []

    def blocking_stage(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5.0)
        return original_stage(*args, **kwargs)

    def run() -> None:
        try:
            controller.execute(item)
        except BaseException as exc:  # captured for the test thread
            errors.append(exc)

    with patch.object(
        artifact_exports,
        "stage_vector_package",
        side_effect=blocking_stage,
    ):
        worker = Thread(target=run)
        worker.start()
        assert entered.wait(timeout=5.0)
        summary = controller.cancel(item, reason="user cancelled")
        assert summary.state is ArtifactExportState.CANCELLING
        release.set()
        worker.join(timeout=5.0)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ExportCancelledError)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
    assert controller.summary(item).state is ArtifactExportState.CANCELLED


def test_rubbing_worker_recomputes_receipt_then_publishes(tmp_path: Path) -> None:
    session = _rubbing_session()
    workbench = ArtifactWorkbench(session=session)
    controller = ArtifactExportController(workbench, id_factory=SequentialIds())
    destination = tmp_path / "rubbing.amr-rubbing"
    item = controller.begin_rubbing(destination, "record:rubbing:export")

    result = controller.execute(item)

    assert not destination.exists()
    validate_rubbing_export_package(
        result.staging_directory,
        document=session.document,
    )
    publication = controller.publish_result(item, result)
    assert publication.durability_confirmed is True
    validate_rubbing_export_package(destination, document=session.document)


def test_rubbing_budget_failure_has_no_filesystem_effect(tmp_path: Path) -> None:
    session = _rubbing_session()
    controller = ArtifactExportController(
        ArtifactWorkbench(session=session),
        id_factory=SequentialIds(),
        rubbing_memory_budget_bytes=1,
    )
    destination = tmp_path / "too-large.amr-rubbing"
    item = controller.begin_rubbing(destination, "record:rubbing:export")

    with pytest.raises(ExportResourceLimitError, match="exceeds"):
        controller.execute(item)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
    assert controller.summary(item).state is ArtifactExportState.FAILED


def test_post_rename_durability_error_is_completed_with_warning(
    tmp_path: Path,
) -> None:
    session = _vector_session()
    controller = ArtifactExportController(
        ArtifactWorkbench(session=session),
        id_factory=SequentialIds(),
    )
    destination = tmp_path / "uncertain.amr-vector"
    item = controller.begin_vector(destination, "record:vector:export")
    result = controller.execute(item)

    def publish_then_report_uncertain(prepared):
        publish_prepared_vector_package(prepared)
        raise ArtifactVectorExportError(
            "directory fsync failed after rename",
            committed=True,
        )

    with patch.object(
        artifact_exports,
        "publish_prepared_vector_package",
        side_effect=publish_then_report_uncertain,
    ):
        publication = controller.publish_result(item, result)

    assert destination.is_dir()
    assert not result.staging_directory.exists()
    assert publication.durability_confirmed is False
    assert publication.warning_message is not None
    assert "fsync" in publication.warning_message
    assert controller.summary(item).state is ArtifactExportState.COMPLETED


def test_staging_moved_visible_before_authority_is_failed_not_stale(
    tmp_path: Path,
) -> None:
    session = _vector_session()
    workbench = ArtifactWorkbench(session=session)
    controller = ArtifactExportController(workbench, id_factory=SequentialIds())
    destination = tmp_path / "unauthorized-visible.amr-vector"
    item = controller.begin_vector(destination, "record:vector:export")
    result = controller.execute(item)
    result.staging_directory.rename(destination)
    _change_align(workbench)

    with pytest.raises(ArtifactExportError, match="before final authority"):
        controller.publish_result(item, result)

    assert destination.is_dir()
    assert controller.summary(item).state is ArtifactExportState.FAILED
