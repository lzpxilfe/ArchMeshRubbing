from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

import numpy as np
import pytest

from src.application.artifact_workbench import (
    ArtifactWorkbench,
    ArtifactWorkbenchError,
    ConfirmedSourceMetadata,
    SaveDurability,
    SavedProjectCheckpoint,
    StaleWorkflowOperationError,
    WorkflowBusyError,
    WorkflowPhase,
    WorkflowSaveStatus,
    WorkflowStage,
    WorkflowTransitionKind,
)
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_extractor import (
    commit_vector_computation,
    compute_artifact_cutline,
)
from src.core.artifact_vector_record import PlanarFrame
from src.core.mesh_import_recipe import (
    MeshImportRecipeError,
    current_mesh_import_recipe,
)
from src.core.mesh_loader import MeshData, MeshLoader
from src.core.source_identity import SourceFingerprint


STAMP = "2026-07-12T00:00:00Z"
SOURCE_SHA = "a" * 64


def _resolved(path: str) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


class SequentialIds:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, prefix: str) -> str:
        self.value += 1
        return f"{prefix}:test-{self.value}"


def _mesh(
    *,
    sha256: str = SOURCE_SHA,
    source_format: str = "ply",
) -> MeshData:
    return MeshData(
        vertices=np.asarray(
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [10.0, 10.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        unit="cm",
        filepath=Path("/source/artifact.ply"),
        source_identity=SourceFingerprint(
            sha256=sha256,
            size_bytes=123,
            mtime_ns=1,
            original_name="artifact.ply",
            format="ply",
        ),
        source_format=source_format,
        source_import_recipe=current_mesh_import_recipe(source_format),
    )


def _metadata() -> ConfirmedSourceMetadata:
    return ConfirmedSourceMetadata(
        unit="cm",
        source_x="+X",
        source_y="+Y",
        source_z="+Z",
        handedness="right",
    )


def _session(*, explicit_align: bool = False) -> ArtifactSession:
    session = ArtifactSession.create_from_source(
        _mesh(),
        resolved_source_path="/source/artifact.ply",
        unit="cm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="0.1.0",
        operator="pytest",
        created_at=STAMP,
        document_id="artifact:workbench-test",
        metadata_revision_id="metadata:initial",
        align_revision_id="align:initial",
    )
    if not explicit_align:
        return session
    return session.commit_preview(
        translation_mm=(0.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:confirmed",
    )


def _record_candidate(
    session: ArtifactSession,
    *,
    record_id: str = "record:workbench-cutline",
) -> ArtifactSession:
    computation = compute_artifact_cutline(
        session,
        PlanarFrame(
            origin_world_mm=(150.0, 0.0, 0.0),
            u_axis_world=(0.0, 1.0, 0.0),
            v_axis_world=(0.0, 0.0, 1.0),
            normal_world=(1.0, 0.0, 0.0),
        ),
    )
    return commit_vector_computation(
        session,
        computation,
        record_id=record_id,
        created_at=STAMP,
        operator="pytest",
    )


def _publish(workbench: ArtifactWorkbench, transition):
    activation = workbench.activate_projection(transition)
    return workbench.finalize_projection(activation)


def test_initial_state_is_headless_empty_and_not_measurement_ready() -> None:
    workbench = ArtifactWorkbench(id_factory=SequentialIds())
    snapshot = workbench.snapshot

    assert snapshot.phase is WorkflowPhase.INITIAL
    assert snapshot.stage is WorkflowStage.EMPTY
    assert snapshot.state_version == 0
    assert snapshot.authority_epoch == 0
    assert snapshot.session is None
    assert not snapshot.can_save
    assert not snapshot.can_measure
    assert snapshot.save_status is WorkflowSaveStatus.EMPTY
    assert not snapshot.has_unsaved_changes


def test_initial_checkpoint_distinguishes_saved_and_unsaved_sessions() -> None:
    session = _session(explicit_align=True)

    unsaved = ArtifactWorkbench(
        session=session,
        id_factory=SequentialIds(),
    ).snapshot
    saved = ArtifactWorkbench(
        session=session,
        project_path="/projects/saved.amr",
        id_factory=SequentialIds(),
    ).snapshot

    assert unsaved.project_path is None
    assert unsaved.save_checkpoint is None
    assert unsaved.save_status is WorkflowSaveStatus.UNSAVED
    assert unsaved.has_unsaved_changes
    assert not unsaved.save_checkpoint_current

    assert saved.save_checkpoint == SavedProjectCheckpoint(
        document_sha256=session.document.canonical_sha256,
        project_path="/projects/saved.amr",
    )
    assert saved.save_status is WorkflowSaveStatus.SAVED
    assert not saved.has_unsaved_changes
    assert saved.save_checkpoint_current


def test_saved_project_path_adoption_is_an_authority_checked_cas() -> None:
    session = _session(explicit_align=True)
    workbench = ArtifactWorkbench(
        session=session,
        project_path="/projects/before.amr",
        id_factory=SequentialIds(),
    )
    observed = []
    workbench.subscribe(observed.append)
    captured = workbench.snapshot

    adopted = workbench.adopt_saved_project_path(
        session,
        "/projects/after.amr",
        expected_state_version=captured.state_version,
        expected_authority_epoch=captured.authority_epoch,
    )

    assert adopted.session is session
    assert adopted.project_path == _resolved("/projects/after.amr")
    assert adopted.state_version == captured.state_version + 1
    assert adopted.authority_epoch == captured.authority_epoch
    assert adopted.save_checkpoint_current
    assert adopted.save_status is WorkflowSaveStatus.SAVED
    assert observed == [captured, adopted]

    with pytest.raises(StaleWorkflowOperationError, match="stale artifact authority"):
        workbench.adopt_saved_project_path(
            session,
            "/projects/stale.amr",
            expected_state_version=captured.state_version,
            expected_authority_epoch=captured.authority_epoch,
        )
    assert workbench.snapshot is adopted


def test_same_path_save_creates_checkpoint_and_uncertain_durability_stays_dirty() -> None:
    session = _session(explicit_align=True)
    workbench = ArtifactWorkbench(session=session, id_factory=SequentialIds())
    workbench.synchronize_legacy_session(
        session,
        project_path="/projects/live.amr",
    )
    before_save = workbench.snapshot
    assert before_save.project_path == _resolved("/projects/live.amr")
    assert before_save.save_checkpoint is None
    assert before_save.has_unsaved_changes

    uncertain = workbench.adopt_saved_project_path(
        session,
        "/projects/live.amr",
        expected_state_version=before_save.state_version,
        expected_authority_epoch=before_save.authority_epoch,
        durability_confirmed=False,
    )

    assert uncertain.state_version == before_save.state_version + 1
    assert uncertain.save_checkpoint is not None
    assert uncertain.save_checkpoint.durability is SaveDurability.UNCERTAIN
    assert uncertain.save_status is WorkflowSaveStatus.DURABILITY_UNCERTAIN
    assert uncertain.has_unsaved_changes
    assert not uncertain.save_checkpoint_current

    confirmed = workbench.adopt_saved_project_path(
        session,
        "/projects/live.amr",
        expected_state_version=uncertain.state_version,
        expected_authority_epoch=uncertain.authority_epoch,
    )
    assert confirmed.save_checkpoint is not None
    assert confirmed.save_checkpoint.durability is SaveDurability.CONFIRMED
    assert confirmed.save_checkpoint_current
    assert not confirmed.has_unsaved_changes


def test_saved_project_path_noop_still_requires_exact_authority() -> None:
    session = _session(explicit_align=True)
    workbench = ArtifactWorkbench(
        session=session,
        project_path="/projects/live.amr",
        id_factory=SequentialIds(),
    )
    captured = workbench.snapshot

    unchanged = workbench.adopt_saved_project_path(
        session,
        "/projects/live.amr",
        expected_state_version=captured.state_version,
        expected_authority_epoch=captured.authority_epoch,
    )

    assert unchanged is captured

    replacement = _session(explicit_align=True)
    workbench.synchronize_legacy_session(
        replacement,
        project_path="/projects/live.amr",
    )
    with pytest.raises(StaleWorkflowOperationError, match="stale artifact authority"):
        workbench.adopt_saved_project_path(
            session,
            "/projects/live.amr",
            expected_state_version=captured.state_version,
            expected_authority_epoch=captured.authority_epoch,
        )
    assert workbench.snapshot.save_checkpoint is None
    assert workbench.snapshot.has_unsaved_changes


def test_saved_checkpoint_rejects_invalid_hash_and_durability() -> None:
    with pytest.raises(ArtifactWorkbenchError, match="64 lowercase hexadecimal"):
        SavedProjectCheckpoint(
            document_sha256="not-a-document-hash",
            project_path="/projects/live.amr",
        )
    with pytest.raises(ArtifactWorkbenchError, match="confirmed or uncertain"):
        SavedProjectCheckpoint(
            document_sha256="a" * 64,
            project_path="/projects/live.amr",
            durability="maybe",
        )


def test_new_import_is_ticketed_and_publishes_only_after_finalize() -> None:
    ids = SequentialIds()
    workbench = ArtifactWorkbench(id_factory=ids)
    events = []
    workbench.subscribe(events.append)

    ticket = workbench.begin_new_import(
        "/source/artifact.ply",
        _metadata(),
        software_version="0.1.0",
        operator="pytest",
        request_id="open:new",
        created_at=STAMP,
        document_id="artifact:opened",
        metadata_revision_id="metadata:opened",
        align_revision_id="align:opened-identity",
    )
    assert workbench.snapshot.phase is WorkflowPhase.IMPORTING
    assert workbench.snapshot.session is None
    assert dict(ticket.import_recipe) == current_mesh_import_recipe("ply")
    with pytest.raises(TypeError):
        ticket.import_recipe["format"] = "obj"  # type: ignore[index]

    transition = workbench.prepare_loaded_source(ticket, _mesh())
    assert workbench.snapshot.session is None
    activation = workbench.activate_projection(transition)

    # Scene-swap callbacks may read tentative authority, but observers do not
    # see it until the projection port has succeeded.
    assert workbench.session is transition.candidate_session
    assert len(events) == 2
    ready = workbench.finalize_projection(activation)

    assert ready.phase is WorkflowPhase.READY
    assert ready.stage is WorkflowStage.ALIGN_REQUIRED
    assert ready.can_save
    assert not ready.can_measure
    assert ready.authority_epoch == 1
    assert len(events) == 3
    assert events[-1] is ready
    assert ready.session is not None
    assert ready.save_checkpoint is None
    assert ready.save_status is WorkflowSaveStatus.UNSAVED
    assert ready.has_unsaved_changes
    np.testing.assert_array_equal(_mesh().vertices, ready.session.source_mesh.vertices)


def test_pending_failed_and_cancelled_open_preserve_saved_checkpoint() -> None:
    live = _session(explicit_align=True)
    workbench = ArtifactWorkbench(
        session=live,
        project_path="/projects/live.amr",
        id_factory=SequentialIds(),
    )
    saved = workbench.snapshot
    assert saved.save_checkpoint_current

    failed_ticket = workbench.begin_new_import(
        "/source/replacement.ply",
        _metadata(),
        software_version="0.1.0",
        operator="pytest",
    )
    importing = workbench.snapshot
    assert importing.save_checkpoint is saved.save_checkpoint
    assert importing.save_checkpoint_current
    assert not importing.has_unsaved_changes

    failed = workbench.fail_load(failed_ticket, RuntimeError("load failed"))
    assert failed.save_checkpoint is saved.save_checkpoint
    assert failed.save_checkpoint_current
    assert not failed.has_unsaved_changes

    cancelled_ticket = workbench.begin_new_import(
        "/source/replacement.ply",
        _metadata(),
        software_version="0.1.0",
        operator="pytest",
    )
    cancelled = workbench.cancel_load(cancelled_ticket)
    assert cancelled.save_checkpoint is saved.save_checkpoint
    assert cancelled.save_checkpoint_current
    assert not cancelled.has_unsaved_changes


def test_new_import_rejects_parser_runtime_failure_without_state_change() -> None:
    workbench = ArtifactWorkbench(id_factory=SequentialIds())
    initial = workbench.snapshot

    with (
        patch(
            "src.application.artifact_workbench.current_mesh_import_recipe",
            side_effect=MeshImportRecipeError("runtime lock drift"),
        ),
        pytest.raises(ArtifactWorkbenchError, match="parser runtime.*lock drift"),
    ):
        workbench.begin_new_import(
            "/source/artifact.ply",
            _metadata(),
            software_version="0.1.0",
            operator="pytest",
        )

    assert workbench.snapshot is initial


def test_zero_delta_align_is_a_durable_explicit_confirmation() -> None:
    source = _session()
    workbench = ArtifactWorkbench(session=source, id_factory=SequentialIds())
    assert workbench.snapshot.stage is WorkflowStage.ALIGN_REQUIRED

    transition = workbench.prepare_align_commit(
        translation_mm=(0.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(100.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:explicit-identity",
    )
    assert transition is not None
    ready = _publish(workbench, transition)

    assert ready.stage is WorkflowStage.MEASUREMENT_READY
    assert ready.can_measure
    assert ready.session is not None
    active = ready.session.document.align_revision_index["align:explicit-identity"]
    assert active.parent_id == "align:initial"
    assert active.recipe["kind"] == "manual_scene_trs_delta"
    assert active.recipe["translation_mm"] == (0.0, 0.0, 0.0)
    assert active.recipe["rotation_deg"] == (0.0, 0.0, 0.0)
    np.testing.assert_array_equal(source.source_mesh.vertices, _mesh().vertices)

    state_before = workbench.snapshot
    assert (
        workbench.prepare_align_commit(
            translation_mm=(0.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            pivot_mm=(0.0, 0.0, 0.0),
            operator="pytest",
        )
        is None
    )
    assert workbench.snapshot is state_before


def test_align_and_record_append_dirty_the_exact_saved_document_checkpoint() -> None:
    source = _session()
    workbench = ArtifactWorkbench(
        session=source,
        project_path="/projects/live.amr",
        id_factory=SequentialIds(),
    )
    original_checkpoint = workbench.snapshot.save_checkpoint
    assert original_checkpoint is not None
    assert workbench.snapshot.save_checkpoint_current

    align = workbench.prepare_align_commit(
        translation_mm=(0.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:checkpoint-dirty",
    )
    assert align is not None
    aligned = _publish(workbench, align)
    assert aligned.save_checkpoint is original_checkpoint
    assert aligned.document_sha256 != original_checkpoint.document_sha256
    assert aligned.has_unsaved_changes
    assert aligned.session is not None

    saved = workbench.adopt_saved_project_path(
        aligned.session,
        "/projects/live.amr",
        expected_state_version=aligned.state_version,
        expected_authority_epoch=aligned.authority_epoch,
    )
    assert not saved.has_unsaved_changes
    assert saved.session is not None

    candidate = _record_candidate(saved.session, record_id="record:checkpoint-dirty")
    binding = workbench.prepare_record_commit(
        saved.session,
        candidate,
        expected_new_record_ids=("record:checkpoint-dirty",),
    )
    with_record = workbench.finalize_record_binding(
        workbench.activate_record_binding(binding)
    )
    assert with_record.save_checkpoint is saved.save_checkpoint
    assert with_record.has_unsaved_changes
    assert with_record.save_status is WorkflowSaveStatus.UNSAVED


def test_session_update_cannot_forge_a_clean_project_reopen_checkpoint() -> None:
    live = _session(explicit_align=True)
    workbench = ArtifactWorkbench(
        session=live,
        project_path="/projects/live.amr",
        id_factory=SequentialIds(),
    )
    transition = workbench.prepare_align_commit(
        translation_mm=(1.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        revision_id="align:forged-reopen",
    )
    assert transition is not None
    forged = replace(
        transition,
        kind=WorkflowTransitionKind.REOPEN_PROJECT,
        project_path="/projects/forged.amr",
    )

    with pytest.raises(ArtifactWorkbenchError, match="no live load ticket"):
        workbench.activate_projection(forged)

    assert workbench.snapshot.project_path == _resolved("/projects/live.amr")
    assert workbench.snapshot.save_checkpoint_current


def test_busy_begin_and_stale_results_are_atomic() -> None:
    workbench = ArtifactWorkbench(id_factory=SequentialIds())
    first = workbench.begin_new_import(
        "/source/artifact.ply",
        _metadata(),
        software_version="0.1.0",
        operator="pytest",
        request_id="open:first",
    )
    importing = workbench.snapshot

    with pytest.raises(WorkflowBusyError, match="already pending"):
        workbench.begin_new_import(
            "/source/other.ply",
            _metadata(),
            software_version="0.1.0",
            operator="pytest",
        )
    assert workbench.snapshot is importing

    workbench.cancel_load(first)
    second = workbench.begin_new_import(
        "/source/artifact.ply",
        _metadata(),
        software_version="0.1.0",
        operator="pytest",
        request_id="open:second",
    )
    current = workbench.snapshot
    with pytest.raises(StaleWorkflowOperationError, match="stale"):
        workbench.prepare_loaded_source(first, _mesh())
    with pytest.raises(StaleWorkflowOperationError, match="stale"):
        workbench.fail_load(first, RuntimeError("late failure"))
    assert workbench.snapshot is current
    assert workbench.snapshot.pending_load is second


def test_loaded_source_must_match_ticketed_parser_receipt() -> None:
    workbench = ArtifactWorkbench(id_factory=SequentialIds())
    ticket = workbench.begin_new_import(
        "/source/artifact.ply",
        _metadata(),
        software_version="0.1.0",
        operator="pytest",
    )
    mesh = _mesh()
    receipt = dict(mesh.source_import_recipe or {})
    receipt["runtime_lock_sha256"] = "b" * 64
    mesh.source_import_recipe = receipt

    with pytest.raises(ArtifactWorkbenchError, match="parser receipt.*Open ticket"):
        workbench.prepare_loaded_source(ticket, mesh)

    assert workbench.snapshot.pending_load is ticket
    assert workbench.snapshot.session is None


def test_new_import_accepts_manifest_finalized_from_ticketed_parser_base(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "artifact.obj"
    source_path.write_text(
        "mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    (tmp_path / "material.mtl").write_text(
        "newmtl plain\nKd 0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    workbench = ArtifactWorkbench(id_factory=SequentialIds())
    ticket = workbench.begin_new_import(
        str(source_path),
        _metadata(),
        software_version="0.1.0",
        operator="pytest",
        created_at=STAMP,
    )
    mesh = MeshLoader(default_unit="mm").load(source_path, unit="cm")

    transition = workbench.prepare_loaded_source(ticket, mesh)

    assert ticket.capture_dependencies
    assert ticket.import_recipe["dependency_policy"] == "deny_external"
    geometry = transition.candidate_session.document.geometry_revisions[0]
    assert geometry.import_recipe["dependency_policy"] == "closed_manifest"
    assert geometry.import_recipe["source_manifest"]["entries"][0][
        "logical_path"
    ] == "artifact.obj"


def test_reused_request_id_cannot_revive_a_cancelled_ticket() -> None:
    workbench = ArtifactWorkbench(id_factory=SequentialIds())
    first = workbench.begin_new_import(
        "/source/artifact.ply",
        _metadata(),
        software_version="0.1.0",
        operator="pytest",
        request_id="open:reused",
    )
    workbench.cancel_load(first)
    second = workbench.begin_new_import(
        "/source/artifact.ply",
        _metadata(),
        software_version="0.1.0",
        operator="pytest",
        request_id="open:reused",
    )
    assert first == second
    assert first is not second

    with pytest.raises(StaleWorkflowOperationError, match="stale"):
        workbench.prepare_loaded_source(first, _mesh())
    with pytest.raises(StaleWorkflowOperationError, match="stale"):
        workbench.fail_load(first, RuntimeError("late failure"))
    assert workbench.snapshot.pending_load is second


def test_failed_replacement_preserves_live_authority_and_save_target() -> None:
    live = _session(explicit_align=True)
    workbench = ArtifactWorkbench(
        session=live,
        project_path="/projects/live.amr",
        id_factory=SequentialIds(),
    )
    ticket = workbench.begin_project_reopen(
        live.document,
        project_path="/projects/replacement.amr",
        resolved_source_path="/relocated/artifact.ply",
        request_id="open:replacement",
    )
    failed = workbench.fail_load(ticket, RuntimeError("source mismatch"))

    assert failed.session is live
    assert failed.project_path == _resolved("/projects/live.amr")
    assert failed.phase is WorkflowPhase.READY
    assert failed.failure is not None
    assert failed.failure.message == "source mismatch"
    assert failed.can_save
    assert failed.can_measure


def test_project_reopen_accepts_relocated_identical_source_and_saved_parser() -> None:
    saved = _session(explicit_align=True)
    workbench = ArtifactWorkbench(id_factory=SequentialIds())
    ticket = workbench.begin_project_reopen(
        saved.document,
        project_path="/projects/saved.amr",
        resolved_source_path="/relocated/artifact.raw-scan",
        request_id="open:project",
    )
    assert ticket.source_format == "ply"
    metadata_id = saved.document.active_source_metadata_revision_id
    assert metadata_id is not None
    metadata = saved.document.source_metadata_revision_index[metadata_id]
    geometry = saved.document.geometry_revision_index[metadata.geometry_revision_id]
    assert dict(ticket.import_recipe) == dict(geometry.import_recipe)
    transition = workbench.prepare_loaded_source(
        ticket,
        _mesh(),
        resolved_source_path="/relocated/artifact.raw-scan",
    )
    ready = _publish(workbench, transition)

    assert ready.session is not None
    assert ready.session.document is saved.document
    assert ready.session.resolved_source_path == _resolved(
        "/relocated/artifact.raw-scan"
    )
    assert ready.project_path == _resolved("/projects/saved.amr")
    assert ready.can_measure
    assert ready.save_checkpoint_current
    assert ready.save_status is WorkflowSaveStatus.SAVED
    assert not ready.has_unsaved_changes


def test_project_reopen_rejects_replaced_prepared_capability_atomically() -> None:
    saved = _session(explicit_align=True)
    workbench = ArtifactWorkbench(id_factory=SequentialIds())
    ticket = workbench.begin_project_reopen(
        saved.document,
        project_path="/projects/saved.amr",
        resolved_source_path="/relocated/artifact.ply",
        request_id="open:capability",
    )
    transition = workbench.prepare_loaded_source(
        ticket,
        _mesh(),
        resolved_source_path="/relocated/artifact.ply",
    )

    # Replacing both candidate and projection defeats structural self-consistency
    # checks: the forged pair is a valid session/projection, but it was never
    # issued by this Workbench for the live project file.
    forged_session = _record_candidate(
        transition.candidate_session,
        record_id="record:forged-reopen",
    )
    forged = replace(
        transition,
        candidate_session=forged_session,
        projection=forged_session.materialize(),
    )

    before_forgery = workbench.snapshot
    with pytest.raises(StaleWorkflowOperationError, match="prepared capability"):
        workbench.activate_projection(forged)

    assert workbench.snapshot is before_forgery
    assert workbench.snapshot.pending_load is ticket
    assert workbench.snapshot.session is None
    assert not workbench.snapshot.tentative

    # Rejection must not consume the genuine one-shot capability.
    ready = _publish(workbench, transition)
    assert ready.session is not None
    assert ready.session is transition.candidate_session
    assert ready.session.document is saved.document
    assert ready.session.document.records == ()
    assert ready.save_checkpoint_current
    assert not ready.has_unsaved_changes


def test_second_open_preparation_supersedes_the_first_capability() -> None:
    saved = _session(explicit_align=True)
    workbench = ArtifactWorkbench(id_factory=SequentialIds())
    ticket = workbench.begin_project_reopen(
        saved.document,
        project_path="/projects/saved.amr",
        resolved_source_path="/relocated/artifact.ply",
    )
    first = workbench.prepare_loaded_source(ticket, _mesh())
    second = workbench.prepare_loaded_source(ticket, _mesh())
    captured = workbench.snapshot

    with pytest.raises(StaleWorkflowOperationError, match="prepared capability"):
        workbench.activate_projection(first)
    assert workbench.snapshot is captured

    ready = _publish(workbench, second)
    assert ready.session is second.candidate_session
    assert ready.save_checkpoint_current


def test_project_source_mismatch_never_replaces_live_authority() -> None:
    live = _session(explicit_align=True)
    workbench = ArtifactWorkbench(
        session=live,
        project_path="/projects/live.amr",
        id_factory=SequentialIds(),
    )
    ticket = workbench.begin_project_reopen(
        live.document,
        project_path="/projects/replacement.amr",
        resolved_source_path="/relocated/lookalike.ply",
    )
    with pytest.raises(ArtifactWorkbenchError, match="source bytes"):
        workbench.prepare_loaded_source(ticket, _mesh(sha256="f" * 64))
    assert workbench.session is live
    assert workbench.snapshot.pending_load is ticket


def test_authority_epoch_rejects_second_same_session_commit() -> None:
    live = _session(explicit_align=True)
    workbench = ArtifactWorkbench(session=live, id_factory=SequentialIds())
    first = workbench.prepare_align_commit(
        translation_mm=(1.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:first-worker",
    )
    second = workbench.prepare_align_commit(
        translation_mm=(2.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:second-worker",
    )
    assert first is not None and second is not None
    first_ready = _publish(workbench, first)

    with pytest.raises(StaleWorkflowOperationError, match="stale authority"):
        workbench.activate_projection(second)
    assert workbench.snapshot is first_ready
    assert workbench.session is first.candidate_session


def test_tentative_authority_disables_save_measurement_and_new_commands() -> None:
    live = _session(explicit_align=True)
    workbench = ArtifactWorkbench(session=live, id_factory=SequentialIds())
    transition = workbench.prepare_align_commit(
        translation_mm=(1.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:tentative",
    )
    assert transition is not None
    activation = workbench.activate_projection(transition)

    assert workbench.snapshot.tentative
    assert not workbench.snapshot.can_save
    assert not workbench.snapshot.can_measure
    with pytest.raises(WorkflowBusyError, match="publication"):
        workbench.require_stable_session(transition.candidate_session)
    with pytest.raises(WorkflowBusyError, match="publication"):
        workbench.prepare_activate_parent_align()

    ready = workbench.finalize_projection(activation)
    assert not ready.tentative
    assert ready.can_save
    assert ready.can_measure
    assert ready.session is not None
    assert workbench.require_stable_session(ready.session, measurement=True) is ready.session


def test_session_update_cannot_smuggle_an_align_authority_change() -> None:
    live = _session(explicit_align=True)
    rogue = live.commit_preview(
        translation_mm=(8.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:rogue-session-update",
    )
    workbench = ArtifactWorkbench(session=live, id_factory=SequentialIds())

    with pytest.raises(ArtifactWorkbenchError, match="cannot change Align"):
        workbench.prepare_session_commit(
            live,
            rogue,
            kind=WorkflowTransitionKind.SESSION_UPDATE,
            expected_new_record_ids=(),
        )
    assert workbench.session is live


def test_record_commit_rebinds_document_without_materializing_a_mesh() -> None:
    live = _session(explicit_align=True)
    candidate = _record_candidate(live)
    workbench = ArtifactWorkbench(
        session=live,
        project_path="/projects/live.amr",
        id_factory=SequentialIds(),
    )
    events = []
    workbench.subscribe(events.append)

    with patch.object(
        ArtifactSession,
        "materialize",
        side_effect=AssertionError("record commit must not materialize a mesh"),
    ) as materialize:
        transition = workbench.prepare_record_commit(
            live,
            candidate,
            expected_new_record_ids=("record:workbench-cutline",),
        )
        activation = workbench.activate_record_binding(transition)
        ready = workbench.finalize_record_binding(activation)
    materialize.assert_not_called()
    assert transition.expected_snapshot.document_sha256 != (
        transition.candidate_snapshot.document_sha256
    )
    assert transition.expected_snapshot.has_same_render_projection(
        transition.candidate_snapshot
    )

    assert not ready.tentative
    assert ready.session is candidate
    assert ready.project_path == _resolved("/projects/live.amr")
    assert events == [activation.previous, ready]


def test_record_binding_rollback_restores_previous_authority() -> None:
    live = _session(explicit_align=True)
    candidate = _record_candidate(live, record_id="record:rollback")
    workbench = ArtifactWorkbench(
        session=live,
        project_path="/projects/live.amr",
        id_factory=SequentialIds(),
    )
    checkpoint = workbench.snapshot.save_checkpoint
    assert checkpoint is not None
    transition = workbench.prepare_record_commit(
        live,
        candidate,
        expected_new_record_ids=("record:rollback",),
    )
    activation = workbench.activate_record_binding(transition)

    rolled_back = workbench.rollback_record_binding(
        activation,
        RuntimeError("injected binding failure"),
    )

    assert rolled_back.session is live
    assert not rolled_back.tentative
    assert rolled_back.save_checkpoint is checkpoint
    assert rolled_back.save_checkpoint_current
    assert not rolled_back.has_unsaved_changes
    assert rolled_back.failure is not None
    assert rolled_back.failure.operation == "record_binding_publish"


def test_record_binding_activation_rejects_a_forged_snapshot_capability() -> None:
    live = _session(explicit_align=True)
    candidate = _record_candidate(live, record_id="record:forged-binding")
    workbench = ArtifactWorkbench(session=live, id_factory=SequentialIds())
    transition = workbench.prepare_record_commit(
        live,
        candidate,
        expected_new_record_ids=("record:forged-binding",),
    )
    forged = replace(
        transition,
        candidate_snapshot=transition.expected_snapshot,
    )

    with pytest.raises(ArtifactWorkbenchError, match="immutable sessions"):
        workbench.activate_record_binding(forged)

    assert workbench.session is live
    assert not workbench.snapshot.tentative


def test_export_effect_allows_same_align_append_and_rejects_align_change() -> None:
    base = _session(explicit_align=True)
    captured = _record_candidate(base, record_id="record:export-root")
    expected_record = captured.document.record_index["record:export-root"]
    snapshot = captured.projection_snapshot()
    workbench = ArtifactWorkbench(session=captured, id_factory=SequentialIds())

    appended = _record_candidate(captured, record_id="record:unrelated")
    transition = workbench.prepare_record_commit(
        captured,
        appended,
        expected_new_record_ids=("record:unrelated",),
    )
    activation = workbench.activate_record_binding(transition)
    workbench.finalize_record_binding(activation)

    published: list[str] = []
    result = workbench.publish_record_effect_if_current(
        captured,
        snapshot,
        record_id="record:export-root",
        expected_record=expected_record,
        publish=lambda: published.append("same-align") or "published",
    )
    assert result == "published"
    assert published == ["same-align"]

    align = workbench.prepare_align_commit(
        translation_mm=(1.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:after-export-stage",
    )
    assert align is not None
    _publish(workbench, align)

    with pytest.raises(StaleWorkflowOperationError, match="projection authority"):
        workbench.publish_record_effect_if_current(
            captured,
            snapshot,
            record_id="record:export-root",
            expected_record=expected_record,
            publish=lambda: published.append("stale"),
        )
    assert published == ["same-align"]


def test_export_effect_is_busy_during_pending_open() -> None:
    base = _session(explicit_align=True)
    captured = _record_candidate(base, record_id="record:export-pending")
    workbench = ArtifactWorkbench(session=captured, id_factory=SequentialIds())
    ticket = workbench.begin_new_import(
        "/source/replacement.ply",
        _metadata(),
        software_version="0.1.0",
        operator="pytest",
    )
    called = False

    def publish() -> None:
        nonlocal called
        called = True

    with pytest.raises(WorkflowBusyError, match="pending"):
        workbench.publish_record_effect_if_current(
            captured,
            captured.projection_snapshot(),
            record_id="record:export-pending",
            expected_record=captured.document.record_index["record:export-pending"],
            publish=publish,
        )
    assert not called
    workbench.cancel_load(ticket)


def test_export_effect_lease_blocks_reentrant_authority_mutators() -> None:
    base = _session(explicit_align=True)
    captured = _record_candidate(base, record_id="record:export-reentrant")
    record = captured.document.record_index["record:export-reentrant"]
    workbench = ArtifactWorkbench(
        session=captured,
        project_path="/projects/live.amr",
        id_factory=SequentialIds(),
    )
    before = workbench.snapshot

    def publish() -> str:
        with pytest.raises(WorkflowBusyError, match="external effect publication"):
            workbench.synchronize_legacy_session(
                base,
                project_path="/projects/reentrant.amr",
            )
        with pytest.raises(WorkflowBusyError, match="external effect publication"):
            workbench.begin_new_import(
                "/source/reentrant.ply",
                _metadata(),
                software_version="0.1.0",
                operator="pytest",
            )
        with pytest.raises(WorkflowBusyError, match="external effect publication"):
            workbench.prepare_align_commit(
                translation_mm=(1.0, 0.0, 0.0),
                rotation_deg=(0.0, 0.0, 0.0),
                scale=1.0,
                pivot_mm=(0.0, 0.0, 0.0),
                operator="pytest",
                revision_id="align:reentrant",
            )
        return "published"

    result = workbench.publish_record_effect_if_current(
        captured,
        captured.projection_snapshot(),
        record_id=record.id,
        expected_record=record,
        publish=publish,
    )

    assert result == "published"
    assert workbench.snapshot is before


def test_export_effect_callback_exception_clears_publication_lease() -> None:
    base = _session(explicit_align=True)
    captured = _record_candidate(base, record_id="record:export-exception")
    record = captured.document.record_index["record:export-exception"]
    workbench = ArtifactWorkbench(session=captured, id_factory=SequentialIds())
    before = workbench.snapshot

    def publish() -> None:
        raise RuntimeError("injected publish failure")

    with pytest.raises(RuntimeError, match="injected publish failure"):
        workbench.publish_record_effect_if_current(
            captured,
            captured.projection_snapshot(),
            record_id=record.id,
            expected_record=record,
            publish=publish,
        )

    assert workbench.snapshot is before
    transition = workbench.prepare_align_commit(
        translation_mm=(1.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        revision_id="align:after-publish-failure",
    )
    assert transition is not None


def test_export_effect_serializes_a_concurrent_authority_mutator() -> None:
    base = _session(explicit_align=True)
    captured = _record_candidate(base, record_id="record:export-concurrent")
    record = captured.document.record_index["record:export-concurrent"]
    workbench = ArtifactWorkbench(session=captured, id_factory=SequentialIds())
    callback_entered = Event()
    release_callback = Event()
    mutator_started = Event()
    mutator_finished = Event()
    results: list[str] = []
    errors: list[BaseException] = []

    def publish() -> str:
        callback_entered.set()
        if not release_callback.wait(timeout=2.0):
            raise RuntimeError("test did not release export callback")
        return "published"

    def run_publish() -> None:
        try:
            results.append(
                workbench.publish_record_effect_if_current(
                    captured,
                    captured.projection_snapshot(),
                    record_id=record.id,
                    expected_record=record,
                    publish=publish,
                )
            )
        except BaseException as exc:  # pragma: no cover - failure diagnostics
            errors.append(exc)

    def run_mutator() -> None:
        try:
            mutator_started.set()
            workbench.begin_new_import(
                "/source/concurrent.ply",
                _metadata(),
                software_version="0.1.0",
                operator="pytest",
            )
        except BaseException as exc:  # pragma: no cover - failure diagnostics
            errors.append(exc)
        finally:
            mutator_finished.set()

    publish_thread = Thread(target=run_publish)
    publish_thread.start()
    assert callback_entered.wait(timeout=2.0)
    mutator_thread = Thread(target=run_mutator)
    mutator_thread.start()
    assert mutator_started.wait(timeout=2.0)
    assert not mutator_finished.wait(timeout=0.05)

    release_callback.set()
    publish_thread.join(timeout=2.0)
    mutator_thread.join(timeout=2.0)

    assert not publish_thread.is_alive()
    assert not mutator_thread.is_alive()
    assert errors == []
    assert results == ["published"]
    assert workbench.snapshot.pending_load is not None


def test_export_effect_detects_emergency_authority_change_and_stays_faulted() -> None:
    base = _session(explicit_align=True)
    captured = _record_candidate(base, record_id="record:export-fault")
    record = captured.document.record_index["record:export-fault"]
    workbench = ArtifactWorkbench(session=captured, id_factory=SequentialIds())

    def publish() -> None:
        workbench.enter_faulted_state(
            session=captured,
            project_path=None,
            error="injected uncertain external effect",
        )

    with pytest.raises(ArtifactWorkbenchError, match="authority changed"):
        workbench.publish_record_effect_if_current(
            captured,
            captured.projection_snapshot(),
            record_id=record.id,
            expected_record=record,
            publish=publish,
        )

    assert workbench.snapshot.faulted
    assert workbench.snapshot.failure is not None
    assert workbench.snapshot.failure.fatal


def test_publish_rollback_restores_previous_authority_without_observing_candidate() -> None:
    live = _session(explicit_align=True)
    workbench = ArtifactWorkbench(
        session=live,
        project_path="/projects/live.amr",
        id_factory=SequentialIds(),
    )
    events = []
    workbench.subscribe(events.append)
    transition = workbench.prepare_align_commit(
        translation_mm=(5.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:candidate",
    )
    assert transition is not None
    activation = workbench.activate_projection(transition)
    assert workbench.session is transition.candidate_session
    assert events == [activation.previous]

    rolled_back = workbench.rollback_projection(
        activation,
        RuntimeError("injected scene swap failure"),
    )
    assert rolled_back.session is live
    assert rolled_back.project_path == _resolved("/projects/live.amr")
    assert rolled_back.authority_epoch > activation.current.authority_epoch
    assert rolled_back.failure is not None
    assert events == [activation.previous, rolled_back]


def test_fatal_authority_state_blocks_save_and_measure_until_verified_reopen() -> None:
    live = _session(explicit_align=True)
    workbench = ArtifactWorkbench(
        session=live,
        project_path="/projects/live.amr",
        id_factory=SequentialIds(),
    )
    faulted = workbench.enter_faulted_state(
        session=live,
        project_path="/projects/live.amr",
        error=RuntimeError("rollback could not be proven"),
        operation_id="fault:rollback",
    )

    assert faulted.phase is WorkflowPhase.ERROR
    assert faulted.failure is not None and faulted.failure.fatal
    assert not faulted.can_save
    assert not faulted.can_measure
    with pytest.raises(ArtifactWorkbenchError, match="faulted"):
        workbench.require_stable_session(live)

    ticket = workbench.begin_new_import(
        "/source/artifact.ply",
        _metadata(),
        software_version="0.1.0",
        operator="pytest",
    )
    assert workbench.snapshot.pending_load is ticket
    failed_recovery = workbench.fail_load(ticket, RuntimeError("reopen failed"))
    assert failed_recovery.faulted
    assert failed_recovery.phase is WorkflowPhase.ERROR
    assert not failed_recovery.can_save
    assert not failed_recovery.can_measure

    retry = workbench.begin_new_import(
        "/source/artifact.ply",
        _metadata(),
        software_version="0.1.0",
        operator="pytest",
    )
    recovered = _publish(
        workbench,
        workbench.prepare_loaded_source(retry, _mesh()),
    )
    assert not recovered.faulted
    assert recovered.phase is WorkflowPhase.READY
    assert recovered.can_save
    assert recovered.stage is WorkflowStage.ALIGN_REQUIRED


def test_reentrant_observer_notifications_remain_fifo_for_every_observer() -> None:
    workbench = ArtifactWorkbench(id_factory=SequentialIds())
    observed_versions: list[int] = []

    def cancel_pending(snapshot) -> None:
        if snapshot.pending_load is not None:
            workbench.cancel_load(snapshot.pending_load)

    workbench.subscribe(cancel_pending, replay=False)
    workbench.subscribe(
        lambda snapshot: observed_versions.append(snapshot.state_version),
        replay=False,
    )
    workbench.begin_new_import(
        "/source/artifact.ply",
        _metadata(),
        software_version="0.1.0",
        operator="pytest",
    )

    assert observed_versions == [1, 2]
    assert workbench.snapshot.pending_load is None


def test_subscribe_replay_precedes_concurrent_state_notifications() -> None:
    workbench = ArtifactWorkbench(id_factory=SequentialIds())
    observed_versions: list[int] = []
    replay_entered = Event()
    release_replay = Event()
    original_notify_one = ArtifactWorkbench._notify_one

    def blocking_notify_one(observer, snapshot) -> None:
        if snapshot.state_version == 0:
            replay_entered.set()
            assert release_replay.wait(timeout=2.0)
        original_notify_one(observer, snapshot)

    with patch.object(
        ArtifactWorkbench,
        "_notify_one",
        side_effect=blocking_notify_one,
    ):
        subscriber = Thread(
            target=lambda: workbench.subscribe(
                lambda snapshot: observed_versions.append(snapshot.state_version)
            ),
            daemon=True,
        )
        subscriber.start()
        assert replay_entered.wait(timeout=2.0)
        try:
            workbench.begin_new_import(
                "/source/artifact.ply",
                _metadata(),
                software_version="0.1.0",
                operator="pytest",
                request_id="open:concurrent-subscribe",
            )
        finally:
            release_replay.set()
        subscriber.join(timeout=2.0)

    assert not subscriber.is_alive()
    assert observed_versions == [0, 1]


def test_observer_failure_is_isolated_from_state_and_other_observers() -> None:
    workbench = ArtifactWorkbench(id_factory=SequentialIds())
    received = []

    def broken(_snapshot) -> None:
        raise RuntimeError("observer bug")

    workbench.subscribe(broken)
    workbench.subscribe(received.append)
    ticket = workbench.begin_new_import(
        "/source/artifact.ply",
        _metadata(),
        software_version="0.1.0",
        operator="pytest",
    )

    assert workbench.snapshot.pending_load is ticket
    assert received[-1] is workbench.snapshot


def test_invalid_or_unconfirmed_metadata_is_rejected_before_state_changes() -> None:
    workbench = ArtifactWorkbench(id_factory=SequentialIds())
    initial = workbench.snapshot
    with pytest.raises(ArtifactWorkbenchError, match="explicitly confirmed"):
        workbench.begin_new_import(
            "/source/artifact.ply",
            {
                "unit": "cm",
                "axes": {
                    "source_x": "+X",
                    "source_y": "+Y",
                    "source_z": "+Z",
                },
                "handedness": "right",
                "confirmation_status": "unconfirmed",
            },
            software_version="0.1.0",
            operator="pytest",
        )
    assert workbench.snapshot is initial


def test_gltf_new_import_uses_dependency_capture_ticket() -> None:
    workbench = ArtifactWorkbench(id_factory=SequentialIds())

    ticket = workbench.begin_new_import(
        "/source/artifact.gltf",
        _metadata(),
        software_version="0.1.0",
        operator="pytest",
    )

    assert ticket.source_format == "gltf"
    assert ticket.capture_dependencies
    assert workbench.snapshot.pending_load is ticket


def test_saved_gltf_recipe_can_begin_closed_project_reopen() -> None:
    saved = _session(explicit_align=True)
    metadata_id = saved.document.active_source_metadata_revision_id
    assert metadata_id is not None
    metadata = saved.document.source_metadata_revision_index[metadata_id]
    geometry = saved.document.geometry_revision_index[metadata.geometry_revision_id]
    gltf_geometry = replace(
        geometry,
        import_recipe=current_mesh_import_recipe("gltf"),
    )
    document = replace(
        saved.document,
        geometry_revisions=(gltf_geometry,),
    )
    workbench = ArtifactWorkbench(id_factory=SequentialIds())
    ticket = workbench.begin_project_reopen(
        document,
        project_path="/projects/external-gltf.amr",
        resolved_source_path="/source/artifact.gltf",
    )

    assert ticket.source_format == "gltf"
    assert not ticket.capture_dependencies
    assert workbench.snapshot.pending_load is ticket


def test_application_workbench_has_no_qt_opengl_or_gui_imports() -> None:
    path = Path("src/application/artifact_workbench.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(
        name.startswith(("PyQt", "OpenGL", "src.gui")) for name in imported
    )
