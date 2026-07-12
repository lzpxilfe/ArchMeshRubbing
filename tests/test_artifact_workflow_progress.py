from __future__ import annotations

import ast
from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

from src.application.artifact_workflow_progress import (
    ArtifactWorkflowProgress,
    ArtifactWorkflowStep,
    REQUIRED_CUTLINE_VIEWS,
    REQUIRED_SIX_VIEWS,
    derive_artifact_workflow_progress,
)
from src.core.artifact_document import (
    ArtifactDocument,
    RecordLifecycleStatus,
)
from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_rubbing_extractor import (
    commit_artifact_rubbing,
    compute_artifact_rubbing,
)
from src.core.artifact_session import ArtifactSession, ArtifactSessionError
from src.core.artifact_vector_extractor import (
    commit_vector_computation,
    compute_artifact_cutline,
)
from src.core.artifact_vector_record import PlanarFrame
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


STAMP = "2026-07-12T00:00:00Z"


def _session() -> ArtifactSession:
    source = MeshData(
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
        filepath=Path("/source/progress.ply"),
        source_identity=SourceFingerprint(
            sha256="a" * 64,
            size_bytes=123,
            mtime_ns=1,
            original_name="progress.ply",
            format="ply",
        ),
        source_format="ply",
    )
    session = ArtifactSession.create_from_source(
        source,
        resolved_source_path="/source/progress.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="test",
        operator="pytest",
        created_at=STAMP,
        document_id="artifact:workflow-progress",
        metadata_revision_id="metadata:workflow-progress",
        align_revision_id="align:identity",
    )
    return session.commit_preview(
        translation_mm=(0.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        operator="pytest",
        created_at=STAMP,
        revision_id="align:confirmed",
    )


def _cutline_frame(view: str, *, offset_mm: float = 0.0) -> PlanarFrame:
    if view == "top":
        return PlanarFrame(
            origin_world_mm=(0.0, 0.0, offset_mm),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 1.0, 0.0),
            normal_world=(0.0, 0.0, 1.0),
        )
    if view == "front":
        return PlanarFrame(
            origin_world_mm=(0.0, offset_mm, 0.0),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 0.0, 1.0),
            normal_world=(0.0, -1.0, 0.0),
        )
    if view == "right":
        return PlanarFrame(
            origin_world_mm=(offset_mm, 0.0, 0.0),
            u_axis_world=(0.0, 1.0, 0.0),
            v_axis_world=(0.0, 0.0, 1.0),
            normal_world=(1.0, 0.0, 0.0),
        )
    raise ValueError(f"unsupported Cutline view: {view}")


def _append_cutline(
    session: ArtifactSession,
    view: str,
    *,
    suffix: str = "ready",
    depends_on_record_ids: tuple[str, ...] = (),
) -> ArtifactSession:
    computation = compute_artifact_cutline(session, _cutline_frame(view))
    return commit_vector_computation(
        session,
        computation,
        record_id=f"record:cutline:{view}:{suffix}",
        created_at=STAMP,
        operator="pytest",
        depends_on_record_ids=depends_on_record_ids,
    )


def _append_outline(
    session: ArtifactSession,
    view: str,
    *,
    suffix: str = "ready",
) -> ArtifactSession:
    computation = compute_artifact_outline(
        session,
        view,
        precision_grid_mm=0.01,
    )
    return commit_vector_computation(
        session,
        computation,
        record_id=f"record:outline:{view}:{suffix}",
        created_at=STAMP,
        operator="pytest",
    )


def _append_rubbing(
    session: ArtifactSession,
    view: str,
    *,
    suffix: str = "ready",
) -> ArtifactSession:
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
    return commit_artifact_rubbing(
        session,
        computation,
        record_id=f"record:rubbing:{view}:{suffix}",
        created_at=STAMP,
        operator="pytest",
    )


def _append_unknown_record(
    session: ArtifactSession,
    *,
    record_id: str,
    lifecycle_status: RecordLifecycleStatus,
    recipe: dict[str, object] | None = None,
) -> ArtifactSession:
    resolved_recipe = dict(recipe or {"kind": "test"})
    context = session.document.capture_operation_context(recipe=resolved_recipe)
    document = session.document.append_record_from_context(
        context=context,
        id=record_id,
        type="unknown.future.v1",
        geometry_ref=f"urn:test:{record_id}",
        recipe=resolved_recipe,
        qc={},
        lifecycle_status=lifecycle_status,
        created_at=STAMP,
        operator="pytest",
    )
    return session.with_document(document)


def _replace_record(
    session: ArtifactSession,
    record_id: str,
    **changes: object,
) -> ArtifactSession:
    found = False
    records = []
    for record in session.document.records:
        if record.id == record_id:
            records.append(replace(record, **changes))
            found = True
        else:
            records.append(record)
    if not found:
        raise AssertionError(f"missing test record: {record_id}")
    return session.with_document(
        replace(session.document, records=tuple(records))
    )


def _complete_session() -> ArtifactSession:
    session = _session()
    for view in REQUIRED_CUTLINE_VIEWS:
        session = _append_cutline(session, view)
    for view in REQUIRED_SIX_VIEWS:
        session = _append_outline(session, view)
    for view in REQUIRED_SIX_VIEWS:
        session = _append_rubbing(session, view)
    return session


def test_empty_progress_enables_only_cutline_after_explicit_align() -> None:
    session = _session()

    blocked = derive_artifact_workflow_progress(session, align_ready=False)
    assert blocked == ArtifactWorkflowProgress.empty()
    assert not blocked.cutline.enabled
    assert not blocked.outline.enabled
    assert not blocked.rubbing.enabled

    progress = derive_artifact_workflow_progress(session, align_ready=True)
    assert progress.cutline.enabled
    assert not progress.outline.enabled
    assert not progress.rubbing.enabled
    assert progress.cutline.completed_views == ()
    assert progress.cutline.missing_views == REQUIRED_CUTLINE_VIEWS
    assert progress.outline.missing_views == REQUIRED_SIX_VIEWS
    assert progress.rubbing.missing_views == REQUIRED_SIX_VIEWS


def test_unique_ready_fresh_production_records_drive_the_3_6_6_sequence() -> None:
    session = _append_outline(_session(), "top", suffix="early")
    progress = derive_artifact_workflow_progress(session, align_ready=True)
    assert progress.outline.completed_views == ("top",)
    assert not progress.outline.enabled

    session = _append_cutline(session, "top", suffix="first")
    session = _append_cutline(session, "top", suffix="duplicate")
    session = _append_cutline(session, "front", suffix="draft")
    session = _replace_record(
        session,
        "record:cutline:front:draft",
        lifecycle_status=RecordLifecycleStatus.DRAFT,
    )
    session = _append_cutline(session, "right", suffix="failed")
    session = _replace_record(
        session,
        "record:cutline:right:failed",
        lifecycle_status=RecordLifecycleStatus.FAILED,
    )
    progress = derive_artifact_workflow_progress(session, align_ready=True)
    assert progress.cutline.completed_views == ("top",)
    assert progress.cutline.completed_count == 1
    assert not progress.cutline.complete
    assert not progress.outline.enabled

    session = _append_cutline(session, "front")
    session = _append_cutline(session, "right")
    progress = derive_artifact_workflow_progress(session, align_ready=True)
    assert progress.cutline.completed_views == REQUIRED_CUTLINE_VIEWS
    assert progress.cutline.complete
    assert progress.outline.enabled
    assert not progress.rubbing.enabled

    for view in REQUIRED_SIX_VIEWS[1:]:
        session = _append_outline(session, view)
    progress = derive_artifact_workflow_progress(session, align_ready=True)
    assert progress.outline.completed_views == REQUIRED_SIX_VIEWS
    assert progress.outline.complete
    assert progress.rubbing.enabled

    for view in reversed(REQUIRED_SIX_VIEWS):
        session = _append_rubbing(session, view)
    progress = derive_artifact_workflow_progress(session, align_ready=True)
    assert progress.rubbing.completed_views == REQUIRED_SIX_VIEWS
    assert progress.rubbing.complete
    assert progress.for_step(ArtifactWorkflowStep.DIGITAL_RUBBING) is progress.rubbing


def test_non_ready_blocked_unknown_and_noncanonical_records_fail_closed() -> None:
    session = _append_unknown_record(
        _session(),
        record_id="record:dependency:draft",
        lifecycle_status=RecordLifecycleStatus.DRAFT,
    )
    session = _append_cutline(
        session,
        "top",
        suffix="blocked",
        depends_on_record_ids=("record:dependency:draft",),
    )
    session = _append_cutline(
        session,
        "front",
        suffix="missing",
        depends_on_record_ids=("record:dependency:missing",),
    )
    session = _append_cutline(session, "right", suffix="draft")
    session = _replace_record(
        session,
        "record:cutline:right:draft",
        lifecycle_status=RecordLifecycleStatus.DRAFT,
    )
    session = _append_cutline(session, "right", suffix="failed")
    session = _replace_record(
        session,
        "record:cutline:right:failed",
        lifecycle_status=RecordLifecycleStatus.FAILED,
    )
    diagonal = math.sqrt(0.5)
    oblique = PlanarFrame(
        origin_world_mm=(0.125, 0.0, 0.125),
        u_axis_world=(diagonal, 0.0, -diagonal),
        v_axis_world=(0.0, 1.0, 0.0),
        normal_world=(diagonal, 0.0, diagonal),
    )
    computation = compute_artifact_cutline(session, oblique)
    session = commit_vector_computation(
        session,
        computation,
        record_id="record:cutline:oblique",
        created_at=STAMP,
        operator="pytest",
    )
    session = _append_unknown_record(
        session,
        record_id="record:unknown:outline-like",
        lifecycle_status=RecordLifecycleStatus.READY,
        recipe={"kind": "outline", "view": "top"},
    )

    progress = derive_artifact_workflow_progress(session, align_ready=True)
    assert progress.cutline.completed_views == ()
    assert progress.outline.completed_views == ()
    assert progress.rubbing.completed_views == ()
    assert progress.cutline.enabled
    assert not progress.outline.enabled
    assert not progress.rubbing.enabled

    valid = _append_cutline(_session(), "top", suffix="to-corrupt")
    record_id = "record:cutline:top:to-corrupt"
    record = valid.document.record_index[record_id]
    malformed_document = replace(
        valid.document,
        records=(replace(record, extensions={}),),
    )
    with pytest.raises(ArtifactSessionError, match="inline payload descriptor"):
        valid.with_document(malformed_document)
    with pytest.raises(TypeError, match="ArtifactSession"):
        derive_artifact_workflow_progress(
            malformed_document,  # type: ignore[arg-type]
            align_ready=True,
        )


def test_align_staleness_roundtrip_and_parent_reactivation_restore_progress() -> None:
    complete = _complete_session()
    expected = derive_artifact_workflow_progress(complete, align_ready=True)
    assert expected.cutline.complete
    assert expected.outline.complete
    assert expected.rubbing.complete

    reopened_document = ArtifactDocument.from_json_bytes(
        complete.document.canonical_json_bytes()
    )
    reopened = complete.with_document(reopened_document)
    assert derive_artifact_workflow_progress(reopened, align_ready=True) == expected

    active_align_id = complete.document.active_align_revision_id
    assert active_align_id is not None
    changed = complete.commit_preview(
        translation_mm=(0.25, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        operator="pytest",
        created_at=STAMP,
        revision_id="align:changed",
    )
    stale = derive_artifact_workflow_progress(changed, align_ready=True)
    assert stale.cutline.completed_views == ()
    assert stale.outline.completed_views == ()
    assert stale.rubbing.completed_views == ()
    assert stale.cutline.enabled
    assert not stale.outline.enabled
    assert not stale.rubbing.enabled

    restored = changed.activate_align(active_align_id)
    assert derive_artifact_workflow_progress(restored, align_ready=True) == expected

    align_blocked = derive_artifact_workflow_progress(restored, align_ready=False)
    assert align_blocked.cutline.complete
    assert align_blocked.outline.complete
    assert align_blocked.rubbing.complete
    assert not align_blocked.cutline.enabled
    assert not align_blocked.outline.enabled
    assert not align_blocked.rubbing.enabled


def test_progress_module_has_no_qt_opengl_or_gui_imports() -> None:
    path = Path("src/application/artifact_workflow_progress.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(
        name.startswith(("PyQt", "OpenGL", "src.gui")) for name in imported
    )
