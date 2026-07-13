"""Deterministic end-to-end health check for the archaeological workflow.

The native package self-test uses this module to prove more than importability:
one tiny source is opened through the application authority boundary, explicitly
aligned, measured through the required 3/6/6 sequence, embedded in an AMR, and
reopened after the external source has been removed.  The reopened session then
reproduces relocatable 1:1 SVG and PNG packages through the same export
controllers used by the GUI.

The fixture is intentionally tiny and the module imports neither Qt nor OpenGL,
so the check remains deterministic and suitable for an offline Windows build.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from src.application.artifact_exports import ArtifactExportController
from src.application.artifact_measurements import (
    ArtifactMeasurementController,
    ArtifactMeasurementWorkItem,
)
from src.application.artifact_workbench import (
    ArtifactWorkbench,
    ConfirmedSourceMetadata,
    RecordBindingTransition,
    WorkflowStage,
)
from src.application.artifact_workflow_progress import (
    REQUIRED_CUTLINE_VIEWS,
    REQUIRED_SIX_VIEWS,
    ArtifactWorkflowStep,
    derive_artifact_workflow_progress,
    workflow_step_record_ids,
)
from src.core.artifact_rubbing_export import validate_rubbing_export_package
from src.core.artifact_session import ArtifactSession
from src.core.artifact_verification import build_artifact_verification_report
from src.core.artifact_vector_export import validate_vector_export_package
from src.core.artifact_vector_record import PlanarFrame
from src.core.mesh_loader import MeshLoader
from src.core.project_file import (
    load_artifact_session_project,
    save_artifact_session_project,
)


_STAMP = "2026-07-13T00:00:00Z"
_OPERATOR = "packaged-workflow-self-test"
_PLY_BYTES = (
    b"ply\n"
    b"format ascii 1.0\n"
    b"comment deterministic packaged workflow fixture\n"
    b"element vertex 8\n"
    b"property float x\n"
    b"property float y\n"
    b"property float z\n"
    b"element face 12\n"
    b"property list uchar int vertex_indices\n"
    b"end_header\n"
    b"-1 -1 -1\n"
    b"1 -1 -1\n"
    b"1 1 -1\n"
    b"-1 1 -1\n"
    b"-1 -1 1\n"
    b"1 -1 1\n"
    b"1 1 1\n"
    b"-1 1 1\n"
    b"3 0 2 1\n"
    b"3 0 3 2\n"
    b"3 4 5 6\n"
    b"3 4 6 7\n"
    b"3 0 1 5\n"
    b"3 0 5 4\n"
    b"3 1 2 6\n"
    b"3 1 6 5\n"
    b"3 2 3 7\n"
    b"3 2 7 6\n"
    b"3 3 0 4\n"
    b"3 3 4 7\n"
)


class _SequentialIds:
    def __init__(self) -> None:
        self._value = 0

    def __call__(self, prefix: str) -> str:
        self._value += 1
        return f"{prefix}:workflow-self-test-{self._value}"


@dataclass(frozen=True, slots=True)
class ArtifactWorkflowSelfTestResult:
    """Small immutable receipt returned after all temporary files are removed."""

    source_sha256: str
    document_sha256: str
    align_revision_id: str
    cutline_count: int
    outline_count: int
    rubbing_count: int
    svg_sha256: str
    png_sha256: str

    @property
    def record_count(self) -> int:
        return self.cutline_count + self.outline_count + self.rubbing_count

    def detail(self) -> str:
        return (
            f"workflow=Open>Align>Cutline {self.cutline_count}/3>"
            f"Outline {self.outline_count}/6>Rubbing {self.rubbing_count}/6, "
            f"records={self.record_count}, source={self.source_sha256[:12]}, "
            f"document={self.document_sha256[:12]}, "
            f"svg={self.svg_sha256[:12]}, png={self.png_sha256[:12]}"
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
    raise RuntimeError(f"unsupported self-test Cutline view: {view}")


def _publish_measurement(
    workbench: ArtifactWorkbench,
    transition: object,
) -> None:
    if not isinstance(transition, RecordBindingTransition):
        raise RuntimeError("measurement did not produce a record-binding transition")
    activation = workbench.activate_record_binding(transition)
    workbench.finalize_record_binding(activation)


def _measure_and_publish(
    controller: ArtifactMeasurementController,
    work_item: ArtifactMeasurementWorkItem,
) -> str:
    result = controller.execute(work_item)
    publication = controller.publish_result(
        work_item,
        result,
        lambda transition: _publish_measurement(controller.workbench, transition),
    )
    return publication.record_id


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} is not an object")
    return value


def _verification_evidence(
    report: Mapping[str, Any],
    *,
    artifact_kind: str,
) -> Mapping[str, Any]:
    if report.get("ok") is not True or report.get("artifact_kind") != artifact_kind:
        raise RuntimeError(f"unified offline verification failed: {report.get('error')!r}")
    return _require_mapping(report.get("evidence"), label="verification evidence")


def _validate_export_evidence(
    sidecar_bytes: bytes,
    *,
    session: ArtifactSession,
    record_id: str,
    source_sha256: str,
    physical_scale: str,
) -> None:
    try:
        sidecar = json.loads(sidecar_bytes.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("validated export sidecar is not strict UTF-8 JSON") from exc
    root = _require_mapping(sidecar, label="export sidecar")
    record = session.document.record_index[record_id]
    provenance = _require_mapping(root.get("provenance"), label="export provenance")
    assets = provenance.get("source_assets")
    if not isinstance(assets, list) or len(assets) != 1:
        raise RuntimeError("export provenance does not name exactly one source asset")
    asset = _require_mapping(assets[0], label="export source asset")
    if asset.get("sha256") != source_sha256:
        raise RuntimeError("export provenance lost the original source SHA-256")
    if root.get("recipe") != record.to_dict()["recipe"]:
        raise RuntimeError("export sidecar recipe does not match its durable record")
    qc = _require_mapping(root.get("qc"), label="export QC")
    if qc.get("record") != record.to_dict()["qc"]:
        raise RuntimeError("export sidecar QC does not match its durable record")
    scale = _require_mapping(qc.get("scale"), label="export scale QC")
    if scale.get("physical_scale") != physical_scale:
        raise RuntimeError("export sidecar does not carry the required 1:1 scale claim")


def _assert_progress_complete(session: ArtifactSession) -> tuple[int, int, int]:
    progress = derive_artifact_workflow_progress(session, align_ready=True)
    counts = (
        progress.cutline.completed_count,
        progress.outline.completed_count,
        progress.rubbing.completed_count,
    )
    if counts != (3, 6, 6):
        raise RuntimeError(f"reopened workflow is incomplete: {counts!r}")
    if not (
        progress.cutline.complete
        and progress.outline.complete
        and progress.rubbing.complete
    ):
        raise RuntimeError("reopened workflow completion flags are inconsistent")
    return counts


def _run_in_directory(directory: Path) -> ArtifactWorkflowSelfTestResult:
    source_path = directory / "workflow-fixture.ply"
    project_path = directory / "workflow-fixture.amr"
    source_path.write_bytes(_PLY_BYTES)
    source_sha256 = hashlib.sha256(_PLY_BYTES).hexdigest()

    ids = _SequentialIds()
    workbench = ArtifactWorkbench(id_factory=ids)
    ticket = workbench.begin_new_import(
        str(source_path),
        ConfirmedSourceMetadata(
            unit="mm",
            source_x="+X",
            source_y="+Y",
            source_z="+Z",
            handedness="right",
        ),
        software_version="packaged-self-test/1",
        operator=_OPERATOR,
        request_id="open:workflow-self-test",
        created_at=_STAMP,
        document_id="artifact:packaged-workflow-self-test",
        metadata_revision_id="metadata:workflow-self-test-mm",
        align_revision_id="align:workflow-self-test-initial",
    )
    mesh = MeshLoader(default_unit="mm").load(
        source_path,
        unit=ticket.source_unit,
        source_format=ticket.source_format,
        import_recipe=ticket.import_recipe,
        capture_dependencies=ticket.capture_dependencies,
    )
    if mesh.source_identity is None or mesh.source_identity.sha256 != source_sha256:
        raise RuntimeError("Open did not bind the exact fixture source bytes")
    opened = workbench.prepare_loaded_source(ticket, mesh)
    opened_activation = workbench.activate_projection(opened)
    opened_state = workbench.finalize_projection(opened_activation)
    if opened_state.stage is not WorkflowStage.ALIGN_REQUIRED:
        raise RuntimeError("Open did not require an explicit Align confirmation")

    aligned = workbench.prepare_align_commit(
        translation_mm=(0.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator=_OPERATOR,
        created_at=_STAMP,
        revision_id="align:workflow-self-test-explicit",
        transition_id="projection:workflow-self-test-align",
    )
    if aligned is None:
        raise RuntimeError("explicit Align confirmation did not create a revision")
    aligned_activation = workbench.activate_projection(aligned)
    aligned_state = workbench.finalize_projection(aligned_activation)
    if (
        aligned_state.stage is not WorkflowStage.MEASUREMENT_READY
        or not aligned_state.can_measure
    ):
        raise RuntimeError("explicit Align did not unlock measurement")

    measurements = ArtifactMeasurementController(workbench, id_factory=ids)
    cutline_ids: list[str] = []
    for view in REQUIRED_CUTLINE_VIEWS:
        item = measurements.begin_cutline(
            _cutline_frame(view),
            record_id=f"record:cutline:{view}:workflow-self-test",
            created_at=_STAMP,
            operator=_OPERATOR,
        )
        cutline_ids.append(_measure_and_publish(measurements, item))

    current = workbench.snapshot.session
    if not isinstance(current, ArtifactSession):
        raise RuntimeError("Cutline publication lost the active ArtifactSession")
    if workflow_step_record_ids(current, ArtifactWorkflowStep.CUTLINE) != tuple(
        cutline_ids
    ):
        raise RuntimeError("Cutline 3-view progress does not match published records")

    outline_ids: list[str] = []
    for view in REQUIRED_SIX_VIEWS:
        item = measurements.begin_outline(
            view,
            precision_grid_mm=0.01,
            record_id=f"record:outline:{view}:workflow-self-test",
            created_at=_STAMP,
            operator=_OPERATOR,
        )
        outline_ids.append(_measure_and_publish(measurements, item))

    current = workbench.snapshot.session
    if not isinstance(current, ArtifactSession):
        raise RuntimeError("Outline publication lost the active ArtifactSession")
    if workflow_step_record_ids(current, ArtifactWorkflowStep.OUTLINE) != tuple(
        outline_ids
    ):
        raise RuntimeError("Outline 6-view progress does not match published records")

    rubbing_ids: list[str] = []
    for view in REQUIRED_SIX_VIEWS:
        item = measurements.begin_rubbing(
            view,
            pixels_per_mm=2,
            margin_um=0,
            reference_radius_um=500,
            depth_quantization_um=10,
            black_point_um=100,
            ink_strength_percent=100,
            relief_polarity="bidirectional",
            record_id=f"record:rubbing:{view}:workflow-self-test",
            created_at=_STAMP,
            operator=_OPERATOR,
        )
        rubbing_ids.append(_measure_and_publish(measurements, item))

    current = workbench.snapshot.session
    if not isinstance(current, ArtifactSession):
        raise RuntimeError("Digital Rubbing publication lost the active ArtifactSession")
    counts = _assert_progress_complete(current)
    if workflow_step_record_ids(
        current,
        ArtifactWorkflowStep.DIGITAL_RUBBING,
    ) != tuple(rubbing_ids):
        raise RuntimeError("Digital Rubbing 6-view progress does not match records")

    save_artifact_session_project(project_path, current)
    source_path.unlink()
    if source_path.exists():
        raise RuntimeError("external fixture source could not be removed")
    restored = load_artifact_session_project(project_path)
    if restored.document.canonical_json_bytes() != current.document.canonical_json_bytes():
        raise RuntimeError("offline AMR reopen changed the completed ArtifactDocument")
    restored_counts = _assert_progress_complete(restored)
    if restored_counts != counts:
        raise RuntimeError("offline AMR reopen changed workflow progress")
    project_evidence = _verification_evidence(
        build_artifact_verification_report(project_path),
        artifact_kind="project",
    )
    if (
        project_evidence.get("document_sha256")
        != restored.document.canonical_sha256
        or project_evidence.get("embedded_source_materialized") is not True
    ):
        raise RuntimeError("unified AMR verification lost embedded-source authority")

    offline_workbench = ArtifactWorkbench(
        session=restored,
        project_path=str(project_path),
        id_factory=ids,
    )
    if not offline_workbench.snapshot.can_measure:
        raise RuntimeError("offline AMR reopen lost explicit Align authority")
    exports = ArtifactExportController(offline_workbench, id_factory=ids)

    vector_destination = directory / "cutline-top.amr-vector"
    vector_item = exports.begin_vector(vector_destination, cutline_ids[0])
    vector_result = exports.execute(vector_item)
    exports.publish_result(vector_item, vector_result)
    relocated_vector = directory / "relocated-cutline-top.amr-vector"
    vector_destination.rename(relocated_vector)
    vector_bundle = validate_vector_export_package(relocated_vector)
    vector_evidence = _verification_evidence(
        build_artifact_verification_report(
            relocated_vector,
            against_project=project_path,
        ),
        artifact_kind="vector_export",
    )
    if (
        vector_evidence.get("bound_project_document_sha256")
        != restored.document.canonical_sha256
        or vector_evidence.get("svg_sha256") != vector_bundle.svg_sha256
    ):
        raise RuntimeError("unified vector verification lost project binding")
    _validate_export_evidence(
        vector_bundle.sidecar_bytes,
        session=restored,
        record_id=cutline_ids[0],
        source_sha256=source_sha256,
        physical_scale="1:1",
    )

    rubbing_destination = directory / "rubbing-top.amr-rubbing"
    rubbing_item = exports.begin_rubbing(rubbing_destination, rubbing_ids[0])
    rubbing_result = exports.execute(rubbing_item)
    exports.publish_result(rubbing_item, rubbing_result)
    relocated_rubbing = directory / "relocated-rubbing-top.amr-rubbing"
    rubbing_destination.rename(relocated_rubbing)
    rubbing_bundle = validate_rubbing_export_package(relocated_rubbing)
    rubbing_evidence = _verification_evidence(
        build_artifact_verification_report(
            relocated_rubbing,
            against_project=project_path,
        ),
        artifact_kind="rubbing_export",
    )
    if (
        rubbing_evidence.get("bound_project_document_sha256")
        != restored.document.canonical_sha256
        or rubbing_evidence.get("png_sha256") != rubbing_bundle.png_sha256
    ):
        raise RuntimeError("unified rubbing verification lost project binding")
    _validate_export_evidence(
        rubbing_bundle.sidecar_bytes,
        session=restored,
        record_id=rubbing_ids[0],
        source_sha256=source_sha256,
        physical_scale="1:1_planar_sampling",
    )
    if rubbing_bundle.pixels_per_meter != 2_000:
        raise RuntimeError("Digital Rubbing export changed its 2 px/mm sampling scale")

    align_id = restored.document.active_align_revision_id
    if align_id != "align:workflow-self-test-explicit":
        raise RuntimeError("offline workflow changed the explicit Align revision")
    return ArtifactWorkflowSelfTestResult(
        source_sha256=source_sha256,
        document_sha256=restored.document.canonical_sha256,
        align_revision_id=align_id,
        cutline_count=counts[0],
        outline_count=counts[1],
        rubbing_count=counts[2],
        svg_sha256=vector_bundle.svg_sha256,
        png_sha256=rubbing_bundle.png_sha256,
    )


def run_artifact_workflow_self_test() -> ArtifactWorkflowSelfTestResult:
    """Execute the complete offline workflow in an auto-removed directory."""

    with tempfile.TemporaryDirectory(
        prefix="archmeshrubbing-workflow-self-test-"
    ) as temporary:
        return _run_in_directory(Path(temporary))


__all__ = [
    "ArtifactWorkflowSelfTestResult",
    "run_artifact_workflow_self_test",
]
