from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

import numpy as np
import pytest

from src.application.artifact_measurements import (
    ArtifactMeasurementController,
    ArtifactMeasurementError,
    ArtifactMeasurementResult,
    MeasurementCancelledError,
    MeasurementOperationKind,
    MeasurementOperationState,
    MeasurementResourceLimitError,
    StaleMeasurementOperationError,
    execute_measurement_work_item,
)
from src.application.artifact_workbench import (
    ArtifactWorkbench,
    ArtifactWorkbenchError,
    ConfirmedSourceMetadata,
    RecordBindingTransition,
    StaleWorkflowOperationError,
    WorkflowBusyError,
)
from src.application.artifact_workflow_progress import (
    ArtifactWorkflowStep,
    REQUIRED_CUTLINE_VIEWS,
    REQUIRED_SIX_VIEWS,
    workflow_step_record_ids,
)
from src.core.artifact_cancellation import ArtifactComputationCancelledError
from src.core.artifact_geometry_metrics import (
    ArtifactGeometryMetricsComputation,
    geometry_metrics_receipt_from_record,
)
from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_rubbing_extractor import (
    ArtifactRubbingComputation,
    RUBBING_ESTIMATED_PEAK_BYTES_PER_PIXEL,
    RUBBING_ESTIMATE_FIXED_OVERHEAD_BYTES,
    RUBBING_ESTIMATE_GEOMETRY_MULTIPLIER,
    RUBBING_ESTIMATE_MATERIALIZED_ATTRIBUTE_MULTIPLIER,
    estimate_digital_rubbing_resources,
)
from src.core.artifact_session import ArtifactSession
from src.core.artifact_surface_measurement import (
    ArtifactSurfaceMeasurementComputation,
    SURFACE_DIAMETER_RECORD_TYPE,
    SURFACE_DISTANCE_RECORD_TYPE,
    resolve_surface_anchor_from_ray,
    surface_measurement_receipt_from_record,
)
from src.core.artifact_vector_extractor import (
    ArtifactVectorComputation,
    commit_vector_computation,
    compute_artifact_cutline,
)
from src.core.artifact_vector_record import PlanarFrame
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


STAMP = "2026-07-12T00:00:00Z"
SOURCE_SHA = "b" * 64


class SequentialIds:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, prefix: str) -> str:
        self.value += 1
        return f"{prefix}:test-{self.value}"


def _box_mesh() -> MeshData:
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
        unit="cm",
        filepath=Path("/source/box.ply"),
        source_identity=SourceFingerprint(
            sha256=SOURCE_SHA,
            size_bytes=321,
            mtime_ns=1,
            original_name="box.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )


def _session(
    *,
    explicit_align: bool = True,
    source_mesh: MeshData | None = None,
    axes: dict[str, str] | None = None,
    handedness: str = "right",
) -> ArtifactSession:
    session = ArtifactSession.create_from_source(
        _box_mesh() if source_mesh is None else source_mesh,
        resolved_source_path="/source/box.ply",
        unit="cm",
        axes=(
            {"source_x": "+X", "source_y": "+Y", "source_z": "+Z"}
            if axes is None
            else axes
        ),
        handedness=handedness,
        software_version="0.1.0",
        operator="pytest",
        created_at=STAMP,
        document_id="artifact:measurement-test",
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


def _cutline_frame(view: str = "top") -> PlanarFrame:
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
    raise AssertionError(f"unsupported test Cutline view: {view}")


def _session_with_cutlines(*, source_mesh: MeshData | None = None) -> ArtifactSession:
    session = _session(source_mesh=source_mesh)
    for view in REQUIRED_CUTLINE_VIEWS:
        computation = compute_artifact_cutline(session, _cutline_frame(view))
        session = commit_vector_computation(
            session,
            computation,
            record_id=f"record:prerequisite:cutline:{view}",
            created_at=STAMP,
            operator="pytest",
        )
    return session


def _session_with_outlines(*, source_mesh: MeshData | None = None) -> ArtifactSession:
    session = _session_with_cutlines(source_mesh=source_mesh)
    cutline_ids = workflow_step_record_ids(
        session,
        ArtifactWorkflowStep.CUTLINE,
    )
    for view in REQUIRED_SIX_VIEWS:
        computation = compute_artifact_outline(
            session,
            view,
            precision_grid_mm=0.01,
        )
        session = commit_vector_computation(
            session,
            computation,
            record_id=f"record:prerequisite:outline:{view}",
            created_at=STAMP,
            operator="pytest",
            depends_on_record_ids=cutline_ids,
        )
    return session


def _headless_publisher(workbench: ArtifactWorkbench):
    def publish(transition) -> None:
        if isinstance(transition, RecordBindingTransition):
            activation = workbench.activate_record_binding(transition)
            workbench.finalize_record_binding(activation)
        else:
            activation = workbench.activate_projection(transition)
            workbench.finalize_projection(activation)

    return publish


def _begin_rubbing(
    controller: ArtifactMeasurementController,
    *,
    record_id: str = "record:rubbing:test",
):
    return controller.begin_rubbing(
        "top",
        pixels_per_mm=1,
        margin_um=1_000,
        reference_radius_um=1_000,
        depth_quantization_um=10,
        black_point_um=250,
        ink_strength_percent=100,
        relief_polarity="bidirectional",
        record_id=record_id,
        created_at=STAMP,
        operator="pytest",
    )


def _surface_anchor(
    session: ArtifactSession,
    point_world_mm: tuple[float, float, float],
) -> dict[str, object]:
    projection = session.materialize()
    point = np.asarray(point_world_mm, dtype=np.float64)
    return resolve_surface_anchor_from_ray(
        projection.mesh.vertices,
        projection.mesh.faces,
        source_faces=session.source_mesh.faces,
        ray_origin_world_mm=point + np.asarray([0.0, 0.0, 50.0]),
        ray_direction_world=[0.0, 0.0, -1.0],
        depth_point_world_mm=point,
        pixel_footprint_um=10,
    )


def test_begin_requires_explicit_align_before_reserving_work() -> None:
    workbench = ArtifactWorkbench(session=_session(explicit_align=False))
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())

    with pytest.raises(ArtifactWorkbenchError, match="explicit Align"):
        controller.begin_cutline(_cutline_frame(), record_id="record:blocked")

    blocked_anchor = {
        "barycentric_numerators": [1_000_000_000, 0, 0],
        "capture_point_grid": [-10_000, -10_000, 10_000],
        "depth_match_tolerance_um": 50,
        "depth_search_offset_px": [0, 0],
        "face_index": 2,
        "face_vertex_indices": [4, 5, 6],
        "pixel_footprint_um": 10,
    }
    with pytest.raises(ArtifactWorkbenchError, match="explicit Align"):
        controller.begin_surface_distance(
            [blocked_anchor, {**blocked_anchor, "barycentric_numerators": [0, 1_000_000_000, 0]}],
            record_id="record:surface-blocked",
        )

    assert controller.active_summaries == ()
    assert workbench.snapshot.session is not None
    assert workbench.snapshot.session.document.records == ()


def test_begin_enforces_sequence_and_captures_canonical_prerequisites() -> None:
    empty_controller = ArtifactMeasurementController(
        ArtifactWorkbench(session=_session()),
        id_factory=SequentialIds(),
    )
    with pytest.raises(ArtifactMeasurementError, match="Outline requires"):
        empty_controller.begin_outline(
            "top",
            precision_grid_mm=0.01,
            record_id="record:outline:too-early",
        )
    with pytest.raises(ArtifactMeasurementError, match="Digital Rubbing requires"):
        _begin_rubbing(empty_controller, record_id="record:rubbing:too-early")
    assert empty_controller.active_summaries == ()

    cutline_session = _session_with_cutlines()
    cutline_controller = ArtifactMeasurementController(
        ArtifactWorkbench(session=cutline_session),
        id_factory=SequentialIds(),
    )
    outline = cutline_controller.begin_outline(
        "top",
        precision_grid_mm=0.01,
        record_id="record:outline:with-dependencies",
    )
    assert outline.depends_on_record_ids == workflow_step_record_ids(
        cutline_session,
        ArtifactWorkflowStep.CUTLINE,
    )
    cutline_controller.cancel(outline)

    outline_session = _session_with_outlines()
    outline_controller = ArtifactMeasurementController(
        ArtifactWorkbench(session=outline_session),
        id_factory=SequentialIds(),
    )
    rubbing = _begin_rubbing(
        outline_controller,
        record_id="record:rubbing:with-dependencies",
    )
    assert rubbing.depends_on_record_ids == workflow_step_record_ids(
        outline_session,
        ArtifactWorkflowStep.OUTLINE,
    )
    outline_controller.cancel(rubbing)


def test_begin_tile_unwrap_captures_explicit_seam_in_canonical_recipe() -> None:
    session = _session()
    controller = ArtifactMeasurementController(
        ArtifactWorkbench(session=session),
        id_factory=SequentialIds(),
    )

    work_item = controller.begin_tile_unwrap(
        longitudinal_axis="y",
        record_view="top",
        selected_face_indices=(5, 1, 3),
        n_sections=24,
        seam_angle_microdegrees=12_345_678,
        record_id="record:tile:fixed-seam",
        created_at=STAMP,
        operator="pytest",
    )

    recipe = work_item.recipe_dict()
    assert recipe["seam_angle_microdegrees"] == 12_345_678
    selection = recipe["selection"]
    assert isinstance(selection, dict)
    assert selection["face_ranges"] == [[1, 2], [3, 4], [5, 6]]
    assert selection["selected_face_count"] == 3
    assert work_item.context.selection_hash == selection["selection_sha256"]
    controller.cancel(work_item)


def test_cutline_executes_and_publishes_only_the_reserved_record_id() -> None:
    session = _session()
    workbench = ArtifactWorkbench(session=session, id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:cutline:reserved",
        created_at=STAMP,
        operator="pytest",
    )

    result = controller.execute(item)
    assert isinstance(result.computation, ArtifactVectorComputation)
    active = workbench.snapshot.session
    assert isinstance(active, ArtifactSession)
    assert active is session
    assert active.document.records == ()

    publication = controller.publish_result(
        item,
        result,
        _headless_publisher(workbench),
    )

    assert publication.record_id == "record:cutline:reserved"
    assert workbench.snapshot.session is publication.session
    assert set(publication.session.document.record_index) == {
        "record:cutline:reserved"
    }
    assert (
        publication.session.document.record_freshness(publication.record_id).value
        == "fresh"
    )
    assert controller.summary(item).state is MeasurementOperationState.COMPLETED


def test_geometry_metrics_executes_and_publishes_guarded_exact_record() -> None:
    session = _session()
    workbench = ArtifactWorkbench(session=session, id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_geometry_metrics(
        coordinate_grid_um=1,
        record_id="record:geometry-metrics:reserved",
        created_at=STAMP,
        operator="pytest",
    )

    assert item.kind is MeasurementOperationKind.GEOMETRY_METRICS
    result = controller.execute(item)
    assert isinstance(result.computation, ArtifactGeometryMetricsComputation)
    receipt = result.computation.receipt_dict()
    assert receipt["surface_area"]["decimal_mm2"] == "2400.000000"
    assert receipt["volume"]["decimal_mm3"] == "8000.000000000"
    assert receipt["volume"]["exact_rational_mm3"] == {
        "denominator": "1",
        "numerator": "8000",
    }

    publication = controller.publish_result(
        item,
        result,
        _headless_publisher(workbench),
    )
    record = publication.session.document.record_index[publication.record_id]
    assert geometry_metrics_receipt_from_record(record) == receipt
    assert controller.summary(item).state is MeasurementOperationState.COMPLETED


def test_surface_distance_and_diameter_execute_publish_and_same_align_rebase() -> None:
    session = _session()
    workbench = ArtifactWorkbench(session=session, id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    distance_anchors = [
        _surface_anchor(session, (-5.0, -2.0, 10.0)),
        _surface_anchor(session, (5.0, -2.0, 10.0)),
    ]
    diameter_anchors = [
        _surface_anchor(session, (5.0, 0.0, 10.0)),
        _surface_anchor(session, (0.0, 5.0, 10.0)),
        _surface_anchor(session, (-5.0, 0.0, 10.0)),
        _surface_anchor(session, (0.0, -5.0, 10.0)),
    ]
    distance_item = controller.begin_surface_distance(
        distance_anchors,
        record_id="record:surface-distance:test",
        created_at=STAMP,
        operator="pytest",
    )
    diameter_item = controller.begin_surface_diameter(
        diameter_anchors,
        record_id="record:surface-diameter:test",
        created_at=STAMP,
        operator="pytest",
    )
    assert distance_item.kind is MeasurementOperationKind.SURFACE_DISTANCE
    assert diameter_item.kind is MeasurementOperationKind.SURFACE_DIAMETER
    assert distance_item.context.selection_hash is not None
    assert diameter_item.context.selection_hash is not None

    distance_result = controller.execute(distance_item)
    diameter_result = controller.execute(diameter_item)
    assert isinstance(distance_result.computation, ArtifactSurfaceMeasurementComputation)
    assert isinstance(diameter_result.computation, ArtifactSurfaceMeasurementComputation)

    controller.publish_result(
        distance_item,
        distance_result,
        _headless_publisher(workbench),
    )
    controller.publish_result(
        diameter_item,
        diameter_result,
        _headless_publisher(workbench),
    )
    current = workbench.snapshot.session
    assert isinstance(current, ArtifactSession)
    distance_record = current.document.record_index[distance_item.record_id]
    diameter_record = current.document.record_index[diameter_item.record_id]
    assert distance_record.type == SURFACE_DISTANCE_RECORD_TYPE
    assert diameter_record.type == SURFACE_DIAMETER_RECORD_TYPE
    assert surface_measurement_receipt_from_record(distance_record)["measurement"][
        "distance_mm_decimal"
    ] == "10.000000"
    assert surface_measurement_receipt_from_record(diameter_record)["measurement"][
        "diameter_mm_decimal"
    ] == "10.000000"
    assert controller.summary(distance_item).state is MeasurementOperationState.COMPLETED
    assert controller.summary(diameter_item).state is MeasurementOperationState.COMPLETED


def test_surface_result_becomes_stale_after_align_change() -> None:
    session = _session()
    workbench = ArtifactWorkbench(session=session, id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_surface_distance(
        [
            _surface_anchor(session, (-5.0, -2.0, 10.0)),
            _surface_anchor(session, (5.0, -2.0, 10.0)),
        ],
        record_id="record:surface-distance:stale",
        created_at=STAMP,
        operator="pytest",
    )
    result = controller.execute(item)
    changed = workbench.prepare_align_commit(
        translation_mm=(1.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:surface-stale",
    )
    assert changed is not None
    _headless_publisher(workbench)(changed)

    with pytest.raises(StaleMeasurementOperationError, match="stale"):
        controller.publish_result(item, result, _headless_publisher(workbench))
    current = workbench.snapshot.session
    assert isinstance(current, ArtifactSession)
    assert item.record_id not in current.document.record_index
    assert controller.summary(item).state is MeasurementOperationState.STALE


def test_surface_distance_reopens_with_reflected_projected_face_winding() -> None:
    session = _session(
        axes={"source_x": "-X", "source_y": "+Y", "source_z": "+Z"},
        handedness="left",
    )
    projection = session.materialize()
    assert np.array_equal(
        projection.mesh.faces,
        np.asarray(session.source_mesh.faces)[:, [0, 2, 1]],
    )
    controller = ArtifactMeasurementController(ArtifactWorkbench(session=session))
    item = controller.begin_surface_distance(
        [
            _surface_anchor(session, (-5.0, -2.0, 10.0)),
            _surface_anchor(session, (5.0, -2.0, 10.0)),
        ],
        record_id="record:surface-distance:reflected",
        created_at=STAMP,
        operator="pytest",
    )
    result = controller.execute(item)
    publication = controller.publish_result(
        item,
        result,
        _headless_publisher(controller.workbench),
    )
    receipt = surface_measurement_receipt_from_record(
        publication.session.document.record_index[item.record_id]
    )
    assert receipt["measurement"]["distance_mm_decimal"] == "10.000000"
    rebound = ArtifactSession.bind_loaded_document(
        publication.session.document,
        session.source_mesh,
        resolved_source_path=session.resolved_source_path,
    )
    assert rebound.document.record_index[item.record_id].type == (
        SURFACE_DISTANCE_RECORD_TYPE
    )


def test_same_align_parallel_results_rebase_without_lost_updates() -> None:
    session = _session_with_cutlines()
    initial_record_ids = set(session.document.record_index)
    expected_outline_dependencies = workflow_step_record_ids(
        session,
        ArtifactWorkflowStep.CUTLINE,
    )
    source_mesh = session.source_mesh
    workbench = ArtifactWorkbench(session=session, id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    cutline = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:z-cutline",
        created_at=STAMP,
        operator="pytest",
    )
    outline = controller.begin_outline(
        "top",
        precision_grid_mm=0.01,
        record_id="record:a-outline",
        created_at=STAMP,
        operator="pytest",
    )
    assert outline.depends_on_record_ids == expected_outline_dependencies
    cutline_result = controller.execute(cutline)
    outline_result = controller.execute(outline)

    controller.publish_result(
        cutline,
        cutline_result,
        _headless_publisher(workbench),
    )
    controller.publish_result(
        outline,
        outline_result,
        _headless_publisher(workbench),
    )

    current = workbench.snapshot.session
    assert current is not None
    assert current.source_mesh is source_mesh
    assert set(current.document.record_index) == initial_record_ids | {
        "record:z-cutline",
        "record:a-outline",
    }
    assert all(
        current.document.record_freshness(record_id).value == "fresh"
        for record_id in current.document.record_index
    )


def test_align_change_makes_late_result_stale_without_adding_a_record() -> None:
    workbench = ArtifactWorkbench(session=_session(), id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:late",
        created_at=STAMP,
        operator="pytest",
    )
    result = controller.execute(item)

    align_transition = workbench.prepare_align_commit(
        translation_mm=(1.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:changed",
    )
    assert align_transition is not None
    _headless_publisher(workbench)(align_transition)

    with pytest.raises(StaleMeasurementOperationError, match="stale"):
        controller.publish_result(item, result, _headless_publisher(workbench))

    current = workbench.snapshot.session
    assert current is not None
    assert "record:late" not in current.document.record_index
    assert controller.summary(item).state is MeasurementOperationState.STALE


def test_align_change_then_undo_does_not_revive_a_revoked_result() -> None:
    workbench = ArtifactWorkbench(session=_session(), id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:no-revival",
        created_at=STAMP,
        operator="pytest",
    )
    result = controller.execute(item)

    changed = workbench.prepare_align_commit(
        translation_mm=(2.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:temporary-change",
    )
    assert changed is not None
    _headless_publisher(workbench)(changed)
    assert controller.summary(item).state is MeasurementOperationState.STALE

    undo = workbench.prepare_activate_parent_align()
    _headless_publisher(workbench)(undo)
    current = workbench.snapshot.session
    assert current is not None
    assert current.document.active_align_revision_id == "align:confirmed"
    with pytest.raises(StaleMeasurementOperationError, match="stale"):
        controller.publish_result(item, result, _headless_publisher(workbench))
    assert "record:no-revival" not in current.document.record_index


def test_cancel_revokes_a_computed_result_and_releases_record_reservation() -> None:
    workbench = ArtifactWorkbench(session=_session(), id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:reusable",
        created_at=STAMP,
        operator="pytest",
    )
    result = controller.execute(item)
    summary = controller.cancel(item)
    assert summary.state is MeasurementOperationState.CANCELLED

    with pytest.raises(StaleMeasurementOperationError, match="stale"):
        controller.publish_result(item, result, _headless_publisher(workbench))

    replacement = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:reusable",
        created_at=STAMP,
        operator="pytest",
    )
    assert replacement is not item
    controller.cancel(replacement)


def test_duplicate_active_record_reservation_is_atomic() -> None:
    workbench = ArtifactWorkbench(session=_session_with_cutlines())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    first = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:collision",
    )

    with pytest.raises(ArtifactMeasurementError, match="reserved"):
        controller.begin_outline(
            "front",
            precision_grid_mm=0.01,
            record_id="record:collision",
        )

    assert len(controller.active_summaries) == 1
    assert controller.active_summaries[0].operation_id == first.id
    controller.cancel(first)


def test_controllers_share_record_reservations_for_one_workbench() -> None:
    workbench = ArtifactWorkbench(session=_session_with_cutlines())
    first = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    second = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = first.begin_cutline(
        _cutline_frame(),
        record_id="record:shared-reservation",
        created_at=STAMP,
    )

    with pytest.raises(ArtifactMeasurementError, match="is reserved by"):
        second.begin_outline(
            "top",
            precision_grid_mm=0.01,
            record_id="record:shared-reservation",
            created_at=STAMP,
        )

    assert second.active_summaries == first.active_summaries
    first.cancel(item)
    replacement = second.begin_outline(
        "top",
        precision_grid_mm=0.01,
        record_id="record:shared-reservation",
        created_at=STAMP,
    )
    second.cancel(replacement)


def test_record_reservation_rejects_every_existing_durable_namespace() -> None:
    session = _session()
    durable_ids = (
        next(iter(session.document.source_asset_index)),
        next(iter(session.document.geometry_revision_index)),
        next(iter(session.document.source_metadata_revision_index)),
        next(iter(session.document.align_revision_index)),
    )
    workbench = ArtifactWorkbench(session=session)
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())

    for durable_id in durable_ids:
        with pytest.raises(ArtifactMeasurementError, match="durable document ID"):
            controller.begin_cutline(
                _cutline_frame(),
                record_id=durable_id,
            )

    assert controller.active_summaries == ()


def test_forged_operation_id_is_rejected_without_consuming_valid_result() -> None:
    workbench = ArtifactWorkbench(session=_session(), id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:forgery-test",
        created_at=STAMP,
        operator="pytest",
    )
    result = controller.execute(item)
    forged = replace(result, operation_id="operation:other")

    with pytest.raises(ArtifactMeasurementError, match="exact result capability"):
        controller.publish_result(item, forged, _headless_publisher(workbench))

    assert controller.summary(item).state is MeasurementOperationState.RUNNING
    controller.publish_result(item, result, _headless_publisher(workbench))
    assert controller.summary(item).state is MeasurementOperationState.COMPLETED


def test_standalone_computation_cannot_forge_controller_result_capability() -> None:
    workbench = ArtifactWorkbench(session=_session(), id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:capability",
        created_at=STAMP,
        operator="pytest",
    )
    untrusted = execute_measurement_work_item(item)

    with pytest.raises(ArtifactMeasurementError, match="exact result capability"):
        controller.publish_result(item, untrusted, _headless_publisher(workbench))

    trusted = controller.execute(item)
    controller.publish_result(item, trusted, _headless_publisher(workbench))
    assert controller.summary(item).state is MeasurementOperationState.COMPLETED


def test_invalid_created_at_fails_before_registration() -> None:
    workbench = ArtifactWorkbench(session=_session())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())

    with pytest.raises(ArtifactMeasurementError, match="canonical UTC seconds"):
        controller.begin_cutline(
            _cutline_frame(),
            record_id="record:bad-time",
            created_at="not-a-timestamp",
        )

    assert controller.active_summaries == ()


def test_publication_rollback_keeps_result_retryable_and_reserved() -> None:
    workbench = ArtifactWorkbench(session=_session(), id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:retry",
        created_at=STAMP,
        operator="pytest",
    )
    result = controller.execute(item)

    def rejected(_transition) -> None:
        raise RuntimeError("injected scene preparation failure")

    with pytest.raises(RuntimeError, match="injected"):
        controller.publish_result(item, result, rejected)

    assert controller.summary(item).state is MeasurementOperationState.RUNNING
    controller.publish_result(item, result, _headless_publisher(workbench))
    assert controller.summary(item).state is MeasurementOperationState.COMPLETED


def test_pending_open_keeps_result_retryable_until_open_is_resolved() -> None:
    workbench = ArtifactWorkbench(session=_session(), id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:pending-open-retry",
        created_at=STAMP,
        operator="pytest",
    )
    result = controller.execute(item)
    ticket = workbench.begin_new_import(
        "/source/next.ply",
        ConfirmedSourceMetadata(
            unit="cm",
            source_x="+X",
            source_y="+Y",
            source_z="+Z",
            handedness="right",
        ),
        software_version="0.1.0",
        operator="pytest",
    )

    with pytest.raises(WorkflowBusyError, match="Open request is pending"):
        controller.publish_result(item, result, _headless_publisher(workbench))

    assert controller.summary(item).state is MeasurementOperationState.RUNNING
    workbench.cancel_load(ticket)
    with pytest.raises(ArtifactMeasurementError, match="is reserved by"):
        controller.begin_cutline(
            _cutline_frame("front"),
            record_id="record:pending-open-retry",
            created_at=STAMP,
        )

    controller.publish_result(item, result, _headless_publisher(workbench))
    assert controller.summary(item).state is MeasurementOperationState.COMPLETED


def test_same_align_external_commit_causes_automatic_rebase_retry() -> None:
    session = _session_with_cutlines()
    initial_record_ids = set(session.document.record_index)
    workbench = ArtifactWorkbench(session=session, id_factory=SequentialIds())
    first = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    external = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    first_item = first.begin_cutline(
        _cutline_frame(),
        record_id="record:first",
        created_at=STAMP,
        operator="pytest",
    )
    external_item = external.begin_outline(
        "top",
        precision_grid_mm=0.01,
        record_id="record:external",
        created_at=STAMP,
        operator="pytest",
    )
    first_result = first.execute(first_item)
    external_result = external.execute(external_item)
    calls = 0

    def racing_publisher(transition) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            external.publish_result(
                external_item,
                external_result,
                _headless_publisher(workbench),
            )
        _headless_publisher(workbench)(transition)

    first.publish_result(first_item, first_result, racing_publisher)

    current = workbench.snapshot.session
    assert current is not None
    assert calls == 2
    assert set(current.document.record_index) == initial_record_ids | {
        "record:first",
        "record:external",
    }


def test_publisher_leaving_tentative_authority_fails_closed_without_busy_lock() -> None:
    workbench = ArtifactWorkbench(session=_session(), id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:tentative-fault",
        created_at=STAMP,
        operator="pytest",
    )
    result = controller.execute(item)

    def broken_publisher(transition) -> None:
        workbench.activate_record_binding(transition)
        raise RuntimeError("publisher crashed after activation")

    with pytest.raises(RuntimeError, match="publisher crashed"):
        controller.publish_result(item, result, broken_publisher)

    assert workbench.snapshot.faulted
    assert not workbench.snapshot.tentative
    assert controller.summary(item).state is MeasurementOperationState.FAILED
    recovery = workbench.begin_new_import(
        "/source/recovery.ply",
        ConfirmedSourceMetadata(
            unit="cm",
            source_x="+X",
            source_y="+Y",
            source_z="+Z",
            handedness="right",
        ),
        software_version="0.1.0",
        operator="pytest",
    )
    workbench.cancel_load(recovery)


def test_unrelated_tentative_transition_is_preserved_and_result_remains_retryable() -> None:
    workbench = ArtifactWorkbench(session=_session(), id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:wait-for-align",
        created_at=STAMP,
        operator="pytest",
    )
    result = controller.execute(item)
    align = workbench.prepare_align_commit(
        translation_mm=(3.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at=STAMP,
        revision_id="align:external-tentative",
    )
    assert align is not None
    activation = workbench.activate_projection(align)

    with pytest.raises(WorkflowBusyError, match="publication"):
        controller.publish_result(item, result, _headless_publisher(workbench))

    assert workbench.snapshot is activation.current
    assert workbench.snapshot.tentative
    assert not workbench.snapshot.faulted
    assert controller.summary(item).state is MeasurementOperationState.RUNNING
    workbench.rollback_projection(activation, RuntimeError("external scene rejected"))
    controller.publish_result(item, result, _headless_publisher(workbench))
    assert controller.summary(item).state is MeasurementOperationState.COMPLETED


def test_exception_after_finalize_reports_committed_operation_as_completed() -> None:
    workbench = ArtifactWorkbench(session=_session(), id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:committed-before-error",
        created_at=STAMP,
        operator="pytest",
    )
    result = controller.execute(item)

    def publish_then_raise(transition) -> None:
        _headless_publisher(workbench)(transition)
        raise StaleWorkflowOperationError("late callback error")

    publication = controller.publish_result(item, result, publish_then_raise)

    current = workbench.snapshot.session
    assert current is not None
    assert publication.record_id == "record:committed-before-error"
    assert "record:committed-before-error" in current.document.record_index
    assert controller.summary(item).state is MeasurementOperationState.COMPLETED


def test_open_started_after_finalize_does_not_reclassify_committed_result() -> None:
    workbench = ArtifactWorkbench(session=_session(), id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:before-open",
        created_at=STAMP,
        operator="pytest",
    )
    result = controller.execute(item)
    pending = []

    def publish_then_open(transition) -> None:
        _headless_publisher(workbench)(transition)
        pending.append(
            workbench.begin_new_import(
                "/source/next.ply",
                ConfirmedSourceMetadata(
                    unit="cm",
                    source_x="+X",
                    source_y="+Y",
                    source_z="+Z",
                    handedness="right",
                ),
                software_version="0.1.0",
                operator="pytest",
            )
        )

    publication = controller.publish_result(item, result, publish_then_open)

    assert publication.record_id == "record:before-open"
    assert controller.summary(item).state is MeasurementOperationState.COMPLETED
    assert workbench.snapshot.pending_load is pending[0]
    assert "record:before-open" in publication.session.document.record_index
    workbench.cancel_load(pending[0])


def test_align_finalized_inside_publisher_cannot_return_false_success() -> None:
    workbench = ArtifactWorkbench(session=_session(), id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:align-race",
        created_at=STAMP,
        operator="pytest",
    )
    result = controller.execute(item)

    def publish_then_align(transition) -> None:
        _headless_publisher(workbench)(transition)
        changed = workbench.prepare_align_commit(
            translation_mm=(3.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            pivot_mm=(0.0, 0.0, 0.0),
            operator="pytest",
            created_at=STAMP,
            revision_id="align:after-measurement",
        )
        assert changed is not None
        _headless_publisher(workbench)(changed)

    with pytest.raises(StaleMeasurementOperationError, match="finalized"):
        controller.publish_result(item, result, publish_then_align)

    current = workbench.snapshot.session
    assert isinstance(current, ArtifactSession)
    assert current.document.active_align_revision_id == "align:after-measurement"
    assert current.document.record_freshness(item.record_id).value == "stale_alignment"
    assert controller.summary(item).state is MeasurementOperationState.STALE


def test_rubbing_preflight_matches_raster_and_enforces_memory_budget() -> None:
    workbench = ArtifactWorkbench(
        session=_session_with_outlines(),
        id_factory=SequentialIds(),
    )
    tiny_budget = ArtifactMeasurementController(
        workbench,
        id_factory=SequentialIds(),
        rubbing_memory_budget_bytes=1,
    )
    with pytest.raises(MeasurementResourceLimitError, match="memory estimate"):
        _begin_rubbing(tiny_budget)
    assert tiny_budget.active_summaries == ()

    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    with patch(
        "src.application.artifact_measurements.estimate_digital_rubbing_resources",
        wraps=estimate_digital_rubbing_resources,
    ) as estimate_preflight:
        item = _begin_rubbing(controller)
        estimate_preflight.assert_not_called()
        assert item.resource_estimate is None
        minimum_reservation = controller.summary(item).estimated_peak_bytes
        result = controller.execute(item)
        estimate_preflight.assert_called_once()
    assert isinstance(result.computation, ArtifactRubbingComputation)
    estimate = controller.rubbing_resource_estimate(item)
    assert estimate is not None
    assert estimate.width_pixels == result.computation.raster.width_pixels
    assert estimate.height_pixels == result.computation.raster.height_pixels
    assert estimate.pixel_count == (
        result.computation.raster.width_pixels
        * result.computation.raster.height_pixels
    )
    assert estimate.estimated_peak_bytes > minimum_reservation

    publication = controller.publish_result(
        item,
        result,
        _headless_publisher(workbench),
    )
    assert publication.record_id == "record:rubbing:test"
    assert controller.summary(item).estimated_peak_bytes > 32 * 1024 * 1024


def test_rubbing_texture_copy_is_budgeted_before_materialization() -> None:
    textured_mesh = _box_mesh()
    textured_mesh.uv_coords = np.zeros((8, 2), dtype=np.float64)
    textured_mesh.texture = np.zeros((64, 64, 4), dtype=np.uint8)
    textured_session = _session_with_outlines(source_mesh=textured_mesh)
    source_mesh = textured_session.source_mesh
    attribute_bytes = int(
        source_mesh.uv_coords.nbytes + source_mesh.texture.nbytes  # type: ignore[union-attr]
    )
    attribute_peak_bytes = (
        attribute_bytes * RUBBING_ESTIMATE_MATERIALIZED_ATTRIBUTE_MULTIPLIER
    )
    minimum_without_attributes = (
        RUBBING_ESTIMATE_FIXED_OVERHEAD_BYTES
        + RUBBING_ESTIMATED_PEAK_BYTES_PER_PIXEL
        + int(source_mesh.vertices.nbytes + source_mesh.faces.nbytes)
        * RUBBING_ESTIMATE_GEOMETRY_MULTIPLIER
    )
    constrained = ArtifactMeasurementController(
        ArtifactWorkbench(session=textured_session),
        id_factory=SequentialIds(),
        rubbing_memory_budget_bytes=(
            minimum_without_attributes + attribute_peak_bytes - 1
        ),
    )

    with patch.object(
        ArtifactSession,
        "materialize",
        side_effect=AssertionError("materialize must not run before budget rejection"),
    ) as materialize:
        with pytest.raises(MeasurementResourceLimitError, match="before projection"):
            _begin_rubbing(constrained)
    materialize.assert_not_called()
    assert constrained.active_summaries == ()

    plain = ArtifactMeasurementController(
        ArtifactWorkbench(session=_session_with_outlines()),
        id_factory=SequentialIds(),
    )
    textured = ArtifactMeasurementController(
        ArtifactWorkbench(session=textured_session),
        id_factory=SequentialIds(),
    )
    plain_item = _begin_rubbing(plain, record_id="record:rubbing:plain")
    with patch.object(
        ArtifactSession,
        "materialize",
        side_effect=AssertionError("begin must finish admission before materialization"),
    ) as materialize:
        textured_item = _begin_rubbing(
            textured,
            record_id="record:rubbing:textured",
        )
    materialize.assert_not_called()
    assert plain_item.resource_estimate is None
    assert textured_item.resource_estimate is None
    plain.execute(plain_item)
    textured.execute(textured_item)
    plain_estimate = plain.rubbing_resource_estimate(plain_item)
    textured_estimate = textured.rubbing_resource_estimate(textured_item)
    assert plain_estimate is not None
    assert textured_estimate is not None
    assert (
        textured_estimate.estimated_peak_bytes
        - plain_estimate.estimated_peak_bytes
        == attribute_peak_bytes
    )
    plain.cancel(plain_item)
    textured.cancel(textured_item)


def test_only_one_active_rubbing_owns_the_raster_budget() -> None:
    workbench = ArtifactWorkbench(session=_session_with_outlines())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    first = _begin_rubbing(controller, record_id="record:rubbing:first")

    with pytest.raises(MeasurementResourceLimitError, match="already owns"):
        _begin_rubbing(controller, record_id="record:rubbing:second")

    controller.cancel(first)
    second = _begin_rubbing(controller, record_id="record:rubbing:second")
    controller.cancel(second)


def test_controllers_share_rubbing_admission_for_one_workbench() -> None:
    workbench = ArtifactWorkbench(session=_session_with_outlines())
    ids = SequentialIds()
    first = ArtifactMeasurementController(workbench, id_factory=ids)
    second = ArtifactMeasurementController(
        workbench,
        id_factory=ids,
        max_active_rubbing_operations=2,
    )
    item = _begin_rubbing(first, record_id="record:rubbing:shared-first")

    with pytest.raises(MeasurementResourceLimitError, match="already owns"):
        _begin_rubbing(second, record_id="record:rubbing:shared-second")

    assert second.active_summaries == first.active_summaries
    first.cancel(item)
    replacement = _begin_rubbing(
        second,
        record_id="record:rubbing:shared-second",
    )
    second.cancel(replacement)


def test_cancelled_running_rubbing_keeps_memory_slot_until_worker_exits() -> None:
    workbench = ArtifactWorkbench(session=_session_with_outlines())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = _begin_rubbing(controller, record_id="record:rubbing:blocked")
    started = Event()
    release = Event()
    errors: list[BaseException] = []

    def blocked(_item, *, cancellation_probe):
        started.set()
        if not release.wait(timeout=2.0):
            raise RuntimeError("test worker release timed out")
        if cancellation_probe():
            raise MeasurementCancelledError("cancelled in blocked extractor")
        raise AssertionError("blocked worker should have been cancelled")

    def run() -> None:
        try:
            controller.execute(item)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    with patch(
        "src.application.artifact_measurements.execute_measurement_work_item",
        side_effect=blocked,
    ):
        worker = Thread(target=run, daemon=True)
        worker.start()
        assert started.wait(timeout=2.0)
        cancelling = controller.cancel(item)
        assert cancelling.state is MeasurementOperationState.CANCELLING
        with pytest.raises(MeasurementResourceLimitError, match="already owns"):
            _begin_rubbing(controller, record_id="record:rubbing:too-early")
        release.set()
        worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], MeasurementCancelledError)
    assert controller.summary(item).state is MeasurementOperationState.CANCELLED
    replacement = _begin_rubbing(
        controller,
        record_id="record:rubbing:after-worker-exit",
    )
    controller.cancel(replacement)


def test_cancel_during_worker_rubbing_preflight_releases_only_after_estimator_exits(
) -> None:
    workbench = ArtifactWorkbench(session=_session_with_outlines())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = _begin_rubbing(
        controller,
        record_id="record:rubbing:blocked-preflight",
    )
    started = Event()
    release = Event()
    errors: list[BaseException] = []

    def blocked_estimate(*args, **kwargs):
        started.set()
        if not release.wait(timeout=2.0):
            raise RuntimeError("test preflight release timed out")
        return estimate_digital_rubbing_resources(*args, **kwargs)

    def run() -> None:
        try:
            controller.execute(item)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    with patch(
        "src.application.artifact_measurements.estimate_digital_rubbing_resources",
        side_effect=blocked_estimate,
    ):
        worker = Thread(target=run, daemon=True)
        worker.start()
        assert started.wait(timeout=2.0)
        cancelling = controller.cancel(item)
        assert cancelling.state is MeasurementOperationState.CANCELLING
        assert controller.rubbing_resource_estimate(item) is None
        with pytest.raises(MeasurementResourceLimitError, match="already owns"):
            _begin_rubbing(
                controller,
                record_id="record:rubbing:preflight-still-owned",
            )
        release.set()
        worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], MeasurementCancelledError)
    assert controller.summary(item).state is MeasurementOperationState.CANCELLED
    replacement = _begin_rubbing(
        controller,
        record_id="record:rubbing:after-preflight-exit",
    )
    controller.cancel(replacement)


def test_worker_preflight_failure_is_terminal_and_releases_record_reservation() -> None:
    workbench = ArtifactWorkbench(session=_session())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:preflight-failure",
    )

    def fail_preflight() -> None:
        raise RuntimeError("canonical scene mismatch")

    with pytest.raises(RuntimeError, match="canonical scene mismatch"):
        controller.execute(item, preflight=fail_preflight)

    summary = controller.summary(item)
    assert summary.state is MeasurementOperationState.FAILED
    assert summary.message == "canonical scene mismatch"
    assert controller.active_summaries == ()
    replacement = controller.begin_cutline(
        _cutline_frame(),
        record_id=item.record_id,
    )
    controller.cancel(replacement)


def test_worker_failure_outranks_cancel_requested_during_execution() -> None:
    workbench = ArtifactWorkbench(session=_session())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:cancel-vs-failure",
    )
    started = Event()
    release = Event()
    errors: list[BaseException] = []

    def failing_worker(_item, *, cancellation_probe):
        started.set()
        assert release.wait(timeout=2.0)
        assert cancellation_probe()
        raise RuntimeError("worker failed after cancellation was requested")

    def run() -> None:
        try:
            controller.execute(item)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    with patch(
        "src.application.artifact_measurements.execute_measurement_work_item",
        side_effect=failing_worker,
    ):
        worker = Thread(target=run, daemon=True)
        worker.start()
        assert started.wait(timeout=2.0)
        assert (
            controller.cancel(item).state
            is MeasurementOperationState.CANCELLING
        )
        release.set()
        worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "worker failed after cancellation was requested"
    summary = controller.summary(item)
    assert summary.state is MeasurementOperationState.FAILED
    assert summary.error_type == "RuntimeError"
    assert summary.message == "worker failed after cancellation was requested"
    assert workbench.snapshot.session is not None
    assert workbench.snapshot.session.document.records == ()

    replacement = controller.begin_cutline(
        _cutline_frame(),
        record_id=item.record_id,
    )
    controller.cancel(replacement)


def test_worker_failure_outranks_stale_requested_during_execution() -> None:
    workbench = ArtifactWorkbench(session=_session(), id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:stale-vs-failure",
    )
    started = Event()
    release = Event()
    errors: list[BaseException] = []

    def failing_worker(_item, *, cancellation_probe):
        started.set()
        assert release.wait(timeout=2.0)
        assert cancellation_probe()
        raise RuntimeError("worker failed after its Align became stale")

    def run() -> None:
        try:
            controller.execute(item)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    with patch(
        "src.application.artifact_measurements.execute_measurement_work_item",
        side_effect=failing_worker,
    ):
        worker = Thread(target=run, daemon=True)
        worker.start()
        assert started.wait(timeout=2.0)
        changed = workbench.prepare_align_commit(
            translation_mm=(1.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            pivot_mm=(0.0, 0.0, 0.0),
            operator="pytest",
            created_at=STAMP,
            revision_id="align:stale-vs-failure",
        )
        assert changed is not None
        _headless_publisher(workbench)(changed)
        assert (
            controller.summary(item).state
            is MeasurementOperationState.CANCELLING
        )
        release.set()
        worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "worker failed after its Align became stale"
    summary = controller.summary(item)
    assert summary.state is MeasurementOperationState.FAILED
    assert summary.error_type == "RuntimeError"
    assert summary.message == "worker failed after its Align became stale"
    assert workbench.snapshot.session is not None
    assert workbench.snapshot.session.document.records == ()

    replacement = controller.begin_cutline(
        _cutline_frame(),
        record_id=item.record_id,
    )
    controller.cancel(replacement)


def test_execute_claim_is_exactly_once_under_concurrency() -> None:
    workbench = ArtifactWorkbench(session=_session_with_outlines())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = _begin_rubbing(controller, record_id="record:rubbing:execute-once")
    original_execute = execute_measurement_work_item
    started = Event()
    release = Event()
    results = []
    errors: list[BaseException] = []
    calls = 0

    def blocked(work_item, *, cancellation_probe):
        nonlocal calls
        calls += 1
        started.set()
        if not release.wait(timeout=2.0):
            raise RuntimeError("test worker release timed out")
        return original_execute(
            work_item,
            cancellation_probe=cancellation_probe,
        )

    def run() -> None:
        try:
            results.append(controller.execute(item))
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    with patch(
        "src.application.artifact_measurements.execute_measurement_work_item",
        side_effect=blocked,
    ):
        worker = Thread(target=run, daemon=True)
        worker.start()
        assert started.wait(timeout=2.0)
        with pytest.raises(StaleMeasurementOperationError, match="already been executed"):
            controller.execute(item)
        release.set()
        worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert calls == 1
    assert len(results) == 1
    assert errors == []
    controller.cancel(item)


def test_cumulative_rubbing_estimates_cannot_overbook_configured_budget() -> None:
    workbench = ArtifactWorkbench(session=_session_with_outlines())
    ids = SequentialIds()
    probe = ArtifactMeasurementController(
        workbench,
        id_factory=ids,
        max_active_rubbing_operations=2,
    )
    probe_item = _begin_rubbing(probe, record_id="record:rubbing:probe")
    probe.execute(probe_item)
    estimate = probe.summary(probe_item).estimated_peak_bytes
    probe.cancel(probe_item)

    controller = ArtifactMeasurementController(
        workbench,
        id_factory=ids,
        max_active_rubbing_operations=2,
        rubbing_memory_budget_bytes=estimate * 2 - 1,
    )
    first = _begin_rubbing(controller, record_id="record:rubbing:budget-first")
    second = _begin_rubbing(controller, record_id="record:rubbing:budget-second")
    controller.execute(first)
    with pytest.raises(MeasurementResourceLimitError, match="cumulative"):
        controller.execute(second)
    assert len(controller.active_summaries) == 1
    controller.cancel(first)


def test_rubbing_admission_honors_every_active_controller_memory_budget() -> None:
    workbench = ArtifactWorkbench(session=_session_with_outlines())
    ids = SequentialIds()
    probe = ArtifactMeasurementController(
        workbench,
        id_factory=ids,
        max_active_rubbing_operations=2,
    )
    probe_item = _begin_rubbing(probe, record_id="record:rubbing:owner-probe")
    probe.execute(probe_item)
    estimate = probe.summary(probe_item).estimated_peak_bytes
    probe.cancel(probe_item)

    constrained = ArtifactMeasurementController(
        workbench,
        id_factory=ids,
        max_active_rubbing_operations=2,
        rubbing_memory_budget_bytes=estimate * 2 - 1,
    )
    permissive = ArtifactMeasurementController(
        workbench,
        id_factory=ids,
        max_active_rubbing_operations=3,
        rubbing_memory_budget_bytes=estimate * 3,
    )
    first = _begin_rubbing(
        constrained,
        record_id="record:rubbing:owner-budget-first",
    )
    second = _begin_rubbing(
        permissive,
        record_id="record:rubbing:owner-budget-second",
    )
    constrained.execute(first)

    with pytest.raises(MeasurementResourceLimitError, match="cumulative"):
        permissive.execute(second)

    assert permissive.active_summaries == constrained.active_summaries
    constrained.cancel(first)


def test_cancellation_probe_discards_result_at_safe_boundary() -> None:
    workbench = ArtifactWorkbench(session=_session())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:cancel-boundary",
    )
    probes = iter((False, False, True))

    with pytest.raises(MeasurementCancelledError, match="cancelled"):
        execute_measurement_work_item(item, cancellation_probe=lambda: next(probes))

    active = workbench.snapshot.session
    assert isinstance(active, ArtifactSession)
    assert active is item.captured_session
    assert active.document.records == ()
    controller.cancel(item)


@pytest.mark.parametrize(
    ("kind", "extractor_name"),
    (
        (MeasurementOperationKind.CUTLINE, "extract_cutline_geometry"),
        (MeasurementOperationKind.OUTLINE, "extract_outline_geometry"),
        (MeasurementOperationKind.DIGITAL_RUBBING, "extract_digital_rubbing"),
    ),
)
def test_controller_cancel_reaches_each_extractor_and_maps_core_signal(
    kind: MeasurementOperationKind,
    extractor_name: str,
) -> None:
    session = (
        _session_with_cutlines()
        if kind is MeasurementOperationKind.OUTLINE
        else (
            _session_with_outlines()
            if kind is MeasurementOperationKind.DIGITAL_RUBBING
            else _session()
        )
    )
    initial_records = session.document.records
    workbench = ArtifactWorkbench(session=session)
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    record_id = f"record:cooperative-cancel:{kind.value}"
    if kind is MeasurementOperationKind.CUTLINE:
        item = controller.begin_cutline(_cutline_frame(), record_id=record_id)
    elif kind is MeasurementOperationKind.OUTLINE:
        item = controller.begin_outline(
            "top",
            precision_grid_mm=0.01,
            record_id=record_id,
        )
    else:
        item = _begin_rubbing(controller, record_id=record_id)

    observed_probe = None

    def cancel_inside_extractor(*_args, **kwargs):
        nonlocal observed_probe
        observed_probe = kwargs.get("cancellation_probe")
        assert callable(observed_probe)
        controller.cancel(item)
        assert observed_probe()
        raise ArtifactComputationCancelledError("cancelled inside core extractor")

    with patch(
        f"src.application.artifact_measurements.{extractor_name}",
        side_effect=cancel_inside_extractor,
    ):
        with pytest.raises(MeasurementCancelledError, match="user_cancelled"):
            controller.execute(item)

    assert callable(observed_probe)
    summary = controller.summary(item)
    assert summary.state is MeasurementOperationState.CANCELLED
    assert workbench.snapshot.session is not None
    assert workbench.snapshot.session.document.records == initial_records


def test_core_cancellation_signal_is_mapped_at_the_application_boundary() -> None:
    workbench = ArtifactWorkbench(session=_session())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:core-cancel-mapping",
    )

    with patch(
        "src.application.artifact_measurements.extract_cutline_geometry",
        side_effect=ArtifactComputationCancelledError("core boundary cancelled"),
    ):
        with pytest.raises(MeasurementCancelledError, match="core boundary"):
            execute_measurement_work_item(item, cancellation_probe=lambda: False)

    controller.cancel(item)


def test_measurement_application_layer_has_no_qt_opengl_or_gui_imports() -> None:
    path = Path("src/application/artifact_measurements.py")
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


def test_invalid_cutline_recipe_fails_before_operation_registration() -> None:
    workbench = ArtifactWorkbench(session=_session())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())

    with pytest.raises(Exception, match="at least"):
        controller.begin_cutline(
            _cutline_frame(),
            classification_tolerance_mm=0.1,
            stitch_tolerance_mm=0.01,
            record_id="record:invalid",
        )

    assert controller.active_summaries == ()


def test_result_type_validation_is_eager() -> None:
    workbench = ArtifactWorkbench(session=_session())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_cutline(
        _cutline_frame(),
        record_id="record:type-check",
    )
    result = controller.execute(item)

    with pytest.raises(ArtifactMeasurementError, match="result kind"):
        ArtifactMeasurementResult(
            operation_id=item.id,
            kind="digital_rubbing",  # type: ignore[arg-type]
            computation=result.computation,
        )

    controller.cancel(item)


def test_condition_annotation_executes_and_publishes_the_painted_region() -> None:
    """A painted face set goes through the same worker path as every record."""

    from src.core.artifact_condition_annotation import (
        ConditionAnnotationComputation,
        condition_payload_from_record,
    )

    session = _session()
    workbench = ArtifactWorkbench(session=session, id_factory=SequentialIds())
    controller = ArtifactMeasurementController(workbench, id_factory=SequentialIds())
    item = controller.begin_condition_annotation(
        condition="restored",
        selected_face_indices=(3, 1, 2, 2),
        precision_grid_mm=0.01,
        record_id="record:condition:reserved",
        created_at=STAMP,
        operator="pytest",
    )

    assert item.kind is MeasurementOperationKind.CONDITION_ANNOTATION
    recipe = item.recipe_dict()
    assert recipe["kind"] == "condition_annotation"
    assert recipe["condition"] == "restored"
    selection = recipe["selection"]
    assert isinstance(selection, dict)
    # Canonical: sorted, deduplicated, merged.
    assert selection["face_ranges"] == [[1, 4]]
    assert item.context.selection_hash == selection["selection_sha256"]

    result = controller.execute(item)
    assert isinstance(result.computation, ConditionAnnotationComputation)
    assert result.computation.payload.face_count == 3

    publication = controller.publish_result(
        item,
        result,
        _headless_publisher(workbench),
    )
    assert publication.record_id == "record:condition:reserved"
    record = publication.session.document.record_index["record:condition:reserved"]
    payload = condition_payload_from_record(record)
    assert list(payload.face_indices()) == [1, 2, 3]
    assert payload.condition == "restored"
    assert controller.summary(item).state is MeasurementOperationState.COMPLETED


def test_condition_annotation_refuses_a_face_outside_the_mesh_before_work_begins() -> None:
    session = _session()
    controller = ArtifactMeasurementController(
        ArtifactWorkbench(session=session),
        id_factory=SequentialIds(),
    )

    with pytest.raises(ArtifactMeasurementError, match="outside the geometry"):
        controller.begin_condition_annotation(
            condition="missing",
            selected_face_indices=(0, 12),
            precision_grid_mm=0.01,
        )
    with pytest.raises(ArtifactMeasurementError, match="at least one face"):
        controller.begin_condition_annotation(
            condition="missing",
            selected_face_indices=(),
            precision_grid_mm=0.01,
        )
    with pytest.raises(ArtifactMeasurementError, match="condition kind"):
        controller.begin_condition_annotation(
            condition="chipped",
            selected_face_indices=(0,),
            precision_grid_mm=0.01,
        )
