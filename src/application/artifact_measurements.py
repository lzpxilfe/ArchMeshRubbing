"""Qt-free lifecycle for authoritative measurements and surface unwrapping.

The GUI may collect parameters and dispatch :func:`execute_measurement_work_item`
to a worker, but it does not own recipe capture, record IDs, stale-result policy,
or commit rebasing.  A worker returns only an immutable computation.  Publication
always rebases that computation onto the current same-Align session and then
uses the existing ``ArtifactWorkbench`` two-phase projection transaction.

Cancellation is cooperative inside deterministic extractor chunks as well as
at their outer boundaries.  A request immediately revokes commit authority;
the worker stops at its next safe Python boundary after any current NumPy or
GEOS call returns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import re
from threading import Event, RLock
from typing import TypeAlias
import uuid
from weakref import WeakKeyDictionary

from src.core.artifact_cancellation import ArtifactComputationCancelledError
from src.core.artifact_condition_annotation import (
    ArtifactConditionAnnotationError,
    ConditionAnnotationComputation,
    commit_condition_annotation,
    condition_computation_matches_active_projection,
    condition_recipe,
    condition_selection,
    face_ranges_from_indices,
    project_condition_from_recipe,
    validate_condition_recipe,
)
from src.core.artifact_developed_rubbing import (
    ArtifactDevelopedRubbingError,
    DevelopedRubbingComputation,
    commit_developed_rubbing,
    derive_developed_rubbing,
    developed_rubbing_computation_matches_active_projection,
    developed_rubbing_recipe_for_record,
    development_record_for_recipe,
    estimate_developed_rubbing_resources,
    validate_developed_rubbing_recipe,
)
from src.core.artifact_document import OperationContext, canonical_recipe_hash
from src.core.artifact_geometry_metrics import (
    ArtifactGeometryMetricsComputation,
    ArtifactGeometryMetricsError,
    commit_artifact_geometry_metrics,
    extract_geometry_metrics,
    geometry_metrics_computation_matches_active_projection,
    geometry_metrics_recipe,
)
from src.core.artifact_outline_extractor import (
    OutlineView,
    extract_outline_geometry,
    outline_recipe,
)
from src.core.artifact_rubbing_extractor import (
    ArtifactRubbingComputation,
    DigitalRubbingResourceEstimate,
    RUBBING_ESTIMATED_PEAK_BYTES_PER_PIXEL,
    RUBBING_ESTIMATE_FIXED_OVERHEAD_BYTES,
    RUBBING_ESTIMATE_GEOMETRY_MULTIPLIER,
    RUBBING_ESTIMATE_MATERIALIZED_ATTRIBUTE_MULTIPLIER,
    commit_artifact_rubbing,
    estimate_digital_rubbing_resources,
    extract_digital_rubbing,
    rubbing_computation_matches_active_projection,
    rubbing_materialized_attribute_bytes,
    rubbing_recipe,
)
from src.core.artifact_scene_adapter import ArtifactProjectionSnapshot
from src.core.artifact_session import ArtifactSession, ArtifactSessionError
from src.core.artifact_surface_measurement import (
    ArtifactSurfaceMeasurementComputation,
    ArtifactSurfaceMeasurementError,
    commit_artifact_surface_measurement,
    extract_surface_measurement_from_source,
    surface_diameter_recipe,
    surface_distance_recipe,
    surface_measurement_computation_matches_active_projection,
    surface_measurement_selection_hash,
)
from src.core.artifact_axis_alignment import AXIS_ALIGN_RECIPE_KIND
from src.core.artifact_tile_unwrap_extractor import (
    SECTION_CENTER_CANONICAL_AXIS,
    SECTION_CENTER_FIT_PER_SECTION,
    STATION_CENTERLINE_ARC,
    ArtifactTileUnwrapComputation,
    ArtifactTileUnwrapError,
    commit_artifact_tile_unwrap,
    extract_tile_unwrap,
    tile_unwrap_computation_matches_active_projection,
    tile_unwrap_recipe,
    validate_tile_unwrap_recipe,
)
from src.core.artifact_vector_extractor import (
    ArtifactVectorComputation,
    commit_vector_computation,
    computation_matches_active_projection,
    cutline_recipe,
    extract_cutline_geometry,
)
from src.core.artifact_vector_record import PlanarFrame, VectorRecordKind
from src.core.canonical_json import CanonicalJSONError, canonical_json_bytes

from .artifact_workbench import (
    ArtifactWorkbench,
    ArtifactWorkbenchError,
    RecordBindingTransition,
    StaleWorkflowOperationError,
    WorkflowSnapshot,
)
from .artifact_workflow_progress import (
    ArtifactWorkflowStep,
    REQUIRED_CUTLINE_VIEWS,
    REQUIRED_SIX_VIEWS,
    workflow_step_record_ids,
)


DEFAULT_RUBBING_MEMORY_BUDGET_BYTES = 1024 * 1024 * 1024
MAX_PUBLICATION_REBASE_ATTEMPTS = 8
_UTC_SECONDS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_LOGGER = logging.getLogger(__name__)


class ArtifactMeasurementError(ArtifactWorkbenchError):
    """A measurement operation violated its application-level contract."""


class MeasurementCancelledError(ArtifactMeasurementError):
    """The operation was cancelled and no record may be published."""


class MeasurementResourceLimitError(ArtifactMeasurementError):
    """A preflight estimate exceeds the configured local resource budget."""


class StaleMeasurementOperationError(StaleWorkflowOperationError):
    """A result no longer matches the active source/metadata/Align authority."""


class MeasurementOperationKind(str, Enum):
    CUTLINE = "cutline"
    OUTLINE = "outline"
    DIGITAL_RUBBING = "digital_rubbing"
    TILE_UNWRAP = "tile_unwrap"
    GEOMETRY_METRICS = "geometry_metrics"
    SURFACE_DISTANCE = "surface_distance"
    SURFACE_DIAMETER = "surface_diameter"
    CONDITION_ANNOTATION = "condition_annotation"
    DEVELOPED_RUBBING = "developed_rubbing"


# Kinds that allocate a raster and therefore share the rubbing memory budget.
_RASTER_KINDS = frozenset(
    {
        MeasurementOperationKind.DIGITAL_RUBBING,
        MeasurementOperationKind.DEVELOPED_RUBBING,
    }
)


class MeasurementOperationState(str, Enum):
    RUNNING = "running"
    CANCELLING = "cancelling"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    STALE = "stale"


_TERMINAL_STATE_PRIORITY = {
    MeasurementOperationState.CANCELLED: 1,
    MeasurementOperationState.STALE: 2,
    MeasurementOperationState.FAILED: 3,
}


MeasurementComputation: TypeAlias = (
    ArtifactVectorComputation
    | ArtifactRubbingComputation
    | ArtifactTileUnwrapComputation
    | ArtifactGeometryMetricsComputation
    | ArtifactSurfaceMeasurementComputation
    | ConditionAnnotationComputation
    | DevelopedRubbingComputation
)
CancellationProbe: TypeAlias = Callable[[], bool]
MeasurementPublisher: TypeAlias = Callable[[RecordBindingTransition], None]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4()}"


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactMeasurementError(f"{field_name} must be a non-empty string")
    return value.strip()


def _canonical_timestamp(value: object, *, field_name: str) -> str:
    timestamp = _required_text(value, field_name=field_name)
    if _UTC_SECONDS_RE.fullmatch(timestamp) is None:
        raise ArtifactMeasurementError(
            f"{field_name} must use canonical UTC seconds (YYYY-MM-DDTHH:MM:SSZ)"
        )
    return timestamp


def _recipe_bytes(recipe: Mapping[str, object]) -> bytes:
    if not isinstance(recipe, Mapping):
        raise ArtifactMeasurementError("measurement recipe must be an object")
    try:
        encoded = canonical_json_bytes(recipe)
        decoded = json.loads(encoded)
    except (CanonicalJSONError, json.JSONDecodeError) as exc:
        raise ArtifactMeasurementError(str(exc)) from exc
    if not isinstance(decoded, dict):
        raise ArtifactMeasurementError("measurement recipe must be an object")
    return encoded


def _recipe_dict(encoded: bytes) -> dict[str, object]:
    try:
        value = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ArtifactMeasurementError("measurement recipe JSON is invalid") from exc
    if not isinstance(value, dict):
        raise ArtifactMeasurementError("measurement recipe JSON must contain an object")
    return value


def _recipe_float(recipe: Mapping[str, object], field_name: str) -> float:
    value = recipe.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactMeasurementError(
            f"measurement recipe {field_name!r} must be a number"
        )
    return float(value)


def _operation_kind_for_computation(
    computation: MeasurementComputation,
) -> MeasurementOperationKind:
    if isinstance(computation, ArtifactGeometryMetricsComputation):
        return MeasurementOperationKind.GEOMETRY_METRICS
    if isinstance(computation, ArtifactSurfaceMeasurementComputation):
        return MeasurementOperationKind(computation.kind)
    if isinstance(computation, ArtifactRubbingComputation):
        return MeasurementOperationKind.DIGITAL_RUBBING
    if isinstance(computation, ArtifactTileUnwrapComputation):
        return MeasurementOperationKind.TILE_UNWRAP
    if isinstance(computation, ConditionAnnotationComputation):
        return MeasurementOperationKind.CONDITION_ANNOTATION
    if isinstance(computation, DevelopedRubbingComputation):
        return MeasurementOperationKind.DEVELOPED_RUBBING
    if isinstance(computation, ArtifactVectorComputation):
        kind = VectorRecordKind(computation.payload.kind)
        if kind is VectorRecordKind.CUTLINE:
            return MeasurementOperationKind.CUTLINE
        if kind is VectorRecordKind.OUTLINE:
            return MeasurementOperationKind.OUTLINE
    raise ArtifactMeasurementError("measurement computation kind is unsupported")


@dataclass(frozen=True, slots=True, eq=False)
class ArtifactMeasurementWorkItem:
    """Opaque immutable compute authority; equality is intentionally identity-only."""

    id: str
    kind: MeasurementOperationKind
    captured_session: ArtifactSession
    context: OperationContext
    projection_snapshot: ArtifactProjectionSnapshot
    recipe_json: bytes
    record_id: str
    created_at: str
    operator: str
    depends_on_record_ids: tuple[str, ...]
    base_state_version: int
    base_authority_epoch: int
    resource_estimate: DigitalRubbingResourceEstimate | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "id", _required_text(self.id, field_name="operation ID")
        )
        object.__setattr__(
            self,
            "kind",
            MeasurementOperationKind(self.kind),
        )
        if not isinstance(self.captured_session, ArtifactSession):
            raise ArtifactMeasurementError(
                "captured_session must be an ArtifactSession"
            )
        if not isinstance(self.context, OperationContext):
            raise ArtifactMeasurementError("context must be an OperationContext")
        if not isinstance(self.projection_snapshot, ArtifactProjectionSnapshot):
            raise ArtifactMeasurementError(
                "projection_snapshot must be an ArtifactProjectionSnapshot"
            )
        if not isinstance(self.recipe_json, bytes):
            raise ArtifactMeasurementError("recipe_json must be canonical JSON bytes")
        recipe = _recipe_dict(self.recipe_json)
        if canonical_json_bytes(recipe) != self.recipe_json:
            raise ArtifactMeasurementError("recipe_json is not canonical RFC 8785 JSON")
        if canonical_recipe_hash(recipe) != self.context.recipe_hash:
            raise ArtifactMeasurementError(
                "measurement recipe does not match its captured OperationContext"
            )
        if recipe.get("kind") != self.kind.value:
            raise ArtifactMeasurementError(
                "measurement recipe kind does not match the work item"
            )
        snapshot = self.projection_snapshot
        if (
            tuple(self.context.source_asset_ids) != (snapshot.source_asset_id,)
            or self.context.geometry_revision_id != snapshot.geometry_revision_id
            or self.context.source_metadata_revision_id
            != snapshot.source_metadata_revision_id
            or self.context.align_revision_id != snapshot.align_revision_id
        ):
            raise ArtifactMeasurementError(
                "measurement context does not match its projection snapshot"
            )
        if self.captured_session.projection_snapshot() != snapshot:
            raise ArtifactMeasurementError(
                "captured session does not match the measurement projection snapshot"
            )
        object.__setattr__(
            self,
            "record_id",
            _required_text(self.record_id, field_name="reserved record ID"),
        )
        object.__setattr__(
            self,
            "created_at",
            _canonical_timestamp(self.created_at, field_name="created_at"),
        )
        object.__setattr__(
            self,
            "operator",
            _required_text(self.operator, field_name="operator"),
        )
        dependencies = tuple(
            _required_text(value, field_name="dependency record ID")
            for value in self.depends_on_record_ids
        )
        if len(set(dependencies)) != len(dependencies):
            raise ArtifactMeasurementError("dependency record IDs must be unique")
        object.__setattr__(self, "depends_on_record_ids", dependencies)
        if type(self.base_state_version) is not int or self.base_state_version < 0:
            raise ArtifactMeasurementError("base_state_version must be non-negative")
        if type(self.base_authority_epoch) is not int or self.base_authority_epoch < 0:
            raise ArtifactMeasurementError("base_authority_epoch must be non-negative")
        if (
            self.kind in _RASTER_KINDS
            and self.resource_estimate is not None
            and not isinstance(
                self.resource_estimate,
                DigitalRubbingResourceEstimate,
            )
        ):
            raise ArtifactMeasurementError(
                "Digital Rubbing resource estimate has an invalid type"
            )
        if (
            self.kind not in _RASTER_KINDS
            and self.resource_estimate is not None
        ):
            raise ArtifactMeasurementError(
                "non-raster work items cannot carry a raster resource estimate"
            )

    @property
    def recipe_hash(self) -> str:
        return self.context.recipe_hash

    def recipe_dict(self) -> dict[str, object]:
        return _recipe_dict(self.recipe_json)


@dataclass(frozen=True, slots=True)
class ArtifactMeasurementResult:
    operation_id: str
    kind: MeasurementOperationKind
    computation: MeasurementComputation

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_id",
            _required_text(self.operation_id, field_name="operation ID"),
        )
        resolved = MeasurementOperationKind(self.kind)
        object.__setattr__(self, "kind", resolved)
        if _operation_kind_for_computation(self.computation) is not resolved:
            raise ArtifactMeasurementError(
                "measurement result kind does not match its computation"
            )


@dataclass(frozen=True, slots=True)
class ArtifactMeasurementPublication:
    operation_id: str
    kind: MeasurementOperationKind
    record_id: str
    session: ArtifactSession
    document_sha256: str
    align_revision_id: str


@dataclass(frozen=True, slots=True)
class ArtifactMeasurementSummary:
    operation_id: str
    kind: MeasurementOperationKind
    state: MeasurementOperationState
    record_id: str
    created_at: str
    estimated_peak_bytes: int
    error_type: str | None = None
    message: str | None = None


@dataclass(slots=True)
class _MeasurementRuntime:
    work_item: ArtifactMeasurementWorkItem
    cancellation: Event
    rubbing_memory_budget_bytes: int | None = None
    max_active_rubbing_operations: int | None = None
    resource_estimate: DigitalRubbingResourceEstimate | None = None
    reserved_peak_bytes: int = 0
    state: MeasurementOperationState = MeasurementOperationState.RUNNING
    executing: bool = False
    result: ArtifactMeasurementResult | None = None
    pending_terminal_state: MeasurementOperationState | None = None
    pending_error: BaseException | str | None = None


@dataclass(slots=True)
class _WorkbenchMeasurementState:
    """One admission and capability registry shared by a Workbench."""

    lock: RLock = field(default_factory=RLock)
    publication_lock: RLock = field(default_factory=RLock)
    rubbing_admission_lock: RLock = field(default_factory=RLock)
    active: dict[str, _MeasurementRuntime] = field(default_factory=dict)
    history: dict[str, ArtifactMeasurementSummary] = field(default_factory=dict)
    record_reservations: dict[str, str] = field(default_factory=dict)
    observer_installed: bool = False


_WORKBENCH_MEASUREMENT_STATES_LOCK = RLock()
_WORKBENCH_MEASUREMENT_STATES: WeakKeyDictionary[
    ArtifactWorkbench,
    _WorkbenchMeasurementState,
] = WeakKeyDictionary()


def _measurement_state_for(
    workbench: ArtifactWorkbench,
) -> _WorkbenchMeasurementState:
    with _WORKBENCH_MEASUREMENT_STATES_LOCK:
        state = _WORKBENCH_MEASUREMENT_STATES.get(workbench)
        if state is None:
            state = _WorkbenchMeasurementState()
            _WORKBENCH_MEASUREMENT_STATES[workbench] = state
        return state


def _raise_if_cancelled(cancellation_probe: CancellationProbe | None) -> None:
    if cancellation_probe is not None and bool(cancellation_probe()):
        raise MeasurementCancelledError(
            "measurement operation was cancelled before its next safe boundary"
        )


def execute_measurement_work_item(
    work_item: ArtifactMeasurementWorkItem,
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> ArtifactMeasurementResult:
    """Execute one immutable work item without mutating a document or GUI."""

    if not isinstance(work_item, ArtifactMeasurementWorkItem):
        raise ArtifactMeasurementError(
            "work_item must be an ArtifactMeasurementWorkItem"
        )
    _raise_if_cancelled(cancellation_probe)
    try:
        projection = work_item.captured_session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactMeasurementError(str(exc)) from exc
    if projection.snapshot != work_item.projection_snapshot:
        raise StaleMeasurementOperationError(
            "captured measurement projection changed before execution"
        )
    _raise_if_cancelled(cancellation_probe)

    recipe = work_item.recipe_dict()
    if work_item.kind is MeasurementOperationKind.CUTLINE:
        frame_value = recipe.get("frame")
        if not isinstance(frame_value, Mapping):
            raise ArtifactMeasurementError("Cutline recipe has no valid frame")
        frame = PlanarFrame.from_dict(frame_value)
        try:
            geometry = extract_cutline_geometry(
                projection.mesh.vertices,
                projection.mesh.faces,
                frame,
                classification_tolerance_mm=_recipe_float(
                    recipe,
                    "classification_tolerance_mm",
                ),
                stitch_tolerance_mm=_recipe_float(recipe, "stitch_tolerance_mm"),
                cancellation_probe=cancellation_probe,
            )
        except ArtifactComputationCancelledError as exc:
            raise MeasurementCancelledError(str(exc)) from exc
        computation: MeasurementComputation = ArtifactVectorComputation(
            context=work_item.context,
            projection_snapshot=work_item.projection_snapshot,
            payload=geometry.payload,
            recipe=recipe,
            qc=geometry.qc,
        )
    elif work_item.kind is MeasurementOperationKind.OUTLINE:
        try:
            geometry = extract_outline_geometry(
                projection.mesh.vertices,
                projection.mesh.faces,
                str(recipe["view"]),
                precision_grid_mm=_recipe_float(recipe, "precision_grid_mm"),
                cancellation_probe=cancellation_probe,
            )
        except ArtifactComputationCancelledError as exc:
            raise MeasurementCancelledError(str(exc)) from exc
        computation = ArtifactVectorComputation(
            context=work_item.context,
            projection_snapshot=work_item.projection_snapshot,
            payload=geometry.payload,
            recipe=recipe,
            qc=geometry.qc,
        )
    elif work_item.kind is MeasurementOperationKind.TILE_UNWRAP:
        try:
            unwrap, qc = extract_tile_unwrap(
                projection.mesh,
                recipe,
                cancellation_probe=cancellation_probe,
            )
        except ArtifactComputationCancelledError as exc:
            raise MeasurementCancelledError(str(exc)) from exc
        except ArtifactTileUnwrapError as exc:
            raise ArtifactMeasurementError(str(exc)) from exc
        computation = ArtifactTileUnwrapComputation(
            context=work_item.context,
            projection_snapshot=work_item.projection_snapshot,
            unwrap=unwrap,
            recipe=recipe,
            qc=qc,
        )
    elif work_item.kind is MeasurementOperationKind.GEOMETRY_METRICS:
        try:
            receipt, qc = extract_geometry_metrics(
                projection.mesh.vertices,
                projection.mesh.faces,
                recipe,
                cancellation_probe=cancellation_probe,
            )
        except ArtifactComputationCancelledError as exc:
            raise MeasurementCancelledError(str(exc)) from exc
        except ArtifactGeometryMetricsError as exc:
            raise ArtifactMeasurementError(str(exc)) from exc
        computation = ArtifactGeometryMetricsComputation(
            context=work_item.context,
            projection_snapshot=work_item.projection_snapshot,
            receipt=receipt,
            recipe=recipe,
            qc=qc,
        )
    elif work_item.kind in {
        MeasurementOperationKind.SURFACE_DISTANCE,
        MeasurementOperationKind.SURFACE_DIAMETER,
    }:
        try:
            receipt, qc = extract_surface_measurement_from_source(
                work_item.captured_session.source_mesh.vertices,
                work_item.captured_session.source_mesh.faces,
                work_item.projection_snapshot.matrix,
                recipe,
                cancellation_probe=cancellation_probe,
            )
        except ArtifactComputationCancelledError as exc:
            raise MeasurementCancelledError(str(exc)) from exc
        except ArtifactSurfaceMeasurementError as exc:
            raise ArtifactMeasurementError(str(exc)) from exc
        computation = ArtifactSurfaceMeasurementComputation(
            context=work_item.context,
            projection_snapshot=work_item.projection_snapshot,
            receipt=receipt,
            recipe=recipe,
            qc=qc,
        )
    elif work_item.kind is MeasurementOperationKind.CONDITION_ANNOTATION:
        try:
            payload = project_condition_from_recipe(
                projection.mesh.vertices,
                projection.mesh.faces,
                recipe,
                cancellation_probe=cancellation_probe,
            )
        except ArtifactComputationCancelledError as exc:
            raise MeasurementCancelledError(str(exc)) from exc
        except ArtifactConditionAnnotationError as exc:
            raise ArtifactMeasurementError(str(exc)) from exc
        computation = ConditionAnnotationComputation(
            context=work_item.context,
            projection_snapshot=work_item.projection_snapshot,
            payload=payload,
            recipe=recipe,
            qc=payload.qc_summary(),
        )
    elif work_item.kind is MeasurementOperationKind.DEVELOPED_RUBBING:
        try:
            developed_raster, developed_qc = derive_developed_rubbing(
                work_item.captured_session.document,
                projection.mesh,
                recipe,
                cancellation_probe=cancellation_probe,
            )
        except ArtifactComputationCancelledError as exc:
            raise MeasurementCancelledError(str(exc)) from exc
        except ArtifactDevelopedRubbingError as exc:
            raise ArtifactMeasurementError(str(exc)) from exc
        computation = DevelopedRubbingComputation(
            context=work_item.context,
            projection_snapshot=work_item.projection_snapshot,
            raster=developed_raster,
            recipe=recipe,
            qc=developed_qc,
        )
    elif work_item.kind is MeasurementOperationKind.DIGITAL_RUBBING:
        try:
            raster, qc = extract_digital_rubbing(
                projection.mesh.vertices,
                projection.mesh.faces,
                recipe,
                cancellation_probe=cancellation_probe,
            )
        except ArtifactComputationCancelledError as exc:
            raise MeasurementCancelledError(str(exc)) from exc
        computation = ArtifactRubbingComputation(
            context=work_item.context,
            projection_snapshot=work_item.projection_snapshot,
            raster=raster,
            recipe=recipe,
            qc=qc,
        )
    else:  # pragma: no cover - closed enum guard
        raise ArtifactMeasurementError(
            f"measurement kind {work_item.kind.value!r} is unsupported"
        )

    _raise_if_cancelled(cancellation_probe)
    return ArtifactMeasurementResult(
        operation_id=work_item.id,
        kind=work_item.kind,
        computation=computation,
    )


class ArtifactMeasurementController:
    """Own operation capabilities, resource admission, cancellation, and rebase."""

    def __init__(
        self,
        workbench: ArtifactWorkbench,
        *,
        id_factory: Callable[[str], str] = _new_id,
        rubbing_memory_budget_bytes: int = DEFAULT_RUBBING_MEMORY_BUDGET_BYTES,
        max_active_rubbing_operations: int = 1,
    ) -> None:
        if not isinstance(workbench, ArtifactWorkbench):
            raise ArtifactMeasurementError("workbench must be an ArtifactWorkbench")
        if not callable(id_factory):
            raise ArtifactMeasurementError("id_factory must be callable")
        if (
            type(rubbing_memory_budget_bytes) is not int
            or rubbing_memory_budget_bytes <= 0
        ):
            raise ArtifactMeasurementError(
                "rubbing_memory_budget_bytes must be a positive integer"
            )
        if (
            type(max_active_rubbing_operations) is not int
            or max_active_rubbing_operations <= 0
        ):
            raise ArtifactMeasurementError(
                "max_active_rubbing_operations must be a positive integer"
            )
        shared = _measurement_state_for(workbench)
        self._workbench = workbench
        self._id_factory = id_factory
        self._rubbing_memory_budget_bytes = rubbing_memory_budget_bytes
        self._max_active_rubbing_operations = max_active_rubbing_operations
        self._lock = shared.lock
        self._publication_lock = shared.publication_lock
        self._rubbing_admission_lock = shared.rubbing_admission_lock
        self._active = shared.active
        self._history = shared.history
        self._record_reservations = shared.record_reservations
        self._unsubscribe_workbench: Callable[[], None] = lambda: None
        with _WORKBENCH_MEASUREMENT_STATES_LOCK:
            if not shared.observer_installed:
                self._unsubscribe_workbench = workbench.subscribe(
                    self._on_workbench_snapshot
                )
                shared.observer_installed = True

    @property
    def workbench(self) -> ArtifactWorkbench:
        return self._workbench

    @property
    def active_summaries(self) -> tuple[ArtifactMeasurementSummary, ...]:
        with self._lock:
            return tuple(
                self._summary_for_runtime(runtime)
                for _, runtime in sorted(self._active.items())
            )

    def summary(
        self,
        operation: ArtifactMeasurementWorkItem | str,
    ) -> ArtifactMeasurementSummary:
        operation_id = (
            operation.id
            if isinstance(operation, ArtifactMeasurementWorkItem)
            else _required_text(operation, field_name="operation ID")
        )
        with self._lock:
            runtime = self._active.get(operation_id)
            if runtime is not None:
                if (
                    isinstance(operation, ArtifactMeasurementWorkItem)
                    and runtime.work_item is not operation
                ):
                    raise StaleMeasurementOperationError(
                        "operation capability is stale"
                    )
                return self._summary_for_runtime(runtime)
            summary = self._history.get(operation_id)
            if summary is None:
                raise StaleMeasurementOperationError("measurement operation is unknown")
            return summary

    def rubbing_resource_estimate(
        self,
        work_item: ArtifactMeasurementWorkItem,
    ) -> DigitalRubbingResourceEstimate | None:
        """Return the worker-computed estimate while its exact operation is active."""

        with self._lock:
            runtime = self._require_runtime_locked(
                work_item,
                states=frozenset(
                    {
                        MeasurementOperationState.RUNNING,
                        MeasurementOperationState.CANCELLING,
                        MeasurementOperationState.PUBLISHING,
                    }
                ),
            )
            if work_item.kind not in _RASTER_KINDS:
                return None
            return runtime.resource_estimate or work_item.resource_estimate

    @staticmethod
    def _summary_for_runtime(
        runtime: _MeasurementRuntime,
    ) -> ArtifactMeasurementSummary:
        estimate = runtime.resource_estimate or runtime.work_item.resource_estimate
        return ArtifactMeasurementSummary(
            operation_id=runtime.work_item.id,
            kind=runtime.work_item.kind,
            state=runtime.state,
            record_id=runtime.work_item.record_id,
            created_at=runtime.work_item.created_at,
            estimated_peak_bytes=(
                estimate.estimated_peak_bytes
                if estimate is not None
                else runtime.reserved_peak_bytes
            ),
        )

    def _require_runtime_locked(
        self,
        work_item: ArtifactMeasurementWorkItem,
        *,
        states: frozenset[MeasurementOperationState],
    ) -> _MeasurementRuntime:
        if not isinstance(work_item, ArtifactMeasurementWorkItem):
            raise ArtifactMeasurementError(
                "work_item must be an ArtifactMeasurementWorkItem"
            )
        runtime = self._active.get(work_item.id)
        if runtime is None or runtime.work_item is not work_item:
            raise StaleMeasurementOperationError(
                "measurement operation capability is stale"
            )
        if runtime.state not in states:
            raise StaleMeasurementOperationError(
                f"measurement operation is already {runtime.state.value}"
            )
        return runtime

    def _finish_locked(
        self,
        runtime: _MeasurementRuntime,
        state: MeasurementOperationState,
        error: BaseException | str | None = None,
    ) -> ArtifactMeasurementSummary:
        work_item = runtime.work_item
        active = self._active.get(work_item.id)
        if active is not runtime:
            summary = self._history.get(work_item.id)
            if summary is None:
                raise StaleMeasurementOperationError(
                    "measurement operation capability is stale"
                )
            return summary
        runtime.state = state
        estimate = runtime.resource_estimate or work_item.resource_estimate
        summary = ArtifactMeasurementSummary(
            operation_id=work_item.id,
            kind=work_item.kind,
            state=state,
            record_id=work_item.record_id,
            created_at=work_item.created_at,
            estimated_peak_bytes=(
                estimate.estimated_peak_bytes
                if estimate is not None
                else runtime.reserved_peak_bytes
            ),
            error_type=(
                type(error).__name__
                if isinstance(error, BaseException)
                else ("Error" if error is not None else None)
            ),
            message=(str(error) if error is not None else None),
        )
        self._active.pop(work_item.id, None)
        if self._record_reservations.get(work_item.record_id) == work_item.id:
            self._record_reservations.pop(work_item.record_id, None)
        self._history[work_item.id] = summary
        return summary

    def _request_terminal_locked(
        self,
        runtime: _MeasurementRuntime,
        state: MeasurementOperationState,
        error: BaseException | str,
    ) -> ArtifactMeasurementSummary:
        """Revoke commit authority while retaining resources until a worker exits."""

        runtime.cancellation.set()
        if runtime.executing:
            current = runtime.pending_terminal_state
            if (
                current is None
                or _TERMINAL_STATE_PRIORITY[state] >= _TERMINAL_STATE_PRIORITY[current]
            ):
                runtime.pending_terminal_state = state
                runtime.pending_error = error
            runtime.state = MeasurementOperationState.CANCELLING
            return self._summary_for_runtime(runtime)
        return self._finish_locked(runtime, state, error)

    @staticmethod
    def _terminal_exception(
        state: MeasurementOperationState,
        error: BaseException | str | None,
    ) -> ArtifactWorkbenchError:
        message = str(error or state.value)
        if state is MeasurementOperationState.CANCELLED:
            if isinstance(error, MeasurementCancelledError):
                return error
            return MeasurementCancelledError(message)
        if state is MeasurementOperationState.STALE:
            if isinstance(error, StaleMeasurementOperationError):
                return error
            return StaleMeasurementOperationError(message)
        if isinstance(error, ArtifactMeasurementError):
            return error
        return ArtifactMeasurementError(message)

    def _finish_execution_locked(
        self,
        runtime: _MeasurementRuntime,
        *,
        default_state: MeasurementOperationState,
        default_error: BaseException | str,
    ) -> tuple[ArtifactMeasurementSummary, ArtifactWorkbenchError | None]:
        runtime.executing = False
        pending_state = runtime.pending_terminal_state
        if (
            pending_state is None
            or _TERMINAL_STATE_PRIORITY[pending_state]
            < _TERMINAL_STATE_PRIORITY[default_state]
        ):
            pending_wins = False
            state = default_state
            error = default_error
        else:
            pending_wins = True
            state = pending_state
            error = runtime.pending_error
        summary = self._finish_locked(runtime, state, error)
        if not pending_wins:
            return summary, None
        return summary, self._terminal_exception(state, error)

    @staticmethod
    def _projection_authority_key(
        snapshot: ArtifactProjectionSnapshot,
    ) -> tuple[object, ...]:
        """Coordinate authority key; document hash intentionally excluded."""

        return snapshot.render_key

    def _on_workbench_snapshot(self, snapshot: WorkflowSnapshot) -> None:
        """Permanently revoke work after a finalized Open/Align authority change."""

        if not isinstance(snapshot, WorkflowSnapshot):
            return
        current_session = snapshot.session
        current_key: tuple[object, ...] | None = None
        if isinstance(current_session, ArtifactSession) and not snapshot.faulted:
            try:
                current_key = self._projection_authority_key(
                    current_session.projection_snapshot()
                )
            except ArtifactSessionError:
                current_key = None
        with self._lock:
            for runtime in tuple(self._active.values()):
                work_item = runtime.work_item
                if snapshot.faulted:
                    self._request_terminal_locked(
                        runtime,
                        MeasurementOperationState.FAILED,
                        ArtifactMeasurementError(
                            "artifact authority faulted while measurement work was active"
                        ),
                    )
                    continue
                if current_session is None:
                    if snapshot.pending_load is None:
                        self._request_terminal_locked(
                            runtime,
                            MeasurementOperationState.STALE,
                            StaleMeasurementOperationError(
                                "active artifact authority was cleared"
                            ),
                        )
                    continue
                if (
                    current_session.source_mesh
                    is not work_item.captured_session.source_mesh
                    or current_key
                    != self._projection_authority_key(work_item.projection_snapshot)
                ):
                    self._request_terminal_locked(
                        runtime,
                        MeasurementOperationState.STALE,
                        StaleMeasurementOperationError(
                            "finalized source, metadata, or Align authority changed"
                        ),
                    )

    @staticmethod
    def _durable_id_exists(session: ArtifactSession, item_id: str) -> bool:
        document = session.document
        return any(
            item_id in index
            for index in (
                document.source_asset_index,
                document.geometry_revision_index,
                document.source_metadata_revision_index,
                document.align_revision_index,
                document.record_index,
            )
        )

    def _rubbing_admission_locked(self) -> tuple[int, int, int, int]:
        active = [
            runtime
            for runtime in self._active.values()
            if runtime.work_item.kind in _RASTER_KINDS
        ]
        memory_budgets = [
            runtime.rubbing_memory_budget_bytes
            for runtime in active
            if runtime.rubbing_memory_budget_bytes is not None
        ]
        operation_limits = [
            runtime.max_active_rubbing_operations
            for runtime in active
            if runtime.max_active_rubbing_operations is not None
        ]
        return (
            len(active),
            sum(
                runtime.reserved_peak_bytes
                for runtime in active
            ),
            min((self._rubbing_memory_budget_bytes, *memory_budgets)),
            min((self._max_active_rubbing_operations, *operation_limits)),
        )

    def _begin(
        self,
        *,
        kind: MeasurementOperationKind,
        recipe: Mapping[str, object],
        record_id: str | None,
        created_at: str | None,
        operator: str,
        selection_hash: str | None = None,
        depends_on_record_ids: Sequence[str] = (),
        estimate_rubbing: bool = False,
    ) -> ArtifactMeasurementWorkItem:
        session = self._workbench.snapshot.session
        if not isinstance(session, ArtifactSession):
            raise ArtifactMeasurementError("no active ArtifactDocument session")
        self._workbench.require_stable_session(session, measurement=True)
        if kind is MeasurementOperationKind.TILE_UNWRAP:
            try:
                tile_recipe = validate_tile_unwrap_recipe(recipe)
            except ArtifactTileUnwrapError as exc:
                raise ArtifactMeasurementError(str(exc)) from exc
            selection = tile_recipe["selection"]
            assert isinstance(selection, Mapping)
            if int(selection["total_face_count"]) != int(
                session.source_mesh.faces.shape[0]
            ):
                raise StaleMeasurementOperationError(
                    "tile unwrap selection does not match the active source mesh"
                )
        elif kind is MeasurementOperationKind.CONDITION_ANNOTATION:
            try:
                condition_selection_block = validate_condition_recipe(recipe)["selection"]
            except ArtifactConditionAnnotationError as exc:
                raise ArtifactMeasurementError(str(exc)) from exc
            if int(condition_selection_block["total_face_count"]) != int(
                session.source_mesh.faces.shape[0]
            ):
                raise StaleMeasurementOperationError(
                    "condition selection does not match the active source mesh"
                )
        development_prerequisites: tuple[str, ...] = ()
        if kind is MeasurementOperationKind.DEVELOPED_RUBBING:
            # The development must still be the one the recipe names, by
            # hash, in the session the work is captured on.  A development
            # that was recomputed or superseded is another development.
            try:
                developed_recipe = validate_developed_rubbing_recipe(recipe)
                development_record, _development_receipt = (
                    development_record_for_recipe(session.document, developed_recipe)
                )
            except ArtifactDevelopedRubbingError as exc:
                raise ArtifactMeasurementError(str(exc)) from exc
            development_prerequisites = (development_record.id,)
        prerequisite_ids: tuple[str, ...] = ()
        if kind is MeasurementOperationKind.DEVELOPED_RUBBING:
            prerequisite_ids = development_prerequisites
        elif kind is MeasurementOperationKind.OUTLINE:
            prerequisite_ids = workflow_step_record_ids(
                session,
                ArtifactWorkflowStep.CUTLINE,
            )
            if len(prerequisite_ids) != len(REQUIRED_CUTLINE_VIEWS):
                raise ArtifactMeasurementError(
                    "Outline requires READY + FRESH Top, Front, and Right "
                    "Cutline records"
                )
        elif kind is MeasurementOperationKind.DIGITAL_RUBBING:
            prerequisite_ids = workflow_step_record_ids(
                session,
                ArtifactWorkflowStep.OUTLINE,
            )
            if len(prerequisite_ids) != len(REQUIRED_SIX_VIEWS):
                raise ArtifactMeasurementError(
                    "Digital Rubbing requires six dependency-valid READY + FRESH "
                    "Outline records"
                )
        encoded_recipe = _recipe_bytes(recipe)
        canonical_recipe = _recipe_dict(encoded_recipe)
        try:
            context = session.capture_operation(
                recipe=canonical_recipe,
                selection_hash=selection_hash,
            )
            projection_snapshot = session.projection_snapshot()
        except ArtifactSessionError as exc:
            raise ArtifactMeasurementError(str(exc)) from exc
        state = self._workbench.snapshot
        if state.session is not session:
            raise StaleMeasurementOperationError(
                "artifact authority changed while measurement work was captured"
            )

        requested_dependencies = tuple(
            _required_text(value, field_name="dependency record ID")
            for value in depends_on_record_ids
        )
        if len(set(requested_dependencies)) != len(requested_dependencies):
            raise ArtifactMeasurementError("dependency record IDs must be unique")
        dependencies = (
            *prerequisite_ids,
            *(
                dependency_id
                for dependency_id in requested_dependencies
                if dependency_id not in prerequisite_ids
            ),
        )
        for dependency_id in dependencies:
            dependency = session.document.record_index.get(dependency_id)
            if dependency is None:
                raise ArtifactMeasurementError(
                    f"dependency record {dependency_id!r} does not exist"
                )
            if session.document.record_freshness(dependency_id).value != "fresh":
                raise ArtifactMeasurementError(
                    f"dependency record {dependency_id!r} is not fresh"
                )

        resolved_record_id = _required_text(
            (
                self._id_factory(f"record:{kind.value}")
                if record_id is None
                else record_id
            ),
            field_name="reserved record ID",
        )
        operation_id = _required_text(
            self._id_factory(f"operation:{kind.value}"),
            field_name="operation ID",
        )

        estimate: DigitalRubbingResourceEstimate | None = None
        minimum_estimated_bytes = 0
        if estimate_rubbing:
            with self._lock:
                (
                    active_rubbing,
                    reserved_bytes,
                    effective_memory_budget,
                    effective_operation_limit,
                ) = self._rubbing_admission_locked()
                if active_rubbing >= effective_operation_limit:
                    raise MeasurementResourceLimitError(
                        "another Digital Rubbing operation already owns the raster budget"
                    )
                if reserved_bytes >= effective_memory_budget:
                    raise MeasurementResourceLimitError(
                        "active Digital Rubbing work already reserves the memory budget"
                    )
                source_geometry_bytes = int(
                    session.source_mesh.vertices.nbytes
                    + session.source_mesh.faces.nbytes
                )
                source_materialized_attribute_bytes = (
                    rubbing_materialized_attribute_bytes(
                        uv_coords=session.source_mesh.uv_coords,
                        texture=session.source_mesh.texture,
                    )
                )
                minimum_estimated_bytes = (
                    RUBBING_ESTIMATE_FIXED_OVERHEAD_BYTES
                    + RUBBING_ESTIMATED_PEAK_BYTES_PER_PIXEL
                    + source_geometry_bytes * RUBBING_ESTIMATE_GEOMETRY_MULTIPLIER
                    + source_materialized_attribute_bytes
                    * RUBBING_ESTIMATE_MATERIALIZED_ATTRIBUTE_MULTIPLIER
                )
                if reserved_bytes + minimum_estimated_bytes > effective_memory_budget:
                    raise MeasurementResourceLimitError(
                        "Digital Rubbing minimum cumulative memory estimate exceeds "
                        "the configured budget before projection preflight"
                    )

        work_item = ArtifactMeasurementWorkItem(
            id=operation_id,
            kind=kind,
            captured_session=session,
            context=context,
            projection_snapshot=projection_snapshot,
            recipe_json=encoded_recipe,
            record_id=resolved_record_id,
            created_at=str(_utc_now() if created_at is None else created_at),
            operator=operator,
            depends_on_record_ids=dependencies,
            base_state_version=state.state_version,
            base_authority_epoch=state.authority_epoch,
            resource_estimate=estimate,
        )

        with self._lock:
            self._workbench.require_stable_session(session, measurement=True)
            if operation_id in self._active or operation_id in self._history:
                raise ArtifactMeasurementError(
                    f"operation ID {operation_id!r} has already been used"
                )
            if self._durable_id_exists(session, resolved_record_id):
                raise ArtifactMeasurementError(
                    f"record ID {resolved_record_id!r} collides with a durable document ID"
                )
            reservation_owner = self._record_reservations.get(resolved_record_id)
            if reservation_owner is not None:
                raise ArtifactMeasurementError(
                    f"record ID {resolved_record_id!r} is reserved by {reservation_owner!r}"
                )
            if kind in _RASTER_KINDS:
                (
                    active_rubbing,
                    reserved_bytes,
                    effective_memory_budget,
                    effective_operation_limit,
                ) = self._rubbing_admission_locked()
                if active_rubbing >= effective_operation_limit:
                    raise MeasurementResourceLimitError(
                        "another Digital Rubbing operation already owns the raster budget"
                    )
                if (
                    reserved_bytes + minimum_estimated_bytes
                    > effective_memory_budget
                ):
                    raise MeasurementResourceLimitError(
                        "Digital Rubbing cumulative minimum memory reservation "
                        f"{reserved_bytes + minimum_estimated_bytes} bytes "
                        "exceeds the configured budget "
                        f"{effective_memory_budget} bytes"
                    )
            runtime = _MeasurementRuntime(
                work_item=work_item,
                cancellation=Event(),
                rubbing_memory_budget_bytes=(
                    self._rubbing_memory_budget_bytes
                    if kind in _RASTER_KINDS
                    else None
                ),
                max_active_rubbing_operations=(
                    self._max_active_rubbing_operations
                    if kind in _RASTER_KINDS
                    else None
                ),
                resource_estimate=estimate,
                reserved_peak_bytes=(
                    minimum_estimated_bytes if kind in _RASTER_KINDS else 0
                ),
            )
            self._active[operation_id] = runtime
            self._record_reservations[resolved_record_id] = operation_id
        return work_item

    def begin_cutline(
        self,
        frame: PlanarFrame,
        *,
        classification_tolerance_mm: float = 1e-9,
        stitch_tolerance_mm: float = 1e-7,
        selection_hash: str | None = None,
        record_id: str | None = None,
        created_at: str | None = None,
        operator: str = "local-user",
        depends_on_record_ids: Sequence[str] = (),
    ) -> ArtifactMeasurementWorkItem:
        recipe = cutline_recipe(
            frame,
            classification_tolerance_mm=classification_tolerance_mm,
            stitch_tolerance_mm=stitch_tolerance_mm,
        )
        return self._begin(
            kind=MeasurementOperationKind.CUTLINE,
            recipe=recipe,
            record_id=record_id,
            created_at=created_at,
            operator=operator,
            selection_hash=selection_hash,
            depends_on_record_ids=depends_on_record_ids,
        )

    def begin_outline(
        self,
        view: OutlineView | str,
        *,
        precision_grid_mm: float,
        record_id: str | None = None,
        created_at: str | None = None,
        operator: str = "local-user",
        depends_on_record_ids: Sequence[str] = (),
    ) -> ArtifactMeasurementWorkItem:
        recipe = outline_recipe(view, precision_grid_mm=precision_grid_mm)
        return self._begin(
            kind=MeasurementOperationKind.OUTLINE,
            recipe=recipe,
            record_id=record_id,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=depends_on_record_ids,
        )

    def begin_tile_unwrap(
        self,
        *,
        longitudinal_axis: str,
        record_view: str,
        selected_face_indices: Sequence[int] | None = None,
        n_sections: int = 32,
        seam_angle_microdegrees: int | None = None,
        section_center_policy: str = SECTION_CENTER_FIT_PER_SECTION,
        station_policy: str = STATION_CENTERLINE_ARC,
        record_id: str | None = None,
        created_at: str | None = None,
        operator: str = "local-user",
        depends_on_record_ids: Sequence[str] = (),
    ) -> ArtifactMeasurementWorkItem:
        session = self._workbench.snapshot.session
        if not isinstance(session, ArtifactSession):
            raise ArtifactMeasurementError("no active ArtifactDocument session")
        if section_center_policy == SECTION_CENTER_CANONICAL_AXIS:
            # Unrolling about the canonical origin is only true of an artifact
            # that was stood on a measured axis through that origin.  Under a
            # manual drag the origin is wherever the drag left it, and the
            # strip would be unrolled about a point that means nothing.
            align_id = session.document.active_align_revision_id
            align = (
                session.document.align_revision_index.get(align_id)
                if isinstance(align_id, str)
                else None
            )
            if align is None or align.recipe.get("kind") != AXIS_ALIGN_RECIPE_KIND:
                raise ArtifactMeasurementError(
                    "unrolling about the canonical axis needs an artifact "
                    "positioned on its measured rotation axis; the active Align "
                    "was not made from one"
                )
        try:
            recipe = tile_unwrap_recipe(
                longitudinal_axis=longitudinal_axis,
                record_view=record_view,
                total_face_count=int(session.source_mesh.faces.shape[0]),
                selected_face_indices=selected_face_indices,
                n_sections=n_sections,
                seam_angle_microdegrees=seam_angle_microdegrees,
                section_center_policy=section_center_policy,
                station_policy=station_policy,
            )
        except ArtifactTileUnwrapError as exc:
            raise ArtifactMeasurementError(str(exc)) from exc
        selection = recipe["selection"]
        assert isinstance(selection, Mapping)
        return self._begin(
            kind=MeasurementOperationKind.TILE_UNWRAP,
            recipe=recipe,
            record_id=record_id,
            created_at=created_at,
            operator=operator,
            selection_hash=str(selection["selection_sha256"]),
            depends_on_record_ids=depends_on_record_ids,
        )

    def begin_condition_annotation(
        self,
        *,
        condition: str,
        selected_face_indices: Sequence[int],
        precision_grid_mm: float,
        record_id: str | None = None,
        created_at: str | None = None,
        operator: str = "local-user",
        depends_on_record_ids: Sequence[str] = (),
    ) -> ArtifactMeasurementWorkItem:
        """Reserve one condition region - missing, restored, crack, or worn.

        The face set is fixed here, canonically encoded, and carried in the
        recipe, so the worker projects exactly what the user painted and the
        record's recipe hash names exactly that region.
        """

        session = self._workbench.snapshot.session
        if not isinstance(session, ArtifactSession):
            raise ArtifactMeasurementError("no active ArtifactDocument session")
        total_face_count = int(session.source_mesh.faces.shape[0])
        try:
            selection = condition_selection(
                total_face_count=total_face_count,
                face_ranges=face_ranges_from_indices(
                    selected_face_indices,
                    total_face_count=total_face_count,
                ),
            )
            recipe = condition_recipe(
                condition=condition,
                precision_grid_mm=precision_grid_mm,
                selection=selection,
            )
        except ArtifactConditionAnnotationError as exc:
            raise ArtifactMeasurementError(str(exc)) from exc
        return self._begin(
            kind=MeasurementOperationKind.CONDITION_ANNOTATION,
            recipe=recipe,
            record_id=record_id,
            created_at=created_at,
            operator=operator,
            selection_hash=str(selection["selection_sha256"]),
            depends_on_record_ids=depends_on_record_ids,
        )

    def begin_geometry_metrics(
        self,
        *,
        coordinate_grid_um: int = 1,
        record_id: str | None = None,
        created_at: str | None = None,
        operator: str = "local-user",
        depends_on_record_ids: Sequence[str] = (),
    ) -> ArtifactMeasurementWorkItem:
        """Reserve one whole-active-geometry area/volume record."""

        recipe = geometry_metrics_recipe(coordinate_grid_um=coordinate_grid_um)
        return self._begin(
            kind=MeasurementOperationKind.GEOMETRY_METRICS,
            recipe=recipe,
            record_id=record_id,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=depends_on_record_ids,
        )

    def begin_surface_distance(
        self,
        anchors: Sequence[Mapping[str, object]],
        *,
        coordinate_grid_um: int = 1,
        record_id: str | None = None,
        created_at: str | None = None,
        operator: str = "local-user",
        depends_on_record_ids: Sequence[str] = (),
    ) -> ArtifactMeasurementWorkItem:
        """Reserve one two-anchor Euclidean chord distance record."""

        session = self._workbench.snapshot.session
        if not isinstance(session, ArtifactSession):
            raise ArtifactMeasurementError("no active ArtifactDocument session")
        recipe = surface_distance_recipe(
            anchors,
            source_vertex_count=int(session.source_mesh.vertices.shape[0]),
            source_face_count=int(session.source_mesh.faces.shape[0]),
            coordinate_grid_um=coordinate_grid_um,
        )
        return self._begin(
            kind=MeasurementOperationKind.SURFACE_DISTANCE,
            recipe=recipe,
            record_id=record_id,
            created_at=created_at,
            operator=operator,
            selection_hash=surface_measurement_selection_hash(recipe),
            depends_on_record_ids=depends_on_record_ids,
        )

    def begin_surface_diameter(
        self,
        anchors: Sequence[Mapping[str, object]],
        *,
        coordinate_grid_um: int = 1,
        fit_review_threshold_um: int = 250,
        maximum_fit_condition: int = 100_000_000,
        record_id: str | None = None,
        created_at: str | None = None,
        operator: str = "local-user",
        depends_on_record_ids: Sequence[str] = (),
    ) -> ArtifactMeasurementWorkItem:
        """Reserve one best-fit planar circle diameter record."""

        session = self._workbench.snapshot.session
        if not isinstance(session, ArtifactSession):
            raise ArtifactMeasurementError("no active ArtifactDocument session")
        recipe = surface_diameter_recipe(
            anchors,
            source_vertex_count=int(session.source_mesh.vertices.shape[0]),
            source_face_count=int(session.source_mesh.faces.shape[0]),
            coordinate_grid_um=coordinate_grid_um,
            fit_review_threshold_um=fit_review_threshold_um,
            maximum_fit_condition=maximum_fit_condition,
        )
        return self._begin(
            kind=MeasurementOperationKind.SURFACE_DIAMETER,
            recipe=recipe,
            record_id=record_id,
            created_at=created_at,
            operator=operator,
            selection_hash=surface_measurement_selection_hash(recipe),
            depends_on_record_ids=depends_on_record_ids,
        )

    def begin_rubbing(
        self,
        view: OutlineView | str,
        *,
        pixels_per_mm: int,
        margin_um: int,
        reference_radius_um: int,
        depth_quantization_um: int,
        black_point_um: int,
        ink_strength_percent: int,
        relief_polarity: str,
        record_id: str | None = None,
        created_at: str | None = None,
        operator: str = "local-user",
        depends_on_record_ids: Sequence[str] = (),
    ) -> ArtifactMeasurementWorkItem:
        with self._rubbing_admission_lock:
            recipe = rubbing_recipe(
                view,
                pixels_per_mm=pixels_per_mm,
                margin_um=margin_um,
                reference_radius_um=reference_radius_um,
                depth_quantization_um=depth_quantization_um,
                black_point_um=black_point_um,
                ink_strength_percent=ink_strength_percent,
                relief_polarity=relief_polarity,
            )
            return self._begin(
                kind=MeasurementOperationKind.DIGITAL_RUBBING,
                recipe=recipe,
                record_id=record_id,
                created_at=created_at,
                operator=operator,
                depends_on_record_ids=depends_on_record_ids,
                estimate_rubbing=True,
            )

    def begin_developed_rubbing(
        self,
        development_record_id: str,
        *,
        pixels_per_mm: int,
        margin_um: int,
        reference_radius_um: int,
        depth_quantization_um: int,
        black_point_um: int,
        ink_strength_percent: int,
        relief_polarity: str,
        record_id: str | None = None,
        created_at: str | None = None,
        operator: str = "local-user",
        depends_on_record_ids: Sequence[str] = (),
    ) -> ArtifactMeasurementWorkItem:
        """Draw a rubbing on a READY + FRESH tile-unwrap record's development."""

        session = self._workbench.snapshot.session
        if not isinstance(session, ArtifactSession):
            raise ArtifactMeasurementError("no active ArtifactDocument session")
        with self._rubbing_admission_lock:
            try:
                recipe = developed_rubbing_recipe_for_record(
                    session.document,
                    development_record_id,
                    pixels_per_mm=pixels_per_mm,
                    margin_um=margin_um,
                    reference_radius_um=reference_radius_um,
                    depth_quantization_um=depth_quantization_um,
                    black_point_um=black_point_um,
                    ink_strength_percent=ink_strength_percent,
                    relief_polarity=relief_polarity,
                )
            except ArtifactDevelopedRubbingError as exc:
                raise ArtifactMeasurementError(str(exc)) from exc
            return self._begin(
                kind=MeasurementOperationKind.DEVELOPED_RUBBING,
                recipe=recipe,
                record_id=record_id,
                created_at=created_at,
                operator=operator,
                depends_on_record_ids=depends_on_record_ids,
                estimate_rubbing=True,
            )

    def _prepare_rubbing_resource_estimate(
        self,
        runtime: _MeasurementRuntime,
        cancellation: Event,
    ) -> None:
        """Expand a cheap admission reservation on the executing worker."""

        work_item = runtime.work_item
        if work_item.kind not in _RASTER_KINDS:
            return
        _raise_if_cancelled(cancellation.is_set)
        session = work_item.captured_session
        if work_item.kind is MeasurementOperationKind.DEVELOPED_RUBBING:
            # The development receipt states the strip's exact extent, so the
            # artboard is known without unrolling anything on the worker.
            try:
                _development, development_receipt = development_record_for_recipe(
                    session.document,
                    work_item.recipe_dict(),
                )
                estimate = estimate_developed_rubbing_resources(
                    development_receipt,
                    work_item.recipe_dict(),
                    source_vertex_count=int(session.source_mesh.vertices.shape[0]),
                    source_face_count=int(session.source_mesh.faces.shape[0]),
                    source_geometry_bytes=int(
                        session.source_mesh.vertices.nbytes
                        + session.source_mesh.faces.nbytes
                    ),
                )
            except ArtifactDevelopedRubbingError as exc:
                raise ArtifactMeasurementError(str(exc)) from exc
        else:
            estimate = estimate_digital_rubbing_resources(
                session.source_mesh.vertices,
                session.source_mesh.faces,
                work_item.recipe_dict(),
                source_to_world_mm_matrix4x4=work_item.projection_snapshot.matrix4x4,
                uv_coords=session.source_mesh.uv_coords,
                texture=session.source_mesh.texture,
            )
        if session.projection_snapshot() != work_item.projection_snapshot:
            raise StaleMeasurementOperationError(
                "artifact projection changed during resource preflight"
            )
        _raise_if_cancelled(cancellation.is_set)

        with self._rubbing_admission_lock, self._lock:
            active = self._active.get(work_item.id)
            if active is not runtime or cancellation.is_set():
                raise MeasurementCancelledError(
                    "Digital Rubbing resource preflight lost operation authority"
                )
            if runtime.state is not MeasurementOperationState.RUNNING:
                raise MeasurementCancelledError(
                    "Digital Rubbing resource preflight was cancelled"
                )
            (
                _active_rubbing,
                reserved_bytes,
                effective_memory_budget,
                _effective_operation_limit,
            ) = self._rubbing_admission_locked()
            other_reserved_bytes = max(
                0,
                reserved_bytes - runtime.reserved_peak_bytes,
            )
            cumulative_peak_bytes = (
                other_reserved_bytes + estimate.estimated_peak_bytes
            )
            if cumulative_peak_bytes > effective_memory_budget:
                raise MeasurementResourceLimitError(
                    "Digital Rubbing cumulative estimated peak memory "
                    f"{cumulative_peak_bytes} bytes exceeds the configured budget "
                    f"{effective_memory_budget} bytes"
                )
            runtime.resource_estimate = estimate
            runtime.reserved_peak_bytes = estimate.estimated_peak_bytes

    def execute(
        self,
        work_item: ArtifactMeasurementWorkItem,
        *,
        preflight: Callable[[], None] | None = None,
    ) -> ArtifactMeasurementResult:
        with self._lock:
            runtime = self._require_runtime_locked(
                work_item,
                states=frozenset({MeasurementOperationState.RUNNING}),
            )
            if runtime.executing or runtime.result is not None:
                raise StaleMeasurementOperationError(
                    "measurement operation has already been executed"
                )
            runtime.executing = True
            cancellation = runtime.cancellation
        try:
            _raise_if_cancelled(cancellation.is_set)
            if preflight is not None:
                preflight()
                _raise_if_cancelled(cancellation.is_set)
            self._prepare_rubbing_resource_estimate(runtime, cancellation)
            result = execute_measurement_work_item(
                work_item,
                cancellation_probe=cancellation.is_set,
            )
        except MeasurementCancelledError as exc:
            terminal_error: ArtifactWorkbenchError | None = None
            with self._lock:
                active = self._active.get(work_item.id)
                if active is runtime:
                    _summary, terminal_error = self._finish_execution_locked(
                        runtime,
                        default_state=MeasurementOperationState.CANCELLED,
                        default_error=exc,
                    )
            if terminal_error is not None:
                raise terminal_error from exc
            raise
        except Exception as exc:
            terminal_error = None
            with self._lock:
                active = self._active.get(work_item.id)
                if active is runtime:
                    _summary, terminal_error = self._finish_execution_locked(
                        runtime,
                        default_state=MeasurementOperationState.FAILED,
                        default_error=exc,
                    )
            if terminal_error is not None:
                raise terminal_error from exc
            raise
        terminal_error = None
        with self._lock:
            active = self._active.get(work_item.id)
            if active is not runtime:
                raise MeasurementCancelledError(
                    "measurement result lost commit authority while computing"
                )
            runtime.executing = False
            if runtime.pending_terminal_state is not None:
                state = runtime.pending_terminal_state
                error = runtime.pending_error
                self._finish_locked(runtime, state, error)
                terminal_error = self._terminal_exception(state, error)
            elif runtime.state is not MeasurementOperationState.RUNNING:
                terminal_error = MeasurementCancelledError(
                    "measurement result lost commit authority while computing"
                )
            else:
                runtime.result = result
        if terminal_error is not None:
            raise terminal_error
        return result

    def cancel(
        self,
        work_item: ArtifactMeasurementWorkItem,
        *,
        reason: str = "user_cancelled",
    ) -> ArtifactMeasurementSummary:
        with self._lock:
            runtime = self._require_runtime_locked(
                work_item,
                states=frozenset(
                    {
                        MeasurementOperationState.RUNNING,
                        MeasurementOperationState.CANCELLING,
                    }
                ),
            )
            if runtime.state is MeasurementOperationState.CANCELLING:
                return self._summary_for_runtime(runtime)
            return self._request_terminal_locked(
                runtime,
                MeasurementOperationState.CANCELLED,
                reason,
            )

    def fail(
        self,
        work_item: ArtifactMeasurementWorkItem,
        error: BaseException | str,
    ) -> ArtifactMeasurementSummary:
        with self._lock:
            runtime = self._require_runtime_locked(
                work_item,
                states=frozenset(
                    {
                        MeasurementOperationState.RUNNING,
                        MeasurementOperationState.CANCELLING,
                    }
                ),
            )
            return self._request_terminal_locked(
                runtime,
                MeasurementOperationState.FAILED,
                error,
            )

    @staticmethod
    def _require_captured_document_ancestor(
        work_item: ArtifactMeasurementWorkItem,
        current: ArtifactSession,
    ) -> None:
        captured = work_item.captured_session
        if current.source_mesh is not captured.source_mesh:
            raise StaleMeasurementOperationError(
                "measurement source session changed after work began"
            )
        old = captured.document
        new = current.document
        if (
            new.document_id != old.document_id
            or new.schema_version != old.schema_version
            or new.software_version != old.software_version
            or new.extensions != old.extensions
        ):
            raise StaleMeasurementOperationError(
                "measurement document identity changed after work began"
            )
        for old_index, new_index, label in (
            (old.source_asset_index, new.source_asset_index, "source asset"),
            (
                old.geometry_revision_index,
                new.geometry_revision_index,
                "geometry revision",
            ),
            (
                old.source_metadata_revision_index,
                new.source_metadata_revision_index,
                "metadata revision",
            ),
            (old.align_revision_index, new.align_revision_index, "Align revision"),
            (old.record_index, new.record_index, "derived record"),
        ):
            for item_id, value in old_index.items():
                if new_index.get(item_id) != value:
                    raise StaleMeasurementOperationError(
                        f"captured {label} {item_id!r} was removed or rewritten"
                    )

    @staticmethod
    def _validate_result(
        work_item: ArtifactMeasurementWorkItem,
        result: ArtifactMeasurementResult,
    ) -> None:
        if not isinstance(result, ArtifactMeasurementResult):
            raise ArtifactMeasurementError(
                "worker result must be an ArtifactMeasurementResult"
            )
        if result.operation_id != work_item.id or result.kind is not work_item.kind:
            raise ArtifactMeasurementError(
                "worker result does not belong to the reserved measurement operation"
            )
        computation = result.computation
        if computation.context != work_item.context:
            raise ArtifactMeasurementError(
                "worker result context does not match the work item"
            )
        if computation.projection_snapshot != work_item.projection_snapshot:
            raise ArtifactMeasurementError(
                "worker result projection does not match the work item"
            )
        recipe = (
            computation.recipe_dict()
            if isinstance(computation, ArtifactVectorComputation)
            else computation.recipe_dict()
        )
        if _recipe_bytes(recipe) != work_item.recipe_json:
            raise ArtifactMeasurementError(
                "worker result recipe does not match the work item"
            )
        if _operation_kind_for_computation(computation) is not work_item.kind:
            raise ArtifactMeasurementError(
                "worker computation kind does not match the work item"
            )

    def _prepare_rebased_transition(
        self,
        work_item: ArtifactMeasurementWorkItem,
        result: ArtifactMeasurementResult,
    ) -> RecordBindingTransition:
        current = self._workbench.snapshot.session
        if not isinstance(current, ArtifactSession):
            raise StaleMeasurementOperationError(
                "measurement result has no active ArtifactDocument target"
            )
        self._workbench.require_stable_session(current, measurement=True)
        self._require_captured_document_ancestor(work_item, current)
        if self._durable_id_exists(current, work_item.record_id):
            raise ArtifactMeasurementError(
                f"reserved record ID {work_item.record_id!r} already exists"
            )
        for dependency_id in work_item.depends_on_record_ids:
            captured_dependency = work_item.captured_session.document.record_index.get(
                dependency_id
            )
            current_dependency = current.document.record_index.get(dependency_id)
            if current_dependency is None or current_dependency != captured_dependency:
                raise StaleMeasurementOperationError(
                    f"dependency record {dependency_id!r} changed after work began"
                )
            if current.document.record_freshness(dependency_id).value != "fresh":
                raise StaleMeasurementOperationError(
                    f"dependency record {dependency_id!r} is no longer fresh"
                )

        computation = result.computation
        if isinstance(computation, ArtifactVectorComputation):
            if not computation_matches_active_projection(current, computation):
                raise StaleMeasurementOperationError(
                    "vector result is stale for the active projection"
                )
            candidate = commit_vector_computation(
                current,
                computation,
                record_id=work_item.record_id,
                created_at=work_item.created_at,
                operator=work_item.operator,
                depends_on_record_ids=work_item.depends_on_record_ids,
            )
        elif isinstance(computation, ArtifactRubbingComputation):
            if not rubbing_computation_matches_active_projection(current, computation):
                raise StaleMeasurementOperationError(
                    "Digital Rubbing result is stale for the active projection"
                )
            candidate = commit_artifact_rubbing(
                current,
                computation,
                record_id=work_item.record_id,
                created_at=work_item.created_at,
                operator=work_item.operator,
                depends_on_record_ids=work_item.depends_on_record_ids,
            )
        elif isinstance(computation, ArtifactTileUnwrapComputation):
            if not tile_unwrap_computation_matches_active_projection(
                current, computation
            ):
                raise StaleMeasurementOperationError(
                    "tile unwrap result is stale for the active projection"
                )
            candidate = commit_artifact_tile_unwrap(
                current,
                computation,
                record_id=work_item.record_id,
                created_at=work_item.created_at,
                operator=work_item.operator,
                depends_on_record_ids=work_item.depends_on_record_ids,
            )
        elif isinstance(computation, ArtifactGeometryMetricsComputation):
            if not geometry_metrics_computation_matches_active_projection(
                current, computation
            ):
                raise StaleMeasurementOperationError(
                    "geometry metrics result is stale for the active projection"
                )
            candidate = commit_artifact_geometry_metrics(
                current,
                computation,
                record_id=work_item.record_id,
                created_at=work_item.created_at,
                operator=work_item.operator,
                depends_on_record_ids=work_item.depends_on_record_ids,
            )
        elif isinstance(computation, ArtifactSurfaceMeasurementComputation):
            if not surface_measurement_computation_matches_active_projection(
                current, computation
            ):
                raise StaleMeasurementOperationError(
                    "surface measurement result is stale for the active projection"
                )
            candidate = commit_artifact_surface_measurement(
                current,
                computation,
                record_id=work_item.record_id,
                created_at=work_item.created_at,
                operator=work_item.operator,
                depends_on_record_ids=work_item.depends_on_record_ids,
            )
        elif isinstance(computation, ConditionAnnotationComputation):
            if not condition_computation_matches_active_projection(
                current, computation
            ):
                raise StaleMeasurementOperationError(
                    "condition annotation result is stale for the active projection"
                )
            try:
                candidate = commit_condition_annotation(
                    current,
                    computation,
                    record_id=work_item.record_id,
                    created_at=work_item.created_at,
                    operator=work_item.operator,
                    depends_on_record_ids=work_item.depends_on_record_ids,
                )
            except ArtifactConditionAnnotationError as exc:
                raise ArtifactMeasurementError(str(exc)) from exc
        elif isinstance(computation, DevelopedRubbingComputation):
            if not developed_rubbing_computation_matches_active_projection(
                current, computation
            ):
                raise StaleMeasurementOperationError(
                    "developed rubbing result is stale for the active projection"
                )
            try:
                candidate = commit_developed_rubbing(
                    current,
                    computation,
                    record_id=work_item.record_id,
                    created_at=work_item.created_at,
                    operator=work_item.operator,
                    depends_on_record_ids=work_item.depends_on_record_ids,
                )
            except ArtifactDevelopedRubbingError as exc:
                raise ArtifactMeasurementError(str(exc)) from exc
        else:  # pragma: no cover - guarded by ArtifactMeasurementResult
            raise ArtifactMeasurementError("unsupported measurement computation")

        return self._workbench.prepare_record_commit(
            current,
            candidate,
            expected_new_record_ids=(work_item.record_id,),
            transition_id=work_item.id,
        )

    def _publication_can_retry(
        self,
        work_item: ArtifactMeasurementWorkItem,
        result: ArtifactMeasurementResult,
    ) -> bool:
        """Return whether a rolled-back publication may reuse its computation."""

        snapshot = self._workbench.snapshot
        current = snapshot.session
        if not isinstance(current, ArtifactSession):
            return False
        if snapshot.tentative or snapshot.faulted:
            return False
        # A pending Open only reserves the command slot.  Until its candidate
        # is finalized, the current session remains authoritative and this
        # computation must stay retryable if it still matches that session.
        if snapshot.pending_load is None and not snapshot.can_measure:
            return False
        try:
            self._require_captured_document_ancestor(work_item, current)
        except ArtifactWorkbenchError:
            return False
        if self._durable_id_exists(current, work_item.record_id):
            return False
        computation = result.computation
        if isinstance(computation, ArtifactVectorComputation):
            return computation_matches_active_projection(current, computation)
        if isinstance(computation, ArtifactRubbingComputation):
            return rubbing_computation_matches_active_projection(current, computation)
        if isinstance(computation, ArtifactTileUnwrapComputation):
            return tile_unwrap_computation_matches_active_projection(
                current, computation
            )
        if isinstance(computation, ArtifactGeometryMetricsComputation):
            return geometry_metrics_computation_matches_active_projection(
                current, computation
            )
        if isinstance(computation, ArtifactSurfaceMeasurementComputation):
            return surface_measurement_computation_matches_active_projection(
                current, computation
            )
        if isinstance(computation, DevelopedRubbingComputation):
            return developed_rubbing_computation_matches_active_projection(
                current, computation
            )
        return False

    def _candidate_was_published(
        self,
        candidate: ArtifactSession,
        record_id: str,
    ) -> bool:
        """Detect durable commit independently of later Open/readiness changes."""

        snapshot = self._workbench.snapshot
        if snapshot.tentative or not isinstance(snapshot.session, ArtifactSession):
            return False
        current = snapshot.session
        if (
            current.source_mesh is not candidate.source_mesh
            or current.document.document_id != candidate.document.document_id
            or current.document.schema_version != candidate.document.schema_version
        ):
            return False
        expected = candidate.document.record_index.get(record_id)
        return (
            expected is not None
            and current.document.record_index.get(record_id) == expected
        )

    def publish_result(
        self,
        work_item: ArtifactMeasurementWorkItem,
        result: ArtifactMeasurementResult,
        publisher: MeasurementPublisher,
    ) -> ArtifactMeasurementPublication:
        """Rebase and publish on the caller's application/UI dispatcher thread."""

        if not callable(publisher):
            raise ArtifactMeasurementError("publisher must be callable")
        with self._lock:
            runtime = self._require_runtime_locked(
                work_item,
                states=frozenset({MeasurementOperationState.RUNNING}),
            )
            if runtime.result is not result:
                raise ArtifactMeasurementError(
                    "publisher requires the exact result capability returned by execute()"
                )
            self._validate_result(work_item, result)

        candidate: ArtifactSession | None = None
        transition: RecordBindingTransition | None = None
        with self._publication_lock:
            with self._lock:
                runtime = self._require_runtime_locked(
                    work_item,
                    states=frozenset({MeasurementOperationState.RUNNING}),
                )
                runtime.state = MeasurementOperationState.PUBLISHING
            attempt = 0
            while True:
                candidate = None
                transition = None
                try:
                    transition = self._prepare_rebased_transition(work_item, result)
                    candidate = transition.candidate_session
                    publisher(transition)
                    if not self._candidate_was_published(
                        candidate,
                        work_item.record_id,
                    ):
                        raise ArtifactMeasurementError(
                            "measurement publisher returned without finalizing its record"
                        )
                    break
                except Exception as exc:
                    published = bool(
                        candidate is not None
                        and self._candidate_was_published(
                            candidate,
                            work_item.record_id,
                        )
                    )
                    if published:
                        _LOGGER.warning(
                            "Measurement publisher raised after record commit; "
                            "treating operation %s as completed",
                            work_item.id,
                            exc_info=True,
                        )
                        break

                    current_snapshot = self._workbench.snapshot
                    if current_snapshot.tentative:
                        owns_tentative = (
                            transition is not None
                            and candidate is not None
                            and current_snapshot.session is candidate
                        )
                        if owns_tentative:
                            assert transition is not None
                            try:
                                self._workbench.enter_faulted_state(
                                    session=None,
                                    project_path=None,
                                    error=RuntimeError(
                                        "measurement publisher exited with tentative authority"
                                    ),
                                    operation_id=transition.id,
                                )
                            finally:
                                with self._lock:
                                    active = self._active.get(work_item.id)
                                    if active is runtime:
                                        self._finish_locked(
                                            runtime,
                                            MeasurementOperationState.FAILED,
                                            exc,
                                        )
                            raise
                        # An unrelated Open/Align/record transaction owns the
                        # tentative state. Preserve both it and this result;
                        # its finalize/rollback notification will decide whether
                        # the measurement becomes stale or retryable.
                        with self._lock:
                            active = self._active.get(work_item.id)
                            if active is runtime:
                                runtime.state = MeasurementOperationState.RUNNING
                        raise

                    retryable = self._publication_can_retry(work_item, result)
                    with self._lock:
                        active = self._active.get(work_item.id)
                    if active is not runtime:
                        raise StaleMeasurementOperationError(
                            "measurement operation lost authority during publication"
                        ) from exc
                    if (
                        isinstance(exc, StaleWorkflowOperationError)
                        and retryable
                        and attempt < MAX_PUBLICATION_REBASE_ATTEMPTS
                    ):
                        attempt += 1
                        continue

                    with self._lock:
                        active = self._active.get(work_item.id)
                        if active is runtime:
                            if retryable:
                                runtime.state = MeasurementOperationState.RUNNING
                            else:
                                self._finish_locked(
                                    runtime,
                                    (
                                        MeasurementOperationState.STALE
                                        if isinstance(
                                            exc,
                                            (
                                                StaleMeasurementOperationError,
                                                StaleWorkflowOperationError,
                                            ),
                                        )
                                        else MeasurementOperationState.FAILED
                                    ),
                                    exc,
                                )
                    raise

            assert candidate is not None
            with self._lock:
                completion = self._finish_locked(
                    runtime,
                    MeasurementOperationState.COMPLETED,
                )
            if completion.state is not MeasurementOperationState.COMPLETED:
                raise self._terminal_exception(
                    completion.state,
                    completion.message,
                )

        align_id = candidate.document.active_align_revision_id
        if align_id is None:  # pragma: no cover - stable measurement requires Align
            raise ArtifactMeasurementError("published measurement has no active Align")
        return ArtifactMeasurementPublication(
            operation_id=work_item.id,
            kind=work_item.kind,
            record_id=work_item.record_id,
            session=candidate,
            document_sha256=candidate.document.canonical_sha256,
            align_revision_id=align_id,
        )


__all__ = [
    "ArtifactMeasurementController",
    "ArtifactMeasurementError",
    "ArtifactMeasurementPublication",
    "ArtifactMeasurementResult",
    "ArtifactMeasurementSummary",
    "ArtifactMeasurementWorkItem",
    "DEFAULT_RUBBING_MEMORY_BUDGET_BYTES",
    "MeasurementCancelledError",
    "MeasurementOperationKind",
    "MeasurementOperationState",
    "MeasurementResourceLimitError",
    "StaleMeasurementOperationError",
    "execute_measurement_work_item",
]
