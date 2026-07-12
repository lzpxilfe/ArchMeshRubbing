"""Qt-free staging and final-authority publication for artifact exports.

Export construction can be expensive, especially when a Digital Rubbing must
be reproduced from its durable recipe.  This controller keeps that work away
from the UI dispatcher and, more importantly, away from the authoritative
publication boundary.  A worker creates only a hidden, same-parent staging
directory.  The visible destination is created later by one no-replace rename
after :class:`ArtifactWorkbench` revalidates the exact record and unchanged
render projection.

Work items and results are identity capabilities.  Reconstructing an equal
dataclass does not grant publication or cleanup authority.  Staging cleanup is
delegated to the core ownership registry, which preserves paths whose inode was
replaced or which were not created by the matching public stage call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import os
from pathlib import Path
import stat
from threading import Event, RLock
import uuid

from src.core.artifact_document import (
    DerivedRecord,
    RecordFreshness,
    RecordLifecycleStatus,
)
from src.core.artifact_rubbing_export import (
    PreparedRubbingPublication,
    RUBBING_EXPORT_DIRECTORY_SUFFIX,
    discard_prepared_rubbing_package,
    discard_staged_rubbing_package,
    prepare_staged_rubbing_publication,
    publish_prepared_rubbing_package,
    stage_rubbing_package,
    validate_rubbing_export_package,
)
from src.core.artifact_rubbing_extractor import (
    compute_artifact_rubbing_from_recipe,
    estimate_digital_rubbing_resources,
    require_current_rubbing_computation,
)
from src.core.artifact_rubbing_record import (
    RUBBING_RECORD_TYPE,
    rubbing_receipt_from_record,
)
from src.core.artifact_scene_adapter import ArtifactProjectionSnapshot
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_export import (
    PreparedVectorPublication,
    VECTOR_EXPORT_DIRECTORY_SUFFIX,
    VectorSVGOptions,
    discard_prepared_vector_package,
    discard_staged_vector_package,
    prepare_staged_vector_publication,
    publish_prepared_vector_package,
    stage_vector_package,
    validate_vector_export_package,
)
from src.core.artifact_vector_record import VectorRecordKind

from .artifact_workbench import (
    ArtifactWorkbench,
    ArtifactWorkbenchError,
    StaleWorkflowOperationError,
    WorkflowBusyError,
)


DEFAULT_EXPORT_RUBBING_MEMORY_BUDGET_BYTES = 1024 * 1024 * 1024
_LOGGER = logging.getLogger(__name__)
PreparedExportPublication = PreparedVectorPublication | PreparedRubbingPublication


class ArtifactExportError(ArtifactWorkbenchError):
    """An export operation violated its application-level authority contract."""


class ExportCancelledError(ArtifactExportError):
    """An export capability was revoked before visible publication."""


class ExportResourceLimitError(ArtifactExportError):
    """Digital Rubbing reproduction exceeds the configured local budget."""


class StaleExportOperationError(StaleWorkflowOperationError):
    """The source, Align, record, or exact export capability is no longer current."""


class ArtifactExportKind(str, Enum):
    VECTOR = "vector"
    DIGITAL_RUBBING = "digital_rubbing"


class ArtifactExportState(str, Enum):
    READY = "ready"
    STAGING = "staging"
    STAGED = "staged"
    CANCELLING = "cancelling"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    STALE = "stale"


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4()}"


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactExportError(f"{field_name} must be a non-empty string")
    return value.strip()


def _destination_path(
    value: str | os.PathLike[str],
    *,
    kind: ArtifactExportKind,
) -> Path:
    try:
        path = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactExportError(f"export destination is invalid: {exc}") from exc
    suffix = (
        VECTOR_EXPORT_DIRECTORY_SUFFIX
        if kind is ArtifactExportKind.VECTOR
        else RUBBING_EXPORT_DIRECTORY_SUFFIX
    )
    if not path.name.endswith(suffix):
        raise ArtifactExportError(f"export destination must end with {suffix}")
    if path.exists() or path.is_symlink():
        raise ArtifactExportError("export destination already exists")
    return path


def _record_kind(record: DerivedRecord) -> ArtifactExportKind:
    if record.type == RUBBING_RECORD_TYPE:
        return ArtifactExportKind.DIGITAL_RUBBING
    if record.type in {kind.record_type for kind in VectorRecordKind}:
        return ArtifactExportKind.VECTOR
    raise ArtifactExportError("record type is not exportable by the artifact workbench")


def _require_exportable_record(
    session: ArtifactSession,
    record_id: str,
    *,
    kind: ArtifactExportKind,
) -> DerivedRecord:
    record_key = _required_text(record_id, field_name="record ID")
    record = session.document.record_index.get(record_key)
    if record is None:
        raise ArtifactExportError(f"record {record_key!r} does not exist")
    if _record_kind(record) is not kind:
        raise ArtifactExportError("record type does not match the requested export kind")
    if record.lifecycle_status is not RecordLifecycleStatus.READY:
        raise ArtifactExportError("only READY records may export")
    if session.document.record_freshness(record.id) is not RecordFreshness.FRESH:
        raise ArtifactExportError("only FRESH records may export")
    return record


@dataclass(frozen=True, slots=True, eq=False)
class ArtifactExportWorkItem:
    """Opaque immutable authority to stage exactly one record and destination."""

    id: str
    kind: ArtifactExportKind
    captured_session: ArtifactSession
    projection_snapshot: ArtifactProjectionSnapshot
    record_id: str
    expected_record: DerivedRecord
    destination: Path
    vector_options: VectorSVGOptions | None
    base_state_version: int
    base_authority_epoch: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, field_name="operation ID"))
        resolved_kind = ArtifactExportKind(self.kind)
        object.__setattr__(self, "kind", resolved_kind)
        if not isinstance(self.captured_session, ArtifactSession):
            raise ArtifactExportError("captured_session must be an ArtifactSession")
        if not isinstance(self.projection_snapshot, ArtifactProjectionSnapshot):
            raise ArtifactExportError(
                "projection_snapshot must be an ArtifactProjectionSnapshot"
            )
        if self.captured_session.projection_snapshot() != self.projection_snapshot:
            raise ArtifactExportError(
                "captured session does not match the export projection snapshot"
            )
        record_key = _required_text(self.record_id, field_name="record ID")
        object.__setattr__(self, "record_id", record_key)
        if not isinstance(self.expected_record, DerivedRecord):
            raise ArtifactExportError("expected_record must be a DerivedRecord")
        if (
            self.expected_record.id != record_key
            or self.captured_session.document.record_index.get(record_key)
            != self.expected_record
        ):
            raise ArtifactExportError(
                "expected record does not match the captured export document"
            )
        _require_exportable_record(
            self.captured_session,
            record_key,
            kind=resolved_kind,
        )
        resolved_destination = _destination_path(self.destination, kind=resolved_kind)
        object.__setattr__(self, "destination", resolved_destination)
        if resolved_kind is ArtifactExportKind.VECTOR:
            if self.vector_options is not None and not isinstance(
                self.vector_options, VectorSVGOptions
            ):
                raise ArtifactExportError(
                    "vector_options must be VectorSVGOptions or None"
                )
        elif self.vector_options is not None:
            raise ArtifactExportError(
                "Digital Rubbing exports cannot carry vector SVG options"
            )
        for field_name, value in (
            ("base_state_version", self.base_state_version),
            ("base_authority_epoch", self.base_authority_epoch),
        ):
            if type(value) is not int or value < 0:
                raise ArtifactExportError(f"{field_name} must be non-negative")


@dataclass(frozen=True, slots=True, eq=False)
class ArtifactExportResult:
    """Exact staged-package capability returned only by ``execute``."""

    operation_id: str
    kind: ArtifactExportKind
    staging_directory: Path
    prepared_publication: PreparedExportPublication | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_id",
            _required_text(self.operation_id, field_name="operation ID"),
        )
        object.__setattr__(self, "kind", ArtifactExportKind(self.kind))
        if not isinstance(self.staging_directory, Path):
            raise ArtifactExportError("staging_directory must be a Path")
        if self.prepared_publication is not None:
            if self.kind is ArtifactExportKind.VECTOR and not isinstance(
                self.prepared_publication,
                PreparedVectorPublication,
            ):
                raise ArtifactExportError(
                    "vector result requires a PreparedVectorPublication"
                )
            if self.kind is ArtifactExportKind.DIGITAL_RUBBING and not isinstance(
                self.prepared_publication,
                PreparedRubbingPublication,
            ):
                raise ArtifactExportError(
                    "rubbing result requires a PreparedRubbingPublication"
                )


@dataclass(frozen=True, slots=True)
class ArtifactExportPublication:
    operation_id: str
    kind: ArtifactExportKind
    record_id: str
    destination: Path
    document_sha256: str
    align_revision_id: str
    durability_confirmed: bool = True
    warning_message: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactExportSummary:
    operation_id: str
    kind: ArtifactExportKind
    state: ArtifactExportState
    record_id: str
    destination: Path
    error_type: str | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class _StagingIdentity:
    device: int
    inode: int


@dataclass(slots=True)
class _ExportRuntime:
    work_item: ArtifactExportWorkItem
    state: ArtifactExportState = ArtifactExportState.READY
    cancellation: Event = field(default_factory=Event)
    executing: bool = False
    result: ArtifactExportResult | None = None
    staging_identity: _StagingIdentity | None = None


class ArtifactExportController:
    """Own exact stage, cleanup, retry, and final publication capabilities."""

    def __init__(
        self,
        workbench: ArtifactWorkbench,
        *,
        id_factory: Callable[[str], str] = _new_id,
        rubbing_memory_budget_bytes: int = DEFAULT_EXPORT_RUBBING_MEMORY_BUDGET_BYTES,
    ) -> None:
        if not isinstance(workbench, ArtifactWorkbench):
            raise ArtifactExportError("workbench must be an ArtifactWorkbench")
        if not callable(id_factory):
            raise ArtifactExportError("id_factory must be callable")
        if (
            type(rubbing_memory_budget_bytes) is not int
            or rubbing_memory_budget_bytes <= 0
        ):
            raise ArtifactExportError(
                "rubbing_memory_budget_bytes must be a positive integer"
            )
        self._workbench = workbench
        self._id_factory = id_factory
        self._rubbing_memory_budget_bytes = rubbing_memory_budget_bytes
        self._lock = RLock()
        self._active: dict[str, _ExportRuntime] = {}
        self._history: dict[str, ArtifactExportSummary] = {}
        self._destination_reservations: dict[Path, str] = {}

    @property
    def workbench(self) -> ArtifactWorkbench:
        return self._workbench

    @property
    def active_summaries(self) -> tuple[ArtifactExportSummary, ...]:
        with self._lock:
            return tuple(
                self._summary_for_runtime(runtime)
                for _, runtime in sorted(self._active.items())
            )

    @staticmethod
    def _summary_for_runtime(runtime: _ExportRuntime) -> ArtifactExportSummary:
        item = runtime.work_item
        return ArtifactExportSummary(
            operation_id=item.id,
            kind=item.kind,
            state=runtime.state,
            record_id=item.record_id,
            destination=item.destination,
        )

    def summary(
        self,
        operation: ArtifactExportWorkItem | str,
    ) -> ArtifactExportSummary:
        operation_id = (
            operation.id
            if isinstance(operation, ArtifactExportWorkItem)
            else _required_text(operation, field_name="operation ID")
        )
        with self._lock:
            runtime = self._active.get(operation_id)
            if runtime is not None:
                if (
                    isinstance(operation, ArtifactExportWorkItem)
                    and runtime.work_item is not operation
                ):
                    raise StaleExportOperationError("export capability is stale")
                return self._summary_for_runtime(runtime)
            summary = self._history.get(operation_id)
            if summary is None:
                raise StaleExportOperationError("export operation is unknown")
            return summary

    def _require_runtime_locked(
        self,
        work_item: ArtifactExportWorkItem,
        *,
        states: frozenset[ArtifactExportState],
    ) -> _ExportRuntime:
        if not isinstance(work_item, ArtifactExportWorkItem):
            raise ArtifactExportError("work_item must be an ArtifactExportWorkItem")
        runtime = self._active.get(work_item.id)
        if runtime is None or runtime.work_item is not work_item:
            raise StaleExportOperationError("export operation capability is stale")
        if runtime.state not in states:
            raise StaleExportOperationError(
                f"export operation is already {runtime.state.value}"
            )
        return runtime

    def _finish_locked(
        self,
        runtime: _ExportRuntime,
        state: ArtifactExportState,
        error: BaseException | str | None = None,
    ) -> ArtifactExportSummary:
        work_item = runtime.work_item
        if self._active.get(work_item.id) is not runtime:
            known = self._history.get(work_item.id)
            if known is None:
                raise StaleExportOperationError("export operation capability is stale")
            return known
        runtime.state = state
        summary = ArtifactExportSummary(
            operation_id=work_item.id,
            kind=work_item.kind,
            state=state,
            record_id=work_item.record_id,
            destination=work_item.destination,
            error_type=(
                type(error).__name__
                if isinstance(error, BaseException)
                else ("Error" if error is not None else None)
            ),
            message=(str(error) if error is not None else None),
        )
        self._active.pop(work_item.id, None)
        if self._destination_reservations.get(work_item.destination) == work_item.id:
            self._destination_reservations.pop(work_item.destination, None)
        self._history[work_item.id] = summary
        return summary

    def _begin(
        self,
        kind: ArtifactExportKind,
        destination: str | os.PathLike[str],
        record_id: str,
        *,
        vector_options: VectorSVGOptions | None = None,
    ) -> ArtifactExportWorkItem:
        state = self._workbench.snapshot
        session = state.session
        if not isinstance(session, ArtifactSession):
            raise ArtifactExportError("no active ArtifactSession for export")
        self._workbench.require_stable_session(session, measurement=True)
        record = _require_exportable_record(session, record_id, kind=kind)
        resolved_destination = _destination_path(destination, kind=kind)
        projection_snapshot = session.projection_snapshot()
        operation_id = _required_text(
            self._id_factory("export"),
            field_name="generated operation ID",
        )
        work_item = ArtifactExportWorkItem(
            id=operation_id,
            kind=kind,
            captured_session=session,
            projection_snapshot=projection_snapshot,
            record_id=record.id,
            expected_record=record,
            destination=resolved_destination,
            vector_options=vector_options,
            base_state_version=state.state_version,
            base_authority_epoch=state.authority_epoch,
        )
        with self._lock:
            self._workbench.require_stable_session(session, measurement=True)
            current = session.document.record_index.get(record.id)
            if current != record:
                raise StaleExportOperationError(
                    "export record changed while its capability was captured"
                )
            if operation_id in self._active or operation_id in self._history:
                raise ArtifactExportError(
                    f"operation ID {operation_id!r} has already been used"
                )
            owner = self._destination_reservations.get(resolved_destination)
            if owner is not None:
                raise ArtifactExportError(
                    f"export destination is already reserved by {owner!r}"
                )
            runtime = _ExportRuntime(work_item=work_item, cancellation=Event())
            self._active[operation_id] = runtime
            self._destination_reservations[resolved_destination] = operation_id
        return work_item

    def begin_vector(
        self,
        destination: str | os.PathLike[str],
        record_id: str,
        *,
        options: VectorSVGOptions | None = None,
    ) -> ArtifactExportWorkItem:
        return self._begin(
            ArtifactExportKind.VECTOR,
            destination,
            record_id,
            vector_options=options,
        )

    def begin_rubbing(
        self,
        destination: str | os.PathLike[str],
        record_id: str,
    ) -> ArtifactExportWorkItem:
        return self._begin(
            ArtifactExportKind.DIGITAL_RUBBING,
            destination,
            record_id,
        )

    @staticmethod
    def _capture_staging_identity(path: Path) -> _StagingIdentity:
        try:
            identity = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise ArtifactExportError(
                f"cannot inspect staged export package: {exc}"
            ) from exc
        if not stat.S_ISDIR(identity.st_mode):
            raise ArtifactExportError("staged export package is not a real directory")
        return _StagingIdentity(device=identity.st_dev, inode=identity.st_ino)

    def _stage(self, work_item: ArtifactExportWorkItem) -> Path:
        session = work_item.captured_session
        record = work_item.expected_record
        if work_item.kind is ArtifactExportKind.VECTOR:
            return stage_vector_package(
                work_item.destination,
                session.document,
                record.id,
                options=work_item.vector_options,
            )

        snapshot = work_item.projection_snapshot
        mesh = session.source_mesh
        estimate = estimate_digital_rubbing_resources(
            mesh.vertices,
            mesh.faces,
            record.recipe,
            source_to_world_mm_matrix4x4=snapshot.matrix4x4,
            uv_coords=mesh.uv_coords,
            texture=mesh.texture,
        )
        if estimate.estimated_peak_bytes > self._rubbing_memory_budget_bytes:
            raise ExportResourceLimitError(
                "Digital Rubbing reproduction estimated peak memory "
                f"{estimate.estimated_peak_bytes} bytes exceeds the configured "
                f"budget {self._rubbing_memory_budget_bytes} bytes"
            )
        computation = compute_artifact_rubbing_from_recipe(session, record.recipe)
        require_current_rubbing_computation(session, computation)
        if computation.raster.receipt() != rubbing_receipt_from_record(record):
            raise ArtifactExportError(
                "recomputed Digital Rubbing does not match its durable receipt"
            )
        return stage_rubbing_package(
            work_item.destination,
            session.document,
            record.id,
            computation.raster,
        )

    @staticmethod
    def _raise_if_cancelled(runtime: _ExportRuntime) -> None:
        if runtime.cancellation.is_set():
            raise ExportCancelledError(
                "export operation was cancelled before visible publication"
            )

    @staticmethod
    def _core_error(error: BaseException) -> ArtifactExportError:
        if isinstance(error, ArtifactExportError):
            return error
        return ArtifactExportError(str(error))

    def _discard_staging(
        self,
        work_item: ArtifactExportWorkItem,
        result: ArtifactExportResult,
        *,
        prepared: PreparedExportPublication | None = None,
    ) -> bool:
        if prepared is None:
            prepared = result.prepared_publication
        if isinstance(prepared, PreparedVectorPublication):
            discarded = discard_prepared_vector_package(prepared)
        elif isinstance(prepared, PreparedRubbingPublication):
            discarded = discard_prepared_rubbing_package(prepared)
        elif work_item.kind is ArtifactExportKind.VECTOR:
            discarded = discard_staged_vector_package(
                result.staging_directory, work_item.destination
            )
        else:
            discarded = discard_staged_rubbing_package(
                result.staging_directory, work_item.destination
            )
        if not discarded:
            raise ArtifactExportError(
                "staging cleanup was not proven; an unowned or replaced path was preserved"
            )
        return True

    @staticmethod
    def _prepare_staged_publication(
        work_item: ArtifactExportWorkItem,
        staging_directory: Path,
    ) -> PreparedExportPublication:
        if work_item.kind is ArtifactExportKind.VECTOR:
            return prepare_staged_vector_publication(
                staging_directory,
                work_item.destination,
                document=work_item.captured_session.document,
            )
        return prepare_staged_rubbing_publication(
            staging_directory,
            work_item.destination,
            document=work_item.captured_session.document,
        )

    @staticmethod
    def _publish_prepared(
        prepared: PreparedExportPublication,
    ) -> Path:
        if isinstance(prepared, PreparedVectorPublication):
            return publish_prepared_vector_package(prepared)
        return publish_prepared_rubbing_package(prepared)

    def execute(self, work_item: ArtifactExportWorkItem) -> ArtifactExportResult:
        """Build and verify only a hidden staging package on a worker thread."""

        with self._lock:
            runtime = self._require_runtime_locked(
                work_item,
                states=frozenset({ArtifactExportState.READY}),
            )
            if runtime.executing or runtime.result is not None:
                raise StaleExportOperationError(
                    "export operation has already been executed"
                )
            runtime.executing = True
            runtime.state = ArtifactExportState.STAGING

        result: ArtifactExportResult | None = None
        cleanup_result: ArtifactExportResult | None = None
        prepared: PreparedExportPublication | None = None
        try:
            self._raise_if_cancelled(runtime)
            staging = self._stage(work_item)
            cleanup_result = ArtifactExportResult(
                operation_id=work_item.id,
                kind=work_item.kind,
                staging_directory=staging,
            )
            staging_identity = self._capture_staging_identity(staging)
            prepared = self._prepare_staged_publication(work_item, staging)
            result = ArtifactExportResult(
                operation_id=work_item.id,
                kind=work_item.kind,
                staging_directory=staging,
                prepared_publication=prepared,
            )
            self._raise_if_cancelled(runtime)
        except Exception as exc:
            cleanup_error: BaseException | None = None
            if cleanup_result is not None:
                try:
                    self._discard_staging(
                        work_item,
                        cleanup_result,
                        prepared=prepared,
                    )
                except Exception as discard_exc:  # pragma: no cover - rare I/O failure
                    cleanup_error = discard_exc
                    _LOGGER.exception("Failed to discard staged export after worker error")
            with self._lock:
                active = self._active.get(work_item.id)
                if active is runtime:
                    runtime.executing = False
                    cancelled = runtime.cancellation.is_set() or isinstance(
                        exc, ExportCancelledError
                    )
                    terminal = (
                        ArtifactExportState.CANCELLED
                        if cancelled and cleanup_error is None
                        else ArtifactExportState.FAILED
                    )
                    self._finish_locked(runtime, terminal, cleanup_error or exc)
            if cleanup_error is not None:
                raise ArtifactExportError(
                    "export failed and its owned staging package could not be discarded"
                ) from cleanup_error
            if runtime.cancellation.is_set() or isinstance(exc, ExportCancelledError):
                raise ExportCancelledError(
                    "export operation was cancelled before visible publication"
                ) from exc
            error = self._core_error(exc)
            if error is exc:
                raise
            raise error from exc

        with self._lock:
            active = self._active.get(work_item.id)
            if active is not runtime:
                should_cancel = True
            else:
                runtime.executing = False
                should_cancel = (
                    runtime.cancellation.is_set()
                    or runtime.state is ArtifactExportState.CANCELLING
                )
                if not should_cancel:
                    runtime.result = result
                    runtime.staging_identity = staging_identity
                    runtime.state = ArtifactExportState.STAGED
                    return result

        assert result is not None
        try:
            self._discard_staging(work_item, result)
        except Exception as exc:
            with self._lock:
                if self._active.get(work_item.id) is runtime:
                    self._finish_locked(runtime, ArtifactExportState.FAILED, exc)
            raise ArtifactExportError(
                "cancelled export staging package could not be discarded"
            ) from exc
        with self._lock:
            if self._active.get(work_item.id) is runtime:
                self._finish_locked(
                    runtime,
                    ArtifactExportState.CANCELLED,
                    "export cancelled while staging",
                )
        raise ExportCancelledError(
            "export operation was cancelled before visible publication"
        )

    def _validate_exact_result_locked(
        self,
        runtime: _ExportRuntime,
        result: ArtifactExportResult,
    ) -> None:
        if not isinstance(result, ArtifactExportResult):
            raise ArtifactExportError("result must be an ArtifactExportResult")
        if runtime.result is not result:
            raise ArtifactExportError(
                "publisher requires the exact result capability returned by execute()"
            )
        work_item = runtime.work_item
        if (
            result.operation_id != work_item.id
            or result.kind is not work_item.kind
        ):
            raise ArtifactExportError("export result does not match its work item")
        prepared = result.prepared_publication
        if prepared is None:
            raise ArtifactExportError(
                "export result is missing its exact prepared publication capability"
            )
        if (
            prepared.staging_directory != result.staging_directory
            or prepared.destination != work_item.destination
        ):
            raise ArtifactExportError(
                "prepared publication does not match the staged export result"
            )

    def _published_destination_is_owned(
        self,
        runtime: _ExportRuntime,
        result: ArtifactExportResult,
    ) -> bool:
        identity = runtime.staging_identity
        if identity is None or result.staging_directory.exists():
            return False
        try:
            current = runtime.work_item.destination.stat(follow_symlinks=False)
        except OSError:
            return False
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_dev != identity.device
            or current.st_ino != identity.inode
        ):
            return False
        try:
            if runtime.work_item.kind is ArtifactExportKind.VECTOR:
                validate_vector_export_package(
                    runtime.work_item.destination,
                    document=runtime.work_item.captured_session.document,
                )
            else:
                validate_rubbing_export_package(
                    runtime.work_item.destination,
                    document=runtime.work_item.captured_session.document,
                )
        except Exception:
            return False
        return True

    def _captured_authority_is_stale(
        self,
        work_item: ArtifactExportWorkItem,
    ) -> bool:
        current = self._workbench.snapshot.session
        if not isinstance(current, ArtifactSession):
            return True
        if current.source_mesh is not work_item.captured_session.source_mesh:
            return True
        try:
            current_snapshot = current.projection_snapshot()
        except Exception:
            return True
        if not work_item.projection_snapshot.has_same_render_projection(
            current_snapshot
        ):
            return True
        return (
            current.document.record_index.get(work_item.record_id)
            != work_item.expected_record
        )

    @staticmethod
    def _publication(
        work_item: ArtifactExportWorkItem,
        *,
        durability_confirmed: bool = True,
        warning_message: str | None = None,
    ) -> ArtifactExportPublication:
        return ArtifactExportPublication(
            operation_id=work_item.id,
            kind=work_item.kind,
            record_id=work_item.record_id,
            destination=work_item.destination,
            document_sha256=work_item.captured_session.document.canonical_sha256,
            align_revision_id=work_item.projection_snapshot.align_revision_id,
            durability_confirmed=durability_confirmed,
            warning_message=warning_message,
        )

    def publish_result(
        self,
        work_item: ArtifactExportWorkItem,
        result: ArtifactExportResult,
    ) -> ArtifactExportPublication:
        """Publish visibly only after the Workbench final authority fence."""

        with self._lock:
            runtime = self._require_runtime_locked(
                work_item,
                states=frozenset({ArtifactExportState.STAGED}),
            )
            self._validate_exact_result_locked(runtime, result)
            prepared = result.prepared_publication
            if prepared is None:  # guarded by _validate_exact_result_locked
                raise ArtifactExportError(
                    "export result is missing its prepared publication capability"
                )
            runtime.state = ArtifactExportState.PUBLISHING

        authority_callback_entered = False
        try:
            def publish_exact_prepared_capability() -> Path:
                nonlocal authority_callback_entered
                authority_callback_entered = True
                return self._publish_prepared(prepared)

            published = self._workbench.publish_record_effect_if_current(
                work_item.captured_session,
                work_item.projection_snapshot,
                record_id=work_item.record_id,
                expected_record=work_item.expected_record,
                publish=publish_exact_prepared_capability,
            )
            if Path(published) != work_item.destination:
                raise ArtifactExportError(
                    "core exporter returned an unexpected publication destination"
                )
        except WorkflowBusyError:
            with self._lock:
                if self._active.get(work_item.id) is runtime:
                    runtime.state = ArtifactExportState.STAGED
            raise
        except StaleWorkflowOperationError as exc:
            try:
                self._discard_staging(
                    work_item,
                    result,
                    prepared=prepared,
                )
            except Exception as cleanup_exc:
                with self._lock:
                    if self._active.get(work_item.id) is runtime:
                        self._finish_locked(
                            runtime,
                            ArtifactExportState.FAILED,
                            cleanup_exc,
                        )
                if getattr(cleanup_exc, "committed", False):
                    raise ArtifactExportError(
                        "export staging became visible before final authority "
                        "publication; the destination was preserved for forensic "
                        "inspection"
                    ) from cleanup_exc
                raise ArtifactExportError(
                    "stale export was revoked but staging cleanup failed"
                ) from cleanup_exc
            with self._lock:
                if self._active.get(work_item.id) is runtime:
                    self._finish_locked(runtime, ArtifactExportState.STALE, exc)
            raise StaleExportOperationError(str(exc)) from exc
        except Exception as exc:
            committed_after_authority = bool(
                authority_callback_entered and getattr(exc, "committed", False)
            )
            if (
                committed_after_authority
                and self._published_destination_is_owned(runtime, result)
            ):
                _LOGGER.warning(
                    "Export rename committed before a durability/reporting error; "
                    "treating operation %s as completed",
                    work_item.id,
                    exc_info=True,
                )
                with self._lock:
                    if self._active.get(work_item.id) is runtime:
                        self._finish_locked(runtime, ArtifactExportState.COMPLETED)
                warning = str(exc) or (
                    "export was published, but crash durability could not be confirmed"
                )
                return self._publication(
                    work_item,
                    durability_confirmed=False,
                    warning_message=warning,
                )
            if not authority_callback_entered and getattr(exc, "committed", False):
                error = ArtifactExportError(
                    "export staging became visible before final authority publication; "
                    "the destination was preserved for forensic inspection"
                )
                with self._lock:
                    if self._active.get(work_item.id) is runtime:
                        self._finish_locked(runtime, ArtifactExportState.FAILED, error)
                raise error from exc
            if (
                isinstance(exc, ArtifactWorkbenchError)
                and self._captured_authority_is_stale(work_item)
            ):
                try:
                    self._discard_staging(
                        work_item,
                        result,
                        prepared=prepared,
                    )
                except Exception as cleanup_exc:
                    with self._lock:
                        if self._active.get(work_item.id) is runtime:
                            self._finish_locked(
                                runtime,
                                ArtifactExportState.FAILED,
                            cleanup_exc,
                        )
                    if getattr(cleanup_exc, "committed", False):
                        raise ArtifactExportError(
                            "export staging became visible before final authority "
                            "publication; the destination was preserved for forensic "
                            "inspection"
                        ) from cleanup_exc
                    raise ArtifactExportError(
                        "stale export was revoked but staging cleanup failed"
                    ) from cleanup_exc
                with self._lock:
                    if self._active.get(work_item.id) is runtime:
                        self._finish_locked(runtime, ArtifactExportState.STALE, exc)
                raise StaleExportOperationError(str(exc)) from exc
            try:
                self._discard_staging(
                    work_item,
                    result,
                    prepared=prepared,
                )
            except Exception as cleanup_exc:
                with self._lock:
                    if self._active.get(work_item.id) is runtime:
                        self._finish_locked(
                            runtime,
                            ArtifactExportState.FAILED,
                            cleanup_exc,
                        )
                raise ArtifactExportError(
                    "export publication failed and staging cleanup also failed"
                ) from cleanup_exc
            with self._lock:
                if self._active.get(work_item.id) is runtime:
                    self._finish_locked(runtime, ArtifactExportState.FAILED, exc)
            error = self._core_error(exc)
            if error is exc:
                raise
            raise error from exc

        if not self._published_destination_is_owned(runtime, result):
            error = ArtifactExportError(
                "export publisher returned without the owned package at its destination"
            )
            with self._lock:
                if self._active.get(work_item.id) is runtime:
                    self._finish_locked(runtime, ArtifactExportState.FAILED, error)
            raise error
        with self._lock:
            if self._active.get(work_item.id) is runtime:
                self._finish_locked(runtime, ArtifactExportState.COMPLETED)
        return self._publication(work_item)

    def discard_result(
        self,
        work_item: ArtifactExportWorkItem,
        result: ArtifactExportResult,
        *,
        reason: str = "discarded",
    ) -> ArtifactExportSummary:
        """Revoke an exact staged result and remove only its owned temp path."""

        resolved_reason = _required_text(reason, field_name="discard reason")
        with self._lock:
            runtime = self._require_runtime_locked(
                work_item,
                states=frozenset({ArtifactExportState.STAGED}),
            )
            self._validate_exact_result_locked(runtime, result)
            runtime.state = ArtifactExportState.CANCELLING
        try:
            self._discard_staging(work_item, result)
        except Exception as exc:
            with self._lock:
                if self._active.get(work_item.id) is runtime:
                    self._finish_locked(runtime, ArtifactExportState.FAILED, exc)
            raise ArtifactExportError(
                f"staged export could not be discarded safely: {exc}"
            ) from exc
        with self._lock:
            return self._finish_locked(
                runtime,
                ArtifactExportState.CANCELLED,
                resolved_reason,
            )

    def cancel(
        self,
        work_item: ArtifactExportWorkItem,
        *,
        reason: str = "cancelled",
    ) -> ArtifactExportSummary:
        """Cooperatively cancel compute or discard an already staged result."""

        resolved_reason = _required_text(reason, field_name="cancel reason")
        staged_result: ArtifactExportResult | None = None
        with self._lock:
            runtime = self._require_runtime_locked(
                work_item,
                states=frozenset(
                    {
                        ArtifactExportState.READY,
                        ArtifactExportState.STAGING,
                        ArtifactExportState.STAGED,
                    }
                ),
            )
            runtime.cancellation.set()
            if runtime.executing or runtime.state is ArtifactExportState.STAGING:
                runtime.state = ArtifactExportState.CANCELLING
                return self._summary_for_runtime(runtime)
            staged_result = runtime.result
            runtime.state = ArtifactExportState.CANCELLING

        if staged_result is not None:
            try:
                self._discard_staging(work_item, staged_result)
            except Exception as exc:
                with self._lock:
                    if self._active.get(work_item.id) is runtime:
                        self._finish_locked(runtime, ArtifactExportState.FAILED, exc)
                raise ArtifactExportError(
                    f"cancelled export could not discard its staging package: {exc}"
                ) from exc
        with self._lock:
            return self._finish_locked(
                runtime,
                ArtifactExportState.CANCELLED,
                resolved_reason,
            )


__all__ = [
    "ArtifactExportController",
    "ArtifactExportError",
    "ArtifactExportKind",
    "ArtifactExportPublication",
    "ArtifactExportResult",
    "ArtifactExportState",
    "ArtifactExportSummary",
    "ArtifactExportWorkItem",
    "DEFAULT_EXPORT_RUBBING_MEMORY_BUDGET_BYTES",
    "ExportCancelledError",
    "ExportResourceLimitError",
    "StaleExportOperationError",
]
