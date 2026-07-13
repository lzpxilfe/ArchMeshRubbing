"""Qt-free complete-survey staging and final-authority publication.

This controller selects one dependency-valid READY+FRESH record for every
required 3/6/6 view, estimates each Digital Rubbing reproduction against a
bounded local memory budget, and builds one hidden ``*.amr-survey`` tree.  The
tree becomes visible only after the Workbench confirms that all fifteen exact
records and the render projection are still authoritative.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import stat
from threading import Event, RLock
import uuid

from src.core.artifact_document import DerivedRecord
from src.core.artifact_rubbing_extractor import estimate_digital_rubbing_resources
from src.core.artifact_scene_adapter import ArtifactProjectionSnapshot
from src.core.artifact_session import ArtifactSession
from src.core.artifact_survey_export import (
    SURVEY_EXPORT_DIRECTORY_SUFFIX,
    ArtifactSurveyExportError,
    PreparedSurveyPublication,
    SurveyExportSelection,
    discard_prepared_survey_package,
    discard_staged_survey_package,
    prepare_staged_survey_publication,
    publish_prepared_survey_package,
    stage_survey_export_package,
    validate_survey_export_package,
)

from .artifact_exports import (
    DEFAULT_EXPORT_RUBBING_MEMORY_BUDGET_BYTES,
    ArtifactExportError,
    ArtifactExportState,
    ExportCancelledError,
    ExportResourceLimitError,
    StaleExportOperationError,
)
from .artifact_workbench import (
    ArtifactWorkbench,
    ArtifactWorkbenchError,
    StaleWorkflowOperationError,
    WorkflowBusyError,
)
from .artifact_workflow_progress import (
    ArtifactWorkflowStep,
    workflow_step_record_ids,
)


_LOGGER = logging.getLogger(__name__)


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4()}"


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactExportError(f"{field_name} must be a non-empty string")
    return value.strip()


def _destination_path(value: str | os.PathLike[str]) -> Path:
    try:
        path = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactExportError(f"survey destination is invalid: {exc}") from exc
    if not path.name.endswith(SURVEY_EXPORT_DIRECTORY_SUFFIX):
        raise ArtifactExportError(
            f"survey destination must end with {SURVEY_EXPORT_DIRECTORY_SUFFIX}"
        )
    if os.path.lexists(path):
        raise ArtifactExportError("survey destination already exists")
    return path


@dataclass(frozen=True, slots=True, eq=False)
class ArtifactSurveyExportWorkItem:
    id: str
    captured_session: ArtifactSession
    projection_snapshot: ArtifactProjectionSnapshot
    selection: SurveyExportSelection
    expected_records: tuple[DerivedRecord, ...]
    destination: Path
    base_state_version: int
    base_authority_epoch: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, field_name="operation ID"))
        if not isinstance(self.captured_session, ArtifactSession):
            raise ArtifactExportError("captured_session must be an ArtifactSession")
        if not isinstance(self.projection_snapshot, ArtifactProjectionSnapshot):
            raise ArtifactExportError(
                "projection_snapshot must be an ArtifactProjectionSnapshot"
            )
        if self.captured_session.projection_snapshot() != self.projection_snapshot:
            raise ArtifactExportError(
                "captured survey session does not match its projection snapshot"
            )
        if not isinstance(self.selection, SurveyExportSelection):
            raise ArtifactExportError("selection must be a SurveyExportSelection")
        records = tuple(self.expected_records)
        if len(records) != 15 or any(
            not isinstance(record, DerivedRecord) for record in records
        ):
            raise ArtifactExportError(
                "expected_records must contain exactly fifteen DerivedRecords"
            )
        if tuple(record.id for record in records) != self.selection.record_ids:
            raise ArtifactExportError(
                "expected survey records do not match the canonical selection"
            )
        for record in records:
            if self.captured_session.document.record_index.get(record.id) != record:
                raise ArtifactExportError(
                    "expected survey record does not match the captured document"
                )
        object.__setattr__(self, "expected_records", records)
        object.__setattr__(self, "destination", _destination_path(self.destination))
        for field_name, value in (
            ("base_state_version", self.base_state_version),
            ("base_authority_epoch", self.base_authority_epoch),
        ):
            if type(value) is not int or value < 0:
                raise ArtifactExportError(f"{field_name} must be non-negative")


@dataclass(frozen=True, slots=True, eq=False)
class ArtifactSurveyExportResult:
    operation_id: str
    staging_directory: Path
    prepared_publication: PreparedSurveyPublication | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_id",
            _required_text(self.operation_id, field_name="operation ID"),
        )
        if not isinstance(self.staging_directory, Path):
            raise ArtifactExportError("staging_directory must be a Path")
        if self.prepared_publication is not None and not isinstance(
            self.prepared_publication,
            PreparedSurveyPublication,
        ):
            raise ArtifactExportError(
                "survey result requires a PreparedSurveyPublication"
            )


@dataclass(frozen=True, slots=True)
class ArtifactSurveyExportPublication:
    operation_id: str
    destination: Path
    document_sha256: str
    align_revision_id: str
    record_ids: tuple[str, ...]
    durability_confirmed: bool = True
    warning_message: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactSurveyExportSummary:
    operation_id: str
    state: ArtifactExportState
    destination: Path
    record_count: int
    error_type: str | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class _StagingIdentity:
    device: int
    inode: int


@dataclass(slots=True)
class _SurveyRuntime:
    work_item: ArtifactSurveyExportWorkItem
    state: ArtifactExportState = ArtifactExportState.READY
    cancellation: Event = field(default_factory=Event)
    executing: bool = False
    result: ArtifactSurveyExportResult | None = None
    staging_identity: _StagingIdentity | None = None


class ArtifactSurveyExportController:
    """Own one-use capabilities for atomic complete-survey publication."""

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
        self._active: dict[str, _SurveyRuntime] = {}
        self._history: dict[str, ArtifactSurveyExportSummary] = {}
        self._destination_reservations: dict[Path, str] = {}

    @property
    def workbench(self) -> ArtifactWorkbench:
        return self._workbench

    @property
    def active_summaries(self) -> tuple[ArtifactSurveyExportSummary, ...]:
        with self._lock:
            return tuple(
                self._summary_for_runtime(runtime)
                for _, runtime in sorted(self._active.items())
            )

    @staticmethod
    def _summary_for_runtime(
        runtime: _SurveyRuntime,
    ) -> ArtifactSurveyExportSummary:
        item = runtime.work_item
        return ArtifactSurveyExportSummary(
            operation_id=item.id,
            state=runtime.state,
            destination=item.destination,
            record_count=len(item.expected_records),
        )

    def summary(
        self,
        operation: ArtifactSurveyExportWorkItem | str,
    ) -> ArtifactSurveyExportSummary:
        operation_id = (
            operation.id
            if isinstance(operation, ArtifactSurveyExportWorkItem)
            else _required_text(operation, field_name="operation ID")
        )
        with self._lock:
            runtime = self._active.get(operation_id)
            if runtime is not None:
                if (
                    isinstance(operation, ArtifactSurveyExportWorkItem)
                    and runtime.work_item is not operation
                ):
                    raise StaleExportOperationError(
                        "survey export capability is stale"
                    )
                return self._summary_for_runtime(runtime)
            summary = self._history.get(operation_id)
            if summary is None:
                raise StaleExportOperationError("survey export operation is unknown")
            return summary

    def _require_runtime_locked(
        self,
        work_item: ArtifactSurveyExportWorkItem,
        *,
        states: frozenset[ArtifactExportState],
    ) -> _SurveyRuntime:
        if not isinstance(work_item, ArtifactSurveyExportWorkItem):
            raise ArtifactExportError(
                "work_item must be an ArtifactSurveyExportWorkItem"
            )
        runtime = self._active.get(work_item.id)
        if runtime is None or runtime.work_item is not work_item:
            raise StaleExportOperationError(
                "survey export operation capability is stale"
            )
        if runtime.state not in states:
            raise StaleExportOperationError(
                f"survey export operation is already {runtime.state.value}"
            )
        return runtime

    def _finish_locked(
        self,
        runtime: _SurveyRuntime,
        state: ArtifactExportState,
        error: BaseException | str | None = None,
    ) -> ArtifactSurveyExportSummary:
        item = runtime.work_item
        if self._active.get(item.id) is not runtime:
            known = self._history.get(item.id)
            if known is None:
                raise StaleExportOperationError(
                    "survey export operation capability is stale"
                )
            return known
        runtime.state = state
        summary = ArtifactSurveyExportSummary(
            operation_id=item.id,
            state=state,
            destination=item.destination,
            record_count=len(item.expected_records),
            error_type=(
                type(error).__name__
                if isinstance(error, BaseException)
                else ("Error" if error is not None else None)
            ),
            message=str(error) if error is not None else None,
        )
        self._active.pop(item.id, None)
        if self._destination_reservations.get(item.destination) == item.id:
            self._destination_reservations.pop(item.destination, None)
        self._history[item.id] = summary
        return summary

    @staticmethod
    def _selection(session: ArtifactSession) -> SurveyExportSelection:
        return SurveyExportSelection(
            cutline_record_ids=workflow_step_record_ids(
                session,
                ArtifactWorkflowStep.CUTLINE,
            ),
            outline_record_ids=workflow_step_record_ids(
                session,
                ArtifactWorkflowStep.OUTLINE,
            ),
            rubbing_record_ids=workflow_step_record_ids(
                session,
                ArtifactWorkflowStep.DIGITAL_RUBBING,
            ),
        )

    def begin(
        self,
        destination: str | os.PathLike[str],
    ) -> ArtifactSurveyExportWorkItem:
        state = self._workbench.snapshot
        session = state.session
        if not isinstance(session, ArtifactSession):
            raise ArtifactExportError("no active ArtifactSession for survey export")
        self._workbench.require_stable_session(session, measurement=True)
        try:
            selection = self._selection(session)
        except (ArtifactSurveyExportError, TypeError, ValueError) as exc:
            raise ArtifactExportError(
                f"complete survey requires dependency-valid 3/6/6 records: {exc}"
            ) from exc
        records = tuple(
            session.document.record_index[record_id]
            for record_id in selection.record_ids
        )
        resolved_destination = _destination_path(destination)
        operation_id = _required_text(
            self._id_factory("survey-export"),
            field_name="generated operation ID",
        )
        work_item = ArtifactSurveyExportWorkItem(
            id=operation_id,
            captured_session=session,
            projection_snapshot=session.projection_snapshot(),
            selection=selection,
            expected_records=records,
            destination=resolved_destination,
            base_state_version=state.state_version,
            base_authority_epoch=state.authority_epoch,
        )
        with self._lock:
            self._workbench.require_stable_session(session, measurement=True)
            if any(
                session.document.record_index.get(record.id) != record
                for record in records
            ):
                raise StaleExportOperationError(
                    "survey records changed while the capability was captured"
                )
            if operation_id in self._active or operation_id in self._history:
                raise ArtifactExportError(
                    f"operation ID {operation_id!r} has already been used"
                )
            owner = self._destination_reservations.get(resolved_destination)
            if owner is not None:
                raise ArtifactExportError(
                    f"survey destination is already reserved by {owner!r}"
                )
            runtime = _SurveyRuntime(work_item=work_item)
            self._active[operation_id] = runtime
            self._destination_reservations[resolved_destination] = operation_id
        return work_item

    @staticmethod
    def _raise_if_cancelled(runtime: _SurveyRuntime) -> None:
        if runtime.cancellation.is_set():
            raise ExportCancelledError(
                "survey export was cancelled before visible publication"
            )

    def _preflight_resources(
        self,
        work_item: ArtifactSurveyExportWorkItem,
        runtime: _SurveyRuntime,
    ) -> None:
        session = work_item.captured_session
        mesh = session.source_mesh
        snapshot = work_item.projection_snapshot
        for record_id in work_item.selection.rubbing_record_ids:
            self._raise_if_cancelled(runtime)
            record = session.document.record_index[record_id]
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

    @staticmethod
    def _capture_staging_identity(path: Path) -> _StagingIdentity:
        try:
            value = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise ArtifactExportError(
                f"cannot inspect staged survey package: {exc}"
            ) from exc
        if not stat.S_ISDIR(value.st_mode):
            raise ArtifactExportError("staged survey package is not a real directory")
        return _StagingIdentity(device=value.st_dev, inode=value.st_ino)

    @staticmethod
    def _discard_staging(
        work_item: ArtifactSurveyExportWorkItem,
        result: ArtifactSurveyExportResult,
    ) -> bool:
        prepared = result.prepared_publication
        discarded = (
            discard_prepared_survey_package(prepared)
            if prepared is not None
            else discard_staged_survey_package(
                result.staging_directory,
                work_item.destination,
            )
        )
        if not discarded:
            raise ArtifactExportError(
                "survey staging cleanup was not proven; the path was preserved"
            )
        return True

    def execute(
        self,
        work_item: ArtifactSurveyExportWorkItem,
    ) -> ArtifactSurveyExportResult:
        with self._lock:
            runtime = self._require_runtime_locked(
                work_item,
                states=frozenset({ArtifactExportState.READY}),
            )
            if runtime.executing or runtime.result is not None:
                raise StaleExportOperationError(
                    "survey export operation has already been executed"
                )
            runtime.executing = True
            runtime.state = ArtifactExportState.STAGING

        result: ArtifactSurveyExportResult | None = None
        cleanup_result: ArtifactSurveyExportResult | None = None
        prepared: PreparedSurveyPublication | None = None
        try:
            self._raise_if_cancelled(runtime)
            self._preflight_resources(work_item, runtime)
            staging = stage_survey_export_package(
                work_item.destination,
                work_item.captured_session,
                work_item.selection,
                cancellation_probe=runtime.cancellation.is_set,
            )
            cleanup_result = ArtifactSurveyExportResult(
                operation_id=work_item.id,
                staging_directory=staging,
            )
            staging_identity = self._capture_staging_identity(staging)
            prepared = prepare_staged_survey_publication(
                staging,
                work_item.destination,
                document=work_item.captured_session.document,
            )
            result = ArtifactSurveyExportResult(
                operation_id=work_item.id,
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
                        ArtifactSurveyExportResult(
                            operation_id=work_item.id,
                            staging_directory=cleanup_result.staging_directory,
                            prepared_publication=prepared,
                        ),
                    )
                except Exception as discard_exc:  # pragma: no cover - rare I/O
                    cleanup_error = discard_exc
                    _LOGGER.exception(
                        "Failed to discard staged complete survey after worker error"
                    )
            with self._lock:
                if self._active.get(work_item.id) is runtime:
                    runtime.executing = False
                    cancelled = runtime.cancellation.is_set() or isinstance(
                        exc,
                        ExportCancelledError,
                    )
                    terminal = (
                        ArtifactExportState.CANCELLED
                        if cancelled and cleanup_error is None
                        else ArtifactExportState.FAILED
                    )
                    self._finish_locked(runtime, terminal, cleanup_error or exc)
            if cleanup_error is not None:
                raise ArtifactExportError(
                    "survey export failed and its staging tree could not be discarded"
                ) from cleanup_error
            if runtime.cancellation.is_set() or isinstance(exc, ExportCancelledError):
                raise ExportCancelledError(
                    "survey export was cancelled before visible publication"
                ) from exc
            if isinstance(exc, ArtifactExportError):
                raise
            raise ArtifactExportError(str(exc)) from exc

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
                "cancelled survey staging tree could not be discarded"
            ) from exc
        with self._lock:
            if self._active.get(work_item.id) is runtime:
                self._finish_locked(
                    runtime,
                    ArtifactExportState.CANCELLED,
                    "survey export cancelled while staging",
                )
        raise ExportCancelledError(
            "survey export was cancelled before visible publication"
        )

    def _validate_exact_result_locked(
        self,
        runtime: _SurveyRuntime,
        result: ArtifactSurveyExportResult,
    ) -> PreparedSurveyPublication:
        if not isinstance(result, ArtifactSurveyExportResult):
            raise ArtifactExportError(
                "result must be an ArtifactSurveyExportResult"
            )
        if runtime.result is not result:
            raise ArtifactExportError(
                "publisher requires the exact survey result returned by execute()"
            )
        work_item = runtime.work_item
        prepared = result.prepared_publication
        if (
            result.operation_id != work_item.id
            or prepared is None
            or prepared.staging_directory != result.staging_directory
            or prepared.destination != work_item.destination
        ):
            raise ArtifactExportError(
                "survey result does not match its work item and prepared capability"
            )
        return prepared

    def _published_destination_is_owned(
        self,
        runtime: _SurveyRuntime,
        result: ArtifactSurveyExportResult,
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
            validate_survey_export_package(
                runtime.work_item.destination,
                document=runtime.work_item.captured_session.document,
            )
        except Exception:
            return False
        return True

    def _captured_authority_is_stale(
        self,
        work_item: ArtifactSurveyExportWorkItem,
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
        return any(
            current.document.record_index.get(record.id) != record
            for record in work_item.expected_records
        )

    @staticmethod
    def _publication(
        work_item: ArtifactSurveyExportWorkItem,
        *,
        durability_confirmed: bool = True,
        warning_message: str | None = None,
    ) -> ArtifactSurveyExportPublication:
        return ArtifactSurveyExportPublication(
            operation_id=work_item.id,
            destination=work_item.destination,
            document_sha256=work_item.captured_session.document.canonical_sha256,
            align_revision_id=work_item.projection_snapshot.align_revision_id,
            record_ids=work_item.selection.record_ids,
            durability_confirmed=durability_confirmed,
            warning_message=warning_message,
        )

    def publish_result(
        self,
        work_item: ArtifactSurveyExportWorkItem,
        result: ArtifactSurveyExportResult,
    ) -> ArtifactSurveyExportPublication:
        with self._lock:
            runtime = self._require_runtime_locked(
                work_item,
                states=frozenset({ArtifactExportState.STAGED}),
            )
            prepared = self._validate_exact_result_locked(runtime, result)
            runtime.state = ArtifactExportState.PUBLISHING

        authority_callback_entered = False
        try:
            def publish_exact_capability() -> Path:
                nonlocal authority_callback_entered
                authority_callback_entered = True
                return publish_prepared_survey_package(prepared)

            published = self._workbench.publish_records_effect_if_current(
                work_item.captured_session,
                work_item.projection_snapshot,
                expected_records=work_item.expected_records,
                expected_record_ids=work_item.selection.record_ids,
                publish=publish_exact_capability,
            )
            if Path(published) != work_item.destination:
                raise ArtifactExportError(
                    "survey core returned an unexpected publication destination"
                )
        except WorkflowBusyError:
            with self._lock:
                if self._active.get(work_item.id) is runtime:
                    runtime.state = ArtifactExportState.STAGED
            raise
        except StaleWorkflowOperationError as exc:
            try:
                self._discard_staging(work_item, result)
            except Exception as cleanup_exc:
                with self._lock:
                    if self._active.get(work_item.id) is runtime:
                        self._finish_locked(
                            runtime,
                            ArtifactExportState.FAILED,
                            cleanup_exc,
                        )
                raise ArtifactExportError(
                    "stale survey export was revoked but cleanup failed"
                ) from cleanup_exc
            with self._lock:
                if self._active.get(work_item.id) is runtime:
                    self._finish_locked(runtime, ArtifactExportState.STALE, exc)
            raise StaleExportOperationError(str(exc)) from exc
        except Exception as exc:
            committed_after_authority = bool(
                authority_callback_entered and getattr(exc, "committed", False)
            )
            if committed_after_authority and self._published_destination_is_owned(
                runtime,
                result,
            ):
                with self._lock:
                    if self._active.get(work_item.id) is runtime:
                        self._finish_locked(runtime, ArtifactExportState.COMPLETED)
                return self._publication(
                    work_item,
                    durability_confirmed=False,
                    warning_message=str(exc),
                )
            if not authority_callback_entered and getattr(exc, "committed", False):
                error = ArtifactExportError(
                    "survey staging became visible before final authority publication"
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
                    self._discard_staging(work_item, result)
                except Exception as cleanup_exc:
                    with self._lock:
                        if self._active.get(work_item.id) is runtime:
                            self._finish_locked(
                                runtime,
                                ArtifactExportState.FAILED,
                                cleanup_exc,
                            )
                    raise ArtifactExportError(
                        "stale survey export was revoked but cleanup failed"
                    ) from cleanup_exc
                with self._lock:
                    if self._active.get(work_item.id) is runtime:
                        self._finish_locked(runtime, ArtifactExportState.STALE, exc)
                raise StaleExportOperationError(str(exc)) from exc
            try:
                self._discard_staging(work_item, result)
            except Exception as cleanup_exc:
                with self._lock:
                    if self._active.get(work_item.id) is runtime:
                        self._finish_locked(
                            runtime,
                            ArtifactExportState.FAILED,
                            cleanup_exc,
                        )
                raise ArtifactExportError(
                    "survey publication failed and staging cleanup also failed"
                ) from cleanup_exc
            with self._lock:
                if self._active.get(work_item.id) is runtime:
                    self._finish_locked(runtime, ArtifactExportState.FAILED, exc)
            if isinstance(exc, ArtifactExportError):
                raise
            if isinstance(exc, ArtifactSurveyExportError):
                raise ArtifactExportError(str(exc)) from exc
            raise ArtifactExportError(str(exc)) from exc

        if not self._published_destination_is_owned(runtime, result):
            error = ArtifactExportError(
                "survey publisher returned without the owned package at destination"
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
        work_item: ArtifactSurveyExportWorkItem,
        result: ArtifactSurveyExportResult,
        *,
        reason: str = "discarded",
    ) -> ArtifactSurveyExportSummary:
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
                f"staged survey could not be discarded safely: {exc}"
            ) from exc
        with self._lock:
            return self._finish_locked(
                runtime,
                ArtifactExportState.CANCELLED,
                resolved_reason,
            )

    def cancel(
        self,
        work_item: ArtifactSurveyExportWorkItem,
        *,
        reason: str = "cancelled",
    ) -> ArtifactSurveyExportSummary:
        resolved_reason = _required_text(reason, field_name="cancel reason")
        staged_result: ArtifactSurveyExportResult | None = None
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
                    f"cancelled survey could not discard its staging tree: {exc}"
                ) from exc
        with self._lock:
            return self._finish_locked(
                runtime,
                ArtifactExportState.CANCELLED,
                resolved_reason,
            )


__all__ = [
    "ArtifactSurveyExportController",
    "ArtifactSurveyExportPublication",
    "ArtifactSurveyExportResult",
    "ArtifactSurveyExportSummary",
    "ArtifactSurveyExportWorkItem",
]
