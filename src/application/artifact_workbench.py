"""Qt-free authority and transaction coordinator for one native artifact.

The scientific core validates source bytes, geometry, units, Align revisions,
and derived records.  This module adds the application concerns that do not
belong in that core: one pending Open request, monotonic authority epochs,
compare-and-swap publication, explicit Align readiness, and observer-safe
rollback.  It intentionally imports neither Qt nor OpenGL.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import logging
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import TypeAlias, TypeVar
import uuid

import numpy as np

from src.core.artifact_document import (
    ArtifactDocument,
    DerivedRecord,
    GeometryRevision,
    Handedness,
    MetadataConfirmationStatus,
    SourceAsset,
    SourceMetadataRevision,
    RecordFreshness,
    RecordLifecycleStatus,
    source_to_canonical_mm_matrix,
)
from src.core.artifact_scene_adapter import (
    ArtifactProjectionSnapshot,
    ArtifactSceneProjection,
)
from src.core.artifact_session import ArtifactSession, ArtifactSessionError
from src.core.mesh_import_recipe import (
    MeshImportRecipeError,
    current_mesh_import_recipe,
    mesh_import_receipt_matches_base,
    validate_mesh_import_recipe,
)
from src.core.mesh_loader import MeshData, MeshLoader


_LOGGER = logging.getLogger(__name__)
_INITIAL_ALIGN_RECIPE = "initial_identity"
_AXIS_KEYS = ("source_x", "source_y", "source_z")
_KEEP_PROJECT_PATH = object()
_T = TypeVar("_T")


class ArtifactWorkbenchError(ValueError):
    """Base class for typed application-workflow failures."""


class WorkflowBusyError(ArtifactWorkbenchError):
    """A command conflicts with a pending Open or projection publication."""


class StaleWorkflowOperationError(ArtifactWorkbenchError):
    """A result was produced for an authority epoch that is no longer active."""


class WorkflowPhase(str, Enum):
    INITIAL = "initial"
    IMPORTING = "importing"
    READY = "ready"
    ERROR = "error"


class WorkflowStage(str, Enum):
    EMPTY = "empty"
    ALIGN_REQUIRED = "align_required"
    MEASUREMENT_READY = "measurement_ready"


class SaveDurability(str, Enum):
    """Whether a completed project write has proven directory durability."""

    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"


class WorkflowSaveStatus(str, Enum):
    """Derived save state for the exact immutable document authority."""

    EMPTY = "empty"
    UNSAVED = "unsaved"
    SAVED = "saved"
    DURABILITY_UNCERTAIN = "durability_uncertain"


class WorkflowTransitionKind(str, Enum):
    NEW_SOURCE = "new_source"
    REOPEN_PROJECT = "reopen_project"
    ALIGN_COMMIT = "align_commit"
    ALIGN_ACTIVATE_PARENT = "align_activate_parent"
    SESSION_UPDATE = "session_update"


class _LoadKind(str, Enum):
    NEW_SOURCE = "new_source"
    REOPEN_PROJECT = "reopen_project"


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactWorkbenchError(f"{field_name} must be a non-empty string")
    return value.strip()


def _normalized_path(value: object, *, field_name: str) -> str:
    raw = _required_text(value, field_name=field_name)
    return str(Path(raw).expanduser().resolve(strict=False))


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4()}"


def _authoritative_source_format(value: str, *, field_name: str) -> str:
    source_format = _required_text(value, field_name=field_name).lower()
    if f".{source_format}" not in MeshLoader.SUPPORTED_FORMATS:
        raise ArtifactWorkbenchError(
            f"unsupported authoritative source format: {source_format!r}"
        )
    return source_format


def _document_source_context(
    document: ArtifactDocument,
) -> tuple[SourceAsset, GeometryRevision, SourceMetadataRevision]:
    metadata_id = document.active_source_metadata_revision_id
    if metadata_id is None:
        raise ArtifactWorkbenchError("ArtifactDocument has no active metadata revision")
    try:
        metadata = document.source_metadata_revision_index[metadata_id]
        geometry = document.geometry_revision_index[metadata.geometry_revision_id]
    except KeyError as exc:
        raise ArtifactWorkbenchError(
            "ArtifactDocument active source context is incomplete"
        ) from exc
    if len(geometry.source_asset_ids) != 1:
        raise ArtifactWorkbenchError("native workbench supports exactly one source asset")
    try:
        source_asset = document.source_asset_index[geometry.source_asset_ids[0]]
    except KeyError as exc:
        raise ArtifactWorkbenchError("ArtifactDocument source asset is missing") from exc
    return source_asset, geometry, metadata


def _active_align_is_explicit(session: ArtifactSession) -> bool:
    active_id = session.document.active_align_revision_id
    if active_id is None:
        return False
    align = session.document.align_revision_index.get(active_id)
    if align is None:
        return False
    return str(align.recipe.get("kind", "") or "") != _INITIAL_ALIGN_RECIPE


@dataclass(frozen=True, slots=True)
class ConfirmedSourceMetadata:
    """User-confirmed unit and signed-axis mapping, independent of widgets."""

    unit: str
    source_x: str
    source_y: str
    source_z: str
    handedness: Handedness | str

    def __post_init__(self) -> None:
        unit = _required_text(self.unit, field_name="source metadata unit").lower()
        axes = self.axes
        try:
            matrix = source_to_canonical_mm_matrix(unit, axes)
            handedness = Handedness(self.handedness)
        except (TypeError, ValueError) as exc:
            raise ArtifactWorkbenchError(str(exc)) from exc
        if handedness is Handedness.UNKNOWN:
            raise ArtifactWorkbenchError("confirmed source metadata needs handedness")
        orientation = float(np.linalg.det(matrix[:3, :3]))
        expected = Handedness.RIGHT if orientation > 0.0 else Handedness.LEFT
        if handedness is not expected:
            raise ArtifactWorkbenchError(
                "source metadata handedness does not match its signed-axis mapping"
            )
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "handedness", handedness)

    @property
    def axes(self) -> dict[str, str]:
        return {
            "source_x": _required_text(self.source_x, field_name="axes.source_x"),
            "source_y": _required_text(self.source_y, field_name="axes.source_y"),
            "source_z": _required_text(self.source_z, field_name="axes.source_z"),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ConfirmedSourceMetadata":
        if not isinstance(value, Mapping):
            raise ArtifactWorkbenchError("source metadata must be an object")
        if value.get("confirmation_status") != MetadataConfirmationStatus.CONFIRMED.value:
            raise ArtifactWorkbenchError("source metadata must be explicitly confirmed")
        axes = value.get("axes")
        if not isinstance(axes, Mapping) or set(axes) != set(_AXIS_KEYS):
            raise ArtifactWorkbenchError(
                "source metadata axes must contain source_x, source_y and source_z"
            )
        return cls(
            unit=str(value.get("unit", "")),
            source_x=str(axes["source_x"]),
            source_y=str(axes["source_y"]),
            source_z=str(axes["source_z"]),
            handedness=str(value.get("handedness", "unknown")),
        )


@dataclass(frozen=True, slots=True)
class WorkflowFailure:
    operation_id: str
    operation: str
    error_type: str
    message: str
    fatal: bool = False


@dataclass(frozen=True, slots=True)
class SavedProjectCheckpoint:
    """Exact document/path pair proven by one completed project save.

    A project path alone is not evidence that the current immutable document
    was saved.  The canonical document hash closes that gap, while durability
    records whether the parent-directory sync completed successfully.
    """

    document_sha256: str
    project_path: str
    durability: SaveDurability | str = SaveDurability.CONFIRMED

    def __post_init__(self) -> None:
        document_sha256 = _required_text(
            self.document_sha256,
            field_name="saved checkpoint document_sha256",
        ).lower()
        if len(document_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in document_sha256
        ):
            raise ArtifactWorkbenchError(
                "saved checkpoint document_sha256 must be 64 lowercase hexadecimal characters"
            )
        try:
            durability = SaveDurability(self.durability)
        except (TypeError, ValueError) as exc:
            raise ArtifactWorkbenchError(
                "saved checkpoint durability must be confirmed or uncertain"
            ) from exc
        object.__setattr__(self, "document_sha256", document_sha256)
        object.__setattr__(
            self,
            "project_path",
            _normalized_path(self.project_path, field_name="saved checkpoint project_path"),
        )
        object.__setattr__(self, "durability", durability)

    @property
    def durability_confirmed(self) -> bool:
        return self.durability is SaveDurability.CONFIRMED


@dataclass(frozen=True, slots=True)
class ArtifactLoadTicket:
    id: str
    base_authority_epoch: int
    kind: _LoadKind
    source_path: str
    source_format: str
    import_recipe: Mapping[str, object]
    source_unit: str
    project_path: str | None
    metadata: ConfirmedSourceMetadata | None
    document: ArtifactDocument | None
    software_version: str | None
    operator: str | None
    created_at: str | None = None
    document_id: str | None = None
    metadata_revision_id: str | None = None
    align_revision_id: str | None = None
    capture_dependencies: bool = False

    def __post_init__(self) -> None:
        # A ticket crosses the GUI/background-worker boundary.  Preserve the
        # exact validated parser receipt and prevent either side from mutating
        # its execution contract while the source bytes are being parsed.
        object.__setattr__(
            self,
            "import_recipe",
            MappingProxyType(dict(self.import_recipe)),
        )


@dataclass(frozen=True, slots=True)
class WorkflowSnapshot:
    state_version: int
    authority_epoch: int
    session: ArtifactSession | None
    project_path: str | None
    pending_load: ArtifactLoadTicket | None
    failure: WorkflowFailure | None
    save_checkpoint: SavedProjectCheckpoint | None = None
    tentative: bool = False
    faulted: bool = False

    @property
    def phase(self) -> WorkflowPhase:
        if self.pending_load is not None:
            return WorkflowPhase.IMPORTING
        if self.faulted or (self.failure is not None and self.failure.fatal):
            return WorkflowPhase.ERROR
        if self.session is not None:
            return WorkflowPhase.READY
        if self.failure is not None:
            return WorkflowPhase.ERROR
        return WorkflowPhase.INITIAL

    @property
    def stage(self) -> WorkflowStage:
        if self.session is None:
            return WorkflowStage.EMPTY
        if _active_align_is_explicit(self.session):
            return WorkflowStage.MEASUREMENT_READY
        return WorkflowStage.ALIGN_REQUIRED

    @property
    def can_save(self) -> bool:
        return (
            self.session is not None
            and self.pending_load is None
            and not self.tentative
            and not self.faulted
            and not bool(self.failure is not None and self.failure.fatal)
        )

    @property
    def can_measure(self) -> bool:
        return (
            self.session is not None
            and self.pending_load is None
            and not self.tentative
            and not self.faulted
            and not bool(self.failure is not None and self.failure.fatal)
            and self.stage is WorkflowStage.MEASUREMENT_READY
        )

    @property
    def document_sha256(self) -> str | None:
        return self.session.document.canonical_sha256 if self.session is not None else None

    @property
    def save_checkpoint_current(self) -> bool:
        checkpoint = self.save_checkpoint
        return bool(
            self.session is not None
            and self.project_path is not None
            and checkpoint is not None
            and checkpoint.document_sha256 == self.document_sha256
            and checkpoint.project_path == self.project_path
            and checkpoint.durability is SaveDurability.CONFIRMED
        )

    @property
    def save_status(self) -> WorkflowSaveStatus:
        if self.session is None:
            return WorkflowSaveStatus.EMPTY
        checkpoint = self.save_checkpoint
        checkpoint_matches = bool(
            self.project_path is not None
            and checkpoint is not None
            and checkpoint.document_sha256 == self.document_sha256
            and checkpoint.project_path == self.project_path
        )
        if self.faulted:
            return WorkflowSaveStatus.UNSAVED
        if not checkpoint_matches:
            return WorkflowSaveStatus.UNSAVED
        if checkpoint is not None and checkpoint.durability is SaveDurability.UNCERTAIN:
            return WorkflowSaveStatus.DURABILITY_UNCERTAIN
        return WorkflowSaveStatus.SAVED

    @property
    def has_unsaved_changes(self) -> bool:
        return self.save_status not in {
            WorkflowSaveStatus.EMPTY,
            WorkflowSaveStatus.SAVED,
        }


@dataclass(frozen=True, slots=True)
class ProjectionTransition:
    id: str
    kind: WorkflowTransitionKind
    base_state_version: int
    base_authority_epoch: int
    expected_session: ArtifactSession | None
    load_ticket_id: str | None
    candidate_session: ArtifactSession
    projection: ArtifactSceneProjection
    project_path: str | None


@dataclass(frozen=True, slots=True)
class RecordBindingTransition:
    """DerivedRecord append that reuses an unchanged live mesh projection."""

    id: str
    base_state_version: int
    base_authority_epoch: int
    expected_session: ArtifactSession
    candidate_session: ArtifactSession
    expected_snapshot: ArtifactProjectionSnapshot
    candidate_snapshot: ArtifactProjectionSnapshot
    expected_new_record_ids: tuple[str, ...]
    project_path: str | None


@dataclass(frozen=True, slots=True)
class ProjectionActivation:
    id: str
    transition_id: str
    previous: WorkflowSnapshot
    current: WorkflowSnapshot


Observer: TypeAlias = Callable[[WorkflowSnapshot], None]
ObserverTarget: TypeAlias = tuple[str, Observer]
Notification: TypeAlias = tuple[WorkflowSnapshot, tuple[ObserverTarget, ...]]


class ArtifactWorkbench:
    """Own the native application authority while remaining GUI-independent."""

    def __init__(
        self,
        *,
        session: ArtifactSession | None = None,
        project_path: str | None = None,
        id_factory: Callable[[str], str] = _new_id,
    ) -> None:
        if session is not None:
            session.projection_snapshot()
        normalized_project = (
            _normalized_path(project_path, field_name="project_path")
            if project_path is not None
            else None
        )
        initial_checkpoint = (
            SavedProjectCheckpoint(
                document_sha256=session.document.canonical_sha256,
                project_path=normalized_project,
            )
            if session is not None and normalized_project is not None
            else None
        )
        self._lock = RLock()
        self._id_factory = id_factory
        self._state = WorkflowSnapshot(
            state_version=0,
            authority_epoch=0,
            session=session,
            project_path=normalized_project,
            pending_load=None,
            failure=None,
            save_checkpoint=initial_checkpoint,
        )
        self._tentative_activation_id: str | None = None
        # Open transitions cross an asynchronous worker/GUI boundary.  Their
        # frozen dataclass fields remain inspectable evidence, but field equality
        # is not an authority capability because ``dataclasses.replace`` can
        # reproduce and alter them.  Only the exact object most recently issued
        # by ``prepare_loaded_source`` may consume the one live Open ticket.
        self._prepared_open_transition: ProjectionTransition | None = None
        self._external_effect_publication_lease: object | None = None
        self._observers: dict[str, Observer] = {}
        self._observer_versions: dict[str, int] = {}
        self._notification_queue: deque[Notification] = deque()
        self._notifying = False

    @property
    def snapshot(self) -> WorkflowSnapshot:
        with self._lock:
            return self._state

    @property
    def session(self) -> ArtifactSession | None:
        return self.snapshot.session

    def subscribe(self, observer: Observer, *, replay: bool = True) -> Callable[[], None]:
        if not callable(observer):
            raise ArtifactWorkbenchError("observer must be callable")
        observer_id = self._id_factory("observer")
        should_drain = False
        with self._lock:
            self._observers[observer_id] = observer
            self._observer_versions[observer_id] = -1
            snapshot = self._state
            if replay:
                should_drain = self._queue_notification_locked(
                    snapshot,
                    ((observer_id, observer),),
                )
        if should_drain:
            self._drain_notifications()

        def unsubscribe() -> None:
            with self._lock:
                self._observers.pop(observer_id, None)
                self._observer_versions.pop(observer_id, None)

        return unsubscribe

    @staticmethod
    def _notify_one(observer: Observer, snapshot: WorkflowSnapshot) -> None:
        try:
            observer(snapshot)
        except Exception:
            _LOGGER.exception("ArtifactWorkbench observer failed")

    def _queue_notification_locked(
        self,
        snapshot: WorkflowSnapshot,
        targets: tuple[ObserverTarget, ...],
    ) -> bool:
        self._notification_queue.append((snapshot, targets))
        if not self._notifying:
            self._notifying = True
            return True
        return False

    def _drain_notifications(self) -> None:
        while True:
            with self._lock:
                if not self._notification_queue:
                    self._notifying = False
                    return
                snapshot, targets = self._notification_queue.popleft()
            for observer_id, observer in targets:
                with self._lock:
                    if self._observers.get(observer_id) is not observer:
                        continue
                    last_version = self._observer_versions.get(observer_id, -1)
                    if snapshot.state_version <= last_version:
                        continue
                    self._observer_versions[observer_id] = snapshot.state_version
                self._notify_one(observer, snapshot)

    def _notify(self, snapshot: WorkflowSnapshot) -> None:
        with self._lock:
            targets = tuple(self._observers.items())
            should_drain = self._queue_notification_locked(snapshot, targets)
        if should_drain:
            self._drain_notifications()

    def _require_no_external_effect_publication(self) -> None:
        if self._external_effect_publication_lease is not None:
            raise WorkflowBusyError("an external effect publication is in progress")

    def _require_open_slot(self) -> WorkflowSnapshot:
        self._require_no_external_effect_publication()
        if self._tentative_activation_id is not None:
            raise WorkflowBusyError("a projection publication is in progress")
        if self._state.pending_load is not None:
            raise WorkflowBusyError("an artifact Open request is already pending")
        return self._state

    def _require_command_slot(self) -> WorkflowSnapshot:
        state = self._require_open_slot()
        if state.faulted:
            raise ArtifactWorkbenchError(
                "artifact authority is faulted; only a verified Open may recover it"
            )
        return state

    def require_stable_session(
        self,
        session: ArtifactSession,
        *,
        measurement: bool = False,
    ) -> ArtifactSession:
        """Require one finalized authority before save, measurement or export."""

        if not isinstance(session, ArtifactSession):
            raise ArtifactWorkbenchError("session must be an ArtifactSession")
        with self._lock:
            state = self._state
            if self._tentative_activation_id is not None or state.tentative:
                raise WorkflowBusyError("a projection publication is in progress")
            if state.pending_load is not None:
                raise WorkflowBusyError("an artifact Open request is pending")
            if state.session is not session:
                raise StaleWorkflowOperationError(
                    "requested session is not the active artifact authority"
                )
            if state.faulted or (state.failure is not None and state.failure.fatal):
                raise ArtifactWorkbenchError(
                    "artifact authority is faulted; reopen a verified source or project"
                )
            if measurement and not state.can_measure:
                raise ArtifactWorkbenchError(
                    "explicit Align confirmation is required before measurement or export"
                )
            if not measurement and not state.can_save:
                raise ArtifactWorkbenchError("artifact authority is not stable for save")
            return session

    def adopt_saved_project_path(
        self,
        captured_session: ArtifactSession,
        project_path: str,
        *,
        expected_state_version: int,
        expected_authority_epoch: int,
        durability_confirmed: bool = True,
    ) -> WorkflowSnapshot:
        """Adopt a completed Save destination only for its captured authority.

        Project serialization happens before this fast compare-and-swap.  A
        successful checkpoint change advances the state version, even when a
        same-path save establishes new document or durability evidence.  A
        transition prepared against the former checkpoint therefore cannot
        silently restore it.  The authority epoch stays unchanged because the
        immutable document and render projection do not change.
        """

        if not isinstance(captured_session, ArtifactSession):
            raise ArtifactWorkbenchError(
                "captured_session must be an ArtifactSession"
            )
        if type(expected_state_version) is not int or expected_state_version < 0:
            raise ArtifactWorkbenchError(
                "expected_state_version must be a non-negative integer"
            )
        if type(expected_authority_epoch) is not int or expected_authority_epoch < 0:
            raise ArtifactWorkbenchError(
                "expected_authority_epoch must be a non-negative integer"
            )
        if type(durability_confirmed) is not bool:
            raise ArtifactWorkbenchError("durability_confirmed must be a boolean")
        normalized_project = _normalized_path(
            project_path,
            field_name="project_path",
        )
        checkpoint = SavedProjectCheckpoint(
            document_sha256=captured_session.document.canonical_sha256,
            project_path=normalized_project,
            durability=(
                SaveDurability.CONFIRMED
                if durability_confirmed
                else SaveDurability.UNCERTAIN
            ),
        )
        with self._lock:
            state = self._require_command_slot()
            if (
                state.session is not captured_session
                or state.state_version != expected_state_version
                or state.authority_epoch != expected_authority_epoch
            ):
                raise StaleWorkflowOperationError(
                    "saved project path belongs to stale artifact authority"
                )
            if not state.can_save:
                raise ArtifactWorkbenchError(
                    "artifact authority is not stable for saved-path adoption"
                )
            if (
                state.project_path == normalized_project
                and state.save_checkpoint == checkpoint
            ):
                return state
            self._state = replace(
                state,
                state_version=state.state_version + 1,
                project_path=normalized_project,
                save_checkpoint=checkpoint,
            )
            changed = self._state
        self._notify(changed)
        return changed

    @staticmethod
    def _require_export_document_ancestor(
        captured: ArtifactSession,
        current: ArtifactSession,
    ) -> None:
        if current.source_mesh is not captured.source_mesh:
            raise StaleWorkflowOperationError(
                "export source session changed after staging began"
            )
        old = captured.document
        new = current.document
        if (
            new.document_id != old.document_id
            or new.schema_version != old.schema_version
            or new.software_version != old.software_version
            or new.extensions != old.extensions
        ):
            raise StaleWorkflowOperationError(
                "export document identity changed after staging began"
            )
        for old_index, new_index, label in (
            (old.source_asset_index, new.source_asset_index, "source asset"),
            (old.geometry_revision_index, new.geometry_revision_index, "geometry revision"),
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
                    raise StaleWorkflowOperationError(
                        f"captured {label} {item_id!r} was removed or rewritten"
                    )

    def publish_record_effect_if_current(
        self,
        captured_session: ArtifactSession,
        captured_snapshot: ArtifactProjectionSnapshot,
        *,
        record_id: str,
        expected_record: DerivedRecord,
        publish: Callable[[], _T],
    ) -> _T:
        """Linearize one fast external publish against current record authority.

        Expensive package construction and validation must happen before this
        call.  The callback runs under the Workbench lock and should perform
        only the final same-filesystem no-replace rename.
        """

        return self.publish_records_effect_if_current(
            captured_session,
            captured_snapshot,
            expected_records=(expected_record,),
            publish=publish,
            expected_record_ids=(record_id,),
        )

    def publish_records_effect_if_current(
        self,
        captured_session: ArtifactSession,
        captured_snapshot: ArtifactProjectionSnapshot,
        *,
        expected_records: tuple[DerivedRecord, ...],
        publish: Callable[[], _T],
        expected_record_ids: tuple[str, ...] | None = None,
    ) -> _T:
        """Linearize one external publish against several immutable records.

        Complete-survey publication uses this boundary after all fifteen child
        packages have already been built and validated.  Append-only records
        may appear meanwhile, but every captured record and the render
        projection must remain exactly current until the parent-directory
        rename completes.
        """

        if not isinstance(captured_session, ArtifactSession):
            raise ArtifactWorkbenchError("captured_session must be an ArtifactSession")
        if not isinstance(captured_snapshot, ArtifactProjectionSnapshot):
            raise ArtifactWorkbenchError(
                "captured_snapshot must be an ArtifactProjectionSnapshot"
            )
        if captured_session.projection_snapshot() != captured_snapshot:
            raise StaleWorkflowOperationError(
                "captured export session no longer matches its snapshot"
            )
        records = tuple(expected_records)
        if not records or any(not isinstance(record, DerivedRecord) for record in records):
            raise ArtifactWorkbenchError(
                "expected_records must contain DerivedRecord values"
            )
        record_keys = (
            tuple(record.id for record in records)
            if expected_record_ids is None
            else tuple(
                _required_text(record_id, field_name="record ID")
                for record_id in expected_record_ids
            )
        )
        if len(record_keys) != len(records) or len(set(record_keys)) != len(record_keys):
            raise ArtifactWorkbenchError(
                "expected export record IDs must be unique and match the records"
            )
        for record_key, expected_record in zip(record_keys, records, strict=True):
            if (
                expected_record.id != record_key
                or captured_session.document.record_index.get(record_key)
                != expected_record
            ):
                raise ArtifactWorkbenchError(
                    "expected export record does not match the captured document"
                )
        if not callable(publish):
            raise ArtifactWorkbenchError("export publisher must be callable")

        for _attempt in range(8):
            with self._lock:
                observed = self._require_command_slot()
                current = observed.session
                if current is None:
                    raise ArtifactWorkbenchError(
                        "no active ArtifactSession for export publish"
                    )
                if not observed.can_measure:
                    raise ArtifactWorkbenchError(
                        "explicit Align confirmation is required before export"
                    )

            self._require_export_document_ancestor(captured_session, current)
            current_snapshot = current.projection_snapshot()
            if not captured_snapshot.has_same_render_projection(current_snapshot):
                raise StaleWorkflowOperationError(
                    "export projection authority changed after staging began"
                )
            for record_key, expected_record in zip(
                record_keys,
                records,
                strict=True,
            ):
                current_record = current.document.record_index.get(record_key)
                if current_record is None or current_record != expected_record:
                    raise StaleWorkflowOperationError(
                        "export record changed after staging began"
                    )
                if (
                    current_record.lifecycle_status is not RecordLifecycleStatus.READY
                    or current.document.record_freshness(record_key)
                    is not RecordFreshness.FRESH
                ):
                    raise StaleWorkflowOperationError(
                        "export record is no longer READY + FRESH"
                    )

            with self._lock:
                if self._state is not observed:
                    continue
                self._require_command_slot()
                lease = object()
                self._external_effect_publication_lease = lease
                try:
                    try:
                        result = publish()
                    except BaseException as callback_error:
                        try:
                            self._require_external_effect_postcondition(
                                lease=lease,
                                expected_state=observed,
                            )
                        except ArtifactWorkbenchError as authority_error:
                            raise authority_error from callback_error
                        raise
                    self._require_external_effect_postcondition(
                        lease=lease,
                        expected_state=observed,
                    )
                    return result
                finally:
                    self._external_effect_publication_lease = None

        raise StaleWorkflowOperationError(
            "artifact authority kept changing while export publication was prepared"
        )

    def _require_external_effect_postcondition(
        self,
        *,
        lease: object,
        expected_state: WorkflowSnapshot,
    ) -> None:
        if self._external_effect_publication_lease is not lease:
            raise ArtifactWorkbenchError(
                "external effect publication lost its authority lease"
            )
        if self._state is not expected_state:
            raise ArtifactWorkbenchError(
                "artifact authority changed during external effect publication"
            )

    def begin_new_import(
        self,
        source_path: str,
        metadata: ConfirmedSourceMetadata | Mapping[str, object],
        *,
        software_version: str,
        operator: str,
        request_id: str | None = None,
        created_at: str | None = None,
        document_id: str | None = None,
        metadata_revision_id: str | None = None,
        align_revision_id: str | None = None,
    ) -> ArtifactLoadTicket:
        confirmed = (
            metadata
            if isinstance(metadata, ConfirmedSourceMetadata)
            else ConfirmedSourceMetadata.from_mapping(metadata)
        )
        resolved = _normalized_path(source_path, field_name="source_path")
        source_format = _authoritative_source_format(
            Path(resolved).suffix.lower().removeprefix("."),
            field_name="source_format",
        )
        try:
            import_recipe = current_mesh_import_recipe(source_format)
        except MeshImportRecipeError as exc:
            raise ArtifactWorkbenchError(
                f"mesh parser runtime is not executable: {exc}"
            ) from exc
        with self._lock:
            state = self._require_open_slot()
            ticket = ArtifactLoadTicket(
                id=request_id or self._id_factory("open"),
                base_authority_epoch=state.authority_epoch,
                kind=_LoadKind.NEW_SOURCE,
                source_path=resolved,
                source_format=source_format,
                import_recipe=import_recipe,
                source_unit=confirmed.unit,
                project_path=None,
                metadata=confirmed,
                document=None,
                software_version=_required_text(
                    software_version,
                    field_name="software_version",
                ),
                operator=_required_text(operator, field_name="operator"),
                created_at=created_at,
                document_id=document_id,
                metadata_revision_id=metadata_revision_id,
                align_revision_id=align_revision_id,
                capture_dependencies=True,
            )
            self._prepared_open_transition = None
            self._state = WorkflowSnapshot(
                state_version=state.state_version + 1,
                authority_epoch=state.authority_epoch,
                session=state.session,
                project_path=state.project_path,
                pending_load=ticket,
                failure=None,
                save_checkpoint=state.save_checkpoint,
                faulted=state.faulted,
            )
            changed = self._state
        self._notify(changed)
        return ticket

    def begin_project_reopen(
        self,
        document: ArtifactDocument,
        *,
        project_path: str,
        resolved_source_path: str,
        request_id: str | None = None,
    ) -> ArtifactLoadTicket:
        if not isinstance(document, ArtifactDocument):
            raise ArtifactWorkbenchError("document must be an ArtifactDocument")
        _source_asset, geometry, metadata = _document_source_context(document)
        try:
            execution = validate_mesh_import_recipe(
                geometry.import_recipe,
                allow_legacy=True,
            )
        except MeshImportRecipeError as exc:
            raise ArtifactWorkbenchError(
                f"ArtifactDocument parser recipe is not executable: {exc}"
            ) from exc
        source_format = _authoritative_source_format(
            execution.source_format,
            field_name="ArtifactDocument parser format",
        )
        if metadata.confirmation_status is not MetadataConfirmationStatus.CONFIRMED:
            raise ArtifactWorkbenchError("ArtifactDocument metadata is not confirmed")
        with self._lock:
            state = self._require_open_slot()
            ticket = ArtifactLoadTicket(
                id=request_id or self._id_factory("open"),
                base_authority_epoch=state.authority_epoch,
                kind=_LoadKind.REOPEN_PROJECT,
                source_path=_normalized_path(
                    resolved_source_path,
                    field_name="resolved_source_path",
                ),
                source_format=source_format,
                import_recipe=geometry.import_recipe,
                source_unit=str(metadata.unit),
                project_path=_normalized_path(project_path, field_name="project_path"),
                metadata=None,
                document=document,
                software_version=None,
                operator=None,
                capture_dependencies=False,
            )
            self._prepared_open_transition = None
            self._state = WorkflowSnapshot(
                state_version=state.state_version + 1,
                authority_epoch=state.authority_epoch,
                session=state.session,
                project_path=state.project_path,
                pending_load=ticket,
                failure=None,
                save_checkpoint=state.save_checkpoint,
                faulted=state.faulted,
            )
            changed = self._state
        self._notify(changed)
        return ticket

    def _require_current_ticket(self, ticket: ArtifactLoadTicket) -> WorkflowSnapshot:
        if not isinstance(ticket, ArtifactLoadTicket):
            raise ArtifactWorkbenchError("ticket must be an ArtifactLoadTicket")
        self._require_no_external_effect_publication()
        if self._tentative_activation_id is not None:
            raise WorkflowBusyError("a projection publication is in progress")
        state = self._state
        pending = state.pending_load
        if pending is not ticket:
            raise StaleWorkflowOperationError(
                "artifact Open result belongs to a stale or cancelled request"
            )
        if state.authority_epoch != ticket.base_authority_epoch:
            raise StaleWorkflowOperationError(
                "artifact authority changed while the Open request was running"
            )
        return state

    def prepare_loaded_source(
        self,
        ticket: ArtifactLoadTicket,
        mesh: MeshData,
        *,
        resolved_source_path: str | None = None,
        transition_id: str | None = None,
    ) -> ProjectionTransition:
        with self._lock:
            state = self._require_current_ticket(ticket)
        source_path = _normalized_path(
            resolved_source_path or ticket.source_path,
            field_name="resolved_source_path",
        )
        if not isinstance(mesh.source_import_recipe, Mapping):
            raise ArtifactWorkbenchError("loaded source has no parser receipt")
        if ticket.kind is _LoadKind.NEW_SOURCE:
            receipt_matches = mesh_import_receipt_matches_base(
                ticket.import_recipe,
                mesh.source_import_recipe,
            )
        else:
            receipt_matches = dict(mesh.source_import_recipe) == dict(
                ticket.import_recipe
            )
        if not receipt_matches:
            raise ArtifactWorkbenchError(
                "loaded source parser receipt does not match its Open ticket"
            )
        try:
            if ticket.kind is _LoadKind.NEW_SOURCE:
                metadata = ticket.metadata
                if metadata is None or ticket.software_version is None or ticket.operator is None:
                    raise ArtifactWorkbenchError("new source ticket is incomplete")
                candidate = ArtifactSession.create_from_source(
                    mesh,
                    resolved_source_path=source_path,
                    unit=metadata.unit,
                    axes=metadata.axes,
                    handedness=metadata.handedness,
                    software_version=ticket.software_version,
                    operator=ticket.operator,
                    created_at=ticket.created_at,
                    document_id=ticket.document_id,
                    metadata_revision_id=ticket.metadata_revision_id,
                    align_revision_id=ticket.align_revision_id,
                )
                kind = WorkflowTransitionKind.NEW_SOURCE
            else:
                if ticket.document is None:
                    raise ArtifactWorkbenchError("project reopen ticket has no document")
                candidate = ArtifactSession.bind_loaded_document(
                    ticket.document,
                    mesh,
                    resolved_source_path=source_path,
                )
                kind = WorkflowTransitionKind.REOPEN_PROJECT
            projection = candidate.materialize()
        except (ArtifactSessionError, ValueError) as exc:
            raise ArtifactWorkbenchError(str(exc)) from exc
        with self._lock:
            current = self._require_current_ticket(ticket)
            if current.state_version != state.state_version:
                raise StaleWorkflowOperationError(
                    "artifact Open state changed while its candidate was prepared"
                )
            transition = ProjectionTransition(
                id=transition_id or self._id_factory("projection"),
                kind=kind,
                base_state_version=current.state_version,
                base_authority_epoch=current.authority_epoch,
                expected_session=current.session,
                load_ticket_id=ticket.id,
                candidate_session=candidate,
                projection=projection,
                project_path=ticket.project_path,
            )
            # A second successful preparation for the same ticket deliberately
            # supersedes the first.  Failed preparation leaves the last valid
            # capability usable because no authority state changed.
            self._prepared_open_transition = transition
            return transition

    def fail_load(
        self,
        ticket: ArtifactLoadTicket,
        error: BaseException | str,
    ) -> WorkflowSnapshot:
        with self._lock:
            state = self._require_current_ticket(ticket)
            failure = WorkflowFailure(
                operation_id=ticket.id,
                operation="open",
                error_type=(type(error).__name__ if isinstance(error, BaseException) else "Error"),
                message=str(error),
            )
            self._state = WorkflowSnapshot(
                state_version=state.state_version + 1,
                authority_epoch=state.authority_epoch,
                session=state.session,
                project_path=state.project_path,
                pending_load=None,
                failure=failure,
                save_checkpoint=state.save_checkpoint,
                faulted=state.faulted,
            )
            self._prepared_open_transition = None
            changed = self._state
        self._notify(changed)
        return changed

    def cancel_load(self, ticket: ArtifactLoadTicket) -> WorkflowSnapshot:
        with self._lock:
            state = self._require_current_ticket(ticket)
            self._state = WorkflowSnapshot(
                state_version=state.state_version + 1,
                authority_epoch=state.authority_epoch,
                session=state.session,
                project_path=state.project_path,
                pending_load=None,
                failure=None,
                save_checkpoint=state.save_checkpoint,
                faulted=state.faulted,
            )
            self._prepared_open_transition = None
            changed = self._state
        self._notify(changed)
        return changed

    @staticmethod
    def _identity_preview(
        translation_mm: Sequence[float],
        rotation_deg: Sequence[float],
        scale: float,
    ) -> bool:
        try:
            translation = np.asarray(translation_mm, dtype=np.float64).reshape(-1)
            rotation = np.asarray(rotation_deg, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return False
        return bool(
            translation.shape == (3,)
            and rotation.shape == (3,)
            and np.isfinite(translation).all()
            and np.isfinite(rotation).all()
            and np.allclose(translation, 0.0, rtol=0.0, atol=1e-12)
            and np.allclose(rotation, 0.0, rtol=0.0, atol=1e-12)
            and np.isclose(float(scale), 1.0, rtol=0.0, atol=1e-12)
        )

    def prepare_align_commit(
        self,
        *,
        translation_mm: Sequence[float],
        rotation_deg: Sequence[float],
        scale: float,
        pivot_mm: Sequence[float],
        operator: str,
        created_at: str | None = None,
        revision_id: str | None = None,
        transition_id: str | None = None,
    ) -> ProjectionTransition | None:
        with self._lock:
            state = self._require_command_slot()
            captured = state.session
            if captured is None:
                raise ArtifactWorkbenchError("no active ArtifactSession")
        if self._identity_preview(translation_mm, rotation_deg, scale) and _active_align_is_explicit(
            captured
        ):
            return None
        try:
            candidate = captured.commit_preview(
                translation_mm=tuple(float(value) for value in translation_mm),
                rotation_deg=tuple(float(value) for value in rotation_deg),
                scale=scale,
                pivot_mm=tuple(float(value) for value in pivot_mm),
                operator=operator,
                created_at=created_at,
                revision_id=revision_id,
            )
        except ArtifactSessionError as exc:
            raise ArtifactWorkbenchError(str(exc)) from exc
        return self.prepare_session_commit(
            captured,
            candidate,
            kind=WorkflowTransitionKind.ALIGN_COMMIT,
            transition_id=transition_id,
        )

    def prepare_activate_parent_align(
        self,
        *,
        transition_id: str | None = None,
    ) -> ProjectionTransition:
        with self._lock:
            state = self._require_command_slot()
            captured = state.session
            if captured is None:
                raise ArtifactWorkbenchError("no active ArtifactSession")
        try:
            candidate = captured.activate_parent_align()
        except ArtifactSessionError as exc:
            raise ArtifactWorkbenchError(str(exc)) from exc
        return self.prepare_session_commit(
            captured,
            candidate,
            kind=WorkflowTransitionKind.ALIGN_ACTIVATE_PARENT,
            transition_id=transition_id,
        )

    @staticmethod
    def _validate_session_extension(
        captured: ArtifactSession,
        candidate: ArtifactSession,
        *,
        kind: WorkflowTransitionKind,
        expected_new_record_ids: tuple[str, ...] | None,
    ) -> None:
        if candidate.source_mesh is not captured.source_mesh:
            raise ArtifactWorkbenchError("candidate changed the immutable source mesh")
        if candidate.verified_geometry != captured.verified_geometry:
            raise ArtifactWorkbenchError("candidate changed verified geometry identity")
        if candidate.resolved_source_path != captured.resolved_source_path:
            raise ArtifactWorkbenchError("candidate changed the resolved source path")
        old = captured.document
        new = candidate.document
        if (
            new.schema_version != old.schema_version
            or new.document_id != old.document_id
            or new.software_version != old.software_version
            or new.extensions != old.extensions
        ):
            raise ArtifactWorkbenchError("candidate changed the ArtifactDocument identity")
        if (
            new.source_assets != old.source_assets
            or new.geometry_revisions != old.geometry_revisions
            or new.source_metadata_revisions != old.source_metadata_revisions
        ):
            raise ArtifactWorkbenchError("candidate rewrote the immutable source context")
        new_align_index = new.align_revision_index
        for revision_id, revision in old.align_revision_index.items():
            if new_align_index.get(revision_id) != revision:
                raise ArtifactWorkbenchError("candidate removed or rewrote an Align revision")
        new_record_index = new.record_index
        for record_id, record in old.record_index.items():
            if new_record_index.get(record_id) != record:
                raise ArtifactWorkbenchError("candidate removed or rewrote a derived record")

        added_align_ids = tuple(
            sorted(set(new_align_index) - set(old.align_revision_index))
        )
        added_record_ids = tuple(
            sorted(set(new_record_index) - set(old.record_index))
        )
        expected_records = (
            None
            if expected_new_record_ids is None
            else tuple(
                sorted(
                    _required_text(record_id, field_name="expected record ID")
                    for record_id in expected_new_record_ids
                )
            )
        )
        if expected_records is not None and len(set(expected_records)) != len(
            expected_records
        ):
            raise ArtifactWorkbenchError("expected record IDs must be unique")

        if kind is WorkflowTransitionKind.SESSION_UPDATE:
            if expected_records is None:
                raise ArtifactWorkbenchError(
                    "session updates must declare their expected derived record IDs"
                )
            if (
                new.align_revisions != old.align_revisions
                or new.active_align_revision_id != old.active_align_revision_id
                or new.active_source_metadata_revision_id
                != old.active_source_metadata_revision_id
            ):
                raise ArtifactWorkbenchError(
                    "a derived-record update cannot change Align authority"
                )
            if added_record_ids != expected_records:
                raise ArtifactWorkbenchError(
                    "candidate did not add exactly the expected derived record IDs"
                )
            return

        if kind is WorkflowTransitionKind.ALIGN_COMMIT:
            if expected_records not in (None, ()) or new.records != old.records:
                raise ArtifactWorkbenchError("Align commit cannot change derived records")
            if len(added_align_ids) != 1:
                raise ArtifactWorkbenchError(
                    "Align commit must append exactly one Align revision"
                )
            revision = new_align_index[added_align_ids[0]]
            if (
                revision.parent_id != old.active_align_revision_id
                or new.active_align_revision_id != revision.id
                or new.active_source_metadata_revision_id
                != revision.source_metadata_revision_id
            ):
                raise ArtifactWorkbenchError(
                    "Align commit must activate one child of the captured Align"
                )
            return

        if kind is WorkflowTransitionKind.ALIGN_ACTIVATE_PARENT:
            active_id = old.active_align_revision_id
            parent_id = (
                old.align_revision_index[active_id].parent_id
                if active_id is not None
                else None
            )
            if parent_id is None:
                raise ArtifactWorkbenchError("captured Align has no parent to activate")
            parent = old.align_revision_index[parent_id]
            if (
                expected_records not in (None, ())
                or new.align_revisions != old.align_revisions
                or new.records != old.records
                or new.active_align_revision_id != parent_id
                or new.active_source_metadata_revision_id
                != parent.source_metadata_revision_id
            ):
                raise ArtifactWorkbenchError(
                    "parent activation may only activate the captured Align parent"
                )
            return

        raise ArtifactWorkbenchError(
            f"unsupported session transition kind: {kind.value!r}"
        )

    def prepare_session_commit(
        self,
        captured_session: ArtifactSession,
        candidate_session: ArtifactSession,
        *,
        kind: WorkflowTransitionKind = WorkflowTransitionKind.SESSION_UPDATE,
        expected_new_record_ids: tuple[str, ...] | None = None,
        transition_id: str | None = None,
        project_path: str | None | object = _KEEP_PROJECT_PATH,
    ) -> ProjectionTransition:
        if not isinstance(captured_session, ArtifactSession) or not isinstance(
            candidate_session,
            ArtifactSession,
        ):
            raise ArtifactWorkbenchError("session commit needs ArtifactSession values")
        with self._lock:
            state = self._require_command_slot()
            if state.session is not captured_session:
                raise StaleWorkflowOperationError(
                    "captured session is no longer the active authority"
                )
            base_state_version = state.state_version
            base_authority_epoch = state.authority_epoch
            candidate_project_path = (
                state.project_path
                if project_path is _KEEP_PROJECT_PATH
                else (
                    _normalized_path(project_path, field_name="project_path")
                    if project_path is not None
                    else None
                )
            )
        transition_kind = WorkflowTransitionKind(kind)
        self._validate_session_extension(
            captured_session,
            candidate_session,
            kind=transition_kind,
            expected_new_record_ids=expected_new_record_ids,
        )
        projection = candidate_session.materialize()
        with self._lock:
            current = self._require_command_slot()
            if (
                current.state_version != base_state_version
                or current.authority_epoch != base_authority_epoch
                or current.session is not captured_session
            ):
                raise StaleWorkflowOperationError(
                    "artifact authority changed while the projection was prepared"
                )
            return ProjectionTransition(
                id=transition_id or self._id_factory("projection"),
                kind=transition_kind,
                base_state_version=base_state_version,
                base_authority_epoch=base_authority_epoch,
                expected_session=captured_session,
                load_ticket_id=None,
                candidate_session=candidate_session,
                projection=projection,
                project_path=candidate_project_path,
            )

    def prepare_record_commit(
        self,
        captured_session: ArtifactSession,
        candidate_session: ArtifactSession,
        *,
        expected_new_record_ids: tuple[str, ...],
        transition_id: str | None = None,
        project_path: str | None | object = _KEEP_PROJECT_PATH,
    ) -> RecordBindingTransition:
        """Prepare an append-only document rebind without materializing a mesh."""

        if not isinstance(captured_session, ArtifactSession) or not isinstance(
            candidate_session,
            ArtifactSession,
        ):
            raise ArtifactWorkbenchError("record commit needs ArtifactSession values")
        expected_ids = tuple(
            sorted(
                _required_text(record_id, field_name="expected record ID")
                for record_id in expected_new_record_ids
            )
        )
        if not expected_ids:
            raise ArtifactWorkbenchError("record commit must append at least one record")
        if len(set(expected_ids)) != len(expected_ids):
            raise ArtifactWorkbenchError("expected record IDs must be unique")
        with self._lock:
            state = self._require_command_slot()
            if state.session is not captured_session:
                raise StaleWorkflowOperationError(
                    "captured session is no longer the active authority"
                )
            base_state_version = state.state_version
            base_authority_epoch = state.authority_epoch
            candidate_project_path = (
                state.project_path
                if project_path is _KEEP_PROJECT_PATH
                else (
                    _normalized_path(project_path, field_name="project_path")
                    if project_path is not None
                    else None
                )
            )
        self._validate_session_extension(
            captured_session,
            candidate_session,
            kind=WorkflowTransitionKind.SESSION_UPDATE,
            expected_new_record_ids=expected_ids,
        )
        expected_snapshot = captured_session.projection_snapshot()
        candidate_snapshot = candidate_session.projection_snapshot()
        if not expected_snapshot.has_same_render_projection(candidate_snapshot):
            raise ArtifactWorkbenchError(
                "a record commit changed the live render projection"
            )
        if expected_snapshot.document_sha256 == candidate_snapshot.document_sha256:
            raise ArtifactWorkbenchError(
                "record commit did not change the canonical document"
            )
        with self._lock:
            current = self._require_command_slot()
            if (
                current.state_version != base_state_version
                or current.authority_epoch != base_authority_epoch
                or current.session is not captured_session
            ):
                raise StaleWorkflowOperationError(
                    "artifact authority changed while the record binding was prepared"
                )
            return RecordBindingTransition(
                id=transition_id or self._id_factory("record-binding"),
                base_state_version=base_state_version,
                base_authority_epoch=base_authority_epoch,
                expected_session=captured_session,
                candidate_session=candidate_session,
                expected_snapshot=expected_snapshot,
                candidate_snapshot=candidate_snapshot,
                expected_new_record_ids=expected_ids,
                project_path=candidate_project_path,
            )

    def activate_projection(
        self,
        transition: ProjectionTransition,
        *,
        activation_id: str | None = None,
    ) -> ProjectionActivation:
        if not isinstance(transition, ProjectionTransition):
            raise ArtifactWorkbenchError("transition must be a ProjectionTransition")
        with self._lock:
            self._require_no_external_effect_publication()
            if self._tentative_activation_id is not None:
                raise WorkflowBusyError("another projection publication is in progress")
            state = self._state
            if (
                state.state_version != transition.base_state_version
                or state.authority_epoch != transition.base_authority_epoch
                or state.session is not transition.expected_session
            ):
                raise StaleWorkflowOperationError(
                    "projection transition was prepared for stale authority"
                )
            if transition.load_ticket_id is not None:
                if self._prepared_open_transition is not transition:
                    raise StaleWorkflowOperationError(
                        "Open projection transition is not the current prepared "
                        "capability"
                    )
                pending = state.pending_load
                if pending is None or pending.id != transition.load_ticket_id:
                    raise StaleWorkflowOperationError(
                        "projection transition belongs to a stale Open request"
                    )
                expected_kind = WorkflowTransitionKind(pending.kind.value)
                if transition.kind is not expected_kind:
                    raise ArtifactWorkbenchError(
                        "projection transition kind does not match its Open ticket"
                    )
            elif state.pending_load is not None:
                raise WorkflowBusyError("cannot publish an update during artifact Open")
            elif transition.kind in {
                WorkflowTransitionKind.NEW_SOURCE,
                WorkflowTransitionKind.REOPEN_PROJECT,
            }:
                raise ArtifactWorkbenchError(
                    "Open projection transition has no live load ticket"
                )
            if transition.kind is WorkflowTransitionKind.NEW_SOURCE:
                checkpoint = None
            elif transition.kind is WorkflowTransitionKind.REOPEN_PROJECT:
                if transition.project_path is None:
                    raise ArtifactWorkbenchError(
                        "project reopen transition has no project path"
                    )
                checkpoint = SavedProjectCheckpoint(
                    document_sha256=(
                        transition.candidate_session.document.canonical_sha256
                    ),
                    project_path=transition.project_path,
                )
            else:
                checkpoint = state.save_checkpoint
            activation = ProjectionActivation(
                id=activation_id or self._id_factory("activation"),
                transition_id=transition.id,
                previous=state,
                current=WorkflowSnapshot(
                    state_version=state.state_version + 1,
                    authority_epoch=state.authority_epoch + 1,
                    session=transition.candidate_session,
                    project_path=transition.project_path,
                    pending_load=None,
                    failure=None,
                    save_checkpoint=checkpoint,
                    tentative=True,
                    faulted=state.faulted,
                ),
            )
            if transition.load_ticket_id is not None:
                # Consume the controller-owned capability only after every
                # validation and activation construction has succeeded.  A
                # rejected forged copy therefore cannot invalidate the genuine
                # transition returned to the caller.
                self._prepared_open_transition = None
            self._state = activation.current
            self._tentative_activation_id = activation.id
            return activation

    def activate_record_binding(
        self,
        transition: RecordBindingTransition,
        *,
        activation_id: str | None = None,
    ) -> ProjectionActivation:
        if not isinstance(transition, RecordBindingTransition):
            raise ArtifactWorkbenchError(
                "transition must be a RecordBindingTransition"
            )
        if (
            transition.expected_session.projection_snapshot()
            != transition.expected_snapshot
            or transition.candidate_session.projection_snapshot()
            != transition.candidate_snapshot
        ):
            raise ArtifactWorkbenchError(
                "record binding snapshots do not match their immutable sessions"
            )
        if not transition.expected_snapshot.has_same_render_projection(
            transition.candidate_snapshot
        ):
            raise ArtifactWorkbenchError(
                "record binding cannot change the live render projection"
            )
        self._validate_session_extension(
            transition.expected_session,
            transition.candidate_session,
            kind=WorkflowTransitionKind.SESSION_UPDATE,
            expected_new_record_ids=transition.expected_new_record_ids,
        )
        with self._lock:
            self._require_no_external_effect_publication()
            if self._tentative_activation_id is not None:
                raise WorkflowBusyError("another authority publication is in progress")
            state = self._state
            if (
                state.state_version != transition.base_state_version
                or state.authority_epoch != transition.base_authority_epoch
                or state.session is not transition.expected_session
            ):
                raise StaleWorkflowOperationError(
                    "record binding was prepared for stale authority"
                )
            if state.pending_load is not None:
                raise WorkflowBusyError(
                    "cannot publish a record binding during artifact Open"
                )
            activation = ProjectionActivation(
                id=activation_id or self._id_factory("activation"),
                transition_id=transition.id,
                previous=state,
                current=WorkflowSnapshot(
                    state_version=state.state_version + 1,
                    authority_epoch=state.authority_epoch + 1,
                    session=transition.candidate_session,
                    project_path=transition.project_path,
                    pending_load=None,
                    failure=None,
                    save_checkpoint=state.save_checkpoint,
                    tentative=True,
                    faulted=state.faulted,
                ),
            )
            self._state = activation.current
            self._tentative_activation_id = activation.id
            return activation

    def finalize_projection(self, activation: ProjectionActivation) -> WorkflowSnapshot:
        with self._lock:
            self._require_no_external_effect_publication()
            if (
                self._tentative_activation_id != activation.id
                or self._state is not activation.current
            ):
                raise StaleWorkflowOperationError(
                    "projection activation is no longer tentative authority"
                )
            current = replace(
                activation.current,
                tentative=False,
                faulted=False,
            )
            self._state = current
            self._tentative_activation_id = None
        self._notify(current)
        return current

    def finalize_record_binding(
        self,
        activation: ProjectionActivation,
    ) -> WorkflowSnapshot:
        return self.finalize_projection(activation)

    def rollback_projection(
        self,
        activation: ProjectionActivation,
        error: BaseException | str,
    ) -> WorkflowSnapshot:
        return self._rollback_activation(
            activation,
            error,
            operation="projection_publish",
        )

    def rollback_record_binding(
        self,
        activation: ProjectionActivation,
        error: BaseException | str,
    ) -> WorkflowSnapshot:
        return self._rollback_activation(
            activation,
            error,
            operation="record_binding_publish",
        )

    def _rollback_activation(
        self,
        activation: ProjectionActivation,
        error: BaseException | str,
        *,
        operation: str,
    ) -> WorkflowSnapshot:
        with self._lock:
            self._require_no_external_effect_publication()
            if (
                self._tentative_activation_id != activation.id
                or self._state is not activation.current
            ):
                raise StaleWorkflowOperationError(
                    "authority activation cannot be rolled back after authority changed"
                )
            failure = WorkflowFailure(
                operation_id=activation.transition_id,
                operation=operation,
                error_type=(type(error).__name__ if isinstance(error, BaseException) else "Error"),
                message=str(error),
            )
            self._state = WorkflowSnapshot(
                state_version=activation.current.state_version + 1,
                authority_epoch=activation.current.authority_epoch + 1,
                session=activation.previous.session,
                project_path=activation.previous.project_path,
                pending_load=None,
                failure=failure,
                save_checkpoint=activation.previous.save_checkpoint,
                faulted=activation.previous.faulted,
            )
            self._tentative_activation_id = None
            changed = self._state
        self._notify(changed)
        return changed

    def enter_faulted_state(
        self,
        *,
        session: ArtifactSession | None,
        project_path: str | None,
        error: BaseException | str,
        operation_id: str | None = None,
    ) -> WorkflowSnapshot:
        """Emergency fail-closed recovery after an uncertain publication.

        Normal failures must use ``rollback_projection``. This method exists
        for the rarer case where rollback, scene restoration, or finalization
        itself fails and the application can no longer prove a coherent live
        authority. A verified Open may subsequently replace this faulted state.

        Unlike ordinary authority mutators, this emergency path is deliberately
        allowed while an external effect callback holds its publication lease.
        The publisher's postcondition then detects the authority change while
        this method leaves the Workbench failed closed.
        """

        if session is not None:
            if not isinstance(session, ArtifactSession):
                raise ArtifactWorkbenchError(
                    "fault recovery session must be an ArtifactSession or None"
                )
            session.projection_snapshot()
        normalized_project = (
            _normalized_path(project_path, field_name="project_path")
            if project_path is not None
            else None
        )
        with self._lock:
            state = self._state
            checkpoint = state.save_checkpoint
            if not (
                checkpoint is not None
                and session is not None
                and normalized_project is not None
                and checkpoint.document_sha256
                == session.document.canonical_sha256
                and checkpoint.project_path == normalized_project
            ):
                checkpoint = None
            failure = WorkflowFailure(
                operation_id=operation_id or self._id_factory("fault"),
                operation="authority_fault",
                error_type=(
                    type(error).__name__ if isinstance(error, BaseException) else "Error"
                ),
                message=str(error),
                fatal=True,
            )
            self._state = WorkflowSnapshot(
                state_version=state.state_version + 1,
                authority_epoch=state.authority_epoch + 1,
                session=session,
                project_path=normalized_project,
                pending_load=None,
                failure=failure,
                save_checkpoint=checkpoint,
                faulted=True,
            )
            self._prepared_open_transition = None
            self._tentative_activation_id = None
            changed = self._state
        self._notify(changed)
        return changed

    def synchronize_legacy_session(
        self,
        session: ArtifactSession | None,
        *,
        project_path: str | None,
    ) -> WorkflowSnapshot:
        """Temporary migration bridge for legacy call sites and GUI tests.

        Production native Open/Align uses ticketed transitions.  This bridge
        exists only while older record commands still assign MainWindow's
        compatibility session field directly; it rejects pending work.
        """

        if session is not None and not isinstance(session, ArtifactSession):
            raise ArtifactWorkbenchError("session must be an ArtifactSession or None")
        if session is not None:
            session.projection_snapshot()
        with self._lock:
            state = self._require_command_slot()
            self._prepared_open_transition = None
            normalized_project = (
                _normalized_path(project_path, field_name="project_path")
                if project_path is not None
                else None
            )
            if state.session is session and state.project_path == normalized_project:
                return state
            self._state = WorkflowSnapshot(
                state_version=state.state_version + 1,
                authority_epoch=state.authority_epoch + 1,
                session=session,
                project_path=normalized_project,
                pending_load=None,
                failure=None,
                save_checkpoint=None,
            )
            changed = self._state
        self._notify(changed)
        return changed


__all__ = [
    "ArtifactLoadTicket",
    "ArtifactWorkbench",
    "ArtifactWorkbenchError",
    "ConfirmedSourceMetadata",
    "ProjectionActivation",
    "ProjectionTransition",
    "RecordBindingTransition",
    "SaveDurability",
    "SavedProjectCheckpoint",
    "StaleWorkflowOperationError",
    "WorkflowBusyError",
    "WorkflowFailure",
    "WorkflowPhase",
    "WorkflowSaveStatus",
    "WorkflowSnapshot",
    "WorkflowStage",
    "WorkflowTransitionKind",
]
