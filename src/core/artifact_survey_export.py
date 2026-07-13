"""Atomic complete-survey packages built from authoritative workflow records.

``*.amr-survey`` is a relocatable directory that contains the existing nine
Cutline/Outline ``*.amr-vector`` packages and six Digital Rubbing
``*.amr-rubbing`` packages without changing either child format.  A canonical
manifest binds each canonical view to one exact record and to the hashes of
both child files.  The complete hidden tree is validated before one
same-filesystem, no-replace rename makes any part of the survey visible.

The package remains useful without the original scanner file or application
GUI.  Passing an :class:`ArtifactDocument` to the validator additionally binds
all fifteen children to that exact project authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
from threading import RLock
from typing import Any
import uuid

from .artifact_cancellation import CancellationProbe, raise_if_cancelled
from .artifact_document import (
    ArtifactDocument,
    DerivedRecord,
    RecordFreshness,
    RecordLifecycleStatus,
)
from .artifact_outline_extractor import OutlineView, outline_frame
from .artifact_rubbing_export import (
    RUBBING_EXPORT_DIRECTORY_SUFFIX,
    RUBBING_EXPORT_PNG_NAME,
    RUBBING_EXPORT_SIDECAR_NAME,
    ArtifactRubbingExportError,
    build_rubbing_export,
    validate_rubbing_export_package,
)
from .artifact_rubbing_extractor import (
    compute_artifact_rubbing_from_recipe,
    require_current_rubbing_computation,
)
from .artifact_rubbing_record import (
    RUBBING_RECORD_TYPE,
    rubbing_receipt_from_record,
    validate_rubbing_recipe,
)
from .artifact_session import ArtifactSession
from .artifact_vector_export import (
    MAX_IGNORABLE_OS_METADATA_BYTES,
    VECTOR_EXPORT_DIRECTORY_SUFFIX,
    VECTOR_EXPORT_SIDECAR_NAME,
    VECTOR_EXPORT_SVG_NAME,
    ArtifactVectorExportError,
    build_vector_export,
    fsync_export_directory,
    publish_export_directory_noreplace,
    read_bounded_export_file,
    validate_vector_export_package,
    write_new_export_file,
)
from .artifact_vector_record import (
    ArtifactVectorRecordError,
    PlanarFrame,
    VectorRecordKind,
    validate_vector_recipe,
)
from .canonical_json import CanonicalJSONError, canonical_json_bytes, canonical_json_sha256


SURVEY_EXPORT_DIRECTORY_SUFFIX = ".amr-survey"
SURVEY_EXPORT_FORMAT = "archmeshrubbing_survey_export"
SURVEY_EXPORT_SCHEMA_VERSION = "1.0.0"
SURVEY_EXPORT_MANIFEST_NAME = "survey.amr-survey.json"
SURVEY_EXPORT_MANIFEST_MEDIA_TYPE = (
    "application/vnd.archmeshrubbing.survey-export+json"
)
MAX_SURVEY_EXPORT_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_SURVEY_EXPORT_TREE_ENTRIES = 64
SURVEY_CUTLINE_VIEWS = ("top", "front", "right")
SURVEY_SIX_VIEWS = tuple(view.value for view in OutlineView)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")
_STAGING_PREFIX = ".amrs-stage-"
_QUARANTINE_PREFIX = ".amrs-discard-"
_IGNORABLE_OS_METADATA_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


class ArtifactSurveyExportError(ValueError):
    """A complete survey package violates authority or filesystem safety."""

    def __init__(self, message: str, *, committed: bool = False) -> None:
        super().__init__(message)
        self.committed = bool(committed)


@dataclass(frozen=True, slots=True)
class SurveyExportSelection:
    """One deterministic 3/6/6 record selection for a complete survey."""

    cutline_record_ids: tuple[str, ...]
    outline_record_ids: tuple[str, ...]
    rubbing_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = (
            ("cutline_record_ids", self.cutline_record_ids, SURVEY_CUTLINE_VIEWS),
            ("outline_record_ids", self.outline_record_ids, SURVEY_SIX_VIEWS),
            ("rubbing_record_ids", self.rubbing_record_ids, SURVEY_SIX_VIEWS),
        )
        all_ids: list[str] = []
        for field_name, values, views in groups:
            resolved = tuple(values)
            if len(resolved) != len(views):
                raise ArtifactSurveyExportError(
                    f"{field_name} must contain exactly {len(views)} records"
                )
            if any(not isinstance(value, str) or not value.strip() for value in resolved):
                raise ArtifactSurveyExportError(
                    f"{field_name} must contain non-empty record IDs"
                )
            object.__setattr__(self, field_name, resolved)
            all_ids.extend(resolved)
        if len(set(all_ids)) != len(all_ids):
            raise ArtifactSurveyExportError(
                "a survey record ID may appear in only one canonical slot"
            )

    @property
    def record_ids(self) -> tuple[str, ...]:
        return (
            *self.cutline_record_ids,
            *self.outline_record_ids,
            *self.rubbing_record_ids,
        )


@dataclass(frozen=True, slots=True)
class SurveyExportBundle:
    manifest: Mapping[str, Any]
    manifest_bytes: bytes
    manifest_sha256: str
    artifact_set_sha256: str
    vector_count: int
    rubbing_count: int

    @property
    def artifact_count(self) -> int:
        return self.vector_count + self.rubbing_count


@dataclass(frozen=True, slots=True)
class _SurveySlot:
    step: str
    view: str
    artifact_kind: str
    record_id: str

    @property
    def directory_name(self) -> str:
        suffix = (
            VECTOR_EXPORT_DIRECTORY_SUFFIX
            if self.artifact_kind == "vector_export"
            else RUBBING_EXPORT_DIRECTORY_SUFFIX
        )
        return f"{self.step}-{self.view}{suffix}"


@dataclass(frozen=True, slots=True)
class _OwnedStagingDirectory:
    path: Path
    destination: Path
    device: int
    inode: int
    parent_device: int
    parent_inode: int
    staging_tree_fsync_confirmed: bool = False


@dataclass(frozen=True, slots=True)
class _TreeEntryFingerprint:
    relative_path: str
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True, eq=False)
class PreparedSurveyPublication:
    """Exact, one-use capability for the final survey-directory rename."""

    staging_directory: Path
    destination: Path
    _owned: _OwnedStagingDirectory
    _fingerprint: tuple[_TreeEntryFingerprint, ...]
    _nonce: object


_STAGING_LOCK = RLock()
_STAGING_OWNERS: dict[str, _OwnedStagingDirectory] = {}
_PREPARED_PUBLICATIONS: dict[object, PreparedSurveyPublication] = {}


def _slots(selection: SurveyExportSelection) -> tuple[_SurveySlot, ...]:
    return tuple(
        [
            _SurveySlot("cutline", view, "vector_export", record_id)
            for view, record_id in zip(
                SURVEY_CUTLINE_VIEWS,
                selection.cutline_record_ids,
                strict=True,
            )
        ]
        + [
            _SurveySlot("outline", view, "vector_export", record_id)
            for view, record_id in zip(
                SURVEY_SIX_VIEWS,
                selection.outline_record_ids,
                strict=True,
            )
        ]
        + [
            _SurveySlot("rubbing", view, "rubbing_export", record_id)
            for view, record_id in zip(
                SURVEY_SIX_VIEWS,
                selection.rubbing_record_ids,
                strict=True,
            )
        ]
    )


def _expected_slots_without_ids() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        [("cutline", view, "vector_export") for view in SURVEY_CUTLINE_VIEWS]
        + [("outline", view, "vector_export") for view in SURVEY_SIX_VIEWS]
        + [("rubbing", view, "rubbing_export") for view in SURVEY_SIX_VIEWS]
    )


def _required_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactSurveyExportError(f"{label} must be an object")
    return value


def _exact_keys(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    result = _required_mapping(value, label=label)
    observed = set(result)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise ArtifactSurveyExportError(
            f"{label} is missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise ArtifactSurveyExportError(
            f"{label} has unknown fields: {', '.join(unknown)}"
        )
    return result


def _required_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ArtifactSurveyExportError(f"{label} must be a lowercase SHA-256")
    return value


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactSurveyExportError(f"{label} must be a non-empty string")
    return value


def _strict_json_bytes(payload: bytes) -> dict[str, Any]:
    if not payload or len(payload) > MAX_SURVEY_EXPORT_MANIFEST_BYTES:
        raise ArtifactSurveyExportError("survey manifest size is outside the limit")

    def reject_constant(value: str) -> None:
        raise ArtifactSurveyExportError(
            f"survey manifest contains invalid constant {value}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactSurveyExportError(
                    f"survey manifest contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactSurveyExportError(
            "survey manifest is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ArtifactSurveyExportError("survey manifest root must be an object")
    try:
        canonical = canonical_json_bytes(value) + b"\n"
    except CanonicalJSONError as exc:
        raise ArtifactSurveyExportError(str(exc)) from exc
    if payload != canonical:
        raise ArtifactSurveyExportError(
            "survey manifest is not canonical RFC 8785 JSON plus one newline"
        )
    return value


def _sidecar_mapping(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover
        raise ArtifactSurveyExportError(
            f"validated {label} sidecar could not be decoded"
        ) from exc
    return _required_mapping(value, label=f"{label} sidecar")


def _record_for_slot(
    document: ArtifactDocument,
    slot: _SurveySlot,
) -> DerivedRecord:
    record = document.record_index.get(slot.record_id)
    if record is None:
        raise ArtifactSurveyExportError(
            f"survey record {slot.record_id!r} does not exist"
        )
    expected_type = (
        VectorRecordKind.CUTLINE.record_type
        if slot.step == "cutline"
        else (
            VectorRecordKind.OUTLINE.record_type
            if slot.step == "outline"
            else RUBBING_RECORD_TYPE
        )
    )
    if record.type != expected_type:
        raise ArtifactSurveyExportError(
            f"survey slot {slot.step}/{slot.view} has the wrong record type"
        )
    if record.lifecycle_status is not RecordLifecycleStatus.READY:
        raise ArtifactSurveyExportError("survey records must be READY")
    if document.record_freshness(record.id) is not RecordFreshness.FRESH:
        raise ArtifactSurveyExportError("survey records must be FRESH")

    if slot.step == "cutline":
        try:
            validate_vector_recipe(
                record.recipe,
                expected_kind=VectorRecordKind.CUTLINE,
            )
            frame = PlanarFrame.from_dict(
                _required_mapping(
                    record.recipe.get("frame"),
                    label="Cutline recipe frame",
                )
            )
        except (ArtifactVectorRecordError, TypeError, ValueError) as exc:
            raise ArtifactSurveyExportError(str(exc)) from exc
        expected_frame = outline_frame(slot.view)
        if (
            frame.u_axis_world,
            frame.v_axis_world,
            frame.normal_world,
        ) != (
            expected_frame.u_axis_world,
            expected_frame.v_axis_world,
            expected_frame.normal_world,
        ):
            raise ArtifactSurveyExportError(
                f"Cutline record does not match canonical {slot.view} axes"
            )
    elif slot.step == "outline":
        try:
            validate_vector_recipe(
                record.recipe,
                expected_kind=VectorRecordKind.OUTLINE,
            )
        except ArtifactVectorRecordError as exc:
            raise ArtifactSurveyExportError(str(exc)) from exc
        if record.recipe.get("view") != slot.view:
            raise ArtifactSurveyExportError(
                f"Outline record does not match canonical {slot.view} view"
            )
    else:
        try:
            recipe = validate_rubbing_recipe(record.recipe)
        except Exception as exc:
            raise ArtifactSurveyExportError(str(exc)) from exc
        if recipe.get("view") != slot.view:
            raise ArtifactSurveyExportError(
                f"Digital Rubbing record does not match canonical {slot.view} view"
            )
    return record


def _validate_selection(
    document: ArtifactDocument,
    selection: SurveyExportSelection,
) -> tuple[DerivedRecord, ...]:
    if not isinstance(document, ArtifactDocument):
        raise ArtifactSurveyExportError("document must be an ArtifactDocument")
    if not isinstance(selection, SurveyExportSelection):
        raise ArtifactSurveyExportError(
            "selection must be a SurveyExportSelection"
        )
    records = tuple(_record_for_slot(document, slot) for slot in _slots(selection))
    geometry_revision_ids = {record.geometry_revision_id for record in records}
    align_ids = {record.align_revision_id for record in records}
    if len(geometry_revision_ids) != 1 or len(align_ids) != 1:
        raise ArtifactSurveyExportError(
            "survey records do not share one geometry and Align authority"
        )
    if align_ids != {document.active_align_revision_id}:
        raise ArtifactSurveyExportError(
            "survey records do not belong to the active Align revision"
        )
    return records


def _authority_from_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    document = _required_mapping(provenance.get("document"), label="provenance document")
    record = _required_mapping(provenance.get("record"), label="provenance record")
    source_assets = provenance.get("source_assets")
    if not isinstance(source_assets, list) or not source_assets:
        raise ArtifactSurveyExportError("provenance source_assets must be non-empty")
    assets: list[dict[str, Any]] = []
    for index, value in enumerate(source_assets):
        asset = _required_mapping(value, label=f"source asset {index}")
        assets.append(
            {
                "id": _required_text(asset.get("id"), label="source asset id"),
                "sha256": _required_sha256(
                    asset.get("sha256"),
                    label="source asset sha256",
                ),
                "size_bytes": asset.get("size_bytes"),
            }
        )
        if type(assets[-1]["size_bytes"]) is not int or assets[-1]["size_bytes"] < 0:
            raise ArtifactSurveyExportError("source asset size_bytes is invalid")
    return {
        "active_align_revision_id": _required_text(
            document.get("active_align_revision_id"),
            label="active Align revision ID",
        ),
        "active_source_metadata_revision_id": _required_text(
            document.get("active_source_metadata_revision_id"),
            label="active source metadata revision ID",
        ),
        "document_id": _required_text(
            document.get("document_id"),
            label="document ID",
        ),
        "document_manifest_sha256": _required_sha256(
            document.get("manifest_sha256"),
            label="document manifest SHA-256",
        ),
        "geometry_revision_id": _required_text(
            record.get("geometry_revision_id"),
            label="geometry revision ID",
        ),
        "source_assets": assets,
    }


def _entry_from_validated_child(
    child: Path,
    slot: _SurveySlot,
    *,
    document: ArtifactDocument | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if slot.artifact_kind == "vector_export":
        try:
            bundle = validate_vector_export_package(child, document=document)
        except ArtifactVectorExportError as exc:
            raise ArtifactSurveyExportError(str(exc)) from exc
        sidecar = _sidecar_mapping(bundle.sidecar_bytes, label="vector export")
        primary_name = VECTOR_EXPORT_SVG_NAME
        primary_sha256 = bundle.svg_sha256
        primary_size = len(bundle.svg_bytes)
        sidecar_name = VECTOR_EXPORT_SIDECAR_NAME
        sidecar_sha256 = bundle.sidecar_sha256
        sidecar_size = len(bundle.sidecar_bytes)
        physical_scale = "1:1"
    else:
        try:
            bundle = validate_rubbing_export_package(child, document=document)
        except ArtifactRubbingExportError as exc:
            raise ArtifactSurveyExportError(str(exc)) from exc
        sidecar = _sidecar_mapping(bundle.sidecar_bytes, label="rubbing export")
        primary_name = RUBBING_EXPORT_PNG_NAME
        primary_sha256 = bundle.png_sha256
        primary_size = len(bundle.png_bytes)
        sidecar_name = RUBBING_EXPORT_SIDECAR_NAME
        sidecar_sha256 = bundle.sidecar_sha256
        sidecar_size = len(bundle.sidecar_bytes)
        physical_scale = "1:1_planar_sampling"

    provenance = _required_mapping(sidecar.get("provenance"), label="child provenance")
    record = _required_mapping(provenance.get("record"), label="child record provenance")
    if record.get("id") != slot.record_id:
        raise ArtifactSurveyExportError(
            f"child package {slot.directory_name} has a different record ID"
        )
    recipe = _required_mapping(sidecar.get("recipe"), label="child recipe")
    if slot.step == "cutline":
        if recipe.get("kind") != VectorRecordKind.CUTLINE.value:
            raise ArtifactSurveyExportError("Cutline child recipe kind is invalid")
        try:
            frame = PlanarFrame.from_dict(
                _required_mapping(recipe.get("frame"), label="Cutline child frame")
            )
        except (ArtifactVectorRecordError, TypeError, ValueError) as exc:
            raise ArtifactSurveyExportError(str(exc)) from exc
        expected = outline_frame(slot.view)
        if (
            frame.u_axis_world,
            frame.v_axis_world,
            frame.normal_world,
        ) != (
            expected.u_axis_world,
            expected.v_axis_world,
            expected.normal_world,
        ):
            raise ArtifactSurveyExportError("Cutline child view axes are invalid")
    elif recipe.get("view") != slot.view:
        raise ArtifactSurveyExportError(
            f"child package {slot.directory_name} has a different canonical view"
        )

    entry = {
        "artifact_kind": slot.artifact_kind,
        "directory": slot.directory_name,
        "physical_scale": physical_scale,
        "primary_file": primary_name,
        "primary_sha256": primary_sha256,
        "primary_size_bytes": primary_size,
        "record_id": slot.record_id,
        "sidecar_file": sidecar_name,
        "sidecar_sha256": sidecar_sha256,
        "sidecar_size_bytes": sidecar_size,
        "step": slot.step,
        "view": slot.view,
    }
    return entry, _authority_from_provenance(provenance)


def _manifest(
    entries: list[dict[str, Any]],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_set_sha256 = canonical_json_sha256(entries)
    return {
        "artifact_set_sha256": artifact_set_sha256,
        "artifacts": entries,
        "authority": dict(authority),
        "format": SURVEY_EXPORT_FORMAT,
        "privacy": {
            "absolute_source_path_embedded": False,
            "external_resources": False,
        },
        "qc": {
            "artifact_count": 15,
            "coverage_complete": True,
            "rubbing_count": 6,
            "vector_count": 9,
        },
        "schema_version": SURVEY_EXPORT_SCHEMA_VERSION,
        "workflow": {
            "cutline_views": list(SURVEY_CUTLINE_VIEWS),
            "outline_views": list(SURVEY_SIX_VIEWS),
            "rubbing_views": list(SURVEY_SIX_VIEWS),
        },
    }


def _selection_from_manifest_entries(entries: object) -> SurveyExportSelection:
    if not isinstance(entries, list) or len(entries) != 15:
        raise ArtifactSurveyExportError(
            "survey manifest must contain exactly fifteen artifact entries"
        )
    expected_slots = _expected_slots_without_ids()
    record_ids: list[str] = []
    for index, (value, expected) in enumerate(zip(entries, expected_slots, strict=True)):
        entry = _exact_keys(
            value,
            {
                "artifact_kind",
                "directory",
                "physical_scale",
                "primary_file",
                "primary_sha256",
                "primary_size_bytes",
                "record_id",
                "sidecar_file",
                "sidecar_sha256",
                "sidecar_size_bytes",
                "step",
                "view",
            },
            label=f"artifact entry {index}",
        )
        step, view, artifact_kind = expected
        if (
            entry.get("step"),
            entry.get("view"),
            entry.get("artifact_kind"),
        ) != expected:
            raise ArtifactSurveyExportError(
                "survey artifact entries are not in canonical 3/6/6 order"
            )
        suffix = (
            VECTOR_EXPORT_DIRECTORY_SUFFIX
            if artifact_kind == "vector_export"
            else RUBBING_EXPORT_DIRECTORY_SUFFIX
        )
        if entry.get("directory") != f"{step}-{view}{suffix}":
            raise ArtifactSurveyExportError("survey child directory name is invalid")
        expected_primary = (
            VECTOR_EXPORT_SVG_NAME
            if artifact_kind == "vector_export"
            else RUBBING_EXPORT_PNG_NAME
        )
        expected_sidecar = (
            VECTOR_EXPORT_SIDECAR_NAME
            if artifact_kind == "vector_export"
            else RUBBING_EXPORT_SIDECAR_NAME
        )
        expected_scale = (
            "1:1" if artifact_kind == "vector_export" else "1:1_planar_sampling"
        )
        if (
            entry.get("primary_file") != expected_primary
            or entry.get("sidecar_file") != expected_sidecar
            or entry.get("physical_scale") != expected_scale
        ):
            raise ArtifactSurveyExportError("survey child descriptor is invalid")
        for key in ("primary_sha256", "sidecar_sha256"):
            _required_sha256(entry.get(key), label=f"artifact entry {key}")
        for key in ("primary_size_bytes", "sidecar_size_bytes"):
            size = entry.get(key)
            if type(size) is not int or size <= 0:
                raise ArtifactSurveyExportError(
                    f"artifact entry {key} must be a positive integer"
                )
        record_ids.append(_required_text(entry.get("record_id"), label="record ID"))
    return SurveyExportSelection(
        cutline_record_ids=tuple(record_ids[:3]),
        outline_record_ids=tuple(record_ids[3:9]),
        rubbing_record_ids=tuple(record_ids[9:15]),
    )


def validate_survey_export_package(
    directory: str | os.PathLike[str],
    *,
    document: ArtifactDocument | None = None,
) -> SurveyExportBundle:
    """Validate all fifteen children and their canonical aggregate manifest."""

    path = Path(directory)
    try:
        identity = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactSurveyExportError(
            f"survey export package cannot be inspected: {exc}"
        ) from exc
    if not stat.S_ISDIR(identity.st_mode) or path.is_symlink():
        raise ArtifactSurveyExportError("survey export package must be a real directory")
    try:
        manifest_bytes = read_bounded_export_file(
            path / SURVEY_EXPORT_MANIFEST_NAME,
            limit=MAX_SURVEY_EXPORT_MANIFEST_BYTES,
            label="survey manifest",
        )
    except ArtifactVectorExportError as exc:
        raise ArtifactSurveyExportError(str(exc)) from exc
    manifest = _strict_json_bytes(manifest_bytes)
    root = _exact_keys(
        manifest,
        {
            "artifact_set_sha256",
            "artifacts",
            "authority",
            "format",
            "privacy",
            "qc",
            "schema_version",
            "workflow",
        },
        label="survey manifest",
    )
    if root.get("format") != SURVEY_EXPORT_FORMAT:
        raise ArtifactSurveyExportError("survey export format is invalid")
    if root.get("schema_version") != SURVEY_EXPORT_SCHEMA_VERSION:
        raise ArtifactSurveyExportError("survey export schema version is invalid")
    selection = _selection_from_manifest_entries(root.get("artifacts"))
    if document is not None:
        _validate_selection(document, selection)

    expected_names = {
        SURVEY_EXPORT_MANIFEST_NAME,
        *(slot.directory_name for slot in _slots(selection)),
    }
    try:
        entries = list(path.iterdir())
    except OSError as exc:
        raise ArtifactSurveyExportError(
            f"survey export package cannot be enumerated: {exc}"
        ) from exc
    normative = {entry.name for entry in entries if entry.name not in _IGNORABLE_OS_METADATA_NAMES}
    if normative != expected_names:
        raise ArtifactSurveyExportError(
            "survey export package has missing or unexpected normative entries"
        )
    for entry in entries:
        if entry.name in _IGNORABLE_OS_METADATA_NAMES:
            item = entry.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(item.st_mode)
                or entry.is_symlink()
                or item.st_size > MAX_IGNORABLE_OS_METADATA_BYTES
            ):
                raise ArtifactSurveyExportError("OS metadata entry is unsafe")

    actual_entries: list[dict[str, Any]] = []
    common_authority: dict[str, Any] | None = None
    for slot in _slots(selection):
        actual_entry, authority = _entry_from_validated_child(
            path / slot.directory_name,
            slot,
            document=document,
        )
        if common_authority is None:
            common_authority = authority
        elif authority != common_authority:
            raise ArtifactSurveyExportError(
                "survey child packages do not share one document authority"
            )
        actual_entries.append(actual_entry)
    assert common_authority is not None

    expected_manifest = _manifest(actual_entries, common_authority)
    if manifest != expected_manifest:
        raise ArtifactSurveyExportError(
            "survey manifest does not exactly describe the validated child packages"
        )
    if document is not None and common_authority["document_manifest_sha256"] != (
        document.canonical_sha256
    ):
        raise ArtifactSurveyExportError(
            "survey document authority does not match the supplied project"
        )
    return SurveyExportBundle(
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        artifact_set_sha256=str(expected_manifest["artifact_set_sha256"]),
        vector_count=9,
        rubbing_count=6,
    )


def _absolute_destination(directory: str | os.PathLike[str]) -> Path:
    try:
        destination = Path(os.path.abspath(os.fspath(Path(directory).expanduser())))
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactSurveyExportError(f"survey destination is invalid: {exc}") from exc
    if not destination.name.endswith(SURVEY_EXPORT_DIRECTORY_SUFFIX):
        raise ArtifactSurveyExportError(
            f"survey destination must end with {SURVEY_EXPORT_DIRECTORY_SUFFIX}"
        )
    return destination


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _registry_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _uuid_hex() -> str:
    token = uuid.uuid4().hex.lower()
    if _UUID_HEX_RE.fullmatch(token) is None:
        raise ArtifactSurveyExportError("UUID provider returned an invalid token")
    return token


def _require_real_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        value = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactSurveyExportError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISDIR(value.st_mode) or path.is_symlink():
        raise ArtifactSurveyExportError(f"{label} must be a real directory")
    return value


def _create_staging(destination: Path) -> _OwnedStagingDirectory:
    if _path_exists(destination):
        raise ArtifactSurveyExportError("survey destination already exists")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ArtifactSurveyExportError(
            f"cannot create survey destination parent: {exc}"
        ) from exc
    parent = _require_real_directory(destination.parent, label="survey parent")
    for _attempt in range(16):
        staging = destination.parent / f"{_STAGING_PREFIX}{_uuid_hex()}"
        try:
            os.mkdir(staging, 0o700)
        except FileExistsError:
            continue
        except OSError as exc:
            raise ArtifactSurveyExportError(
                f"cannot create survey staging directory: {exc}"
            ) from exc
        identity = _require_real_directory(staging, label="survey staging directory")
        return _OwnedStagingDirectory(
            path=staging,
            destination=destination,
            device=identity.st_dev,
            inode=identity.st_ino,
            parent_device=parent.st_dev,
            parent_inode=parent.st_ino,
        )
    raise ArtifactSurveyExportError("could not allocate a unique survey staging directory")


def _register_staging(owned: _OwnedStagingDirectory) -> None:
    key = _registry_key(owned.path)
    with _STAGING_LOCK:
        if key in _STAGING_OWNERS:
            raise ArtifactSurveyExportError("survey staging path is already registered")
        _STAGING_OWNERS[key] = owned


def _require_owned_identity(owned: _OwnedStagingDirectory) -> None:
    parent = _require_real_directory(owned.path.parent, label="survey parent")
    if (parent.st_dev, parent.st_ino) != (
        owned.parent_device,
        owned.parent_inode,
    ):
        raise ArtifactSurveyExportError("survey destination parent identity changed")
    staging = _require_real_directory(owned.path, label="survey staging directory")
    if (staging.st_dev, staging.st_ino) != (owned.device, owned.inode):
        raise ArtifactSurveyExportError("survey staging directory identity changed")


def _raise_if_owned_destination_visible(owned: _OwnedStagingDirectory) -> None:
    if not _path_exists(owned.destination):
        return
    try:
        value = owned.destination.stat(follow_symlinks=False)
    except OSError:
        return
    if stat.S_ISDIR(value.st_mode) and (value.st_dev, value.st_ino) == (
        owned.device,
        owned.inode,
    ):
        raise ArtifactSurveyExportError(
            "survey package is already visible at its destination",
            committed=True,
        )


def _capture_tree_fingerprint(root: Path) -> tuple[_TreeEntryFingerprint, ...]:
    found: list[_TreeEntryFingerprint] = []

    def visit(directory: Path, prefix: str) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise ArtifactSurveyExportError(
                f"cannot enumerate survey package tree: {exc}"
            ) from exc
        for child in children:
            relative = f"{prefix}/{child.name}" if prefix else child.name
            try:
                value = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArtifactSurveyExportError(
                    f"cannot inspect survey tree entry {relative!r}: {exc}"
                ) from exc
            if stat.S_ISLNK(value.st_mode):
                raise ArtifactSurveyExportError("survey package tree contains a symlink")
            if not (stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode)):
                raise ArtifactSurveyExportError(
                    "survey package tree contains an unsupported filesystem entry"
                )
            found.append(
                _TreeEntryFingerprint(
                    relative_path=relative,
                    device=value.st_dev,
                    inode=value.st_ino,
                    mode=value.st_mode,
                    size=value.st_size,
                    mtime_ns=value.st_mtime_ns,
                    ctime_ns=value.st_ctime_ns,
                )
            )
            if len(found) > MAX_SURVEY_EXPORT_TREE_ENTRIES:
                raise ArtifactSurveyExportError(
                    "survey package tree exceeds its entry limit"
                )
            if stat.S_ISDIR(value.st_mode):
                visit(child, relative)

    visit(root, "")
    return tuple(found)


def _invalidate_prepared_locked(owned: _OwnedStagingDirectory) -> None:
    for nonce, prepared in tuple(_PREPARED_PUBLICATIONS.items()):
        if prepared._owned is owned:
            _PREPARED_PUBLICATIONS.pop(nonce, None)


def _discard_owned_staging(owned: _OwnedStagingDirectory) -> bool:
    _require_owned_identity(owned)
    before = _capture_tree_fingerprint(owned.path)
    quarantine = owned.path.parent / f"{_QUARANTINE_PREFIX}{_uuid_hex()}"
    try:
        publish_export_directory_noreplace(owned.path, quarantine)
    except ArtifactVectorExportError as exc:
        raise ArtifactSurveyExportError(str(exc)) from exc
    identity = _require_real_directory(quarantine, label="survey quarantine")
    if (identity.st_dev, identity.st_ino) != (owned.device, owned.inode):
        raise ArtifactSurveyExportError(
            "survey staging was quarantined but its identity changed"
        )
    if _capture_tree_fingerprint(quarantine) != before:
        raise ArtifactSurveyExportError(
            "survey staging changed while it was being quarantined"
        )
    try:
        shutil.rmtree(quarantine)
    except OSError as exc:
        raise ArtifactSurveyExportError(
            f"survey staging was quarantined but cleanup is not proven: {exc}"
        ) from exc
    if _path_exists(quarantine):
        raise ArtifactSurveyExportError(
            "survey quarantine remains after cleanup"
        )
    return True


def _write_child(
    staging: Path,
    slot: _SurveySlot,
    session: ArtifactSession,
    *,
    cancellation_probe: CancellationProbe | None,
) -> bool:
    raise_if_cancelled(cancellation_probe)
    child = staging / slot.directory_name
    try:
        os.mkdir(child, 0o700)
    except OSError as exc:
        raise ArtifactSurveyExportError(
            f"cannot create survey child package: {exc}"
        ) from exc
    if slot.artifact_kind == "vector_export":
        bundle = build_vector_export(session.document, slot.record_id)
        write_new_export_file(child / VECTOR_EXPORT_SVG_NAME, bundle.svg_bytes)
        write_new_export_file(
            child / VECTOR_EXPORT_SIDECAR_NAME,
            bundle.sidecar_bytes,
        )
        confirmed = fsync_export_directory(child)
        validate_vector_export_package(child, document=session.document)
        return confirmed

    record = session.document.record_index[slot.record_id]
    computation = compute_artifact_rubbing_from_recipe(
        session,
        record.recipe,
        cancellation_probe=cancellation_probe,
    )
    require_current_rubbing_computation(session, computation)
    if computation.raster.receipt() != rubbing_receipt_from_record(record):
        raise ArtifactSurveyExportError(
            "recomputed Digital Rubbing does not match its durable record"
        )
    bundle = build_rubbing_export(
        session.document,
        slot.record_id,
        computation.raster,
    )
    write_new_export_file(child / RUBBING_EXPORT_PNG_NAME, bundle.png_bytes)
    write_new_export_file(
        child / RUBBING_EXPORT_SIDECAR_NAME,
        bundle.sidecar_bytes,
    )
    confirmed = fsync_export_directory(child)
    validate_rubbing_export_package(child, document=session.document)
    return confirmed


def stage_survey_export_package(
    directory: str | os.PathLike[str],
    session: ArtifactSession,
    selection: SurveyExportSelection,
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> Path:
    """Build and validate one complete hidden survey tree without publishing it."""

    if not isinstance(session, ArtifactSession):
        raise ArtifactSurveyExportError("session must be an ArtifactSession")
    _validate_selection(session.document, selection)
    destination = _absolute_destination(directory)
    owned = _create_staging(destination)
    _register_staging(owned)
    try:
        tree_fsync_confirmed = True
        for slot in _slots(selection):
            tree_fsync_confirmed = (
                _write_child(
                    owned.path,
                    slot,
                    session,
                    cancellation_probe=cancellation_probe,
                )
                and tree_fsync_confirmed
            )
        raise_if_cancelled(cancellation_probe)
        entries: list[dict[str, Any]] = []
        authority: dict[str, Any] | None = None
        for slot in _slots(selection):
            entry, child_authority = _entry_from_validated_child(
                owned.path / slot.directory_name,
                slot,
                document=session.document,
            )
            if authority is None:
                authority = child_authority
            elif authority != child_authority:
                raise ArtifactSurveyExportError(
                    "staged survey children do not share one authority"
                )
            entries.append(entry)
        assert authority is not None
        manifest = _manifest(entries, authority)
        manifest_bytes = canonical_json_bytes(manifest) + b"\n"
        write_new_export_file(
            owned.path / SURVEY_EXPORT_MANIFEST_NAME,
            manifest_bytes,
        )
        tree_fsync_confirmed = (
            fsync_export_directory(owned.path) and tree_fsync_confirmed
        )
        owned = replace(
            owned,
            staging_tree_fsync_confirmed=tree_fsync_confirmed,
        )
        with _STAGING_LOCK:
            _STAGING_OWNERS[_registry_key(owned.path)] = owned
        validate_survey_export_package(owned.path, document=session.document)
        raise_if_cancelled(cancellation_probe)
        return owned.path
    except Exception as exc:
        try:
            discarded = _discard_owned_staging(owned)
        except Exception as cleanup_exc:
            raise ArtifactSurveyExportError(
                "survey staging failed and cleanup is not proven"
            ) from cleanup_exc
        finally:
            with _STAGING_LOCK:
                _STAGING_OWNERS.pop(_registry_key(owned.path), None)
                _invalidate_prepared_locked(owned)
        if not discarded:
            raise ArtifactSurveyExportError(
                "survey staging failed and cleanup is not proven"
            ) from exc
        raise


def prepare_staged_survey_publication(
    staging_directory: str | os.PathLike[str],
    directory: str | os.PathLike[str],
    *,
    document: ArtifactDocument | None = None,
) -> PreparedSurveyPublication:
    destination = _absolute_destination(directory)
    staging = Path(os.path.abspath(os.fspath(staging_directory)))
    key = _registry_key(staging)
    with _STAGING_LOCK:
        owned = _STAGING_OWNERS.get(key)
        if owned is None:
            raise ArtifactSurveyExportError(
                "survey staging directory was not created by this process"
            )
        if owned.destination != destination:
            raise ArtifactSurveyExportError(
                "survey staging authority belongs to a different destination"
            )
    _require_owned_identity(owned)
    if _path_exists(destination):
        _raise_if_owned_destination_visible(owned)
        raise ArtifactSurveyExportError("survey destination already exists")
    before = _capture_tree_fingerprint(staging)
    validate_survey_export_package(staging, document=document)
    after = _capture_tree_fingerprint(staging)
    if before != after:
        raise ArtifactSurveyExportError(
            "survey staging tree changed while being validated"
        )
    nonce = object()
    prepared = PreparedSurveyPublication(
        staging_directory=staging,
        destination=destination,
        _owned=owned,
        _fingerprint=after,
        _nonce=nonce,
    )
    with _STAGING_LOCK:
        if _STAGING_OWNERS.get(key) is not owned:
            raise ArtifactSurveyExportError(
                "survey staging authority changed during validation"
            )
        _require_owned_identity(owned)
        if _capture_tree_fingerprint(staging) != after:
            raise ArtifactSurveyExportError(
                "survey staging tree changed after validation"
            )
        _PREPARED_PUBLICATIONS[nonce] = prepared
    return prepared


def discard_staged_survey_package(
    staging_directory: str | os.PathLike[str],
    directory: str | os.PathLike[str],
) -> bool:
    destination = _absolute_destination(directory)
    staging = Path(os.path.abspath(os.fspath(staging_directory)))
    key = _registry_key(staging)
    with _STAGING_LOCK:
        owned = _STAGING_OWNERS.get(key)
        if owned is None or owned.destination != destination:
            return False
        try:
            result = _discard_owned_staging(owned)
        finally:
            _STAGING_OWNERS.pop(key, None)
            _invalidate_prepared_locked(owned)
        return result


def discard_prepared_survey_package(
    prepared: PreparedSurveyPublication,
) -> bool:
    if not isinstance(prepared, PreparedSurveyPublication):
        raise ArtifactSurveyExportError(
            "prepared publication must be a PreparedSurveyPublication"
        )
    with _STAGING_LOCK:
        if _PREPARED_PUBLICATIONS.get(prepared._nonce) is not prepared:
            _raise_if_owned_destination_visible(prepared._owned)
            return False
    return discard_staged_survey_package(
        prepared.staging_directory,
        prepared.destination,
    )


def publish_prepared_survey_package(
    prepared: PreparedSurveyPublication,
) -> Path:
    if not isinstance(prepared, PreparedSurveyPublication):
        raise ArtifactSurveyExportError(
            "prepared publication must be a PreparedSurveyPublication"
        )
    owned = prepared._owned
    key = _registry_key(prepared.staging_directory)
    with _STAGING_LOCK:
        if _PREPARED_PUBLICATIONS.get(prepared._nonce) is not prepared:
            _raise_if_owned_destination_visible(owned)
            raise ArtifactSurveyExportError(
                "prepared survey publication capability is invalid or consumed"
            )
        if _STAGING_OWNERS.get(key) is not owned:
            _raise_if_owned_destination_visible(owned)
            raise ArtifactSurveyExportError(
                "survey staging authority is no longer current"
            )
        _require_owned_identity(owned)
        if _path_exists(prepared.destination):
            _raise_if_owned_destination_visible(owned)
            raise ArtifactSurveyExportError("survey destination already exists")
        if _capture_tree_fingerprint(prepared.staging_directory) != prepared._fingerprint:
            raise ArtifactSurveyExportError(
                "survey staging tree changed after preparation"
            )
        try:
            publish_export_directory_noreplace(
                prepared.staging_directory,
                prepared.destination,
            )
        except ArtifactVectorExportError as exc:
            raise ArtifactSurveyExportError(str(exc)) from exc
        try:
            published = prepared.destination.stat(follow_symlinks=False)
        except OSError as exc:
            raise ArtifactSurveyExportError(
                "survey was renamed but destination identity is unavailable",
                committed=True,
            ) from exc
        if not stat.S_ISDIR(published.st_mode) or (
            published.st_dev,
            published.st_ino,
        ) != (owned.device, owned.inode):
            raise ArtifactSurveyExportError(
                "survey destination is not the authorized staging inode",
                committed=True,
            )
        _STAGING_OWNERS.pop(key, None)
        _invalidate_prepared_locked(owned)
    try:
        parent_fsync_confirmed = fsync_export_directory(prepared.destination.parent)
    except OSError as exc:
        raise ArtifactSurveyExportError(
            "survey was atomically published but directory fsync failed; "
            "crash durability is uncertain",
            committed=True,
        ) from exc
    if not owned.staging_tree_fsync_confirmed or not parent_fsync_confirmed:
        raise ArtifactSurveyExportError(
            "survey was atomically published but directory fsync is unsupported; "
            "crash durability is uncertain",
            committed=True,
        )
    return prepared.destination


def export_survey_package(
    directory: str | os.PathLike[str],
    session: ArtifactSession,
    selection: SurveyExportSelection,
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> Path:
    """Stage and atomically publish one complete ``*.amr-survey`` package."""

    staging = stage_survey_export_package(
        directory,
        session,
        selection,
        cancellation_probe=cancellation_probe,
    )
    prepared: PreparedSurveyPublication | None = None
    try:
        prepared = prepare_staged_survey_publication(
            staging,
            directory,
            document=session.document,
        )
        return publish_prepared_survey_package(prepared)
    except Exception as exc:
        if isinstance(exc, ArtifactSurveyExportError) and exc.committed:
            raise
        try:
            discarded = (
                discard_prepared_survey_package(prepared)
                if prepared is not None
                else discard_staged_survey_package(staging, directory)
            )
        except ArtifactSurveyExportError as cleanup_exc:
            if cleanup_exc.committed:
                raise
            raise ArtifactSurveyExportError(
                "survey export failed and staging cleanup is not proven"
            ) from cleanup_exc
        if not discarded:
            raise ArtifactSurveyExportError(
                "survey export failed and staging cleanup is not proven"
            ) from exc
        raise


__all__ = [
    "ArtifactSurveyExportError",
    "MAX_SURVEY_EXPORT_MANIFEST_BYTES",
    "PreparedSurveyPublication",
    "SURVEY_CUTLINE_VIEWS",
    "SURVEY_EXPORT_DIRECTORY_SUFFIX",
    "SURVEY_EXPORT_FORMAT",
    "SURVEY_EXPORT_MANIFEST_MEDIA_TYPE",
    "SURVEY_EXPORT_MANIFEST_NAME",
    "SURVEY_EXPORT_SCHEMA_VERSION",
    "SURVEY_SIX_VIEWS",
    "SurveyExportBundle",
    "SurveyExportSelection",
    "discard_prepared_survey_package",
    "discard_staged_survey_package",
    "export_survey_package",
    "prepare_staged_survey_publication",
    "publish_prepared_survey_package",
    "stage_survey_export_package",
    "validate_survey_export_package",
]
