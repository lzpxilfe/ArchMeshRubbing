"""Strict, transactional ArchMeshRubbing project file I/O.

``.amr`` files are ZIP containers.  Container version 2 separates the AMR
container contract from the payload schema and adds an integrity manifest.
Plain JSON remains available only when the caller explicitly supplies a
``.json`` path; it is a debugging/import format, not an AMR container.

The public ``save_project``/``load_project`` API intentionally remains the
same as version 1 so the current GUI can adopt the safer storage layer without
also changing scene materialisation.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import errno
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tempfile
from typing import Any, BinaryIO, NoReturn
import zipfile

from .artifact_document import (
    ARTIFACT_DOCUMENT_SCHEMA_VERSION,
    ArtifactDocument,
    ArtifactDocumentError,
)
from .artifact_vector_record import (
    ArtifactVectorRecordError,
)
from .artifact_record_validation import ArtifactKnownRecordError, validate_known_records


PROJECT_FORMAT = "archmeshrubbing_project"
PROJECT_VERSION = 2
LEGACY_PROJECT_VERSION = 1
MANIFEST_NAME = "project.json"
CHECKSUMS_NAME = "checksums.json"
PAYLOAD_TYPE = "legacy_ui_state"
PAYLOAD_SCHEMA_VERSION = "1.0.0"
ARTIFACT_PAYLOAD_TYPE = "artifact_document"
ARTIFACT_PAYLOAD_SCHEMA_VERSION = ARTIFACT_DOCUMENT_SCHEMA_VERSION
MIGRATION_MARKER_NAME = "_migration"
_LEGACY_MIGRATION_MARKER = {
    "from_version": LEGACY_PROJECT_VERSION,
    "to_version": PROJECT_VERSION,
    "status": "legacy_unverified",
    "runtime_only": True,
    "requires_save_as": True,
}

# Project state can contain large face-index lists.  These caps are high enough
# for that use while ensuring malformed archives are rejected before unbounded
# decompression.  They are part of the v2 reader's defensive contract.
MAX_ZIP_MEMBERS = 64
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_CHECKSUMS_BYTES = 1024 * 1024
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_COMPRESSION_RATIO = 500.0
MAX_PROJECT_FILE_BYTES = 520 * 1024 * 1024
MAX_CENTRAL_DIRECTORY_BYTES = 8 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:[-+].*)?$")


class ProjectFormatError(RuntimeError):
    """The input is not a supported, trustworthy project document."""


class ProjectSerializationError(ProjectFormatError):
    """The supplied state cannot be represented as strict JSON."""


class ProjectSaveError(RuntimeError):
    """A transactional save failed before it could be committed.

    ``stage`` is stable enough for UI diagnostics and retry policy.  The
    original exception is retained as ``__cause__``.
    """

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        retryable: bool = True,
        committed: bool = False,
    ) -> None:
        super().__init__(message)
        self.stage = str(stage)
        self.retryable = bool(retryable)
        self.committed = bool(committed)


class UnsupportedProjectVersionError(ProjectFormatError):
    """A project container version cannot be executed by this build."""

    def __init__(
        self,
        found_version: object,
        *,
        inspection: dict[str, Any] | None = None,
        newer: bool = False,
    ) -> None:
        relation = "newer" if newer else "unsupported"
        super().__init__(
            f"{relation.capitalize()} project container version: {found_version!r} "
            f"(supported: {PROJECT_VERSION})"
        )
        self.found_version = found_version
        self.supported_version = PROJECT_VERSION
        self.newer = bool(newer)
        # An unknown major is never returned as executable state.  This small,
        # scalar-only view is safe for a future read-only inspection UI.
        self.read_only_inspection = bool(newer)
        self.inspection = dict(inspection or {})


class UnsupportedPayloadError(ProjectFormatError):
    """A v2 payload cannot be materialised by the legacy UI adapter."""

    def __init__(
        self,
        payload_type: object,
        payload_schema_version: object,
        *,
        newer: bool = False,
        inspection: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            "Unsupported project payload: "
            f"type={payload_type!r}, schema={payload_schema_version!r}"
        )
        self.payload_type = payload_type
        self.payload_schema_version = payload_schema_version
        self.newer = bool(newer)
        self.read_only_inspection = True
        self.inspection = dict(inspection or {})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _raise_non_finite(value: str) -> NoReturn:
    raise ProjectFormatError(f"Invalid JSON numeric constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ProjectFormatError(f"JSON number is outside the finite range: {value}")
    return parsed


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProjectFormatError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _strict_json_loads(raw_bytes: bytes, *, label: str) -> Any:
    try:
        raw = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProjectFormatError(f"Invalid UTF-8 in {label}: {exc}") from exc

    try:
        return json.loads(
            raw,
            parse_constant=_raise_non_finite,
            parse_float=_parse_finite_float,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except ProjectFormatError:
        raise
    except json.JSONDecodeError as exc:
        raise ProjectFormatError(f"Invalid JSON in {label}: {exc}") from exc
    except (RecursionError, ValueError) as exc:
        raise ProjectFormatError(f"Unsafe JSON value in {label}: {exc}") from exc


def _strict_json_dumps(value: Any, *, label: str) -> bytes:
    _reject_non_string_mapping_keys(value, label=label)
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise ProjectSerializationError(f"Invalid strict JSON value in {label}: {exc}") from exc
    return (raw + "\n").encode("utf-8")


def _reject_non_string_mapping_keys(value: Any, *, label: str) -> None:
    """Reject Python mapping keys that JSON would otherwise silently coerce."""

    pending: list[tuple[Any, str]] = [(value, "$")]
    visited: set[int] = set()
    while pending:
        current, path = pending.pop()
        if isinstance(current, dict):
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            for key, child in current.items():
                if not isinstance(key, str):
                    raise ProjectSerializationError(
                        f"Invalid JSON object key in {label} at {path}: "
                        f"expected string, got {type(key).__name__}"
                    )
                pending.append((child, f"{path}.{key}"))
        elif isinstance(current, (list, tuple)):
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            pending.extend(
                (child, f"{path}[{index}]")
                for index, child in enumerate(current)
            )


def _inspection_from_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Return scalar envelope fields only; never expose unknown executable state."""

    inspection: dict[str, Any] = {"read_only": True}
    for key in (
        "format",
        "version",
        "payload_type",
        "payload_schema_version",
        "saved_at",
    ):
        value = doc.get(key)
        if value is None or isinstance(value, (str, int, float, bool)):
            inspection[key] = value
    return inspection


def _require_envelope_identity(doc: dict[str, Any]) -> int:
    fmt = doc.get("format")
    if fmt != PROJECT_FORMAT:
        raise ProjectFormatError(f"Unsupported project format: {fmt!r}")

    version = doc.get("version")
    if type(version) is not int:  # bool is deliberately not an integer version.
        raise UnsupportedProjectVersionError(version)
    if version > PROJECT_VERSION:
        raise UnsupportedProjectVersionError(
            version,
            inspection=_inspection_from_document(doc),
            newer=True,
        )
    if version not in (LEGACY_PROJECT_VERSION, PROJECT_VERSION):
        raise UnsupportedProjectVersionError(version)
    return version


def _validate_payload_identity(
    doc: dict[str, Any],
    *,
    expected_payload_type: str = PAYLOAD_TYPE,
    expected_schema_version: str = PAYLOAD_SCHEMA_VERSION,
) -> None:
    payload_type = doc.get("payload_type")
    schema_version = doc.get("payload_schema_version")
    inspection = _inspection_from_document(doc)

    if payload_type != expected_payload_type:
        raise UnsupportedPayloadError(
            payload_type,
            schema_version,
            inspection=inspection,
        )
    if not isinstance(schema_version, str):
        raise ProjectFormatError("Invalid project document: payload schema version must be a string")
    match = _SEMVER_RE.fullmatch(schema_version)
    if match is None:
        raise ProjectFormatError(
            f"Invalid project payload schema version: {schema_version!r}"
        )

    supported_major = int(expected_schema_version.split(".", 1)[0])
    found_major = int(match.group(1))
    if found_major != supported_major:
        raise UnsupportedPayloadError(
            payload_type,
            schema_version,
            newer=found_major > supported_major,
            inspection=inspection,
        )


def _validate_v2_document(
    doc: dict[str, Any],
    *,
    allow_runtime_migration_marker: bool = False,
    expected_payload_type: str = PAYLOAD_TYPE,
    expected_schema_version: str = PAYLOAD_SCHEMA_VERSION,
) -> None:
    _validate_payload_identity(
        doc,
        expected_payload_type=expected_payload_type,
        expected_schema_version=expected_schema_version,
    )

    if MIGRATION_MARKER_NAME in doc and not allow_runtime_migration_marker:
        raise ProjectFormatError(
            f"{MIGRATION_MARKER_NAME} is runtime-only and must not appear in a native v2 document"
        )

    saved_at = doc.get("saved_at")
    if not isinstance(saved_at, str) or not saved_at.strip():
        raise ProjectFormatError("Invalid project document: missing 'saved_at' string")

    state = doc.get("state")
    if not isinstance(state, dict):
        raise ProjectFormatError("Invalid project document: missing 'state' object")

    meta = doc.get("meta")
    if not isinstance(meta, dict):
        raise ProjectFormatError("Invalid project document: missing 'meta' object")

    if expected_payload_type == ARTIFACT_PAYLOAD_TYPE:
        if doc.get("payload_schema_version") != ARTIFACT_PAYLOAD_SCHEMA_VERSION:
            raise UnsupportedPayloadError(
                doc.get("payload_type"),
                doc.get("payload_schema_version"),
                inspection=_inspection_from_document(doc),
            )
        try:
            artifact = ArtifactDocument.from_dict(state)
        except ArtifactDocumentError as exc:
            raise ProjectFormatError(f"Invalid ArtifactDocument payload: {exc}") from exc
        if artifact.schema_version != doc.get("payload_schema_version"):
            raise ProjectFormatError(
                "ArtifactDocument schema_version does not match payload_schema_version"
            )


def migrate_project_document(document: dict[str, Any]) -> dict[str, Any]:
    """Purely and deterministically normalise v1/v2 documents to AMR v2.

    Unknown nested payload fields (including any ``source_identity`` object)
    are copied verbatim.  The migration deliberately does not infer units,
    source hashes, or alignment transforms that were absent from v1.
    """

    if not isinstance(document, dict):
        raise ProjectFormatError("Invalid project document (expected JSON object)")

    version = _require_envelope_identity(document)
    try:
        migrated = copy.deepcopy(document)
    except RecursionError as exc:
        raise ProjectFormatError("Project document nesting is too deep") from exc

    if version == LEGACY_PROJECT_VERSION:
        state = migrated.get("state")
        if not isinstance(state, dict):
            raise ProjectFormatError("Invalid legacy project document: missing 'state' object")

        legacy_meta = migrated.get("meta")
        if legacy_meta is None:
            migrated["meta"] = {}
        elif not isinstance(legacy_meta, dict):
            # v1 tolerated arbitrary metadata.  Preserve it explicitly rather
            # than discarding it, while producing a strict v2 envelope.
            migrated["meta"] = {"_raw": copy.deepcopy(legacy_meta)}

        saved_at = migrated.get("saved_at")
        if not isinstance(saved_at, str) or not saved_at.strip():
            # Do not invent a historical timestamp during a deterministic
            # migration.  The explicit marker is honest and stable.
            migrated["saved_at"] = "legacy-unknown"

        migrated["version"] = PROJECT_VERSION
        migrated["payload_type"] = PAYLOAD_TYPE
        migrated["payload_schema_version"] = PAYLOAD_SCHEMA_VERSION
        migrated[MIGRATION_MARKER_NAME] = dict(_LEGACY_MIGRATION_MARKER)

        # v1 stored only a path hint.  Make the absence of an authoritative
        # content identity explicit without manufacturing a hash or changing
        # the legacy path/scale fields.  This is intentionally a plain payload
        # shape so the storage layer does not depend on source_identity.py.
        objects = state.get("objects")
        if isinstance(objects, list):
            for item in objects:
                if not isinstance(item, dict):
                    continue
                mesh = item.get("mesh")
                if isinstance(mesh, dict) and "source" not in mesh:
                    mesh["source"] = {
                        "identity": None,
                        "binding_status": "legacy_unverified",
                    }
                # v1 never stored a trustworthy immutable alignment revision.
                # Override any draft/unknown field rather than upgrading it.
                item["alignment"] = {"status": "legacy_unverifiable"}

    runtime_marker = migrated.get(MIGRATION_MARKER_NAME)
    allow_runtime_marker = runtime_marker == _LEGACY_MIGRATION_MARKER
    if runtime_marker is not None and not allow_runtime_marker:
        raise ProjectFormatError(f"Invalid runtime migration marker: {runtime_marker!r}")

    _validate_v2_document(
        migrated,
        allow_runtime_migration_marker=allow_runtime_marker,
    )
    return migrated


def _reject_runtime_marker_from_durable_document(doc: dict[str, Any]) -> None:
    if doc.get("version") == PROJECT_VERSION and MIGRATION_MARKER_NAME in doc:
        raise ProjectFormatError(
            f"{MIGRATION_MARKER_NAME} is runtime-only and cannot be read from durable v2 input"
        )


def _validate_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ProjectFormatError(f"Unsafe ZIP member name: {name!r}")


def _preflight_zip_directory(path: Path) -> None:
    """Bound entry count before ``zipfile`` allocates every ``ZipInfo``.

    Python's ``ZipFile`` parses the complete central directory in its
    constructor. Reading EOCD/ZIP64 counts first keeps the public 64-member
    limit meaningful even for hostile archives with millions of empty entries.
    """
    try:
        file_size = int(path.stat().st_size)
    except OSError:
        raise
    if file_size > MAX_PROJECT_FILE_BYTES:
        raise ProjectFormatError(
            f"Project file exceeds the {MAX_PROJECT_FILE_BYTES}-byte safety limit"
        )
    if file_size < 22:
        raise ProjectFormatError("Invalid AMR ZIP container: missing end-of-central-directory record")

    tail_size = min(file_size, 22 + 65535)
    with path.open("rb") as stream:
        stream.seek(file_size - tail_size)
        tail = stream.read(tail_size)

        signature = b"PK\x05\x06"
        search_end = len(tail)
        eocd_index = -1
        eocd_fields: tuple[int, ...] | None = None
        while True:
            candidate = tail.rfind(signature, 0, search_end)
            if candidate < 0:
                break
            if candidate + 22 <= len(tail):
                fields = struct.unpack_from("<4H2LH", tail, candidate + 4)
                comment_length = int(fields[-1])
                if candidate + 22 + comment_length == len(tail):
                    eocd_index = candidate
                    eocd_fields = tuple(int(value) for value in fields)
                    break
            search_end = candidate
        if eocd_index < 0 or eocd_fields is None:
            raise ProjectFormatError("Invalid AMR ZIP container: EOCD record not found")

        disk_number, cd_disk, entries_disk, entries_total, cd_size, cd_offset, _comment = (
            eocd_fields
        )
        if disk_number != 0 or cd_disk != 0 or entries_disk != entries_total:
            raise ProjectFormatError("Multi-disk ZIP containers are not supported")

        absolute_eocd = file_size - tail_size + eocd_index
        if entries_total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
            locator_offset = absolute_eocd - 20
            if locator_offset < 0:
                raise ProjectFormatError("Invalid ZIP64 container: locator is missing")
            stream.seek(locator_offset)
            locator = stream.read(20)
            if len(locator) != 20 or locator[:4] != b"PK\x06\x07":
                raise ProjectFormatError("Invalid ZIP64 container: locator is missing")
            _locator_disk, zip64_offset, total_disks = struct.unpack_from("<LQL", locator, 4)
            if _locator_disk != 0 or total_disks != 1:
                raise ProjectFormatError("Multi-disk ZIP64 containers are not supported")
            stream.seek(int(zip64_offset))
            zip64 = stream.read(56)
            if len(zip64) < 56 or zip64[:4] != b"PK\x06\x06":
                raise ProjectFormatError("Invalid ZIP64 end-of-central-directory record")
            (
                _record_size,
                _made_by,
                _needed,
                zip64_disk,
                zip64_cd_disk,
                zip64_entries_disk,
                zip64_entries_total,
                zip64_cd_size,
                zip64_cd_offset,
            ) = struct.unpack_from("<Q2H2L4Q", zip64, 4)
            if (
                zip64_disk != 0
                or zip64_cd_disk != 0
                or zip64_entries_disk != zip64_entries_total
            ):
                raise ProjectFormatError("Multi-disk ZIP64 containers are not supported")
            entries_total = int(zip64_entries_total)
            cd_size = int(zip64_cd_size)
            cd_offset = int(zip64_cd_offset)

        if entries_total > MAX_ZIP_MEMBERS:
            raise ProjectFormatError(
                f"Project ZIP has too many members ({entries_total} > {MAX_ZIP_MEMBERS})"
            )
        if cd_size > MAX_CENTRAL_DIRECTORY_BYTES:
            raise ProjectFormatError("Project ZIP central directory exceeds the safety limit")
        if cd_offset < 0 or cd_size < 0 or cd_offset + cd_size > absolute_eocd:
            raise ProjectFormatError("Invalid ZIP central-directory bounds")


def _validate_zip_infos(infos: list[zipfile.ZipInfo]) -> dict[str, zipfile.ZipInfo]:
    if len(infos) > MAX_ZIP_MEMBERS:
        raise ProjectFormatError(
            f"Project ZIP has too many members ({len(infos)} > {MAX_ZIP_MEMBERS})"
        )

    by_name: dict[str, zipfile.ZipInfo] = {}
    total_size = 0
    for info in infos:
        _validate_member_name(info.filename)
        if info.filename in by_name:
            raise ProjectFormatError(f"Duplicate ZIP member: {info.filename!r}")
        if info.is_dir():
            raise ProjectFormatError(f"Directory ZIP members are not supported: {info.filename!r}")
        if info.flag_bits & 0x1:
            raise ProjectFormatError(f"Encrypted ZIP member is not supported: {info.filename!r}")
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise ProjectFormatError(
                f"Unsupported ZIP compression for {info.filename!r}: {info.compress_type}"
            )
        if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
            raise ProjectFormatError(
                f"ZIP member is too large: {info.filename!r} ({info.file_size} bytes)"
            )

        total_size += info.file_size
        if total_size > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ProjectFormatError("Project ZIP uncompressed size exceeds the safety limit")

        if info.file_size:
            if info.compress_size <= 0:
                raise ProjectFormatError(
                    f"Invalid compressed size for ZIP member: {info.filename!r}"
                )
            ratio = float(info.file_size) / float(info.compress_size)
            if ratio > MAX_COMPRESSION_RATIO:
                raise ProjectFormatError(
                    f"ZIP member compression ratio exceeds the safety limit: {info.filename!r}"
                )
        by_name[info.filename] = info

    manifest_info = by_name.get(MANIFEST_NAME)
    if manifest_info is None:
        raise ProjectFormatError(f"Missing {MANIFEST_NAME} in project file")
    if manifest_info.file_size > MAX_MANIFEST_BYTES:
        raise ProjectFormatError(
            f"{MANIFEST_NAME} exceeds the {MAX_MANIFEST_BYTES}-byte safety limit"
        )
    checksum_info = by_name.get(CHECKSUMS_NAME)
    if checksum_info is not None and checksum_info.file_size > MAX_CHECKSUMS_BYTES:
        raise ProjectFormatError(
            f"{CHECKSUMS_NAME} exceeds the {MAX_CHECKSUMS_BYTES}-byte safety limit"
        )
    return by_name


def _read_zip_member(zf: zipfile.ZipFile, name: str) -> bytes:
    try:
        with zf.open(name, "r") as source:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_MEMBER_BYTES:
                    raise ProjectFormatError(f"ZIP member expanded beyond its safety limit: {name!r}")
                chunks.append(chunk)
            return b"".join(chunks)
    except (KeyError, RuntimeError, zipfile.BadZipFile, OSError) as exc:
        raise ProjectFormatError(f"Unable to read ZIP member {name!r}: {exc}") from exc


def _hash_zip_member(zf: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    try:
        with zf.open(name, "r") as source:
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
    except (KeyError, RuntimeError, zipfile.BadZipFile, OSError) as exc:
        raise ProjectFormatError(f"Unable to verify ZIP member {name!r}: {exc}") from exc
    return digest.hexdigest()


def _validate_v2_checksums(
    zf: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
) -> None:
    if CHECKSUMS_NAME not in members:
        raise ProjectFormatError(f"Missing {CHECKSUMS_NAME} in AMR v2 project file")

    checksum_doc = _strict_json_loads(
        _read_zip_member(zf, CHECKSUMS_NAME),
        label=CHECKSUMS_NAME,
    )
    if not isinstance(checksum_doc, dict):
        raise ProjectFormatError(f"Invalid {CHECKSUMS_NAME}: expected JSON object")
    if checksum_doc.get("algorithm") != "sha256":
        raise ProjectFormatError(f"Invalid {CHECKSUMS_NAME}: unsupported algorithm")
    files = checksum_doc.get("files")
    if not isinstance(files, dict):
        raise ProjectFormatError(f"Invalid {CHECKSUMS_NAME}: missing 'files' object")

    expected_names = set(members) - {CHECKSUMS_NAME}
    if set(files) != expected_names:
        raise ProjectFormatError(
            f"Invalid {CHECKSUMS_NAME}: file list does not match ZIP members"
        )

    for name in sorted(expected_names):
        expected_digest = files.get(name)
        if not isinstance(expected_digest, str) or _SHA256_RE.fullmatch(expected_digest) is None:
            raise ProjectFormatError(
                f"Invalid SHA-256 digest for ZIP member {name!r}"
            )
        actual_digest = _hash_zip_member(zf, name)
        if not hmac.compare_digest(actual_digest, expected_digest):
            raise ProjectFormatError(f"Checksum mismatch for ZIP member {name!r}")


def _normalize_loaded_document(
    doc: dict[str, Any],
    *,
    expected_payload_type: str,
    expected_schema_version: str,
) -> dict[str, Any]:
    """Validate one strict envelope for a specific public payload API."""

    version = _require_envelope_identity(doc)
    if expected_payload_type == PAYLOAD_TYPE:
        # Preserve the existing v1 -> v2 legacy UI migration contract exactly.
        return migrate_project_document(doc)
    if version != PROJECT_VERSION:
        raise UnsupportedPayloadError(
            doc.get("payload_type"),
            doc.get("payload_schema_version"),
            inspection=_inspection_from_document(doc),
        )
    _validate_v2_document(
        doc,
        expected_payload_type=expected_payload_type,
        expected_schema_version=expected_schema_version,
    )
    return doc


def _load_zip_document(
    path: Path,
    *,
    expected_payload_type: str = PAYLOAD_TYPE,
    expected_schema_version: str = PAYLOAD_SCHEMA_VERSION,
) -> dict[str, Any]:
    _preflight_zip_directory(path)
    try:
        with zipfile.ZipFile(path, "r") as zf:
            members = _validate_zip_infos(zf.infolist())
            manifest_bytes = _read_zip_member(zf, MANIFEST_NAME)
            doc = _strict_json_loads(manifest_bytes, label=MANIFEST_NAME)
            if not isinstance(doc, dict):
                raise ProjectFormatError("Invalid project document (expected JSON object)")

            # Detect a newer container before applying v2 checksum/schema
            # assumptions.  The manifest itself has already passed bounds,
            # strict UTF-8/JSON parsing, and CRC verification.
            version = _require_envelope_identity(doc)
            _reject_runtime_marker_from_durable_document(doc)
            if version == PROJECT_VERSION:
                _validate_v2_checksums(zf, members)
            else:
                bad_member = zf.testzip()
                if bad_member is not None:
                    raise ProjectFormatError(f"CRC failure in ZIP member {bad_member!r}")
    except UnsupportedProjectVersionError:
        raise
    except ProjectFormatError:
        raise
    except (zipfile.BadZipFile, OSError, EOFError) as exc:
        raise ProjectFormatError(f"Invalid AMR ZIP container: {exc}") from exc

    return _normalize_loaded_document(
        doc,
        expected_payload_type=expected_payload_type,
        expected_schema_version=expected_schema_version,
    )


def _load_json_document(
    path: Path,
    *,
    expected_payload_type: str = PAYLOAD_TYPE,
    expected_schema_version: str = PAYLOAD_SCHEMA_VERSION,
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError:
        raise
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ProjectFormatError(
            f"JSON project exceeds the {MAX_MANIFEST_BYTES}-byte safety limit"
        )
    doc = _strict_json_loads(raw, label=path.name)
    if not isinstance(doc, dict):
        raise ProjectFormatError("Invalid project document (expected JSON object)")
    _reject_runtime_marker_from_durable_document(doc)
    return _normalize_loaded_document(
        doc,
        expected_payload_type=expected_payload_type,
        expected_schema_version=expected_schema_version,
    )


def _write_zip_archive(
    handle: BinaryIO,
    manifest_bytes: bytes,
    checksums_bytes: bytes,
) -> None:
    with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, manifest_bytes)
        zf.writestr(CHECKSUMS_NAME, checksums_bytes)


def _flush_and_fsync(handle: BinaryIO) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _validate_staged_project(
    path: Path,
    expected_document: dict[str, Any],
    *,
    expected_payload_type: str = PAYLOAD_TYPE,
    expected_schema_version: str = PAYLOAD_SCHEMA_VERSION,
) -> None:
    loaded = _load_zip_document(
        path,
        expected_payload_type=expected_payload_type,
        expected_schema_version=expected_schema_version,
    )
    if loaded != expected_document:
        raise ProjectFormatError("Staged project validation did not reproduce the saved document")


def _best_effort_fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    fd: int | None = None
    try:
        fd = os.open(directory, flags)
        os.fsync(fd)
    except (AttributeError, NotImplementedError):
        # Directory fsync is unavailable on some supported platforms.
        pass
    except OSError as exc:
        unsupported_errnos = {
            errno.EACCES,
            errno.EBADF,
            errno.EINVAL,
            getattr(errno, "ENOSYS", errno.EINVAL),
            errno.EPERM,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if exc.errno not in unsupported_errnos:
            raise
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _save_payload_project(
    path: str | Path,
    state: dict[str, Any],
    *,
    payload_type: str,
    payload_schema_version: str,
    meta: dict[str, Any] | None = None,
) -> str:
    """Atomically save one already-validated AMR v2 payload.

    The new archive is written to a unique same-directory temporary file,
    flushed and fsynced, reopened through the production parser, and only then
    committed with ``os.replace``.  Any failure before replace leaves an
    existing destination byte-for-byte unchanged and removes the temporary.
    """

    if not isinstance(state, dict):
        raise ProjectSerializationError("Project state must be a JSON object")
    if meta is not None and not isinstance(meta, dict):
        raise ProjectSerializationError("Project metadata must be a JSON object")

    out_path = Path(path)
    if out_path.suffix.lower() == ".json":
        raise ProjectSaveError(
            "prepare",
            "save_project writes AMR ZIP containers; choose an .amr destination",
            retryable=False,
        )
    document: dict[str, Any] = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "payload_type": payload_type,
        "payload_schema_version": payload_schema_version,
        "saved_at": _utc_now_iso(),
        "meta": dict(meta or {}),
        "state": state,
    }
    manifest_bytes = _strict_json_dumps(document, label=MANIFEST_NAME)
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ProjectSerializationError(
            f"{MANIFEST_NAME} exceeds the {MAX_MANIFEST_BYTES}-byte safety limit"
        )
    # Compare validation against JSON-normalised values, not caller-specific
    # Python containers such as tuples that encode as JSON arrays.
    expected_document = _strict_json_loads(manifest_bytes, label=MANIFEST_NAME)
    assert isinstance(expected_document, dict)
    checksums_bytes = _strict_json_dumps(
        {
            "algorithm": "sha256",
            "files": {MANIFEST_NAME: hashlib.sha256(manifest_bytes).hexdigest()},
        },
        label=CHECKSUMS_NAME,
    )
    if len(checksums_bytes) > MAX_CHECKSUMS_BYTES:
        raise ProjectSerializationError(
            f"{CHECKSUMS_NAME} exceeds the {MAX_CHECKSUMS_BYTES}-byte safety limit"
        )

    temp_path: Path | None = None
    raw_fd: int | None = None
    committed = False
    stage = "prepare"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_fd, raw_temp_path = tempfile.mkstemp(
            prefix=f".{out_path.name}.",
            suffix=".tmp",
            dir=str(out_path.parent),
        )
        temp_path = Path(raw_temp_path)

        stage = "temp_write"
        with os.fdopen(raw_fd, "w+b") as handle:
            raw_fd = None
            _write_zip_archive(handle, manifest_bytes, checksums_bytes)
            stage = "temp_fsync"
            _flush_and_fsync(handle)

        stage = "validation"
        _validate_staged_project(
            temp_path,
            expected_document,
            expected_payload_type=payload_type,
            expected_schema_version=payload_schema_version,
        )

        stage = "replace"
        os.replace(temp_path, out_path)
        committed = True
        stage = "directory_fsync"
        try:
            _best_effort_fsync_directory(out_path.parent)
        except OSError as exc:
            raise ProjectSaveError(
                stage,
                "Project file was atomically replaced, but directory fsync failed; "
                f"crash durability is uncertain: {exc}",
                retryable=True,
                committed=True,
            ) from exc
        return str(out_path)
    except ProjectSerializationError:
        raise
    except ProjectSaveError:
        raise
    except Exception as exc:
        raise ProjectSaveError(
            stage,
            f"Project save failed during {stage}: {exc}",
            retryable=stage in {"prepare", "temp_write", "temp_fsync", "validation", "replace"},
            committed=committed,
        ) from exc
    finally:
        if raw_fd is not None:
            try:
                os.close(raw_fd)
            except OSError:
                pass
        if temp_path is not None and not committed:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def save_project(
    path: str | Path,
    state: dict[str, Any],
    *,
    meta: dict[str, Any] | None = None,
) -> str:
    """Atomically save the existing ``legacy_ui_state`` AMR v2 payload."""

    return _save_payload_project(
        path,
        state,
        payload_type=PAYLOAD_TYPE,
        payload_schema_version=PAYLOAD_SCHEMA_VERSION,
        meta=meta,
    )


def save_artifact_project(
    path: str | Path,
    document: ArtifactDocument,
    *,
    meta: dict[str, Any] | None = None,
) -> str:
    """Atomically save one authoritative ``ArtifactDocument`` AMR v2 payload."""

    if not isinstance(document, ArtifactDocument):
        raise ProjectSerializationError("document must be an ArtifactDocument")
    try:
        # Reparse the public representation so even a corrupted or unsafely
        # mutated frozen instance cannot bypass graph/schema validation.
        validated = ArtifactDocument.from_dict(document.to_dict())
        validate_known_records(validated)
    except (ArtifactDocumentError, ArtifactVectorRecordError, ArtifactKnownRecordError) as exc:
        raise ProjectSerializationError(f"Invalid ArtifactDocument: {exc}") from exc
    if validated.canonical_json_bytes() != document.canonical_json_bytes():
        raise ProjectSerializationError(
            "ArtifactDocument canonical serialization changed during validation"
        )
    return _save_payload_project(
        path,
        validated.to_dict(),
        payload_type=ARTIFACT_PAYLOAD_TYPE,
        payload_schema_version=ARTIFACT_PAYLOAD_SCHEMA_VERSION,
        meta=meta,
    )


def load_project(path: str | Path) -> dict[str, Any]:
    """Load, strictly validate, and migrate a project to an AMR v2 document.

    Only an explicit ``.json`` suffix selects the developer JSON import path.
    Every other path, especially ``.amr``, must be a valid ZIP container; a
    damaged ZIP can therefore never be mistaken for legacy plain JSON.
    """

    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))
    if in_path.suffix.lower() == ".json":
        return _load_json_document(in_path)
    return _load_zip_document(in_path)


def load_artifact_project(path: str | Path) -> ArtifactDocument:
    """Load and validate an authoritative ``artifact_document`` AMR v2 payload.

    Legacy UI payloads are deliberately not migrated here. Callers must choose
    the legacy or artifact API explicitly so no unit, geometry, Align, or record
    identity is ever invented during storage loading.
    """

    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))
    if in_path.suffix.lower() == ".json":
        envelope = _load_json_document(
            in_path,
            expected_payload_type=ARTIFACT_PAYLOAD_TYPE,
            expected_schema_version=ARTIFACT_PAYLOAD_SCHEMA_VERSION,
        )
    else:
        envelope = _load_zip_document(
            in_path,
            expected_payload_type=ARTIFACT_PAYLOAD_TYPE,
            expected_schema_version=ARTIFACT_PAYLOAD_SCHEMA_VERSION,
        )
    state = envelope.get("state")
    if not isinstance(state, dict):
        # The envelope validator already enforces this. Keep the public return
        # boundary defensive if internal loading changes later.
        raise ProjectFormatError("Invalid ArtifactDocument payload state")
    try:
        artifact = ArtifactDocument.from_dict(state)
        validate_known_records(artifact)
        return artifact
    except (ArtifactDocumentError, ArtifactVectorRecordError, ArtifactKnownRecordError) as exc:
        raise ProjectFormatError(f"Invalid ArtifactDocument payload: {exc}") from exc
