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
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import hmac
import json
import math
import ntpath
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tempfile
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, Iterator, NoReturn, cast
import zipfile

from .artifact_document import (
    ARTIFACT_DOCUMENT_SCHEMA_VERSION,
    PRIMARY_SOURCE_ASSET_ROLE,
    ArtifactDocument,
    ArtifactDocumentError,
)
from .artifact_vector_record import (
    ArtifactVectorRecordError,
)
from .artifact_record_validation import ArtifactKnownRecordError, validate_known_records
from .source_bundle import (
    SOURCE_INDEX_NAME,
    SourceBundleError,
    SourceBundleIndex,
    source_blob_member,
)
from .source_identity import SourceChangedError, open_fingerprinted_file
from .source_manifest import ResolvedSourceResource, SourceManifestEntry

if TYPE_CHECKING:
    from .artifact_session import ArtifactSession


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
MAX_SOURCE_INDEX_BYTES = 1024 * 1024
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
# Primary scans are commonly much larger than JSON/derived payloads. Keep the
# original defensive limits for ordinary members, while giving content-
# addressed, ZIP_STORED source blobs an explicit first-release budget.
MAX_SOURCE_MEMBER_BYTES = 16 * 1024 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES = 16 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 500.0
MAX_CENTRAL_DIRECTORY_BYTES = 8 * 1024 * 1024
MAX_PROJECT_FILE_BYTES = (
    MAX_TOTAL_SOURCE_BYTES
    + MAX_TOTAL_UNCOMPRESSED_BYTES
    + MAX_CENTRAL_DIRECTORY_BYTES
)
_COPY_CHUNK_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:[-+].*)?$")

# The Windows release target commits a fully written and production-validated
# same-directory staging file with the documented Win32 write-through rename.
# Keep the identifier stable: packaged workflow reports use it as an exact
# machine-readable durability gate.
WINDOWS_PROJECT_COMMIT_BACKEND = "windows-movefileex-write-through"
POSIX_PROJECT_COMMIT_BACKEND = "posix-replace-directory-fsync"
MOVEFILE_REPLACE_EXISTING = 0x00000001
MOVEFILE_WRITE_THROUGH = 0x00000008
_WINDOWS_PROJECT_COMMIT_FLAGS = MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH

_MoveFileExW = Callable[[str, str, int], int]
_GetLastError = Callable[[], int]


class ProjectFormatError(RuntimeError):
    """The input is not a supported, trustworthy project document."""


class ProjectSerializationError(ProjectFormatError):
    """The supplied state cannot be represented as strict JSON."""


class ProjectSaveError(RuntimeError):
    """A transactional save failed before or just after its commit boundary.

    ``stage`` is stable enough for UI diagnostics and retry policy.  The
    original exception is retained as ``__cause__``. ``committed=True`` is
    reserved for a successful POSIX replacement whose following directory
    fsync failed; a failed Windows write-through move remains uncommitted.
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


class EmbeddedSourceRequiredError(ProjectFormatError):
    """A session reopen was requested from a manifest-only project."""


@dataclass(frozen=True, slots=True)
class ArtifactProjectPackage:
    """Validated artifact document plus optional embedded-source inventory."""

    document: ArtifactDocument
    archive_path: Path
    source_bundle: SourceBundleIndex | None

    def __post_init__(self) -> None:
        if not isinstance(self.document, ArtifactDocument):
            raise TypeError("document must be an ArtifactDocument")
        object.__setattr__(self, "archive_path", Path(self.archive_path))
        if self.source_bundle is not None and not isinstance(
            self.source_bundle,
            SourceBundleIndex,
        ):
            raise TypeError("source_bundle must be a SourceBundleIndex or None")


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


def _source_digest_from_member(name: str) -> str | None:
    prefix = "sources/blobs/sha256/"
    if not name.startswith(prefix):
        return None
    digest = name.removeprefix(prefix)
    try:
        expected = source_blob_member(digest)
    except SourceBundleError as exc:
        raise ProjectFormatError(f"Invalid source blob member name: {name!r}") from exc
    if expected != name:
        raise ProjectFormatError(f"Invalid source blob member name: {name!r}")
    return digest


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

        stream.seek(cd_offset)
        central_directory = stream.read(cd_size)
        if len(central_directory) != cd_size:
            raise ProjectFormatError("Truncated ZIP central directory")

        # Do not trust the EOCD entry count: an attacker can lower it while
        # leaving thousands of real records for ZipFile to allocate. Parse the
        # already bounded central directory before constructing ZipFile.
        cursor = 0
        actual_entries = 0
        while cursor < len(central_directory):
            if len(central_directory) - cursor < 46:
                raise ProjectFormatError("Truncated ZIP central-directory record")
            if central_directory[cursor : cursor + 4] != b"PK\x01\x02":
                raise ProjectFormatError("Invalid ZIP central-directory record signature")
            filename_length, extra_length, comment_length = struct.unpack_from(
                "<3H",
                central_directory,
                cursor + 28,
            )
            record_size = 46 + filename_length + extra_length + comment_length
            if record_size > len(central_directory) - cursor:
                raise ProjectFormatError("Truncated ZIP central-directory record")
            actual_entries += 1
            if actual_entries > MAX_ZIP_MEMBERS:
                raise ProjectFormatError(
                    f"Project ZIP has too many members ({actual_entries} > {MAX_ZIP_MEMBERS})"
                )
            cursor += record_size

        if cursor != len(central_directory):
            raise ProjectFormatError("Invalid ZIP central-directory bounds")
        if actual_entries != entries_total:
            raise ProjectFormatError(
                "ZIP central-directory entry count does not match EOCD "
                f"({actual_entries} != {entries_total})"
            )


def _validate_zip_infos(infos: list[zipfile.ZipInfo]) -> dict[str, zipfile.ZipInfo]:
    if len(infos) > MAX_ZIP_MEMBERS:
        raise ProjectFormatError(
            f"Project ZIP has too many members ({len(infos)} > {MAX_ZIP_MEMBERS})"
        )

    by_name: dict[str, zipfile.ZipInfo] = {}
    total_size = 0
    total_source_size = 0
    for info in infos:
        _validate_member_name(info.filename)
        if info.filename in by_name:
            raise ProjectFormatError(f"Duplicate ZIP member: {info.filename!r}")
        if info.is_dir():
            raise ProjectFormatError(f"Directory ZIP members are not supported: {info.filename!r}")
        if info.create_system == 3:
            unix_mode = (int(info.external_attr) >> 16) & 0xFFFF
            unix_kind = stat.S_IFMT(unix_mode)
            if unix_kind not in (0, stat.S_IFREG):
                raise ProjectFormatError(
                    "Non-regular Unix ZIP members are not supported: "
                    f"{info.filename!r}"
                )
        if info.flag_bits & 0x1:
            raise ProjectFormatError(f"Encrypted ZIP member is not supported: {info.filename!r}")
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise ProjectFormatError(
                f"Unsupported ZIP compression for {info.filename!r}: {info.compress_type}"
            )
        source_digest = _source_digest_from_member(info.filename)
        if (
            info.filename.startswith("sources/")
            and info.filename != SOURCE_INDEX_NAME
            and source_digest is None
        ):
            raise ProjectFormatError(
                f"Unsupported member in reserved sources namespace: {info.filename!r}"
            )
        if source_digest is not None:
            if info.compress_type != zipfile.ZIP_STORED:
                raise ProjectFormatError(
                    f"Source blob must use ZIP_STORED compression: {info.filename!r}"
                )
            if info.file_size < 0 or info.file_size > MAX_SOURCE_MEMBER_BYTES:
                raise ProjectFormatError(
                    f"Source blob is too large: {info.filename!r} ({info.file_size} bytes)"
                )
            total_source_size += info.file_size
            if total_source_size > MAX_TOTAL_SOURCE_BYTES:
                raise ProjectFormatError(
                    "Project ZIP embedded source size exceeds the safety limit"
                )
        else:
            if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
                raise ProjectFormatError(
                    f"ZIP member is too large: {info.filename!r} ({info.file_size} bytes)"
                )
            total_size += info.file_size
            if total_size > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ProjectFormatError(
                    "Project ZIP uncompressed non-source size exceeds the safety limit"
                )

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
    source_index_info = by_name.get(SOURCE_INDEX_NAME)
    if (
        source_index_info is not None
        and source_index_info.file_size > MAX_SOURCE_INDEX_BYTES
    ):
        raise ProjectFormatError(
            f"{SOURCE_INDEX_NAME} exceeds the {MAX_SOURCE_INDEX_BYTES}-byte safety limit"
        )
    return by_name


def _read_zip_member(zf: zipfile.ZipFile, name: str) -> bytes:
    if _source_digest_from_member(name) is not None:
        raise ProjectFormatError(
            f"Source blob members must be consumed as bounded streams: {name!r}"
        )
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
) -> dict[str, str]:
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

    validated_files: dict[str, str] = {}
    for name in sorted(expected_names):
        expected_digest = files.get(name)
        if not isinstance(expected_digest, str) or _SHA256_RE.fullmatch(expected_digest) is None:
            raise ProjectFormatError(
                f"Invalid SHA-256 digest for ZIP member {name!r}"
            )
        actual_digest = _hash_zip_member(zf, name)
        if not hmac.compare_digest(actual_digest, expected_digest):
            raise ProjectFormatError(f"Checksum mismatch for ZIP member {name!r}")
        validated_files[name] = expected_digest
    return validated_files


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


def project_commit_backend_identifier(*, platform_name: str | None = None) -> str:
    """Return the exact staged-project commit backend for one platform.

    Windows is the release target and therefore has a required Win32
    write-through backend.  The POSIX identifier describes the historical
    compatibility path; it is not a macOS/Linux release-completion claim.
    """

    selected_platform = os.name if platform_name is None else str(platform_name)
    if selected_platform == "nt":
        return WINDOWS_PROJECT_COMMIT_BACKEND
    return POSIX_PROJECT_COMMIT_BACKEND


def _windows_extended_path(path: str | os.PathLike[str]) -> str:
    """Return an absolute Win32 extended-length path without losing Unicode.

    ``MoveFileExW`` receives Unicode strings with the Win32 extended-length
    prefix, so long local and UNC project paths do not fall back to legacy
    ``MAX_PATH`` parsing. Device namespace paths are deliberately not
    manufactured by the project writer, while an already-normalized extended
    path is preserved.
    """

    raw_path = os.fspath(path)
    if not isinstance(raw_path, str):
        raise TypeError("Windows project paths must be text paths")
    if "\x00" in raw_path:
        raise ValueError("Windows project paths cannot contain NUL")

    normalized = raw_path.replace("/", "\\")
    if normalized.startswith("\\\\?\\"):
        return normalized
    if normalized.startswith("\\\\.\\"):
        raise ValueError("Windows device namespace paths are not supported")
    if normalized.startswith("\\\\"):
        return "\\\\?\\UNC\\" + normalized[2:]

    drive, tail = ntpath.splitdrive(normalized)
    if not drive or not tail.startswith("\\"):
        # Production reaches this branch only on Windows. ``os.path.abspath``
        # then applies the process drive/current-directory rules before the
        # extended prefix disables Win32 path normalization.
        normalized = os.path.abspath(raw_path).replace("/", "\\")
        drive, tail = ntpath.splitdrive(normalized)
    if not drive or not tail.startswith("\\"):
        raise ValueError("Windows project paths must resolve to an absolute drive path")
    return "\\\\?\\" + normalized


def _load_move_file_ex_w() -> tuple[_MoveFileExW, _GetLastError]:
    """Load ``kernel32!MoveFileExW`` without importing pywin32."""

    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    win_dll_factory = getattr(ctypes, "WinDLL", None)
    get_last_error = getattr(ctypes, "get_last_error", None)
    if win_dll_factory is None or get_last_error is None:
        raise OSError("Win32 MoveFileExW is unavailable in this Python runtime")
    kernel32 = win_dll_factory("kernel32", use_last_error=True)
    move_file_ex = kernel32.MoveFileExW
    move_file_ex.argtypes = (
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
    )
    move_file_ex.restype = wintypes.BOOL
    return cast(_MoveFileExW, move_file_ex), cast(_GetLastError, get_last_error)


def _windows_replace_write_through(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    move_file_ex: _MoveFileExW | None = None,
    get_last_error: _GetLastError | None = None,
) -> None:
    """Replace ``destination`` with a same-volume write-through Win32 move."""

    if (move_file_ex is None) != (get_last_error is None):
        raise ValueError("MoveFileExW and GetLastError must be supplied together")
    if move_file_ex is None or get_last_error is None:
        move_file_ex, get_last_error = _load_move_file_ex_w()

    source_path = _windows_extended_path(source)
    destination_path = _windows_extended_path(destination)
    source_parent = ntpath.normcase(ntpath.dirname(source_path))
    destination_parent = ntpath.normcase(ntpath.dirname(destination_path))
    if not source_parent or source_parent != destination_parent:
        raise ValueError(
            "Windows project commit requires staging in the destination directory"
        )
    result = int(
        move_file_ex(
            source_path,
            destination_path,
            _WINDOWS_PROJECT_COMMIT_FLAGS,
        )
    )
    if result:
        return

    winerror = int(get_last_error())
    raise OSError(
        winerror,
        f"MoveFileExW write-through replacement failed (Win32 error {winerror})",
        os.fspath(destination),
    )


def _commit_staged_project(
    source: Path,
    destination: Path,
    *,
    platform_name: str | None = None,
    move_file_ex: _MoveFileExW | None = None,
    get_last_error: _GetLastError | None = None,
) -> str:
    """Commit one validated same-directory project staging file.

    A failed Windows call is a pre-commit failure and never falls back to a
    weaker rename.  On POSIX, the historical ``os.replace`` plus parent
    directory fsync behavior is retained, including its typed committed-but-
    uncertain error after a successful replacement.
    """

    backend = project_commit_backend_identifier(platform_name=platform_name)
    if backend == WINDOWS_PROJECT_COMMIT_BACKEND:
        _windows_replace_write_through(
            source,
            destination,
            move_file_ex=move_file_ex,
            get_last_error=get_last_error,
        )
        return backend

    if move_file_ex is not None or get_last_error is not None:
        raise ValueError("Win32 callables cannot be supplied to the POSIX backend")
    os.replace(source, destination)
    try:
        _best_effort_fsync_directory(destination.parent)
    except OSError as exc:
        raise ProjectSaveError(
            "directory_fsync",
            "Project file was atomically replaced, but POSIX directory fsync failed; "
            f"crash durability is uncertain: {exc}",
            retryable=True,
            committed=True,
        ) from exc
    return backend


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
    committed through the platform backend. Windows uses ``MoveFileExW`` with
    replace-existing and write-through flags; historical POSIX compatibility
    uses ``os.replace`` followed by parent-directory fsync. Any failure before
    commit leaves an existing destination byte-for-byte unchanged and removes
    the temporary.
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
        _commit_staged_project(temp_path, out_path)
        committed = True
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

    validated = _validated_artifact_document(document)
    return _save_payload_project(
        path,
        validated.to_dict(),
        payload_type=ARTIFACT_PAYLOAD_TYPE,
        payload_schema_version=ARTIFACT_PAYLOAD_SCHEMA_VERSION,
        meta=meta,
    )


def _validated_artifact_document(document: ArtifactDocument) -> ArtifactDocument:
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
    return validated


def _write_embedded_artifact_archive(
    handle: BinaryIO,
    *,
    manifest_bytes: bytes,
    source_index_bytes: bytes,
    source_blobs: tuple[tuple[str, BinaryIO, str, int], ...],
    checksums_bytes: bytes,
) -> None:
    """Stream verified, content-addressed source blobs into a ZIP64 archive."""

    with zipfile.ZipFile(handle, "w", allowZip64=True) as zf:
        zf.writestr(
            MANIFEST_NAME,
            manifest_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
        )
        zf.writestr(
            SOURCE_INDEX_NAME,
            source_index_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
        )

        for (
            source_member,
            source_stream,
            expected_source_sha256,
            expected_source_size_bytes,
        ) in source_blobs:
            source_info = zipfile.ZipInfo(source_member)
            source_info.compress_type = zipfile.ZIP_STORED
            source_info.file_size = expected_source_size_bytes
            digest = hashlib.sha256()
            copied = 0
            with zf.open(source_info, "w", force_zip64=True) as destination:
                while True:
                    chunk = source_stream.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > expected_source_size_bytes:
                        raise SourceChangedError(
                            "Source grew while it was being embedded in the project"
                        )
                    digest.update(chunk)
                    destination.write(chunk)
            if copied != expected_source_size_bytes:
                raise SourceChangedError(
                    "Source size changed while it was being embedded in the project"
                )
            if not hmac.compare_digest(digest.hexdigest(), expected_source_sha256):
                raise SourceChangedError(
                    "Source SHA-256 changed while it was being embedded in the project"
                )

        zf.writestr(
            CHECKSUMS_NAME,
            checksums_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
        )


def _path_matches_file_identity(path: Path, identity: tuple[int, int]) -> bool:
    """Return whether ``path`` currently resolves to a captured file object."""

    try:
        observed = path.stat()
    except FileNotFoundError:
        return False
    return (int(observed.st_dev), int(observed.st_ino)) == identity


@contextmanager
def _open_artifact_session_source(
    *,
    locator: str,
    source_member: str,
    expected_identity_scope: str | None,
    expected_sha256: str,
    expected_size_bytes: int,
) -> Iterator[tuple[BinaryIO, tuple[int, int] | None]]:
    """Open either an external source or a validated embedded source member.

    The archive and member descriptors remain open through staged package
    validation. The caller therefore can close them before atomically replacing
    the same archive on Windows.
    """

    embedded_namespace = "!/sources/blobs/sha256/"
    expected_suffix = f"!/{source_member}"
    if embedded_namespace in locator:
        if not locator.endswith(expected_suffix):
            raise ProjectFormatError(
                "embedded source locator does not match the ArtifactDocument source identity"
            )
        archive_text = locator[: -len(expected_suffix)]
        if not archive_text:
            raise ProjectFormatError("embedded source locator has no archive path")
        archive_path = Path(archive_text)
        package = load_artifact_project_package(archive_path)
        bundle = package.source_bundle
        if bundle is None:
            raise EmbeddedSourceRequiredError(
                "embedded source locator archive has no source bundle"
            )
        matches = [
            entry
            for entry in bundle.entries
            if entry.member == source_member
            and entry.sha256 == expected_sha256
            and entry.size_bytes == expected_size_bytes
            and entry.source_asset_id == f"sha256:{expected_sha256}"
        ]
        if not matches:
            raise ProjectFormatError(
                "embedded source locator archive does not contain the expected source blob"
            )

        _preflight_zip_directory(archive_path)
        with zipfile.ZipFile(archive_path, "r") as source_archive:
            with source_archive.open(source_member, "r") as source_stream:
                yield cast(BinaryIO, source_stream), None
        return

    with open_fingerprinted_file(locator) as (source_stream, source_fingerprint):
        if (
            (
                expected_identity_scope is not None
                and source_fingerprint.identity_scope != expected_identity_scope
            )
            or source_fingerprint.sha256 != expected_sha256
            or source_fingerprint.size_bytes != expected_size_bytes
        ):
            raise ProjectFormatError(
                "current external source bytes do not match the ArtifactDocument "
                "SourceAsset identity"
            )
        source_stat = os.fstat(source_stream.fileno())
        yield source_stream, (int(source_stat.st_dev), int(source_stat.st_ino))


def _visual_attributes_match(expected: object, observed: object) -> bool:
    """Verify that the source closure reproduces texture and UV attributes."""

    import numpy as np  # noqa: PLC0415

    for attribute in ("uv_coords", "texture"):
        expected_value = getattr(expected, attribute, None)
        observed_value = getattr(observed, attribute, None)
        if (expected_value is None) != (observed_value is None):
            return False
        if expected_value is not None and not np.array_equal(
            np.asarray(expected_value),
            np.asarray(observed_value),
            equal_nan=True,
        ):
            return False
    return True


def save_artifact_session_project(
    path: str | Path,
    session: ArtifactSession,
    *,
    meta: dict[str, Any] | None = None,
) -> str:
    """Atomically save an artifact session with its verified source closure.

    External and archive paths are runtime locators only. Every unique content
    descriptor is checked against ``SourceAsset`` or the import manifest and
    copied into a content-addressed ZIP_STORED member. The staged package is
    reopened through the production reader before the destination is replaced.
    """

    # Late import keeps the manifest-only storage API independent of trimesh
    # until callers explicitly request session packaging/materialisation.
    from .artifact_session import ArtifactSession  # noqa: PLC0415

    if not isinstance(session, ArtifactSession):
        raise ProjectSerializationError("session must be an ArtifactSession")
    if meta is not None and not isinstance(meta, dict):
        raise ProjectSerializationError("Project metadata must be a JSON object")

    out_path = Path(path)
    if out_path.suffix.lower() != ".amr":
        raise ProjectSaveError(
            "prepare",
            "embedded artifact sessions require an .amr ZIP destination",
            retryable=False,
        )

    try:
        expected_projection_snapshot = session.projection_snapshot()
    except ValueError as exc:
        raise ProjectSerializationError(
            f"Invalid artifact session source/document binding: {exc}"
        ) from exc

    validated = _validated_artifact_document(session.document)
    try:
        source_bundle = SourceBundleIndex.for_document(validated)
    except (SourceBundleError, ValueError) as exc:
        raise ProjectSerializationError(f"Invalid embedded source inventory: {exc}") from exc

    source_asset_id = session.verified_geometry.source_asset_id
    source_asset = validated.source_asset_index.get(source_asset_id)
    if source_asset is None:
        raise ProjectSerializationError(
            "session verified source asset is missing from its ArtifactDocument"
        )
    if len(validated.source_assets) != 1:
        raise ProjectSerializationError(
            "embedded session projects currently require exactly one source asset"
        )
    runtime_resources = {
        (resource.entry.logical_path, resource.entry.sha256): resource
        for resource in session.source_mesh.source_resources
    }
    source_plans_by_member: dict[
        str,
        tuple[str, str, int, str | None],
    ] = {}
    for entry in source_bundle.entries:
        if entry.size_bytes > MAX_SOURCE_MEMBER_BYTES:
            raise ProjectSerializationError(
                f"source exceeds the {MAX_SOURCE_MEMBER_BYTES}-byte embedded-source limit"
            )
        if (
            entry.role == PRIMARY_SOURCE_ASSET_ROLE
            and entry.source_asset_id == source_asset.id
        ):
            locator = session.resolved_source_path
            identity_scope: str | None = source_asset.identity_scope
        else:
            resource = runtime_resources.get((entry.logical_path, entry.sha256))
            if resource is None:
                raise ProjectSerializationError(
                    "session has no verified runtime locator for source dependency "
                    f"{entry.logical_path!r}"
                )
            expected_entry = SourceManifestEntry(
                logical_path=entry.logical_path,
                media_type=entry.media_type,
                role=entry.role,
                sha256=entry.sha256,
                size_bytes=entry.size_bytes,
            )
            if resource.entry != expected_entry:
                raise ProjectSerializationError(
                    "runtime source dependency does not match the durable source manifest"
                )
            locator = resource.locator
            identity_scope = None
        existing = source_plans_by_member.get(entry.member)
        plan = (locator, entry.sha256, entry.size_bytes, identity_scope)
        if existing is None or identity_scope is not None:
            source_plans_by_member[entry.member] = plan
        elif existing[1:3] != plan[1:3]:
            raise ProjectSerializationError(
                "content-addressed source member has conflicting identities"
            )
    if sum(plan[2] for plan in source_plans_by_member.values()) > MAX_TOTAL_SOURCE_BYTES:
        raise ProjectSerializationError(
            "embedded source closure exceeds the total source-byte safety limit"
        )

    envelope: dict[str, Any] = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "payload_type": ARTIFACT_PAYLOAD_TYPE,
        "payload_schema_version": ARTIFACT_PAYLOAD_SCHEMA_VERSION,
        "saved_at": _utc_now_iso(),
        "meta": dict(meta or {}),
        "state": validated.to_dict(),
    }
    manifest_bytes = _strict_json_dumps(envelope, label=MANIFEST_NAME)
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ProjectSerializationError(
            f"{MANIFEST_NAME} exceeds the {MAX_MANIFEST_BYTES}-byte safety limit"
        )
    source_index_bytes = source_bundle.canonical_json_bytes()
    if len(source_index_bytes) > MAX_SOURCE_INDEX_BYTES:
        raise ProjectSerializationError(
            f"{SOURCE_INDEX_NAME} exceeds the {MAX_SOURCE_INDEX_BYTES}-byte safety limit"
        )
    checksum_files = {
        MANIFEST_NAME: hashlib.sha256(manifest_bytes).hexdigest(),
        SOURCE_INDEX_NAME: hashlib.sha256(source_index_bytes).hexdigest(),
        **{
            member: plan[1]
            for member, plan in sorted(source_plans_by_member.items())
        },
    }
    checksums_bytes = _strict_json_dumps(
        {"algorithm": "sha256", "files": checksum_files},
        label=CHECKSUMS_NAME,
    )
    if len(checksums_bytes) > MAX_CHECKSUMS_BYTES:
        raise ProjectSerializationError(
            f"{CHECKSUMS_NAME} exceeds the {MAX_CHECKSUMS_BYTES}-byte safety limit"
        )

    temp_path: Path | None = None
    raw_fd: int | None = None
    committed = False
    external_source_identities: list[tuple[int, int]] = []
    stage = "source_open"
    try:
        stage = "source_verification"
        with ExitStack() as source_stack:
            opened_blobs: list[tuple[str, BinaryIO, str, int]] = []
            for member, plan in sorted(source_plans_by_member.items()):
                locator, expected_sha256, expected_size_bytes, identity_scope = plan
                source_stream, external_identity = source_stack.enter_context(
                    _open_artifact_session_source(
                        locator=locator,
                        source_member=member,
                        expected_identity_scope=identity_scope,
                        expected_sha256=expected_sha256,
                        expected_size_bytes=expected_size_bytes,
                    )
                )
                opened_blobs.append(
                    (member, source_stream, expected_sha256, expected_size_bytes)
                )
                if external_identity is not None:
                    external_source_identities.append(external_identity)
            if any(
                _path_matches_file_identity(out_path, identity)
                for identity in external_source_identities
            ):
                raise ProjectSaveError(
                    "prepare",
                    "project destination resolves to an external source resource",
                    retryable=False,
                )

            stage = "prepare"
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
                _write_embedded_artifact_archive(
                    handle,
                    manifest_bytes=manifest_bytes,
                    source_index_bytes=source_index_bytes,
                    source_blobs=tuple(opened_blobs),
                    checksums_bytes=checksums_bytes,
                )
                stage = "temp_fsync"
                _flush_and_fsync(handle)

            stage = "validation"
            package = load_artifact_project_package(temp_path)
            if package.document.canonical_json_bytes() != validated.canonical_json_bytes():
                raise ProjectFormatError(
                    "Staged package validation changed the ArtifactDocument"
                )
            if package.source_bundle != source_bundle:
                raise ProjectFormatError(
                    "Staged package validation changed the source bundle index"
                )
            staged_session = _materialize_artifact_project_package(package)
            if staged_session.projection_snapshot() != expected_projection_snapshot:
                raise ProjectFormatError(
                    "Staged embedded source did not reproduce the saved artifact projection"
                )
            if not _visual_attributes_match(
                session.source_mesh,
                staged_session.source_mesh,
            ):
                raise ProjectFormatError(
                    "Staged embedded source did not reproduce source UV/texture attributes; "
                    "sidecar source closure is incomplete"
                )

            # Keep this label active while open_fingerprinted_file performs its
            # final descriptor/path checks on context-manager exit.
            stage = "source_verification"

        stage = "replace"
        assert temp_path is not None
        if any(
            _path_matches_file_identity(out_path, identity)
            for identity in external_source_identities
        ):
            raise ProjectSaveError(
                "replace",
                "project destination changed to resolve to an external source resource",
                retryable=False,
            )
        _commit_staged_project(temp_path, out_path)
        committed = True
        return str(out_path)
    except ProjectSerializationError:
        raise
    except ProjectSaveError:
        raise
    except Exception as exc:
        raise ProjectSaveError(
            stage,
            f"Embedded artifact project save failed during {stage}: {exc}",
            retryable=stage
            in {
                "source_open",
                "source_verification",
                "prepare",
                "temp_write",
                "temp_fsync",
                "validation",
                "replace",
            },
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


def _artifact_from_envelope(envelope: dict[str, Any]) -> ArtifactDocument:
    state = envelope.get("state")
    if not isinstance(state, dict):
        raise ProjectFormatError("Invalid ArtifactDocument payload state")
    try:
        artifact = ArtifactDocument.from_dict(state)
        validate_known_records(artifact)
        return artifact
    except (
        ArtifactDocumentError,
        ArtifactVectorRecordError,
        ArtifactKnownRecordError,
    ) as exc:
        raise ProjectFormatError(f"Invalid ArtifactDocument payload: {exc}") from exc


def _load_artifact_project_package_zip(path: Path) -> ArtifactProjectPackage:
    """Validate one artifact archive and its optional embedded source index."""

    _preflight_zip_directory(path)
    try:
        with zipfile.ZipFile(path, "r") as zf:
            members = _validate_zip_infos(zf.infolist())
            manifest_bytes = _read_zip_member(zf, MANIFEST_NAME)
            raw_envelope = _strict_json_loads(manifest_bytes, label=MANIFEST_NAME)
            if not isinstance(raw_envelope, dict):
                raise ProjectFormatError(
                    "Invalid project document (expected JSON object)"
                )
            version = _require_envelope_identity(raw_envelope)
            _reject_runtime_marker_from_durable_document(raw_envelope)
            if version != PROJECT_VERSION:
                raise UnsupportedPayloadError(
                    raw_envelope.get("payload_type"),
                    raw_envelope.get("payload_schema_version"),
                    inspection=_inspection_from_document(raw_envelope),
                )

            checksums = _validate_v2_checksums(zf, members)
            envelope = _normalize_loaded_document(
                raw_envelope,
                expected_payload_type=ARTIFACT_PAYLOAD_TYPE,
                expected_schema_version=ARTIFACT_PAYLOAD_SCHEMA_VERSION,
            )
            artifact = _artifact_from_envelope(envelope)

            source_member_names = {
                name
                for name in members
                if _source_digest_from_member(name) is not None
            }
            if SOURCE_INDEX_NAME not in members:
                if source_member_names:
                    raise ProjectFormatError(
                        "Embedded source blob exists without sources/index.json"
                    )
                return ArtifactProjectPackage(
                    document=artifact,
                    archive_path=path,
                    source_bundle=None,
                )

            source_index_bytes = _read_zip_member(zf, SOURCE_INDEX_NAME)
            source_index_data = _strict_json_loads(
                source_index_bytes,
                label=SOURCE_INDEX_NAME,
            )
            if not isinstance(source_index_data, dict):
                raise ProjectFormatError(
                    f"Invalid {SOURCE_INDEX_NAME}: expected JSON object"
                )
            try:
                source_bundle = SourceBundleIndex.from_dict(source_index_data)
                expected_bundle = SourceBundleIndex.for_document(artifact)
            except (SourceBundleError, ValueError) as exc:
                raise ProjectFormatError(
                    f"Invalid embedded source bundle: {exc}"
                ) from exc
            if source_bundle.canonical_json_bytes() != source_index_bytes:
                raise ProjectFormatError(
                    f"{SOURCE_INDEX_NAME} is not canonical JSON"
                )
            if source_bundle != expected_bundle:
                raise ProjectFormatError(
                    "Embedded source index does not match the ArtifactDocument snapshot"
                )

            indexed_members = {entry.member for entry in source_bundle.entries}
            missing_members = sorted(indexed_members - source_member_names)
            orphan_members = sorted(source_member_names - indexed_members)
            if missing_members:
                raise ProjectFormatError(
                    "Embedded source index references missing blob members: "
                    + ", ".join(missing_members)
                )
            if orphan_members:
                raise ProjectFormatError(
                    "Embedded source archive contains orphan blob members: "
                    + ", ".join(orphan_members)
                )

            for entry in source_bundle.entries:
                info = members[entry.member]
                if info.file_size != entry.size_bytes:
                    raise ProjectFormatError(
                        f"Embedded source size does not match index: {entry.member!r}"
                    )
                checksum = checksums.get(entry.member)
                if checksum is None or not hmac.compare_digest(
                    checksum,
                    entry.sha256,
                ):
                    raise ProjectFormatError(
                        f"Embedded source SHA-256 does not match index: {entry.member!r}"
                    )

            return ArtifactProjectPackage(
                document=artifact,
                archive_path=path,
                source_bundle=source_bundle,
            )
    except (UnsupportedProjectVersionError, UnsupportedPayloadError):
        raise
    except ProjectFormatError:
        raise
    except (zipfile.BadZipFile, OSError, EOFError) as exc:
        raise ProjectFormatError(f"Invalid AMR ZIP container: {exc}") from exc


def load_artifact_project_package(path: str | Path) -> ArtifactProjectPackage:
    """Load an artifact document and validate any embedded source bundle."""

    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))
    if in_path.suffix.lower() == ".json":
        envelope = _load_json_document(
            in_path,
            expected_payload_type=ARTIFACT_PAYLOAD_TYPE,
            expected_schema_version=ARTIFACT_PAYLOAD_SCHEMA_VERSION,
        )
        return ArtifactProjectPackage(
            document=_artifact_from_envelope(envelope),
            archive_path=in_path,
            source_bundle=None,
        )
    return _load_artifact_project_package_zip(in_path)


def load_artifact_project(path: str | Path) -> ArtifactDocument:
    """Load and validate an authoritative ``artifact_document`` AMR v2 payload.

    Legacy UI payloads are deliberately not migrated here. Callers must choose
    the legacy or artifact API explicitly so no unit, geometry, Align, or record
    identity is ever invented during storage loading.
    """

    return load_artifact_project_package(path).document


def _materialize_artifact_project_package(
    package: ArtifactProjectPackage,
) -> ArtifactSession:
    """Parse and bind an embedded source closure from a validated package."""

    from .artifact_session import (  # noqa: PLC0415
        ArtifactSession,
        ArtifactSessionError,
    )
    from .mesh_loader import MeshLoader  # noqa: PLC0415

    if not isinstance(package, ArtifactProjectPackage):
        raise TypeError("package must be an ArtifactProjectPackage")
    source_bundle = package.source_bundle
    if source_bundle is None:
        raise EmbeddedSourceRequiredError(
            "Artifact project has no embedded source bundle; session materialization "
            "requires sources/index.json and its primary blob"
        )
    primary_entries = [
        entry
        for entry in source_bundle.entries
        if entry.source_asset_id == source_bundle.primary_source_asset_id
        and entry.role == PRIMARY_SOURCE_ASSET_ROLE
    ]
    if len(primary_entries) != 1:
        # SourceBundleIndex already enforces this; retain a defensive public
        # boundary if the model evolves independently.
        raise ProjectFormatError(
            "Embedded source bundle does not identify exactly one primary source"
        )
    entry = primary_entries[0]

    document = package.document
    active_metadata_id = document.active_source_metadata_revision_id
    if active_metadata_id is None:
        raise ProjectFormatError(
            "ArtifactDocument has no active source metadata revision"
        )
    metadata = document.source_metadata_revision_index[active_metadata_id]
    geometry = document.geometry_revision_index[metadata.geometry_revision_id]
    from .mesh_import_recipe import (  # noqa: PLC0415
        MeshImportRecipeError,
        RUNTIME_POLICY_RECORD_ONLY,
        validate_mesh_import_recipe,
    )

    try:
        import_execution = validate_mesh_import_recipe(
            geometry.import_recipe,
            allow_legacy=True,
            runtime_policy=RUNTIME_POLICY_RECORD_ONLY,
        )
    except MeshImportRecipeError as exc:
        raise ProjectFormatError(
            f"ArtifactDocument geometry import recipe is not executable: {exc}"
        ) from exc
    parser_format = import_execution.source_format

    try:
        # The archive path may have changed after package validation. Reapply
        # the central-directory bound before the second, stream-only open; the
        # mesh loader independently re-verifies the expected blob hash/size.
        _preflight_zip_directory(package.archive_path)
        with zipfile.ZipFile(package.archive_path, "r") as zf:
            def dependency_loader(
                dependency: SourceManifestEntry,
            ) -> tuple[bytes, str]:
                matches = [
                    candidate
                    for candidate in source_bundle.entries
                    if candidate.logical_path == dependency.logical_path
                    and candidate.sha256 == dependency.sha256
                    and candidate.size_bytes == dependency.size_bytes
                    and candidate.role == dependency.role
                ]
                if len(matches) != 1:
                    raise ProjectFormatError(
                        "embedded source bundle does not contain exactly one "
                        f"manifest dependency {dependency.logical_path!r}"
                    )
                candidate = matches[0]
                with zf.open(candidate.member, "r") as dependency_stream:
                    payload = dependency_stream.read()
                return (
                    payload,
                    f"{package.archive_path}!/{candidate.member}",
                )

            with zf.open(entry.member, "r") as source_stream:
                source_mesh = MeshLoader(default_unit="mm").load_verified_stream(
                    cast(BinaryIO, source_stream),
                    unit=metadata.unit,
                    source_format=parser_format,
                    expected_sha256=entry.sha256,
                    expected_size_bytes=entry.size_bytes,
                    original_name=entry.logical_path,
                    import_recipe=geometry.import_recipe,
                    dependency_loader=(
                        dependency_loader
                        if import_execution.source_manifest is not None
                        else None
                    ),
                    primary_locator=f"{package.archive_path}!/{entry.member}",
                )
        resource_index = {
            (resource.entry.logical_path, resource.entry.sha256): resource
            for resource in source_mesh.source_resources
        }
        for bundled in source_bundle.entries:
            resource_entry = SourceManifestEntry(
                logical_path=bundled.logical_path,
                media_type=bundled.media_type,
                role=bundled.role,
                sha256=bundled.sha256,
                size_bytes=bundled.size_bytes,
            )
            resource_index.setdefault(
                (resource_entry.logical_path, resource_entry.sha256),
                ResolvedSourceResource(
                    entry=resource_entry,
                    locator=f"{package.archive_path}!/{bundled.member}",
                ),
            )
        source_mesh.source_resources = tuple(
            resource_index[key] for key in sorted(resource_index)
        )
        session = ArtifactSession.bind_loaded_document(
            document,
            source_mesh,
            resolved_source_path=f"{package.archive_path}!/{entry.member}",
        )
        # A successful load means the exact source can reproduce the active
        # unit/Align projection, not merely that its container checksums pass.
        session.materialize()
        return session
    except ProjectFormatError:
        raise
    except (
        ArtifactSessionError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        raise ProjectFormatError(
            f"Unable to materialize embedded artifact session: {exc}"
        ) from exc


def load_artifact_session_project(path: str | Path) -> ArtifactSession:
    """Reconstruct and materialize a session solely from an embedded AMR."""

    return _materialize_artifact_project_package(
        load_artifact_project_package(path)
    )
