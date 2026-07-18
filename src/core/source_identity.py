"""Content identity and verification for external source files.

The authoritative identity is the SHA-256 digest plus byte length.  File names,
formats, paths, and modification times are deliberately treated as hints so a
byte-identical source can be relocated without becoming a different artifact.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any, BinaryIO, Iterator, Mapping


DEFAULT_HASH_CHUNK_SIZE = 4 * 1024 * 1024
EXTERNAL_FILE_KIND = "external_file"
PRIMARY_FILE_IDENTITY_SCOPE = "primary_file_bytes"
SOURCE_FINGERPRINT_SCHEMA_VERSION = 1
SOURCE_VERIFICATION_SCHEMA_VERSION = 1

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class SourceIdentityError(RuntimeError):
    """Base error for source identity operations."""


class SourceChangedError(SourceIdentityError):
    """Raised when a source changes while its fingerprint is being calculated."""


class SourceSizeLimitError(SourceIdentityError):
    """Raised before hashing when the opened descriptor exceeds a caller limit."""


class SourceVerificationStatus(str, Enum):
    """Typed outcome of checking an external source against a fingerprint."""

    VERIFIED = "verified"
    MISSING = "missing"
    MISMATCH = "mismatch"
    UNREADABLE = "unreadable"
    LEGACY_UNVERIFIED = "legacy_unverified"


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_string(value: object, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _require_schema_version(
    data: Mapping[str, object],
    *,
    expected: int,
    model_name: str,
) -> None:
    observed = data.get("schema_version")
    if isinstance(observed, bool) or not isinstance(observed, int) or observed != expected:
        raise ValueError(f"unsupported {model_name} schema version: {observed!r}")


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    """Stable identity and non-authoritative file hints for one primary file."""

    sha256: str
    size_bytes: int
    mtime_ns: int
    original_name: str
    format: str

    def __post_init__(self) -> None:
        digest = _require_string(self.sha256, "sha256").strip().lower()
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(
            self,
            "size_bytes",
            _require_non_negative_int(self.size_bytes, "size_bytes"),
        )
        object.__setattr__(
            self,
            "mtime_ns",
            _require_int(self.mtime_ns, "mtime_ns"),
        )
        object.__setattr__(
            self,
            "original_name",
            _require_string(self.original_name, "original_name"),
        )

        source_format = _require_string(self.format, "format", allow_empty=True)
        object.__setattr__(self, "format", source_format.removeprefix(".").lower())

    @property
    def id(self) -> str:
        return f"sha256:{self.sha256}"

    @property
    def kind(self) -> str:
        return EXTERNAL_FILE_KIND

    @property
    def identity_scope(self) -> str:
        return PRIMARY_FILE_IDENTITY_SCOPE

    @property
    def filename(self) -> str:
        """Compatibility-friendly name for display code."""
        return self.original_name

    @property
    def extension(self) -> str:
        """Normalized extension including the leading dot, or an empty string."""
        return f".{self.format}" if self.format else ""

    def content_matches(self, other: "SourceFingerprint") -> bool:
        return self.sha256 == other.sha256 and self.size_bytes == other.size_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
            "id": self.id,
            "kind": self.kind,
            "identity_scope": self.identity_scope,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "original_name": self.original_name,
            "format": self.format,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SourceFingerprint":
        if not isinstance(data, Mapping):
            raise ValueError("source fingerprint must be a mapping")
        _require_schema_version(
            data,
            expected=SOURCE_FINGERPRINT_SCHEMA_VERSION,
            model_name="source fingerprint",
        )

        fingerprint = cls(
            sha256=_require_string(data.get("sha256"), "sha256"),
            size_bytes=_require_non_negative_int(data.get("size_bytes"), "size_bytes"),
            mtime_ns=_require_int(data.get("mtime_ns"), "mtime_ns"),
            original_name=_require_string(data.get("original_name"), "original_name"),
            format=_require_string(data.get("format"), "format", allow_empty=True),
        )

        serialized_id = _require_string(data.get("id"), "id")
        if serialized_id != fingerprint.id:
            raise ValueError("source fingerprint id does not match sha256")
        kind = _require_string(data.get("kind"), "kind")
        if kind != EXTERNAL_FILE_KIND:
            raise ValueError(f"unsupported source kind: {kind!r}")
        scope = _require_string(data.get("identity_scope"), "identity_scope")
        if scope != PRIMARY_FILE_IDENTITY_SCOPE:
            raise ValueError(f"unsupported identity scope: {scope!r}")
        return fingerprint


@dataclass(frozen=True, slots=True)
class _StatSnapshot:
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int


def _time_ns(stat_result: os.stat_result, name: str) -> int:
    ns_value = getattr(stat_result, f"st_{name}_ns", None)
    if ns_value is not None:
        return int(ns_value)
    return int(float(getattr(stat_result, f"st_{name}")) * 1_000_000_000)


def _stat_snapshot(stat_result: os.stat_result) -> _StatSnapshot:
    return _StatSnapshot(
        device=int(stat_result.st_dev),
        inode=int(stat_result.st_ino),
        size_bytes=int(stat_result.st_size),
        mtime_ns=_time_ns(stat_result, "mtime"),
        ctime_ns=_time_ns(stat_result, "ctime"),
    )


def _raise_if_changed(
    source_path: Path,
    expected: _StatSnapshot,
    observed: _StatSnapshot,
    *,
    compare_ctime: bool = True,
) -> None:
    if (
        expected.device != observed.device
        or expected.inode != observed.inode
        or expected.size_bytes != observed.size_bytes
        or expected.mtime_ns != observed.mtime_ns
        or (compare_ctime and expected.ctime_ns != observed.ctime_ns)
    ):
        raise SourceChangedError(
            f"Source changed while calculating its fingerprint: {source_path}"
        )


def _path_descriptor_ctime_comparable() -> bool:
    """Whether path ``stat`` and descriptor ``fstat`` expose the same ctime.

    CPython on Windows reports creation time for ``Path.stat().st_ctime`` but
    the file change time for ``os.fstat().st_ctime``. Descriptor-to-descriptor
    comparisons remain meaningful there, so only mixed comparisons omit ctime.
    """

    return os.name != "nt"


@contextmanager
def open_fingerprinted_file(
    path: str | os.PathLike[str],
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
    max_size_bytes: int | None = None,
) -> Iterator[tuple[BinaryIO, SourceFingerprint]]:
    """Yield one rewound descriptor and the identity computed from its bytes.

    Consumers such as mesh parsers must read from this descriptor rather than
    reopening ``path``. This prevents a same-size/same-mtime path replacement
    from pairing one file's hash with another file's parsed geometry. The path
    and descriptor are checked both before yielding and after the consumer
    returns.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if max_size_bytes is not None and (
        isinstance(max_size_bytes, bool)
        or not isinstance(max_size_bytes, int)
        or max_size_bytes < 1
    ):
        raise ValueError("max_size_bytes must be a positive integer or None")

    source_path = Path(path)
    before_path = _stat_snapshot(source_path.stat())
    digest = hashlib.sha256()
    bytes_read = 0

    with source_path.open("rb") as stream:
        opened_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened_stat.st_mode):
            raise SourceIdentityError(
                f"Source is not an opened regular file: {source_path}"
            )
        opened_file = _stat_snapshot(opened_stat)
        _raise_if_changed(
            source_path,
            before_path,
            opened_file,
            compare_ctime=_path_descriptor_ctime_comparable(),
        )
        if (
            max_size_bytes is not None
            and opened_file.size_bytes > max_size_bytes
        ):
            raise SourceSizeLimitError(
                f"Source exceeds the {max_size_bytes:,}-byte safety limit: "
                f"{opened_file.size_bytes:,} bytes"
            )

        while True:
            read_size = chunk_size
            if max_size_bytes is not None:
                # Once the opened file reaches the fixed boundary, read at
                # most one sentinel byte.  A source that grows while it is
                # being hashed must not turn the pre-hash size check into an
                # unbounded read.
                read_size = min(
                    chunk_size,
                    max_size_bytes - bytes_read + 1,
                )
            chunk = stream.read(read_size)
            if not chunk:
                break
            bytes_read += len(chunk)
            if max_size_bytes is not None and bytes_read > max_size_bytes:
                raise SourceSizeLimitError(
                    f"Source grew beyond the {max_size_bytes:,}-byte safety "
                    f"limit while hashing: {source_path}"
                )
            digest.update(chunk)

        after_hash_file = _stat_snapshot(os.fstat(stream.fileno()))
        _raise_if_changed(source_path, opened_file, after_hash_file)
        try:
            after_hash_path = _stat_snapshot(source_path.stat())
        except FileNotFoundError as exc:
            raise SourceChangedError(
                f"Source disappeared while calculating its fingerprint: {source_path}"
            ) from exc
        _raise_if_changed(
            source_path,
            after_hash_file,
            after_hash_path,
            compare_ctime=_path_descriptor_ctime_comparable(),
        )

        if bytes_read != after_hash_file.size_bytes:
            raise SourceChangedError(
                f"Source size changed while calculating its fingerprint: {source_path}"
            )

        fingerprint = SourceFingerprint(
            sha256=digest.hexdigest(),
            size_bytes=bytes_read,
            mtime_ns=after_hash_file.mtime_ns,
            original_name=source_path.name,
            format=source_path.suffix.removeprefix(".").lower(),
        )
        stream.seek(0)
        yield stream, fingerprint

        after_consumer_file = _stat_snapshot(os.fstat(stream.fileno()))
        _raise_if_changed(source_path, after_hash_file, after_consumer_file)
        try:
            after_consumer_path = _stat_snapshot(source_path.stat())
        except FileNotFoundError as exc:
            raise SourceChangedError(
                f"Source disappeared while it was being consumed: {source_path}"
            ) from exc
        _raise_if_changed(
            source_path,
            after_consumer_file,
            after_consumer_path,
            compare_ctime=_path_descriptor_ctime_comparable(),
        )


def fingerprint_file(
    path: str | os.PathLike[str],
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> SourceFingerprint:
    """Hash a file incrementally and fail if it changes during the read."""
    with open_fingerprinted_file(path, chunk_size=chunk_size) as (_stream, fingerprint):
        return fingerprint


@dataclass(frozen=True, slots=True)
class SourceVerification:
    """Serializable outcome of source verification."""

    status: SourceVerificationStatus
    checked_path: str
    expected: SourceFingerprint | None = None
    actual: SourceFingerprint | None = None
    mismatch_fields: tuple[str, ...] = ()
    hint_differences: tuple[str, ...] = ()
    relocated: bool = False
    detail: str | None = None

    def __post_init__(self) -> None:
        try:
            status = SourceVerificationStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported source verification status: {self.status!r}") from exc
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "checked_path",
            _require_string(self.checked_path, "checked_path", allow_empty=True),
        )
        mismatch_fields = tuple(self.mismatch_fields)
        hint_differences = tuple(self.hint_differences)
        if not all(isinstance(item, str) for item in mismatch_fields):
            raise ValueError("mismatch_fields must contain only strings")
        if not all(isinstance(item, str) for item in hint_differences):
            raise ValueError("hint_differences must contain only strings")
        object.__setattr__(self, "mismatch_fields", mismatch_fields)
        object.__setattr__(self, "hint_differences", hint_differences)
        if not isinstance(self.relocated, bool):
            raise ValueError("relocated must be a boolean")
        if self.expected is not None and not isinstance(self.expected, SourceFingerprint):
            raise ValueError("expected must be a SourceFingerprint or null")
        if self.actual is not None and not isinstance(self.actual, SourceFingerprint):
            raise ValueError("actual must be a SourceFingerprint or null")
        if self.detail is not None and not isinstance(self.detail, str):
            raise ValueError("detail must be a string or null")

        if status in {SourceVerificationStatus.VERIFIED, SourceVerificationStatus.MISMATCH}:
            if self.expected is None or self.actual is None:
                raise ValueError(f"{status.value} verification requires expected and actual fingerprints")
        elif status is SourceVerificationStatus.LEGACY_UNVERIFIED:
            if self.expected is not None or self.actual is not None:
                raise ValueError("legacy_unverified verification cannot contain fingerprints")

        if status is SourceVerificationStatus.VERIFIED and self.mismatch_fields:
            raise ValueError("verified source cannot contain mismatch_fields")
        if status is SourceVerificationStatus.MISMATCH and not self.mismatch_fields:
            raise ValueError("mismatch source must identify mismatch_fields")
        if status in {
            SourceVerificationStatus.MISSING,
            SourceVerificationStatus.UNREADABLE,
            SourceVerificationStatus.LEGACY_UNVERIFIED,
        }:
            if self.actual is not None:
                raise ValueError(f"{status.value} verification cannot contain an actual fingerprint")
            if self.mismatch_fields or self.hint_differences or self.relocated:
                raise ValueError(f"{status.value} verification cannot contain comparison results")

        authoritative_fields = {"sha256", "size_bytes"}
        if any(field not in authoritative_fields for field in self.mismatch_fields):
            raise ValueError("mismatch_fields contains a non-authoritative identity field")
        hint_fields = {"mtime_ns", "original_name", "format"}
        if any(field not in hint_fields for field in self.hint_differences):
            raise ValueError("hint_differences contains an unsupported hint field")

    @property
    def verified(self) -> bool:
        return self.status is SourceVerificationStatus.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOURCE_VERIFICATION_SCHEMA_VERSION,
            "status": self.status.value,
            "checked_path": self.checked_path,
            "expected": self.expected.to_dict() if self.expected is not None else None,
            "actual": self.actual.to_dict() if self.actual is not None else None,
            "mismatch_fields": list(self.mismatch_fields),
            "hint_differences": list(self.hint_differences),
            "relocated": self.relocated,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SourceVerification":
        if not isinstance(data, Mapping):
            raise ValueError("source verification must be a mapping")
        _require_schema_version(
            data,
            expected=SOURCE_VERIFICATION_SCHEMA_VERSION,
            model_name="source verification",
        )

        raw_expected = data.get("expected")
        raw_actual = data.get("actual")
        if raw_expected is not None and not isinstance(raw_expected, Mapping):
            raise ValueError("expected must be a source fingerprint mapping or null")
        if raw_actual is not None and not isinstance(raw_actual, Mapping):
            raise ValueError("actual must be a source fingerprint mapping or null")

        raw_mismatch_fields = data.get("mismatch_fields", [])
        raw_hint_differences = data.get("hint_differences", [])
        if not isinstance(raw_mismatch_fields, list) or not all(
            isinstance(item, str) for item in raw_mismatch_fields
        ):
            raise ValueError("mismatch_fields must be a list of strings")
        if not isinstance(raw_hint_differences, list) or not all(
            isinstance(item, str) for item in raw_hint_differences
        ):
            raise ValueError("hint_differences must be a list of strings")

        raw_detail = data.get("detail")
        if raw_detail is not None and not isinstance(raw_detail, str):
            raise ValueError("detail must be a string or null")
        raw_relocated = data.get("relocated", False)
        if not isinstance(raw_relocated, bool):
            raise ValueError("relocated must be a boolean")

        return cls(
            status=SourceVerificationStatus(
                _require_string(data.get("status"), "status")
            ),
            checked_path=_require_string(
                data.get("checked_path", ""), "checked_path", allow_empty=True
            ),
            expected=SourceFingerprint.from_dict(raw_expected) if raw_expected is not None else None,
            actual=SourceFingerprint.from_dict(raw_actual) if raw_actual is not None else None,
            mismatch_fields=tuple(raw_mismatch_fields),
            hint_differences=tuple(raw_hint_differences),
            relocated=raw_relocated,
            detail=raw_detail,
        )

    @classmethod
    def missing(
        cls,
        checked_path: str,
        *,
        expected: SourceFingerprint | None = None,
        detail: str = "source file does not exist",
    ) -> "SourceVerification":
        return cls(
            status=SourceVerificationStatus.MISSING,
            checked_path=checked_path,
            expected=expected,
            detail=detail,
        )

    @classmethod
    def legacy_unverified(
        cls,
        checked_path: str = "",
        *,
        detail: str = "legacy project has no source fingerprint",
    ) -> "SourceVerification":
        return cls(
            status=SourceVerificationStatus.LEGACY_UNVERIFIED,
            checked_path=checked_path,
            detail=detail,
        )


def compare_fingerprints(
    expected: SourceFingerprint,
    actual: SourceFingerprint,
    *,
    checked_path: str = "",
    relocated: bool = False,
) -> SourceVerification:
    """Compare already-computed identities without reading the source again."""
    mismatch_fields = tuple(
        field_name
        for field_name in ("sha256", "size_bytes")
        if getattr(expected, field_name) != getattr(actual, field_name)
    )
    hint_differences = tuple(
        field_name
        for field_name in ("mtime_ns", "original_name", "format")
        if getattr(expected, field_name) != getattr(actual, field_name)
    )

    if mismatch_fields:
        return SourceVerification(
            status=SourceVerificationStatus.MISMATCH,
            checked_path=checked_path,
            expected=expected,
            actual=actual,
            mismatch_fields=mismatch_fields,
            hint_differences=hint_differences,
            relocated=relocated,
            detail="source bytes do not match the expected identity",
        )
    return SourceVerification(
        status=SourceVerificationStatus.VERIFIED,
        checked_path=checked_path,
        expected=expected,
        actual=actual,
        hint_differences=hint_differences,
        relocated=relocated,
        detail=(
            "byte-identical source found at a different path" if relocated else None
        ),
    )


def verify_source(
    path: str | os.PathLike[str],
    expected: SourceFingerprint,
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
    expected_path_hint: str | os.PathLike[str] | None = None,
) -> SourceVerification:
    """Fingerprint a candidate path and return a typed verification result."""
    source_path = Path(path)
    checked_path = str(source_path)
    try:
        actual = fingerprint_file(source_path, chunk_size=chunk_size)
    except FileNotFoundError:
        return SourceVerification.missing(checked_path, expected=expected)
    except (SourceChangedError, OSError) as exc:
        return SourceVerification(
            status=SourceVerificationStatus.UNREADABLE,
            checked_path=checked_path,
            expected=expected,
            detail=f"{type(exc).__name__}: {exc}",
        )
    relocated = False
    if expected_path_hint is not None:
        expected_path = os.path.normcase(
            os.path.abspath(os.path.normpath(os.fspath(expected_path_hint)))
        )
        actual_path = os.path.normcase(os.path.abspath(os.path.normpath(checked_path)))
        relocated = expected_path != actual_path
    return compare_fingerprints(
        expected,
        actual,
        checked_path=checked_path,
        relocated=relocated,
    )


def legacy_unverified_source(
    checked_path: str = "",
    *,
    detail: str = "legacy project has no source fingerprint",
) -> SourceVerification:
    """Create an explicit non-verified result for a v1/legacy source."""
    return SourceVerification.legacy_unverified(checked_path, detail=detail)


def missing_source(
    checked_path: str,
    *,
    expected: SourceFingerprint | None = None,
    detail: str = "source file does not exist",
) -> SourceVerification:
    """Create a typed missing result, including for legacy sources without identity."""
    return SourceVerification.missing(
        checked_path,
        expected=expected,
        detail=detail,
    )
