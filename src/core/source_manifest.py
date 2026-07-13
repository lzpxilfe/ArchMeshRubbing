"""Closed, portable inventory of every byte stream used by a mesh parser.

The primary mesh and its material, texture, and buffer sidecars form one
scientific parser input.  Host paths are runtime locators only; the durable
contract records normalized logical paths, byte lengths, media types, and
SHA-256 digests.  A parser resolver may read exactly these entries and nothing
else.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from .canonical_json import canonical_json_bytes


SOURCE_MANIFEST_FORMAT = "archmeshrubbing_source_manifest"
SOURCE_MANIFEST_SCHEMA_VERSION = "1.0.0"
SOURCE_RESOLVER_PROFILE = "relative-contained-v1"
PRIMARY_RESOURCE_ROLE = "primary_mesh"
DEPENDENCY_RESOURCE_ROLE = "import_dependency"
MAX_SOURCE_MANIFEST_ENTRIES = 61

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_ENTRY_KEYS = {
    "logical_path",
    "media_type",
    "role",
    "sha256",
    "size_bytes",
}
_MANIFEST_KEYS = {
    "entries",
    "format",
    "primary_logical_path",
    "resolver_profile",
    "schema_version",
}


class SourceManifestError(ValueError):
    """A source closure or logical reference violates the portable contract."""


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceManifestError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise SourceManifestError(f"{field_name} must not contain surrounding whitespace")
    return value


def _sha256(value: object, *, field_name: str) -> str:
    digest = _required_text(value, field_name=field_name)
    if _SHA256_RE.fullmatch(digest) is None:
        raise SourceManifestError(
            f"{field_name} must be 64 lowercase hexadecimal characters"
        )
    return digest


def _size_bytes(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceManifestError(f"{field_name} must be a non-negative integer")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    model_name: str,
) -> None:
    raw = set(value)
    if not all(isinstance(key, str) for key in raw):
        raise SourceManifestError(f"{model_name} field names must be strings")
    observed = {key for key in raw if isinstance(key, str)}
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise SourceManifestError(f"{model_name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise SourceManifestError(f"{model_name} has unknown fields: {', '.join(unknown)}")


def canonical_logical_path(value: object, *, field_name: str = "logical_path") -> str:
    """Validate one already-normalized, relative POSIX manifest path."""

    text = _required_text(value, field_name=field_name)
    if len(text.encode("utf-8")) > 4096:
        raise SourceManifestError(f"{field_name} exceeds the 4096-byte UTF-8 limit")
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise SourceManifestError(f"{field_name} must not contain control characters")
    if "\\" in text:
        raise SourceManifestError(f"{field_name} must use POSIX '/' separators")
    if text.startswith("//"):
        raise SourceManifestError(f"{field_name} must not be a UNC path")
    if _URI_SCHEME_RE.match(text) is not None:
        raise SourceManifestError(f"{field_name} must not contain a URI or drive scheme")
    path = PurePosixPath(text)
    if path.is_absolute():
        raise SourceManifestError(f"{field_name} must be relative")
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SourceManifestError(
            f"{field_name} must be a normalized relative POSIX path without traversal"
        )
    if path.as_posix() != text:
        raise SourceManifestError(f"{field_name} must be a normalized POSIX path")
    return text


def resolve_logical_reference(
    namespace: str,
    reference: object,
) -> str:
    """Resolve a parser-provided relative reference without filesystem access.

    Backslashes are interpreted as separators because many Windows scanning
    tools emit them in otherwise portable OBJ/MTL files.  The durable result is
    always POSIX.  Dot segments may occur in source text but can never escape
    the source root.
    """

    if isinstance(reference, bytes):
        try:
            raw = reference.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise SourceManifestError("dependency reference must be valid UTF-8") from exc
    elif isinstance(reference, str):
        raw = reference
    else:
        raise SourceManifestError("dependency reference must be text")
    raw = raw.strip()
    if not raw:
        raise SourceManifestError("dependency reference must not be empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in raw):
        raise SourceManifestError("dependency reference must not contain control characters")
    portable = raw.replace("\\", "/")
    if portable.startswith("/") or portable.startswith("//"):
        raise SourceManifestError("dependency reference must be relative")
    if _URI_SCHEME_RE.match(portable) is not None:
        raise SourceManifestError("dependency reference must not be a URI or drive path")

    base_parts = [] if not namespace else canonical_logical_path(
        namespace,
        field_name="resolver namespace",
    ).split("/")
    parts = list(base_parts)
    for part in portable.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise SourceManifestError("dependency reference escapes the source root")
            parts.pop()
            continue
        parts.append(part)
    if not parts:
        raise SourceManifestError("dependency reference resolves to the source root")
    return canonical_logical_path("/".join(parts))


def fixed_media_type(logical_path: str) -> str:
    """Return a platform-independent media type for common mesh resources."""

    suffix = PurePosixPath(logical_path).suffix.lower()
    return {
        ".bin": "application/octet-stream",
        ".bmp": "image/bmp",
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".mtl": "model/mtl",
        ".obj": "model/obj",
        ".off": "model/vnd.off",
        ".ply": "model/ply",
        ".png": "image/png",
        ".stl": "model/stl",
        ".tga": "image/x-tga",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")


@dataclass(frozen=True, slots=True)
class SourceManifestEntry:
    logical_path: str
    media_type: str
    role: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        logical_path = canonical_logical_path(self.logical_path)
        object.__setattr__(self, "logical_path", logical_path)
        expected_media_type = fixed_media_type(logical_path)
        media_type = _required_text(self.media_type, field_name="entry.media_type")
        if media_type != expected_media_type:
            raise SourceManifestError(
                "entry.media_type must match the fixed logical-path media type "
                f"({media_type!r} != {expected_media_type!r})"
            )
        object.__setattr__(
            self,
            "media_type",
            media_type,
        )
        role = _required_text(self.role, field_name="entry.role")
        if role not in {PRIMARY_RESOURCE_ROLE, DEPENDENCY_RESOURCE_ROLE}:
            raise SourceManifestError(f"unsupported source manifest role: {role!r}")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "sha256", _sha256(self.sha256, field_name="entry.sha256"))
        object.__setattr__(
            self,
            "size_bytes",
            _size_bytes(self.size_bytes, field_name="entry.size_bytes"),
        )

    @property
    def content_id(self) -> str:
        return f"sha256:{self.sha256}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_path": self.logical_path,
            "media_type": self.media_type,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SourceManifestEntry":
        if not isinstance(value, Mapping):
            raise SourceManifestError("source manifest entry must be an object")
        _exact_keys(value, _ENTRY_KEYS, model_name="source manifest entry")
        return cls(
            logical_path=_required_text(
                value["logical_path"],
                field_name="entry.logical_path",
            ),
            media_type=_required_text(
                value["media_type"],
                field_name="entry.media_type",
            ),
            role=_required_text(value["role"], field_name="entry.role"),
            sha256=_sha256(value["sha256"], field_name="entry.sha256"),
            size_bytes=_size_bytes(value["size_bytes"], field_name="entry.size_bytes"),
        )


@dataclass(frozen=True, slots=True)
class SourceManifest:
    primary_logical_path: str
    entries: tuple[SourceManifestEntry, ...]
    resolver_profile: str = SOURCE_RESOLVER_PROFILE

    def __post_init__(self) -> None:
        primary_path = canonical_logical_path(
            self.primary_logical_path,
            field_name="primary_logical_path",
        )
        object.__setattr__(self, "primary_logical_path", primary_path)
        if self.resolver_profile != SOURCE_RESOLVER_PROFILE:
            raise SourceManifestError(
                f"unsupported resolver profile: {self.resolver_profile!r}"
            )
        try:
            entries = tuple(self.entries)
        except TypeError as exc:
            raise SourceManifestError("source manifest entries must be iterable") from exc
        if not entries:
            raise SourceManifestError("source manifest entries must not be empty")
        if len(entries) > MAX_SOURCE_MANIFEST_ENTRIES:
            raise SourceManifestError(
                "source manifest has too many entries "
                f"({len(entries)} > {MAX_SOURCE_MANIFEST_ENTRIES})"
            )
        if not all(isinstance(entry, SourceManifestEntry) for entry in entries):
            raise SourceManifestError(
                "source manifest entries must contain only SourceManifestEntry values"
            )
        entries = tuple(sorted(entries, key=lambda entry: entry.logical_path))
        paths = [entry.logical_path for entry in entries]
        if len(paths) != len(set(paths)):
            raise SourceManifestError("source manifest logical paths must be unique")
        primary = [entry for entry in entries if entry.role == PRIMARY_RESOURCE_ROLE]
        if len(primary) != 1:
            raise SourceManifestError(
                "source manifest must contain exactly one primary_mesh entry"
            )
        if primary[0].logical_path != primary_path:
            raise SourceManifestError(
                "primary_logical_path must identify the primary_mesh entry"
            )
        object.__setattr__(self, "entries", entries)

    @property
    def primary_entry(self) -> SourceManifestEntry:
        return next(entry for entry in self.entries if entry.role == PRIMARY_RESOURCE_ROLE)

    @property
    def dependency_entries(self) -> tuple[SourceManifestEntry, ...]:
        return tuple(
            entry for entry in self.entries if entry.role == DEPENDENCY_RESOURCE_ROLE
        )

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": SOURCE_MANIFEST_FORMAT,
            "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
            "resolver_profile": self.resolver_profile,
            "primary_logical_path": self.primary_logical_path,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SourceManifest":
        if not isinstance(value, Mapping):
            raise SourceManifestError("source manifest must be an object")
        _exact_keys(value, _MANIFEST_KEYS, model_name="source manifest")
        if value["format"] != SOURCE_MANIFEST_FORMAT:
            raise SourceManifestError(
                f"unsupported source manifest format: {value['format']!r}"
            )
        if value["schema_version"] != SOURCE_MANIFEST_SCHEMA_VERSION:
            raise SourceManifestError(
                "unsupported source manifest schema version: "
                f"{value['schema_version']!r}"
            )
        raw_entries = value["entries"]
        if not isinstance(raw_entries, (list, tuple)):
            raise SourceManifestError("source manifest entries must be an array")
        entries: list[SourceManifestEntry] = []
        for index, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, Mapping):
                raise SourceManifestError(
                    f"source manifest entries[{index}] must be an object"
                )
            entries.append(SourceManifestEntry.from_dict(raw_entry))
        return cls(
            primary_logical_path=_required_text(
                value["primary_logical_path"],
                field_name="primary_logical_path",
            ),
            entries=tuple(entries),
            resolver_profile=_required_text(
                value["resolver_profile"],
                field_name="resolver_profile",
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedSourceResource:
    """Runtime-only locator for one durable manifest entry."""

    entry: SourceManifestEntry
    locator: str

    def __post_init__(self) -> None:
        if not isinstance(self.entry, SourceManifestEntry):
            raise SourceManifestError("resolved resource entry is invalid")
        locator = _required_text(self.locator, field_name="resource locator")
        object.__setattr__(self, "locator", locator)


__all__ = [
    "DEPENDENCY_RESOURCE_ROLE",
    "MAX_SOURCE_MANIFEST_ENTRIES",
    "PRIMARY_RESOURCE_ROLE",
    "SOURCE_MANIFEST_FORMAT",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "SOURCE_RESOLVER_PROFILE",
    "ResolvedSourceResource",
    "SourceManifest",
    "SourceManifestEntry",
    "SourceManifestError",
    "canonical_logical_path",
    "fixed_media_type",
    "resolve_logical_reference",
]
