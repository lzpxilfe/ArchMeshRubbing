"""Canonical index model for source bytes embedded in an AMR container.

The artifact document remains the scientific authority.  This module only
describes a deterministic, content-addressed copy of its source assets so an
offline project can carry the exact bytes needed to reproduce later work.
Archive I/O belongs to :mod:`src.core.project_file`; the values here contain no
host paths and perform no filesystem access.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from .artifact_document import (
    PRIMARY_SOURCE_ASSET_ROLE,
    ArtifactDocument,
)
from .canonical_json import canonical_json_bytes as _canonical_json_bytes
from .mesh_import_recipe import MeshImportRecipeError, validate_mesh_import_recipe
from .source_manifest import (
    DEPENDENCY_RESOURCE_ROLE,
    MAX_SOURCE_MANIFEST_ENTRIES,
    SourceManifestEntry,
)


SOURCE_BUNDLE_FORMAT = "archmeshrubbing_source_bundle"
SOURCE_BUNDLE_SCHEMA_VERSION = "1.0.0"
SOURCE_BUNDLE_SCHEMA_VERSION_CLOSED_MANIFEST = "2.0.0"
SOURCE_INDEX_NAME = "sources/index.json"
SOURCE_BLOB_PREFIX = "sources/blobs/sha256/"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_ENTRY_KEYS = {
    "source_asset_id",
    "role",
    "logical_path",
    "media_type",
    "member",
    "sha256",
    "size_bytes",
}
_INDEX_KEYS = {
    "format",
    "schema_version",
    "document_id",
    "document_sha256",
    "primary_source_asset_id",
    "entries",
}


class SourceBundleError(ValueError):
    """A source bundle index violates its closed, portable contract."""


def _required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceBundleError(f"{field_name} must be a non-empty string")
    return value


def _sha256(value: object, *, field_name: str) -> str:
    digest = _required_string(value, field_name=field_name)
    if _SHA256_RE.fullmatch(digest) is None:
        raise SourceBundleError(
            f"{field_name} must be 64 lowercase hexadecimal characters"
        )
    return digest


def _size_bytes(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceBundleError(f"{field_name} must be a non-negative integer")
    return value


def _relative_posix_path(value: object, *, field_name: str) -> str:
    path_text = _required_string(value, field_name=field_name)
    if any(ord(character) < 32 or ord(character) == 127 for character in path_text):
        raise SourceBundleError(f"{field_name} must not contain control characters")
    if "\\" in path_text:
        raise SourceBundleError(f"{field_name} must use POSIX '/' separators")
    if path_text.startswith("//"):
        raise SourceBundleError(f"{field_name} must not be a UNC path")
    if _URI_SCHEME_RE.match(path_text) is not None:
        raise SourceBundleError(f"{field_name} must not contain a URI or drive scheme")

    path = PurePosixPath(path_text)
    if path.is_absolute():
        raise SourceBundleError(f"{field_name} must be relative")
    parts = path_text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SourceBundleError(
            f"{field_name} must be a normalized relative POSIX path without traversal"
        )
    if path.as_posix() != path_text:
        raise SourceBundleError(f"{field_name} must be a normalized POSIX path")
    return path_text


def _exact_keys(
    data: Mapping[str, object],
    expected: set[str],
    *,
    model_name: str,
) -> None:
    raw_keys = set(data)
    if not all(isinstance(key, str) for key in raw_keys):
        raise SourceBundleError(f"{model_name} field names must be strings")
    observed = {key for key in raw_keys if isinstance(key, str)}
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise SourceBundleError(f"{model_name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise SourceBundleError(f"{model_name} has unknown fields: {', '.join(unknown)}")


def source_blob_member(sha256: str) -> str:
    """Return the one canonical archive member for a source content digest."""

    digest = _sha256(sha256, field_name="sha256")
    return f"{SOURCE_BLOB_PREFIX}{digest}"


@dataclass(frozen=True, slots=True)
class EmbeddedSourceEntry:
    """One content-addressed source file included in an offline project."""

    source_asset_id: str
    role: str
    logical_path: str
    media_type: str
    member: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        digest = _sha256(self.sha256, field_name="entry.sha256")
        object.__setattr__(self, "sha256", digest)

        asset_id = _required_string(
            self.source_asset_id,
            field_name="entry.source_asset_id",
        )
        if asset_id != f"sha256:{digest}":
            raise SourceBundleError(
                "entry.source_asset_id must equal sha256:<entry.sha256>"
            )
        object.__setattr__(self, "source_asset_id", asset_id)
        object.__setattr__(
            self,
            "role",
            _required_string(self.role, field_name="entry.role"),
        )
        object.__setattr__(
            self,
            "logical_path",
            _relative_posix_path(self.logical_path, field_name="entry.logical_path"),
        )
        object.__setattr__(
            self,
            "media_type",
            _required_string(self.media_type, field_name="entry.media_type"),
        )

        member = _relative_posix_path(self.member, field_name="entry.member")
        expected_member = source_blob_member(digest)
        if member != expected_member:
            raise SourceBundleError(
                "entry.member must equal the content-addressed member for entry.sha256"
            )
        object.__setattr__(self, "member", member)
        object.__setattr__(
            self,
            "size_bytes",
            _size_bytes(self.size_bytes, field_name="entry.size_bytes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_asset_id": self.source_asset_id,
            "role": self.role,
            "logical_path": self.logical_path,
            "media_type": self.media_type,
            "member": self.member,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EmbeddedSourceEntry":
        if not isinstance(data, Mapping):
            raise SourceBundleError("embedded_source_entry must be a JSON object")
        _exact_keys(data, _ENTRY_KEYS, model_name="embedded_source_entry")
        return cls(
            source_asset_id=_required_string(
                data["source_asset_id"],
                field_name="entry.source_asset_id",
            ),
            role=_required_string(data["role"], field_name="entry.role"),
            logical_path=_required_string(
                data["logical_path"],
                field_name="entry.logical_path",
            ),
            media_type=_required_string(
                data["media_type"],
                field_name="entry.media_type",
            ),
            member=_required_string(data["member"], field_name="entry.member"),
            sha256=_required_string(data["sha256"], field_name="entry.sha256"),
            size_bytes=_size_bytes(data["size_bytes"], field_name="entry.size_bytes"),
        )


@dataclass(frozen=True, slots=True)
class SourceBundleIndex:
    """Closed RFC 8785 index binding embedded bytes to one document snapshot."""

    document_id: str
    document_sha256: str
    primary_source_asset_id: str
    entries: tuple[EmbeddedSourceEntry, ...]
    schema_version: str = SOURCE_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_id",
            _required_string(self.document_id, field_name="document_id"),
        )
        object.__setattr__(
            self,
            "document_sha256",
            _sha256(self.document_sha256, field_name="document_sha256"),
        )
        object.__setattr__(
            self,
            "primary_source_asset_id",
            _required_string(
                self.primary_source_asset_id,
                field_name="primary_source_asset_id",
            ),
        )
        if self.schema_version not in {
            SOURCE_BUNDLE_SCHEMA_VERSION,
            SOURCE_BUNDLE_SCHEMA_VERSION_CLOSED_MANIFEST,
        }:
            raise SourceBundleError(
                f"unsupported source bundle schema version: {self.schema_version!r}"
            )

        try:
            entries = tuple(self.entries)
        except TypeError as exc:
            raise SourceBundleError("entries must be an iterable of source entries") from exc
        if not entries:
            raise SourceBundleError("entries must not be empty")
        if len(entries) > MAX_SOURCE_MANIFEST_ENTRIES:
            raise SourceBundleError(
                "source bundle has too many entries "
                f"({len(entries)} > {MAX_SOURCE_MANIFEST_ENTRIES})"
            )
        if not all(isinstance(entry, EmbeddedSourceEntry) for entry in entries):
            raise SourceBundleError(
                "entries must contain only EmbeddedSourceEntry values"
            )
        entries = tuple(
            sorted(
                entries,
                key=lambda entry: (
                    entry.logical_path,
                    entry.member,
                    entry.source_asset_id,
                    entry.role,
                ),
            )
        )
        object.__setattr__(self, "entries", entries)

        if self.schema_version == SOURCE_BUNDLE_SCHEMA_VERSION:
            logical_paths = [entry.logical_path for entry in entries]
            if len(set(logical_paths)) != len(logical_paths):
                raise SourceBundleError("entries must have unique logical_path values")
            members = [entry.member for entry in entries]
            if len(set(members)) != len(members):
                raise SourceBundleError("entries must have unique member values")
            source_asset_ids = [entry.source_asset_id for entry in entries]
            if len(set(source_asset_ids)) != len(source_asset_ids):
                raise SourceBundleError("entries must have unique source_asset_id values")
        else:
            if len(entries) < 2:
                raise SourceBundleError(
                    "v2 source bundle must include a primary and at least one dependency"
                )
            if any(
                entry.role not in {
                    PRIMARY_SOURCE_ASSET_ROLE,
                    DEPENDENCY_RESOURCE_ROLE,
                }
                for entry in entries
            ):
                raise SourceBundleError("v2 source bundle contains an unsupported role")
            aliases = [(entry.logical_path, entry.sha256) for entry in entries]
            if len(set(aliases)) != len(aliases):
                raise SourceBundleError(
                    "v2 entries must have unique logical_path and sha256 pairs"
                )

        primary_entries = [
            entry for entry in entries if entry.role == PRIMARY_SOURCE_ASSET_ROLE
        ]
        if len(primary_entries) != 1:
            raise SourceBundleError("entries must contain exactly one primary_mesh role")
        if primary_entries[0].source_asset_id != self.primary_source_asset_id:
            raise SourceBundleError(
                "primary_source_asset_id must identify the primary_mesh entry"
            )

    @classmethod
    def for_document(cls, document: ArtifactDocument) -> "SourceBundleIndex":
        """Create a transport index for every active parser input byte stream."""

        if not isinstance(document, ArtifactDocument):
            raise SourceBundleError("document must be an ArtifactDocument")
        if len(document.source_assets) != 1:
            raise SourceBundleError(
                "the current source bundle contract requires exactly one SourceAsset"
            )
        source = document.source_assets[0]
        calculated_document_sha256 = hashlib.sha256(
            document.canonical_json_bytes()
        ).hexdigest()
        if calculated_document_sha256 != document.canonical_sha256:
            raise SourceBundleError(
                "document canonical_sha256 does not match its canonical bytes"
            )

        manifests = []
        for geometry in document.geometry_revisions:
            try:
                execution = validate_mesh_import_recipe(
                    geometry.import_recipe,
                    allow_legacy=True,
                    require_current_runtime=False,
                )
            except MeshImportRecipeError as exc:
                raise SourceBundleError(
                    f"geometry {geometry.id!r} has an invalid import recipe: {exc}"
                ) from exc
            if execution.source_manifest is not None:
                primary = execution.source_manifest.primary_entry
                if (
                    primary.sha256 != source.sha256
                    or primary.size_bytes != source.size_bytes
                ):
                    raise SourceBundleError(
                        "import recipe primary entry does not match the ArtifactDocument "
                        "SourceAsset"
                    )
                manifests.append(execution.source_manifest)

        if manifests:
            active_manifest = manifests[0]
            metadata_id = document.active_source_metadata_revision_id
            if metadata_id is not None:
                active_geometry_id = document.source_metadata_revision_index[
                    metadata_id
                ].geometry_revision_id
                active_geometry = document.geometry_revision_index[active_geometry_id]
                try:
                    active_execution = validate_mesh_import_recipe(
                        active_geometry.import_recipe,
                        allow_legacy=True,
                        require_current_runtime=False,
                    )
                except MeshImportRecipeError as exc:
                    raise SourceBundleError(str(exc)) from exc
                if active_execution.source_manifest is not None:
                    active_manifest = active_execution.source_manifest

            manifest_entries: dict[tuple[str, str], SourceManifestEntry] = {}
            content_sizes: dict[str, int] = {source.sha256: source.size_bytes}
            for manifest in manifests:
                for manifest_entry in manifest.entries:
                    previous_size = content_sizes.get(manifest_entry.sha256)
                    if (
                        previous_size is not None
                        and previous_size != manifest_entry.size_bytes
                    ):
                        raise SourceBundleError(
                            "one content digest has conflicting source sizes"
                        )
                    content_sizes[manifest_entry.sha256] = manifest_entry.size_bytes
                for dependency in manifest.dependency_entries:
                    key = (dependency.logical_path, dependency.sha256)
                    previous = manifest_entries.get(key)
                    if previous is not None and previous != dependency:
                        raise SourceBundleError(
                            "source manifests contain conflicting dependency metadata"
                        )
                    manifest_entries[key] = dependency
            primary = active_manifest.primary_entry
            entries = [
                EmbeddedSourceEntry(
                    source_asset_id=source.id,
                    role=source.role,
                    logical_path=primary.logical_path,
                    media_type=primary.media_type,
                    member=source_blob_member(primary.sha256),
                    sha256=primary.sha256,
                    size_bytes=primary.size_bytes,
                )
            ]
            entries.extend(
                EmbeddedSourceEntry(
                    source_asset_id=entry.content_id,
                    role=entry.role,
                    logical_path=entry.logical_path,
                    media_type=entry.media_type,
                    member=source_blob_member(entry.sha256),
                    sha256=entry.sha256,
                    size_bytes=entry.size_bytes,
                )
                for entry in manifest_entries.values()
            )
            schema_version = SOURCE_BUNDLE_SCHEMA_VERSION_CLOSED_MANIFEST
        else:
            entries = [
                EmbeddedSourceEntry(
                    source_asset_id=source.id,
                    role=source.role,
                    logical_path=source.original_name,
                    media_type=source.media_type,
                    member=source_blob_member(source.sha256),
                    sha256=source.sha256,
                    size_bytes=source.size_bytes,
                )
            ]
            schema_version = SOURCE_BUNDLE_SCHEMA_VERSION
        return cls(
            document_id=document.document_id,
            document_sha256=calculated_document_sha256,
            primary_source_asset_id=source.id,
            entries=tuple(entries),
            schema_version=schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": SOURCE_BUNDLE_FORMAT,
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "document_sha256": self.document_sha256,
            "primary_source_asset_id": self.primary_source_asset_id,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def canonical_json_bytes(self) -> bytes:
        """Return the RFC 8785 representation used as the archive index bytes."""

        return _canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SourceBundleIndex":
        if not isinstance(data, Mapping):
            raise SourceBundleError("source_bundle_index must be a JSON object")
        _exact_keys(data, _INDEX_KEYS, model_name="source_bundle_index")
        if data["format"] != SOURCE_BUNDLE_FORMAT:
            raise SourceBundleError(f"unsupported source bundle format: {data['format']!r}")
        if data["schema_version"] not in {
            SOURCE_BUNDLE_SCHEMA_VERSION,
            SOURCE_BUNDLE_SCHEMA_VERSION_CLOSED_MANIFEST,
        }:
            raise SourceBundleError(
                "unsupported source bundle schema version: "
                f"{data['schema_version']!r}"
            )

        raw_entries = data["entries"]
        if not isinstance(raw_entries, list):
            raise SourceBundleError("source_bundle_index.entries must be an array")
        entries: list[EmbeddedSourceEntry] = []
        for index, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, Mapping):
                raise SourceBundleError(
                    f"source_bundle_index.entries[{index}] must be an object"
                )
            entries.append(EmbeddedSourceEntry.from_dict(raw_entry))

        return cls(
            document_id=_required_string(data["document_id"], field_name="document_id"),
            document_sha256=_required_string(
                data["document_sha256"],
                field_name="document_sha256",
            ),
            primary_source_asset_id=_required_string(
                data["primary_source_asset_id"],
                field_name="primary_source_asset_id",
            ),
            entries=tuple(entries),
            schema_version=str(data["schema_version"]),
        )


__all__ = [
    "SOURCE_BLOB_PREFIX",
    "SOURCE_BUNDLE_FORMAT",
    "SOURCE_BUNDLE_SCHEMA_VERSION",
    "SOURCE_BUNDLE_SCHEMA_VERSION_CLOSED_MANIFEST",
    "SOURCE_INDEX_NAME",
    "EmbeddedSourceEntry",
    "SourceBundleError",
    "SourceBundleIndex",
    "source_blob_member",
]
