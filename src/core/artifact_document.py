"""Authoritative, headless document model for one recorded artifact.

The legacy GUI stores mutable ``SceneObject`` fields.  This module defines the
durable scientific authority beneath that projection:

    SourceAsset -> GeometryRevision -> SourceMetadataRevision -> AlignRevision
                                                                  |
                                                                  v
                                                            DerivedRecord

Revisions and records are immutable values.  Editing appends a new value and
moves an explicit active pointer; it never rewrites or deletes earlier work.
Large geometry/face/point payloads remain external references and are never
embedded as manifest-sized JSON arrays here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import heapq
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import re

import numpy as np

from .alignment_utils import require_affine_matrix4x4, require_rigid_matrix4x4
from .canonical_json import CanonicalJSONError, canonical_json_sha256
from .source_identity import PRIMARY_FILE_IDENTITY_SCOPE, SourceFingerprint


ARTIFACT_DOCUMENT_SCHEMA_VERSION = "1.0.0"
CANONICAL_UNIT = "mm"
PRIMARY_SOURCE_ASSET_ROLE = "primary_mesh"
GEOMETRY_HASH_SCOPE_V1 = "positions-f64le+triangles-i32le/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_SECONDS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_AXIS_VALUES = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}
_AXIS_KEYS = ("source_x", "source_y", "source_z")
_UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}


class ArtifactDocumentError(ValueError):
    """The document violates a durable domain invariant."""


class UnconfirmedMetadataError(ArtifactDocumentError):
    """Canonical materialization was requested from unconfirmed metadata."""


class MetadataConfirmationStatus(str, Enum):
    UNCONFIRMED = "unconfirmed"
    CONFIRMED = "confirmed"


class Handedness(str, Enum):
    UNKNOWN = "unknown"
    RIGHT = "right"
    LEFT = "left"


class RecordLifecycleStatus(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    FAILED = "failed"


class RecordFreshness(str, Enum):
    FRESH = "fresh"
    STALE_ALIGNMENT = "stale_alignment"
    STALE_METADATA = "stale_metadata"
    MISSING_DEPENDENCY = "missing_dependency"
    BLOCKED_DEPENDENCY = "blocked_dependency"
    INVALID = "invalid"


def _canonical_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ArtifactDocumentError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ArtifactDocumentError(f"{field_name} must be a finite number")
    return 0.0 if number == 0.0 else number


def _required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactDocumentError(f"{field_name} must be a non-empty string")
    return value


def _optional_id(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name=field_name)


def _canonical_timestamp(value: object, *, field_name: str) -> str:
    timestamp = _required_string(value, field_name=field_name)
    if _UTC_SECONDS_RE.fullmatch(timestamp) is None:
        raise ArtifactDocumentError(
            f"{field_name} must use canonical UTC seconds (YYYY-MM-DDTHH:MM:SSZ)"
        )
    return timestamp


def _sha256(value: object, *, field_name: str) -> str:
    digest = _required_string(value, field_name=field_name).lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ArtifactDocumentError(f"{field_name} must be 64 lowercase hexadecimal characters")
    return digest


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactDocumentError(f"{field_name} must be a non-negative integer")
    return value


def _freeze_json(value: Any, *, path: str = "$", depth: int = 0) -> Any:
    """Normalize JSON to immutable values with deterministic mapping order."""

    if depth > 100:
        raise ArtifactDocumentError(f"JSON extension nesting is too deep at {path}")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        return _canonical_float(value, field_name=path)
    if isinstance(value, Mapping):
        keys = list(value)
        for key in keys:
            if not isinstance(key, str):
                raise ArtifactDocumentError(f"JSON object key at {path} must be a string")
        normalized: dict[str, Any] = {}
        for key in sorted(keys):
            normalized[key] = _freeze_json(value[key], path=f"{path}.{key}", depth=depth + 1)
        return MappingProxyType(normalized)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )
    raise ArtifactDocumentError(f"Unsupported JSON value at {path}: {type(value).__name__}")


def _freeze_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactDocumentError(f"{field_name} must be a JSON object")
    frozen = _freeze_json(value, path=field_name)
    assert isinstance(frozen, Mapping)
    return frozen


def _freeze_extensions(value: object, *, field_name: str) -> Mapping[str, Any]:
    frozen = _freeze_mapping(value, field_name=field_name)
    for key in frozen:
        if ":" not in key:
            raise ArtifactDocumentError(
                f"{field_name} key {key!r} must be namespaced (for example 'org.example:key')"
            )
    return frozen


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _matrix_tuple(
    value: object,
    *,
    field_name: str,
    rigid: bool,
) -> tuple[tuple[float, float, float, float], ...]:
    try:
        matrix = (
            require_rigid_matrix4x4(value, field_name=field_name)  # type: ignore[arg-type]
            if rigid
            else require_affine_matrix4x4(value, field_name=field_name)  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactDocumentError(str(exc)) from exc
    matrix[matrix == 0.0] = 0.0
    return tuple(tuple(float(cell) for cell in row) for row in matrix)  # type: ignore[return-value]


def _exact_keys(data: Mapping[str, object], expected: set[str], *, model_name: str) -> None:
    observed = set(data)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise ArtifactDocumentError(f"{model_name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ArtifactDocumentError(f"{model_name} has unknown fields: {', '.join(unknown)}")


def _as_string_tuple(value: object, *, field_name: str, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ArtifactDocumentError(f"{field_name} must be an array")
    items = tuple(_required_string(item, field_name=f"{field_name}[]") for item in value)
    if not allow_empty and not items:
        raise ArtifactDocumentError(f"{field_name} must not be empty")
    if len(set(items)) != len(items):
        raise ArtifactDocumentError(f"{field_name} must not contain duplicate IDs")
    return tuple(sorted(items))


def _axis_matrix(axes: Mapping[str, object]) -> np.ndarray:
    if set(axes) != set(_AXIS_KEYS):
        raise ArtifactDocumentError(
            "axes must contain exactly source_x, source_y and source_z"
        )
    vectors = {
        "+X": np.array([1.0, 0.0, 0.0]),
        "-X": np.array([-1.0, 0.0, 0.0]),
        "+Y": np.array([0.0, 1.0, 0.0]),
        "-Y": np.array([0.0, -1.0, 0.0]),
        "+Z": np.array([0.0, 0.0, 1.0]),
        "-Z": np.array([0.0, 0.0, -1.0]),
    }
    values: list[str] = []
    for key in _AXIS_KEYS:
        raw = axes[key]
        if not isinstance(raw, str) or raw not in _AXIS_VALUES:
            raise ArtifactDocumentError(f"axes.{key} must be one of {sorted(_AXIS_VALUES)}")
        values.append(raw)
    if len({value[-1] for value in values}) != 3:
        raise ArtifactDocumentError("axes must map each source axis to a unique canonical axis")
    return np.column_stack([vectors[value] for value in values])


def source_to_canonical_mm_matrix(
    unit: str,
    axes: Mapping[str, object],
) -> np.ndarray:
    """Build the explicit source-unit/signed-axis transform to millimeters."""

    unit_name = _required_string(unit, field_name="unit").lower()
    if unit_name not in _UNIT_TO_MM:
        raise ArtifactDocumentError("unit must be mm, cm or m")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _axis_matrix(axes) * _UNIT_TO_MM[unit_name]
    return matrix


@dataclass(frozen=True, slots=True)
class SourceAsset:
    id: str
    sha256: str
    size_bytes: int
    media_type: str
    original_name: str
    asset_ref: str
    role: str = PRIMARY_SOURCE_ASSET_ROLE
    identity_scope: str = PRIMARY_FILE_IDENTITY_SCOPE
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_string(self.id, field_name="source_asset.id"))
        digest = _sha256(self.sha256, field_name="source_asset.sha256")
        object.__setattr__(self, "sha256", digest)
        if self.id != f"sha256:{digest}":
            raise ArtifactDocumentError("source_asset.id must equal sha256:<sha256>")
        object.__setattr__(
            self,
            "size_bytes",
            _non_negative_int(self.size_bytes, field_name="source_asset.size_bytes"),
        )
        object.__setattr__(
            self,
            "media_type",
            _required_string(self.media_type, field_name="source_asset.media_type"),
        )
        object.__setattr__(
            self,
            "original_name",
            _required_string(self.original_name, field_name="source_asset.original_name"),
        )
        object.__setattr__(
            self,
            "asset_ref",
            _required_string(self.asset_ref, field_name="source_asset.asset_ref"),
        )
        role = _required_string(self.role, field_name="source_asset.role")
        if role != PRIMARY_SOURCE_ASSET_ROLE:
            raise ArtifactDocumentError(
                f"unsupported source asset role: {role!r}"
            )
        object.__setattr__(self, "role", role)
        scope = _required_string(self.identity_scope, field_name="source_asset.identity_scope")
        if scope != PRIMARY_FILE_IDENTITY_SCOPE:
            raise ArtifactDocumentError(
                f"unsupported source asset identity_scope: {scope!r}"
            )
        object.__setattr__(self, "identity_scope", scope)
        object.__setattr__(
            self,
            "extensions",
            _freeze_extensions(self.extensions, field_name="source_asset.extensions"),
        )

    @classmethod
    def from_fingerprint(
        cls,
        fingerprint: SourceFingerprint,
        *,
        asset_ref: str,
        media_type: str = "application/octet-stream",
        role: str = PRIMARY_SOURCE_ASSET_ROLE,
    ) -> "SourceAsset":
        if not isinstance(fingerprint, SourceFingerprint):
            raise ArtifactDocumentError("fingerprint must be a SourceFingerprint")
        return cls(
            id=fingerprint.id,
            sha256=fingerprint.sha256,
            size_bytes=fingerprint.size_bytes,
            media_type=media_type,
            original_name=fingerprint.original_name,
            asset_ref=asset_ref,
            role=role,
            identity_scope=fingerprint.identity_scope,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "original_name": self.original_name,
            "asset_ref": self.asset_ref,
            "role": self.role,
            "identity_scope": self.identity_scope,
            "extensions": _thaw_json(self.extensions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SourceAsset":
        _exact_keys(
            data,
            {
                "id",
                "sha256",
                "size_bytes",
                "media_type",
                "original_name",
                "asset_ref",
                "role",
                "identity_scope",
                "extensions",
            },
            model_name="source_asset",
        )
        return cls(**dict(data))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class GeometryRevision:
    id: str
    source_asset_ids: tuple[str, ...]
    geometry_sha256: str
    geometry_hash_scope: str
    import_recipe: Mapping[str, Any]
    topology_map_ref: str | None
    qc: Mapping[str, Any]
    created_at: str
    operator: str
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_string(self.id, field_name="geometry.id"))
        object.__setattr__(
            self,
            "source_asset_ids",
            _as_string_tuple(
                self.source_asset_ids,
                field_name="geometry.source_asset_ids",
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "geometry_sha256",
            _sha256(self.geometry_sha256, field_name="geometry.geometry_sha256"),
        )
        geometry_hash_scope = _required_string(
            self.geometry_hash_scope,
            field_name="geometry.geometry_hash_scope",
        )
        if geometry_hash_scope != GEOMETRY_HASH_SCOPE_V1:
            raise ArtifactDocumentError(
                f"unsupported geometry hash scope: {geometry_hash_scope!r}"
            )
        object.__setattr__(self, "geometry_hash_scope", geometry_hash_scope)
        object.__setattr__(
            self,
            "import_recipe",
            _freeze_mapping(self.import_recipe, field_name="geometry.import_recipe"),
        )
        object.__setattr__(
            self,
            "topology_map_ref",
            _optional_id(self.topology_map_ref, field_name="geometry.topology_map_ref"),
        )
        object.__setattr__(self, "qc", _freeze_mapping(self.qc, field_name="geometry.qc"))
        object.__setattr__(
            self,
            "created_at",
            _canonical_timestamp(self.created_at, field_name="geometry.created_at"),
        )
        object.__setattr__(
            self,
            "operator",
            _required_string(self.operator, field_name="geometry.operator"),
        )
        object.__setattr__(
            self,
            "extensions",
            _freeze_extensions(self.extensions, field_name="geometry.extensions"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_asset_ids": list(self.source_asset_ids),
            "geometry_sha256": self.geometry_sha256,
            "geometry_hash_scope": self.geometry_hash_scope,
            "import_recipe": _thaw_json(self.import_recipe),
            "topology_map_ref": self.topology_map_ref,
            "qc": _thaw_json(self.qc),
            "created_at": self.created_at,
            "operator": self.operator,
            "extensions": _thaw_json(self.extensions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "GeometryRevision":
        _exact_keys(
            data,
            {
                "id",
                "source_asset_ids",
                "geometry_sha256",
                "geometry_hash_scope",
                "import_recipe",
                "topology_map_ref",
                "qc",
                "created_at",
                "operator",
                "extensions",
            },
            model_name="geometry_revision",
        )
        return cls(**dict(data))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SourceMetadataRevision:
    id: str
    parent_id: str | None
    geometry_revision_id: str
    unit: str
    axes: Mapping[str, str]
    handedness: Handedness | str
    confirmation_status: MetadataConfirmationStatus | str
    source_to_canonical_mm: tuple[tuple[float, float, float, float], ...]
    created_at: str
    operator: str
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_string(self.id, field_name="metadata.id"))
        object.__setattr__(
            self,
            "parent_id",
            _optional_id(self.parent_id, field_name="metadata.parent_id"),
        )
        object.__setattr__(
            self,
            "geometry_revision_id",
            _required_string(
                self.geometry_revision_id,
                field_name="metadata.geometry_revision_id",
            ),
        )
        unit = _required_string(self.unit, field_name="metadata.unit").lower()
        if unit not in {*_UNIT_TO_MM, "unknown"}:
            raise ArtifactDocumentError("metadata.unit must be mm, cm, m or unknown")
        object.__setattr__(self, "unit", unit)

        if not isinstance(self.axes, Mapping):
            raise ArtifactDocumentError("metadata.axes must be an object")
        axes = {key: self.axes[key] for key in self.axes}
        axis_matrix = _axis_matrix(axes)
        object.__setattr__(self, "axes", MappingProxyType(dict(sorted(axes.items()))))

        try:
            handedness = Handedness(self.handedness)
        except (TypeError, ValueError) as exc:
            raise ArtifactDocumentError(f"unsupported metadata.handedness: {self.handedness!r}") from exc
        object.__setattr__(self, "handedness", handedness)
        try:
            confirmation = MetadataConfirmationStatus(self.confirmation_status)
        except (TypeError, ValueError) as exc:
            raise ArtifactDocumentError(
                f"unsupported metadata.confirmation_status: {self.confirmation_status!r}"
            ) from exc
        object.__setattr__(self, "confirmation_status", confirmation)

        matrix = _matrix_tuple(
            self.source_to_canonical_mm,
            field_name="metadata.source_to_canonical_mm",
            rigid=False,
        )
        object.__setattr__(self, "source_to_canonical_mm", matrix)
        object.__setattr__(
            self,
            "created_at",
            _canonical_timestamp(self.created_at, field_name="metadata.created_at"),
        )
        object.__setattr__(
            self,
            "operator",
            _required_string(self.operator, field_name="metadata.operator"),
        )
        object.__setattr__(
            self,
            "extensions",
            _freeze_extensions(self.extensions, field_name="metadata.extensions"),
        )

        if confirmation is MetadataConfirmationStatus.CONFIRMED:
            if unit == "unknown":
                raise ArtifactDocumentError("confirmed metadata must declare a known unit")
            if handedness is Handedness.UNKNOWN:
                raise ArtifactDocumentError("confirmed metadata must declare handedness")
            expected_linear = axis_matrix * _UNIT_TO_MM[unit]
            observed = np.asarray(matrix, dtype=np.float64)
            if not np.allclose(
                observed[:3, :3],
                expected_linear,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ArtifactDocumentError(
                    "confirmed metadata source_to_canonical_mm must exactly encode unit and axes"
                )
            if not np.allclose(observed[:3, 3], 0.0, rtol=0.0, atol=1e-9):
                raise ArtifactDocumentError(
                    "confirmed metadata source_to_canonical_mm cannot translate source geometry"
                )
            orientation = float(np.linalg.det(axis_matrix))
            expected_handedness = Handedness.RIGHT if orientation > 0.0 else Handedness.LEFT
            if handedness is not expected_handedness:
                raise ArtifactDocumentError(
                    "metadata handedness does not match its signed axis mapping"
                )

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.source_to_canonical_mm, dtype=np.float64)

    def require_confirmed_matrix(self) -> np.ndarray:
        if self.confirmation_status is not MetadataConfirmationStatus.CONFIRMED:
            raise UnconfirmedMetadataError(
                f"metadata revision {self.id!r} is not confirmed"
            )
        return self.matrix

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "geometry_revision_id": self.geometry_revision_id,
            "unit": self.unit,
            "axes": dict(self.axes),
            "handedness": Handedness(self.handedness).value,
            "confirmation_status": MetadataConfirmationStatus(
                self.confirmation_status
            ).value,
            "source_to_canonical_mm": [list(row) for row in self.source_to_canonical_mm],
            "created_at": self.created_at,
            "operator": self.operator,
            "extensions": _thaw_json(self.extensions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SourceMetadataRevision":
        _exact_keys(
            data,
            {
                "id",
                "parent_id",
                "geometry_revision_id",
                "unit",
                "axes",
                "handedness",
                "confirmation_status",
                "source_to_canonical_mm",
                "created_at",
                "operator",
                "extensions",
            },
            model_name="source_metadata_revision",
        )
        return cls(**dict(data))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class AlignRevision:
    id: str
    parent_id: str | None
    source_metadata_revision_id: str
    matrix4x4: tuple[tuple[float, float, float, float], ...]
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]
    created_at: str
    operator: str
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_string(self.id, field_name="align.id"))
        object.__setattr__(
            self,
            "parent_id",
            _optional_id(self.parent_id, field_name="align.parent_id"),
        )
        object.__setattr__(
            self,
            "source_metadata_revision_id",
            _required_string(
                self.source_metadata_revision_id,
                field_name="align.source_metadata_revision_id",
            ),
        )
        object.__setattr__(
            self,
            "matrix4x4",
            _matrix_tuple(self.matrix4x4, field_name="align.matrix4x4", rigid=True),
        )
        object.__setattr__(self, "recipe", _freeze_mapping(self.recipe, field_name="align.recipe"))
        object.__setattr__(self, "qc", _freeze_mapping(self.qc, field_name="align.qc"))
        object.__setattr__(
            self,
            "created_at",
            _canonical_timestamp(self.created_at, field_name="align.created_at"),
        )
        object.__setattr__(
            self,
            "operator",
            _required_string(self.operator, field_name="align.operator"),
        )
        object.__setattr__(
            self,
            "extensions",
            _freeze_extensions(self.extensions, field_name="align.extensions"),
        )

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.matrix4x4, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "source_metadata_revision_id": self.source_metadata_revision_id,
            "matrix4x4": [list(row) for row in self.matrix4x4],
            "recipe": _thaw_json(self.recipe),
            "qc": _thaw_json(self.qc),
            "created_at": self.created_at,
            "operator": self.operator,
            "extensions": _thaw_json(self.extensions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AlignRevision":
        _exact_keys(
            data,
            {
                "id",
                "parent_id",
                "source_metadata_revision_id",
                "matrix4x4",
                "recipe",
                "qc",
                "created_at",
                "operator",
                "extensions",
            },
            model_name="align_revision",
        )
        return cls(**dict(data))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class OperationContext:
    source_asset_ids: tuple[str, ...]
    geometry_revision_id: str
    source_metadata_revision_id: str
    align_revision_id: str
    recipe_hash: str
    selection_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_asset_ids",
            _as_string_tuple(
                self.source_asset_ids,
                field_name="operation_context.source_asset_ids",
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "geometry_revision_id",
            _required_string(
                self.geometry_revision_id,
                field_name="operation_context.geometry_revision_id",
            ),
        )
        object.__setattr__(
            self,
            "source_metadata_revision_id",
            _required_string(
                self.source_metadata_revision_id,
                field_name="operation_context.source_metadata_revision_id",
            ),
        )
        object.__setattr__(
            self,
            "align_revision_id",
            _required_string(
                self.align_revision_id,
                field_name="operation_context.align_revision_id",
            ),
        )
        object.__setattr__(
            self,
            "recipe_hash",
            _sha256(self.recipe_hash, field_name="operation_context.recipe_hash"),
        )
        if self.selection_hash is not None:
            object.__setattr__(
                self,
                "selection_hash",
                _sha256(
                    self.selection_hash,
                    field_name="operation_context.selection_hash",
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_asset_ids": list(self.source_asset_ids),
            "geometry_revision_id": self.geometry_revision_id,
            "source_metadata_revision_id": self.source_metadata_revision_id,
            "align_revision_id": self.align_revision_id,
            "recipe_hash": self.recipe_hash,
            "selection_hash": self.selection_hash,
        }


@dataclass(frozen=True, slots=True)
class DerivedRecord:
    id: str
    type: str
    geometry_revision_id: str
    align_revision_id: str
    depends_on_record_ids: tuple[str, ...]
    geometry_ref: str
    recipe: Mapping[str, Any]
    recipe_hash: str
    selection_hash: str | None
    qc: Mapping[str, Any]
    lifecycle_status: RecordLifecycleStatus | str
    created_at: str
    operator: str
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_string(self.id, field_name="record.id"))
        object.__setattr__(self, "type", _required_string(self.type, field_name="record.type"))
        object.__setattr__(
            self,
            "geometry_revision_id",
            _required_string(
                self.geometry_revision_id,
                field_name="record.geometry_revision_id",
            ),
        )
        object.__setattr__(
            self,
            "align_revision_id",
            _required_string(self.align_revision_id, field_name="record.align_revision_id"),
        )
        object.__setattr__(
            self,
            "depends_on_record_ids",
            _as_string_tuple(
                self.depends_on_record_ids,
                field_name="record.depends_on_record_ids",
            ),
        )
        object.__setattr__(
            self,
            "geometry_ref",
            _required_string(self.geometry_ref, field_name="record.geometry_ref"),
        )
        object.__setattr__(self, "recipe", _freeze_mapping(self.recipe, field_name="record.recipe"))
        object.__setattr__(
            self,
            "recipe_hash",
            _sha256(self.recipe_hash, field_name="record.recipe_hash"),
        )
        if canonical_recipe_hash(self.recipe) != self.recipe_hash:
            raise ArtifactDocumentError(
                "record.recipe_hash must match the canonical recipe bytes"
            )
        if self.selection_hash is not None:
            object.__setattr__(
                self,
                "selection_hash",
                _sha256(self.selection_hash, field_name="record.selection_hash"),
            )
        object.__setattr__(self, "qc", _freeze_mapping(self.qc, field_name="record.qc"))
        try:
            lifecycle = RecordLifecycleStatus(self.lifecycle_status)
        except (TypeError, ValueError) as exc:
            raise ArtifactDocumentError(
                f"unsupported record.lifecycle_status: {self.lifecycle_status!r}"
            ) from exc
        object.__setattr__(self, "lifecycle_status", lifecycle)
        object.__setattr__(
            self,
            "created_at",
            _canonical_timestamp(self.created_at, field_name="record.created_at"),
        )
        object.__setattr__(
            self,
            "operator",
            _required_string(self.operator, field_name="record.operator"),
        )
        object.__setattr__(
            self,
            "extensions",
            _freeze_extensions(self.extensions, field_name="record.extensions"),
        )

    @classmethod
    def from_operation_context(
        cls,
        *,
        id: str,
        type: str,
        context: OperationContext,
        geometry_ref: str,
        recipe: Mapping[str, Any],
        qc: Mapping[str, Any],
        lifecycle_status: RecordLifecycleStatus | str,
        created_at: str,
        operator: str,
        depends_on_record_ids: Sequence[str] = (),
        extensions: Mapping[str, Any] | None = None,
    ) -> "DerivedRecord":
        if not isinstance(context, OperationContext):
            raise ArtifactDocumentError("context must be an OperationContext")
        return cls(
            id=id,
            type=type,
            geometry_revision_id=context.geometry_revision_id,
            align_revision_id=context.align_revision_id,
            depends_on_record_ids=tuple(depends_on_record_ids),
            geometry_ref=geometry_ref,
            recipe=recipe,
            recipe_hash=context.recipe_hash,
            selection_hash=context.selection_hash,
            qc=qc,
            lifecycle_status=lifecycle_status,
            created_at=created_at,
            operator=operator,
            extensions=dict(extensions or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "geometry_revision_id": self.geometry_revision_id,
            "align_revision_id": self.align_revision_id,
            "depends_on_record_ids": list(self.depends_on_record_ids),
            "geometry_ref": self.geometry_ref,
            "recipe": _thaw_json(self.recipe),
            "recipe_hash": self.recipe_hash,
            "selection_hash": self.selection_hash,
            "qc": _thaw_json(self.qc),
            "lifecycle_status": RecordLifecycleStatus(self.lifecycle_status).value,
            "created_at": self.created_at,
            "operator": self.operator,
            "extensions": _thaw_json(self.extensions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "DerivedRecord":
        _exact_keys(
            data,
            {
                "id",
                "type",
                "geometry_revision_id",
                "align_revision_id",
                "depends_on_record_ids",
                "geometry_ref",
                "recipe",
                "recipe_hash",
                "selection_hash",
                "qc",
                "lifecycle_status",
                "created_at",
                "operator",
                "extensions",
            },
            model_name="derived_record",
        )
        return cls(**dict(data))  # type: ignore[arg-type]


def _index_by_id(items: Iterable[Any], *, label: str) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for item in items:
        item_id = getattr(item, "id", None)
        if not isinstance(item_id, str):
            raise ArtifactDocumentError(f"{label} contains an item without a string id")
        if item_id in index:
            raise ArtifactDocumentError(f"duplicate {label} id: {item_id!r}")
        index[item_id] = item
    return index


def _validate_parent_graph(
    index: Mapping[str, Any],
    *,
    label: str,
    compatibility_field: str,
) -> None:
    for item_id, item in index.items():
        parent_id = getattr(item, "parent_id", None)
        if parent_id is None:
            continue
        if parent_id not in index:
            raise ArtifactDocumentError(
                f"{label} {item_id!r} references missing parent {parent_id!r}"
            )
        if parent_id == item_id:
            raise ArtifactDocumentError(f"{label} {item_id!r} cannot parent itself")
        parent = index[parent_id]
        if getattr(parent, compatibility_field) != getattr(item, compatibility_field):
            raise ArtifactDocumentError(
                f"{label} {item_id!r} and parent {parent_id!r} have incompatible "
                f"{compatibility_field}"
            )

    # Iterative tri-color walk avoids recursion limits on long revision chains.
    color: dict[str, int] = {item_id: 0 for item_id in index}
    for start_id in sorted(index):
        if color[start_id] == 2:
            continue
        path: list[str] = []
        current_id: str | None = start_id
        while current_id is not None:
            state = color[current_id]
            if state == 2:
                break
            if state == 1:
                cycle_start = path.index(current_id) if current_id in path else 0
                cycle = path[cycle_start:] + [current_id]
                raise ArtifactDocumentError(
                    f"{label} parent cycle: {' -> '.join(cycle)}"
                )
            color[current_id] = 1
            path.append(current_id)
            current_id = getattr(index[current_id], "parent_id", None)
        for visited_id in path:
            color[visited_id] = 2


def _record_topological_order(
    records: Mapping[str, DerivedRecord],
) -> tuple[str, ...]:
    indegree = {record_id: 0 for record_id in records}
    children: dict[str, list[str]] = {record_id: [] for record_id in records}
    for record_id, record in records.items():
        for dependency_id in record.depends_on_record_ids:
            if dependency_id not in records:
                continue
            indegree[record_id] += 1
            children[dependency_id].append(record_id)

    ready = [record_id for record_id, count in indegree.items() if count == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        current = heapq.heappop(ready)
        order.append(current)
        for child in sorted(children[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(records):
        cyclic_ids = sorted(record_id for record_id, count in indegree.items() if count > 0)
        raise ArtifactDocumentError(
            f"derived record dependency cycle: {', '.join(cyclic_ids)}"
        )
    return tuple(order)


def canonical_recipe_hash(recipe: Mapping[str, Any]) -> str:
    """Return an RFC 8785 semantic SHA-256 for a strict JSON recipe object."""

    normalized = _thaw_json(_freeze_mapping(recipe, field_name="recipe"))
    try:
        return canonical_json_sha256(normalized)
    except CanonicalJSONError as exc:
        raise ArtifactDocumentError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class ArtifactDocument:
    schema_version: str
    document_id: str
    software_version: str
    source_assets: tuple[SourceAsset, ...]
    geometry_revisions: tuple[GeometryRevision, ...]
    source_metadata_revisions: tuple[SourceMetadataRevision, ...]
    align_revisions: tuple[AlignRevision, ...]
    active_source_metadata_revision_id: str | None
    active_align_revision_id: str | None
    records: tuple[DerivedRecord, ...]
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_DOCUMENT_SCHEMA_VERSION:
            raise ArtifactDocumentError(
                f"unsupported artifact document schema version: {self.schema_version!r}"
            )
        object.__setattr__(
            self,
            "document_id",
            _required_string(self.document_id, field_name="document_id"),
        )
        object.__setattr__(
            self,
            "software_version",
            _required_string(self.software_version, field_name="software_version"),
        )

        collections: tuple[tuple[str, type[Any]], ...] = (
            ("source_assets", SourceAsset),
            ("geometry_revisions", GeometryRevision),
            ("source_metadata_revisions", SourceMetadataRevision),
            ("align_revisions", AlignRevision),
            ("records", DerivedRecord),
        )
        for field_name, expected_type in collections:
            raw_items = tuple(getattr(self, field_name))
            if not all(isinstance(item, expected_type) for item in raw_items):
                raise ArtifactDocumentError(
                    f"{field_name} must contain only {expected_type.__name__} values"
                )
            object.__setattr__(
                self,
                field_name,
                tuple(sorted(raw_items, key=lambda item: item.id)),
            )

        object.__setattr__(
            self,
            "active_source_metadata_revision_id",
            _optional_id(
                self.active_source_metadata_revision_id,
                field_name="active_source_metadata_revision_id",
            ),
        )
        object.__setattr__(
            self,
            "active_align_revision_id",
            _optional_id(
                self.active_align_revision_id,
                field_name="active_align_revision_id",
            ),
        )
        object.__setattr__(
            self,
            "extensions",
            _freeze_extensions(self.extensions, field_name="document.extensions"),
        )
        self._validate()

    @classmethod
    def empty(cls, *, document_id: str, software_version: str) -> "ArtifactDocument":
        return cls(
            schema_version=ARTIFACT_DOCUMENT_SCHEMA_VERSION,
            document_id=document_id,
            software_version=software_version,
            source_assets=(),
            geometry_revisions=(),
            source_metadata_revisions=(),
            align_revisions=(),
            active_source_metadata_revision_id=None,
            active_align_revision_id=None,
            records=(),
        )

    def _validate(self) -> None:
        source_index = _index_by_id(self.source_assets, label="source_asset")
        geometry_index = _index_by_id(self.geometry_revisions, label="geometry_revision")
        metadata_index = _index_by_id(
            self.source_metadata_revisions,
            label="source_metadata_revision",
        )
        align_index = _index_by_id(self.align_revisions, label="align_revision")
        record_index = _index_by_id(self.records, label="derived_record")

        all_ids: dict[str, str] = {}
        for label, index in (
            ("source_asset", source_index),
            ("geometry_revision", geometry_index),
            ("source_metadata_revision", metadata_index),
            ("align_revision", align_index),
            ("derived_record", record_index),
        ):
            for item_id in index:
                previous = all_ids.get(item_id)
                if previous is not None:
                    raise ArtifactDocumentError(
                        f"id {item_id!r} is shared by {previous} and {label}"
                    )
                all_ids[item_id] = label

        for geometry in self.geometry_revisions:
            for source_id in geometry.source_asset_ids:
                if source_id not in source_index:
                    raise ArtifactDocumentError(
                        f"geometry revision {geometry.id!r} references missing source asset "
                        f"{source_id!r}"
                    )

        for metadata in self.source_metadata_revisions:
            if metadata.geometry_revision_id not in geometry_index:
                raise ArtifactDocumentError(
                    f"metadata revision {metadata.id!r} references missing geometry revision "
                    f"{metadata.geometry_revision_id!r}"
                )
        _validate_parent_graph(
            metadata_index,
            label="metadata revision",
            compatibility_field="geometry_revision_id",
        )

        for align in self.align_revisions:
            if align.source_metadata_revision_id not in metadata_index:
                raise ArtifactDocumentError(
                    f"align revision {align.id!r} references missing metadata revision "
                    f"{align.source_metadata_revision_id!r}"
                )
        _validate_parent_graph(
            align_index,
            label="align revision",
            compatibility_field="source_metadata_revision_id",
        )

        if metadata_index:
            if self.active_source_metadata_revision_id not in metadata_index:
                raise ArtifactDocumentError(
                    "documents with metadata revisions require a valid active metadata ID"
                )
        elif self.active_source_metadata_revision_id is not None:
            raise ArtifactDocumentError("active metadata ID exists without metadata revisions")

        if self.active_align_revision_id is not None:
            active_align = align_index.get(self.active_align_revision_id)
            if active_align is None:
                raise ArtifactDocumentError(
                    f"active align revision {self.active_align_revision_id!r} does not exist"
                )
            if active_align.source_metadata_revision_id != self.active_source_metadata_revision_id:
                raise ArtifactDocumentError(
                    "active align and active metadata must describe one atomic context"
                )

        for record in self.records:
            align = align_index.get(record.align_revision_id)
            if align is None:
                raise ArtifactDocumentError(
                    f"record {record.id!r} references missing align revision "
                    f"{record.align_revision_id!r}"
                )
            if record.geometry_revision_id not in geometry_index:
                raise ArtifactDocumentError(
                    f"record {record.id!r} references missing geometry revision "
                    f"{record.geometry_revision_id!r}"
                )
            metadata = metadata_index[align.source_metadata_revision_id]
            if record.geometry_revision_id != metadata.geometry_revision_id:
                raise ArtifactDocumentError(
                    f"record {record.id!r} geometry does not match its align context"
                )
            for dependency_id in record.depends_on_record_ids:
                dependency = record_index.get(dependency_id)
                if dependency is None:
                    # Missing dependencies remain loadable and report an explicit
                    # freshness state instead of being silently discarded.
                    continue
                if dependency.align_revision_id != record.align_revision_id:
                    raise ArtifactDocumentError(
                        f"record {record.id!r} has implicit cross-align dependency "
                        f"{dependency_id!r}; an explicit transform record is required"
                    )
        _record_topological_order(record_index)

    @property
    def source_asset_index(self) -> dict[str, SourceAsset]:
        return _index_by_id(self.source_assets, label="source_asset")

    @property
    def geometry_revision_index(self) -> dict[str, GeometryRevision]:
        return _index_by_id(self.geometry_revisions, label="geometry_revision")

    @property
    def source_metadata_revision_index(self) -> dict[str, SourceMetadataRevision]:
        return _index_by_id(
            self.source_metadata_revisions,
            label="source_metadata_revision",
        )

    @property
    def align_revision_index(self) -> dict[str, AlignRevision]:
        return _index_by_id(self.align_revisions, label="align_revision")

    @property
    def record_index(self) -> dict[str, DerivedRecord]:
        return _index_by_id(self.records, label="derived_record")

    def append_source_asset(self, asset: SourceAsset) -> "ArtifactDocument":
        return replace(self, source_assets=(*self.source_assets, asset))

    def append_geometry_revision(self, revision: GeometryRevision) -> "ArtifactDocument":
        return replace(self, geometry_revisions=(*self.geometry_revisions, revision))

    def append_source_metadata_revision(
        self,
        revision: SourceMetadataRevision,
        *,
        activate: bool = True,
    ) -> "ArtifactDocument":
        return replace(
            self,
            source_metadata_revisions=(*self.source_metadata_revisions, revision),
            active_source_metadata_revision_id=(
                revision.id if activate else self.active_source_metadata_revision_id
            ),
            # Metadata changes invalidate the active Align context.  Activating
            # an Align later restores metadata+Align atomically.
            active_align_revision_id=(None if activate else self.active_align_revision_id),
        )

    def activate_source_metadata_revision(self, revision_id: str) -> "ArtifactDocument":
        revision_id = _required_string(revision_id, field_name="revision_id")
        if revision_id not in self.source_metadata_revision_index:
            raise ArtifactDocumentError(f"metadata revision {revision_id!r} does not exist")
        return replace(
            self,
            active_source_metadata_revision_id=revision_id,
            active_align_revision_id=None,
        )

    def append_align_revision(
        self,
        revision: AlignRevision,
        *,
        activate: bool = True,
    ) -> "ArtifactDocument":
        return replace(
            self,
            align_revisions=(*self.align_revisions, revision),
            active_source_metadata_revision_id=(
                revision.source_metadata_revision_id
                if activate
                else self.active_source_metadata_revision_id
            ),
            active_align_revision_id=(revision.id if activate else self.active_align_revision_id),
        )

    def activate_align_revision(self, revision_id: str) -> "ArtifactDocument":
        revision_id = _required_string(revision_id, field_name="revision_id")
        revision = self.align_revision_index.get(revision_id)
        if revision is None:
            raise ArtifactDocumentError(f"align revision {revision_id!r} does not exist")
        return replace(
            self,
            active_source_metadata_revision_id=revision.source_metadata_revision_id,
            active_align_revision_id=revision.id,
        )

    def append_record(self, record: DerivedRecord) -> "ArtifactDocument":
        return replace(self, records=(*self.records, record))

    def capture_operation_context(
        self,
        *,
        recipe: Mapping[str, Any],
        selection_hash: str | None = None,
    ) -> OperationContext:
        if self.active_align_revision_id is None:
            raise ArtifactDocumentError("an active Align is required to capture an operation")
        align = self.align_revision_index[self.active_align_revision_id]
        metadata = self.source_metadata_revision_index[align.source_metadata_revision_id]
        metadata.require_confirmed_matrix()
        geometry = self.geometry_revision_index[metadata.geometry_revision_id]
        return OperationContext(
            source_asset_ids=geometry.source_asset_ids,
            geometry_revision_id=geometry.id,
            source_metadata_revision_id=metadata.id,
            align_revision_id=align.id,
            recipe_hash=canonical_recipe_hash(recipe),
            selection_hash=selection_hash,
        )

    def append_record_from_context(
        self,
        *,
        context: OperationContext,
        id: str,
        type: str,
        geometry_ref: str,
        recipe: Mapping[str, Any],
        qc: Mapping[str, Any],
        lifecycle_status: RecordLifecycleStatus | str,
        created_at: str,
        operator: str,
        depends_on_record_ids: Sequence[str] = (),
        extensions: Mapping[str, Any] | None = None,
    ) -> "ArtifactDocument":
        geometry = self.geometry_revision_index.get(context.geometry_revision_id)
        metadata = self.source_metadata_revision_index.get(
            context.source_metadata_revision_id
        )
        align = self.align_revision_index.get(context.align_revision_id)
        if geometry is None or metadata is None or align is None:
            raise ArtifactDocumentError("operation context references a missing revision")
        if tuple(geometry.source_asset_ids) != tuple(context.source_asset_ids):
            raise ArtifactDocumentError("operation context source assets do not match geometry")
        if metadata.geometry_revision_id != geometry.id:
            raise ArtifactDocumentError("operation context metadata does not match geometry")
        if align.source_metadata_revision_id != metadata.id:
            raise ArtifactDocumentError("operation context Align does not match metadata")
        if canonical_recipe_hash(recipe) != context.recipe_hash:
            raise ArtifactDocumentError("completed recipe does not match captured recipe hash")
        return self.append_record(
            DerivedRecord.from_operation_context(
                id=id,
                type=type,
                context=context,
                geometry_ref=geometry_ref,
                recipe=recipe,
                qc=qc,
                lifecycle_status=lifecycle_status,
                created_at=created_at,
                operator=operator,
                depends_on_record_ids=depends_on_record_ids,
                extensions=extensions,
            )
        )

    def record_freshnesses(self) -> dict[str, RecordFreshness]:
        records = self.record_index
        aligns = self.align_revision_index
        statuses: dict[str, RecordFreshness] = {}
        for record_id in _record_topological_order(records):
            record = records[record_id]
            align = aligns[record.align_revision_id]
            if any(
                dependency_id not in records
                for dependency_id in record.depends_on_record_ids
            ):
                statuses[record_id] = RecordFreshness.MISSING_DEPENDENCY
                continue
            if align.source_metadata_revision_id != self.active_source_metadata_revision_id:
                statuses[record_id] = RecordFreshness.STALE_METADATA
                continue
            if self.active_align_revision_id is None or record.align_revision_id != self.active_align_revision_id:
                statuses[record_id] = RecordFreshness.STALE_ALIGNMENT
                continue
            blocked = False
            for dependency_id in record.depends_on_record_ids:
                dependency = records[dependency_id]
                if dependency.lifecycle_status is not RecordLifecycleStatus.READY:
                    blocked = True
                    break
                if statuses[dependency_id] is not RecordFreshness.FRESH:
                    blocked = True
                    break
            statuses[record_id] = (
                RecordFreshness.BLOCKED_DEPENDENCY
                if blocked
                else RecordFreshness.FRESH
            )
        return statuses

    def record_freshness(self, record_id: str) -> RecordFreshness:
        record_id = _required_string(record_id, field_name="record_id")
        if record_id not in self.record_index:
            raise ArtifactDocumentError(f"record {record_id!r} does not exist")
        return self.record_freshnesses()[record_id]

    def active_canonical_matrix(self) -> np.ndarray:
        if self.active_align_revision_id is None:
            raise ArtifactDocumentError("an active Align is required for materialization")
        align = self.align_revision_index[self.active_align_revision_id]
        metadata = self.source_metadata_revision_index[align.source_metadata_revision_id]
        return align.matrix @ metadata.require_confirmed_matrix()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "software_version": self.software_version,
            "source_assets": [item.to_dict() for item in self.source_assets],
            "geometry_revisions": [item.to_dict() for item in self.geometry_revisions],
            "source_metadata_revisions": [
                item.to_dict() for item in self.source_metadata_revisions
            ],
            "align_revisions": [item.to_dict() for item in self.align_revisions],
            "active_source_metadata_revision_id": self.active_source_metadata_revision_id,
            "active_align_revision_id": self.active_align_revision_id,
            "records": [item.to_dict() for item in self.records],
            "extensions": _thaw_json(self.extensions),
        }

    def canonical_json_bytes(self) -> bytes:
        """Serialize deterministic UTF-8/LF canonical document bytes."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ArtifactDocument":
        if not isinstance(data, Mapping):
            raise ArtifactDocumentError("artifact document must be a JSON object")
        _exact_keys(
            data,
            {
                "schema_version",
                "document_id",
                "software_version",
                "source_assets",
                "geometry_revisions",
                "source_metadata_revisions",
                "align_revisions",
                "active_source_metadata_revision_id",
                "active_align_revision_id",
                "records",
                "extensions",
            },
            model_name="artifact_document",
        )

        def parse_list(
            key: str,
            model: type[Any],
        ) -> tuple[Any, ...]:
            raw = data[key]
            if not isinstance(raw, list):
                raise ArtifactDocumentError(f"artifact_document.{key} must be an array")
            output: list[Any] = []
            for index, item in enumerate(raw):
                if not isinstance(item, Mapping):
                    raise ArtifactDocumentError(
                        f"artifact_document.{key}[{index}] must be an object"
                    )
                output.append(model.from_dict(item))
            return tuple(output)

        return cls(
            schema_version=data["schema_version"],  # type: ignore[arg-type]
            document_id=data["document_id"],  # type: ignore[arg-type]
            software_version=data["software_version"],  # type: ignore[arg-type]
            source_assets=parse_list("source_assets", SourceAsset),
            geometry_revisions=parse_list("geometry_revisions", GeometryRevision),
            source_metadata_revisions=parse_list(
                "source_metadata_revisions",
                SourceMetadataRevision,
            ),
            align_revisions=parse_list("align_revisions", AlignRevision),
            active_source_metadata_revision_id=data[
                "active_source_metadata_revision_id"
            ],  # type: ignore[arg-type]
            active_align_revision_id=data["active_align_revision_id"],  # type: ignore[arg-type]
            records=parse_list("records", DerivedRecord),
            extensions=data["extensions"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "ArtifactDocument":
        try:
            raw = payload.decode("utf-8", errors="strict")
            data = json.loads(
                raw,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ArtifactDocumentError(f"invalid JSON constant: {value}")
                ),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except ArtifactDocumentError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise ArtifactDocumentError(f"invalid artifact document JSON: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ArtifactDocumentError("artifact document JSON must contain an object")
        return cls.from_dict(data)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ArtifactDocumentError(f"duplicate JSON object key: {key!r}")
        output[key] = value
    return output
