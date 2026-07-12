"""Deterministic planar vector payloads for Cutline and Outline records.

The payload coordinate system is an explicit orthonormal plane embedded in
canonical world millimetres.  Paths are stored as 2D millimetre coordinates
inside that plane, while the frame preserves their relationship to the active
ArtifactDocument Align revision.  Payload bytes are content-addressed and are
kept separate from the immutable record manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .canonical_json import (
    CanonicalJSONError,
    canonical_json_bytes as rfc8785_json_bytes,
    canonical_json_sha256,
)
from .artifact_document import (
    ArtifactDocument,
    ArtifactDocumentError,
    DerivedRecord,
    OperationContext,
    RecordLifecycleStatus,
)


VECTOR_PAYLOAD_SCHEMA_VERSION = "1.0.0"
VECTOR_COORDINATE_SPACE = "canonical_mm_planar/v1"
VECTOR_PAYLOAD_MEDIA_TYPE = "application/vnd.archmeshrubbing.vector+json"
VECTOR_GEOMETRY_REF_PREFIX = "urn:archmeshrubbing:vector-payload:sha256:"
VECTOR_PAYLOAD_EXTENSION_KEY = "org.archmeshrubbing:vector-payload-v1"
MAX_VECTOR_PATHS = 4096
MAX_VECTOR_POINTS = 250_000
MAX_VECTOR_PAYLOAD_BYTES = 16 * 1024 * 1024
VECTOR_QC_LENGTH_DECIMAL_PLACES = 12

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactVectorRecordError(ValueError):
    """A vector payload or record command violates the M0-4 contract."""


class VectorRecordKind(str, Enum):
    CUTLINE = "cutline"
    OUTLINE = "outline"

    @property
    def record_type(self) -> str:
        return f"vector.{self.value}.v1"


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactVectorRecordError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ArtifactVectorRecordError(
            f"{field_name} must not contain surrounding whitespace"
        )
    return value


def _exact_keys(data: Mapping[str, object], expected: set[str], model_name: str) -> None:
    observed = set(data)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise ArtifactVectorRecordError(
            f"{model_name} is missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise ArtifactVectorRecordError(
            f"{model_name} has unknown fields: {', '.join(unknown)}"
        )


def _finite_vector(
    value: object,
    *,
    size: int,
    field_name: str,
) -> tuple[float, ...]:
    try:
        raw = np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ArtifactVectorRecordError(
            f"{field_name} must contain exactly {size} finite numbers"
        ) from exc
    if raw.size != size:
        raise ArtifactVectorRecordError(
            f"{field_name} must contain exactly {size} finite numbers"
        )
    numbers: list[float] = []
    for item in raw:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, float, np.integer, np.floating)
        ):
            raise ArtifactVectorRecordError(
                f"{field_name} must contain exactly {size} finite numbers"
            )
        number = float(item)
        if not math.isfinite(number):
            raise ArtifactVectorRecordError(
                f"{field_name} must contain exactly {size} finite numbers"
            )
        numbers.append(number)
    array = np.asarray(numbers, dtype=np.float64)
    array[array == 0.0] = 0.0
    return tuple(float(item) for item in array)


@dataclass(frozen=True, slots=True)
class PlanarFrame:
    """Right-handed orthonormal plane embedded in canonical world millimetres."""

    origin_world_mm: tuple[float, float, float]
    u_axis_world: tuple[float, float, float]
    v_axis_world: tuple[float, float, float]
    normal_world: tuple[float, float, float]

    def __post_init__(self) -> None:
        origin = _finite_vector(
            self.origin_world_mm,
            size=3,
            field_name="frame.origin_world_mm",
        )
        u_axis = _finite_vector(
            self.u_axis_world,
            size=3,
            field_name="frame.u_axis_world",
        )
        v_axis = _finite_vector(
            self.v_axis_world,
            size=3,
            field_name="frame.v_axis_world",
        )
        normal = _finite_vector(
            self.normal_world,
            size=3,
            field_name="frame.normal_world",
        )
        u = np.asarray(u_axis, dtype=np.float64)
        v = np.asarray(v_axis, dtype=np.float64)
        n = np.asarray(normal, dtype=np.float64)
        for name, axis in (("u_axis_world", u), ("v_axis_world", v), ("normal_world", n)):
            if not np.isclose(float(np.linalg.norm(axis)), 1.0, rtol=0.0, atol=1e-10):
                raise ArtifactVectorRecordError(f"frame.{name} must be a unit vector")
        if not np.isclose(float(np.dot(u, v)), 0.0, rtol=0.0, atol=1e-10):
            raise ArtifactVectorRecordError("frame u/v axes must be orthogonal")
        if not np.allclose(np.cross(u, v), n, rtol=0.0, atol=1e-10):
            raise ArtifactVectorRecordError(
                "frame axes must be right-handed with cross(u, v) == normal"
            )
        object.__setattr__(self, "origin_world_mm", origin)
        object.__setattr__(self, "u_axis_world", u_axis)
        object.__setattr__(self, "v_axis_world", v_axis)
        object.__setattr__(self, "normal_world", normal)

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_world_mm": list(self.origin_world_mm),
            "u_axis_world": list(self.u_axis_world),
            "v_axis_world": list(self.v_axis_world),
            "normal_world": list(self.normal_world),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PlanarFrame":
        if not isinstance(data, Mapping):
            raise ArtifactVectorRecordError("frame must be an object")
        _exact_keys(
            data,
            {"origin_world_mm", "u_axis_world", "v_axis_world", "normal_world"},
            "frame",
        )
        return cls(
            origin_world_mm=_finite_vector(
                data["origin_world_mm"], size=3, field_name="frame.origin_world_mm"
            ),  # type: ignore[arg-type]
            u_axis_world=_finite_vector(
                data["u_axis_world"], size=3, field_name="frame.u_axis_world"
            ),  # type: ignore[arg-type]
            v_axis_world=_finite_vector(
                data["v_axis_world"], size=3, field_name="frame.v_axis_world"
            ),  # type: ignore[arg-type]
            normal_world=_finite_vector(
                data["normal_world"], size=3, field_name="frame.normal_world"
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class VectorPath:
    id: str
    role: str
    closed: bool
    points_mm: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        path_id = _required_string(self.id, "path.id")
        role = _required_string(self.role, "path.role")
        if type(self.closed) is not bool:
            raise ArtifactVectorRecordError("path.closed must be a boolean")
        try:
            raw_values = np.asarray(self.points_mm, dtype=object)
        except (TypeError, ValueError) as exc:
            raise ArtifactVectorRecordError("path.points_mm must be an Nx2 array") from exc
        if raw_values.ndim != 2 or raw_values.shape[1] != 2:
            raise ArtifactVectorRecordError("path.points_mm must be a finite Nx2 array")
        raw = np.asarray(
            [
                _finite_vector(point, size=2, field_name="path.points_mm[]")
                for point in raw_values
            ],
            dtype=np.float64,
        )
        raw = raw.copy()
        raw[raw == 0.0] = 0.0

        normalized: list[np.ndarray] = []
        for point in raw:
            if normalized and np.array_equal(normalized[-1], point):
                continue
            normalized.append(point.copy())
        if self.closed and len(normalized) >= 2 and np.array_equal(
            normalized[0], normalized[-1]
        ):
            normalized.pop()
        if (
            not self.closed
            and len(normalized) >= 2
            and np.array_equal(normalized[0], normalized[-1])
        ):
            raise ArtifactVectorRecordError(
                "open path endpoints must be distinct"
            )
        minimum = 3 if self.closed else 2
        if len(normalized) < minimum:
            raise ArtifactVectorRecordError(
                f"{'closed' if self.closed else 'open'} path requires at least {minimum} points"
            )
        points = tuple((float(point[0]), float(point[1])) for point in normalized)
        object.__setattr__(self, "id", path_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "points_mm", points)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "closed": self.closed,
            "points_mm": [list(point) for point in self.points_mm],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "VectorPath":
        if not isinstance(data, Mapping):
            raise ArtifactVectorRecordError("path must be an object")
        _exact_keys(data, {"id", "role", "closed", "points_mm"}, "path")
        closed = data["closed"]
        if type(closed) is not bool:
            raise ArtifactVectorRecordError("path.closed must be a boolean")
        points = data["points_mm"]
        if not isinstance(points, (list, tuple)):
            raise ArtifactVectorRecordError("path.points_mm must be an array")
        return cls(
            id=_required_string(data["id"], "path.id"),
            role=_required_string(data["role"], "path.role"),
            closed=closed,
            points_mm=tuple(
                _finite_vector(point, size=2, field_name="path.points_mm[]")  # type: ignore[misc]
                for point in points
            ),
        )


def _path_signed_area(points: Sequence[tuple[float, float]]) -> float:
    # Translate before the shoelace sum so small artifacts retain their area
    # when canonical world coordinates contain a large survey offset.
    origin_x, origin_y = points[0]
    return 0.5 * math.fsum(
        (points[index][0] - origin_x)
        * (points[(index + 1) % len(points)][1] - origin_y)
        - (points[(index + 1) % len(points)][0] - origin_x)
        * (points[index][1] - origin_y)
        for index in range(len(points))
    )


def _canonical_path(path: VectorPath, *, kind: VectorRecordKind) -> VectorPath:
    points = list(path.points_mm)
    if path.closed:
        start = min(range(len(points)), key=lambda index: points[index])
        points = points[start:] + points[:start]
        area = _path_signed_area(points)
        if area == 0.0:
            raise ArtifactVectorRecordError("closed vector path must have non-zero area")
        wants_clockwise = kind is VectorRecordKind.OUTLINE and path.role == "hole"
        if (wants_clockwise and area > 0.0) or (not wants_clockwise and area < 0.0):
            points = [points[0], *reversed(points[1:])]
    else:
        reversed_points = list(reversed(points))
        if tuple(reversed_points) < tuple(points):
            points = reversed_points
    return VectorPath(
        id=path.id,
        role=path.role,
        closed=path.closed,
        points_mm=tuple(points),
    )


def _path_order_key(path: VectorPath) -> tuple[Any, ...]:
    minimum_x = min(point[0] for point in path.points_mm)
    minimum_y = min(point[1] for point in path.points_mm)
    maximum_x = max(point[0] for point in path.points_mm)
    maximum_y = max(point[1] for point in path.points_mm)
    role_order = {"section": 0, "exterior": 0, "hole": 1}[path.role]
    flattened = tuple(coordinate for point in path.points_mm for coordinate in point)
    return (
        role_order,
        0 if path.closed else 1,
        minimum_x,
        minimum_y,
        maximum_x,
        maximum_y,
        flattened,
        path.id,
    )


@dataclass(frozen=True, slots=True)
class VectorGeometryPayload:
    schema_version: str
    kind: VectorRecordKind | str
    coordinate_space: str
    frame: PlanarFrame
    paths: tuple[VectorPath, ...]

    def __post_init__(self) -> None:
        if self.schema_version != VECTOR_PAYLOAD_SCHEMA_VERSION:
            raise ArtifactVectorRecordError(
                f"unsupported vector payload schema: {self.schema_version!r}"
            )
        try:
            kind = VectorRecordKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ArtifactVectorRecordError(f"unsupported vector record kind: {self.kind!r}") from exc
        if self.coordinate_space != VECTOR_COORDINATE_SPACE:
            raise ArtifactVectorRecordError(
                f"unsupported vector coordinate space: {self.coordinate_space!r}"
            )
        if not isinstance(self.frame, PlanarFrame):
            raise ArtifactVectorRecordError("vector payload frame must be a PlanarFrame")
        paths = tuple(self.paths)
        if not paths or len(paths) > MAX_VECTOR_PATHS:
            raise ArtifactVectorRecordError(
                f"vector payload must contain 1..{MAX_VECTOR_PATHS} paths"
            )
        if any(not isinstance(path, VectorPath) for path in paths):
            raise ArtifactVectorRecordError("vector payload paths must be VectorPath values")
        ids = [path.id for path in paths]
        if len(set(ids)) != len(ids):
            raise ArtifactVectorRecordError("vector payload path IDs must be unique")
        point_count = sum(len(path.points_mm) for path in paths)
        if point_count > MAX_VECTOR_POINTS:
            raise ArtifactVectorRecordError(
                f"vector payload has too many points ({point_count} > {MAX_VECTOR_POINTS})"
            )
        if kind is VectorRecordKind.CUTLINE:
            if any(path.role != "section" for path in paths):
                raise ArtifactVectorRecordError(
                    "cutline vector paths must use the 'section' role"
                )
        else:
            if any(not path.closed for path in paths):
                raise ArtifactVectorRecordError("outline vector paths must be closed")
            if any(path.role not in {"exterior", "hole"} for path in paths):
                raise ArtifactVectorRecordError(
                    "outline vector paths must use 'exterior' or 'hole' roles"
                )
            if not any(path.role == "exterior" for path in paths):
                raise ArtifactVectorRecordError(
                    "outline vector payload requires at least one exterior path"
                )
        canonical_paths = tuple(
            sorted(
                (_canonical_path(path, kind=kind) for path in paths),
                key=_path_order_key,
            )
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "paths", canonical_paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": VectorRecordKind(self.kind).value,
            "coordinate_space": self.coordinate_space,
            "frame": self.frame.to_dict(),
            "paths": [path.to_dict() for path in self.paths],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "VectorGeometryPayload":
        if not isinstance(data, Mapping):
            raise ArtifactVectorRecordError("vector payload must be an object")
        _exact_keys(
            data,
            {"schema_version", "kind", "coordinate_space", "frame", "paths"},
            "vector payload",
        )
        raw_paths = data["paths"]
        if not isinstance(raw_paths, (list, tuple)):
            raise ArtifactVectorRecordError("vector payload paths must be an array")
        if any(not isinstance(path, Mapping) for path in raw_paths):
            raise ArtifactVectorRecordError("vector payload paths must contain only objects")
        raw_frame = data["frame"]
        if not isinstance(raw_frame, Mapping):
            raise ArtifactVectorRecordError("vector payload frame must be an object")
        payload = cls(
            schema_version=_required_string(data["schema_version"], "schema_version"),
            kind=_required_string(data["kind"], "kind"),
            coordinate_space=_required_string(data["coordinate_space"], "coordinate_space"),
            frame=PlanarFrame.from_dict(raw_frame),
            paths=tuple(
                VectorPath.from_dict(path) for path in raw_paths  # type: ignore[arg-type]
            ),
        )
        return payload

    def canonical_json_bytes(self) -> bytes:
        try:
            encoded = rfc8785_json_bytes(self.to_dict())
        except CanonicalJSONError as exc:
            raise ArtifactVectorRecordError(str(exc)) from exc
        if len(encoded) > MAX_VECTOR_PAYLOAD_BYTES:
            raise ArtifactVectorRecordError(
                "vector payload exceeds the "
                f"{MAX_VECTOR_PAYLOAD_BYTES}-byte inline safety limit"
            )
        return encoded

    @property
    def sha256(self) -> str:
        try:
            return canonical_json_sha256(self.to_dict())
        except CanonicalJSONError as exc:
            raise ArtifactVectorRecordError(str(exc)) from exc

    @property
    def geometry_ref(self) -> str:
        return f"{VECTOR_GEOMETRY_REF_PREFIX}{self.sha256}"

    def qc_summary(self) -> dict[str, Any]:
        all_points = np.vstack(
            [np.asarray(path.points_mm, dtype=np.float64) for path in self.paths]
        )
        minimum = all_points.min(axis=0)
        maximum = all_points.max(axis=0)
        segment_lengths: list[float] = []
        for path in self.paths:
            points = path.points_mm
            segment_lengths.extend(
                math.hypot(
                    points[index + 1][0] - points[index][0],
                    points[index + 1][1] - points[index][1],
                )
                for index in range(len(points) - 1)
            )
            if path.closed:
                segment_lengths.append(
                    math.hypot(
                        points[0][0] - points[-1][0],
                        points[0][1] - points[-1][1],
                    )
                )
        total_length = round(
            math.fsum(segment_lengths),
            VECTOR_QC_LENGTH_DECIMAL_PLACES,
        )
        if total_length == 0.0:
            total_length = 0.0
        return {
            "bounds_mm": [
                float(minimum[0]),
                float(minimum[1]),
                float(maximum[0]),
                float(maximum[1]),
            ],
            "closed_path_count": sum(1 for path in self.paths if path.closed),
            "coordinate_space": VECTOR_COORDINATE_SPACE,
            "finite": True,
            "path_count": len(self.paths),
            "payload_sha256": self.sha256,
            "point_count": sum(len(path.points_mm) for path in self.paths),
            "total_length_mm": total_length,
            "total_length_rounding_decimal_places": VECTOR_QC_LENGTH_DECIMAL_PLACES,
            "unit": "mm",
        }


def payload_sha256_from_geometry_ref(geometry_ref: str) -> str:
    reference = _required_string(geometry_ref, "geometry_ref")
    if not reference.startswith(VECTOR_GEOMETRY_REF_PREFIX):
        raise ArtifactVectorRecordError("geometry_ref is not a vector payload reference")
    digest = reference.removeprefix(VECTOR_GEOMETRY_REF_PREFIX)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ArtifactVectorRecordError("vector geometry_ref has an invalid SHA-256")
    return digest


def validate_vector_recipe(
    recipe: Mapping[str, Any],
    *,
    expected_kind: VectorRecordKind,
) -> None:
    if not isinstance(recipe, Mapping):
        raise ArtifactVectorRecordError("vector recipe must be an object")
    if recipe.get("kind") != expected_kind.value:
        raise ArtifactVectorRecordError(
            "recipe kind must match the vector payload kind"
        )
    for field_name in ("algorithm", "algorithm_version"):
        _required_string(recipe.get(field_name), f"recipe.{field_name}")


def _computed_payload_qc(payload: VectorGeometryPayload) -> dict[str, Any]:
    computed = payload.qc_summary()
    if VectorRecordKind(payload.kind) is VectorRecordKind.OUTLINE:
        # Local import avoids a module cycle: the topology validator consumes
        # VectorGeometryPayload but record persistence owns this trust boundary.
        from .artifact_outline_topology import (  # noqa: PLC0415
            ArtifactOutlineTopologyError,
            validate_outline_topology,
        )

        try:
            computed["outline_topology"] = validate_outline_topology(payload).to_dict()
        except ArtifactOutlineTopologyError as exc:
            raise ArtifactVectorRecordError(
                f"outline payload topology is invalid: {exc}"
            ) from exc
    return computed


def _validate_payload_recipe_contract(
    payload: VectorGeometryPayload,
    recipe: Mapping[str, Any],
) -> None:
    if VectorRecordKind(payload.kind) is not VectorRecordKind.OUTLINE:
        return
    from .artifact_outline_extractor import (  # noqa: PLC0415
        ArtifactVectorExtractionError,
        OUTLINE_ALGORITHM,
        validate_outline_record_contract,
    )

    # vector.outline.v1 is intentionally extensible to other open algorithms;
    # the stricter grid/ID/recipe proof applies when this implementation's
    # production algorithm is claimed. Every outline still receives the
    # algorithm-independent topology proof in _computed_payload_qc().
    if recipe.get("algorithm") != OUTLINE_ALGORITHM:
        return

    try:
        validate_outline_record_contract(payload, recipe)
    except ArtifactVectorExtractionError as exc:
        raise ArtifactVectorRecordError(
            f"outline payload/recipe contract is invalid: {exc}"
        ) from exc


def append_vector_record_from_context(
    document: ArtifactDocument,
    *,
    context: OperationContext,
    payload: VectorGeometryPayload,
    recipe: Mapping[str, Any],
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
    qc: Mapping[str, Any] | None = None,
) -> ArtifactDocument:
    """Append a ready vector record bound to a previously captured context."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactVectorRecordError("document must be an ArtifactDocument")
    if not isinstance(context, OperationContext):
        raise ArtifactVectorRecordError("context must be an OperationContext")
    if not isinstance(payload, VectorGeometryPayload):
        raise ArtifactVectorRecordError("payload must be a VectorGeometryPayload")
    validate_vector_recipe(
        recipe,
        expected_kind=VectorRecordKind(payload.kind),
    )
    _validate_payload_recipe_contract(payload, recipe)
    payload_bytes = payload.canonical_json_bytes()
    computed_qc = _computed_payload_qc(payload)
    for key, value in dict(qc or {}).items():
        if key in computed_qc and computed_qc[key] != value:
            raise ArtifactVectorRecordError(
                f"caller QC cannot override computed field {key!r}"
            )
        computed_qc[key] = value
    extensions = {
        VECTOR_PAYLOAD_EXTENSION_KEY: {
            "byte_length": len(payload_bytes),
            "media_type": VECTOR_PAYLOAD_MEDIA_TYPE,
            "payload": payload.to_dict(),
            "schema_version": VECTOR_PAYLOAD_SCHEMA_VERSION,
            "sha256": payload.sha256,
        }
    }
    try:
        return document.append_record_from_context(
            context=context,
            id=record_id,
            type=VectorRecordKind(payload.kind).record_type,
            geometry_ref=payload.geometry_ref,
            recipe=recipe,
            qc=computed_qc,
            lifecycle_status=RecordLifecycleStatus.READY,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=depends_on_record_ids,
            extensions=extensions,
        )
    except ArtifactDocumentError as exc:
        raise ArtifactVectorRecordError(str(exc)) from exc


def vector_payload_from_record(record: DerivedRecord) -> VectorGeometryPayload:
    """Resolve and verify the bounded inline payload of one known vector record."""

    if not isinstance(record, DerivedRecord):
        raise ArtifactVectorRecordError("record must be a DerivedRecord")
    known_types = {kind.record_type: kind for kind in VectorRecordKind}
    expected_kind = known_types.get(record.type)
    if expected_kind is None:
        raise ArtifactVectorRecordError(f"record is not a known vector type: {record.type!r}")
    descriptor = record.extensions.get(VECTOR_PAYLOAD_EXTENSION_KEY)
    if not isinstance(descriptor, Mapping):
        raise ArtifactVectorRecordError("vector record has no inline payload descriptor")
    _exact_keys(
        descriptor,
        {"byte_length", "media_type", "payload", "schema_version", "sha256"},
        "vector payload descriptor",
    )
    if descriptor.get("media_type") != VECTOR_PAYLOAD_MEDIA_TYPE:
        raise ArtifactVectorRecordError("vector payload media_type is invalid")
    if descriptor.get("schema_version") != VECTOR_PAYLOAD_SCHEMA_VERSION:
        raise ArtifactVectorRecordError("vector payload descriptor schema is invalid")
    raw_payload = descriptor.get("payload")
    if not isinstance(raw_payload, Mapping):
        raise ArtifactVectorRecordError("vector payload descriptor payload must be an object")
    payload = VectorGeometryPayload.from_dict(raw_payload)
    payload_bytes = payload.canonical_json_bytes()
    byte_length = descriptor.get("byte_length")
    if type(byte_length) is not int or byte_length != len(payload_bytes):
        raise ArtifactVectorRecordError("vector payload byte_length does not match payload")
    digest = descriptor.get("sha256")
    if not isinstance(digest, str) or digest != payload.sha256:
        raise ArtifactVectorRecordError("vector payload SHA-256 does not match payload")
    if record.geometry_ref != payload.geometry_ref:
        raise ArtifactVectorRecordError("vector record geometry_ref does not match payload")
    if expected_kind is not VectorRecordKind(payload.kind):
        raise ArtifactVectorRecordError("vector record type does not match payload kind")
    validate_vector_recipe(record.recipe, expected_kind=expected_kind)
    _validate_payload_recipe_contract(payload, record.recipe)
    thawed_qc = record.to_dict()["qc"]
    assert isinstance(thawed_qc, dict)
    for key, expected_value in _computed_payload_qc(payload).items():
        if thawed_qc.get(key) != expected_value:
            raise ArtifactVectorRecordError(
                f"vector record QC field {key!r} does not match payload"
            )
    return payload


def validate_vector_records(document: ArtifactDocument) -> None:
    """Strictly validate every known vector record embedded in a document."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactVectorRecordError("document must be an ArtifactDocument")
    known_types = {kind.record_type for kind in VectorRecordKind}
    for record in document.records:
        if record.type in known_types:
            vector_payload_from_record(record)


__all__ = [
    "ArtifactVectorRecordError",
    "MAX_VECTOR_PATHS",
    "MAX_VECTOR_PAYLOAD_BYTES",
    "MAX_VECTOR_POINTS",
    "PlanarFrame",
    "VECTOR_COORDINATE_SPACE",
    "VECTOR_GEOMETRY_REF_PREFIX",
    "VECTOR_PAYLOAD_MEDIA_TYPE",
    "VECTOR_PAYLOAD_EXTENSION_KEY",
    "VECTOR_PAYLOAD_SCHEMA_VERSION",
    "VECTOR_QC_LENGTH_DECIMAL_PLACES",
    "VectorGeometryPayload",
    "VectorPath",
    "VectorRecordKind",
    "append_vector_record_from_context",
    "payload_sha256_from_geometry_ref",
    "validate_vector_records",
    "validate_vector_recipe",
    "vector_payload_from_record",
]
