"""Reproducible whole-mesh dimensions, area, and guarded volume records.

The legacy GUI can display convenient Trimesh statistics, but those values are
not durable scientific authority.  This module measures a fresh canonical-mm
projection, quantizes coordinates on an explicit micrometre grid, records the
complete topology audit, and publishes an immutable DerivedRecord receipt.

Volume deliberately fails closed.  It is emitted only for one connected,
closed, consistently oriented edge-manifold component with no duplicate or
quantization-degenerate faces.  The exact value is stored as a rational number
of cubic millimetres; the decimal value is only a presentation field.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_cancellation import (
    CancellationProbe,
    poll_cancellation,
    raise_if_cancelled,
)
from .artifact_document import (
    ArtifactDocument,
    ArtifactDocumentError,
    DerivedRecord,
    OperationContext,
    RecordLifecycleStatus,
    canonical_recipe_hash,
)
from .artifact_scene_adapter import ArtifactProjectionSnapshot
from .artifact_session import ArtifactSession, ArtifactSessionError
from .canonical_json import canonical_json_bytes, canonical_json_sha256


GEOMETRY_METRICS_ALGORITHM = "archmeshrubbing.quantized_triangle_metrics"
GEOMETRY_METRICS_ALGORITHM_VERSION = "1.0.0"
GEOMETRY_METRICS_SCHEMA_VERSION = "1.0.0"
GEOMETRY_METRICS_RECORD_TYPE = "measurement.geometry_metrics.v1"
GEOMETRY_METRICS_EXTENSION_KEY = "org.archmeshrubbing:geometry-metrics-v1"
GEOMETRY_METRICS_MEDIA_TYPE = (
    "application/vnd.archmeshrubbing.geometry-metrics-receipt+json"
)
GEOMETRY_METRICS_REF_PREFIX = (
    "urn:archmeshrubbing:geometry-metrics-receipt:sha256:"
)
GEOMETRY_METRICS_COORDINATE_SPACE = "canonical_aligned_mm/v1"
GEOMETRY_METRICS_SCOPE = "entire_active_geometry"
GEOMETRY_METRICS_ROUNDING = "round_ties_to_even"
GEOMETRY_METRICS_VOLUME_POLICY = (
    "single_closed_consistently_oriented_edge_manifold_component/v1"
)
DEFAULT_GEOMETRY_METRICS_GRID_UM = 1
SURFACE_AREA_DECIMAL_PLACES = 6
VOLUME_DECIMAL_PLACES = 9
MAX_GEOMETRY_METRICS_VERTICES = 5_000_000
MAX_GEOMETRY_METRICS_FACES = 2_000_000
MAX_GEOMETRY_METRICS_GRID_UM = 1_000
MAX_QUANTIZED_AXIS_EXTENT = 2_000_000_000
MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_991
MAX_GEOMETRY_METRICS_RECEIPT_BYTES = 64 * 1024
_AREA_QUANTUM = Decimal(1).scaleb(-SURFACE_AREA_DECIMAL_PLACES)
_VOLUME_QUANTUM = Decimal(1).scaleb(-VOLUME_DECIMAL_PLACES)
_UNSIGNED_INTEGER_RE = re.compile(r"^(0|[1-9][0-9]*)$")
_SIGNED_INTEGER_RE = re.compile(r"^(0|-?[1-9][0-9]*)$")
_AREA_DECIMAL_RE = re.compile(r"^(0|[1-9][0-9]*)\.[0-9]{6}$")
_VOLUME_DECIMAL_RE = re.compile(r"^(0|[1-9][0-9]*)\.[0-9]{9}$")
_DISPLACEMENT_DECIMAL_RE = re.compile(r"^(0|[1-9][0-9]*)\.[0-9]{6}$")


class ArtifactGeometryMetricsError(ValueError):
    """A geometry metrics computation or record violates its strict contract."""


def _exact_mapping(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactGeometryMetricsError(f"{name} must be an object")
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise ArtifactGeometryMetricsError(
            f"{name} is missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise ArtifactGeometryMetricsError(
            f"{name} has unknown fields: {', '.join(unknown)}"
        )
    return value


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactGeometryMetricsError(f"{name} must be an integer")
    number = int(value)
    if number < minimum or number > maximum:
        raise ArtifactGeometryMetricsError(
            f"{name} must be in the inclusive range {minimum}..{maximum}"
        )
    return number


def _decimal_string(
    value: object,
    *,
    name: str,
    pattern: re.Pattern[str],
) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ArtifactGeometryMetricsError(f"{name} has an invalid decimal form")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ArtifactGeometryMetricsError(f"{name} is not a decimal") from exc
    if not number.is_finite() or number < 0:
        raise ArtifactGeometryMetricsError(f"{name} must be non-negative")
    return value


def _integer_string(value: object, *, name: str, signed: bool) -> str:
    pattern = _SIGNED_INTEGER_RE if signed else _UNSIGNED_INTEGER_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ArtifactGeometryMetricsError(f"{name} has an invalid integer form")
    return value


def _freeze_json(value: Any, *, path: str = "$", depth: int = 0) -> Any:
    if depth > 100:
        raise ArtifactGeometryMetricsError(f"JSON nesting is too deep at {path}")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        number = int(value)
        if abs(number) > MAX_SAFE_JSON_INTEGER:
            raise ArtifactGeometryMetricsError(
                f"integer at {path} exceeds the I-JSON safe range"
            )
        return number
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ArtifactGeometryMetricsError(
                f"number at {path} must be finite"
            )
        return 0.0 if number == 0.0 else number
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise ArtifactGeometryMetricsError(
                    f"object key at {path} must be a string"
                )
            output[key] = _freeze_json(
                value[key], path=f"{path}.{key}", depth=depth + 1
            )
        return MappingProxyType(output)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )
    raise ArtifactGeometryMetricsError(
        f"unsupported JSON value at {path}: {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _frozen_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    try:
        encoded = canonical_json_bytes(value)
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactGeometryMetricsError(f"{name} is not strict JSON") from exc
    frozen = _freeze_json(decoded, path=name)
    if not isinstance(frozen, Mapping):
        raise ArtifactGeometryMetricsError(f"{name} must be an object")
    return frozen


def _fixed_decimal(value: Decimal | float, quantum: Decimal) -> str:
    number = value if isinstance(value, Decimal) else Decimal(str(value))
    return format(number.quantize(quantum, rounding=ROUND_HALF_EVEN), "f")


def geometry_metrics_recipe(
    *,
    coordinate_grid_um: int = DEFAULT_GEOMETRY_METRICS_GRID_UM,
) -> dict[str, Any]:
    grid = _strict_int(
        coordinate_grid_um,
        name="coordinate_grid_um",
        minimum=1,
        maximum=MAX_GEOMETRY_METRICS_GRID_UM,
    )
    return {
        "algorithm": GEOMETRY_METRICS_ALGORITHM,
        "algorithm_version": GEOMETRY_METRICS_ALGORITHM_VERSION,
        "coordinate_grid_um": grid,
        "coordinate_space": GEOMETRY_METRICS_COORDINATE_SPACE,
        "kind": "geometry_metrics",
        "rounding_mode": GEOMETRY_METRICS_ROUNDING,
        "scope": GEOMETRY_METRICS_SCOPE,
        "surface_area_decimal_places": SURFACE_AREA_DECIMAL_PLACES,
        "volume_decimal_places": VOLUME_DECIMAL_PLACES,
        "volume_policy": GEOMETRY_METRICS_VOLUME_POLICY,
    }


def validate_geometry_metrics_recipe(value: object) -> dict[str, Any]:
    recipe = _exact_mapping(
        value,
        {
            "algorithm",
            "algorithm_version",
            "coordinate_grid_um",
            "coordinate_space",
            "kind",
            "rounding_mode",
            "scope",
            "surface_area_decimal_places",
            "volume_decimal_places",
            "volume_policy",
        },
        name="geometry metrics recipe",
    )
    expected = geometry_metrics_recipe(
        coordinate_grid_um=_strict_int(
            recipe["coordinate_grid_um"],
            name="recipe.coordinate_grid_um",
            minimum=1,
            maximum=MAX_GEOMETRY_METRICS_GRID_UM,
        )
    )
    if dict(recipe) != expected:
        raise ArtifactGeometryMetricsError(
            "geometry metrics recipe does not match the versioned contract"
        )
    return expected


def _vec3_int(value: object, *, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ArtifactGeometryMetricsError(f"{name} must contain three integers")
    return [
        _strict_int(
            item,
            name=f"{name}[{index}]",
            minimum=-MAX_SAFE_JSON_INTEGER,
            maximum=MAX_SAFE_JSON_INTEGER,
        )
        for index, item in enumerate(value)
    ]


def validate_geometry_metrics_receipt(value: object) -> dict[str, Any]:
    receipt = _exact_mapping(
        value,
        {
            "bounds_grid",
            "coordinate_grid_um",
            "coordinate_space",
            "input_face_count",
            "input_vertex_count",
            "kind",
            "measurement_basis",
            "quantization",
            "rounding_mode",
            "schema_version",
            "scope",
            "surface_area",
            "topology",
            "volume",
        },
        name="geometry metrics receipt",
    )
    if receipt["schema_version"] != GEOMETRY_METRICS_SCHEMA_VERSION:
        raise ArtifactGeometryMetricsError("geometry metrics schema is unsupported")
    if receipt["kind"] != "geometry_metrics":
        raise ArtifactGeometryMetricsError("geometry metrics kind is invalid")
    if receipt["coordinate_space"] != GEOMETRY_METRICS_COORDINATE_SPACE:
        raise ArtifactGeometryMetricsError("geometry metrics coordinate space is invalid")
    if receipt["measurement_basis"] != "quantized_triangle_mesh/v1":
        raise ArtifactGeometryMetricsError("geometry metrics basis is invalid")
    if receipt["rounding_mode"] != GEOMETRY_METRICS_ROUNDING:
        raise ArtifactGeometryMetricsError("geometry metrics rounding mode is invalid")
    if receipt["scope"] != GEOMETRY_METRICS_SCOPE:
        raise ArtifactGeometryMetricsError("geometry metrics scope is invalid")
    grid = _strict_int(
        receipt["coordinate_grid_um"],
        name="receipt.coordinate_grid_um",
        minimum=1,
        maximum=MAX_GEOMETRY_METRICS_GRID_UM,
    )
    vertex_count = _strict_int(
        receipt["input_vertex_count"],
        name="receipt.input_vertex_count",
        minimum=3,
        maximum=MAX_GEOMETRY_METRICS_VERTICES,
    )
    face_count = _strict_int(
        receipt["input_face_count"],
        name="receipt.input_face_count",
        minimum=1,
        maximum=MAX_GEOMETRY_METRICS_FACES,
    )

    bounds = _exact_mapping(
        receipt["bounds_grid"], {"maximum", "minimum"}, name="bounds_grid"
    )
    minimum = _vec3_int(bounds["minimum"], name="bounds_grid.minimum")
    maximum = _vec3_int(bounds["maximum"], name="bounds_grid.maximum")
    if any(low > high for low, high in zip(minimum, maximum)):
        raise ArtifactGeometryMetricsError("geometry metrics bounds are reversed")
    if any(high - low > MAX_QUANTIZED_AXIS_EXTENT for low, high in zip(minimum, maximum)):
        raise ArtifactGeometryMetricsError("geometry metrics bounds exceed the extent limit")

    quantization = _exact_mapping(
        receipt["quantization"],
        {
            "changed_vertex_count",
            "maximum_displacement_um",
            "maximum_per_axis_error_um_exact",
        },
        name="quantization",
    )
    changed = _strict_int(
        quantization["changed_vertex_count"],
        name="quantization.changed_vertex_count",
        minimum=0,
        maximum=vertex_count,
    )
    displacement = _decimal_string(
        quantization["maximum_displacement_um"],
        name="quantization.maximum_displacement_um",
        pattern=_DISPLACEMENT_DECIMAL_RE,
    )
    error_exact = _exact_mapping(
        quantization["maximum_per_axis_error_um_exact"],
        {"denominator", "numerator"},
        name="quantization.maximum_per_axis_error_um_exact",
    )
    if error_exact != {"denominator": 2, "numerator": grid}:
        raise ArtifactGeometryMetricsError(
            "quantization per-axis error does not match the grid"
        )
    max_displacement = Decimal(displacement)
    theoretical = Decimal(grid) * Decimal(3).sqrt() / Decimal(2)
    if max_displacement > theoretical + Decimal("0.000001"):
        raise ArtifactGeometryMetricsError(
            "quantization displacement exceeds the declared grid bound"
        )

    surface = _exact_mapping(
        receipt["surface_area"],
        {"decimal_mm2", "decimal_places", "status"},
        name="surface_area",
    )
    if surface["status"] != "available":
        raise ArtifactGeometryMetricsError("surface area status must be available")
    if surface["decimal_places"] != SURFACE_AREA_DECIMAL_PLACES:
        raise ArtifactGeometryMetricsError("surface area precision is invalid")
    surface_decimal = _decimal_string(
        surface["decimal_mm2"],
        name="surface_area.decimal_mm2",
        pattern=_AREA_DECIMAL_RE,
    )
    if Decimal(surface_decimal) <= 0:
        raise ArtifactGeometryMetricsError("surface area must be greater than zero")

    topology = _exact_mapping(
        receipt["topology"],
        {
            "boundary_edge_count",
            "closed_edge_manifold",
            "connected_component_count",
            "consistently_oriented",
            "degenerate_face_count",
            "duplicate_face_count",
            "edge_count",
            "non_manifold_edge_count",
            "orientation_mismatch_edge_count",
            "referenced_vertex_count",
            "unreferenced_vertex_count",
            "zero_length_edge_count",
        },
        name="topology",
    )
    referenced = _strict_int(
        topology["referenced_vertex_count"],
        name="topology.referenced_vertex_count",
        minimum=3,
        maximum=vertex_count,
    )
    unreferenced = _strict_int(
        topology["unreferenced_vertex_count"],
        name="topology.unreferenced_vertex_count",
        minimum=0,
        maximum=vertex_count,
    )
    if referenced + unreferenced != vertex_count:
        raise ArtifactGeometryMetricsError("topology vertex counts are inconsistent")
    edge_count = _strict_int(
        topology["edge_count"],
        name="topology.edge_count",
        minimum=1,
        maximum=face_count * 3,
    )
    boundary = _strict_int(
        topology["boundary_edge_count"],
        name="topology.boundary_edge_count",
        minimum=0,
        maximum=edge_count,
    )
    non_manifold = _strict_int(
        topology["non_manifold_edge_count"],
        name="topology.non_manifold_edge_count",
        minimum=0,
        maximum=edge_count,
    )
    mismatch = _strict_int(
        topology["orientation_mismatch_edge_count"],
        name="topology.orientation_mismatch_edge_count",
        minimum=0,
        maximum=edge_count,
    )
    zero_edges = _strict_int(
        topology["zero_length_edge_count"],
        name="topology.zero_length_edge_count",
        minimum=0,
        maximum=edge_count,
    )
    degenerate = _strict_int(
        topology["degenerate_face_count"],
        name="topology.degenerate_face_count",
        minimum=0,
        maximum=face_count,
    )
    duplicate = _strict_int(
        topology["duplicate_face_count"],
        name="topology.duplicate_face_count",
        minimum=0,
        maximum=face_count - 1,
    )
    components = _strict_int(
        topology["connected_component_count"],
        name="topology.connected_component_count",
        minimum=1,
        maximum=face_count,
    )
    closed_expected = (
        boundary == 0
        and non_manifold == 0
        and zero_edges == 0
        and degenerate == 0
        and duplicate == 0
    )
    if topology["closed_edge_manifold"] is not closed_expected:
        raise ArtifactGeometryMetricsError("closed-edge-manifold QC is inconsistent")
    oriented_expected = closed_expected and mismatch == 0
    if topology["consistently_oriented"] is not oriented_expected:
        raise ArtifactGeometryMetricsError("orientation QC is inconsistent")

    volume = _exact_mapping(
        receipt["volume"],
        {
            "decimal_mm3",
            "decimal_places",
            "exact_rational_mm3",
            "policy",
            "signed_six_grid_units3",
            "status",
            "winding",
        },
        name="volume",
    )
    if volume["decimal_places"] != VOLUME_DECIMAL_PLACES:
        raise ArtifactGeometryMetricsError("volume precision is invalid")
    if volume["policy"] != GEOMETRY_METRICS_VOLUME_POLICY:
        raise ArtifactGeometryMetricsError("volume policy is invalid")
    expected_topology_available = oriented_expected and components == 1
    status = volume["status"]
    if status == "available":
        if not expected_topology_available:
            raise ArtifactGeometryMetricsError(
                "volume cannot be available for the recorded topology"
            )
        decimal_mm3 = _decimal_string(
            volume["decimal_mm3"],
            name="volume.decimal_mm3",
            pattern=_VOLUME_DECIMAL_RE,
        )
        signed_text = _integer_string(
            volume["signed_six_grid_units3"],
            name="volume.signed_six_grid_units3",
            signed=True,
        )
        signed_six = int(signed_text)
        if signed_six == 0:
            raise ArtifactGeometryMetricsError("available volume cannot be zero")
        winding = volume["winding"]
        expected_winding = "positive" if signed_six > 0 else "negative"
        if winding != expected_winding:
            raise ArtifactGeometryMetricsError("volume winding is inconsistent")
        rational = _exact_mapping(
            volume["exact_rational_mm3"],
            {"denominator", "numerator"},
            name="volume.exact_rational_mm3",
        )
        numerator = int(
            _integer_string(
                rational["numerator"],
                name="volume.exact_rational_mm3.numerator",
                signed=False,
            )
        )
        denominator = int(
            _integer_string(
                rational["denominator"],
                name="volume.exact_rational_mm3.denominator",
                signed=False,
            )
        )
        if numerator <= 0 or denominator <= 0 or math.gcd(numerator, denominator) != 1:
            raise ArtifactGeometryMetricsError("volume rational must be positive and reduced")
        raw_numerator = abs(signed_six) * grid**3
        raw_denominator = 6 * 1000**3
        divisor = math.gcd(raw_numerator, raw_denominator)
        if (numerator, denominator) != (
            raw_numerator // divisor,
            raw_denominator // divisor,
        ):
            raise ArtifactGeometryMetricsError(
                "volume rational does not match the quantized signed sum"
            )
        expected_decimal = _fixed_decimal(
            Decimal(numerator) / Decimal(denominator), _VOLUME_QUANTUM
        )
        if decimal_mm3 != expected_decimal:
            raise ArtifactGeometryMetricsError("volume decimal does not match its rational")
    elif status in {"unavailable_topology", "unavailable_zero"}:
        if any(
            value is not None
            for value in (
                volume["decimal_mm3"],
                volume["exact_rational_mm3"],
                volume["signed_six_grid_units3"],
            )
        ) or volume["winding"] != "not_evaluated":
            raise ArtifactGeometryMetricsError(
                "unavailable volume must not carry a numeric result"
            )
        if status == "unavailable_topology" and expected_topology_available:
            raise ArtifactGeometryMetricsError(
                "topology-valid volume cannot use unavailable_topology"
            )
        if status == "unavailable_zero" and not expected_topology_available:
            raise ArtifactGeometryMetricsError(
                "unavailable_zero requires topology-valid input"
            )
    else:
        raise ArtifactGeometryMetricsError("volume status is invalid")

    return {
        "bounds_grid": {"maximum": maximum, "minimum": minimum},
        "coordinate_grid_um": grid,
        "coordinate_space": GEOMETRY_METRICS_COORDINATE_SPACE,
        "input_face_count": face_count,
        "input_vertex_count": vertex_count,
        "kind": "geometry_metrics",
        "measurement_basis": "quantized_triangle_mesh/v1",
        "quantization": {
            "changed_vertex_count": changed,
            "maximum_displacement_um": displacement,
            "maximum_per_axis_error_um_exact": {"denominator": 2, "numerator": grid},
        },
        "rounding_mode": GEOMETRY_METRICS_ROUNDING,
        "schema_version": GEOMETRY_METRICS_SCHEMA_VERSION,
        "scope": GEOMETRY_METRICS_SCOPE,
        "surface_area": {
            "decimal_mm2": surface_decimal,
            "decimal_places": SURFACE_AREA_DECIMAL_PLACES,
            "status": "available",
        },
        "topology": {
            "boundary_edge_count": boundary,
            "closed_edge_manifold": closed_expected,
            "connected_component_count": components,
            "consistently_oriented": oriented_expected,
            "degenerate_face_count": degenerate,
            "duplicate_face_count": duplicate,
            "edge_count": edge_count,
            "non_manifold_edge_count": non_manifold,
            "orientation_mismatch_edge_count": mismatch,
            "referenced_vertex_count": referenced,
            "unreferenced_vertex_count": unreferenced,
            "zero_length_edge_count": zero_edges,
        },
        "volume": {
            "decimal_mm3": volume["decimal_mm3"],
            "decimal_places": VOLUME_DECIMAL_PLACES,
            "exact_rational_mm3": (
                None
                if volume["exact_rational_mm3"] is None
                else {
                    "denominator": str(volume["exact_rational_mm3"]["denominator"]),
                    "numerator": str(volume["exact_rational_mm3"]["numerator"]),
                }
            ),
            "policy": GEOMETRY_METRICS_VOLUME_POLICY,
            "signed_six_grid_units3": volume["signed_six_grid_units3"],
            "status": status,
            "winding": volume["winding"],
        },
    }


def _validated_arrays(
    vertices_world_mm: object,
    faces: object,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices_world_mm, dtype=np.float64)
    face_array = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ArtifactGeometryMetricsError("vertices must have shape (N, 3)")
    if vertices.shape[0] < 3 or vertices.shape[0] > MAX_GEOMETRY_METRICS_VERTICES:
        raise ArtifactGeometryMetricsError("vertex count is outside safety limits")
    if not np.isfinite(vertices).all():
        raise ArtifactGeometryMetricsError("vertices must be finite")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ArtifactGeometryMetricsError("faces must have shape (M, 3)")
    if face_array.shape[0] < 1 or face_array.shape[0] > MAX_GEOMETRY_METRICS_FACES:
        raise ArtifactGeometryMetricsError("face count is outside safety limits")
    if not np.issubdtype(face_array.dtype, np.integer):
        raise ArtifactGeometryMetricsError("faces must contain integer indices")
    face_array = np.asarray(face_array, dtype=np.int64)
    if np.any(face_array < 0) or np.any(face_array >= vertices.shape[0]):
        raise ArtifactGeometryMetricsError("faces contain out-of-range indices")
    return vertices, face_array


def _quantized_vertices(
    vertices: np.ndarray,
    *,
    grid_um: int,
) -> tuple[np.ndarray, int, str]:
    scaled = vertices * (1000.0 / float(grid_um))
    if not np.isfinite(scaled).all() or np.max(np.abs(scaled)) > MAX_SAFE_JSON_INTEGER:
        raise ArtifactGeometryMetricsError(
            "vertices exceed the safe quantized coordinate range"
        )
    quantized = np.rint(scaled).astype(np.int64)
    extents = np.max(quantized, axis=0) - np.min(quantized, axis=0)
    if np.any(extents > MAX_QUANTIZED_AXIS_EXTENT):
        raise ArtifactGeometryMetricsError(
            "quantized geometry exceeds the supported artifact-scale extent"
        )
    reconstructed = quantized.astype(np.float64) * (float(grid_um) / 1000.0)
    displacement_um = np.linalg.norm(reconstructed - vertices, axis=1) * 1000.0
    changed = int(np.count_nonzero(np.any(reconstructed != vertices, axis=1)))
    maximum_displacement = float(np.max(displacement_um))
    return (
        quantized,
        changed,
        _fixed_decimal(maximum_displacement, Decimal("0.000001")),
    )


class _FaceUnionFind:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int64)
        self.rank = np.zeros(count, dtype=np.uint8)

    def find(self, value: int) -> int:
        parent = self.parent
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    def union(self, first: int, second: int) -> None:
        left = self.find(first)
        right = self.find(second)
        if left == right:
            return
        rank = self.rank
        if int(rank[left]) < int(rank[right]):
            left, right = right, left
        self.parent[right] = left
        if rank[left] == rank[right]:
            rank[left] = np.uint8(int(rank[left]) + 1)


def _topology_audit(
    quantized: np.ndarray,
    faces: np.ndarray,
    *,
    cancellation_probe: CancellationProbe | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    raise_if_cancelled(cancellation_probe)
    face_count = int(faces.shape[0])
    referenced = int(np.unique(faces.reshape(-1)).size)
    canonical_faces = np.sort(faces, axis=1)
    _, duplicate_counts = np.unique(canonical_faces, axis=0, return_counts=True)
    duplicate_face_count = int(np.sum(np.maximum(duplicate_counts - 1, 0)))
    raise_if_cancelled(cancellation_probe)

    directed_edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    face_ids = np.concatenate(
        (
            np.arange(face_count, dtype=np.int64),
            np.arange(face_count, dtype=np.int64),
            np.arange(face_count, dtype=np.int64),
        )
    )
    directions = np.where(
        directed_edges[:, 0] < directed_edges[:, 1],
        1,
        np.where(directed_edges[:, 0] > directed_edges[:, 1], -1, 0),
    ).astype(np.int8)
    undirected = np.sort(directed_edges, axis=1)
    order = np.lexsort((undirected[:, 1], undirected[:, 0]))
    sorted_edges = undirected[order]
    sorted_directions = directions[order]
    sorted_face_ids = face_ids[order]
    del directed_edges, undirected, face_ids, directions, order
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1))
            + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.array([sorted_edges.shape[0]])))
    counts = ends - starts
    orientation_sums = np.add.reduceat(sorted_directions.astype(np.int64), starts)
    edge_count = int(starts.size)
    boundary_edge_count = int(np.count_nonzero(counts == 1))
    non_manifold_edge_count = int(np.count_nonzero(counts > 2))
    zero_length_edge_count = int(
        np.count_nonzero(sorted_edges[starts, 0] == sorted_edges[starts, 1])
    )
    orientation_mismatch_edge_count = int(
        np.count_nonzero((counts == 2) & (orientation_sums != 0))
    )
    raise_if_cancelled(cancellation_probe)

    first = quantized[faces[:, 0]]
    second = quantized[faces[:, 1]]
    third = quantized[faces[:, 2]]
    cross = np.cross(second - first, third - first)
    degenerate_mask = np.all(cross == 0, axis=1)
    degenerate_face_count = int(np.count_nonzero(degenerate_mask))
    raise_if_cancelled(cancellation_probe)

    union_find = _FaceUnionFind(face_count)
    for group_index, (start, end) in enumerate(zip(starts, ends)):
        poll_cancellation(cancellation_probe, group_index, interval=4096)
        incident = sorted_face_ids[int(start) : int(end)]
        if incident.size < 2:
            continue
        anchor = int(incident[0])
        for other in incident[1:]:
            union_find.union(anchor, int(other))
    roots: set[int] = set()
    for face_index in range(face_count):
        poll_cancellation(cancellation_probe, face_index, interval=4096)
        roots.add(union_find.find(face_index))
    connected_component_count = len(roots)
    closed = (
        boundary_edge_count == 0
        and non_manifold_edge_count == 0
        and zero_length_edge_count == 0
        and degenerate_face_count == 0
        and duplicate_face_count == 0
    )
    oriented = closed and orientation_mismatch_edge_count == 0
    return (
        {
            "boundary_edge_count": boundary_edge_count,
            "closed_edge_manifold": closed,
            "connected_component_count": connected_component_count,
            "consistently_oriented": oriented,
            "degenerate_face_count": degenerate_face_count,
            "duplicate_face_count": duplicate_face_count,
            "edge_count": edge_count,
            "non_manifold_edge_count": non_manifold_edge_count,
            "orientation_mismatch_edge_count": orientation_mismatch_edge_count,
            "referenced_vertex_count": referenced,
            "unreferenced_vertex_count": int(quantized.shape[0]) - referenced,
            "zero_length_edge_count": zero_length_edge_count,
        },
        cross,
        degenerate_mask,
    )


def _surface_area_decimal(
    cross_grid2: np.ndarray,
    *,
    grid_um: int,
    cancellation_probe: CancellationProbe | None,
) -> str:
    partials: list[float] = []
    chunk_size = 65_536
    for start in range(0, cross_grid2.shape[0], chunk_size):
        poll_cancellation(
            cancellation_probe, start // chunk_size, interval=1
        )
        chunk = cross_grid2[start : start + chunk_size].astype(np.float64)
        lengths = np.sqrt(np.einsum("ij,ij->i", chunk, chunk))
        partials.append(math.fsum(float(value) for value in lengths))
    twice_area_grid2 = math.fsum(partials)
    area_mm2 = twice_area_grid2 * (float(grid_um) / 1000.0) ** 2 / 2.0
    if not math.isfinite(area_mm2) or area_mm2 <= 0.0:
        raise ArtifactGeometryMetricsError(
            "quantized geometry has no positive surface area"
        )
    return _fixed_decimal(area_mm2, _AREA_QUANTUM)


def _signed_six_volume_grid3(
    quantized: np.ndarray,
    faces: np.ndarray,
    *,
    cancellation_probe: CancellationProbe | None,
) -> int:
    origin = quantized[int(faces[0, 0])]
    total = 0
    for face_index, face in enumerate(faces):
        poll_cancellation(cancellation_probe, face_index, interval=4096)
        a = quantized[int(face[0])] - origin
        b = quantized[int(face[1])] - origin
        c = quantized[int(face[2])] - origin
        ax, ay, az = (int(value) for value in a)
        bx, by, bz = (int(value) for value in b)
        cx, cy, cz = (int(value) for value in c)
        total += ax * (by * cz - bz * cy)
        total += ay * (bz * cx - bx * cz)
        total += az * (bx * cy - by * cx)
    return total


def extract_geometry_metrics(
    vertices_world_mm: object,
    faces: object,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure one canonical-mm triangle mesh without mutating it."""

    raise_if_cancelled(cancellation_probe)
    validated_recipe = validate_geometry_metrics_recipe(recipe)
    vertices, face_array = _validated_arrays(vertices_world_mm, faces)
    grid_um = int(validated_recipe["coordinate_grid_um"])
    quantized, changed, maximum_displacement = _quantized_vertices(
        vertices, grid_um=grid_um
    )
    raise_if_cancelled(cancellation_probe)
    topology, cross, _degenerate_mask = _topology_audit(
        quantized,
        face_array,
        cancellation_probe=cancellation_probe,
    )
    surface_decimal = _surface_area_decimal(
        cross,
        grid_um=grid_um,
        cancellation_probe=cancellation_probe,
    )
    del cross, _degenerate_mask
    raise_if_cancelled(cancellation_probe)

    volume: dict[str, Any]
    if bool(topology["consistently_oriented"]) and int(
        topology["connected_component_count"]
    ) == 1:
        signed_six = _signed_six_volume_grid3(
            quantized,
            face_array,
            cancellation_probe=cancellation_probe,
        )
        if signed_six == 0:
            volume = {
                "decimal_mm3": None,
                "decimal_places": VOLUME_DECIMAL_PLACES,
                "exact_rational_mm3": None,
                "policy": GEOMETRY_METRICS_VOLUME_POLICY,
                "signed_six_grid_units3": None,
                "status": "unavailable_zero",
                "winding": "not_evaluated",
            }
        else:
            raw_numerator = abs(signed_six) * grid_um**3
            raw_denominator = 6 * 1000**3
            divisor = math.gcd(raw_numerator, raw_denominator)
            numerator = raw_numerator // divisor
            denominator = raw_denominator // divisor
            volume = {
                "decimal_mm3": _fixed_decimal(
                    Decimal(numerator) / Decimal(denominator), _VOLUME_QUANTUM
                ),
                "decimal_places": VOLUME_DECIMAL_PLACES,
                "exact_rational_mm3": {
                    "denominator": str(denominator),
                    "numerator": str(numerator),
                },
                "policy": GEOMETRY_METRICS_VOLUME_POLICY,
                "signed_six_grid_units3": str(signed_six),
                "status": "available",
                "winding": "positive" if signed_six > 0 else "negative",
            }
    else:
        volume = {
            "decimal_mm3": None,
            "decimal_places": VOLUME_DECIMAL_PLACES,
            "exact_rational_mm3": None,
            "policy": GEOMETRY_METRICS_VOLUME_POLICY,
            "signed_six_grid_units3": None,
            "status": "unavailable_topology",
            "winding": "not_evaluated",
        }

    receipt = validate_geometry_metrics_receipt(
        {
            "bounds_grid": {
                "maximum": [int(value) for value in np.max(quantized, axis=0)],
                "minimum": [int(value) for value in np.min(quantized, axis=0)],
            },
            "coordinate_grid_um": grid_um,
            "coordinate_space": GEOMETRY_METRICS_COORDINATE_SPACE,
            "input_face_count": int(face_array.shape[0]),
            "input_vertex_count": int(vertices.shape[0]),
            "kind": "geometry_metrics",
            "measurement_basis": "quantized_triangle_mesh/v1",
            "quantization": {
                "changed_vertex_count": changed,
                "maximum_displacement_um": maximum_displacement,
                "maximum_per_axis_error_um_exact": {
                    "denominator": 2,
                    "numerator": grid_um,
                },
            },
            "rounding_mode": GEOMETRY_METRICS_ROUNDING,
            "schema_version": GEOMETRY_METRICS_SCHEMA_VERSION,
            "scope": GEOMETRY_METRICS_SCOPE,
            "surface_area": {
                "decimal_mm2": surface_decimal,
                "decimal_places": SURFACE_AREA_DECIMAL_PLACES,
                "status": "available",
            },
            "topology": topology,
            "volume": volume,
        }
    )
    qc = {
        "boundary_edge_count": int(topology["boundary_edge_count"]),
        "closed_edge_manifold": bool(topology["closed_edge_manifold"]),
        "connected_component_count": int(topology["connected_component_count"]),
        "consistently_oriented": bool(topology["consistently_oriented"]),
        "coordinate_grid_um": grid_um,
        "degenerate_face_count": int(topology["degenerate_face_count"]),
        "duplicate_face_count": int(topology["duplicate_face_count"]),
        "input_face_count": int(face_array.shape[0]),
        "input_vertex_count": int(vertices.shape[0]),
        "non_manifold_edge_count": int(topology["non_manifold_edge_count"]),
        "orientation_mismatch_edge_count": int(
            topology["orientation_mismatch_edge_count"]
        ),
        "quantized_changed_vertex_count": changed,
        "surface_area_mm2_decimal": surface_decimal,
        "volume_mm3_decimal": volume["decimal_mm3"],
        "volume_status": volume["status"],
    }
    raise_if_cancelled(cancellation_probe)
    return receipt, qc


def _validate_qc_against_receipt(
    qc: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    topology = receipt["topology"]
    quantization = receipt["quantization"]
    surface = receipt["surface_area"]
    volume = receipt["volume"]
    assert isinstance(topology, Mapping)
    assert isinstance(quantization, Mapping)
    assert isinstance(surface, Mapping)
    assert isinstance(volume, Mapping)
    expected = {
        "boundary_edge_count": topology["boundary_edge_count"],
        "closed_edge_manifold": topology["closed_edge_manifold"],
        "connected_component_count": topology["connected_component_count"],
        "consistently_oriented": topology["consistently_oriented"],
        "coordinate_grid_um": receipt["coordinate_grid_um"],
        "degenerate_face_count": topology["degenerate_face_count"],
        "duplicate_face_count": topology["duplicate_face_count"],
        "input_face_count": receipt["input_face_count"],
        "input_vertex_count": receipt["input_vertex_count"],
        "non_manifold_edge_count": topology["non_manifold_edge_count"],
        "orientation_mismatch_edge_count": topology[
            "orientation_mismatch_edge_count"
        ],
        "quantized_changed_vertex_count": quantization["changed_vertex_count"],
        "surface_area_mm2_decimal": surface["decimal_mm2"],
        "volume_mm3_decimal": volume["decimal_mm3"],
        "volume_status": volume["status"],
    }
    if set(qc) != set(expected):
        raise ArtifactGeometryMetricsError(
            "geometry metrics QC fields do not match the closed contract"
        )
    for key, value in expected.items():
        if qc.get(key) != value:
            raise ArtifactGeometryMetricsError(
                f"geometry metrics QC field {key!r} does not match its receipt"
            )


@dataclass(frozen=True, slots=True)
class ArtifactGeometryMetricsComputation:
    context: OperationContext
    projection_snapshot: ArtifactProjectionSnapshot
    receipt: Mapping[str, Any]
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.context, OperationContext):
            raise ArtifactGeometryMetricsError("context must be OperationContext")
        if not isinstance(self.projection_snapshot, ArtifactProjectionSnapshot):
            raise ArtifactGeometryMetricsError(
                "projection_snapshot must be ArtifactProjectionSnapshot"
            )
        recipe = validate_geometry_metrics_recipe(self.recipe)
        receipt = validate_geometry_metrics_receipt(self.receipt)
        if int(recipe["coordinate_grid_um"]) != int(receipt["coordinate_grid_um"]):
            raise ArtifactGeometryMetricsError(
                "geometry metrics recipe and receipt grids differ"
            )
        if canonical_recipe_hash(recipe) != self.context.recipe_hash:
            raise ArtifactGeometryMetricsError(
                "geometry metrics recipe does not match its OperationContext"
            )
        snapshot = self.projection_snapshot
        if (
            tuple(self.context.source_asset_ids) != (snapshot.source_asset_id,)
            or self.context.geometry_revision_id != snapshot.geometry_revision_id
            or self.context.source_metadata_revision_id
            != snapshot.source_metadata_revision_id
            or self.context.align_revision_id != snapshot.align_revision_id
        ):
            raise ArtifactGeometryMetricsError(
                "projection snapshot does not match the metrics context"
            )
        if not isinstance(self.qc, Mapping):
            raise ArtifactGeometryMetricsError("geometry metrics QC must be an object")
        qc = _frozen_mapping(self.qc, name="geometry_metrics.qc")
        _validate_qc_against_receipt(qc, receipt)
        object.__setattr__(self, "recipe", _frozen_mapping(recipe, name="recipe"))
        object.__setattr__(self, "receipt", _frozen_mapping(receipt, name="receipt"))
        object.__setattr__(self, "qc", qc)

    @property
    def geometry_ref(self) -> str:
        return GEOMETRY_METRICS_REF_PREFIX + canonical_json_sha256(self.receipt)

    def recipe_dict(self) -> dict[str, Any]:
        value = _thaw_json(self.recipe)
        assert isinstance(value, dict)
        return value

    def receipt_dict(self) -> dict[str, Any]:
        value = _thaw_json(self.receipt)
        assert isinstance(value, dict)
        return value

    def qc_dict(self) -> dict[str, Any]:
        value = _thaw_json(self.qc)
        assert isinstance(value, dict)
        return value


def compute_artifact_geometry_metrics(
    session: ArtifactSession,
    *,
    coordinate_grid_um: int = DEFAULT_GEOMETRY_METRICS_GRID_UM,
    cancellation_probe: CancellationProbe | None = None,
) -> ArtifactGeometryMetricsComputation:
    raise_if_cancelled(cancellation_probe)
    if not isinstance(session, ArtifactSession):
        raise ArtifactGeometryMetricsError("session must be an ArtifactSession")
    recipe = geometry_metrics_recipe(coordinate_grid_um=coordinate_grid_um)
    try:
        context = session.capture_operation(recipe=recipe)
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactGeometryMetricsError(str(exc)) from exc
    receipt, qc = extract_geometry_metrics(
        projection.mesh.vertices,
        projection.mesh.faces,
        recipe,
        cancellation_probe=cancellation_probe,
    )
    return ArtifactGeometryMetricsComputation(
        context=context,
        projection_snapshot=projection.snapshot,
        receipt=receipt,
        recipe=recipe,
        qc=qc,
    )


def geometry_metrics_computation_matches_active_projection(
    session: ArtifactSession,
    computation: ArtifactGeometryMetricsComputation,
) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(
        computation, ArtifactGeometryMetricsComputation
    ):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    return current.render_key == computation.projection_snapshot.render_key


def append_geometry_metrics_record_from_context(
    document: ArtifactDocument,
    *,
    context: OperationContext,
    computation: ArtifactGeometryMetricsComputation,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactDocument:
    if not isinstance(document, ArtifactDocument):
        raise ArtifactGeometryMetricsError("document must be an ArtifactDocument")
    if not isinstance(context, OperationContext) or context != computation.context:
        raise ArtifactGeometryMetricsError(
            "geometry metrics context does not match its computation"
        )
    receipt = computation.receipt_dict()
    qc = computation.qc_dict()
    _validate_qc_against_receipt(qc, receipt)
    receipt_bytes = canonical_json_bytes(receipt)
    if len(receipt_bytes) > MAX_GEOMETRY_METRICS_RECEIPT_BYTES:
        raise ArtifactGeometryMetricsError("geometry metrics receipt exceeds its limit")
    receipt_sha256 = canonical_json_sha256(receipt)
    extensions = {
        GEOMETRY_METRICS_EXTENSION_KEY: {
            "media_type": GEOMETRY_METRICS_MEDIA_TYPE,
            "receipt": receipt,
            "receipt_byte_length": len(receipt_bytes),
            "receipt_sha256": receipt_sha256,
            "schema_version": GEOMETRY_METRICS_SCHEMA_VERSION,
        }
    }
    try:
        return document.append_record_from_context(
            context=context,
            id=record_id,
            type=GEOMETRY_METRICS_RECORD_TYPE,
            geometry_ref=GEOMETRY_METRICS_REF_PREFIX + receipt_sha256,
            recipe=computation.recipe_dict(),
            qc=qc,
            lifecycle_status=RecordLifecycleStatus.READY,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=depends_on_record_ids,
            extensions=extensions,
        )
    except ArtifactDocumentError as exc:
        raise ArtifactGeometryMetricsError(str(exc)) from exc


def commit_artifact_geometry_metrics(
    session: ArtifactSession,
    computation: ArtifactGeometryMetricsComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    if not geometry_metrics_computation_matches_active_projection(session, computation):
        raise ArtifactGeometryMetricsError(
            "geometry metrics computation is stale for the active projection"
        )
    document = append_geometry_metrics_record_from_context(
        session.document,
        context=computation.context,
        computation=computation,
        record_id=record_id,
        created_at=created_at,
        operator=operator,
        depends_on_record_ids=depends_on_record_ids,
    )
    return session.with_document(document)


def geometry_metrics_receipt_from_record(record: DerivedRecord) -> dict[str, Any]:
    if not isinstance(record, DerivedRecord):
        raise ArtifactGeometryMetricsError("record must be a DerivedRecord")
    if record.type != GEOMETRY_METRICS_RECORD_TYPE:
        raise ArtifactGeometryMetricsError("record is not a geometry metrics record")
    descriptor = _exact_mapping(
        record.extensions.get(GEOMETRY_METRICS_EXTENSION_KEY),
        {
            "media_type",
            "receipt",
            "receipt_byte_length",
            "receipt_sha256",
            "schema_version",
        },
        name="geometry metrics descriptor",
    )
    if descriptor["media_type"] != GEOMETRY_METRICS_MEDIA_TYPE:
        raise ArtifactGeometryMetricsError("geometry metrics media type is invalid")
    if descriptor["schema_version"] != GEOMETRY_METRICS_SCHEMA_VERSION:
        raise ArtifactGeometryMetricsError("geometry metrics descriptor schema is invalid")
    receipt = validate_geometry_metrics_receipt(descriptor["receipt"])
    receipt_bytes = canonical_json_bytes(receipt)
    if descriptor["receipt_byte_length"] != len(receipt_bytes):
        raise ArtifactGeometryMetricsError("geometry metrics receipt length is invalid")
    receipt_sha256 = canonical_json_sha256(receipt)
    if descriptor["receipt_sha256"] != receipt_sha256:
        raise ArtifactGeometryMetricsError("geometry metrics receipt hash is invalid")
    if record.geometry_ref != GEOMETRY_METRICS_REF_PREFIX + receipt_sha256:
        raise ArtifactGeometryMetricsError("geometry metrics geometry_ref is invalid")
    recipe = validate_geometry_metrics_recipe(record.recipe)
    if int(recipe["coordinate_grid_um"]) != int(receipt["coordinate_grid_um"]):
        raise ArtifactGeometryMetricsError("record recipe and receipt grids differ")
    _validate_qc_against_receipt(record.qc, receipt)
    return receipt


def validate_geometry_metrics_records(document: ArtifactDocument) -> None:
    if not isinstance(document, ArtifactDocument):
        raise ArtifactGeometryMetricsError("document must be an ArtifactDocument")
    for record in document.records:
        if record.type == GEOMETRY_METRICS_RECORD_TYPE:
            geometry_metrics_receipt_from_record(record)


__all__ = [
    "ArtifactGeometryMetricsComputation",
    "ArtifactGeometryMetricsError",
    "DEFAULT_GEOMETRY_METRICS_GRID_UM",
    "GEOMETRY_METRICS_ALGORITHM",
    "GEOMETRY_METRICS_ALGORITHM_VERSION",
    "GEOMETRY_METRICS_COORDINATE_SPACE",
    "GEOMETRY_METRICS_EXTENSION_KEY",
    "GEOMETRY_METRICS_MEDIA_TYPE",
    "GEOMETRY_METRICS_RECORD_TYPE",
    "GEOMETRY_METRICS_REF_PREFIX",
    "GEOMETRY_METRICS_SCHEMA_VERSION",
    "GEOMETRY_METRICS_SCOPE",
    "GEOMETRY_METRICS_VOLUME_POLICY",
    "MAX_GEOMETRY_METRICS_FACES",
    "MAX_GEOMETRY_METRICS_VERTICES",
    "append_geometry_metrics_record_from_context",
    "commit_artifact_geometry_metrics",
    "compute_artifact_geometry_metrics",
    "extract_geometry_metrics",
    "geometry_metrics_computation_matches_active_projection",
    "geometry_metrics_receipt_from_record",
    "geometry_metrics_recipe",
    "validate_geometry_metrics_receipt",
    "validate_geometry_metrics_recipe",
    "validate_geometry_metrics_records",
]
