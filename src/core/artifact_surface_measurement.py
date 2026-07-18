"""Durable surface-anchored chord distance and fitted circle diameter records.

The legacy viewport keeps convenient floating world points, but those points
cannot be reattached to source geometry after a project is reopened.  This
module instead binds every pick to one triangle face and fixed-denominator
barycentric weights.  Measurements are recomputed from a fresh canonical-mm
projection and published as immutable :class:`DerivedRecord` receipts.

``surface_distance`` means the three-dimensional Euclidean chord between two
surface anchors.  It is deliberately not a geodesic distance.  A
``surface_diameter`` is the diameter of a normalized algebraic Kasa circle on
the PCA best-fit plane through three or more anchors; it is not a geometric
radial least-squares fit or an unconstrained maximum object diameter.  Both
meanings are closed, versioned recipe fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, DecimalException, ROUND_HALF_EVEN, localcontext
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_cancellation import CancellationProbe, poll_cancellation, raise_if_cancelled
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
from .alignment_utils import require_affine_matrix4x4


SURFACE_MEASUREMENT_ALGORITHM = "archmeshrubbing.surface_anchor_measurement"
SURFACE_MEASUREMENT_ALGORITHM_VERSION = "1.0.0"
SURFACE_MEASUREMENT_SCHEMA_VERSION = "1.0.0"
SURFACE_DISTANCE_RECORD_TYPE = "measurement.surface_distance.v1"
SURFACE_DIAMETER_RECORD_TYPE = "measurement.circle_diameter.v1"
SURFACE_MEASUREMENT_EXTENSION_KEY = "org.archmeshrubbing:surface-measurement-v1"
SURFACE_MEASUREMENT_MEDIA_TYPE = (
    "application/vnd.archmeshrubbing.surface-measurement-receipt+json"
)
SURFACE_MEASUREMENT_REF_PREFIX = (
    "urn:archmeshrubbing:surface-measurement-receipt:sha256:"
)
SURFACE_MEASUREMENT_COORDINATE_SPACE = "canonical_aligned_mm/v1"
SURFACE_ANCHOR_BASIS = "triangle_face_index+barycentric_ppb/v1"
SURFACE_PICK_METHOD = "frame_depth_unproject+cpu_ray_triangle/v1"
SURFACE_DISTANCE_MEANING = "euclidean_3d_chord_between_surface_anchors/v1"
SURFACE_DIAMETER_FIT_POLICY = (
    "normalized_algebraic_kasa_on_pca_best_fit_plane/v1"
)
SURFACE_MEASUREMENT_ROUNDING = "round_ties_to_even"

DEFAULT_COORDINATE_GRID_UM = 1
BARYCENTRIC_DENOMINATOR = 1_000_000_000
DEFAULT_EDGE_REVIEW_THRESHOLD_PPB = 1_000_000
DEFAULT_FIT_REVIEW_THRESHOLD_UM = 250
DEFAULT_MAXIMUM_FIT_CONDITION = 100_000_000
DEFAULT_MINIMUM_DEPTH_MATCH_TOLERANCE_UM = 50
MAXIMUM_DEPTH_MATCH_TOLERANCE_UM = 10_000
MAXIMUM_PIXEL_FOOTPRINT_UM = 10_000_000
MAXIMUM_SCREEN_SEARCH_OFFSET_PX = 64
MAXIMUM_DIAMETER_ANCHORS = 64
MAXIMUM_SURFACE_MEASUREMENT_VERTICES = 5_000_000
MAXIMUM_SURFACE_MEASUREMENT_FACES = 2_000_000
MAXIMUM_SAFE_JSON_INTEGER = 9_007_199_254_740_991
MAXIMUM_RECEIPT_BYTES = 128 * 1024
MAXIMUM_NUMERIC_TEXT_LENGTH = 128
RESULT_DECIMAL_PLACES = 6
NORMAL_DECIMAL_PLACES = 9

_RESULT_QUANTUM = Decimal("0.000001")
_NORMAL_QUANTUM = Decimal("0.000000001")
_UNSIGNED_INTEGER_RE = re.compile(r"^(0|[1-9][0-9]*)$")
_SIGNED_INTEGER_RE = re.compile(r"^(0|-?[1-9][0-9]*)$")
_UNSIGNED_DECIMAL_RE = re.compile(r"^(0|[1-9][0-9]*)\.[0-9]{6}$")
_SIGNED_DECIMAL_RE = re.compile(r"^(0|-?(?:[1-9][0-9]*|0))\.[0-9]{6}$")
_SIGNED_NORMAL_RE = re.compile(r"^(0|-?(?:[1-9][0-9]*|0))\.[0-9]{9}$")


class ArtifactSurfaceMeasurementError(ValueError):
    """A surface anchor, computation, or durable record is invalid."""


def _exact_mapping(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactSurfaceMeasurementError(f"{name} must be an object")
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise ArtifactSurfaceMeasurementError(
            f"{name} is missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise ArtifactSurfaceMeasurementError(
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
        raise ArtifactSurfaceMeasurementError(f"{name} must be an integer")
    number = int(value)
    if number < minimum or number > maximum:
        raise ArtifactSurfaceMeasurementError(
            f"{name} must be in the inclusive range {minimum}..{maximum}"
        )
    return number


def _fixed_decimal(value: Decimal | float, quantum: Decimal = _RESULT_QUANTUM) -> str:
    number = value if isinstance(value, Decimal) else Decimal(str(value))
    if not number.is_finite():
        raise ArtifactSurfaceMeasurementError("measurement result must be finite")
    try:
        with localcontext() as context:
            context.prec = max(80, len(number.as_tuple().digits) + 20)
            quantized = number.quantize(quantum, rounding=ROUND_HALF_EVEN)
    except DecimalException as exc:
        raise ArtifactSurfaceMeasurementError(
            "measurement result cannot be represented by the fixed decimal policy"
        ) from exc
    if quantized == 0:
        quantized = Decimal(0).quantize(quantum)
    return format(quantized, "f")


def _decimal_text(
    value: object,
    *,
    name: str,
    signed: bool = False,
    normal: bool = False,
) -> str:
    pattern = (
        _SIGNED_NORMAL_RE
        if normal
        else (_SIGNED_DECIMAL_RE if signed else _UNSIGNED_DECIMAL_RE)
    )
    if (
        not isinstance(value, str)
        or len(value) > MAXIMUM_NUMERIC_TEXT_LENGTH
        or pattern.fullmatch(value) is None
    ):
        raise ArtifactSurfaceMeasurementError(f"{name} has an invalid decimal form")
    number = Decimal(value)
    if not number.is_finite() or (not signed and number < 0):
        raise ArtifactSurfaceMeasurementError(f"{name} has an invalid decimal value")
    return value


def _integer_text(value: object, *, name: str, signed: bool = False) -> str:
    pattern = _SIGNED_INTEGER_RE if signed else _UNSIGNED_INTEGER_RE
    if (
        not isinstance(value, str)
        or len(value) > MAXIMUM_NUMERIC_TEXT_LENGTH
        or pattern.fullmatch(value) is None
    ):
        raise ArtifactSurfaceMeasurementError(f"{name} has an invalid integer form")
    return value


def _freeze_json(value: Any, *, path: str = "$", depth: int = 0) -> Any:
    if depth > 100:
        raise ArtifactSurfaceMeasurementError(f"JSON nesting is too deep at {path}")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        number = int(value)
        if abs(number) > MAXIMUM_SAFE_JSON_INTEGER:
            raise ArtifactSurfaceMeasurementError(
                f"integer at {path} exceeds the I-JSON safe range"
            )
        return number
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ArtifactSurfaceMeasurementError(f"number at {path} must be finite")
        return 0.0 if number == 0.0 else number
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise ArtifactSurfaceMeasurementError(
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
    raise ArtifactSurfaceMeasurementError(
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
        decoded = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactSurfaceMeasurementError(f"{name} is not strict JSON") from exc
    frozen = _freeze_json(decoded, path=name)
    if not isinstance(frozen, Mapping):
        raise ArtifactSurfaceMeasurementError(f"{name} must be an object")
    return frozen


def _vec3_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ArtifactSurfaceMeasurementError(f"{name} must contain three integers")
    return [
        _strict_int(item, name=f"{name}[{index}]", minimum=minimum, maximum=maximum)
        for index, item in enumerate(value)
    ]


def _vec2_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ArtifactSurfaceMeasurementError(f"{name} must contain two integers")
    return [
        _strict_int(item, name=f"{name}[{index}]", minimum=minimum, maximum=maximum)
        for index, item in enumerate(value)
    ]


def _vec3_text(value: object, *, name: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ArtifactSurfaceMeasurementError(f"{name} must contain three integers")
    return [
        _integer_text(item, name=f"{name}[{index}]", signed=True)
        for index, item in enumerate(value)
    ]


def _vec3_decimal(
    value: object,
    *,
    name: str,
    normal: bool = False,
) -> list[str]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ArtifactSurfaceMeasurementError(f"{name} must contain three decimals")
    return [
        _decimal_text(
            item,
            name=f"{name}[{index}]",
            signed=True,
            normal=normal,
        )
        for index, item in enumerate(value)
    ]


def _validated_anchor(
    value: object,
    *,
    index: int,
    source_vertex_count: int,
    source_face_count: int,
) -> dict[str, Any]:
    anchor = _exact_mapping(
        value,
        {
            "barycentric_numerators",
            "capture_point_grid",
            "depth_match_tolerance_um",
            "depth_search_offset_px",
            "face_index",
            "face_vertex_indices",
            "pixel_footprint_um",
        },
        name=f"surface anchor {index}",
    )
    face_index = _strict_int(
        anchor["face_index"],
        name=f"surface anchor {index}.face_index",
        minimum=0,
        maximum=source_face_count - 1,
    )
    face_vertices = _vec3_int(
        anchor["face_vertex_indices"],
        name=f"surface anchor {index}.face_vertex_indices",
        minimum=0,
        maximum=source_vertex_count - 1,
    )
    if len(set(face_vertices)) != 3:
        raise ArtifactSurfaceMeasurementError(
            f"surface anchor {index} face vertices must be distinct"
        )
    barycentric = _vec3_int(
        anchor["barycentric_numerators"],
        name=f"surface anchor {index}.barycentric_numerators",
        minimum=0,
        maximum=BARYCENTRIC_DENOMINATOR,
    )
    if sum(barycentric) != BARYCENTRIC_DENOMINATOR:
        raise ArtifactSurfaceMeasurementError(
            f"surface anchor {index} barycentric numerators must sum exactly to "
            f"{BARYCENTRIC_DENOMINATOR}"
        )
    capture = _vec3_int(
        anchor["capture_point_grid"],
        name=f"surface anchor {index}.capture_point_grid",
        minimum=-MAXIMUM_SAFE_JSON_INTEGER,
        maximum=MAXIMUM_SAFE_JSON_INTEGER,
    )
    search_offset = _vec2_int(
        anchor["depth_search_offset_px"],
        name=f"surface anchor {index}.depth_search_offset_px",
        minimum=-MAXIMUM_SCREEN_SEARCH_OFFSET_PX,
        maximum=MAXIMUM_SCREEN_SEARCH_OFFSET_PX,
    )
    pixel_footprint = _strict_int(
        anchor["pixel_footprint_um"],
        name=f"surface anchor {index}.pixel_footprint_um",
        minimum=1,
        maximum=MAXIMUM_PIXEL_FOOTPRINT_UM,
    )
    depth_tolerance = _strict_int(
        anchor["depth_match_tolerance_um"],
        name=f"surface anchor {index}.depth_match_tolerance_um",
        minimum=DEFAULT_MINIMUM_DEPTH_MATCH_TOLERANCE_UM,
        maximum=MAXIMUM_DEPTH_MATCH_TOLERANCE_UM,
    )
    return {
        "barycentric_numerators": barycentric,
        "capture_point_grid": capture,
        "depth_match_tolerance_um": depth_tolerance,
        "depth_search_offset_px": search_offset,
        "face_index": face_index,
        "face_vertex_indices": face_vertices,
        "pixel_footprint_um": pixel_footprint,
    }


def _surface_measurement_recipe(
    kind: str,
    anchors: Sequence[Mapping[str, Any]],
    *,
    source_vertex_count: int,
    source_face_count: int,
    coordinate_grid_um: int = DEFAULT_COORDINATE_GRID_UM,
    edge_review_threshold_ppb: int = DEFAULT_EDGE_REVIEW_THRESHOLD_PPB,
    fit_review_threshold_um: int = DEFAULT_FIT_REVIEW_THRESHOLD_UM,
    maximum_fit_condition: int = DEFAULT_MAXIMUM_FIT_CONDITION,
) -> dict[str, Any]:
    if kind not in {"surface_distance", "surface_diameter"}:
        raise ArtifactSurfaceMeasurementError("surface measurement kind is unsupported")
    vertex_count = _strict_int(
        source_vertex_count,
        name="source_vertex_count",
        minimum=3,
        maximum=MAXIMUM_SURFACE_MEASUREMENT_VERTICES,
    )
    face_count = _strict_int(
        source_face_count,
        name="source_face_count",
        minimum=1,
        maximum=MAXIMUM_SURFACE_MEASUREMENT_FACES,
    )
    minimum_anchors = 2 if kind == "surface_distance" else 3
    maximum_anchors = 2 if kind == "surface_distance" else MAXIMUM_DIAMETER_ANCHORS
    anchor_values = list(anchors)
    if len(anchor_values) < minimum_anchors or len(anchor_values) > maximum_anchors:
        raise ArtifactSurfaceMeasurementError(
            f"{kind} requires {minimum_anchors}..{maximum_anchors} anchors"
        )
    validated_anchors = [
        _validated_anchor(
            value,
            index=index,
            source_vertex_count=vertex_count,
            source_face_count=face_count,
        )
        for index, value in enumerate(anchor_values)
    ]
    grid_um = _strict_int(
        coordinate_grid_um,
        name="coordinate_grid_um",
        minimum=1,
        maximum=1000,
    )
    for index, anchor in enumerate(validated_anchors):
        expected_tolerance = _pick_depth_tolerance_um(
            pixel_footprint_um=int(anchor["pixel_footprint_um"]),
            grid_um=grid_um,
        )
        if int(anchor["depth_match_tolerance_um"]) != expected_tolerance:
            raise ArtifactSurfaceMeasurementError(
                f"surface anchor {index} depth tolerance does not match its "
                "pixel/grid policy"
            )
    edge_threshold = _strict_int(
        edge_review_threshold_ppb,
        name="edge_review_threshold_ppb",
        minimum=0,
        maximum=BARYCENTRIC_DENOMINATOR // 3,
    )
    fit_threshold = _strict_int(
        fit_review_threshold_um,
        name="fit_review_threshold_um",
        minimum=1,
        maximum=MAXIMUM_PIXEL_FOOTPRINT_UM,
    )
    condition = _strict_int(
        maximum_fit_condition,
        name="maximum_fit_condition",
        minimum=100,
        maximum=MAXIMUM_SAFE_JSON_INTEGER,
    )
    return {
        "algorithm": SURFACE_MEASUREMENT_ALGORITHM,
        "algorithm_version": SURFACE_MEASUREMENT_ALGORITHM_VERSION,
        "anchor_basis": SURFACE_ANCHOR_BASIS,
        "anchors": validated_anchors,
        "barycentric_denominator": BARYCENTRIC_DENOMINATOR,
        "coordinate_grid_um": grid_um,
        "coordinate_space": SURFACE_MEASUREMENT_COORDINATE_SPACE,
        "edge_review_threshold_ppb": edge_threshold,
        "fit_policy": (
            "none" if kind == "surface_distance" else SURFACE_DIAMETER_FIT_POLICY
        ),
        "fit_review_threshold_um": fit_threshold,
        "kind": kind,
        "maximum_fit_condition": condition,
        "measurement_meaning": (
            SURFACE_DISTANCE_MEANING
            if kind == "surface_distance"
            else "best_fit_planar_circle_diameter_from_surface_anchors/v1"
        ),
        "pick_method": SURFACE_PICK_METHOD,
        "result_decimal_places": RESULT_DECIMAL_PLACES,
        "rounding_mode": SURFACE_MEASUREMENT_ROUNDING,
        "source_face_count": face_count,
        "source_vertex_count": vertex_count,
    }


def surface_distance_recipe(
    anchors: Sequence[Mapping[str, Any]],
    *,
    source_vertex_count: int,
    source_face_count: int,
    coordinate_grid_um: int = DEFAULT_COORDINATE_GRID_UM,
    edge_review_threshold_ppb: int = DEFAULT_EDGE_REVIEW_THRESHOLD_PPB,
) -> dict[str, Any]:
    return _surface_measurement_recipe(
        "surface_distance",
        anchors,
        source_vertex_count=source_vertex_count,
        source_face_count=source_face_count,
        coordinate_grid_um=coordinate_grid_um,
        edge_review_threshold_ppb=edge_review_threshold_ppb,
    )


def surface_diameter_recipe(
    anchors: Sequence[Mapping[str, Any]],
    *,
    source_vertex_count: int,
    source_face_count: int,
    coordinate_grid_um: int = DEFAULT_COORDINATE_GRID_UM,
    edge_review_threshold_ppb: int = DEFAULT_EDGE_REVIEW_THRESHOLD_PPB,
    fit_review_threshold_um: int = DEFAULT_FIT_REVIEW_THRESHOLD_UM,
    maximum_fit_condition: int = DEFAULT_MAXIMUM_FIT_CONDITION,
) -> dict[str, Any]:
    return _surface_measurement_recipe(
        "surface_diameter",
        anchors,
        source_vertex_count=source_vertex_count,
        source_face_count=source_face_count,
        coordinate_grid_um=coordinate_grid_um,
        edge_review_threshold_ppb=edge_review_threshold_ppb,
        fit_review_threshold_um=fit_review_threshold_um,
        maximum_fit_condition=maximum_fit_condition,
    )


def validate_surface_measurement_recipe(value: object) -> dict[str, Any]:
    recipe = _exact_mapping(
        value,
        {
            "algorithm",
            "algorithm_version",
            "anchor_basis",
            "anchors",
            "barycentric_denominator",
            "coordinate_grid_um",
            "coordinate_space",
            "edge_review_threshold_ppb",
            "fit_policy",
            "fit_review_threshold_um",
            "kind",
            "maximum_fit_condition",
            "measurement_meaning",
            "pick_method",
            "result_decimal_places",
            "rounding_mode",
            "source_face_count",
            "source_vertex_count",
        },
        name="surface measurement recipe",
    )
    kind = str(recipe["kind"])
    anchors = recipe["anchors"]
    if not isinstance(anchors, (list, tuple)):
        raise ArtifactSurfaceMeasurementError("surface measurement anchors must be a list")
    expected = _surface_measurement_recipe(
        kind,
        anchors,
        source_vertex_count=_strict_int(
            recipe["source_vertex_count"],
            name="recipe.source_vertex_count",
            minimum=3,
            maximum=MAXIMUM_SURFACE_MEASUREMENT_VERTICES,
        ),
        source_face_count=_strict_int(
            recipe["source_face_count"],
            name="recipe.source_face_count",
            minimum=1,
            maximum=MAXIMUM_SURFACE_MEASUREMENT_FACES,
        ),
        coordinate_grid_um=_strict_int(
            recipe["coordinate_grid_um"],
            name="recipe.coordinate_grid_um",
            minimum=1,
            maximum=1000,
        ),
        edge_review_threshold_ppb=_strict_int(
            recipe["edge_review_threshold_ppb"],
            name="recipe.edge_review_threshold_ppb",
            minimum=0,
            maximum=BARYCENTRIC_DENOMINATOR // 3,
        ),
        fit_review_threshold_um=_strict_int(
            recipe["fit_review_threshold_um"],
            name="recipe.fit_review_threshold_um",
            minimum=1,
            maximum=MAXIMUM_PIXEL_FOOTPRINT_UM,
        ),
        maximum_fit_condition=_strict_int(
            recipe["maximum_fit_condition"],
            name="recipe.maximum_fit_condition",
            minimum=100,
            maximum=MAXIMUM_SAFE_JSON_INTEGER,
        ),
    )
    if canonical_json_bytes(recipe) != canonical_json_bytes(expected):
        raise ArtifactSurfaceMeasurementError(
            "surface measurement recipe constants or semantics are invalid"
        )
    return expected


def surface_measurement_selection_hash(recipe: Mapping[str, Any]) -> str:
    validated = validate_surface_measurement_recipe(recipe)
    return canonical_json_sha256(
        {
            "anchor_basis": validated["anchor_basis"],
            "anchors": validated["anchors"],
            "barycentric_denominator": validated["barycentric_denominator"],
        }
    )


def _quantized_barycentric(weights: Sequence[float]) -> list[int]:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ArtifactSurfaceMeasurementError("barycentric weights must be finite vec3")
    if np.any(values < -1e-8) or np.any(values > 1.0 + 1e-8):
        raise ArtifactSurfaceMeasurementError("ray hit is outside its triangle")
    values = np.clip(values, 0.0, 1.0)
    total = float(math.fsum(float(item) for item in values))
    if not math.isfinite(total) or total <= 0.0:
        raise ArtifactSurfaceMeasurementError("barycentric weights have no mass")
    values = values / total
    scaled = values * float(BARYCENTRIC_DENOMINATOR)
    floors = np.floor(scaled).astype(np.int64)
    remainder = BARYCENTRIC_DENOMINATOR - int(np.sum(floors, dtype=np.int64))
    fractions = scaled - floors.astype(np.float64)
    order = sorted(range(3), key=lambda index: (-float(fractions[index]), index))
    for offset in range(remainder):
        floors[order[offset % 3]] += 1
    output = [int(item) for item in floors]
    if min(output) < 0 or sum(output) != BARYCENTRIC_DENOMINATOR:
        raise ArtifactSurfaceMeasurementError("barycentric quantization failed")
    return output


def _quantized_point(point_world_mm: object, *, grid_um: int) -> list[int]:
    point = np.asarray(point_world_mm, dtype=np.float64).reshape(-1)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ArtifactSurfaceMeasurementError("capture point must be a finite vec3")
    scaled = point * (1000.0 / float(grid_um))
    if np.max(np.abs(scaled)) > MAXIMUM_SAFE_JSON_INTEGER:
        raise ArtifactSurfaceMeasurementError("capture point exceeds the safe grid range")
    return [int(item) for item in np.rint(scaled).astype(np.int64)]


def _pick_depth_tolerance_um(*, pixel_footprint_um: int, grid_um: int) -> int:
    tolerance = max(
        DEFAULT_MINIMUM_DEPTH_MATCH_TOLERANCE_UM,
        int(math.ceil(float(pixel_footprint_um) * 2.0 + float(grid_um) * 4.0)),
    )
    if tolerance > MAXIMUM_DEPTH_MATCH_TOLERANCE_UM:
        raise ArtifactSurfaceMeasurementError(
            "screen pixel footprint is too coarse for an authoritative anchor"
        )
    return tolerance


def resolve_surface_anchor_from_ray(
    vertices_world_mm: object,
    faces: object,
    *,
    source_faces: object,
    ray_origin_world_mm: object,
    ray_direction_world: object,
    depth_point_world_mm: object,
    pixel_footprint_um: int,
    depth_search_offset_px: Sequence[int] = (0, 0),
    coordinate_grid_um: int = DEFAULT_COORDINATE_GRID_UM,
    cancellation_probe: CancellationProbe | None = None,
) -> dict[str, Any]:
    """Resolve a depth sample to the closest exact CPU ray/triangle hit.

    Every triangle is considered in bounded chunks.  A centroid-nearest legacy
    heuristic is intentionally not used because it can silently select the
    wrong face on large or overlapping triangles.
    """

    raise_if_cancelled(cancellation_probe)
    vertices = np.asarray(vertices_world_mm, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        raise ArtifactSurfaceMeasurementError("vertices must have shape (N, 3)")
    if not np.isfinite(vertices).all():
        raise ArtifactSurfaceMeasurementError("vertices must be finite")
    if face_array.ndim != 2 or face_array.shape[1] != 3 or face_array.shape[0] < 1:
        raise ArtifactSurfaceMeasurementError("faces must have shape (M, 3)")
    if vertices.shape[0] > MAXIMUM_SURFACE_MEASUREMENT_VERTICES:
        raise ArtifactSurfaceMeasurementError("surface anchor vertex limit exceeded")
    if face_array.shape[0] > MAXIMUM_SURFACE_MEASUREMENT_FACES:
        raise ArtifactSurfaceMeasurementError("surface anchor face limit exceeded")
    if np.any(face_array < 0) or np.any(face_array >= vertices.shape[0]):
        raise ArtifactSurfaceMeasurementError("faces contain an invalid vertex index")
    if source_faces is None:
        raise ArtifactSurfaceMeasurementError(
            "source_faces is required for durable source-order anchors"
        )
    source_face_array = np.asarray(source_faces, dtype=np.int64)
    if source_face_array.shape != face_array.shape:
        raise ArtifactSurfaceMeasurementError(
            "source face rows must match projected face rows"
        )
    if np.any(source_face_array < 0) or np.any(source_face_array >= vertices.shape[0]):
        raise ArtifactSurfaceMeasurementError(
            "source faces contain an invalid vertex index"
        )
    origin = np.asarray(ray_origin_world_mm, dtype=np.float64).reshape(-1)
    direction = np.asarray(ray_direction_world, dtype=np.float64).reshape(-1)
    depth_point = np.asarray(depth_point_world_mm, dtype=np.float64).reshape(-1)
    if origin.shape != (3,) or direction.shape != (3,) or depth_point.shape != (3,):
        raise ArtifactSurfaceMeasurementError("ray and depth values must be vec3")
    if not np.isfinite(origin).all() or not np.isfinite(direction).all() or not np.isfinite(depth_point).all():
        raise ArtifactSurfaceMeasurementError("ray and depth values must be finite")
    direction_norm = float(np.linalg.norm(direction))
    if not math.isfinite(direction_norm) or direction_norm <= 1e-15:
        raise ArtifactSurfaceMeasurementError("ray direction is degenerate")
    direction = direction / direction_norm
    footprint = _strict_int(
        pixel_footprint_um,
        name="pixel_footprint_um",
        minimum=1,
        maximum=MAXIMUM_PIXEL_FOOTPRINT_UM,
    )
    if not isinstance(depth_search_offset_px, (list, tuple)) or len(depth_search_offset_px) != 2:
        raise ArtifactSurfaceMeasurementError("depth_search_offset_px must be int vec2")
    search_offset = [
        _strict_int(
            value,
            name=f"depth_search_offset_px[{index}]",
            minimum=-MAXIMUM_SCREEN_SEARCH_OFFSET_PX,
            maximum=MAXIMUM_SCREEN_SEARCH_OFFSET_PX,
        )
        for index, value in enumerate(depth_search_offset_px)
    ]
    grid = _strict_int(
        coordinate_grid_um,
        name="coordinate_grid_um",
        minimum=1,
        maximum=1000,
    )

    best_face = -1
    best_residual2 = float("inf")
    best_t = float("inf")
    best_weights: tuple[float, float, float] | None = None
    chunk_size = 65_536
    for start in range(0, face_array.shape[0], chunk_size):
        poll_cancellation(cancellation_probe, start // chunk_size, interval=1)
        chunk_faces = face_array[start : start + chunk_size]
        triangles = vertices[chunk_faces]
        edge1 = triangles[:, 1] - triangles[:, 0]
        edge2 = triangles[:, 2] - triangles[:, 0]
        pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        determinant = np.einsum("ij,ij->i", edge1, pvec)
        scale = np.linalg.norm(edge1, axis=1) * np.linalg.norm(edge2, axis=1)
        epsilon = np.maximum(scale * 1e-12, 1e-15)
        valid = np.abs(determinant) > epsilon
        if not bool(np.any(valid)):
            continue
        inverse = np.zeros_like(determinant)
        inverse[valid] = 1.0 / determinant[valid]
        tvec = origin - triangles[:, 0]
        u = np.einsum("ij,ij->i", tvec, pvec) * inverse
        qvec = np.cross(tvec, edge1)
        v = np.einsum("j,ij->i", direction, qvec) * inverse
        t = np.einsum("ij,ij->i", edge2, qvec) * inverse
        valid &= u >= -1e-10
        valid &= v >= -1e-10
        valid &= (u + v) <= 1.0 + 1e-10
        valid &= t >= 0.0
        candidate_indices = np.flatnonzero(valid)
        for local_index in candidate_indices.tolist():
            hit = origin + float(t[local_index]) * direction
            delta = hit - depth_point
            residual2 = float(np.dot(delta, delta))
            face_index = start + int(local_index)
            candidate_t = float(t[local_index])
            if (
                residual2 < best_residual2
                or (
                    residual2 == best_residual2
                    and (candidate_t < best_t or (candidate_t == best_t and face_index < best_face))
                )
            ):
                best_face = face_index
                best_residual2 = residual2
                best_t = candidate_t
                best_weights = (
                    1.0 - float(u[local_index]) - float(v[local_index]),
                    float(u[local_index]),
                    float(v[local_index]),
                )
    raise_if_cancelled(cancellation_probe)
    if best_face < 0 or best_weights is None:
        raise ArtifactSurfaceMeasurementError("screen ray did not intersect the selected mesh")
    depth_tolerance_um = _pick_depth_tolerance_um(
        pixel_footprint_um=footprint,
        grid_um=grid,
    )
    residual_um = math.sqrt(best_residual2) * 1000.0
    if residual_um > float(depth_tolerance_um):
        raise ArtifactSurfaceMeasurementError(
            "framebuffer depth and CPU triangle hit disagree beyond the explicit tolerance"
        )
    projected_row = [int(value) for value in face_array[best_face]]
    source_row = [int(value) for value in source_face_array[best_face]]
    if set(projected_row) != set(source_row):
        raise ArtifactSurfaceMeasurementError(
            "source and projected face rows do not identify the same triangle"
        )
    projected_barycentric = _quantized_barycentric(best_weights)
    source_barycentric = [
        projected_barycentric[projected_row.index(vertex_index)]
        for vertex_index in source_row
    ]
    return {
        "barycentric_numerators": source_barycentric,
        "capture_point_grid": _quantized_point(depth_point, grid_um=grid),
        "depth_match_tolerance_um": depth_tolerance_um,
        "depth_search_offset_px": search_offset,
        "face_index": best_face,
        "face_vertex_indices": source_row,
        "pixel_footprint_um": footprint,
    }


def _validated_arrays(
    vertices_world_mm: object,
    faces: object,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices_world_mm, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        raise ArtifactSurfaceMeasurementError("vertices must have shape (N, 3), N >= 3")
    if not np.isfinite(vertices).all():
        raise ArtifactSurfaceMeasurementError("vertices must be finite")
    if face_array.ndim != 2 or face_array.shape[1] != 3 or face_array.shape[0] < 1:
        raise ArtifactSurfaceMeasurementError("faces must have shape (M, 3), M >= 1")
    if np.any(face_array < 0) or np.any(face_array >= vertices.shape[0]):
        raise ArtifactSurfaceMeasurementError("faces contain an invalid vertex index")
    if vertices.shape[0] > MAXIMUM_SURFACE_MEASUREMENT_VERTICES:
        raise ArtifactSurfaceMeasurementError("surface measurement vertex limit exceeded")
    if face_array.shape[0] > MAXIMUM_SURFACE_MEASUREMENT_FACES:
        raise ArtifactSurfaceMeasurementError("surface measurement face limit exceeded")
    return vertices, face_array


def _quantized_vertices(vertices: np.ndarray, *, grid_um: int) -> np.ndarray:
    scaled = vertices * (1000.0 / float(grid_um))
    if not np.isfinite(scaled).all() or np.max(np.abs(scaled)) > MAXIMUM_SAFE_JSON_INTEGER:
        raise ArtifactSurfaceMeasurementError("vertices exceed the safe quantized range")
    return np.rint(scaled).astype(np.int64)


def _quantized_source_vertices(
    source_vertices: object,
    matrix4x4: object,
    *,
    grid_um: int,
) -> np.ndarray:
    """Transform and quantize with a fixed per-axis operation order.

    Full-array BLAS and a three-row reopen check are allowed to choose
    different reduction kernels.  Explicit NumPy ufunc steps make the
    multiply/add order identical for a whole source and any referenced subset,
    including coordinates exactly at a half-grid boundary.
    """

    source = np.asarray(source_vertices, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or source.shape[0] < 1:
        raise ArtifactSurfaceMeasurementError(
            "source vertices must have shape (N, 3), N >= 1"
        )
    if not np.isfinite(source).all():
        raise ArtifactSurfaceMeasurementError("source vertices must be finite")
    try:
        matrix_value = np.asarray(matrix4x4, dtype=np.float64)
        matrix = require_affine_matrix4x4(matrix_value)
    except (TypeError, ValueError) as exc:
        raise ArtifactSurfaceMeasurementError(
            "source-to-canonical matrix is invalid"
        ) from exc
    factor = 1000.0 / float(grid_um)
    scaled = np.empty(source.shape, dtype=np.float64)
    for axis in range(3):
        first = np.multiply(source[:, 0], float(matrix[axis, 0]))
        second = np.multiply(source[:, 1], float(matrix[axis, 1]))
        third = np.multiply(source[:, 2], float(matrix[axis, 2]))
        combined = np.add(first, second)
        combined = np.add(combined, third)
        combined = np.add(combined, float(matrix[axis, 3]))
        scaled[:, axis] = np.multiply(combined, factor)
    if (
        not np.isfinite(scaled).all()
        or float(np.max(np.abs(scaled))) > MAXIMUM_SAFE_JSON_INTEGER
    ):
        raise ArtifactSurfaceMeasurementError(
            "transformed source vertices exceed the safe quantized range"
        )
    return np.rint(scaled).astype(np.int64)


def _reduced_fraction(numerator: int, denominator: int) -> dict[str, str]:
    if numerator < 0 or denominator <= 0:
        raise ArtifactSurfaceMeasurementError("exact fraction is invalid")
    divisor = math.gcd(numerator, denominator)
    return {
        "denominator": str(denominator // divisor),
        "numerator": str(numerator // divisor),
    }


def _quantized_triangle_has_area(triangle: np.ndarray) -> bool:
    values = np.asarray(triangle, dtype=object)
    if values.shape != (3, 3):
        raise ArtifactSurfaceMeasurementError("quantized triangle must be 3x3")
    edge1 = [int(values[1, axis]) - int(values[0, axis]) for axis in range(3)]
    edge2 = [int(values[2, axis]) - int(values[0, axis]) for axis in range(3)]
    cross = (
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0],
    )
    return any(value != 0 for value in cross)


def _anchor_receipts(
    recipe: Mapping[str, Any],
    quantized_vertices: np.ndarray | Mapping[int, np.ndarray],
    face_array: np.ndarray,
    *,
    cancellation_probe: CancellationProbe | None,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[int], list[int]]:
    anchors = recipe["anchors"]
    assert isinstance(anchors, list)
    denominator = int(recipe["barycentric_denominator"])
    grid_um = int(recipe["coordinate_grid_um"])
    edge_threshold = int(recipe["edge_review_threshold_ppb"])
    receipts: list[dict[str, Any]] = []
    point_numerators: list[np.ndarray] = []
    residual_um_values: list[int] = []
    pixel_footprints: list[int] = []
    for index, anchor in enumerate(anchors):
        poll_cancellation(cancellation_probe, index, interval=1)
        assert isinstance(anchor, Mapping)
        face_index = int(anchor["face_index"])
        expected_vertices = [int(value) for value in anchor["face_vertex_indices"]]
        observed_vertices = [int(value) for value in face_array[face_index]]
        if observed_vertices != expected_vertices:
            raise ArtifactSurfaceMeasurementError(
                f"surface anchor {index} face vertex identity changed"
            )
        if isinstance(quantized_vertices, Mapping):
            try:
                triangle = np.asarray(
                    [quantized_vertices[value] for value in observed_vertices],
                    dtype=np.int64,
                )
            except KeyError as exc:  # pragma: no cover - internal lookup invariant
                raise ArtifactSurfaceMeasurementError(
                    f"surface anchor {index} vertex was not quantized"
                ) from exc
        else:
            triangle = quantized_vertices[
                np.asarray(observed_vertices, dtype=np.int64)
            ]
        if not _quantized_triangle_has_area(triangle):
            raise ArtifactSurfaceMeasurementError(
                f"surface anchor {index} references a quantization-degenerate face"
            )
        weights = np.asarray(anchor["barycentric_numerators"], dtype=np.int64)
        point_numerator = np.sum(
            triangle.astype(object) * weights.astype(object)[:, None], axis=0
        )
        point_numerator_int = np.asarray(
            [int(value) for value in point_numerator], dtype=object
        )
        capture = np.asarray(anchor["capture_point_grid"], dtype=object)
        delta = point_numerator_int - capture * denominator
        squared = sum(int(value) * int(value) for value in delta)
        with localcontext() as context:
            context.prec = 60
            residual_um = (
                Decimal(squared).sqrt()
                * Decimal(grid_um)
                / Decimal(denominator)
            )
        tolerance_um = int(anchor["depth_match_tolerance_um"])
        if residual_um > Decimal(tolerance_um):
            raise ArtifactSurfaceMeasurementError(
                f"surface anchor {index} exceeds its captured depth tolerance"
            )
        minimum_weight = int(min(int(value) for value in weights))
        edge_status = (
            "on_edge"
            if minimum_weight == 0
            else ("near_edge" if minimum_weight <= edge_threshold else "interior")
        )
        resolved_decimal = [
            _fixed_decimal(
                Decimal(int(value))
                * Decimal(grid_um)
                / Decimal(denominator)
                / Decimal(1000)
            )
            for value in point_numerator_int
        ]
        residual_text = _fixed_decimal(residual_um / Decimal(1000))
        receipts.append(
            {
                "barycentric_numerators": [int(value) for value in weights],
                "capture_point_grid": [int(value) for value in capture],
                "capture_residual_mm_decimal": residual_text,
                "depth_match_tolerance_um": tolerance_um,
                "depth_search_offset_px": [
                    int(value) for value in anchor["depth_search_offset_px"]
                ],
                "edge_status": edge_status,
                "face_index": face_index,
                "face_vertex_indices": observed_vertices,
                "minimum_barycentric_ppb": minimum_weight,
                "pixel_footprint_um": int(anchor["pixel_footprint_um"]),
                "resolved_point_mm_decimal": resolved_decimal,
                "resolved_point_numerator_grid_bary": [
                    str(int(value)) for value in point_numerator_int
                ],
            }
        )
        point_numerators.append(point_numerator_int)
        residual_um_values.append(int(residual_um.to_integral_value(rounding=ROUND_HALF_EVEN)))
        pixel_footprints.append(int(anchor["pixel_footprint_um"]))
    return receipts, point_numerators, residual_um_values, pixel_footprints


def _distance_measurement(
    point_numerators: Sequence[np.ndarray],
    *,
    grid_um: int,
) -> dict[str, Any]:
    if len(point_numerators) != 2:
        raise ArtifactSurfaceMeasurementError("surface distance requires two anchors")
    delta = point_numerators[1] - point_numerators[0]
    squared_grid_bary = sum(int(value) * int(value) for value in delta)
    if squared_grid_bary <= 0:
        raise ArtifactSurfaceMeasurementError("surface distance anchors must be distinct")
    numerator = squared_grid_bary * grid_um**2
    denominator = BARYCENTRIC_DENOMINATOR**2 * 1000**2
    exact = _reduced_fraction(numerator, denominator)
    with localcontext() as context:
        context.prec = 60
        distance = (
            Decimal(int(exact["numerator"]))
            / Decimal(int(exact["denominator"]))
        ).sqrt()
    return {
        "distance_mm_decimal": _fixed_decimal(distance),
        "meaning": SURFACE_DISTANCE_MEANING,
        "result_decimal_places": RESULT_DECIMAL_PLACES,
        "squared_distance_exact_mm2": exact,
        "status": "available",
    }


def _normalized_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis_index = int(np.argmin(np.abs(normal)))
    axis = np.zeros(3, dtype=np.float64)
    axis[axis_index] = 1.0
    first = np.cross(normal, axis)
    first_norm = float(np.linalg.norm(first))
    if not math.isfinite(first_norm) or first_norm <= 1e-15:
        raise ArtifactSurfaceMeasurementError("diameter plane basis is degenerate")
    first = first / first_norm
    second = np.cross(normal, first)
    second_norm = float(np.linalg.norm(second))
    if not math.isfinite(second_norm) or second_norm <= 1e-15:
        raise ArtifactSurfaceMeasurementError("diameter plane basis is degenerate")
    return first, second / second_norm


def _diameter_measurement(
    point_numerators: Sequence[np.ndarray],
    *,
    grid_um: int,
    maximum_fit_condition: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    if len(point_numerators) < 3:
        raise ArtifactSurfaceMeasurementError("surface diameter requires three anchors")
    # Subtract one exact integer anchor before any float conversion.  Directly
    # converting absolute canonical coordinates loses micrometre-scale circles
    # when an archaeological survey uses a very large site/grid origin.
    integer_origin = np.asarray(point_numerators[0], dtype=object)
    relative_numerators = [
        np.asarray(point, dtype=object) - integer_origin
        for point in point_numerators
    ]
    points = np.asarray(
        [
            [
                float(int(value)) * float(grid_um)
                / float(BARYCENTRIC_DENOMINATOR)
                / 1000.0
                for value in point
            ]
            for point in relative_numerators
        ],
        dtype=np.float64,
    )
    centroid = np.asarray(
        [math.fsum(float(value) for value in points[:, axis]) / len(points) for axis in range(3)],
        dtype=np.float64,
    )
    centered = points - centroid
    covariance = np.asarray(
        [
            [math.fsum(float(value) for value in centered[:, i] * centered[:, j]) for j in range(3)]
            for i in range(3)
        ],
        dtype=np.float64,
    )
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError as exc:
        raise ArtifactSurfaceMeasurementError("diameter plane fit failed") from exc
    if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
        raise ArtifactSurfaceMeasurementError("diameter plane fit is non-finite")
    scale_eigen = float(max(float(eigenvalues[-1]), 0.0))
    if scale_eigen <= 0.0 or float(eigenvalues[1]) <= scale_eigen * 1e-14:
        raise ArtifactSurfaceMeasurementError("diameter anchors are collinear or unstable")
    if float(eigenvalues[1] - eigenvalues[0]) <= scale_eigen * 1e-12:
        raise ArtifactSurfaceMeasurementError(
            "diameter best-fit plane is ambiguous or unstable"
        )
    normal = np.asarray(eigenvectors[:, 0], dtype=np.float64)
    dominant = int(np.argmax(np.abs(normal)))
    if float(normal[dominant]) < 0.0:
        normal = -normal
    first, second = _normalized_plane_basis(normal)
    plane_offsets = centered @ normal
    x = centered @ first
    y = centered @ second
    radial_scale = float(np.max(np.hypot(x, y)))
    if not math.isfinite(radial_scale) or radial_scale <= 1e-12:
        raise ArtifactSurfaceMeasurementError("diameter anchors have no radial extent")
    xn = x / radial_scale
    yn = y / radial_scale
    design = np.column_stack((2.0 * xn, 2.0 * yn, np.ones(len(points))))
    target = xn * xn + yn * yn
    try:
        solution, _residuals, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ArtifactSurfaceMeasurementError("diameter circle fit failed") from exc
    if int(rank) != 3 or singular.shape != (3,) or float(singular[-1]) <= 0.0:
        raise ArtifactSurfaceMeasurementError("diameter anchors are circle-fit degenerate")
    condition = float(singular[0] / singular[-1])
    if not math.isfinite(condition) or condition > float(maximum_fit_condition):
        raise ArtifactSurfaceMeasurementError("diameter fit condition exceeds its policy")
    center_x = float(solution[0]) * radial_scale
    center_y = float(solution[1]) * radial_scale
    radius_squared_normalized = float(
        solution[2] + solution[0] * solution[0] + solution[1] * solution[1]
    )
    if not math.isfinite(radius_squared_normalized) or radius_squared_normalized <= 0.0:
        raise ArtifactSurfaceMeasurementError("diameter fit produced a non-positive radius")
    radius = math.sqrt(radius_squared_normalized) * radial_scale
    center_relative = centroid + center_x * first + center_y * second
    radial_distances = np.hypot(x - center_x, y - center_y)
    radial_residuals = radial_distances - radius
    plane_rms = math.sqrt(math.fsum(float(value * value) for value in plane_offsets) / len(points))
    plane_max = float(np.max(np.abs(plane_offsets)))
    radial_rms = math.sqrt(math.fsum(float(value * value) for value in radial_residuals) / len(points))
    radial_max = float(np.max(np.abs(radial_residuals)))
    if not np.isfinite(
        [
            radius,
            *center_relative,
            *normal,
            plane_rms,
            plane_max,
            radial_rms,
            radial_max,
        ]
    ).all():
        raise ArtifactSurfaceMeasurementError("diameter fit produced non-finite evidence")
    with localcontext() as decimal_context:
        decimal_context.prec = 60
        origin_scale = (
            Decimal(grid_um)
            / Decimal(BARYCENTRIC_DENOMINATOR)
            / Decimal(1000)
        )
        center_decimal = [
            Decimal(int(integer_origin[axis])) * origin_scale
            + Decimal(str(float(center_relative[axis])))
            for axis in range(3)
        ]
    measurement = {
        "center_mm_decimal": [_fixed_decimal(value) for value in center_decimal],
        "condition_number_decimal": _fixed_decimal(condition),
        "diameter_mm_decimal": _fixed_decimal(radius * 2.0),
        "fit_policy": SURFACE_DIAMETER_FIT_POLICY,
        "normal_unit_decimal": [
            _fixed_decimal(value, _NORMAL_QUANTUM) for value in normal
        ],
        "plane_max_residual_mm_decimal": _fixed_decimal(plane_max),
        "plane_rms_residual_mm_decimal": _fixed_decimal(plane_rms),
        "radial_max_residual_mm_decimal": _fixed_decimal(radial_max),
        "radial_rms_residual_mm_decimal": _fixed_decimal(radial_rms),
        "radius_mm_decimal": _fixed_decimal(radius),
        "result_decimal_places": RESULT_DECIMAL_PLACES,
        "sample_count": len(points),
        "status": "available",
    }
    evidence = {
        "plane_max_mm": float(Decimal(measurement["plane_max_residual_mm_decimal"])),
        "plane_rms_mm": float(Decimal(measurement["plane_rms_residual_mm_decimal"])),
        "radial_max_mm": float(Decimal(measurement["radial_max_residual_mm_decimal"])),
        "radial_rms_mm": float(Decimal(measurement["radial_rms_residual_mm_decimal"])),
    }
    return measurement, evidence


def _quality_receipt(
    recipe: Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
    *,
    fit_evidence: Mapping[str, float] | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    near_edge_count = sum(1 for anchor in anchors if anchor["edge_status"] != "interior")
    screen_search_count = sum(
        1
        for anchor in anchors
        if any(int(value) != 0 for value in anchor["depth_search_offset_px"])
    )
    if near_edge_count:
        reasons.append("anchor_near_triangle_edge")
    if screen_search_count:
        reasons.append("depth_search_offset_used")
    fit_threshold_mm = float(recipe["fit_review_threshold_um"]) / 1000.0
    if fit_evidence is not None:
        if max(
            float(fit_evidence["plane_rms_mm"]),
            float(fit_evidence["plane_max_mm"]),
        ) > fit_threshold_mm:
            reasons.append("plane_fit_residual")
        if max(
            float(fit_evidence["radial_rms_mm"]),
            float(fit_evidence["radial_max_mm"]),
        ) > fit_threshold_mm:
            reasons.append("circle_fit_residual")
    maximum_capture = max(Decimal(str(anchor["capture_residual_mm_decimal"])) for anchor in anchors)
    maximum_pixel = max(int(anchor["pixel_footprint_um"]) for anchor in anchors)
    if maximum_pixel > int(recipe["fit_review_threshold_um"]):
        reasons.append("coarse_pixel_footprint")
    reasons = sorted(set(reasons))
    return {
        "edge_review_threshold_ppb": int(recipe["edge_review_threshold_ppb"]),
        "fit_review_threshold_um": int(recipe["fit_review_threshold_um"]),
        "maximum_capture_residual_mm_decimal": _fixed_decimal(maximum_capture),
        "maximum_pixel_footprint_um": maximum_pixel,
        "near_edge_anchor_count": near_edge_count,
        "review_reasons": reasons,
        "screen_search_anchor_count": screen_search_count,
        "status": "review" if reasons else "pass",
    }


def _referenced_vertex_ids(recipe: Mapping[str, Any]) -> np.ndarray:
    anchors = recipe["anchors"]
    assert isinstance(anchors, list)
    values = sorted(
        {
            int(vertex_id)
            for anchor in anchors
            for vertex_id in anchor["face_vertex_indices"]
        }
    )
    if not values:  # pragma: no cover - closed recipe requires at least two anchors
        raise ArtifactSurfaceMeasurementError("surface recipe has no referenced vertices")
    return np.asarray(values, dtype=np.int64)


def extract_surface_measurement(
    vertices_world_mm: object,
    faces: object,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reattach captured anchors and compute one closed measurement receipt."""

    raise_if_cancelled(cancellation_probe)
    validated_recipe = validate_surface_measurement_recipe(recipe)
    vertices, face_array = _validated_arrays(vertices_world_mm, faces)
    if int(validated_recipe["source_vertex_count"]) != int(vertices.shape[0]):
        raise ArtifactSurfaceMeasurementError("recipe vertex count does not match geometry")
    if int(validated_recipe["source_face_count"]) != int(face_array.shape[0]):
        raise ArtifactSurfaceMeasurementError("recipe face count does not match geometry")
    grid_um = int(validated_recipe["coordinate_grid_um"])
    referenced_ids = _referenced_vertex_ids(validated_recipe)
    quantized_rows = _quantized_vertices(vertices[referenced_ids], grid_um=grid_um)
    quantized = {
        int(vertex_id): quantized_rows[index]
        for index, vertex_id in enumerate(referenced_ids.tolist())
    }
    return _extract_surface_measurement_from_quantized(
        quantized,
        face_array,
        validated_recipe,
        cancellation_probe=cancellation_probe,
    )


def extract_surface_measurement_from_source(
    source_vertices: object,
    source_faces: object,
    source_to_canonical_mm_matrix4x4: object,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute with the same deterministic source transform used on reopen."""

    raise_if_cancelled(cancellation_probe)
    validated_recipe = validate_surface_measurement_recipe(recipe)
    vertices, face_array = _validated_arrays(source_vertices, source_faces)
    if int(validated_recipe["source_vertex_count"]) != int(vertices.shape[0]):
        raise ArtifactSurfaceMeasurementError(
            "recipe vertex count does not match source geometry"
        )
    if int(validated_recipe["source_face_count"]) != int(face_array.shape[0]):
        raise ArtifactSurfaceMeasurementError(
            "recipe face count does not match source geometry"
        )
    referenced_ids = _referenced_vertex_ids(validated_recipe)
    quantized_rows = _quantized_source_vertices(
        vertices[referenced_ids],
        source_to_canonical_mm_matrix4x4,
        grid_um=int(validated_recipe["coordinate_grid_um"]),
    )
    quantized = {
        int(vertex_id): quantized_rows[index]
        for index, vertex_id in enumerate(referenced_ids.tolist())
    }
    return _extract_surface_measurement_from_quantized(
        quantized,
        face_array,
        validated_recipe,
        cancellation_probe=cancellation_probe,
    )


def _extract_surface_measurement_from_quantized(
    quantized: np.ndarray | Mapping[int, np.ndarray],
    face_array: np.ndarray,
    validated_recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    grid_um = int(validated_recipe["coordinate_grid_um"])
    anchors, point_numerators, _residual_um, _footprints = _anchor_receipts(
        validated_recipe,
        quantized,
        face_array,
        cancellation_probe=cancellation_probe,
    )
    raise_if_cancelled(cancellation_probe)
    kind = str(validated_recipe["kind"])
    fit_evidence: Mapping[str, float] | None = None
    if kind == "surface_distance":
        measurement = _distance_measurement(point_numerators, grid_um=grid_um)
    else:
        measurement, fit_evidence = _diameter_measurement(
            point_numerators,
            grid_um=grid_um,
            maximum_fit_condition=int(validated_recipe["maximum_fit_condition"]),
        )
    quality = _quality_receipt(validated_recipe, anchors, fit_evidence=fit_evidence)
    receipt = validate_surface_measurement_receipt(
        {
            "anchor_basis": SURFACE_ANCHOR_BASIS,
            "anchors": anchors,
            "barycentric_denominator": BARYCENTRIC_DENOMINATOR,
            "coordinate_grid_um": grid_um,
            "coordinate_space": SURFACE_MEASUREMENT_COORDINATE_SPACE,
            "input_face_count": int(face_array.shape[0]),
            "input_vertex_count": int(validated_recipe["source_vertex_count"]),
            "kind": kind,
            "measurement": measurement,
            "pick_method": SURFACE_PICK_METHOD,
            "quality": quality,
            "rounding_mode": SURFACE_MEASUREMENT_ROUNDING,
            "schema_version": SURFACE_MEASUREMENT_SCHEMA_VERSION,
        }
    )
    qc = _qc_from_receipt(receipt)
    raise_if_cancelled(cancellation_probe)
    return receipt, qc


def _validate_exact_fraction(value: object, *, name: str) -> dict[str, str]:
    fraction = _exact_mapping(value, {"denominator", "numerator"}, name=name)
    numerator = int(_integer_text(fraction["numerator"], name=f"{name}.numerator"))
    denominator = int(_integer_text(fraction["denominator"], name=f"{name}.denominator"))
    if denominator <= 0 or numerator < 0 or math.gcd(numerator, denominator) != 1:
        raise ArtifactSurfaceMeasurementError(f"{name} must be a reduced non-negative fraction")
    return {"denominator": str(denominator), "numerator": str(numerator)}


def _validate_receipt_anchor(value: object, *, index: int) -> dict[str, Any]:
    anchor = _exact_mapping(
        value,
        {
            "barycentric_numerators",
            "capture_point_grid",
            "capture_residual_mm_decimal",
            "depth_match_tolerance_um",
            "depth_search_offset_px",
            "edge_status",
            "face_index",
            "face_vertex_indices",
            "minimum_barycentric_ppb",
            "pixel_footprint_um",
            "resolved_point_mm_decimal",
            "resolved_point_numerator_grid_bary",
        },
        name=f"receipt anchor {index}",
    )
    barycentric = _vec3_int(
        anchor["barycentric_numerators"],
        name=f"receipt anchor {index}.barycentric_numerators",
        minimum=0,
        maximum=BARYCENTRIC_DENOMINATOR,
    )
    if sum(barycentric) != BARYCENTRIC_DENOMINATOR:
        raise ArtifactSurfaceMeasurementError("receipt barycentric sum is invalid")
    minimum_barycentric = _strict_int(
        anchor["minimum_barycentric_ppb"],
        name=f"receipt anchor {index}.minimum_barycentric_ppb",
        minimum=0,
        maximum=BARYCENTRIC_DENOMINATOR,
    )
    if minimum_barycentric != min(barycentric):
        raise ArtifactSurfaceMeasurementError("receipt minimum barycentric is invalid")
    edge_status = str(anchor["edge_status"])
    if edge_status not in {"interior", "near_edge", "on_edge"}:
        raise ArtifactSurfaceMeasurementError("receipt edge status is invalid")
    return {
        "barycentric_numerators": barycentric,
        "capture_point_grid": _vec3_int(
            anchor["capture_point_grid"],
            name=f"receipt anchor {index}.capture_point_grid",
            minimum=-MAXIMUM_SAFE_JSON_INTEGER,
            maximum=MAXIMUM_SAFE_JSON_INTEGER,
        ),
        "capture_residual_mm_decimal": _decimal_text(
            anchor["capture_residual_mm_decimal"],
            name=f"receipt anchor {index}.capture_residual_mm_decimal",
        ),
        "depth_match_tolerance_um": _strict_int(
            anchor["depth_match_tolerance_um"],
            name=f"receipt anchor {index}.depth_match_tolerance_um",
            minimum=DEFAULT_MINIMUM_DEPTH_MATCH_TOLERANCE_UM,
            maximum=MAXIMUM_DEPTH_MATCH_TOLERANCE_UM,
        ),
        "depth_search_offset_px": _vec2_int(
            anchor["depth_search_offset_px"],
            name=f"receipt anchor {index}.depth_search_offset_px",
            minimum=-MAXIMUM_SCREEN_SEARCH_OFFSET_PX,
            maximum=MAXIMUM_SCREEN_SEARCH_OFFSET_PX,
        ),
        "edge_status": edge_status,
        "face_index": _strict_int(
            anchor["face_index"],
            name=f"receipt anchor {index}.face_index",
            minimum=0,
            maximum=MAXIMUM_SURFACE_MEASUREMENT_FACES - 1,
        ),
        "face_vertex_indices": _vec3_int(
            anchor["face_vertex_indices"],
            name=f"receipt anchor {index}.face_vertex_indices",
            minimum=0,
            maximum=MAXIMUM_SURFACE_MEASUREMENT_VERTICES - 1,
        ),
        "minimum_barycentric_ppb": minimum_barycentric,
        "pixel_footprint_um": _strict_int(
            anchor["pixel_footprint_um"],
            name=f"receipt anchor {index}.pixel_footprint_um",
            minimum=1,
            maximum=MAXIMUM_PIXEL_FOOTPRINT_UM,
        ),
        "resolved_point_mm_decimal": _vec3_decimal(
            anchor["resolved_point_mm_decimal"],
            name=f"receipt anchor {index}.resolved_point_mm_decimal",
        ),
        "resolved_point_numerator_grid_bary": _vec3_text(
            anchor["resolved_point_numerator_grid_bary"],
            name=f"receipt anchor {index}.resolved_point_numerator_grid_bary",
        ),
    }


def _validate_receipt_anchor_derivations(
    anchor: Mapping[str, Any],
    *,
    index: int,
    coordinate_grid_um: int,
    edge_review_threshold_ppb: int,
    input_vertex_count: int,
    input_face_count: int,
) -> None:
    """Reject syntactically valid but coherently forged derived pick fields."""

    face_index = int(anchor["face_index"])
    vertices = [int(value) for value in anchor["face_vertex_indices"]]
    if face_index >= input_face_count:
        raise ArtifactSurfaceMeasurementError(
            f"receipt anchor {index} face exceeds input_face_count"
        )
    if len(set(vertices)) != 3 or any(
        value >= input_vertex_count for value in vertices
    ):
        raise ArtifactSurfaceMeasurementError(
            f"receipt anchor {index} vertex identity is invalid"
        )
    footprint = int(anchor["pixel_footprint_um"])
    expected_tolerance = _pick_depth_tolerance_um(
        pixel_footprint_um=footprint,
        grid_um=coordinate_grid_um,
    )
    if int(anchor["depth_match_tolerance_um"]) != expected_tolerance:
        raise ArtifactSurfaceMeasurementError(
            f"receipt anchor {index} depth tolerance is not derived from its "
            "pixel/grid policy"
        )

    weights = [int(value) for value in anchor["barycentric_numerators"]]
    minimum_weight = min(weights)
    expected_edge = (
        "on_edge"
        if minimum_weight == 0
        else (
            "near_edge"
            if minimum_weight <= edge_review_threshold_ppb
            else "interior"
        )
    )
    if anchor["edge_status"] != expected_edge:
        raise ArtifactSurfaceMeasurementError(
            f"receipt anchor {index} edge status is not derived from barycentrics"
        )

    point_numerators = [
        int(value) for value in anchor["resolved_point_numerator_grid_bary"]
    ]
    maximum_point_numerator = MAXIMUM_SAFE_JSON_INTEGER * BARYCENTRIC_DENOMINATOR
    if any(abs(value) > maximum_point_numerator for value in point_numerators):
        raise ArtifactSurfaceMeasurementError(
            f"receipt anchor {index} exact point exceeds the safe source/grid range"
        )
    expected_point_decimals = [
        _fixed_decimal(
            Decimal(value)
            * Decimal(coordinate_grid_um)
            / Decimal(BARYCENTRIC_DENOMINATOR)
            / Decimal(1000)
        )
        for value in point_numerators
    ]
    if list(anchor["resolved_point_mm_decimal"]) != expected_point_decimals:
        raise ArtifactSurfaceMeasurementError(
            f"receipt anchor {index} resolved decimal is not derived from its "
            "exact numerator"
        )

    capture = [int(value) for value in anchor["capture_point_grid"]]
    delta = [
        point_numerators[axis] - capture[axis] * BARYCENTRIC_DENOMINATOR
        for axis in range(3)
    ]
    squared = sum(value * value for value in delta)
    with localcontext() as context:
        context.prec = 60
        residual_um = (
            Decimal(squared).sqrt()
            * Decimal(coordinate_grid_um)
            / Decimal(BARYCENTRIC_DENOMINATOR)
        )
    expected_residual = _fixed_decimal(residual_um / Decimal(1000))
    if anchor["capture_residual_mm_decimal"] != expected_residual:
        raise ArtifactSurfaceMeasurementError(
            f"receipt anchor {index} capture residual is not derived from its points"
        )
    if residual_um > Decimal(expected_tolerance):
        raise ArtifactSurfaceMeasurementError(
            f"receipt anchor {index} exceeds its depth-match tolerance"
        )


def _validate_measurement(value: object, *, kind: str, anchor_count: int) -> dict[str, Any]:
    if kind == "surface_distance":
        measurement = _exact_mapping(
            value,
            {
                "distance_mm_decimal",
                "meaning",
                "result_decimal_places",
                "squared_distance_exact_mm2",
                "status",
            },
            name="distance measurement",
        )
        exact = _validate_exact_fraction(
            measurement["squared_distance_exact_mm2"],
            name="squared_distance_exact_mm2",
        )
        with localcontext() as context:
            context.prec = 60
            expected_distance = _fixed_decimal(
                (
                    Decimal(int(exact["numerator"]))
                    / Decimal(int(exact["denominator"]))
                ).sqrt()
            )
        distance = _decimal_text(
            measurement["distance_mm_decimal"], name="distance_mm_decimal"
        )
        if distance != expected_distance or Decimal(distance) <= 0:
            raise ArtifactSurfaceMeasurementError(
                "distance decimal does not match its exact squared distance"
            )
        if (
            measurement["meaning"] != SURFACE_DISTANCE_MEANING
            or measurement["status"] != "available"
            or measurement["result_decimal_places"] != RESULT_DECIMAL_PLACES
            or anchor_count != 2
        ):
            raise ArtifactSurfaceMeasurementError("distance measurement semantics are invalid")
        return {
            "distance_mm_decimal": distance,
            "meaning": SURFACE_DISTANCE_MEANING,
            "result_decimal_places": RESULT_DECIMAL_PLACES,
            "squared_distance_exact_mm2": exact,
            "status": "available",
        }
    measurement = _exact_mapping(
        value,
        {
            "center_mm_decimal",
            "condition_number_decimal",
            "diameter_mm_decimal",
            "fit_policy",
            "normal_unit_decimal",
            "plane_max_residual_mm_decimal",
            "plane_rms_residual_mm_decimal",
            "radial_max_residual_mm_decimal",
            "radial_rms_residual_mm_decimal",
            "radius_mm_decimal",
            "result_decimal_places",
            "sample_count",
            "status",
        },
        name="diameter measurement",
    )
    diameter = _decimal_text(measurement["diameter_mm_decimal"], name="diameter_mm_decimal")
    radius = _decimal_text(measurement["radius_mm_decimal"], name="radius_mm_decimal")
    if Decimal(diameter) <= 0 or Decimal(radius) <= 0:
        raise ArtifactSurfaceMeasurementError("diameter and radius must be positive")
    if abs(Decimal(diameter) - Decimal(radius) * 2) > _RESULT_QUANTUM:
        raise ArtifactSurfaceMeasurementError("diameter does not match radius")
    sample_count = _strict_int(
        measurement["sample_count"],
        name="diameter.sample_count",
        minimum=3,
        maximum=MAXIMUM_DIAMETER_ANCHORS,
    )
    if sample_count != anchor_count:
        raise ArtifactSurfaceMeasurementError("diameter sample count does not match anchors")
    if (
        measurement["fit_policy"] != SURFACE_DIAMETER_FIT_POLICY
        or measurement["status"] != "available"
        or measurement["result_decimal_places"] != RESULT_DECIMAL_PLACES
    ):
        raise ArtifactSurfaceMeasurementError("diameter measurement semantics are invalid")
    return {
        "center_mm_decimal": _vec3_decimal(
            measurement["center_mm_decimal"], name="diameter.center_mm_decimal"
        ),
        "condition_number_decimal": _decimal_text(
            measurement["condition_number_decimal"], name="diameter.condition_number_decimal"
        ),
        "diameter_mm_decimal": diameter,
        "fit_policy": SURFACE_DIAMETER_FIT_POLICY,
        "normal_unit_decimal": _vec3_decimal(
            measurement["normal_unit_decimal"],
            name="diameter.normal_unit_decimal",
            normal=True,
        ),
        "plane_max_residual_mm_decimal": _decimal_text(
            measurement["plane_max_residual_mm_decimal"], name="diameter.plane_max_residual"
        ),
        "plane_rms_residual_mm_decimal": _decimal_text(
            measurement["plane_rms_residual_mm_decimal"], name="diameter.plane_rms_residual"
        ),
        "radial_max_residual_mm_decimal": _decimal_text(
            measurement["radial_max_residual_mm_decimal"], name="diameter.radial_max_residual"
        ),
        "radial_rms_residual_mm_decimal": _decimal_text(
            measurement["radial_rms_residual_mm_decimal"], name="diameter.radial_rms_residual"
        ),
        "radius_mm_decimal": radius,
        "result_decimal_places": RESULT_DECIMAL_PLACES,
        "sample_count": sample_count,
        "status": "available",
    }


def _validate_quality(value: object, anchors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    quality = _exact_mapping(
        value,
        {
            "edge_review_threshold_ppb",
            "fit_review_threshold_um",
            "maximum_capture_residual_mm_decimal",
            "maximum_pixel_footprint_um",
            "near_edge_anchor_count",
            "review_reasons",
            "screen_search_anchor_count",
            "status",
        },
        name="surface measurement quality",
    )
    reasons_value = quality["review_reasons"]
    if not isinstance(reasons_value, (list, tuple)) or any(
        reason not in {
            "anchor_near_triangle_edge",
            "circle_fit_residual",
            "coarse_pixel_footprint",
            "depth_search_offset_used",
            "plane_fit_residual",
        }
        for reason in reasons_value
    ):
        raise ArtifactSurfaceMeasurementError("quality review reasons are invalid")
    reasons = [str(reason) for reason in reasons_value]
    if reasons != sorted(set(reasons)):
        raise ArtifactSurfaceMeasurementError("quality review reasons must be sorted and unique")
    status = str(quality["status"])
    if status != ("review" if reasons else "pass"):
        raise ArtifactSurfaceMeasurementError("quality status does not match review reasons")
    near_count = _strict_int(
        quality["near_edge_anchor_count"],
        name="quality.near_edge_anchor_count",
        minimum=0,
        maximum=len(anchors),
    )
    observed_near = sum(1 for anchor in anchors if anchor["edge_status"] != "interior")
    if near_count != observed_near:
        raise ArtifactSurfaceMeasurementError("quality near-edge count is invalid")
    search_count = _strict_int(
        quality["screen_search_anchor_count"],
        name="quality.screen_search_anchor_count",
        minimum=0,
        maximum=len(anchors),
    )
    observed_search = sum(
        1
        for anchor in anchors
        if any(int(value) != 0 for value in anchor["depth_search_offset_px"])
    )
    if search_count != observed_search:
        raise ArtifactSurfaceMeasurementError("quality screen-search count is invalid")
    maximum_capture = max(
        Decimal(str(anchor["capture_residual_mm_decimal"])) for anchor in anchors
    )
    maximum_pixel = max(int(anchor["pixel_footprint_um"]) for anchor in anchors)
    if quality["maximum_capture_residual_mm_decimal"] != _fixed_decimal(maximum_capture):
        raise ArtifactSurfaceMeasurementError("quality maximum capture residual is invalid")
    if quality["maximum_pixel_footprint_um"] != maximum_pixel:
        raise ArtifactSurfaceMeasurementError("quality maximum pixel footprint is invalid")
    return {
        "edge_review_threshold_ppb": _strict_int(
            quality["edge_review_threshold_ppb"],
            name="quality.edge_review_threshold_ppb",
            minimum=0,
            maximum=BARYCENTRIC_DENOMINATOR // 3,
        ),
        "fit_review_threshold_um": _strict_int(
            quality["fit_review_threshold_um"],
            name="quality.fit_review_threshold_um",
            minimum=1,
            maximum=MAXIMUM_PIXEL_FOOTPRINT_UM,
        ),
        "maximum_capture_residual_mm_decimal": _decimal_text(
            quality["maximum_capture_residual_mm_decimal"],
            name="quality.maximum_capture_residual_mm_decimal",
        ),
        "maximum_pixel_footprint_um": maximum_pixel,
        "near_edge_anchor_count": near_count,
        "review_reasons": reasons,
        "screen_search_anchor_count": search_count,
        "status": status,
    }


def validate_surface_measurement_receipt(value: object) -> dict[str, Any]:
    receipt = _exact_mapping(
        value,
        {
            "anchor_basis",
            "anchors",
            "barycentric_denominator",
            "coordinate_grid_um",
            "coordinate_space",
            "input_face_count",
            "input_vertex_count",
            "kind",
            "measurement",
            "pick_method",
            "quality",
            "rounding_mode",
            "schema_version",
        },
        name="surface measurement receipt",
    )
    kind = str(receipt["kind"])
    if kind not in {"surface_distance", "surface_diameter"}:
        raise ArtifactSurfaceMeasurementError("receipt kind is unsupported")
    anchor_values = receipt["anchors"]
    if not isinstance(anchor_values, (list, tuple)):
        raise ArtifactSurfaceMeasurementError("receipt anchors must be a list")
    coordinate_grid_um = _strict_int(
        receipt["coordinate_grid_um"],
        name="receipt.coordinate_grid_um",
        minimum=1,
        maximum=1000,
    )
    input_face_count = _strict_int(
        receipt["input_face_count"],
        name="receipt.input_face_count",
        minimum=1,
        maximum=MAXIMUM_SURFACE_MEASUREMENT_FACES,
    )
    input_vertex_count = _strict_int(
        receipt["input_vertex_count"],
        name="receipt.input_vertex_count",
        minimum=3,
        maximum=MAXIMUM_SURFACE_MEASUREMENT_VERTICES,
    )
    minimum = 2 if kind == "surface_distance" else 3
    maximum = 2 if kind == "surface_distance" else MAXIMUM_DIAMETER_ANCHORS
    if len(anchor_values) < minimum or len(anchor_values) > maximum:
        raise ArtifactSurfaceMeasurementError("receipt anchor count is invalid")
    anchors = [
        _validate_receipt_anchor(anchor, index=index)
        for index, anchor in enumerate(anchor_values)
    ]
    measurement = _validate_measurement(
        receipt["measurement"], kind=kind, anchor_count=len(anchors)
    )
    quality = _validate_quality(receipt["quality"], anchors)
    for index, anchor in enumerate(anchors):
        _validate_receipt_anchor_derivations(
            anchor,
            index=index,
            coordinate_grid_um=coordinate_grid_um,
            edge_review_threshold_ppb=int(quality["edge_review_threshold_ppb"]),
            input_vertex_count=input_vertex_count,
            input_face_count=input_face_count,
        )
    point_numerators = [
        np.asarray(
            [int(value) for value in anchor["resolved_point_numerator_grid_bary"]],
            dtype=object,
        )
        for anchor in anchors
    ]
    fit_evidence: Mapping[str, float] | None = None
    if kind == "surface_distance":
        expected_measurement = _distance_measurement(
            point_numerators,
            grid_um=coordinate_grid_um,
        )
    else:
        expected_measurement, fit_evidence = _diameter_measurement(
            point_numerators,
            grid_um=coordinate_grid_um,
            maximum_fit_condition=MAXIMUM_SAFE_JSON_INTEGER,
        )
    if measurement != expected_measurement:
        raise ArtifactSurfaceMeasurementError(
            "surface measurement result is not derived from its anchors"
        )
    expected_quality = _quality_receipt(
        quality,
        anchors,
        fit_evidence=fit_evidence,
    )
    if quality != expected_quality:
        raise ArtifactSurfaceMeasurementError(
            "surface measurement quality is not derived from its anchors and fit"
        )
    if (
        receipt["schema_version"] != SURFACE_MEASUREMENT_SCHEMA_VERSION
        or receipt["anchor_basis"] != SURFACE_ANCHOR_BASIS
        or receipt["barycentric_denominator"] != BARYCENTRIC_DENOMINATOR
        or receipt["coordinate_space"] != SURFACE_MEASUREMENT_COORDINATE_SPACE
        or receipt["pick_method"] != SURFACE_PICK_METHOD
        or receipt["rounding_mode"] != SURFACE_MEASUREMENT_ROUNDING
    ):
        raise ArtifactSurfaceMeasurementError("receipt constants are invalid")
    return {
        "anchor_basis": SURFACE_ANCHOR_BASIS,
        "anchors": anchors,
        "barycentric_denominator": BARYCENTRIC_DENOMINATOR,
        "coordinate_grid_um": coordinate_grid_um,
        "coordinate_space": SURFACE_MEASUREMENT_COORDINATE_SPACE,
        "input_face_count": input_face_count,
        "input_vertex_count": input_vertex_count,
        "kind": kind,
        "measurement": measurement,
        "pick_method": SURFACE_PICK_METHOD,
        "quality": quality,
        "rounding_mode": SURFACE_MEASUREMENT_ROUNDING,
        "schema_version": SURFACE_MEASUREMENT_SCHEMA_VERSION,
    }


def _qc_from_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    measurement = receipt["measurement"]
    quality = receipt["quality"]
    assert isinstance(measurement, Mapping)
    assert isinstance(quality, Mapping)
    kind = str(receipt["kind"])
    return {
        "anchor_count": len(receipt["anchors"]),
        "coordinate_grid_um": int(receipt["coordinate_grid_um"]),
        "diameter_mm_decimal": (
            measurement["diameter_mm_decimal"] if kind == "surface_diameter" else None
        ),
        "distance_mm_decimal": (
            measurement["distance_mm_decimal"] if kind == "surface_distance" else None
        ),
        "kind": kind,
        "maximum_capture_residual_mm_decimal": quality[
            "maximum_capture_residual_mm_decimal"
        ],
        "maximum_pixel_footprint_um": int(quality["maximum_pixel_footprint_um"]),
        "near_edge_anchor_count": int(quality["near_edge_anchor_count"]),
        "quality_status": quality["status"],
        "review_reason_count": len(quality["review_reasons"]),
        "screen_search_anchor_count": int(quality["screen_search_anchor_count"]),
    }


def _validate_qc_against_receipt(
    qc: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    expected = _qc_from_receipt(receipt)
    if set(qc) != set(expected):
        raise ArtifactSurfaceMeasurementError(
            "surface measurement QC fields do not match the closed contract"
        )
    for key, value in expected.items():
        if qc.get(key) != value:
            raise ArtifactSurfaceMeasurementError(
                f"surface measurement QC field {key!r} does not match its receipt"
            )


def _validate_recipe_against_receipt(
    recipe: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    anchors = receipt["anchors"]
    recipe_anchors = recipe["anchors"]
    assert isinstance(anchors, list)
    assert isinstance(recipe_anchors, list)
    if len(anchors) != len(recipe_anchors):
        raise ArtifactSurfaceMeasurementError("recipe and receipt anchor counts differ")
    common_fields = (
        "barycentric_numerators",
        "capture_point_grid",
        "depth_match_tolerance_um",
        "depth_search_offset_px",
        "face_index",
        "face_vertex_indices",
        "pixel_footprint_um",
    )
    for index, (recipe_anchor, receipt_anchor) in enumerate(
        zip(recipe_anchors, anchors, strict=True)
    ):
        assert isinstance(recipe_anchor, Mapping)
        assert isinstance(receipt_anchor, Mapping)
        for field_name in common_fields:
            if recipe_anchor[field_name] != receipt_anchor[field_name]:
                raise ArtifactSurfaceMeasurementError(
                    f"recipe and receipt anchor {index} field {field_name!r} differ"
                )
    quality = receipt["quality"]
    assert isinstance(quality, Mapping)
    if (
        quality["edge_review_threshold_ppb"]
        != recipe["edge_review_threshold_ppb"]
        or quality["fit_review_threshold_um"]
        != recipe["fit_review_threshold_um"]
    ):
        raise ArtifactSurfaceMeasurementError(
            "recipe and receipt quality thresholds differ"
        )
    point_numerators = [
        np.asarray(
            [int(value) for value in anchor["resolved_point_numerator_grid_bary"]],
            dtype=object,
        )
        for anchor in anchors
    ]
    fit_evidence: Mapping[str, float] | None = None
    if recipe["kind"] == "surface_distance":
        expected_measurement = _distance_measurement(
            point_numerators,
            grid_um=int(recipe["coordinate_grid_um"]),
        )
    else:
        expected_measurement, fit_evidence = _diameter_measurement(
            point_numerators,
            grid_um=int(recipe["coordinate_grid_um"]),
            maximum_fit_condition=int(recipe["maximum_fit_condition"]),
        )
    if expected_measurement != receipt["measurement"]:
        raise ArtifactSurfaceMeasurementError(
            "surface measurement result does not match its resolved anchors"
        )
    expected_quality = _quality_receipt(
        recipe,
        anchors,
        fit_evidence=fit_evidence,
    )
    if expected_quality != quality:
        raise ArtifactSurfaceMeasurementError(
            "surface measurement quality does not match its anchors and fit"
        )


@dataclass(frozen=True, slots=True)
class ArtifactSurfaceMeasurementComputation:
    context: OperationContext
    projection_snapshot: ArtifactProjectionSnapshot
    receipt: Mapping[str, Any]
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.context, OperationContext):
            raise ArtifactSurfaceMeasurementError("context must be OperationContext")
        if not isinstance(self.projection_snapshot, ArtifactProjectionSnapshot):
            raise ArtifactSurfaceMeasurementError(
                "projection_snapshot must be ArtifactProjectionSnapshot"
            )
        recipe = validate_surface_measurement_recipe(self.recipe)
        receipt = validate_surface_measurement_receipt(self.receipt)
        if (
            recipe["kind"] != receipt["kind"]
            or recipe["coordinate_grid_um"] != receipt["coordinate_grid_um"]
            or recipe["source_vertex_count"] != receipt["input_vertex_count"]
            or recipe["source_face_count"] != receipt["input_face_count"]
        ):
            raise ArtifactSurfaceMeasurementError("surface recipe and receipt differ")
        _validate_recipe_against_receipt(recipe, receipt)
        if canonical_recipe_hash(recipe) != self.context.recipe_hash:
            raise ArtifactSurfaceMeasurementError(
                "surface measurement recipe does not match its OperationContext"
            )
        if self.context.selection_hash != surface_measurement_selection_hash(recipe):
            raise ArtifactSurfaceMeasurementError(
                "surface measurement selection hash does not match its anchors"
            )
        snapshot = self.projection_snapshot
        if (
            tuple(self.context.source_asset_ids) != (snapshot.source_asset_id,)
            or self.context.geometry_revision_id != snapshot.geometry_revision_id
            or self.context.source_metadata_revision_id != snapshot.source_metadata_revision_id
            or self.context.align_revision_id != snapshot.align_revision_id
        ):
            raise ArtifactSurfaceMeasurementError(
                "projection snapshot does not match the surface measurement context"
            )
        qc = _frozen_mapping(self.qc, name="surface_measurement.qc")
        _validate_qc_against_receipt(qc, receipt)
        object.__setattr__(self, "recipe", _frozen_mapping(recipe, name="recipe"))
        object.__setattr__(self, "receipt", _frozen_mapping(receipt, name="receipt"))
        object.__setattr__(self, "qc", qc)

    @property
    def kind(self) -> str:
        return str(self.receipt["kind"])

    @property
    def record_type(self) -> str:
        return (
            SURFACE_DISTANCE_RECORD_TYPE
            if self.kind == "surface_distance"
            else SURFACE_DIAMETER_RECORD_TYPE
        )

    @property
    def geometry_ref(self) -> str:
        return SURFACE_MEASUREMENT_REF_PREFIX + canonical_json_sha256(self.receipt)

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


def surface_measurement_computation_matches_active_projection(
    session: ArtifactSession,
    computation: ArtifactSurfaceMeasurementComputation,
) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(
        computation, ArtifactSurfaceMeasurementComputation
    ):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    return current.render_key == computation.projection_snapshot.render_key


def commit_artifact_surface_measurement(
    session: ArtifactSession,
    computation: ArtifactSurfaceMeasurementComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    if not surface_measurement_computation_matches_active_projection(session, computation):
        raise ArtifactSurfaceMeasurementError(
            "surface measurement computation is stale for the active projection"
        )
    receipt = computation.receipt_dict()
    qc = computation.qc_dict()
    _validate_qc_against_receipt(qc, receipt)
    receipt_bytes = canonical_json_bytes(receipt)
    if len(receipt_bytes) > MAXIMUM_RECEIPT_BYTES:
        raise ArtifactSurfaceMeasurementError("surface measurement receipt exceeds its limit")
    receipt_sha256 = canonical_json_sha256(receipt)
    extensions = {
        SURFACE_MEASUREMENT_EXTENSION_KEY: {
            "media_type": SURFACE_MEASUREMENT_MEDIA_TYPE,
            "receipt": receipt,
            "receipt_byte_length": len(receipt_bytes),
            "receipt_sha256": receipt_sha256,
            "schema_version": SURFACE_MEASUREMENT_SCHEMA_VERSION,
        }
    }
    try:
        document = session.document.append_record_from_context(
            context=computation.context,
            id=record_id,
            type=computation.record_type,
            geometry_ref=SURFACE_MEASUREMENT_REF_PREFIX + receipt_sha256,
            recipe=computation.recipe_dict(),
            qc=qc,
            lifecycle_status=RecordLifecycleStatus.READY,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=depends_on_record_ids,
            extensions=extensions,
        )
    except ArtifactDocumentError as exc:
        raise ArtifactSurfaceMeasurementError(str(exc)) from exc
    return session.with_document(document)


def surface_measurement_receipt_from_record(record: DerivedRecord) -> dict[str, Any]:
    if not isinstance(record, DerivedRecord):
        raise ArtifactSurfaceMeasurementError("record must be a DerivedRecord")
    if record.type not in {SURFACE_DISTANCE_RECORD_TYPE, SURFACE_DIAMETER_RECORD_TYPE}:
        raise ArtifactSurfaceMeasurementError("record is not a surface measurement record")
    descriptor = _exact_mapping(
        record.extensions.get(SURFACE_MEASUREMENT_EXTENSION_KEY),
        {"media_type", "receipt", "receipt_byte_length", "receipt_sha256", "schema_version"},
        name="surface measurement descriptor",
    )
    if (
        descriptor["media_type"] != SURFACE_MEASUREMENT_MEDIA_TYPE
        or descriptor["schema_version"] != SURFACE_MEASUREMENT_SCHEMA_VERSION
    ):
        raise ArtifactSurfaceMeasurementError("surface measurement descriptor is invalid")
    declared_length = _strict_int(
        descriptor["receipt_byte_length"],
        name="surface measurement receipt_byte_length",
        minimum=2,
        maximum=MAXIMUM_RECEIPT_BYTES,
    )
    try:
        unvalidated_receipt_bytes = canonical_json_bytes(descriptor["receipt"])
    except (TypeError, ValueError) as exc:
        raise ArtifactSurfaceMeasurementError(
            "surface measurement receipt is not canonical JSON"
        ) from exc
    if len(unvalidated_receipt_bytes) != declared_length:
        raise ArtifactSurfaceMeasurementError(
            "surface measurement receipt length is invalid"
        )
    receipt = validate_surface_measurement_receipt(descriptor["receipt"])
    expected_type = (
        SURFACE_DISTANCE_RECORD_TYPE
        if receipt["kind"] == "surface_distance"
        else SURFACE_DIAMETER_RECORD_TYPE
    )
    if record.type != expected_type:
        raise ArtifactSurfaceMeasurementError("surface measurement record type is invalid")
    receipt_bytes = canonical_json_bytes(receipt)
    if declared_length != len(receipt_bytes):
        raise ArtifactSurfaceMeasurementError("surface measurement receipt length is invalid")
    receipt_sha256 = canonical_json_sha256(receipt)
    if descriptor["receipt_sha256"] != receipt_sha256:
        raise ArtifactSurfaceMeasurementError("surface measurement receipt hash is invalid")
    if record.geometry_ref != SURFACE_MEASUREMENT_REF_PREFIX + receipt_sha256:
        raise ArtifactSurfaceMeasurementError("surface measurement geometry_ref is invalid")
    recipe = validate_surface_measurement_recipe(record.recipe)
    if (
        recipe["kind"] != receipt["kind"]
        or recipe["coordinate_grid_um"] != receipt["coordinate_grid_um"]
        or recipe["source_vertex_count"] != receipt["input_vertex_count"]
        or recipe["source_face_count"] != receipt["input_face_count"]
    ):
        raise ArtifactSurfaceMeasurementError("record recipe and receipt differ")
    _validate_recipe_against_receipt(recipe, receipt)
    if record.selection_hash != surface_measurement_selection_hash(recipe):
        raise ArtifactSurfaceMeasurementError(
            "surface measurement selection hash is invalid"
        )
    _validate_qc_against_receipt(record.qc, receipt)
    return receipt


def validate_surface_measurement_records(document: ArtifactDocument) -> None:
    if not isinstance(document, ArtifactDocument):
        raise ArtifactSurfaceMeasurementError("document must be an ArtifactDocument")
    for record in document.records:
        if record.type in {SURFACE_DISTANCE_RECORD_TYPE, SURFACE_DIAMETER_RECORD_TYPE}:
            surface_measurement_receipt_from_record(record)


def validate_surface_measurement_records_against_session(
    session: ArtifactSession,
) -> None:
    """Reattach every record to its historical source triangles.

    Structural receipt validation already recomputes the result and QC from
    the stored resolved points.  This source-bound pass therefore only needs
    to prove those points still equal their referenced source triangles under
    the record's historical Metadata+Align matrix.  Quantizing the whole mesh
    once per record would make project open/commit scale with
    ``records × vertices``; resolving at most 64 referenced triangles keeps
    the check strict and bounded.
    """

    if not isinstance(session, ArtifactSession):
        raise ArtifactSurfaceMeasurementError("session must be an ArtifactSession")
    source_vertices = np.asarray(session.source_mesh.vertices, dtype=np.float64)
    source_faces = np.asarray(session.source_mesh.faces, dtype=np.int64)
    for record in session.document.records:
        if record.type not in {SURFACE_DISTANCE_RECORD_TYPE, SURFACE_DIAMETER_RECORD_TYPE}:
            continue
        recipe = validate_surface_measurement_recipe(record.recipe)
        stored_receipt = surface_measurement_receipt_from_record(record)
        if (
            int(recipe["source_vertex_count"]) != int(source_vertices.shape[0])
            or int(recipe["source_face_count"]) != int(source_faces.shape[0])
        ):
            raise ArtifactSurfaceMeasurementError(
                "surface measurement source counts do not match the bound geometry"
            )
        align = session.document.align_revision_index.get(record.align_revision_id)
        if align is None:
            raise ArtifactSurfaceMeasurementError(
                "surface measurement record references a missing Align revision"
            )
        metadata = session.document.source_metadata_revision_index.get(
            align.source_metadata_revision_id
        )
        if metadata is None:
            raise ArtifactSurfaceMeasurementError(
                "surface measurement record references missing metadata"
            )
        geometry = session.document.geometry_revision_index.get(
            metadata.geometry_revision_id
        )
        if geometry is None or geometry.id != record.geometry_revision_id:
            raise ArtifactSurfaceMeasurementError(
                "surface measurement record geometry authority is invalid"
            )
        matrix = align.matrix @ metadata.require_confirmed_matrix()
        grid_um = int(recipe["coordinate_grid_um"])
        recipe_anchors = recipe["anchors"]
        receipt_anchors = stored_receipt["anchors"]
        assert isinstance(recipe_anchors, list)
        assert isinstance(receipt_anchors, list)
        for index, (recipe_anchor, receipt_anchor) in enumerate(
            zip(recipe_anchors, receipt_anchors, strict=True)
        ):
            assert isinstance(recipe_anchor, Mapping)
            assert isinstance(receipt_anchor, Mapping)
            face_index = int(recipe_anchor["face_index"])
            if face_index < 0 or face_index >= source_faces.shape[0]:
                raise ArtifactSurfaceMeasurementError(
                    f"surface measurement anchor {index} face is out of range"
                )
            source_row = [int(value) for value in source_faces[face_index]]
            if source_row != [
                int(value) for value in recipe_anchor["face_vertex_indices"]
            ]:
                raise ArtifactSurfaceMeasurementError(
                    f"surface measurement anchor {index} source face identity changed"
                )
            quantized_triangle = _quantized_source_vertices(
                source_vertices[source_row],
                matrix,
                grid_um=grid_um,
            )
            if len(set(source_row)) != 3:
                raise ArtifactSurfaceMeasurementError(
                    f"surface measurement anchor {index} source face is degenerate"
                )
            if not _quantized_triangle_has_area(quantized_triangle):
                raise ArtifactSurfaceMeasurementError(
                    f"surface measurement anchor {index} face collapses on its grid"
                )
            weights = [
                int(value) for value in recipe_anchor["barycentric_numerators"]
            ]
            expected_numerators = [
                str(
                    sum(
                        weights[corner] * int(quantized_triangle[corner, axis])
                        for corner in range(3)
                    )
                )
                for axis in range(3)
            ]
            if expected_numerators != list(
                receipt_anchor["resolved_point_numerator_grid_bary"]
            ):
                raise ArtifactSurfaceMeasurementError(
                    f"surface measurement record {record.id!r} anchor {index} "
                    "does not match source geometry"
                )


__all__ = [
    "ArtifactSurfaceMeasurementComputation",
    "ArtifactSurfaceMeasurementError",
    "BARYCENTRIC_DENOMINATOR",
    "DEFAULT_COORDINATE_GRID_UM",
    "SURFACE_ANCHOR_BASIS",
    "SURFACE_DIAMETER_FIT_POLICY",
    "SURFACE_DIAMETER_RECORD_TYPE",
    "SURFACE_DISTANCE_MEANING",
    "SURFACE_DISTANCE_RECORD_TYPE",
    "SURFACE_MEASUREMENT_ALGORITHM",
    "SURFACE_MEASUREMENT_ALGORITHM_VERSION",
    "SURFACE_MEASUREMENT_COORDINATE_SPACE",
    "SURFACE_MEASUREMENT_EXTENSION_KEY",
    "SURFACE_MEASUREMENT_MEDIA_TYPE",
    "SURFACE_MEASUREMENT_REF_PREFIX",
    "SURFACE_MEASUREMENT_SCHEMA_VERSION",
    "commit_artifact_surface_measurement",
    "extract_surface_measurement",
    "extract_surface_measurement_from_source",
    "resolve_surface_anchor_from_ray",
    "surface_diameter_recipe",
    "surface_distance_recipe",
    "surface_measurement_computation_matches_active_projection",
    "surface_measurement_selection_hash",
    "surface_measurement_receipt_from_record",
    "validate_surface_measurement_receipt",
    "validate_surface_measurement_recipe",
    "validate_surface_measurement_records",
    "validate_surface_measurement_records_against_session",
]
