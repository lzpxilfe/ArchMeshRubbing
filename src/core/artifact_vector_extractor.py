"""Headless canonical-mm vector extraction for authoritative artifact records.

This module intentionally does not call the legacy viewport, OpenGL, OpenCV,
or ``MeshSlicer``.  Cutlines are computed from a fresh ArtifactSession
materialization and an explicit planar frame.  Ambiguous coplanar faces,
on-plane edges, and branching section graphs fail closed instead of producing
renderer- or normal-direction-dependent measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_cancellation import (
    CancellationProbe,
    poll_cancellation,
    raise_if_cancelled,
)
from .artifact_document import OperationContext, canonical_recipe_hash
from .artifact_scene_adapter import ArtifactProjectionSnapshot
from .artifact_session import ArtifactSession, ArtifactSessionError
from .artifact_vector_record import (
    MAX_VECTOR_POINTS,
    PlanarFrame,
    VECTOR_COORDINATE_SPACE,
    VECTOR_PAYLOAD_SCHEMA_VERSION,
    VectorGeometryPayload,
    VectorPath,
    VectorRecordKind,
)


CUTLINE_ALGORITHM = "archmeshrubbing.triangle_plane_cutline"
CUTLINE_ALGORITHM_VERSION = "1.0.0"
DEFAULT_PLANE_CLASSIFICATION_TOLERANCE_MM = 1e-9
DEFAULT_STITCH_TOLERANCE_MM = 1e-7
MAX_CUTLINE_SEGMENTS = MAX_VECTOR_POINTS


class ArtifactVectorExtractionError(ValueError):
    """A measurement-grade vector result cannot be formed unambiguously."""


def _finite_positive(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ArtifactVectorExtractionError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ArtifactVectorExtractionError(
            f"{field_name} must be a finite number greater than zero"
        )
    return number


def _freeze_json(value: Any, *, field_name: str, depth: int = 0) -> Any:
    if depth > 100:
        raise ArtifactVectorExtractionError(f"{field_name} is nested too deeply")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ArtifactVectorExtractionError(
                f"{field_name} contains a non-finite number"
            )
        return 0.0 if number == 0.0 else number
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise ArtifactVectorExtractionError(
                    f"{field_name} object keys must be strings"
                )
            result[key] = _freeze_json(
                value[key],
                field_name=f"{field_name}.{key}",
                depth=depth + 1,
            )
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(
                item,
                field_name=f"{field_name}[{index}]",
                depth=depth + 1,
            )
            for index, item in enumerate(value)
        )
    raise ArtifactVectorExtractionError(
        f"{field_name} contains unsupported {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _json_mapping_copy(
    value: Mapping[str, Any], *, field_name: str
) -> Mapping[str, Any]:
    try:
        encoded = json.dumps(
            _thaw_json(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, RecursionError, json.JSONDecodeError) as exc:
        raise ArtifactVectorExtractionError(f"{field_name} is not strict JSON") from exc
    frozen = _freeze_json(decoded, field_name=field_name)
    if not isinstance(frozen, Mapping):
        raise ArtifactVectorExtractionError(f"{field_name} must be an object")
    return frozen


@dataclass(frozen=True, slots=True)
class CutlineGeometryResult:
    payload: VectorGeometryPayload
    qc: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, VectorGeometryPayload):
            raise ArtifactVectorExtractionError("payload must be VectorGeometryPayload")
        if VectorRecordKind(self.payload.kind) is not VectorRecordKind.CUTLINE:
            raise ArtifactVectorExtractionError(
                "cutline result payload kind must be cutline"
            )
        if not isinstance(self.qc, Mapping):
            raise ArtifactVectorExtractionError("qc must be an object")
        object.__setattr__(self, "qc", _json_mapping_copy(self.qc, field_name="qc"))

    def qc_dict(self) -> dict[str, Any]:
        result = _thaw_json(self.qc)
        assert isinstance(result, dict)
        return result


@dataclass(frozen=True, slots=True)
class ArtifactVectorComputation:
    """One immutable compute result bound to its captured document context."""

    context: OperationContext
    projection_snapshot: ArtifactProjectionSnapshot
    payload: VectorGeometryPayload
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.context, OperationContext):
            raise ArtifactVectorExtractionError("context must be OperationContext")
        if not isinstance(self.projection_snapshot, ArtifactProjectionSnapshot):
            raise ArtifactVectorExtractionError(
                "projection_snapshot must be ArtifactProjectionSnapshot"
            )
        if not isinstance(self.payload, VectorGeometryPayload):
            raise ArtifactVectorExtractionError("payload must be VectorGeometryPayload")
        if not isinstance(self.recipe, Mapping) or not isinstance(self.qc, Mapping):
            raise ArtifactVectorExtractionError("recipe and qc must be objects")
        recipe = _json_mapping_copy(self.recipe, field_name="recipe")
        qc = _json_mapping_copy(self.qc, field_name="qc")
        if canonical_recipe_hash(recipe) != self.context.recipe_hash:
            raise ArtifactVectorExtractionError(
                "computation recipe does not match captured OperationContext"
            )
        snapshot = self.projection_snapshot
        if (
            snapshot.geometry_revision_id != self.context.geometry_revision_id
            or snapshot.source_metadata_revision_id
            != self.context.source_metadata_revision_id
            or snapshot.align_revision_id != self.context.align_revision_id
            or tuple(self.context.source_asset_ids) != (snapshot.source_asset_id,)
        ):
            raise ArtifactVectorExtractionError(
                "projection snapshot does not match captured OperationContext"
            )
        expected_kind = VectorRecordKind(self.payload.kind).value
        if recipe.get("kind") != expected_kind:
            raise ArtifactVectorExtractionError(
                "computation recipe kind does not match payload kind"
            )
        object.__setattr__(self, "recipe", recipe)
        object.__setattr__(self, "qc", qc)

    def recipe_dict(self) -> dict[str, Any]:
        result = _thaw_json(self.recipe)
        assert isinstance(result, dict)
        return result

    def qc_dict(self) -> dict[str, Any]:
        result = _thaw_json(self.qc)
        assert isinstance(result, dict)
        return result


def cutline_recipe(
    frame: PlanarFrame,
    *,
    classification_tolerance_mm: float = DEFAULT_PLANE_CLASSIFICATION_TOLERANCE_MM,
    stitch_tolerance_mm: float = DEFAULT_STITCH_TOLERANCE_MM,
) -> dict[str, Any]:
    """Build the complete deterministic recipe before context capture."""

    if not isinstance(frame, PlanarFrame):
        raise ArtifactVectorExtractionError("frame must be a PlanarFrame")
    classification = _finite_positive(
        classification_tolerance_mm,
        field_name="classification_tolerance_mm",
    )
    stitch = _finite_positive(
        stitch_tolerance_mm,
        field_name="stitch_tolerance_mm",
    )
    if stitch < classification:
        raise ArtifactVectorExtractionError(
            "stitch_tolerance_mm must be at least classification_tolerance_mm"
        )
    return {
        "algorithm": CUTLINE_ALGORITHM,
        "algorithm_version": CUTLINE_ALGORITHM_VERSION,
        "classification_tolerance_mm": classification,
        "closed_orientation": "counterclockwise_in_frame_uv/v1",
        "collinear_reduction": "within_stitch_tolerance/v1",
        "component_order": "closed_bounds_coordinates_then_open/v1",
        "coordinate_space": VECTOR_COORDINATE_SPACE,
        "coplanar_face_policy": "reject",
        "frame": frame.to_dict(),
        "kind": VectorRecordKind.CUTLINE.value,
        "on_plane_edge_policy": "reject",
        "open_path_orientation": "lexicographically_smallest_endpoint_first/v1",
        "open_paths": "allow",
        "stitch_tolerance_mm": stitch,
    }


def _validated_mesh_arrays(
    vertices: object,
    faces: object,
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    raise_if_cancelled(cancellation_probe)
    try:
        raw_vertices = np.asarray(vertices)
        raise_if_cancelled(cancellation_probe)
        raw_faces = np.asarray(faces)
        raise_if_cancelled(cancellation_probe)
    except (TypeError, ValueError) as exc:
        raise ArtifactVectorExtractionError(
            "vertices and faces must be numeric arrays"
        ) from exc
    if raw_vertices.dtype.kind not in {"f", "i", "u"}:
        raise ArtifactVectorExtractionError("vertices must contain only real numbers")
    if raw_faces.dtype.kind not in {"i", "u"}:
        raise ArtifactVectorExtractionError("faces must contain only integer indices")
    try:
        vertex_array = np.asarray(raw_vertices, dtype=np.float64)
        raise_if_cancelled(cancellation_probe)
        face_array = np.asarray(raw_faces, dtype=np.int64)
        raise_if_cancelled(cancellation_probe)
    except (TypeError, ValueError) as exc:
        raise ArtifactVectorExtractionError(
            "vertices and faces must be numeric arrays"
        ) from exc
    if (
        vertex_array.ndim != 2
        or vertex_array.shape[1] != 3
        or vertex_array.shape[0] == 0
    ):
        raise ArtifactVectorExtractionError(
            "vertices must be a finite non-empty Nx3 array"
        )
    vertices_are_finite = bool(np.isfinite(vertex_array).all())
    raise_if_cancelled(cancellation_probe)
    if not vertices_are_finite:
        raise ArtifactVectorExtractionError(
            "vertices must be a finite non-empty Nx3 array"
        )
    if face_array.ndim != 2 or face_array.shape[1] != 3 or face_array.shape[0] == 0:
        raise ArtifactVectorExtractionError("faces must be a non-empty Mx3 array")
    has_negative_index = bool(np.any(face_array < 0))
    raise_if_cancelled(cancellation_probe)
    has_high_index = bool(np.any(face_array >= vertex_array.shape[0]))
    raise_if_cancelled(cancellation_probe)
    if has_negative_index or has_high_index:
        raise ArtifactVectorExtractionError(
            "faces contain an out-of-range vertex index"
        )
    duplicate_index = face_array[:, 0] == face_array[:, 1]
    raise_if_cancelled(cancellation_probe)
    duplicate_index |= face_array[:, 1] == face_array[:, 2]
    raise_if_cancelled(cancellation_probe)
    duplicate_index |= face_array[:, 2] == face_array[:, 0]
    raise_if_cancelled(cancellation_probe)
    triangles = vertex_array[face_array]
    raise_if_cancelled(cancellation_probe)
    first_edge = triangles[:, 1] - triangles[:, 0]
    raise_if_cancelled(cancellation_probe)
    second_edge = triangles[:, 2] - triangles[:, 0]
    raise_if_cancelled(cancellation_probe)
    twice_area = np.cross(first_edge, second_edge)
    del first_edge, second_edge
    raise_if_cancelled(cancellation_probe)
    area_squared = np.einsum("ij,ij->i", twice_area, twice_area)
    raise_if_cancelled(cancellation_probe)
    has_duplicate_index = bool(np.any(duplicate_index))
    raise_if_cancelled(cancellation_probe)
    has_zero_area = bool(np.any(area_squared <= 0.0))
    raise_if_cancelled(cancellation_probe)
    if has_duplicate_index or has_zero_area:
        raise ArtifactVectorExtractionError("faces contain a degenerate triangle")
    return vertex_array, face_array


def _segment_for_triangle(
    triangle: np.ndarray,
    signed: np.ndarray,
    on_plane: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    on_indices = np.flatnonzero(on_plane)
    if on_indices.size == 1:
        on_index = int(on_indices[0])
        others = [index for index in range(3) if index != on_index]
        first, second = others
        if signed[first] * signed[second] >= 0.0:
            return None
        denominator = float(signed[first] - signed[second])
        interpolation = float(signed[first] / denominator)
        crossing = triangle[first] + interpolation * (
            triangle[second] - triangle[first]
        )
        return triangle[on_index].copy(), crossing

    crossings: list[np.ndarray] = []
    for first, second in ((0, 1), (1, 2), (2, 0)):
        first_distance = float(signed[first])
        second_distance = float(signed[second])
        if first_distance * second_distance >= 0.0:
            continue
        denominator = first_distance - second_distance
        interpolation = first_distance / denominator
        crossings.append(
            triangle[first] + interpolation * (triangle[second] - triangle[first])
        )
    if len(crossings) != 2:
        return None
    return crossings[0], crossings[1]


def _cluster_segment_endpoints(
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    tolerance: float,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[int, int]], float, int]:
    raise_if_cancelled(cancellation_probe)
    raw_points: list[tuple[float, float, int]] = []
    for segment_index, (first, second) in enumerate(segments):
        poll_cancellation(cancellation_probe, segment_index)
        raw_points.append((float(first[0]), float(first[1]), segment_index * 2))
        raw_points.append((float(second[0]), float(second[1]), segment_index * 2 + 1))
    ordered = sorted(raw_points, key=lambda item: (item[0], item[1], item[2]))
    raise_if_cancelled(cancellation_probe)
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            return
        if root_first < root_second:
            parent[root_second] = root_first
        else:
            parent[root_first] = root_second

    grid: dict[tuple[int, int], list[int]] = {}
    tolerance_squared = tolerance * tolerance
    neighbor_checks = 0
    for ordered_index, (x, y, _original_index) in enumerate(ordered):
        poll_cancellation(cancellation_probe, ordered_index)
        cell = (math.floor(x / tolerance), math.floor(y / tolerance))
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                for neighbor in grid.get((cell[0] + delta_x, cell[1] + delta_y), ()):
                    poll_cancellation(cancellation_probe, neighbor_checks)
                    neighbor_checks += 1
                    other_x, other_y, _ = ordered[neighbor]
                    if (x - other_x) ** 2 + (y - other_y) ** 2 <= tolerance_squared:
                        union(ordered_index, neighbor)
        grid.setdefault(cell, []).append(ordered_index)

    members: dict[int, list[int]] = {}
    for index in range(len(ordered)):
        poll_cancellation(cancellation_probe, index)
        members.setdefault(find(index), []).append(index)
    centers_by_root: dict[int, tuple[float, float]] = {}
    max_snap = 0.0
    member_checks = 0
    for root_index, (root, indices) in enumerate(members.items()):
        poll_cancellation(cancellation_probe, root_index)
        minimum_x = min(ordered[index][0] for index in indices)
        maximum_x = max(ordered[index][0] for index in indices)
        minimum_y = min(ordered[index][1] for index in indices)
        maximum_y = max(ordered[index][1] for index in indices)
        if math.hypot(maximum_x - minimum_x, maximum_y - minimum_y) > tolerance:
            raise ArtifactVectorExtractionError(
                "endpoint stitch cluster exceeds the declared tolerance"
            )
        x = math.fsum(ordered[index][0] for index in indices) / len(indices)
        y = math.fsum(ordered[index][1] for index in indices) / len(indices)
        center = (0.0 if x == 0.0 else x, 0.0 if y == 0.0 else y)
        centers_by_root[root] = center
        for index in indices:
            poll_cancellation(cancellation_probe, member_checks)
            member_checks += 1
            max_snap = max(
                max_snap,
                math.hypot(
                    ordered[index][0] - center[0], ordered[index][1] - center[1]
                ),
            )
    sorted_roots = sorted(centers_by_root, key=lambda root: centers_by_root[root])
    raise_if_cancelled(cancellation_probe)
    cluster_id = {root: index for index, root in enumerate(sorted_roots)}
    centers = [centers_by_root[root] for root in sorted_roots]
    endpoint_cluster: dict[int, int] = {}
    for ordered_index, (_x, _y, original_index) in enumerate(ordered):
        poll_cancellation(cancellation_probe, ordered_index)
        endpoint_cluster[original_index] = cluster_id[find(ordered_index)]

    edges: list[tuple[int, int]] = []
    collapsed = 0
    for segment_index in range(len(segments)):
        poll_cancellation(cancellation_probe, segment_index)
        first = endpoint_cluster[segment_index * 2]
        second = endpoint_cluster[segment_index * 2 + 1]
        if first == second:
            collapsed += 1
            continue
        edges.append((min(first, second), max(first, second)))
    return centers, edges, max_snap, collapsed


def _signed_area(
    points: Sequence[tuple[float, float]],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> float:
    terms: list[float] = []
    for index in range(len(points)):
        poll_cancellation(cancellation_probe, index)
        terms.append(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
        )
    return 0.5 * math.fsum(terms)


def _collinear(
    previous: tuple[float, float],
    current: tuple[float, float],
    following: tuple[float, float],
    *,
    tolerance: float,
) -> bool:
    dx = following[0] - previous[0]
    dy = following[1] - previous[1]
    baseline = math.hypot(dx, dy)
    if baseline <= tolerance:
        return False
    distance = (
        abs(dx * (previous[1] - current[1]) - (previous[0] - current[0]) * dy)
        / baseline
    )
    if distance > tolerance:
        return False
    dot = (current[0] - previous[0]) * (current[0] - following[0]) + (
        current[1] - previous[1]
    ) * (current[1] - following[1])
    return dot <= tolerance * tolerance


def _remove_collinear_points(
    points: list[tuple[float, float]],
    *,
    closed: bool,
    tolerance: float,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[list[tuple[float, float]], int]:
    raise_if_cancelled(cancellation_probe)
    result = list(points)
    removed = 0
    pass_index = 0
    while True:
        poll_cancellation(cancellation_probe, pass_index)
        pass_index += 1
        if len(result) <= (3 if closed else 2):
            break
        removable: set[int] = set()
        indices = range(len(result)) if closed else range(1, len(result) - 1)
        for check_index, index in enumerate(indices):
            poll_cancellation(cancellation_probe, check_index)
            previous = result[(index - 1) % len(result)]
            following = result[(index + 1) % len(result)]
            if _collinear(
                previous,
                result[index],
                following,
                tolerance=tolerance,
            ):
                removable.add(index)
        if not removable:
            break
        candidate = [
            point for index, point in enumerate(result) if index not in removable
        ]
        raise_if_cancelled(cancellation_probe)
        minimum = 3 if closed else 2
        if len(candidate) < minimum:
            break
        removed += len(result) - len(candidate)
        result = candidate
    return result, removed


def _paths_from_edges(
    centers: Sequence[tuple[float, float]],
    edges: Sequence[tuple[int, int]],
    *,
    tolerance: float,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[list[VectorPath], int, int]:
    raise_if_cancelled(cancellation_probe)
    unique_edge_set: set[tuple[int, int]] = set()
    for edge_index, edge in enumerate(edges):
        poll_cancellation(cancellation_probe, edge_index)
        unique_edge_set.add(edge)
    unique_edges = sorted(unique_edge_set)
    duplicate_count = len(edges) - len(unique_edges)
    adjacency: dict[int, set[int]] = {}
    for edge_index, (first, second) in enumerate(unique_edges):
        poll_cancellation(cancellation_probe, edge_index)
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    branching: list[int] = []
    for node_index, (node, neighbors) in enumerate(adjacency.items()):
        poll_cancellation(cancellation_probe, node_index)
        if len(neighbors) > 2:
            branching.append(node)
    branching.sort()
    if branching:
        raise ArtifactVectorExtractionError(
            f"section graph has {len(branching)} non-manifold branching junctions"
        )

    remaining_nodes = set(adjacency)
    raw_paths: list[tuple[bool, list[tuple[float, float]]]] = []
    collinear_removed = 0
    component_index = 0
    while remaining_nodes:
        poll_cancellation(cancellation_probe, component_index)
        component_index += 1
        seed = min(remaining_nodes, key=lambda node: centers[node])
        component: set[int] = set()
        stack = [seed]
        stack_iteration = 0
        while stack:
            poll_cancellation(cancellation_probe, stack_iteration)
            stack_iteration += 1
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        remaining_nodes -= component
        endpoints = sorted(
            (node for node in component if len(adjacency[node]) == 1),
            key=lambda node: centers[node],
        )
        if len(endpoints) not in {0, 2}:
            raise ArtifactVectorExtractionError(
                "section component is neither a simple open path nor a closed loop"
            )
        closed = len(endpoints) == 0
        start = (
            min(component, key=lambda node: centers[node]) if closed else endpoints[0]
        )
        ordered_nodes = [start]
        previous: int | None = None
        current = start
        traversal_index = 0
        while True:
            poll_cancellation(cancellation_probe, traversal_index)
            traversal_index += 1
            candidates = sorted(
                (neighbor for neighbor in adjacency[current] if neighbor != previous),
                key=lambda node: centers[node],
            )
            if not candidates:
                break
            following = candidates[0]
            if closed and following == start:
                break
            ordered_nodes.append(following)
            previous, current = current, following
            if not closed and len(adjacency[current]) == 1:
                break
            if len(ordered_nodes) > len(component):
                raise ArtifactVectorExtractionError(
                    "section graph traversal did not terminate"
                )
        if len(ordered_nodes) != len(component):
            raise ArtifactVectorExtractionError(
                "section graph traversal omitted an edge"
            )
        points = [centers[node] for node in ordered_nodes]
        points, removed = _remove_collinear_points(
            points,
            closed=closed,
            tolerance=tolerance,
            cancellation_probe=cancellation_probe,
        )
        collinear_removed += removed
        if closed:
            area = _signed_area(points, cancellation_probe=cancellation_probe)
            if abs(area) <= tolerance * tolerance:
                raise ArtifactVectorExtractionError(
                    "closed section path has zero planar area"
                )
            if area < 0.0:
                points = [points[0], *reversed(points[1:])]
        raw_paths.append((closed, points))

    def path_key(item: tuple[bool, list[tuple[float, float]]]) -> tuple[Any, ...]:
        closed, points = item
        minimum_x = min(point[0] for point in points)
        minimum_y = min(point[1] for point in points)
        maximum_x = max(point[0] for point in points)
        maximum_y = max(point[1] for point in points)
        flattened_coordinates: list[float] = []
        for point_index, point in enumerate(points):
            poll_cancellation(cancellation_probe, point_index)
            flattened_coordinates.extend(point)
        flattened = tuple(flattened_coordinates)
        return (
            0 if closed else 1,
            minimum_x,
            minimum_y,
            maximum_x,
            maximum_y,
            flattened,
        )

    raw_paths.sort(key=path_key)
    raise_if_cancelled(cancellation_probe)
    paths: list[VectorPath] = []
    for index, (closed, points) in enumerate(raw_paths):
        poll_cancellation(cancellation_probe, index)
        paths.append(
            VectorPath(
                id=f"cutline:path:{index:04d}",
                role="section",
                closed=closed,
                points_mm=tuple(points),
            )
        )
    return paths, duplicate_count, collinear_removed


def extract_cutline_geometry(
    vertices_world_mm: object,
    faces: object,
    frame: PlanarFrame,
    *,
    classification_tolerance_mm: float = DEFAULT_PLANE_CLASSIFICATION_TOLERANCE_MM,
    stitch_tolerance_mm: float = DEFAULT_STITCH_TOLERANCE_MM,
    cancellation_probe: CancellationProbe | None = None,
) -> CutlineGeometryResult:
    """Intersect canonical-mm triangles with an explicit plane deterministically."""

    raise_if_cancelled(cancellation_probe)
    if not isinstance(frame, PlanarFrame):
        raise ArtifactVectorExtractionError("frame must be a PlanarFrame")
    classification = _finite_positive(
        classification_tolerance_mm,
        field_name="classification_tolerance_mm",
    )
    stitch = _finite_positive(stitch_tolerance_mm, field_name="stitch_tolerance_mm")
    if stitch < classification:
        raise ArtifactVectorExtractionError(
            "stitch_tolerance_mm must be at least classification_tolerance_mm"
        )
    vertices, face_array = _validated_mesh_arrays(
        vertices_world_mm,
        faces,
        cancellation_probe=cancellation_probe,
    )
    raise_if_cancelled(cancellation_probe)
    origin = np.asarray(frame.origin_world_mm, dtype=np.float64)
    normal = np.asarray(frame.normal_world, dtype=np.float64)
    u_axis = np.asarray(frame.u_axis_world, dtype=np.float64)
    v_axis = np.asarray(frame.v_axis_world, dtype=np.float64)
    vertex_signed = (vertices - origin) @ normal
    raise_if_cancelled(cancellation_probe)
    face_signed = vertex_signed[face_array]
    raise_if_cancelled(cancellation_probe)
    absolute_face_signed = np.abs(face_signed)
    raise_if_cancelled(cancellation_probe)
    on_plane = absolute_face_signed <= classification
    del absolute_face_signed
    raise_if_cancelled(cancellation_probe)
    on_counts = np.count_nonzero(on_plane, axis=1)
    raise_if_cancelled(cancellation_probe)
    coplanar_count = int(np.count_nonzero(on_counts == 3))
    raise_if_cancelled(cancellation_probe)
    if coplanar_count:
        raise ArtifactVectorExtractionError(
            f"cut plane contains {coplanar_count} coplanar faces; offset the plane"
        )
    on_edge_count = int(np.count_nonzero(on_counts == 2))
    raise_if_cancelled(cancellation_probe)
    if on_edge_count:
        raise ArtifactVectorExtractionError(
            f"cut plane contains {on_edge_count} on-plane triangle edges; offset the plane"
        )
    negative = np.any(face_signed < -classification, axis=1)
    raise_if_cancelled(cancellation_probe)
    positive = np.any(face_signed > classification, axis=1)
    raise_if_cancelled(cancellation_probe)
    tangent_sign = np.logical_xor(negative, positive)
    raise_if_cancelled(cancellation_probe)
    point_tangent_count = int(
        np.count_nonzero((on_counts == 1) & tangent_sign)
    )
    raise_if_cancelled(cancellation_probe)
    candidate_mask = (negative & positive) | ((on_counts == 1) & (negative | positive))
    raise_if_cancelled(cancellation_probe)
    candidate_indices = np.flatnonzero(candidate_mask)
    raise_if_cancelled(cancellation_probe)
    segments_uv: list[tuple[np.ndarray, np.ndarray]] = []
    max_plane_residual = 0.0
    intersected_faces = 0
    for candidate_index, face_index in enumerate(candidate_indices):
        poll_cancellation(cancellation_probe, candidate_index)
        triangle = vertices[face_array[face_index]]
        segment = _segment_for_triangle(
            triangle,
            face_signed[face_index],
            on_plane[face_index],
        )
        if segment is None:
            continue
        first_world, second_world = segment
        residual = max(
            abs(float(np.dot(first_world - origin, normal))),
            abs(float(np.dot(second_world - origin, normal))),
        )
        max_plane_residual = max(max_plane_residual, residual)
        first_relative = first_world - origin
        second_relative = second_world - origin
        first_uv = np.array(
            [np.dot(first_relative, u_axis), np.dot(first_relative, v_axis)],
            dtype=np.float64,
        )
        second_uv = np.array(
            [np.dot(second_relative, u_axis), np.dot(second_relative, v_axis)],
            dtype=np.float64,
        )
        if float(np.linalg.norm(first_uv - second_uv)) <= classification:
            continue
        segments_uv.append((first_uv, second_uv))
        intersected_faces += 1
        if len(segments_uv) > MAX_CUTLINE_SEGMENTS:
            raise ArtifactVectorExtractionError(
                f"cutline exceeds the {MAX_CUTLINE_SEGMENTS}-segment safety limit"
            )
    if not segments_uv:
        raise ArtifactVectorExtractionError("cut plane does not form a line section")
    if max_plane_residual > classification:
        raise ArtifactVectorExtractionError(
            "computed section exceeds the declared plane classification tolerance"
        )

    centers, edges, max_snap, collapsed = _cluster_segment_endpoints(
        segments_uv,
        tolerance=stitch,
        cancellation_probe=cancellation_probe,
    )
    if collapsed:
        raise ArtifactVectorExtractionError(
            f"{collapsed} section segments collapse at the stitch tolerance"
        )
    paths, duplicate_count, collinear_removed = _paths_from_edges(
        centers,
        edges,
        tolerance=stitch,
        cancellation_probe=cancellation_probe,
    )
    if duplicate_count:
        raise ArtifactVectorExtractionError(
            f"section contains {duplicate_count} coincident segments"
        )
    payload = VectorGeometryPayload(
        schema_version=VECTOR_PAYLOAD_SCHEMA_VERSION,
        kind=VectorRecordKind.CUTLINE,
        coordinate_space=VECTOR_COORDINATE_SPACE,
        frame=frame,
        paths=tuple(paths),
    )
    raise_if_cancelled(cancellation_probe)
    qc = {
        "candidate_face_count": int(candidate_indices.size),
        "classification_tolerance_mm": classification,
        "collinear_point_removal_count": collinear_removed,
        "coplanar_face_count": 0,
        "duplicate_segment_count": duplicate_count,
        "input_face_count": int(face_array.shape[0]),
        "input_vertex_count": int(vertices.shape[0]),
        "intersected_face_count": intersected_faces,
        "max_endpoint_snap_mm": max_snap,
        "max_plane_residual_mm": max_plane_residual,
        "non_manifold_junction_count": 0,
        "on_plane_edge_face_count": 0,
        "point_tangent_count": point_tangent_count,
        "raw_segment_count": len(segments_uv),
        "stitch_tolerance_mm": stitch,
        "unique_segment_count": len(set(edges)),
    }
    raise_if_cancelled(cancellation_probe)
    result = CutlineGeometryResult(payload=payload, qc=qc)
    raise_if_cancelled(cancellation_probe)
    return result


def compute_artifact_cutline(
    session: ArtifactSession,
    frame: PlanarFrame,
    *,
    classification_tolerance_mm: float = DEFAULT_PLANE_CLASSIFICATION_TOLERANCE_MM,
    stitch_tolerance_mm: float = DEFAULT_STITCH_TOLERANCE_MM,
    selection_hash: str | None = None,
    cancellation_probe: CancellationProbe | None = None,
) -> ArtifactVectorComputation:
    """Capture context and compute a canonical cutline from a fresh projection."""

    raise_if_cancelled(cancellation_probe)
    if not isinstance(session, ArtifactSession):
        raise ArtifactVectorExtractionError("session must be an ArtifactSession")
    recipe = cutline_recipe(
        frame,
        classification_tolerance_mm=classification_tolerance_mm,
        stitch_tolerance_mm=stitch_tolerance_mm,
    )
    try:
        context = session.capture_vector_operation(
            recipe=recipe,
            selection_hash=selection_hash,
        )
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactVectorExtractionError(str(exc)) from exc
    raise_if_cancelled(cancellation_probe)
    geometry = extract_cutline_geometry(
        projection.mesh.vertices,
        projection.mesh.faces,
        frame,
        classification_tolerance_mm=classification_tolerance_mm,
        stitch_tolerance_mm=stitch_tolerance_mm,
        cancellation_probe=cancellation_probe,
    )
    raise_if_cancelled(cancellation_probe)
    computation = ArtifactVectorComputation(
        context=context,
        projection_snapshot=projection.snapshot,
        payload=geometry.payload,
        recipe=recipe,
        qc=geometry.qc,
    )
    raise_if_cancelled(cancellation_probe)
    return computation


def computation_matches_active_projection(
    session: ArtifactSession,
    computation: ArtifactVectorComputation,
) -> bool:
    """Return whether a result may be projected into the current GUI scene.

    Document SHA is intentionally ignored so committing another record does not
    invalidate unchanged geometry.  Every identity, revision, and matrix field
    that affects coordinates remains part of the comparison.
    """

    if not isinstance(session, ArtifactSession) or not isinstance(
        computation, ArtifactVectorComputation
    ):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    expected = computation.projection_snapshot
    return (
        current.document_id == expected.document_id
        and current.document_schema_version == expected.document_schema_version
        and current.source_asset_id == expected.source_asset_id
        and current.geometry_revision_id == expected.geometry_revision_id
        and current.source_metadata_revision_id == expected.source_metadata_revision_id
        and current.align_revision_id == expected.align_revision_id
        and current.geometry_sha256 == expected.geometry_sha256
        and current.geometry_hash_scope == expected.geometry_hash_scope
        and current.matrix4x4 == expected.matrix4x4
    )


def require_current_computation(
    session: ArtifactSession,
    computation: ArtifactVectorComputation,
) -> None:
    if not computation_matches_active_projection(session, computation):
        raise ArtifactVectorExtractionError(
            "vector computation is stale for the active scene projection"
        )


def commit_vector_computation(
    session: ArtifactSession,
    computation: ArtifactVectorComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    """Append a computation at its captured context, even if now historical."""

    if not isinstance(session, ArtifactSession):
        raise ArtifactVectorExtractionError("session must be an ArtifactSession")
    if not isinstance(computation, ArtifactVectorComputation):
        raise ArtifactVectorExtractionError(
            "computation must be an ArtifactVectorComputation"
        )
    try:
        return session.commit_vector_record(
            context=computation.context,
            payload=computation.payload,
            recipe=computation.recipe_dict(),
            record_id=record_id,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=tuple(depends_on_record_ids),
            qc=computation.qc_dict(),
        )
    except ArtifactSessionError as exc:
        raise ArtifactVectorExtractionError(str(exc)) from exc


__all__ = [
    "ArtifactVectorComputation",
    "ArtifactVectorExtractionError",
    "CUTLINE_ALGORITHM",
    "CUTLINE_ALGORITHM_VERSION",
    "CutlineGeometryResult",
    "DEFAULT_PLANE_CLASSIFICATION_TOLERANCE_MM",
    "DEFAULT_STITCH_TOLERANCE_MM",
    "MAX_CUTLINE_SEGMENTS",
    "commit_vector_computation",
    "computation_matches_active_projection",
    "compute_artifact_cutline",
    "cutline_recipe",
    "extract_cutline_geometry",
    "require_current_computation",
]
