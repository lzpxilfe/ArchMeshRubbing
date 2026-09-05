"""Authoritative six-view silhouette extraction in canonical millimetres.

Every mesh triangle is orthographically projected into one of six explicit,
right-handed planar frames.  Projected polygons are snapped with GEOS' fixed
precision model and unioned without rasterisation, sampling, convex hulls, or
screen-space fallbacks.  The resulting Polygon/MultiPolygon is converted into
deterministically ordered exterior and hole paths for immutable vector records.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import shapely
from shapely import is_valid_reason, set_precision, union_all
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .artifact_cancellation import (
    CancellationProbe,
    poll_cancellation,
    raise_if_cancelled,
)
from .artifact_session import ArtifactSession, ArtifactSessionError
from .artifact_document import canonical_recipe_hash
from .artifact_outline_topology import validate_outline_topology
from .artifact_vector_extractor import (
    ArtifactVectorComputation,
    ArtifactVectorExtractionError,
    _validated_mesh_arrays,
)
from .artifact_vector_record import (
    MAX_VECTOR_POINTS,
    PlanarFrame,
    VECTOR_COORDINATE_SPACE,
    VECTOR_PAYLOAD_SCHEMA_VERSION,
    VectorGeometryPayload,
    VectorPath,
    VectorRecordKind,
)


OUTLINE_ALGORITHM = "archmeshrubbing.projected_triangle_union"
#: The lattice union alone, as every outline before grid closing was computed.
OUTLINE_LEGACY_ALGORITHM_VERSION = "1.0.0"
#: The lattice union followed by a one-cell morphological closing.  Snapping
#: thin silhouette triangles to the grid leaves hairline slivers between them
#: - holes one cell wide, pinches where the outline touches itself - that are
#: the grid's, not the artifact's; the closing removes exactly the features
#: narrower than two cells.  Records at 1.0.0 still recompute as they were.
OUTLINE_CLOSING_ALGORITHM_VERSION = "1.1.0"
#: The closed union, refused when it is more than one piece.  A connected mesh
#: projects to a connected silhouette, so a second exterior component is
#: never the artifact: it is a loose fragment the scanner left in the mesh -
#: the museum's 빗살무늬토기 (신수22891) carries a 24-face crumb beside a
#: 391,432-face body - or a snap that severed the artifact at a grid coarser
#: than the mesh.  1.1.0 drew both into the measured drawing without a word,
#: the crumb as a second outline 0.8 mm2 large.  Nothing else changes: an
#: outline that is one piece has the same bytes at 1.2.0 as at 1.1.0, and
#: records at 1.0.0 and 1.1.0 still recompute as they were written.
OUTLINE_PIECE_GATE_ALGORITHM_VERSION = "1.2.0"
#: The piece-gated union, refused when a hole in it is the grid's.  A grid
#: coarser than the mesh collapses the thin triangles of a surface seen
#: edge-on, and where enough of them go a hole opens in the lattice union
#: that the closing cannot span - 1.2.0 drew it: the museum pot's top view at
#: 1.0 mm has ten such holes, 4 to 7 mm wide, and passed.  The unsnapped
#: union of the same triangles - already computed for the area comparison -
#: has no hole there, because the surface is continuous.  1.3.0 measures how
#: much of each hole the unsnapped union covers: a hole in the artifact is
#: covered only by the snap error at its rim (a 12 mm hole through both
#: walls of a test vessel: 0.4 to 7 %); a hole the grid punched is covered
#: entirely (every one measured: 100 %).  Above the fraction below the
#: outline is refused with the hole's size in millimetres.  Nothing else
#: changes: an outline whose holes are the artifact's has the same bytes at
#: 1.3.0 as at 1.2.0, and every earlier version still recomputes as written.
OUTLINE_ALGORITHM_VERSION = "1.3.0"
OUTLINE_ALGORITHM_VERSIONS = (
    OUTLINE_LEGACY_ALGORITHM_VERSION,
    OUTLINE_CLOSING_ALGORITHM_VERSION,
    OUTLINE_PIECE_GATE_ALGORITHM_VERSION,
    OUTLINE_ALGORITHM_VERSION,
)
OUTLINE_GRID_CLOSING_RADIUS_CELLS = 1.0
#: The largest fraction of a hole's area the unsnapped union may cover before
#: the hole is the grid's.  Measured values sit at the two ends - under 0.07
#: for a hole in the artifact, 1.0 for a hole the grid punched - and the
#: middle is a hole too small to tell apart at this grid, which is refused
#: with the same advice: a finer grid.
OUTLINE_GRID_HOLE_COVER_FRACTION_MAX = 0.5
#: Snap error contracts, per algorithm version.  The union moves a vertex by
#: at most half a cell on each axis; the closing may move a boundary point by
#: its radius, and re-snapping by another half cell.
_SNAP_CONTRACTS: Mapping[str, tuple[str, float]] = {
    OUTLINE_LEGACY_ALGORITHM_VERSION: ("axis<=grid/2;radial<=grid/sqrt(2)", 0.5),
    OUTLINE_CLOSING_ALGORITHM_VERSION: (
        "axis<=1.5*grid;radial<=1.5*grid*sqrt(2)",
        OUTLINE_GRID_CLOSING_RADIUS_CELLS + 0.5,
    ),
    # The piece gate and the hole gate move no vertex, so 1.2.0 and 1.3.0
    # keep 1.1.0's contract.
    OUTLINE_PIECE_GATE_ALGORITHM_VERSION: (
        "axis<=1.5*grid;radial<=1.5*grid*sqrt(2)",
        OUTLINE_GRID_CLOSING_RADIUS_CELLS + 0.5,
    ),
    OUTLINE_ALGORITHM_VERSION: (
        "axis<=1.5*grid;radial<=1.5*grid*sqrt(2)",
        OUTLINE_GRID_CLOSING_RADIUS_CELLS + 0.5,
    ),
}


def _outline_algorithm_version(value: object) -> str:
    if not isinstance(value, str) or value not in OUTLINE_ALGORITHM_VERSIONS:
        raise ArtifactVectorExtractionError(
            "outline algorithm_version must be one of "
            + ", ".join(OUTLINE_ALGORITHM_VERSIONS)
        )
    return value
DEFAULT_OUTLINE_PRECISION_GRID_MM = 0.01
REQUIRED_SHAPELY_VERSION = "2.1.2"
REQUIRED_GEOS_VERSION = "3.13.1"

# Every (Shapely, GEOS) pair this project has computed authoritative outlines
# with.  Append here when the pin moves; never remove an entry.
#
# Computing an outline uses `REQUIRED_*` alone, because a new record must come
# from the one backend the release actually tests.  Verifying a record that
# already exists uses this table instead: the pair it names was reviewed when
# it was current, and a package must not stop verifying merely because the
# project later moved on.  Comparing a stored version against `REQUIRED_*`
# would have failed every outline package the moment the pin changed.
REVIEWED_OUTLINE_BACKENDS: frozenset[tuple[str, str]] = frozenset(
    {
        (REQUIRED_SHAPELY_VERSION, REQUIRED_GEOS_VERSION),
    }
)
MAX_OUTLINE_FACES = 2_000_000
MAX_OUTLINE_VERTICES = 5_000_000
MAX_OUTLINE_INTERMEDIATE_COORDINATES = 1_000_000
MAX_OUTLINE_INTERMEDIATE_POLYGONS = 16_384
OUTLINE_UNION_BATCH_SIZE = 25_000
MAX_GRID_INDEX = 2**48

_EXTERIOR_ID_RE = re.compile(r"^outline:component:(\d{4}):exterior$")
_HOLE_ID_RE = re.compile(r"^outline:component:(\d{4}):hole:(\d{4})$")


class OutlineView(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"
    FRONT = "front"
    BACK = "back"
    RIGHT = "right"
    LEFT = "left"


_VIEW_AXES: dict[
    OutlineView,
    tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
] = {
    OutlineView.TOP: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    OutlineView.BOTTOM: ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
    OutlineView.FRONT: ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    OutlineView.BACK: ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    OutlineView.RIGHT: ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    OutlineView.LEFT: ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)),
}


def _outline_view(value: object) -> OutlineView:
    try:
        return OutlineView(value)
    except (TypeError, ValueError) as exc:
        supported = ", ".join(item.value for item in OutlineView)
        raise ArtifactVectorExtractionError(
            f"outline view must be one of: {supported}"
        ) from exc


def _precision_grid(value: object) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ArtifactVectorExtractionError(
            "precision_grid_mm must be a finite number greater than zero"
        )
    grid = float(value)
    if not math.isfinite(grid) or grid <= 0.0:
        raise ArtifactVectorExtractionError(
            "precision_grid_mm must be a finite number greater than zero"
        )
    return grid


def _require_outline_backend() -> None:
    if (
        shapely.__version__ != REQUIRED_SHAPELY_VERSION
        or shapely.geos_version_string != REQUIRED_GEOS_VERSION
    ):
        raise ArtifactVectorExtractionError(
            "authoritative Outline requires "
            f"Shapely {REQUIRED_SHAPELY_VERSION} with GEOS {REQUIRED_GEOS_VERSION}; "
            f"found Shapely {shapely.__version__} with GEOS "
            f"{shapely.geos_version_string}"
        )


def outline_frame(view: OutlineView | str) -> PlanarFrame:
    """Return the canonical right-handed projection frame for a six-view side."""

    resolved = _outline_view(view)
    u_axis, v_axis, normal = _VIEW_AXES[resolved]
    return PlanarFrame(
        origin_world_mm=(0.0, 0.0, 0.0),
        u_axis_world=u_axis,
        v_axis_world=v_axis,
        normal_world=normal,
    )


def outline_recipe(
    view: OutlineView | str,
    *,
    precision_grid_mm: float,
    algorithm_version: str = OUTLINE_ALGORITHM_VERSION,
) -> dict[str, Any]:
    _require_outline_backend()
    resolved = _outline_view(view)
    grid = _precision_grid(precision_grid_mm)
    version = _outline_algorithm_version(algorithm_version)
    closing = version != OUTLINE_LEGACY_ALGORITHM_VERSION
    return {
        "algorithm": OUTLINE_ALGORITHM,
        "algorithm_version": version,
        "backend": {
            "name": "shapely",
            "geos_version": shapely.geos_version_string,
            "normalized_grid_size": 1.0,
            "operation": (
                "set_precision+union_all+buffer" if closing else "set_precision+union_all"
            ),
            "shapely_version": shapely.__version__,
        },
        "coordinate_space": VECTOR_COORDINATE_SPACE,
        "face_scope": "all_geometry_faces",
        "face_order": "geometry_revision_order/v1",
        "frame": outline_frame(resolved).to_dict(),
        # Present from 1.1.0 on; a 1.0.0 recipe has no closing and no key, so
        # it hashes exactly as it did.
        **(
            {
                "grid_closing": {
                    "join_style": "mitre",
                    "operation": "buffer_out_then_in_then_set_precision/v1",
                    "radius_cells": OUTLINE_GRID_CLOSING_RADIUS_CELLS,
                }
            }
            if closing
            else {}
        ),
        # Present from 1.3.0 on: the hole gate's threshold, so a record says
        # what it was measured against.
        **(
            {
                "grid_hole_gate": {
                    "cover_fraction_max": OUTLINE_GRID_HOLE_COVER_FRACTION_MAX,
                    "operation": "hole_area_covered_by_unsnapped_union/v1",
                }
            }
            if version == OUTLINE_ALGORITHM_VERSION
            else {}
        ),
        "kind": VectorRecordKind.OUTLINE.value,
        "precision_grid_mm": grid,
        "precision_model": (
            "translated_integer_lattice_set_precision_valid_output_"
            "then_balanced_union_all_then_one_cell_closing/v1"
            if closing
            else "translated_integer_lattice_set_precision_valid_output_"
            "then_balanced_union_all/v1"
        ),
        "projection": "orthographic_all_triangles/v1",
        "ring_policy": {
            "exterior_orientation": "counterclockwise_in_frame_uv",
            "hole_orientation": "clockwise_in_frame_uv",
            "collinear_reduction": "exact_integer_grid/v1",
            "component_order": "canonical_exterior_then_holes/v1",
        },
        "sampling": "none",
        "simplification_tolerance_mm": 0.0,
        "union_batch_size": OUTLINE_UNION_BATCH_SIZE,
        "union_merge_order": "balanced_pairwise/v1",
        "union_operation": "polygonal_union_all_fixed_grid/v1",
        "resource_limits": {
            "max_abs_lattice_index": MAX_GRID_INDEX,
            "max_input_faces": MAX_OUTLINE_FACES,
            "max_input_vertices": MAX_OUTLINE_VERTICES,
            "max_intermediate_coordinates": MAX_OUTLINE_INTERMEDIATE_COORDINATES,
            "max_intermediate_polygons": MAX_OUTLINE_INTERMEDIATE_POLYGONS,
        },
        "view": resolved.value,
    }


def validate_outline_record_contract(
    payload: VectorGeometryPayload,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> None:
    """Re-prove the production recipe, grid, topology, and path identity."""

    raise_if_cancelled(cancellation_probe)
    if (
        not isinstance(payload, VectorGeometryPayload)
        or VectorRecordKind(payload.kind) is not VectorRecordKind.OUTLINE
    ):
        raise ArtifactVectorExtractionError(
            "outline record contract requires an outline vector payload"
        )
    if not isinstance(recipe, Mapping):
        raise ArtifactVectorExtractionError("outline record recipe must be an object")
    view = _outline_view(recipe.get("view"))
    grid = _precision_grid(recipe.get("precision_grid_mm"))
    expected_recipe = outline_recipe(
        view,
        precision_grid_mm=grid,
        algorithm_version=_outline_algorithm_version(recipe.get("algorithm_version")),
    )
    if canonical_recipe_hash(recipe) != canonical_recipe_hash(expected_recipe):
        raise ArtifactVectorExtractionError(
            "outline record recipe does not match the production algorithm contract"
        )
    if payload.frame != outline_frame(view):
        raise ArtifactVectorExtractionError(
            "outline payload frame does not match its declared six-view recipe"
        )

    integers_by_id: dict[str, tuple[tuple[int, int], ...]] = {}
    for path_index, path in enumerate(payload.paths):
        poll_cancellation(cancellation_probe, path_index)
        integer_points: list[tuple[int, int]] = []
        for point_index, (x, y) in enumerate(path.points_mm):
            poll_cancellation(cancellation_probe, point_index)
            scaled_x = x / grid
            scaled_y = y / grid
            if max(abs(scaled_x), abs(scaled_y)) > MAX_GRID_INDEX:
                raise ArtifactVectorExtractionError(
                    "outline payload exceeds the fixed-grid integer safety range"
                )
            x_index = int(round(scaled_x))
            y_index = int(round(scaled_y))
            residual = max(abs(x - x_index * grid), abs(y - y_index * grid))
            if residual > max(grid * 1e-8, 1e-12):
                raise ArtifactVectorExtractionError(
                    "outline payload contains an off-grid coordinate"
                )
            integer_points.append((x_index, y_index))
        points = tuple(integer_points)
        for index, current in enumerate(points):
            poll_cancellation(cancellation_probe, index)
            previous = points[index - 1]
            following = points[(index + 1) % len(points)]
            cross = (current[0] - previous[0]) * (following[1] - current[1]) - (
                current[1] - previous[1]
            ) * (following[0] - current[0])
            if cross == 0:
                raise ArtifactVectorExtractionError(
                    "outline payload contains a non-canonical collinear ring point"
                )
        integers_by_id[path.id] = points

    topology = validate_outline_topology(
        payload,
        cancellation_probe=cancellation_probe,
    )
    exterior_paths = sorted(
        (path for path in payload.paths if path.role == "exterior"),
        key=lambda path: integers_by_id[path.id],
    )
    exterior_id_by_component: dict[int, str] = {}
    for component_index, path in enumerate(exterior_paths):
        poll_cancellation(cancellation_probe, component_index)
        match = _EXTERIOR_ID_RE.fullmatch(path.id)
        if match is None or int(match.group(1)) != component_index:
            raise ArtifactVectorExtractionError(
                "outline exterior IDs do not match canonical component order"
            )
        exterior_id_by_component[component_index] = path.id

    owner_by_hole = dict(topology.hole_assignments)
    holes_by_component: dict[int, list[VectorPath]] = {
        index: [] for index in exterior_id_by_component
    }
    for path_index, path in enumerate(payload.paths):
        poll_cancellation(cancellation_probe, path_index)
        if path.role != "hole":
            continue
        match = _HOLE_ID_RE.fullmatch(path.id)
        if match is None:
            raise ArtifactVectorExtractionError(
                "outline hole ID does not use the production component format"
            )
        component_index = int(match.group(1))
        exterior_id = exterior_id_by_component.get(component_index)
        if exterior_id is None or owner_by_hole.get(path.id) != exterior_id:
            raise ArtifactVectorExtractionError(
                "outline hole ID does not match its geometric exterior owner"
            )
        holes_by_component[component_index].append(path)
    for component_index, holes in holes_by_component.items():
        poll_cancellation(cancellation_probe, component_index)
        ordered = sorted(holes, key=lambda path: integers_by_id[path.id])
        for hole_index, path in enumerate(ordered):
            expected_id = (
                f"outline:component:{component_index:04d}:hole:{hole_index:04d}"
            )
            if path.id != expected_id:
                raise ArtifactVectorExtractionError(
                    "outline hole IDs do not match canonical within-component order"
                )
    raise_if_cancelled(cancellation_probe)


@dataclass(frozen=True, slots=True)
class OutlineGeometryResult:
    payload: VectorGeometryPayload
    qc: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, VectorGeometryPayload):
            raise ArtifactVectorExtractionError("payload must be VectorGeometryPayload")
        if VectorRecordKind(self.payload.kind) is not VectorRecordKind.OUTLINE:
            raise ArtifactVectorExtractionError(
                "outline result payload kind must be outline"
            )
        if not isinstance(self.qc, dict):
            raise ArtifactVectorExtractionError("qc must be an object")


def _polygon_count(geometry: BaseGeometry) -> int:
    if geometry.is_empty:
        return 0
    if isinstance(geometry, Polygon):
        return 1
    if isinstance(geometry, MultiPolygon):
        return len(geometry.geoms)
    raise ArtifactVectorExtractionError(
        f"outline union produced non-polygonal geometry: {geometry.geom_type}"
    )


def _polygon_sequence(geometry: BaseGeometry) -> tuple[Polygon, ...]:
    if geometry.is_empty:
        return ()
    if isinstance(geometry, Polygon):
        return (geometry,)
    if isinstance(geometry, MultiPolygon):
        return tuple(geometry.geoms)
    raise ArtifactVectorExtractionError(
        f"outline union produced non-polygonal geometry: {geometry.geom_type}"
    )


def _validate_intermediate_geometry(
    geometry: BaseGeometry,
    *,
    stage: str,
    cancellation_probe: CancellationProbe | None = None,
) -> None:
    raise_if_cancelled(cancellation_probe)
    polygons = _polygon_sequence(geometry)
    if not polygons:
        raise ArtifactVectorExtractionError(f"outline {stage} geometry is empty")
    if len(polygons) > MAX_OUTLINE_INTERMEDIATE_POLYGONS:
        raise ArtifactVectorExtractionError(
            f"outline {stage} exceeds the "
            f"{MAX_OUTLINE_INTERMEDIATE_POLYGONS}-polygon safety limit"
        )
    coordinate_count = 0
    for polygon_index, polygon in enumerate(polygons):
        poll_cancellation(cancellation_probe, polygon_index)
        coordinate_count += len(polygon.exterior.coords)
        for interior_index, interior in enumerate(polygon.interiors):
            poll_cancellation(cancellation_probe, interior_index)
            coordinate_count += len(interior.coords)
    if coordinate_count > MAX_OUTLINE_INTERMEDIATE_COORDINATES:
        raise ArtifactVectorExtractionError(
            f"outline {stage} exceeds the "
            f"{MAX_OUTLINE_INTERMEDIATE_COORDINATES}-coordinate safety limit"
        )
    if not geometry.is_valid:
        raise ArtifactVectorExtractionError(
            f"outline {stage} geometry is invalid: {is_valid_reason(geometry)}"
        )
    raise_if_cancelled(cancellation_probe)


def _batched_union(
    geometries: Sequence[BaseGeometry],
    *,
    grid_size: float | None,
    cancellation_probe: CancellationProbe | None = None,
) -> BaseGeometry:
    raise_if_cancelled(cancellation_probe)
    if not geometries:
        raise ArtifactVectorExtractionError(
            "outline has no polygonal projected triangles"
        )
    level = list(geometries)
    try:
        batches: list[BaseGeometry] = []
        for start in range(0, len(level), OUTLINE_UNION_BATCH_SIZE):
            poll_cancellation(
                cancellation_probe,
                start // OUTLINE_UNION_BATCH_SIZE,
            )
            chunk = level[start : start + OUTLINE_UNION_BATCH_SIZE]
            raise_if_cancelled(cancellation_probe)
            merged = (
                union_all(chunk)
                if grid_size is None
                else union_all(chunk, grid_size=grid_size)
            )
            raise_if_cancelled(cancellation_probe)
            _validate_intermediate_geometry(
                merged,
                stage="batch-union",
                cancellation_probe=cancellation_probe,
            )
            batches.append(merged)
        level = batches
        while len(level) > 1:
            raise_if_cancelled(cancellation_probe)
            next_level: list[BaseGeometry] = []
            for index in range(0, len(level), 2):
                poll_cancellation(cancellation_probe, index // 2)
                pair = level[index : index + 2]
                raise_if_cancelled(cancellation_probe)
                merged = (
                    pair[0]
                    if len(pair) == 1
                    else (
                        union_all(pair)
                        if grid_size is None
                        else union_all(pair, grid_size=grid_size)
                    )
                )
                raise_if_cancelled(cancellation_probe)
                _validate_intermediate_geometry(
                    merged,
                    stage="balanced-union",
                    cancellation_probe=cancellation_probe,
                )
                next_level.append(merged)
            level = next_level
        raise_if_cancelled(cancellation_probe)
        return level[0]
    except (GEOSException, ValueError, TypeError) as exc:
        raise ArtifactVectorExtractionError(
            f"outline polygon union failed: {exc}"
        ) from exc


def _balanced_union(
    geometries: Sequence[BaseGeometry],
    *,
    grid_size: float | None,
    cancellation_probe: CancellationProbe | None = None,
) -> BaseGeometry:
    raise_if_cancelled(cancellation_probe)
    if not geometries:
        raise ArtifactVectorExtractionError("outline has no non-empty union batches")
    level = list(geometries)
    try:
        while len(level) > 1:
            raise_if_cancelled(cancellation_probe)
            next_level: list[BaseGeometry] = []
            for index in range(0, len(level), 2):
                poll_cancellation(cancellation_probe, index // 2)
                pair = level[index : index + 2]
                raise_if_cancelled(cancellation_probe)
                merged = (
                    pair[0]
                    if len(pair) == 1
                    else (
                        union_all(pair)
                        if grid_size is None
                        else union_all(pair, grid_size=grid_size)
                    )
                )
                raise_if_cancelled(cancellation_probe)
                _validate_intermediate_geometry(
                    merged,
                    stage="balanced-union",
                    cancellation_probe=cancellation_probe,
                )
                next_level.append(merged)
            level = next_level
        raise_if_cancelled(cancellation_probe)
        return level[0]
    except (GEOSException, ValueError, TypeError) as exc:
        raise ArtifactVectorExtractionError(
            f"outline balanced polygon union failed: {exc}"
        ) from exc


def _remove_integer_collinear(
    points: Sequence[tuple[int, int]],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[tuple[tuple[int, int], ...], int]:
    raise_if_cancelled(cancellation_probe)
    result = list(points)
    removed = 0
    pass_index = 0
    while len(result) > 3:
        poll_cancellation(cancellation_probe, pass_index)
        pass_index += 1
        removable: set[int] = set()
        for index, current in enumerate(result):
            poll_cancellation(cancellation_probe, index)
            previous = result[index - 1]
            following = result[(index + 1) % len(result)]
            first_x = current[0] - previous[0]
            first_y = current[1] - previous[1]
            second_x = following[0] - current[0]
            second_y = following[1] - current[1]
            cross = first_x * second_y - first_y * second_x
            between = (current[0] - previous[0]) * (current[0] - following[0]) + (
                current[1] - previous[1]
            ) * (current[1] - following[1]) <= 0
            if cross == 0 and between:
                removable.add(index)
        if not removable or len(result) - len(removable) < 3:
            break
        raise_if_cancelled(cancellation_probe)
        retained: list[tuple[int, int]] = []
        for index, point in enumerate(result):
            poll_cancellation(cancellation_probe, index)
            if index not in removable:
                retained.append(point)
        result = retained
        raise_if_cancelled(cancellation_probe)
        removed += len(removable)
    raise_if_cancelled(cancellation_probe)
    return tuple(result), removed


def _integer_ring_area2(
    points: Sequence[tuple[int, int]],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> int:
    area2 = 0
    for index in range(len(points)):
        poll_cancellation(cancellation_probe, index)
        area2 += (
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
        )
    raise_if_cancelled(cancellation_probe)
    return area2


def _canonical_integer_ring(
    coordinates: Iterable[Sequence[float]],
    *,
    grid: float,
    grid_origin: tuple[int, int],
    clockwise: bool,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[tuple[tuple[int, int], ...], int, float]:
    raise_if_cancelled(cancellation_probe)
    points: list[tuple[int, int]] = []
    max_residual = 0.0
    for coordinate_index, coordinate in enumerate(coordinates):
        poll_cancellation(cancellation_probe, coordinate_index)
        x = float(coordinate[0])
        y = float(coordinate[1])
        local_x_index = int(round(x))
        local_y_index = int(round(y))
        x_index = local_x_index + grid_origin[0]
        y_index = local_y_index + grid_origin[1]
        if abs(x_index) > MAX_GRID_INDEX or abs(y_index) > MAX_GRID_INDEX:
            raise ArtifactVectorExtractionError(
                "outline coordinates exceed the fixed-grid integer safety range"
            )
        residual = max(
            abs(x - local_x_index) * grid,
            abs(y - local_y_index) * grid,
        )
        max_residual = max(max_residual, residual)
        if residual > max(grid * 1e-8, 1e-12):
            raise ArtifactVectorExtractionError(
                "outline union emitted a coordinate outside the declared precision grid"
            )
        point = (x_index, y_index)
        if not points or points[-1] != point:
            points.append(point)
    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3:
        raise ArtifactVectorExtractionError("outline union contains a collapsed ring")
    reduced, removed = _remove_integer_collinear(
        points,
        cancellation_probe=cancellation_probe,
    )
    area2 = _integer_ring_area2(
        reduced,
        cancellation_probe=cancellation_probe,
    )
    if area2 == 0:
        raise ArtifactVectorExtractionError("outline union contains a zero-area ring")
    if (clockwise and area2 > 0) or (not clockwise and area2 < 0):
        reduced = tuple(reversed(reduced))
    start = 0
    for index in range(1, len(reduced)):
        poll_cancellation(cancellation_probe, index)
        if reduced[index] < reduced[start]:
            start = index
    reduced = reduced[start:] + reduced[:start]
    raise_if_cancelled(cancellation_probe)
    return reduced, removed, max_residual


def _float_ring(
    points: Sequence[tuple[int, int]],
    *,
    grid: float,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[tuple[float, float], ...]:
    floats: list[tuple[float, float]] = []
    for index, (x, y) in enumerate(points):
        poll_cancellation(cancellation_probe, index)
        floats.append(
            (
                0.0 if x == 0 else float(x * grid),
                0.0 if y == 0 else float(y * grid),
            )
        )
    raise_if_cancelled(cancellation_probe)
    return tuple(floats)


def _ring_segment_lengths(
    paths: Sequence[VectorPath],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> Iterable[float]:
    segment_index = 0
    for path in paths:
        for index in range(len(path.points_mm)):
            poll_cancellation(cancellation_probe, segment_index)
            segment_index += 1
            yield math.hypot(
                path.points_mm[(index + 1) % len(path.points_mm)][0]
                - path.points_mm[index][0],
                path.points_mm[(index + 1) % len(path.points_mm)][1]
                - path.points_mm[index][1],
            )
    raise_if_cancelled(cancellation_probe)


def _close_grid_slivers(
    geometry: BaseGeometry,
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[BaseGeometry, dict[str, Any]]:
    """Close the hairline gaps the fixed grid leaves in the lattice union.

    Near a silhouette the projected triangles are thinner than a cell, and
    snapping each to the grid leaves slivers between them: holes one cell
    wide and a few cells long, and pinches where two parts of the outline
    meet at a single lattice point.  They are the grid's, not the artifact's
    - the surface is continuous there - and a finer grid only finds smaller
    ones.  A morphological closing by ``OUTLINE_GRID_CLOSING_RADIUS_CELLS``
    removes exactly the features narrower than twice that radius, and
    re-snapping puts the result back on the lattice.  On the corded test body
    the closing changes the area by a few parts per million; a real hole -
    wider than two cells - is untouched.
    """

    raise_if_cancelled(cancellation_probe)
    radius = OUTLINE_GRID_CLOSING_RADIUS_CELLS
    holes_before = sum(len(polygon.interiors) for polygon in _polygon_sequence(geometry))
    components_before = _polygon_count(geometry)
    try:
        closed = geometry.buffer(radius, join_style="mitre").buffer(
            -radius, join_style="mitre"
        )
        raise_if_cancelled(cancellation_probe)
        closed = set_precision(closed, 1.0, mode="valid_output")
    except GEOSException as exc:
        raise ArtifactVectorExtractionError(f"outline grid closing failed: {exc}") from exc
    if closed.is_empty:
        raise ArtifactVectorExtractionError("outline grid closing emptied the union")
    if not closed.is_valid:
        raise ArtifactVectorExtractionError(
            f"outline union is invalid after grid closing: {is_valid_reason(closed)}"
        )
    holes_after = sum(len(polygon.interiors) for polygon in _polygon_sequence(closed))
    components_after = _polygon_count(closed)
    return closed, {
        "grid_closing_radius_cells": radius,
        "grid_closing_hole_fill_count": max(0, holes_before - holes_after),
        "grid_closing_component_merge_count": max(0, components_before - components_after),
        "grid_closing_area_delta_cells": float(closed.area - geometry.area),
    }


def _component_summary(
    components: Sequence[tuple[Any, Sequence[Any]]], grid: float
) -> str:
    """How many pieces the projection has, and how small the smallest is.

    A drafter reading a refusal needs to know whether the drawing is one
    artifact or an artifact and a speck of scanner noise.
    """

    return _pieces_summary(
        [
            abs(_integer_ring_area2(exterior)) * grid * grid / 2.0
            for exterior, _holes in components
        ]
    )


def _refuse_grid_holes(
    geometry: BaseGeometry,
    unsnapped: BaseGeometry | None,
    *,
    grid: float,
    unsnapped_status: str,
    cancellation_probe: CancellationProbe | None = None,
) -> dict[str, Any]:
    """Refuse a hole the artifact's surface is continuous under.

    Every hole of the closed lattice union is measured against the unsnapped
    union of the same triangles.  A hole in the artifact is a hole there too,
    and the unsnapped union covers only the snap error at its rim; a hole the
    grid punched - thin triangles collapsed, and the closing could not span
    the gap - is covered entirely, because the surface never opened.  The
    outline is refused when any hole is covered by more than
    ``OUTLINE_GRID_HOLE_COVER_FRACTION_MAX`` of its area, and the largest
    covered fraction of the holes that stay is returned as QC.
    """

    raise_if_cancelled(cancellation_probe)
    holes = [
        Polygon(interior.coords)
        for polygon in _polygon_sequence(geometry)
        for interior in polygon.interiors
    ]
    if not holes:
        return {"grid_hole_unsnapped_cover_max": 0.0}
    if unsnapped is None:
        raise ArtifactVectorExtractionError(
            f"outline has {len(holes)} holes and the unsnapped comparison is "
            f"{unsnapped_status}, so it cannot tell the artifact's holes from "
            "the grid's; use a finer grid"
        )
    covered: list[tuple[float, Polygon]] = []
    try:
        for index, hole in enumerate(holes):
            poll_cancellation(cancellation_probe, index)
            area = float(hole.area)
            fraction = float(hole.intersection(unsnapped).area) / area if area > 0 else 1.0
            covered.append((min(max(fraction, 0.0), 1.0), hole))
    except GEOSException as exc:
        raise ArtifactVectorExtractionError(
            f"outline hole comparison failed: {exc}"
        ) from exc
    punched = [
        item for item in covered if item[0] > OUTLINE_GRID_HOLE_COVER_FRACTION_MAX
    ]
    if punched:
        fraction, hole = max(punched, key=lambda item: float(item[1].area))
        x0, y0, x1, y1 = hole.bounds
        raise ArtifactVectorExtractionError(
            f"{len(punched)} of {len(holes)} holes in the outline are the grid's, "
            "not the artifact's: the unsnapped projection covers "
            f"{fraction:.0%} of a hole {(x1 - x0) * grid:.1f} x {(y1 - y0) * grid:.1f} mm, "
            "so the surface is continuous where the drawing shows a hole. The "
            f"{grid} mm grid is coarser than the mesh there; use a finer "
            "precision_grid_mm"
        )
    return {
        "grid_hole_unsnapped_cover_max": round(max(item[0] for item in covered), 6)
    }


def _pieces_summary(areas: Sequence[float]) -> str:
    areas_mm2 = sorted(float(area) for area in areas)
    if not areas_mm2:
        return "no exterior at all"
    if len(areas_mm2) == 1:
        return f"one piece of {areas_mm2[0]:.1f} mm2"
    return (
        f"{len(areas_mm2)} separate pieces, the largest {areas_mm2[-1]:.1f} mm2 "
        f"and the smallest {areas_mm2[0]:.4g} mm2; a piece far smaller than the "
        "artifact is usually a loose fragment in the mesh rather than part of "
        "what is being drawn"
    )


def _payload_from_union(
    geometry: BaseGeometry,
    *,
    frame: PlanarFrame,
    grid: float,
    grid_origin: tuple[int, int],
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[VectorGeometryPayload, dict[str, Any]]:
    raise_if_cancelled(cancellation_probe)
    components: list[
        tuple[
            tuple[tuple[int, int], ...],
            tuple[tuple[tuple[int, int], ...], ...],
        ]
    ] = []
    collinear_removed = 0
    max_grid_residual = 0.0
    for polygon_index, polygon in enumerate(_polygon_sequence(geometry)):
        poll_cancellation(cancellation_probe, polygon_index)
        exterior, removed, residual = _canonical_integer_ring(
            polygon.exterior.coords,
            grid=grid,
            grid_origin=grid_origin,
            clockwise=False,
            cancellation_probe=cancellation_probe,
        )
        collinear_removed += removed
        max_grid_residual = max(max_grid_residual, residual)
        holes: list[tuple[tuple[int, int], ...]] = []
        for interior_index, interior in enumerate(polygon.interiors):
            poll_cancellation(cancellation_probe, interior_index)
            hole, removed, residual = _canonical_integer_ring(
                interior.coords,
                grid=grid,
                grid_origin=grid_origin,
                clockwise=True,
                cancellation_probe=cancellation_probe,
            )
            holes.append(hole)
            collinear_removed += removed
            max_grid_residual = max(max_grid_residual, residual)
        components.append((exterior, tuple(sorted(holes))))
    raise_if_cancelled(cancellation_probe)
    components.sort(key=lambda item: (item[0], item[1]))
    raise_if_cancelled(cancellation_probe)

    paths: list[VectorPath] = []
    for component_index, (exterior, component_holes) in enumerate(components):
        poll_cancellation(cancellation_probe, component_index)
        paths.append(
            VectorPath(
                id=f"outline:component:{component_index:04d}:exterior",
                role="exterior",
                closed=True,
                points_mm=_float_ring(
                    exterior,
                    grid=grid,
                    cancellation_probe=cancellation_probe,
                ),
            )
        )
        for hole_index, hole in enumerate(component_holes):
            poll_cancellation(cancellation_probe, hole_index)
            paths.append(
                VectorPath(
                    id=(
                        f"outline:component:{component_index:04d}:hole:{hole_index:04d}"
                    ),
                    role="hole",
                    closed=True,
                    points_mm=_float_ring(
                        hole,
                        grid=grid,
                        cancellation_probe=cancellation_probe,
                    ),
                )
            )
    point_count = sum(len(path.points_mm) for path in paths)
    if point_count > MAX_VECTOR_POINTS:
        raise ArtifactVectorExtractionError(
            f"outline exceeds the {MAX_VECTOR_POINTS}-point vector safety limit"
        )
    raise_if_cancelled(cancellation_probe)
    payload = VectorGeometryPayload(
        schema_version=VECTOR_PAYLOAD_SCHEMA_VERSION,
        kind=VectorRecordKind.OUTLINE,
        coordinate_space=VECTOR_COORDINATE_SPACE,
        frame=frame,
        paths=tuple(paths),
    )
    raise_if_cancelled(cancellation_probe)

    try:
        topology = validate_outline_topology(
            payload,
            precision_grid_mm=grid,
            cancellation_probe=cancellation_probe,
        )
    except ValueError as exc:
        # Say what the drawing is made of as well as which rule broke.  A real
        # scan carries specks: the museum's 신수22891 is one body of 391,432
        # faces and one loose crumb of 24, and the crumb lands beside the pot
        # on the lattice and touches it at a corner.  "exteriors_touch" alone
        # sends a drafter looking for a fault in the pot.
        raise ArtifactVectorExtractionError(
            f"canonical outline topology is invalid: {exc}"
            f"; this projection has {_component_summary(components, grid)}"
        ) from exc
    area_lattice2 = 0
    for component_index, (exterior, component_holes) in enumerate(components):
        poll_cancellation(cancellation_probe, component_index)
        component_area2 = abs(
            _integer_ring_area2(
                exterior,
                cancellation_probe=cancellation_probe,
            )
        )
        for hole_index, hole in enumerate(component_holes):
            poll_cancellation(cancellation_probe, hole_index)
            component_area2 -= abs(
                _integer_ring_area2(
                    hole,
                    cancellation_probe=cancellation_probe,
                )
            )
        area_lattice2 += component_area2
    outline_area_mm2 = round(area_lattice2 * grid * grid / 2.0, 12)
    outline_perimeter_mm = round(
        math.fsum(
            _ring_segment_lengths(
                payload.paths,
                cancellation_probe=cancellation_probe,
            )
        ),
        12,
    )
    # The lattice area is exact; the validator's is a float sum over the same
    # rings.  On a fine grid a body-sized outline (thousands of mm^2, tens of
    # thousands of vertices) accumulates a few 1e-9 mm^2 of rounding, which is
    # not a disagreement about the shape, so the tolerance is relative to the
    # area as well as to the grid cell.
    if not math.isclose(
        outline_area_mm2,
        topology.area_mm2,
        rel_tol=0.0,
        abs_tol=max(grid * grid * 1e-8, abs(topology.area_mm2) * 1e-9, 1e-12),
    ):
        raise ArtifactVectorExtractionError(
            "integer-lattice and topology-validator outline areas disagree"
        )
    raise_if_cancelled(cancellation_probe)
    return payload, {
        "component_count": len(components),
        "hole_count": sum(
            len(component_holes) for _exterior, component_holes in components
        ),
        "outline_area_mm2": outline_area_mm2,
        "outline_perimeter_mm": outline_perimeter_mm,
        "outline_collinear_point_removal_count": collinear_removed,
        "output_grid_residual_max_mm": max_grid_residual,
        "topology_valid": True,
    }


def extract_outline_geometry(
    vertices_world_mm: object,
    faces: object,
    view: OutlineView | str,
    *,
    precision_grid_mm: float,
    algorithm_version: str = OUTLINE_ALGORITHM_VERSION,
    cancellation_probe: CancellationProbe | None = None,
) -> OutlineGeometryResult:
    """Project and fixed-grid union every triangle into one canonical view.

    ``algorithm_version`` selects the contract the result must reproduce:
    1.0.0 is the lattice union alone, 1.1.0 adds the one-cell grid closing.
    """

    raise_if_cancelled(cancellation_probe)
    _require_outline_backend()
    resolved = _outline_view(view)
    grid = _precision_grid(precision_grid_mm)
    version = _outline_algorithm_version(algorithm_version)
    closing = version != OUTLINE_LEGACY_ALGORITHM_VERSION
    piece_gate = version not in (
        OUTLINE_LEGACY_ALGORITHM_VERSION,
        OUTLINE_CLOSING_ALGORITHM_VERSION,
    )
    hole_gate = version == OUTLINE_ALGORITHM_VERSION
    snap_contract, snap_cells = _SNAP_CONTRACTS[version]
    vertices, face_array = _validated_mesh_arrays(
        vertices_world_mm,
        faces,
        cancellation_probe=cancellation_probe,
    )
    raise_if_cancelled(cancellation_probe)
    if vertices.shape[0] > MAX_OUTLINE_VERTICES:
        raise ArtifactVectorExtractionError(
            f"outline exceeds the {MAX_OUTLINE_VERTICES}-vertex safety limit"
        )
    if face_array.shape[0] > MAX_OUTLINE_FACES:
        raise ArtifactVectorExtractionError(
            f"outline exceeds the {MAX_OUTLINE_FACES}-face safety limit"
        )
    frame = outline_frame(resolved)
    origin = np.asarray(frame.origin_world_mm, dtype=np.float64)
    u_axis = np.asarray(frame.u_axis_world, dtype=np.float64)
    v_axis = np.asarray(frame.v_axis_world, dtype=np.float64)
    relative = vertices - origin
    raise_if_cancelled(cancellation_probe)
    projected_vertices = np.column_stack((relative @ u_axis, relative @ v_axis))
    raise_if_cancelled(cancellation_probe)
    referenced_indices = np.unique(face_array.reshape(-1))
    raise_if_cancelled(cancellation_probe)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        referenced_scaled = projected_vertices[referenced_indices] / grid
    raise_if_cancelled(cancellation_probe)
    referenced_scaled_is_finite = bool(np.isfinite(referenced_scaled).all())
    raise_if_cancelled(cancellation_probe)
    if not referenced_scaled_is_finite:
        raise ArtifactVectorExtractionError(
            "outline projection is non-finite at the selected precision grid"
        )
    maximum_grid_index = float(np.max(np.abs(referenced_scaled)))
    raise_if_cancelled(cancellation_probe)
    if maximum_grid_index > MAX_GRID_INDEX:
        raise ArtifactVectorExtractionError(
            "outline coordinates exceed the fixed-grid integer safety range"
        )
    minimum_u_index = math.floor(float(np.min(referenced_scaled[:, 0])))
    raise_if_cancelled(cancellation_probe)
    minimum_v_index = math.floor(float(np.min(referenced_scaled[:, 1])))
    raise_if_cancelled(cancellation_probe)
    grid_origin = (minimum_u_index, minimum_v_index)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lattice_vertices = projected_vertices / grid - np.asarray(
            grid_origin, dtype=np.float64
        )
    raise_if_cancelled(cancellation_probe)
    lattice_vertices_are_finite = bool(
        np.isfinite(lattice_vertices[referenced_indices]).all()
    )
    raise_if_cancelled(cancellation_probe)
    if not lattice_vertices_are_finite:
        raise ArtifactVectorExtractionError(
            "outline lattice coordinates are non-finite"
        )
    raise_if_cancelled(cancellation_probe)

    snapped_batches: list[BaseGeometry] = []
    raw_batches: list[BaseGeometry] = []
    raw_comparison_available = True
    raw_comparison_failure = ""
    grid_collapsed_count = 0
    projected_degenerate_count = 0
    projected_non_degenerate_count = 0
    fixed_grid_triangle_count = 0
    face_chunk_count = 0
    for start in range(0, face_array.shape[0], OUTLINE_UNION_BATCH_SIZE):
        raise_if_cancelled(cancellation_probe)
        face_chunk_count += 1
        face_chunk = face_array[start : start + OUTLINE_UNION_BATCH_SIZE]
        triangles = lattice_vertices[face_chunk]
        twice_area = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
            triangles[:, 2, 1] - triangles[:, 0, 1]
        ) - (triangles[:, 1, 1] - triangles[:, 0, 1]) * (
            triangles[:, 2, 0] - triangles[:, 0, 0]
        )
        projected_degenerate_count += int(np.count_nonzero(twice_area == 0.0))
        candidate_triangles = triangles[twice_area != 0.0]
        projected_non_degenerate_count += int(candidate_triangles.shape[0])
        if candidate_triangles.shape[0] == 0:
            raise_if_cancelled(cancellation_probe)
            continue

        raw_polygons: list[Polygon] = []
        snapped_polygons: list[BaseGeometry] = []
        try:
            for triangle_index, coordinates in enumerate(candidate_triangles):
                poll_cancellation(cancellation_probe, triangle_index)
                polygon = Polygon(coordinates)
                raw_polygons.append(polygon)
                snapped = set_precision(polygon, 1.0, mode="valid_output")
                if snapped.is_empty:
                    grid_collapsed_count += 1
                    continue
                if not isinstance(snapped, (Polygon, MultiPolygon)):
                    raise ArtifactVectorExtractionError(
                        "fixed-grid projection produced non-polygonal triangle geometry"
                    )
                snapped_polygons.append(snapped)
        except GEOSException as exc:
            raise ArtifactVectorExtractionError(
                f"outline precision snapping failed: {exc}"
            ) from exc
        fixed_grid_triangle_count += len(snapped_polygons)
        if snapped_polygons:
            snapped_batches.append(
                _batched_union(
                    snapped_polygons,
                    grid_size=1.0,
                    cancellation_probe=cancellation_probe,
                )
            )
        if raw_comparison_available:
            try:
                raw_batches.append(
                    _batched_union(
                        raw_polygons,
                        grid_size=None,
                        cancellation_probe=cancellation_probe,
                    )
                )
            except ArtifactVectorExtractionError as exc:
                raw_comparison_available = False
                raw_comparison_failure = type(exc).__name__
                raw_batches.clear()
        raise_if_cancelled(cancellation_probe)
    raise_if_cancelled(cancellation_probe)

    if projected_non_degenerate_count == 0:
        raise ArtifactVectorExtractionError(
            "outline projection contains no non-degenerate triangle area"
        )
    if not snapped_batches:
        raise ArtifactVectorExtractionError(
            "all projected triangle areas collapse at the selected precision grid"
        )

    snapped_union = _balanced_union(
        snapped_batches,
        grid_size=1.0,
        cancellation_probe=cancellation_probe,
    )
    raise_if_cancelled(cancellation_probe)
    if snapped_union.is_empty:
        raise ArtifactVectorExtractionError(
            "outline polygon union is empty at the selected precision grid"
        )
    if not snapped_union.is_valid:
        raise ArtifactVectorExtractionError(
            f"outline polygon union is invalid: {is_valid_reason(snapped_union)}"
        )
    closing_qc: dict[str, Any] = {}
    if closing:
        snapped_union, closing_cells = _close_grid_slivers(
            snapped_union, cancellation_probe=cancellation_probe
        )
        closing_qc = {
            "grid_closing_area_delta_mm2": round(
                float(closing_cells.pop("grid_closing_area_delta_cells")) * grid * grid,
                12,
            ),
            **closing_cells,
        }
    snapped_component_count = _polygon_count(snapped_union)

    unsnapped_status = "available"
    unsnapped_union: BaseGeometry | None = None
    unsnapped_component_count: int | None
    unsnapped_area_mm2: float | None
    try:
        if not raw_comparison_available or not raw_batches:
            raise ArtifactVectorExtractionError(
                raw_comparison_failure or "raw comparison has no batches"
            )
        unsnapped_union = _balanced_union(
            raw_batches,
            grid_size=None,
            cancellation_probe=cancellation_probe,
        )
        unsnapped_component_count = _polygon_count(unsnapped_union)
        unsnapped_area_mm2 = round(
            float(unsnapped_union.area) * grid * grid,
            12,
        )
    except ArtifactVectorExtractionError:
        unsnapped_status = "unavailable_geos_union_failure"
        unsnapped_union = None
        unsnapped_component_count = None
        unsnapped_area_mm2 = None

    if piece_gate and snapped_component_count > 1:
        # A connected mesh has a connected silhouette.  Two pieces here are a
        # loose fragment in the mesh or a snap that cut the artifact in two,
        # and neither belongs on a measured drawing - so say which it looks
        # like, in millimetres, rather than draw it.  The unsnapped union
        # tells the two apart: a fragment is a piece of its own there as
        # well, a severed sliver is not.
        if unsnapped_component_count is None:
            cause = (
                ". Remove the loose fragment from the mesh, or use a finer grid "
                "if the mesh itself is one piece"
            )
        elif unsnapped_component_count == 1:
            cause = (
                ". The unsnapped projection is one piece, so the grid severed "
                "the artifact: use a finer grid"
            )
        else:
            cause = (
                f". The unsnapped projection is {unsnapped_component_count} "
                "pieces as well, so the mesh carries a loose fragment: remove it"
            )
        raise ArtifactVectorExtractionError(
            "outline is more than one piece, and a drawing may hold only the "
            f"artifact: {_pieces_summary([polygon.area * grid * grid for polygon in _polygon_sequence(snapped_union)])}"
            + cause
        )
    hole_gate_qc: dict[str, Any] = {}
    if hole_gate:
        hole_gate_qc = _refuse_grid_holes(
            snapped_union,
            unsnapped_union,
            grid=grid,
            unsnapped_status=unsnapped_status,
            cancellation_probe=cancellation_probe,
        )

    payload, topology_qc = _payload_from_union(
        snapped_union,
        frame=frame,
        grid=grid,
        grid_origin=grid_origin,
        cancellation_probe=cancellation_probe,
    )
    component_merge_count = (
        max(0, unsnapped_component_count - snapped_component_count)
        if unsnapped_component_count is not None
        else None
    )
    component_split_count = (
        max(0, snapped_component_count - unsnapped_component_count)
        if unsnapped_component_count is not None
        else None
    )
    snapped_area_mm2 = float(topology_qc["outline_area_mm2"])
    qc = {
        "backend_geos_version": shapely.geos_version_string,
        "backend_shapely_version": shapely.__version__,
        "all_projected_faces_included": True,
        "face_chunk_count": face_chunk_count,
        "fixed_grid_triangle_count": fixed_grid_triangle_count,
        "grid_area_delta_mm2": (
            round(snapped_area_mm2 - unsnapped_area_mm2, 12)
            if unsnapped_area_mm2 is not None
            else None
        ),
        "grid_collapsed_triangle_count": grid_collapsed_count,
        # The closing's four keys exist only from algorithm 1.1.0 on, and the
        # hole gate's one only from 1.3.0, so an older recomputation keeps
        # its QC bytes.
        **closing_qc,
        **hole_gate_qc,
        "grid_component_merge_count": component_merge_count,
        "grid_component_split_count": component_split_count,
        "grid_snap_axis_upper_bound_mm": grid * snap_cells,
        "grid_snap_error_contract": snap_contract,
        "grid_snap_radial_upper_bound_squared_mm2": 2.0 * (grid * snap_cells) ** 2,
        "grid_origin_index_uv": [grid_origin[0], grid_origin[1]],
        "input_face_count": int(face_array.shape[0]),
        "input_vertex_count": int(vertices.shape[0]),
        "precision_grid_mm": grid,
        "projected_degenerate_triangle_count": projected_degenerate_count,
        "projected_non_degenerate_triangle_count": projected_non_degenerate_count,
        "sampling_applied": False,
        "unsnapped_area_mm2": unsnapped_area_mm2,
        "unsnapped_component_count": unsnapped_component_count,
        "unsnapped_comparison_status": unsnapped_status,
        "view": resolved.value,
        **topology_qc,
    }
    raise_if_cancelled(cancellation_probe)
    return OutlineGeometryResult(payload=payload, qc=qc)


def compute_artifact_outline(
    session: ArtifactSession,
    view: OutlineView | str,
    *,
    precision_grid_mm: float,
    cancellation_probe: CancellationProbe | None = None,
) -> ArtifactVectorComputation:
    """Capture document context and compute one six-view authoritative outline."""

    raise_if_cancelled(cancellation_probe)
    if not isinstance(session, ArtifactSession):
        raise ArtifactVectorExtractionError("session must be an ArtifactSession")
    resolved = _outline_view(view)
    recipe = outline_recipe(resolved, precision_grid_mm=precision_grid_mm)
    try:
        context = session.capture_vector_operation(
            recipe=recipe,
        )
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactVectorExtractionError(str(exc)) from exc
    raise_if_cancelled(cancellation_probe)
    geometry = extract_outline_geometry(
        projection.mesh.vertices,
        projection.mesh.faces,
        resolved,
        precision_grid_mm=precision_grid_mm,
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


REGION_ALGORITHM = "archmeshrubbing.region_projection"
REGION_ALGORITHM_VERSION = "1.0.0"


def extract_region_geometry(
    vertices_world_mm: object,
    faces: object,
    view: OutlineView | str,
    *,
    precision_grid_mm: float,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[VectorGeometryPayload, dict[str, Any]]:
    """The silhouette of a painted face subset in one view.

    An outline is measured: every triangle is snapped to the lattice on its
    own and the union of the snapped triangles is the contract.  A painted
    region is not a measurement of the artifact's edge, and snapping its
    thousands of tiny triangles one by one leaves slivers between them that
    the one-cell closing cannot always mend - a smooth wall would then refuse
    to carry a finger mark.  So a region is unioned exactly first and snapped
    once as a whole, then closed by one cell and re-snapped, and only the
    result has to pass the topology rules.  The boundary is still on the
    lattice, within the closing contract of outline 1.1.0.
    """

    raise_if_cancelled(cancellation_probe)
    _require_outline_backend()
    resolved = _outline_view(view)
    grid = _precision_grid(precision_grid_mm)
    vertices, face_array = _validated_mesh_arrays(
        vertices_world_mm, faces, cancellation_probe=cancellation_probe
    )
    if vertices.shape[0] > MAX_OUTLINE_VERTICES:
        raise ArtifactVectorExtractionError(
            f"region exceeds the {MAX_OUTLINE_VERTICES}-vertex safety limit"
        )
    if face_array.shape[0] > MAX_OUTLINE_FACES:
        raise ArtifactVectorExtractionError(
            f"region exceeds the {MAX_OUTLINE_FACES}-face safety limit"
        )
    frame = outline_frame(resolved)
    origin = np.asarray(frame.origin_world_mm, dtype=np.float64)
    u_axis = np.asarray(frame.u_axis_world, dtype=np.float64)
    v_axis = np.asarray(frame.v_axis_world, dtype=np.float64)
    relative = vertices - origin
    projected_vertices = np.column_stack((relative @ u_axis, relative @ v_axis))
    referenced_indices = np.unique(face_array.reshape(-1))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        referenced_scaled = projected_vertices[referenced_indices] / grid
    if not bool(np.isfinite(referenced_scaled).all()):
        raise ArtifactVectorExtractionError(
            "region projection is non-finite at the selected precision grid"
        )
    if float(np.max(np.abs(referenced_scaled))) > MAX_GRID_INDEX:
        raise ArtifactVectorExtractionError(
            "region coordinates exceed the fixed-grid integer safety range"
        )
    grid_origin = (
        math.floor(float(np.min(referenced_scaled[:, 0]))),
        math.floor(float(np.min(referenced_scaled[:, 1]))),
    )
    lattice_vertices = projected_vertices / grid - np.asarray(grid_origin, dtype=np.float64)
    raise_if_cancelled(cancellation_probe)

    batches: list[BaseGeometry] = []
    non_degenerate = 0
    for start in range(0, face_array.shape[0], OUTLINE_UNION_BATCH_SIZE):
        raise_if_cancelled(cancellation_probe)
        triangles = lattice_vertices[face_array[start : start + OUTLINE_UNION_BATCH_SIZE]]
        twice_area = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
            triangles[:, 2, 1] - triangles[:, 0, 1]
        ) - (triangles[:, 1, 1] - triangles[:, 0, 1]) * (triangles[:, 2, 0] - triangles[:, 0, 0])
        candidates = triangles[twice_area != 0.0]
        non_degenerate += int(candidates.shape[0])
        if candidates.shape[0] == 0:
            continue
        batches.append(
            _batched_union(
                [Polygon(coordinates) for coordinates in candidates],
                grid_size=None,
                cancellation_probe=cancellation_probe,
            )
        )
    if non_degenerate == 0 or not batches:
        raise ArtifactVectorExtractionError(
            "region projection contains no non-degenerate triangle area"
        )
    union = _balanced_union(batches, grid_size=None, cancellation_probe=cancellation_probe)
    try:
        snapped = set_precision(union, 1.0, mode="valid_output")
    except GEOSException as exc:
        raise ArtifactVectorExtractionError(f"region precision snapping failed: {exc}") from exc
    if snapped.is_empty:
        raise ArtifactVectorExtractionError(
            "region area collapses at the selected precision grid"
        )
    if not snapped.is_valid:
        raise ArtifactVectorExtractionError(
            f"region union is invalid: {is_valid_reason(snapped)}"
        )
    closed, closing_cells = _close_grid_slivers(snapped, cancellation_probe=cancellation_probe)
    payload, topology_qc = _payload_from_union(
        closed,
        frame=frame,
        grid=grid,
        grid_origin=grid_origin,
        cancellation_probe=cancellation_probe,
    )
    qc = {
        "algorithm": REGION_ALGORITHM,
        "algorithm_version": REGION_ALGORITHM_VERSION,
        "backend_geos_version": shapely.geos_version_string,
        "backend_shapely_version": shapely.__version__,
        "grid_closing_area_delta_mm2": round(
            float(closing_cells.pop("grid_closing_area_delta_cells")) * grid * grid, 12
        ),
        **closing_cells,
        "grid_origin_index_uv": [grid_origin[0], grid_origin[1]],
        "input_face_count": int(face_array.shape[0]),
        "precision_grid_mm": grid,
        "projected_non_degenerate_triangle_count": non_degenerate,
        **topology_qc,
    }
    return payload, qc


__all__ = [
    "DEFAULT_OUTLINE_PRECISION_GRID_MM",
    "REGION_ALGORITHM",
    "REGION_ALGORITHM_VERSION",
    "extract_region_geometry",
    "MAX_GRID_INDEX",
    "MAX_OUTLINE_FACES",
    "MAX_OUTLINE_INTERMEDIATE_COORDINATES",
    "MAX_OUTLINE_INTERMEDIATE_POLYGONS",
    "MAX_OUTLINE_VERTICES",
    "OUTLINE_ALGORITHM",
    "OUTLINE_ALGORITHM_VERSION",
    "OUTLINE_ALGORITHM_VERSIONS",
    "OUTLINE_CLOSING_ALGORITHM_VERSION",
    "OUTLINE_GRID_CLOSING_RADIUS_CELLS",
    "OUTLINE_GRID_HOLE_COVER_FRACTION_MAX",
    "OUTLINE_PIECE_GATE_ALGORITHM_VERSION",
    "OUTLINE_LEGACY_ALGORITHM_VERSION",
    "OUTLINE_UNION_BATCH_SIZE",
    "OutlineGeometryResult",
    "OutlineView",
    "REQUIRED_GEOS_VERSION",
    "REVIEWED_OUTLINE_BACKENDS",
    "REQUIRED_SHAPELY_VERSION",
    "compute_artifact_outline",
    "extract_outline_geometry",
    "outline_frame",
    "outline_recipe",
    "validate_outline_record_contract",
]
