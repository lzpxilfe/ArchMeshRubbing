"""Strict Shapely topology validation for authoritative outline payloads.

``VectorGeometryPayload`` establishes the syntactic outline contract.  This
module adds the geometric contract which cannot be proven from path roles and
orientation alone: every ring is simple and non-zero, each hole belongs to
exactly one exterior, components are disjoint, and the assembled Polygon or
MultiPolygon is valid.

Validation is deliberately read-only.  It never repairs, buffers, unions, or
otherwise changes measurement geometry.  Invalid topology fails with a typed
error carrying deterministic, JSON-compatible diagnostics.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import math
from typing import Any, Sequence

from shapely import LinearRing, MultiPolygon, Polygon, contains_properly
from shapely.errors import GEOSException
from shapely.validation import explain_validity

from .artifact_cancellation import (
    CancellationProbe,
    poll_cancellation,
    raise_if_cancelled,
)
from .artifact_vector_record import (
    VectorGeometryPayload,
    VectorPath,
    VectorRecordKind,
)


class ArtifactOutlineTopologyError(ValueError):
    """An outline cannot form an unambiguous valid polygon topology."""

    def __init__(
        self,
        code: str,
        reason: str,
        *,
        path_ids: Sequence[str] = (),
    ) -> None:
        canonical_ids = tuple(sorted(set(path_ids)))
        self.code = str(code)
        self.reason = str(reason)
        self.path_ids = canonical_ids
        suffix = f" (paths: {', '.join(canonical_ids)})" if canonical_ids else ""
        super().__init__(f"{self.code}: {self.reason}{suffix}")

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return stable JSON-compatible failure diagnostics."""

        return {
            "code": self.code,
            "path_ids": list(self.path_ids),
            "reason": self.reason,
            "topology_valid": False,
        }


@dataclass(frozen=True, slots=True)
class OutlineTopologyDiagnostics:
    """Deterministic summary of one successfully validated outline."""

    geometry_type: str
    exterior_path_ids: tuple[str, ...]
    hole_assignments: tuple[tuple[str, str], ...]
    component_hole_counts: tuple[int, ...]
    component_areas_mm2: tuple[float, ...]
    bounds_mm: tuple[float, float, float, float]
    area_mm2: float
    validity_reason: str

    @property
    def exterior_count(self) -> int:
        return len(self.exterior_path_ids)

    @property
    def hole_count(self) -> int:
        return len(self.hole_assignments)

    @property
    def component_count(self) -> int:
        return len(self.exterior_path_ids)

    @property
    def ring_count(self) -> int:
        return self.exterior_count + self.hole_count

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible QC representation."""

        return {
            "area_mm2": self.area_mm2,
            "bounds_mm": list(self.bounds_mm),
            "component_areas_mm2": list(self.component_areas_mm2),
            "component_count": self.component_count,
            "component_exterior_path_ids": list(self.exterior_path_ids),
            "component_hole_counts": list(self.component_hole_counts),
            "exterior_count": self.exterior_count,
            "geometry_type": self.geometry_type,
            "hole_assignments": [
                {
                    "exterior_path_id": exterior_id,
                    "hole_path_id": hole_id,
                }
                for hole_id, exterior_id in self.hole_assignments
            ],
            "hole_count": self.hole_count,
            "ring_count": self.ring_count,
            "topology_valid": True,
            "validity_reason": self.validity_reason,
        }


@dataclass(frozen=True, slots=True)
class _RingGeometry:
    path: VectorPath
    polygon: Polygon


# A ring no wider than this many precision-grid cells is at the grid's own
# resolution, so a refusal can say where it came from.
_GRID_PINHOLE_CELLS = 4.0


def _path_order_key(path: VectorPath) -> tuple[Any, ...]:
    role_rank = {"exterior": 0, "hole": 1}.get(path.role, 2)
    return (role_rank, path.points_mm, path.id)


def _issue(
    code: str,
    reason: str,
    *paths: VectorPath,
) -> ArtifactOutlineTopologyError:
    return ArtifactOutlineTopologyError(
        code,
        reason,
        path_ids=tuple(path.id for path in paths),
    )


def _ring_geometry(
    path: VectorPath,
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> _RingGeometry:
    raise_if_cancelled(cancellation_probe)
    if path.role not in {"exterior", "hole"}:
        raise _issue(
            "invalid_ring_role",
            "outline paths must use only the 'exterior' or 'hole' role",
            path,
        )
    if not path.closed:
        raise _issue(
            "open_outline_ring",
            "outline exterior and hole paths must be closed",
            path,
        )

    try:
        polygon = Polygon(path.points_mm)
        ring = LinearRing(path.points_mm)
    except (GEOSException, TypeError, ValueError) as exc:
        raise _issue(
            "ring_construction_failed",
            f"Shapely could not construct the outline ring: {exc}",
            path,
        ) from exc

    raise_if_cancelled(cancellation_probe)
    area = float(polygon.area)
    if not math.isfinite(area) or area <= 0.0:
        raise _issue(
            "ring_zero_area",
            "outline rings must enclose a finite non-zero area",
            path,
        )
    raise_if_cancelled(cancellation_probe)
    is_simple = bool(ring.is_simple)
    raise_if_cancelled(cancellation_probe)
    is_ring = bool(ring.is_ring)
    if not is_simple or not is_ring:
        raise _issue(
            "ring_not_simple",
            f"outline ring is not simple: {explain_validity(ring)}",
            path,
        )
    raise_if_cancelled(cancellation_probe)
    if not bool(polygon.is_valid):
        raise _issue(
            "ring_invalid",
            f"outline ring is invalid: {explain_validity(polygon)}",
            path,
        )
    raise_if_cancelled(cancellation_probe)
    return _RingGeometry(path=path, polygon=polygon)


def _grid_scale_note(
    hole: _RingGeometry,
    *,
    precision_grid_mm: float | None,
) -> str:
    """Say when a refused hole is the size of the grid rather than a feature.

    Snapping a triangle to the precision grid can collapse it to zero area,
    and a collapsed triangle drops out of the union and leaves its two
    neighbours joined at a point only.  Near a silhouette every smooth
    surface has such triangles, so a grid coarser than their projected width
    puts a pinhole on the outline edge.  The refusal is right - a pinhole
    touching the boundary is not a hole in an artifact - but the reader
    cannot act on it without being told where it came from.
    """

    if precision_grid_mm is None or not math.isfinite(precision_grid_mm):
        return ""
    grid = float(precision_grid_mm)
    if grid <= 0.0:
        return ""
    minimum_x, minimum_y, maximum_x, maximum_y = hole.polygon.bounds
    width_cells = (maximum_x - minimum_x) / grid
    height_cells = (maximum_y - minimum_y) / grid
    if max(width_cells, height_cells) > _GRID_PINHOLE_CELLS:
        return ""
    return (
        f"; this hole measures {width_cells:.1f} x {height_cells:.1f} cells of "
        f"the declared {grid:g} mm precision grid, so it is a pinhole at the "
        "grid's own resolution rather than a feature of the artifact: a "
        "triangle whose projection collapsed on that grid dropped out of the "
        "union. A finer precision_grid_mm resolves it, and a dense mesh or a "
        "relieved surface needs one"
    )


def _validate_hole_owners(
    exteriors: Sequence[_RingGeometry],
    holes: Sequence[_RingGeometry],
    *,
    precision_grid_mm: float | None = None,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[tuple[_RingGeometry, _RingGeometry], ...]:
    assignments: list[tuple[_RingGeometry, _RingGeometry]] = []
    for hole_index, hole in enumerate(holes):
        poll_cancellation(cancellation_probe, hole_index)
        owners_list: list[_RingGeometry] = []
        for exterior in exteriors:
            raise_if_cancelled(cancellation_probe)
            if bool(contains_properly(exterior.polygon, hole.polygon)):
                owners_list.append(exterior)
        owners = tuple(owners_list)
        if len(owners) > 1:
            raise _issue(
                "hole_multiple_exteriors",
                "a hole must be strictly inside exactly one exterior",
                hole.path,
                *(owner.path for owner in owners),
            )
        if not owners:
            intersecting_list: list[_RingGeometry] = []
            for exterior in exteriors:
                raise_if_cancelled(cancellation_probe)
                if bool(exterior.polygon.intersects(hole.polygon)):
                    intersecting_list.append(exterior)
            intersecting = tuple(intersecting_list)
            if intersecting:
                raise _issue(
                    "hole_not_strictly_inside",
                    "a hole must not cross or touch an exterior boundary"
                    + _grid_scale_note(hole, precision_grid_mm=precision_grid_mm),
                    hole.path,
                    *(exterior.path for exterior in intersecting),
                )
            raise _issue(
                "hole_without_exterior",
                "a hole must be strictly inside exactly one exterior",
                hole.path,
            )
        assignments.append((hole, owners[0]))
    raise_if_cancelled(cancellation_probe)
    return tuple(assignments)


def _validate_hole_pairs(
    holes: Sequence[_RingGeometry],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> None:
    for index, first in enumerate(holes):
        for second in holes[index + 1 :]:
            raise_if_cancelled(cancellation_probe)
            if bool(first.polygon.disjoint(second.polygon)):
                continue
            raise_if_cancelled(cancellation_probe)
            if bool(first.polygon.touches(second.polygon)):
                raise _issue(
                    "holes_touch",
                    "hole rings must not touch",
                    first.path,
                    second.path,
                )
            raise _issue(
                "holes_overlap",
                "hole interiors must not overlap",
                first.path,
                second.path,
            )
    raise_if_cancelled(cancellation_probe)


def _validate_exterior_pairs(
    exteriors: Sequence[_RingGeometry],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> None:
    for index, first in enumerate(exteriors):
        for second in exteriors[index + 1 :]:
            raise_if_cancelled(cancellation_probe)
            if bool(first.polygon.disjoint(second.polygon)):
                continue
            raise_if_cancelled(cancellation_probe)
            if bool(first.polygon.touches(second.polygon)):
                raise _issue(
                    "exteriors_touch",
                    "separate exterior components must not touch",
                    first.path,
                    second.path,
                )
            raise _issue(
                "exteriors_overlap",
                "exterior component interiors must not overlap or nest",
                first.path,
                second.path,
            )
    raise_if_cancelled(cancellation_probe)


def _finite_float(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ArtifactOutlineTopologyError(
            "non_finite_diagnostics",
            "validated outline produced non-finite geometry diagnostics",
        )
    return 0.0 if number == 0.0 else number


def _path_area_mm2(
    path: VectorPath,
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> float:
    origin_x, origin_y = path.points_mm[0]

    def area_terms() -> Iterator[float]:
        for index in range(len(path.points_mm)):
            poll_cancellation(cancellation_probe, index)
            yield (
                (path.points_mm[index][0] - origin_x)
                * (path.points_mm[(index + 1) % len(path.points_mm)][1] - origin_y)
                - (path.points_mm[(index + 1) % len(path.points_mm)][0] - origin_x)
                * (path.points_mm[index][1] - origin_y)
            )
        raise_if_cancelled(cancellation_probe)

    area = 0.5 * math.fsum(area_terms())
    return abs(_finite_float(area))


def validate_outline_topology(
    payload: VectorGeometryPayload,
    *,
    precision_grid_mm: float | None = None,
    cancellation_probe: CancellationProbe | None = None,
) -> OutlineTopologyDiagnostics:
    """Validate and summarize an outline Polygon or MultiPolygon.

    The function does not normalize or repair geometry.  A successful return
    proves that every path participates in one valid polygon topology under
    the explicit ``exterior`` and ``hole`` roles.

    ``precision_grid_mm`` is used only to explain a refusal: it lets the
    message say when the offending ring is the size of the grid it was
    snapped to.  It changes no decision.
    """

    raise_if_cancelled(cancellation_probe)
    if not isinstance(payload, VectorGeometryPayload):
        raise ArtifactOutlineTopologyError(
            "invalid_payload_type",
            "payload must be a VectorGeometryPayload",
        )
    if VectorRecordKind(payload.kind) is not VectorRecordKind.OUTLINE:
        raise ArtifactOutlineTopologyError(
            "invalid_payload_kind",
            "topology validation requires an outline payload",
        )

    paths = tuple(sorted(payload.paths, key=_path_order_key))
    raise_if_cancelled(cancellation_probe)
    rings_list: list[_RingGeometry] = []
    for path_index, path in enumerate(paths):
        poll_cancellation(cancellation_probe, path_index)
        rings_list.append(_ring_geometry(path, cancellation_probe=cancellation_probe))
    rings = tuple(rings_list)
    exteriors = tuple(ring for ring in rings if ring.path.role == "exterior")
    holes = tuple(ring for ring in rings if ring.path.role == "hole")
    if not exteriors:
        raise ArtifactOutlineTopologyError(
            "missing_exterior",
            "outline topology requires at least one exterior ring",
        )

    assignments = _validate_hole_owners(
        exteriors,
        holes,
        precision_grid_mm=precision_grid_mm,
        cancellation_probe=cancellation_probe,
    )
    _validate_hole_pairs(holes, cancellation_probe=cancellation_probe)
    _validate_exterior_pairs(exteriors, cancellation_probe=cancellation_probe)

    owner_by_hole_id = {
        hole.path.id: exterior.path.id for hole, exterior in assignments
    }
    holes_by_exterior_id: dict[str, list[_RingGeometry]] = {
        exterior.path.id: [] for exterior in exteriors
    }
    for assignment_index, (hole, exterior) in enumerate(assignments):
        poll_cancellation(cancellation_probe, assignment_index)
        holes_by_exterior_id[exterior.path.id].append(hole)

    components: list[Polygon] = []
    component_hole_counts: list[int] = []
    for exterior in exteriors:
        raise_if_cancelled(cancellation_probe)
        component_holes = tuple(holes_by_exterior_id[exterior.path.id])
        try:
            component = Polygon(
                exterior.path.points_mm,
                [hole.path.points_mm for hole in component_holes],
            )
        except (GEOSException, TypeError, ValueError) as exc:
            raise _issue(
                "component_construction_failed",
                f"Shapely could not construct the outline component: {exc}",
                exterior.path,
                *(hole.path for hole in component_holes),
            ) from exc
        raise_if_cancelled(cancellation_probe)
        if not bool(component.is_valid):
            raise _issue(
                "component_invalid",
                f"assembled outline component is invalid: {explain_validity(component)}",
                exterior.path,
                *(hole.path for hole in component_holes),
            )
        components.append(component)
        component_hole_counts.append(len(component_holes))
        raise_if_cancelled(cancellation_probe)

    geometry: Polygon | MultiPolygon
    if len(components) == 1:
        geometry = components[0]
    else:
        try:
            geometry = MultiPolygon(components)
        except (GEOSException, TypeError, ValueError) as exc:
            raise ArtifactOutlineTopologyError(
                "multipolygon_construction_failed",
                f"Shapely could not construct the outline MultiPolygon: {exc}",
                path_ids=tuple(exterior.path.id for exterior in exteriors),
            ) from exc

    raise_if_cancelled(cancellation_probe)
    validity_reason = explain_validity(geometry)
    if not bool(geometry.is_valid):
        raise ArtifactOutlineTopologyError(
            "geometry_invalid",
            f"assembled outline geometry is invalid: {validity_reason}",
            path_ids=tuple(path.id for path in paths),
        )

    assignment_ids = tuple(
        (hole.path.id, owner_by_hole_id[hole.path.id]) for hole in holes
    )
    all_points_list: list[tuple[float, float]] = []
    point_index = 0
    for path in paths:
        for point in path.points_mm:
            poll_cancellation(cancellation_probe, point_index)
            point_index += 1
            all_points_list.append(point)
    all_points = tuple(all_points_list)
    bounds = (
        min(point[0] for point in all_points),
        min(point[1] for point in all_points),
        max(point[0] for point in all_points),
        max(point[1] for point in all_points),
    )
    component_areas_list: list[float] = []
    for exterior_index, exterior in enumerate(exteriors):
        poll_cancellation(cancellation_probe, exterior_index)
        component_areas_list.append(
            _finite_float(
                _path_area_mm2(
                    exterior.path,
                    cancellation_probe=cancellation_probe,
                )
                - math.fsum(
                    _path_area_mm2(
                        hole.path,
                        cancellation_probe=cancellation_probe,
                    )
                    for hole in holes_by_exterior_id[exterior.path.id]
                )
            )
        )
    component_areas = tuple(component_areas_list)
    total_area = _finite_float(math.fsum(component_areas))
    raise_if_cancelled(cancellation_probe)
    return OutlineTopologyDiagnostics(
        geometry_type=geometry.geom_type,
        exterior_path_ids=tuple(exterior.path.id for exterior in exteriors),
        hole_assignments=assignment_ids,
        component_hole_counts=tuple(component_hole_counts),
        component_areas_mm2=component_areas,
        bounds_mm=(
            _finite_float(bounds[0]),
            _finite_float(bounds[1]),
            _finite_float(bounds[2]),
            _finite_float(bounds[3]),
        ),
        area_mm2=total_area,
        validity_reason=validity_reason,
    )


__all__ = [
    "ArtifactOutlineTopologyError",
    "OutlineTopologyDiagnostics",
    "validate_outline_topology",
]
