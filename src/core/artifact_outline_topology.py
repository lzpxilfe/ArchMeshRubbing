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

from dataclasses import dataclass
import math
from typing import Any, Sequence

from shapely import LinearRing, MultiPolygon, Polygon, contains_properly
from shapely.errors import GEOSException
from shapely.validation import explain_validity

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


def _ring_geometry(path: VectorPath) -> _RingGeometry:
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

    area = float(polygon.area)
    if not math.isfinite(area) or area <= 0.0:
        raise _issue(
            "ring_zero_area",
            "outline rings must enclose a finite non-zero area",
            path,
        )
    if not bool(ring.is_simple) or not bool(ring.is_ring):
        raise _issue(
            "ring_not_simple",
            f"outline ring is not simple: {explain_validity(ring)}",
            path,
        )
    if not bool(polygon.is_valid):
        raise _issue(
            "ring_invalid",
            f"outline ring is invalid: {explain_validity(polygon)}",
            path,
        )
    return _RingGeometry(path=path, polygon=polygon)


def _validate_hole_owners(
    exteriors: Sequence[_RingGeometry],
    holes: Sequence[_RingGeometry],
) -> tuple[tuple[_RingGeometry, _RingGeometry], ...]:
    assignments: list[tuple[_RingGeometry, _RingGeometry]] = []
    for hole in holes:
        owners = tuple(
            exterior
            for exterior in exteriors
            if bool(contains_properly(exterior.polygon, hole.polygon))
        )
        if len(owners) > 1:
            raise _issue(
                "hole_multiple_exteriors",
                "a hole must be strictly inside exactly one exterior",
                hole.path,
                *(owner.path for owner in owners),
            )
        if not owners:
            intersecting = tuple(
                exterior
                for exterior in exteriors
                if bool(exterior.polygon.intersects(hole.polygon))
            )
            if intersecting:
                raise _issue(
                    "hole_not_strictly_inside",
                    "a hole must not cross or touch an exterior boundary",
                    hole.path,
                    *(exterior.path for exterior in intersecting),
                )
            raise _issue(
                "hole_without_exterior",
                "a hole must be strictly inside exactly one exterior",
                hole.path,
            )
        assignments.append((hole, owners[0]))
    return tuple(assignments)


def _validate_hole_pairs(holes: Sequence[_RingGeometry]) -> None:
    for index, first in enumerate(holes):
        for second in holes[index + 1 :]:
            if bool(first.polygon.disjoint(second.polygon)):
                continue
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


def _validate_exterior_pairs(exteriors: Sequence[_RingGeometry]) -> None:
    for index, first in enumerate(exteriors):
        for second in exteriors[index + 1 :]:
            if bool(first.polygon.disjoint(second.polygon)):
                continue
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


def _finite_float(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ArtifactOutlineTopologyError(
            "non_finite_diagnostics",
            "validated outline produced non-finite geometry diagnostics",
        )
    return 0.0 if number == 0.0 else number


def _path_area_mm2(path: VectorPath) -> float:
    origin_x, origin_y = path.points_mm[0]
    area = 0.5 * math.fsum(
        (path.points_mm[index][0] - origin_x)
        * (path.points_mm[(index + 1) % len(path.points_mm)][1] - origin_y)
        - (path.points_mm[(index + 1) % len(path.points_mm)][0] - origin_x)
        * (path.points_mm[index][1] - origin_y)
        for index in range(len(path.points_mm))
    )
    return abs(_finite_float(area))


def validate_outline_topology(
    payload: VectorGeometryPayload,
) -> OutlineTopologyDiagnostics:
    """Validate and summarize an outline Polygon or MultiPolygon.

    The function does not normalize or repair geometry.  A successful return
    proves that every path participates in one valid polygon topology under
    the explicit ``exterior`` and ``hole`` roles.
    """

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
    rings = tuple(_ring_geometry(path) for path in paths)
    exteriors = tuple(ring for ring in rings if ring.path.role == "exterior")
    holes = tuple(ring for ring in rings if ring.path.role == "hole")
    if not exteriors:
        raise ArtifactOutlineTopologyError(
            "missing_exterior",
            "outline topology requires at least one exterior ring",
        )

    assignments = _validate_hole_owners(exteriors, holes)
    _validate_hole_pairs(holes)
    _validate_exterior_pairs(exteriors)

    owner_by_hole_id = {
        hole.path.id: exterior.path.id for hole, exterior in assignments
    }
    holes_by_exterior_id: dict[str, list[_RingGeometry]] = {
        exterior.path.id: [] for exterior in exteriors
    }
    for hole, exterior in assignments:
        holes_by_exterior_id[exterior.path.id].append(hole)

    components: list[Polygon] = []
    component_hole_counts: list[int] = []
    for exterior in exteriors:
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
        if not bool(component.is_valid):
            raise _issue(
                "component_invalid",
                f"assembled outline component is invalid: {explain_validity(component)}",
                exterior.path,
                *(hole.path for hole in component_holes),
            )
        components.append(component)
        component_hole_counts.append(len(component_holes))

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
    all_points = tuple(point for path in paths for point in path.points_mm)
    bounds = (
        min(point[0] for point in all_points),
        min(point[1] for point in all_points),
        max(point[0] for point in all_points),
        max(point[1] for point in all_points),
    )
    component_areas = tuple(
        _finite_float(
            _path_area_mm2(exterior.path)
            - math.fsum(
                _path_area_mm2(hole.path)
                for hole in holes_by_exterior_id[exterior.path.id]
            )
        )
        for exterior in exteriors
    )
    total_area = _finite_float(math.fsum(component_areas))
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
