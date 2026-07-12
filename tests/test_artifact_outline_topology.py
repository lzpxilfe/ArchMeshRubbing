from __future__ import annotations

import unittest

from src.core.artifact_outline_topology import (
    ArtifactOutlineTopologyError,
    validate_outline_topology,
)
from src.core.artifact_vector_record import (
    PlanarFrame,
    VECTOR_COORDINATE_SPACE,
    VECTOR_PAYLOAD_SCHEMA_VERSION,
    VectorGeometryPayload,
    VectorPath,
    VectorRecordKind,
)


def _frame() -> PlanarFrame:
    return PlanarFrame(
        origin_world_mm=(0.0, 0.0, 0.0),
        u_axis_world=(1.0, 0.0, 0.0),
        v_axis_world=(0.0, 1.0, 0.0),
        normal_world=(0.0, 0.0, 1.0),
    )


def _path(
    path_id: str,
    role: str,
    points: tuple[tuple[float, float], ...],
    *,
    closed: bool = True,
) -> VectorPath:
    return VectorPath(
        id=path_id,
        role=role,
        closed=closed,
        points_mm=points,
    )


def _outline(*paths: VectorPath) -> VectorGeometryPayload:
    return VectorGeometryPayload(
        schema_version=VECTOR_PAYLOAD_SCHEMA_VERSION,
        kind=VectorRecordKind.OUTLINE,
        coordinate_space=VECTOR_COORDINATE_SPACE,
        frame=_frame(),
        paths=tuple(paths),
    )


def _unchecked_outline(*paths: VectorPath) -> VectorGeometryPayload:
    """Build malformed payloads to verify this module's defensive boundary."""

    payload = object.__new__(VectorGeometryPayload)
    object.__setattr__(payload, "schema_version", VECTOR_PAYLOAD_SCHEMA_VERSION)
    object.__setattr__(payload, "kind", VectorRecordKind.OUTLINE)
    object.__setattr__(payload, "coordinate_space", VECTOR_COORDINATE_SPACE)
    object.__setattr__(payload, "frame", _frame())
    object.__setattr__(payload, "paths", tuple(paths))
    return payload


class TestValidOutlineTopology(unittest.TestCase):
    def test_polygon_with_hole_returns_qc_ready_diagnostics(self):
        payload = _outline(
            _path(
                "exterior:main",
                "exterior",
                ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
            ),
            _path(
                "hole:mark",
                "hole",
                ((2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)),
            ),
        )

        diagnostics = validate_outline_topology(payload)

        self.assertEqual(
            diagnostics.to_dict(),
            {
                "area_mm2": 96.0,
                "bounds_mm": [0.0, 0.0, 10.0, 10.0],
                "component_areas_mm2": [96.0],
                "component_count": 1,
                "component_exterior_path_ids": ["exterior:main"],
                "component_hole_counts": [1],
                "exterior_count": 1,
                "geometry_type": "Polygon",
                "hole_assignments": [
                    {
                        "exterior_path_id": "exterior:main",
                        "hole_path_id": "hole:mark",
                    }
                ],
                "hole_count": 1,
                "ring_count": 2,
                "topology_valid": True,
                "validity_reason": "Valid Geometry",
            },
        )

    def test_disjoint_components_and_diagnostics_are_order_independent(self):
        paths = (
            _path(
                "hole:right",
                "hole",
                ((22.0, 2.0), (24.0, 2.0), (24.0, 4.0), (22.0, 4.0)),
            ),
            _path(
                "exterior:right",
                "exterior",
                ((20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0)),
            ),
            _path(
                "exterior:left",
                "exterior",
                ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
            ),
        )
        first = validate_outline_topology(_outline(*paths)).to_dict()
        second = validate_outline_topology(_outline(*reversed(paths))).to_dict()

        self.assertEqual(first, second)
        self.assertEqual(first["geometry_type"], "MultiPolygon")
        self.assertEqual(first["component_count"], 2)
        self.assertEqual(first["component_exterior_path_ids"], [
            "exterior:left",
            "exterior:right",
        ])
        self.assertEqual(first["component_hole_counts"], [0, 1])
        self.assertEqual(first["area_mm2"], 196.0)


class TestRingContract(unittest.TestCase):
    def test_requires_outline_payload(self):
        cutline = VectorGeometryPayload(
            schema_version=VECTOR_PAYLOAD_SCHEMA_VERSION,
            kind=VectorRecordKind.CUTLINE,
            coordinate_space=VECTOR_COORDINATE_SPACE,
            frame=_frame(),
            paths=(
                _path(
                    "section:one",
                    "section",
                    ((0.0, 0.0), (1.0, 0.0)),
                    closed=False,
                ),
            ),
        )

        with self.assertRaises(ArtifactOutlineTopologyError) as raised:
            validate_outline_topology(cutline)
        self.assertEqual(raised.exception.code, "invalid_payload_kind")

    def test_open_or_unknown_role_ring_is_rejected_defensively(self):
        exterior = _path(
            "exterior:main",
            "exterior",
            ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0)),
            closed=False,
        )
        with self.assertRaises(ArtifactOutlineTopologyError) as raised:
            validate_outline_topology(_unchecked_outline(exterior))
        self.assertEqual(raised.exception.code, "open_outline_ring")

        unknown = _path(
            "ring:unknown",
            "island",
            ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0)),
        )
        with self.assertRaises(ArtifactOutlineTopologyError) as raised:
            validate_outline_topology(_unchecked_outline(unknown))
        self.assertEqual(raised.exception.code, "invalid_ring_role")

    def test_zero_area_and_self_intersection_are_distinguished(self):
        collinear = _path(
            "exterior:flat",
            "exterior",
            ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
        )
        with self.assertRaises(ArtifactOutlineTopologyError) as raised:
            validate_outline_topology(_unchecked_outline(collinear))
        self.assertEqual(raised.exception.code, "ring_zero_area")

        self_touching = _path(
            "exterior:self-touching",
            "exterior",
            ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 0.0), (0.0, 4.0)),
        )
        payload = _outline(self_touching)
        with self.assertRaises(ArtifactOutlineTopologyError) as raised:
            validate_outline_topology(payload)
        self.assertEqual(raised.exception.code, "ring_not_simple")
        self.assertEqual(raised.exception.path_ids, ("exterior:self-touching",))


class TestHoleTopology(unittest.TestCase):
    def setUp(self) -> None:
        self.exterior = _path(
            "exterior:main",
            "exterior",
            ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
        )

    def test_hole_must_be_strictly_inside_one_exterior(self):
        outside = _path(
            "hole:outside",
            "hole",
            ((20.0, 20.0), (22.0, 20.0), (22.0, 22.0), (20.0, 22.0)),
        )
        with self.assertRaises(ArtifactOutlineTopologyError) as raised:
            validate_outline_topology(_outline(self.exterior, outside))
        self.assertEqual(raised.exception.code, "hole_without_exterior")
        self.assertEqual(
            raised.exception.diagnostics,
            {
                "code": "hole_without_exterior",
                "path_ids": ["hole:outside"],
                "reason": "a hole must be strictly inside exactly one exterior",
                "topology_valid": False,
            },
        )

        touching = _path(
            "hole:touching",
            "hole",
            ((0.0, 2.0), (2.0, 2.0), (2.0, 4.0), (0.0, 4.0)),
        )
        with self.assertRaises(ArtifactOutlineTopologyError) as raised:
            validate_outline_topology(_outline(self.exterior, touching))
        self.assertEqual(raised.exception.code, "hole_not_strictly_inside")
        self.assertEqual(
            raised.exception.path_ids,
            ("exterior:main", "hole:touching"),
        )

    def test_hole_cannot_belong_to_nested_exteriors(self):
        outer = _path(
            "exterior:outer",
            "exterior",
            ((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)),
        )
        inner = _path(
            "exterior:inner",
            "exterior",
            ((2.0, 2.0), (18.0, 2.0), (18.0, 18.0), (2.0, 18.0)),
        )
        hole = _path(
            "hole:ambiguous",
            "hole",
            ((4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)),
        )
        with self.assertRaises(ArtifactOutlineTopologyError) as raised:
            validate_outline_topology(_outline(outer, inner, hole))
        self.assertEqual(raised.exception.code, "hole_multiple_exteriors")
        self.assertEqual(
            raised.exception.path_ids,
            ("exterior:inner", "exterior:outer", "hole:ambiguous"),
        )

    def test_holes_must_not_overlap_or_touch(self):
        first = _path(
            "hole:first",
            "hole",
            ((2.0, 2.0), (5.0, 2.0), (5.0, 5.0), (2.0, 5.0)),
        )
        overlap = _path(
            "hole:overlap",
            "hole",
            ((4.0, 4.0), (7.0, 4.0), (7.0, 7.0), (4.0, 7.0)),
        )
        with self.assertRaises(ArtifactOutlineTopologyError) as raised:
            validate_outline_topology(_outline(self.exterior, first, overlap))
        self.assertEqual(raised.exception.code, "holes_overlap")

        touching = _path(
            "hole:touching",
            "hole",
            ((5.0, 2.0), (7.0, 2.0), (7.0, 4.0), (5.0, 4.0)),
        )
        with self.assertRaises(ArtifactOutlineTopologyError) as raised:
            validate_outline_topology(_outline(self.exterior, first, touching))
        self.assertEqual(raised.exception.code, "holes_touch")


class TestExteriorTopology(unittest.TestCase):
    def test_exteriors_must_not_overlap_nest_or_touch(self):
        first = _path(
            "exterior:first",
            "exterior",
            ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)),
        )
        overlap = _path(
            "exterior:overlap",
            "exterior",
            ((2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0)),
        )
        with self.assertRaises(ArtifactOutlineTopologyError) as raised:
            validate_outline_topology(_outline(first, overlap))
        self.assertEqual(raised.exception.code, "exteriors_overlap")

        nested = _path(
            "exterior:nested",
            "exterior",
            ((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)),
        )
        with self.assertRaises(ArtifactOutlineTopologyError) as raised:
            validate_outline_topology(_outline(first, nested))
        self.assertEqual(raised.exception.code, "exteriors_overlap")

        touching = _path(
            "exterior:touching",
            "exterior",
            ((4.0, 0.0), (8.0, 0.0), (8.0, 4.0), (4.0, 4.0)),
        )
        with self.assertRaises(ArtifactOutlineTopologyError) as raised:
            validate_outline_topology(_outline(first, touching))
        self.assertEqual(raised.exception.code, "exteriors_touch")


if __name__ == "__main__":
    unittest.main()
