from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

import src.core.artifact_outline_extractor as outline_extractor
from src.core.artifact_cancellation import ArtifactComputationCancelledError
from src.core.artifact_document import RecordFreshness
from src.core.artifact_outline_extractor import (
    OUTLINE_ALGORITHM,
    REQUIRED_GEOS_VERSION,
    REQUIRED_SHAPELY_VERSION,
    OutlineView,
    compute_artifact_outline,
    extract_outline_geometry,
    outline_frame,
    outline_recipe,
    validate_outline_record_contract,
)
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_export import (
    VECTOR_EXPORT_SVG_NAME,
    VectorSVGOptions,
    export_vector_package,
    validate_vector_export_package,
)
from src.core.artifact_vector_extractor import (
    ArtifactVectorExtractionError,
    commit_vector_computation,
    computation_matches_active_projection,
    require_current_computation,
)
from src.core.artifact_vector_record import (
    VectorGeometryPayload,
    VectorPath,
    vector_payload_from_record,
)
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


STAMP = "2026-07-12T00:00:00Z"
SVG_NS = "http://www.w3.org/2000/svg"


def _box(extents: tuple[float, float, float] = (2.0, 4.0, 6.0)) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=extents)


def _l_shape() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]],
        dtype=np.int32,
    )
    return vertices, faces


def _annulus() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [-3.0, -3.0, 0.0],
            [3.0, -3.0, 0.0],
            [3.0, 3.0, 0.0],
            [-3.0, 3.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _rectangle(
    minimum_x: float,
    minimum_y: float,
    maximum_x: float,
    maximum_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [minimum_x, minimum_y, 0.0],
            [maximum_x, minimum_y, 0.0],
            [maximum_x, maximum_y, 0.0],
            [minimum_x, maximum_y, 0.0],
        ],
        dtype=np.float64,
    )
    return vertices, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def _combine(
    meshes: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for mesh_vertices, mesh_faces in meshes:
        vertices.append(np.asarray(mesh_vertices, dtype=np.float64))
        faces.append(np.asarray(mesh_faces, dtype=np.int32) + offset)
        offset += mesh_vertices.shape[0]
    return np.vstack(vertices), np.vstack(faces)


def _session(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    unit: str = "mm",
    document_id: str = "artifact:outline-extractor",
) -> ArtifactSession:
    mesh = MeshData(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int32),
        unit=unit,
        filepath=Path("/source/outline.ply"),
        source_identity=SourceFingerprint(
            sha256="e" * 64,
            size_bytes=789,
            mtime_ns=1,
            original_name="outline.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    return ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/outline.ply",
        unit=unit,
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="0.5-test",
        operator="tester",
        created_at=STAMP,
        document_id=document_id,
        metadata_revision_id="metadata:m1",
        align_revision_id="align:a1",
    )


class TestOutlineFramesAndRecipe(unittest.TestCase):
    def test_six_frames_are_exact_and_right_handed(self):
        expected = {
            "top": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            "bottom": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
            "front": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
            "back": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
            "right": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
            "left": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)),
        }
        for view in OutlineView:
            with self.subTest(view=view.value):
                frame = outline_frame(view)
                self.assertEqual(
                    (frame.u_axis_world, frame.v_axis_world, frame.normal_world),
                    expected[view.value],
                )
                np.testing.assert_array_equal(
                    np.cross(frame.u_axis_world, frame.v_axis_world),
                    frame.normal_world,
                )

    def test_recipe_declares_backend_grid_and_no_approximation(self):
        recipe = outline_recipe("top", precision_grid_mm=0.02)
        self.assertEqual(recipe["algorithm"], OUTLINE_ALGORITHM)
        self.assertEqual(recipe["kind"], "outline")
        self.assertEqual(recipe["precision_grid_mm"], 0.02)
        self.assertEqual(recipe["sampling"], "none")
        self.assertEqual(recipe["simplification_tolerance_mm"], 0.0)
        self.assertEqual(recipe["backend"]["shapely_version"], REQUIRED_SHAPELY_VERSION)
        self.assertEqual(recipe["backend"]["geos_version"], REQUIRED_GEOS_VERSION)
        self.assertEqual(recipe["backend"]["normalized_grid_size"], 1.0)

    def test_backend_version_is_an_authoritative_compute_gate(self):
        with patch.object(outline_extractor.shapely, "__version__", "9.9.9"):
            with self.assertRaisesRegex(
                ArtifactVectorExtractionError, "requires Shapely"
            ):
                outline_recipe("top", precision_grid_mm=0.01)

    def test_production_record_contract_rejects_recipe_id_and_grid_tampering(self):
        vertices, faces = _l_shape()
        result = extract_outline_geometry(
            vertices,
            faces,
            "top",
            precision_grid_mm=0.01,
        )
        recipe = outline_recipe("top", precision_grid_mm=0.01)
        validate_outline_record_contract(result.payload, recipe)

        with self.assertRaisesRegex(ArtifactVectorExtractionError, "recipe"):
            validate_outline_record_contract(
                result.payload,
                {**recipe, "sampling": "pixels"},
            )

        source_path = result.payload.paths[0]
        off_grid_points = list(source_path.points_mm)
        off_grid_points[0] = (off_grid_points[0][0] + 0.003, off_grid_points[0][1])
        off_grid = VectorGeometryPayload(
            schema_version=result.payload.schema_version,
            kind=result.payload.kind,
            coordinate_space=result.payload.coordinate_space,
            frame=result.payload.frame,
            paths=(
                VectorPath(
                    id=source_path.id,
                    role=source_path.role,
                    closed=True,
                    points_mm=tuple(off_grid_points),
                ),
            ),
        )
        with self.assertRaisesRegex(ArtifactVectorExtractionError, "off-grid"):
            validate_outline_record_contract(off_grid, recipe)

        wrong_id = VectorGeometryPayload(
            schema_version=result.payload.schema_version,
            kind=result.payload.kind,
            coordinate_space=result.payload.coordinate_space,
            frame=result.payload.frame,
            paths=(
                VectorPath(
                    id="outline:wrong-id",
                    role=source_path.role,
                    closed=True,
                    points_mm=source_path.points_mm,
                ),
            ),
        )
        with self.assertRaisesRegex(ArtifactVectorExtractionError, "exterior IDs"):
            validate_outline_record_contract(wrong_id, recipe)


class TestExactOutlineGeometry(unittest.TestCase):
    def test_box_bounds_are_correct_in_all_six_views(self):
        box = _box()
        expected = {
            "top": [-1.0, -2.0, 1.0, 2.0],
            "bottom": [-1.0, -2.0, 1.0, 2.0],
            "front": [-1.0, -3.0, 1.0, 3.0],
            "back": [-1.0, -3.0, 1.0, 3.0],
            "right": [-2.0, -3.0, 2.0, 3.0],
            "left": [-2.0, -3.0, 2.0, 3.0],
        }
        for view in OutlineView:
            with self.subTest(view=view.value):
                result = extract_outline_geometry(
                    box.vertices,
                    box.faces,
                    view,
                    precision_grid_mm=0.01,
                )
                self.assertEqual(
                    result.payload.qc_summary()["bounds_mm"], expected[view.value]
                )
                self.assertEqual(len(result.payload.paths), 1)
                self.assertEqual(result.payload.paths[0].role, "exterior")
                self.assertEqual(result.qc["projected_degenerate_triangle_count"], 8)

    def test_l_shape_retains_concavity_instead_of_convex_hull(self):
        vertices, faces = _l_shape()
        result = extract_outline_geometry(
            vertices,
            faces,
            "top",
            precision_grid_mm=0.01,
        )
        self.assertEqual(result.qc["outline_area_mm2"], 5.0)
        self.assertEqual(len(result.payload.paths[0].points_mm), 6)
        self.assertIn((1.0, 1.0), result.payload.paths[0].points_mm)
        self.assertEqual(result.qc["component_count"], 1)

    def test_hole_and_disconnected_islands_are_all_preserved(self):
        vertices, faces = _annulus()
        annulus = extract_outline_geometry(
            vertices,
            faces,
            "top",
            precision_grid_mm=0.01,
        )
        self.assertEqual(
            [path.role for path in annulus.payload.paths], ["exterior", "hole"]
        )
        self.assertEqual(annulus.qc["hole_count"], 1)
        self.assertEqual(annulus.qc["outline_area_mm2"], 32.0)
        self.assertEqual(
            annulus.payload.sha256,
            "b46217eaa9021e203d0999bf0c4a5e75cc8af7c9cc97fd3f5bcfc60e77ac1f3d",
        )

        islands_vertices, islands_faces = _combine(
            (_rectangle(-4.0, -1.0, -2.0, 1.0), _rectangle(2.0, -1.0, 4.0, 1.0))
        )
        islands = extract_outline_geometry(
            islands_vertices,
            islands_faces,
            "top",
            precision_grid_mm=0.01,
        )
        self.assertEqual(islands.qc["component_count"], 2)
        self.assertEqual(
            [path.id for path in islands.payload.paths],
            [
                "outline:component:0000:exterior",
                "outline:component:0001:exterior",
            ],
        )

    def test_face_order_winding_and_duplicates_do_not_change_payload(self):
        vertices, faces = _annulus()
        baseline = extract_outline_geometry(
            vertices,
            faces,
            "top",
            precision_grid_mm=0.01,
        ).payload
        for seed in range(8):
            with self.subTest(seed=seed):
                candidate_faces = faces.copy()
                np.random.default_rng(seed).shuffle(candidate_faces)
                candidate_faces = np.vstack(
                    (candidate_faces[:, ::-1], candidate_faces[:2])
                )
                candidate = extract_outline_geometry(
                    vertices,
                    candidate_faces,
                    "top",
                    precision_grid_mm=0.01,
                ).payload
                self.assertEqual(
                    candidate.canonical_json_bytes(), baseline.canonical_json_bytes()
                )

    def test_face_chunks_are_union_reduced_before_balanced_merge(self):
        vertices, faces = _annulus()
        baseline = extract_outline_geometry(
            vertices,
            faces,
            "top",
            precision_grid_mm=0.01,
        ).payload
        with patch.object(outline_extractor, "OUTLINE_UNION_BATCH_SIZE", 2):
            chunked = extract_outline_geometry(
                vertices,
                faces,
                "top",
                precision_grid_mm=0.01,
            )
        self.assertEqual(chunked.qc["face_chunk_count"], 4)
        self.assertEqual(
            chunked.payload.canonical_json_bytes(), baseline.canonical_json_bytes()
        )

    def test_cancellation_signal_is_not_wrapped_as_an_extraction_error(self):
        vertices, faces = _l_shape()

        with self.assertRaises(ArtifactComputationCancelledError):
            extract_outline_geometry(
                vertices,
                faces,
                "top",
                precision_grid_mm=0.01,
                cancellation_probe=lambda: True,
            )

    def test_cancellation_interrupts_the_polygon_precision_loop(self):
        vertices, base_faces = _rectangle(0.0, 0.0, 1.0, 1.0)
        faces = np.repeat(base_faces[:1], 600, axis=0)
        set_precision_calls = 0
        actual_set_precision = outline_extractor.set_precision

        def counted_set_precision(*args, **kwargs):
            nonlocal set_precision_calls
            set_precision_calls += 1
            return actual_set_precision(*args, **kwargs)

        with patch.object(
            outline_extractor,
            "set_precision",
            side_effect=counted_set_precision,
        ):
            with self.assertRaises(ArtifactComputationCancelledError):
                extract_outline_geometry(
                    vertices,
                    faces,
                    "top",
                    precision_grid_mm=0.01,
                    cancellation_probe=lambda: set_precision_calls > 0,
                )

        self.assertGreater(set_precision_calls, 0)
        self.assertLess(set_precision_calls, len(faces))

    def test_cancellation_after_projection_stops_before_referenced_vertex_scan(self):
        vertices, faces = _annulus()
        cancellation_requested = False
        actual_column_stack = outline_extractor.np.column_stack

        def project_then_cancel(*args, **kwargs):
            nonlocal cancellation_requested
            result = actual_column_stack(*args, **kwargs)
            cancellation_requested = True
            return result

        with (
            patch.object(
                outline_extractor.np,
                "column_stack",
                side_effect=project_then_cancel,
            ),
            patch.object(
                outline_extractor.np,
                "unique",
                wraps=outline_extractor.np.unique,
            ) as unique,
        ):
            with self.assertRaises(ArtifactComputationCancelledError):
                extract_outline_geometry(
                    vertices,
                    faces,
                    "top",
                    precision_grid_mm=0.01,
                    cancellation_probe=lambda: cancellation_requested,
                )

        self.assertTrue(cancellation_requested)
        unique.assert_not_called()

    def test_false_cancellation_probe_preserves_canonical_result(self):
        vertices, faces = _annulus()
        baseline = extract_outline_geometry(
            vertices,
            faces,
            "top",
            precision_grid_mm=0.01,
        )
        probe_calls = 0

        def keep_running() -> bool:
            nonlocal probe_calls
            probe_calls += 1
            return False

        candidate = extract_outline_geometry(
            vertices,
            faces,
            "top",
            precision_grid_mm=0.01,
            cancellation_probe=keep_running,
        )

        self.assertGreater(probe_calls, 0)
        self.assertEqual(
            candidate.payload.canonical_json_bytes(),
            baseline.payload.canonical_json_bytes(),
        )
        self.assertEqual(candidate.qc, baseline.qc)

    def test_large_survey_offset_uses_translated_integer_lattice(self):
        vertices, faces = _l_shape()
        translated = vertices + np.array([1_000_000_000.0, -1_000_000_000.0, 0.0])
        result = extract_outline_geometry(
            translated,
            faces,
            "top",
            precision_grid_mm=0.01,
        )
        self.assertEqual(result.qc["outline_area_mm2"], 5.0)
        self.assertEqual(
            result.qc["grid_origin_index_uv"], [100_000_000_000, -100_000_000_000]
        )
        self.assertEqual(
            result.payload.qc_summary()["bounds_mm"],
            [
                1_000_000_000.0,
                -1_000_000_000.0,
                1_000_000_003.0,
                -999_999_997.0,
            ],
        )


class TestGridAndSafetyPolicy(unittest.TestCase):
    def test_grid_collapse_and_component_merge_are_explicit_qc(self):
        tiny_vertices = np.array(
            [[5.0, 0.0, 0.0], [5.04, 0.0, 0.0], [5.0, 0.04, 0.0]],
            dtype=np.float64,
        )
        tiny_faces = np.array([[0, 1, 2]], dtype=np.int32)
        vertices, faces = _combine(
            (
                _rectangle(0.0, 0.0, 1.0, 1.0),
                _rectangle(1.04, 0.0, 2.04, 1.0),
                (tiny_vertices, tiny_faces),
            )
        )
        result = extract_outline_geometry(
            vertices,
            faces,
            "top",
            precision_grid_mm=0.1,
        )
        self.assertEqual(result.qc["grid_collapsed_triangle_count"], 1)
        self.assertEqual(result.qc["unsnapped_component_count"], 3)
        self.assertEqual(result.qc["component_count"], 1)
        self.assertEqual(result.qc["grid_component_merge_count"], 2)
        # The current algorithm closes the lattice union by one cell, so its
        # error bound is the union's half cell plus the closing radius plus
        # the re-snap.
        self.assertEqual(
            result.qc["grid_snap_error_contract"],
            "axis<=1.5*grid;radial<=1.5*grid*sqrt(2)",
        )
        self.assertAlmostEqual(result.qc["grid_snap_axis_upper_bound_mm"], 0.15)
        self.assertAlmostEqual(
            result.qc["grid_snap_radial_upper_bound_squared_mm2"], 0.045
        )
        self.assertEqual(result.qc["grid_closing_radius_cells"], 1.0)
        # A record made before the closing recomputes under the old contract.
        legacy = extract_outline_geometry(
            vertices,
            faces,
            "top",
            precision_grid_mm=0.1,
            algorithm_version="1.0.0",
        )
        self.assertEqual(
            legacy.qc["grid_snap_error_contract"],
            "axis<=grid/2;radial<=grid/sqrt(2)",
        )
        self.assertEqual(legacy.qc["grid_snap_axis_upper_bound_mm"], 0.05)
        self.assertEqual(
            legacy.qc["grid_snap_radial_upper_bound_squared_mm2"],
            0.005000000000000001,
        )
        self.assertNotIn("grid_closing_radius_cells", legacy.qc)
        with self.assertRaisesRegex(ArtifactVectorExtractionError, "algorithm_version"):
            extract_outline_geometry(
                vertices, faces, "top", precision_grid_mm=0.1, algorithm_version="2.0.0"
            )

    def test_empty_invalid_grid_and_limits_fail_without_fallback(self):
        vertical_vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        with self.assertRaisesRegex(ArtifactVectorExtractionError, "no non-degenerate"):
            extract_outline_geometry(
                vertical_vertices,
                np.array([[0, 1, 2]], dtype=np.int32),
                "top",
                precision_grid_mm=0.01,
            )

        vertices, faces = _rectangle(0.0, 0.0, 1.0, 1.0)
        for invalid in (0.0, -1.0, float("nan"), True):
            with self.subTest(grid=invalid):
                with self.assertRaisesRegex(
                    ArtifactVectorExtractionError, "greater than zero"
                ):
                    extract_outline_geometry(
                        vertices,
                        faces,
                        "top",
                        precision_grid_mm=invalid,
                    )
        with patch.object(outline_extractor, "MAX_OUTLINE_FACES", 1):
            with self.assertRaisesRegex(
                ArtifactVectorExtractionError, "face safety limit"
            ):
                extract_outline_geometry(
                    vertices,
                    faces,
                    "top",
                    precision_grid_mm=0.01,
                )
        with patch.object(outline_extractor, "MAX_GRID_INDEX", 10):
            with self.assertRaisesRegex(
                ArtifactVectorExtractionError, "integer safety range"
            ):
                extract_outline_geometry(
                    vertices + np.array([100.0, 0.0, 0.0]),
                    faces,
                    "top",
                    precision_grid_mm=0.01,
                )
        islands_vertices, islands_faces = _combine(
            (_rectangle(0.0, 0.0, 1.0, 1.0), _rectangle(3.0, 0.0, 4.0, 1.0))
        )
        with patch.object(outline_extractor, "MAX_OUTLINE_INTERMEDIATE_POLYGONS", 1):
            with self.assertRaisesRegex(
                ArtifactVectorExtractionError, "polygon safety limit"
            ):
                extract_outline_geometry(
                    islands_vertices,
                    islands_faces,
                    "top",
                    precision_grid_mm=0.01,
                )

    def test_all_faces_collapsing_at_grid_is_a_typed_failure(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.0, 0.04, 0.0]],
            dtype=np.float64,
        )
        with self.assertRaisesRegex(ArtifactVectorExtractionError, "all projected"):
            extract_outline_geometry(
                vertices,
                np.array([[0, 1, 2]], dtype=np.int32),
                "top",
                precision_grid_mm=0.1,
            )


class TestArtifactOutlineCommandAndExport(unittest.TestCase):
    def test_compute_cancellation_during_final_computation_cannot_escape(self):
        box = _box((2.0, 2.0, 2.0))
        session = _session(box.vertices, box.faces)
        cancellation_requested = False
        actual_computation_type = outline_extractor.ArtifactVectorComputation

        def computation_then_cancel(*args, **kwargs):
            nonlocal cancellation_requested
            computation = actual_computation_type(*args, **kwargs)
            cancellation_requested = True
            return computation

        with patch.object(
            outline_extractor,
            "ArtifactVectorComputation",
            side_effect=computation_then_cancel,
        ):
            with self.assertRaises(ArtifactComputationCancelledError):
                compute_artifact_outline(
                    session,
                    "top",
                    precision_grid_mm=0.01,
                    cancellation_probe=lambda: cancellation_requested,
                )

        self.assertTrue(cancellation_requested)

    def test_cm_source_is_converted_once_committed_and_revalidated(self):
        box = _box((2.0, 2.0, 2.0))
        session = _session(box.vertices, box.faces, unit="cm")
        source_before = session.source_mesh.vertices.copy()
        computation = compute_artifact_outline(
            session,
            "top",
            precision_grid_mm=0.01,
        )
        self.assertEqual(
            computation.payload.qc_summary()["bounds_mm"],
            [-10.0, -10.0, 10.0, 10.0],
        )
        np.testing.assert_array_equal(session.source_mesh.vertices, source_before)
        committed = commit_vector_computation(
            session,
            computation,
            record_id="record:outline:top",
            created_at=STAMP,
            operator="tester",
        )
        record = committed.document.record_index["record:outline:top"]
        self.assertEqual(record.type, "vector.outline.v1")
        self.assertEqual(
            committed.document.record_freshness(record.id),
            RecordFreshness.FRESH,
        )
        self.assertTrue(record.qc["outline_topology"]["topology_valid"])
        self.assertEqual(vector_payload_from_record(record), computation.payload)

    def test_late_align_result_stays_historical_and_is_not_display_current(self):
        box = _box((2.0, 2.0, 2.0))
        session = _session(box.vertices, box.faces)
        computation = compute_artifact_outline(
            session,
            "right",
            precision_grid_mm=0.01,
        )
        switched = session.commit_preview(
            translation_mm=(5.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            operator="tester",
            created_at=STAMP,
            revision_id="align:a2",
        )
        self.assertFalse(computation_matches_active_projection(switched, computation))
        with self.assertRaisesRegex(ArtifactVectorExtractionError, "stale"):
            require_current_computation(switched, computation)
        committed = commit_vector_computation(
            switched,
            computation,
            record_id="record:outline:late",
            created_at=STAMP,
            operator="worker",
        )
        self.assertEqual(
            committed.document.record_freshness("record:outline:late"),
            RecordFreshness.STALE_ALIGNMENT,
        )

    def test_hole_outline_exports_as_offline_valid_one_to_one_svg(self):
        vertices, faces = _annulus()
        session = _session(vertices, faces)
        computation = compute_artifact_outline(
            session,
            "top",
            precision_grid_mm=0.01,
        )
        committed = commit_vector_computation(
            session,
            computation,
            record_id="record:outline:annulus",
            created_at=STAMP,
            operator="tester",
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "src.core.artifact_vector_export._fsync_parent",
            return_value=True,
        ):
            destination = Path(temporary) / "annulus.amr-vector"
            package = export_vector_package(
                destination,
                committed.document,
                "record:outline:annulus",
                options=VectorSVGOptions(stroke_width_mm=0.2, margin_mm=0.1),
            )
            validate_vector_export_package(package)
            root = ET.fromstring((package / VECTOR_EXPORT_SVG_NAME).read_bytes())
            self.assertEqual(root.attrib["width"], "6.2mm")
            self.assertEqual(root.attrib["height"], "6.2mm")
            self.assertEqual(len(root.findall(f".//{{{SVG_NS}}}path")), 2)


if __name__ == "__main__":
    unittest.main()
