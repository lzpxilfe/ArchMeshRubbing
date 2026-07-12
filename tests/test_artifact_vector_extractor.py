from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import trimesh

from src.core.artifact_document import RecordFreshness
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_extractor import (
    ArtifactVectorExtractionError,
    commit_vector_computation,
    computation_matches_active_projection,
    compute_artifact_cutline,
    cutline_recipe,
    extract_cutline_geometry,
    require_current_computation,
)
from src.core.artifact_vector_record import PlanarFrame
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


STAMP = "2026-07-12T00:00:00Z"


def _top_frame() -> PlanarFrame:
    return PlanarFrame(
        origin_world_mm=(0.0, 0.0, 0.0),
        u_axis_world=(1.0, 0.0, 0.0),
        v_axis_world=(0.0, 1.0, 0.0),
        normal_world=(0.0, 0.0, 1.0),
    )


def _right_frame() -> PlanarFrame:
    return PlanarFrame(
        origin_world_mm=(0.0, 0.0, 0.0),
        u_axis_world=(0.0, 1.0, 0.0),
        v_axis_world=(0.0, 0.0, 1.0),
        normal_world=(1.0, 0.0, 0.0),
    )


def _box(extents: tuple[float, float, float] = (2.0, 2.0, 2.0)) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=extents)


def _session_cm_box() -> ArtifactSession:
    box = _box()
    mesh = MeshData(
        vertices=np.asarray(box.vertices, dtype=np.float64),
        faces=np.asarray(box.faces, dtype=np.int32),
        unit="cm",
        filepath=Path("/source/box.ply"),
        source_identity=SourceFingerprint(
            sha256="d" * 64,
            size_bytes=456,
            mtime_ns=1,
            original_name="box.ply",
            format="ply",
        ),
        source_format="ply",
    )
    return ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/box.ply",
        unit="cm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="0.4-test",
        operator="tester",
        created_at=STAMP,
        document_id="artifact:cutline-extractor",
        metadata_revision_id="metadata:m1",
        align_revision_id="align:a1",
    )


class TestCanonicalCutlineGeometry(unittest.TestCase):
    def test_exact_box_section_is_four_point_ccw_golden(self):
        box = _box()
        result = extract_cutline_geometry(box.vertices, box.faces, _top_frame())

        self.assertEqual(len(result.payload.paths), 1)
        path = result.payload.paths[0]
        self.assertTrue(path.closed)
        self.assertEqual(path.id, "cutline:path:0000")
        self.assertEqual(
            path.points_mm,
            ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),
        )
        self.assertEqual(result.qc["raw_segment_count"], 8)
        self.assertEqual(result.qc["collinear_point_removal_count"], 4)
        self.assertEqual(result.qc["max_plane_residual_mm"], 0.0)
        self.assertEqual(
            result.payload.sha256,
            "0253922427a4deab3069bffb68fb91359b2afb753a07d60370d332c61e9ce491",
        )

    def test_face_order_winding_and_multiple_components_are_deterministic(self):
        box = _box()
        baseline = extract_cutline_geometry(box.vertices, box.faces, _top_frame()).payload
        for seed in range(10):
            with self.subTest(seed=seed):
                faces = np.asarray(box.faces, dtype=np.int64).copy()
                np.random.default_rng(seed).shuffle(faces)
                faces = faces[:, ::-1]
                candidate = extract_cutline_geometry(
                    box.vertices,
                    faces,
                    _top_frame(),
                ).payload
                self.assertEqual(candidate.canonical_json_bytes(), baseline.canonical_json_bytes())
                self.assertEqual(candidate.sha256, baseline.sha256)

        left = _box()
        left.apply_translation((-3.0, 0.0, 0.0))
        right = _box()
        right.apply_translation((3.0, 0.0, 0.0))
        combined = left + right
        multi = extract_cutline_geometry(
            combined.vertices,
            combined.faces,
            _top_frame(),
        ).payload
        self.assertEqual(len(multi.paths), 2)
        self.assertEqual(
            multi.paths[0].points_mm,
            ((-4.0, -1.0), (-2.0, -1.0), (-2.0, 1.0), (-4.0, 1.0)),
        )
        self.assertEqual(
            multi.paths[1].points_mm,
            ((2.0, -1.0), (4.0, -1.0), (4.0, 1.0), (2.0, 1.0)),
        )

    def test_right_and_oblique_frames_preserve_plane_coordinates(self):
        box = _box((2.0, 4.0, 6.0))
        right = extract_cutline_geometry(box.vertices, box.faces, _right_frame()).payload
        self.assertEqual(right.qc_summary()["bounds_mm"], [-2.0, -3.0, 2.0, 3.0])
        for path in right.paths:
            for u, v in path.points_mm:
                world = (
                    np.asarray(right.frame.origin_world_mm)
                    + u * np.asarray(right.frame.u_axis_world)
                    + v * np.asarray(right.frame.v_axis_world)
                )
                self.assertAlmostEqual(float(world[0]), 0.0, places=12)

        normal = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        u_axis = np.array([1.0, -1.0, 0.0], dtype=np.float64)
        u_axis /= np.linalg.norm(u_axis)
        v_axis = np.cross(normal, u_axis)
        frame = PlanarFrame(
            origin_world_mm=(0.0, 0.0, 0.0),
            u_axis_world=(float(u_axis[0]), float(u_axis[1]), float(u_axis[2])),
            v_axis_world=(float(v_axis[0]), float(v_axis[1]), float(v_axis[2])),
            normal_world=(float(normal[0]), float(normal[1]), float(normal[2])),
        )
        oblique_box = _box()
        oblique = extract_cutline_geometry(
            oblique_box.vertices,
            oblique_box.faces,
            frame,
        ).payload
        self.assertGreaterEqual(len(oblique.paths[0].points_mm), 6)
        for u, v in oblique.paths[0].points_mm:
            world = u * u_axis + v * v_axis
            self.assertLess(abs(float(np.dot(world, normal))), 1e-12)


class TestCutlineAmbiguityPolicy(unittest.TestCase):
    def test_coplanar_on_edge_and_point_tangent_cases_fail_closed(self):
        with self.assertRaisesRegex(ArtifactVectorExtractionError, "coplanar"):
            extract_cutline_geometry(
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                np.array([[0, 1, 2]], dtype=np.int32),
                _top_frame(),
            )
        with self.assertRaisesRegex(ArtifactVectorExtractionError, "on-plane"):
            extract_cutline_geometry(
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
                np.array([[0, 1, 2]], dtype=np.int32),
                _top_frame(),
            )
        with self.assertRaisesRegex(ArtifactVectorExtractionError, "does not form"):
            extract_cutline_geometry(
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
                np.array([[0, 1, 2]], dtype=np.int32),
                _top_frame(),
            )

    def test_non_manifold_branch_and_coincident_segments_are_rejected(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 1.0],
                [2.0, 0.0, -1.0],
                [0.0, 2.0, 1.0],
                [0.0, 2.0, -1.0],
                [-2.0, 0.0, 1.0],
                [-2.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 3, 4], [0, 5, 6]], dtype=np.int32)
        with self.assertRaisesRegex(ArtifactVectorExtractionError, "branching"):
            extract_cutline_geometry(vertices, faces, _top_frame())

        triangle = np.array(
            [[0.0, 0.0, -1.0], [2.0, 0.0, 1.0], [0.0, 2.0, 1.0]],
            dtype=np.float64,
        )
        with self.assertRaisesRegex(ArtifactVectorExtractionError, "coincident"):
            extract_cutline_geometry(
                triangle,
                np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32),
                _top_frame(),
            )

    def test_input_types_degenerate_faces_and_tolerance_contract_are_strict(self):
        box = _box()
        with self.assertRaisesRegex(ArtifactVectorExtractionError, "integer indices"):
            extract_cutline_geometry(
                box.vertices,
                np.asarray(box.faces, dtype=np.float64),
                _top_frame(),
            )
        with self.assertRaisesRegex(ArtifactVectorExtractionError, "degenerate"):
            extract_cutline_geometry(
                np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 1.0]]),
                np.array([[0, 1, 2]], dtype=np.int32),
                _top_frame(),
            )
        with self.assertRaisesRegex(ArtifactVectorExtractionError, "at least"):
            cutline_recipe(
                _top_frame(),
                classification_tolerance_mm=1e-3,
                stitch_tolerance_mm=1e-4,
            )


class TestArtifactCutlineCommand(unittest.TestCase):
    def test_cm_source_is_materialized_to_mm_exactly_once_and_committed(self):
        session = _session_cm_box()
        source_before = session.source_mesh.vertices.copy()
        computation = compute_artifact_cutline(session, _top_frame())

        self.assertEqual(
            computation.payload.qc_summary()["bounds_mm"],
            [-10.0, -10.0, 10.0, 10.0],
        )
        np.testing.assert_array_equal(session.source_mesh.vertices, source_before)
        self.assertEqual(computation.recipe["coordinate_space"], "canonical_mm_planar/v1")
        self.assertEqual(computation.context.align_revision_id, "align:a1")
        self.assertTrue(computation_matches_active_projection(session, computation))

        committed = commit_vector_computation(
            session,
            computation,
            record_id="record:cutline-z0",
            created_at=STAMP,
            operator="tester",
        )
        self.assertEqual(
            committed.document.record_freshness("record:cutline-z0"),
            RecordFreshness.FRESH,
        )
        self.assertTrue(computation_matches_active_projection(committed, computation))
        np.testing.assert_array_equal(committed.source_mesh.vertices, source_before)

    def test_late_completion_stays_historical_and_never_projects_into_new_align(self):
        session = _session_cm_box()
        computation = compute_artifact_cutline(session, _top_frame())
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
            record_id="record:late-cutline",
            created_at=STAMP,
            operator="worker",
        )
        record = committed.document.record_index["record:late-cutline"]
        self.assertEqual(record.align_revision_id, "align:a1")
        self.assertEqual(
            committed.document.record_freshness(record.id),
            RecordFreshness.STALE_ALIGNMENT,
        )
        restored = committed.activate_align("align:a1")
        self.assertEqual(
            restored.document.record_freshness(record.id),
            RecordFreshness.FRESH,
        )


if __name__ == "__main__":
    unittest.main()
