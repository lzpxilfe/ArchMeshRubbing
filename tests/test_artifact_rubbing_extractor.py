from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

import src.core.artifact_rubbing_extractor as rubbing_extractor
from src.core.artifact_document import RecordFreshness
from src.core.artifact_rubbing_extractor import (
    ArtifactRubbingError,
    RUBBING_ALGORITHM,
    commit_artifact_rubbing,
    compute_artifact_rubbing,
    compute_artifact_rubbing_from_recipe,
    extract_digital_rubbing,
    require_current_rubbing_computation,
    rubbing_computation_matches_active_projection,
    rubbing_recipe,
    validate_rubbing_recipe,
)
from src.core.artifact_rubbing_record import (
    RUBBING_RECORD_TYPE,
    rubbing_receipt_from_record,
)
from src.core.artifact_session import ArtifactSession
from src.core.mesh_loader import MeshData
from src.core.project_file import load_artifact_project, save_artifact_project
from src.core.source_identity import SourceFingerprint


STAMP = "2026-07-12T00:00:00Z"


def _recipe(
    view: str = "top",
    *,
    pixels_per_mm: int = 10,
    margin_um: int = 0,
    reference_radius_um: int = 500,
    depth_quantization_um: int = 10,
    black_point_um: int = 100,
    ink_strength_percent: int = 100,
    relief_polarity: str = "bidirectional",
) -> dict[str, Any]:
    return rubbing_recipe(
        view,
        pixels_per_mm=pixels_per_mm,
        margin_um=margin_um,
        reference_radius_um=reference_radius_um,
        depth_quantization_um=depth_quantization_um,
        black_point_um=black_point_um,
        ink_strength_percent=ink_strength_percent,
        relief_polarity=relief_polarity,
    )


def _plane() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )


def _bump() -> tuple[np.ndarray, np.ndarray]:
    vertices, _faces = _plane()
    return (
        np.vstack((vertices, np.array([[0.0, 0.0, 0.5]], dtype=np.float64))),
        np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int32),
    )


def _annulus() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [-2.0, -2.0, 0.0],
            [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0],
            [-2.0, 2.0, 0.0],
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
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


def _session(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    unit: str = "mm",
) -> ArtifactSession:
    mesh = MeshData(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int32),
        unit=unit,
        filepath=Path("/source/rubbing.ply"),
        source_identity=SourceFingerprint(
            sha256="f" * 64,
            size_bytes=1234,
            mtime_ns=1,
            original_name="rubbing.ply",
            format="ply",
        ),
        source_format="ply",
    )
    return ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/rubbing.ply",
        unit=unit,
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="0.6-test",
        operator="tester",
        created_at=STAMP,
        document_id="artifact:rubbing-extractor",
        metadata_revision_id="metadata:m1",
        align_revision_id="align:a1",
    )


class TestRubbingRecipe(unittest.TestCase):
    def test_recipe_resolves_physical_grid_and_integer_tone(self):
        recipe = rubbing_recipe(
            "front",
            pixels_per_mm=20,
            margin_um=1_250,
            reference_radius_um=2_250,
            depth_quantization_um=10,
            black_point_um=300,
            ink_strength_percent=150,
            relief_polarity="incised",
        )
        self.assertEqual(recipe["algorithm"], RUBBING_ALGORITHM)
        self.assertEqual(recipe["pixel_policy"]["pixels_per_meter"], 20_000)
        self.assertEqual(recipe["pixel_policy"]["margin_pixels"], 25)
        self.assertEqual(recipe["relief_policy"]["reference_radius_pixels"], 45)
        self.assertEqual(recipe["relief_policy"]["effective_black_point_um"], 200)
        self.assertEqual(recipe["relief_policy"]["effective_black_point_ticks"], 20)
        self.assertEqual(validate_rubbing_recipe(recipe), recipe)

    def test_recipe_rejects_coercion_unknown_polarity_and_tampering(self):
        with self.assertRaisesRegex(ArtifactRubbingError, "integer"):
            rubbing_recipe(
                "top",
                pixels_per_mm=True,
                margin_um=0,
                reference_radius_um=100,
                depth_quantization_um=10,
                black_point_um=100,
                ink_strength_percent=100,
                relief_polarity="raised",
            )
        with self.assertRaisesRegex(ArtifactRubbingError, "polarity"):
            _recipe(relief_polarity="auto")
        recipe = _recipe()
        recipe["pixel_policy"]["margin_pixels"] = 99
        with self.assertRaisesRegex(ArtifactRubbingError, "production contract"):
            validate_rubbing_recipe(recipe)


class TestRubbingRaster(unittest.TestCase):
    def test_flat_plane_has_exact_scale_mask_and_golden_semantic_hash(self):
        vertices, faces = _plane()
        raster, qc = extract_digital_rubbing(vertices, faces, _recipe())
        self.assertEqual(raster.pixels.shape, (20, 20, 2))
        self.assertTrue(np.all(raster.pixels[:, :, 0] == 255))
        self.assertTrue(np.all(raster.pixels[:, :, 1] == 255))
        self.assertEqual(raster.receipt()["width_mm_exact"], {
            "denominator": 10_000,
            "numerator": 20_000,
        })
        self.assertEqual(qc["covered_pixel_count"], 400)
        self.assertEqual(qc["sampling_applied"], False)
        self.assertEqual(
            raster.raw_pixel_sha256,
            "51e8e3057e7f4381071438308da8d8efb7df35b998764735822190bc21f0f8ed",
        )
        self.assertEqual(
            raster.raster_sha256,
            "6fdadfcca36c6655415f069aecf1b7b30c2f3378b9b44d89fd5dd0d8b96f1be7",
        )

    def test_known_bump_produces_ink_without_global_normalization(self):
        vertices, faces = _bump()
        raster, qc = extract_digital_rubbing(vertices, faces, _recipe())
        self.assertGreater(qc["inked_pixel_count"], 0)
        self.assertLess(qc["covered_gray_min"], 255)
        self.assertEqual(qc["depth_span_quantized_ticks"], 45)
        self.assertEqual(
            raster.raw_pixel_sha256,
            "e59856277df84789b5e1207aee2901e3a7c0beb2ea964a1971f23c8a6b692570",
        )

    def test_face_order_winding_duplicates_and_large_offset_preserve_pixels(self):
        vertices, faces = _bump()
        baseline, _qc = extract_digital_rubbing(vertices, faces, _recipe())
        variants = (faces[::-1], faces[:, ::-1], np.vstack((faces, faces)))
        for candidate_faces in variants:
            with self.subTest(face_count=len(candidate_faces)):
                candidate, _ = extract_digital_rubbing(
                    vertices,
                    candidate_faces,
                    _recipe(),
                )
                self.assertEqual(candidate.raw_pixel_sha256, baseline.raw_pixel_sha256)
        translated, _ = extract_digital_rubbing(
            vertices + np.array([1_000_000_000.0, -1_000_000_000.0, 0.0]),
            faces,
            _recipe(),
        )
        self.assertEqual(translated.raw_pixel_sha256, baseline.raw_pixel_sha256)
        self.assertNotEqual(translated.raster_sha256, baseline.raster_sha256)

    def test_hole_is_transparent_and_closed_box_reports_hidden_layer(self):
        vertices, faces = _annulus()
        raster, qc = extract_digital_rubbing(vertices, faces, _recipe())
        self.assertEqual(raster.pixels.shape, (40, 40, 2))
        self.assertEqual(int(raster.pixels[20, 20, 1]), 0)
        self.assertLess(qc["covered_pixel_count"], qc["pixel_count"])

        box = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
        _box_raster, box_qc = extract_digital_rubbing(
            box.vertices,
            box.faces,
            _recipe(),
        )
        self.assertEqual(box_qc["multi_layer_pixel_count"], 400)
        self.assertEqual(box_qc["maximum_second_layer_gap_um_rounded"], 1000)

    def test_invalid_empty_and_resource_exhaustion_fail_without_downsampling(self):
        vertices, faces = _plane()
        with patch.object(rubbing_extractor, "MAX_RUBBING_PIXELS", 100):
            with self.assertRaisesRegex(ArtifactRubbingError, "lower physical resolution"):
                extract_digital_rubbing(vertices, faces, _recipe())
        with patch.object(rubbing_extractor, "MAX_RUBBING_TRIANGLE_PIXEL_TESTS", 10):
            with self.assertRaisesRegex(ArtifactRubbingError, "triangle-pixel"):
                extract_digital_rubbing(vertices, faces, _recipe())
        with self.assertRaisesRegex(ValueError, "degenerate"):
            extract_digital_rubbing(
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                np.array([[0, 1, 2]], dtype=np.int32),
                _recipe(),
            )


class TestRubbingRecordWorkflow(unittest.TestCase):
    def test_cm_source_is_converted_once_committed_saved_and_recomputed(self):
        box = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
        session = _session(box.vertices, box.faces, unit="cm")
        source_before = session.source_mesh.vertices.copy()
        computation = compute_artifact_rubbing(
            session,
            "top",
            pixels_per_mm=2,
            margin_um=0,
            reference_radius_um=1000,
            depth_quantization_um=10,
            black_point_um=250,
            ink_strength_percent=100,
            relief_polarity="bidirectional",
        )
        self.assertEqual(computation.raster.pixels.shape, (40, 40, 2))
        np.testing.assert_array_equal(session.source_mesh.vertices, source_before)
        committed = commit_artifact_rubbing(
            session,
            computation,
            record_id="record:rubbing:top",
            created_at=STAMP,
            operator="tester",
        )
        record = committed.document.record_index["record:rubbing:top"]
        self.assertEqual(record.type, RUBBING_RECORD_TYPE)
        self.assertEqual(
            committed.document.record_freshness(record.id),
            RecordFreshness.FRESH,
        )
        receipt = rubbing_receipt_from_record(record)
        self.assertEqual(receipt["raster_sha256"], computation.raster.raster_sha256)

        with tempfile.TemporaryDirectory() as temporary:
            project_path = Path(temporary) / "rubbing.amr"
            save_artifact_project(project_path, committed.document)
            loaded_document = load_artifact_project(project_path)
        rebound = committed.with_document(loaded_document)
        recomputed = compute_artifact_rubbing_from_recipe(rebound, record.recipe)
        self.assertEqual(
            recomputed.raster.raster_sha256,
            receipt["raster_sha256"],
        )
        np.testing.assert_array_equal(recomputed.raster.pixels, computation.raster.pixels)

    def test_late_align_result_is_historical_and_never_current(self):
        vertices, faces = _bump()
        session = _session(vertices, faces)
        computation = compute_artifact_rubbing(
            session,
            "top",
            pixels_per_mm=10,
            margin_um=0,
            reference_radius_um=500,
            depth_quantization_um=10,
            black_point_um=100,
            ink_strength_percent=100,
            relief_polarity="bidirectional",
        )
        switched = session.commit_preview(
            translation_mm=(5.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            operator="tester",
            created_at=STAMP,
            revision_id="align:a2",
        )
        self.assertFalse(rubbing_computation_matches_active_projection(switched, computation))
        with self.assertRaisesRegex(ArtifactRubbingError, "stale"):
            require_current_rubbing_computation(switched, computation)
        committed = commit_artifact_rubbing(
            switched,
            computation,
            record_id="record:rubbing:late",
            created_at=STAMP,
            operator="worker",
        )
        self.assertEqual(
            committed.document.record_freshness("record:rubbing:late"),
            RecordFreshness.STALE_ALIGNMENT,
        )


if __name__ == "__main__":
    unittest.main()
