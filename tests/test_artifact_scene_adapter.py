from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import ast
import unittest

import numpy as np

import src.core.artifact_scene_adapter as adapter_module
from src.core.artifact_document import (
    ARTIFACT_DOCUMENT_SCHEMA_VERSION,
    GEOMETRY_HASH_SCOPE_V1,
    AlignRevision,
    ArtifactDocument,
    GeometryRevision,
    Handedness,
    MetadataConfirmationStatus,
    SourceAsset,
    SourceMetadataRevision,
)
from src.core.artifact_scene_adapter import (
    ArtifactSceneAdapter,
    GeometryBindingError,
    SourceBindingError,
    StaleProjectionContextError,
    VerifiedGeometryIdentity,
)
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


STAMP = "2026-07-11T00:00:00Z"
SOURCE_SHA = "a" * 64
GEOMETRY_SHA = "b" * 64
SOURCE_ASSET_ID = f"sha256:{SOURCE_SHA}"

SOURCE_VERTICES_CM = np.asarray(
    [
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [10.0, 0.0, 10.0],
    ],
    dtype=np.float64,
)
SOURCE_FACES = np.asarray(
    [[0, 1, 2], [0, 1, 3]],
    dtype=np.int32,
)
EXPECTED_WORLD_VERTICES_MM = np.asarray(
    [
        [105.0, -2.0, 3.0],
        [205.0, -2.0, 3.0],
        [105.0, 98.0, 3.0],
        [105.0, -2.0, 103.0],
    ],
    dtype=np.float64,
)
EXPECTED_SOURCE_TO_WORLD_MM = np.asarray(
    [
        [10.0, 0.0, 0.0, 5.0],
        [0.0, 10.0, 0.0, -2.0],
        [0.0, 0.0, 10.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _fingerprint(
    *,
    sha256: str = SOURCE_SHA,
    size_bytes: int = 123,
    original_name: str = "artifact.ply",
) -> SourceFingerprint:
    return SourceFingerprint(
        sha256=sha256,
        size_bytes=size_bytes,
        mtime_ns=1,
        original_name=original_name,
        format="ply",
    )


def _source_mesh(
    *,
    fingerprint: SourceFingerprint | None = None,
) -> MeshData:
    return MeshData(
        vertices=SOURCE_VERTICES_CM.copy(),
        faces=SOURCE_FACES.copy(),
        normals=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        face_normals=np.asarray(
            [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        ),
        uv_coords=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float64,
        ),
        texture=np.asarray([[[10, 20, 30]]], dtype=np.uint8),
        unit="cm",
        filepath=Path("/relocated/artifact.ply"),
        source_identity=_fingerprint() if fingerprint is None else fingerprint,
        source_format="ply",
    )


def _source_asset() -> SourceAsset:
    return SourceAsset.from_fingerprint(
        _fingerprint(),
        asset_ref="external:artifact.ply",
        media_type="model/ply",
    )


def _geometry() -> GeometryRevision:
    return GeometryRevision(
        id="geometry:g1",
        source_asset_ids=(SOURCE_ASSET_ID,),
        geometry_sha256=GEOMETRY_SHA,
        geometry_hash_scope=GEOMETRY_HASH_SCOPE_V1,
        import_recipe={"format": "ply", "process": False},
        topology_map_ref=None,
        qc={"finite_vertices": True},
        created_at=STAMP,
        operator="tester",
    )


def _metadata(*, confirmed: bool = True) -> SourceMetadataRevision:
    return SourceMetadataRevision(
        id="metadata:m1",
        parent_id=None,
        geometry_revision_id="geometry:g1",
        unit="cm" if confirmed else "unknown",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness=Handedness.RIGHT if confirmed else Handedness.UNKNOWN,
        confirmation_status=(
            MetadataConfirmationStatus.CONFIRMED
            if confirmed
            else MetadataConfirmationStatus.UNCONFIRMED
        ),
        source_to_canonical_mm=(
            (
                (10.0, 0.0, 0.0, 0.0),
                (0.0, 10.0, 0.0, 0.0),
                (0.0, 0.0, 10.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
            if confirmed
            else (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
        created_at=STAMP,
        operator="tester",
    )


def _left_handed_metadata() -> SourceMetadataRevision:
    return replace(
        _metadata(),
        axes={"source_x": "-X", "source_y": "+Y", "source_z": "+Z"},
        handedness=Handedness.LEFT,
        source_to_canonical_mm=(
            (-10.0, 0.0, 0.0, 0.0),
            (0.0, 10.0, 0.0, 0.0),
            (0.0, 0.0, 10.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _align(
    align_id: str = "align:a1",
    *,
    parent_id: str | None = None,
    translation: tuple[float, float, float] = (5.0, -2.0, 3.0),
) -> AlignRevision:
    tx, ty, tz = translation
    return AlignRevision(
        id=align_id,
        parent_id=parent_id,
        source_metadata_revision_id="metadata:m1",
        matrix4x4=(
            (1.0, 0.0, 0.0, tx),
            (0.0, 1.0, 0.0, ty),
            (0.0, 0.0, 1.0, tz),
            (0.0, 0.0, 0.0, 1.0),
        ),
        recipe={"kind": "manual_matrix"},
        qc={"rigid": True},
        created_at=STAMP,
        operator="tester",
    )


def _document(*, confirmed_metadata: bool = True) -> ArtifactDocument:
    return ArtifactDocument(
        schema_version=ARTIFACT_DOCUMENT_SCHEMA_VERSION,
        document_id="artifact:test-scene-adapter",
        software_version="0.1.0",
        source_assets=(_source_asset(),),
        geometry_revisions=(_geometry(),),
        source_metadata_revisions=(_metadata(confirmed=confirmed_metadata),),
        align_revisions=(_align(),),
        active_source_metadata_revision_id="metadata:m1",
        active_align_revision_id="align:a1",
        records=(),
    )


def _verified_identity(**changes: str) -> VerifiedGeometryIdentity:
    values = {
        "source_asset_id": SOURCE_ASSET_ID,
        "geometry_revision_id": "geometry:g1",
        "geometry_sha256": GEOMETRY_SHA,
        "geometry_hash_scope": GEOMETRY_HASH_SCOPE_V1,
    }
    values.update(changes)
    return VerifiedGeometryIdentity(**values)


class TestArtifactSceneAdapter(unittest.TestCase):
    def test_left_handed_metadata_reverses_winding_without_changing_face_ids(self):
        document = replace(
            _document(),
            source_metadata_revisions=(_left_handed_metadata(),),
        )

        projection = ArtifactSceneAdapter(document).materialize(
            _source_mesh(),
            _verified_identity(),
        )

        np.testing.assert_array_equal(
            projection.mesh.faces,
            SOURCE_FACES[:, [0, 2, 1]],
        )
        np.testing.assert_array_equal(_source_mesh().faces, SOURCE_FACES)

    def test_materializes_exact_align_after_cm_to_mm_without_recentering(self):
        document = _document()
        source = _source_mesh()
        assert source.uv_coords is not None
        assert source.texture is not None
        source_vertices_before = source.vertices.copy()
        source_faces_before = source.faces.copy()
        source_uv_before = source.uv_coords.copy()
        source_texture_before = source.texture.copy()

        projection = ArtifactSceneAdapter(document).materialize(
            source,
            _verified_identity(),
        )

        np.testing.assert_array_equal(
            projection.mesh.vertices,
            EXPECTED_WORLD_VERTICES_MM,
        )
        np.testing.assert_array_equal(
            projection.snapshot.matrix,
            EXPECTED_SOURCE_TO_WORLD_MM,
        )
        np.testing.assert_array_equal(
            projection.mesh.centroid,
            np.asarray([130.0, 23.0, 28.0]),
        )
        self.assertFalse(np.allclose(projection.mesh.centroid, 0.0))
        self.assertEqual(projection.mesh.unit, "mm")
        self.assertEqual(projection.snapshot.document_id, document.document_id)
        self.assertEqual(
            projection.snapshot.document_schema_version,
            ARTIFACT_DOCUMENT_SCHEMA_VERSION,
        )
        self.assertEqual(projection.snapshot.document_sha256, document.canonical_sha256)
        self.assertEqual(projection.snapshot.source_asset_id, SOURCE_ASSET_ID)
        self.assertEqual(projection.snapshot.geometry_revision_id, "geometry:g1")
        self.assertEqual(
            projection.snapshot.source_metadata_revision_id,
            "metadata:m1",
        )
        self.assertEqual(projection.snapshot.align_revision_id, "align:a1")

        np.testing.assert_array_equal(source.vertices, source_vertices_before)
        np.testing.assert_array_equal(source.faces, source_faces_before)
        np.testing.assert_array_equal(source.uv_coords, source_uv_before)
        np.testing.assert_array_equal(source.texture, source_texture_before)
        self.assertFalse(np.shares_memory(projection.mesh.vertices, source.vertices))
        self.assertFalse(np.shares_memory(projection.mesh.faces, source.faces))
        self.assertFalse(np.shares_memory(projection.mesh.uv_coords, source.uv_coords))
        self.assertFalse(np.shares_memory(projection.mesh.texture, source.texture))
        self.assertIsNone(projection.mesh.normals)
        self.assertIsNone(projection.mesh.face_normals)

    def test_large_align_translation_preserves_submillimeter_world_features(self):
        source = MeshData(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.0125, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.1],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            unit="cm",
            filepath=Path("/relocated/large-coordinate-artifact.ply"),
            source_identity=_fingerprint(),
            source_format="ply",
        )
        source_vertices_before = source.vertices.copy()
        source_faces_before = source.faces.copy()
        translation_mm = np.asarray(
            [1_000_000_000.0, -1_000_000_000.0, 500_000_000.0],
            dtype=np.float64,
        )
        document = replace(
            _document(),
            align_revisions=(
                _align(
                    translation=(
                        float(translation_mm[0]),
                        float(translation_mm[1]),
                        float(translation_mm[2]),
                    )
                ),
            ),
        )

        projection = ArtifactSceneAdapter(document).materialize(
            source,
            _verified_identity(),
        )

        canonical_features_mm = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.125, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        expected_world_mm = canonical_features_mm + translation_mm
        self.assertEqual(projection.mesh.vertices.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(projection.mesh.vertices, expected_world_mm)
        np.testing.assert_array_equal(
            projection.mesh.vertices - translation_mm,
            canonical_features_mm,
        )
        self.assertEqual(
            float(projection.mesh.vertices[1, 0] - projection.mesh.vertices[0, 0]),
            0.125,
        )
        self.assertEqual(
            float(projection.mesh.vertices[2, 1] - projection.mesh.vertices[0, 1]),
            1.0,
        )
        np.testing.assert_array_equal(
            projection.snapshot.matrix[:3, 3],
            translation_mm,
        )
        np.testing.assert_array_equal(source.vertices, source_vertices_before)
        np.testing.assert_array_equal(source.faces, source_faces_before)
        self.assertFalse(np.shares_memory(projection.mesh.vertices, source.vertices))
        self.assertFalse(np.shares_memory(projection.mesh.faces, source.faces))

    def test_materialization_is_deterministic_and_never_chains_projected_state(self):
        adapter = ArtifactSceneAdapter(_document())
        source = _source_mesh()
        first = adapter.materialize(source, _verified_identity())
        second = adapter.materialize(source, _verified_identity())

        np.testing.assert_array_equal(first.mesh.vertices, second.mesh.vertices)
        np.testing.assert_array_equal(first.mesh.faces, second.mesh.faces)
        self.assertEqual(first.snapshot, second.snapshot)
        self.assertIsNot(first.mesh, second.mesh)
        self.assertFalse(np.shares_memory(first.mesh.vertices, second.mesh.vertices))

        first.mesh.vertices[:] = -999.0
        first.mesh.faces[:] = 0
        assert first.mesh.uv_coords is not None
        assert first.mesh.texture is not None
        first.mesh.uv_coords[:] = -1.0
        first.mesh.texture[:] = 0

        third = adapter.materialize(source, _verified_identity())
        np.testing.assert_array_equal(third.mesh.vertices, EXPECTED_WORLD_VERTICES_MM)
        np.testing.assert_array_equal(third.mesh.faces, SOURCE_FACES)
        np.testing.assert_array_equal(source.vertices, SOURCE_VERTICES_CM)
        np.testing.assert_array_equal(source.faces, SOURCE_FACES)

    def test_rejects_stale_snapshot_after_document_or_active_context_changes(self):
        document = _document()
        verified = _verified_identity()
        snapshot = ArtifactSceneAdapter(document).capture_snapshot(verified)
        source = _source_mesh()

        changed_align = document.append_align_revision(
            _align(
                "align:a2",
                parent_id="align:a1",
                translation=(9.0, 0.0, 0.0),
            )
        )
        changed_metadata_only = replace(document, active_align_revision_id=None)
        changed_document = replace(document, software_version="0.1.1")

        with self.subTest("active Align changed"):
            with self.assertRaisesRegex(
                StaleProjectionContextError,
                "stale",
            ):
                ArtifactSceneAdapter(changed_align).materialize(
                    source,
                    verified,
                    expected_snapshot=snapshot,
                )

        with self.subTest("document canonical identity changed"):
            with self.assertRaisesRegex(
                StaleProjectionContextError,
                "stale",
            ):
                ArtifactSceneAdapter(changed_document).materialize(
                    source,
                    verified,
                    expected_snapshot=snapshot,
                )

        with self.subTest("active Align missing"):
            with self.assertRaisesRegex(GeometryBindingError, "active Align"):
                ArtifactSceneAdapter(changed_metadata_only).capture_snapshot(verified)

    def test_rejects_mismatched_or_missing_source_fingerprint(self):
        adapter = ArtifactSceneAdapter(_document())
        verified = _verified_identity()
        cases = {
            "wrong hash": _source_mesh(fingerprint=_fingerprint(sha256="c" * 64)),
            "wrong byte length": _source_mesh(
                fingerprint=_fingerprint(size_bytes=124)
            ),
            "missing fingerprint": _source_mesh(),
        }
        cases["missing fingerprint"].source_identity = None

        for label, source in cases.items():
            with self.subTest(label):
                with self.assertRaises(SourceBindingError):
                    adapter.materialize(source, verified)

    def test_accepts_relocated_source_when_content_identity_is_unchanged(self):
        relocated = _source_mesh(
            fingerprint=_fingerprint(original_name="renamed-without-extension")
        )

        projection = ArtifactSceneAdapter(_document()).materialize(
            relocated,
            _verified_identity(),
        )

        np.testing.assert_array_equal(
            projection.mesh.vertices,
            EXPECTED_WORLD_VERTICES_MM,
        )

    def test_rejects_each_geometry_identity_mismatch(self):
        adapter = ArtifactSceneAdapter(_document())
        source = _source_mesh()
        cases = {
            "source asset": _verified_identity(source_asset_id="sha256:" + "c" * 64),
            "geometry revision": _verified_identity(
                geometry_revision_id="geometry:missing"
            ),
            "geometry hash": _verified_identity(geometry_sha256="c" * 64),
            "hash scope": _verified_identity(
                geometry_hash_scope="positions-and-faces/unknown"
            ),
        }

        for label, verified in cases.items():
            with self.subTest(label):
                with self.assertRaises(GeometryBindingError):
                    adapter.materialize(source, verified)

    def test_unconfirmed_metadata_cannot_materialize_world_geometry(self):
        with self.assertRaisesRegex(GeometryBindingError, "not confirmed"):
            ArtifactSceneAdapter(_document(confirmed_metadata=False)).materialize(
                _source_mesh(),
                _verified_identity(),
            )

    def test_rejects_non_finite_source_coordinates(self):
        source = _source_mesh()
        source.vertices[0, 0] = np.nan

        with self.assertRaisesRegex(SourceBindingError, "finite"):
            ArtifactSceneAdapter(_document()).materialize(
                source,
                _verified_identity(),
            )

    def test_adapter_module_has_no_qt_opengl_or_gui_import(self):
        module_path = Path(adapter_module.__file__).resolve()
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
                imported_modules.add(node.module)

        self.assertTrue(imported_roots.isdisjoint({"PyQt5", "PyQt6", "OpenGL"}))
        self.assertFalse(any(name.startswith("src.gui") for name in imported_modules))


if __name__ == "__main__":
    unittest.main()
