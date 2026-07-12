from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.core.artifact_session import ArtifactSession, ArtifactSessionError
from src.core.mesh_loader import MeshData
from src.core.project_file import load_artifact_project, save_artifact_project
from src.core.source_identity import SourceFingerprint


STAMP = "2026-07-11T00:00:00Z"
SOURCE_SHA = "a" * 64


def _mesh() -> MeshData:
    return MeshData(
        vertices=np.array(
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [10.0, 10.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        unit="cm",
        filepath=Path("/source/artifact.ply"),
        source_identity=SourceFingerprint(
            sha256=SOURCE_SHA,
            size_bytes=123,
            mtime_ns=1,
            original_name="artifact.ply",
            format="ply",
        ),
        source_format="ply",
    )


def _session() -> ArtifactSession:
    return ArtifactSession.create_from_source(
        _mesh(),
        resolved_source_path="/source/artifact.ply",
        unit="cm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="0.1.0",
        operator="tester",
        created_at=STAMP,
        document_id="artifact:session-test",
        metadata_revision_id="metadata:m1",
        align_revision_id="align:a1",
    )


class TestArtifactSession(unittest.TestCase):
    def test_create_materialize_and_source_snapshot_are_canonical_and_immutable(self):
        original = _mesh()
        session = ArtifactSession.create_from_source(
            original,
            resolved_source_path="/source/artifact.ply",
            unit="cm",
            axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
            handedness="right",
            software_version="0.1.0",
            operator="tester",
            created_at=STAMP,
            document_id="artifact:create-test",
            metadata_revision_id="metadata:m1",
            align_revision_id="align:a1",
        )

        original.vertices[:] = -999.0
        projection = session.materialize()

        np.testing.assert_array_equal(
            projection.mesh.vertices,
            [[100.0, 0.0, 0.0], [200.0, 0.0, 0.0], [100.0, 100.0, 0.0]],
        )
        self.assertFalse(session.source_mesh.vertices.flags.writeable)
        self.assertFalse(session.source_mesh.faces.flags.writeable)
        self.assertEqual(projection.mesh.unit, "mm")

    def test_commit_and_parent_activation_are_drift_free(self):
        initial = _session()
        committed = initial.commit_preview(
            translation_mm=(5.0, -2.0, 3.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            operator="tester",
            created_at=STAMP,
            revision_id="align:a2",
        )

        np.testing.assert_array_equal(
            committed.materialize().mesh.vertices,
            [[105.0, -2.0, 3.0], [205.0, -2.0, 3.0], [105.0, 98.0, 3.0]],
        )
        restored = committed.activate_parent_align()
        np.testing.assert_array_equal(
            restored.materialize().mesh.vertices,
            initial.materialize().mesh.vertices,
        )
        recommitted = restored.activate_align("align:a2")
        np.testing.assert_array_equal(
            recommitted.materialize().mesh.vertices,
            committed.materialize().mesh.vertices,
        )
        np.testing.assert_array_equal(initial.source_mesh.vertices, _mesh().vertices)

    def test_preview_scale_is_rejected(self):
        with self.assertRaisesRegex(ArtifactSessionError, "cannot contain scale"):
            _session().commit_preview(
                translation_mm=(0.0, 0.0, 0.0),
                rotation_deg=(0.0, 0.0, 0.0),
                scale=2.0,
                operator="tester",
            )

    def test_preview_rotation_uses_explicit_world_mm_pivot(self):
        committed = _session().commit_preview(
            translation_mm=(5.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 90.0),
            scale=1.0,
            pivot_mm=(100.0, 0.0, 0.0),
            operator="tester",
            created_at=STAMP,
            revision_id="align:pivoted",
        )

        np.testing.assert_allclose(
            committed.materialize().mesh.vertices,
            [[105.0, 0.0, 0.0], [105.0, 100.0, 0.0], [5.0, 0.0, 0.0]],
            rtol=0.0,
            atol=1e-12,
        )
        self.assertEqual(
            committed.document.align_revision_index["align:pivoted"].recipe["pivot_mm"],
            (100.0, 0.0, 0.0),
        )

    def test_saved_document_rebinds_source_and_restores_exact_matrix_and_geometry(self):
        committed = _session().commit_preview(
            translation_mm=(5.0, -2.0, 3.0),
            rotation_deg=(0.0, 0.0, 90.0),
            scale=1.0,
            operator="tester",
            created_at=STAMP,
            revision_id="align:a2",
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.amr"
            save_artifact_project(path, committed.document)
            loaded = load_artifact_project(path)
            rebound = ArtifactSession.bind_loaded_document(
                loaded,
                _mesh(),
                resolved_source_path="/relocated/artifact.ply",
            )

        self.assertEqual(rebound.document.canonical_sha256, committed.document.canonical_sha256)
        np.testing.assert_array_equal(
            rebound.document.active_canonical_matrix(),
            committed.document.active_canonical_matrix(),
        )
        np.testing.assert_array_equal(
            rebound.materialize().mesh.vertices,
            committed.materialize().mesh.vertices,
        )
        rebound_identity = rebound.source_mesh.source_identity
        committed_identity = committed.source_mesh.source_identity
        self.assertIsNotNone(rebound_identity)
        self.assertIsNotNone(committed_identity)
        assert rebound_identity is not None
        assert committed_identity is not None
        self.assertEqual(rebound_identity.sha256, committed_identity.sha256)

    def test_rebinding_changed_decoded_geometry_is_rejected(self):
        changed = _mesh()
        changed.vertices[0, 0] += 0.5
        with self.assertRaisesRegex(ArtifactSessionError, "SHA-256"):
            ArtifactSession.bind_loaded_document(
                _session().document,
                changed,
                resolved_source_path="/source/artifact.ply",
            )

    def test_rebinding_same_geometry_from_different_source_bytes_is_rejected(self):
        changed = _mesh()
        changed.source_identity = SourceFingerprint(
            sha256="b" * 64,
            size_bytes=123,
            mtime_ns=2,
            original_name="lookalike.ply",
            format="ply",
        )

        with self.assertRaisesRegex(ArtifactSessionError, "source bytes"):
            ArtifactSession.bind_loaded_document(
                _session().document,
                changed,
                resolved_source_path="/relocated/lookalike.ply",
            )

    def test_rebinding_requires_saved_parser_format(self):
        changed = _mesh()
        changed.source_format = "obj"

        with self.assertRaisesRegex(ArtifactSessionError, "parser format"):
            ArtifactSession.bind_loaded_document(
                _session().document,
                changed,
                resolved_source_path="/relocated/artifact.dat",
            )

    def test_public_constructor_and_post_construction_mutation_cannot_bypass_hash(self):
        trusted = _session()
        changed = _mesh()
        changed.vertices[0, 0] += 0.5
        with self.assertRaisesRegex(ArtifactSessionError, "SHA-256"):
            ArtifactSession(
                document=trusted.document,
                source_mesh=changed,
                verified_geometry=trusted.verified_geometry,
                resolved_source_path="/source/artifact.ply",
            )

        replacement = trusted.source_mesh.vertices.copy()
        replacement[0, 0] += 0.5
        trusted.source_mesh.vertices = replacement
        with self.assertRaisesRegex(ArtifactSessionError, "SHA-256"):
            trusted.materialize()


if __name__ == "__main__":
    unittest.main()
