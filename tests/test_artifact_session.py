from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from src.core.artifact_document import ArtifactDocument
from src.core.artifact_session import ArtifactSession, ArtifactSessionError
from src.core.mesh_import_recipe import (
    MESH_IMPORT_RECIPE_ID,
    current_mesh_import_recipe,
)
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
        source_import_recipe=current_mesh_import_recipe("ply"),
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
    def test_new_session_records_closed_parser_and_runtime_recipe(self):
        session = _session()
        geometry = session.document.geometry_revisions[0]
        recipe = dict(geometry.import_recipe)

        self.assertEqual(
            recipe["recipe_id"],
            MESH_IMPORT_RECIPE_ID,
        )
        self.assertEqual(recipe["recipe_version"], "1.0.0")
        self.assertEqual(recipe["loader"], "trimesh")
        self.assertEqual(recipe["loader_version"], "4.11.5")
        self.assertEqual(len(recipe["runtime_lock_sha256"]), 64)
        self.assertIs(recipe["process"], False)
        self.assertIs(recipe["maintain_order"], True)
        self.assertEqual(recipe["force"], "mesh")

    def test_rebind_accepts_exact_legacy_profile_but_rejects_recipe_drift(self):
        session = _session()
        legacy_data = session.document.to_dict()
        legacy_data["geometry_revisions"][0]["import_recipe"] = {
            "format": "ply",
            "loader": "trimesh",
            "maintain_order": True,
            "process": False,
            "sanitizer": "meshdata-v1",
        }
        legacy_document = ArtifactDocument.from_dict(legacy_data)
        legacy_mesh = _mesh()
        legacy_mesh.source_import_recipe = dict(
            legacy_data["geometry_revisions"][0]["import_recipe"]
        )
        rebound = ArtifactSession.bind_loaded_document(
            legacy_document,
            legacy_mesh,
            resolved_source_path="/relocated/artifact.ply",
        )
        self.assertEqual(rebound.document, legacy_document)

        # Runtime identity is no longer compared against the installed parser,
        # but the loaded source receipt must still agree with the document's
        # recipe field for field, so drift is still caught here.
        for field, value, message in (
            ("loader_version", "0.0.0", "does not match"),
            ("parser_runtime_sha256", "0" * 64, "does not match"),
            ("process", True, "process"),
        ):
            changed = session.document.to_dict()
            changed["geometry_revisions"][0]["import_recipe"][field] = value
            changed_document = ArtifactDocument.from_dict(changed)
            with self.assertRaisesRegex(ArtifactSessionError, message):
                ArtifactSession.bind_loaded_document(
                    changed_document,
                    _mesh(),
                    resolved_source_path="/relocated/artifact.ply",
                )

    def test_document_identity_does_not_depend_on_native_source_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_path = root / "site-a" / "artifact.ply"
            second_path = root / "site-b" / "artifact.ply"

            def create(source_path: Path) -> ArtifactSession:
                return ArtifactSession.create_from_source(
                    _mesh(),
                    resolved_source_path=str(source_path),
                    unit="cm",
                    axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
                    handedness="right",
                    software_version="0.1.0",
                    operator="tester",
                    created_at=STAMP,
                    document_id="artifact:path-independent",
                    metadata_revision_id="metadata:m1",
                    align_revision_id="align:a1",
                )

            first = create(first_path)
            second = create(second_path)

        self.assertNotEqual(first.resolved_source_path, second.resolved_source_path)
        self.assertEqual(first.document.source_assets[0].asset_ref, "external:artifact.ply")
        self.assertEqual(second.document.source_assets[0].asset_ref, "external:artifact.ply")
        self.assertNotIn(str(root).encode("utf-8"), first.document.canonical_json_bytes())
        self.assertNotIn(str(root).encode("utf-8"), second.document.canonical_json_bytes())
        self.assertEqual(
            first.document.canonical_json_bytes(),
            second.document.canonical_json_bytes(),
        )
        self.assertEqual(first.document.canonical_sha256, second.document.canonical_sha256)

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

    def test_upgrading_the_parser_does_not_orphan_a_project_already_saved(self):
        """A newer Trimesh must not lock a researcher out of their own data.

        The saved recipe here names a parser version this runtime does not
        have, which is exactly what an app upgrade produces.  Reopening is
        allowed because it proves the parser by reproducing the recorded
        geometry, not because the version strings agree.
        """

        saved_recipe = current_mesh_import_recipe("ply")
        saved_recipe["loader_version"] = "99.0.0"

        def mesh_saved_under_another_parser() -> MeshData:
            mesh = _mesh()
            mesh.source_import_recipe = dict(saved_recipe)
            return mesh

        committed = ArtifactSession.create_from_source(
            mesh_saved_under_another_parser(),
            resolved_source_path="/source/artifact.ply",
            unit="cm",
            axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
            handedness="right",
            software_version="0.1.0",
            operator="tester",
            created_at=STAMP,
            document_id="artifact:reopen-across-runtimes",
            metadata_revision_id="metadata:m1",
            align_revision_id="align:a1",
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.amr"
            save_artifact_project(path, committed.document)
            loaded = load_artifact_project(path)
            rebound = ArtifactSession.bind_loaded_document(
                loaded,
                mesh_saved_under_another_parser(),
                resolved_source_path="/relocated/artifact.ply",
            )

        self.assertEqual(
            rebound.document.canonical_sha256,
            committed.document.canonical_sha256,
        )
        np.testing.assert_array_equal(
            rebound.materialize().mesh.vertices,
            committed.materialize().mesh.vertices,
        )
        # The difference is reported rather than hidden.
        self.assertEqual(rebound.reopened_under_runtime, trimesh.__version__)

    def test_a_parser_that_decodes_different_geometry_is_still_refused(self):
        """Relaxing the version check must not relax the actual proof."""

        saved_recipe = current_mesh_import_recipe("ply")
        saved_recipe["loader_version"] = "99.0.0"
        mesh = _mesh()
        mesh.source_import_recipe = dict(saved_recipe)

        committed = ArtifactSession.create_from_source(
            mesh,
            resolved_source_path="/source/artifact.ply",
            unit="cm",
            axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
            handedness="right",
            software_version="0.1.0",
            operator="tester",
            created_at=STAMP,
            document_id="artifact:reopen-geometry-drift",
            metadata_revision_id="metadata:m1",
            align_revision_id="align:a1",
        )

        drifted = _mesh()
        drifted.source_import_recipe = dict(saved_recipe)
        drifted.vertices = np.array(
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [10.0, 10.000001, 0.0]],
            dtype=np.float64,
        )

        with self.assertRaisesRegex(ArtifactSessionError, "SHA-256 mismatch"):
            ArtifactSession.bind_loaded_document(
                committed.document,
                drifted,
                resolved_source_path="/relocated/artifact.ply",
            ).projection_snapshot()

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
