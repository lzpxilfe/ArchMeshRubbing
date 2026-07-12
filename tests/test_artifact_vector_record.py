from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.core.artifact_document import AlignRevision, RecordFreshness, canonical_recipe_hash
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_record import (
    ArtifactVectorRecordError,
    PlanarFrame,
    VECTOR_COORDINATE_SPACE,
    VECTOR_GEOMETRY_REF_PREFIX,
    VECTOR_PAYLOAD_EXTENSION_KEY,
    VECTOR_PAYLOAD_SCHEMA_VERSION,
    VectorGeometryPayload,
    VectorPath,
    VectorRecordKind,
    append_vector_record_from_context,
    payload_sha256_from_geometry_ref,
    vector_payload_from_record,
)
from src.core.mesh_loader import MeshData
from src.core.project_file import (
    ProjectSerializationError,
    load_artifact_project,
    save_artifact_project,
)
from src.core.source_identity import SourceFingerprint


STAMP = "2026-07-12T00:00:00Z"
RECIPE = {
    "algorithm": "trimesh.section",
    "algorithm_version": "4",
    "kind": "cutline",
    "plane_offset_mm": 12.5,
    "precision_mm": 0.1,
}


def _session() -> ArtifactSession:
    mesh = MeshData(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        unit="cm",
        filepath=Path("/source/vector.ply"),
        source_identity=SourceFingerprint(
            sha256="a" * 64,
            size_bytes=321,
            mtime_ns=1,
            original_name="vector.ply",
            format="ply",
        ),
        source_format="ply",
    )
    return ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/vector.ply",
        unit="cm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="test",
        operator="tester",
        created_at=STAMP,
        document_id="artifact:vector-record",
        metadata_revision_id="metadata:m1",
        align_revision_id="align:a1",
    )


def _frame() -> PlanarFrame:
    return PlanarFrame(
        origin_world_mm=(100.0, 200.0, 300.0),
        u_axis_world=(1.0, 0.0, 0.0),
        v_axis_world=(0.0, 1.0, 0.0),
        normal_world=(0.0, 0.0, 1.0),
    )


def _payload(kind: VectorRecordKind = VectorRecordKind.CUTLINE) -> VectorGeometryPayload:
    return VectorGeometryPayload(
        schema_version=VECTOR_PAYLOAD_SCHEMA_VERSION,
        kind=kind,
        coordinate_space=VECTOR_COORDINATE_SPACE,
        frame=_frame(),
        paths=(
            VectorPath(
                id="path:outer",
                role="section" if kind is VectorRecordKind.CUTLINE else "exterior",
                closed=True,
                points_mm=(
                    (-0.0, 0.0),
                    (20.0, 0.0),
                    (20.0, 10.0),
                    (20.0, 10.0),
                    (0.0, 10.0),
                    (-0.0, 0.0),
                ),
            ),
        ),
    )


class TestArtifactVectorPayload(unittest.TestCase):
    def test_payload_is_canonical_content_addressed_and_roundtrips(self):
        payload = _payload()
        restored = VectorGeometryPayload.from_dict(payload.to_dict())

        self.assertEqual(restored, payload)
        self.assertEqual(len(payload.paths[0].points_mm), 4)
        self.assertEqual(payload.paths[0].points_mm[0], (0.0, 0.0))
        self.assertNotIn(b"-0.0", payload.canonical_json_bytes())
        self.assertEqual(restored.canonical_json_bytes(), payload.canonical_json_bytes())
        self.assertEqual(
            payload.geometry_ref,
            f"{VECTOR_GEOMETRY_REF_PREFIX}{payload.sha256}",
        )
        self.assertEqual(
            payload.sha256,
            "38c42316c05e939ad4ddffc41faf26b0c7fa814802f85506cc51f5ca3fa74b8d",
        )
        self.assertEqual(
            payload_sha256_from_geometry_ref(payload.geometry_ref),
            payload.sha256,
        )
        self.assertEqual(
            payload.qc_summary(),
            {
                "bounds_mm": [0.0, 0.0, 20.0, 10.0],
                "closed_path_count": 1,
                "coordinate_space": VECTOR_COORDINATE_SPACE,
                "finite": True,
                "path_count": 1,
                "payload_sha256": payload.sha256,
                "point_count": 4,
                "total_length_mm": 60.0,
                "total_length_rounding_decimal_places": 12,
                "unit": "mm",
            },
        )

    def test_frame_and_path_validation_fail_closed(self):
        with self.assertRaisesRegex(ArtifactVectorRecordError, "right-handed"):
            replace(_frame(), normal_world=(0.0, 0.0, -1.0))
        with self.assertRaisesRegex(ArtifactVectorRecordError, "unit vector"):
            replace(_frame(), u_axis_world=(2.0, 0.0, 0.0))
        with self.assertRaisesRegex(ArtifactVectorRecordError, "at least 3"):
            VectorPath(
                id="bad",
                role="outer",
                closed=True,
                points_mm=((0.0, 0.0), (1.0, 0.0), (0.0, 0.0)),
            )
        with self.assertRaisesRegex(ArtifactVectorRecordError, "finite"):
            VectorPath(
                id="bad",
                role="line",
                closed=False,
                points_mm=((0.0, 0.0), (float("nan"), 1.0)),
            )
        with self.assertRaisesRegex(ArtifactVectorRecordError, "endpoints"):
            VectorPath(
                id="bad-open-loop",
                role="section",
                closed=False,
                points_mm=((0.0, 0.0), (1.0, 1.0), (0.0, 0.0)),
            )

    def test_parser_rejects_unknown_or_non_object_paths(self):
        data = _payload().to_dict()
        data["unknown"] = True
        with self.assertRaisesRegex(ArtifactVectorRecordError, "unknown fields"):
            VectorGeometryPayload.from_dict(data)

        data = _payload().to_dict()
        data["paths"] = ["not-an-object"]
        with self.assertRaisesRegex(ArtifactVectorRecordError, "only objects"):
            VectorGeometryPayload.from_dict(data)

    def test_parser_rejects_type_coercion_and_noncanonical_whitespace(self):
        for replacement in ("20.0", True):
            with self.subTest(replacement=replacement):
                data = _payload().to_dict()
                data["paths"][0]["points_mm"][1][0] = replacement
                with self.assertRaisesRegex(ArtifactVectorRecordError, "finite numbers"):
                    VectorGeometryPayload.from_dict(data)

        data = _payload().to_dict()
        data["paths"][0]["id"] = " path:outer "
        with self.assertRaisesRegex(ArtifactVectorRecordError, "surrounding whitespace"):
            VectorGeometryPayload.from_dict(data)

        data = _payload().to_dict()
        data["paths"][0]["points_mm"][1][0] = 20
        integer_lexical_form = VectorGeometryPayload.from_dict(data)
        self.assertEqual(integer_lexical_form, _payload())
        self.assertEqual(integer_lexical_form.sha256, _payload().sha256)

    def test_path_direction_start_and_order_are_canonicalized_before_hashing(self):
        base = _payload()
        reversed_rotated = replace(
            base,
            paths=(
                VectorPath(
                    id="path:outer",
                    role="section",
                    closed=True,
                    points_mm=((20.0, 10.0), (20.0, 0.0), (0.0, 0.0), (0.0, 10.0)),
                ),
            ),
        )
        self.assertEqual(reversed_rotated.paths, base.paths)
        self.assertEqual(reversed_rotated.sha256, base.sha256)

        first = VectorPath(
            id="path:a",
            role="section",
            closed=False,
            points_mm=((10.0, 0.0), (0.0, 0.0)),
        )
        second = VectorPath(
            id="path:b",
            role="section",
            closed=False,
            points_mm=((30.0, 0.0), (20.0, 0.0)),
        )
        forward = replace(base, paths=(first, second))
        shuffled = replace(base, paths=(second, first))
        self.assertEqual(forward.paths, shuffled.paths)
        self.assertEqual(forward.sha256, shuffled.sha256)
        self.assertEqual(forward.paths[0].points_mm[0], (0.0, 0.0))

    def test_outline_roles_closure_and_winding_are_controlled(self):
        outline = VectorGeometryPayload(
            schema_version=VECTOR_PAYLOAD_SCHEMA_VERSION,
            kind=VectorRecordKind.OUTLINE,
            coordinate_space=VECTOR_COORDINATE_SPACE,
            frame=_frame(),
            paths=(
                VectorPath(
                    id="hole:0",
                    role="hole",
                    closed=True,
                    points_mm=((2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)),
                ),
                VectorPath(
                    id="exterior:0",
                    role="exterior",
                    closed=True,
                    points_mm=((0.0, 10.0), (10.0, 10.0), (10.0, 0.0), (0.0, 0.0)),
                ),
            ),
        )
        self.assertEqual([path.role for path in outline.paths], ["exterior", "hole"])

        def signed_area(path: VectorPath) -> float:
            points = path.points_mm
            return 0.5 * sum(
                points[index][0] * points[(index + 1) % len(points)][1]
                - points[(index + 1) % len(points)][0] * points[index][1]
                for index in range(len(points))
            )

        self.assertGreater(signed_area(outline.paths[0]), 0.0)
        self.assertLess(signed_area(outline.paths[1]), 0.0)
        with self.assertRaisesRegex(ArtifactVectorRecordError, "must be closed"):
            replace(
                outline,
                paths=(
                    VectorPath(
                        id="bad",
                        role="exterior",
                        closed=False,
                        points_mm=((0.0, 0.0), (1.0, 0.0)),
                    ),
                ),
            )
        with self.assertRaisesRegex(ArtifactVectorRecordError, "roles"):
            replace(
                outline,
                paths=(
                    VectorPath(
                        id="bad",
                        role="outer",
                        closed=True,
                        points_mm=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
                    ),
                ),
            )


class TestArtifactVectorRecordCommand(unittest.TestCase):
    def test_session_command_commits_record_without_mutating_projection(self):
        session = _session()
        before = session.materialize().mesh.vertices.copy()
        context = session.capture_vector_operation(recipe=RECIPE)
        committed = session.commit_vector_record(
            context=context,
            payload=_payload(),
            recipe=RECIPE,
            record_id="record:session-cutline",
            created_at=STAMP,
            operator="tester",
        )

        np.testing.assert_array_equal(committed.materialize().mesh.vertices, before)
        np.testing.assert_array_equal(session.materialize().mesh.vertices, before)
        self.assertEqual(len(session.document.records), 0)
        self.assertEqual(len(committed.document.records), 1)
        self.assertEqual(
            committed.document.record_freshness("record:session-cutline"),
            RecordFreshness.FRESH,
        )

    def test_ready_record_binds_payload_recipe_qc_and_active_align(self):
        session = _session()
        context = session.document.capture_operation_context(recipe=RECIPE)
        payload = _payload()
        document = append_vector_record_from_context(
            session.document,
            context=context,
            payload=payload,
            recipe=RECIPE,
            record_id="record:cutline-0",
            created_at=STAMP,
            operator="tester",
            qc={"plane_residual_max_mm": 0.0},
        )

        record = document.record_index["record:cutline-0"]
        self.assertEqual(record.type, "vector.cutline.v1")
        self.assertEqual(record.geometry_ref, payload.geometry_ref)
        self.assertEqual(record.align_revision_id, "align:a1")
        self.assertEqual(record.qc["payload_sha256"], payload.sha256)
        self.assertEqual(record.qc["plane_residual_max_mm"], 0.0)
        self.assertEqual(
            record.extensions[VECTOR_PAYLOAD_EXTENSION_KEY]["byte_length"],
            len(payload.canonical_json_bytes()),
        )
        self.assertEqual(vector_payload_from_record(record), payload)
        self.assertEqual(document.record_freshness(record.id), RecordFreshness.FRESH)

        changed_session = session.with_document(document).commit_preview(
            translation_mm=(1.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            operator="tester",
            created_at=STAMP,
            revision_id="align:a2",
        )
        self.assertEqual(
            changed_session.document.record_freshness(record.id),
            RecordFreshness.STALE_ALIGNMENT,
        )

    def test_late_completion_remains_bound_to_captured_context(self):
        session = _session()
        outline_recipe = {**RECIPE, "kind": "outline"}
        context = session.document.capture_operation_context(recipe=outline_recipe)
        parent = session.document.align_revision_index["align:a1"]
        switched = session.document.append_align_revision(
            AlignRevision(
                id="align:a2",
                parent_id=parent.id,
                source_metadata_revision_id=parent.source_metadata_revision_id,
                matrix4x4=(
                    (1.0, 0.0, 0.0, 5.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                recipe={"kind": "test"},
                qc={"proper_rigid": True},
                created_at=STAMP,
                operator="tester",
            )
        )

        completed = append_vector_record_from_context(
            switched,
            context=context,
            payload=_payload(VectorRecordKind.OUTLINE),
            recipe=outline_recipe,
            record_id="record:late-outline",
            created_at=STAMP,
            operator="worker",
        )

        self.assertEqual(
            completed.record_index["record:late-outline"].align_revision_id,
            "align:a1",
        )
        self.assertEqual(
            completed.record_freshness("record:late-outline"),
            RecordFreshness.STALE_ALIGNMENT,
        )

    def test_recipe_or_computed_qc_cannot_be_relabelled(self):
        session = _session()
        context = session.document.capture_operation_context(recipe=RECIPE)
        with self.assertRaisesRegex(ArtifactVectorRecordError, "recipe"):
            append_vector_record_from_context(
                session.document,
                context=context,
                payload=_payload(),
                recipe={**RECIPE, "precision_mm": 1.0},
                record_id="record:wrong-recipe",
                created_at=STAMP,
                operator="tester",
            )
        with self.assertRaisesRegex(ArtifactVectorRecordError, "cannot override"):
            append_vector_record_from_context(
                session.document,
                context=context,
                payload=_payload(),
                recipe=RECIPE,
                record_id="record:wrong-qc",
                created_at=STAMP,
                operator="tester",
                qc={"point_count": 999},
            )

    def test_inline_payload_tampering_is_detected(self):
        session = _session()
        context = session.document.capture_operation_context(recipe=RECIPE)
        payload = _payload()
        document = append_vector_record_from_context(
            session.document,
            context=context,
            payload=payload,
            recipe=RECIPE,
            record_id="record:tamper",
            created_at=STAMP,
            operator="tester",
        )
        record = document.record_index["record:tamper"]
        descriptor = dict(record.extensions[VECTOR_PAYLOAD_EXTENSION_KEY])
        tampered_payload = payload.to_dict()
        tampered_payload["paths"][0]["points_mm"][1][0] = 21.0
        descriptor["payload"] = tampered_payload
        tampered = replace(
            record,
            extensions={VECTOR_PAYLOAD_EXTENSION_KEY: descriptor},
        )

        with self.assertRaisesRegex(ArtifactVectorRecordError, "SHA-256"):
            vector_payload_from_record(tampered)

        tampered_document = replace(document, records=(tampered,))
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tampered.amr"
            with self.assertRaisesRegex(ProjectSerializationError, "SHA-256"):
                save_artifact_project(path, tampered_document)

    def test_inline_payload_survives_artifact_project_roundtrip(self):
        session = _session()
        context = session.document.capture_operation_context(recipe=RECIPE)
        payload = _payload()
        document = append_vector_record_from_context(
            session.document,
            context=context,
            payload=payload,
            recipe=RECIPE,
            record_id="record:roundtrip",
            created_at=STAMP,
            operator="tester",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "vector.amr"
            save_artifact_project(path, document)
            loaded = load_artifact_project(path)

        loaded_record = loaded.record_index["record:roundtrip"]
        self.assertEqual(vector_payload_from_record(loaded_record), payload)
        self.assertEqual(loaded.canonical_sha256, document.canonical_sha256)

    def test_loaded_record_cannot_relabel_payload_derived_qc(self):
        session = _session()
        context = session.document.capture_operation_context(recipe=RECIPE)
        document = append_vector_record_from_context(
            session.document,
            context=context,
            payload=_payload(),
            recipe=RECIPE,
            record_id="record:forged-qc",
            created_at=STAMP,
            operator="tester",
        )
        record = document.record_index["record:forged-qc"]
        qc = dict(record.qc)
        qc["total_length_mm"] = 999.0
        forged = replace(record, qc=qc)

        with self.assertRaisesRegex(ArtifactVectorRecordError, "total_length_mm"):
            vector_payload_from_record(forged)

        forged_document = replace(document, records=(forged,))
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "forged-qc.amr"
            with self.assertRaisesRegex(ProjectSerializationError, "total_length_mm"):
                save_artifact_project(path, forged_document)

    def test_loaded_record_requires_reproducible_algorithm_recipe(self):
        session = _session()
        context = session.document.capture_operation_context(recipe=RECIPE)
        document = append_vector_record_from_context(
            session.document,
            context=context,
            payload=_payload(),
            recipe=RECIPE,
            record_id="record:missing-algorithm",
            created_at=STAMP,
            operator="tester",
        )
        record = document.record_index["record:missing-algorithm"]
        incomplete = {"kind": "cutline"}
        forged = replace(
            record,
            recipe=incomplete,
            recipe_hash=canonical_recipe_hash(incomplete),
        )

        with self.assertRaisesRegex(ArtifactVectorRecordError, "recipe.algorithm"):
            vector_payload_from_record(forged)

    def test_geometry_ref_parser_rejects_non_content_addressed_values(self):
        with self.assertRaisesRegex(ArtifactVectorRecordError, "not a vector"):
            payload_sha256_from_geometry_ref("payload:cutline.svg")
        with self.assertRaisesRegex(ArtifactVectorRecordError, "invalid SHA"):
            payload_sha256_from_geometry_ref(
                f"{VECTOR_GEOMETRY_REF_PREFIX}not-a-digest"
            )


if __name__ == "__main__":
    unittest.main()
