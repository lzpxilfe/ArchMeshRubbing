from __future__ import annotations

import importlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.core.artifact_document import ArtifactDocument
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_export import build_vector_export
from src.core.artifact_vector_record import (
    PlanarFrame,
    VECTOR_COORDINATE_SPACE,
    VECTOR_PAYLOAD_SCHEMA_VERSION,
    VectorGeometryPayload,
    VectorPath,
    VectorRecordKind,
)
from src.core.mesh_loader import MeshData
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.source_identity import SourceFingerprint


ROOT = Path(__file__).resolve().parents[1]


def _document_and_payload() -> tuple[ArtifactDocument, VectorGeometryPayload]:
    mesh = MeshData(
        vertices=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        unit="mm",
        source_identity=SourceFingerprint(
            sha256="e" * 64,
            size_bytes=10,
            mtime_ns=1,
            original_name="schema.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/schema.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="schema-test",
        operator="tester",
        created_at="2026-07-12T00:00:00Z",
        document_id="artifact:schema-test",
        metadata_revision_id="metadata:schema-test",
        align_revision_id="align:schema-test",
    )
    payload = VectorGeometryPayload(
        schema_version=VECTOR_PAYLOAD_SCHEMA_VERSION,
        kind=VectorRecordKind.CUTLINE,
        coordinate_space=VECTOR_COORDINATE_SPACE,
        frame=PlanarFrame(
            origin_world_mm=(0.0, 0.0, 0.0),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 1.0, 0.0),
            normal_world=(0.0, 0.0, 1.0),
        ),
        paths=(
            VectorPath(
                id="cutline:path:0000",
                role="section",
                closed=True,
                points_mm=((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)),
            ),
        ),
    )
    recipe = {
        "algorithm": "schema-test",
        "algorithm_version": "1.0.0",
        "kind": "cutline",
    }
    context = session.capture_vector_operation(recipe=recipe)
    document = session.commit_vector_record(
        context=context,
        payload=payload,
        recipe=recipe,
        record_id="record:schema-test",
        created_at="2026-07-12T00:00:00Z",
        operator="tester",
    ).document
    return document, payload


class TestVectorSchemas(unittest.TestCase):
    def test_payload_and_export_sidecar_validate_against_versioned_schemas(self):
        jsonschema = importlib.import_module("jsonschema")
        referencing = importlib.import_module("referencing")
        payload_schema = json.loads(
            (ROOT / "schemas/vector_payload-1.0.0.schema.json").read_text(
                encoding="utf-8"
            )
        )
        export_schema = json.loads(
            (ROOT / "schemas/vector_export-1.0.0.schema.json").read_text(
                encoding="utf-8"
            )
        )
        import_recipe_schema = json.loads(
            (ROOT / "schemas/mesh_import_recipe-1.0.0.schema.json").read_text(
                encoding="utf-8"
            )
        )
        import_recipe_v2_schema = json.loads(
            (ROOT / "schemas/mesh_import_recipe-2.0.0.schema.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema.Draft202012Validator.check_schema(payload_schema)
        jsonschema.Draft202012Validator.check_schema(export_schema)
        jsonschema.Draft202012Validator.check_schema(import_recipe_schema)
        jsonschema.Draft202012Validator.check_schema(import_recipe_v2_schema)
        document, payload = _document_and_payload()
        payload_validator = jsonschema.Draft202012Validator(payload_schema)
        self.assertEqual(list(payload_validator.iter_errors(payload.to_dict())), [])

        registry = referencing.Registry().with_resource(
            payload_schema["$id"],
            referencing.Resource.from_contents(payload_schema),
        )
        registry = registry.with_resource(
            import_recipe_schema["$id"],
            referencing.Resource.from_contents(import_recipe_schema),
        )
        registry = registry.with_resource(
            import_recipe_v2_schema["$id"],
            referencing.Resource.from_contents(import_recipe_v2_schema),
        )
        export_validator = jsonschema.Draft202012Validator(
            export_schema,
            registry=registry,
        )
        sidecar = json.loads(
            build_vector_export(document, "record:schema-test").sidecar_bytes
        )
        self.assertEqual(list(export_validator.iter_errors(sidecar)), [])
        self.assertEqual(
            sidecar["provenance"]["geometry_revision"]["import_recipe"],
            document.geometry_revisions[0].to_dict()["import_recipe"],
        )

        sidecar["provenance"].pop("dependency_closure")
        self.assertTrue(list(export_validator.iter_errors(sidecar)))

        tampered = json.loads(
            build_vector_export(document, "record:schema-test").sidecar_bytes
        )
        tampered["provenance"]["geometry_revision"]["import_recipe"][
            "dependency_policy"
        ] = "allow_external"
        self.assertTrue(list(export_validator.iter_errors(tampered)))

    def test_outline_role_constraints_are_machine_readable(self):
        jsonschema = importlib.import_module("jsonschema")
        schema = json.loads(
            (ROOT / "schemas/vector_payload-1.0.0.schema.json").read_text(
                encoding="utf-8"
            )
        )
        _document, payload = _document_and_payload()
        invalid = payload.to_dict()
        invalid["kind"] = "outline"
        validator = jsonschema.Draft202012Validator(schema)
        self.assertTrue(list(validator.iter_errors(invalid)))


if __name__ == "__main__":
    unittest.main()
