from __future__ import annotations

import copy
import importlib
import hashlib
import json
from pathlib import Path
from typing import Any
import unittest

import numpy as np
import pytest

import src.core.artifact_vector_export as vector_export
from src.core.artifact_document import ArtifactDocument
from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_export import (
    ArtifactVectorExportError,
    build_vector_export,
    validate_vector_export_bytes,
)
from src.core.artifact_vector_extractor import (
    commit_vector_computation,
    compute_artifact_cutline,
)
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


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _rebound_vector_export(sidecar: dict[str, Any]) -> tuple[bytes, bytes]:
    payload = vector_export._validated_sidecar_payload(sidecar)
    claims_sha256 = vector_export._sidecar_claims_sha256(sidecar)
    options, _presentation = vector_export._options_from_presentation(
        sidecar["presentation"]
    )
    svg_bytes, _bounds, _width, _height = vector_export._render_svg(
        payload,
        options=options,
        provenance=sidecar["provenance"],
        sidecar_claims_sha256=claims_sha256,
    )
    sidecar["artifact"]["sha256"] = hashlib.sha256(svg_bytes).hexdigest()
    sidecar["artifact"]["size_bytes"] = len(svg_bytes)
    return svg_bytes, _canonical_json(sidecar)


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


def _production_vector_session() -> ArtifactSession:
    vertices = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=np.int32,
    )
    mesh = MeshData(
        vertices=vertices,
        faces=faces,
        unit="mm",
        source_identity=SourceFingerprint(
            sha256="f" * 64,
            size_bytes=64,
            mtime_ns=1,
            original_name="production-schema.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/production-schema.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="schema-test",
        operator="tester",
        created_at="2026-07-12T00:00:00Z",
        document_id="artifact:production-schema-test",
        metadata_revision_id="metadata:production-schema-test",
        align_revision_id="align:production-schema-test",
    )
    frame = PlanarFrame(
        origin_world_mm=(0.0, 0.0, 0.0),
        u_axis_world=(1.0, 0.0, 0.0),
        v_axis_world=(0.0, 1.0, 0.0),
        normal_world=(0.0, 0.0, 1.0),
    )
    cutline = compute_artifact_cutline(session, frame)
    session = commit_vector_computation(
        session,
        cutline,
        record_id="record:production-cutline",
        created_at="2026-07-12T00:00:01Z",
        operator="tester",
    )
    outline = compute_artifact_outline(session, "top", precision_grid_mm=0.01)
    return commit_vector_computation(
        session,
        outline,
        record_id="record:production-outline",
        created_at="2026-07-12T00:00:02Z",
        operator="tester",
    )


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
            (ROOT / "schemas/vector_export-1.3.0.schema.json").read_text(
                encoding="utf-8"
            )
        )
        previous_export_schema = json.loads(
            (ROOT / "schemas/vector_export-1.1.0.schema.json").read_text(
                encoding="utf-8"
            )
        )
        legacy_export_schema = json.loads(
            (ROOT / "schemas/vector_export-1.0.0.schema.json").read_text(
                encoding="utf-8"
            )
        )
        admission_schema = json.loads(
            (ROOT / "schemas/mesh_admission_receipt-1.0.0.schema.json").read_text(
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
        jsonschema.Draft202012Validator.check_schema(previous_export_schema)
        jsonschema.Draft202012Validator.check_schema(legacy_export_schema)
        jsonschema.Draft202012Validator.check_schema(admission_schema)
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
            legacy_export_schema["$id"],
            referencing.Resource.from_contents(legacy_export_schema),
        )
        registry = registry.with_resource(
            admission_schema["$id"],
            referencing.Resource.from_contents(admission_schema),
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
        self.assertEqual(sidecar["schema_version"], "1.3.0")
        self.assertIn(
            "import_admission",
            sidecar["provenance"]["geometry_revision"]["qc"],
        )
        # 1.2.0 is 1.1.0 plus the outline grid closing, 1.3.0 is 1.2.0 plus a
        # user preset's definition.  A sidecar carries the 1.1.0 contract
        # exactly when its record has no closing to declare.
        previous_validator = jsonschema.Draft202012Validator(
            previous_export_schema,
            registry=registry,
        )
        self.assertTrue(list(previous_validator.iter_errors(sidecar)))
        previous_shaped = copy.deepcopy(sidecar)
        previous_shaped["schema_version"] = "1.1.0"
        previous_errors = list(previous_validator.iter_errors(previous_shaped))
        if sidecar["recipe"].get("algorithm_version") == "1.1.0":
            self.assertTrue(previous_errors)
        else:
            self.assertEqual(previous_errors, [])
        invalid_root = copy.deepcopy(sidecar)
        invalid_root["provenance"]["align_ancestry"][0]["matrix4x4"][0][3] = 1.0
        self.assertTrue(list(export_validator.iter_errors(invalid_root)))
        legacy_sidecar = json.loads(json.dumps(sidecar))
        legacy_sidecar["schema_version"] = "1.0.0"
        legacy_sidecar["provenance"].pop("align_ancestry")
        legacy_sidecar["provenance"]["geometry_revision"]["qc"].pop(
            "import_admission"
        )
        legacy_validator = jsonschema.Draft202012Validator(
            legacy_export_schema,
            registry=registry,
        )
        self.assertEqual(list(legacy_validator.iter_errors(legacy_sidecar)), [])

        sidecar["provenance"].pop("dependency_closure")
        self.assertTrue(list(export_validator.iter_errors(sidecar)))

        tampered = json.loads(
            build_vector_export(document, "record:schema-test").sidecar_bytes
        )
        tampered["provenance"]["geometry_revision"]["import_recipe"][
            "dependency_policy"
        ] = "allow_external"
        self.assertTrue(list(export_validator.iter_errors(tampered)))

        closed_cases = []
        for path in (
            ("provenance", "align_revision", "recipe"),
            ("provenance", "align_revision", "qc"),
            ("provenance", "geometry_revision", "qc"),
            ("qc", "payload"),
            ("qc", "record"),
        ):
            candidate = json.loads(
                build_vector_export(document, "record:schema-test").sidecar_bytes
            )
            target = candidate
            for key in path:
                target = target[key]
            target["unknown_contract_field"] = True
            closed_cases.append(candidate)
        for candidate in closed_cases:
            self.assertTrue(list(export_validator.iter_errors(candidate)))

        false_production_claim = json.loads(
            build_vector_export(document, "record:schema-test").sidecar_bytes
        )
        false_production_claim["recipe"][
            "algorithm"
        ] = "archmeshrubbing.triangle_plane_cutline"
        self.assertTrue(
            list(export_validator.iter_errors(false_production_claim))
        )

        production = _production_vector_session()
        for record_id in ("record:production-cutline", "record:production-outline"):
            with self.subTest(record_id=record_id):
                bundle = build_vector_export(production.document, record_id)
                production_sidecar = json.loads(bundle.sidecar_bytes)
                self.assertEqual(
                    list(export_validator.iter_errors(production_sidecar)),
                    [],
                )
                validate_vector_export_bytes(
                    bundle.svg_bytes,
                    bundle.sidecar_bytes,
                )

                production_sidecar["recipe"] = {
                    key: production_sidecar["recipe"][key]
                    for key in ("algorithm", "algorithm_version", "kind")
                }
                self.assertTrue(
                    list(export_validator.iter_errors(production_sidecar))
                )
                tampered_bytes = (
                    json.dumps(
                        production_sidecar,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
                with self.assertRaisesRegex(
                    ArtifactVectorExportError,
                    "payload/recipe contract",
                ):
                    validate_vector_export_bytes(bundle.svg_bytes, tampered_bytes)

    def test_legacy_vector_schema_bytes_are_immutable(self):
        schema_bytes = (ROOT / "schemas/vector_export-1.0.0.schema.json").read_bytes()
        self.assertEqual(
            hashlib.sha256(schema_bytes).hexdigest(),
            "da27146ddd88873c964c489a8b4b0798fffff2022f48f7f9ed86eea073c36ecc",
        )

    def test_production_qc_is_cross_bound_to_recipe_and_admission(self):
        production = _production_vector_session()
        cutline = json.loads(
            build_vector_export(
                production.document,
                "record:production-cutline",
            ).sidecar_bytes
        )
        outline = json.loads(
            build_vector_export(
                production.document,
                "record:production-outline",
            ).sidecar_bytes
        )

        cases: list[tuple[str, dict[str, Any], str]] = []
        changed = copy.deepcopy(cutline)
        changed["qc"]["record"]["classification_tolerance_mm"] *= 2.0
        cases.append(("cutline_classification", changed, "classification tolerance"))
        changed = copy.deepcopy(cutline)
        changed["qc"]["record"]["stitch_tolerance_mm"] *= 2.0
        cases.append(("cutline_stitch", changed, "stitch tolerance"))
        changed = copy.deepcopy(cutline)
        changed["qc"]["record"]["input_face_count"] += 1
        cases.append(("cutline_faces", changed, "input_face_count"))
        changed = copy.deepcopy(outline)
        changed["qc"]["record"]["precision_grid_mm"] *= 2.0
        cases.append(("outline_grid", changed, "precision grid"))
        changed = copy.deepcopy(outline)
        changed["qc"]["record"]["view"] = "front"
        cases.append(("outline_view", changed, "view does not match"))
        changed = copy.deepcopy(outline)
        changed["qc"]["record"]["backend_geos_version"] = "3.13.0"
        cases.append(("outline_backend", changed, "unreviewed geometry backend"))
        changed = copy.deepcopy(outline)
        changed["qc"]["record"]["input_vertex_count"] += 1
        cases.append(("outline_vertices", changed, "input_vertex_count"))
        changed = copy.deepcopy(outline)
        changed["qc"]["record"]["grid_snap_axis_upper_bound_mm"] *= 2.0
        cases.append(("outline_grid_bound", changed, "axis snap bound"))

        for label, sidecar, message in cases:
            with self.subTest(label=label):
                svg_bytes, sidecar_bytes = _rebound_vector_export(sidecar)
                with self.assertRaisesRegex(ArtifactVectorExportError, message):
                    validate_vector_export_bytes(svg_bytes, sidecar_bytes)

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


def test_outline_backend_pin_has_one_definition() -> None:
    """The exporter must not restate the extractor's backend pin.

    Duplicated version literals meant bumping one side alone would make freshly
    written outline packages fail their own validator.
    """

    from src.core import artifact_outline_extractor, artifact_vector_export

    assert (
        artifact_vector_export.REVIEWED_OUTLINE_BACKENDS
        is artifact_outline_extractor.REVIEWED_OUTLINE_BACKENDS
    )


def test_the_computed_backend_is_always_a_reviewed_one() -> None:
    """A pin bump that forgets the reviewed table would write unverifiable packages."""

    from src.core.artifact_outline_extractor import (
        REQUIRED_GEOS_VERSION,
        REQUIRED_SHAPELY_VERSION,
        REVIEWED_OUTLINE_BACKENDS,
    )

    assert (
        REQUIRED_SHAPELY_VERSION,
        REQUIRED_GEOS_VERSION,
    ) in REVIEWED_OUTLINE_BACKENDS


def test_the_export_schema_accepts_exactly_the_reviewed_backends() -> None:
    """The JSON Schema and the code table are one policy, stated twice.

    They are both shipped, and an offline verifier may consult either, so a
    package must not pass one and fail the other.
    """

    from src.core.artifact_outline_extractor import REVIEWED_OUTLINE_BACKENDS

    schema = json.loads(
        (ROOT / "schemas/vector_export-1.3.0.schema.json").read_text(encoding="utf-8")
    )
    properties = schema["$defs"]["recordQc"]["properties"]

    assert set(properties["backend_shapely_version"]["enum"]) == {
        pair[0] for pair in REVIEWED_OUTLINE_BACKENDS
    }
    assert set(properties["backend_geos_version"]["enum"]) == {
        pair[1] for pair in REVIEWED_OUTLINE_BACKENDS
    }


def test_an_unreviewed_backend_is_refused_but_a_reviewed_one_keeps_verifying() -> None:
    """Upgrading Shapely must not retire packages the project already wrote."""

    from src.core.artifact_outline_extractor import (
        REQUIRED_GEOS_VERSION,
        REQUIRED_SHAPELY_VERSION,
    )
    from src.core.artifact_vector_export import (
        ArtifactVectorExportError,
        _require_reviewed_outline_backend,
    )

    _require_reviewed_outline_backend(
        {
            "backend_shapely_version": REQUIRED_SHAPELY_VERSION,
            "backend_geos_version": REQUIRED_GEOS_VERSION,
        }
    )

    with pytest.raises(ArtifactVectorExportError, match="unreviewed geometry backend"):
        _require_reviewed_outline_backend(
            {
                "backend_shapely_version": "9.9.9",
                "backend_geos_version": REQUIRED_GEOS_VERSION,
            }
        )

    # The pair is the unit: a reviewed Shapely against an unreviewed GEOS is
    # not a reviewed backend, because GEOS decides the fixed-precision union.
    with pytest.raises(ArtifactVectorExportError, match="unreviewed geometry backend"):
        _require_reviewed_outline_backend(
            {
                "backend_shapely_version": REQUIRED_SHAPELY_VERSION,
                "backend_geos_version": "9.9.9",
            }
        )
