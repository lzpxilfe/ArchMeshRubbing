from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import unittest

import numpy as np

from src.core.artifact_developed_rubbing import (
    commit_developed_rubbing,
    compute_developed_rubbing,
    validate_developed_rubbing_receipt,
)
from src.core.artifact_outline_extractor import OutlineView, outline_frame
from src.core.artifact_rubbing_export import build_rubbing_export
from src.core.artifact_rubbing_extractor import (
    DigitalRubbingRaster,
    commit_artifact_rubbing,
    compute_artifact_rubbing,
)
from src.core.artifact_rubbing_record import validate_rubbing_receipt
from src.core.artifact_session import ArtifactSession
from src.core.artifact_tile_unwrap_extractor import (
    SECTION_CENTER_CANONICAL_AXIS,
    STATION_MERIDIAN_ARC,
    commit_artifact_tile_unwrap,
    compute_artifact_tile_unwrap,
)
from src.core.mesh_loader import MeshData
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.source_identity import SourceFingerprint
from synthetic_vessel import meridional_strip_faces, positioned_vessel_session


ROOT = Path(__file__).resolve().parents[1]
STAMP = "2026-07-12T00:00:00Z"


def _load_schema(name: str) -> dict[str, object]:
    value = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _generated_receipt_and_sidecar() -> tuple[dict[str, object], dict[str, object]]:
    vertices = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=np.int32,
    )
    mesh = MeshData(
        vertices=vertices,
        faces=faces,
        unit="mm",
        filepath=Path("/private/schema/scan.ply"),
        source_identity=SourceFingerprint(
            sha256="7" * 64,
            size_bytes=512,
            mtime_ns=1,
            original_name="schema-rubbing.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/private/schema/scan.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="schema-test",
        operator="tester",
        created_at=STAMP,
        document_id="artifact:rubbing-schema",
        metadata_revision_id="metadata:rubbing-schema",
        align_revision_id="align:rubbing-schema",
    )
    computation = compute_artifact_rubbing(
        session,
        "top",
        pixels_per_mm=10,
        margin_um=1_000,
        reference_radius_um=500,
        depth_quantization_um=10,
        black_point_um=100,
        ink_strength_percent=100,
        relief_polarity="bidirectional",
    )
    committed = commit_artifact_rubbing(
        session,
        computation,
        record_id="record:rubbing:schema",
        created_at=STAMP,
        operator="tester",
    )
    bundle = build_rubbing_export(
        committed.document,
        "record:rubbing:schema",
        computation.raster,
    )
    receipt = validate_rubbing_receipt(computation.raster.receipt())
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    assert isinstance(sidecar, dict)
    return receipt, sidecar


def _generated_developed_receipt_and_sidecar() -> tuple[dict[str, object], dict[str, object]]:
    """A rubbing on the developed strip of a positioned pot, packaged."""

    session, vertices, faces = positioned_vessel_session(segments=24, rings=12)
    selected = meridional_strip_faces(
        vertices, faces, center_angle_rad=math.pi / 2.0, width_mm=20.0
    )
    unwrap = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="z",
        record_view="top",
        selected_face_indices=selected,
        n_sections=12,
        section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
        station_policy=STATION_MERIDIAN_ARC,
    )
    session = commit_artifact_tile_unwrap(
        session,
        unwrap,
        record_id="record:unwrap:schema",
        created_at=STAMP,
        operator="tester",
    )
    computation = compute_developed_rubbing(
        session,
        "record:unwrap:schema",
        pixels_per_mm=2,
        margin_um=0,
        reference_radius_um=3_000,
        depth_quantization_um=10,
        black_point_um=250,
        ink_strength_percent=100,
        relief_polarity="bidirectional",
    )
    committed = commit_developed_rubbing(
        session,
        computation,
        record_id="record:developed:schema",
        created_at=STAMP,
        operator="tester",
    )
    bundle = build_rubbing_export(
        committed.document,
        "record:developed:schema",
        computation.raster,
    )
    receipt = validate_developed_rubbing_receipt(computation.raster.receipt())
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    assert isinstance(sidecar, dict)
    return receipt, sidecar


class TestRubbingSchemas(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        jsonschema = importlib.import_module("jsonschema")
        referencing = importlib.import_module("referencing")
        cls.receipt_schema = _load_schema("rubbing_receipt-1.0.0.schema.json")
        cls.developed_receipt_schema = _load_schema(
            "developed_rubbing_receipt-1.0.0.schema.json"
        )
        cls.export_schema = _load_schema("rubbing_export-1.3.0.schema.json")
        cls.legacy_export_schema = _load_schema(
            "rubbing_export-1.0.0.schema.json"
        )
        cls.legacy_1_1_export_schema = _load_schema(
            "rubbing_export-1.1.0.schema.json"
        )
        cls.legacy_1_2_export_schema = _load_schema(
            "rubbing_export-1.2.0.schema.json"
        )
        cls.mesh_admission_schema = _load_schema(
            "mesh_admission_receipt-1.0.0.schema.json"
        )
        cls.import_recipe_schema = _load_schema(
            "mesh_import_recipe-1.0.0.schema.json"
        )
        cls.import_recipe_v2_schema = _load_schema(
            "mesh_import_recipe-2.0.0.schema.json"
        )
        jsonschema.Draft202012Validator.check_schema(cls.receipt_schema)
        jsonschema.Draft202012Validator.check_schema(cls.developed_receipt_schema)
        jsonschema.Draft202012Validator.check_schema(cls.export_schema)
        jsonschema.Draft202012Validator.check_schema(cls.legacy_export_schema)
        jsonschema.Draft202012Validator.check_schema(cls.legacy_1_1_export_schema)
        jsonschema.Draft202012Validator.check_schema(cls.legacy_1_2_export_schema)
        jsonschema.Draft202012Validator.check_schema(cls.mesh_admission_schema)
        jsonschema.Draft202012Validator.check_schema(cls.import_recipe_schema)
        jsonschema.Draft202012Validator.check_schema(cls.import_recipe_v2_schema)
        cls.receipt_validator = jsonschema.Draft202012Validator(cls.receipt_schema)
        cls.developed_receipt_validator = jsonschema.Draft202012Validator(
            cls.developed_receipt_schema
        )
        registry = referencing.Registry().with_resource(
            cls.receipt_schema["$id"],
            referencing.Resource.from_contents(cls.receipt_schema),
        )
        registry = registry.with_resource(
            cls.developed_receipt_schema["$id"],
            referencing.Resource.from_contents(cls.developed_receipt_schema),
        )
        registry = registry.with_resource(
            cls.legacy_export_schema["$id"],
            referencing.Resource.from_contents(cls.legacy_export_schema),
        )
        registry = registry.with_resource(
            cls.legacy_1_1_export_schema["$id"],
            referencing.Resource.from_contents(cls.legacy_1_1_export_schema),
        )
        registry = registry.with_resource(
            cls.legacy_1_2_export_schema["$id"],
            referencing.Resource.from_contents(cls.legacy_1_2_export_schema),
        )
        registry = registry.with_resource(
            cls.mesh_admission_schema["$id"],
            referencing.Resource.from_contents(cls.mesh_admission_schema),
        )
        registry = registry.with_resource(
            cls.import_recipe_schema["$id"],
            referencing.Resource.from_contents(cls.import_recipe_schema),
        )
        registry = registry.with_resource(
            cls.import_recipe_v2_schema["$id"],
            referencing.Resource.from_contents(cls.import_recipe_v2_schema),
        )
        cls.export_validator = jsonschema.Draft202012Validator(
            cls.export_schema,
            registry=registry,
        )
        cls.receipt, cls.sidecar = _generated_receipt_and_sidecar()
        cls.developed_receipt, cls.developed_sidecar = (
            _generated_developed_receipt_and_sidecar()
        )

    def test_legacy_export_schema_remains_byte_exact(self) -> None:
        payload = (ROOT / "schemas" / "rubbing_export-1.0.0.schema.json").read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "31cc5dd55a8ce2acce934fd7fdd3211093f58a80a7b5259d8d6f52f7cb50ac89",
        )
        payload = (ROOT / "schemas" / "rubbing_export-1.1.0.schema.json").read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "6dc7ceb0ed456f451d91935fa62d0c9585d12acd6755a265a5afebb5c6efc811",
        )
        # 1.2.0 shipped before the paper wash existed, so its six-view recipe
        # comes from the frozen 1.0.0 definition and cannot describe one.
        payload = (ROOT / "schemas" / "rubbing_export-1.2.0.schema.json").read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "7bc4edda8ec1dc0044457f7a494a6ec3056b7c972627ee1f7e4943b0d5de9707",
        )

    def test_developed_rubbing_receipt_and_sidecar_validate(self) -> None:
        self.assert_schema_valid(self.developed_receipt_validator, self.developed_receipt)
        self.assert_schema_invalid(self.receipt_validator, self.developed_receipt)
        self.assert_schema_valid(self.export_validator, self.developed_sidecar)
        self.assertEqual(self.developed_sidecar["schema_version"], "1.3.0")
        recipe = self.developed_sidecar["recipe"]
        assert isinstance(recipe, dict)
        self.assertEqual(recipe["kind"], "developed_rubbing")

        # A developed receipt cannot pose as a six-view one, and the two
        # recipe shapes do not blend.
        posing = copy.deepcopy(self.developed_sidecar)
        posing["raster_receipt"] = copy.deepcopy(self.receipt)
        self.assert_schema_invalid(self.export_validator, posing)
        blended = copy.deepcopy(self.developed_sidecar)
        blended_recipe = blended["recipe"]
        assert isinstance(blended_recipe, dict)
        blended_recipe["view"] = "top"
        self.assert_schema_invalid(self.export_validator, blended)
        wrong_ref = copy.deepcopy(self.developed_sidecar)
        provenance = wrong_ref["provenance"]
        assert isinstance(provenance, dict)
        record = provenance["record"]
        assert isinstance(record, dict)
        record["geometry_ref"] = (
            "urn:archmeshrubbing:digital-rubbing-raster:sha256:" + "0" * 64
        )
        self.assert_schema_invalid(self.export_validator, wrong_ref)
        # The developed sidecar is a 1.2.0 shape; the 1.1.0 contract has no
        # room for it.
        legacy_validator = importlib.import_module("jsonschema").Draft202012Validator(
            self.legacy_1_1_export_schema,
            registry=self.export_validator._registry,  # type: ignore[attr-defined]
        )
        self.assert_schema_invalid(legacy_validator, self.developed_sidecar)

    def assert_schema_valid(self, validator, value: object) -> None:
        errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
        self.assertEqual([error.message for error in errors], [])

    def assert_schema_invalid(self, validator, value: object) -> None:
        self.assertTrue(list(validator.iter_errors(value)))

    def test_generated_receipt_and_export_sidecar_validate(self) -> None:
        self.assert_schema_valid(self.receipt_validator, self.receipt)
        self.assert_schema_valid(self.export_validator, self.sidecar)
        self.assertEqual(self.sidecar["schema_version"], "1.3.0")
        provenance = self.sidecar["provenance"]
        assert isinstance(provenance, dict)
        geometry = provenance["geometry_revision"]
        assert isinstance(geometry, dict)
        import_recipe = geometry["import_recipe"]
        assert isinstance(import_recipe, dict)
        self.assertEqual(import_recipe["recipe_version"], "1.0.0")
        self.assertEqual(import_recipe["dependency_policy"], "deny_external")

        pixels = np.array([[[255, 255]]], dtype=np.uint8)
        for view in OutlineView:
            with self.subTest(view=view.value):
                raster = DigitalRubbingRaster(
                    pixels=pixels,
                    frame=outline_frame(view),
                    view=view,
                    pixels_per_meter=1_000,
                    minimum_u_pixel_index=0,
                    minimum_v_pixel_index=0,
                )
                self.assert_schema_valid(self.receipt_validator, raster.receipt())

    def test_receipt_schema_rejects_closed_contract_tampering(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = []

        unknown_root = copy.deepcopy(self.receipt)
        unknown_root["unexpected"] = True
        cases.append(("unknown_root", unknown_root))

        missing_field = copy.deepcopy(self.receipt)
        missing_field.pop("raw_pixel_sha256")
        cases.append(("missing_field", missing_field))

        non_integral_density = copy.deepcopy(self.receipt)
        non_integral_density["pixels_per_meter"] = 1_500
        cases.append(("non_integral_density", non_integral_density))

        wrong_view_frame = copy.deepcopy(self.receipt)
        wrong_view_frame["view"] = "front"
        cases.append(("wrong_view_frame", wrong_view_frame))

        uppercase_digest = copy.deepcopy(self.receipt)
        uppercase_digest["raster_sha256"] = "A" * 64
        cases.append(("uppercase_digest", uppercase_digest))

        nested_extra = copy.deepcopy(self.receipt)
        width_exact = nested_extra["width_mm_exact"]
        assert isinstance(width_exact, dict)
        width_exact["unit"] = "mm"
        cases.append(("nested_extra", nested_extra))

        for label, value in cases:
            with self.subTest(label=label):
                self.assert_schema_invalid(self.receipt_validator, value)

    def test_export_schema_rejects_tampered_or_private_shapes(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = []

        unknown_root = copy.deepcopy(self.sidecar)
        unknown_root["unexpected"] = True
        cases.append(("unknown_root", unknown_root))

        artifact_extra = copy.deepcopy(self.sidecar)
        artifact = artifact_extra["artifact"]
        assert isinstance(artifact, dict)
        artifact["absolute_path"] = "/private/schema/artifact.png"
        cases.append(("artifact_extra", artifact_extra))

        privacy_claim = copy.deepcopy(self.sidecar)
        privacy = privacy_claim["privacy"]
        assert isinstance(privacy, dict)
        privacy["annotations_embedded_in_primary_png"] = True
        cases.append(("privacy_claim", privacy_claim))

        wrong_record_type = copy.deepcopy(self.sidecar)
        provenance = wrong_record_type["provenance"]
        assert isinstance(provenance, dict)
        record = provenance["record"]
        assert isinstance(record, dict)
        record["type"] = "vector.outline.v1"
        cases.append(("wrong_record_type", wrong_record_type))

        recipe_extra = copy.deepcopy(self.sidecar)
        recipe = recipe_extra["recipe"]
        assert isinstance(recipe, dict)
        recipe["gpu_backend"] = "OpenGL"
        cases.append(("recipe_extra", recipe_extra))

        wrong_recipe_frame = copy.deepcopy(self.sidecar)
        recipe = wrong_recipe_frame["recipe"]
        assert isinstance(recipe, dict)
        recipe["view"] = "left"
        cases.append(("wrong_recipe_frame", wrong_recipe_frame))

        raster_qc_extra = copy.deepcopy(self.sidecar)
        qc = raster_qc_extra["qc"]
        assert isinstance(qc, dict)
        raster_qc = qc["raster"]
        assert isinstance(raster_qc, dict)
        raster_qc["sampled"] = False
        cases.append(("raster_qc_extra", raster_qc_extra))

        public_align_extra = copy.deepcopy(self.sidecar)
        provenance = public_align_extra["provenance"]
        assert isinstance(provenance, dict)
        align = provenance["align_revision"]
        assert isinstance(align, dict)
        align_recipe = align["recipe"]
        assert isinstance(align_recipe, dict)
        align_recipe["source_path"] = "/private/schema/scan.ply"
        cases.append(("public_align_extra", public_align_extra))

        import_recipe_tampered = copy.deepcopy(self.sidecar)
        provenance = import_recipe_tampered["provenance"]
        assert isinstance(provenance, dict)
        geometry = provenance["geometry_revision"]
        assert isinstance(geometry, dict)
        import_recipe = geometry["import_recipe"]
        assert isinstance(import_recipe, dict)
        import_recipe["runtime_lock_sha256"] = "not-a-sha256"
        cases.append(("import_recipe_tampered", import_recipe_tampered))

        missing_admission = copy.deepcopy(self.sidecar)
        provenance = missing_admission["provenance"]
        assert isinstance(provenance, dict)
        geometry = provenance["geometry_revision"]
        assert isinstance(geometry, dict)
        geometry_qc = geometry["qc"]
        assert isinstance(geometry_qc, dict)
        geometry_qc.pop("import_admission")
        cases.append(("missing_admission", missing_admission))

        legacy_align_qc = copy.deepcopy(self.sidecar)
        provenance = legacy_align_qc["provenance"]
        assert isinstance(provenance, dict)
        align = provenance["align_revision"]
        assert isinstance(align, dict)
        align["qc"] = {"rigid": True}
        cases.append(("legacy_align_qc", legacy_align_qc))

        invalid_root = copy.deepcopy(self.sidecar)
        provenance = invalid_root["provenance"]
        assert isinstance(provenance, dict)
        ancestry = provenance["align_ancestry"]
        assert isinstance(ancestry, list)
        root = ancestry[0]
        assert isinstance(root, dict)
        matrix = root["matrix4x4"]
        assert isinstance(matrix, list)
        matrix[0][3] = 1.0
        cases.append(("invalid_align_root", invalid_root))

        for label, value in cases:
            with self.subTest(label=label):
                self.assert_schema_invalid(self.export_validator, value)

    def test_record_qc_remains_an_explicit_extension_boundary(self) -> None:
        extended = copy.deepcopy(self.sidecar)
        qc = extended["qc"]
        assert isinstance(qc, dict)
        record_qc = qc["record"]
        assert isinstance(record_qc, dict)
        record_qc["org.example:review-note"] = "non-authoritative QC extension"
        self.assert_schema_valid(self.export_validator, extended)


if __name__ == "__main__":
    unittest.main()
