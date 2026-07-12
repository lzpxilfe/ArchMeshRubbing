from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.core.artifact_outline_extractor import OutlineView, outline_frame
from src.core.artifact_rubbing_export import build_rubbing_export
from src.core.artifact_rubbing_extractor import (
    DigitalRubbingRaster,
    commit_artifact_rubbing,
    compute_artifact_rubbing,
)
from src.core.artifact_rubbing_record import validate_rubbing_receipt
from src.core.artifact_session import ArtifactSession
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


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


class TestRubbingSchemas(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        jsonschema = importlib.import_module("jsonschema")
        referencing = importlib.import_module("referencing")
        cls.receipt_schema = _load_schema("rubbing_receipt-1.0.0.schema.json")
        cls.export_schema = _load_schema("rubbing_export-1.0.0.schema.json")
        jsonschema.Draft202012Validator.check_schema(cls.receipt_schema)
        jsonschema.Draft202012Validator.check_schema(cls.export_schema)
        cls.receipt_validator = jsonschema.Draft202012Validator(cls.receipt_schema)
        registry = referencing.Registry().with_resource(
            cls.receipt_schema["$id"],
            referencing.Resource.from_contents(cls.receipt_schema),
        )
        cls.export_validator = jsonschema.Draft202012Validator(
            cls.export_schema,
            registry=registry,
        )
        cls.receipt, cls.sidecar = _generated_receipt_and_sidecar()

    def assert_schema_valid(self, validator, value: object) -> None:
        errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
        self.assertEqual([error.message for error in errors], [])

    def assert_schema_invalid(self, validator, value: object) -> None:
        self.assertTrue(list(validator.iter_errors(value)))

    def test_generated_receipt_and_export_sidecar_validate(self) -> None:
        self.assert_schema_valid(self.receipt_validator, self.receipt)
        self.assert_schema_valid(self.export_validator, self.sidecar)

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
