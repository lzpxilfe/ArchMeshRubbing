from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest

import numpy as np

from src.core.artifact_rubbing_export import (
    ArtifactRubbingExportError,
    RUBBING_EXPORT_PNG_NAME,
    RUBBING_EXPORT_SIDECAR_NAME,
    build_rubbing_export,
    export_rubbing_package,
    validate_rubbing_export_bytes,
    validate_rubbing_export_package,
)
from src.core.artifact_rubbing_extractor import (
    commit_artifact_rubbing,
    compute_artifact_rubbing,
)
from src.core.artifact_session import ArtifactSession
from src.core.canonical_json import canonical_json_bytes
from src.core.canonical_png import (
    decode_canonical_ga8_png,
    encode_canonical_ga8_png,
)
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


STAMP = "2026-07-12T00:00:00Z"


def _session() -> ArtifactSession:
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
        filepath=Path("/private/lab/alice/secret-scan.ply"),
        source_identity=SourceFingerprint(
            sha256="9" * 64,
            size_bytes=987,
            mtime_ns=1,
            original_name="유물-탁본.ply",
            format="ply",
        ),
        source_format="ply",
    )
    return ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/private/lab/alice/secret-scan.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="0.6-test",
        operator="archaeologist",
        created_at=STAMP,
        document_id="artifact:rubbing-export",
        metadata_revision_id="metadata:r1",
        align_revision_id="align:r1",
    )


def _committed():
    session = _session()
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
        record_id="record:rubbing:export",
        created_at=STAMP,
        operator="archaeologist",
    )
    return committed, computation


def _sidecar(bundle) -> dict[str, Any]:
    value = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    assert isinstance(value, dict)
    return value


class TestRubbingExport(unittest.TestCase):
    def test_build_is_exact_scaled_private_and_golden(self):
        session, computation = _committed()
        bundle = build_rubbing_export(
            session.document,
            "record:rubbing:export",
            computation.raster,
        )
        self.assertEqual(bundle.width_pixels, 40)
        self.assertEqual(bundle.height_pixels, 40)
        self.assertEqual(bundle.pixels_per_meter, 10_000)
        self.assertEqual(
            bundle.png_sha256,
            "acec498e5fe02d77685b873a420d674d3da4c54144342b94dba383cb963a9ff5",
        )
        self.assertEqual(
            bundle.sidecar_sha256,
            "34f5777092d193e7307d4014a82d51416c47f3a29acd6cd11906feb5b513e91f",
        )
        pixels, ppm, metadata = decode_canonical_ga8_png(bundle.png_bytes)
        np.testing.assert_array_equal(pixels, computation.raster.pixels)
        self.assertEqual(ppm, 10_000)
        self.assertEqual(metadata["physical_scale"], "1:1_planar_sampling")
        sidecar = _sidecar(bundle)
        self.assertEqual(
            sidecar["presentation"]["pixel_pitch_mm_exact"],
            {"denominator": 10_000, "numerator": 1000},
        )
        self.assertNotIn(b"/private/lab/alice", bundle.sidecar_bytes)
        self.assertNotIn(b"secret-scan", bundle.png_bytes)

    def test_offline_package_relocation_and_independent_process_validation(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = export_rubbing_package(
                root / "source.amr-rubbing",
                session.document,
                "record:rubbing:export",
                computation.raster,
            )
            relocated = root / "relocated.amr-rubbing"
            package.rename(relocated)
            verified = validate_rubbing_export_package(relocated)
            self.assertEqual(verified.raster_sha256, computation.raster.raster_sha256)

            program = (
                "import json,sys;"
                "from src.core.artifact_rubbing_export import "
                "validate_rubbing_export_package;"
                "b=validate_rubbing_export_package(sys.argv[1]);"
                "print(json.dumps({'raster':b.raster_sha256,'ppm':b.pixels_per_meter}))"
            )
            completed = subprocess.run(
                [sys.executable, "-c", program, str(relocated)],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )
            receipt = json.loads(completed.stdout)
            self.assertEqual(receipt["raster"], computation.raster.raster_sha256)
            self.assertEqual(receipt["ppm"], 10_000)

    def test_png_pixel_phys_and_sidecar_claim_tampering_are_detected(self):
        session, computation = _committed()
        bundle = build_rubbing_export(
            session.document,
            "record:rubbing:export",
            computation.raster,
        )
        pixels, ppm, metadata = decode_canonical_ga8_png(bundle.png_bytes)
        changed = pixels.copy()
        changed[10, 10, 0] ^= 1
        changed_png = encode_canonical_ga8_png(
            changed,
            pixels_per_meter=ppm,
            metadata=metadata,
        )
        sidecar = _sidecar(bundle)
        sidecar["artifact"]["sha256"] = hashlib.sha256(changed_png).hexdigest()
        sidecar["artifact"]["size_bytes"] = len(changed_png)
        with self.assertRaisesRegex(ArtifactRubbingExportError, "receipt"):
            validate_rubbing_export_bytes(
                changed_png,
                canonical_json_bytes(sidecar),
            )

        changed_phys = encode_canonical_ga8_png(
            pixels,
            pixels_per_meter=20_000,
            metadata=metadata,
        )
        sidecar = _sidecar(bundle)
        sidecar["artifact"]["sha256"] = hashlib.sha256(changed_phys).hexdigest()
        sidecar["artifact"]["size_bytes"] = len(changed_phys)
        with self.assertRaisesRegex(ArtifactRubbingExportError, "pHYs|receipt"):
            validate_rubbing_export_bytes(
                changed_phys,
                canonical_json_bytes(sidecar),
            )

        sidecar = _sidecar(bundle)
        sidecar["privacy"]["annotations_embedded_in_primary_png"] = True
        with self.assertRaisesRegex(ArtifactRubbingExportError, "privacy|metadata"):
            validate_rubbing_export_bytes(
                bundle.png_bytes,
                canonical_json_bytes(sidecar),
            )

    def test_stale_record_extra_member_and_existing_destination_fail_closed(self):
        session, computation = _committed()
        switched = session.commit_preview(
            translation_mm=(1.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            operator="tester",
            created_at=STAMP,
            revision_id="align:r2",
        )
        with self.assertRaisesRegex(ArtifactRubbingExportError, "FRESH"):
            build_rubbing_export(
                switched.document,
                "record:rubbing:export",
                computation.raster,
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "result.amr-rubbing"
            destination.mkdir()
            sentinel = destination / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactRubbingExportError, "already exists"):
                export_rubbing_package(
                    destination,
                    session.document,
                    "record:rubbing:export",
                    computation.raster,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

            package = export_rubbing_package(
                root / "valid.amr-rubbing",
                session.document,
                "record:rubbing:export",
                computation.raster,
            )
            (package / ".DS_Store").write_bytes(b"metadata")
            with self.assertRaisesRegex(ArtifactRubbingExportError, "exactly two"):
                validate_rubbing_export_package(package)

    def test_package_members_are_exactly_canonical_names(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            package = export_rubbing_package(
                Path(temporary) / "artifact.amr-rubbing",
                session.document,
                "record:rubbing:export",
                computation.raster,
            )
            self.assertEqual(
                sorted(path.name for path in package.iterdir()),
                sorted([RUBBING_EXPORT_PNG_NAME, RUBBING_EXPORT_SIDECAR_NAME]),
            )


if __name__ == "__main__":
    unittest.main()
