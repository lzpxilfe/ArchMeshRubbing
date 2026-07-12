from __future__ import annotations

from dataclasses import replace
import errno
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np

import src.core.artifact_rubbing_export as rubbing_export
from src.core.artifact_rubbing_export import (
    ArtifactRubbingExportError,
    MAX_IGNORABLE_OS_METADATA_BYTES,
    RUBBING_EXPORT_PNG_NAME,
    RUBBING_EXPORT_SIDECAR_NAME,
    build_rubbing_export,
    discard_prepared_rubbing_package,
    discard_staged_rubbing_package,
    export_rubbing_package,
    prepare_staged_rubbing_publication,
    publish_prepared_rubbing_package,
    publish_staged_rubbing_package,
    stage_rubbing_package,
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
    def test_prepared_capability_is_exact_and_single_use(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "exact.amr-rubbing"
            staging = stage_rubbing_package(
                destination,
                session.document,
                "record:rubbing:export",
                computation.raster,
            )
            prepared = prepare_staged_rubbing_publication(
                staging,
                destination,
                document=session.document,
            )
            with self.assertRaisesRegex(
                ArtifactRubbingExportError,
                "invalid or consumed",
            ):
                publish_prepared_rubbing_package(replace(prepared))

            self.assertEqual(
                publish_prepared_rubbing_package(prepared),
                destination,
            )
            with self.assertRaises(ArtifactRubbingExportError) as raised:
                publish_prepared_rubbing_package(prepared)
            self.assertTrue(raised.exception.committed)

    def test_public_publish_rejects_never_owned_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            foreign = root / f".amrr-stage-{'f' * 32}"
            foreign.mkdir()
            destination = root / "foreign.amr-rubbing"
            with self.assertRaisesRegex(
                ArtifactRubbingExportError,
                "not created by this process",
            ):
                publish_staged_rubbing_package(foreign, destination)
            self.assertFalse(destination.exists())

    def test_fixed_length_stage_supports_long_destination_name(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / ("탁" * 70 + ".amr-rubbing")
            staging = stage_rubbing_package(
                destination,
                session.document,
                "record:rubbing:export",
                computation.raster,
            )
            self.assertEqual(len(staging.name), len(".amrr-stage-") + 32)
            self.assertNotIn(destination.name, staging.name)
            prepared = prepare_staged_rubbing_publication(
                staging,
                destination,
                document=session.document,
            )
            self.assertEqual(
                publish_prepared_rubbing_package(prepared),
                destination,
            )

    def test_pre_moved_stage_is_reported_as_committed_visible_effect(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "pre-moved.amr-rubbing"
            staging = stage_rubbing_package(
                destination,
                session.document,
                "record:rubbing:export",
                computation.raster,
            )
            prepared = prepare_staged_rubbing_publication(
                staging,
                destination,
                document=session.document,
            )
            staging.rename(destination)
            with self.assertRaises(ArtifactRubbingExportError) as raised:
                discard_prepared_rubbing_package(prepared)
            self.assertTrue(raised.exception.committed)
            self.assertTrue(destination.is_dir())

    def test_discard_detects_top_directory_swap_and_preserves_foreign(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "swap.amr-rubbing"
            staging = stage_rubbing_package(
                destination,
                session.document,
                "record:rubbing:export",
                computation.raster,
            )
            moved_owned = root / "moved-after-open"
            original_empty = rubbing_export._empty_rubbing_directory_fd

            def swap_then_empty(directory_fd: int) -> None:
                quarantine = next(root.glob(".amrr-discard-*"))
                quarantine.rename(moved_owned)
                quarantine.mkdir()
                (quarantine / "foreign.txt").write_text(
                    "preserve",
                    encoding="utf-8",
                )
                original_empty(directory_fd)

            with patch.object(
                rubbing_export,
                "_empty_rubbing_directory_fd",
                side_effect=swap_then_empty,
            ), patch.object(
                rubbing_export.shutil,
                "rmtree",
                side_effect=AssertionError("POSIX cleanup must not use rmtree"),
            ):
                self.assertFalse(
                    discard_staged_rubbing_package(staging, destination)
                )
            self.assertEqual(
                (staging / "foreign.txt").read_text(encoding="utf-8"),
                "preserve",
            )
            self.assertTrue(moved_owned.is_dir())

    def test_windows_fallback_quarantines_and_cleans_owned_inode(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "windows.amr-rubbing"
            staging = stage_rubbing_package(
                destination,
                session.document,
                "record:rubbing:export",
                computation.raster,
            )
            real_rmtree = rubbing_export.shutil.rmtree
            with patch.object(
                rubbing_export,
                "_descriptor_cleanup_available",
                return_value=False,
            ), patch.object(
                rubbing_export,
                "_windows_cleanup_fallback_required",
                return_value=True,
            ), patch.object(
                rubbing_export.shutil,
                "rmtree",
                wraps=real_rmtree,
            ) as rmtree:
                self.assertTrue(
                    discard_staged_rubbing_package(staging, destination)
                )
            rmtree.assert_called_once()
            self.assertFalse(staging.exists())

    def test_unsupported_directory_fsync_is_committed_uncertainty(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            rubbing_export,
            "fsync_export_directory",
            return_value=False,
        ):
            destination = Path(temporary) / "unsupported.amr-rubbing"
            staging = stage_rubbing_package(
                destination,
                session.document,
                "record:rubbing:export",
                computation.raster,
            )
            prepared = prepare_staged_rubbing_publication(
                staging,
                destination,
                document=session.document,
            )
            with self.assertRaises(ArtifactRubbingExportError) as raised:
                publish_prepared_rubbing_package(prepared)
            self.assertTrue(raised.exception.committed)
            self.assertIn("unsupported", str(raised.exception))
            self.assertTrue(destination.is_dir())

    def test_post_rename_destination_inode_is_verified(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "verify-inode.amr-rubbing"
            staging = stage_rubbing_package(
                destination,
                session.document,
                "record:rubbing:export",
                computation.raster,
            )
            prepared = prepare_staged_rubbing_publication(
                staging,
                destination,
                document=session.document,
            )
            moved_owned = root / "published-owned-moved"
            real_rename = rubbing_export.publish_export_directory_noreplace

            def replace_after_rename(source: Path, target: Path) -> None:
                real_rename(source, target)
                target.rename(moved_owned)
                target.mkdir()
                (target / "foreign.txt").write_text("preserve", encoding="utf-8")

            with patch.object(
                rubbing_export,
                "publish_export_directory_noreplace",
                side_effect=replace_after_rename,
            ):
                with self.assertRaises(ArtifactRubbingExportError) as raised:
                    publish_prepared_rubbing_package(prepared)
            self.assertTrue(raised.exception.committed)
            self.assertEqual(
                (destination / "foreign.txt").read_text(encoding="utf-8"),
                "preserve",
            )
            self.assertTrue(moved_owned.is_dir())

    def test_stage_is_same_parent_verified_and_does_not_publish_destination(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "staged.amr-rubbing"
            collision = "c" * 32
            owned = "a" * 32
            foreign = root / f".amrr-stage-{collision}"
            foreign.mkdir()
            sentinel = foreign / "foreign-sentinel.txt"
            sentinel.write_text("do not remove", encoding="utf-8")

            with patch.object(
                rubbing_export.uuid,
                "uuid4",
                side_effect=[
                    SimpleNamespace(hex=collision),
                    SimpleNamespace(hex=owned),
                ],
            ):
                staging = stage_rubbing_package(
                    destination,
                    session.document,
                    "record:rubbing:export",
                    computation.raster,
                )

            self.assertEqual(staging.parent, destination.parent)
            self.assertEqual(staging.name, f".amrr-stage-{owned}")
            self.assertFalse(destination.exists())
            validate_rubbing_export_package(staging, document=session.document)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not remove")

            with patch.object(
                rubbing_export,
                "fsync_export_directory",
            ) as fsync_directory:
                published = publish_staged_rubbing_package(
                    staging,
                    destination,
                    document=session.document,
                )
            self.assertEqual(published, destination)
            fsync_directory.assert_called_once_with(destination.parent)
            self.assertFalse(staging.exists())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not remove")

    def test_stage_collision_budget_preserves_foreign_directory(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "busy.amr-rubbing"
            collision = "c" * 32
            foreign = root / f".amrr-stage-{collision}"
            foreign.mkdir()
            sentinel = foreign / "sentinel.txt"
            sentinel.write_text("foreign", encoding="utf-8")

            with patch.object(
                rubbing_export.uuid,
                "uuid4",
                return_value=SimpleNamespace(hex=collision),
            ) as uuid4:
                with self.assertRaisesRegex(
                    ArtifactRubbingExportError,
                    "after 16 attempts",
                ):
                    stage_rubbing_package(
                        destination,
                        session.document,
                        "record:rubbing:export",
                        computation.raster,
                    )

            self.assertEqual(uuid4.call_count, 16)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "foreign")
            self.assertFalse(destination.exists())

    def test_discard_removes_only_the_registered_staging_inode(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "discard.amr-rubbing"
            staging = stage_rubbing_package(
                destination,
                session.document,
                "record:rubbing:export",
                computation.raster,
            )
            self.assertTrue(
                discard_staged_rubbing_package(staging, destination)
            )
            self.assertFalse(staging.exists())

            replaced = stage_rubbing_package(
                destination,
                session.document,
                "record:rubbing:export",
                computation.raster,
            )
            original = root / "moved-owned-rubbing-staging"
            replaced.rename(original)
            replaced.mkdir()
            sentinel = replaced / "foreign.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            self.assertFalse(
                discard_staged_rubbing_package(replaced, destination)
            )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            self.assertTrue(original.is_dir())

    def test_publish_reports_committed_directory_fsync_uncertainty(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "uncertain.amr-rubbing"
            staging = stage_rubbing_package(
                destination,
                session.document,
                "record:rubbing:export",
                computation.raster,
            )
            with patch.object(
                rubbing_export,
                "fsync_export_directory",
                side_effect=OSError(errno.EIO, "injected fsync failure"),
            ):
                with self.assertRaises(ArtifactRubbingExportError) as raised:
                    publish_staged_rubbing_package(
                        staging,
                        destination,
                        document=session.document,
                    )
            self.assertTrue(raised.exception.committed)
            self.assertTrue(destination.is_dir())
            self.assertFalse(staging.exists())

    def test_invalid_record_does_not_create_destination_parent(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "new-parent" / "bad.amr-rubbing"
            with self.assertRaisesRegex(
                ArtifactRubbingExportError,
                "does not exist",
            ):
                stage_rubbing_package(
                    destination,
                    session.document,
                    "record:missing",
                    computation.raster,
                )
            self.assertFalse(destination.parent.exists())

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
            validate_rubbing_export_package(package)
            (package / "unexpected.txt").write_text("unsafe", encoding="utf-8")
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

    def test_ignorable_os_metadata_is_bounded_and_must_be_regular(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            package = export_rubbing_package(
                Path(temporary) / "metadata.amr-rubbing",
                session.document,
                "record:rubbing:export",
                computation.raster,
            )
            metadata = package / ".DS_Store"
            with metadata.open("wb") as stream:
                stream.truncate(MAX_IGNORABLE_OS_METADATA_BYTES + 1)
            with self.assertRaisesRegex(
                ArtifactRubbingExportError,
                "OS metadata entry is unsafe",
            ):
                validate_rubbing_export_package(package)

    def test_export_publish_race_preserves_winner_and_cleans_owned_staging(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "raced.amr-rubbing"
            real_publish = rubbing_export.publish_export_directory_noreplace

            def create_competing_destination(source: Path, target: Path) -> None:
                target.mkdir()
                (target / "winner.txt").write_text(
                    "other process",
                    encoding="utf-8",
                )
                real_publish(source, target)

            with patch.object(
                rubbing_export,
                "publish_export_directory_noreplace",
                side_effect=create_competing_destination,
            ):
                with self.assertRaisesRegex(
                    ArtifactRubbingExportError,
                    "already exists",
                ):
                    export_rubbing_package(
                        destination,
                        session.document,
                        "record:rubbing:export",
                        computation.raster,
                    )

            self.assertEqual(
                (destination / "winner.txt").read_text(encoding="utf-8"),
                "other process",
            )
            self.assertEqual(
                list(root.glob(".amrr-stage-*")),
                [],
            )

    def test_publish_race_preserves_destination_and_returned_staging(self):
        session, computation = _committed()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "raced-stage.amr-rubbing"
            staging = stage_rubbing_package(
                destination,
                session.document,
                "record:rubbing:export",
                computation.raster,
            )
            destination.mkdir()
            sentinel = destination / "winner.txt"
            sentinel.write_text("other process", encoding="utf-8")

            with self.assertRaisesRegex(
                ArtifactRubbingExportError,
                "already exists",
            ):
                publish_staged_rubbing_package(
                    staging,
                    destination,
                    document=session.document,
                )

            self.assertTrue(staging.is_dir())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "other process")


if __name__ == "__main__":
    unittest.main()
