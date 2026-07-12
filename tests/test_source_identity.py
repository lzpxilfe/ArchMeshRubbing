from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import trimesh

import src.core.source_identity as source_identity
from src.core.mesh_loader import MeshLoader
from src.core.source_identity import (
    DEFAULT_HASH_CHUNK_SIZE,
    SourceChangedError,
    SourceFingerprint,
    SourceVerification,
    SourceVerificationStatus,
    compare_fingerprints,
    fingerprint_file,
    legacy_unverified_source,
    missing_source,
    verify_source,
)


class _TrackingHash:
    def __init__(self) -> None:
        self._delegate = hashlib.new("sha256")
        self.update_sizes: list[int] = []

    def update(self, data: bytes) -> None:
        self.update_sizes.append(len(data))
        self._delegate.update(data)

    def hexdigest(self) -> str:
        return self._delegate.hexdigest()


class _MtimeMutatingHash:
    def __init__(self, source_path: Path) -> None:
        self._delegate = hashlib.new("sha256")
        self._source_path = source_path
        self._mutated = False

    def update(self, data: bytes) -> None:
        if not self._mutated:
            current = self._source_path.stat()
            os.utime(
                self._source_path,
                ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000),
            )
            self._mutated = True
        self._delegate.update(data)

    def hexdigest(self) -> str:
        return self._delegate.hexdigest()


class TestSourceIdentity(unittest.TestCase):
    def test_known_bytes_have_exact_stable_fingerprint(self) -> None:
        payload = b"ArchMeshRubbing\x00primary-source\n"
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "청동거울.Scan.PLY"
            source.write_bytes(payload)
            fingerprint = fingerprint_file(source, chunk_size=7)

        self.assertEqual(fingerprint.sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(fingerprint.size_bytes, len(payload))
        self.assertEqual(fingerprint.original_name, "청동거울.Scan.PLY")
        self.assertEqual(fingerprint.filename, "청동거울.Scan.PLY")
        self.assertEqual(fingerprint.format, "ply")
        self.assertEqual(fingerprint.extension, ".ply")
        self.assertEqual(fingerprint.id, f"sha256:{fingerprint.sha256}")
        self.assertEqual(fingerprint.identity_scope, "primary_file_bytes")

    def test_empty_file_uses_standard_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "empty"
            source.write_bytes(b"")
            fingerprint = fingerprint_file(source, chunk_size=1)

        self.assertEqual(fingerprint.sha256, hashlib.sha256(b"").hexdigest())
        self.assertEqual(fingerprint.size_bytes, 0)
        self.assertEqual(fingerprint.format, "")
        self.assertEqual(fingerprint.extension, "")

    def test_pre_epoch_mtime_is_a_valid_non_authoritative_hint(self) -> None:
        fingerprint = SourceFingerprint(
            sha256="f" * 64,
            size_bytes=0,
            mtime_ns=-1_000_000_000,
            original_name="historic-scan.ply",
            format="ply",
        )

        self.assertEqual(fingerprint.mtime_ns, -1_000_000_000)
        self.assertEqual(SourceFingerprint.from_dict(fingerprint.to_dict()), fingerprint)

    def test_large_input_is_consumed_in_bounded_chunks(self) -> None:
        chunk_size = DEFAULT_HASH_CHUNK_SIZE
        payload = b"large-source-block" * ((chunk_size * 2) // 18 + 1)
        tracker = _TrackingHash()
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "large.bin"
            source.write_bytes(payload)
            with patch("src.core.source_identity.hashlib.sha256", return_value=tracker):
                fingerprint = fingerprint_file(source, chunk_size=chunk_size)

        self.assertEqual(fingerprint.sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(sum(tracker.update_sizes), len(payload))
        self.assertGreater(len(tracker.update_sizes), 1)
        self.assertLessEqual(max(tracker.update_sizes), chunk_size)

    def test_rejects_invalid_chunk_size(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.ply"
            source.write_bytes(b"mesh")
            for invalid in (0, -1, True):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        fingerprint_file(source, chunk_size=invalid)

    def test_change_during_hashing_raises_explicit_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "changing.ply"
            source.write_bytes(b"0123456789" * 100)
            mutating_hash = _MtimeMutatingHash(source)
            with patch(
                "src.core.source_identity.hashlib.sha256",
                return_value=mutating_hash,
            ):
                with self.assertRaises((SourceChangedError, PermissionError)):
                    fingerprint_file(source, chunk_size=8)

    def test_windows_path_and_descriptor_ctimes_are_not_compared(self) -> None:
        payload = b"same file, incompatible Windows ctime meanings"
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "stable.ply"
            source.write_bytes(payload)
            baseline = source_identity._stat_snapshot(source.stat())
            path_snapshot = replace(baseline, ctime_ns=10)
            descriptor_snapshot = replace(baseline, ctime_ns=20)
            snapshots = (
                path_snapshot,
                descriptor_snapshot,
                descriptor_snapshot,
                path_snapshot,
                descriptor_snapshot,
                path_snapshot,
            )
            with patch.object(
                source_identity,
                "_path_descriptor_ctime_comparable",
                return_value=False,
            ), patch.object(
                source_identity,
                "_stat_snapshot",
                side_effect=snapshots,
            ):
                fingerprint = fingerprint_file(source)

        self.assertEqual(fingerprint.sha256, hashlib.sha256(payload).hexdigest())

    def test_windows_descriptor_ctime_change_is_still_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "changed.ply"
            source.write_bytes(b"descriptor change remains authoritative")
            baseline = source_identity._stat_snapshot(source.stat())
            path_snapshot = replace(baseline, ctime_ns=10)
            opened_snapshot = replace(baseline, ctime_ns=20)
            changed_snapshot = replace(baseline, ctime_ns=21)
            with patch.object(
                source_identity,
                "_path_descriptor_ctime_comparable",
                return_value=False,
            ), patch.object(
                source_identity,
                "_stat_snapshot",
                side_effect=(
                    path_snapshot,
                    opened_snapshot,
                    changed_snapshot,
                ),
            ):
                with self.assertRaises(SourceChangedError):
                    fingerprint_file(source)

    def test_windows_mixed_stat_comparison_keeps_non_ctime_fields(self) -> None:
        baseline = source_identity._StatSnapshot(
            device=1,
            inode=2,
            size_bytes=3,
            mtime_ns=4,
            ctime_ns=5,
        )
        for field_name in ("device", "inode", "size_bytes", "mtime_ns"):
            with self.subTest(field_name=field_name):
                changed = replace(
                    baseline,
                    **{field_name: getattr(baseline, field_name) + 1},
                )
                with self.assertRaises(SourceChangedError):
                    source_identity._raise_if_changed(
                        Path("source.ply"),
                        baseline,
                        changed,
                        compare_ctime=False,
                    )

    def test_missing_source_has_typed_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            known = Path(td) / "known.ply"
            known.write_bytes(b"known")
            expected = fingerprint_file(known)
            missing = Path(td) / "missing.ply"
            result = verify_source(missing, expected)

        self.assertEqual(result.status, SourceVerificationStatus.MISSING)
        self.assertFalse(result.verified)
        self.assertEqual(result.expected, expected)
        self.assertIsNone(result.actual)

    def test_same_size_changed_source_is_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.ply"
            source.write_bytes(b"AAAA")
            expected = fingerprint_file(source)
            source.write_bytes(b"BBBB")
            result = verify_source(source, expected)

        self.assertEqual(result.status, SourceVerificationStatus.MISMATCH)
        self.assertEqual(result.mismatch_fields, ("sha256",))
        self.assertIsNotNone(result.actual)

    def test_relocated_identical_bytes_are_verified_with_relocation_hint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            original = Path(td) / "first" / "artifact.ply"
            original.parent.mkdir()
            original.write_bytes(b"same bytes")
            expected = fingerprint_file(original)

            relocated = Path(td) / "second" / "renamed.PLY"
            relocated.parent.mkdir()
            relocated.write_bytes(b"same bytes")
            result = verify_source(
                relocated,
                expected,
                expected_path_hint=original,
            )

        self.assertEqual(result.status, SourceVerificationStatus.VERIFIED)
        self.assertTrue(result.verified)
        self.assertTrue(result.relocated)
        self.assertIn("original_name", result.hint_differences)
        self.assertNotIn("original_name", result.mismatch_fields)

    def test_compare_fingerprints_does_not_rehash_worker_result(self) -> None:
        expected = SourceFingerprint(
            sha256="a" * 64,
            size_bytes=123,
            mtime_ns=10,
            original_name="old.ply",
            format="ply",
        )
        actual = SourceFingerprint(
            sha256="a" * 64,
            size_bytes=123,
            mtime_ns=20,
            original_name="new.ply",
            format="ply",
        )

        result = compare_fingerprints(
            expected,
            actual,
            checked_path=r"C:\Survey Data\new.ply",
            relocated=True,
        )

        self.assertEqual(result.status, SourceVerificationStatus.VERIFIED)
        self.assertTrue(result.relocated)
        self.assertEqual(result.hint_differences, ("mtime_ns", "original_name"))

    def test_unreadable_source_has_typed_result(self) -> None:
        expected = SourceFingerprint(
            sha256="b" * 64,
            size_bytes=4,
            mtime_ns=1,
            original_name="source.ply",
            format="ply",
        )
        with patch(
            "src.core.source_identity.fingerprint_file",
            side_effect=PermissionError("denied"),
        ):
            result = verify_source("source.ply", expected)

        self.assertEqual(result.status, SourceVerificationStatus.UNREADABLE)
        self.assertIn("PermissionError", result.detail or "")

    def test_fingerprint_and_verification_json_roundtrip(self) -> None:
        expected = SourceFingerprint(
            sha256="c" * 64,
            size_bytes=42,
            mtime_ns=123456,
            original_name="유물 자료.PLY",
            format="PLY",
        )
        actual = SourceFingerprint(
            sha256="c" * 64,
            size_bytes=42,
            mtime_ns=654321,
            original_name="relocated.ply",
            format="ply",
        )
        result = compare_fingerprints(
            expected,
            actual,
            checked_path=r"C:\문화유산 자료\relocated.ply",
            relocated=True,
        )

        encoded = json.dumps(result.to_dict(), ensure_ascii=False)
        restored = SourceVerification.from_dict(json.loads(encoded))

        self.assertEqual(SourceFingerprint.from_dict(expected.to_dict()), expected)
        self.assertEqual(restored, result)
        self.assertEqual(restored.checked_path, r"C:\문화유산 자료\relocated.ply")

    def test_legacy_and_missing_without_fingerprint_are_explicit(self) -> None:
        legacy = legacy_unverified_source(r"D:\legacy\artifact.obj")
        missing = missing_source(r"D:\legacy\missing.obj")

        self.assertEqual(legacy.status, SourceVerificationStatus.LEGACY_UNVERIFIED)
        self.assertIsNone(legacy.expected)
        self.assertEqual(missing.status, SourceVerificationStatus.MISSING)
        self.assertIsNone(missing.expected)
        self.assertEqual(
            SourceVerification.from_dict(legacy.to_dict()),
            legacy,
        )

    def test_serialized_identity_validation_is_strict(self) -> None:
        valid = SourceFingerprint(
            sha256="d" * 64,
            size_bytes=1,
            mtime_ns=2,
            original_name="artifact.obj",
            format="obj",
        ).to_dict()

        invalid_id = dict(valid)
        invalid_id["id"] = f"sha256:{'e' * 64}"
        with self.assertRaises(ValueError):
            SourceFingerprint.from_dict(invalid_id)

        invalid_size = dict(valid)
        invalid_size["size_bytes"] = True
        with self.assertRaises(ValueError):
            SourceFingerprint.from_dict(invalid_size)

        invalid_schema = dict(valid)
        invalid_schema["schema_version"] = True
        with self.assertRaises(ValueError):
            SourceFingerprint.from_dict(invalid_schema)

        for required_field in ("id", "kind", "identity_scope"):
            with self.subTest(required_field=required_field):
                missing_field = dict(valid)
                del missing_field[required_field]
                with self.assertRaises(ValueError):
                    SourceFingerprint.from_dict(missing_field)

        invalid_verification = {
            "schema_version": 1,
            "status": "verified",
            "checked_path": "artifact.obj",
            "expected": valid,
            "actual": None,
            "mismatch_fields": [],
            "hint_differences": [],
            "relocated": False,
            "detail": None,
        }
        with self.assertRaises(ValueError):
            SourceVerification.from_dict(invalid_verification)

    def test_mesh_loader_attaches_raw_identity_before_geometry_mutation(self) -> None:
        obj_bytes = (
            b"v 0 0 0\n"
            b"v 1 0 0\n"
            b"v 0 1 0\n"
            b"f 1 2 3\n"
        )
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "artifact.obj"
            source.write_bytes(obj_bytes)
            mesh = MeshLoader(default_unit="mm").load(source)

        identity = mesh.source_identity
        self.assertIsNotNone(identity)
        assert identity is not None
        self.assertEqual(identity.sha256, hashlib.sha256(obj_bytes).hexdigest())
        self.assertEqual(identity.size_bytes, len(obj_bytes))

        mesh.vertices *= 10.0
        centered = mesh.center_at_origin()
        subset = mesh.extract_submesh(mesh.faces[:1])
        self.assertIs(mesh.source_identity, identity)
        self.assertIs(centered.source_identity, identity)
        self.assertIs(subset.source_identity, identity)
        self.assertEqual(mesh.source_format, "obj")
        self.assertEqual(centered.source_format, "obj")
        self.assertEqual(subset.source_format, "obj")

    def test_mesh_loader_uses_saved_format_hint_for_renamed_identical_source(self) -> None:
        obj_bytes = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        with tempfile.TemporaryDirectory() as td:
            renamed = Path(td) / "artifact-renamed.ply"
            renamed.write_bytes(obj_bytes)
            mesh = MeshLoader(default_unit="mm").load(
                renamed,
                source_format="obj",
            )

        self.assertEqual(mesh.n_vertices, 3)
        self.assertEqual(mesh.n_faces, 1)
        self.assertIsNotNone(mesh.source_identity)
        assert mesh.source_identity is not None
        self.assertEqual(mesh.source_identity.sha256, hashlib.sha256(obj_bytes).hexdigest())
        # The current filename remains a hint; the parser choice came from the
        # saved source identity's original format.
        self.assertEqual(mesh.source_identity.format, "ply")
        self.assertEqual(mesh.source_format, "obj")

    def test_same_descriptor_import_supports_primary_binary_formats(self) -> None:
        source_mesh = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
        with tempfile.TemporaryDirectory() as td:
            for extension in ("ply", "stl", "off", "glb"):
                with self.subTest(extension=extension):
                    path = Path(td) / f"box.{extension}"
                    source_mesh.export(path)
                    loaded = MeshLoader(default_unit="mm").load(path)

                    self.assertGreater(loaded.n_vertices, 0)
                    self.assertGreater(loaded.n_faces, 0)
                    self.assertIsNotNone(loaded.source_identity)
                    self.assertEqual(loaded.source_format, extension)

    def test_mesh_loader_rejects_source_changed_during_parse(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "changing.obj"
            source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")

            def mutate_while_parsing(*_args, **_kwargs):
                current = source.stat()
                os.utime(
                    source,
                    ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000),
                )
                return trimesh.Trimesh(
                    vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    faces=[[0, 1, 2]],
                    process=False,
                )

            with patch("src.core.mesh_loader.trimesh.load", side_effect=mutate_while_parsing):
                with self.assertRaises((SourceChangedError, PermissionError)):
                    MeshLoader(default_unit="mm").load(source)

    @unittest.skipIf(
        os.name == "nt",
        "Windows denies replacement of the open source descriptor; mutation coverage remains active",
    )
    def test_mesh_loader_rejects_same_metadata_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "race.obj"
            original_bytes = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
            replacement_bytes = b"v 9 9 9\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
            self.assertEqual(len(original_bytes), len(replacement_bytes))
            source.write_bytes(original_bytes)
            original_stat = source.stat()

            replacement = Path(td) / "replacement.obj"
            replacement.write_bytes(replacement_bytes)
            os.utime(
                replacement,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )

            def replace_path_while_parsing(source_stream, **_kwargs):
                os.replace(replacement, source)
                self.assertEqual(source_stream.read(), original_bytes)
                source_stream.seek(0)
                return trimesh.Trimesh(
                    vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    faces=[[0, 1, 2]],
                    process=False,
                )

            with patch(
                "src.core.mesh_loader.trimesh.load",
                side_effect=replace_path_while_parsing,
            ):
                with self.assertRaises(SourceChangedError):
                    MeshLoader(default_unit="mm").load(source)


if __name__ == "__main__":
    unittest.main()
