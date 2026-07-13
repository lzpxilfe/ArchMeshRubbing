import copy
from dataclasses import replace
import errno
import hashlib
import importlib
import json
import stat
import struct
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock
import warnings
import zipfile

import numpy as np
from PIL import Image

import src.core.project_file as project_file
from src.core.artifact_document import ArtifactDocument
from src.core.artifact_session import ArtifactSession
from src.core.artifact_verification import build_artifact_verification_report
from src.core.mesh_loader import MeshLoader
from src.core.project_file import (
    ARTIFACT_PAYLOAD_SCHEMA_VERSION,
    ARTIFACT_PAYLOAD_TYPE,
    CHECKSUMS_NAME,
    LEGACY_PROJECT_VERSION,
    MANIFEST_NAME,
    MIGRATION_MARKER_NAME,
    PAYLOAD_SCHEMA_VERSION,
    PAYLOAD_TYPE,
    PROJECT_FORMAT,
    PROJECT_VERSION,
    ArtifactProjectPackage,
    EmbeddedSourceRequiredError,
    ProjectFormatError,
    ProjectSaveError,
    ProjectSerializationError,
    UnsupportedPayloadError,
    UnsupportedProjectVersionError,
    load_artifact_project,
    load_artifact_project_package,
    load_artifact_session_project,
    load_project,
    migrate_project_document,
    save_artifact_project,
    save_artifact_session_project,
    save_project,
)
from src.core.source_bundle import SOURCE_INDEX_NAME, SourceBundleIndex, source_blob_member
from src.core.source_identity import SourceFingerprint


ARTIFACT_GOLDEN_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "projects"
    / "artifact_document_1_0_0.json"
)


def _artifact_document() -> ArtifactDocument:
    return ArtifactDocument.from_json_bytes(ARTIFACT_GOLDEN_PATH.read_bytes())


TEST_PLY_BYTES = textwrap.dedent(
    """\
    ply
    format ascii 1.0
    comment embedded artifact project fixture
    element vertex 4
    property float x
    property float y
    property float z
    element face 4
    property list uchar int vertex_indices
    end_header
    0 0 0
    1 0 0
    0 1 0
    0 0 1
    3 0 2 1
    3 0 1 3
    3 1 2 3
    3 2 0 3
    """
).encode("ascii")


def _artifact_session_from_path(source_path: Path) -> ArtifactSession:
    mesh = MeshLoader(default_unit="mm").load(source_path, unit="cm")
    return ArtifactSession.create_from_source(
        mesh,
        resolved_source_path=str(source_path),
        unit="cm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="0.1.0-test",
        operator="embedded-project-test",
        created_at="2026-07-13T00:00:00Z",
        document_id="artifact:embedded-project-test",
        metadata_revision_id="metadata:embedded-source",
        align_revision_id="align:identity",
    )


def _write_textured_obj(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "fragment.obj"
    source_path.write_text(
        textwrap.dedent(
            """\
            mtllib material.mtl
            v 0 0 0
            v 1 0 0
            v 0 1 0
            vt 0 0
            vt 1 0
            vt 0 1
            usemtl painted
            f 1/1 2/2 3/3
            """
        ),
        encoding="utf-8",
    )
    (directory / "material.mtl").write_text(
        "newmtl painted\nmap_Kd texture.png\n",
        encoding="utf-8",
    )
    Image.new("RGB", (2, 2), color=(12, 34, 56)).save(directory / "texture.png")
    return source_path


def _write_external_buffer_gltf(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    binary = struct.pack(
        "<9f3H",
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0,
        1,
        2,
    )
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary), "uri": "geometry.bin"}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "max": [1, 1, 0],
                "min": [0, 0, 0],
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR",
                "max": [2],
                "min": [0],
            },
        ],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}
                ]
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    source_path = directory / "fragment.gltf"
    source_path.write_text(json.dumps(document), encoding="utf-8")
    (directory / "geometry.bin").write_bytes(binary)
    return source_path


class TestProjectFile(unittest.TestCase):
    def test_roundtrip_zip_manifest(self):
        state = {
            "viewport": {"slice": {"enabled": True, "z": 12.34}},
            "objects": [
                {
                    "name": "Test",
                    "mesh": {"path": "C:/tmp/foo.stl", "source_scale_factor": 0.1},
                    "transform": {"t": [1.0, 2.0, 3.0], "r": [10.0, 20.0, 30.0], "s": 1.5},
                    "faces": {"outer": [1, 2, 3], "inner": [], "migu": []},
                }
            ],
        }
        meta = {"app": "ArchMeshRubbing", "version": "0.0.0"}

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.amr"
            save_project(path, state, meta=meta)
            doc = load_project(path)

        self.assertEqual(doc.get("format"), PROJECT_FORMAT)
        self.assertEqual(doc.get("version"), PROJECT_VERSION)
        self.assertEqual(doc.get("meta"), meta)
        self.assertEqual(doc.get("state"), state)

    def test_load_plain_json_fallback(self):
        state = {"hello": "world"}
        doc = {
            "format": PROJECT_FORMAT,
            "version": PROJECT_VERSION,
            "payload_type": PAYLOAD_TYPE,
            "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
            "meta": {},
            "saved_at": "2026-01-01T00:00:00Z",
            "state": state,
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.json"
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            loaded = load_project(path)

        self.assertEqual(loaded.get("state"), state)

    def test_roundtrip_state_with_tile_slots(self):
        state = {
            "objects": [
                {
                    "name": "Tile",
                    "mesh": {"path": "C:/tmp/tile.obj", "source_scale_factor": 1.0},
                    "faces": {"selected": [1, 2, 3], "outer": [], "inner": [], "migu": []},
                    "tile_interpretation": {
                        "tile_class": "sugkiwa",
                        "split_scheme": "quarter",
                        "record_view": "top",
                        "record_strategy": "canonical_visible",
                        "saved_slots": [
                            {
                                "slot_key": "slot_1",
                                "label": "상면 기록 | 선택 3면",
                                "selected_faces": [1, 2, 3],
                                "tile_class": "sugkiwa",
                                "split_scheme": "quarter",
                                "axis_hint": {
                                    "source": "selected_patch_pca",
                                    "vector_world": [0.0, 1.0, 0.0],
                                    "origin_world": [0.0, 0.0, 0.0],
                                    "confidence": 0.8,
                                    "face_count": 3,
                                    "note": "slot axis",
                                },
                                "section_observations": [],
                                "mandrel_fit": {
                                    "radius_world": 22.5,
                                    "radius_spread_world": 0.6,
                                    "axis_origin_world": [0.0, 0.0, 0.0],
                                    "axis_vector_world": [0.0, 1.0, 0.0],
                                    "confidence": 0.7,
                                    "used_sections": 3,
                                    "used_points": 72,
                                    "scope": "현재 선택 표면",
                                    "note": "slot fit",
                                },
                                "record_view": "top",
                                "record_strategy": "canonical_visible",
                                "workflow_stage": "record_surface",
                                "note": "",
                                "updated_at_iso": "2026-03-20T12:00:00",
                            }
                        ],
                    },
                    "tile_synthetic_truth": {
                        "spec": {
                            "tile_class": "sugkiwa",
                            "split_scheme": "quarter",
                            "length_world": 180.0,
                            "radius_base_world": 65.0,
                            "radius_amplitude_world": 8.0,
                            "theta_span_deg": 0.0,
                            "axial_samples": 20,
                            "angular_samples": 24,
                            "twist_deg": 4.0,
                            "bend_world": 6.0,
                            "axial_slope_world": 2.0,
                            "noise_std_world": 0.0,
                            "thickness_world": 0.0,
                            "seed": 3,
                            "unit": "mm",
                            "record_view": "top",
                            "name": "synthetic_demo",
                        },
                        "ground_truth_state": {"tile_class": "sugkiwa", "split_scheme": "quarter"},
                        "axis_vector_world": [0.0, 1.0, 0.0],
                        "axis_origin_world": [0.0, 0.0, 0.0],
                        "section_stations": [-50.0, 0.0, 50.0],
                        "section_radii_world": [62.0, 68.0, 62.0],
                        "selected_faces": [0, 1, 2],
                        "mesh_name": "synthetic_demo",
                        "note": "ground_truth",
                    },
                    "tile_evaluation_report": {
                        "tile_class_match": True,
                        "split_scheme_match": True,
                        "record_view_match": True,
                        "axis_angle_error_deg": 0.5,
                        "axis_origin_offset_world": 0.0,
                        "section_station_mae_world": 0.2,
                        "section_radius_mae_world": 0.3,
                        "mandrel_radius_abs_error_world": 0.4,
                        "completeness": 1.0,
                        "overall_score": 0.97,
                        "note": "synthetic_evaluation",
                    },
                }
            ]
        }

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tile_slots.amr"
            save_project(path, state, meta={})
            doc = load_project(path)

        loaded_state = doc.get("state", {})
        loaded_slots = (
            loaded_state.get("objects", [{}])[0]
            .get("tile_interpretation", {})
            .get("saved_slots", [])
        )
        self.assertEqual(len(loaded_slots), 1)
        self.assertEqual(loaded_slots[0]["slot_key"], "slot_1")
        self.assertEqual(loaded_slots[0]["selected_faces"], [1, 2, 3])
        self.assertEqual(loaded_slots[0]["record_view"], "top")
        loaded_truth = loaded_state.get("objects", [{}])[0].get("tile_synthetic_truth", {})
        loaded_report = loaded_state.get("objects", [{}])[0].get("tile_evaluation_report", {})
        self.assertEqual(loaded_truth.get("mesh_name"), "synthetic_demo")
        self.assertEqual(loaded_truth.get("selected_faces"), [0, 1, 2])
        self.assertAlmostEqual(float(loaded_report.get("overall_score", 0.0)), 0.97)


class TestProjectFileV2(unittest.TestCase):
    @staticmethod
    def _v2_document(state=None):
        return {
            "format": PROJECT_FORMAT,
            "version": PROJECT_VERSION,
            "payload_type": PAYLOAD_TYPE,
            "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
            "saved_at": "2026-07-11T00:00:00Z",
            "meta": {},
            "state": dict(state or {}),
        }

    @staticmethod
    def _artifact_v2_document(
        document: ArtifactDocument | None = None,
    ) -> dict[str, object]:
        artifact = document or _artifact_document()
        return {
            "format": PROJECT_FORMAT,
            "version": PROJECT_VERSION,
            "payload_type": ARTIFACT_PAYLOAD_TYPE,
            "payload_schema_version": ARTIFACT_PAYLOAD_SCHEMA_VERSION,
            "saved_at": "2026-07-11T00:00:00Z",
            "meta": {},
            "state": artifact.to_dict(),
        }

    @staticmethod
    def _write_json(path: Path, document: dict) -> None:
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _temp_artifacts(destination: Path) -> list[Path]:
        return list(destination.parent.glob(f".{destination.name}.*.tmp"))

    def test_v2_archive_contains_integrity_manifest_and_preserves_source_identity(self):
        source_identity = SourceFingerprint(
            sha256="a" * 64,
            size_bytes=123,
            mtime_ns=456,
            original_name="artifact.ply",
            format="ply",
        )
        state = {
            "objects": [
                {
                    "mesh": {
                        "path": "/data/artifact.ply",
                        "source_scale_factor": 1.0,
                        "source": {
                            "identity": source_identity.to_dict(),
                            "binding_status": "captured_at_import",
                            "parse_format": "ply",
                        },
                    }
                }
            ]
        }

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "identity.amr"
            save_project(path, state)
            with zipfile.ZipFile(path, "r") as zf:
                self.assertEqual(set(zf.namelist()), {MANIFEST_NAME, CHECKSUMS_NAME})
                manifest = zf.read(MANIFEST_NAME)
                checksums = json.loads(zf.read(CHECKSUMS_NAME))
            loaded = load_project(path)

        self.assertEqual(checksums["algorithm"], "sha256")
        self.assertEqual(
            checksums["files"][MANIFEST_NAME],
            hashlib.sha256(manifest).hexdigest(),
        )
        self.assertEqual(loaded["state"], state)
        loaded_source = loaded["state"]["objects"][0]["mesh"]["source"]
        self.assertEqual(
            SourceFingerprint.from_dict(loaded_source["identity"]),
            source_identity,
        )
        self.assertEqual(loaded_source["parse_format"], "ply")

    def test_artifact_document_roundtrip_preserves_canonical_payload_and_legacy_api_boundary(self):
        artifact = _artifact_document()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.amr"
            save_artifact_project(path, artifact, meta={"app": "ArchMeshRubbing"})

            with zipfile.ZipFile(path, "r") as zf:
                manifest = json.loads(zf.read(MANIFEST_NAME))
                checksums = json.loads(zf.read(CHECKSUMS_NAME))
            restored = load_artifact_project(path)
            with self.assertRaises(UnsupportedPayloadError):
                load_project(path)

        self.assertEqual(manifest["payload_type"], ARTIFACT_PAYLOAD_TYPE)
        self.assertEqual(
            manifest["payload_schema_version"],
            ARTIFACT_PAYLOAD_SCHEMA_VERSION,
        )
        self.assertEqual(manifest["state"], artifact.to_dict())
        self.assertEqual(
            checksums["files"][MANIFEST_NAME],
            hashlib.sha256(
                (json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        self.assertEqual(restored, artifact)
        self.assertEqual(restored.canonical_json_bytes(), artifact.canonical_json_bytes())

    def test_artifact_save_reopens_staged_archive_through_artifact_loader(self):
        artifact = _artifact_document()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "staged-reopen.amr"
            with mock.patch.object(
                project_file,
                "_load_zip_document",
                wraps=project_file._load_zip_document,
            ) as load_zip:
                save_artifact_project(path, artifact)

            self.assertEqual(load_zip.call_count, 1)
            _staged_path = load_zip.call_args.args[0]
            self.assertNotEqual(Path(_staged_path), path)
            self.assertEqual(
                load_zip.call_args.kwargs,
                {
                    "expected_payload_type": ARTIFACT_PAYLOAD_TYPE,
                    "expected_schema_version": ARTIFACT_PAYLOAD_SCHEMA_VERSION,
                },
            )
            self.assertEqual(load_artifact_project(path), artifact)

    def test_artifact_atomic_validation_failure_preserves_previous_destination(self):
        original = _artifact_document()
        replacement = replace(original, software_version="0.2.0")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact-existing.amr"
            save_artifact_project(path, original)
            previous = path.read_bytes()

            with mock.patch.object(
                project_file,
                "_validate_staged_project",
                side_effect=ProjectFormatError("injected artifact validation failure"),
            ):
                with self.assertRaises(ProjectSaveError) as raised:
                    save_artifact_project(path, replacement)

            self.assertEqual(raised.exception.stage, "validation")
            self.assertFalse(raised.exception.committed)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(self._temp_artifacts(path), [])

    def test_invalid_artifact_is_rejected_before_commit_and_on_load(self):
        original = _artifact_document()
        invalid = _artifact_document()
        object.__setattr__(invalid, "active_align_revision_id", "align:missing")

        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            path = directory / "artifact-existing.amr"
            save_artifact_project(path, original)
            previous = path.read_bytes()

            with self.assertRaisesRegex(ProjectSerializationError, "active align revision"):
                save_artifact_project(path, invalid)

            invalid_json = directory / "invalid-artifact.json"
            self._write_json(invalid_json, self._artifact_v2_document(invalid))
            with self.assertRaisesRegex(ProjectFormatError, "active align revision"):
                load_artifact_project(invalid_json)

            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(self._temp_artifacts(path), [])

    def test_newer_artifact_payload_major_is_typed_read_only_rejection(self):
        document = self._artifact_v2_document()
        document["payload_schema_version"] = "2.0.0"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "future-artifact.json"
            self._write_json(path, document)
            with self.assertRaises(UnsupportedPayloadError) as raised:
                load_artifact_project(path)

        error = raised.exception
        self.assertTrue(error.newer)
        self.assertTrue(error.read_only_inspection)
        self.assertEqual(error.payload_type, ARTIFACT_PAYLOAD_TYPE)
        self.assertEqual(error.payload_schema_version, "2.0.0")
        self.assertNotIn("state", error.inspection)

    def test_v1_migration_is_pure_deterministic_idempotent_and_generic(self):
        source_identity = {
            "status": "legacy_unverified",
            "path_hint": "artifact.obj",
            "future_extension": {"keep": [1, 2, 3]},
        }
        legacy = {
            "format": PROJECT_FORMAT,
            "version": LEGACY_PROJECT_VERSION,
            "saved_at": "2025-01-02T03:04:05Z",
            "meta": {"legacy": True},
            "state": {
                "objects": [
                    {
                        "alignment": {"status": "draft_v1_value_must_not_be_trusted"},
                        "mesh": {
                            "path": "artifact.obj",
                            "source_scale_factor": 0.1,
                            "source_identity": source_identity,
                        }
                    }
                ],
                "unknown_payload_field": {"preserve": True},
            },
            "unknown_envelope_field": {"preserve": "also"},
        }
        before = copy.deepcopy(legacy)

        migrated_once = migrate_project_document(legacy)
        migrated_twice = migrate_project_document(migrated_once)

        self.assertEqual(legacy, before)
        self.assertEqual(migrated_once, migrated_twice)
        self.assertEqual(migrated_once["version"], PROJECT_VERSION)
        self.assertEqual(migrated_once["payload_type"], PAYLOAD_TYPE)
        self.assertEqual(
            migrated_once["payload_schema_version"],
            PAYLOAD_SCHEMA_VERSION,
        )
        self.assertEqual(
            migrated_once[MIGRATION_MARKER_NAME],
            {
                "from_version": LEGACY_PROJECT_VERSION,
                "to_version": PROJECT_VERSION,
                "status": "legacy_unverified",
                "runtime_only": True,
                "requires_save_as": True,
            },
        )
        self.assertEqual(
            migrated_once["state"]["objects"][0]["mesh"]["source_identity"],
            source_identity,
        )
        self.assertEqual(
            migrated_once["state"]["objects"][0]["mesh"]["source"],
            {"identity": None, "binding_status": "legacy_unverified"},
        )
        self.assertEqual(
            migrated_once["state"]["objects"][0]["alignment"],
            {"status": "legacy_unverifiable"},
        )
        self.assertEqual(
            migrated_once["state"]["objects"][0]["mesh"]["path"],
            "artifact.obj",
        )
        self.assertEqual(
            migrated_once["state"]["objects"][0]["mesh"]["source_scale_factor"],
            0.1,
        )
        self.assertEqual(
            migrated_once["unknown_envelope_field"],
            {"preserve": "also"},
        )

    def test_loads_v1_amr_without_v2_checksums_and_migrates(self):
        legacy = {
            "format": PROJECT_FORMAT,
            "version": LEGACY_PROJECT_VERSION,
            "saved_at": "2025-01-01T00:00:00Z",
            "state": {"legacy": True},
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "legacy.amr"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(MANIFEST_NAME, json.dumps(legacy).encode("utf-8"))
            loaded = load_project(path)

        self.assertEqual(loaded["version"], PROJECT_VERSION)
        self.assertEqual(loaded["meta"], {})
        self.assertEqual(loaded["state"], {"legacy": True})
        self.assertEqual(loaded[MIGRATION_MARKER_NAME]["requires_save_as"], True)

    def test_explicit_v1_json_import_migrates_but_amr_never_falls_back_to_json(self):
        legacy = {
            "format": PROJECT_FORMAT,
            "version": LEGACY_PROJECT_VERSION,
            "saved_at": "2025-01-01T00:00:00Z",
            "meta": {},
            "state": {"legacy": True},
        }
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            json_path = directory / "legacy.json"
            amr_path = directory / "not-a-container.amr"
            self._write_json(json_path, legacy)
            self._write_json(amr_path, legacy)

            loaded = load_project(json_path)
            with self.assertRaises(ProjectFormatError):
                load_project(amr_path)

        self.assertEqual(loaded["version"], PROJECT_VERSION)
        self.assertEqual(loaded["state"], {"legacy": True})

    def test_truncated_zip_is_not_misidentified_as_json(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "truncated.amr"
            path.write_bytes(b"PK\x03\x04truncated")
            with self.assertRaises(ProjectFormatError):
                load_project(path)

    def test_strict_json_rejects_non_finite_values_on_save_without_touching_destination(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as td:
                path = Path(td) / "existing.amr"
                previous = b"previous-project-bytes"
                path.write_bytes(previous)

                with self.assertRaises(ProjectSerializationError):
                    save_project(path, {"invalid": value})

                self.assertEqual(path.read_bytes(), previous)
                self.assertEqual(self._temp_artifacts(path), [])

    def test_strict_json_rejects_non_string_mapping_keys_before_creating_temp(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "existing.amr"
            previous = b"previous-project-bytes"
            path.write_bytes(previous)

            with self.assertRaisesRegex(ProjectSerializationError, "expected string"):
                save_project(path, {"nested": {1: "silently coercible otherwise"}})

            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(self._temp_artifacts(path), [])

    def test_native_v2_save_has_no_runtime_migration_marker(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "native.amr"
            save_project(path, {"native": True})
            loaded = load_project(path)

        self.assertNotIn(MIGRATION_MARKER_NAME, loaded)

    def test_runtime_migration_marker_is_not_written_by_durable_v2_save(self):
        legacy = {
            "format": PROJECT_FORMAT,
            "version": LEGACY_PROJECT_VERSION,
            "saved_at": "2025-01-01T00:00:00Z",
            "meta": {"legacy": True},
            "state": {"objects": [{"mesh": {"path": "artifact.ply"}}]},
        }
        migrated = migrate_project_document(legacy)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "migrated-v2.amr"
            save_project(path, migrated["state"], meta=migrated["meta"])
            reloaded = load_project(path)

        self.assertIn(MIGRATION_MARKER_NAME, migrated)
        self.assertNotIn(MIGRATION_MARKER_NAME, reloaded)
        self.assertEqual(
            reloaded["state"]["objects"][0]["mesh"]["source"],
            {"identity": None, "binding_status": "legacy_unverified"},
        )

    def test_strict_json_rejects_non_finite_constants_and_duplicate_keys_on_load(self):
        samples = (
            '{"format":"archmeshrubbing_project","version":1,"state":{"x":NaN}}',
            '{"format":"archmeshrubbing_project","version":1,"state":{"x":1e9999}}',
            '{"format":"archmeshrubbing_project","version":1,"state":{},"state":{}}',
        )
        for index, raw in enumerate(samples):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as td:
                path = Path(td) / "invalid.json"
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(ProjectFormatError):
                    load_project(path)

    def test_strict_json_wraps_excessive_depth_and_integer_range_errors(self):
        prefix = '{"format":"archmeshrubbing_project","version":1,"state":{"x":'
        suffix = "}}"
        samples = (
            prefix + ("[" * 1500) + "0" + ("]" * 1500) + suffix,
            prefix + ("9" * 5000) + suffix,
        )
        for index, raw in enumerate(samples):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as td:
                path = Path(td) / "unsafe.json"
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(ProjectFormatError):
                    load_project(path)

    def test_v2_rejects_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "corrupt.amr"
            save_project(path, {"value": "original"})
            with zipfile.ZipFile(path, "r") as zf:
                checksums = zf.read(CHECKSUMS_NAME)

            changed = self._v2_document({"value": "tampered"})
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(MANIFEST_NAME, json.dumps(changed).encode("utf-8"))
                zf.writestr(CHECKSUMS_NAME, checksums)

            with self.assertRaisesRegex(ProjectFormatError, "Checksum mismatch"):
                load_project(path)

    def test_duplicate_zip_manifest_is_rejected(self):
        legacy = {
            "format": PROJECT_FORMAT,
            "version": LEGACY_PROJECT_VERSION,
            "saved_at": "2025-01-01T00:00:00Z",
            "meta": {},
            "state": {},
        }
        raw = json.dumps(legacy).encode("utf-8")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "duplicate.amr"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(path, "w") as zf:
                    zf.writestr(MANIFEST_NAME, raw)
                    zf.writestr(MANIFEST_NAME, raw)
            with self.assertRaisesRegex(ProjectFormatError, "Duplicate ZIP member"):
                load_project(path)

    def test_member_count_is_rejected_before_zipfile_allocates_infos(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "many-members.amr"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
                for index in range(65):
                    zf.writestr(f"empty-{index}.txt", b"")

            with mock.patch(
                "src.core.project_file.zipfile.ZipFile",
                side_effect=AssertionError("ZipFile constructor must not be reached"),
            ):
                with self.assertRaisesRegex(ProjectFormatError, "too many members"):
                    load_project(path)

    def test_newer_container_major_is_typed_read_only_rejection(self):
        document = self._v2_document({"must_not_execute": True})
        document["version"] = PROJECT_VERSION + 1
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "future.json"
            self._write_json(path, document)
            with self.assertRaises(UnsupportedProjectVersionError) as raised:
                load_project(path)

        error = raised.exception
        self.assertTrue(error.newer)
        self.assertTrue(error.read_only_inspection)
        self.assertEqual(error.found_version, PROJECT_VERSION + 1)
        self.assertNotIn("state", error.inspection)
        self.assertEqual(error.inspection["read_only"], True)

    def test_newer_payload_major_is_typed_read_only_rejection(self):
        document = self._v2_document({"must_not_execute": True})
        document["payload_schema_version"] = "2.0.0"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "future-payload.json"
            self._write_json(path, document)
            with self.assertRaises(UnsupportedPayloadError) as raised:
                load_project(path)

        self.assertTrue(raised.exception.newer)
        self.assertTrue(raised.exception.read_only_inspection)
        self.assertNotIn("state", raised.exception.inspection)

    def test_native_v2_rejects_runtime_migration_marker(self):
        document = self._v2_document({"must_not_bypass_source_verification": True})
        document[MIGRATION_MARKER_NAME] = {
            "from_version": LEGACY_PROJECT_VERSION,
            "to_version": PROJECT_VERSION,
            "status": "legacy_unverified",
            "runtime_only": True,
            "requires_save_as": True,
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "forged-marker.json"
            self._write_json(path, document)
            with self.assertRaisesRegex(ProjectFormatError, "runtime-only"):
                load_project(path)

    def test_save_project_rejects_json_destination(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "project.json"
            with self.assertRaises(ProjectSaveError) as raised:
                save_project(path, {"state": "would be a ZIP"})

            self.assertEqual(raised.exception.stage, "prepare")
            self.assertFalse(raised.exception.retryable)
            self.assertFalse(path.exists())

    def test_atomic_failures_preserve_destination_and_remove_same_dir_temp(self):
        failure_cases = (
            ("temp_write", "_write_zip_archive", OSError("injected write failure")),
            ("temp_fsync", "_flush_and_fsync", OSError("injected fsync failure")),
            (
                "validation",
                "_validate_staged_project",
                ProjectFormatError("injected validation failure"),
            ),
            ("replace", "os.replace", OSError("injected replace failure")),
        )
        for expected_stage, target, error in failure_cases:
            with self.subTest(stage=expected_stage), tempfile.TemporaryDirectory() as td:
                path = Path(td) / "existing.amr"
                save_project(path, {"original": True})
                previous = path.read_bytes()

                patch_target = (
                    mock.patch("src.core.project_file.os.replace", side_effect=error)
                    if target == "os.replace"
                    else mock.patch.object(project_file, target, side_effect=error)
                )
                with patch_target:
                    with self.assertRaises(ProjectSaveError) as raised:
                        save_project(path, {"replacement": expected_stage})

                self.assertEqual(raised.exception.stage, expected_stage)
                self.assertFalse(raised.exception.committed)
                self.assertEqual(path.read_bytes(), previous)
                self.assertEqual(self._temp_artifacts(path), [])

    def test_directory_fsync_is_best_effort_after_successful_replace(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "project.amr"
            with mock.patch(
                "src.core.project_file.os.open",
                side_effect=OSError(errno.EINVAL, "unsupported directory fsync"),
            ):
                project_file._best_effort_fsync_directory(path.parent)

            result = save_project(path, {"committed": True})

            self.assertEqual(result, str(path))
            self.assertEqual(load_project(path)["state"], {"committed": True})

    def test_directory_fsync_io_failure_reports_committed_but_uncertain(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "project.amr"
            save_project(path, {"before": True})

            with mock.patch.object(
                project_file,
                "_best_effort_fsync_directory",
                side_effect=OSError(errno.EIO, "injected directory I/O failure"),
            ):
                with self.assertRaises(ProjectSaveError) as raised:
                    save_project(path, {"after": True})

            self.assertEqual(raised.exception.stage, "directory_fsync")
            self.assertTrue(raised.exception.committed)
            self.assertEqual(load_project(path)["state"], {"after": True})


class TestEmbeddedArtifactProject(unittest.TestCase):
    @staticmethod
    def _temp_artifacts(destination: Path) -> list[Path]:
        return list(destination.parent.glob(f".{destination.name}.*.tmp"))

    @staticmethod
    def _read_archive(path: Path) -> dict[str, bytes]:
        with zipfile.ZipFile(path, "r") as zf:
            return {name: zf.read(name) for name in zf.namelist()}

    @staticmethod
    def _rewrite_archive(path: Path, files: dict[str, bytes]) -> None:
        payloads = dict(files)
        payloads.pop(CHECKSUMS_NAME, None)
        checksums = {
            "algorithm": "sha256",
            "files": {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in payloads.items()
            },
        }
        with zipfile.ZipFile(path, "w", allowZip64=True) as zf:
            for name, payload in payloads.items():
                compress_type = (
                    zipfile.ZIP_STORED
                    if name.startswith("sources/blobs/sha256/")
                    else zipfile.ZIP_DEFLATED
                )
                zf.writestr(name, payload, compress_type=compress_type)
            zf.writestr(
                CHECKSUMS_NAME,
                json.dumps(
                    checksums,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                ).encode("utf-8")
                + b"\n",
                compress_type=zipfile.ZIP_DEFLATED,
            )

    def test_session_package_has_exact_four_members_and_closed_checksums(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source_path = directory / "source" / "fragment.ply"
            source_path.parent.mkdir()
            source_path.write_bytes(TEST_PLY_BYTES)
            session = _artifact_session_from_path(source_path)
            project_path = directory / "artifact.amr"

            save_artifact_session_project(project_path, session)
            source_asset = session.document.source_assets[0]
            blob_member = source_blob_member(source_asset.sha256)
            with zipfile.ZipFile(project_path, "r") as zf:
                infos = {info.filename: info for info in zf.infolist()}
                checksums = json.loads(zf.read(CHECKSUMS_NAME))
                manifest = json.loads(zf.read(MANIFEST_NAME))
                source_index_bytes = zf.read(SOURCE_INDEX_NAME)
                blob_bytes = zf.read(blob_member)

            self.assertEqual(
                set(infos),
                {MANIFEST_NAME, CHECKSUMS_NAME, SOURCE_INDEX_NAME, blob_member},
            )
            self.assertEqual(
                set(checksums["files"]),
                {MANIFEST_NAME, SOURCE_INDEX_NAME, blob_member},
            )
            self.assertEqual(infos[SOURCE_INDEX_NAME].compress_type, zipfile.ZIP_DEFLATED)
            self.assertEqual(infos[blob_member].compress_type, zipfile.ZIP_STORED)
            self.assertEqual(blob_bytes, TEST_PLY_BYTES)
            self.assertEqual(manifest["state"], session.document.to_dict())
            self.assertEqual(
                checksums["files"][blob_member],
                hashlib.sha256(TEST_PLY_BYTES).hexdigest(),
            )
            self.assertEqual(
                checksums["files"][SOURCE_INDEX_NAME],
                hashlib.sha256(source_index_bytes).hexdigest(),
            )

            package = load_artifact_project_package(project_path)
            self.assertIsInstance(package, ArtifactProjectPackage)
            self.assertEqual(package.document, session.document)
            self.assertEqual(
                package.document.canonical_json_bytes(),
                session.document.canonical_json_bytes(),
            )
            self.assertEqual(
                package.source_bundle,
                SourceBundleIndex.for_document(session.document),
            )

    def test_unified_offline_report_materializes_embedded_project_source(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source_path = directory / "source" / "유물.ply"
            source_path.parent.mkdir()
            source_path.write_bytes(TEST_PLY_BYTES)
            session = _artifact_session_from_path(source_path)
            project_path = directory / "record.amr"
            save_artifact_session_project(project_path, session)
            source_path.unlink()

            report = build_artifact_verification_report(project_path)
            repeated = build_artifact_verification_report(project_path)

            self.assertEqual(report, repeated)
            self.assertTrue(report["ok"])
            self.assertEqual(report["artifact_kind"], "project")
            self.assertEqual(report["authority"], "self_contained")
            evidence = report["evidence"]
            self.assertIsInstance(evidence, dict)
            assert isinstance(evidence, dict)
            self.assertTrue(evidence["embedded_source_materialized"])
            self.assertEqual(
                evidence["document_sha256"],
                session.document.canonical_sha256,
            )
            self.assertEqual(
                evidence["source"]["sha256"],
                hashlib.sha256(TEST_PLY_BYTES).hexdigest(),
            )
            self.assertEqual(evidence["geometry"]["vertex_count"], 4)
            self.assertEqual(evidence["geometry"]["face_count"], 4)
            serialized = json.dumps(report, ensure_ascii=False)
            self.assertNotIn(str(directory), serialized)

            jsonschema = importlib.import_module("jsonschema")
            schema = json.loads(
                (
                    Path(__file__).resolve().parents[1]
                    / "schemas/offline_verification_report-1.0.0.schema.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                list(jsonschema.Draft202012Validator(schema).iter_errors(report)),
                [],
            )

            manifest_only = directory / "manifest-only.amr"
            save_artifact_project(manifest_only, session.document)
            failure = build_artifact_verification_report(manifest_only)
            self.assertFalse(failure["ok"])
            self.assertEqual(failure["artifact_kind"], "project")
            self.assertEqual(failure["error"]["code"], "verification_failed")

    def test_manifest_only_artifact_remains_compatible_but_session_load_is_typed(self):
        artifact = _artifact_document()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest-only.amr"
            save_artifact_project(path, artifact)
            package = load_artifact_project_package(path)

            self.assertEqual(package.document, artifact)
            self.assertIsNone(package.source_bundle)
            self.assertEqual(load_artifact_project(path), artifact)
            with self.assertRaises(EmbeddedSourceRequiredError):
                load_artifact_session_project(path)

    def test_save_rejects_changed_external_source_without_touching_destination(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source_path = directory / "fragment.ply"
            source_path.write_bytes(TEST_PLY_BYTES)
            session = _artifact_session_from_path(source_path)
            project_path = directory / "existing.amr"
            save_artifact_session_project(project_path, session)
            previous = project_path.read_bytes()

            changed = bytearray(TEST_PLY_BYTES)
            changed[-2] = ord("2") if changed[-2] != ord("2") else ord("3")
            source_path.write_bytes(bytes(changed))
            with self.assertRaises(ProjectSaveError) as raised:
                save_artifact_session_project(project_path, session)

            self.assertEqual(raised.exception.stage, "source_verification")
            self.assertFalse(raised.exception.committed)
            self.assertEqual(project_path.read_bytes(), previous)
            self.assertEqual(self._temp_artifacts(project_path), [])

    def test_staged_package_validation_failure_is_atomic(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source_path = directory / "fragment.ply"
            source_path.write_bytes(TEST_PLY_BYTES)
            session = _artifact_session_from_path(source_path)
            project_path = directory / "existing.amr"
            save_artifact_session_project(project_path, session)
            previous = project_path.read_bytes()

            with mock.patch.object(
                project_file,
                "load_artifact_project_package",
                side_effect=ProjectFormatError("injected package validation failure"),
            ):
                with self.assertRaises(ProjectSaveError) as raised:
                    save_artifact_session_project(project_path, session)

            self.assertEqual(raised.exception.stage, "validation")
            self.assertFalse(raised.exception.committed)
            self.assertEqual(project_path.read_bytes(), previous)
            self.assertEqual(self._temp_artifacts(project_path), [])

    def test_staged_offline_materialization_failure_is_atomic(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source_path = directory / "fragment.ply"
            source_path.write_bytes(TEST_PLY_BYTES)
            session = _artifact_session_from_path(source_path)
            project_path = directory / "existing.amr"
            save_artifact_session_project(project_path, session)
            previous = project_path.read_bytes()

            with mock.patch.object(
                project_file,
                "_materialize_artifact_project_package",
                side_effect=ProjectFormatError(
                    "injected embedded parser/materialization failure"
                ),
            ):
                with self.assertRaises(ProjectSaveError) as raised:
                    save_artifact_session_project(project_path, session)

            self.assertEqual(raised.exception.stage, "validation")
            self.assertFalse(raised.exception.committed)
            self.assertEqual(project_path.read_bytes(), previous)
            self.assertEqual(self._temp_artifacts(project_path), [])

    def test_mutated_session_geometry_is_rejected_before_temp_creation(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source_path = directory / "fragment.ply"
            source_path.write_bytes(TEST_PLY_BYTES)
            session = _artifact_session_from_path(source_path)
            project_path = directory / "existing.amr"
            previous = b"previous-destination"
            project_path.write_bytes(previous)

            changed_vertices = np.asarray(session.source_mesh.vertices).copy()
            changed_vertices[0, 0] += 1.0
            session.source_mesh.vertices = changed_vertices
            with self.assertRaisesRegex(
                ProjectSerializationError,
                "source/document binding",
            ):
                save_artifact_session_project(project_path, session)

            self.assertEqual(project_path.read_bytes(), previous)
            self.assertEqual(self._temp_artifacts(project_path), [])

    def test_save_requires_amr_and_never_replaces_external_source_alias(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source_path = directory / "fragment.ply"
            source_path.write_bytes(TEST_PLY_BYTES)
            session = _artifact_session_from_path(source_path)

            with self.assertRaisesRegex(ProjectSaveError, r"require an \.amr") as raised:
                save_artifact_session_project(source_path, session)
            self.assertEqual(raised.exception.stage, "prepare")
            self.assertFalse(raised.exception.retryable)
            self.assertEqual(source_path.read_bytes(), TEST_PLY_BYTES)

            alias_path = directory / "source-alias.amr"
            try:
                alias_path.hardlink_to(source_path)
            except OSError as exc:
                self.skipTest(f"hard links unavailable: {exc}")
            with self.assertRaisesRegex(ProjectSaveError, "external source") as raised:
                save_artifact_session_project(alias_path, session)
            self.assertEqual(raised.exception.stage, "prepare")
            self.assertFalse(raised.exception.retryable)
            self.assertEqual(alias_path.read_bytes(), TEST_PLY_BYTES)
            self.assertEqual(source_path.read_bytes(), TEST_PLY_BYTES)
            self.assertEqual(self._temp_artifacts(alias_path), [])

    def test_visual_sidecar_loss_fails_before_destination_replace(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source_path = directory / "fragment.ply"
            source_path.write_bytes(TEST_PLY_BYTES)
            session = _artifact_session_from_path(source_path)
            session.source_mesh.uv_coords = np.zeros(
                (session.source_mesh.vertices.shape[0], 2),
                dtype=np.float64,
            )
            session.source_mesh.texture = np.zeros((2, 2, 3), dtype=np.uint8)
            project_path = directory / "existing.amr"
            previous = b"previous-destination"
            project_path.write_bytes(previous)

            with self.assertRaisesRegex(ProjectSaveError, "sidecar") as raised:
                save_artifact_session_project(project_path, session)

            self.assertEqual(raised.exception.stage, "validation")
            self.assertFalse(raised.exception.committed)
            self.assertEqual(project_path.read_bytes(), previous)
            self.assertEqual(self._temp_artifacts(project_path), [])

    def test_changed_captured_dependency_fails_before_destination_replace(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source_dir = directory / "source"
            source_path = _write_textured_obj(source_dir)
            session = _artifact_session_from_path(source_path)
            project_path = directory / "existing.amr"
            previous = b"previous-destination"
            project_path.write_bytes(previous)
            Image.new("RGB", (2, 2), color=(255, 0, 0)).save(
                source_dir / "texture.png"
            )

            with self.assertRaises(ProjectSaveError) as raised:
                save_artifact_session_project(project_path, session)

            self.assertEqual(raised.exception.stage, "source_verification")
            self.assertFalse(raised.exception.committed)
            self.assertEqual(project_path.read_bytes(), previous)
            self.assertEqual(self._temp_artifacts(project_path), [])

    def test_corrupt_dependency_blob_fails_closed_even_with_rewritten_checksums(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source_path = _write_textured_obj(directory / "source")
            session = _artifact_session_from_path(source_path)
            project_path = directory / "base.amr"
            save_artifact_session_project(project_path, session)
            files = self._read_archive(project_path)
            source_index = json.loads(files[SOURCE_INDEX_NAME])
            dependency = next(
                entry
                for entry in source_index["entries"]
                if entry["role"] == "import_dependency"
            )
            dependency_member = dependency["member"]
            changed = bytearray(files[dependency_member])
            changed[0] ^= 1
            files[dependency_member] = bytes(changed)
            corrupt_path = directory / "corrupt-dependency.amr"
            self._rewrite_archive(corrupt_path, files)

            with self.assertRaisesRegex(ProjectFormatError, "SHA-256"):
                load_artifact_project_package(corrupt_path)

    def test_corrupt_missing_and_orphan_source_blobs_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source_path = directory / "fragment.ply"
            source_path.write_bytes(TEST_PLY_BYTES)
            session = _artifact_session_from_path(source_path)
            source_asset = session.document.source_assets[0]
            blob_member = source_blob_member(source_asset.sha256)

            base_path = directory / "base.amr"
            save_artifact_session_project(base_path, session)
            original_files = self._read_archive(base_path)

            corrupt_path = directory / "corrupt.amr"
            corrupt_files = dict(original_files)
            corrupt_blob = bytearray(corrupt_files[blob_member])
            corrupt_blob[-2] ^= 1
            corrupt_files[blob_member] = bytes(corrupt_blob)
            self._rewrite_archive(corrupt_path, corrupt_files)
            with self.assertRaisesRegex(ProjectFormatError, "SHA-256"):
                load_artifact_project_package(corrupt_path)

            missing_path = directory / "missing.amr"
            missing_files = dict(original_files)
            missing_files.pop(blob_member)
            self._rewrite_archive(missing_path, missing_files)
            with self.assertRaisesRegex(ProjectFormatError, "missing blob"):
                load_artifact_project_package(missing_path)

            orphan_bytes = b"orphan-source-bytes"
            orphan_member = source_blob_member(hashlib.sha256(orphan_bytes).hexdigest())
            orphan_path = directory / "orphan.amr"
            orphan_files = dict(original_files)
            orphan_files[orphan_member] = orphan_bytes
            self._rewrite_archive(orphan_path, orphan_files)
            with self.assertRaisesRegex(ProjectFormatError, "orphan blob"):
                load_artifact_project_package(orphan_path)

    def test_session_materializes_after_external_source_is_deleted(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source_path = directory / "fragment.ply"
            source_path.write_bytes(TEST_PLY_BYTES)
            original = _artifact_session_from_path(source_path).commit_preview(
                translation_mm=(3.0, -2.0, 1.0),
                rotation_deg=(10.0, 20.0, 30.0),
                scale=1.0,
                operator="embedded-project-test",
                created_at="2026-07-13T00:00:01Z",
                revision_id="align:nontrivial",
            )
            expected_vertices = np.asarray(
                original.materialize().mesh.vertices,
                dtype=np.float64,
            ).copy()
            project_path = directory / "offline.amr"
            save_artifact_session_project(project_path, original)
            source_path.unlink()

            with mock.patch.object(
                project_file,
                "_read_zip_member",
                wraps=project_file._read_zip_member,
            ) as read_member:
                restored = load_artifact_session_project(project_path)

            self.assertFalse(source_path.exists())
            self.assertEqual(restored.document, original.document)
            self.assertIn("!/sources/blobs/sha256/", restored.resolved_source_path)
            np.testing.assert_allclose(
                restored.materialize().mesh.vertices,
                expected_vertices,
                rtol=0.0,
                atol=1e-12,
            )
            blob_member = source_blob_member(original.document.source_assets[0].sha256)
            self.assertNotIn(blob_member, [call.args[1] for call in read_member.call_args_list])

            resave_path = directory / "resave.amr"
            save_artifact_session_project(resave_path, restored)
            resaved = load_artifact_session_project(resave_path)
            self.assertEqual(resaved.document, restored.document)
            self.assertEqual(
                resaved.document.source_assets,
                restored.document.source_assets,
            )

            # Save must also work over the archive that currently owns the
            # embedded source. The old archive is closed before os.replace so
            # this remains valid on Windows.
            save_artifact_session_project(resave_path, resaved)
            same_path = load_artifact_session_project(resave_path)
            self.assertEqual(same_path.document, restored.document)
            np.testing.assert_allclose(
                same_path.materialize().mesh.vertices,
                expected_vertices,
                rtol=0.0,
                atol=1e-12,
            )

    def test_obj_sidecar_closure_materializes_offline_and_resaves(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source_dir = directory / "source"
            source_dir.mkdir()
            source_path = source_dir / "fragment.obj"
            source_path.write_text(
                textwrap.dedent(
                    """\
                    mtllib material.mtl
                    v 0 0 0
                    v 1 0 0
                    v 0 1 0
                    vt 0 0
                    vt 1 0
                    vt 0 1
                    usemtl painted
                    f 1/1 2/2 3/3
                    """
                ),
                encoding="utf-8",
            )
            (source_dir / "material.mtl").write_text(
                "newmtl painted\nmap_Kd texture.png\n",
                encoding="utf-8",
            )
            Image.new("RGB", (2, 2), color=(12, 34, 56)).save(
                source_dir / "texture.png"
            )
            session = _artifact_session_from_path(source_path)
            self.assertEqual(
                session.document.geometry_revisions[0].import_recipe[
                    "dependency_policy"
                ],
                "closed_manifest",
            )
            expected_texture = np.asarray(session.source_mesh.texture).copy()
            project_path = directory / "portable.amr"

            save_artifact_session_project(project_path, session)
            with zipfile.ZipFile(project_path, "r") as zf:
                source_index = json.loads(zf.read(SOURCE_INDEX_NAME))
                source_members = {
                    name
                    for name in zf.namelist()
                    if name.startswith("sources/blobs/sha256/")
                }
            self.assertEqual(source_index["schema_version"], "2.0.0")
            self.assertEqual(len(source_index["entries"]), 3)
            self.assertEqual(len(source_members), 3)

            source_path.unlink()
            (source_dir / "material.mtl").unlink()
            (source_dir / "texture.png").unlink()
            restored = load_artifact_session_project(project_path)

            self.assertEqual(restored.document, session.document)
            np.testing.assert_array_equal(restored.source_mesh.texture, expected_texture)
            self.assertEqual(len(restored.source_mesh.source_resources), 3)
            resaved_path = directory / "portable-resaved.amr"
            save_artifact_session_project(resaved_path, restored)
            resaved = load_artifact_session_project(resaved_path)
            self.assertEqual(resaved.document, restored.document)
            np.testing.assert_array_equal(resaved.source_mesh.texture, expected_texture)

    def test_gltf_external_buffer_materializes_offline(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source_dir = directory / "source"
            source_path = _write_external_buffer_gltf(source_dir)
            session = _artifact_session_from_path(source_path)
            expected_vertices = np.asarray(session.source_mesh.vertices).copy()
            project_path = directory / "portable-gltf.amr"

            save_artifact_session_project(project_path, session)
            source_path.unlink()
            (source_dir / "geometry.bin").unlink()
            restored = load_artifact_session_project(project_path)

            self.assertEqual(restored.document, session.document)
            self.assertEqual(
                restored.document.geometry_revisions[0].import_recipe[
                    "dependency_policy"
                ],
                "closed_manifest",
            )
            self.assertEqual(len(restored.source_mesh.source_resources), 2)
            np.testing.assert_allclose(
                restored.source_mesh.vertices,
                expected_vertices,
                rtol=0.0,
                atol=1e-12,
            )

    def test_preflight_counts_real_central_directory_records(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "forged-count.amr"
            with zipfile.ZipFile(path, "w") as zf:
                for index in range(project_file.MAX_ZIP_MEMBERS + 1):
                    zf.writestr(f"entry-{index:03d}", b"")

            forged = bytearray(path.read_bytes())
            eocd = forged.rfind(b"PK\x05\x06")
            self.assertGreaterEqual(eocd, 0)
            struct.pack_into("<H", forged, eocd + 8, 1)
            struct.pack_into("<H", forged, eocd + 10, 1)
            path.write_bytes(forged)

            with mock.patch.object(
                project_file.zipfile,
                "ZipFile",
                side_effect=AssertionError("ZipFile must not be constructed"),
            ):
                with self.assertRaisesRegex(ProjectFormatError, "too many members"):
                    load_project(path)

    def test_source_blob_has_a_separate_large_file_budget(self):
        manifest = zipfile.ZipInfo(MANIFEST_NAME)
        manifest.file_size = 1
        manifest.compress_size = 1
        manifest.compress_type = zipfile.ZIP_STORED
        checksums = zipfile.ZipInfo(CHECKSUMS_NAME)
        checksums.file_size = 1
        checksums.compress_size = 1
        checksums.compress_type = zipfile.ZIP_STORED
        digest = "a" * 64
        source = zipfile.ZipInfo(source_blob_member(digest))
        source.file_size = project_file.MAX_MEMBER_BYTES + 1
        source.compress_size = source.file_size
        source.compress_type = zipfile.ZIP_STORED

        members = project_file._validate_zip_infos([manifest, checksums, source])

        self.assertIn(source.filename, members)
        self.assertLess(source.file_size, project_file.MAX_SOURCE_MEMBER_BYTES)

    def test_non_regular_unix_zip_member_is_rejected(self):
        manifest = zipfile.ZipInfo(MANIFEST_NAME)
        manifest.file_size = 1
        manifest.compress_size = 1
        manifest.compress_type = zipfile.ZIP_STORED
        source = zipfile.ZipInfo(source_blob_member("a" * 64))
        source.file_size = 1
        source.compress_size = 1
        source.compress_type = zipfile.ZIP_STORED
        source.create_system = 3
        source.external_attr = (stat.S_IFLNK | 0o777) << 16

        with self.assertRaisesRegex(ProjectFormatError, "Non-regular Unix"):
            project_file._validate_zip_infos([manifest, source])
