from __future__ import annotations

from dataclasses import replace
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

import src.core.artifact_vector_export as vector_export
from src.core.artifact_document import RecordLifecycleStatus
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_export import (
    ArtifactVectorExportError,
    MAX_VECTOR_EXPORT_SVG_BYTES,
    VECTOR_EXPORT_SIDECAR_NAME,
    VECTOR_EXPORT_SVG_NAME,
    VectorSVGOptions,
    build_vector_export,
    discard_prepared_vector_package,
    discard_staged_vector_package,
    export_vector_package,
    prepare_staged_vector_publication,
    publish_prepared_vector_package,
    publish_staged_vector_package,
    stage_vector_package,
    validate_vector_export_bytes,
    validate_vector_export_package,
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
from src.core.source_identity import SourceFingerprint


STAMP = "2026-07-12T00:00:00Z"
RECIPE = {
    "algorithm": "archmeshrubbing.triangle_plane_intersection",
    "algorithm_version": "1.0.0",
    "kind": "cutline",
    "plane_offset_mm": 0.0,
    "stitch_tolerance_mm": 0.01,
}
SVG_NS = "http://www.w3.org/2000/svg"


def _canonical_json(value: dict[str, object]) -> bytes:
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


def _session() -> ArtifactSession:
    mesh = MeshData(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        unit="cm",
        filepath=Path("/source/유물 & 기록.ply"),
        source_identity=SourceFingerprint(
            sha256="a" * 64,
            size_bytes=321,
            mtime_ns=1,
            original_name="유물 & 기록.ply",
            format="ply",
        ),
        source_format="ply",
    )
    return ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/유물 & 기록.ply",
        unit="cm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="0.4-test",
        operator="고고학자",
        created_at=STAMP,
        document_id="artifact:vector-export",
        metadata_revision_id="metadata:m1",
        align_revision_id="align:a1",
    )


def _payload() -> VectorGeometryPayload:
    # This is an X-normal vertical plane.  Storing planar points with this frame
    # proves the exporter never collapses Front/Right cutlines onto world XY.
    frame = PlanarFrame(
        origin_world_mm=(10.0, 20.0, 30.0),
        u_axis_world=(0.0, 0.0, 1.0),
        v_axis_world=(0.0, 1.0, 0.0),
        normal_world=(-1.0, 0.0, 0.0),
    )
    return VectorGeometryPayload(
        schema_version=VECTOR_PAYLOAD_SCHEMA_VERSION,
        kind=VectorRecordKind.CUTLINE,
        coordinate_space=VECTOR_COORDINATE_SPACE,
        frame=frame,
        paths=(
            VectorPath(
                id="section:주단면&1",
                role="section",
                closed=True,
                points_mm=(
                    (0.0, 0.0),
                    (100.0, 0.0),
                    (100.0, 50.0),
                    (0.0, 50.0),
                ),
            ),
        ),
    )


def _committed_session() -> ArtifactSession:
    session = _session()
    context = session.capture_vector_operation(recipe=RECIPE)
    return session.commit_vector_record(
        context=context,
        payload=_payload(),
        recipe=RECIPE,
        record_id="record:cutline-0",
        created_at=STAMP,
        operator="고고학자",
        qc={"plane_residual_max_mm": 0.0},
    )


class TestArtifactVectorExportScaleAndProvenance(unittest.TestCase):
    def test_exact_1_to_1_svg_tokens_and_provenance(self):
        session = _committed_session()
        bundle = build_vector_export(
            session.document,
            "record:cutline-0",
            options=VectorSVGOptions(
                margin_mm=5.0,
                stroke_width_mm=0.25,
                stroke_color="#123ABC",
                title="유물 & <주단면>",
            ),
        )

        root = ET.fromstring(bundle.svg_bytes)
        self.assertEqual(root.attrib["width"], "110mm")
        self.assertEqual(root.attrib["height"], "60mm")
        self.assertEqual(root.attrib["viewBox"], "0 0 110 60")
        self.assertAlmostEqual(bundle.width_mm / 25.4, 110.0 / 25.4, places=12)
        self.assertAlmostEqual(bundle.height_mm / 25.4, 60.0 / 25.4, places=12)
        self.assertNotIn(b"transform=", bundle.svg_bytes)
        title = root.find(f"{{{SVG_NS}}}title")
        self.assertIsNotNone(title)
        assert title is not None and title.text is not None
        self.assertIn("유물 & <주단면>", title.text)

        path = root.find(f".//{{{SVG_NS}}}path")
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path.attrib["id"], "section:주단면&1")
        self.assertEqual(path.attrib["data-role"], "section")
        self.assertEqual(path.attrib["d"], "M 5 55 L 105 55 L 105 5 L 5 5 Z")

        sidecar = json.loads(bundle.sidecar_bytes)
        self.assertEqual(sidecar["presentation"]["content_bounds_mm"], [0.0, 0.0, 100.0, 50.0])
        self.assertEqual(sidecar["presentation"]["view_box"], ["0", "0", "110", "60"])
        self.assertEqual(sidecar["provenance"]["source_assets"][0]["sha256"], "a" * 64)
        self.assertEqual(sidecar["provenance"]["source_assets"][0]["original_name"], "유물 & 기록.ply")
        self.assertNotIn("asset_ref", sidecar["provenance"]["source_assets"][0])
        metadata = sidecar["provenance"]["source_metadata_revision"]
        self.assertEqual(metadata["unit"], "cm")
        self.assertEqual(metadata["source_to_canonical_mm"][0][0], 10.0)
        self.assertEqual(sidecar["vector_payload"]["frame"]["normal_world"], [-1.0, 0.0, 0.0])
        self.assertEqual(sidecar["qc"]["record"]["plane_residual_max_mm"], 0.0)
        self.assertEqual(sidecar["qc"]["scale"]["physical_scale"], "1:1")
        self.assertEqual(sidecar["recipe"], RECIPE)
        self.assertTrue(bundle.sidecar_bytes.endswith(b"\n"))
        self.assertNotIn(b"\r", bundle.sidecar_bytes)
        self.assertEqual(hashlib.sha256(bundle.svg_bytes).hexdigest(), bundle.svg_sha256)
        self.assertEqual(
            hashlib.sha256(bundle.sidecar_bytes).hexdigest(),
            bundle.sidecar_sha256,
        )

    def test_known_metric_rectangle_converts_to_inches_exactly_once(self):
        session = _session()
        payload = replace(
            _payload(),
            paths=(
                VectorPath(
                    id="inch-reference",
                    role="section",
                    closed=True,
                    points_mm=((0.0, 0.0), (25.4, 0.0), (25.4, 50.8), (0.0, 50.8)),
                ),
            ),
        )
        context = session.capture_vector_operation(recipe=RECIPE)
        document = session.commit_vector_record(
            context=context,
            payload=payload,
            recipe=RECIPE,
            record_id="record:inch-reference",
            created_at=STAMP,
            operator="tester",
        ).document

        bundle = build_vector_export(
            document,
            "record:inch-reference",
            options=VectorSVGOptions(margin_mm=0.1),
        )
        root = ET.fromstring(bundle.svg_bytes)
        self.assertEqual(root.attrib["width"], "25.6mm")
        self.assertEqual(root.attrib["height"], "51mm")
        self.assertEqual(root.attrib["viewBox"], "0 0 25.6 51")
        sidecar = json.loads(bundle.sidecar_bytes)
        bounds = sidecar["presentation"]["content_bounds_mm"]
        self.assertEqual((bounds[2] - bounds[0]) / 25.4, 1.0)
        self.assertEqual((bounds[3] - bounds[1]) / 25.4, 2.0)

    def test_canonical_bytes_are_repeatable(self):
        document = _committed_session().document
        first = build_vector_export(document, "record:cutline-0")
        second = build_vector_export(document, "record:cutline-0")

        self.assertEqual(first.svg_bytes, second.svg_bytes)
        self.assertEqual(first.sidecar_bytes, second.sidecar_bytes)
        self.assertEqual(first.svg_sha256, second.svg_sha256)
        self.assertEqual(first.sidecar_sha256, second.sidecar_sha256)
        self.assertEqual(
            first.svg_sha256,
            "2c5b670d7fdb70f42917b8166cf8c5c63aad0b84792487b714f58c9293162bc9",
        )
        self.assertEqual(
            first.sidecar_sha256,
            "48f48218c06ce1bfe6d2aff644c30874bb00b7aa7f46d53090ed66531b69b02b",
        )

    def test_multiple_cutline_components_survive_without_world_xy_collapse(self):
        session = _session()
        payload = replace(
            _payload(),
            paths=(
                VectorPath(
                    id="closed",
                    role="section",
                    closed=True,
                    points_mm=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
                ),
                VectorPath(
                    id="open",
                    role="section",
                    closed=False,
                    points_mm=((20.0, 2.0), (30.0, 8.0)),
                ),
            ),
        )
        context = session.capture_vector_operation(recipe=RECIPE)
        document = session.commit_vector_record(
            context=context,
            payload=payload,
            recipe=RECIPE,
            record_id="record:multi-cutline",
            created_at=STAMP,
            operator="tester",
        ).document

        bundle = build_vector_export(document, "record:multi-cutline")
        root = ET.fromstring(bundle.svg_bytes)
        paths = root.findall(f".//{{{SVG_NS}}}path")
        self.assertEqual(len(paths), 2)
        self.assertEqual([path.attrib["id"] for path in paths], ["closed", "open"])
        self.assertTrue(paths[0].attrib["d"].endswith(" Z"))
        self.assertFalse(paths[1].attrib["d"].endswith(" Z"))
        sidecar = json.loads(bundle.sidecar_bytes)
        self.assertEqual(sidecar["presentation"]["content_bounds_mm"], [0.0, 0.0, 30.0, 10.0])
        self.assertEqual(sidecar["vector_payload"]["frame"]["normal_world"], [-1.0, 0.0, 0.0])

    def test_outline_islands_and_hole_survive_as_distinct_unfilled_paths(self):
        session = _session()
        recipe = {
            "algorithm": "test.fixed_grid_triangle_union",
            "algorithm_version": "1.0.0",
            "kind": "outline",
            "precision_grid_mm": 0.01,
            "view": "top",
        }
        payload = VectorGeometryPayload(
            schema_version=VECTOR_PAYLOAD_SCHEMA_VERSION,
            kind=VectorRecordKind.OUTLINE,
            coordinate_space=VECTOR_COORDINATE_SPACE,
            frame=PlanarFrame(
                origin_world_mm=(0.0, 0.0, 0.0),
                u_axis_world=(1.0, 0.0, 0.0),
                v_axis_world=(0.0, 1.0, 0.0),
                normal_world=(0.0, 0.0, 1.0),
            ),
            paths=(
                VectorPath(
                    id="outline:component:0000:hole:0000",
                    role="hole",
                    closed=True,
                    points_mm=((2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)),
                ),
                VectorPath(
                    id="outline:component:0001:exterior",
                    role="exterior",
                    closed=True,
                    points_mm=((20.0, 0.0), (25.0, 0.0), (25.0, 5.0), (20.0, 5.0)),
                ),
                VectorPath(
                    id="outline:component:0000:exterior",
                    role="exterior",
                    closed=True,
                    points_mm=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
                ),
            ),
        )
        context = session.capture_vector_operation(recipe=recipe)
        document = session.commit_vector_record(
            context=context,
            payload=payload,
            recipe=recipe,
            record_id="record:outline-top",
            created_at=STAMP,
            operator="tester",
        ).document

        bundle = build_vector_export(document, "record:outline-top")
        root = ET.fromstring(bundle.svg_bytes)
        group = root.find(f"{{{SVG_NS}}}g")
        assert group is not None
        self.assertEqual(group.attrib["fill"], "none")
        self.assertNotIn("transform", group.attrib)
        paths = root.findall(f".//{{{SVG_NS}}}path")
        self.assertEqual(len(paths), 3)
        self.assertEqual(
            [path.attrib["data-role"] for path in paths],
            ["exterior", "exterior", "hole"],
        )
        sidecar = json.loads(bundle.sidecar_bytes)
        self.assertEqual(sidecar["presentation"]["content_bounds_mm"], [0.0, 0.0, 25.0, 10.0])
        self.assertEqual(sidecar["qc"]["payload"]["closed_path_count"], 3)

    def test_dependency_closure_and_export_time_active_context_are_portable(self):
        session = _session()
        first_context = session.capture_vector_operation(recipe=RECIPE)
        session = session.commit_vector_record(
            context=first_context,
            payload=_payload(),
            recipe=RECIPE,
            record_id="record:dependency",
            created_at=STAMP,
            operator="tester",
        )
        root_context = session.capture_vector_operation(recipe=RECIPE)
        document = session.commit_vector_record(
            context=root_context,
            payload=_payload(),
            recipe=RECIPE,
            record_id="record:dependent",
            created_at=STAMP,
            operator="tester",
            depends_on_record_ids=("record:dependency",),
        ).document

        bundle = build_vector_export(document, "record:dependent")
        sidecar = json.loads(bundle.sidecar_bytes)
        provenance = sidecar["provenance"]
        self.assertEqual(
            provenance["document"]["active_align_revision_id"],
            "align:a1",
        )
        self.assertEqual(
            provenance["document"]["active_source_metadata_revision_id"],
            "metadata:m1",
        )
        self.assertEqual(
            provenance["record"]["depends_on_record_ids"],
            ["record:dependency"],
        )
        self.assertEqual(len(provenance["dependency_closure"]), 1)
        receipt = provenance["dependency_closure"][0]
        self.assertEqual(receipt["id"], "record:dependency")
        self.assertEqual(receipt["freshness"], "fresh")
        self.assertEqual(receipt["lifecycle_status"], "ready")
        validate_vector_export_bytes(
            bundle.svg_bytes,
            bundle.sidecar_bytes,
            document=document,
        )

        provenance["dependency_closure"] = []
        with self.assertRaisesRegex(ArtifactVectorExportError, "missing receipt"):
            validate_vector_export_bytes(bundle.svg_bytes, _canonical_json(sidecar))

    def test_private_revision_extensions_and_paths_never_leak(self):
        document = _committed_session().document
        secret = "/Users/alice/private/site-notes.txt"
        geometry = document.geometry_revisions[0]
        metadata = document.source_metadata_revisions[0]
        align = document.align_revisions[0]
        asset = document.source_assets[0]
        private_document = replace(
            document,
            geometry_revisions=(
                replace(
                    geometry,
                    import_recipe={**dict(geometry.import_recipe), "source_path": secret},
                    topology_map_ref=secret,
                    extensions={"org.example:private-path": secret},
                ),
            ),
            source_metadata_revisions=(
                replace(
                    metadata,
                    extensions={"org.example:private-note": secret},
                ),
            ),
            align_revisions=(
                replace(
                    align,
                    recipe={**dict(align.recipe), "source_path": secret},
                    qc={**dict(align.qc), "private_note": secret},
                    extensions={"org.example:private-note": secret},
                ),
            ),
            source_assets=(
                replace(
                    asset,
                    asset_ref=f"external:{secret}",
                    extensions={"org.example:private-path": secret},
                ),
            ),
        )

        bundle = build_vector_export(private_document, "record:cutline-0")
        self.assertNotIn(secret.encode("utf-8"), bundle.sidecar_bytes)
        self.assertNotIn(secret.encode("utf-8"), bundle.svg_bytes)


class TestArtifactVectorExportFailClosed(unittest.TestCase):
    def test_svg_byte_tampering_is_detected(self):
        bundle = build_vector_export(_committed_session().document, "record:cutline-0")
        tampered = bundle.svg_bytes.replace(b"M 5 55", b"M 6 55", 1)

        with self.assertRaisesRegex(ArtifactVectorExportError, "SVG SHA-256"):
            validate_vector_export_bytes(tampered, bundle.sidecar_bytes)

    def test_semantic_payload_tampering_is_detected(self):
        bundle = build_vector_export(_committed_session().document, "record:cutline-0")
        sidecar = json.loads(bundle.sidecar_bytes)
        sidecar["vector_payload"]["paths"][0]["points_mm"][1][0] = 99.0

        with self.assertRaisesRegex(ArtifactVectorExportError, "semantic SHA-256"):
            validate_vector_export_bytes(bundle.svg_bytes, _canonical_json(sidecar))

    def test_active_content_is_rejected_even_if_artifact_hash_is_relabelled(self):
        bundle = build_vector_export(_committed_session().document, "record:cutline-0")
        tampered = bundle.svg_bytes.replace(b"</svg>", b"<script>bad()</script>\n</svg>")
        sidecar = json.loads(bundle.sidecar_bytes)
        sidecar["artifact"]["size_bytes"] = len(tampered)
        sidecar["artifact"]["sha256"] = hashlib.sha256(tampered).hexdigest()

        with self.assertRaisesRegex(ArtifactVectorExportError, "canonical payload derivative"):
            validate_vector_export_bytes(tampered, _canonical_json(sidecar))

    def test_sidecar_provenance_tampering_is_bound_into_svg_metadata(self):
        bundle = build_vector_export(_committed_session().document, "record:cutline-0")
        sidecar = json.loads(bundle.sidecar_bytes)
        sidecar["provenance"]["align_revision"]["matrix4x4"][0][3] = 1000.0

        with self.assertRaisesRegex(ArtifactVectorExportError, "canonical payload derivative"):
            validate_vector_export_bytes(bundle.svg_bytes, _canonical_json(sidecar))

        sidecar = json.loads(bundle.sidecar_bytes)
        sidecar["provenance"]["document"]["active_align_revision_id"] = "align:forged"
        with self.assertRaisesRegex(ArtifactVectorExportError, "active Align"):
            validate_vector_export_bytes(bundle.svg_bytes, _canonical_json(sidecar))

    def test_unconfirmed_source_metadata_cannot_support_a_scale_claim(self):
        bundle = build_vector_export(_committed_session().document, "record:cutline-0")
        sidecar = json.loads(bundle.sidecar_bytes)
        sidecar["provenance"]["source_metadata_revision"][
            "confirmation_status"
        ] = "unconfirmed"

        with self.assertRaisesRegex(ArtifactVectorExportError, "requires confirmed"):
            validate_vector_export_bytes(bundle.svg_bytes, _canonical_json(sidecar))

    def test_raw_numeric_strings_and_boole_are_not_normalized_into_valid_payloads(self):
        bundle = build_vector_export(_committed_session().document, "record:cutline-0")
        for replacement in ("100.0", True):
            with self.subTest(replacement=replacement):
                sidecar = json.loads(bundle.sidecar_bytes)
                sidecar["vector_payload"]["paths"][0]["points_mm"][1][0] = replacement
                with self.assertRaisesRegex(ArtifactVectorExportError, "finite numbers"):
                    validate_vector_export_bytes(bundle.svg_bytes, _canonical_json(sidecar))

    def test_pathological_json_numbers_and_nesting_return_domain_errors(self):
        bundle = build_vector_export(_committed_session().document, "record:cutline-0")
        pathological = (
            b'{"x":' + b"9" * 5000 + b"}\n",
            b'{"x":' + b"[" * 1500 + b"0" + b"]" * 1500 + b"}\n",
        )
        for sidecar in pathological:
            with self.subTest(size=len(sidecar)):
                with self.assertRaises(ArtifactVectorExportError):
                    validate_vector_export_bytes(bundle.svg_bytes, sidecar)

    def test_noncanonical_utf16_svg_is_rejected_before_xml_parsing(self):
        bundle = build_vector_export(_committed_session().document, "record:cutline-0")
        malicious_text = (
            '<?xml version="1.0" encoding="UTF-16"?>'
            '<!DOCTYPE svg [<!ENTITY probe "expanded">]>'
            f'<svg xmlns="{SVG_NS}"><title>&probe;</title></svg>'
        )
        malicious = malicious_text.encode("utf-16")
        sidecar = json.loads(bundle.sidecar_bytes)
        sidecar["artifact"]["size_bytes"] = len(malicious)
        sidecar["artifact"]["sha256"] = hashlib.sha256(malicious).hexdigest()

        with self.assertRaisesRegex(ArtifactVectorExportError, "canonical payload derivative"):
            validate_vector_export_bytes(malicious, _canonical_json(sidecar))

    def test_offline_provenance_rejects_malformed_scalar_types(self):
        bundle = build_vector_export(_committed_session().document, "record:cutline-0")
        mutations = (
            (("provenance", "record", "operator"), {"name": "forged"}, "operator"),
            (("provenance", "record", "selection_hash"), "bad", "selection_hash"),
            (("provenance", "document", "software_version"), True, "software_version"),
            (
                ("provenance", "source_assets", 0, "original_name"),
                ["forged.ply"],
                "original_name",
            ),
        )
        for path, replacement, message in mutations:
            with self.subTest(path=path):
                sidecar = json.loads(bundle.sidecar_bytes)
                target = sidecar
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                with self.assertRaisesRegex(ArtifactVectorExportError, message):
                    validate_vector_export_bytes(bundle.svg_bytes, _canonical_json(sidecar))

    def test_stale_draft_and_qc_relabelled_records_cannot_export(self):
        session = _committed_session()
        stale = session.commit_preview(
            translation_mm=(1.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            operator="tester",
            created_at=STAMP,
            revision_id="align:a2",
        )
        with self.assertRaisesRegex(ArtifactVectorExportError, "FRESH"):
            build_vector_export(stale.document, "record:cutline-0")

        record = session.document.record_index["record:cutline-0"]
        draft = replace(record, lifecycle_status=RecordLifecycleStatus.DRAFT)
        draft_document = replace(session.document, records=(draft,))
        with self.assertRaisesRegex(ArtifactVectorExportError, "READY"):
            build_vector_export(draft_document, record.id)

        relabelled_qc = dict(record.qc)
        relabelled_qc["bounds_mm"] = [0.0, 0.0, 999.0, 50.0]
        bad_record = replace(record, qc=relabelled_qc)
        bad_document = replace(session.document, records=(bad_record,))
        with self.assertRaisesRegex(ArtifactVectorExportError, "record QC"):
            build_vector_export(bad_document, record.id)

    def test_invalid_options_fail_closed(self):
        with self.assertRaisesRegex(ArtifactVectorExportError, "stroke_color"):
            VectorSVGOptions(stroke_color="red")
        with self.assertRaisesRegex(ArtifactVectorExportError, "greater than zero"):
            VectorSVGOptions(stroke_width_mm=0.0)
        with self.assertRaisesRegex(ArtifactVectorExportError, "at least"):
            VectorSVGOptions(margin_mm=-1.0)
        with self.assertRaisesRegex(ArtifactVectorExportError, "half"):
            VectorSVGOptions(margin_mm=0.0)
        with self.assertRaisesRegex(ArtifactVectorExportError, "VectorSVGOptions"):
            build_vector_export(
                _committed_session().document,
                "record:cutline-0",
                options=False,  # type: ignore[arg-type]
            )

    def test_zero_extent_open_cutline_uses_half_stroke_safe_artboard(self):
        session = _session()
        payload = replace(
            _payload(),
            paths=(
                VectorPath(
                    id="vertical-section",
                    role="section",
                    closed=False,
                    points_mm=((0.0, 0.0), (0.0, 10.0)),
                ),
            ),
        )
        context = session.capture_vector_operation(recipe=RECIPE)
        document = session.commit_vector_record(
            context=context,
            payload=payload,
            recipe=RECIPE,
            record_id="record:vertical",
            created_at=STAMP,
            operator="tester",
        ).document

        bundle = build_vector_export(
            document,
            "record:vertical",
            options=VectorSVGOptions(margin_mm=1.0, stroke_width_mm=2.0),
        )
        root = ET.fromstring(bundle.svg_bytes)
        self.assertEqual(root.attrib["width"], "2mm")
        self.assertEqual(root.attrib["height"], "12mm")
        path = root.find(f".//{{{SVG_NS}}}path")
        assert path is not None
        self.assertEqual(path.attrib["d"], "M 1 11 L 1 1")


class TestArtifactVectorExportPackage(unittest.TestCase):
    def setUp(self) -> None:
        self._confirmed_directory_fsync = patch.object(
            vector_export,
            "_fsync_parent",
            return_value=True,
        )
        self._confirmed_directory_fsync.start()
        self.addCleanup(self._confirmed_directory_fsync.stop)

    def test_prepared_capability_is_exact_destination_bound_and_single_use(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "exact.amr-vector"
            staging = stage_vector_package(
                destination,
                document,
                "record:cutline-0",
            )
            prepared = prepare_staged_vector_publication(
                staging,
                destination,
                document=document,
            )

            with self.assertRaisesRegex(
                ArtifactVectorExportError,
                "invalid or consumed",
            ):
                publish_prepared_vector_package(replace(prepared))
            with self.assertRaisesRegex(
                ArtifactVectorExportError,
                "different destination",
            ):
                prepare_staged_vector_publication(
                    staging,
                    root / "other.amr-vector",
                    document=document,
                )

            self.assertEqual(
                publish_prepared_vector_package(prepared),
                destination,
            )
            with self.assertRaises(ArtifactVectorExportError) as raised:
                publish_prepared_vector_package(prepared)
            self.assertTrue(raised.exception.committed)

    def test_public_publish_rejects_never_owned_and_replaced_staging(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "owned.amr-vector"
            foreign = root / f".amrv-stage-{'f' * 32}"
            foreign.mkdir()
            with self.assertRaisesRegex(
                ArtifactVectorExportError,
                "not created by this process",
            ):
                publish_staged_vector_package(foreign, destination)

            staging = stage_vector_package(
                destination,
                document,
                "record:cutline-0",
            )
            prepared = prepare_staged_vector_publication(
                staging,
                destination,
                document=document,
            )
            moved_owned = root / "moved-owned"
            staging.rename(moved_owned)
            staging.mkdir()
            sentinel = staging / "foreign.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(
                ArtifactVectorExportError,
                "replaced",
            ):
                publish_prepared_vector_package(prepared)
            self.assertFalse(discard_prepared_vector_package(prepared))
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            self.assertTrue(moved_owned.is_dir())

    def test_fixed_length_stage_supports_long_destination_name(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / ("유" * 70 + ".amr-vector")
            staging = stage_vector_package(
                destination,
                document,
                "record:cutline-0",
            )
            self.assertEqual(len(staging.name), len(".amrv-stage-") + 32)
            self.assertNotIn(destination.name, staging.name)
            prepared = prepare_staged_vector_publication(
                staging,
                destination,
                document=document,
            )
            self.assertEqual(
                publish_prepared_vector_package(prepared),
                destination,
            )

    def test_pre_moved_stage_is_reported_as_committed_visible_effect(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "pre-moved.amr-vector"
            staging = stage_vector_package(
                destination,
                document,
                "record:cutline-0",
            )
            prepared = prepare_staged_vector_publication(
                staging,
                destination,
                document=document,
            )
            staging.rename(destination)

            with self.assertRaises(ArtifactVectorExportError) as raised:
                discard_prepared_vector_package(prepared)
            self.assertTrue(raised.exception.committed)
            self.assertTrue(destination.is_dir())

    def test_missing_owned_stage_is_not_successfully_discarded(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "missing.amr-vector"
            staging = stage_vector_package(
                destination,
                document,
                "record:cutline-0",
            )
            staging.rename(root / "moved-somewhere-else")
            self.assertFalse(discard_staged_vector_package(staging, destination))

    def test_windows_missing_rename_is_a_typed_export_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                vector_export.sys,
                "platform",
                "win32",
            ), patch.object(vector_export.os, "name", "nt"):
                with self.assertRaisesRegex(
                    ArtifactVectorExportError,
                    "cannot atomically publish",
                ):
                    vector_export._rename_directory_noreplace(
                        root / "missing-stage",
                        root / "destination",
                    )

    @unittest.skipIf(
        os.name == "nt",
        "requires descriptor-relative POSIX cleanup",
    )
    def test_discard_detects_top_directory_swap_and_preserves_foreign(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "swap.amr-vector"
            staging = stage_vector_package(
                destination,
                document,
                "record:cutline-0",
            )
            moved_owned = root / "moved-after-open"
            original_empty = vector_export._empty_vector_directory_fd

            def swap_then_empty(directory_fd: int) -> None:
                quarantine = next(root.glob(".amrv-discard-*"))
                quarantine.rename(moved_owned)
                quarantine.mkdir()
                (quarantine / "foreign.txt").write_text(
                    "preserve",
                    encoding="utf-8",
                )
                original_empty(directory_fd)

            with patch.object(
                vector_export,
                "_empty_vector_directory_fd",
                side_effect=swap_then_empty,
            ), patch.object(
                vector_export.shutil,
                "rmtree",
                side_effect=AssertionError("POSIX cleanup must not use rmtree"),
            ):
                self.assertFalse(
                    discard_staged_vector_package(staging, destination)
                )

            self.assertEqual(
                (staging / "foreign.txt").read_text(encoding="utf-8"),
                "preserve",
            )
            self.assertTrue(moved_owned.is_dir())

    def test_windows_fallback_quarantines_and_cleans_owned_inode(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "windows.amr-vector"
            staging = stage_vector_package(
                destination,
                document,
                "record:cutline-0",
            )
            real_rmtree = vector_export.shutil.rmtree
            with patch.object(
                vector_export,
                "_descriptor_cleanup_available",
                return_value=False,
            ), patch.object(
                vector_export,
                "_windows_cleanup_fallback_required",
                return_value=True,
            ), patch.object(
                vector_export.shutil,
                "rmtree",
                wraps=real_rmtree,
            ) as rmtree:
                self.assertTrue(
                    discard_staged_vector_package(staging, destination)
                )
            rmtree.assert_called_once()
            self.assertFalse(staging.exists())

    def test_unsupported_directory_fsync_is_committed_uncertainty(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            vector_export,
            "_fsync_parent",
            return_value=False,
        ):
            destination = Path(temporary) / "unsupported-fsync.amr-vector"
            staging = stage_vector_package(
                destination,
                document,
                "record:cutline-0",
            )
            prepared = prepare_staged_vector_publication(
                staging,
                destination,
                document=document,
            )
            with self.assertRaises(ArtifactVectorExportError) as raised:
                publish_prepared_vector_package(prepared)
            self.assertTrue(raised.exception.committed)
            self.assertIn("unsupported", str(raised.exception))
            self.assertTrue(destination.is_dir())

    def test_einval_directory_fsync_is_explicitly_unconfirmed(self):
        self._confirmed_directory_fsync.stop()
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            vector_export.os,
            "fsync",
            side_effect=OSError(errno.EINVAL, "unsupported"),
        ):
            self.assertFalse(vector_export._fsync_parent(Path(temporary)))

    def test_post_rename_destination_inode_is_verified(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "verify-inode.amr-vector"
            staging = stage_vector_package(
                destination,
                document,
                "record:cutline-0",
            )
            prepared = prepare_staged_vector_publication(
                staging,
                destination,
                document=document,
            )
            moved_owned = root / "published-owned-moved"
            real_rename = vector_export._rename_directory_noreplace

            def replace_after_rename(source: Path, target: Path) -> None:
                real_rename(source, target)
                target.rename(moved_owned)
                target.mkdir()
                (target / "foreign.txt").write_text("preserve", encoding="utf-8")

            with patch.object(
                vector_export,
                "_rename_directory_noreplace",
                side_effect=replace_after_rename,
            ):
                with self.assertRaises(ArtifactVectorExportError) as raised:
                    publish_prepared_vector_package(prepared)
            self.assertTrue(raised.exception.committed)
            self.assertEqual(
                (destination / "foreign.txt").read_text(encoding="utf-8"),
                "preserve",
            )
            self.assertTrue(moved_owned.is_dir())

    def test_stage_is_same_parent_verified_and_does_not_publish_destination(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "staged.amr-vector"
            collision = "c" * 32
            owned = "a" * 32
            foreign = root / f".amrv-stage-{collision}"
            foreign.mkdir()
            sentinel = foreign / "foreign-sentinel.txt"
            sentinel.write_text("do not remove", encoding="utf-8")

            with patch.object(
                vector_export.uuid,
                "uuid4",
                side_effect=[
                    SimpleNamespace(hex=collision),
                    SimpleNamespace(hex=owned),
                ],
            ):
                staging = stage_vector_package(
                    destination,
                    document,
                    "record:cutline-0",
                )

            self.assertEqual(staging.parent, destination.parent)
            self.assertEqual(staging.name, f".amrv-stage-{owned}")
            self.assertFalse(destination.exists())
            validate_vector_export_package(staging, document=document)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not remove")

            with patch.object(vector_export, "_fsync_parent") as fsync_parent:
                published = publish_staged_vector_package(
                    staging,
                    destination,
                    document=document,
                )
            self.assertEqual(published, destination)
            fsync_parent.assert_called_once_with(destination.parent)
            self.assertFalse(staging.exists())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not remove")

    def test_stage_collision_budget_preserves_foreign_directory(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "busy.amr-vector"
            collision = "c" * 32
            foreign = root / f".amrv-stage-{collision}"
            foreign.mkdir()
            sentinel = foreign / "sentinel.txt"
            sentinel.write_text("foreign", encoding="utf-8")

            with patch.object(
                vector_export.uuid,
                "uuid4",
                return_value=SimpleNamespace(hex=collision),
            ) as uuid4:
                with self.assertRaisesRegex(
                    ArtifactVectorExportError,
                    "after 16 attempts",
                ):
                    stage_vector_package(
                        destination,
                        document,
                        "record:cutline-0",
                    )

            self.assertEqual(uuid4.call_count, 16)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "foreign")
            self.assertFalse(destination.exists())

    def test_discard_removes_only_the_registered_staging_inode(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "discard.amr-vector"
            staging = stage_vector_package(
                destination,
                document,
                "record:cutline-0",
            )
            self.assertTrue(
                discard_staged_vector_package(staging, destination)
            )
            self.assertFalse(staging.exists())

            replaced = stage_vector_package(
                destination,
                document,
                "record:cutline-0",
            )
            original = root / "moved-owned-staging"
            replaced.rename(original)
            replaced.mkdir()
            sentinel = replaced / "foreign.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            self.assertFalse(
                discard_staged_vector_package(replaced, destination)
            )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            self.assertTrue(original.is_dir())

    def test_publish_reports_committed_directory_fsync_uncertainty(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "uncertain.amr-vector"
            staging = stage_vector_package(
                destination,
                document,
                "record:cutline-0",
            )
            with patch.object(
                vector_export,
                "_fsync_parent",
                side_effect=OSError(errno.EIO, "injected fsync failure"),
            ):
                with self.assertRaises(ArtifactVectorExportError) as raised:
                    publish_staged_vector_package(
                        staging,
                        destination,
                        document=document,
                    )
            self.assertTrue(raised.exception.committed)
            self.assertTrue(destination.is_dir())
            self.assertFalse(staging.exists())

    def test_invalid_record_does_not_create_destination_parent(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "new-parent" / "bad.amr-vector"
            with self.assertRaisesRegex(ArtifactVectorExportError, "does not exist"):
                stage_vector_package(destination, document, "record:missing")
            self.assertFalse(destination.parent.exists())

    def test_atomic_two_file_package_is_relocatable_and_non_overwriting(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "주단면.amr-vector"
            exported = export_vector_package(package, document, "record:cutline-0")

            self.assertEqual(exported, package)
            self.assertEqual(
                sorted(item.name for item in package.iterdir()),
                sorted([VECTOR_EXPORT_SVG_NAME, VECTOR_EXPORT_SIDECAR_NAME]),
            )
            original = validate_vector_export_package(package, document=document)

            relocated = root / "relocated-anywhere.amr-vector"
            package.rename(relocated)
            offline = validate_vector_export_package(relocated)
            self.assertEqual(offline.svg_sha256, original.svg_sha256)
            self.assertEqual(offline.vector_payload_sha256, original.vector_payload_sha256)

            with self.assertRaisesRegex(ArtifactVectorExportError, "already exists"):
                export_vector_package(relocated, document, "record:cutline-0")

    def test_package_rejects_extra_members_and_wrong_suffix(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ArtifactVectorExportError, "must end"):
                export_vector_package(root / "not-a-package", document, "record:cutline-0")

            package = export_vector_package(
                root / "measured.amr-vector",
                document,
                "record:cutline-0",
            )
            (package / ".DS_Store").write_bytes(b"Finder metadata")
            validate_vector_export_package(package)
            (package / "unexpected.txt").write_text("not allowed", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactVectorExportError, "exactly two"):
                validate_vector_export_package(package)

    def test_concurrent_destination_creation_is_never_overwritten(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "raced.amr-vector"
            real_publish = vector_export._rename_directory_noreplace

            def create_competing_destination(source: Path, target: Path) -> None:
                target.mkdir()
                real_publish(source, target)

            with patch.object(
                vector_export,
                "_rename_directory_noreplace",
                side_effect=create_competing_destination,
            ):
                with self.assertRaisesRegex(ArtifactVectorExportError, "already exists"):
                    export_vector_package(destination, document, "record:cutline-0")

            self.assertTrue(destination.is_dir())
            self.assertEqual(list(destination.iterdir()), [])
            self.assertEqual(
                list(destination.parent.glob(".amrv-stage-*")),
                [],
            )

    def test_publish_race_preserves_destination_and_returned_staging(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "raced-stage.amr-vector"
            staging = stage_vector_package(
                destination,
                document,
                "record:cutline-0",
            )
            destination.mkdir()
            sentinel = destination / "winner.txt"
            sentinel.write_text("other process", encoding="utf-8")

            with self.assertRaisesRegex(ArtifactVectorExportError, "already exists"):
                publish_staged_vector_package(
                    staging,
                    destination,
                    document=document,
                )

            self.assertTrue(staging.is_dir())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "other process")

    def test_package_size_cap_is_checked_before_unbounded_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "oversized.amr-vector"
            package.mkdir()
            with (package / VECTOR_EXPORT_SVG_NAME).open("wb") as stream:
                stream.truncate(MAX_VECTOR_EXPORT_SVG_BYTES + 1)
            (package / VECTOR_EXPORT_SIDECAR_NAME).write_bytes(b"{}\n")

            with self.assertRaisesRegex(ArtifactVectorExportError, "safety limit"):
                validate_vector_export_package(package)

    def test_relocated_package_validates_in_an_independent_offline_process(self):
        document = _committed_session().document
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = export_vector_package(
                root / "first-location.amr-vector",
                document,
                "record:cutline-0",
            )
            relocated = root / "offline-relocated.amr-vector"
            package.rename(relocated)
            script = (
                "import json,sys; sys.path.insert(0,sys.argv[2]); "
                "from src.core.artifact_vector_export import "
                "validate_vector_export_package; "
                "b=validate_vector_export_package(sys.argv[1]); "
                "print(json.dumps({'svg':b.svg_sha256,'payload':b.vector_payload_sha256},sort_keys=True))"
            )
            environment = dict(os.environ)
            project_root = str(Path(__file__).resolve().parents[1])
            environment["PYTHONPATH"] = project_root
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    script,
                    str(relocated),
                    project_root,
                ],
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            verified = json.loads(completed.stdout)
            expected = validate_vector_export_package(relocated)
            self.assertEqual(verified["svg"], expected.svg_sha256)
            self.assertEqual(verified["payload"], expected.vector_payload_sha256)


if __name__ == "__main__":
    unittest.main()
