from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.core.artifact_document import (
    ARTIFACT_DOCUMENT_SCHEMA_VERSION,
    GEOMETRY_HASH_SCOPE_V1,
    PRIMARY_SOURCE_ASSET_ROLE,
    AlignRevision,
    ArtifactDocument,
    ArtifactDocumentError,
    DerivedRecord,
    GeometryRevision,
    Handedness,
    MetadataConfirmationStatus,
    RecordFreshness,
    RecordLifecycleStatus,
    SourceAsset,
    SourceMetadataRevision,
    UnconfirmedMetadataError,
)
from src.core.alignment_utils import transform_points


STAMP = "2026-07-11T00:00:00Z"
SOURCE_SHA = "a" * 64
GEOMETRY_SHA = "b" * 64
RECIPE = {"kind": "cutline", "precision_mm": 0.1}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "artifact_document-1.0.0.schema.json"
GOLDEN_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "projects"
    / "artifact_document_1_0_0.json"
)


def _source() -> SourceAsset:
    return SourceAsset(
        id=f"sha256:{SOURCE_SHA}",
        sha256=SOURCE_SHA,
        size_bytes=123,
        media_type="model/ply",
        original_name="artifact.ply",
        asset_ref="external:artifact.ply",
        role=PRIMARY_SOURCE_ASSET_ROLE,
    )


def _geometry() -> GeometryRevision:
    return GeometryRevision(
        id="geometry:g1",
        source_asset_ids=(f"sha256:{SOURCE_SHA}",),
        geometry_sha256=GEOMETRY_SHA,
        geometry_hash_scope=GEOMETRY_HASH_SCOPE_V1,
        import_recipe={"format": "ply", "process": False},
        topology_map_ref="payload:topology-g1",
        qc={"finite_vertices": True},
        created_at=STAMP,
        operator="tester",
    )


def _metadata(
    metadata_id: str = "metadata:m1",
    *,
    parent_id: str | None = None,
    confirmed: bool = True,
) -> SourceMetadataRevision:
    return SourceMetadataRevision(
        id=metadata_id,
        parent_id=parent_id,
        geometry_revision_id="geometry:g1",
        unit="cm" if confirmed else "unknown",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness=Handedness.RIGHT if confirmed else Handedness.UNKNOWN,
        confirmation_status=(
            MetadataConfirmationStatus.CONFIRMED
            if confirmed
            else MetadataConfirmationStatus.UNCONFIRMED
        ),
        source_to_canonical_mm=(
            (10.0, 0.0, 0.0, 0.0),
            (0.0, 10.0, 0.0, 0.0),
            (0.0, 0.0, 10.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        if confirmed
        else (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        created_at=STAMP,
        operator="tester",
    )


def _align(
    align_id: str = "align:a1",
    *,
    metadata_id: str = "metadata:m1",
    parent_id: str | None = None,
    translation: tuple[float, float, float] = (5.0, -2.0, 3.0),
) -> AlignRevision:
    tx, ty, tz = translation
    return AlignRevision(
        id=align_id,
        parent_id=parent_id,
        source_metadata_revision_id=metadata_id,
        matrix4x4=(
            (1.0, 0.0, 0.0, tx),
            (0.0, 1.0, 0.0, ty),
            (0.0, 0.0, 1.0, tz),
            (0.0, 0.0, 0.0, 1.0),
        ),
        recipe={"kind": "manual_matrix"},
        qc={"rigid": True},
        created_at=STAMP,
        operator="tester",
    )


def _document(*, with_record: bool = False) -> ArtifactDocument:
    document = ArtifactDocument(
        schema_version=ARTIFACT_DOCUMENT_SCHEMA_VERSION,
        document_id="artifact:test-1",
        software_version="0.1.0",
        source_assets=(_source(),),
        geometry_revisions=(_geometry(),),
        source_metadata_revisions=(_metadata(),),
        align_revisions=(_align(),),
        active_source_metadata_revision_id="metadata:m1",
        active_align_revision_id="align:a1",
        records=(),
        extensions={"org.archmeshrubbing:test": {"korean": "유물"}},
    )
    if not with_record:
        return document
    context = document.capture_operation_context(
        recipe=RECIPE,
        selection_hash="c" * 64,
    )
    return document.append_record_from_context(
        context=context,
        id="record:r1",
        type="cutline",
        geometry_ref="payload:cutline-r1.svg",
        recipe=RECIPE,
        qc={"components": 1},
        lifecycle_status=RecordLifecycleStatus.READY,
        created_at=STAMP,
        operator="tester",
    )


def _record(
    record_id: str,
    *,
    depends_on: tuple[str, ...] = (),
    lifecycle: RecordLifecycleStatus = RecordLifecycleStatus.READY,
    align_id: str = "align:a1",
) -> DerivedRecord:
    return DerivedRecord(
        id=record_id,
        type="derived.test",
        geometry_revision_id="geometry:g1",
        align_revision_id=align_id,
        depends_on_record_ids=depends_on,
        geometry_ref=f"payload:{record_id}",
        recipe={"kind": "test"},
        recipe_hash=hashlib.sha256(b'{"kind":"test"}').hexdigest(),
        selection_hash=None,
        qc={},
        lifecycle_status=lifecycle,
        created_at=STAMP,
        operator="tester",
    )


class TestArtifactDocument(unittest.TestCase):
    def test_source_role_and_geometry_hash_scope_are_mandatory_and_controlled(self):
        source_payload = _source().to_dict()
        self.assertEqual(source_payload["role"], PRIMARY_SOURCE_ASSET_ROLE)
        source_payload.pop("role")
        with self.assertRaisesRegex(ArtifactDocumentError, "missing fields: role"):
            SourceAsset.from_dict(source_payload)
        with self.assertRaisesRegex(ArtifactDocumentError, "source asset role"):
            replace(_source(), role="texture")

        geometry_payload = _geometry().to_dict()
        self.assertEqual(
            geometry_payload["geometry_hash_scope"],
            GEOMETRY_HASH_SCOPE_V1,
        )
        geometry_payload.pop("geometry_hash_scope")
        with self.assertRaisesRegex(
            ArtifactDocumentError,
            "missing fields: geometry_hash_scope",
        ):
            GeometryRevision.from_dict(geometry_payload)
        with self.assertRaisesRegex(ArtifactDocumentError, "geometry hash scope"):
            replace(_geometry(), geometry_hash_scope="unspecified")

    def test_minimum_document_materializes_source_cm_to_canonical_mm_then_align(self):
        document = _document()
        point_source = np.array([[10.0, 0.0, 0.0]], dtype=np.float64)

        point_world_mm = transform_points(point_source, document.active_canonical_matrix())

        np.testing.assert_allclose(
            point_world_mm,
            np.array([[105.0, -2.0, 3.0]], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertNotEqual(document.source_assets[0].sha256, document.geometry_revisions[0].geometry_sha256)

    def test_append_activate_align_is_immutable_and_old_record_becomes_fresh_again(self):
        original = _document(with_record=True)
        original_bytes = original.canonical_json_bytes()
        old_record = original.records[0]
        a2 = _align(
            "align:a2",
            parent_id="align:a1",
            translation=(7.0, -2.0, 3.0),
        )

        changed = original.append_align_revision(a2)

        self.assertEqual(original.active_align_revision_id, "align:a1")
        self.assertEqual(original.canonical_json_bytes(), original_bytes)
        self.assertIs(changed.records[0], old_record)
        self.assertEqual(changed.record_freshness("record:r1"), RecordFreshness.STALE_ALIGNMENT)

        restored = changed.activate_align_revision("align:a1")
        self.assertEqual(restored.active_source_metadata_revision_id, "metadata:m1")
        self.assertEqual(restored.record_freshness("record:r1"), RecordFreshness.FRESH)

    def test_metadata_activation_clears_align_and_align_activation_restores_atomic_context(self):
        original = _document(with_record=True)
        m2 = _metadata("metadata:m2", parent_id="metadata:m1")

        metadata_changed = original.append_source_metadata_revision(m2)

        self.assertEqual(metadata_changed.active_source_metadata_revision_id, "metadata:m2")
        self.assertIsNone(metadata_changed.active_align_revision_id)
        self.assertEqual(
            metadata_changed.record_freshness("record:r1"),
            RecordFreshness.STALE_METADATA,
        )

        restored = metadata_changed.activate_align_revision("align:a1")
        self.assertEqual(restored.active_source_metadata_revision_id, "metadata:m1")
        self.assertEqual(restored.active_align_revision_id, "align:a1")
        self.assertEqual(restored.record_freshness("record:r1"), RecordFreshness.FRESH)

    def test_unconfirmed_metadata_blocks_materialization_and_operation_capture(self):
        document = ArtifactDocument(
            schema_version=ARTIFACT_DOCUMENT_SCHEMA_VERSION,
            document_id="artifact:unconfirmed",
            software_version="0.1.0",
            source_assets=(_source(),),
            geometry_revisions=(_geometry(),),
            source_metadata_revisions=(_metadata(confirmed=False),),
            align_revisions=(_align(),),
            active_source_metadata_revision_id="metadata:m1",
            active_align_revision_id="align:a1",
            records=(),
        )

        with self.assertRaises(UnconfirmedMetadataError):
            document.active_canonical_matrix()
        with self.assertRaises(UnconfirmedMetadataError):
            document.capture_operation_context(recipe=RECIPE)

    def test_async_completion_stays_bound_to_captured_align(self):
        original = _document()
        context = original.capture_operation_context(recipe=RECIPE, selection_hash="d" * 64)
        switched = original.append_align_revision(
            _align("align:a2", parent_id="align:a1", translation=(9.0, 0.0, 0.0))
        )

        completed = switched.append_record_from_context(
            context=context,
            id="record:late",
            type="cutline",
            geometry_ref="payload:late.svg",
            recipe=RECIPE,
            qc={},
            lifecycle_status=RecordLifecycleStatus.READY,
            created_at=STAMP,
            operator="worker",
        )

        self.assertEqual(completed.record_index["record:late"].align_revision_id, "align:a1")
        self.assertEqual(
            completed.record_freshness("record:late"),
            RecordFreshness.STALE_ALIGNMENT,
        )
        with self.assertRaisesRegex(ArtifactDocumentError, "missing revision"):
            switched.append_record_from_context(
                context=replace(context, align_revision_id="align:missing"),
                id="record:bad",
                type="cutline",
                geometry_ref="payload:bad.svg",
                recipe=RECIPE,
                qc={},
                lifecycle_status=RecordLifecycleStatus.READY,
                created_at=STAMP,
                operator="worker",
            )

    def test_missing_and_non_ready_dependencies_have_explicit_deterministic_freshness(self):
        failed = _record("record:failed", lifecycle=RecordLifecycleStatus.FAILED)
        child = _record("record:child", depends_on=("record:failed",))
        draft = _record("record:draft", lifecycle=RecordLifecycleStatus.DRAFT)
        draft_child = _record("record:draft-child", depends_on=("record:draft",))
        missing = _record("record:missing", depends_on=("record:not-present",))
        document = replace(
            _document(),
            records=(missing, child, failed, draft_child, draft),
        )

        statuses = document.record_freshnesses()

        self.assertEqual(statuses["record:failed"], RecordFreshness.FRESH)
        self.assertEqual(statuses["record:child"], RecordFreshness.BLOCKED_DEPENDENCY)
        self.assertEqual(statuses["record:draft"], RecordFreshness.FRESH)
        self.assertEqual(
            statuses["record:draft-child"],
            RecordFreshness.BLOCKED_DEPENDENCY,
        )
        self.assertEqual(statuses["record:missing"], RecordFreshness.MISSING_DEPENDENCY)

        metadata_changed = document.append_source_metadata_revision(
            _metadata("metadata:m2", parent_id="metadata:m1")
        )
        self.assertEqual(
            metadata_changed.record_freshness("record:missing"),
            RecordFreshness.MISSING_DEPENDENCY,
        )

    def test_dependency_cycle_and_implicit_cross_align_dependency_are_rejected(self):
        r1 = _record("record:r1", depends_on=("record:r2",))
        r2 = _record("record:r2", depends_on=("record:r1",))
        with self.assertRaisesRegex(ArtifactDocumentError, "dependency cycle"):
            replace(_document(), records=(r1, r2))

        with_a2 = _document().append_align_revision(
            _align("align:a2", parent_id="align:a1")
        )
        cross_align = _record(
            "record:cross",
            depends_on=("record:base",),
            align_id="align:a2",
        )
        with self.assertRaisesRegex(ArtifactDocumentError, "cross-align"):
            replace(
                with_a2,
                records=(_record("record:base", align_id="align:a1"), cross_align),
            )

    def test_long_dependency_chain_is_iterative(self):
        chain: list[DerivedRecord] = []
        previous: str | None = None
        for index in range(1200):
            record_id = f"record:{index:04d}"
            chain.append(
                _record(record_id, depends_on=((previous,) if previous is not None else ()))
            )
            previous = record_id
        document = replace(_document(), records=tuple(reversed(chain)))

        statuses = document.record_freshnesses()

        self.assertEqual(len(statuses), 1200)
        self.assertTrue(all(status is RecordFreshness.FRESH for status in statuses.values()))

    def test_invalid_references_ids_and_revision_graphs_are_rejected(self):
        with self.assertRaisesRegex(ArtifactDocumentError, "missing source asset"):
            replace(
                _document(),
                geometry_revisions=(
                    replace(_geometry(), source_asset_ids=(f"sha256:{'f' * 64}",)),
                ),
            )
        with self.assertRaisesRegex(ArtifactDocumentError, "duplicate align_revision"):
            replace(_document(), align_revisions=(_align(), _align()))
        with self.assertRaisesRegex(ArtifactDocumentError, "active align revision"):
            replace(_document(), active_align_revision_id="align:missing")

        m1 = _metadata("metadata:m1", parent_id="metadata:m2")
        m2 = _metadata("metadata:m2", parent_id="metadata:m1")
        with self.assertRaisesRegex(ArtifactDocumentError, "parent cycle"):
            ArtifactDocument(
                schema_version=ARTIFACT_DOCUMENT_SCHEMA_VERSION,
                document_id="artifact:cycle",
                software_version="0.1.0",
                source_assets=(_source(),),
                geometry_revisions=(_geometry(),),
                source_metadata_revisions=(m1, m2),
                align_revisions=(),
                active_source_metadata_revision_id="metadata:m1",
                active_align_revision_id=None,
                records=(),
            )

    def test_align_rejects_scale_shear_reflection_and_non_finite_values(self):
        invalid_linear_parts = (
            np.diag([2.0, 2.0, 2.0]),
            np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.diag([-1.0, 1.0, 1.0]),
            np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )
        for linear in invalid_linear_parts:
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = linear
            with self.subTest(linear=linear), self.assertRaises(ArtifactDocumentError):
                replace(_align(), matrix4x4=matrix)

    def test_confirmed_metadata_matrix_must_match_unit_axes_and_handedness(self):
        wrong_scale = np.eye(4, dtype=np.float64)
        with self.assertRaisesRegex(ArtifactDocumentError, "exactly encode unit and axes"):
            replace(_metadata(), source_to_canonical_mm=wrong_scale)
        with self.assertRaisesRegex(ArtifactDocumentError, "handedness"):
            replace(_metadata(), handedness=Handedness.LEFT)

    def test_canonical_serialization_is_order_independent_idempotent_and_strict(self):
        document = _document(with_record=True)
        with_second_record = document.append_record(_record("record:a0"))
        reordered = replace(
            with_second_record,
            source_assets=tuple(reversed(with_second_record.source_assets)),
            records=tuple(reversed(with_second_record.records)),
            extensions={"org.archmeshrubbing:test": {"z": -0.0, "a": 1}},
        )
        normalized = replace(
            with_second_record,
            extensions={"org.archmeshrubbing:test": {"a": 1, "z": 0.0}},
        )

        self.assertEqual(reordered.canonical_json_bytes(), normalized.canonical_json_bytes())
        parsed = ArtifactDocument.from_json_bytes(reordered.canonical_json_bytes())
        self.assertEqual(parsed, reordered)
        self.assertEqual(parsed.canonical_json_bytes(), reordered.canonical_json_bytes())
        self.assertEqual(parsed.canonical_sha256, reordered.canonical_sha256)

        raw = reordered.to_dict()
        raw["unknown_core_field"] = True
        with self.assertRaisesRegex(ArtifactDocumentError, "unknown fields"):
            ArtifactDocument.from_dict(raw)
        with self.assertRaisesRegex(ArtifactDocumentError, "namespaced"):
            replace(reordered, extensions={"not_namespaced": True})
        with self.assertRaisesRegex(ArtifactDocumentError, "key.*must be a string"):
            replace(
                reordered,
                extensions={"org.archmeshrubbing:test": {1: "invalid", "ok": True}},
            )
        with self.assertRaisesRegex(ArtifactDocumentError, "recipe_hash must match"):
            replace(_record("record:bad-hash"), recipe_hash="d" * 64)
        with self.assertRaisesRegex(ArtifactDocumentError, "invalid JSON constant"):
            ArtifactDocument.from_json_bytes(
                reordered.canonical_json_bytes().replace(b'"extensions":{', b'"extensions":{"org.example:nan":NaN,', 1)
            )
        with self.assertRaisesRegex(ArtifactDocumentError, "duplicate JSON object key"):
            ArtifactDocument.from_json_bytes(b'{"schema_version":"1.0.0","schema_version":"1.0.0"}')

    def test_canonical_bytes_match_versioned_golden_fixture(self):
        expected = GOLDEN_PATH.read_bytes()
        document = _document(with_record=True)

        self.assertEqual(document.canonical_json_bytes(), expected)
        restored = ArtifactDocument.from_json_bytes(expected)
        self.assertEqual(restored, document)
        self.assertEqual(restored.canonical_json_bytes(), expected)

    def test_json_schema_2020_12_matches_runtime_document(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("role", schema["$defs"]["sourceAsset"]["required"])
        self.assertIn(
            "geometry_hash_scope",
            schema["$defs"]["geometryRevision"]["required"],
        )
        for definition_name in (
            "sourceAsset",
            "geometryRevision",
            "sourceMetadataRevision",
            "alignRevision",
            "derivedRecord",
        ):
            self.assertFalse(schema["$defs"][definition_name]["additionalProperties"])

        jsonschema = importlib.import_module("jsonschema")
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema)
        validator.validate(_document(with_record=True).to_dict())
        validator.validate(json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))

        missing_role = _document().to_dict()
        missing_role["source_assets"][0].pop("role")
        self.assertTrue(list(validator.iter_errors(missing_role)))

        missing_hash_scope = _document().to_dict()
        missing_hash_scope["geometry_revisions"][0].pop("geometry_hash_scope")
        self.assertTrue(list(validator.iter_errors(missing_hash_scope)))

        unknown_core_field = _document().to_dict()
        unknown_core_field["future_core_field"] = True
        self.assertTrue(list(validator.iter_errors(unknown_core_field)))


if __name__ == "__main__":
    unittest.main()
