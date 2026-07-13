from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path
import unittest
from unittest.mock import PropertyMock, patch

from tests.test_artifact_document import _document

from src.core.artifact_document import ArtifactDocument
from src.core.source_bundle import (
    SOURCE_BLOB_PREFIX,
    SOURCE_BUNDLE_FORMAT,
    SOURCE_BUNDLE_SCHEMA_VERSION,
    SOURCE_BUNDLE_SCHEMA_VERSION_CLOSED_MANIFEST,
    SOURCE_INDEX_NAME,
    EmbeddedSourceEntry,
    SourceBundleError,
    SourceBundleIndex,
    source_blob_member,
)
from src.core.mesh_import_recipe import (
    current_mesh_import_recipe,
    mesh_import_recipe_with_manifest,
)
from src.core.source_manifest import (
    DEPENDENCY_RESOURCE_ROLE,
    PRIMARY_RESOURCE_ROLE,
    SourceManifest,
    SourceManifestEntry,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "a" * 64
OTHER_SHA = "b" * 64


def _entry(
    *,
    digest: str = SOURCE_SHA,
    role: str = "primary_mesh",
    logical_path: str = "artifact.ply",
) -> EmbeddedSourceEntry:
    return EmbeddedSourceEntry(
        source_asset_id=f"sha256:{digest}",
        role=role,
        logical_path=logical_path,
        media_type="model/ply",
        member=source_blob_member(digest),
        sha256=digest,
        size_bytes=123,
    )


class TestSourceBundle(unittest.TestCase):
    def test_document_index_has_canonical_roundtrip_and_schema(self):
        document = _document(with_record=True)
        index = SourceBundleIndex.for_document(document)

        self.assertEqual(SOURCE_INDEX_NAME, "sources/index.json")
        self.assertEqual(SOURCE_BLOB_PREFIX, "sources/blobs/sha256/")
        self.assertEqual(index.document_id, document.document_id)
        self.assertEqual(index.document_sha256, document.canonical_sha256)
        self.assertEqual(index.primary_source_asset_id, document.source_assets[0].id)
        self.assertEqual(index.entries[0].logical_path, "artifact.ply")
        self.assertEqual(index.entries[0].member, source_blob_member(SOURCE_SHA))

        expected = {
            "document_id": "artifact:test-1",
            "document_sha256": document.canonical_sha256,
            "entries": [
                {
                    "logical_path": "artifact.ply",
                    "media_type": "model/ply",
                    "member": f"sources/blobs/sha256/{SOURCE_SHA}",
                    "role": "primary_mesh",
                    "sha256": SOURCE_SHA,
                    "size_bytes": 123,
                    "source_asset_id": f"sha256:{SOURCE_SHA}",
                }
            ],
            "format": SOURCE_BUNDLE_FORMAT,
            "primary_source_asset_id": f"sha256:{SOURCE_SHA}",
            "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
        }
        self.assertEqual(index.to_dict(), expected)
        self.assertEqual(
            index.canonical_json_bytes(),
            json.dumps(
                expected,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )

        restored = SourceBundleIndex.from_dict(
            json.loads(index.canonical_json_bytes())
        )
        self.assertEqual(restored, index)
        self.assertEqual(restored.canonical_json_bytes(), index.canonical_json_bytes())

        schema = json.loads(
            (ROOT / "schemas" / "source_bundle-1.0.0.schema.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema = importlib.import_module("jsonschema")
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(index.to_dict())

    def test_entries_are_immutably_and_deterministically_sorted(self):
        primary = _entry(logical_path="z/source.ply")
        auxiliary = _entry(
            digest=OTHER_SHA,
            role="texture",
            logical_path="a/texture.png",
        )

        index = SourceBundleIndex(
            document_id="artifact:sort",
            document_sha256="c" * 64,
            primary_source_asset_id=primary.source_asset_id,
            entries=(primary, auxiliary),
        )

        self.assertEqual(
            tuple(entry.logical_path for entry in index.entries),
            ("a/texture.png", "z/source.ply"),
        )
        with self.assertRaisesRegex(AttributeError, "cannot assign"):
            index.document_id = "changed"  # type: ignore[misc]

    def test_v2_document_index_carries_closed_dependencies_and_content_aliases(self):
        document = _document(with_record=True)
        manifest = SourceManifest(
            primary_logical_path="artifact.ply",
            entries=(
                SourceManifestEntry(
                    logical_path="artifact.ply",
                    media_type="model/ply",
                    role=PRIMARY_RESOURCE_ROLE,
                    sha256=SOURCE_SHA,
                    size_bytes=123,
                ),
                SourceManifestEntry(
                    logical_path="textures/a.png",
                    media_type="image/png",
                    role=DEPENDENCY_RESOURCE_ROLE,
                    sha256=OTHER_SHA,
                    size_bytes=10,
                ),
                SourceManifestEntry(
                    logical_path="textures/b.png",
                    media_type="image/png",
                    role=DEPENDENCY_RESOURCE_ROLE,
                    sha256=OTHER_SHA,
                    size_bytes=10,
                ),
            ),
        )
        recipe = mesh_import_recipe_with_manifest(
            current_mesh_import_recipe("ply"),
            manifest,
        )
        geometry = replace(document.geometry_revisions[0], import_recipe=recipe)
        document = replace(document, geometry_revisions=(geometry,))

        index = SourceBundleIndex.for_document(document)

        self.assertEqual(
            index.schema_version,
            SOURCE_BUNDLE_SCHEMA_VERSION_CLOSED_MANIFEST,
        )
        self.assertEqual(len(index.entries), 3)
        alias_members = [
            entry.member
            for entry in index.entries
            if entry.sha256 == OTHER_SHA
        ]
        self.assertEqual(len(alias_members), 2)
        self.assertEqual(len(set(alias_members)), 1)
        schema = json.loads(
            (ROOT / "schemas" / "source_bundle-2.0.0.schema.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema = importlib.import_module("jsonschema")
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(index.to_dict())
        self.assertEqual(
            SourceBundleIndex.from_dict(index.to_dict()),
            index,
        )

    def test_paths_reject_traversal_absolute_drive_unc_uri_and_backslash(self):
        invalid_paths = (
            "../artifact.ply",
            "mesh/../artifact.ply",
            "/artifact.ply",
            "C:/artifact.ply",
            "//server/share/artifact.ply",
            r"server\share\artifact.ply",
            "https://example.test/artifact.ply",
            "mesh//artifact.ply",
            "./artifact.ply",
            "mesh/scan\u0000.ply",
        )
        for invalid in invalid_paths:
            with self.subTest(path=invalid):
                with self.assertRaises(SourceBundleError):
                    _entry(logical_path=invalid)

        with self.assertRaises(SourceBundleError):
            replace(_entry(), member="sources/blobs/sha256/../escape")

    def test_v2_index_enforces_the_portable_entry_budget(self):
        primary = _entry()
        dependencies = tuple(
            _entry(
                digest=f"{index:064x}",
                role=DEPENDENCY_RESOURCE_ROLE,
                logical_path=f"dependencies/{index}.ply",
            )
            for index in range(1, 62)
        )
        with self.assertRaisesRegex(SourceBundleError, "too many entries"):
            SourceBundleIndex(
                document_id="artifact:entry-budget",
                document_sha256="c" * 64,
                primary_source_asset_id=primary.source_asset_id,
                entries=(primary, *dependencies),
                schema_version=SOURCE_BUNDLE_SCHEMA_VERSION_CLOSED_MANIFEST,
            )

    def test_v2_index_requires_a_dependency_and_closed_roles(self):
        primary = _entry()
        with self.assertRaisesRegex(SourceBundleError, "at least one dependency"):
            SourceBundleIndex(
                document_id="artifact:v2-primary-only",
                document_sha256="c" * 64,
                primary_source_asset_id=primary.source_asset_id,
                entries=(primary,),
                schema_version=SOURCE_BUNDLE_SCHEMA_VERSION_CLOSED_MANIFEST,
            )

        unsupported = _entry(
            digest=OTHER_SHA,
            role="texture",
            logical_path="texture.ply",
        )
        with self.assertRaisesRegex(SourceBundleError, "unsupported role"):
            SourceBundleIndex(
                document_id="artifact:v2-role",
                document_sha256="c" * 64,
                primary_source_asset_id=primary.source_asset_id,
                entries=(primary, unsupported),
                schema_version=SOURCE_BUNDLE_SCHEMA_VERSION_CLOSED_MANIFEST,
            )

    def test_duplicate_paths_members_and_primary_mismatch_fail_closed(self):
        primary = _entry()
        same_path = _entry(
            digest=OTHER_SHA,
            role="texture",
            logical_path=primary.logical_path,
        )
        with self.assertRaisesRegex(SourceBundleError, "unique logical_path"):
            SourceBundleIndex(
                document_id="artifact:duplicate-path",
                document_sha256="c" * 64,
                primary_source_asset_id=primary.source_asset_id,
                entries=(primary, same_path),
            )

        same_member = replace(
            _entry(digest=OTHER_SHA, role="texture", logical_path="texture.png"),
            source_asset_id=primary.source_asset_id,
            member=primary.member,
            sha256=primary.sha256,
        )
        with self.assertRaisesRegex(SourceBundleError, "unique member"):
            SourceBundleIndex(
                document_id="artifact:duplicate-member",
                document_sha256="c" * 64,
                primary_source_asset_id=primary.source_asset_id,
                entries=(primary, same_member),
            )

        with self.assertRaisesRegex(SourceBundleError, "exactly one primary_mesh"):
            SourceBundleIndex(
                document_id="artifact:no-primary",
                document_sha256="c" * 64,
                primary_source_asset_id=primary.source_asset_id,
                entries=(replace(primary, role="texture"),),
            )
        with self.assertRaisesRegex(SourceBundleError, "must identify"):
            SourceBundleIndex(
                document_id="artifact:wrong-primary",
                document_sha256="c" * 64,
                primary_source_asset_id=f"sha256:{OTHER_SHA}",
                entries=(primary,),
            )

    def test_hash_size_member_and_exact_keys_mismatches_are_rejected(self):
        with self.assertRaisesRegex(SourceBundleError, "lowercase"):
            source_blob_member("A" * 64)
        with self.assertRaisesRegex(SourceBundleError, "non-negative"):
            replace(_entry(), size_bytes=-1)
        with self.assertRaisesRegex(SourceBundleError, "source_asset_id"):
            replace(_entry(), source_asset_id=f"sha256:{OTHER_SHA}")
        with self.assertRaisesRegex(SourceBundleError, "content-addressed member"):
            replace(_entry(), member=source_blob_member(OTHER_SHA))

        entry_payload = _entry().to_dict()
        entry_payload["unknown"] = True
        with self.assertRaisesRegex(SourceBundleError, "unknown fields"):
            EmbeddedSourceEntry.from_dict(entry_payload)

        index_payload = SourceBundleIndex(
            document_id="artifact:closed",
            document_sha256="c" * 64,
            primary_source_asset_id=f"sha256:{SOURCE_SHA}",
            entries=(_entry(),),
        ).to_dict()
        index_payload.pop("document_sha256")
        with self.assertRaisesRegex(SourceBundleError, "missing fields"):
            SourceBundleIndex.from_dict(index_payload)

    def test_for_document_requires_exactly_one_source_asset(self):
        empty = ArtifactDocument.empty(
            document_id="artifact:empty",
            software_version="test",
        )
        with self.assertRaisesRegex(SourceBundleError, "exactly one SourceAsset"):
            SourceBundleIndex.for_document(empty)

        with patch.object(
            ArtifactDocument,
            "canonical_sha256",
            new_callable=PropertyMock,
            return_value="f" * 64,
        ):
            with self.assertRaisesRegex(SourceBundleError, "canonical bytes"):
                SourceBundleIndex.for_document(_document())


if __name__ == "__main__":
    unittest.main()
