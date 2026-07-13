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
    SOURCE_INDEX_NAME,
    EmbeddedSourceEntry,
    SourceBundleError,
    SourceBundleIndex,
    source_blob_member,
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
