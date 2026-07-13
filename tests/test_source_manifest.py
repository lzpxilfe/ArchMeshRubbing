from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from src.core.source_manifest import (
    DEPENDENCY_RESOURCE_ROLE,
    PRIMARY_RESOURCE_ROLE,
    SourceManifest,
    SourceManifestEntry,
    SourceManifestError,
    fixed_media_type,
    resolve_logical_reference,
)


ROOT = Path(__file__).resolve().parents[1]


def _entry(
    logical_path: str,
    digest: str,
    *,
    role: str = DEPENDENCY_RESOURCE_ROLE,
) -> SourceManifestEntry:
    return SourceManifestEntry(
        logical_path=logical_path,
        media_type=fixed_media_type(logical_path),
        role=role,
        sha256=digest * 64,
        size_bytes=1,
    )


def _manifest() -> SourceManifest:
    return SourceManifest(
        primary_logical_path="scan.obj",
        entries=(
            _entry("texture.png", "b"),
            _entry("scan.obj", "a", role=PRIMARY_RESOURCE_ROLE),
            _entry("materials/scan.mtl", "c"),
        ),
    )


def test_manifest_is_canonical_and_schema_valid() -> None:
    manifest = _manifest()
    payload = manifest.to_dict()
    schema = json.loads(
        (ROOT / "schemas" / "source_manifest-1.0.0.schema.json").read_text(
            encoding="utf-8"
        )
    )

    jsonschema.Draft202012Validator(schema).validate(payload)
    assert [entry["logical_path"] for entry in payload["entries"]] == [
        "materials/scan.mtl",
        "scan.obj",
        "texture.png",
    ]
    assert SourceManifest.from_dict(payload) == manifest
    assert len(manifest.canonical_sha256) == 64


@pytest.mark.parametrize(
    "path",
    [
        "../outside.png",
        "nested/../../outside.png",
        "/absolute.png",
        "C:/drive.png",
        "https://example.invalid/a.png",
        "nested\\texture.png",
        "nested//texture.png",
        "./texture.png",
        "texture.png ",
        "bad\x00name.png",
    ],
)
def test_durable_logical_paths_reject_nonportable_values(path: str) -> None:
    with pytest.raises(SourceManifestError):
        _entry(path, "a")


def test_parser_reference_normalization_is_contained_and_portable() -> None:
    assert resolve_logical_reference("materials", "../textures\\scan.png") == (
        "textures/scan.png"
    )
    assert resolve_logical_reference("", "./textures/scan.png") == (
        "textures/scan.png"
    )
    with pytest.raises(SourceManifestError, match="escapes"):
        resolve_logical_reference("", "../../secret.png")
    with pytest.raises(SourceManifestError, match="URI"):
        resolve_logical_reference("", "file:///secret.png")


def test_manifest_allows_content_aliases_but_not_path_alias_collisions() -> None:
    aliased = SourceManifest(
        primary_logical_path="scan.obj",
        entries=(
            _entry("scan.obj", "a", role=PRIMARY_RESOURCE_ROLE),
            _entry("a/texture.png", "b"),
            _entry("b/texture.png", "b"),
        ),
    )
    assert len({entry.sha256 for entry in aliased.entries}) == 2

    with pytest.raises(SourceManifestError, match="logical paths"):
        SourceManifest(
            primary_logical_path="scan.obj",
            entries=(
                _entry("scan.obj", "a", role=PRIMARY_RESOURCE_ROLE),
                _entry("texture.png", "b"),
                _entry("texture.png", "c"),
            ),
        )


def test_manifest_reader_rejects_type_coercion_and_media_type_drift() -> None:
    payload = _manifest().to_dict()
    payload["primary_logical_path"] = 123
    with pytest.raises(SourceManifestError, match="must be a non-empty string"):
        SourceManifest.from_dict(payload)

    payload = _manifest().to_dict()
    raw_entries = payload["entries"]
    assert isinstance(raw_entries, list)
    first_entry = raw_entries[0]
    assert isinstance(first_entry, dict)
    first_entry["sha256"] = 7
    with pytest.raises(SourceManifestError, match="must be a non-empty string"):
        SourceManifest.from_dict(payload)

    with pytest.raises(SourceManifestError, match="fixed logical-path media type"):
        SourceManifestEntry(
            logical_path="texture.png",
            media_type="application/octet-stream",
            role=DEPENDENCY_RESOURCE_ROLE,
            sha256="a" * 64,
            size_bytes=1,
        )
