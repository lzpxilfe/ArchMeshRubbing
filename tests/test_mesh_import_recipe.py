from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from src.core.mesh_import_recipe import (
    MESH_IMPORT_RECIPE_ID,
    MESH_IMPORT_RECIPE_VERSION,
    MeshImportRecipeError,
    current_mesh_import_recipe,
    mesh_import_receipt_matches_base,
    mesh_import_recipe_with_manifest,
    validate_mesh_import_recipe,
)
from src.core.source_manifest import (
    DEPENDENCY_RESOURCE_ROLE,
    PRIMARY_RESOURCE_ROLE,
    SourceManifest,
    SourceManifestEntry,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "mesh_import_recipe-1.0.0.schema.json"
)


def test_current_recipe_is_closed_schema_valid_and_runtime_bound() -> None:
    recipe = current_mesh_import_recipe(".PLY")

    assert set(recipe) == {
        "recipe_id",
        "recipe_version",
        "format",
        "loader",
        "loader_version",
        "parser_runtime_sha256",
        "runtime_lock_sha256",
        "force",
        "process",
        "maintain_order",
        "scene_merge",
        "sanitizer",
        "dependency_policy",
    }
    assert recipe["recipe_id"] == MESH_IMPORT_RECIPE_ID
    assert recipe["recipe_version"] == MESH_IMPORT_RECIPE_VERSION
    assert recipe["format"] == "ply"
    assert recipe["loader"] == "trimesh"
    assert recipe["loader_version"] == "4.11.5"
    assert recipe["parser_runtime_sha256"] == (
        "930cc48cbc94f91c867ebb79e2976c589f7135cd8642e6565c04a44110efddef"
    )
    assert recipe["runtime_lock_sha256"] == (
        "d460c6f403b0c7b2aea3bf30d5f63c06b85c76ef7d7317989b66461db3754400"
    )

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(recipe)

    execution = validate_mesh_import_recipe(recipe, allow_legacy=False)
    assert execution.source_format == "ply"
    assert not execution.legacy_unversioned
    assert execution.strict_recipe() == recipe


def test_only_exact_pre_v1_recipe_is_accepted_as_legacy() -> None:
    legacy = {
        "format": "ply",
        "loader": "trimesh",
        "maintain_order": True,
        "process": False,
        "sanitizer": "meshdata-v1",
    }

    execution = validate_mesh_import_recipe(legacy, allow_legacy=True)
    assert execution.source_format == "ply"
    assert execution.legacy_unversioned

    with pytest.raises(MeshImportRecipeError, match="legacy unversioned"):
        validate_mesh_import_recipe(legacy, allow_legacy=False)

    official_document_profile = {"format": "ply", "process": False}
    official_execution = validate_mesh_import_recipe(
        official_document_profile,
        allow_legacy=True,
    )
    assert official_execution.source_format == "ply"
    assert official_execution.legacy_unversioned

    for field, bad_value in (
        ("loader", "other"),
        ("maintain_order", False),
        ("process", True),
        ("sanitizer", "other"),
    ):
        changed = dict(legacy)
        changed[field] = bad_value
        with pytest.raises(MeshImportRecipeError):
            validate_mesh_import_recipe(changed, allow_legacy=True)


def test_strict_recipe_rejects_unknown_flags_and_runtime_drift() -> None:
    recipe = current_mesh_import_recipe("ply")

    missing = dict(recipe)
    missing.pop("force")
    with pytest.raises(MeshImportRecipeError, match="missing fields"):
        validate_mesh_import_recipe(missing, allow_legacy=True)

    unknown = {**recipe, "source_path": "/secret/artifact.ply"}
    with pytest.raises(MeshImportRecipeError, match="unknown fields"):
        validate_mesh_import_recipe(unknown, allow_legacy=True)

    for field, bad_value, message in (
        ("process", True, "process"),
        ("maintain_order", False, "maintain_order"),
        ("force", "scene", "force"),
        ("loader_version", "0.0.0", "Trimesh version"),
        ("parser_runtime_sha256", "0" * 64, "parser-runtime"),
    ):
        changed = copy.deepcopy(recipe)
        changed[field] = bad_value
        with pytest.raises(MeshImportRecipeError, match=message):
            validate_mesh_import_recipe(changed, allow_legacy=True)

    for stored_format in ("PLY", ".ply", " ply", "ply "):
        changed = copy.deepcopy(recipe)
        changed["format"] = stored_format
        with pytest.raises(MeshImportRecipeError, match="normalized lowercase"):
            validate_mesh_import_recipe(changed, allow_legacy=True)


def test_saved_recipe_can_be_inspected_without_dispatching_wrong_runtime() -> None:
    recipe = current_mesh_import_recipe("off")
    recipe["loader_version"] = "99.0.0"
    recipe["parser_runtime_sha256"] = "b" * 64
    recipe["runtime_lock_sha256"] = "a" * 64

    inspected = validate_mesh_import_recipe(
        recipe,
        allow_legacy=False,
        require_current_runtime=False,
    )
    assert inspected.source_format == "off"
    assert inspected.loader_version == "99.0.0"

    with pytest.raises(MeshImportRecipeError, match="Trimesh version"):
        validate_mesh_import_recipe(recipe, allow_legacy=False)


def test_full_runtime_lock_is_provenance_not_an_unrelated_execution_gate() -> None:
    recipe = current_mesh_import_recipe("ply")
    recipe["runtime_lock_sha256"] = "a" * 64

    execution = validate_mesh_import_recipe(recipe, allow_legacy=False)

    assert execution.runtime_lock_sha256 == "a" * 64


def test_closed_manifest_recipe_is_schema_valid_and_matches_capture_base() -> None:
    base = current_mesh_import_recipe("obj")
    manifest = SourceManifest(
        primary_logical_path="scan.obj",
        entries=(
            SourceManifestEntry(
                logical_path="scan.obj",
                media_type="model/obj",
                role=PRIMARY_RESOURCE_ROLE,
                sha256="a" * 64,
                size_bytes=10,
            ),
            SourceManifestEntry(
                logical_path="material.mtl",
                media_type="model/mtl",
                role=DEPENDENCY_RESOURCE_ROLE,
                sha256="b" * 64,
                size_bytes=20,
            ),
        ),
    )

    receipt = mesh_import_recipe_with_manifest(base, manifest)
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "mesh_import_recipe-2.0.0.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(receipt)
    execution = validate_mesh_import_recipe(receipt, allow_legacy=False)

    assert receipt["recipe_version"] == "2.0.0"
    assert receipt["dependency_policy"] == "closed_manifest"
    assert execution.source_manifest == manifest
    assert execution.strict_recipe() == receipt
    assert mesh_import_receipt_matches_base(base, receipt)

    changed_base = dict(base)
    changed_base["format"] = "ply"
    assert not mesh_import_receipt_matches_base(changed_base, receipt)


def test_manifest_builder_keeps_v1_for_a_self_contained_primary() -> None:
    base = current_mesh_import_recipe("ply")
    manifest = SourceManifest(
        primary_logical_path="scan.ply",
        entries=(
            SourceManifestEntry(
                logical_path="scan.ply",
                media_type="model/ply",
                role=PRIMARY_RESOURCE_ROLE,
                sha256="a" * 64,
                size_bytes=10,
            ),
        ),
    )

    assert mesh_import_recipe_with_manifest(base, manifest) == base
