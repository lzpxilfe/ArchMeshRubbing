from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import src.core.artifact_survey_export as artifact_survey_export
from src.core.project_file import project_commit_backend_identifier
from src.core.project_recovery import project_recovery_publish_backend_identifier
from src.application.artifact_workflow_self_test import (
    run_artifact_workflow_self_test,
)


def test_complete_workflow_self_test_has_deterministic_offline_receipts() -> None:
    result = run_artifact_workflow_self_test()
    repeated = run_artifact_workflow_self_test()

    assert repeated == result
    assert result.source_sha256 == (
        "60b5dca0fcef8346eea22a944ce0faa160e350cb97bd028b4a319c3e26883eb5"
    )
    assert result.document_sha256 == (
        "23079da84ce02583110a105ac930a1eb9388f071e0444e793a3f1165ea6bec24"
    )
    assert result.align_revision_id == "align:workflow-self-test-explicit"
    assert (result.cutline_count, result.outline_count, result.rubbing_count) == (
        3,
        6,
        6,
    )
    assert result.geometry_metrics_count == 1
    assert (result.surface_distance_count, result.surface_diameter_count) == (1, 1)
    assert result.record_count == 18
    assert (result.vector_export_count, result.rubbing_export_count) == (9, 6)
    assert (result.tile_unwrap_count, result.tile_unwrap_export_count) == (1, 1)
    assert result.vector_set_sha256 == (
        "520cad387e3901ca15f851eb06e64e8b3a28de9e6ee19eff580359cbcc75276f"
    )
    assert result.rubbing_set_sha256 == (
        "11684a56c60aedf5e2efaa0f5a5f89df2cebcce3760509f65dc18eeacf25e918"
    )
    assert result.tile_unwrap_source_sha256 == (
        "5d1432cc1c6fe601cd2777da86a255f07689fcbfa775d0c38ae3178b28661eb6"
    )
    assert result.tile_unwrap_document_sha256 == (
        "ad871134d55502bf4d0bdebd19a2730fdb09bd6b8b761d4081dbe6552df01eb6"
    )
    assert result.tile_unwrap_sha256 == (
        "54bebe8cc01e0b3bb4ea9cbba2b982a47014d0637b5dea17c923ee1512ff93eb"
    )
    assert (
        result.tile_unwrap_sha256
        == result.tile_unwrap_recomputed_sha256
        == result.tile_unwrap_export_sha256
    )
    assert result.survey_manifest_sha256 == (
        "1bcc3b85df21f1415fba02638764364ee19b27ecb6da1d59ff196b37b7b3a928"
    )
    assert result.survey_artifact_set_sha256 == (
        "8e32fb283e3bf13eb2a72f8fc2ba220c8df2e91c6b1a9def7aa367456226a491"
    )
    for digest in (
        result.document_sha256,
        result.vector_set_sha256,
        result.rubbing_set_sha256,
        result.survey_manifest_sha256,
        result.survey_artifact_set_sha256,
        result.svg_sha256,
        result.png_sha256,
        result.tile_unwrap_document_sha256,
        result.tile_unwrap_sha256,
    ):
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")
    assert result.field_pilot_contract == "artifact-pass-human-driver-pending"
    assert result.svg_sha256 == (
        "7dcd665fe211d0eb5e9ce4b5a3e903c57685e194bbf23a52eb664119e29503eb"
    )
    assert result.png_sha256 == (
        "c0a391b54b1d535dda3956bbbb14efe14d76449bc8fdaccd9b05397d0bbf9b44"
    )
    assert result.tile_unwrap_row_shift_max_um == 6364
    assert result.tile_unwrap_row_shift_station_count == 13
    assert result.surface_area_mm2_decimal == "24.000000"
    assert result.volume_mm3_decimal == "8.000000000"
    assert result.surface_distance_mm_decimal == "1.000000"
    assert result.surface_diameter_mm_decimal == "1.000000"
    detail_tokens = {part.strip() for part in result.detail().split(",")}
    assert "checkpoint=dirty>saved>dirty>saved" in detail_tokens
    assert "exports=vector 9/9>rubbing 6/6>unwrap 1/1" in detail_tokens
    assert any(
        token.startswith(
            "unwrap=record 1/1>reopen 1/1>export 1/1>hash-match>row-shift "
        )
        for token in detail_tokens
    )
    assert result.project_commit_backend == project_commit_backend_identifier()
    assert (
        f"project_commit={project_commit_backend_identifier()}" in detail_tokens
    )
    assert (
        result.recovery_publish_backend
        == project_recovery_publish_backend_identifier()
    )
    assert (
        f"recovery_commit={project_recovery_publish_backend_identifier()}"
        in detail_tokens
    )


def test_complete_workflow_accepts_explicit_committed_directory_fsync_warning(
) -> None:
    with patch.object(
        artifact_survey_export,
        "fsync_export_directory",
        return_value=False,
    ):
        result = run_artifact_workflow_self_test()

    assert result.record_count == 18
    assert result.survey_manifest_sha256 == (
        "1bcc3b85df21f1415fba02638764364ee19b27ecb6da1d59ff196b37b7b3a928"
    )
    assert (
        result.tile_unwrap_sha256
        == result.tile_unwrap_recomputed_sha256
        == result.tile_unwrap_export_sha256
    )


def test_complete_workflow_self_test_is_qt_and_opengl_free() -> None:
    path = Path("src/application/artifact_workflow_self_test.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert not any(
        name == "PyQt6"
        or name.startswith("PyQt6.")
        or name == "OpenGL"
        or name.startswith("OpenGL.")
        or name.startswith("src.gui")
        for name in imported
    )
