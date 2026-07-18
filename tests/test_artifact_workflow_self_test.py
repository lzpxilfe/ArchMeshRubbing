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

    assert result.source_sha256 == (
        "60b5dca0fcef8346eea22a944ce0faa160e350cb97bd028b4a319c3e26883eb5"
    )
    assert result.document_sha256 == (
        "6a78295dfecc901c7e9714111f88d517ea7af509f0c983184fbd882c77fce349"
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
    assert result.vector_set_sha256 == (
        "78936893e44a2d5e0f6eb1bc30961204d3ce752a2e279b5389b74395d9fbb3f3"
    )
    assert result.rubbing_set_sha256 == (
        "14a9fcbd576e9e7584dfe86edf0a65f13839661b92394bc91f6407a62985f25c"
    )
    assert result.survey_manifest_sha256 == (
        "124ab0217c50eed098c37ca502cdd58d5231a942fadc3e2a8a88f5b375389950"
    )
    assert result.survey_artifact_set_sha256 == (
        "35aecfd83e8450b59d924f7cba81cf337a2b517e3bdc0e0d360a154aee70345b"
    )
    assert result.field_pilot_contract == "artifact-pass-human-driver-pending"
    assert result.svg_sha256 == (
        "787b19c70f36a6479cc9d196f86d187932a70f1024d6782810322f335a15f5b6"
    )
    assert result.png_sha256 == (
        "447cbb8244758926ff036ed681b5e677094fd3f454491392b861b9338e3835e7"
    )
    assert result.surface_area_mm2_decimal == "24.000000"
    assert result.volume_mm3_decimal == "8.000000000"
    assert result.surface_distance_mm_decimal == "1.000000"
    assert result.surface_diameter_mm_decimal == "1.000000"
    detail_tokens = {part.strip() for part in result.detail().split(",")}
    assert "checkpoint=dirty>saved>dirty>saved" in detail_tokens
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
        "124ab0217c50eed098c37ca502cdd58d5231a942fadc3e2a8a88f5b375389950"
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
