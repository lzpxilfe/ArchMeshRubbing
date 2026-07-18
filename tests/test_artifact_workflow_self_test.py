from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import src.core.artifact_survey_export as artifact_survey_export
from src.application.artifact_workflow_self_test import (
    run_artifact_workflow_self_test,
)


def test_complete_workflow_self_test_has_deterministic_offline_receipts() -> None:
    result = run_artifact_workflow_self_test()

    assert result.source_sha256 == (
        "60b5dca0fcef8346eea22a944ce0faa160e350cb97bd028b4a319c3e26883eb5"
    )
    assert result.document_sha256 == (
        "8d40f937da122701942ddd453ce0d79a60a2e87a820e7d9631faf1f7aff04045"
    )
    assert result.align_revision_id == "align:workflow-self-test-explicit"
    assert (result.cutline_count, result.outline_count, result.rubbing_count) == (
        3,
        6,
        6,
    )
    assert result.geometry_metrics_count == 1
    assert result.record_count == 16
    assert (result.vector_export_count, result.rubbing_export_count) == (9, 6)
    assert result.vector_set_sha256 == (
        "11b6a4627a7998bf32b33d1e8db424942bf93753f1ad9eb7ecc16da8f64fcb8b"
    )
    assert result.rubbing_set_sha256 == (
        "0544d4b594d947ae82407c37b68c18ca94af98c856c375dbe2ada1fe15bd77a8"
    )
    assert result.survey_manifest_sha256 == (
        "c01c186f539b9cabdcab81bd4da2777aec247a166ebfebc32434445394d85796"
    )
    assert result.survey_artifact_set_sha256 == (
        "9546d4070894be10c8881ff2e5c10ee708d7fd8a92b73e91e6800bf42fede8ae"
    )
    assert result.field_pilot_contract == "artifact-pass-human-driver-pending"
    assert result.svg_sha256 == (
        "e67be16ad9d7ee764369b883329220610b315737b97a92facfe1038bde049f48"
    )
    assert result.png_sha256 == (
        "f27f1050618db7f2e99cf25b0620ce6304a80dfc9702469d8f1e2410851af1f4"
    )
    assert result.surface_area_mm2_decimal == "24.000000"
    assert result.volume_mm3_decimal == "8.000000000"


def test_complete_workflow_accepts_explicit_committed_directory_fsync_warning(
) -> None:
    with patch.object(
        artifact_survey_export,
        "fsync_export_directory",
        return_value=False,
    ):
        result = run_artifact_workflow_self_test()

    assert result.record_count == 16
    assert result.survey_manifest_sha256 == (
        "c01c186f539b9cabdcab81bd4da2777aec247a166ebfebc32434445394d85796"
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
