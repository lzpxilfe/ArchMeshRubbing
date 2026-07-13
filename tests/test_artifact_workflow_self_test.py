from __future__ import annotations

import ast
from pathlib import Path

from src.application.artifact_workflow_self_test import (
    run_artifact_workflow_self_test,
)


def test_complete_workflow_self_test_has_deterministic_offline_receipts() -> None:
    result = run_artifact_workflow_self_test()

    assert result.source_sha256 == (
        "60b5dca0fcef8346eea22a944ce0faa160e350cb97bd028b4a319c3e26883eb5"
    )
    assert result.document_sha256 == (
        "f9f8e845843a2170751319c88be4d15abe66e327de6ef1bd074c75e997ea416a"
    )
    assert result.align_revision_id == "align:workflow-self-test-explicit"
    assert (result.cutline_count, result.outline_count, result.rubbing_count) == (
        3,
        6,
        6,
    )
    assert result.record_count == 15
    assert (result.vector_export_count, result.rubbing_export_count) == (9, 6)
    assert result.vector_set_sha256 == (
        "180db90e564ebb0483fe43d167d08f970468da17cffa3f9870d4dba6655faba8"
    )
    assert result.rubbing_set_sha256 == (
        "849a66fa9b79ebbe4520a5acfc5feec29c7fc2cfa3c25d610ca07e1fac10cce5"
    )
    assert result.survey_manifest_sha256 == (
        "bfa36ee1595405d4568f90cae54ca175954b75e6ca5d87364330b3532e260e10"
    )
    assert result.survey_artifact_set_sha256 == (
        "5d607381f90a3f78f5211d47ea9507f3a55c1196eb3ab8d8c991ca1c077d08b1"
    )
    assert result.svg_sha256 == (
        "aaf2c7c1136242074fbc2862894c821673646687c55cea36f1f6e3408c569d23"
    )
    assert result.png_sha256 == (
        "268ad9fc5910358a2a2882809c6928d9c9fd66685ed25bc2b182ccc211336e8d"
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
