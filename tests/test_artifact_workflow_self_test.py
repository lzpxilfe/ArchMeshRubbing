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


#: The pinned digests below moved once, when outline algorithm 1.4.0 and the
#: 1.8.0 sidecar arrived: the welded-fragment gate refuses a crumb the
#: closing joined to the artifact, and its record names the new version and
#: carries the count that gate judges (`grid_pre_closing_component_count`).
#: Only the outline payload changed - the source, the tile unwrap and its
#: document are byte-identical - and every other digest here moved because
#: an export embeds the document's.  Outlines written at 1.3.0 and earlier
#: still recompute to the bytes they were written with.  They moved again
#: with vector sidecar 1.9.0, which holds the section mark in the line-kind
#: vocabulary, and again with 1.10.0, which holds the presumed stretch of the
#: cut: every export names the schema it was written under.
def test_complete_workflow_self_test_has_deterministic_offline_receipts() -> None:
    result = run_artifact_workflow_self_test()
    repeated = run_artifact_workflow_self_test()

    assert repeated == result
    assert result.source_sha256 == (
        "60b5dca0fcef8346eea22a944ce0faa160e350cb97bd028b4a319c3e26883eb5"
    )
    assert result.document_sha256 == (
        "91aa4360a2fd005bbe2e1009c9d52ca352137ecb035ad3362ba76b9e161e3a91"
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
        "9516e4813185faf0781686871678400b83c9a9a6e5aaf74d1def42e93994fb03"
    )
    assert result.rubbing_set_sha256 == (
        "4b81e1a1a408f34252c79d84b140d871bc21fbc0e4178dfff0fd92844a3bbb5a"
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
        "ced818d9715fc152db5631d58d4751b1cd16b40a3011c399b10881cc59e609e3"
    )
    assert result.survey_artifact_set_sha256 == (
        "8785d1fba01168bbb2c500a5c67c853421d3198268c8bec9a0db0539464ea1ca"
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
        "110eef61b0e33cbc936152b0ea338210a4e847caee07e0bb7cab34abdf72fd17"
    )
    assert result.png_sha256 == (
        "b03ebe131be472d43145f66e59a26b6d3787a6ea5a1e019e05d5f180d446d68f"
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
        "ced818d9715fc152db5631d58d4751b1cd16b40a3011c399b10881cc59e609e3"
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
