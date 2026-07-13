from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.artifact_rubbing_export import RUBBING_EXPORT_SIDECAR_NAME
from src.core.artifact_vector_export import VECTOR_EXPORT_SIDECAR_NAME
from src.core.artifact_verification import build_artifact_verification_report


ROOT = Path(__file__).resolve().parents[1]


def _validator():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (ROOT / "schemas/offline_verification_report-1.0.0.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def test_missing_input_is_deterministic_private_and_schema_valid(tmp_path: Path) -> None:
    missing = tmp_path / "유물-없음.amr"

    first = build_artifact_verification_report(missing)
    second = build_artifact_verification_report(missing)

    assert first == second
    assert first == {
        "artifact_kind": "unknown",
        "authority": "self_contained",
        "error": {
            "code": "input_missing",
            "message": "input does not exist",
        },
        "format": "archmeshrubbing_offline_verification",
        "input_name": "유물-없음.amr",
        "ok": False,
        "schema_version": "1.0.0",
    }
    assert str(tmp_path) not in json.dumps(first, ensure_ascii=False)
    assert list(_validator().iter_errors(first)) == []


def test_symlink_input_is_rejected_without_following_it(tmp_path: Path) -> None:
    target = tmp_path / "real.amr-vector"
    target.mkdir()
    link = tmp_path / "linked.amr-vector"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - restricted Windows developer hosts
        pytest.skip(f"symlink creation is unavailable: {exc}")

    report = build_artifact_verification_report(link)

    assert report["ok"] is False
    assert report["artifact_kind"] == "unknown"
    assert report["error"] == {
        "code": "input_symlink",
        "message": "symbolic-link inputs are not accepted",
    }
    assert list(_validator().iter_errors(report)) == []


def test_multiple_export_markers_fail_as_ambiguous(tmp_path: Path) -> None:
    package = tmp_path / "mixed-package"
    package.mkdir()
    (package / VECTOR_EXPORT_SIDECAR_NAME).write_bytes(b"{}\n")
    (package / RUBBING_EXPORT_SIDECAR_NAME).write_bytes(b"{}\n")

    report = build_artifact_verification_report(package)

    assert report["ok"] is False
    assert report["error"] == {
        "code": "input_ambiguous",
        "message": "directory contains more than one export-package marker",
    }
    assert list(_validator().iter_errors(report)) == []


def test_validator_failure_keeps_kind_and_redacts_absolute_paths(
    tmp_path: Path,
) -> None:
    package = tmp_path / "broken.amr-vector"
    package.mkdir()
    (package / VECTOR_EXPORT_SIDECAR_NAME).write_bytes(b"{}\n")

    with patch(
        "src.core.artifact_verification.validate_vector_export_package",
        side_effect=RuntimeError(f"cannot read {package / 'private.svg'}"),
    ):
        report = build_artifact_verification_report(package)

    assert report["ok"] is False
    assert report["artifact_kind"] == "vector_export"
    assert report["error"]["code"] == "verification_failed"
    assert "<path>" in report["error"]["message"]
    assert str(tmp_path) not in report["error"]["message"]
    assert list(_validator().iter_errors(report)) == []


def test_report_schema_is_closed() -> None:
    report = build_artifact_verification_report("definitely-missing.amr")
    report["unexpected"] = True

    assert list(_validator().iter_errors(report))
