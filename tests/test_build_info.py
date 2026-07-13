from __future__ import annotations

from contextlib import redirect_stdout
import io
import re
import tempfile
import json
import os
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import main
import src
from src import build_info


def test_version_and_diagnostics_use_the_source_package_version() -> None:
    assert build_info.APP_VERSION == src.__version__
    assert build_info.version_text() == f"ArchMeshRubbing {src.__version__}"

    diagnostics = build_info.collect_diagnostics()
    assert diagnostics["schema_version"] == "1.2.0"
    assert diagnostics["application"] == {
        "name": "ArchMeshRubbing",
        "distribution": "ArchMeshRubbing",
        "version": src.__version__,
    }
    resources = cast(dict[str, dict[str, object]], diagnostics["resources"])
    runtime = cast(dict[str, object], diagnostics["runtime"])
    assert resources["app_icon_png"]["present"] is True
    build = cast(dict[str, object], diagnostics["build"])
    assert re.fullmatch(r"[0-9a-f]{64}", str(build["dependency_lock_sha256"]))
    assert re.fullmatch(
        r"[0-9a-f]{64}", str(build["windows_wheel_lock_sha256"])
    )
    assert set(runtime) == {
        "numpy",
        "scipy",
        "trimesh",
        "Pillow",
        "rfc8785",
        "shapely",
        "PyQt6",
        "PyOpenGL",
    }
    assert json.loads(build_info.diagnostics_json()) == diagnostics


def test_self_test_passes_with_complete_offline_artifact_workflow() -> None:
    report = build_info.run_self_test()

    assert report["ok"] is True, report
    report_checks = cast(list[dict[str, Any]], report["checks"])
    checks = {item["name"]: item for item in report_checks}
    assert set(checks) == {
        "build_identity",
        "required_runtime",
        "resources",
        "release_evidence",
        "qt_offscreen",
        "gui_stack",
        "mesh_parsers",
        "png_codec",
        "artifact_document_canonical",
        "artifact_vector_canonical",
        "artifact_rubbing_canonical",
        "artifact_embedded_project_roundtrip",
        "artifact_complete_workflow_offline",
    }
    assert all(item["ok"] is True for item in checks.values())
    assert checks["artifact_embedded_project_roundtrip"]["detail"].endswith(
        "align=align:self-test-explicit, vertices=5"
    )
    assert checks["artifact_complete_workflow_offline"]["detail"].startswith(
        "workflow=Open>Align>Cutline 3/3>Outline 6/6>Rubbing 6/6, records=15"
    )
    diagnostics = cast(dict[str, object], report["diagnostics"])
    assert diagnostics["application"] == {
        "name": "ArchMeshRubbing",
        "distribution": "ArchMeshRubbing",
        "version": src.__version__,
    }
    assert json.loads(build_info.diagnostics_json(report)) == report


def test_runtime_self_test_rejects_installed_version_drift_from_lock() -> None:
    original = build_info._distribution_version

    def drifted(distribution: str) -> str | None:
        if distribution == "numpy":
            return "0.0.0"
        return original(distribution)

    with patch.object(build_info, "_distribution_version", side_effect=drifted):
        check = build_info._run_check(
            "required_runtime",
            build_info._check_required_runtime,
        )
    assert check.ok is False
    assert "runtime lock mismatch for numpy" in check.detail


def test_self_test_reports_a_failed_check_instead_of_raising() -> None:
    with patch(
        "src.build_info._check_resources",
        side_effect=RuntimeError("resource intentionally unavailable"),
    ):
        report = build_info.run_self_test()

    assert report["ok"] is False
    report_checks = cast(list[dict[str, Any]], report["checks"])
    resource_check = next(
        item for item in report_checks if item["name"] == "resources"
    )
    assert resource_check == {
        "name": "resources",
        "ok": False,
        "detail": "RuntimeError: resource intentionally unavailable",
    }


def test_release_cli_commands_are_machine_readable_and_return_status() -> None:
    with patch("src.core.logging_utils.setup_logging") as setup_logging:
        output = io.StringIO()
        with patch("sys.argv", ["main.py", "--version"]), redirect_stdout(output):
            assert main.run_cli() == 0
        assert output.getvalue() == f"ArchMeshRubbing {src.__version__}\n"

        output = io.StringIO()
        with patch("sys.argv", ["main.py", "--diagnostics-json"]), redirect_stdout(output):
            assert main.run_cli() == 0
        diagnostics = json.loads(output.getvalue())
        assert diagnostics["application"]["version"] == src.__version__
        setup_logging.assert_not_called()

    failure = {
        "schema_version": "1.2.0",
        "application": {"name": "ArchMeshRubbing", "version": src.__version__},
        "ok": False,
        "checks": [],
    }
    output = io.StringIO()
    with (
        patch("sys.argv", ["main.py", "--self-test"]),
        patch.object(main.build_info, "run_self_test", return_value=failure),
        redirect_stdout(output),
    ):
        assert main.run_cli() == 1
    assert json.loads(output.getvalue()) == failure


def test_frozen_no_argument_entrypoint_launches_the_gui() -> None:
    with (
        patch("sys.argv", ["ArchMeshRubbing"]),
        patch.object(main.sys, "frozen", True, create=True),
        patch.object(main, "launch_gui") as launch_gui,
    ):
        assert main.run_cli() == 0
    launch_gui.assert_called_once_with()


def test_self_test_report_cli_writes_machine_json_without_overwrite() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "frozen-self-test.json"
        with patch(
            "sys.argv",
            ["ArchMeshRubbing", "--self-test-report", str(destination)],
        ):
            assert main.run_cli() == 0
        report = json.loads(destination.read_text(encoding="utf-8"))
        assert report["ok"] is True

        with patch(
            "sys.argv",
            ["ArchMeshRubbing", "--self-test-report", str(destination)],
        ):
            assert main.run_cli() == 2


def test_native_opengl_report_cli_selects_the_windows_qpa() -> None:
    with (
        patch.object(main.sys, "platform", "win32"),
        patch(
            "src.gui.opengl_driver_smoke.main",
            return_value=0,
        ) as driver_smoke,
        patch(
            "sys.argv",
            ["ArchMeshRubbing.exe", "--opengl-driver-smoke-report", "gl.json"],
        ),
    ):
        assert main.run_cli() == 0

    driver_smoke.assert_called_once_with(
        ["--report", "gl.json", "--qt-platform", "windows"]
    )


def test_native_opengl_report_cli_rejects_missing_path() -> None:
    with patch("sys.argv", ["ArchMeshRubbing", "--opengl-driver-smoke-report"]):
        assert main.run_cli() == 2
