from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from tools import build_native


def test_artifact_layout_uses_onedir_and_platform_launchers() -> None:
    dist = Path("/tmp/native-dist")

    windows = build_native.artifact_layout(dist, platform_name="win32")
    assert windows.onedir == dist / "ArchMeshRubbing"
    assert windows.executable == windows.onedir / "ArchMeshRubbing.exe"
    assert windows.app_bundle is None

    linux = build_native.artifact_layout(dist, platform_name="linux")
    assert linux.executable == linux.onedir / "ArchMeshRubbing"
    assert linux.app_bundle is None

    macos = build_native.artifact_layout(dist, platform_name="darwin")
    assert macos.onedir == dist / "ArchMeshRubbing"
    assert macos.app_bundle == dist / "ArchMeshRubbing.app"
    assert macos.executable == (
        dist / "ArchMeshRubbing.app" / "Contents" / "MacOS" / "ArchMeshRubbing"
    )


def test_exact_lock_is_recursive_platform_aware_and_rejects_ranges() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "runtime.lock").write_text(
            "numpy==2.4.3\nwidget==1.0; sys_platform == 'win32'\n",
            encoding="utf-8",
        )
        build_lock = root / "build.lock"
        build_lock.write_text(
            "-r runtime.lock\nPyInstaller==6.21.0\n"
            "macholib==1.16.4; sys_platform == \"darwin\"\n",
            encoding="utf-8",
        )
        assert build_native.exact_lock_pins(
            build_lock,
            platform_name="darwin",
        ) == {
            "numpy": ("numpy", "2.4.3"),
            "pyinstaller": ("PyInstaller", "6.21.0"),
            "macholib": ("macholib", "1.16.4"),
        }

        build_lock.write_text("PyInstaller>=6\n", encoding="utf-8")
        with pytest.raises(build_native.NativeBuildError, match="exact dependency pin"):
            build_native.exact_lock_pins(build_lock)


def test_environment_requires_python_312_and_every_exact_pin() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        lock = Path(temporary) / "build.lock"
        lock.write_text("PyInstaller==6.21.0\n", encoding="utf-8")
        with pytest.raises(build_native.NativeBuildError, match="Python 3.12"):
            build_native.validate_build_environment(
                lock,
                python_version=(3, 13),
                installed_versions={"pyinstaller": "6.21.0"},
            )
        with pytest.raises(build_native.NativeBuildError, match="expected 6.21.0"):
            build_native.validate_build_environment(
                lock,
                python_version=(3, 12),
                installed_versions={"pyinstaller": "6.20.0"},
            )
        pins = build_native.validate_build_environment(
            lock,
            python_version=(3, 13),
            installed_versions={"pyinstaller": "6.21.0"},
            allow_python_version_mismatch=True,
        )
        assert pins["pyinstaller"] == ("PyInstaller", "6.21.0")


def test_existing_outputs_fail_closed_without_deleting_or_running_build() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "requirements").mkdir()
        (root / "requirements" / "runtime-py312.lock").write_text(
            "demo==1\n", encoding="utf-8"
        )
        (root / "requirements" / "build-py312.lock").write_text(
            "demo==1\n", encoding="utf-8"
        )
        (root / "requirements" / "windows-py312-x64-hashed.lock").write_text(
            "demo==1 --hash=sha256:" + "0" * 64 + "\n", encoding="utf-8"
        )
        (root / "ArchMeshRubbing.spec").write_text("# spec\n", encoding="utf-8")
        existing = root / "dist" / "ArchMeshRubbing"
        existing.mkdir(parents=True)
        sentinel = existing / "keep-me.txt"
        sentinel.write_text("user evidence", encoding="utf-8")

        with (
            patch.object(build_native, "validate_build_environment"),
            patch.object(build_native.subprocess, "run") as run,
            pytest.raises(build_native.NativeBuildError, match="refusing to overwrite"),
        ):
            build_native.build_native(
                root=root,
                platform_name="linux",
                commit="a" * 40,
            )
        run.assert_not_called()
        assert sentinel.read_text(encoding="utf-8") == "user evidence"


def test_build_invokes_current_python_locates_app_and_runs_file_self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        requirements = root / "requirements"
        requirements.mkdir()
        runtime_lock = requirements / "runtime-py312.lock"
        runtime_lock.write_text("demo==1\n", encoding="utf-8")
        (requirements / "build-py312.lock").write_text(
            "demo==1\n", encoding="utf-8"
        )
        (requirements / "windows-py312-x64-hashed.lock").write_text(
            "demo==1 --hash=sha256:" + "0" * 64 + "\n", encoding="utf-8"
        )
        spec = root / "ArchMeshRubbing.spec"
        spec.write_text("# spec\n", encoding="utf-8")

        layout = build_native.artifact_layout(root / "dist", platform_name="darwin")
        assert layout.app_bundle is not None

        def fake_pyinstaller(command: list[str], **kwargs: object) -> None:
            assert command[:3] == [build_native.sys.executable, "-m", "PyInstaller"]
            assert command[-1] == str(spec)
            assert "--clean" not in command
            assert "--noconfirm" not in command
            layout.onedir.mkdir(parents=True)
            layout.executable.parent.mkdir(parents=True)
            layout.executable.write_bytes(b"launcher")

        def fake_self_test(executable: Path, report: Path) -> dict[str, object]:
            assert executable == layout.executable
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text('{"ok":true}', encoding="utf-8")
            return {"ok": True}

        with (
            patch.object(build_native, "validate_build_environment"),
            patch.object(build_native.subprocess, "run", side_effect=fake_pyinstaller),
            patch.object(
                build_native,
                "run_packaged_self_test",
                side_effect=fake_self_test,
            ),
        ):
            result = build_native.build_native(
                root=root,
                platform_name="darwin",
                channel="ci-smoke",
                commit="b" * 40,
                source_tree="dirty",
            )

        assert result.layout == layout
        assert result.self_test == {"ok": True}
        assert result.self_test_report is not None
        manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
        assert manifest["channel"] == "ci-smoke"
        assert manifest["commit"] == "b" * 40
        assert manifest["source_tree"] == "dirty"


def test_self_test_uses_report_file_and_rejects_reported_failure() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        executable = root / "ArchMeshRubbing"
        executable.write_bytes(b"launcher")
        report = root / "report.json"

        def write_failure(command: list[str], **kwargs: object) -> None:
            assert command == [
                str(executable),
                "--self-test-report",
                str(report),
            ]
            report.write_text('{"ok":false}', encoding="utf-8")

        with (
            patch.object(build_native.subprocess, "run", side_effect=write_failure),
            pytest.raises(build_native.NativeBuildError, match="reported failure"),
        ):
            build_native.run_packaged_self_test(executable, report)


def test_legacy_wrapper_is_non_destructive_compatibility_entrypoint() -> None:
    import build_and_shortcut

    with patch.object(build_and_shortcut, "build_native_main", return_value=7) as main:
        assert build_and_shortcut.main() == 7
    main.assert_called_once_with()
