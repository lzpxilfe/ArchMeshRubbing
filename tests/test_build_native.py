from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any
from unittest.mock import patch

import pytest

from tools import build_native


def _write_required_build_inputs(root: Path) -> None:
    requirements = root / "requirements"
    requirements.mkdir(parents=True)
    (requirements / "runtime-py312.lock").write_text(
        "demo==1\n",
        encoding="utf-8",
    )
    (requirements / "build-py312.lock").write_text(
        "demo==1\n",
        encoding="utf-8",
    )
    (requirements / "windows-py312-x64-hashed.lock").write_text(
        "demo==1 --hash=sha256:" + "0" * 64 + "\n",
        encoding="utf-8",
    )
    (root / "ArchMeshRubbing.spec").write_text("# spec\n", encoding="utf-8")
    (root / ".gitignore").write_text(
        "__pycache__/\n*.log\nbuild/\ndist/\n",
        encoding="utf-8",
    )


def _initialize_clean_build_checkout(root: Path) -> str:
    _write_required_build_inputs(root)
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=ArchMeshRubbing Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "test checkout",
        ],
        cwd=root,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_artifact_layout_is_windows_onedir_only() -> None:
    dist = Path("/tmp/native-dist")

    windows = build_native.artifact_layout(dist, platform_name="win32")
    assert windows.onedir == dist / "ArchMeshRubbing"
    assert windows.executable == windows.onedir / "ArchMeshRubbing.exe"
    assert windows.replace_targets == (windows.onedir,)

    for unsupported in ("darwin", "linux"):
        with pytest.raises(build_native.NativeBuildError, match="native AMD64 Windows"):
            build_native.artifact_layout(dist, platform_name=unsupported)


@pytest.mark.parametrize("platform_name", ["darwin", "linux"])
def test_non_windows_build_fails_before_writes_or_process(
    platform_name: str,
) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        with (
            patch.object(build_native.subprocess, "run") as run,
            pytest.raises(build_native.NativeBuildError, match="native AMD64 Windows"),
        ):
            build_native.build_native(
                root=root,
                platform_name=platform_name,
                machine_name="x86_64",
                pointer_bits=64,
                native_machine_name="AMD64",
            )

        run.assert_not_called()
        assert not (root / "build").exists()
        assert not (root / "dist").exists()


@pytest.mark.parametrize(
    ("machine_name", "pointer_bits", "message"),
    [("ARM64", 64, "Windows x64"), ("AMD64", 32, "64-bit CPython")],
)
def test_windows_build_rejects_unsupported_architecture(
    machine_name: str,
    pointer_bits: int,
    message: str,
) -> None:
    with pytest.raises(build_native.NativeBuildError, match=message):
        build_native.validate_windows_build_host(
            platform_name="win32",
            machine_name=machine_name,
            pointer_bits=pointer_bits,
            native_machine_name="AMD64",
        )


def test_windows_build_rejects_x64_emulation_on_arm64_host() -> None:
    with pytest.raises(build_native.NativeBuildError, match="native AMD64"):
        build_native.validate_windows_build_host(
            platform_name="win32",
            machine_name="AMD64",
            pointer_bits=64,
            native_machine_name="ARM64",
        )


def test_native_machine_probe_fails_before_subprocess_or_outputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "arm64-host"
    root.mkdir()
    with (
        patch.object(
            build_native,
            "_detect_windows_native_machine",
            return_value="ARM64",
        ) as detect,
        patch.object(build_native.subprocess, "run") as run,
        pytest.raises(build_native.NativeBuildError, match="native AMD64"),
    ):
        build_native.build_native(
            root=root,
            platform_name="win32",
            machine_name="AMD64",
            pointer_bits=64,
        )

    detect.assert_called_once_with()
    run.assert_not_called()
    assert not (root / "build").exists()
    assert not (root / "dist").exists()


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
            'pefile==2024.8.26; sys_platform == "win32"\n',
            encoding="utf-8",
        )
        assert build_native.exact_lock_pins(
            build_lock,
            platform_name="win32",
        ) == {
            "numpy": ("numpy", "2.4.3"),
            "pyinstaller": ("PyInstaller", "6.21.0"),
            "pefile": ("pefile", "2024.8.26"),
            "widget": ("widget", "1.0"),
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
        with (
            patch.object(
                build_native.platform,
                "python_implementation",
                return_value="PyPy",
            ),
            pytest.raises(build_native.NativeBuildError, match="implementation"),
        ):
            build_native.validate_build_environment(
                lock,
                python_version=(3, 12),
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
            python_version=(3, 12),
            installed_versions={"pyinstaller": "6.21.0"},
        )
        assert pins["pyinstaller"] == ("PyInstaller", "6.21.0")


def test_pypy_build_fails_before_subprocess_manifest_or_outputs(tmp_path: Path) -> None:
    root = tmp_path / "pypy-build"
    root.mkdir()
    _write_required_build_inputs(root)

    with (
        patch.object(
            build_native.platform, "python_implementation", return_value="PyPy"
        ),
        patch.object(build_native.subprocess, "run") as run,
        patch.object(build_native, "_write_or_reuse_manifest") as write_manifest,
        pytest.raises(build_native.NativeBuildError, match="implementation"),
    ):
        build_native.build_native(
            root=root,
            platform_name="win32",
            machine_name="AMD64",
            pointer_bits=64,
            native_machine_name="AMD64",
            commit="a" * 40,
        )

    run.assert_not_called()
    write_manifest.assert_not_called()
    assert not (root / "build").exists()
    assert not (root / "dist").exists()


def test_cli_has_no_python_version_bypass() -> None:
    with pytest.raises(SystemExit):
        build_native._parser().parse_args(["--allow-python-version-mismatch"])


def test_pyinstaller_spec_requires_windows_x64_python_312() -> None:
    spec = (Path(__file__).resolve().parents[1] / "ArchMeshRubbing.spec").read_text(
        encoding="utf-8"
    )
    assert 'sys.platform != "win32"' in spec
    assert 'platform.machine().casefold() not in {"amd64", "x86_64"}' in spec
    assert 'struct.calcsize("P") * 8 != 64' in spec
    assert "IsWow64Process2" in spec
    assert "_native_windows_machine() != 0x8664" in spec
    assert 'platform.python_implementation() != "CPython"' in spec
    assert "sys.version_info[:2] != (3, 12)" in spec

    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "package-smoke.yml"
    ).read_text(encoding="utf-8")
    assert 'PYTHONDONTWRITEBYTECODE: "1"' in workflow
    assert workflow.count('- ".gitignore"') == 2


def test_commit_mismatch_fails_before_manifest_or_build_outputs(tmp_path: Path) -> None:
    root = tmp_path / "mismatched-checkout"
    root.mkdir()
    head = _initialize_clean_build_checkout(root)
    requested = "0" * 40
    assert requested != head
    subprocess_commands: list[list[str]] = []
    real_run = build_native.subprocess.run

    def record_subprocess(
        command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[Any]:
        subprocess_commands.append(command)
        return real_run(command, **kwargs)

    with (
        patch.object(build_native, "validate_build_environment"),
        patch.object(
            build_native.subprocess,
            "run",
            side_effect=record_subprocess,
        ),
        patch.object(build_native, "_write_or_reuse_manifest") as write_manifest,
        pytest.raises(build_native.NativeBuildError, match="does not match"),
    ):
        build_native.build_native(
            root=root,
            platform_name="win32",
            machine_name="AMD64",
            pointer_bits=64,
            native_machine_name="AMD64",
            commit=requested,
        )

    write_manifest.assert_not_called()
    assert subprocess_commands
    assert all(command[0] == "git" for command in subprocess_commands)
    assert not (root / "build").exists()
    assert not (root / "dist").exists()


def test_dirty_checkout_fails_before_manifest_or_build_outputs(tmp_path: Path) -> None:
    root = tmp_path / "dirty-checkout"
    root.mkdir()
    head = _initialize_clean_build_checkout(root)
    (root / "ArchMeshRubbing.spec").write_text(
        "# dirty spec\n",
        encoding="utf-8",
    )
    subprocess_commands: list[list[str]] = []
    real_run = build_native.subprocess.run

    def record_subprocess(
        command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[Any]:
        subprocess_commands.append(command)
        return real_run(command, **kwargs)

    with (
        patch.object(build_native, "validate_build_environment"),
        patch.object(
            build_native.subprocess,
            "run",
            side_effect=record_subprocess,
        ),
        patch.object(build_native, "_write_or_reuse_manifest") as write_manifest,
        pytest.raises(build_native.NativeBuildError, match="clean Git worktree"),
    ):
        build_native.build_native(
            root=root,
            platform_name="win32",
            machine_name="AMD64",
            pointer_bits=64,
            native_machine_name="AMD64",
            commit=head,
        )

    write_manifest.assert_not_called()
    assert subprocess_commands
    assert all(command[0] == "git" for command in subprocess_commands)
    assert not (root / "build").exists()
    assert not (root / "dist").exists()


def test_git_repository_override_fails_before_git_manifest_or_outputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    head = _initialize_clean_build_checkout(root)
    other = tmp_path / "other"
    other.mkdir()
    _initialize_clean_build_checkout(other)

    with (
        patch.object(build_native, "validate_build_environment"),
        patch.dict(
            build_native.os.environ,
            {"GIT_DIR": str(other / ".git")},
        ),
        patch.object(build_native.subprocess, "run") as run,
        patch.object(build_native, "_write_or_reuse_manifest") as write_manifest,
        pytest.raises(build_native.NativeBuildError, match="Git repository/config"),
    ):
        build_native.build_native(
            root=root,
            platform_name="win32",
            machine_name="AMD64",
            pointer_bits=64,
            native_machine_name="AMD64",
            commit=head,
        )

    run.assert_not_called()
    write_manifest.assert_not_called()
    assert not (root / "build").exists()
    assert not (root / "dist").exists()


@pytest.mark.parametrize(
    "variable",
    [
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_OBJECT_DIRECTORY",
        "GIT_WORK_TREE",
    ],
)
def test_git_config_and_object_overrides_are_rejected(variable: str) -> None:
    with (
        patch.dict(build_native.os.environ, {variable: "redirect"}),
        pytest.raises(build_native.NativeBuildError, match=variable),
    ):
        build_native._git_environment()


def test_git_plumbing_disables_replace_objects_and_optional_locks() -> None:
    environment = build_native._git_environment()
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"


def test_nested_git_directory_fails_before_manifest_or_build_outputs(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent-checkout"
    root = parent / "nested-build-root"
    root.mkdir(parents=True)
    _write_required_build_inputs(root)
    subprocess.run(["git", "init", "--quiet"], cwd=parent, check=True)
    subprocess.run(["git", "add", "."], cwd=parent, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=ArchMeshRubbing Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "nested checkout",
        ],
        cwd=parent,
        check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commands: list[list[str]] = []
    real_run = build_native.subprocess.run

    def record_subprocess(
        command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[Any]:
        commands.append(command)
        return real_run(command, **kwargs)

    with (
        patch.object(build_native, "validate_build_environment"),
        patch.object(
            build_native.subprocess,
            "run",
            side_effect=record_subprocess,
        ),
        patch.object(build_native, "_write_or_reuse_manifest") as write_manifest,
        pytest.raises(build_native.NativeBuildError, match="Git top-level"),
    ):
        build_native.build_native(
            root=root,
            platform_name="win32",
            machine_name="AMD64",
            pointer_bits=64,
            native_machine_name="AMD64",
            commit=head,
        )

    write_manifest.assert_not_called()
    assert commands and all(command[0] == "git" for command in commands)
    assert not (root / "build").exists()
    assert not (root / "dist").exists()


@pytest.mark.parametrize(
    "index_flag",
    ["--assume-unchanged", "--skip-worktree"],
)
def test_hidden_tracked_change_fails_before_manifest_or_build_outputs(
    tmp_path: Path,
    index_flag: str,
) -> None:
    root = tmp_path / index_flag.removeprefix("--")
    root.mkdir()
    head = _initialize_clean_build_checkout(root)
    subprocess.run(
        ["git", "update-index", index_flag, "ArchMeshRubbing.spec"],
        cwd=root,
        check=True,
    )
    (root / "ArchMeshRubbing.spec").write_text(
        "# hidden live build input\n",
        encoding="utf-8",
    )
    assert (
        subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )
    commands: list[list[str]] = []
    real_run = build_native.subprocess.run

    def record_subprocess(
        command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[Any]:
        commands.append(command)
        return real_run(command, **kwargs)

    with (
        patch.object(build_native, "validate_build_environment"),
        patch.object(
            build_native.subprocess,
            "run",
            side_effect=record_subprocess,
        ),
        patch.object(build_native, "_write_or_reuse_manifest") as write_manifest,
        pytest.raises(
            build_native.NativeBuildError, match="content does not match HEAD"
        ),
    ):
        build_native.build_native(
            root=root,
            platform_name="win32",
            machine_name="AMD64",
            pointer_bits=64,
            native_machine_name="AMD64",
            commit=head,
        )

    write_manifest.assert_not_called()
    assert commands and all(command[0] == "git" for command in commands)
    assert not (root / "build").exists()
    assert not (root / "dist").exists()


def test_untracked_path_fails_before_manifest_or_build_outputs(tmp_path: Path) -> None:
    root = tmp_path / "untracked"
    root.mkdir()
    head = _initialize_clean_build_checkout(root)
    (root / "unexpected.txt").write_text("not committed\n", encoding="utf-8")

    with (
        patch.object(build_native, "validate_build_environment"),
        patch.object(build_native, "_write_or_reuse_manifest") as write_manifest,
        pytest.raises(build_native.NativeBuildError, match="untracked paths"),
    ):
        build_native.build_native(
            root=root,
            platform_name="win32",
            machine_name="AMD64",
            pointer_bits=64,
            native_machine_name="AMD64",
            commit=head,
        )

    write_manifest.assert_not_called()
    assert not (root / "build").exists()
    assert not (root / "dist").exists()


def test_index_executable_mode_change_is_not_clean(tmp_path: Path) -> None:
    root = tmp_path / "mode-change"
    root.mkdir()
    head = _initialize_clean_build_checkout(root)
    subprocess.run(
        ["git", "update-index", "--chmod=+x", "ArchMeshRubbing.spec"],
        cwd=root,
        check=True,
    )

    with pytest.raises(build_native.NativeBuildError, match="index path/blob/mode"):
        build_native.resolve_clean_source_checkout(root, head)


def test_ignored_frozen_input_fails_before_manifest_or_build_outputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ignored-input"
    root.mkdir()
    head = _initialize_clean_build_checkout(root)
    resources = root / "resources"
    resources.mkdir()
    (resources / "injected.log").write_text("ignored payload\n", encoding="utf-8")

    with (
        patch.object(build_native, "validate_build_environment"),
        patch.object(build_native, "_write_or_reuse_manifest") as write_manifest,
        pytest.raises(build_native.NativeBuildError, match="ignored files"),
    ):
        build_native.build_native(
            root=root,
            platform_name="win32",
            machine_name="AMD64",
            pointer_bits=64,
            native_machine_name="AMD64",
            commit=head,
        )

    write_manifest.assert_not_called()
    assert not (root / "build").exists()
    assert not (root / "dist").exists()


def test_ignored_source_bytecode_is_never_a_clean_build_input(tmp_path: Path) -> None:
    root = tmp_path / "ignored-bytecode"
    root.mkdir()
    head = _initialize_clean_build_checkout(root)
    cache = root / "src" / "core" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "injected.cpython-312.pyc").write_bytes(b"untrusted bytecode")

    with pytest.raises(build_native.NativeBuildError, match="ignored files"):
        build_native.resolve_clean_source_checkout(root, head)


def test_filtered_crlf_checkout_hashes_to_head_blob(tmp_path: Path) -> None:
    root = tmp_path / "crlf-checkout"
    root.mkdir()
    _write_required_build_inputs(root)
    (root / ".gitattributes").write_text(
        "*.txt text eol=crlf\n",
        encoding="utf-8",
    )
    sample = root / "sample.txt"
    sample.write_bytes(b"canonical line\n")
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=ArchMeshRubbing Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "CRLF checkout",
        ],
        cwd=root,
        check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    sample.unlink()
    subprocess.run(["git", "checkout", "--", "sample.txt"], cwd=root, check=True)
    assert sample.read_bytes() == b"canonical line\r\n"

    assert build_native.resolve_clean_source_checkout(root, head) == head


def test_opaque_git_content_filter_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "opaque-filter"
    root.mkdir()
    head = _initialize_clean_build_checkout(root)
    info = root / ".git" / "info"
    info.mkdir(exist_ok=True)
    (info / "attributes").write_text(
        "ArchMeshRubbing.spec filter=opaque\n",
        encoding="utf-8",
    )

    with pytest.raises(build_native.NativeBuildError, match="content transformation"):
        build_native.resolve_clean_source_checkout(root, head)


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
            patch.object(
                build_native,
                "resolve_clean_source_checkout",
                return_value="a" * 40,
            ),
            patch.object(build_native.subprocess, "run") as run,
            pytest.raises(build_native.NativeBuildError, match="refusing to overwrite"),
        ):
            build_native.build_native(
                root=root,
                platform_name="win32",
                machine_name="AMD64",
                pointer_bits=64,
                native_machine_name="AMD64",
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
        (requirements / "build-py312.lock").write_text("demo==1\n", encoding="utf-8")
        (requirements / "windows-py312-x64-hashed.lock").write_text(
            "demo==1 --hash=sha256:" + "0" * 64 + "\n", encoding="utf-8"
        )
        spec = root / "ArchMeshRubbing.spec"
        spec.write_text("# spec\n", encoding="utf-8")

        layout = build_native.artifact_layout(root / "dist", platform_name="win32")

        def fake_pyinstaller(command: list[str], **kwargs: Any) -> None:
            assert command[:3] == [build_native.sys.executable, "-m", "PyInstaller"]
            assert command[-1] == str(spec)
            assert "--clean" not in command
            assert "--noconfirm" not in command
            assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
            layout.onedir.mkdir(parents=True)
            layout.executable.write_bytes(b"launcher")

        def fake_self_test(executable: Path, report: Path) -> dict[str, object]:
            assert executable == layout.executable
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text('{"ok":true}', encoding="utf-8")
            return {"ok": True}

        def fake_source(
            _repository: Path,
            archive: Path,
            sidecar: Path,
            *,
            commit: str,
        ) -> None:
            assert commit == "b" * 40
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"source archive")
            sidecar.write_bytes(b"{}")

        def fake_evidence(
            _payload: Path,
            output: Path,
            *,
            created_at: str,
        ) -> None:
            assert created_at == "2026-07-01T00:00:00Z"
            output.mkdir()

        with (
            patch.object(build_native, "validate_build_environment"),
            patch.object(
                build_native,
                "resolve_clean_source_checkout",
                return_value="b" * 40,
            ),
            patch.object(build_native.subprocess, "run", side_effect=fake_pyinstaller),
            patch.object(
                build_native,
                "resolve_commit_timestamp",
                return_value="2026-07-01T00:00:00Z",
            ),
            patch.object(
                build_native,
                "run_packaged_self_test",
                side_effect=fake_self_test,
            ),
            patch(
                "src.source_archive.build_source_archive",
                side_effect=fake_source,
            ),
            patch(
                "src.release_evidence.generate_release_evidence",
                side_effect=fake_evidence,
            ),
        ):
            result = build_native.build_native(
                root=root,
                platform_name="win32",
                machine_name="AMD64",
                pointer_bits=64,
                native_machine_name="AMD64",
                channel="ci-smoke",
                commit="b" * 40,
            )

        assert result.layout == layout
        assert result.self_test == {"ok": True}
        assert result.self_test_report is not None
        manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
        assert manifest["channel"] == "ci-smoke"
        assert manifest["commit"] == "b" * 40
        assert manifest["source_tree"] == "clean"


def test_post_build_checkout_change_blocks_source_and_evidence() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        _write_required_build_inputs(root)
        layout = build_native.artifact_layout(root / "dist", platform_name="win32")
        commit = "c" * 40

        def fake_pyinstaller(command: list[str], **_kwargs: object) -> None:
            assert command[-1] == str(root / "ArchMeshRubbing.spec")
            layout.onedir.mkdir(parents=True)
            layout.executable.write_bytes(b"launcher")

        checkout = build_native.NativeBuildError(
            "native builds require a clean Git worktree after PyInstaller"
        )
        with (
            patch.object(build_native, "validate_build_environment"),
            patch.object(
                build_native,
                "resolve_clean_source_checkout",
                side_effect=[commit, checkout],
            ) as resolve_checkout,
            patch.object(build_native.subprocess, "run", side_effect=fake_pyinstaller),
            patch("src.source_archive.build_source_archive") as build_source,
            patch("src.release_evidence.generate_release_evidence") as build_evidence,
            pytest.raises(build_native.NativeBuildError, match="after PyInstaller"),
        ):
            build_native.build_native(
                root=root,
                platform_name="win32",
                machine_name="AMD64",
                pointer_bits=64,
                native_machine_name="AMD64",
                channel="ci-smoke",
                commit=commit,
                skip_self_test=True,
            )

        assert resolve_checkout.call_count == 2
        build_source.assert_not_called()
        build_evidence.assert_not_called()
        assert (root / "build" / "generated" / "build_info.json").is_file()
        assert layout.executable.is_file()
        assert not (layout.onedir / "source").exists()
        assert not (layout.onedir / "release-evidence").exists()


def test_post_build_hidden_tracked_change_is_caught_by_live_blob_recheck(
    tmp_path: Path,
) -> None:
    root = tmp_path / "post-build-hidden-change"
    root.mkdir()
    commit = _initialize_clean_build_checkout(root)
    subprocess.run(
        [
            "git",
            "update-index",
            "--assume-unchanged",
            "ArchMeshRubbing.spec",
        ],
        cwd=root,
        check=True,
    )
    layout = build_native.artifact_layout(root / "dist", platform_name="win32")
    real_run = build_native.subprocess.run

    def mutate_during_pyinstaller(
        command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[Any] | None:
        if command[:3] == [build_native.sys.executable, "-m", "PyInstaller"]:
            (root / "ArchMeshRubbing.spec").write_text(
                "# changed while PyInstaller was running\n",
                encoding="utf-8",
            )
            layout.onedir.mkdir(parents=True)
            layout.executable.write_bytes(b"launcher")
            return None
        return real_run(command, **kwargs)

    with (
        patch.object(build_native, "validate_build_environment"),
        patch.object(
            build_native.subprocess,
            "run",
            side_effect=mutate_during_pyinstaller,
        ),
        patch("src.source_archive.build_source_archive") as build_source,
        patch("src.release_evidence.generate_release_evidence") as build_evidence,
        pytest.raises(
            build_native.NativeBuildError, match="content does not match HEAD"
        ),
    ):
        build_native.build_native(
            root=root,
            platform_name="win32",
            machine_name="AMD64",
            pointer_bits=64,
            native_machine_name="AMD64",
            channel="ci-smoke",
            commit=commit,
            skip_self_test=True,
        )

    build_source.assert_not_called()
    build_evidence.assert_not_called()
    assert (root / "build" / "generated" / "build_info.json").is_file()
    assert layout.executable.is_file()
    assert not (layout.onedir / "source").exists()
    assert not (layout.onedir / "release-evidence").exists()


def test_windows_build_bundles_exact_source_before_release_evidence() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        requirements = root / "requirements"
        requirements.mkdir()
        for name in (
            "runtime-py312.lock",
            "build-py312.lock",
            "windows-py312-x64-hashed.lock",
        ):
            (requirements / name).write_text("demo==1\n", encoding="utf-8")
        spec = root / "ArchMeshRubbing.spec"
        spec.write_text("# spec\n", encoding="utf-8")
        layout = build_native.artifact_layout(root / "dist", platform_name="win32")
        commit = "c" * 40
        events: list[str] = []

        def fake_pyinstaller(command: list[str], **_kwargs: object) -> None:
            assert command[-1] == str(spec)
            layout.onedir.mkdir(parents=True)
            layout.executable.write_bytes(b"launcher")

        def fake_source(
            repository: Path,
            archive: Path,
            sidecar: Path,
            *,
            commit: str,
        ) -> None:
            assert repository == root
            assert commit == "c" * 40
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"source archive")
            sidecar.write_bytes(b"{}")
            events.append("source")

        def fake_evidence(
            payload: Path,
            output: Path,
            *,
            created_at: str,
        ) -> None:
            assert payload == layout.onedir
            assert created_at == "2026-07-01T00:00:00Z"
            assert (payload / "source" / "ArchMeshRubbing-source.zip").is_file()
            output.mkdir()
            events.append("evidence")

        with (
            patch.object(build_native, "validate_build_environment"),
            patch.object(
                build_native,
                "resolve_clean_source_checkout",
                return_value=commit,
            ),
            patch.object(build_native.subprocess, "run", side_effect=fake_pyinstaller),
            patch.object(
                build_native,
                "resolve_commit_timestamp",
                return_value="2026-07-01T00:00:00Z",
            ),
            patch(
                "src.source_archive.build_source_archive",
                side_effect=fake_source,
            ),
            patch(
                "src.release_evidence.generate_release_evidence",
                side_effect=fake_evidence,
            ),
        ):
            result = build_native.build_native(
                root=root,
                platform_name="win32",
                machine_name="AMD64",
                pointer_bits=64,
                native_machine_name="AMD64",
                channel="ci-smoke",
                commit=commit,
                skip_self_test=True,
            )

        assert events == ["source", "evidence"]
        assert result.source_archive == (
            layout.onedir / "source" / "ArchMeshRubbing-source.zip"
        )


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
