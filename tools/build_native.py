"""Build and verify a local, unsigned ArchMeshRubbing native package.

This tool deliberately stops at a local PyInstaller artifact.  It does not
sign, notarize, publish, upload, install, or create desktop shortcuts.  Public
distribution needs a separate, explicit release process and license review.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_NAME = "ArchMeshRubbing"
SPEC_PATH = ROOT / "ArchMeshRubbing.spec"
RUNTIME_LOCK_PATH = ROOT / "requirements" / "runtime-py312.lock"
BUILD_LOCK_PATH = ROOT / "requirements" / "build-py312.lock"
MANIFEST_PATH = ROOT / "build" / "generated" / "build_info.json"
DIST_PATH = ROOT / "dist"
WORK_PATH = ROOT / "build"
SUPPORTED_BUILD_PYTHON = (3, 12)

_PIN_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)"
    r"==(?P<version>[^;\s]+)"
    r"(?:\s*;\s*(?P<marker>.+))?$"
)
_INCLUDE_RE = re.compile(r"^-r\s+(?P<path>\S+)$")
_SYS_PLATFORM_MARKER_RE = re.compile(
    r"^sys_platform\s*(?P<operator>==|!=)\s*"
    r"(?P<quote>['\"])(?P<value>[^'\"]+)(?P=quote)$"
)
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class NativeBuildError(RuntimeError):
    """A safe, actionable native-build failure."""


@dataclass(frozen=True, slots=True)
class ArtifactLayout:
    """Expected PyInstaller output paths for one host platform."""

    dist_dir: Path
    onedir: Path
    executable: Path
    app_bundle: Path | None

    @property
    def replace_targets(self) -> tuple[Path, ...]:
        values = [self.onedir]
        if self.app_bundle is not None:
            values.append(self.app_bundle)
        return tuple(values)


@dataclass(frozen=True, slots=True)
class NativeBuildResult:
    """Verified outputs returned to callers and rendered by the CLI."""

    layout: ArtifactLayout
    manifest: Path
    source_archive: Path | None
    release_evidence: Path | None
    self_test_report: Path | None
    self_test: Mapping[str, object] | None
    command: tuple[str, ...]


def artifact_layout(
    dist_dir: Path = DIST_PATH,
    *,
    platform_name: str = sys.platform,
) -> ArtifactLayout:
    """Return the onedir and launchable executable paths for a platform."""

    dist_dir = Path(dist_dir)
    onedir = dist_dir / APP_NAME
    if platform_name == "darwin":
        app_bundle = dist_dir / f"{APP_NAME}.app"
        executable = app_bundle / "Contents" / "MacOS" / APP_NAME
    elif platform_name == "win32":
        app_bundle = None
        executable = onedir / f"{APP_NAME}.exe"
    else:
        app_bundle = None
        executable = onedir / APP_NAME
    return ArtifactLayout(
        dist_dir=dist_dir,
        onedir=onedir,
        executable=executable,
        app_bundle=app_bundle,
    )


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _marker_applies(marker: str | None, *, platform_name: str) -> bool:
    if marker is None:
        return True
    match = _SYS_PLATFORM_MARKER_RE.fullmatch(marker.strip())
    if match is None:
        raise NativeBuildError(
            "unsupported dependency marker in native-build lock: " + marker
        )
    matches = platform_name == match.group("value")
    return matches if match.group("operator") == "==" else not matches


def exact_lock_pins(
    lock_path: Path,
    *,
    platform_name: str = sys.platform,
) -> dict[str, tuple[str, str]]:
    """Read recursive ``-r`` files and return applicable exact pins.

    The native build lock intentionally uses only exact ``==`` pins and simple
    ``sys_platform`` markers.  Rejecting anything else prevents a seemingly
    locked build from silently resolving a new version.
    """

    pins: dict[str, tuple[str, str]] = {}
    visiting: set[Path] = set()

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visiting:
            raise NativeBuildError(f"dependency lock include cycle: {resolved}")
        try:
            text = resolved.read_text(encoding="utf-8", errors="strict")
        except OSError as exc:
            raise NativeBuildError(
                f"dependency lock is missing or unreadable: {resolved}"
            ) from exc
        visiting.add(resolved)
        try:
            for line_number, raw_line in enumerate(text.splitlines(), start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                include = _INCLUDE_RE.fullmatch(line)
                if include is not None:
                    visit(resolved.parent / include.group("path"))
                    continue
                match = _PIN_RE.fullmatch(line)
                if match is None:
                    raise NativeBuildError(
                        f"{resolved}:{line_number} is not an exact dependency pin"
                    )
                if not _marker_applies(
                    match.group("marker"),
                    platform_name=platform_name,
                ):
                    continue
                display_name = match.group("name")
                version = match.group("version")
                key = _canonical_distribution_name(display_name)
                previous = pins.get(key)
                if previous is not None and previous != (display_name, version):
                    raise NativeBuildError(
                        f"dependency lock repeats conflicting pin: {display_name}"
                    )
                pins[key] = (display_name, version)
        finally:
            visiting.remove(resolved)

    visit(Path(lock_path))
    if not pins:
        raise NativeBuildError("dependency lock contains no applicable exact pins")
    return pins


def validate_build_environment(
    build_lock: Path = BUILD_LOCK_PATH,
    *,
    platform_name: str = sys.platform,
    python_version: tuple[int, int] | None = None,
    installed_versions: Mapping[str, str | None] | None = None,
    allow_python_version_mismatch: bool = False,
) -> dict[str, tuple[str, str]]:
    """Require Python 3.12 and every applicable version in the build lock."""

    observed_python = python_version or (sys.version_info.major, sys.version_info.minor)
    if (
        observed_python != SUPPORTED_BUILD_PYTHON
        and not allow_python_version_mismatch
    ):
        raise NativeBuildError(
            "native builds require Python 3.12; create a clean environment and run "
            f"`python3.12 -m pip install -r {build_lock}`"
        )

    pins = exact_lock_pins(build_lock, platform_name=platform_name)
    problems: list[str] = []
    for key, (display_name, expected) in sorted(pins.items()):
        if installed_versions is None:
            try:
                observed = importlib.metadata.version(display_name)
            except importlib.metadata.PackageNotFoundError:
                observed = None
        else:
            observed = installed_versions.get(key)
            if observed is None:
                observed = installed_versions.get(display_name)
        if observed is None:
            problems.append(f"{display_name} is not installed (expected {expected})")
        elif observed != expected:
            problems.append(
                f"{display_name}=={observed} is installed (expected {expected})"
            )
    if problems:
        detail = "\n  - ".join(problems)
        raise NativeBuildError(
            "native-build environment does not match the exact lock:\n"
            f"  - {detail}\n"
            f"Install it with `{sys.executable} -m pip install -r {build_lock}`."
        )
    return pins


def resolve_commit(root: Path = ROOT, supplied: str | None = None) -> str:
    """Validate an explicit commit or read immutable identity from Git."""

    if supplied is None:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise NativeBuildError(
                "Git commit could not be detected; pass --commit with a full hash"
            ) from exc
        supplied = completed.stdout.strip()
    if _COMMIT_RE.fullmatch(supplied) is None:
        raise NativeBuildError("commit must be a lowercase 40- or 64-character Git hash")
    return supplied


def resolve_commit_timestamp(root: Path, commit: str) -> str:
    """Read the immutable commit timestamp used for reproducible SPDX output."""

    try:
        completed = subprocess.run(
            ["git", "show", "-s", "--format=%cI", commit],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NativeBuildError("source commit timestamp could not be read") from exc
    timestamp = completed.stdout.strip()
    if not timestamp:
        raise NativeBuildError("source commit timestamp is empty")
    return timestamp


def _existing_generated_outputs(
    layout: ArtifactLayout,
    *,
    work_dir: Path,
) -> tuple[Path, ...]:
    candidates = (*layout.replace_targets, Path(work_dir) / APP_NAME)
    return tuple(path for path in candidates if path.exists())


def _write_or_reuse_manifest(
    *,
    channel: str,
    commit: str,
    runtime_lock: Path,
    wheel_lock: Path,
    manifest_path: Path,
    replace_existing: bool,
    source_tree: str,
) -> None:
    from tools.generate_build_info import build_manifest, write_manifest

    try:
        manifest = build_manifest(
            channel=channel,
            commit=commit,
            lock_path=runtime_lock,
            wheel_lock_path=wheel_lock,
            source_tree=source_tree,
        )
    except (OSError, ValueError) as exc:
        raise NativeBuildError(f"build manifest could not be generated: {exc}") from exc
    if manifest_path.exists():
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            current = None
        if current == manifest:
            return
        if not replace_existing:
            raise NativeBuildError(
                f"generated build manifest already exists with other values: "
                f"{manifest_path}; pass --replace-existing to replace generated outputs"
            )
    write_manifest(manifest_path, manifest)


def locate_native_artifact(
    dist_dir: Path = DIST_PATH,
    *,
    platform_name: str = sys.platform,
) -> ArtifactLayout:
    """Locate the platform's launchable onedir or ``.app`` artifact."""

    layout = artifact_layout(dist_dir, platform_name=platform_name)
    missing: list[Path] = []
    if not layout.onedir.is_dir():
        missing.append(layout.onedir)
    if layout.app_bundle is not None and not layout.app_bundle.is_dir():
        missing.append(layout.app_bundle)
    if not layout.executable.is_file():
        missing.append(layout.executable)
    if missing:
        raise NativeBuildError(
            "PyInstaller completed without the expected native artifact: "
            + ", ".join(str(path) for path in missing)
        )
    return layout


def _next_report_path(
    work_dir: Path,
    *,
    channel: str,
    commit: str,
    platform_name: str,
) -> Path:
    report_dir = Path(work_dir) / "reports"
    stem = f"native-self-test-{channel}-{commit[:12]}-{platform_name}"
    candidate = report_dir / f"{stem}.json"
    suffix = 2
    while candidate.exists():
        candidate = report_dir / f"{stem}-{suffix}.json"
        suffix += 1
    return candidate


def run_packaged_self_test(executable: Path, report_path: Path) -> dict[str, object]:
    """Run the windowed frozen binary through its file-based self-test CLI."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        subprocess.run(
            [str(executable), "--self-test-report", str(report_path)],
            check=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NativeBuildError(
            f"packaged self-test failed; expected report: {report_path}"
        ) from exc
    try:
        value = json.loads(report_path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBuildError("packaged self-test did not write valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or value.get("ok") is not True:
        raise NativeBuildError(f"packaged self-test reported failure: {report_path}")
    return value


def build_native(
    *,
    channel: str = "local-smoke",
    commit: str | None = None,
    replace_existing: bool = False,
    clean_cache: bool = False,
    skip_self_test: bool = False,
    allow_python_version_mismatch: bool = False,
    source_tree: str | None = None,
    root: Path = ROOT,
    platform_name: str = sys.platform,
) -> NativeBuildResult:
    """Create one local unsigned package and verify its frozen entrypoint."""

    root = Path(root).resolve()
    spec_path = root / "ArchMeshRubbing.spec"
    runtime_lock = root / "requirements" / "runtime-py312.lock"
    build_lock = root / "requirements" / "build-py312.lock"
    wheel_lock = root / "requirements" / "windows-py312-x64-hashed.lock"
    manifest_path = root / "build" / "generated" / "build_info.json"
    dist_dir = root / "dist"
    work_dir = root / "build"
    layout = artifact_layout(dist_dir, platform_name=platform_name)

    for required in (spec_path, runtime_lock, build_lock, wheel_lock):
        if not required.is_file():
            raise NativeBuildError(f"required native-build input is missing: {required}")
    validate_build_environment(
        build_lock,
        platform_name=platform_name,
        allow_python_version_mismatch=allow_python_version_mismatch,
    )
    resolved_commit = resolve_commit(root, commit)

    existing = _existing_generated_outputs(layout, work_dir=work_dir)
    if existing and not replace_existing:
        raise NativeBuildError(
            "native-build outputs already exist; refusing to overwrite:\n  - "
            + "\n  - ".join(str(path) for path in existing)
            + "\nPass --replace-existing only after reviewing those generated outputs."
        )
    if source_tree is None:
        from tools.generate_build_info import detect_source_tree

        source_tree = detect_source_tree(root)
    _write_or_reuse_manifest(
        channel=channel,
        commit=resolved_commit,
        runtime_lock=runtime_lock,
        wheel_lock=wheel_lock,
        manifest_path=manifest_path,
        replace_existing=replace_existing,
        source_tree=source_tree,
    )

    command = [sys.executable, "-m", "PyInstaller"]
    if replace_existing:
        command.append("--noconfirm")
    if clean_cache:
        command.append("--clean")
    command.append(str(spec_path))
    try:
        subprocess.run(command, cwd=root, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NativeBuildError("PyInstaller native build failed") from exc

    built_layout = locate_native_artifact(
        dist_dir,
        platform_name=platform_name,
    )
    evidence_path: Path | None = None
    source_archive_path: Path | None = None
    if platform_name == "win32":
        from src.source_archive import (
            SOURCE_ARCHIVE_DIRECTORY,
            SOURCE_ARCHIVE_FILENAME,
            SOURCE_ARCHIVE_SIDECAR_FILENAME,
            SourceArchiveError,
            build_source_archive,
        )
        from src.release_evidence import (
            EVIDENCE_DIRECTORY_NAME,
            ReleaseEvidenceError,
            generate_release_evidence,
        )

        source_directory = built_layout.onedir / SOURCE_ARCHIVE_DIRECTORY
        source_archive_path = source_directory / SOURCE_ARCHIVE_FILENAME
        try:
            build_source_archive(
                root,
                source_archive_path,
                source_directory / SOURCE_ARCHIVE_SIDECAR_FILENAME,
                commit=resolved_commit,
            )
        except SourceArchiveError as exc:
            raise NativeBuildError(
                f"corresponding-source generation failed: {exc}"
            ) from exc
        evidence_path = built_layout.onedir / EVIDENCE_DIRECTORY_NAME
        try:
            generate_release_evidence(
                built_layout.onedir,
                evidence_path,
                created_at=resolve_commit_timestamp(root, resolved_commit),
            )
        except ReleaseEvidenceError as exc:
            raise NativeBuildError(f"release evidence generation failed: {exc}") from exc
    report_path: Path | None = None
    self_test: dict[str, object] | None = None
    if not skip_self_test:
        report_path = _next_report_path(
            work_dir,
            channel=channel,
            commit=resolved_commit,
            platform_name=platform_name,
        )
        self_test = run_packaged_self_test(built_layout.executable, report_path)
    return NativeBuildResult(
        layout=built_layout,
        manifest=manifest_path,
        source_archive=source_archive_path,
        release_evidence=evidence_path,
        self_test_report=report_path,
        self_test=self_test,
        command=tuple(command),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and verify a local unsigned ArchMeshRubbing PyInstaller package. "
            "This command never publishes or signs artifacts."
        )
    )
    parser.add_argument("--channel", default="local-smoke")
    parser.add_argument("--commit", help="full lowercase Git hash; defaults to HEAD")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="allow PyInstaller to replace existing generated build/dist outputs",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="ask PyInstaller to clear its cache (never enabled by default)",
    )
    parser.add_argument(
        "--skip-self-test",
        action="store_true",
        help="build without executing the frozen offline self-test",
    )
    parser.add_argument(
        "--allow-python-version-mismatch",
        action="store_true",
        help=(
            "permit a local diagnostic build outside Python 3.12; the exact "
            "dependency lock is still required and the result is not release-eligible"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_native(
            channel=args.channel,
            commit=args.commit,
            replace_existing=args.replace_existing,
            clean_cache=args.clean_cache,
            skip_self_test=args.skip_self_test,
            allow_python_version_mismatch=args.allow_python_version_mismatch,
        )
    except NativeBuildError as exc:
        print(f"Native build stopped: {exc}", file=sys.stderr)
        return 2

    print(f"Local unsigned artifact: {result.layout.executable}")
    if result.layout.app_bundle is not None:
        print(f"macOS app bundle: {result.layout.app_bundle}")
    print(f"Embedded build manifest: {result.manifest}")
    if result.source_archive is not None:
        print(f"Verified corresponding source: {result.source_archive}")
    if result.release_evidence is not None:
        print(f"Verified release evidence: {result.release_evidence}")
    if result.self_test_report is not None:
        print(f"Frozen self-test passed: {result.self_test_report}")
    else:
        print("Frozen self-test skipped by explicit request.")
    print("No artifact was signed, installed, uploaded, or published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
