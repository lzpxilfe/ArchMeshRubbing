"""Build and verify a local, unsigned ArchMeshRubbing native package.

This tool deliberately stops at a local PyInstaller artifact.  It does not
sign, notarize, publish, upload, install, or create desktop shortcuts.  Public
distribution needs a separate, explicit release process and license review.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import struct
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
SUPPORTED_WINDOWS_MACHINES = frozenset({"amd64", "x86_64"})
WINDOWS_NATIVE_AMD64 = "AMD64"

_IMAGE_FILE_MACHINE_NAMES = {
    0x014C: "x86",
    0x0200: "IA64",
    0x8664: WINDOWS_NATIVE_AMD64,
    0xAA64: "ARM64",
}
_GIT_REPOSITORY_OVERRIDE_KEYS = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_ATTR_SOURCE",
        "GIT_COMMON_DIR",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_QUARANTINE_PATH",
        "GIT_WORK_TREE",
    }
)
_IGNORED_BUILD_INPUT_ROOTS = (
    "resources",
    "schemas",
    "src",
    "third_party_licenses",
)

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
    """Expected Windows x64 PyInstaller onedir output paths."""

    dist_dir: Path
    onedir: Path
    executable: Path

    @property
    def replace_targets(self) -> tuple[Path, ...]:
        return (self.onedir,)


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


@dataclass(frozen=True, slots=True)
class _TrackedEntry:
    mode: str
    object_id: str
    path: str


def artifact_layout(
    dist_dir: Path = DIST_PATH,
    *,
    platform_name: str = sys.platform,
) -> ArtifactLayout:
    """Return the only supported Windows x64 onedir layout."""

    if platform_name != "win32":
        raise NativeBuildError(
            "native packages are supported only on native AMD64 Windows build hosts"
        )
    dist_dir = Path(dist_dir)
    onedir = dist_dir / APP_NAME
    return ArtifactLayout(
        dist_dir=dist_dir,
        onedir=onedir,
        executable=onedir / f"{APP_NAME}.exe",
    )


def _detect_windows_native_machine() -> str:
    """Return the native Windows host architecture via ``IsWow64Process2``."""

    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        raise NativeBuildError(
            "native packages require IsWow64Process2 to verify a native AMD64 host"
        )
    try:
        kernel32 = win_dll("kernel32", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = ctypes.c_void_p
        is_wow64_process2 = kernel32.IsWow64Process2
        is_wow64_process2.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ushort),
            ctypes.POINTER(ctypes.c_ushort),
        ]
        is_wow64_process2.restype = ctypes.c_int
        process_machine = ctypes.c_ushort()
        native_machine = ctypes.c_ushort()
        if not is_wow64_process2(
            get_current_process(),
            ctypes.byref(process_machine),
            ctypes.byref(native_machine),
        ):
            raise NativeBuildError(
                "IsWow64Process2 could not verify the native Windows architecture"
            )
    except NativeBuildError:
        raise
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise NativeBuildError(
            "native packages require IsWow64Process2 to verify a native AMD64 host"
        ) from exc
    return _IMAGE_FILE_MACHINE_NAMES.get(
        int(native_machine.value),
        f"unknown-0x{int(native_machine.value):04x}",
    )


def validate_windows_build_host(
    *,
    platform_name: str = sys.platform,
    machine_name: str | None = None,
    pointer_bits: int | None = None,
    native_machine_name: str | None = None,
) -> None:
    """Fail before writes unless this is native AMD64 Windows with x64 Python."""

    observed_machine = platform.machine() if machine_name is None else machine_name
    observed_bits = struct.calcsize("P") * 8 if pointer_bits is None else pointer_bits
    if platform_name != "win32":
        raise NativeBuildError(
            "native packages are supported only on native AMD64 Windows build hosts"
        )
    if observed_machine.casefold() not in SUPPORTED_WINDOWS_MACHINES:
        raise NativeBuildError(
            "native packages require Windows x64 (AMD64/x86_64); "
            f"observed architecture: {observed_machine or 'unknown'}"
        )
    if observed_bits != 64:
        raise NativeBuildError(
            f"native packages require 64-bit CPython; observed {observed_bits}-bit"
        )
    observed_native = (
        _detect_windows_native_machine()
        if native_machine_name is None
        else native_machine_name
    )
    if observed_native != WINDOWS_NATIVE_AMD64:
        raise NativeBuildError(
            "native packages require a native AMD64 Windows host; "
            f"observed native architecture: {observed_native or 'unknown'}"
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
) -> dict[str, tuple[str, str]]:
    """Require CPython 3.12 and every applicable version in the build lock."""

    observed_python = python_version or (sys.version_info.major, sys.version_info.minor)
    observed_implementation = platform.python_implementation()
    if observed_implementation != "CPython":
        raise NativeBuildError(
            "native builds require CPython 3.12; observed Python implementation: "
            f"{observed_implementation or 'unknown'}"
        )
    if observed_python != SUPPORTED_BUILD_PYTHON:
        raise NativeBuildError(
            "native builds require CPython 3.12; create a clean environment and run "
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


def _git_environment() -> dict[str, str]:
    """Return a deterministic Git environment or reject repository redirection."""

    configured_count = os.environ.get("GIT_CONFIG_COUNT", "")
    offenders = {
        key for key in _GIT_REPOSITORY_OVERRIDE_KEYS if os.environ.get(key, "")
    }
    if configured_count not in {"", "0"}:
        offenders.add("GIT_CONFIG_COUNT")
    offenders.update(
        key
        for key, value in os.environ.items()
        if value
        and (key.startswith("GIT_CONFIG_KEY_") or key.startswith("GIT_CONFIG_VALUE_"))
    )
    if offenders:
        raise NativeBuildError(
            "native builds reject Git repository/config override environment: "
            + ", ".join(sorted(offenders))
        )
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["LC_ALL"] = "C"
    return environment


def _run_git(
    root: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=False,
            input=input_bytes,
            env=_git_environment(),
        )
    except NativeBuildError:
        raise
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NativeBuildError(
            "Git could not verify the native-build checkout: " + " ".join(arguments)
        ) from exc
    return bytes(completed.stdout)


def _decode_git_path(raw: bytes) -> str:
    try:
        value = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise NativeBuildError("Git checkout contains a non-UTF-8 path") from exc
    logical = PurePosixPath(value)
    if (
        not value
        or logical.is_absolute()
        or logical.as_posix() != value
        or any(part in {"", ".", ".."} for part in logical.parts)
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise NativeBuildError(f"Git checkout contains an unsafe path: {value!r}")
    return value


def _head_tracked_entries(root: Path, commit: str) -> dict[str, _TrackedEntry]:
    raw = _run_git(root, "ls-tree", "-rz", "--full-tree", "-r", commit)
    entries: dict[str, _TrackedEntry] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            raw_mode, raw_kind, raw_object = metadata.split(b" ", 2)
            mode = raw_mode.decode("ascii", errors="strict")
            kind = raw_kind.decode("ascii", errors="strict")
            object_id = raw_object.decode("ascii", errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise NativeBuildError("Git HEAD tree record is malformed") from exc
        path = _decode_git_path(raw_path)
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise NativeBuildError(
                "native builds require regular tracked files only; "
                f"HEAD entry {path!r} has mode/type {mode} {kind}"
            )
        if _COMMIT_RE.fullmatch(object_id) is None:
            raise NativeBuildError(f"Git HEAD blob ID is malformed for {path!r}")
        if path in entries:
            raise NativeBuildError(f"Git HEAD repeats tracked path: {path!r}")
        entries[path] = _TrackedEntry(mode=mode, object_id=object_id, path=path)
    if not entries:
        raise NativeBuildError("Git HEAD contains no tracked files")
    return entries


def _index_tracked_entries(root: Path) -> dict[str, _TrackedEntry]:
    raw = _run_git(root, "ls-files", "--stage", "-z")
    entries: dict[str, _TrackedEntry] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            raw_mode, raw_object, raw_stage = metadata.split(b" ", 2)
            mode = raw_mode.decode("ascii", errors="strict")
            object_id = raw_object.decode("ascii", errors="strict")
            stage = raw_stage.decode("ascii", errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise NativeBuildError("Git index record is malformed") from exc
        path = _decode_git_path(raw_path)
        if stage != "0" or path in entries:
            raise NativeBuildError(
                f"native builds reject unresolved or repeated Git index path: {path!r}"
            )
        entries[path] = _TrackedEntry(mode=mode, object_id=object_id, path=path)
    return entries


def _path_fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    try:
        status = path.lstat()
    except OSError as exc:
        raise NativeBuildError(f"tracked worktree path is missing: {path}") from exc
    if not stat.S_ISREG(status.st_mode):
        raise NativeBuildError(f"tracked worktree path is not a regular file: {path}")
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_mode),
    )


def _unexpected_untracked_paths(root: Path) -> tuple[str, ...]:
    raw = _run_git(root, "ls-files", "--others", "--exclude-standard", "-z")
    return tuple(_decode_git_path(value) for value in raw.split(b"\0") if value)


def _unexpected_ignored_build_inputs(root: Path) -> tuple[str, ...]:
    raw = _run_git(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
        "--",
        *_IGNORED_BUILD_INPUT_ROOTS,
    )
    unexpected: list[str] = []
    for value in raw.split(b"\0"):
        if not value:
            continue
        path = _decode_git_path(value)
        unexpected.append(path)
    return tuple(unexpected)


def _require_safe_worktree_attributes(
    root: Path,
    entries: tuple[_TrackedEntry, ...],
) -> None:
    """Allow Git's EOL normalization but reject opaque content transforms."""

    attributes = ("filter", "ident", "working-tree-encoding")
    stdin_paths = b"\0".join(entry.path.encode("utf-8") for entry in entries) + b"\0"
    raw = _run_git(
        root,
        "check-attr",
        "-z",
        "--stdin",
        *attributes,
        input_bytes=stdin_paths,
    )
    fields = raw.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) != len(entries) * len(attributes) * 3:
        raise NativeBuildError("Git returned malformed worktree attributes")
    expected_paths = {entry.path for entry in entries}
    for offset in range(0, len(fields), 3):
        path = _decode_git_path(fields[offset])
        try:
            attribute = fields[offset + 1].decode("ascii", errors="strict")
            value = fields[offset + 2].decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise NativeBuildError(
                "Git returned malformed worktree attributes"
            ) from exc
        if path not in expected_paths or attribute not in attributes:
            raise NativeBuildError("Git returned attributes for an unexpected path")
        if value != "unspecified":
            raise NativeBuildError(
                "native builds reject opaque Git content transformation attribute "
                f"{attribute}={value!r} on {path!r}"
            )


def _require_exact_head_worktree(root: Path, commit: str) -> None:
    """Compare every live tracked path to HEAD without trusting index flags."""

    head_entries = _head_tracked_entries(root, commit)
    index_entries = _index_tracked_entries(root)
    if index_entries != head_entries:
        raise NativeBuildError(
            "native builds require a clean Git worktree; index path/blob/mode "
            "does not match HEAD"
        )

    untracked = _unexpected_untracked_paths(root)
    if untracked:
        raise NativeBuildError(
            "native builds require a clean Git worktree; untracked paths exist: "
            + ", ".join(repr(path) for path in untracked[:8])
        )
    ignored_inputs = _unexpected_ignored_build_inputs(root)
    if ignored_inputs:
        raise NativeBuildError(
            "native builds reject ignored files inside frozen build inputs: "
            + ", ".join(repr(path) for path in ignored_inputs[:8])
        )

    ordered = tuple(head_entries[path] for path in sorted(head_entries))
    _require_safe_worktree_attributes(root, ordered)
    before = {
        entry.path: _path_fingerprint(root.joinpath(*PurePosixPath(entry.path).parts))
        for entry in ordered
    }
    stdin_paths = "".join(f"{entry.path}\n" for entry in ordered).encode("utf-8")
    observed_hashes = _run_git(
        root,
        "hash-object",
        "--stdin-paths",
        input_bytes=stdin_paths,
    ).splitlines()
    if len(observed_hashes) != len(ordered):
        raise NativeBuildError("Git did not hash every tracked worktree path")
    after = {
        entry.path: _path_fingerprint(root.joinpath(*PurePosixPath(entry.path).parts))
        for entry in ordered
    }
    if after != before:
        raise NativeBuildError(
            "native builds require an unchanged worktree during tracked-file hashing"
        )

    for entry, raw_observed in zip(ordered, observed_hashes, strict=True):
        try:
            observed = raw_observed.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise NativeBuildError("Git returned a malformed worktree blob ID") from exc
        if observed != entry.object_id:
            raise NativeBuildError(
                "native builds require a clean Git worktree; tracked worktree "
                f"content does not match HEAD: {entry.path!r}"
            )
        if os.name != "nt":
            executable = bool(after[entry.path][4] & 0o111)
            if executable != (entry.mode == "100755"):
                raise NativeBuildError(
                    "native builds require tracked executable mode to match HEAD: "
                    f"{entry.path!r}"
                )


def resolve_commit(root: Path = ROOT, supplied: str | None = None) -> str:
    """Validate an explicit commit or read immutable identity from Git."""

    if supplied is None:
        try:
            supplied = (
                _run_git(root, "rev-parse", "--verify", "HEAD^{commit}")
                .decode("ascii", errors="strict")
                .strip()
            )
        except UnicodeDecodeError as exc:
            raise NativeBuildError("Git HEAD commit ID is malformed") from exc
    if _COMMIT_RE.fullmatch(supplied) is None:
        raise NativeBuildError(
            "commit must be a lowercase 40- or 64-character Git hash"
        )
    return supplied


def resolve_clean_source_checkout(
    root: Path = ROOT,
    supplied: str | None = None,
) -> str:
    """Bind the build to the live HEAD of one provably clean checkout."""

    try:
        canonical_root = Path(root).resolve(strict=True)
    except OSError as exc:
        raise NativeBuildError("native-build checkout root is missing") from exc
    raw_top = _run_git(canonical_root, "rev-parse", "--show-toplevel").rstrip(b"\r\n")
    try:
        top = Path(os.fsdecode(raw_top)).resolve(strict=True)
    except OSError as exc:
        raise NativeBuildError("Git returned an unreadable checkout root") from exc
    if top != canonical_root:
        raise NativeBuildError(
            "native-build root must be the canonical Git top-level directory: "
            f"root {canonical_root}, Git top-level {top}"
        )

    live_head = resolve_commit(canonical_root)
    requested = (
        live_head if supplied is None else resolve_commit(canonical_root, supplied)
    )
    if requested != live_head:
        raise NativeBuildError(
            "native build commit does not match the checked-out Git HEAD: "
            f"requested {requested}, HEAD {live_head}"
        )

    _require_exact_head_worktree(canonical_root, live_head)
    if resolve_commit(canonical_root) != live_head:
        raise NativeBuildError(
            "Git HEAD changed during native-build checkout verification"
        )
    return live_head


def resolve_commit_timestamp(root: Path, commit: str) -> str:
    """Read the immutable commit timestamp used for reproducible SPDX output."""

    try:
        timestamp = (
            _run_git(root, "show", "-s", "--format=%cI", commit)
            .decode("ascii", errors="strict")
            .strip()
        )
    except UnicodeDecodeError as exc:
        raise NativeBuildError("source commit timestamp is malformed") from exc
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
    """Locate the supported Windows x64 onedir artifact."""

    layout = artifact_layout(dist_dir, platform_name=platform_name)
    missing: list[Path] = []
    if not layout.onedir.is_dir():
        missing.append(layout.onedir)
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
        raise NativeBuildError(
            "packaged self-test did not write valid UTF-8 JSON"
        ) from exc
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
    root: Path = ROOT,
    platform_name: str = sys.platform,
    machine_name: str | None = None,
    pointer_bits: int | None = None,
    native_machine_name: str | None = None,
) -> NativeBuildResult:
    """Create one Windows x64 unsigned package and verify its entrypoint."""

    # Local source imports happen only after the exact checkout preflight.  Do
    # not create or later consume source-tree bytecode outside that trust gate.
    sys.dont_write_bytecode = True
    validate_windows_build_host(
        platform_name=platform_name,
        machine_name=machine_name,
        pointer_bits=pointer_bits,
        native_machine_name=native_machine_name,
    )
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
            raise NativeBuildError(
                f"required native-build input is missing: {required}"
            )
    validate_build_environment(
        build_lock,
        platform_name=platform_name,
    )
    resolved_commit = resolve_clean_source_checkout(root, commit)

    existing = _existing_generated_outputs(layout, work_dir=work_dir)
    if existing and not replace_existing:
        raise NativeBuildError(
            "native-build outputs already exist; refusing to overwrite:\n  - "
            + "\n  - ".join(str(path) for path in existing)
            + "\nPass --replace-existing only after reviewing those generated outputs."
        )
    _write_or_reuse_manifest(
        channel=channel,
        commit=resolved_commit,
        runtime_lock=runtime_lock,
        wheel_lock=wheel_lock,
        manifest_path=manifest_path,
        replace_existing=replace_existing,
        source_tree="clean",
    )

    command = [sys.executable, "-m", "PyInstaller"]
    if replace_existing:
        command.append("--noconfirm")
    if clean_cache:
        command.append("--clean")
    command.append(str(spec_path))
    build_environment = os.environ.copy()
    build_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        subprocess.run(command, cwd=root, check=True, env=build_environment)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NativeBuildError("PyInstaller native build failed") from exc

    built_layout = locate_native_artifact(
        dist_dir,
        platform_name=platform_name,
    )
    # Do not issue corresponding-source or release evidence if HEAD or tracked
    # source changed while PyInstaller was running. Generated build/dist paths
    # are ignored, so they do not make this clean-tree recheck fail.
    resolve_clean_source_checkout(root, resolved_commit)
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
        )
    except NativeBuildError as exc:
        print(f"Native build stopped: {exc}", file=sys.stderr)
        return 2

    print(f"Local unsigned artifact: {result.layout.executable}")
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
