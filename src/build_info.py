"""Build metadata, diagnostics, and an offline packaged-application self-test.

The module itself intentionally imports only the standard library.  Heavy
runtime modules are imported inside the self-test so ``--version`` and
``--diagnostics-json`` can still explain a damaged installation.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.metadata
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
from typing import Any, Callable

from src import __version__


APP_NAME = "ArchMeshRubbing"
APP_VERSION = __version__
DISTRIBUTION_NAME = "ArchMeshRubbing"
BUILD_INFO_SCHEMA_VERSION = "1.2.0"

# These values are deliberately immutable module constants.  A later native
# build step may replace them in the staged source tree before freezing; they
# must not be inferred from a possibly absent or dirty runtime Git checkout.
BUILD_CHANNEL = "source"
BUILD_COMMIT = "unknown"
BUILD_LOCK_SHA256 = "unknown"
RUNTIME_LOCK_PARTS = ("requirements", "runtime-py312.lock")
WINDOWS_WHEEL_LOCK_PARTS = (
    "requirements",
    "windows-py312-x64-hashed.lock",
)
FROZEN_BUILD_MANIFEST_PARTS = ("resources", "build_info.json")

_EXPECTED_DOCUMENT_SHA256 = (
    "860531781ee05e937ed2144f7e45ba870391b9fc2c07b3957442645be8953717"
)
_EXPECTED_VECTOR_SHA256 = (
    "0253922427a4deab3069bffb68fb91359b2afb753a07d60370d332c61e9ce491"
)
_EXPECTED_RUBBING_RAW_SHA256 = (
    "51e8e3057e7f4381071438308da8d8efb7df35b998764735822190bc21f0f8ed"
)
_EXPECTED_RUBBING_RASTER_SHA256 = (
    "6fdadfcca36c6655415f069aecf1b7b30c2f3378b9b44d89fd5dd0d8b96f1be7"
)

# Import module, installed distribution.  Keeping the two names explicit is
# important for Pillow and PyOpenGL, whose import names differ from their
# distribution names.
_REQUIRED_RUNTIMES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("trimesh", "trimesh"),
    ("PIL", "Pillow"),
    ("rfc8785", "rfc8785"),
    ("shapely", "shapely"),
    ("PyQt6", "PyQt6"),
    ("OpenGL", "PyOpenGL"),
)
_LOCK_ONLY_DISTRIBUTIONS = ("PyQt6-Qt6", "PyQt6-sip")
_PIN_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)==(?P<version>[^;\s]+)$"
)
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_CHANNEL_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,31}$")

_SELF_TEST_QT_APP: Any | None = None


@dataclass(frozen=True, slots=True)
class SelfTestCheck:
    """One serializable self-test result."""

    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def version_text() -> str:
    """Return the stable human-readable CLI version string."""

    return f"{APP_NAME} {APP_VERSION}"


def _resource_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        roots.append(Path(str(frozen_root)))
    roots.append(Path(__file__).resolve().parents[1])

    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return tuple(unique)


def resource_path(*parts: str) -> Path:
    """Resolve a read-only resource in source and PyInstaller layouts."""

    roots = _resource_roots()
    for root in roots:
        candidate = root.joinpath(*parts)
        if candidate.exists():
            return candidate
    return roots[0].joinpath(*parts)


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def runtime_lock() -> tuple[Path, dict[str, tuple[str, str]], str]:
    """Load the exact runtime resolution bundled with source/frozen builds."""

    path = resource_path(*RUNTIME_LOCK_PARTS)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError("runtime dependency lock is missing or unreadable") from exc
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError("runtime dependency lock is not UTF-8") from exc
    pins: dict[str, tuple[str, str]] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN_RE.fullmatch(line)
        if match is None:
            raise RuntimeError(
                f"runtime dependency lock line {line_number} is not an exact pin"
            )
        name = match.group("name")
        key = _canonical_distribution_name(name)
        if key in pins:
            raise RuntimeError(f"runtime dependency lock repeats {name}")
        pins[key] = (name, match.group("version"))
    required = {
        _canonical_distribution_name(distribution)
        for _module, distribution in _REQUIRED_RUNTIMES
    } | {
        _canonical_distribution_name(distribution)
        for distribution in _LOCK_ONLY_DISTRIBUTIONS
    }
    missing = sorted(required - set(pins))
    if missing:
        raise RuntimeError(
            "runtime dependency lock is missing required pins: " + ", ".join(missing)
        )
    return path, pins, hashlib.sha256(payload).hexdigest()


def windows_wheel_lock() -> tuple[Path, dict[str, tuple[str, str, str]], str]:
    """Load the exact, hash-checked Windows x64/Python 3.12 wheel set."""

    from src.release_evidence import ReleaseEvidenceError, parse_hashed_lock

    path = resource_path(*WINDOWS_WHEEL_LOCK_PARTS)
    try:
        pins, payload = parse_hashed_lock(path)
    except ReleaseEvidenceError as exc:
        raise RuntimeError(str(exc)) from exc
    _runtime_path, runtime_pins, _runtime_sha256 = runtime_lock()
    for key, (name, version) in runtime_pins.items():
        wheel_pin = pins.get(key)
        if wheel_pin is None or wheel_pin[1] != version:
            raise RuntimeError(
                f"Windows wheel lock does not match runtime pin {name}=={version}"
            )
    return path, pins, hashlib.sha256(payload).hexdigest()


def build_metadata() -> dict[str, object]:
    """Return validated immutable metadata embedded by the native build."""

    _lock_path, _pins, observed_lock_sha256 = runtime_lock()
    _wheel_path, _wheel_pins, observed_wheel_lock_sha256 = windows_wheel_lock()
    manifest_path = resource_path(*FROZEN_BUILD_MANIFEST_PARTS)
    if not manifest_path.is_file():
        return {
            "channel": BUILD_CHANNEL,
            "commit": BUILD_COMMIT,
            "dependency_lock_sha256": observed_lock_sha256,
            "manifest_present": False,
            "source_tree": "unknown",
            "windows_wheel_lock_sha256": observed_wheel_lock_sha256,
        }
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise RuntimeError("frozen build manifest is unreadable") from exc
    if not raw or len(raw) > 64 * 1024:
        raise RuntimeError("frozen build manifest size is invalid")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("frozen build manifest is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "channel",
        "commit",
        "dependency_lock_sha256",
        "schema_version",
        "source_tree",
        "version",
        "windows_wheel_lock_sha256",
    }:
        raise RuntimeError("frozen build manifest fields are invalid")
    channel = value["channel"]
    commit = value["commit"]
    lock_sha256 = value["dependency_lock_sha256"]
    wheel_lock_sha256 = value["windows_wheel_lock_sha256"]
    source_tree = value["source_tree"]
    if value["schema_version"] != BUILD_INFO_SCHEMA_VERSION:
        raise RuntimeError("frozen build manifest schema is unsupported")
    if value["version"] != APP_VERSION:
        raise RuntimeError("frozen build manifest version does not match the application")
    if not isinstance(channel, str) or _CHANNEL_RE.fullmatch(channel) is None:
        raise RuntimeError("frozen build channel is invalid")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeError("frozen build commit is invalid")
    if lock_sha256 != observed_lock_sha256:
        raise RuntimeError("frozen build dependency lock hash does not match its bytes")
    if wheel_lock_sha256 != observed_wheel_lock_sha256:
        raise RuntimeError("frozen build Windows wheel lock hash does not match its bytes")
    if not isinstance(source_tree, str) or source_tree not in {
        "clean",
        "dirty",
        "unknown",
    }:
        raise RuntimeError("frozen build source tree state is invalid")
    return {
        "channel": channel,
        "commit": commit,
        "dependency_lock_sha256": lock_sha256,
        "manifest_present": True,
        "source_tree": source_tree,
        "windows_wheel_lock_sha256": wheel_lock_sha256,
    }


def runtime_diagnostics() -> dict[str, dict[str, object]]:
    """Report dependency discoverability without requiring imports to work."""

    result: dict[str, dict[str, object]] = {}
    for module_name, distribution_name in _REQUIRED_RUNTIMES:
        try:
            importable = importlib.util.find_spec(module_name) is not None
        except (ImportError, AttributeError, ValueError):
            importable = False
        result[distribution_name] = {
            "import_name": module_name,
            "importable": importable,
            "version": _distribution_version(distribution_name),
        }
    return result


def collect_diagnostics() -> dict[str, object]:
    """Collect JSON-safe release diagnostics without writing to disk."""

    icon_path = resource_path("resources", "icons", "app_icon.png")
    metadata_error: str | None = None
    try:
        metadata = build_metadata()
    except RuntimeError as exc:
        metadata = {
            "channel": "invalid",
            "commit": "invalid",
            "dependency_lock_sha256": "invalid",
            "manifest_present": False,
            "source_tree": "invalid",
            "windows_wheel_lock_sha256": "invalid",
        }
        metadata_error = str(exc)
    build = {
        **metadata,
        "frozen": bool(getattr(sys, "frozen", False)),
    }
    if metadata_error is not None:
        build["metadata_error"] = metadata_error
    return {
        "schema_version": BUILD_INFO_SCHEMA_VERSION,
        "application": {
            "name": APP_NAME,
            "distribution": DISTRIBUTION_NAME,
            "version": APP_VERSION,
        },
        "build": build,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "machine": platform.machine() or "unknown",
            "system": platform.system() or "unknown",
        },
        "resources": {
            "app_icon_png": {
                "present": icon_path.is_file(),
            }
        },
        "runtime": runtime_diagnostics(),
    }


def diagnostics_json(payload: dict[str, object] | None = None) -> str:
    """Serialize diagnostics or a self-test report as deterministic JSON."""

    value = collect_diagnostics() if payload is None else payload
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_json_report(path: str | Path, payload: dict[str, object]) -> Path:
    """Write one machine report without overwriting an existing result."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (diagnostics_json(payload) + "\n").encode("utf-8")
    try:
        with destination.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise RuntimeError(f"report destination already exists: {destination}") from exc
    return destination


def _check_required_runtime() -> str:
    _lock_path, pins, observed_lock_sha256 = runtime_lock()
    if (
        BUILD_LOCK_SHA256 != "unknown"
        and BUILD_LOCK_SHA256 != observed_lock_sha256
    ):
        raise RuntimeError(
            "bundled runtime dependency lock does not match build metadata"
        )
    versions: list[str] = []
    for module_name, distribution_name in _REQUIRED_RUNTIMES:
        importlib.import_module(module_name)
        installed_version = _distribution_version(distribution_name)
        if installed_version is None:
            raise RuntimeError(
                f"installed metadata is missing for {distribution_name}"
            )
        expected = pins[_canonical_distribution_name(distribution_name)][1]
        if installed_version != expected:
            raise RuntimeError(
                f"runtime lock mismatch for {distribution_name}: "
                f"expected {expected}, found {installed_version}"
            )
        versions.append(f"{distribution_name}={installed_version}")
    for distribution_name in _LOCK_ONLY_DISTRIBUTIONS:
        installed_version = _distribution_version(distribution_name)
        expected = pins[_canonical_distribution_name(distribution_name)][1]
        if installed_version != expected:
            raise RuntimeError(
                f"runtime lock mismatch for {distribution_name}: "
                f"expected {expected}, found {installed_version or 'missing'}"
            )

    import shapely

    from src.core.artifact_outline_extractor import (
        REQUIRED_GEOS_VERSION,
        REQUIRED_SHAPELY_VERSION,
    )

    if (
        shapely.__version__ != REQUIRED_SHAPELY_VERSION
        or shapely.geos_version_string != REQUIRED_GEOS_VERSION
    ):
        raise RuntimeError(
            "authoritative Outline runtime mismatch: "
            f"expected Shapely {REQUIRED_SHAPELY_VERSION}/GEOS "
            f"{REQUIRED_GEOS_VERSION}, found Shapely {shapely.__version__}/GEOS "
            f"{shapely.geos_version_string}"
        )
    return f"lock={observed_lock_sha256}; " + ", ".join(versions)


def _check_build_identity() -> str:
    metadata = build_metadata()
    frozen = bool(getattr(sys, "frozen", False))
    if frozen and metadata["manifest_present"] is not True:
        raise RuntimeError("frozen application has no immutable build manifest")
    if frozen and metadata["channel"] == "source":
        raise RuntimeError("frozen application still claims the source build channel")
    return (
        f"channel={metadata['channel']}, commit={metadata['commit']}, "
        f"source_tree={metadata['source_tree']}, "
        f"lock={metadata['dependency_lock_sha256']}, "
        f"wheel_lock={metadata['windows_wheel_lock_sha256']}"
    )


def _check_resources() -> str:
    icon_path = resource_path("resources", "icons", "app_icon.png")
    if not icon_path.is_file():
        raise RuntimeError("resources/icons/app_icon.png is missing")
    with icon_path.open("rb") as stream:
        signature = stream.read(8)
    if signature != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("resources/icons/app_icon.png is not a valid PNG resource")
    lock_path, _pins, _sha256 = runtime_lock()
    wheel_lock_path, _wheel_pins, _wheel_sha256 = windows_wheel_lock()
    policy_path = resource_path("requirements", "runtime-license-policy.json")
    public_policy_path = resource_path(
        "requirements", "public-release-policy.json"
    )
    fallback_path = resource_path(
        "third_party_licenses", "PyOpenGL-3.1.10-LICENSE.txt"
    )
    if (
        not policy_path.is_file()
        or not public_policy_path.is_file()
        or not fallback_path.is_file()
    ):
        raise RuntimeError("runtime license policy or reviewed fallback is missing")
    from src.public_release_policy import (
        PublicReleasePolicyError,
        load_public_release_policy,
        verify_combined_work_license,
        verify_project_license,
        verify_runtime_license_observations,
    )

    try:
        public_policy, _public_policy_raw = load_public_release_policy(
            public_policy_path
        )
        verify_project_license(public_policy, resource_path("LICENSE"))
        if public_policy.combined_work_license is not None:
            combined_parts = tuple(
                Path(public_policy.combined_work_license.path).parts
            )
            combined_path = resource_path(*combined_parts)
            if not combined_path.is_file():
                raise PublicReleasePolicyError(
                    "combined-work license file is missing or linked"
                )
            verify_combined_work_license(
                public_policy,
                combined_path.parents[len(combined_parts) - 1],
            )
        observed_licenses: dict[str, tuple[str, str | None]] = {}
        for item in public_policy.runtime_license_observations:
            metadata = importlib.metadata.metadata(item.canonical_name)
            expression = metadata.get("License-Expression")
            observed_licenses[item.canonical_name] = (
                str(metadata.get("Version") or ""),
                str(expression).strip() if expression else None,
            )
        verify_runtime_license_observations(public_policy, observed_licenses)
    except (PublicReleasePolicyError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError(f"public release policy failed: {exc}") from exc
    required_schemas = (
        "artifact_document-1.0.0.schema.json",
        "geometry_metrics_receipt-1.0.0.schema.json",
        "surface_measurement_receipt-1.0.0.schema.json",
        "vector_payload-1.0.0.schema.json",
        "vector_export-1.0.0.schema.json",
        "vector_export-1.1.0.schema.json",
        "vector_export-1.2.0.schema.json",
        "vector_export-1.3.0.schema.json",
        "vector_export-1.4.0.schema.json",
        "vector_export-1.5.0.schema.json",
        "vector_export-1.6.0.schema.json",
        "rubbing_receipt-1.0.0.schema.json",
        "rubbing_export-1.0.0.schema.json",
        "rubbing_export-1.1.0.schema.json",
        "rubbing_export-1.2.0.schema.json",
        "rubbing_export-1.3.0.schema.json",
        "survey_export-1.0.0.schema.json",
        "tile_unwrap_receipt-1.0.0.schema.json",
        "tile_unwrap_export-1.0.0.schema.json",
        "tile_unwrap_receipt-1.1.0.schema.json",
        "tile_unwrap_export-1.1.0.schema.json",
        "tile_unwrap_export-1.2.0.schema.json",
        "tile_unwrap_export-1.3.0.schema.json",
        "tile_unwrap_export-1.4.0.schema.json",
        "tile_unwrap_export-1.5.0.schema.json",
        "offline_verification_report-1.0.0.schema.json",
        "source_bundle-1.0.0.schema.json",
        "source_bundle-2.0.0.schema.json",
        "source_manifest-1.0.0.schema.json",
        "mesh_import_recipe-1.0.0.schema.json",
        "mesh_import_recipe-2.0.0.schema.json",
        "mesh_admission_receipt-1.0.0.schema.json",
        "portable_archive_manifest-1.0.0.schema.json",
        "source_archive-1.0.0.schema.json",
        "build_provenance-1.0.0.schema.json",
        "field_pilot_review-1.0.0.schema.json",
        "field_pilot_report-1.0.0.schema.json",
        "field_pilot_report-1.1.0.schema.json",
        "field_pilot_verification-1.0.0.schema.json",
    )
    for name in required_schemas:
        if not resource_path("schemas", name).is_file():
            raise RuntimeError(f"packaged schema is missing: {name}")
    return (
        f"app icon, runtime lock, hashed wheel lock, fail-closed release policy, and "
        f"{len(required_schemas)} schemas present "
        f"({lock_path.name}, {wheel_lock_path.name}; "
        f"public-binary={public_policy.decision})"
    )


def _check_release_evidence() -> str:
    """Verify the generated evidence from the installed/frozen payload."""

    if not bool(getattr(sys, "frozen", False)):
        return "source build; release evidence is generated after freezing"
    if sys.platform != "win32":
        raise RuntimeError("frozen builds are supported only on Windows 10/11 x64")
    from src.release_evidence import verify_release_evidence

    payload_root = Path(sys.executable).resolve().parent
    return verify_release_evidence(payload_root).detail()


def _check_source_archive() -> str:
    """Verify bundled corresponding source against the frozen build commit."""

    if not bool(getattr(sys, "frozen", False)):
        return "source build; corresponding source is generated after freezing"
    if sys.platform != "win32":
        raise RuntimeError("frozen builds are supported only on Windows 10/11 x64")
    metadata = build_metadata()
    if metadata["source_tree"] != "clean":
        raise RuntimeError(
            "frozen build is not clean and cannot claim exact corresponding source"
        )
    from src.source_archive import (
        SOURCE_ARCHIVE_DIRECTORY,
        SOURCE_ARCHIVE_FILENAME,
        SOURCE_ARCHIVE_SIDECAR_FILENAME,
        verify_source_archive,
    )

    payload_root = Path(sys.executable).resolve().parent
    source_directory = payload_root / SOURCE_ARCHIVE_DIRECTORY
    if source_directory.is_symlink() or not source_directory.is_dir():
        raise RuntimeError("bundled corresponding-source directory is missing")
    try:
        observed = {path.name for path in source_directory.iterdir()}
    except OSError as exc:
        raise RuntimeError("bundled corresponding-source directory is unreadable") from exc
    expected = {SOURCE_ARCHIVE_FILENAME, SOURCE_ARCHIVE_SIDECAR_FILENAME}
    if observed != expected:
        raise RuntimeError("bundled corresponding-source file set is invalid")
    result = verify_source_archive(
        source_directory / SOURCE_ARCHIVE_FILENAME,
        source_directory / SOURCE_ARCHIVE_SIDECAR_FILENAME,
    )
    if result.source_commit != metadata["commit"]:
        raise RuntimeError(
            "bundled corresponding source does not match the frozen build commit"
        )
    return result.detail()


def _check_qt_offscreen() -> str:
    global _SELF_TEST_QT_APP

    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    created = app is None
    if app is None:
        # The command is intended to work on CI and systems without a display.
        # Passing the platform explicitly avoids relying on a user's shell.
        _SELF_TEST_QT_APP = QApplication(
            ["archmeshrubbing-self-test", "-platform", "offscreen"]
        )
        app = _SELF_TEST_QT_APP
    if not isinstance(app, QApplication):
        raise RuntimeError("an incompatible QCoreApplication already exists")
    app.processEvents()
    platform_name = str(QGuiApplication.platformName() or "unknown")
    if created and platform_name != "offscreen":
        raise RuntimeError(
            f"Qt requested the offscreen platform but initialized {platform_name!r}"
        )
    return f"Qt application constructed ({platform_name})"


def _check_gui_stack() -> str:
    """Import and construct the real GUI shell without showing a window."""

    from OpenGL import GL, GLU
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    from PyQt6.QtWidgets import QApplication, QMainWindow

    from app_interactive import MainWindow
    from src.gui.viewport_3d import Viewport3D

    app = QApplication.instance()
    if not isinstance(app, QApplication):
        raise RuntimeError("Qt application must exist before the GUI stack check")
    if not issubclass(Viewport3D, QOpenGLWidget):
        raise RuntimeError("Viewport3D is not backed by QOpenGLWidget")
    if not issubclass(MainWindow, QMainWindow):
        raise RuntimeError("MainWindow is not a QMainWindow")
    if not callable(getattr(GL, "glGetString", None)):
        raise RuntimeError("OpenGL.GL entry points are unavailable")
    if not callable(getattr(GLU, "gluPerspective", None)):
        raise RuntimeError("OpenGL.GLU entry points are unavailable")

    window = MainWindow()
    try:
        if not isinstance(window.viewport, Viewport3D):
            raise RuntimeError("MainWindow did not construct the native 3D viewport")
        panel = getattr(window, "section_panel", None)
        required_tile_controls = (
            "btn_native_survey_export",
            "combo_native_tile_target",
            "combo_native_tile_axis",
            "combo_native_tile_record_view",
            "spin_native_tile_sections",
            "btn_native_tile_unwrap",
            "combo_native_tile_unwrap_record",
            "btn_native_tile_unwrap_export",
            "label_native_tile_unwrap_preview",
            "label_native_tile_unwrap_info",
        )
        missing = [
            name
            for name in required_tile_controls
            if panel is None or not hasattr(panel, name)
        ]
        if missing:
            raise RuntimeError(
                "MainWindow is missing native tile unwrap controls: "
                + ", ".join(missing)
            )
        workflow_panel = getattr(window, "workflow_panel", None)
        if workflow_panel is None or not hasattr(
            workflow_panel,
            "btn_authoritative_measurements",
        ):
            raise RuntimeError(
                "MainWindow is missing the authoritative measurement shortcut"
            )
        app.processEvents()
    finally:
        # Do not call close(): the user-facing closeEvent asks for confirmation
        # and persists layout state.  Deferred destruction keeps this check
        # non-interactive and avoids mutating user preferences.
        window.deleteLater()
        app.processEvents()
    return (
        "MainWindow, atomic survey export, native tile unwrap panel, "
        "QOpenGLWidget, OpenGL.GL, "
        "and OpenGL.GLU constructed"
    )


def _check_mesh_parsers() -> str:
    """Exercise every advertised parser through the authoritative load gate."""

    import numpy as np
    import trimesh
    from trimesh.exchange.gltf import export_gltf

    from src.core.mesh_import_recipe import current_mesh_import_recipe
    from src.core.mesh_loader import MeshLoader

    source = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array(
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
            dtype=np.int64,
        ),
        process=False,
    )
    encoded: dict[str, bytes] = {}
    for file_type in ("obj", "ply", "stl", "off", "glb"):
        value = source.export(file_type=file_type)
        if isinstance(value, str):
            encoded[file_type] = value.encode("utf-8")
        elif isinstance(value, bytes):
            encoded[file_type] = value
        else:
            raise RuntimeError(
                f"unexpected {file_type} encoder result: {type(value).__name__}"
            )
    gltf_files = export_gltf(source, embed_buffers=True)
    gltf_bytes = gltf_files.get("model.gltf")
    if not isinstance(gltf_bytes, bytes):
        raise RuntimeError("embedded glTF encoder did not return model.gltf bytes")
    encoded["gltf"] = gltf_bytes

    parsed: list[str] = []
    for file_type in ("obj", "ply", "stl", "off", "gltf", "glb"):
        payload = encoded[file_type]
        loaded = MeshLoader(default_unit="mm").load_verified_stream(
            io.BytesIO(encoded[file_type]),
            unit="mm",
            source_format=file_type,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size_bytes=len(payload),
            original_name=f"self-test.{file_type}",
            import_recipe=current_mesh_import_recipe(file_type),
        )
        if loaded.source_import_recipe != current_mesh_import_recipe(file_type):
            raise RuntimeError(f"{file_type} parser lost its import receipt")
        if loaded.n_vertices < 4 or loaded.n_faces < 4:
            raise RuntimeError(f"{file_type} parser lost the fixture geometry")
        parsed.append(file_type)
    return "parsed=" + ",".join(parsed)


def _check_png_codec() -> str:
    """Prove that the frozen Pillow PNG codec can encode and decode pixels."""

    from PIL import Image

    expected = bytes((0, 64, 128, 255))
    source = Image.frombytes("L", (2, 2), expected)
    stream = io.BytesIO()
    source.save(stream, format="PNG")
    encoded = stream.getvalue()
    if not encoded.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Pillow did not produce a PNG byte stream")
    stream.seek(0)
    with Image.open(stream) as restored:
        restored.load()
        if restored.mode != "L" or restored.size != (2, 2):
            raise RuntimeError("Pillow PNG round-trip changed mode or dimensions")
        if restored.tobytes() != expected:
            raise RuntimeError("Pillow PNG round-trip changed pixel values")
    return f"PNG round-trip bytes={len(encoded)}"


def _check_artifact_document() -> str:
    from src.core.artifact_document import ArtifactDocument

    document = ArtifactDocument.empty(
        document_id="artifact:self-test",
        software_version="self-test/1",
    )
    canonical = document.canonical_json_bytes()
    restored = ArtifactDocument.from_json_bytes(canonical)
    if restored.canonical_json_bytes() != canonical:
        raise RuntimeError("ArtifactDocument canonical round-trip changed bytes")
    if document.canonical_sha256 != _EXPECTED_DOCUMENT_SHA256:
        raise RuntimeError(
            "ArtifactDocument canonical golden mismatch: "
            f"{document.canonical_sha256}"
        )
    return f"sha256={document.canonical_sha256}"


def _check_artifact_vector() -> str:
    import numpy as np

    from src.core.artifact_vector_extractor import extract_cutline_geometry
    from src.core.artifact_vector_record import PlanarFrame

    vertices = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    frame = PlanarFrame(
        origin_world_mm=(0.0, 0.0, 0.0),
        u_axis_world=(1.0, 0.0, 0.0),
        v_axis_world=(0.0, 1.0, 0.0),
        normal_world=(0.0, 0.0, 1.0),
    )
    payload = extract_cutline_geometry(vertices, faces, frame).payload
    if payload.sha256 != _EXPECTED_VECTOR_SHA256:
        raise RuntimeError(f"canonical cutline golden mismatch: {payload.sha256}")
    if len(payload.paths) != 1 or len(payload.paths[0].points_mm) != 4:
        raise RuntimeError("canonical cutline topology is not the expected square")
    return f"sha256={payload.sha256}"


def _check_artifact_rubbing() -> str:
    import numpy as np

    from src.core.artifact_rubbing_extractor import (
        extract_digital_rubbing,
        rubbing_recipe,
    )

    vertices = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    recipe = rubbing_recipe(
        "top",
        pixels_per_mm=10,
        margin_um=0,
        reference_radius_um=500,
        depth_quantization_um=10,
        black_point_um=100,
        ink_strength_percent=100,
        relief_polarity="bidirectional",
    )
    raster, qc = extract_digital_rubbing(vertices, faces, recipe)
    if raster.raw_pixel_sha256 != _EXPECTED_RUBBING_RAW_SHA256:
        raise RuntimeError(
            "canonical rubbing pixel golden mismatch: "
            f"{raster.raw_pixel_sha256}"
        )
    if raster.raster_sha256 != _EXPECTED_RUBBING_RASTER_SHA256:
        raise RuntimeError(
            f"canonical rubbing receipt golden mismatch: {raster.raster_sha256}"
        )
    if tuple(raster.pixels.shape) != (20, 20, 2):
        raise RuntimeError(f"canonical rubbing shape mismatch: {raster.pixels.shape}")
    if qc.get("covered_pixel_count") != 400:
        raise RuntimeError("canonical rubbing coverage mismatch")
    return f"sha256={raster.raster_sha256}"


def _check_artifact_embedded_project_roundtrip() -> str:
    """Prove that an AMR carries enough source data for an offline reopen."""

    import numpy as np

    from src.core.artifact_session import ArtifactSession
    from src.core.mesh_loader import MeshLoader
    from src.core.project_file import (
        load_artifact_session_project,
        save_artifact_session_project,
    )

    ply_bytes = (
        b"ply\n"
        b"format ascii 1.0\n"
        b"comment frozen offline self-test fixture\n"
        b"element vertex 5\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"element face 4\n"
        b"property list uchar int vertex_indices\n"
        b"end_header\n"
        b"1.25 -2.5 0.75\n"
        b"4.5 -1.25 1.5\n"
        b"3.75 2.0 2.25\n"
        b"-0.5 1.5 -1.0\n"
        b"2.0 0.25 4.0\n"
        b"3 0 1 4\n"
        b"3 1 2 4\n"
        b"3 2 3 4\n"
        b"3 3 0 4\n"
    )

    with tempfile.TemporaryDirectory(prefix="archmeshrubbing-self-test-") as temporary:
        directory = Path(temporary)
        source_path = directory / "self-test-primary.ply"
        project_path = directory / "self-test-project.amr"
        source_path.write_bytes(ply_bytes)

        source_mesh = MeshLoader(default_unit="mm").load(source_path, unit="cm")
        session = ArtifactSession.create_from_source(
            source_mesh,
            resolved_source_path=str(source_path),
            unit="cm",
            axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
            handedness="right",
            software_version="self-test/1",
            operator="frozen-self-test",
            created_at="2026-01-01T00:00:00Z",
            document_id="artifact:frozen-offline-self-test",
            metadata_revision_id="metadata:self-test-cm",
            align_revision_id="align:self-test-identity",
        )
        session = session.commit_preview(
            translation_mm=(13.25, -7.5, 4.75),
            rotation_deg=(17.0, -23.0, 61.0),
            scale=1.0,
            pivot_mm=(12.5, -5.0, 7.5),
            operator="frozen-self-test",
            created_at="2026-01-01T00:00:01Z",
            revision_id="align:self-test-explicit",
        )
        before_projection = session.materialize()
        save_artifact_session_project(project_path, session)

        source_path.unlink()
        if source_path.exists():
            raise RuntimeError("external source deletion did not take effect")
        restored = load_artifact_session_project(project_path)
        after_projection = restored.materialize()

        if (
            restored.document.canonical_json_bytes()
            != session.document.canonical_json_bytes()
        ):
            raise RuntimeError("embedded reopen changed the ArtifactDocument")

        source_identity = session.source_mesh.source_identity
        restored_identity = restored.source_mesh.source_identity
        if source_identity is None or restored_identity is None:
            raise RuntimeError("embedded reopen lost primary source identity")
        if (
            not source_identity.content_matches(restored_identity)
            or source_identity.identity_scope != restored_identity.identity_scope
            or source_identity.original_name != restored_identity.original_name
            or source_identity.format != restored_identity.format
        ):
            raise RuntimeError("embedded reopen changed primary source identity")
        if not np.array_equal(session.source_mesh.vertices, restored.source_mesh.vertices):
            raise RuntimeError("embedded reopen changed source vertices")
        if not np.array_equal(session.source_mesh.faces, restored.source_mesh.faces):
            raise RuntimeError("embedded reopen changed source faces")
        if (
            restored.source_mesh.source_import_recipe
            != session.source_mesh.source_import_recipe
        ):
            raise RuntimeError("embedded reopen changed the parser/runtime receipt")

        if restored.verified_geometry != session.verified_geometry:
            raise RuntimeError("embedded reopen changed verified geometry identity")
        geometry_id = session.verified_geometry.geometry_revision_id
        if (
            restored.document.geometry_revision_index[geometry_id]
            != session.document.geometry_revision_index[geometry_id]
        ):
            raise RuntimeError("embedded reopen changed the geometry revision")

        align_id = session.document.active_align_revision_id
        if align_id != "align:self-test-explicit":
            raise RuntimeError("self-test did not create an explicit Align revision")
        if restored.document.active_align_revision_id != align_id:
            raise RuntimeError("embedded reopen changed the active Align revision")
        if (
            restored.document.align_revision_index[align_id]
            != session.document.align_revision_index[align_id]
        ):
            raise RuntimeError("embedded reopen changed the active Align data")

        before_vertices = np.asarray(before_projection.mesh.vertices, dtype=np.float64)
        after_vertices = np.asarray(after_projection.mesh.vertices, dtype=np.float64)
        if not np.array_equal(after_vertices, before_vertices):
            raise RuntimeError("embedded reopen changed materialized vertices")

        source_short = source_identity.sha256[:12]
        geometry_short = session.verified_geometry.geometry_sha256[:12]
        return (
            f"source={source_short}, geometry={geometry_short}, "
            f"align={align_id}, vertices={before_vertices.shape[0]}"
        )


def _check_artifact_complete_workflow() -> str:
    """Run the complete public workflow and offline export reproduction."""

    from src.application.artifact_workflow_self_test import (
        run_artifact_workflow_self_test,
    )

    return run_artifact_workflow_self_test().detail()


def _run_check(name: str, check: Callable[[], str]) -> SelfTestCheck:
    try:
        return SelfTestCheck(name=name, ok=True, detail=str(check()))
    except Exception as exc:
        return SelfTestCheck(
            name=name,
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def run_self_test() -> dict[str, object]:
    """Run deterministic, offline health checks.

    Scientific checks use tiny in-memory fixtures.  The embedded-project check
    writes only inside an automatically removed temporary directory.
    """

    checks = (
        _run_check("build_identity", _check_build_identity),
        _run_check("required_runtime", _check_required_runtime),
        _run_check("resources", _check_resources),
        _run_check("release_evidence", _check_release_evidence),
        _run_check("source_archive", _check_source_archive),
        _run_check("qt_offscreen", _check_qt_offscreen),
        _run_check("gui_stack", _check_gui_stack),
        _run_check("mesh_parsers", _check_mesh_parsers),
        _run_check("png_codec", _check_png_codec),
        _run_check("artifact_document_canonical", _check_artifact_document),
        _run_check("artifact_vector_canonical", _check_artifact_vector),
        _run_check("artifact_rubbing_canonical", _check_artifact_rubbing),
        _run_check(
            "artifact_embedded_project_roundtrip",
            _check_artifact_embedded_project_roundtrip,
        ),
        _run_check(
            "artifact_complete_workflow_offline",
            _check_artifact_complete_workflow,
        ),
    )
    return {
        "schema_version": BUILD_INFO_SCHEMA_VERSION,
        "application": {
            "name": APP_NAME,
            "version": APP_VERSION,
        },
        "diagnostics": collect_diagnostics(),
        "ok": all(check.ok for check in checks),
        "checks": [check.to_dict() for check in checks],
    }
