"""Native AMD64 Windows PyInstaller onedir build for package smoke tests."""

from pathlib import Path
import ctypes
import platform
import struct
import sys

from PyInstaller.utils.hooks import collect_submodules, copy_metadata


ROOT = Path(SPECPATH).resolve()
ICON = ROOT / "resources" / "icons" / "app_icon.png"
BUILD_INFO = ROOT / "build" / "generated" / "build_info.json"
RUNTIME_LOCK = ROOT / "requirements" / "runtime-py312.lock"
WINDOWS_WHEEL_LOCK = ROOT / "requirements" / "windows-py312-x64-hashed.lock"
RUNTIME_LICENSE_POLICY = ROOT / "requirements" / "runtime-license-policy.json"
PUBLIC_RELEASE_POLICY = ROOT / "requirements" / "public-release-policy.json"


def _native_windows_machine():
    """Fail closed when an x64 process is emulated on an ARM64 host."""

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
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
            return None
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return int(native_machine.value)


if (
    sys.platform != "win32"
    or platform.machine().casefold() not in {"amd64", "x86_64"}
    or struct.calcsize("P") * 8 != 64
    or _native_windows_machine() != 0x8664
    or platform.python_implementation() != "CPython"
    or sys.version_info[:2] != (3, 12)
):
    raise SystemExit(
        "ArchMeshRubbing.spec requires a native AMD64 Windows build host with "
        "64-bit CPython 3.12"
    )

if not BUILD_INFO.is_file():
    raise SystemExit(
        "build/generated/build_info.json is missing; run "
        "tools/generate_build_info.py before PyInstaller"
    )

datas = [
    (str(ROOT / "resources"), "resources"),
    (str(BUILD_INFO), "resources"),
    (str(ROOT / "schemas"), "schemas"),
    (str(RUNTIME_LOCK), "requirements"),
    (str(WINDOWS_WHEEL_LOCK), "requirements"),
    (str(RUNTIME_LICENSE_POLICY), "requirements"),
    (str(PUBLIC_RELEASE_POLICY), "requirements"),
    (str(ROOT / "third_party_licenses"), "third_party_licenses"),
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "README.md"), "."),
]

for distribution in (
    "numpy",
    "scipy",
    "trimesh",
    "Pillow",
    "rfc8785",
    "shapely",
    "PyQt6",
    "PyQt6-Qt6",
    "PyQt6-sip",
    "PyOpenGL",
):
    datas += copy_metadata(distribution)

hiddenimports = sorted(
    set(
        [
            "app_interactive",
            "OpenGL.GL",
            "OpenGL.GLU",
            "OpenGL.platform",
            "PyQt6.QtOpenGLWidgets",
            "src.public_release_policy",
            "src.release_evidence",
            "src.source_archive",
        ]
        + collect_submodules("src.application")
        + collect_submodules("src.core")
        + collect_submodules("src.gui")
    )
)

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "cv2",
        "OpenGL_accelerate",
        "PyQt5",
        "PySide2",
        "PySide6",
        # Build and test tools can be present in a developer virtualenv, but
        # are not runtime dependencies and must not leak into the application.
        "_pytest",
        "jsonschema",
        "jsonschema_specifications",
        "pytest",
        "pygments",
        "referencing",
        "rpds",
        "setuptools",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ArchMeshRubbing",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ArchMeshRubbing",
)
