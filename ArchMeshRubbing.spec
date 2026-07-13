"""Cross-platform PyInstaller onedir build for native package smoke tests."""

from pathlib import Path
import re
import sys

from PyInstaller.utils.hooks import collect_submodules, copy_metadata


ROOT = Path(SPECPATH).resolve()
ICON = ROOT / "resources" / "icons" / "app_icon.png"
BUILD_INFO = ROOT / "build" / "generated" / "build_info.json"
RUNTIME_LOCK = ROOT / "requirements" / "runtime-py312.lock"

if not BUILD_INFO.is_file():
    raise SystemExit(
        "build/generated/build_info.json is missing; run "
        "tools/generate_build_info.py before PyInstaller"
    )

version_source = (ROOT / "src" / "__init__.py").read_text(encoding="utf-8")
version_match = re.search(
    r'^__version__\s*=\s*"([^"]+)"',
    version_source,
    flags=re.MULTILINE,
)
if version_match is None:
    raise SystemExit("src.__version__ could not be read")
APP_VERSION = version_match.group(1)

datas = [
    (str(ROOT / "resources"), "resources"),
    (str(BUILD_INFO), "resources"),
    (str(ROOT / "schemas"), "schemas"),
    (str(RUNTIME_LOCK), "requirements"),
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

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="ArchMeshRubbing.app",
        icon=str(ICON),
        bundle_identifier="io.github.lzpxilfe.ArchMeshRubbing",
        version=APP_VERSION,
        info_plist={"NSPrincipalClass": "NSApplication"},
    )
