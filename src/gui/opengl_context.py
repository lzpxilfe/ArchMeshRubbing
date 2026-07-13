"""OpenGL surface contract shared by the application and driver smoke.

The native viewport still uses the OpenGL 2.1 fixed-function compatibility
API.  Request that contract explicitly before constructing QApplication so Qt
does not silently choose an incompatible core or OpenGL ES context.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
from typing import Any, MutableMapping

from PyQt6.QtGui import QGuiApplication, QSurfaceFormat


OPENGL_MINIMUM_VERSION = (2, 1)
OPENGL_MINIMUM_DEPTH_BITS = 24
_WINDOWS_SOFTWARE_GL_DLL: Any | None = None
_WINDOWS_SOFTWARE_GL_DIRECTORY: Any | None = None


def _bind_pyopengl_windows_dll(gl_platform: Any, dll: Any) -> None:
    """Point PyOpenGL's Win32 dispatch at the same DLL selected by Qt."""

    function_type = getattr(ctypes, "WINFUNCTYPE", None)
    if function_type is None:
        raise RuntimeError("ctypes.WINFUNCTYPE is unavailable")
    dll.FunctionType = function_type
    get_current_context = dll.wglGetCurrentContext
    get_current_context.restype = ctypes.c_void_p
    get_extension_procedure = dll.wglGetProcAddress
    get_extension_procedure.restype = ctypes.c_void_p

    platform = gl_platform.PLATFORM
    bindings = {
        "GL": dll,
        "OpenGL": dll,
        "WGL": dll,
        "GetCurrentContext": get_current_context,
        "CurrentContextIsValid": get_current_context,
        "getExtensionProcedure": get_extension_procedure,
    }
    for name, value in bindings.items():
        setattr(platform, name, value)
        if name in {
            "GetCurrentContext",
            "CurrentContextIsValid",
            "getExtensionProcedure",
        }:
            setattr(gl_platform, name, value)


def install_windows_software_pyopengl_bridge(
    *,
    environ: MutableMapping[str, str] | None = None,
) -> str | None:
    """Share Qt's Windows software-OpenGL DLL with PyOpenGL.

    Qt dynamically loads ``opengl32sw.dll`` when ``QT_OPENGL=software``.
    PyOpenGL otherwise dispatches native calls through the system
    ``opengl32.dll``. Mixing those two WGL implementations in one widget can
    terminate the process, so bind PyOpenGL to Qt's exact DLL before importing
    ``OpenGL.GL`` or the viewport module.
    """

    global _WINDOWS_SOFTWARE_GL_DLL
    global _WINDOWS_SOFTWARE_GL_DIRECTORY

    target_environ = os.environ if environ is None else environ
    if sys.platform != "win32":
        return None
    if str(target_environ.get("QT_OPENGL", "")).strip().casefold() != "software":
        return None
    if QGuiApplication.instance() is not None:
        raise RuntimeError(
            "Windows software OpenGL bridge must be installed before QApplication"
        )

    from PyQt6.QtCore import QLibraryInfo

    dll_path = (
        Path(
            QLibraryInfo.path(
                QLibraryInfo.LibraryPath.BinariesPath,
            )
        )
        / "opengl32sw.dll"
    ).resolve()
    if not dll_path.is_file():
        raise RuntimeError(f"Qt software OpenGL DLL is missing: {dll_path}")
    target_environ["QT_OPENGL_DLL"] = str(dll_path)

    if _WINDOWS_SOFTWARE_GL_DLL is None:
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is None:
            raise RuntimeError("os.add_dll_directory is unavailable")
        win_dll = getattr(ctypes, "WinDLL", None)
        if win_dll is None:
            raise RuntimeError("ctypes.WinDLL is unavailable")
        _WINDOWS_SOFTWARE_GL_DIRECTORY = add_dll_directory(str(dll_path.parent))
        _WINDOWS_SOFTWARE_GL_DLL = win_dll(str(dll_path))

    from OpenGL import platform as gl_platform

    _bind_pyopengl_windows_dll(gl_platform, _WINDOWS_SOFTWARE_GL_DLL)
    return str(dll_path)


def compatibility_surface_format() -> QSurfaceFormat:
    """Return the native viewport's explicit compatibility surface format."""

    surface_format = QSurfaceFormat()
    surface_format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    surface_format.setVersion(*OPENGL_MINIMUM_VERSION)
    surface_format.setProfile(
        QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile
    )
    surface_format.setDepthBufferSize(OPENGL_MINIMUM_DEPTH_BITS)
    surface_format.setStencilBufferSize(8)
    surface_format.setSamples(0)
    return surface_format


def install_compatibility_surface_format() -> QSurfaceFormat:
    """Install the viewport format before the first GUI application exists."""

    if QGuiApplication.instance() is not None:
        raise RuntimeError(
            "OpenGL surface format must be installed before QApplication"
        )
    surface_format = compatibility_surface_format()
    QSurfaceFormat.setDefaultFormat(surface_format)
    return surface_format


__all__ = [
    "OPENGL_MINIMUM_DEPTH_BITS",
    "OPENGL_MINIMUM_VERSION",
    "compatibility_surface_format",
    "install_compatibility_surface_format",
    "install_windows_software_pyopengl_bridge",
]
