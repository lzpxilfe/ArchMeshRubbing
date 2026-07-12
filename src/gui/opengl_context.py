"""OpenGL surface contract shared by the application and driver smoke.

The native viewport still uses the OpenGL 2.1 fixed-function compatibility
API.  Request that contract explicitly before constructing QApplication so Qt
does not silently choose an incompatible core or OpenGL ES context.
"""

from __future__ import annotations

from PyQt6.QtGui import QGuiApplication, QSurfaceFormat


OPENGL_MINIMUM_VERSION = (2, 1)
OPENGL_MINIMUM_DEPTH_BITS = 24


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
]
