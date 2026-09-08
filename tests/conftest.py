"""Shared collection rules for the test suite.

The product target is Windows with a working Qt/OpenGL stack, and CI runs
there.  Most of the codebase is Qt-free core, though, so a contributor on a
machine without Qt libraries should still be able to exercise it instead of
watching an import error take out the whole run.  Skip only the modules that
genuinely need Qt, and say why.
"""

from __future__ import annotations

import os
from pathlib import Path


# Modules that import PyQt6 (directly or through app_interactive/viewport_3d)
# at collection time and therefore cannot be collected without Qt libraries.
_QT_DEPENDENT_MODULES = (
    "test_gui_smoke.py",
    "test_opengl_driver_smoke.py",
    "test_pixel_icons.py",
    "test_rotation_convention.py",
    "test_viewport_render_origin.py",
)


#: What counts as "yes, skip the Qt modules".  A plain truthiness test would
#: read ``0`` as yes, and the header below tells the reader to set exactly
#: that when they want the modules to run.
_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})


def _qt_is_importable() -> bool:
    if os.environ.get("ARCHMESHRUBBING_SKIP_QT_TESTS", "").strip().lower() in _TRUE_WORDS:
        return False
    try:  # pragma: no cover - depends on the host, not on branch logic
        import PyQt6.QtWidgets  # noqa: F401
    except Exception:
        return False
    return True


collect_ignore: list[str] = []

if not _qt_is_importable():
    collect_ignore.extend(_QT_DEPENDENT_MODULES)


def pytest_report_header(config: object) -> str | None:
    """Make a partial run obvious in the log rather than silently narrower."""

    if not collect_ignore:
        return None
    names = ", ".join(sorted(Path(name).stem for name in collect_ignore))
    return (
        "Qt is unavailable: skipping GUI/OpenGL modules "
        f"({names}). Unset ARCHMESHRUBBING_SKIP_QT_TESTS and install PyQt6 to "
        "run them; the Windows CI job always does."
    )
