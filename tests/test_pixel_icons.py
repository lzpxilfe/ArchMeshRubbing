from __future__ import annotations

import os
from pathlib import Path
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QPushButton

from src.gui.pixel_icons import (
    PIXEL_ICON_SIZE,
    available_pixel_icons,
    pixel_icon,
    pixel_icon_grid,
    require_pixel_icons,
    set_pixel_icon,
)


ROOT = Path(__file__).resolve().parents[1]
_UI_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u20E3]"
)


def _application() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication(["pixel-icon-test", "-platform", "offscreen"])


def test_pixel_icon_catalog_is_fixed_grid_and_complete_for_core_workflow() -> None:
    required = {
        "open_mesh",
        "open_project",
        "recover",
        "save",
        "align",
        "cutline",
        "outline",
        "rubbing",
        "export",
        "view_front",
        "view_back",
        "view_right",
        "view_left",
        "view_top",
        "view_bottom",
    }
    names = set(available_pixel_icons())
    assert required <= names
    require_pixel_icons(required)
    for name in names:
        rows = pixel_icon_grid(name)
        assert len(rows) == 12
        assert {len(row) for row in rows} == {12}


def test_pixel_icons_render_without_font_or_theme_glyphs() -> None:
    app = _application()
    for name in available_pixel_icons():
        icon = pixel_icon(name)
        assert not icon.isNull()
        assert icon.cacheKey() == pixel_icon(name).cacheKey()
        pixmap = icon.pixmap(QSize(PIXEL_ICON_SIZE, PIXEL_ICON_SIZE))
        assert not pixmap.isNull()
        active = icon.pixmap(
            QSize(PIXEL_ICON_SIZE, PIXEL_ICON_SIZE),
            QIcon.Mode.Active,
            QIcon.State.On,
        )
        assert not active.isNull()
        image = pixmap.toImage()
        opaque = sum(
            1
            for y in range(image.height())
            for x in range(image.width())
            if image.pixelColor(x, y).alpha() > 0
        )
        assert opaque >= 8, name

    button = QPushButton("검증")
    set_pixel_icon(button, "align")
    assert button.property("pixelIconName") == "align"
    assert button.iconSize() == QSize(PIXEL_ICON_SIZE, PIXEL_ICON_SIZE)
    app.processEvents()


def test_production_ui_source_contains_no_emoji_glyphs() -> None:
    for relative in ("app_interactive.py", "src/gui/viewport_3d.py"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        match = _UI_EMOJI_RE.search(text)
        assert match is None, f"{relative} still contains emoji U+{ord(match.group()):04X}"
