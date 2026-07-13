"""Hand-drawn, deterministic pixel icons for the native Qt workbench.

The UI deliberately avoids emoji and host-font icon glyphs.  Every icon is a
12-by-12 authored pixel grid centered on a 16-by-16 transparent canvas, using a
small archaeology-inspired palette.  High-DPI variants use nearest-neighbour
scaling so Windows font or theme changes cannot alter their appearance.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QImage, QPixmap


PIXEL_ICON_SIZE = 16
_GRID_SIZE = 12
_GRID_OFFSET = (PIXEL_ICON_SIZE - _GRID_SIZE) // 2

_PALETTE = {
    ".": "#00000000",
    "A": "#718087",
    "B": "#39738C",
    "C": "#B8643E",
    "G": "#4F7D57",
    "K": "#263238",
    "L": "#A9CBD5",
    "R": "#B44747",
    "S": "#D4B483",
    "W": "#F5F0E6",
    "Y": "#D3A33C",
}
_DISABLED_PALETTE = {
    key: ("#00000000" if key == "." else "#8A9499") for key in _PALETTE
}


def _grid(*rows: str) -> tuple[str, ...]:
    if len(rows) != _GRID_SIZE:
        raise ValueError(f"pixel icon must have {_GRID_SIZE} rows")
    if any(len(row) != _GRID_SIZE for row in rows):
        raise ValueError(f"pixel icon rows must have {_GRID_SIZE} columns")
    unknown = set("".join(rows)) - set(_PALETTE)
    if unknown:
        raise ValueError(f"pixel icon uses unknown palette entries: {sorted(unknown)}")
    return tuple(rows)


_ICONS: dict[str, tuple[str, ...]] = {
    "open_mesh": _grid(
        "............",
        "..KK........",
        ".KSSKK......",
        ".KSSSSKKKK..",
        ".KBBBBBBBK..",
        ".KBBBBBBBK..",
        ".KBBBLLBBK..",
        ".KBBLLLLBK..",
        ".KBBBBBBBK..",
        "..KKKKKKK...",
        "............",
        "............",
    ),
    "open_project": _grid(
        "............",
        "..KK........",
        ".KSSKK......",
        ".KSSSSKKKK..",
        ".KCCCCCCCK..",
        ".KCCKKKCCK..",
        ".KCCKWKCCK..",
        ".KCCKKKCCK..",
        ".KCCCCCCCK..",
        "..KKKKKKK...",
        "............",
        "............",
    ),
    "save": _grid(
        ".KKKKKKKKKK.",
        ".KBBBBBBBBK.",
        ".KBWWWWBBBK.",
        ".KBWWWWBBBK.",
        ".KBBBBBBBBK.",
        ".KBBBBBBBBK.",
        ".KBKKKKKBBK.",
        ".KBKWWWKBBK.",
        ".KBKWWWKBBK.",
        ".KBKKKKKBBK.",
        ".KKKKKKKKKK.",
        "............",
    ),
    "fit": _grid(
        ".KK......KK.",
        ".K........K.",
        "............",
        "....CC......",
        "...CCCC.....",
        "..CCCCCC....",
        "..CCCCCC....",
        "...CCCC.....",
        "....CC......",
        "............",
        ".K........K.",
        ".KK......KK.",
    ),
    "reset": _grid(
        "....KKKK....",
        "..KKBBBBKK..",
        ".KBB....BBK.",
        ".KB......BK.",
        "KK........K.",
        "KYY.......K.",
        "KYYY......K.",
        "KKYK.....BK.",
        "...K....BBK.",
        "....KKBBKK..",
        "......KK....",
        "............",
    ),
    "view_front": _grid(
        "..KKKKKKKK..",
        "..KCCCCCCK..",
        "..KCCCCCCK..",
        "..KCCBBCCK..",
        "..KCCBBCCK..",
        "..KCCBBCCK..",
        "..KCCBBCCK..",
        "..KCCCCCCK..",
        "..KCCCCCCK..",
        "..KKKKKKKK..",
        "............",
        "............",
    ),
    "view_back": _grid(
        "..KKKKKKKK..",
        "..KSSSSSSK..",
        "..KS.KK.SK..",
        "..K.KSSK.K..",
        "..KKSSSSKK..",
        "..KKSSSSKK..",
        "..K.KSSK.K..",
        "..KS.KK.SK..",
        "..KSSSSSSK..",
        "..KKKKKKKK..",
        "............",
        "............",
    ),
    "view_right": _grid(
        "...KKKKKK...",
        "...KCCCCKK..",
        "...KCCCCCCK.",
        "...KCCCCCCK.",
        "...KCCCCCCK.",
        "...KCCCCCCK.",
        "...KCCCCCCK.",
        "...KCCCCCCK.",
        "...KCCCCKK..",
        "...KKKKKK...",
        "............",
        "............",
    ),
    "view_left": _grid(
        "...KKKKKK...",
        "..KKCCCCK...",
        ".KCCCCCCK...",
        ".KCCCCCCK...",
        ".KCCCCCCK...",
        ".KCCCCCCK...",
        ".KCCCCCCK...",
        ".KCCCCCCK...",
        "..KKCCCCK...",
        "...KKKKKK...",
        "............",
        "............",
    ),
    "view_top": _grid(
        "..KKKKKKKK..",
        "..KBBBBBBK..",
        "..KBBBBBBK..",
        "..KSSSSSSK..",
        "..KSSSSSSK..",
        "..KSSSSSSK..",
        "..KSSSSSSK..",
        "..KSSSSSSK..",
        "..KSSSSSSK..",
        "..KKKKKKKK..",
        "............",
        "............",
    ),
    "view_bottom": _grid(
        "..KKKKKKKK..",
        "..KSSSSSSK..",
        "..KSSSSSSK..",
        "..KSSSSSSK..",
        "..KSSSSSSK..",
        "..KSSSSSSK..",
        "..KSSSSSSK..",
        "..KBBBBBBK..",
        "..KBBBBBBK..",
        "..KKKKKKKK..",
        "............",
        "............",
    ),
    "align": _grid(
        ".....K......",
        "....KYK.....",
        "...KYYYK....",
        ".....K......",
        "....CCC.....",
        "...CCCCC....",
        "..CCCCCCC...",
        "...CCCCC....",
        "....CCC.....",
        "KKKKKKKKKKKK",
        "BBBBBBBBBBBB",
        "............",
    ),
    "ground": _grid(
        ".....K......",
        ".....K......",
        ".....K......",
        "..K..K..K...",
        "...K.K.K....",
        "....KKK.....",
        ".....K......",
        "............",
        "....CCC.....",
        "KKKKKKKKKKKK",
        "BBBBBBBBBBBB",
        "............",
    ),
    "lock": _grid(
        "....KKKK....",
        "...K....K...",
        "..K......K..",
        "..K......K..",
        "..KKKKKKKK..",
        "..KYYYYYYK..",
        "..KYYYKYYK..",
        "..KYYKKK.K..",
        "..KYYYKYYK..",
        "..KYYYYYYK..",
        "..KKKKKKKK..",
        "............",
    ),
    "flat": _grid(
        "..KKKKKKKK..",
        "..KWWWWCCK..",
        "..KWWWWCCK..",
        "..KWWWWCCK..",
        "..KWWWWCCK..",
        "..KCCCCWWK..",
        "..KCCCCWWK..",
        "..KCCCCWWK..",
        "..KCCCCWWK..",
        "..KKKKKKKK..",
        "............",
        "............",
    ),
    "xray": _grid(
        ".....KK.....",
        "....KLLK....",
        "....KLLK....",
        "..KKKLLKKK..",
        ".KLLKLLKLLK.",
        ".KLLKLLKLLK.",
        "..KKKLLKKK..",
        ".KLLKLLKLLK.",
        ".KLLKLLKLLK.",
        "..KKKLLKKK..",
        "....KLLK....",
        "....KKKK....",
    ),
    "cutline": _grid(
        "....KKKK....",
        "..KKCCCCKK..",
        ".KCCCCCCCCK.",
        ".KCCCCCCCCK.",
        "KKKKKKKKKKKK",
        "RRRRRRRRRRRR",
        "KKKKKKKKKKKK",
        ".KCCCCCCCCK.",
        ".KCCCCCCCCK.",
        "..KKCCCCKK..",
        "....KKKK....",
        "............",
    ),
    "outline": _grid(
        "....KKKK....",
        "..KK....KK..",
        ".K........K.",
        ".K..CCCC..K.",
        "K..C....C..K",
        "K.C......C.K",
        "K.C......C.K",
        "K..C....C..K",
        ".K..CCCC..K.",
        ".K........K.",
        "..KKKKKKKK..",
        "............",
    ),
    "rubbing": _grid(
        "..KKKKKKKK..",
        "..KWWWWWWK..",
        "..KWKWWKWK..",
        "..KWWKKWWK..",
        "..KWKKKKWK..",
        "..KWWKKWWK..",
        "..KWKWWKWK..",
        "..KWWKKWWK..",
        "..KWWWWWWK..",
        "..KKKKKKKK..",
        "....CCCC....",
        "............",
    ),
    "flatten": _grid(
        "..KKKKKKKK..",
        ".KCCCCCCCCK.",
        "KCCCCCCCCCCK",
        ".KK......KK.",
        "............",
        ".....K......",
        "...KYYYK....",
        "....KYK.....",
        ".....K......",
        "............",
        "KKKKKKKKKKKK",
        "BBBBBBBBBBBB",
    ),
    "record_top": _grid(
        ".....K......",
        "....KYK.....",
        "...KYYYK....",
        ".....K......",
        "............",
        "..KKKKKKKK..",
        ".KCCCCCCCCK.",
        "KCCCCCCCCCCK",
        "KCCCCCCCCCCK",
        ".KKKKKKKKKK.",
        "............",
        "............",
    ),
    "record_bottom": _grid(
        ".KKKKKKKKKK.",
        "KCCCCCCCCCCK",
        "KCCCCCCCCCCK",
        ".KCCCCCCCCK.",
        "..KKKKKKKK..",
        "............",
        ".....K......",
        "...KYYYK....",
        "....KYK.....",
        ".....K......",
        "............",
        "............",
    ),
    "preview": _grid(
        "............",
        "....KKKK....",
        "..KKLLLLKK..",
        ".KLLLKKLLLK.",
        "KLLLKBBKLLLK",
        "KLLLKBBKLLLK",
        ".KLLLKKLLLK.",
        "..KKLLLLKK..",
        "....KKKK....",
        "............",
        "............",
        "............",
    ),
    "export": _grid(
        ".....KK.....",
        ".....KYK....",
        ".....KYYK...",
        "KKKKKKYYYK..",
        "KYYYYYYYYYK.",
        "KKKKKKYYYK..",
        ".....KYYK...",
        "..KKKKYK....",
        "..K....K....",
        "..KBBBBBBBK.",
        "..KKKKKKKKK.",
        "............",
    ),
    "measure": _grid(
        ".........KK.",
        "........KSK.",
        ".......KSSK.",
        "......KSSK..",
        ".....KSSK...",
        "....KSSK....",
        "...KSSK.....",
        "..KSSK......",
        ".KSSK.......",
        ".KSK........",
        ".KK.........",
        "............",
    ),
    "details": _grid(
        "..K......K..",
        "..K..KK..K..",
        "KKKKKYYKKKKK",
        "..K..KK..K..",
        "..K......K..",
        "............",
        "..K......K..",
        "KKKKKK..KKKK",
        "..K..K..K...",
        "..K......K..",
        "..K......K..",
        "............",
    ),
    "delete": _grid(
        "....KKKK....",
        "...KCCCCK...",
        "..KKKKKKKK..",
        "..KRRRRRRK..",
        "..KRKRRKRK..",
        "..KRKRRKRK..",
        "..KRKRRKRK..",
        "..KRKRRKRK..",
        "..KRRRRRRK..",
        "..KKKKKKKK..",
        "............",
        "............",
    ),
    "visible": _grid(
        "............",
        "....KKKK....",
        "..KKLLLLKK..",
        ".KLLLKKLLLK.",
        "KLLLKBBKLLLK",
        "KLLLKBBKLLLK",
        ".KLLLKKLLLK.",
        "..KKLLLLKK..",
        "....KKKK....",
        "............",
        "............",
        "............",
    ),
    "hidden": _grid(
        "............",
        "....KKKK..R.",
        "..KKAAAARR..",
        ".KAAAKKRRRK.",
        "KAAAKRRKAAAK",
        "KAAKRRAKAAAK",
        ".KRRRKKAAAK.",
        "..RRAAAAKK..",
        ".R..KKKK....",
        "............",
        "............",
        "............",
    ),
    "selection": _grid(
        "K...........",
        "KK..........",
        "KCK.........",
        "KCCK........",
        "KCCCK.......",
        "KCCCCK......",
        "KCCCCCK.....",
        "KCCKKKK.....",
        "KCK.KK......",
        "KK...KK.....",
        "K.....KK....",
        "............",
    ),
    "copy": _grid(
        "....KKKKKK..",
        "....KWWWWK..",
        "..KKKWWWWK..",
        "..KWWWWWWK..",
        "..KWWKKKKK..",
        "..KWWK......",
        "..KWWK......",
        "..KWWK......",
        "..KWWK......",
        "..KKKK......",
        "............",
        "............",
    ),
    "camera": _grid(
        "............",
        "....KKKK....",
        "...KCCCCK...",
        ".KKKKKKKKKK.",
        ".KBBBBBBBBK.",
        ".KBBKKKKBBK.",
        ".KBKLLLLKBK.",
        ".KBKLLLLKBK.",
        ".KBBKKKKBBK.",
        ".KBBBBBBBBK.",
        ".KKKKKKKKKK.",
        "............",
    ),
    "help": _grid(
        "...KKKKKK...",
        "..KYYYYYYK..",
        ".KYYKKKKYYK.",
        ".KYYK..KYYK.",
        "....K.KYYK..",
        ".....KYYK...",
        ".....KYK....",
        ".....KK.....",
        "............",
        ".....KK.....",
        ".....KK.....",
        "............",
    ),
}


def available_pixel_icons() -> tuple[str, ...]:
    """Return the stable public icon-name set."""

    return tuple(sorted(_ICONS))


def pixel_icon_grid(name: str) -> tuple[str, ...]:
    """Return immutable source pixels for tests and design review."""

    try:
        return _ICONS[name]
    except KeyError as exc:
        raise KeyError(f"unknown pixel icon: {name}") from exc


def _render_image(name: str, *, disabled: bool = False) -> QImage:
    rows = pixel_icon_grid(name)
    palette = _DISABLED_PALETTE if disabled else _PALETTE
    image = QImage(
        PIXEL_ICON_SIZE,
        PIXEL_ICON_SIZE,
        QImage.Format.Format_ARGB32,
    )
    image.fill(QColor("#00000000"))
    for y, row in enumerate(rows, start=_GRID_OFFSET):
        for x, token in enumerate(row, start=_GRID_OFFSET):
            if token != ".":
                image.setPixelColor(x, y, QColor(palette[token]))
    return image


def _add_mode_pixmaps(icon: QIcon, name: str, mode: QIcon.Mode) -> None:
    image = _render_image(name, disabled=mode == QIcon.Mode.Disabled)
    pixmap = QPixmap.fromImage(image)
    icon.addPixmap(pixmap, mode, QIcon.State.Off)

    high_dpi = pixmap.scaled(
        PIXEL_ICON_SIZE * 2,
        PIXEL_ICON_SIZE * 2,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )
    high_dpi.setDevicePixelRatio(2.0)
    icon.addPixmap(high_dpi, mode, QIcon.State.Off)


@lru_cache(maxsize=None)
def _pixel_icon_template(name: str) -> QIcon:
    """Build and retain one immutable, theme-independent icon template."""

    icon = QIcon()
    _add_mode_pixmaps(icon, name, QIcon.Mode.Normal)
    _add_mode_pixmaps(icon, name, QIcon.Mode.Disabled)
    return icon


def pixel_icon(name: str) -> QIcon:
    """Return a cheap shared-data copy of a cached pixel-icon template.

    Qt falls back from Active/Selected to Normal and from On to Off, so storing
    duplicate pixmaps for those states only increases GUI construction cost.
    Returning a QIcon copy prevents callers from mutating the cached template.
    """

    return QIcon(_pixel_icon_template(name))


def set_pixel_icon(target: Any, name: str, *, size: int = PIXEL_ICON_SIZE) -> None:
    """Attach an icon to a QAction or button and record its semantic name."""

    target.setIcon(pixel_icon(name))
    if callable(getattr(target, "setIconSize", None)):
        target.setIconSize(QSize(size, size))
    if callable(getattr(target, "setProperty", None)):
        target.setProperty("pixelIconName", name)


def require_pixel_icons(names: Iterable[str]) -> None:
    """Fail early when a UI surface references an undefined icon."""

    missing = sorted(set(names) - set(_ICONS))
    if missing:
        raise KeyError("undefined pixel icons: " + ", ".join(missing))
