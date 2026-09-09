"""The panel asks for what each reading needs, and nothing it does not.

Two of the seven readings read the scanner's images: they need the OBJ's
texture coordinates and one map beside it, and which map decides what is
read.  The rest need none of that, and must not be made to ask for it.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.application.artifact_readings import (  # noqa: E402
    PAINT_CUTOUT,
    PROFILE_BREAK,
    READING_KINDS,
    TEXTURE_LINES,
)
from src.core.artifact_texture_lines import (  # noqa: E402
    TEXTURE_LINES_DOMAIN_AXIS,
    TEXTURE_LINES_RIDGE_RULE,
)
from src.gui.readings_panel import ReadingsPanel  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def panel(app: QApplication) -> ReadingsPanel:
    return ReadingsPanel()


def _choose(panel: ReadingsPanel, kind: str) -> None:
    panel.combo_kind.setCurrentIndex(panel.combo_kind.findData(kind))


def _shown(panel: ReadingsPanel, widget) -> bool:
    """Whether the panel would show this control once the panel is shown.

    `isVisible` is false for every child of a widget nobody has shown, so it
    cannot answer the question these tests ask.
    """

    return bool(widget.isVisibleTo(panel))


def test_all_seven_readings_are_offered(panel: ReadingsPanel) -> None:
    offered = [panel.combo_kind.itemData(index) for index in range(panel.combo_kind.count())]
    assert offered == list(READING_KINDS)
    assert TEXTURE_LINES in offered and PAINT_CUTOUT in offered


def test_the_files_are_asked_for_only_by_the_readings_that_read_them(panel: ReadingsPanel) -> None:
    _choose(panel, PROFILE_BREAK)
    assert not _shown(panel, panel.group_texture)
    _choose(panel, TEXTURE_LINES)
    assert _shown(panel, panel.group_texture)
    assert _shown(panel, panel.combo_texture_map), "either map can trace a line"
    _choose(panel, PAINT_CUTOUT)
    assert _shown(panel, panel.group_texture)
    assert not _shown(panel, panel.combo_texture_map), "a cutout is read from colour only"
    assert not _shown(panel, panel.combo_texture_rule)


def test_the_window_owns_the_dialogue_and_the_panel_owns_the_path(panel: ReadingsPanel) -> None:
    """The panel touches no disk: it asks for a file and is told the path."""

    asked: list[str] = []
    panel.fileRequested.connect(asked.append)
    _choose(panel, TEXTURE_LINES)
    panel.btn_atlas_path.click()
    panel.btn_map_path.click()
    assert asked == ["atlas", "normal"]

    panel.set_path("atlas", "/scan/pot.obj")
    panel.set_path("normal", "/scan/pot_nor.png")
    options = panel.options()
    assert options["atlas_path"] == "/scan/pot.obj"
    assert options["normal_map_path"] == "/scan/pot_nor.png"
    assert "colour_map_path" not in options


def test_choosing_the_colour_map_changes_what_is_read(panel: ReadingsPanel) -> None:
    """One reading, two maps: the normal map gives the incised line, the
    colour map the painted one, and the option says which."""

    _choose(panel, TEXTURE_LINES)
    panel.combo_texture_map.setCurrentIndex(panel.combo_texture_map.findData("colour"))
    panel.set_path("colour", "/scan/pot_bc.png")
    options = panel.options()
    assert options["colour_map_path"] == "/scan/pot_bc.png"
    assert "normal_map_path" not in options

    asked: list[str] = []
    panel.fileRequested.connect(asked.append)
    panel.btn_map_path.click()
    assert asked == ["colour"], "the dialogue asks for the map this reading needs"


def test_a_cutout_always_asks_for_colour(panel: ReadingsPanel) -> None:
    _choose(panel, PAINT_CUTOUT)
    assert panel.current_map_kind() == "colour"
    panel.set_path("atlas", "/scan/pot.obj")
    panel.set_path("colour", "/scan/pot_bc.png")
    options = panel.options()
    assert options["colour_map_path"] == "/scan/pot_bc.png"
    assert "rule" not in options and "domain" not in options


def test_the_rule_and_the_domain_travel_with_a_pattern_reading(panel: ReadingsPanel) -> None:
    """A painted line is read as the rubbing's paper reads a stroke, on the
    wall unrolled - not with the curvature rule that reads an incision.  The
    archaeologist chooses, and the choice reaches the core."""

    _choose(panel, TEXTURE_LINES)
    panel.combo_texture_rule.setCurrentIndex(
        panel.combo_texture_rule.findData(TEXTURE_LINES_RIDGE_RULE)
    )
    panel.combo_texture_domain.setCurrentIndex(
        panel.combo_texture_domain.findData(TEXTURE_LINES_DOMAIN_AXIS)
    )
    options = panel.options()
    assert options["rule"] == TEXTURE_LINES_RIDGE_RULE
    assert options["domain"] == TEXTURE_LINES_DOMAIN_AXIS


def test_a_reading_can_be_put_back_on_the_panel_to_take_again(panel: ReadingsPanel) -> None:
    """Reading the same pattern at a finer resolution should not mean
    typing every path and every choice a second time."""

    _choose(panel, TEXTURE_LINES)
    panel.set_options(
        {
            "atlas_path": "/scan/bowl.obj",
            "colour_map_path": "/scan/bowl_bc.png",
            "pixels_per_mm": 40,
            "rule": TEXTURE_LINES_RIDGE_RULE,
            "domain": TEXTURE_LINES_DOMAIN_AXIS,
            "view": "right",
        }
    )
    options = panel.options()
    assert options["atlas_path"] == "/scan/bowl.obj"
    assert options["colour_map_path"] == "/scan/bowl_bc.png"
    assert options["pixels_per_mm"] == 40
    assert options["rule"] == TEXTURE_LINES_RIDGE_RULE
    assert options["view"] == "right"


def test_taking_a_reading_carries_the_kind_the_id_and_the_options(panel: ReadingsPanel) -> None:
    taken: list[tuple[str, str, dict]] = []
    panel.readingRequested.connect(lambda *args: taken.append(args))
    _choose(panel, TEXTURE_LINES)
    panel.edit_record_id.setText("record:pattern")
    panel.set_path("atlas", "/scan/pot.obj")
    panel.btn_take.click()
    assert len(taken) == 1
    kind, record_id, options = taken[0]
    assert kind == TEXTURE_LINES
    assert record_id == "record:pattern"
    assert options["atlas_path"] == "/scan/pot.obj"
