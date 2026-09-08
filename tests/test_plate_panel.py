"""The plate panel holds every decision, and gives it back unchanged.

The composer takes some forty decisions and the window used to know
thirteen; the rest could only be written in Python.  These tests hold the
panel to the contract that makes it worth having: what goes in comes out,
the plate a real bottle was drawn with survives the trip, and a decision
the composer would refuse is refused here with a word the archaeologist
can act on.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.core.drawing_sheet import DrawingSheetOptions, TitleBlock  # noqa: E402
from src.core.drawing_sheet_spec import (  # noqa: E402
    REACH_TO_WALL,
    PlateSpecError,
    plate_spec,
    plate_spec_options,
)
from src.core.drawing_style import available_presets  # noqa: E402
from src.gui.plate_panel import PlatePanel  # noqa: E402

ELEVATION = "record:front"
SECTION = "record:section"
FAR = "record:far"
CUTOUT = "record:peony"


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def panel(app: QApplication) -> PlatePanel:
    widget = PlatePanel()
    widget.set_records(
        [
            (ELEVATION, "vector.outline.v1", "정면 외형선"),
            (SECTION, "vector.cutline.v1", "정면 단면"),
            (FAR, "measurement.far_silhouette.v1", "뒷면 실루엣"),
            (CUTOUT, "measurement.paint_cutout.v1", "청화 모란"),
        ]
    )
    widget.set_presets([(preset_id, preset_id) for preset_id in available_presets()])
    return widget


def _bottle_spec() -> dict:
    """A plate like the white porcelain bottle's: a step to the wall, two
    presumed lines, a pasted painting, a corner read, and an interpretation."""

    return plate_spec(
        [ELEVATION],
        DrawingSheetOptions(
            title_block=TitleBlock(artifact_label="백자청화모란문병", rows=(("자료", "제공 FBX"),)),
            scale_denominator=2.0,
            show_center_axis=True,
            line_cap="butt",
            mirror_sections=((ELEVATION, SECTION),),
            mirror_jogs=((ELEVATION, 18.05, 130.05, math.inf),),
            presumed_lines=(
                (ELEVATION, "wall_on", 186.5, 20.0),
                (ELEVATION, "floor", 12.0, 0.0),
            ),
            break_records=("record:breaks",),
            break_styles=(("record:breaks", 1, "broken_once"),),
            paint_cutouts=((CUTOUT, ELEVATION, "in_place"),),
            paint_cutout_ink_percent=100,
        ),
    )


def test_a_plate_the_bottle_was_drawn_with_goes_in_and_comes_out_the_same(panel: PlatePanel) -> None:
    spec = _bottle_spec()
    panel.set_spec(spec)
    again = panel.spec()
    # The panel does not carry the preset unless it was offered one it knows;
    # everything else must be exactly what it was given.
    for key in sorted(set(spec) - {"style_preset"}):
        assert again[key] == spec[key], key
    assert panel.checked_records() == [ELEVATION]
    # And the options the composer would be handed are the same options.
    assert plate_spec_options(again)[1].mirror_jogs == plate_spec_options(spec)[1].mirror_jogs


def test_the_panel_shows_a_step_to_the_wall_as_a_word_and_reads_it_back(panel: PlatePanel) -> None:
    panel.set_spec(_bottle_spec())
    assert panel.table_jogs.item(0, 3).text() == PlatePanel.WALL_WORD, "the table says 벽, not wall"
    assert panel.spec()["mirror_jogs"][0][3] == REACH_TO_WALL
    _records, options = panel.options()
    assert options.mirror_jogs[0][3] == math.inf
    # A number is still a number, and any of the three words is the wall.
    for written, expected in ((" 12.5 ", 12.5), ("벽", REACH_TO_WALL), ("wall", REACH_TO_WALL)):
        panel.table_jogs.item(0, 3).setText(written)
        assert panel.spec()["mirror_jogs"][0][3] == expected


def test_the_summary_says_what_the_plate_will_be_without_opening_a_tab(panel: PlatePanel) -> None:
    assert "고른 기록이 없습니다" in panel.summary.text()
    assert not panel.btn_plate.isEnabled()
    panel.set_spec(_bottle_spec())
    summary = panel.summary.text()
    for part in ("1개 기록", "A4 세로", "1:2", "미러 1쌍", "벽까지 1", "추정선 2", "채색 1"):
        assert part in summary, summary
    assert panel.btn_plate.isEnabled()


def test_a_half_filled_row_is_not_a_decision_and_a_bad_number_is_named(panel: PlatePanel) -> None:
    panel.set_spec(_bottle_spec())
    # A row with a record id but no heights is someone part way through
    # typing, not a jog: it is left out rather than guessed at.
    row = panel.table_jogs.rowCount() - 1
    panel.table_jogs.item(row, 0).setText(ELEVATION)
    assert len(panel.spec()["mirror_jogs"]) == 1
    panel.table_jogs.item(row, 1).setText("여기")
    panel.table_jogs.item(row, 2).setText("40")
    panel.table_jogs.item(row, 3).setText("10")
    with pytest.raises(PlateSpecError, match="숫자가 아닌"):
        panel.spec()
    assert "숫자가 아닌" in panel.summary.text(), "the summary says so instead of lying"


def test_the_panel_refuses_a_specification_it_cannot_mean(panel: PlatePanel) -> None:
    with pytest.raises(PlateSpecError, match="unknown keys"):
        panel.set_spec({**_bottle_spec(), "colour": "red"})
    with pytest.raises(PlateSpecError):
        panel.set_spec({"format": "other"})


def test_records_survive_a_refresh_of_the_list(panel: PlatePanel) -> None:
    panel.set_spec(_bottle_spec())
    assert panel.checked_records() == [ELEVATION]
    panel.set_records(
        [
            (ELEVATION, "vector.outline.v1", "정면 외형선"),
            (SECTION, "vector.cutline.v1", "정면 단면"),
        ]
    )
    assert panel.checked_records() == [ELEVATION], "a reload does not lose the choice"


def test_the_far_silhouette_column_is_used_only_when_the_outline_reaches_that_far(
    panel: PlatePanel,
) -> None:
    spec = plate_spec(
        [ELEVATION],
        DrawingSheetOptions(
            title_block=TitleBlock(artifact_label="청자 접시"),
            mirror_sections=((ELEVATION, SECTION),),
            outline_reach="far",
            far_silhouettes=((ELEVATION, FAR),),
        ),
    )
    panel.set_spec(spec)
    assert panel.spec()["far_silhouettes"] == [[ELEVATION, FAR]]
    index = panel.combo_outline_reach.findData("axis")
    panel.combo_outline_reach.setCurrentIndex(index)
    # The composer refuses a silhouette that would not be drawn, so the panel
    # stops naming it rather than making a plate the composer will reject.
    assert panel.spec()["far_silhouettes"] == []
    panel.options()
    # The record is still in the table, ready for the reach to go back.
    assert panel.table_mirror.item(0, 2).text() == FAR
