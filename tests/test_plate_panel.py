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

from PyQt6.QtWidgets import QApplication, QComboBox  # noqa: E402

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


def _set_record(table, row: int, column: int, value: str) -> None:
    """Choose a record in a table cell, the way the panel now offers it."""

    widget = table.cellWidget(row, column)
    assert isinstance(widget, QComboBox), "a record column is chosen, not typed"
    PlatePanel._select_record(widget, value)


def _record_text(table, row: int, column: int) -> str:
    widget = table.cellWidget(row, column)
    assert isinstance(widget, QComboBox), "a record column is chosen, not typed"
    return str(widget.currentData() or "")


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
    _set_record(panel.table_jogs, row, 0, ELEVATION)
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
    assert _record_text(panel.table_mirror, 0, 2) == FAR


def test_a_record_is_chosen_from_the_session_not_typed_out(panel: PlatePanel) -> None:
    """`record:cutline:<uuid4>` is forty-five characters.  Retyping it into
    ten tables is not a smaller version of the job - it is a different job,
    and one no reader can check by eye."""

    panel.combo_layout.setCurrentIndex(panel.combo_layout.findData("mirror"))
    widget = panel.table_mirror.cellWidget(0, 0)
    assert isinstance(widget, QComboBox)
    offered = [widget.itemData(index) for index in range(widget.count())]
    assert offered[0] == "", "a row can be left empty"
    assert ELEVATION in offered and SECTION in offered
    assert widget.itemText(offered.index(ELEVATION)).startswith("정면 외형선")

    _set_record(panel.table_mirror, 0, 0, ELEVATION)
    _set_record(panel.table_mirror, 0, 1, SECTION)
    assert panel.spec()["mirror_sections"] == [[ELEVATION, SECTION]]


def test_a_record_this_session_does_not_hold_is_kept_and_marked(panel: PlatePanel) -> None:
    """A spec written on another machine, or before a re-import, names
    records this session has never seen.  Dropping them silently would edit
    the archaeologist's plate behind their back."""

    stranger = "record:outline:0000-not-here"
    panel.set_spec(
        plate_spec(
            [ELEVATION],
            DrawingSheetOptions(
                title_block=TitleBlock(artifact_label="남의 명세"),
                mirror_sections=((stranger, SECTION),),
            ),
        )
    )
    assert _record_text(panel.table_mirror, 0, 0) == stranger
    widget = panel.table_mirror.cellWidget(0, 0)
    assert "이 세션에 없음" in widget.currentText()
    assert panel.spec()["mirror_sections"] == [[stranger, SECTION]]


def test_reloading_the_record_list_does_not_lose_what_is_chosen(panel: PlatePanel) -> None:
    """The list refreshes whenever a reading is committed; a chooser that
    reset itself would undo the row the archaeologist had just filled."""

    _set_record(panel.table_stipples, 0, 0, CUTOUT)
    _set_record(panel.table_stipples, 0, 1, ELEVATION)
    panel.set_records(
        [
            (ELEVATION, "vector.outline.v1", "정면 외형선"),
            (SECTION, "vector.cutline.v1", "정면 단면"),
            (CUTOUT, "measurement.relief_shade.v1", "요철 음영"),
            ("record:new", "measurement.crease.v1", "새 능선"),
        ]
    )
    assert panel.spec()["relief_stipples"] == [[CUTOUT, ELEVATION]]
    widget = panel.table_stipples.cellWidget(0, 0)
    assert "record:new" in [widget.itemData(i) for i in range(widget.count())]


def test_rows_a_loaded_spec_adds_get_the_same_controls(panel: PlatePanel) -> None:
    """A table that grows must grow its choosers too, or exactly the rows a
    loaded spec added become free text while the first row is a chooser."""

    rows = ((ELEVATION, "top"), (SECTION, "left"), (CUTOUT, "bottom"))
    panel.set_spec(
        plate_spec(
            [ELEVATION],
            DrawingSheetOptions(
                title_block=TitleBlock(artifact_label="파편 셋"),
                sherd_breaks=rows,
            ),
        )
    )
    # The spec sorts its decisions, so the table shows them in that order.
    for row, (record_id, side) in enumerate(sorted(rows)):
        assert _record_text(panel.table_sherd, row, 0) == record_id
        side_widget = panel.table_sherd.cellWidget(row, 1)
        assert isinstance(side_widget, QComboBox), "the side is a closed choice on every row"
        assert side_widget.currentData() == side
    assert panel.spec()["sherd_breaks"] == [list(row) for row in sorted(rows)]


def test_the_scale_list_and_the_scale_box_stay_the_same_number(panel: PlatePanel) -> None:
    """Two widgets, one decision.  Picking 1:3 has to put 3 in the box, and
    typing 7 - which no report on the list uses - has to move the list to
    직접 입력 rather than leave it pointing at a scale the plate is not at."""

    index = panel.combo_scale.findData(3.0)
    assert index >= 0, "the list a report uses has 1:3 on it"
    panel.combo_scale.setCurrentIndex(index)
    assert panel.spin_scale.value() == 3
    assert panel.spec()["scale_denominator"] == 3.0

    panel.spin_scale.setValue(7)
    assert panel.combo_scale.currentData() is None, "7 is not on the list"
    assert panel.spec()["scale_denominator"] == 7.0

    # Choosing 직접 입력 itself changes nothing: it is a label, not a value.
    panel.combo_scale.setCurrentIndex(panel.combo_scale.findData(None))
    assert panel.spin_scale.value() == 7


def test_a_loaded_spec_moves_the_scale_list_to_what_it_says(panel: PlatePanel) -> None:
    panel.set_spec(
        plate_spec(
            [ELEVATION],
            DrawingSheetOptions(
                title_block=TitleBlock(artifact_label="축척 시험"),
                scale_denominator=6.0,
            ),
        )
    )
    assert panel.spin_scale.value() == 6
    assert panel.combo_scale.currentData() == 6.0
