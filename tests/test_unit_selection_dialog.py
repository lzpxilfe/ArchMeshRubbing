"""원본 단위·좌표축 확인: the mapping may turn the artifact, never mirror it.

The dialogue takes the file's unit and says where each of its axes goes.
Three distinct axes is not enough to make that a turn: swapping two of them
- 원본 Y → +Z with 원본 Z → +Y, which is the obvious-looking way to stand a
Y-up scan on its base - has determinant -1, and the artifact comes in left
for right.  On a vessel turned on a wheel nothing on screen says so, and the
drawing that follows is wrong rather than differently labelled, so the
mapping that does it is refused here rather than recorded as left-handed.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from app_interactive import UnitSelectionDialog  # noqa: E402


#: One application for the module.  A QApplication that goes out of scope is
#: garbage collected, and it takes every widget with it.
_APP: QApplication | None = None


@pytest.fixture(scope="module", autouse=True)
def _app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    yield _APP


def _dialog(**axes: str) -> UnitSelectionDialog:
    dialog = UnitSelectionDialog()
    for key, value in axes.items():
        dialog.axis_combos[key].setCurrentText(value)
    dialog.confirm_metadata.setChecked(True)
    return dialog


def test_a_turn_is_accepted_and_a_swap_is_not() -> None:
    # Standing a Y-up scan on its base is a quarter turn about x, and one of
    # the two axes it moves changes sign.
    turned = _dialog(source_x="+X", source_y="+Z", source_z="-Y")
    assert turned._axes_keep_handedness()
    assert turned.ok_btn.isEnabled()
    assert turned.handedness_warning.text() == ""
    assert turned.get_source_metadata()["handedness"] == "right"

    # Leaving the sign off is the mistake, and it mirrors the artifact.
    mirrored = _dialog(source_x="+X", source_y="+Z", source_z="+Y")
    assert mirrored._axes_are_bijective()
    assert not mirrored._axes_keep_handedness()
    assert not mirrored.ok_btn.isEnabled()
    # The warning has to name the fix, not just the fault.
    assert "부호" in mirrored.handedness_warning.text()

    turned.deleteLater()
    mirrored.deleteLater()


def test_the_identity_and_a_half_turn_are_both_turns() -> None:
    for axes in (
        {"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        {"source_x": "-X", "source_y": "-Y", "source_z": "+Z"},
        {"source_x": "+Y", "source_y": "+Z", "source_z": "+X"},
    ):
        dialog = _dialog(**axes)
        assert dialog._axes_keep_handedness(), axes
        dialog.deleteLater()

    # A single reflection is not, however few axes it moves.
    for axes in (
        {"source_x": "-X", "source_y": "+Y", "source_z": "+Z"},
        {"source_x": "+Y", "source_y": "+X", "source_z": "+Z"},
    ):
        dialog = _dialog(**axes)
        assert not dialog._axes_keep_handedness(), axes
        dialog.deleteLater()


def test_a_mapping_that_repeats_an_axis_is_still_refused_first() -> None:
    dialog = _dialog(source_x="+X", source_y="+X", source_z="+Z")
    assert not dialog._axes_are_bijective()
    assert not dialog._axes_keep_handedness()
    assert not dialog.ok_btn.isEnabled()
    # A mapping that is not a bijection has no handedness to warn about, so
    # the red line stays empty and the bijection message does the work.
    assert dialog.handedness_warning.text() == ""
    dialog.deleteLater()
