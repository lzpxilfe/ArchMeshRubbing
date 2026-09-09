"""The readings panel: make what the plate can draw.

The plate draws a corner, a groove, a ridge, a far silhouette and the
shade of a relief.  This is where they are made.  Each reading has its own
few numbers - the least angle a corner must turn, how deep a groove is a
groove - and each is taken on the artifact as the active Align stands it,
then recorded under the archaeologist's name.

The panel decides nothing about a reading.  It collects the numbers, asks
for a record id, and hands both to `src.application.artifact_readings`;
what a reading means, and when it refuses, is the core's.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.application.artifact_readings import (
    CREASE,
    FAR_SILHOUETTE,
    PROFILE_BREAK,
    PROFILE_GROOVE,
    READING_KINDS,
    READING_LABELS,
    RELIEF_SHADE,
)
from src.core.artifact_profile_break import (
    PROFILE_BREAK_SURFACE_INWARD,
    PROFILE_BREAK_SURFACE_OUTWARD,
)

#: The six views a silhouette or a shade may be read in.
_VIEWS = ("front", "back", "left", "right", "top", "bottom")
_VIEW_WORDS = {
    "front": "정면",
    "back": "배면",
    "left": "좌측",
    "right": "우측",
    "top": "평면",
    "bottom": "저면",
}


class ReadingsPanel(QWidget):
    """Choose a reading, set its few numbers, and take it.

    ``readingRequested`` carries ``(kind, record id, options)``; the window
    holds the session and does the taking, so this widget never touches a
    document.
    """

    readingRequested = pyqtSignal(str, str, dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        heading = QLabel("판독 — 도판이 그릴 수 있는 것을 읽는다")
        heading.setStyleSheet("font-weight: bold; font-size: 13px;")
        outer.addWidget(heading)

        self.combo_kind = QComboBox()
        for kind in READING_KINDS:
            self.combo_kind.addItem(READING_LABELS[kind][0], kind)
        self.combo_kind.currentIndexChanged.connect(self._kind_changed)
        form = QFormLayout()
        form.addRow("판독", self.combo_kind)
        self.edit_record_id = QLineEdit()
        self.edit_record_id.setPlaceholderText("record id (예: record:breaks)")
        form.addRow("기록 id", self.edit_record_id)
        outer.addLayout(form)

        self.about = QLabel()
        self.about.setWordWrap(True)
        self.about.setStyleSheet("color: #4a5568; font-size: 11px;")
        outer.addWidget(self.about)

        outer.addWidget(self._break_group())
        outer.addWidget(self._groove_group())
        outer.addWidget(self._crease_group())
        outer.addWidget(self._view_group())

        self.btn_take = QPushButton("이 판독 읽고 기록하기")
        self.btn_take.setStyleSheet("font-weight: bold;")
        self.btn_take.clicked.connect(self._take)
        outer.addWidget(self.btn_take)
        outer.addStretch()
        self._kind_changed()

    def _break_group(self) -> QGroupBox:
        self.group_break = QGroupBox("단면 꺾임")
        form = QFormLayout(self.group_break)
        self.spin_break_angle = QSpinBox()
        self.spin_break_angle.setRange(1, 179)
        self.spin_break_angle.setValue(30)
        self.spin_break_angle.setSuffix(" °")
        self.spin_break_angle.setToolTip("이 각도부터 꺾임으로 봅니다.  낮출수록 은근한 꺾임까지 잡습니다.")
        form.addRow("최소 각", self.spin_break_angle)
        self.combo_break_surface = QComboBox()
        self.combo_break_surface.addItem("바깥면", PROFILE_BREAK_SURFACE_OUTWARD)
        self.combo_break_surface.addItem("안쪽면", PROFILE_BREAK_SURFACE_INWARD)
        self.combo_break_surface.setToolTip("어느 면의 옆모습을 읽을지.  안쪽은 스캔이 닿은 곳까지만 읽힙니다.")
        form.addRow("면", self.combo_break_surface)
        return self.group_break

    def _groove_group(self) -> QGroupBox:
        self.group_groove = QGroupBox("홈 (침선)")
        form = QFormLayout(self.group_groove)
        self.spin_groove_depth = QSpinBox()
        self.spin_groove_depth.setRange(10, 5000)
        self.spin_groove_depth.setValue(150)
        self.spin_groove_depth.setSuffix(" µm")
        self.spin_groove_depth.setToolTip("이보다 얕은 골은 홈으로 보지 않습니다.")
        form.addRow("최소 깊이", self.spin_groove_depth)
        self.spin_groove_width = QSpinBox()
        self.spin_groove_width.setRange(100, 60000)
        self.spin_groove_width.setValue(8000)
        self.spin_groove_width.setSuffix(" µm")
        self.spin_groove_width.setToolTip("이보다 넓으면 홈이 아니라 벽의 굴곡입니다.")
        form.addRow("최대 너비", self.spin_groove_width)
        return self.group_groove

    def _crease_group(self) -> QGroupBox:
        self.group_crease = QGroupBox("능선")
        form = QFormLayout(self.group_crease)
        self.spin_crease_angle = QDoubleSpinBox()
        self.spin_crease_angle.setRange(1.0, 179.0)
        self.spin_crease_angle.setValue(35.0)
        self.spin_crease_angle.setSuffix(" °")
        self.spin_crease_angle.setToolTip("두 면이 이만큼 꺾이면 능선입니다.")
        form.addRow("최소 이면각", self.spin_crease_angle)
        self.spin_crease_length = QDoubleSpinBox()
        self.spin_crease_length.setRange(0.1, 500.0)
        self.spin_crease_length.setValue(5.0)
        self.spin_crease_length.setSuffix(" mm")
        self.spin_crease_length.setToolTip("이보다 짧은 능선은 그리지 않습니다.")
        form.addRow("최소 길이", self.spin_crease_length)
        self.spin_crease_link = QDoubleSpinBox()
        self.spin_crease_link.setRange(0.0, 50.0)
        self.spin_crease_link.setValue(0.0)
        self.spin_crease_link.setSuffix(" mm")
        self.spin_crease_link.setToolTip("끊긴 능선의 끝을 이만큼까지 이어 한 능선으로 봅니다.  0이면 잇지 않습니다.")
        form.addRow("끝점 잇기", self.spin_crease_link)
        return self.group_crease

    def _view_group(self) -> QGroupBox:
        self.group_view = QGroupBox("뷰")
        form = QFormLayout(self.group_view)
        self.combo_view = QComboBox()
        for view in _VIEWS:
            self.combo_view.addItem(_VIEW_WORDS[view], view)
        form.addRow("어느 쪽에서", self.combo_view)
        self.spin_grid = QDoubleSpinBox()
        self.spin_grid.setRange(0.05, 5.0)
        self.spin_grid.setSingleStep(0.05)
        self.spin_grid.setDecimals(2)
        self.spin_grid.setValue(0.5)
        self.spin_grid.setSuffix(" mm")
        self.spin_grid.setToolTip("실루엣을 스냅하는 격자.  촘촘할수록 점이 많아집니다.")
        form.addRow("격자", self.spin_grid)
        return self.group_view

    def _kind_changed(self, *_args: object) -> None:
        kind = self.current_kind()
        self.about.setText(READING_LABELS[kind][1])
        self.group_break.setVisible(kind == PROFILE_BREAK)
        self.group_groove.setVisible(kind == PROFILE_GROOVE)
        self.group_crease.setVisible(kind == CREASE)
        self.group_view.setVisible(kind in {FAR_SILHOUETTE, RELIEF_SHADE})
        self.spin_grid.setVisible(kind == FAR_SILHOUETTE)
        if not self.edit_record_id.text().strip():
            self.edit_record_id.setPlaceholderText(f"record id (예: record:{kind})")

    def current_kind(self) -> str:
        return str(self.combo_kind.currentData() or READING_KINDS[0])

    def options(self) -> dict[str, Any]:
        """The numbers this reading was given, in the core's own words."""

        kind = self.current_kind()
        if kind == PROFILE_BREAK:
            return {
                "angle_min_deg": int(self.spin_break_angle.value()),
                "surface": str(self.combo_break_surface.currentData()),
            }
        if kind == PROFILE_GROOVE:
            return {
                "minimum_depth_um": int(self.spin_groove_depth.value()),
                "maximum_width_um": int(self.spin_groove_width.value()),
            }
        if kind == CREASE:
            return {
                "dihedral_min_deg": float(self.spin_crease_angle.value()),
                "min_length_mm": float(self.spin_crease_length.value()),
                "link_mm": float(self.spin_crease_link.value()),
            }
        if kind == FAR_SILHOUETTE:
            return {
                "view": str(self.combo_view.currentData()),
                "precision_grid_mm": float(self.spin_grid.value()),
            }
        return {"view": str(self.combo_view.currentData())}

    def record_id(self) -> str:
        text = self.edit_record_id.text().strip()
        return text or f"record:{self.current_kind()}"

    def set_options(self, options: Mapping[str, Any]) -> None:
        """Put a reading's numbers back on the panel (for a repeat reading)."""

        if "angle_min_deg" in options:
            self.spin_break_angle.setValue(int(options["angle_min_deg"]))
        if "minimum_depth_um" in options:
            self.spin_groove_depth.setValue(int(options["minimum_depth_um"]))
        if "maximum_width_um" in options:
            self.spin_groove_width.setValue(int(options["maximum_width_um"]))
        if "dihedral_min_deg" in options:
            self.spin_crease_angle.setValue(float(options["dihedral_min_deg"]))
        if "min_length_mm" in options:
            self.spin_crease_length.setValue(float(options["min_length_mm"]))
        if "link_mm" in options:
            self.spin_crease_link.setValue(float(options["link_mm"]))
        if "precision_grid_mm" in options:
            self.spin_grid.setValue(float(options["precision_grid_mm"]))
        if "view" in options:
            index = self.combo_view.findData(str(options["view"]))
            if index >= 0:
                self.combo_view.setCurrentIndex(index)

    def _take(self) -> None:
        self.readingRequested.emit(self.current_kind(), self.record_id(), self.options())


__all__ = ["ReadingsPanel"]
