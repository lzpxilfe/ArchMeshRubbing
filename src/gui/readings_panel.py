"""The readings panel: make what the plate can draw.

The plate draws a corner, a groove, a ridge, a far silhouette, the shade
of a relief, the lines of a pattern and the paint cut out of the colour.
This is where they are made.  Each reading has its own
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
    PAINT_CUTOUT,
    PROFILE_BREAK,
    PROFILE_GROOVE,
    READING_KINDS,
    READING_LABELS,
    RELIEF_SHADE,
    TEXTURE_LINES,
    TEXTURE_READINGS,
)
from src.core.artifact_profile_break import (
    PROFILE_BREAK_SURFACE_INWARD,
    PROFILE_BREAK_SURFACE_OUTWARD,
)
from src.core.artifact_texture_lines import (
    TEXTURE_LINES_DOMAIN_AXIS,
    TEXTURE_LINES_DOMAIN_VIEW,
    TEXTURE_LINES_RIDGE_RULE,
    TEXTURE_LINES_STROKE_RULE,
    TEXTURE_LINES_VALLEY_RULE,
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
    #: Which file the archaeologist wants to pick: "atlas", "normal" or
    #: "colour".  The window owns the dialogue; this panel owns the path.
    fileRequested = pyqtSignal(str)

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
        outer.addWidget(self._texture_group())

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

    def _texture_group(self) -> QGroupBox:
        """The two readings that read the scanner's images, not the mesh.

        The pattern is in the texture and nowhere else in the file, so
        these need the OBJ's texture coordinates and one map beside them.
        A normal map gives the incised or raised lines; a colour map gives
        the painted ones, and is the only thing a cutout can be read from.
        """

        self.group_texture = QGroupBox("텍스처에서 읽기")
        form = QFormLayout(self.group_texture)

        self.edit_atlas_path = QLineEdit()
        self.edit_atlas_path.setPlaceholderText("텍스처 좌표가 있는 OBJ")
        self.btn_atlas_path = QPushButton("OBJ 고르기")
        self.btn_atlas_path.clicked.connect(lambda: self.fileRequested.emit("atlas"))
        form.addRow(self.edit_atlas_path, self.btn_atlas_path)

        self.combo_texture_map = QComboBox()
        self.combo_texture_map.addItem("법선 지도 (음각·양각)", "normal")
        self.combo_texture_map.addItem("색 지도 (채색)", "colour")
        self.combo_texture_map.setToolTip(
            "법선 지도는 파이거나 솟은 선을, 색 지도는 칠해진 선을 읽습니다."
        )
        self.combo_texture_map.currentIndexChanged.connect(self._kind_changed)
        form.addRow("무엇에서", self.combo_texture_map)

        self.edit_map_path = QLineEdit()
        self.edit_map_path.setPlaceholderText("지도 이미지 (PNG 등)")
        self.btn_map_path = QPushButton("이미지 고르기")
        self.btn_map_path.clicked.connect(
            lambda: self.fileRequested.emit(self.current_map_kind())
        )
        form.addRow(self.edit_map_path, self.btn_map_path)

        self.combo_texture_rule = QComboBox()
        self.combo_texture_rule.addItem("곡률 골 (파인 선)", TEXTURE_LINES_VALLEY_RULE)
        self.combo_texture_rule.addItem("종이 획 (솟은 자리를 덮는 종이)", TEXTURE_LINES_STROKE_RULE)
        self.combo_texture_rule.addItem("획 능선 (조밀한 문양·채색)", TEXTURE_LINES_RIDGE_RULE)
        self.combo_texture_rule.setToolTip(
            "곡률 골: 파인 시문선을 곡률로 읽습니다.\n"
            "종이 획: 탁본의 종이가 읽듯 획 하나를 끝까지 한 줄로 읽습니다.\n"
            "획 능선: 획이 서로 닿을 만큼 조밀할 때, 그리고 채색선을 읽을 때."
        )
        form.addRow("규칙", self.combo_texture_rule)

        self.combo_texture_domain = QComboBox()
        self.combo_texture_domain.addItem("뷰에서", TEXTURE_LINES_DOMAIN_VIEW)
        self.combo_texture_domain.addItem("축 전개에서 (탁본처럼)", TEXTURE_LINES_DOMAIN_AXIS)
        self.combo_texture_domain.setToolTip(
            "뷰에서 읽으면 실루엣 쪽으로 갈수록 문양이 눌립니다.\n"
            "축 전개는 회전축으로 정치한 기물을 펴서 읽으므로, 탁본의 종이가 보는 대로 읽습니다."
        )
        form.addRow("어디서", self.combo_texture_domain)

        self.spin_texture_ppmm = QSpinBox()
        self.spin_texture_ppmm.setRange(1, 200)
        self.spin_texture_ppmm.setValue(20)
        self.spin_texture_ppmm.setSuffix(" px/mm")
        self.spin_texture_ppmm.setToolTip("텍스처를 뷰에 옮길 때의 해상도.  높을수록 느립니다.")
        form.addRow("해상도", self.spin_texture_ppmm)
        return self.group_texture

    def current_map_kind(self) -> str:
        """Which map this reading is being taken from."""

        if self.current_kind() == PAINT_CUTOUT:
            return "colour"
        return str(self.combo_texture_map.currentData() or "normal")

    def set_path(self, which: str, path: str) -> None:
        """Put a path the window chose into the row it belongs to."""

        if str(which) == "atlas":
            self.edit_atlas_path.setText(str(path))
        else:
            self.edit_map_path.setText(str(path))

    def _kind_changed(self, *_args: object) -> None:
        kind = self.current_kind()
        self.about.setText(READING_LABELS[kind][1])
        self.group_break.setVisible(kind == PROFILE_BREAK)
        self.group_groove.setVisible(kind == PROFILE_GROOVE)
        self.group_crease.setVisible(kind == CREASE)
        self.group_view.setVisible(kind in {FAR_SILHOUETTE, RELIEF_SHADE} | TEXTURE_READINGS)
        self.spin_grid.setVisible(kind == FAR_SILHOUETTE)
        self.group_texture.setVisible(kind in TEXTURE_READINGS)
        # A cutout is read from colour and nothing else, so there is no
        # choice to offer for it.
        self.combo_texture_map.setVisible(kind == TEXTURE_LINES)
        self.combo_texture_rule.setVisible(kind == TEXTURE_LINES)
        self.combo_texture_domain.setVisible(kind == TEXTURE_LINES)
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
        if kind in TEXTURE_READINGS:
            options: dict[str, Any] = {
                "view": str(self.combo_view.currentData()),
                "atlas_path": self.edit_atlas_path.text().strip(),
                "pixels_per_mm": int(self.spin_texture_ppmm.value()),
            }
            path = self.edit_map_path.text().strip()
            if self.current_map_kind() == "colour":
                options["colour_map_path"] = path
            else:
                options["normal_map_path"] = path
            if kind == TEXTURE_LINES:
                options["rule"] = str(self.combo_texture_rule.currentData())
                options["domain"] = str(self.combo_texture_domain.currentData())
            return options
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
        if "pixels_per_mm" in options:
            self.spin_texture_ppmm.setValue(int(options["pixels_per_mm"]))
        if "atlas_path" in options:
            self.edit_atlas_path.setText(str(options["atlas_path"]))
        for key, combo in (
            ("rule", self.combo_texture_rule),
            ("domain", self.combo_texture_domain),
        ):
            if key in options:
                index = combo.findData(str(options[key]))
                if index >= 0:
                    combo.setCurrentIndex(index)
        for key, which in (("normal_map_path", "normal"), ("colour_map_path", "colour")):
            if key in options:
                self.edit_map_path.setText(str(options[key]))
                index = self.combo_texture_map.findData(which)
                if index >= 0:
                    self.combo_texture_map.setCurrentIndex(index)
        if "view" in options:
            index = self.combo_view.findData(str(options["view"]))
            if index >= 0:
                self.combo_view.setCurrentIndex(index)

    def _take(self) -> None:
        self.readingRequested.emit(self.current_kind(), self.record_id(), self.options())


__all__ = ["ReadingsPanel"]
