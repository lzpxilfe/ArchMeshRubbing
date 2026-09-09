"""The plate panel: every decision a measured drawing takes, on one page.

The composer accepts some forty decisions, and until now the window knew
thirteen of them.  Everything else - the fold's steps round a motif, the
presumed lines, the pasted painting, the far silhouette, the corner styles,
the strokes struck out - could only be set by writing Python.  That is why
every plate of the last weeks was made by a script.

This panel is those decisions, laid out the way the work goes rather than
the way the dataclass is ordered: what goes on the sheet, what paper it is
drawn on, how the two halves meet, which inner lines are drawn, what is
pasted, and how far the pen went past what was measured.  It speaks the
plate specification (`src.core.drawing_sheet_spec`) in both directions, so
the panel, the sidecar of a finished plate and a file on disk are the same
document: open a plate's spec, change one number, draw it again.

The panel holds no session and touches no document.  It is given the
records it may offer (`set_records`), it returns a spec (`spec()`), and it
says when the archaeologist changed something (`specChanged`).  What that
spec means is the composer's business, and the composer refuses what it
cannot draw; this panel's job is to make every decision reachable and to
show, at a glance, what the plate will be.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.core.drawing_sheet import (
    BREAK_REACHES,
    BREAK_STYLES,
    CENTER_AXIS_STYLES,
    CONVENTIONAL_SCALE_DENOMINATORS,
    LINE_CAPS,
    MIRROR_ELEVATION_SIDES,
    PAINT_CUTOUT_PLACEMENTS,
    PRESUMED_FLOOR,
    PRESUMED_KINDS,
    REACHES,
    RUBBING_ON_AXIS_FITS,
    RUBBING_ON_AXIS_TRIMS,
    SHERD_SIDES,
    PAGE_SIZES_MM,
)
from src.core.drawing_sheet_spec import (
    PLATE_SPEC_FORMAT,
    PLATE_SPEC_SCHEMA_VERSION,
    REACH_TO_WALL,
    PlateSpecError,
    plate_spec_options,
)

#: What each choice is called on the page.  The composer's words are English
#: keys; the archaeologist reads Korean.
_WORDS: dict[str, str] = {
    "axis": "축에서 멈춤",
    "section": "단면 앞까지",
    "far": "뒷면 실루엣",
    "left": "왼쪽",
    "right": "오른쪽",
    "solid": "실선",
    "dash_dot": "일점쇄선",
    "round": "둥글게",
    "butt": "자르기",
    "square": "네모",
    "broken_once": "한 번 끊기",
    "broken_twice": "두 번 끊기",
    "omit": "생략",
    "in_place": "제자리에",
    "below": "도면 아래",
    "paper": "종이 그대로",
    "axis_height": "높이에 맞춰",
    "none": "자르지 않음",
    "rectangle": "직사각형으로",
    "floor": "바닥",
    "wall_on": "안벽 연장",
    "portrait": "세로",
    "landscape": "가로",
}

#: The rows a table starts with, so an empty table is still a table.
_TABLE_MIN_ROWS = 1


def _word(value: str) -> str:
    return _WORDS.get(value, value)


def _combo(values: Sequence[str], *, tip: str = "") -> QComboBox:
    combo = QComboBox()
    for value in values:
        combo.addItem(_word(value), value)
    if tip:
        combo.setToolTip(tip)
    return combo


def _spin(low: float, high: float, value: float, *, step: float = 1.0, decimals: int = 1, suffix: str = "") -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setValue(value)
    if suffix:
        spin.setSuffix(suffix)
    return spin


def _table(headers: Sequence[str], *, tip: str = "") -> QTableWidget:
    table = QTableWidget(_TABLE_MIN_ROWS, len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    rows_header = table.verticalHeader()
    if rows_header is not None:
        rows_header.setVisible(False)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setMaximumHeight(150)
    header = table.horizontalHeader()
    if header is not None:
        for column in range(len(headers)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
    if tip:
        table.setToolTip(tip)
    return table


def _cell_text(table: QTableWidget, row: int, column: int) -> str:
    widget = table.cellWidget(row, column)
    if isinstance(widget, QComboBox):
        return str(widget.currentData())
    item = table.item(row, column)
    return "" if item is None else item.text().strip()


class PlatePanel(QWidget):
    """Everything the plate composer can be told, as one panel.

    ``specChanged`` fires whenever a decision changes; ``plateRequested``
    when the archaeologist asks for the plate.  ``loadRequested`` and
    ``saveRequested`` are the spec file, which the window turns into a file
    dialogue - the panel does not touch the disk.
    """

    specChanged = pyqtSignal()
    plateRequested = pyqtSignal()
    loadRequested = pyqtSignal()
    saveRequested = pyqtSignal()

    #: Decisions this panel does not edit, because the window chooses them
    #: elsewhere (the 상태·기법 chooser).  A spec that carries them keeps
    #: them: a panel that silently dropped a decision it was handed would
    #: lose work nobody asked it to lose.
    CARRIED_KEYS: tuple[str, ...] = (
        "condition_records",
        "technique_records",
        "technique_angles_deg",
        "technique_representations",
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._records: list[tuple[str, str, str]] = []
        self._carried: dict[str, Any] = {}
        # Which table columns name a record, and which hold a closed choice.
        # Both are re-applied whenever a table grows, so a row added by a
        # loaded spec gets the same control as the rows built with the panel.
        self._record_columns: list[tuple[QTableWidget, int]] = []
        self._kind_columns: list[tuple[QTableWidget, int, tuple[str, ...], str]] = []
        self._build()

    # --- building ---------------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        heading = QLabel("실측 도판")
        heading.setStyleSheet("font-weight: bold; font-size: 13px;")
        outer.addWidget(heading)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._sheet_tab(), "도면")
        self.tabs.addTab(self._paper_tab(), "종이")
        self.tabs.addTab(self._halves_tab(), "반쪽")
        self.tabs.addTab(self._inner_tab(), "내선")
        self.tabs.addTab(self._motif_tab(), "문양")
        self.tabs.addTab(self._reading_tab(), "해석")
        outer.addWidget(self.tabs, 1)

        # What the plate will be, in one line, always visible: the panel has
        # six tabs and the reader must not have to open them to know.
        self.summary = QLabel("고른 기록이 없습니다")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #4a5568; font-size: 11px;")
        outer.addWidget(self.summary)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        self.btn_load = QPushButton("명세 불러오기")
        self.btn_load.setToolTip("도판 명세(.json)를 읽어 이 패널을 그대로 채웁니다.")
        self.btn_load.clicked.connect(self.loadRequested.emit)
        self.btn_save = QPushButton("명세 저장")
        self.btn_save.setToolTip(
            "지금 패널의 결정을 도판 명세로 저장합니다.\n"
            "같은 명세로 명령줄에서도 같은 도판이 나옵니다."
        )
        self.btn_save.clicked.connect(self.saveRequested.emit)
        self.btn_plate = QPushButton("도판 만들기")
        self.btn_plate.setStyleSheet("font-weight: bold;")
        self.btn_plate.clicked.connect(self.plateRequested.emit)
        buttons.addWidget(self.btn_load)
        buttons.addWidget(self.btn_save)
        buttons.addStretch()
        buttons.addWidget(self.btn_plate)
        outer.addLayout(buttons)

    def _sheet_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 6, 4, 4)

        layout.addWidget(QLabel("도판에 올릴 기록 (체크한 순서가 배치 순서)"))
        self.list_records = QListWidget()
        self.list_records.setToolTip(
            "READY + FRESH 기록만 오릅니다.  체크한 순서대로 왼쪽에서 오른쪽으로 놓이고,\n"
            "한 줄이 차면 다음 줄로 넘어갑니다.  미러 도형의 단면 기록은 여기 체크하지 않습니다."
        )
        self.list_records.itemChanged.connect(self._changed)
        layout.addWidget(self.list_records, 1)

        form = QFormLayout()
        self.edit_label = QLineEdit()
        self.edit_label.setPlaceholderText("유물명 (제목란에 인쇄됩니다)")
        self.edit_label.setMaxLength(120)
        self.edit_label.textChanged.connect(self._changed)
        form.addRow("유물", self.edit_label)

        self.combo_layout = QComboBox()
        for value, label in (
            ("rows", "줄 배치 (기본)"),
            ("mirror", "좌 반입면 · 우 반단면"),
            ("plan_over_elevation", "평면 위 · 입면 아래"),
            ("plan_with_sections", "평면 + 아래 횡단면 + 오른쪽 종단면"),
        ):
            self.combo_layout.addItem(label, value)
        self.combo_layout.setToolTip(
            "도형을 어떻게 놓을지.  미러는 아래 '반쪽' 탭에서 단면 기록을 짝지어야 하고,\n"
            "나머지 두 배치는 그 배치가 요구하는 기록만 그립니다."
        )
        self.combo_layout.currentIndexChanged.connect(self._changed)
        form.addRow("배치", self.combo_layout)

        self.check_center_axis = QCheckBox("중심축선 그리기")
        self.check_center_axis.setToolTip(
            "회전축으로 정치한 경우에만 그립니다.  수동 정치에서는 켜도 그리지 않습니다."
        )
        self.check_center_axis.toggled.connect(self._changed)
        form.addRow("", self.check_center_axis)
        layout.addLayout(form)

        layout.addWidget(QLabel("제목란에 더 넣을 줄 (최대 6줄)"))
        self.table_title_rows = _table(
            ("이름", "값"),
            tip="제목란에 인쇄할 줄.  축척·문서 해시·해석·추정은 도판이 스스로 넣습니다.",
        )
        self.table_title_rows.setRowCount(3)
        self.table_title_rows.itemChanged.connect(self._changed)
        layout.addWidget(self.table_title_rows)

        layout.addWidget(QLabel("파편 — 깨진 자리 (도형 record, 어느 쪽)"))
        self.table_sherd = _table(
            ("도형 record", "깨진 쪽"),
            tip=(
                "그 쪽은 만든 사람이 낸 가장자리가 아니라 깨진 자리입니다.\n"
                "그 앞 1.5 mm에서 선을 멈추고, 가로질러 아무것도 긋지 않습니다.\n"
                "제목란에 '파편' 줄이 함께 인쇄됩니다."
            ),
        )
        self._fill_kind_column(
            self.table_sherd, column=1, values=SHERD_SIDES, default=SHERD_SIDES[0]
        )
        self._fill_record_column(self.table_sherd, column=0)
        self.table_sherd.itemChanged.connect(self._changed)
        layout.addWidget(self.table_sherd)

        layout.addWidget(QLabel("단면 위치 표시 (단면 record, 그 단면을 뜬 도형 record)"))
        self.table_section_marks = _table(
            ("단면 record", "표시할 도형 record"),
            tip=(
                "그 도형 위에 절단면의 자취를 일점쇄선으로 긋고 양끝에 A-A′를 붙입니다.\n"
                "도형 평면이 절단면과 나란하면 자취가 없으므로 도판이 거부합니다."
            ),
        )
        for column in (0, 1):
            self._fill_record_column(self.table_section_marks, column=column)
        self.table_section_marks.itemChanged.connect(self._changed)
        layout.addWidget(self.table_section_marks)
        return page

    def _paper_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(4, 6, 4, 4)

        self.combo_page = QComboBox()
        # The data is one string, not a tuple: Qt compares item data as
        # variants, and a tuple never matches itself in ``findData``.
        for size in sorted(PAGE_SIZES_MM):
            for orientation in ("portrait", "landscape"):
                self.combo_page.addItem(f"{size} {_word(orientation)}", f"{size}|{orientation}")
        default = self.combo_page.findData("A4|portrait")
        if default >= 0:
            self.combo_page.setCurrentIndex(default)
        self.combo_page.currentIndexChanged.connect(self._changed)
        form.addRow("용지", self.combo_page)

        # The scale is two widgets: the list a report actually uses, and the
        # box for the case it does not cover.  Picking from the list sets the
        # box; typing in the box moves the list to 직접 입력.  The spec only
        # ever carries the number, so the pair is a convenience and never a
        # second source of truth.
        self.combo_scale = QComboBox()
        for denominator in CONVENTIONAL_SCALE_DENOMINATORS:
            self.combo_scale.addItem(f"1 : {denominator:g}", float(denominator))
        self.combo_scale.addItem("직접 입력", None)
        self.combo_scale.setToolTip(
            "보고서가 보통 쓰는 축척입니다.  고르면 아래 칸이 따라옵니다.\n"
            "목록에 없는 값이 필요하면 직접 입력을 고르고 칸에 적으세요."
        )
        form.addRow("축척", self.combo_scale)

        self.spin_scale = QSpinBox()
        self.spin_scale.setRange(1, 1000)
        self.spin_scale.setValue(1)
        self.spin_scale.setPrefix("1 : ")
        self.spin_scale.setToolTip(
            "선 굵기는 축척과 무관하게 종이 mm 그대로입니다.\n"
            "도판에 들어가지 않으면 쓸 수 있는 축척을 알려주고 거부합니다."
        )
        self.combo_scale.currentIndexChanged.connect(self._scale_template_chosen)
        self.spin_scale.valueChanged.connect(self._sync_scale_template)
        self.spin_scale.valueChanged.connect(self._changed)
        self._sync_scale_template()
        form.addRow("", self.spin_scale)

        self.combo_preset = QComboBox()
        self.combo_preset.setToolTip("선 굵기 preset.  고른 것이 도판 provenance에 그대로 남습니다.")
        self.combo_preset.currentIndexChanged.connect(self._changed)
        form.addRow("선 굵기", self.combo_preset)

        self.spin_margin = _spin(0.0, 100.0, 12.0, decimals=1, suffix=" mm")
        self.spin_margin.valueChanged.connect(self._changed)
        form.addRow("여백", self.spin_margin)

        self.spin_gutter = _spin(0.0, 100.0, 8.0, decimals=1, suffix=" mm")
        self.spin_gutter.setToolTip("도형과 도형 사이의 간격.")
        self.spin_gutter.valueChanged.connect(self._changed)
        form.addRow("도형 간격", self.spin_gutter)

        self.combo_center_axis_style = _combo(
            CENTER_AXIS_STYLES,
            tip="기본은 조금 두꺼운 실선.  일점쇄선은 preset의 중심선 스타일을 씁니다.",
        )
        self.combo_center_axis_style.currentIndexChanged.connect(self._changed)
        form.addRow("중심축선", self.combo_center_axis_style)

        self.combo_line_cap = _combo(
            LINE_CAPS,
            tip="굵은 선이 가는 선과 만날 때 둥근 끝이 삐져나오면 '자르기'로 둡니다.",
        )
        self.combo_line_cap.currentIndexChanged.connect(self._changed)
        form.addRow("선 끝", self.combo_line_cap)

        self.edit_stroke_color = QLineEdit("#111111")
        self.edit_stroke_color.setMaxLength(7)
        self.edit_stroke_color.setToolTip("여섯 자리 16진 색.  기본은 먹빛에 가까운 #111111.")
        self.edit_stroke_color.textChanged.connect(self._changed)
        form.addRow("선 색", self.edit_stroke_color)

        self.edit_title = QLineEdit("ArchMeshRubbing measured drawing sheet")
        self.edit_title.setToolTip("SVG 문서의 제목.  도면 위에는 인쇄되지 않습니다.")
        self.edit_title.textChanged.connect(self._changed)
        form.addRow("SVG 제목", self.edit_title)
        return page

    def _halves_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 6, 4, 4)

        form = QFormLayout()
        self.combo_elevation_side = _combo(
            MIRROR_ELEVATION_SIDES, tip="입면이 어느 쪽에 오는지.  단면은 반대쪽입니다."
        )
        self.combo_elevation_side.currentIndexChanged.connect(self._changed)
        form.addRow("입면 쪽", self.combo_elevation_side)

        self.combo_outline_reach = _combo(
            REACHES,
            tip=(
                "축을 가로지르는 외형선 가장자리를 단면 쪽으로 얼마나 잇는지.\n"
                "'뒷면 실루엣'은 아래 표에 뒷면 record를 짝지어야 합니다."
            ),
        )
        self.combo_outline_reach.currentIndexChanged.connect(self._changed)
        form.addRow("외형선", self.combo_outline_reach)
        layout.addLayout(form)

        layout.addWidget(QLabel("입면 ↔ 단면 · 뒷면 실루엣 짝짓기"))
        self.table_mirror = _table(
            ("입면 record", "단면 record", "뒷면 실루엣 record"),
            tip="한 줄이 한 도형입니다.  뒷면 실루엣은 외형선을 '뒷면 실루엣'으로 둘 때만 씁니다.",
        )
        for column in (0, 1, 2):
            self._fill_record_column(self.table_mirror, column=column)
        self.table_mirror.itemChanged.connect(self._changed)
        layout.addWidget(self.table_mirror)

        layout.addWidget(QLabel("가운데선 꺾기 (계단) — 축 방향 mm, 도달은 mm 또는 '벽'"))
        self.table_jogs = _table(
            ("입면 record", "시작 mm", "끝 mm", "도달"),
            tip=(
                "문양이 축에 걸릴 때 가운데선을 그 문양 둘레로 꺾습니다.\n"
                "도달에 '벽'이라고 쓰면 그 띠에서 단면을 빼고 외형선이 계단의 모서리가 됩니다.\n"
                "계단 모서리가 문양의 먹을 지나가면 도판이 거부합니다."
            ),
        )
        self._fill_record_column(self.table_jogs, column=0)
        self.table_jogs.itemChanged.connect(self._changed)
        layout.addWidget(self.table_jogs)

        layout.addWidget(QLabel("추정선 — 잰 데까지만 그리고, 아는 것은 점선으로"))
        self.table_presumed = _table(
            ("입면 record", "종류", "높이 mm", "길이 mm"),
            tip=(
                "안벽 연장: 잰 안벽의 끊긴 끝에서 그 두께로 바깥벽을 따라 길이만큼 더.\n"
                "바닥: 자를 꽂아 잰 정치면 위 높이.  길이 0이면 벽 앞까지."
            ),
        )
        self._fill_record_column(self.table_presumed, column=0)
        self.table_presumed.itemChanged.connect(self._changed)
        self._fill_kind_column(self.table_presumed, column=1, values=PRESUMED_KINDS, default=PRESUMED_FLOOR)
        layout.addWidget(self.table_presumed)
        return page

    def _inner_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 6, 4, 4)

        layout.addWidget(QLabel("내선으로 얹을 판독 (record id, 한 줄에 하나)"))
        self.table_inner = _table(
            ("능선", "문양", "홈", "꺾임"),
            tip=(
                "능선(석기의 날 능선), 문양(시문선), 홈(침선), 꺾임(굽 경계·모서리).\n"
                "각 칸에 record id를 적습니다.  줄이 모자라면 표가 늘어납니다."
            ),
        )
        self.table_inner.setRowCount(3)
        for column in (0, 1, 2, 3):
            self._fill_record_column(self.table_inner, column=column)
        self.table_inner.itemChanged.connect(self._changed)
        layout.addWidget(self.table_inner)

        form = QFormLayout()
        self.spin_break_solid = QSpinBox()
        self.spin_break_solid.setRange(0, 180)
        self.spin_break_solid.setValue(30)
        self.spin_break_solid.setSuffix(" °")
        self.spin_break_solid.setToolTip(
            "이 각도부터 꺾임을 실선으로 긋습니다.  더 은근한 꺾임은 끊어 긋습니다."
        )
        self.spin_break_solid.valueChanged.connect(self._changed)
        form.addRow("실선 최소 각", self.spin_break_solid)

        self.combo_break_reach = _combo(
            BREAK_REACHES, tip="꺾임 내선을 축에서 멈출지, 단면 앞까지 이을지."
        )
        self.combo_break_reach.currentIndexChanged.connect(self._changed)
        form.addRow("꺾임 내선", self.combo_break_reach)
        layout.addLayout(form)

        layout.addWidget(QLabel("꺾임 하나하나에 대한 실측자의 판단"))
        self.table_break_styles = _table(
            ("꺾임 record", "번호", "어떻게"),
            tip="번호는 아래에서부터 0.  규칙이 정한 것과 달리 보고 싶을 때만 적습니다.",
        )
        self._fill_record_column(self.table_break_styles, column=0)
        self.table_break_styles.itemChanged.connect(self._changed)
        self._fill_kind_column(self.table_break_styles, column=2, values=BREAK_STYLES, default="solid")
        layout.addWidget(self.table_break_styles)

        layout.addWidget(QLabel("문양에서 지울 획 (검수)"))
        self.table_hidden = _table(
            ("문양 record", "패턴 번호"),
            tip="-1은 흩어진 획, -2는 이음매.  나머지는 패턴 번호입니다.",
        )
        self._fill_record_column(self.table_hidden, column=0)
        self.table_hidden.itemChanged.connect(self._changed)
        layout.addWidget(self.table_hidden)
        return page

    def _motif_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 6, 4, 4)

        paint = QGroupBox("채색 따 붙이기")
        paint_layout = QVBoxLayout(paint)
        self.table_cutouts = _table(
            ("채색 record", "붙일 도형", "자리"),
            tip="제자리에: 도형 위 제 위치에.  도면 아래: 굽 안 묵서처럼 다른 뷰의 것.",
        )
        for column in (0, 1):
            self._fill_record_column(self.table_cutouts, column=column)
        self.table_cutouts.itemChanged.connect(self._changed)
        self._fill_kind_column(self.table_cutouts, column=2, values=PAINT_CUTOUT_PLACEMENTS, default="in_place")
        paint_layout.addWidget(self.table_cutouts)
        ink_row = QHBoxLayout()
        ink_row.addWidget(QLabel("먹 농도"))
        self.spin_ink = QSpinBox()
        self.spin_ink.setRange(10, 100)
        self.spin_ink.setValue(70)
        self.spin_ink.setSuffix(" %")
        self.spin_ink.setToolTip("색 그대로 붙인 채색은 이 값과 무관하게 100%로 붙습니다.")
        self.spin_ink.valueChanged.connect(self._changed)
        ink_row.addWidget(self.spin_ink)
        ink_row.addStretch()
        paint_layout.addLayout(ink_row)
        layout.addWidget(paint)

        relief = QGroupBox("양각 점묘")
        relief_layout = QVBoxLayout(relief)
        self.table_stipples = _table(
            ("양각 음영 record", "얹을 도형"),
            tip="양각은 선이 아니라 점입니다.  미러 도형에서는 입면 쪽에만 찍습니다.",
        )
        for column in (0, 1):
            self._fill_record_column(self.table_stipples, column=column)
        self.table_stipples.itemChanged.connect(self._changed)
        relief_layout.addWidget(self.table_stipples)
        dot_row = QHBoxLayout()
        dot_row.addWidget(QLabel("간격"))
        self.spin_pitch = _spin(0.05, 2.0, 0.15, step=0.01, decimals=2, suffix=" mm")
        self.spin_pitch.valueChanged.connect(self._changed)
        dot_row.addWidget(self.spin_pitch)
        dot_row.addWidget(QLabel("지름"))
        self.spin_dot = _spin(0.01, 2.0, 0.12, step=0.01, decimals=2, suffix=" mm")
        self.spin_dot.valueChanged.connect(self._changed)
        dot_row.addWidget(self.spin_dot)
        dot_row.addStretch()
        relief_layout.addLayout(dot_row)
        layout.addWidget(relief)

        rubbing = QGroupBox("탁본 붙이기")
        rubbing_layout = QVBoxLayout(rubbing)
        self.table_rubbings = _table(
            ("탁본 record", "붙일 입면", "어느 면인지"),
            tip="한 변을 중심선에 딱 붙여 붙입니다.  '어느 면인지'는 캡션 앞에 인쇄됩니다.",
        )
        for column in (0, 1):
            self._fill_record_column(self.table_rubbings, column=column)
        self.table_rubbings.itemChanged.connect(self._changed)
        rubbing_layout.addWidget(self.table_rubbings)
        fit_row = QHBoxLayout()
        fit_row.addWidget(QLabel("높이"))
        self.combo_fit = _combo(
            RUBBING_ON_AXIS_FITS,
            tip="종이 그대로는 실제 종이처럼 제 길이로, 높이에 맞춰는 띠마다 제 높이에.",
        )
        self.combo_fit.currentIndexChanged.connect(self._changed)
        fit_row.addWidget(self.combo_fit)
        fit_row.addWidget(QLabel("가위질"))
        self.combo_trim = _combo(RUBBING_ON_AXIS_TRIMS, tip="직사각형으로는 먹이 닿은 가장 큰 사각형만 남깁니다.")
        self.combo_trim.currentIndexChanged.connect(self._changed)
        fit_row.addWidget(self.combo_trim)
        fit_row.addStretch()
        rubbing_layout.addLayout(fit_row)
        layout.addWidget(rubbing)
        layout.addStretch()
        return page

    def _reading_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(4, 6, 4, 4)

        note = QLabel(
            "여기서 무엇이든 켜면 제목란에 '해석' 줄이 인쇄됩니다.\n"
            "읽는 사람이 잰 선과 손댄 선을 종이 위에서 가릴 수 있어야 합니다."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #4a5568; font-size: 11px;")
        form.addRow(note)

        self.spin_smoothing = _spin(0.0, 5.0, 0.0, step=0.1, decimals=1, suffix=" mm")
        self.spin_smoothing.setToolTip("메쉬의 잔떨림을 펜이 따라가지 못하는 만큼 고릅니다.")
        self.spin_smoothing.valueChanged.connect(self._changed)
        form.addRow("선 평활", self.spin_smoothing)

        self.spin_straightening = _spin(0.0, 30.0, 0.0, step=1.0, decimals=0, suffix=" °")
        self.spin_straightening.setToolTip("빗질한 획을 이웃과 나란히 긋습니다.  굽은 획은 그대로 둡니다.")
        self.spin_straightening.valueChanged.connect(self._changed)
        form.addRow("획 직선화", self.spin_straightening)

        self.spin_emphasis = QSpinBox()
        self.spin_emphasis.setRange(0, 100)
        self.spin_emphasis.setSuffix(" %")
        self.spin_emphasis.setToolTip("홈의 두 능선을 잰 것보다 이만큼 더 세웁니다.  골은 움직이지 않습니다.")
        self.spin_emphasis.valueChanged.connect(self._changed)
        form.addRow("홈 능선 강조", self.spin_emphasis)

        self.check_straight_far = QCheckBox("뒷선을 직선으로")
        self.check_straight_far.setToolTip("뒷면 가장자리를 잰 두 끝 사이의 직선 하나로 긋습니다.")
        self.check_straight_far.toggled.connect(self._changed)
        form.addRow("", self.check_straight_far)

        self.edit_note = QLineEdit()
        self.edit_note.setMaxLength(60)
        self.edit_note.setPlaceholderText("무엇을 해석했는지, 실측자의 말로")
        self.edit_note.textChanged.connect(self._changed)
        form.addRow("해석 메모", self.edit_note)
        return page

    def _fill_kind_column(
        self, table: QTableWidget, *, column: int, values: Sequence[str], default: str
    ) -> None:
        """Put a closed choice in one column of every row of ``table``."""

        entry = (table, int(column), tuple(str(value) for value in values), str(default))
        if entry not in self._kind_columns:
            self._kind_columns.append(entry)
        for row in range(table.rowCount()):
            if table.cellWidget(row, column) is None:
                combo = _combo(values)
                index = combo.findData(default)
                if index >= 0:
                    combo.setCurrentIndex(index)
                combo.currentIndexChanged.connect(self._changed)
                table.setCellWidget(row, column, combo)

    def _fill_record_column(self, table: QTableWidget, *, column: int) -> None:
        """Make one column of ``table`` a chooser over the session's records.

        A record is named `record:cutline:<uuid4>` - forty-five characters
        the archaeologist would otherwise have to copy by hand into every
        table that mentions it.  Typing it is not a smaller version of the
        job; it is a different job, and one nobody can check by eye.
        """

        entry = (table, int(column))
        if entry not in self._record_columns:
            self._record_columns.append(entry)
        self._ensure_row_widgets(table)

    def _record_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.setToolTip("이 세션이 가진 기록에서 고릅니다.")
        combo.addItem("", "")
        for record_id, kind, label in self._records:
            combo.addItem(f"{label}  ·  {kind}", record_id)
        combo.currentIndexChanged.connect(self._changed)
        return combo

    @staticmethod
    def _select_record(combo: QComboBox, value: str) -> None:
        """Show ``value`` even when this session does not hold that record.

        A spec written elsewhere - another operator, another machine, the
        same artifact before a re-import - names records this session may
        not have.  Dropping them silently would edit the archaeologist's
        plate; so the id stays, marked as missing.
        """

        wanted = str(value or "")
        index = combo.findData(wanted)
        if index < 0 and wanted:
            combo.addItem(f"{wanted}  ·  이 세션에 없음", wanted)
            index = combo.findData(wanted)
        combo.setCurrentIndex(max(index, 0))

    def _ensure_row_widgets(self, table: QTableWidget) -> None:
        """Give every row of ``table`` the controls its columns call for."""

        for owner, column, values, default in self._kind_columns:
            if owner is not table:
                continue
            for row in range(table.rowCount()):
                if table.cellWidget(row, column) is None:
                    combo = _combo(values)
                    index = combo.findData(default)
                    if index >= 0:
                        combo.setCurrentIndex(index)
                    combo.currentIndexChanged.connect(self._changed)
                    table.setCellWidget(row, column, combo)
        for owner, column in self._record_columns:
            if owner is not table:
                continue
            for row in range(table.rowCount()):
                if table.cellWidget(row, column) is None:
                    table.setCellWidget(row, column, self._record_combo())

    def _refresh_record_columns(self) -> None:
        """Re-offer every record chooser, keeping what each one already holds."""

        for table, column in self._record_columns:
            self._ensure_row_widgets(table)
            for row in range(table.rowCount()):
                widget = table.cellWidget(row, column)
                if not isinstance(widget, QComboBox):
                    continue
                kept = str(widget.currentData() or "")
                widget.blockSignals(True)
                try:
                    widget.clear()
                    widget.addItem("", "")
                    for record_id, kind, label in self._records:
                        widget.addItem(f"{label}  ·  {kind}", record_id)
                    self._select_record(widget, kept)
                finally:
                    widget.blockSignals(False)

    # --- what the window gives it ----------------------------------------

    def set_records(self, records: Sequence[tuple[str, str, str]]) -> None:
        """Offer these records: ``(record id, kind, label)`` triples.

        Checked records survive the refresh, so a list that reloads while the
        archaeologist is choosing does not lose the choice.
        """

        checked = set(self.checked_records())
        self.list_records.blockSignals(True)
        self.list_records.clear()
        self._records = [(str(a), str(b), str(c)) for a, b, c in records]
        for record_id, kind, label in self._records:
            item = QListWidgetItem(f"{label}  ·  {kind}")
            item.setData(Qt.ItemDataRole.UserRole, record_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if record_id in checked else Qt.CheckState.Unchecked
            )
            item.setToolTip(record_id)
            self.list_records.addItem(item)
        self.list_records.blockSignals(False)
        self._refresh_record_columns()
        self._changed()

    def set_presets(self, presets: Sequence[tuple[str, str]], *, current: str = "") -> None:
        """Offer these line presets: ``(preset id, label)`` pairs."""

        self.combo_preset.blockSignals(True)
        self.combo_preset.clear()
        for preset_id, label in presets:
            self.combo_preset.addItem(label, preset_id)
        index = self.combo_preset.findData(current)
        if index >= 0:
            self.combo_preset.setCurrentIndex(index)
        self.combo_preset.blockSignals(False)
        self._changed()

    def _items(self) -> list[QListWidgetItem]:
        rows = (self.list_records.item(row) for row in range(self.list_records.count()))
        return [item for item in rows if item is not None]

    def checked_records(self) -> list[str]:
        return [
            str(item.data(Qt.ItemDataRole.UserRole))
            for item in self._items()
            if item.checkState() == Qt.CheckState.Checked
        ]

    # --- the specification ------------------------------------------------

    def _rows(self, table: QTableWidget, *, columns: int) -> list[list[str]]:
        """The filled rows of a table: every column of a row must be given."""

        rows: list[list[str]] = []
        for row in range(table.rowCount()):
            values = [_cell_text(table, row, column) for column in range(columns)]
            if any(values) and all(values):
                rows.append(values)
        return rows

    #: What a step to the silhouette is called in the table.  The composer's
    #: word is ``wall``; the archaeologist writes 벽, and reads it back.
    WALL_WORD = "벽"

    def _number(self, text: str) -> float | str:
        cleaned = text.strip()
        if cleaned in {REACH_TO_WALL, self.WALL_WORD, "벽까지"}:
            return REACH_TO_WALL
        return float(cleaned)

    def spec(self) -> dict[str, Any]:
        """The plate specification this panel describes.

        Malformed numbers raise `PlateSpecError` naming the table they are
        in, so the window can say where the trouble is instead of showing a
        `ValueError` from three layers down.
        """

        records = self.checked_records()
        size, _, orientation = str(self.combo_page.currentData() or "A4|portrait").partition("|")
        mirror_rows = self._rows(self.table_mirror, columns=2)
        far_rows = [row for row in self._rows(self.table_mirror, columns=3)]
        layout_choice = str(self.combo_layout.currentData())
        spec: dict[str, Any] = {
            "format": PLATE_SPEC_FORMAT,
            "schema_version": PLATE_SPEC_SCHEMA_VERSION,
            "records": records,
            "title_block": {
                "artifact_label": self.edit_label.text().strip() or "이름 없는 유물",
                "rows": self._rows(self.table_title_rows, columns=2),
            },
            "page": {
                "size": str(size),
                "orientation": str(orientation),
                "margin_mm": float(self.spin_margin.value()),
            },
            "scale_denominator": float(self.spin_scale.value()),
            "gutter_mm": float(self.spin_gutter.value()),
            "show_center_axis": bool(self.check_center_axis.isChecked()),
            "center_axis_style": str(self.combo_center_axis_style.currentData()),
            "line_cap": str(self.combo_line_cap.currentData()),
            "stroke_color": self.edit_stroke_color.text().strip() or "#111111",
            "title": self.edit_title.text(),
            "mirror_elevation_side": str(self.combo_elevation_side.currentData()),
            "outline_reach": str(self.combo_outline_reach.currentData()),
            "break_solid_min_deg": int(self.spin_break_solid.value()),
            "break_reach": str(self.combo_break_reach.currentData()),
            "paint_cutout_ink_percent": int(self.spin_ink.value()),
            "stipple_pitch_mm": float(self.spin_pitch.value()),
            "stipple_dot_mm": float(self.spin_dot.value()),
            "rubbing_on_axis_fit": str(self.combo_fit.currentData()),
            "rubbing_on_axis_trim": str(self.combo_trim.currentData()),
            "interpretation": {
                "line_smoothing_mm": float(self.spin_smoothing.value()),
                "stroke_straightening_deg": float(self.spin_straightening.value()),
                "groove_edge_emphasis": float(self.spin_emphasis.value()) / 100.0,
                "straight_far_edges": bool(self.check_straight_far.isChecked()),
                "note": self.edit_note.text().strip(),
            },
        }
        # Each layout is written, the two stacked ones as null unless chosen,
        # so the spec a panel gives back is the spec it was handed.
        spec["mirror_sections"] = (
            [[row[0], row[1]] for row in mirror_rows] if layout_choice == "mirror" else []
        )
        spec["plan_over_elevation"] = (
            records[:2] if layout_choice == "plan_over_elevation" and len(records) >= 2 else None
        )
        spec["plan_with_sections"] = (
            records[:3] if layout_choice == "plan_with_sections" and len(records) >= 3 else None
        )
        # A silhouette that would not be drawn is not named: the composer
        # refuses one, and the third column stays in the table for when the
        # reach goes back to the far half.
        spec["far_silhouettes"] = (
            [[row[0], row[2]] for row in far_rows] if spec["outline_reach"] == "far" else []
        )
        try:
            spec["mirror_jogs"] = [
                [row[0], float(row[1]), float(row[2]), self._number(row[3])]
                for row in self._rows(self.table_jogs, columns=4)
            ]
            spec["presumed_lines"] = [
                [row[0], row[1], float(row[2]), float(row[3])]
                for row in self._rows(self.table_presumed, columns=4)
            ]
            spec["sherd_breaks"] = [
                [row[0], row[1]] for row in self._rows(self.table_sherd, columns=2)
            ]
            spec["section_marks"] = [
                [row[0], row[1]] for row in self._rows(self.table_section_marks, columns=2)
            ]
            spec["break_styles"] = [
                [row[0], int(row[1]), row[2]] for row in self._rows(self.table_break_styles, columns=3)
            ]
            spec["texture_line_hidden_patterns"] = [
                [row[0], int(row[1])] for row in self._rows(self.table_hidden, columns=2)
            ]
        except ValueError as exc:
            raise PlateSpecError(f"표에 숫자가 아닌 값이 있습니다: {exc}") from exc
        spec["paint_cutouts"] = [list(row) for row in self._rows(self.table_cutouts, columns=3)]
        spec["relief_stipples"] = [list(row) for row in self._rows(self.table_stipples, columns=2)]
        rubbings = self._rows(self.table_rubbings, columns=2)
        spec["rubbings_on_axis"] = [[row[0], row[1]] for row in rubbings]
        spec["rubbing_notes"] = [
            [row[0], row[2]] for row in self._rows(self.table_rubbings, columns=3)
        ]
        for key, column in (
            ("crease_records", 0),
            ("texture_line_records", 1),
            ("groove_records", 2),
            ("break_records", 3),
        ):
            spec[key] = [
                _cell_text(self.table_inner, row, column)
                for row in range(self.table_inner.rowCount())
                if _cell_text(self.table_inner, row, column)
            ]
        preset = self.combo_preset.currentData()
        if preset:
            spec["style_preset"] = str(preset)
        spec.update(self._carried)
        return spec

    def set_carried(self, **decisions: Any) -> None:
        """Hold decisions the window makes elsewhere, to put back in the spec."""

        unknown = sorted(set(decisions) - set(self.CARRIED_KEYS))
        if unknown:
            raise PlateSpecError(f"이 패널이 옮겨 담을 수 없는 항목입니다: {', '.join(unknown)}")
        self._carried.update(decisions)
        self._changed()

    def options(self) -> tuple[list[str], Any]:
        """The records and `DrawingSheetOptions` this panel asks for."""

        return plate_spec_options(self.spec())

    # --- reading a specification back in ---------------------------------

    def _set_table(self, table: QTableWidget, rows: Sequence[Sequence[Any]], *, columns: int) -> None:
        table.blockSignals(True)
        table.setRowCount(max(len(rows) + 1, _TABLE_MIN_ROWS))
        # A grown table's new rows have no controls yet; without this the
        # choices a column stands for become free text on exactly the rows a
        # loaded spec added.
        self._ensure_row_widgets(table)
        for row in range(table.rowCount()):
            for column in range(columns):
                value = ""
                if row < len(rows) and column < len(rows[row]):
                    raw = rows[row][column]
                    value = raw if isinstance(raw, str) else f"{raw:g}" if isinstance(raw, float) else str(raw)
                widget = table.cellWidget(row, column)
                if isinstance(widget, QComboBox):
                    if (table, column) in self._record_columns:
                        self._select_record(widget, value)
                    else:
                        index = widget.findData(value)
                        if index >= 0:
                            widget.setCurrentIndex(index)
                else:
                    table.setItem(row, column, QTableWidgetItem(value))
        table.blockSignals(False)

    def set_spec(self, spec: Mapping[str, Any]) -> None:
        """Fill the panel from a plate specification, refusing a bad one first."""

        plate_spec_options(spec)
        self._carried = {key: spec[key] for key in self.CARRIED_KEYS if key in spec}
        block = spec.get("title_block") or {}
        page = spec.get("page") or {}
        interpretation = spec.get("interpretation") or {}
        self.blockSignals(True)
        try:
            self.edit_label.setText(str(block.get("artifact_label", "")))
            self._set_table(self.table_title_rows, list(block.get("rows") or []), columns=2)
            index = self.combo_page.findData(
                f"{page.get('size', 'A4')}|{page.get('orientation', 'portrait')}"
            )
            if index >= 0:
                self.combo_page.setCurrentIndex(index)
            self.spin_margin.setValue(float(page.get("margin_mm", 12.0)))
            self.spin_scale.setValue(int(float(spec.get("scale_denominator", 1.0))))
            self.spin_gutter.setValue(float(spec.get("gutter_mm", 8.0)))
            self.check_center_axis.setChecked(bool(spec.get("show_center_axis", False)))
            self.edit_stroke_color.setText(str(spec.get("stroke_color", "#111111")))
            self.edit_title.setText(str(spec.get("title", self.edit_title.text())))
            for combo, key, fallback in (
                (self.combo_center_axis_style, "center_axis_style", "solid"),
                (self.combo_line_cap, "line_cap", "round"),
                (self.combo_elevation_side, "mirror_elevation_side", "left"),
                (self.combo_outline_reach, "outline_reach", "section"),
                (self.combo_break_reach, "break_reach", "axis"),
                (self.combo_fit, "rubbing_on_axis_fit", "paper"),
                (self.combo_trim, "rubbing_on_axis_trim", "none"),
            ):
                found = combo.findData(str(spec.get(key, fallback)))
                if found >= 0:
                    combo.setCurrentIndex(found)
            self.spin_break_solid.setValue(int(spec.get("break_solid_min_deg", 30)))
            self.spin_ink.setValue(int(spec.get("paint_cutout_ink_percent", 70)))
            self.spin_pitch.setValue(float(spec.get("stipple_pitch_mm", 0.15)))
            self.spin_dot.setValue(float(spec.get("stipple_dot_mm", 0.12)))
            self.spin_smoothing.setValue(float(interpretation.get("line_smoothing_mm", 0.0)))
            self.spin_straightening.setValue(float(interpretation.get("stroke_straightening_deg", 0.0)))
            self.spin_emphasis.setValue(int(round(float(interpretation.get("groove_edge_emphasis", 0.0)) * 100.0)))
            self.check_straight_far.setChecked(bool(interpretation.get("straight_far_edges", False)))
            self.edit_note.setText(str(interpretation.get("note", "")))

            mirror = [list(pair) for pair in spec.get("mirror_sections") or []]
            far = {str(pair[0]): str(pair[1]) for pair in spec.get("far_silhouettes") or []}
            self._set_table(
                self.table_mirror,
                [[pair[0], pair[1], far.get(str(pair[0]), "")] for pair in mirror],
                columns=3,
            )
            self._set_table(
                self.table_jogs,
                [
                    [row[0], row[1], row[2], self.WALL_WORD if row[3] == REACH_TO_WALL else row[3]]
                    for row in spec.get("mirror_jogs") or []
                ],
                columns=4,
            )
            self._set_table(self.table_presumed, spec.get("presumed_lines") or [], columns=4)
            self._fill_kind_column(self.table_presumed, column=1, values=PRESUMED_KINDS, default=PRESUMED_FLOOR)
            self._set_table(self.table_presumed, spec.get("presumed_lines") or [], columns=4)
            self._set_table(self.table_break_styles, spec.get("break_styles") or [], columns=3)
            self._fill_kind_column(self.table_break_styles, column=2, values=BREAK_STYLES, default="solid")
            self._set_table(self.table_break_styles, spec.get("break_styles") or [], columns=3)
            self._set_table(
                self.table_section_marks, spec.get("section_marks") or [], columns=2
            )
            self._set_table(self.table_sherd, spec.get("sherd_breaks") or [], columns=2)
            self._fill_kind_column(
                self.table_sherd, column=1, values=SHERD_SIDES, default=SHERD_SIDES[0]
            )
            self._set_table(self.table_sherd, spec.get("sherd_breaks") or [], columns=2)
            self._set_table(self.table_hidden, spec.get("texture_line_hidden_patterns") or [], columns=2)
            self._set_table(self.table_cutouts, spec.get("paint_cutouts") or [], columns=3)
            self._fill_kind_column(self.table_cutouts, column=2, values=PAINT_CUTOUT_PLACEMENTS, default="in_place")
            self._set_table(self.table_cutouts, spec.get("paint_cutouts") or [], columns=3)
            self._set_table(self.table_stipples, spec.get("relief_stipples") or [], columns=2)
            notes = {str(pair[0]): str(pair[1]) for pair in spec.get("rubbing_notes") or []}
            self._set_table(
                self.table_rubbings,
                [[pair[0], pair[1], notes.get(str(pair[0]), "")] for pair in spec.get("rubbings_on_axis") or []],
                columns=3,
            )
            inner = [
                list(spec.get(key) or [])
                for key in ("crease_records", "texture_line_records", "groove_records", "break_records")
            ]
            depth = max([len(column) for column in inner] + [3])
            self._set_table(
                self.table_inner,
                [[column[row] if row < len(column) else "" for column in inner] for row in range(depth)],
                columns=4,
            )
            layout_choice = (
                "plan_with_sections"
                if spec.get("plan_with_sections")
                else "plan_over_elevation"
                if spec.get("plan_over_elevation")
                else "mirror"
                if mirror
                else "rows"
            )
            found = self.combo_layout.findData(layout_choice)
            if found >= 0:
                self.combo_layout.setCurrentIndex(found)
            wanted = {str(record_id) for record_id in spec.get("records") or []}
            for item in self._items():
                item.setCheckState(
                    Qt.CheckState.Checked
                    if str(item.data(Qt.ItemDataRole.UserRole)) in wanted
                    else Qt.CheckState.Unchecked
                )
            preset = spec.get("style_preset")
            if isinstance(preset, str):
                found = self.combo_preset.findData(preset)
                if found >= 0:
                    self.combo_preset.setCurrentIndex(found)
        finally:
            self.blockSignals(False)
        self._changed()

    # --- the summary ------------------------------------------------------

    def _sync_scale_template(self, *_args: object) -> None:
        """Move the template list to whatever the box now says."""

        index = self.combo_scale.findData(float(self.spin_scale.value()))
        if index < 0:
            index = self.combo_scale.findData(None)
        if index >= 0 and index != self.combo_scale.currentIndex():
            blocked = self.combo_scale.blockSignals(True)
            self.combo_scale.setCurrentIndex(index)
            self.combo_scale.blockSignals(blocked)

    def _scale_template_chosen(self, *_args: object) -> None:
        """A template puts its denominator in the box; 직접 입력 changes nothing."""

        chosen = self.combo_scale.currentData()
        if chosen is None:
            return
        self.spin_scale.setValue(int(round(float(chosen))))

    def _changed(self, *_args: object) -> None:
        self.summary.setText(self._summary_text())
        self.btn_plate.setEnabled(bool(self.checked_records()))
        self.specChanged.emit()

    def _summary_text(self) -> str:
        try:
            spec = self.spec()
        except PlateSpecError as exc:
            return str(exc)
        records = spec["records"]
        if not records:
            return "고른 기록이 없습니다"
        page = spec["page"]
        parts = [
            f"{len(records)}개 기록",
            f"{page['size']} {_word(page['orientation'])}",
            f"1:{spec['scale_denominator']:g}",
        ]
        if spec.get("mirror_sections"):
            parts.append(f"미러 {len(spec['mirror_sections'])}쌍")
        if spec.get("mirror_jogs"):
            walls = sum(1 for jog in spec["mirror_jogs"] if jog[3] == REACH_TO_WALL)
            step = f"계단 {len(spec['mirror_jogs'])}단"
            parts.append(step + (f" (벽까지 {walls})" if walls else ""))
        if spec.get("presumed_lines"):
            parts.append(f"추정선 {len(spec['presumed_lines'])}")
        drawn = sum(len(spec.get(key) or []) for key in ("crease_records", "texture_line_records", "groove_records", "break_records"))
        if drawn:
            parts.append(f"내선 {drawn}")
        if spec.get("paint_cutouts"):
            parts.append(f"채색 {len(spec['paint_cutouts'])}")
        if spec.get("relief_stipples"):
            parts.append(f"점묘 {len(spec['relief_stipples'])}")
        if spec.get("rubbings_on_axis"):
            parts.append(f"탁본 {len(spec['rubbings_on_axis'])}")
        interpretation = spec["interpretation"]
        if any(
            [
                interpretation["line_smoothing_mm"] > 0.0,
                interpretation["stroke_straightening_deg"] > 0.0,
                interpretation["groove_edge_emphasis"] > 0.0,
                interpretation["straight_far_edges"],
                interpretation["note"],
            ]
        ):
            parts.append("해석 있음")
        return " · ".join(parts)


__all__ = ["PlatePanel"]
