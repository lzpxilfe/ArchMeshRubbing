"""
ArchMeshRubbing v2 - Complete Interactive Application
CloudCompare 스타일 인터랙티브 3D 뷰어 + 펼침 + 표면 선택
"""

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QDockWidget, QTreeWidget,
    QTreeWidgetItem, QGroupBox, QDoubleSpinBox, QFormLayout,
    QSlider, QSpinBox, QStatusBar, QToolBar, QSplitter, QFrame,
    QMessageBox, QTabWidget, QTextEdit, QProgressBar, QComboBox,
    QCheckBox, QScrollArea, QSizePolicy, QButtonGroup
)
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal, QThread
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QFont, QPixmap

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.gui.viewport_3d import Viewport3D
from src.core.mesh_loader import MeshLoader


def get_icon_path():
    """아이콘 경로 반환"""
    icon_path = Path(__file__).parent / "resources" / "icons" / "app_icon.png"
    if icon_path.exists():
        return str(icon_path)
    return None


class HelpWidget(QTextEdit):
    """도움말 위젯"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumHeight(150)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 10px;
                font-size: 11px;
            }
        """)
        self.set_default_help()
    
    def set_default_help(self):
        self.setHtml("""
            <h3 style="margin:0; color:#2c5282;">🎮 조작법</h3>
            <table style="font-size:11px;">
                <tr><td><b>좌클릭 드래그</b></td><td>3D 회전</td></tr>
                <tr><td><b>우클릭 드래그</b></td><td>화면 이동</td></tr>
                <tr><td><b>스크롤</b></td><td>확대/축소</td></tr>
                <tr><td><b>1~6</b></td><td>정면/후면/우측/좌측/상면/하면</td></tr>
                <tr><td><b>R</b></td><td>뷰 초기화</td></tr>
                <tr><td><b>F</b></td><td>메쉬에 맞춤</td></tr>
            </table>
        """)
    
    def set_transform_help(self):
        self.setHtml("""
            <h3 style="margin:0; color:#2c5282;">📐 정치 (Positioning)</h3>
            <p style="font-size:11px;">
            기와를 정확한 위치에 배치합니다.<br>
            <b>이동:</b> X, Y, Z 좌표를 직접 입력<br>
            <b>회전:</b> 각 축 기준 회전 각도 입력<br>
            <b>중심 이동:</b> 메쉬 중심을 원점으로<br>
            <b>바닥 정렬:</b> 메쉬 하단을 Y=0에 맞춤
            </p>
        """)
    
    def set_flatten_help(self):
        self.setHtml("""
            <h3 style="margin:0; color:#2c5282;">🗺️ 펼침 설정</h3>
            <p style="font-size:11px;">
            곡면을 평면으로 펼치는 설정입니다.<br>
            <b>곡률 반경:</b> 기와의 곡률 반경 (mm)<br>
            <b>펼침 방향:</b> 주축 방향 선택<br>
            <b>왜곡 허용:</b> 면적/각도 왜곡 균형<br>
            <b>컷 라인:</b> 토수기와 등 복잡한 형태용
            </p>
        """)
    
    def set_selection_help(self):
        self.setHtml("""
            <h3 style="margin:0; color:#2c5282;">✋ 표면 선택</h3>
            <p style="font-size:11px;">
            내면/외면, 미구 등 영역을 선택합니다.<br>
            <b>Shift+클릭:</b> 면 선택/해제<br>
            <b>브러시:</b> 드래그로 여러 면 선택<br>
            <b>자동 분리:</b> 법선 방향으로 자동 구분<br>
            <b>선택 확장/축소:</b> 인접 면 포함/제외
            </p>
        """)


class TransformPanel(QWidget):
    """메쉬 변환 패널 (이동/회전)"""
    
    transformChanged = pyqtSignal()
    
    def __init__(self, viewport: Viewport3D, help_widget: HelpWidget, parent=None):
        super().__init__(parent)
        self.viewport = viewport
        self.help_widget = help_widget
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # 이동 그룹
        trans_group = QGroupBox("📍 이동 (cm)")
        trans_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        trans_layout = QFormLayout(trans_group)
        
        self.trans_x = self._create_spinbox(-1000, 1000, 2)
        self.trans_y = self._create_spinbox(-1000, 1000, 2)
        self.trans_z = self._create_spinbox(-1000, 1000, 2)
        
        trans_layout.addRow("X:", self.trans_x)
        trans_layout.addRow("Y:", self.trans_y)
        trans_layout.addRow("Z:", self.trans_z)
        layout.addWidget(trans_group)
        
        # 회전 그룹
        rot_group = QGroupBox("🔄 회전 (°)")
        rot_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        rot_layout = QFormLayout(rot_group)
        
        self.rot_x = self._create_spinbox(-180, 180, 1)
        self.rot_y = self._create_spinbox(-180, 180, 1)
        self.rot_z = self._create_spinbox(-180, 180, 1)
        
        rot_layout.addRow("X:", self.rot_x)
        rot_layout.addRow("Y:", self.rot_y)
        rot_layout.addRow("Z:", self.rot_z)
        layout.addWidget(rot_group)
        
        # 스케일 그룹
        scale_group = QGroupBox("📏 스케일")
        scale_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        scale_layout = QFormLayout(scale_group)
        
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(10, 1000)  # 0.1x ~ 10x (10배율로 저장)
        self.scale_slider.setValue(100)  # 1.0x
        self.scale_slider.valueChanged.connect(self.on_scale_changed)
        
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 10.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setDecimals(2)
        self.scale_spin.valueChanged.connect(self.on_scale_spin_changed)
        
        scale_inner = QHBoxLayout()
        scale_inner.addWidget(self.scale_slider, 3)
        scale_inner.addWidget(self.scale_spin, 1)
        scale_layout.addRow("배율:", scale_inner)
        
        layout.addWidget(scale_group)
        
        # 빠른 정렬 버튼
        align_group = QGroupBox("⚡ 빠른 정렬")
        align_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        align_layout = QVBoxLayout(align_group)
        
        btn_center = QPushButton("🎯 중심으로 이동")
        btn_center.clicked.connect(self.center_mesh)
        btn_center.setToolTip("메쉬 중심을 원점(0,0,0)으로 이동")
        align_layout.addWidget(btn_center)
        
        btn_floor = QPushButton("⬇️ 바닥에 정렬")
        btn_floor.clicked.connect(self.align_to_floor)
        btn_floor.setToolTip("메쉬 하단을 Y=0 평면에 맞춤")
        align_layout.addWidget(btn_floor)
        
        btn_reset = QPushButton("🔄 변환 초기화")
        btn_reset.clicked.connect(self.reset_transform)
        btn_reset.setToolTip("모든 변환을 초기값으로 되돌림")
        align_layout.addWidget(btn_reset)
        
        layout.addWidget(align_group)
        layout.addStretch()
    
    def _create_spinbox(self, min_val, max_val, decimals):
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(decimals)
        spin.valueChanged.connect(self.on_transform_changed)
        return spin
    
    def on_transform_changed(self):
        self.viewport.set_mesh_translation(
            self.trans_x.value(),
            self.trans_y.value(),
            self.trans_z.value()
        )
        self.viewport.set_mesh_rotation(
            self.rot_x.value(),
            self.rot_y.value(),
            self.rot_z.value()
        )
        self.transformChanged.emit()
    
    def center_mesh(self):
        if self.viewport.mesh is None:
            return
        center = self.viewport.mesh.centroid
        self.trans_x.setValue(-center[0])
        self.trans_y.setValue(-center[1])
        self.trans_z.setValue(-center[2])
    
    def align_to_floor(self):
        if self.viewport.mesh is None:
            return
        min_y = self.viewport.mesh.bounds[0][1]
        current_y = self.trans_y.value()
        self.trans_y.setValue(current_y - min_y)
    
    def reset_transform(self):
        self.trans_x.setValue(0)
        self.trans_y.setValue(0)
        self.trans_z.setValue(0)
        self.rot_x.setValue(0)
        self.rot_y.setValue(0)
        self.rot_z.setValue(0)
        self.scale_slider.setValue(100)
        self.scale_spin.setValue(1.0)
    
    def on_scale_changed(self, value):
        """슬라이더에서 스케일 변경"""
        scale = value / 100.0
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(scale)
        self.scale_spin.blockSignals(False)
        self.viewport.set_mesh_scale(scale)
    
    def on_scale_spin_changed(self, value):
        """스핀박스에서 스케일 변경"""
        self.scale_slider.blockSignals(True)
        self.scale_slider.setValue(int(value * 100))
        self.scale_slider.blockSignals(False)
        self.viewport.set_mesh_scale(value)
    
    def enterEvent(self, event):
        self.help_widget.set_transform_help()
        super().enterEvent(event)


class FlattenPanel(QWidget):
    """펼침 설정 패널 (Phase B)"""
    
    flattenRequested = pyqtSignal(dict)
    
    def __init__(self, help_widget: HelpWidget, parent=None):
        super().__init__(parent)
        self.help_widget = help_widget
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # 곡률 설정
        curve_group = QGroupBox("📐 곡률 설정")
        curve_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        curve_layout = QFormLayout(curve_group)
        
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(10, 1000)
        self.spin_radius.setValue(150)
        self.spin_radius.setSuffix(" mm")
        self.spin_radius.setToolTip("기와의 곡률 반경 (와통 반경)")
        curve_layout.addRow("곡률 반경:", self.spin_radius)
        
        self.combo_direction = QComboBox()
        self.combo_direction.addItems(["자동 감지", "X축 기준", "Y축 기준", "Z축 기준"])
        self.combo_direction.setToolTip("펼침 시 기준이 되는 주축")
        curve_layout.addRow("펼침 방향:", self.combo_direction)
        
        # 곡률 측정 버튼 추가
        measure_layout = QHBoxLayout()
        self.btn_measure = QPushButton("📏 곡률 측정")
        self.btn_measure.setCheckable(True)
        self.btn_measure.setToolTip("Shift+클릭으로 메쉬 위에 점을 3개 이상 찍으면 곡률을 계산합니다")
        measure_layout.addWidget(self.btn_measure)
        
        self.btn_fit_arc = QPushButton("🔄 원호 피팅")
        self.btn_fit_arc.setToolTip("찍은 점들로 원호를 피팅하고 반지름을 계산합니다")
        measure_layout.addWidget(self.btn_fit_arc)
        
        self.btn_clear_points = QPushButton("🗑️")
        self.btn_clear_points.setToolTip("찍은 점 초기화")
        self.btn_clear_points.setFixedWidth(40)
        measure_layout.addWidget(self.btn_clear_points)
        
        curve_layout.addRow(measure_layout)
        
        layout.addWidget(curve_group)
        
        # 펼침 방법
        method_group = QGroupBox("🗺️ 펼침 방법")
        method_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        method_layout = QVBoxLayout(method_group)
        
        self.combo_method = QComboBox()
        self.combo_method.addItems([
            "ARAP (형태 보존)",
            "LSCM (각도 보존)",
            "면적 보존",
            "원통 펼침"
        ])
        self.combo_method.setToolTip("펼침 알고리즘 선택")
        method_layout.addWidget(self.combo_method)
        
        # 왜곡 허용도
        distort_layout = QHBoxLayout()
        distort_layout.addWidget(QLabel("왜곡 허용:"))
        self.slider_distortion = QSlider(Qt.Orientation.Horizontal)
        self.slider_distortion.setRange(0, 100)
        self.slider_distortion.setValue(50)
        self.slider_distortion.setToolTip("낮음: 면적 보존 우선 / 높음: 각도 보존 우선")
        distort_layout.addWidget(self.slider_distortion)
        self.label_distortion = QLabel("50%")
        self.slider_distortion.valueChanged.connect(
            lambda v: self.label_distortion.setText(f"{v}%")
        )
        distort_layout.addWidget(self.label_distortion)
        method_layout.addLayout(distort_layout)
        
        layout.addWidget(method_group)
        
        # 고급 옵션
        adv_group = QGroupBox("⚙️ 고급 옵션")
        adv_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        adv_layout = QVBoxLayout(adv_group)
        
        self.check_auto_cut = QCheckBox("자동 컷 라인 (토수기와용)")
        self.check_auto_cut.setToolTip("곡률이 크게 변하는 곳에 자동으로 절단선 생성")
        adv_layout.addWidget(self.check_auto_cut)
        
        self.check_multiband = QCheckBox("다중 밴드 펼침")
        self.check_multiband.setToolTip("영역별로 나눠서 펼친 후 병합")
        adv_layout.addWidget(self.check_multiband)
        
        self.spin_iterations = QSpinBox()
        self.spin_iterations.setRange(10, 100)
        self.spin_iterations.setValue(30)
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("반복 횟수:"))
        iter_layout.addWidget(self.spin_iterations)
        adv_layout.addLayout(iter_layout)
        
        layout.addWidget(adv_group)
        
        # 실행 버튼
        self.btn_flatten = QPushButton("🚀 펼침 실행")
        self.btn_flatten.setStyleSheet("""
            QPushButton {
                background-color: #38a169;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2f855a;
            }
        """)
        self.btn_flatten.clicked.connect(self.on_flatten_clicked)
        layout.addWidget(self.btn_flatten)
        
        # 진행 상태
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        layout.addStretch()
    
    def on_flatten_clicked(self):
        options = {
            'radius': self.spin_radius.value(),
            'direction': self.combo_direction.currentText(),
            'method': self.combo_method.currentText(),
            'distortion': self.slider_distortion.value() / 100.0,
            'auto_cut': self.check_auto_cut.isChecked(),
            'multiband': self.check_multiband.isChecked(),
            'iterations': self.spin_iterations.value(),
        }
        self.flattenRequested.emit(options)
    
    def enterEvent(self, event):
        self.help_widget.set_flatten_help()
        super().enterEvent(event)


class SelectionPanel(QWidget):
    """표면/영역 선택 패널 (Phase C)"""
    
    selectionChanged = pyqtSignal(str, object)
    
    def __init__(self, help_widget: HelpWidget, parent=None):
        super().__init__(parent)
        self.help_widget = help_widget
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # 선택 도구
        tool_group = QGroupBox("🖱️ 선택 도구")
        tool_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        tool_layout = QVBoxLayout(tool_group)
        
        # 버튼 그룹 (상호 배타적)
        self.tool_button_group = QButtonGroup(self)
        
        self.btn_click = QPushButton("👆 클릭 선택")
        self.btn_click.setCheckable(True)
        self.btn_click.setChecked(True)
        self.btn_click.setToolTip("Shift+클릭으로 면 선택")
        self.tool_button_group.addButton(self.btn_click, 0)
        tool_layout.addWidget(self.btn_click)
        
        self.btn_brush = QPushButton("🖌️ 브러시 선택")
        self.btn_brush.setCheckable(True)
        self.btn_brush.setToolTip("드래그로 여러 면 선택")
        self.tool_button_group.addButton(self.btn_brush, 1)
        tool_layout.addWidget(self.btn_brush)
        
        # 브러시 크기
        brush_layout = QHBoxLayout()
        brush_layout.addWidget(QLabel("브러시 크기:"))
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 50)
        self.spin_brush.setValue(10)
        self.spin_brush.setSuffix(" mm")
        brush_layout.addWidget(self.spin_brush)
        tool_layout.addLayout(brush_layout)
        
        self.btn_lasso = QPushButton("⭕ 올가미 선택")
        self.btn_lasso.setCheckable(True)
        self.btn_lasso.setToolTip("자유형 영역으로 선택")
        self.tool_button_group.addButton(self.btn_lasso, 2)
        tool_layout.addWidget(self.btn_lasso)
        
        layout.addWidget(tool_group)
        
        # 자동 분리
        auto_group = QGroupBox("🤖 자동 분리")
        auto_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        auto_layout = QVBoxLayout(auto_group)
        
        btn_auto_surface = QPushButton("📊 내면/외면 자동 감지")
        btn_auto_surface.setToolTip("법선 방향으로 내면/외면 자동 분류")
        btn_auto_surface.clicked.connect(lambda: self.selectionChanged.emit('auto_surface', None))
        auto_layout.addWidget(btn_auto_surface)
        
        btn_auto_edge = QPushButton("📏 미구 자동 감지")
        btn_auto_edge.setToolTip("경계 근처 영역 자동 선택")
        btn_auto_edge.clicked.connect(lambda: self.selectionChanged.emit('auto_edge', None))
        auto_layout.addWidget(btn_auto_edge)
        
        layout.addWidget(auto_group)
        
        # 선택 편집
        edit_group = QGroupBox("✏️ 선택 편집")
        edit_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        edit_layout = QVBoxLayout(edit_group)
        
        btn_row = QHBoxLayout()
        btn_grow = QPushButton("➕ 확장")
        btn_grow.setToolTip("선택 영역을 인접 면으로 확장")
        btn_grow.clicked.connect(lambda: self.selectionChanged.emit('grow', None))
        btn_row.addWidget(btn_grow)
        
        btn_shrink = QPushButton("➖ 축소")
        btn_shrink.setToolTip("선택 영역 가장자리 제거")
        btn_shrink.clicked.connect(lambda: self.selectionChanged.emit('shrink', None))
        btn_row.addWidget(btn_shrink)
        edit_layout.addLayout(btn_row)
        
        btn_row2 = QHBoxLayout()
        btn_invert = QPushButton("🔄 반전")
        btn_invert.setToolTip("선택/비선택 반전")
        btn_invert.clicked.connect(lambda: self.selectionChanged.emit('invert', None))
        btn_row2.addWidget(btn_invert)
        
        btn_clear = QPushButton("🗑️ 해제")
        btn_clear.setToolTip("모든 선택 해제")
        btn_clear.clicked.connect(lambda: self.selectionChanged.emit('clear', None))
        btn_row2.addWidget(btn_clear)
        edit_layout.addLayout(btn_row2)
        
        layout.addWidget(edit_group)
        
        # 선택 영역 지정
        assign_group = QGroupBox("🏷️ 영역 지정")
        assign_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        assign_layout = QVBoxLayout(assign_group)
        
        btn_outer = QPushButton("🌞 선택 → 외면")
        btn_outer.setStyleSheet("background-color: #ebf8ff; color: #2b6cb0;")
        btn_outer.clicked.connect(lambda: self.selectionChanged.emit('assign_outer', None))
        assign_layout.addWidget(btn_outer)
        
        btn_inner = QPushButton("🌙 선택 → 내면")
        btn_inner.setStyleSheet("background-color: #faf5ff; color: #6b46c1;")
        btn_inner.clicked.connect(lambda: self.selectionChanged.emit('assign_inner', None))
        assign_layout.addWidget(btn_inner)
        
        btn_migu = QPushButton("📐 선택 → 미구")
        btn_migu.setStyleSheet("background-color: #fffaf0; color: #c05621;")
        btn_migu.clicked.connect(lambda: self.selectionChanged.emit('assign_migu', None))
        assign_layout.addWidget(btn_migu)
        
        layout.addWidget(assign_group)
        
        # 선택 정보
        self.label_selection = QLabel("선택된 면: 0개")
        self.label_selection.setStyleSheet("font-weight: bold; color: #2c5282;")
        layout.addWidget(self.label_selection)
        
        layout.addStretch()
    
    def update_selection_count(self, count: int):
        self.label_selection.setText(f"선택된 면: {count:,}개")
    
    def enterEvent(self, event):
        self.help_widget.set_selection_help()
        super().enterEvent(event)


class PropertiesPanel(QWidget):
    """메쉬 속성 패널"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 파일 정보
        file_group = QGroupBox("📁 파일 정보")
        file_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        file_layout = QFormLayout(file_group)
        
        self.label_filename = QLabel("-")
        self.label_filename.setWordWrap(True)
        file_layout.addRow("파일:", self.label_filename)
        
        layout.addWidget(file_group)
        
        # 메쉬 정보
        mesh_group = QGroupBox("🔷 메쉬 정보")
        mesh_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        mesh_layout = QFormLayout(mesh_group)
        
        self.label_vertices = QLabel("-")
        self.label_faces = QLabel("-")
        self.label_size = QLabel("-")
        self.label_area = QLabel("-")
        self.label_texture = QLabel("-")
        
        mesh_layout.addRow("정점:", self.label_vertices)
        mesh_layout.addRow("면:", self.label_faces)
        mesh_layout.addRow("크기:", self.label_size)
        mesh_layout.addRow("면적:", self.label_area)
        mesh_layout.addRow("텍스처:", self.label_texture)
        
        layout.addWidget(mesh_group)
        
        # 영역 정보
        region_group = QGroupBox("🗂️ 영역 정보")
        region_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        region_layout = QFormLayout(region_group)
        
        self.label_outer = QLabel("-")
        self.label_inner = QLabel("-")
        self.label_migu = QLabel("-")
        
        region_layout.addRow("외면:", self.label_outer)
        region_layout.addRow("내면:", self.label_inner)
        region_layout.addRow("미구:", self.label_migu)
        
        layout.addWidget(region_group)
        layout.addStretch()
    
    def update_mesh_info(self, mesh, filepath=None):
        if mesh is None:
            self.label_filename.setText("-")
            self.label_vertices.setText("-")
            self.label_faces.setText("-")
            self.label_size.setText("-")
            self.label_area.setText("-")
            self.label_texture.setText("-")
            return
        
        if filepath:
            self.label_filename.setText(Path(filepath).name)
        
        self.label_vertices.setText(f"{mesh.n_vertices:,}")
        self.label_faces.setText(f"{mesh.n_faces:,}")
        
        extents = mesh.extents
        self.label_size.setText(f"{extents[0]:.1f} × {extents[1]:.1f} × {extents[2]:.1f} cm")
        self.label_area.setText(f"{mesh.surface_area:.1f} cm²")
        self.label_texture.setText("있음" if mesh.has_texture else "없음")


class ExportPanel(QWidget):
    """내보내기 패널"""
    
    exportRequested = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # 이미지 내보내기
        img_group = QGroupBox("🖼️ 이미지 내보내기")
        img_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        img_layout = QFormLayout(img_group)
        
        self.spin_dpi = QSpinBox()
        self.spin_dpi.setRange(72, 600)
        self.spin_dpi.setValue(300)
        self.spin_dpi.setSuffix(" DPI")
        img_layout.addRow("해상도:", self.spin_dpi)
        
        self.combo_format = QComboBox()
        self.combo_format.addItems(["PNG", "TIFF", "JPEG"])
        img_layout.addRow("포맷:", self.combo_format)
        
        self.check_scale_bar = QCheckBox("스케일 바 포함")
        self.check_scale_bar.setChecked(True)
        img_layout.addRow("", self.check_scale_bar)
        
        layout.addWidget(img_group)
        
        # 버튼
        btn_export_rubbing = QPushButton("📤 탁본 이미지 내보내기")
        btn_export_rubbing.setStyleSheet("""
            QPushButton {
                background-color: #4299e1;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #3182ce; }
        """)
        btn_export_rubbing.clicked.connect(lambda: self.exportRequested.emit({'type': 'rubbing'}))
        layout.addWidget(btn_export_rubbing)
        
        btn_export_ortho = QPushButton("📤 정사투영 내보내기")
        btn_export_ortho.clicked.connect(lambda: self.exportRequested.emit({'type': 'ortho'}))
        layout.addWidget(btn_export_ortho)
        
        # 메쉬 내보내기
        mesh_group = QGroupBox("💾 메쉬 내보내기")
        mesh_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        mesh_layout = QVBoxLayout(mesh_group)
        
        btn_export_outer = QPushButton("외면 메쉬 저장")
        btn_export_outer.clicked.connect(lambda: self.exportRequested.emit({'type': 'mesh_outer'}))
        mesh_layout.addWidget(btn_export_outer)
        
        btn_export_inner = QPushButton("내면 메쉬 저장")
        btn_export_inner.clicked.connect(lambda: self.exportRequested.emit({'type': 'mesh_inner'}))
        mesh_layout.addWidget(btn_export_inner)
        
        btn_export_flat = QPushButton("펼쳐진 메쉬 저장")
        btn_export_flat.clicked.connect(lambda: self.exportRequested.emit({'type': 'mesh_flat'}))
        mesh_layout.addWidget(btn_export_flat)
        
        layout.addWidget(mesh_group)
        layout.addStretch()


class MainWindow(QMainWindow):
    """메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("ArchMeshRubbing v2 - 고고학 메쉬 탁본 도구")
        self.setMinimumSize(1400, 900)
        
        # 드래그 앤 드롭 활성화
        self.setAcceptDrops(True)
        
        # 아이콘 설정
        icon_path = get_icon_path()
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))
        
        self.mesh_loader = MeshLoader(default_unit='cm')
        self.current_mesh = None
        self.current_filepath = None
        
        self.init_ui()
        self.init_menu()
        self.init_toolbar()
        self.init_statusbar()
    
    def init_ui(self):
        # 중앙 위젯
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 왼쪽: 3D 뷰포트
        viewport_container = QWidget()
        viewport_layout = QVBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        
        self.viewport = Viewport3D()
        self.viewport.meshLoaded.connect(self.on_mesh_loaded)
        viewport_layout.addWidget(self.viewport, 1)
        
        # 도움말 위젯
        self.help_widget = HelpWidget()
        viewport_layout.addWidget(self.help_widget)
        
        main_layout.addWidget(viewport_container, 3)
        
        # 오른쪽: 도구 패널들 (탭)
        right_panel = QTabWidget()
        right_panel.setMinimumWidth(320)
        right_panel.setMaximumWidth(400)
        
        # 탭 1: 속성 + 변환
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll1 = QScrollArea()
        scroll1.setWidgetResizable(True)
        scroll1_content = QWidget()
        scroll1_layout = QVBoxLayout(scroll1_content)
        
        self.props_panel = PropertiesPanel()
        scroll1_layout.addWidget(self.props_panel)
        
        self.transform_panel = TransformPanel(self.viewport, self.help_widget)
        scroll1_layout.addWidget(self.transform_panel)
        
        scroll1.setWidget(scroll1_content)
        tab1_layout.addWidget(scroll1)
        
        right_panel.addTab(tab1, "📐 정치")
        
        # 탭 2: 선택
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll2 = QScrollArea()
        scroll2.setWidgetResizable(True)
        
        self.selection_panel = SelectionPanel(self.help_widget)
        self.selection_panel.selectionChanged.connect(self.on_selection_action)
        scroll2.setWidget(self.selection_panel)
        tab2_layout.addWidget(scroll2)
        
        right_panel.addTab(tab2, "✋ 선택")
        
        # 탭 3: 펼침
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        tab3_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll3 = QScrollArea()
        scroll3.setWidgetResizable(True)
        
        self.flatten_panel = FlattenPanel(self.help_widget)
        self.flatten_panel.flattenRequested.connect(self.on_flatten_requested)
        
        # 곡률 측정 버튼 연결
        self.flatten_panel.btn_measure.toggled.connect(self.toggle_curvature_mode)
        self.flatten_panel.btn_fit_arc.clicked.connect(self.fit_curvature_arc)
        self.flatten_panel.btn_clear_points.clicked.connect(self.clear_curvature_points)
        
        scroll3.setWidget(self.flatten_panel)
        tab3_layout.addWidget(scroll3)
        
        right_panel.addTab(tab3, "🗺️ 펼침")
        
        # 탭 4: 내보내기
        tab4 = QWidget()
        tab4_layout = QVBoxLayout(tab4)
        tab4_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll4 = QScrollArea()
        scroll4.setWidgetResizable(True)
        
        self.export_panel = ExportPanel()
        self.export_panel.exportRequested.connect(self.on_export_requested)
        scroll4.setWidget(self.export_panel)
        tab4_layout.addWidget(scroll4)
        
        right_panel.addTab(tab4, "📤 내보내기")
        
        main_layout.addWidget(right_panel, 1)
    
    def init_menu(self):
        menubar = self.menuBar()
        
        # 파일 메뉴
        file_menu = menubar.addMenu("파일(&F)")
        
        action_open = QAction("📂 열기(&O)", self)
        action_open.setShortcut(QKeySequence.StandardKey.Open)
        action_open.triggered.connect(self.open_file)
        file_menu.addAction(action_open)
        
        file_menu.addSeparator()
        
        action_exit = QAction("종료(&X)", self)
        action_exit.setShortcut(QKeySequence.StandardKey.Quit)
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)
        
        # 보기 메뉴
        view_menu = menubar.addMenu("보기(&V)")
        
        action_reset_view = QAction("🔄 뷰 초기화(&R)", self)
        action_reset_view.setShortcut("R")
        action_reset_view.triggered.connect(self.reset_view)
        view_menu.addAction(action_reset_view)
        
        action_fit = QAction("🎯 메쉬에 맞춤(&F)", self)
        action_fit.setShortcut("F")
        action_fit.triggered.connect(self.fit_view)
        view_menu.addAction(action_fit)
        
        view_menu.addSeparator()
        
        # 6방향 뷰
        action_front = QAction("1️⃣ 정면 뷰", self)
        action_front.setShortcut("1")
        action_front.triggered.connect(lambda: self.set_view(0, 0))
        view_menu.addAction(action_front)
        
        action_back = QAction("2️⃣ 후면 뷰", self)
        action_back.setShortcut("2")
        action_back.triggered.connect(lambda: self.set_view(180, 0))
        view_menu.addAction(action_back)
        
        action_right = QAction("3️⃣ 우측면 뷰", self)
        action_right.setShortcut("3")
        action_right.triggered.connect(lambda: self.set_view(90, 0))
        view_menu.addAction(action_right)
        
        action_left = QAction("4️⃣ 좌측면 뷰", self)
        action_left.setShortcut("4")
        action_left.triggered.connect(lambda: self.set_view(-90, 0))
        view_menu.addAction(action_left)
        
        action_top = QAction("5️⃣ 상면 뷰", self)
        action_top.setShortcut("5")
        action_top.triggered.connect(lambda: self.set_view(0, 89))
        view_menu.addAction(action_top)
        
        action_bottom = QAction("6️⃣ 하면 뷰", self)
        action_bottom.setShortcut("6")
        action_bottom.triggered.connect(lambda: self.set_view(0, -89))
        view_menu.addAction(action_bottom)
        
        # 도움말 메뉴
        help_menu = menubar.addMenu("도움말(&H)")
        
        action_about = QAction("ℹ️ 정보(&A)", self)
        action_about.triggered.connect(self.show_about)
        help_menu.addAction(action_about)
    
    def init_toolbar(self):
        toolbar = QToolBar("메인 툴바")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        action_open = QAction("📂 열기", self)
        action_open.triggered.connect(self.open_file)
        toolbar.addAction(action_open)
        
        toolbar.addSeparator()
        
        action_reset = QAction("🔄 뷰 초기화", self)
        action_reset.triggered.connect(self.reset_view)
        toolbar.addAction(action_reset)
        
        action_fit = QAction("🎯 맞춤", self)
        action_fit.triggered.connect(self.fit_view)
        toolbar.addAction(action_fit)
        
        toolbar.addSeparator()
        
        # 6방향 뷰 버튼
        action_front = QAction("정면", self)
        action_front.setToolTip("정면 뷰 (1)")
        action_front.triggered.connect(lambda: self.set_view(0, 0))
        toolbar.addAction(action_front)
        
        action_back = QAction("후면", self)
        action_back.setToolTip("후면 뷰 (2)")
        action_back.triggered.connect(lambda: self.set_view(180, 0))
        toolbar.addAction(action_back)
        
        action_right = QAction("우측", self)
        action_right.setToolTip("우측면 뷰 (3)")
        action_right.triggered.connect(lambda: self.set_view(90, 0))
        toolbar.addAction(action_right)
        
        action_left = QAction("좌측", self)
        action_left.setToolTip("좌측면 뷰 (4)")
        action_left.triggered.connect(lambda: self.set_view(-90, 0))
        toolbar.addAction(action_left)
        
        action_top = QAction("상면", self)
        action_top.setToolTip("상면 뷰 (5)")
        action_top.triggered.connect(lambda: self.set_view(0, 89))
        toolbar.addAction(action_top)
        
        action_bottom = QAction("하면", self)
        action_bottom.setToolTip("하면 뷰 (6)")
        action_bottom.triggered.connect(lambda: self.set_view(0, -89))
        toolbar.addAction(action_bottom)
    
    def init_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        self.status_info = QLabel("📂 파일을 열거나 드래그하세요")
        self.status_mesh = QLabel("") # 메쉬 정보 (정점, 면)
        self.status_grid = QLabel("격자: -")
        self.status_unit = QLabel("단위: cm")
        
        self.statusbar.addWidget(self.status_info, 1)
        self.statusbar.addPermanentWidget(self.status_mesh)
        self.statusbar.addPermanentWidget(self.status_grid)
        self.statusbar.addPermanentWidget(self.status_unit)
    
    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "3D 메쉬 파일 열기",
            "",
            "3D Files (*.obj *.ply *.stl *.off);;All Files (*)"
        )
        
        if filepath:
            self.load_mesh(filepath)
    
    def dragEnterEvent(self, event):
        """드래그 진입 이벤트"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                filepath = urls[0].toLocalFile()
                ext = Path(filepath).suffix.lower()
                if ext in ['.obj', '.ply', '.stl', '.off', '.gltf', '.glb']:
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dropEvent(self, event):
        """드롭 이벤트"""
        urls = event.mimeData().urls()
        if urls:
            filepath = urls[0].toLocalFile()
            self.load_mesh(filepath)
    
    def load_mesh(self, filepath: str):
        try:
            self.status_info.setText(f"⏳ 로딩 중: {Path(filepath).name}")
            self.status_mesh.setText("")
            QApplication.processEvents()
            
            mesh = self.mesh_loader.load(filepath, unit='cm')
            self.current_mesh = mesh
            self.current_filepath = filepath
            
            self.viewport.load_mesh(mesh)
            
            # 상태바 업데이트
            self.status_info.setText(f"✅ 로드됨: {Path(filepath).name}")
            self.status_mesh.setText(f"V: {len(mesh.vertices):,} | F: {len(mesh.faces):,}")
            self.status_grid.setText(f"격자: {self.viewport.grid_spacing}cm")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"파일 로드 실패:\n{e}")
            self.status_info.setText("❌ 로드 실패")
            self.status_mesh.setText("")
    
    def on_mesh_loaded(self, mesh):
        self.props_panel.update_mesh_info(mesh, self.current_filepath)
        self.transform_panel.center_mesh()
    
    def on_selection_action(self, action: str, data):
        self.status_info.setText(f"선택 작업: {action}")
        # TODO: 실제 선택 로직 구현
        
    def on_flatten_requested(self, options: dict):
        self.status_info.setText("펼침 처리 중...")
        QMessageBox.information(self, "펼침", f"펼침 설정:\n{options}")
        # TODO: 실제 펼침 로직 구현
    
    def on_export_requested(self, options: dict):
        export_type = options.get('type', 'rubbing')
        
        if export_type == 'rubbing':
            filepath, _ = QFileDialog.getSaveFileName(
                self, "탁본 이미지 저장", "", "PNG (*.png);;TIFF (*.tiff)"
            )
            if filepath:
                self.status_info.setText(f"내보내기: {filepath}")
                # TODO: 실제 내보내기 구현
    
    def reset_view(self):
        self.viewport.camera.reset()
        self.viewport.update()
    
    def fit_view(self):
        if self.current_mesh is not None:
            self.viewport.camera.fit_to_bounds(self.current_mesh.bounds)
            self.viewport.update()
    
    def set_view(self, azimuth: float, elevation: float):
        self.viewport.camera.azimuth = azimuth
        self.viewport.camera.elevation = elevation
        self.viewport.update()
    
    def toggle_curvature_mode(self, enabled: bool):
        """곡률 측정 모드 토글"""
        self.viewport.curvature_pick_mode = enabled
        if enabled:
            self.status_info.setText("📏 곡률 측정 모드: Shift+클릭으로 메쉬에 점을 찍으세요")
        else:
            self.status_info.setText("📏 곡률 측정 모드 종료")
    
    def fit_curvature_arc(self):
        """찍은 점들로 원호 피팅"""
        if len(self.viewport.picked_points) < 3:
            QMessageBox.warning(self, "경고", "최소 3개의 점이 필요합니다.\nShift+클릭으로 메쉬 위에 점을 찍으세요.")
            return
        
        from src.core.curvature_fitter import CurvatureFitter
        
        fitter = CurvatureFitter()
        arc = fitter.fit_arc(self.viewport.picked_points)
        
        if arc is None:
            QMessageBox.warning(self, "경고", "원호 피팅에 실패했습니다.\n점들이 일직선 위에 있거나 너무 가까울 수 있습니다.")
            return
        
        self.viewport.fitted_arc = arc
        self.viewport.update()
        
        # 펼침 패널의 곡률 반경에 자동 입력 (mm → cm 변환 없이 그대로)
        radius_mm = arc.radius * 10  # cm → mm
        self.flatten_panel.spin_radius.setValue(radius_mm)
        
        self.status_info.setText(f"✅ 원호 피팅 완료: 반지름 = {arc.radius:.2f} cm ({radius_mm:.1f} mm)")
    
    def clear_curvature_points(self):
        """곡률 측정용 점 초기화"""
        self.viewport.clear_curvature_picks()
        self.status_info.setText("🗑️ 측정 점 초기화됨")
    
    def show_about(self):
        icon_path = get_icon_path()
        msg = QMessageBox(self)
        msg.setWindowTitle("ArchMeshRubbing v2")
        
        if icon_path:
            msg.setIconPixmap(QPixmap(icon_path).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio))
        
        msg.setText("""
            <h2>ArchMeshRubbing v2</h2>
            <p>고고학 메쉬 탁본 도구</p>
            <hr>
            <p><b>조작법:</b></p>
            <ul>
                <li>좌클릭 드래그: 회전</li>
                <li>우클릭 드래그: 이동</li>
                <li>스크롤: 확대/축소</li>
                <li>1/3/7: 전면/측면/상단 뷰</li>
            </ul>
        """)
        msg.exec()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 아이콘 설정
    icon_path = get_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
