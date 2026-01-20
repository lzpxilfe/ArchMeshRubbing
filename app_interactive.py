"""
ArchMeshRubbing v2 - Complete Interactive Application
Copyright (C) 2026 balguljang2 (lzpxilfe)
Licensed under the GNU General Public License v2.0 (GPL2)
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
    QCheckBox, QScrollArea, QSizePolicy, QButtonGroup, QDialog
)
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal, QThread
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QFont, QPixmap
import numpy as np
import trimesh

# Add src to path
if getattr(sys, 'frozen', False):
    basedir = sys._MEIPASS
else:
    basedir = str(Path(__file__).parent)
sys.path.insert(0, str(Path(basedir) / 'src'))

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
    
    def set_scene_help(self):
        self.setHtml("""
            <h3 style="margin:0; color:#2c5282;">🌲 씬 트리 (Scene)</h3>
            <p style="font-size:11px;">
            현재 작업 중인 객체 목록입니다.<br>
            <b>클릭:</b> 객체 선택 및 기즈모 활성화<br>
            <b>눈 아이콘:</b> 가시성 토글<br>
            <b>더블클릭:</b> 객체 이름 변경
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


class SplashScreen(QWidget):
    """프로세스 시작 시 보여주는 스플래시 화면"""
    
    def __init__(self):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.SplashScreen | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(500, 300)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 메인 카드 (그림자 효과용)
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 15px;
                border: 1px solid #e0e0e0;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(30, 30, 30, 20)
        
        # 아이콘
        self.icon_label = QLabel()
        icon_path = get_icon_path()
        if icon_path:
            pix = QPixmap(icon_path).scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.icon_label.setPixmap(pix)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.icon_label)
        
        # 타이틀
        title = QLabel("ArchMeshRubbing v2")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #2c5282;
            margin-top: 10px;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title)
        
        # 버전 정보 추가 (사용자 확인용)
        version = QLabel("Version: 2026.01.19.v3")
        version.setStyleSheet("color: #a0aec0; font-size: 10px; margin-bottom: 5px;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(version)
        
        # 서브타이틀
        subtitle = QLabel("고고학용 3D 메쉬 탁본 도구")
        subtitle.setStyleSheet("color: #718096; font-size: 14px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(subtitle)
        
        # 로딩 상태
        self.loading_label = QLabel("Initializing engine...")
        self.loading_label.setStyleSheet("color: #a0aec0; font-size: 11px;")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.loading_label)
        
        # 저작권 정보 (사용자 요청 사항)
        copyright_label = QLabel("© 2026 balguljang2 (github.com/lzpxilfe).")
        copyright_label.setStyleSheet("color: #cbd5e0; font-size: 10px; margin-top: 5px;")
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(copyright_label)
        
        license_label = QLabel("Licensed under GNU GPL v2")
        license_label.setStyleSheet("""
            color: #a0aec0; 
            font-size: 9px; 
            font-weight: bold;
            border-top: 1px solid #f7fafc;
            padding-top: 3px;
        """)
        license_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(license_label)
        
        layout.addWidget(card)
        
    def showMessage(self, message):
        self.loading_label.setText(message)
        QApplication.processEvents()


class UnitSelectionDialog(QDialog):
    """메쉬 로딩 시 단위를 선택하는 다이얼로그"""
    last_index = 0  # 클래스 변수로 마지막 선택 기억
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("단위 선택")
        self.setFixedWidth(280)
        
        layout = QVBoxLayout(self)
        label = QLabel("파일의 원본 단위를 선택하세요:\n(숫자 184.9가 18.49cm가 되려면 mm 선택)")
        label.setStyleSheet("color: #4a5568; font-size: 11px;")
        layout.addWidget(label)
        
        self.combo = QComboBox()
        self.combo.addItems(["Millimeters (mm) -> 1/10 축소", "Centimeters (cm) -> 그대로", "Meters (m) -> 100배 확대"])
        self.combo.setCurrentIndex(UnitSelectionDialog.last_index) 
        layout.addWidget(self.combo)
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("확인")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept_and_save)
        cancel_btn = QPushButton("취소")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)

    def accept_and_save(self):
        UnitSelectionDialog.last_index = self.combo.currentIndex()
        self.accept()

    def get_scale_factor(self):
        idx = self.combo.currentIndex()
        if idx == 0: return 0.1
        if idx == 1: return 1.0
        if idx == 2: return 100.0
        return 1.0


class ScenePanel(QWidget):
    """씬 내의 객체 목록과 부착된 요소를 보여주는 트리 패널"""
    selectionChanged = pyqtSignal(int)
    visibilityChanged = pyqtSignal(int, bool)
    arcDeleted = pyqtSignal(int, int) # object_idx, arc_idx
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["이름", "상태", "값"])
        self.tree.setColumnWidth(1, 40)
        self.tree.setAlternatingRowColors(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)
        
        layout.addWidget(self.tree)
        self.tree.itemClicked.connect(self.on_item_clicked)
    
    def update_list(self, objects, selected_index):
        """객체 및 부착된 원호 리스트 갱신"""
        self.tree.blockSignals(True)
        self.tree.clear()
        for i, obj in enumerate(objects):
            # 메쉬 노드
            mesh_item = QTreeWidgetItem([
                obj.name,
                "👁️" if obj.visible else "👓",
                f"{len(obj.mesh.faces):,}"
            ])
            mesh_item.setData(0, Qt.ItemDataRole.UserRole, ("mesh", i))
            self.tree.addTopLevelItem(mesh_item)
            
            # 부착된 원호들
            for j, arc in enumerate(obj.fitted_arcs):
                arc_item = QTreeWidgetItem(mesh_item)
                arc_item.setText(0, f"원호 #{j+1}")
                arc_item.setText(1, "📏")
                arc_item.setText(2, f"R={arc.radius:.2f}cm") # cm로 표시
                arc_item.setData(0, Qt.ItemDataRole.UserRole, ("arc", i, j))
            
            mesh_item.setExpanded(True)
            if i == selected_index:
                self.tree.setCurrentItem(mesh_item)
        self.tree.blockSignals(False)
                
    def on_item_clicked(self, item, column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data: return
        
        if data[0] == "mesh":
            index = data[1]
            if column == 1: # 가시성 토글
                visible = item.text(1) == "👓"
                item.setText(1, "👁️" if visible else "👓")
                self.visibilityChanged.emit(index, visible)
            else:
                self.selectionChanged.emit(index)

    def show_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if not item: return
        
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data[0] == "arc":
            menu = QMenu(self) # 원인: 부모 위젯 지정
            delete_action = menu.addAction("🗑️ 원호 삭제")
            action = menu.exec(self.tree.mapToGlobal(pos))
            if action == delete_action:
                self.arcDeleted.emit(data[1], data[2])


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
        
        btn_floor = QPushButton("⬇️ 바닥에 자동 정렬")
        btn_floor.clicked.connect(self.align_to_floor)
        btn_floor.setToolTip("메쉬의 가장 낮은 점을 찾아 Y=0 평면에 맞춤")
        align_layout.addWidget(btn_floor)
        
        self.btn_pick_floor = QPushButton("🎯 바닥 지점 직접 클릭")
        self.btn_pick_floor.clicked.connect(self.start_floor_picking)
        self.btn_pick_floor.setToolTip("메쉬 위에서 바닥에 닿을 지점을 직접 클릭하세요")
        align_layout.addWidget(self.btn_pick_floor)
        
        btn_reset = QPushButton("🔄 변환 초기화")
        btn_reset.clicked.connect(self.reset_transform)
        btn_reset.setToolTip("모든 변환을 초기값으로 되돌림")
        align_layout.addWidget(btn_reset)
        
        btn_bake = QPushButton("🔥 회전 적용 (축 재설정)")
        btn_bake.clicked.connect(self.bake_rotation)
        btn_bake.setToolTip("현재 회전을 메쉬에 굽고 회전값을 0으로 리셋")
        btn_bake.setStyleSheet("QPushButton { background-color: #faf0e6; }")
        align_layout.addWidget(btn_bake)
        
        layout.addWidget(align_group)
        layout.addStretch()
    
    def _create_spinbox(self, min_val, max_val, decimals):
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(decimals)
        spin.valueChanged.connect(self.on_transform_changed)
        return spin
    
    def on_transform_changed(self):
        if self.viewport.selected_obj:
            self.viewport.selected_obj.translation = np.array([
                self.trans_x.value(),
                self.trans_y.value(),
                self.trans_z.value()
            ])
            self.viewport.selected_obj.rotation = np.array([
                self.rot_x.value(),
                self.rot_y.value(),
                self.rot_z.value()
            ])
            self.viewport.update()
            self.transformChanged.emit()
    
    def start_floor_picking(self):
        """바닥 지점 피킹 모드 시작"""
        if self.viewport.selected_obj is None:
            return
        self.viewport.picking_mode = 'floor'
        self.viewport.status_info = "📍 바닥에 닿을 메쉬의 지점을 클릭하세요..."
        self.viewport.update()
        
    def center_mesh(self):
        """메쉬를 월드 원점(0,0,0)으로 이동"""
        if self.viewport.selected_obj is None:
            return
        self.trans_x.setValue(0.0)
        self.trans_y.setValue(0.0)
        self.trans_z.setValue(0.0)
        self.viewport.camera.center = np.array([0.0, 0.0, 0.0])
        self.viewport.update()
    
    def align_to_floor(self):
        """
        메쉬를 바닥(Y=0)에 '놓기'
        현재 회전 상태를 유지한 채로, 메쉬의 가장 낮은 점이 Y=0에 닿도록 이동합니다.
        마치 실제 유물을 바닥에 놓는 것처럼 동작합니다.
        """
        obj = self.viewport.selected_obj
        if obj is None:
            return
        
        # 로컬 정점들에 현재 회전을 적용하여 월드 좌표 계산
        vertices = obj.mesh.vertices.copy()
        
        # 회전 적용 (X -> Y -> Z 순서, OpenGL과 동일)
        rx, ry, rz = np.radians(obj.rotation)
        
        # X축 회전
        cos_x, sin_x = np.cos(rx), np.sin(rx)
        rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        
        # Y축 회전
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        
        # Z축 회전
        cos_z, sin_z = np.cos(rz), np.sin(rz)
        rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        
        # 전체 회전 행렬 (OpenGL 순서: X -> Y -> Z)
        rotation_matrix = rot_z @ rot_y @ rot_x
        
        # 모든 정점에 회전 및 스케일 적용
        rotated_vertices = (rotation_matrix @ vertices.T).T * obj.scale
        
        # 회전된 정점들 중 가장 낮은 Y값 찾기
        min_y = rotated_vertices[:, 1].min()
        
        # Y를 -min_y로 설정하면 가장 낮은 점이 Y=0에 닿음
        #setValue가 이벤트를 발생시켜 viewport.update()를 호출함
        self.trans_y.setValue(-min_y)
        # 즉시 UI 동기화
        self.viewport.update()
    
    def reset_transform(self):
        self.trans_x.setValue(0)
        self.trans_y.setValue(0)
        self.trans_z.setValue(0)
        self.rot_x.setValue(0)
        self.rot_y.setValue(0)
        self.rot_z.setValue(0)
        self.scale_slider.setValue(100)
        self.scale_spin.setValue(1.0)
    
    def bake_rotation(self):
        """
        현재 회전을 메쉬 정점에 적용하고 회전값을 0으로 리셋
        이렇게 하면 현재 자세가 새로운 '기본' 자세가 되고,
        XYZ 축이 현재 메쉬 방향에 맞춰 재설정됩니다.
        """
        obj = self.viewport.selected_obj
        if obj is None:
            return
        
        # 회전 행렬 계산
        rx, ry, rz = np.radians(obj.rotation)
        
        cos_x, sin_x = np.cos(rx), np.sin(rx)
        rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        
        cos_z, sin_z = np.cos(rz), np.sin(rz)
        rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        
        rotation_matrix = rot_z @ rot_y @ rot_x
        
        # 모든 정점에 회전 적용
        obj.mesh.vertices = (rotation_matrix @ obj.mesh.vertices.T).T
        
        # 법선 벡터도 회전 적용
        obj.mesh.face_normals = (rotation_matrix @ obj.mesh.face_normals.T).T
        if hasattr(obj.mesh, 'vertex_normals') and obj.mesh.vertex_normals is not None:
            obj.mesh.vertex_normals = (rotation_matrix @ obj.mesh.vertex_normals.T).T
        
        # 회전값 리셋
        obj.rotation = np.array([0.0, 0.0, 0.0])
        
        # VBO 업데이트
        self.viewport.update_vbo(obj)
        
        # UI 업데이트
        self.rot_x.setValue(0)
        self.rot_y.setValue(0)
        self.rot_z.setValue(0)
        
        self.viewport.update()
    
    def on_scale_changed(self, value):
        """슬라이더에서 스케일 변경"""
        scale = value / 100.0
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(scale)
        self.scale_spin.blockSignals(False)
        if self.viewport.selected_obj:
            self.viewport.selected_obj.scale = scale
            self.viewport.update()
    
    def on_scale_spin_changed(self, value):
        """스핀박스에서 스케일 변경"""
        self.scale_slider.blockSignals(True)
        self.scale_slider.setValue(int(value * 100))
        self.scale_slider.blockSignals(False)
        if self.viewport.selected_obj:
            self.viewport.selected_obj.scale = value
            self.viewport.update()
    
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
        
        # 원호 관리
        arc_layout = QHBoxLayout()
        arc_label = QLabel("부착된 원호:")
        arc_layout.addWidget(arc_label)
        arc_layout.addStretch()
        
        self.btn_clear_arcs = QPushButton("🗑️ 모든 원호 삭제")
        self.btn_clear_arcs.setToolTip("선택된 객체의 모든 원호 삭제")
        arc_layout.addWidget(self.btn_clear_arcs)
        curve_layout.addRow(arc_layout)
        
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
        
        self.setWindowTitle("ArchMeshRubbing v1.0.0")
        self.resize(1400, 900)
        
        # 메인 위젯
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
        # 중앙 위젯 (3D 뷰포트)
        self.viewport = Viewport3D()
        self.setCentralWidget(self.viewport)
        
        # 씬 매니저 연결
        self.viewport.selectionChanged.connect(self.on_selection_changed)
        self.viewport.meshLoaded.connect(self.on_mesh_loaded)
        self.viewport.meshTransformChanged.connect(self.sync_transform_panel)
        self.viewport.floorPointPicked.connect(self.on_floor_point_picked)
        
        # 도움말 위젯 (오버레이처럼 작동하도록 뷰포트 위에 띄우거나 하단에 배치 가능)
        # 일단은 뷰포트 하단에 고정
        self.help_widget = HelpWidget()
        
        # 도킹 위젯 설정
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowTabbedDocks)
        
        # 1. 씬 패널 (도킹)
        self.scene_dock = QDockWidget("🌲 씬 (레이어)", self)
        self.scene_panel = ScenePanel()
        self.scene_panel.selectionChanged.connect(self.viewport.select_object)
        self.scene_panel.visibilityChanged.connect(self.on_visibility_changed)
        self.scene_panel.arcDeleted.connect(self.on_arc_deleted)
        self.scene_dock.setWidget(self.scene_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.scene_dock)
        
        # 2. 정치 패널 (도킹)
        self.transform_dock = QDockWidget("📐 정치 (변환)", self)
        transform_scroll = QScrollArea()
        transform_scroll.setWidgetResizable(True)
        transform_content = QWidget()
        transform_layout = QVBoxLayout(transform_content)
        
        self.props_panel = PropertiesPanel()
        transform_layout.addWidget(self.props_panel)
        
        self.transform_panel = TransformPanel(self.viewport, self.help_widget)
        transform_layout.addWidget(self.transform_panel)
        transform_layout.addStretch()
        
        transform_scroll.setWidget(transform_content)
        self.transform_dock.setWidget(transform_scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.transform_dock)
        
        # 3. 선택 패널 (도킹)
        self.selection_dock = QDockWidget("✋ 선택 및 영역", self)
        self.selection_panel = SelectionPanel(self.help_widget)
        self.selection_panel.selectionChanged.connect(self.on_selection_action)
        self.selection_dock.setWidget(self.selection_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.selection_dock)
        
        # 4. 펼침 패널 (도킹)
        self.flatten_dock = QDockWidget("🗺️ 펼침 (Flatten)", self)
        self.flatten_panel = FlattenPanel(self.help_widget)
        self.flatten_panel.flattenRequested.connect(self.on_flatten_requested)
        self.flatten_panel.btn_measure.toggled.connect(self.toggle_curvature_mode)
        self.flatten_panel.btn_fit_arc.clicked.connect(self.fit_curvature_arc)
        self.flatten_panel.btn_clear_points.clicked.connect(self.clear_curvature_points)
        self.flatten_panel.btn_clear_arcs.clicked.connect(self.clear_all_arcs)
        
        self.flatten_dock.setWidget(self.flatten_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.flatten_dock)
        
        # 5. 내보내기 패널 (도킹)
        self.export_dock = QDockWidget("📤 내보내기", self)
        self.export_panel = ExportPanel()
        self.export_panel.exportRequested.connect(self.on_export_requested)
        self.export_dock.setWidget(self.export_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.export_dock)
        
        # 씬 패널은 좌측에 독립적으로 유지
        self.scene_dock.show()
        self.scene_dock.raise_()
        
        # 우측 패널들만 탭으로 묶기
        self.tabifyDockWidget(self.transform_dock, self.selection_dock)
        self.tabifyDockWidget(self.selection_dock, self.flatten_dock)
        self.tabifyDockWidget(self.flatten_dock, self.export_dock)
        
        self.transform_dock.raise_() 
    
    def on_floor_point_picked(self, point):
        """사용자가 클릭한 점을 Y=0 바닥에 맞춤"""
        obj = self.viewport.selected_obj
        if not obj: return
        
        # 현재 Y 위치에서 점의 월드 Y만큼 빼주면 바닥에 닿음
        # 하지만 point는 이미 월드 좌표이므로, 현재 translation.y에서 point.y를 빼주면 됨
        old_y = obj.translation[1]
        new_y = old_y - point[1]
        
        # UI 업데이트가 자동으로 obj.translation을 바꿈
        self.transform_panel.trans_y.setValue(new_y)
        self.viewport.status_info = "✅ 바닥 정렬 완료"
        self.viewport.update()

    def on_arc_deleted(self, obj_idx, arc_idx):
        """특정 객체의 특정 원호 삭제"""
        if 0 <= obj_idx < len(self.viewport.objects):
            obj = self.viewport.objects[obj_idx]
            if 0 <= arc_idx < len(obj.fitted_arcs):
                del obj.fitted_arcs[arc_idx]
                self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
                self.viewport.update()
                self.status_info.setText(f"🗑️ 원호 #{arc_idx+1} 삭제됨")
    
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
        
        action_reset_fit = QAction("🎯 정치 초기화 (Match)", self)
        action_reset_fit.setToolTip("메쉬의 변환을 리셋하고 원점으로 맞춤 (정치)")
        action_reset_fit.triggered.connect(self.reset_transform_and_center)
        toolbar.addAction(action_reset_fit)
        
        action_fit = QAction("🔍 뷰 맞춤", self)
        action_fit.setToolTip("메쉬가 화면에 꽉 차도록 카메라 조정")
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
        
        # 버전 표시 (사용자 확인용)
        self.status_ver = QLabel("v1.0.0")
        self.status_ver.setStyleSheet("color: #a0aec0; font-size: 10px; margin-left: 10px;")
        self.statusbar.addPermanentWidget(self.status_ver)
    
    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "3D 메쉬 파일 열기",
            "",
            "3D Files (*.obj *.ply *.stl *.off);;All Files (*)"
        )
        
        if filepath:
            # 단위 선택 다이얼로그
            dialog = UnitSelectionDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                scale_factor = dialog.get_scale_factor()
                self.load_mesh(filepath, scale_factor)
    
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
            # 드롭 시에도 단위 선택 다이얼로그 표시
            dialog = UnitSelectionDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                scale_factor = dialog.get_scale_factor()
                self.load_mesh(filepath, scale_factor)
    
    def load_mesh(self, filepath: str, scale_factor: float = 1.0):
        try:
            self.status_info.setText(f"⏳ 로딩 중: {Path(filepath).name}")
            self.status_mesh.setText("")
            QApplication.processEvents()
            
            # 메쉬 로드
            mesh = trimesh.load(filepath)
            
            # 단위 변환 적용 (예: mm 파일의 184.9 -> cm 기준 18.49로 변환)
            if scale_factor != 1.0:
                mesh.apply_scale(scale_factor)
            
            # 메쉬의 물리적 중심(Centroid)을 (0,0,0)으로 이동
            # 이렇게 해야 '중심 이동' 버튼을 눌렀을 때 화면 한가운데에 정확히 옵니다.
            centroid = mesh.vertices.mean(axis=0)
            mesh.vertices -= centroid
                
            self.current_mesh = mesh
            self.current_filepath = filepath
            
            # 뷰포트에 추가
            self.viewport.add_mesh_object(mesh, name=Path(filepath).name)
            
            # 상태바 업데이트
            self.status_info.setText(f"✅ 로드됨: {Path(filepath).name} (원점 정렬 완료)")
            self.status_mesh.setText(f"V: {len(mesh.vertices):,} | F: {len(mesh.faces):,}")
            self.status_grid.setText(f"격자: {self.viewport.grid_spacing}cm")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"파일 로드 실패:\n{e}")
            self.status_info.setText("❌ 로드 실패")
            self.status_mesh.setText("")
    
    def on_mesh_loaded(self, mesh):
        self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
        self.props_panel.update_mesh_info(mesh, self.current_filepath)
        self.sync_transform_panel()
        
    def on_selection_changed(self, index):
        self.scene_panel.update_list(self.viewport.objects, index)
        self.sync_transform_panel()
        
    def on_visibility_changed(self, index, visible):
        if 0 <= index < len(self.viewport.objects):
            self.viewport.objects[index].visible = visible
            self.viewport.update()
            
    def sync_transform_panel(self):
        obj = self.viewport.selected_obj
        if not obj: 
            # Clear transform panel if no object is selected
            self.transform_panel.reset_transform()
            return
        
        # 스핀박스 시그널 블록하고 값 설정
        self.transform_panel.trans_x.blockSignals(True)
        self.transform_panel.trans_y.blockSignals(True)
        self.transform_panel.trans_z.blockSignals(True)
        self.transform_panel.rot_x.blockSignals(True)
        self.transform_panel.rot_y.blockSignals(True)
        self.transform_panel.rot_z.blockSignals(True)
        self.transform_panel.scale_spin.blockSignals(True)
        self.transform_panel.scale_slider.blockSignals(True)
        
        self.transform_panel.trans_x.setValue(obj.translation[0])
        self.transform_panel.trans_y.setValue(obj.translation[1])
        self.transform_panel.trans_z.setValue(obj.translation[2])
        self.transform_panel.rot_x.setValue(obj.rotation[0])
        self.transform_panel.rot_y.setValue(obj.rotation[1])
        self.transform_panel.rot_z.setValue(obj.rotation[2])
        self.transform_panel.scale_spin.setValue(obj.scale)
        self.transform_panel.scale_slider.setValue(int(obj.scale * 100))
        
        self.transform_panel.trans_x.blockSignals(False)
        self.transform_panel.trans_y.blockSignals(False)
        self.transform_panel.trans_z.blockSignals(False)
        self.transform_panel.rot_x.blockSignals(False)
        self.transform_panel.rot_y.blockSignals(False)
        self.transform_panel.rot_z.blockSignals(False)
        self.transform_panel.scale_spin.blockSignals(False)
        self.transform_panel.scale_slider.blockSignals(False)
    
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
    
    def reset_transform_and_center(self):
        """정치 초기화: 변환 리셋 + 원점 중심 이동"""
        if self.viewport.selected_obj:
            self.transform_panel.reset_transform()
            self.transform_panel.center_mesh()
            self.status_info.setText("✅ 정치 초기화 완료 (0,0,0)")
            
    def reset_view(self):
        self.viewport.camera.reset()
        self.viewport.update()
    
    def fit_view(self):
        obj = self.viewport.selected_obj
        if obj:
            self.viewport.camera.fit_to_bounds(obj.mesh.bounds)
            self.viewport.camera.center = obj.translation.copy()
            self.viewport.update()
        elif self.current_mesh is not None:
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
        """찍은 点들로 원호 피팅 (월드 좌표계 고정)"""
        if len(self.viewport.picked_points) < 3:
            QMessageBox.warning(self, "경고", "최소 3개의 점이 필요합니다.\nShift+클릭으로 메쉬 위에 점을 찍으세요.")
            return
        
        obj = self.viewport.selected_obj
        if obj is None:
            QMessageBox.warning(self, "경고", "먼저 메쉬를 선택하세요.")
            return
        
        from src.core.curvature_fitter import CurvatureFitter
        
        # 월드 좌표 점들을 그대로 사용 (메쉬와 분리하기 위해)
        world_points = self.viewport.picked_points
        
        fitter = CurvatureFitter()
        arc = fitter.fit_arc(world_points)
        
        if arc is None:
            QMessageBox.warning(self, "경고", "원호 피팅에 실패했습니다.\n점들이 일직선 위에 있거나 너무 가까울 수 있습니다.")
            return
        
        # 객체에 원호 부착 (데이터 구조는 유지하되 렌더링 시 변환 적용 안 함)
        obj.fitted_arcs.append(arc)
        
        # 임시 데이터 초기화
        self.viewport.fitted_arc = None
        self.viewport.picked_points = []
        self.viewport.update()
        
        # 펼침 패널의 곡률 반경에 자동 입력
        radius_mm = arc.radius * 10  # cm → mm
        self.flatten_panel.spin_radius.setValue(radius_mm)
        
        self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
        arc_count = len(obj.fitted_arcs)
        self.status_info.setText(f"✅ 원호 #{arc_count} 생성됨 (월드 고정): 반지름 = {arc.radius:.2f} cm ({radius_mm:.1f} mm)")
    
    def clear_curvature_points(self):
        """곡률 측정용 점 초기화"""
        self.viewport.clear_curvature_picks()
        self.status_info.setText("🗑️ 측정 점 초기화됨")
    
    def clear_all_arcs(self):
        """선택된 객체의 모든 원호 삭제"""
        obj = self.viewport.selected_obj
        if obj and obj.fitted_arcs:
            count = len(obj.fitted_arcs)
            obj.fitted_arcs = []
            self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
            self.viewport.update()
            self.status_info.setText(f"🗑️ {count}개 원호 삭제됨")
    
    def show_about(self):
        icon_path = get_icon_path()
        msg = QMessageBox(self)
        msg.setWindowTitle("ArchMeshRubbing v2")
        
        if icon_path:
            msg.setIconPixmap(QPixmap(icon_path).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio))
        
        msg.setText("""
            <h2>ArchMeshRubbing v2</h2>
            <p>고고학 메쉬 탁본 도구</p>
            <p style="font-size: 11px; color: #718096;">© 2026 balguljang2 (lzpxilfe) / Licensed under GPLv2</p>
            <hr>
            <p><b>조작법:</b></p>
            <ul>
                <li>좌클릭 드래그: 회전</li>
                <li>우클릭 드래그: 이동</li>
                <li>스크롤: 확대/축소</li>
                <li>1~6: 다방향 프리셋 뷰</li>
            </ul>
        """)
        msg.exec()


def main():
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # 아이콘 설정
        icon_path = get_icon_path()
        if icon_path:
            app.setWindowIcon(QIcon(icon_path))
        
        # 1. 스플래시 화면 표시
        splash = SplashScreen()
        splash.show()
        splash.setCursor(Qt.CursorShape.WaitCursor)
        
        splash.showMessage("Loading standard libraries...")
        QTimer.singleShot(500, lambda: splash.showMessage("Configuring OpenGL context..."))
        
        # 2. 메인 윈도우 생성
        splash.showMessage("Initializing UI components...")
        window = MainWindow()
        
        # 3. 마무리 및 스플래시 닫기
        QTimer.singleShot(1500, lambda: (splash.close(), window.show()))
        
        sys.exit(app.exec())
    except Exception as e:
        # 치명적 오류 팝업 (EXE 등에서 유용)
        import traceback
        err_msg = f"Application crashed on startup:\n\n{e}\n\n{traceback.format_exc()}"
        print(err_msg)
        try:
            temp_app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(None, "Fatal Startup Error", err_msg)
        except:
            pass
        sys.exit(1)


if __name__ == '__main__':
    main()
