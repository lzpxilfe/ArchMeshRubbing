"""
ArchMeshRubbing v1.0.1 - Complete Interactive Application
Copyright (C) 2026 balguljang2 (lzpxilfe)
Licensed under the GNU General Public License v2.0 (GPL2)
"""

import sys
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QDockWidget, QTreeWidget,
    QTreeWidgetItem, QGroupBox, QDoubleSpinBox, QFormLayout,
    QSlider, QSpinBox, QStatusBar, QToolBar, QFrame,
    QMessageBox, QTextEdit, QProgressBar, QComboBox,
    QCheckBox, QScrollArea, QSizePolicy, QButtonGroup, QDialog,
    QGridLayout, QProgressDialog, QMenu
)
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal, QThread, QBuffer, QByteArray, QIODevice
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QPixmap, QShortcut
import numpy as np
from PIL import Image
import io

_LOGGER = logging.getLogger(__name__)
_log_path: Path | None = None
APP_NAME = "ArchMeshRubbing"
APP_VERSION = "1.0.1"


def _safe_git_info(repo_dir: str) -> tuple[str | None, bool]:
    try:
        sha = (
            subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8", errors="replace")
            .strip()
        )
        dirty = bool(
            subprocess.check_output(["git", "-C", repo_dir, "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode("utf-8", errors="replace")
            .strip()
        )
        return (sha or None), dirty
    except Exception:
        return None, False


def _collect_debug_info(*, basedir: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sha, dirty = _safe_git_info(basedir)
    sha_s = f"{sha}{'*' if dirty else ''}" if sha else "unknown"

    def mod_path(name: str) -> str:
        try:
            import importlib

            m = importlib.import_module(name)
            return str(getattr(m, "__file__", "<no __file__>"))
        except Exception as e:
            return f"<import failed: {type(e).__name__}: {e}>"

    parts = [
        f"time: {ts}",
        f"app: {APP_NAME} v{APP_VERSION} (git {sha_s})",
        f"python: {sys.executable}",
        f"cwd: {Path.cwd()}",
        f"basedir: {basedir}",
        "modules:",
        f"  app_interactive: {__file__}",
        f"  src.gui.viewport_3d: {mod_path('src.gui.viewport_3d')}",
        f"  src.core.surface_separator: {mod_path('src.core.surface_separator')}",
        f"  src.core.flattener: {mod_path('src.core.flattener')}",
    ]
    return "\n".join(parts)

# Add src to path
# Add basedir to path so 'src' package can be found
if getattr(sys, 'frozen', False):
    basedir = getattr(sys, "_MEIPASS", str(Path(__file__).parent))
else:
    basedir = str(Path(__file__).parent)
sys.path.insert(0, basedir)

from src.gui.viewport_3d import Viewport3D  # noqa: E402
from src.core.mesh_loader import MeshLoader, MeshProcessor  # noqa: E402
from src.core.profile_exporter import ProfileExporter  # noqa: E402
from src.gui.profile_graph_widget import ProfileGraphWidget  # noqa: E402


class MeshLoadThread(QThread):
    loaded = pyqtSignal(object, str)
    failed = pyqtSignal(str)

    def __init__(self, filepath: str, scale_factor: float, default_unit: str):
        super().__init__()
        self._filepath = str(filepath)
        self._scale_factor = float(scale_factor)
        self._default_unit = str(default_unit)

    def run(self):
        try:
            loader = MeshLoader(default_unit=self._default_unit)
            mesh_data = loader.load(self._filepath)

            if self._scale_factor != 1.0:
                mesh_data.vertices *= self._scale_factor
                mesh_data._bounds = None
                mesh_data._centroid = None
                mesh_data._surface_area = None

            self.loaded.emit(mesh_data, self._filepath)
        except Exception as e:
            _LOGGER.exception("Mesh load failed: %s", self._filepath)
            self.failed.emit(f"{type(e).__name__}: {e}")


class SliceComputeThread(QThread):
    computed = pyqtSignal(float, object)  # z_height, world_contours
    failed = pyqtSignal(float, str)       # z_height, message

    def __init__(self, mesh_data, translation, rotation, scale: float, z_height: float):
        super().__init__()
        self._mesh_data = mesh_data
        self._translation = np.asarray(translation, dtype=np.float64)
        self._rotation = np.asarray(rotation, dtype=np.float64)
        self._scale = float(scale)
        self._z = float(z_height)

    def run(self):
        try:
            from src.core.mesh_slicer import MeshSlicer
            from scipy.spatial.transform import Rotation as R

            slicer = MeshSlicer(self._mesh_data.to_trimesh())

            inv_rot = R.from_euler('xyz', self._rotation, degrees=True).inv().as_matrix()
            inv_scale = 1.0 / self._scale if self._scale != 0 else 1.0

            world_origin = np.array([0.0, 0.0, self._z], dtype=np.float64)
            local_origin = inv_scale * inv_rot @ (world_origin - self._translation)

            world_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            local_normal = inv_rot @ world_normal

            contours_local = slicer.slice_with_plane(local_origin, local_normal)

            rot_mat = R.from_euler('xyz', self._rotation, degrees=True).as_matrix()
            world_contours = []
            for cnt in contours_local:
                w_cnt = (rot_mat @ (cnt * self._scale).T).T + self._translation
                world_contours.append(w_cnt)

            self.computed.emit(self._z, world_contours)
        except Exception as e:
            _LOGGER.exception("Slice compute failed (z=%s)", self._z)
            self.failed.emit(self._z, f"{type(e).__name__}: {e}")


class ProfileExportThread(QThread):
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        mesh_data,
        view: str,
        output_path: str,
        translation: np.ndarray,
        rotation: np.ndarray,
        scale: float,
        viewport_image: Image.Image,
        opengl_matrices: tuple[Any, Any, Any],
        cut_lines_world: list[Any],
        cut_profiles_world: list[Any],
        resolution: int = 2048,
        grid_spacing: float = 1.0,
        include_grid: bool = True,
    ):
        super().__init__()
        self._mesh_data = mesh_data
        self._view = str(view)
        self._output_path = str(output_path)
        self._translation = np.asarray(translation, dtype=np.float64)
        self._rotation = np.asarray(rotation, dtype=np.float64)
        self._scale = float(scale)
        self._viewport_image = viewport_image
        self._opengl_matrices = opengl_matrices
        self._cut_lines_world = cut_lines_world
        self._cut_profiles_world = cut_profiles_world
        self._resolution = int(resolution)
        self._grid_spacing = float(grid_spacing)
        self._include_grid = bool(include_grid)

    def run(self):
        try:
            exporter = ProfileExporter(resolution=self._resolution)
            result_path = exporter.export_profile(
                self._mesh_data,
                view=self._view,
                output_path=self._output_path,
                translation=self._translation,
                rotation=self._rotation,
                scale=self._scale,
                grid_spacing=self._grid_spacing,
                include_grid=self._include_grid,
                viewport_image=self._viewport_image,
                opengl_matrices=self._opengl_matrices,
                cut_lines_world=self._cut_lines_world,
                cut_profiles_world=self._cut_profiles_world,
            )
            self.done.emit(str(result_path))
        except Exception as e:
            _LOGGER.exception("Profile export failed (%s -> %s)", self._view, self._output_path)
            self.failed.emit(f"{type(e).__name__}: {e}")


class TaskThread(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, task_name: str, fn: Callable[[], Any]):
        super().__init__()
        self._task_name = str(task_name)
        self._fn = fn

    def run(self):
        try:
            result = self._fn()
            self.done.emit(result)
        except Exception as e:
            _LOGGER.exception("Task failed: %s", self._task_name)
            self.failed.emit(f"{type(e).__name__}: {e}")


def get_icon_path():
    """아이콘 경로 반환"""
    icon_path = Path(basedir) / "resources" / "icons" / "app_icon.png"
    if icon_path.exists():
        return str(icon_path)
    return None


class HelpWidget(QTextEdit):
    """도움말 위젯"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        try:
            self.setMinimumHeight(120)
        except Exception:
            pass
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
            <h3 style="margin:0; color:#2c5282;">✋ 표면(내/외면) 선택</h3>
            <p style="font-size:11px;">
            내면/외면/미구(경계)를 지정하는 도구입니다.<br><br>

            <b>📊 내면/외면 자동 감지</b><br>
            - 클릭: <b>법선</b> 기반 자동 분리 (일반 메쉬에 빠름)<br>
            - <b>Shift + 클릭:</b> <b>상면/하면(보이는 면)</b> 기반 자동 분리 (기와/얇은 쉘에 유리)<br><br>

            <b>🖱️ 찍기(표면 클릭)</b><br>
            - 클릭: <b>한 면</b>만 토글(추가/해제)<br>
            - <b>Shift/Ctrl + 클릭:</b> <b>매직완드처럼 조금씩 확장</b> (Shift/Ctrl 클릭을 반복할수록 더 넓게)<br>
            - <b>Alt:</b> 삭제 모드<br><br>

            <b>🖌️ 브러시</b><br>
            - 드래그: 칠하는 면을 추가, <b>Alt+드래그</b>: 삭제<br><br>

            <b>⭕ 올가미(면적)</b><br>
            - 좌클릭으로 점 추가 → 첫 점 근처 클릭 또는 우클릭으로 확정<br>
            </p>
        """)


class SplashScreen(QWidget):
    """프로세스 시작 시 보여주는 스플래시 화면"""
    
    def __init__(self):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.SplashScreen
            | Qt.WindowType.WindowStaysOnTopHint,
        )
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
            pix = QPixmap(icon_path).scaled(
                80,
                80,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.icon_label.setPixmap(pix)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.icon_label)
        
        # 타이틀
        title = QLabel("ArchMeshRubbing v1")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #2c5282;
            margin-top: 10px;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title)
        
        # 버전 정보 추가 (사용자 확인용)
        version = QLabel("Version: 1.0.1")
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
        if idx == 0:
            return 0.1
        if idx == 1:
            return 1.0
        if idx == 2:
            return 100.0
        return 1.0


class ScenePanel(QWidget):
    """씬 내의 객체 목록과 부착된 요소를 보여주는 트리 패널"""
    selectionChanged = pyqtSignal(int)
    visibilityChanged = pyqtSignal(int, bool)
    arcDeleted = pyqtSignal(int, int) # object_idx, arc_idx
    layerVisibilityChanged = pyqtSignal(int, int, bool)  # object_idx, layer_idx, visible
    layerDeleted = pyqtSignal(int, int)  # object_idx, layer_idx
    layerMoveRequested = pyqtSignal(int, int, float, float)  # object_idx, layer_idx, dx, dy
    layerOffsetResetRequested = pyqtSignal(int, int)  # object_idx, layer_idx
    
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

            # 저장된 단면/가이드 레이어
            for k, layer in enumerate(getattr(obj, "polyline_layers", []) or []):
                layer_item = QTreeWidgetItem(mesh_item)
                name = str(layer.get("name", "")).strip() or f"레이어 #{k+1}"
                layer_item.setText(0, name)

                visible = bool(layer.get("visible", True))
                layer_item.setText(1, "👁️" if visible else "👓")

                pts = layer.get("points", []) or []
                kind = str(layer.get("kind", "")).strip()
                if kind == "section_profile":
                    kind_label = "단면"
                elif kind == "cut_line":
                    kind_label = "단면선"
                else:
                    kind_label = kind or "레이어"
                layer_item.setText(2, f"{kind_label} ({len(pts):,})")
                layer_item.setData(0, Qt.ItemDataRole.UserRole, ("layer", i, k))
            
            mesh_item.setExpanded(True)
            if i == selected_index:
                self.tree.setCurrentItem(mesh_item)
        self.tree.blockSignals(False)
                
    def on_item_clicked(self, item, column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        
        if data[0] == "mesh":
            index = data[1]
            if column == 1: # 가시성 토글
                visible = item.text(1) == "👓"
                item.setText(1, "👁️" if visible else "👓")
                self.visibilityChanged.emit(index, visible)
            else:
                self.selectionChanged.emit(index)
        elif data[0] == "layer":
            obj_idx = int(data[1])
            layer_idx = int(data[2])
            if column == 1:
                visible = item.text(1) == "👓"
                item.setText(1, "👁️" if visible else "👓")
                self.layerVisibilityChanged.emit(obj_idx, layer_idx, visible)

    def show_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if not item:
            return
        
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        if data[0] == "arc":
            menu = QMenu(self) # 원인: 부모 위젯 지정
            delete_action = menu.addAction("🗑️ 원호 삭제")
            action = menu.exec(self.tree.mapToGlobal(pos))
            if action == delete_action:
                self.arcDeleted.emit(data[1], data[2])
        elif data[0] == "layer":
            menu = QMenu(self)
            move_left = menu.addAction("왼쪽 5cm")
            move_right = menu.addAction("오른쪽 5cm")
            move_up = menu.addAction("위로 5cm")
            move_down = menu.addAction("아래로 5cm")
            reset_offset = menu.addAction("오프셋 초기화")
            menu.addSeparator()
            delete_action = menu.addAction("🗑️ 레이어 삭제")
            action = menu.exec(self.tree.mapToGlobal(pos))
            if action == move_left:
                self.layerMoveRequested.emit(int(data[1]), int(data[2]), -5.0, 0.0)
            elif action == move_right:
                self.layerMoveRequested.emit(int(data[1]), int(data[2]), 5.0, 0.0)
            elif action == move_up:
                self.layerMoveRequested.emit(int(data[1]), int(data[2]), 0.0, 5.0)
            elif action == move_down:
                self.layerMoveRequested.emit(int(data[1]), int(data[2]), 0.0, -5.0)
            elif action == reset_offset:
                self.layerOffsetResetRequested.emit(int(data[1]), int(data[2]))
            elif action == delete_action:
                self.layerDeleted.emit(int(data[1]), int(data[2]))


class TransformToolbar(QToolBar):
    """상단 고정 정치(변환) 툴바"""
    def __init__(self, viewport: Viewport3D, parent=None):
        super().__init__("정치 도구", parent)
        self.viewport = viewport
        self.setIconSize(QSize(24, 24))
        self.init_ui()

    def init_ui(self):
        # 이동 (cm)
        self.addWidget(QLabel(" 📍 이동: "))
        self.trans_x = self._create_spin(-10000, 10000, "X")
        self.trans_y = self._create_spin(-10000, 10000, "Y")
        self.trans_z = self._create_spin(-10000, 10000, "Z")
        self.addWidget(self.trans_x)
        self.addWidget(self.trans_y)
        self.addWidget(self.trans_z)
        
        self.addSeparator()
        
        # 회전 (deg)
        self.addWidget(QLabel(" 🔄 회전: "))
        self.rot_x = self._create_spin(-360, 360, "Rx")
        self.rot_y = self._create_spin(-360, 360, "Ry")
        self.rot_z = self._create_spin(-360, 360, "Rz")
        self.addWidget(self.rot_x)
        self.addWidget(self.rot_y)
        self.addWidget(self.rot_z)
        
        self.addSeparator()
        
        # 배율
        self.addWidget(QLabel(" 🔍 배율: "))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.01, 100.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setFixedWidth(70)
        self.addWidget(self.scale_spin)
        
        self.addSeparator()
        
        # 버튼들
        self.btn_bake = QPushButton("📌 정치 확정")
        self.btn_bake.setToolTip("현재 변환을 메쉬에 영구 적용하고 위치를 고정합니다")
        self.btn_bake.setStyleSheet("QPushButton { font-weight: bold; padding: 2px 10px; }")
        self.addWidget(self.btn_bake)

        self.btn_fixed = QPushButton("🔒 고정상태로")
        self.btn_fixed.setToolTip("정치 확정(Bake) 이후의 고정 상태로 되돌립니다 (실수로 이동/회전했을 때)")
        self.btn_fixed.setEnabled(False)
        self.addWidget(self.btn_fixed)
        
        self.btn_reset = QPushButton("🔄 초기화")
        self.addWidget(self.btn_reset)
        
        self.btn_flat = QPushButton("🌓 Flat Shading")
        self.btn_flat.setCheckable(True)
        self.btn_flat.setToolTip("명암 없이 메쉬를 밝게 봅니다 (회전 시 어두워짐 방지)")
        self.addWidget(self.btn_flat)

    def _create_spin(self, min_v, max_v, prefix=""):
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setDecimals(2)
        spin.setPrefix(f"{prefix}: ")
        spin.setFixedWidth(90)
        return spin


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

        hint = QLabel(
            "정치/바닥 정렬은 상단 툴바를 사용하세요.\n"
            "✏️ 바닥 면 그리기: 상단 툴바 버튼 → 메쉬 클릭으로 점 추가 → Enter로 확정"
        )
        hint.setStyleSheet("color: #718096; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch()
    
    def enterEvent(self, event):
        self.help_widget.set_transform_help()
        super().enterEvent(event)
    
class FlattenPanel(QWidget):
    """펼침 설정 패널 (Phase B)"""
    
    flattenRequested = pyqtSignal(dict)
    selectionRequested = pyqtSignal(str, object)
    
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

        # 표면 선택/지정 (내/외면/미구)
        surface_group = QGroupBox("✋ 표면 선택/지정 (내/외면)")
        surface_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        surface_layout = QVBoxLayout(surface_group)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("대상:"))
        self.combo_surface_target = QComboBox()
        self.combo_surface_target.addItems(["🌞 외면", "🌙 내면", "🧩 미구"])
        self.combo_surface_target.setToolTip("지정할 표면 그룹 선택")
        self.combo_surface_target.currentIndexChanged.connect(
            lambda _i: self.selectionRequested.emit("surface_target", self.current_surface_target())
        )
        target_row.addWidget(self.combo_surface_target)
        surface_layout.addLayout(target_row)

        tool_row = QHBoxLayout()
        self.btn_surface_click = QPushButton("👆 찍기(자동 확장)")
        self.btn_surface_click.setToolTip(
            "클릭한 면이 속한 '매끈한 연결 영역'을 자동 확장해 지정합니다.\n"
            "Shift/Ctrl=추가, Alt=제거, ESC=종료"
        )
        self.btn_surface_click.clicked.connect(
            lambda: self.selectionRequested.emit("surface_tool", {"tool": "click", "target": self.current_surface_target()})
        )
        tool_row.addWidget(self.btn_surface_click)

        self.btn_surface_brush = QPushButton("🖌️ 보정(브러시)")
        self.btn_surface_brush.setToolTip("드래그로 칠해서 보정합니다. Alt=지우기, ESC=종료")
        self.btn_surface_brush.clicked.connect(
            lambda: self.selectionRequested.emit("surface_tool", {"tool": "brush", "target": self.current_surface_target()})
        )
        tool_row.addWidget(self.btn_surface_brush)

        self.btn_surface_area = QPushButton("📐 면적(Area)")
        self.btn_surface_area.setToolTip(
            "메쉬 위에 점을 찍어 다각형을 만들고, 보이는 면을 한 번에 지정합니다.\n"
            "시작점 근처 클릭=스냅 닫힘(자동 확정)\n"
            "좌클릭=점 추가(드래그=회전), 우클릭/Enter=확정(우클릭 위치가 완드 기준), Backspace=되돌리기, ESC=취소"
        )
        self.btn_surface_area.clicked.connect(
            lambda: self.selectionRequested.emit(
                "surface_tool",
                {"tool": "area", "target": self.current_surface_target()},
            )
        )
        tool_row.addWidget(self.btn_surface_area)
        surface_layout.addLayout(tool_row)

        self.label_surface_assignment = QLabel("외면: 0 / 내면: 0 / 미구: 0")
        self.label_surface_assignment.setStyleSheet("font-weight: bold; color: #2c5282;")
        surface_layout.addWidget(self.label_surface_assignment)

        action_row = QHBoxLayout()
        btn_clear_target = QPushButton("🗑️ 현재 비우기")
        btn_clear_target.setToolTip("현재 대상(외/내/미구) 지정 면을 모두 비웁니다.")
        btn_clear_target.clicked.connect(
            lambda: self.selectionRequested.emit("surface_clear_target", self.current_surface_target())
        )
        action_row.addWidget(btn_clear_target)

        btn_clear_all = QPushButton("🧼 전체 초기화")
        btn_clear_all.setToolTip("외면/내면/미구 지정을 모두 초기화합니다.")
        btn_clear_all.clicked.connect(lambda: self.selectionRequested.emit("surface_clear_all", None))
        action_row.addWidget(btn_clear_all)
        surface_layout.addLayout(action_row)

        auto_row = QHBoxLayout()
        btn_auto = QPushButton("🤖 자동 분리(실험)")
        btn_auto.setToolTip("완전 자동은 메쉬/정렬 상태에 따라 실패할 수 있습니다. 결과가 이상하면 수동 '찍기'로 지정하세요.")
        btn_auto.clicked.connect(lambda: self.selectionRequested.emit("auto_surface", None))
        auto_row.addWidget(btn_auto)

        btn_auto_migu = QPushButton("📏 미구 자동 감지")
        btn_auto_migu.setToolTip(
            "미구(계단/경계) 영역을 자동으로 찾아 미구로 지정합니다.\n"
            "- 클릭: Y축(기본) 강조 감지\n"
            "- Ctrl+클릭: X축 강조 감지\n"
            "- Shift+클릭: 둘레 경계(Edge belt) 감지"
        )
        btn_auto_migu.clicked.connect(lambda: self.selectionRequested.emit("auto_edge", None))
        auto_row.addWidget(btn_auto_migu)
        surface_layout.addLayout(auto_row)

        layout.addWidget(surface_group)
        
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

    def current_surface_target(self) -> str:
        try:
            idx = int(self.combo_surface_target.currentIndex())
        except Exception:
            idx = 0
        return "inner" if idx == 1 else ("migu" if idx == 2 else "outer")

    def update_surface_assignment_counts(self, outer: int, inner: int, migu: int) -> None:
        try:
            o = int(outer)
        except Exception:
            o = 0
        try:
            i = int(inner)
        except Exception:
            i = 0
        try:
            m = int(migu)
        except Exception:
            m = 0
        try:
            self.label_surface_assignment.setText(f"외면: {o:,} / 내면: {i:,} / 미구: {m:,}")
        except Exception:
            pass
    
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
        self.btn_click.clicked.connect(lambda: self.selectionChanged.emit("tool", {"tool": "click"}))
        self.tool_button_group.addButton(self.btn_click, 0)
        tool_layout.addWidget(self.btn_click)
        
        self.btn_brush = QPushButton("🖌️ 브러시 선택")
        self.btn_brush.setCheckable(True)
        self.btn_brush.setToolTip("드래그로 여러 면 선택")
        self.btn_brush.clicked.connect(lambda: self.selectionChanged.emit("tool", {"tool": "brush"}))
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
        self.btn_lasso.clicked.connect(lambda: self.selectionChanged.emit("tool", {"tool": "lasso"}))
        self.tool_button_group.addButton(self.btn_lasso, 2)
        tool_layout.addWidget(self.btn_lasso)
        
        layout.addWidget(tool_group)
        
        # 자동 분리
        auto_group = QGroupBox("🤖 자동 분리")
        auto_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        auto_layout = QVBoxLayout(auto_group)
        
        btn_auto_surface = QPushButton("📊 내면/외면 자동 감지")
        btn_auto_surface.setToolTip("클릭=법선 기반, Shift+클릭=상/하면(보이는 면) 기반으로 자동 분류")
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



class InfoBarWidget(QWidget):
    """상단 고정용 파일/메쉬 정보 바"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filepath = None
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(10)

        self.label_summary = QLabel("File: - | V: - | F: - | Size: - | Area: - | Tex: -")
        self.label_summary.setWordWrap(False)
        self.label_summary.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.label_summary.setStyleSheet("color: #2d3748;")
        layout.addWidget(self.label_summary, 1)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMaximumHeight(34)

        self.setStyleSheet("""
            InfoBarWidget {
                background-color: #f8f9fa;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
            }
            QLabel { font-size: 11px; }
        """)

    def update_mesh_info(self, mesh, filepath=None):
        self._filepath = filepath
        if mesh is None:
            self.label_summary.setText("File: - | V: - | F: - | Size: - | Area: - | Tex: -")
            return

        file_name = "-"
        if filepath:
            try:
                file_name = Path(filepath).name
                self.label_summary.setToolTip(str(filepath))
            except Exception:
                file_name = str(filepath)

        extents = mesh.extents
        size_txt = f"{extents[0]:.1f}×{extents[1]:.1f}×{extents[2]:.1f}cm"
        try:
            area_txt = f"{mesh.surface_area:.1f}cm²"
        except Exception:
            area_txt = "-"

        tex_txt = "있음" if getattr(mesh, "has_texture", False) else "없음"
        self.label_summary.setText(
            f"File: {file_name} | V: {mesh.n_vertices:,} | F: {mesh.n_faces:,} | "
            f"Size: {size_txt} | Area: {area_txt} | Tex: {tex_txt}"
        )


class SlicingPanel(QWidget):
    """단면 슬라이싱 제어 패널"""
    sliceChanged = pyqtSignal(bool, float)  # enabled, height
    exportRequested = pyqtSignal(float)     # height
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. 활성화 스위치
        self.group = QGroupBox("📏 단면 슬라이싱 (CT)")
        self.group.setCheckable(True)
        self.group.setChecked(False)
        self.group.toggled.connect(self.on_toggled)
        group_layout = QVBoxLayout(self.group)
        
        # 2. 높이 조절 슬라이더
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(-500, 500)  # -5cm ~ 5cm (0.1mm 단위)
        self.slider.setValue(0)
        self.slider.setToolTip("슬라이스 높이 조절 (0.1mm 단위)")
        
        self.spin = QDoubleSpinBox()
        self.spin.setRange(-50.0, 50.0)
        self.spin.setSingleStep(0.1)
        self.spin.setSuffix(" cm")
        self.spin.setDecimals(2)
        
        # 슬라이더 - 스핀박스 양방향 연결
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin.valueChanged.connect(self._on_spin_changed)
        
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.spin)
        group_layout.addLayout(slider_layout)
        
        # 3. 버튼들
        btn_layout = QHBoxLayout()
        self.btn_export = QPushButton("💾 단면 SVG 내보내기")
        self.btn_export.setStyleSheet("background-color: #ebf8ff; font-weight: bold;")
        self.btn_export.clicked.connect(self.on_export_clicked)
        btn_layout.addWidget(self.btn_export)
        
        group_layout.addLayout(btn_layout)
        
        # 도움말
        help_label = QLabel("상면(Top) 뷰에서 보면서 높이를 조절하세요.")
        help_label.setStyleSheet("color: #718096; font-size: 10px;")
        help_label.setWordWrap(True)
        group_layout.addWidget(help_label)
        
        layout.addWidget(self.group)
        layout.addStretch()
        
    def _on_slider_changed(self, val):
        self.spin.blockSignals(True)
        self.spin.setValue(val / 100.0)
        self.spin.blockSignals(False)
        self.sliceChanged.emit(self.group.isChecked(), val / 100.0)
        
    def _on_spin_changed(self, val):
        self.slider.blockSignals(True)
        self.slider.setValue(int(val * 100))
        self.slider.blockSignals(False)
        self.sliceChanged.emit(self.group.isChecked(), val)
        
    def on_toggled(self, checked):
        self.sliceChanged.emit(checked, self.spin.value())
        
    def on_export_clicked(self):
        self.exportRequested.emit(self.spin.value())

    def update_range(self, z_min, z_max):
        """메쉬 범위에 맞춰 슬라이더 범위 업데이트"""
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        
        self.slider.setRange(int(z_min * 100), int(z_max * 100))
        self.spin.setRange(z_min, z_max)
        
        mid = (z_min + z_max) / 2
        self.slider.setValue(int(mid * 100))
        self.spin.setValue(mid)
        
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)


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
        
        btn_export_flat_svg = QPushButton("펼친 결과 SVG 저장")
        btn_export_flat_svg.setToolTip("평면화(Flatten) 결과의 외곽선을 실측 SVG로 저장합니다")
        btn_export_flat_svg.clicked.connect(lambda: self.exportRequested.emit({'type': 'flat_svg'}))
        mesh_layout.addWidget(btn_export_flat_svg)

        btn_export_sheet_svg = QPushButton("통합 SVG (실측+단면+내/외면 탁본)")
        btn_export_sheet_svg.setToolTip("Top outline + cut lines/sections + outer/inner rubbing in one SVG")
        btn_export_sheet_svg.clicked.connect(lambda: self.exportRequested.emit({'type': 'sheet_svg'}))
        mesh_layout.addWidget(btn_export_sheet_svg)
        
        layout.addWidget(mesh_group)
        
        # 2D 외곽선 내보내기 (SVG/PDF)
        profile_group = QGroupBox("🛡️ 2D 실측 도면 내보내기 (SVG)")
        profile_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2b6cb0; }")
        profile_layout = QVBoxLayout(profile_group)
        
        # 안내 문구
        lbl_info = QLabel("격자는 이미지, 외곽선은 벡터로 저장됩니다.\n(지정된 뷰 방향에서 투영)")
        lbl_info.setStyleSheet("font-size: 11px; color: #718096;")
        profile_layout.addWidget(lbl_info)
        
        # 6방향 버튼 그리드
        grid_layout = QGridLayout()
        views = [
            ('Top (상면)', 'top'), ('Bottom (하면)', 'bottom'),
            ('Front (정면)', 'front'), ('Back (후면)', 'back'),
            ('Left (좌측)', 'left'), ('Right (우측)', 'right')
        ]
        
        for i, (label, view_code) in enumerate(views):
            btn = QPushButton(label)
            btn.setStyleSheet("text-align: left; padding: 5px;")
            btn.clicked.connect(
                lambda checked, v=view_code: self.exportRequested.emit(
                    {"type": "profile_2d", "view": v}
                )
            )
            grid_layout.addWidget(btn, i // 2, i % 2)
            
        profile_layout.addLayout(grid_layout)
        layout.addWidget(profile_group)
        
        layout.addStretch()


class SectionPanel(QWidget):
    crosshairToggled = pyqtSignal(bool)
    lineSectionToggled = pyqtSignal(bool)
    cutLineActiveChanged = pyqtSignal(int)
    cutLineClearRequested = pyqtSignal(int)
    cutLinesClearAllRequested = pyqtSignal()
    saveSectionLayersRequested = pyqtSignal()
    roiToggled = pyqtSignal(bool)
    silhouetteRequested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. 활성화 버튼
        self.btn_toggle = QPushButton("🎯 십자선 단면 모드 시작")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setStyleSheet("""
            QPushButton:checked {
                background-color: #f6e05e;
                font-weight: bold;
            }
        """)
        self.btn_toggle.toggled.connect(self.on_btn_toggled)
        layout.addWidget(self.btn_toggle)
        
        # 2. 도움말
        help_label = QLabel("모드 활성 후 메쉬를 클릭/드래그하여 단면을 확인하세요.")
        help_label.setStyleSheet("color: #718096; font-size: 10px;")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        
        # 3. 그래프 공간
        self.label_x = QLabel("X-Profile (Yellow Line)")
        layout.addWidget(self.label_x)
        self.graph_x = ProfileGraphWidget("가로 단면 (X-Profile)")
        layout.addWidget(self.graph_x)
        
        self.label_y = QLabel("Y-Profile (Cyan Line)")
        layout.addWidget(self.label_y)
        self.graph_y = ProfileGraphWidget("세로 단면 (Y-Profile)")
        layout.addWidget(self.graph_y)
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # 4. 단면선(2개) - 상면에서 가로/세로(꺾임 가능) 가이드 라인
        line_group = QGroupBox("✏️ 단면선 (2개)")
        line_layout = QVBoxLayout(line_group)

        self.btn_line = QPushButton("✏️ 단면선 그리기 시작")
        self.btn_line.setCheckable(True)
        self.btn_line.setStyleSheet(
            "QPushButton:checked { background-color: #ed8936; "
            "color: white; font-weight: bold; }"
        )
        self.btn_line.toggled.connect(self.on_line_toggled)
        line_layout.addWidget(self.btn_line)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("활성 선:"))
        self.combo_cutline = QComboBox()
        self.combo_cutline.addItems(["가로(1)", "세로(2)"])
        self.combo_cutline.currentIndexChanged.connect(self.cutLineActiveChanged.emit)
        sel_row.addWidget(self.combo_cutline, 1)

        self.btn_cutline_clear = QPushButton("🧹 현재 선 지우기")
        self.btn_cutline_clear.clicked.connect(
            lambda: self.cutLineClearRequested.emit(int(self.combo_cutline.currentIndex()))
        )
        sel_row.addWidget(self.btn_cutline_clear)

        self.btn_cutline_clear_all = QPushButton("🧹 모두 지우기")
        self.btn_cutline_clear_all.clicked.connect(self.cutLinesClearAllRequested.emit)
        sel_row.addWidget(self.btn_cutline_clear_all)
        line_layout.addLayout(sel_row)

        line_help = QLabel(
            "상면(Top) 뷰에서 좌클릭으로 점을 추가해 단면선(꺾인 폴리라인)을 그리세요. (자동 수평/수직)\n"
            "Enter/우클릭=현재 선 확정, Backspace/Delete=마지막 점 취소, Tab=선 전환\n"
            "가로/세로는 각각 1개 선만 유지됩니다.\n"
            "Shift/Ctrl/Alt + 드래그: 메쉬 이동/회전 (점 추가 안 됨)"
        )
        line_help.setStyleSheet("color: #718096; font-size: 10px;")
        line_help.setWordWrap(True)
        line_layout.addWidget(line_help)

        self.btn_save_section_layers = QPushButton("단면을 레이어로 저장")
        self.btn_save_section_layers.setToolTip("현재 단면선/단면 결과를 레이어로 스냅샷 저장합니다.")
        self.btn_save_section_layers.clicked.connect(self.saveSectionLayersRequested.emit)
        line_layout.addWidget(self.btn_save_section_layers)

        layout.addWidget(line_group)

        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line2)
        
        # 5. 2D ROI 영역 지정 (NEW)
        roi_group = QGroupBox("✂️ 2D 영역 지정 (Cropping)")
        roi_layout = QVBoxLayout(roi_group)
        
        self.btn_roi = QPushButton("📐 영역 지정 모드 시작")
        self.btn_roi.setCheckable(True)
        self.btn_roi.setStyleSheet("QPushButton:checked { background-color: #4299e1; color: white; }")
        self.btn_roi.toggled.connect(self.on_roi_toggled)
        roi_layout.addWidget(self.btn_roi)
        
        self.btn_silhouette = QPushButton("✅ 영역 확정 및 외곽 추출")
        self.btn_silhouette.setEnabled(False)
        self.btn_silhouette.clicked.connect(self.silhouetteRequested.emit)
        roi_layout.addWidget(self.btn_silhouette)
        
        roi_help = QLabel("상면(Top) 뷰에서 4개 화살표를 드래그하여 영역을 지정하세요.")
        roi_help.setStyleSheet("color: #718096; font-size: 10px;")
        roi_help.setWordWrap(True)
        roi_layout.addWidget(roi_help)
        
        layout.addWidget(roi_group)
        
        layout.addStretch()
        
    def on_btn_toggled(self, checked):
        if checked:
            self.btn_toggle.setText("🎯 십자선 단면 모드 중지")
        else:
            self.btn_toggle.setText("🎯 십자선 단면 모드 시작")
        self.crosshairToggled.emit(checked)

    def on_line_toggled(self, checked):
        if checked:
            self.btn_line.setText("✏️ 단면선 그리기 중지")
        else:
            self.btn_line.setText("✏️ 단면선 그리기 시작")
        self.lineSectionToggled.emit(checked)
        
    def on_roi_toggled(self, checked):
        if checked:
            self.btn_roi.setText("📐 영역 지정 모드 중지")
            self.btn_silhouette.setEnabled(True)
        else:
            self.btn_roi.setText("📐 영역 지정 모드 시작")
            self.btn_silhouette.setEnabled(False)
        self.roiToggled.emit(checked)
        
    def update_profiles(self, x_data, y_data):
        self.graph_x.set_data(x_data)
        self.graph_y.set_data(y_data)

    def update_line_profile(self, line_data):
        # 호환 유지: 이전 '직선 단면' 그래프는 더 이상 사용하지 않음
        pass


class MainWindow(QMainWindow):
    """메인 윈도우"""

    UI_STATE_VERSION = 3
    
    def __init__(self):
        super().__init__()
        
        sha, dirty = _safe_git_info(str(Path(basedir)))
        sha_s = f"{sha}{'*' if dirty else ''}" if sha else "unknown"
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} ({sha_s})")
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

        self._mesh_load_dialog: QProgressDialog | None = None
        self._mesh_load_thread: MeshLoadThread | None = None
        self._profile_export_dialog: QProgressDialog | None = None
        self._profile_export_thread: ProfileExportThread | None = None
        self._task_dialog: QProgressDialog | None = None
        self._task_thread: TaskThread | None = None

        # 평면화(Flatten) 결과 캐시: (obj id + transform + options) -> FlattenedMesh
        self._flattened_cache = {}

        # Slice(CT) 계산은 디바운스 + 백그라운드 스레드로 처리 (UI 끊김 방지)
        self._slice_debounce_timer = QTimer(self)
        self._slice_debounce_timer.setSingleShot(True)
        self._slice_debounce_timer.timeout.connect(self._request_slice_compute)
        self._slice_compute_thread = None
        self._slice_pending_height = None
        
        self.init_ui()
        self.init_menu()
        self.init_toolbar()
        self.init_statusbar()
        self._restore_ui_state()
    
    def init_ui(self):
        # 중앙 위젯 (3D 뷰포트)
        self.viewport = Viewport3D()
        self.setCentralWidget(self.viewport)
        
        # 씬 매니저 연결
        self.viewport.selectionChanged.connect(self.on_selection_changed)
        self.viewport.meshLoaded.connect(self.on_mesh_loaded)
        self.viewport.meshTransformChanged.connect(self.sync_transform_panel)
        self.viewport.floorPointPicked.connect(self.on_floor_point_picked)
        self.viewport.floorFacePicked.connect(self.on_floor_face_picked)
        self.viewport.alignToBrushSelected.connect(self.on_align_to_brush_selected)
        self.viewport.floorAlignmentConfirmed.connect(self.on_floor_alignment_confirmed)
        self.viewport.surfaceAssignmentChanged.connect(self.on_surface_assignment_changed)
        
        # 단축키 설정 (Undo: Ctrl+Z)
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.viewport.undo)
        
        # 상단 정치 툴바 추가
        self.trans_toolbar = TransformToolbar(self.viewport, self)
        self.trans_toolbar.setObjectName("toolbar_transform")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.trans_toolbar)
        
        # 툴바 신호 연결
        self.trans_toolbar.trans_x.valueChanged.connect(self.on_toolbar_transform_changed)
        self.trans_toolbar.trans_y.valueChanged.connect(self.on_toolbar_transform_changed)
        self.trans_toolbar.trans_z.valueChanged.connect(self.on_toolbar_transform_changed)
        self.trans_toolbar.rot_x.valueChanged.connect(self.on_toolbar_transform_changed)
        self.trans_toolbar.rot_y.valueChanged.connect(self.on_toolbar_transform_changed)
        self.trans_toolbar.rot_z.valueChanged.connect(self.on_toolbar_transform_changed)
        self.trans_toolbar.scale_spin.valueChanged.connect(self.on_toolbar_transform_changed)
        
        self.trans_toolbar.btn_bake.clicked.connect(self.on_bake_all_clicked)
        self.trans_toolbar.btn_fixed.clicked.connect(self.restore_fixed_state)
        self.trans_toolbar.btn_reset.clicked.connect(self.reset_transform)
        self.trans_toolbar.btn_flat.toggled.connect(self.toggle_flat_shading)
        
        # 도움말 위젯 (오버레이처럼 작동하도록 뷰포트 위에 띄우거나 하단에 배치 가능)
        # 일단은 뷰포트 하단에 고정
        self.help_widget = HelpWidget()
        self.help_dock = QDockWidget("❓ 도움말", self)
        self.help_dock.setObjectName("dock_help")
        self.help_dock.setWidget(self.help_widget)
        try:
            self.help_dock.setMinimumHeight(100)
        except Exception:
            pass
        try:
            self._help_dock_last_floating = True
            self.help_dock.topLevelChanged.connect(self._on_help_dock_top_level_changed)
        except Exception:
            self._help_dock_last_floating = True
        self.action_toggle_help_panel = self.help_dock.toggleViewAction()
        if self.action_toggle_help_panel is None:
            self.action_toggle_help_panel = QAction("❓ 도움말", self)
            self.action_toggle_help_panel.setCheckable(True)
            self.action_toggle_help_panel.toggled.connect(self._on_help_panel_toggled)
            try:
                self.help_dock.visibilityChanged.connect(self.action_toggle_help_panel.setChecked)
            except Exception:
                pass
        else:
            self.action_toggle_help_panel.setText("❓ 도움말")
            self.action_toggle_help_panel.setToolTip("도움말 창 표시/숨김")
            try:
                self.action_toggle_help_panel.toggled.connect(self._on_help_panel_toggled)
            except Exception:
                pass

        # 도킹 위젯 설정
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AllowNestedDocks
        )
        self.setDockNestingEnabled(True)

        # 1) 상단 정보(파일/메쉬)
        self.info_dock = QDockWidget("📄 파일/메쉬 정보", self)
        self.info_dock.setObjectName("dock_info")
        self.props_panel = InfoBarWidget()
        self.info_dock.setWidget(self.props_panel)

        # 2) 정치(변환)
        self.transform_dock = QDockWidget("📐 정치 (변환)", self)
        self.transform_dock.setObjectName("dock_transform")
        self.transform_panel = TransformPanel(self.viewport, self.help_widget)
        self.transform_dock.setWidget(self.transform_panel)

        # 3) 펼침
        self.flatten_dock = QDockWidget("🗺️ 펼침 (Flatten)", self)
        self.flatten_dock.setObjectName("dock_flatten")
        self.flatten_panel = FlattenPanel(self.help_widget)
        self.flatten_panel.flattenRequested.connect(self.on_flatten_requested)
        self.flatten_panel.selectionRequested.connect(self.on_selection_action)
        self.flatten_panel.btn_measure.toggled.connect(self.toggle_curvature_mode)
        self.flatten_panel.btn_fit_arc.clicked.connect(self.fit_curvature_arc)
        self.flatten_panel.btn_clear_points.clicked.connect(self.clear_curvature_points)
        self.flatten_panel.btn_clear_arcs.clicked.connect(self.clear_all_arcs)
        self.flatten_dock.setWidget(self.flatten_panel)

        # 4) 내보내기
        self.export_dock = QDockWidget("📤 내보내기", self)
        self.export_dock.setObjectName("dock_export")
        self.export_panel = ExportPanel()
        self.export_panel.exportRequested.connect(self.on_export_requested)
        self.export_dock.setWidget(self.export_panel)

        # 5) 단면 도구 (슬라이싱 + 십자선 + 라인)
        self.section_dock = QDockWidget("📏 단면 도구 (Section)", self)
        self.section_dock.setObjectName("dock_section")
        section_scroll = QScrollArea()
        section_scroll.setWidgetResizable(True)
        section_content = QWidget()
        section_layout = QVBoxLayout(section_content)

        self.slice_panel = SlicingPanel()
        self.slice_panel.sliceChanged.connect(self.on_slice_changed)
        self.slice_panel.exportRequested.connect(self.on_slice_export_requested)
        section_layout.addWidget(self.slice_panel)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        section_layout.addWidget(line)

        self.section_panel = SectionPanel()
        self.section_panel.crosshairToggled.connect(self.on_crosshair_toggled)
        self.section_panel.lineSectionToggled.connect(self.on_line_section_toggled)
        self.section_panel.cutLineActiveChanged.connect(self.on_cut_line_active_changed)
        self.section_panel.cutLineClearRequested.connect(self.on_cut_line_clear_requested)
        self.section_panel.cutLinesClearAllRequested.connect(self.on_cut_lines_clear_all_requested)
        self.section_panel.roiToggled.connect(self.on_roi_toggled)
        self.section_panel.silhouetteRequested.connect(self.viewport.extract_roi_silhouette)
        self.section_panel.saveSectionLayersRequested.connect(self.on_save_section_layers_requested)

        self.viewport.profileUpdated.connect(self.section_panel.update_profiles)
        self.viewport.lineProfileUpdated.connect(self.section_panel.update_line_profile)
        self.viewport.roiSilhouetteExtracted.connect(self.on_silhouette_extracted)
        self.viewport.cutLinesAutoEnded.connect(self._on_cut_lines_auto_ended)
        section_layout.addWidget(self.section_panel)

        section_layout.addStretch()
        section_scroll.setWidget(section_content)
        self.section_dock.setWidget(section_scroll)

        # 7) 씬(레이어)
        self.scene_dock = QDockWidget("🌲 씬 (레이어)", self)
        self.scene_dock.setObjectName("dock_scene")
        self.scene_panel = ScenePanel()
        self.scene_panel.selectionChanged.connect(self.viewport.select_object)
        self.scene_panel.visibilityChanged.connect(self.on_visibility_changed)
        self.scene_panel.arcDeleted.connect(self.on_arc_deleted)
        self.scene_panel.layerVisibilityChanged.connect(self.on_layer_visibility_changed)
        self.scene_panel.layerDeleted.connect(self.on_layer_deleted)
        self.scene_panel.layerMoveRequested.connect(self.on_layer_move_requested)
        self.scene_panel.layerOffsetResetRequested.connect(self.on_layer_offset_reset_requested)
        self.scene_dock.setWidget(self.scene_panel)

        # 공통 도킹/플로팅 옵션
        for dock in [
            self.info_dock,
            self.transform_dock,
            self.flatten_dock,
            self.section_dock,
            self.export_dock,
            self.scene_dock,
            self.help_dock,
        ]:
            dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
            dock.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                | QDockWidget.DockWidgetFeature.DockWidgetClosable
            )

        # 기본 레이아웃(일러스트레이터 스타일: 상단 정보/정치, 우측 분리, 씬은 우측 하단)
        self._apply_default_dock_layout()

    def _settings(self) -> QSettings:
        return QSettings("ArchMeshRubbing", "ArchMeshRubbing")

    def _apply_default_dock_layout(self):
        """기본 도킹 레이아웃 적용 (저장된 레이아웃이 없을 때의 초기 배치)"""
        for dock in [
            self.info_dock,
            self.transform_dock,
            self.flatten_dock,
            self.section_dock,
            self.export_dock,
            self.scene_dock,
            self.help_dock,
        ]:
            # 기존 배치가 남아있으면(중복 split/tabify 등) 레이아웃이 꼬일 수 있어 초기화
            try:
                self.removeDockWidget(dock)
            except Exception:
                pass
            dock.setFloating(False)
            if dock is self.help_dock:
                dock.hide()
            else:
                dock.show()

        # 상단: 파일/메쉬 정보 + 정치(변환) (가로 배치)
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.info_dock)
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.transform_dock)
        self.splitDockWidget(self.info_dock, self.transform_dock, Qt.Orientation.Horizontal)

        # 우측: 펼침 + 단면(도구) + 내보내기는 탭, 씬은 우측 하단
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.flatten_dock)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.section_dock)
        self.tabifyDockWidget(self.flatten_dock, self.section_dock)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.export_dock)
        self.tabifyDockWidget(self.flatten_dock, self.export_dock)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.scene_dock)
        self.splitDockWidget(self.flatten_dock, self.scene_dock, Qt.Orientation.Vertical)

        # 하단: 컨텍스트 도움말(선택/툴 사용법)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.help_dock)
        self.help_dock.hide()

        # 크기 비율(대략적인 기본값)
        self.resizeDocks([self.info_dock, self.transform_dock], [650, 750], Qt.Orientation.Horizontal)
        self.resizeDocks([self.flatten_dock, self.scene_dock], [780, 220], Qt.Orientation.Vertical)

        self.flatten_dock.raise_()

    def _restore_ui_state(self):
        settings = self._settings()
        stored_version = settings.value("ui/state_version")
        if stored_version is not None:
            try:
                stored_version = int(stored_version)
            except (TypeError, ValueError):
                stored_version = None

        # 버전이 다르면(레이아웃 구조 변경 등) 기존 저장값 무시
        if stored_version is not None and stored_version != self.UI_STATE_VERSION:
            return

        geometry = settings.value("ui/geometry")
        state = settings.value("ui/state")

        if geometry is not None:
            try:
                self.restoreGeometry(geometry)
            except Exception:
                pass
        if state is not None:
            try:
                self.restoreState(state, self.UI_STATE_VERSION)
            except Exception:
                pass

    def _save_ui_state(self):
        settings = self._settings()
        settings.setValue("ui/state_version", self.UI_STATE_VERSION)
        settings.setValue("ui/geometry", self.saveGeometry())
        settings.setValue("ui/state", self.saveState(self.UI_STATE_VERSION))

    def reset_panel_layout(self):
        """사용자 레이아웃 저장값 삭제 후 기본 레이아웃으로 복구"""
        settings = self._settings()
        settings.remove("ui/geometry")
        settings.remove("ui/state")
        settings.remove("ui/state_version")
        self._apply_default_dock_layout()

    def closeEvent(self, a0):
        self._save_ui_state()
        if a0 is None:
            return
        super().closeEvent(a0)

    def start_floor_picking(self):
        """바닥면 그리기(점 찍기) 모드 시작"""
        if self.viewport.selected_obj is None:
            return
        self.viewport.picking_mode = 'floor_3point'
        self.viewport.floor_picks = []
        self.viewport.status_info = "📍 바닥면 점 찍기: 메쉬 위를 클릭하여 점을 추가하세요 (Enter로 확정)"
        self.viewport.update()

    def start_floor_picking_face(self):
        """면 선택 바닥 정렬 모드 시작"""
        if self.viewport.selected_obj is None:
            return
        self.viewport.picking_mode = 'floor_face'
        self.viewport.status_info = "📐 바닥면이 될 삼각형 면(Triangle)을 클릭하세요..."
        self.viewport.update()

    def start_floor_picking_brush(self):
        """브러시 바닥 정렬 모드 시작"""
        if self.viewport.selected_obj is None:
            return
        self.viewport.picking_mode = 'floor_brush'
        self.viewport.brush_selected_faces.clear()
        self.viewport.status_info = "🖌️ 바닥이 될 영역을 마우스 왼쪽 버튼으로 드래그하듯이 그리세요..."
        self.viewport.update()

    def on_align_to_brush_selected(self):
        """브러시로 선택된 영역의 평균 법선으로 정렬"""
        obj = self.viewport.selected_obj
        if not obj or not self.viewport.brush_selected_faces:
            return
            
        faces = obj.mesh.faces
        vertices = obj.mesh.vertices
        
        total_normal = np.array([0.0, 0.0, 0.0])
        total_area = 0.0
        
        for face_idx in self.viewport.brush_selected_faces:
            f = faces[face_idx]
            v0 = vertices[f[0]]
            v1 = vertices[f[1]]
            v2 = vertices[f[2]]
            
            n = np.cross(v1 - v0, v2 - v0)
            area = np.linalg.norm(n) / 2.0
            if area > 1e-9:
                total_normal += n # n의 길이가 area*2이므로 가중 합산됨
                total_area += area
        
        if total_area < 1e-9:
            self.viewport.status_info = "❌ 유효한 면이 선택되지 않았습니다."
            self.viewport.update()
            return
            
        avg_normal = total_normal / np.linalg.norm(total_normal)
        self.align_mesh_to_normal(avg_normal)
        
        count = len(self.viewport.brush_selected_faces)
        self.viewport.brush_selected_faces.clear()
        self.viewport.status_info = f"✅ 브러시 영역({count}개 면) 기준 바닥 정렬 완료"
        self.viewport.update()

    def align_mesh_to_normal(self, normal):
        """주어진 법선 벡터를 월드 Z축(0,0,1)으로 정렬 (Bake)"""
        obj = self.viewport.selected_obj
        if not obj:
            return
        
        target = np.array([0.0, 0.0, 1.0])
        axis = np.cross(normal, target)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm > 1e-6:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0))
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

            obj.mesh.vertices = (R @ obj.mesh.vertices.T).T
            try:
                obj.mesh._bounds = None
                obj.mesh._centroid = None
                obj.mesh._surface_area = None
            except Exception:
                pass
            obj.mesh.compute_normals(compute_vertex_normals=False, force=True)
            obj._trimesh = None
            obj.rotation = np.array([0.0, 0.0, 0.0])
            self.viewport.update_vbo(obj)
            self.sync_transform_panel()
            return R
        return np.eye(3)

    def on_floor_face_picked(self, vertices):
        """바닥면(면 선택) - Enter를 눌러야 정렬됨"""
        if len(vertices) != 3:
            return
        self.viewport.floor_picks = [v.copy() for v in vertices]
        self.viewport.status_info = "✅ 면 선택됨. Enter를 누르면 정렬됩니다."
        self.viewport.update()

    def on_floor_point_picked(self, point):
        """바닥면 점 선택 - 점이 추가되면 상태바 업데이트"""
        obj = self.viewport.selected_obj
        if not obj:
            return
        
        if not hasattr(self.viewport, 'floor_picks'):
            self.viewport.floor_picks = []
        
        # 중복 방지
        if not any(np.array_equal(point, p) for p in self.viewport.floor_picks):
            self.viewport.floor_picks.append(point.copy())
            
        count = len(self.viewport.floor_picks)
        
        if count < 3:
            self.viewport.status_info = f"📍 바닥면 점 찍기 (현재 {count}개 선택됨, 더 찍어주세요)..."
        else:
            self.viewport.status_info = f"✅ 점 {count}개 선택됨. 계속 추가하거나 Enter로 확정하세요."
        
        self.viewport.update()

    def on_floor_alignment_confirmed(self):
        """Enter 키 입력 시 호출: 선택된 점들을 기반으로 평면 정렬 수행"""
        obj = self.viewport.selected_obj
        if not obj or not self.viewport.floor_picks:
            return

        points = np.array(self.viewport.floor_picks)
        if len(points) < 3:
            self.viewport.status_info = "❌ 점이 부족합니다. 더 찍어주세요."
            self.viewport.update()
            return
            
        # 1. 메쉬 정치 확정 (Bake)
        # 선택된 점들이 로컬 좌표계이므로, 현재 메쉬의 모든 변환을 정점에 미리 적용해둠
        self.viewport.bake_object_transform(obj)
        
        # 2. 평면 피팅
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        _, _, vh = np.linalg.svd(centered_points)
        normal = vh[2, :] # 초기 법선 (방향은 아직 불확실)
        
        # 3. 정렬 수행
        self.viewport.save_undo_state()
        R = self.align_mesh_to_normal(normal)
        
        # 4. 상하 반전 체크 (Bulk-Height Comparison)
        if R is not None:
            # 회전 후 찍은 점들의 평균 Z
            points_rotated = (R @ points.T).T
            avg_pick_z = np.mean(points_rotated[:, 2])
            
            # 회전 후 전체 메쉬의 평균 Z
            avg_mesh_z = np.mean(obj.mesh.vertices[:, 2])
            
            # 메쉬 몸통(평균)이 찍은 점들보다 낮으면 upside-down 상태
            if avg_mesh_z < avg_pick_z:
                # 180도 추가 회전 (X축 기준)
                R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                obj.mesh.vertices = (R_flip @ obj.mesh.vertices.T).T
                try:
                    obj.mesh._bounds = None
                    obj.mesh._centroid = None
                    obj.mesh._surface_area = None
                except Exception:
                    pass
                obj.mesh.compute_normals(compute_vertex_normals=False, force=True)
                obj._trimesh = None
                self.viewport.update_vbo(obj)
        
        # 5. 바닥 높이 맞춤 (가라앉지 않도록 Z >= 0 보장)
        if R is not None:
            min_z = obj.mesh.vertices[:, 2].min()
            obj.mesh.vertices[:, 2] -= min_z
            try:
                obj.mesh._bounds = None
                obj.mesh._centroid = None
            except Exception:
                pass
            obj._trimesh = None
            obj.translation[2] = 0

            self.viewport.update_vbo(obj)
            self.sync_transform_panel()
            self.viewport.status_info = f"✅ 바닥 정렬 완료 (점 {len(points)}개 기반 평면 보정)"
            self.viewport.update()
        
        self.viewport.floor_picks = []
        self.viewport.picking_mode = 'none'
        self.viewport.update()
        self.viewport.meshTransformChanged.emit()

    def on_arc_deleted(self, obj_idx, arc_idx):
        """특정 객체의 특정 원호 삭제"""
        if 0 <= obj_idx < len(self.viewport.objects):
            obj = self.viewport.objects[obj_idx]
            if 0 <= arc_idx < len(obj.fitted_arcs):
                del obj.fitted_arcs[arc_idx]
                self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
                self.viewport.update()
                self.status_info.setText(f"🗑️ 원호 #{arc_idx+1} 삭제됨")
    
    def on_layer_visibility_changed(self, obj_idx: int, layer_idx: int, visible: bool):
        try:
            self.viewport.set_polyline_layer_visible(int(obj_idx), int(layer_idx), bool(visible))
            self.viewport.update()
        except Exception:
            pass

    def on_layer_deleted(self, obj_idx: int, layer_idx: int):
        try:
            self.viewport.delete_polyline_layer(int(obj_idx), int(layer_idx))
            self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
            self.viewport.update()
            self.status_info.setText("레이어 삭제됨")
        except Exception:
            pass

    def on_layer_move_requested(self, obj_idx: int, layer_idx: int, dx: float, dy: float):
        try:
            self.viewport.move_polyline_layer(int(obj_idx), int(layer_idx), float(dx), float(dy))
            self.viewport.update()
        except Exception:
            pass

    def on_layer_offset_reset_requested(self, obj_idx: int, layer_idx: int):
        try:
            self.viewport.reset_polyline_layer_offset(int(obj_idx), int(layer_idx))
            self.viewport.update()
        except Exception:
            pass

    def init_menu(self):
        menubar = self.menuBar()
        if menubar is None:
            return
        
        # 파일 메뉴
        file_menu = menubar.addMenu("파일(&F)")
        if file_menu is None:
            return
        
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
        if view_menu is None:
            return
        
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
        action_front.triggered.connect(lambda: self.set_view(-90, 0))
        view_menu.addAction(action_front)
        
        action_back = QAction("2️⃣ 후면 뷰", self)
        action_back.setShortcut("2")
        action_back.triggered.connect(lambda: self.set_view(90, 0))
        view_menu.addAction(action_back)
        
        action_right = QAction("3️⃣ 우측면 뷰", self)
        action_right.setShortcut("3")
        action_right.triggered.connect(lambda: self.set_view(0, 0))
        view_menu.addAction(action_right)
        
        action_left = QAction("4️⃣ 좌측면 뷰", self)
        action_left.setShortcut("4")
        action_left.triggered.connect(lambda: self.set_view(180, 0))
        view_menu.addAction(action_left)
        
        action_top = QAction("5️⃣ 상면 뷰", self)
        action_top.setShortcut("5")
        action_top.triggered.connect(lambda: self.set_view(0, 89))
        view_menu.addAction(action_top)
        
        action_bottom = QAction("6️⃣ 하면 뷰", self)
        action_bottom.setShortcut("6")
        action_bottom.triggered.connect(lambda: self.set_view(0, -89))
        view_menu.addAction(action_bottom)

        view_menu.addSeparator()

        action_reset_layout = QAction("패널 레이아웃 초기화", self)
        action_reset_layout.triggered.connect(self.reset_panel_layout)
        view_menu.addAction(action_reset_layout)

        panels_menu = view_menu.addMenu("패널 표시/숨김")
        if panels_menu is not None:
            panels_menu.addAction(self.info_dock.toggleViewAction())
            panels_menu.addAction(self.transform_dock.toggleViewAction())
            panels_menu.addAction(self.flatten_dock.toggleViewAction())
            panels_menu.addAction(self.section_dock.toggleViewAction())
            panels_menu.addAction(self.export_dock.toggleViewAction())
            panels_menu.addAction(self.scene_dock.toggleViewAction())
            panels_menu.addAction(self.action_toggle_help_panel)
        
        # 도움말 메뉴
        help_menu = menubar.addMenu("도움말(&H)")
        if help_menu is not None:
            action_about = QAction("ℹ️ 정보(&A)", self)
            action_about.triggered.connect(self.show_about)
            help_menu.addAction(action_about)

            action_debug = QAction("디버그 정보 복사", self)
            action_debug.setToolTip("실행 중인 코드/버전/모듈 경로 정보를 클립보드로 복사합니다.")
            action_debug.triggered.connect(self.copy_debug_info)
            help_menu.addAction(action_debug)

    def _on_help_dock_top_level_changed(self, floating: bool) -> None:
        try:
            self._help_dock_last_floating = bool(floating)
        except Exception:
            pass

    def _on_help_panel_toggled(self, checked: bool) -> None:
        try:
            if checked:
                self.help_dock.show()
                prefer_floating = bool(getattr(self, "_help_dock_last_floating", True))
                if prefer_floating:
                    try:
                        self.help_dock.setFloating(True)
                    except Exception:
                        pass
                    try:
                        self.help_dock.resize(560, 260)
                    except Exception:
                        pass
                    try:
                        g = self.geometry()
                        x = int(g.x() + g.width() - self.help_dock.width() - 20)
                        y = int(g.y() + g.height() - self.help_dock.height() - 60)
                        self.help_dock.move(max(0, x), max(0, y))
                    except Exception:
                        pass
                try:
                    self.help_dock.raise_()
                except Exception:
                    pass
            else:
                self.help_dock.hide()
        except Exception:
            pass

    def init_toolbar(self):
        toolbar = QToolBar("메인 툴바")
        toolbar.setObjectName("toolbar_main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        action_open = QAction("📂 열기", self)
        action_open.triggered.connect(self.open_file)
        toolbar.addAction(action_open)

        toolbar.addSeparator()

        action_fit = QAction("🔍 뷰 맞춤", self)
        action_fit.setToolTip("메쉬가 화면에 꽉 차도록 카메라 조정")
        action_fit.triggered.connect(self.fit_view)
        toolbar.addAction(action_fit)

        action_draw_floor = QAction("✏️ 바닥 면 그리기", self)
        action_draw_floor.setToolTip("바닥면이 될 점들을 클릭하여 바닥면 지정을 시작 (Enter로 확정)")
        action_draw_floor.triggered.connect(self.start_floor_picking)
        toolbar.addAction(action_draw_floor)


        toolbar.addSeparator()
        
        # 6방향 뷰 버튼
        action_front = QAction("정면", self)
        action_front.setToolTip("정면 뷰 (1)")
        action_front.triggered.connect(lambda: self.set_view(-90, 0))
        toolbar.addAction(action_front)
        
        action_back = QAction("후면", self)
        action_back.setToolTip("후면 뷰 (2)")
        action_back.triggered.connect(lambda: self.set_view(90, 0))
        toolbar.addAction(action_back)
        
        action_right = QAction("우측", self)
        action_right.setToolTip("우측면 뷰 (3)")
        action_right.triggered.connect(lambda: self.set_view(0, 0))
        toolbar.addAction(action_right)
        
        action_left = QAction("좌측", self)
        action_left.setToolTip("좌측면 뷰 (4)")
        action_left.triggered.connect(lambda: self.set_view(180, 0))
        toolbar.addAction(action_left)
        
        action_top = QAction("상면", self)
        action_top.setToolTip("상면 뷰 (5)")
        action_top.triggered.connect(lambda: self.set_view(0, 89))
        toolbar.addAction(action_top)
        
        action_bottom = QAction("하면", self)
        action_bottom.setToolTip("하면 뷰 (6)")
        action_bottom.triggered.connect(lambda: self.set_view(0, -89))
        toolbar.addAction(action_bottom)

        toolbar.addSeparator()
        toolbar.addAction(self.action_toggle_help_panel)

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
        sha, dirty = _safe_git_info(str(Path(basedir)))
        sha_s = f"{sha}{'*' if dirty else ''}" if sha else "unknown"
        self.status_ver = QLabel(f"v{APP_VERSION} ({sha_s})")
        self.status_ver.setStyleSheet("color: #a0aec0; font-size: 10px; margin-left: 10px;")
        self.statusbar.addPermanentWidget(self.status_ver)

    def copy_debug_info(self) -> None:
        try:
            info = _collect_debug_info(basedir=str(Path(basedir)))
            cb = QApplication.clipboard()
            if cb is not None:
                cb.setText(info)
            QMessageBox.information(self, "디버그 정보", "클립보드에 복사했습니다.\n\n(이 내용과 함께 문제 상황을 알려주시면 재현/디버깅이 빨라집니다.)")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"디버그 정보 생성 실패:\n{type(e).__name__}: {e}")
    
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
    
    def dragEnterEvent(self, a0):
        """드래그 진입 이벤트"""
        if a0 is None:
            return

        mime_data = a0.mimeData()
        if mime_data is None:
            return

        if mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                filepath = urls[0].toLocalFile()
                ext = Path(filepath).suffix.lower()
                if ext in ['.obj', '.ply', '.stl', '.off', '.gltf', '.glb']:
                    a0.acceptProposedAction()
                    return
        a0.ignore()
    
    def dropEvent(self, a0):
        """드롭 이벤트"""
        if a0 is None:
            return

        mime_data = a0.mimeData()
        if mime_data is None:
            return

        urls = mime_data.urls()
        if urls:
            filepath = urls[0].toLocalFile()
            # 드롭 시에도 단위 선택 다이얼로그 표시
            dialog = UnitSelectionDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                scale_factor = dialog.get_scale_factor()
                self.load_mesh(filepath, scale_factor)
    
    def load_mesh(self, filepath: str, scale_factor: float = 1.0):
        self._start_async_load(filepath, scale_factor)
        return
    
    def _start_async_load(self, filepath: str, scale_factor: float):
        thread = getattr(self, "_mesh_load_thread", None)
        if thread is not None and thread.isRunning():
            QMessageBox.information(self, "로딩 중", "이미 다른 메쉬를 로딩 중입니다.")
            return

        name = Path(filepath).name
        self.status_info.setText(f"로딩 중: {name}")
        self.status_mesh.setText("")

        dlg = QProgressDialog(f"메쉬 로딩 중: {name}", None, 0, 0, self)
        dlg.setWindowTitle("로딩")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.show()
        self._mesh_load_dialog = dlg

        self._mesh_load_thread = MeshLoadThread(
            filepath=str(filepath),
            scale_factor=float(scale_factor),
            default_unit=str(getattr(self.mesh_loader, "default_unit", "cm")),
        )
        self._mesh_load_thread.loaded.connect(self._on_mesh_load_thread_loaded)
        self._mesh_load_thread.failed.connect(self._on_mesh_load_thread_failed)
        self._mesh_load_thread.finished.connect(self._on_mesh_load_thread_finished)
        self._mesh_load_thread.start()

    def _on_mesh_load_thread_loaded(self, mesh_data, filepath: str):
        try:
            dlg = getattr(self, "_mesh_load_dialog", None)
            if dlg is not None:
                dlg.setLabelText("장면에 추가하는 중...")
                QApplication.processEvents()

            self.current_mesh = mesh_data
            self.current_filepath = filepath

            self.viewport.add_mesh_object(mesh_data, name=Path(filepath).name)

            self.status_info.setText(f"로드됨: {Path(filepath).name}")
            self.status_mesh.setText(f"V: {mesh_data.n_vertices:,} | F: {mesh_data.n_faces:,}")
            self.status_grid.setText(f"격자: {self.viewport.grid_spacing}cm")
        finally:
            dlg = getattr(self, "_mesh_load_dialog", None)
            if dlg is not None:
                dlg.close()
                self._mesh_load_dialog = None

    def _on_mesh_load_thread_failed(self, message: str):
        dlg = getattr(self, "_mesh_load_dialog", None)
        if dlg is not None:
            dlg.close()
            self._mesh_load_dialog = None

        msg = f"파일 로드 실패:\n{message}"
        try:
            from src.core.logging_utils import format_exception_message

            msg = format_exception_message("파일 로드 실패:", message, log_path=_log_path)
        except Exception:
            pass

        QMessageBox.critical(self, "오류", msg)
        self.status_info.setText("로드 실패")
        self.status_mesh.setText("")

    def _on_mesh_load_thread_finished(self):
        thread = getattr(self, "_mesh_load_thread", None)
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                pass
        self._mesh_load_thread = None

    def _on_profile_export_done(self, result_path: str):
        dlg = getattr(self, "_profile_export_dialog", None)
        if dlg is not None:
            dlg.close()
            self._profile_export_dialog = None

        QMessageBox.information(self, "완료", f"2D 도면(SVG)이 저장되었습니다:\n{result_path}")
        try:
            self.status_info.setText(f"내보내기 완료: {Path(result_path).name}")
        except Exception:
            self.status_info.setText("내보내기 완료")

    def _on_profile_export_failed(self, message: str):
        dlg = getattr(self, "_profile_export_dialog", None)
        if dlg is not None:
            dlg.close()
            self._profile_export_dialog = None

        self.status_info.setText("내보내기 실패")
        msg = f"2D 도면(SVG) 내보내기 실패:\n{message}"
        try:
            from src.core.logging_utils import format_exception_message

            msg = format_exception_message("2D 도면(SVG) 내보내기 실패:", message, log_path=_log_path)
        except Exception:
            pass

        QMessageBox.critical(self, "오류", msg)

    def _on_profile_export_finished(self):
        thread = getattr(self, "_profile_export_thread", None)
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                pass
        self._profile_export_thread = None

    def _format_error_message(self, prefix: str, message: str) -> str:
        try:
            from src.core.logging_utils import format_exception_message

            return format_exception_message(prefix, message, log_path=_log_path)
        except Exception:
            return f"{prefix}\n\n{message}"

    def _start_task(
        self,
        *,
        title: str,
        label: str,
        thread: TaskThread,
        on_done: Callable[[Any], None],
        on_failed: Callable[[str], None] | None = None,
    ) -> bool:
        existing = getattr(self, "_task_thread", None)
        if existing is not None and existing.isRunning():
            QMessageBox.information(self, "작업 중", "이미 다른 작업이 진행 중입니다. 완료 후 다시 시도하세요.")
            return False

        dlg = QProgressDialog(str(label), None, 0, 0, self)
        dlg.setWindowTitle(str(title))
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.show()

        self._task_dialog = dlg
        self._task_thread = thread

        def _close_dialog():
            d = getattr(self, "_task_dialog", None)
            if d is not None:
                try:
                    d.close()
                except Exception:
                    pass
                self._task_dialog = None

        def _cleanup_thread():
            t = getattr(self, "_task_thread", None)
            if t is not None:
                try:
                    t.deleteLater()
                except Exception:
                    pass
            self._task_thread = None

        def _default_failed(message: str):
            QMessageBox.critical(self, "오류", self._format_error_message("작업 실패:", message))

        def _safe_invoke(callback: Callable[[Any], None], arg: Any):
            try:
                callback(arg)
            except Exception as e:
                _LOGGER.exception("Task callback failed")
                QMessageBox.critical(
                    self,
                    "오류",
                    self._format_error_message(
                        "내부 오류:",
                        f"{type(e).__name__}: {e}",
                    ),
                )

        thread.done.connect(lambda result: (_close_dialog(), _safe_invoke(on_done, result)))
        thread.failed.connect(
            lambda msg: (_close_dialog(), _safe_invoke(on_failed or _default_failed, msg))
        )
        thread.finished.connect(lambda: (_close_dialog(), _cleanup_thread()))
        thread.start()
        return True

    def on_mesh_loaded(self, mesh):
        self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
        self.props_panel.update_mesh_info(mesh, self.current_filepath)
        self.sync_transform_panel()
        self.update_slice_range()
        
    def on_selection_changed(self, index):
        self.scene_panel.update_list(self.viewport.objects, index)
        self.sync_transform_panel()
        self.update_slice_range()
        try:
            obj = self.viewport.selected_obj
            self.flatten_panel.update_surface_assignment_counts(
                len(getattr(obj, "outer_face_indices", set()) or set()),
                len(getattr(obj, "inner_face_indices", set()) or set()),
                len(getattr(obj, "migu_face_indices", set()) or set()),
            )
        except Exception:
            pass

    def on_surface_assignment_changed(self, outer: int, inner: int, migu: int) -> None:
        try:
            self.flatten_panel.update_surface_assignment_counts(int(outer), int(inner), int(migu))
        except Exception:
            pass

    def update_slice_range(self):
        """현재 선택된 객체의 Z 범위로 슬라이더 업데이트"""
        obj = self.viewport.selected_obj
        if obj and obj.mesh:
            # 대용량 메쉬에서 전체 버텍스 스캔은 느림 -> 월드 바운드로 근사
            try:
                wb = obj.get_world_bounds()
                z_min = float(wb[0][2])
                z_max = float(wb[1][2])
            except Exception:
                z_min = float(obj.mesh.bounds[0][2])
                z_max = float(obj.mesh.bounds[1][2])
            self.slice_panel.update_range(z_min, z_max)
            
    def on_visibility_changed(self, index, visible):
        if 0 <= index < len(self.viewport.objects):
            self.viewport.objects[index].visible = visible
            self.viewport.update()
            
    def sync_transform_panel(self):
        obj = self.viewport.selected_obj
        if not obj: 
            return

        # 고정 상태 버튼 활성/비활성
        try:
            self.trans_toolbar.btn_fixed.setEnabled(bool(getattr(obj, "fixed_state_valid", False)))
        except Exception:
            pass
        
        # 툴바 동기화
        self.trans_toolbar.trans_x.blockSignals(True)
        self.trans_toolbar.trans_y.blockSignals(True)
        self.trans_toolbar.trans_z.blockSignals(True)
        self.trans_toolbar.rot_x.blockSignals(True)
        self.trans_toolbar.rot_y.blockSignals(True)
        self.trans_toolbar.rot_z.blockSignals(True)
        self.trans_toolbar.scale_spin.blockSignals(True)
        
        self.trans_toolbar.trans_x.setValue(obj.translation[0])
        self.trans_toolbar.trans_y.setValue(obj.translation[1])
        self.trans_toolbar.trans_z.setValue(obj.translation[2])
        self.trans_toolbar.rot_x.setValue(obj.rotation[0])
        self.trans_toolbar.rot_y.setValue(obj.rotation[1])
        self.trans_toolbar.rot_z.setValue(obj.rotation[2])
        self.trans_toolbar.scale_spin.setValue(obj.scale)
        
        self.trans_toolbar.trans_x.blockSignals(False)
        self.trans_toolbar.trans_y.blockSignals(False)
        self.trans_toolbar.trans_z.blockSignals(False)
        self.trans_toolbar.rot_x.blockSignals(False)
        self.trans_toolbar.rot_y.blockSignals(False)
        self.trans_toolbar.rot_z.blockSignals(False)
        self.trans_toolbar.scale_spin.blockSignals(False)

    def on_toolbar_transform_changed(self):
        """툴바에서 값이 변경된 경우"""
        obj = self.viewport.selected_obj
        if not obj:
            return
        
        obj.translation = np.array([
            self.trans_toolbar.trans_x.value(),
            self.trans_toolbar.trans_y.value(),
            self.trans_toolbar.trans_z.value()
        ])
        obj.rotation = np.array([
            self.trans_toolbar.rot_x.value(),
            self.trans_toolbar.rot_y.value(),
            self.trans_toolbar.rot_z.value()
        ])
        obj.scale = self.trans_toolbar.scale_spin.value()
        self.viewport.update()
        self.viewport.meshTransformChanged.emit()

    def on_bake_all_clicked(self):
        """현재 변환을 메쉬에 영구 정착 (정치 신청)"""
        obj = self.viewport.selected_obj
        if not obj:
            return
        
        self.viewport.bake_object_transform(obj)
        self.sync_transform_panel() # 툴바 값 리셋됨
        self.viewport.status_info = f"{obj.name} 정치(Bake) 완료. 변환값이 초기화되었습니다."
        self.viewport.update()

    def restore_fixed_state(self):
        """정치 확정 이후의 고정 상태로 복귀"""
        obj = self.viewport.selected_obj
        if not obj:
            return

        self.viewport.restore_fixed_state(obj)
        self.sync_transform_panel()
        self.viewport.status_info = f"{obj.name} 고정 상태로 복귀"

    def toggle_flat_shading(self, enabled):
        """Flat Shading 모드 토글"""
        self.viewport.flat_shading = enabled
        self.viewport.update()

    def reset_transform(self):
        """모든 변환 초기화"""
        obj = self.viewport.selected_obj
        if not obj:
            return
        
        obj.translation = np.array([0.0, 0.0, 0.0])
        obj.rotation = np.array([0.0, 0.0, 0.0])
        obj.scale = 1.0
        self.sync_transform_panel()
        self.viewport.update()
        self.viewport.meshTransformChanged.emit()
    
    def on_selection_action(self, action: str, data):
        action = str(action or "").strip()

        # 1) Surface target / tool switch (no mesh required)
        if action == "surface_target":
            target = str(data or "").strip().lower()
            if target not in {"outer", "inner", "migu"}:
                target = "outer"
            self.viewport._surface_paint_target = target
            self.viewport.status_info = f"✋ 표면 지정 대상: {target} (찍기/브러시 버튼으로 시작)"
            self.viewport.update()
            return

        if action in {"surface_tool", "tool"}:
            tool = ""
            target = "outer"
            try:
                tool = str((data or {}).get("tool", "")).strip().lower()
                target = str((data or {}).get("target", "outer")).strip().lower()
            except Exception:
                tool = ""
                target = "outer"

            if target not in {"outer", "inner", "migu"}:
                target = "outer"
            self.viewport._surface_paint_target = target

            if tool == "click":
                self.viewport.picking_mode = "paint_surface_face"
                try:
                    if not bool(getattr(self.viewport, "cut_lines_enabled", False)):
                        self.viewport.setMouseTracking(False)
                except Exception:
                    pass
                self.viewport.status_info = (
                    f"👆 찍기(자동 확장) [{target}]: 클릭=영역 지정, Shift/Ctrl=추가, Alt=제거 (ESC로 종료)"
                )
            elif tool == "brush":
                self.viewport.picking_mode = "paint_surface_brush"
                try:
                    if not bool(getattr(self.viewport, "cut_lines_enabled", False)):
                        self.viewport.setMouseTracking(False)
                except Exception:
                    pass
                self.viewport.status_info = f"🖌️ 보정(브러시) [{target}]: 드래그=칠하기, Alt=지우기 (ESC로 종료)"
            elif tool == "area":
                self.viewport.picking_mode = "paint_surface_area"
                try:
                    self.viewport.clear_surface_lasso()
                    self.viewport.setMouseTracking(True)
                    self.viewport.setFocus()
                except Exception:
                    pass
                self.viewport.status_info = (
                    f"📐 면적(Area) [{target}]: 메쉬 위 좌클릭=점 추가(드래그=회전), "
                    f"우클릭/Enter=확정, Backspace=되돌리기, Alt=제거 (ESC로 종료)"
                )
            else:
                QMessageBox.information(self, "안내", "선택 도구를 확인할 수 없습니다.")
                return

            self.viewport.update()
            return

        # 2) Actions that need a selected mesh
        obj = self.viewport.selected_obj
        if not obj or not getattr(obj, "mesh", None):
            QMessageBox.warning(self, "경고", "먼저 메쉬를 선택해 주세요.")
            return

        if not hasattr(obj, "outer_face_indices") or obj.outer_face_indices is None:
            obj.outer_face_indices = set()
        if not hasattr(obj, "inner_face_indices") or obj.inner_face_indices is None:
            obj.inner_face_indices = set()
        if not hasattr(obj, "migu_face_indices") or obj.migu_face_indices is None:
            obj.migu_face_indices = set()

        if action == "surface_clear_target":
            target = str(data or "").strip().lower()
            if target not in {"outer", "inner", "migu"}:
                target = "outer"
            if target == "inner":
                obj.inner_face_indices.clear()
            elif target == "migu":
                obj.migu_face_indices.clear()
            else:
                obj.outer_face_indices.clear()
            try:
                self.viewport.clear_surface_paint_points(target)
                self.viewport.clear_surface_lasso()
            except Exception:
                pass
            self.viewport.status_info = f"표면 지정 비움: {target}"
            try:
                self.viewport._emit_surface_assignment_changed(obj)
            except Exception:
                pass

        elif action == "surface_clear_all":
            obj.outer_face_indices.clear()
            obj.inner_face_indices.clear()
            obj.migu_face_indices.clear()
            try:
                self.viewport.clear_surface_paint_points(None)
                self.viewport.clear_surface_lasso()
            except Exception:
                pass
            self.viewport.status_info = "표면 지정 전체 초기화"
            try:
                self.viewport._emit_surface_assignment_changed(obj)
            except Exception:
                pass

        elif action == "auto_surface":
            try:
                from src.core.surface_separator import SurfaceSeparator

                separator = SurfaceSeparator()
                mesh = self._build_world_mesh(obj)
                modifiers = QApplication.keyboardModifiers()
                use_views = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
                result = separator.auto_detect_surfaces(mesh, method="views" if use_views else "normals")
                obj.outer_face_indices = set(int(x) for x in result.outer_face_indices.tolist())
                obj.inner_face_indices = set(int(x) for x in result.inner_face_indices.tolist())

                self.viewport.status_info = (
                    f"✅ 자동 분리 적용({('view' if use_views else 'normal')}): outer {len(obj.outer_face_indices):,} / inner {len(obj.inner_face_indices):,} (현재 메쉬에 저장됨)"
                )
                try:
                    self.viewport._emit_surface_assignment_changed(obj)
                except Exception:
                    pass
                QMessageBox.information(
                    self,
                    "완료",
                    f"자동 분리 결과를 현재 메쉬에 적용했습니다. (파일 저장은 아직 하지 않았습니다.)\n\n"
                    f"- outer(외면): {len(obj.outer_face_indices):,} faces\n"
                    f"- inner(내면): {len(obj.inner_face_indices):,} faces\n\n"
                    f"표시: 외면=파랑, 내면=보라 오버레이\n"
                    f"저장: 내보내기 탭에서 SVG/이미지로 내보내세요.",
                )
            except Exception as e:
                QMessageBox.critical(self, "오류", f"자동 분리 실패:\n{e}")
                return

        elif action == "auto_edge":
            try:
                from src.core.surface_separator import SurfaceSeparator

                mesh_local = getattr(obj, "mesh", None)
                if mesh_local is None:
                    QMessageBox.warning(self, "경고", "먼저 메쉬를 선택해 주세요.")
                    return

                modifiers = QApplication.keyboardModifiers()
                broad_edge = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
                major_axis = "x" if (modifiers & Qt.KeyboardModifier.ControlModifier) else "y"

                # Rotation matrix (local -> world)
                rot_deg = np.asarray(getattr(obj, "rotation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
                if rot_deg.size < 3:
                    rot_deg = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                rx, ry, rz = np.radians(rot_deg[:3])
                cx, sx = float(np.cos(rx)), float(np.sin(rx))
                cy, sy = float(np.cos(ry)), float(np.sin(ry))
                cz, sz = float(np.cos(rz)), float(np.sin(rz))
                rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
                rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
                rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                rot_mat = rot_x @ rot_y @ rot_z

                # Face normals (world)
                try:
                    if getattr(mesh_local, "face_normals", None) is None:
                        mesh_local.compute_normals(compute_vertex_normals=False)
                except Exception:
                    pass
                fn_local = np.asarray(getattr(mesh_local, "face_normals", None), dtype=np.float64)
                if fn_local.ndim != 2 or fn_local.shape[0] != int(getattr(mesh_local, "n_faces", 0) or 0) or fn_local.shape[1] < 3:
                    raise RuntimeError("면 법선(face_normals) 계산에 실패했습니다.")
                fn_world = fn_local[:, :3] @ rot_mat.T

                # Estimate "thickness" direction and rotate to world
                separator = SurfaceSeparator()
                d_local = np.asarray(separator._estimate_reference_direction(mesh_local), dtype=np.float64).reshape(-1)
                if d_local.size < 3 or not np.isfinite(d_local[:3]).all():
                    d_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                d_world = rot_mat @ d_local[:3]
                dn = float(np.linalg.norm(d_world))
                if dn > 1e-12 and np.isfinite(dn):
                    d_world = d_world / dn
                else:
                    d_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)

                abs_dot = np.abs(fn_world @ d_world.reshape(3,))

                if broad_edge:
                    # Broad "edge belt": faces whose normals are near-perpendicular to thickness axis.
                    absdot_max = float(getattr(self, "_migu_edge_absdot_max", 0.35) or 0.35)
                    absdot_max = float(np.clip(absdot_max, 0.0, 1.0))
                    mask = abs_dot <= absdot_max
                    mode_desc = f"경계(둘레) | absdot≤{absdot_max:.2f}"
                else:
                    # "미구" heuristic: dominant X/Y-facing faces that are not outer/inner.
                    major_thr = float(getattr(self, "_migu_major_axis_min", 0.55) or 0.55)
                    major_thr = float(np.clip(major_thr, 0.0, 1.0))
                    absdot_max = float(getattr(self, "_migu_absdot_max", 0.90) or 0.90)
                    absdot_max = float(np.clip(absdot_max, 0.0, 1.0))
                    ax_i = 0 if major_axis == "x" else 1
                    major = np.abs(fn_world[:, ax_i])
                    mask = (major >= major_thr) & (abs_dot <= absdot_max)
                    mode_desc = f"{major_axis.upper()}축 강조 | major≥{major_thr:.2f}, absdot≤{absdot_max:.2f}"

                idx = np.where(mask)[0].astype(np.int32, copy=False)
                n_sel = int(idx.size)
                if n_sel <= 0:
                    QMessageBox.information(
                        self,
                        "결과 없음",
                        "미구 자동 감지 결과가 없습니다.\n\n"
                        "팁:\n"
                        "- 기와를 정치 후(상면/하면이 위/아래) 다시 시도\n"
                        "- Ctrl을 누르고 다시 클릭(축 전환)\n"
                        "- Shift를 누르고 클릭(둘레 경계 전체 감지)",
                    )
                    return

                try:
                    obj.migu_face_indices.clear()
                    obj.migu_face_indices.update(int(x) for x in idx)
                except Exception:
                    obj.migu_face_indices = set(int(x) for x in idx)

                # Keep sets exclusive (migu wins).
                try:
                    obj.outer_face_indices.difference_update(obj.migu_face_indices)
                    obj.inner_face_indices.difference_update(obj.migu_face_indices)
                except Exception:
                    pass

                self.viewport.status_info = (
                    f"✅ 미구 자동 감지({mode_desc}): migu {len(obj.migu_face_indices):,} faces "
                    f"(Shift=경계, Ctrl=축전환)"
                )
                try:
                    self.viewport._emit_surface_assignment_changed(obj)
                except Exception:
                    pass
                QMessageBox.information(
                    self,
                    "완료",
                    "미구 자동 감지 결과를 현재 메쉬에 적용했습니다.\n\n"
                    f"- migu(미구): {len(obj.migu_face_indices):,} faces\n\n"
                    "표시: 미구=초록 오버레이\n"
                    "팁: 필요하면 '찍기/브러시/면적' 도구로 추가 보정하세요.\n"
                    "단축: Shift=둘레 경계, Ctrl=축 전환(X↔Y)",
                )
            except Exception as e:
                QMessageBox.critical(self, "오류", f"미구 자동 감지 실패:\n{e}")
                return

        else:
            self.status_info.setText(f"선택 작업: {action}")

        try:
            self.flatten_panel.update_surface_assignment_counts(
                len(obj.outer_face_indices),
                len(obj.inner_face_indices),
                len(obj.migu_face_indices),
            )
        except Exception:
            pass
        self.viewport.update()
        
    def _flatten_cache_key(self, obj, options: dict[str, Any]) -> tuple[object, ...]:
        method = str(options.get('method', 'ARAP')).strip()
        iterations = int(options.get('iterations', 30))
        boundary = str(options.get('boundary', 'free')).strip()
        initial = str(options.get('initial', 'lscm')).strip()
        distortion = float(options.get("distortion", 0.5))
        radius = float(options.get("radius", 0.0))
        direction = str(options.get("direction", "auto")).strip()
        auto_cut = bool(options.get("auto_cut", False))
        multiband = bool(options.get("multiband", False))

        t = tuple(np.round(np.asarray(obj.translation, dtype=np.float64), 6).tolist())
        r = tuple(np.round(np.asarray(obj.rotation, dtype=np.float64), 6).tolist())
        s = float(np.round(float(obj.scale), 6))

        return (
            id(obj),
            t,
            r,
            s,
            method,
            iterations,
            boundary,
            initial,
            float(np.round(distortion, 6)),
            float(np.round(radius, 6)),
            direction,
            auto_cut,
            multiband,
        )

    def _build_world_mesh(self, obj):
        """
        현재 화면에 보이는 변환값(T/R/S)을 적용한 MeshData 복사본을 생성합니다.
        (원본 obj.mesh는 변경하지 않습니다)
        """
        base = obj.mesh
        return MainWindow._build_world_mesh_from_transform(
            base,
            translation=getattr(obj, "translation", None),
            rotation=getattr(obj, "rotation", None),
            scale=float(getattr(obj, "scale", 1.0)),
        )

    @staticmethod
    def _build_world_mesh_from_transform(base, *, translation, rotation, scale: float):
        from src.core.mesh_loader import MeshData
        from scipy.spatial.transform import Rotation as R

        vertices = base.vertices.astype(np.float64) * float(scale)

        if rotation is not None and not np.allclose(rotation, [0, 0, 0]):
            rot = R.from_euler('xyz', rotation, degrees=True).as_matrix()
            vertices = (rot @ vertices.T).T

        if translation is not None and not np.allclose(translation, [0, 0, 0]):
            vertices = vertices + np.asarray(translation, dtype=np.float64)

        mesh = MeshData(
            vertices=vertices,
            faces=base.faces.copy(),
            normals=None,
            face_normals=None,
            uv_coords=base.uv_coords.copy() if base.uv_coords is not None else None,
            texture=base.texture,
            unit=base.unit,
            filepath=base.filepath
        )
        mesh.compute_normals(compute_vertex_normals=False)
        return mesh

    @staticmethod
    def _compute_flattened_mesh(mesh, options: dict[str, Any]):
        from src.core.flattener import flatten_with_method

        method = str(options.get('method', 'ARAP (형태 보존)'))
        iterations = int(options.get('iterations', 30))
        boundary_type = str(options.get('boundary', 'free'))
        initial = str(options.get('initial', 'lscm'))
        distortion = float(options.get("distortion", 0.5))
        radius_mm = float(options.get("radius", 0.0))
        direction = str(options.get("direction", "auto"))

        def normalize_method(text: str) -> str:
            t = str(text or "").strip().lower()
            if "arap" in t:
                return "arap"
            if "lscm" in t:
                return "lscm"
            if ("면적" in text) or ("area" in t):
                return "area"
            if ("원통" in text) or ("cyl" in t):
                return "cylinder"
            return "arap"

        # FlattenPanel의 radius는 mm 입력이므로, mesh.unit 기준으로 world 단위로 환산
        unit = str(getattr(mesh, "unit", "cm") or "cm").strip().lower()
        if unit == "mm":
            radius_world = radius_mm
        elif unit == "m":
            radius_world = radius_mm / 1000.0
        else:
            # default: cm
            radius_world = radius_mm / 10.0

        return flatten_with_method(
            mesh,
            method=normalize_method(method),
            iterations=iterations,
            distortion=distortion,
            boundary_type=boundary_type,
            initial_method=initial,
            cylinder_axis=direction,
            cylinder_radius=radius_world,
        )

    def _compute_flattened(self, obj, options: dict[str, Any]):
        mesh = self._build_world_mesh(obj)
        return self._compute_flattened_mesh(mesh, options)

    def _get_or_compute_flattened(self, obj, options: dict[str, Any]):
        key = self._flatten_cache_key(obj, options)
        cached = self._flattened_cache.get(key)
        if cached is not None:
            return cached

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            flattened = self._compute_flattened(obj, options)
        finally:
            QApplication.restoreOverrideCursor()

        # 캐시는 최근 결과만 유지 (객체/옵션이 바뀌면 새 키로 저장됨)
        self._flattened_cache[key] = flattened
        return flattened

    def on_flatten_requested(self, options: dict[str, Any]):
        obj = self.viewport.selected_obj
        if not obj or not obj.mesh:
            QMessageBox.warning(self, "경고", "먼저 메쉬를 선택하세요.")
            return

        key = self._flatten_cache_key(obj, options)
        cached = self._flattened_cache.get(key)
        if cached is not None:
            self._on_flatten_task_done({"key": key, "flattened": cached})
            return

        base = obj.mesh
        translation = (
            np.asarray(obj.translation, dtype=np.float64).copy()
            if getattr(obj, "translation", None) is not None
            else None
        )
        rotation = (
            np.asarray(obj.rotation, dtype=np.float64).copy()
            if getattr(obj, "rotation", None) is not None
            else None
        )
        scale = float(getattr(obj, "scale", 1.0))
        options_copy = dict(options)

        def task():
            mesh = MainWindow._build_world_mesh_from_transform(
                base, translation=translation, rotation=rotation, scale=scale
            )
            flattened = MainWindow._compute_flattened_mesh(mesh, options_copy)
            return {"key": key, "flattened": flattened}

        self.status_info.setText("🗺️ 펼침 처리 중...")
        self._start_task(
            title="펼침",
            label="펼침 처리 중...",
            thread=TaskThread("flatten", task),
            on_done=self._on_flatten_task_done,
            on_failed=self._on_flatten_task_failed,
        )

    def _on_flatten_task_done(self, result: Any):
        key = None
        flattened = None
        try:
            if isinstance(result, dict):
                key = result.get("key")
                flattened = result.get("flattened")
        except Exception:
            key = None
            flattened = None

        if flattened is None:
            self.status_info.setText("❌ 펼침 실패")
            QMessageBox.critical(self, "오류", self._format_error_message("펼침 처리 실패:", "Flatten result is empty."))
            return

        if key is not None:
            self._flattened_cache[key] = flattened

        self.status_info.setText(
            f"✅ 펼침 완료: {flattened.width:.2f} x {flattened.height:.2f} {flattened.original_mesh.unit} "
            f"(왜곡 평균 {flattened.mean_distortion:.1%})"
        )
        QMessageBox.information(
            self,
            "펼침 완료",
            f"펼침이 완료되었습니다.\n\n"
            f"- 크기: {flattened.width:.2f} x {flattened.height:.2f} {flattened.original_mesh.unit}\n"
            f"- 왜곡(평균/최대): {flattened.mean_distortion:.1%} / {flattened.max_distortion:.1%}\n\n"
            f"이제 '펼친 결과 SVG 저장' 또는 '탁본 이미지 내보내기'를 사용할 수 있습니다."
        )

    def _on_flatten_task_failed(self, message: str):
        self.status_info.setText("❌ 펼침 실패")
        QMessageBox.critical(self, "오류", self._format_error_message("펼침 처리 중 오류 발생:", message))

    def on_export_requested(self, data):
        """내보내기 요청 처리"""
        export_type = data.get('type')
        
        if export_type == 'profile_2d':
            self.export_2d_profile(data.get('view'))
            return
            
        if not self.viewport.selected_obj:
            QMessageBox.warning(self, "경고", "선택된 메쉬가 없습니다.")
            return

        obj = self.viewport.selected_obj
        if not obj.mesh:
            QMessageBox.warning(self, "경고", "선택된 객체에 메쉬 데이터가 없습니다.")
            return

        # 공통: 현재 펼침 옵션 (패널 값 기반)
        flatten_options = {
            'method': self.flatten_panel.combo_method.currentText(),
            'iterations': self.flatten_panel.spin_iterations.value(),
            'radius': self.flatten_panel.spin_radius.value(),
            'direction': self.flatten_panel.combo_direction.currentText(),
            'distortion': self.flatten_panel.slider_distortion.value() / 100.0,
            'auto_cut': self.flatten_panel.check_auto_cut.isChecked(),
            'multiband': self.flatten_panel.check_multiband.isChecked(),
            'boundary': 'free',
            'initial': 'lscm',
        }
        
        if export_type == 'rubbing':
            filepath, _ = QFileDialog.getSaveFileName(
                self, "탁본 이미지 저장", "", "PNG (*.png);;TIFF (*.tiff)"
            )
            if filepath:
                self.status_info.setText(f"내보내기: {filepath}")

                dpi = int(self.export_panel.spin_dpi.value())
                include_scale = bool(self.export_panel.check_scale_bar.isChecked())

                key = self._flatten_cache_key(obj, flatten_options)
                cached_flat = self._flattened_cache.get(key)
                base = obj.mesh
                translation = (
                    np.asarray(obj.translation, dtype=np.float64).copy()
                    if getattr(obj, "translation", None) is not None
                    else None
                )
                rotation = (
                    np.asarray(obj.rotation, dtype=np.float64).copy()
                    if getattr(obj, "rotation", None) is not None
                    else None
                )
                scale = float(getattr(obj, "scale", 1.0))
                opts = dict(flatten_options)

                def task_export_rubbing():
                    from src.core.surface_visualizer import SurfaceVisualizer

                    if cached_flat is not None:
                        flattened = cached_flat
                    else:
                        mesh = MainWindow._build_world_mesh_from_transform(
                            base, translation=translation, rotation=rotation, scale=scale
                        )
                        flattened = MainWindow._compute_flattened_mesh(mesh, opts)

                    # DPI 기준으로 출력 폭 계산 (실측 스케일 유지를 위해)
                    unit = (flattened.original_mesh.unit or "mm").lower()
                    width_real = float(flattened.width)
                    if unit == 'mm':
                        width_in = width_real / 25.4
                    elif unit == 'cm':
                        width_in = width_real / 2.54
                    elif unit == 'm':
                        width_in = (width_real * 100.0) / 2.54
                    else:
                        width_in = width_real / 25.4

                    width_pixels = max(800, int(width_in * dpi))
                    width_pixels = min(width_pixels, 12000)  # 메모리 보호용 상한

                    visualizer = SurfaceVisualizer(default_dpi=dpi)
                    rubbing = visualizer.generate_rubbing(flattened, width_pixels=width_pixels, style='traditional')
                    rubbing.save(filepath, include_scale_bar=include_scale)
                    return {"path": filepath, "key": key, "flattened": flattened if cached_flat is None else None}

                def on_done_export_rubbing(result: Any):
                    if isinstance(result, dict):
                        flat = result.get("flattened")
                        if flat is not None:
                            self._flattened_cache[key] = flat

                    QMessageBox.information(self, "완료", f"탁본 이미지가 저장되었습니다:\n{filepath}")
                    self.status_info.setText(f"✅ 저장 완료: {Path(filepath).name}")

                def on_failed(message: str):
                    self.status_info.setText("❌ 저장 실패")
                    QMessageBox.critical(self, "오류", self._format_error_message("탁본 저장 중 오류 발생:", message))

                self._start_task(
                    title="내보내기",
                    label="탁본 이미지 생성/저장 중...",
                    thread=TaskThread("export_rubbing", task_export_rubbing),
                    on_done=on_done_export_rubbing,
                    on_failed=on_failed,
                )

        elif export_type == 'ortho':
            filepath, _ = QFileDialog.getSaveFileName(
                self, "정사투영 이미지 저장", "", "PNG (*.png);;TIFF (*.tiff)"
            )
            if filepath:
                dpi = int(self.export_panel.spin_dpi.value())
                base = obj.mesh
                translation = (
                    np.asarray(obj.translation, dtype=np.float64).copy()
                    if getattr(obj, "translation", None) is not None
                    else None
                )
                rotation = (
                    np.asarray(obj.rotation, dtype=np.float64).copy()
                    if getattr(obj, "rotation", None) is not None
                    else None
                )
                scale = float(getattr(obj, "scale", 1.0))

                def task_export_ortho():
                    from src.core.orthographic_projector import OrthographicProjector

                    mesh = MainWindow._build_world_mesh_from_transform(
                        base, translation=translation, rotation=rotation, scale=scale
                    )
                    projector = OrthographicProjector(resolution=2048)
                    aligned = projector.align_mesh(mesh, method='pca')
                    result = projector.project(aligned, direction='top', render_mode='depth')
                    result.save(filepath, dpi=dpi)
                    return filepath

                def on_done_export_ortho(_result: Any):
                    QMessageBox.information(self, "완료", f"정사투영 이미지가 저장되었습니다:\n{filepath}")
                    self.status_info.setText(f"✅ 저장 완료: {Path(filepath).name}")

                def on_failed(message: str):
                    self.status_info.setText("❌ 저장 실패")
                    QMessageBox.critical(self, "오류", self._format_error_message("정사투영 저장 중 오류 발생:", message))

                self._start_task(
                    title="내보내기",
                    label="정사투영 이미지 생성/저장 중...",
                    thread=TaskThread("export_ortho", task_export_ortho),
                    on_done=on_done_export_ortho,
                    on_failed=on_failed,
                )

        elif export_type == 'flat_svg':
            filepath, _ = QFileDialog.getSaveFileName(
                self, "펼친 결과 SVG 저장", "flattened.svg", "Scalable Vector Graphics (*.svg)"
            )
            if filepath:
                key = self._flatten_cache_key(obj, flatten_options)
                cached_flat = self._flattened_cache.get(key)
                base = obj.mesh
                translation = (
                    np.asarray(obj.translation, dtype=np.float64).copy()
                    if getattr(obj, "translation", None) is not None
                    else None
                )
                rotation = (
                    np.asarray(obj.rotation, dtype=np.float64).copy()
                    if getattr(obj, "rotation", None) is not None
                    else None
                )
                scale = float(getattr(obj, "scale", 1.0))
                opts = dict(flatten_options)

                def task_export_flat_svg():
                    from src.core.flattened_svg_exporter import FlattenedSVGExporter, SVGExportOptions

                    if cached_flat is not None:
                        flattened = cached_flat
                    else:
                        mesh = MainWindow._build_world_mesh_from_transform(
                            base, translation=translation, rotation=rotation, scale=scale
                        )
                        flattened = MainWindow._compute_flattened_mesh(mesh, opts)
                    exporter = FlattenedSVGExporter()

                    # 1cm 격자를 기본 제공 (단위가 mm면 10mm)
                    unit = (flattened.original_mesh.unit or "cm").lower()
                    svg_unit = unit if unit in ('mm', 'cm') else 'cm'
                    grid = 10.0 if svg_unit == 'mm' else 1.0

                    exporter.export(
                        flattened,
                        filepath,
                        options=SVGExportOptions(
                            unit=svg_unit,
                            include_grid=True,
                            grid_spacing=grid,
                            include_outline=True,
                            include_wireframe=False,
                            stroke_width=0.05,
                        ),
                    )
                    return {"path": filepath, "key": key, "flattened": flattened if cached_flat is None else None}

                def on_done_export_flat_svg(result: Any):
                    if isinstance(result, dict):
                        flat = result.get("flattened")
                        if flat is not None:
                            self._flattened_cache[key] = flat
                    QMessageBox.information(self, "완료", f"펼친 결과 SVG가 저장되었습니다:\n{filepath}")
                    self.status_info.setText(f"✅ 저장 완료: {Path(filepath).name}")

                def on_failed(message: str):
                    self.status_info.setText("❌ 저장 실패")
                    QMessageBox.critical(self, "오류", self._format_error_message("SVG 저장 중 오류 발생:", message))

                self._start_task(
                    title="내보내기",
                    label="펼침 계산/ SVG 저장 중...",
                    thread=TaskThread("export_flat_svg", task_export_flat_svg),
                    on_done=on_done_export_flat_svg,
                    on_failed=on_failed,
                )

        elif export_type == 'sheet_svg':
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "통합 SVG 저장 (실측+단면+내/외면 탁본)",
                "rubbing_sheet.svg",
                "Scalable Vector Graphics (*.svg)",
            )
            if filepath:
                dpi = int(self.export_panel.spin_dpi.value())
                iterations = int(flatten_options.get("iterations", 30))

                base = obj.mesh
                translation = (
                    np.asarray(obj.translation, dtype=np.float64).copy()
                    if getattr(obj, "translation", None) is not None
                    else None
                )
                rotation = (
                    np.asarray(obj.rotation, dtype=np.float64).copy()
                    if getattr(obj, "rotation", None) is not None
                    else None
                )
                scale = float(getattr(obj, "scale", 1.0))
                cut_lines_world = self.viewport.get_cut_lines_world()
                cut_profiles_world = self.viewport.get_cut_sections_world()
                outer_idx = sorted(list(getattr(obj, "outer_face_indices", set()) or []))
                inner_idx = sorted(list(getattr(obj, "inner_face_indices", set()) or []))

                unit = str(getattr(base, "unit", "cm") or "cm").strip().lower()
                radius_mm = float(flatten_options.get("radius", 0.0))
                if unit == "mm":
                    cylinder_radius = radius_mm
                elif unit == "m":
                    cylinder_radius = radius_mm / 1000.0
                else:
                    cylinder_radius = radius_mm / 10.0

                def task_export_sheet_svg():
                    from src.core.rubbing_sheet_exporter import (
                        RubbingSheetExporter,
                        SheetExportOptions,
                    )

                    mesh = MainWindow._build_world_mesh_from_transform(
                        base, translation=translation, rotation=rotation, scale=scale
                    )
                    exporter = RubbingSheetExporter()
                    exporter.export(
                        mesh,
                        filepath,
                        cut_lines_world=cut_lines_world,
                        cut_profiles_world=cut_profiles_world,
                        outer_face_indices=outer_idx if outer_idx else None,
                        inner_face_indices=inner_idx if inner_idx else None,
                        options=SheetExportOptions(
                            dpi=dpi,
                            flatten_iterations=iterations,
                            flatten_method=str(flatten_options.get("method", "arap")),
                            flatten_distortion=float(flatten_options.get("distortion", 0.5)),
                            cylinder_axis=str(flatten_options.get("direction", "auto")),
                            cylinder_radius=cylinder_radius,
                        ),
                    )
                    return filepath

                def on_done_export_sheet_svg(_result: Any):
                    QMessageBox.information(self, "완료", f"통합 SVG가 저장되었습니다:\n{filepath}")
                    self.status_info.setText(f"✅ 저장 완료: {Path(filepath).name}")

                def on_failed(message: str):
                    self.status_info.setText("❌ 저장 실패")
                    QMessageBox.critical(self, "오류", self._format_error_message("통합 SVG 저장 중 오류 발생:", message))

                self._start_task(
                    title="내보내기",
                    label="통합 SVG 생성/저장 중...",
                    thread=TaskThread("export_sheet_svg", task_export_sheet_svg),
                    on_done=on_done_export_sheet_svg,
                    on_failed=on_failed,
                )

        elif export_type == 'mesh_outer':
            filepath, _ = QFileDialog.getSaveFileName(
                self, "외면 메쉬 저장", "", "OBJ (*.obj);;STL (*.stl);;PLY (*.ply)"
            )
            if filepath:
                base = obj.mesh
                translation = (
                    np.asarray(obj.translation, dtype=np.float64).copy()
                    if getattr(obj, "translation", None) is not None
                    else None
                )
                rotation = (
                    np.asarray(obj.rotation, dtype=np.float64).copy()
                    if getattr(obj, "rotation", None) is not None
                    else None
                )
                scale = float(getattr(obj, "scale", 1.0))

                def task_export_mesh_outer():
                    from src.core.surface_separator import SurfaceSeparator

                    mesh = MainWindow._build_world_mesh_from_transform(
                        base, translation=translation, rotation=rotation, scale=scale
                    )
                    separator = SurfaceSeparator()
                    result = separator.auto_detect_surfaces(mesh)
                    outer = getattr(result, "outer_surface", None)
                    if outer is None:
                        return {"status": "no_outer"}
                    MeshProcessor().save_mesh(outer, filepath)
                    return {"status": "ok"}

                def on_done_export_mesh_outer(result: Any):
                    if isinstance(result, dict) and result.get("status") == "no_outer":
                        QMessageBox.warning(self, "경고", "외면을 감지하지 못했습니다.")
                        return
                    QMessageBox.information(self, "완료", f"외면 메쉬가 저장되었습니다:\n{filepath}")

                def on_failed(message: str):
                    QMessageBox.critical(self, "오류", self._format_error_message("외면 저장 중 오류 발생:", message))

                self._start_task(
                    title="내보내기",
                    label="외면 메쉬 분리/저장 중...",
                    thread=TaskThread("export_mesh_outer", task_export_mesh_outer),
                    on_done=on_done_export_mesh_outer,
                    on_failed=on_failed,
                )
        elif export_type == 'mesh_inner':
            filepath, _ = QFileDialog.getSaveFileName(
                self, "내면 메쉬 저장", "", "OBJ (*.obj);;STL (*.stl);;PLY (*.ply)"
            )
            if filepath:
                base = obj.mesh
                translation = (
                    np.asarray(obj.translation, dtype=np.float64).copy()
                    if getattr(obj, "translation", None) is not None
                    else None
                )
                rotation = (
                    np.asarray(obj.rotation, dtype=np.float64).copy()
                    if getattr(obj, "rotation", None) is not None
                    else None
                )
                scale = float(getattr(obj, "scale", 1.0))

                def task_export_mesh_inner():
                    from src.core.surface_separator import SurfaceSeparator

                    mesh = MainWindow._build_world_mesh_from_transform(
                        base, translation=translation, rotation=rotation, scale=scale
                    )
                    separator = SurfaceSeparator()
                    result = separator.auto_detect_surfaces(mesh)
                    inner = getattr(result, "inner_surface", None)
                    if inner is None:
                        return {"status": "no_inner"}
                    MeshProcessor().save_mesh(inner, filepath)
                    return {"status": "ok"}

                def on_done_export_mesh_inner(result: Any):
                    if isinstance(result, dict) and result.get("status") == "no_inner":
                        QMessageBox.warning(self, "경고", "내면을 감지하지 못했습니다.")
                        return
                    QMessageBox.information(self, "완료", f"내면 메쉬가 저장되었습니다:\n{filepath}")

                def on_failed(message: str):
                    QMessageBox.critical(self, "오류", self._format_error_message("내면 저장 중 오류 발생:", message))

                self._start_task(
                    title="내보내기",
                    label="내면 메쉬 분리/저장 중...",
                    thread=TaskThread("export_mesh_inner", task_export_mesh_inner),
                    on_done=on_done_export_mesh_inner,
                    on_failed=on_failed,
                )
        elif export_type == 'mesh_flat':
            filepath, _ = QFileDialog.getSaveFileName(
                self, "펼쳐진 메쉬 저장", "", "OBJ (*.obj);;STL (*.stl);;PLY (*.ply)"
            )
            if filepath:
                key = self._flatten_cache_key(obj, flatten_options)
                cached_flat = self._flattened_cache.get(key)
                base = obj.mesh
                translation = (
                    np.asarray(obj.translation, dtype=np.float64).copy()
                    if getattr(obj, "translation", None) is not None
                    else None
                )
                rotation = (
                    np.asarray(obj.rotation, dtype=np.float64).copy()
                    if getattr(obj, "rotation", None) is not None
                    else None
                )
                scale = float(getattr(obj, "scale", 1.0))
                opts = dict(flatten_options)

                def task_export_mesh_flat():
                    from src.core.mesh_loader import MeshData

                    if cached_flat is not None:
                        flattened = cached_flat
                    else:
                        mesh = MainWindow._build_world_mesh_from_transform(
                            base, translation=translation, rotation=rotation, scale=scale
                        )
                        flattened = MainWindow._compute_flattened_mesh(mesh, opts)

                    uv_real = flattened.uv.astype(np.float64) * float(flattened.scale)
                    uv_real -= uv_real.min(axis=0)
                    vertices_3d = np.column_stack([uv_real[:, 0], uv_real[:, 1], np.zeros(len(uv_real))])

                    flat_mesh = MeshData(
                        vertices=vertices_3d,
                        faces=flattened.faces.copy(),
                        normals=None,
                        face_normals=None,
                        uv_coords=None,
                        texture=None,
                        unit=flattened.original_mesh.unit,
                        filepath=None
                    )
                    flat_mesh.compute_normals(compute_vertex_normals=False)

                    MeshProcessor().save_mesh(flat_mesh, filepath)
                    return {"status": "ok", "flattened": flattened if cached_flat is None else None}

                def on_done_export_mesh_flat(result: Any):
                    if isinstance(result, dict):
                        flat = result.get("flattened")
                        if flat is not None:
                            self._flattened_cache[key] = flat
                    QMessageBox.information(self, "완료", f"펼쳐진 메쉬가 저장되었습니다:\n{filepath}")

                def on_failed(message: str):
                    QMessageBox.critical(self, "오류", self._format_error_message("펼친 메쉬 저장 중 오류 발생:", message))

                self._start_task(
                    title="내보내기",
                    label="펼침/메쉬 생성/저장 중...",
                    thread=TaskThread("export_mesh_flat", task_export_mesh_flat),
                    on_done=on_done_export_mesh_flat,
                    on_failed=on_failed,
                )
    
    def export_2d_profile(self, view):
        """2D 실측 도면(SVG) 내보내기"""
        obj = self.viewport.selected_obj
        if not obj:
            QMessageBox.warning(self, "경고", "선택된 메쉬가 없습니다.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            f"2D 도면 저장 ({view})",
            f"{view}_profile.svg",
            "Scalable Vector Graphics (*.svg)"
        )
        
        if not filepath:
            return

        cam_state = None
        try:
            # 지정된 뷰로 자동 정렬 후 캡처 (내보내기 완료 후 원래 카메라 상태 복원)
            cam = self.viewport.camera
            cam_state = (
                float(cam.distance),
                float(cam.azimuth),
                float(cam.elevation),
                cam.center.copy(),
                cam.pan_offset.copy(),
            )
            view_map = {
                'top': (0.0, 89.0),
                'bottom': (0.0, -89.0),
                'front': (-90.0, 0.0),
                'back': (90.0, 0.0),
                'left': (180.0, 0.0),
                'right': (0.0, 0.0),
            }
            if view in view_map:
                # 메쉬 + 단면(바닥 배치)까지 화면에 들어오도록 bounds 확장
                bounds = np.asarray(obj.get_world_bounds(), dtype=np.float64)
                try:
                    extra_pts = []
                    for ln in self.viewport.get_cut_sections_world() or []:
                        for p in ln or []:
                            extra_pts.append(np.asarray(p, dtype=np.float64))
                    if extra_pts:
                        ep = np.vstack(extra_pts)
                        bounds[0] = np.minimum(bounds[0], ep.min(axis=0))
                        bounds[1] = np.maximum(bounds[1], ep.max(axis=0))
                except Exception:
                    pass

                cam.fit_to_bounds(bounds)
                cam.azimuth, cam.elevation = view_map[view]

            # 1. 고해상도 이미지 캡처 및 정렬용 행렬 획득
            qimage, mv, proj, vp = self.viewport.capture_high_res_image(
                width=2048,
                height=2048,
                only_selected=True,
                orthographic=True,
            )

            # QImage -> PIL Image 변환 (Qt QBuffer 사용)
            ba = QByteArray()
            qbuf = QBuffer(ba)
            qbuf.open(QIODevice.OpenModeFlag.WriteOnly)
            qimage.save(qbuf, "PNG")
            qbuf.close()
            pil_img = Image.open(io.BytesIO(ba.data()))

            # 2. 프로파일 추출 및 SVG 내보내기
            exporter = ProfileExporter(resolution=2048) # 추출 해상도

            running = getattr(self, "_profile_export_thread", None)
            if running is not None and running.isRunning():
                QMessageBox.information(self, "내보내기", "이미 내보내기 작업이 진행 중입니다.")
                return

            dlg = QProgressDialog("2D 도면(SVG) 내보내는 중...", None, 0, 0, self)
            dlg.setWindowTitle("내보내기")
            dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
            dlg.setCancelButton(None)
            dlg.setMinimumDuration(0)
            dlg.show()
            self._profile_export_dialog = dlg

            self._profile_export_thread = ProfileExportThread(
                mesh_data=obj.mesh,
                view=view,
                output_path=filepath,
                translation=obj.translation.copy(),
                rotation=obj.rotation.copy(),
                scale=float(obj.scale),
                viewport_image=pil_img,
                opengl_matrices=(mv, proj, vp),
                cut_lines_world=self.viewport.get_cut_lines_world(),
                cut_profiles_world=self.viewport.get_cut_sections_world(),
                resolution=2048,
                grid_spacing=1.0,
                include_grid=True,
            )
            self._profile_export_thread.done.connect(self._on_profile_export_done)
            self._profile_export_thread.failed.connect(self._on_profile_export_failed)
            self._profile_export_thread.finished.connect(self._on_profile_export_finished)
            self._profile_export_thread.start()
            self.status_info.setText(f"내보내기 시작: {Path(filepath).name}")
            return

            result_path = exporter.export_profile(
                obj.mesh,
                view=view,
                output_path=filepath,
                translation=obj.translation,
                rotation=obj.rotation,
                scale=obj.scale,
                grid_spacing=1.0, # 1cm 격자
                include_grid=True,
                viewport_image=pil_img,
                opengl_matrices=(mv, proj, vp) # 정밀 정렬을 위한 행렬 전달
            )

            QMessageBox.information(self, "완료", f"2D 도면이 저장되었습니다:\n{result_path}")
            self.status_info.setText(f"✅ 저장 완료: {Path(result_path).name}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_info.setText("❌ 저장 실패")
            QMessageBox.critical(self, "오류", f"도면 저장 중 오류 발생:\n{str(e)}")
        finally:
            # 카메라 복원
            if cam_state is not None:
                try:
                    cam = self.viewport.camera
                    cam.distance, cam.azimuth, cam.elevation = cam_state[0], cam_state[1], cam_state[2]
                    cam.center = cam_state[3]
                    cam.pan_offset = cam_state[4]
                    self.viewport.update()
                except Exception:
                    pass
    
    def reset_transform_and_center(self):
        """변환 리셋 + 뷰 맞춤"""
        obj = self.viewport.selected_obj
        if obj is None:
            return

        self.reset_transform()
        self.fit_view()
        self.status_info.setText("🔄 변환 초기화 + 뷰 맞춤 완료")
    
    def bake_and_center(self):
        """정치: 현재 회전을 메쉬 버텍스에 영구 적용하고 변환 리셋"""
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
        
        # OpenGL 렌더링(glRotate X->Y->Z)과 동일한 합성 회전
        rotation_matrix = rot_x @ rot_y @ rot_z
        
        # 메쉬 버텍스에 회전과 스케일 적용
        obj.mesh.vertices = (rotation_matrix @ obj.mesh.vertices.T).T * obj.scale
        try:
            obj.mesh._bounds = None
            obj.mesh._centroid = None
            obj.mesh._surface_area = None
        except Exception:
            pass
        
        # 법선 다시 계산
        obj.mesh.compute_normals(compute_vertex_normals=False, force=True)
        obj._trimesh = None
        
        # 중심을 원점으로 이동
        centroid = obj.mesh.vertices.mean(axis=0)
        obj.mesh.vertices -= centroid
        try:
            obj.mesh._bounds = None
            obj.mesh._centroid = None
        except Exception:
            pass
        
        # VBO 업데이트
        self.viewport.update_vbo(obj)
        
        # 변환 리셋
        obj.translation = np.array([0.0, 0.0, 0.0])
        obj.rotation = np.array([0.0, 0.0, 0.0])
        obj.scale = 1.0
        
        self.sync_transform_panel()
        self.viewport.update()
        self.status_info.setText("✅ 정치 완료 - 회전이 메쉬에 적용됨")
    
    def return_to_origin(self):
        """카메라를 원점으로 이동"""
        self.viewport.camera.center = np.array([0.0, 0.0, 0.0])
        self.viewport.camera.pan_offset = np.array([0.0, 0.0, 0.0])
        self.viewport.update()
        self.status_info.setText("🏠 카메라 원점 복귀")
            
    def reset_view(self):
        self.viewport.camera.reset()
        self.viewport.update()
    
    def fit_view(self):
        obj = self.viewport.selected_obj
        if obj:
            # 월드 좌표계 바운드로 획득
            self.viewport.camera.fit_to_bounds(obj.get_world_bounds())
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
        self.viewport.picking_mode = 'curvature' if enabled else 'none'
        if enabled:
            self.status_info.setText("📏 곡률 측정 모드: 메쉬 위를 클릭하여 점을 찍으세요")
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
        world_points = np.asarray(self.viewport.picked_points, dtype=np.float64)
        
        fitter = CurvatureFitter()
        arc = fitter.fit_arc(world_points)
        
        if arc is None:
            QMessageBox.warning(
                self,
                "경고",
                "원호 피팅에 실패했습니다.\n"
                "점들이 일직선 위에 있거나 너무 가까울 수 있습니다.",
            )
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
        self.status_info.setText(
            f"✅ 원호 #{arc_count} 생성됨 (월드 고정): 반지름 = {arc.radius:.2f} cm "
            f"({radius_mm:.1f} mm)"
        )
    
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
    
    def on_roi_toggled(self, enabled):
        """2D ROI 모드 토글 핸들러"""
        self.viewport.roi_enabled = enabled
        if enabled:
            # ROI는 바닥 평면 드래그를 사용 -> 다른 입력 모드 비활성화
            if self.viewport.crosshair_enabled:
                self.viewport.crosshair_enabled = False
                try:
                    self.section_panel.btn_toggle.blockSignals(True)
                    self.section_panel.btn_toggle.setChecked(False)
                    self.section_panel.btn_toggle.blockSignals(False)
                except Exception:
                    pass

            if getattr(self.viewport, "cut_lines_enabled", False):
                self.viewport.set_cut_lines_enabled(False)
                try:
                    self.section_panel.btn_line.blockSignals(True)
                    self.section_panel.btn_line.setChecked(False)
                    self.section_panel.btn_line.blockSignals(False)
                except Exception:
                    pass

            # ROI가 활성화되면 초기 범위를 메쉬 크기에 맞춤
            if self.viewport.selected_obj and self.viewport.selected_obj.mesh:
                b = self.viewport.selected_obj.get_world_bounds()
                # [min_x, max_x, min_y, max_y]
                self.viewport.roi_bounds = [float(b[0][0]), float(b[1][0]), float(b[0][1]), float(b[1][1])]
            try:
                self.viewport.schedule_roi_edges_update(0)
            except Exception:
                pass
        else:
            try:
                self.viewport.roi_cut_edges = {"x1": [], "x2": [], "y1": [], "y2": []}
            except Exception:
                pass
        self.viewport.picking_mode = 'none' 
        self.viewport.update()

    def on_silhouette_extracted(self, points):
        """추출된 외곽선 처리 핸들러"""
        if not points:
            return
        self.status_info.setText(f"✅ {len(points)}개의 점으로 외곽선 추출 완료")
        print(f"Extracted Silhouette: {len(points)} points")

    def on_crosshair_toggled(self, enabled):
        """십자선 모드 토글 핸들러 (Viewport3D와 연동)"""
        # 십자선/선형 단면은 입력(드래그) 충돌 -> 상호 배타로 처리
        if enabled and getattr(self.viewport, "cut_lines_enabled", False):
            self.viewport.set_cut_lines_enabled(False)
            try:
                self.section_panel.btn_line.blockSignals(True)
                self.section_panel.btn_line.setChecked(False)
                self.section_panel.btn_line.blockSignals(False)
            except Exception:
                pass

        # ROI와도 입력이 충돌하므로 상호 배타로 처리
        if enabled and getattr(self.viewport, "roi_enabled", False):
            self.viewport.roi_enabled = False
            self.viewport.active_roi_edge = None
            try:
                self.section_panel.btn_roi.blockSignals(True)
                self.section_panel.btn_roi.setChecked(False)
                self.section_panel.btn_roi.blockSignals(False)
                self.section_panel.btn_silhouette.setEnabled(False)
            except Exception:
                pass

        self.viewport.crosshair_enabled = enabled
        if enabled:
            self.viewport.picking_mode = 'crosshair'
            self.viewport.schedule_crosshair_profile_update(0)
        else:
            if self.viewport.picking_mode == 'crosshair':
                self.viewport.picking_mode = 'none'
        self.viewport.update()

    def on_line_section_toggled(self, enabled):
        """단면선(2개) 모드 토글 핸들러"""
        # 십자선/단면선/ROI는 입력 충돌 -> 상호 배타로 처리
        if enabled and self.viewport.crosshair_enabled:
            self.viewport.crosshair_enabled = False
            try:
                self.section_panel.btn_toggle.blockSignals(True)
                self.section_panel.btn_toggle.setChecked(False)
                self.section_panel.btn_toggle.blockSignals(False)
            except Exception:
                pass

        # ROI와도 입력이 충돌하므로 상호 배타로 처리
        if enabled and getattr(self.viewport, "roi_enabled", False):
            self.viewport.roi_enabled = False
            self.viewport.active_roi_edge = None
            try:
                self.section_panel.btn_roi.blockSignals(True)
                self.section_panel.btn_roi.setChecked(False)
                self.section_panel.btn_roi.blockSignals(False)
                self.section_panel.btn_silhouette.setEnabled(False)
            except Exception:
                pass

        self.viewport.set_cut_lines_enabled(enabled)

    def on_cut_line_active_changed(self, index: int):
        """단면선(2개) 중 활성 선 변경"""
        try:
            self.viewport.cut_line_active = int(index)
            self.viewport.cut_line_preview = None
            idx = int(index)
            idx = idx if idx in (0, 1) else 0
            line = self.viewport.cut_lines[idx]
            final = getattr(self.viewport, "_cut_line_final", [False, False])
            self.viewport.cut_line_drawing = bool(line) and not bool(final[idx])
            self.viewport.update()
        except Exception:
            pass

    def on_cut_line_clear_requested(self, index: int):
        """현재 활성 단면선 지우기"""
        try:
            self.viewport.clear_cut_line(int(index))
            self.viewport.update()
        except Exception:
            pass

    def on_cut_lines_clear_all_requested(self):
        """단면선 전체 지우기"""
        try:
            self.viewport.clear_cut_lines()
            self.viewport.update()
        except Exception:
            pass

    def on_save_section_layers_requested(self):
        """현재 단면/가이드 결과를 레이어로 저장(스냅샷)."""
        try:
            added = int(self.viewport.save_current_sections_to_layers())
        except Exception:
            added = 0

        if added <= 0:
            self.status_info.setText("저장할 단면 레이어가 없습니다.")
            return

        self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
        self.status_info.setText(f"단면 레이어 {added}개 저장됨")

    def _on_cut_lines_auto_ended(self):
        """Viewport에서 단면선(2개) 입력이 자동 종료되면 버튼 상태도 맞춰줌"""
        try:
            if self.section_panel.btn_line.isChecked():
                self.section_panel.btn_line.setChecked(False)
        except Exception:
            pass

    def _request_slice_compute(self):
        if not getattr(self.viewport, "slice_enabled", False):
            return

        obj = self.viewport.selected_obj
        if obj is None or obj.mesh is None:
            self.viewport.slice_contours = []
            self.viewport.update()
            return

        height = (
            float(self._slice_pending_height)
            if self._slice_pending_height is not None
            else float(self.viewport.slice_z)
        )

        thread = getattr(self, "_slice_compute_thread", None)
        if thread is not None and thread.isRunning():
            # 이미 계산 중이면 최신 요청만 기억해두고 종료 후 재요청
            self._slice_pending_height = height
            return

        # 지금 값으로 계산 시작
        self._slice_pending_height = None
        self._slice_compute_thread = SliceComputeThread(
            mesh_data=obj.mesh,
            translation=obj.translation.copy(),
            rotation=obj.rotation.copy(),
            scale=float(obj.scale),
            z_height=height,
        )
        self._slice_compute_thread.computed.connect(self._on_slice_computed)
        self._slice_compute_thread.failed.connect(self._on_slice_compute_failed)
        self._slice_compute_thread.finished.connect(self._on_slice_compute_finished)
        self._slice_compute_thread.start()

    def _on_slice_computed(self, z_height: float, contours):
        if not getattr(self.viewport, "slice_enabled", False):
            return

        # 사용자가 높이를 바꿨으면(또는 pending이 있으면) 오래된 결과는 버림
        if self._slice_pending_height is not None:
            return
        if not np.isclose(float(self.viewport.slice_z), float(z_height), atol=1e-6):
            return

        self.viewport.slice_contours = contours or []
        self.viewport.update()

    def _on_slice_compute_failed(self, z_height: float, message: str):
        if not getattr(self.viewport, "slice_enabled", False):
            return
        self.viewport.slice_contours = []
        self.viewport.update()
        # 너무 잦은 팝업 방지: 상태바에만 표시
        try:
            self.status_info.setText(f"단면 계산 실패 (Z={float(z_height):.2f}cm): {message}")
        except Exception:
            pass

    def _on_slice_compute_finished(self):
        thread = getattr(self, "_slice_compute_thread", None)
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                pass
        self._slice_compute_thread = None

        if getattr(self.viewport, "slice_enabled", False) and self._slice_pending_height is not None:
            # 다음 요청이 대기 중이면 바로 처리
            self._slice_debounce_timer.start(1)

    def on_slice_changed(self, enabled, height):
        """단면 슬라이싱 상태/높이 변경 핸들러"""
        self.viewport.slice_enabled = enabled
        self.viewport.slice_z = float(height)

        if enabled:
            # plane은 즉시 갱신, 실제 단면 계산은 디바운스 + 스레드
            self.viewport.slice_contours = []
            self.viewport.update()

            self._slice_pending_height = float(height)
            self._slice_debounce_timer.start(150)
            return

        self._slice_pending_height = None
        try:
            self._slice_debounce_timer.stop()
        except Exception:
            pass
        self.viewport.slice_contours = []
        self.viewport.update()

    def on_slice_export_requested(self, height):
        """단면 SVG 내보내기 핸들러"""
        obj = self.viewport.selected_obj
        if not obj or not obj.mesh:
            QMessageBox.warning(self, "경고", "내보낼 대상 메쉬가 없습니다.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "단면 SVG 내보내기", f"section_z_{height:.2f}.svg", "SVG Files (*.svg)"
        )
        
        if file_path:
            try:
                from src.core.mesh_slicer import MeshSlicer
                slicer = MeshSlicer(obj.mesh)
                
                # 로컬 좌표계로 평면 변환
                from scipy.spatial.transform import Rotation as R
                inv_rot = R.from_euler('xyz', obj.rotation, degrees=True).inv().as_matrix()
                inv_scale = 1.0 / obj.scale if obj.scale != 0 else 1.0
                
                world_origin = np.array([0, 0, height])
                local_origin = inv_scale * inv_rot @ (world_origin - obj.translation)
                
                world_normal = np.array([0, 0, 1])
                local_normal = inv_rot @ world_normal
                
                # Slicer를 통해 SVG 직접 내보내기는 slice_at_z 대신 slice_with_plane 기반 SVG 구현 필요
                # 일단 slice_multiple_z 형태를 응용하거나 수동 SVG 생성
                
                # MeshSlicer 클래스에 slice_with_plane_svg 추가하거나, 
                # 여기서 contours 추출 후 slicer.export_slice_svg_from_contours(file_path, contours) 같은 식
                
                # 우선 slicer.py를 수정하여 slice_with_plane_svg를 추가하는 것이 깔끔함.
                # 임시로 contours 추출 후 slicer의 일반 SVG 메서드 활용 시뮬레이션
                
                contours = slicer.slice_with_plane(local_origin, local_normal)
                if not contours:
                    QMessageBox.warning(self, "경고", f"Z={height:.2f} 높이에서 단면을 찾을 수 없습니다.")
                    return
                
                # slicer.export_slice_svg는 slice_at_z(수평)만 지원하므로,
                # contours를 직접 전달하는 방식이 필요함. 
                # (slicer.py 수정을 예약하고 일단 구현 유보 혹은 slicer.py 즉시 수정)
                
                # TODO: slicer.py에 export_contours_svg 추가
                # 일단 slicer.export_slice_svg(height, file_path) 호출 (단, local transform 고려 안됨)
                # 정답: slicer.py에 contours를 인자로 받는 메서드 추가 필요
                
                self._save_contours_as_svg(file_path, contours, height)
                
                QMessageBox.information(self, "성공", f"단면 SVG가 저장되었습니다:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "오류", f"SVG 저장 중 오류 발생: {e}")

    def _save_contours_as_svg(self, path, contours, z_val):
        """임시 SVG 저장 (로컬 contours를 월드 비율로)"""
        # 바운딩 박스 (로컬 XY)
        # 하지만 스케일이 곱해져야 하므로...
        obj = self.viewport.selected_obj
        if obj is None:
            return
        scale = float(obj.scale)
        all_pts = np.vstack(contours) * scale
        
        min_x, min_y = all_pts[:, 0].min(), all_pts[:, 1].min()
        max_x, max_y = all_pts[:, 0].max(), all_pts[:, 1].max()
        
        width = (max_x - min_x) * 1.1
        height = (max_y - min_y) * 1.1
        
        svg = [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.2f}cm" '
                f'height="{height:.2f}cm" viewBox="0 0 {width:.4f} {height:.4f}">'
            ),
            '<g stroke="red" fill="none" stroke-width="0.1">',
        ]
        
        for cnt in contours:
            pts = cnt[:, :2] * scale
            pts[:, 0] -= min_x
            pts[:, 1] = height - (pts[:, 1] - min_y)
            pts_str = " ".join([f"{p[0]:.3f},{p[1]:.3f}" for p in pts])
            svg.append(f'<polyline points="{pts_str}" fill="none" />')
             
        svg.append('</g></svg>')
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(svg))

    def show_about(self):
        icon_path = get_icon_path()
        msg = QMessageBox(self)
        sha, dirty = _safe_git_info(str(Path(basedir)))
        sha_s = f"{sha}{'*' if dirty else ''}" if sha else "unknown"
        msg.setWindowTitle(f"{APP_NAME} v{APP_VERSION} ({sha_s})")
        
        if icon_path:
            msg.setIconPixmap(QPixmap(icon_path).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio))
        
        debug_info = _collect_debug_info(basedir=str(Path(basedir)))
        msg.setText(f"""
            <h2>{APP_NAME} v{APP_VERSION}</h2>
            <p>고고학 메쉬 탁본 도구</p>
            <p style="font-size: 11px; color: #718096;">© 2026 balguljang2 (lzpxilfe) / Licensed under GPLv2</p>
            <hr>
            <p style="font-size: 11px; color: #718096; white-space: pre-wrap;">{debug_info}</p>
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
        global _log_path
        try:
            from src.core.logging_utils import setup_logging

            _log_path = setup_logging()
        except Exception:
            _log_path = None

        def _excepthook(exc_type, exc, tb):
            _LOGGER.critical("Unhandled exception", exc_info=(exc_type, exc, tb))

        sys.excepthook = _excepthook

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
        
        splash.showMessage("Loading engine...")
        
        # 2. 메인 윈도우 생성
        splash.showMessage("Initializing Main Window...")
        window = MainWindow()
        
        # 3. 마무리 및 스플래시 닫기
        splash.showMessage("Ready!")
        QTimer.singleShot(1000, lambda: (splash.close(), window.show()))
        
        sys.exit(app.exec())
    except Exception as e:
        import traceback
        _LOGGER.exception("Application crashed on startup")
        err_msg = f"Application crashed on startup:\n\n{e}\n\n{traceback.format_exc()}"
        try:
            try:
                from src.core.logging_utils import format_exception_message

                err_msg = format_exception_message(
                    "Application crashed on startup:",
                    f"{e}\n\n{traceback.format_exc()}",
                    log_path=_log_path,
                )
            except Exception:
                pass
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            QMessageBox.critical(None, "Fatal Startup Error", err_msg)
        except Exception:
            pass
        sys.exit(1)


if __name__ == '__main__':
    main()
