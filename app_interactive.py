"""
ArchMeshRubbing v0.1.0 - Complete Interactive Application
Copyright (C) 2026 balguljang2 (lzpxilfe)
Licensed under the GNU General Public License v2.0 (GPL2)
"""

import sys
import logging
import subprocess
import json
import time
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
APP_VERSION = "0.1.0"
ORTHO_VIEW_SCALE_DEFAULT = 1.15
DEFAULT_MESH_UNIT = "cm"
DEFAULT_PROJECT_FILENAME = "project.amr"
MIN_EXPORT_WIDTH_PX = 800
MAX_EXPORT_WIDTH_PX = 12000
_UNIT_TO_INCHES: dict[str, float] = {
    "mm": 1.0 / 25.4,
    "cm": 1.0 / 2.54,
    "m": 100.0 / 2.54,
}


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


def _safe_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _width_in_inches(width_real: float, unit: str) -> float:
    factor = _UNIT_TO_INCHES.get(str(unit).strip().lower(), _UNIT_TO_INCHES["mm"])
    return float(width_real) * float(factor)

# Add src to path
# Add basedir to path so 'src' package can be found
if getattr(sys, 'frozen', False):
    basedir = getattr(sys, "_MEIPASS", str(Path(__file__).parent))
else:
    basedir = str(Path(__file__).parent)
sys.path.insert(0, basedir)

try:
    import src as _amr_src  # noqa: E402

    APP_VERSION = str(getattr(_amr_src, "__version__", APP_VERSION))
except Exception:
    pass

from src.gui.viewport_3d import Viewport3D  # noqa: E402
from src.core.mesh_loader import MeshLoader, MeshProcessor  # noqa: E402
from src.core.profile_exporter import ProfileExporter  # noqa: E402
from src.core.project_file import (  # noqa: E402
    ProjectFormatError,
    load_project as load_amr_project,
    save_project as save_amr_project,
)
from src.gui.profile_graph_widget import ProfileGraphWidget  # noqa: E402
from src.core.alignment_utils import (  # noqa: E402
    compute_floor_contact_shift,
    fit_plane_normal,
    orient_plane_normal_toward,
    rotation_matrix_align_vectors,
)


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

            try:
                setattr(mesh_data, "_amr_source_scale_factor", float(self._scale_factor))
            except Exception:
                pass

            # Precompute heavy caches in the loader thread so the UI stays responsive.
            # - face_normals: required for display and many tools (compute once, in background)
            # - face_centroids: speeds up surface tools on huge meshes (lasso/brush)
            try:
                if getattr(mesh_data, "face_normals", None) is None:
                    mesh_data.compute_normals(compute_vertex_normals=False)
            except Exception:
                _LOGGER.debug("Mesh normals precompute failed (continuing)", exc_info=True)

            try:
                n_faces = int(getattr(mesh_data, "n_faces", 0) or 0)
            except Exception:
                n_faces = 0

            try:
                threshold = int(getattr(mesh_data, "_amr_precompute_face_centroids_threshold", 300000) or 300000)
            except Exception:
                threshold = 300000

            if n_faces >= threshold:
                try:
                    faces = np.asarray(getattr(mesh_data, "faces", None), dtype=np.int32)
                    verts = np.asarray(getattr(mesh_data, "vertices", None), dtype=np.float64)
                    if faces.ndim == 2 and faces.shape[1] >= 3 and verts.ndim == 2 and verts.shape[1] >= 3:
                        centroids = np.empty((int(faces.shape[0]), 3), dtype=np.float32)
                        try:
                            chunk = int(getattr(mesh_data, "_amr_precompute_face_centroids_chunk", 250000) or 250000)
                        except Exception:
                            chunk = 250000
                        chunk = max(50000, min(chunk, 500000))

                        for start in range(0, int(faces.shape[0]), int(chunk)):
                            if self.isInterruptionRequested():
                                break
                            end = min(int(faces.shape[0]), start + int(chunk))
                            f = faces[start:end, :3]
                            v0 = verts[f[:, 0], :3]
                            v1 = verts[f[:, 1], :3]
                            v2 = verts[f[:, 2], :3]
                            centroids[start:end, :] = ((v0 + v1 + v2) / 3.0).astype(np.float32, copy=False)

                        if not self.isInterruptionRequested():
                            setattr(mesh_data, "_amr_face_centroids", centroids)
                            setattr(mesh_data, "_amr_face_centroids_faces_count", int(faces.shape[0]))
                except Exception:
                    _LOGGER.debug("Mesh face-centroids precompute failed (continuing)", exc_info=True)

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
        include_feature_lines: bool = False,
        feature_angle_deg: float = 60.0,
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
        self._include_feature_lines = bool(include_feature_lines)
        self._feature_angle_deg = float(feature_angle_deg)

    def run(self):
        try:
            exporter = ProfileExporter(resolution=self._resolution)
            feature_edges = None
            feature_style = None
            if self._include_feature_lines:
                try:
                    from src.core.feature_line_extractor import extract_sharp_edges

                    feature_edges = extract_sharp_edges(
                        self._mesh_data,
                        angle_deg=float(self._feature_angle_deg),
                        include_boundary=False,
                        min_edge_length=0.0,
                    )
                    feature_style = {"stroke": "#4a5568", "stroke_width": 0.01, "max_segments": 20000}
                except Exception:
                    feature_edges = None
                    feature_style = None

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
                feature_edges=feature_edges,
                feature_style=feature_style,
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
            <h3 style="margin:0; color:#2c5282;">🌲 레이어 트리 (Layer)</h3>
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

            <b>🤖 자동 분리(실험)</b><br>
            - 클릭: 자동 분리(auto)<br>
            - <b>Shift + 클릭:</b> 가시성(보이는 면) 기반 강제<br>
            - <b>Ctrl + 클릭:</b> 원통 기반 강제<br><br>

            <b>🧲 경계(면적+자석)</b><br>
            - <b>좌클릭:</b> 점 추가(자석 스냅) / <b>드래그:</b> 카메라 회전<br>
            - <b>첫 점 근처 클릭</b> 또는 <b>우클릭/Enter</b>: 확정<br>
            - <b>Backspace</b>: 되돌리기 / <b>Alt</b>: 제거 모드<br>
            - <b>Shift/Ctrl</b>: 완드 정제 / <b>[ / ]</b>: 자석 반경 / <b>ESC</b>: 종료<br>
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
        title = QLabel(f"{APP_NAME} v{APP_VERSION}")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #2c5282;
            margin-top: 10px;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title)
        
        # 버전 정보 추가 (사용자 확인용)
        version = QLabel(f"Version: {APP_VERSION}")
        version.setStyleSheet("color: #a0aec0; font-size: 10px; margin-bottom: 5px;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(version)
        
        # 서브타이틀
        subtitle = QLabel("고고학용 3d 메쉬 도구")
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
    """레이어 기준으로 객체 목록과 부착된 요소를 보여주는 트리 패널"""
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
        from src.core.unit_utils import mesh_units_to_mm

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
                r_mm = mesh_units_to_mm(float(getattr(arc, "radius", 0.0)), getattr(obj.mesh, "unit", None))
                arc_item.setText(2, f"R={r_mm:.1f}mm")
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
        self.trans_x = self._create_spin(-10000, 10000, "X", step=0.1)
        self.trans_y = self._create_spin(-10000, 10000, "Y", step=0.1)
        self.trans_z = self._create_spin(-10000, 10000, "Z", step=0.1)
        self.addWidget(self.trans_x)
        self.addWidget(self.trans_y)
        self.addWidget(self.trans_z)
        
        self.addSeparator()
        
        # 회전 (deg)
        self.addWidget(QLabel(" 🔄 회전: "))
        self.rot_x = self._create_spin(-360, 360, "Rx", step=1.0)
        self.rot_y = self._create_spin(-360, 360, "Ry", step=1.0)
        self.rot_z = self._create_spin(-360, 360, "Rz", step=1.0)
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

        self.btn_fit_ground = QPushButton("⬆ 바닥면 맞춤")
        self.btn_fit_ground.setToolTip("현재 자세를 유지한 채 메쉬 최저점을 XY 바닥(Z=0)에 맞춥니다.")
        self.addWidget(self.btn_fit_ground)
        
        self.btn_flat = QPushButton("🌓 Flat Shading")
        self.btn_flat.setCheckable(True)
        self.btn_flat.setToolTip("명암 없이 메쉬를 밝게 봅니다 (회전 시 어두워짐 방지)")
        self.addWidget(self.btn_flat)

        self.btn_xray = QPushButton("🩻 X-Ray")
        self.btn_xray.setCheckable(True)
        self.btn_xray.setToolTip("선택된 메쉬를 X-Ray(투명)로 표시합니다 (선택 객체만).")
        self.addWidget(self.btn_xray)

    def _create_spin(self, min_v, max_v, prefix="", step=None):
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setDecimals(2)
        spin.setPrefix(f"{prefix}: ")
        spin.setFixedWidth(90)
        try:
            if step is not None:
                spin.setSingleStep(float(step))
        except Exception:
            pass
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
        self.btn_surface_boundary = QPushButton("🧲 경계(면적+자석)")
        self.btn_surface_boundary.setToolTip(
            "면적(점-올가미) + 자석(경계 스냅)을 하나로 합친 도구입니다.\n"
            "좌클릭=점 추가(자석 스냅), 드래그=카메라 회전/시점, 우클릭/Enter=확정,\n"
            "Backspace=되돌리기, Shift/Ctrl=완드 정제, Alt=제거, [ / ]=자석 반경, ESC=종료"
        )
        self.btn_surface_boundary.clicked.connect(
            lambda: self.selectionRequested.emit(
                "surface_tool",
                {"tool": "boundary", "target": self.current_surface_target()},
            )
        )
        tool_row.addWidget(self.btn_surface_boundary)
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
        btn_auto_surface.setToolTip(
            "클릭=스마트(auto: 가시성(위상)→원통→법선), Shift+클릭=가시성(±두께축) 강제, Ctrl+클릭=원통(반경) 강제"
        )
        btn_auto_surface.clicked.connect(lambda: self.selectionChanged.emit('auto_surface', None))
        auto_layout.addWidget(btn_auto_surface)
        
        btn_auto_edge = QPushButton("📏 미구 자동 감지")
        btn_auto_edge.setToolTip(
            "미구(계단/경계) 영역을 자동으로 찾아 미구로 지정합니다.\n"
            "- 클릭: (가능하면) 원통 기반 미구, 아니면 Y축(기본) 강조 감지\n"
            "- Ctrl+클릭: X축 강조 감지\n"
            "- Shift+클릭: 둘레 경계(Edge belt) 감지"
        )
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
    captureRequested = pyqtSignal(float)    # height (capture current mesh slice)
    saveLayersRequested = pyqtSignal()      # snapshot to layers (for SVG export)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._presets: list[dict[str, Any]] = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. 활성화 스위치
        self.group = QGroupBox("📏 메쉬 단면 슬라이싱")
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

        # 2.5 Presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("프리셋:"))
        self.combo_presets = QComboBox()
        self.combo_presets.setToolTip("저장한 단면(클립) 높이 프리셋을 불러옵니다.")
        preset_layout.addWidget(self.combo_presets, 1)

        self.btn_preset_add = QPushButton("➕ 저장")
        self.btn_preset_add.setToolTip("현재 높이(Z)를 프리셋으로 저장합니다.")
        self.btn_preset_add.clicked.connect(self._on_preset_add_clicked)
        preset_layout.addWidget(self.btn_preset_add)

        self.btn_preset_apply = QPushButton("적용")
        self.btn_preset_apply.setToolTip("선택한 프리셋 높이를 적용합니다.")
        self.btn_preset_apply.clicked.connect(self._on_preset_apply_clicked)
        preset_layout.addWidget(self.btn_preset_apply)

        self.btn_preset_delete = QPushButton("삭제")
        self.btn_preset_delete.setToolTip("선택한 프리셋을 삭제합니다.")
        self.btn_preset_delete.clicked.connect(self._on_preset_delete_clicked)
        preset_layout.addWidget(self.btn_preset_delete)

        group_layout.addLayout(preset_layout)
        self._refresh_presets_ui()
        
        # 3. 버튼들
        btn_layout = QHBoxLayout()
        self.btn_export = QPushButton("💾 단면 SVG 내보내기")
        self.btn_export.setStyleSheet("background-color: #ebf8ff; font-weight: bold;")
        self.btn_export.clicked.connect(self.on_export_clicked)
        btn_layout.addWidget(self.btn_export)

        self.btn_capture = QPushButton("📸 현재 단면 촬영")
        self.btn_capture.setStyleSheet("background-color: #fff7ed; font-weight: bold;")
        self.btn_capture.setToolTip("현재 보이는 메쉬 단면을 레이어로 바로 저장합니다.")
        self.btn_capture.clicked.connect(self.on_capture_clicked)
        btn_layout.addWidget(self.btn_capture)

        self.btn_save_layers = QPushButton("🗂️ 레이어로 저장")
        self.btn_save_layers.setToolTip("현재 단면 결과(슬라이스/가이드/ROI)를 레이어로 스냅샷 저장합니다.")
        self.btn_save_layers.clicked.connect(self.saveLayersRequested.emit)
        btn_layout.addWidget(self.btn_save_layers)

        group_layout.addLayout(btn_layout)
        
        # 도움말
        help_label = QLabel(
            "상면(Top) 뷰에서 보면서 높이를 조절하세요. "
            "Ctrl+휠=실시간 단면 이동, Shift+Ctrl=미세, Alt+Ctrl=고속\n"
            "실시간 단면=3D 절단 관측/촬영, 2D 지정(단면선/ROI)=아래 도구에서 설정"
        )
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

    def on_capture_clicked(self):
        self.captureRequested.emit(self.spin.value())

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

    def get_presets(self) -> list[dict[str, Any]]:
        return [dict(p) for p in (self._presets or [])]

    def set_presets(self, presets: list[dict[str, Any]] | None) -> None:
        out: list[dict[str, Any]] = []
        for p in presets or []:
            if not isinstance(p, dict):
                continue
            try:
                z = float(p.get("z", p.get("height", 0.0)) or 0.0)
            except Exception:
                continue
            name = str(p.get("name", "")).strip() or f"Z={z:.2f}cm"
            out.append({"name": name, "z": z})
        self._presets = out
        self._refresh_presets_ui()

    def _refresh_presets_ui(self) -> None:
        combo = getattr(self, "combo_presets", None)
        if combo is None:
            return
        combo.blockSignals(True)
        try:
            combo.clear()
            for p in self._presets or []:
                combo.addItem(str(p.get("name", "")).strip() or "Preset", userData=float(p.get("z", 0.0) or 0.0))
        finally:
            combo.blockSignals(False)

        has = bool(self._presets)
        try:
            self.btn_preset_apply.setEnabled(has)
            self.btn_preset_delete.setEnabled(has)
        except Exception:
            pass

    def _unique_preset_name(self, base: str) -> str:
        base = str(base).strip() or "Preset"
        existing = {str(p.get("name", "")).strip() for p in (self._presets or [])}
        if base not in existing:
            return base
        n = 2
        while f"{base} ({n})" in existing:
            n += 1
        return f"{base} ({n})"

    def _on_preset_add_clicked(self) -> None:
        try:
            z = float(self.spin.value())
        except Exception:
            z = 0.0
        name = self._unique_preset_name(f"Z={z:.2f}cm")
        self._presets.append({"name": name, "z": z})
        self._refresh_presets_ui()
        try:
            self.combo_presets.setCurrentIndex(len(self._presets) - 1)
        except Exception:
            pass

    def _on_preset_apply_clicked(self) -> None:
        if not (self._presets and getattr(self, "combo_presets", None) is not None):
            return
        try:
            idx = int(self.combo_presets.currentIndex())
        except Exception:
            idx = -1
        if not (0 <= idx < len(self._presets)):
            return

        try:
            z = float(self._presets[idx].get("z", 0.0) or 0.0)
        except Exception:
            z = 0.0

        # Apply and enable slice mode.
        try:
            self.group.setChecked(True)
        except Exception:
            pass
        try:
            self.spin.setValue(z)
        except Exception:
            pass

    def _on_preset_delete_clicked(self) -> None:
        if not (self._presets and getattr(self, "combo_presets", None) is not None):
            return
        try:
            idx = int(self.combo_presets.currentIndex())
        except Exception:
            idx = -1
        if not (0 <= idx < len(self._presets)):
            return
        try:
            del self._presets[idx]
        except Exception:
            return
        self._refresh_presets_ui()


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

        self.combo_rubbing_target = QComboBox()
        self.combo_rubbing_target.addItems(["전체", "🌞 외면", "🌙 내면", "🧩 미구"])
        self.combo_rubbing_target.setToolTip(
            "탁본/디지털 탁본 내보내기 대상 표면을 선택합니다.\n"
            "- 전체: 전체 메쉬를 그대로 사용\n"
            "- 외면/내면/미구: 표면 지정 결과(face set)만 추출해 내보내기\n"
            "※ 대상 표면이 비어 있으면 먼저 '표면 선택/지정'으로 지정해 주세요."
        )
        img_layout.addRow("탁본 대상:", self.combo_rubbing_target)
        
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
        btn_export_rubbing.clicked.connect(
            lambda: self.exportRequested.emit(
                {'type': 'rubbing', 'target': self.current_rubbing_target()}
            )
        )
        layout.addWidget(btn_export_rubbing)

        btn_export_rubbing_digital = QPushButton("📤 디지털 탁본(곡률 제거) 내보내기")
        btn_export_rubbing_digital.setToolTip("원통 펼침(빠름) + 곡률 제거(참조면 스무딩) 기반 탁본")
        btn_export_rubbing_digital.setStyleSheet("""
            QPushButton {
                background-color: #805ad5;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #6b46c1; }
        """)
        btn_export_rubbing_digital.clicked.connect(
            lambda: self.exportRequested.emit(
                {'type': 'rubbing_digital', 'target': self.current_rubbing_target()}
            )
        )
        layout.addWidget(btn_export_rubbing_digital)
        
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

        btn_export_sheet_svg_digital = QPushButton("통합 SVG (디지털 탁본/원통)")
        btn_export_sheet_svg_digital.setToolTip("원통 펼침 + 곡률 제거(디지털 탁본)로 outer/inner 이미지를 생성합니다")
        btn_export_sheet_svg_digital.clicked.connect(lambda: self.exportRequested.emit({'type': 'sheet_svg_digital'}))
        mesh_layout.addWidget(btn_export_sheet_svg_digital)
        
        layout.addWidget(mesh_group)
        
        # 2D 외곽선 내보내기 (SVG/PDF)
        profile_group = QGroupBox("🛡️ 2D 실측 도면 내보내기 (SVG)")
        profile_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2b6cb0; }")
        profile_layout = QVBoxLayout(profile_group)
        
        # 안내 문구
        lbl_info = QLabel("격자는 이미지, 외곽선은 벡터로 저장됩니다.\n(지정된 뷰 방향에서 투영)")
        lbl_info.setStyleSheet("font-size: 11px; color: #718096;")
        profile_layout.addWidget(lbl_info)

        # 옵션: 격자/배경 포함
        opt_row = QHBoxLayout()
        self.check_profile_include_grid = QCheckBox("격자/배경 포함 (기본)")
        self.check_profile_include_grid.setChecked(True)
        self.check_profile_include_grid.setToolTip(
            "체크 시 1cm 격자+화면 캡처가 SVG에 배경 이미지로 포함됩니다(파일이 커짐).\n"
            "해제 시 벡터(외곽선/가이드)만 저장됩니다."
        )
        opt_row.addWidget(self.check_profile_include_grid)
        opt_row.addStretch(1)
        profile_layout.addLayout(opt_row)

        # 옵션: 샤프 엣지(능선) 라인 포함
        feature_row = QHBoxLayout()
        self.check_profile_feature_lines = QCheckBox("✨ 샤프 엣지(능선) 라인 포함")
        self.check_profile_feature_lines.setChecked(False)
        self.check_profile_feature_lines.setToolTip(
            "인접 면의 각도(디하이드럴)로 '날카로운 엣지'를 검출해 SVG에 선 레이어로 추가합니다.\n"
            "값이 낮을수록 선이 많아지고, 스캔 노이즈가 많으면 파일이 커질 수 있습니다."
        )
        feature_row.addWidget(self.check_profile_feature_lines, 1)

        feature_row.addWidget(QLabel("임계각:"))
        self.spin_profile_feature_angle = QDoubleSpinBox()
        self.spin_profile_feature_angle.setRange(0.0, 180.0)
        self.spin_profile_feature_angle.setSingleStep(5.0)
        self.spin_profile_feature_angle.setValue(60.0)
        self.spin_profile_feature_angle.setSuffix(" °")
        self.spin_profile_feature_angle.setToolTip("디하이드럴 각도 임계값(도).")
        self.spin_profile_feature_angle.setEnabled(False)
        self.check_profile_feature_lines.toggled.connect(self.spin_profile_feature_angle.setEnabled)
        feature_row.addWidget(self.spin_profile_feature_angle)
        profile_layout.addLayout(feature_row)
        
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

        btn_export_pkg = QPushButton("📦 6방향 패키지 내보내기")
        btn_export_pkg.setToolTip("Top/Bottom/Front/Back/Left/Right를 한 폴더에 '뷰별 하위 폴더'로 저장합니다")
        btn_export_pkg.clicked.connect(lambda: self.exportRequested.emit({"type": "profile_2d_package"}))
        profile_layout.addWidget(btn_export_pkg)
        layout.addWidget(profile_group)
        layout.addStretch(1)

    def current_rubbing_target(self) -> str:
        try:
            idx = int(getattr(self.combo_rubbing_target, "currentIndex", lambda: 0)())
        except Exception:
            idx = 0
        return {0: "all", 1: "outer", 2: "inner", 3: "migu"}.get(idx, "all")


class MeasurePanel(QWidget):
    """기본 치수(거리/지름) 측정 패널"""

    measureModeToggled = pyqtSignal(bool)
    fitCircleRequested = pyqtSignal()
    clearPointsRequested = pyqtSignal()
    copyResultsRequested = pyqtSignal()
    clearResultsRequested = pyqtSignal()
    computeVolumeRequested = pyqtSignal()
    modeChanged = pyqtSignal(str)  # "distance" | "diameter"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        hint = QLabel(
            "Shift+클릭으로 메쉬 위에 점을 찍어 치수를 측정합니다.\n"
            "거리=2점 선택 즉시 계산, 지름=3점 이상 선택 후 '지름 계산'을 누르세요."
        )
        hint.setStyleSheet("color: #718096; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.btn_measure_mode = QPushButton("📏 측정 모드 시작")
        self.btn_measure_mode.setCheckable(True)
        self.btn_measure_mode.setStyleSheet(
            "QPushButton:checked { background-color: #38a169; color: white; font-weight: bold; }"
        )
        self.btn_measure_mode.toggled.connect(self._on_measure_toggled)
        layout.addWidget(self.btn_measure_mode)

        mode_group = QGroupBox("측정 방식")
        mode_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        mode_layout = QFormLayout(mode_group)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["거리 (2점)", "지름/직경 (원 맞춤, 3점+)"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addRow("모드:", self.combo_mode)

        self.label_point_count = QLabel("선택된 포인트: 0")
        mode_layout.addRow("", self.label_point_count)

        btn_row = QHBoxLayout()
        self.btn_fit_circle = QPushButton("⭕ 지름 계산")
        self.btn_fit_circle.setToolTip("선택된 포인트(3점 이상)로 원을 맞추고 지름을 계산합니다.")
        self.btn_fit_circle.clicked.connect(self.fitCircleRequested.emit)
        self.btn_fit_circle.setEnabled(False)
        btn_row.addWidget(self.btn_fit_circle)

        self.btn_clear_points = QPushButton("🧹 포인트 초기화")
        self.btn_clear_points.clicked.connect(self.clearPointsRequested.emit)
        btn_row.addWidget(self.btn_clear_points)
        btn_row.addStretch(1)
        mode_layout.addRow(btn_row)

        self.btn_compute_volume = QPushButton("📦 부피/면적 계산")
        self.btn_compute_volume.setToolTip("선택된 메쉬의 표면적/부피를 계산합니다. (부피는 watertight 메쉬에서만 신뢰)")
        self.btn_compute_volume.clicked.connect(self.computeVolumeRequested.emit)
        mode_layout.addRow(self.btn_compute_volume)

        layout.addWidget(mode_group)

        result_group = QGroupBox("결과")
        result_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        result_layout = QVBoxLayout(result_group)

        self.text_results = QTextEdit()
        self.text_results.setReadOnly(True)
        self.text_results.setPlaceholderText("측정 결과가 여기에 기록됩니다.")
        self.text_results.setMinimumHeight(120)
        result_layout.addWidget(self.text_results)

        result_btn_row = QHBoxLayout()
        self.btn_copy = QPushButton("📋 복사")
        self.btn_copy.clicked.connect(self.copyResultsRequested.emit)
        result_btn_row.addWidget(self.btn_copy)

        self.btn_clear_results = QPushButton("🗑️ 지우기")
        self.btn_clear_results.clicked.connect(self.clearResultsRequested.emit)
        result_btn_row.addWidget(self.btn_clear_results)

        result_btn_row.addStretch(1)
        result_layout.addLayout(result_btn_row)

        layout.addWidget(result_group)
        layout.addStretch(1)

    @property
    def mode(self) -> str:
        try:
            return "diameter" if int(self.combo_mode.currentIndex()) == 1 else "distance"
        except Exception:
            return "distance"

    def set_points_count(self, n: int) -> None:
        try:
            self.label_point_count.setText(f"선택된 포인트: {int(n)}")
        except Exception:
            pass

    def append_result(self, text: str) -> None:
        try:
            if text:
                self.text_results.append(str(text))
        except Exception:
            pass

    def clear_results(self) -> None:
        try:
            self.text_results.clear()
        except Exception:
            pass

    def results_text(self) -> str:
        try:
            return str(self.text_results.toPlainText())
        except Exception:
            return ""

    def set_measure_checked(self, checked: bool) -> None:
        try:
            self.btn_measure_mode.blockSignals(True)
            self.btn_measure_mode.setChecked(bool(checked))
        except Exception:
            pass
        finally:
            try:
                self.btn_measure_mode.blockSignals(False)
            except Exception:
                pass
        try:
            self.btn_measure_mode.setText("📏 측정 모드 중지" if checked else "📏 측정 모드 시작")
        except Exception:
            pass

    def _on_measure_toggled(self, checked: bool):
        try:
            self.btn_measure_mode.setText("📏 측정 모드 중지" if checked else "📏 측정 모드 시작")
        except Exception:
            pass
        self.measureModeToggled.emit(bool(checked))

    def _on_mode_changed(self, _index: int):
        mode = self.mode
        try:
            self.btn_fit_circle.setEnabled(mode == "diameter")
        except Exception:
            pass
        self.modeChanged.emit(mode)


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
        
        # 2. 도움말
        help_label = QLabel("모드 활성 후 메쉬를 클릭/드래그하여 단면을 확인하세요.")
        help_label.setStyleSheet("color: #718096; font-size: 10px;")
        help_label.setWordWrap(True)
        
        # 3. 그래프 공간
        self.label_x = QLabel("X-Profile (Yellow Line)")
        self.graph_x = ProfileGraphWidget("가로 단면 (X-Profile)")
        
        self.label_y = QLabel("Y-Profile (Cyan Line)")
        self.graph_y = ProfileGraphWidget("세로 단면 (Y-Profile)")
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        # XY 십자선/프로파일 UI는 단면 도구 단순화 요청으로 숨김 처리
        self.btn_toggle.setVisible(False)
        help_label.setVisible(False)
        self.label_x.setVisible(False)
        self.graph_x.setVisible(False)
        self.label_y.setVisible(False)
        self.graph_y.setVisible(False)
        line.setVisible(False)

        # 4. 2D 단면선(2개) - 상면에서 가로/세로(꺾임 가능) 가이드 라인
        line_group = QGroupBox("✏️ 2D 단면선 지정 (상면, 2개)")
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
            "메쉬 위를 클릭해도 자동으로 상면(XY)으로 투영됩니다.\n"
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
        
        # 5. 2D ROI 영역 지정 (상면 투영)
        roi_group = QGroupBox("✂️ 2D 영역 지정 (상면 Cropping)")
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
        
        roi_help = QLabel(
            "상면(Top) 뷰에서 4개 화살표 드래그=크기 조절, 가운데 마름모 드래그=이동.\n"
            "Shift+드래그=새 영역 지정 (드래그=카메라 회전 / 우클릭 드래그=이동)"
        )
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

    UI_STATE_VERSION = 6
    
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
        
        self.mesh_loader = MeshLoader(default_unit=DEFAULT_MESH_UNIT)
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

        # Slice 계산은 디바운스 + 백그라운드 스레드로 처리 (UI 끊김 방지)
        self._slice_debounce_timer = QTimer(self)
        self._slice_debounce_timer.setSingleShot(True)
        self._slice_debounce_timer.timeout.connect(self._request_slice_compute)
        self._slice_compute_thread = None
        self._slice_pending_height = None
        self._slice_capture_pending = False

        # Project (.amr)
        self._current_project_path: str | None = None
        self._project_load_active: bool = False
        self._project_load_queue: list[dict[str, Any]] = []
        self._project_load_state: dict[str, Any] | None = None
        self._project_load_current: dict[str, Any] | None = None
        
        self.init_ui()
        self.init_menu()
        self.init_toolbar()
        self.init_statusbar()
        self._restore_ui_state()
        self._hide_unused_docks()
    
    def init_ui(self):
        # 중앙 위젯 (3D 뷰포트)
        self.viewport = Viewport3D()
        self.setCentralWidget(self.viewport)
        
        # 레이어 매니저 연결
        self.viewport.selectionChanged.connect(self.on_selection_changed)
        self.viewport.meshLoaded.connect(self.on_mesh_loaded)
        self.viewport.meshTransformChanged.connect(self.sync_transform_panel)
        self.viewport.floorPointPicked.connect(self.on_floor_point_picked)
        self.viewport.floorFacePicked.connect(self.on_floor_face_picked)
        self.viewport.alignToBrushSelected.connect(self.on_align_to_brush_selected)
        self.viewport.floorAlignmentConfirmed.connect(self.on_floor_alignment_confirmed)
        self.viewport.surfaceAssignmentChanged.connect(self.on_surface_assignment_changed)
        self.viewport.measurePointPicked.connect(self.on_measure_point_picked)
        
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
        self.trans_toolbar.btn_fit_ground.clicked.connect(self.fit_ground_plane)
        self.trans_toolbar.btn_flat.toggled.connect(self.toggle_flat_shading)
        self.trans_toolbar.btn_xray.toggled.connect(self.toggle_xray_mode)
        
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
        try:
            self.flatten_dock.visibilityChanged.connect(self._on_flatten_dock_visibility_changed)
        except Exception:
            pass

        # 4) 내보내기
        self.export_dock = QDockWidget("📤 내보내기", self)
        self.export_dock.setObjectName("dock_export")
        self.export_panel = ExportPanel()
        self.export_panel.exportRequested.connect(self.on_export_requested)
        self.export_dock.setWidget(self.export_panel)

        # 4.5) 치수 측정
        self.measure_dock = QDockWidget("📏 치수 측정", self)
        self.measure_dock.setObjectName("dock_measure")
        self.measure_panel = MeasurePanel()
        self.measure_panel.measureModeToggled.connect(self.toggle_measure_mode)
        self.measure_panel.fitCircleRequested.connect(self.fit_measure_circle)
        self.measure_panel.clearPointsRequested.connect(self.clear_measure_points)
        self.measure_panel.copyResultsRequested.connect(self.copy_measure_results)
        self.measure_panel.clearResultsRequested.connect(self.clear_measure_results)
        self.measure_panel.computeVolumeRequested.connect(self.compute_volume_stats)
        self.measure_panel.modeChanged.connect(self.on_measure_mode_changed)
        self.measure_dock.setWidget(self.measure_panel)

        # 5) 단면/2D 지정 도구 (슬라이싱 + 십자선 + 라인 + ROI)
        self.section_dock = QDockWidget("📏 단면/2D 지정 도구 (Section)", self)
        self.section_dock.setObjectName("dock_section")
        section_scroll = QScrollArea()
        section_scroll.setWidgetResizable(True)
        section_content = QWidget()
        section_layout = QVBoxLayout(section_content)

        # Section dock is simplified to line/ROI only.
        self.slice_panel = None

        mode_hint = QLabel("구분: 2D 지정 = 상면에서 단면선/ROI 가이드 지정")
        mode_hint.setStyleSheet("color: #4a5568; font-size: 10px;")
        mode_hint.setWordWrap(True)
        section_layout.addWidget(mode_hint)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        section_layout.addWidget(line)

        self.section_panel = SectionPanel()
        self.section_panel.lineSectionToggled.connect(self.on_line_section_toggled)
        self.section_panel.cutLineActiveChanged.connect(self.on_cut_line_active_changed)
        self.section_panel.cutLineClearRequested.connect(self.on_cut_line_clear_requested)
        self.section_panel.cutLinesClearAllRequested.connect(self.on_cut_lines_clear_all_requested)
        self.section_panel.roiToggled.connect(self.on_roi_toggled)
        self.section_panel.silhouetteRequested.connect(self.viewport.extract_roi_silhouette)
        self.section_panel.saveSectionLayersRequested.connect(self.on_save_section_layers_requested)

        self.viewport.lineProfileUpdated.connect(self.section_panel.update_line_profile)
        self.viewport.roiSilhouetteExtracted.connect(self.on_silhouette_extracted)
        self.viewport.cutLinesAutoEnded.connect(self._on_cut_lines_auto_ended)
        self.viewport.cutLinesEnabledChanged.connect(self._sync_cutline_button_state)
        self.viewport.roiSectionCommitRequested.connect(self.on_roi_section_commit_requested)
        section_layout.addWidget(self.section_panel)

        section_layout.addStretch()
        section_scroll.setWidget(section_content)
        self.section_dock.setWidget(section_scroll)

        # 7) 레이어
        self.scene_dock = QDockWidget("🌲 레이어", self)
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
            self.flatten_dock,
            self.section_dock,
            self.export_dock,
            self.measure_dock,
            self.scene_dock,
        ]:
            dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
            dock.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                | QDockWidget.DockWidgetFeature.DockWidgetClosable
            )

        # 기본 레이아웃(일러스트레이터 스타일: 상단 정보/정치, 우측 분리, 레이어는 우측 하단)
        self._apply_default_dock_layout()

    def _settings(self) -> QSettings:
        return QSettings("ArchMeshRubbing", "ArchMeshRubbing")

    def _apply_default_dock_layout(self):
        """기본 도킹 레이아웃 적용 (저장된 레이아웃이 없을 때의 초기 배치)"""
        for dock in [
            self.info_dock,
            self.flatten_dock,
            self.section_dock,
            self.export_dock,
            self.measure_dock,
            self.scene_dock,
        ]:
            # 기존 배치가 남아있으면(중복 split/tabify 등) 레이아웃이 꼬일 수 있어 초기화
            try:
                self.removeDockWidget(dock)
            except Exception:
                pass
            dock.setFloating(False)
            dock.show()

        # 상단: 파일/메쉬 정보
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.info_dock)

        # 우측: 단면 + 펼침 + 내보내기(+치수)는 탭, 레이어는 우측 하단
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.section_dock)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.flatten_dock)
        self.tabifyDockWidget(self.section_dock, self.flatten_dock)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.export_dock)
        self.tabifyDockWidget(self.section_dock, self.export_dock)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.measure_dock)
        self.tabifyDockWidget(self.section_dock, self.measure_dock)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.scene_dock)
        self.splitDockWidget(self.section_dock, self.scene_dock, Qt.Orientation.Vertical)

        # 크기 비율(대략적인 기본값)
        self.resizeDocks([self.section_dock, self.scene_dock], [780, 220], Qt.Orientation.Vertical)

        self.section_dock.raise_()
        self._hide_unused_docks()

    def _on_flatten_dock_visibility_changed(self, visible: bool) -> None:
        """펼침 탭이 활성화되면(보이면) 기본 도구를 '경계(면적+자석)'로 맞춥니다.

        다른 피킹 모드가 이미 켜져 있으면(예: 단면/ROI 등) 덮어쓰지 않습니다.
        """
        try:
            if not bool(visible):
                return
        except Exception:
            return

        try:
            if str(getattr(self.viewport, "picking_mode", "none")) != "none":
                return
        except Exception:
            return

        obj = getattr(self.viewport, "selected_obj", None)
        if obj is None or getattr(obj, "mesh", None) is None:
            return

        try:
            target = self.flatten_panel.current_surface_target()
        except Exception:
            target = "outer"

        try:
            self.on_selection_action("surface_tool", {"tool": "boundary", "target": target})
        except Exception:
            pass

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

    def _hide_unused_docks(self):
        for dock in (getattr(self, "transform_dock", None), getattr(self, "help_dock", None)):
            if dock is None:
                continue
            try:
                self.removeDockWidget(dock)
            except Exception:
                pass
            try:
                dock.setFloating(False)
            except Exception:
                pass
            try:
                dock.hide()
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
        if a0 is None:
            return
        reply = QMessageBox.question(
            self,
            "종료 확인",
            "정말 종료하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            a0.ignore()
            return
        self._save_ui_state()
        super().closeEvent(a0)

    def start_floor_picking(self):
        """바닥면 그리기(점 찍기) 모드 시작"""
        if self.viewport.selected_obj is None:
            return
        # X-Ray는 바닥면 판독을 방해하고 "방충망"처럼 보여 정렬 오판을 유발할 수 있어 자동 해제.
        try:
            if bool(getattr(self.viewport, "xray_mode", False)):
                self.viewport.xray_mode = False
                btn_xray = getattr(getattr(self, "trans_toolbar", None), "btn_xray", None)
                if btn_xray is not None:
                    btn_xray.blockSignals(True)
                    btn_xray.setChecked(False)
                    btn_xray.blockSignals(False)
        except Exception:
            pass
        try:
            self._disable_measure_mode()
        except Exception:
            pass
        self.viewport.picking_mode = 'floor_3point'
        self.viewport.floor_picks = []
        self.viewport.status_info = "📍 바닥면 점 찍기: 메쉬 위를 클릭하여 점을 추가하세요 (Enter로 확정)"
        self.viewport.update()

    def start_floor_picking_face(self):
        """면 선택 바닥 정렬 모드 시작"""
        if self.viewport.selected_obj is None:
            return
        try:
            if bool(getattr(self.viewport, "xray_mode", False)):
                self.viewport.xray_mode = False
                btn_xray = getattr(getattr(self, "trans_toolbar", None), "btn_xray", None)
                if btn_xray is not None:
                    btn_xray.blockSignals(True)
                    btn_xray.setChecked(False)
                    btn_xray.blockSignals(False)
        except Exception:
            pass
        try:
            self._disable_measure_mode()
        except Exception:
            pass
        self.viewport.picking_mode = 'floor_face'
        self.viewport.status_info = "📐 바닥면이 될 삼각형 면(Triangle)을 클릭하세요..."
        self.viewport.update()

    def start_floor_picking_brush(self):
        """브러시 바닥 정렬 모드 시작"""
        if self.viewport.selected_obj is None:
            return
        try:
            if bool(getattr(self.viewport, "xray_mode", False)):
                self.viewport.xray_mode = False
                btn_xray = getattr(getattr(self, "trans_toolbar", None), "btn_xray", None)
                if btn_xray is not None:
                    btn_xray.blockSignals(True)
                    btn_xray.setChecked(False)
                    btn_xray.blockSignals(False)
        except Exception:
            pass
        try:
            self._disable_measure_mode()
        except Exception:
            pass
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

    def align_mesh_to_normal(self, normal, *, pivot=None) -> np.ndarray | None:
        """주어진 법선을 월드 +Z로 정렬 (메쉬에 직접 반영/Bake)."""
        obj = self.viewport.selected_obj
        if not obj:
            return

        target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        R = rotation_matrix_align_vectors(normal, target)

        try:
            pivot_v = (
                np.asarray(pivot, dtype=np.float64).reshape(3)
                if pivot is not None
                else np.array([0.0, 0.0, 0.0], dtype=np.float64)
            )
        except Exception:
            pivot_v = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        vertices = np.asarray(obj.mesh.vertices, dtype=np.float64)
        obj.mesh.vertices = ((R @ (vertices - pivot_v).T).T + pivot_v).astype(np.float32)
        try:
            obj.mesh._bounds = None
            obj.mesh._centroid = None
            obj.mesh._surface_area = None
        except Exception:
            pass
        obj.mesh.compute_normals(compute_vertex_normals=False, force=True)
        obj._trimesh = None
        obj.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.viewport.update_vbo(obj)
        self.sync_transform_panel()
        return R

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

        points = np.asarray(self.viewport.floor_picks, dtype=np.float64).reshape(-1, 3)
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) < 3:
            self.viewport.status_info = "❌ 점이 부족합니다. 더 찍어주세요."
            self.viewport.update()
            return
            
        # 1) floor_picks는 월드 좌표이므로 메쉬도 월드 기준 정점으로 맞춘다.
        self.viewport.bake_object_transform(obj)

        # 2) 점 기반 평면을 robust하게 추정한다.
        plane = fit_plane_normal(points, robust=True)
        if plane is None:
            self.viewport.status_info = "❌ 선택 점이 거의 일직선입니다. 점을 다시 찍어주세요."
            self.viewport.update()
            return
        normal, centroid = plane

        # 법선 방향을 메쉬 중심 쪽으로 맞춰 뒤집힘을 줄인다.
        try:
            mesh_centroid = np.asarray(obj.mesh.centroid, dtype=np.float64).reshape(3)
        except Exception:
            mesh_centroid = np.mean(np.asarray(obj.mesh.vertices, dtype=np.float64), axis=0)
        normal = orient_plane_normal_toward(normal, centroid, mesh_centroid)

        # 3) 법선 정렬
        self.viewport.save_undo_state()
        R = self.align_mesh_to_normal(normal, pivot=centroid)
        if R is None:
            self.viewport.status_info = "바닥 정렬 중 회전 계산에 실패했습니다."
            self.viewport.update()
            return
        points_rotated = (R @ (points - centroid).T).T + centroid

        # 4) 선택 바닥 평면을 Z=0으로 이동
        try:
            plane_z = float(np.nanmedian(np.asarray(points_rotated, dtype=np.float64)[:, 2]))
        except Exception:
            plane_z = 0.0
        if np.isfinite(plane_z):
            obj.mesh.vertices[:, 2] -= plane_z

        # 5) 경미한 침투만 자동 보정한다(큰 보정은 사용자 의도와 다를 수 있어 보정 안 함).
        auto_shift = 0.0
        try:
            z_vals = np.asarray(obj.mesh.vertices[:, 2], dtype=np.float64)
            auto_shift = compute_floor_contact_shift(z_vals, tolerance=0.02, max_auto_shift=0.2)
            if auto_shift > 0.0:
                obj.mesh.vertices[:, 2] += float(auto_shift)
        except Exception:
            auto_shift = 0.0

        try:
            obj.mesh._bounds = None
            obj.mesh._centroid = None
            obj.mesh._surface_area = None
        except Exception:
            pass
        obj._trimesh = None
        obj.translation = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self.viewport.update_vbo(obj)
        self.sync_transform_panel()
        if auto_shift > 0.0:
            self.viewport.status_info = (
                f"✅ 바닥 정렬 완료 (점 {len(points)}개 / 경미한 침투 보정 +{auto_shift:.3f})"
            )
        else:
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

        action_open_project = QAction("📁 프로젝트 열기…", self)
        action_open_project.setShortcut(QKeySequence("Ctrl+Shift+O"))
        action_open_project.triggered.connect(self.open_project)
        file_menu.addAction(action_open_project)

        file_menu.addSeparator()

        action_save_project = QAction("💾 프로젝트 저장", self)
        action_save_project.setShortcut(QKeySequence.StandardKey.Save)
        action_save_project.triggered.connect(self.save_project)
        file_menu.addAction(action_save_project)

        action_save_project_as = QAction("💾 프로젝트 다른 이름 저장…", self)
        action_save_project_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        action_save_project_as.triggered.connect(self.save_project_as)
        file_menu.addAction(action_save_project_as)
        
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
        action_top.triggered.connect(lambda: self.set_view(0, 90))
        view_menu.addAction(action_top)
        
        action_bottom = QAction("6️⃣ 하면 뷰", self)
        action_bottom.setShortcut("6")
        action_bottom.triggered.connect(lambda: self.set_view(0, -90))
        view_menu.addAction(action_bottom)

        view_menu.addSeparator()

        action_reset_layout = QAction("패널 레이아웃 초기화", self)
        action_reset_layout.triggered.connect(self.reset_panel_layout)
        view_menu.addAction(action_reset_layout)

        panels_menu = view_menu.addMenu("패널 표시/숨김")
        if panels_menu is not None:
            panels_menu.addAction(self.info_dock.toggleViewAction())
            panels_menu.addAction(self.flatten_dock.toggleViewAction())
            panels_menu.addAction(self.section_dock.toggleViewAction())
            panels_menu.addAction(self.export_dock.toggleViewAction())
            panels_menu.addAction(self.scene_dock.toggleViewAction())
        
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
        action_top.triggered.connect(lambda: self.set_view(0, 90))
        toolbar.addAction(action_top)
        
        action_bottom = QAction("하면", self)
        action_bottom.setToolTip("하면 뷰 (6)")
        action_bottom.triggered.connect(lambda: self.set_view(0, -90))
        toolbar.addAction(action_bottom)

        toolbar.addSeparator()

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

        # 우측 하단 작업 진행바(작고 비침투적으로)
        self._status_task_count = 0
        self._status_task_widget = QWidget()
        task_layout = QHBoxLayout(self._status_task_widget)
        task_layout.setContentsMargins(0, 0, 0, 0)
        task_layout.setSpacing(6)
        self._status_task_label = QLabel("")
        self._status_task_label.setStyleSheet("color: #718096; font-size: 10px;")
        self._status_task_bar = QProgressBar()
        self._status_task_bar.setTextVisible(False)
        self._status_task_bar.setFixedWidth(120)
        self._status_task_bar.setFixedHeight(12)
        self._status_task_bar.setRange(0, 0)  # indeterminate by default
        task_layout.addWidget(self._status_task_label)
        task_layout.addWidget(self._status_task_bar)
        self._status_task_widget.setVisible(False)
        self.statusbar.addPermanentWidget(self._status_task_widget)

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
            "3D Files (*.obj *.ply *.stl *.off *.gltf *.glb);;All Files (*)"
        )
        
        if filepath:
            # 단위 선택 다이얼로그
            dialog = UnitSelectionDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                scale_factor = dialog.get_scale_factor()
                self.load_mesh(filepath, scale_factor)

    def open_project(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "프로젝트 열기",
            "",
            "ArchMeshRubbing Project (*.amr);;All Files (*)",
        )
        if not filepath:
            return
        self.open_project_path(filepath)

    def open_project_path(self, filepath: str) -> None:
        """Open a project file (.amr) from a known path (no file dialog)."""
        if not filepath:
            return

        try:
            doc = load_amr_project(filepath)
            state = doc.get("state", {})
        except (OSError, ProjectFormatError) as e:
            QMessageBox.critical(self, "오류", f"프로젝트를 열 수 없습니다:\n{e}")
            return
        except Exception as e:
            QMessageBox.critical(self, "오류", f"프로젝트 열기 중 오류 발생:\n{type(e).__name__}: {e}")
            return

        objects = state.get("objects", [])
        if not isinstance(objects, list) or not objects:
            QMessageBox.warning(self, "경고", "프로젝트에 로드할 객체(objects)가 없습니다.")
            return

        # Reset scene and start queued mesh loads
        try:
            self.viewport.clear_scene()
        except Exception:
            try:
                self.viewport.objects = []
                self.viewport.selected_index = -1
                self.viewport.picking_mode = "none"
                self.viewport.update()
            except Exception:
                pass

        self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
        self.current_mesh = None
        self.current_filepath = None

        self._current_project_path = str(filepath)
        self._project_load_active = True
        self._project_load_state = state if isinstance(state, dict) else {}
        self._project_load_queue = [o for o in objects if isinstance(o, dict)]
        self._project_load_current = None

        self.status_info.setText(f"📁 프로젝트 로딩 중: {Path(filepath).name}")
        self._start_next_project_object_load()

    def save_project(self) -> None:
        if not getattr(self, "_current_project_path", None):
            self.save_project_as()
            return
        self._write_project(str(self._current_project_path))

    def save_project_as(self) -> None:
        default_name = DEFAULT_PROJECT_FILENAME
        try:
            if self.current_filepath:
                default_name = str(Path(str(self.current_filepath)).with_suffix(".amr").name)
        except Exception:
            default_name = DEFAULT_PROJECT_FILENAME

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "프로젝트 저장",
            default_name,
            "ArchMeshRubbing Project (*.amr);;All Files (*)",
        )
        if not filepath:
            return

        if not str(filepath).lower().endswith(".amr"):
            filepath = str(filepath) + ".amr"

        if self._write_project(filepath):
            self._current_project_path = str(filepath)

    def _write_project(self, filepath: str) -> bool:
        try:
            state = self._collect_project_state()

            sha, dirty = _safe_git_info(str(Path(basedir)))
            meta = {
                "app": APP_NAME,
                "version": APP_VERSION,
                "git": f"{sha}{'*' if dirty else ''}" if sha else "unknown",
            }
            save_amr_project(filepath, state, meta=meta)
            self.status_info.setText(f"✅ 프로젝트 저장: {Path(filepath).name}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "오류", f"프로젝트 저장 실패:\n{type(e).__name__}: {e}")
            self.status_info.setText("❌ 프로젝트 저장 실패")
            return False

    def _collect_project_state(self) -> dict[str, Any]:
        vp = self.viewport

        def f3(v) -> list[float]:
            try:
                arr = np.asarray(v, dtype=np.float64).reshape(-1)
                if arr.size >= 3:
                    return [float(arr[0]), float(arr[1]), float(arr[2])]
            except Exception:
                pass
            return [0.0, 0.0, 0.0]

        def f2(v) -> list[float]:
            try:
                arr = np.asarray(v, dtype=np.float64).reshape(-1)
                if arr.size >= 2:
                    return [float(arr[0]), float(arr[1])]
            except Exception:
                pass
            return [0.0, 0.0]

        def to_int_list(s) -> list[int]:
            try:
                return [int(x) for x in sorted(list(s or []))]
            except Exception:
                return []

        def to_safe_assist_meta(meta_src: Any) -> dict[str, Any]:
            if not isinstance(meta_src, dict):
                return {}
            keep_keys = (
                "status",
                "method",
                "auto_method",
                "auto_mapping",
                "assist_mode",
                "conservative",
                "seed_outer_count",
                "seed_inner_count",
                "added_outer_count",
                "added_inner_count",
                "unknown_count",
                "unresolved_count",
                "unresolved_truncated",
                "migu_count",
                "direct_hits",
                "swapped_hits",
                "rule_used",
                "rule_sep_ratio",
            )
            out: dict[str, Any] = {}
            for k in keep_keys:
                if k not in meta_src:
                    continue
                v = meta_src.get(k)
                if isinstance(v, (str, bool, int)):
                    out[str(k)] = v
                elif isinstance(v, (np.integer,)):
                    out[str(k)] = int(v)
                elif isinstance(v, (float, np.floating)):
                    fv = float(v)
                    if np.isfinite(fv):
                        out[str(k)] = fv
            return out

        objects: list[dict[str, Any]] = []
        for obj in getattr(vp, "objects", []) or []:
            mesh = getattr(obj, "mesh", None)
            mesh_path = None
            try:
                fp = getattr(mesh, "filepath", None)
                if fp:
                    mesh_path = str(fp)
            except Exception:
                mesh_path = None

            try:
                source_scale = float(getattr(mesh, "_amr_source_scale_factor", 1.0))
            except Exception:
                source_scale = 1.0

            # Polyline layers (sections/guides)
            poly_layers: list[dict[str, Any]] = []
            for layer in getattr(obj, "polyline_layers", []) or []:
                try:
                    pts = []
                    for p in layer.get("points", []) or []:
                        arr = np.asarray(p, dtype=np.float64).reshape(-1)
                        if arr.size >= 3:
                            pts.append([float(arr[0]), float(arr[1]), float(arr[2])])
                        elif arr.size == 2:
                            pts.append([float(arr[0]), float(arr[1]), 0.0])
                    poly_layers.append(
                        {
                            "name": str(layer.get("name", "")).strip(),
                            "kind": str(layer.get("kind", "")).strip(),
                            "visible": bool(layer.get("visible", True)),
                            "offset": f2(layer.get("offset", [0.0, 0.0])),
                            "color": [float(x) for x in (layer.get("color", [0.1, 0.1, 0.1, 0.9]) or [])][:4],
                            "width": float(layer.get("width", 2.0) or 2.0),
                            "points": pts,
                        }
                    )
                except Exception:
                    continue

            # Fitted arcs (curvature)
            arcs_state: list[dict[str, Any]] = []
            for arc in getattr(obj, "fitted_arcs", []) or []:
                try:
                    arcs_state.append(
                        {
                            "center": f3(getattr(arc, "center", [0, 0, 0])),
                            "radius": float(getattr(arc, "radius", 0.0) or 0.0),
                            "normal": f3(getattr(arc, "normal", [0, 0, 1])),
                            "plane_origin": f3(getattr(arc, "plane_origin", [0, 0, 0])),
                            "plane_u": f3(getattr(arc, "plane_u", [1, 0, 0])),
                            "plane_v": f3(getattr(arc, "plane_v", [0, 1, 0])),
                            "points_2d": (
                                np.asarray(getattr(arc, "points_2d", np.zeros((0, 2))), dtype=np.float64)
                                .reshape(-1, 2)
                                .tolist()
                            ),
                        }
                    )
                except Exception:
                    continue

            objects.append(
                {
                    "name": str(getattr(obj, "name", "")).strip() or "Object",
                    "visible": bool(getattr(obj, "visible", True)),
                    "mesh": {"path": mesh_path, "source_scale_factor": source_scale},
                    "transform": {
                        "translation": f3(getattr(obj, "translation", [0, 0, 0])),
                        "rotation_deg": f3(getattr(obj, "rotation", [0, 0, 0])),
                        "scale": float(getattr(obj, "scale", 1.0) or 1.0),
                        "fixed_state_valid": bool(getattr(obj, "fixed_state_valid", False)),
                        "fixed_translation": f3(getattr(obj, "fixed_translation", [0, 0, 0])),
                        "fixed_rotation_deg": f3(getattr(obj, "fixed_rotation", [0, 0, 0])),
                        "fixed_scale": float(getattr(obj, "fixed_scale", 1.0) or 1.0),
                    },
                    "faces": {
                        "selected": to_int_list(getattr(obj, "selected_faces", set())),
                        "outer": to_int_list(getattr(obj, "outer_face_indices", set())),
                        "inner": to_int_list(getattr(obj, "inner_face_indices", set())),
                        "migu": to_int_list(getattr(obj, "migu_face_indices", set())),
                        "assist_unresolved": to_int_list(
                            getattr(obj, "surface_assist_unresolved_face_indices", set())
                        ),
                        "assist_meta": to_safe_assist_meta(getattr(obj, "surface_assist_meta", {})),
                    },
                    "polylines": poly_layers,
                    "arcs": arcs_state,
                }
            )

        cam = getattr(vp, "camera", None)
        viewport_state: dict[str, Any] = {
            "selected_index": int(getattr(vp, "selected_index", -1) or -1),
            "grid_spacing": float(getattr(vp, "grid_spacing", 1.0) or 1.0),
            "grid_size": float(getattr(vp, "grid_size", 500.0) or 500.0),
            "flat_shading": bool(getattr(vp, "flat_shading", False)),
            "xray_mode": bool(getattr(vp, "xray_mode", False)),
            "xray_alpha": float(getattr(vp, "xray_alpha", 0.25) or 0.25),
            "camera": {
                "distance": float(getattr(cam, "distance", 50.0) or 50.0) if cam is not None else 50.0,
                "azimuth": float(getattr(cam, "azimuth", 45.0) or 45.0) if cam is not None else 45.0,
                "elevation": float(getattr(cam, "elevation", 30.0) or 30.0) if cam is not None else 30.0,
                "center": f3(getattr(cam, "center", [0, 0, 0])) if cam is not None else [0.0, 0.0, 0.0],
                "pan_offset": f3(getattr(cam, "pan_offset", [0, 0, 0])) if cam is not None else [0.0, 0.0, 0.0],
            },
            "slice": {
                "enabled": bool(getattr(vp, "slice_enabled", False)),
                "z": float(getattr(vp, "slice_z", 0.0) or 0.0),
            },
            "crosshair": {
                "enabled": bool(getattr(vp, "crosshair_enabled", False)),
                "pos": f2(getattr(vp, "crosshair_pos", [0.0, 0.0])),
            },
            "roi": {
                "enabled": bool(getattr(vp, "roi_enabled", False)),
                "bounds": [float(x) for x in (getattr(vp, "roi_bounds", [-10, 10, -10, 10]) or [])][:4],
                "caps": bool(getattr(vp, "roi_caps_enabled", False)),
            },
            "cut_lines": {
                "enabled": bool(getattr(vp, "cut_lines_enabled", False)),
                "active": int(getattr(vp, "cut_line_active", 0) or 0),
                "final": [bool(x) for x in (getattr(vp, "_cut_line_final", [False, False]) or [False, False])][:2],
                "lines": [
                    [f3(p) for p in (line or [])]
                    for line in (getattr(vp, "cut_lines", [[], []]) or [[], []])[:2]
                ],
            },
        }

        ui_state: dict[str, Any] = {}

        # Flatten panel state
        flatten_panel = getattr(self, "flatten_panel", None)
        if flatten_panel is not None:
            try:
                radius_mm = float(flatten_panel.spin_radius.value())
            except Exception:
                radius_mm = 150.0
            try:
                direction_index = int(flatten_panel.combo_direction.currentIndex())
            except Exception:
                direction_index = 0
            try:
                method_index = int(flatten_panel.combo_method.currentIndex())
            except Exception:
                method_index = 0
            try:
                distortion_percent = int(flatten_panel.slider_distortion.value())
            except Exception:
                distortion_percent = 50
            try:
                auto_cut = bool(flatten_panel.check_auto_cut.isChecked())
            except Exception:
                auto_cut = False
            try:
                multiband = bool(flatten_panel.check_multiband.isChecked())
            except Exception:
                multiband = False
            try:
                iterations = int(flatten_panel.spin_iterations.value())
            except Exception:
                iterations = 30
        else:
            radius_mm = 150.0
            direction_index = 0
            method_index = 0
            distortion_percent = 50
            auto_cut = False
            multiband = False
            iterations = 30

        ui_state["flatten"] = {
            "radius_mm": float(radius_mm),
            "direction_index": int(direction_index),
            "method_index": int(method_index),
            "distortion_percent": int(distortion_percent),
            "auto_cut": bool(auto_cut),
            "multiband": bool(multiband),
            "iterations": int(iterations),
        }

        # Export panel state
        export_panel = getattr(self, "export_panel", None)
        if export_panel is not None:
            try:
                dpi = int(export_panel.spin_dpi.value())
            except Exception:
                dpi = 300
            try:
                format_index = int(export_panel.combo_format.currentIndex())
            except Exception:
                format_index = 0
            try:
                scale_bar = bool(export_panel.check_scale_bar.isChecked())
            except Exception:
                scale_bar = True
            try:
                profile_include_grid = bool(export_panel.check_profile_include_grid.isChecked())
            except Exception:
                profile_include_grid = True
            try:
                profile_feature_lines = bool(export_panel.check_profile_feature_lines.isChecked())
            except Exception:
                profile_feature_lines = False
            try:
                profile_feature_angle = float(export_panel.spin_profile_feature_angle.value())
            except Exception:
                profile_feature_angle = 60.0
        else:
            dpi = 300
            format_index = 0
            scale_bar = True
            profile_include_grid = True
            profile_feature_lines = False
            profile_feature_angle = 60.0

        ui_state["export"] = {
            "dpi": int(dpi),
            "format_index": int(format_index),
            "scale_bar": bool(scale_bar),
            "profile_include_grid": bool(profile_include_grid),
            "profile_feature_lines": bool(profile_feature_lines),
            "profile_feature_angle": float(profile_feature_angle),
        }

        slice_panel = getattr(self, "slice_panel", None)
        ui_state["slice"] = {
            "presets": slice_panel.get_presets() if slice_panel is not None else [],
        }

        return {
            "objects": objects,
            "viewport": viewport_state,
            "ui": ui_state,
        }

    def _start_next_project_object_load(self) -> None:
        if not bool(getattr(self, "_project_load_active", False)):
            return

        queue = getattr(self, "_project_load_queue", None)
        if not queue:
            return

        obj_state = queue.pop(0)
        self._project_load_current = obj_state

        mesh_info = obj_state.get("mesh", {}) if isinstance(obj_state, dict) else {}
        if not isinstance(mesh_info, dict):
            mesh_info = {}

        mesh_path = str(mesh_info.get("path", "") or "").strip()
        if not mesh_path or not Path(mesh_path).exists():
            mesh_path, _ = QFileDialog.getOpenFileName(
                self,
                "프로젝트 메쉬 파일 찾기",
                "",
                "3D Files (*.obj *.ply *.stl *.off *.gltf *.glb);;All Files (*)",
            )
            if not mesh_path:
                # Skip this object
                self._project_load_current = None
                self._start_next_project_object_load()
                return
            mesh_info["path"] = mesh_path
            obj_state["mesh"] = mesh_info

        try:
            scale_factor = float(mesh_info.get("source_scale_factor", 1.0) or 1.0)
        except Exception:
            scale_factor = 1.0

        self._start_async_load(mesh_path, scale_factor)

    def _apply_loaded_object_state(self, obj, obj_state: dict[str, Any]) -> None:
        if obj is None or not isinstance(obj_state, dict):
            return

        # Visibility/name
        try:
            obj.visible = bool(obj_state.get("visible", True))
        except Exception:
            pass

        # Transform
        tr = obj_state.get("transform", {})
        if not isinstance(tr, dict):
            tr = {}

        def f3(v, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
            try:
                arr = np.asarray(v, dtype=np.float64).reshape(-1)
                if arr.size >= 3 and np.isfinite(arr[:3]).all():
                    return arr[:3].astype(np.float64, copy=True)
            except Exception:
                pass
            return np.asarray(default, dtype=np.float64)

        try:
            obj.translation = f3(tr.get("translation", obj.translation))
        except Exception:
            pass
        try:
            obj.rotation = f3(tr.get("rotation_deg", obj.rotation))
        except Exception:
            pass
        try:
            obj.scale = float(tr.get("scale", getattr(obj, "scale", 1.0)) or 1.0)
        except Exception:
            pass

        try:
            obj.fixed_state_valid = bool(tr.get("fixed_state_valid", getattr(obj, "fixed_state_valid", False)))
            obj.fixed_translation = f3(tr.get("fixed_translation", getattr(obj, "fixed_translation", [0, 0, 0])))
            obj.fixed_rotation = f3(tr.get("fixed_rotation_deg", getattr(obj, "fixed_rotation", [0, 0, 0])))
            obj.fixed_scale = float(tr.get("fixed_scale", getattr(obj, "fixed_scale", 1.0)) or 1.0)
        except Exception:
            pass

        # Face selection / outer-inner assignment
        faces = obj_state.get("faces", {})
        if not isinstance(faces, dict):
            faces = {}

        try:
            n_faces_local = int(getattr(getattr(obj, "mesh", None), "n_faces", 0) or 0)
        except Exception:
            n_faces_local = 0
        n_faces_limit: int | None = n_faces_local if n_faces_local > 0 else None

        def to_int_set(v, *, max_face_count: int | None = n_faces_limit) -> set[int]:
            if not v:
                return set()
            out: set[int] = set()
            try:
                for x in v:
                    try:
                        i = int(x)
                    except Exception:
                        continue
                    if max_face_count is not None and (i < 0 or i >= max_face_count):
                        continue
                    out.add(i)
            except Exception:
                return set()
            return out

        try:
            obj.selected_faces = to_int_set(faces.get("selected", []))
        except Exception:
            pass
        try:
            obj.outer_face_indices = to_int_set(faces.get("outer", []))
            obj.inner_face_indices = to_int_set(faces.get("inner", []))
            obj.migu_face_indices = to_int_set(faces.get("migu", []))
        except Exception:
            pass
        try:
            obj.outer_face_indices.difference_update(obj.migu_face_indices)
            obj.inner_face_indices.difference_update(obj.migu_face_indices)
            overlap = obj.outer_face_indices.intersection(obj.inner_face_indices)
            if overlap:
                obj.inner_face_indices.difference_update(overlap)
        except Exception:
            pass
        try:
            unresolved = to_int_set(faces.get("assist_unresolved", []))
            unresolved.difference_update(obj.outer_face_indices)
            unresolved.difference_update(obj.inner_face_indices)
            unresolved.difference_update(obj.migu_face_indices)
            obj.surface_assist_unresolved_face_indices = unresolved
        except Exception:
            obj.surface_assist_unresolved_face_indices = set()
        try:
            raw_assist_meta = faces.get("assist_meta", {})
            obj.surface_assist_meta = dict(raw_assist_meta) if isinstance(raw_assist_meta, dict) else {}
        except Exception:
            obj.surface_assist_meta = {}
        try:
            obj.surface_assist_runtime = {}
        except Exception:
            pass

        try:
            obj._surface_overlay_index_cache = {}
            obj._surface_overlay_index_cache_version = -1
        except Exception:
            pass
        try:
            self.viewport._emit_surface_assignment_changed(obj)
        except Exception:
            try:
                obj._surface_assignment_version = int(getattr(obj, "_surface_assignment_version", 0) or 0) + 1
            except Exception:
                pass
            try:
                self.viewport.surfaceAssignmentChanged.emit(
                    len(getattr(obj, "outer_face_indices", set()) or set()),
                    len(getattr(obj, "inner_face_indices", set()) or set()),
                    len(getattr(obj, "migu_face_indices", set()) or set()),
                )
            except Exception:
                pass

        # Polyline layers
        polylines = obj_state.get("polylines", [])
        layers: list[dict[str, Any]] = []
        if isinstance(polylines, list):
            for layer in polylines:
                if not isinstance(layer, dict):
                    continue
                try:
                    pts_in = layer.get("points", []) or []
                    pts: list[list[float]] = []
                    for p in pts_in:
                        arr = np.asarray(p, dtype=np.float64).reshape(-1)
                        if arr.size >= 3 and np.isfinite(arr[:3]).all():
                            pts.append([float(arr[0]), float(arr[1]), float(arr[2])])
                        elif arr.size >= 2 and np.isfinite(arr[:2]).all():
                            pts.append([float(arr[0]), float(arr[1]), 0.0])
                    layers.append(
                        {
                            "name": str(layer.get("name", "")).strip(),
                            "kind": str(layer.get("kind", "")).strip(),
                            "visible": bool(layer.get("visible", True)),
                            "offset": [float(x) for x in (layer.get("offset", [0.0, 0.0]) or [])][:2],
                            "color": [float(x) for x in (layer.get("color", [0.1, 0.1, 0.1, 0.9]) or [])][:4],
                            "width": float(layer.get("width", 2.0) or 2.0),
                            "points": pts,
                        }
                    )
                except Exception:
                    continue
        try:
            obj.polyline_layers = layers
        except Exception:
            pass

        # Fitted arcs
        arcs = obj_state.get("arcs", [])
        fitted = []
        if isinstance(arcs, list) and arcs:
            try:
                from src.core.curvature_fitter import FittedArc

                for a in arcs:
                    if not isinstance(a, dict):
                        continue
                    try:
                        center = f3(a.get("center", [0, 0, 0]))
                        normal = f3(a.get("normal", [0, 0, 1]), default=(0.0, 0.0, 1.0))
                        plane_origin = f3(a.get("plane_origin", [0, 0, 0]))
                        plane_u = f3(a.get("plane_u", [1, 0, 0]), default=(1.0, 0.0, 0.0))
                        plane_v = f3(a.get("plane_v", [0, 1, 0]), default=(0.0, 1.0, 0.0))
                        pts2 = np.asarray(a.get("points_2d", []), dtype=np.float64).reshape(-1, 2)
                        fitted.append(
                            FittedArc(
                                center=center,
                                radius=float(a.get("radius", 0.0) or 0.0),
                                normal=normal,
                                points_2d=pts2,
                                plane_origin=plane_origin,
                                plane_u=plane_u,
                                plane_v=plane_v,
                            )
                        )
                    except Exception:
                        continue
            except Exception:
                fitted = []
        try:
            obj.fitted_arcs = fitted
        except Exception:
            pass

    def _finish_project_load(self) -> None:
        state = getattr(self, "_project_load_state", None)
        self._project_load_active = False
        self._project_load_queue = []
        self._project_load_current = None
        self._project_load_state = None

        if not isinstance(state, dict):
            state = {}

        try:
            self._apply_project_state(state)
        except Exception:
            _LOGGER.exception("Failed applying project global state")

        self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
        try:
            self.sync_transform_panel()
        except Exception:
            pass

        try:
            self.status_info.setText("✅ 프로젝트 로딩 완료")
        except Exception:
            pass

    def _apply_project_state(self, state: dict[str, Any]) -> None:
        # UI widgets (flatten/export)
        ui = state.get("ui", {})
        if isinstance(ui, dict):
            self._apply_ui_state(ui)

        vp_state = state.get("viewport", {})
        if not isinstance(vp_state, dict):
            vp_state = {}

        vp = self.viewport

        # Grid / rendering toggles
        try:
            vp.grid_spacing = float(vp_state.get("grid_spacing", vp.grid_spacing) or vp.grid_spacing)
            vp.grid_size = float(vp_state.get("grid_size", vp.grid_size) or vp.grid_size)
        except Exception:
            pass
        try:
            vp.flat_shading = bool(vp_state.get("flat_shading", getattr(vp, "flat_shading", False)))
        except Exception:
            pass
        try:
            vp.xray_mode = bool(vp_state.get("xray_mode", getattr(vp, "xray_mode", False)))
            vp.xray_alpha = float(vp_state.get("xray_alpha", getattr(vp, "xray_alpha", 0.25)) or 0.25)
        except Exception:
            pass

        # Camera
        cam_s = vp_state.get("camera", {})
        if isinstance(cam_s, dict) and getattr(vp, "camera", None) is not None:
            try:
                self._restore_camera_state_from_project(cam_s)
            except Exception:
                pass

        # Selected object (apply early so derived computations target the right mesh)
        try:
            sel = int(vp_state.get("selected_index", getattr(vp, "selected_index", -1)) or -1)
        except Exception:
            sel = -1
        if 0 <= sel < len(getattr(vp, "objects", []) or []):
            try:
                vp.select_object(sel)
            except Exception:
                vp.selected_index = sel

        # Cut lines data (edit mode restored only if explicitly enabled)
        cut_s = vp_state.get("cut_lines", {})
        if isinstance(cut_s, dict):
            try:
                vp.cut_line_active = int(cut_s.get("active", getattr(vp, "cut_line_active", 0)) or 0)
            except Exception:
                vp.cut_line_active = 0
            try:
                vp._cut_line_final = [bool(x) for x in (cut_s.get("final", [False, False]) or [False, False])][:2]
            except Exception:
                vp._cut_line_final = [False, False]

            lines = cut_s.get("lines", None)
            if isinstance(lines, list):
                out_lines = [[], []]
                for i in (0, 1):
                    pts = lines[i] if i < len(lines) else []
                    line_pts = []
                    if isinstance(pts, list):
                        for p in pts:
                            arr = np.asarray(p, dtype=np.float64).reshape(-1)
                            if arr.size >= 3 and np.isfinite(arr[:3]).all():
                                line_pts.append(arr[:3].copy())
                            elif arr.size >= 2 and np.isfinite(arr[:2]).all():
                                line_pts.append(np.array([float(arr[0]), float(arr[1]), 0.0], dtype=np.float64))
                    out_lines[i] = line_pts
                vp.cut_lines = out_lines

                # Recompute section profiles from restored cut lines.
                try:
                    for i in (0, 1):
                        if i < len(out_lines) and len(out_lines[i]) >= 2:
                            vp.schedule_cut_section_update(i, delay_ms=0)
                except Exception:
                    pass

            try:
                vp.set_cut_lines_enabled(bool(cut_s.get("enabled", False)))
            except Exception:
                vp.set_cut_lines_enabled(False)

        # Slice/Crosshair are intentionally disabled in section mode (line/ROI only).
        try:
            vp.slice_enabled = False
            vp.slice_contours = []
        except Exception:
            pass
        try:
            if getattr(vp, "picking_mode", "") == "slice":
                vp.picking_mode = "none"
        except Exception:
            pass
        try:
            self._slice_pending_height = None
            self._slice_capture_pending = False
            self._slice_debounce_timer.stop()
        except Exception:
            pass

        try:
            vp.crosshair_enabled = False
        except Exception:
            pass
        try:
            if getattr(vp, "picking_mode", "") == "crosshair":
                vp.picking_mode = "none"
        except Exception:
            pass
        try:
            self.section_panel.btn_toggle.blockSignals(True)
            self.section_panel.btn_toggle.setChecked(False)
        except Exception:
            pass
        finally:
            try:
                self.section_panel.btn_toggle.blockSignals(False)
            except Exception:
                pass

        # ROI
        roi_s = vp_state.get("roi", {})
        if isinstance(roi_s, dict):
            try:
                vp.roi_enabled = bool(roi_s.get("enabled", False))
            except Exception:
                vp.roi_enabled = False
            try:
                b = roi_s.get("bounds", None)
                if isinstance(b, (list, tuple)) and len(b) >= 4:
                    vp.roi_bounds = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
            except Exception:
                pass
            try:
                vp.roi_caps_enabled = bool(roi_s.get("caps", False))
            except Exception:
                pass
            try:
                if vp.roi_enabled:
                    vp.schedule_roi_edges_update(0)
            except Exception:
                pass

            try:
                self.section_panel.btn_roi.blockSignals(True)
                self.section_panel.btn_roi.setChecked(bool(getattr(vp, "roi_enabled", False)))
                self.section_panel.btn_roi.setText(
                    "📐 영역 지정 모드 중지" if bool(getattr(vp, "roi_enabled", False)) else "📐 영역 지정 모드 시작"
                )
                self.section_panel.btn_silhouette.setEnabled(bool(getattr(vp, "roi_enabled", False)))
            except Exception:
                pass
            finally:
                try:
                    self.section_panel.btn_roi.blockSignals(False)
                except Exception:
                    pass

        # Cutline edit mode button
        try:
            self._sync_cutline_button_state(bool(getattr(vp, "cut_lines_enabled", False)))
            try:
                self.section_panel.combo_cutline.blockSignals(True)
                self.section_panel.combo_cutline.setCurrentIndex(int(getattr(vp, "cut_line_active", 0) or 0))
            finally:
                try:
                    self.section_panel.combo_cutline.blockSignals(False)
                except Exception:
                    pass
        except Exception:
            pass

        # Normalize mutually-exclusive section input modes restored from project files.
        try:
            self._normalize_section_modes_after_restore()
        except Exception:
            _LOGGER.exception("Failed normalizing section modes after restore")

        # Final UI sync after normalization so button state always matches actual mode.
        try:
            self._sync_section_mode_buttons()
        except Exception:
            pass

        vp.update()

    def _restore_camera_state_from_project(self, cam_s: dict[str, Any]) -> None:
        vp = self.viewport
        cam = vp.camera

        def _vec3(value: object, fallback: np.ndarray) -> np.ndarray:
            try:
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
                if arr.size >= 3 and np.isfinite(arr[:3]).all():
                    return arr[:3].copy()
            except Exception:
                pass
            return np.asarray(fallback, dtype=np.float64).reshape(3)

        try:
            dist_raw = float(cam_s.get("distance", cam.distance) or cam.distance)
        except Exception:
            dist_raw = float(getattr(cam, "distance", 50.0) or 50.0)
        try:
            az_raw = float(cam_s.get("azimuth", cam.azimuth) or cam.azimuth)
        except Exception:
            az_raw = float(getattr(cam, "azimuth", 45.0) or 45.0)
        try:
            el_raw = float(cam_s.get("elevation", cam.elevation) or cam.elevation)
        except Exception:
            el_raw = float(getattr(cam, "elevation", 30.0) or 30.0)

        min_d = float(getattr(cam, "min_distance", 0.01) or 0.01)
        max_d = float(getattr(cam, "max_distance", 1_000_000.0) or 1_000_000.0)
        if not np.isfinite(dist_raw):
            dist_raw = float(getattr(cam, "distance", 50.0) or 50.0)
        dist = float(max(min_d, min(max_d, dist_raw)))

        if not np.isfinite(az_raw):
            az_raw = float(getattr(cam, "azimuth", 45.0) or 45.0)
        az = ((float(az_raw) + 180.0) % 360.0) - 180.0

        if not np.isfinite(el_raw):
            el_raw = float(getattr(cam, "elevation", 30.0) or 30.0)
        el = float(el_raw)
        el = float(max(-90.0, min(90.0, el)))

        cam.distance = dist
        cam.azimuth = az
        cam.elevation = el
        cam.center = _vec3(cam_s.get("center", cam.center), np.asarray(getattr(cam, "center", [0.0, 0.0, 0.0]), dtype=np.float64))
        cam.pan_offset = _vec3(
            cam_s.get("pan_offset", cam.pan_offset),
            np.asarray(getattr(cam, "pan_offset", [0.0, 0.0, 0.0]), dtype=np.float64),
        )

        # Restore should not force camera back into orthographic lock.
        vp._front_back_ortho_enabled = False

    def _normalize_section_modes_after_restore(self) -> None:
        vp = self.viewport

        cut_enabled = bool(getattr(vp, "cut_lines_enabled", False))
        roi_enabled = bool(getattr(vp, "roi_enabled", False))
        cross_enabled = bool(getattr(vp, "crosshair_enabled", False))

        # Priority: cut-lines > ROI > crosshair (matches active input intent).
        if cut_enabled:
            vp.crosshair_enabled = False
            vp.roi_enabled = False
            vp.active_roi_edge = None
            vp.set_cut_lines_enabled(True)
            return

        vp.set_cut_lines_enabled(False)
        vp.active_roi_edge = None

        if roi_enabled:
            vp.crosshair_enabled = False
            vp.roi_enabled = True
            if str(getattr(vp, "picking_mode", "")).strip().lower() in {"crosshair", "cut_lines"}:
                vp.picking_mode = "none"
            try:
                vp.schedule_roi_edges_update(0)
            except Exception:
                pass
            return

        if cross_enabled:
            vp.roi_enabled = False
            vp.crosshair_enabled = True
            vp.picking_mode = "crosshair"
            try:
                vp.schedule_crosshair_profile_update(0)
            except Exception:
                pass
            return

        vp.crosshair_enabled = False
        vp.roi_enabled = False
        if str(getattr(vp, "picking_mode", "")).strip().lower() in {"crosshair", "cut_lines"}:
            vp.picking_mode = "none"

    def _sync_section_mode_buttons(self) -> None:
        vp = self.viewport
        cross_enabled = bool(getattr(vp, "crosshair_enabled", False))
        roi_enabled = bool(getattr(vp, "roi_enabled", False))
        cut_enabled = bool(getattr(vp, "cut_lines_enabled", False))

        try:
            self.section_panel.btn_toggle.blockSignals(True)
            self.section_panel.btn_toggle.setChecked(cross_enabled)
            self.section_panel.btn_toggle.setText(
                "🎯 십자선 단면 모드 중지" if cross_enabled else "🎯 십자선 단면 모드 시작"
            )
        except Exception:
            pass
        finally:
            try:
                self.section_panel.btn_toggle.blockSignals(False)
            except Exception:
                pass

        try:
            self.section_panel.btn_roi.blockSignals(True)
            self.section_panel.btn_roi.setChecked(roi_enabled)
            self.section_panel.btn_roi.setText(
                "🗺 영역 지정 모드 중지" if roi_enabled else "🗺 영역 지정 모드 시작"
            )
            self.section_panel.btn_silhouette.setEnabled(roi_enabled)
        except Exception:
            pass
        finally:
            try:
                self.section_panel.btn_roi.blockSignals(False)
            except Exception:
                pass

        self._sync_cutline_button_state(cut_enabled)

    def _apply_ui_state(self, ui: dict[str, Any]) -> None:
        # Flatten panel
        flat = ui.get("flatten", {})
        if isinstance(flat, dict) and getattr(self, "flatten_panel", None) is not None:
            try:
                self.flatten_panel.spin_radius.setValue(float(flat.get("radius_mm", self.flatten_panel.spin_radius.value()) or 150.0))
                self.flatten_panel.combo_direction.setCurrentIndex(int(flat.get("direction_index", self.flatten_panel.combo_direction.currentIndex()) or 0))
                self.flatten_panel.combo_method.setCurrentIndex(int(flat.get("method_index", self.flatten_panel.combo_method.currentIndex()) or 0))
                self.flatten_panel.slider_distortion.setValue(int(flat.get("distortion_percent", self.flatten_panel.slider_distortion.value()) or 50))
                self.flatten_panel.check_auto_cut.setChecked(bool(flat.get("auto_cut", self.flatten_panel.check_auto_cut.isChecked())))
                self.flatten_panel.check_multiband.setChecked(bool(flat.get("multiband", self.flatten_panel.check_multiband.isChecked())))
                self.flatten_panel.spin_iterations.setValue(int(flat.get("iterations", self.flatten_panel.spin_iterations.value()) or 30))
            except Exception:
                pass

        # Export panel
        exp = ui.get("export", {})
        if isinstance(exp, dict) and getattr(self, "export_panel", None) is not None:
            try:
                self.export_panel.spin_dpi.setValue(int(exp.get("dpi", self.export_panel.spin_dpi.value()) or 300))
                self.export_panel.combo_format.setCurrentIndex(int(exp.get("format_index", self.export_panel.combo_format.currentIndex()) or 0))
                self.export_panel.check_scale_bar.setChecked(bool(exp.get("scale_bar", self.export_panel.check_scale_bar.isChecked())))
                self.export_panel.check_profile_include_grid.setChecked(
                    bool(exp.get("profile_include_grid", self.export_panel.check_profile_include_grid.isChecked()))
                )
                self.export_panel.check_profile_feature_lines.setChecked(
                    bool(exp.get("profile_feature_lines", self.export_panel.check_profile_feature_lines.isChecked()))
                )
                self.export_panel.spin_profile_feature_angle.setValue(
                    float(exp.get("profile_feature_angle", self.export_panel.spin_profile_feature_angle.value()) or 60.0)
                )
            except Exception:
                pass

        # Slice presets
        sl = ui.get("slice", {})
        if isinstance(sl, dict) and getattr(self, "slice_panel", None) is not None:
            try:
                self.slice_panel.set_presets(sl.get("presets", []))
            except Exception:
                pass
    
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
        try:
            self._status_task_begin(f"메쉬 로딩: {name}", maximum=None, value=None)
        except Exception:
            pass

        self._mesh_load_thread = MeshLoadThread(
            filepath=str(filepath),
            scale_factor=float(scale_factor),
            default_unit=str(getattr(self.mesh_loader, "default_unit", DEFAULT_MESH_UNIT)),
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

            # Normal file load vs project load(.amr)
            obj_name = Path(filepath).name
            project_obj_state = getattr(self, "_project_load_current", None) if getattr(self, "_project_load_active", False) else None
            if isinstance(project_obj_state, dict):
                obj_name = str(project_obj_state.get("name", "")).strip() or obj_name

            self.viewport.add_mesh_object(mesh_data, name=obj_name)
            try:
                obj_loaded = self.viewport.selected_obj
                if obj_loaded is not None and int(getattr(obj_loaded, "vertex_count", 0) or 0) <= 0:
                    # Defensive: if VBO was not prepared, rebuild once.
                    self.viewport.update_vbo(obj_loaded)
            except Exception:
                pass

            if isinstance(project_obj_state, dict):
                try:
                    self._apply_loaded_object_state(self.viewport.selected_obj, project_obj_state)
                except Exception:
                    _LOGGER.exception("Failed applying object state from project")
                self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
                self.status_info.setText(f"로드됨(프로젝트): {obj_name}")
            else:
                # 일반 메쉬 로드 시에는 X-Ray를 기본 해제해 내부 비침 혼란을 줄입니다.
                try:
                    self.viewport.xray_mode = False
                    if getattr(self, "trans_toolbar", None) is not None:
                        self.trans_toolbar.btn_xray.blockSignals(True)
                        self.trans_toolbar.btn_xray.setChecked(False)
                        self.trans_toolbar.btn_xray.blockSignals(False)
                except Exception:
                    pass
                try:
                    # Keep newly loaded meshes immediately visible.
                    self.fit_view()
                except Exception:
                    pass
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

        # Abort project load if a mesh fails to load.
        if bool(getattr(self, "_project_load_active", False)):
            self._project_load_active = False
            self._project_load_queue = []
            self._project_load_current = None
            self._project_load_state = None

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
        try:
            self._status_task_end()
        except Exception:
            pass

        # Continue queued project loads after each mesh finishes loading.
        if bool(getattr(self, "_project_load_active", False)):
            try:
                if getattr(self, "_project_load_queue", None):
                    self._start_next_project_object_load()
                else:
                    self._finish_project_load()
            except Exception:
                _LOGGER.exception("Project load continuation failed")

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
        try:
            self._status_task_end()
        except Exception:
            pass

    def _format_error_message(self, prefix: str, message: str) -> str:
        try:
            from src.core.logging_utils import format_exception_message

            return format_exception_message(prefix, message, log_path=_log_path)
        except Exception:
            return f"{prefix}\n\n{message}"

    def _status_task_begin(self, text: str, *, maximum: int | None = None, value: int | None = None) -> None:
        try:
            self._status_task_count = int(getattr(self, "_status_task_count", 0) or 0) + 1
        except Exception:
            self._status_task_count = 1

        widget = getattr(self, "_status_task_widget", None)
        label = getattr(self, "_status_task_label", None)
        bar = getattr(self, "_status_task_bar", None)
        if widget is None or label is None or bar is None:
            return

        try:
            label.setText(str(text or "").strip())
        except Exception:
            pass

        try:
            if maximum is None:
                bar.setRange(0, 0)  # indeterminate
            else:
                m = int(maximum)
                m = max(1, m)
                bar.setRange(0, m)
                bar.setValue(int(value or 0))
        except Exception:
            pass

        try:
            widget.setVisible(True)
        except Exception:
            pass

    def _status_task_update(self, *, text: str | None = None, maximum: int | None = None, value: int | None = None) -> None:
        widget = getattr(self, "_status_task_widget", None)
        label = getattr(self, "_status_task_label", None)
        bar = getattr(self, "_status_task_bar", None)
        if widget is None or label is None or bar is None:
            return

        try:
            if text is not None:
                label.setText(str(text or "").strip())
        except Exception:
            pass

        try:
            if maximum is not None:
                m = int(maximum)
                m = max(1, m)
                bar.setRange(0, m)
            if value is not None:
                bar.setValue(int(value))
        except Exception:
            pass

        try:
            if not widget.isVisible():
                widget.setVisible(True)
        except Exception:
            pass

    def _status_task_end(self) -> None:
        try:
            c = int(getattr(self, "_status_task_count", 0) or 0)
        except Exception:
            c = 0
        c = max(0, c - 1)
        self._status_task_count = c

        if c > 0:
            return

        widget = getattr(self, "_status_task_widget", None)
        label = getattr(self, "_status_task_label", None)
        bar = getattr(self, "_status_task_bar", None)
        try:
            if label is not None:
                label.setText("")
        except Exception:
            pass
        try:
            if bar is not None:
                bar.setRange(0, 0)
        except Exception:
            pass
        try:
            if widget is not None:
                widget.setVisible(False)
        except Exception:
            pass

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

        try:
            self._status_task_begin(str(label), maximum=None, value=None)
        except Exception:
            pass

        self._task_dialog = dlg
        self._task_thread = thread

        progress_ended = False

        def _end_progress():
            nonlocal progress_ended
            if progress_ended:
                return
            progress_ended = True
            try:
                self._status_task_end()
            except Exception:
                pass

        def _close_dialog():
            d = getattr(self, "_task_dialog", None)
            if d is not None:
                try:
                    d.close()
                except Exception:
                    pass
                self._task_dialog = None
            _end_progress()

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

        try:
            self.viewport.clear_measure_picks()
            panel = getattr(self, "measure_panel", None)
            if panel is not None:
                panel.set_points_count(0)
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
        panel = getattr(self, "slice_panel", None)
        if obj and obj.mesh and panel is not None:
            # 대용량 메쉬에서 전체 버텍스 스캔은 느림 -> 월드 바운드로 근사
            try:
                wb = obj.get_world_bounds()
                z_min = float(wb[0][2])
                z_max = float(wb[1][2])
            except Exception:
                z_min = float(obj.mesh.bounds[0][2])
                z_max = float(obj.mesh.bounds[1][2])
            panel.update_range(z_min, z_max)
            
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

    def toggle_xray_mode(self, enabled):
        """X-Ray 모드 토글 (선택된 메쉬만 투명 표시)"""
        try:
            self.viewport.xray_mode = bool(enabled)
        except Exception:
            return
        self.viewport.update()
        try:
            self.status_info.setText("🩻 X-Ray 모드: 선택된 메쉬를 투명 표시" if enabled else "🩻 X-Ray 모드 종료")
        except Exception:
            pass

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

    def fit_ground_plane(self):
        """현재 자세를 유지하고 메쉬를 XY 바닥(Z=0)에 안착."""
        obj = self.viewport.selected_obj
        if not obj:
            return

        try:
            self.viewport.save_undo_state()
        except Exception:
            pass

        # 월드 기준 안착을 위해 현재 T/R/S를 먼저 bake.
        self.viewport.bake_object_transform(obj)

        try:
            z_vals = np.asarray(obj.mesh.vertices[:, 2], dtype=np.float64)
            z_vals = z_vals[np.isfinite(z_vals)]
            if z_vals.size == 0:
                return
            min_z = float(np.min(z_vals))
        except Exception:
            return

        if not np.isfinite(min_z):
            return

        if abs(min_z) > 1e-9:
            obj.mesh.vertices[:, 2] -= min_z
            try:
                obj.mesh._bounds = None
                obj.mesh._centroid = None
                obj.mesh._surface_area = None
            except Exception:
                pass
            obj._trimesh = None

        obj.translation = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.viewport.update_vbo(obj)
        self.sync_transform_panel()
        self.viewport.status_info = "✅ 기준평면 맞추기 완료 (최저점 Z=0)"
        self.viewport.update()
        self.viewport.meshTransformChanged.emit()

    def _infer_migu_from_outer_inner(
        self,
        *,
        obj,
        mesh_local,
        outer_ids: set[int] | list[int] | np.ndarray,
        inner_ids: set[int] | list[int] | np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """
        현재 outer/inner 라벨 경계로부터 미구(두께/측벽) face를 추론합니다.

        Returns:
            (indices, description)
        """
        try:
            from src.core.surface_separator import SurfaceSeparator

            separator = SurfaceSeparator()
            hops = int(getattr(self, "_migu_boundary_hops", 1) or 1)
            dom_ratio = float(getattr(self, "_migu_vertex_dom_ratio", 1.20) or 1.20)
            side_thr = float(getattr(self, "_migu_side_absdot_max", 0.45) or 0.45)
            max_ratio = float(getattr(self, "_migu_boundary_max_ratio", 0.35) or 0.35)

            idx, meta = separator.infer_migu_from_outer_inner(
                mesh_local,
                outer_face_indices=outer_ids,
                inner_face_indices=inner_ids,
                hops=hops,
                vertex_dom_ratio=dom_ratio,
                side_absdot_max=side_thr,
                max_ratio=max_ratio,
            )
            mode = str((meta or {}).get("mode", "")).strip()
            mode_tag = f",{mode}" if mode else ""
            desc = f"경계-보조(hops={max(0, min(hops, 3))}{mode_tag})"
            return np.asarray(idx, dtype=np.int32).reshape(-1), desc
        except Exception:
            return np.zeros((0,), dtype=np.int32), "경계-보조"

    def _apply_surface_stability_presets(self, mesh_local) -> str | None:
        """
        대형 메쉬(수백만 face)에서 내/외면 분리 안정성을 높이기 위한 기본 프리셋을 적용합니다.
        사용자/고급 설정이 이미 존재하면 덮어쓰지 않습니다.
        """
        try:
            n_faces = int(getattr(mesh_local, "n_faces", 0) or 0)
        except Exception:
            n_faces = 0
        if n_faces < 1_000_000:
            return None

        applied: list[str] = []
        try:
            if getattr(mesh_local, "_views_fallback_use_normals", None) is None:
                mesh_local._views_fallback_use_normals = False
                applied.append("fallback_t_only")
        except Exception:
            pass
        try:
            if getattr(mesh_local, "_views_migu_absdot_max", None) is None:
                # Disable normal-only migu carving for very large meshes; use boundary-based supplement instead.
                mesh_local._views_migu_absdot_max = 1.0
                applied.append("migu_disable_normals")
        except Exception:
            pass
        try:
            if getattr(mesh_local, "_views_migu_max_frac", None) is None:
                mesh_local._views_migu_max_frac = 0.05
                applied.append("migu_frac_guard")
        except Exception:
            pass
        try:
            if getattr(mesh_local, "_views_visibility_neighborhood", None) is None:
                # Reduce view-bin jitter on very large meshes.
                mesh_local._views_visibility_neighborhood = 2
                applied.append("vis_nbhd2")
        except Exception:
            pass

        if applied:
            return "large-mesh-stable"
        return "large-mesh-stable(user-set)"
    
    def on_selection_action(self, action: str, data):
        action = str(action or "").strip()

        # 1) Surface target / tool switch (no mesh required)
        if action == "surface_target":
            target = str(data or "").strip().lower()
            if target not in {"outer", "inner", "migu"}:
                target = "outer"
            self.viewport._surface_paint_target = target
            self.viewport.status_info = f"✋ 표면 지정 대상: {target} (경계(면적+자석)로 시작)"
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

            try:
                self._disable_measure_mode()
            except Exception:
                pass

            # Tool unification: click/brush removed, area+magnetic merged into one boundary tool.
            tool = {
                "click": "boundary",
                "brush": "boundary",
                "area": "boundary",
                "magnetic": "boundary",
            }.get(tool, tool)

            if tool == "boundary":
                self.viewport.picking_mode = "paint_surface_magnetic"
                try:
                    self.viewport.clear_surface_lasso()
                except Exception:
                    pass
                try:
                    self.viewport.start_surface_magnetic_lasso()
                    self.viewport.setMouseTracking(True)
                    self.viewport.setFocus()
                except Exception:
                    pass
                self.viewport.status_info = (
                    f"🧲 경계(면적+자석) [{target}]: 좌클릭=점 추가(자석 스냅), 드래그=회전/시점, "
                    f"우클릭/Enter=확정, Backspace=되돌리기, Shift/Ctrl=완드 정제, Alt=제거, [ / ]=반경, "
                    f"실시간 단면은 '단면/2D 지정 도구' 탭에서 ON 후 Ctrl+휠/[, .]/C 사용 (ESC=종료)"
                )
            else:
                QMessageBox.information(self, "안내", "선택 도구를 확인할 수 없습니다.")
                return

            self.viewport.update()
            return

        if action == "open_section_tools":
            try:
                self.section_dock.show()
                self.section_dock.raise_()
            except Exception:
                pass
            try:
                self.status_info.setText(
                    "단면/2D 지정 도구로 이동: 실시간 단면(3D)과 2D 단면선/ROI를 여기서 함께 제어합니다."
                )
            except Exception:
                pass
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
        if not hasattr(obj, "surface_assist_unresolved_face_indices") or obj.surface_assist_unresolved_face_indices is None:
            obj.surface_assist_unresolved_face_indices = set()
        if not hasattr(obj, "surface_assist_meta") or obj.surface_assist_meta is None:
            obj.surface_assist_meta = {}
        if not hasattr(obj, "surface_assist_runtime") or obj.surface_assist_runtime is None:
            obj.surface_assist_runtime = {}

        if action == "surface_slice_toggle":
            panel = getattr(self, "slice_panel", None)
            if panel is None:
                QMessageBox.warning(self, "경고", "단면 패널을 찾을 수 없습니다.")
                return
            try:
                self.update_slice_range()
            except Exception:
                pass
            current_enabled = bool(getattr(self.viewport, "slice_enabled", False))
            requested = data if isinstance(data, bool) else None
            enabled = (not current_enabled) if requested is None else bool(requested)
            try:
                lo = float(panel.spin.minimum())
                hi = float(panel.spin.maximum())
                z_cur = float(getattr(self.viewport, "slice_z", 0.0) or 0.0)
                z_next = float(np.clip(z_cur, lo, hi))
            except Exception:
                z_next = float(getattr(self.viewport, "slice_z", 0.0) or 0.0)
            try:
                panel.spin.setValue(z_next)
            except Exception:
                pass
            try:
                panel.group.setChecked(bool(enabled))
            except Exception:
                pass
            if enabled:
                self.viewport.status_info = (
                    f"🧭 실시간 단면 모드 ON (Z={z_next:.2f}cm): "
                    "Ctrl+휠/[, .]=스캔, C=촬영"
                )
            else:
                self.viewport.status_info = "🧭 실시간 단면 모드 OFF"
            try:
                self.viewport.setFocus()
            except Exception:
                pass
            self.viewport.update()
            return

        if action == "surface_slice_capture":
            try:
                z_now = float(getattr(self.viewport, "slice_z", 0.0) or 0.0)
            except Exception:
                z_now = 0.0
            self.on_slice_capture_requested(z_now)
            try:
                self.viewport.setFocus()
            except Exception:
                pass
            return

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
                self.viewport.clear_surface_magnetic_lasso(clear_cache=False)
            except Exception:
                pass
            self.viewport.status_info = f"표면 지정 비움: {target}"
            try:
                obj.surface_assist_unresolved_face_indices = set()
                obj.surface_assist_meta = {}
                obj.surface_assist_runtime = {}
            except Exception:
                pass
            try:
                self.viewport._emit_surface_assignment_changed(obj)
            except Exception:
                pass

        elif action == "surface_clear_all":
            obj.outer_face_indices.clear()
            obj.inner_face_indices.clear()
            obj.migu_face_indices.clear()
            try:
                obj.surface_assist_unresolved_face_indices = set()
                obj.surface_assist_meta = {}
                obj.surface_assist_runtime = {}
            except Exception:
                pass
            try:
                self.viewport.clear_surface_paint_points(None)
                self.viewport.clear_surface_lasso()
                self.viewport.clear_surface_magnetic_lasso(clear_cache=False)
            except Exception:
                pass
            self.viewport.status_info = "표면 지정 전체 초기화"
            try:
                self.viewport._emit_surface_assignment_changed(obj)
            except Exception:
                pass

        elif action == "assist_surface":
            try:
                from src.core.surface_separator import SurfaceSeparator

                mesh_local = getattr(obj, "mesh", None)
                if mesh_local is None:
                    QMessageBox.warning(self, "경고", "먼저 메쉬를 선택해 주세요.")
                    return

                try:
                    n_faces = int(getattr(mesh_local, "n_faces", 0) or 0)
                except Exception:
                    n_faces = 0
                min_seed = int(max(24, min(300, int(0.00005 * max(1, n_faces)))))

                modifiers = QApplication.keyboardModifiers()
                conservative = not bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
                force_cyl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
                force_auto = bool(modifiers & Qt.KeyboardModifier.AltModifier)
                if force_cyl:
                    method = "cylinder"
                elif force_auto:
                    method = "auto"
                else:
                    method = "views"

                old_outer = set(int(x) for x in (getattr(obj, "outer_face_indices", set()) or set()))
                old_inner = set(int(x) for x in (getattr(obj, "inner_face_indices", set()) or set()))
                old_migu = set(int(x) for x in (getattr(obj, "migu_face_indices", set()) or set()))
                assist_total_t0 = time.perf_counter()

                try:
                    self._apply_surface_stability_presets(mesh_local)
                except Exception:
                    pass

                separator = SurfaceSeparator()
                assist_core_t0 = time.perf_counter()
                outer_idx, inner_idx, meta = separator.assist_outer_inner_from_seeds(
                    mesh_local,
                    outer_face_indices=old_outer,
                    inner_face_indices=old_inner,
                    migu_face_indices=old_migu,
                    method=method,
                    conservative=bool(conservative),
                    min_seed=min_seed,
                )
                assist_core_ms = (time.perf_counter() - assist_core_t0) * 1000.0

                status = str((meta or {}).get("status", "")).strip().lower()
                if status == "missing_seeds":
                    so = int((meta or {}).get("seed_outer_count", len(old_outer)) or 0)
                    si = int((meta or {}).get("seed_inner_count", len(old_inner)) or 0)
                    req = int((meta or {}).get("min_seed_required", min_seed) or min_seed)
                    QMessageBox.information(
                        self,
                        "씨드 부족",
                        "수동 보조 분리를 위해 outer/inner 씨드가 더 필요합니다.\n\n"
                        f"- 현재 outer seed: {so:,}\n"
                        f"- 현재 inner seed: {si:,}\n"
                        f"- 권장 최소 seed: {req:,}\n\n"
                        "경계(면적+자석)로 양쪽에 조금씩 먼저 지정한 뒤 다시 실행하세요.",
                    )
                    return
                if status and status != "ok":
                    err = str((meta or {}).get("error", "")).strip()
                    msg = (
                        "수동 보조 분리 중 자동 분류를 완료하지 못했습니다.\n\n"
                        f"- 상태: {status}\n"
                    )
                    if err:
                        msg += f"- 상세: {err}\n"
                    msg += "\n씨드를 더 지정하거나 보조 방식(Shift/Ctrl/Alt)을 바꿔 다시 시도하세요."
                    QMessageBox.warning(self, "수동 보조 분리 실패", msg)
                    return

                assist_apply_t0 = time.perf_counter()
                new_outer = set(map(int, np.asarray(outer_idx, dtype=np.int32).reshape(-1)))
                new_inner = set(map(int, np.asarray(inner_idx, dtype=np.int32).reshape(-1)))
                # Keep migu exclusive.
                new_outer.difference_update(old_migu)
                new_inner.difference_update(old_migu)
                overlap = new_outer.intersection(new_inner)
                if overlap:
                    new_inner.difference_update(overlap)

                obj.outer_face_indices = new_outer
                obj.inner_face_indices = new_inner
                unresolved_truncated = bool((meta or {}).get("unresolved_truncated", False))
                try:
                    unresolved_raw = (meta or {}).get("unresolved_indices", None)
                    if unresolved_raw is None:
                        unresolved_idx = np.zeros((0,), dtype=np.int32)
                    else:
                        unresolved_idx = np.asarray(unresolved_raw, dtype=np.int32).reshape(-1)
                except Exception:
                    unresolved_idx = np.zeros((0,), dtype=np.int32)
                if unresolved_idx.size > 0:
                    unresolved_set = set(int(x) for x in unresolved_idx.tolist())
                else:
                    unresolved_set = set()
                if unresolved_set:
                    unresolved_set.difference_update(new_outer)
                    unresolved_set.difference_update(new_inner)
                    unresolved_set.difference_update(old_migu)
                obj.surface_assist_unresolved_face_indices = unresolved_set
                obj.surface_assist_meta = dict(meta or {})
                assist_apply_ms = (time.perf_counter() - assist_apply_t0) * 1000.0
                assist_total_ms = (time.perf_counter() - assist_total_t0) * 1000.0

                add_o = len(new_outer.difference(old_outer))
                add_i = len(new_inner.difference(old_inner))
                unresolved = int((meta or {}).get("unresolved_count", 0) or 0)
                mode = str((meta or {}).get("assist_mode", "seeded")).strip()
                mapping = str((meta or {}).get("auto_mapping", "direct")).strip()
                mode_txt = "보수" if conservative else "공격"
                unresolved_suffix = (
                    " (표시 일부 생략)"
                    if unresolved > 0 and unresolved_truncated and len(unresolved_set) <= 0
                    else ""
                )
                try:
                    obj.surface_assist_runtime = {
                        "total_ms": float(assist_total_ms),
                        "core_ms": float(assist_core_ms),
                        "apply_ms": float(assist_apply_ms),
                        "method": str(method),
                        "mode_txt": str(mode_txt),
                        "assist_mode": str(mode),
                        "mapping": str(mapping),
                        "added_outer_count": int(add_o),
                        "added_inner_count": int(add_i),
                        "unresolved_count": int(unresolved),
                        "unresolved_drawn_count": int(len(unresolved_set)),
                    }
                except Exception:
                    pass

                self.viewport.status_info = (
                    f"🤝 수동 보조 분리({mode_txt}/{method}, {mode}, {mapping}): "
                    f"outer +{add_o:,}, inner +{add_i:,}, 미확정 {unresolved:,}{unresolved_suffix} "
                    f"({assist_total_ms:.1f}ms)"
                )
                try:
                    self.viewport._emit_surface_assignment_changed(obj)
                except Exception:
                    pass
            except Exception as e:
                QMessageBox.critical(self, "오류", f"수동 보조 분리 실패:\n{e}")
                return

        elif action == "auto_surface":
            try:
                from src.core.surface_separator import SurfaceSeparator

                separator = SurfaceSeparator()
                mesh_local = getattr(obj, "mesh", None)
                if mesh_local is None:
                    QMessageBox.warning(self, "경고", "먼저 메쉬를 선택해 주세요.")
                    return
                preset_desc = None
                try:
                    preset_desc = self._apply_surface_stability_presets(mesh_local)
                except Exception:
                    preset_desc = None
                modifiers = QApplication.keyboardModifiers()
                force_views = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
                force_cyl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
                if force_cyl:
                    method = "cylinder"
                elif force_views:
                    method = "views"
                else:
                    method = "auto"

                result = separator.auto_detect_surfaces(mesh_local, method=method, return_submeshes=False)
                obj.outer_face_indices = set(map(int, getattr(result, "outer_face_indices", np.zeros((0,), dtype=np.int32))))
                obj.inner_face_indices = set(map(int, getattr(result, "inner_face_indices", np.zeros((0,), dtype=np.int32))))
                try:
                    obj.surface_assist_unresolved_face_indices = set()
                    obj.surface_assist_meta = {}
                    obj.surface_assist_runtime = {}
                except Exception:
                    pass

                migu_idx = getattr(result, "migu_face_indices", None)
                if isinstance(migu_idx, np.ndarray) and migu_idx.size:
                    obj.migu_face_indices = set(map(int, migu_idx))
                else:
                    obj.migu_face_indices.clear()

                # Keep sets exclusive (migu wins).
                try:
                    obj.outer_face_indices.difference_update(obj.migu_face_indices)
                    obj.inner_face_indices.difference_update(obj.migu_face_indices)
                except Exception:
                    pass

                # Safety: eliminate any overlap between outer/inner.
                try:
                    overlap = obj.outer_face_indices.intersection(obj.inner_face_indices)
                    if overlap:
                        obj.outer_face_indices.difference_update(overlap)
                        obj.inner_face_indices.difference_update(overlap)
                        obj.migu_face_indices.update(overlap)
                except Exception:
                    pass

                # Supplemental migu inference from current outer/inner boundary
                # (so users can get usable inner/migu split in one click).
                supplemental_desc = None
                try:
                    n_faces = int(getattr(mesh_local, "n_faces", 0) or 0)
                    min_migu = max(8, int(0.003 * max(1, n_faces)))
                    if len(obj.migu_face_indices) < min_migu:
                        sup_idx, sup_desc = self._infer_migu_from_outer_inner(
                            obj=obj,
                            mesh_local=mesh_local,
                            outer_ids=obj.outer_face_indices,
                            inner_ids=obj.inner_face_indices,
                        )
                        if isinstance(sup_idx, np.ndarray) and sup_idx.size > 0:
                            obj.migu_face_indices.update(int(x) for x in sup_idx)
                            obj.outer_face_indices.difference_update(obj.migu_face_indices)
                            obj.inner_face_indices.difference_update(obj.migu_face_indices)
                            supplemental_desc = str(sup_desc or "경계-보조")
                except Exception:
                    supplemental_desc = None

                meta = getattr(result, "meta", {}) or {}
                method_used = str(meta.get("method", method))
                if preset_desc:
                    method_used = f"{method_used} + {preset_desc}"
                if supplemental_desc:
                    method_used = f"{method_used} + {supplemental_desc}"

                self.viewport.status_info = (
                    f"✅ 자동 분리 적용({method_used}): outer {len(obj.outer_face_indices):,} / inner {len(obj.inner_face_indices):,} / migu {len(obj.migu_face_indices):,} (현재 메쉬에 저장됨)"
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
                    f"- inner(내면): {len(obj.inner_face_indices):,} faces\n"
                    f"- migu(미구): {len(obj.migu_face_indices):,} faces\n\n"
                    f"- method: {method_used}\n\n"
                    f"표시: 외면=파랑, 내면=주황 오버레이\n"
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
                use_x = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
                allow_bootstrap = bool(modifiers & Qt.KeyboardModifier.AltModifier)

                idx = None
                mode_desc = None
                bootstrap_used = False

                # Optional: if outer/inner labels are weak or missing, bootstrap once first.
                # Keep this opt-in (Alt) instead of implicit auto behavior.
                if (not broad_edge) and (not use_x) and allow_bootstrap:
                    try:
                        n_faces = int(getattr(mesh_local, "n_faces", 0) or 0)
                        min_seed = max(12, int(0.005 * max(1, n_faces)))
                        cur_outer = set(int(x) for x in (getattr(obj, "outer_face_indices", set()) or set()))
                        cur_inner = set(int(x) for x in (getattr(obj, "inner_face_indices", set()) or set()))
                        if len(cur_outer) < min_seed or len(cur_inner) < min_seed:
                            try:
                                self._apply_surface_stability_presets(mesh_local)
                            except Exception:
                                pass
                            separator = SurfaceSeparator()
                            boot = separator.auto_detect_surfaces(mesh_local, method="auto", return_submeshes=False)
                            boot_outer = set(map(int, getattr(boot, "outer_face_indices", np.zeros((0,), dtype=np.int32))))
                            boot_inner = set(map(int, getattr(boot, "inner_face_indices", np.zeros((0,), dtype=np.int32))))
                            if boot_outer and boot_inner:
                                try:
                                    boot_outer.difference_update(getattr(obj, "migu_face_indices", set()) or set())
                                    boot_inner.difference_update(getattr(obj, "migu_face_indices", set()) or set())
                                except Exception:
                                    pass
                                overlap = boot_outer.intersection(boot_inner)
                                if overlap:
                                    boot_outer.difference_update(overlap)
                                    boot_inner.difference_update(overlap)
                                if boot_outer and boot_inner:
                                    obj.outer_face_indices = boot_outer
                                    obj.inner_face_indices = boot_inner
                                    bootstrap_used = True
                    except Exception:
                        bootstrap_used = False

                # Preferred path: if outer/inner already exist, infer migu directly from their boundary.
                if (not broad_edge) and (not use_x):
                    try:
                        idx_b, desc_b = self._infer_migu_from_outer_inner(
                            obj=obj,
                            mesh_local=mesh_local,
                            outer_ids=getattr(obj, "outer_face_indices", set()) or set(),
                            inner_ids=getattr(obj, "inner_face_indices", set()) or set(),
                        )
                        if isinstance(idx_b, np.ndarray) and idx_b.size > 0:
                            idx = idx_b.astype(np.int32, copy=False)
                            mode_desc = str(desc_b or "경계-보조")
                    except Exception:
                        idx = None
                        mode_desc = None

                # Fast path for tiles: reuse the cylinder separator's migu band when it looks valid.
                if idx is None and (not broad_edge) and (not use_x):
                    try:
                        separator = SurfaceSeparator()
                        cyl = separator.auto_detect_surfaces(mesh_local, method="cylinder", return_submeshes=False)
                        meta = getattr(cyl, "meta", {}) or {}
                        migu_idx = getattr(cyl, "migu_face_indices", None)
                        if bool(meta.get("cylinder_ok", False)) and isinstance(migu_idx, np.ndarray) and migu_idx.size:
                            idx = migu_idx.astype(np.int32, copy=False)
                            mode_desc = "원통(반경) | 자동"
                    except Exception:
                        idx = None
                        mode_desc = None

                if idx is None:
                    major_axis = "x" if use_x else "y"

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
                if bootstrap_used:
                    mode_desc = f"{mode_desc} + outer/inner 자동보강" if mode_desc else "outer/inner 자동보강"
                n_sel = int(idx.size)
                if n_sel <= 0:
                    QMessageBox.information(
                        self,
                        "결과 없음",
                        "미구 자동 감지 결과가 없습니다.\n\n"
                        "팁:\n"
                        "- 기와를 정치 후(상면/하면이 위/아래) 다시 시도\n"
                        "- Ctrl을 누르고 다시 클릭(축 전환)\n"
                        "- Shift를 누르고 클릭(둘레 경계 전체 감지)\n"
                        "- Alt를 누르고 클릭(내/외면 자동보강 후 미구 감지)",
                    )
                    return

                try:
                    obj.migu_face_indices.clear()
                    obj.migu_face_indices.update(int(x) for x in idx)
                except Exception:
                    obj.migu_face_indices = set(int(x) for x in idx)
                try:
                    obj.surface_assist_unresolved_face_indices = set()
                    obj.surface_assist_meta = {}
                    obj.surface_assist_runtime = {}
                except Exception:
                    pass

                # Keep sets exclusive (migu wins).
                try:
                    obj.outer_face_indices.difference_update(obj.migu_face_indices)
                    obj.inner_face_indices.difference_update(obj.migu_face_indices)
                except Exception:
                    pass

                self.viewport.status_info = (
                    f"✅ 미구 자동 감지({mode_desc}): migu {len(obj.migu_face_indices):,} faces "
                    f"(Shift=경계, Ctrl=축전환, Alt=내/외면 자동보강)"
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
                    "팁: 필요하면 '경계(면적+자석)'로 추가 보정하세요.\n"
                    "단축: Shift=둘레 경계, Ctrl=축 전환(X↔Y), Alt=내/외면 자동보강",
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
        surface_target = str(options.get("surface_target", "all")).strip().lower()
        if surface_target not in {"all", "outer", "inner", "migu"}:
            surface_target = "all"

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
            surface_target,
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

        meta = dict(getattr(flattened, "meta", {}) or {})
        size_warning = bool(meta.get("flatten_size_warning", False))
        size_guard_applied = bool(meta.get("flatten_size_guard_applied", False))
        dim_ratio_before = meta.get("flatten_size_dim_ratio_before", None)
        dim_ratio_after = meta.get("flatten_size_dim_ratio_after", None)
        guard_scale = meta.get("flatten_size_guard_scale", None)
        dim_ratio_before_f = _safe_float_or_none(dim_ratio_before)
        dim_ratio_after_f = _safe_float_or_none(dim_ratio_after)
        guard_scale_f = _safe_float_or_none(guard_scale)

        status_prefix = "⚠️ 펼침 완료" if size_warning else "✅ 펼침 완료"
        self.status_info.setText(
            f"{status_prefix}: {flattened.width:.2f} x {flattened.height:.2f} {flattened.original_mesh.unit} "
            f"(왜곡 평균 {flattened.mean_distortion:.1%})"
        )

        size_note = ""
        if size_warning:
            if size_guard_applied:
                try:
                    size_note = (
                        f"\n- 크기 안정화 보정: 적용됨"
                        f"\n  (비율 {float(dim_ratio_before_f or 0.0):.2f}x → {float(dim_ratio_after_f or 0.0):.2f}x,"
                        f" 스케일 {float(guard_scale_f or 0.0):.4f})"
                    )
                except Exception:
                    size_note = "\n- 크기 안정화 보정: 적용됨"
            else:
                try:
                    size_note = (
                        f"\n- 크기 경고: 원본 대비 펼침 최대 길이 비율이 큽니다"
                        f"\n  (현재 약 {float(dim_ratio_before_f or 0.0):.2f}x)"
                    )
                except Exception:
                    size_note = "\n- 크기 경고: 원본 대비 펼침 크기가 큰 편입니다."

        QMessageBox.information(
            self,
            "펼침 완료",
            f"펼침이 완료되었습니다.\n\n"
            f"- 크기: {flattened.width:.2f} x {flattened.height:.2f} {flattened.original_mesh.unit}\n"
            f"- 왜곡(평균/최대): {flattened.mean_distortion:.1%} / {flattened.max_distortion:.1%}"
            f"{size_note}\n\n"
            f"이제 '펼친 결과 SVG 저장' 또는 '탁본 이미지 내보내기'를 사용할 수 있습니다."
        )

    def _on_flatten_task_failed(self, message: str):
        self.status_info.setText("❌ 펼침 실패")
        QMessageBox.critical(self, "오류", self._format_error_message("펼침 처리 중 오류 발생:", message))

    def on_export_requested(self, data):
        """내보내기 요청 처리"""
        export_type = data.get('type')
        target = str((data or {}).get("target", "all") or "all").strip().lower()
        if target not in {"all", "outer", "inner", "migu"}:
            target = "all"
        
        if export_type == 'profile_2d':
            self.export_2d_profile(data.get('view'))
            return

        if export_type == "profile_2d_package":
            self.export_2d_profile_package()
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

                face_set = None
                if target != "all":
                    attr = f"{target}_face_indices"
                    face_set = getattr(obj, attr, None) or set()
                    if not face_set:
                        QMessageBox.warning(
                            self,
                            "경고",
                            f"'{target}' 표면 지정이 비어 있습니다.\n\n"
                            "우측 '표면 선택/지정'에서 외면/내면/미구를 먼저 지정하거나,\n"
                            "탁본 대상=전체로 내보내세요.",
                        )
                        return

                flatten_options_target = dict(flatten_options)
                flatten_options_target["surface_target"] = target
                key = self._flatten_cache_key(obj, flatten_options_target)
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
                        if target != "all":
                            ids = np.asarray(sorted(list(face_set or [])), dtype=np.int32).reshape(-1)
                            mesh = mesh.extract_submesh(ids)
                        flattened = MainWindow._compute_flattened_mesh(mesh, opts)

                    # DPI 기준으로 출력 폭 계산 (실측 스케일 유지를 위해)
                    unit = (flattened.original_mesh.unit or "mm").lower()
                    width_in = _width_in_inches(float(flattened.width), unit)
                    width_pixels = max(MIN_EXPORT_WIDTH_PX, int(width_in * dpi))
                    width_pixels = min(width_pixels, MAX_EXPORT_WIDTH_PX)  # output width guard

                    visualizer = SurfaceVisualizer(default_dpi=dpi)
                    rubbing = visualizer.generate_rubbing(
                        flattened,
                        width_pixels=width_pixels,
                        preset="자연(이미지)",
                    )
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

        elif export_type == 'rubbing_digital':
            filepath, _ = QFileDialog.getSaveFileName(
                self, "디지털 탁본 저장 (곡률 제거)", "", "PNG (*.png);;TIFF (*.tiff)"
            )
            if filepath:
                self.status_info.setText(f"내보내기: {filepath}")

                dpi = int(self.export_panel.spin_dpi.value())
                include_scale = bool(self.export_panel.check_scale_bar.isChecked())

                base = obj.mesh

                face_set = None
                if target != "all":
                    attr = f"{target}_face_indices"
                    face_set = getattr(obj, attr, None) or set()
                    if not face_set:
                        QMessageBox.warning(
                            self,
                            "경고",
                            f"'{target}' 표면 지정이 비어 있습니다.\n\n"
                            "우측 '표면 선택/지정'에서 외면/내면/미구를 먼저 지정하거나,\n"
                            "탁본 대상=전체로 내보내세요.",
                        )
                        return
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

                # Always use fast cylindrical unwrapping for digital rubbing.
                opts = dict(flatten_options)
                opts["method"] = "원통 펼침"

                def task_export_rubbing_digital():
                    from src.core.surface_visualizer import SurfaceVisualizer

                    mesh = MainWindow._build_world_mesh_from_transform(
                        base, translation=translation, rotation=rotation, scale=scale
                    )
                    if target != "all":
                        ids = np.asarray(sorted(list(face_set or [])), dtype=np.int32).reshape(-1)
                        mesh = mesh.extract_submesh(ids)
                    flattened = MainWindow._compute_flattened_mesh(mesh, opts)

                    # DPI 기준으로 출력 폭 계산 (실측 스케일 유지를 위해)
                    unit = (flattened.original_mesh.unit or "mm").lower()
                    width_in = _width_in_inches(float(flattened.width), unit)
                    width_pixels = max(MIN_EXPORT_WIDTH_PX, int(width_in * dpi))
                    width_pixels = min(width_pixels, MAX_EXPORT_WIDTH_PX)  # output width guard

                    visualizer = SurfaceVisualizer(default_dpi=dpi)
                    # Prefer the image-based preset to reduce aliasing/noise on scanned meshes.
                    rubbing = visualizer.generate_rubbing(
                        flattened,
                        width_pixels=width_pixels,
                        preset="디지털(곡률 제거)",
                    )
                    rubbing.save(filepath, include_scale_bar=include_scale)
                    return filepath

                def on_done_export_rubbing_digital(_result: Any):
                    QMessageBox.information(self, "완료", f"디지털 탁본 이미지가 저장되었습니다:\n{filepath}")
                    self.status_info.setText(f"✅ 저장 완료: {Path(filepath).name}")

                def on_failed(message: str):
                    self.status_info.setText("❌ 저장 실패")
                    QMessageBox.critical(self, "오류", self._format_error_message("디지털 탁본 저장 중 오류 발생:", message))

                self._start_task(
                    title="내보내기",
                    label="디지털 탁본 생성/저장 중...",
                    thread=TaskThread("export_rubbing_digital", task_export_rubbing_digital),
                    on_done=on_done_export_rubbing_digital,
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
                    unit = (flattened.original_mesh.unit or DEFAULT_MESH_UNIT).lower()
                    svg_unit = unit if unit in ("mm", "cm") else DEFAULT_MESH_UNIT
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
                            rubbing_preset="자연(이미지)",
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

        elif export_type == 'sheet_svg_digital':
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "통합 SVG 저장 (디지털 탁본/원통)",
                "rubbing_sheet_digital.svg",
                "Scalable Vector Graphics (*.svg)",
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

                def task_export_sheet_svg_digital():
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
                            flatten_iterations=0,
                            flatten_method="cylinder",
                            flatten_distortion=0.0,
                            cylinder_axis=str(flatten_options.get("direction", "auto")),
                            cylinder_radius=cylinder_radius,
                            rubbing_preset="디지털(곡률 제거)",
                        ),
                    )
                    return filepath

                def on_done_export_sheet_svg_digital(_result: Any):
                    QMessageBox.information(self, "완료", f"통합 SVG(디지털 탁본)가 저장되었습니다:\n{filepath}")
                    self.status_info.setText(f"✅ 저장 완료: {Path(filepath).name}")

                def on_failed(message: str):
                    self.status_info.setText("❌ 저장 실패")
                    QMessageBox.critical(self, "오류", self._format_error_message("통합 SVG 저장 중 오류 발생:", message))

                self._start_task(
                    title="내보내기",
                    label="통합 SVG(디지털 탁본) 생성/저장 중...",
                    thread=TaskThread("export_sheet_svg_digital", task_export_sheet_svg_digital),
                    on_done=on_done_export_sheet_svg_digital,
                    on_failed=on_failed,
                )

        elif export_type == 'mesh_outer':
            filepath, _ = QFileDialog.getSaveFileName(
                self, "외면 메쉬 저장", "", "OBJ (*.obj);;STL (*.stl);;PLY (*.ply)"
            )
            if filepath:
                manual_outer_idx = np.asarray(
                    sorted(list(getattr(obj, "outer_face_indices", set()) or [])),
                    dtype=np.int32,
                ).reshape(-1)
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
                    source = "manual"
                    if manual_outer_idx.size > 0:
                        outer_idx = manual_outer_idx.astype(np.int32, copy=True)
                    else:
                        source = "auto"
                        try:
                            self._apply_surface_stability_presets(mesh)
                        except Exception:
                            pass
                        separator = SurfaceSeparator()
                        result = separator.auto_detect_surfaces(mesh, return_submeshes=False)
                        outer_idx = np.asarray(
                            getattr(result, "outer_face_indices", np.zeros((0,), dtype=np.int32)),
                            dtype=np.int32,
                        ).reshape(-1)

                    n_faces = int(getattr(mesh, "n_faces", 0) or 0)
                    if n_faces > 0 and outer_idx.size > 0:
                        valid = (outer_idx >= 0) & (outer_idx < n_faces)
                        outer_idx = np.unique(outer_idx[valid]).astype(np.int32, copy=False)
                    if outer_idx.size <= 0:
                        return {"status": "no_outer"}
                    outer = mesh.extract_submesh(outer_idx)
                    MeshProcessor().save_mesh(outer, filepath)
                    return {"status": "ok", "source": source}

                def on_done_export_mesh_outer(result: Any):
                    if isinstance(result, dict) and result.get("status") == "no_outer":
                        QMessageBox.warning(self, "경고", "외면을 감지하지 못했습니다.")
                        return
                    src = "수동 지정" if isinstance(result, dict) and result.get("source") == "manual" else "자동 감지"
                    QMessageBox.information(self, "완료", f"외면 메쉬가 저장되었습니다 ({src}):\n{filepath}")

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
                manual_inner_idx = np.asarray(
                    sorted(list(getattr(obj, "inner_face_indices", set()) or [])),
                    dtype=np.int32,
                ).reshape(-1)
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
                    source = "manual"
                    if manual_inner_idx.size > 0:
                        inner_idx = manual_inner_idx.astype(np.int32, copy=True)
                    else:
                        source = "auto"
                        try:
                            self._apply_surface_stability_presets(mesh)
                        except Exception:
                            pass
                        separator = SurfaceSeparator()
                        result = separator.auto_detect_surfaces(mesh, return_submeshes=False)
                        inner_idx = np.asarray(
                            getattr(result, "inner_face_indices", np.zeros((0,), dtype=np.int32)),
                            dtype=np.int32,
                        ).reshape(-1)

                    n_faces = int(getattr(mesh, "n_faces", 0) or 0)
                    if n_faces > 0 and inner_idx.size > 0:
                        valid = (inner_idx >= 0) & (inner_idx < n_faces)
                        inner_idx = np.unique(inner_idx[valid]).astype(np.int32, copy=False)
                    if inner_idx.size <= 0:
                        return {"status": "no_inner"}
                    inner = mesh.extract_submesh(inner_idx)
                    MeshProcessor().save_mesh(inner, filepath)
                    return {"status": "ok", "source": source}

                def on_done_export_mesh_inner(result: Any):
                    if isinstance(result, dict) and result.get("status") == "no_inner":
                        QMessageBox.warning(self, "경고", "내면을 감지하지 못했습니다.")
                        return
                    src = "수동 지정" if isinstance(result, dict) and result.get("source") == "manual" else "자동 감지"
                    QMessageBox.information(self, "완료", f"내면 메쉬가 저장되었습니다 ({src}):\n{filepath}")

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
                'top': (0.0, 90.0),
                'bottom': (0.0, -90.0),
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
            try:
                self._status_task_begin("2D 도면(SVG) 내보내기", maximum=None, value=None)
            except Exception:
                pass

            include_grid = True
            include_feature_lines = False
            feature_angle_deg = 60.0
            try:
                include_grid = bool(self.export_panel.check_profile_include_grid.isChecked())
            except Exception:
                include_grid = True
            try:
                include_feature_lines = bool(self.export_panel.check_profile_feature_lines.isChecked())
            except Exception:
                include_feature_lines = False
            try:
                feature_angle_deg = float(self.export_panel.spin_profile_feature_angle.value())
            except Exception:
                feature_angle_deg = 60.0

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
                include_grid=bool(include_grid),
                include_feature_lines=bool(include_feature_lines),
                feature_angle_deg=float(feature_angle_deg),
            )
            self._profile_export_thread.done.connect(self._on_profile_export_done)
            self._profile_export_thread.failed.connect(self._on_profile_export_failed)
            self._profile_export_thread.finished.connect(self._on_profile_export_finished)
            self._profile_export_thread.start()
            self.status_info.setText(f"내보내기 시작: {Path(filepath).name}")
            return

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
    
    def export_2d_profile_package(self):
        """2D 실측 도면(SVG) 6방향 패키지 내보내기"""
        obj = self.viewport.selected_obj
        if not obj:
            QMessageBox.warning(self, "경고", "선택된 메쉬가 없습니다.")
            return

        mesh_data = getattr(obj, "mesh", None)
        if mesh_data is None:
            QMessageBox.warning(self, "경고", "선택된 객체에 메쉬 데이터가 없습니다.")
            return

        running_single = getattr(self, "_profile_export_thread", None)
        running_pkg = getattr(self, "_profile_package_export_thread", None)
        if (
            (running_single is not None and running_single.isRunning())
            or (running_pkg is not None and running_pkg.isRunning())
        ):
            QMessageBox.information(self, "내보내기", "이미 내보내기 작업이 진행 중입니다.")
            return

        default_dir = str(Path.home())
        mesh_fp = None
        try:
            mesh_fp = getattr(mesh_data, "filepath", None)
            if mesh_fp:
                default_dir = str(Path(str(mesh_fp)).parent)
        except Exception:
            mesh_fp = None

        parent_dir = QFileDialog.getExistingDirectory(
            self,
            "2D 도면 패키지 저장 폴더 선택",
            default_dir,
        )
        if not parent_dir:
            return

        base_name = "mesh"
        try:
            if mesh_fp:
                base_name = Path(str(mesh_fp)).stem
        except Exception:
            base_name = "mesh"

        # 폴더명 생성 (Windows 금지 문자 치환)
        invalid = '<>:"/\\\\|?*'
        safe_name = "".join("_" if c in invalid else c for c in str(base_name)).strip() or "mesh"

        parent = Path(parent_dir)
        stem = f"{safe_name}_profiles"
        package_dir = parent / stem
        if package_dir.exists():
            for i in range(1, 1000):
                cand = parent / f"{stem}_{i}"
                if not cand.exists():
                    package_dir = cand
                    break
            else:
                QMessageBox.critical(self, "오류", "패키지 폴더 이름을 만들 수 없습니다. 다른 폴더를 선택하세요.")
                return

        try:
            package_dir.mkdir(parents=True, exist_ok=False)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"폴더 생성 실패:\n{type(e).__name__}: {e}")
            return

        # 카메라/뷰 상태 저장
        cam_state = None
        try:
            cam = self.viewport.camera
            cam_state = (
                float(cam.distance),
                float(cam.azimuth),
                float(cam.elevation),
                cam.center.copy(),
                cam.pan_offset.copy(),
            )
        except Exception:
            cam_state = None

        translation = np.asarray(getattr(obj, "translation", np.zeros(3)), dtype=np.float64).copy()
        rotation = np.asarray(getattr(obj, "rotation", np.zeros(3)), dtype=np.float64).copy()
        scale = float(getattr(obj, "scale", 1.0))

        # 단면/가이드 라인을 포함하도록 bounds 확장
        try:
            bounds = np.asarray(obj.get_world_bounds(), dtype=np.float64)
            extra_pts = []
            for ln in self.viewport.get_cut_sections_world() or []:
                for p in ln or []:
                    extra_pts.append(np.asarray(p, dtype=np.float64))
            if extra_pts:
                ep = np.vstack(extra_pts)
                bounds[0] = np.minimum(bounds[0], ep.min(axis=0))
                bounds[1] = np.maximum(bounds[1], ep.max(axis=0))
        except Exception:
            bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float64)

        views = ["top", "bottom", "front", "back", "left", "right"]
        view_map = {
            "top": (0.0, 90.0),
            "bottom": (0.0, -90.0),
            "front": (-90.0, 0.0),
            "back": (90.0, 0.0),
            "left": (180.0, 0.0),
            "right": (0.0, 0.0),
        }

        resolution = 2048
        grid_spacing = 1.0  # cm
        include_grid = True
        try:
            cb = getattr(self.export_panel, "check_profile_include_grid", None)
            if cb is not None:
                include_grid = bool(cb.isChecked())
        except Exception:
            include_grid = True

        include_feature_lines = False
        feature_angle_deg = 60.0
        try:
            cbf = getattr(self.export_panel, "check_profile_feature_lines", None)
            if cbf is not None:
                include_feature_lines = bool(cbf.isChecked())
            sp = getattr(self.export_panel, "spin_profile_feature_angle", None)
            if sp is not None:
                feature_angle_deg = float(sp.value())
        except Exception:
            include_feature_lines = False
            feature_angle_deg = 60.0

        dlg = QProgressDialog("2D 도면(SVG) 패키지 내보내는 중...", None, 0, len(views), self)
        dlg.setWindowTitle("내보내기")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        dlg.show()

        self._profile_package_export_dialog = dlg
        try:
            self._status_task_begin("패키지 내보내기", maximum=len(views), value=0)
        except Exception:
            pass
        self._profile_package_export_state = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "package_dir": str(package_dir),
            "mesh_filepath": str(mesh_fp) if mesh_fp else None,
            "mesh_unit": str(getattr(mesh_data, "unit", "mm")),
            "mesh_data": mesh_data,
            "translation": translation,
            "rotation": rotation,
            "scale": scale,
            "bounds": bounds,
            "cam_state": cam_state,
            "views": views,
            "view_map": view_map,
            "index": 0,
            "results": {},
            "resolution": resolution,
            "grid_spacing": grid_spacing,
            "include_grid": include_grid,
            "include_feature_lines": include_feature_lines,
            "feature_angle_deg": feature_angle_deg,
            "cut_lines_world": self.viewport.get_cut_lines_world(),
            "cut_profiles_world": self.viewport.get_cut_sections_world(),
        }

        self.status_info.setText(f"내보내기 시작(패키지): {package_dir.name}")
        QTimer.singleShot(0, self._start_next_profile_package_view)

    def _start_next_profile_package_view(self):
        state = getattr(self, "_profile_package_export_state", None)
        if not isinstance(state, dict):
            return

        views = list(state.get("views") or [])
        idx = int(state.get("index", 0))
        if idx >= len(views):
            self._finish_profile_package_export()
            return

        view = str(views[idx])
        dlg = getattr(self, "_profile_package_export_dialog", None)
        if dlg is not None:
            dlg.setLabelText(f"[{idx+1}/{len(views)}] {view} 내보내는 중...")
            try:
                dlg.setValue(idx)
            except Exception:
                pass

        view_map = state.get("view_map") or {}
        bounds = np.asarray(state.get("bounds"), dtype=np.float64)
        resolution = int(state.get("resolution", 2048))

        try:
            try:
                cam = self.viewport.camera
                cam.fit_to_bounds(bounds)
                if view in view_map:
                    az, el = view_map[view]
                    cam.azimuth, cam.elevation = float(az), float(el)
            except Exception:
                pass

            qimage, mv, proj, vp = self.viewport.capture_high_res_image(
                width=resolution,
                height=resolution,
                only_selected=True,
                orthographic=True,
            )

            ba = QByteArray()
            qbuf = QBuffer(ba)
            qbuf.open(QIODevice.OpenModeFlag.WriteOnly)
            qimage.save(qbuf, "PNG")
            qbuf.close()
            pil_img = Image.open(io.BytesIO(ba.data()))
        except Exception as e:
            self._abort_profile_package_export(view, f"{type(e).__name__}: {e}")
            return

        package_dir = Path(str(state.get("package_dir")))
        view_dir = package_dir / str(view)
        try:
            view_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._abort_profile_package_export(view, f"{type(e).__name__}: {e}")
            return

        out_name = f"{view}.svg"
        out_path = str(view_dir / out_name)

        thread = ProfileExportThread(
            mesh_data=state.get("mesh_data"),
            view=view,
            output_path=out_path,
            translation=np.asarray(state.get("translation"), dtype=np.float64),
            rotation=np.asarray(state.get("rotation"), dtype=np.float64),
            scale=float(state.get("scale", 1.0)),
            viewport_image=pil_img,
            opengl_matrices=(mv, proj, vp),
            cut_lines_world=state.get("cut_lines_world") or [],
            cut_profiles_world=state.get("cut_profiles_world") or [],
            resolution=resolution,
            grid_spacing=float(state.get("grid_spacing", 1.0)),
            include_grid=bool(state.get("include_grid", True)),
            include_feature_lines=bool(state.get("include_feature_lines", False)),
            feature_angle_deg=float(state.get("feature_angle_deg", 60.0)),
        )

        self._profile_package_export_thread = thread
        thread.done.connect(lambda p, v=view: self._on_profile_package_view_done(v, p))
        thread.failed.connect(lambda m, v=view: self._abort_profile_package_export(v, m))
        thread.finished.connect(self._on_profile_package_view_finished)
        thread.start()

    def _on_profile_package_view_done(self, view: str, result_path: str):
        state = getattr(self, "_profile_package_export_state", None)
        if not isinstance(state, dict):
            return

        idx = int(state.get("index", 0))
        try:
            package_dir = Path(str(state.get("package_dir")))
            rp = Path(str(result_path))
            try:
                rel = rp.relative_to(package_dir)
                rel_s = rel.as_posix()
            except Exception:
                rel_s = rp.name
        except Exception:
            rel_s = str(Path(str(result_path)).name)

        state.setdefault("results", {})[str(view)] = rel_s
        state["index"] = idx + 1

        dlg = getattr(self, "_profile_package_export_dialog", None)
        if dlg is not None:
            try:
                dlg.setValue(int(state["index"]))
            except Exception:
                pass
        try:
            total = int(len(state.get("views") or []))
            cur = int(state.get("index", 0))
            if total > 0:
                self._status_task_update(text=f"패키지 내보내기 {cur}/{total}", maximum=total, value=cur)
        except Exception:
            pass

    def _on_profile_package_view_finished(self):
        self._profile_package_export_thread = None
        QTimer.singleShot(0, self._start_next_profile_package_view)

    def _finish_profile_package_export(self):
        state = getattr(self, "_profile_package_export_state", None)
        if not isinstance(state, dict):
            return

        package_dir = Path(str(state.get("package_dir")))
        views = list(state.get("views") or [])
        results = dict(state.get("results") or {})

        try:
            manifest = {
                "app": {"name": APP_NAME, "version": APP_VERSION},
                "exported_at": datetime.now().isoformat(timespec="seconds"),
                "mesh": {"filepath": state.get("mesh_filepath"), "unit": state.get("mesh_unit")},
                "transform": {
                    "translation": np.asarray(state.get("translation"), dtype=np.float64).reshape(-1).tolist(),
                    "rotation": np.asarray(state.get("rotation"), dtype=np.float64).reshape(-1).tolist(),
                    "scale": float(state.get("scale", 1.0)),
                },
                "settings": {
                    "resolution": int(state.get("resolution", 2048)),
                    "grid_spacing_cm": float(state.get("grid_spacing", 1.0)),
                    "include_grid": bool(state.get("include_grid", True)),
                    "include_feature_lines": bool(state.get("include_feature_lines", False)),
                    "feature_angle_deg": float(state.get("feature_angle_deg", 60.0)),
                },
                "views": [{"view": v, "file": results.get(v)} for v in views],
            }
            (package_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        self._cleanup_profile_package_export()
        QMessageBox.information(self, "완료", f"2D 도면 패키지가 저장되었습니다:\n{package_dir}")
        self.status_info.setText(f"✅ 패키지 저장 완료: {package_dir.name}")

    def _abort_profile_package_export(self, view: str, message: str):
        package_dir = None
        try:
            state = getattr(self, "_profile_package_export_state", None)
            if isinstance(state, dict):
                package_dir = state.get("package_dir")
        except Exception:
            package_dir = None

        self._cleanup_profile_package_export()
        hint = f"\n\n폴더: {package_dir}" if package_dir else ""
        QMessageBox.critical(
            self,
            "오류",
            self._format_error_message(f"패키지 내보내기 실패 ({view}):", f"{message}{hint}"),
        )
        self.status_info.setText("❌ 패키지 내보내기 실패")

    def _cleanup_profile_package_export(self):
        dlg = getattr(self, "_profile_package_export_dialog", None)
        if dlg is not None:
            try:
                dlg.close()
            except Exception:
                pass
        self._profile_package_export_dialog = None

        state = getattr(self, "_profile_package_export_state", None)
        cam_state = None
        if isinstance(state, dict):
            cam_state = state.get("cam_state")
        self._profile_package_export_state = None

        if cam_state is not None:
            try:
                cam = self.viewport.camera
                cam.distance, cam.azimuth, cam.elevation = cam_state[0], cam_state[1], cam_state[2]
                cam.center = cam_state[3]
                cam.pan_offset = cam_state[4]
                self.viewport.update()
            except Exception:
                pass
        try:
            self._status_task_end()
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
        self.viewport._front_back_ortho_enabled = False
        self.viewport.camera.reset()
        self.viewport.update()

    def fit_view(self):
        self.viewport._front_back_ortho_enabled = False
        obj = self.viewport.selected_obj
        if obj:
            try:
                wb = np.asarray(obj.get_world_bounds(), dtype=np.float64)
                if wb.shape == (2, 3) and np.isfinite(wb).all():
                    self.viewport.camera.fit_to_bounds(wb)
                    self.viewport.camera.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                    self.viewport.update()
                    return
            except Exception:
                pass
            try:
                self.viewport.fit_view_to_selected_object()
            except Exception:
                pass
        elif self.current_mesh is not None:
            try:
                b = np.asarray(self.current_mesh.bounds, dtype=np.float64)
                if b.shape == (2, 3) and np.isfinite(b).all():
                    self.viewport.camera.fit_to_bounds(b)
                    self.viewport.camera.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                    self.viewport.update()
            except Exception:
                pass

    def set_view(self, azimuth: float, elevation: float):
        try:
            az = float(azimuth)
            el = float(elevation)
        except Exception:
            return

        az = ((az + 180.0) % 360.0) - 180.0
        for tgt in (-180.0, -90.0, 0.0, 90.0, 180.0):
            if abs(az - tgt) <= 1e-6:
                az = tgt
                break
        if abs(el) <= 1e-6:
            el = 0.0
        if abs(el - 90.0) <= 1e-6:
            el = 90.0
        elif abs(el + 90.0) <= 1e-6:
            el = -90.0

        cam = self.viewport.camera
        cam.azimuth = az
        cam.elevation = max(-90.0, min(90.0, el))

        # Keep 6-face views tightly framed around visible geometry.
        try:
            bounds = None
            obj = self.viewport.selected_obj
            if obj is not None:
                b = np.asarray(obj.get_world_bounds(), dtype=np.float64)
                if b.shape == (2, 3) and np.isfinite(b).all():
                    bounds = b
            else:
                boxes = []
                for o in list(getattr(self.viewport, "objects", []) or []):
                    if not bool(getattr(o, "visible", True)):
                        continue
                    b = np.asarray(o.get_world_bounds(), dtype=np.float64)
                    if b.shape == (2, 3) and np.isfinite(b).all():
                        boxes.append(b)
                if boxes:
                    wb = np.vstack(boxes)
                    bounds = np.array([wb.min(axis=0), wb.max(axis=0)], dtype=np.float64)
            if bounds is not None:
                center = (bounds[0] + bounds[1]) * 0.5
                ext = bounds[1] - bounds[0]
                max_dim = float(np.max(ext))
                if not np.isfinite(max_dim) or max_dim <= 1e-6:
                    max_dim = 10.0
                cam.center = np.asarray(center, dtype=np.float64)
                cam.distance = float(max(cam.min_distance, min(cam.max_distance, max_dim * 1.35)))
        except Exception:
            pass

        try:
            cam.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        except Exception:
            pass
        # Keep directional view buttons free-orbit friendly (no sticky ortho lock).
        self.viewport._front_back_ortho_enabled = False
        self.viewport.update()

    def toggle_curvature_mode(self, enabled: bool):
        """곡률 측정 모드 토글"""
        if enabled:
            try:
                self._disable_measure_mode()
            except Exception:
                pass
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
        
        # 펼침 패널 반경 입력은 mm 기준. arc.radius는 "입력 점(월드/메쉬) 단위" 그대로라서 mesh.unit에 맞춰 mm로 변환.
        from src.core.unit_utils import mesh_units_to_mm

        radius_mm = mesh_units_to_mm(float(arc.radius), getattr(getattr(obj, "mesh", None), "unit", None))
        if np.isfinite(radius_mm) and radius_mm > 0:
            self.flatten_panel.spin_radius.setValue(float(radius_mm))
        
        self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
        arc_count = len(obj.fitted_arcs)
        self.status_info.setText(
            f"✅ 원호 #{arc_count} 생성됨 (월드 고정): 반지름 = {radius_mm:.1f} mm"
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
    
    def _disable_measure_mode(self) -> None:
        panel = getattr(self, "measure_panel", None)
        if panel is not None:
            try:
                panel.set_measure_checked(False)
                panel.set_points_count(0)
            except Exception:
                pass

        try:
            if self.viewport.picking_mode == "measure":
                self.viewport.picking_mode = "none"
        except Exception:
            pass

        try:
            self.viewport.clear_measure_picks()
        except Exception:
            pass

    """
    NOTE: 아래 블록은 이전 패치 과정에서 깨진 상태로 남은 치수 측정 메서드들입니다.
    안전하게 보존만 하고(문자열로 처리), 아래에 정상 구현을 다시 정의합니다.
    (legacy measurement block continues below)

    def toggle_measure_mode(self, enabled: bool) -> None:
        \"\"\"치수(거리/지름) 측정 모드 토글\"\"\"
        if enabled and self.viewport.selected_obj is None:
            QMessageBox.warning(self, \"경고\", \"먼저 메쉬를 선택하세요.\")
            self._disable_measure_mode()
            self.viewport.update()
            return

        if enabled:
            # 다른 입력 모드와 충돌 방지
            try:
                if self.flatten_panel.btn_measure.isChecked():
                    self.flatten_panel.btn_measure.blockSignals(True)
                    self.flatten_panel.btn_measure.setChecked(False)
                    self.flatten_panel.btn_measure.blockSignals(False)
            except Exception:
                pass
            try:
                self.viewport.curvature_pick_mode = False
            except Exception:
                pass

            try:
                if bool(getattr(self.viewport, \"crosshair_enabled\", False)):
                    self.viewport.crosshair_enabled = False
                    self.section_panel.btn_toggle.blockSignals(True)
                    self.section_panel.btn_toggle.setChecked(False)
                    self.section_panel.btn_toggle.blockSignals(False)
            except Exception:
                pass

            try:
                if bool(getattr(self.viewport, \"cut_lines_enabled\", False)):
                    self.viewport.set_cut_lines_enabled(False)
                    self.section_panel.btn_line.blockSignals(True)
                    self.section_panel.btn_line.setChecked(False)
                    self.section_panel.btn_line.blockSignals(False)
            except Exception:
                pass

            try:
                if bool(getattr(self.viewport, \"roi_enabled\", False)):
                    self.viewport.roi_enabled = False
                    self.viewport.active_roi_edge = None
                    self.section_panel.btn_roi.blockSignals(True)
                    self.section_panel.btn_roi.setChecked(False)
                    self.section_panel.btn_roi.blockSignals(False)
                    self.section_panel.btn_silhouette.setEnabled(False)
            except Exception:
                pass

            try:
                self.viewport.clear_measure_picks()
            except Exception:
                pass
            try:
                self.measure_panel.set_points_count(0)
            except Exception:
                pass

            self.viewport.picking_mode = \"measure\"
            self.status_info.setText(\"📏 치수 측정 모드: Shift+클릭으로 점을 찍으세요.\")
        else:
            try:
                if self.viewport.picking_mode == \"measure\":
                    self.viewport.picking_mode = \"none\"
            except Exception:
                pass
            try:
                self.viewport.clear_measure_picks()
            except Exception:
                pass
            try:
                self.measure_panel.set_points_count(0)
            except Exception:
                pass
            self.status_info.setText(\"📏 치수 측정 모드 종료\")

        self.viewport.update()

    def on_measure_mode_changed(self, mode: str) -> None:
        try:
            self.viewport.clear_measure_picks()
            self.measure_panel.set_points_count(0)
            self.viewport.update()
        except Exception:
            pass

        if str(mode) == \"diameter\":
            self.status_info.setText(\"📏 지름 모드: 점 3개 이상 선택 후 '지름 계산'.\")
        else:
            self.status_info.setText(\"📏 거리 모드: 점 2개 선택하면 자동 계산.\")

    def on_measure_point_picked(self, _point: np.ndarray) -> None:
        panel = getattr(self, \"measure_panel\", None)
        if panel is None:
            return

        try:
            pts = list(getattr(self.viewport, \"measure_picked_points\", []) or [])
        except Exception:
            pts = []

        panel.set_points_count(len(pts))

        if panel.mode != \"distance\":
            return

        if len(pts) < 2:
            return

        p0 = np.asarray(pts[-2], dtype=np.float64).reshape(-1)
        p1 = np.asarray(pts[-1], dtype=np.float64).reshape(-1)
        if p0.size < 3 or p1.size < 3:
            return
        if not np.isfinite(p0[:3]).all() or not np.isfinite(p1[:3]).all():
            return

        dist_cm = float(np.linalg.norm(p1[:3] - p0[:3]))
        if not np.isfinite(dist_cm):
            return

        dist_mm = dist_cm * 10.0
        msg = f\"거리: {dist_cm:.2f} cm ({dist_mm:.1f} mm)\"
        panel.append_result(msg)
        self.status_info.setText(f\"📏 {msg}\")

        try:
            self.viewport.clear_measure_picks()
            panel.set_points_count(0)
            self.viewport.update()
        except Exception:
            pass

    def fit_measure_circle(self) -> None:
        panel = getattr(self, \"measure_panel\", None)
        if panel is None:
            return

        if panel.mode != \"diameter\":
            QMessageBox.information(self, \"안내\", \"지름/직경 모드에서만 사용할 수 있습니다.\")
            return

        try:
            pts = np.asarray(getattr(self.viewport, \"measure_picked_points\", []) or [], dtype=np.float64)
        except Exception:
            pts = np.zeros((0, 3), dtype=np.float64)

        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 3:
            QMessageBox.warning(
                self,
                \"경고\",
                \"최소 3개의 포인트가 필요합니다.\\nShift+클릭으로 점을 더 찍어주세요.\",
            )
            return

        from src.core.curvature_fitter import CurvatureFitter

        fitter = CurvatureFitter()
        arc = fitter.fit_arc(pts[:, :3])
        if arc is None:
            QMessageBox.warning(self, \"경고\", \"원 맞추기에 실패했습니다. 포인트를 다시 선택해보세요.\")
            return

        diameter_cm = float(arc.radius) * 2.0
        diameter_mm = diameter_cm * 10.0
        msg = f\"지름: {diameter_cm:.2f} cm ({diameter_mm:.1f} mm)\"
        panel.append_result(msg)
        self.status_info.setText(f\"📏 {msg}\")

        try:
            self.viewport.clear_measure_picks()
            panel.set_points_count(0)
            self.viewport.update()
        except Exception:
            pass

    def clear_measure_points(self) -> None:
        try:
            self.viewport.clear_measure_picks()
            self.measure_panel.set_points_count(0)
            self.viewport.update()
            self.status_info.setText(\"🧹 측정 포인트 초기화\")
        except Exception:
            pass

    def copy_measure_results(self) -> None:
        panel = getattr(self, \"measure_panel\", None)
        if panel is None:
            return

        text = panel.results_text().strip()
        if not text:
            return

        try:
            QApplication.clipboard().setText(text)
            self.status_info.setText(\"📋 측정 결과 복사됨\")
        except Exception:
            pass

    def clear_measure_results(self) -> None:
        try:
            self.measure_panel.clear_results()
            self.status_info.setText(\"🗑️ 측정 결과 지움\")
        except Exception:
            pass

    """

    def toggle_measure_mode(self, enabled: bool) -> None:
        """치수(거리/지름) 측정 모드 토글"""
        if enabled and self.viewport.selected_obj is None:
            QMessageBox.warning(self, "경고", "먼저 메쉬를 선택하세요.")
            try:
                self.measure_panel.set_measure_checked(False)
            except Exception:
                pass
            self._disable_measure_mode()
            self.viewport.update()
            return

        if enabled:
            # 다른 입력 모드와 충돌 방지
            try:
                if self.flatten_panel.btn_measure.isChecked():
                    self.flatten_panel.btn_measure.blockSignals(True)
                    self.flatten_panel.btn_measure.setChecked(False)
                    self.flatten_panel.btn_measure.blockSignals(False)
            except Exception:
                pass
            try:
                self.viewport.curvature_pick_mode = False
            except Exception:
                pass

            # Crosshair / Cut-lines / ROI 는 입력 충돌이 잦아서 측정 모드에서는 강제 해제
            try:
                if bool(getattr(self.viewport, "crosshair_enabled", False)):
                    self.viewport.crosshair_enabled = False
                    self.section_panel.btn_toggle.blockSignals(True)
                    self.section_panel.btn_toggle.setChecked(False)
                    self.section_panel.btn_toggle.blockSignals(False)
            except Exception:
                pass

            try:
                if bool(getattr(self.viewport, "cut_lines_enabled", False)):
                    self.viewport.set_cut_lines_enabled(False)
                    self.section_panel.btn_line.blockSignals(True)
                    self.section_panel.btn_line.setChecked(False)
                    self.section_panel.btn_line.blockSignals(False)
            except Exception:
                pass

            try:
                if bool(getattr(self.viewport, "roi_enabled", False)):
                    self.viewport.roi_enabled = False
                    self.viewport.active_roi_edge = None
                    self.section_panel.btn_roi.blockSignals(True)
                    self.section_panel.btn_roi.setChecked(False)
                    self.section_panel.btn_roi.blockSignals(False)
                    self.section_panel.btn_silhouette.setEnabled(False)
            except Exception:
                pass

            try:
                self.viewport.clear_measure_picks()
                self.measure_panel.set_points_count(0)
            except Exception:
                pass

            self.viewport.picking_mode = "measure"
            self.status_info.setText("📏 치수 측정 모드: Shift+클릭으로 점을 찍으세요.")
        else:
            try:
                if self.viewport.picking_mode == "measure":
                    self.viewport.picking_mode = "none"
            except Exception:
                pass
            try:
                self.viewport.clear_measure_picks()
                self.measure_panel.set_points_count(0)
            except Exception:
                pass
            self.status_info.setText("📏 치수 측정 모드 종료")

        self.viewport.update()

    def on_measure_mode_changed(self, mode: str) -> None:
        try:
            self.viewport.clear_measure_picks()
            self.measure_panel.set_points_count(0)
            self.viewport.update()
        except Exception:
            pass

        if str(mode) == "diameter":
            self.status_info.setText("📏 지름 모드: 점 3개 이상 선택 후 '지름 계산'.")
        else:
            self.status_info.setText("📏 거리 모드: 점 2개 선택하면 자동 계산.")

    def on_measure_point_picked(self, _point: np.ndarray) -> None:
        panel = getattr(self, "measure_panel", None)
        if panel is None:
            return

        try:
            pts = list(getattr(self.viewport, "measure_picked_points", []) or [])
        except Exception:
            pts = []

        panel.set_points_count(len(pts))

        if panel.mode != "distance":
            return

        if len(pts) < 2:
            return

        p0 = np.asarray(pts[-2], dtype=np.float64).reshape(-1)
        p1 = np.asarray(pts[-1], dtype=np.float64).reshape(-1)
        if p0.size < 3 or p1.size < 3:
            return
        if not np.isfinite(p0[:3]).all() or not np.isfinite(p1[:3]).all():
            return

        dist_cm = float(np.linalg.norm(p1[:3] - p0[:3]))
        if not np.isfinite(dist_cm):
            return

        dist_mm = dist_cm * 10.0
        msg = f"거리: {dist_cm:.2f} cm ({dist_mm:.1f} mm)"
        panel.append_result(msg)
        self.status_info.setText(f"📏 {msg}")

        try:
            self.viewport.clear_measure_picks()
            panel.set_points_count(0)
            self.viewport.update()
        except Exception:
            pass

    def fit_measure_circle(self) -> None:
        panel = getattr(self, "measure_panel", None)
        if panel is None:
            return

        if panel.mode != "diameter":
            QMessageBox.information(self, "안내", "지름/직경 모드에서만 사용할 수 있습니다.")
            return

        try:
            pts = np.asarray(getattr(self.viewport, "measure_picked_points", []) or [], dtype=np.float64)
        except Exception:
            pts = np.zeros((0, 3), dtype=np.float64)

        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 3:
            QMessageBox.warning(self, "경고", "최소 3개의 포인트가 필요합니다.\nShift+클릭으로 점을 더 찍어주세요.")
            return

        from src.core.curvature_fitter import CurvatureFitter

        fitter = CurvatureFitter()
        arc = fitter.fit_arc(pts[:, :3])
        if arc is None:
            QMessageBox.warning(self, "경고", "원 맞추기에 실패했습니다. 포인트를 다시 선택해보세요.")
            return

        from src.core.unit_utils import mesh_units_to_mm

        obj = getattr(self.viewport, "selected_obj", None)
        unit = getattr(getattr(obj, "mesh", None), "unit", None)
        diameter_mm = mesh_units_to_mm(float(arc.radius) * 2.0, unit)
        msg = f"지름: {diameter_mm:.1f} mm"
        panel.append_result(msg)
        self.status_info.setText(f"📏 {msg}")

        try:
            self.viewport.clear_measure_picks()
            panel.set_points_count(0)
            self.viewport.update()
        except Exception:
            pass

    def clear_measure_points(self) -> None:
        try:
            self.viewport.clear_measure_picks()
            self.measure_panel.set_points_count(0)
            self.viewport.update()
            self.status_info.setText("🧹 측정 포인트 초기화")
        except Exception:
            pass

    def copy_measure_results(self) -> None:
        panel = getattr(self, "measure_panel", None)
        if panel is None:
            return

        text = panel.results_text().strip()
        if not text:
            return

        try:
            cb = QApplication.clipboard()
            if cb is not None:
                cb.setText(text)
            label = getattr(self, "status_info", None)
            if label is not None:
                label.setText("📋 측정 결과 복사됨")
        except Exception:
            pass

    def clear_measure_results(self) -> None:
        try:
            self.measure_panel.clear_results()
            self.status_info.setText("🗑️ 측정 결과 지움")
        except Exception:
            pass

    def compute_volume_stats(self) -> None:
        panel = getattr(self, "measure_panel", None)
        if panel is None:
            return

        obj = self.viewport.selected_obj
        if obj is None:
            QMessageBox.warning(self, "경고", "선택된 메쉬가 없습니다.")
            return

        mesh = getattr(obj, "mesh", None)
        if mesh is None:
            QMessageBox.warning(self, "경고", "선택된 객체에 메쉬 데이터가 없습니다.")
            return

        unit = str(getattr(mesh, "unit", "cm") or "cm").strip().lower()
        scale = float(getattr(obj, "scale", 1.0))
        name = str(getattr(obj, "name", "mesh"))

        def task():
            tm = obj.to_trimesh()
            if tm is None:
                raise ValueError("trimesh conversion failed")

            watertight = bool(getattr(tm, "is_watertight", False))

            area0 = float(getattr(mesh, "surface_area", 0.0))
            if not np.isfinite(area0) or area0 < 0.0:
                area0 = float(getattr(tm, "area", 0.0))

            volume0 = None
            if watertight:
                try:
                    volume0 = abs(float(getattr(tm, "volume", 0.0)))
                except Exception:
                    volume0 = None

            hull0 = None
            if not watertight:
                try:
                    vcount = int(getattr(tm, "vertices", np.zeros((0, 3))).shape[0])
                    fcount = int(getattr(tm, "faces", np.zeros((0, 3))).shape[0])
                except Exception:
                    vcount = 0
                    fcount = 0

                # Convex hull volume is a rough upper bound and can be expensive.
                if vcount > 0 and fcount > 0 and vcount <= 200000 and fcount <= 400000:
                    try:
                        hull0 = abs(float(tm.convex_hull.volume))
                    except Exception:
                        hull0 = None

            ext0 = np.asarray(getattr(mesh, "extents", np.zeros(3)), dtype=np.float64)
            v = int(getattr(mesh, "n_vertices", 0))
            f = int(getattr(mesh, "n_faces", 0))
            return {
                "name": name,
                "unit": unit,
                "scale": scale,
                "watertight": watertight,
                "area0": area0,
                "volume0": volume0,
                "hull0": hull0,
                "ext0": ext0,
                "v": v,
                "f": f,
            }

        def on_done(result: Any) -> None:
            if not isinstance(result, dict):
                return

            unit_s = str(result.get("unit") or "cm").strip().lower()
            scale_s = float(result.get("scale", 1.0))

            # Convert to cm-based reporting.
            unit_to_cm = 1.0
            if unit_s == "mm":
                unit_to_cm = 0.1
            elif unit_s == "m":
                unit_to_cm = 100.0

            ext0 = np.asarray(result.get("ext0") or np.zeros(3), dtype=np.float64).reshape(-1)[:3]
            ext_cm = ext0 * float(scale_s) * float(unit_to_cm)
            ext_mm = ext_cm * 10.0

            area0 = float(result.get("area0", 0.0))
            area_cm2 = area0 * (float(scale_s) ** 2) * (float(unit_to_cm) ** 2)
            area_mm2 = area_cm2 * 100.0

            vol0 = result.get("volume0")
            hull0 = result.get("hull0")
            vol_cm3 = None
            hull_cm3 = None
            if vol0 is not None:
                vol_cm3 = float(vol0) * (float(scale_s) ** 3) * (float(unit_to_cm) ** 3)
            if hull0 is not None:
                hull_cm3 = float(hull0) * (float(scale_s) ** 3) * (float(unit_to_cm) ** 3)

            watertight = bool(result.get("watertight", False))
            v = int(result.get("v", 0))
            f = int(result.get("f", 0))
            n = str(result.get("name") or "mesh")

            panel.append_result(f"[Mesh Stats] {n} (V:{v:,}, F:{f:,}, scale:{scale_s:.3f})")
            panel.append_result(
                f"- Size: {ext_cm[0]:.2f}×{ext_cm[1]:.2f}×{ext_cm[2]:.2f} cm "
                f"({ext_mm[0]:.1f}×{ext_mm[1]:.1f}×{ext_mm[2]:.1f} mm)"
            )
            panel.append_result(f"- Surface area: {area_cm2:.2f} cm² ({area_mm2:.0f} mm²)")

            if vol_cm3 is not None:
                panel.append_result(
                    f"- Volume: {vol_cm3:.2f} cm³ ({vol_cm3 * 1000.0:.0f} mm³) (watertight={watertight})"
                )
            else:
                panel.append_result(f"- Volume: (watertight={watertight}) 계산 불가/참고용")
                if hull_cm3 is not None:
                    panel.append_result(
                        f"  - Convex hull (upper bound): {hull_cm3:.2f} cm³ ({hull_cm3 * 1000.0:.0f} mm³)"
                    )

            try:
                self.status_info.setText("📦 부피/면적 계산 완료")
            except Exception:
                pass

        def on_failed(message: str) -> None:
            QMessageBox.critical(self, "오류", self._format_error_message("부피/면적 계산 실패:", message))
            try:
                self.status_info.setText("❌ 부피/면적 계산 실패")
            except Exception:
                pass

        self._start_task(
            title="계산",
            label="부피/면적 계산 중...",
            thread=TaskThread("mesh_stats", task),
            on_done=on_done,
            on_failed=on_failed,
        )

    def on_roi_toggled(self, enabled):
        """2D ROI 모드 토글 핸들러"""
        if enabled:
            try:
                self._disable_measure_mode()
            except Exception:
                pass
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
                fit = [float(b[0][0]), float(b[1][0]), float(b[0][1]), float(b[1][1])]

                cur = None
                try:
                    cur = [float(x) for x in (getattr(self.viewport, "roi_bounds", None) or [])][:4]
                except Exception:
                    cur = None

                need_fit = True
                if cur is not None and len(cur) >= 4 and np.isfinite(np.asarray(cur[:4], dtype=np.float64)).all():
                    try:
                        x1, x2 = float(cur[0]), float(cur[1])
                        y1, y2 = float(cur[2]), float(cur[3])
                        if x1 > x2:
                            x1, x2 = x2, x1
                        if y1 > y2:
                            y1, y2 = y2, y1
                        cur0 = [x1, x2, y1, y2]
                    except Exception:
                        cur0 = None

                    default0 = [-10.0, 10.0, -10.0, 10.0]
                    if cur0 is not None and all(abs(float(cur0[i]) - float(default0[i])) < 1e-8 for i in range(4)):
                        need_fit = True
                    else:
                        # If the current ROI overlaps the mesh bounds, keep it (prevents "reset every time").
                        try:
                            bx1, bx2, by1, by2 = [float(v) for v in fit]
                            overlap_x = not (float(x2) < float(bx1) or float(x1) > float(bx2))
                            overlap_y = not (float(y2) < float(by1) or float(y1) > float(by2))
                            need_fit = not (overlap_x and overlap_y)
                        except Exception:
                            need_fit = True

                if need_fit:
                    self.viewport.roi_bounds = fit
            try:
                self.viewport.schedule_roi_edges_update(0)
            except Exception:
                pass
        else:
            try:
                self.viewport.active_roi_edge = None
                self.viewport.roi_rect_dragging = False
                self.viewport.roi_rect_start = None
                self.viewport._roi_move_dragging = False
                self.viewport._roi_move_last_xy = None
                self.viewport._roi_bounds_changed = False
            except Exception:
                pass
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
        try:
            _LOGGER.info("Extracted silhouette: %s points", len(points))
        except Exception:
            pass

    def on_crosshair_toggled(self, enabled):
        """십자선 모드 토글 핸들러 (Viewport3D와 연동)"""
        if enabled:
            try:
                self._disable_measure_mode()
            except Exception:
                pass
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
        if enabled:
            try:
                self._disable_measure_mode()
            except Exception:
                pass
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
        self._sync_cutline_button_state(bool(getattr(self.viewport, "cut_lines_enabled", False)))

    def _sync_cutline_button_state(self, enabled: bool):
        try:
            self.section_panel.btn_line.blockSignals(True)
            self.section_panel.btn_line.setChecked(bool(enabled))
            self.section_panel.btn_line.setText(
                "✏️ 단면선 그리기 중지" if bool(enabled) else "✏️ 단면선 그리기 시작"
            )
        except Exception:
            pass
        finally:
            try:
                self.section_panel.btn_line.blockSignals(False)
            except Exception:
                pass

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

    def on_roi_section_commit_requested(self):
        """ROI Enter 커밋 요청을 현재 조정 축 기준 ROI 단면 레이어 저장으로 처리."""
        try:
            added = int(self.viewport.save_roi_sections_to_layers())
        except Exception:
            added = 0

        if added <= 0:
            self.status_info.setText("저장할 ROI 단면 레이어가 없습니다.")
            return

        self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
        self.status_info.setText(f"ROI 단면 레이어 {added}개 저장됨")

    def _on_cut_lines_auto_ended(self):
        self._sync_cutline_button_state(False)

    def _slice_debounce_delay_ms(self) -> int:
        """메쉬 크기에 따라 단면 계산 디바운스 시간을 동적으로 조정."""
        try:
            obj = self.viewport.selected_obj
            n_faces = int(getattr(getattr(obj, "mesh", None), "n_faces", 0) or 0)
        except Exception:
            n_faces = 0

        if n_faces >= 3_000_000:
            return 120
        if n_faces >= 1_000_000:
            return 90
        if n_faces >= 300_000:
            return 60
        return 35

    def _capture_current_slice_to_layer(self) -> int:
        """현재 슬라이스를 레이어로 저장하고 UI를 갱신."""
        try:
            added = int(self.viewport.save_current_slice_to_layer())
        except Exception:
            added = 0

        if added <= 0:
            self.status_info.setText("촬영할 단면이 없습니다.")
            return 0

        try:
            self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
        except Exception:
            pass
        self.status_info.setText(f"단면 촬영 완료: 레이어 {added}개 저장")
        return int(added)

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

    def on_slice_scan_requested(self, delta_cm: float):
        """Ctrl+휠 스캔 입력으로 슬라이스 높이를 연속 조절."""
        try:
            delta = float(delta_cm)
        except Exception:
            return
        if abs(delta) <= 1e-9:
            return

        panel = getattr(self, "slice_panel", None)
        if panel is None:
            return

        try:
            if not panel.group.isChecked():
                panel.group.setChecked(True)
        except Exception:
            pass

        try:
            cur = float(panel.spin.value())
            lo = float(panel.spin.minimum())
            hi = float(panel.spin.maximum())
        except Exception:
            return

        nxt = float(np.clip(cur + delta, lo, hi))
        if np.isclose(nxt, cur, atol=1e-9):
            return
        try:
            panel.spin.setValue(nxt)
        except Exception:
            return
        try:
            self.status_info.setText(f"단면 스캔 Z={nxt:.2f}cm (Ctrl+휠)")
        except Exception:
            pass

    def on_slice_capture_requested(self, height: float):
        """현재 단면 촬영(레이어 저장) 요청."""
        obj = self.viewport.selected_obj
        if obj is None or getattr(obj, "mesh", None) is None:
            QMessageBox.warning(self, "경고", "촬영할 대상 메쉬가 없습니다.")
            return
        panel = getattr(self, "slice_panel", None)
        if panel is None:
            return

        try:
            target_z = float(height)
        except Exception:
            target_z = float(getattr(self.viewport, "slice_z", 0.0) or 0.0)

        try:
            if not panel.group.isChecked():
                panel.group.setChecked(True)
        except Exception:
            pass

        try:
            cur_z = float(getattr(self.viewport, "slice_z", 0.0) or 0.0)
            if not np.isclose(cur_z, target_z, atol=1e-9):
                panel.spin.setValue(target_z)
        except Exception:
            pass

        # 즉시 저장 가능하면 바로 촬영
        thread = getattr(self, "_slice_compute_thread", None)
        has_live_contours = bool(getattr(self.viewport, "slice_contours", None))
        if has_live_contours and (thread is None or not thread.isRunning()) and self._slice_pending_height is None:
            self._slice_capture_pending = False
            self._capture_current_slice_to_layer()
            return

        # 계산 후 자동 촬영 큐
        self._slice_capture_pending = True
        self._slice_pending_height = float(getattr(self.viewport, "slice_z", target_z) or target_z)
        self._slice_debounce_timer.start(1)
        try:
            self.status_info.setText("단면 계산 완료 후 자동 촬영합니다...")
        except Exception:
            pass

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
        if self._slice_capture_pending:
            self._slice_capture_pending = False
            self._capture_current_slice_to_layer()

    def _on_slice_compute_failed(self, z_height: float, message: str):
        if not getattr(self.viewport, "slice_enabled", False):
            return
        self.viewport.slice_contours = []
        self.viewport.update()
        self._slice_capture_pending = False
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
            self._slice_debounce_timer.start(self._slice_debounce_delay_ms())
            return

        self._slice_pending_height = None
        self._slice_capture_pending = False
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
                from scipy.spatial.transform import Rotation as R
                slicer = MeshSlicer(obj.mesh)

                inv_rot = R.from_euler('xyz', obj.rotation, degrees=True).inv().as_matrix()
                inv_scale = 1.0 / obj.scale if obj.scale != 0 else 1.0

                world_origin = np.array([0.0, 0.0, float(height)], dtype=np.float64)
                world_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                translation = np.asarray(obj.translation, dtype=np.float64).reshape(3,)

                local_origin = inv_scale * (inv_rot @ (world_origin - translation))
                local_normal = inv_rot @ world_normal

                contours_local = slicer.slice_with_plane(local_origin, local_normal)
                if not contours_local:
                    QMessageBox.warning(self, "경고", f"Z={height:.2f} 높이에서 단면을 찾을 수 없습니다.")
                    return

                rot = R.from_euler('xyz', obj.rotation, degrees=True).as_matrix()
                scale = float(obj.scale)
                t = translation.reshape(1, 3)

                contours_world: list[np.ndarray] = []
                for contour in contours_local:
                    arr = np.asarray(contour, dtype=np.float64)
                    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
                        continue
                    world_pts = (rot @ (arr[:, :3] * scale).T).T + t
                    contours_world.append(world_pts)

                if not contours_world:
                    QMessageBox.warning(self, "경고", "유효한 단면 폴리라인이 없습니다.")
                    return

                saved = slicer.export_contours_svg(
                    contours_world,
                    file_path,
                    unit=getattr(obj.mesh, "unit", None),
                    stroke_color="#FF0000",
                    stroke_width=0.1,
                    grid_spacing_cm=1.0,
                    mesh_unit=getattr(obj.mesh, "unit", None),
                    title=f"Cross Section at Z={float(height):.2f}",
                    desc=f"Scale: 1:1 (mesh unit: {getattr(obj.mesh, 'unit', 'mm')})",
                )
                if not saved:
                    QMessageBox.warning(self, "경고", "SVG 저장에 실패했습니다.")
                    return

                QMessageBox.information(self, "성공", f"단면 SVG가 저장되었습니다:\n{file_path}")

            except Exception as e:
                QMessageBox.critical(self, "오류", f"SVG 저장 중 오류 발생: {e}")

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

        # Optional: open project passed via CLI (`python main.py --open-project foo.amr`)
        try:
            if "--open-project" in sys.argv:
                i = sys.argv.index("--open-project")
                if i + 1 < len(sys.argv):
                    p = str(sys.argv[i + 1])
                    if p:
                        window.open_project_path(p)
        except Exception:
            _LOGGER.exception("Failed to auto-open project from CLI args")
        
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
