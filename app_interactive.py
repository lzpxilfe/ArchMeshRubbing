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
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QDockWidget, QTreeWidget,
    QTreeWidgetItem, QGroupBox, QDoubleSpinBox, QFormLayout,
    QSlider, QSpinBox, QStatusBar, QToolBar, QFrame,
    QMessageBox, QTextEdit, QProgressBar, QComboBox,
    QCheckBox, QScrollArea, QSizePolicy, QButtonGroup, QDialog, QLineEdit,
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
DEFAULT_PROJECT_FILENAME = "project.amr"
MIN_EXPORT_WIDTH_PX = 800
MAX_EXPORT_WIDTH_PX = 12000
VIEW_ANGLE_EPS = 1e-6
VIEW_CANONICAL_AZIMUTHS = (-180.0, -90.0, 0.0, 90.0, 180.0)
VIEW_DISTANCE_SCALE = 1.35
VIEW_MIN_DIM = 10.0
VIEW_ORTHO_SCALE_TOP_BOTTOM = 0.95
VIEW_ORTHO_SCALE_SIDE = 1.05
FLOOR_ALIGN_AXIS_Z = 2
FLOOR_OPTIMIZE_STEP_DEGREES = (1.2, 0.4, 0.15, 0.05)
CANONICAL_VIEW_PRESETS: dict[str, tuple[float, float]] = {
    "front": (-90.0, 0.0),
    "back": (90.0, 0.0),
    "right": (0.0, 0.0),
    "left": (180.0, 0.0),
    "top": (0.0, 90.0),
    "bottom": (0.0, -90.0),
}
# 6-view planes must always be one of XY / YZ / ZX.
CANONICAL_VIEW_AXES: dict[str, tuple[int, int]] = {
    "top": (0, 1),     # XY
    "bottom": (0, 1),  # XY
    "left": (1, 2),    # YZ
    "right": (1, 2),   # YZ
    "front": (2, 0),   # ZX
    "back": (2, 0),    # ZX
}
_UNIT_TO_INCHES: dict[str, float] = {
    "mm": 1.0 / 25.4,
    "cm": 1.0 / 2.54,
    "m": 100.0 / 2.54,
}
_EXPORT_SURFACE_TARGET_LABELS: dict[str, str] = {
    "all": "전체 메쉬",
    "selected": "현재 선택",
    "outer": "외면",
    "inner": "내면",
    "migu": "미구",
}


def _normalize_surface_target(value: object) -> str:
    target = str(value or "all").strip().lower()
    return target if target in {"all", "selected", "outer", "inner", "migu"} else "all"


def _surface_target_label(value: object) -> str:
    return _EXPORT_SURFACE_TARGET_LABELS.get(_normalize_surface_target(value), "전체 메쉬")


def _surface_target_face_ids(obj: object, value: object) -> np.ndarray:
    target = _normalize_surface_target(value)
    if target == "all" or obj is None:
        return np.zeros((0,), dtype=np.int32)

    if target == "selected":
        source = getattr(obj, "selected_faces", set()) or set()
    else:
        source = getattr(obj, f"{target}_face_indices", set()) or set()

    try:
        ids = np.asarray(sorted(int(x) for x in source), dtype=np.int32).reshape(-1)
    except Exception:
        ids = np.zeros((0,), dtype=np.int32)
    return ids


def _face_index_signature(face_ids: np.ndarray) -> tuple[object, ...] | None:
    ids = np.asarray(face_ids, dtype=np.int32).reshape(-1)
    if ids.size <= 0:
        return (0, "empty")
    digest = hashlib.sha1(ids.tobytes()).hexdigest()[:12]
    return (int(ids.size), digest)


def _canonical_view_key_from_angles(azimuth: float, elevation: float) -> str | None:
    az = ((float(azimuth) + 180.0) % 360.0) - 180.0
    el = float(elevation)
    if abs(el - 90.0) <= VIEW_ANGLE_EPS:
        return "top"
    if abs(el + 90.0) <= VIEW_ANGLE_EPS:
        return "bottom"
    if abs(el) > VIEW_ANGLE_EPS:
        return None
    if abs(az - 0.0) <= VIEW_ANGLE_EPS:
        return "right"
    if abs(abs(az) - 180.0) <= VIEW_ANGLE_EPS:
        return "left"
    if abs(az + 90.0) <= VIEW_ANGLE_EPS:
        return "front"
    if abs(az - 90.0) <= VIEW_ANGLE_EPS:
        return "back"
    return None


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
from src.core.runtime_defaults import DEFAULTS as RUNTIME_DEFAULTS  # noqa: E402
from src.gui.profile_graph_widget import ProfileGraphWidget  # noqa: E402
from src.core.alignment_utils import (  # noqa: E402
    compute_minimax_center_shift,
    compute_nonpenetration_lift,
    fit_plane_normal,
    orient_plane_normal_toward,
    rotation_matrix_align_vectors,
)
from src.core.unit_utils import DEFAULT_MESH_UNIT, mm_to_mesh_units  # noqa: E402
from src.core.tile_form_model import (  # noqa: E402
    AxisHint,
    AxisSource,
    MandrelFitResult,
    SectionObservation,
    SplitScheme,
    TileClass,
    TileInterpretationState,
)
from src.core.tile_synthetic import (  # noqa: E402
    SyntheticTileArtifact,
    SyntheticBenchmarkSuiteReport,
    SyntheticTileGroundTruth,
    SyntheticTileSpec,
    TileEvaluationReport,
    evaluate_tile_interpretation,
    generate_synthetic_tile,
    save_synthetic_benchmark_suite,
    save_synthetic_tile_bundle,
    synthetic_tile_spec_from_preset,
)
from src.core.tile_profile_fitting import fit_circle_2d  # noqa: E402

DEFAULT_EXPORT_DPI = RUNTIME_DEFAULTS.export_dpi


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
            <h3 style="margin:0; color:#2c5282;">🧭 기본 흐름</h3>
            <p style="font-size:11px;">
            <b>1. 정위치</b> → <b>2. 실측용 도면</b> → <b>3. 탁본</b> → <b>4. 제원측정</b><br><br>
            메쉬 체계에서는 정위치, 단면, 외곽, 제원측정을 다루고, 기록면 체계에서는 탁본과 기록면 도면을 만듭니다.<br><br>
            <b>조작</b><br>
            좌클릭 드래그: 회전 / 우클릭 드래그: 이동 / 스크롤: 확대·축소<br>
            1~6: 정면·후면·우측·좌측·상면·하면 / F: 메쉬 맞춤 / R: 뷰 초기화
            </p>
        """)

    def set_transform_help(self):
        self.setHtml("""
            <h3 style="margin:0; color:#2c5282;">📐 정위치 (Positioning)</h3>
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
            <h3 style="margin:0; color:#2c5282;">🗺️ 기록면 전개 설정</h3>
            <p style="font-size:11px;">
            이 단계는 삼각형 면을 따로 터뜨리는 것이 아니라, 기록할 표면을 연속된 좌표계로 전개하기 위한 설정입니다.<br>
            <b>기록면 미리보기:</b> 전개 결과를 연속 탁본 이미지로 바로 확인<br><br>
            기본 출력은 <b>탁본 이미지 + 외곽선</b> 중심이며, 와이어프레임은 기본적으로 사용하지 않습니다.<br>
            곡률 측정, 고급 옵션, 표면 라벨링은 <b>보정/실험 도구</b>로 숨겨져 있습니다.
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
            <h3 style="margin:0; color:#2c5282;">✋ 기록할 표면 선택</h3>
            <p style="font-size:11px;">
            먼저 기록할 표면 패치를 고르는 도구입니다.<br>
            권장 흐름은 <b>표준 시점 버튼 → 가시면 선택 → 현재 선택으로 전개/탁본 저장</b> 입니다.<br><br>

            <b>👁️ 가시면 선택</b><br>
            - <b>현재 시점 가시면</b>: 지금 카메라에서 실제로 보이는 면만 선택<br>
            - <b>상면/하면/정면/후면/좌측/우측</b>: 표준 시점으로 맞춘 뒤 그 시점의 가시면 선택<br><br>

            외면/내면/미구 라벨링은 기본 흐름이 아니라 <b>연구용 표면 라벨링</b>입니다. 필요한 경우에만 별도로 펼쳐 사용하세요.<br><br>

            <b>🧲 경계(면적+자석)</b><br>
            - <b>좌클릭:</b> 점 추가(자석 스냅) / <b>드래그:</b> 카메라 회전<br>
            - <b>첫 점 근처 클릭</b> 또는 <b>우클릭/Enter</b>: 확정<br>
            - <b>Backspace</b>: 되돌리기 / <b>Alt</b>: 제거 모드<br>
            - <b>Shift/Ctrl</b>: 완드 정제 / <b>[ / ]</b>: 자석 반경 / <b>ESC</b>: 종료<br>
            </p>
        """)

    def set_tile_help(self):
        self.setHtml("""
            <h3 style="margin:0; color:#2c5282;">🏺 실측용 도면 / 기와 제작 추정</h3>
            <p style="font-size:11px;">
            기와를 단순 곡면이 아니라 제작 과정을 가진 유물로 읽어 실측용 도면을 만들기 위한 단계입니다.<br>
            <b>기본 실측 흐름:</b> 유형/분할 가설 → 길이축 힌트 → 대표 단면 후보 → 와통 피팅<br>
            <b>탁본 준비:</b> 메인 4축 작업 흐름의 탁본 축에서 상면/하면 기록 준비로 진행<br><br>

            이 패널은 먼저 <b>핵심 실측 단계</b>만 보여주고, 기록면 보조·작업 슬롯·synthetic benchmark 같은 도구는
            <b>연구/검증 도구 보기</b>에서 펼치도록 정리했습니다.
            </p>
        """)

    def set_workflow_help(self):
        self.setHtml("""
            <h3 style="margin:0; color:#2c5282;">🧭 작업 흐름</h3>
            <p style="font-size:11px;">
            기본 화면은 고고학 실무의 핵심 4축만 남겼습니다.<br>
            <b>1. 정위치</b> → 유물을 도면 기준에 맞게 두고 시점을 정합니다.<br>
            <b>2. 실측용 도면</b> → 제작 가설, 단면, 외곽을 정리해 도면을 만듭니다.<br>
            <b>3. 탁본</b> → 상면/하면 기록면을 준비하고 검토 시트를 만듭니다.<br>
            <b>4. 제원측정</b> → 거리, 지름, 면적, 부피 같은 수치를 확인합니다.<br><br>
            이 앱은 <b>메쉬 체계</b>(정위치, 실측용 도면, 제원측정)와 <b>기록면 체계</b>(탁본)를 함께 다룹니다.
            보조 보정 도구는 필요할 때만 따로 여세요.
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
    layerSelected = pyqtSignal(int, int)  # object_idx, layer_idx
    
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
            else:
                self.selectionChanged.emit(obj_idx)
                self.layerSelected.emit(obj_idx, layer_idx)

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


class WorkflowPanel(QWidget):
    """정위치 -> 실측용 도면 -> 탁본 -> 제원측정의 4축 기본 작업 패널"""

    workflowRequested = pyqtSignal(str, object)

    def __init__(self, help_widget: HelpWidget, parent=None):
        super().__init__(parent)
        self.help_widget = help_widget
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        intro = QLabel(
            "기본 화면은 정위치, 실측용 도면, 탁본, 제원측정의 4축만 남겼습니다. "
            "메쉬 체계와 기록면 체계를 오갈 때만 세부 도구를 여세요."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 11px; color: #4a5568;")
        layout.addWidget(intro)

        self.label_object_summary = QLabel("현재 메쉬가 없습니다.")
        self.label_object_summary.setWordWrap(True)
        self.label_object_summary.setStyleSheet("color: #2c5282; font-weight: bold;")
        layout.addWidget(self.label_object_summary)

        self.label_system_summary = QLabel("정위치와 실측 체계가 아직 시작되지 않았습니다.")
        self.label_system_summary.setWordWrap(True)
        self.label_system_summary.setStyleSheet("font-size: 11px; color: #4a5568;")
        layout.addWidget(self.label_system_summary)

        self.label_interpret_summary = QLabel("실측용 도면 준비가 아직 시작되지 않았습니다.")
        self.label_interpret_summary.setWordWrap(True)
        self.label_interpret_summary.setStyleSheet("font-size: 11px; color: #4a5568;")
        layout.addWidget(self.label_interpret_summary)

        self.label_record_summary = QLabel("탁본이 아직 시작되지 않았습니다.")
        self.label_record_summary.setWordWrap(True)
        self.label_record_summary.setStyleSheet("font-size: 11px; color: #4a5568;")
        layout.addWidget(self.label_record_summary)

        self.label_measure_summary = QLabel("제원측정은 필요할 때만 실행하면 됩니다.")
        self.label_measure_summary.setWordWrap(True)
        self.label_measure_summary.setStyleSheet("font-size: 11px; color: #4a5568;")
        layout.addWidget(self.label_measure_summary)

        self.label_next_summary = QLabel("다음 단계: 메쉬를 열고 기준 시점을 맞추세요.")
        self.label_next_summary.setWordWrap(True)
        self.label_next_summary.setStyleSheet("font-size: 11px; color: #4a5568;")
        layout.addWidget(self.label_next_summary)

        align_group = QGroupBox("1. 정위치")
        align_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        align_layout = QVBoxLayout(align_group)
        btn_open_mesh = QPushButton("메쉬 열기")
        btn_open_mesh.clicked.connect(lambda: self.workflowRequested.emit("open_mesh", None))
        align_layout.addWidget(btn_open_mesh)
        btn_open_project = QPushButton("프로젝트 열기")
        btn_open_project.clicked.connect(lambda: self.workflowRequested.emit("open_project", None))
        align_layout.addWidget(btn_open_project)
        btn_fit = QPushButton("메쉬에 맞춤")
        btn_fit.clicked.connect(lambda: self.workflowRequested.emit("fit_view", None))
        align_layout.addWidget(btn_fit)
        view_grid = QGridLayout()
        views = [
            ("상면", "top"), ("정면", "front"), ("우측", "right"),
            ("하면", "bottom"), ("후면", "back"), ("좌측", "left"),
        ]
        for idx, (label, key) in enumerate(views):
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _checked=False, view_key=key: self.workflowRequested.emit("canonical_view", {"view": view_key})
            )
            view_grid.addWidget(btn, idx // 3, idx % 3)
        align_layout.addLayout(view_grid)
        layout.addWidget(align_group)

        interpret_group = QGroupBox("2. 실측용 도면")
        interpret_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        interpret_layout = QVBoxLayout(interpret_group)
        self.progress_interpret = QProgressBar()
        self.progress_interpret.setRange(0, 100)
        self.progress_interpret.setValue(0)
        interpret_layout.addWidget(self.progress_interpret)
        self.btn_interpret_next = QPushButton("다음 실측 단계 실행")
        self.btn_interpret_next.clicked.connect(lambda: self.workflowRequested.emit("run_interpretation_next", None))
        self.btn_interpret_next.setEnabled(False)
        interpret_layout.addWidget(self.btn_interpret_next)

        btn_drawing_svg = QPushButton("실측용 SVG 저장")
        btn_drawing_svg.clicked.connect(lambda: self.workflowRequested.emit("export_flat_svg", None))
        interpret_layout.addWidget(btn_drawing_svg)

        btn_drawing_package = QPushButton("6방향 도면 패키지 저장")
        btn_drawing_package.clicked.connect(lambda: self.workflowRequested.emit("export_profile_package", None))
        interpret_layout.addWidget(btn_drawing_package)
        layout.addWidget(interpret_group)

        record_group = QGroupBox("3. 탁본")
        record_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        record_layout = QVBoxLayout(record_group)
        btn_record_top = QPushButton("상면 기록 준비")
        btn_record_top.clicked.connect(
            lambda: self.workflowRequested.emit("prepare_record_surface", {"view": "top"})
        )
        record_layout.addWidget(btn_record_top)
        btn_record_bottom = QPushButton("하면 기록 준비")
        btn_record_bottom.clicked.connect(
            lambda: self.workflowRequested.emit("prepare_record_surface", {"view": "bottom"})
        )
        record_layout.addWidget(btn_record_bottom)
        btn_preview = QPushButton("기록면 미리보기")
        btn_preview.clicked.connect(lambda: self.workflowRequested.emit("preview_recording_surface", None))
        record_layout.addWidget(btn_preview)
        btn_export_review = QPushButton("기록면 검토 시트 저장")
        btn_export_review.clicked.connect(lambda: self.workflowRequested.emit("export_review_sheet", None))
        record_layout.addWidget(btn_export_review)
        layout.addWidget(record_group)

        measure_group = QGroupBox("4. 제원측정")
        measure_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        measure_layout = QVBoxLayout(measure_group)
        btn_measure = QPushButton("제원측정 도구 열기")
        btn_measure.clicked.connect(lambda: self.workflowRequested.emit("show_measure_tools", None))
        measure_layout.addWidget(btn_measure)
        layout.addWidget(measure_group)

        btn_advanced = QPushButton("세부 패널 열기")
        btn_advanced.clicked.connect(lambda: self.workflowRequested.emit("show_advanced_panels", None))
        layout.addWidget(btn_advanced)

        layout.addStretch(1)

    def update_state(
        self,
        *,
        has_object: bool,
        object_name: str = "",
        selected_faces: int = 0,
        total_faces: int = 0,
        canonical_view: str = "",
        record_view: str = "",
        tile_summary: str = "",
        wizard_summary: str = "",
        wizard_progress: int = 0,
        wizard_next_label: str = "",
        wizard_next_enabled: bool = False,
    ) -> None:
        if not has_object:
            self.label_object_summary.setText("현재 메쉬가 없습니다.")
            self.label_system_summary.setText("정위치와 실측 체계가 아직 시작되지 않았습니다.")
            self.label_interpret_summary.setText("실측용 도면 준비가 아직 시작되지 않았습니다.")
            self.label_record_summary.setText("탁본이 아직 시작되지 않았습니다.")
            self.label_measure_summary.setText("제원측정은 필요할 때만 실행하면 됩니다.")
            self.label_next_summary.setText("다음 단계: 메쉬를 열고 기준 시점을 맞추세요.")
            self.progress_interpret.setValue(0)
            self.btn_interpret_next.setText("다음 실측 단계 실행")
            self.btn_interpret_next.setEnabled(False)
            return

        self.label_object_summary.setText(
            f"현재 메쉬: {object_name or 'Object'} | 선택 {int(selected_faces):,} / 전체 {int(total_faces):,}면"
        )
        view_label = {
            "top": "상면",
            "bottom": "하면",
            "front": "정면",
            "back": "후면",
            "left": "좌측",
            "right": "우측",
        }.get(str(canonical_view or "").strip().lower(), "자유 시점")
        self.label_system_summary.setText(
            f"정위치: {view_label} 시점 기준 | 실측 체계: 단면/외곽/투영 도면을 다룹니다."
        )
        if str(tile_summary or "").strip():
            self.label_interpret_summary.setText(f"실측용 도면 상태: {tile_summary}")
        else:
            self.label_interpret_summary.setText("실측용 도면 상태: 아직 유형/분할/와통 가설이 정리되지 않았습니다.")

        if str(record_view or "").strip().lower() in {"top", "bottom"}:
            record_label = (
                "상면 기록면 준비됨" if str(record_view).strip().lower() == "top" else "하면 기록면 준비됨"
            )
        elif int(selected_faces) > 0:
            record_label = f"수동 선택 {int(selected_faces):,}면"
        else:
            record_label = "아직 기록면이 준비되지 않았습니다."
        self.label_record_summary.setText(f"탁본 상태: {record_label}")
        self.label_measure_summary.setText(
            f"제원측정: 현재 선택 {int(selected_faces):,}면 | 필요 시 치수 측정 도구를 여세요."
        )
        self.progress_interpret.setValue(max(0, min(100, int(wizard_progress))))
        next_label = str(wizard_next_label or "다음 실측 단계 실행")
        next_label = next_label.replace("다음 단계:", "다음 실측 단계:")
        self.btn_interpret_next.setText(next_label)
        self.btn_interpret_next.setEnabled(bool(wizard_next_enabled))
        self.label_next_summary.setText(
            str(wizard_summary or "다음 단계: 실측용 도면을 정리하고 탁본 기록면을 준비하세요.")
        )

    def enterEvent(self, event):
        self.help_widget.set_workflow_help()
        super().enterEvent(event)


class FlattenPanel(QWidget):
    """기록면 전개 설정 패널 (Phase B)"""
    
    flattenRequested = pyqtSignal(dict)
    previewRequested = pyqtSignal()
    selectionRequested = pyqtSignal(str, object)
    
    def __init__(self, help_widget: HelpWidget, parent=None):
        super().__init__(parent)
        self.help_widget = help_widget
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        intro = QLabel(
            "이 단계는 메쉬를 조각내는 explode가 아니라, 기록할 표면을 연속된 기록면으로 전개하는 설정입니다."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 11px; color: #4a5568;")
        layout.addWidget(intro)

        compact_note = QLabel(
            "기본 도면 생성 흐름에는 전개 방법과 미리보기만 남겨두고, 곡률 측정과 라벨링은 보정/실험 도구로 뒤로 숨겼습니다."
        )
        compact_note.setWordWrap(True)
        compact_note.setStyleSheet("font-size: 11px; color: #4a5568;")
        layout.addWidget(compact_note)
        
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
        self.combo_direction.setToolTip("기록면 전개 시 기준이 되는 길이축/주축")
        curve_layout.addRow("전개 방향:", self.combo_direction)
        
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
        self.curve_group = curve_group
        
        # 기록면 전개 방법
        method_group = QGroupBox("🗺️ 기록면 전개 방법")
        method_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        method_layout = QVBoxLayout(method_group)
        
        self.combo_method = QComboBox()
        self.combo_method.addItems([
            "ARAP (형태 보존)",
            "LSCM (각도 보존)",
            "면적 보존",
            "원통 펼침",
            "단면 기반 펼침 (기와)"
        ])
        self.combo_method.setToolTip("기록면 전개 알고리즘 선택")
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
        self.advanced_options_group = adv_group

        # 고급 표면 라벨링 (외면/내면/미구)
        surface_group = QGroupBox("🏷️ 고급 표면 라벨링 (외면/내면/미구)")
        surface_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        surface_layout = QVBoxLayout(surface_group)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("대상:"))
        self.combo_surface_target = QComboBox()
        self.combo_surface_target.addItems(["🌞 외면", "🌙 내면", "🧩 미구"])
        self.combo_surface_target.setToolTip("라벨링할 표면 그룹 선택")
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
        self.surface_group = surface_group
        auto_hint = QLabel(
            "권장: 먼저 선택 패널에서 현재 시점 가시면을 고른 뒤, 내보내기 패널에서 '현재 선택'으로 기록면 전개/탁본을 저장하세요."
        )
        auto_hint.setStyleSheet("color: #4a5568; font-size: 11px;")
        auto_hint.setWordWrap(True)
        layout.addWidget(auto_hint)

        self.btn_toggle_experimental_tools = QPushButton("보정/실험 도구 보기")
        self.btn_toggle_experimental_tools.setCheckable(True)
        self.btn_toggle_experimental_tools.setToolTip(
            "곡률 측정, 고급 옵션, 표면 라벨링 같은 보정/실험용 설정을 표시합니다."
        )
        self.btn_toggle_experimental_tools.toggled.connect(self._set_experimental_tools_visible)
        layout.addWidget(self.btn_toggle_experimental_tools)
        
        # 실행 버튼
        self.btn_flatten = QPushButton("🚀 기록면 전개 실행")
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

        self.btn_preview = QPushButton("🖼️ 기록면 미리보기")
        self.btn_preview.setToolTip(
            "현재 설정과 대상을 기준으로 기록면 전개 결과를 연속 이미지로 미리 봅니다.\n"
            "기본 미리보기는 와이어프레임이 아닌 탁본형 이미지입니다."
        )
        self.btn_preview.clicked.connect(self.previewRequested.emit)
        layout.addWidget(self.btn_preview)
        
        # 진행 상태
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self._set_experimental_tools_visible(False)
        
        layout.addStretch()

    def _set_experimental_tools_visible(self, visible: bool) -> None:
        groups = [
            getattr(self, "curve_group", None),
            getattr(self, "advanced_options_group", None),
            getattr(self, "surface_group", None),
        ]
        for group in groups:
            if group is None:
                continue
            group.setVisible(bool(visible))
        try:
            self.btn_toggle_experimental_tools.setText(
                "보정/실험 도구 숨기기" if visible else "보정/실험 도구 보기"
            )
        except Exception:
            pass
    
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

        hint_selection = QLabel(
            "권장 작업 순서: 표준 시점 버튼 또는 현재 시점 가시면 선택 → 필요 시 브러시/올가미 보정 → '현재 선택'으로 기록면 전개/탁본 저장"
        )
        hint_selection.setWordWrap(True)
        hint_selection.setStyleSheet("font-size: 11px; color: #4a5568;")
        tool_layout.addWidget(hint_selection)
        
        layout.addWidget(tool_group)
        
        # 고급 자동 라벨링
        auto_group = QGroupBox("🤖 고급 표면 라벨링")
        auto_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        auto_layout = QVBoxLayout(auto_group)
        
        btn_auto_surface = QPushButton("📊 외면/내면 자동 라벨링")
        btn_auto_surface.setToolTip(
            "외면/내면을 자동 추정해 현재 메쉬에 라벨로 저장합니다.\n"
            "권장 기본 흐름은 아니며, 외면/내면 구분이 꼭 필요할 때만 사용하세요.\n"
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
        self.auto_group = auto_group
        
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

        btn_visible = QPushButton("👁️ 현재 시점 가시면")
        btn_visible.setToolTip(
            "현재 카메라에서 실제로 보이는 면만 선택 영역으로 가져옵니다.\n"
            "클릭=교체, Shift/Ctrl=추가, Alt=제거"
        )
        btn_visible.clicked.connect(lambda: self.selectionChanged.emit('select_visible_faces', None))
        edit_layout.addWidget(btn_visible)

        visible_view_grid = QGridLayout()
        visible_view_grid.setHorizontalSpacing(4)
        visible_view_grid.setVerticalSpacing(4)
        visible_views = [
            ("상면", "top"),
            ("하면", "bottom"),
            ("정면", "front"),
            ("후면", "back"),
            ("좌측", "left"),
            ("우측", "right"),
        ]
        for i, (label, view_code) in enumerate(visible_views):
            btn = QPushButton(label)
            btn.setToolTip(
                f"{label} 표준 시점으로 맞춘 뒤, 그 시점에서 실제로 보이는 면만 선택합니다."
            )
            btn.clicked.connect(
                lambda _checked=False, v=view_code: self.selectionChanged.emit(
                    "select_visible_from_view",
                    {"view": v},
                )
            )
            visible_view_grid.addWidget(btn, i // 2, i % 2)
        edit_layout.addLayout(visible_view_grid)
        
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
        self.assign_group = assign_group

        self.btn_toggle_labeling_tools = QPushButton("연구용 표면 라벨링 보기")
        self.btn_toggle_labeling_tools.setCheckable(True)
        self.btn_toggle_labeling_tools.setToolTip(
            "외면/내면/미구 자동 라벨링과 수동 지정 같은 연구용 기능을 표시합니다."
        )
        self.btn_toggle_labeling_tools.toggled.connect(self._set_labeling_tools_visible)
        layout.addWidget(self.btn_toggle_labeling_tools)

        # 선택 정보
        self.label_selection = QLabel("선택된 면: 0개")
        self.label_selection.setStyleSheet("font-weight: bold; color: #2c5282;")
        layout.addWidget(self.label_selection)

        self._set_labeling_tools_visible(False)
        
        layout.addStretch()

    def _set_labeling_tools_visible(self, visible: bool) -> None:
        groups = [getattr(self, "auto_group", None), getattr(self, "assign_group", None)]
        for group in groups:
            if group is None:
                continue
            group.setVisible(bool(visible))
        try:
            self.btn_toggle_labeling_tools.setText(
                "연구용 표면 라벨링 숨기기" if visible else "연구용 표면 라벨링 보기"
            )
        except Exception:
            pass
    
    def update_selection_count(self, count: int):
        self.label_selection.setText(f"선택된 면: {count:,}개")
    
    def enterEvent(self, event):
        self.help_widget.set_selection_help()
        super().enterEvent(event)


class TileInterpretationPanel(QWidget):
    """기와 실측용 도면 추정 패널"""

    interpretationChanged = pyqtSignal(str, object)

    def __init__(self, help_widget: HelpWidget, parent=None):
        super().__init__(parent)
        self.help_widget = help_widget
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        intro = QLabel(
            "실측용 도면 축의 기본 흐름은 제작 가설 -> 길이축 -> 대표 단면 -> 와통 추정입니다. "
            "탁본 준비는 메인 4축 작업 흐름의 탁본 축에서 진행하세요."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 11px; color: #4a5568;")
        layout.addWidget(intro)

        essential_note = QLabel(
            "핵심 실측 단계만 먼저 보입니다. 수동 단면 조정은 '세부 실측 도구 보기', "
            "슬롯과 synthetic benchmark는 '연구/검증 도구 보기'에서 여세요."
        )
        essential_note.setWordWrap(True)
        essential_note.setStyleSheet("font-size: 11px; color: #4a5568;")
        layout.addWidget(essential_note)

        hypo_group = QGroupBox("🏺 제작 가설")
        hypo_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        hypo_layout = QFormLayout(hypo_group)

        self.combo_tile_class = QComboBox()
        self.combo_tile_class.addItem("미상", TileClass.UNKNOWN.value)
        self.combo_tile_class.addItem("수키와", TileClass.SUGKIWA.value)
        self.combo_tile_class.addItem("암키와", TileClass.AMKIWA.value)
        self.combo_tile_class.currentIndexChanged.connect(
            lambda _i: self.interpretationChanged.emit("set_tile_class", self.combo_tile_class.currentData())
        )
        hypo_layout.addRow("유형:", self.combo_tile_class)

        self.combo_split_scheme = QComboBox()
        self.combo_split_scheme.addItem("미상", SplitScheme.UNKNOWN.value)
        self.combo_split_scheme.addItem("2분할", SplitScheme.HALF.value)
        self.combo_split_scheme.addItem("4분할", SplitScheme.QUARTER.value)
        self.combo_split_scheme.currentIndexChanged.connect(
            lambda _i: self.interpretationChanged.emit("set_split_scheme", self.combo_split_scheme.currentData())
        )
        hypo_layout.addRow("분할 가설:", self.combo_split_scheme)

        layout.addWidget(hypo_group)

        axis_group = QGroupBox("📏 길이축 힌트")
        axis_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        axis_layout = QVBoxLayout(axis_group)

        self.label_axis_summary = QLabel("아직 저장된 길이축 힌트가 없습니다.")
        self.label_axis_summary.setWordWrap(True)
        self.label_axis_summary.setStyleSheet("color: #2c5282; font-weight: bold;")
        axis_layout.addWidget(self.label_axis_summary)

        btn_axis_auto = QPushButton("길이축 자동 추정")
        btn_axis_auto.setToolTip("전체 메쉬 기준 장축을 길이축 후보로 저장합니다.")
        btn_axis_auto.clicked.connect(
            lambda: self.interpretationChanged.emit("estimate_axis", {"mode": "mesh"})
        )
        axis_layout.addWidget(btn_axis_auto)

        axis_detail_widget = QWidget()
        axis_detail_layout = QVBoxLayout(axis_detail_widget)
        axis_detail_layout.setContentsMargins(0, 0, 0, 0)
        axis_detail_layout.setSpacing(6)

        axis_btn_row = QHBoxLayout()
        btn_axis_selected = QPushButton("현재 선택에서 추정")
        btn_axis_selected.setToolTip("현재 선택한 표면 패치의 장축을 길이축 후보로 저장합니다.")
        btn_axis_selected.clicked.connect(
            lambda: self.interpretationChanged.emit("estimate_axis", {"mode": "selected"})
        )
        axis_btn_row.addWidget(btn_axis_selected)
        axis_detail_layout.addLayout(axis_btn_row)

        btn_axis_clear = QPushButton("🗑️ 길이축 힌트 초기화")
        btn_axis_clear.clicked.connect(lambda: self.interpretationChanged.emit("clear_axis", None))
        axis_detail_layout.addWidget(btn_axis_clear)
        axis_layout.addWidget(axis_detail_widget)
        self.axis_detail_widget = axis_detail_widget

        layout.addWidget(axis_group)

        section_group = QGroupBox("🧭 대표 단면 후보")
        section_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        section_layout = QVBoxLayout(section_group)

        self.label_section_summary = QLabel("대표 단면 후보가 없습니다.")
        self.label_section_summary.setWordWrap(True)
        self.label_section_summary.setStyleSheet("color: #2c5282; font-weight: bold;")
        section_layout.addWidget(self.label_section_summary)

        btn_section_auto = QPushButton("대표 단면 자동 준비")
        btn_section_auto.setToolTip("길이축을 따라 대표 단면 후보 5개를 자동 제안합니다.")
        btn_section_auto.clicked.connect(
            lambda: self.interpretationChanged.emit("auto_section_candidates", {"mode": "mesh", "count": 5})
        )
        section_layout.addWidget(btn_section_auto)

        btn_section_analyze = QPushButton("단면 프로파일 분석")
        btn_section_analyze.setToolTip("대표 단면 후보 위치에서 실제 단면 프로파일을 추출해 요약값을 저장합니다.")
        btn_section_analyze.clicked.connect(
            lambda: self.interpretationChanged.emit("analyze_section_profiles", {"mode": "selected_preferred"})
        )
        section_layout.addWidget(btn_section_analyze)

        section_detail_widget = QWidget()
        section_detail_layout = QVBoxLayout(section_detail_widget)
        section_detail_layout.setContentsMargins(0, 0, 0, 0)
        section_detail_layout.setSpacing(6)

        btn_section_selected = QPushButton("현재 선택 중심 단면 추가")
        btn_section_selected.setToolTip("현재 선택 패치의 중심 위치를 대표 단면 후보로 추가합니다.")
        btn_section_selected.clicked.connect(
            lambda: self.interpretationChanged.emit("add_section_candidate", {"mode": "selected"})
        )
        section_detail_layout.addWidget(btn_section_selected)

        btn_section_mesh = QPushButton("전체 메쉬 중심 단면 추가")
        btn_section_mesh.setToolTip("전체 메쉬 기준 중심 위치를 대표 단면 후보로 추가합니다.")
        btn_section_mesh.clicked.connect(
            lambda: self.interpretationChanged.emit("add_section_candidate", {"mode": "mesh"})
        )
        section_detail_layout.addWidget(btn_section_mesh)

        accept_row = QHBoxLayout()
        btn_section_accept_all = QPushButton("후보 모두 채택")
        btn_section_accept_all.clicked.connect(
            lambda: self.interpretationChanged.emit("accept_all_sections", None)
        )
        accept_row.addWidget(btn_section_accept_all)

        btn_section_accept_middle = QPushButton("중앙 3개 우선 채택")
        btn_section_accept_middle.setToolTip("대표 단면 후보 중 길이축 중앙에 가까운 3개를 우선 채택합니다.")
        btn_section_accept_middle.clicked.connect(
            lambda: self.interpretationChanged.emit("accept_middle_sections", {"count": 3})
        )
        accept_row.addWidget(btn_section_accept_middle)
        section_detail_layout.addLayout(accept_row)

        btn_section_clear = QPushButton("🗑️ 단면 후보 초기화")
        btn_section_clear.clicked.connect(lambda: self.interpretationChanged.emit("clear_sections", None))
        section_detail_layout.addWidget(btn_section_clear)
        section_layout.addWidget(section_detail_widget)
        self.section_detail_widget = section_detail_widget

        layout.addWidget(section_group)

        fit_group = QGroupBox("🪵 와통 초벌 피팅")
        fit_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        fit_layout = QVBoxLayout(fit_group)

        self.label_mandrel_summary = QLabel("와통 초벌 피팅 결과가 없습니다.")
        self.label_mandrel_summary.setWordWrap(True)
        self.label_mandrel_summary.setStyleSheet("color: #2c5282; font-weight: bold;")
        fit_layout.addWidget(self.label_mandrel_summary)

        btn_fit_selected = QPushButton("와통 추정 실행")
        btn_fit_selected.setToolTip("현재 선택 표면이 있으면 우선 사용하고, 없으면 전체 메쉬로 와통 반경 후보를 추정합니다.")
        btn_fit_selected.clicked.connect(
            lambda: self.interpretationChanged.emit("fit_mandrel", {"mode": "selected_preferred"})
        )
        fit_layout.addWidget(btn_fit_selected)

        fit_detail_widget = QWidget()
        fit_detail_layout = QVBoxLayout(fit_detail_widget)
        fit_detail_layout.setContentsMargins(0, 0, 0, 0)
        fit_detail_layout.setSpacing(6)

        btn_fit_mesh = QPushButton("전체 메쉬로 추정")
        btn_fit_mesh.setToolTip("대표 단면 후보를 이용해 전체 메쉬 기준 와통 반경 후보를 추정합니다.")
        btn_fit_mesh.clicked.connect(
            lambda: self.interpretationChanged.emit("fit_mandrel", {"mode": "mesh"})
        )
        fit_detail_layout.addWidget(btn_fit_mesh)

        btn_fit_clear = QPushButton("🗑️ 피팅 결과 초기화")
        btn_fit_clear.clicked.connect(lambda: self.interpretationChanged.emit("clear_mandrel_fit", None))
        fit_detail_layout.addWidget(btn_fit_clear)
        fit_layout.addWidget(fit_detail_widget)
        self.fit_detail_widget = fit_detail_widget

        layout.addWidget(fit_group)

        self.btn_toggle_interpret_detail_tools = QPushButton("세부 실측 도구 보기")
        self.btn_toggle_interpret_detail_tools.setCheckable(True)
        self.btn_toggle_interpret_detail_tools.setToolTip(
            "수동 단면 추가, 후보 채택 조정, 길이축 초기화 같은 세부 실측 도구를 표시합니다."
        )
        self.btn_toggle_interpret_detail_tools.toggled.connect(self._set_interpret_detail_tools_visible)
        layout.addWidget(self.btn_toggle_interpret_detail_tools)

        record_group = QGroupBox("🧾 탁본 기록면 보조")
        record_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        record_layout = QVBoxLayout(record_group)

        self.label_record_summary = QLabel("아직 준비된 기록면이 없습니다.")
        self.label_record_summary.setWordWrap(True)
        self.label_record_summary.setStyleSheet("color: #2c5282; font-weight: bold;")
        record_layout.addWidget(self.label_record_summary)

        btn_record_top = QPushButton("상면 기록 준비")
        btn_record_top.setToolTip("상면 표준 시점으로 맞춘 뒤, 그 시점에서 보이는 기록면을 자동 준비합니다.")
        btn_record_top.clicked.connect(
            lambda: self.interpretationChanged.emit("prepare_record_surface", {"view": "top"})
        )
        record_layout.addWidget(btn_record_top)

        btn_record_bottom = QPushButton("하면 기록 준비")
        btn_record_bottom.setToolTip("하면 표준 시점으로 맞춘 뒤, 그 시점에서 보이는 기록면을 자동 준비합니다.")
        btn_record_bottom.clicked.connect(
            lambda: self.interpretationChanged.emit("prepare_record_surface", {"view": "bottom"})
        )
        record_layout.addWidget(btn_record_bottom)

        btn_record_clear = QPushButton("🗑️ 기록면 준비 해제")
        btn_record_clear.clicked.connect(lambda: self.interpretationChanged.emit("clear_record_surface", None))
        record_layout.addWidget(btn_record_clear)

        record_note = QLabel(
            "기와 모드에서는 사용자가 면을 먼저 고르지 않아도 됩니다. 상면/하면을 고르면 앱이 내부적으로 현재 선택을 준비합니다."
        )
        record_note.setWordWrap(True)
        record_note.setStyleSheet("font-size: 11px; color: #4a5568;")
        record_layout.addWidget(record_note)

        layout.addWidget(record_group)
        self.record_group = record_group

        slot_group = QGroupBox("💾 작업 슬롯")
        slot_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        slot_layout = QVBoxLayout(slot_group)

        self.label_slot_summary = QLabel("저장된 작업 슬롯이 없습니다.")
        self.label_slot_summary.setWordWrap(True)
        self.label_slot_summary.setStyleSheet("color: #2c5282; font-weight: bold;")
        slot_layout.addWidget(self.label_slot_summary)

        self._slot_save_buttons: dict[int, QPushButton] = {}
        self._slot_load_buttons: dict[int, QPushButton] = {}
        self._slot_info_labels: dict[int, QLabel] = {}

        for slot_index in range(1, 4):
            row = QHBoxLayout()
            btn_save_slot = QPushButton(f"슬롯 {slot_index} 저장")
            btn_save_slot.setToolTip("현재 선택과 기와 해석 가설 상태를 이 슬롯에 저장합니다.")
            btn_save_slot.clicked.connect(
                lambda _checked=False, idx=slot_index: self.interpretationChanged.emit(
                    "save_slot", {"slot": idx}
                )
            )
            row.addWidget(btn_save_slot)

            btn_load_slot = QPushButton("불러오기")
            btn_load_slot.setToolTip("이 슬롯에 저장된 선택/가설 상태를 복원합니다.")
            btn_load_slot.clicked.connect(
                lambda _checked=False, idx=slot_index: self.interpretationChanged.emit(
                    "load_slot", {"slot": idx}
                )
            )
            row.addWidget(btn_load_slot)
            slot_layout.addLayout(row)

            info_label = QLabel(f"슬롯 {slot_index}: 비어 있음")
            info_label.setWordWrap(True)
            info_label.setStyleSheet("font-size: 11px; color: #4a5568; margin-left: 4px;")
            slot_layout.addWidget(info_label)

            self._slot_save_buttons[slot_index] = btn_save_slot
            self._slot_load_buttons[slot_index] = btn_load_slot
            self._slot_info_labels[slot_index] = info_label

        btn_clear_slots = QPushButton("🗑️ 작업 슬롯 모두 비우기")
        btn_clear_slots.clicked.connect(lambda: self.interpretationChanged.emit("clear_slots", None))
        slot_layout.addWidget(btn_clear_slots)
        self.btn_clear_slots = btn_clear_slots

        btn_export_slots = QPushButton("📦 저장 슬롯 검토 시트 묶음 저장")
        btn_export_slots.setToolTip("저장된 슬롯별로 기록면 검토 시트를 한 번에 생성합니다.")
        btn_export_slots.clicked.connect(
            lambda: self.interpretationChanged.emit("export_saved_slots_review", None)
        )
        slot_layout.addWidget(btn_export_slots)
        self.btn_export_slots = btn_export_slots

        layout.addWidget(slot_group)
        self.slot_group = slot_group

        wizard_group = QGroupBox("🪄 기와 실측 위저드")
        wizard_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        wizard_layout = QVBoxLayout(wizard_group)

        self.label_wizard_summary = QLabel("위저드가 아직 시작되지 않았습니다.")
        self.label_wizard_summary.setWordWrap(True)
        self.label_wizard_summary.setStyleSheet("color: #2c5282; font-weight: bold;")
        wizard_layout.addWidget(self.label_wizard_summary)

        self.progress_wizard = QProgressBar()
        self.progress_wizard.setRange(0, 100)
        self.progress_wizard.setValue(0)
        wizard_layout.addWidget(self.progress_wizard)

        self.btn_wizard_next = QPushButton("다음 단계 실행")
        self.btn_wizard_next.clicked.connect(lambda: self.interpretationChanged.emit("run_wizard_next", None))
        wizard_layout.addWidget(self.btn_wizard_next)

        self.btn_wizard_run_all = QPushButton("남은 단계 자동 실행")
        self.btn_wizard_run_all.clicked.connect(lambda: self.interpretationChanged.emit("run_wizard_all", None))
        wizard_layout.addWidget(self.btn_wizard_run_all)

        layout.addWidget(wizard_group)
        self.wizard_group = wizard_group

        synth_group = QGroupBox("🧪 합성 데이터 / 정답 평가")
        synth_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        synth_layout = QVBoxLayout(synth_group)

        preset_row = QHBoxLayout()
        self.combo_synthetic_preset = QComboBox()
        self.combo_synthetic_preset.addItem("수키와 · 4분할", "sugkiwa_quarter")
        self.combo_synthetic_preset.addItem("수키와 · 2분할", "sugkiwa_half")
        self.combo_synthetic_preset.addItem("암키와 · 4분할", "amkiwa_quarter")
        self.combo_synthetic_preset.addItem("암키와 · 2분할", "amkiwa_half")
        preset_row.addWidget(self.combo_synthetic_preset, 1)

        self.spin_synthetic_seed = QSpinBox()
        self.spin_synthetic_seed.setRange(0, 999999)
        self.spin_synthetic_seed.setValue(1)
        self.spin_synthetic_seed.setPrefix("seed ")
        preset_row.addWidget(self.spin_synthetic_seed)
        synth_layout.addLayout(preset_row)

        btn_generate_synthetic = QPushButton("합성 기와 생성")
        btn_generate_synthetic.clicked.connect(
            lambda: self.interpretationChanged.emit(
                "generate_synthetic_tile",
                {
                    "preset": self.combo_synthetic_preset.currentData(),
                    "seed": int(self.spin_synthetic_seed.value()),
                },
            )
        )
        synth_layout.addWidget(btn_generate_synthetic)

        btn_evaluate_truth = QPushButton("정답 평가 실행")
        btn_evaluate_truth.clicked.connect(lambda: self.interpretationChanged.emit("evaluate_against_truth", None))
        synth_layout.addWidget(btn_evaluate_truth)
        self.btn_evaluate_truth = btn_evaluate_truth

        btn_apply_truth = QPushButton("정답 가설 적용")
        btn_apply_truth.setToolTip("합성 정답이 연결된 경우, 정답 상태를 현재 해석 상태로 복원합니다.")
        btn_apply_truth.clicked.connect(lambda: self.interpretationChanged.emit("apply_synthetic_truth_hypothesis", None))
        synth_layout.addWidget(btn_apply_truth)
        self.btn_apply_truth = btn_apply_truth

        btn_export_bundle = QPushButton("합성 벤치마크 묶음 저장")
        btn_export_bundle.setToolTip("메쉬, 정답, 현재 해석, 평가 결과를 한 묶음으로 저장합니다.")
        btn_export_bundle.clicked.connect(lambda: self.interpretationChanged.emit("export_synthetic_bundle", None))
        synth_layout.addWidget(btn_export_bundle)
        self.btn_export_synthetic_bundle = btn_export_bundle

        suite_row = QHBoxLayout()
        self.edit_synthetic_suite_seeds = QLineEdit("1,2,3")
        self.edit_synthetic_suite_seeds.setPlaceholderText("seed 목록 / Seeds, e.g. 1,2,3")
        self.edit_synthetic_suite_seeds.setToolTip(
            "모든 preset에 대해 생성할 synthetic benchmark seed 목록 / Seeds for every preset in the benchmark suite"
        )
        suite_row.addWidget(self.edit_synthetic_suite_seeds, 1)

        self.spin_synthetic_pass_threshold = QDoubleSpinBox()
        self.spin_synthetic_pass_threshold.setRange(0.0, 1.0)
        self.spin_synthetic_pass_threshold.setDecimals(2)
        self.spin_synthetic_pass_threshold.setSingleStep(0.05)
        self.spin_synthetic_pass_threshold.setValue(0.90)
        self.spin_synthetic_pass_threshold.setPrefix("pass ")
        self.spin_synthetic_pass_threshold.setToolTip(
            "합격 기준 점수 / Pass threshold for synthetic benchmark suite"
        )
        suite_row.addWidget(self.spin_synthetic_pass_threshold)

        btn_export_suite = QPushButton("합성 benchmark suite 저장")
        btn_export_suite.setToolTip(
            "모든 기와 preset × seed 목록을 한 번에 생성하고 review 시트까지 저장합니다. / "
            "Generate every preset × seed case and save review sheets together."
        )
        btn_export_suite.clicked.connect(
            lambda: self.interpretationChanged.emit(
                "export_synthetic_benchmark_suite",
                {
                    "seeds": str(self.edit_synthetic_suite_seeds.text() or "1"),
                    "pass_threshold": float(self.spin_synthetic_pass_threshold.value()),
                },
            )
        )
        suite_row.addWidget(btn_export_suite)
        synth_layout.addLayout(suite_row)
        self.btn_export_synthetic_suite = btn_export_suite

        self.label_synthetic_truth = QLabel("선택된 메쉬에 연결된 합성 정답이 없습니다.")
        self.label_synthetic_truth.setWordWrap(True)
        self.label_synthetic_truth.setStyleSheet("font-size: 11px; color: #4a5568;")
        synth_layout.addWidget(self.label_synthetic_truth)

        self.label_evaluation_summary = QLabel("아직 실행된 정답 평가가 없습니다.")
        self.label_evaluation_summary.setWordWrap(True)
        self.label_evaluation_summary.setStyleSheet("font-size: 11px; color: #4a5568;")
        synth_layout.addWidget(self.label_evaluation_summary)

        self.label_synthetic_suite_summary = QLabel(
            "Synthetic benchmark suite: 모든 preset × seed 조합을 생성하고 review 시트까지 함께 저장합니다."
        )
        self.label_synthetic_suite_summary.setWordWrap(True)
        self.label_synthetic_suite_summary.setStyleSheet("font-size: 11px; color: #4a5568;")
        synth_layout.addWidget(self.label_synthetic_suite_summary)

        layout.addWidget(synth_group)
        self.synth_group = synth_group

        self.btn_toggle_research_tools = QPushButton("연구/검증 도구 보기")
        self.btn_toggle_research_tools.setCheckable(True)
        self.btn_toggle_research_tools.setToolTip(
            "기록면 보조, 작업 슬롯, 기와 위저드, synthetic benchmark 같은 연구/검증용 도구를 표시합니다."
        )
        self.btn_toggle_research_tools.toggled.connect(self._set_research_tools_visible)
        layout.addWidget(self.btn_toggle_research_tools)

        self.label_context = QLabel("선택된 메쉬가 없습니다.")
        self.label_context.setWordWrap(True)
        self.label_context.setStyleSheet("font-size: 11px; color: #4a5568;")
        layout.addWidget(self.label_context)

        self.label_workflow = QLabel(
            "다음 단계: 길이축이 잡히면 대표 단면을 골라 와통 기반 제작형 추정을 시작합니다."
        )
        self.label_workflow.setWordWrap(True)
        self.label_workflow.setStyleSheet("font-size: 11px; color: #4a5568;")
        layout.addWidget(self.label_workflow)

        self._set_interpret_detail_tools_visible(False)
        self._set_research_tools_visible(False)

        layout.addStretch()

    def _set_interpret_detail_tools_visible(self, visible: bool) -> None:
        widgets = [
            getattr(self, "axis_detail_widget", None),
            getattr(self, "section_detail_widget", None),
            getattr(self, "fit_detail_widget", None),
        ]
        for widget in widgets:
            if widget is None:
                continue
            widget.setVisible(bool(visible))
        try:
            self.btn_toggle_interpret_detail_tools.setText(
                "세부 실측 도구 숨기기" if visible else "세부 실측 도구 보기"
            )
        except Exception:
            pass

    def _set_research_tools_visible(self, visible: bool) -> None:
        groups = [
            getattr(self, "record_group", None),
            getattr(self, "slot_group", None),
            getattr(self, "wizard_group", None),
            getattr(self, "synth_group", None),
        ]
        for group in groups:
            if group is None:
                continue
            group.setVisible(bool(visible))
        try:
            self.btn_toggle_research_tools.setText(
                "연구/검증 도구 숨기기" if visible else "연구/검증 도구 보기"
            )
        except Exception:
            pass

    def update_state(
        self,
        state: TileInterpretationState | None,
        *,
        object_name: str,
        object_unit: str,
        selected_faces: int,
        total_faces: int,
        wizard_summary: str = "",
        wizard_progress: int = 0,
        wizard_next_label: str = "",
        wizard_next_enabled: bool = False,
        synthetic_truth_summary: str = "",
        evaluation_summary: str = "",
    ) -> None:
        enabled = state is not None

        for widget in (self.combo_tile_class, self.combo_split_scheme):
            widget.blockSignals(True)
        try:
            tile_value = (state.tile_class.value if state is not None else TileClass.UNKNOWN.value)
            split_value = (state.split_scheme.value if state is not None else SplitScheme.UNKNOWN.value)
            tile_index = self.combo_tile_class.findData(tile_value)
            split_index = self.combo_split_scheme.findData(split_value)
            self.combo_tile_class.setCurrentIndex(tile_index if tile_index >= 0 else 0)
            self.combo_split_scheme.setCurrentIndex(split_index if split_index >= 0 else 0)
        finally:
            for widget in (self.combo_tile_class, self.combo_split_scheme):
                widget.blockSignals(False)

        self.combo_tile_class.setEnabled(enabled)
        self.combo_split_scheme.setEnabled(enabled)

        if state is None:
            self.label_axis_summary.setText("아직 저장된 길이축 힌트가 없습니다.")
            self.label_section_summary.setText("대표 단면 후보가 없습니다.")
            self.label_mandrel_summary.setText("와통 초벌 피팅 결과가 없습니다.")
            self.label_record_summary.setText("아직 준비된 기록면이 없습니다.")
            self.label_slot_summary.setText("저장된 작업 슬롯이 없습니다.")
            for slot_index in range(1, 4):
                self._slot_save_buttons[slot_index].setEnabled(False)
                self._slot_load_buttons[slot_index].setEnabled(False)
                self._slot_info_labels[slot_index].setText(f"슬롯 {slot_index}: 비어 있음")
            self.btn_clear_slots.setEnabled(False)
            self.btn_export_slots.setEnabled(False)
            self.label_wizard_summary.setText("위저드가 아직 시작되지 않았습니다.")
            self.progress_wizard.setValue(0)
            self.btn_wizard_next.setText("다음 단계 실행")
            self.btn_wizard_next.setEnabled(False)
            self.btn_wizard_run_all.setEnabled(False)
            self.label_synthetic_truth.setText("선택된 메쉬에 연결된 합성 정답이 없습니다.")
            self.label_evaluation_summary.setText("아직 실행된 정답 평가가 없습니다.")
            self.label_synthetic_suite_summary.setText(
                "Synthetic benchmark suite: 모든 preset × seed 조합을 생성하고 review 시트까지 함께 저장합니다."
            )
            self.btn_evaluate_truth.setEnabled(False)
            self.btn_apply_truth.setEnabled(False)
            self.btn_export_synthetic_bundle.setEnabled(False)
            self.label_context.setText("선택된 메쉬가 없습니다.")
            self.label_workflow.setText(
                "다음 단계: 길이축이 잡히면 대표 단면을 골라 와통 기반 제작형 추정을 시작합니다."
            )
            return

        for slot_index in range(1, 4):
            self._slot_save_buttons[slot_index].setEnabled(True)
        self.btn_clear_slots.setEnabled(True)
        self.btn_wizard_next.setEnabled(bool(wizard_next_enabled))
        self.btn_wizard_next.setText(str(wizard_next_label or "다음 단계 실행"))
        self.btn_wizard_run_all.setEnabled(bool(wizard_next_enabled))
        self.progress_wizard.setValue(max(0, min(100, int(wizard_progress))))
        self.label_wizard_summary.setText(str(wizard_summary or "위저드 단계를 계산하지 못했습니다."))
        has_truth = bool(str(synthetic_truth_summary or "").strip())
        self.label_synthetic_truth.setText(
            str(synthetic_truth_summary or "선택된 메쉬에 연결된 합성 정답이 없습니다.")
        )
        self.label_evaluation_summary.setText(
            str(evaluation_summary or "아직 실행된 정답 평가가 없습니다.")
        )
        self.btn_evaluate_truth.setEnabled(has_truth)
        self.btn_apply_truth.setEnabled(has_truth)
        self.btn_export_synthetic_bundle.setEnabled(has_truth)

        axis_hint = state.axis_hint
        if axis_hint.is_defined():
            vec = axis_hint.vector_world or (0.0, 0.0, 0.0)
            axis_text = (
                f"{axis_hint.source.label_ko} | "
                f"x={vec[0]:+.3f}, y={vec[1]:+.3f}, z={vec[2]:+.3f} | "
                f"신뢰도 {axis_hint.confidence * 100.0:.0f}%"
            )
        else:
            axis_text = "아직 저장된 길이축 힌트가 없습니다."
        self.label_axis_summary.setText(axis_text)

        sections = list(state.section_observations or [])
        if sections:
            accepted = sum(1 for item in sections if bool(item.accepted))
            analyzed = sum(1 for item in sections if int(getattr(item, "profile_point_count", 0) or 0) > 0)
            preview: list[str] = []
            for item in sections[:3]:
                if item.station is None:
                    preview.append("station ?")
                else:
                    preview.append(f"s={float(item.station):+.2f}")
            suffix = " / ".join(preview)
            if len(sections) > 3:
                suffix += " / ..."
            self.label_section_summary.setText(
                f"후보 {len(sections)}개 (채택 {accepted}개, 분석 {analyzed}개) | {suffix}"
            )
            self.label_workflow.setText(
                "다음 단계: 대표 단면 후보를 검토한 뒤, 보존 상태가 좋은 단면부터 와통 피팅에 사용합니다."
            )
        else:
            self.label_section_summary.setText("대표 단면 후보가 없습니다.")
            self.label_workflow.setText(
                "다음 단계: 길이축이 잡히면 대표 단면을 골라 와통 기반 제작형 추정을 시작합니다."
            )

        fit_result = state.mandrel_fit
        if fit_result.is_defined():
            self.label_mandrel_summary.setText(
                f"R={float(fit_result.radius_world):.3f} {object_unit or 'unit'} | "
                f"spread {float(fit_result.radius_spread_world):.3f} | "
                f"후보 {int(fit_result.used_sections)}개 | "
                f"신뢰도 {float(fit_result.confidence) * 100.0:.0f}%"
            )
            self.label_workflow.setText(
                "다음 단계: 초벌 반경 후보를 기준으로 대표 단면을 검토하고, 공통 와통 형상으로 보정합니다."
            )
        else:
            self.label_mandrel_summary.setText("와통 초벌 피팅 결과가 없습니다.")

        record_view = str(getattr(state, "record_view", "") or "").strip().lower()
        if record_view in {"top", "bottom"}:
            label = "상면" if record_view == "top" else "하면"
            self.label_record_summary.setText(
                f"{label} 기록면 준비됨 | 방식: {str(getattr(state, 'record_strategy', '') or 'auto')}"
            )
            self.label_workflow.setText(
                f"다음 단계: {label} 기록면이 준비되어 있습니다. 바로 전개/탁본 내보내기를 실행할 수 있습니다."
            )
        else:
            self.label_record_summary.setText("아직 준비된 기록면이 없습니다.")

        slot_items = {str(getattr(item, "slot_key", "") or ""): item for item in list(getattr(state, "saved_slots", []) or [])}
        filled_slots = 0
        for slot_index in range(1, 4):
            slot = slot_items.get(f"slot_{slot_index}")
            load_button = self._slot_load_buttons[slot_index]
            info_label = self._slot_info_labels[slot_index]
            if slot is None:
                load_button.setEnabled(False)
                info_label.setText(f"슬롯 {slot_index}: 비어 있음")
                continue
            filled_slots += 1
            load_button.setEnabled(True)
            updated_at = str(getattr(slot, "updated_at_iso", "") or "").strip()
            suffix = f" | {updated_at}" if updated_at else ""
            info_label.setText(f"슬롯 {slot_index}: {slot.summary_label()}{suffix}")
        self.label_slot_summary.setText(
            f"저장된 작업 슬롯 {filled_slots}개 | 현재 선택 {int(selected_faces):,}면"
        )
        self.btn_export_slots.setEnabled(filled_slots > 0)

        self.label_context.setText(
            f"현재 메쉬: {object_name or 'Object'} | 선택 면 {int(selected_faces):,} / 전체 면 {int(total_faces):,}"
        )

    def enterEvent(self, event):
        self.help_widget.set_tile_help()
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
    """기본 도면 출력 패널"""
    
    exportRequested = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        intro = QLabel(
            "기본 도면 출력에는 검토 시트, 기록면 SVG, 6방향 도면 패키지만 남겼습니다. "
            "실험적이거나 우회적인 출력은 기본 UI에서 제거했습니다."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 11px; color: #4a5568;")
        layout.addWidget(intro)

        img_group = QGroupBox("🧾 기본 출력 설정")
        img_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        img_layout = QFormLayout(img_group)

        self.spin_dpi = QSpinBox()
        self.spin_dpi.setRange(72, 1200)
        self.spin_dpi.setValue(DEFAULT_EXPORT_DPI)
        self.spin_dpi.setSuffix(" DPI")
        self.spin_dpi.setToolTip("권장: 300 / 600 / 1200 PPI")
        img_layout.addRow("해상도:", self.spin_dpi)

        self.combo_format = QComboBox()
        self.combo_format.addItems(["PNG", "TIFF", "JPEG"])

        self.check_scale_bar = QCheckBox("스케일 바 포함")
        self.check_scale_bar.setChecked(True)
        img_layout.addRow("", self.check_scale_bar)

        self.combo_review_render_mode = QComboBox()
        self.combo_review_render_mode.addItem("자동", "auto")
        self.combo_review_render_mode.addItem("다중광(기록면)", "다중광(기록면)")
        self.combo_review_render_mode.addItem("자연(이미지)+CLAHE", "자연(이미지)+CLAHE")
        self.combo_review_render_mode.addItem("로컬 대비(텍스처)", "로컬 대비(텍스처)")
        self.combo_review_render_mode.addItem("노멀 언샵", "노멀 언샵")
        self.combo_review_render_mode.addItem("스펙큘러 강조", "스펙큘러 강조")
        self.combo_review_render_mode.addItem("노멀 보기", "노멀 보기")
        self.combo_review_render_mode.addItem("자연(이미지)", "자연(이미지)")
        self.combo_review_render_mode.setToolTip(
            "검토 시트와 미리보기에서 사용할 기록면 렌더 모드입니다.\n"
            "자동은 기와/기록면일 때 다중광, 일반 경로는 자연(이미지)를 사용합니다.\n"
            "텍스처 보강은 CLAHE/로컬 대비 기반 후처리(실험적) 옵션입니다."
        )
        img_layout.addRow("기록면 렌더:", self.combo_review_render_mode)

        self.spin_texture_detail_scale = QDoubleSpinBox()
        self.spin_texture_detail_scale.setRange(0.25, 3.0)
        self.spin_texture_detail_scale.setSingleStep(0.05)
        self.spin_texture_detail_scale.setDecimals(2)
        self.spin_texture_detail_scale.setValue(1.0)
        self.spin_texture_detail_scale.setSuffix(" x")
        self.spin_texture_detail_scale.setToolTip(
            "선택한 렌더 모드 위에 추가로 질감/요철 강조를 곱해 적용합니다.\n"
            "1.0은 기본, 1보다 크면 더 선명해집니다."
        )
        img_layout.addRow("질감 강조:", self.spin_texture_detail_scale)

        self.spin_texture_smooth_extra = QDoubleSpinBox()
        self.spin_texture_smooth_extra.setRange(0.0, 4.0)
        self.spin_texture_smooth_extra.setSingleStep(0.1)
        self.spin_texture_smooth_extra.setDecimals(2)
        self.spin_texture_smooth_extra.setValue(0.0)
        self.spin_texture_smooth_extra.setSuffix(" px")
        self.spin_texture_smooth_extra.setToolTip(
            "기본 렌더 위에 추가로 적용할 부드럽게 하기 정도입니다.\n"
            "거친 노이즈를 줄이고 싶은 경우에 올립니다."
        )
        img_layout.addRow("추가 스무딩:", self.spin_texture_smooth_extra)

        self.combo_texture_postprocess = QComboBox()
        self.combo_texture_postprocess.addItem("기본 유지", "")
        self.combo_texture_postprocess.addItem("CLAHE 추가", "clahe")
        self.combo_texture_postprocess.addItem("로컬 대비 추가", "local_contrast")
        self.combo_texture_postprocess.addItem("샤프닝 추가", "unsharp")
        self.combo_texture_postprocess.addItem("부드럽게", "soften")
        self.combo_texture_postprocess.addItem("CLAHE + 샤프닝", "clahe,unsharp")
        self.combo_texture_postprocess.addItem("로컬 대비 + 부드럽게", "local_contrast,soften")
        self.combo_texture_postprocess.setToolTip(
            "렌더 모드 프리셋 뒤에 덧붙일 추가 텍스처 보정입니다.\n"
            "기본 유지는 프리셋 자체의 후처리만 사용합니다."
        )
        img_layout.addRow("텍스처 보정:", self.combo_texture_postprocess)

        self.combo_rubbing_target = QComboBox()
        self.combo_rubbing_target.addItems(["전체 메쉬", "✨ 현재 선택"])
        self.combo_rubbing_target.setToolTip(
            "기본 도면 생성에서 사용할 대상을 고릅니다.\n"
            "기본 흐름은 '현재 선택' 또는 기와 모드의 상면/하면 기록 준비 결과를 사용하는 것입니다."
        )
        img_layout.addRow("도면 대상:", self.combo_rubbing_target)

        layout.addWidget(img_group)

        btn_export_review_sheet = QPushButton("📤 기록면 검토 시트 저장")
        btn_export_review_sheet.setToolTip(
            "연속 탁본형 기록면 + 외곽 확인 이미지를 한 장의 검토 시트로 저장합니다.\n"
            "미리보기와 같은 철학의 출력물을 파일로 남길 때 사용합니다."
        )
        btn_export_review_sheet.setStyleSheet("""
            QPushButton {
                background-color: #d69e2e;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #b7791f; }
        """)
        btn_export_review_sheet.clicked.connect(
            lambda: self.exportRequested.emit(
                {'type': 'review_sheet', 'target': self.current_rubbing_target()}
            )
        )
        layout.addWidget(btn_export_review_sheet)

        btn_export_flat_svg = QPushButton("기록면 전개 SVG 저장")
        btn_export_flat_svg.setToolTip(
            "전개 결과의 외곽선을 실측 SVG로 저장합니다.\n"
            "기본 출력은 연속 표면의 외곽선만 포함하며, 와이어프레임은 넣지 않습니다."
        )
        btn_export_flat_svg.clicked.connect(
            lambda: self.exportRequested.emit({'type': 'flat_svg', 'target': self.current_rubbing_target()})
        )
        layout.addWidget(btn_export_flat_svg)

        profile_group = QGroupBox("📦 6방향 도면 패키지")
        profile_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2b6cb0; }")
        profile_layout = QVBoxLayout(profile_group)

        lbl_info = QLabel(
            "Top / Bottom / Front / Back / Left / Right 기준의 2D 실측 도면을 한 폴더에 묶어 저장합니다."
        )
        lbl_info.setStyleSheet("font-size: 11px; color: #718096;")
        lbl_info.setWordWrap(True)
        profile_layout.addWidget(lbl_info)

        opt_row = QHBoxLayout()
        self.check_profile_include_grid = QCheckBox("격자/배경 포함 (기본)")
        self.check_profile_include_grid.setChecked(True)
        self.check_profile_include_grid.hide()
        opt_row.addWidget(self.check_profile_include_grid)
        profile_layout.addLayout(opt_row)

        feature_row = QHBoxLayout()
        self.check_profile_feature_lines = QCheckBox("✨ 샤프 엣지(능선) 라인 포함")
        self.check_profile_feature_lines.setChecked(False)
        self.check_profile_feature_lines.hide()
        feature_row.addWidget(self.check_profile_feature_lines, 1)

        feature_label = QLabel("임계각:")
        feature_label.hide()
        feature_row.addWidget(feature_label)
        self.spin_profile_feature_angle = QDoubleSpinBox()
        self.spin_profile_feature_angle.setRange(0.0, 180.0)
        self.spin_profile_feature_angle.setSingleStep(5.0)
        self.spin_profile_feature_angle.setValue(60.0)
        self.spin_profile_feature_angle.setSuffix(" °")
        self.spin_profile_feature_angle.setEnabled(False)
        self.check_profile_feature_lines.toggled.connect(self.spin_profile_feature_angle.setEnabled)
        self.spin_profile_feature_angle.hide()
        feature_row.addWidget(self.spin_profile_feature_angle)
        profile_layout.addLayout(feature_row)

        btn_export_pkg = QPushButton("📦 6방향 패키지 내보내기")
        btn_export_pkg.setToolTip("Top/Bottom/Front/Back/Left/Right를 한 폴더에 '뷰별 하위 폴더'로 저장합니다")
        btn_export_pkg.clicked.connect(lambda: self.exportRequested.emit({"type": "profile_2d_package"}))
        profile_layout.addWidget(btn_export_pkg)
        layout.addWidget(profile_group)

        layout.addStretch(1)

    def current_rubbing_target(self) -> str:
        try:
            idx = int(self.combo_rubbing_target.currentIndex())
        except Exception:
            idx = 0
        return {
            1: "selected",
        }.get(idx, "all")

    def set_rubbing_target(self, target: str) -> None:
        key = _normalize_surface_target(target)
        index = {
            "all": 0,
            "selected": 1,
            "outer": 1,
            "inner": 1,
            "migu": 1,
        }.get(key, 0)
        self.combo_rubbing_target.setCurrentIndex(int(index))

    def current_review_render_mode(self) -> str:
        try:
            value = self.combo_review_render_mode.currentData()
        except Exception:
            value = None
        text = str(value or "auto").strip()
        return text or "auto"

    def set_review_render_mode(self, mode: str) -> None:
        key = str(mode or "auto").strip() or "auto"
        idx = self.combo_review_render_mode.findData(key)
        if idx < 0:
            idx = self.combo_review_render_mode.findData("auto")
        if idx >= 0:
            self.combo_review_render_mode.setCurrentIndex(int(idx))

    def current_texture_detail_scale(self) -> float:
        try:
            value = float(self.spin_texture_detail_scale.value())
        except Exception:
            value = 1.0
        if not np.isfinite(value) or value <= 0.0:
            return 1.0
        return value

    def current_texture_smooth_extra(self) -> float:
        try:
            value = float(self.spin_texture_smooth_extra.value())
        except Exception:
            value = 0.0
        if not np.isfinite(value) or value <= 0.0:
            return 0.0
        return value

    def current_texture_postprocess(self) -> str:
        try:
            value = self.combo_texture_postprocess.currentData()
        except Exception:
            value = ""
        text = str(value or "").strip()
        return text

    def set_texture_adjustments(
        self,
        *,
        detail_scale: float = 1.0,
        smooth_extra: float = 0.0,
        postprocess: str = "",
    ) -> None:
        try:
            detail_value = float(detail_scale)
        except Exception:
            detail_value = 1.0
        if not np.isfinite(detail_value) or detail_value <= 0.0:
            detail_value = 1.0
        self.spin_texture_detail_scale.setValue(detail_value)

        try:
            smooth_value = float(smooth_extra)
        except Exception:
            smooth_value = 0.0
        if not np.isfinite(smooth_value) or smooth_value <= 0.0:
            smooth_value = 0.0
        self.spin_texture_smooth_extra.setValue(smooth_value)

        key = str(postprocess or "").strip()
        idx = self.combo_texture_postprocess.findData(key)
        if idx < 0:
            idx = self.combo_texture_postprocess.findData("")
        if idx >= 0:
            self.combo_texture_postprocess.setCurrentIndex(int(idx))


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

    UI_STATE_VERSION = 10
    
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
        self.viewport.faceSelectionChanged.connect(self.on_face_selection_count_changed)
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

        # 1.5) 기본 작업 흐름
        self.workflow_dock = QDockWidget("🧭 4축 작업 흐름", self)
        self.workflow_dock.setObjectName("dock_workflow")
        self.workflow_panel = WorkflowPanel(self.help_widget)
        self.workflow_panel.workflowRequested.connect(self.on_workflow_action)
        self.workflow_dock.setWidget(self.workflow_panel)

        # 2) 정치(변환)
        self.transform_dock = QDockWidget("세부 · 정위치", self)
        self.transform_dock.setObjectName("dock_transform")
        self.transform_panel = TransformPanel(self.viewport, self.help_widget)
        self.transform_dock.setWidget(self.transform_panel)

        # 3) 펼침
        self.selection_dock = QDockWidget("보조 · 탁본 표면 보정", self)
        self.selection_dock.setObjectName("dock_selection")
        self.selection_panel = SelectionPanel(self.help_widget)
        self.selection_panel.selectionChanged.connect(self.on_selection_action)
        self.selection_dock.setWidget(self.selection_panel)

        # 4) 기록면 전개
        self.flatten_dock = QDockWidget("세부 · 탁본", self)
        self.flatten_dock.setObjectName("dock_flatten")
        self.flatten_panel = FlattenPanel(self.help_widget)
        self.flatten_panel.flattenRequested.connect(self.on_flatten_requested)
        self.flatten_panel.previewRequested.connect(self.on_flatten_preview_requested)
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

        # 4) 기와 해석
        self.tile_dock = QDockWidget("세부 · 실측용 도면", self)
        self.tile_dock.setObjectName("dock_tile")
        self.tile_panel = TileInterpretationPanel(self.help_widget)
        self.tile_panel.interpretationChanged.connect(self.on_tile_interpretation_action)
        self.tile_dock.setWidget(self.tile_panel)

        # 5) 내보내기
        self.export_dock = QDockWidget("세부 · 실측/탁본 출력", self)
        self.export_dock.setObjectName("dock_export")
        self.export_panel = ExportPanel()
        self.export_panel.exportRequested.connect(self.on_export_requested)
        self.export_dock.setWidget(self.export_panel)

        # 5.5) 치수 측정
        self.measure_dock = QDockWidget("세부 · 제원측정", self)
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

        # 6) 단면/2D 지정 도구 (슬라이싱 + 십자선 + 라인 + ROI)
        self.section_dock = QDockWidget("보조 · 실측 단면/외곽", self)
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
        self.scene_panel.layerSelected.connect(self.on_layer_selected)
        self.scene_dock.setWidget(self.scene_panel)

        # 공통 도킹/플로팅 옵션
        for dock in [
            self.info_dock,
            self.workflow_dock,
            self.transform_dock,
            self.selection_dock,
            self.flatten_dock,
            self.tile_dock,
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

        # 기본 레이아웃: 단계형 작업 흐름 + 고급 패널 숨김
        self._apply_default_dock_layout()

    def _settings(self) -> QSettings:
        return QSettings("ArchMeshRubbing", "ArchMeshRubbing")

    def _apply_default_dock_layout(self):
        """기본 도킹 레이아웃 적용: 작업 흐름 중심 화면"""
        for dock in [
            self.info_dock,
            self.workflow_dock,
            self.transform_dock,
            self.selection_dock,
            self.flatten_dock,
            self.tile_dock,
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

        # 우측: 기본 작업 흐름만 유지
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.workflow_dock)

        for dock in [
            self.transform_dock,
            self.scene_dock,
            self.selection_dock,
            self.flatten_dock,
            self.tile_dock,
            self.section_dock,
            self.export_dock,
            self.measure_dock,
        ]:
            dock.hide()

        self.workflow_dock.raise_()
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
        for dock in (getattr(self, "help_dock", None),):
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
        try:
            toolbar = getattr(self, "trans_toolbar", None)
            if toolbar is not None:
                toolbar.hide()
        except Exception:
            pass

    def _save_ui_state(self):
        settings = self._settings()
        settings.setValue("ui/state_version", self.UI_STATE_VERSION)
        settings.setValue("ui/geometry", self.saveGeometry())
        settings.setValue("ui/state", self.saveState(self.UI_STATE_VERSION))

    def reset_panel_layout(self):
        """사용자 레이아웃 저장값 삭제 후 기본 화면으로 복구"""
        settings = self._settings()
        settings.remove("ui/geometry")
        settings.remove("ui/state")
        settings.remove("ui/state_version")
        self._apply_default_dock_layout()
        try:
            self.status_info.setText("기본 화면으로 복귀했습니다.")
        except Exception:
            pass

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
        try:
            self.viewport.mark_floor_pick_pending(0.08)
        except Exception:
            pass
        self.viewport.status_info = "Preparing floor pick... please wait, then click on mesh."
        QTimer.singleShot(
            90,
            lambda: (
                setattr(
                    self.viewport,
                    "status_info",
                    "Floor pick ready: click 3 points on mesh (Enter to confirm).",
                ),
                self.viewport.update(),
            )
            if getattr(self.viewport, "picking_mode", "") == "floor_3point"
            else None,
        )
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
        try:
            self.viewport.mark_floor_pick_pending(0.10)
        except Exception:
            pass
        self.viewport.status_info = "Preparing floor face pick... please wait, then click a face."
        QTimer.singleShot(
            110,
            lambda: (
                setattr(
                    self.viewport,
                    "status_info",
                    "Floor face pick ready: click a support face.",
                ),
                self.viewport.update(),
            )
            if getattr(self.viewport, "picking_mode", "") == "floor_face"
            else None,
        )
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
        """Align by brushed-face normal and keep brushed region touching XY plane."""
        obj = self.viewport.selected_obj
        if not obj or not self.viewport.brush_selected_faces:
            return

        # Brushed faces are picked in world view. Bake first so mesh-space == world-space.
        self.viewport.bake_object_transform(obj)

        try:
            faces = np.asarray(obj.mesh.faces, dtype=np.int64)
            vertices = np.asarray(obj.mesh.vertices, dtype=np.float64)
        except Exception:
            return

        selected = []
        for idx in list(self.viewport.brush_selected_faces):
            try:
                fi = int(idx)
            except Exception:
                continue
            if 0 <= fi < int(len(faces)):
                selected.append(fi)

        if not selected:
            self.viewport.status_info = "선택된 브러시 면이 없습니다."
            self.viewport.update()
            return

        total_normal = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        total_area = 0.0
        selected_vidx: set[int] = set()

        for face_idx in selected:
            f = faces[int(face_idx)]
            i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])
            selected_vidx.add(i0)
            selected_vidx.add(i1)
            selected_vidx.add(i2)

            v0 = vertices[i0]
            v1 = vertices[i1]
            v2 = vertices[i2]
            n = np.cross(v1 - v0, v2 - v0)
            n_len = float(np.linalg.norm(n))
            if n_len > 1e-12:
                total_normal += n
                total_area += (n_len * 0.5)

        if total_area < 1e-9 or float(np.linalg.norm(total_normal)) <= 1e-12:
            self.viewport.status_info = "유효한 바닥 브러시 면을 찾지 못했습니다."
            self.viewport.update()
            return

        if selected_vidx:
            sel_idx = np.asarray(sorted(selected_vidx), dtype=np.int64)
            selected_pts = np.asarray(vertices[sel_idx], dtype=np.float64)
        else:
            selected_pts = np.asarray(vertices, dtype=np.float64)

        if selected_pts.size == 0:
            self.viewport.status_info = "브러시 영역 정점을 찾지 못했습니다."
            self.viewport.update()
            return

        centroid = np.mean(selected_pts, axis=0)
        avg_normal = total_normal / float(np.linalg.norm(total_normal))

        try:
            mesh_centroid = np.asarray(obj.mesh.centroid, dtype=np.float64).reshape(3)
        except Exception:
            mesh_centroid = np.mean(np.asarray(vertices, dtype=np.float64), axis=0)
        avg_normal = orient_plane_normal_toward(avg_normal, centroid, mesh_centroid)

        self.viewport.save_undo_state()
        R = self.align_mesh_to_normal(avg_normal, pivot=centroid)
        if R is None:
            self.viewport.status_info = "브러시 바닥 정렬 회전 계산에 실패했습니다."
            self.viewport.update()
            return

        selected_rot = (R @ (selected_pts - centroid).T).T + centroid

        # Final parallel pass: make selected floor support truly parallel to XY.
        try:
            plane_after = fit_plane_normal(selected_rot, robust=False)
            if plane_after is not None:
                normal_after, centroid_after = plane_after
                target_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                if float(np.dot(normal_after, target_up)) < 0.0:
                    normal_after = -normal_after
                R2 = rotation_matrix_align_vectors(normal_after, target_up)
                verts2 = np.asarray(obj.mesh.vertices, dtype=np.float64)
                pivot2 = np.asarray(centroid_after, dtype=np.float64).reshape(3)
                obj.mesh.vertices = ((R2 @ (verts2 - pivot2).T).T + pivot2).astype(np.float32)
                selected_rot = (R2 @ (selected_rot - pivot2).T).T + pivot2
        except Exception:
            pass

        z_residual = float("nan")
        try:
            z_vals = np.asarray(selected_rot, dtype=np.float64)[:, FLOOR_ALIGN_AXIS_Z]
            floor_z = compute_minimax_center_shift(z_vals)
        except Exception:
            floor_z = 0.0
        if np.isfinite(floor_z):
            obj.mesh.vertices[:, FLOOR_ALIGN_AXIS_Z] -= float(floor_z)
            selected_rot[:, FLOOR_ALIGN_AXIS_Z] -= float(floor_z)
        # Keep the entire mesh above XY after floor alignment.
        try:
            mesh_z = np.asarray(obj.mesh.vertices, dtype=np.float64)[:, FLOOR_ALIGN_AXIS_Z]
            lift_z = compute_nonpenetration_lift(mesh_z, floor_z=0.0)
        except Exception:
            lift_z = 0.0
        if np.isfinite(lift_z) and lift_z > 0.0:
            obj.mesh.vertices[:, FLOOR_ALIGN_AXIS_Z] += float(lift_z)
            selected_rot[:, FLOOR_ALIGN_AXIS_Z] += float(lift_z)
        try:
            z_after = np.asarray(selected_rot, dtype=np.float64)[:, FLOOR_ALIGN_AXIS_Z]
            z_residual = float(np.nanmax(np.abs(z_after)))
        except Exception:
            z_residual = float("nan")

        try:
            obj.mesh._bounds = None
            obj.mesh._centroid = None
            obj.mesh._surface_area = None
        except Exception:
            pass
        try:
            obj.mesh.compute_normals(compute_vertex_normals=False, force=True)
        except Exception:
            pass
        obj._trimesh = None
        obj.translation = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self.viewport.update_vbo(obj)
        self.sync_transform_panel()

        count = int(len(selected))
        self.viewport.brush_selected_faces.clear()
        self.viewport.picking_mode = 'none'
        if np.isfinite(z_residual):
            self.viewport.status_info = (
                f"✅ 브러시 바닥 정렬 완료 ({count}개 면 / 선택점 Z잔차 ±{z_residual:.4f})"
            )
        else:
            self.viewport.status_info = f"✅ 브러시 바닥 정렬 완료 ({count}개 면)"
        self.viewport.update()
        self.viewport.meshTransformChanged.emit()

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

    def _optimize_points_xy_contact(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Minimize picked-point Z spread (plane flatness) via small X/Y tilt search."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(pts) < 3:
            return np.eye(3, dtype=np.float64), np.array([0.0, 0.0, 0.0], dtype=np.float64), pts

        pivot = np.mean(pts, axis=0)
        centered = pts - pivot

        def _rot_x(rad: float) -> np.ndarray:
            c = float(np.cos(rad))
            s = float(np.sin(rad))
            return np.array(
                [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
                dtype=np.float64,
            )

        def _rot_y(rad: float) -> np.ndarray:
            c = float(np.cos(rad))
            s = float(np.sin(rad))
            return np.array(
                [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                dtype=np.float64,
            )

        def _eval(ax: float, ay: float) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
            R = _rot_y(ay) @ _rot_x(ax)
            pts_r = (R @ centered.T).T + pivot
            z = np.asarray(pts_r[:, FLOOR_ALIGN_AXIS_Z], dtype=np.float64)
            if z.size == 0 or not np.isfinite(z).all():
                return (float("inf"), float("inf")), pts_r, R
            # Height offset is irrelevant (we translate to Z=0 later).
            # Optimize flatness of picked points around their own center level.
            z_rel = z - float(np.median(z))
            return (float(np.max(np.abs(z_rel))), float(np.mean(np.abs(z_rel)))), pts_r, R

        ax = 0.0
        ay = 0.0
        best_metric, best_pts, best_R = _eval(ax, ay)

        for step_deg in FLOOR_OPTIMIZE_STEP_DEGREES:
            step = float(np.deg2rad(step_deg))
            improved = True
            while improved:
                improved = False
                for dax, day in (
                    (step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step),
                    (step, step), (step, -step), (-step, step), (-step, -step),
                ):
                    metric, pts_try, R_try = _eval(ax + dax, ay + day)
                    better = (
                        (metric[0] + 1e-12 < best_metric[0])
                        or (abs(metric[0] - best_metric[0]) <= 1e-12 and metric[1] + 1e-12 < best_metric[1])
                    )
                    if better:
                        ax += dax
                        ay += day
                        best_metric = metric
                        best_pts = pts_try
                        best_R = R_try
                        improved = True

        return best_R, pivot, best_pts

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

        # 2) 선택한 점 전체를 반영한 least-squares 평면을 추정한다.
        plane = fit_plane_normal(points, robust=False)
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

        # Final parallel pass: enforce selected floor points parallel to XY.
        try:
            R2, pivot2, points_opt = self._optimize_points_xy_contact(points_rotated)
            if R2 is not None:
                verts2 = np.asarray(obj.mesh.vertices, dtype=np.float64)
                pivot2 = np.asarray(pivot2, dtype=np.float64).reshape(3)
                obj.mesh.vertices = ((R2 @ (verts2 - pivot2).T).T + pivot2).astype(np.float32)
                points_rotated = np.asarray(points_opt, dtype=np.float64)
        except Exception:
            pass

        # 4) 선택 점들의 Z 잔차를 XY 기준으로 최소화하도록 중심 정렬한다.
        #    (기존 min(z)=0 방식은 한두 점만 닿고 나머지가 뜨기 쉬움)
        z_residual = float("nan")
        try:
            z_vals = np.asarray(points_rotated, dtype=np.float64)[:, FLOOR_ALIGN_AXIS_Z]
            # Minimax center: minimize max_i |z_i - t|
            floor_z = compute_minimax_center_shift(z_vals)
        except Exception:
            floor_z = 0.0
        if np.isfinite(floor_z):
            obj.mesh.vertices[:, FLOOR_ALIGN_AXIS_Z] -= float(floor_z)
            points_rotated[:, FLOOR_ALIGN_AXIS_Z] -= float(floor_z)
        # Keep the entire mesh above XY after floor alignment.
        try:
            mesh_z = np.asarray(obj.mesh.vertices, dtype=np.float64)[:, FLOOR_ALIGN_AXIS_Z]
            lift_z = compute_nonpenetration_lift(mesh_z, floor_z=0.0)
        except Exception:
            lift_z = 0.0
        if np.isfinite(lift_z) and lift_z > 0.0:
            obj.mesh.vertices[:, FLOOR_ALIGN_AXIS_Z] += float(lift_z)
            points_rotated[:, FLOOR_ALIGN_AXIS_Z] += float(lift_z)
        try:
            z_after = np.asarray(points_rotated, dtype=np.float64)[:, FLOOR_ALIGN_AXIS_Z]
            z_residual = float(np.nanmax(np.abs(z_after)))
        except Exception:
            z_residual = float("nan")

        try:
            obj.mesh._bounds = None
            obj.mesh._centroid = None
            obj.mesh._surface_area = None
        except Exception:
            pass
        try:
            obj.mesh.compute_normals(compute_vertex_normals=False, force=True)
        except Exception:
            pass
        obj._trimesh = None
        obj.translation = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self.viewport.update_vbo(obj)
        self.sync_transform_panel()
        if np.isfinite(z_residual):
            self.viewport.status_info = (
                f"✅ 바닥 정렬 완료 (점 {len(points)}개 / 선택점 Z잔차 ±{z_residual:.4f})"
            )
        else:
            self.viewport.status_info = f"✅ 바닥 정렬 완료 (점 {len(points)}개 기반)"
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

    def on_layer_selected(self, obj_idx: int, layer_idx: int):
        try:
            oi = int(obj_idx)
            li = int(layer_idx)
            self.viewport.select_object(oi)
            self.viewport.set_active_polyline_layer(oi, li)
            self.viewport.update()
            self.status_info.setText("Section layer selected: drag in viewport (Shift = axis lock)")
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
        action_front.triggered.connect(lambda: self._set_canonical_view("front"))
        view_menu.addAction(action_front)
        
        action_back = QAction("2️⃣ 후면 뷰", self)
        action_back.setShortcut("2")
        action_back.triggered.connect(lambda: self._set_canonical_view("back"))
        view_menu.addAction(action_back)
        
        action_right = QAction("3️⃣ 우측면 뷰", self)
        action_right.setShortcut("3")
        action_right.triggered.connect(lambda: self._set_canonical_view("right"))
        view_menu.addAction(action_right)
        
        action_left = QAction("4️⃣ 좌측면 뷰", self)
        action_left.setShortcut("4")
        action_left.triggered.connect(lambda: self._set_canonical_view("left"))
        view_menu.addAction(action_left)
        
        action_top = QAction("5️⃣ 상면 뷰", self)
        action_top.setShortcut("5")
        action_top.triggered.connect(lambda: self._set_canonical_view("top"))
        view_menu.addAction(action_top)
        
        action_bottom = QAction("6️⃣ 하면 뷰", self)
        action_bottom.setShortcut("6")
        action_bottom.triggered.connect(lambda: self._set_canonical_view("bottom"))
        view_menu.addAction(action_bottom)

        view_menu.addSeparator()

        action_show_advanced = QAction("정위치/실측/탁본 도구 열기", self)
        action_show_advanced.triggered.connect(self._show_advanced_panels)
        view_menu.addAction(action_show_advanced)

        action_open_selection_tools = QAction("표면 보정 도구 열기", self)
        action_open_selection_tools.triggered.connect(self._show_selection_panel)
        view_menu.addAction(action_open_selection_tools)

        action_open_section_tools = QAction("단면/외곽 도구 열기", self)
        action_open_section_tools.triggered.connect(lambda: self.on_selection_action("open_section_tools", None))
        view_menu.addAction(action_open_section_tools)

        action_open_measure_tools = QAction("치수 측정 도구 열기", self)
        action_open_measure_tools.triggered.connect(self._show_measure_panel)
        view_menu.addAction(action_open_measure_tools)

        action_reset_layout = QAction("기본 화면 복귀", self)
        action_reset_layout.triggered.connect(self.reset_panel_layout)
        view_menu.addAction(action_reset_layout)

        panels_menu = view_menu.addMenu("패널 표시/숨김")
        if panels_menu is not None:
            panels_menu.addAction(self.workflow_dock.toggleViewAction())
            panels_menu.addAction(self.info_dock.toggleViewAction())
            panels_menu.addAction(self.scene_dock.toggleViewAction())
            panels_menu.addSeparator()
            panels_menu.addAction(self.transform_dock.toggleViewAction())
            panels_menu.addAction(self.tile_dock.toggleViewAction())
            panels_menu.addAction(self.selection_dock.toggleViewAction())
            panels_menu.addAction(self.flatten_dock.toggleViewAction())
            panels_menu.addAction(self.export_dock.toggleViewAction())
            panels_menu.addAction(self.section_dock.toggleViewAction())
            panels_menu.addAction(self.measure_dock.toggleViewAction())
        
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

        action_open_project = QAction("📁 프로젝트", self)
        action_open_project.triggered.connect(self.open_project)
        toolbar.addAction(action_open_project)

        toolbar.addSeparator()

        action_fit = QAction("🔍 뷰 맞춤", self)
        action_fit.setToolTip("메쉬가 화면에 꽉 차도록 카메라 조정")
        action_fit.triggered.connect(self.fit_view)
        toolbar.addAction(action_fit)

        toolbar.addSeparator()
        
        # 6방향 뷰 버튼
        action_front = QAction("정면", self)
        action_front.setToolTip("정면 뷰 (1)")
        action_front.triggered.connect(lambda: self._set_canonical_view("front"))
        toolbar.addAction(action_front)
        
        action_back = QAction("후면", self)
        action_back.setToolTip("후면 뷰 (2)")
        action_back.triggered.connect(lambda: self._set_canonical_view("back"))
        toolbar.addAction(action_back)
        
        action_right = QAction("우측", self)
        action_right.setToolTip("우측면 뷰 (3)")
        action_right.triggered.connect(lambda: self._set_canonical_view("right"))
        toolbar.addAction(action_right)
        
        action_left = QAction("좌측", self)
        action_left.setToolTip("좌측면 뷰 (4)")
        action_left.triggered.connect(lambda: self._set_canonical_view("left"))
        toolbar.addAction(action_left)
        
        action_top = QAction("상면", self)
        action_top.setToolTip("상면 뷰 (5)")
        action_top.triggered.connect(lambda: self._set_canonical_view("top"))
        toolbar.addAction(action_top)
        
        action_bottom = QAction("하면", self)
        action_bottom.setToolTip("하면 뷰 (6)")
        action_bottom.triggered.connect(lambda: self._set_canonical_view("bottom"))
        toolbar.addAction(action_bottom)

        toolbar.addSeparator()

        action_record_top = QAction("상면 기록", self)
        action_record_top.triggered.connect(
            lambda: self.on_tile_interpretation_action("prepare_record_surface", {"view": "top"})
        )
        toolbar.addAction(action_record_top)

        action_record_bottom = QAction("하면 기록", self)
        action_record_bottom.triggered.connect(
            lambda: self.on_tile_interpretation_action("prepare_record_surface", {"view": "bottom"})
        )
        toolbar.addAction(action_record_bottom)

        action_preview = QAction("미리보기", self)
        action_preview.triggered.connect(self.on_flatten_preview_requested)
        toolbar.addAction(action_preview)

        action_review = QAction("검토 시트", self)
        action_review.triggered.connect(
            lambda: self.on_export_requested({"type": "review_sheet", "target": "selected"})
        )
        toolbar.addAction(action_review)

    def init_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        self.status_info = QLabel("📂 파일을 열거나 드래그하세요")
        self.status_mesh = QLabel("") # 메쉬 정보 (정점, 면)
        self.status_grid = QLabel("격자: -")
        self.status_unit = QLabel("단위: -")
        
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
            self.open_file_path(filepath, prompt_unit=True)

    def open_file_path(self, filepath: str, *, prompt_unit: bool = True) -> None:
        """Open a mesh file from a known path."""
        if not filepath:
            return

        scale_factor = 1.0
        if bool(prompt_unit):
            dialog = UnitSelectionDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
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
            synthetic_truth = self._coerce_synthetic_truth(getattr(obj, "tile_synthetic_truth", None))
            evaluation_report = self._coerce_tile_evaluation_report(getattr(obj, "tile_evaluation_report", None))
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
                    "tile_interpretation": self._ensure_tile_interpretation_state(obj).to_dict(),
                    "tile_synthetic_truth": synthetic_truth.to_dict() if synthetic_truth is not None else None,
                    "tile_evaluation_report": evaluation_report.to_dict() if evaluation_report is not None else None,
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
                dpi = DEFAULT_EXPORT_DPI
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
            try:
                review_render_mode = str(export_panel.current_review_render_mode() or "auto")
            except Exception:
                review_render_mode = "auto"
            try:
                texture_detail_scale = float(export_panel.current_texture_detail_scale())
            except Exception:
                texture_detail_scale = 1.0
            try:
                texture_smooth_extra = float(export_panel.current_texture_smooth_extra())
            except Exception:
                texture_smooth_extra = 0.0
            try:
                texture_postprocess = str(export_panel.current_texture_postprocess() or "")
            except Exception:
                texture_postprocess = ""
        else:
            dpi = DEFAULT_EXPORT_DPI
            format_index = 0
            scale_bar = True
            profile_include_grid = True
            profile_feature_lines = False
            profile_feature_angle = 60.0
            review_render_mode = "auto"
            texture_detail_scale = 1.0
            texture_smooth_extra = 0.0
            texture_postprocess = ""

        ui_state["export"] = {
            "dpi": int(dpi),
            "format_index": int(format_index),
            "scale_bar": bool(scale_bar),
            "profile_include_grid": bool(profile_include_grid),
            "profile_feature_lines": bool(profile_feature_lines),
            "profile_feature_angle": float(profile_feature_angle),
            "review_render_mode": str(review_render_mode or "auto"),
            "texture_detail_scale": float(texture_detail_scale),
            "texture_smooth_extra": float(texture_smooth_extra),
            "texture_postprocess": str(texture_postprocess or ""),
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

        try:
            obj.tile_interpretation_state = TileInterpretationState.from_dict(obj_state.get("tile_interpretation"))
        except Exception:
            obj.tile_interpretation_state = TileInterpretationState()
        try:
            raw_truth = obj_state.get("tile_synthetic_truth")
            obj.tile_synthetic_truth = (
                SyntheticTileGroundTruth.from_dict(raw_truth) if isinstance(raw_truth, dict) else None
            )
        except Exception:
            obj.tile_synthetic_truth = None
        try:
            raw_report = obj_state.get("tile_evaluation_report")
            obj.tile_evaluation_report = (
                TileEvaluationReport.from_dict(raw_report) if isinstance(raw_report, dict) else None
            )
        except Exception:
            obj.tile_evaluation_report = None

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
        self._sync_tile_panel()

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
                vp.roi_caps_enabled = bool(roi_s.get("caps", True))
            except Exception:
                vp.roi_caps_enabled = True
            try:
                if bool(vp.roi_enabled) and not bool(vp.roi_caps_enabled):
                    vp.roi_caps_enabled = True
            except Exception:
                vp.roi_caps_enabled = True
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
        vp._canonical_view_key = None

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
                self.export_panel.spin_dpi.setValue(
                    int(exp.get("dpi", self.export_panel.spin_dpi.value()) or DEFAULT_EXPORT_DPI)
                )
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
                self.export_panel.set_review_render_mode(
                    str(exp.get("review_render_mode", self.export_panel.current_review_render_mode()) or "auto")
                )
                self.export_panel.set_texture_adjustments(
                    detail_scale=float(exp.get("texture_detail_scale", self.export_panel.current_texture_detail_scale()) or 1.0),
                    smooth_extra=float(exp.get("texture_smooth_extra", self.export_panel.current_texture_smooth_extra()) or 0.0),
                    postprocess=str(exp.get("texture_postprocess", self.export_panel.current_texture_postprocess()) or ""),
                )
            except Exception:
                pass

        # Slice presets
        sl = ui.get("slice", {})
        slice_panel = getattr(self, "slice_panel", None)
        if isinstance(sl, dict) and slice_panel is not None:
            try:
                slice_panel.set_presets(sl.get("presets", []))
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
            unit_s = str(getattr(mesh_data, "unit", "") or "").strip().lower()
            if unit_s not in ("mm", "cm", "m"):
                unit_s = str(getattr(self.mesh_loader, "default_unit", DEFAULT_MESH_UNIT) or DEFAULT_MESH_UNIT).strip().lower()
            self.status_unit.setText(f"단위: {unit_s}")

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
                self.status_info.setText(
                    f"프로젝트 로드됨: {obj_name} | 다음: 1단계 정치에서 기준 시점을 확인하세요."
                )
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
                self.status_info.setText(
                    f"메쉬 로드됨: {Path(filepath).name} | 다음: 1단계 정치에서 기준 시점을 맞추세요."
                )
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
        try:
            obj = self.viewport.selected_obj
            count = len(getattr(obj, "selected_faces", set()) or set()) if obj is not None else 0
            self.selection_panel.update_selection_count(int(count))
        except Exception:
            pass
        self._sync_tile_panel()
        self._sync_workflow_panel()
        
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
            obj = self.viewport.selected_obj
            count = len(getattr(obj, "selected_faces", set()) or set()) if obj is not None else 0
            self.selection_panel.update_selection_count(int(count))
        except Exception:
            pass
        self._sync_tile_panel()
        self._sync_workflow_panel()

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

    def on_face_selection_count_changed(self, count: int) -> None:
        try:
            self.selection_panel.update_selection_count(int(count))
        except Exception:
            pass
        self._sync_tile_panel()
        self._sync_workflow_panel()

    def _ensure_tile_interpretation_state(self, obj) -> TileInterpretationState:
        raw_state = getattr(obj, "tile_interpretation_state", None)
        if isinstance(raw_state, TileInterpretationState):
            return raw_state
        state = TileInterpretationState.from_dict(raw_state if isinstance(raw_state, dict) else {})
        setattr(obj, "tile_interpretation_state", state)
        return state

    @staticmethod
    def _coerce_synthetic_truth(raw: object) -> SyntheticTileGroundTruth | None:
        if isinstance(raw, SyntheticTileGroundTruth):
            return raw
        if isinstance(raw, dict):
            try:
                return SyntheticTileGroundTruth.from_dict(raw)
            except Exception:
                return None
        return None

    @staticmethod
    def _coerce_tile_evaluation_report(raw: object) -> TileEvaluationReport | None:
        if isinstance(raw, TileEvaluationReport):
            return raw
        if isinstance(raw, dict):
            try:
                return TileEvaluationReport.from_dict(raw)
            except Exception:
                return None
        return None

    @staticmethod
    def _synthetic_tile_spec_from_preset(preset: object, *, seed: int) -> SyntheticTileSpec:
        return synthetic_tile_spec_from_preset(preset, seed=int(seed))

    def _tile_wizard_status(
        self,
        obj,
        state: TileInterpretationState,
    ) -> dict[str, Any]:
        selected_faces = len(getattr(obj, "selected_faces", set()) or set()) if obj is not None else 0
        tile_ready = state.tile_class != TileClass.UNKNOWN and state.split_scheme != SplitScheme.UNKNOWN
        analyzed_sections = sum(
            1 for item in list(getattr(state, "section_observations", []) or [])
            if int(getattr(item, "profile_point_count", 0) or 0) > 0
        )
        accepted_sections = sum(
            1 for item in list(getattr(state, "section_observations", []) or [])
            if bool(getattr(item, "accepted", False))
        )

        if not tile_ready:
            return {
                "summary": "1/6 유형과 2분할/4분할 가설을 먼저 정하세요.",
                "progress": 8,
                "next_label": "유형/분할 먼저 지정",
                "next_enabled": False,
                "next_action": None,
                "next_data": None,
            }
        if not state.axis_hint.is_defined():
            mode = "selected" if selected_faces > 0 else "mesh"
            return {
                "summary": "2/6 길이축 힌트를 추정해야 합니다.",
                "progress": 20,
                "next_label": f"다음 단계: 길이축 추정 ({'현재 선택' if mode == 'selected' else '전체 메쉬'})",
                "next_enabled": True,
                "next_action": "estimate_axis",
                "next_data": {"mode": mode},
            }
        if accepted_sections <= 0:
            mode = "selected" if selected_faces > 0 else "mesh"
            return {
                "summary": "3/6 대표 단면 후보를 자동 제안하고 채택할 단계입니다.",
                "progress": 35,
                "next_label": "다음 단계: 대표 단면 5개 자동 제안",
                "next_enabled": True,
                "next_action": "auto_section_candidates",
                "next_data": {"mode": mode, "count": 5},
            }
        if analyzed_sections <= 0:
            return {
                "summary": f"4/6 채택된 단면 {accepted_sections}개가 있습니다. 프로파일 분석이 필요합니다.",
                "progress": 52,
                "next_label": "다음 단계: 단면 프로파일 분석",
                "next_enabled": True,
                "next_action": "analyze_section_profiles",
                "next_data": {"mode": "selected_preferred"},
            }
        if not state.mandrel_fit.is_defined():
            return {
                "summary": f"5/6 분석된 단면 {analyzed_sections}개를 기준으로 와통 반경을 피팅합니다.",
                "progress": 72,
                "next_label": "다음 단계: 와통 초벌 피팅",
                "next_enabled": True,
                "next_action": "fit_mandrel",
                "next_data": {"mode": "selected_preferred"},
            }
        record_view_key = str(state.record_view or "").strip().lower()
        if record_view_key not in {"top", "bottom"}:
            return {
                "summary": "6/6 상면 또는 하면 기록면을 준비하면 위저드가 완료됩니다.",
                "progress": 88,
                "next_label": "다음 단계: 상면 기록 준비",
                "next_enabled": True,
                "next_action": "prepare_record_surface",
                "next_data": {"view": "top"},
            }
        if selected_faces <= 0:
            record_label = "상면" if record_view_key == "top" else "하면"
            return {
                "summary": f"6/6 {record_label} 기록면을 계산 중이거나 선택이 비어 있습니다. 다시 준비하거나 보정 후 진행하세요.",
                "progress": 92,
                "next_label": f"다음 단계: {record_label} 기록면 다시 준비",
                "next_enabled": True,
                "next_action": "prepare_record_surface",
                "next_data": {"view": record_view_key},
            }
        record_label = "상면" if record_view_key == "top" else "하면"
        return {
            "summary": f"완료: {record_label} 기록면이 준비되었습니다. 검토 시트 저장이나 평가를 실행할 수 있습니다.",
            "progress": 100,
            "next_label": "위저드 완료",
            "next_enabled": False,
            "next_action": None,
            "next_data": None,
        }

    @staticmethod
    def _synthetic_truth_summary(truth: SyntheticTileGroundTruth | None) -> str:
        if truth is None:
            return ""
        return " | ".join(truth.summary_lines())

    @staticmethod
    def _tile_evaluation_summary(report: TileEvaluationReport | None, *, unit: str) -> str:
        if report is None:
            return ""
        return " | ".join(report.summary_lines(unit=unit))

    @staticmethod
    def _synthetic_suite_summary(report: SyntheticBenchmarkSuiteReport | None) -> str:
        if report is None:
            return ""
        lines = list(report.summary_lines())
        lines.extend(report.failing_case_lines(limit=3))
        return "\n".join(str(line) for line in lines if str(line or "").strip())

    def _add_synthetic_tile_artifact(self, artifact) -> None:
        self.viewport.add_mesh_object(artifact.mesh, artifact.name)
        obj = getattr(self.viewport, "selected_obj", None)
        if obj is None:
            raise RuntimeError("합성 기와 객체를 장면에 추가하지 못했습니다.")

        state = TileInterpretationState(
            tile_class=TileClass.UNKNOWN,
            split_scheme=SplitScheme.UNKNOWN,
            workflow_stage="hypothesis",
            note="synthetic_tile_benchmark",
        )
        state.touch()
        setattr(obj, "tile_interpretation_state", state)
        setattr(obj, "tile_synthetic_truth", artifact.truth)
        setattr(obj, "tile_evaluation_report", TileEvaluationReport())
        try:
            obj.selected_faces = set()
        except Exception:
            pass

        self.current_mesh = artifact.mesh
        self.current_filepath = None
        try:
            self.selection_panel.update_selection_count(0)
        except Exception:
            pass
        try:
            self.viewport.faceSelectionChanged.emit(0)
        except Exception:
            pass
        self._sync_tile_panel()

    @staticmethod
    def _tile_slot_key(slot_index: object) -> str:
        try:
            index = int(slot_index)
        except Exception:
            index = 0
        index = max(1, min(3, index))
        return f"slot_{index}"

    @staticmethod
    def _build_tile_slot_label(
        state: TileInterpretationState,
        *,
        slot_index: int,
        selected_face_count: int,
    ) -> str:
        parts: list[str] = []
        record_view = str(getattr(state, "record_view", "") or "").strip().lower()
        if record_view == "top":
            parts.append("상면 기록")
        elif record_view == "bottom":
            parts.append("하면 기록")

        tile_class = getattr(state, "tile_class", TileClass.UNKNOWN)
        if tile_class != TileClass.UNKNOWN:
            parts.append(tile_class.label_ko)

        accepted_sections = sum(
            1 for item in list(getattr(state, "section_observations", []) or []) if bool(getattr(item, "accepted", False))
        )
        if accepted_sections > 0:
            parts.append(f"단면 {accepted_sections}개")

        if bool(getattr(getattr(state, "mandrel_fit", None), "is_defined", lambda: False)()):
            parts.append("와통 피팅")

        if int(selected_face_count) > 0:
            parts.append(f"선택 {int(selected_face_count)}면")

        if not parts:
            return f"슬롯 {int(slot_index)}"
        return " | ".join(parts[:4])

    def _set_object_selected_faces(self, obj, face_ids: object) -> int:
        if obj is None:
            return 0

        try:
            max_face_count = int(getattr(getattr(obj, "mesh", None), "n_faces", 0) or 0)
        except Exception:
            max_face_count = 0

        selected: set[int] = set()
        try:
            for item in list(face_ids or []):
                try:
                    face_id = int(item)
                except Exception:
                    continue
                if face_id < 0:
                    continue
                if max_face_count > 0 and face_id >= max_face_count:
                    continue
                selected.add(face_id)
        except Exception:
            selected = set()

        try:
            obj.selected_faces = selected
        except Exception:
            return 0

        try:
            self.viewport.brush_selected_faces.clear()
        except Exception:
            pass
        try:
            self.selection_panel.update_selection_count(len(selected))
        except Exception:
            pass
        try:
            self.viewport.faceSelectionChanged.emit(len(selected))
        except Exception:
            pass
        try:
            if selected:
                self.export_panel.set_rubbing_target("selected")
        except Exception:
            pass
        try:
            self.viewport.update()
        except Exception:
            pass
        return int(len(selected))

    def _sync_tile_panel(self) -> None:
        panel = getattr(self, "tile_panel", None)
        if panel is None:
            try:
                self._sync_workflow_panel()
            except Exception:
                pass
            return

        obj = getattr(self.viewport, "selected_obj", None)
        if obj is None or getattr(obj, "mesh", None) is None:
            panel.update_state(None, object_name="", object_unit="", selected_faces=0, total_faces=0)
            try:
                self._sync_workflow_panel()
            except Exception:
                pass
            return

        state = self._ensure_tile_interpretation_state(obj)
        try:
            selected_faces = len(getattr(obj, "selected_faces", set()) or set())
        except Exception:
            selected_faces = 0
        try:
            total_faces = int(getattr(getattr(obj, "mesh", None), "n_faces", 0) or 0)
        except Exception:
            total_faces = 0
        record_view_key = str(getattr(state, "record_view", "") or "").strip().lower()
        if record_view_key in {"top", "bottom"}:
            if int(selected_faces) > 0:
                if str(getattr(state, "workflow_stage", "") or "") != "record_surface":
                    state.workflow_stage = "record_surface"
            elif str(getattr(state, "workflow_stage", "") or "") == "record_surface":
                state.workflow_stage = "record_surface_pending"
        truth = self._coerce_synthetic_truth(getattr(obj, "tile_synthetic_truth", None))
        report = self._coerce_tile_evaluation_report(getattr(obj, "tile_evaluation_report", None))
        wizard = self._tile_wizard_status(obj, state)

        panel.update_state(
            state,
            object_name=str(getattr(obj, "name", "") or "Object"),
            object_unit=str(getattr(getattr(obj, "mesh", None), "unit", "") or ""),
            selected_faces=int(selected_faces),
            total_faces=int(total_faces),
            wizard_summary=str(wizard.get("summary", "") or ""),
            wizard_progress=int(wizard.get("progress", 0) or 0),
            wizard_next_label=str(wizard.get("next_label", "") or ""),
            wizard_next_enabled=bool(wizard.get("next_enabled", False)),
            synthetic_truth_summary=self._synthetic_truth_summary(truth),
            evaluation_summary=self._tile_evaluation_summary(
                report,
                unit=str(getattr(getattr(obj, "mesh", None), "unit", "") or "mm"),
            ),
        )
        try:
            self._sync_workflow_panel()
        except Exception:
            pass

    def _show_dock_on_right(self, dock: QDockWidget, *, tab_with: QDockWidget | None = None) -> None:
        if dock is None:
            return
        try:
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        except Exception:
            pass
        try:
            dock.show()
        except Exception:
            pass
        try:
            if tab_with is not None and tab_with is not dock and tab_with.isVisible():
                self.tabifyDockWidget(tab_with, dock)
        except Exception:
            pass
        try:
            dock.raise_()
        except Exception:
            pass

    def _show_measure_panel(self) -> None:
        anchor = None
        try:
            if self.tile_dock.isVisible():
                anchor = self.tile_dock
        except Exception:
            anchor = None
        self._show_dock_on_right(self.measure_dock, tab_with=anchor)
        try:
            self.status_info.setText("제원측정 도구를 열었습니다. 기본 작업은 4축 작업 흐름 패널에서 이어집니다.")
        except Exception:
            pass

    def _show_selection_panel(self) -> None:
        anchor = None
        try:
            if self.flatten_dock.isVisible():
                anchor = self.flatten_dock
            elif self.tile_dock.isVisible():
                anchor = self.tile_dock
        except Exception:
            anchor = None
        self._show_dock_on_right(self.selection_dock, tab_with=anchor)
        try:
            self.status_info.setText("탁본 표면 보정 도구를 열었습니다. 기본 작업은 4축 작업 흐름 패널에서 이어집니다.")
        except Exception:
            pass

    def _show_advanced_panels(self) -> None:
        primary = [self.transform_dock, self.tile_dock, self.flatten_dock, self.export_dock]
        try:
            anchor = None
            for dock in primary:
                self._show_dock_on_right(dock, tab_with=anchor)
                if anchor is None:
                    anchor = dock
            self.transform_dock.raise_()
        except Exception:
            pass
        try:
            self.workflow_dock.raise_()
        except Exception:
            pass
        try:
            toolbar = getattr(self, "trans_toolbar", None)
            if toolbar is not None:
                toolbar.show()
        except Exception:
            pass
        try:
            self.status_info.setText("정위치/실측/탁본 세부 도구를 열었습니다. 기본 흐름은 오른쪽 4축 작업 패널에 남아 있습니다.")
        except Exception:
            pass

    def _sync_workflow_panel(self) -> None:
        panel = getattr(self, "workflow_panel", None)
        if panel is None:
            return

        obj = getattr(self.viewport, "selected_obj", None)
        if obj is None or getattr(obj, "mesh", None) is None:
            panel.update_state(has_object=False)
            return

        state = self._ensure_tile_interpretation_state(obj)
        try:
            selected_faces = len(getattr(obj, "selected_faces", set()) or set())
        except Exception:
            selected_faces = 0
        try:
            total_faces = int(getattr(getattr(obj, "mesh", None), "n_faces", 0) or 0)
        except Exception:
            total_faces = 0

        cam = getattr(self.viewport, "camera", None)
        canonical_view = None
        try:
            if cam is not None:
                canonical_view = _canonical_view_key_from_angles(
                    float(getattr(cam, "azimuth", 0.0) or 0.0),
                    float(getattr(cam, "elevation", 0.0) or 0.0),
                )
        except Exception:
            canonical_view = None

        tile_bits: list[str] = []
        if getattr(state, "tile_class", TileClass.UNKNOWN) != TileClass.UNKNOWN:
            tile_bits.append(state.tile_class.label_ko)
        if getattr(state, "split_scheme", SplitScheme.UNKNOWN) != SplitScheme.UNKNOWN:
            tile_bits.append(state.split_scheme.label_ko)
        if bool(getattr(getattr(state, "mandrel_fit", None), "is_defined", lambda: False)()):
            tile_bits.append("와통")
        tile_summary = " / ".join(tile_bits)

        wizard = self._tile_wizard_status(obj, state)
        panel.update_state(
            has_object=True,
            object_name=str(getattr(obj, "name", "") or "Object"),
            selected_faces=int(selected_faces),
            total_faces=int(total_faces),
            canonical_view=str(canonical_view or ""),
            record_view=str(getattr(state, "record_view", "") or ""),
            tile_summary=tile_summary,
            wizard_summary=str(wizard.get("summary", "") or ""),
            wizard_progress=int(wizard.get("progress", 0) or 0),
            wizard_next_label=str(wizard.get("next_label", "") or ""),
            wizard_next_enabled=bool(wizard.get("next_enabled", False)),
        )

    def on_workflow_action(self, action: str, data: object) -> None:
        if action == "open_mesh":
            self.open_file()
            return
        if action == "open_project":
            self.open_project()
            return
        if action == "fit_view":
            self.fit_view()
            return
        if action == "canonical_view":
            try:
                view_key = str((data or {}).get("view", "")).strip().lower()
            except Exception:
                view_key = ""
            if view_key:
                self._set_canonical_view(view_key)
            return
        if action == "run_interpretation_next":
            self.on_tile_interpretation_action("run_wizard_next", None)
            return
        if action == "show_section_tools":
            self.on_selection_action("open_section_tools", None)
            return
        if action == "show_measure_tools":
            self._show_measure_panel()
            return
        if action == "prepare_record_surface":
            self.on_tile_interpretation_action("prepare_record_surface", data)
            return
        if action == "select_visible_faces":
            self.on_selection_action("select_visible_faces", None)
            return
        if action == "preview_recording_surface":
            try:
                self.export_panel.set_rubbing_target("selected")
            except Exception:
                pass
            self.on_flatten_preview_requested()
            return
        if action == "unwrap_recording_surface":
            try:
                self.export_panel.set_rubbing_target("selected")
            except Exception:
                pass
            self.on_flatten_requested(self._current_flatten_panel_options(surface_target="selected"))
            return
        if action == "export_review_sheet":
            try:
                self.export_panel.set_rubbing_target("selected")
            except Exception:
                pass
            self.on_export_requested({"type": "review_sheet", "target": "selected"})
            return
        if action == "export_flat_svg":
            try:
                self.export_panel.set_rubbing_target("selected")
            except Exception:
                pass
            self.on_export_requested({"type": "flat_svg", "target": "selected"})
            return
        if action == "export_profile_package":
            self.on_export_requested({"type": "profile_2d_package"})
            return
        if action == "show_advanced_panels":
            self._show_advanced_panels()
            self.status_info.setText("정위치/실측용 도면/탁본 세부 도구를 열었습니다.")
            return
        if action == "show_selection_panel":
            self._show_selection_panel()
            return

    def _build_tile_scope_mesh(self, obj, *, mode: str):
        world_mesh = self._build_world_mesh(obj)
        selected_face_ids = _surface_target_face_ids(obj, "selected")
        use_selected = str(mode or "").strip().lower() != "mesh" and selected_face_ids.size > 0
        if use_selected:
            try:
                return world_mesh.extract_submesh(selected_face_ids), "현재 선택 표면", True
            except Exception:
                pass
        return world_mesh, "전체 메쉬", False

    def _prepare_tile_record_surface(self, *, view: str) -> str:
        view_key = str(view or "").strip().lower()
        if view_key not in {"top", "bottom"}:
            raise ValueError("기록면은 상면(top) 또는 하면(bottom)만 지원합니다.")

        export_panel = getattr(self, "export_panel", None)
        if export_panel is not None:
            try:
                export_panel.set_rubbing_target("selected")
            except Exception:
                pass

        obj = getattr(self.viewport, "selected_obj", None)
        if obj is not None and getattr(obj, "mesh", None) is not None:
            try:
                self._set_object_selected_faces(obj, [])
            except Exception:
                pass

        self.on_selection_action("select_visible_from_view", {"view": view_key})
        return "상면" if view_key == "top" else "하면"

    def _current_flatten_panel_options(self, *, surface_target: str) -> dict[str, Any]:
        return {
            "method": self.flatten_panel.combo_method.currentText(),
            "iterations": self.flatten_panel.spin_iterations.value(),
            "radius": self.flatten_panel.spin_radius.value(),
            "direction": self.flatten_panel.combo_direction.currentText(),
            "distortion": self.flatten_panel.slider_distortion.value() / 100.0,
            "auto_cut": self.flatten_panel.check_auto_cut.isChecked(),
            "multiband": self.flatten_panel.check_multiband.isChecked(),
            "boundary": "free",
            "initial": "lscm",
            "surface_target": _normalize_surface_target(surface_target),
        }

    @staticmethod
    def _flatten_strategy_suffix(options: dict[str, Any]) -> str:
        if not bool((options or {}).get("tile_guided", False)):
            return ""

        parts: list[str] = []
        record_view = str((options or {}).get("tile_record_view", "") or "").strip().lower()
        if record_view == "top":
            parts.append("상면")
        elif record_view == "bottom":
            parts.append("하면")

        guides = (options or {}).get("section_guides", None)
        if isinstance(guides, list) and guides:
            parts.append(f"단면 {len(guides)}개")
        if (options or {}).get("direction_override", None) is not None:
            parts.append("길이축")
        if (options or {}).get("radius_world_override", None) is not None:
            parts.append("와통 반경")

        if not parts:
            return " (기와 해석 기반)"
        return f" (기와 해석 기반: {', '.join(parts)})"

    def _single_surface_export_label(self, obj, target: str) -> str:
        normalized = _normalize_surface_target(target)
        if normalized != "selected":
            return f"{_surface_target_label(normalized)} 탁본"

        try:
            state = self._ensure_tile_interpretation_state(obj)
        except Exception:
            state = None

        record_view = str(getattr(state, "record_view", "") or "").strip().lower() if state is not None else ""
        if record_view == "top":
            return "상면 기록 탁본"
        if record_view == "bottom":
            return "하면 기록 탁본"
        return "선택 표면 탁본"

    @staticmethod
    def _tile_record_strategy_label(value: object) -> str:
        text = str(value or "").strip().lower()
        if text == "canonical_visible":
            return "표준 시점 가시면 자동 준비"
        if text:
            return text
        return ""

    @staticmethod
    def _review_rubbing_preset_for_options(options: dict[str, Any] | None) -> str:
        data = dict(options or {})
        if bool(data.get("tile_guided", False)) or str(data.get("tile_record_view", "") or "").strip():
            return "다중광(기록면)"
        return "자연(이미지)"

    def _selected_review_rubbing_preset(self, options: dict[str, Any] | None) -> str:
        try:
            export_panel = getattr(self, "export_panel", None)
            mode = export_panel.current_review_render_mode() if export_panel is not None else "auto"
        except Exception:
            mode = "auto"
        mode = str(mode or "auto").strip() or "auto"
        if mode == "auto":
            return self._review_rubbing_preset_for_options(options)
        return mode

    def _current_review_texture_options(self) -> dict[str, Any]:
        export_panel = getattr(self, "export_panel", None)
        if export_panel is None:
            return {
                "rubbing_detail_scale": 1.0,
                "rubbing_smooth_sigma_extra": 0.0,
                "rubbing_texture_postprocess": None,
            }

        try:
            detail_scale = float(export_panel.current_texture_detail_scale())
        except Exception:
            detail_scale = 1.0
        if not np.isfinite(detail_scale) or detail_scale <= 0.0:
            detail_scale = 1.0

        try:
            smooth_extra = float(export_panel.current_texture_smooth_extra())
        except Exception:
            smooth_extra = 0.0
        if not np.isfinite(smooth_extra) or smooth_extra <= 0.0:
            smooth_extra = 0.0

        try:
            postprocess = str(export_panel.current_texture_postprocess() or "").strip()
        except Exception:
            postprocess = ""

        return {
            "rubbing_detail_scale": float(detail_scale),
            "rubbing_smooth_sigma_extra": float(smooth_extra),
            "rubbing_texture_postprocess": (postprocess or None),
        }

    def _build_review_summary_context(
        self,
        obj,
        *,
        options: dict[str, Any],
        target_label: str,
        record_label: str,
        strategy_suffix: str,
        state_override: TileInterpretationState | None = None,
    ) -> dict[str, Any]:
        mode_label = "기와 해석 기반" if bool((options or {}).get("tile_guided", False)) else "일반 전개"
        guide_count = len(options.get("section_guides", [])) if isinstance(options.get("section_guides", None), list) else 0

        tile_class_label = ""
        split_scheme_label = ""
        record_strategy_label = ""
        mandrel_radius_world = None

        state = state_override
        if state is None:
            try:
                state = self._ensure_tile_interpretation_state(obj)
            except Exception:
                state = None

        if state is not None:
            tile_class = getattr(state, "tile_class", TileClass.UNKNOWN)
            split_scheme = getattr(state, "split_scheme", SplitScheme.UNKNOWN)
            if tile_class != TileClass.UNKNOWN:
                tile_class_label = tile_class.label_ko
            if split_scheme != SplitScheme.UNKNOWN:
                split_scheme_label = split_scheme.label_ko
            record_strategy_label = self._tile_record_strategy_label(getattr(state, "record_strategy", ""))

            mandrel_fit = getattr(state, "mandrel_fit", None)
            if mandrel_fit is not None and bool(getattr(mandrel_fit, "is_defined", lambda: False)()):
                try:
                    radius_value = float(getattr(mandrel_fit, "radius_world", None))
                except Exception:
                    radius_value = None
                if radius_value is not None and np.isfinite(radius_value) and radius_value > 0.0:
                    mandrel_radius_world = radius_value

        return {
            "record_label": str(record_label or ""),
            "target_label": str(target_label or ""),
            "strategy_suffix": str(strategy_suffix or ""),
            "mode_label": mode_label,
            "tile_class_label": tile_class_label,
            "split_scheme_label": split_scheme_label,
            "record_strategy_label": record_strategy_label,
            "guide_count": guide_count,
            "mandrel_radius_world": mandrel_radius_world,
        }

    @staticmethod
    def _slugify_filename_fragment(value: object, *, fallback: str) -> str:
        text = str(value or "").strip().lower()
        chars: list[str] = []
        for ch in text:
            if ch.isalnum():
                chars.append(ch)
            elif ch in {" ", "-", "_"}:
                chars.append("_")
        slug = "".join(chars).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug or str(fallback or "item")

    def _build_saved_slot_review_filename(self, obj, slot, *, extension: str = ".png") -> str:
        object_name = self._slugify_filename_fragment(getattr(obj, "name", "object"), fallback="object")
        slot_key = self._slugify_filename_fragment(getattr(slot, "slot_key", "slot"), fallback="slot")
        slot_label = self._slugify_filename_fragment(getattr(slot, "label", ""), fallback="")
        label_suffix = f".{slot_label}" if slot_label else ""
        return f"{object_name}.{slot_key}{label_suffix}.review{extension}"

    @staticmethod
    def _tile_section_guides(state: TileInterpretationState | None) -> list[dict[str, Any]]:
        if state is None:
            return []

        guides: list[dict[str, Any]] = []
        for item in list(getattr(state, "section_observations", []) or []):
            if not bool(getattr(item, "accepted", False)):
                continue
            try:
                station = float(getattr(item, "station", None))
            except Exception:
                continue
            if not np.isfinite(station):
                continue

            try:
                confidence = float(getattr(item, "confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0

            try:
                radius_world = getattr(item, "profile_radius_median_world", None)
                radius_value = float(radius_world) if radius_world is not None else None
            except Exception:
                radius_value = None
            if radius_value is not None and (not np.isfinite(radius_value) or radius_value <= 0.0):
                radius_value = None

            guides.append(
                {
                    "station": float(station),
                    "radius_world": radius_value,
                    "confidence": float(np.clip(confidence, 0.0, 1.0)),
                    "point_count": int(max(0, int(getattr(item, "profile_point_count", 0) or 0))),
                    "width_world": float(max(0.0, float(getattr(item, "profile_width_world", 0.0) or 0.0))),
                    "depth_world": float(max(0.0, float(getattr(item, "profile_depth_world", 0.0) or 0.0))),
                }
            )

        guides.sort(key=lambda item: float(item["station"]))
        return guides

    @staticmethod
    def _section_guides_signature(guides: object) -> tuple[object, ...] | None:
        if not isinstance(guides, list) or not guides:
            return None

        sig: list[object] = []
        for item in guides:
            if not isinstance(item, dict):
                continue
            try:
                station = float(item.get("station", None))
            except Exception:
                continue
            if not np.isfinite(station):
                continue
            radius_value = _safe_float_or_none(item.get("radius_world", None))
            confidence = _safe_float_or_none(item.get("confidence", 0.0))
            sig.append(
                (
                    float(np.round(station, 6)),
                    None if radius_value is None else float(np.round(radius_value, 6)),
                    0.0 if confidence is None else float(np.round(confidence, 4)),
                )
            )
        return tuple(sig) if sig else None

    def _resolve_flatten_options_with_state(
        self,
        obj,
        options: dict[str, Any],
        *,
        state: TileInterpretationState | None = None,
        selected_face_ids: np.ndarray | None = None,
    ) -> dict[str, Any]:
        resolved = dict(options or {})
        resolved["surface_target"] = _normalize_surface_target(resolved.get("surface_target", "all"))

        if obj is None:
            return resolved

        if state is None:
            try:
                state = self._ensure_tile_interpretation_state(obj)
            except Exception:
                return resolved

        record_view = str(getattr(state, "record_view", "") or "").strip().lower()
        if record_view not in {"top", "bottom"}:
            return resolved

        resolved["tile_guided"] = True
        resolved["tile_record_view"] = record_view
        resolved["tile_record_strategy"] = str(getattr(state, "record_strategy", "") or "canonical_visible")

        if selected_face_ids is None:
            selected_face_ids = _surface_target_face_ids(obj, "selected")
        else:
            selected_face_ids = np.asarray(selected_face_ids, dtype=np.int32).reshape(-1)
        if selected_face_ids.size > 0:
            resolved["surface_target"] = "selected"

        # Tile mode prefers the fabrication-aware unwrap path by default.
        resolved["method"] = "단면 기반 펼침 (기와)"
        resolved["initial"] = "section"

        axis_hint = getattr(state, "axis_hint", None)
        if axis_hint is not None and bool(getattr(axis_hint, "is_defined", lambda: False)()):
            resolved["direction_override"] = tuple(axis_hint.vector_world or ())

        mandrel_fit = getattr(state, "mandrel_fit", None)
        if mandrel_fit is not None and bool(getattr(mandrel_fit, "is_defined", lambda: False)()):
            try:
                radius_world = float(getattr(mandrel_fit, "radius_world", None))
            except Exception:
                radius_world = None
            if radius_world is not None and np.isfinite(radius_world) and radius_world > 0.0:
                resolved["radius_world_override"] = float(radius_world)

        section_guides = self._tile_section_guides(state)
        if section_guides:
            resolved["section_guides"] = section_guides

        return resolved

    def _resolve_flatten_options(self, obj, options: dict[str, Any]) -> dict[str, Any]:
        return self._resolve_flatten_options_with_state(obj, options)

    @staticmethod
    def _estimate_pca_axis_hint(mesh, *, face_ids: np.ndarray, source: AxisSource, note: str) -> AxisHint:
        vertices = np.asarray(getattr(mesh, "vertices", None), dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(getattr(mesh, "faces", None), dtype=np.int32).reshape(-1, 3)
        if vertices.shape[0] < 3 or faces.shape[0] <= 0:
            raise ValueError("축을 추정할 메쉬 데이터가 충분하지 않습니다.")

        if face_ids.size > 0:
            valid = face_ids[(face_ids >= 0) & (face_ids < faces.shape[0])]
            if valid.size <= 0:
                raise ValueError("선택된 면이 없어 길이축을 추정할 수 없습니다.")
            vertex_ids = np.unique(faces[valid].reshape(-1))
            face_count = int(valid.size)
        else:
            vertex_ids = np.arange(vertices.shape[0], dtype=np.int32)
            face_count = int(faces.shape[0])

        points = vertices[vertex_ids]
        finite_mask = np.isfinite(points).all(axis=1)
        points = points[finite_mask]
        if points.shape[0] < 3:
            raise ValueError("축을 추정할 점이 충분하지 않습니다.")

        origin = np.mean(points, axis=0)
        centered = points - origin
        cov = centered.T @ centered
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = np.asarray(evals[order], dtype=np.float64)
        axis = np.asarray(evecs[:, order[0]], dtype=np.float64).reshape(3)

        anchor = int(np.argmax(np.abs(axis)))
        if float(axis[anchor]) < 0.0:
            axis = -axis

        denom = float(max(evals[0], 1e-12))
        confidence = float(np.clip((evals[0] - evals[1]) / denom, 0.0, 1.0))
        return AxisHint(
            source=source,
            vector_world=(float(axis[0]), float(axis[1]), float(axis[2])),
            origin_world=(float(origin[0]), float(origin[1]), float(origin[2])),
            confidence=confidence,
            face_count=face_count,
            note=note,
        )

    @staticmethod
    def _points_for_face_subset(mesh, face_ids: np.ndarray) -> tuple[np.ndarray, int]:
        vertices = np.asarray(getattr(mesh, "vertices", None), dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(getattr(mesh, "faces", None), dtype=np.int32).reshape(-1, 3)
        if vertices.shape[0] < 3 or faces.shape[0] <= 0:
            raise ValueError("단면 후보를 계산할 메쉬 데이터가 충분하지 않습니다.")

        if face_ids.size > 0:
            valid = face_ids[(face_ids >= 0) & (face_ids < faces.shape[0])]
            if valid.size <= 0:
                raise ValueError("선택된 면이 없어 단면 후보를 계산할 수 없습니다.")
            vertex_ids = np.unique(faces[valid].reshape(-1))
            face_count = int(valid.size)
        else:
            vertex_ids = np.arange(vertices.shape[0], dtype=np.int32)
            face_count = int(faces.shape[0])

        points = np.asarray(vertices[vertex_ids], dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] < 3:
            raise ValueError("단면 후보를 계산할 점이 충분하지 않습니다.")
        return points, face_count

    @staticmethod
    def _section_candidates_from_axis(
        mesh,
        *,
        axis_hint: AxisHint,
        face_ids: np.ndarray,
        quantiles: list[float],
        note_prefix: str,
        confidence_scale: float,
    ) -> list[SectionObservation]:
        if not axis_hint.is_defined():
            raise ValueError("먼저 길이축 힌트를 저장해 주세요.")

        points, _face_count = MainWindow._points_for_face_subset(mesh, face_ids)
        axis_vec = np.asarray(axis_hint.vector_world, dtype=np.float64).reshape(3)
        axis_norm = float(np.linalg.norm(axis_vec))
        if axis_norm <= 1e-12 or not np.isfinite(axis_norm):
            raise ValueError("길이축 벡터가 유효하지 않습니다.")
        axis_vec = axis_vec / axis_norm

        axis_origin = np.asarray(axis_hint.origin_world or np.mean(points, axis=0), dtype=np.float64).reshape(3)
        projections = (points - axis_origin) @ axis_vec
        finite_proj = projections[np.isfinite(projections)]
        if finite_proj.size <= 0:
            raise ValueError("길이축 투영값을 계산할 수 없습니다.")

        candidates: list[SectionObservation] = []
        for q in quantiles:
            station = float(np.quantile(finite_proj, float(np.clip(q, 0.0, 1.0))))
            plane_origin = axis_origin + (axis_vec * station)
            candidates.append(
                SectionObservation(
                    station=station,
                    origin_world=(float(plane_origin[0]), float(plane_origin[1]), float(plane_origin[2])),
                    normal_world=(float(axis_vec[0]), float(axis_vec[1]), float(axis_vec[2])),
                    confidence=float(np.clip(float(axis_hint.confidence) * float(confidence_scale), 0.0, 1.0)),
                    accepted=True,
                    note=f"{note_prefix} q={float(q):.2f}",
                )
            )
        return candidates

    @staticmethod
    def _merge_section_observations(
        existing: list[SectionObservation],
        incoming: list[SectionObservation],
    ) -> list[SectionObservation]:
        merged = list(existing or [])
        stations = [item.station for item in merged if item.station is not None]
        if stations:
            span = float(max(stations) - min(stations))
        else:
            span = 0.0
        station_tol = max(1e-4, span * 0.02)

        for item in incoming:
            station = item.station
            replaced = False
            if station is not None:
                for idx, prev in enumerate(merged):
                    prev_station = prev.station
                    if prev_station is None:
                        continue
                    if abs(float(prev_station) - float(station)) <= station_tol:
                        if float(item.confidence) >= float(prev.confidence):
                            merged[idx] = item
                        replaced = True
                        break
            if not replaced:
                merged.append(item)

        merged.sort(key=lambda obs: float(obs.station) if obs.station is not None else 0.0)
        return merged

    @staticmethod
    def _mark_all_sections(
        sections: list[SectionObservation],
        *,
        accepted: bool,
    ) -> list[SectionObservation]:
        updated: list[SectionObservation] = []
        for item in sections or []:
            updated.append(
                SectionObservation(
                    station=item.station,
                    origin_world=item.origin_world,
                    normal_world=item.normal_world,
                    confidence=item.confidence,
                    accepted=bool(accepted),
                    profile_contour_count=item.profile_contour_count,
                    profile_point_count=item.profile_point_count,
                    profile_width_world=item.profile_width_world,
                    profile_depth_world=item.profile_depth_world,
                    profile_center_world=item.profile_center_world,
                    profile_radius_median_world=item.profile_radius_median_world,
                    profile_radius_iqr_world=item.profile_radius_iqr_world,
                    profile_fit_rmse_world=item.profile_fit_rmse_world,
                    profile_arc_span_deg=item.profile_arc_span_deg,
                    profile_fit_confidence=item.profile_fit_confidence,
                    note=item.note,
                )
            )
        return updated

    @staticmethod
    def _mark_middle_sections(
        sections: list[SectionObservation],
        *,
        keep_count: int,
    ) -> list[SectionObservation]:
        items = list(sections or [])
        if not items:
            return []

        indexed = [
            (idx, item) for idx, item in enumerate(items) if item.station is not None
        ]
        if not indexed:
            return MainWindow._mark_all_sections(items, accepted=True)

        indexed.sort(key=lambda pair: float(pair[1].station))
        keep_count = max(1, min(int(keep_count), len(indexed)))
        start = max(0, (len(indexed) - keep_count) // 2)
        keep_ids = {indexed[i][0] for i in range(start, start + keep_count)}

        updated: list[SectionObservation] = []
        for idx, item in enumerate(items):
            updated.append(
                SectionObservation(
                    station=item.station,
                    origin_world=item.origin_world,
                    normal_world=item.normal_world,
                    confidence=item.confidence,
                    accepted=(idx in keep_ids),
                    profile_contour_count=item.profile_contour_count,
                    profile_point_count=item.profile_point_count,
                    profile_width_world=item.profile_width_world,
                    profile_depth_world=item.profile_depth_world,
                    profile_center_world=item.profile_center_world,
                    profile_radius_median_world=item.profile_radius_median_world,
                    profile_radius_iqr_world=item.profile_radius_iqr_world,
                    profile_fit_rmse_world=item.profile_fit_rmse_world,
                    profile_arc_span_deg=item.profile_arc_span_deg,
                    profile_fit_confidence=item.profile_fit_confidence,
                    note=item.note,
                )
            )
        return updated

    @staticmethod
    def _plane_basis_from_normal(normal_world: tuple[float, float, float] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normal = np.asarray(normal_world, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12 or not np.isfinite(norm):
            raise ValueError("단면 법선이 유효하지 않습니다.")
        normal = normal / norm
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(normal, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(ref, normal)
        u_norm = float(np.linalg.norm(u))
        if u_norm <= 1e-12:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            u = np.cross(ref, normal)
            u_norm = float(np.linalg.norm(u))
        if u_norm <= 1e-12 or not np.isfinite(u_norm):
            raise ValueError("단면 기저를 만들 수 없습니다.")
        u = u / u_norm
        v = np.cross(normal, u)
        v_norm = float(np.linalg.norm(v))
        if v_norm <= 1e-12 or not np.isfinite(v_norm):
            raise ValueError("단면 기저를 만들 수 없습니다.")
        v = v / v_norm
        return u, v, normal

    @staticmethod
    def _analyze_section_profiles(
        mesh,
        *,
        axis_hint: AxisHint,
        section_observations: list[SectionObservation],
    ) -> list[SectionObservation]:
        if not axis_hint.is_defined():
            raise ValueError("먼저 길이축 힌트를 저장해 주세요.")

        from src.core.mesh_slicer import MeshSlicer

        axis_vec = np.asarray(axis_hint.vector_world, dtype=np.float64).reshape(3)
        axis_norm = float(np.linalg.norm(axis_vec))
        if axis_norm <= 1e-12 or not np.isfinite(axis_norm):
            raise ValueError("길이축 벡터가 유효하지 않습니다.")
        axis_vec = axis_vec / axis_norm

        mesh_vertices = np.asarray(getattr(mesh, "vertices", None), dtype=np.float64).reshape(-1, 3)
        if mesh_vertices.shape[0] < 3:
            raise ValueError("단면 프로파일을 분석할 메쉬 데이터가 충분하지 않습니다.")
        axis_origin = np.asarray(axis_hint.origin_world or np.mean(mesh_vertices, axis=0), dtype=np.float64).reshape(3)

        slicer = MeshSlicer(mesh.to_trimesh())
        updated: list[SectionObservation] = []

        for item in section_observations or []:
            origin = item.origin_world
            if origin is None and item.station is not None:
                plane_origin = axis_origin + (axis_vec * float(item.station))
                origin = (float(plane_origin[0]), float(plane_origin[1]), float(plane_origin[2]))
            normal = item.normal_world or axis_hint.vector_world

            profile_contour_count = 0
            profile_point_count = 0
            profile_width_world = 0.0
            profile_depth_world = 0.0
            profile_center_world = None
            profile_radius_median_world = None
            profile_radius_iqr_world = 0.0
            profile_fit_rmse_world = 0.0
            profile_arc_span_deg = 0.0
            profile_fit_confidence = 0.0

            try:
                if origin is not None and normal is not None:
                    origin_arr = np.asarray(origin, dtype=np.float64).reshape(3)
                    u_axis, v_axis, n_axis = MainWindow._plane_basis_from_normal(normal)
                    contours_local = slicer.slice_with_plane(origin_arr.tolist(), n_axis.tolist())

                    contour_pts: list[np.ndarray] = []
                    for contour in contours_local or []:
                        arr = np.asarray(contour, dtype=np.float64)
                        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
                            continue
                        arr = arr[:, :3]
                        arr = arr[np.isfinite(arr).all(axis=1)]
                        if arr.shape[0] < 2:
                            continue
                        contour_pts.append(arr)

                    if contour_pts:
                        profile_contour_count = len(contour_pts)
                        best_fit = None
                        best_points_uv = None
                        best_score = -1.0
                        all_points_uv: list[np.ndarray] = []

                        for contour in contour_pts:
                            rel = contour - origin_arr
                            u_vals = rel @ u_axis
                            v_vals = rel @ v_axis
                            uv = np.column_stack([u_vals, v_vals]).astype(np.float64, copy=False)
                            uv = uv[np.isfinite(uv).all(axis=1)]
                            if uv.shape[0] < 3:
                                continue
                            all_points_uv.append(uv)
                            fit = fit_circle_2d(uv)
                            score = float(fit.confidence) * max(float(fit.used_points), 1.0)
                            if fit.is_defined() and score > best_score:
                                best_fit = fit
                                best_points_uv = uv
                                best_score = score

                        if best_fit is None and all_points_uv:
                            stacked = np.vstack(all_points_uv)
                            fallback_fit = fit_circle_2d(stacked, min_points=6)
                            if fallback_fit.is_defined():
                                best_fit = fallback_fit
                                best_points_uv = stacked

                        if best_fit is not None and best_points_uv is not None:
                            q05_u, q95_u = np.quantile(best_points_uv[:, 0], [0.05, 0.95])
                            q05_v, q95_v = np.quantile(best_points_uv[:, 1], [0.05, 0.95])
                            profile_point_count = int(max(best_fit.used_points, best_points_uv.shape[0]))
                            profile_width_world = float(max(0.0, q95_u - q05_u))
                            profile_depth_world = float(max(0.0, q95_v - q05_v))
                            profile_radius_median_world = float(best_fit.radius or 0.0)
                            profile_radius_iqr_world = float(max(0.0, best_fit.radius_iqr))
                            profile_fit_rmse_world = float(max(0.0, best_fit.rmse))
                            profile_arc_span_deg = float(max(0.0, best_fit.arc_span_deg))
                            profile_fit_confidence = float(np.clip(best_fit.confidence, 0.0, 1.0))
                            if best_fit.center_xy is not None:
                                cu, cv = best_fit.center_xy
                                center_world = origin_arr + (u_axis * float(cu)) + (v_axis * float(cv))
                                profile_center_world = (
                                    float(center_world[0]),
                                    float(center_world[1]),
                                    float(center_world[2]),
                                )
                        elif all_points_uv:
                            stacked = np.vstack(all_points_uv)
                            radii = np.linalg.norm(stacked, axis=1)
                            radii = radii[np.isfinite(radii)]
                            if radii.size > 0:
                                profile_point_count = int(radii.size)
                                q05_u, q95_u = np.quantile(stacked[:, 0], [0.05, 0.95])
                                q05_v, q95_v = np.quantile(stacked[:, 1], [0.05, 0.95])
                                q25_r, q50_r, q75_r = np.quantile(radii, [0.25, 0.50, 0.75])
                                profile_width_world = float(max(0.0, q95_u - q05_u))
                                profile_depth_world = float(max(0.0, q95_v - q05_v))
                                profile_radius_median_world = float(q50_r)
                                profile_radius_iqr_world = float(max(0.0, q75_r - q25_r))
            except Exception:
                pass

            updated.append(
                SectionObservation(
                    station=item.station,
                    origin_world=origin,
                    normal_world=normal,
                    confidence=item.confidence,
                    accepted=item.accepted,
                    profile_contour_count=profile_contour_count,
                    profile_point_count=profile_point_count,
                    profile_width_world=profile_width_world,
                    profile_depth_world=profile_depth_world,
                    profile_center_world=profile_center_world,
                    profile_radius_median_world=profile_radius_median_world,
                    profile_radius_iqr_world=profile_radius_iqr_world,
                    profile_fit_rmse_world=profile_fit_rmse_world,
                    profile_arc_span_deg=profile_arc_span_deg,
                    profile_fit_confidence=profile_fit_confidence,
                    note=item.note,
                )
            )

        return updated

    @staticmethod
    def _fit_mandrel_from_sections(
        mesh,
        *,
        axis_hint: AxisHint,
        section_observations: list[SectionObservation],
        face_ids: np.ndarray,
        scope: str,
    ) -> MandrelFitResult:
        if not axis_hint.is_defined():
            raise ValueError("먼저 길이축 힌트를 저장해 주세요.")

        accepted = [item for item in (section_observations or []) if bool(item.accepted) and item.station is not None]
        if not accepted:
            raise ValueError("먼저 채택된 대표 단면 후보를 만들어 주세요.")

        profile_ready = [
            item for item in accepted
            if item.profile_radius_median_world is not None and int(item.profile_point_count or 0) > 0
        ]
        if profile_ready:
            centered_ready = [
                item
                for item in profile_ready
                if item.profile_center_world is not None and float(getattr(item, "profile_fit_confidence", 0.0) or 0.0) > 0.15
            ]
            radius_values = np.asarray(
                [float(item.profile_radius_median_world) for item in profile_ready],
                dtype=np.float64,
            )
            spread_values = np.asarray(
                [float(max(0.0, item.profile_radius_iqr_world)) for item in profile_ready],
                dtype=np.float64,
            )
            radius_world = float(np.median(radius_values))
            section_spread = float(
                np.quantile(radius_values, 0.75) - np.quantile(radius_values, 0.25)
            ) if radius_values.size > 1 else 0.0
            radius_spread_world = float(
                max(section_spread, float(np.median(spread_values)) if spread_values.size > 0 else 0.0)
            )
            rel_spread = radius_spread_world / max(abs(radius_world), 1e-6)
            consistency = float(np.clip(1.0 - (rel_spread * 3.0), 0.0, 1.0))
            coverage = float(np.clip(len(profile_ready) / max(3, len(accepted)), 0.0, 1.0))
            confidence = float(
                np.clip(
                    (float(axis_hint.confidence) * 0.45) + (consistency * 0.35) + (coverage * 0.20),
                    0.0,
                    1.0,
                )
            )
            axis_vec = np.asarray(axis_hint.vector_world, dtype=np.float64).reshape(3)
            axis_vec = axis_vec / max(float(np.linalg.norm(axis_vec)), 1e-12)
            axis_origin = np.asarray(axis_hint.origin_world or np.zeros(3, dtype=np.float64), dtype=np.float64).reshape(3)
            if centered_ready:
                origin_candidates = []
                for item in centered_ready:
                    center_world = np.asarray(item.profile_center_world, dtype=np.float64).reshape(3)
                    station = float(item.station or 0.0)
                    origin_candidates.append(center_world - (axis_vec * station))
                if origin_candidates:
                    origin_arr = np.vstack(origin_candidates)
                    axis_origin = np.median(origin_arr, axis=0)
            return MandrelFitResult(
                radius_world=radius_world,
                radius_spread_world=radius_spread_world,
                axis_origin_world=(float(axis_origin[0]), float(axis_origin[1]), float(axis_origin[2])),
                axis_vector_world=(float(axis_vec[0]), float(axis_vec[1]), float(axis_vec[2])),
                confidence=confidence,
                used_sections=len(profile_ready),
                used_points=sum(int(item.profile_point_count or 0) for item in profile_ready),
                scope=str(scope or ""),
                note=f"{scope} 기준 단면 프로파일 기반 공통 반경 후보",
            )

        points, _face_count = MainWindow._points_for_face_subset(mesh, face_ids)
        axis_vec = np.asarray(axis_hint.vector_world, dtype=np.float64).reshape(3)
        axis_norm = float(np.linalg.norm(axis_vec))
        if axis_norm <= 1e-12 or not np.isfinite(axis_norm):
            raise ValueError("길이축 벡터가 유효하지 않습니다.")
        axis_vec = axis_vec / axis_norm
        axis_origin = np.asarray(axis_hint.origin_world or np.mean(points, axis=0), dtype=np.float64).reshape(3)

        projections = (points - axis_origin) @ axis_vec
        finite_proj = projections[np.isfinite(projections)]
        if finite_proj.size <= 0:
            raise ValueError("단면 투영값을 계산할 수 없습니다.")

        proj_q05 = float(np.quantile(finite_proj, 0.05))
        proj_q95 = float(np.quantile(finite_proj, 0.95))
        proj_span = max(abs(proj_q95 - proj_q05), 1e-6)
        band_half = max(proj_span * 0.03, 1e-4)

        per_section_radius: list[float] = []
        per_section_spread: list[float] = []
        used_sections = 0
        used_points = 0

        for obs in accepted:
            station = float(obs.station)
            delta = np.abs(projections - station)
            idx = np.flatnonzero(delta <= band_half)
            if idx.size < 12:
                order = np.argsort(delta)
                take = min(max(24, int(points.shape[0] * 0.08)), int(points.shape[0]))
                idx = order[:take]
            sample = points[idx]
            if sample.shape[0] < 6:
                continue

            rel = sample - axis_origin
            axial = rel @ axis_vec
            radial_vec = rel - np.outer(axial, axis_vec)
            radii = np.linalg.norm(radial_vec, axis=1)
            radii = radii[np.isfinite(radii)]
            if radii.size < 6:
                continue

            q25, q50, q75 = np.quantile(radii, [0.25, 0.5, 0.75])
            per_section_radius.append(float(q50))
            per_section_spread.append(float(max(0.0, q75 - q25)))
            used_sections += 1
            used_points += int(radii.size)

        if not per_section_radius:
            raise ValueError("단면 후보 주변에서 반경을 추정할 점을 충분히 찾지 못했습니다.")

        radius_values = np.asarray(per_section_radius, dtype=np.float64)
        spread_values = np.asarray(per_section_spread, dtype=np.float64)
        radius_world = float(np.median(radius_values))
        section_spread = float(np.quantile(radius_values, 0.75) - np.quantile(radius_values, 0.25)) if radius_values.size > 1 else 0.0
        radius_spread_world = float(max(section_spread, float(np.median(spread_values)) if spread_values.size > 0 else 0.0))

        rel_spread = radius_spread_world / max(abs(radius_world), 1e-6)
        consistency = float(np.clip(1.0 - (rel_spread * 3.0), 0.0, 1.0))
        coverage = float(np.clip(used_sections / max(3, len(accepted)), 0.0, 1.0))
        confidence = float(
            np.clip(
                (float(axis_hint.confidence) * 0.45) + (consistency * 0.35) + (coverage * 0.20),
                0.0,
                1.0,
            )
        )

        return MandrelFitResult(
            radius_world=radius_world,
            radius_spread_world=radius_spread_world,
            axis_origin_world=(float(axis_origin[0]), float(axis_origin[1]), float(axis_origin[2])),
            axis_vector_world=(float(axis_vec[0]), float(axis_vec[1]), float(axis_vec[2])),
            confidence=confidence,
            used_sections=used_sections,
            used_points=used_points,
            scope=str(scope or ""),
            note=f"{scope} 기준 공통 반경 후보",
        )

    def on_tile_interpretation_action(self, action: str, data: object) -> None:
        if action == "generate_synthetic_tile":
            try:
                preset = str((data or {}).get("preset", "sugkiwa_quarter") or "sugkiwa_quarter")
            except Exception:
                preset = "sugkiwa_quarter"
            try:
                seed = int((data or {}).get("seed", 1) or 1)
            except Exception:
                seed = 1
            try:
                spec = self._synthetic_tile_spec_from_preset(preset, seed=seed)
                artifact = generate_synthetic_tile(spec)
                self._add_synthetic_tile_artifact(artifact)
                self.status_info.setText(
                    f"합성 기와 생성: {artifact.name} "
                    f"({spec.tile_class.label_ko}, {spec.split_scheme.label_ko}, seed {int(spec.seed)})"
                )
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "기와 해석",
                    self._format_error_message("합성 기와를 생성하지 못했습니다:", f"{type(e).__name__}: {e}"),
                )
            return
        if action == "export_synthetic_benchmark_suite":
            seeds_arg = "1"
            pass_threshold = 0.9
            try:
                seeds_arg = str((data or {}).get("seeds", "1") or "1").strip() or "1"
            except Exception:
                seeds_arg = "1"
            try:
                pass_threshold = float((data or {}).get("pass_threshold", 0.9) or 0.9)
            except Exception:
                pass_threshold = 0.9
            output_dir = QFileDialog.getExistingDirectory(
                self,
                "합성 benchmark suite 저장 폴더 선택",
                "",
            )
            if not output_dir:
                return

            def task_export_synthetic_suite():
                seeds: list[int] = []
                for token in str(seeds_arg or "1").split(","):
                    token = str(token or "").strip()
                    if not token:
                        continue
                    seeds.append(int(token))
                if not seeds:
                    seeds = [1]
                report = save_synthetic_benchmark_suite(
                    output_dir,
                    seeds=tuple(seeds),
                    include_review_sheets=True,
                    review_dpi=int(self.export_panel.spin_dpi.value()) if hasattr(self, "export_panel") else DEFAULT_EXPORT_DPI,
                    pass_threshold=pass_threshold,
                )
                return report.to_dict()

            def on_done_export_synthetic_suite(result: Any):
                report = (
                    SyntheticBenchmarkSuiteReport.from_dict(result)
                    if isinstance(result, dict)
                    else SyntheticBenchmarkSuiteReport()
                )
                summary = (
                    self._synthetic_suite_summary(report)
                    if report.case_count > 0
                    else "Synthetic benchmark suite를 생성하지 않았습니다."
                )
                self.label_synthetic_suite_summary.setText(summary)
                self.status_info.setText(
                    f"✅ synthetic benchmark suite {report.case_count}건 저장 완료"
                )
                fail_text = (
                    "\n실패 케이스는 label 또는 summary 파일을 확인하세요."
                    if int(report.fail_count or 0) > 0
                    else "\n모든 케이스가 기준 점수를 통과했습니다."
                )
                QMessageBox.information(
                    self,
                    "합성 benchmark suite 저장",
                    (
                        f"synthetic benchmark suite 저장 완료\n\n"
                        f"케이스 수: {report.case_count}\n"
                        f"평균 점수: {report.average_score * 100.0:.1f} / 100\n"
                        f"기준 점수: {report.pass_threshold * 100.0:.0f} / 100\n"
                        f"통과/실패: {report.pass_count}/{report.fail_count}\n"
                        f"폴더: {output_dir}"
                        f"{fail_text}"
                    ),
                )

            def on_failed_export_synthetic_suite(message: str):
                self.status_info.setText("❌ synthetic benchmark suite 저장 실패")
                QMessageBox.critical(
                    self,
                    "오류",
                    self._format_error_message("synthetic benchmark suite 저장 중 오류 발생:", message),
                )

            self._start_task(
                title="합성 benchmark",
                label=f"synthetic benchmark suite 생성/저장 중... ({seeds_arg})",
                thread=TaskThread("export_synthetic_benchmark_suite", task_export_synthetic_suite),
                on_done=on_done_export_synthetic_suite,
                on_failed=on_failed_export_synthetic_suite,
            )
            return

        obj = getattr(self.viewport, "selected_obj", None)
        if obj is None or getattr(obj, "mesh", None) is None:
            QMessageBox.warning(self, "경고", "먼저 메쉬를 선택해 주세요.")
            return

        state = self._ensure_tile_interpretation_state(obj)

        try:
            if action == "set_tile_class":
                state.tile_class = TileClass.from_value(data)
                if str(state.workflow_stage or "") in {"", "hypothesis"}:
                    state.workflow_stage = "hypothesis"
            elif action == "set_split_scheme":
                state.split_scheme = SplitScheme.from_value(data)
                if str(state.workflow_stage or "") in {"", "hypothesis"}:
                    state.workflow_stage = "hypothesis"
            elif action == "clear_axis":
                state.axis_hint = AxisHint()
                state.section_observations = []
                state.mandrel_fit = MandrelFitResult()
                state.workflow_stage = "hypothesis"
                self.status_info.setText("기와 해석: 길이축 힌트를 초기화했습니다.")
            elif action == "estimate_axis":
                mode = ""
                try:
                    mode = str((data or {}).get("mode", "")).strip().lower()
                except Exception:
                    mode = ""
                use_selected = mode == "selected"
                world_mesh = self._build_world_mesh(obj)
                face_ids = _surface_target_face_ids(obj, "selected") if use_selected else np.zeros((0,), dtype=np.int32)
                axis_hint = self._estimate_pca_axis_hint(
                    world_mesh,
                    face_ids=face_ids,
                    source=(AxisSource.SELECTED_PATCH_PCA if use_selected else AxisSource.FULL_MESH_PCA),
                    note=("현재 선택 표면 패치 기반 PCA" if use_selected else "전체 메쉬 기반 PCA"),
                )
                state.axis_hint = axis_hint
                state.section_observations = []
                state.mandrel_fit = MandrelFitResult()
                state.workflow_stage = "axis_hint"
                self.status_info.setText(
                    f"기와 해석: {axis_hint.source.label_ko} 저장 "
                    f"(신뢰도 {axis_hint.confidence * 100.0:.0f}%)"
                )
            elif action == "add_section_candidate":
                mode = ""
                try:
                    mode = str((data or {}).get("mode", "")).strip().lower()
                except Exception:
                    mode = ""
                use_selected = mode == "selected"
                world_mesh = self._build_world_mesh(obj)
                face_ids = _surface_target_face_ids(obj, "selected") if use_selected else np.zeros((0,), dtype=np.int32)
                candidates = self._section_candidates_from_axis(
                    world_mesh,
                    axis_hint=state.axis_hint,
                    face_ids=face_ids,
                    quantiles=[0.5],
                    note_prefix=("현재 선택 중심 단면" if use_selected else "전체 메쉬 중심 단면"),
                    confidence_scale=0.9 if use_selected else 0.8,
                )
                state.section_observations = self._merge_section_observations(
                    list(state.section_observations or []),
                    candidates,
                )
                state.mandrel_fit = MandrelFitResult()
                state.workflow_stage = "section_candidates"
                self.status_info.setText(
                    f"기와 해석: 대표 단면 후보 {len(candidates)}개 추가 "
                    f"(총 {len(state.section_observations)}개)"
                )
            elif action == "auto_section_candidates":
                mode = ""
                try:
                    mode = str((data or {}).get("mode", "")).strip().lower()
                except Exception:
                    mode = ""
                use_selected = mode != "mesh"
                world_mesh = self._build_world_mesh(obj)
                face_ids = _surface_target_face_ids(obj, "selected") if use_selected else np.zeros((0,), dtype=np.int32)
                try:
                    count = int((data or {}).get("count", 5) or 5)
                except Exception:
                    count = 5
                count = max(3, min(9, count))
                quantiles = np.linspace(0.15, 0.85, count, dtype=np.float64).tolist()
                candidates = self._section_candidates_from_axis(
                    world_mesh,
                    axis_hint=state.axis_hint,
                    face_ids=face_ids,
                    quantiles=quantiles,
                    note_prefix=("현재 선택 대표 단면" if use_selected else "전체 메쉬 대표 단면"),
                    confidence_scale=0.8 if use_selected else 0.7,
                )
                state.section_observations = self._merge_section_observations(
                    list(state.section_observations or []),
                    candidates,
                )
                state.mandrel_fit = MandrelFitResult()
                state.workflow_stage = "section_candidates"
                self.status_info.setText(
                    f"기와 해석: 대표 단면 후보 {len(candidates)}개 자동 제안 "
                    f"(총 {len(state.section_observations)}개)"
                )
            elif action == "clear_sections":
                state.section_observations = []
                state.mandrel_fit = MandrelFitResult()
                if state.axis_hint.is_defined():
                    state.workflow_stage = "axis_hint"
                else:
                    state.workflow_stage = "hypothesis"
                self.status_info.setText("기와 해석: 대표 단면 후보를 초기화했습니다.")
            elif action == "analyze_section_profiles":
                mode = ""
                try:
                    mode = str((data or {}).get("mode", "")).strip().lower()
                except Exception:
                    mode = ""
                scope_mesh, scope_label, _used_selected = self._build_tile_scope_mesh(obj, mode=mode or "selected_preferred")
                state.section_observations = self._analyze_section_profiles(
                    scope_mesh,
                    axis_hint=state.axis_hint,
                    section_observations=list(state.section_observations or []),
                )
                state.mandrel_fit = MandrelFitResult()
                analyzed_count = sum(
                    1 for item in state.section_observations if int(item.profile_point_count or 0) > 0
                )
                if analyzed_count > 0:
                    state.workflow_stage = "section_profiles"
                self.status_info.setText(
                    f"기와 해석: 단면 프로파일 {analyzed_count}개 분석 완료 ({scope_label})"
                )
            elif action == "accept_all_sections":
                state.section_observations = self._mark_all_sections(
                    list(state.section_observations or []),
                    accepted=True,
                )
                state.mandrel_fit = MandrelFitResult()
                if state.section_observations:
                    state.workflow_stage = "section_candidates"
                self.status_info.setText(
                    f"기와 해석: 대표 단면 후보 {len(state.section_observations)}개를 모두 채택했습니다."
                )
            elif action == "accept_middle_sections":
                try:
                    keep_count = int((data or {}).get("count", 3) or 3)
                except Exception:
                    keep_count = 3
                state.section_observations = self._mark_middle_sections(
                    list(state.section_observations or []),
                    keep_count=keep_count,
                )
                state.mandrel_fit = MandrelFitResult()
                accepted_count = sum(1 for item in state.section_observations if bool(item.accepted))
                if state.section_observations:
                    state.workflow_stage = "section_candidates"
                self.status_info.setText(
                    f"기와 해석: 중앙 단면 {accepted_count}개를 우선 채택했습니다."
                )
            elif action == "fit_mandrel":
                mode = ""
                try:
                    mode = str((data or {}).get("mode", "")).strip().lower()
                except Exception:
                    mode = ""
                world_mesh, scope_label, _used_selected = self._build_tile_scope_mesh(obj, mode=mode or "selected_preferred")
                state.section_observations = self._analyze_section_profiles(
                    world_mesh,
                    axis_hint=state.axis_hint,
                    section_observations=list(state.section_observations or []),
                )
                fit_result = self._fit_mandrel_from_sections(
                    world_mesh,
                    axis_hint=state.axis_hint,
                    section_observations=list(state.section_observations or []),
                    face_ids=np.zeros((0,), dtype=np.int32),
                    scope=scope_label,
                )
                state.mandrel_fit = fit_result
                state.workflow_stage = "mandrel_fit"
                self.status_info.setText(
                    f"기와 해석: 와통 반경 후보 {float(fit_result.radius_world):.3f} "
                    f"({fit_result.scope}, 후보 {int(fit_result.used_sections)}개)"
                )
            elif action == "clear_mandrel_fit":
                state.mandrel_fit = MandrelFitResult()
                if state.section_observations:
                    state.workflow_stage = "section_candidates"
                elif state.axis_hint.is_defined():
                    state.workflow_stage = "axis_hint"
                else:
                    state.workflow_stage = "hypothesis"
                self.status_info.setText("기와 해석: 와통 초벌 피팅 결과를 초기화했습니다.")
            elif action == "prepare_record_surface":
                view_key = ""
                try:
                    view_key = str((data or {}).get("view", "")).strip().lower()
                except Exception:
                    view_key = ""
                label = self._prepare_tile_record_surface(view=view_key)
                state.record_view = view_key
                state.record_strategy = "canonical_visible"
                state.workflow_stage = "record_surface_pending"
                self.status_info.setText(
                    f"기와 해석: {label} 기록면 자동 준비 중 (내부적으로 현재 선택 사용)"
                )
            elif action == "clear_record_surface":
                state.record_view = ""
                state.record_strategy = ""
                if state.mandrel_fit.is_defined():
                    state.workflow_stage = "mandrel_fit"
                elif state.section_observations:
                    analyzed_count = sum(
                        1 for item in state.section_observations if int(getattr(item, "profile_point_count", 0) or 0) > 0
                    )
                    state.workflow_stage = "section_profiles" if analyzed_count > 0 else "section_candidates"
                elif state.axis_hint.is_defined():
                    state.workflow_stage = "axis_hint"
                else:
                    state.workflow_stage = "hypothesis"
                try:
                    obj.selected_faces = set()
                except Exception:
                    pass
                try:
                    self.selection_panel.update_selection_count(0)
                except Exception:
                    pass
                try:
                    self.viewport.faceSelectionChanged.emit(0)
                except Exception:
                    pass
                try:
                    self.viewport.update()
                except Exception:
                    pass
                self.status_info.setText("기와 해석: 기록면 준비를 해제했습니다.")
            elif action == "save_slot":
                try:
                    slot_index = int((data or {}).get("slot", 1) or 1)
                except Exception:
                    slot_index = 1
                slot_key = self._tile_slot_key(slot_index)
                selected_faces = sorted(list(getattr(obj, "selected_faces", set()) or set()))
                label = self._build_tile_slot_label(
                    state,
                    slot_index=slot_index,
                    selected_face_count=len(selected_faces),
                )
                slot = state.save_slot(
                    slot_key=slot_key,
                    label=label,
                    selected_faces=selected_faces,
                )
                self.status_info.setText(
                    f"기와 해석: 슬롯 {slot_index} 저장 완료 "
                    f"({slot.summary_label()})"
                )
            elif action == "load_slot":
                try:
                    slot_index = int((data or {}).get("slot", 1) or 1)
                except Exception:
                    slot_index = 1
                slot_key = self._tile_slot_key(slot_index)
                restored_state, selected_faces = state.restore_slot(slot_key)
                restored_count = self._set_object_selected_faces(obj, selected_faces)
                state = restored_state
                self.status_info.setText(
                    f"기와 해석: 슬롯 {slot_index} 복원 완료 "
                    f"(선택 {restored_count}면)"
                )
            elif action == "clear_slots":
                cleared = len(list(getattr(state, "saved_slots", []) or []))
                state.clear_slots()
                self.status_info.setText(
                    f"기와 해석: 작업 슬롯 {cleared}개를 모두 비웠습니다."
                )
            elif action == "export_saved_slots_review":
                saved_slots = [
                    type(item).from_dict(item.to_dict())
                    for item in list(getattr(state, "saved_slots", []) or [])
                    if str(getattr(item, "slot_key", "") or "").strip()
                ]
                if not saved_slots:
                    QMessageBox.warning(self, "기와 해석", "먼저 저장된 작업 슬롯을 하나 이상 만들어 주세요.")
                    return

                output_dir = QFileDialog.getExistingDirectory(
                    self,
                    "저장 슬롯 검토 시트 저장 폴더 선택",
                    "",
                )
                if not output_dir:
                    return

                dpi = int(self.export_panel.spin_dpi.value()) if hasattr(self, "export_panel") else DEFAULT_EXPORT_DPI
                include_scale = bool(self.export_panel.check_scale_bar.isChecked()) if hasattr(self, "export_panel") else True
                base_options = self._current_flatten_panel_options(surface_target="all")
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
                output_dir_path = Path(output_dir)

                def task_export_saved_slots():
                    from src.core.recording_surface_review import (
                        RecordingSurfaceReviewOptions,
                        build_recording_surface_summary_lines,
                        render_recording_surface_review,
                    )

                    results: list[str] = []
                    for slot in saved_slots:
                        slot_state = slot.to_state()
                        face_ids = np.asarray(getattr(slot, "selected_faces", []) or [], dtype=np.int32).reshape(-1)
                        opts = self._resolve_flatten_options_with_state(
                            obj,
                            dict(base_options),
                            state=slot_state,
                            selected_face_ids=face_ids,
                        )
                        slot_target = _normalize_surface_target(opts.get("surface_target", "all"))

                        mesh = MainWindow._build_world_mesh_from_transform(
                            base, translation=translation, rotation=rotation, scale=scale
                        )
                        if slot_target != "all" and face_ids.size > 0:
                            mesh = mesh.extract_submesh(face_ids)
                        flattened = MainWindow._compute_flattened_mesh(mesh, opts)

                        slot_target_label = _surface_target_label(slot_target)
                        record_label = MainWindow._flatten_preview_record_label(opts, slot_target_label)
                        strategy_suffix = MainWindow._flatten_strategy_suffix(opts)
                        review_context = self._build_review_summary_context(
                            obj,
                            options=opts,
                            target_label=slot_target_label,
                            record_label=record_label,
                            strategy_suffix=strategy_suffix,
                            state_override=slot_state,
                        )
                        slot_desc = str(getattr(slot, "label", "") or getattr(slot, "slot_key", "") or "").strip()
                        summary_lines = build_recording_surface_summary_lines(
                            flattened,
                            **review_context,
                            extra_lines=((f"작업 슬롯: {slot_desc}",) if slot_desc else ()),
                        )
                        review = render_recording_surface_review(
                            flattened,
                            options=RecordingSurfaceReviewOptions(
                                dpi=dpi,
                                width_pixels=1600,
                                rubbing_preset=self._selected_review_rubbing_preset(opts),
                                **self._current_review_texture_options(),
                                title=f"기록면 검토 시트 - {record_label}",
                                summary_lines=summary_lines,
                                show_scale_bar=include_scale,
                            ),
                        )
                        save_path = output_dir_path / self._build_saved_slot_review_filename(obj, slot)
                        review.combined_image.save(save_path)
                        results.append(str(save_path))
                    return results

                def on_done_export_saved_slots(result: Any):
                    paths = list(result or []) if isinstance(result, list) else []
                    count = len(paths)
                    if count <= 0:
                        QMessageBox.information(self, "완료", "저장된 슬롯 검토 시트를 생성하지 않았습니다.")
                        self.status_info.setText("ℹ️ 저장된 슬롯 검토 시트 없음")
                        return
                    QMessageBox.information(
                        self,
                        "완료",
                        f"저장된 슬롯 검토 시트 {count}개를 저장했습니다.\n\n폴더: {output_dir}",
                    )
                    self.status_info.setText(f"✅ 슬롯 검토 시트 {count}개 저장 완료")

                def on_failed_export_saved_slots(message: str):
                    self.status_info.setText("❌ 슬롯 검토 시트 저장 실패")
                    QMessageBox.critical(
                        self,
                        "오류",
                        self._format_error_message("저장 슬롯 검토 시트 저장 중 오류 발생:", message),
                    )

                self._start_task(
                    title="내보내기",
                    label=f"저장 슬롯 검토 시트 {len(saved_slots)}개 생성/저장 중...",
                    thread=TaskThread("export_saved_slots_review", task_export_saved_slots),
                    on_done=on_done_export_saved_slots,
                    on_failed=on_failed_export_saved_slots,
                )
            elif action == "evaluate_against_truth":
                truth = self._coerce_synthetic_truth(getattr(obj, "tile_synthetic_truth", None))
                if truth is None:
                    raise ValueError("현재 메쉬에는 연결된 합성 정답이 없습니다.")
                report = evaluate_tile_interpretation(state, truth)
                setattr(obj, "tile_evaluation_report", report)
                unit = str(getattr(getattr(obj, "mesh", None), "unit", "") or "mm")
                self.status_info.setText(
                    f"기와 해석 평가: {report.overall_score * 100.0:.0f}점 "
                    f"(반경 오차 {report.mandrel_radius_abs_error_world if report.mandrel_radius_abs_error_world is not None else 'n/a'} {unit})"
                )
            elif action == "apply_synthetic_truth_hypothesis":
                truth = self._coerce_synthetic_truth(getattr(obj, "tile_synthetic_truth", None))
                if truth is None:
                    raise ValueError("현재 메쉬에는 연결된 합성 정답이 없습니다.")
                restored = TileInterpretationState.from_dict(truth.ground_truth_state.to_dict())
                restored.saved_slots = [type(item).from_dict(item.to_dict()) for item in list(state.saved_slots or [])]
                restored.note = "synthetic_truth_applied"
                restored.touch()
                state = restored
                restored_count = self._set_object_selected_faces(obj, truth.selected_faces)
                report = evaluate_tile_interpretation(state, truth)
                setattr(obj, "tile_evaluation_report", report)
                self.status_info.setText(
                    f"기와 해석: 합성 정답 가설 적용 완료 (선택 {restored_count}면, 점수 {report.overall_score * 100.0:.0f})"
                )
            elif action == "export_synthetic_bundle":
                truth = self._coerce_synthetic_truth(getattr(obj, "tile_synthetic_truth", None))
                if truth is None:
                    raise ValueError("현재 메쉬에는 연결된 합성 정답이 없습니다.")
                default_name = str(getattr(obj, "name", "") or truth.mesh_name or "synthetic_tile")
                filepath, _ = QFileDialog.getSaveFileName(
                    self,
                    "합성 벤치마크 저장",
                    f"{default_name}.obj",
                    "Wavefront OBJ (*.obj);;PLY (*.ply);;STL (*.stl)",
                )
                if not filepath:
                    return
                report = evaluate_tile_interpretation(state, truth)
                setattr(obj, "tile_evaluation_report", report)
                artifact = SyntheticTileArtifact(
                    mesh=obj.mesh,
                    truth=truth,
                    name=str(getattr(obj, "name", "") or truth.mesh_name or "synthetic_tile"),
                )
                saved_paths = save_synthetic_tile_bundle(
                    artifact,
                    filepath,
                    interpretation_state=state,
                    evaluation_report=report,
                )
                self.status_info.setText(
                    f"합성 벤치마크 저장 완료: {Path(saved_paths.get('bundle', filepath)).name}"
                )
            elif action == "run_wizard_next":
                wizard = self._tile_wizard_status(obj, state)
                next_action = wizard.get("next_action")
                next_data = wizard.get("next_data")
                if not wizard.get("next_enabled", False) or not next_action:
                    raise ValueError(str(wizard.get("summary", "") or "현재 단계에서 더 진행할 자동 작업이 없습니다."))
                self.on_tile_interpretation_action(str(next_action), next_data)
                return
            elif action == "run_wizard_all":
                executed_steps: list[str] = []
                for _ in range(12):
                    wizard = self._tile_wizard_status(obj, state)
                    next_action = wizard.get("next_action")
                    next_data = wizard.get("next_data")
                    if not wizard.get("next_enabled", False) or not next_action:
                        break
                    executed_steps.append(str(next_action))
                    self.on_tile_interpretation_action(str(next_action), next_data)
                    obj = getattr(self.viewport, "selected_obj", None)
                    if obj is None or getattr(obj, "mesh", None) is None:
                        break
                    state = self._ensure_tile_interpretation_state(obj)
                if not executed_steps:
                    raise ValueError("위저드를 자동 진행할 수 없습니다. 유형/분할 가설부터 확인하세요.")
                truth = self._coerce_synthetic_truth(getattr(obj, "tile_synthetic_truth", None))
                if truth is not None:
                    report = evaluate_tile_interpretation(state, truth)
                    setattr(obj, "tile_evaluation_report", report)
                    self.status_info.setText(
                        f"기와 위저드 자동 진행 완료 ({len(executed_steps)}단계, 평가 {report.overall_score * 100.0:.0f}점)"
                    )
                else:
                    self.status_info.setText(
                        f"기와 위저드 자동 진행 완료 ({len(executed_steps)}단계)"
                    )
                self._sync_tile_panel()
                return
            else:
                return
        except Exception as e:
            QMessageBox.warning(
                self,
                "기와 해석",
                self._format_error_message("기와 해석 상태를 갱신하지 못했습니다:", f"{type(e).__name__}: {e}"),
            )
            return

        state.touch()
        setattr(obj, "tile_interpretation_state", state)
        self._sync_tile_panel()

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
                anchor = self.tile_dock if self.tile_dock.isVisible() else None
            except Exception:
                anchor = None
            try:
                self._show_dock_on_right(self.section_dock, tab_with=anchor)
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

        if action == "select_visible_from_view":
            view = ""
            try:
                view = str((data or {}).get("view", "")).strip().lower()
            except Exception:
                view = ""
            if view not in CANONICAL_VIEW_PRESETS:
                QMessageBox.warning(self, "경고", "표준 시점 정보를 확인할 수 없습니다.")
                return
            try:
                combo = getattr(getattr(self, "export_panel", None), "combo_rubbing_target", None)
                if combo is not None:
                    combo.setCurrentIndex(1)  # 현재 선택
            except Exception:
                pass
            try:
                modifiers = QApplication.keyboardModifiers()
            except Exception:
                modifiers = Qt.KeyboardModifier.NoModifier
            try:
                self._set_canonical_view(view)
                self.viewport.repaint()
                self.viewport.select_visible_faces_in_view(modifiers=modifiers)
                self.viewport.setFocus()
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "오류",
                    self._format_error_message(
                        "표준 시점 가시면 선택 중 오류 발생:",
                        f"{type(e).__name__}: {e}",
                    ),
                )
            return

        if action == "select_visible_faces":
            try:
                combo = getattr(getattr(self, "export_panel", None), "combo_rubbing_target", None)
                if combo is not None:
                    combo.setCurrentIndex(1)  # 현재 선택
            except Exception:
                pass
            try:
                modifiers = QApplication.keyboardModifiers()
            except Exception:
                modifiers = Qt.KeyboardModifier.NoModifier
            try:
                self.viewport.select_visible_faces_in_view(modifiers=modifiers)
                self.viewport.setFocus()
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "오류",
                    self._format_error_message("현재 시점 가시면 선택 중 오류 발생:", f"{type(e).__name__}: {e}"),
                )
            return

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
                    f"✅ 표면 라벨 자동 적용({method_used}): outer {len(obj.outer_face_indices):,} / inner {len(obj.inner_face_indices):,} / migu {len(obj.migu_face_indices):,} (현재 메쉬에 저장됨)"
                )
                try:
                    self.viewport._emit_surface_assignment_changed(obj)
                except Exception:
                    pass
                QMessageBox.information(
                    self,
                    "완료",
                    f"표면 라벨 자동 적용 결과를 현재 메쉬에 반영했습니다. (파일 저장은 아직 하지 않았습니다.)\n\n"
                    f"- outer(외면): {len(obj.outer_face_indices):,} faces\n"
                    f"- inner(내면): {len(obj.inner_face_indices):,} faces\n"
                    f"- migu(미구): {len(obj.migu_face_indices):,} faces\n\n"
                    f"- method: {method_used}\n\n"
                    f"표시: 외면=파랑, 내면=주황 오버레이\n"
                    f"권장: 외면/내면 구분이 꼭 필요하지 않다면, 선택 패널에서 가시면을 고른 뒤 '현재 선택'으로 저장하세요.",
                )
            except Exception as e:
                QMessageBox.critical(self, "오류", f"표면 라벨 자동 적용 실패:\n{e}")
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
        options = self._resolve_flatten_options(obj, options)
        method = str(options.get('method', 'ARAP')).strip()
        iterations = int(options.get('iterations', 30))
        boundary = str(options.get('boundary', 'free')).strip()
        initial = str(options.get('initial', 'lscm')).strip()
        distortion = float(options.get("distortion", 0.5))
        auto_cut = bool(options.get("auto_cut", False))
        multiband = bool(options.get("multiband", False))
        surface_target = _normalize_surface_target(options.get("surface_target", "all"))
        face_signature = None
        if surface_target != "all":
            face_signature = _face_index_signature(_surface_target_face_ids(obj, surface_target))

        radius_world_override = options.get("radius_world_override", None)
        if radius_world_override is None:
            radius_key: object = ("mm", float(np.round(float(options.get("radius", 0.0)), 6)))
        else:
            radius_key = ("world", float(np.round(float(radius_world_override), 6)))

        direction_value = options.get("direction_override", options.get("direction", "auto"))
        try:
            axis_arr = np.asarray(direction_value, dtype=np.float64).reshape(-1)
            if axis_arr.size >= 3 and np.isfinite(axis_arr[:3]).all():
                axis_vec = axis_arr[:3].astype(np.float64, copy=True)
                nrm = float(np.linalg.norm(axis_vec))
                if np.isfinite(nrm) and nrm > 1e-12:
                    axis_vec = axis_vec / nrm
                direction_key: object = tuple(np.round(axis_vec[:3], 6).tolist())
            else:
                direction_key = str(direction_value or "auto").strip()
        except Exception:
            direction_key = str(direction_value or "auto").strip()

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
            radius_key,
            direction_key,
            auto_cut,
            multiband,
            surface_target,
            face_signature,
            bool(options.get("tile_guided", False)),
            str(options.get("tile_record_view", "") or ""),
            self._section_guides_signature(options.get("section_guides", None)),
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
        direction = options.get("direction_override", options.get("direction", "auto"))

        def normalize_method(text: str) -> str:
            t = str(text or "").strip().lower()
            if "arap" in t:
                return "arap"
            if "lscm" in t:
                return "lscm"
            if ("면적" in text) or ("area" in t):
                return "area"
            if ("단면" in text) or ("기와" in text) or ("section" in t) or ("tile" in t):
                return "section"
            if ("원통" in text) or ("cyl" in t):
                return "cylinder"
            return "arap"

        # FlattenPanel의 radius는 mm 입력이므로, mesh.unit 기준으로 world 단위로 환산
        radius_world = options.get("radius_world_override", None)
        if radius_world is None:
            radius_world = mm_to_mesh_units(radius_mm, getattr(mesh, "unit", None))
        else:
            radius_world = float(radius_world)

        return flatten_with_method(
            mesh,
            method=normalize_method(method),
            iterations=iterations,
            distortion=distortion,
            boundary_type=boundary_type,
            initial_method=initial,
            cylinder_axis=direction,
            cylinder_radius=radius_world,
            section_guides=options.get("section_guides", None),
            section_record_view=options.get("tile_record_view", None),
        )

    def _compute_flattened(self, obj, options: dict[str, Any]):
        options = self._resolve_flatten_options(obj, options)
        mesh = self._build_world_mesh(obj)
        surface_target = _normalize_surface_target(options.get("surface_target", "all"))
        if surface_target != "all":
            face_ids = _surface_target_face_ids(obj, surface_target)
            if face_ids.size <= 0:
                raise ValueError(f"No faces are assigned for surface target '{surface_target}'.")
            mesh = mesh.extract_submesh(face_ids)
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

        options = dict(options)
        surface_target = (
            self.export_panel.current_rubbing_target() if hasattr(self, "export_panel") else "all"
        )
        surface_target = _normalize_surface_target(surface_target)
        options["surface_target"] = surface_target
        options = self._resolve_flatten_options(obj, options)

        surface_target = _normalize_surface_target(options.get("surface_target", surface_target))
        target_label = _surface_target_label(surface_target)
        target_face_ids = _surface_target_face_ids(obj, surface_target)
        if surface_target != "all" and target_face_ids.size <= 0:
            if surface_target == "selected":
                body = (
                    "현재 선택된 면이 없습니다.\n\n"
                    "브러시/올가미/경계 도구로 펼칠 표면을 먼저 선택한 뒤 다시 시도하세요."
                )
            else:
                body = (
                    f"'{target_label}' 지정이 비어 있습니다.\n\n"
                    "우측 '표면 선택/지정'에서 먼저 영역을 지정하거나,\n"
                    "내보내기 패널 대상을 '전체 메쉬' 또는 '현재 선택'으로 바꿔 다시 시도하세요."
                )
            QMessageBox.warning(self, "경고", body)
            return

        key = self._flatten_cache_key(obj, options)
        cached = self._flattened_cache.get(key)
        strategy_suffix = self._flatten_strategy_suffix(options)
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
        face_ids = target_face_ids.copy()

        def task():
            mesh = MainWindow._build_world_mesh_from_transform(
                base, translation=translation, rotation=rotation, scale=scale
            )
            if surface_target != "all":
                mesh = mesh.extract_submesh(face_ids)
            flattened = MainWindow._compute_flattened_mesh(mesh, options_copy)
            return {"key": key, "flattened": flattened}

        status_target = f" ({target_label})" if surface_target != "all" else ""
        self.status_info.setText(f"🗺️ 기록면 전개 중{status_target}{strategy_suffix}...")
        self._start_task(
            title="기록면 전개",
            label=f"기록면 전개 중{status_target}{strategy_suffix}...",
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
            self.status_info.setText("❌ 기록면 전개 실패")
            QMessageBox.critical(self, "오류", self._format_error_message("기록면 전개 실패:", "Recording-surface unwrap result is empty."))
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

        status_prefix = "⚠️ 기록면 전개 완료" if size_warning else "✅ 기록면 전개 완료"
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
            "기록면 전개 완료",
            f"기록면 전개가 완료되었습니다.\n\n"
            f"- 크기: {flattened.width:.2f} x {flattened.height:.2f} {flattened.original_mesh.unit}\n"
            f"- 왜곡(평균/최대): {flattened.mean_distortion:.1%} / {flattened.max_distortion:.1%}"
            f"{size_note}\n\n"
            f"이 결과는 삼각형을 분해한 것이 아니라, 선택된 기록면을 연속 좌표계로 전개한 결과입니다.\n"
            f"이제 '기록면 전개 SVG 저장' 또는 '탁본 이미지 내보내기'를 사용할 수 있습니다."
        )

    def _on_flatten_task_failed(self, message: str):
        self.status_info.setText("❌ 기록면 전개 실패")
        QMessageBox.critical(self, "오류", self._format_error_message("기록면 전개 중 오류 발생:", message))

    @staticmethod
    def _flatten_preview_record_label(options: dict[str, Any], target_label: str) -> str:
        record_view = str((options or {}).get("tile_record_view", "") or "").strip().lower()
        if record_view == "top":
            return "상면 기록면"
        if record_view == "bottom":
            return "하면 기록면"
        return target_label

    @staticmethod
    def _pixmap_from_pil_image(image: Image.Image) -> QPixmap:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        pixmap = QPixmap()
        if not pixmap.loadFromData(buffer.getvalue(), "PNG"):
            raise RuntimeError("미리보기 이미지를 QPixmap으로 변환하지 못했습니다.")
        return pixmap

    def on_flatten_preview_requested(self):
        obj = self.viewport.selected_obj
        if not obj or not getattr(obj, "mesh", None):
            QMessageBox.warning(self, "경고", "먼저 메쉬를 선택하세요.")
            return

        target = self.export_panel.current_rubbing_target() if hasattr(self, "export_panel") else "all"
        options = self._current_flatten_panel_options(surface_target=target)
        options = self._resolve_flatten_options(obj, options)
        target = _normalize_surface_target(options.get("surface_target", target))
        target_label = _surface_target_label(target)
        record_label = self._flatten_preview_record_label(options, target_label)
        target_face_ids = _surface_target_face_ids(obj, target)
        strategy_suffix = self._flatten_strategy_suffix(options)

        if target != "all" and target_face_ids.size <= 0:
            if target == "selected":
                body = (
                    "현재 선택된 면이 없습니다.\n\n"
                    "표준 시점 버튼이나 가시면 선택으로 먼저 기록면을 준비한 뒤 다시 시도하세요."
                )
            else:
                body = (
                    f"'{target_label}' 지정이 비어 있습니다.\n\n"
                    "대상을 '전체 메쉬' 또는 '현재 선택'으로 바꾸거나,\n"
                    "표면 선택/지정에서 먼저 영역을 지정해 주세요."
                )
            QMessageBox.warning(self, "경고", body)
            return

        try:
            flattened = self._get_or_compute_flattened(obj, options)
        except Exception as e:
            QMessageBox.critical(
                self,
                "오류",
                self._format_error_message("기록면 미리보기 생성 중 오류 발생:", f"{type(e).__name__}: {e}"),
            )
            return

        try:
            from src.core.recording_surface_review import (
                build_recording_surface_summary_lines,
                RecordingSurfaceReviewOptions,
                render_recording_surface_review,
            )

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                review_context = self._build_review_summary_context(
                    obj,
                    options=options,
                    target_label=target_label,
                    record_label=record_label,
                    strategy_suffix=strategy_suffix,
                )
                review = render_recording_surface_review(
                    flattened,
                    options=RecordingSurfaceReviewOptions(
                        dpi=int(self.export_panel.spin_dpi.value()) if hasattr(self, "export_panel") else DEFAULT_EXPORT_DPI,
                        width_pixels=1600,
                        rubbing_preset=self._selected_review_rubbing_preset(options),
                        **self._current_review_texture_options(),
                        title=f"기록면 전개 미리보기 - {record_label}",
                        summary_lines=build_recording_surface_summary_lines(
                            flattened,
                            **review_context,
                            extra_lines=("왼쪽은 연속 탁본형 기록면, 오른쪽은 외곽 확인용 뷰입니다.",),
                        ),
                    ),
                )
            finally:
                QApplication.restoreOverrideCursor()

            pixmap_rubbing = self._pixmap_from_pil_image(review.rubbing_image)
            pixmap_outline = self._pixmap_from_pil_image(review.outline_image)
        except Exception as e:
            QMessageBox.critical(
                self,
                "오류",
                self._format_error_message("기록면 미리보기 렌더링 중 오류 발생:", f"{type(e).__name__}: {e}"),
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"기록면 전개 미리보기 - {record_label}")
        dialog.resize(1320, 900)

        layout = QVBoxLayout(dialog)
        info = QLabel(
            f"기록면: {record_label} | 대상: {target_label}{strategy_suffix}\n"
            f"왼쪽은 연속 탁본형 기록면, 오른쪽은 외곽 확인용 뷰입니다.\n"
            f"둘 다 삼각형 와이어프레임이 아니라 기록면 전개 결과를 읽기 쉽게 보여주기 위한 미리보기입니다."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 11px; color: #2d3748;")
        layout.addWidget(info)

        preview_row = QHBoxLayout()

        def _make_preview_panel(title: str, pixmap: QPixmap) -> QWidget:
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.setSpacing(6)

            title_label = QLabel(title)
            title_label.setStyleSheet("font-weight: bold; color: #2c5282;")
            panel_layout.addWidget(title_label)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setPixmap(pixmap)
            scroll.setWidget(image_label)
            panel_layout.addWidget(scroll, 1)
            return panel

        preview_row.addWidget(_make_preview_panel("연속 탁본형 기록면", pixmap_rubbing), 1)
        preview_row.addWidget(_make_preview_panel("외곽 확인", pixmap_outline), 1)
        layout.addLayout(preview_row, 1)

        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignRight)

        dialog.exec()

    def on_export_requested(self, data):
        """내보내기 요청 처리"""
        export_type = data.get('type')
        requested_target = (data or {}).get(
            "target",
            self.export_panel.current_rubbing_target() if hasattr(self, "export_panel") else "all",
        )
        retired_export_types = {
            "rubbing",
            "rubbing_digital",
            "rubbing_view_cyl",
            "ortho",
            "sheet_svg",
            "sheet_svg_digital",
            "mesh_outer",
            "mesh_inner",
            "mesh_flat",
        }
        if export_type in retired_export_types:
            QMessageBox.information(
                self,
                "기본 워크플로우에서 제거됨",
                "이 출력 방식은 기본 고고학 워크플로우에서 제거되었습니다.\n\n"
                "대신 '기록면 검토 시트 저장', '기록면 전개 SVG 저장', "
                "또는 '6방향 도면 패키지 내보내기'를 사용하세요.",
            )
            try:
                self.status_info.setText("기본 워크플로우에서 제거된 내보내기 방식입니다.")
            except Exception:
                pass
            return
        target = _normalize_surface_target(requested_target)
        requested_target_normalized = target
        
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

        flatten_options = self._current_flatten_panel_options(surface_target=target)
        flatten_options = self._resolve_flatten_options(obj, flatten_options)
        target = _normalize_surface_target(flatten_options.get("surface_target", target))
        if target != requested_target_normalized and hasattr(self, "export_panel"):
            try:
                self.export_panel.set_rubbing_target(target)
            except Exception:
                pass

        target_label = _surface_target_label(target)
        target_face_ids = _surface_target_face_ids(obj, target)
        strategy_suffix = self._flatten_strategy_suffix(flatten_options)

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

        def _ensure_recording_surface_ready(action_label: str) -> bool:
            if target == "all" or target_face_ids.size > 0:
                return True
            if target == "selected":
                body = (
                    "현재 선택된 면이 없습니다.\n\n"
                    f"표준 시점 버튼이나 가시면 선택으로 먼저 {action_label} 기록면을 준비한 뒤 다시 시도하세요."
                )
            else:
                body = (
                    f"'{target_label}' 지정이 비어 있습니다.\n\n"
                    "대상을 '전체 메쉬' 또는 '현재 선택'으로 바꾸거나,\n"
                    "표면 선택/지정에서 먼저 영역을 지정해 주세요."
                )
            QMessageBox.warning(self, "경고", body)
            return False

        if export_type == 'review_sheet':
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "기록면 검토 시트 저장",
                "recording_surface_review.png",
                "PNG (*.png);;TIFF (*.tiff)",
            )
            if not filepath:
                return
            if not _ensure_recording_surface_ready("검토 시트를 만들"):
                return

            flatten_options_target = dict(flatten_options)
            flatten_options_target["surface_target"] = target
            key = self._flatten_cache_key(obj, flatten_options_target)
            cached_flat = self._flattened_cache.get(key)
            opts = dict(flatten_options_target)
            record_label = self._flatten_preview_record_label(flatten_options_target, target_label)
            review_context = self._build_review_summary_context(
                obj,
                options=flatten_options_target,
                target_label=target_label,
                record_label=record_label,
                strategy_suffix=strategy_suffix,
            )

            def task_export_review_sheet():
                from src.core.recording_surface_review import (
                    RecordingSurfaceReviewOptions,
                    build_recording_surface_summary_lines,
                    render_recording_surface_review,
                )

                if cached_flat is not None:
                    flattened = cached_flat
                else:
                    mesh = MainWindow._build_world_mesh_from_transform(
                        base, translation=translation, rotation=rotation, scale=scale
                    )
                    if target != "all":
                        mesh = mesh.extract_submesh(target_face_ids)
                    flattened = MainWindow._compute_flattened_mesh(mesh, opts)

                review = render_recording_surface_review(
                    flattened,
                    options=RecordingSurfaceReviewOptions(
                        dpi=int(self.export_panel.spin_dpi.value()) if hasattr(self, "export_panel") else DEFAULT_EXPORT_DPI,
                        width_pixels=1600,
                        rubbing_preset=self._selected_review_rubbing_preset(flatten_options_target),
                        **self._current_review_texture_options(),
                        title=f"기록면 검토 시트 - {record_label}",
                        summary_lines=build_recording_surface_summary_lines(
                            flattened,
                            **review_context,
                        ),
                    ),
                )
                review.combined_image.save(filepath)
                return {"path": filepath, "key": key, "flattened": flattened if cached_flat is None else None}

            def on_done_export_review_sheet(result: Any):
                if isinstance(result, dict):
                    flat = result.get("flattened")
                    if flat is not None:
                        self._flattened_cache[key] = flat
                QMessageBox.information(self, "완료", f"기록면 검토 시트가 저장되었습니다:\n{filepath}")
                self.status_info.setText(f"✅ 저장 완료: {Path(filepath).name}")

            def on_failed(message: str):
                self.status_info.setText("❌ 저장 실패")
                QMessageBox.critical(self, "오류", self._format_error_message("기록면 검토 시트 저장 중 오류 발생:", message))

            self._start_task(
                title="내보내기",
                label=f"기록면 검토 시트 생성/저장 중{strategy_suffix}...",
                thread=TaskThread("export_review_sheet", task_export_review_sheet),
                on_done=on_done_export_review_sheet,
                on_failed=on_failed,
            )
            return

        if export_type == 'flat_svg':
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "기록면 전개 SVG 저장",
                "flattened.svg",
                "Scalable Vector Graphics (*.svg)",
            )
            if not filepath:
                return
            if not _ensure_recording_surface_ready("전개 SVG를 만들"):
                return

            flatten_options_target = dict(flatten_options)
            flatten_options_target["surface_target"] = target
            key = self._flatten_cache_key(obj, flatten_options_target)
            cached_flat = self._flattened_cache.get(key)
            opts = dict(flatten_options_target)

            def task_export_flat_svg():
                from src.core.flattened_svg_exporter import FlattenedSVGExporter, SVGExportOptions

                if cached_flat is not None:
                    flattened = cached_flat
                else:
                    mesh = MainWindow._build_world_mesh_from_transform(
                        base, translation=translation, rotation=rotation, scale=scale
                    )
                    if target != "all":
                        mesh = mesh.extract_submesh(target_face_ids)
                    flattened = MainWindow._compute_flattened_mesh(mesh, opts)

                exporter = FlattenedSVGExporter()
                unit = (flattened.original_mesh.unit or DEFAULT_MESH_UNIT).lower()
                svg_unit = unit if unit in ("mm", "cm") else DEFAULT_MESH_UNIT
                grid = 10.0 if svg_unit == "mm" else 1.0

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
                QMessageBox.information(self, "완료", f"기록면 전개 SVG가 저장되었습니다:\n{filepath}")
                self.status_info.setText(f"✅ 저장 완료: {Path(filepath).name}")

            def on_failed(message: str):
                self.status_info.setText("❌ 저장 실패")
                QMessageBox.critical(self, "오류", self._format_error_message("SVG 저장 중 오류 발생:", message))

            self._start_task(
                title="내보내기",
                label=f"기록면 전개 계산/SVG 저장 중{strategy_suffix}...",
                thread=TaskThread("export_flat_svg", task_export_flat_svg),
                on_done=on_done_export_flat_svg,
                on_failed=on_failed,
            )
            return

        QMessageBox.information(
            self,
            "지원되지 않는 출력",
            "현재 기본 워크플로우에서는 이 출력 방식을 사용하지 않습니다.\n\n"
            "실측용 도면 SVG, 기록면 검토 시트, 6방향 도면 패키지를 사용해 주세요.",
        )
        try:
            self.status_info.setText("기본 워크플로우에 없는 출력 요청입니다.")
        except Exception:
            pass
    
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
        view_map = {k: CANONICAL_VIEW_PRESETS[k] for k in views}

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
        self.viewport._canonical_view_key = None
        self.viewport.camera.reset()
        self.viewport.update()

    def fit_view(self):
        self.viewport._front_back_ortho_enabled = False
        self.viewport._canonical_view_key = None
        obj = self.viewport.selected_obj
        if obj:
            try:
                wb = np.asarray(obj.get_world_bounds(), dtype=np.float64)
                if wb.shape == (2, 3) and np.isfinite(wb).all():
                    self.viewport.camera.fit_to_bounds(wb)
                    self.viewport.camera.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                    self.viewport.update()
                    self._sync_workflow_panel()
                    return
            except Exception:
                pass
            try:
                self.viewport.fit_view_to_selected_object()
            except Exception:
                pass
            self._sync_workflow_panel()
        elif self.current_mesh is not None:
            try:
                b = np.asarray(self.current_mesh.bounds, dtype=np.float64)
                if b.shape == (2, 3) and np.isfinite(b).all():
                    self.viewport.camera.fit_to_bounds(b)
                    self.viewport.camera.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                    self.viewport.update()
            except Exception:
                pass
            self._sync_workflow_panel()

    def _set_canonical_view(self, key: str) -> None:
        preset = CANONICAL_VIEW_PRESETS.get(str(key).strip().lower())
        if preset is None:
            return
        self.set_view(float(preset[0]), float(preset[1]))

    def set_view(self, azimuth: float, elevation: float):
        try:
            az = float(azimuth)
            el = float(elevation)
        except Exception:
            return

        az = ((az + 180.0) % 360.0) - 180.0
        for tgt in VIEW_CANONICAL_AZIMUTHS:
            if abs(az - tgt) <= VIEW_ANGLE_EPS:
                az = tgt
                break
        if abs(el) <= VIEW_ANGLE_EPS:
            el = 0.0
        if abs(el - 90.0) <= VIEW_ANGLE_EPS:
            el = 90.0
        elif abs(el + 90.0) <= VIEW_ANGLE_EPS:
            el = -90.0

        cam = self.viewport.camera
        cam.azimuth = az
        cam.elevation = max(-90.0, min(90.0, el))
        view_key = _canonical_view_key_from_angles(cam.azimuth, cam.elevation)
        view_axes = CANONICAL_VIEW_AXES.get(view_key) if view_key is not None else None

        # Keep 6-face views framed using absolute-axis-stable sizing
        # (independent from mesh rotation/orientation).
        try:
            center = None
            max_dim = None

            def _span_from_bounds(bounds_min: np.ndarray, bounds_max: np.ndarray) -> float:
                span = np.abs(np.asarray(bounds_max, dtype=np.float64) - np.asarray(bounds_min, dtype=np.float64))
                if span.shape != (3,):
                    return float(np.max(np.abs(span)))
                if view_axes is not None:
                    a0 = int(view_axes[0])
                    a1 = int(view_axes[1])
                    return float(max(float(span[a0]), float(span[a1])))
                return float(np.max(span))

            def _stable_center_dim(o):
                world_center = None
                try:
                    wb = np.asarray(o.get_world_bounds(), dtype=np.float64)
                    if wb.shape == (2, 3) and np.isfinite(wb).all():
                        world_center = (wb[0] + wb[1]) * 0.5
                except Exception:
                    world_center = None

                try:
                    mesh = getattr(o, "mesh", None)
                    if mesh is not None and hasattr(mesh, "bounds"):
                        lb = np.asarray(mesh.bounds, dtype=np.float64)
                        if lb.shape == (2, 3) and np.isfinite(lb).all():
                            sc = float(getattr(o, "scale", 1.0) or 1.0)
                            if abs(sc) < 1e-12:
                                sc = 1.0
                            d = float(_span_from_bounds(lb[0], lb[1]) * abs(sc))
                            if (
                                world_center is not None
                                and np.isfinite(world_center).all()
                                and np.isfinite(d)
                                and d > 1e-9
                            ):
                                return np.asarray(world_center, dtype=np.float64), float(d)
                except Exception:
                    pass

                try:
                    b = np.asarray(o.get_world_bounds(), dtype=np.float64)
                    if b.shape == (2, 3) and np.isfinite(b).all():
                        c = (b[0] + b[1]) * 0.5
                        d = float(_span_from_bounds(b[0], b[1]))
                        if np.isfinite(c).all() and np.isfinite(d) and d > 1e-9:
                            return np.asarray(c, dtype=np.float64), float(d)
                except Exception:
                    pass
                return None

            obj = self.viewport.selected_obj
            if obj is not None and bool(getattr(obj, "visible", True)):
                stable = _stable_center_dim(obj)
                if stable is not None:
                    center, max_dim = stable
            else:
                try:
                    bmin, bmax = self.viewport._collect_projection_bounds()
                    bmin = np.asarray(bmin, dtype=np.float64).reshape(3)
                    bmax = np.asarray(bmax, dtype=np.float64).reshape(3)
                    if np.isfinite(bmin).all() and np.isfinite(bmax).all():
                        center = (bmin + bmax) * 0.5
                        max_dim = float(_span_from_bounds(bmin, bmax))
                except Exception:
                    center = None
                    max_dim = None

            # Fallback: selected object bounds even when hidden, then current mesh bounds.
            if center is None or max_dim is None:
                try:
                    obj_any = self.viewport.selected_obj
                    if obj_any is not None:
                        b = np.asarray(obj_any.get_world_bounds(), dtype=np.float64)
                        if b.shape == (2, 3) and np.isfinite(b).all():
                            center = (b[0] + b[1]) * 0.5
                            max_dim = float(_span_from_bounds(b[0], b[1]))
                except Exception:
                    pass
            if center is None or max_dim is None:
                try:
                    mesh_current = getattr(self, "current_mesh", None)
                    if mesh_current is not None:
                        cm = np.asarray(mesh_current.bounds, dtype=np.float64)
                        if cm.shape == (2, 3) and np.isfinite(cm).all():
                            center = (cm[0] + cm[1]) * 0.5
                            max_dim = float(_span_from_bounds(cm[0], cm[1]))
                except Exception:
                    pass

            if center is not None and max_dim is not None:
                if not np.isfinite(max_dim) or max_dim <= 1e-6:
                    max_dim = VIEW_MIN_DIM
                cam.center = np.asarray(center, dtype=np.float64)
                cam.distance = float(
                    max(cam.min_distance, min(cam.max_distance, max_dim * VIEW_DISTANCE_SCALE))
                )
        except Exception:
            pass

        try:
            cam.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        except Exception:
            pass
        # 6-face view should stay orthographic and axis-aligned.
        enable_ortho_lock = True
        try:
            is_top_bottom = abs(abs(float(cam.elevation)) - 90.0) <= VIEW_ANGLE_EPS
            az_norm = ((float(cam.azimuth) + 180.0) % 360.0) - 180.0
            is_side = abs(float(cam.elevation)) <= VIEW_ANGLE_EPS and any(
                abs(az_norm - tgt) <= VIEW_ANGLE_EPS for tgt in VIEW_CANONICAL_AZIMUTHS
            )
            self.viewport._ortho_view_scale = (
                VIEW_ORTHO_SCALE_TOP_BOTTOM if is_top_bottom else VIEW_ORTHO_SCALE_SIDE
            )
            self.viewport._ortho_frame_override = None
            enable_ortho_lock = bool(is_top_bottom or is_side)
        except Exception:
            pass
        self.viewport._front_back_ortho_enabled = enable_ortho_lock
        self.viewport._canonical_view_key = view_key if (enable_ortho_lock and view_key is not None) else None
        self.viewport.update()
        self._sync_workflow_panel()

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

        unit = str(getattr(mesh, "unit", DEFAULT_MESH_UNIT) or DEFAULT_MESH_UNIT).strip().lower()
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

            unit_s = str(result.get("unit") or DEFAULT_MESH_UNIT).strip().lower()
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
            self.viewport.roi_caps_enabled = True
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

        if enabled:
            # Start cutline mode as a fresh session so stale profiles do not appear.
            try:
                self.viewport.clear_cut_lines()
            except Exception:
                pass
            try:
                self.viewport.cut_line_active = 0
                self.viewport.cutLineActiveChanged.emit(0)
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
            added = int(
                self.viewport.save_current_sections_to_layers(
                    include_cut_lines=False,
                    include_cut_profiles=True,
                    include_roi_profiles=False,
                    include_slices=False,
                    separate_section_profiles=True,
                )
            )
        except Exception:
            added = 0

        if added <= 0:
            self.status_info.setText("No section layer to save.")
            return

        self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
        self.status_info.setText(f"Saved {added} section layer(s). You can move each layer in Scene panel.")

    def on_roi_section_commit_requested(self):
        """ROI Enter 커밋 요청을 현재 조정 축 기준 ROI 단면 레이어 저장으로 처리."""
        # Capture cut location hint before save_roi_sections_to_layers() clears commit markers.
        try:
            plane_hint = str(getattr(self.viewport, "_roi_commit_plane_hint", "") or "").strip().lower()
        except Exception:
            plane_hint = ""
        if plane_hint not in ("x1", "x2", "y1", "y2"):
            try:
                plane_hint = str(getattr(self.viewport, "_roi_last_adjust_plane", "") or "").strip().lower()
            except Exception:
                plane_hint = ""
        try:
            roi_bounds = [float(v) for v in (getattr(self.viewport, "roi_bounds", None) or [])][:4]
        except Exception:
            roi_bounds = []

        try:
            added = int(self.viewport.save_roi_sections_to_layers())
        except Exception:
            added = 0

        if added <= 0:
            self.status_info.setText("No ROI section layer to save.")
            return

        self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
        loc_text = ""
        try:
            if len(roi_bounds) >= 4:
                x1, x2, y1, y2 = float(roi_bounds[0]), float(roi_bounds[1]), float(roi_bounds[2]), float(roi_bounds[3])
                if plane_hint == "x1":
                    loc_text = f" x1={x1:.2f}"
                elif plane_hint == "x2":
                    loc_text = f" x2={x2:.2f}"
                elif plane_hint == "y1":
                    loc_text = f" y1={y1:.2f}"
                elif plane_hint == "y2":
                    loc_text = f" y2={y2:.2f}"
                else:
                    loc_text = f" x[{x1:.2f},{x2:.2f}] y[{y1:.2f},{y2:.2f}]"
        except Exception:
            loc_text = ""
        self.status_info.setText(
            f"Saved ROI section layer(s): {added}.{loc_text}  Move/offset in Scene panel."
        )

    def _on_cut_lines_auto_ended(self):
        self._sync_cutline_button_state(False)
        try:
            added = int(
                self.viewport.save_current_sections_to_layers(
                    include_cut_lines=False,
                    include_cut_profiles=True,
                    include_roi_profiles=False,
                    include_slices=False,
                    separate_section_profiles=True,
                )
            )
        except Exception:
            added = 0
        if added > 0:
            try:
                self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
            except Exception:
                pass
            self.status_info.setText(
                f"Cut section committed: {added} layer(s). Move them in Scene panel."
            )

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

        # Optional: open project/mesh passed via CLI.
        try:
            if "--open-project" in sys.argv:
                i = sys.argv.index("--open-project")
                if i + 1 < len(sys.argv):
                    p = str(sys.argv[i + 1])
                    if p:
                        window.open_project_path(p)
            elif "--open-mesh" in sys.argv:
                i = sys.argv.index("--open-mesh")
                if i + 1 < len(sys.argv):
                    p = str(sys.argv[i + 1])
                    if p:
                        window.open_file_path(p, prompt_unit=True)
        except Exception:
            _LOGGER.exception("Failed to auto-open file from CLI args")
        
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
