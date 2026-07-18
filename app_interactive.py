"""
ArchMeshRubbing v0.1.0 - Complete Interactive Application
Copyright (C) 2026 balguljang2 (lzpxilfe)
Licensed under the GNU General Public License v2.0 (GPL2)
"""

import sys
import copy
import logging
import subprocess
import json
import time
import hashlib
import os
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping
import uuid

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QDockWidget, QTreeWidget,
    QTreeWidgetItem, QGroupBox, QDoubleSpinBox, QFormLayout,
    QSlider, QSpinBox, QStatusBar, QToolBar, QFrame,
    QMessageBox, QTextEdit, QProgressBar, QComboBox,
    QCheckBox, QScrollArea, QSizePolicy, QButtonGroup, QDialog, QLineEdit,
    QGridLayout, QProgressDialog, QMenu, QInputDialog
)
from PyQt6.QtCore import (
    Qt,
    QTimer,
    QSize,
    pyqtSignal,
    QThread,
    QBuffer,
    QByteArray,
    QIODevice,
    QEvent,
    QObject,
)
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QPixmap, QShortcut
import numpy as np
from PIL import Image, ImageDraw
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
TASK_SHUTDOWN_WAIT_MS = 30_000
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
_METHOD_NAME_ARAP = "저왜곡 펼침"
_METHOD_NAME_LSCM = "각도 보존 펼침"
_METHOD_NAME_AREA = "기록면 기반 펼침"
_METHOD_NAME_CYLINDER = "곡면 추적 펼침"
_METHOD_NAME_SECTION = "기와 추천 펼침"
_SECTION_RECOMMEND_TAG = "기와 추천"


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

from src.gui.opengl_context import (  # noqa: E402
    install_windows_software_pyopengl_bridge,
)

install_windows_software_pyopengl_bridge()

from src.gui.viewport_3d import SurfaceAnchorObservation, Viewport3D  # noqa: E402
from src.core.mesh_loader import MeshData, MeshLoader  # noqa: E402
from src.core.profile_exporter import ProfileExporter  # noqa: E402
from src.core.project_file import (  # noqa: E402
    EmbeddedSourceRequiredError,
    MIGRATION_MARKER_NAME,
    ProjectFormatError,
    ProjectSaveError,
    ProjectSerializationError,
    UnsupportedPayloadError,
    load_artifact_project_package as load_amr_artifact_project_package,
    load_artifact_session_project as load_amr_artifact_session_project,
    load_project as load_amr_project,
    save_artifact_session_project as save_amr_artifact_session_project,
    save_project as save_amr_project,
)
from src.core.project_recovery import (  # noqa: E402
    InterruptedProjectSave,
    ProjectRecoveryError,
    ProjectRecoveryResult,
    discover_interrupted_project_saves as discover_amr_interrupted_saves,
    recover_interrupted_project_save as recover_amr_interrupted_save,
)
from src.core.artifact_document import ArtifactDocument  # noqa: E402
from src.core.artifact_geometry_metrics import (  # noqa: E402
    ArtifactGeometryMetricsComputation,
)
from src.core.artifact_surface_measurement import (  # noqa: E402
    BARYCENTRIC_DENOMINATOR,
    ArtifactSurfaceMeasurementComputation,
    resolve_surface_anchor_from_ray,
)
from src.core.artifact_scene_adapter import (  # noqa: E402
    ArtifactProjectionSnapshot,
)
from src.core.artifact_session import (  # noqa: E402
    ArtifactSession,
    ArtifactSessionError,
)
from src.application.artifact_workbench import (  # noqa: E402
    ArtifactLoadTicket,
    ArtifactWorkbench,
    ArtifactWorkbenchError,
    ProjectionActivation,
    ProjectionTransition,
    RecordBindingTransition,
    StaleWorkflowOperationError,
    WorkflowBusyError,
    WorkflowSaveStatus,
    WorkflowSnapshot,
    WorkflowStage,
    WorkflowTransitionKind,
)
from src.application.artifact_measurements import (  # noqa: E402
    ArtifactMeasurementController,
    ArtifactMeasurementResult,
    ArtifactMeasurementWorkItem,
    DEFAULT_RUBBING_MEMORY_BUDGET_BYTES,
    MeasurementCancelledError,
    MeasurementOperationKind,
    MeasurementOperationState,
)
from src.application.artifact_workflow_progress import (  # noqa: E402
    ArtifactWorkflowProgress,
    derive_artifact_workflow_progress,
)
from src.application.artifact_exports import (  # noqa: E402
    ArtifactExportController,
    ArtifactExportError,
    ArtifactExportPublication,
    ArtifactExportResult,
    ArtifactExportState,
    ArtifactExportWorkItem,
)
from src.application.artifact_survey_exports import (  # noqa: E402
    ArtifactSurveyExportController,
    ArtifactSurveyExportPublication,
    ArtifactSurveyExportResult,
    ArtifactSurveyExportWorkItem,
)
from src.core.artifact_vector_extractor import (  # noqa: E402
    ArtifactVectorComputation,
    ArtifactVectorExtractionError,
)
from src.core.artifact_outline_extractor import (  # noqa: E402
    DEFAULT_OUTLINE_PRECISION_GRID_MM,
)
from src.core.artifact_rubbing_extractor import (  # noqa: E402
    ArtifactRubbingComputation,
    ArtifactRubbingError,
    DEFAULT_RUBBING_BLACK_POINT_UM,
    DEFAULT_RUBBING_DEPTH_QUANTIZATION_UM,
    DEFAULT_RUBBING_INK_STRENGTH_PERCENT,
    DEFAULT_RUBBING_MARGIN_UM,
    DEFAULT_RUBBING_PIXELS_PER_MM,
    DEFAULT_RUBBING_POLARITY,
    DEFAULT_RUBBING_REFERENCE_RADIUS_UM,
    DigitalRubbingRaster,
    compute_artifact_rubbing_from_recipe,
    estimate_digital_rubbing_resources,
    require_current_rubbing_computation,
)
from src.core.artifact_rubbing_record import (  # noqa: E402
    RUBBING_RECORD_TYPE,
    rubbing_receipt_from_record,
)
from src.core.artifact_rubbing_export import (  # noqa: E402
    ArtifactRubbingExportError,
    RUBBING_EXPORT_DIRECTORY_SUFFIX,
)
from src.core.artifact_survey_export import (  # noqa: E402
    SURVEY_EXPORT_DIRECTORY_SUFFIX,
)
from src.core.artifact_tile_unwrap_export import (  # noqa: E402
    ArtifactTileUnwrapExportError,
    TILE_UNWRAP_EXPORT_DIRECTORY_SUFFIX,
)
from src.core.artifact_tile_unwrap_extractor import (  # noqa: E402
    ArtifactTileUnwrapComputation,
    ArtifactTileUnwrapError,
    TileUnwrapMesh,
    compute_artifact_tile_unwrap_from_recipe,
    require_current_tile_unwrap_computation,
    selection_face_indices,
)
from src.core.artifact_tile_unwrap_record import (  # noqa: E402
    TILE_UNWRAP_RECORD_TYPE,
    tile_unwrap_receipt_from_record,
)
from src.core.artifact_vector_export import (  # noqa: E402
    ArtifactVectorExportError,
    VECTOR_EXPORT_DIRECTORY_SUFFIX,
)
from src.core.artifact_vector_record import (  # noqa: E402
    PlanarFrame,
    VectorRecordKind,
    vector_payload_from_record,
)


_NATIVE_CUTLINE_AXIS_INDEX = {"top": 2, "front": 1, "right": 0}
_NativeExportController = ArtifactExportController | ArtifactSurveyExportController
_NativeExportWorkItem = ArtifactExportWorkItem | ArtifactSurveyExportWorkItem
_NativeExportPublication = ArtifactExportPublication | ArtifactSurveyExportPublication


def _native_cutline_frame(view: str, offset_mm: float) -> PlanarFrame:
    """Return the fixed right-handed survey plane for Top/Front/Right cuts."""

    key = str(view or "").strip().lower()
    offset = float(offset_mm)
    if not np.isfinite(offset):
        raise ArtifactVectorExtractionError("cutline offset must be finite millimetres")
    if key == "top":
        return PlanarFrame(
            origin_world_mm=(0.0, 0.0, offset),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 1.0, 0.0),
            normal_world=(0.0, 0.0, 1.0),
        )
    if key == "front":
        return PlanarFrame(
            origin_world_mm=(0.0, offset, 0.0),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 0.0, 1.0),
            normal_world=(0.0, -1.0, 0.0),
        )
    if key == "right":
        return PlanarFrame(
            origin_world_mm=(offset, 0.0, 0.0),
            u_axis_world=(0.0, 1.0, 0.0),
            v_axis_world=(0.0, 0.0, 1.0),
            normal_world=(1.0, 0.0, 0.0),
        )
    raise ArtifactVectorExtractionError(f"unsupported native Cutline view: {view!r}")
from src.core.source_identity import (  # noqa: E402
    SourceFingerprint,
    SourceVerification,
    SourceVerificationStatus,
    compare_fingerprints,
    legacy_unverified_source,
    missing_source,
)
from src.core.runtime_defaults import DEFAULTS as RUNTIME_DEFAULTS  # noqa: E402
from src.gui.profile_graph_widget import ProfileGraphWidget  # noqa: E402
from src.gui.pixel_icons import pixel_icon, set_pixel_icon  # noqa: E402
from src.core.alignment_utils import (  # noqa: E402
    compute_minimax_center_shift,
    compute_nonpenetration_lift,
    fit_plane_normal,
    orient_plane_normal_toward,
    rotation_matrix_align_vectors,
    scene_rotation_matrix,
    scene_trs_matrix,
    transform_plane_world_to_local,
    transform_points,
)
from src.core.unit_utils import (  # noqa: E402
    DEFAULT_MESH_UNIT,
    mesh_units_to_mm,
    mm_to_mesh_units,
)
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
from src.core.flatten_policy import recommend_flatten_mode  # noqa: E402

DEFAULT_EXPORT_DPI = RUNTIME_DEFAULTS.export_dpi

_ARTIFACT_AUTHORITY_REOPEN_STATUS = (
    "문서 권위 복원 실패 | 저장·실측·내보내기 차단, "
    "검증된 원본을 다시 여세요"
)

_SOURCE_BINDING_CAPTURED = "captured_at_import"
_SOURCE_BINDING_LEGACY = "legacy_unverified"
_SOURCE_BINDING_GENERATED = "generated_ephemeral"
_SOURCE_BINDING_VALUES = {
    _SOURCE_BINDING_CAPTURED,
    _SOURCE_BINDING_LEGACY,
    _SOURCE_BINDING_GENERATED,
}
_ALIGNMENT_STATUS_MUTABLE_TRS = "legacy_mutable_trs"
_ALIGNMENT_STATUS_UNVERIFIABLE = "legacy_unverifiable"
_ALIGNMENT_STATUS_BAKED_UNVERIFIABLE = "legacy_baked_unverifiable"
_VIEWPORT_PROJECT_SWAP_FIELDS = (
    "_mesh_center",
    "_amr_scene_render_origin_world_mm",
    "picking_mode",
    "curvature_pick_mode",
    "picked_points",
    "fitted_arc",
    "measure_picked_points",
    "measure_picked_anchors",
    "slice_enabled",
    "slice_z",
    "slice_contours",
    "crosshair_enabled",
    "crosshair_pos",
    "x_profile",
    "y_profile",
    "_world_x_profile",
    "_world_y_profile",
    "roi_enabled",
    "active_roi_edge",
    "roi_rect_dragging",
    "roi_rect_start",
    "_roi_bounds_changed",
    "_roi_move_dragging",
    "_roi_move_last_xy",
    "_roi_commit_axis_hint",
    "_roi_last_adjust_axis",
    "_roi_commit_plane_hint",
    "_roi_last_adjust_plane",
    "roi_bounds",
    "roi_cut_edges",
    "roi_cap_verts",
    "roi_section_world",
    "roi_caps_enabled",
    "cut_lines_enabled",
    "cut_lines",
    "cut_line_axis_lock",
    "cut_line_active",
    "cut_line_drawing",
    "cut_line_preview",
    "_cut_line_final",
    "cut_section_profiles",
    "cut_section_world",
    "cut_section_contours_world",
    "cut_section_contours_local",
    "_active_polyline_layer_obj_index",
    "_active_polyline_layer_index",
    "_polyline_layer_dragging",
    "_polyline_layer_drag_world_start",
    "_polyline_layer_drag_offset_start",
    "_polyline_layer_drag_axis_lock",
    "_polyline_layer_drag_moved",
    "_cut_section_pending_indices",
    "line_section_enabled",
    "line_section_dragging",
    "line_section_start",
    "line_section_end",
    "line_profile",
    "line_section_contours",
    "floor_picks",
    "_floor_pick_ready",
    "_floor_pick_ready_at",
    "surface_paint_points",
    "surface_lasso_points",
    "surface_lasso_face_indices",
    "surface_lasso_preview",
    "grid_spacing",
    "grid_size",
    "undo_stack",
    "_front_back_ortho_enabled",
    "_canonical_view_key",
    "_ortho_frame_override",
)


def _normalized_path_hint(path: str) -> str:
    try:
        value = os.path.normcase(str(Path(path).expanduser().resolve(strict=False)))
    except Exception:
        value = os.path.normcase(str(path))
    return unicodedata.normalize("NFC", value)


def _same_filesystem_target(first: str, second: str) -> bool:
    try:
        first_path = Path(first).expanduser()
        second_path = Path(second).expanduser()
        if first_path.exists() and second_path.exists():
            return os.path.samefile(first_path, second_path)
    except (OSError, ValueError):
        pass
    return _normalized_path_hint(first) == _normalized_path_hint(second)


def _mesh_source_payload(mesh: Any, mesh_path: str | None) -> dict[str, Any]:
    """Serialize the identity captured when this in-memory geometry was imported.

    Deliberately never re-hash ``mesh_path`` here: the file may have changed
    after import, and attaching its new hash to old in-memory geometry would
    create false provenance.
    """
    identity = getattr(mesh, "source_identity", None) if mesh is not None else None
    binding_status = str(
        getattr(mesh, "_amr_source_binding_status", "") or ""
    ).strip()

    if mesh_path:
        if not isinstance(identity, SourceFingerprint):
            raise ProjectSerializationError(
                f"External mesh {mesh_path!r} has no immutable source identity; "
                "reload the source before saving"
            )
        if not binding_status:
            binding_status = _SOURCE_BINDING_CAPTURED
    else:
        if identity is not None and not isinstance(identity, SourceFingerprint):
            raise ProjectSerializationError("Mesh source identity has an invalid runtime type")
        if not binding_status:
            binding_status = _SOURCE_BINDING_GENERATED

    if binding_status not in _SOURCE_BINDING_VALUES:
        raise ProjectSerializationError(
            f"Unsupported mesh source binding status: {binding_status!r}"
        )
    parse_format = str(
        getattr(mesh, "source_format", "")
        or (identity.format if identity is not None else "")
    ).strip().lower().removeprefix(".")
    if identity is not None and f".{parse_format}" not in MeshLoader.SUPPORTED_FORMATS:
        raise ProjectSerializationError(
            f"Unsupported mesh source parser format: {parse_format!r}"
        )
    return {
        "identity": identity.to_dict() if identity is not None else None,
        "binding_status": binding_status,
        "parse_format": parse_format or None,
    }


def _validate_project_source_declarations(
    state: dict[str, Any],
    *,
    migrated_from_v1: bool,
) -> None:
    """Reject invalid v2 source declarations before the live scene is cleared."""
    objects = state.get("objects", [])
    if not isinstance(objects, list):
        raise ProjectFormatError("Invalid project state: 'objects' must be a list")

    for index, obj_state in enumerate(objects):
        if not isinstance(obj_state, dict):
            raise ProjectFormatError(f"Invalid project object at index {index}")
        mesh_info = obj_state.get("mesh", {})
        if not isinstance(mesh_info, dict):
            raise ProjectFormatError(f"Invalid mesh declaration at object index {index}")
        path_hint = str(mesh_info.get("path", "") or "").strip()
        if not path_hint:
            continue

        source = mesh_info.get("source")
        if not isinstance(source, dict):
            if migrated_from_v1:
                continue
            raise ProjectFormatError(
                f"Object {index} has an external mesh path but no source declaration"
            )
        binding_status = str(source.get("binding_status", "") or "").strip()
        if binding_status not in _SOURCE_BINDING_VALUES:
            raise ProjectFormatError(
                f"Object {index} has unsupported source binding status: {binding_status!r}"
            )
        raw_identity = source.get("identity")
        if raw_identity is None:
            if binding_status == _SOURCE_BINDING_LEGACY:
                continue
            raise ProjectFormatError(
                f"Object {index} has no source identity for external mesh {path_hint!r}"
            )
        if not isinstance(raw_identity, dict):
            raise ProjectFormatError(f"Object {index} source identity must be an object")
        try:
            identity = SourceFingerprint.from_dict(raw_identity)
        except ValueError as exc:
            raise ProjectFormatError(
                f"Object {index} has invalid source identity: {exc}"
            ) from exc
        parse_format = str(source.get("parse_format", identity.format) or "").strip()
        parse_ext = f".{parse_format.lower().removeprefix('.')}"
        if parse_ext not in MeshLoader.SUPPORTED_FORMATS:
            raise ProjectFormatError(
                f"Object {index} has unsupported source parser format: {parse_format!r}"
            )


def _verify_loaded_project_source(
    mesh_data: Any,
    obj_state: dict[str, Any],
    loaded_path: str,
    *,
    migrated_from_v1: bool,
) -> tuple[SourceVerification, str]:
    """Compare saved and imported identities before any scene mutation."""
    mesh_info = obj_state.get("mesh", {})
    if not isinstance(mesh_info, dict):
        return (
            SourceVerification(
                status=SourceVerificationStatus.UNREADABLE,
                checked_path=str(loaded_path),
                detail="project mesh declaration is not an object",
            ),
            _SOURCE_BINDING_LEGACY if migrated_from_v1 else _SOURCE_BINDING_CAPTURED,
        )

    source = mesh_info.get("source")
    if not isinstance(source, dict):
        source = {}
    binding_status = str(source.get("binding_status", "") or "").strip()
    if migrated_from_v1:
        return legacy_unverified_source(str(loaded_path)), _SOURCE_BINDING_LEGACY

    raw_identity = source.get("identity")
    if raw_identity is None and binding_status == _SOURCE_BINDING_LEGACY:
        return legacy_unverified_source(str(loaded_path)), _SOURCE_BINDING_LEGACY
    if not isinstance(raw_identity, dict):
        return (
            SourceVerification(
                status=SourceVerificationStatus.UNREADABLE,
                checked_path=str(loaded_path),
                detail="project source identity is missing or invalid",
            ),
            binding_status or _SOURCE_BINDING_CAPTURED,
        )

    try:
        expected = SourceFingerprint.from_dict(raw_identity)
    except ValueError as exc:
        return (
            SourceVerification(
                status=SourceVerificationStatus.UNREADABLE,
                checked_path=str(loaded_path),
                detail=f"invalid expected source identity: {exc}",
            ),
            binding_status or _SOURCE_BINDING_CAPTURED,
        )

    actual = getattr(mesh_data, "source_identity", None)
    if not isinstance(actual, SourceFingerprint):
        return (
            SourceVerification(
                status=SourceVerificationStatus.UNREADABLE,
                checked_path=str(loaded_path),
                expected=expected,
                detail="mesh loader did not provide a source identity",
            ),
            binding_status or _SOURCE_BINDING_CAPTURED,
        )

    stored_path = str(mesh_info.get("path", "") or "").strip()
    relocated = bool(stored_path) and _normalized_path_hint(stored_path) != _normalized_path_hint(
        str(loaded_path)
    )
    verification = compare_fingerprints(
        expected,
        actual,
        checked_path=str(loaded_path),
        relocated=relocated,
    )
    return verification, binding_status or _SOURCE_BINDING_CAPTURED


class MeshLoadThread(QThread):
    loaded = pyqtSignal(object, str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        filepath: str,
        scale_factor: float,
        default_unit: str,
        source_format: str | None = None,
        import_recipe: Mapping[str, object] | None = None,
        capture_dependencies: bool = False,
    ):
        super().__init__()
        self._filepath = str(filepath)
        self._scale_factor = float(scale_factor)
        self._default_unit = str(default_unit)
        self._source_format = str(source_format or "").strip().lower() or None
        self._import_recipe = (
            dict(import_recipe) if isinstance(import_recipe, Mapping) else None
        )
        self._capture_dependencies = bool(capture_dependencies)

    def run(self):
        try:
            loader = MeshLoader(default_unit=self._default_unit)
            mesh_data = loader.load(
                self._filepath,
                source_format=self._source_format,
                import_recipe=self._import_recipe,
                capture_dependencies=self._capture_dependencies,
            )

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
                        # CPU picking authority stays absolute float64.  Casting
                        # these centroids at survey offsets would collapse
                        # millimetre-adjacent faces before the renderer sees them.
                        centroids = np.empty((int(faces.shape[0]), 3), dtype=np.float64)
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
                            centroids[start:end, :] = (v0 + v1 + v2) / 3.0

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

            slicer = MeshSlicer(self._mesh_data.to_trimesh())

            local_to_world = scene_trs_matrix(
                self._translation,
                self._rotation,
                self._scale,
            )

            world_origin = np.array([0.0, 0.0, self._z], dtype=np.float64)
            world_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            local_origin, local_normal = transform_plane_world_to_local(
                world_origin,
                world_normal,
                local_to_world,
            )

            contours_local = slicer.slice_with_plane(local_origin, local_normal)

            world_contours = []
            for cnt in contours_local:
                world_contours.append(transform_points(cnt, local_to_world))

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
        except MeasurementCancelledError as e:
            _LOGGER.info("Task cancelled: %s", self._task_name)
            self.failed.emit(f"{type(e).__name__}: {e}")
        except Exception as e:
            _LOGGER.exception("Task failed: %s", self._task_name)
            self.failed.emit(f"{type(e).__name__}: {e}")


def _load_project_open_candidate(filepath: str) -> dict[str, Any]:
    """Validate and materialize one Project Open candidate off the GUI thread.

    Native packages are attempted first so an invalid embedded bundle fails
    closed instead of being reinterpreted as a legacy project.  A typed
    manifest-only result is the sole native path that may later ask the user
    to resolve an external source.
    """

    try:
        package = load_amr_artifact_project_package(filepath)
    except UnsupportedPayloadError as exc:
        if exc.payload_type != "legacy_ui_state":
            raise
        return {
            "kind": "legacy",
            "document": load_amr_project(filepath),
        }

    try:
        session = load_amr_artifact_session_project(filepath)
    except EmbeddedSourceRequiredError:
        if package.source_bundle is not None:
            raise ProjectFormatError(
                "Embedded source bundle was validated but could not be materialized"
            )
        return {
            "kind": "artifact_manifest_only",
            "document": package.document,
        }

    if session.document != package.document:
        raise ProjectFormatError(
            "Embedded session materialization changed the ArtifactDocument"
        )
    return {
        "kind": "artifact_embedded",
        "document": package.document,
        "session": session,
    }


class _TaskDialogCloseGuard(QObject):
    """Keep a cancelled task's wait dialog visible until its worker exits."""

    def __init__(self) -> None:
        super().__init__()
        self.waiting_for_worker = False
        self.close_allowed = False

    def eventFilter(self, watched, event):
        if self.waiting_for_worker and not self.close_allowed:
            if event.type() == QEvent.Type.Close:
                event.ignore()
                return True
            if (
                event.type() == QEvent.Type.KeyPress
                and event.key() == Qt.Key.Key_Escape
            ):
                event.accept()
                return True
        return super().eventFilter(watched, event)


@dataclass(frozen=True, slots=True)
class _NativeProjectSaveResult:
    destination: str
    durability_warning: str | None = None


@dataclass(frozen=True, slots=True)
class _NativeTransientIssue:
    """One GUI-only value that has no ArtifactDocument serialization contract."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class _NativeTransientWorkState:
    """Cheap, fail-closed summary of native work that Save cannot persist."""

    issues: tuple[_NativeTransientIssue, ...] = ()

    @property
    def has_unpersisted_work(self) -> bool:
        return bool(self.issues)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(issue.message for issue in self.issues)

    def detail(self, *, limit: int = 8) -> str:
        shown = self.reasons[: max(1, int(limit))]
        lines = [f"- {reason}" for reason in shown]
        hidden = len(self.issues) - len(shown)
        if hidden > 0:
            lines.append(f"- 그 밖의 미확정 작업 {hidden}개")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _NativeAlignSceneCapture:
    scene_object: Any
    session: ArtifactSession
    binding: ArtifactProjectionSnapshot
    mesh: MeshData
    translation_mm: tuple[float, float, float]
    rotation_deg: tuple[float, float, float]
    scale: float
    pivot_mm: tuple[float, float, float]
    project_path: str | None
    project_requires_save_as: bool
    legacy_project_path: str | None
    state_version: int
    authority_epoch: int


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
            <h3 style="margin:0; color:#2c5282;">기본 흐름</h3>
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
            <h3 style="margin:0; color:#2c5282;">정위치 (Positioning)</h3>
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
            <h3 style="margin:0; color:#2c5282;">기록면 전개 설정</h3>
            <p style="font-size:11px;">
            이 단계는 삼각형 면을 따로 터뜨리는 것이 아니라, 기록할 표면을 연속된 좌표계로 전개하기 위한 설정입니다.<br>
            <b>기록면 미리보기:</b> 전개 결과를 연속 탁본 이미지로 바로 확인<br><br>
            기본 출력은 <b>탁본 이미지 + 외곽선</b> 중심이며, 와이어프레임은 기본적으로 사용하지 않습니다.<br>
            곡률 측정, 고급 옵션, 표면 라벨링은 <b>보정/실험 도구</b>로 숨겨져 있습니다.
            </p>
        """)
    
    def set_scene_help(self):
        self.setHtml("""
            <h3 style="margin:0; color:#2c5282;">레이어 트리 (Layer)</h3>
            <p style="font-size:11px;">
            현재 작업 중인 객체 목록입니다.<br>
            <b>클릭:</b> 객체 선택 및 기즈모 활성화<br>
            <b>눈 아이콘:</b> 가시성 토글<br>
            <b>더블클릭:</b> 객체 이름 변경
            </p>
        """)
    
    def set_selection_help(self):
        self.setHtml("""
            <h3 style="margin:0; color:#2c5282;">기록할 표면 선택</h3>
            <p style="font-size:11px;">
            먼저 기록할 표면 패치를 고르는 도구입니다.<br>
            권장 흐름은 <b>표준 시점 버튼 → 가시면 선택 → 현재 선택으로 전개/탁본 저장</b> 입니다.<br><br>

            <b>가시면 선택</b><br>
            - <b>현재 시점 가시면</b>: 지금 카메라에서 실제로 보이는 면만 선택<br>
            - <b>상면/하면/정면/후면/좌측/우측</b>: 표준 시점으로 맞춘 뒤 그 시점의 가시면 선택<br><br>

            외면/내면/미구 라벨링은 기본 흐름이 아니라 <b>연구용 표면 라벨링</b>입니다. 필요한 경우에만 별도로 펼쳐 사용하세요.<br><br>

            <b>경계(면적+자석)</b><br>
            - <b>좌클릭:</b> 점 추가(자석 스냅) / <b>드래그:</b> 카메라 회전<br>
            - <b>첫 점 근처 클릭</b> 또는 <b>우클릭/Enter</b>: 확정<br>
            - <b>Backspace</b>: 되돌리기 / <b>Alt</b>: 제거 모드<br>
            - <b>Shift/Ctrl</b>: 완드 정제 / <b>[ / ]</b>: 자석 반경 / <b>ESC</b>: 종료<br>
            </p>
        """)

    def set_tile_help(self):
        self.setHtml("""
            <h3 style="margin:0; color:#2c5282;">실측용 도면 / 기와 제작 추정</h3>
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
            <h3 style="margin:0; color:#2c5282;">작업 흐름</h3>
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
    """Confirm source unit and signed-axis mapping before scientific use."""
    last_index = 0  # 클래스 변수로 마지막 선택 기억
    last_axes = {"source_x": "+X", "source_y": "+Y", "source_z": "+Z"}
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("원본 단위·좌표축 확인")
        self.setFixedWidth(420)
        
        layout = QVBoxLayout(self)
        label = QLabel(
            "파일에 기록된 원본 단위와 좌표축을 확인하세요.\n"
            "이 선택은 원본을 변경하지 않고 metadata revision으로 저장됩니다."
        )
        label.setStyleSheet("color: #4a5568; font-size: 11px;")
        label.setWordWrap(True)
        layout.addWidget(label)
        
        self.combo = QComboBox()
        self.combo.addItems(["Millimeters (mm)", "Centimeters (cm)", "Meters (m)"])
        self.combo.setCurrentIndex(UnitSelectionDialog.last_index) 
        layout.addWidget(self.combo)

        axis_group = QGroupBox("원본 축 → canonical 축")
        axis_layout = QFormLayout(axis_group)
        self.axis_combos: dict[str, QComboBox] = {}
        axis_values = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
        for key, label_text in (
            ("source_x", "원본 X"),
            ("source_y", "원본 Y"),
            ("source_z", "원본 Z"),
        ):
            axis_combo = QComboBox()
            axis_combo.addItems(axis_values)
            saved = UnitSelectionDialog.last_axes.get(key, f"+{key[-1].upper()}")
            axis_combo.setCurrentText(saved)
            axis_combo.currentIndexChanged.connect(self._update_accept_enabled)
            self.axis_combos[key] = axis_combo
            axis_layout.addRow(label_text, axis_combo)
        layout.addWidget(axis_group)

        self.confirm_metadata = QCheckBox(
            "위 단위와 축 매핑을 확인했습니다 (오른손/왼손은 매핑에서 계산)"
        )
        self.confirm_metadata.setChecked(False)
        self.confirm_metadata.toggled.connect(self._update_accept_enabled)
        layout.addWidget(self.confirm_metadata)
        
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("확인")
        self.ok_btn.setDefault(True)
        self.ok_btn.setEnabled(False)
        self.ok_btn.clicked.connect(self.accept_and_save)
        cancel_btn = QPushButton("취소")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)

    def _axes(self) -> dict[str, str]:
        return {
            key: str(combo.currentText()).strip()
            for key, combo in self.axis_combos.items()
        }

    def _axes_are_bijective(self) -> bool:
        values = list(self._axes().values())
        return len({value[-1] for value in values if value}) == 3

    def _update_accept_enabled(self, *_args) -> None:
        self.ok_btn.setEnabled(
            bool(self.confirm_metadata.isChecked()) and self._axes_are_bijective()
        )

    def accept_and_save(self):
        if not self._axes_are_bijective():
            QMessageBox.warning(
                self,
                "좌표축 확인",
                "원본 X/Y/Z는 canonical X/Y/Z에 각각 한 번씩 대응해야 합니다.",
            )
            return
        if not self.confirm_metadata.isChecked():
            return
        UnitSelectionDialog.last_index = self.combo.currentIndex()
        UnitSelectionDialog.last_axes = self._axes()
        self.accept()

    def get_source_metadata(self) -> dict[str, Any]:
        units = ("mm", "cm", "m")
        index = int(self.combo.currentIndex())
        unit = units[index] if 0 <= index < len(units) else "unknown"
        axes = self._axes()
        vectors = {
            "+X": np.array([1.0, 0.0, 0.0]),
            "-X": np.array([-1.0, 0.0, 0.0]),
            "+Y": np.array([0.0, 1.0, 0.0]),
            "-Y": np.array([0.0, -1.0, 0.0]),
            "+Z": np.array([0.0, 0.0, 1.0]),
            "-Z": np.array([0.0, 0.0, -1.0]),
        }
        axes_are_bijective = self._axes_are_bijective()
        if axes_are_bijective:
            matrix = np.column_stack(
                [vectors[axes[key]] for key in ("source_x", "source_y", "source_z")]
            )
            handedness = "right" if float(np.linalg.det(matrix)) > 0.0 else "left"
        else:
            handedness = "unknown"
        return {
            "unit": unit,
            "axes": axes,
            "handedness": handedness,
            "confirmation_status": (
                "confirmed"
                if self.confirm_metadata.isChecked() and axes_are_bijective
                else "unconfirmed"
            ),
        }

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
    _VISIBILITY_ROLE = int(Qt.ItemDataRole.UserRole) + 1
    
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
            mesh_item = QTreeWidgetItem([obj.name, "", f"{len(obj.mesh.faces):,}"])
            mesh_item.setData(0, Qt.ItemDataRole.UserRole, ("mesh", i))
            self._set_visibility_state(mesh_item, bool(obj.visible))
            self.tree.addTopLevelItem(mesh_item)
            
            # 부착된 원호들
            for j, arc in enumerate(obj.fitted_arcs):
                arc_item = QTreeWidgetItem(mesh_item)
                arc_item.setText(0, f"원호 #{j+1}")
                arc_item.setText(1, "")
                arc_item.setIcon(1, pixel_icon("measure"))
                arc_item.setToolTip(1, "곡률 원호")
                r_mm = mesh_units_to_mm(float(getattr(arc, "radius", 0.0)), getattr(obj.mesh, "unit", None))
                arc_item.setText(2, f"R={r_mm:.1f}mm")
                arc_item.setData(0, Qt.ItemDataRole.UserRole, ("arc", i, j))

            # 저장된 단면/가이드 레이어
            for k, layer in enumerate(getattr(obj, "polyline_layers", []) or []):
                layer_item = QTreeWidgetItem(mesh_item)
                name = str(layer.get("name", "")).strip() or f"레이어 #{k+1}"
                layer_item.setText(0, name)

                visible = bool(layer.get("visible", True))
                self._set_visibility_state(layer_item, visible)

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

    def _set_visibility_state(self, item: QTreeWidgetItem, visible: bool) -> None:
        item.setText(1, "")
        item.setData(1, self._VISIBILITY_ROLE, bool(visible))
        item.setIcon(1, pixel_icon("visible" if visible else "hidden"))
        item.setToolTip(1, "표시 중" if visible else "숨김")
                
    def on_item_clicked(self, item, column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        
        if data[0] == "mesh":
            index = data[1]
            if column == 1: # 가시성 토글
                visible = not bool(item.data(1, self._VISIBILITY_ROLE))
                self._set_visibility_state(item, visible)
                self.visibilityChanged.emit(index, visible)
            else:
                self.selectionChanged.emit(index)
        elif data[0] == "layer":
            obj_idx = int(data[1])
            layer_idx = int(data[2])
            if column == 1:
                visible = not bool(item.data(1, self._VISIBILITY_ROLE))
                self._set_visibility_state(item, visible)
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
            delete_action = menu.addAction(pixel_icon("delete"), "원호 삭제")
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
            delete_action = menu.addAction(pixel_icon("delete"), "레이어 삭제")
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
        self.setIconSize(QSize(16, 16))
        self.init_ui()

    def init_ui(self):
        # 이동 (cm)
        self.addWidget(QLabel("이동: "))
        self.trans_x = self._create_spin(-10000, 10000, "X", step=0.1)
        self.trans_y = self._create_spin(-10000, 10000, "Y", step=0.1)
        self.trans_z = self._create_spin(-10000, 10000, "Z", step=0.1)
        self.addWidget(self.trans_x)
        self.addWidget(self.trans_y)
        self.addWidget(self.trans_z)
        
        self.addSeparator()
        
        # 회전 (deg)
        self.addWidget(QLabel("회전: "))
        self.rot_x = self._create_spin(-360, 360, "Rx", step=1.0)
        self.rot_y = self._create_spin(-360, 360, "Ry", step=1.0)
        self.rot_z = self._create_spin(-360, 360, "Rz", step=1.0)
        self.addWidget(self.rot_x)
        self.addWidget(self.rot_y)
        self.addWidget(self.rot_z)
        
        self.addSeparator()
        
        # 배율
        self.addWidget(QLabel("배율: "))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.01, 100.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setFixedWidth(70)
        self.addWidget(self.scale_spin)
        
        self.addSeparator()
        
        # 버튼들
        self.btn_bake = QPushButton("정치 확정")
        set_pixel_icon(self.btn_bake, "align")
        self.btn_bake.setToolTip("현재 변환을 메쉬에 영구 적용하고 위치를 고정합니다")
        self.btn_bake.setStyleSheet("QPushButton { font-weight: bold; padding: 2px 10px; }")
        self.addWidget(self.btn_bake)

        self.btn_fixed = QPushButton("고정상태로")
        set_pixel_icon(self.btn_fixed, "lock")
        self.btn_fixed.setToolTip("정치 확정(Bake) 이후의 고정 상태로 되돌립니다 (실수로 이동/회전했을 때)")
        self.btn_fixed.setEnabled(False)
        self.addWidget(self.btn_fixed)
        
        self.btn_reset = QPushButton("초기화")
        set_pixel_icon(self.btn_reset, "reset")
        self.addWidget(self.btn_reset)

        self.btn_fit_ground = QPushButton("바닥면 맞춤")
        set_pixel_icon(self.btn_fit_ground, "ground")
        self.btn_fit_ground.setToolTip("현재 자세를 유지한 채 메쉬 최저점을 XY 바닥(Z=0)에 맞춥니다.")
        self.addWidget(self.btn_fit_ground)
        
        self.btn_flat = QPushButton("Flat Shading")
        set_pixel_icon(self.btn_flat, "flat")
        self.btn_flat.setCheckable(True)
        self.btn_flat.setToolTip("명암 없이 메쉬를 밝게 봅니다 (회전 시 어두워짐 방지)")
        self.addWidget(self.btn_flat)

        self.btn_xray = QPushButton("X-Ray")
        set_pixel_icon(self.btn_xray, "xray")
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
            "바닥 면 그리기: 상단 툴바 버튼 → 메쉬 클릭으로 점 추가 → Enter로 확정"
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
        set_pixel_icon(btn_open_mesh, "open_mesh")
        btn_open_mesh.clicked.connect(lambda: self.workflowRequested.emit("open_mesh", None))
        align_layout.addWidget(btn_open_mesh)
        btn_open_project = QPushButton("프로젝트 열기")
        set_pixel_icon(btn_open_project, "open_project")
        btn_open_project.clicked.connect(lambda: self.workflowRequested.emit("open_project", None))
        align_layout.addWidget(btn_open_project)
        btn_fit = QPushButton("메쉬에 맞춤")
        set_pixel_icon(btn_fit, "fit")
        btn_fit.clicked.connect(lambda: self.workflowRequested.emit("fit_view", None))
        align_layout.addWidget(btn_fit)
        view_grid = QGridLayout()
        views = [
            ("상면", "top"), ("정면", "front"), ("우측", "right"),
            ("하면", "bottom"), ("후면", "back"), ("좌측", "left"),
        ]
        for idx, (label, key) in enumerate(views):
            btn = QPushButton(label)
            set_pixel_icon(btn, f"view_{key}")
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
        set_pixel_icon(self.btn_interpret_next, "details")
        self.btn_interpret_next.clicked.connect(lambda: self.workflowRequested.emit("run_interpretation_next", None))
        self.btn_interpret_next.setEnabled(False)
        interpret_layout.addWidget(self.btn_interpret_next)

        self.btn_authoritative_measurements = QPushButton(
            "검증된 실측 · 기와 전개 열기"
        )
        set_pixel_icon(self.btn_authoritative_measurements, "flatten")
        self.btn_authoritative_measurements.setToolTip(
            "ArtifactDocument의 Cutline·Outline·Digital Rubbing·기와 전개 record와 "
            "1:1 검증 export 패널을 엽니다."
        )
        self.btn_authoritative_measurements.clicked.connect(
            lambda: self.workflowRequested.emit("show_section_tools", None)
        )
        self.btn_authoritative_measurements.setEnabled(False)
        interpret_layout.addWidget(self.btn_authoritative_measurements)

        btn_drawing_svg = QPushButton("실측용 SVG 저장")
        set_pixel_icon(btn_drawing_svg, "export")
        btn_drawing_svg.clicked.connect(lambda: self.workflowRequested.emit("export_flat_svg", None))
        interpret_layout.addWidget(btn_drawing_svg)

        btn_drawing_package = QPushButton("6방향 도면 패키지 저장")
        set_pixel_icon(btn_drawing_package, "save")
        btn_drawing_package.clicked.connect(lambda: self.workflowRequested.emit("export_profile_package", None))
        interpret_layout.addWidget(btn_drawing_package)
        layout.addWidget(interpret_group)

        record_group = QGroupBox("3. 탁본")
        record_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        record_layout = QVBoxLayout(record_group)
        btn_record_top = QPushButton("상면 기록 준비")
        set_pixel_icon(btn_record_top, "record_top")
        btn_record_top.clicked.connect(
            lambda: self.workflowRequested.emit("prepare_record_surface", {"view": "top"})
        )
        record_layout.addWidget(btn_record_top)
        btn_record_bottom = QPushButton("하면 기록 준비")
        set_pixel_icon(btn_record_bottom, "record_bottom")
        btn_record_bottom.clicked.connect(
            lambda: self.workflowRequested.emit("prepare_record_surface", {"view": "bottom"})
        )
        record_layout.addWidget(btn_record_bottom)
        btn_preview = QPushButton("기록면 미리보기")
        set_pixel_icon(btn_preview, "preview")
        btn_preview.clicked.connect(lambda: self.workflowRequested.emit("preview_recording_surface", None))
        record_layout.addWidget(btn_preview)
        btn_export_review = QPushButton("기록면 검토 시트 저장")
        set_pixel_icon(btn_export_review, "export")
        btn_export_review.clicked.connect(lambda: self.workflowRequested.emit("export_review_sheet", None))
        record_layout.addWidget(btn_export_review)
        layout.addWidget(record_group)

        measure_group = QGroupBox("4. 제원측정")
        measure_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        measure_layout = QVBoxLayout(measure_group)
        btn_measure = QPushButton("제원측정 도구 열기")
        set_pixel_icon(btn_measure, "measure")
        btn_measure.clicked.connect(lambda: self.workflowRequested.emit("show_measure_tools", None))
        measure_layout.addWidget(btn_measure)
        layout.addWidget(measure_group)

        btn_advanced = QPushButton("세부 패널 열기")
        set_pixel_icon(btn_advanced, "details")
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
            self.btn_authoritative_measurements.setEnabled(False)
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
        self.btn_authoritative_measurements.setEnabled(True)
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
    methodChanged = pyqtSignal(int)
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
        curve_group = QGroupBox("곡률 설정")
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
        self.btn_measure = QPushButton("곡률 측정")
        set_pixel_icon(self.btn_measure, "measure")
        self.btn_measure.setCheckable(True)
        self.btn_measure.setToolTip("Shift+클릭으로 메쉬 위에 점을 3개 이상 찍으면 곡률을 계산합니다")
        measure_layout.addWidget(self.btn_measure)
        
        self.btn_fit_arc = QPushButton("원호 피팅")
        set_pixel_icon(self.btn_fit_arc, "cutline")
        self.btn_fit_arc.setToolTip("찍은 점들로 원호를 피팅하고 반지름을 계산합니다")
        measure_layout.addWidget(self.btn_fit_arc)
        
        self.btn_clear_points = QPushButton("")
        set_pixel_icon(self.btn_clear_points, "delete")
        self.btn_clear_points.setToolTip("찍은 점 초기화")
        self.btn_clear_points.setFixedWidth(40)
        measure_layout.addWidget(self.btn_clear_points)
        
        curve_layout.addRow(measure_layout)
        
        # 원호 관리
        arc_layout = QHBoxLayout()
        arc_label = QLabel("부착된 원호:")
        arc_layout.addWidget(arc_label)
        arc_layout.addStretch()
        
        self.btn_clear_arcs = QPushButton("모든 원호 삭제")
        set_pixel_icon(self.btn_clear_arcs, "delete")
        self.btn_clear_arcs.setToolTip("선택된 객체의 모든 원호 삭제")
        arc_layout.addWidget(self.btn_clear_arcs)
        curve_layout.addRow(arc_layout)
        
        layout.addWidget(curve_group)
        self.curve_group = curve_group
        
        # 기록면 전개 방법
        method_group = QGroupBox("기록면 전개 방법")
        method_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        method_layout = QVBoxLayout(method_group)
        
        self.combo_method = QComboBox()
        self.combo_method.addItems([
            _METHOD_NAME_ARAP,
            _METHOD_NAME_LSCM,
            _METHOD_NAME_AREA,
            _METHOD_NAME_CYLINDER,
            _METHOD_NAME_SECTION,
        ])
        self.combo_method.setToolTip("기록면 전개 알고리즘 선택")
        self.combo_method.currentIndexChanged.connect(
            lambda idx: self.methodChanged.emit(int(idx))
        )
        method_layout.addWidget(self.combo_method)

        self.label_recommendation = QLabel("")
        self.label_recommendation.setWordWrap(True)
        self.label_recommendation.setVisible(False)
        self.label_recommendation.setStyleSheet(
            "color: #2d3748; background-color: #fffaf0; border: 1px solid #f6ad55;"
            " border-radius: 4px; padding: 6px; font-size: 11px;"
        )
        method_layout.addWidget(self.label_recommendation)
        
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
        adv_group = QGroupBox("고급 옵션")
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
        surface_group = QGroupBox("고급 표면 라벨링 (외면/내면/미구)")
        surface_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        surface_layout = QVBoxLayout(surface_group)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("대상:"))
        self.combo_surface_target = QComboBox()
        self.combo_surface_target.addItems(["외면", "내면", "미구"])
        self.combo_surface_target.setToolTip("라벨링할 표면 그룹 선택")
        self.combo_surface_target.currentIndexChanged.connect(
            lambda _i: self.selectionRequested.emit("surface_target", self.current_surface_target())
        )
        target_row.addWidget(self.combo_surface_target)
        surface_layout.addLayout(target_row)

        tool_row = QHBoxLayout()
        self.btn_surface_boundary = QPushButton("경계(면적+자석)")
        set_pixel_icon(self.btn_surface_boundary, "outline")
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
        btn_clear_target = QPushButton("현재 비우기")
        set_pixel_icon(btn_clear_target, "delete")
        btn_clear_target.setToolTip("현재 대상(외/내/미구) 지정 면을 모두 비웁니다.")
        btn_clear_target.clicked.connect(
            lambda: self.selectionRequested.emit("surface_clear_target", self.current_surface_target())
        )
        action_row.addWidget(btn_clear_target)

        btn_clear_all = QPushButton("전체 초기화")
        set_pixel_icon(btn_clear_all, "reset")
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
        self.btn_flatten = QPushButton("기록면 전개 실행")
        set_pixel_icon(self.btn_flatten, "flatten")
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

        self.btn_preview = QPushButton("기록면 미리보기")
        set_pixel_icon(self.btn_preview, "preview")
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

    def set_flatten_method_recommendation(
        self,
        method_label: str,
        reason: str,
        *,
        auto_applied: bool,
        alternatives: list[str] | None = None,
        fallback_hint: str = "",
    ) -> None:
        method_label = str(method_label or "").strip()
        reason = str(reason or "").strip()
        if not method_label:
            self.clear_flatten_method_recommendation()
            return
        if not reason:
            reason = "기와형/곡면 단면 반복성 기반으로 기와 추천 펼침이 기본 추천됩니다."
        status = "현재 기본값에 적용됨" if bool(auto_applied) else "수동 선택 후 변경 가능"
        extras: list[str] = []
        alt_labels = [str(item or "").strip() for item in list(alternatives or []) if str(item or "").strip()]
        if alt_labels:
            extras.append(f"다른 선택: {', '.join(alt_labels[:3])}")
        fallback_hint = str(fallback_hint or "").strip()
        if fallback_hint:
            extras.append(f"문제 시 {fallback_hint}")
        tail = f"<br><span style='color:#4a5568;'>{' | '.join(extras)}</span>" if extras else ""
        self.label_recommendation.setText(
            f"<b>[{_SECTION_RECOMMEND_TAG}]</b> {status} | {method_label}: {reason}{tail}"
        )
        self.label_recommendation.setVisible(True)

    def clear_flatten_method_recommendation(self) -> None:
        self.label_recommendation.setText("")
        self.label_recommendation.setVisible(False)
    
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
        tool_group = QGroupBox("선택 도구")
        tool_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        tool_layout = QVBoxLayout(tool_group)
        
        # 버튼 그룹 (상호 배타적)
        self.tool_button_group = QButtonGroup(self)
        
        self.btn_click = QPushButton("클릭 선택")
        set_pixel_icon(self.btn_click, "selection")
        self.btn_click.setCheckable(True)
        self.btn_click.setChecked(True)
        self.btn_click.setToolTip("Shift+클릭으로 면 선택")
        self.btn_click.clicked.connect(lambda: self.selectionChanged.emit("tool", {"tool": "click"}))
        self.tool_button_group.addButton(self.btn_click, 0)
        tool_layout.addWidget(self.btn_click)
        
        self.btn_brush = QPushButton("브러시 선택")
        set_pixel_icon(self.btn_brush, "selection")
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
        
        self.btn_lasso = QPushButton("올가미 선택")
        set_pixel_icon(self.btn_lasso, "outline")
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
        auto_group = QGroupBox("고급 표면 라벨링")
        auto_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        auto_layout = QVBoxLayout(auto_group)
        
        btn_auto_surface = QPushButton("외면/내면 자동 라벨링")
        set_pixel_icon(btn_auto_surface, "details")
        btn_auto_surface.setToolTip(
            "외면/내면을 자동 추정해 현재 메쉬에 라벨로 저장합니다.\n"
            "권장 기본 흐름은 아니며, 외면/내면 구분이 꼭 필요할 때만 사용하세요.\n"
            "클릭=스마트(auto: 가시성(위상)→원통→법선), Shift+클릭=가시성(±두께축) 강제, Ctrl+클릭=원통(반경) 강제"
        )
        btn_auto_surface.clicked.connect(lambda: self.selectionChanged.emit('auto_surface', None))
        auto_layout.addWidget(btn_auto_surface)
        
        btn_auto_edge = QPushButton("미구 자동 감지")
        set_pixel_icon(btn_auto_edge, "measure")
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
        edit_group = QGroupBox("선택 편집")
        edit_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        edit_layout = QVBoxLayout(edit_group)
        
        btn_row = QHBoxLayout()
        btn_grow = QPushButton("확장")
        btn_grow.setToolTip("선택 영역을 인접 면으로 확장")
        btn_grow.clicked.connect(lambda: self.selectionChanged.emit('grow', None))
        btn_row.addWidget(btn_grow)
        
        btn_shrink = QPushButton("축소")
        btn_shrink.setToolTip("선택 영역 가장자리 제거")
        btn_shrink.clicked.connect(lambda: self.selectionChanged.emit('shrink', None))
        btn_row.addWidget(btn_shrink)
        edit_layout.addLayout(btn_row)
        
        btn_row2 = QHBoxLayout()
        btn_invert = QPushButton("반전")
        set_pixel_icon(btn_invert, "reset")
        btn_invert.setToolTip("선택/비선택 반전")
        btn_invert.clicked.connect(lambda: self.selectionChanged.emit('invert', None))
        btn_row2.addWidget(btn_invert)
        
        btn_clear = QPushButton("해제")
        set_pixel_icon(btn_clear, "delete")
        btn_clear.setToolTip("모든 선택 해제")
        btn_clear.clicked.connect(lambda: self.selectionChanged.emit('clear', None))
        btn_row2.addWidget(btn_clear)
        edit_layout.addLayout(btn_row2)

        btn_visible = QPushButton("현재 시점 가시면")
        set_pixel_icon(btn_visible, "visible")
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
        assign_group = QGroupBox("영역 지정")
        assign_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        assign_layout = QVBoxLayout(assign_group)
        
        btn_outer = QPushButton("선택 → 외면")
        btn_outer.setStyleSheet("background-color: #ebf8ff; color: #2b6cb0;")
        btn_outer.clicked.connect(lambda: self.selectionChanged.emit('assign_outer', None))
        assign_layout.addWidget(btn_outer)
        
        btn_inner = QPushButton("선택 → 내면")
        btn_inner.setStyleSheet("background-color: #faf5ff; color: #6b46c1;")
        btn_inner.clicked.connect(lambda: self.selectionChanged.emit('assign_inner', None))
        assign_layout.addWidget(btn_inner)
        
        btn_migu = QPushButton("선택 → 미구")
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

        hypo_group = QGroupBox("제작 가설")
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

        axis_group = QGroupBox("길이축 힌트")
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

        btn_axis_clear = QPushButton("길이축 힌트 초기화")
        btn_axis_clear.clicked.connect(lambda: self.interpretationChanged.emit("clear_axis", None))
        axis_detail_layout.addWidget(btn_axis_clear)
        axis_layout.addWidget(axis_detail_widget)
        self.axis_detail_widget = axis_detail_widget

        layout.addWidget(axis_group)

        section_group = QGroupBox("대표 단면 후보")
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

        btn_section_clear = QPushButton("단면 후보 초기화")
        btn_section_clear.clicked.connect(lambda: self.interpretationChanged.emit("clear_sections", None))
        section_detail_layout.addWidget(btn_section_clear)
        section_layout.addWidget(section_detail_widget)
        self.section_detail_widget = section_detail_widget

        layout.addWidget(section_group)

        fit_group = QGroupBox("와통 초벌 피팅")
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

        btn_fit_clear = QPushButton("피팅 결과 초기화")
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

        record_group = QGroupBox("탁본 기록면 보조")
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

        btn_record_clear = QPushButton("기록면 준비 해제")
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

        slot_group = QGroupBox("작업 슬롯")
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

        btn_clear_slots = QPushButton("작업 슬롯 모두 비우기")
        btn_clear_slots.clicked.connect(lambda: self.interpretationChanged.emit("clear_slots", None))
        slot_layout.addWidget(btn_clear_slots)
        self.btn_clear_slots = btn_clear_slots

        btn_export_slots = QPushButton("저장 슬롯 검토 시트 묶음 저장")
        btn_export_slots.setToolTip("저장된 슬롯별로 기록면 검토 시트를 한 번에 생성합니다.")
        btn_export_slots.clicked.connect(
            lambda: self.interpretationChanged.emit("export_saved_slots_review", None)
        )
        slot_layout.addWidget(btn_export_slots)
        self.btn_export_slots = btn_export_slots

        layout.addWidget(slot_group)
        self.slot_group = slot_group

        wizard_group = QGroupBox("기와 실측 위저드")
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

        synth_group = QGroupBox("합성 데이터 / 정답 평가")
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
        file_group = QGroupBox("파일 정보")
        file_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        file_layout = QFormLayout(file_group)
        
        self.label_filename = QLabel("-")
        self.label_filename.setWordWrap(True)
        file_layout.addRow("파일:", self.label_filename)
        
        layout.addWidget(file_group)
        
        # 메쉬 정보
        mesh_group = QGroupBox("메쉬 정보")
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
        region_group = QGroupBox("영역 정보")
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
        self.group = QGroupBox("메쉬 단면 슬라이싱")
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

        self.btn_preset_add = QPushButton("저장")
        set_pixel_icon(self.btn_preset_add, "save")
        self.btn_preset_add.setToolTip("현재 높이(Z)를 프리셋으로 저장합니다.")
        self.btn_preset_add.clicked.connect(self._on_preset_add_clicked)
        preset_layout.addWidget(self.btn_preset_add)

        self.btn_preset_apply = QPushButton("적용")
        self.btn_preset_apply.setToolTip("선택한 프리셋 높이를 적용합니다.")
        self.btn_preset_apply.clicked.connect(self._on_preset_apply_clicked)
        preset_layout.addWidget(self.btn_preset_apply)

        self.btn_preset_delete = QPushButton("삭제")
        set_pixel_icon(self.btn_preset_delete, "delete")
        self.btn_preset_delete.setToolTip("선택한 프리셋을 삭제합니다.")
        self.btn_preset_delete.clicked.connect(self._on_preset_delete_clicked)
        preset_layout.addWidget(self.btn_preset_delete)

        group_layout.addLayout(preset_layout)
        self._refresh_presets_ui()
        
        # 3. 버튼들
        btn_layout = QHBoxLayout()
        self.btn_export = QPushButton("단면 SVG 내보내기")
        set_pixel_icon(self.btn_export, "export")
        self.btn_export.setStyleSheet("background-color: #ebf8ff; font-weight: bold;")
        self.btn_export.clicked.connect(self.on_export_clicked)
        btn_layout.addWidget(self.btn_export)

        self.btn_capture = QPushButton("현재 단면 촬영")
        set_pixel_icon(self.btn_capture, "camera")
        self.btn_capture.setStyleSheet("background-color: #fff7ed; font-weight: bold;")
        self.btn_capture.setToolTip("현재 보이는 메쉬 단면을 레이어로 바로 저장합니다.")
        self.btn_capture.clicked.connect(self.on_capture_clicked)
        btn_layout.addWidget(self.btn_capture)

        self.btn_save_layers = QPushButton("레이어로 저장")
        set_pixel_icon(self.btn_save_layers, "save")
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

        img_group = QGroupBox("기본 출력 설정")
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
        self.combo_review_render_mode.addItem("노멀 언샵", "노멀 언샵")
        self.combo_review_render_mode.addItem("스펙큘러 강조", "스펙큘러 강조")
        self.combo_review_render_mode.addItem("노멀 보기", "노멀 보기")
        self.combo_review_render_mode.addItem("자연(이미지)", "자연(이미지)")
        self.combo_review_render_mode.setToolTip(
            "검토 시트와 미리보기에서 사용할 기록면 렌더 모드입니다.\n"
            "자동은 기와/기록면일 때 다중광, 일반 경로는 자연(이미지)를 사용합니다."
        )
        img_layout.addRow("기록면 렌더:", self.combo_review_render_mode)

        self.combo_rubbing_target = QComboBox()
        self.combo_rubbing_target.addItems(["전체 메쉬", "현재 선택"])
        self.combo_rubbing_target.setToolTip(
            "기본 도면 생성에서 사용할 대상을 고릅니다.\n"
            "기본 흐름은 '현재 선택' 또는 기와 모드의 상면/하면 기록 준비 결과를 사용하는 것입니다."
        )
        img_layout.addRow("도면 대상:", self.combo_rubbing_target)

        layout.addWidget(img_group)

        btn_export_review_sheet = QPushButton("기록면 검토 시트 저장")
        set_pixel_icon(btn_export_review_sheet, "export")
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
        set_pixel_icon(btn_export_flat_svg, "export")
        btn_export_flat_svg.setToolTip(
            "전개 결과의 외곽선을 실측 SVG로 저장합니다.\n"
            "기본 출력은 연속 표면의 외곽선만 포함하며, 와이어프레임은 넣지 않습니다."
        )
        btn_export_flat_svg.clicked.connect(
            lambda: self.exportRequested.emit({'type': 'flat_svg', 'target': self.current_rubbing_target()})
        )
        layout.addWidget(btn_export_flat_svg)

        profile_group = QGroupBox("6방향 도면 패키지")
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
        self.check_profile_feature_lines = QCheckBox("샤프 엣지(능선) 라인 포함")
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

        btn_export_pkg = QPushButton("6방향 패키지 내보내기")
        set_pixel_icon(btn_export_pkg, "export")
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
            "Native 문서의 Shift+클릭은 원본 삼각형·barycentric 좌표로 결박됩니다.\n"
            "거리·지름·표면적·체적은 명시 단위와 Align에 연결된 재검산 가능 기록입니다."
        )
        hint.setStyleSheet("color: #718096; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.btn_measure_mode = QPushButton("측정 모드 시작")
        set_pixel_icon(self.btn_measure_mode, "measure")
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
        self.combo_mode.addItems(["검증 거리 (2점)", "검증 원 맞춤 지름 (3~64점)"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addRow("모드:", self.combo_mode)

        self.label_point_count = QLabel("선택된 포인트: 0")
        mode_layout.addRow("", self.label_point_count)

        btn_row = QHBoxLayout()
        self.btn_fit_circle = QPushButton("지름 계산 · 기록")
        set_pixel_icon(self.btn_fit_circle, "measure")
        self.btn_fit_circle.setToolTip(
            "3~64개의 표면 anchor에 PCA 평면·정규화 대수 Kasa 원을 맞추고 잔차 QC와 함께 기록합니다."
        )
        self.btn_fit_circle.clicked.connect(self.fitCircleRequested.emit)
        self.btn_fit_circle.setEnabled(False)
        btn_row.addWidget(self.btn_fit_circle)

        self.btn_clear_points = QPushButton("포인트 초기화")
        set_pixel_icon(self.btn_clear_points, "reset")
        self.btn_clear_points.clicked.connect(self.clearPointsRequested.emit)
        btn_row.addWidget(self.btn_clear_points)
        btn_row.addStretch(1)
        mode_layout.addRow(btn_row)

        self.btn_compute_volume = QPushButton("검증 표면적·체적 계산 · 기록")
        set_pixel_icon(self.btn_compute_volume, "measure")
        self.btn_compute_volume.setToolTip(
            "canonical mm 전체 메쉬를 1 µm 격자로 측정하고 위상 QC와 함께 기록합니다.\n"
            "열린 메쉬·비다양체·방향 불일치·다중 조각이면 체적을 제공하지 않습니다."
        )
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
        self.btn_copy = QPushButton("복사")
        set_pixel_icon(self.btn_copy, "copy")
        self.btn_copy.clicked.connect(self.copyResultsRequested.emit)
        result_btn_row.addWidget(self.btn_copy)

        self.btn_clear_results = QPushButton("지우기")
        set_pixel_icon(self.btn_clear_results, "delete")
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
            self.btn_measure_mode.setText("측정 모드 중지" if checked else "측정 모드 시작")
        except Exception:
            pass

    def _on_measure_toggled(self, checked: bool):
        try:
            self.btn_measure_mode.setText("측정 모드 중지" if checked else "측정 모드 시작")
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
    nativeCutlineRequested = pyqtSignal()
    nativeCutlineViewChanged = pyqtSignal(str)
    nativeOutlineRequested = pyqtSignal()
    nativeMeasurementRetryRequested = pyqtSignal()
    nativeVectorRecordSelected = pyqtSignal(str)
    nativeVectorExportRequested = pyqtSignal()
    nativeRubbingRequested = pyqtSignal()
    nativeRubbingRecordSelected = pyqtSignal(str)
    nativeRubbingExportRequested = pyqtSignal()
    nativeSurveyExportRequested = pyqtSignal()
    nativeTileUnwrapRequested = pyqtSignal()
    nativeTileUnwrapRecordSelected = pyqtSignal(str)
    nativeTileUnwrapExportRequested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. 활성화 버튼
        self.btn_toggle = QPushButton("십자선 단면 모드 시작")
        set_pixel_icon(self.btn_toggle, "cutline")
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

        native_group = QGroupBox("검증된 실측 · ArtifactDocument")
        native_group.setStyleSheet(
            """
            QPushButton[workflowComplete="true"] {
                background-color: #38a169;
                color: white;
                font-weight: bold;
            }
            QPushButton[workflowComplete="true"]:disabled {
                background-color: #9ae6b4;
                color: #f0fff4;
            }
            """
        )
        native_layout = QVBoxLayout(native_group)
        native_form = QFormLayout()
        self.combo_native_cutline_view = QComboBox()
        self.combo_native_cutline_view.addItem("Top · XY / Z 위치", "top")
        self.combo_native_cutline_view.addItem("Front · XZ / Y 위치", "front")
        self.combo_native_cutline_view.addItem("Right · YZ / X 위치", "right")
        self.combo_native_cutline_view.currentIndexChanged.connect(
            lambda _index: self.nativeCutlineViewChanged.emit(
                str(self.combo_native_cutline_view.currentData() or "top")
            )
        )
        native_form.addRow("단면 방향", self.combo_native_cutline_view)
        self.spin_native_cutline_offset = QDoubleSpinBox()
        self.spin_native_cutline_offset.setDecimals(4)
        self.spin_native_cutline_offset.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.spin_native_cutline_offset.setSingleStep(0.1)
        self.spin_native_cutline_offset.setSuffix(" mm")
        native_form.addRow("평면 위치", self.spin_native_cutline_offset)
        native_layout.addLayout(native_form)
        self.btn_native_cutline = QPushButton("단면 계산 · 기록")
        set_pixel_icon(self.btn_native_cutline, "cutline")
        self.btn_native_cutline.setToolTip(
            "원본에서 canonical-mm 단면을 다시 계산하고 recipe·QC와 함께 기록합니다."
        )
        self.btn_native_cutline.clicked.connect(self.nativeCutlineRequested.emit)
        native_layout.addWidget(self.btn_native_cutline)

        outline_line = QFrame()
        outline_line.setFrameShape(QFrame.Shape.HLine)
        outline_line.setFrameShadow(QFrame.Shadow.Sunken)
        native_layout.addWidget(outline_line)
        native_outline_form = QFormLayout()
        self.combo_native_outline_view = QComboBox()
        self.combo_native_outline_view.addItem("Top · +Z", "top")
        self.combo_native_outline_view.addItem("Bottom · -Z", "bottom")
        self.combo_native_outline_view.addItem("Front · -Y", "front")
        self.combo_native_outline_view.addItem("Back · +Y", "back")
        self.combo_native_outline_view.addItem("Right · +X", "right")
        self.combo_native_outline_view.addItem("Left · -X", "left")
        native_outline_form.addRow("외곽 방향", self.combo_native_outline_view)
        self.spin_native_outline_grid = QDoubleSpinBox()
        self.spin_native_outline_grid.setDecimals(6)
        self.spin_native_outline_grid.setRange(0.000001, 1_000_000.0)
        self.spin_native_outline_grid.setSingleStep(0.01)
        self.spin_native_outline_grid.setValue(DEFAULT_OUTLINE_PRECISION_GRID_MM)
        self.spin_native_outline_grid.setSuffix(" mm")
        self.spin_native_outline_grid.setToolTip(
            "삼각형 투영을 합집합하기 전 적용할 고정 mm 격자입니다. "
            "격자보다 좁은 특징은 합쳐지거나 사라질 수 있으며 QC에 기록됩니다."
        )
        native_outline_form.addRow("외곽 정밀도", self.spin_native_outline_grid)
        native_layout.addLayout(native_outline_form)
        self.btn_native_outline = QPushButton("외곽 계산 · 기록")
        set_pixel_icon(self.btn_native_outline, "outline")
        self.btn_native_outline.setToolTip(
            "전체 삼각형을 canonical-mm 평면에 투영하고 오목부·구멍·분리 성분을 "
            "보존한 Outline record를 만듭니다."
        )
        self.btn_native_outline.clicked.connect(self.nativeOutlineRequested.emit)
        native_layout.addWidget(self.btn_native_outline)
        self.btn_native_measurement_retry = QPushButton("보류 결과 게시 재시도")
        set_pixel_icon(self.btn_native_measurement_retry, "reset")
        self.btn_native_measurement_retry.setEnabled(False)
        self.btn_native_measurement_retry.setToolTip(
            "Open 또는 scene 전환 때문에 계산 완료 결과를 아직 게시하지 못한 경우, "
            "같은 operation capability로 다시 게시합니다."
        )
        self.btn_native_measurement_retry.clicked.connect(
            self.nativeMeasurementRetryRequested.emit
        )
        native_layout.addWidget(self.btn_native_measurement_retry)
        self.combo_native_vector_record = QComboBox()
        self.combo_native_vector_record.addItem(
            "READY + FRESH 벡터 기록을 선택하세요", None
        )
        self.combo_native_vector_record.setToolTip(
            "프로젝트에 저장된 Cutline·Outline 기록 중 미리보기와 "
            "1:1 SVG 내보내기에 사용할 기록을 명시적으로 고릅니다."
        )
        self.combo_native_vector_record.currentIndexChanged.connect(
            lambda _index: self.nativeVectorRecordSelected.emit(
                str(self.combo_native_vector_record.currentData() or "")
            )
        )
        native_layout.addWidget(self.combo_native_vector_record)
        self.btn_native_vector_export = QPushButton("선택한 검증 벡터 1:1 SVG 내보내기")
        set_pixel_icon(self.btn_native_vector_export, "export")
        self.btn_native_vector_export.clicked.connect(self.nativeVectorExportRequested.emit)
        native_layout.addWidget(self.btn_native_vector_export)

        rubbing_line = QFrame()
        rubbing_line.setFrameShape(QFrame.Shape.HLine)
        rubbing_line.setFrameShadow(QFrame.Shadow.Sunken)
        native_layout.addWidget(rubbing_line)
        rubbing_title = QLabel("디지털 탁본 · 재현 가능한 1:1 raster")
        rubbing_title.setStyleSheet("font-weight: bold;")
        native_layout.addWidget(rubbing_title)
        native_rubbing_form = QFormLayout()
        self.combo_native_rubbing_view = QComboBox()
        for label, value in (
            ("Top · +Z", "top"),
            ("Bottom · -Z", "bottom"),
            ("Front · -Y", "front"),
            ("Back · +Y", "back"),
            ("Right · +X", "right"),
            ("Left · -X", "left"),
        ):
            self.combo_native_rubbing_view.addItem(label, value)
        native_rubbing_form.addRow("탁본 방향", self.combo_native_rubbing_view)

        self.spin_native_rubbing_pixels_per_mm = QSpinBox()
        self.spin_native_rubbing_pixels_per_mm.setRange(1, 100)
        self.spin_native_rubbing_pixels_per_mm.setValue(DEFAULT_RUBBING_PIXELS_PER_MM)
        self.spin_native_rubbing_pixels_per_mm.setSuffix(" px/mm")
        native_rubbing_form.addRow("해상도", self.spin_native_rubbing_pixels_per_mm)

        def _micrometre_spin(value: int, *, minimum: int = 1) -> QSpinBox:
            spin = QSpinBox()
            spin.setRange(minimum, 1_000_000_000)
            spin.setValue(int(value))
            spin.setSuffix(" µm")
            return spin

        self.spin_native_rubbing_margin_um = _micrometre_spin(
            DEFAULT_RUBBING_MARGIN_UM,
            minimum=0,
        )
        native_rubbing_form.addRow("여백", self.spin_native_rubbing_margin_um)
        self.spin_native_rubbing_reference_radius_um = _micrometre_spin(
            DEFAULT_RUBBING_REFERENCE_RADIUS_UM
        )
        native_rubbing_form.addRow(
            "표면 기준 반경", self.spin_native_rubbing_reference_radius_um
        )
        self.spin_native_rubbing_depth_quantization_um = _micrometre_spin(
            DEFAULT_RUBBING_DEPTH_QUANTIZATION_UM
        )
        self.spin_native_rubbing_depth_quantization_um.setMaximum(1_000_000)
        native_rubbing_form.addRow(
            "깊이 양자화", self.spin_native_rubbing_depth_quantization_um
        )
        self.spin_native_rubbing_black_point_um = _micrometre_spin(
            DEFAULT_RUBBING_BLACK_POINT_UM
        )
        native_rubbing_form.addRow(
            "검정 기준 깊이", self.spin_native_rubbing_black_point_um
        )
        self.spin_native_rubbing_strength = QSpinBox()
        self.spin_native_rubbing_strength.setRange(1, 400)
        self.spin_native_rubbing_strength.setValue(
            DEFAULT_RUBBING_INK_STRENGTH_PERCENT
        )
        self.spin_native_rubbing_strength.setSuffix(" %")
        native_rubbing_form.addRow("먹 농도", self.spin_native_rubbing_strength)
        self.combo_native_rubbing_polarity = QComboBox()
        self.combo_native_rubbing_polarity.addItem("양각·음각 모두", "bidirectional")
        self.combo_native_rubbing_polarity.addItem("양각", "raised")
        self.combo_native_rubbing_polarity.addItem("음각", "incised")
        polarity_index = self.combo_native_rubbing_polarity.findData(
            DEFAULT_RUBBING_POLARITY
        )
        self.combo_native_rubbing_polarity.setCurrentIndex(max(0, polarity_index))
        native_rubbing_form.addRow("표면 극성", self.combo_native_rubbing_polarity)
        native_layout.addLayout(native_rubbing_form)

        self.btn_native_rubbing = QPushButton("탁본 계산 · 기록")
        set_pixel_icon(self.btn_native_rubbing, "rubbing")
        self.btn_native_rubbing.setToolTip(
            "현재 source·단위·Align에서 CPU로 다시 계산하고 recipe·QC·raster receipt를 기록합니다."
        )
        self.btn_native_rubbing.clicked.connect(self.nativeRubbingRequested.emit)
        native_layout.addWidget(self.btn_native_rubbing)
        self.combo_native_rubbing_record = QComboBox()
        self.combo_native_rubbing_record.addItem(
            "READY + FRESH 탁본 기록을 선택하세요", None
        )
        self.combo_native_rubbing_record.setToolTip(
            "프로젝트에 저장된 Digital Rubbing 기록 중 미리보기와 "
            "1:1 PNG 내보내기에 사용할 기록을 명시적으로 고릅니다."
        )
        self.combo_native_rubbing_record.currentIndexChanged.connect(
            lambda _index: self.nativeRubbingRecordSelected.emit(
                str(self.combo_native_rubbing_record.currentData() or "")
            )
        )
        native_layout.addWidget(self.combo_native_rubbing_record)
        self.btn_native_rubbing_export = QPushButton(
            "선택한 검증 탁본 1:1 PNG 패키지 내보내기"
        )
        set_pixel_icon(self.btn_native_rubbing_export, "export")
        self.btn_native_rubbing_export.clicked.connect(
            self.nativeRubbingExportRequested.emit
        )
        native_layout.addWidget(self.btn_native_rubbing_export)
        self.label_native_rubbing_preview = QLabel(
            "계산된 탁본 미리보기는 여기에 표시됩니다.\n"
            "미리보기 픽셀은 export 권위가 아니며 record recipe에서 다시 계산합니다."
        )
        self.label_native_rubbing_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_native_rubbing_preview.setWordWrap(True)
        self.label_native_rubbing_preview.setMinimumHeight(160)
        self.label_native_rubbing_preview.setStyleSheet(
            "background: #f7fafc; border: 1px solid #cbd5e0; color: #4a5568;"
        )
        native_layout.addWidget(self.label_native_rubbing_preview)
        self.label_native_rubbing_info = QLabel("READY + FRESH 탁본 기록 없음")
        self.label_native_rubbing_info.setWordWrap(True)
        self.label_native_rubbing_info.setStyleSheet(
            "color: #4a5568; font-size: 10px;"
        )
        native_layout.addWidget(self.label_native_rubbing_info)

        survey_line = QFrame()
        survey_line.setFrameShape(QFrame.Shape.HLine)
        survey_line.setFrameShadow(QFrame.Shadow.Sunken)
        native_layout.addWidget(survey_line)
        self.btn_native_survey_export = QPushButton(
            "완료 실측 15개 원자 묶음 내보내기"
        )
        set_pixel_icon(self.btn_native_survey_export, "export")
        self.btn_native_survey_export.setToolTip(
            "Top·Front·Right Cutline 3개, 6면 Outline 6개, 6면 Digital Rubbing "
            "6개를 하나의 검증 가능한 .amr-survey 디렉터리로 원자 게시합니다."
        )
        self.btn_native_survey_export.clicked.connect(
            self.nativeSurveyExportRequested.emit
        )
        native_layout.addWidget(self.btn_native_survey_export)

        tile_line = QFrame()
        tile_line.setFrameShape(QFrame.Shape.HLine)
        tile_line.setFrameShadow(QFrame.Shadow.Sunken)
        native_layout.addWidget(tile_line)
        tile_title = QLabel("기와 기록면 전개 · 원본 보존형 1:1 좌표")
        tile_title.setStyleSheet("font-weight: bold;")
        native_layout.addWidget(tile_title)
        tile_form = QFormLayout()
        self.combo_native_tile_target = QComboBox()
        self.combo_native_tile_target.addItem("전체 메쉬", "all")
        self.combo_native_tile_target.addItem("현재 선택 면", "selected")
        self.combo_native_tile_target.setToolTip(
            "현재 선택 면을 고르면 선택 face ID가 recipe와 selection hash에 고정됩니다."
        )
        tile_form.addRow("기록 영역", self.combo_native_tile_target)
        self.combo_native_tile_axis = QComboBox()
        self.combo_native_tile_axis.addItem("X축", "x")
        self.combo_native_tile_axis.addItem("Y축", "y")
        self.combo_native_tile_axis.addItem("Z축", "z")
        self.combo_native_tile_axis.setCurrentIndex(1)
        self.combo_native_tile_axis.setToolTip(
            "자동 추정 없이 정렬된 canonical X/Y/Z 중 기와 길이축을 명시합니다."
        )
        tile_form.addRow("길이축", self.combo_native_tile_axis)
        self.combo_native_tile_record_view = QComboBox()
        self.combo_native_tile_record_view.addItem("상면", "top")
        self.combo_native_tile_record_view.addItem("하면", "bottom")
        tile_form.addRow("기록면", self.combo_native_tile_record_view)
        self.spin_native_tile_sections = QSpinBox()
        self.spin_native_tile_sections.setRange(12, 512)
        self.spin_native_tile_sections.setValue(32)
        self.spin_native_tile_sections.setSuffix(" 구간")
        self.spin_native_tile_sections.setToolTip(
            "길이축을 따라 가변 반지름 단면을 적합할 구간 수입니다. 값은 recipe에 기록됩니다."
        )
        tile_form.addRow("단면 수", self.spin_native_tile_sections)
        native_layout.addLayout(tile_form)
        self.label_native_tile_selection = QLabel("현재 선택 0면 · 전체 사용 가능")
        self.label_native_tile_selection.setStyleSheet(
            "color: #4a5568; font-size: 10px;"
        )
        native_layout.addWidget(self.label_native_tile_selection)
        self.btn_native_tile_unwrap = QPushButton("기와 전개 계산 · 기록")
        set_pixel_icon(self.btn_native_tile_unwrap, "flatten")
        self.btn_native_tile_unwrap.setToolTip(
            "원본 canonical-mm 메쉬에서 variable-radius sectionwise 전개를 계산하고 "
            "선택·축·기록면·왜곡·foldover QC와 함께 기록합니다."
        )
        self.btn_native_tile_unwrap.clicked.connect(
            self.nativeTileUnwrapRequested.emit
        )
        native_layout.addWidget(self.btn_native_tile_unwrap)
        self.combo_native_tile_unwrap_record = QComboBox()
        self.combo_native_tile_unwrap_record.addItem(
            "READY + FRESH 기와 전개 기록을 선택하세요", None
        )
        self.combo_native_tile_unwrap_record.setToolTip(
            "저장된 전개 record를 recipe로 재계산해 미리보기·QC·1:1 export에 사용합니다."
        )
        self.combo_native_tile_unwrap_record.currentIndexChanged.connect(
            lambda _index: self.nativeTileUnwrapRecordSelected.emit(
                str(self.combo_native_tile_unwrap_record.currentData() or "")
            )
        )
        native_layout.addWidget(self.combo_native_tile_unwrap_record)
        self.btn_native_tile_unwrap_export = QPushButton(
            "선택한 검증 전개 1:1 OBJ · SVG 패키지 내보내기"
        )
        set_pixel_icon(self.btn_native_tile_unwrap_export, "export")
        self.btn_native_tile_unwrap_export.clicked.connect(
            self.nativeTileUnwrapExportRequested.emit
        )
        native_layout.addWidget(self.btn_native_tile_unwrap_export)
        self.label_native_tile_unwrap_preview = QLabel(
            "전개 미리보기는 여기에 표시됩니다.\n"
            "화면 이미지는 측정 권위가 아니며 µm 좌표 record를 다시 계산합니다."
        )
        self.label_native_tile_unwrap_preview.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.label_native_tile_unwrap_preview.setWordWrap(True)
        self.label_native_tile_unwrap_preview.setMinimumHeight(160)
        self.label_native_tile_unwrap_preview.setStyleSheet(
            "background: #f7fafc; border: 1px solid #cbd5e0; color: #4a5568;"
        )
        native_layout.addWidget(self.label_native_tile_unwrap_preview)
        self.label_native_tile_unwrap_info = QLabel(
            "READY + FRESH 기와 전개 기록 없음"
        )
        self.label_native_tile_unwrap_info.setWordWrap(True)
        self.label_native_tile_unwrap_info.setStyleSheet(
            "color: #4a5568; font-size: 10px;"
        )
        native_layout.addWidget(self.label_native_tile_unwrap_info)
        native_note = QLabel(
            "초록 선은 기록 payload에서 다시 만든 화면 투영입니다. "
            "Cutline과 6면 Outline은 원본에서 다시 계산되며 화면 픽셀을 측정값으로 쓰지 않습니다."
        )
        native_note.setWordWrap(True)
        native_note.setStyleSheet("color: #4a5568; font-size: 10px;")
        native_layout.addWidget(native_note)
        self.apply_native_workflow_progress(ArtifactWorkflowProgress.empty())
        native_group.setEnabled(False)
        self.native_group = native_group
        layout.addWidget(native_group)

        # 4. 2D 단면선(2개) - 상면에서 가로/세로(꺾임 가능) 가이드 라인
        line_group = QGroupBox("2D 단면선 지정 (상면, 2개)")
        self.legacy_line_group = line_group
        line_layout = QVBoxLayout(line_group)

        self.btn_line = QPushButton("단면선 그리기 시작")
        set_pixel_icon(self.btn_line, "cutline")
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

        self.btn_cutline_clear = QPushButton("현재 선 지우기")
        set_pixel_icon(self.btn_cutline_clear, "delete")
        self.btn_cutline_clear.clicked.connect(
            lambda: self.cutLineClearRequested.emit(int(self.combo_cutline.currentIndex()))
        )
        sel_row.addWidget(self.btn_cutline_clear)

        self.btn_cutline_clear_all = QPushButton("모두 지우기")
        set_pixel_icon(self.btn_cutline_clear_all, "delete")
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
        set_pixel_icon(self.btn_save_section_layers, "save")
        self.btn_save_section_layers.setToolTip("현재 단면선/단면 결과를 레이어로 스냅샷 저장합니다.")
        self.btn_save_section_layers.clicked.connect(self.saveSectionLayersRequested.emit)
        line_layout.addWidget(self.btn_save_section_layers)

        layout.addWidget(line_group)

        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line2)
        
        # 5. 2D ROI 영역 지정 (상면 투영)
        roi_group = QGroupBox("2D 영역 지정 (상면 Cropping)")
        self.legacy_roi_group = roi_group
        roi_layout = QVBoxLayout(roi_group)
        
        self.btn_roi = QPushButton("영역 지정 모드 시작")
        set_pixel_icon(self.btn_roi, "selection")
        self.btn_roi.setCheckable(True)
        self.btn_roi.setStyleSheet("QPushButton:checked { background-color: #4299e1; color: white; }")
        self.btn_roi.toggled.connect(self.on_roi_toggled)
        roi_layout.addWidget(self.btn_roi)
        
        self.btn_silhouette = QPushButton("영역 확정 및 외곽 추출")
        set_pixel_icon(self.btn_silhouette, "outline")
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
        
    @staticmethod
    def _apply_native_step_progress(button, label: str, progress) -> None:
        button.setText(
            f"{label} ({progress.completed_count}/{progress.required_count})"
        )
        button.setEnabled(progress.enabled)
        button.setProperty("workflowComplete", progress.complete)
        style = button.style()
        style.unpolish(button)
        style.polish(button)
        button.update()

    def apply_native_workflow_progress(
        self,
        progress: ArtifactWorkflowProgress,
    ) -> None:
        if not isinstance(progress, ArtifactWorkflowProgress):
            raise TypeError("progress must be ArtifactWorkflowProgress")
        self._apply_native_step_progress(
            self.btn_native_cutline,
            "단면 계산 · 기록",
            progress.cutline,
        )
        self._apply_native_step_progress(
            self.btn_native_outline,
            "외곽 계산 · 기록",
            progress.outline,
        )
        self._apply_native_step_progress(
            self.btn_native_rubbing,
            "탁본 계산 · 기록",
            progress.rubbing,
        )
        self.btn_native_survey_export.setEnabled(progress.rubbing.complete)
        self.btn_native_survey_export.setProperty(
            "workflowComplete",
            progress.rubbing.complete,
        )
        survey_style = self.btn_native_survey_export.style()
        survey_style.unpolish(self.btn_native_survey_export)
        survey_style.polish(self.btn_native_survey_export)
        self.btn_native_survey_export.update()

    def on_btn_toggled(self, checked):
        if checked:
            self.btn_toggle.setText("십자선 단면 모드 중지")
        else:
            self.btn_toggle.setText("십자선 단면 모드 시작")
        self.crosshairToggled.emit(checked)

    def on_line_toggled(self, checked):
        if checked:
            self.btn_line.setText("단면선 그리기 중지")
        else:
            self.btn_line.setText("단면선 그리기 시작")
        self.lineSectionToggled.emit(checked)
        
    def on_roi_toggled(self, checked):
        if checked:
            self.btn_roi.setText("영역 지정 모드 중지")
            self.btn_silhouette.setEnabled(True)
        else:
            self.btn_roi.setText("영역 지정 모드 시작")
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
        self._base_window_title = f"{APP_NAME} v{APP_VERSION} ({sha_s})"
        self.setWindowTitle(self._base_window_title)
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
        self._project_open_thread: TaskThread | None = None
        self._project_open_request_id: str | None = None
        self._project_open_base_authority_epoch: int | None = None
        self._profile_export_dialog: QProgressDialog | None = None
        self._profile_export_thread: ProfileExportThread | None = None
        self._task_dialog: QProgressDialog | None = None
        self._task_thread: TaskThread | None = None
        self._task_cancel_request: Callable[[], None] | None = None
        self._task_close_dialog: Callable[[], None] | None = None
        self._task_shutdown_verify: Callable[[], None] | None = None
        self._application_closing = False

        # 평면화(Flatten) 결과 캐시: (obj id + transform + options) -> FlattenedMesh
        self._flattened_cache = {}
        self._flatten_recommendation_cache: dict[int, tuple[tuple[Any, ...], dict[str, Any]]] = {}
        self._flatten_method_user_override = False
        self._flatten_method_signal_guard = False
        self._flatten_method_target_obj_id: int | None = None

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
        self._project_load_from_legacy: bool = False
        self._project_requires_save_as: bool = False
        self._legacy_project_path: str | None = None
        self._last_source_verification: SourceVerification | None = None
        self._project_has_legacy_bindings: bool = False
        self._project_pending_path: str | None = None
        self._project_load_failed: bool = False
        self._project_staged_objects: list[
            tuple[Any, str, dict[str, Any], SourceVerification, str]
        ] = []
        self._project_previous_context: dict[str, Any] | None = None
        # Native ArtifactDocument mode is mutually exclusive with writable
        # legacy_ui_state.  The session owns the immutable document, verified
        # source-space geometry and resolved external source path.
        self._artifact_session: ArtifactSession | None = None
        self._artifact_workbench = ArtifactWorkbench()
        self._artifact_measurements = ArtifactMeasurementController(
            self._artifact_workbench
        )
        self._artifact_exports = ArtifactExportController(
            self._artifact_workbench,
            rubbing_memory_budget_bytes=DEFAULT_RUBBING_MEMORY_BUDGET_BYTES,
        )
        self._artifact_survey_exports = ArtifactSurveyExportController(
            self._artifact_workbench,
            rubbing_memory_budget_bytes=DEFAULT_RUBBING_MEMORY_BUDGET_BYTES,
        )
        self._pending_native_measurement_publications: dict[
            str,
            tuple[ArtifactMeasurementWorkItem, ArtifactMeasurementResult],
        ] = {}
        # Fence late surface-pick workers from repopulating a transient point
        # set that the operator already cleared or replaced.
        self._surface_pick_generation = 0
        self._artifact_authority_faulted = False
        self._artifact_load_ticket: ArtifactLoadTicket | None = None
        self._mesh_load_request_id: str | None = None
        self._native_vector_preview_document_id: str | None = None
        self._native_rubbing_preview_record_id: str | None = None
        self._native_rubbing_preview_document_id: str | None = None
        self._native_rubbing_preview_geometry_ref: str | None = None
        self._native_rubbing_preview_pending_record_id: str | None = None
        self._native_rubbing_preview_pending_record: Any | None = None
        self._native_rubbing_preview_pending_token: object | None = None
        self._native_tile_unwrap_preview_record_id: str | None = None
        self._native_tile_unwrap_preview_document_id: str | None = None
        self._native_tile_unwrap_preview_geometry_ref: str | None = None
        self._native_tile_unwrap_preview_pending_record_id: str | None = None
        self._native_tile_unwrap_preview_pending_record: Any | None = None
        self._native_tile_unwrap_preview_pending_token: object | None = None
        self._artifact_load_active: bool = False
        self._artifact_pending_document: ArtifactDocument | None = None
        self._artifact_pending_project_path: str | None = None
        self._artifact_pending_source_metadata: dict[str, Any] | None = None
        
        self.init_ui()
        self.init_menu()
        self.init_toolbar()
        self.init_statusbar()
        self._workbench_unsubscribe = self._artifact_workbench.subscribe(
            self._on_workbench_snapshot_changed
        )
        self.destroyed.connect(
            lambda _object=None, unsubscribe=self._workbench_unsubscribe: (
                unsubscribe()
            )
        )
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
        self.viewport.meshTransformChanged.connect(
            self._refresh_native_save_indicator
        )
        self.viewport.floorPointPicked.connect(self.on_floor_point_picked)
        self.viewport.floorFacePicked.connect(self.on_floor_face_picked)
        self.viewport.alignToBrushSelected.connect(self.on_align_to_brush_selected)
        self.viewport.floorAlignmentConfirmed.connect(self.on_floor_alignment_confirmed)
        self.viewport.surfaceAssignmentChanged.connect(self.on_surface_assignment_changed)
        self.viewport.measurePointPicked.connect(self.on_measure_point_picked)
        self.viewport.curvaturePickStateChanged.connect(
            lambda _count: self._refresh_native_save_indicator()
        )
        self.viewport.surfaceAnchorPickRequested.connect(
            self.on_surface_anchor_pick_requested
        )
        self.viewport.undoRequested.connect(self.undo_last_action)
        
        # 단축키 설정 (Undo: Ctrl+Z)
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo_last_action)
        
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
        self.help_dock = QDockWidget("도움말", self)
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
            self.action_toggle_help_panel = QAction("도움말", self)
            self.action_toggle_help_panel.setCheckable(True)
            self.action_toggle_help_panel.toggled.connect(self._on_help_panel_toggled)
            try:
                self.help_dock.visibilityChanged.connect(self.action_toggle_help_panel.setChecked)
            except Exception:
                pass
        else:
            self.action_toggle_help_panel.setText("도움말")
            self.action_toggle_help_panel.setToolTip("도움말 창 표시/숨김")
            try:
                self.action_toggle_help_panel.toggled.connect(self._on_help_panel_toggled)
            except Exception:
                pass
        set_pixel_icon(self.action_toggle_help_panel, "help")

        # 도킹 위젯 설정
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AllowNestedDocks
        )
        self.setDockNestingEnabled(True)

        # 1) 상단 정보(파일/메쉬)
        self.info_dock = QDockWidget("파일/메쉬 정보", self)
        self.info_dock.setObjectName("dock_info")
        self.props_panel = InfoBarWidget()
        self.info_dock.setWidget(self.props_panel)

        # 1.5) 기본 작업 흐름
        self.workflow_dock = QDockWidget("4축 작업 흐름", self)
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
        self.flatten_panel.methodChanged.connect(self._on_flatten_method_changed)
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
        self.section_dock = QDockWidget("검증된 실측 · 전개", self)
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
        self.section_panel.silhouetteRequested.connect(self.on_silhouette_requested)
        self.section_panel.saveSectionLayersRequested.connect(self.on_save_section_layers_requested)
        self.section_panel.nativeCutlineRequested.connect(self.on_native_cutline_requested)
        self.section_panel.nativeCutlineViewChanged.connect(
            self.on_native_cutline_view_changed
        )
        self.section_panel.nativeOutlineRequested.connect(
            self.on_native_outline_requested
        )
        self.section_panel.nativeMeasurementRetryRequested.connect(
            self.on_native_measurement_retry_requested
        )
        self.section_panel.nativeVectorRecordSelected.connect(
            self.on_native_vector_record_selected
        )
        self.section_panel.nativeVectorExportRequested.connect(
            self.on_native_vector_export_requested
        )
        self.section_panel.nativeRubbingRequested.connect(
            self.on_native_rubbing_requested
        )
        self.section_panel.nativeRubbingRecordSelected.connect(
            self.on_native_rubbing_record_selected
        )
        self.section_panel.nativeRubbingExportRequested.connect(
            self.on_native_rubbing_export_requested
        )
        self.section_panel.nativeSurveyExportRequested.connect(
            self.on_native_survey_export_requested
        )
        self.section_panel.nativeTileUnwrapRequested.connect(
            self.on_native_tile_unwrap_requested
        )
        self.section_panel.nativeTileUnwrapRecordSelected.connect(
            self.on_native_tile_unwrap_record_selected
        )
        self.section_panel.nativeTileUnwrapExportRequested.connect(
            self.on_native_tile_unwrap_export_requested
        )

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
        self.scene_dock = QDockWidget("레이어", self)
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
        if self._native_artifact_mode():
            transient_state = self._native_transient_work_state()
            if transient_state.has_unpersisted_work:
                reply = self._ask_native_transient_action(
                    "프로그램을 종료",
                    transient_state,
                )
                if reply != QMessageBox.StandardButton.Discard:
                    a0.ignore()
                    return
            elif self._native_document_has_unsaved_changes():
                reply = self._ask_native_unsaved_action("프로그램을 종료")
                if reply == QMessageBox.StandardButton.Save:
                    # A native save finishes on a worker. Keep this close event
                    # rejected and close only after the exact captured document
                    # owns a Windows write-through-confirmed save checkpoint.
                    a0.ignore()
                    self.save_project(on_saved=self.close)
                    return
                if reply != QMessageBox.StandardButton.Discard:
                    a0.ignore()
                    return
        else:
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
        self._application_closing = True
        if not self._shutdown_active_task_worker():
            self._application_closing = False
            a0.ignore()
            try:
                self.status_info.setText(
                    "종료 보류 | 실행 중 작업의 안전한 종료를 기다리는 중"
                )
            except Exception:
                pass
            QMessageBox.warning(
                self,
                "종료 보류",
                "실행 중 작업이 아직 안전하게 끝나지 않았습니다.\n"
                "작업 종료가 표시된 뒤 다시 종료하세요.",
            )
            return
        if not self._shutdown_mesh_load_worker():
            self._application_closing = False
            a0.ignore()
            try:
                self.status_info.setText(
                    "종료 보류 | 원본 로더의 안전한 종료를 기다리는 중"
                )
            except Exception:
                pass
            QMessageBox.warning(
                self,
                "종료 보류",
                "원본 로더가 아직 안전하게 끝나지 않았습니다. 현재 문서 권위는 "
                "유지했고 늦은 로드 결과는 폐기합니다. 로더 종료 후 다시 종료하세요.",
            )
            return
        if not self._shutdown_project_open_worker():
            self._application_closing = False
            a0.ignore()
            try:
                self.status_info.setText(
                    "종료 보류 | 프로젝트 검증기의 안전한 종료를 기다리는 중"
                )
            except Exception:
                pass
            QMessageBox.warning(
                self,
                "종료 보류",
                "프로젝트 검증기가 아직 안전하게 끝나지 않았습니다. 현재 문서 "
                "권위는 유지했고 늦은 검증 결과는 폐기합니다. 종료 후 다시 시도하세요.",
            )
            return
        self._save_ui_state()
        unsubscribe = getattr(self, "_workbench_unsubscribe", None)
        if callable(unsubscribe):
            unsubscribe()
            self._workbench_unsubscribe = None
        super().closeEvent(a0)

    @staticmethod
    def _wait_for_thread_shutdown(thread: object, timeout_ms: int) -> bool:
        """Wait for one worker without ever force-terminating its QThread."""

        try:
            if not bool(thread.isRunning()):  # type: ignore[attr-defined]
                return True
        except Exception:
            return False
        try:
            joined = thread.wait(int(timeout_ms))  # type: ignore[attr-defined]
        except Exception:
            _LOGGER.exception("Worker shutdown wait failed")
            return False
        return bool(joined)

    def _shutdown_active_task_worker(self) -> bool:
        """Revoke authority, cancel when supported, and join before teardown.

        The task's completion callbacks stay connected while waiting so a timed-out
        close attempt can safely return to the live window.  Once the worker has
        joined, callbacks are disconnected and the task identity is cleared before
        any queued signal can publish a measurement or export into a closing app.
        """

        thread = getattr(self, "_task_thread", None)
        if thread is None:
            return True

        cancel_request = getattr(self, "_task_cancel_request", None)
        if callable(cancel_request):
            try:
                cancel_request()
            except Exception:
                _LOGGER.exception("Active task cancellation request failed")
                return False
        try:
            thread.requestInterruption()
        except Exception:
            pass

        if not self._wait_for_thread_shutdown(thread, TASK_SHUTDOWN_WAIT_MS):
            return False

        shutdown_verify = getattr(self, "_task_shutdown_verify", None)
        if callable(shutdown_verify):
            try:
                shutdown_verify()
            except Exception:
                _LOGGER.exception(
                    "Joined task did not reach a shutdown-safe terminal state"
                )
                return False

        for signal_name in ("done", "failed", "finished"):
            try:
                getattr(thread, signal_name).disconnect()
            except Exception:
                pass

        if getattr(self, "_task_thread", None) is thread:
            self._task_thread = None
            self._task_cancel_request = None
            self._task_shutdown_verify = None
            close_dialog = getattr(self, "_task_close_dialog", None)
            self._task_close_dialog = None
            if callable(close_dialog):
                try:
                    close_dialog()
                except Exception:
                    _LOGGER.debug(
                        "Task dialog shutdown cleanup failed",
                        exc_info=True,
                    )
        try:
            thread.deleteLater()
        except Exception:
            pass
        return True

    def _shutdown_mesh_load_worker(self) -> bool:
        """Fence and join the independent source loader before window teardown.

        ``MeshLoadThread`` is not part of the shared ``_task_thread`` slot.  Revoke
        its request ID and Open ticket before waiting so a queued ``loaded`` signal
        can never publish into the live scene after a close attempt.  The worker is
        cooperative and may be inside a parser that cannot stop immediately; in
        that case retain the QThread object and leave the window alive.
        """

        thread = getattr(self, "_mesh_load_thread", None)
        if thread is None:
            return True

        # Fence queued and future callbacks before requesting interruption.
        self._mesh_load_request_id = None
        for signal_name in ("loaded", "failed", "finished"):
            try:
                getattr(thread, signal_name).disconnect()
            except Exception:
                pass

        try:
            thread.requestInterruption()
        except Exception:
            pass

        # Roll pending Open authority back to the exact current session.  Legacy
        # project staging is CPU-only here, so discard it without touching scene.
        if bool(getattr(self, "_artifact_load_active", False)):
            self._clear_artifact_pending_load(cancel_workbench=True)
        if bool(getattr(self, "_project_load_active", False)):
            self._project_load_active = False
            self._project_load_queue = []
            self._project_load_current = None
            self._project_load_state = None
            self._project_load_from_legacy = False
            self._discard_project_staging_and_restore_context()

        dialog = getattr(self, "_mesh_load_dialog", None)
        self._mesh_load_dialog = None
        if dialog is not None:
            try:
                dialog.close()
            except Exception:
                pass
        try:
            self._status_task_end()
        except Exception:
            pass

        if not self._wait_for_thread_shutdown(thread, TASK_SHUTDOWN_WAIT_MS):
            # Keep the only explicit owner while the QThread is still running.
            # A later close attempt can join and release it safely.
            return False

        if getattr(self, "_mesh_load_thread", None) is thread:
            self._mesh_load_thread = None
        try:
            thread.deleteLater()
        except Exception:
            pass
        return True

    def _shutdown_project_open_worker(self) -> bool:
        """Fence and bounded-join package inspection before window teardown."""

        thread = getattr(self, "_project_open_thread", None)
        if thread is None:
            return True
        self._project_open_request_id = None
        self._project_open_base_authority_epoch = None
        self._clear_artifact_pending_load(cancel_workbench=True)
        for signal_name in ("done", "failed", "finished"):
            try:
                getattr(thread, signal_name).disconnect()
            except Exception:
                pass
        try:
            thread.requestInterruption()
        except Exception:
            pass
        if not self._wait_for_thread_shutdown(thread, TASK_SHUTDOWN_WAIT_MS):
            # Retain the QThread owner while the package parser finishes.  Its
            # request ID is already revoked, so no late result can be published.
            return False
        if getattr(self, "_project_open_thread", None) is thread:
            self._project_open_thread = None
        try:
            thread.deleteLater()
        except Exception:
            pass
        return True

    def _native_artifact_mode(self) -> bool:
        return isinstance(getattr(self, "_artifact_session", None), ArtifactSession)

    def _artifact_workbench_controller(self) -> ArtifactWorkbench:
        """Return the Qt-free authority controller during the shell migration."""

        controller = getattr(self, "_artifact_workbench", None)
        if not isinstance(controller, ArtifactWorkbench):
            controller = ArtifactWorkbench()
            self._artifact_workbench = controller
        session = getattr(self, "_artifact_session", None)
        session = session if isinstance(session, ArtifactSession) else None
        snapshot = controller.snapshot
        # Older GUI tests and not-yet-ported record commands still assign the
        # compatibility field directly.  Production Open/Align never needs
        # this bridge; it exists so one vertical slice can migrate safely.
        if (
            not bool(getattr(self, "_artifact_authority_faulted", False))
            and snapshot.pending_load is None
            and snapshot.session is not session
        ):
            controller.synchronize_legacy_session(
                session,
                project_path=getattr(self, "_current_project_path", None),
            )
        return controller

    def _artifact_measurement_controller(self) -> ArtifactMeasurementController:
        """Return the Qt-free derived-operation controller for this workbench."""

        workbench = self._artifact_workbench_controller()
        controller = getattr(self, "_artifact_measurements", None)
        if (
            not isinstance(controller, ArtifactMeasurementController)
            or controller.workbench is not workbench
        ):
            controller = ArtifactMeasurementController(workbench)
            self._artifact_measurements = controller
        return controller

    def _artifact_export_controller(self) -> ArtifactExportController:
        """Return the Qt-free stage/final-publish export controller."""

        workbench = self._artifact_workbench_controller()
        controller = getattr(self, "_artifact_exports", None)
        if (
            not isinstance(controller, ArtifactExportController)
            or controller.workbench is not workbench
        ):
            controller = ArtifactExportController(
                workbench,
                rubbing_memory_budget_bytes=DEFAULT_RUBBING_MEMORY_BUDGET_BYTES,
            )
            self._artifact_exports = controller
        return controller

    def _artifact_survey_export_controller(self) -> ArtifactSurveyExportController:
        """Return the Qt-free complete-survey atomic export controller."""

        workbench = self._artifact_workbench_controller()
        controller = getattr(self, "_artifact_survey_exports", None)
        if (
            not isinstance(controller, ArtifactSurveyExportController)
            or controller.workbench is not workbench
        ):
            controller = ArtifactSurveyExportController(
                workbench,
                rubbing_memory_budget_bytes=DEFAULT_RUBBING_MEMORY_BUDGET_BYTES,
            )
            self._artifact_survey_exports = controller
        return controller

    def _native_workflow_stage(self) -> WorkflowStage:
        if not self._native_artifact_mode():
            return WorkflowStage.EMPTY
        return self._artifact_workbench_controller().snapshot.stage

    def _native_measurement_ready(self) -> bool:
        if not self._native_artifact_mode() or bool(
            getattr(self, "_artifact_authority_faulted", False)
        ):
            return False
        return self._artifact_workbench_controller().snapshot.can_measure

    def _native_record_workflow_progress(self) -> ArtifactWorkflowProgress:
        session = getattr(self, "_artifact_session", None)
        if not isinstance(session, ArtifactSession):
            return ArtifactWorkflowProgress.empty()
        return derive_artifact_workflow_progress(
            session,
            align_ready=self._native_measurement_ready(),
        )

    def _require_native_projection_session(self, obj: Any) -> ArtifactSession:
        if bool(getattr(self, "_artifact_authority_faulted", False)):
            raise ArtifactSessionError(
                "artifact authority is faulted; reopen a verified source or project"
            )
        session = getattr(self, "_artifact_session", None)
        if not isinstance(session, ArtifactSession):
            raise ArtifactSessionError("no active ArtifactDocument session")
        objects = list(getattr(self.viewport, "objects", []) or [])
        if len(objects) != 1 or objects[0] is not obj:
            raise ArtifactSessionError(
                "native ArtifactDocument must own exactly one projected object"
            )
        binding = getattr(obj, "_amr_artifact_projection_snapshot", None)
        if not isinstance(binding, ArtifactProjectionSnapshot):
            raise ArtifactSessionError("native projection has no document binding")
        if binding != session.projection_snapshot():
            raise ArtifactSessionError("native projection binding is stale")
        return session

    def _require_native_measurement_session(self, obj: Any) -> ArtifactSession:
        session = self._require_native_projection_session(obj)
        controller = self._artifact_workbench_controller()
        try:
            controller.require_stable_session(session, measurement=True)
        except ArtifactWorkbenchError as exc:
            raise ArtifactSessionError(str(exc)) from exc
        return session

    def _reject_native_unported_mutation(self, action_name: str) -> bool:
        """Fail closed when a legacy tool would mutate a native projection."""

        if not self._native_artifact_mode():
            return False
        message = (
            f"{action_name}은 아직 ArtifactDocument revision 명령으로 전환되지 않았습니다. "
            "원본과 재현성을 보호하기 위해 현재 문서에서는 실행하지 않습니다. "
            "이동·회전 preview와 '정치 확정'을 사용하세요."
        )
        try:
            self.viewport.status_info = message
            self.status_info.setText(f"{action_name} 차단 | 원본 projection 유지")
            self.viewport.update()
        except Exception:
            pass
        QMessageBox.warning(self, "지원 전 정렬 도구", message)
        return True

    def start_floor_picking(self):
        """바닥면 그리기(점 찍기) 모드 시작"""
        if self._reject_native_unported_mutation("3점 바닥 정렬"):
            return
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
        if self._reject_native_unported_mutation("면 바닥 정렬"):
            return
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
        if self._reject_native_unported_mutation("브러시 바닥 정렬"):
            return
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
        self.viewport.status_info = "바닥이 될 영역을 마우스 왼쪽 버튼으로 드래그하듯이 그리세요..."
        self.viewport.update()

    def on_align_to_brush_selected(self):
        """Align by brushed-face normal and keep brushed region touching XY plane."""
        if self._reject_native_unported_mutation("브러시 바닥 정렬"):
            return
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
                f"브러시 바닥 정렬 완료 ({count}개 면 / 선택점 Z잔차 ±{z_residual:.4f})"
            )
        else:
            self.viewport.status_info = f"브러시 바닥 정렬 완료 ({count}개 면)"
        self.viewport.update()
        self.viewport.meshTransformChanged.emit()

    def align_mesh_to_normal(self, normal, *, pivot=None) -> np.ndarray | None:
        """주어진 법선을 월드 +Z로 정렬 (메쉬에 직접 반영/Bake)."""
        if self._reject_native_unported_mutation("법선 바닥 정렬"):
            return None
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
        obj._amr_has_unpersisted_bake = True
        obj._amr_alignment_status = _ALIGNMENT_STATUS_BAKED_UNVERIFIABLE
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
        self.viewport.status_info = "면 선택됨. Enter를 누르면 정렬됩니다."
        self.viewport.update()
        self._refresh_native_save_indicator()

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
            self.viewport.status_info = f"바닥면 점 찍기 (현재 {count}개 선택됨, 더 찍어주세요)..."
        else:
            self.viewport.status_info = f"점 {count}개 선택됨. 계속 추가하거나 Enter로 확정하세요."
        
        self.viewport.update()
        self._refresh_native_save_indicator()

    def on_floor_alignment_confirmed(self):
        """Enter 키 입력 시 호출: 선택된 점들을 기반으로 평면 정렬 수행"""
        if self._reject_native_unported_mutation("바닥 정렬 확정"):
            return
        obj = self.viewport.selected_obj
        if not obj or not self.viewport.floor_picks:
            return

        points = np.asarray(self.viewport.floor_picks, dtype=np.float64).reshape(-1, 3)
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) < 3:
            self.viewport.status_info = "점이 부족합니다. 더 찍어주세요."
            self.viewport.update()
            return
            
        # 1) floor_picks는 월드 좌표이므로 메쉬도 월드 기준 정점으로 맞춘다.
        self.viewport.bake_object_transform(obj)

        # 2) 선택한 점 전체를 반영한 least-squares 평면을 추정한다.
        plane = fit_plane_normal(points, robust=False)
        if plane is None:
            self.viewport.status_info = "선택 점이 거의 일직선입니다. 점을 다시 찍어주세요."
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
                f"바닥 정렬 완료 (점 {len(points)}개 / 선택점 Z잔차 ±{z_residual:.4f})"
            )
        else:
            self.viewport.status_info = f"바닥 정렬 완료 (점 {len(points)}개 기반)"
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
                self.status_info.setText(f"원호 #{arc_idx+1} 삭제됨")
                self._refresh_native_save_indicator()
    
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
        
        action_open = QAction("열기(&O)", self)
        set_pixel_icon(action_open, "open_mesh")
        action_open.setShortcut(QKeySequence.StandardKey.Open)
        action_open.triggered.connect(self.open_file)
        file_menu.addAction(action_open)

        action_open_project = QAction("프로젝트 열기…", self)
        set_pixel_icon(action_open_project, "open_project")
        action_open_project.setShortcut(QKeySequence("Ctrl+Shift+O"))
        action_open_project.triggered.connect(self.open_project)
        file_menu.addAction(action_open_project)

        action_recover_project = QAction("중단된 프로젝트 저장 복구…", self)
        set_pixel_icon(action_recover_project, "recover")
        action_recover_project.triggered.connect(
            self.recover_interrupted_project_save
        )
        file_menu.addAction(action_recover_project)

        file_menu.addSeparator()

        action_save_project = QAction("프로젝트 저장", self)
        set_pixel_icon(action_save_project, "save")
        action_save_project.setShortcut(QKeySequence.StandardKey.Save)
        action_save_project.triggered.connect(self.save_project)
        file_menu.addAction(action_save_project)

        action_save_project_as = QAction("프로젝트 다른 이름 저장…", self)
        set_pixel_icon(action_save_project_as, "save")
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
        
        action_reset_view = QAction("뷰 초기화(&R)", self)
        set_pixel_icon(action_reset_view, "reset")
        action_reset_view.setShortcut("R")
        action_reset_view.triggered.connect(self.reset_view)
        view_menu.addAction(action_reset_view)
        
        action_fit = QAction("메쉬에 맞춤(&F)", self)
        set_pixel_icon(action_fit, "fit")
        action_fit.setShortcut("F")
        action_fit.triggered.connect(self.fit_view)
        view_menu.addAction(action_fit)
        
        view_menu.addSeparator()
        
        # 6방향 뷰
        action_front = QAction("1 정면 뷰", self)
        set_pixel_icon(action_front, "view_front")
        action_front.setShortcut("1")
        action_front.triggered.connect(lambda: self._set_canonical_view("front"))
        view_menu.addAction(action_front)
        
        action_back = QAction("2 후면 뷰", self)
        set_pixel_icon(action_back, "view_back")
        action_back.setShortcut("2")
        action_back.triggered.connect(lambda: self._set_canonical_view("back"))
        view_menu.addAction(action_back)
        
        action_right = QAction("3 우측면 뷰", self)
        set_pixel_icon(action_right, "view_right")
        action_right.setShortcut("3")
        action_right.triggered.connect(lambda: self._set_canonical_view("right"))
        view_menu.addAction(action_right)
        
        action_left = QAction("4 좌측면 뷰", self)
        set_pixel_icon(action_left, "view_left")
        action_left.setShortcut("4")
        action_left.triggered.connect(lambda: self._set_canonical_view("left"))
        view_menu.addAction(action_left)
        
        action_top = QAction("5 상면 뷰", self)
        set_pixel_icon(action_top, "view_top")
        action_top.setShortcut("5")
        action_top.triggered.connect(lambda: self._set_canonical_view("top"))
        view_menu.addAction(action_top)
        
        action_bottom = QAction("6 하면 뷰", self)
        set_pixel_icon(action_bottom, "view_bottom")
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
            action_about = QAction("정보(&A)", self)
            set_pixel_icon(action_about, "help")
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
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        action_open = QAction("열기", self)
        set_pixel_icon(action_open, "open_mesh")
        action_open.triggered.connect(self.open_file)
        toolbar.addAction(action_open)

        action_open_project = QAction("프로젝트", self)
        set_pixel_icon(action_open_project, "open_project")
        action_open_project.triggered.connect(self.open_project)
        toolbar.addAction(action_open_project)

        toolbar.addSeparator()

        action_fit = QAction("뷰 맞춤", self)
        set_pixel_icon(action_fit, "fit")
        action_fit.setToolTip("메쉬가 화면에 꽉 차도록 카메라 조정")
        action_fit.triggered.connect(self.fit_view)
        toolbar.addAction(action_fit)

        toolbar.addSeparator()
        
        # 6방향 뷰 버튼
        action_front = QAction("정면", self)
        set_pixel_icon(action_front, "view_front")
        action_front.setToolTip("정면 뷰 (1)")
        action_front.triggered.connect(lambda: self._set_canonical_view("front"))
        toolbar.addAction(action_front)
        
        action_back = QAction("후면", self)
        set_pixel_icon(action_back, "view_back")
        action_back.setToolTip("후면 뷰 (2)")
        action_back.triggered.connect(lambda: self._set_canonical_view("back"))
        toolbar.addAction(action_back)
        
        action_right = QAction("우측", self)
        set_pixel_icon(action_right, "view_right")
        action_right.setToolTip("우측면 뷰 (3)")
        action_right.triggered.connect(lambda: self._set_canonical_view("right"))
        toolbar.addAction(action_right)
        
        action_left = QAction("좌측", self)
        set_pixel_icon(action_left, "view_left")
        action_left.setToolTip("좌측면 뷰 (4)")
        action_left.triggered.connect(lambda: self._set_canonical_view("left"))
        toolbar.addAction(action_left)
        
        action_top = QAction("상면", self)
        set_pixel_icon(action_top, "view_top")
        action_top.setToolTip("상면 뷰 (5)")
        action_top.triggered.connect(lambda: self._set_canonical_view("top"))
        toolbar.addAction(action_top)
        
        action_bottom = QAction("하면", self)
        set_pixel_icon(action_bottom, "view_bottom")
        action_bottom.setToolTip("하면 뷰 (6)")
        action_bottom.triggered.connect(lambda: self._set_canonical_view("bottom"))
        toolbar.addAction(action_bottom)

        toolbar.addSeparator()

        action_record_top = QAction("상면 기록", self)
        set_pixel_icon(action_record_top, "record_top")
        action_record_top.triggered.connect(
            lambda: self.on_tile_interpretation_action("prepare_record_surface", {"view": "top"})
        )
        toolbar.addAction(action_record_top)

        action_record_bottom = QAction("하면 기록", self)
        set_pixel_icon(action_record_bottom, "record_bottom")
        action_record_bottom.triggered.connect(
            lambda: self.on_tile_interpretation_action("prepare_record_surface", {"view": "bottom"})
        )
        toolbar.addAction(action_record_bottom)

        action_preview = QAction("미리보기", self)
        set_pixel_icon(action_preview, "preview")
        action_preview.triggered.connect(self.on_flatten_preview_requested)
        toolbar.addAction(action_preview)

        action_review = QAction("검토 시트", self)
        set_pixel_icon(action_review, "export")
        action_review.triggered.connect(
            lambda: self.on_export_requested({"type": "review_sheet", "target": "selected"})
        )
        toolbar.addAction(action_review)

    def init_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        self.status_info = QLabel("파일을 열거나 드래그하세요")
        self.status_mesh = QLabel("") # 메쉬 정보 (정점, 면)
        self.status_grid = QLabel("격자: -")
        self.status_unit = QLabel("단위: -")
        self.status_save = QLabel("저장: -")
        
        self.statusbar.addWidget(self.status_info, 1)
        self.statusbar.addPermanentWidget(self.status_mesh)
        self.statusbar.addPermanentWidget(self.status_grid)
        self.statusbar.addPermanentWidget(self.status_unit)
        self.statusbar.addPermanentWidget(self.status_save)
        
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

    @staticmethod
    def _contains_native_transient_value(value: Any) -> bool:
        """Return whether a GUI value contains a material, unrecorded result."""

        if value is None:
            return False
        if isinstance(value, np.ndarray):
            return bool(value.size)
        if isinstance(value, Mapping):
            return any(
                MainWindow._contains_native_transient_value(item)
                for item in value.values()
            )
        if isinstance(value, (list, tuple, set, frozenset)):
            if not value:
                return False
            if all(
                isinstance(
                    item,
                    (Mapping, list, tuple, set, frozenset, np.ndarray),
                )
                or item is None
                for item in value
            ):
                return any(
                    MainWindow._contains_native_transient_value(item)
                    for item in value
                )
            return True
        if isinstance(value, str):
            return bool(value.strip())
        return bool(value)

    def _native_transient_work_state(
        self,
        *,
        obj: Any | None = None,
        allowed_selected_face_indices: tuple[int, ...] | None = None,
        allow_surface_measurement_picks: bool = False,
    ) -> _NativeTransientWorkState:
        """Capture every cheap GUI-only guard shared by Save and navigation.

        The immutable document hash cannot see these values.  Detection is
        deliberately fail-closed: malformed transform/selection/operation
        state becomes an issue instead of being treated as clean.
        """

        issues: dict[str, _NativeTransientIssue] = {}

        def add(code: str, message: str) -> None:
            issues.setdefault(
                str(code),
                _NativeTransientIssue(code=str(code), message=str(message)),
            )

        controller_factory = getattr(
            self,
            "_artifact_measurement_controller",
            None,
        )
        if callable(controller_factory):
            try:
                active_measurements = tuple(
                    controller_factory().active_summaries or ()
                )
            except Exception:
                _LOGGER.exception(
                    "Could not inspect active native measurement operations"
                )
                add(
                    "measurement_state_unreadable",
                    "실측 작업 상태를 확인할 수 없어 안전하게 중단해야 합니다.",
                )
            else:
                if active_measurements:
                    states = ", ".join(
                        f"{getattr(getattr(summary, 'kind', None), 'value', 'unknown')}:"
                        f"{getattr(getattr(summary, 'state', None), 'value', 'unknown')}"
                        for summary in active_measurements
                    )
                    add(
                        "active_measurement",
                        "계산 또는 게시가 끝나지 않은 실측 작업이 있습니다 "
                        f"({states}).",
                    )

        try:
            pending_publications = getattr(
                self,
                "_pending_native_measurement_publications",
                {},
            )
            if pending_publications:
                add(
                    "active_measurement",
                    "계산은 끝났지만 ArtifactDocument에 게시되지 않은 실측 "
                    "결과가 있습니다.",
                )
        except Exception:
            add(
                "measurement_state_unreadable",
                "보류 실측 결과 상태를 확인할 수 없어 안전하게 중단해야 합니다.",
            )

        active_task = getattr(self, "_task_thread", None)
        if active_task is not None:
            try:
                task_name = str(getattr(active_task, "_task_name", ""))
            except Exception:
                task_name = ""
            # Derived measurement commands own controller capabilities and are
            # covered above.  Surface-anchor resolution is the one pre-command
            # measurement worker whose result would otherwise be invisible.
            # Generic Save/export/Align workers are not scene dirtiness.
            if task_name == "native_surface_anchor":
                add(
                    "pending_surface_anchor",
                    "표면 anchor 선택을 확인하는 실측 작업이 끝나지 않았습니다.",
                )

        viewport = getattr(self, "viewport", None)
        if viewport is None:
            add(
                "scene_state_unreadable",
                "3D 장면 상태를 확인할 수 없어 안전하게 중단해야 합니다.",
            )
            return _NativeTransientWorkState(tuple(issues.values()))

        if obj is None:
            try:
                objects = list(getattr(viewport, "objects", []) or [])
            except Exception:
                objects = []
                add(
                    "scene_state_unreadable",
                    "3D 장면 객체 상태를 확인할 수 없어 안전하게 중단해야 합니다.",
                )
            if len(objects) == 1:
                obj = objects[0]
            else:
                add(
                    "scene_object_count",
                    "Native 유물 문서는 검증된 장면 객체 하나만 "
                    f"소유해야 하지만 현재 {len(objects)}개입니다.",
                )
        if obj is None:
            return _NativeTransientWorkState(tuple(issues.values()))

        def identity_triplet(value: Any) -> bool:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            return bool(
                array.shape == (3,)
                and np.isfinite(array).all()
                and np.allclose(array, [0.0, 0.0, 0.0], rtol=0.0, atol=1e-12)
            )

        try:
            if not identity_triplet(getattr(obj, "translation")):
                add(
                    "align_translation_preview",
                    "현재 보이는 이동 preview를 먼저 정치 확정하거나 초기화하세요.",
                )
        except Exception:
            add(
                "align_translation_unreadable",
                "이동 preview 상태를 확인할 수 없어 안전하게 중단해야 합니다.",
            )
        try:
            if not identity_triplet(getattr(obj, "rotation")):
                add(
                    "align_rotation_preview",
                    "현재 보이는 회전 preview를 먼저 정치 확정하거나 초기화하세요.",
                )
        except Exception:
            add(
                "align_rotation_unreadable",
                "회전 preview 상태를 확인할 수 없어 안전하게 중단해야 합니다.",
            )
        try:
            scale = float(getattr(obj, "scale"))
            if not np.isfinite(scale) or not np.isclose(
                scale,
                1.0,
                rtol=0.0,
                atol=1e-12,
            ):
                add(
                    "align_scale_preview",
                    "Native Align scale preview는 저장할 수 없습니다.",
                )
        except Exception:
            add(
                "align_scale_unreadable",
                "배율 preview 상태를 확인할 수 없어 안전하게 중단해야 합니다.",
            )
        try:
            if bool(getattr(obj, "_amr_has_unpersisted_bake", False)):
                add(
                    "unpersisted_vertex_bake",
                    "Native projection에 문서화되지 않은 vertex bake 흔적이 있습니다.",
                )
        except Exception:
            add(
                "unpersisted_vertex_bake_unreadable",
                "vertex bake 상태를 확인할 수 없어 안전하게 중단해야 합니다.",
            )

        raw_tile_state = getattr(obj, "tile_interpretation_state", None)
        if raw_tile_state is not None:
            try:
                normalized_tile_state = (
                    raw_tile_state
                    if isinstance(raw_tile_state, TileInterpretationState)
                    else TileInterpretationState.from_dict(raw_tile_state)
                )
                if (
                    normalized_tile_state.to_dict()
                    != TileInterpretationState().to_dict()
                ):
                    add(
                        "tile_interpretation",
                        "기와 판독 상태가 아직 ArtifactDocument record로 승격되지 "
                        "않았습니다.",
                    )
            except Exception:
                add(
                    "tile_interpretation_unreadable",
                    "기와 판독 상태를 확인할 수 없어 안전하게 중단해야 합니다.",
                )

        try:
            selected_faces = tuple(
                sorted(
                    int(value)
                    for value in (getattr(obj, "selected_faces", set()) or set())
                )
            )
        except Exception:
            selected_faces = ()
            add(
                "selected_faces_unreadable",
                "선택 face 상태를 확인할 수 없어 안전하게 중단해야 합니다.",
            )
        if allowed_selected_face_indices is not None:
            try:
                allowed_selection = tuple(
                    sorted(int(value) for value in allowed_selected_face_indices)
                )
            except Exception:
                allowed_selection = ()
                add(
                    "allowed_selection_unreadable",
                    "캡처한 face selection 상태를 확인할 수 없습니다.",
                )
            if selected_faces != allowed_selection:
                add(
                    "selected_faces_changed",
                    "기와 전개 face selection이 preflight capture와 다릅니다.",
                )
        elif selected_faces:
            add(
                "selected_faces",
                "선택 face가 아직 ArtifactDocument record로 승격되지 않았습니다.",
            )

        object_values = (
            ("polyline_layers", "폴리라인 레이어"),
            ("fitted_arcs", "맞춤 원호"),
            ("outer_face_indices", "외면 face 지정"),
            ("inner_face_indices", "내면 face 지정"),
            ("migu_face_indices", "미구 face 지정"),
            (
                "surface_assist_unresolved_face_indices",
                "미확정 표면 보조 face",
            ),
            ("surface_assist_meta", "표면 보조 메타데이터"),
            ("surface_assist_runtime", "표면 보조 실행 결과"),
            ("tile_synthetic_truth", "기와 합성 기준값"),
            ("tile_evaluation_report", "기와 평가 결과"),
        )
        viewport_values = (
            ("picked_points", "곡률 선택점"),
            ("fitted_arc", "곡률 맞춤 원호"),
            ("slice_contours", "단면 contour"),
            ("x_profile", "X 단면 profile"),
            ("y_profile", "Y 단면 profile"),
            ("_world_x_profile", "월드 X 단면 profile"),
            ("_world_y_profile", "월드 Y 단면 profile"),
            ("roi_cut_edges", "ROI 절단 edge"),
            ("roi_cap_verts", "ROI cap vertex"),
            ("roi_section_world", "ROI 단면"),
            ("cut_lines", "단면선"),
            ("cut_line_preview", "단면선 preview"),
            ("cut_section_profiles", "단면 profile"),
            ("cut_section_world", "월드 단면"),
            ("cut_section_contours_world", "월드 단면 contour"),
            ("cut_section_contours_local", "로컬 단면 contour"),
            ("line_profile", "선형 단면 profile"),
            ("line_section_contours", "선형 단면 contour"),
            ("floor_picks", "바닥면 선택점"),
            ("brush_selected_faces", "브러시 선택 face"),
            ("surface_paint_points", "표면 지정점"),
            ("surface_lasso_points", "표면 lasso 점"),
            ("surface_lasso_face_indices", "표면 lasso face"),
            ("surface_magnetic_points", "표면 자석점"),
        )
        for field_name, label in object_values:
            try:
                value = getattr(obj, field_name, None)
                if MainWindow._contains_native_transient_value(value):
                    add(
                        f"object:{field_name}",
                        f"{label}이 아직 ArtifactDocument record로 승격되지 않았습니다.",
                    )
            except Exception:
                add(
                    f"object:{field_name}:unreadable",
                    f"{label} 상태를 확인할 수 없어 안전하게 중단해야 합니다.",
                )
        for field_name, label in viewport_values:
            try:
                value = getattr(viewport, field_name, None)
                if MainWindow._contains_native_transient_value(value):
                    add(
                        f"viewport:{field_name}",
                        f"{label}이 아직 ArtifactDocument record로 승격되지 않았습니다.",
                    )
            except Exception:
                add(
                    f"viewport:{field_name}:unreadable",
                    f"{label} 상태를 확인할 수 없어 안전하게 중단해야 합니다.",
                )
        if not allow_surface_measurement_picks:
            for field_name, label in (
                ("measure_picked_points", "실측 선택점"),
                ("measure_picked_anchors", "실측 surface anchor"),
            ):
                try:
                    value = getattr(viewport, field_name, None)
                    if MainWindow._contains_native_transient_value(value):
                        add(
                            f"viewport:{field_name}",
                            f"{label}가 아직 ArtifactDocument record로 승격되지 않았습니다.",
                        )
                except Exception:
                    add(
                        f"viewport:{field_name}:unreadable",
                        f"{label} 상태를 확인할 수 없어 안전하게 중단해야 합니다.",
                    )

        return _NativeTransientWorkState(tuple(issues.values()))

    def _refresh_native_save_indicator(self) -> None:
        """Refresh shell dirtiness after a GUI-only native mutation."""

        if not self._native_artifact_mode():
            return
        try:
            snapshot = self._artifact_workbench_controller().snapshot
            self._on_workbench_snapshot_changed(snapshot)
        except Exception:
            _LOGGER.exception("Could not refresh native transient save status")
            base_title = str(getattr(self, "_base_window_title", APP_NAME))
            self.setWindowTitle(f"* {base_title}")
            status_label = getattr(self, "status_save", None)
            if isinstance(status_label, QLabel):
                status_label.setText("저장: 상태 확인 필요")

    def _on_workbench_snapshot_changed(self, snapshot: WorkflowSnapshot) -> None:
        """Reflect document checkpoints and GUI-only work in the Windows shell."""

        if not isinstance(snapshot, WorkflowSnapshot):
            return
        base_title = str(
            getattr(
                self,
                "_base_window_title",
                f"{APP_NAME} v{APP_VERSION}",
            )
        )
        status_label = getattr(self, "status_save", None)
        if snapshot.save_status is WorkflowSaveStatus.EMPTY:
            self.setWindowTitle(base_title)
            if isinstance(status_label, QLabel):
                status_label.setText("저장: -")
                status_label.setToolTip("")
            return

        transient_state = _NativeTransientWorkState()
        if snapshot.session is not None and self._native_artifact_mode():
            transient_state = self._native_transient_work_state()

        display_path = snapshot.project_path
        if not display_path and snapshot.session is not None:
            display_path = snapshot.session.resolved_source_path
        display_name = Path(display_path).name if display_path else "새 유물"
        dirty_prefix = (
            "* "
            if snapshot.has_unsaved_changes
            or transient_state.has_unpersisted_work
            else ""
        )
        self.setWindowTitle(f"{dirty_prefix}{display_name} — {base_title}")

        if not isinstance(status_label, QLabel):
            return
        if transient_state.has_unpersisted_work:
            status_label.setText("저장: 미확정 작업")
        elif snapshot.save_status is WorkflowSaveStatus.SAVED:
            status_label.setText("저장: 저장됨")
        elif snapshot.save_status is WorkflowSaveStatus.DURABILITY_UNCERTAIN:
            status_label.setText("저장: 내구성 미확정")
        else:
            status_label.setText("저장: 미저장")
        document_sha256 = snapshot.document_sha256 or ""
        tooltip_parts = (
            [f"문서 SHA-256: {document_sha256}"] if document_sha256 else []
        )
        if transient_state.has_unpersisted_work:
            tooltip_parts.extend(
                [
                    "프로젝트 저장에 포함되지 않는 미확정 작업:",
                    transient_state.detail(),
                ]
            )
        status_label.setToolTip("\n".join(tooltip_parts))

    def _native_document_has_unsaved_changes(self) -> bool:
        """Fail closed when native document save authority cannot be read."""

        if not self._native_artifact_mode():
            return False
        try:
            return bool(
                self._artifact_workbench_controller().snapshot.has_unsaved_changes
            )
        except Exception:
            _LOGGER.exception("Could not determine native document save status")
            return True

    def _ask_native_unsaved_action(
        self,
        action_label: str,
    ) -> QMessageBox.StandardButton:
        return QMessageBox.warning(
            self,
            "저장되지 않은 변경",
            "현재 유물 문서에 저장되지 않은 변경이 있습니다.\n"
            f"{action_label}하기 전에 저장하시겠습니까?",
            (
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            ),
            QMessageBox.StandardButton.Cancel,
        )

    def _ask_native_transient_action(
        self,
        action_label: str,
        state: _NativeTransientWorkState,
    ) -> QMessageBox.StandardButton:
        """Offer only explicit discard/cancel for work Save cannot serialize."""

        if not isinstance(state, _NativeTransientWorkState) or not (
            state.has_unpersisted_work
        ):
            return QMessageBox.StandardButton.Cancel
        return QMessageBox.warning(
            self,
            "저장할 수 없는 미확정 작업",
            "현재 장면에 프로젝트 파일로 저장할 수 없는 미확정 작업이 "
            "있습니다.\n\n"
            f"{state.detail()}\n\n"
            "계속 작업하려면 먼저 정치 확정·record 기록 또는 초기화를 "
            "완료하세요.\n"
            f"'{action_label}' 작업에서 [버리기]를 누르면 위 작업을 복구하지 "
            "않고 현재 장면을 버립니다.",
            (
                QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            ),
            QMessageBox.StandardButton.Cancel,
        )

    def _continue_after_native_unsaved_guard(
        self,
        action_label: str,
        continuation: Callable[[], None],
    ) -> bool:
        """Run a destructive action now, or after one exact durable Save."""

        if self._native_artifact_mode():
            transient_state = self._native_transient_work_state()
            if transient_state.has_unpersisted_work:
                reply = self._ask_native_transient_action(
                    action_label,
                    transient_state,
                )
                if reply == QMessageBox.StandardButton.Discard:
                    continuation()
                    return True
                return False
        if not self._native_document_has_unsaved_changes():
            continuation()
            return True
        reply = self._ask_native_unsaved_action(action_label)
        if reply == QMessageBox.StandardButton.Discard:
            continuation()
            return True
        if reply == QMessageBox.StandardButton.Save:
            self.save_project(on_saved=continuation)
        return False

    def _defer_exact_native_save_continuation(
        self,
        callback: Callable[[], None],
        *,
        captured_session: ArtifactSession,
        project_path: str,
        save_thread: object | None,
    ) -> None:
        """Continue after task cleanup only while the saved checkpoint is current."""

        normalized_project_path = str(
            Path(project_path).expanduser().resolve(strict=False)
        )
        cleanup_wait_attempts = 0

        def invoke_when_safe() -> None:
            nonlocal cleanup_wait_attempts
            if bool(getattr(self, "_application_closing", False)):
                return
            active_task = getattr(self, "_task_thread", None)
            if save_thread is not None and active_task is save_thread:
                # TaskThread emits done just before QThread emits finished.
                # Waiting for finished prevents close/open from re-entering
                # while the Save worker still owns the shared task slot.
                cleanup_wait_attempts += 1
                if cleanup_wait_attempts > 500:
                    _LOGGER.error(
                        "Native Save continuation stayed blocked after worker completion"
                    )
                    return
                QTimer.singleShot(1, invoke_when_safe)
                return
            if active_task is not None:
                return
            try:
                snapshot = self._artifact_workbench_controller().snapshot
                gui_project_path = getattr(self, "_current_project_path", None)
                normalized_gui_project_path = (
                    str(
                        Path(str(gui_project_path))
                        .expanduser()
                        .resolve(strict=False)
                    )
                    if gui_project_path is not None
                    else None
                )
                transient_state = self._native_transient_work_state()
                checkpoint_is_current = bool(
                    snapshot.session is captured_session
                    and getattr(self, "_artifact_session", None) is captured_session
                    and snapshot.project_path == normalized_project_path
                    and normalized_gui_project_path == normalized_project_path
                    and snapshot.can_save
                    and snapshot.save_checkpoint_current
                    and not transient_state.has_unpersisted_work
                )
            except Exception:
                _LOGGER.exception(
                    "Could not validate native Save continuation authority"
                )
                return
            if checkpoint_is_current:
                callback()
            else:
                self._refresh_native_save_indicator()

        QTimer.singleShot(0, invoke_when_safe)

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

    def open_file_path(
        self,
        filepath: str,
        *,
        prompt_unit: bool = True,
        source_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Open a mesh file from a known path."""
        if not filepath:
            return
        path = str(filepath)
        metadata = dict(source_metadata) if isinstance(source_metadata, dict) else None
        self._continue_after_native_unsaved_guard(
            "다른 원본을 열기",
            lambda: self._open_file_path_after_unsaved_guard(
                path,
                prompt_unit=prompt_unit,
                source_metadata=metadata,
            ),
        )

    def _open_file_path_after_unsaved_guard(
        self,
        filepath: str,
        *,
        prompt_unit: bool,
        source_metadata: dict[str, Any] | None,
    ) -> None:
        """Continue Open only after the current native document is safe."""

        if bool(prompt_unit):
            dialog = UnitSelectionDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            source_metadata = dialog.get_source_metadata()
        if not isinstance(source_metadata, dict):
            # Compatibility-only programmatic path. User-facing Open always
            # requires explicit metadata and enters native document mode.
            self.load_mesh(filepath, 1.0)
            return
        self._start_artifact_source_import(filepath, source_metadata)

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

    @staticmethod
    def _recovery_candidate_label(
        candidate: InterruptedProjectSave,
        index: int,
    ) -> str:
        def safe_name(value: str) -> str:
            name = Path(value).name
            cleaned = "".join(
                character if character.isprintable() else "_"
                for character in name
            )
            return cleaned[:96] or "unnamed"

        try:
            modified = datetime.fromtimestamp(
                candidate.modified_time_ns / 1_000_000_000
            ).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except (OSError, OverflowError, ValueError):
            modified = "시간 확인 불가"
        if candidate.size_bytes >= 1024 * 1024:
            size = f"{candidate.size_bytes / (1024 * 1024):.1f} MiB"
        elif candidate.size_bytes >= 1024:
            size = f"{candidate.size_bytes / 1024:.1f} KiB"
        else:
            size = f"{candidate.size_bytes} B"
        return (
            f"{index + 1}. {safe_name(candidate.intended_destination)} | "
            f"{size} | {modified} | {safe_name(candidate.candidate_path)}"
        )

    @staticmethod
    def _default_recovery_destination(candidate: InterruptedProjectSave) -> str:
        intended = Path(candidate.intended_destination)
        base = intended.with_name(f"{intended.stem}-recovered.amr")
        if not base.exists() and not base.is_symlink():
            return str(base)
        for index in range(2, 1000):
            alternate = intended.with_name(
                f"{intended.stem}-recovered-{index}.amr"
            )
            if not alternate.exists() and not alternate.is_symlink():
                return str(alternate)
        return str(
            intended.with_name(
                f"{intended.stem}-recovered-{uuid.uuid4().hex[:8]}.amr"
            )
        )

    def recover_interrupted_project_save(self) -> None:
        """Explicitly recover one verified native save temp to a new AMR."""

        project_thread = getattr(self, "_project_open_thread", None)
        mesh_thread = getattr(self, "_mesh_load_thread", None)
        task_thread = getattr(self, "_task_thread", None)
        workers_busy = False
        for thread in (project_thread, mesh_thread, task_thread):
            try:
                workers_busy = workers_busy or bool(
                    thread is not None and thread.isRunning()
                )
            except Exception:
                workers_busy = True
        if (
            workers_busy
            or bool(getattr(self, "_artifact_load_active", False))
            or bool(getattr(self, "_project_load_active", False))
        ):
            QMessageBox.information(
                self,
                "작업 중",
                "현재 검증·저장 작업이 끝난 뒤 중단 저장 복구를 시작하세요.",
            )
            return

        start_directory = ""
        for raw_path in (
            getattr(self, "_current_project_path", None),
            getattr(self, "current_filepath", None),
        ):
            if raw_path:
                try:
                    start_directory = str(Path(str(raw_path)).expanduser().parent)
                    break
                except (OSError, ValueError):
                    continue
        folder = QFileDialog.getExistingDirectory(
            self,
            "중단된 프로젝트 저장이 남은 폴더 선택",
            start_directory,
        )
        if not folder:
            return
        try:
            candidates = discover_amr_interrupted_saves(folder)
        except (ProjectRecoveryError, OSError, ValueError) as exc:
            QMessageBox.critical(
                self,
                "복구 후보 검색 실패",
                f"선택한 폴더를 안전하게 검사할 수 없습니다.\n\n{exc}",
            )
            return
        if not candidates:
            QMessageBox.information(
                self,
                "복구 후보 없음",
                "이 폴더에는 ArchMeshRubbing 저장 이름과 정확히 일치하는 "
                "중단 임시본이 없습니다. 일반 파일과 심볼릭 링크는 후보로 "
                "간주하지 않았습니다.",
            )
            return

        labels = tuple(
            self._recovery_candidate_label(candidate, index)
            for index, candidate in enumerate(candidates)
        )
        selected_label, accepted = QInputDialog.getItem(
            self,
            "복구 후보 선택",
            "아직 유효하다고 판정하지 않은 후보입니다. 완전 검증할 항목을 선택하세요.",
            labels,
            0,
            False,
        )
        if not accepted:
            return
        try:
            selected_index = labels.index(str(selected_label))
        except ValueError:
            QMessageBox.critical(
                self,
                "복구 후보 선택 실패",
                "선택한 후보를 현재 검색 결과와 결합할 수 없습니다.",
            )
            return
        candidate = candidates[selected_index]

        output, _ = QFileDialog.getSaveFileName(
            self,
            "검증된 복구본을 새 파일로 저장",
            self._default_recovery_destination(candidate),
            "ArchMeshRubbing Project (*.amr);;All Files (*)",
        )
        if not output:
            return
        if not output.lower().endswith(".amr"):
            output += ".amr"
        output_path = Path(output)
        if output_path.exists() or output_path.is_symlink():
            QMessageBox.warning(
                self,
                "덮어쓰기 차단",
                "복구는 기존 파일을 덮어쓰지 않습니다. 존재하지 않는 새 .amr "
                "파일 이름을 선택하세요.",
            )
            return

        answer = QMessageBox.question(
            self,
            "중단 저장 검증 및 복구",
            "다음 임시본을 별도 staging에 복사한 뒤 내장 원본·문서·Align을 "
            "완전히 검증합니다. 검증된 경우에만 새 파일을 생성합니다.\n\n"
            f"후보: {Path(candidate.candidate_path).name}\n"
            f"원래 저장 대상: {Path(candidate.intended_destination).name}\n"
            f"새 복구본: {output_path.name}\n\n"
            "중단 임시본과 기존 프로젝트는 성공 후에도 자동 삭제하거나 "
            "변경하지 않습니다. 계속할까요?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._start_interrupted_project_recovery(candidate, str(output_path))

    def _start_interrupted_project_recovery(
        self,
        candidate: InterruptedProjectSave,
        output: str,
    ) -> bool:
        def on_done(value: object) -> None:
            if not isinstance(value, ProjectRecoveryResult):
                raise ProjectRecoveryError(
                    "result",
                    "Project recovery worker returned an invalid result",
                )
            self._finish_interrupted_project_recovery(value)

        def on_failed(message: str) -> None:
            self.status_info.setText(
                "프로젝트 복구 실패 | 후보·기존 파일 유지"
            )
            QMessageBox.critical(
                self,
                "프로젝트 복구 실패",
                self._format_error_message(
                    "중단 임시본을 검증하거나 새 복구본으로 게시하지 못했습니다. "
                    "후보와 기존 파일은 유지했습니다:",
                    message,
                ),
            )

        try:
            return bool(
                self._start_task(
                    title="중단된 프로젝트 저장 복구",
                    label=(
                        "임시본을 복사하고 내장 원본·문서·Align을 완전 검증하는 중..."
                    ),
                    thread=TaskThread(
                        "recover_interrupted_project_save",
                        lambda: recover_amr_interrupted_save(candidate, output),
                    ),
                    on_done=on_done,
                    on_failed=on_failed,
                    lock_dialog_until_finished=True,
                )
            )
        except Exception as exc:
            self.status_info.setText("프로젝트 복구 작업 시작 실패")
            QMessageBox.critical(
                self,
                "프로젝트 복구 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return False

    def _finish_interrupted_project_recovery(
        self,
        result: ProjectRecoveryResult,
    ) -> None:
        self.status_info.setText(
            f"검증된 프로젝트 복구본 생성: {Path(result.destination).name}"
        )
        detail = (
            "내장 원본·문서·Align을 완전 검증한 새 프로젝트를 생성했습니다.\n\n"
            f"파일: {result.destination}\n"
            f"문서 ID: {result.document_id}\n"
            f"파일 SHA-256: {result.project_sha256}\n\n"
            "중단 임시본과 기존 프로젝트는 그대로 유지했습니다. "
            "복구본을 지금 열까요?"
        )
        if result.durability_warning:
            detail = (
                "복구본은 검증되어 생성됐지만 디렉터리 동기화를 확인하지 "
                "못했습니다. 즉시 다른 저장장치에도 복사하는 편이 안전합니다.\n\n"
                f"{result.durability_warning}\n\n{detail}"
            )
        answer = QMessageBox.question(
            self,
            "프로젝트 복구 완료",
            detail,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.open_project_path(result.destination)

    def open_project_path(self, filepath: str) -> None:
        """Open a project file (.amr) from a known path (no file dialog)."""
        if not filepath:
            return
        path = str(filepath)
        self._continue_after_native_unsaved_guard(
            "다른 프로젝트를 열기",
            lambda: self._open_project_path_after_unsaved_guard(path),
        )

    def _open_project_path_after_unsaved_guard(self, filepath: str) -> None:
        """Start Project Open only after the current native document is safe."""

        thread = getattr(self, "_mesh_load_thread", None)
        project_thread = getattr(self, "_project_open_thread", None)
        if (
            bool(getattr(self, "_artifact_load_active", False))
            or bool(getattr(self, "_project_load_active", False))
            or (thread is not None and thread.isRunning())
            or (project_thread is not None and project_thread.isRunning())
        ):
            QMessageBox.information(
                self,
                "로딩 중",
                "이미 다른 원본 또는 프로젝트를 검증하고 있습니다.",
            )
            return

        request_id = f"project-open:{uuid.uuid4()}"
        load_thread = TaskThread(
            "project-open",
            lambda path=str(filepath): _load_project_open_candidate(path),
        )
        controller = self._artifact_workbench_controller()
        self._project_open_thread = load_thread
        self._project_open_request_id = request_id
        self._project_open_base_authority_epoch = controller.snapshot.authority_epoch
        self._artifact_load_active = True
        self._artifact_pending_document = None
        self._artifact_pending_project_path = str(filepath)
        self._artifact_pending_source_metadata = None
        self._artifact_load_ticket = None
        self.status_info.setText(
            f"프로젝트 패키지 검증 중: {Path(filepath).name}"
        )
        load_thread.done.connect(
            lambda result, owner=load_thread, rid=request_id, path=str(filepath): (
                self._dispatch_project_open_success(owner, rid, path, result)
            )
        )
        load_thread.failed.connect(
            lambda message, owner=load_thread, rid=request_id: (
                self._dispatch_project_open_failure(owner, rid, message)
            )
        )
        load_thread.finished.connect(
            lambda owner=load_thread, rid=request_id: (
                self._dispatch_project_open_finished(owner, rid)
            )
        )
        load_thread.start()

    def _project_open_result_is_current(
        self,
        owner: QThread,
        request_id: str,
    ) -> bool:
        if not self._project_open_worker_is_current(owner, request_id):
            return False
        controller = getattr(self, "_artifact_workbench", None)
        base_epoch = getattr(self, "_project_open_base_authority_epoch", None)
        return (
            isinstance(controller, ArtifactWorkbench)
            and isinstance(base_epoch, int)
            and controller.snapshot.authority_epoch == base_epoch
        )

    def _project_open_worker_is_current(
        self,
        owner: QThread,
        request_id: str,
    ) -> bool:
        return (
            owner is getattr(self, "_project_open_thread", None)
            and request_id == getattr(self, "_project_open_request_id", None)
        )

    def _dispatch_project_open_success(
        self,
        owner: QThread,
        request_id: str,
        filepath: str,
        result: object,
    ) -> None:
        if not self._project_open_result_is_current(owner, request_id):
            _LOGGER.info("Discarded stale project-open success: %s", request_id)
            if self._project_open_worker_is_current(owner, request_id):
                self._clear_artifact_pending_load(cancel_workbench=True)
                self.status_info.setText(
                    "Project Open 결과 폐기 | 더 최신 문서 권위 유지"
                )
            return
        if not isinstance(result, dict):
            self._on_project_open_failure("invalid Project Open worker result")
            return

        kind = str(result.get("kind", ""))
        if kind == "artifact_embedded":
            session = result.get("session")
            document = result.get("document")
            if not isinstance(session, ArtifactSession) or not isinstance(
                document,
                ArtifactDocument,
            ):
                self._on_project_open_failure(
                    "embedded Project Open worker returned invalid native state"
                )
                return
            self._finish_embedded_artifact_project_open(
                session,
                document=document,
                project_path=filepath,
                request_id=request_id,
            )
            return

        # Package inspection is complete. The manifest-only and legacy paths
        # now continue through their existing GUI-thread source-resolution
        # adapters; neither path hashes or parses a large embedded source here.
        self._clear_artifact_pending_load(cancel_workbench=False)
        document = result.get("document")
        if kind == "artifact_manifest_only" and isinstance(
            document,
            ArtifactDocument,
        ):
            self._start_artifact_project_load(document, filepath)
            return
        if kind == "legacy" and isinstance(document, dict):
            self._start_legacy_project_load(document, filepath)
            return
        self._on_project_open_failure("unknown Project Open worker result")

    def _dispatch_project_open_failure(
        self,
        owner: QThread,
        request_id: str,
        message: str,
    ) -> None:
        if not self._project_open_result_is_current(owner, request_id):
            _LOGGER.info("Discarded stale project-open failure: %s", request_id)
            if self._project_open_worker_is_current(owner, request_id):
                self._clear_artifact_pending_load(cancel_workbench=True)
            return
        self._on_project_open_failure(message)

    def _dispatch_project_open_finished(
        self,
        owner: QThread,
        request_id: str,
    ) -> None:
        if not self._project_open_worker_is_current(owner, request_id):
            try:
                owner.deleteLater()
            except Exception:
                pass
            _LOGGER.info("Ignored stale project-open finished signal: %s", request_id)
            return
        try:
            owner.deleteLater()
        except Exception:
            pass
        self._project_open_thread = None
        self._project_open_request_id = None
        self._project_open_base_authority_epoch = None

    def _on_project_open_failure(self, message: str) -> None:
        self._clear_artifact_pending_load(cancel_workbench=True)
        self.status_info.setText("프로젝트 패키지 검증 실패 | 기존 scene 유지")
        QMessageBox.critical(
            self,
            "오류",
            "프로젝트를 열 수 없습니다. 기존 작업은 유지했습니다."
            f"\n\n{message}",
        )

    def _start_legacy_project_load(
        self,
        doc: dict[str, Any],
        filepath: str,
    ) -> None:
        try:
            state = doc.get("state", {})
            migration = doc.get(MIGRATION_MARKER_NAME, {})
            migrated_from_v1 = bool(
                isinstance(migration, dict)
                and migration.get("from_version") == 1
                and migration.get("requires_save_as") is True
            )
            if not isinstance(state, dict):
                raise ProjectFormatError("Invalid project state")
            _validate_project_source_declarations(
                state,
                migrated_from_v1=migrated_from_v1,
            )
        except (ProjectFormatError, ValueError) as exc:
            self._on_project_open_failure(str(exc))
            return

        objects = state.get("objects", [])
        if not isinstance(objects, list) or not objects:
            QMessageBox.warning(self, "경고", "프로젝트에 로드할 객체(objects)가 없습니다.")
            return

        # Keep the live scene intact while every external source is loaded and
        # verified in CPU staging. It is swapped only after the full queue
        # succeeds, so a late mismatch cannot destroy unsaved work.
        self._project_previous_context = {
            "current_project_path": self._current_project_path,
            "requires_save_as": self._project_requires_save_as,
            "legacy_project_path": self._legacy_project_path,
            "has_legacy_bindings": self._project_has_legacy_bindings,
            "load_failed": self._project_load_failed,
        }
        self._project_staged_objects = []

        # A migrated v1 file is never overwritten by the first v2 save.
        self._project_load_from_legacy = migrated_from_v1
        self._project_requires_save_as = migrated_from_v1
        self._legacy_project_path = str(filepath) if migrated_from_v1 else None
        # Commit the destination path only after every queued source has been
        # verified and applied. Saving is disabled while this transaction is
        # active or after it fails.
        self._project_pending_path = str(filepath)
        self._project_load_failed = False
        self._last_source_verification = None
        self._project_has_legacy_bindings = migrated_from_v1
        self._project_load_active = True
        self._project_load_state = state
        self._project_load_queue = [o for o in objects if isinstance(o, dict)]
        self._project_load_current = None

        self.status_info.setText(f"프로젝트 로딩 중: {Path(filepath).name}")
        self._start_next_project_object_load()

    def _clear_artifact_pending_load(self, *, cancel_workbench: bool = True) -> None:
        ticket = getattr(self, "_artifact_load_ticket", None)
        if cancel_workbench and isinstance(ticket, ArtifactLoadTicket):
            try:
                controller = self._artifact_workbench_controller()
                if controller.snapshot.pending_load is ticket:
                    controller.cancel_load(ticket)
            except (ArtifactWorkbenchError, StaleWorkflowOperationError):
                _LOGGER.debug("Artifact Open ticket was already completed", exc_info=True)
        self._artifact_load_active = False
        self._artifact_pending_document = None
        self._artifact_pending_project_path = None
        self._artifact_pending_source_metadata = None
        self._artifact_load_ticket = None
        try:
            self._prune_pending_native_measurement_publications()
        except Exception:
            _LOGGER.debug(
                "Could not refresh pending measurement publication state",
                exc_info=True,
            )

    def _start_artifact_source_import(
        self,
        filepath: str,
        source_metadata: dict[str, Any],
    ) -> None:
        if bool(getattr(self, "_artifact_load_active", False)) or bool(
            getattr(self, "_project_load_active", False)
        ):
            QMessageBox.information(
                self,
                "로딩 중",
                "이미 다른 원본 또는 프로젝트를 검증하고 있습니다.",
            )
            return
        if str(source_metadata.get("confirmation_status", "")) != "confirmed":
            QMessageBox.warning(
                self,
                "원본 metadata 미확정",
                "단위와 좌표축을 확인해야 원본을 실측 문서로 열 수 있습니다.",
            )
            return
        thread = getattr(self, "_mesh_load_thread", None)
        if thread is not None and thread.isRunning():
            QMessageBox.information(self, "로딩 중", "이미 다른 메쉬를 로딩 중입니다.")
            return
        source_format = Path(filepath).suffix.lower().removeprefix(".")
        if f".{source_format}" not in MeshLoader.SUPPORTED_FORMATS:
            QMessageBox.critical(self, "오류", f"지원하지 않는 원본 형식: {source_format!r}")
            return
        try:
            ticket = self._artifact_workbench_controller().begin_new_import(
                filepath,
                source_metadata,
                software_version=APP_VERSION,
                operator="local-user",
            )
        except (ArtifactWorkbenchError, WorkflowBusyError) as exc:
            QMessageBox.warning(self, "원본 열기 차단", str(exc))
            return
        self._artifact_load_active = True
        self._artifact_load_ticket = ticket
        self._artifact_pending_document = None
        self._artifact_pending_project_path = None
        self._artifact_pending_source_metadata = copy.deepcopy(source_metadata)
        started = self._start_async_load(
            filepath,
            1.0,
            source_format=ticket.source_format,
            import_recipe=ticket.import_recipe,
            source_unit=ticket.source_unit,
            artifact_ticket=ticket,
        )
        if not started:
            self._clear_artifact_pending_load(cancel_workbench=True)

    def _artifact_source_context(
        self,
        document: ArtifactDocument,
    ) -> tuple[Any, Any, Any]:
        metadata_id = document.active_source_metadata_revision_id
        if metadata_id is None:
            raise ProjectFormatError("ArtifactDocument has no active metadata revision")
        metadata = document.source_metadata_revision_index[metadata_id]
        geometry = document.geometry_revision_index[metadata.geometry_revision_id]
        if len(geometry.source_asset_ids) != 1:
            raise ProjectFormatError("M0-3 supports exactly one source asset")
        source_asset = document.source_asset_index[geometry.source_asset_ids[0]]
        return source_asset, geometry, metadata

    def _resolve_artifact_source_path(
        self,
        source_asset: Any,
        project_path: str,
    ) -> str | None:
        raw_ref = str(getattr(source_asset, "asset_ref", "") or "").strip()
        raw_path = raw_ref.removeprefix("external:")
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(project_path).resolve(strict=False).parent / candidate
        if candidate.exists():
            return str(candidate)

        selected, _ = QFileDialog.getOpenFileName(
            self,
            f"원본 파일 찾기: {getattr(source_asset, 'original_name', 'artifact')}",
            str(candidate.parent if candidate.parent.exists() else ""),
            "3D Files (*.obj *.ply *.stl *.off *.gltf *.glb);;All Files (*)",
        )
        return str(selected) if selected else None

    def _finish_embedded_artifact_project_open(
        self,
        session: ArtifactSession,
        *,
        document: ArtifactDocument,
        project_path: str,
        request_id: str,
    ) -> None:
        """Publish a worker-materialized embedded package through Workbench."""

        ticket: ArtifactLoadTicket | None = None
        try:
            if session.document != document:
                raise ArtifactSessionError(
                    "embedded worker session does not match its validated document"
                )
            controller = self._artifact_workbench_controller()
            ticket = controller.begin_project_reopen(
                document,
                project_path=project_path,
                resolved_source_path=session.resolved_source_path,
                request_id=request_id,
            )
            self._artifact_load_ticket = ticket
            self._artifact_pending_document = document
            self._artifact_pending_project_path = str(project_path)
            transition = controller.prepare_loaded_source(
                ticket,
                session.source_mesh,
                resolved_source_path=session.resolved_source_path,
            )
            candidate = transition.candidate_session
            self._publish_artifact_session_projection(
                candidate,
                project_path=str(project_path),
                fit_camera=True,
                status_text=(
                    f"내장 원본 프로젝트 로딩 완료: {Path(project_path).name} "
                    "| package·source·geometry·Align 검증됨"
                ),
                workflow_transition=transition,
            )
            self._clear_artifact_pending_load(cancel_workbench=False)
        except Exception as exc:
            if isinstance(ticket, ArtifactLoadTicket):
                try:
                    controller = self._artifact_workbench_controller()
                    if controller.snapshot.pending_load is ticket:
                        controller.fail_load(ticket, exc)
                except (ArtifactWorkbenchError, StaleWorkflowOperationError):
                    _LOGGER.debug("Embedded Project Open failure was stale", exc_info=True)
            self._clear_artifact_pending_load(cancel_workbench=False)
            self.status_info.setText(
                "내장 원본 프로젝트 staging 실패 | 기존 scene 유지"
            )
            QMessageBox.critical(
                self,
                "ArtifactDocument 로딩 실패",
                "내장 원본·geometry·장면 검증 중 실패하여 기존 작업을 유지했습니다."
                f"\n\n{type(exc).__name__}: {exc}",
            )

    def _start_artifact_project_load(
        self,
        document: ArtifactDocument,
        project_path: str,
    ) -> None:
        if bool(getattr(self, "_artifact_load_active", False)) or bool(
            getattr(self, "_project_load_active", False)
        ):
            QMessageBox.information(
                self,
                "로딩 중",
                "이미 다른 원본 또는 프로젝트를 검증하고 있습니다.",
            )
            return
        try:
            source_asset, geometry, metadata = self._artifact_source_context(document)
            resolved = self._resolve_artifact_source_path(source_asset, project_path)
            if not resolved:
                raise ProjectFormatError("ArtifactDocument source was not resolved")
            source_format = str(geometry.import_recipe.get("format", "") or "").strip().lower()
            if f".{source_format}" not in MeshLoader.SUPPORTED_FORMATS:
                raise ProjectFormatError(
                    f"ArtifactDocument has unsupported parser format: {source_format!r}"
                )
            if str(metadata.confirmation_status.value) != "confirmed":
                raise ProjectFormatError("ArtifactDocument metadata is not confirmed")
            thread = getattr(self, "_mesh_load_thread", None)
            if thread is not None and thread.isRunning():
                raise ProjectFormatError("another mesh load is already active")
        except (ArtifactSessionError, ProjectFormatError, KeyError, ValueError) as exc:
            QMessageBox.critical(self, "오류", f"ArtifactDocument를 열 수 없습니다:\n{exc}")
            return

        try:
            ticket = self._artifact_workbench_controller().begin_project_reopen(
                document,
                project_path=project_path,
                resolved_source_path=resolved,
            )
        except (ArtifactWorkbenchError, WorkflowBusyError) as exc:
            QMessageBox.critical(self, "오류", f"ArtifactDocument를 열 수 없습니다:\n{exc}")
            return

        self._artifact_load_active = True
        self._artifact_load_ticket = ticket
        self._artifact_pending_document = document
        self._artifact_pending_project_path = str(project_path)
        self._artifact_pending_source_metadata = None
        started = self._start_async_load(
            ticket.source_path,
            1.0,
            source_format=ticket.source_format,
            import_recipe=ticket.import_recipe,
            source_unit=ticket.source_unit,
            artifact_ticket=ticket,
        )
        if not started:
            self._clear_artifact_pending_load(cancel_workbench=True)

    def save_project(
        self,
        *,
        on_saved: Callable[[], None] | None = None,
    ) -> None:
        if bool(getattr(self, "_artifact_load_active", False)) or bool(
            getattr(self, "_project_load_active", False)
        ) or bool(
            getattr(self, "_project_load_failed", False)
        ) or bool(
            getattr(self, "_artifact_authority_faulted", False)
        ):
            QMessageBox.warning(
                self,
                "프로젝트 저장 차단",
                "프로젝트 원본 검증이 진행 중이거나 실패했습니다. 부분 scene을 저장하지 "
                "않도록 차단했습니다. 정상 프로젝트를 다시 여세요.",
            )
            return
        if bool(getattr(self, "_project_requires_save_as", False)):
            QMessageBox.information(
                self,
                "구버전 프로젝트 보존",
                "이 프로젝트는 AMR v1에서 읽었습니다. 원본을 보존하기 위해 "
                "첫 저장은 새 AMR v2 파일로만 할 수 있습니다.",
            )
            self.save_project_as(on_saved=on_saved)
            return
        if not getattr(self, "_current_project_path", None):
            self.save_project_as(on_saved=on_saved)
            return
        destination = str(self._current_project_path)
        if isinstance(getattr(self, "_artifact_session", None), ArtifactSession):
            self._start_native_project_save(destination, on_saved=on_saved)
            return
        if self._write_project(destination) and callable(on_saved):
            on_saved()

    def save_project_as(
        self,
        *,
        on_saved: Callable[[], None] | None = None,
    ) -> None:
        if bool(getattr(self, "_artifact_load_active", False)) or bool(
            getattr(self, "_project_load_active", False)
        ) or bool(
            getattr(self, "_project_load_failed", False)
        ) or bool(
            getattr(self, "_artifact_authority_faulted", False)
        ):
            QMessageBox.warning(
                self,
                "프로젝트 저장 차단",
                "검증에 실패한 부분 scene은 새 프로젝트로도 저장할 수 없습니다. "
                "정상 프로젝트를 다시 여세요.",
            )
            return
        default_name = DEFAULT_PROJECT_FILENAME
        legacy_path = str(getattr(self, "_legacy_project_path", "") or "").strip()
        if bool(getattr(self, "_project_requires_save_as", False)) and legacy_path:
            legacy = Path(legacy_path)
            default_name = str(legacy.with_name(f"{legacy.stem}-v2.amr"))
        try:
            if self.current_filepath and not legacy_path:
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

        if isinstance(getattr(self, "_artifact_session", None), ArtifactSession):
            self._start_native_project_save(filepath, on_saved=on_saved)
            return
        if self._write_project(filepath):
            self._current_project_path = str(filepath)
            if callable(on_saved):
                on_saved()

    @staticmethod
    def _write_native_project_snapshot(
        filepath: str,
        session: ArtifactSession,
        meta: Mapping[str, Any],
        preflight: Callable[[], None],
    ) -> _NativeProjectSaveResult:
        """Validate and persist one immutable native snapshot on a worker."""

        preflight()
        worker_meta = dict(meta)
        sha, dirty = _safe_git_info(str(Path(basedir)))
        worker_meta["git"] = (
            f"{sha}{'*' if dirty else ''}" if sha else "unknown"
        )
        try:
            destination = save_amr_artifact_session_project(
                filepath,
                session,
                meta=worker_meta,
            )
        except ProjectSaveError as exc:
            if not exc.committed:
                raise
            return _NativeProjectSaveResult(
                destination=str(filepath),
                durability_warning=str(exc),
            )
        return _NativeProjectSaveResult(destination=str(destination))

    def _start_native_project_save(
        self,
        filepath: str,
        *,
        on_saved: Callable[[], None] | None = None,
    ) -> bool:
        """Start a native project save without blocking the Qt event loop."""

        if bool(getattr(self, "_artifact_load_active", False)) or bool(
            getattr(self, "_project_load_active", False)
        ) or bool(
            getattr(self, "_project_load_failed", False)
        ) or bool(
            getattr(self, "_artifact_authority_faulted", False)
        ):
            QMessageBox.warning(
                self,
                "프로젝트 저장 차단",
                "검증 중이거나 권위가 불확실한 프로젝트는 저장할 수 없습니다.",
            )
            return False

        session = getattr(self, "_artifact_session", None)
        if not isinstance(session, ArtifactSession):
            return False
        destination = str(filepath)
        legacy_path = str(getattr(self, "_legacy_project_path", "") or "").strip()
        if bool(getattr(self, "_project_requires_save_as", False)) and legacy_path:
            if _same_filesystem_target(destination, legacy_path):
                QMessageBox.warning(
                    self,
                    "구버전 프로젝트 보존",
                    "AMR v1 원본은 덮어쓸 수 없습니다. 다른 파일 이름을 선택하세요.",
                )
                return False

        try:
            controller = self._artifact_workbench_controller()
            controller.require_stable_session(session)
            preflight = self._capture_native_scene_preflight(session)
            snapshot = controller.snapshot
            if snapshot.session is not session or snapshot.tentative:
                raise ArtifactWorkbenchError(
                    "native save snapshot does not own stable project authority"
                )
            current_project_path = getattr(self, "_current_project_path", None)
            normalized_current_path = (
                str(Path(str(current_project_path)).expanduser().resolve(strict=False))
                if current_project_path is not None
                else None
            )
            if snapshot.project_path != normalized_current_path:
                raise ArtifactWorkbenchError(
                    "GUI and Workbench project paths do not share one authority"
                )
            meta = {
                "app": APP_NAME,
                "version": APP_VERSION,
            }
        except Exception as exc:
            self.status_info.setText("프로젝트 저장 준비 실패 | 기존 파일 유지")
            QMessageBox.warning(
                self,
                "프로젝트 저장 준비 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return False

        base_state_version = snapshot.state_version
        base_authority_epoch = snapshot.authority_epoch
        base_project_path = getattr(self, "_current_project_path", None)
        base_requires_save_as = bool(
            getattr(self, "_project_requires_save_as", False)
        )
        base_legacy_path = getattr(self, "_legacy_project_path", None)
        save_thread: TaskThread | None = None

        def authority_is_current() -> bool:
            current = controller.snapshot
            return bool(
                not current.tentative
                and current.pending_load is None
                and current.session is session
                and getattr(self, "_artifact_session", None) is session
                and current.state_version == base_state_version
                and current.authority_epoch == base_authority_epoch
                and getattr(self, "_current_project_path", None) == base_project_path
                and bool(getattr(self, "_project_requires_save_as", False))
                == base_requires_save_as
                and getattr(self, "_legacy_project_path", None) == base_legacy_path
                and not bool(getattr(self, "_project_load_failed", False))
                and not bool(getattr(self, "_artifact_authority_faulted", False))
            )

        def report_snapshot_only(
            value: _NativeProjectSaveResult,
            *,
            reason: str,
        ) -> None:
            detail = (
                "저장 작업이 캡처한 snapshot은 파일에 기록됐지만 "
                f"{reason} 현재 작업을 다시 저장하세요."
            )
            if value.durability_warning:
                detail += (
                    "\n\n또한 디렉터리 동기화를 확인하지 못했습니다:\n"
                    f"{value.durability_warning}"
                )
            self.status_info.setText(
                "이전 snapshot 저장됨 | 현재 문서는 다시 저장 필요"
            )
            QMessageBox.warning(
                self,
                "프로젝트 snapshot 저장됨",
                detail,
            )

        def on_done(value: object) -> None:
            if not isinstance(value, _NativeProjectSaveResult):
                raise ProjectSerializationError(
                    "native project save worker returned an invalid result"
                )
            if not authority_is_current():
                report_snapshot_only(
                    value,
                    reason="그 사이 현재 문서 권위가 변경됐습니다.",
                )
                return

            try:
                controller.adopt_saved_project_path(
                    session,
                    value.destination,
                    expected_state_version=base_state_version,
                    expected_authority_epoch=base_authority_epoch,
                    durability_confirmed=not bool(value.durability_warning),
                )
            except ArtifactWorkbenchError:
                _LOGGER.warning(
                    "Native project save path adoption was rejected",
                    exc_info=True,
                )
                report_snapshot_only(
                    value,
                    reason=(
                        "경로를 채택하기 직전에 현재 문서 권위가 변경됐습니다."
                    ),
                )
                return

            # Keep the user's selected spelling for the shell while the
            # Workbench owns the normalized authority locator.
            self._current_project_path = value.destination
            self._project_requires_save_as = False
            self._legacy_project_path = None
            if value.durability_warning:
                QMessageBox.warning(
                    self,
                    "저장 내구성 경고",
                    "프로젝트 파일은 원자적으로 교체되었지만 디렉터리 동기화에 "
                    "실패했습니다. 파일은 다시 열 수 있으나 직후 시스템 장애에 대한 "
                    "내구성은 확정할 수 없습니다.\n\n"
                    f"{value.durability_warning}",
                )
                self.status_info.setText(
                    "프로젝트 저장됨 | crash durability 미확정"
                )
                return
            self.status_info.setText(
                f"프로젝트 저장: {Path(value.destination).name}"
            )
            if callable(on_saved):
                self._defer_exact_native_save_continuation(
                    on_saved,
                    captured_session=session,
                    project_path=value.destination,
                    save_thread=save_thread,
                )

        def on_failed(message: str) -> None:
            self.status_info.setText("프로젝트 저장 실패 | 기존 파일 유지")
            QMessageBox.critical(
                self,
                "프로젝트 저장 실패",
                self._format_error_message(
                    "프로젝트 snapshot 저장 중 오류가 발생했습니다:",
                    message,
                ),
            )

        try:
            save_thread = TaskThread(
                "save_native_project",
                lambda: MainWindow._write_native_project_snapshot(
                    destination,
                    session,
                    meta,
                    preflight,
                ),
            )
            started = self._start_task(
                title="프로젝트 저장",
                label="원본·문서·Align을 검증하고 AMR 패키지를 저장하는 중...",
                thread=save_thread,
                on_done=on_done,
                on_failed=on_failed,
                lock_dialog_until_finished=True,
            )
        except Exception as exc:
            self.status_info.setText("프로젝트 저장 작업 시작 실패")
            QMessageBox.critical(
                self,
                "프로젝트 저장 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return False
        return bool(started)

    def _write_project(self, filepath: str) -> bool:
        if bool(getattr(self, "_artifact_load_active", False)) or bool(
            getattr(self, "_project_load_active", False)
        ) or bool(
            getattr(self, "_project_load_failed", False)
        ) or bool(
            getattr(self, "_artifact_authority_faulted", False)
        ):
            return False
        legacy_path = str(getattr(self, "_legacy_project_path", "") or "").strip()
        if bool(getattr(self, "_project_requires_save_as", False)) and legacy_path:
            if _same_filesystem_target(filepath, legacy_path):
                QMessageBox.warning(
                    self,
                    "구버전 프로젝트 보존",
                    "AMR v1 원본은 덮어쓸 수 없습니다. 다른 파일 이름을 선택하세요.",
                )
                return False
        try:
            sha, dirty = _safe_git_info(str(Path(basedir)))
            meta = {
                "app": APP_NAME,
                "version": APP_VERSION,
                "git": f"{sha}{'*' if dirty else ''}" if sha else "unknown",
            }
            session = getattr(self, "_artifact_session", None)
            if isinstance(session, ArtifactSession):
                self._artifact_workbench_controller().require_stable_session(session)
                self._validate_native_scene_for_save(session)
                save_amr_artifact_session_project(filepath, session, meta=meta)
            else:
                state = self._collect_project_state()
                save_amr_project(filepath, state, meta=meta)
            self._project_requires_save_as = False
            self._legacy_project_path = None
            self.status_info.setText(f"프로젝트 저장: {Path(filepath).name}")
            return True
        except ProjectSaveError as e:
            if e.committed:
                self._project_requires_save_as = False
                self._legacy_project_path = None
                QMessageBox.warning(
                    self,
                    "저장 내구성 경고",
                    "프로젝트 파일은 원자적으로 교체되었지만 디렉터리 동기화에 "
                    "실패했습니다. 파일은 다시 열 수 있으나 직후 시스템 장애에 대한 "
                    f"내구성은 확정할 수 없습니다.\n\n{e}",
                )
                self.status_info.setText("프로젝트 저장됨 | crash durability 미확정")
                return True
            QMessageBox.critical(self, "오류", f"프로젝트 저장 실패:\n{e}")
            self.status_info.setText("프로젝트 저장 실패")
            return False
        except Exception as e:
            QMessageBox.critical(self, "오류", f"프로젝트 저장 실패:\n{type(e).__name__}: {e}")
            self.status_info.setText("프로젝트 저장 실패")
            return False

    def _capture_native_scene_preflight(
        self,
        session: ArtifactSession,
        *,
        allowed_selected_face_indices: tuple[int, ...] | None = None,
        allow_surface_measurement_picks: bool = False,
    ) -> Callable[[], None]:
        """Capture GUI-only guards and defer canonical mesh comparison to a worker."""

        objects = list(getattr(self.viewport, "objects", []) or [])
        if len(objects) != 1:
            raise ProjectSerializationError(
                "ArtifactDocument M0-3 save requires exactly one projected artifact"
            )
        obj = objects[0]
        binding = getattr(obj, "_amr_artifact_projection_snapshot", None)
        if not isinstance(binding, ArtifactProjectionSnapshot):
            raise ProjectSerializationError(
                "Native scene object has no authoritative projection binding"
            )
        expected = session.projection_snapshot()
        if binding != expected:
            raise ProjectSerializationError(
                "Native scene projection is stale for the active ArtifactDocument"
            )
        transient_state = MainWindow._native_transient_work_state(
            self,
            obj=obj,
            allowed_selected_face_indices=allowed_selected_face_indices,
            allow_surface_measurement_picks=allow_surface_measurement_picks,
        )
        if transient_state.has_unpersisted_work:
            raise ProjectSerializationError(
                "현재 장면에 저장할 수 없는 미확정 작업이 있습니다. "
                + " ".join(transient_state.reasons)
            )

        actual_mesh = getattr(obj, "mesh", None)

        def validate_geometry() -> None:
            if session.projection_snapshot() != expected:
                raise ProjectSerializationError(
                    "Native scene preflight projection changed before execution"
                )
            expected_mesh = session.materialize().mesh
            try:
                vertices_match = np.array_equal(
                    np.asarray(actual_mesh.vertices),
                    np.asarray(expected_mesh.vertices),
                )
                faces_match = np.array_equal(
                    np.asarray(actual_mesh.faces),
                    np.asarray(expected_mesh.faces),
                )
            except Exception:
                vertices_match = False
                faces_match = False
            if not vertices_match or not faces_match:
                raise ProjectSerializationError(
                    "Native scene geometry가 ArtifactDocument의 canonical projection과 "
                    "다릅니다"
                )

        return validate_geometry

    def _validate_native_scene_for_save(self, session: ArtifactSession) -> None:
        MainWindow._capture_native_scene_preflight(self, session)()

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
            if bool(getattr(obj, "_amr_has_unpersisted_bake", False)):
                raise ProjectSerializationError(
                    "정치 확정으로 변경된 메쉬 정점은 현재 AMR legacy payload에 "
                    "재현 가능하게 저장할 수 없습니다. 원본을 다시 열어 TRS 상태로 "
                    "저장하거나 M0-3 정렬 revision 경로를 사용하세요."
                )
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
            source_payload = _mesh_source_payload(mesh, mesh_path)
            alignment_status = str(
                getattr(obj, "_amr_alignment_status", "")
                or _ALIGNMENT_STATUS_MUTABLE_TRS
            ).strip()
            if alignment_status not in {
                _ALIGNMENT_STATUS_MUTABLE_TRS,
                _ALIGNMENT_STATUS_UNVERIFIABLE,
                _ALIGNMENT_STATUS_BAKED_UNVERIFIABLE,
            }:
                raise ProjectSerializationError(
                    f"Unsupported alignment status: {alignment_status!r}"
                )

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
                    "mesh": {
                        "path": mesh_path,
                        "source_scale_factor": source_scale,
                        "source": source_payload,
                    },
                    "alignment": {"status": alignment_status},
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
        else:
            dpi = DEFAULT_EXPORT_DPI
            format_index = 0
            scale_bar = True
            profile_include_grid = True
            profile_feature_lines = False
            profile_feature_angle = 60.0
            review_render_mode = "auto"

        ui_state["export"] = {
            "dpi": int(dpi),
            "format_index": int(format_index),
            "scale_bar": bool(scale_bar),
            "profile_include_grid": bool(profile_include_grid),
            "profile_feature_lines": bool(profile_feature_lines),
            "profile_feature_angle": float(profile_feature_angle),
            "review_render_mode": str(review_render_mode or "auto"),
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

    def _discard_project_staging_and_restore_context(self) -> None:
        previous = getattr(self, "_project_previous_context", None)
        previous_load_failed = False
        if isinstance(previous, dict):
            self._current_project_path = previous.get("current_project_path")
            self._project_requires_save_as = bool(previous.get("requires_save_as", False))
            legacy_path = previous.get("legacy_project_path")
            self._legacy_project_path = str(legacy_path) if legacy_path else None
            self._project_has_legacy_bindings = bool(
                previous.get("has_legacy_bindings", False)
            )
            previous_load_failed = bool(previous.get("load_failed", False))
        self._project_previous_context = None
        self._project_staged_objects = []
        self._project_pending_path = None
        self._project_load_failed = previous_load_failed

    def _abort_project_source_load(
        self,
        verification: SourceVerification,
        *,
        message: str,
    ) -> None:
        """Stop queued loading before unverified state reaches a scene object."""
        self._last_source_verification = verification
        self._project_load_active = False
        self._project_load_queue = []
        self._project_load_current = None
        self._project_load_state = None
        self._project_load_from_legacy = False
        self._discard_project_staging_and_restore_context()
        self.status_info.setText(f"원본 검증 실패: {verification.status.value}")
        if verification.status is SourceVerificationStatus.MISSING:
            QMessageBox.warning(self, "원본 파일 없음", message)
        else:
            QMessageBox.critical(self, "원본 검증 실패", message)

    def _expected_source_for_project_object(
        self,
        obj_state: dict[str, Any],
    ) -> SourceFingerprint | None:
        if bool(getattr(self, "_project_load_from_legacy", False)):
            return None
        mesh_info = obj_state.get("mesh", {})
        if not isinstance(mesh_info, dict):
            return None
        source = mesh_info.get("source", {})
        if not isinstance(source, dict):
            return None
        raw_identity = source.get("identity")
        if not isinstance(raw_identity, dict):
            return None
        try:
            return SourceFingerprint.from_dict(raw_identity)
        except ValueError:
            return None

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
                verification = missing_source(
                    str(mesh_info.get("path", "") or ""),
                    expected=self._expected_source_for_project_object(obj_state),
                    detail="source file is missing and no replacement was selected",
                )
                self._abort_project_source_load(
                    verification,
                    message=(
                        "프로젝트의 원본 메쉬를 찾지 못했습니다. 불완전한 상태로 작업을 "
                        "계속하지 않도록 프로젝트 로딩을 중단했습니다."
                    ),
                )
                return

        try:
            scale_factor = float(mesh_info.get("source_scale_factor", 1.0) or 1.0)
        except Exception:
            scale_factor = 1.0

        expected_source = self._expected_source_for_project_object(obj_state)
        source_info = mesh_info.get("source", {})
        saved_parse_format = (
            str(source_info.get("parse_format", "") or "").strip().lower()
            if isinstance(source_info, dict)
            else ""
        )
        source_format = saved_parse_format or (
            expected_source.format if expected_source is not None else None
        )
        self._start_async_load(mesh_path, scale_factor, source_format=source_format)

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

        alignment = obj_state.get("alignment", {})
        if not isinstance(alignment, dict):
            alignment = {}
        alignment_status = str(
            alignment.get("status", _ALIGNMENT_STATUS_MUTABLE_TRS)
            or _ALIGNMENT_STATUS_MUTABLE_TRS
        ).strip()
        if alignment_status not in {
            _ALIGNMENT_STATUS_MUTABLE_TRS,
            _ALIGNMENT_STATUS_UNVERIFIABLE,
            _ALIGNMENT_STATUS_BAKED_UNVERIFIABLE,
        }:
            alignment_status = _ALIGNMENT_STATUS_UNVERIFIABLE
        try:
            obj._amr_alignment_status = alignment_status
        except Exception:
            pass

        try:
            if alignment_status in {
                _ALIGNMENT_STATUS_UNVERIFIABLE,
                _ALIGNMENT_STATUS_BAKED_UNVERIFIABLE,
            }:
                # A v1 baked/fixed alignment cannot be reconstructed from the
                # raw source because neither baked vertices nor its matrix was
                # stored. Never expose the old fixed state as trustworthy.
                obj.fixed_state_valid = False
            else:
                obj.fixed_state_valid = bool(
                    tr.get("fixed_state_valid", getattr(obj, "fixed_state_valid", False))
                )
                obj.fixed_translation = f3(
                    tr.get("fixed_translation", getattr(obj, "fixed_translation", [0, 0, 0]))
                )
                obj.fixed_rotation = f3(
                    tr.get("fixed_rotation_deg", getattr(obj, "fixed_rotation", [0, 0, 0]))
                )
                obj.fixed_scale = float(
                    tr.get("fixed_scale", getattr(obj, "fixed_scale", 1.0)) or 1.0
                )
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

    def _snapshot_live_scene_for_project_swap(self) -> dict[str, Any]:
        vp = self.viewport
        viewport_fields: dict[str, Any] = {}
        for name in _VIEWPORT_PROJECT_SWAP_FIELDS:
            if not hasattr(vp, name):
                continue
            value = getattr(vp, name)
            if name == "_cut_section_pending_indices":
                try:
                    value = set(value)
                except Exception:
                    pass
            elif isinstance(value, np.ndarray):
                value = value.copy()
            viewport_fields[name] = value

        camera_state: dict[str, Any] = {}
        camera = getattr(vp, "camera", None)
        if camera is not None:
            try:
                camera_items = vars(camera).items()
            except TypeError:
                camera_items = ()
            for name, value in camera_items:
                try:
                    camera_state[name] = copy.deepcopy(value)
                except Exception:
                    camera_state[name] = value

        return {
            "objects": list(getattr(vp, "objects", []) or []),
            "selected_index": int(getattr(vp, "selected_index", -1)),
            "viewport_fields": viewport_fields,
            "camera_state": camera_state,
            "current_mesh": self.current_mesh,
            "current_filepath": self.current_filepath,
            "artifact_session": getattr(self, "_artifact_session", None),
        }

    def _restore_live_scene_after_failed_swap(
        self,
        snapshot: dict[str, Any],
        new_objects: list[Any],
    ) -> None:
        vp = self.viewport
        old_objects = list(snapshot.get("objects", []) or [])
        old_object_ids = {id(obj) for obj in old_objects}
        try:
            vp.makeCurrent()
        except Exception:
            pass
        cleaned_ids: set[int] = set()
        for obj in new_objects:
            obj_id = id(obj)
            if obj_id in cleaned_ids or obj_id in old_object_ids:
                continue
            cleaned_ids.add(obj_id)
            try:
                obj.cleanup()
            except Exception:
                pass

        vp.objects = old_objects
        vp.selected_index = int(snapshot.get("selected_index", -1))
        fields = snapshot.get("viewport_fields", {})
        if isinstance(fields, dict):
            for name, value in fields.items():
                try:
                    setattr(vp, name, value)
                except Exception:
                    pass
        # The framebuffer may already contain pixels from the failed candidate
        # scene.  Never pair those depths with a restored scene/camera frame.
        vp._amr_render_frame_snapshot = None
        vp._amr_render_frame_depth_signature = None
        vp._cached_render_frame = None
        camera = getattr(vp, "camera", None)
        camera_state = snapshot.get("camera_state", {})
        if camera is not None and isinstance(camera_state, dict):
            for name, value in camera_state.items():
                try:
                    setattr(camera, name, value)
                except Exception:
                    pass

        self.current_mesh = snapshot.get("current_mesh")
        current_filepath = snapshot.get("current_filepath")
        self.current_filepath = str(current_filepath) if current_filepath else None
        artifact_session = snapshot.get("artifact_session")
        self._artifact_session = (
            artifact_session if isinstance(artifact_session, ArtifactSession) else None
        )
        try:
            self.scene_panel.update_list(vp.objects, vp.selected_index)
            vp.selectionChanged.emit(vp.selected_index)
            vp.update()
            self.sync_transform_panel()
            self._sync_tile_panel()
        except Exception:
            pass

    def _finish_project_load(self) -> None:
        state = getattr(self, "_project_load_state", None)
        loaded_from_legacy = bool(getattr(self, "_project_load_from_legacy", False))
        pending_path = str(getattr(self, "_project_pending_path", "") or "").strip()
        staged = list(getattr(self, "_project_staged_objects", []) or [])
        has_legacy_bindings = bool(
            getattr(self, "_project_has_legacy_bindings", False)
        )

        expected_objects = state.get("objects", []) if isinstance(state, dict) else []
        if not isinstance(expected_objects, list) or len(staged) != len(expected_objects):
            self._project_load_active = False
            self._project_load_queue = []
            self._project_load_current = None
            self._project_load_state = None
            self._project_load_from_legacy = False
            self._discard_project_staging_and_restore_context()
            self.status_info.setText("프로젝트 staging 불완전 | 기존 scene 유지")
            QMessageBox.critical(
                self,
                "프로젝트 로딩 실패",
                "모든 원본이 staging되지 않아 기존 scene을 유지했습니다.",
            )
            return

        self._project_load_active = False
        self._project_load_queue = []
        self._project_load_current = None
        self._project_load_state = None
        self._project_load_from_legacy = False

        if not isinstance(state, dict):
            state = {}

        try:
            scene_snapshot = self._snapshot_live_scene_for_project_swap()
        except Exception as exc:
            _LOGGER.exception("Failed snapshotting live scene before project swap")
            self._discard_project_staging_and_restore_context()
            self.status_info.setText("기존 scene snapshot 실패 | 기존 scene 유지")
            QMessageBox.critical(
                self,
                "프로젝트 scene 교체 실패",
                "기존 scene을 안전하게 보존할 수 없어 프로젝트 교체를 중단했습니다."
                f"\n\n{type(exc).__name__}: {exc}",
            )
            return

        old_objects = list(scene_snapshot.get("objects", []) or [])
        new_objects: list[Any] = []
        try:
            # Detach the old scene without releasing its GPU resources.  The
            # detached objects remain the rollback target until every staged
            # object and global project setting has materialized successfully.
            # Publish legacy authority before add_mesh_object emits callbacks;
            # rollback restores the native session from scene_snapshot.
            self._artifact_session = None
            self.viewport.objects = []
            self.viewport.selected_index = -1
            self.viewport.clear_scene()
            self.current_mesh = None
            self.current_filepath = None

            for mesh_data, filepath, obj_state, _verification, _binding in staged:
                self.current_mesh = mesh_data
                self.current_filepath = filepath
                obj_name = str(obj_state.get("name", "")).strip() or Path(filepath).name
                self.viewport.add_mesh_object(mesh_data, name=obj_name)
                obj_loaded = self.viewport.selected_obj
                if obj_loaded is None:
                    raise RuntimeError("mesh materialization produced no scene object")
                new_objects.append(obj_loaded)
                try:
                    vertex_count = int(getattr(obj_loaded, "vertex_count", 0) or 0)
                    vbo_id = int(getattr(obj_loaded, "vbo_id", 0) or 0)
                except Exception:
                    vertex_count = 0
                    vbo_id = 0
                if vertex_count <= 0 or vbo_id <= 0:
                    if not self.viewport.update_vbo(obj_loaded):
                        raise RuntimeError(f"VBO upload failed for {obj_name!r}")
                    try:
                        vertex_count = int(getattr(obj_loaded, "vertex_count", 0) or 0)
                        vbo_id = int(getattr(obj_loaded, "vbo_id", 0) or 0)
                    except Exception:
                        vertex_count = 0
                        vbo_id = 0
                if vertex_count <= 0 or vbo_id <= 0:
                    raise RuntimeError(f"VBO upload produced invalid state for {obj_name!r}")
                self._apply_loaded_object_state(obj_loaded, obj_state)

            self._apply_project_state(state)
        except Exception as exc:
            _LOGGER.exception("Failed materializing staged project")
            old_object_ids = {id(obj) for obj in old_objects}
            for obj in list(getattr(self.viewport, "objects", []) or []):
                if id(obj) not in old_object_ids:
                    new_objects.append(obj)

            restore_error: Exception | None = None
            try:
                self._restore_live_scene_after_failed_swap(scene_snapshot, new_objects)
                restored_objects = list(getattr(self.viewport, "objects", []) or [])
                if len(restored_objects) != len(old_objects) or any(
                    actual is not expected
                    for actual, expected in zip(restored_objects, old_objects, strict=True)
                ):
                    raise RuntimeError("live scene rollback identity check failed")
            except Exception as restore_exc:
                restore_error = restore_exc
                _LOGGER.exception("Failed restoring live scene after project swap failure")

            self._discard_project_staging_and_restore_context()
            if restore_error is None:
                self.status_info.setText("project scene 교체 실패 | 기존 scene 복원")
                detail = "기존 scene을 복원했으며 현재 프로젝트는 계속 저장할 수 있습니다."
            else:
                self._project_load_failed = True
                self._current_project_path = None
                self._project_requires_save_as = False
                self._legacy_project_path = None
                self._project_has_legacy_bindings = False
                self.status_info.setText("project scene 복원 실패 | 저장 차단")
                detail = (
                    "기존 scene 복원까지 실패해 부분 상태 저장을 차단했습니다."
                    f"\n복원 오류: {type(restore_error).__name__}: {restore_error}"
                )
            QMessageBox.critical(
                self,
                "프로젝트 scene 교체 실패",
                "모든 원본 검증 후 scene을 교체하는 단계에서 실패했습니다. "
                f"{detail}\n\n{type(exc).__name__}: {exc}",
            )
            return

        # Commit point: the new scene is complete.  Only now may resources
        # owned by the detached previous scene be released.
        try:
            self.viewport.makeCurrent()
        except Exception:
            pass
        for old_obj in old_objects:
            try:
                old_obj.cleanup()
            except Exception:
                _LOGGER.warning("Previous SceneObject cleanup failed after project swap", exc_info=True)

        self._project_staged_objects = []
        self._project_previous_context = None
        self._project_pending_path = None
        self._project_load_failed = False
        self._current_project_path = None if loaded_from_legacy else (pending_path or None)
        self._flattened_cache.clear()
        self._flatten_recommendation_cache.clear()

        self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
        try:
            self.sync_transform_panel()
        except Exception:
            pass
        self._sync_tile_panel()

        try:
            if loaded_from_legacy or has_legacy_bindings:
                self.status_info.setText(
                    "프로젝트 로딩 완료 | 원본 SHA 기록, 구버전 기록 연결은 미검증"
                    + (" | 첫 저장은 새 v2 파일" if loaded_from_legacy else "")
                )
            else:
                self.status_info.setText("프로젝트 로딩 완료 | 원본 SHA-256 검증됨")
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
                    "영역 지정 모드 중지" if bool(getattr(vp, "roi_enabled", False)) else "영역 지정 모드 시작"
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
                "십자선 단면 모드 중지" if cross_enabled else "십자선 단면 모드 시작"
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
                "영역 지정 모드 중지" if roi_enabled else "영역 지정 모드 시작"
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
                self._set_flatten_method_combo_index(int(flat.get("method_index", self.flatten_panel.combo_method.currentIndex()) or 0))
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
            # Drag-and-drop follows the same explicit metadata/native document
            # boundary as File > Open; it must never append a ghost legacy mesh.
            self.open_file_path(filepath, prompt_unit=True)
            a0.acceptProposedAction()
    
    def load_mesh(self, filepath: str, scale_factor: float = 1.0):
        if self._native_artifact_mode():
            QMessageBox.warning(
                self,
                "메쉬 추가 차단",
                "ArtifactDocument는 한 문서당 하나의 검증된 원본만 소유합니다. "
                "새 원본은 파일 열기로 별도 문서로 여세요.",
            )
            self.status_info.setText("legacy 메쉬 추가 차단 | native 문서 유지")
            return False
        self._start_async_load(filepath, scale_factor)
        return True
    
    def _start_async_load(
        self,
        filepath: str,
        scale_factor: float,
        *,
        source_format: str | None = None,
        import_recipe: Mapping[str, object] | None = None,
        source_unit: str | None = None,
        artifact_ticket: ArtifactLoadTicket | None = None,
    ) -> bool:
        thread = getattr(self, "_mesh_load_thread", None)
        if thread is not None and thread.isRunning():
            QMessageBox.information(self, "로딩 중", "이미 다른 메쉬를 로딩 중입니다.")
            return False

        # The Open ticket is the authority for native imports.  Even an
        # internal caller which accidentally supplies a different mapping
        # cannot change the parser execution contract after ticket issuance.
        if isinstance(artifact_ticket, ArtifactLoadTicket):
            source_format = artifact_ticket.source_format
            import_recipe = artifact_ticket.import_recipe

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

        load_thread = MeshLoadThread(
            filepath=str(filepath),
            scale_factor=float(scale_factor),
            default_unit=str(
                source_unit
                or getattr(self.mesh_loader, "default_unit", DEFAULT_MESH_UNIT)
            ),
            source_format=source_format,
            import_recipe=import_recipe,
            capture_dependencies=(
                artifact_ticket.capture_dependencies
                if isinstance(artifact_ticket, ArtifactLoadTicket)
                else import_recipe is None
            ),
        )
        request_id = (
            artifact_ticket.id
            if isinstance(artifact_ticket, ArtifactLoadTicket)
            else f"mesh-load:{uuid.uuid4()}"
        )
        self._mesh_load_thread = load_thread
        self._mesh_load_request_id = request_id
        load_thread.loaded.connect(
            lambda mesh, path, owner=load_thread, rid=request_id, ticket=artifact_ticket: (
                self._dispatch_mesh_load_success(owner, rid, ticket, mesh, path)
            )
        )
        load_thread.failed.connect(
            lambda message, owner=load_thread, rid=request_id, ticket=artifact_ticket: (
                self._dispatch_mesh_load_failure(owner, rid, ticket, message)
            )
        )
        load_thread.finished.connect(
            lambda owner=load_thread, rid=request_id: self._dispatch_mesh_load_finished(
                owner,
                rid,
            )
        )
        load_thread.start()
        return True

    def _mesh_load_result_is_current(self, owner: QThread, request_id: str) -> bool:
        return (
            owner is getattr(self, "_mesh_load_thread", None)
            and request_id == getattr(self, "_mesh_load_request_id", None)
        )

    def _dispatch_mesh_load_success(
        self,
        owner: QThread,
        request_id: str,
        artifact_ticket: ArtifactLoadTicket | None,
        mesh_data: object,
        filepath: str,
    ) -> None:
        if not self._mesh_load_result_is_current(owner, request_id):
            _LOGGER.info("Discarded stale mesh-load success: %s", request_id)
            return
        if (
            artifact_ticket is not None
            and artifact_ticket is not getattr(self, "_artifact_load_ticket", None)
        ):
            _LOGGER.info("Discarded superseded Artifact Open success: %s", request_id)
            return
        self._on_mesh_load_thread_loaded(
            mesh_data,
            filepath,
            artifact_ticket=artifact_ticket,
        )

    def _dispatch_mesh_load_failure(
        self,
        owner: QThread,
        request_id: str,
        artifact_ticket: ArtifactLoadTicket | None,
        message: str,
    ) -> None:
        if not self._mesh_load_result_is_current(owner, request_id):
            _LOGGER.info("Discarded stale mesh-load failure: %s", request_id)
            return
        if (
            artifact_ticket is not None
            and artifact_ticket is not getattr(self, "_artifact_load_ticket", None)
        ):
            _LOGGER.info("Discarded superseded Artifact Open failure: %s", request_id)
            return
        self._on_mesh_load_thread_failed(message, artifact_ticket=artifact_ticket)

    def _dispatch_mesh_load_finished(self, owner: QThread, request_id: str) -> None:
        if not self._mesh_load_result_is_current(owner, request_id):
            try:
                owner.deleteLater()
            except Exception:
                pass
            _LOGGER.info("Ignored stale mesh-load finished signal: %s", request_id)
            return
        self._on_mesh_load_thread_finished(owner=owner, request_id=request_id)

    def _mark_artifact_authority_faulted(
        self,
        controller: ArtifactWorkbench,
        *,
        session: ArtifactSession | None,
        project_path: str | None,
        error: BaseException,
        operation_id: str | None,
    ) -> None:
        """Block all writes after an application/scene rollback becomes uncertain."""

        self._artifact_authority_faulted = True
        self._project_load_failed = True
        # A faulted target must never be overwritten by an ordinary Save.
        self._current_project_path = None
        try:
            controller.enter_faulted_state(
                session=session,
                project_path=project_path,
                error=error,
                operation_id=operation_id,
            )
        except Exception:
            _LOGGER.critical(
                "Artifact controller could not enter its fatal state",
                exc_info=True,
            )
            # Preserve a fail-closed controller even if the existing instance
            # was itself corrupted or replaced by a test double.
            fallback = ArtifactWorkbench()
            try:
                fallback.enter_faulted_state(
                    session=session,
                    project_path=project_path,
                    error=error,
                    operation_id=operation_id,
                )
            except Exception:
                fallback.enter_faulted_state(
                    session=None,
                    project_path=None,
                    error=error,
                    operation_id=operation_id,
                )
            self._artifact_workbench = fallback
        try:
            self.status_info.setText(_ARTIFACT_AUTHORITY_REOPEN_STATUS)
        except Exception:
            pass

    def _restore_artifact_authority_fault_status(self) -> bool:
        """Keep the reopen-required banner dominant after a fatal authority fault."""

        faulted = bool(getattr(self, "_artifact_authority_faulted", False))
        controller = getattr(self, "_artifact_workbench", None)
        if isinstance(controller, ArtifactWorkbench) and controller.snapshot.faulted:
            faulted = True
            self._artifact_authority_faulted = True
        if not faulted:
            return False
        try:
            self.status_info.setText(_ARTIFACT_AUTHORITY_REOPEN_STATUS)
        except Exception:
            pass
        return True

    def _report_artifact_authority_callback_failure(
        self,
        *,
        context: str,
        detail: str,
    ) -> bool:
        """Report callback fallout without replacing the fail-closed UI state."""

        if not self._restore_artifact_authority_fault_status():
            return False
        QMessageBox.critical(
            self,
            "문서 권위 복원 실패",
            "문서와 화면의 권위를 확정적으로 복원하지 못했습니다. "
            "저장·실측·내보내기를 차단했습니다. 검증된 원본 또는 "
            f"프로젝트를 다시 여세요.\n\n{context}\n{detail}",
        )
        return True

    def _publish_artifact_record_binding(
        self,
        session: ArtifactSession,
        transition: RecordBindingTransition,
        *,
        status_text: str,
    ) -> None:
        """Publish an append-only document update without rebuilding the live VBO."""

        controller = self._artifact_workbench_controller()
        old_session = self._artifact_session
        if (
            not isinstance(old_session, ArtifactSession)
            or transition.expected_session is not old_session
            or transition.candidate_session is not session
        ):
            raise StaleWorkflowOperationError(
                "record binding does not match the active GUI authority"
            )
        if old_session.projection_snapshot() != transition.expected_snapshot:
            raise StaleWorkflowOperationError(
                "record binding expected snapshot is stale"
            )
        if session.projection_snapshot() != transition.candidate_snapshot:
            raise ArtifactWorkbenchError(
                "record binding candidate snapshot is invalid"
            )
        objects = getattr(self.viewport, "objects", None)
        if not isinstance(objects, list) or len(objects) != 1:
            raise ArtifactWorkbenchError(
                "record binding requires exactly one live artifact object"
            )
        obj = objects[0]
        if (
            getattr(obj, "mesh", None) is not self.current_mesh
            or getattr(obj, "_amr_artifact_projection_snapshot", None)
            != transition.expected_snapshot
        ):
            raise StaleWorkflowOperationError(
                "live scene object does not match the record binding authority"
            )
        if not transition.expected_snapshot.has_same_render_projection(
            transition.candidate_snapshot
        ):
            raise ArtifactWorkbenchError(
                "record binding cannot change the live render projection"
            )

        old_project_path = self._current_project_path
        old_binding = transition.expected_snapshot
        activation: ProjectionActivation | None = None
        binding_changed = False
        finalize_attempted = False
        try:
            activation = controller.activate_record_binding(transition)
            self._artifact_session = session
            self._current_project_path = transition.project_path
            obj.compare_and_swap_artifact_binding(
                transition.expected_snapshot,
                transition.candidate_snapshot,
            )
            binding_changed = True
            finalize_attempted = True
            controller.finalize_record_binding(activation)
        except Exception as publication_error:
            if activation is None:
                raise
            restore_error: BaseException | None = None
            try:
                if binding_changed:
                    obj.compare_and_swap_artifact_binding(
                        transition.candidate_snapshot,
                        old_binding,
                    )
                self._artifact_session = old_session
                self._current_project_path = old_project_path
                if (
                    getattr(obj, "_amr_artifact_projection_snapshot", None)
                    != old_binding
                    or self._artifact_session is not old_session
                ):
                    raise RuntimeError(
                        "record binding rollback did not restore the exact GUI authority"
                    )
            except Exception as exc:
                restore_error = exc
                _LOGGER.critical("Record binding GUI rollback failed", exc_info=True)

            rollback_error: BaseException | None = None
            if restore_error is None:
                try:
                    controller.rollback_record_binding(
                        activation,
                        RuntimeError("record binding publication failed"),
                    )
                except Exception as exc:
                    rollback_error = exc
                    _LOGGER.critical(
                        "Record binding Workbench rollback failed",
                        exc_info=True,
                    )
            if (
                restore_error is not None
                or rollback_error is not None
                or finalize_attempted
            ):
                self._mark_artifact_authority_faulted(
                    controller,
                    session=None,
                    project_path=None,
                    error=restore_error or rollback_error or publication_error,
                    operation_id=transition.id,
                )
            raise

        snapshot = controller.snapshot
        if (
            snapshot.tentative
            or snapshot.session is not session
            or self._artifact_session is not session
            or getattr(obj, "_amr_artifact_projection_snapshot", None)
            != transition.candidate_snapshot
        ):
            self._mark_artifact_authority_faulted(
                controller,
                session=None,
                project_path=None,
                error=RuntimeError(
                    "record binding finalize did not prove coherent authority"
                ),
                operation_id=transition.id,
            )
            raise ArtifactWorkbenchError(
                "record binding finalize did not prove coherent authority"
            )

        try:
            self._sync_native_cutline_controls(reset_offset=False)
            self.status_unit.setText("단위: mm (canonical)")
            self.status_info.setText(status_text)
            self.viewport.update()
        except Exception:
            _LOGGER.debug("Record binding UI refresh failed", exc_info=True)

    def _publish_artifact_session_projection(
        self,
        session: ArtifactSession,
        *,
        project_path: str | None,
        fit_camera: bool,
        status_text: str,
        workflow_transition: ProjectionTransition | RecordBindingTransition | None = None,
        expected_new_record_ids: tuple[str, ...] | None = None,
    ) -> None:
        if isinstance(workflow_transition, RecordBindingTransition):
            if workflow_transition.candidate_session is not session:
                raise ArtifactWorkbenchError(
                    "record binding candidate does not match the published session"
                )
            self._publish_artifact_record_binding(
                session,
                workflow_transition,
                status_text=status_text,
            )
            return
        old_session = self._artifact_session
        controller = self._artifact_workbench_controller()
        if workflow_transition is None:
            if not isinstance(old_session, ArtifactSession):
                raise ArtifactWorkbenchError(
                    "native projection replacement requires a ticketed Open transition"
                )
            workflow_transition = controller.prepare_session_commit(
                old_session,
                session,
                expected_new_record_ids=expected_new_record_ids,
                project_path=project_path,
            )
        elif workflow_transition.candidate_session is not session:
            raise ArtifactWorkbenchError(
                "workflow transition candidate does not match the published session"
            )
        projection = workflow_transition.projection
        effective_project_path = workflow_transition.project_path
        source_asset, _geometry, _metadata = self._artifact_source_context(
            session.document
        )
        prepared = None
        activation: ProjectionActivation | None = None
        scene_snapshot = self._snapshot_live_scene_for_project_swap()
        old_project_path = self._current_project_path
        old_requires_save_as = self._project_requires_save_as
        old_legacy_path = self._legacy_project_path
        old_has_legacy = self._project_has_legacy_bindings
        old_project_load_failed = self._project_load_failed
        is_open_transition = workflow_transition.kind in {
            WorkflowTransitionKind.NEW_SOURCE,
            WorkflowTransitionKind.REOPEN_PROJECT,
        }
        authority_published = False
        finalize_attempted = False
        old_objects: list[Any] = []
        try:
            prepared = self.viewport.prepare_mesh_object(
                projection.mesh,
                str(getattr(source_asset, "original_name", "") or "Artifact"),
                artifact_binding=projection.snapshot,
            )
            self.viewport.validate_prepared_scene([prepared])

            # Publish the authority before scene notifications. Signal
            # callbacks must never observe a new projection with an old doc.
            activation = controller.activate_projection(workflow_transition)
            self._artifact_session = session
            self.current_mesh = projection.mesh
            self.current_filepath = session.resolved_source_path
            self._current_project_path = (
                str(effective_project_path) if effective_project_path else None
            )
            self._project_requires_save_as = False
            self._legacy_project_path = None
            self._project_has_legacy_bindings = False
            self._project_load_failed = (
                False if is_open_transition else old_project_load_failed
            )
            authority_published = True

            old_objects = self.viewport.swap_prepared_scene(
                [prepared],
                selected_index=0,
                fit_camera=fit_camera,
            )
            self._flattened_cache.clear()
            self._flatten_recommendation_cache.clear()
            assert activation is not None
            finalize_attempted = True
            controller.finalize_projection(activation)
            if is_open_transition:
                self._artifact_authority_faulted = False
        except Exception as publication_error:
            self._artifact_session = old_session
            self._current_project_path = old_project_path
            self._project_requires_save_as = old_requires_save_as
            self._legacy_project_path = old_legacy_path
            self._project_has_legacy_bindings = old_has_legacy
            self._project_load_failed = old_project_load_failed
            rollback_error: BaseException | None = None
            if activation is not None:
                try:
                    controller.rollback_projection(
                        activation,
                        RuntimeError("scene projection publication failed"),
                    )
                except Exception as exc:
                    rollback_error = exc
                    _LOGGER.critical(
                        "Artifact application authority rollback failed",
                        exc_info=True,
                    )
            if not authority_published:
                if prepared is not None:
                    self.viewport.cleanup_scene_objects([prepared])
                if rollback_error is not None or finalize_attempted:
                    self._mark_artifact_authority_faulted(
                        controller,
                        session=old_session,
                        project_path=old_project_path,
                        error=rollback_error or publication_error,
                        operation_id=(
                            workflow_transition.id
                            if workflow_transition is not None
                            else None
                        ),
                    )
                raise
            restore_error: BaseException | None = None
            try:
                self._restore_live_scene_after_failed_swap(
                    scene_snapshot,
                    [prepared] if prepared is not None else [],
                )
                expected_objects = list(scene_snapshot.get("objects", []) or [])
                restored_objects = list(getattr(self.viewport, "objects", []) or [])
                if (
                    len(restored_objects) != len(expected_objects)
                    or any(
                        restored is not expected
                        for restored, expected in zip(
                            restored_objects,
                            expected_objects,
                            strict=True,
                        )
                    )
                    or self._artifact_session is not old_session
                    or self.current_mesh is not scene_snapshot.get("current_mesh")
                ):
                    raise RuntimeError(
                        "scene restoration did not recover the exact previous authority"
                    )
            except Exception as exc:
                restore_error = exc
                _LOGGER.critical(
                    "Artifact live-scene restoration failed",
                    exc_info=True,
                )
            if rollback_error is not None or restore_error is not None or finalize_attempted:
                self._mark_artifact_authority_faulted(
                    controller,
                    session=old_session,
                    project_path=old_project_path,
                    error=rollback_error or restore_error or publication_error,
                    operation_id=workflow_transition.id,
                )
            raise

        self.viewport.cleanup_scene_objects(old_objects)
        try:
            self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
            self.sync_transform_panel()
            self._sync_tile_panel()
            self._sync_native_cutline_controls(reset_offset=False)
            preview_id = getattr(
                self.viewport,
                "native_vector_preview_record_id",
                None,
            )
            preview_record = (
                session.document.record_index.get(preview_id)
                if isinstance(preview_id, str)
                else None
            )
            if (
                self._native_vector_preview_document_id
                == session.document.document_id
                and preview_record is not None
                and self._native_vector_record_is_exportable(session, preview_record)
            ):
                self._preview_native_vector_record(session, preview_record.id)
            else:
                self._clear_native_vector_preview()
            self.status_unit.setText("단위: mm (canonical)")
            self.status_info.setText(status_text)
        except Exception:
            _LOGGER.debug("Artifact projection UI refresh failed", exc_info=True)

    def _finish_artifact_source_loaded(
        self,
        mesh_data,
        filepath: str,
        *,
        artifact_ticket: ArtifactLoadTicket | None = None,
    ) -> None:
        pending_document = self._artifact_pending_document
        pending_project_path = self._artifact_pending_project_path
        pending_metadata = self._artifact_pending_source_metadata
        try:
            transition: ProjectionTransition | None = None
            if isinstance(artifact_ticket, ArtifactLoadTicket):
                transition = self._artifact_workbench_controller().prepare_loaded_source(
                    artifact_ticket,
                    mesh_data,
                    resolved_source_path=str(filepath),
                )
                session = transition.candidate_session
                if artifact_ticket.document is not None:
                    status = (
                        f"ArtifactDocument 로딩 완료: "
                        f"{Path(pending_project_path or '').name} "
                        "| source·geometry·Align 검증됨"
                    )
                else:
                    status = (
                        f"원본 등록 완료: {Path(filepath).name} | canonical mm | "
                        "다음: 정치 preview 후 정치 확정"
                    )
            elif pending_document is not None:
                session = ArtifactSession.bind_loaded_document(
                    pending_document,
                    mesh_data,
                    resolved_source_path=str(filepath),
                )
                status = (
                    f"ArtifactDocument 로딩 완료: {Path(pending_project_path or '').name} "
                    "| source·geometry·Align 검증됨"
                )
            elif isinstance(pending_metadata, dict):
                session = ArtifactSession.create_from_source(
                    mesh_data,
                    resolved_source_path=str(filepath),
                    unit=str(pending_metadata.get("unit", "")),
                    axes=dict(pending_metadata.get("axes", {})),
                    handedness=str(pending_metadata.get("handedness", "unknown")),
                    software_version=APP_VERSION,
                    operator="local-user",
                )
                status = (
                    f"원본 등록 완료: {Path(filepath).name} | canonical mm | "
                    "다음: 정치 preview 후 정치 확정"
                )
            else:
                raise ArtifactSessionError("artifact load request has no document or metadata")

            self._publish_artifact_session_projection(
                session,
                project_path=pending_project_path,
                fit_camera=True,
                status_text=status,
                workflow_transition=transition,
            )
            self._clear_artifact_pending_load(cancel_workbench=False)
        except Exception as exc:
            if isinstance(artifact_ticket, ArtifactLoadTicket):
                try:
                    controller = self._artifact_workbench_controller()
                    if controller.snapshot.pending_load == artifact_ticket:
                        controller.fail_load(artifact_ticket, exc)
                except (ArtifactWorkbenchError, StaleWorkflowOperationError):
                    _LOGGER.debug("Artifact Open failure was stale", exc_info=True)
            self._clear_artifact_pending_load(cancel_workbench=False)
            self.status_info.setText("ArtifactDocument staging 실패 | 기존 scene 유지")
            QMessageBox.critical(
                self,
                "ArtifactDocument 로딩 실패",
                "원본·geometry·장면 검증 중 실패하여 기존 작업을 유지했습니다."
                f"\n\n{type(exc).__name__}: {exc}",
            )

    def _on_mesh_load_thread_loaded(
        self,
        mesh_data,
        filepath: str,
        *,
        artifact_ticket: ArtifactLoadTicket | None = None,
    ):
        try:
            dlg = getattr(self, "_mesh_load_dialog", None)
            if dlg is not None:
                dlg.setLabelText("장면에 추가하는 중...")
                QApplication.processEvents()

            if bool(getattr(self, "_artifact_load_active", False)):
                self._finish_artifact_source_loaded(
                    mesh_data,
                    filepath,
                    artifact_ticket=artifact_ticket,
                )
                return

            project_obj_state = (
                getattr(self, "_project_load_current", None)
                if getattr(self, "_project_load_active", False)
                else None
            )
            source_verification: SourceVerification | None = None
            source_binding_status = ""
            if isinstance(project_obj_state, dict):
                source_verification, binding_status = _verify_loaded_project_source(
                    mesh_data,
                    project_obj_state,
                    filepath,
                    migrated_from_v1=bool(
                        getattr(self, "_project_load_from_legacy", False)
                    ),
                )
                source_binding_status = binding_status
                if binding_status == _SOURCE_BINDING_LEGACY:
                    self._project_has_legacy_bindings = True
                self._last_source_verification = source_verification
                try:
                    setattr(mesh_data, "_amr_source_verification", source_verification)
                    setattr(mesh_data, "_amr_source_binding_status", binding_status)
                except Exception:
                    pass

                if source_verification.status not in {
                    SourceVerificationStatus.VERIFIED,
                    SourceVerificationStatus.LEGACY_UNVERIFIED,
                }:
                    expected = source_verification.expected
                    actual = source_verification.actual
                    expected_text = expected.id if expected is not None else "없음"
                    actual_text = actual.id if actual is not None else "읽을 수 없음"
                    self._abort_project_source_load(
                        source_verification,
                        message=(
                            "저장된 프로젝트와 선택한 원본 메쉬의 바이트가 일치하지 않아 "
                            "face ID와 기록 데이터를 적용하지 않았습니다.\n\n"
                            f"기대값: {expected_text}\n실제값: {actual_text}"
                        ),
                    )
                    return

                self._project_staged_objects.append(
                    (
                        mesh_data,
                        str(filepath),
                        project_obj_state,
                        source_verification,
                        source_binding_status,
                    )
                )
                self.status_info.setText(
                    f"원본 검증 완료, scene 교체 대기: {Path(filepath).name}"
                )
                return

            if self._native_artifact_mode():
                QMessageBox.critical(
                    self,
                    "메쉬 추가 차단",
                    "검증되지 않은 legacy 메쉬 로드가 native ArtifactDocument 장면에 "
                    "도달해 적용하지 않았습니다.",
                )
                self.status_info.setText("hybrid scene 차단 | native 문서 유지")
                return

            self.current_mesh = mesh_data
            self.current_filepath = filepath
            unit_s = str(getattr(mesh_data, "unit", "") or "").strip().lower()
            if unit_s not in ("mm", "cm", "m"):
                unit_s = str(getattr(self.mesh_loader, "default_unit", DEFAULT_MESH_UNIT) or DEFAULT_MESH_UNIT).strip().lower()
            self.status_unit.setText(f"단위: {unit_s}")

            # Normal file load vs project load(.amr)
            obj_name = Path(filepath).name
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
                if source_binding_status == _SOURCE_BINDING_LEGACY:
                    source_note = " | 원본 SHA 기록, 구버전 기록 연결 미검증"
                elif source_verification is not None and source_verification.relocated:
                    source_note = " | 이동된 동일 원본 SHA-256 확인"
                elif (
                    source_verification is not None
                    and source_verification.status is SourceVerificationStatus.LEGACY_UNVERIFIED
                ):
                    source_note = " | 구버전 원본 연결 미검증"
                else:
                    source_note = " | 원본 SHA-256 확인"
                self.status_info.setText(
                    f"프로젝트 로드됨: {obj_name}{source_note} | "
                    "다음: 1단계 정치에서 기준 시점을 확인하세요."
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

    def _on_mesh_load_thread_failed(
        self,
        message: str,
        *,
        artifact_ticket: ArtifactLoadTicket | None = None,
    ):
        dlg = getattr(self, "_mesh_load_dialog", None)
        if dlg is not None:
            dlg.close()
            self._mesh_load_dialog = None

        # Abort project staging if a mesh fails to load. The live scene has not
        # been touched yet, so discard only staged CPU data and restore the
        # previous save context.
        failed_artifact_staging = bool(getattr(self, "_artifact_load_active", False))
        if failed_artifact_staging:
            if isinstance(artifact_ticket, ArtifactLoadTicket):
                try:
                    controller = self._artifact_workbench_controller()
                    if controller.snapshot.pending_load == artifact_ticket:
                        controller.fail_load(artifact_ticket, message)
                except (ArtifactWorkbenchError, StaleWorkflowOperationError):
                    _LOGGER.debug("Artifact Open failure was stale", exc_info=True)
            self._clear_artifact_pending_load(cancel_workbench=False)
        failed_project_staging = bool(getattr(self, "_project_load_active", False))
        if failed_project_staging:
            self._project_load_active = False
            self._project_load_queue = []
            self._project_load_current = None
            self._project_load_state = None
            self._project_load_from_legacy = False
            self._discard_project_staging_and_restore_context()

        msg = f"파일 로드 실패:\n{message}"
        try:
            from src.core.logging_utils import format_exception_message

            msg = format_exception_message("파일 로드 실패:", message, log_path=_log_path)
        except Exception:
            pass

        QMessageBox.critical(self, "오류", msg)
        self.status_info.setText(
            "로드 실패 | 기존 scene 유지"
            if failed_project_staging or failed_artifact_staging
            else "로드 실패"
        )
        self.status_mesh.setText("")

    def _on_mesh_load_thread_finished(
        self,
        *,
        owner: QThread | None = None,
        request_id: str | None = None,
    ):
        thread = owner or getattr(self, "_mesh_load_thread", None)
        if owner is not None and not self._mesh_load_result_is_current(
            owner,
            str(request_id or ""),
        ):
            return
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                pass
        self._mesh_load_thread = None
        self._mesh_load_request_id = None
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
        on_cancel_requested: Callable[[], None] | None = None,
        on_shutdown_joined: Callable[[], None] | None = None,
        lock_dialog_until_finished: bool = False,
    ) -> bool:
        if bool(getattr(self, "_application_closing", False)):
            return False
        existing = getattr(self, "_task_thread", None)
        if existing is not None and existing.isRunning():
            QMessageBox.information(self, "작업 중", "이미 다른 작업이 진행 중입니다. 완료 후 다시 시도하세요.")
            return False

        dlg = QProgressDialog(
            str(label),
            "취소" if on_cancel_requested is not None else None,
            0,
            0,
            self,
        )
        dlg.setWindowTitle(str(title))
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        if on_cancel_requested is None:
            dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.show()

        try:
            self._status_task_begin(str(label), maximum=None, value=None)
        except Exception:
            pass

        self._task_dialog = dlg
        self._task_thread = thread
        self._refresh_native_save_indicator()
        dialog_close_guard = _TaskDialogCloseGuard()
        dialog_close_guard.waiting_for_worker = bool(lock_dialog_until_finished)
        try:
            dlg.installEventFilter(dialog_close_guard)
        except Exception:
            pass

        progress_ended = False
        cancel_requested = False

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
            dialog_close_guard.close_allowed = True
            previous_blocked = None
            try:
                previous_blocked = dlg.blockSignals(True)
            except Exception:
                pass
            try:
                dlg.close()
            except Exception:
                pass
            finally:
                if previous_blocked is not None:
                    try:
                        dlg.blockSignals(bool(previous_blocked))
                    except Exception:
                        pass
            if getattr(self, "_task_dialog", None) is dlg:
                self._task_dialog = None
            _end_progress()

        def _cleanup_thread():
            try:
                thread.deleteLater()
            except Exception:
                pass
            if getattr(self, "_task_thread", None) is thread:
                self._task_thread = None
                self._task_cancel_request = None
                self._task_close_dialog = None
                self._task_shutdown_verify = None
                self._refresh_native_save_indicator()

        def _default_failed(message: str):
            QMessageBox.critical(self, "오류", self._format_error_message("작업 실패:", message))

        def _request_cancel() -> None:
            nonlocal cancel_requested
            if cancel_requested or on_cancel_requested is None:
                return
            cancel_requested = True
            dialog_close_guard.waiting_for_worker = True
            try:
                on_cancel_requested()
            except Exception:
                _LOGGER.exception("Task cancellation callback failed: %s", title)
            try:
                dlg.setLabelText("취소 요청됨 · 안전한 계산 경계까지 기다리는 중...")
                dlg.setCancelButton(None)
                dlg.show()
            except Exception:
                pass

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

        def _callback_is_current() -> bool:
            return (
                getattr(self, "_task_thread", None) is thread
                and not bool(getattr(self, "_application_closing", False))
            )

        def _handle_done(result: object) -> None:
            if not _callback_is_current():
                return
            _close_dialog()
            _safe_invoke(on_done, result)

        def _handle_failed(message: str) -> None:
            if not _callback_is_current():
                return
            _close_dialog()
            _safe_invoke(on_failed or _default_failed, message)

        def _handle_finished() -> None:
            if getattr(self, "_task_thread", None) is not thread:
                try:
                    thread.deleteLater()
                except Exception:
                    pass
                return
            _close_dialog()
            _cleanup_thread()

        self._task_cancel_request = (
            _request_cancel if on_cancel_requested is not None else None
        )
        self._task_close_dialog = _close_dialog
        self._task_shutdown_verify = on_shutdown_joined
        thread.done.connect(_handle_done)
        thread.failed.connect(_handle_failed)
        thread.finished.connect(_handle_finished)
        if on_cancel_requested is not None:
            dlg.canceled.connect(_request_cancel)
        try:
            thread.start()
        except Exception:
            _close_dialog()
            _cleanup_thread()
            raise
        return True

    def on_mesh_loaded(self, mesh):
        obj = getattr(self.viewport, "selected_obj", None)
        selected_obj_id = int(id(obj)) if obj is not None else 0
        if int(self._flatten_method_target_obj_id or 0) != selected_obj_id:
            self._flatten_method_user_override = False
            self._flatten_method_target_obj_id = selected_obj_id
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
        self._refresh_native_save_indicator()
        
    def on_selection_changed(self, index):
        self.scene_panel.update_list(self.viewport.objects, index)
        self.sync_transform_panel()
        self.update_slice_range()
        current_obj = getattr(self.viewport, "selected_obj", None)
        selected_obj_id = int(id(current_obj)) if current_obj is not None else 0
        if int(self._flatten_method_target_obj_id or 0) != selected_obj_id:
            self._flatten_method_user_override = False
            self._flatten_method_target_obj_id = selected_obj_id
        try:
            self.flatten_panel.update_surface_assignment_counts(
                len(getattr(current_obj, "outer_face_indices", set()) or set()),
                len(getattr(current_obj, "inner_face_indices", set()) or set()),
                len(getattr(current_obj, "migu_face_indices", set()) or set()),
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
        self._refresh_native_save_indicator()

    def on_surface_assignment_changed(self, outer: int, inner: int, migu: int) -> None:
        try:
            self.flatten_panel.update_surface_assignment_counts(int(outer), int(inner), int(migu))
        except Exception:
            pass
        self._refresh_native_save_indicator()

    def on_face_selection_count_changed(self, count: int) -> None:
        try:
            self.selection_panel.update_selection_count(int(count))
        except Exception:
            pass
        try:
            panel = getattr(self, "section_panel", None)
            obj = getattr(self.viewport, "selected_obj", None)
            total = int(getattr(getattr(obj, "mesh", None), "n_faces", 0) or 0)
            if panel is not None and hasattr(panel, "label_native_tile_selection"):
                panel.label_native_tile_selection.setText(
                    f"현재 선택 {int(count):,}면 · 전체 {total:,}면"
                )
        except Exception:
            pass
        self._sync_tile_panel()
        self._sync_workflow_panel()
        self._refresh_native_save_indicator()

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
        self._refresh_native_save_indicator()

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
            self._sync_flatten_recommendation_for_current_selection(None)
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
        self._sync_flatten_recommendation_for_current_selection(state)

    def _on_flatten_method_changed(self, _index: int) -> None:
        if getattr(self, "_flatten_method_signal_guard", False):
            return
        self._flatten_method_user_override = True

    def _set_flatten_method_combo_index(self, index: int) -> None:
        panel = getattr(self, "flatten_panel", None)
        if panel is None:
            return
        try:
            safe_index = int(index)
        except Exception:
            safe_index = 0
        if safe_index < 0:
            safe_index = 0
        try:
            count = panel.combo_method.count()
        except Exception:
            return
        if count <= 0:
            return
        safe_index = max(0, min(safe_index, count - 1))
        self._flatten_method_signal_guard = True
        try:
            if panel.combo_method.currentIndex() != safe_index:
                panel.combo_method.setCurrentIndex(safe_index)
        finally:
            self._flatten_method_signal_guard = False

    def _set_flatten_method_by_text(self, method_text: str) -> bool:
        panel = getattr(self, "flatten_panel", None)
        if panel is None:
            return False
        method_text = str(method_text or "").strip()
        if not method_text:
            return False
        try:
            idx = panel.combo_method.findText(method_text)
        except Exception:
            idx = -1
        if idx < 0:
            return False
        if panel.combo_method.currentIndex() == idx:
            return True
        self._flatten_method_signal_guard = True
        try:
            panel.combo_method.setCurrentIndex(idx)
        finally:
            self._flatten_method_signal_guard = False
        return True

    def _flatten_recommendation_cache_key(self, mesh) -> tuple[Any, ...] | None:
        mesh_vertices = getattr(mesh, "vertices", None)
        if mesh_vertices is None:
            return None
        try:
            verts = np.asarray(mesh_vertices, dtype=np.float64).reshape(-1, 3)
        except Exception:
            return None
        if verts.size <= 0:
            return None
        finite_mask = np.isfinite(verts).all(axis=1)
        verts = verts[finite_mask]
        if verts.size <= 0:
            return None
        if verts.shape[0] == 0:
            return None
        bb_min = np.min(verts, axis=0)
        bb_max = np.max(verts, axis=0)
        span = bb_max - bb_min
        return (
            int(verts.shape[0]),
            int(verts.shape[1]),
            float(np.round(float(np.linalg.norm(bb_min)), 6)),
            float(np.round(float(np.linalg.norm(bb_max)), 6)),
            float(np.round(float(np.linalg.norm(span)), 6)),
        )

    @staticmethod
    def _flatten_recommendation_state_key(state: TileInterpretationState | None) -> tuple[Any, ...]:
        if state is None:
            return ()
        axis_hint = getattr(state, "axis_hint", None)
        try:
            axis_vec = tuple(
                np.round(
                    np.asarray(getattr(axis_hint, "vector_world", ()) or (), dtype=np.float64).reshape(-1)[:3],
                    4,
                ).tolist()
            )
        except Exception:
            axis_vec = ()
        accepted_sections = sum(
            1 for item in list(getattr(state, "section_observations", []) or []) if bool(getattr(item, "accepted", False))
        )
        analyzed_sections = sum(
            1 for item in list(getattr(state, "section_observations", []) or []) if int(getattr(item, "profile_point_count", 0) or 0) > 0
        )
        mandrel = getattr(state, "mandrel_fit", None)
        try:
            radius_world = float(getattr(mandrel, "radius_world", None))
        except Exception:
            radius_world = None
        return (
            str(getattr(state, "tile_class", "") or ""),
            axis_vec,
            int(accepted_sections),
            int(analyzed_sections),
            None if radius_world is None or not np.isfinite(radius_world) else float(np.round(radius_world, 4)),
            str(getattr(state, "record_view", "") or ""),
        )

    def _tile_flatten_recommendation(self, mesh, state: TileInterpretationState | None) -> dict[str, Any]:
        if mesh is None:
            return {
                "enabled": False,
                "method": _METHOD_NAME_ARAP,
                "reason": "",
                "confidence": 0.0,
                "applied_default_method": _METHOD_NAME_ARAP,
            }

        key = self._flatten_recommendation_cache_key(mesh)
        if key is None:
            return {
                "enabled": False,
                "method": _METHOD_NAME_ARAP,
                "reason": "",
                "confidence": 0.0,
                "applied_default_method": _METHOD_NAME_ARAP,
            }

        rec_key = id(mesh)
        cached = self._flatten_recommendation_cache.get(rec_key)
        cache_key = (key, self._flatten_recommendation_state_key(state))
        if cached is not None and isinstance(cached, tuple) and cached[0] == cache_key:
            cached_result = cached[1]
            if isinstance(cached_result, dict):
                return cached_result

        recommendation = recommend_flatten_mode(mesh, state, _METHOD_NAME_ARAP)
        result = recommendation.as_dict()
        self._flatten_recommendation_cache[rec_key] = (cache_key, result)
        return result

    def _sync_flatten_recommendation_for_current_selection(self, state: TileInterpretationState | None) -> None:
        panel = getattr(self, "flatten_panel", None)
        if panel is None:
            return
        obj = getattr(self.viewport, "selected_obj", None)
        if obj is None or getattr(obj, "mesh", None) is None:
            panel.clear_flatten_method_recommendation()
            self._flatten_method_user_override = False
            return

        rec = self._tile_flatten_recommendation(obj.mesh, state)
        if not rec.get("enabled", False):
            panel.clear_flatten_method_recommendation()
            return

        method_label = str(rec.get("method", _METHOD_NAME_SECTION))
        reason = str(rec.get("reason", "")).strip()
        alternatives = [
            str(item.get("label", "")).strip()
            for item in list(rec.get("alternatives", []) or [])
            if isinstance(item, dict) and str(item.get("label", "")).strip()
        ]
        fallback_chain = [
            str(item or "").strip()
            for item in list(rec.get("fallback_chain", []) or [])
            if str(item or "").strip()
        ]
        fallback_label_map = {
            "section": _METHOD_NAME_SECTION,
            "area": _METHOD_NAME_AREA,
            "cylinder": _METHOD_NAME_CYLINDER,
            "arap": _METHOD_NAME_ARAP,
            "lscm": _METHOD_NAME_LSCM,
        }
        fallback_labels = [fallback_label_map.get(item, item) for item in fallback_chain[1:]]
        fallback_hint = " → ".join(fallback_labels[:3])
        auto_applied = False
        if not bool(self._flatten_method_user_override):
            current_method = str(panel.combo_method.currentText() or "").strip()
            if current_method == _METHOD_NAME_ARAP:
                if self._set_flatten_method_by_text(method_label):
                    auto_applied = True

        panel.set_flatten_method_recommendation(
            method_label,
            reason,
            auto_applied=auto_applied,
            alternatives=alternatives,
            fallback_hint=fallback_hint,
        )

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
        method_text = str(resolved.get("method", "")).strip()
        if not method_text:
            rec = recommend_flatten_mode(getattr(obj, "mesh", None), state, None)
            resolved["method"] = str(rec.ui_label or _METHOD_NAME_SECTION)
        else:
            resolved["method"] = method_text
        if resolved["method"] == _METHOD_NAME_SECTION:
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

        if record_view in {"top", "bottom"}:
            resolved["tile_guided"] = True
            resolved["tile_record_view"] = record_view
            resolved["tile_record_strategy"] = str(getattr(state, "record_strategy", "") or "canonical_visible")
            if selected_face_ids is None:
                selected_face_ids = _surface_target_face_ids(obj, "selected")
            else:
                selected_face_ids = np.asarray(selected_face_ids, dtype=np.int32).reshape(-1)
            if selected_face_ids.size > 0:
                resolved["surface_target"] = "selected"

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
                self._refresh_native_save_indicator()
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
                    f"synthetic benchmark suite {report.case_count}건 저장 완료"
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
                self.status_info.setText("synthetic benchmark suite 저장 실패")
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
                        self.status_info.setText("저장된 슬롯 검토 시트 없음")
                        return
                    QMessageBox.information(
                        self,
                        "완료",
                        f"저장된 슬롯 검토 시트 {count}개를 저장했습니다.\n\n폴더: {output_dir}",
                    )
                    self.status_info.setText(f"슬롯 검토 시트 {count}개 저장 완료")

                def on_failed_export_saved_slots(message: str):
                    self.status_info.setText("슬롯 검토 시트 저장 실패")
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
                self._refresh_native_save_indicator()
                return
            else:
                return
        except Exception as e:
            QMessageBox.warning(
                self,
                "기와 해석",
                self._format_error_message("기와 해석 상태를 갱신하지 못했습니다:", f"{type(e).__name__}: {e}"),
            )
            self._refresh_native_save_indicator()
            return

        state.touch()
        setattr(obj, "tile_interpretation_state", state)
        self._sync_tile_panel()
        self._refresh_native_save_indicator()

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
            if self._native_artifact_mode():
                has_preview = (
                    not np.allclose(obj.translation, [0.0, 0.0, 0.0])
                    or not np.allclose(obj.rotation, [0.0, 0.0, 0.0])
                    or not np.isclose(float(obj.scale), 1.0)
                )
                self.trans_toolbar.btn_fixed.setEnabled(bool(has_preview))
                self.trans_toolbar.scale_spin.setEnabled(False)
            else:
                self.trans_toolbar.btn_fixed.setEnabled(
                    bool(getattr(obj, "fixed_state_valid", False))
                )
                self.trans_toolbar.scale_spin.setEnabled(True)
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

        native = self._native_artifact_mode()
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
        if native:
            obj.scale = 1.0
            if not np.isclose(float(self.trans_toolbar.scale_spin.value()), 1.0):
                self.trans_toolbar.scale_spin.blockSignals(True)
                self.trans_toolbar.scale_spin.setValue(1.0)
                self.trans_toolbar.scale_spin.blockSignals(False)
            try:
                self.status_info.setText(
                    "정치 preview | 저장 전 '정치 확정'으로 Align revision을 만드세요"
                )
            except Exception:
                pass
        else:
            obj.scale = self.trans_toolbar.scale_spin.value()
        self.viewport.update()
        self.viewport.meshTransformChanged.emit()

    def _capture_native_align_scene(
        self,
        obj: Any,
    ) -> tuple[ArtifactWorkbench, _NativeAlignSceneCapture]:
        """Capture a cheap GUI guard; the worker performs full mesh proof."""

        if bool(getattr(self, "_artifact_authority_faulted", False)):
            raise ArtifactSessionError(
                "artifact authority is faulted; reopen a verified source or project"
            )
        session = getattr(self, "_artifact_session", None)
        if not isinstance(session, ArtifactSession):
            raise ArtifactSessionError("no active ArtifactDocument session")
        objects = list(getattr(self.viewport, "objects", []) or [])
        if len(objects) != 1 or objects[0] is not obj:
            raise ArtifactSessionError(
                "native ArtifactDocument must own exactly one projected object"
            )
        binding = getattr(obj, "_amr_artifact_projection_snapshot", None)
        if not isinstance(binding, ArtifactProjectionSnapshot):
            raise ArtifactSessionError("native projection has no document binding")
        mesh = getattr(obj, "mesh", None)
        if not isinstance(mesh, MeshData):
            raise ArtifactSessionError("native projection has no MeshData")

        controller = self._artifact_workbench_controller()
        controller.require_stable_session(session)
        state = controller.snapshot
        if state.session is not session:
            raise StaleWorkflowOperationError(
                "native Align capture does not own Workbench authority"
            )

        document = session.document
        verified = session.verified_geometry
        expected_identity = (
            document.document_id,
            document.schema_version,
            verified.source_asset_id,
            verified.geometry_revision_id,
            document.active_source_metadata_revision_id,
            document.active_align_revision_id,
            verified.geometry_sha256,
            verified.geometry_hash_scope,
        )
        observed_identity = (
            binding.document_id,
            binding.document_schema_version,
            binding.source_asset_id,
            binding.geometry_revision_id,
            binding.source_metadata_revision_id,
            binding.align_revision_id,
            binding.geometry_sha256,
            binding.geometry_hash_scope,
        )
        if observed_identity != expected_identity:
            raise ArtifactSessionError(
                "native scene render binding is stale for the active session"
            )
        try:
            active_matrix = np.asarray(
                document.active_canonical_matrix(),
                dtype=np.float64,
            )
        except Exception as exc:
            raise ArtifactSessionError(
                f"active Align matrix is invalid: {exc}"
            ) from exc
        if not np.array_equal(active_matrix, binding.matrix):
            raise ArtifactSessionError(
                "native scene matrix is stale for the active Align revision"
            )

        def finite_triplet(value: object, *, label: str) -> tuple[float, float, float]:
            try:
                array = np.asarray(value, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ArtifactSessionError(f"{label} must be a finite 3-vector") from exc
            if array.shape != (3,) or not np.isfinite(array).all():
                raise ArtifactSessionError(f"{label} must be a finite 3-vector")
            return (float(array[0]), float(array[1]), float(array[2]))

        pivot_value = getattr(obj, "_amr_preview_pivot_mm", None)
        if pivot_value is None:
            raise ArtifactSessionError("native Align preview has no canonical pivot")
        translation = finite_triplet(obj.translation, label="Align translation")
        rotation = finite_triplet(obj.rotation, label="Align rotation")
        pivot = finite_triplet(pivot_value, label="Align pivot")
        scale = float(obj.scale)
        if not np.isfinite(scale):
            raise ArtifactSessionError("Align scale must be finite")

        return controller, _NativeAlignSceneCapture(
            scene_object=obj,
            session=session,
            binding=binding,
            mesh=mesh,
            translation_mm=translation,
            rotation_deg=rotation,
            scale=scale,
            pivot_mm=pivot,
            project_path=getattr(self, "_current_project_path", None),
            project_requires_save_as=bool(
                getattr(self, "_project_requires_save_as", False)
            ),
            legacy_project_path=getattr(self, "_legacy_project_path", None),
            state_version=state.state_version,
            authority_epoch=state.authority_epoch,
        )

    def _native_align_capture_is_current(
        self,
        controller: ArtifactWorkbench,
        capture: _NativeAlignSceneCapture,
    ) -> bool:
        """Recheck only captured identities and three small preview vectors."""

        try:
            state = controller.snapshot
            objects = list(getattr(self.viewport, "objects", []) or [])
            obj = capture.scene_object
            pivot = np.asarray(
                getattr(obj, "_amr_preview_pivot_mm", None),
                dtype=np.float64,
            ).reshape(-1)
            translation = np.asarray(obj.translation, dtype=np.float64).reshape(-1)
            rotation = np.asarray(obj.rotation, dtype=np.float64).reshape(-1)
            return bool(
                state.session is capture.session
                and state.state_version == capture.state_version
                and state.authority_epoch == capture.authority_epoch
                and state.pending_load is None
                and not state.tentative
                and not state.faulted
                and getattr(self, "_artifact_session", None) is capture.session
                and len(objects) == 1
                and objects[0] is obj
                and getattr(self.viewport, "selected_obj", None) is obj
                and getattr(obj, "mesh", None) is capture.mesh
                and getattr(obj, "_amr_artifact_projection_snapshot", None)
                == capture.binding
                and translation.shape == (3,)
                and rotation.shape == (3,)
                and pivot.shape == (3,)
                and np.array_equal(
                    translation,
                    np.asarray(capture.translation_mm, dtype=np.float64),
                )
                and np.array_equal(
                    rotation,
                    np.asarray(capture.rotation_deg, dtype=np.float64),
                )
                and np.array_equal(
                    pivot,
                    np.asarray(capture.pivot_mm, dtype=np.float64),
                )
                and float(obj.scale) == capture.scale
                and getattr(self, "_current_project_path", None)
                == capture.project_path
                and bool(getattr(self, "_project_requires_save_as", False))
                == capture.project_requires_save_as
                and getattr(self, "_legacy_project_path", None)
                == capture.legacy_project_path
                and not bool(getattr(self, "_project_load_failed", False))
                and not bool(getattr(self, "_artifact_authority_faulted", False))
            )
        except Exception:
            return False

    def _discard_native_align_result(self, *, action: str) -> None:
        self.status_info.setText(
            f"{action} 결과 폐기 | preview 또는 문서 권위가 변경됐습니다."
        )
        QMessageBox.warning(
            self,
            f"{action} 결과 폐기",
            "Align 준비 중 preview·선택 객체·문서 권위 중 하나가 변경되어 "
            "계산 결과를 현재 장면에 적용하지 않았습니다. 현재 상태에서 다시 실행하세요.",
        )

    def _start_native_align_commit(self, obj: Any) -> bool:
        try:
            controller, capture = self._capture_native_align_scene(obj)
        except Exception as exc:
            self.status_info.setText("정치 확정 준비 실패 | 기존 preview 유지")
            QMessageBox.critical(
                self,
                "정치 확정 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return False

        def on_done(value: object) -> None:
            if not self._native_align_capture_is_current(controller, capture):
                self._discard_native_align_result(action="정치 확정")
                return
            if value is None:
                self.status_info.setText(
                    "정치 preview 변경이 없어 현재 revision을 유지합니다"
                )
                return
            if not isinstance(value, ProjectionTransition) or (
                value.kind is not WorkflowTransitionKind.ALIGN_COMMIT
                or value.expected_session is not capture.session
                or value.base_state_version != capture.state_version
                or value.base_authority_epoch != capture.authority_epoch
            ):
                raise ArtifactWorkbenchError(
                    "Align worker returned a transition for different authority"
                )
            try:
                self._publish_artifact_session_projection(
                    value.candidate_session,
                    project_path=capture.project_path,
                    fit_camera=False,
                    status_text="정치 확정 | 새 Align revision 생성",
                    workflow_transition=value,
                )
            except Exception as exc:
                if self._report_artifact_authority_callback_failure(
                    context="Align revision 게시 중 권위 확인 실패",
                    detail=f"{type(exc).__name__}: {exc}",
                ):
                    return
                QMessageBox.critical(
                    self,
                    "정치 확정 실패",
                    "Align revision과 장면을 원자적으로 교체하지 못해 기존 preview를 "
                    f"유지했습니다.\n\n{type(exc).__name__}: {exc}",
                )

        def on_failed(message: str) -> None:
            if not self._native_align_capture_is_current(controller, capture):
                self.status_info.setText(
                    "정치 확정 결과 폐기 | 준비 중 문서 권위가 변경됐습니다."
                )
                return
            self.status_info.setText("정치 확정 실패 | 기존 preview 유지")
            QMessageBox.critical(
                self,
                "정치 확정 실패",
                self._format_error_message(
                    "Align revision 준비 중 오류가 발생했습니다:",
                    message,
                ),
            )

        try:
            return bool(
                self._start_task(
                    title="정치 확정",
                    label="원본 geometry를 검증하고 Align revision을 준비하는 중...",
                    thread=TaskThread(
                        "prepare_native_align_commit",
                        lambda: controller.prepare_align_commit(
                            translation_mm=capture.translation_mm,
                            rotation_deg=capture.rotation_deg,
                            scale=capture.scale,
                            pivot_mm=capture.pivot_mm,
                            operator="local-user",
                        ),
                    ),
                    on_done=on_done,
                    on_failed=on_failed,
                    lock_dialog_until_finished=True,
                )
            )
        except Exception as exc:
            self.status_info.setText("정치 확정 작업 시작 실패")
            QMessageBox.critical(
                self,
                "정치 확정 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return False

    def _start_native_parent_align_activation(
        self,
        controller: ArtifactWorkbench,
        capture: _NativeAlignSceneCapture,
    ) -> bool:
        active_id = capture.session.document.active_align_revision_id
        parent_id = (
            capture.session.document.align_revision_index[active_id].parent_id
            if active_id is not None
            else None
        )
        if parent_id is None:
            self.status_info.setText(
                "Undo할 이전 Align revision이 없습니다: active revision has no parent"
            )
            return False

        def on_done(value: object) -> None:
            if not self._native_align_capture_is_current(controller, capture):
                self._discard_native_align_result(action="Align Undo")
                return
            if not isinstance(value, ProjectionTransition) or (
                value.kind is not WorkflowTransitionKind.ALIGN_ACTIVATE_PARENT
                or value.expected_session is not capture.session
                or value.base_state_version != capture.state_version
                or value.base_authority_epoch != capture.authority_epoch
            ):
                raise ArtifactWorkbenchError(
                    "Align Undo worker returned a transition for different authority"
                )
            try:
                self._publish_artifact_session_projection(
                    value.candidate_session,
                    project_path=capture.project_path,
                    fit_camera=False,
                    status_text="이전 Align revision 활성화",
                    workflow_transition=value,
                )
            except Exception as exc:
                if self._report_artifact_authority_callback_failure(
                    context="이전 Align revision 게시 중 권위 확인 실패",
                    detail=f"{type(exc).__name__}: {exc}",
                ):
                    return
                QMessageBox.critical(
                    self,
                    "Align Undo 실패",
                    f"기존 revision을 유지했습니다.\n\n{type(exc).__name__}: {exc}",
                )

        def on_failed(message: str) -> None:
            if not self._native_align_capture_is_current(controller, capture):
                self.status_info.setText(
                    "Align Undo 결과 폐기 | 준비 중 문서 권위가 변경됐습니다."
                )
                return
            self.status_info.setText("Align Undo 실패 | 기존 revision 유지")
            QMessageBox.critical(
                self,
                "Align Undo 실패",
                self._format_error_message(
                    "이전 Align revision 준비 중 오류가 발생했습니다:",
                    message,
                ),
            )

        try:
            return bool(
                self._start_task(
                    title="이전 Align 활성화",
                    label="원본 geometry를 검증하고 이전 Align revision을 준비하는 중...",
                    thread=TaskThread(
                        "prepare_native_parent_align",
                        controller.prepare_activate_parent_align,
                    ),
                    on_done=on_done,
                    on_failed=on_failed,
                    lock_dialog_until_finished=True,
                )
            )
        except Exception as exc:
            self.status_info.setText("Align Undo 작업 시작 실패")
            QMessageBox.critical(
                self,
                "Align Undo 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return False

    def on_bake_all_clicked(self):
        """현재 변환을 메쉬에 영구 정착 (정치 신청)"""
        obj = self.viewport.selected_obj
        if not obj:
            return

        if self._native_artifact_mode():
            self._start_native_align_commit(obj)
            return

        self.viewport.bake_object_transform(obj)
        self.sync_transform_panel() # 툴바 값 리셋됨
        self.viewport.status_info = f"{obj.name} 정치(Bake) 완료. 변환값이 초기화되었습니다."
        self.viewport.update()

    def undo_last_action(self):
        """Undo a native preview/revision, otherwise delegate to legacy undo."""

        obj = self.viewport.selected_obj
        if not self._native_artifact_mode():
            self.viewport.undo()
            return
        if obj is None:
            return
        try:
            controller, capture = self._capture_native_align_scene(obj)
            has_preview = (
                not np.allclose(
                    capture.translation_mm,
                    [0.0, 0.0, 0.0],
                    rtol=0.0,
                    atol=1e-12,
                )
                or not np.allclose(
                    capture.rotation_deg,
                    [0.0, 0.0, 0.0],
                    rtol=0.0,
                    atol=1e-12,
                )
                or not np.isclose(capture.scale, 1.0, rtol=0.0, atol=1e-12)
            )
            if has_preview:
                self.reset_transform()
                self.status_info.setText("정치 preview 취소 | 확정 revision 유지")
                return
            self._start_native_parent_align_activation(
                controller,
                capture,
            )
        except (ArtifactSessionError, ArtifactWorkbenchError) as exc:
            self.status_info.setText(f"Undo할 이전 Align revision이 없습니다: {exc}")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Align Undo 실패",
                f"기존 revision을 유지했습니다.\n\n{type(exc).__name__}: {exc}",
            )

    def restore_fixed_state(self):
        """정치 확정 이후의 고정 상태로 복귀"""
        obj = self.viewport.selected_obj
        if not obj:
            return

        if self._native_artifact_mode():
            self.reset_transform()
            self.status_info.setText("정치 preview 초기화 | 확정 Align revision 유지")
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
            self.status_info.setText("X-Ray 모드: 선택된 메쉬를 투명 표시" if enabled else "X-Ray 모드 종료")
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
        if self._reject_native_unported_mutation("기준평면 맞추기"):
            return
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
            obj._amr_has_unpersisted_bake = True
            obj._amr_alignment_status = _ALIGNMENT_STATUS_BAKED_UNVERIFIABLE
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
        self.viewport.status_info = "기준평면 맞추기 완료 (최저점 Z=0)"
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
            self.viewport.status_info = f"표면 지정 대상: {target} (경계(면적+자석)로 시작)"
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
                    f"경계(면적+자석) [{target}]: 좌클릭=점 추가(자석 스냅), 드래그=회전/시점, "
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
                    "검증된 실측·기와 전개 패널을 열었습니다. native 문서는 record 기반 명령과 1:1 export를 사용합니다."
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
                    f"실시간 단면 모드 ON (Z={z_next:.2f}cm): "
                    "Ctrl+휠/[, .]=스캔, C=촬영"
                )
            else:
                self.viewport.status_info = "실시간 단면 모드 OFF"
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
                    f"수동 보조 분리({mode_txt}/{method}, {mode}, {mapping}): "
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
                    f"표면 라벨 자동 적용({method_used}): outer {len(obj.outer_face_indices):,} / inner {len(obj.inner_face_indices):,} / migu {len(obj.migu_face_indices):,} (현재 메쉬에 저장됨)"
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

                    # Rotation matrix (local -> world), shared with the renderer.
                    rot_deg = np.asarray(getattr(obj, "rotation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
                    if rot_deg.size < 3:
                        rot_deg = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                    rot_mat = scene_rotation_matrix(rot_deg[:3])

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
                    f"미구 자동 감지({mode_desc}): migu {len(obj.migu_face_indices):,} faces "
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

        local_to_world = scene_trs_matrix(
            [0.0, 0.0, 0.0] if translation is None else translation,
            [0.0, 0.0, 0.0] if rotation is None else rotation,
            scale,
        )
        vertices = transform_points(base.vertices, local_to_world)

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

        method = str(options.get('method', _METHOD_NAME_ARAP))
        iterations = int(options.get('iterations', 30))
        boundary_type = str(options.get('boundary', 'free'))
        initial = str(options.get('initial', 'lscm'))
        distortion = float(options.get("distortion", 0.5))
        radius_mm = float(options.get("radius", 0.0))
        direction = options.get("direction_override", options.get("direction", "auto"))

        def normalize_method(text: str) -> str:
            t = str(text or "").strip().lower()
            source = str(text or "")
            if ("저왜곡" in source) or ("arap" in t):
                return "arap"
            if ("각도 보존" in source) or ("lscm" in t):
                return "lscm"
            if ("기록면 기반" in source) or ("면적" in source) or ("area" in t):
                return "area"
            if ("기와 추천" in source) or ("단면" in source) or ("기와" in source) or ("section" in t) or ("tile" in t):
                return "section"
            if ("곡면 추적" in source) or ("원통" in source) or ("cyl" in t):
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
        self.status_info.setText(f"기록면 전개 중{status_target}{strategy_suffix}...")
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
            self.status_info.setText("기록면 전개 실패")
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

        status_prefix = (
            "기록면 전개 완료 · 크기 경고" if size_warning else "기록면 전개 완료"
        )
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
        self.status_info.setText("기록면 전개 실패")
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

    def _reject_native_legacy_profile_export(self) -> bool:
        if not self._native_artifact_mode():
            return False
        message = (
            "화면 캡처/OpenCV 기반 2D 도면은 ArtifactDocument의 1:1 측정 산출물이 "
            "아닙니다. 검증된 Cutline/Outline record를 만든 뒤 '최근 검증 벡터 1:1 SVG "
            "내보내기'를 사용하세요."
        )
        self.status_info.setText("legacy 화면 SVG 차단 | 검증 벡터 내보내기 사용")
        QMessageBox.warning(self, "검증되지 않은 내보내기 차단", message)
        return True

    def _reject_native_legacy_surface_export(self, export_type: object) -> bool:
        if not self._native_artifact_mode() or export_type not in {
            "flat_svg",
            "review_sheet",
            "rubbing",
            "rubbing_digital",
            "rubbing_view_cyl",
        }:
            return False
        message = (
            "기존 펼침·화면·SurfaceVisualizer 기반 출력은 ArtifactDocument의 "
            "재현 가능한 측정 산출물이 아닙니다. '탁본 계산 · 기록' 후 "
            "'최근 검증 탁본 1:1 PNG 패키지 내보내기'를 사용하세요."
        )
        self.status_info.setText(
            "legacy 탁본/펼침 출력 차단 | 검증 Digital Rubbing 사용"
        )
        QMessageBox.warning(self, "검증되지 않은 탁본 출력 차단", message)
        return True

    def on_export_requested(self, data):
        """내보내기 요청 처리"""
        export_type = data.get('type')
        if self._reject_native_legacy_surface_export(export_type):
            return
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
            if self._reject_native_legacy_profile_export():
                return
            self.export_2d_profile(data.get('view'))
            return

        if export_type == "profile_2d_package":
            if self._reject_native_legacy_profile_export():
                return
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
                self.status_info.setText(f"저장 완료: {Path(filepath).name}")

            def on_failed(message: str):
                self.status_info.setText("저장 실패")
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
                self.status_info.setText(f"저장 완료: {Path(filepath).name}")

            def on_failed(message: str):
                self.status_info.setText("저장 실패")
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
        if self._reject_native_legacy_profile_export():
            return
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
            self.status_info.setText("저장 실패")
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
        if self._reject_native_legacy_profile_export():
            return
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
        self.status_info.setText(f"패키지 저장 완료: {package_dir.name}")

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
        self.status_info.setText("패키지 내보내기 실패")

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
        self.status_info.setText("변환 초기화 + 뷰 맞춤 완료")
    
    def bake_and_center(self):
        """정치: 현재 회전을 메쉬 버텍스에 영구 적용하고 변환 리셋"""
        if self._reject_native_unported_mutation("정치 후 중심 이동"):
            return
        obj = self.viewport.selected_obj
        if obj is None:
            return
        
        # OpenGL 렌더링과 동일한 중앙 회전 계약.
        rotation_matrix = scene_rotation_matrix(obj.rotation)
        
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
        obj._amr_has_unpersisted_bake = True
        obj._amr_alignment_status = _ALIGNMENT_STATUS_BAKED_UNVERIFIABLE
        
        self.sync_transform_panel()
        self.viewport.update()
        self.status_info.setText("정치 완료 - 회전이 메쉬에 적용됨")
    
    def return_to_origin(self):
        """카메라를 원점으로 이동"""
        self.viewport.camera.center = np.array([0.0, 0.0, 0.0])
        self.viewport.camera.pan_offset = np.array([0.0, 0.0, 0.0])
        self.viewport.update()
        self.status_info.setText("카메라 원점 복귀")
            
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
            self.status_info.setText("곡률 측정 모드: 메쉬 위를 클릭하여 점을 찍으세요")
        else:
            self.status_info.setText("곡률 측정 모드 종료")
    
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
            f"원호 #{arc_count} 생성됨 (월드 고정): 반지름 = {radius_mm:.1f} mm"
        )
        self._refresh_native_save_indicator()
    
    def clear_curvature_points(self):
        """곡률 측정용 점 초기화"""
        self.viewport.clear_curvature_picks()
        self.status_info.setText("측정 점 초기화됨")
        self._refresh_native_save_indicator()
    
    def clear_all_arcs(self):
        """선택된 객체의 모든 원호 삭제"""
        obj = self.viewport.selected_obj
        if obj and obj.fitted_arcs:
            count = len(obj.fitted_arcs)
            obj.fitted_arcs = []
            self.scene_panel.update_list(self.viewport.objects, self.viewport.selected_index)
            self.viewport.update()
            self.status_info.setText(f"{count}개 원호 삭제됨")
            self._refresh_native_save_indicator()
    
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
        self._refresh_native_save_indicator()


    def toggle_measure_mode(self, enabled: bool) -> None:
        """치수(거리/지름) 측정 모드 토글"""
        self._invalidate_surface_pick_requests()
        if enabled and self.viewport.selected_obj is None:
            QMessageBox.warning(self, "경고", "먼저 메쉬를 선택하세요.")
            try:
                self.measure_panel.set_measure_checked(False)
            except Exception:
                pass
            self._disable_measure_mode()
            self.viewport.update()
            return

        if enabled and self._native_artifact_mode():
            try:
                obj = self.viewport.selected_obj
                session = self._require_native_measurement_session(obj)
                # Capture all GUI-only fail-closed guards now.  Canonical mesh
                # equality is intentionally deferred to a worker.
                self._capture_native_scene_preflight(session)
            except Exception as exc:
                self.measure_panel.set_measure_checked(False)
                self._disable_measure_mode()
                self.status_info.setText("검증 치수 측정 시작 실패 | 기존 문서 유지")
                QMessageBox.warning(
                    self,
                    "검증 치수 측정 시작 실패",
                    "원본·단위·Align이 확정된 단일 native 문서가 필요합니다.\n\n"
                    f"{type(exc).__name__}: {exc}",
                )
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

            # X-ray does not write authoritative object depth.  Always leave
            # it before a native pick instead of accepting a missing/stale hit.
            try:
                if bool(getattr(self.viewport, "xray_mode", False)):
                    self.viewport.xray_mode = False
                    self.trans_toolbar.btn_xray.blockSignals(True)
                    self.trans_toolbar.btn_xray.setChecked(False)
                    self.trans_toolbar.btn_xray.blockSignals(False)
            except Exception:
                pass

            try:
                self.viewport.clear_measure_picks()
                self.measure_panel.set_points_count(0)
            except Exception:
                pass

            self.viewport.picking_mode = "measure"
            self.status_info.setText(
                "검증 치수 측정: Shift+클릭 · source triangle+barycentric anchor"
                if self._native_artifact_mode()
                else "검토용 치수 측정: Shift+클릭으로 점을 찍으세요."
            )
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
            self.status_info.setText("치수 측정 모드 종료")

        self.viewport.update()
        self._refresh_native_save_indicator()

    def on_measure_mode_changed(self, mode: str) -> None:
        self._invalidate_surface_pick_requests()
        try:
            self.viewport.clear_measure_picks()
            self.measure_panel.set_points_count(0)
            self.viewport.update()
        except Exception:
            pass

        if str(mode) == "diameter":
            self.status_info.setText("원 맞춤 지름: 표면 anchor 3~64개 선택 후 '지름 계산 · 기록'.")
        else:
            self.status_info.setText("검증 거리: 표면 anchor 2개 선택 시 자동 계산·기록.")
        self._refresh_native_save_indicator()

    def on_surface_anchor_pick_requested(self, screen_x: int, screen_y: int) -> None:
        """Resolve one native exact-frame click to a durable source anchor."""

        if not self._native_artifact_mode():
            return
        try:
            obj = self.viewport.selected_obj
            session = self._require_native_measurement_session(obj)
            observation = self.viewport.capture_surface_anchor_observation(
                int(screen_x),
                int(screen_y),
            )
            if observation is None:
                self.status_info.setText(
                    "표면 anchor 없음 | 메쉬 위를 Shift+클릭하세요"
                )
                return
            if not isinstance(observation, SurfaceAnchorObservation):
                raise ArtifactWorkbenchError("surface pick observation is invalid")
            if observation.projection_snapshot != session.projection_snapshot():
                raise ArtifactWorkbenchError(
                    "surface pick frame does not match the active projection"
                )
        except Exception as exc:
            self.status_info.setText("표면 anchor 캡처 실패 | 기록 변경 없음")
            QMessageBox.warning(
                self,
                "표면 anchor 캡처 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return

        cancellation = Event()
        captured_obj = obj
        pick_generation = int(getattr(self, "_surface_pick_generation", 0))

        def resolve_anchor() -> tuple[
            dict[str, Any],
            np.ndarray,
            ArtifactProjectionSnapshot,
        ]:
            projection = session.materialize()
            if projection.snapshot != observation.projection_snapshot:
                raise ArtifactWorkbenchError(
                    "surface projection changed before anchor resolution"
                )
            anchor = resolve_surface_anchor_from_ray(
                projection.mesh.vertices,
                projection.mesh.faces,
                source_faces=session.source_mesh.faces,
                ray_origin_world_mm=observation.ray_origin_world_mm,
                ray_direction_world=observation.ray_direction_world,
                depth_point_world_mm=observation.depth_point_world_mm,
                pixel_footprint_um=observation.pixel_footprint_um,
                depth_search_offset_px=observation.depth_search_offset_px,
                coordinate_grid_um=1,
                cancellation_probe=cancellation.is_set,
            )
            row = np.asarray(anchor["face_vertex_indices"], dtype=np.int64)
            weights = np.asarray(
                anchor["barycentric_numerators"], dtype=np.float64
            ) / float(BARYCENTRIC_DENOMINATOR)
            point = np.einsum(
                "i,ij->j",
                weights,
                np.asarray(projection.mesh.vertices, dtype=np.float64)[row],
            )
            return anchor, np.asarray(point, dtype=np.float64), projection.snapshot

        def on_done(result: object) -> None:
            try:
                anchor, point, snapshot = result  # type: ignore[misc]
                if not isinstance(anchor, dict) or not isinstance(
                    snapshot, ArtifactProjectionSnapshot
                ):
                    raise ArtifactWorkbenchError(
                        "surface anchor worker result is invalid"
                    )
                current = getattr(self, "_artifact_session", None)
                live_obj = self.viewport.selected_obj
                if not isinstance(current, ArtifactSession):
                    raise ArtifactWorkbenchError("native document was closed")
                current_snapshot = current.projection_snapshot()
                live_binding = getattr(
                    live_obj, "_amr_artifact_projection_snapshot", None
                )
                if (
                    live_obj is not captured_obj
                    or not isinstance(live_binding, ArtifactProjectionSnapshot)
                    or current_snapshot.render_key != snapshot.render_key
                    or live_binding.render_key != snapshot.render_key
                    or self.viewport.picking_mode != "measure"
                    or int(getattr(self, "_surface_pick_generation", 0))
                    != pick_generation
                ):
                    raise ArtifactWorkbenchError(
                        "surface anchor became stale before publication"
                    )
                anchors = getattr(self.viewport, "measure_picked_anchors", None)
                points = getattr(self.viewport, "measure_picked_points", None)
                if not isinstance(anchors, list) or not isinstance(points, list):
                    raise ArtifactWorkbenchError("surface pick state is invalid")
                maximum_anchors = 2 if self.measure_panel.mode == "distance" else 64
                if len(anchors) >= maximum_anchors:
                    raise ArtifactWorkbenchError(
                        f"surface measurement accepts at most {maximum_anchors} anchors"
                    )
                anchors.append(dict(anchor))
                points.append(np.asarray(point, dtype=np.float64).reshape(3))
                self.measure_panel.set_points_count(len(anchors))
                self.viewport.update()
                self._refresh_native_save_indicator()
                self.status_info.setText(
                    f"표면 anchor {len(anchors)}개 | face {int(anchor['face_index'])}"
                )
                if self.measure_panel.mode == "distance" and len(anchors) == 2:
                    captured_anchors = tuple(dict(value) for value in anchors)
                    QTimer.singleShot(
                        0,
                        lambda: self._start_native_surface_measurement(
                            "distance",
                            captured_anchors,
                        ),
                    )
            except Exception as exc:
                self.status_info.setText("표면 anchor 폐기 | 현재 문서 유지")
                QMessageBox.warning(
                    self,
                    "표면 anchor 폐기",
                    f"{type(exc).__name__}: {exc}",
                )

        def on_failed(message: str) -> None:
            if (
                int(getattr(self, "_surface_pick_generation", 0))
                != pick_generation
            ):
                self.status_info.setText("표면 anchor 폐기 | 현재 점 목록 유지")
                return
            if cancellation.is_set():
                self.status_info.setText("표면 anchor 계산 취소 | 기록 변경 없음")
                return
            self.status_info.setText("표면 anchor 계산 실패 | 기록 변경 없음")
            QMessageBox.warning(
                self,
                "표면 anchor 계산 실패",
                str(message),
            )

        def cancel_anchor() -> None:
            cancellation.set()
            self._invalidate_surface_pick_requests()

        self.status_info.setText("전체 삼각형에서 표면 anchor 확인 중...")
        started = self._start_task(
            title="표면 anchor",
            label="렌더 깊이와 원본 삼각형 교차 검증 중...",
            thread=TaskThread("native_surface_anchor", resolve_anchor),
            on_done=on_done,
            on_failed=on_failed,
            on_cancel_requested=cancel_anchor,
        )
        if not started:
            cancellation.set()

    def _start_native_surface_measurement(
        self,
        kind: str,
        anchors: tuple[dict[str, Any], ...],
    ) -> None:
        """Compute and publish distance/diameter from captured source anchors."""

        key = str(kind).strip().lower()
        label = "검증 거리" if key == "distance" else "검증 원 맞춤 지름"
        try:
            obj = self.viewport.selected_obj
            session = self._require_native_measurement_session(obj)
            live_anchors = tuple(
                dict(value)
                for value in (
                    getattr(self.viewport, "measure_picked_anchors", []) or []
                )
            )
            if live_anchors != anchors:
                raise ArtifactWorkbenchError(
                    "surface anchors changed before measurement began"
                )
            preflight = self._capture_native_scene_preflight(
                session,
                allow_surface_measurement_picks=True,
            )
            controller = self._artifact_measurement_controller()
            if key == "distance":
                if len(anchors) != 2:
                    raise ArtifactWorkbenchError(
                        "surface distance requires exactly two anchors"
                    )
                work_item = controller.begin_surface_distance(
                    anchors,
                    coordinate_grid_um=1,
                    record_id=f"record:surface-distance:{uuid.uuid4()}",
                    created_at=self._utc_seconds_now(),
                    operator="local-user",
                )
            elif key == "diameter":
                if len(anchors) < 3 or len(anchors) > 64:
                    raise ArtifactWorkbenchError(
                        "surface diameter requires 3..64 anchors"
                    )
                work_item = controller.begin_surface_diameter(
                    anchors,
                    coordinate_grid_um=1,
                    record_id=f"record:surface-diameter:{uuid.uuid4()}",
                    created_at=self._utc_seconds_now(),
                    operator="local-user",
                )
            else:
                raise ArtifactWorkbenchError(
                    f"unsupported surface measurement kind: {kind!r}"
                )
        except Exception as exc:
            self.status_info.setText(f"{label} 준비 실패 | 기존 문서 유지")
            QMessageBox.warning(
                self,
                f"{label} 준비 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return

        def on_done(result: object) -> None:
            if self._native_measurement_callback_is_terminal(
                controller,
                work_item,
                label=label,
            ):
                return
            try:
                if not isinstance(result, ArtifactMeasurementResult):
                    raise ArtifactWorkbenchError(
                        "surface measurement worker result is invalid"
                    )
                self._publish_native_measurement_result(work_item, result)
                self._invalidate_surface_pick_requests()
                self.viewport.clear_measure_picks()
                self.measure_panel.set_points_count(0)
            except Exception as exc:
                if self._report_artifact_authority_callback_failure(
                    context=f"{label} 결과 게시 중 권위 확인 실패",
                    detail=f"{type(exc).__name__}: {exc}",
                ):
                    return
                pending = self._native_measurement_publication_is_pending(work_item)
                self.status_info.setText(
                    f"{label} 결과 게시 보류 | 재시도 버튼 사용"
                    if pending
                    else f"{label} 결과 폐기 | 현재 문서 유지"
                )
                QMessageBox.warning(
                    self,
                    f"{label} 결과 게시 보류" if pending else f"{label} 결과 폐기",
                    f"{type(exc).__name__}: {exc}",
                )

        def on_failed(message: str) -> None:
            if self._report_artifact_authority_callback_failure(
                context=f"{label} worker 종료 콜백",
                detail=str(message),
            ):
                return
            if self._native_measurement_callback_is_terminal(
                controller,
                work_item,
                label=label,
            ):
                return
            self.status_info.setText(f"{label} 계산 실패 | 기록을 만들지 않았습니다")
            QMessageBox.warning(
                self,
                f"{label} 계산 실패",
                self._format_error_message("측정 중 오류가 발생했습니다:", message),
            )

        self.status_info.setText(f"{label} 계산 중 · canonical mm / 1 µm...")
        started = self._start_task(
            title=label,
            label=f"{label}와 pick QC 계산 중...",
            thread=TaskThread(
                f"native_surface_{key}",
                lambda: self._execute_native_measurement_with_preflight(
                    preflight,
                    controller,
                    work_item,
                ),
            ),
            on_done=on_done,
            on_failed=on_failed,
            on_cancel_requested=lambda: self._request_native_measurement_cancel(
                controller,
                work_item,
                label=label,
            ),
            on_shutdown_joined=lambda: self._verify_native_measurement_shutdown(
                controller,
                work_item,
            ),
        )
        if not started:
            controller.cancel(work_item, reason="task_not_started")

    def on_measure_point_picked(self, _point: np.ndarray) -> None:
        panel = getattr(self, "measure_panel", None)
        if panel is None:
            return

        try:
            pts = list(getattr(self.viewport, "measure_picked_points", []) or [])
        except Exception:
            pts = []

        panel.set_points_count(len(pts))
        self._refresh_native_save_indicator()

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

        distance_mesh_units = float(np.linalg.norm(p1[:3] - p0[:3]))
        if not np.isfinite(distance_mesh_units):
            return

        obj = getattr(self.viewport, "selected_obj", None)
        unit = getattr(getattr(obj, "mesh", None), "unit", None)
        dist_mm = mesh_units_to_mm(distance_mesh_units, unit)
        msg = f"[검토용] 거리: {dist_mm:.3f} mm"
        panel.append_result(msg)
        self.status_info.setText(f"{msg}")

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

        if self._native_artifact_mode():
            try:
                anchors = tuple(
                    dict(value)
                    for value in (
                        getattr(self.viewport, "measure_picked_anchors", []) or []
                    )
                )
            except Exception:
                anchors = ()
            if len(anchors) < 3 or len(anchors) > 64:
                QMessageBox.warning(
                    self,
                    "검증 원 맞춤 지름",
                    "표면 anchor 3~64개가 필요합니다.\nShift+클릭으로 점을 선택하세요.",
                )
                return
            self._start_native_surface_measurement("diameter", anchors)
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

        obj = getattr(self.viewport, "selected_obj", None)
        unit = getattr(getattr(obj, "mesh", None), "unit", None)
        diameter_mm = mesh_units_to_mm(float(arc.radius) * 2.0, unit)
        msg = f"[검토용] 지름: {diameter_mm:.3f} mm"
        panel.append_result(msg)
        self.status_info.setText(f"{msg}")

        try:
            self.viewport.clear_measure_picks()
            panel.set_points_count(0)
            self.viewport.update()
        except Exception:
            pass

    def clear_measure_points(self) -> None:
        try:
            self._invalidate_surface_pick_requests()
            self.viewport.clear_measure_picks()
            self.measure_panel.set_points_count(0)
            self.viewport.update()
            self.status_info.setText("측정 포인트 초기화")
            self._refresh_native_save_indicator()
        except Exception:
            pass

    def _invalidate_surface_pick_requests(self) -> int:
        """Invalidate workers bound to an older transient anchor list."""

        try:
            generation = int(getattr(self, "_surface_pick_generation", 0)) + 1
        except Exception:
            generation = 1
        self._surface_pick_generation = generation
        return generation

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
                label.setText("측정 결과 복사됨")
        except Exception:
            pass

    def clear_measure_results(self) -> None:
        try:
            self.measure_panel.clear_results()
            self.status_info.setText("측정 결과 지움")
        except Exception:
            pass

    def on_native_geometry_metrics_requested(self) -> None:
        """Compute and publish one canonical-mm area/guarded-volume record."""

        try:
            obj = self.viewport.selected_obj
            session = self._require_native_measurement_session(obj)
            preflight = self._capture_native_scene_preflight(session)
            controller = self._artifact_measurement_controller()
            work_item = controller.begin_geometry_metrics(
                coordinate_grid_um=1,
                record_id=f"record:geometry-metrics:{uuid.uuid4()}",
                created_at=self._utc_seconds_now(),
                operator="local-user",
            )
        except Exception as exc:
            self.status_info.setText("검증 제원 준비 실패 | 기존 문서 유지")
            QMessageBox.warning(
                self,
                "검증 제원 준비 실패",
                "원본·단위·Align이 확정된 native 문서가 필요합니다.\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            return

        def on_done(result: object) -> None:
            if self._native_measurement_callback_is_terminal(
                controller,
                work_item,
                label="검증 제원",
            ):
                return
            try:
                if not isinstance(result, ArtifactMeasurementResult):
                    raise ArtifactWorkbenchError(
                        "geometry metrics worker result is invalid"
                    )
                self._publish_native_measurement_result(work_item, result)
            except Exception as exc:
                if self._report_artifact_authority_callback_failure(
                    context="검증 제원 결과 게시 중 권위 확인 실패",
                    detail=f"{type(exc).__name__}: {exc}",
                ):
                    return
                pending = self._native_measurement_publication_is_pending(work_item)
                self.status_info.setText(
                    "검증 제원 결과 게시 보류 | 재시도 버튼 사용"
                    if pending
                    else "검증 제원 결과 폐기 | 현재 문서 유지"
                )
                QMessageBox.warning(
                    self,
                    "검증 제원 결과 게시 보류" if pending else "검증 제원 결과 폐기",
                    f"{type(exc).__name__}: {exc}",
                )

        def on_failed(message: str) -> None:
            if self._report_artifact_authority_callback_failure(
                context="검증 제원 worker 종료 콜백",
                detail=str(message),
            ):
                return
            if self._native_measurement_callback_is_terminal(
                controller,
                work_item,
                label="검증 제원",
            ):
                return
            self.status_info.setText("검증 제원 계산 실패 | 기록을 만들지 않았습니다")
            QMessageBox.warning(
                self,
                "검증 제원 계산 실패",
                self._format_error_message(
                    "표면적·체적 계산 중 오류가 발생했습니다:", message
                ),
            )

        self.status_info.setText("검증 제원 계산 중 · canonical mm / 1 µm...")
        started = self._start_task(
            title="검증 제원",
            label="표면적과 위상 검증 체적 계산 중...",
            thread=TaskThread(
                "native_geometry_metrics",
                lambda: self._execute_native_measurement_with_preflight(
                    preflight,
                    controller,
                    work_item,
                ),
            ),
            on_done=on_done,
            on_failed=on_failed,
            on_cancel_requested=lambda: self._request_native_measurement_cancel(
                controller,
                work_item,
                label="검증 제원",
            ),
            on_shutdown_joined=lambda: self._verify_native_measurement_shutdown(
                controller,
                work_item,
            ),
        )
        if not started:
            controller.cancel(work_item, reason="task_not_started")

    def compute_volume_stats(self) -> None:
        if self._native_artifact_mode():
            self.on_native_geometry_metrics_requested()
            return

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
                self.status_info.setText("부피/면적 계산 완료")
            except Exception:
                pass

        def on_failed(message: str) -> None:
            QMessageBox.critical(self, "오류", self._format_error_message("부피/면적 계산 실패:", message))
            try:
                self.status_info.setText("부피/면적 계산 실패")
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
        if enabled and self._native_artifact_mode():
            try:
                self.section_panel.btn_roi.blockSignals(True)
                self.section_panel.btn_roi.setChecked(False)
            finally:
                self.section_panel.btn_roi.blockSignals(False)
            self.status_info.setText("화면 ROI는 측정값이 아닙니다 | 검증된 외곽 도구 사용")
            return
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
        self.status_info.setText(f"{len(points)}개의 점으로 외곽선 추출 완료")
        try:
            _LOGGER.info("Extracted silhouette: %s points", len(points))
        except Exception:
            pass

    def on_silhouette_requested(self) -> None:
        if self._native_artifact_mode():
            message = (
                "현재 ROI 외곽선은 화면/convex-hull 기반이라 오목부, 구멍, 분리 성분을 "
                "보존하지 못합니다. 위의 '외곽 계산 · 기록'에서 6면 방향과 mm 정밀도를 "
                "선택해 검증된 Outline을 만드세요."
            )
            self.status_info.setText("화면 외곽선 차단 | 검증된 Outline 도구 사용")
            QMessageBox.warning(self, "화면 외곽선은 검토용", message)
            return
        self.viewport.extract_roi_silhouette()

    def _latest_native_vector_record(self):
        session = getattr(self, "_artifact_session", None)
        if not isinstance(session, ArtifactSession):
            return None
        known_types = {kind.record_type for kind in VectorRecordKind}
        candidates = []
        for record in session.document.records:
            try:
                if (
                    record.type in known_types
                    and self._native_vector_record_is_exportable(session, record)
                ):
                    candidates.append(record)
            except Exception:
                continue
        if not candidates:
            return None
        return max(candidates, key=lambda record: (str(record.created_at), record.id))

    @staticmethod
    def _native_vector_record_is_exportable(session, record) -> bool:
        return bool(
            isinstance(session, ArtifactSession)
            and record.type in {kind.record_type for kind in VectorRecordKind}
            and str(record.lifecycle_status.value) == "ready"
            and session.document.record_freshness(record.id).value == "fresh"
        )

    def _current_native_vector_record(self):
        session = getattr(self, "_artifact_session", None)
        if not isinstance(session, ArtifactSession):
            return None
        preview_id = getattr(self.viewport, "native_vector_preview_record_id", None)
        if (
            isinstance(preview_id, str)
            and self._native_vector_preview_document_id
            == session.document.document_id
        ):
            record = session.document.record_index.get(preview_id)
            if record is not None and self._native_vector_record_is_exportable(
                session, record
            ):
                return record
        return None

    @staticmethod
    def _native_rubbing_record_is_exportable(session, record) -> bool:
        return bool(
            isinstance(session, ArtifactSession)
            and getattr(record, "type", None) == RUBBING_RECORD_TYPE
            and str(record.lifecycle_status.value) == "ready"
            and session.document.record_freshness(record.id).value == "fresh"
        )

    def _latest_native_rubbing_record(self):
        session = getattr(self, "_artifact_session", None)
        if not isinstance(session, ArtifactSession):
            return None
        candidates = []
        for record in session.document.records:
            try:
                if self._native_rubbing_record_is_exportable(session, record):
                    candidates.append(record)
            except Exception:
                continue
        return (
            max(candidates, key=lambda record: (str(record.created_at), record.id))
            if candidates
            else None
        )

    def _current_native_rubbing_record(self):
        session = getattr(self, "_artifact_session", None)
        if not isinstance(session, ArtifactSession):
            return None
        preview_id = getattr(self, "_native_rubbing_preview_record_id", None)
        if (
            isinstance(preview_id, str)
            and self._native_rubbing_preview_document_id
            == session.document.document_id
        ):
            record = session.document.record_index.get(preview_id)
            if record is not None and self._native_rubbing_record_is_exportable(
                session, record
            ):
                return record
        return None

    @staticmethod
    def _native_tile_unwrap_record_is_exportable(session, record) -> bool:
        return bool(
            isinstance(session, ArtifactSession)
            and getattr(record, "type", None) == TILE_UNWRAP_RECORD_TYPE
            and str(record.lifecycle_status.value) == "ready"
            and session.document.record_freshness(record.id).value == "fresh"
        )

    def _current_native_tile_unwrap_record(self):
        session = getattr(self, "_artifact_session", None)
        if not isinstance(session, ArtifactSession):
            return None
        preview_id = getattr(self, "_native_tile_unwrap_preview_record_id", None)
        if (
            isinstance(preview_id, str)
            and self._native_tile_unwrap_preview_document_id
            == session.document.document_id
        ):
            record = session.document.record_index.get(preview_id)
            if record is not None and self._native_tile_unwrap_record_is_exportable(
                session,
                record,
            ):
                return record
        return None

    def _clear_native_vector_preview(self) -> None:
        self._native_vector_preview_document_id = None
        self.viewport.set_native_vector_preview(None)

    def _clear_native_rubbing_preview(self) -> None:
        self._native_rubbing_preview_record_id = None
        self._native_rubbing_preview_document_id = None
        self._native_rubbing_preview_geometry_ref = None
        self._native_rubbing_preview_pending_record_id = None
        self._native_rubbing_preview_pending_record = None
        self._native_rubbing_preview_pending_token = None
        panel = getattr(self, "section_panel", None)
        if panel is None or not hasattr(panel, "label_native_rubbing_preview"):
            return
        panel.label_native_rubbing_preview.clear()
        panel.label_native_rubbing_preview.setText(
            "READY + FRESH 탁본을 계산하면 미리보기가 표시됩니다.\n"
            "프로젝트를 다시 연 경우 export 또는 재계산으로 픽셀을 검증합니다."
        )
        panel.label_native_rubbing_info.setText("READY + FRESH 탁본 기록 없음")

    def _clear_native_tile_unwrap_preview(self) -> None:
        self._native_tile_unwrap_preview_record_id = None
        self._native_tile_unwrap_preview_document_id = None
        self._native_tile_unwrap_preview_geometry_ref = None
        self._native_tile_unwrap_preview_pending_record_id = None
        self._native_tile_unwrap_preview_pending_record = None
        self._native_tile_unwrap_preview_pending_token = None
        panel = getattr(self, "section_panel", None)
        if panel is None or not hasattr(panel, "label_native_tile_unwrap_preview"):
            return
        panel.label_native_tile_unwrap_preview.clear()
        panel.label_native_tile_unwrap_preview.setText(
            "READY + FRESH 기와 전개를 계산하면 미리보기가 표시됩니다.\n"
            "미리보기는 비권위이며 export 때 record recipe를 다시 계산합니다."
        )
        panel.label_native_tile_unwrap_info.setText(
            "READY + FRESH 기와 전개 기록 없음"
        )

    def _preview_native_rubbing(
        self,
        session: ArtifactSession,
        record_id: str,
        raster: DigitalRubbingRaster,
    ) -> None:
        record = session.document.record_index.get(record_id)
        if record is None or not self._native_rubbing_record_is_exportable(
            session, record
        ):
            raise ArtifactRubbingError(
                "native Digital Rubbing preview requires a READY + FRESH record"
            )
        if raster.receipt() != rubbing_receipt_from_record(record):
            raise ArtifactRubbingError(
                "native Digital Rubbing preview does not match its record receipt"
            )
        image = Image.fromarray(raster.pixels, mode="LA").convert("RGBA")
        pixmap = self._pixmap_from_pil_image(image)
        scaled = pixmap.scaled(
            420,
            260,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        panel = self.section_panel
        panel.label_native_rubbing_preview.setPixmap(scaled)
        panel.label_native_rubbing_info.setText(
            f"{record.recipe['view']} · {raster.width_pixels}×{raster.height_pixels} px · "
            f"{raster.pixels_per_meter // 1000} px/mm · record {record.id}"
        )
        self._native_rubbing_preview_record_id = record.id
        self._native_rubbing_preview_document_id = session.document.document_id
        self._native_rubbing_preview_geometry_ref = record.geometry_ref
        self._native_rubbing_preview_pending_record_id = None
        self._native_rubbing_preview_pending_record = None
        self._native_rubbing_preview_pending_token = None

    def _preview_native_tile_unwrap(
        self,
        session: ArtifactSession,
        record_id: str,
        unwrap: TileUnwrapMesh,
    ) -> None:
        record = session.document.record_index.get(record_id)
        if record is None or not self._native_tile_unwrap_record_is_exportable(
            session,
            record,
        ):
            raise ArtifactTileUnwrapError(
                "native tile unwrap preview requires a READY + FRESH record"
            )
        receipt = tile_unwrap_receipt_from_record(record)
        if unwrap.receipt(
            selection_sha256=str(receipt["selection_sha256"])
        ) != receipt:
            raise ArtifactTileUnwrapError(
                "native tile unwrap preview does not match its record receipt"
            )

        width_px, height_px, margin_px = 420, 260, 12
        image = Image.new("RGBA", (width_px, height_px), (247, 250, 252, 255))
        draw = ImageDraw.Draw(image)
        uv = np.asarray(unwrap.uv_um, dtype=np.float64)
        minimum = np.min(uv, axis=0)
        maximum = np.max(uv, axis=0)
        span = np.maximum(maximum - minimum, 1.0)
        scale = min(
            (width_px - 2 * margin_px) / float(span[0]),
            (height_px - 2 * margin_px) / float(span[1]),
        )

        def point(index: int) -> tuple[float, float]:
            normalized = (uv[index] - minimum) * scale
            return (
                margin_px + float(normalized[0]),
                height_px - margin_px - float(normalized[1]),
            )

        faces = np.asarray(unwrap.faces, dtype=np.int64)
        if faces.shape[0] > 5_000:
            face_ids = np.linspace(
                0,
                faces.shape[0] - 1,
                num=5_000,
                dtype=np.int64,
            )
            faces = faces[face_ids]
        for a, b, c in faces:
            triangle = [point(int(a)), point(int(b)), point(int(c))]
            draw.line(triangle + [triangle[0]], fill=(45, 94, 120, 115), width=1)
        pixmap = self._pixmap_from_pil_image(image)
        self.section_panel.label_native_tile_unwrap_preview.setPixmap(pixmap)

        width_mm = int(receipt["width_mm_exact"]["numerator"]) / int(
            receipt["width_mm_exact"]["denominator"]
        )
        height_mm = int(receipt["height_mm_exact"]["numerator"]) / int(
            receipt["height_mm_exact"]["denominator"]
        )
        qc = record.qc
        distortion_percent = int(qc.get("distortion_p95_millionths", 0)) / 10_000.0
        self.section_panel.label_native_tile_unwrap_info.setText(
            f"{str(record.recipe['record_view']).title()} · 길이축 "
            f"{str(record.recipe['longitudinal_axis']).upper()} · "
            f"선택 {int(qc.get('selected_face_count', 0)):,}면 · "
            f"{width_mm:g}×{height_mm:g} mm · "
            f"section {int(qc.get('section_count', 0))} · "
            f"foldover {int(qc.get('foldover_face_count', 0))} · "
            f"왜곡 p95 {distortion_percent:.3f}%"
        )
        self._native_tile_unwrap_preview_record_id = record.id
        self._native_tile_unwrap_preview_document_id = session.document.document_id
        self._native_tile_unwrap_preview_geometry_ref = record.geometry_ref
        self._native_tile_unwrap_preview_pending_record_id = None
        self._native_tile_unwrap_preview_pending_record = None
        self._native_tile_unwrap_preview_pending_token = None

    def _preview_native_vector_record(
        self,
        session: ArtifactSession,
        record_id: str,
    ) -> None:
        record = session.document.record_index.get(record_id)
        if record is None or not self._native_vector_record_is_exportable(
            session, record
        ):
            raise ArtifactVectorExtractionError(
                "native vector preview requires a READY + FRESH record"
            )
        payload = vector_payload_from_record(record)
        self.viewport.set_native_vector_preview(payload, record_id=record.id)
        self._native_vector_preview_document_id = session.document.document_id

    @staticmethod
    def _native_record_choice_label(record: Any) -> str:
        record_type = str(getattr(record, "type", ""))
        kind = {
            "vector.cutline.v1": "Cutline",
            "vector.outline.v1": "Outline",
            RUBBING_RECORD_TYPE: "Digital Rubbing",
            TILE_UNWRAP_RECORD_TYPE: "Tile Unwrap",
        }.get(record_type, record_type or "Record")
        recipe = getattr(record, "recipe", {})
        view = ""
        if isinstance(recipe, Mapping):
            view = str(
                recipe.get("view", recipe.get("record_view", ""))
            ).strip().title()
        details = " · ".join(part for part in (kind, view) if part)
        return f"{details} · {record.id} · {record.created_at}"

    @staticmethod
    def _replace_native_record_choices(
        combo: QComboBox,
        *,
        placeholder: str,
        records: list[Any],
        selected_id: str | None,
    ) -> None:
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(placeholder, None)
            for record in sorted(
                records,
                key=lambda item: (str(item.created_at), str(item.id)),
                reverse=True,
            ):
                combo.addItem(MainWindow._native_record_choice_label(record), record.id)
            selected_index = combo.findData(selected_id) if selected_id else -1
            combo.setCurrentIndex(selected_index if selected_index >= 1 else 0)
        finally:
            combo.blockSignals(False)

    def _refresh_native_record_selectors(
        self,
        session: ArtifactSession | None,
    ) -> None:
        panel = getattr(self, "section_panel", None)
        if panel is None or not hasattr(panel, "combo_native_vector_record"):
            return
        vector_records: list[Any] = []
        rubbing_records: list[Any] = []
        tile_unwrap_records: list[Any] = []
        selected_vector_id: str | None = None
        selected_rubbing_id: str | None = None
        selected_tile_unwrap_id: str | None = None
        if isinstance(session, ArtifactSession):
            vector_record = self._current_native_vector_record()
            rubbing_record = self._current_native_rubbing_record()
            tile_unwrap_record = self._current_native_tile_unwrap_record()
            selected_vector_id = vector_record.id if vector_record is not None else None
            selected_rubbing_id = rubbing_record.id if rubbing_record is not None else None
            selected_tile_unwrap_id = (
                tile_unwrap_record.id if tile_unwrap_record is not None else None
            )
            if selected_rubbing_id is None:
                pending_id = getattr(
                    self,
                    "_native_rubbing_preview_pending_record_id",
                    None,
                )
                pending_record = getattr(
                    self,
                    "_native_rubbing_preview_pending_record",
                    None,
                )
                pending_token = getattr(
                    self,
                    "_native_rubbing_preview_pending_token",
                    None,
                )
                current_pending = (
                    session.document.record_index.get(pending_id)
                    if isinstance(pending_id, str)
                    else None
                )
                if (
                    pending_token is not None
                    and current_pending is not None
                    and current_pending == pending_record
                    and self._native_rubbing_record_is_exportable(
                        session,
                        current_pending,
                    )
                ):
                    selected_rubbing_id = pending_id
            if selected_tile_unwrap_id is None:
                pending_id = getattr(
                    self,
                    "_native_tile_unwrap_preview_pending_record_id",
                    None,
                )
                pending_record = getattr(
                    self,
                    "_native_tile_unwrap_preview_pending_record",
                    None,
                )
                pending_token = getattr(
                    self,
                    "_native_tile_unwrap_preview_pending_token",
                    None,
                )
                current_pending = (
                    session.document.record_index.get(pending_id)
                    if isinstance(pending_id, str)
                    else None
                )
                if (
                    pending_token is not None
                    and current_pending is not None
                    and current_pending == pending_record
                    and self._native_tile_unwrap_record_is_exportable(
                        session,
                        current_pending,
                    )
                ):
                    selected_tile_unwrap_id = pending_id
            for record in session.document.records:
                try:
                    if self._native_vector_record_is_exportable(session, record):
                        vector_records.append(record)
                    elif self._native_rubbing_record_is_exportable(session, record):
                        rubbing_records.append(record)
                    elif self._native_tile_unwrap_record_is_exportable(
                        session,
                        record,
                    ):
                        tile_unwrap_records.append(record)
                except Exception:
                    continue
        self._replace_native_record_choices(
            panel.combo_native_vector_record,
            placeholder="READY + FRESH 벡터 기록을 명시적으로 선택",
            records=vector_records,
            selected_id=selected_vector_id,
        )
        self._replace_native_record_choices(
            panel.combo_native_rubbing_record,
            placeholder="READY + FRESH 탁본 기록을 명시적으로 선택",
            records=rubbing_records,
            selected_id=selected_rubbing_id,
        )
        self._replace_native_record_choices(
            panel.combo_native_tile_unwrap_record,
            placeholder="READY + FRESH 기와 전개 기록을 명시적으로 선택",
            records=tile_unwrap_records,
            selected_id=selected_tile_unwrap_id,
        )

    @staticmethod
    def _reset_native_record_choice(combo: QComboBox) -> None:
        combo.blockSignals(True)
        try:
            combo.setCurrentIndex(0)
        finally:
            combo.blockSignals(False)

    def on_native_vector_record_selected(self, record_id: str) -> None:
        panel = self.section_panel
        if not record_id:
            self._clear_native_vector_preview()
            panel.btn_native_vector_export.setEnabled(False)
            self.status_info.setText("벡터 기록 선택을 해제했습니다.")
            return
        session = getattr(self, "_artifact_session", None)
        record = (
            session.document.record_index.get(record_id)
            if isinstance(session, ArtifactSession)
            else None
        )
        if (
            not self._native_measurement_ready()
            or record is None
            or not self._native_vector_record_is_exportable(session, record)
        ):
            self._clear_native_vector_preview()
            self._reset_native_record_choice(panel.combo_native_vector_record)
            panel.btn_native_vector_export.setEnabled(False)
            self.status_info.setText("선택한 벡터 기록은 READY + FRESH 상태가 아닙니다.")
            return
        try:
            self._preview_native_vector_record(session, record.id)
        except Exception as exc:
            self._clear_native_vector_preview()
            self._reset_native_record_choice(panel.combo_native_vector_record)
            panel.btn_native_vector_export.setEnabled(False)
            self.status_info.setText(f"벡터 기록 미리보기 실패: {exc}")
            return
        panel.btn_native_vector_export.setEnabled(True)
        self.status_info.setText(f"벡터 기록 선택: {record.id}")

    def on_native_rubbing_record_selected(self, record_id: str) -> None:
        panel = self.section_panel
        if not record_id:
            self._clear_native_rubbing_preview()
            panel.btn_native_rubbing_export.setEnabled(False)
            self.status_info.setText("탁본 기록 선택을 해제했습니다.")
            return
        session = getattr(self, "_artifact_session", None)
        record = (
            session.document.record_index.get(record_id)
            if isinstance(session, ArtifactSession)
            else None
        )
        if (
            not self._native_measurement_ready()
            or record is None
            or not self._native_rubbing_record_is_exportable(session, record)
        ):
            self._clear_native_rubbing_preview()
            self._reset_native_record_choice(panel.combo_native_rubbing_record)
            panel.btn_native_rubbing_export.setEnabled(False)
            self.status_info.setText("선택한 탁본 기록은 READY + FRESH 상태가 아닙니다.")
            return
        if self._artifact_measurement_controller().active_summaries:
            self._clear_native_rubbing_preview()
            self._reset_native_record_choice(panel.combo_native_rubbing_record)
            panel.btn_native_rubbing_export.setEnabled(False)
            self.status_info.setText(
                "진행·보류 중인 실측 결과를 먼저 완료하거나 해제하세요."
            )
            return
        try:
            captured_snapshot = session.projection_snapshot()
        except Exception as exc:
            self._clear_native_rubbing_preview()
            self._reset_native_record_choice(panel.combo_native_rubbing_record)
            panel.btn_native_rubbing_export.setEnabled(False)
            self.status_info.setText(f"탁본 기록 권위 확인 실패: {exc}")
            return

        self._clear_native_rubbing_preview()
        preview_token = object()
        self._native_rubbing_preview_pending_record_id = record_id
        self._native_rubbing_preview_pending_record = record
        self._native_rubbing_preview_pending_token = preview_token
        panel.btn_native_rubbing_export.setEnabled(False)

        def owns_preview_request() -> bool:
            return (
                getattr(self, "_native_rubbing_preview_pending_token", None)
                is preview_token
            )

        def current_record_if_authoritative():
            if not owns_preview_request():
                return None
            current = getattr(self, "_artifact_session", None)
            if not isinstance(current, ArtifactSession):
                return None
            try:
                if (
                    current.source_mesh is not session.source_mesh
                    or not current.projection_snapshot().has_same_render_projection(
                        captured_snapshot
                    )
                    or str(panel.combo_native_rubbing_record.currentData() or "")
                    != record_id
                ):
                    return None
                current_record = current.document.record_index.get(record_id)
                if (
                    current_record != record
                    or not self._native_rubbing_record_is_exportable(
                        current,
                        current_record,
                    )
                ):
                    return None
            except Exception:
                return None
            return current, current_record

        def on_done(result: object) -> None:
            if not owns_preview_request():
                return
            if self._restore_artifact_authority_fault_status():
                return
            authoritative = current_record_if_authoritative()
            if not isinstance(result, DigitalRubbingRaster) or authoritative is None:
                self._clear_native_rubbing_preview()
                self._reset_native_record_choice(panel.combo_native_rubbing_record)
                panel.btn_native_rubbing_export.setEnabled(False)
                self.status_info.setText(
                    "늦은 탁본 기록 미리보기 폐기 | 현재 문서 유지"
                )
                return
            current, current_record = authoritative
            try:
                self._preview_native_rubbing(current, current_record.id, result)
            except Exception as exc:
                self._clear_native_rubbing_preview()
                self._reset_native_record_choice(panel.combo_native_rubbing_record)
                panel.btn_native_rubbing_export.setEnabled(False)
                self.status_info.setText(f"탁본 기록 미리보기 실패: {exc}")
                return
            panel.btn_native_rubbing_export.setEnabled(True)
            self.status_info.setText(f"탁본 기록 선택: {current_record.id}")

        def on_failed(message: str) -> None:
            if not owns_preview_request():
                return
            if self._restore_artifact_authority_fault_status():
                return
            if str(panel.combo_native_rubbing_record.currentData() or "") != record_id:
                return
            self._clear_native_rubbing_preview()
            self._reset_native_record_choice(panel.combo_native_rubbing_record)
            panel.btn_native_rubbing_export.setEnabled(False)
            self.status_info.setText("탁본 기록 재계산 실패")
            QMessageBox.warning(
                self,
                "탁본 기록 미리보기 실패",
                self._format_error_message("탁본 기록 재계산 중 오류:", message),
            )

        self.status_info.setText("탁본 기록 미리보기 재계산 중...")
        try:
            started = self._start_task(
                title="Digital Rubbing 기록 미리보기",
                label="저장된 recipe로 탁본 raster를 재검증하는 중...",
                thread=TaskThread(
                    "native_rubbing_record_preview",
                    lambda: MainWindow._recompute_native_rubbing_record(session, record),
                ),
                on_done=on_done,
                on_failed=on_failed,
            )
        except Exception as exc:
            if not owns_preview_request():
                return
            self._clear_native_rubbing_preview()
            self._reset_native_record_choice(panel.combo_native_rubbing_record)
            panel.btn_native_rubbing_export.setEnabled(False)
            if not self._restore_artifact_authority_fault_status():
                self.status_info.setText("탁본 기록 미리보기 시작 실패")
                QMessageBox.warning(
                    self,
                    "탁본 기록 미리보기 실패",
                    f"{type(exc).__name__}: {exc}",
                )
            return
        if not started and owns_preview_request():
            self._clear_native_rubbing_preview()
            self._reset_native_record_choice(panel.combo_native_rubbing_record)
            panel.btn_native_rubbing_export.setEnabled(False)

    def on_native_tile_unwrap_record_selected(self, record_id: str) -> None:
        panel = self.section_panel
        if not record_id:
            self._clear_native_tile_unwrap_preview()
            panel.btn_native_tile_unwrap_export.setEnabled(False)
            self.status_info.setText("기와 전개 기록 선택을 해제했습니다.")
            return
        session = getattr(self, "_artifact_session", None)
        record = (
            session.document.record_index.get(record_id)
            if isinstance(session, ArtifactSession)
            else None
        )
        if (
            not self._native_measurement_ready()
            or record is None
            or not self._native_tile_unwrap_record_is_exportable(session, record)
        ):
            self._clear_native_tile_unwrap_preview()
            self._reset_native_record_choice(panel.combo_native_tile_unwrap_record)
            panel.btn_native_tile_unwrap_export.setEnabled(False)
            self.status_info.setText(
                "선택한 기와 전개 기록은 READY + FRESH 상태가 아닙니다."
            )
            return
        if self._artifact_measurement_controller().active_summaries:
            self._clear_native_tile_unwrap_preview()
            self._reset_native_record_choice(panel.combo_native_tile_unwrap_record)
            panel.btn_native_tile_unwrap_export.setEnabled(False)
            self.status_info.setText(
                "진행·보류 중인 실측 결과를 먼저 완료하거나 해제하세요."
            )
            return
        try:
            captured_snapshot = session.projection_snapshot()
        except Exception as exc:
            self._clear_native_tile_unwrap_preview()
            self._reset_native_record_choice(panel.combo_native_tile_unwrap_record)
            panel.btn_native_tile_unwrap_export.setEnabled(False)
            self.status_info.setText(f"기와 전개 기록 권위 확인 실패: {exc}")
            return

        self._clear_native_tile_unwrap_preview()
        preview_token = object()
        self._native_tile_unwrap_preview_pending_record_id = record_id
        self._native_tile_unwrap_preview_pending_record = record
        self._native_tile_unwrap_preview_pending_token = preview_token
        panel.btn_native_tile_unwrap_export.setEnabled(False)

        def owns_preview_request() -> bool:
            return (
                getattr(self, "_native_tile_unwrap_preview_pending_token", None)
                is preview_token
            )

        def current_record_if_authoritative():
            if not owns_preview_request():
                return None
            current = getattr(self, "_artifact_session", None)
            if not isinstance(current, ArtifactSession):
                return None
            try:
                if (
                    current.source_mesh is not session.source_mesh
                    or not current.projection_snapshot().has_same_render_projection(
                        captured_snapshot
                    )
                    or str(
                        panel.combo_native_tile_unwrap_record.currentData() or ""
                    )
                    != record_id
                ):
                    return None
                current_record = current.document.record_index.get(record_id)
                if (
                    current_record != record
                    or not self._native_tile_unwrap_record_is_exportable(
                        current,
                        current_record,
                    )
                ):
                    return None
            except Exception:
                return None
            return current, current_record

        def on_done(result: object) -> None:
            if not owns_preview_request():
                return
            if self._restore_artifact_authority_fault_status():
                return
            authoritative = current_record_if_authoritative()
            if not isinstance(result, TileUnwrapMesh) or authoritative is None:
                self._clear_native_tile_unwrap_preview()
                self._reset_native_record_choice(
                    panel.combo_native_tile_unwrap_record
                )
                panel.btn_native_tile_unwrap_export.setEnabled(False)
                self.status_info.setText(
                    "늦은 기와 전개 미리보기 폐기 | 현재 문서 유지"
                )
                return
            current, current_record = authoritative
            try:
                self._preview_native_tile_unwrap(
                    current,
                    current_record.id,
                    result,
                )
            except Exception as exc:
                self._clear_native_tile_unwrap_preview()
                self._reset_native_record_choice(
                    panel.combo_native_tile_unwrap_record
                )
                panel.btn_native_tile_unwrap_export.setEnabled(False)
                self.status_info.setText(f"기와 전개 미리보기 실패: {exc}")
                return
            panel.btn_native_tile_unwrap_export.setEnabled(True)
            self.status_info.setText(f"기와 전개 기록 선택: {current_record.id}")

        def on_failed(message: str) -> None:
            if not owns_preview_request():
                return
            if self._restore_artifact_authority_fault_status():
                return
            if (
                str(panel.combo_native_tile_unwrap_record.currentData() or "")
                != record_id
            ):
                return
            self._clear_native_tile_unwrap_preview()
            self._reset_native_record_choice(panel.combo_native_tile_unwrap_record)
            panel.btn_native_tile_unwrap_export.setEnabled(False)
            self.status_info.setText("기와 전개 기록 재계산 실패")
            QMessageBox.warning(
                self,
                "기와 전개 기록 미리보기 실패",
                self._format_error_message("기와 전개 재계산 중 오류:", message),
            )

        self.status_info.setText("기와 전개 기록 미리보기 재계산 중...")
        try:
            started = self._start_task(
                title="기와 전개 기록 미리보기",
                label="저장된 recipe로 µm 전개 좌표를 재검증하는 중...",
                thread=TaskThread(
                    "native_tile_unwrap_record_preview",
                    lambda: MainWindow._recompute_native_tile_unwrap_record(
                        session,
                        record,
                    ),
                ),
                on_done=on_done,
                on_failed=on_failed,
            )
        except Exception as exc:
            if not owns_preview_request():
                return
            self._clear_native_tile_unwrap_preview()
            self._reset_native_record_choice(panel.combo_native_tile_unwrap_record)
            panel.btn_native_tile_unwrap_export.setEnabled(False)
            if not self._restore_artifact_authority_fault_status():
                self.status_info.setText("기와 전개 미리보기 시작 실패")
                QMessageBox.warning(
                    self,
                    "기와 전개 미리보기 실패",
                    f"{type(exc).__name__}: {exc}",
                )
            return
        if not started and owns_preview_request():
            self._clear_native_tile_unwrap_preview()
            self._reset_native_record_choice(panel.combo_native_tile_unwrap_record)
            panel.btn_native_tile_unwrap_export.setEnabled(False)

    def _sync_native_cutline_controls(self, *, reset_offset: bool) -> None:
        panel = getattr(self, "section_panel", None)
        if panel is None or not hasattr(panel, "native_group"):
            return
        self._prune_pending_native_measurement_publications()
        native = self._native_artifact_mode()
        panel.native_group.setEnabled(native)
        panel.legacy_line_group.setEnabled(not native)
        panel.legacy_roi_group.setEnabled(not native)
        if not native:
            panel.apply_native_workflow_progress(ArtifactWorkflowProgress.empty())
            panel.btn_native_tile_unwrap.setEnabled(False)
            panel.btn_native_vector_export.setEnabled(False)
            panel.btn_native_rubbing_export.setEnabled(False)
            panel.btn_native_tile_unwrap_export.setEnabled(False)
            self._clear_native_vector_preview()
            self._clear_native_rubbing_preview()
            self._clear_native_tile_unwrap_preview()
            self._refresh_native_record_selectors(None)
            return
        if not self._native_measurement_ready():
            panel.apply_native_workflow_progress(ArtifactWorkflowProgress.empty())
            panel.btn_native_tile_unwrap.setEnabled(False)
            panel.btn_native_vector_export.setEnabled(False)
            panel.btn_native_rubbing_export.setEnabled(False)
            panel.btn_native_tile_unwrap_export.setEnabled(False)
            self._clear_native_vector_preview()
            self._clear_native_rubbing_preview()
            self._clear_native_tile_unwrap_preview()
            self._refresh_native_record_selectors(None)
            panel.label_native_rubbing_info.setText(
                "정위치 확정 후 Cutline · Outline · Digital Rubbing · 기와 전개를 사용할 수 있습니다."
            )
            panel.label_native_tile_unwrap_info.setText(
                "정위치 확정 후 기록면·장축을 명시해 기와 전개를 사용할 수 있습니다."
            )
            return
        session = self._artifact_session
        assert isinstance(session, ArtifactSession)
        live_obj = getattr(self.viewport, "selected_obj", None)
        live_mesh = getattr(live_obj, "mesh", None)
        if (
            live_obj is None
            or live_mesh is None
            or getattr(live_obj, "_amr_artifact_projection_snapshot", None)
            != session.projection_snapshot()
        ):
            raise ArtifactSessionError(
                "native cutline controls require the current live projection"
            )
        workflow_progress = derive_artifact_workflow_progress(
            session,
            align_ready=True,
        )
        panel.apply_native_workflow_progress(workflow_progress)
        panel.btn_native_tile_unwrap.setEnabled(True)
        selected_count = len(getattr(live_obj, "selected_faces", set()) or set())
        panel.label_native_tile_selection.setText(
            f"현재 선택 {selected_count:,}면 · 전체 "
            f"{int(session.source_mesh.faces.shape[0]):,}면"
        )
        bounds = np.asarray(live_mesh.bounds, dtype=np.float64)
        view = str(panel.combo_native_cutline_view.currentData() or "top")
        axis = _NATIVE_CUTLINE_AXIS_INDEX.get(view, 2)
        minimum = float(bounds[0, axis])
        maximum = float(bounds[1, axis])
        span = max(maximum - minimum, 1e-6)
        spin = panel.spin_native_cutline_offset
        previous = float(spin.value())
        spin.blockSignals(True)
        try:
            spin.setRange(minimum, maximum)
            spin.setSingleStep(max(span / 200.0, 0.001))
            if reset_offset or previous < minimum or previous > maximum:
                spin.setValue((minimum + maximum) * 0.5)
        finally:
            spin.blockSignals(False)
        current_vector = self._current_native_vector_record()
        if current_vector is None and isinstance(
            getattr(self.viewport, "native_vector_preview_record_id", None), str
        ):
            self._clear_native_vector_preview()
        current_rubbing = self._current_native_rubbing_record()
        if current_rubbing is None and isinstance(
            getattr(self, "_native_rubbing_preview_record_id", None), str
        ):
            self._clear_native_rubbing_preview()
        current_tile_unwrap = self._current_native_tile_unwrap_record()
        if current_tile_unwrap is None and isinstance(
            getattr(self, "_native_tile_unwrap_preview_record_id", None),
            str,
        ):
            self._clear_native_tile_unwrap_preview()
        self._refresh_native_record_selectors(session)
        panel.btn_native_vector_export.setEnabled(current_vector is not None)
        panel.btn_native_rubbing_export.setEnabled(current_rubbing is not None)
        panel.btn_native_tile_unwrap_export.setEnabled(
            current_tile_unwrap is not None
        )
        preview_id = getattr(self, "_native_rubbing_preview_record_id", None)
        if not isinstance(preview_id, str):
            pending_preview_id = getattr(
                self,
                "_native_rubbing_preview_pending_record_id",
                None,
            )
            pending_preview_token = getattr(
                self,
                "_native_rubbing_preview_pending_token",
                None,
            )
            if current_rubbing is None and (
                not isinstance(pending_preview_id, str)
                or pending_preview_token is None
            ):
                self._clear_native_rubbing_preview()
        else:
            preview_record = session.document.record_index.get(preview_id)
            if (
                self._native_rubbing_preview_document_id
                != session.document.document_id
                or preview_record is None
                or self._native_rubbing_preview_geometry_ref
                != getattr(preview_record, "geometry_ref", None)
                or not self._native_rubbing_record_is_exportable(session, preview_record)
            ):
                self._clear_native_rubbing_preview()
        tile_preview_id = getattr(
            self,
            "_native_tile_unwrap_preview_record_id",
            None,
        )
        if not isinstance(tile_preview_id, str):
            pending_tile_preview_id = getattr(
                self,
                "_native_tile_unwrap_preview_pending_record_id",
                None,
            )
            pending_tile_preview_token = getattr(
                self,
                "_native_tile_unwrap_preview_pending_token",
                None,
            )
            if current_tile_unwrap is None and (
                not isinstance(pending_tile_preview_id, str)
                or pending_tile_preview_token is None
            ):
                self._clear_native_tile_unwrap_preview()
        else:
            tile_preview_record = session.document.record_index.get(tile_preview_id)
            if (
                self._native_tile_unwrap_preview_document_id
                != session.document.document_id
                or tile_preview_record is None
                or self._native_tile_unwrap_preview_geometry_ref
                != getattr(tile_preview_record, "geometry_ref", None)
                or not self._native_tile_unwrap_record_is_exportable(
                    session,
                    tile_preview_record,
                )
            ):
                self._clear_native_tile_unwrap_preview()

    def on_native_cutline_view_changed(self, _view: str) -> None:
        try:
            self._sync_native_cutline_controls(reset_offset=True)
        except Exception as exc:
            self.status_info.setText(f"native 단면 범위 갱신 실패: {exc}")

    @staticmethod
    def _utc_seconds_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )

    def _publish_native_measurement_result(
        self,
        work_item: ArtifactMeasurementWorkItem,
        result: ArtifactMeasurementResult,
    ) -> str:
        """Rebase one worker computation and publish its exact reserved record."""

        if not isinstance(work_item, ArtifactMeasurementWorkItem) or not isinstance(
            result,
            ArtifactMeasurementResult,
        ):
            raise ArtifactWorkbenchError(
                "native measurement worker returned an invalid operation result"
            )
        computation = result.computation
        recipe = work_item.recipe_dict()
        if work_item.kind is MeasurementOperationKind.CUTLINE:
            assert isinstance(computation, ArtifactVectorComputation)
            frame = recipe.get("frame")
            normal = (
                tuple(frame.get("normal_world", ()))
                if isinstance(frame, dict)
                else ()
            )
            view_label = {
                (0, 0, 1): "Top",
                (0, -1, 0): "Front",
                (1, 0, 0): "Right",
            }.get(normal, "Cutline")
            status_text = (
                f"{view_label} Cutline 기록 | "
                f"{len(computation.payload.paths)}개 경로 | canonical mm"
            )
        elif work_item.kind is MeasurementOperationKind.OUTLINE:
            assert isinstance(computation, ArtifactVectorComputation)
            status_text = (
                f"{str(recipe['view']).title()} Outline 기록 | "
                f"{int(computation.qc.get('component_count', 0))}개 성분 · "
                f"{int(computation.qc.get('hole_count', 0))}개 구멍 | "
                f"grid {float(recipe['precision_grid_mm']):g} mm"
            )
        elif work_item.kind is MeasurementOperationKind.DIGITAL_RUBBING:
            assert isinstance(computation, ArtifactRubbingComputation)
            status_text = (
                f"{str(computation.recipe['view']).title()} Digital Rubbing 기록 | "
                f"{computation.raster.width_pixels}×{computation.raster.height_pixels} px · "
                f"ink {int(computation.qc.get('inked_pixel_count', 0))} px"
            )
        elif work_item.kind is MeasurementOperationKind.TILE_UNWRAP:
            assert isinstance(computation, ArtifactTileUnwrapComputation)
            status_text = (
                f"{str(computation.recipe['record_view']).title()} 기와 전개 기록 | "
                f"길이축 {str(computation.recipe['longitudinal_axis']).upper()} · "
                f"{int(computation.qc.get('selected_face_count', 0)):,}면 · "
                f"foldover {int(computation.qc.get('foldover_face_count', 0))}"
            )
        elif work_item.kind is MeasurementOperationKind.GEOMETRY_METRICS:
            assert isinstance(computation, ArtifactGeometryMetricsComputation)
            receipt = computation.receipt_dict()
            surface = receipt["surface_area"]
            volume = receipt["volume"]
            assert isinstance(surface, dict)
            assert isinstance(volume, dict)
            volume_text = (
                f"체적 {volume['decimal_mm3']} mm³"
                if volume["status"] == "available"
                else "체적 산출 보류(위상 QC)"
            )
            status_text = (
                f"검증 제원 기록 | 표면적 {surface['decimal_mm2']} mm² · "
                f"{volume_text}"
            )
        elif work_item.kind in {
            MeasurementOperationKind.SURFACE_DISTANCE,
            MeasurementOperationKind.SURFACE_DIAMETER,
        }:
            assert isinstance(computation, ArtifactSurfaceMeasurementComputation)
            receipt = computation.receipt_dict()
            measurement = receipt["measurement"]
            quality = receipt["quality"]
            assert isinstance(measurement, dict)
            assert isinstance(quality, dict)
            if work_item.kind is MeasurementOperationKind.SURFACE_DISTANCE:
                value_text = f"거리 {measurement['distance_mm_decimal']} mm"
            else:
                value_text = f"지름 {measurement['diameter_mm_decimal']} mm"
            status_text = (
                f"검증 표면 측정 기록 | {value_text} · "
                f"pick QC {quality['status']}"
            )
        else:  # pragma: no cover - closed enum guard
            raise ArtifactWorkbenchError(
                f"unsupported native measurement kind: {work_item.kind.value}"
            )

        measurement_controller = self._artifact_measurement_controller()

        def publish(transition: RecordBindingTransition) -> None:
            self._publish_artifact_session_projection(
                transition.candidate_session,
                project_path=self._current_project_path,
                fit_camera=False,
                workflow_transition=transition,
                expected_new_record_ids=(work_item.record_id,),
                status_text=status_text,
            )

        try:
            publication = measurement_controller.publish_result(
                work_item,
                result,
                publish,
            )
        except Exception:
            try:
                summary = measurement_controller.summary(work_item)
                if summary.state is MeasurementOperationState.RUNNING:
                    self._pending_native_measurement_publications[work_item.id] = (
                        work_item,
                        result,
                    )
                    self.section_panel.btn_native_measurement_retry.setEnabled(
                        self._native_measurement_ready()
                    )
                else:
                    self._pending_native_measurement_publications.pop(
                        work_item.id,
                        None,
                    )
            except Exception:
                _LOGGER.debug(
                    "Native measurement operation was already terminal",
                    exc_info=True,
                )
            raise
        self._pending_native_measurement_publications.pop(work_item.id, None)
        self.section_panel.btn_native_measurement_retry.setEnabled(
            bool(self._pending_native_measurement_publications)
            and self._native_measurement_ready()
        )
        if work_item.kind is MeasurementOperationKind.TILE_UNWRAP:
            try:
                selection = recipe.get("selection")
                if isinstance(selection, Mapping):
                    captured_selection = tuple(
                        int(value) for value in selection_face_indices(selection)
                    )
                    live_obj = getattr(self.viewport, "selected_obj", None)
                    live_selection = tuple(
                        sorted(
                            int(value)
                            for value in (
                                getattr(live_obj, "selected_faces", set()) or set()
                            )
                        )
                    )
                    if live_selection and live_selection == captured_selection:
                        self._set_object_selected_faces(live_obj, ())
            except Exception:
                _LOGGER.debug(
                    "Published tile selection could not be consumed",
                    exc_info=True,
                )
        if isinstance(computation, ArtifactRubbingComputation):
            self._preview_native_rubbing(
                publication.session,
                publication.record_id,
                computation.raster,
            )
        elif isinstance(computation, ArtifactTileUnwrapComputation):
            self._preview_native_tile_unwrap(
                publication.session,
                publication.record_id,
                computation.unwrap,
            )
        elif isinstance(computation, ArtifactVectorComputation):
            self._preview_native_vector_record(
                publication.session,
                publication.record_id,
            )
        elif isinstance(computation, ArtifactGeometryMetricsComputation):
            receipt = computation.receipt_dict()
            surface = receipt["surface_area"]
            volume = receipt["volume"]
            topology = receipt["topology"]
            bounds = receipt["bounds_grid"]
            assert isinstance(surface, dict)
            assert isinstance(volume, dict)
            assert isinstance(topology, dict)
            assert isinstance(bounds, dict)
            grid_um = int(receipt["coordinate_grid_um"])
            minimum = np.asarray(bounds["minimum"], dtype=np.int64)
            maximum = np.asarray(bounds["maximum"], dtype=np.int64)
            extents_mm = (maximum - minimum).astype(np.float64) * grid_um / 1000.0
            panel = getattr(self, "measure_panel", None)
            if panel is not None:
                panel.append_result(
                    f"[검증 제원] record={publication.record_id} | grid={grid_um} µm"
                )
                panel.append_result(
                    "- 크기: "
                    f"{extents_mm[0]:.3f}×{extents_mm[1]:.3f}×{extents_mm[2]:.3f} mm"
                )
                panel.append_result(
                    f"- 표면적: {surface['decimal_mm2']} mm²"
                )
                if volume["status"] == "available":
                    panel.append_result(
                        f"- 체적: {volume['decimal_mm3']} mm³ "
                        f"(exact {volume['exact_rational_mm3']})"
                    )
                else:
                    panel.append_result(
                        "- 체적: 산출하지 않음 | "
                        f"boundary={topology['boundary_edge_count']}, "
                        f"non-manifold={topology['non_manifold_edge_count']}, "
                        f"orientation={topology['orientation_mismatch_edge_count']}, "
                        f"components={topology['connected_component_count']}"
                    )
        elif isinstance(computation, ArtifactSurfaceMeasurementComputation):
            receipt = computation.receipt_dict()
            measurement = receipt["measurement"]
            quality = receipt["quality"]
            assert isinstance(measurement, dict)
            assert isinstance(quality, dict)
            panel = getattr(self, "measure_panel", None)
            if panel is not None:
                if computation.kind == "surface_distance":
                    title = "검증 거리"
                    value = f"{measurement['distance_mm_decimal']} mm"
                    detail = "3D Euclidean chord (측지 거리 아님)"
                else:
                    title = "검증 원 맞춤 지름"
                    value = f"{measurement['diameter_mm_decimal']} mm"
                    detail = (
                        f"samples={measurement['sample_count']}, "
                        f"plane RMS={measurement['plane_rms_residual_mm_decimal']} mm, "
                        f"radial RMS={measurement['radial_rms_residual_mm_decimal']} mm"
                    )
                panel.append_result(
                    f"[{title}] record={publication.record_id} | {value}"
                )
                panel.append_result(f"- 의미: {detail}")
                panel.append_result(
                    "- pick QC: "
                    f"{quality['status']} | max residual="
                    f"{quality['maximum_capture_residual_mm_decimal']} mm, "
                    f"pixel={quality['maximum_pixel_footprint_um']} µm, "
                    f"near-edge={quality['near_edge_anchor_count']}"
                )
                reasons = list(quality["review_reasons"])
                if reasons:
                    panel.append_result("- 검토 사유: " + ", ".join(reasons))
        else:  # pragma: no cover - closed computation union
            raise ArtifactWorkbenchError(
                "published an unsupported native measurement computation"
            )
        self._sync_native_cutline_controls(reset_offset=False)
        return publication.record_id

    def _native_measurement_publication_is_pending(
        self,
        work_item: ArtifactMeasurementWorkItem,
    ) -> bool:
        pending = self._pending_native_measurement_publications.get(work_item.id)
        if pending is None or pending[0] is not work_item:
            return False
        try:
            return (
                self._artifact_measurement_controller().summary(work_item).state
                is MeasurementOperationState.RUNNING
            )
        except Exception:
            self._pending_native_measurement_publications.pop(work_item.id, None)
            return False

    def _request_native_measurement_cancel(
        self,
        controller: ArtifactMeasurementController,
        work_item: ArtifactMeasurementWorkItem,
        *,
        label: str,
    ) -> None:
        try:
            summary = controller.cancel(work_item, reason="user_cancelled")
        except Exception:
            try:
                summary = controller.summary(work_item)
            except Exception:
                _LOGGER.debug(
                    "Native measurement cancellation lost its operation",
                    exc_info=True,
                )
                return
        if summary.state in {
            MeasurementOperationState.CANCELLING,
            MeasurementOperationState.CANCELLED,
        }:
            self.status_info.setText(
                f"{label} 취소 요청됨 · 안전한 계산 경계까지 기다리는 중"
            )

    @staticmethod
    def _verify_native_measurement_shutdown(
        controller: ArtifactMeasurementController,
        work_item: ArtifactMeasurementWorkItem,
    ) -> None:
        state = controller.summary(work_item).state
        if state not in {
            MeasurementOperationState.CANCELLED,
            MeasurementOperationState.COMPLETED,
            MeasurementOperationState.FAILED,
            MeasurementOperationState.STALE,
        }:
            raise ArtifactWorkbenchError(
                "joined measurement worker retained publication authority: "
                f"{state.value}"
            )

    @staticmethod
    def _execute_native_measurement_with_preflight(
        preflight: Callable[[], None],
        controller: ArtifactMeasurementController,
        work_item: ArtifactMeasurementWorkItem,
    ) -> ArtifactMeasurementResult:
        """Run canonical scene materialization on the measurement worker."""

        return controller.execute(work_item, preflight=preflight)

    def _native_measurement_callback_is_terminal(
        self,
        controller: ArtifactMeasurementController,
        work_item: ArtifactMeasurementWorkItem,
        *,
        label: str,
    ) -> bool:
        try:
            state = controller.summary(work_item).state
        except Exception:
            return False
        if state is MeasurementOperationState.CANCELLED:
            self._pending_native_measurement_publications.pop(work_item.id, None)
            self.status_info.setText(f"{label} 취소됨 | 기록을 만들지 않았습니다.")
            return True
        if state is MeasurementOperationState.STALE:
            self._pending_native_measurement_publications.pop(work_item.id, None)
            self.status_info.setText(
                f"{label} 결과 폐기 | 계산 중 문서·Align이 변경됐습니다."
            )
            return True
        return False

    def _prune_pending_native_measurement_publications(self) -> bool:
        controller = self._artifact_measurement_controller()
        for operation_id, (work_item, _result) in tuple(
            self._pending_native_measurement_publications.items()
        ):
            try:
                state = controller.summary(work_item).state
            except Exception:
                state = None
            if state is not MeasurementOperationState.RUNNING:
                self._pending_native_measurement_publications.pop(operation_id, None)
        pending = bool(self._pending_native_measurement_publications)
        panel = getattr(self, "section_panel", None)
        if panel is not None and hasattr(panel, "btn_native_measurement_retry"):
            panel.btn_native_measurement_retry.setEnabled(
                pending and self._native_measurement_ready()
            )
        return pending

    def on_native_measurement_retry_requested(self) -> None:
        """Retry the oldest completed computation without minting a new record ID."""

        if not self._prune_pending_native_measurement_publications():
            self.status_info.setText("다시 게시할 보류 실측 결과가 없습니다.")
            return
        _operation_id, (work_item, result) = next(
            iter(self._pending_native_measurement_publications.items())
        )
        try:
            self._publish_native_measurement_result(work_item, result)
        except Exception as exc:
            if self._report_artifact_authority_callback_failure(
                context="실측 결과 재게시 중 권위 확인 실패",
                detail=f"{type(exc).__name__}: {exc}",
            ):
                return
            if self._native_measurement_publication_is_pending(work_item):
                self.status_info.setText(
                    "실측 결과 게시 보류 | Open·scene 전환 완료 후 다시 시도"
                )
                QMessageBox.warning(
                    self,
                    "실측 결과 게시 보류",
                    "계산 결과와 예약 record ID는 그대로 보존했습니다. "
                    "현재 Open 또는 scene 전환이 끝난 뒤 다시 시도하세요.\n\n"
                    f"{type(exc).__name__}: {exc}",
                )
            else:
                self.status_info.setText("실측 결과 게시 권위 상실")
                QMessageBox.warning(
                    self,
                    "실측 결과 게시 실패",
                    f"{type(exc).__name__}: {exc}",
                )

    def _compute_and_commit_native_cutline(
        self,
        *,
        view: str,
        offset_mm: float,
        record_id: str | None = None,
        created_at: str | None = None,
        operator: str = "local-user",
    ) -> str:
        obj = self.viewport.selected_obj
        session = self._require_native_measurement_session(obj)
        self._validate_native_scene_for_save(session)
        frame = _native_cutline_frame(view, offset_mm)
        new_record_id = (
            f"record:cutline:{uuid.uuid4()}" if record_id is None else record_id
        )
        controller = self._artifact_measurement_controller()
        work_item = controller.begin_cutline(
            frame,
            record_id=new_record_id,
            created_at=(self._utc_seconds_now() if created_at is None else created_at),
            operator=operator,
        )
        result = controller.execute(work_item)
        return self._publish_native_measurement_result(work_item, result)

    def on_native_cutline_requested(self) -> None:
        try:
            view = str(
                self.section_panel.combo_native_cutline_view.currentData() or "top"
            )
            offset = float(self.section_panel.spin_native_cutline_offset.value())
            obj = self.viewport.selected_obj
            session = self._require_native_measurement_session(obj)
            preflight = self._capture_native_scene_preflight(session)
            controller = self._artifact_measurement_controller()
            work_item = controller.begin_cutline(
                _native_cutline_frame(view, offset),
                record_id=f"record:cutline:{uuid.uuid4()}",
                created_at=self._utc_seconds_now(),
                operator="local-user",
            )
        except Exception as exc:
            self.status_info.setText("검증 단면 준비 실패 | 기존 문서 유지")
            QMessageBox.warning(
                self,
                "Cutline 준비 실패",
                f"단면이 모호하거나 현재 Align과 맞지 않습니다.\n\n{type(exc).__name__}: {exc}",
            )
            return

        def on_done(result: object) -> None:
            if self._native_measurement_callback_is_terminal(
                controller,
                work_item,
                label="Cutline",
            ):
                return
            try:
                if not isinstance(result, ArtifactMeasurementResult):
                    raise ArtifactWorkbenchError("Cutline worker result is invalid")
                self._publish_native_measurement_result(work_item, result)
            except Exception as exc:
                if self._report_artifact_authority_callback_failure(
                    context="Cutline 결과 게시 중 권위 확인 실패",
                    detail=f"{type(exc).__name__}: {exc}",
                ):
                    return
                pending = self._native_measurement_publication_is_pending(work_item)
                self.status_info.setText(
                    "Cutline 결과 게시 보류 | 재시도 버튼 사용"
                    if pending
                    else "Cutline 결과 폐기 | 현재 문서 유지"
                )
                QMessageBox.warning(
                    self,
                    "Cutline 결과 게시 보류" if pending else "Cutline 결과 폐기",
                    f"{type(exc).__name__}: {exc}",
                )

        def on_failed(message: str) -> None:
            if self._report_artifact_authority_callback_failure(
                context="Cutline worker 종료 콜백",
                detail=str(message),
            ):
                return
            if self._native_measurement_callback_is_terminal(
                controller,
                work_item,
                label="Cutline",
            ):
                return
            self.status_info.setText("검증 단면 계산 실패 | 기존 문서 유지")
            QMessageBox.warning(
                self,
                "Cutline 계산 실패",
                self._format_error_message("단면 계산 중 오류가 발생했습니다:", message),
            )

        self.status_info.setText("Cutline 계산 중 · canonical mm 재투영...")
        started = self._start_task(
            title="Cutline",
            label="재현 가능한 단면 벡터 계산 중...",
            thread=TaskThread(
                "native_cutline",
                lambda: self._execute_native_measurement_with_preflight(
                    preflight,
                    controller,
                    work_item,
                ),
            ),
            on_done=on_done,
            on_failed=on_failed,
            on_cancel_requested=lambda: self._request_native_measurement_cancel(
                controller,
                work_item,
                label="Cutline",
            ),
            on_shutdown_joined=lambda: self._verify_native_measurement_shutdown(
                controller,
                work_item,
            ),
        )
        if not started:
            controller.cancel(work_item, reason="task_not_started")

    def _compute_and_commit_native_outline(
        self,
        *,
        view: str,
        precision_grid_mm: float,
        record_id: str | None = None,
        created_at: str | None = None,
        operator: str = "local-user",
    ) -> str:
        obj = self.viewport.selected_obj
        session = self._require_native_measurement_session(obj)
        self._validate_native_scene_for_save(session)
        new_record_id = (
            f"record:outline:{view}:{uuid.uuid4()}" if record_id is None else record_id
        )
        controller = self._artifact_measurement_controller()
        work_item = controller.begin_outline(
            view,
            precision_grid_mm=precision_grid_mm,
            record_id=new_record_id,
            created_at=(self._utc_seconds_now() if created_at is None else created_at),
            operator=operator,
        )
        result = controller.execute(work_item)
        return self._publish_native_measurement_result(work_item, result)

    def on_native_outline_requested(self) -> None:
        try:
            view = str(
                self.section_panel.combo_native_outline_view.currentData() or "top"
            )
            precision_grid_mm = float(
                self.section_panel.spin_native_outline_grid.value()
            )
            obj = self.viewport.selected_obj
            session = self._require_native_measurement_session(obj)
            preflight = self._capture_native_scene_preflight(session)
            progress = self._native_record_workflow_progress()
            if not progress.outline.enabled:
                raise ArtifactSessionError(
                    "Outline requires READY + FRESH Top, Front, and Right "
                    "Cutline records"
                )
            controller = self._artifact_measurement_controller()
            work_item = controller.begin_outline(
                view,
                precision_grid_mm=precision_grid_mm,
                record_id=f"record:outline:{view}:{uuid.uuid4()}",
                created_at=self._utc_seconds_now(),
                operator="local-user",
            )
        except Exception as exc:
            self.status_info.setText("검증 외곽 준비 실패 | 기존 문서 유지")
            QMessageBox.warning(
                self,
                "Outline 준비 실패",
                "외곽 위상이 유효하지 않거나 현재 Align과 맞지 않습니다.\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            return

        def on_done(result: object) -> None:
            if self._native_measurement_callback_is_terminal(
                controller,
                work_item,
                label="Outline",
            ):
                return
            try:
                if not isinstance(result, ArtifactMeasurementResult):
                    raise ArtifactWorkbenchError("Outline worker result is invalid")
                self._publish_native_measurement_result(work_item, result)
            except Exception as exc:
                if self._report_artifact_authority_callback_failure(
                    context="Outline 결과 게시 중 권위 확인 실패",
                    detail=f"{type(exc).__name__}: {exc}",
                ):
                    return
                pending = self._native_measurement_publication_is_pending(work_item)
                self.status_info.setText(
                    "Outline 결과 게시 보류 | 재시도 버튼 사용"
                    if pending
                    else "Outline 결과 폐기 | 현재 문서 유지"
                )
                QMessageBox.warning(
                    self,
                    "Outline 결과 게시 보류" if pending else "Outline 결과 폐기",
                    f"{type(exc).__name__}: {exc}",
                )

        def on_failed(message: str) -> None:
            if self._report_artifact_authority_callback_failure(
                context="Outline worker 종료 콜백",
                detail=str(message),
            ):
                return
            if self._native_measurement_callback_is_terminal(
                controller,
                work_item,
                label="Outline",
            ):
                return
            self.status_info.setText("검증 외곽 계산 실패 | 기존 문서 유지")
            QMessageBox.warning(
                self,
                "Outline 계산 실패",
                self._format_error_message("외곽 계산 중 오류가 발생했습니다:", message),
            )

        self.status_info.setText("Outline 계산 중 · canonical mm 재투영...")
        started = self._start_task(
            title="Outline",
            label="재현 가능한 6면 외곽 벡터 계산 중...",
            thread=TaskThread(
                "native_outline",
                lambda: self._execute_native_measurement_with_preflight(
                    preflight,
                    controller,
                    work_item,
                ),
            ),
            on_done=on_done,
            on_failed=on_failed,
            on_cancel_requested=lambda: self._request_native_measurement_cancel(
                controller,
                work_item,
                label="Outline",
            ),
            on_shutdown_joined=lambda: self._verify_native_measurement_shutdown(
                controller,
                work_item,
            ),
        )
        if not started:
            controller.cancel(work_item, reason="task_not_started")

    @staticmethod
    def _cancel_native_export_if_staged(
        controller: _NativeExportController,
        work_item: _NativeExportWorkItem,
        *,
        reason: str,
    ) -> str | None:
        """Best-effort cleanup for a GUI callback that cannot consume its stage."""

        try:
            if controller.summary(work_item).state is ArtifactExportState.STAGED:
                controller.cancel(work_item, reason=reason)
        except Exception as exc:
            return str(exc)
        return None

    @staticmethod
    def _cancel_unstarted_native_export(
        controller: _NativeExportController,
        work_item: _NativeExportWorkItem,
        *,
        reason: str,
    ) -> str | None:
        """Release a READY destination reservation when no worker owns it."""

        try:
            controller.cancel(work_item, reason=reason)
        except Exception as exc:
            return str(exc)
        return None

    def _request_native_export_cancel(
        self,
        controller: _NativeExportController,
        work_item: _NativeExportWorkItem,
        *,
        label: str,
    ) -> None:
        """Revoke publication authority before waiting for export staging to exit."""

        try:
            state = controller.summary(work_item).state
            if state in {
                ArtifactExportState.CANCELLED,
                ArtifactExportState.COMPLETED,
                ArtifactExportState.FAILED,
                ArtifactExportState.STALE,
            }:
                return
            controller.cancel(work_item, reason="user_cancelled")
        except ArtifactExportError:
            try:
                if controller.summary(work_item).state in {
                    ArtifactExportState.CANCELLED,
                    ArtifactExportState.COMPLETED,
                    ArtifactExportState.FAILED,
                    ArtifactExportState.STALE,
                }:
                    return
            except Exception:
                pass
            raise
        try:
            self.status_info.setText(
                f"{label} 취소 요청됨 | 임시 패키지 안전 정리 대기"
            )
        except Exception:
            pass

    def _native_export_callback_is_cancelled(
        self,
        controller: _NativeExportController,
        work_item: _NativeExportWorkItem,
        *,
        label: str,
    ) -> bool:
        try:
            cancelled = (
                controller.summary(work_item).state
                is ArtifactExportState.CANCELLED
            )
        except Exception:
            return False
        if cancelled:
            try:
                self.status_info.setText(
                    f"{label} 취소 완료 | 목적지에 패키지를 게시하지 않음"
                )
            except Exception:
                pass
        return cancelled

    @staticmethod
    def _verify_native_export_shutdown(
        controller: _NativeExportController,
        work_item: _NativeExportWorkItem,
    ) -> None:
        summary = controller.summary(work_item)
        if summary.state in {
            ArtifactExportState.CANCELLED,
            ArtifactExportState.COMPLETED,
            ArtifactExportState.STALE,
        }:
            return
        detail = f": {summary.message}" if summary.message else ""
        raise ArtifactExportError(
            "joined export worker did not prove safe staging cleanup "
            f"({summary.state.value}){detail}"
        )

    def _report_native_export_publication(
        self,
        publication: _NativeExportPublication,
        *,
        artifact_label: str,
    ) -> None:
        if self._report_artifact_authority_callback_failure(
            context=f"{artifact_label} 패키지 게시 완료 콜백",
            detail="문서 권위가 fault 상태인 동안 성공 표시를 차단했습니다.",
        ):
            return
        destination = publication.destination
        if publication.durability_confirmed:
            self.status_info.setText(
                f"1:1 {artifact_label} + provenance 저장: {destination.name}"
            )
            return
        warning = publication.warning_message or (
            "패키지는 원자적으로 게시됐지만 crash durability를 확인하지 못했습니다."
        )
        self.status_info.setText(
            f"{artifact_label} 패키지 저장됨 | crash durability 미확정"
        )
        QMessageBox.warning(
            self,
            "내보내기 내구성 경고",
            f"패키지는 저장됐지만 디렉터리 동기화를 확인하지 못했습니다.\n\n{warning}",
        )

    def _export_native_vector_record(
        self,
        destination: str | os.PathLike[str],
        *,
        record_id: str | None = None,
    ) -> Path:
        session = getattr(self, "_artifact_session", None)
        if not isinstance(session, ArtifactSession):
            raise ArtifactVectorExportError("no active ArtifactDocument session")
        try:
            self._artifact_workbench_controller().require_stable_session(
                session,
                measurement=True,
            )
        except ArtifactWorkbenchError as exc:
            raise ArtifactVectorExportError(str(exc)) from exc
        record = (
            session.document.record_index.get(record_id)
            if record_id is not None
            else self._current_native_vector_record()
        )
        if record is None or not self._native_vector_record_is_exportable(
            session, record
        ):
            raise ArtifactVectorExportError("no READY + FRESH vector record to export")
        controller = self._artifact_export_controller()
        work_item: ArtifactExportWorkItem | None = None
        result: ArtifactExportResult | None = None
        try:
            work_item = controller.begin_vector(destination, record.id)
            result = controller.execute(work_item)
            return controller.publish_result(work_item, result).destination
        except WorkflowBusyError as exc:
            if work_item is not None and result is not None:
                controller.discard_result(
                    work_item,
                    result,
                    reason="synchronous export blocked by pending Open",
                )
            raise ArtifactVectorExportError(str(exc)) from exc
        except ArtifactExportError as exc:
            raise ArtifactVectorExportError(str(exc)) from exc

    def on_native_vector_export_requested(self) -> None:
        session = getattr(self, "_artifact_session", None)
        if not isinstance(session, ArtifactSession):
            self.status_info.setText("내보낼 ArtifactDocument가 없습니다.")
            return
        try:
            self._artifact_workbench_controller().require_stable_session(
                session,
                measurement=True,
            )
        except ArtifactWorkbenchError as exc:
            self.status_info.setText(f"벡터 내보내기 차단: {exc}")
            return
        record = self._current_native_vector_record()
        if record is None:
            self.status_info.setText("내보낼 READY + FRESH 벡터 기록이 없습니다.")
            return
        source_path = Path(str(self.current_filepath or "artifact"))
        default_path = source_path.with_name(
            f"{source_path.stem}-{record.type.split('.')[1]}{VECTOR_EXPORT_DIRECTORY_SUFFIX}"
        )
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "1:1 벡터 패키지 저장",
            str(default_path),
            "ArchMeshRubbing Vector Package (*.amr-vector)",
        )
        if not selected:
            return
        if not selected.endswith(VECTOR_EXPORT_DIRECTORY_SUFFIX):
            selected += VECTOR_EXPORT_DIRECTORY_SUFFIX
        try:
            controller = self._artifact_export_controller()
            work_item = controller.begin_vector(selected, record.id)
        except Exception as exc:
            self.status_info.setText("벡터 패키지 준비 실패")
            QMessageBox.warning(
                self,
                "벡터 내보내기 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return

        def on_done(value: object) -> None:
            try:
                if self._artifact_export_controller() is not controller:
                    raise ArtifactExportError(
                        "vector export controller authority was replaced"
                    )
                if not isinstance(value, ArtifactExportResult):
                    raise ArtifactExportError("vector export worker result is invalid")
                publication = controller.publish_result(work_item, value)
            except WorkflowBusyError as exc:
                cleanup_message = ""
                try:
                    if isinstance(value, ArtifactExportResult):
                        controller.discard_result(
                            work_item,
                            value,
                            reason="pending Open blocked final export publication",
                        )
                except Exception as cleanup_exc:
                    cleanup_message = f"\n임시 패키지 정리 확인 실패: {cleanup_exc}"
                if self._report_artifact_authority_callback_failure(
                    context="SVG 패키지 최종 게시 중 권위 확인 실패",
                    detail=f"{type(exc).__name__}: {exc}{cleanup_message}",
                ):
                    return
                self.status_info.setText(
                    "벡터 게시 취소 | Open 완료 후 다시 내보내세요"
                )
                QMessageBox.warning(
                    self,
                    "벡터 내보내기 게시 취소",
                    f"{type(exc).__name__}: {exc}{cleanup_message}",
                )
                return
            except Exception as exc:
                cleanup_error = self._cancel_native_export_if_staged(
                    controller,
                    work_item,
                    reason="invalid or rejected vector export callback",
                )
                detail = f"{type(exc).__name__}: {exc}" + (
                    f"\n임시 패키지 정리 확인 실패: {cleanup_error}"
                    if cleanup_error
                    else ""
                )
                if self._report_artifact_authority_callback_failure(
                    context="SVG 패키지 게시 콜백 실패",
                    detail=detail,
                ):
                    return
                self.status_info.setText("벡터 패키지 저장 실패")
                QMessageBox.warning(
                    self,
                    "벡터 내보내기 실패",
                    f"{type(exc).__name__}: {exc}"
                    + (
                        f"\n임시 패키지 정리 확인 실패: {cleanup_error}"
                        if cleanup_error
                        else ""
                    ),
                )
                return
            self._report_native_export_publication(publication, artifact_label="SVG")

        def on_failed(message: str) -> None:
            if self._native_export_callback_is_cancelled(
                controller,
                work_item,
                label="벡터 내보내기",
            ):
                return
            if self._report_artifact_authority_callback_failure(
                context="SVG 패키지 worker 종료 콜백",
                detail=str(message),
            ):
                return
            self.status_info.setText("벡터 패키지 생성 실패")
            QMessageBox.warning(
                self,
                "벡터 내보내기 실패",
                self._format_error_message("패키지 생성 중 오류가 발생했습니다:", message),
            )

        try:
            started = self._start_task(
                title="벡터 내보내기",
                label="1:1 SVG와 provenance 패키지를 검증하는 중...",
                thread=TaskThread(
                    "export_native_vector",
                    lambda: controller.execute(work_item),
                ),
                on_done=on_done,
                on_failed=on_failed,
                on_cancel_requested=lambda: self._request_native_export_cancel(
                    controller,
                    work_item,
                    label="벡터 내보내기",
                ),
                on_shutdown_joined=lambda: self._verify_native_export_shutdown(
                    controller,
                    work_item,
                ),
            )
        except Exception as exc:
            cleanup_error = self._cancel_unstarted_native_export(
                controller,
                work_item,
                reason="task_start_failed",
            )
            detail = f"{type(exc).__name__}: {exc}" + (
                f"\n내보내기 예약 해제 확인 실패: {cleanup_error}"
                if cleanup_error
                else ""
            )
            if self._report_artifact_authority_callback_failure(
                context="SVG 패키지 worker 시작 실패",
                detail=detail,
            ):
                return
            self.status_info.setText("벡터 패키지 작업 시작 실패")
            QMessageBox.warning(self, "벡터 내보내기 실패", detail)
            return
        if not started:
            cleanup_error = self._cancel_unstarted_native_export(
                controller,
                work_item,
                reason="task_not_started",
            )
            if cleanup_error:
                if self._report_artifact_authority_callback_failure(
                    context="SVG 패키지 worker 미시작",
                    detail=f"내보내기 예약 해제 확인 실패: {cleanup_error}",
                ):
                    return
                self.status_info.setText("벡터 내보내기 예약 해제 확인 실패")
                QMessageBox.warning(
                    self,
                    "벡터 내보내기 정리 실패",
                    cleanup_error,
                )

    def _native_rubbing_options_from_panel(self) -> dict[str, Any]:
        panel = self.section_panel
        return {
            "view": str(panel.combo_native_rubbing_view.currentData() or "top"),
            "pixels_per_mm": int(panel.spin_native_rubbing_pixels_per_mm.value()),
            "margin_um": int(panel.spin_native_rubbing_margin_um.value()),
            "reference_radius_um": int(
                panel.spin_native_rubbing_reference_radius_um.value()
            ),
            "depth_quantization_um": int(
                panel.spin_native_rubbing_depth_quantization_um.value()
            ),
            "black_point_um": int(panel.spin_native_rubbing_black_point_um.value()),
            "ink_strength_percent": int(panel.spin_native_rubbing_strength.value()),
            "relief_polarity": str(
                panel.combo_native_rubbing_polarity.currentData()
                or DEFAULT_RUBBING_POLARITY
            ),
        }

    def _compute_and_commit_native_rubbing(
        self,
        *,
        options: dict[str, Any],
        record_id: str | None = None,
        created_at: str | None = None,
        operator: str = "local-user",
    ) -> str:
        obj = self.viewport.selected_obj
        session = self._require_native_measurement_session(obj)
        self._validate_native_scene_for_save(session)
        new_record_id = (
            f"record:rubbing:{options.get('view', 'top')}:{uuid.uuid4()}"
            if record_id is None
            else record_id
        )
        controller = self._artifact_measurement_controller()
        work_item = controller.begin_rubbing(
            **dict(options),
            record_id=new_record_id,
            created_at=(self._utc_seconds_now() if created_at is None else created_at),
            operator=operator,
        )
        result = controller.execute(work_item)
        return self._publish_native_measurement_result(work_item, result)

    def on_native_rubbing_requested(self) -> None:
        try:
            obj = self.viewport.selected_obj
            session = self._require_native_measurement_session(obj)
            preflight = self._capture_native_scene_preflight(session)
            progress = self._native_record_workflow_progress()
            if not progress.rubbing.enabled:
                raise ArtifactSessionError(
                    "Digital Rubbing requires complete READY + FRESH "
                    "Cutline and six-view Outline records"
                )
            options = self._native_rubbing_options_from_panel()
            record_id = f"record:rubbing:{options['view']}:{uuid.uuid4()}"
            created_at = self._utc_seconds_now()
            controller = self._artifact_measurement_controller()
            work_item = controller.begin_rubbing(
                **options,
                record_id=record_id,
                created_at=created_at,
                operator="local-user",
            )
        except Exception as exc:
            self.status_info.setText("Digital Rubbing 준비 실패 | 기존 문서 유지")
            QMessageBox.warning(
                self,
                "Digital Rubbing 준비 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return

        def task():
            return self._execute_native_measurement_with_preflight(
                preflight,
                controller,
                work_item,
            )

        def on_done(result: object) -> None:
            if self._native_measurement_callback_is_terminal(
                controller,
                work_item,
                label="Digital Rubbing",
            ):
                return
            try:
                if not isinstance(result, ArtifactMeasurementResult):
                    raise ArtifactWorkbenchError(
                        "Digital Rubbing worker result is invalid"
                    )
                self._publish_native_measurement_result(work_item, result)
            except Exception as exc:
                if self._report_artifact_authority_callback_failure(
                    context="Digital Rubbing 결과 게시 중 권위 확인 실패",
                    detail=f"{type(exc).__name__}: {exc}",
                ):
                    return
                pending = self._native_measurement_publication_is_pending(work_item)
                self.status_info.setText(
                    "Digital Rubbing 결과 게시 보류 | 재시도 버튼 사용"
                    if pending
                    else "늦은 Digital Rubbing 결과 폐기 | 현재 문서 유지"
                )
                QMessageBox.warning(
                    self,
                    (
                        "Digital Rubbing 결과 게시 보류"
                        if pending
                        else "Digital Rubbing 결과 폐기"
                    ),
                    f"{type(exc).__name__}: {exc}",
                )

        def on_failed(message: str) -> None:
            if self._report_artifact_authority_callback_failure(
                context="Digital Rubbing worker 종료 콜백",
                detail=str(message),
            ):
                return
            if self._native_measurement_callback_is_terminal(
                controller,
                work_item,
                label="Digital Rubbing",
            ):
                return
            self.status_info.setText("Digital Rubbing 계산 실패 | 기존 문서 유지")
            QMessageBox.warning(
                self,
                "Digital Rubbing 계산 실패",
                self._format_error_message("탁본 계산 중 오류가 발생했습니다:", message),
            )

        self.status_info.setText("Digital Rubbing 계산 중 · 원본 canonical mm 재투영...")
        started = self._start_task(
            title="Digital Rubbing",
            label="재현 가능한 1:1 탁본 raster 계산 중...",
            thread=TaskThread("native_digital_rubbing", task),
            on_done=on_done,
            on_failed=on_failed,
            on_cancel_requested=lambda: self._request_native_measurement_cancel(
                controller,
                work_item,
                label="Digital Rubbing",
            ),
            on_shutdown_joined=lambda: self._verify_native_measurement_shutdown(
                controller,
                work_item,
            ),
        )
        if not started:
            controller.cancel(work_item, reason="task_not_started")

    @staticmethod
    def _recompute_native_rubbing_record(session: ArtifactSession, record):
        if not MainWindow._native_rubbing_record_is_exportable(session, record):
            raise ArtifactRubbingExportError(
                "no READY + FRESH Digital Rubbing record to export"
            )
        snapshot = session.projection_snapshot()
        estimate = estimate_digital_rubbing_resources(
            session.source_mesh.vertices,
            session.source_mesh.faces,
            record.recipe,
            source_to_world_mm_matrix4x4=snapshot.matrix4x4,
            uv_coords=session.source_mesh.uv_coords,
            texture=session.source_mesh.texture,
        )
        if estimate.estimated_peak_bytes > DEFAULT_RUBBING_MEMORY_BUDGET_BYTES:
            raise ArtifactRubbingExportError(
                "Digital Rubbing recomputation exceeds the local 1 GiB memory budget"
            )
        computation = compute_artifact_rubbing_from_recipe(session, record.recipe)
        require_current_rubbing_computation(session, computation)
        if computation.raster.receipt() != rubbing_receipt_from_record(record):
            raise ArtifactRubbingExportError(
                "recomputed Digital Rubbing raster does not match its record receipt"
            )
        return computation.raster

    def _export_native_rubbing_record(
        self,
        destination: str | os.PathLike[str],
        *,
        record_id: str | None = None,
    ) -> Path:
        session = getattr(self, "_artifact_session", None)
        if not isinstance(session, ArtifactSession):
            raise ArtifactRubbingExportError("no active ArtifactDocument session")
        if self._artifact_measurement_controller().active_summaries:
            raise ArtifactRubbingExportError(
                "active measurement work owns the shared raster memory budget"
            )
        try:
            self._artifact_workbench_controller().require_stable_session(
                session,
                measurement=True,
            )
        except ArtifactWorkbenchError as exc:
            raise ArtifactRubbingExportError(str(exc)) from exc
        record = (
            session.document.record_index.get(record_id)
            if record_id is not None
            else self._current_native_rubbing_record()
        )
        if record is None or not self._native_rubbing_record_is_exportable(
            session,
            record,
        ):
            raise ArtifactRubbingExportError(
                "no READY + FRESH Digital Rubbing record to export"
            )
        controller = self._artifact_export_controller()
        work_item: ArtifactExportWorkItem | None = None
        result: ArtifactExportResult | None = None
        try:
            work_item = controller.begin_rubbing(destination, record.id)
            result = controller.execute(work_item)
            return controller.publish_result(work_item, result).destination
        except WorkflowBusyError as exc:
            if work_item is not None and result is not None:
                controller.discard_result(
                    work_item,
                    result,
                    reason="synchronous export blocked by pending Open",
                )
            raise ArtifactRubbingExportError(str(exc)) from exc
        except ArtifactExportError as exc:
            raise ArtifactRubbingExportError(str(exc)) from exc

    def on_native_rubbing_export_requested(self) -> None:
        session = getattr(self, "_artifact_session", None)
        record = self._current_native_rubbing_record()
        if not isinstance(session, ArtifactSession) or record is None:
            self.status_info.setText("내보낼 READY + FRESH 탁본 기록이 없습니다.")
            return
        if self._artifact_measurement_controller().active_summaries:
            self.status_info.setText(
                "진행·보류 중인 실측 결과가 raster 예산을 사용하고 있습니다."
            )
            return
        try:
            self._artifact_workbench_controller().require_stable_session(
                session,
                measurement=True,
            )
        except ArtifactWorkbenchError as exc:
            self.status_info.setText(f"Digital Rubbing 내보내기 차단: {exc}")
            return
        source_path = Path(str(self.current_filepath or "artifact"))
        default_path = source_path.with_name(
            f"{source_path.stem}-digital-rubbing{RUBBING_EXPORT_DIRECTORY_SUFFIX}"
        )
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "1:1 Digital Rubbing 패키지 저장",
            str(default_path),
            "ArchMeshRubbing Rubbing Package (*.amr-rubbing)",
        )
        if not selected:
            return
        if not selected.endswith(RUBBING_EXPORT_DIRECTORY_SUFFIX):
            selected += RUBBING_EXPORT_DIRECTORY_SUFFIX
        try:
            controller = self._artifact_export_controller()
            work_item = controller.begin_rubbing(selected, record.id)
        except Exception as exc:
            self.status_info.setText("Digital Rubbing 내보내기 준비 실패")
            QMessageBox.warning(
                self,
                "Digital Rubbing 내보내기 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return

        def on_done(value: object) -> None:
            try:
                if self._artifact_export_controller() is not controller:
                    raise ArtifactExportError(
                        "rubbing export controller authority was replaced"
                    )
                if not isinstance(value, ArtifactExportResult):
                    raise ArtifactExportError("rubbing export worker result is invalid")
                publication = controller.publish_result(work_item, value)
            except WorkflowBusyError as exc:
                cleanup_message = ""
                try:
                    if isinstance(value, ArtifactExportResult):
                        controller.discard_result(
                            work_item,
                            value,
                            reason="pending Open blocked final export publication",
                        )
                except Exception as cleanup_exc:
                    cleanup_message = f"\n임시 패키지 정리 확인 실패: {cleanup_exc}"
                if self._report_artifact_authority_callback_failure(
                    context="Digital Rubbing 패키지 최종 게시 중 권위 확인 실패",
                    detail=f"{type(exc).__name__}: {exc}{cleanup_message}",
                ):
                    return
                self.status_info.setText(
                    "탁본 게시 취소 | Open 완료 후 다시 내보내세요"
                )
                QMessageBox.warning(
                    self,
                    "Digital Rubbing 게시 취소",
                    f"{type(exc).__name__}: {exc}{cleanup_message}",
                )
                return
            except Exception as exc:
                cleanup_error = self._cancel_native_export_if_staged(
                    controller,
                    work_item,
                    reason="invalid or rejected rubbing export callback",
                )
                detail = f"{type(exc).__name__}: {exc}" + (
                    f"\n임시 패키지 정리 확인 실패: {cleanup_error}"
                    if cleanup_error
                    else ""
                )
                if self._report_artifact_authority_callback_failure(
                    context="Digital Rubbing 패키지 게시 콜백 실패",
                    detail=detail,
                ):
                    return
                self.status_info.setText("Digital Rubbing 패키지 저장 실패")
                QMessageBox.warning(
                    self,
                    "Digital Rubbing 내보내기 실패",
                    f"{type(exc).__name__}: {exc}"
                    + (
                        f"\n임시 패키지 정리 확인 실패: {cleanup_error}"
                        if cleanup_error
                        else ""
                    ),
                )
                return
            self._report_native_export_publication(publication, artifact_label="PNG")

        def on_failed(message: str) -> None:
            if self._native_export_callback_is_cancelled(
                controller,
                work_item,
                label="Digital Rubbing 내보내기",
            ):
                return
            if self._report_artifact_authority_callback_failure(
                context="Digital Rubbing export worker 종료 콜백",
                detail=str(message),
            ):
                return
            self.status_info.setText("Digital Rubbing 패키지 저장 실패")
            QMessageBox.warning(
                self,
                "Digital Rubbing 내보내기 실패",
                self._format_error_message("패키지 생성 중 오류가 발생했습니다:", message),
            )

        try:
            started = self._start_task(
                title="Digital Rubbing 내보내기",
                label="record recipe 재계산 및 1:1 PNG 검증 중...",
                thread=TaskThread(
                    "export_native_digital_rubbing",
                    lambda: controller.execute(work_item),
                ),
                on_done=on_done,
                on_failed=on_failed,
                on_cancel_requested=lambda: self._request_native_export_cancel(
                    controller,
                    work_item,
                    label="Digital Rubbing 내보내기",
                ),
                on_shutdown_joined=lambda: self._verify_native_export_shutdown(
                    controller,
                    work_item,
                ),
            )
        except Exception as exc:
            cleanup_error = self._cancel_unstarted_native_export(
                controller,
                work_item,
                reason="task_start_failed",
            )
            detail = f"{type(exc).__name__}: {exc}" + (
                f"\n내보내기 예약 해제 확인 실패: {cleanup_error}"
                if cleanup_error
                else ""
            )
            if self._report_artifact_authority_callback_failure(
                context="Digital Rubbing export worker 시작 실패",
                detail=detail,
            ):
                return
            self.status_info.setText("Digital Rubbing 패키지 작업 시작 실패")
            QMessageBox.warning(self, "Digital Rubbing 내보내기 실패", detail)
            return
        if not started:
            cleanup_error = self._cancel_unstarted_native_export(
                controller,
                work_item,
                reason="task_not_started",
            )
            if cleanup_error:
                if self._report_artifact_authority_callback_failure(
                    context="Digital Rubbing export worker 미시작",
                    detail=f"내보내기 예약 해제 확인 실패: {cleanup_error}",
                ):
                    return
                self.status_info.setText(
                    "Digital Rubbing 내보내기 예약 해제 확인 실패"
                )
                QMessageBox.warning(
                    self,
                    "Digital Rubbing 내보내기 정리 실패",
                    cleanup_error,
                )

    def on_native_survey_export_requested(self) -> None:
        """Publish the completed 3/6/6 workflow as one atomic directory."""

        session = getattr(self, "_artifact_session", None)
        if not isinstance(session, ArtifactSession):
            self.status_info.setText("내보낼 ArtifactDocument가 없습니다.")
            return
        if self._artifact_measurement_controller().active_summaries:
            self.status_info.setText(
                "진행·보류 중인 실측 결과가 있어 완료 실측 묶음을 만들 수 없습니다."
            )
            return
        progress = self._native_record_workflow_progress()
        if not progress.rubbing.complete:
            self.status_info.setText(
                "완료 실측 묶음은 Cutline 3/3 · Outline 6/6 · 탁본 6/6이 필요합니다."
            )
            return
        try:
            self._artifact_workbench_controller().require_stable_session(
                session,
                measurement=True,
            )
        except ArtifactWorkbenchError as exc:
            self.status_info.setText(f"완료 실측 묶음 내보내기 차단: {exc}")
            return

        source_path = Path(str(self.current_filepath or "artifact"))
        default_path = source_path.with_name(
            f"{source_path.stem}-complete{SURVEY_EXPORT_DIRECTORY_SUFFIX}"
        )
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "완료 실측 15개 원자 묶음 저장",
            str(default_path),
            "ArchMeshRubbing Survey Package (*.amr-survey)",
        )
        if not selected:
            return
        if not selected.endswith(SURVEY_EXPORT_DIRECTORY_SUFFIX):
            selected += SURVEY_EXPORT_DIRECTORY_SUFFIX
        try:
            controller = self._artifact_survey_export_controller()
            work_item = controller.begin(selected)
        except Exception as exc:
            self.status_info.setText("완료 실측 묶음 준비 실패")
            QMessageBox.warning(
                self,
                "완료 실측 묶음 내보내기 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return

        def on_done(value: object) -> None:
            try:
                if self._artifact_survey_export_controller() is not controller:
                    raise ArtifactExportError(
                        "complete-survey export controller authority was replaced"
                    )
                if not isinstance(value, ArtifactSurveyExportResult):
                    raise ArtifactExportError(
                        "complete-survey export worker result is invalid"
                    )
                publication = controller.publish_result(work_item, value)
            except WorkflowBusyError as exc:
                cleanup_message = ""
                try:
                    if isinstance(value, ArtifactSurveyExportResult):
                        controller.discard_result(
                            work_item,
                            value,
                            reason="pending Open blocked final survey publication",
                        )
                except Exception as cleanup_exc:
                    cleanup_message = f"\n임시 묶음 정리 확인 실패: {cleanup_exc}"
                detail = f"{type(exc).__name__}: {exc}{cleanup_message}"
                if self._report_artifact_authority_callback_failure(
                    context="완료 실측 묶음 최종 게시 중 권위 확인 실패",
                    detail=detail,
                ):
                    return
                self.status_info.setText(
                    "완료 실측 묶음 게시 취소 | Open 완료 후 다시 내보내세요"
                )
                QMessageBox.warning(
                    self,
                    "완료 실측 묶음 게시 취소",
                    detail,
                )
                return
            except Exception as exc:
                cleanup_error = self._cancel_native_export_if_staged(
                    controller,
                    work_item,
                    reason="invalid or rejected complete-survey export callback",
                )
                detail = f"{type(exc).__name__}: {exc}" + (
                    f"\n임시 묶음 정리 확인 실패: {cleanup_error}"
                    if cleanup_error
                    else ""
                )
                if self._report_artifact_authority_callback_failure(
                    context="완료 실측 묶음 게시 콜백 실패",
                    detail=detail,
                ):
                    return
                self.status_info.setText("완료 실측 묶음 저장 실패")
                QMessageBox.warning(
                    self,
                    "완료 실측 묶음 내보내기 실패",
                    detail,
                )
                return
            self._report_native_export_publication(
                publication,
                artifact_label="완료 실측 15개",
            )

        def on_failed(message: str) -> None:
            if self._native_export_callback_is_cancelled(
                controller,
                work_item,
                label="완료 실측 묶음 내보내기",
            ):
                return
            if self._report_artifact_authority_callback_failure(
                context="완료 실측 묶음 worker 종료 콜백",
                detail=str(message),
            ):
                return
            self.status_info.setText("완료 실측 묶음 생성 실패")
            QMessageBox.warning(
                self,
                "완료 실측 묶음 내보내기 실패",
                self._format_error_message("묶음 생성 중 오류가 발생했습니다:", message),
            )

        try:
            started = self._start_task(
                title="완료 실측 묶음 내보내기",
                label="3/6/6 기록 재현 · 15개 자식 검증 · 원자 게시 준비 중...",
                thread=TaskThread(
                    "export_native_complete_survey",
                    lambda: controller.execute(work_item),
                ),
                on_done=on_done,
                on_failed=on_failed,
                on_cancel_requested=lambda: self._request_native_export_cancel(
                    controller,
                    work_item,
                    label="완료 실측 묶음 내보내기",
                ),
                on_shutdown_joined=lambda: self._verify_native_export_shutdown(
                    controller,
                    work_item,
                ),
            )
        except Exception as exc:
            cleanup_error = self._cancel_unstarted_native_export(
                controller,
                work_item,
                reason="task_start_failed",
            )
            detail = f"{type(exc).__name__}: {exc}" + (
                f"\n내보내기 예약 해제 확인 실패: {cleanup_error}"
                if cleanup_error
                else ""
            )
            if self._report_artifact_authority_callback_failure(
                context="완료 실측 묶음 worker 시작 실패",
                detail=detail,
            ):
                return
            self.status_info.setText("완료 실측 묶음 작업 시작 실패")
            QMessageBox.warning(
                self,
                "완료 실측 묶음 내보내기 실패",
                detail,
            )
            return
        if not started:
            cleanup_error = self._cancel_unstarted_native_export(
                controller,
                work_item,
                reason="task_not_started",
            )
            if cleanup_error:
                if self._report_artifact_authority_callback_failure(
                    context="완료 실측 묶음 worker 미시작",
                    detail=f"내보내기 예약 해제 확인 실패: {cleanup_error}",
                ):
                    return
                self.status_info.setText("완료 실측 묶음 예약 해제 확인 실패")
                QMessageBox.warning(
                    self,
                    "완료 실측 묶음 정리 실패",
                    cleanup_error,
                )

    def _native_tile_unwrap_options_from_panel(self) -> dict[str, Any]:
        panel = self.section_panel
        target = str(panel.combo_native_tile_target.currentData() or "all")
        selected_face_indices: tuple[int, ...] | None = None
        if target == "selected":
            obj = self.viewport.selected_obj
            selected = sorted(
                int(value)
                for value in (getattr(obj, "selected_faces", set()) or set())
            )
            if not selected:
                raise ArtifactTileUnwrapError(
                    "현재 선택 면을 사용하려면 하나 이상의 face를 선택하세요"
                )
            selected_face_indices = tuple(selected)
        elif target != "all":
            raise ArtifactTileUnwrapError("기와 전개 기록 영역이 올바르지 않습니다")
        return {
            "longitudinal_axis": str(
                panel.combo_native_tile_axis.currentData() or "y"
            ),
            "record_view": str(
                panel.combo_native_tile_record_view.currentData() or "top"
            ),
            "selected_face_indices": selected_face_indices,
            "n_sections": int(panel.spin_native_tile_sections.value()),
        }

    def _compute_and_commit_native_tile_unwrap(
        self,
        *,
        longitudinal_axis: str,
        record_view: str,
        selected_face_indices: tuple[int, ...] | None = None,
        n_sections: int = 32,
        record_id: str | None = None,
        created_at: str | None = None,
        operator: str = "local-user",
    ) -> str:
        obj = self.viewport.selected_obj
        session = self._require_native_measurement_session(obj)
        MainWindow._capture_native_scene_preflight(
            self,
            session,
            allowed_selected_face_indices=selected_face_indices,
        )()
        new_record_id = (
            f"record:tile-unwrap:{record_view}:{uuid.uuid4()}"
            if record_id is None
            else record_id
        )
        controller = self._artifact_measurement_controller()
        work_item = controller.begin_tile_unwrap(
            longitudinal_axis=longitudinal_axis,
            record_view=record_view,
            selected_face_indices=selected_face_indices,
            n_sections=n_sections,
            record_id=new_record_id,
            created_at=(self._utc_seconds_now() if created_at is None else created_at),
            operator=operator,
        )
        result = controller.execute(work_item)
        return self._publish_native_measurement_result(work_item, result)

    def on_native_tile_unwrap_requested(self) -> None:
        try:
            obj = self.viewport.selected_obj
            session = self._require_native_measurement_session(obj)
            options = self._native_tile_unwrap_options_from_panel()
            preflight = self._capture_native_scene_preflight(
                session,
                allowed_selected_face_indices=options["selected_face_indices"],
            )
            record_id = (
                f"record:tile-unwrap:{options['record_view']}:{uuid.uuid4()}"
            )
            controller = self._artifact_measurement_controller()
            work_item = controller.begin_tile_unwrap(
                **options,
                record_id=record_id,
                created_at=self._utc_seconds_now(),
                operator="local-user",
            )
        except Exception as exc:
            self.status_info.setText("기와 전개 준비 실패 | 기존 문서 유지")
            QMessageBox.warning(
                self,
                "기와 전개 준비 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return

        def on_done(result: object) -> None:
            if self._native_measurement_callback_is_terminal(
                controller,
                work_item,
                label="기와 전개",
            ):
                return
            try:
                if not isinstance(result, ArtifactMeasurementResult):
                    raise ArtifactWorkbenchError(
                        "tile unwrap worker result is invalid"
                    )
                self._publish_native_measurement_result(work_item, result)
            except Exception as exc:
                if self._report_artifact_authority_callback_failure(
                    context="기와 전개 결과 게시 중 권위 확인 실패",
                    detail=f"{type(exc).__name__}: {exc}",
                ):
                    return
                pending = self._native_measurement_publication_is_pending(work_item)
                self.status_info.setText(
                    "기와 전개 결과 게시 보류 | 재시도 버튼 사용"
                    if pending
                    else "늦은 기와 전개 결과 폐기 | 현재 문서 유지"
                )
                QMessageBox.warning(
                    self,
                    "기와 전개 결과 게시 보류" if pending else "기와 전개 결과 폐기",
                    f"{type(exc).__name__}: {exc}",
                )

        def on_failed(message: str) -> None:
            if self._report_artifact_authority_callback_failure(
                context="기와 전개 worker 종료 콜백",
                detail=str(message),
            ):
                return
            if self._native_measurement_callback_is_terminal(
                controller,
                work_item,
                label="기와 전개",
            ):
                return
            self.status_info.setText("기와 전개 계산 실패 | 기존 문서 유지")
            QMessageBox.warning(
                self,
                "기와 전개 계산 실패",
                self._format_error_message("기와 전개 계산 중 오류:", message),
            )

        self.status_info.setText("기와 전개 계산 중 · canonical mm 단면 적합...")
        started = self._start_task(
            title="기와 기록면 전개",
            label="선택·축·기록면을 고정한 µm 전개 좌표 계산 중...",
            thread=TaskThread(
                "native_tile_unwrap",
                lambda: self._execute_native_measurement_with_preflight(
                    preflight,
                    controller,
                    work_item,
                ),
            ),
            on_done=on_done,
            on_failed=on_failed,
            on_cancel_requested=lambda: self._request_native_measurement_cancel(
                controller,
                work_item,
                label="기와 전개",
            ),
            on_shutdown_joined=lambda: self._verify_native_measurement_shutdown(
                controller,
                work_item,
            ),
        )
        if not started:
            controller.cancel(work_item, reason="task_not_started")

    @staticmethod
    def _recompute_native_tile_unwrap_record(
        session: ArtifactSession,
        record,
    ) -> TileUnwrapMesh:
        if not MainWindow._native_tile_unwrap_record_is_exportable(session, record):
            raise ArtifactTileUnwrapExportError(
                "no READY + FRESH tile unwrap record to export"
            )
        computation = compute_artifact_tile_unwrap_from_recipe(
            session,
            record.recipe,
        )
        require_current_tile_unwrap_computation(session, computation)
        receipt = tile_unwrap_receipt_from_record(record)
        if computation.unwrap.receipt(
            selection_sha256=str(receipt["selection_sha256"])
        ) != receipt:
            raise ArtifactTileUnwrapExportError(
                "recomputed tile unwrap does not match its durable receipt"
            )
        return computation.unwrap

    def _export_native_tile_unwrap_record(
        self,
        destination: str | os.PathLike[str],
        *,
        record_id: str | None = None,
    ) -> Path:
        session = getattr(self, "_artifact_session", None)
        if not isinstance(session, ArtifactSession):
            raise ArtifactTileUnwrapExportError(
                "no active ArtifactDocument session"
            )
        try:
            self._artifact_workbench_controller().require_stable_session(
                session,
                measurement=True,
            )
        except ArtifactWorkbenchError as exc:
            raise ArtifactTileUnwrapExportError(str(exc)) from exc
        record = (
            session.document.record_index.get(record_id)
            if record_id is not None
            else self._current_native_tile_unwrap_record()
        )
        if record is None or not self._native_tile_unwrap_record_is_exportable(
            session,
            record,
        ):
            raise ArtifactTileUnwrapExportError(
                "no READY + FRESH tile unwrap record to export"
            )
        controller = self._artifact_export_controller()
        work_item: ArtifactExportWorkItem | None = None
        result: ArtifactExportResult | None = None
        try:
            work_item = controller.begin_tile_unwrap(destination, record.id)
            result = controller.execute(work_item)
            return controller.publish_result(work_item, result).destination
        except WorkflowBusyError as exc:
            if work_item is not None and result is not None:
                controller.discard_result(
                    work_item,
                    result,
                    reason="synchronous export blocked by pending Open",
                )
            raise ArtifactTileUnwrapExportError(str(exc)) from exc
        except ArtifactExportError as exc:
            raise ArtifactTileUnwrapExportError(str(exc)) from exc

    def on_native_tile_unwrap_export_requested(self) -> None:
        session = getattr(self, "_artifact_session", None)
        record = self._current_native_tile_unwrap_record()
        if not isinstance(session, ArtifactSession) or record is None:
            self.status_info.setText(
                "내보낼 READY + FRESH 기와 전개 기록이 없습니다."
            )
            return
        try:
            self._artifact_workbench_controller().require_stable_session(
                session,
                measurement=True,
            )
        except ArtifactWorkbenchError as exc:
            self.status_info.setText(f"기와 전개 내보내기 차단: {exc}")
            return
        source_path = Path(str(self.current_filepath or "artifact"))
        default_path = source_path.with_name(
            f"{source_path.stem}-tile-unwrap{TILE_UNWRAP_EXPORT_DIRECTORY_SUFFIX}"
        )
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "1:1 기와 전개 패키지 저장",
            str(default_path),
            "ArchMeshRubbing Tile Unwrap Package (*.amr-unwrap)",
        )
        if not selected:
            return
        if not selected.endswith(TILE_UNWRAP_EXPORT_DIRECTORY_SUFFIX):
            selected += TILE_UNWRAP_EXPORT_DIRECTORY_SUFFIX
        try:
            controller = self._artifact_export_controller()
            work_item = controller.begin_tile_unwrap(selected, record.id)
        except Exception as exc:
            self.status_info.setText("기와 전개 내보내기 준비 실패")
            QMessageBox.warning(
                self,
                "기와 전개 내보내기 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return

        def on_done(value: object) -> None:
            try:
                if self._artifact_export_controller() is not controller:
                    raise ArtifactExportError(
                        "tile unwrap export controller authority was replaced"
                    )
                if not isinstance(value, ArtifactExportResult):
                    raise ArtifactExportError(
                        "tile unwrap export worker result is invalid"
                    )
                publication = controller.publish_result(work_item, value)
            except Exception as exc:
                cleanup_error = self._cancel_native_export_if_staged(
                    controller,
                    work_item,
                    reason="invalid or rejected tile unwrap export callback",
                )
                detail = f"{type(exc).__name__}: {exc}" + (
                    f"\n임시 패키지 정리 확인 실패: {cleanup_error}"
                    if cleanup_error
                    else ""
                )
                if self._report_artifact_authority_callback_failure(
                    context="기와 전개 패키지 게시 콜백 실패",
                    detail=detail,
                ):
                    return
                self.status_info.setText("기와 전개 패키지 저장 실패")
                QMessageBox.warning(
                    self,
                    "기와 전개 내보내기 실패",
                    detail,
                )
                return
            self._report_native_export_publication(
                publication,
                artifact_label="기와 OBJ/SVG",
            )

        def on_failed(message: str) -> None:
            if self._native_export_callback_is_cancelled(
                controller,
                work_item,
                label="기와 전개 내보내기",
            ):
                return
            if self._report_artifact_authority_callback_failure(
                context="기와 전개 export worker 종료 콜백",
                detail=str(message),
            ):
                return
            self.status_info.setText("기와 전개 패키지 저장 실패")
            QMessageBox.warning(
                self,
                "기와 전개 내보내기 실패",
                self._format_error_message("패키지 생성 중 오류:", message),
            )

        try:
            started = self._start_task(
                title="기와 전개 내보내기",
                label="record recipe 재계산 및 1:1 OBJ·SVG 검증 중...",
                thread=TaskThread(
                    "export_native_tile_unwrap",
                    lambda: controller.execute(work_item),
                ),
                on_done=on_done,
                on_failed=on_failed,
                on_cancel_requested=lambda: self._request_native_export_cancel(
                    controller,
                    work_item,
                    label="기와 전개 내보내기",
                ),
                on_shutdown_joined=lambda: self._verify_native_export_shutdown(
                    controller,
                    work_item,
                ),
            )
        except Exception as exc:
            cleanup_error = self._cancel_unstarted_native_export(
                controller,
                work_item,
                reason="task_start_failed",
            )
            detail = f"{type(exc).__name__}: {exc}" + (
                f"\n내보내기 예약 해제 확인 실패: {cleanup_error}"
                if cleanup_error
                else ""
            )
            self.status_info.setText("기와 전개 패키지 작업 시작 실패")
            QMessageBox.warning(self, "기와 전개 내보내기 실패", detail)
            return
        if not started:
            cleanup_error = self._cancel_unstarted_native_export(
                controller,
                work_item,
                reason="task_not_started",
            )
            if cleanup_error:
                self.status_info.setText("기와 전개 예약 해제 확인 실패")
                QMessageBox.warning(
                    self,
                    "기와 전개 내보내기 정리 실패",
                    cleanup_error,
                )

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
        if enabled and self._native_artifact_mode():
            try:
                self.section_panel.btn_line.blockSignals(True)
                self.section_panel.btn_line.setChecked(False)
            finally:
                self.section_panel.btn_line.blockSignals(False)
            self.status_info.setText("화면용 legacy 단면선 차단 | 검증된 단면 사용")
            return
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
                "단면선 그리기 중지" if bool(enabled) else "단면선 그리기 시작"
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
        if self._native_artifact_mode():
            self.status_info.setText(
                "legacy 단면 레이어 차단 | 위의 '단면 계산 · 기록'을 사용하세요"
            )
            return
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
        if self._native_artifact_mode():
            self.status_info.setText(
                "화면용 Cutline 자동 저장 안 함 | 검증된 단면 명령을 사용하세요"
            )
            return
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
                slicer = MeshSlicer(obj.mesh)

                world_origin = np.array([0.0, 0.0, float(height)], dtype=np.float64)
                world_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                translation = np.asarray(obj.translation, dtype=np.float64).reshape(3,)
                local_to_world = scene_trs_matrix(
                    translation,
                    obj.rotation,
                    float(obj.scale),
                )
                local_origin, local_normal = transform_plane_world_to_local(
                    world_origin,
                    world_normal,
                    local_to_world,
                )

                contours_local = slicer.slice_with_plane(local_origin, local_normal)
                if not contours_local:
                    QMessageBox.warning(self, "경고", f"Z={height:.2f} 높이에서 단면을 찾을 수 없습니다.")
                    return

                contours_world: list[np.ndarray] = []
                for contour in contours_local:
                    arr = np.asarray(contour, dtype=np.float64)
                    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
                        continue
                    contours_world.append(transform_points(arr[:, :3], local_to_world))

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

        # Viewport3D uses the OpenGL 2.1 fixed-function compatibility API.
        # Qt must receive that contract before QApplication creates any native
        # graphics resources.
        from src.gui.opengl_context import install_compatibility_surface_format

        install_compatibility_surface_format()
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
