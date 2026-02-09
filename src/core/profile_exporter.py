"""
2D 실루엣 추출 및 SVG 내보내기 모듈 (v5 - 디버깅 완료)
메쉬를 6방향에서 투영하여 외곽선을 폴리라인으로 추출합니다.
"""
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import importlib.util
import logging
import os
import subprocess
import sys
from types import ModuleType
from typing import Any, cast

from .logging_utils import log_once

_LOGGER = logging.getLogger(__name__)

OpenGLMatrices = tuple[Any, Any, Any]

Contours = list[np.ndarray]
Bounds = dict[str, Any]

_cv2_module: ModuleType | None = None
_cv2_checked = False
_CV2_DISABLED = os.environ.get("ARCHMESHRUBBING_DISABLE_OPENCV", "").strip().lower() in {"1", "true", "yes", "on"}
_PROFILE_EXPORT_SAFE = (
    os.environ.get("ARCHMESHRUBBING_PROFILE_EXPORT_SAFE", "").strip().lower()
    in {"1", "true", "yes", "on"}
)

try:
    _cv2_import_timeout_seconds = float(os.environ.get("ARCHMESHRUBBING_CV2_IMPORT_TIMEOUT", "2.0"))
except ValueError:
    _cv2_import_timeout_seconds = 2.0

try:
    from scipy import ndimage
except ImportError:
    ndimage = None  # type: ignore[assignment]


def _world_units_per_cm_from_unit(unit: str | None) -> float:
    u = str(unit or "").strip().lower()
    if u in {"mm", "millimeter", "millimeters"}:
        return 10.0
    if u in {"cm", "centimeter", "centimeters"}:
        return 1.0
    if u in {"m", "meter", "meters"}:
        return 0.01
    return 1.0


def _resolve_world_units_per_cm(mesh: Any, override: float | None = None) -> float:
    if override is not None:
        try:
            v = float(override)
            if v > 0:
                return v
        except Exception:
            _LOGGER.debug("Invalid world_units_per_cm override: %r", override, exc_info=True)
    return _world_units_per_cm_from_unit(getattr(mesh, "unit", None))


def _get_cv2() -> ModuleType | None:
    global _cv2_checked, _cv2_module

    if _cv2_module is not None:
        return _cv2_module
    if _cv2_checked:
        return None
    _cv2_checked = True

    if _CV2_DISABLED:
        return None

    # In frozen/packaged apps, `sys.executable` is the app binary, so the
    # subprocess smoke-test isn't viable. Fall back to a direct import.
    if getattr(sys, "frozen", False):
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return None
        _cv2_module = cv2
        return _cv2_module

    if importlib.util.find_spec("cv2") is None:
        return None

    # Avoid hanging the main process on some OpenCV installs by doing a 1-time
    # import smoke-test in a subprocess.
    try:
        subprocess.run(
            [sys.executable, "-c", "import cv2"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=_cv2_import_timeout_seconds,
        )
    except Exception:
        return None

    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return None

    _cv2_module = cv2
    return _cv2_module


def _rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer–Douglas–Peucker polyline simplification.

    Args:
        points: (N, 2) polyline points
        epsilon: max perpendicular distance (same unit as points)
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return points

    keep = np.zeros((len(points),), dtype=bool)
    keep[0] = True
    keep[-1] = True

    stack: list[tuple[int, int]] = [(0, len(points) - 1)]
    while stack:
        start_idx, end_idx = stack.pop()
        if end_idx <= start_idx + 1:
            continue

        start = points[start_idx]
        end = points[end_idx]
        line = end - start
        line_len = float(np.hypot(line[0], line[1]))

        inner = points[start_idx + 1 : end_idx]
        if inner.size == 0:
            continue

        if line_len < 1e-12:
            dists = np.hypot(*(inner - start).T)
        else:
            vec = inner - start
            dists = np.abs(line[0] * vec[:, 1] - line[1] * vec[:, 0]) / line_len

        rel = int(np.argmax(dists))
        max_dist = float(dists[rel])
        if max_dist > epsilon:
            mid_idx = start_idx + 1 + rel
            keep[mid_idx] = True
            stack.append((start_idx, mid_idx))
            stack.append((mid_idx, end_idx))

    return points[keep]


def _trace_main_contour_binary(mask: np.ndarray) -> np.ndarray:
    """
    Extract an ordered outer contour from a binary mask without OpenCV.

    Returns:
        (N, 2) float array in pixel coordinates (x, y), y-down.
    """
    if ndimage is None:
        raise RuntimeError("SciPy is required for contour extraction when OpenCV is not available")

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros((0, 2), dtype=np.float64)

    labels, num = cast(tuple[np.ndarray, int], ndimage.label(mask))
    if num <= 0:
        return np.zeros((0, 2), dtype=np.float64)

    counts = np.bincount(labels.ravel())
    if len(counts) <= 1:
        return np.zeros((0, 2), dtype=np.float64)
    counts[0] = 0
    main_label = int(np.argmax(counts))
    comp = labels == main_label

    # Find boundary pixels
    eroded = ndimage.binary_erosion(comp, structure=np.ones((3, 3), dtype=bool), border_value=0)
    boundary = comp & np.logical_not(eroded)
    coords = np.argwhere(boundary)
    if coords.size == 0:
        # Fallback: single pixel/component
        coords = np.argwhere(comp)
        if coords.size == 0:
            return np.zeros((0, 2), dtype=np.float64)

    # Choose a deterministic start: top-most, then left-most.
    start_idx = int(np.lexsort((coords[:, 1], coords[:, 0]))[0])
    sy, sx = coords[start_idx]

    # Pad to simplify neighbor checks.
    comp_p = np.pad(comp, 1, mode='constant', constant_values=False)
    sy += 1
    sx += 1

    # Moore-neighbor tracing around the component (8-connectivity).
    # Directions in clockwise order starting from NW.
    dirs = np.array(
        [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
        ],
        dtype=int,
    )
    dir_map = {(int(dy), int(dx)): i for i, (dy, dx) in enumerate(dirs)}

    b0 = (int(sy), int(sx))
    b = b0
    c0 = (b0[0], b0[1] - 1)  # west neighbor
    c = c0

    contour_xy: list[list[float]] = []
    max_steps = int(comp_p.size)  # generous upper bound

    for _ in range(max_steps):
        # store (x, y) in unpadded coordinates
        contour_xy.append([float(b[1] - 1), float(b[0] - 1)])

        dy = c[0] - b[0]
        dx = c[1] - b[1]
        d = dir_map.get((int(dy), int(dx)), 7)

        found = False
        # Scan neighbors clockwise, starting just after the backtrack direction.
        for step in range(1, 9):
            dn = (d + step) % 8
            nb = (b[0] + int(dirs[dn][0]), b[1] + int(dirs[dn][1]))
            if comp_p[nb]:
                # The "previous" neighbor is the one before nb in scanning order.
                prev_dir = (dn - 1) % 8
                c = (b[0] + int(dirs[prev_dir][0]), b[1] + int(dirs[prev_dir][1]))
                b = nb
                found = True
                break

        if not found:
            break

        if b == b0 and c == c0:
            break

    return np.asarray(contour_xy, dtype=np.float64)


def _extract_main_contour(occupancy: np.ndarray) -> np.ndarray:
    """
    occupancy: (H, W) uint8 image where 0=background, 255=filled.
    Returns (N, 2) float points in pixel coords (x, y), y-down.
    """
    if occupancy.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    cv2 = _get_cv2()
    if cv2 is not None:
        chain_mode = cv2.CHAIN_APPROX_NONE if _PROFILE_EXPORT_SAFE else cv2.CHAIN_APPROX_TC89_KCOS
        res = cv2.findContours(occupancy, cv2.RETR_EXTERNAL, chain_mode)
        if len(res) == 2:
            found_contours, _ = res
        else:
            _, found_contours, _ = res
        if not found_contours:
            return np.zeros((0, 2), dtype=np.float64)
        main_cnt = max(found_contours, key=cv2.contourArea)
        epsilon = 0.0002 * cv2.arcLength(main_cnt, True)
        approx = cv2.approxPolyDP(main_cnt, epsilon, True)
        approx_pts = approx.reshape(-1, 2).astype(np.float64)
        if approx_pts.shape[0] >= 3:
            return approx_pts
        return main_cnt.reshape(-1, 2).astype(np.float64)

    contour = _trace_main_contour_binary(occupancy > 0)
    if len(contour) < 3:
        return contour

    # Simplify (closed polygon)
    closed = np.vstack([contour, contour[0]])
    perimeter = float(np.sum(np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))))
    epsilon = max(0.5, 0.0002 * perimeter)
    simplified = _rdp_simplify(closed, epsilon)
    if len(simplified) >= 2 and np.allclose(simplified[0], simplified[-1]):
        simplified = simplified[:-1]
    if len(simplified) < 3:
        return contour
    return simplified


def _project_world_to_px_segments(
    pts_world: np.ndarray,
    *,
    mvp: np.ndarray,
    vp: np.ndarray,
    jump_factor: float = 1.5,
) -> list[np.ndarray]:
    pts_world = np.asarray(pts_world, dtype=np.float64)
    if pts_world.ndim != 2 or pts_world.shape[0] < 2 or pts_world.shape[1] < 2:
        return []

    if pts_world.shape[1] == 2:
        pts_world = np.hstack([pts_world, np.zeros((len(pts_world), 1), dtype=np.float64)])

    vp = np.asarray(vp, dtype=np.float64).reshape(-1)
    if vp.size < 4:
        return []

    v_homo = np.hstack([pts_world[:, :3], np.ones((len(pts_world), 1), dtype=np.float64)])
    v_clip = v_homo @ np.asarray(mvp, dtype=np.float64)
    w = v_clip[:, 3]
    valid = np.abs(w) > 1e-12

    v_ndc = np.full((len(v_clip), 3), np.nan, dtype=np.float64)
    v_ndc[valid] = v_clip[valid, :3] / w[valid, None]
    x = (v_ndc[:, 0] + 1.0) / 2.0 * float(vp[2])
    y = float(vp[3]) - (v_ndc[:, 1] + 1.0) / 2.0 * float(vp[3])
    pts = np.stack([x, y], axis=1)

    finite = valid & np.all(np.isfinite(pts), axis=1)
    max_dim = max(float(vp[2]), float(vp[3]), 1.0)
    finite &= np.max(np.abs(pts), axis=1) <= max_dim * 10.0

    jump = max_dim * float(jump_factor)
    segments: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for ok, p in zip(finite, pts):
        if bool(ok):
            if current and float(np.linalg.norm(p - current[-1])) > jump:
                if len(current) >= 2:
                    segments.append(np.asarray(current, dtype=np.float64))
                current = [p]
            else:
                current.append(p)
        else:
            if len(current) >= 2:
                segments.append(np.asarray(current, dtype=np.float64))
            current = []

    if len(current) >= 2:
        segments.append(np.asarray(current, dtype=np.float64))

    return segments


class ProfileExporter:
    """메쉬의 2D 실루엣을 추출하고 SVG로 내보냅니다."""
    
    # 6방향 뷰 정의
    # axes: (SVG X축에 매핑될 3D축 인덱스, SVG Y축에 매핑될 3D축 인덱스)
    # 모든 뷰에서 SVG 좌표계: 오른쪽=+X, 위쪽=+Y (표준)
    VIEWS = {
        'top':    {'axes': (0, 1)},  # XY 평면, X→오른쪽, Y→위쪽
        'bottom': {'axes': (0, 1)},  # XY 평면, X→오른쪽, Y→아래쪽
        'front':  {'axes': (0, 2)},  # XZ 평면, X→오른쪽, Z→위쪽
        'back':   {'axes': (0, 2)},  # XZ 평면, X→왼쪽, Z→위쪽
        'left':   {'axes': (1, 2)},  # YZ 평면, Y→왼쪽, Z→위쪽
        'right':  {'axes': (1, 2)},  # YZ 평면, Y→오른쪽, Z→위쪽
    }
    
    MAX_FACES_FOR_RASTERIZE = 100000  # 래스터화 최대 면 수
    
    def __init__(self, resolution: int = 1024):
        self.resolution = resolution
    
    def extract_silhouette(self, mesh, view: str = 'top',
                           translation: np.ndarray | None = None,
                           rotation: np.ndarray | None = None,
                           scale: float = 1.0,
                           opengl_matrices: OpenGLMatrices | None = None,
                           viewport_image: Image.Image | None = None,
                           world_units_per_cm: float | None = None) -> tuple[Contours, Bounds]:
        """
        메쉬를 투영하여 단일 외곽선(폴리라인)을 추출합니다.
        opengl_matrices: (modelview, projection, viewport) - 제공 시 실제 렌더링 화면과 완벽히 일치시킴
        viewport_image: OpenGL 캡처 이미지(PIL). 제공 시 화면과 100% 일치하는 외곽선을 이미지 기반으로 추출
        world_units_per_cm: 1cm에 해당하는 world 단위 수. None이면 mesh.unit(mm/cm/m)에서 추정합니다.
        """
        wupc = _resolve_world_units_per_cm(mesh, world_units_per_cm)
        cv2_mod = _get_cv2()
        if opengl_matrices:
            mv_raw, proj_raw, vp = opengl_matrices
            # PyOpenGL matrices are column-major. Reshape+transpose yields conventional (column-vector) matrices;
            # we then build a row-vector MVP to use with (v @ M) throughout this module.
            mv = np.asarray(mv_raw, dtype=np.float64).reshape(4, 4).T
            proj = np.asarray(proj_raw, dtype=np.float64).reshape(4, 4).T
            # Row-vector convention: (P*M*v)^T = v^T*M^T*P^T
            mvp = mv.T @ proj.T

            # 1) 월드 bounds 계산 (전체 정점 변환 없이 bounds corner 8개만 변환)
            w_min = None
            w_max = None
            try:
                lb = np.asarray(mesh.bounds, dtype=np.float64)
                corners = np.array(
                    [
                        [lb[0, 0], lb[0, 1], lb[0, 2]],
                        [lb[1, 0], lb[0, 1], lb[0, 2]],
                        [lb[0, 0], lb[1, 1], lb[0, 2]],
                        [lb[1, 0], lb[1, 1], lb[0, 2]],
                        [lb[0, 0], lb[0, 1], lb[1, 2]],
                        [lb[1, 0], lb[0, 1], lb[1, 2]],
                        [lb[0, 0], lb[1, 1], lb[1, 2]],
                        [lb[1, 0], lb[1, 1], lb[1, 2]],
                    ],
                    dtype=np.float64,
                )
                corners = corners * float(scale)
                if rotation is not None:
                    from scipy.spatial.transform import Rotation as R
                    # Match OpenGL fixed-function order: glRotate(X) -> glRotate(Y) -> glRotate(Z)
                    # which corresponds to intrinsic "XYZ" in SciPy.
                    r = R.from_euler('XYZ', rotation, degrees=True)
                    corners = r.apply(corners)
                if translation is not None:
                    corners = corners + translation
                w_min = corners.min(axis=0)
                w_max = corners.max(axis=0)
            except Exception:
                log_once(
                    _LOGGER,
                    "profile_exporter:opengl_bounds_from_corners",
                    logging.DEBUG,
                    "Failed to compute bounds from mesh.bounds corners; falling back to full-vertex bounds",
                    exc_info=True,
                )

            if w_min is None or w_max is None:
                v_all = np.asarray(mesh.vertices, dtype=np.float64) * float(scale)
                if rotation is not None:
                    from scipy.spatial.transform import Rotation as R
                    r = R.from_euler('XYZ', rotation, degrees=True)
                    v_all = r.apply(v_all)
                if translation is not None:
                    v_all = v_all + translation
                w_min = v_all.min(axis=0)
                w_max = v_all.max(axis=0)

            vp = np.asarray(vp, dtype=np.float64).reshape(-1)
            if vp.size < 4:
                vp = np.array([0.0, 0.0, 1024.0, 1024.0], dtype=np.float64)

            # 2) px_per_cm 계산 (world 좌표에서 1cm = wupc world units로 가정)
            # 정사영(glOrtho)인 경우 projection 행렬에서 직접 스케일을 얻는 것이 가장 정확하다.
            def project_pt(p, vp_local):
                vh = np.append(p, 1.0)
                vc = vh @ mvp
                if abs(float(vc[3])) < 1e-12:
                    return np.array([0.0, 0.0], dtype=np.float64)
                vn = vc[:3] / vc[3]
                return np.array([(vn[0] + 1) / 2 * vp_local[2], vp_local[3] - (vn[1] + 1) / 2 * vp_local[3]])

            def compute_px_per_cm(vp_local) -> float:
                px_per_world_unit = None
                try:
                    if abs(float(proj[3, 3]) - 1.0) < 1e-9 and abs(float(proj[3, 2])) < 1e-9:
                        px_x = abs(float(vp_local[2]) * float(proj[0, 0]) / 2.0)
                        px_y = abs(float(vp_local[3]) * float(proj[1, 1]) / 2.0)
                        px_per_world_unit = float((px_x + px_y) * 0.5)
                except Exception:
                    px_per_world_unit = None

                if px_per_world_unit is None or px_per_world_unit < 1e-6:
                    center_world = (w_min + w_max) / 2.0
                    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    if view in {"left", "right"}:
                        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    p1 = center_world + axis

                    px_per_world_unit = float(
                        np.linalg.norm(project_pt(center_world, vp_local) - project_pt(p1, vp_local))
                    )

                px_per_cm = float(px_per_world_unit or 0.0) * float(wupc)
                if not np.isfinite(px_per_cm) or px_per_cm < 1e-6:
                    px_per_cm = 100.0  # Fallback
                return float(px_per_cm)

            px_per_cm = compute_px_per_cm(vp)

            # 3) 외곽선 추출: viewport_image가 있으면 이미지 기반(가장 견고/빠름)
            grid_w = int(vp[2])
            grid_h = int(vp[3])
            occupancy = np.zeros((grid_h, grid_w), dtype=np.uint8)
            if viewport_image is not None:
                try:
                    img_rgba = viewport_image.convert("RGBA")
                    rgb = np.asarray(img_rgba, dtype=np.uint8)[..., :3]
                    if rgb.shape[1] != grid_w or rgb.shape[0] != grid_h:
                        # 캡처 사이즈가 vp와 다르면 vp를 이미지 기준으로 맞춤
                        grid_w = int(rgb.shape[1])
                        grid_h = int(rgb.shape[0])
                        vp = np.array([0.0, 0.0, float(grid_w), float(grid_h)], dtype=np.float64)
                        px_per_cm = compute_px_per_cm(vp)
                        occupancy = np.zeros((grid_h, grid_w), dtype=np.uint8)

                    corners_rgb = np.stack(
                        [rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], axis=0
                    ).astype(np.int16)
                    bg = np.median(corners_rgb, axis=0).astype(np.int16)

                    diff = np.max(np.abs(rgb.astype(np.int16) - bg[None, None, :]), axis=2)
                    mask = diff > 8
                    if int(mask.sum()) < 64:
                        # 매우 밝은 메쉬/조명 등으로 diff가 거의 없을 때: 단순 백색 임계값으로 fallback
                        mask = rgb.mean(axis=2) < 254

                    if cv2_mod is not None:
                        occ = (mask.astype(np.uint8) * 255)
                        k = np.ones((3, 3), dtype=np.uint8)
                        occ = cv2_mod.morphologyEx(occ, cv2_mod.MORPH_CLOSE, k, iterations=2)
                        occ = cv2_mod.morphologyEx(occ, cv2_mod.MORPH_OPEN, k, iterations=1)
                        occupancy = occ
                    else:
                        if ndimage is not None:
                            mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=2)
                            mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
                        occupancy = (mask.astype(np.uint8) * 255)

                    # If segmentation failed (too few pixels / line-like), fall back to geometry rasterization.
                    try:
                        occ_mask = occupancy > 0
                        filled = int(occ_mask.sum())
                        min_fill = max(256, int(occupancy.size * 0.0001))
                        if filled < min_fill:
                            viewport_image = None
                        else:
                            ys, xs = np.where(occ_mask)
                            if xs.size < 2 or ys.size < 2:
                                viewport_image = None
                            else:
                                if (int(xs.max() - xs.min()) < 4) or (int(ys.max() - ys.min()) < 4):
                                    viewport_image = None
                    except Exception:
                        viewport_image = None
                except Exception:
                    viewport_image = None  # fallback to geometry rasterize

            if viewport_image is None:
                # (Fallback) geometry rasterize 기반 외곽선 추출
                faces = getattr(mesh, "faces", None)
                if faces is None:
                    faces = []
                faces = np.asarray(faces, dtype=np.int32)
                if faces.size == 0:
                    bounds = {
                        'min': np.array([0, 0]),
                        'max': np.array([grid_w / px_per_cm, grid_h / px_per_cm]),
                        'size': np.array([grid_w / px_per_cm, grid_h / px_per_cm]),
                        'px_per_cm': px_per_cm,
                        'world_units_per_cm': float(wupc),
                        'mesh_unit': getattr(mesh, "unit", None),
                        'view': view,
                        'is_pixels': True,
                        'vp_size': (grid_w, grid_h),
                        'grid_size': (grid_w, grid_h),
                        'matrices': (mv_raw, proj_raw, vp),
                        'world_bounds': np.array([w_min, w_max], dtype=np.float64),
                    }
                    return [], bounds

                if len(faces) > self.MAX_FACES_FOR_RASTERIZE:
                    step = max(1, int(len(faces) // self.MAX_FACES_FOR_RASTERIZE))
                    faces = faces[::step]
                    if len(faces) > self.MAX_FACES_FOR_RASTERIZE:
                        faces = faces[: self.MAX_FACES_FOR_RASTERIZE]

                unique_idx = np.unique(faces.reshape(-1))
                faces = np.searchsorted(unique_idx, faces).astype(np.int32, copy=False)

                vertices = np.asarray(mesh.vertices[unique_idx], dtype=np.float64) * float(scale)
                if rotation is not None:
                    from scipy.spatial.transform import Rotation as R
                    r = R.from_euler('XYZ', rotation, degrees=True)
                    vertices = r.apply(vertices)
                if translation is not None:
                    vertices = vertices + translation

                v_homo = np.hstack([vertices, np.ones((len(vertices), 1))])
                v_clip = v_homo @ mvp  # (N, 4)
                w = v_clip[:, 3]
                valid_w = np.abs(w) > 1e-12
                v_ndc = np.full((len(v_clip), 3), np.nan, dtype=np.float64)
                v_ndc[valid_w] = v_clip[valid_w, :3] / w[valid_w, None]

                proj_2d = np.empty((len(vertices), 2), dtype=np.float64)
                proj_2d[:, 0] = (v_ndc[:, 0] + 1) / 2 * grid_w
                proj_2d[:, 1] = (v_ndc[:, 1] + 1) / 2 * grid_h
                proj_2d[:, 1] = grid_h - proj_2d[:, 1]

                finite = valid_w & np.all(np.isfinite(proj_2d), axis=1)
                max_dim = max(float(grid_w), float(grid_h), 1.0)
                finite &= np.max(np.abs(proj_2d), axis=1) <= max_dim * 10.0
                if not bool(np.all(finite)):
                    faces = faces[np.all(finite[faces], axis=1)]

                occupancy = np.zeros((grid_h, grid_w), dtype=np.uint8)
                if cv2_mod is not None:
                    img_v = proj_2d.astype(np.int32)
                    for face in faces:
                        cv2_mod.fillPoly(occupancy, [img_v[face]], 255)
                else:
                    img = Image.new('L', (grid_w, grid_h), 0)
                    draw = ImageDraw.Draw(img)
                    for face in faces:
                        draw.polygon([(int(p[0]), int(p[1])) for p in proj_2d[face]], fill=255)
                    occupancy = np.array(img)

            contours = []
            contour_px = _extract_main_contour(occupancy)
            if len(contour_px) > 0:
                contours.append(contour_px.astype(float))

            bounds = {
                'min': np.array([0, 0]),
                'max': np.array([grid_w / px_per_cm, grid_h / px_per_cm]),
                'size': np.array([grid_w / px_per_cm, grid_h / px_per_cm]),
                'px_per_cm': px_per_cm,
                'world_units_per_cm': float(wupc),
                'mesh_unit': getattr(mesh, "unit", None),
                'view': view,
                'is_pixels': True,
                'vp_size': (grid_w, grid_h),
                'grid_size': (grid_w, grid_h),
                'matrices': (mv_raw, proj_raw, vp),
                'world_bounds': np.array([w_min, w_max], dtype=np.float64),
            }
            return contours, bounds

        # Fallback (Orthogonal-like simple projection)
        if view not in self.VIEWS:
            raise ValueError(f"Unknown view: {view}")
        
        view_config = self.VIEWS[view]
        ax0, ax1 = view_config['axes']
        
        # 정점 변환
        vertices = mesh.vertices.copy() * scale
        if rotation is not None and np.any(rotation != 0):
            from scipy.spatial.transform import Rotation as R
            r = R.from_euler('XYZ', rotation, degrees=True)
            vertices = r.apply(vertices)
        if translation is not None:
            vertices = vertices + translation
        
        # 2D 투영 (실제 좌표)
        proj_2d = vertices[:, [ax0, ax1]]
        if view == 'back':
            proj_2d[:, 0] = -proj_2d[:, 0]
        if view == 'left':
            proj_2d[:, 0] = -proj_2d[:, 0]
        if view == 'bottom':
            proj_2d[:, 1] = -proj_2d[:, 1]
        
        # 바운딩 박스 (실제 좌표)
        min_coords = proj_2d.min(axis=0)
        max_coords = proj_2d.max(axis=0)
        size = max_coords - min_coords
        margin = size * 0.02
        min_coords -= margin
        max_coords += margin
        size = max_coords - min_coords
        
        grid_res = max(self.resolution, 2048)
        aspect = size[0] / size[1] if size[1] > 0 else 1.0
        if aspect >= 1:
            grid_w, grid_h = grid_res, max(2, int(grid_res / aspect))
        else:
            grid_h, grid_w = grid_res, max(2, int(grid_res * aspect))
        
        size_safe = np.asarray(size, dtype=np.float64)
        size_safe = np.where(np.abs(size_safe) < 1e-12, 1e-12, size_safe)

        def real_to_img(pts):
            img_pts = np.zeros_like(pts)
            img_pts[:, 0] = (pts[:, 0] - min_coords[0]) / size_safe[0] * (grid_w - 1)
            img_pts[:, 1] = (1 - (pts[:, 1] - min_coords[1]) / size_safe[1]) * (grid_h - 1)
            return img_pts

        occupancy = np.zeros((grid_h, grid_w), dtype=np.uint8)
        if cv2_mod is not None:
            img_v = real_to_img(proj_2d).astype(np.int32)
            for face in mesh.faces:
                cv2_mod.fillPoly(occupancy, [img_v[face]], 255)
        else:
            img = Image.new('L', (grid_w, grid_h), 0)
            draw = ImageDraw.Draw(img)
            img_v = real_to_img(proj_2d)
            for face in mesh.faces:
                draw.polygon([(int(p[0]), int(p[1])) for p in img_v[face]], fill=255)
            occupancy = np.array(img)

        contour_px = _extract_main_contour(occupancy)
        contours = []
        if len(contour_px) > 0:
            img_pts = contour_px.astype(float)

            # 이미지 좌표 -> 실제 좌표 변환
            real_pts = np.zeros_like(img_pts)
            real_pts[:, 0] = img_pts[:, 0] / (grid_w - 1) * size_safe[0] + min_coords[0]
            real_pts[:, 1] = (1 - img_pts[:, 1] / (grid_h - 1)) * size_safe[1] + min_coords[1]
            contours.append(real_pts)
        
        bounds = {
            'min': min_coords,
            'max': max_coords,
            'size': size,
            'grid_size': (grid_w, grid_h),
            'is_pixels': False,
            'world_units_per_cm': float(wupc),
            'mesh_unit': getattr(mesh, "unit", None),
            'view': view,
        }
        return contours, bounds
    
    def generate_composite_image(
        self,
        viewport_image: Image.Image,
        bounds: Bounds,
        grid_spacing: float = 1.0,
        *,
        view: str | None = None,
    ) -> Image.Image:
        """
        사용자가 실제 본 화면(viewport_image) + 격자 오버레이 이미지 생성 (Multiply)

        grid_spacing은 cm 단위입니다. is_pixels=True인 경우 bounds.world_units_per_cm를 이용해 world spacing으로 변환합니다.
        """
        img_w, img_h = viewport_image.size
        grid_img = Image.new('RGB', (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(grid_img)
        line_color = (200, 200, 200)

        if not np.isfinite(grid_spacing) or grid_spacing <= 0:
            return viewport_image

        if bounds.get('is_pixels') and 'matrices' in bounds:
            # 1. 행렬 기반 격자 투영
            try:
                mv_raw, proj_raw, vp = bounds['matrices']
                mv = np.asarray(mv_raw, dtype=np.float64).reshape(4, 4).T
                proj = np.asarray(proj_raw, dtype=np.float64).reshape(4, 4).T
                # Row-vector convention: (P*M*v)^T = v^T*M^T*P^T
                mvp = mv.T @ proj.T

                wupc = float(bounds.get("world_units_per_cm", 1.0))
                if not np.isfinite(wupc) or wupc <= 0:
                    wupc = 1.0

                spacing_world = float(grid_spacing) * wupc
                if spacing_world <= 0:
                    return viewport_image

                # 투영용 헬퍼
                def w_to_i(pt3: np.ndarray) -> tuple[float, float] | None:
                    vh = np.array([float(pt3[0]), float(pt3[1]), float(pt3[2]), 1.0], dtype=np.float64)
                    vc = vh @ mvp
                    if abs(float(vc[3])) < 1e-6:
                        return None
                    vn = vc[:3] / vc[3]
                    return (vn[0] + 1) / 2 * img_w, img_h - (vn[1] + 1) / 2 * img_h

                # 가시 범위 계산 (메쉬 주변)
                if 'world_bounds' in bounds:
                    wb = np.asarray(bounds['world_bounds'], dtype=np.float64)
                    if wb.shape == (2, 3):
                        w_min, w_max = wb[0], wb[1]
                    else:
                        w_min, w_max = wb.min(axis=0), wb.max(axis=0)
                else:
                    v_world = np.asarray(bounds.get('vertices_world', []), dtype=np.float64)
                    if v_world.ndim != 2 or v_world.shape[1] < 3:
                        return viewport_image
                    w_min, w_max = v_world.min(axis=0), v_world.max(axis=0)

                center = (w_min + w_max) / 2.0
                view_key = (view or bounds.get("view") or "top").lower()
                if view_key in {"front", "back"}:
                    ax0, ax1, ax_const = 0, 2, 1
                elif view_key in {"left", "right"}:
                    ax0, ax1, ax_const = 1, 2, 0
                else:
                    ax0, ax1, ax_const = 0, 1, 2
                const_val = float(center[ax_const])

                pad_cm = 20.0
                pad_world = pad_cm * wupc

                a0_min = float(w_min[ax0] - pad_world)
                a0_max = float(w_max[ax0] + pad_world)
                a1_min = float(w_min[ax1] - pad_world)
                a1_max = float(w_max[ax1] + pad_world)

                def make_pt(v0: float, v1: float) -> np.ndarray:
                    p = np.array(center, dtype=np.float64)
                    p[ax0] = v0
                    p[ax1] = v1
                    p[ax_const] = const_val
                    return p

                # "세로선": ax0를 변화시키고 ax1 범위를 연결
                v0 = np.floor(a0_min / spacing_world) * spacing_world
                v0_end = np.ceil(a0_max / spacing_world) * spacing_world
                while v0 <= v0_end + spacing_world * 0.5:
                    p1 = w_to_i(make_pt(float(v0), a1_min))
                    p2 = w_to_i(make_pt(float(v0), a1_max))
                    if p1 and p2:
                        draw.line([p1, p2], fill=line_color, width=1)
                    v0 += spacing_world

                # "가로선": ax1을 변화시키고 ax0 범위를 연결
                v1 = np.floor(a1_min / spacing_world) * spacing_world
                v1_end = np.ceil(a1_max / spacing_world) * spacing_world
                while v1 <= v1_end + spacing_world * 0.5:
                    p1 = w_to_i(make_pt(a0_min, float(v1)))
                    p2 = w_to_i(make_pt(a0_max, float(v1)))
                    if p1 and p2:
                        draw.line([p1, p2], fill=line_color, width=1)
                    v1 += spacing_world
            except Exception as e:
                _LOGGER.debug("Grid projection failed: %s", e, exc_info=True)
        else:
            # 2. 고전적 평면 투영 (정사영 베이스)
            size = bounds['size']
            min_coords = bounds['min']
            
            size = np.asarray(size, dtype=np.float64).reshape(-1)
            min_coords = np.asarray(min_coords, dtype=np.float64).reshape(-1)
            if size.size < 2 or min_coords.size < 2 or abs(float(size[0])) < 1e-12 or abs(float(size[1])) < 1e-12:
                return viewport_image

            # 세로선 (X)
            x = np.ceil(float(min_coords[0]) / grid_spacing) * grid_spacing
            while x <= min_coords[0] + size[0]:
                px = int((x - float(min_coords[0])) / float(size[0]) * (img_w - 1))
                if 0 <= px < img_w:
                    draw.line([(px, 0), (px, img_h - 1)], fill=line_color, width=1)
                x += grid_spacing
            # 가로선 (Y)
            y = np.ceil(float(min_coords[1]) / grid_spacing) * grid_spacing
            while y <= min_coords[1] + size[1]:
                py = int((1 - (y - float(min_coords[1])) / float(size[1])) * (img_h - 1))
                if 0 <= py < img_h:
                    draw.line([(0, py), (img_w - 1, py)], fill=line_color, width=1)
                y += grid_spacing
            
        # 3. Multiply Blend
        mesh_arr = np.array(viewport_image.convert('RGB'), dtype=np.float32) / 255.0
        grid_arr = np.array(grid_img, dtype=np.float32) / 255.0
        result_arr = (mesh_arr * grid_arr * 255.0).astype(np.uint8)
        return Image.fromarray(result_arr)
    
    def export_svg(self, contours: Contours, bounds: Bounds,
                    background_image: Image.Image | None = None,
                    stroke_color: str = "#000000", # 기본 검정색
                    stroke_width: float = 0.05,    # 0.5mm
                    extra_paths: list[dict[str, Any]] | None = None,
                    output_path: str | None = None) -> str:
        """
        최종 SVG 생성 (배경 이미지 + 단일 벡터 외곽선)
        """
        size = bounds['size']
        min_coords = bounds['min']
        
        svg_width = size[0]
        svg_height = size[1]
        
        svg_parts = [
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
            '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
            '<svg xmlns="http://www.w3.org/2000/svg" ',
            'xmlns:xlink="http://www.w3.org/1999/xlink" ',
            f'width="{svg_width:.2f}cm" height="{svg_height:.2f}cm" ',
            f'viewBox="0 0 {svg_width:.6f} {svg_height:.6f}">',
            f'<!-- Produced by ArchMeshRubbing - Resolution: {background_image.size if background_image else "Vector Only"} -->',
        ]
        
        # 1. 배경 이미지 삽입
        if background_image is not None:
            buffer = io.BytesIO()
            background_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            svg_parts.append(
                f'<image x="0" y="0" width="{svg_width:.6f}" height="{svg_height:.6f}" '
                f'preserveAspectRatio="none" '
                f'xlink:href="data:image/png;base64,{img_base64}" />'
            )
            
        # 2. 외곽선 벡터 삽입
        if contours:
            svg_parts.append(
                f'<g id="outline" stroke="{stroke_color}" fill="none" stroke-width="{stroke_width}" '
                f'stroke-linejoin="round" stroke-linecap="round">'
            )
            for contour in contours:
                if len(contour) < 2:
                    continue
                # 좌표 변환
                svg_pts = contour.copy()
                
                # is_pixels (OpenGL 투영)인 경우 이미 Viewport 상단(0) 기준이므로 그대로 사용 가능하나
                # extract_silhouette에서 vp[3] - proj_2d[:, 1]로 이미 뒤집었는지 확인 필요
                # 현재 코드: proj_2d[:, 1] = vp[3] - proj_2d[:, 1] (라인 69) -> 이미지 좌표계(Top-Left 0)
                # 따라서 min_coords=0이므로 그대로 두면 됨.
                if bounds.get('is_pixels'):
                   # 픽셀 좌표 -> cm 단위로 스케일링만 수행
                   px_per_cm = bounds['px_per_cm']
                   svg_pts = svg_pts / px_per_cm
                else:
                   # 월드 좌표 (Top-Down 투영 등) -> Y축 반전 필요 (SVG는 Y가 아래로 증가, 월드는 위로 증가)
                   svg_pts[:, 0] -= min_coords[0]
                   svg_pts[:, 1] = svg_height - (svg_pts[:, 1] - min_coords[1])
                
                # CAD import 시 "한 개의 선(연속 폴리라인)"으로 들어오도록 단일 path로 출력
                if not np.isfinite(svg_pts).all():
                    continue
                d_parts = [f'M {svg_pts[0,0]:.6f} {svg_pts[0,1]:.6f}']
                for p in svg_pts[1:]:
                    d_parts.append(f'L {p[0]:.6f} {p[1]:.6f}')
                d_parts.append('Z')
                svg_parts.append(f'<path d="{" ".join(d_parts)}" fill="none" />')
            svg_parts.append('</g>')

        # 3. 추가 선(단면 가이드 등) 삽입
        if extra_paths:
            svg_parts.append(
                '<g id="cut_lines" fill="none" stroke-linejoin="round" stroke-linecap="round">'
            )
            for item in extra_paths:
                stroke = str(item.get("stroke", "#ff4040"))
                sw = float(item.get("stroke_width", stroke_width))
                path_id = item.get("id", None)

                # Efficient multi-segment export (one SVG path with many subpaths).
                # Used for feature edge layers to avoid generating thousands of <path> nodes.
                if "segments" in item:
                    try:
                        seg = np.asarray(item.get("segments", []), dtype=np.float64)
                    except Exception:
                        seg = None

                    if seg is None or seg.ndim != 3 or seg.shape[0] == 0 or seg.shape[1] < 2 or seg.shape[2] < 2:
                        continue

                    seg_xy = seg[:, :2, :2].copy()
                    if bounds.get('is_pixels'):
                        px_per_cm = float(bounds.get('px_per_cm', 100.0))
                        if px_per_cm <= 0:
                            px_per_cm = 100.0
                        seg_xy = seg_xy / px_per_cm
                    else:
                        seg_xy[:, :, 0] -= min_coords[0]
                        seg_xy[:, :, 1] = svg_height - (seg_xy[:, :, 1] - min_coords[1])

                    d_parts = []
                    for s in seg_xy:
                        if not np.isfinite(s).all():
                            continue
                        d_parts.append(
                            f'M {s[0,0]:.6f} {s[0,1]:.6f} L {s[1,0]:.6f} {s[1,1]:.6f}'
                        )
                    if not d_parts:
                        continue

                    id_attr = f' id="{path_id}"' if path_id else ""
                    svg_parts.append(
                        f'<path{id_attr} d="{" ".join(d_parts)}" fill="none" stroke="{stroke}" stroke-width="{sw}" />'
                    )
                    continue

                try:
                    pts = np.asarray(item.get("points", []), dtype=np.float64)
                except Exception:
                    continue
                if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
                    continue

                svg_pts = pts[:, :2].copy()
                if bounds.get('is_pixels'):
                    px_per_cm = float(bounds.get('px_per_cm', 100.0))
                    if px_per_cm <= 0:
                        px_per_cm = 100.0
                    svg_pts = svg_pts / px_per_cm
                else:
                    svg_pts[:, 0] -= min_coords[0]
                    svg_pts[:, 1] = svg_height - (svg_pts[:, 1] - min_coords[1])

                if not np.isfinite(svg_pts).all():
                    continue
                d_parts = [f'M {svg_pts[0,0]:.6f} {svg_pts[0,1]:.6f}']
                for p in svg_pts[1:]:
                    d_parts.append(f'L {p[0]:.6f} {p[1]:.6f}')

                id_attr = f' id="{path_id}"' if path_id else ""
                svg_parts.append(
                    f'<path{id_attr} d="{" ".join(d_parts)}" fill="none" stroke="{stroke}" stroke-width="{sw}" />'
                )
            svg_parts.append('</g>')
             
        svg_parts.append('</svg>')
        
        content = '\n'.join(svg_parts)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return output_path
        return content
        
    def export_profile(self, mesh, view: str, output_path: str,
                       translation: np.ndarray | None = None,
                       rotation: np.ndarray | None = None,
                       scale: float = 1.0,
                       grid_spacing: float = 1.0,
                       include_grid: bool = True,
                       viewport_image: Image.Image | None = None,
                       opengl_matrices: OpenGLMatrices | None = None,
                       cut_lines_world: list[Any] | None = None,
                       cut_profiles_world: list[Any] | None = None,
                       feature_edges: Any | None = None,
                       feature_style: dict[str, Any] | None = None,
                       world_units_per_cm: float | None = None) -> str:
        """
        메쉬의 2D 프로파일을 SVG로 내보내는 통합 메서드.

        grid_spacing은 cm 단위입니다. world_units_per_cm를 지정하면 메쉬 단위를 cm로 환산하는 기준을 강제할 수 있습니다.
        """
        contours, bounds = self.extract_silhouette(
            mesh,
            view,
            translation,
            rotation,
            scale,
            opengl_matrices,
            viewport_image=viewport_image,
            world_units_per_cm=world_units_per_cm,
        )

        # export_svg는 cm 단위를 기준으로 동작한다. OpenGL 경로(is_pixels=True)는 px_per_cm로 이미 맞춰져 있고,
        # 단순 투영 경로(is_pixels=False)는 메쉬 단위를 cm로 환산한다.
        if not bool(bounds.get("is_pixels", False)):
            try:
                wupc = float(bounds.get("world_units_per_cm", _resolve_world_units_per_cm(mesh, world_units_per_cm)))
            except Exception:
                wupc = 1.0
            if not np.isfinite(wupc) or wupc <= 0:
                wupc = 1.0

            to_cm = 1.0 / wupc
            bounds = dict(bounds)
            for k in ("min", "max", "size"):
                if k in bounds:
                    bounds[k] = np.asarray(bounds[k], dtype=np.float64) * to_cm
            bounds["world_units_per_cm"] = 1.0
            bounds["mesh_unit"] = "cm"

            contours_cm: list[np.ndarray] = []
            for c in contours or []:
                arr = np.asarray(c, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
                    contours_cm.append(arr * to_cm)
            contours = contours_cm
        
        background_image = None
        if include_grid:
            if viewport_image:
                # 캡처된 실제 이미지와 격자 합성
                background_image = self.generate_composite_image(
                    viewport_image, bounds, grid_spacing, view=view
                )
            else:
                # (Fallback) 실루엣과 격자 합성 - occupancy가 없으므로 빈 이미지 배경
                dummy_viewport = Image.new('RGB', bounds['grid_size'], (255, 255, 255))
                background_image = self.generate_composite_image(
                    dummy_viewport, bounds, grid_spacing, view=view
                )

        extra_paths = []
        matrices: OpenGLMatrices | None = None
        if bool(bounds.get("is_pixels")):
            m = bounds.get("matrices")
            if isinstance(m, tuple) and len(m) == 3:
                matrices = cast(OpenGLMatrices, m)

        matrices_src = matrices if matrices is not None else opengl_matrices

        if cut_lines_world and matrices_src is not None and view in {"top", "bottom"}:
            try:
                mv_raw, proj_raw, vp = matrices_src
                mv = np.asarray(mv_raw, dtype=np.float64).reshape(4, 4).T
                proj = np.asarray(proj_raw, dtype=np.float64).reshape(4, 4).T
                # Row-vector convention: (P*M*v)^T = v^T*M^T*P^T
                mvp = mv.T @ proj.T

                colors = ["#ff4040", "#2b8cff"]
                for i, line in enumerate(cut_lines_world):
                    if not line or len(line) < 2:
                        continue
                    pts_w = np.asarray(line, dtype=np.float64)
                    segments = _project_world_to_px_segments(pts_w, mvp=mvp, vp=vp)
                    for seg_i, pts_px in enumerate(segments):
                        if pts_px.shape[0] < 2:
                            continue
                        path_id = f"cut_line_{i+1}" if len(segments) == 1 else f"cut_line_{i+1}_{seg_i+1}"
                        extra_paths.append(
                            {
                                "id": path_id,
                                "points": pts_px,
                                "stroke": colors[i % len(colors)],
                                "stroke_width": 0.015,  # 0.15mm
                            }
                        )
            except Exception as e:
                _LOGGER.debug("Failed projecting cut lines to pixels: %s", e, exc_info=True)
                extra_paths = []

        if cut_profiles_world and matrices_src is not None and view in {"top", "bottom"}:
            try:
                mv_raw, proj_raw, vp = matrices_src
                mv = np.asarray(mv_raw, dtype=np.float64).reshape(4, 4).T
                proj = np.asarray(proj_raw, dtype=np.float64).reshape(4, 4).T
                # Row-vector convention: (P*M*v)^T = v^T*M^T*P^T
                mvp = mv.T @ proj.T

                for i, line in enumerate(cut_profiles_world):
                    if not line or len(line) < 2:
                        continue
                    pts_w = np.asarray(line, dtype=np.float64)
                    segments = _project_world_to_px_segments(pts_w, mvp=mvp, vp=vp)
                    for seg_i, pts_px in enumerate(segments):
                        if pts_px.shape[0] < 2:
                            continue
                        path_id = (
                            f"section_profile_{i+1}"
                            if len(segments) == 1
                            else f"section_profile_{i+1}_{seg_i+1}"
                        )
                        extra_paths.append(
                            {
                                "id": path_id,
                                "points": pts_px,
                                "stroke": "#111111",
                                "stroke_width": 0.015,  # 0.15mm
                            }
                        )
            except Exception as e:
                _LOGGER.debug("Failed projecting section profiles to pixels: %s", e, exc_info=True)

        if feature_edges is not None:
            try:
                edges = np.asarray(getattr(feature_edges, "edges", []), dtype=np.int32)
                face_pairs = np.asarray(getattr(feature_edges, "face_pairs", []), dtype=np.int32)
            except Exception:
                edges = np.zeros((0, 2), dtype=np.int32)
                face_pairs = np.zeros((0, 2), dtype=np.int32)

            if edges.ndim == 2 and edges.shape[0] > 0 and edges.shape[1] >= 2:
                style = dict(feature_style or {})
                stroke = str(style.get("stroke", "#4a5568"))  # gray-600
                sw = float(style.get("stroke_width", 0.01))  # 0.1mm
                max_segments = int(style.get("max_segments", 20000))

                v = np.asarray(getattr(mesh, "vertices", np.zeros((0, 3))), dtype=np.float64)
                if v.ndim == 2 and v.shape[0] > 0:
                    keep = np.ones((edges.shape[0],), dtype=bool)

                    # Cheap back-face culling (if face normals are available)
                    if (
                        matrices_src is not None
                        and bool(bounds.get("is_pixels"))
                        and face_pairs.ndim == 2
                        and face_pairs.shape[0] == edges.shape[0]
                        and face_pairs.shape[1] >= 2
                    ):
                        try:
                            mv_raw, _proj_raw, _vp = matrices_src
                            mv = np.asarray(mv_raw, dtype=np.float64).reshape(4, 4).T

                            fn = getattr(mesh, "face_normals", None)
                            fn_arr = None
                            try:
                                fn_arr = np.asarray(fn, dtype=np.float64) if fn is not None else None
                            except Exception:
                                fn_arr = None

                            if fn_arr is not None and fn_arr.ndim == 2 and fn_arr.shape[1] >= 3 and fn_arr.shape[0] > 0:
                                fn3 = fn_arr[:, :3].copy()
                                if rotation is not None:
                                    from scipy.spatial.transform import Rotation as R

                                    r = R.from_euler("XYZ", rotation, degrees=True)
                                    fn3 = r.apply(fn3)

                                n_eye = fn3 @ mv[:3, :3].T
                                front = n_eye[:, 2] > 0.0

                                f0 = face_pairs[:, 0].astype(np.int32, copy=False)
                                f1 = face_pairs[:, 1].astype(np.int32, copy=False)
                                ok0 = (f0 >= 0) & (f0 < front.shape[0])
                                ok1 = (f1 >= 0) & (f1 < front.shape[0])
                                keep = (ok0 & front[f0]) | (ok1 & front[f1])
                        except Exception:
                            keep = np.ones((edges.shape[0],), dtype=bool)

                    idx = np.flatnonzero(keep)
                    if idx.size > 0:
                        if max_segments > 0 and idx.size > max_segments:
                            step = int(np.ceil(float(idx.size) / float(max_segments)))
                            idx = idx[:: max(1, step)]

                        e = edges[idx, :2].astype(np.int32, copy=False)
                        valid = np.all((e >= 0) & (e < int(v.shape[0])), axis=1)
                        if not bool(np.all(valid)):
                            e = e[valid]

                        if e.shape[0] > 0:
                            p0 = v[e[:, 0]]
                            p1 = v[e[:, 1]]

                            # Local -> world transform
                            pts = np.vstack([p0, p1]) * float(scale)
                            if rotation is not None:
                                from scipy.spatial.transform import Rotation as R

                                r = R.from_euler("XYZ", rotation, degrees=True)
                                pts = r.apply(pts)
                            if translation is not None:
                                pts = pts + np.asarray(translation, dtype=np.float64).reshape(1, 3)

                            if matrices_src is not None and bool(bounds.get("is_pixels")):
                                try:
                                    mv_raw, proj_raw, vp = matrices_src
                                    mv = np.asarray(mv_raw, dtype=np.float64).reshape(4, 4).T
                                    proj = np.asarray(proj_raw, dtype=np.float64).reshape(4, 4).T
                                    mvp = mv.T @ proj.T

                                    vp = np.asarray(vp, dtype=np.float64).reshape(-1)
                                    if vp.size < 4:
                                        vp = np.array([0.0, 0.0, 2048.0, 2048.0], dtype=np.float64)

                                    v_h = np.hstack([pts[:, :3], np.ones((pts.shape[0], 1), dtype=np.float64)])
                                    v_clip = v_h @ mvp
                                    w = v_clip[:, 3]
                                    valid_w = np.abs(w) > 1e-12
                                    xy = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
                                    if np.any(valid_w):
                                        v_ndc = v_clip[valid_w, :3] / w[valid_w, None]
                                        x = (v_ndc[:, 0] + 1.0) / 2.0 * float(vp[2])
                                        y = float(vp[3]) - (v_ndc[:, 1] + 1.0) / 2.0 * float(vp[3])
                                        xy[valid_w] = np.stack([x, y], axis=1)

                                    e_count = int(e.shape[0])
                                    seg = np.stack([xy[:e_count], xy[e_count:]], axis=1)
                                    ok = np.isfinite(seg).all(axis=(1, 2))
                                    if np.any(ok):
                                        seg = seg[ok]
                                        extra_paths.append(
                                            {
                                                "id": "feature_edges",
                                                "segments": seg,
                                                "stroke": stroke,
                                                "stroke_width": sw,
                                            }
                                        )
                                except Exception:
                                    _LOGGER.debug("Failed projecting feature edges to pixels", exc_info=True)
                            else:
                                # Fallback: project to plane coords (world units)
                                try:
                                    if view in self.VIEWS:
                                        ax0, ax1 = self.VIEWS[view]["axes"]
                                        seg_world = pts.reshape(2, -1, 3).transpose(1, 0, 2)[:, :, [ax0, ax1]]
                                        if view == "back":
                                            seg_world[:, :, 0] *= -1.0
                                        if view == "left":
                                            seg_world[:, :, 0] *= -1.0
                                        if view == "bottom":
                                            seg_world[:, :, 1] *= -1.0
                                        extra_paths.append(
                                            {
                                                "id": "feature_edges",
                                                "segments": seg_world[:, :, :2],
                                                "stroke": stroke,
                                                "stroke_width": sw,
                                            }
                                        )
                                except Exception:
                                    _LOGGER.debug("Failed projecting feature edges (fallback)", exc_info=True)

        return self.export_svg(
            contours, bounds, background_image,
            stroke_width=0.015,  # 0.15mm
            extra_paths=extra_paths,
            output_path=output_path
        )
