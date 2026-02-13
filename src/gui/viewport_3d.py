"""
3D Viewport Widget with OpenGL
Copyright (C) 2026 balguljang2 (lzpxilfe)
Licensed under the GNU General Public License v2.0 (GPL2)
"""

import logging
import sys
import time
import numpy as np
from typing import Any, Optional, cast
from scipy import ndimage

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QColor, QFont, QKeyEvent, QMouseEvent, QOpenGLContext, QPainter, QWheelEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLFramebufferObject

from OpenGL.GL import (
    GL_BACK,
    GL_ALL_ATTRIB_BITS,
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_ARRAY_BUFFER,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_BLEND,
    GL_CLIP_PLANE0,
    GL_CLIP_PLANE1,
    GL_CLIP_PLANE2,
    GL_CLIP_PLANE3,
    GL_CLIP_PLANE4,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_COMPONENT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FALSE,
    GL_FILL,
    GL_FLOAT,
    GL_FRONT_AND_BACK,
    GL_LIGHT_MODEL_AMBIENT,
    GL_LIGHT0,
    GL_LIGHT1,
    GL_LIGHTING,
    GL_LINE,
    GL_LINE_LOOP,
    GL_LINE_SMOOTH,
    GL_LINE_SMOOTH_HINT,
    GL_LINE_STRIP,
    GL_LINES,
    GL_MODELVIEW,
    GL_MODELVIEW_MATRIX,
    GL_NICEST,
    GL_NORMAL_ARRAY,
    GL_NORMALIZE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS,
    GL_POLYGON_OFFSET_FILL,
    GL_POSITION,
    GL_PROJECTION,
    GL_PROJECTION_MATRIX,
    GL_QUADS,
    GL_SHININESS,
    GL_SPECULAR,
    GL_SRC_ALPHA,
    GL_STATIC_DRAW,
    GL_TRIANGLE_FAN,
    GL_TRIANGLES,
    GL_UNSIGNED_INT,
    GL_TRUE,
    GL_VERTEX_ARRAY,
    GL_VIEWPORT,
    glBegin,
    glBindBuffer,
    glBlendFunc,
    glBufferData,
    glClear,
    glClearColor,
    glClipPlane,
    glColor3f,
    glColor4f,
    glColorMaterial,
    glCullFace,
    glDeleteBuffers,
    glDepthMask,
    glDisable,
    glDisableClientState,
    glDrawArrays,
    glDrawElements,
    glEnable,
    glEnableClientState,
    glEnd,
    glFlush,
    glGenBuffers,
    glGetDoublev,
    glGetIntegerv,
    glHint,
    glLightfv,
    glLightModelfv,
    glLineWidth,
    glLoadIdentity,
    glLoadMatrixd,
    glMaterialf,
    glMaterialfv,
    glMatrixMode,
    glNormalPointer,
    glNormal3f,
    glOrtho,
    glPointSize,
    glPolygonMode,
    glPolygonOffset,
    glPopAttrib,
    glPopMatrix,
    glPushAttrib,
    glPushMatrix,
    glReadPixels,
    glRotatef,
    glScalef,
    glTranslatef,
    glVertex3f,
    glVertex3fv,
    glVertexPointer,
    glViewport,
)
from OpenGL.GLU import gluLookAt, gluPerspective, gluProject, gluUnProject
import ctypes

from ..core.mesh_loader import MeshData
from ..core.mesh_slicer import MeshSlicer
from ..core.logging_utils import log_once

_LOGGER = logging.getLogger(__name__)
ORTHO_VIEW_SCALE_DEFAULT = 1.15


def _log_ignored_exception(context: str = "Ignored exception", *, level: int = logging.DEBUG) -> None:
    try:
        frame = sys._getframe(1)
        location = f"{frame.f_code.co_name}:{frame.f_lineno}"
        key = f"{__name__}:{location}"
    except Exception:
        location = "<unknown>"
        key = f"{__name__}:unknown"

    log_once(_LOGGER, key, level, "%s (%s)", context, location, exc_info=True)


class _CrosshairProfileThread(QThread):
    computed = pyqtSignal(object)  # {"cx","cy","x_profile","y_profile","world_x","world_y"}
    failed = pyqtSignal(str)

    def __init__(self, mesh: MeshData, translation: np.ndarray, rotation_deg: np.ndarray, scale: float, cx: float, cy: float):
        super().__init__()
        self._mesh = mesh
        self._translation = np.asarray(translation, dtype=np.float64)
        self._rotation = np.asarray(rotation_deg, dtype=np.float64)
        self._scale = float(scale)
        self._cx = float(cx)
        self._cy = float(cy)

    def run(self):
        try:
            from scipy.spatial.transform import Rotation as R

            obj = self._mesh.to_trimesh()
            slicer = MeshSlicer(obj)

            inv_rot = R.from_euler('XYZ', self._rotation, degrees=True).inv().as_matrix()
            inv_scale = 1.0 / self._scale if self._scale != 0 else 1.0

            def world_to_local(pt_world: np.ndarray) -> np.ndarray:
                return inv_scale * (inv_rot @ (pt_world - self._translation))

            rot_mat = R.from_euler('XYZ', self._rotation, degrees=True).as_matrix()

            def local_to_world(pts_local: np.ndarray) -> np.ndarray:
                return (rot_mat @ (pts_local * self._scale).T).T + self._translation

            # Plane X-profile (Y = cy): origin can be any point on plane
            w_orig_x = np.array([0.0, self._cy, 0.0], dtype=np.float64)
            w_norm_x = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            l_orig_x = world_to_local(w_orig_x)
            l_norm_x = inv_rot @ w_norm_x

            # Plane Y-profile (X = cx)
            w_orig_y = np.array([self._cx, 0.0, 0.0], dtype=np.float64)
            w_norm_y = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            l_orig_y = world_to_local(w_orig_y)
            l_norm_y = inv_rot @ w_norm_y

            contours_x = slicer.slice_with_plane(l_orig_x.tolist(), l_norm_x.tolist())
            contours_y = slicer.slice_with_plane(l_orig_y.tolist(), l_norm_y.tolist())

            x_profile = []
            y_profile = []
            world_x = np.zeros((0, 3), dtype=np.float64)
            world_y = np.zeros((0, 3), dtype=np.float64)

            if contours_x:
                pts_local = np.vstack(contours_x)
                pts_world = local_to_world(pts_local)
                order = np.argsort(pts_world[:, 0])
                world_x = pts_world[order]
                x_profile = world_x[:, [0, 2]].tolist()

            if contours_y:
                pts_local = np.vstack(contours_y)
                pts_world = local_to_world(pts_local)
                order = np.argsort(pts_world[:, 1])
                world_y = pts_world[order]
                y_profile = world_y[:, [1, 2]].tolist()

            self.computed.emit(
                {
                    "cx": self._cx,
                    "cy": self._cy,
                    "x_profile": x_profile,
                    "y_profile": y_profile,
                    "world_x": world_x,
                    "world_y": world_y,
                }
            )
        except Exception as e:
            self.failed.emit(str(e))


class _LineSectionProfileThread(QThread):
    computed = pyqtSignal(object)  # {"p0","p1","profile","contours"}
    failed = pyqtSignal(str)

    def __init__(self, mesh: MeshData, translation: np.ndarray, rotation_deg: np.ndarray, scale: float, p0: np.ndarray, p1: np.ndarray):
        super().__init__()
        self._mesh = mesh
        self._translation = np.asarray(translation, dtype=np.float64)
        self._rotation = np.asarray(rotation_deg, dtype=np.float64)
        self._scale = float(scale)
        self._p0 = np.asarray(p0, dtype=np.float64)
        self._p1 = np.asarray(p1, dtype=np.float64)

    def run(self):
        try:
            from scipy.spatial.transform import Rotation as R

            p0 = self._p0
            p1 = self._p1

            d = p1 - p0
            d[2] = 0.0
            length = float(np.linalg.norm(d))
            if length < 1e-6:
                self.computed.emit({"p0": p0, "p1": p1, "profile": [], "contours": []})
                return

            d_unit = d / length
            world_normal = np.array([d_unit[1], -d_unit[0], 0.0], dtype=np.float64)
            world_origin = p0

            inv_rot = R.from_euler('XYZ', self._rotation, degrees=True).inv().as_matrix()
            inv_scale = 1.0 / self._scale if self._scale != 0 else 1.0
            local_origin = inv_scale * inv_rot @ (world_origin - self._translation)
            local_normal = inv_rot @ world_normal

            slicer = MeshSlicer(self._mesh.to_trimesh())
            contours_local = slicer.slice_with_plane(local_origin, local_normal)

            rot_mat = R.from_euler('XYZ', self._rotation, degrees=True).as_matrix()
            trans = self._translation
            scale = self._scale

            world_contours = []
            for cnt in contours_local:
                world_contours.append((rot_mat @ (cnt * scale).T).T + trans)

            best_profile = []
            best_span = 0.0

            margin = max(length * 0.02, 0.2)
            t_min = -margin
            t_max = length + margin

            for cnt in world_contours:
                if cnt is None or len(cnt) < 2:
                    continue
                t = (cnt - p0) @ d_unit
                mask = (t >= t_min) & (t <= t_max)
                if int(mask.sum()) < 2:
                    continue

                t_f = t[mask]
                z_f = cnt[mask, 2]
                span = float(t_f.max() - t_f.min())
                if span <= best_span:
                    continue

                order = np.argsort(t_f)
                t_sorted = t_f[order]
                z_sorted = z_f[order]
                t_sorted = t_sorted - float(t_sorted.min())

                best_profile = list(zip(t_sorted.tolist(), z_sorted.tolist()))
                best_span = span

            self.computed.emit(
                {
                    "p0": p0,
                    "p1": p1,
                    "profile": best_profile,
                    "contours": world_contours,
                }
            )
        except Exception as e:
            self.failed.emit(str(e))


class _CutPolylineSectionProfileThread(QThread):
    computed = pyqtSignal(object)  # {"index","profile"} profile=[(s,z),...]
    failed = pyqtSignal(str)

    def __init__(
        self,
        mesh: MeshData,
        translation: np.ndarray,
        rotation_deg: np.ndarray,
        scale: float,
        index: int,
        polyline_world: list[np.ndarray] | list[list[float]],
    ):
        super().__init__()
        self._mesh = mesh
        self._translation = np.asarray(translation, dtype=np.float64)
        self._rotation = np.asarray(rotation_deg, dtype=np.float64)
        self._scale = float(scale)
        self._index = int(index)
        self._polyline_world = polyline_world

    def run(self):
        try:
            from scipy.spatial.transform import Rotation as R

            pts = np.asarray(self._polyline_world, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 2:
                self.computed.emit({"index": self._index, "profile": []})
                return
            if pts.shape[1] == 2:
                pts = np.hstack([pts, np.zeros((len(pts), 1), dtype=np.float64)])
            pts = pts[:, :3].copy()
            pts[:, 2] = 0.0

            # remove consecutive duplicates
            keep = [0]
            for i in range(1, len(pts)):
                if float(np.linalg.norm(pts[i, :2] - pts[keep[-1], :2])) > 1e-6:
                    keep.append(i)
            pts = pts[keep]
            if len(pts) < 2:
                self.computed.emit({"index": self._index, "profile": []})
                return

            inv_rot = R.from_euler('XYZ', self._rotation, degrees=True).inv().as_matrix()
            inv_scale = 1.0 / self._scale if self._scale != 0 else 1.0
            rot_mat = R.from_euler('XYZ', self._rotation, degrees=True).as_matrix()

            slicer = MeshSlicer(self._mesh.to_trimesh())

            def _extract_segment_profile(seg_p0: np.ndarray, seg_p1: np.ndarray) -> tuple[list[tuple[float, float]] | None, float]:
                d = (np.asarray(seg_p1, dtype=np.float64) - np.asarray(seg_p0, dtype=np.float64)).astype(np.float64)
                d[2] = 0.0
                seg_len = float(np.linalg.norm(d))
                if seg_len < 1e-6:
                    return None, 0.0

                d_unit = d / seg_len
                world_normal = np.array([d_unit[1], -d_unit[0], 0.0], dtype=np.float64)
                local_origin = inv_scale * inv_rot @ (np.asarray(seg_p0, dtype=np.float64) - self._translation)
                local_normal = inv_rot @ world_normal

                contours_local = slicer.slice_with_plane(local_origin.tolist(), local_normal.tolist())
                if not contours_local:
                    return None, seg_len

                best_profile = None
                best_score = -1.0

                for cnt in contours_local:
                    if cnt is None or len(cnt) < 2:
                        continue
                    arr = (rot_mat @ (np.asarray(cnt, dtype=np.float64).reshape(-1, 3) * self._scale).T).T + self._translation
                    if arr.shape[0] < 2:
                        continue

                    s = (arr - np.asarray(seg_p0, dtype=np.float64)) @ d_unit
                    z = arr[:, 2]
                    finite = np.isfinite(s) & np.isfinite(z)
                    s = s[finite]
                    z = z[finite]
                    if s.size < 2:
                        continue

                    ds = np.hypot(np.diff(s), np.diff(z))
                    keep_seg = np.ones((s.size,), dtype=bool)
                    keep_seg[1:] = ds > 1e-6
                    s = s[keep_seg]
                    z = z[keep_seg]
                    if s.size < 2:
                        continue

                    span_s = float(np.nanmax(s) - np.nanmin(s))
                    span_z = float(np.nanmax(z) - np.nanmin(z))
                    score = max(span_s, span_s * max(1e-6, span_z))
                    if s.size >= 3:
                        try:
                            s2 = np.append(s, s[0])
                            z2 = np.append(z, z[0])
                            area2 = float(np.dot(s2[:-1], z2[1:]) - np.dot(z2[:-1], s2[1:]))
                            score = max(score, abs(area2))
                        except Exception:
                            _log_ignored_exception()

                    if score > best_score:
                        best_score = score
                        best_profile = list(zip(s.tolist(), z.tolist()))

                return best_profile, seg_len

            merged_profile: list[tuple[float, float]] = []
            s_offset = 0.0

            for i in range(1, len(pts)):
                seg_profile, seg_len = _extract_segment_profile(pts[i - 1], pts[i])
                if seg_profile is not None and len(seg_profile) >= 2:
                    for s_val, z_val in seg_profile:
                        merged_profile.append((s_offset + float(s_val), float(z_val)))
                s_offset += max(0.0, float(seg_len))

            # Fallback: segment 湲곕컲 異붿텧???ㅽ뙣?섎㈃ 泥???吏곸꽑 湲곗??쇰줈 1???ъ떆??
            if len(merged_profile) < 2:
                seg_profile, _seg_len = _extract_segment_profile(pts[0], pts[-1])
                if seg_profile is not None:
                    merged_profile = [(float(s), float(z)) for s, z in seg_profile]

            cleaned: list[tuple[float, float]] = []
            for s_val, z_val in merged_profile:
                if not np.isfinite([s_val, z_val]).all():
                    continue
                if not cleaned:
                    cleaned.append((float(s_val), float(z_val)))
                    continue
                ps, pz = cleaned[-1]
                if float(np.hypot(float(s_val) - ps, float(z_val) - pz)) > 1e-6:
                    cleaned.append((float(s_val), float(z_val)))

            try:
                if len(cleaned) >= 2:
                    s_arr = np.asarray([p[0] for p in cleaned], dtype=np.float64)
                    if not np.isfinite(s_arr).all() or float(np.nanmax(s_arr) - np.nanmin(s_arr)) <= 1e-6:
                        cleaned = []
            except Exception:
                _log_ignored_exception()

            if len(cleaned) < 2:
                seg_profile, _seg_len = _extract_segment_profile(pts[0], pts[-1])
                cleaned = []
                if seg_profile is not None:
                    for s_val, z_val in seg_profile:
                        if np.isfinite([s_val, z_val]).all():
                            if not cleaned:
                                cleaned.append((float(s_val), float(z_val)))
                            else:
                                ps, pz = cleaned[-1]
                                if float(np.hypot(float(s_val) - ps, float(z_val) - pz)) > 1e-6:
                                    cleaned.append((float(s_val), float(z_val)))

            if len(cleaned) < 2:
                self.computed.emit({"index": self._index, "profile": []})
                return

            self.computed.emit({"index": self._index, "profile": cleaned})
        except Exception as e:
            self.failed.emit(str(e))


class _RoiCutEdgesThread(QThread):
    computed = pyqtSignal(object)  # {"x1": [...], "x2": [...], "y1": [...], "y2": [...]}
    failed = pyqtSignal(str)

    def __init__(
        self,
        mesh_or_tm: Any,
        translation: np.ndarray,
        rotation_deg: np.ndarray,
        scale: float,
        roi_bounds: list[float],
    ):
        super().__init__()
        self._mesh_or_tm = mesh_or_tm
        self._translation = np.asarray(translation, dtype=np.float64)
        self._rotation = np.asarray(rotation_deg, dtype=np.float64)
        self._scale = float(scale)
        self._roi_bounds = np.asarray(roi_bounds, dtype=np.float64).reshape(-1)

    @staticmethod
    def _clip_polyline_axis_interval(
        points: np.ndarray,
        *,
        axis: int,
        lo: float,
        hi: float,
        eps: float = 1e-6,
    ) -> list[np.ndarray]:
        """3D polyline瑜???異?axis)??[lo, hi] 援ш컙?쇰줈 ?대━?묓빐 議곌컖?ㅻ줈 諛섑솚."""
        try:
            pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        except Exception:
            return []
        if pts.shape[0] < 2:
            return []

        lo_v = float(min(lo, hi))
        hi_v = float(max(lo, hi))
        eps_v = float(max(1e-12, eps))

        # closed contour??留덉?留?以묐났???쒓굅 ??留덉?留?泥섏쓬 ?멸렇癒쇳듃源뚯? 泥섎━
        closed = False
        try:
            if float(np.linalg.norm(pts[0] - pts[-1])) <= eps_v:
                closed = True
                pts = pts[:-1]
        except Exception:
            closed = False
        if pts.shape[0] < 2:
            return []

        segments: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(pts.shape[0] - 1):
            segments.append((pts[i], pts[i + 1]))
        if closed:
            segments.append((pts[-1], pts[0]))

        def inside(v: float) -> bool:
            return (v >= lo_v - eps_v) and (v <= hi_v + eps_v)

        def interp(p0: np.ndarray, p1: np.ndarray, v_target: float) -> np.ndarray:
            v0 = float(p0[axis])
            v1 = float(p1[axis])
            dv = float(v1 - v0)
            if abs(dv) <= eps_v:
                return np.asarray(p0, dtype=np.float64)
            t = float((v_target - v0) / dv)
            t = float(np.clip(t, 0.0, 1.0))
            return p0 + t * (p1 - p0)

        out: list[np.ndarray] = []
        cur: list[np.ndarray] = []

        def push_cur() -> None:
            nonlocal cur
            if len(cur) < 2:
                cur = []
                return
            arr = np.asarray(cur, dtype=np.float64).reshape(-1, 3)
            # ?곗냽 以묐났 ?쒓굅
            if arr.shape[0] >= 2:
                d = np.linalg.norm(np.diff(arr, axis=0), axis=1)
                keep = np.ones((arr.shape[0],), dtype=bool)
                keep[1:] = d > eps_v
                arr = arr[keep]
            if arr.shape[0] >= 2:
                out.append(arr)
            cur = []

        for p0, p1 in segments:
            v0 = float(p0[axis])
            v1 = float(p1[axis])
            in0 = inside(v0)
            in1 = inside(v1)

            if in0 and in1:
                if not cur:
                    cur.append(np.asarray(p0, dtype=np.float64))
                cur.append(np.asarray(p1, dtype=np.float64))
                continue

            if in0 and not in1:
                # inside -> outside: 寃쎄퀎?먯뿉??醫낅즺
                hit = interp(p0, p1, hi_v if v1 > hi_v else lo_v)
                if not cur:
                    cur.append(np.asarray(p0, dtype=np.float64))
                cur.append(np.asarray(hit, dtype=np.float64))
                push_cur()
                continue

            if (not in0) and in1:
                # outside -> inside: 寃쎄퀎?먯뿉???쒖옉
                hit = interp(p0, p1, hi_v if v0 > hi_v else lo_v)
                cur.append(np.asarray(hit, dtype=np.float64))
                cur.append(np.asarray(p1, dtype=np.float64))
                continue

            # outside -> outside: 援ш컙??愿?듯븯硫???援먯감?먯쑝濡?1媛?議곌컖 ?앹꽦
            if (v0 < lo_v and v1 > hi_v) or (v0 > hi_v and v1 < lo_v):
                h0 = interp(p0, p1, lo_v)
                h1 = interp(p0, p1, hi_v)
                if float(h0[axis]) > float(h1[axis]):
                    h0, h1 = h1, h0
                out.append(np.asarray([h0, h1], dtype=np.float64))
                continue

            # outside -> outside and no crossing
            push_cur()

        push_cur()
        return out

    @classmethod
    def _apply_roi_interaction_filters(
        cls,
        edges: dict[str, list[np.ndarray]],
        *,
        x1: float,
        x2: float,
        y1: float,
        y2: float,
    ) -> dict[str, list[np.ndarray]]:
        """媛?ROI 寃쎄퀎 ?⑤㈃???ㅻⅨ 異??덈떒 寃곌낵? ?곹샇 諛섏쁺?섎룄濡??대━??"""
        out: dict[str, list[np.ndarray]] = {"x1": [], "x2": [], "y1": [], "y2": []}

        # x-plane ?⑤㈃? y 援ш컙?쇰줈, y-plane ?⑤㈃? x 援ш컙?쇰줈 ?섎씪??        # ??諛⑺뼢 ?덈떒???곹샇?묒슜???ㅼ젣 ?⑤㈃ 紐⑥뼇???⑸땲??
        for key in ("x1", "x2"):
            for cnt in edges.get(key, []) or []:
                out[key].extend(
                    cls._clip_polyline_axis_interval(
                        cnt,
                        axis=1,  # y
                        lo=float(y1),
                        hi=float(y2),
                        eps=1e-5,
                    )
                )
        for key in ("y1", "y2"):
            for cnt in edges.get(key, []) or []:
                out[key].extend(
                    cls._clip_polyline_axis_interval(
                        cnt,
                        axis=0,  # x
                        lo=float(x1),
                        hi=float(x2),
                        eps=1e-5,
                    )
                )
        return out

    def run(self):
        try:
            from scipy.spatial.transform import Rotation as R

            if self._roi_bounds.size < 4:
                self.computed.emit({"x1": [], "x2": [], "y1": [], "y2": []})
                return

            x1, x2, y1, y2 = [float(v) for v in self._roi_bounds[:4]]
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            inv_rot = R.from_euler('XYZ', self._rotation, degrees=True).inv().as_matrix()
            inv_scale = 1.0 / self._scale if self._scale != 0 else 1.0
            rot_mat = R.from_euler('XYZ', self._rotation, degrees=True).as_matrix()

            tm = self._mesh_or_tm
            if isinstance(tm, MeshData):
                tm = tm.to_trimesh()
            if tm is None:
                self.computed.emit({"x1": [], "x2": [], "y1": [], "y2": []})
                return
            slicer = MeshSlicer(cast(Any, tm))

            def slice_world_plane(world_origin: np.ndarray, world_normal: np.ndarray):
                world_origin = np.asarray(world_origin, dtype=np.float64)
                world_normal = np.asarray(world_normal, dtype=np.float64)
                local_origin = inv_scale * inv_rot @ (world_origin - self._translation)
                local_normal = inv_rot @ world_normal
                contours_local = slicer.slice_with_plane(local_origin, local_normal)
                out = []
                for cnt in contours_local or []:
                    if cnt is None or len(cnt) < 2:
                        continue
                    out.append((rot_mat @ (cnt * self._scale).T).T + self._translation)
                return out

            # 4媛쒖쓽 寃쎄퀎 ?됰㈃?먯꽌 ?⑤㈃??寃쎄퀎?? 異붿텧
            edges = {
                "x1": slice_world_plane(np.array([x1, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
                "x2": slice_world_plane(np.array([x2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
                "y1": slice_world_plane(np.array([0.0, y1, 0.0]), np.array([0.0, 1.0, 0.0])),
                "y2": slice_world_plane(np.array([0.0, y2, 0.0]), np.array([0.0, 1.0, 0.0])),
            }
            # ?듭떖: 媛?諛⑺뼢 ?⑤㈃??"?ㅻⅨ 諛⑺뼢 ?덈떒"怨??곹샇 諛섏쁺?섍쾶 ?꾪꽣留?
            clipped = self._apply_roi_interaction_filters(
                edges,
                x1=float(x1),
                x2=float(x2),
                y1=float(y1),
                y2=float(y2),
            )
            self.computed.emit(clipped)
        except Exception as e:
            self.failed.emit(str(e))


class _SurfaceLassoSelectThread(QThread):
    computed = pyqtSignal(object)  # {"indices": np.ndarray, "stats": {...}}
    failed = pyqtSignal(str)

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        translation: np.ndarray,
        rotation_deg: np.ndarray,
        scale: float,
        camera_pos_world: np.ndarray | None,
        modelview: np.ndarray,
        projection: np.ndarray,
        viewport: np.ndarray,
        poly_gl: np.ndarray,
        bbox_gl: tuple[int, int, int, int],
        depth_origin: tuple[int, int],
        depth_map: np.ndarray | None,
        *,
        face_centroids: np.ndarray | None = None,
        face_normals: np.ndarray | None = None,
        depth_tol: float = 0.01,
        max_selected_faces: int = 400000,
        wand_seed_face_idx: int | None = None,
        wand_enabled: bool = True,
        wand_angle_deg: float = 30.0,
        wand_knn_k: int = 12,
        wand_time_budget_s: float = 2.0,
    ):
        super().__init__()
        self._vertices = np.asarray(vertices)
        self._faces = np.asarray(faces)
        self._translation = np.asarray(translation, dtype=np.float64).reshape(-1)
        self._rotation = np.asarray(rotation_deg, dtype=np.float64).reshape(-1)
        self._scale = float(scale)
        self._camera_pos_world = None if camera_pos_world is None else np.asarray(camera_pos_world, dtype=np.float64).reshape(-1)
        self._mv = np.asarray(modelview, dtype=np.float64).reshape(4, 4)
        self._proj = np.asarray(projection, dtype=np.float64).reshape(4, 4)
        self._viewport = np.asarray(viewport, dtype=np.int32).reshape(-1)
        self._poly = np.asarray(poly_gl, dtype=np.float64).reshape(-1, 2)
        self._bbox_gl = (
            int(bbox_gl[0]),
            int(bbox_gl[1]),
            int(bbox_gl[2]),
            int(bbox_gl[3]),
        )
        self._depth_origin = (int(depth_origin[0]), int(depth_origin[1]))
        self._depth = None if depth_map is None else np.asarray(depth_map)
        self._depth_tol = float(depth_tol)
        self._max_selected = int(max_selected_faces)

        self._face_centroids: np.ndarray | None = None
        try:
            if face_centroids is not None:
                fc = np.asarray(face_centroids)
                if fc.ndim == 2 and int(fc.shape[1]) >= 3:
                    self._face_centroids = fc[:, :3]
        except Exception:
            self._face_centroids = None

        self._face_normals: np.ndarray | None = None
        try:
            if face_normals is not None:
                fn = np.asarray(face_normals)
                if fn.ndim == 2 and int(fn.shape[1]) >= 3:
                    self._face_normals = fn[:, :3]
        except Exception:
            self._face_normals = None
        try:
            self._wand_seed = int(wand_seed_face_idx) if wand_seed_face_idx is not None else -1
        except Exception:
            self._wand_seed = -1
        self._wand_enabled = bool(wand_enabled)
        self._wand_angle_deg = float(wand_angle_deg)
        self._wand_knn_k = int(wand_knn_k)
        self._wand_time_budget_s = float(wand_time_budget_s)

    def _component_refine(
        self,
        faces: np.ndarray,
        candidate_faces: np.ndarray,
        *,
        seed_face_idx: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Keep only the face-edge-connected component that contains the seed face (within candidates)."""
        stats: dict[str, Any] = {}
        try:
            if candidate_faces is None:
                return np.zeros((0,), dtype=np.int32), stats
            cand = np.asarray(candidate_faces, dtype=np.int32).reshape(-1)
            if cand.size < 2:
                return cand, stats

            try:
                max_faces = int(getattr(self, "_component_max_faces", 500000))
            except Exception:
                max_faces = 500000
            max_faces = max(10, int(max_faces))
            if int(cand.size) > int(max_faces):
                return cand, {"component_skipped": True, "component_candidates": int(cand.size), "component_max_faces": int(max_faces)}

            seed = int(seed_face_idx)
            if seed < 0 or seed >= int(faces.shape[0]):
                seed = int(cand[0])

            seed_pos_arr = np.where(cand == seed)[0]
            seed_pos = int(seed_pos_arr[0]) if seed_pos_arr.size else 0

            f = np.asarray(faces[cand, :3], dtype=np.int32)
            if f.ndim != 2 or f.shape[1] < 3:
                return cand, stats
            a = f[:, 0]
            b = f[:, 1]
            c = f[:, 2]

            # Build undirected edge list for faces in candidate set.
            e0 = np.stack([a, b], axis=1)
            e1 = np.stack([b, c], axis=1)
            e2 = np.stack([c, a], axis=1)
            edges = np.concatenate([e0, e1, e2], axis=0)
            edges = np.sort(edges, axis=1)

            face_ids = np.repeat(np.arange(int(f.shape[0]), dtype=np.int32), 3)

            # Sort edges and connect faces that share the same edge.
            order = np.lexsort((edges[:, 1], edges[:, 0]))
            edges_s = edges[order]
            faces_s = face_ids[order]
            same = np.all(edges_s[1:] == edges_s[:-1], axis=1)
            if not np.any(same):
                out = np.asarray([int(cand[seed_pos])], dtype=np.int32)
                return out, {"component_selected": int(out.size), "component_candidates": int(cand.size), "component_components": int(cand.size)}

            u = faces_s[:-1][same]
            v = faces_s[1:][same]

            try:
                from scipy import sparse as _sparse
                from scipy.sparse import csgraph as _csgraph
            except Exception:
                return cand, stats

            row = np.concatenate([u, v]).astype(np.int32, copy=False)
            col = np.concatenate([v, u]).astype(np.int32, copy=False)
            data = np.ones((row.size,), dtype=np.int8)
            g = _sparse.coo_matrix((data, (row, col)), shape=(int(cand.size), int(cand.size))).tocsr()

            order2 = cast(Any, _csgraph).breadth_first_order(g, int(seed_pos), directed=False, return_predecessors=False)
            order2 = np.asarray(order2, dtype=np.int32).reshape(-1)

            keep = np.zeros((int(cand.size),), dtype=bool)
            if order2.size:
                keep[order2] = True
            out = cand[keep]

            stats = {
                "component_seed": int(seed),
                "component_selected": int(out.size),
                "component_candidates": int(cand.size),
            }
            return out, stats
        except Exception:
            return candidate_faces, stats

    def _wand_refine(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        candidate_faces: np.ndarray,
        *,
        seed_face_idx: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        stats: dict[str, Any] = {}
        try:
            if not bool(self._wand_enabled):
                return candidate_faces, stats
            if candidate_faces is None:
                return np.zeros((0,), dtype=np.int32), stats
            cand = np.asarray(candidate_faces, dtype=np.int32).reshape(-1)
            if cand.size < 1:
                return cand, stats

            seed = int(seed_face_idx)
            if seed < 0 or seed >= int(faces.shape[0]):
                seed = int(cand[0])

            # Centroids and normals for candidate faces (local coords)
            face_centroids_all = getattr(self, "_face_centroids", None)
            face_normals_all = getattr(self, "_face_normals", None)

            cent: np.ndarray | None = None
            normals: np.ndarray | None = None

            if face_centroids_all is not None:
                try:
                    fc = np.asarray(face_centroids_all)
                    if fc.ndim == 2 and int(fc.shape[0]) == int(faces.shape[0]) and int(fc.shape[1]) >= 3:
                        cent = np.asarray(fc[cand, :3], dtype=np.float32)
                except Exception:
                    cent = None

            if face_normals_all is not None:
                try:
                    fn = np.asarray(face_normals_all)
                    if fn.ndim == 2 and int(fn.shape[0]) == int(faces.shape[0]) and int(fn.shape[1]) >= 3:
                        normals = np.asarray(fn[cand, :3], dtype=np.float32)
                        nn = np.linalg.norm(normals, axis=1)
                        nn = np.where(nn > 1e-12, nn, 1.0)
                        normals = (normals / nn[:, None]).astype(np.float32, copy=False)
                except Exception:
                    normals = None

            if cent is None or normals is None:
                f = faces[cand, :3].astype(np.int32, copy=False)
                v0 = vertices[f[:, 0], :3].astype(np.float64, copy=False)
                v1 = vertices[f[:, 1], :3].astype(np.float64, copy=False)
                v2 = vertices[f[:, 2], :3].astype(np.float64, copy=False)
                if cent is None:
                    cent = ((v0 + v1 + v2) / 3.0).astype(np.float32, copy=False)
                if normals is None:
                    n = np.cross(v1 - v0, v2 - v0)
                    nn = np.linalg.norm(n, axis=1)
                    nn = np.where(nn > 1e-12, nn, 1.0)
                    normals = (n / nn[:, None]).astype(np.float32, copy=False)

            # KNN adjacency on centroids
            k = int(max(2, self._wand_knn_k))
            k = min(k, int(cent.shape[0]))
            if k <= 1:
                return cand, {"wand_selected": int(cand.size), "wand_seed": seed}

            try:
                from scipy import spatial as _spatial

                tree = cast(Any, _spatial).cKDTree(cent)
                _d, nbr = tree.query(cent, k=k)
                nbr = np.asarray(nbr, dtype=np.int32)
            except Exception:
                return cand, stats

            # Seed position in candidate list
            seed_pos_arr = np.where(cand == seed)[0]
            if seed_pos_arr.size:
                seed_pos = int(seed_pos_arr[0])
            else:
                # Nearest candidate to seed face centroid
                try:
                    sc = None
                    if face_centroids_all is not None:
                        try:
                            fc0 = np.asarray(face_centroids_all)
                            if fc0.ndim == 2 and int(fc0.shape[0]) == int(faces.shape[0]) and int(fc0.shape[1]) >= 3:
                                sc = np.asarray(fc0[int(seed), :3], dtype=np.float32).reshape(-1)
                        except Exception:
                            sc = None
                    if sc is None:
                        sf = faces[seed, :3].astype(np.int32, copy=False)
                        sc = (
                            (vertices[int(sf[0]), :3] + vertices[int(sf[1]), :3] + vertices[int(sf[2]), :3])
                            / 3.0
                        ).astype(np.float32, copy=False)
                    _d2, seed_pos2 = tree.query(np.asarray(sc, dtype=np.float32), k=1)
                    seed_pos = int(np.asarray(seed_pos2).reshape(-1)[0])
                except Exception:
                    seed_pos = 0

            # Region grow by local normal smoothness (magic-wand-like)
            from collections import deque

            cos_thr = float(np.cos(np.radians(float(self._wand_angle_deg))))
            time_budget = float(max(0.1, self._wand_time_budget_s))

            visited = np.zeros((cand.size,), dtype=bool)
            visited[seed_pos] = True
            q = deque([seed_pos])
            visited_count = 1
            start_t = time.monotonic()

            while q:
                if self.isInterruptionRequested():
                    break
                if time.monotonic() - start_t > time_budget:
                    break
                i = int(q.popleft())
                n0 = normals[i]
                nb = nbr[i]
                if nb.ndim == 0:
                    continue
                nb = nb[nb != i]
                if nb.size == 0:
                    continue
                mask = ~visited[nb]
                nb = nb[mask]
                if nb.size == 0:
                    continue
                dots = (normals[nb] @ n0.astype(np.float32, copy=False)).astype(np.float32, copy=False)
                good = nb[dots >= cos_thr]
                if good.size:
                    visited[good] = True
                    visited_count += int(good.size)
                    q.extend(int(x) for x in good.tolist())

            out = cand[visited]
            stats = {
                "wand_seed": int(seed),
                "wand_selected": int(out.size),
                "wand_candidates": int(cand.size),
                "wand_angle_deg": float(self._wand_angle_deg),
                "wand_time_s": float(time.monotonic() - start_t),
            }
            return out, stats
        except Exception:
            return candidate_faces, stats

    def run(self):
        try:
            faces = np.asarray(self._faces, dtype=np.int32)
            vertices = np.asarray(self._vertices, dtype=np.float64)
            if faces.ndim != 2 or faces.shape[1] < 3 or vertices.ndim != 2 or vertices.shape[1] < 3:
                self.computed.emit({"indices": np.zeros((0,), dtype=np.int32), "stats": {"selected": 0}})
                return

            n_faces = int(faces.shape[0])
            if n_faces <= 0:
                self.computed.emit({"indices": np.zeros((0,), dtype=np.int32), "stats": {"selected": 0}})
                return

            poly = np.asarray(self._poly, dtype=np.float64)
            if poly.shape[0] < 3:
                self.computed.emit({"indices": np.zeros((0,), dtype=np.int32), "stats": {"selected": 0}})
                return

            vp = np.asarray(self._viewport, dtype=np.int32).reshape(-1)
            if vp.size < 4:
                self.computed.emit({"indices": np.zeros((0,), dtype=np.int32), "stats": {"selected": 0}})
                return

            vx, vy, vw, vh = [int(v) for v in vp[:4]]
            vw = max(1, vw)
            vh = max(1, vh)
            mv = np.asarray(self._mv, dtype=np.float64)
            proj = np.asarray(self._proj, dtype=np.float64)
            # Row-vector projection matrix: (P*M*v)^T = v^T*M^T*P^T
            try:
                mvp_t = mv.T @ proj.T
            except Exception:
                mvp_t = None

            depth = self._depth
            if depth is not None:
                depth = np.asarray(depth, dtype=np.float32)
                if depth.ndim >= 3:
                    depth = np.squeeze(depth)
                if depth.ndim != 2:
                    depth = None

            dx0, dy0 = self._depth_origin

            # BBox in GL coords (clamped to viewport)
            bb = self._bbox_gl
            bbox_x0 = int(np.clip(bb[0], vx, vx + vw - 1))
            bbox_y0 = int(np.clip(bb[1], vy, vy + vh - 1))
            bbox_x1 = int(np.clip(bb[2], vx, vx + vw - 1))
            bbox_y1 = int(np.clip(bb[3], vy, vy + vh - 1))

            poly_x = poly[:, 0]
            poly_y = poly[:, 1]

            def points_in_poly(x: np.ndarray, y: np.ndarray) -> np.ndarray:
                inside = np.zeros_like(x, dtype=bool)
                j = int(len(poly_x) - 1)
                for i in range(int(len(poly_x))):
                    xi = float(poly_x[i])
                    yi = float(poly_y[i])
                    xj = float(poly_x[j])
                    yj = float(poly_y[j])
                    denom = (yj - yi) if abs(yj - yi) > 1e-12 else 1e-12
                    intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / denom + xi)
                    inside ^= intersect
                    j = i
                return inside

            # Object transform (local -> world)
            trans = self._translation
            if trans.size < 3:
                trans = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            rot = self._rotation
            if rot.size < 3:
                rot = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            scale = float(self._scale) if abs(float(self._scale)) > 1e-12 else 1.0

            rx, ry, rz = np.radians(rot[:3])
            cx, sx = float(np.cos(rx)), float(np.sin(rx))
            cy, sy = float(np.cos(ry)), float(np.sin(ry))
            cz, sz = float(np.cos(rz)), float(np.sin(rz))
            rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
            rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
            rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            rot_mat = rot_x @ rot_y @ rot_z

            # Camera position in local coords (for front-face filtering).
            cam_local: np.ndarray | None = None
            try:
                cam_w = self._camera_pos_world
                if cam_w is not None and cam_w.size >= 3 and np.isfinite(cam_w[:3]).all():
                    cam_local = (rot_mat.T @ (cam_w[:3] - trans[:3])) / float(scale)
            except Exception:
                cam_local = None

            tol = float(max(0.0, self._depth_tol))
            max_sel = max(1, int(self._max_selected))

            face_centroids_all: np.ndarray | None = getattr(self, "_face_centroids", None)
            if (
                face_centroids_all is not None
                and face_centroids_all.ndim == 2
                and int(face_centroids_all.shape[0]) == n_faces
                and int(face_centroids_all.shape[1]) >= 3
            ):
                face_centroids_all = face_centroids_all[:, :3]
            else:
                face_centroids_all = None

            face_normals_all: np.ndarray | None = getattr(self, "_face_normals", None)
            if (
                face_normals_all is not None
                and face_normals_all.ndim == 2
                and int(face_normals_all.shape[0]) == n_faces
                and int(face_normals_all.shape[1]) >= 3
            ):
                face_normals_all = face_normals_all[:, :3]
            else:
                face_normals_all = None

            out: list[np.ndarray] = []
            out_count = 0
            out_no_depth: list[np.ndarray] = []
            out_no_depth_count = 0
            out_no_front: list[np.ndarray] = []
            out_no_front_count = 0
            total_candidates = 0

            chunk = 200000
            for start in range(0, n_faces, chunk):
                if self.isInterruptionRequested():
                    break
                end = min(n_faces, start + chunk)
                f = faces[start:end, :3]
                if f.size == 0:
                    continue

                # Face centroids in local coords
                v0 = v1 = v2 = None
                if face_centroids_all is not None:
                    cent = face_centroids_all[start:end, :3]
                else:
                    try:
                        v0 = vertices[f[:, 0], :3]
                        v1 = vertices[f[:, 1], :3]
                        v2 = vertices[f[:, 2], :3]
                    except Exception:
                        continue
                    cent = (v0 + v1 + v2) / 3.0

                # Local -> world
                cent_w = (rot_mat @ (cent * scale).T).T + trans[:3]

                # Project (row-vector convention)
                ones = np.ones((cent_w.shape[0], 1), dtype=np.float64)
                v_h = np.hstack([cent_w.astype(np.float64, copy=False), ones])
                if mvp_t is None:
                    clip = v_h @ mv @ proj
                else:
                    clip = v_h @ mvp_t
                w = clip[:, 3]
                valid = np.isfinite(w) & (w > 1e-12)
                if not np.any(valid):
                    continue

                idx_chunk = np.arange(start, end, dtype=np.int32)
                idx_valid = idx_chunk[valid]
                clip_v = clip[valid, :]
                w_v = w[valid]

                ndc = clip_v[:, :3] / w_v[:, None]
                win_x = (ndc[:, 0] + 1.0) * 0.5 * float(vw) + float(vx)
                win_y = (ndc[:, 1] + 1.0) * 0.5 * float(vh) + float(vy)
                win_z = (ndc[:, 2] + 1.0) * 0.5

                ok = (
                    np.isfinite(win_x)
                    & np.isfinite(win_y)
                    & np.isfinite(win_z)
                    & (win_z >= 0.0)
                    & (win_z <= 1.0)
                    & (win_x >= float(bbox_x0) - 1.0)
                    & (win_x <= float(bbox_x1) + 1.0)
                    & (win_y >= float(bbox_y0) - 1.0)
                    & (win_y <= float(bbox_y1) + 1.0)
                )
                if not np.any(ok):
                    continue

                idx_ok = idx_valid[ok]
                x_ok = win_x[ok]
                y_ok = win_y[ok]
                z_ok = win_z[ok]

                inside = points_in_poly(x_ok, y_ok)
                if not np.any(inside):
                    continue

                idx_in = idx_ok[inside]
                x_in = x_ok[inside]
                y_in = y_ok[inside]
                z_in = z_ok[inside]
                total_candidates += int(idx_in.size)

                # Front-face filter: keep faces whose (local) normal points toward the camera.
                if cam_local is not None and (depth is None or depth.size == 0) and idx_in.size:
                    # Keep a fallback set (no front filtering) in case normals are flipped / inconsistent.
                    if out_no_front_count < max_sel:
                        try:
                            remain = max_sel - out_no_front_count
                            if remain > 0:
                                take = idx_in[:remain]
                                out_no_front.append(np.asarray(take, dtype=np.int32, order="C"))
                                out_no_front_count += int(take.size)
                        except Exception:
                            _log_ignored_exception()

                    try:
                        cent_in = cent[valid, :][ok, :][inside, :][:, :3]
                        view_vec = (cam_local[:3].reshape(1, 3) - cent_in[:, :3]).astype(np.float64, copy=False)

                        if face_normals_all is not None:
                            n_local = np.asarray(face_normals_all[idx_in, :3], dtype=np.float64)
                        else:
                            f_in = faces[idx_in, :3]
                            v0_in = vertices[f_in[:, 0], :3]
                            v1_in = vertices[f_in[:, 1], :3]
                            v2_in = vertices[f_in[:, 2], :3]
                            n_local = np.cross(v1_in - v0_in, v2_in - v0_in)

                        dots = np.einsum("ij,ij->i", n_local.astype(np.float64, copy=False), view_vec)
                        keep_front = dots > 1e-12
                        if np.any(keep_front):
                            idx_in = idx_in[keep_front]
                            x_in = x_in[keep_front]
                            y_in = y_in[keep_front]
                            z_in = z_in[keep_front]
                        else:
                            idx_in = idx_in[:0]
                            x_in = x_in[:0]
                            y_in = y_in[:0]
                            z_in = z_in[:0]
                    except Exception:
                        _log_ignored_exception()

                if depth is not None and depth.size != 0:
                    # Keep a fallback set (no depth filtering) in case depth test is too strict/mismatched.
                    if idx_in.size and out_no_depth_count < max_sel:
                        try:
                            remain = max_sel - out_no_depth_count
                            if remain > 0:
                                take = idx_in[:remain]
                                out_no_depth.append(np.asarray(take, dtype=np.int32, order="C"))
                                out_no_depth_count += int(take.size)
                        except Exception:
                            _log_ignored_exception()

                    # Depth test against current depth buffer (keep visible-ish faces)
                    px = np.rint(x_in).astype(np.int32)
                    py = np.rint(y_in).astype(np.int32)
                    px = np.clip(px, bbox_x0, bbox_x1) - int(dx0)
                    py = np.clip(py, bbox_y0, bbox_y1) - int(dy0)
                    try:
                        dpx = depth[py, px]
                    except Exception:
                        dpx = None
                    if dpx is None:
                        keep = np.ones((idx_in.size,), dtype=bool)
                    else:
                        dpx = np.asarray(dpx, dtype=np.float32)
                        keep = np.isfinite(dpx) & (dpx < 1.0) & (z_in <= (dpx.astype(np.float64) + tol))
                    idx_in = idx_in[keep]

                if idx_in.size:
                    out.append(np.asarray(idx_in, dtype=np.int32, order="C"))
                    out_count += int(idx_in.size)
                    if out_count >= max_sel:
                        break

            if not out:
                # Front-face filter fallback: if normals were unusable and everything got filtered out,
                # fall back to no front filtering (still respects depth if it was applied).
                if out_no_front and (depth is None or not out_no_depth):
                    indices = np.unique(np.concatenate(out_no_front)).astype(np.int32, copy=False)
                    if indices.size > max_sel:
                        indices = indices[:max_sel].copy()
                    self.computed.emit(
                        {
                            "indices": indices,
                            "stats": {
                                "selected": int(indices.size),
                                "candidates": int(total_candidates),
                                "front_fallback": True,
                            },
                        }
                    )
                    return
                if depth is not None and out_no_depth:
                    indices = np.unique(np.concatenate(out_no_depth)).astype(np.int32, copy=False)
                    if indices.size > max_sel:
                        indices = indices[:max_sel].copy()
                    self.computed.emit(
                        {
                            "indices": indices,
                            "stats": {
                                "selected": int(indices.size),
                                "candidates": int(total_candidates),
                                "depth_fallback": True,
                            },
                        }
                    )
                    return
                else:
                    self.computed.emit(
                        {
                            "indices": np.zeros((0,), dtype=np.int32),
                            "stats": {"selected": 0, "candidates": int(total_candidates)},
                        }
                    )
                    return

            indices = np.unique(np.concatenate(out)).astype(np.int32, copy=False)
            truncated = bool((out_count >= max_sel) and (max_sel < n_faces))
            if indices.size > max_sel:
                indices = indices[:max_sel].copy()
                truncated = True

            comp_stats: dict[str, Any] = {}
            try:
                seed0 = int(getattr(self, "_wand_seed", -1))
                if seed0 >= 0 and indices.size:
                    indices, comp_stats = self._component_refine(faces, indices, seed_face_idx=seed0)
                    indices = np.unique(np.asarray(indices, dtype=np.int32).reshape(-1)).astype(np.int32, copy=False)
                    if indices.size > max_sel:
                        indices = indices[:max_sel].copy()
                        truncated = True
            except Exception:
                comp_stats = {}

            # Magic-wand-like refinement: keep the smooth connected patch inside the polygon.
            wand_stats: dict[str, Any] = {}
            try:
                seed = int(getattr(self, "_wand_seed", -1))
                if seed >= 0 and bool(getattr(self, "_wand_enabled", True)) and indices.size:
                    indices, wand_stats = self._wand_refine(vertices, faces, indices, seed_face_idx=seed)
                    indices = np.unique(np.asarray(indices, dtype=np.int32).reshape(-1)).astype(np.int32, copy=False)
                    if indices.size > max_sel:
                        indices = indices[:max_sel].copy()
                        truncated = True
            except Exception:
                wand_stats = {}

            self.computed.emit(
                {
                    "indices": indices,
                    "stats": {
                        "selected": int(indices.size),
                        "candidates": int(total_candidates),
                        "truncated": bool(truncated),
                        "max_selected_faces": int(max_sel),
                        **(comp_stats or {}),
                        **(wand_stats or {}),
                    },
                }
            )
        except Exception as e:
            self.failed.emit(str(e))


class TrackballCamera:
    """Trackball camera in a Z-up world coordinate system."""

    def __init__(self):
        self.distance = 50.0
        self.azimuth = 45.0
        self.elevation = 30.0
        self.center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.min_distance = 0.01
        self.max_distance = 1000000.0

    @property
    def position(self) -> np.ndarray:
        """Camera world position."""
        az_rad = np.radians(self.azimuth)
        el_rad = np.radians(self.elevation)
        x = self.distance * np.cos(el_rad) * np.cos(az_rad)
        y = self.distance * np.cos(el_rad) * np.sin(az_rad)
        z = self.distance * np.sin(el_rad)
        return np.array([x, y, z], dtype=np.float64) + self.center + self.pan_offset

    @property
    def up_vector(self) -> np.ndarray:
        """Camera up vector (top/bottom views use Y-up)."""
        if abs(self.elevation) > 85:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    @property
    def look_at(self) -> np.ndarray:
        """Camera look-at point."""
        return self.center + self.pan_offset

    def rotate(self, delta_x: float, delta_y: float, sensitivity: float = 0.5):
        """Rotate camera around look-at."""
        self.azimuth -= delta_x * sensitivity
        self.elevation += delta_y * sensitivity
        self.elevation = max(-89.0, min(89.0, self.elevation))

    def pan(self, delta_x: float, delta_y: float, sensitivity: float = 0.3):
        """Pan camera in screen plane."""
        view_dir = self.look_at - self.position
        v_norm = float(np.linalg.norm(view_dir))
        if v_norm < 1e-12:
            return
        forward = view_dir / v_norm

        up_ref = self.up_vector.astype(np.float64)
        u_norm = float(np.linalg.norm(up_ref))
        if u_norm < 1e-12:
            up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            up_ref = up_ref / u_norm

        right = np.cross(forward, up_ref)
        r_norm = float(np.linalg.norm(right))
        if r_norm < 1e-12:
            return
        right = right / r_norm

        up = np.cross(right, forward)
        u2_norm = float(np.linalg.norm(up))
        if u2_norm > 1e-12:
            up = up / u2_norm

        pan_speed = self.distance * sensitivity * 0.005
        self.pan_offset -= right * (delta_x * pan_speed)
        self.pan_offset += up * (delta_y * pan_speed)

    def zoom(self, delta: float, sensitivity: float = 1.1):
        """Zoom camera in/out."""
        if delta > 0:
            self.distance /= sensitivity
        else:
            self.distance *= sensitivity
        self.distance = max(self.min_distance, min(self.max_distance, self.distance))

    def reset(self):
        """Reset camera parameters."""
        self.distance = 50.0
        self.azimuth = 45.0
        self.elevation = 30.0
        self.center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def fit_to_bounds(self, bounds):
        """Fit camera to an axis-aligned bounds pair."""
        if bounds is None:
            return
        center = (bounds[0] + bounds[1]) / 2
        extents = bounds[1] - bounds[0]
        max_dim = np.max(extents)
        self.center = np.asarray(center, dtype=np.float64)
        self.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.distance = max(float(max_dim) * 2.0, self.min_distance)
        self.azimuth = 45.0
        self.elevation = 30.0

    def move_relative(self, dx: float, dy: float, dz: float, sensitivity: float = 1.0):
        """Move camera center in local navigation axes."""
        az_rad = np.radians(self.azimuth)
        right = np.array([-np.sin(az_rad), np.cos(az_rad), 0.0], dtype=np.float64)
        forward_h = np.array([-np.cos(az_rad), -np.sin(az_rad), 0.0], dtype=np.float64)
        up_v = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        move_speed = (self.distance * 0.03 + 2.0) * sensitivity
        self.center += right * (dx * move_speed)
        self.center += up_v * (dy * move_speed)
        self.center += forward_h * (dz * move_speed)

    def apply(self):
        """Apply camera transform to OpenGL."""
        try:
            pos = self.position
            target = self.look_at
            up = self.up_vector
            gluLookAt(
                pos[0], pos[1], pos[2],
                target[0], target[1], target[2],
                up[0], up[1], up[2],
            )
        except Exception:
            _log_ignored_exception()


class SceneObject:
    """???댁쓽 媛쒕퀎 硫붿돩 媛앹껜 愿由?"""
    def __init__(self, mesh, name="Object"):
        self.mesh = mesh
        self.name = name
        self.visible = True
        self.color = [0.72, 0.72, 0.78]
        
        # 媛쒕퀎 蹂???곹깭
        self.translation = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])
        self.scale = 1.0

        # ?뺤튂 怨좎젙 ?곹깭 (Bake ?댄썑 蹂듦???
        self.fixed_state_valid = False
        self.fixed_translation = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.fixed_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.fixed_scale = 1.0
        
        # ?쇳똿???먰샇??(硫붿돩? ?④퍡 ?대룞)
        self.fitted_arcs = []

        # Saved overlay polylines (e.g., section/cut results) attached to this object.
        # Each item is a dict: {name, kind, points([[x,y,z],...]), visible, color([r,g,b,a]), width}
        self.polyline_layers = []
        
        # ?뚮뜑留?由ъ냼??
        self.vbo_id = None
        self.vertex_count = 0
        self.selected_faces = set()
        self.outer_face_indices = set()
        self.inner_face_indices = set()
        self.migu_face_indices = set()
        # Optional assist output: faces still unresolved after seeded assist.
        self.surface_assist_unresolved_face_indices = set()
        self.surface_assist_meta: dict[str, Any] = {}
        self.surface_assist_runtime: dict[str, Any] = {}
        self._surface_assignment_version: int = 0
        self._surface_overlay_index_cache: dict[str, np.ndarray] = {}
        self._surface_overlay_index_cache_version: int = -1
        self._trimesh: Any | None = None  # Lazy-loaded trimesh object
        self._face_centroids: np.ndarray | None = None
        self._face_centroid_kdtree: Any | None = None
        self._face_centroid_faces_count: int = 0
        self._face_adjacency: list[list[int]] | None = None
        self._face_adjacency_faces_count: int = 0
        
    def to_trimesh(self):
        """trimesh 媛앹껜 諛섑솚 (罹먯떛)"""
        if self._trimesh is None and self.mesh:
            self._trimesh = self.mesh.to_trimesh()
        return self._trimesh

    def get_world_bounds(self):
        """?붾뱶 醫뚰몴怨꾩뿉?쒖쓽 寃쎄퀎 諛뺤뒪 諛섑솚"""
        if not self.mesh:
            return np.array([[0,0,0],[0,0,0]])
            
        # 濡쒖뺄 諛붿슫??
        lb = self.mesh.bounds
        
        # 8媛쒖쓽 瑗?쭞???앹꽦
        v = np.array([
            [lb[0,0], lb[0,1], lb[0,2]], [lb[1,0], lb[0,1], lb[0,2]],
            [lb[0,0], lb[1,1], lb[0,2]], [lb[1,0], lb[1,1], lb[0,2]],
            [lb[0,0], lb[0,1], lb[1,2]], [lb[1,0], lb[0,1], lb[1,2]],
            [lb[0,0], lb[1,1], lb[1,2]], [lb[1,0], lb[1,1], lb[1,2]]
        ])

        # ?붾뱶 蹂???곸슜 (R * (S * V) + T)
        rx, ry, rz = np.radians(self.rotation)
        cx, sx = float(np.cos(rx)), float(np.sin(rx))
        cy, sy = float(np.cos(ry)), float(np.sin(ry))
        cz, sz = float(np.cos(rz)), float(np.sin(rz))

        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
        rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
        rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        # OpenGL ?곸슜 ?쒖꽌(glRotatef X->Y->Z)? ?숈씪??intrinsic 'xyz' (Rx @ Ry @ Rz)
        rot_mat = rot_x @ rot_y @ rot_z

        world_v = (rot_mat @ (v * float(self.scale)).T).T + self.translation

        return np.array([world_v.min(axis=0), world_v.max(axis=0)])
        
    def cleanup(self):
        if self.vbo_id is not None:
            # OpenGL 而⑦뀓?ㅽ듃媛 ?쒖꽦?붾맂 ?곹깭?먯꽌 ?몄텧?섏뼱????
            try:
                vbo_id = int(self.vbo_id or 0)
            except Exception:
                vbo_id = 0
            if vbo_id > 0:
                try:
                    glDeleteBuffers(1, [vbo_id])
                except Exception:
                    _log_ignored_exception()


class Viewport3D(QOpenGLWidget):
    """OpenGL-based 3D viewport widget."""

    meshLoaded = pyqtSignal(object)
    meshTransformChanged = pyqtSignal()
    selectionChanged = pyqtSignal(int)
    floorPointPicked = pyqtSignal(np.ndarray)
    floorFacePicked = pyqtSignal(list)
    alignToBrushSelected = pyqtSignal()
    floorAlignmentConfirmed = pyqtSignal()
    profileUpdated = pyqtSignal(list, list)   # x_profile, y_profile
    lineProfileUpdated = pyqtSignal(list)     # line_profile
    roiSilhouetteExtracted = pyqtSignal(list) # 2D silhouette points
    cutLinesAutoEnded = pyqtSignal()
    cutLinesEnabledChanged = pyqtSignal(bool)
    cutLineActiveChanged = pyqtSignal(int)    # 0=Length, 1=Width
    roiSectionCommitRequested = pyqtSignal()
    faceSelectionChanged = pyqtSignal(int)
    surfaceAssignmentChanged = pyqtSignal(int, int, int)  # outer/inner/migu faces count
    measurePointPicked = pyqtSignal(np.ndarray)
    sliceScanRequested = pyqtSignal(float)
    sliceCaptureRequested = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Camera
        self.camera = TrackballCamera()
        
        # 留덉슦???곹깭
        self.last_mouse_pos = None
        self.mouse_button = None
        
        # ???곹깭 ?곹뼢 ?됱???(硫??硫붿돩)
        self.objects = []
        self.selected_index = -1
        self._mesh_center: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # ?뚮뜑留??ㅼ젙
        self.grid_size = 500.0  # cm (???ш쾶 ?뺤옣)
        self.grid_spacing = 1.0  # cm (1.0 = 1cm)
        self.bg_color = [0.96, 0.96, 0.94, 1.0] # #F5F5F0 (Cream/Beige)
        
        # 湲곗쫰紐??ㅼ젙
        self.show_gizmo = True
        self.active_gizmo_axis = None
        self.gizmo_size = 6.0
        self.gizmo_radius_factor = 0.72
        self.gizmo_drag_start = None
        
        # 怨〓쪧 痢≪젙 紐⑤뱶
        self.curvature_pick_mode = False
        self.picked_points = []
        self.fitted_arc = None

        # 移섏닔 痢≪젙 紐⑤뱶 (嫄곕━/吏곴꼍 ??
        self.measure_picked_points: list[np.ndarray] = []
        
        # Status text shown on viewport HUD
        self.status_info = ""
        self.surface_runtime_hud_enabled = True
        # Last assist/overlay performance snapshots for runtime HUD.
        self._surface_overlay_last_stats: dict[str, Any] = {}
        self.flat_shading = False # Flat shading 紐⑤뱶 (紐낆븫 ?놁씠 諛앷쾶 蹂닿린)
        # ?쇳빀 winding 硫붿돩?먯꽌 ?대?媛 鍮꾩퀜 蹂댁씠??臾몄젣瑜??쇳븯湲??꾪빐 湲곕낯媛믪? ?묐㈃ ?뚮뜑留?
        self.solid_shell_render = True
        # X-Ray render mode for selected object
        self.xray_mode = False
        self.xray_alpha = 0.25
        # ?뺣㈃/?꾨㈃ ?꾨━?뗭뿉??X-Z 吏곴탳 ?ъ쁺??媛뺤젣?????ъ슜
        self._front_back_ortho_enabled = False
        self._ortho_view_scale = ORTHO_VIEW_SCALE_DEFAULT
        
        # ?쇳궧 紐⑤뱶 ('none', 'curvature', 'floor_3point', 'floor_face', 'floor_brush')
        self.picking_mode = 'none'
        self.brush_selected_faces = set()  # Brush-selected face indices
        self._selection_brush_mode = "replace"  # replace|add|remove
        self._selection_brush_last_pick = 0.0
        # ??硫붿돩?먯꽌 ?대┃??議곌툑 鍮쀫굹媛硫?諛곌꼍 depth=1.0) 洹쇱쿂 ?쎌????먯깋?댁꽌 ?쇳궧??蹂댁젙?⑸땲??
        self._pick_search_radius_px = 8
        self._surface_paint_target = "outer"  # outer|inner|migu
        self._surface_brush_last_pick = 0.0
        # ?쒕㈃ 吏??李띻린/釉뚮윭?? ?꾧뎄 ?ш린: ?붾㈃(px) 湲곗? 湲곕낯媛??쇳궧 源딆씠?먯꽌 world濡??섏궛).
        # NOTE: ???ㅼ틪 硫붿돩?먯꽌??泥닿컧 ?ш린媛 ?덈Т ?묒? ?딅룄濡?湲곕낯媛믪쓣 ?щ젮?〓땲?? ([ / ]濡?議곗젅 媛??
        self._surface_brush_radius_px = 48.0
        self._surface_click_radius_px = 48.0
        self.surface_paint_points = []  # [(np.ndarray(3,), target), ...] in world coords
        self._surface_paint_points_max = 250
        self._surface_paint_left_press_pos = None
        self._surface_paint_left_dragged = False
        # Surface area(lasso) selection via mesh-snapped world points
        self.surface_lasso_points = []  # [np.ndarray(3,), ...] in world coords
        self.surface_lasso_face_indices = []  # [int, ...] per vertex (best-effort)
        self.surface_lasso_preview = None  # QPoint | None
        self._surface_area_left_press_pos = None
        self._surface_area_left_dragged = False
        self._surface_area_right_press_pos = None
        self._surface_area_right_dragged = False
        self._surface_lasso_thread = None
        self._surface_area_close_snap_px = 12
        self._surface_area_wand_angle_deg = 30.0
        self._surface_area_wand_knn_k = 12
        self._surface_area_wand_time_budget_s = 2.0
        self._surface_lasso_apply_target: str = "outer"
        self._surface_lasso_apply_modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier
        self._surface_lasso_apply_tool: str = "area"  # area|boundary

        # Surface boundary tool (magnetic edge snapping; vertices stored in surface_lasso_points)
        self.surface_magnetic_points: list[tuple[int, int]] = []  # GL window coords (x,y)
        self._surface_magnetic_drawing = False
        self._surface_magnetic_cursor_qt = None  # QPoint | None
        self._surface_magnetic_left_press_pos = None
        self._surface_magnetic_left_dragged = False
        self._surface_magnetic_right_press_pos = None
        self._surface_magnetic_right_dragged = False
        self._surface_magnetic_close_snap_px = 12
        self._surface_magnetic_snap_radius_px = 14
        self._surface_magnetic_min_step_px = 1.5
        self._surface_magnetic_last_add_t = 0.0
        self._surface_magnetic_space_nav = False
        self._surface_magnetic_dist = None  # np.ndarray(h,w) float32
        self._surface_magnetic_nn_y = None  # np.ndarray(h,w) int32
        self._surface_magnetic_nn_x = None  # np.ndarray(h,w) int32
        self._surface_magnetic_cache_viewport = None  # (vx,vy,w,h) in GL window coords
        self._surface_magnetic_cache_sig = None
        self._surface_magnetic_apply_target: str = "outer"
        self._surface_magnetic_apply_modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier
        self._surface_magnetic_thread = None
        # Show outer/inner/migu overlays even outside paint mode (helps visualize auto separation).
        self.show_surface_assignment_overlay = True
        # Magic-wand(stepwise) surface grow state for Shift/Ctrl clicks.
        # This lets the user expand a patch gradually (Photoshop-like).
        self._surface_grow_state: dict[str, Any] = {}
        self.floor_picks = []  # 諛붾떏硫?吏?뺤슜 ??由ъ뒪??        
        self._floor_3point_right_press_pos = None
        self._floor_3point_right_dragged = False
        # Undo/Redo ?쒖뒪??
        self.undo_stack = []
        self.max_undo = 50
        
        # ?⑤㈃ ?щ씪?댁떛
        self.slice_enabled = False
        self.slice_z = 0.0
        self.slice_contours = []  # ?꾩옱 ?щ씪?댁뒪 ?⑤㈃ ?대━?쇱씤
        
        # ??옄???⑤㈃ (Crosshair)
        self.crosshair_enabled = False
        self.crosshair_pos = np.array([0.0, 0.0]) # XY ?꾩튂
        self.x_profile = [] # X異??⑤㈃ ?곗씠??[(dist, z), ...]
        self.y_profile = [] # Y異??⑤㈃ ?곗씠??[(dist, z), ...]
        self._world_x_profile: np.ndarray | list[list[float]] = []
        self._world_y_profile: np.ndarray | list[list[float]] = []
        self._crosshair_last_update = 0.0
        self._crosshair_profile_thread = None
        self._crosshair_pending_pos = None
        self._crosshair_profile_timer = QTimer(self)
        self._crosshair_profile_timer.setSingleShot(True)
        self._crosshair_profile_timer.timeout.connect(self._request_crosshair_profile_compute)
        
        # 2D ROI (Region of Interest)
        self.roi_enabled = False
        self.roi_bounds = [-10.0, 10.0, -10.0, 10.0] # [min_x, max_x, min_y, max_y]
        self.active_roi_edge = None # ?꾩옱 ?쒕옒洹?以묒씤 紐⑥꽌由?('left', 'right', 'top', 'bottom')
        self.roi_rect_dragging = False  # 罹≪퀜 ?⑤벏???쒕옒洹몃줈 ROI 諛뺤뒪 吏??
        self.roi_rect_start = None      # np.array([x,y])
        self._roi_bounds_changed = False
        self._roi_move_dragging = False
        self._roi_move_last_xy = None  # np.ndarray(2,) | None
        self._roi_last_adjust_axis: str | None = None  # "x" | "y"
        self._roi_commit_axis_hint: str | None = None  # Enter 諛곗튂 ???곗꽑 異?
        self._roi_last_adjust_plane: str | None = None  # "x1" | "x2" | "y1" | "y2"
        self._roi_commit_plane_hint: str | None = None  # Enter 諛곗튂 ???곗꽑 ?됰㈃
        self._roi_handle_hit_px = 52
        self.roi_cut_edges: dict[str, list[np.ndarray]] = {"x1": [], "x2": [], "y1": [], "y2": []}  # ROI ?섎┝ 寃쎄퀎???붾뱶 醫뚰몴)
        self.roi_cap_verts: dict[str, np.ndarray | None] = {"x1": None, "x2": None, "y1": None, "y2": None}  # ROI 罹??쇨컖?? 踰꾪뀓??
        self.roi_section_world = {"x": [], "y": []}  # ROI濡??살? ?⑤㈃(諛붾떏 諛곗튂)
        # ROI ?⑤㈃ "梨꾩?"(罹? ?쒖떆. 湲곕낯? ?멸낸?좊쭔 蹂댁씠?꾨줉 OFF.
        self.roi_caps_enabled = False
        self._roi_edges_pending_bounds = None
        self._roi_edges_thread = None
        self._roi_edges_timer = QTimer(self)
        self._roi_edges_timer.setSingleShot(True)
        self._roi_edges_timer.timeout.connect(self._request_roi_edges_compute)
        # ROI ?쒕옒洹?以??⑤㈃ ?ш퀎?곗? 臾닿쾪湲??뚮Ц???꾨━酉??낅뜲?댄듃 媛꾧꺽???섎┰?덈떎.
        self._roi_live_update_delay_ms = 220

        # Cut guide lines (2 polylines on top view, SVG export??
        self.cut_lines_enabled = False
        self.cut_lines = [[], []]  # 媛??붿냼??world 醫뚰몴 ??由ъ뒪??[np.array([x,y,z]), ...]
        # Per-line axis lock: line0=Length(X), line1=Width(Y).
        self.cut_line_axis_lock: list[str | None] = ["x", "y"]
        self.cut_line_active = 0   # 0 or 1
        self.cut_line_drawing = False
        self.cut_line_preview = None  # np.array([x,y,z]) - 留덉?留??먯뿉???댁뼱吏???꾨━酉?        # 媛??대━?쇱씤??"?뺤젙/?꾨즺" ?곹깭. Enter/?고겢由?쑝濡?True媛 ?섎ŉ, Backspace/Delete濡??몄쭛 ??False濡??뚯븘媛?
        self._cut_line_final = [False, False]
        # ?⑤㈃???낅젰: '?대┃' vs '?쒕옒洹? 援щ텇(?대룞/?뚯쟾 以??먯씠 李랁엳??臾몄젣 諛⑹?)
        self._cut_line_left_press_pos = None
        self._cut_line_left_dragged = False
        self._cut_line_right_press_pos = None
        self._cut_line_right_dragged = False
        self.cut_section_profiles = [[], []]  # 媛??좎쓽 (s,z) ?꾨줈?뚯씪 [(dist, z), ...]
        self.cut_section_world = [[], []]     # 諛붾떏??諛곗튂???⑤㈃ ?대━?쇱씤(?붾뱶 醫뚰몴)
        self._cutline_tape_cache: dict[tuple[Any, ...], list[list[np.ndarray]]] = {}
        self._cut_section_pending_indices: set[int] = set()
        self._cut_section_thread = None
        self._cut_section_timer = QTimer(self)
        self._cut_section_timer.setSingleShot(True)
        self._cut_section_timer.timeout.connect(self._request_cut_section_compute)

        # Line section (top-view cut line)
        self.line_section_enabled = False
        self.line_section_dragging = False
        self.line_section_start = None  # np.ndarray([x, y, z])
        self.line_section_end = None    # np.ndarray([x, y, z])
        self.line_profile = []          # [(dist, z), ...]
        self.line_section_contours = [] # world-space contours
        self._line_section_last_update = 0.0
        self._line_profile_thread = None
        self._line_pending_segment = None
        self._line_profile_timer = QTimer(self)
        self._line_profile_timer.setSingleShot(True)
        self._line_profile_timer.timeout.connect(self._request_line_profile_compute)

        # Floor penetration highlight (z < 0)
        self.floor_penetration_highlight = False
        
        # ?쒕옒洹?議곗옉??理쒖쟻??蹂??
        self._drag_depth = 0.0
        self._cached_viewport = None
        self._cached_modelview = None
        self._cached_projection = None
        self._ctrl_drag_active = False
        self._hover_axis = None
        
        # ?ㅻ낫??議곗옉 ??대㉧ (WASD ?곗냽 ?대룞??
        self.keys_pressed = set()
        self.move_timer = QTimer(self)
        self.move_timer.timeout.connect(self.process_keyboard_navigation)
        self.move_timer.setInterval(16) # ~60fps (?꾩슂 ?쒕쭔 start/stop)
        
        # UI ?ㅼ젙
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # 蹂???대룞/?뚯쟾/?ㅼ??? ???⑤㈃/ROI ???뚯깮 ?곗씠?곕? ?붾컮?댁뒪 媛깆떊
        self.meshTransformChanged.connect(self._on_mesh_transform_changed)

        # ?뚮뜑留곸? ?낅젰/?곹깭 蹂寃??쒖뿉留?update()?섎룄濡??좎? (?곸떆 60FPS ?뚮뜑留곸? ??⑸웾 硫붿돩?먯꽌 踰꾨쾮???좊컻)
    
    @property
    def selected_obj(self) -> Optional[SceneObject]:
        if 0 <= self.selected_index < len(self.objects):
            return self.objects[self.selected_index]
        return None
    
    def initializeGL(self):
        """OpenGL 珥덇린??"""
        glClearColor(0.95, 0.95, 0.95, 1.0) # 諛앹? 諛곌꼍 (CloudCompare ?ㅽ???
        # 湲곕낯 ?ㅼ젙
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        
        # Grid performance: keep fixed-function line smoothing disabled.
        # On many drivers, wide+smoothed lines cause frame hitches in large scenes.
        glDisable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # 愿묒썝 紐⑤뜽 ?ㅼ젙 (?꾩뿭 ?섍꼍愿???땄)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.1, 0.1, 0.1, 1.0])
        
        # 愿묒썝 ?ㅼ젙 (湲곕낯媛?
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.0, 0.0, 0.0, 1.0]) # 媛쒕퀎 愿묒썝 ambient??0?쇰줈
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
        
        # ?몃? ?뺢퇋???쒖꽦??
        glEnable(GL_NORMALIZE)
        
        # ?대━怨?紐⑤뱶
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # ?덊떚?⑤━?댁떛 (?쇰? ?쒕씪?대쾭?먯꽌 遺덉븞?뺥븷 ???덉뼱 鍮꾪솢?깊솕)
        # glEnable(GL_LINE_SMOOTH)
        # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    
    def resizeGL(self, w: int, h: int):
        """酉고룷???ш린 蹂寃?"""
        glViewport(0, 0, w, h)

        self._apply_main_projection(w, h)

    @staticmethod
    def _build_look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        eye_v = np.asarray(eye, dtype=np.float64).reshape(3)
        tgt_v = np.asarray(target, dtype=np.float64).reshape(3)
        up_v = np.asarray(up, dtype=np.float64).reshape(3)

        f = tgt_v - eye_v
        fn = float(np.linalg.norm(f))
        if fn < 1e-12:
            f = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            f = f / fn

        un = float(np.linalg.norm(up_v))
        if un < 1e-12:
            up_v = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            up_v = up_v / un

        s = np.cross(f, up_v)
        sn = float(np.linalg.norm(s))
        if sn < 1e-12:
            up_v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            s = np.cross(f, up_v)
            sn = float(np.linalg.norm(s))
            if sn < 1e-12:
                s = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                s = s / sn
        else:
            s = s / sn

        u = np.cross(s, f)
        m = np.eye(4, dtype=np.float64)
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        t = np.eye(4, dtype=np.float64)
        t[:3, 3] = -eye_v
        return m @ t

    def _collect_projection_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        boxes: list[np.ndarray] = []

        for obj in self.objects:
            try:
                if not bool(getattr(obj, "visible", True)):
                    continue
                b = np.asarray(obj.get_world_bounds(), dtype=np.float64)
                if b.shape == (2, 3):
                    boxes.append(b)
            except Exception:
                continue

        if boxes:
            wb = np.vstack(boxes)
            return wb.min(axis=0), wb.max(axis=0)

        return np.array([-1.0, -1.0, -1.0], dtype=np.float64), np.array([1.0, 1.0, 1.0], dtype=np.float64)

    def _collect_projection_sphere(self) -> tuple[np.ndarray, float]:
        """酉??꾨젅?대컢 ?덉젙?붾? ?꾪븳 ?붾뱶 援?bound sphere) ?섏쭛."""
        sphere_center = None
        sphere_radius = None

        for obj in self.objects:
            try:
                if not bool(getattr(obj, "visible", True)):
                    continue
            except Exception:
                continue

            c = None
            r = None
            try:
                mesh = getattr(obj, "mesh", None)
                if mesh is not None and hasattr(mesh, "bounds"):
                    lb = np.asarray(mesh.bounds, dtype=np.float64)
                    if lb.shape == (2, 3) and np.isfinite(lb).all():
                        lc = (lb[0] + lb[1]) * 0.5
                        ext = lb[1] - lb[0]
                        base_r = float(0.5 * np.linalg.norm(ext))
                        sc = float(getattr(obj, "scale", 1.0) or 1.0)
                        if abs(sc) < 1e-12:
                            sc = 1.0
                        r = abs(sc) * base_r

                        tr = np.asarray(getattr(obj, "translation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
                        if tr.size < 3:
                            tr = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                        # Keep orthographic framing stable while the mesh rotates.
                        c = (lc * sc) + tr[:3]
            except Exception:
                c = None
                r = None

            if c is None or r is None or (not np.isfinite(c).all()) or (not np.isfinite(r)) or r <= 1e-9:
                try:
                    b = np.asarray(obj.get_world_bounds(), dtype=np.float64)
                    if b.shape == (2, 3) and np.isfinite(b).all():
                        c = (b[0] + b[1]) * 0.5
                        r = float(0.5 * np.linalg.norm(b[1] - b[0]))
                except Exception:
                    c = None
                    r = None

            if c is None or r is None or (not np.isfinite(c).all()) or (not np.isfinite(r)):
                continue

            c = np.asarray(c, dtype=np.float64).reshape(3)
            r = float(r)
            if sphere_center is None or sphere_radius is None:
                sphere_center = c
                sphere_radius = r
                continue

            dvec = c - sphere_center
            dist = float(np.linalg.norm(dvec))
            cur_r = float(sphere_radius)

            if dist + r <= cur_r:
                continue
            if dist + cur_r <= r:
                sphere_center = c
                sphere_radius = r
                continue

            new_r = 0.5 * (dist + cur_r + r)
            if dist > 1e-12:
                sphere_center = sphere_center + dvec * ((new_r - cur_r) / dist)
            sphere_radius = new_r

        if sphere_center is None or sphere_radius is None:
            c = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            return c, 10.0
        if (not np.isfinite(sphere_radius)) or sphere_radius <= 1e-6:
            sphere_radius = 10.0
        return np.asarray(sphere_center, dtype=np.float64).reshape(3), float(sphere_radius)

    def _sanitize_camera_state(self) -> None:
        """Defensive normalization for camera values restored from files or broken UI states."""
        cam = getattr(self, "camera", None)
        if cam is None:
            return

        try:
            min_d = float(getattr(cam, "min_distance", 0.01) or 0.01)
        except Exception:
            min_d = 0.01
        try:
            max_d = float(getattr(cam, "max_distance", 1_000_000.0) or 1_000_000.0)
        except Exception:
            max_d = 1_000_000.0
        if (not np.isfinite(min_d)) or min_d <= 0.0:
            min_d = 0.01
        if (not np.isfinite(max_d)) or max_d <= min_d:
            max_d = max(min_d * 10.0, 1_000_000.0)

        try:
            dist = float(getattr(cam, "distance", 50.0) or 50.0)
        except Exception:
            dist = 50.0
        if not np.isfinite(dist):
            dist = 50.0
        cam.distance = float(max(min_d, min(max_d, dist)))

        try:
            az = float(getattr(cam, "azimuth", 45.0) or 45.0)
        except Exception:
            az = 45.0
        if not np.isfinite(az):
            az = 45.0
        cam.azimuth = float(((az + 180.0) % 360.0) - 180.0)

        try:
            el = float(getattr(cam, "elevation", 30.0) or 30.0)
        except Exception:
            el = 30.0
        if not np.isfinite(el):
            el = 30.0
        cam.elevation = float(max(-90.0, min(90.0, el)))

        def _vec3(value: object, fallback: np.ndarray) -> np.ndarray:
            try:
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
                if arr.size >= 3 and np.isfinite(arr[:3]).all():
                    return arr[:3].copy()
            except Exception:
                pass
            return np.asarray(fallback, dtype=np.float64).reshape(3)

        cam.center = _vec3(getattr(cam, "center", [0.0, 0.0, 0.0]), np.zeros(3, dtype=np.float64))
        cam.pan_offset = _vec3(getattr(cam, "pan_offset", [0.0, 0.0, 0.0]), np.zeros(3, dtype=np.float64))

    def _apply_main_projection(self, w: int, h: int) -> None:
        self._sanitize_camera_state()
        ww = max(1, int(w))
        hh = max(1, int(h))
        aspect = float(ww) / float(hh)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Dynamic clip range avoids "mesh loaded but invisible" for very small or very large scales.
        clip_near = 0.1
        clip_far = 1000000.0
        try:
            center_w, radius = self._collect_projection_sphere()
            cam_pos = np.asarray(getattr(self.camera, "position", [0.0, 0.0, 50.0]), dtype=np.float64).reshape(-1)[:3]
            if cam_pos.size < 3 or (not np.isfinite(cam_pos).all()):
                raise ValueError("invalid camera position")
            c = np.asarray(center_w, dtype=np.float64).reshape(-1)[:3]
            r = float(max(1e-6, float(radius)))
            dist = float(np.linalg.norm(cam_pos - c))
            if (not np.isfinite(dist)) or dist <= 1e-9:
                dist = float(max(1e-3, getattr(self.camera, "distance", 50.0)))

            clip_near = float(max(1e-5, dist - (r * 4.0)))
            clip_far = float(max(clip_near + 1.0, dist + (r * 6.0)))
            # Keep far clip sufficiently large so floor/grid/axes do not get visibly cut at low elevation.
            cam_dist = float(max(1e-3, float(getattr(self.camera, "distance", dist) or dist)))
            elev_abs = abs(float(getattr(self.camera, "elevation", 30.0) or 30.0))
            if elev_abs < 5.0:
                horizon_factor = 42.0
            elif elev_abs < 10.0:
                horizon_factor = 30.0
            elif elev_abs < 20.0:
                horizon_factor = 18.0
            else:
                horizon_factor = 10.0
            clip_far = max(clip_far, dist + cam_dist * horizon_factor)
            clip_near = min(clip_near, max(1e-4, cam_dist * 1e-3))
            if (not np.isfinite(clip_near)) or clip_near <= 0.0:
                clip_near = 0.001
            if (not np.isfinite(clip_far)) or clip_far <= clip_near:
                clip_far = max(clip_near + 1.0, 1000.0)
            clip_near = float(min(clip_near, 1e7))
            clip_far = float(min(max(clip_far, clip_near + 1.0), 1e9))
        except Exception:
            clip_near = 0.1
            clip_far = 1000000.0

        use_front_back_ortho = bool(getattr(self, "_front_back_ortho_enabled", False))
        if use_front_back_ortho:
            try:
                az = float(getattr(self.camera, "azimuth", 0.0))
                el = float(getattr(self.camera, "elevation", 0.0))
                az = ((az + 180.0) % 360.0) - 180.0
                is_top_bottom = abs(abs(el) - 90.0) <= 1e-3
                is_side = abs(el) <= 1e-3 and any(abs(az - t) <= 1e-3 for t in (-180.0, -90.0, 0.0, 90.0, 180.0))
                use_front_back_ortho = bool(is_top_bottom or is_side)
            except Exception:
                use_front_back_ortho = False
        if not use_front_back_ortho:
            # 湲곕낯 ?먭렐 ?ъ쁺
            gluPerspective(45.0, aspect, float(clip_near), float(clip_far))
            glMatrixMode(GL_MODELVIEW)
            return

        # 6諛⑺뼢 異??뺣젹 ?꾨━?? ?뚯쟾 以묒뿉???ㅼ???援щ룄媛 ?붾뱾由ъ? ?딅룄濡??덉젙 吏곴탳 ?꾨젅?대컢
        try:
            _center_w, radius = self._collect_projection_sphere()
            try:
                ortho_scale = float(getattr(self, "_ortho_view_scale", ORTHO_VIEW_SCALE_DEFAULT) or ORTHO_VIEW_SCALE_DEFAULT)
                # 異??뺣젹 ?꾨━?뗭뿉?쒕뒗 ???꾩쟻 ?ㅽ봽?뗭쑝濡??명븳 援щ룄 ??댁쭚??李⑤떒.
            except Exception:
                ortho_scale = ORTHO_VIEW_SCALE_DEFAULT
            if not np.isfinite(ortho_scale):
                ortho_scale = ORTHO_VIEW_SCALE_DEFAULT
            ortho_scale = float(max(0.2, min(ortho_scale, 40.0)))

            base = max(1e-3, float(radius) * ortho_scale)
            if aspect >= 1.0:
                half_h = base
                half_w = base * aspect
            else:
                half_w = base
                half_h = base / max(1e-9, aspect)

            near = float(clip_near)
            far = float(clip_far)
            glOrtho(float(-half_w), float(half_w), float(-half_h), float(half_h), float(near), float(far))
        except Exception:
            gluPerspective(45.0, aspect, float(clip_near), float(clip_far))

        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """洹몃━湲?"""
        try:
            self._apply_main_projection(self.width(), self.height())
        except Exception:
            _log_ignored_exception()
        # ?댁쟾 ?꾨젅???덉쇅?먯꽌 ?꾩닔???곹깭瑜?留??꾨젅???뺢퇋?뷀빐 源딆씠 ?덉젙?깆쓣 ?뺣낫?⑸땲??
        glMatrixMode(GL_MODELVIEW)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
        glLoadIdentity()
        
        # 移대찓???곸슜
        self.camera.apply()

        # Reset clip planes every frame so stale ROI/floor clipping never leaks into global overlays.
        try:
            glDisable(GL_CLIP_PLANE0)
            glDisable(GL_CLIP_PLANE1)
            glDisable(GL_CLIP_PLANE2)
            glDisable(GL_CLIP_PLANE3)
            glDisable(GL_CLIP_PLANE4)
        except Exception:
            _log_ignored_exception()

        # 諛붾떏 愿??z<0) ?섏씠?쇱씠?몄슜 ?대━???됰㈃ 媛깆떊 (移대찓???대룞/?뚯쟾???곕씪 留??꾨젅???꾩슂)
        if self.floor_penetration_highlight:
            self._update_floor_penetration_clip_plane()
        if self.roi_enabled:
            self._update_roi_clip_planes()
        
        # 0. 愿묒썝 ?꾩튂 ?낅뜲?댄듃 (諛앷퀬 洹좎씪??議곕챸)
        if not self.flat_shading:
            glEnable(GL_LIGHTING)
            
            # ?섍꼍愿??믪엫 (?꾩껜?곸쑝濡?諛앷쾶)
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.4, 0.4, 0.4, 1.0])
            
            # GL_LIGHT0: ?뺣㈃ 二?議곕챸 (Headlight - 移대찓??諛⑺뼢)
            glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 1.0, 0.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.6, 0.6, 0.6, 1.0])
            glLightfv(GL_LIGHT0, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
            
            # GL_LIGHT1: 蹂댁“ 議곕챸 (?ㅼそ?먯꽌 - 洹몃┝???꾪솕)
            glEnable(GL_LIGHT1)
            glLightfv(GL_LIGHT1, GL_POSITION, [0.0, 0.0, -1.0, 0.0])
            glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.3, 0.3, 0.3, 1.0])
            glLightfv(GL_LIGHT1, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
        else:
            glDisable(GL_LIGHTING)
            # Flat shading ?쒖뿉??紐⑤뱺 硫댁씠 ?쇱젙 諛앷린濡?蹂댁씠寃?
            glColor3f(0.8, 0.8, 0.8)
        
        # 1. 寃⑹옄 諛?異?(Depth buffer?먮뒗 湲곕줉?섏? ?딆쓬: 硫붿돩 ?쇳궧/源딆씠 ?덉젙??
        glDepthMask(GL_FALSE)
        self.draw_ground_plane()  # 諛섑닾紐?諛붾떏
        self.draw_grid()
        glDepthMask(GL_TRUE)
        
        # 2. 紐⑤뱺 硫붿돩 媛앹껜 ?뚮뜑留?
        sel = int(self.selected_index) if self.selected_index is not None else -1
        xray_enabled = bool(getattr(self, "xray_mode", False))
        for i, obj in enumerate(self.objects):
            if not obj.visible:
                continue

            # X-Ray???좏깮??媛앹껜瑜?猷⑦봽 ?ㅼ뿉??蹂꾨룄 ?뚮뜑留?源딆씠踰꾪띁 ?곹뼢 理쒖냼??
            if xray_enabled and i == sel:
                continue

            # ROI ?대┰? ?좏깮??媛앹껜?먮쭔 ?곸슜
            if self.roi_enabled and i == sel:
                try:
                    glEnable(GL_CLIP_PLANE1)
                    glEnable(GL_CLIP_PLANE2)
                    glEnable(GL_CLIP_PLANE3)
                    glEnable(GL_CLIP_PLANE4)
                except Exception:
                    _log_ignored_exception("Failed to enable ROI clip planes", level=logging.WARNING)

                self.draw_scene_object(obj, is_selected=True)
                # ROI濡??섎┛ 硫댁쓣 梨꾩썙 ?⑤㈃ ?뺤씤???쎄쾶 蹂댁씠?꾨줉 罹??쒓퍚) ?뚮뜑留?
                if bool(getattr(self, "roi_caps_enabled", False)):
                    self.draw_roi_caps()

                try:
                    glDisable(GL_CLIP_PLANE1)
                    glDisable(GL_CLIP_PLANE2)
                    glDisable(GL_CLIP_PLANE3)
                    glDisable(GL_CLIP_PLANE4)
                except Exception:
                    _log_ignored_exception("Failed to disable ROI clip planes", level=logging.WARNING)
                continue

            self.draw_scene_object(obj, is_selected=(i == sel))

        # X-Ray (?좏깮??媛앹껜留? 留덉?留됱뿉 ?뚮뜑留?
        if xray_enabled and 0 <= sel < len(self.objects):
            try:
                obj = self.objects[sel]
            except Exception:
                obj = None

            if obj is not None and bool(getattr(obj, "visible", True)):
                a = float(getattr(self, "xray_alpha", 0.25))
                if not np.isfinite(a):
                    a = 0.25
                a = max(0.0, min(a, 1.0))

                if self.roi_enabled:
                    try:
                        glEnable(GL_CLIP_PLANE1)
                        glEnable(GL_CLIP_PLANE2)
                        glEnable(GL_CLIP_PLANE3)
                        glEnable(GL_CLIP_PLANE4)
                    except Exception:
                        _log_ignored_exception("Failed to enable ROI clip planes", level=logging.WARNING)

                self.draw_scene_object(obj, is_selected=True, alpha=a, depth_write=False)
                if self.roi_enabled and bool(getattr(self, "roi_caps_enabled", False)):
                    try:
                        glDepthMask(GL_FALSE)
                        self.draw_roi_caps()
                    finally:
                        glDepthMask(GL_TRUE)

                if self.roi_enabled:
                    try:
                        glDisable(GL_CLIP_PLANE1)
                        glDisable(GL_CLIP_PLANE2)
                        glDisable(GL_CLIP_PLANE3)
                        glDisable(GL_CLIP_PLANE4)
                    except Exception:
                        _log_ignored_exception("Failed to disable ROI clip planes", level=logging.WARNING)
            
        # 3. ?ㅻ쾭?덉씠 ?붿냼 (Depth write off: depth buffer??硫붿돩留??좎?)
        # Draw world axes after solids so they remain visible after mesh load.
        self.draw_axes()
        glDepthMask(GL_FALSE)

        # 3.1 怨〓쪧 ?쇳똿 ?붿냼
        self.draw_picked_points()
        self.draw_fitted_arc()

        # 3.2 ?쒕㈃ 吏??李띿? ?? ?쒖떆
        self.draw_surface_paint_points()
        # 3.3 ?쒕㈃ 吏??硫댁쟻/Area) ?ш?誘??ㅻ쾭?덉씠
        self.draw_surface_lasso_overlay()
        # 3.4 ?쒕㈃ 吏??寃쎄퀎/?먯꽍) ?ш?誘??ㅻ쾭?덉씠
        self.draw_surface_magnetic_lasso_overlay()

        # 3.5 諛붾떏 ?뺣젹 ???쒖떆
        self.draw_floor_picks()

        # 3.6 Mesh slicing plane/contours disabled (ROI + line-section workflow only).

        # 3.7 ??옄???⑤㈃
        if self.crosshair_enabled:
            self.draw_crosshair()

        # 3.7.25 ?⑤㈃??2媛? 媛?대뱶
        if self.cut_lines_enabled or any(len(line) > 0 for line in getattr(self, "cut_lines", [])) or self._has_visible_polyline_layers():
            self.draw_cut_lines()

        # 3.7.5 ?좏삎 ?⑤㈃ (Top-view cut line)
        if self.line_section_enabled:
            self.draw_line_section()
             
        # 3.8 2D ROI ?щ줈???곸뿭
        if self.roi_enabled:
            self.draw_roi_cut_edges()
            self.draw_roi_box()

        glDepthMask(GL_TRUE)

        # 3.9 ROI section floor-plots disabled (ROI-only workflow).
        
        # 4. ?뚯쟾 湲곗쫰紐?(?좏깮??媛앹껜?먮쭔, ?쇳궧 紐⑤뱶 ?꾨땺 ?뚮쭔)
        if self.selected_obj and self.picking_mode == 'none':
            if not self.roi_enabled:
                self.draw_rotation_gizmo(self.selected_obj)
            # 硫붿돩 移섏닔/以묒떖???ㅻ쾭?덉씠
            self.draw_mesh_dimensions(self.selected_obj)
            
        # 5. UI ?ㅻ쾭?덉씠 (HUD)
        self.draw_orientation_hud()
        self.draw_surface_runtime_hud()

    def _update_floor_penetration_clip_plane(self):
        """?붾뱶 諛붾떏(Z=0) 湲곗??쇰줈 '?꾨옒履?留??④린???대━???됰㈃ ?뺤쓽"""
        try:
            # Plane: z = 0, keep z <= 0  => -z >= 0
            glClipPlane(GL_CLIP_PLANE0, (0.0, 0.0, -1.0, 0.0))
        except Exception:
            # OpenGL 而⑦뀓?ㅽ듃/?꾨줈?뚯씪???곕씪 吏?먯씠 ???????덉쓬
            pass

    def _update_roi_clip_planes(self):
        """ROI bounds(x/y)濡??좏깮 硫붿돩瑜??щ줈?묓븯??4媛??대━???됰㈃ ?뺤쓽"""
        try:
            x1, x2, y1, y2 = self.roi_bounds
            # Keep: x >= x1
            glClipPlane(GL_CLIP_PLANE1, (1.0, 0.0, 0.0, -float(x1)))
            # Keep: x <= x2
            glClipPlane(GL_CLIP_PLANE2, (-1.0, 0.0, 0.0, float(x2)))
            # Keep: y >= y1
            glClipPlane(GL_CLIP_PLANE3, (0.0, 1.0, 0.0, -float(y1)))
            # Keep: y <= y2
            glClipPlane(GL_CLIP_PLANE4, (0.0, -1.0, 0.0, float(y2)))
        except Exception:
            _log_ignored_exception()
    
    def draw_ground_plane(self):
        """諛섑닾紐?諛붾떏硫?洹몃━湲?(Z=0, XY ?됰㈃) - Z-up 醫뚰몴怨?"""
        # ?섑룊 酉??뺣㈃/痢〓㈃ ???먯꽌??諛붾떏硫댁씠 ?좎쑝濡?蹂댁뿬 ?쒖빞瑜?諛⑺빐?섎?濡??④?
        if abs(self.camera.elevation) < 10:
            return
            
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # ?묐㈃ ?뚮뜑留?(?꾨옒?먯꽌 遊먮룄 蹂댁씠寃?
        glDisable(GL_CULL_FACE)
        
        # 諛붾떏硫??ш린 (移대찓??嫄곕━??鍮꾨?)
        elev = abs(float(getattr(self.camera, "elevation", 30.0) or 30.0))
        if elev < 5.0:
            horizon_factor = 22.0
        elif elev < 10.0:
            horizon_factor = 14.0
        elif elev < 20.0:
            horizon_factor = 9.0
        else:
            horizon_factor = 5.0
        size = max(float(self.camera.distance) * horizon_factor, 200.0)
        size = min(size, 2_000_000.0)
        
        # Keep floor tone neutral to reduce visual noise and preserve depth cues.
        glColor4f(0.82, 0.84, 0.86, 0.16)
        # ?뺤젏 ?쒖꽌: 諛섏떆怨?諛⑺뼢 = ?꾩そ???욌㈃ (Z-up)
        glBegin(GL_QUADS)
        glVertex3f(-size, -size, 0)
        glVertex3f(size, -size, 0)
        glVertex3f(size, size, 0)
        glVertex3f(-size, size, 0)
        glEnd()
        
        glDisable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)
    
    def draw_grid(self):
        """Draw an infinite-feeling world grid on Z=0."""
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glDisable(GL_LINE_SMOOTH)

        try:
            base_spacing = float(getattr(self, "grid_spacing", 1.0) or 1.0)
        except Exception:
            base_spacing = 1.0
        if (not np.isfinite(base_spacing)) or base_spacing <= 1e-9:
            base_spacing = 1.0

        try:
            cam_dist = float(getattr(self.camera, "distance", 500.0) or 500.0)
        except Exception:
            cam_dist = 500.0
        cam_dist = max(1.0, cam_dist)

        try:
            elev = abs(float(getattr(self.camera, "elevation", 30.0) or 30.0))
        except Exception:
            elev = 30.0
        if elev < 5.0:
            horizon_factor = 24.0
        elif elev < 10.0:
            horizon_factor = 16.0
        elif elev < 20.0:
            horizon_factor = 9.0
        else:
            horizon_factor = 5.0
        min_spacing_factor = 0.02 if elev < 10.0 else 0.015

        target_half_range = max(200.0, cam_dist * horizon_factor)
        target_half_range = min(target_half_range, 2_000_000.0)

        cam_center = np.asarray(getattr(self.camera, "look_at", [0.0, 0.0, 0.0]), dtype=np.float64)
        levels = [1, 10, 100, 1000, 10000]
        for level in levels:
            spacing = base_spacing * float(level)
            if (not np.isfinite(spacing)) or spacing <= 1e-9:
                continue
            if spacing < cam_dist * min_spacing_factor:
                continue
            if spacing > target_half_range * 1.2 and level > 1:
                continue

            half_steps = int(np.ceil(target_half_range / spacing))
            if level <= 10:
                max_half_steps = 1200
            elif level <= 100:
                max_half_steps = 900
            else:
                max_half_steps = 700
            half_steps = max(24, min(max_half_steps, half_steps))
            half_range = spacing * float(half_steps)

            if level == 1:
                alpha = 0.20
                line_width = 1.0
            elif level == 10:
                alpha = 0.30
                line_width = 1.0
            elif level == 100:
                alpha = 0.40
                line_width = 1.2
            elif level == 1000:
                alpha = 0.50
                line_width = 1.2
            else:
                alpha = 0.58
                line_width = 1.3

            glColor4f(0.5, 0.5, 0.5, alpha)
            glLineWidth(line_width)

            snap_x = round(float(cam_center[0]) / spacing) * spacing
            snap_y = round(float(cam_center[1]) / spacing) * spacing

            glBegin(GL_LINES)
            for i in range(-half_steps, half_steps + 1):
                offset = i * spacing
                x_val = snap_x + offset
                glVertex3f(float(x_val), float(snap_y - half_range), 0.0)
                glVertex3f(float(x_val), float(snap_y + half_range), 0.0)

                y_val = snap_y + offset
                glVertex3f(float(snap_x - half_range), float(y_val), 0.0)
                glVertex3f(float(snap_x + half_range), float(y_val), 0.0)
            glEnd()

        glLineWidth(1.0)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def draw_slice_plane(self):
        """?꾩옱 ?щ씪?댁뒪 ?믪씠??諛섑닾紐??됰㈃ 洹몃━湲?"""
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # ?됰㈃ ?ш린 (洹몃━???ш린? 留욎땄)
        s = self.grid_size / 2
        z = self.slice_z
        
        # 諛섑닾紐?鍮④컙???됰㈃
        glColor4f(1.0, 0.0, 0.0, 0.15)
        glBegin(GL_QUADS)
        glVertex3f(-s, -s, z)
        glVertex3f(s, -s, z)
        glVertex3f(s, s, z)
        glVertex3f(-s, s, z)
        glEnd()
        
        # 寃쎄퀎??
        glLineWidth(2.0)
        glColor4f(1.0, 0.0, 0.0, 0.5)
        glBegin(GL_LINE_LOOP)
        glVertex3f(-s, -s, z)
        glVertex3f(s, -s, z)
        glVertex3f(s, s, z)
        glVertex3f(-s, s, z)
        glEnd()
        glLineWidth(1.0)
        
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def draw_slice_contours(self):
        """異붿텧???⑤㈃??洹몃━湲?"""
        if not self.slice_contours:
            return
            
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glColor3f(1.0, 0.0, 1.0)  # 留덉젨? ?됱긽 (?덉뿉 ?꾧쾶)
        
        for contour in self.slice_contours:
            if len(contour) < 2:
                continue
            glBegin(GL_LINE_STRIP)
            for pt in contour:
                glVertex3fv(pt)
            glEnd()
            
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def update_slice(self):
        """?꾩옱 Z ?믪씠?먯꽌 ?⑤㈃ ?ъ텛異?"""
        obj = self.selected_obj
        if obj is None or obj.mesh is None:
            self.slice_contours = []
            return

        tm = obj.to_trimesh()
        if tm is None:
            self.slice_contours = []
            return

        slicer = MeshSlicer(cast(Any, tm))
        
        # 媛앹껜??濡쒖뺄 Z 醫뚰몴濡?蹂???꾩슂 (?꾩옱???붾뱶 Z 湲곗? ?щ씪?댁뒪 援ы쁽)
        # TODO: 媛앹껜 蹂???뚯쟾, ?대룞) 諛섏쁺 泥섎━
        # ?곗꽑 媛???⑥닚?섍쾶 ?붾뱶 Z 湲곗? (?됰㈃ origin??媛앹껜 濡쒖뺄 醫뚰몴濡?????섑븯???щ씪?댁뒪)
        
        # (?붾뱶 Z) -> (濡쒖뺄 醫뚰몴)
        # 濡쒖쭅: P_world = R * (S * P_local) + T
        # P_local = (1/S) * R^T * (P_world - T)

        from scipy.spatial.transform import Rotation as R
        inv_rot = R.from_euler('XYZ', obj.rotation, degrees=True).inv().as_matrix()
        inv_scale = 1.0 / obj.scale if obj.scale != 0 else 1.0
        
        # ?붾뱶 ?됰㈃ [0, 0, 1] dot (P - [0, 0, Z_slice]) = 0
        # 濡쒗봽 醫뚰몴?먯꽌???됰㈃ origin怨?normal 怨꾩궛
        world_origin = np.array([0.0, 0.0, float(self.slice_z)], dtype=np.float64)
        local_origin = inv_scale * inv_rot @ (world_origin - obj.translation)

        world_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        local_normal = inv_rot @ world_normal # ?뚯쟾留??곸슜 (踰뺤꽑踰≫꽣?대?濡?

        self.slice_contours = slicer.slice_with_plane(local_origin.tolist(), local_normal.tolist())
        
        # 異붿텧??濡쒖뺄 醫뚰몴 ?⑤㈃???붾뱶 醫뚰몴濡?蹂?섑븯?????(?뚮뜑留곸슜)
        rot_mat = R.from_euler('XYZ', obj.rotation, degrees=True).as_matrix()
        scale = obj.scale
        trans = obj.translation
        
        world_contours = []
        for cnt in self.slice_contours:
            # P_world = R * (S * P_local) + T
            w_cnt = (rot_mat @ (cnt * scale).T).T + trans
            world_contours.append(w_cnt)
            
        self.slice_contours = world_contours
        self.update()

    def draw_crosshair(self):
        """??옄??諛?硫붿돩 ?ъ쁺 ?⑤㈃ ?쒓컖??"""
        glDisable(GL_LIGHTING)
        
        cx, cy = self.crosshair_pos
        s = self.grid_size / 2
        
        # 1. 諛붾떏 ??옄??(?고븳 ?뚯깋)
        glLineWidth(1.0)
        glColor4f(0.5, 0.5, 0.5, 0.5)
        glBegin(GL_LINES)
        glVertex3f(-s, cy, 0)
        glVertex3f(s, cy, 0)
        glVertex3f(cx, -s, 0)
        glVertex3f(cx, s, 0)
        glEnd()
        
        # 2. 硫붿돩 ?ъ쁺 ?⑤㈃ (媛뺥븳 ?몃???
        glLineWidth(3.0)
        glColor3f(1.0, 1.0, 0.0)
        
        # X異??꾨줈?뚯씪 (Y 怨좎젙)
        if self.x_profile:
            pass
            
        # 3. ?ㅼ젣 異붿텧???ъ씤?몃뱾 ?뚮뜑留?(?붾뱶 醫뚰몴怨?
        # X ?꾨줈?뚯씪: X異?諛⑺뼢?쇰줈 媛濡쒖?瑜대뒗 ??(Y = cy)
        world_x_profile = getattr(self, '_world_x_profile', None)
        if world_x_profile is not None and len(world_x_profile) > 0:
            glColor3f(1.0, 1.0, 0.0) # Yellow
            glBegin(GL_LINE_STRIP)
            for pt in world_x_profile:
                glVertex3fv(pt)
            glEnd()
            
        # Y ?꾨줈?뚯씪: Y異?諛⑺뼢?쇰줈 媛濡쒖?瑜대뒗 ??(X = cx)
        world_y_profile = getattr(self, '_world_y_profile', None)
        if world_y_profile is not None and len(world_y_profile) > 0:
            glColor3f(0.0, 1.0, 1.0) # Cyan
            glBegin(GL_LINE_STRIP)
            for pt in world_y_profile:
                glVertex3fv(pt)
            glEnd()
            
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def clear_line_section(self):
        """?좏삎 ?⑤㈃(?쇱씤) ?곗씠??珥덇린??"""
        self.line_section_start = None
        self.line_section_end = None
        self.line_profile = []
        self.line_section_contours = []
        try:
            self.lineProfileUpdated.emit([])
        except Exception:
            _log_ignored_exception()
        self.update()

    def _unique_polyline_layer_name(self, obj: SceneObject, base: str) -> str:
        base = str(base).strip() or "Layer"
        try:
            existing = {str(layer.get("name", "")).strip() for layer in getattr(obj, "polyline_layers", []) or []}
        except Exception:
            existing = set()
        if base not in existing:
            return base
        n = 2
        while f"{base} {n}" in existing:
            n += 1
        return f"{base} {n}"

    @staticmethod
    def _to_points_list(points) -> list[list[float]]:
        out_pts: list[list[float]] = []
        for p in points or []:
            try:
                arr = np.asarray(p, dtype=np.float64).reshape(-1)
            except Exception:
                continue
            if arr.size >= 3:
                out_pts.append([float(arr[0]), float(arr[1]), float(arr[2])])
            elif arr.size == 2:
                out_pts.append([float(arr[0]), float(arr[1]), 0.0])
        return out_pts

    @staticmethod
    def _polyline_bbox_xy(points: list[list[float]]) -> tuple[float, float, float, float] | None:
        if not points:
            return None
        try:
            arr = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        except Exception:
            return None
        if arr.shape[0] < 2:
            return None
        finite = np.all(np.isfinite(arr[:, :2]), axis=1)
        arr = arr[finite]
        if arr.shape[0] < 2:
            return None
        min_x = float(np.min(arr[:, 0]))
        max_x = float(np.max(arr[:, 0]))
        min_y = float(np.min(arr[:, 1]))
        max_y = float(np.max(arr[:, 1]))
        return (min_x, max_x, min_y, max_y)

    @staticmethod
    def _boxes_overlap_2d(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
        pad: float = 0.0,
    ) -> bool:
        pad_v = float(max(0.0, pad))
        return not (
            (a[1] + pad_v) < b[0]
            or (b[1] + pad_v) < a[0]
            or (a[3] + pad_v) < b[2]
            or (b[3] + pad_v) < a[2]
        )

    def _suggest_section_profile_offset(self, obj: SceneObject, points: list[list[float]]) -> list[float]:
        """湲곕낯 諛곗튂 ?꾩튂???⑤㈃ ?덉씠?닿? ?대? ?덉쑝硫?寃뱀튂吏 ?딅룄濡??ㅽ봽?뗭쓣 ?쒖븞."""
        bbox_new = self._polyline_bbox_xy(points)
        if bbox_new is None:
            return [0.0, 0.0]

        existing_boxes: list[tuple[float, float, float, float]] = []
        try:
            layers = getattr(obj, "polyline_layers", None) or []
        except Exception:
            layers = []

        for layer in layers:
            try:
                if str(layer.get("kind", "")).strip() != "section_profile":
                    continue
                pts = self._to_points_list(layer.get("points", []) or [])
                bbox = self._polyline_bbox_xy(pts)
                if bbox is None:
                    continue
                off = layer.get("offset", [0.0, 0.0]) or [0.0, 0.0]
                off_x = float(off[0]) if len(off) >= 1 else 0.0
                off_y = float(off[1]) if len(off) >= 2 else 0.0
                existing_boxes.append(
                    (bbox[0] + off_x, bbox[1] + off_x, bbox[2] + off_y, bbox[3] + off_y)
                )
            except Exception:
                continue

        if not existing_boxes:
            return [0.0, 0.0]

        span_x = max(1e-6, float(bbox_new[1] - bbox_new[0]))
        span_y = max(1e-6, float(bbox_new[3] - bbox_new[2]))
        span = max(span_x, span_y)
        gap = max(2.0, 0.16 * span)
        overlap_pad = max(0.2, 0.04 * span)

        def collides(dx: float, dy: float) -> bool:
            bb = (bbox_new[0] + dx, bbox_new[1] + dx, bbox_new[2] + dy, bbox_new[3] + dy)
            for ex in existing_boxes:
                if self._boxes_overlap_2d(bb, ex, pad=overlap_pad):
                    return True
            return False

        if not collides(0.0, 0.0):
            return [0.0, 0.0]

        for step in range(1, 32):
            d = float(step) * float(gap)
            for cand_x, cand_y in (
                (d, 0.0),
                (0.0, d),
                (d, d),
                (-d, 0.0),
                (0.0, -d),
            ):
                if not collides(cand_x, cand_y):
                    return [float(cand_x), float(cand_y)]

        fallback = float(max(1, len(existing_boxes))) * float(gap)
        return [fallback, fallback]

    def _append_polyline_layer(
        self,
        obj: SceneObject,
        *,
        name: str,
        kind: str,
        points,
        color: list[float],
        width: float = 2.0,
        auto_separate_section: bool = False,
    ) -> bool:
        pts = self._to_points_list(points)
        if len(pts) < 2:
            return False
        unique_name = self._unique_polyline_layer_name(obj, name)
        offset = [0.0, 0.0]
        if bool(auto_separate_section) and str(kind).strip() == "section_profile":
            try:
                offset = self._suggest_section_profile_offset(obj, pts)
            except Exception:
                offset = [0.0, 0.0]
        try:
            col = [float(x) for x in (color or [0.2, 0.2, 0.2, 0.9])][:4]
        except Exception:
            col = [0.2, 0.2, 0.2, 0.9]
        if len(col) < 4:
            col = (col + [0.9, 0.9, 0.9, 0.9])[:4]
        obj.polyline_layers.append(
            {
                "name": unique_name,
                "kind": str(kind),
                "points": pts,
                "visible": True,
                "offset": [float(offset[0]), float(offset[1])],
                "color": col,
                "width": float(width),
            }
        )
        return True

    def save_current_slice_to_layer(self) -> int:
        """?꾩옱 ?щ씪?댁뒪留??덉씠?대줈 ?ㅻ깄?????"""
        obj = self.selected_obj
        if obj is None:
            return 0
        contours = getattr(self, "slice_contours", None) or []
        if not bool(getattr(self, "slice_enabled", False)) or not contours:
            return 0

        if not hasattr(obj, "polyline_layers") or obj.polyline_layers is None:
            obj.polyline_layers = []

        added = 0
        z_val = float(getattr(self, "slice_z", 0.0) or 0.0)
        for i, cnt in enumerate(contours):
            suffix = f" #{i+1}" if len(contours) > 1 else ""
            if self._append_polyline_layer(
                obj,
                name=f"Slice-Z {z_val:.2f}cm{suffix}",
                kind="section_profile",
                points=cnt,
                color=[0.75, 0.15, 0.75, 0.9],
                width=2.0,
                auto_separate_section=False,
            ):
                added += 1

        if added:
            self.update()
        return int(added)

    def save_current_sections_to_layers(
        self,
        *,
        include_cut_lines: bool = True,
        include_cut_profiles: bool = True,
        include_roi_profiles: bool = True,
        include_slices: bool = True,
        separate_section_profiles: bool = False,
        roi_axes: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> int:
        """?꾩옱 ?⑤㈃/媛?대뱶 寃곌낵瑜??ㅻ깄???덉씠?대줈 ???"""
        obj = self.selected_obj
        if obj is None:
            return 0

        if not hasattr(obj, "polyline_layers") or obj.polyline_layers is None:
            obj.polyline_layers = []

        added = 0

        # 1) Cut guide lines (top view)
        if bool(include_cut_lines):
            names = ["CutLine-Length", "CutLine-Width"]
            colors = [
                (1.0, 0.25, 0.25, 0.95),
                (0.15, 0.55, 1.0, 0.95),
            ]
            for i, line in enumerate(getattr(self, "cut_lines", [[], []]) or []):
                if self._append_polyline_layer(
                    obj,
                    name=(names[i] if i < len(names) else f"?⑤㈃??{i+1}"),
                    kind="cut_line",
                    points=line,
                    color=list(colors[i % len(colors)]),
                    width=2.0,
                    auto_separate_section=False,
                ):
                    added += 1

        # 2) Section profiles laid on the floor
        if bool(include_cut_profiles):
            sec_names = ["Section-Length", "Section-Width"]
            for i, line in enumerate(getattr(self, "cut_section_world", [[], []]) or []):
                if self._append_polyline_layer(
                    obj,
                    name=(sec_names[i] if i < len(sec_names) else f"?⑤㈃-{i+1}"),
                    kind="section_profile",
                    points=line,
                    color=[0.1, 0.1, 0.1, 0.9],
                    width=2.0,
                    auto_separate_section=bool(separate_section_profiles),
                ):
                    added += 1

        # 3) ROI section profiles (if any)
        if bool(include_roi_profiles):
            try:
                roi_sec = getattr(self, "roi_section_world", {}) or {}
                roi_keys: tuple[str, ...] = ("x", "y")
                if roi_axes is not None:
                    try:
                        norm = []
                        for vv in roi_axes:
                            kk = str(vv).strip().lower()
                            if kk in ("x", "y"):
                                norm.append(kk)
                        if norm:
                            roi_keys = tuple(dict.fromkeys(norm))
                    except Exception:
                        roi_keys = ("x", "y")
                for key in roi_keys:
                    line = roi_sec.get(key, None)
                    axis = "X" if key == "x" else "Y"
                    if self._append_polyline_layer(
                        obj,
                        name=f"ROI-?⑤㈃-{axis}",
                        kind="section_profile",
                        points=line,
                        color=[0.1, 0.35, 0.1, 0.9],
                        width=2.0,
                        auto_separate_section=bool(separate_section_profiles),
                    ):
                        added += 1
            except Exception:
                _log_ignored_exception()

        # 4) Slice contours (if enabled)
        if bool(include_slices):
            try:
                contours = getattr(self, "slice_contours", None) or []
                if bool(getattr(self, "slice_enabled", False)) and contours:
                    z_val = float(getattr(self, "slice_z", 0.0) or 0.0)
                    for i, cnt in enumerate(contours):
                        suffix = f" #{i+1}" if len(contours) > 1 else ""
                        if self._append_polyline_layer(
                            obj,
                            name=f"Slice-Z {z_val:.2f}cm{suffix}",
                            kind="section_profile",
                            points=cnt,
                            color=[0.75, 0.15, 0.75, 0.9],
                            width=2.0,
                            auto_separate_section=False,
                        ):
                            added += 1
            except Exception:
                _log_ignored_exception()

        if added:
            self.update()
        return int(added)

    def save_roi_sections_to_layers(self) -> int:
        """?꾩옱 ROI ?⑤㈃ 誘몃━蹂닿린瑜??덉씠?대줈 ?뺤젙 諛곗튂."""
        plane_hint = str(getattr(self, "_roi_commit_plane_hint", "") or "").strip().lower()
        axis_hint = str(getattr(self, "_roi_commit_axis_hint", "") or "").strip().lower()
        try:
            # Enter 吏곸쟾??ROI媛 諛⑷툑 蹂寃쎈맂 寃쎌슦, ?ㅻ젅??寃곌낵瑜?湲곕떎由ъ? ?딄퀬 ?꾩옱 bounds 湲곗??쇰줈 利됱떆 怨꾩궛
            # ?댁꽌 "?ъ슜?먭? 留덉?留됱쑝濡?以꾩씤 ?⑤㈃"??洹몃?濡?諛곗튂?⑸땲??
            need_fresh_edges = bool(getattr(self, "_roi_bounds_changed", False)) or (
                getattr(self, "_roi_edges_pending_bounds", None) is not None
            )
            if not need_fresh_edges:
                try:
                    cur_edges = getattr(self, "roi_cut_edges", None) or {}
                    if isinstance(cur_edges, dict):
                        need_fresh_edges = not any(
                            isinstance(v, list) and len(v) > 0 for v in cur_edges.values()
                        )
                    else:
                        need_fresh_edges = True
                except Exception:
                    need_fresh_edges = True

            if need_fresh_edges:
                fresh_edges = self._compute_roi_cut_edges_sync()
                if any(len(fresh_edges.get(k, [])) > 0 for k in ("x1", "x2", "y1", "y2")):
                    self.roi_cut_edges = fresh_edges
                    self._roi_bounds_changed = False
                    self._roi_edges_pending_bounds = None
                    try:
                        self._rebuild_roi_caps()
                    except Exception:
                        _log_ignored_exception()

            built = self._build_roi_sections_for_commit()
            if built.get("x") or built.get("y"):
                self.roi_section_world = {
                    "x": built.get("x", []) or [],
                    "y": built.get("y", []) or [],
                }

            preferred_axes: tuple[str, ...] | None = None
            if plane_hint in ("x1", "x2"):
                preferred_axes = ("x",)
            elif plane_hint in ("y1", "y2"):
                preferred_axes = ("y",)
            elif axis_hint in ("x", "y"):
                preferred_axes = (axis_hint,)

            if preferred_axes is not None:
                added = int(
                    self.save_current_sections_to_layers(
                        include_cut_lines=False,
                        include_cut_profiles=False,
                        include_roi_profiles=True,
                        include_slices=False,
                        separate_section_profiles=True,
                        roi_axes=preferred_axes,
                    )
                )
                if added > 0:
                    return added

            if axis_hint in ("x", "y"):
                added = int(
                    self.save_current_sections_to_layers(
                        include_cut_lines=False,
                        include_cut_profiles=False,
                        include_roi_profiles=True,
                        include_slices=False,
                        separate_section_profiles=True,
                        roi_axes=(axis_hint,),
                    )
                )
                if added > 0:
                    return added

            return int(
                self.save_current_sections_to_layers(
                    include_cut_lines=False,
                    include_cut_profiles=False,
                    include_roi_profiles=True,
                    include_slices=False,
                    separate_section_profiles=True,
                    roi_axes=None,
                )
            )
        finally:
            self._roi_commit_axis_hint = None
            self._roi_commit_plane_hint = None

    def set_polyline_layer_visible(self, object_index: int, layer_index: int, visible: bool):
        try:
            oi = int(object_index)
            li = int(layer_index)
        except Exception:
            return
        if not (0 <= oi < len(self.objects)):
            return
        obj = self.objects[oi]
        layers = getattr(obj, "polyline_layers", None)
        if not layers or not (0 <= li < len(layers)):
            return
        try:
            layers[li]["visible"] = bool(visible)
        except Exception:
            return
        self.update()

    def delete_polyline_layer(self, object_index: int, layer_index: int):
        try:
            oi = int(object_index)
            li = int(layer_index)
        except Exception:
            return
        if not (0 <= oi < len(self.objects)):
            return
        obj = self.objects[oi]
        layers = getattr(obj, "polyline_layers", None)
        if not layers or not (0 <= li < len(layers)):
            return
        try:
            layers.pop(li)
        except Exception:
            return
        self.update()

    def move_polyline_layer(self, object_index: int, layer_index: int, dx: float, dy: float):
        try:
            oi = int(object_index)
            li = int(layer_index)
        except Exception:
            return
        if not (0 <= oi < len(self.objects)):
            return
        obj = self.objects[oi]
        layers = getattr(obj, "polyline_layers", None)
        if not layers or not (0 <= li < len(layers)):
            return

        layer = layers[li]
        try:
            off = layer.get("offset", None)
        except Exception:
            off = None
        if not (isinstance(off, (list, tuple)) and len(off) >= 2):
            off = [0.0, 0.0]

        try:
            off_x = float(off[0]) + float(dx)
            off_y = float(off[1]) + float(dy)
        except Exception:
            return

        layer["offset"] = [off_x, off_y]
        self.update()

    def reset_polyline_layer_offset(self, object_index: int, layer_index: int):
        try:
            oi = int(object_index)
            li = int(layer_index)
        except Exception:
            return
        if not (0 <= oi < len(self.objects)):
            return
        obj = self.objects[oi]
        layers = getattr(obj, "polyline_layers", None)
        if not layers or not (0 <= li < len(layers)):
            return
        try:
            layers[li]["offset"] = [0.0, 0.0]
        except Exception:
            return
        self.update()

    def _has_visible_polyline_layers(self) -> bool:
        obj = self.selected_obj
        if obj is None:
            return False
        layers = getattr(obj, "polyline_layers", None) or []
        for layer in layers:
            try:
                if not bool(layer.get("visible", True)):
                    continue
                pts = layer.get("points", None)
                if pts is not None and len(pts) >= 2:
                    return True
            except Exception:
                continue
        return False

    def _clear_cutline_tape_cache(self) -> None:
        try:
            cache = getattr(self, "_cutline_tape_cache", None)
            if isinstance(cache, dict):
                cache.clear()
            else:
                self._cutline_tape_cache = {}
        except Exception:
            self._cutline_tape_cache = {}

    def set_cut_lines_enabled(self, enabled: bool):
        self.cut_lines_enabled = bool(enabled)
        try:
            locks = getattr(self, "cut_line_axis_lock", None)
            if not isinstance(locks, list) or len(locks) < 2:
                self.cut_line_axis_lock = ["x", "y"]
            else:
                lk0 = str(locks[0]).strip().lower() if locks[0] is not None else ""
                lk1 = str(locks[1]).strip().lower() if locks[1] is not None else ""
                self.cut_line_axis_lock[0] = "x" if lk0 not in ("x", "y") else lk0
                self.cut_line_axis_lock[1] = "y" if lk1 not in ("x", "y") else lk1
        except Exception:
            self.cut_line_axis_lock = ["x", "y"]
        if enabled:
            self.picking_mode = 'cut_lines'
            # ?꾨━酉곕? ?꾪빐 留덉슦???몃옒???쒖꽦??
            self.setMouseTracking(True)
            try:
                self.setFocus()
            except Exception:
                _log_ignored_exception()
            try:
                idx = int(getattr(self, "cut_line_active", 0))
                idx = idx if idx in (0, 1) else 0
                self.cut_line_active = idx
                line = self.cut_lines[idx]
                final = getattr(self, "_cut_line_final", [False, False])
                self.cut_line_drawing = bool(line) and not bool(final[idx])
                try:
                    self.cutLineActiveChanged.emit(int(idx))
                except Exception:
                    _log_ignored_exception()
            except Exception:
                self.cut_line_drawing = False
        else:
            if self.picking_mode == 'cut_lines':
                self.picking_mode = 'none'
            self.cut_line_drawing = False
            self.cut_line_preview = None
            self._cut_line_left_press_pos = None
            self._cut_line_left_dragged = False
            self._cut_line_right_press_pos = None
            self._cut_line_right_dragged = False
            self.setMouseTracking(False)
        try:
            self.cutLinesEnabledChanged.emit(bool(self.cut_lines_enabled))
        except Exception:
            _log_ignored_exception()
        self.update()

    def clear_cut_line(self, index: int):
        try:
            idx = int(index)
            if idx not in (0, 1):
                return
            self.cut_lines[idx] = []
            try:
                self._cut_line_final[idx] = False
            except Exception:
                _log_ignored_exception()
            self.cut_section_profiles[idx] = []
            self.cut_section_world[idx] = []
            self._clear_cutline_tape_cache()
            try:
                self._cut_section_pending_indices.discard(idx)
            except Exception:
                _log_ignored_exception()
            if self.cut_line_active == idx:
                self.cut_line_drawing = False
                self.cut_line_preview = None
        except Exception:
            return
        self.update()

    def clear_cut_lines(self):
        self.cut_lines = [[], []]
        self.cut_line_drawing = False
        self.cut_line_preview = None
        try:
            self._cut_line_final = [False, False]
        except Exception:
            _log_ignored_exception()
        self.cut_section_profiles = [[], []]
        self.cut_section_world = [[], []]
        self._clear_cutline_tape_cache()
        try:
            self._cut_section_pending_indices.clear()
        except Exception:
            _log_ignored_exception()
        self.update()

    def get_cut_lines_world(self):
        """?대낫?닿린?? ?⑤㈃??2媛? ?붾뱶 醫뚰몴 諛섑솚 (?쒖닔 python list)"""
        out = []
        for line in getattr(self, "cut_lines", [[], []]):
            pts = []
            for p in line:
                try:
                    arr = np.asarray(p, dtype=np.float64).reshape(-1)
                    if arr.size >= 3:
                        pts.append([float(arr[0]), float(arr[1]), float(arr[2])])
                    elif arr.size == 2:
                        pts.append([float(arr[0]), float(arr[1]), 0.0])
                except Exception:
                    continue
            out.append(pts)

        # Saved polyline layers (visible)
        obj = self.selected_obj
        if obj is not None:
            for layer in getattr(obj, "polyline_layers", []) or []:
                try:
                    if not bool(layer.get("visible", True)):
                        continue
                    if str(layer.get("kind", "")) != "cut_line":
                        continue
                    off = layer.get("offset", None)
                    if not (isinstance(off, (list, tuple)) and len(off) >= 2):
                        off = (0.0, 0.0)
                    try:
                        off_x = float(off[0])
                        off_y = float(off[1])
                    except Exception:
                        off_x, off_y = (0.0, 0.0)
                    pts = []
                    for p in layer.get("points", []) or []:
                        arr = np.asarray(p, dtype=np.float64).reshape(-1)
                        if arr.size >= 3:
                            pts.append([float(arr[0]) + off_x, float(arr[1]) + off_y, float(arr[2])])
                        elif arr.size == 2:
                            pts.append([float(arr[0]) + off_x, float(arr[1]) + off_y, 0.0])
                    if len(pts) >= 2:
                        out.append(pts)
                except Exception:
                    continue
        return out

    def get_cut_sections_world(self):
        """?대낫?닿린?? 諛붾떏??諛곗튂???⑤㈃(?꾨줈?뚯씪) ?대━?쇱씤???⑤㈃??ROI) ?붾뱶 醫뚰몴 諛섑솚"""
        out = []
        for line in getattr(self, "cut_section_world", [[], []]):
            pts = []
            for p in line:
                try:
                    arr = np.asarray(p, dtype=np.float64).reshape(-1)
                    if arr.size >= 3:
                        pts.append([float(arr[0]), float(arr[1]), float(arr[2])])
                    elif arr.size == 2:
                        pts.append([float(arr[0]), float(arr[1]), 0.0])
                except Exception:
                    continue
            out.append(pts)

        # ROI濡??앹꽦???⑤㈃(?덈뒗 寃쎌슦)???④퍡 ?대낫?닿린
        try:
            roi_sec = getattr(self, "roi_section_world", {}) or {}
            for key in ("x", "y"):
                line = roi_sec.get(key, None)
                if not line or len(line) < 2:
                    continue
                pts = []
                for p in line:
                    try:
                        arr = np.asarray(p, dtype=np.float64).reshape(-1)
                        if arr.size >= 3:
                            pts.append([float(arr[0]), float(arr[1]), float(arr[2])])
                        elif arr.size == 2:
                            pts.append([float(arr[0]), float(arr[1]), 0.0])
                    except Exception:
                        continue
                if len(pts) >= 2:
                    out.append(pts)
        except Exception:
            _log_ignored_exception()

        # Saved polyline layers (visible)
        obj = self.selected_obj
        if obj is not None:
            for layer in getattr(obj, "polyline_layers", []) or []:
                try:
                    if not bool(layer.get("visible", True)):
                        continue
                    if str(layer.get("kind", "")) != "section_profile":
                        continue
                    off = layer.get("offset", None)
                    if not (isinstance(off, (list, tuple)) and len(off) >= 2):
                        off = (0.0, 0.0)
                    try:
                        off_x = float(off[0])
                        off_y = float(off[1])
                    except Exception:
                        off_x, off_y = (0.0, 0.0)
                    pts = []
                    for p in layer.get("points", []) or []:
                        arr = np.asarray(p, dtype=np.float64).reshape(-1)
                        if arr.size >= 3:
                            pts.append([float(arr[0]) + off_x, float(arr[1]) + off_y, float(arr[2])])
                        elif arr.size == 2:
                            pts.append([float(arr[0]) + off_x, float(arr[1]) + off_y, 0.0])
                    if len(pts) >= 2:
                        out.append(pts)
                except Exception:
                    continue
        return out

    def schedule_cut_section_update(self, index: int, delay_ms: int = 0):
        try:
            idx = int(index)
        except Exception:
            return
        if idx not in (0, 1):
            return
        self._cut_section_pending_indices.add(idx)
        self._cut_section_timer.start(max(0, int(delay_ms)))

    def _on_mesh_transform_changed(self):
        """硫붿돩 蹂???? ?⑤㈃/ROI ???섏〈 ?곗씠?곕? ?붾컮?댁뒪 媛깆떊."""
        self._clear_cutline_tape_cache()
        try:
            if getattr(self, "crosshair_enabled", False):
                self.schedule_crosshair_profile_update(150)
        except Exception:
            _log_ignored_exception()

        try:
            if getattr(self, "line_section_enabled", False):
                self.schedule_line_profile_update(150)
        except Exception:
            _log_ignored_exception()

        try:
            if getattr(self, "roi_enabled", False):
                self.schedule_roi_edges_update(150)
        except Exception:
            _log_ignored_exception()

        try:
            lines = getattr(self, "cut_lines", [[], []]) or [[], []]
            for idx in (0, 1):
                if idx < len(lines) and lines[idx] and len(lines[idx]) >= 2:
                    self.schedule_cut_section_update(idx, delay_ms=150)
        except Exception:
            _log_ignored_exception()

    def _cut_line_axis_for_index(self, index: int) -> str:
        try:
            idx = int(index)
        except Exception:
            idx = 0
        if idx not in (0, 1):
            idx = 0
        try:
            locks = getattr(self, "cut_line_axis_lock", [None, None]) or [None, None]
            axis = str(locks[idx]).strip().lower() if idx < len(locks) else ""
            if axis in ("x", "y"):
                return axis
        except Exception:
            pass
        return "x" if idx == 0 else "y"

    def _layout_cut_section_world(self, profile: list[tuple[float, float]], index: int):
        obj = self.selected_obj
        if obj is None or not profile:
            return []

        try:
            b = obj.get_world_bounds()
            min_x, min_y = float(b[0][0]), float(b[0][1])
            max_x, max_y = float(b[1][0]), float(b[1][1])
            extent_x = float(max_x - min_x)
            extent_y = float(max_y - min_y)
            margin = max(2.0, 0.05 * max(extent_x, extent_y))

            s = np.array([p[0] for p in profile], dtype=np.float64)
            z = np.array([p[1] for p in profile], dtype=np.float64)
            finite = np.isfinite(s) & np.isfinite(z)
            s = s[finite]
            z = z[finite]
            if s.size < 2 or z.size < 2:
                return []

            s_min = float(np.nanmin(s))
            s_max = float(np.nanmax(s))
            z_min = float(np.nanmin(z))
            s_span = max(1e-6, float(s_max - s_min))

            pts_world = []
            idx = int(index)
            axis = self._cut_line_axis_for_index(idx)
            extent_along_axis = extent_x if axis == "x" else extent_y
            if axis == "x":
                # X異?諛⑺뼢 ?⑤㈃: 硫붿돩 ?곷떒(+Y)??諛곗튂 (s -> X, z -> Y)
                base_x = min_x
                base_y = max_y + margin
            else:
                # Y異?諛⑺뼢 ?⑤㈃: 硫붿돩 ?곗륫(+X)??諛곗튂 (z -> X, s -> Y)
                base_x = max_x + margin
                base_y = min_y
            scale_s = max(1e-6, extent_along_axis) / s_span

            flip_s = False
            if axis == "y":
                # ?ъ슜?먭? ?⑤㈃?좎쓣 ??>?꾨옒(?먮뒗 ?ㅻⅨ履?>?쇱そ)濡?洹몃━硫?
                # s 異?諛⑺뼢???ㅼ쭛? 寃곌낵媛 ???섍? ?ㅼ쭛? 蹂댁씪 ???덉쓬.
                # ?⑤㈃?좎쓽 吏꾪뻾 諛⑺뼢??+Y(?몃줈) / +X(媛濡?媛 ?섎룄濡?s 異뺤쓣 嫄곗슱?곸쑝濡??ㅼ쭛?붾떎.
                try:
                    lines = getattr(self, "cut_lines", [[], []]) or [[], []]
                    if idx < len(lines) and lines[idx] and len(lines[idx]) >= 2:
                        p0 = np.asarray(lines[idx][0], dtype=np.float64).reshape(-1)
                        p1 = np.asarray(lines[idx][-1], dtype=np.float64).reshape(-1)
                        if p0.size >= 2 and p1.size >= 2:
                            dx = float(p1[0] - p0[0])
                            dy = float(p1[1] - p0[1])
                            if abs(dy) >= abs(dx):
                                flip_s = dy < 0.0
                            else:
                                flip_s = dx < 0.0
                except Exception:
                    flip_s = False

            for si, zi in zip(s.tolist(), z.tolist()):
                if axis == "x":
                    pts_world.append([base_x + (float(si) - s_min) * scale_s, base_y + (float(zi) - z_min), 0.0])
                else:
                    s_term = (s_max - float(si)) if flip_s else (float(si) - s_min)
                    pts_world.append([base_x + (float(zi) - z_min), base_y + s_term * scale_s, 0.0])
            return pts_world
        except Exception:
            return []

    def _request_cut_section_compute(self):
        obj = self.selected_obj
        if obj is None or obj.mesh is None:
            return

        thread = getattr(self, "_cut_section_thread", None)
        if thread is not None and thread.isRunning():
            return

        pending = getattr(self, "_cut_section_pending_indices", None)
        if not pending:
            return

        try:
            preferred = int(getattr(self, "cut_line_active", 0))
        except Exception:
            preferred = 0
        if preferred in pending:
            idx = preferred
        else:
            idx = min(pending)
        pending.discard(idx)
        if idx not in (0, 1):
            return

        poly = self.cut_lines[idx]
        if poly is None or len(poly) < 2:
            self.cut_section_profiles[idx] = []
            self.cut_section_world[idx] = []
            self.update()
            return

        self._cut_section_thread = _CutPolylineSectionProfileThread(
            obj.mesh,
            translation=obj.translation.copy(),
            rotation_deg=obj.rotation.copy(),
            scale=float(obj.scale),
            index=idx,
            polyline_world=[np.asarray(p, dtype=np.float64).copy() for p in poly],
        )
        self._cut_section_thread.computed.connect(self._on_cut_section_computed)
        self._cut_section_thread.failed.connect(self._on_cut_section_failed)
        self._cut_section_thread.finished.connect(self._on_cut_section_finished)
        self._cut_section_thread.start()

    def _on_cut_section_finished(self):
        thread = getattr(self, "_cut_section_thread", None)
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                _log_ignored_exception()
        self._cut_section_thread = None

        # ?湲?以묒씤 理쒖떊 ?붿껌???덉쑝硫?利됱떆 ?ㅼ떆 ?쒕룄
        if getattr(self, "_cut_section_pending_indices", None):
            self._cut_section_timer.start(1)

    def _on_cut_section_computed(self, result: object):
        if not isinstance(result, dict):
            return
        try:
            idx = int(result.get("index", -1))
            profile = result.get("profile", [])
        except Exception:
            return

        if idx not in (0, 1):
            return

        self.cut_section_profiles[idx] = profile
        self.cut_section_world[idx] = self._layout_cut_section_world(profile, idx)
        self.update()

    def _on_cut_section_failed(self, message: str):
        # ?ㅽ뙣?대룄 UI??怨꾩냽 ?숈옉?댁빞 ??        # print(f"Cut section compute failed: {message}")
        pass

    def draw_line_section(self):
        """?곷㈃(Top)?먯꽌 洹몄? 吏곸꽑 ?⑤㈃ ?쒓컖??"""
        if self.line_section_start is None or self.line_section_end is None:
            return

        glDisable(GL_LIGHTING)

        p0 = self.line_section_start
        p1 = self.line_section_end

        # 1) 諛붾떏 ?됰㈃ ??而ㅽ똿 ?쇱씤 (?ㅻ젋吏)
        z = 0.05
        glLineWidth(2.5)
        glColor4f(1.0, 0.55, 0.0, 0.85)
        glBegin(GL_LINES)
        glVertex3f(float(p0[0]), float(p0[1]), z)
        glVertex3f(float(p1[0]), float(p1[1]), z)
        glEnd()

        # 2) ?붾뱶?ъ씤??留덉빱
        # 3) 硫붿돩 ?⑤㈃??(?쇱엫)
        if self.line_section_contours:
            glLineWidth(3.0)
            glColor3f(0.2, 1.0, 0.2)
            for contour in self.line_section_contours:
                if len(contour) < 2:
                    continue
                glBegin(GL_LINE_STRIP)
                for pt in contour:
                    glVertex3fv(pt)
                glEnd()

        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def _cutline_constrain_ortho(
        self,
        last_pt: np.ndarray,
        candidate: np.ndarray,
        *,
        force_lock_axis: bool = True,
    ) -> np.ndarray:
        """CAD Ortho: 留덉?留???湲곗??쇰줈 X/Y 以??섎굹留?蹂??"""
        last_pt = np.asarray(last_pt, dtype=np.float64)
        candidate = np.asarray(candidate, dtype=np.float64).copy()
        try:
            idx = int(getattr(self, "cut_line_active", 0))
        except Exception:
            idx = 0
        if idx not in (0, 1):
            idx = 0
        lock = None
        try:
            locks = getattr(self, "cut_line_axis_lock", [None, None]) or [None, None]
            lock = str(locks[idx]).strip().lower() if idx < len(locks) and locks[idx] is not None else None
        except Exception:
            lock = None

        if force_lock_axis and lock == "x":
            candidate[1] = last_pt[1]
        elif force_lock_axis and lock == "y":
            candidate[0] = last_pt[0]
        else:
            dx = float(candidate[0] - last_pt[0])
            dy = float(candidate[1] - last_pt[1])
            if abs(dx) >= abs(dy):
                candidate[1] = last_pt[1]
            else:
                candidate[0] = last_pt[0]
        # Keep line input free in top view; surface attachment is render-only.
        return candidate

    @staticmethod
    def _closest_point_on_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Christer Ericson ?뚭퀬由ъ쬁 湲곕컲 ?쇨컖??理쒓렐?묒젏."""
        p = np.asarray(point, dtype=np.float64).reshape(-1)[:3]
        a = np.asarray(a, dtype=np.float64).reshape(-1)[:3]
        b = np.asarray(b, dtype=np.float64).reshape(-1)[:3]
        c = np.asarray(c, dtype=np.float64).reshape(-1)[:3]

        ab = b - a
        ac = c - a
        ap = p - a
        d1 = float(np.dot(ab, ap))
        d2 = float(np.dot(ac, ap))
        if d1 <= 0.0 and d2 <= 0.0:
            return a

        bp = p - b
        d3 = float(np.dot(ab, bp))
        d4 = float(np.dot(ac, bp))
        if d3 >= 0.0 and d4 <= d3:
            return b

        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            return a + v * ab

        cp = p - c
        d5 = float(np.dot(ab, cp))
        d6 = float(np.dot(ac, cp))
        if d6 >= 0.0 and d5 <= d6:
            return c

        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            return a + w * ac

        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return b + w * (c - b)

        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        return a + ab * v + ac * w

    def _snap_cutline_point_to_surface(
        self,
        point: np.ndarray,
        *,
        z_hint: float | None = None,
        max_xy_distance: float | None = None,
    ) -> np.ndarray | None:
        """源딆씠 ???ㅽ뙣/?쒖빟 ??XY ?먯쓣 ?좏깮 硫붿돩 ?쒕㈃ 理쒓렐?묒젏?쇰줈 ?ы닾??"""
        obj = self.selected_obj
        if obj is None or getattr(obj, "mesh", None) is None:
            return None

        p = np.asarray(point, dtype=np.float64).reshape(-1)
        if p.size < 2:
            return None
        if p.size >= 3:
            seed = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float64)
        else:
            seed = np.array([float(p[0]), float(p[1]), 0.0], dtype=np.float64)

        try:
            max_xy = float(max_xy_distance) if max_xy_distance is not None else float("nan")
        except Exception:
            max_xy = float("nan")
        if np.isfinite(max_xy) and max_xy >= 0.0:
            try:
                wb = np.asarray(obj.get_world_bounds(), dtype=np.float64)
                if wb.shape == (2, 3) and np.isfinite(wb).all():
                    margin = float(max_xy)
                    if (
                        float(seed[0]) < float(wb[0, 0]) - margin
                        or float(seed[0]) > float(wb[1, 0]) + margin
                        or float(seed[1]) < float(wb[0, 1]) - margin
                        or float(seed[1]) > float(wb[1, 1]) + margin
                    ):
                        return None
            except Exception:
                _log_ignored_exception()

        try:
            zh = float(z_hint) if z_hint is not None else float("nan")
        except Exception:
            zh = float("nan")
        if np.isfinite(zh):
            seed[2] = zh
        else:
            try:
                wb = np.asarray(obj.get_world_bounds(), dtype=np.float64)
                if wb.shape == (2, 3) and np.isfinite(wb).all():
                    seed[2] = float((wb[0, 2] + wb[1, 2]) * 0.5)
            except Exception:
                _log_ignored_exception()

        try:
            res = self.pick_face_at_point(seed, return_index=True)
            if not res:
                return None
            _fi, verts = res
            if not isinstance(verts, (list, tuple)) or len(verts) < 3:
                return None
            a = np.asarray(verts[0], dtype=np.float64).reshape(-1)[:3]
            b = np.asarray(verts[1], dtype=np.float64).reshape(-1)[:3]
            c = np.asarray(verts[2], dtype=np.float64).reshape(-1)[:3]
            cp = self._closest_point_on_triangle(seed, a, b, c)
            if cp.size >= 3 and np.isfinite(cp[:3]).all():
                return cp[:3].copy()
        except Exception:
            _log_ignored_exception()
        return None

    def _finish_cut_lines_current(self):
        """Enter/?고겢由?쑝濡??꾩옱 ?쒖꽦 ?⑤㈃???낅젰??留덈Т由?"""
        try:
            idx_done = int(getattr(self, "cut_line_active", 0))
        except Exception:
            idx_done = 0
        if idx_done not in (0, 1):
            idx_done = 0
        self.cut_line_drawing = False
        self.cut_line_preview = None

        # Mark this polyline as "finalized" if it is valid (>=2 points).
        try:
            if len(self.cut_lines[idx_done]) >= 2:
                self._cut_line_final[idx_done] = True
            else:
                self._cut_line_final[idx_done] = False
        except Exception:
            _log_ignored_exception()

        # ?⑤㈃(?꾨줈?뚯씪) 怨꾩궛 ?붿껌
        try:
            if idx_done in (0, 1) and len(self.cut_lines[idx_done]) >= 2:
                self.schedule_cut_section_update(idx_done, delay_ms=0)
        except Exception:
            _log_ignored_exception()

        # ?ㅻⅨ ?좎씠 "誘명솗???대㈃ ?먮룞 ?꾪솚 (鍮????ы븿)
        try:
            other = 1 - int(idx_done)
            final = getattr(self, "_cut_line_final", [False, False])
            if other in (0, 1) and not bool(final[other]):
                self.cut_line_active = other
                try:
                    self.cutLineActiveChanged.emit(int(other))
                except Exception:
                    _log_ignored_exception()
                line = self.cut_lines[other]
                self.cut_line_drawing = bool(line) and not bool(final[other])
                try:
                    role = "Length" if other == 0 else "Width"
                    locks = getattr(self, "cut_line_axis_lock", [None, None]) or [None, None]
                    lk = locks[other] if other < len(locks) else None
                    axis_txt = "X" if str(lk).lower().startswith("x") else ("Y" if str(lk).lower().startswith("y") else "X/Y")
                    self.status_info = f"?㎛ ?ㅼ쓬 ?⑤㈃?? {role} ({axis_txt}異??ㅻ깄)"
                except Exception:
                    _log_ignored_exception()
        except Exception:
            _log_ignored_exception()

        # ?????뺤젙?섎㈃ 紐⑤뱶 ?먮룞 醫낅즺
        try:
            final = getattr(self, "_cut_line_final", [False, False])
            if bool(final[0]) and bool(final[1]):
                self.set_cut_lines_enabled(False)
                self.status_info = "??Length/Width ?⑤㈃???낅젰 ?꾨즺"
                try:
                    self.cutLinesAutoEnded.emit()
                except Exception:
                    _log_ignored_exception()
        except Exception:
            _log_ignored_exception()

        self.update()

    def _densify_cut_polyline(self, line: list[np.ndarray] | list[list[float]], step_world: float = 0.6) -> list[np.ndarray]:
        """?쒓컖?붿슜 ?대━?쇱씤 蹂닿컙(?뚯씠??怨좊Т諛대뱶 ?쒗쁽 ?덉쭏 ?μ긽)."""
        try:
            step = float(step_world)
        except Exception:
            step = 0.6
        if not np.isfinite(step) or step <= 1e-3:
            step = 0.6

        pts = []
        try:
            for p in line or []:
                arr = np.asarray(p, dtype=np.float64).reshape(-1)
                if arr.size >= 3 and np.isfinite(arr[:3]).all():
                    pts.append(arr[:3].copy())
                elif arr.size >= 2 and np.isfinite(arr[:2]).all():
                    pts.append(np.array([float(arr[0]), float(arr[1]), 0.0], dtype=np.float64))
        except Exception:
            return []
        if len(pts) < 2:
            return pts

        out: list[np.ndarray] = [pts[0]]
        for i in range(1, len(pts)):
            p0 = np.asarray(pts[i - 1], dtype=np.float64)
            p1 = np.asarray(pts[i], dtype=np.float64)
            d = p1 - p0
            seg_len = float(np.linalg.norm(d))
            if seg_len <= 1e-9:
                continue
            n = max(1, int(np.ceil(seg_len / step)))
            for k in range(1, n + 1):
                t = float(k) / float(n)
                out.append((1.0 - t) * p0 + t * p1)
        return out

    def _build_cutline_surface_tape_strips(
        self,
        line: list[np.ndarray] | list[list[float]],
        *,
        step_world: float = 0.5,
    ) -> list[list[np.ndarray]]:
        """Return mesh-attached strips only for polyline portions that pass near the mesh."""
        obj = self.selected_obj
        if obj is None or getattr(obj, "mesh", None) is None:
            return []

        dense = self._densify_cut_polyline(line, step_world=step_world)
        if len(dense) < 2:
            return []

        try:
            wb = np.asarray(obj.get_world_bounds(), dtype=np.float64)
            if wb.shape == (2, 3) and np.isfinite(wb).all():
                span_xy = float(max(float(wb[1, 0] - wb[0, 0]), float(wb[1, 1] - wb[0, 1])))
                z_hint = float((wb[0, 2] + wb[1, 2]) * 0.5)
            else:
                span_xy = 10.0
                z_hint = 0.0
        except Exception:
            span_xy = 10.0
            z_hint = 0.0
        tol_xy = float(max(float(step_world) * 1.75, span_xy * 0.015, 0.35))
        tol_xy = float(min(tol_xy, max(2.0, span_xy * 0.20)))

        line_sig: list[tuple[float, float, float]] = []
        for p in line or []:
            arr = np.asarray(p, dtype=np.float64).reshape(-1)
            if arr.size >= 2 and np.isfinite(arr[:2]).all():
                zz = float(arr[2]) if arr.size >= 3 and np.isfinite(arr[2]) else 0.0
                line_sig.append((float(np.round(arr[0], 4)), float(np.round(arr[1], 4)), float(np.round(zz, 4))))
        if len(line_sig) < 2:
            return []

        try:
            tr = np.asarray(getattr(obj, "translation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            rot = np.asarray(getattr(obj, "rotation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            sc = float(getattr(obj, "scale", 1.0) or 1.0)
        except Exception:
            tr = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            rot = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            sc = 1.0
        obj_sig = (
            float(np.round(tr[0] if tr.size > 0 else 0.0, 5)),
            float(np.round(tr[1] if tr.size > 1 else 0.0, 5)),
            float(np.round(tr[2] if tr.size > 2 else 0.0, 5)),
            float(np.round(rot[0] if rot.size > 0 else 0.0, 4)),
            float(np.round(rot[1] if rot.size > 1 else 0.0, 4)),
            float(np.round(rot[2] if rot.size > 2 else 0.0, 4)),
            float(np.round(sc, 6)),
            int(getattr(getattr(obj, "mesh", None), "n_faces", 0) or 0),
            float(np.round(tol_xy, 4)),
            float(np.round(step_world, 3)),
        )
        cache_key = (int(id(obj)), int(id(getattr(obj, "mesh", None))), obj_sig, tuple(line_sig))

        cache = getattr(self, "_cutline_tape_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._cutline_tape_cache = cache
        cached = cache.get(cache_key, None)
        if cached is not None:
            return cached

        strips: list[list[np.ndarray]] = []
        cur: list[np.ndarray] = []
        for p in dense:
            arr = np.asarray(p, dtype=np.float64).reshape(-1)
            if arr.size < 2 or (not np.isfinite(arr[:2]).all()):
                if len(cur) >= 2:
                    strips.append(cur)
                cur = []
                continue

            zz = float(arr[2]) if arr.size >= 3 and np.isfinite(arr[2]) else z_hint
            guess = np.array([float(arr[0]), float(arr[1]), float(zz)], dtype=np.float64)
            snapped = self._snap_cutline_point_to_surface(
                guess,
                z_hint=z_hint,
                max_xy_distance=tol_xy * 2.0,
            )
            if snapped is None:
                if len(cur) >= 2:
                    strips.append(cur)
                cur = []
                continue

            sp = np.asarray(snapped, dtype=np.float64).reshape(-1)
            if sp.size < 3 or (not np.isfinite(sp[:3]).all()):
                if len(cur) >= 2:
                    strips.append(cur)
                cur = []
                continue

            dxy = float(np.linalg.norm(sp[:2] - guess[:2]))
            if dxy > tol_xy:
                if len(cur) >= 2:
                    strips.append(cur)
                cur = []
                continue

            wp = np.array([float(sp[0]), float(sp[1]), float(sp[2])], dtype=np.float64)
            if (not cur) or float(np.linalg.norm(wp - cur[-1])) > 1e-6:
                cur.append(wp)

        if len(cur) >= 2:
            strips.append(cur)

        cache[cache_key] = strips
        if len(cache) > 48:
            try:
                overflow = int(len(cache) - 48)
                for old_key in list(cache.keys())[: max(0, overflow)]:
                    if old_key == cache_key:
                        continue
                    cache.pop(old_key, None)
            except Exception:
                pass

        return strips

    def draw_cut_lines(self):
        """?⑤㈃??2媛? 媛?대뱶 ?쇱씤 ?쒓컖??(??긽 ?붾㈃ ?꾨줈)"""
        lines = getattr(self, "cut_lines", [[], []])
        if not lines:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        z = 0.08  # 諛붾떏?먯꽌 ?댁쭩 ?꾩?
        colors = [
            (1.0, 0.45, 0.15, 0.95),  # length: orange-ish
            (0.10, 0.72, 0.45, 0.95),  # width: green-ish
        ]

        for i, line in enumerate(lines):
            if not line:
                continue
            col = colors[i % 2]
            glColor4f(*col)
            glLineWidth(2.5 if int(self.cut_line_active) == i else 2.0)

            if len(line) == 1:
                p0 = np.asarray(line[0], dtype=np.float64)
                glPointSize(7.0)
                glBegin(GL_POINTS)
                glVertex3f(float(p0[0]), float(p0[1]), z)
                glEnd()
                glPointSize(1.0)
                continue

            glBegin(GL_LINE_STRIP)
            for p in line:
                p0 = np.asarray(p, dtype=np.float64)
                glVertex3f(float(p0[0]), float(p0[1]), z)
            glEnd()

        # "Sticker / rubber-band" overlay: line follows picked surface points on the mesh.
        glEnable(GL_DEPTH_TEST)
        for i, line in enumerate(lines):
            if line is None or len(line) < 2:
                continue
            col = colors[i % 2]
            strips = self._build_cutline_surface_tape_strips(line, step_world=0.5)
            if not strips:
                continue
            for dense in strips:
                # tape base
                glColor4f(float(col[0] * 0.65), float(col[1] * 0.65), float(col[2] * 0.65), 0.58)
                glLineWidth(6.0 if int(self.cut_line_active) == i else 4.8)
                glBegin(GL_LINE_STRIP)
                for p in dense:
                    p0 = np.asarray(p, dtype=np.float64).reshape(-1)
                    zz = float(p0[2]) if p0.size >= 3 else 0.0
                    if not np.isfinite(zz):
                        zz = 0.0
                    glVertex3f(float(p0[0]), float(p0[1]), float(zz + 0.012))
                glEnd()
                # tape color stroke
                glColor4f(float(col[0]), float(col[1]), float(col[2]), 0.96)
                glLineWidth(3.6 if int(self.cut_line_active) == i else 3.0)
                glBegin(GL_LINE_STRIP)
                for p in dense:
                    p0 = np.asarray(p, dtype=np.float64).reshape(-1)
                    zz = float(p0[2]) if p0.size >= 3 else 0.0
                    if not np.isfinite(zz):
                        zz = 0.0
                    glVertex3f(float(p0[0]), float(p0[1]), float(zz + 0.015))
                glEnd()
                # slim highlight for sticker-like sheen
                glColor4f(1.0, 1.0, 1.0, 0.38)
                glLineWidth(1.4 if int(self.cut_line_active) == i else 1.1)
                glBegin(GL_LINE_STRIP)
                for p in dense:
                    p0 = np.asarray(p, dtype=np.float64).reshape(-1)
                    zz = float(p0[2]) if p0.size >= 3 else 0.0
                    if not np.isfinite(zz):
                        zz = 0.0
                    glVertex3f(float(p0[0]), float(p0[1]), float(zz + 0.018))
                glEnd()
        glDisable(GL_DEPTH_TEST)

        # ?꾨━酉??멸렇癒쇳듃
        if self.cut_line_drawing and self.cut_line_preview is not None:
            try:
                idx = int(self.cut_line_active)
                active = lines[idx]
                if active:
                    p_last = np.asarray(active[-1], dtype=np.float64)
                    p_prev = np.asarray(self.cut_line_preview, dtype=np.float64)
                    glColor4f(*colors[idx % 2])
                    glLineWidth(2.0)
                    glBegin(GL_LINES)
                    glVertex3f(float(p_last[0]), float(p_last[1]), z)
                    glVertex3f(float(p_prev[0]), float(p_prev[1]), z)
                    glEnd()
            except Exception:
                _log_ignored_exception()

        # ?⑤㈃ ?꾨줈?뚯씪(諛붾떏 諛곗튂) ?뚮뜑留?
        profiles = getattr(self, "cut_section_world", [[], []])
        z_profile = 0.12
        if profiles:
            glColor4f(0.1, 0.1, 0.1, 0.9)
            glLineWidth(2.0)
            for pts in profiles:
                if pts is None or len(pts) < 2:
                    continue
                glBegin(GL_LINE_STRIP)
                for p in pts:
                    p0 = np.asarray(p, dtype=np.float64)
                    glVertex3f(float(p0[0]), float(p0[1]), z_profile)
                glEnd()

        # Saved polyline layers
        obj = self.selected_obj
        if obj is not None:
            layers = getattr(obj, "polyline_layers", None) or []
            for layer in layers:
                try:
                    if not bool(layer.get("visible", True)):
                        continue
                    pts = layer.get("points", None) or []
                    if len(pts) < 2:
                        continue
                    kind = str(layer.get("kind", ""))
                    off = layer.get("offset", None)
                    if not (isinstance(off, (list, tuple)) and len(off) >= 2):
                        off = (0.0, 0.0)
                    try:
                        off_x = float(off[0])
                        off_y = float(off[1])
                    except Exception:
                        off_x, off_y = (0.0, 0.0)
                    col = layer.get("color", None)
                    if col is None:
                        col = (0.2, 0.2, 0.2, 0.7)
                    try:
                        r, g, b, a = [float(x) for x in col[:4]]
                    except Exception:
                        r, g, b, a = (0.2, 0.2, 0.2, 0.7)

                    w = float(layer.get("width", 1.6))
                    glColor4f(r, g, b, a)
                    glLineWidth(max(1.0, w))
                    z_use = z if kind == "cut_line" else z_profile

                    glBegin(GL_LINE_STRIP)
                    for p in pts:
                        arr = np.asarray(p, dtype=np.float64).reshape(-1)
                        if arr.size >= 2:
                            glVertex3f(float(arr[0]) + off_x, float(arr[1]) + off_y, float(z_use))
                    glEnd()
                except Exception:
                    continue

        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def schedule_crosshair_profile_update(self, delay_ms: int = 120):
        if not self.crosshair_enabled:
            return
        self._crosshair_pending_pos = np.array(self.crosshair_pos, dtype=np.float64)
        self._crosshair_profile_timer.start(max(0, int(delay_ms)))

    def _request_crosshair_profile_compute(self):
        if not self.crosshair_enabled:
            return

        obj = self.selected_obj
        if obj is None or obj.mesh is None:
            self.x_profile = []
            self.y_profile = []
            self._world_x_profile = []
            self._world_y_profile = []
            self.profileUpdated.emit([], [])
            self.update()
            return

        thread = getattr(self, "_crosshair_profile_thread", None)
        if thread is not None and thread.isRunning():
            return

        pos = self._crosshair_pending_pos
        if pos is None:
            pos = np.array(self.crosshair_pos, dtype=np.float64)
        self._crosshair_pending_pos = None

        self._crosshair_profile_thread = _CrosshairProfileThread(
            obj.mesh,
            obj.translation.copy(),
            obj.rotation.copy(),
            float(obj.scale),
            float(pos[0]),
            float(pos[1]),
        )
        self._crosshair_profile_thread.computed.connect(self._on_crosshair_profile_computed)
        self._crosshair_profile_thread.failed.connect(self._on_crosshair_profile_failed)
        self._crosshair_profile_thread.finished.connect(self._on_crosshair_profile_finished)
        self._crosshair_profile_thread.start()

    def _on_crosshair_profile_computed(self, result: dict[str, Any]):
        if not self.crosshair_enabled:
            return
        if self._crosshair_pending_pos is not None:
            return

        cx = float(result.get("cx", 0.0))
        cy = float(result.get("cy", 0.0))
        if not np.allclose(np.asarray(self.crosshair_pos, dtype=np.float64), [cx, cy], atol=1e-6):
            return

        self.x_profile = result.get("x_profile", []) or []
        self.y_profile = result.get("y_profile", []) or []
        world_x = result.get("world_x", None)
        world_y = result.get("world_y", None)
        self._world_x_profile = world_x if world_x is not None else []
        self._world_y_profile = world_y if world_y is not None else []

        self.profileUpdated.emit(self.x_profile, self.y_profile)
        self.update()

    def _on_crosshair_profile_failed(self, message: str):
        self.x_profile = []
        self.y_profile = []
        self._world_x_profile = []
        self._world_y_profile = []
        try:
            self.profileUpdated.emit([], [])
        except Exception:
            _log_ignored_exception()
        self.update()

    def _on_crosshair_profile_finished(self):
        thread = getattr(self, "_crosshair_profile_thread", None)
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                _log_ignored_exception()
        self._crosshair_profile_thread = None

        if self.crosshair_enabled and self._crosshair_pending_pos is not None:
            self._crosshair_profile_timer.start(1)

    def schedule_line_profile_update(self, delay_ms: int = 120):
        if not getattr(self, "line_section_enabled", False):
            return
        if self.line_section_start is None or self.line_section_end is None:
            return
        self._line_pending_segment = (
            np.array(self.line_section_start, dtype=np.float64),
            np.array(self.line_section_end, dtype=np.float64),
        )
        self._line_profile_timer.start(max(0, int(delay_ms)))

    def _request_line_profile_compute(self):
        if not getattr(self, "line_section_enabled", False):
            return

        obj = self.selected_obj
        if obj is None or obj.mesh is None:
            self.line_profile = []
            self.line_section_contours = []
            try:
                self.lineProfileUpdated.emit([])
            except Exception:
                _log_ignored_exception()
            self.update()
            return

        thread = getattr(self, "_line_profile_thread", None)
        if thread is not None and thread.isRunning():
            return

        segment = self._line_pending_segment
        if segment is None:
            if self.line_section_start is None or self.line_section_end is None:
                return
            segment = (
                np.array(self.line_section_start, dtype=np.float64),
                np.array(self.line_section_end, dtype=np.float64),
            )
        self._line_pending_segment = None

        p0, p1 = segment
        self._line_profile_thread = _LineSectionProfileThread(
            obj.mesh,
            obj.translation.copy(),
            obj.rotation.copy(),
            float(obj.scale),
            p0,
            p1,
        )
        self._line_profile_thread.computed.connect(self._on_line_profile_computed)
        self._line_profile_thread.failed.connect(self._on_line_profile_failed)
        self._line_profile_thread.finished.connect(self._on_line_profile_finished)
        self._line_profile_thread.start()

    def _on_line_profile_computed(self, result: dict[str, Any]):
        if not getattr(self, "line_section_enabled", False):
            return
        if self._line_pending_segment is not None:
            return

        if self.line_section_start is None or self.line_section_end is None:
            return

        p0 = np.asarray(result.get("p0", self.line_section_start), dtype=np.float64)
        p1 = np.asarray(result.get("p1", self.line_section_end), dtype=np.float64)

        if not np.allclose(np.asarray(self.line_section_start, dtype=np.float64), p0, atol=1e-6):
            return
        if not np.allclose(np.asarray(self.line_section_end, dtype=np.float64), p1, atol=1e-6):
            return

        self.line_profile = result.get("profile", []) or []
        self.line_section_contours = result.get("contours", []) or []
        self.lineProfileUpdated.emit(self.line_profile)
        self.update()

    def _on_line_profile_failed(self, message: str):
        self.line_profile = []
        self.line_section_contours = []
        try:
            self.lineProfileUpdated.emit([])
        except Exception:
            _log_ignored_exception()
        self.update()

    def _on_line_profile_finished(self):
        thread = getattr(self, "_line_profile_thread", None)
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                _log_ignored_exception()
        self._line_profile_thread = None

        if getattr(self, "line_section_enabled", False) and self._line_pending_segment is not None:
            self._line_profile_timer.start(1)

    def update_line_section_profile(self):
        """?꾩옱 ?좏삎 ?⑤㈃(?쇱씤)?쇰줈遺???꾨줈?뚯씪 異붿텧"""
        if not self.selected_obj or self.selected_obj.mesh is None:
            self.clear_line_section()
            return

        if self.line_section_start is None or self.line_section_end is None:
            self.line_profile = []
            self.line_section_contours = []
            self.lineProfileUpdated.emit([])
            self.update()
            return

        p0 = np.array(self.line_section_start, dtype=float)
        p1 = np.array(self.line_section_end, dtype=float)

        d = p1 - p0
        d[2] = 0.0
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            self.line_profile = []
            self.line_section_contours = []
            self.lineProfileUpdated.emit([])
            self.update()
            return

        d_unit = d / length
        # ?섏쭅 ?⑤㈃ ?됰㈃??踰뺤꽑 (XY?먯꽌 ?쇱씤???섏쭅)
        world_normal = np.array([d_unit[1], -d_unit[0], 0.0], dtype=float)
        world_origin = p0

        obj = self.selected_obj
        from scipy.spatial.transform import Rotation as R
        inv_rot = R.from_euler('XYZ', obj.rotation, degrees=True).inv().as_matrix()
        inv_scale = 1.0 / obj.scale if obj.scale != 0 else 1.0

        local_origin = inv_scale * inv_rot @ (world_origin - obj.translation)
        local_normal = inv_rot @ world_normal

        tm = obj.to_trimesh()
        if tm is None:
            return []
        slicer = MeshSlicer(cast(Any, tm))
        contours_local = slicer.slice_with_plane(local_origin.tolist(), local_normal.tolist())

        # ?붾뱶 醫뚰몴濡?蹂???뚮뜑留??꾨줈?뚯씪??
        rot_mat = R.from_euler('XYZ', obj.rotation, degrees=True).as_matrix()
        trans = obj.translation
        scale = obj.scale

        world_contours = []
        for cnt in contours_local:
            w_cnt = (rot_mat @ (cnt * scale).T).T + trans
            world_contours.append(w_cnt)
        self.line_section_contours = world_contours

        # ?꾨줈?뚯씪: (嫄곕━, ?믪씠) - ?쇱씤 諛⑺뼢?쇰줈 ?ъ쁺
        best_profile = []
        best_span = 0.0

        # ?쇱씤 ?멸렇癒쇳듃 踰붿쐞濡??꾪꽣留?(?쎄컙 ?ъ쑀)
        margin = max(length * 0.02, 0.2)
        t_min = -margin
        t_max = length + margin

        for cnt in world_contours:
            if cnt is None or len(cnt) < 2:
                continue
            t = (cnt - p0) @ d_unit
            mask = (t >= t_min) & (t <= t_max)
            if int(mask.sum()) < 2:
                continue

            t_f = t[mask]
            z_f = cnt[mask, 2]
            span = float(t_f.max() - t_f.min())
            if span <= best_span:
                continue

            order = np.argsort(t_f)
            t_sorted = t_f[order]
            z_sorted = z_f[order]

            # 洹몃옒?꾨뒗 0遺???쒖옉?섎룄濡?shift
            t_sorted = t_sorted - float(t_sorted.min())

            best_profile = list(zip(t_sorted.tolist(), z_sorted.tolist()))
            best_span = span

        self.line_profile = best_profile
        self.lineProfileUpdated.emit(self.line_profile)
        self.update()

    def update_crosshair_profile(self):
        """?꾩옱 ??옄???꾩튂?먯꽌 ?⑤㈃ ?꾨줈?뚯씪 異붿텧"""
        if not self.selected_obj or self.selected_obj.mesh is None:
            self.x_profile = []
            self.y_profile = []
            return
            
        obj = self.selected_obj
        cx, cy = self.crosshair_pos
        
        from scipy.spatial.transform import Rotation as R
        inv_rot = R.from_euler('XYZ', obj.rotation, degrees=True).inv().as_matrix()
        inv_scale = 1.0 / obj.scale if obj.scale != 0 else 1.0
        
        def get_world_to_local(pts_world):
            # P_local = (1/S) * R^T * (P_world - T)
            return inv_scale * (inv_rot @ (pts_world - obj.translation).T).T

        def get_local_to_world(pts_local):
            # P_world = R * (S * P_local) + T
            rot_mat = R.from_euler('XYZ', obj.rotation, degrees=True).as_matrix()
            return (rot_mat @ (pts_local * obj.scale).T).T + obj.translation

        # 2. X異?諛⑺뼢 ?⑤㈃ (?됰㈃: Y = cy)
        # ?붾뱶 ?곸쓽 ?됰㈃: Origin=[0, cy, 0], Normal=[0, 1, 0]
        w_orig_x = np.array([0, cy, 0])
        w_norm_x = np.array([0, 1, 0])
        l_orig_x = get_world_to_local(w_orig_x.reshape(1,3))[0]
        l_norm_x = inv_rot @ w_norm_x # 踰뺤꽑? ?뚯쟾留?        
        # MeshData.section ?먮윭 ?섏젙???꾪빐 to_trimesh() ?ъ슜
        tm = obj.to_trimesh()
        if tm is None:
            self.x_profile = []
            self.y_profile = []
            return

        slicer = MeshSlicer(cast(Any, tm))
        contours_x = slicer.slice_with_plane(l_orig_x.tolist(), l_norm_x.tolist())
        
        # 3. Y異?諛⑺뼢 ?⑤㈃ (?됰㈃: X = cx)
        w_orig_y = np.array([cx, 0, 0])
        w_norm_y = np.array([1, 0, 0]) # X異뺤뿉 ?섏쭅???됰㈃
        l_orig_y = get_world_to_local(w_orig_y.reshape(1,3))[0]
        l_norm_y = inv_rot @ w_norm_y
        contours_y = slicer.slice_with_plane(l_orig_y.tolist(), l_norm_y.tolist())
        
        # 4. 寃곌낵 泥섎━ (洹몃옒?꾩슜 媛怨?
        # X ?꾨줈?뚯씪 (X異??곕씪 ?대룞 ?쒖쓽 Z媛?
        self.x_profile = []
        self._world_x_profile = []
        if contours_x:
            pts_local = np.vstack(contours_x)
            pts_world = get_local_to_world(pts_local)
            # X媛?湲곗??쇰줈 ?뺣젹
            idx = np.argsort(pts_world[:, 0])
            sorted_pts = pts_world[idx]
            self._world_x_profile = sorted_pts
            # 洹몃옒???곗씠?? (X醫뚰몴, Z醫뚰몴)
            self.x_profile = sorted_pts[:, [0, 2]].tolist()
            
        # Y ?꾨줈?뚯씪 (Y異??곕씪 ?대룞 ?쒖쓽 Z媛?
        self.y_profile = []
        self._world_y_profile = []
        if contours_y:
            pts_local = np.vstack(contours_y)
            pts_world = get_local_to_world(pts_local)
            # Y媛?湲곗??쇰줈 ?뺣젹
            idx = np.argsort(pts_world[:, 1])
            sorted_pts = pts_world[idx]
            self._world_y_profile = sorted_pts
            # 洹몃옒???곗씠?? (Y醫뚰몴, Z醫뚰몴)
            self.y_profile = sorted_pts[:, [1, 2]].tolist()
            
        self.profileUpdated.emit(self.x_profile, self.y_profile)
        self.update()

    def draw_roi_box(self):
        """2D ROI (?щ줈???곸뿭) 諛??몃뱾 ?쒓컖??"""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        x1, x2, y1, y2 = self.roi_bounds
        z = 0.08 # 諛붾떏?먯꽌 ?댁쭩 ?꾩? (Z-fight 諛⑹?)

        # Ensure min/max ordering (defensive)
        try:
            x1, x2 = (float(x1), float(x2)) if float(x1) <= float(x2) else (float(x2), float(x1))
            y1, y2 = (float(y1), float(y2)) if float(y1) <= float(y2) else (float(y2), float(y1))
        except Exception:
            pass

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # ROI rectangle (fill + outline) so the area feels "fixed"
        try:
            glColor4f(0.98, 0.82, 0.35, 0.10)
            glBegin(GL_QUADS)
            glVertex3f(float(x1), float(y1), float(z))
            glVertex3f(float(x2), float(y1), float(z))
            glVertex3f(float(x2), float(y2), float(z))
            glVertex3f(float(x1), float(y2), float(z))
            glEnd()

            glLineWidth(2.0)
            glColor4f(0.95, 0.62, 0.16, 0.68)
            glBegin(GL_LINE_LOOP)
            glVertex3f(float(x1), float(y1), float(z))
            glVertex3f(float(x2), float(y1), float(z))
            glVertex3f(float(x2), float(y2), float(z))
            glVertex3f(float(x1), float(y2), float(z))
            glEnd()
        except Exception:
            try:
                glEnd()
            except Exception:
                _log_ignored_exception()
        finally:
            glLineWidth(1.0)

        # 4-way edge handles (spear style).
        handle_offset = max(1.0, self.camera.distance * 0.010)
        handle_pos = {
            "bottom": (mid_x, float(y1) - handle_offset),
            "top": (mid_x, float(y2) + handle_offset),
            "left": (float(x1) - handle_offset, mid_y),
            "right": (float(x2) + handle_offset, mid_y),
        }

        def draw_spear_handle(cx: float, cy: float, edge: str):
            active = (str(getattr(self, "active_roi_edge", "")).strip().lower() == str(edge).strip().lower())
            if active:
                fill = (1.0, 0.70, 0.18, 0.98)
                stroke = (0.22, 0.15, 0.05, 1.0)
            else:
                fill = (0.95, 0.95, 0.95, 0.96)
                stroke = (0.16, 0.16, 0.16, 0.95)

            vx, vy = 0.0, 0.0
            if edge == "top":
                vy = 1.0
            elif edge == "bottom":
                vy = -1.0
            elif edge == "left":
                vx = -1.0
            elif edge == "right":
                vx = 1.0
            if vx == 0.0 and vy == 0.0:
                return

            shaft = max(1.8, self.camera.distance * 0.0135)
            head = max(1.2, self.camera.distance * 0.0090)
            half_w = max(0.55, self.camera.distance * 0.0036)

            px, py = -vy, vx
            tail = np.array([float(cx), float(cy)], dtype=np.float64)
            neck = tail + np.array([vx, vy], dtype=np.float64) * shaft
            tip = neck + np.array([vx, vy], dtype=np.float64) * head

            tail_l = tail - np.array([px, py], dtype=np.float64) * half_w
            tail_r = tail + np.array([px, py], dtype=np.float64) * half_w
            neck_l = neck - np.array([px, py], dtype=np.float64) * half_w
            neck_r = neck + np.array([px, py], dtype=np.float64) * half_w
            head_l = neck - np.array([px, py], dtype=np.float64) * (half_w * 1.85)
            head_r = neck + np.array([px, py], dtype=np.float64) * (half_w * 1.85)

            # soft drop-shadow for depth cue
            sdx = -px * half_w * 0.32
            sdy = -py * half_w * 0.32
            glColor4f(0.0, 0.0, 0.0, 0.22 if active else 0.16)
            glBegin(GL_QUADS)
            glVertex3f(float(tail_l[0] + sdx), float(tail_l[1] + sdy), float(z))
            glVertex3f(float(neck_l[0] + sdx), float(neck_l[1] + sdy), float(z))
            glVertex3f(float(neck_r[0] + sdx), float(neck_r[1] + sdy), float(z))
            glVertex3f(float(tail_r[0] + sdx), float(tail_r[1] + sdy), float(z))
            glEnd()
            glBegin(GL_TRIANGLES)
            glVertex3f(float(head_l[0] + sdx), float(head_l[1] + sdy), float(z))
            glVertex3f(float(tip[0] + sdx), float(tip[1] + sdy), float(z))
            glVertex3f(float(head_r[0] + sdx), float(head_r[1] + sdy), float(z))
            glEnd()

            glColor4f(*fill)
            glBegin(GL_QUADS)
            glVertex3f(float(tail_l[0]), float(tail_l[1]), float(z))
            glVertex3f(float(neck_l[0]), float(neck_l[1]), float(z))
            glVertex3f(float(neck_r[0]), float(neck_r[1]), float(z))
            glVertex3f(float(tail_r[0]), float(tail_r[1]), float(z))
            glEnd()
            glBegin(GL_TRIANGLES)
            glVertex3f(float(head_l[0]), float(head_l[1]), float(z))
            glVertex3f(float(tip[0]), float(tip[1]), float(z))
            glVertex3f(float(head_r[0]), float(head_r[1]), float(z))
            glEnd()

            # bevel highlight/shadow strips to mimic 3D spear
            sh = half_w * 0.28
            glColor4f(1.0, 1.0, 1.0, 0.42 if active else 0.30)
            glBegin(GL_QUADS)
            glVertex3f(float(tail_l[0]), float(tail_l[1]), float(z))
            glVertex3f(float(neck_l[0]), float(neck_l[1]), float(z))
            glVertex3f(float(neck_l[0] + px * sh), float(neck_l[1] + py * sh), float(z))
            glVertex3f(float(tail_l[0] + px * sh), float(tail_l[1] + py * sh), float(z))
            glEnd()
            glBegin(GL_TRIANGLES)
            glVertex3f(float(head_l[0]), float(head_l[1]), float(z))
            glVertex3f(float(tip[0]), float(tip[1]), float(z))
            glVertex3f(float(head_l[0] + px * sh), float(head_l[1] + py * sh), float(z))
            glEnd()

            glColor4f(0.0, 0.0, 0.0, 0.26 if active else 0.20)
            glBegin(GL_QUADS)
            glVertex3f(float(tail_r[0] - px * sh), float(tail_r[1] - py * sh), float(z))
            glVertex3f(float(neck_r[0] - px * sh), float(neck_r[1] - py * sh), float(z))
            glVertex3f(float(neck_r[0]), float(neck_r[1]), float(z))
            glVertex3f(float(tail_r[0]), float(tail_r[1]), float(z))
            glEnd()

            glColor4f(*stroke)
            glLineWidth(1.6)
            glBegin(GL_LINE_LOOP)
            glVertex3f(float(tail_l[0]), float(tail_l[1]), float(z))
            glVertex3f(float(neck_l[0]), float(neck_l[1]), float(z))
            glVertex3f(float(head_l[0]), float(head_l[1]), float(z))
            glVertex3f(float(tip[0]), float(tip[1]), float(z))
            glVertex3f(float(head_r[0]), float(head_r[1]), float(z))
            glVertex3f(float(neck_r[0]), float(neck_r[1]), float(z))
            glVertex3f(float(tail_r[0]), float(tail_r[1]), float(z))
            glEnd()
            glLineWidth(1.0)

        # Center move handle (click/drag to translate the ROI)
        try:
            csize = max(2.0, self.camera.distance * 0.02)
            if str(getattr(self, "active_roi_edge", "")).strip().lower() == "move":
                glColor4f(1.0, 0.75, 0.2, 1.0)
            else:
                glColor4f(1.0, 1.0, 1.0, 0.95)
            glLineWidth(2.0)
            glBegin(GL_LINE_LOOP)
            glVertex3f(float(mid_x - csize), float(mid_y), float(z))
            glVertex3f(float(mid_x), float(mid_y + csize), float(z))
            glVertex3f(float(mid_x + csize), float(mid_y), float(z))
            glVertex3f(float(mid_x), float(mid_y - csize), float(z))
            glEnd()
        except Exception:
            try:
                glEnd()
            except Exception:
                _log_ignored_exception()
        finally:
            glLineWidth(1.0)
        
        for _edge, (hx, hy) in handle_pos.items():
            draw_spear_handle(float(hx), float(hy), str(_edge))
        
        glLineWidth(1.0)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_roi_cut_edges(self):
        """ROI ?대━?묒쑝濡??앷린???섎┝ 寃쎄퀎???⑤㈃?? ?ㅻ쾭?덉씠"""
        edges = getattr(self, "roi_cut_edges", None)
        if not edges:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.5)

        active = getattr(self, "active_roi_edge", None)
        plane_active = None
        if isinstance(active, str):
            plane_active = {
                "left": "x1",
                "right": "x2",
                "bottom": "y1",
                "top": "y2",
            }.get(active)

        colors = {
            "x1": (1.0, 0.35, 0.25, 0.9),
            "x2": (1.0, 0.35, 0.25, 0.9),
            "y1": (0.25, 0.75, 1.0, 0.9),
            "y2": (0.25, 0.75, 1.0, 0.9),
        }

        for key, contours in edges.items():
            if not contours:
                continue
            col = colors.get(key, (1.0, 1.0, 0.0, 0.9))
            if plane_active == key:
                col = (1.0, 1.0, 0.2, 1.0)
                glLineWidth(3.5)
            else:
                glLineWidth(2.5)

            glColor4f(*col)
            for cnt in contours:
                try:
                    if cnt is None or len(cnt) < 2:
                        continue
                    glBegin(GL_LINE_STRIP)
                    for pt in cnt:
                        glVertex3fv(pt)
                    glEnd()
                except Exception:
                    continue

        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_roi_caps(self):
        """ROI濡??섎┛ ?⑤㈃??'罹??쒓퍚) 硫??쇰줈 梨꾩썙 ?뚮뜑留?

        NOTE: ROI ?대┰ ?뚮젅??GL_CLIP_PLANE1~4)??enable ???곹깭?먯꽌 ?몄텧?섎뒗 寃껋쓣 沅뚯옣.
        """
        caps = getattr(self, "roi_cap_verts", None)
        if not caps:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-1.0, -1.0)

        glBegin(GL_TRIANGLES)
        for key, verts in caps.items():
            if verts is None:
                continue

            # ?덈떒 ?대?媛 鍮꾩뼱 蹂댁씠吏 ?딅룄濡??뚯깋 ?ㅻ㈃?쇰줈 罹≪쓣 梨꾩썎?덈떎.
            if str(key).startswith("x"):
                glColor4f(0.60, 0.60, 0.60, 1.0)
            else:
                glColor4f(0.55, 0.55, 0.55, 1.0)

            try:
                for v in verts:
                    glVertex3f(float(v[0]), float(v[1]), float(v[2]))
            except Exception:
                continue
        glEnd()

        glDisable(GL_POLYGON_OFFSET_FILL)
        glEnable(GL_BLEND)
        glEnable(GL_LIGHTING)

    @staticmethod
    def _polygon_area2(points_2d: np.ndarray) -> float:
        """2D ?대━怨?signed area*2 (shoelace)."""
        pts = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 3:
            return 0.0
        x = pts[:, 0]
        y = pts[:, 1]
        return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @staticmethod
    def _sanitize_polygon_2d(points_2d: np.ndarray, max_points: int = 800, eps: float = 1e-6) -> Optional[np.ndarray]:
        """triangulation??2D ?대━?쇱씤 ?뺣━(以묐났 ?쒓굅/?ロ옒 ?쒓굅/?ㅼ슫?섑뵆)."""
        try:
            pts = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
        except Exception:
            return None

        if pts.shape[0] < 3:
            return None

        finite = np.all(np.isfinite(pts), axis=1)
        pts = pts[finite]
        if pts.shape[0] < 3:
            return None

        # ?ロ엺 猷⑦봽硫?留덉?留????쒓굅
        try:
            if float(np.linalg.norm(pts[0] - pts[-1])) <= eps:
                pts = pts[:-1]
        except Exception:
            _log_ignored_exception()

        if pts.shape[0] < 3:
            return None

        # ?곗냽 以묐났 ?쒓굅
        if pts.shape[0] >= 2:
            d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            keep = np.ones((pts.shape[0],), dtype=bool)
            keep[1:] = d > eps
            pts = pts[keep]

        if pts.shape[0] < 3:
            return None

        # ?덈Т 議곕??섎㈃ ?ㅼ슫?섑뵆
        if int(pts.shape[0]) > int(max_points):
            step = int(np.ceil(float(pts.shape[0]) / float(max_points)))
            step = max(1, step)
            pts = pts[::step]

        if pts.shape[0] < 3:
            return None

        # ?ㅼ슫?섑뵆 ???ㅼ떆 ?뺣━
        try:
            if float(np.linalg.norm(pts[0] - pts[-1])) <= eps:
                pts = pts[:-1]
        except Exception:
            _log_ignored_exception()

        if pts.shape[0] < 3:
            return None

        if pts.shape[0] >= 2:
            d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            keep = np.ones((pts.shape[0],), dtype=bool)
            keep[1:] = d > eps
            pts = pts[keep]

        if pts.shape[0] < 3:
            return None
        return pts

    @staticmethod
    def _point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-12) -> bool:
        """2D point-in-triangle (including boundary)."""

        def sign(p1, p2, p3):
            return float((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]))

        d1 = sign(p, a, b)
        d2 = sign(p, b, c)
        d3 = sign(p, c, a)
        has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
        has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
        return not (has_neg and has_pos)

    @classmethod
    def _triangulate_ear_clip(
        cls, polygon_2d: np.ndarray, eps: float = 1e-12
    ) -> Optional[list[tuple[int, int, int]]]:
        """?⑥씪 猷⑦봽(? ?놁쓬)??ear-clipping ?쇨컖遺꾪븷. ?ㅽ뙣 ??None."""
        pts = np.asarray(polygon_2d, dtype=np.float64).reshape(-1, 2)
        n = int(pts.shape[0])
        if n < 3:
            return None

        # CCW濡??뺣젹
        idx_map = list(range(n))  # pts ?몃뜳??-> ?낅젰 polygon_2d ?몃뜳??
        if cls._polygon_area2(pts) < 0.0:
            pts = pts[::-1].copy()
            idx_map = idx_map[::-1]

        V = list(range(int(pts.shape[0])))
        tris = []

        def cross(a, b, c):
            return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

        max_iter = max(1000, n * n)
        it = 0
        while len(V) > 3 and it < max_iter:
            it += 1
            ear_found = False
            m = len(V)
            for k in range(m):
                i_prev = V[(k - 1) % m]
                i = V[k]
                i_next = V[(k + 1) % m]
                a = pts[i_prev]
                b = pts[i]
                c = pts[i_next]

                # convex check (CCW)
                if cross(a, b, c) <= eps:
                    continue

                # ear test: no other point inside triangle
                is_ear = True
                for j in V:
                    if j in (i_prev, i, i_next):
                        continue
                    if cls._point_in_triangle(pts[j], a, b, c, eps=eps):
                        is_ear = False
                        break

                if not is_ear:
                    continue

                tris.append((idx_map[i_prev], idx_map[i], idx_map[i_next]))
                del V[k]
                ear_found = True
                break

            if not ear_found:
                break

        if len(V) == 3:
            tris.append((idx_map[V[0]], idx_map[V[1]], idx_map[V[2]]))

        return tris if tris else None

    @staticmethod
    def _triangulate_convex_hull(points_2d: np.ndarray) -> Optional[list[tuple[int, int, int]]]:
        """fallback: convex hull fan triangulation."""
        pts = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 3:
            return None
        try:
            from scipy.spatial import ConvexHull
        except Exception:
            return None
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None
        hull_idx = list(getattr(hull, "vertices", []) or [])
        if len(hull_idx) < 3:
            return None
        tris = []
        a0 = hull_idx[0]
        for i in range(1, len(hull_idx) - 1):
            tris.append((a0, hull_idx[i], hull_idx[i + 1]))
        return tris if tris else None

    def _rebuild_roi_caps(self):
        """?꾩옱 ROI ?섎┝ 寃쎄퀎??roi_cut_edges)濡?罹??쇨컖?? 踰꾪뀓?ㅻ? 媛깆떊."""
        self.roi_cap_verts = {"x1": None, "x2": None, "y1": None, "y2": None}
        if not getattr(self, "roi_enabled", False):
            return

        edges = getattr(self, "roi_cut_edges", None) or {}
        if not edges:
            return

        try:
            x1, x2, y1, y2 = [float(v) for v in self.roi_bounds]
        except Exception:
            return

        plane_value = {"x1": x1, "x2": x2, "y1": y1, "y2": y2}

        for key in ("x1", "x2", "y1", "y2"):
            contours = edges.get(key, None)
            if not contours:
                continue

            # 媛????硫댁쟻 湲곗?) 猷⑦봽 1媛쒕쭔 罹≪쑝濡??ъ슜
            best_pts2d = None
            best_area = 0.0
            cloud_pts2d = []
            for cnt in contours:
                try:
                    pts3 = np.asarray(cnt, dtype=np.float64).reshape(-1, 3)
                except Exception:
                    continue

                if key.startswith("x"):
                    pts2 = pts3[:, [1, 2]]  # (y,z)
                else:
                    pts2 = pts3[:, [0, 2]]  # (x,z)

                if pts2.shape[0] >= 2:
                    cloud_pts2d.append(pts2)
                if pts2.shape[0] < 3:
                    continue

                pts2 = self._sanitize_polygon_2d(pts2, max_points=800, eps=1e-6)
                if pts2 is None or pts2.shape[0] < 3:
                    continue

                area = abs(self._polygon_area2(pts2)) * 0.5
                if area > best_area:
                    best_area = area
                    best_pts2d = pts2

            # 猷⑦봽瑜?紐?留뚮뱾硫??? ?좊텇 議곌컖留?議댁옱) ?꾩껜 ?먯쑝濡?convex hull 罹?
            if best_pts2d is None or best_pts2d.shape[0] < 3:
                if not cloud_pts2d:
                    continue
                try:
                    pts_cloud = np.vstack(cloud_pts2d)
                except Exception:
                    continue
                finite = np.all(np.isfinite(pts_cloud), axis=1)
                pts_cloud = pts_cloud[finite]
                if pts_cloud.shape[0] < 3:
                    continue
                if int(pts_cloud.shape[0]) > 5000:
                    step = int(np.ceil(float(pts_cloud.shape[0]) / 5000.0))
                    step = max(1, step)
                    pts_cloud = pts_cloud[::step]
                best_pts2d = np.asarray(pts_cloud, dtype=np.float64).reshape(-1, 2)
                tris = self._triangulate_convex_hull(best_pts2d)
                if tris is None:
                    continue
            else:
                tris = self._triangulate_ear_clip(best_pts2d, eps=1e-12)
                if tris is None:
                    tris = self._triangulate_convex_hull(best_pts2d)
                if tris is None:
                    continue

            pv = float(plane_value[key])
            verts = []
            try:
                if key.startswith("x"):
                    for i0, i1, i2 in tris:
                        for idx in (i0, i1, i2):
                            y, z = best_pts2d[int(idx)]
                            verts.append((pv, float(y), float(z)))
                else:
                    for i0, i1, i2 in tris:
                        for idx in (i0, i1, i2):
                            x, z = best_pts2d[int(idx)]
                            verts.append((float(x), pv, float(z)))
            except Exception:
                continue

            if verts:
                self.roi_cap_verts[key] = np.asarray(verts, dtype=np.float32)

    def draw_roi_section_plots(self):
        """ROI ?⑤㈃ 諛붾떏 諛곗튂 誘몃━蹂닿린 (x異??⑤㈃=+X 諛⑺뼢, y異??⑤㈃=+Y 諛⑺뼢)."""
        roi_sec = getattr(self, "roi_section_world", None)
        if not roi_sec:
            return

        lines = []
        try:
            lx = roi_sec.get("x", [])
            ly = roi_sec.get("y", [])
            if lx and len(lx) >= 2:
                lines.append(lx)
            if ly and len(ly) >= 2:
                lines.append(ly)
        except Exception:
            return

        if not lines:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glColor4f(0.05, 0.05, 0.05, 0.9)
        glLineWidth(2.0)
        z_plot = 0.12
        for pts in lines:
            glBegin(GL_LINE_STRIP)
            for p in pts:
                p0 = np.asarray(p, dtype=np.float64)
                glVertex3f(float(p0[0]), float(p0[1]), z_plot)
            glEnd()
        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    @staticmethod
    def _roi_edge_to_axis(edge: Any) -> str | None:
        s = str(edge).strip().lower()
        if s in ("left", "right"):
            return "x"
        if s in ("top", "bottom"):
            return "y"
        return None

    @staticmethod
    def _roi_edge_to_plane(edge: Any) -> str | None:
        s = str(edge).strip().lower()
        if s == "left":
            return "x1"
        if s == "right":
            return "x2"
        if s == "bottom":
            return "y1"
        if s == "top":
            return "y2"
        return None

    def _remember_roi_adjust_axis(self, edge: Any) -> None:
        axis = self._roi_edge_to_axis(edge)
        if axis in ("x", "y"):
            self._roi_last_adjust_axis = axis
        plane = self._roi_edge_to_plane(edge)
        if plane in ("x1", "x2", "y1", "y2"):
            self._roi_last_adjust_plane = plane

    @staticmethod
    def _pick_main_roi_contour(contours: list[np.ndarray] | None) -> np.ndarray | None:
        if not contours:
            return None
        best = None
        best_n = 0
        for c in contours:
            try:
                n = int(len(c))
            except Exception:
                n = 0
            if n > best_n:
                best_n = n
                best = c
        if best is None or best_n < 2:
            return None
        try:
            return np.asarray(best, dtype=np.float64).reshape(-1, 3)
        except Exception:
            return None

    def _layout_roi_contour_world(self, contour: np.ndarray, *, axis: str, margin: float = 5.0) -> list[list[float]]:
        obj = self.selected_obj
        if obj is None:
            return []
        try:
            b = np.asarray(obj.get_world_bounds(), dtype=np.float64)
            max_x, max_y = float(b[1][0]), float(b[1][1])
            min_x, min_y = float(b[0][0]), float(b[0][1])
        except Exception:
            return []

        pts = np.asarray(contour, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] < 2:
            return []

        out: list[list[float]] = []
        if str(axis).lower() == "x":
            y_min = float(np.min(pts[:, 1]))
            y_max = float(np.max(pts[:, 1]))
            z_min = float(np.min(pts[:, 2]))
            y_span = max(1e-6, y_max - y_min)
            scale_y = max(1e-6, (max_y - min_y)) / y_span
            # X異?ROI ?⑤㈃? ?쇱씤 ?⑤㈃ '?몃줈(?곗륫 諛곗튂)'? 媛숈? 洹쒖튃?쇰줈 諛곗튂
            # (z -> X, y-湲몄씠 -> Y[硫붿돩 湲몄씠??留욎떠 ?ㅼ???).
            base_x = max_x + float(margin)
            base_y = min_y
            for pt in pts:
                out.append(
                    [
                        base_x + (float(pt[2]) - z_min),
                        base_y + (float(pt[1]) - y_min) * scale_y,
                        0.0,
                    ]
                )
            return out

        if str(axis).lower() == "y":
            x_min = float(np.min(pts[:, 0]))
            z_min = float(np.min(pts[:, 2]))
            base_x = min_x
            base_y = max_y + float(margin)  # Y 諛⑺뼢 ?⑤㈃: +Y 諛⑺뼢
            for pt in pts:
                out.append(
                    [
                        base_x + (float(pt[0]) - x_min),  # X -> X
                        base_y + (float(pt[2]) - z_min),  # Z -> Y
                        0.0,
                    ]
                )
            return out

        return []

    def _compute_roi_cut_edges_sync(self, bounds: list[float] | None = None) -> dict[str, list[np.ndarray]]:
        """Enter ?뺤젙 吏곸쟾???꾩옱 ROI 寃쎄퀎 ?⑤㈃???숆린 怨꾩궛."""
        out: dict[str, list[np.ndarray]] = {"x1": [], "x2": [], "y1": [], "y2": []}

        obj = self.selected_obj
        if obj is None or getattr(obj, "mesh", None) is None:
            return out

        try:
            rb = [float(v) for v in ((bounds if bounds is not None else self.roi_bounds) or [])[:4]]
        except Exception:
            return out
        if len(rb) < 4:
            return out

        x1, x2, y1, y2 = rb
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        try:
            from scipy.spatial.transform import Rotation as R
        except Exception:
            return out

        try:
            roi_mesh_source = obj.to_trimesh()
        except Exception:
            roi_mesh_source = None
        if roi_mesh_source is None:
            roi_mesh_source = obj.mesh
        if isinstance(roi_mesh_source, MeshData):
            try:
                roi_mesh_source = roi_mesh_source.to_trimesh()
            except Exception:
                roi_mesh_source = None
        if roi_mesh_source is None:
            return out

        try:
            inv_rot = R.from_euler("XYZ", obj.rotation, degrees=True).inv().as_matrix()
            inv_scale = 1.0 / float(obj.scale) if float(obj.scale) != 0.0 else 1.0
            rot_mat = R.from_euler("XYZ", obj.rotation, degrees=True).as_matrix()
            trans = np.asarray(obj.translation, dtype=np.float64).reshape(3)
        except Exception:
            return out

        try:
            slicer = MeshSlicer(cast(Any, roi_mesh_source))
        except Exception:
            return out

        def slice_world_plane(world_origin: np.ndarray, world_normal: np.ndarray) -> list[np.ndarray]:
            world_origin = np.asarray(world_origin, dtype=np.float64).reshape(3)
            world_normal = np.asarray(world_normal, dtype=np.float64).reshape(3)
            local_origin = inv_scale * (inv_rot @ (world_origin - trans))
            local_normal = inv_rot @ world_normal
            contours_local = slicer.slice_with_plane(local_origin, local_normal)
            sliced: list[np.ndarray] = []
            for cnt in contours_local or []:
                try:
                    arr = np.asarray(cnt, dtype=np.float64).reshape(-1, 3)
                except Exception:
                    continue
                if arr.shape[0] < 2:
                    continue
                sliced.append((rot_mat @ (arr * float(obj.scale)).T).T + trans)
            return sliced

        try:
            edges_raw = {
                "x1": slice_world_plane(np.array([x1, 0.0, 0.0], dtype=np.float64), np.array([1.0, 0.0, 0.0], dtype=np.float64)),
                "x2": slice_world_plane(np.array([x2, 0.0, 0.0], dtype=np.float64), np.array([1.0, 0.0, 0.0], dtype=np.float64)),
                "y1": slice_world_plane(np.array([0.0, y1, 0.0], dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64)),
                "y2": slice_world_plane(np.array([0.0, y2, 0.0], dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64)),
            }
            clipped = _RoiCutEdgesThread._apply_roi_interaction_filters(
                edges_raw,
                x1=float(x1),
                x2=float(x2),
                y1=float(y1),
                y2=float(y2),
            )
        except Exception:
            return out

        for key in ("x1", "x2", "y1", "y2"):
            vals = clipped.get(key, []) if isinstance(clipped, dict) else []
            if not isinstance(vals, list):
                continue
            cleaned: list[np.ndarray] = []
            for cnt in vals:
                try:
                    arr = np.asarray(cnt, dtype=np.float64).reshape(-1, 3)
                except Exception:
                    continue
                if arr.shape[0] >= 2:
                    cleaned.append(arr)
            out[key] = cleaned
        return out

    def _build_roi_sections_for_commit(self) -> dict[str, list[list[float]]]:
        out: dict[str, list[list[float]]] = {"x": [], "y": []}
        edges = getattr(self, "roi_cut_edges", None) or {}
        if not isinstance(edges, dict):
            return out

        plane_hint = str(getattr(self, "_roi_commit_plane_hint", "") or "").strip().lower()
        axis_hint = str(getattr(self, "_roi_commit_axis_hint", "") or "").strip().lower()

        if plane_hint in ("x1", "x2", "y1", "y2"):
            main = self._pick_main_roi_contour(edges.get(plane_hint, []))
            if main is not None:
                axis = "x" if plane_hint.startswith("x") else "y"
                out[axis] = self._layout_roi_contour_world(main, axis=axis, margin=5.0)
                return out

        if axis_hint in ("x", "y"):
            keys = ("x1", "x2") if axis_hint == "x" else ("y1", "y2")
            all_cnt: list[np.ndarray] = []
            for kk in keys:
                vals = edges.get(kk, [])
                if isinstance(vals, list):
                    all_cnt.extend(vals)
            main = self._pick_main_roi_contour(all_cnt)
            if main is not None:
                out[axis_hint] = self._layout_roi_contour_world(main, axis=axis_hint, margin=5.0)
                return out

        # Fallback: choose the most informative contour per axis.
        main_x = self._pick_main_roi_contour((edges.get("x1", []) or []) + (edges.get("x2", []) or []))
        main_y = self._pick_main_roi_contour((edges.get("y1", []) or []) + (edges.get("y2", []) or []))
        if main_x is not None:
            out["x"] = self._layout_roi_contour_world(main_x, axis="x", margin=5.0)
        if main_y is not None:
            out["y"] = self._layout_roi_contour_world(main_y, axis="y", margin=5.0)
        return out

    def _roi_live_delay_ms(self) -> int:
        """硫붿돩 ?ш린??留욎떠 ROI ?ㅼ떆媛??낅뜲?댄듃 吏?곗쓣 ?꾪솕."""
        try:
            base = int(getattr(self, "_roi_live_update_delay_ms", 220) or 220)
        except Exception:
            base = 220
        base = max(60, base)

        try:
            obj = self.selected_obj
            n_faces = int(getattr(getattr(obj, "mesh", None), "n_faces", 0) or 0)
        except Exception:
            n_faces = 0

        if n_faces >= 3_000_000:
            return max(base, 420)
        if n_faces >= 1_000_000:
            return max(base, 340)
        if n_faces >= 300_000:
            return max(base, 280)
        if n_faces >= 100_000:
            return max(base, 230)
        return int(base)

    def schedule_roi_edges_update(self, delay_ms: int = 150):
        if not getattr(self, "roi_enabled", False):
            return
        self._roi_edges_pending_bounds = [float(v) for v in self.roi_bounds]
        self._roi_edges_timer.start(max(0, int(delay_ms)))

    def _request_roi_edges_compute(self):
        if not getattr(self, "roi_enabled", False):
            return

        obj = self.selected_obj
        if obj is None or obj.mesh is None:
            return

        thread = getattr(self, "_roi_edges_thread", None)
        if thread is not None and thread.isRunning():
            return

        bounds = self._roi_edges_pending_bounds
        self._roi_edges_pending_bounds = None
        if bounds is None:
            bounds = [float(v) for v in self.roi_bounds]

        try:
            roi_mesh_source = obj.to_trimesh()
        except Exception:
            roi_mesh_source = None
        if roi_mesh_source is None:
            roi_mesh_source = obj.mesh

        self._roi_edges_thread = _RoiCutEdgesThread(
            roi_mesh_source,
            translation=obj.translation.copy(),
            rotation_deg=obj.rotation.copy(),
            scale=float(obj.scale),
            roi_bounds=bounds,
        )
        self._roi_edges_thread.computed.connect(self._on_roi_edges_computed)
        self._roi_edges_thread.failed.connect(self._on_roi_edges_failed)
        self._roi_edges_thread.finished.connect(self._on_roi_edges_finished)
        self._roi_edges_thread.start()

    def _on_roi_edges_finished(self):
        thread = getattr(self, "_roi_edges_thread", None)
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                _log_ignored_exception()
        self._roi_edges_thread = None

        # ?쒕옒洹?以?理쒖떊 bounds媛 ?湲?以묒씠硫??ㅼ떆 怨꾩궛
        if getattr(self, "roi_enabled", False) and self._roi_edges_pending_bounds is not None:
            self._roi_edges_timer.start(1)

    def _on_roi_edges_computed(self, edges: object):
        if not isinstance(edges, dict):
            return

        cleaned: dict[str, list[np.ndarray]] = {"x1": [], "x2": [], "y1": [], "y2": []}
        for key in ("x1", "x2", "y1", "y2"):
            val = edges.get(key, [])
            if isinstance(val, list):
                cleaned[key] = cast(list[np.ndarray], val)

        self.roi_cut_edges = cleaned

        # ROI 罹??섎┛ 硫?梨꾩슦湲? 媛깆떊
        try:
            dragging_roi = bool(getattr(self, "_roi_move_dragging", False) or getattr(self, "roi_rect_dragging", False))
            active_edge = str(getattr(self, "active_roi_edge", "") or "").strip().lower()
            if active_edge and active_edge != "move":
                dragging_roi = True
            if bool(getattr(self, "roi_caps_enabled", False)):
                if not dragging_roi:
                    self._rebuild_roi_caps()
            else:
                self.roi_cap_verts = {"x1": None, "x2": None, "y1": None, "y2": None}
        except Exception:
            _log_ignored_exception()

        # ROI媛 '?뉗븘議뚯쓣 ?? ?⑤㈃??諛붾떏??諛곗튂 (?뚯쟾?댁꽌 遊먮룄 ?⑤㈃ ?뺤씤 媛??
        self.roi_section_world = {"x": [], "y": []}
        try:
            obj = self.selected_obj
            if obj is not None:
                b = np.asarray(obj.get_world_bounds(), dtype=np.float64)
                max_x, max_y = float(b[1][0]), float(b[1][1])
                margin = 5.0

                x1, x2, y1, y2 = [float(v) for v in self.roi_bounds]
                thin_th = 1.0  # cm: ?대낫???뉗쑝硫?"?⑤㈃"?쇰줈 媛꾩＜

                def pick_main(contours):
                    if not contours:
                        return None
                    best = None
                    best_n = 0
                    for c in contours:
                        try:
                            n = int(len(c))
                        except Exception:
                            n = 0
                        if n > best_n:
                            best_n = n
                            best = c
                    return best

                # X 諛⑺뼢 ?⑤㈃ (醫뚯슦 ??씠 留ㅼ슦 ?뉗쓣 ?? -> +X 諛⑺뼢??諛곗튂 (Y-Z)
                if abs(x2 - x1) <= thin_th:
                    main = pick_main(self.roi_cut_edges.get("x1", []) or self.roi_cut_edges.get("x2", []))
                    if main is not None and len(main) >= 2:
                        main = np.asarray(main, dtype=np.float64)
                        y_min = float(np.min(main[:, 1]))
                        y_max = float(np.max(main[:, 1]))
                        z_min = float(np.min(main[:, 2]))
                        y_span = max(1e-6, y_max - y_min)
                        scale_y = max(1e-6, (max_y - float(b[0][1]))) / y_span
                        base_x = max_x + margin
                        base_y = float(b[0][1])
                        pts = []
                        for pt in main:
                            # ?쇱씤 ?⑤㈃ ?몃줈 諛곗튂? ?숈씪: z -> X, y-湲몄씠 -> Y(硫붿돩 湲몄씠 ?ㅼ???
                            pts.append(
                                [
                                    base_x + (float(pt[2]) - z_min),
                                    base_y + (float(pt[1]) - y_min) * scale_y,
                                    0.0,
                                ]
                            )
                        self.roi_section_world["x"] = pts

                # Y 諛⑺뼢 ?⑤㈃ (?곹븯 ??씠 留ㅼ슦 ?뉗쓣 ?? -> +Y 諛⑺뼢??諛곗튂 (X-Z)
                if abs(y2 - y1) <= thin_th:
                    main = pick_main(self.roi_cut_edges.get("y1", []) or self.roi_cut_edges.get("y2", []))
                    if main is not None and len(main) >= 2:
                        main = np.asarray(main, dtype=np.float64)
                        x_min = float(np.min(main[:, 0]))
                        z_min = float(np.min(main[:, 2]))
                        base_x = float(b[0][0])
                        base_y = max_y + margin
                        pts = []
                        for pt in main:
                            # XZ ?⑤㈃??"媛濡?X, ?몃줈=Z"濡?諛곗튂 (?믪씠媛 ?꾨줈 ?ν븯?꾨줉)
                            pts.append(
                                [
                                    base_x + (float(pt[0]) - x_min),
                                    base_y + (float(pt[2]) - z_min),
                                    0.0,
                                ]
                            )
                        self.roi_section_world["y"] = pts
        except Exception:
            _log_ignored_exception()
        self.update()

    def _on_roi_edges_failed(self, message: str):
        # print(f"ROI edge compute failed: {message}")
        pass

    def extract_roi_silhouette(self):
        """吏?뺣맂 ROI ?곸뿭??硫붿돩 ?멸낸(?ㅻ（?? 異붿텧"""
        if not self.selected_obj or self.selected_obj.mesh is None:
            return
            
        obj = self.selected_obj
        x1, x2, y1, y2 = self.roi_bounds
        
        from scipy.spatial.transform import Rotation as R
        rot_mat = R.from_euler('XYZ', obj.rotation, degrees=True).as_matrix()
        world_v = (rot_mat @ (obj.mesh.vertices * obj.scale).T).T + obj.translation
        
        # 2. ROI ?곸뿭 ?댁쓽 ?먮뱾 ?꾪꽣留?
        mask = (world_v[:, 0] >= x1) & (world_v[:, 0] <= x2) & \
               (world_v[:, 1] >= y1) & (world_v[:, 1] <= y2)
        
        inside_v = world_v[mask]
        if len(inside_v) < 3:
            return
            
        # 3. 2D ?ъ쁺 (XY ?됰㈃) 諛?Convex Hull ?먮뒗 Alpha Shape濡??멸낸 異붿텧
        # ?ш린?쒕뒗 媛꾨떒??Convex Hull ?ъ슜 (異뷀썑 蹂듭옟???뺤긽? Alpha Shape ?꾩슂)
        from scipy.spatial import ConvexHull
        points_2d = inside_v[:, :2]
        try:
            hull = ConvexHull(points_2d)
            silhouette = points_2d[hull.vertices]
            # ?ㅼ떆 2D 由ъ뒪???뺥깭濡?諛섑솚
            self.roiSilhouetteExtracted.emit(silhouette.tolist())
        except Exception:
            _log_ignored_exception()

    
    def draw_axes(self):
        """Draw world XYZ axes (Z-up)."""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        try:
            cam_dist = float(getattr(self.camera, "distance", 500.0) or 500.0)
        except Exception:
            cam_dist = 500.0
        cam_dist = max(1.0, cam_dist)

        try:
            elev = abs(float(getattr(self.camera, "elevation", 30.0) or 30.0))
        except Exception:
            elev = 30.0
        if elev < 5.0:
            horizon_factor = 22.0
        elif elev < 10.0:
            horizon_factor = 14.0
        elif elev < 20.0:
            horizon_factor = 9.0
        else:
            horizon_factor = 5.0

        world_extent = max(200.0, cam_dist * horizon_factor)
        axis_length = max(120.0, world_extent * 0.35)
        axis_length = min(axis_length, 200_000.0)

        glPushMatrix()
        glLineWidth(2.8)
        glBegin(GL_LINES)

        glColor3f(0.95, 0.2, 0.2)
        glVertex3f(float(-axis_length), 0.0, 0.0)
        glVertex3f(float(axis_length), 0.0, 0.0)

        glColor3f(0.2, 0.85, 0.2)
        glVertex3f(0.0, float(-axis_length), 0.0)
        glVertex3f(0.0, float(axis_length), 0.0)

        glColor3f(0.2, 0.2, 0.95)
        glVertex3f(0.0, 0.0, float(-axis_length))
        glVertex3f(0.0, 0.0, float(axis_length))

        glEnd()
        glPopMatrix()

        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def _draw_axis_label_marker(self, x, y, z, size, axis):
        """X/Y 異??쇰꺼??媛꾨떒??湲고븯?숈쟻 ?뺥깭濡?洹몃━湲?(XY ?됰㈃???쒖떆)"""
        glLineWidth(2.5)
        glBegin(GL_LINES)
        
        if axis == 'X':
            # X 紐⑥뼇 (YZ ?됰㈃???쒖떆, X異??앹뿉??
            glVertex3f(x, -size, z + size)
            glVertex3f(x, size, z - size)
            glVertex3f(x, -size, z - size)
            glVertex3f(x, size, z + size)
        elif axis == 'Y':
            # Y 紐⑥뼇 (XZ ?됰㈃???쒖떆, Y異??앹뿉??
            glVertex3f(-size, y, z + size)
            glVertex3f(0, y, z)
            glVertex3f(size, y, z + size)
            glVertex3f(0, y, z)
            glVertex3f(0, y, z)
            glVertex3f(0, y, z - size)
        
        glEnd()
    
    def _draw_axis_label_marker_z(self, x, y, z, size, axis):
        """Z 異??쇰꺼??媛꾨떒??湲고븯?숈쟻 ?뺥깭濡?洹몃━湲?(XY ?됰㈃???쒖떆, Z異??앹뿉??"""
        glLineWidth(2.5)
        glBegin(GL_LINES)
        
        # Z 紐⑥뼇 (XY ?됰㈃???쒖떆)
        glVertex3f(x - size, y + size, z)
        glVertex3f(x + size, y + size, z)
        glVertex3f(x + size, y + size, z)
        glVertex3f(x - size, y - size, z)
        glVertex3f(x - size, y - size, z)
        glVertex3f(x + size, y - size, z)
        
        glEnd()

        
    def draw_orientation_hud(self):
        """?곗륫 ?섎떒???묒? 諛⑺뼢 媛?대뱶(HUD) 洹몃━湲?"""
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # 1. 酉고룷???뺣낫 ???
        viewport = glGetIntegerv(GL_VIEWPORT)
        w, h = viewport[2], viewport[3]
        
        # 2. ?꾩옱 酉??됰젹 媛?몄삤湲?(?뚯쟾 ?뺣낫 異붿텧??
        view_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        
        # 3. HUD???꾩슜 酉고룷???ㅼ젙 (?곗륫 ?섎떒 80x80)
        hud_size = 100
        margin = 10
        glViewport(w - hud_size - margin, margin, hud_size, hud_size)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(-1.5, 1.5, -1.5, 1.5, -2, 2)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # 4. 酉??됰젹?먯꽌 ?뚯쟾留??곸슜 (?대룞 ?쒓굅)
        try:
            rot_matrix = np.array(view_matrix, dtype=np.float64).reshape(4, 4)
            # ?대룞 ?깅텇 ?쒓굅 (Column-major 湲곗? 12, 13, 14踰덉㎏ ?붿냼媛 4??1,2,3??
            # numpy array??寃쎌슦 r, c ?몃뜳???ъ슜
            rot_matrix[3, 0] = 0.0
            rot_matrix[3, 1] = 0.0
            rot_matrix[3, 2] = 0.0
            glLoadMatrixd(rot_matrix)
        except Exception:
            # ?됰젹 泥섎━???ㅽ뙣??寃쎌슦 HUD ?뚯쟾 ?곸슜 ?앸왂 (理쒖냼???щ옒?쒕뒗 諛⑹?)
            pass
        
        # 5. 異?洹몃━湲?(X:Red, Y:Green, Z:Blue)
        glLineWidth(3.0)
        glBegin(GL_LINES)
        # X: Red
        glColor3f(1.0, 0.2, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(1, 0, 0)
        # Y: Green
        glColor3f(0.2, 1.0, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 1, 0)
        # Z: Blue
        glColor3f(0.2, 0.2, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 1)
        glEnd()
        
        # 5.5 異??쇰꺼 (X, Y, Z) - 媛?異??앹뿉 ?쒖떆
        label_size = 0.12
        
        # X ?쇰꺼
        glColor3f(1.0, 0.2, 0.2)
        glBegin(GL_LINES)
        glVertex3f(1.1 - label_size, label_size, 0)
        glVertex3f(1.1 + label_size, -label_size, 0)
        glVertex3f(1.1 - label_size, -label_size, 0)
        glVertex3f(1.1 + label_size, label_size, 0)
        glEnd()
        
        # Y ?쇰꺼
        glColor3f(0.2, 1.0, 0.2)
        glBegin(GL_LINES)
        glVertex3f(-label_size, 1.1 + label_size, 0)
        glVertex3f(0, 1.1, 0)
        glVertex3f(label_size, 1.1 + label_size, 0)
        glVertex3f(0, 1.1, 0)
        glVertex3f(0, 1.1, 0)
        glVertex3f(0, 1.1 - label_size, 0)
        glEnd()
        
        # Z ?쇰꺼
        glColor3f(0.2, 0.2, 1.0)
        glBegin(GL_LINES)
        glVertex3f(-label_size, label_size, 1.1)
        glVertex3f(label_size, label_size, 1.1)
        glVertex3f(label_size, label_size, 1.1)
        glVertex3f(-label_size, -label_size, 1.1)
        glVertex3f(-label_size, -label_size, 1.1)
        glVertex3f(label_size, -label_size, 1.1)
        glEnd()
        
        # 6. 蹂듦뎄
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glViewport(0, 0, w, h)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_surface_runtime_hud(self):
        """?쒕㈃ 遺꾨━(assist/overlay) ?고???怨꾩륫媛믪쓣 ?붾㈃ 醫뚯륫 ?곷떒???쒖떆."""
        if not bool(getattr(self, "surface_runtime_hud_enabled", True)):
            return
        obj = self.selected_obj
        if obj is None:
            return

        assist = getattr(obj, "surface_assist_runtime", None)
        if not isinstance(assist, dict):
            assist = {}
        overlay = getattr(self, "_surface_overlay_last_stats", None)
        if not isinstance(overlay, dict):
            overlay = {}

        has_assist = bool(assist)
        has_overlay = bool(overlay)
        if not has_assist and not has_overlay:
            return

        try:
            outer_n = int(len(getattr(obj, "outer_face_indices", set()) or set()))
            inner_n = int(len(getattr(obj, "inner_face_indices", set()) or set()))
            migu_n = int(len(getattr(obj, "migu_face_indices", set()) or set()))
            unresolved_n = int(len(getattr(obj, "surface_assist_unresolved_face_indices", set()) or set()))
        except Exception:
            outer_n = inner_n = migu_n = unresolved_n = 0

        def _fms(v: Any) -> str:
            try:
                fv = float(v)
            except Exception:
                return "n/a"
            if not np.isfinite(fv):
                return "n/a"
            return f"{fv:.1f}ms"

        def _fnum(v: Any) -> float:
            try:
                fv = float(v)
            except Exception:
                return float("nan")
            if not np.isfinite(fv):
                return float("nan")
            return fv

        def _latency_color(ms: float, *, warn_ms: float, crit_ms: float) -> QColor:
            if not np.isfinite(ms):
                return QColor(236, 240, 245)
            if ms >= float(crit_ms):
                return QColor(255, 132, 132)  # critical
            if ms >= float(warn_ms):
                return QColor(255, 196, 120)  # warning
            return QColor(140, 232, 170)      # good

        assist_total_ms = _fnum(assist.get("total_ms")) if has_assist else float("nan")
        overlay_total_ms = _fnum(overlay.get("total_ms")) if has_overlay else float("nan")

        lines: list[tuple[str, QColor]] = []
        lines.append(("Surface Runtime", QColor(132, 211, 255)))
        if has_assist:
            lines.append((
                "assist total/core/apply: "
                f"{_fms(assist.get('total_ms'))} / {_fms(assist.get('core_ms'))} / {_fms(assist.get('apply_ms'))}"
                ,
                _latency_color(assist_total_ms, warn_ms=7000.0, crit_ms=9000.0),
            ))
            lines.append((
                f"assist mode: {assist.get('mode_txt', '?')} / {assist.get('method', '?')} / {assist.get('assist_mode', '?')} / {assist.get('mapping', '?')}"
                ,
                QColor(236, 240, 245),
            ))
        if has_overlay:
            overlay_age_ms = float("nan")
            try:
                ts = float(overlay.get("perf_ts", 0.0) or 0.0)
                if ts > 0.0:
                    overlay_age_ms = max(0.0, (time.perf_counter() - ts) * 1000.0)
            except Exception:
                overlay_age_ms = float("nan")
            lines.append((
                "overlay total/index/draw: "
                f"{_fms(overlay.get('total_ms'))} / {_fms(overlay.get('index_ms'))} / {_fms(overlay.get('draw_ms'))}"
                ,
                _latency_color(overlay_total_ms, warn_ms=180.0, crit_ms=300.0),
            ))
            cache_build = int(overlay.get("cache_build", 0) or 0)
            cache_hit = int(overlay.get("cache_hit", 0) or 0)
            cache_color = QColor(236, 240, 245)
            if cache_build > 0 and cache_hit == 0:
                cache_color = QColor(255, 196, 120)
            elif cache_hit > 0 and cache_build == 0:
                cache_color = QColor(140, 232, 170)
            lines.append((
                f"overlay cache build/hit: {int(overlay.get('cache_build', 0) or 0)} / {int(overlay.get('cache_hit', 0) or 0)}"
                ,
                cache_color,
            ))
            lines.append((
                f"overlay age: {_fms(overlay_age_ms)}"
                ,
                QColor(195, 200, 208),
            ))
        lines.append((
            f"faces O/I/M/U: {outer_n:,} / {inner_n:,} / {migu_n:,} / {unresolved_n:,}",
            QColor(236, 240, 245),
        ))

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            font = QFont("Consolas")
            font.setPointSize(9)
            painter.setFont(font)
            fm = painter.fontMetrics()
            line_h = int(max(14, fm.height()))
            text_w = int(max((fm.horizontalAdvance(s) for (s, _c) in lines), default=120))
            pad_x = 8
            pad_y = 6
            box_w = int(text_w + pad_x * 2)
            box_h = int(line_h * len(lines) + pad_y * 2)
            x = 12
            y = 12
            painter.fillRect(x, y, box_w, box_h, QColor(8, 14, 22, 176))
            yy = y + pad_y + line_h - 2
            for s, color in lines:
                painter.setPen(color)
                painter.drawText(x + pad_x, yy, s)
                yy += line_h
        finally:
            painter.end()

    
    def draw_scene_object(
        self,
        obj: SceneObject,
        is_selected: bool = False,
        *,
        alpha: float = 1.0,
        depth_write: bool = True,
    ):
        """媛쒕퀎 硫붿돩 媛앹껜 ?뚮뜑留?(?곹깭 ?꾩닔 諛⑹? ?섑띁)."""
        glPushMatrix()
        try:
            self._draw_scene_object_impl(
                obj,
                is_selected=is_selected,
                alpha=alpha,
                depth_write=depth_write,
            )
        except Exception:
            _log_ignored_exception("Failed to draw scene object", level=logging.WARNING)
        finally:
            # ??媛앹껜 ?뚮뜑 ?ㅽ뙣媛 ?ㅼ쓬 ?꾨젅??媛앹껜 源딆씠 ?곹깭瑜??ㅼ뿼?쒗궎吏 ?딅룄濡?媛뺤젣 蹂듦뎄.
            try:
                glBindBuffer(GL_ARRAY_BUFFER, 0)
            except Exception:
                pass
            try:
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            except Exception:
                pass
            try:
                glDisableClientState(GL_NORMAL_ARRAY)
            except Exception:
                pass
            try:
                glDisableClientState(GL_VERTEX_ARRAY)
            except Exception:
                pass
            try:
                glDisable(GL_POLYGON_OFFSET_FILL)
            except Exception:
                pass
            try:
                glDisable(GL_CLIP_PLANE0)
            except Exception:
                pass
            try:
                glDisable(GL_CLIP_PLANE1)
            except Exception:
                pass
            try:
                glDisable(GL_CLIP_PLANE2)
            except Exception:
                pass
            try:
                glDisable(GL_CLIP_PLANE3)
            except Exception:
                pass
            try:
                glDisable(GL_CLIP_PLANE4)
            except Exception:
                pass
            try:
                glDisable(GL_CULL_FACE)
            except Exception:
                pass
            try:
                glEnable(GL_DEPTH_TEST)
            except Exception:
                pass
            try:
                glDepthMask(GL_TRUE)
            except Exception:
                pass
            try:
                glPopMatrix()
            except Exception:
                pass

    def _draw_scene_object_impl(
        self,
        obj: SceneObject,
        is_selected: bool = False,
        *,
        alpha: float = 1.0,
        depth_write: bool = True,
    ):
        """媛쒕퀎 硫붿돩 媛앹껜 ?뚮뜑留?"""

        # 蹂???곸슜
        try:
            tr = np.asarray(getattr(obj, "translation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            if tr.size < 3 or (not np.isfinite(tr[:3]).all()):
                tr = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            else:
                tr = tr[:3]
        except Exception:
            tr = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        try:
            rot = np.asarray(getattr(obj, "rotation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            if rot.size < 3 or (not np.isfinite(rot[:3]).all()):
                rot = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            else:
                rot = rot[:3]
        except Exception:
            rot = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        try:
            sc = float(getattr(obj, "scale", 1.0) or 1.0)
        except Exception:
            sc = 1.0
        if (not np.isfinite(sc)) or abs(sc) < 1e-9:
            sc = 1.0

        try:
            obj.translation = np.asarray(tr, dtype=np.float64)
            obj.rotation = np.asarray(rot, dtype=np.float64)
            obj.scale = float(sc)
        except Exception:
            _log_ignored_exception()

        glTranslatef(float(tr[0]), float(tr[1]), float(tr[2]))
        glRotatef(float(rot[0]), 1, 0, 0)
        glRotatef(float(rot[1]), 0, 1, 0)
        glRotatef(float(rot[2]), 0, 0, 1)
        glScalef(float(sc), float(sc), float(sc))

        alpha_f = float(alpha)
        if not np.isfinite(alpha_f):
            alpha_f = 1.0
        alpha_f = max(0.0, min(alpha_f, 1.0))
        solid_shell = bool(getattr(self, "solid_shell_render", True)) and alpha_f >= 0.999

        if not depth_write:
            glDepthMask(GL_FALSE)

        if solid_shell:
            # 湲곕낯 ?쒖떆?먯꽌??back-face瑜??④꺼 "?띿씠 苑?李? ?뺥깭濡?蹂댁씠寃??⑸땲??
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
        else:
            glDisable(GL_CULL_FACE)
        
        # 硫붿돩 ?ъ쭏 諛?諛앷린 理쒖쟻??(愿묓깮 異붽?濡?援닿끝 媛뺤“)
        if not self.flat_shading:
            glEnable(GL_LIGHTING)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 32.0)
        else:
            glDisable(GL_LIGHTING)
            glDisable(GL_COLOR_MATERIAL)
        
        # 硫붿돩 ?됱긽
        if is_selected:
            col = (0.85, 0.85, 0.95)  # ?덈Т ?섏뼏吏 ?딄쾶 ?쎄컙 ?ㅻ떎??
        else:
            col = tuple(float(c) for c in (obj.color or [0.72, 0.72, 0.78])[:3])

        if alpha_f < 1.0:
            glColor4f(float(col[0]), float(col[1]), float(col[2]), float(alpha_f))
        else:
            # glColor3f??alpha瑜?嫄대뱶由ъ? ?딆쑝誘濡??댁쟾 draw??alpha媛 ?⑥쓣 ???덉뒿?덈떎.
            # 遺덊닾紐?硫붿돩??alpha=1.0??紐낆떆???섎룄移??딆? ?대? 鍮꾩묠??留됱뒿?덈떎.
            glColor4f(float(col[0]), float(col[1]), float(col[2]), 1.0)
            
        # 釉뚮윭?쒕줈 ?좏깮??硫??섏씠?쇱씠??(?꾩떆 ?ㅻ쾭?덉씠)
        if is_selected and self.picking_mode == 'floor_brush' and self.brush_selected_faces:
            glPushMatrix()
            glDisable(GL_LIGHTING)
            # 硫붿돩蹂대떎 ?꾩＜ ?쎄컙 ?욎뿉 洹몃━湲?(Z-fight 諛⑹?)
            glPolygonOffset(-1.0, -1.0)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glColor3f(1.0, 0.2, 0.2)
            glBegin(GL_TRIANGLES)
            for face_idx in self.brush_selected_faces:
                f = obj.mesh.faces[face_idx]
                for v_idx in f:
                    glVertex3fv(obj.mesh.vertices[v_idx])
            glEnd()
            glDisable(GL_POLYGON_OFFSET_FILL)
            glEnable(GL_LIGHTING)
            glPopMatrix()

        # ?좏깮??硫??섏씠?쇱씠??(SelectionPanel)
        if is_selected and self.picking_mode in {'select_face', 'select_brush'} and obj.selected_faces:
            glPushMatrix()
            glDisable(GL_LIGHTING)
            glPolygonOffset(-1.0, -1.0)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glColor3f(1.0, 0.8, 0.0)
            glBegin(GL_TRIANGLES)
            for face_idx in obj.selected_faces:
                f = obj.mesh.faces[int(face_idx)]
                for v_idx in f:
                    glVertex3fv(obj.mesh.vertices[v_idx])
            glEnd()
            glDisable(GL_POLYGON_OFFSET_FILL)
            glEnable(GL_LIGHTING)
            glPopMatrix()

        try:
            vbo_id = int(getattr(obj, "vbo_id", 0) or 0)
        except Exception:
            vbo_id = 0
        can_draw_vbo = vbo_id > 0 and int(getattr(obj, "vertex_count", 0) or 0) > 0
        if can_draw_vbo:
            # VBO 諛⑹떇 ?뚮뜑留?
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)

            glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
            glVertexPointer(3, GL_FLOAT, 24, ctypes.c_void_p(0))
            glNormalPointer(GL_FLOAT, 24, ctypes.c_void_p(12))

            # 1) 湲곕낯 ?됱긽 ?뚮뜑留?
            glDrawArrays(GL_TRIANGLES, 0, obj.vertex_count)

            # 2) 諛붾떏 愿??z<0) ?곸뿭??珥덈줉?됱쑝濡???뼱?곌린(?대━???됰㈃ ?댁슜, CPU ?ㅼ틪 ?놁쓬)
            if self.floor_penetration_highlight:
                try:
                    wb = obj.get_world_bounds()
                    if float(wb[0][2]) < 0.0:
                        glEnable(GL_CLIP_PLANE0)
                        glDepthMask(GL_FALSE)
                        glEnable(GL_POLYGON_OFFSET_FILL)
                        glPolygonOffset(-1.0, -1.0)
                        glColor3f(0.0, 1.0, 0.2)
                        glDrawArrays(GL_TRIANGLES, 0, obj.vertex_count)
                        glDisable(GL_POLYGON_OFFSET_FILL)
                        glDepthMask(GL_TRUE if depth_write else GL_FALSE)
                        glDisable(GL_CLIP_PLANE0)
                except Exception:
                    _log_ignored_exception()

            # Surface assignment overlays (outer/inner/migu + assist-unresolved).
            # Draw after the base mesh so the shading stays visible; use indexed draw for performance.
            if is_selected and bool(getattr(self, "show_surface_assignment_overlay", True)):
                try:
                    overlay_t0 = time.perf_counter()
                    max_faces_vbo = int(getattr(self, "_surface_overlay_max_faces_vbo", 3000000))

                    outer_set = self._get_surface_target_set(obj, "outer")
                    inner_set = self._get_surface_target_set(obj, "inner")
                    migu_set = self._get_surface_target_set(obj, "migu")
                    unresolved_set = set(
                        int(x)
                        for x in (getattr(obj, "surface_assist_unresolved_face_indices", set()) or set())
                    )
                    if unresolved_set:
                        # Safety: unresolved should not overlap with labeled sets.
                        unresolved_set.difference_update(outer_set)
                        unresolved_set.difference_update(inner_set)
                        unresolved_set.difference_update(migu_set)

                    paint_target = None
                    if self.picking_mode in {"paint_surface_face", "paint_surface_brush", "paint_surface_area", "paint_surface_magnetic"}:
                        paint_target = str(getattr(self, "_surface_paint_target", "outer")).strip().lower()
                        if paint_target not in {"outer", "inner", "migu"}:
                            paint_target = "outer"

                    # High-contrast colors so users can clearly distinguish auto-separated surfaces.
                    # outer: blue, inner: orange, migu: green
                    outer_c = (0.20, 0.55, 1.00, 0.36)
                    inner_c = (1.00, 0.55, 0.15, 0.36)
                    migu_c = (0.20, 0.85, 0.35, 0.32)
                    unresolved_c = (1.00, 0.92, 0.22, 0.30)
                    if paint_target == "outer":
                        outer_c = (outer_c[0], outer_c[1], outer_c[2], 0.50)
                    elif paint_target == "inner":
                        inner_c = (inner_c[0], inner_c[1], inner_c[2], 0.50)
                    elif paint_target == "migu":
                        migu_c = (migu_c[0], migu_c[1], migu_c[2], 0.46)

                    cache_hit = 0
                    cache_build = 0

                    def get_indices(key: str, face_set: set[int]) -> np.ndarray | None:
                        nonlocal cache_hit, cache_build
                        if not face_set:
                            return None
                        n = int(len(face_set))
                        if n <= 0 or n > max_faces_vbo:
                            return None

                        try:
                            ver = int(getattr(obj, "_surface_assignment_version", 0) or 0)
                        except Exception:
                            ver = 0

                        cache = getattr(obj, "_surface_overlay_index_cache", None)
                        try:
                            cache_ver = int(getattr(obj, "_surface_overlay_index_cache_version", -1))
                        except Exception:
                            cache_ver = -1
                        if not isinstance(cache, dict):
                            cache = {}
                            try:
                                obj._surface_overlay_index_cache = cache
                            except Exception:
                                _log_ignored_exception()
                        if cache_ver != ver:
                            try:
                                cache.clear()
                            except Exception:
                                pass
                            try:
                                obj._surface_overlay_index_cache_version = ver
                            except Exception:
                                _log_ignored_exception()

                        arr = cache.get(key)
                        if isinstance(arr, np.ndarray) and arr.dtype == np.uint32 and arr.ndim == 1 and arr.size == n * 3:
                            cache_hit += 1
                            return arr

                        face_idx = np.fromiter(face_set, dtype=np.int32, count=n)
                        base = face_idx.astype(np.uint32, copy=False) * np.uint32(3)
                        idx_arr = np.empty((n * 3,), dtype=np.uint32)
                        idx_arr[0::3] = base
                        idx_arr[1::3] = base + np.uint32(1)
                        idx_arr[2::3] = base + np.uint32(2)
                        cache[key] = idx_arr
                        cache_build += 1
                        return idx_arr

                    idx_t0 = time.perf_counter()
                    idx_migu = get_indices("migu", migu_set)
                    idx_inner = get_indices("inner", inner_set)
                    idx_outer = get_indices("outer", outer_set)
                    idx_unresolved = get_indices("unresolved", unresolved_set)
                    index_ms = (time.perf_counter() - idx_t0) * 1000.0
                    draw_ms = 0.0
                    if idx_migu is not None or idx_inner is not None or idx_outer is not None or idx_unresolved is not None:
                        draw_t0 = time.perf_counter()
                        glDisable(GL_LIGHTING)
                        glEnable(GL_BLEND)
                        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                        glDepthMask(GL_FALSE)
                        glEnable(GL_POLYGON_OFFSET_FILL)
                        glPolygonOffset(-1.0, -1.0)
                        try:
                            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
                        except Exception:
                            pass

                        # Draw order: outer -> inner -> migu -> unresolved
                        # (If sets overlap due to user edits, this keeps focused overlays on top.)
                        if idx_outer is not None:
                            glColor4f(*outer_c)
                            glDrawElements(GL_TRIANGLES, int(idx_outer.size), GL_UNSIGNED_INT, idx_outer)
                        if idx_inner is not None:
                            glColor4f(*inner_c)
                            glDrawElements(GL_TRIANGLES, int(idx_inner.size), GL_UNSIGNED_INT, idx_inner)
                        if idx_migu is not None:
                            glColor4f(*migu_c)
                            glDrawElements(GL_TRIANGLES, int(idx_migu.size), GL_UNSIGNED_INT, idx_migu)
                        if idx_unresolved is not None:
                            glColor4f(*unresolved_c)
                            glDrawElements(GL_TRIANGLES, int(idx_unresolved.size), GL_UNSIGNED_INT, idx_unresolved)

                        glDisable(GL_POLYGON_OFFSET_FILL)
                        glDepthMask(GL_TRUE if depth_write else GL_FALSE)
                        glDisable(GL_BLEND)
                        glEnable(GL_LIGHTING)
                        draw_ms = (time.perf_counter() - draw_t0) * 1000.0
                    total_ms = (time.perf_counter() - overlay_t0) * 1000.0
                    try:
                        self._surface_overlay_last_stats = {
                            "total_ms": float(total_ms),
                            "index_ms": float(index_ms),
                            "draw_ms": float(draw_ms),
                            "cache_hit": int(cache_hit),
                            "cache_build": int(cache_build),
                            "outer_count": int(len(outer_set)),
                            "inner_count": int(len(inner_set)),
                            "migu_count": int(len(migu_set)),
                            "unresolved_count": int(len(unresolved_set)),
                            "perf_ts": float(time.perf_counter()),
                        }
                    except Exception:
                        _log_ignored_exception()
                except Exception:
                    _log_ignored_exception()

            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
        else:
            # Fallback path: keep mesh visible even when VBO creation fails.
            try:
                faces = np.asarray(obj.mesh.faces, dtype=np.int32)
                vertices = np.asarray(obj.mesh.vertices, dtype=np.float32)
                if faces.ndim == 2 and int(faces.shape[1]) >= 3 and int(faces.shape[0]) > 0 and int(vertices.shape[0]) > 0:
                    if obj.mesh.face_normals is None or int(len(obj.mesh.face_normals)) != int(faces.shape[0]):
                        obj.mesh.compute_normals(compute_vertex_normals=False)
                    normals = np.asarray(obj.mesh.face_normals, dtype=np.float32)

                    glBegin(GL_TRIANGLES)
                    for fi in range(int(faces.shape[0])):
                        f = faces[fi]
                        n = normals[fi]
                        glNormal3f(float(n[0]), float(n[1]), float(n[2]))
                        glVertex3fv(vertices[int(f[0])])
                        glVertex3fv(vertices[int(f[1])])
                        glVertex3fv(vertices[int(f[2])])
                    glEnd()

                    # Preserve key diagnostics/overlays in fallback mode as well.
                    if self.floor_penetration_highlight:
                        self._draw_floor_penetration_immediate(obj, faces, vertices, depth_write=depth_write)
                    if is_selected and bool(getattr(self, "show_surface_assignment_overlay", True)):
                        self._draw_surface_assignment_overlay_immediate(obj, faces, vertices, depth_write=depth_write)
            except Exception:
                _log_ignored_exception("Immediate-mode mesh fallback failed", level=logging.WARNING)
        
        # 諛붾떏 ?묒큺 硫??섏씠?쇱씠?몃뒗 ?뺤튂(諛붾떏 ?뺣젹) 愿??紐⑤뱶?먯꽌留??쒖떆 (??⑸웾 硫붿돩 ?깅뒫)
        if is_selected and self.picking_mode in {'floor_3point', 'floor_face', 'floor_brush'}:
            self._draw_floor_contact_faces(obj)

        glDisable(GL_CULL_FACE)

    def _draw_floor_penetration_immediate(
        self,
        obj: SceneObject,
        faces: np.ndarray,
        vertices: np.ndarray,
        *,
        depth_write: bool = True,
    ) -> None:
        try:
            wb = obj.get_world_bounds()
            if float(wb[0][2]) >= 0.0:
                return
        except Exception:
            return

        try:
            glEnable(GL_CLIP_PLANE0)
            glDepthMask(GL_FALSE)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(-1.0, -1.0)
            glColor3f(0.0, 1.0, 0.2)
            glBegin(GL_TRIANGLES)
            for f in faces:
                glVertex3fv(vertices[int(f[0])])
                glVertex3fv(vertices[int(f[1])])
                glVertex3fv(vertices[int(f[2])])
            glEnd()
        finally:
            try:
                glDisable(GL_POLYGON_OFFSET_FILL)
                glDepthMask(GL_TRUE if depth_write else GL_FALSE)
                glDisable(GL_CLIP_PLANE0)
            except Exception:
                _log_ignored_exception()

    def _draw_surface_assignment_overlay_immediate(
        self,
        obj: SceneObject,
        faces: np.ndarray,
        vertices: np.ndarray,
        *,
        depth_write: bool = True,
    ) -> None:
        try:
            outer_set = self._get_surface_target_set(obj, "outer")
            inner_set = self._get_surface_target_set(obj, "inner")
            migu_set = self._get_surface_target_set(obj, "migu")
            unresolved_set = set(
                int(x)
                for x in (getattr(obj, "surface_assist_unresolved_face_indices", set()) or set())
            )
            if unresolved_set:
                unresolved_set.difference_update(outer_set)
                unresolved_set.difference_update(inner_set)
                unresolved_set.difference_update(migu_set)

            paint_target = None
            if self.picking_mode in {"paint_surface_face", "paint_surface_brush", "paint_surface_area", "paint_surface_magnetic"}:
                paint_target = str(getattr(self, "_surface_paint_target", "outer")).strip().lower()
                if paint_target not in {"outer", "inner", "migu"}:
                    paint_target = "outer"

            outer_c = (0.20, 0.55, 1.00, 0.36)
            inner_c = (1.00, 0.55, 0.15, 0.36)
            migu_c = (0.20, 0.85, 0.35, 0.32)
            unresolved_c = (1.00, 0.92, 0.22, 0.30)
            if paint_target == "outer":
                outer_c = (outer_c[0], outer_c[1], outer_c[2], 0.50)
            elif paint_target == "inner":
                inner_c = (inner_c[0], inner_c[1], inner_c[2], 0.50)
            elif paint_target == "migu":
                migu_c = (migu_c[0], migu_c[1], migu_c[2], 0.46)

            def draw_face_set(face_set: set[int], rgba: tuple[float, float, float, float]) -> None:
                if not face_set:
                    return
                glColor4f(*rgba)
                glBegin(GL_TRIANGLES)
                for face_idx in face_set:
                    i = int(face_idx)
                    if i < 0 or i >= int(faces.shape[0]):
                        continue
                    f = faces[i]
                    glVertex3fv(vertices[int(f[0])])
                    glVertex3fv(vertices[int(f[1])])
                    glVertex3fv(vertices[int(f[2])])
                glEnd()

            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(GL_FALSE)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(-1.0, -1.0)

            draw_face_set(outer_set, outer_c)
            draw_face_set(inner_set, inner_c)
            draw_face_set(migu_set, migu_c)
            draw_face_set(unresolved_set, unresolved_c)
        finally:
            try:
                glDisable(GL_POLYGON_OFFSET_FILL)
                glDepthMask(GL_TRUE if depth_write else GL_FALSE)
                glDisable(GL_BLEND)
                glEnable(GL_LIGHTING)
            except Exception:
                _log_ignored_exception()
    
    def _draw_floor_contact_faces(self, obj: SceneObject):
        """諛붾떏(Z=0) 洹쇱쿂 硫댁쓣 珥덈줉?됱쑝濡??섏씠?쇱씠??(?뺤튂 怨쇱젙 以??쒖떆)"""
        if obj.mesh is None or obj.mesh.faces is None:
            return
        
        faces = obj.mesh.faces
        vertices = obj.mesh.vertices
        
        # ?뚯쟾 ?됰젹 怨꾩궛
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler('XYZ', obj.rotation, degrees=True).as_matrix()
        
        total_faces = len(faces)
        
        # ?섑뵆留?(???硫붿돩)
        sample_size = min(80000, total_faces)
        if total_faces > sample_size:
            # Deterministic stride sampling avoids per-frame flicker and random CPU spikes.
            step = int(max(1, total_faces // sample_size))
            indices = np.arange(0, total_faces, step, dtype=np.int32)[:sample_size]
        else:
            indices = np.arange(total_faces, dtype=np.int32)
        
        sample_faces = faces[indices]
        v_indices = sample_faces[:, 0]
        v_points = vertices[v_indices] * obj.scale
        
        # ?붾뱶 Z 醫뚰몴 怨꾩궛
        world_z = (r[2, 0] * v_points[:, 0] + 
                   r[2, 1] * v_points[:, 1] + 
                   r[2, 2] * v_points[:, 2]) + obj.translation[2]
        
        # 諛붾떏 洹쇱쿂 媛먯? (Z < 0.5cm ?먮뒗 Z < 0)
        # ?뺤튂 紐⑤뱶?먯꽌??諛붾떏 洹쇱쿂(0.5cm ?대궡)源뚯? ?쒖떆
        threshold = 0.5 if self.picking_mode == 'floor_3point' else 0.0
        near_floor_mask = world_z < threshold
        near_pos = np.where(near_floor_mask)[0]
        near_floor_indices = indices[near_pos]
        near_floor_z = world_z[near_pos]
        
        if len(near_floor_indices) == 0:
            return
        
        # 珥덈줉??梨꾩슦湲?
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-4.0, -4.0)
        
        # ?됱긽: 諛붾떏 ?꾨옒(Z<0)??吏꾪븳 珥덈줉, 洹쇱쿂(0<Z<0.5)???고븳 珥덈줉
        glBegin(GL_TRIANGLES)
        max_fill_faces = min(10000, int(len(near_floor_indices)))
        for k in range(max_fill_faces):
            face_idx = int(near_floor_indices[k])
            f = faces[face_idx]
            v0_z = float(near_floor_z[k])
            # ?섑룊 ?쒖젏?대굹 ?섎㈃ ?쒖젏(Elevation < 0)?먯꽌?????щ챸?섍쾶 泥섎━?섏뿬 硫붿돩瑜?媛由ъ? ?딄쾶 ??
            is_bottom_view = self.camera.elevation < -45
            alpha_penetrate = 0.1 if is_bottom_view else 0.4
            alpha_near = 0.05 if is_bottom_view else 0.2
            
            if v0_z < 0:
                glColor4f(0.0, 1.0, 0.2, alpha_penetrate)  # 吏꾪븳 珥덈줉 (愿??
            else:
                glColor4f(0.5, 1.0, 0.5, alpha_near)  # ?고븳 珥덈줉 (洹쇱쿂)
            for v_idx in f:
                glVertex3fv(vertices[v_idx])
        glEnd()

        # Keep contact highlight as translucent fill only.
        # Drawing per-triangle wireframe here made meshes look like "screen door".
        glPopAttrib()
    
    def draw_mesh_dimensions(self, obj: SceneObject):
        """硫붿돩 以묒떖????옄???쒖떆 (?쒕옒洹몃줈 ?대룞 媛??"""
        if obj.mesh is None:
            return

        # ?붾뱶 醫뚰몴?먯꽌 諛붿슫??諛뺤뒪 怨꾩궛 (??⑸웾 硫붿돩?먯꽌??O(1))
        wb = obj.get_world_bounds()
        min_pt = wb[0]
        max_pt = wb[1]
        
        center_x = (min_pt[0] + max_pt[0]) / 2
        center_y = (min_pt[1] + max_pt[1]) / 2
        z = min_pt[2] + 0.1  # 諛붾떏 ?댁쭩 ??        
        # 以묒떖?????(?쒕옒洹몄슜)
        self._mesh_center = np.array([center_x, center_y, z])
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # ?묒? ??옄??(鍮④컙??
        glColor3f(1.0, 0.3, 0.3)
        glLineWidth(2.0)
        marker_size = 1.5  # 怨좎젙 ?ш린 1.5cm
        glBegin(GL_LINES)
        glVertex3f(center_x - marker_size, center_y, z)
        glVertex3f(center_x + marker_size, center_y, z)
        glVertex3f(center_x, center_y - marker_size, z)
        glVertex3f(center_x, center_y + marker_size, z)
        glEnd()
        
        # ?먯젏 ?쒖떆 (?뱀깋 ?묒? ??
        glColor3f(0.3, 0.9, 0.3)
        glBegin(GL_LINE_LOOP)
        for i in range(16):
            angle = 2.0 * np.pi * i / 16
            glVertex3f(0.5 * np.cos(angle), 0.5 * np.sin(angle), z)
        glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_rotation_gizmo(self, obj: SceneObject):
        """?뚯쟾 湲곗쫰紐?洹몃━湲?"""
        if not self.show_gizmo:
            return
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        glPushMatrix()
        # ?좏깮??媛앹껜???꾩튂濡??대룞
        glTranslatef(*obj.translation)
        
        # 湲곗쫰紐??ш린 ?ㅼ젙 (媛앹껜 ?ㅼ???諛섏쁺)
        size = self.gizmo_size * obj.scale
        
        # ?섏씠?쇱씠?몄슜 異?(hover ?먮뒗 active)
        highlight_axis = self.active_gizmo_axis or getattr(self, '_hover_axis', None)
        
        # X異?
        glColor3f(1.0, 0.2, 0.2)
        if highlight_axis == 'X':
            glLineWidth(5.0)
            glColor3f(1.0, 0.8, 0.0)
        else:
            glLineWidth(2.5)
        glPushMatrix()
        glRotatef(90, 0, 1, 0)
        self._draw_gizmo_circle(size)
        glPopMatrix()
        
        # Y異?
        glColor3f(0.2, 1.0, 0.2)
        if highlight_axis == 'Y':
            glLineWidth(5.0)
            glColor3f(1.0, 0.8, 0.0)
        else:
            glLineWidth(2.5)
        glPushMatrix()
        glRotatef(90, 1, 0, 0)
        self._draw_gizmo_circle(size)
        glPopMatrix()
        
        # Z異?
        glColor3f(0.2, 0.2, 1.0)
        if highlight_axis == 'Z':
            glLineWidth(5.0)
            glColor3f(1.0, 0.8, 0.0)
        else:
            glLineWidth(2.5)
        self._draw_gizmo_circle(size)
        
        glPopMatrix()
        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def _draw_gizmo_circle(self, radius, segments=64):
        """湲곗쫰紐⑥슜 ??洹몃━湲?"""
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            glVertex3f(x, y, 0)
        glEnd()

    def draw_wireframe(self, obj: SceneObject):
        """??댁뼱?꾨젅???ㅻ쾭?덉씠"""
        if obj is None or obj.mesh is None:
            return
        
        glDisable(GL_LIGHTING)
        glColor3f(0.3, 0.3, 0.3)
        glLineWidth(1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        
        glPushMatrix()
        glTranslatef(*obj.translation)
        glRotatef(obj.rotation[0], 1, 0, 0)
        glRotatef(obj.rotation[1], 0, 1, 0)
        glRotatef(obj.rotation[2], 0, 0, 1)
        glScalef(obj.scale, obj.scale, obj.scale)
        
        vertices = obj.mesh.vertices
        faces = obj.mesh.faces
        
        glBegin(GL_TRIANGLES)
        for face in faces:
            for vi in face:
                glVertex3fv(vertices[vi])
        glEnd()
        
        glPopMatrix()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_LIGHTING)
    
    def add_mesh_object(self, mesh, name=None):
        """??硫붿돩瑜??ъ뿉 異붽?"""
        if name is None:
            name = f"Object_{len(self.objects) + 1}"
            
        # Optional precomputed caches from the loader thread (performance on huge meshes).
        pre_centroids = getattr(mesh, "_amr_face_centroids", None)
        try:
            pre_centroids_faces = int(getattr(mesh, "_amr_face_centroids_faces_count", 0) or 0)
        except Exception:
            pre_centroids_faces = 0

        # 硫붿돩 ?먯껜瑜??먯젏?쇰줈 ?쇳꽣留?(濡쒖뺄 醫뚰몴怨??앹꽦)
        center = mesh.centroid
        mesh.vertices -= center
        # 罹먯떆 臾댄슚??(vertices 蹂寃?
        try:
            mesh._bounds = None
            mesh._centroid = None
            mesh._surface_area = None
        except Exception:
            _log_ignored_exception()
        # 濡쒕뵫 ?쒖젏?먮뒗 face normals留??꾩슂 (vertex normals???꾩슂????怨꾩궛)
        centroids_cache = None
        try:
            if pre_centroids is not None:
                fc = np.asarray(pre_centroids, dtype=np.float32)
                n_faces = int(getattr(mesh, "n_faces", int(fc.shape[0])) or int(fc.shape[0]))
                if (
                    fc.ndim == 2
                    and int(fc.shape[1]) >= 3
                    and int(fc.shape[0]) == n_faces
                    and (pre_centroids_faces <= 0 or int(fc.shape[0]) == int(pre_centroids_faces))
                ):
                    c = np.asarray(center[:3], dtype=np.float32).reshape(1, 3)
                    fc = fc[:, :3]
                    try:
                        fc -= c
                        centroids_cache = fc
                    except Exception:
                        centroids_cache = (fc - c).astype(np.float32, copy=False)
        except Exception:
            centroids_cache = None

        try:
            if getattr(mesh, "face_normals", None) is None:
                mesh.compute_normals(compute_vertex_normals=False)
        except Exception:
            _log_ignored_exception()
        
        new_obj = SceneObject(mesh, name)
        self.objects.append(new_obj)
        self.selected_index = len(self.objects) - 1
        
        # VBO ?곗씠???앹꽦
        self.update_vbo(new_obj)

        # Attach centroid cache after update_vbo (it invalidates caches defensively).
        if centroids_cache is not None:
            try:
                new_obj._face_centroids = centroids_cache
                new_obj._face_centroid_faces_count = int(
                    getattr(mesh, "n_faces", int(centroids_cache.shape[0])) or int(centroids_cache.shape[0])
                )
                new_obj._face_centroid_kdtree = None
            except Exception:
                _log_ignored_exception()
        
        # 移대찓???쇳똿 (泥?踰덉㎏ 媛앹껜??寃쎌슦留?
        if len(self.objects) == 1:
            self.update_grid_scale()
            try:
                # Ensure first loaded mesh is immediately visible.
                self._front_back_ortho_enabled = False
                self.camera.fit_to_bounds(new_obj.get_world_bounds())
                self.camera.pan_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            except Exception:
                _log_ignored_exception()
        
        self.meshLoaded.emit(mesh)
        self.selectionChanged.emit(self.selected_index)
        self.update()

    def clear_scene(self) -> None:
        """?ъ쓽 紐⑤뱺 媛앹껜/?ㅻ쾭?덉씠瑜??쒓굅?섍퀬 湲곕낯 ?곹깭濡?由ъ뀑?⑸땲??"""
        try:
            self.makeCurrent()
        except Exception:
            pass

        # Cleanup GL resources
        try:
            for obj in getattr(self, "objects", []) or []:
                try:
                    obj.cleanup()
                except Exception:
                    _log_ignored_exception("SceneObject cleanup failed", level=logging.WARNING)
        except Exception:
            _log_ignored_exception("Scene cleanup loop failed", level=logging.WARNING)

        self.objects = []
        self.selected_index = -1
        self._mesh_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Tool / overlay state
        self.picking_mode = "none"
        self.curvature_pick_mode = False
        self.picked_points = []
        self.fitted_arc = None
        self.measure_picked_points = []

        self.slice_enabled = False
        self.slice_z = 0.0
        self.slice_contours = []

        self.crosshair_enabled = False
        self.crosshair_pos = np.array([0.0, 0.0], dtype=np.float64)
        self.x_profile = []
        self.y_profile = []
        self._world_x_profile = []
        self._world_y_profile = []

        self.roi_enabled = False
        self.active_roi_edge = None
        self.roi_rect_dragging = False
        self.roi_rect_start = None
        self._roi_bounds_changed = False
        self._roi_move_dragging = False
        self._roi_move_last_xy = None
        self._roi_commit_axis_hint = None
        self._roi_last_adjust_axis = None
        self._roi_commit_plane_hint = None
        self._roi_last_adjust_plane = None
        self.roi_bounds = [-10.0, 10.0, -10.0, 10.0]
        self.roi_cut_edges = {"x1": [], "x2": [], "y1": [], "y2": []}
        self.roi_cap_verts = {"x1": None, "x2": None, "y1": None, "y2": None}
        self.roi_section_world = {"x": [], "y": []}
        self.roi_caps_enabled = False

        self.cut_lines_enabled = False
        self.cut_lines = [[], []]
        self.cut_line_axis_lock = ["x", "y"]
        self.cut_line_active = 0
        self.cut_line_drawing = False
        self.cut_line_preview = None
        self._cut_line_final = [False, False]
        self.cut_section_profiles = [[], []]
        self.cut_section_world = [[], []]
        try:
            self._cut_section_pending_indices.clear()
        except Exception:
            pass
        try:
            self.cutLinesEnabledChanged.emit(False)
        except Exception:
            _log_ignored_exception()

        self.line_section_enabled = False
        self.line_section_dragging = False
        self.line_section_start = None
        self.line_section_end = None
        self.line_profile = []
        self.line_section_contours = []

        # Floor alignment picks
        try:
            self.floor_picks = []
        except Exception:
            pass

        # Surface paint overlays
        try:
            self.surface_paint_points = []
            self.surface_lasso_points = []
            self.surface_lasso_face_indices = []
            self.surface_lasso_preview = None
        except Exception:
            pass

        try:
            self.selectionChanged.emit(-1)
        except Exception:
            _log_ignored_exception()
        self.update()

    def update_grid_scale(self):
        """?좏깮??硫붿돩 ?ш린??留욎떠 寃⑹옄 ?ㅼ???議곗젙"""
        obj = self.selected_obj
        if not obj:
            return
            
        bounds = obj.mesh.bounds
        extents = bounds[1] - bounds[0]
        max_dim = np.max(extents)
        
        if max_dim < 50:
            # 1cm grid for small objects
            self.grid_spacing = 1.0
            self.grid_size = 100.0
        elif max_dim < 200:
            # 5cm grid
            self.grid_spacing = 5.0
            self.grid_size = 500.0
        else:
            # 10cm grid for large objects
            self.grid_spacing = 10.0
            self.grid_size = max_dim * 1.5
            
        self.camera.distance = max_dim * 2
        self.camera.center = obj.translation.copy()
        self.camera.pan_offset = np.array([0.0, 0.0, 0.0])
        self.update_gizmo_size()

    def update_gizmo_size(self):
        """?좏깮??硫붿돩 ?ш린??留욎떠 ?뚯쟾 湲곗쫰紐?諛섍꼍 議곗젙"""
        obj = self.selected_obj
        if not obj or getattr(obj, "mesh", None) is None:
            return

        try:
            bounds = obj.mesh.bounds
            extents = bounds[1] - bounds[0]
            max_dim = float(np.max(extents))
        except Exception:
            return

        factor = float(getattr(self, "gizmo_radius_factor", 0.72))
        factor = max(0.60, min(2.5, factor))
        self.gizmo_radius_factor = factor
        self.gizmo_size = max_dim * 0.5 * factor
    
    def hit_test_gizmo(self, screen_x, screen_y):
        """湲곗쫰紐?怨좊━ ?대┃ 寃??"""
        obj = self.selected_obj
        if not obj:
            return None
        
        try:
            self.makeCurrent()
            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            gl_x, gl_y = self._qt_to_gl_window_xy(float(screen_x), float(screen_y), viewport=viewport)

            # Project object center to screen to get a reference point for gizmo size
            obj_screen_pos = gluProject(*obj.translation, modelview, projection, viewport)
            if not obj_screen_pos:
                return None

            # Calculate a world-space ray from screen coordinates
            near_pt = gluUnProject(gl_x, gl_y, 0.0, modelview, projection, viewport)
            far_pt = gluUnProject(gl_x, gl_y, 1.0, modelview, projection, viewport)
            if not near_pt or not far_pt:
                return None
            
            ray_origin = np.array(near_pt)
            ray_dir = np.array(far_pt) - ray_origin
            ray_dir /= np.linalg.norm(ray_dir)
            
            best_axis = None
            min_ray_t = float('inf')
            
            # Gizmo size scales with distance from camera to maintain constant screen size
            # This is a rough approximation, a more accurate way would involve projecting points
            # and calculating screen-space distances.
            # For now, let's use a fixed screen-space threshold or a world-space threshold
            # that scales with camera distance.
            
            # Calculate a dynamic threshold based on screen size of gizmo
            # This is a simplified approach. A more robust solution would involve
            # rendering the gizmo to a small off-screen buffer with unique colors for picking.
            
            # Let's try to estimate a world-space threshold based on screen pixel size
            pixel_world_size = 0.005 * self.camera.distance # Heuristic value
            threshold = pixel_world_size * 5 # Allow for a few pixels tolerance
            
            scaled_gizmo_radius = self.gizmo_size * obj.scale
            center = obj.translation
            
            # Define the planes for each axis's circle
            # X-axis circle is in YZ plane, normal is X-axis
            # Y-axis circle is in XZ plane, normal is Y-axis
            # Z-axis circle is in XY plane, normal is Z-axis
            
            planes = {
                'X': {'normal': np.array([1.0, 0.0, 0.0]), 'axis_vec': np.array([0.0, 1.0, 0.0])}, # Y-axis for rotation
                'Y': {'normal': np.array([0.0, 1.0, 0.0]), 'axis_vec': np.array([1.0, 0.0, 0.0])}, # X-axis for rotation
                'Z': {'normal': np.array([0.0, 0.0, 1.0]), 'axis_vec': np.array([1.0, 0.0, 0.0])}  # X-axis for rotation
            }
            
            for axis, plane_info in planes.items():
                normal = plane_info['normal']
                
                # Intersection of ray with the plane containing the circle
                denom = np.dot(ray_dir, normal)
                if abs(denom) > 1e-6: # Ray is not parallel to the plane
                    t = np.dot(center - ray_origin, normal) / denom
                    if t > 0: # Intersection is in front of the camera
                        hit_pt = ray_origin + t * ray_dir
                        
                        # Check if hit_pt is on the circle
                        dist_from_center = np.linalg.norm(hit_pt - center)
                        
                        if abs(dist_from_center - scaled_gizmo_radius) < threshold:
                            if t < min_ray_t:
                                min_ray_t = t
                                best_axis = axis
            return best_axis
        except Exception:
            return None

    def update_vbo(self, obj: SceneObject):
        """媛앹껜??VBO ?앹꽦 諛??곗씠???꾩넚"""
        made_current = False
        created_vbo = False
        prev_vbo_id = getattr(obj, "vbo_id", None) if obj is not None else None
        try:
            prev_vertex_count = int(getattr(obj, "vertex_count", 0) or 0) if obj is not None else 0
        except Exception:
            prev_vertex_count = 0
        try:
            if obj is None or obj.mesh is None:
                return

            try:
                ctx = self.context()
                if ctx is not None and QOpenGLContext.currentContext() != ctx:
                    self.makeCurrent()
                    made_current = True
            except Exception:
                pass

            faces = obj.mesh.faces
            n_faces = int(faces.shape[0]) if getattr(faces, "ndim", 0) == 2 else int(len(faces))
            try:
                n_vertices = int(getattr(obj.mesh, "n_vertices", int(np.asarray(obj.mesh.vertices).shape[0])) or 0)
            except Exception:
                n_vertices = int(np.asarray(obj.mesh.vertices).shape[0])

            want_vertex_normals = n_faces <= 1_200_000
            try:
                need_face = obj.mesh.face_normals is None or int(len(obj.mesh.face_normals)) != n_faces
            except Exception:
                need_face = True
            try:
                need_vertex = (
                    want_vertex_normals
                    and (
                        getattr(obj.mesh, "normals", None) is None
                        or int(len(obj.mesh.normals)) != n_vertices
                    )
                )
            except Exception:
                need_vertex = bool(want_vertex_normals)
            if need_face or need_vertex:
                obj.mesh.compute_normals(compute_vertex_normals=bool(want_vertex_normals))

            v_indices = faces.reshape(-1)
            vertex_count = int(v_indices.size)

            # [vx,vy,vz,nx,ny,nz] float32 interleaved (avoid huge temporaries)
            data = np.empty((vertex_count, 6), dtype=np.float32)
            np.take(obj.mesh.vertices, v_indices, axis=0, out=data[:, :3])
            use_vertex_normals = False
            try:
                vn = np.asarray(getattr(obj.mesh, "normals", None), dtype=np.float32)
                use_vertex_normals = (
                    want_vertex_normals
                    and vn.ndim == 2
                    and int(vn.shape[1]) >= 3
                    and int(vn.shape[0]) == n_vertices
                )
            except Exception:
                vn = None
                use_vertex_normals = False

            if use_vertex_normals and vn is not None:
                np.take(vn[:, :3], v_indices, axis=0, out=data[:, 3:])
            else:
                # face normals repeated 3 times (broadcast assignment, no big temp)
                fn = np.asarray(obj.mesh.face_normals, dtype=np.float32)
                if fn.ndim != 2 or int(fn.shape[0]) != n_faces or int(fn.shape[1]) < 3:
                    obj.mesh.compute_normals(compute_vertex_normals=False, force=True)
                    fn = np.asarray(obj.mesh.face_normals, dtype=np.float32)
                data[:, 3:].reshape((n_faces, 3, 3))[:] = fn[:, None, :3]

            try:
                vbo_id = int(getattr(obj, "vbo_id", 0) or 0)
            except Exception:
                vbo_id = 0

            if vbo_id <= 0:
                vbo_id = int(glGenBuffers(1) or 0)
                if vbo_id <= 0:
                    raise RuntimeError("glGenBuffers returned 0 (invalid VBO id)")
                obj.vbo_id = vbo_id
                created_vbo = True

            glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            obj.vertex_count = vertex_count

            # Picking cache invalidate (mesh vertices may have changed)
            try:
                obj._face_centroid_kdtree = None
                obj._face_centroids = None
                obj._face_centroid_faces_count = 0
                obj._face_adjacency = None
                obj._face_adjacency_faces_count = 0
            except Exception:
                _log_ignored_exception()
        except Exception:
            try:
                _LOGGER.exception("VBO creation failed for %s", getattr(obj, "name", "<unknown>"))
            except Exception:
                pass
            try:
                if created_vbo and int(getattr(obj, "vbo_id", 0) or 0) > 0:
                    glDeleteBuffers(1, [int(obj.vbo_id)])
                    obj.vbo_id = None
                else:
                    prev_vbo = int(prev_vbo_id or 0) if prev_vbo_id is not None else 0
                    obj.vbo_id = prev_vbo if prev_vbo > 0 else None
                obj.vertex_count = prev_vertex_count
            except Exception:
                _log_ignored_exception()
        finally:
            try:
                glBindBuffer(GL_ARRAY_BUFFER, 0)
            except Exception:
                _log_ignored_exception()
            if made_current:
                try:
                    self.doneCurrent()
                except Exception:
                    _log_ignored_exception()
    
    def fit_view_to_selected_object(self):
        """?좏깮??媛앹껜??移대찓??珥덉젏 留욎땄"""
        obj = self.selected_obj
        if obj:
            self.camera.fit_to_bounds(obj.mesh.bounds)
            self.camera.center = obj.translation.copy()
            self.camera.pan_offset = np.array([0.0, 0.0, 0.0])
            self.update()
            
    def set_mesh_translation(self, x, y, z):
        if self.selected_obj:
            self.selected_obj.translation = np.array([x, y, z])
            self.update()
    
    def set_mesh_rotation(self, rx, ry, rz):
        if self.selected_obj:
            self.selected_obj.rotation = np.array([rx, ry, rz])
            self.update()
            
    def set_mesh_scale(self, scale):
        if self.selected_obj:
            self.selected_obj.scale = scale
            self.update()
            
    def select_object(self, index):
        if 0 <= index < len(self.objects):
            self.selected_index = index
            self.update()
            self.selectionChanged.emit(index)

    def bake_object_transform(self, obj: SceneObject):
        """
        ?뺤튂 ?뺤젙: ?꾩옱 蹂댁씠??洹몃?濡?硫붿돩瑜?怨좎젙?섍퀬 紐⑤뱺 蹂?섍컪??0?쇰줈 由ъ뀑
        
        - ?꾩옱 ?붾㈃??蹂댁씠???꾩튂 洹몃?濡??좎?
        - ?대룞/?뚯쟾/諛곗쑉 媛믪씠 紐⑤몢 0?쇰줈 由ъ뀑
        - ?댄썑 0?먯꽌遺???몃? 議곗젙 媛??        """
        if not obj:
            return
        
        # 蹂?섏씠 ?놁쑝硫??ㅽ궢
        has_transform = (
            not np.allclose(obj.translation, [0, 0, 0]) or 
            not np.allclose(obj.rotation, [0, 0, 0]) or 
            obj.scale != 1.0
        )
        if not has_transform:
            # 蹂?섏씠 ?놁뼱??"?꾩옱 ?곹깭"瑜?怨좎젙 ?곹깭濡?湲곕줉
            try:
                obj.fixed_translation = np.asarray(obj.translation, dtype=np.float64).copy()
                obj.fixed_rotation = np.asarray(obj.rotation, dtype=np.float64).copy()
                obj.fixed_scale = float(obj.scale)
                obj.fixed_state_valid = True
            except Exception:
                _log_ignored_exception()
            return
        
        # 1. ?뚯쟾 ?됰젹 怨꾩궛
        rx, ry, rz = np.radians(obj.rotation)
        
        cos_x, sin_x = np.cos(rx), np.sin(rx)
        rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        
        cos_z, sin_z = np.cos(rz), np.sin(rz)
        rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        
        # OpenGL ?뚮뜑留?glRotate X->Y->Z)怨??숈씪???⑹꽦 ?뚯쟾
        rotation_matrix = rot_x @ rot_y @ rot_z
        
        # 2. ?뺤젏 蹂??(S -> R -> T ?쒖꽌, ?뚮뜑留곴낵 ?숈씪)
        # ?ㅼ???
        vertices = obj.mesh.vertices * obj.scale
        
        # ?뚯쟾
        vertices = (rotation_matrix @ vertices.T).T
        
        # ?대룞 (?붾뱶 醫뚰몴???곸슜)
        vertices = vertices + obj.translation
        
        # 3. ?곗씠???낅뜲?댄듃
        obj.mesh.vertices = vertices.astype(np.float32)
        # 罹먯떆 臾댄슚??(vertices 蹂寃?
        try:
            obj.mesh._bounds = None
            obj.mesh._centroid = None
            obj.mesh._surface_area = None
        except Exception:
            _log_ignored_exception()
        
        # Recompute face normals after baking transformed vertices.
        
        obj.mesh.compute_normals(compute_vertex_normals=False, force=True)
        obj._trimesh = None
        
        # 4. 紐⑤뱺 蹂?섍컪 0?쇰줈 由ъ뀑 (?댁젣 硫붿돩 ?뺤젏 ?먯껜媛 ?붾뱶 醫뚰몴)
        obj.translation = np.array([0.0, 0.0, 0.0])
        obj.rotation = np.array([0.0, 0.0, 0.0])
        obj.scale = 1.0

        # 4.5 怨좎젙 ?곹깭 媛깆떊 (?ㅼ닔濡??吏곸뿬??蹂듦? 媛??
        try:
            obj.fixed_translation = obj.translation.copy()
            obj.fixed_rotation = obj.rotation.copy()
            obj.fixed_scale = float(obj.scale)
            obj.fixed_state_valid = True
        except Exception:
            _log_ignored_exception()
        
        # 5. VBO ?낅뜲?댄듃
        self.update_vbo(obj)
        self.update()
        self.meshTransformChanged.emit()

    def restore_fixed_state(self, obj: SceneObject):
        """?뺤튂 ?뺤젙 ?댄썑??'怨좎젙 ?곹깭'濡?蹂?섍컪 蹂듦?"""
        if not obj:
            return
        if not getattr(obj, "fixed_state_valid", False):
            return

        try:
            obj.translation = np.asarray(obj.fixed_translation, dtype=np.float64).copy()
            obj.rotation = np.asarray(obj.fixed_rotation, dtype=np.float64).copy()
            obj.scale = float(obj.fixed_scale)
        except Exception:
            return

        self.update()
        self.meshTransformChanged.emit()

    def save_undo_state(self):
        """?꾩옱 ?좏깮??媛앹껜??蹂???곹깭瑜??ㅽ깮?????"""
        obj = self.selected_obj
        if not obj:
            return
        
        state = {
            'obj': obj,
            'translation': obj.translation.copy(),
            'rotation': obj.rotation.copy(),
            'scale': obj.scale if isinstance(obj.scale, (int, float)) else obj.scale.copy() if hasattr(obj.scale, 'copy') else obj.scale
        }
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def undo(self):
        """留덉?留?蹂??痍⑥냼 (Ctrl+Z)"""
        if not self.undo_stack:
            return
            
        state = self.undo_stack.pop()
        obj = state['obj']
        obj.translation = state['translation']
        obj.rotation = state['rotation']
        obj.scale = state['scale']
        
        self.update()
        self.meshTransformChanged.emit()
        self.status_info = "Undo transform"

    def _begin_ctrl_drag(self, event: QMouseEvent, obj: SceneObject) -> bool:
        """Ctrl+?쒕옒洹몄슜 源딆씠/?됰젹 罹먯떆瑜?以鍮꾪빀?덈떎."""
        if event is None or obj is None:
            return False
        try:
            self.makeCurrent()
            self._cached_viewport = glGetIntegerv(GL_VIEWPORT)
            self._cached_modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            self._cached_projection = glGetDoublev(GL_PROJECTION_MATRIX)

            gl_x, gl_y = self._qt_to_gl_window_xy(
                float(event.pos().x()),
                float(event.pos().y()),
                viewport=self._cached_viewport,
            )
            depth = cast(Any, glReadPixels(gl_x, gl_y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT))
            try:
                self._drag_depth = float(depth[0][0])
            except Exception:
                self._drag_depth = 1.0

            # 諛곌꼍???대┃??寃쎌슦 媛앹껜 以묒떖 源딆씠濡??泥?
            if self._drag_depth >= 1.0:
                obj_win_pos = gluProject(
                    *obj.translation,
                    self._cached_modelview,
                    self._cached_projection,
                    self._cached_viewport,
                )
                if obj_win_pos:
                    self._drag_depth = float(obj_win_pos[2])
            return True
        except Exception:
            _log_ignored_exception("Failed to initialize ctrl-drag cache", level=logging.WARNING)
            self._cached_viewport = None
            self._cached_modelview = None
            self._cached_projection = None
            self._drag_depth = 1.0
            return False
    
    def mousePressEvent(self, a0: QMouseEvent | None):
        if a0 is None:
            return
        event = a0
        """留덉슦??踰꾪듉 ?뚮┝"""
        try:
            self.last_mouse_pos = event.pos()
            self.mouse_button = event.button()
            modifiers = event.modifiers()
            obj_for_ctrl_drag = self.selected_obj
            self._ctrl_drag_active = False

            # Ctrl+?고겢由??쒕옒洹몃뒗 紐⑤뱺 紐⑤뱶?먯꽌 "硫붿돩 ?대룞" ?곗꽑.
            # (?쒕㈃/?⑤㈃ ?꾧뎄???고겢由??뺤젙怨?異⑸룎?섏? ?딅룄濡?press ?쒖젏?먯꽌 ?좎젏)
            if (
                event.button() == Qt.MouseButton.RightButton
                and bool(modifiers & Qt.KeyboardModifier.ControlModifier)
                and obj_for_ctrl_drag is not None
                and (not getattr(self, "roi_enabled", False))
            ):
                self.save_undo_state()
                self._ctrl_drag_active = bool(self._begin_ctrl_drag(event, obj_for_ctrl_drag))
                if self._ctrl_drag_active:
                    return

            # 1. ?쇰컲 ?대┃ (媛앹껜 ?좏깮 ?먮뒗 ?쇳궧 紐⑤뱶 泥섎━) - 醫뚰겢由?쭔 泥섎━
            if event.button() == Qt.MouseButton.LeftButton:
                # 湲곗쫰紐??좏깮 寃??(媛???곗꽑?쒖쐞) - ROI 紐⑤뱶?먯꽌???④?/鍮꾪솢??
                if self.picking_mode == 'none' and not getattr(self, "roi_enabled", False):
                    axis = self.hit_test_gizmo(event.pos().x(), event.pos().y())
                    if axis:
                        self.save_undo_state() # 蹂???쒖옉 ???곹깭 ???
                        self.active_gizmo_axis = axis

                        # 罹먯떆 留ㅽ듃由?뒪 ???(?깅뒫 理쒖쟻??
                        self.makeCurrent()
                        self._cached_viewport = glGetIntegerv(GL_VIEWPORT)
                        self._cached_modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
                        self._cached_projection = glGetDoublev(GL_PROJECTION_MATRIX)

                        angle = self._calculate_gizmo_angle(event.pos().x(), event.pos().y())
                        if angle is not None:
                            self.gizmo_drag_start = angle
                            self.update()
                            return

                # ?쇳궧 紐⑤뱶 泥섎━
                if self.picking_mode == 'curvature' and (modifiers & Qt.KeyboardModifier.ShiftModifier):
                    # Shift+?대┃?쇰줈留???李띻린
                    point = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                    if point is not None:
                        self.picked_points.append(point)
                        self.update()
                    return

                if self.picking_mode == "measure" and (modifiers & Qt.KeyboardModifier.ShiftModifier):
                    # Shift+?대┃?쇰줈留???李띻린
                    point = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                    if point is not None:
                        self.measure_picked_points.append(point)
                        try:
                            self.measurePointPicked.emit(point)
                        except Exception:
                            pass
                        self.update()
                    return
                        
                elif self.picking_mode == 'floor_3point':
                    point = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                    if point is not None:
                        # CAD AREA ?ㅽ??? ?쇳궧 ?먯? ?붾뱶 醫뚰몴 洹몃?濡??꾩쟻.
                        # (濡쒖뺄/?붾뱶 ?쇱슜 ???뚯쟾/?ㅼ??쇰맂 硫붿돩?먯꽌 ?됰㈃ 怨꾩궛??遺덉븞?뺥빐吏?
                        world_pt = np.asarray(point[:3], dtype=np.float64)

                        # 1. ?ㅻ깄 寃??(泥?踰덉㎏ ?먭낵 媛源뚯슦硫??뺤젙)
                        if len(self.floor_picks) >= 3:
                            first_pt = self.floor_picks[0]
                            dist = np.linalg.norm(world_pt - first_pt)
                            if dist < 0.15: # ?ㅻ깄 嫄곕━ ?뺣? (15cm)
                                self.floorAlignmentConfirmed.emit()
                                return
                                
                        self.floorPointPicked.emit(world_pt)
                        self.update()
                    return
                        
                elif self.picking_mode == 'floor_face':
                    point = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                    if point is not None:
                        vertices = self.pick_face_at_point(point)
                        if vertices:
                            self.floorFacePicked.emit(vertices)
                        else:
                            self.floorPointPicked.emit(point)
                        self.picking_mode = 'none'
                        self.update()
                    return
                
                elif self.picking_mode == 'floor_brush':
                    self.brush_selected_faces.clear()
                    self._pick_brush_face(event.pos())
                    self.update()
                    return

                elif self.picking_mode == 'select_face':
                    point = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                    if point is not None:
                        res = self.pick_face_at_point(point, return_index=True)
                        if res:
                            idx, _verts = res
                            obj = self.selected_obj
                            if obj is None:
                                return
                            idx = int(idx)
                            if modifiers & Qt.KeyboardModifier.AltModifier:
                                obj.selected_faces.discard(idx)
                            elif modifiers & (
                                Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
                            ):
                                obj.selected_faces.add(idx)
                            else:
                                if idx in obj.selected_faces:
                                    obj.selected_faces.discard(idx)
                                else:
                                    obj.selected_faces.add(idx)
                            self.faceSelectionChanged.emit(len(obj.selected_faces))
                        self.update()
                    return

                elif self.picking_mode == 'select_brush':
                    obj = self.selected_obj
                    if obj is None or obj.mesh is None:
                        return
                    if modifiers & Qt.KeyboardModifier.AltModifier:
                        self._selection_brush_mode = "remove"
                    elif modifiers & Qt.KeyboardModifier.ShiftModifier:
                        self._selection_brush_mode = "add"
                    else:
                        self._selection_brush_mode = "replace"

                    if self._selection_brush_mode == "replace":
                        obj.selected_faces.clear()
                    self._pick_selection_brush_face(event.pos())
                    self.faceSelectionChanged.emit(len(obj.selected_faces))
                    self.update()
                    return

                elif self.picking_mode == 'paint_surface_face':
                    # Click-to-apply on mouseRelease only (avoid accidental selection while orbiting).
                    if Qt.Key.Key_Space in getattr(self, "keys_pressed", set()):
                        return
                    self._surface_paint_left_press_pos = event.pos()
                    self._surface_paint_left_dragged = False
                    return

                elif self.picking_mode == 'paint_surface_area':
                    # Lasso polygon: add vertex on mouseRelease only (avoid accidental points while orbiting).
                    if Qt.Key.Key_Space in getattr(self, "keys_pressed", set()):
                        return
                    self._surface_area_left_press_pos = event.pos()
                    self._surface_area_left_dragged = False
                    try:
                        self.surface_lasso_preview = event.pos()
                    except Exception:
                        _log_ignored_exception()
                    return

                elif self.picking_mode == "paint_surface_magnetic":
                    # Boundary tool: click-to-add snapped vertices; drag to orbit camera.
                    if Qt.Key.Key_Space in getattr(self, "keys_pressed", set()):
                        self._surface_magnetic_space_nav = True
                        return
                    self._surface_magnetic_left_press_pos = event.pos()
                    self._surface_magnetic_left_dragged = False
                    try:
                        self._surface_magnetic_cursor_qt = event.pos()
                    except Exception:
                        _log_ignored_exception()
                    try:
                        self.surface_lasso_preview = event.pos()
                    except Exception:
                        _log_ignored_exception()
                    try:
                        self._ensure_surface_magnetic_cache(force=False)
                    except Exception:
                        _log_ignored_exception()
                    self._surface_magnetic_space_nav = False
                    self.update()
                    return

                elif self.picking_mode == 'paint_surface_brush':
                    if Qt.Key.Key_Space in getattr(self, "keys_pressed", set()):
                        return
                    remove = bool(modifiers & Qt.KeyboardModifier.AltModifier)
                    self._pick_surface_brush_face(event.pos(), remove=remove, modifiers=modifiers)
                    self.update()
                    return

                elif self.picking_mode == 'line_section':
                    pt = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                    if pt is not None:
                        self.line_section_enabled = True
                        self.line_section_dragging = True
                        self.line_section_start = np.array([pt[0], pt[1], 0.0], dtype=float)
                        self.line_section_end = self.line_section_start.copy()
                        self.line_section_contours = []
                        self.line_profile = []
                        self.lineProfileUpdated.emit([])
                        self._line_section_last_update = 0.0
                        self.update()
                    return

                elif self.picking_mode == 'cut_lines' and not (
                    modifiers
                    & (
                        Qt.KeyboardModifier.ShiftModifier
                        | Qt.KeyboardModifier.ControlModifier
                        | Qt.KeyboardModifier.AltModifier
                    )
                ):
                    if event.button() != Qt.MouseButton.LeftButton:
                        return
                    # ?ㅼ젣 ??異붽???mouseRelease?먯꽌 "?대┃"?쇰줈 ?먯젙?먯쓣 ?뚮쭔 ?섑뻾
                    self._cut_line_left_press_pos = event.pos()
                    self._cut_line_left_dragged = False
                    return
                 
                elif self.picking_mode == 'crosshair':
                    point = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                    if point is None:
                        # 硫붿돩 ?쎌씠 ?ㅽ뙣?대룄(?붿〈 ?뚯넀 ?? 諛붾떏 ?됰㈃?먯꽌 ??옄???대룞 媛??
                        point = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                    if point is not None:
                        self.crosshair_pos = point[:2]
                        self.schedule_crosshair_profile_update(0)
                        self._crosshair_last_update = time.monotonic()
                        self.update()
                    return

                elif self.roi_enabled:
                    # ROI handle click test (arrows + center-move handle)
                    handle = self._hit_test_roi(event.pos())
                    if handle:
                        self._roi_bounds_changed = False
                        if str(handle).strip().lower() == "move":
                            # Start moving the whole ROI rectangle
                            pt = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                            if pt is not None:
                                self.active_roi_edge = "move"
                                self.roi_rect_dragging = False
                                self.roi_rect_start = None
                                self._roi_move_dragging = True
                                self._roi_move_last_xy = np.array([float(pt[0]), float(pt[1])], dtype=np.float64)
                                self._roi_commit_axis_hint = None
                                self._roi_commit_plane_hint = None
                                self.update()
                                return
                        else:
                            self.active_roi_edge = str(handle)
                            self._roi_move_dragging = False
                            self._roi_move_last_xy = None
                            self._remember_roi_adjust_axis(handle)
                            self._roi_commit_axis_hint = self._roi_edge_to_axis(handle)
                            self._roi_commit_plane_hint = self._roi_edge_to_plane(handle)
                            self.update()
                            return

                    # New ROI capture by drag: Shift+drag only (prevents accidental resets while orbiting camera).
                    if modifiers & Qt.KeyboardModifier.ShiftModifier:
                        pt = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                        if pt is not None:
                            self._roi_bounds_changed = True
                            self._roi_move_dragging = False
                            self._roi_move_last_xy = None
                            self.active_roi_edge = None
                            self.roi_rect_dragging = True
                            self.roi_rect_start = np.array([float(pt[0]), float(pt[1])], dtype=np.float64)
                            self._roi_commit_axis_hint = None
                            self._roi_commit_plane_hint = None
                            # 珥덇린?먮뒗 理쒖냼 ?ш린瑜??뺣낫??0.1cm) ?댄썑 ?쒕옒洹몃줈 ?뺤옣
                            self.roi_bounds = [float(pt[0]), float(pt[0]) + 0.1, float(pt[1]), float(pt[1]) + 0.1]
                            self.schedule_roi_edges_update(0)
                            self.update()
                            return
                    # Otherwise: allow normal camera drag (left=rotate, right=pan)

            # ?⑤㈃??紐⑤뱶: ?고겢由?? "?뺤젙"(click) ?⑸룄濡??ъ슜 (?쒕옒洹??쒖뿉??Pan ?좎?)
            if self.picking_mode == "floor_3point" and event.button() == Qt.MouseButton.RightButton:
                # Right-click confirms only when released without dragging.
                # Right-drag remains available for camera pan in floor mode.
                self._floor_3point_right_press_pos = event.pos()
                self._floor_3point_right_dragged = False
                return

            if self.picking_mode == "cut_lines" and event.button() == Qt.MouseButton.RightButton:
                self._cut_line_right_press_pos = event.pos()
                self._cut_line_right_dragged = False
                return

            # ?쒕㈃ 吏??硫댁쟻/Area): ?고겢由?? "?뺤젙"(click) ?⑸룄濡??ъ슜 (?쒕옒洹??쒖뿉??Pan ?좎?)
            if self.picking_mode == "paint_surface_area" and event.button() == Qt.MouseButton.RightButton:
                self._surface_area_right_press_pos = event.pos()
                self._surface_area_right_dragged = False
                try:
                    self.surface_lasso_preview = event.pos()
                except Exception:
                    _log_ignored_exception()
                return

            # ?쒕㈃ 吏??寃쎄퀎/?먯꽍): ?고겢由?? "?뺤젙"(click) ?⑸룄濡??ъ슜 (?쒕옒洹??쒖뿉??Pan ?좎?)
            if self.picking_mode == "paint_surface_magnetic" and event.button() == Qt.MouseButton.RightButton:
                self._surface_magnetic_right_press_pos = event.pos()
                self._surface_magnetic_right_dragged = False
                try:
                    self._surface_magnetic_cursor_qt = event.pos()
                except Exception:
                    _log_ignored_exception()
                try:
                    self.surface_lasso_preview = event.pos()
                except Exception:
                    _log_ignored_exception()
                return

            # 3. 媛앹껜 議곗옉 (Shift/Ctrl + ?쒕옒洹?
            obj = self.selected_obj
            if (
                obj
                and (not getattr(self, "roi_enabled", False))
                and (modifiers & Qt.KeyboardModifier.ShiftModifier or modifiers & Qt.KeyboardModifier.ControlModifier)
            ):
                 self.save_undo_state() # 蹂???쒖옉 ???곹깭 ???                 
                 # Ctrl+?쒕옒洹??대룞)瑜??꾪븳 珥덇린 源딆씠媛????(留덉슦?ㅺ? 媛由ы궎??吏?먯쓽 源딆씠)
                 if modifiers & Qt.KeyboardModifier.ControlModifier:
                     self._ctrl_drag_active = bool(self._begin_ctrl_drag(event, obj))
                 return
            
            # 4. ???대┃: ?ъ빱???대룞 (Focus move)
            if event.button() == Qt.MouseButton.MiddleButton and modifiers == Qt.KeyboardModifier.NoModifier:
                point = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                if point is not None:
                    self.camera.center = point
                    self.camera.pan_offset = np.array([0.0, 0.0, 0.0])
                    self.update()
                    return

        except Exception:
            _log_ignored_exception("Mouse press error", level=logging.WARNING)

    def mouseReleaseEvent(self, a0: QMouseEvent | None):
        if a0 is None:
            return
        event = a0
        """留덉슦??踰꾪듉 ?볦쓬"""

        # ?⑤㈃??2媛?: 醫뚰겢由???異붽?(?대┃?쇰줈留?, ?고겢由??꾩옱 ???뺤젙
        if self.picking_mode == "cut_lines":
            modifiers = event.modifiers()

            if event.button() == Qt.MouseButton.RightButton:
                try:
                    if getattr(self, "_cut_line_right_press_pos", None) is not None and not bool(
                        getattr(self, "_cut_line_right_dragged", False)
                    ):
                        self._finish_cut_lines_current()
                except Exception:
                    _log_ignored_exception()
                self._cut_line_right_press_pos = None
                self._cut_line_right_dragged = False

            if event.button() == Qt.MouseButton.LeftButton:
                try:
                    modified = bool(
                        modifiers
                        & (
                            Qt.KeyboardModifier.ShiftModifier
                            | Qt.KeyboardModifier.ControlModifier
                            | Qt.KeyboardModifier.AltModifier
                        )
                    )
                    if (
                        not modified
                        and getattr(self, "_cut_line_left_press_pos", None) is not None
                        and not bool(getattr(self, "_cut_line_left_dragged", False))
                    ):
                        # Keep top-view guide input free from mesh occlusion.
                        pt = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                        if pt is not None:
                            picked = np.array([pt[0], pt[1], 0.0], dtype=np.float64)
                        else:
                            picked = self.pick_point_on_mesh(event.pos().x(), event.pos().y())

                        if picked is not None:
                            idx = int(getattr(self, "cut_line_active", 0))
                            if idx not in (0, 1):
                                idx = 0
                            line = self.cut_lines[idx]

                            # 湲곗〈 ?⑤㈃ 寃곌낵???몄쭛 ??臾댄슚??
                            self.cut_section_profiles[idx] = []
                            self.cut_section_world[idx] = []

                            pz = float(picked[2]) if (np.asarray(picked).reshape(-1).size >= 3) else 0.0
                            if not np.isfinite(pz):
                                pz = 0.0
                            p = np.array([float(picked[0]), float(picked[1]), pz], dtype=np.float64)

                            # Polyline input (?긱꽩 紐⑥뼇 ??: 醫뚰겢由?쑝濡??먯쓣 怨꾩냽 異붽??섍퀬,
                            # Enter/?고겢由?쑝濡?"?뺤젙"?⑸땲?? (媛??멸렇癒쇳듃??Ortho濡??섑룊/?섏쭅 ?쒖빟)
                            try:
                                final = getattr(self, "_cut_line_final", [False, False])
                                if bool(final[idx]) and len(line) >= 2:
                                    # ?대? ?뺤젙???쇱씤? Backspace/Delete濡??몄쭛?섍굅??吏?곌퀬 ?ㅼ떆 ?쒖옉.
                                    self.cut_line_drawing = False
                                    self.cut_line_preview = None
                                    self.update()
                                    return
                                final[idx] = False
                            except Exception:
                                _log_ignored_exception()

                            # Polyline input: keep adding orthogonal segments (????議고빀).
                            if len(line) == 0:
                                line.append(p)
                            else:
                                anchor = np.asarray(line[-1], dtype=np.float64)
                                p2 = self._cutline_constrain_ortho(
                                    anchor,
                                    p,
                                    force_lock_axis=(len(line) <= 1),
                                )
                                if float(np.linalg.norm(p2[:2] - anchor[:2])) <= 1e-6:
                                    self.cut_line_preview = p2
                                    self.cut_line_drawing = True
                                    self.update()
                                    return
                                line.append(p2)

                            self.cut_line_drawing = True
                            self.cut_line_preview = None
                            self.update()
                except Exception:
                    _log_ignored_exception()
                self._cut_line_left_press_pos = None
                self._cut_line_left_dragged = False

        if self.picking_mode == "floor_3point" and event.button() == Qt.MouseButton.RightButton:
            try:
                if (
                    getattr(self, "_floor_3point_right_press_pos", None) is not None
                    and not bool(getattr(self, "_floor_3point_right_dragged", False))
                ):
                    if len(getattr(self, "floor_picks", []) or []) >= 3:
                        self.floorAlignmentConfirmed.emit()
                    else:
                        self.status_info = "Pick at least 3 floor points, then right-click (or press Enter) to confirm."
                        self.update()
            except Exception:
                _log_ignored_exception()
            self._floor_3point_right_press_pos = None
            self._floor_3point_right_dragged = False

        # ?쒕㈃ 吏??李띻린): ?쒕옒洹멸? ?꾨땲硫?由대━利덉뿉??1???곸슜
        if self.picking_mode == "paint_surface_face" and event.button() == Qt.MouseButton.LeftButton:
            try:
                if (
                    getattr(self, "_surface_paint_left_press_pos", None) is not None
                    and not bool(getattr(self, "_surface_paint_left_dragged", False))
                ):
                    modifiers = event.modifiers()
                    info = self.pick_point_on_mesh_info(event.pos().x(), event.pos().y())
                    if info is not None:
                        point, depth_value, gl_x, gl_y, viewport, modelview, projection = info
                        res = self.pick_face_at_point(point, return_index=True)
                        if res:
                            idx, _verts = res
                            # Default: pick a small local patch (so it feels like selecting a surface, not a single tri).
                            # Shift/Ctrl: stepwise expand (magic-wand style).
                            radius_override = None
                            try:
                                r_click = float(getattr(self, "_surface_click_radius_world", 0.0) or 0.0)
                            except Exception:
                                r_click = 0.0
                            if np.isfinite(r_click) and r_click > 0.0:
                                radius_override = float(r_click)
                            else:
                                try:
                                    px_r = float(getattr(self, "_surface_click_radius_px", 48.0) or 0.0)
                                except Exception:
                                    px_r = 0.0
                                r_px = self._world_radius_from_px_at_depth(
                                    int(gl_x),
                                    int(gl_y),
                                    float(depth_value),
                                    viewport,
                                    modelview,
                                    projection,
                                    float(px_r),
                                )
                                if np.isfinite(r_px) and float(r_px) > 0.0:
                                    radius_override = float(r_px)
                            self._apply_surface_seed_pick(
                                int(idx),
                                modifiers,
                                picked_point_world=point,
                                radius_world_override=radius_override,
                            )
                        self.update()
            except Exception:
                _log_ignored_exception()
            self._surface_paint_left_press_pos = None
            self._surface_paint_left_dragged = False

        # ?쒕㈃ 吏??硫댁쟻/Area): 醫뚰겢由???異붽?(?대┃???뚮쭔), ?고겢由??뺤젙(?대┃???뚮쭔)
        if self.picking_mode == "paint_surface_area":
            if event.button() == Qt.MouseButton.LeftButton:
                try:
                    if (
                        getattr(self, "_surface_area_left_press_pos", None) is not None
                        and not bool(getattr(self, "_surface_area_left_dragged", False))
                    ):
                        thr = getattr(self, "_surface_lasso_thread", None)
                        if thr is not None and bool(getattr(thr, "isRunning", lambda: False)()):
                            self.status_info = "Surface area selection is computing..."
                        else:
                            # Snap-close: click near the first point to close & confirm.
                            try:
                                snap_px = int(getattr(self, "_surface_area_close_snap_px", 12))
                            except Exception:
                                snap_px = 12
                            if snap_px > 0 and len(getattr(self, "surface_lasso_points", []) or []) >= 3:
                                try:
                                    p0 = np.asarray(self.surface_lasso_points[0], dtype=np.float64).reshape(-1)
                                    if p0.size >= 3:
                                        self.makeCurrent()
                                        vp0 = glGetIntegerv(GL_VIEWPORT)
                                        mv0 = glGetDoublev(GL_MODELVIEW_MATRIX)
                                        pr0 = glGetDoublev(GL_PROJECTION_MATRIX)
                                        win = gluProject(float(p0[0]), float(p0[1]), float(p0[2]), mv0, pr0, vp0)
                                        if win:
                                            qx0, qy0 = self._gl_window_to_qt_xy(float(win[0]), float(win[1]), viewport=vp0)
                                            dx0 = float(event.pos().x()) - float(qx0)
                                            dy0 = float(event.pos().y()) - float(qy0)
                                            if float(np.hypot(dx0, dy0)) <= float(snap_px):
                                                self._finish_surface_lasso(event.modifiers(), seed_pos=event.pos())
                                                self._surface_area_left_press_pos = None
                                                self._surface_area_left_dragged = False
                                                return
                                except Exception:
                                    _log_ignored_exception()

                            p = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                            if p is None:
                                self.status_info = "?좑툘 硫붿돩 ?꾨? ?대┃???먯쓣 李띿뼱 二쇱꽭??"
                            else:
                                try:
                                    self.surface_lasso_points.append(np.asarray(p[:3], dtype=np.float64))
                                except Exception:
                                    self.surface_lasso_points.append(p)
                                try:
                                    res = self.pick_face_at_point(np.asarray(p, dtype=np.float64), return_index=True)
                                    if res:
                                        fi, _v = res
                                        self.surface_lasso_face_indices.append(int(fi))
                                    else:
                                        self.surface_lasso_face_indices.append(-1)
                                except Exception:
                                    try:
                                        self.surface_lasso_face_indices.append(-1)
                                    except Exception:
                                        _log_ignored_exception()
                                try:
                                    self.surface_lasso_preview = event.pos()
                                except Exception:
                                    _log_ignored_exception()
                        self.update()
                except Exception:
                    _log_ignored_exception()
                self._surface_area_left_press_pos = None
                self._surface_area_left_dragged = False

            if event.button() == Qt.MouseButton.RightButton:
                try:
                    if (
                        getattr(self, "_surface_area_right_press_pos", None) is not None
                        and not bool(getattr(self, "_surface_area_right_dragged", False))
                    ):
                        self._finish_surface_lasso(event.modifiers(), seed_pos=event.pos())
                except Exception:
                    _log_ignored_exception()
                self._surface_area_right_press_pos = None
                self._surface_area_right_dragged = False

        # ?쒕㈃ 吏??寃쎄퀎/?먯꽍): 醫뚰겢由???異붽?(?대┃???뚮쭔), ?고겢由??뺤젙(?대┃???뚮쭔)
        if self.picking_mode == "paint_surface_magnetic":
            if event.button() == Qt.MouseButton.LeftButton:
                try:
                    self._surface_magnetic_cursor_qt = event.pos()
                except Exception:
                    _log_ignored_exception()
                try:
                    self.surface_lasso_preview = event.pos()
                except Exception:
                    _log_ignored_exception()

                try:
                    if (
                        getattr(self, "_surface_magnetic_left_press_pos", None) is not None
                        and not bool(getattr(self, "_surface_magnetic_left_dragged", False))
                        and not bool(getattr(self, "_surface_magnetic_space_nav", False))
                        and (Qt.Key.Key_Space not in getattr(self, "keys_pressed", set()))
                    ):
                        thr = getattr(self, "_surface_lasso_thread", None)
                        try:
                            if thr is not None and bool(getattr(thr, "isRunning", lambda: False)()):
                                self.status_info = "Surface boundary selection is computing..."
                                self.update()
                                self._surface_magnetic_left_press_pos = None
                                self._surface_magnetic_left_dragged = False
                                self._surface_magnetic_space_nav = False
                                return
                        except Exception:
                            _log_ignored_exception()

                        # Snap-close: click near the first point to close & confirm.
                        try:
                            snap_px = int(getattr(self, "_surface_magnetic_close_snap_px", 12))
                        except Exception:
                            snap_px = 12
                        if snap_px > 0 and len(getattr(self, "surface_lasso_points", []) or []) >= 3:
                            try:
                                p0 = np.asarray(self.surface_lasso_points[0], dtype=np.float64).reshape(-1)
                                if p0.size >= 3:
                                    self.makeCurrent()
                                    vp0 = glGetIntegerv(GL_VIEWPORT)
                                    mv0 = glGetDoublev(GL_MODELVIEW_MATRIX)
                                    pr0 = glGetDoublev(GL_PROJECTION_MATRIX)
                                    win = gluProject(float(p0[0]), float(p0[1]), float(p0[2]), mv0, pr0, vp0)
                                    if win:
                                        qx0, qy0 = self._gl_window_to_qt_xy(float(win[0]), float(win[1]), viewport=vp0)
                                        dx0 = float(event.pos().x()) - float(qx0)
                                        dy0 = float(event.pos().y()) - float(qy0)
                                        if float(np.hypot(dx0, dy0)) <= float(snap_px):
                                            self._finish_surface_lasso(event.modifiers(), seed_pos=event.pos())
                                            self._surface_magnetic_left_press_pos = None
                                            self._surface_magnetic_left_dragged = False
                                            self._surface_magnetic_space_nav = False
                                            return
                            except Exception:
                                _log_ignored_exception()

                        self._surface_boundary_try_add_world_point(event.pos())
                        self.update()
                except Exception:
                    _log_ignored_exception()
                self._surface_magnetic_space_nav = False
                self._surface_magnetic_left_press_pos = None
                self._surface_magnetic_left_dragged = False

            if event.button() == Qt.MouseButton.RightButton:
                try:
                    if (
                        getattr(self, "_surface_magnetic_right_press_pos", None) is not None
                        and not bool(getattr(self, "_surface_magnetic_right_dragged", False))
                    ):
                        self._finish_surface_lasso(event.modifiers(), seed_pos=event.pos())
                except Exception:
                    _log_ignored_exception()
                self._surface_magnetic_right_press_pos = None
                self._surface_magnetic_right_dragged = False

        if self.mouse_button == Qt.MouseButton.LeftButton and self.picking_mode == 'floor_brush':
            if self.brush_selected_faces:
                self.alignToBrushSelected.emit()
            self.picking_mode = 'none'
            self.update()

        if self.mouse_button == Qt.MouseButton.LeftButton and self.picking_mode == 'select_brush':
            obj = self.selected_obj
            self.faceSelectionChanged.emit(len(obj.selected_faces) if obj is not None else 0)
            self.update()

        if self.mouse_button == Qt.MouseButton.LeftButton and self.picking_mode == 'line_section':
            if self.line_section_dragging:
                self.line_section_dragging = False
                self._line_section_last_update = 0.0
                self.schedule_line_profile_update(0)

        if self.mouse_button == Qt.MouseButton.LeftButton and self.picking_mode == 'crosshair':
            # ?쒕옒洹??ㅻ줈?濡??명빐 留덉?留??꾩튂媛 諛섏쁺?섏? ?딆쓣 ???덉뼱, 由대━利???1???뺤젙 ?낅뜲?댄듃
            self._crosshair_last_update = 0.0
            self.schedule_crosshair_profile_update(0)

        if self.mouse_button == Qt.MouseButton.LeftButton and getattr(self, "roi_enabled", False):
            # ROI ?쒕옒洹??몃뱾/?ш컖?? 醫낅즺 ??理쒖쥌 ?곹깭濡?1???뺤젙 怨꾩궛
            roi_changed = bool(getattr(self, "_roi_bounds_changed", False))
            if getattr(self, "roi_rect_dragging", False):
                self.roi_rect_dragging = False
                self.roi_rect_start = None
            if bool(getattr(self, "_roi_move_dragging", False)):
                self._roi_move_dragging = False
                self._roi_move_last_xy = None
            if getattr(self, "active_roi_edge", None):
                self.active_roi_edge = None
            if roi_changed:
                self.schedule_roi_edges_update(0)
            self._roi_bounds_changed = False

        self.mouse_button = None
        self.active_gizmo_axis = None
        self.gizmo_drag_start = None
        self.last_mouse_pos = None
        
        # 罹먯떆 珥덇린??
        self._cached_viewport = None
        self._cached_modelview = None
        self._cached_projection = None
        self._ctrl_drag_active = False
        
        self.update()
    
    def mouseMoveEvent(self, a0: QMouseEvent | None):
        if a0 is None:
            return
        event = a0
        """留덉슦???대룞 (?쒕옒洹?"""
        try:
            # ?⑤㈃??紐⑤뱶: ?쒕옒洹?以묒뿉??"?대┃"?쇰줈 ?ㅼ씤?섏? ?딅룄濡??뚮옒洹?泥섎━
            if self.picking_mode == "cut_lines":
                threshold_px = 10
                thr2 = float(threshold_px * threshold_px)
                if (
                    self.mouse_button == Qt.MouseButton.LeftButton
                    and getattr(self, "_cut_line_left_press_pos", None) is not None
                    and not bool(getattr(self, "_cut_line_left_dragged", False))
                ):
                    pos0 = self._cut_line_left_press_pos
                    if pos0 is not None:
                        dx0 = float(event.pos().x() - pos0.x())
                        dy0 = float(event.pos().y() - pos0.y())
                        if float(dx0 * dx0 + dy0 * dy0) > thr2:
                            self._cut_line_left_dragged = True
                if (
                    self.mouse_button == Qt.MouseButton.RightButton
                    and getattr(self, "_cut_line_right_press_pos", None) is not None
                    and not bool(getattr(self, "_cut_line_right_dragged", False))
                ):
                    pos0 = self._cut_line_right_press_pos
                    if pos0 is not None:
                        dx0 = float(event.pos().x() - pos0.x())
                        dy0 = float(event.pos().y() - pos0.y())
                        if float(dx0 * dx0 + dy0 * dy0) > thr2:
                            self._cut_line_right_dragged = True

            if self.picking_mode == "floor_3point":
                threshold_px = 10
                thr2 = float(threshold_px * threshold_px)
                if (
                    self.mouse_button == Qt.MouseButton.RightButton
                    and getattr(self, "_floor_3point_right_press_pos", None) is not None
                    and not bool(getattr(self, "_floor_3point_right_dragged", False))
                ):
                    pos0 = self._floor_3point_right_press_pos
                    if pos0 is not None:
                        dx0 = float(event.pos().x() - pos0.x())
                        dy0 = float(event.pos().y() - pos0.y())
                        if float(dx0 * dx0 + dy0 * dy0) > thr2:
                            self._floor_3point_right_dragged = True

            # ?쒕㈃ 吏??李띻린): ?쒕옒洹몃뒗 移대찓???뚯쟾?쇰줈 媛꾩＜?섍퀬, 由대━利덉뿉?쒕쭔 "?대┃" 泥섎━
            if self.picking_mode == "paint_surface_face":
                threshold_px = 10
                thr2 = float(threshold_px * threshold_px)
                if (
                    self.mouse_button == Qt.MouseButton.LeftButton
                    and getattr(self, "_surface_paint_left_press_pos", None) is not None
                    and not bool(getattr(self, "_surface_paint_left_dragged", False))
                ):
                    pos0 = self._surface_paint_left_press_pos
                    if pos0 is not None:
                        dx0 = float(event.pos().x() - pos0.x())
                        dy0 = float(event.pos().y() - pos0.y())
                        if float(dx0 * dx0 + dy0 * dy0) > thr2:
                            self._surface_paint_left_dragged = True

            # ?쒕㈃ 吏??硫댁쟻/Area): 誘몃━蹂닿린 + ?쒕옒洹??먯젙
            if self.picking_mode == "paint_surface_area":
                threshold_px = 10
                thr2 = float(threshold_px * threshold_px)
                try:
                    self.surface_lasso_preview = event.pos()
                except Exception:
                    _log_ignored_exception()

                if (
                    self.mouse_button == Qt.MouseButton.LeftButton
                    and getattr(self, "_surface_area_left_press_pos", None) is not None
                    and not bool(getattr(self, "_surface_area_left_dragged", False))
                ):
                    pos0 = self._surface_area_left_press_pos
                    if pos0 is not None:
                        dx0 = float(event.pos().x() - pos0.x())
                        dy0 = float(event.pos().y() - pos0.y())
                        if float(dx0 * dx0 + dy0 * dy0) > thr2:
                            self._surface_area_left_dragged = True

                if (
                    self.mouse_button == Qt.MouseButton.RightButton
                    and getattr(self, "_surface_area_right_press_pos", None) is not None
                    and not bool(getattr(self, "_surface_area_right_dragged", False))
                ):
                    pos0 = self._surface_area_right_press_pos
                    if pos0 is not None:
                        dx0 = float(event.pos().x() - pos0.x())
                        dy0 = float(event.pos().y() - pos0.y())
                        if float(dx0 * dx0 + dy0 * dy0) > thr2:
                            self._surface_area_right_dragged = True

                self.update()
                if self.mouse_button is None:
                    return

            # ?쒕㈃ 吏??寃쎄퀎/?먯꽍): 而ㅼ꽌 ?꾨━酉?+ ?대┃?쇰줈 ??異붽?(?ㅻ깄)
            if self.picking_mode == "paint_surface_magnetic":
                threshold_px = 10
                thr2 = float(threshold_px * threshold_px)
                try:
                    self._surface_magnetic_cursor_qt = event.pos()
                except Exception:
                    _log_ignored_exception()
                try:
                    self.surface_lasso_preview = event.pos()
                except Exception:
                    _log_ignored_exception()

                if (
                    self.mouse_button == Qt.MouseButton.LeftButton
                    and getattr(self, "_surface_magnetic_left_press_pos", None) is not None
                    and not bool(getattr(self, "_surface_magnetic_left_dragged", False))
                ):
                    pos0 = self._surface_magnetic_left_press_pos
                    if pos0 is not None:
                        dx0 = float(event.pos().x() - pos0.x())
                        dy0 = float(event.pos().y() - pos0.y())
                        if float(dx0 * dx0 + dy0 * dy0) > thr2:
                            self._surface_magnetic_left_dragged = True

                if (
                    self.mouse_button == Qt.MouseButton.RightButton
                    and getattr(self, "_surface_magnetic_right_press_pos", None) is not None
                    and not bool(getattr(self, "_surface_magnetic_right_dragged", False))
                ):
                    pos0 = self._surface_magnetic_right_press_pos
                    if pos0 is not None:
                        dx0 = float(event.pos().x() - pos0.x())
                        dy0 = float(event.pos().y() - pos0.y())
                        if float(dx0 * dx0 + dy0 * dy0) > thr2:
                            self._surface_magnetic_right_dragged = True

                if self.mouse_button == Qt.MouseButton.LeftButton and Qt.Key.Key_Space in getattr(self, "keys_pressed", set()):
                    self._surface_magnetic_space_nav = True

                self.update()
                if self.mouse_button is None:
                    return

            # ?⑤㈃??2媛? ?꾨━酉? 留덉슦???몃옒??踰꾪듉 ?놁씠 ?대룞)?먯꽌???숈옉
            if self.picking_mode == 'cut_lines' and getattr(self, "cut_line_drawing", False):
                if self.mouse_button is None or self.mouse_button == Qt.MouseButton.LeftButton:
                    pt = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                    if pt is not None:
                        picked = np.array([pt[0], pt[1], 0.0], dtype=np.float64)
                    else:
                        picked = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                    if picked is not None:
                        pz = float(picked[2]) if (np.asarray(picked).reshape(-1).size >= 3) else 0.0
                        if not np.isfinite(pz):
                            pz = 0.0
                        p = np.array([float(picked[0]), float(picked[1]), pz], dtype=np.float64)
                        try:
                            idx = int(getattr(self, "cut_line_active", 0))
                            idx = idx if idx in (0, 1) else 0
                            line = self.cut_lines[idx]
                            if line:
                                anchor = np.asarray(line[-1], dtype=np.float64)
                                self.cut_line_preview = self._cutline_constrain_ortho(
                                    anchor,
                                    p,
                                    force_lock_axis=(len(line) <= 1),
                                )
                            else:
                                self.cut_line_preview = p
                        except Exception:
                            self.cut_line_preview = p
                        self.update()
                # 踰꾪듉 ?놁씠 ?대룞 以묒씠硫?移대찓???쒕옒洹?濡쒖쭅???吏 ?딅룄濡?議곌린 醫낅즺
                if self.mouse_button is None:
                    return

            if self.last_mouse_pos is None:
                self.last_mouse_pos = event.pos()
                return

            # ?댁쟾 ?꾩튂 ???諛??꾩옱 ?꾩튂 媛깆떊 (?쒕옒洹?怨꾩궛??
            prev_pos = self.last_mouse_pos
            dx = event.pos().x() - prev_pos.x()
            dy = event.pos().y() - prev_pos.y()
            self.last_mouse_pos = event.pos()
            
            obj = self.selected_obj
            modifiers = event.modifiers()
            
            # 1. 湲곗쫰紐??쒕옒洹?(醫뚰겢由?+ 湲곗쫰紐??쒕옒洹??쒖옉??
            if self.gizmo_drag_start is not None and self.active_gizmo_axis and obj and self.mouse_button == Qt.MouseButton.LeftButton:
                angle_info = self._calculate_gizmo_angle(event.pos().x(), event.pos().y())
                if angle_info is not None:
                    current_angle = angle_info
                    delta_angle = np.degrees(current_angle - self.gizmo_drag_start)
                    
                    # "?먮룞李??몃뱾" 吏곴??? 留덉슦?ㅼ쓽 ?뚯쟾 諛⑺뼢??硫붿돩 ?뚯쟾??1:1 留ㅼ묶
                    # 移대찓???쒖꽑怨??뚯쟾異뺤쓽 諛⑺뼢???꾪듃怨????듯빐 visual CW/CCW瑜?寃곗젙
                    view_dir = self.camera.look_at - self.camera.position
                    view_dir /= np.linalg.norm(view_dir)
                    
                    axis_vec = np.zeros(3)
                    if self.active_gizmo_axis == 'X':
                        axis_vec[0] = 1.0
                    elif self.active_gizmo_axis == 'Y':
                        axis_vec[1] = 1.0
                    elif self.active_gizmo_axis == 'Z':
                        axis_vec[2] = 1.0
                    
                    # ?쒓컖??諛섏쟾 ?щ? 寃곗젙 (?몃뱾???뚮━??諛⑺뼢怨?硫붿돩媛 ?꾨뒗 諛⑺뼢 ?쇱튂)
                    dot = np.dot(view_dir, axis_vec)
                    flip = 1.0 if dot > 0 else -1.0
                    
                    if self.active_gizmo_axis == 'X':
                        obj.rotation[0] += delta_angle * flip
                    elif self.active_gizmo_axis == 'Y':
                        obj.rotation[1] += delta_angle * flip
                    elif self.active_gizmo_axis == 'Z':
                        obj.rotation[2] += delta_angle * flip
                        
                    self.gizmo_drag_start = current_angle
                    self.meshTransformChanged.emit()
                    self.update()
                    return
                
            # 2. 湲곗쫰紐??몃쾭 ?섏씠?쇱씠??(踰꾪듉 ???뚮졇???뚮쭔)
            if event.buttons() == Qt.MouseButton.NoButton:
                axis = self.hit_test_gizmo(event.pos().x(), event.pos().y())
                # ?몃쾭 ???섏씠?쇱씠?몃쭔 蹂寃? active_gizmo_axis???대┃ ?쒖뿉留??ㅼ젙
                if axis != getattr(self, '_hover_axis', None):
                    self._hover_axis = axis
                    self.update()
                return
            
            # 3. 媛앹껜 吏곸젒 議곗옉 (Ctrl+?쒕옒洹?= 硫붿돩 ?대룞, 留덉슦??而ㅼ꽌瑜??뺥솗???곕씪媛?
            if (
                (not getattr(self, "roi_enabled", False))
                and (
                    bool(modifiers & Qt.KeyboardModifier.ControlModifier)
                    or bool(getattr(self, "_ctrl_drag_active", False))
                )
                and obj
                and self._cached_viewport is not None
            ):
                # 留덉슦???꾨젅????罹≪쿂??源딆씠? 留ㅽ듃由?뒪 ?ъ궗??(?깅뒫 ?μ긽)
                curr_x, curr_y = self._qt_to_gl_window_xy(
                    float(event.pos().x()),
                    float(event.pos().y()),
                    viewport=self._cached_viewport,
                )
                prev_x, prev_y = self._qt_to_gl_window_xy(
                    float(prev_pos.x()),
                    float(prev_pos.y()),
                    viewport=self._cached_viewport,
                )

                curr_world = gluUnProject(
                    curr_x,
                    curr_y,
                    self._drag_depth,
                    self._cached_modelview,
                    self._cached_projection,
                    self._cached_viewport,
                )
                prev_world = gluUnProject(
                    prev_x,
                    prev_y,
                    self._drag_depth,
                    self._cached_modelview,
                    self._cached_projection,
                    self._cached_viewport,
                )
                
                if curr_world and prev_world:
                    delta_world = np.array(curr_world) - np.array(prev_world)
                    
                    # 6媛?醫뚰몴怨??뺣젹 酉곗뿉?쒕뒗 2李⑥썝 ?대룞 媛뺤젣 (吏곴????μ긽)
                    el = self.camera.elevation
                    az = self.camera.azimuth % 360
                    
                    if abs(el) > 85: # ?곷㈃(90) / ?섎㈃(-90)
                        delta_world[2] = 0
                    elif abs(el) < 5: # ?뺣㈃, ?꾨㈃, 醫? ??                        # ?뺣㈃(-90/270), ?꾨㈃(90) -> Y異?怨좎젙
                        if abs(az - 90) < 5 or abs(az - 270) < 5:
                            delta_world[1] = 0
                        # ?곗륫(0/360), 醫뚯륫(180) -> X異?怨좎젙
                        elif abs(az) < 5 or abs(az - 360) < 5 or abs(az - 180) < 5:
                            delta_world[0] = 0
                            
                    obj.translation += delta_world
                
                self.meshTransformChanged.emit()
                self.update()
                return
            
            elif (not getattr(self, "roi_enabled", False)) and (modifiers & Qt.KeyboardModifier.AltModifier) and obj:
                # ?몃옓蹂??ㅽ????뚯쟾 - 諛⑺뼢 諛섏쟾
                rot_speed = 0.5
                
                # ?붾㈃ ?섑룊 ?쒕옒洹?-> Z異??뚯쟾 (諛⑺뼢 諛섏쟾)
                obj.rotation[2] += dx * rot_speed
                
                # ?붾㈃ ?섏쭅 ?쒕옒洹?-> 移대찓??諛⑺뼢 湲곗? ?쇱묶 (諛⑺뼢 諛섏쟾)
                az_rad = np.radians(self.camera.azimuth)
                obj.rotation[0] -= dy * rot_speed * np.sin(az_rad)
                obj.rotation[1] -= dy * rot_speed * np.cos(az_rad)
                
                self.meshTransformChanged.emit()
                self.update()
                return
            
            elif (
                (not getattr(self, "roi_enabled", False))
                and (modifiers & Qt.KeyboardModifier.ShiftModifier)
                and obj
                and self.mouse_button == Qt.MouseButton.LeftButton
                and self.picking_mode != 'line_section'
            ):
                # 吏곴????뚯쟾: ?쒕옒洹?諛⑺뼢 = ?뚯쟾 諛⑺뼢 (移대찓??湲곗?)
                rot_speed = 0.5
                
                # 醫뚯슦 ?쒕옒洹?-> ?붾㈃ Y異?湲곗? ?뚯쟾 (Z異??뚯쟾)
                obj.rotation[2] -= dx * rot_speed
                
                # ?곹븯 ?쒕옒洹?-> ?붾㈃?먯꽌 ?욌뮘濡?援대┝ (移대찓??諛⑺뼢 怨좊젮)
                az_rad = np.radians(self.camera.azimuth)
                obj.rotation[0] += dy * rot_speed * np.cos(az_rad)
                obj.rotation[1] += dy * rot_speed * np.sin(az_rad)
                
                self.meshTransformChanged.emit()
                self.update()
                return


                
            # 0. 釉뚮윭???쇳궧 泥섎━
            if self.mouse_button == Qt.MouseButton.LeftButton and self.picking_mode == 'floor_brush':
                self._pick_brush_face(event.pos())
                self.update()
                return

            # 0.1 ?좏깮 釉뚮윭??泥섎━ (SelectionPanel)
            if self.mouse_button == Qt.MouseButton.LeftButton and self.picking_mode == 'select_brush':
                now = time.monotonic()
                if now - float(getattr(self, "_selection_brush_last_pick", 0.0)) >= 0.03:
                    self._selection_brush_last_pick = now
                    self._pick_selection_brush_face(event.pos())
                    try:
                        obj = self.selected_obj
                        self.faceSelectionChanged.emit(len(obj.selected_faces) if obj is not None else 0)
                    except Exception:
                        _log_ignored_exception()
                self.update()
                return

            # 0.2 ?쒕㈃ 吏??釉뚮윭??泥섎━ (outer/inner/migu)
            if (
                self.mouse_button == Qt.MouseButton.LeftButton
                and self.picking_mode == 'paint_surface_brush'
                and (Qt.Key.Key_Space not in getattr(self, "keys_pressed", set()))
            ):
                now = time.monotonic()
                if now - float(getattr(self, "_surface_brush_last_pick", 0.0)) >= 0.03:
                    self._surface_brush_last_pick = now
                    remove = bool(modifiers & Qt.KeyboardModifier.AltModifier)
                    self._pick_surface_brush_face(event.pos(), remove=remove, modifiers=modifiers)
                self.update()
                return

            # 0.4 ?좏삎 ?⑤㈃(?쇱씤) ?쒕옒洹?泥섎━
            if self.picking_mode == 'line_section' and self.line_section_dragging and self.mouse_button == Qt.MouseButton.LeftButton:
                pt = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                if pt is not None and self.line_section_start is not None:
                    end = np.array([pt[0], pt[1], 0.0], dtype=float)

                    # Shift: CAD Ortho (?섑룊/?섏쭅 怨좎젙)
                    if modifiers & Qt.KeyboardModifier.ShiftModifier:
                        start = np.array(self.line_section_start, dtype=float)
                        dx_l = float(end[0] - start[0])
                        dy_l = float(end[1] - start[1])
                        if abs(dx_l) >= abs(dy_l):
                            end[1] = start[1]
                        else:
                            end[0] = start[0]

                    self.line_section_end = end

                    # ?곗궛 鍮꾩슜(?щ씪?댁떛) ?덉빟???꾪빐 ?쎄컙 ?ㅻ줈?留?
                    now = time.monotonic()
                    if now - self._line_section_last_update > 0.08:
                        self._line_section_last_update = now
                        self.schedule_line_profile_update(150)
                    else:
                        self.update()
                return
            
            # 0.5 ??옄???쒕옒洹?泥섎━
            if self.picking_mode == 'crosshair' and self.mouse_button == Qt.MouseButton.LeftButton:
                point = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                if point is None:
                    point = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                if point is not None:
                    self.crosshair_pos = point[:2]
                    now = time.monotonic()
                    if now - self._crosshair_last_update > 0.08:
                        self._crosshair_last_update = now
                        self.schedule_crosshair_profile_update(150)
                    self.update()
                return

            # 0.54 ROI ?대룞 ?쒕옒洹?(以묒븰 ?몃뱾)
            if (
                getattr(self, "roi_enabled", False)
                and bool(getattr(self, "_roi_move_dragging", False))
                and self.mouse_button == Qt.MouseButton.LeftButton
            ):
                pt = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                last_xy = getattr(self, "_roi_move_last_xy", None)
                if pt is not None and last_xy is not None:
                    try:
                        dxw = float(pt[0]) - float(last_xy[0])
                        dyw = float(pt[1]) - float(last_xy[1])
                    except Exception:
                        dxw, dyw = 0.0, 0.0
                    if np.isfinite(dxw) and np.isfinite(dyw) and (abs(dxw) > 1e-12 or abs(dyw) > 1e-12):
                        try:
                            x1, x2, y1, y2 = [float(v) for v in self.roi_bounds]
                            self.roi_bounds = [x1 + dxw, x2 + dxw, y1 + dyw, y2 + dyw]
                        except Exception:
                            _log_ignored_exception()
                        try:
                            self._roi_move_last_xy = np.array([float(pt[0]), float(pt[1])], dtype=np.float64)
                        except Exception:
                            self._roi_move_last_xy = None
                        self._roi_bounds_changed = True
                        self.schedule_roi_edges_update(self._roi_live_delay_ms())
                        self.update()
                return

            # 0.55 ROI ?ш컖???쒕옒洹?罹≪퀜泥섎읆 吏??
            if getattr(self, "roi_enabled", False) and getattr(self, "roi_rect_dragging", False) and self.mouse_button == Qt.MouseButton.LeftButton:
                pt = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                if pt is not None and self.roi_rect_start is not None:
                    x0, y0 = float(self.roi_rect_start[0]), float(self.roi_rect_start[1])
                    x1, y1 = float(pt[0]), float(pt[1])
                    min_x = min(x0, x1)
                    max_x = max(x0, x1)
                    min_y = min(y0, y1)
                    max_y = max(y0, y1)
                    # 理쒖냼 ?ш린 ?뺣낫 (0.1cm)
                    if max_x - min_x < 0.1:
                        max_x = min_x + 0.1
                    if max_y - min_y < 0.1:
                        max_y = min_y + 0.1
                    self.roi_bounds = [min_x, max_x, min_y, max_y]
                    self._roi_bounds_changed = True
                    self.schedule_roi_edges_update(self._roi_live_delay_ms())
                    self.update()
                return
             
            # 0.6 ROI ?몃뱾 ?쒕옒洹?泥섎━
            if self.roi_enabled and self.active_roi_edge and self.active_roi_edge != "move" and self.mouse_button == Qt.MouseButton.LeftButton:
                # XY ?됰㈃?쇰줈 ?ъ쁺?섏뿬 留덉슦???붾뱶 醫뚰몴 ?띾뱷 (z=0?쇰줈 媛??
                # pick_point_on_mesh??硫붿돩 ?쒕㈃??李띿쑝誘濡? ?ш린?쒕뒗 ?⑥닚???덉씠-?됰㈃ 援먯감 ?ъ슜
                # ?꾩????꾪빐 pick_point_on_mesh ?쒖슜 媛?ν븯??諛붾떏????怨좊젮
                ray_origin, ray_dir = self.get_ray(event.pos().x(), event.pos().y())
                if ray_origin is None or ray_dir is None:
                    return
                # Plane: Z=0, Normal=[0,0,1]
                denom = ray_dir[2]
                if abs(denom) > 1e-6:
                    t = -ray_origin[2] / denom
                    hit_pt = ray_origin + t * ray_dir
                    wx, wy = hit_pt[0], hit_pt[1]
                    
                    if self.active_roi_edge == 'left':
                        self.roi_bounds[0] = min(wx, self.roi_bounds[1] - 0.1)
                    elif self.active_roi_edge == 'right':
                        self.roi_bounds[1] = max(wx, self.roi_bounds[0] + 0.1)
                    elif self.active_roi_edge == 'bottom':
                        self.roi_bounds[2] = min(wy, self.roi_bounds[3] - 0.1)
                    elif self.active_roi_edge == 'top':
                        self.roi_bounds[3] = max(wy, self.roi_bounds[2] + 0.1)
                     
                    axis_hint = self._roi_edge_to_axis(self.active_roi_edge)
                    if axis_hint in ("x", "y"):
                        self._roi_last_adjust_axis = axis_hint
                        self._roi_commit_axis_hint = axis_hint
                    plane_hint = self._roi_edge_to_plane(self.active_roi_edge)
                    if plane_hint in ("x1", "x2", "y1", "y2"):
                        self._roi_last_adjust_plane = plane_hint
                        self._roi_commit_plane_hint = plane_hint
                    self._roi_bounds_changed = True
                    self.schedule_roi_edges_update(self._roi_live_delay_ms())
                    self.update()
                return

            # 4. ?쇰컲 移대찓??議곗옉 (?쒕옒洹?
            ortho_locked = bool(getattr(self, "_front_back_ortho_enabled", False))
            if self.mouse_button == Qt.MouseButton.LeftButton:
                # 6-face view starts axis-aligned, but dragging should immediately return to free orbit.
                if ortho_locked:
                    self._front_back_ortho_enabled = False
                self.camera.rotate(dx, dy)
                self.update()
            elif self.mouse_button == Qt.MouseButton.RightButton:
                # Right-drag should also return to free camera mode.
                if ortho_locked:
                    self._front_back_ortho_enabled = False
                self.camera.pan(dx, dy)
                self.update()
            elif self.mouse_button == Qt.MouseButton.MiddleButton:
                if ortho_locked:
                    self._front_back_ortho_enabled = False
                self.camera.rotate(dx, dy)
                self.update()
            
        except Exception:
            _log_ignored_exception("Mouse move error", level=logging.WARNING)
    
    def _calculate_gizmo_angle(self, screen_x, screen_y):
        """湲곗쫰紐?以묒떖 湲곗? 2D ?붾㈃ 怨듦컙?먯꽌??媛곷룄 怨꾩궛 (媛??吏곴??곸씤 ?먰삎 ?쒕옒洹?諛⑹떇)"""
        obj = self.selected_obj
        if not obj or not self.active_gizmo_axis:
            return None
        
        try:
            # 罹먯떆??留ㅽ듃由?뒪媛 ?덉쑝硫??ъ슜 (?깅뒫 理쒖쟻??
            if self._cached_viewport is not None:
                viewport = self._cached_viewport
                modelview = self._cached_modelview
                projection = self._cached_projection
            else:
                self.makeCurrent()
                viewport = glGetIntegerv(GL_VIEWPORT)
                modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
                projection = glGetDoublev(GL_PROJECTION_MATRIX)
            
            # 湲곗쫰紐?以묒떖(?ㅻ툕?앺듃 ?꾩튂)???붾㈃?쇰줈 ?ъ쁺
            obj_pos = obj.translation
            win_pos = gluProject(obj_pos[0], obj_pos[1], obj_pos[2], modelview, projection, viewport)
            if not win_pos:
                return None

            center_x, center_y = self._gl_window_to_qt_xy(float(win_pos[0]), float(win_pos[1]), viewport=viewport)

            # 以묒떖?먯뿉??留덉슦???ъ씤?곌퉴吏??2D 媛곷룄 (atan2)
            # ?붾㈃ 醫뚰몴怨꾨뒗 Y媛 ?꾨옒濡?利앷??섎?濡?遺??二쇱쓽
            dx = float(screen_x) - float(center_x)
            dy = float(screen_y) - float(center_y)

            angle = np.arctan2(dy, dx)
            return angle
        except Exception:
            return None
    
    def wheelEvent(self, a0: QWheelEvent | None):
        if a0 is None:
            return
        event = a0
        """留덉슦???? 湲곕낯 以? Ctrl+?좎? ?щ씪?댁뒪 ?ㅼ틪."""
        try:
            mods = event.modifiers()
            if bool(getattr(self, "slice_enabled", False)) and bool(mods & Qt.KeyboardModifier.ControlModifier):
                steps = float(event.angleDelta().y()) / 120.0
                if abs(steps) > 1e-9:
                    # Default 0.1cm, Shift=fine(0.02cm), Alt=coarse(0.5cm)
                    if bool(mods & Qt.KeyboardModifier.ShiftModifier):
                        step_cm = 0.02
                    elif bool(mods & Qt.KeyboardModifier.AltModifier):
                        step_cm = 0.5
                    else:
                        step_cm = 0.1
                    self.sliceScanRequested.emit(float(steps * step_cm))
                    event.accept()
                    return
        except Exception:
            _log_ignored_exception()

        # fallback: camera zoom
        delta = event.angleDelta().y()
        if bool(getattr(self, "_front_back_ortho_enabled", False)):
            steps = float(delta) / 120.0
            if abs(steps) > 1e-9:
                try:
                    scale = float(getattr(self, "_ortho_view_scale", ORTHO_VIEW_SCALE_DEFAULT) or ORTHO_VIEW_SCALE_DEFAULT)
                except Exception:
                    scale = ORTHO_VIEW_SCALE_DEFAULT
                scale = float(scale * (1.10 ** (-steps)))
                self._ortho_view_scale = float(max(0.2, min(scale, 40.0)))
            self.update()
            return

        self._front_back_ortho_enabled = False
        self.camera.zoom(delta)
        self.update()

    def keyPressEvent(self, a0: QKeyEvent | None):
        if a0 is None:
            return
        event = a0
        """?ㅻ낫???낅젰"""
        self.keys_pressed.add(event.key())
        if event.key() in (Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D, Qt.Key.Key_Q, Qt.Key.Key_E):
            if not self.move_timer.isActive():
                self.move_timer.start()

        # 0. ?щ씪?댁뒪 ?⑥텞??(?쒕㈃ 遺꾨━ ?묒뾽 以?鍮좊Ⅸ ?⑤㈃ ?ㅼ틪/珥ъ쁺)
        if bool(getattr(self, "slice_enabled", False)):
            key = event.key()
            mods = event.modifiers()
            if key in (Qt.Key.Key_Comma, Qt.Key.Key_Period):
                if bool(mods & Qt.KeyboardModifier.ShiftModifier):
                    step_cm = 0.02
                elif bool(mods & Qt.KeyboardModifier.AltModifier):
                    step_cm = 0.5
                else:
                    step_cm = 0.1
                sign = -1.0 if key == Qt.Key.Key_Comma else 1.0
                try:
                    self.sliceScanRequested.emit(float(sign * step_cm))
                    self.status_info = f"?⑤㈃ ?ㅼ틪 ?ㅽ뀦 {sign * step_cm:+.2f}cm (, / .)"
                    self.update()
                except Exception:
                    _log_ignored_exception()
                return
            if key == Qt.Key.Key_C and not bool(mods & Qt.KeyboardModifier.ControlModifier):
                try:
                    z_now = float(getattr(self, "slice_z", 0.0) or 0.0)
                except Exception:
                    z_now = 0.0
                try:
                    self.sliceCaptureRequested.emit(float(z_now))
                    self.status_info = f"?벝 ?⑤㈃ 珥ъ쁺 ?붿껌 (Z={z_now:.2f}cm, C)"
                    self.update()
                except Exception:
                    _log_ignored_exception()
                return

        # 0. ?⑤㈃??2媛? ?꾧뎄 ?⑥텞??
        if self.picking_mode == 'cut_lines':
            key = event.key()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._finish_cut_lines_current()
                return
            if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
                try:
                    idx = int(getattr(self, "cut_line_active", 0))
                    idx = idx if idx in (0, 1) else 0
                    line = self.cut_lines[idx]
                    if line:
                        line.pop()
                        self.cut_section_profiles[idx] = []
                        self.cut_section_world[idx] = []
                        try:
                            self._cut_line_final[idx] = False
                        except Exception:
                            _log_ignored_exception()
                    self.cut_line_drawing = bool(line)
                    self.cut_line_preview = None
                    if len(line) >= 2:
                        self.schedule_cut_section_update(idx, delay_ms=150)
                    self.update()
                except Exception:
                    _log_ignored_exception()
                return
            if key == Qt.Key.Key_Tab:
                try:
                    self.cut_line_active = 1 - int(getattr(self, "cut_line_active", 0))
                    try:
                        self.cutLineActiveChanged.emit(int(self.cut_line_active))
                    except Exception:
                        _log_ignored_exception()
                    self.cut_line_preview = None
                    idx = int(getattr(self, "cut_line_active", 0))
                    idx = idx if idx in (0, 1) else 0
                    line = self.cut_lines[idx]
                    final = getattr(self, "_cut_line_final", [False, False])
                    self.cut_line_drawing = bool(line) and not bool(final[idx])
                    try:
                        role = "Length" if idx == 0 else "Width"
                        locks = getattr(self, "cut_line_axis_lock", [None, None]) or [None, None]
                        lk = locks[idx] if idx < len(locks) else None
                        axis_txt = "X" if str(lk).lower().startswith("x") else ("Y" if str(lk).lower().startswith("y") else "X/Y")
                        self.status_info = f"?㎛ ?쒖꽦 ?⑤㈃?? {role} ({axis_txt}異??ㅻ깄)"
                    except Exception:
                        _log_ignored_exception()
                    self.update()
                except Exception:
                    _log_ignored_exception()
                return
            if key == Qt.Key.Key_Escape:
                # Cancel the entire section-line process.
                self.clear_cut_lines()
                self.set_cut_lines_enabled(False)
                self.status_info = "?썞 ?⑤㈃???낅젰??痍⑥냼?섏뿀?듬땲??"
                self.update()
                return

        # 0.05 ROI ?⑤㈃ 誘몃━蹂닿린 ?뺤젙(Enter): ?꾩옱 蹂댁씠??ROI ?⑤㈃???덉씠?대줈 諛곗튂
        if bool(getattr(self, "roi_enabled", False)) and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            try:
                axis_hint = self._roi_edge_to_axis(getattr(self, "active_roi_edge", None))
                if axis_hint not in ("x", "y"):
                    last_axis = str(getattr(self, "_roi_last_adjust_axis", "") or "").strip().lower()
                    if last_axis in ("x", "y"):
                        axis_hint = last_axis
                self._roi_commit_axis_hint = axis_hint if axis_hint in ("x", "y") else None
                plane_hint = self._roi_edge_to_plane(getattr(self, "active_roi_edge", None))
                if plane_hint not in ("x1", "x2", "y1", "y2"):
                    last_plane = str(getattr(self, "_roi_last_adjust_plane", "") or "").strip().lower()
                    if last_plane in ("x1", "x2", "y1", "y2"):
                        plane_hint = last_plane
                self._roi_commit_plane_hint = plane_hint if plane_hint in ("x1", "x2", "y1", "y2") else None
                self.roiSectionCommitRequested.emit()
                self.status_info = "?뱦 ROI ?⑤㈃ 諛곗튂 ?붿껌"
                self.update()
            except Exception:
                _log_ignored_exception()
            return

        # 0.1 ?섎윭??吏???곸뿭) ?꾧뎄 ?⑥텞??
        if self.picking_mode == "paint_surface_area":
            key = event.key()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._finish_surface_lasso(event.modifiers(), seed_pos=getattr(self, "surface_lasso_preview", None))
                return
            if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
                try:
                    if getattr(self, "surface_lasso_points", None):
                        self.surface_lasso_points.pop()
                    if getattr(self, "surface_lasso_face_indices", None):
                        self.surface_lasso_face_indices.pop()
                    self.update()
                except Exception:
                    _log_ignored_exception()
                return
            if key == Qt.Key.Key_Escape:
                self.clear_surface_lasso()
                self.picking_mode = "none"
                if not bool(getattr(self, "cut_lines_enabled", False)):
                    self.setMouseTracking(False)
                self.status_info = "Area selection canceled"
                self.update()
                return

        # 0.2 寃쎄퀎(硫댁쟻+?먯꽍) ?ш?誘??꾧뎄 ?⑥텞??
        if self.picking_mode == "paint_surface_magnetic":
            key = event.key()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._finish_surface_lasso(event.modifiers(), seed_pos=getattr(self, "surface_lasso_preview", None))
                return
            if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
                try:
                    if getattr(self, "surface_lasso_points", None):
                        self.surface_lasso_points.pop()
                    if getattr(self, "surface_lasso_face_indices", None):
                        self.surface_lasso_face_indices.pop()
                    self.update()
                except Exception:
                    _log_ignored_exception()
                return
            if key == Qt.Key.Key_Escape:
                self.clear_surface_lasso()
                self.clear_surface_magnetic_lasso(clear_cache=False)
                self.picking_mode = "none"
                if not bool(getattr(self, "cut_lines_enabled", False)):
                    self.setMouseTracking(False)
                self.status_info = "Boundary selection canceled"
                self.update()
                return

        # 0.9 ESC: ROI ?뱀뀡 紐⑤뱶 痍⑥냼
        if event.key() == Qt.Key.Key_Escape and bool(getattr(self, "roi_enabled", False)):
            try:
                self.roi_enabled = False
                self.active_roi_edge = None
                self.roi_rect_dragging = False
                self.roi_rect_start = None
                self._roi_move_dragging = False
                self._roi_move_last_xy = None
                self._roi_bounds_changed = False
                self._roi_commit_axis_hint = None
                self._roi_last_adjust_axis = None
                self._roi_commit_plane_hint = None
                self._roi_last_adjust_plane = None
                self.status_info = "?썞 ROI ?뱀뀡??痍⑥냼?섏뿀?듬땲??"
                self.update()
            except Exception:
                _log_ignored_exception()
            return

        # 1. Backspace/Delete: 諛붾떏吏?????섎굹 ?섎룎由ш린
        if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            if self.picking_mode == 'floor_3point':
                if self.floor_picks:
                    self.floor_picks.pop()
                    n = int(len(self.floor_picks))
                    if n <= 0:
                        self.status_info = "諛붾떏吏???먯씠 紐⑤몢 吏?뚯죱?듬땲??"
                    elif n < 3:
                        self.status_info = f"諛붾떏吏????{n}媛? 3媛??댁긽 ?꾩슂?⑸땲??"
                    else:
                        self.status_info = f"諛붾떏吏????{n}媛? Enter/?고겢由?쑝濡??뺤젙?섏꽭??"
                    self.update()
                return

        # 2. Enter/Return ?? 諛붾떏 ?뺣젹 ?뺤젙
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.picking_mode in ('floor_3point', 'floor_face', 'floor_brush'):
                self.floorAlignmentConfirmed.emit()
                return

        # 3. ESC: ?묒뾽 痍⑥냼
        if event.key() == Qt.Key.Key_Escape:
            if self.picking_mode != 'none':
                self.picking_mode = 'none'
                self.floor_picks = []
                self._surface_paint_left_press_pos = None
                self._surface_paint_left_dragged = False
                self.status_info = "Task canceled"
                self.update()
                return
        
        # 4. Ctrl+Z: Undo
        if event.key() == Qt.Key.Key_Z and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.undo()
            return

        # 湲곗쫰紐⑤굹 移대찓??酉?愿????
        key = event.key()
        if key == Qt.Key.Key_R:
            self._front_back_ortho_enabled = False
            self.camera.reset()
            self.update()
        elif key == Qt.Key.Key_F:
            self._front_back_ortho_enabled = False
            self.fit_view_to_selected_object()
        else:
            key_dec = getattr(Qt.Key, "Key_BracketLeft", -1)
            key_inc = getattr(Qt.Key, "Key_BracketRight", -1)
            if key in (key_dec, key_inc):
                # ?쒕㈃ 吏???꾧뎄?먯꽌??[ ] ?ㅻ줈 釉뚮윭??李띻린 ?ш린瑜?議곗젅?⑸땲??
                if self.picking_mode in {"paint_surface_brush", "paint_surface_face"}:
                    attr = (
                        "_surface_brush_radius_px"
                        if self.picking_mode == "paint_surface_brush"
                        else "_surface_click_radius_px"
                    )
                    label = "Brush" if self.picking_mode == "paint_surface_brush" else "Click"
                    try:
                        cur = float(getattr(self, attr, 48.0) or 48.0)
                    except Exception:
                        cur = 48.0
                    factor = 0.9 if key == key_dec else (1.0 / 0.9)
                    new = float(cur) * float(factor)
                    new = float(max(2.0, min(new, 600.0)))
                    try:
                        setattr(self, attr, new)
                    except Exception:
                        _log_ignored_exception()
                    self.status_info = f"?뼂截??쒕㈃ {label} ?ш린: {new:.0f}px ([ / ] 議곗젅)"
                    self.update()
                    return

                # 寃쎄퀎(硫댁쟻+?먯꽍) ?꾧뎄?먯꽌??[ ] ?ㅻ줈 ?ㅻ깄 諛섍꼍??議곗젅?⑸땲??
                if self.picking_mode == "paint_surface_magnetic":
                    try:
                        cur = float(getattr(self, "_surface_magnetic_snap_radius_px", 14.0) or 14.0)
                    except Exception:
                        cur = 14.0
                    factor = 0.9 if key == key_dec else (1.0 / 0.9)
                    new = float(cur) * float(factor)
                    new = float(max(2.0, min(new, 200.0)))
                    try:
                        self._surface_magnetic_snap_radius_px = int(round(new))
                    except Exception:
                        _log_ignored_exception()
                    self.status_info = f"?㎠ ?먯꽍 諛섍꼍: {new:.0f}px ([ / ] 議곗젅)"
                    self.update()
                    return

                # 湲곕낯: [ ] ?ㅻ줈 湲곗쫰紐??ш린 議곗젅
                if key == key_dec:
                    self.gizmo_radius_factor = max(0.60, float(self.gizmo_radius_factor) * 0.9)
                    self.update_gizmo_size()
                    self.status_info = f"? 湲곗쫰紐??ш린: x{self.gizmo_radius_factor:.2f}"
                    self.update()
                elif key == key_inc:
                    self.gizmo_radius_factor = min(2.5, float(self.gizmo_radius_factor) / 0.9)
                    self.update_gizmo_size()
                    self.status_info = f"? 湲곗쫰紐??ш린: x{self.gizmo_radius_factor:.2f}"
                    self.update()
              
        super().keyPressEvent(event)

    def keyReleaseEvent(self, a0: QKeyEvent | None):
        if a0 is None:
            return
        event = a0
        """????泥섎━"""
        if event.key() in self.keys_pressed:
            self.keys_pressed.remove(event.key())
        if not (self.keys_pressed & {Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D, Qt.Key.Key_Q, Qt.Key.Key_E}):
            if self.move_timer.isActive():
                self.move_timer.stop()
        super().keyReleaseEvent(event)

    def process_keyboard_navigation(self):
        """WASD ?곗냽 ?대룞 泥섎━"""
        if not self.keys_pressed:
            return
            
        dx, dy, dz = 0, 0, 0
        if Qt.Key.Key_W in self.keys_pressed:
            dz += 1
        if Qt.Key.Key_S in self.keys_pressed:
            dz -= 1
        if Qt.Key.Key_A in self.keys_pressed:
            dx -= 1
        if Qt.Key.Key_D in self.keys_pressed:
            dx += 1
        if Qt.Key.Key_Q in self.keys_pressed:
            dy += 1
        if Qt.Key.Key_E in self.keys_pressed:
            dy -= 1
        
        if dx != 0 or dy != 0 or dz != 0:
            self.camera.move_relative(dx, dy, dz)
            self.update()
    
    def _hit_test_roi(self, pos):
        """ROI ?몃뱾(?붿궡??以묒븰) ?대┃ 寃??(screen-space ?곗꽑)."""
        # 1) Screen-space hit test (stable regardless of mesh scale / zoom).
        try:
            self.makeCurrent()
            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)

            x1, x2, y1, y2 = self.roi_bounds
            try:
                x1, x2 = (float(x1), float(x2)) if float(x1) <= float(x2) else (float(x2), float(x1))
                y1, y2 = (float(y1), float(y2)) if float(y1) <= float(y2) else (float(y2), float(y1))
            except Exception:
                pass

            mid_x = (float(x1) + float(x2)) / 2.0
            mid_y = (float(y1) + float(y2)) / 2.0
            z = 0.08
            handle_offset = max(1.0, float(getattr(self.camera, "distance", 50.0) or 50.0) * 0.010)

            handles = {
                "bottom": (mid_x, float(y1) - handle_offset, z),
                "top": (mid_x, float(y2) + handle_offset, z),
                "left": (float(x1) - handle_offset, mid_y, z),
                "right": (float(x2) + handle_offset, mid_y, z),
                "move": (mid_x, mid_y, z),
            }

            try:
                thr = int(getattr(self, "_roi_handle_hit_px", 52) or 52)
            except Exception:
                thr = 52
            thr = max(12, min(thr, 180))
            edge_radius = float(thr + max(10, int(thr * 0.35)))
            move_radius = float(max(14, int(thr * 0.85)))

            def _pick_one(keys: tuple[str, ...], radius_px: float) -> str | None:
                best_key = None
                best_d2 = float("inf")
                rr = float(radius_px * radius_px)
                for edge in keys:
                    try:
                        wx, wy, wz = handles[edge]
                        win = gluProject(float(wx), float(wy), float(wz), modelview, projection, viewport)
                        if not win:
                            continue
                        qx, qy = self._gl_window_to_qt_xy(float(win[0]), float(win[1]), viewport=viewport)
                        dx = float(pos.x()) - float(qx)
                        dy = float(pos.y()) - float(qy)
                        d2 = float(dx * dx + dy * dy)
                        if d2 <= rr and d2 < best_d2:
                            best_key = str(edge)
                            best_d2 = d2
                    except Exception:
                        continue
                return best_key

            # Edge handles first: "move" ?몃뱾???붿궡???좏깮??媛濡쒖콈吏 ?딅룄濡??곗꽑?쒖쐞 遺??
            picked_edge = _pick_one(("bottom", "top", "left", "right"), edge_radius)
            if picked_edge is not None:
                return picked_edge
            picked_move = _pick_one(("move",), move_radius)
            if picked_move is not None:
                return picked_move
        except Exception:
            _log_ignored_exception()

        # 2) Fallback: world-space hit test on Z=0 plane (previous behavior)
        try:
            ray_origin, ray_dir = self.get_ray(pos.x(), pos.y())
            if ray_origin is None or ray_dir is None:
                return None
            if abs(float(ray_dir[2])) < 1e-6:
                return None
            t = -float(ray_origin[2]) / float(ray_dir[2])
            hit_pt = ray_origin + t * ray_dir
            wx, wy = float(hit_pt[0]), float(hit_pt[1])

            x1, x2, y1, y2 = self.roi_bounds
            try:
                x1, x2 = (float(x1), float(x2)) if float(x1) <= float(x2) else (float(x2), float(x1))
                y1, y2 = (float(y1), float(y2)) if float(y1) <= float(y2) else (float(y2), float(y1))
            except Exception:
                pass
            mid_x = (float(x1) + float(x2)) / 2.0
            mid_y = (float(y1) + float(y2)) / 2.0

            threshold = float(getattr(self.camera, "distance", 50.0) or 50.0) * 0.07
            off = max(1.0, float(getattr(self.camera, "distance", 50.0) or 50.0) * 0.010)
            if np.hypot(wx - mid_x, wy - (float(y1) - off)) < threshold:
                return "bottom"
            if np.hypot(wx - mid_x, wy - (float(y2) + off)) < threshold:
                return "top"
            if np.hypot(wx - (float(x1) - off), wy - mid_y) < threshold:
                return "left"
            if np.hypot(wx - (float(x2) + off), wy - mid_y) < threshold:
                return "right"
            if np.hypot(wx - mid_x, wy - mid_y) < threshold:
                return "move"
        except Exception:
            return None

        return None

    def capture_high_res_image(
        self,
        width: int = 2048,
        height: int = 2048,
        *,
        only_selected: bool = False,
        orthographic: bool = False,
    ):
        """怨좏빐?곷룄 ?ㅽ봽?ㅽ겕由??뚮뜑留?
        `orthographic=True`瑜??ъ슜?섎㈃ ?뺤궗??glOrtho)?쇰줈 罹≪쿂?섏뿬 1:1 ?ㅼ????꾨㈃???좊━?⑸땲??
        """
        self.makeCurrent()
        prev_viewport = None
        try:
            prev_viewport = glGetIntegerv(GL_VIEWPORT)
        except Exception:
            prev_viewport = None

        # 1. FBO ?앹꽦
        fbo = QOpenGLFramebufferObject(width, height, QOpenGLFramebufferObject.Attachment.Depth)
        fbo.bind()
        
        # 2. ?뚮뜑留??ㅼ젙 (?ㅽ봽?ㅽ겕由곗슜)
        glViewport(0, 0, width, height)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        aspect = width / height
        if not orthographic:
            gluPerspective(45.0, aspect, 0.1, 1000000.0)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        
        # 3. 洹몃━湲?(UI ?쒖쇅?섍퀬 源⑤걮?섍쾶)
        glClearColor(1.0, 1.0, 1.0, 1.0) # ?붿씠??諛곌꼍
        glDepthMask(GL_TRUE)
        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
        glLoadIdentity()
        
        self.camera.apply()

        if orthographic:
            try:
                mv_raw = glGetDoublev(GL_MODELVIEW_MATRIX)
                mv = np.asarray(mv_raw, dtype=np.float64).reshape(4, 4).T

                sel = int(self.selected_index) if self.selected_index is not None else -1
                bounds_list = []
                for i, obj in enumerate(self.objects):
                    if not obj.visible:
                        continue
                    if only_selected and sel >= 0 and i != sel:
                        continue
                    try:
                        b = np.asarray(obj.get_world_bounds(), dtype=np.float64)
                        if b.shape == (2, 3):
                            bounds_list.append(b)
                    except Exception:
                        continue

                if bounds_list:
                    wb = np.vstack(bounds_list)
                    w_min = wb.min(axis=0)
                    w_max = wb.max(axis=0)
                else:
                    w_min = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
                    w_max = np.array([1.0, 1.0, 1.0], dtype=np.float64)

                corners = np.array(
                    [
                        [w_min[0], w_min[1], w_min[2]],
                        [w_max[0], w_min[1], w_min[2]],
                        [w_min[0], w_max[1], w_min[2]],
                        [w_max[0], w_max[1], w_min[2]],
                        [w_min[0], w_min[1], w_max[2]],
                        [w_max[0], w_min[1], w_max[2]],
                        [w_min[0], w_max[1], w_max[2]],
                        [w_max[0], w_max[1], w_max[2]],
                    ],
                    dtype=np.float64,
                )
                v_h = np.hstack([corners, np.ones((8, 1), dtype=np.float64)])
                eye = v_h @ mv.T

                x_min, x_max = float(np.min(eye[:, 0])), float(np.max(eye[:, 0]))
                y_min, y_max = float(np.min(eye[:, 1])), float(np.max(eye[:, 1]))
                z_min, z_max = float(np.min(eye[:, 2])), float(np.max(eye[:, 2]))

                pad_x = max(1e-6, (x_max - x_min) * 0.05)
                pad_y = max(1e-6, (y_max - y_min) * 0.05)
                x_min -= pad_x
                x_max += pad_x
                y_min -= pad_y
                y_max += pad_y

                world_w = max(1e-6, x_max - x_min)
                world_h = max(1e-6, y_max - y_min)
                target_aspect = float(aspect) if aspect > 1e-9 else 1.0
                cur_aspect = float(world_w / world_h)
                if cur_aspect > target_aspect:
                    new_h = world_w / target_aspect
                    d = (new_h - world_h) * 0.5
                    y_min -= d
                    y_max += d
                else:
                    new_w = world_h * target_aspect
                    d = (new_w - world_w) * 0.5
                    x_min -= d
                    x_max += d

                pad_z = max(1e-6, (z_max - z_min) * 0.1)
                near = max(0.1, float(-z_max - pad_z))
                far = max(near + 0.1, float(-z_min + pad_z))

                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                glOrtho(float(x_min), float(x_max), float(y_min), float(y_max), float(near), float(far))
                glMatrixMode(GL_MODELVIEW)
            except Exception:
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(45.0, aspect, 0.1, 1000000.0)
                glMatrixMode(GL_MODELVIEW)
        
        # 愿묒썝
        glEnable(GL_LIGHTING)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.4, 0.4, 0.4, 1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.6, 0.6, 0.6, 1.0])
        
        # 硫붿돩留??뚮뜑留?(洹몃━?쒕굹 HUD ?쒖쇅)
        sel = int(self.selected_index) if self.selected_index is not None else -1
        for i, obj in enumerate(self.objects):
            if not obj.visible:
                continue
            if only_selected and sel >= 0 and i != sel:
                continue
            self.draw_scene_object(obj, is_selected=(i == sel))
        
        # 4. ?됰젹 罹≪쿂 (SVG ?ъ쁺 ?뺣젹??
        mv = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj = glGetDoublev(GL_PROJECTION_MATRIX)
        vp = glGetIntegerv(GL_VIEWPORT)
        
        glFlush()
        qimage = fbo.toImage()
        
        # 5. 蹂듦뎄
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        fbo.release()

        # ?먮옒 酉고룷??蹂듦뎄
        try:
            if prev_viewport is not None:
                vx0, vy0, vw0, vh0 = [int(v) for v in prev_viewport[:4]]
                glViewport(vx0, vy0, vw0, vh0)
            else:
                glViewport(0, 0, int(self.width()), int(self.height()))
        except Exception:
            try:
                glViewport(0, 0, int(self.width()), int(self.height()))
            except Exception:
                pass
        self.update()

        return qimage, mv, proj, vp

    def _gl_viewport_and_scale(self, viewport=None) -> tuple[tuple[int, int, int, int], float, float]:
        """Return (vx,vy,vw,vh) and (scale_x, scale_y) between Qt widget coords and GL window coords.

        Qt mouse events are in widget coordinates (logical pixels). OpenGL window coords are in the
        current GL viewport (often device pixels on HiDPI). This helper keeps picking consistent.
        """
        if viewport is None:
            try:
                self.makeCurrent()
            except Exception:
                pass
            try:
                viewport = glGetIntegerv(GL_VIEWPORT)
            except Exception:
                viewport = None

        try:
            vx, vy, vw, vh = [int(v) for v in (viewport[:4] if viewport is not None else [])]
        except Exception:
            vx, vy, vw, vh = 0, 0, int(self.width()), int(self.height())
        vw = max(1, int(vw))
        vh = max(1, int(vh))

        w = max(1, int(self.width()))
        h = max(1, int(self.height()))
        sx = float(vw) / float(w) if w > 0 else 1.0
        sy = float(vh) / float(h) if h > 0 else 1.0
        if not (np.isfinite(sx) and sx > 0.0):
            sx = 1.0
        if not (np.isfinite(sy) and sy > 0.0):
            sy = 1.0
        return (vx, vy, vw, vh), sx, sy

    def _qt_to_gl_window_xy(self, screen_x: float, screen_y: float, *, viewport=None) -> tuple[int, int]:
        """Convert Qt widget coords (top-left origin) to GL window coords (bottom-left origin)."""
        (vx, vy, vw, vh), sx, sy = self._gl_viewport_and_scale(viewport)
        try:
            gx = float(screen_x) * sx + float(vx)
            gy = float(vy) + float(vh - 1) - float(screen_y) * sy
        except Exception:
            gx = float(vx)
            gy = float(vy)
        gx_i = int(np.clip(int(round(gx)), vx, vx + vw - 1))
        gy_i = int(np.clip(int(round(gy)), vy, vy + vh - 1))
        return gx_i, gy_i

    def _gl_window_to_qt_xy(self, gl_x: float, gl_y: float, *, viewport=None) -> tuple[float, float]:
        """Convert GL window coords (bottom-left origin) to Qt widget coords (top-left origin)."""
        (vx, vy, vw, vh), sx, sy = self._gl_viewport_and_scale(viewport)
        try:
            qx = (float(gl_x) - float(vx)) / sx
            qy = (float(vy + vh - 1) - float(gl_y)) / sy
            return float(qx), float(qy)
        except Exception:
            return float(gl_x), float(gl_y)

    def get_ray(self, screen_x: int, screen_y: int):
        """?붾㈃ 醫뚰몴?먯꽌 ?붾뱶 ?덉씠(origin, dir) 怨꾩궛"""
        try:
            self.makeCurrent()

            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            gl_x, gl_y = self._qt_to_gl_window_xy(float(screen_x), float(screen_y), viewport=viewport)

            near_pt = gluUnProject(gl_x, gl_y, 0.0, modelview, projection, viewport)
            far_pt = gluUnProject(gl_x, gl_y, 1.0, modelview, projection, viewport)
            if not near_pt or not far_pt:
                return None, None

            ray_origin = np.array(near_pt, dtype=float)
            ray_dir = np.array(far_pt, dtype=float) - ray_origin
            norm = float(np.linalg.norm(ray_dir))
            if norm < 1e-12:
                return None, None
            ray_dir /= norm

            return ray_origin, ray_dir
        except Exception:
            return None, None

    def pick_point_on_plane_z(self, screen_x: int, screen_y: int, z: float = 0.0):
        """?붾㈃ 醫뚰몴?먯꽌 Z=z ?됰㈃怨쇱쓽 援먯젏(?붾뱶 醫뚰몴) 怨꾩궛"""
        ray_origin, ray_dir = self.get_ray(screen_x, screen_y)
        if ray_origin is None or ray_dir is None:
            return None
        denom = float(ray_dir[2])
        if abs(denom) < 1e-8:
            return None
        t = (float(z) - float(ray_origin[2])) / denom
        if t < 0:
            return None
        return ray_origin + t * ray_dir
        
    def pick_point_on_mesh_info(self, screen_x: int, screen_y: int):
        """
        ?붾㈃ 醫뚰몴瑜?硫붿돩 ?쒕㈃??3D 醫뚰몴濡?蹂?섑빀?덈떎.

        Returns:
            (pt_world, depth_value, gl_x, gl_y, viewport, modelview, projection) ?먮뒗 None
        """
        if not self.objects:
            return None
        obj = self.selected_obj
        if not obj:
            return None

        # OpenGL 酉고룷?? ?ъ쁺, 紐⑤뜽酉??됰젹 媛?몄삤湲?
        self.makeCurrent()

        viewport = glGetIntegerv(GL_VIEWPORT)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        gl_x, gl_y = self._qt_to_gl_window_xy(float(screen_x), float(screen_y), viewport=viewport)

        def is_bg(depth_val: float) -> bool:
            try:
                return (not bool(np.isfinite(depth_val))) or float(depth_val) >= 1.0
            except Exception:
                return True

        # 源딆씠 踰꾪띁?먯꽌 源딆씠 媛??쎄린
        depth = cast(Any, glReadPixels(int(gl_x), int(gl_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT))
        try:
            depth_value = float(depth[0][0])
        except Exception:
            depth_value = float("nan")

        # 諛곌꼍???대┃??寃쎌슦: 洹쇱쿂 ?쎌? depth瑜??먯깋?댁꽌 ?쇳궧??蹂댁젙
        if is_bg(depth_value):
            try:
                r = int(getattr(self, "_pick_search_radius_px", 0) or 0)
            except Exception:
                r = 0
            if r > 0:
                try:
                    vx, vy, vw, vh = [int(v) for v in viewport[:4]]
                except Exception:
                    vx, vy, vw, vh = 0, 0, int(self.width()), int(self.height())
                vw = max(1, vw)
                vh = max(1, vh)

                cx = int(gl_x)
                cy = int(gl_y)
                x0 = int(max(vx, cx - int(r)))
                y0 = int(max(vy, cy - int(r)))
                x1 = int(min(vx + vw - 1, cx + int(r)))
                y1 = int(min(vy + vh - 1, cy + int(r)))
                w = int(max(1, x1 - x0 + 1))
                h = int(max(1, y1 - y0 + 1))

                try:
                    depth_raw = cast(Any, glReadPixels(x0, y0, w, h, GL_DEPTH_COMPONENT, GL_FLOAT))
                    depth_arr = np.asarray(depth_raw, dtype=np.float32)
                    depth_arr = np.squeeze(depth_arr)
                    if depth_arr.ndim != 2:
                        depth_arr = depth_arr.reshape((h, w))
                except Exception:
                    depth_arr = None

                if depth_arr is not None and depth_arr.size:
                    mask = np.isfinite(depth_arr) & (depth_arr < 1.0)
                    if bool(np.any(mask)):
                        ys, xs = np.where(mask)
                        dx = (xs.astype(np.int32) + int(x0)) - int(cx)
                        dy = (ys.astype(np.int32) + int(y0)) - int(cy)
                        dist2 = dx.astype(np.int64) * dx.astype(np.int64) + dy.astype(np.int64) * dy.astype(np.int64)
                        k = int(np.argmin(dist2))
                        gl_x = int(x0 + int(xs[k]))
                        gl_y = int(y0 + int(ys[k]))
                        try:
                            depth_value = float(depth_arr[int(ys[k]), int(xs[k])])
                        except Exception:
                            depth_value = float("nan")

        if is_bg(depth_value):
            return None

        # Convert screen coordinates to world coordinates.
        world_x, world_y, world_z = gluUnProject(
            float(gl_x),
            float(gl_y),
            float(depth_value),
            modelview,
            projection,
            viewport,
        )

        pt = np.array([world_x, world_y, world_z], dtype=np.float64)

        # Extra guard: ensure the picked point is near the selected object's world bounds.
        # This prevents accidental picks on other geometry / stale depth artifacts.
        try:
            wb = np.asarray(obj.get_world_bounds(), dtype=np.float64)
            if wb.shape == (2, 3) and np.isfinite(wb).all() and np.isfinite(pt).all():
                ext = wb[1] - wb[0]
                max_dim = float(np.max(np.abs(ext))) if ext.size == 3 else 0.0
                margin = max(0.05, max_dim * 0.005)  # ~0.5mm@cm-units; relative for large meshes
                if np.any(pt < (wb[0] - margin)) or np.any(pt > (wb[1] + margin)):
                    return None
        except Exception:
            _log_ignored_exception()

        return (
            pt,
            float(depth_value),
            int(gl_x),
            int(gl_y),
            viewport,
            modelview,
            projection,
        )

    def pick_point_on_mesh(self, screen_x: int, screen_y: int):
        """?붾㈃ 醫뚰몴瑜?硫붿돩 ?쒕㈃??3D 醫뚰몴濡?蹂??"""
        info = self.pick_point_on_mesh_info(screen_x, screen_y)
        if info is None:
            return None
        return info[0]

    def _world_radius_from_px_at_depth(
        self,
        gl_x: int,
        gl_y: int,
        depth_value: float,
        viewport,
        modelview,
        projection,
        px_radius: float,
    ) -> float:
        """媛숈? depth plane?먯꽌 px 嫄곕━ -> world 嫄곕━濡??섏궛(洹쇱궗)."""
        try:
            px = float(px_radius)
        except Exception:
            return 0.0
        if not np.isfinite(px) or px <= 0.0:
            return 0.0
        try:
            d = float(depth_value)
        except Exception:
            d = float("nan")
        if not np.isfinite(d) or d >= 1.0 or d < 0.0:
            return 0.0

        try:
            vx, vy, vw, vh = [int(v) for v in viewport[:4]]
        except Exception:
            vx, vy, vw, vh = 0, 0, int(self.width()), int(self.height())
        vw = max(1, vw)
        vh = max(1, vh)

        x0 = int(np.clip(int(gl_x), vx, vx + vw - 1))
        y0 = int(np.clip(int(gl_y), vy, vy + vh - 1))
        x1 = int(np.clip(int(round(float(x0) + px)), vx, vx + vw - 1))
        if x1 == x0:
            x1 = int(np.clip(x0 + 1, vx, vx + vw - 1))

        try:
            w0 = np.asarray(
                gluUnProject(float(x0), float(y0), float(d), modelview, projection, viewport), dtype=np.float64
            ).reshape(-1)
            w1 = np.asarray(
                gluUnProject(float(x1), float(y0), float(d), modelview, projection, viewport), dtype=np.float64
            ).reshape(-1)
            if w0.size < 3 or w1.size < 3 or (not np.isfinite(w0[:3]).all()) or (not np.isfinite(w1[:3]).all()):
                return 0.0
            return float(np.linalg.norm(w1[:3] - w0[:3]))
        except Exception:
            return 0.0
    

    def _pick_brush_face(self, pos):
        """釉뚮윭?쒕줈 硫??좏깮"""
        point = self.pick_point_on_mesh(pos.x(), pos.y())
        if point is not None:
            res = self.pick_face_at_point(point, return_index=True)
            if res:
                idx, v = res
                self.brush_selected_faces.add(idx)

    def _pick_selection_brush_face(self, pos):
        """SelectionPanel??釉뚮윭???좏깮 (obj.selected_faces??諛섏쁺)"""
        obj = self.selected_obj
        if not obj or obj.mesh is None:
            return

        point = self.pick_point_on_mesh(pos.x(), pos.y())
        if point is None:
            return

        res = self.pick_face_at_point(point, return_index=True)
        if not res:
            return
        idx, _verts = res
        idx = int(idx)

        mode = str(getattr(self, "_selection_brush_mode", "replace"))
        if mode == "remove":
            obj.selected_faces.discard(idx)
        else:
            obj.selected_faces.add(idx)

    def _emit_surface_assignment_changed(self, obj: SceneObject | None) -> None:
        if obj is None:
            return
        # Keep assist-unresolved set consistent with current labels.
        try:
            unresolved = set(int(x) for x in (getattr(obj, "surface_assist_unresolved_face_indices", set()) or set()))
            if unresolved:
                unresolved.difference_update(getattr(obj, "outer_face_indices", set()) or set())
                unresolved.difference_update(getattr(obj, "inner_face_indices", set()) or set())
                unresolved.difference_update(getattr(obj, "migu_face_indices", set()) or set())
            obj.surface_assist_unresolved_face_indices = unresolved
        except Exception:
            _log_ignored_exception()
        try:
            obj._surface_assignment_version = int(getattr(obj, "_surface_assignment_version", 0) or 0) + 1
        except Exception:
            _log_ignored_exception()
        try:
            self.surfaceAssignmentChanged.emit(
                len(getattr(obj, "outer_face_indices", set()) or set()),
                len(getattr(obj, "inner_face_indices", set()) or set()),
                len(getattr(obj, "migu_face_indices", set()) or set()),
            )
        except Exception:
            _log_ignored_exception()

    def _record_surface_paint_point(self, point: np.ndarray, target: str) -> None:
        try:
            p = np.asarray(point, dtype=np.float64).reshape(-1)
            if p.size < 3 or not np.isfinite(p[:3]).all():
                return
            t = str(target or "").strip().lower()
            if t not in {"outer", "inner", "migu"}:
                t = "outer"
            self.surface_paint_points.append((p[:3].copy(), t))
            max_n = int(getattr(self, "_surface_paint_points_max", 250))
            if max_n > 0 and len(self.surface_paint_points) > max_n:
                del self.surface_paint_points[: max(1, len(self.surface_paint_points) - max_n)]
        except Exception:
            _log_ignored_exception()

    def clear_surface_paint_points(self, target: str | None = None) -> None:
        """?쒕㈃ 吏??李띿? ?? ?쒖떆瑜?吏?곷땲??"""
        try:
            if target is None:
                self.surface_paint_points = []
                return
            t = str(target or "").strip().lower()
            if t not in {"outer", "inner", "migu"}:
                t = "outer"
            self.surface_paint_points = [
                (p, tt) for (p, tt) in (self.surface_paint_points or []) if str(tt).strip().lower() != t
            ]
        except Exception:
            self.surface_paint_points = []
        try:
            self._surface_grow_state = {}
        except Exception:
            pass

    def clear_surface_lasso(self) -> None:
        """?섎윭??吏???곸뿭) ?ш?誘??ㅺ컖?? ?ㅻ쾭?덉씠瑜?珥덇린?뷀빀?덈떎."""
        try:
            self._cancel_surface_lasso_thread()
        except Exception:
            _log_ignored_exception()
        try:
            self.surface_lasso_points = []
            self.surface_lasso_face_indices = []
            self.surface_lasso_preview = None
            self._surface_area_left_press_pos = None
            self._surface_area_left_dragged = False
            self._surface_area_right_press_pos = None
            self._surface_area_right_dragged = False
        except Exception:
            self.surface_lasso_points = []
            self.surface_lasso_face_indices = []
            self.surface_lasso_preview = None
        try:
            self._surface_grow_state = {}
        except Exception:
            pass

    def clear_surface_magnetic_lasso(self, *, clear_cache: bool = False) -> None:
        """寃쎄퀎(硫댁쟻+?먯꽍) ?ш?誘??곹깭瑜?珥덇린?뷀빀?덈떎."""
        try:
            self._cancel_surface_magnetic_thread()
        except Exception:
            _log_ignored_exception()
        try:
            self.surface_magnetic_points = []
            self._surface_magnetic_drawing = False
            self._surface_magnetic_cursor_qt = None
            self._surface_magnetic_left_press_pos = None
            self._surface_magnetic_left_dragged = False
            self._surface_magnetic_right_press_pos = None
            self._surface_magnetic_right_dragged = False
            self._surface_magnetic_last_add_t = 0.0
            self._surface_magnetic_space_nav = False
        except Exception:
            self.surface_magnetic_points = []
            self._surface_magnetic_drawing = False
        if clear_cache:
            try:
                self._surface_magnetic_dist = None
                self._surface_magnetic_nn_y = None
                self._surface_magnetic_nn_x = None
                self._surface_magnetic_cache_viewport = None
                self._surface_magnetic_cache_sig = None
            except Exception:
                pass

    def _cancel_surface_magnetic_thread(self) -> None:
        thr = getattr(self, "_surface_magnetic_thread", None)
        if thr is None:
            return
        self._surface_magnetic_thread = None
        try:
            if thr.isRunning():
                thr.requestInterruption()
        except Exception:
            _log_ignored_exception()

    def start_surface_magnetic_lasso(self) -> None:
        """寃쎄퀎(硫댁쟻+?먯꽍) ?ш?誘??꾧뎄 ?쒖옉: 罹먯떆 以鍮?+ 湲곗〈 ??珥덇린??"""
        try:
            self.clear_surface_magnetic_lasso(clear_cache=False)
        except Exception:
            _log_ignored_exception()

        try:
            target = str(getattr(self, "_surface_paint_target", "outer")).strip().lower()
            if target not in {"outer", "inner", "migu"}:
                target = "outer"
            self._surface_magnetic_apply_target = target
        except Exception:
            self._surface_magnetic_apply_target = "outer"

        try:
            self._ensure_surface_magnetic_cache(force=True)
        except Exception:
            _log_ignored_exception()

        self.update()

    def _surface_magnetic_cache_signature(self) -> tuple:
        """Signature used to decide whether the magnetic edge cache is stale."""
        try:
            self.makeCurrent()
        except Exception:
            pass
        try:
            vp = glGetIntegerv(GL_VIEWPORT)
            vx, vy, vw, vh = [int(v) for v in vp[:4]]
        except Exception:
            vx, vy, vw, vh = 0, 0, int(self.width()), int(self.height())

        obj = getattr(self, "selected_obj", None)
        obj_id = id(obj) if obj is not None else 0
        try:
            mesh_id = id(getattr(obj, "mesh", None)) if obj is not None else 0
        except Exception:
            mesh_id = 0
        try:
            tr = np.asarray(getattr(obj, "translation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            if tr.size >= 3:
                obj_t = (float(round(float(tr[0]), 6)), float(round(float(tr[1]), 6)), float(round(float(tr[2]), 6)))
            else:
                obj_t = (0.0, 0.0, 0.0)
        except Exception:
            obj_t = (0.0, 0.0, 0.0)
        try:
            rr = np.asarray(getattr(obj, "rotation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            if rr.size >= 3:
                obj_r = (float(round(float(rr[0]), 4)), float(round(float(rr[1]), 4)), float(round(float(rr[2]), 4)))
            else:
                obj_r = (0.0, 0.0, 0.0)
        except Exception:
            obj_r = (0.0, 0.0, 0.0)
        try:
            sc = float(getattr(obj, "scale", 1.0) or 1.0)
            obj_s = float(round(sc, 6)) if np.isfinite(sc) else 1.0
        except Exception:
            obj_s = 1.0
        try:
            cam = getattr(self, "camera", None)
            az = float(getattr(cam, "azimuth", 0.0) or 0.0)
            el = float(getattr(cam, "elevation", 0.0) or 0.0)
            dist = float(getattr(cam, "distance", 0.0) or 0.0)
            la = np.asarray(getattr(cam, "look_at", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            if la.size >= 3:
                look_at = (float(round(float(la[0]), 5)), float(round(float(la[1]), 5)), float(round(float(la[2]), 5)))
            else:
                look_at = (0.0, 0.0, 0.0)
        except Exception:
            az, el, dist = 0.0, 0.0, 0.0
            look_at = (0.0, 0.0, 0.0)

        # Rounded for stability (avoid recompute due to tiny float noise)
        return (
            int(obj_id),
            int(mesh_id),
            int(vx),
            int(vy),
            int(vw),
            int(vh),
            float(round(az, 3)),
            float(round(el, 3)),
            float(round(dist, 4)),
            look_at,
            obj_t,
            obj_r,
            float(obj_s),
        )

    def _ensure_surface_magnetic_cache(self, *, force: bool = False) -> bool:
        sig = None
        try:
            sig = self._surface_magnetic_cache_signature()
        except Exception:
            sig = None

        if (
            (not force)
            and sig is not None
            and sig == getattr(self, "_surface_magnetic_cache_sig", None)
            and getattr(self, "_surface_magnetic_dist", None) is not None
            and getattr(self, "_surface_magnetic_nn_x", None) is not None
            and getattr(self, "_surface_magnetic_nn_y", None) is not None
            and getattr(self, "_surface_magnetic_cache_viewport", None) is not None
        ):
            return True

        ok = self._compute_surface_magnetic_cache()
        try:
            self._surface_magnetic_cache_sig = sig
        except Exception:
            self._surface_magnetic_cache_sig = None
        return bool(ok)

    def _compute_surface_magnetic_cache(self) -> bool:
        """Compute distance-to-edge cache from the current depth buffer."""
        if ndimage is None:
            self._surface_magnetic_dist = None
            self._surface_magnetic_nn_y = None
            self._surface_magnetic_nn_x = None
            self._surface_magnetic_cache_viewport = None
            return False

        obj = getattr(self, "selected_obj", None)
        if obj is None or getattr(obj, "mesh", None) is None:
            self._surface_magnetic_dist = None
            self._surface_magnetic_nn_y = None
            self._surface_magnetic_nn_x = None
            self._surface_magnetic_cache_viewport = None
            return False

        try:
            self.makeCurrent()
        except Exception:
            return False

        try:
            vp = glGetIntegerv(GL_VIEWPORT)
            vx, vy, vw, vh = [int(v) for v in vp[:4]]
        except Exception:
            vx, vy, vw, vh = 0, 0, int(self.width()), int(self.height())
        vw = max(1, int(vw))
        vh = max(1, int(vh))

        try:
            depth_raw = cast(Any, glReadPixels(vx, vy, vw, vh, GL_DEPTH_COMPONENT, GL_FLOAT))
            depth = np.asarray(depth_raw, dtype=np.float32)
            depth = np.squeeze(depth)
            if depth.ndim != 2:
                depth = depth.reshape((vh, vw))
        except Exception:
            self._surface_magnetic_dist = None
            self._surface_magnetic_nn_y = None
            self._surface_magnetic_nn_x = None
            self._surface_magnetic_cache_viewport = None
            return False

        if depth.size == 0:
            return False

        depth = np.asarray(depth, dtype=np.float32)
        mask_obj = np.isfinite(depth) & (depth < 1.0)
        if not bool(np.any(mask_obj)):
            self._surface_magnetic_dist = None
            self._surface_magnetic_nn_y = None
            self._surface_magnetic_nn_x = None
            self._surface_magnetic_cache_viewport = None
            return False

        # Background as far plane for stable gradients.
        d = depth.copy()
        d[~mask_obj] = 1.0

        # Smooth a bit to reduce triangle noise, then compute depth edges.
        try:
            sigma = float(getattr(self, "_surface_magnetic_depth_smooth_sigma", 1.0) or 1.0)
        except Exception:
            sigma = 1.0
        if np.isfinite(sigma) and sigma > 0.0:
            try:
                d = ndimage.gaussian_filter(d, sigma=float(sigma))
            except Exception:
                pass

        try:
            gx = ndimage.sobel(d, axis=1)
            gy = ndimage.sobel(d, axis=0)
            g = np.hypot(gx, gy).astype(np.float32, copy=False)
        except Exception:
            g = np.zeros((vh, vw), dtype=np.float32)

        g[~mask_obj] = 0.0

        # Silhouette edge (object-vs-background)
        try:
            sil = mask_obj & np.logical_not(
                ndimage.binary_erosion(mask_obj, structure=np.ones((3, 3), dtype=bool))
            )
        except Exception:
            sil = np.zeros((vh, vw), dtype=bool)

        # Gradient edge (creases / depth discontinuities)
        thr = 0.0
        try:
            vals = g[mask_obj]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                p90 = float(np.percentile(vals, 90.0))
                p98 = float(np.percentile(vals, 98.0))
                thr = float(p90 + 0.35 * (p98 - p90))
        except Exception:
            thr = 0.0
        try:
            thr_abs = float(getattr(self, "_surface_magnetic_edge_thr_abs", 1e-4) or 1e-4)
        except Exception:
            thr_abs = 1e-4
        thr = float(max(thr, thr_abs))

        edges = sil | (g >= thr)
        if not bool(np.any(edges)):
            self._surface_magnetic_dist = None
            self._surface_magnetic_nn_y = None
            self._surface_magnetic_nn_x = None
            self._surface_magnetic_cache_viewport = (vx, vy, vw, vh)
            return False

        try:
            inv = np.logical_not(edges)
            dist, (iy, ix) = cast(
                tuple[np.ndarray, tuple[np.ndarray, np.ndarray]],
                ndimage.distance_transform_edt(inv, return_indices=True),
            )
            self._surface_magnetic_dist = np.asarray(dist, dtype=np.float32)
            self._surface_magnetic_nn_y = np.asarray(iy, dtype=np.int32)
            self._surface_magnetic_nn_x = np.asarray(ix, dtype=np.int32)
            self._surface_magnetic_cache_viewport = (vx, vy, vw, vh)
            return True
        except Exception:
            self._surface_magnetic_dist = None
            self._surface_magnetic_nn_y = None
            self._surface_magnetic_nn_x = None
            self._surface_magnetic_cache_viewport = (vx, vy, vw, vh)
            return False

    def _surface_magnetic_snap_gl(self, gl_x: int, gl_y: int) -> tuple[int, int]:
        """Snap (gl_x,gl_y) to nearest edge pixel when within snap radius."""
        dist = getattr(self, "_surface_magnetic_dist", None)
        nnx = getattr(self, "_surface_magnetic_nn_x", None)
        nny = getattr(self, "_surface_magnetic_nn_y", None)
        vp = getattr(self, "_surface_magnetic_cache_viewport", None)
        if dist is None or nnx is None or nny is None or vp is None:
            return int(gl_x), int(gl_y)

        try:
            vx, vy, vw, vh = [int(v) for v in vp[:4]]
        except Exception:
            return int(gl_x), int(gl_y)

        x = int(gl_x) - int(vx)
        y = int(gl_y) - int(vy)
        if x < 0 or y < 0 or x >= int(vw) or y >= int(vh):
            return int(gl_x), int(gl_y)

        try:
            d = float(dist[int(y), int(x)])
        except Exception:
            return int(gl_x), int(gl_y)

        try:
            snap_r = float(getattr(self, "_surface_magnetic_snap_radius_px", 14) or 0.0)
        except Exception:
            snap_r = 14.0
        if not np.isfinite(snap_r) or snap_r <= 0.0:
            return int(gl_x), int(gl_y)

        if np.isfinite(d) and d <= float(snap_r):
            try:
                sx = int(nnx[int(y), int(x)])
                sy = int(nny[int(y), int(x)])
                sx = int(np.clip(sx, 0, int(vw) - 1))
                sy = int(np.clip(sy, 0, int(vh) - 1))
                return int(vx + sx), int(vy + sy)
            except Exception:
                return int(gl_x), int(gl_y)
        return int(gl_x), int(gl_y)

    def _surface_magnetic_try_add_point(self, qt_pos: object) -> None:
        """Append a snapped point from a Qt mouse position (best-effort, throttled)."""
        try:
            x = int(getattr(qt_pos, "x", lambda: 0)())
            y = int(getattr(qt_pos, "y", lambda: 0)())
        except Exception:
            return

        try:
            gl_x, gl_y = self._qt_to_gl_window_xy(float(x), float(y), viewport=None)
        except Exception:
            gl_x, gl_y = int(x), int(y)

        try:
            sx, sy = self._surface_magnetic_snap_gl(int(gl_x), int(gl_y))
        except Exception:
            sx, sy = int(gl_x), int(gl_y)

        pt = (int(sx), int(sy))

        # Guard: ignore clicks fully outside the mesh (background depth).
        try:
            self.makeCurrent()
            d = cast(Any, glReadPixels(int(pt[0]), int(pt[1]), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT))
            try:
                depth_val = float(d[0][0])
            except Exception:
                depth_val = float("nan")
            if (not bool(np.isfinite(depth_val))) or float(depth_val) >= 1.0:
                return
        except Exception:
            # If the guard fails, fall back to the previous permissive behavior.
            pass

        try:
            min_step = float(getattr(self, "_surface_magnetic_min_step_px", 1.5) or 0.0)
        except Exception:
            min_step = 1.5
        if not np.isfinite(min_step) or min_step < 0.0:
            min_step = 0.0

        pts = list(getattr(self, "surface_magnetic_points", None) or [])
        if pts:
            try:
                last = pts[-1]
                dx = float(pt[0] - int(last[0]))
                dy = float(pt[1] - int(last[1]))
                if float(np.hypot(dx, dy)) < float(min_step):
                    return
            except Exception:
                pass

        try:
            max_pts = int(getattr(self, "_surface_magnetic_max_points", 12000) or 12000)
        except Exception:
            max_pts = 12000
        max_pts = max(2000, min(max_pts, 200000))
        if int(len(pts)) >= int(max_pts):
            return

        pts.append(pt)
        self.surface_magnetic_points = pts
        try:
            self._surface_magnetic_last_add_t = float(time.monotonic())
        except Exception:
            self._surface_magnetic_last_add_t = 0.0

    def _surface_boundary_try_add_world_point(self, qt_pos: object) -> bool:
        """Add a boundary vertex snapped to magnetic edges, stored as a world-space point.

        This keeps the polygon attached to the mesh even when the camera moves.
        Points are appended to `surface_lasso_points` / `surface_lasso_face_indices`.
        """
        try:
            qx = int(getattr(qt_pos, "x", lambda: 0)())
            qy = int(getattr(qt_pos, "y", lambda: 0)())
        except Exception:
            return False

        info = self.pick_point_on_mesh_info(int(qx), int(qy))
        if info is None:
            self.status_info = "?좑툘 硫붿돩 ?꾨? ?대┃???먯쓣 李띿뼱 二쇱꽭??"
            return False

        try:
            picked_world, _picked_depth, gl_x, gl_y, viewport, _modelview, _projection = info
        except Exception:
            return False

        try:
            self._ensure_surface_magnetic_cache(force=False)
        except Exception:
            pass

        try:
            sx, sy = self._surface_magnetic_snap_gl(int(gl_x), int(gl_y))
        except Exception:
            sx, sy = int(gl_x), int(gl_y)

        # Convert the snapped screen location back to a mesh point. This intentionally reuses
        # the same robust picking logic (including background depth search + bounds guards).
        pt = None
        try:
            qx2, qy2 = self._gl_window_to_qt_xy(float(sx), float(sy), viewport=viewport)
            info2 = self.pick_point_on_mesh_info(int(round(qx2)), int(round(qy2)))
            if info2 is not None:
                pt = np.asarray(info2[0], dtype=np.float64).reshape(3)
        except Exception:
            pt = None

        if pt is None:
            pt = np.asarray(picked_world, dtype=np.float64).reshape(3)

        try:
            self.surface_lasso_points.append(np.asarray(pt[:3], dtype=np.float64))
        except Exception:
            try:
                self.surface_lasso_points.append(pt)
            except Exception:
                return False

        try:
            res = self.pick_face_at_point(np.asarray(pt, dtype=np.float64), return_index=True)
            if res:
                fi, _v = res
                self.surface_lasso_face_indices.append(int(fi))
            else:
                self.surface_lasso_face_indices.append(-1)
        except Exception:
            try:
                self.surface_lasso_face_indices.append(-1)
            except Exception:
                _log_ignored_exception()

        try:
            self.surface_lasso_preview = qt_pos
        except Exception:
            pass

        return True

    def _cancel_surface_lasso_thread(self) -> None:
        thr = getattr(self, "_surface_lasso_thread", None)
        if thr is None:
            return
        # Detach first so late signals are ignored.
        self._surface_lasso_thread = None
        try:
            if thr.isRunning():
                thr.requestInterruption()
        except Exception:
            _log_ignored_exception()

    def _surface_lasso_tool_strings(self, tool: object | None = None) -> tuple[str, str]:
        """Return (icon, label) for the lasso selection tool."""
        try:
            t = str(tool if tool is not None else getattr(self, "_surface_lasso_apply_tool", "area")).strip().lower()
        except Exception:
            t = "area"
        if t in {"boundary", "magnetic", "paint_surface_magnetic"}:
            return "?㎠", "寃쎄퀎(硫댁쟻+?먯꽍)"
        return "Area", "Lasso(area)"

    def _finish_surface_lasso(self, modifiers, *, seed_pos=None) -> None:
        """Finalize current lasso and compute visible-face selection."""
        obj = self.selected_obj
        if obj is None or getattr(obj, "mesh", None) is None:
            self.status_info = "?좑툘 癒쇱? 硫붿돩瑜??좏깮??二쇱꽭??"
            self.clear_surface_lasso()
            self.update()
            return

        pts_world = list(getattr(self, "surface_lasso_points", None) or [])
        if len(pts_world) < 3:
            icon, lbl = self._surface_lasso_tool_strings(
                "boundary" if str(getattr(self, "picking_mode", "none")) == "paint_surface_magnetic" else "area"
            )
            self.status_info = f"{icon} {lbl}: ?먯쓣 3媛??댁긽 李띿뼱二쇱꽭?? (?고겢由?Enter=?뺤젙)"
            self.update()
            return

        thr = getattr(self, "_surface_lasso_thread", None)
        try:
            if thr is not None and thr.isRunning():
                _icon, lbl = self._surface_lasso_tool_strings()
                self.status_info = f"{lbl}: selection is computing..."
                self.update()
                return
        except Exception:
            _log_ignored_exception()

        # Snapshot state at confirm time
        target = str(getattr(self, "_surface_paint_target", "outer")).strip().lower()
        if target not in {"outer", "inner", "migu"}:
            target = "outer"
        self._surface_lasso_apply_target = target
        self._surface_lasso_apply_modifiers = modifiers
        self._surface_lasso_apply_tool = (
            "boundary" if str(getattr(self, "picking_mode", "none")) == "paint_surface_magnetic" else "area"
        )
        icon, lbl = self._surface_lasso_tool_strings()

        # Magic-wand seed: prefer the confirm click position (user intent) if available,
        # then try polygon centroid, then fall back to last picked vertex face.
        seed_face_idx = -1
        if seed_pos is not None:
            try:
                px = int(getattr(seed_pos, "x", lambda: -1)())
                py = int(getattr(seed_pos, "y", lambda: -1)())
                if px >= 0 and py >= 0:
                    p_seed = self.pick_point_on_mesh(px, py)
                    if p_seed is not None:
                        res = self.pick_face_at_point(np.asarray(p_seed, dtype=np.float64), return_index=True)
                        if res:
                            seed_face_idx = int(res[0])
            except Exception:
                seed_face_idx = -1

        try:
            pts_arr = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
            if pts_arr.size >= 9:
                centroid_world = np.mean(pts_arr[:, :3], axis=0)
                res = self.pick_face_at_point(np.asarray(centroid_world, dtype=np.float64), return_index=True)
                if res:
                    seed_face_idx = int(res[0])
        except Exception:
            seed_face_idx = -1
        if seed_face_idx < 0:
            try:
                fi_list = getattr(self, "surface_lasso_face_indices", None) or []
                if fi_list:
                    seed_face_idx = int(fi_list[-1])
            except Exception:
                seed_face_idx = -1

        self.makeCurrent()
        viewport = glGetIntegerv(GL_VIEWPORT)
        mv_raw = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj_raw = glGetDoublev(GL_PROJECTION_MATRIX)

        try:
            mv = np.asarray(mv_raw, dtype=np.float64).reshape(4, 4).T
            proj = np.asarray(proj_raw, dtype=np.float64).reshape(4, 4).T
        except Exception:
            mv = np.eye(4, dtype=np.float64)
            proj = np.eye(4, dtype=np.float64)

        try:
            vx, vy, vw, vh = [int(v) for v in viewport[:4]]
        except Exception:
            vx, vy, vw, vh = 0, 0, int(self.width()), int(self.height())
        vw = max(1, vw)
        vh = max(1, vh)

        # Project world points -> GL window coords (bottom-left origin)
        proj_xy: list[tuple[float, float]] = []
        for p in pts_world:
            try:
                px, py, _pz = gluProject(
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    mv_raw,
                    proj_raw,
                    viewport,
                )
                if not (np.isfinite(px) and np.isfinite(py)):
                    continue
                proj_xy.append((float(px), float(py)))
            except Exception:
                continue

        if len(proj_xy) < 3:
            self.status_info = f"{icon} {lbl}: ???낅젰???щ컮瑜댁? ?딆뒿?덈떎."
            self.update()
            return

        poly_gl = np.asarray(proj_xy, dtype=np.float64)

        # Bounding box in GL coords (clamped to viewport)
        try:
            gl_x0 = int(np.clip(int(np.floor(float(np.min(poly_gl[:, 0])))), vx, vx + vw - 1))
            gl_x1 = int(np.clip(int(np.ceil(float(np.max(poly_gl[:, 0])))), vx, vx + vw - 1))
            gl_y0 = int(np.clip(int(np.floor(float(np.min(poly_gl[:, 1])))), vy, vy + vh - 1))
            gl_y1 = int(np.clip(int(np.ceil(float(np.max(poly_gl[:, 1])))), vy, vy + vh - 1))
        except Exception:
            self.status_info = f"{icon} {lbl}: ???낅젰???щ컮瑜댁? ?딆뒿?덈떎."
            self.update()
            return

        width = int(max(1, gl_x1 - gl_x0 + 1))
        height = int(max(1, gl_y1 - gl_y0 + 1))

        # Read depth buffer for occlusion filtering (best-effort).
        depth_map = None
        try:
            depth_raw = cast(Any, glReadPixels(gl_x0, gl_y0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT))
            depth_map = np.asarray(depth_raw, dtype=np.float32)
            depth_map = np.squeeze(depth_map)
            if depth_map.ndim != 2:
                depth_map = depth_map.reshape((height, width))
        except Exception:
            depth_map = None

        try:
            tol = float(getattr(self, "_surface_area_depth_tol", 0.01))
        except Exception:
            tol = 0.01
        try:
            max_sel_raw = int(getattr(self, "_surface_area_max_selected_faces", 0))
        except Exception:
            max_sel_raw = 0

        n_faces = 0
        try:
            n_faces = int(getattr(getattr(obj, "mesh", None), "n_faces", 0) or 0)
        except Exception:
            n_faces = 0

        # Unlimited when <= 0 (use the whole mesh size, with a minimal positive fallback).
        if max_sel_raw <= 0:
            max_sel = int(max(1, n_faces))
        else:
            max_sel = int(max(1, max_sel_raw))
            if n_faces > 0:
                max_sel = min(max_sel, int(n_faces))

        self.status_info = f"{icon} {lbl}: computing visible faces..."
        self.update()

        wand_enabled = bool(
            modifiers
            & (
                Qt.KeyboardModifier.ShiftModifier
                | Qt.KeyboardModifier.ControlModifier
            )
        )
        try:
            thr2 = _SurfaceLassoSelectThread(
                obj.mesh.vertices,
                obj.mesh.faces,
                np.asarray(getattr(obj, "translation", [0.0, 0.0, 0.0]), dtype=np.float64),
                np.asarray(getattr(obj, "rotation", [0.0, 0.0, 0.0]), dtype=np.float64),
                float(getattr(obj, "scale", 1.0)),
                np.asarray(getattr(getattr(self, "camera", None), "position", np.array([0.0, 0.0, 0.0])), dtype=np.float64),
                mv,
                proj,
                np.asarray(viewport, dtype=np.int32),
                poly_gl,
                (int(gl_x0), int(gl_y0), int(gl_x1), int(gl_y1)),
                (int(gl_x0), int(gl_y0)),
                depth_map,
                face_centroids=getattr(obj, "_face_centroids", None),
                face_normals=getattr(getattr(obj, "mesh", None), "face_normals", None),
                depth_tol=tol,
                max_selected_faces=max_sel,
                wand_seed_face_idx=seed_face_idx if seed_face_idx >= 0 else None,
                wand_enabled=wand_enabled,
                wand_angle_deg=float(getattr(self, "_surface_area_wand_angle_deg", 30.0) or 30.0),
                wand_knn_k=int(getattr(self, "_surface_area_wand_knn_k", 12) or 12),
                wand_time_budget_s=float(getattr(self, "_surface_area_wand_time_budget_s", 2.0) or 2.0),
            )
            thr2.computed.connect(self._on_surface_lasso_computed)
            thr2.failed.connect(self._on_surface_lasso_failed)
            self._surface_lasso_thread = thr2
            thr2.start()
        except Exception as e:
            self._surface_lasso_thread = None
            _icon, lbl = self._surface_lasso_tool_strings()
            self.status_info = f"?좑툘 {lbl} 怨꾩궛 ?쒖옉 ?ㅽ뙣: {e}"
            self.update()

    def _on_surface_lasso_failed(self, msg: str) -> None:
        if self.sender() is not getattr(self, "_surface_lasso_thread", None):
            return
        self._surface_lasso_thread = None
        _icon, lbl = self._surface_lasso_tool_strings()
        self.status_info = f"?좑툘 {lbl} ?ㅽ뙣: {msg}"
        self.update()

    def _on_surface_lasso_computed(self, result: object) -> None:
        if self.sender() is not getattr(self, "_surface_lasso_thread", None):
            return
        self._surface_lasso_thread = None

        obj = self.selected_obj
        if obj is None or getattr(obj, "mesh", None) is None:
            _icon, lbl = self._surface_lasso_tool_strings()
            self.status_info = f"?좑툘 {lbl}: ?좏깮 ??곸씠 ?놁뒿?덈떎."
            self.clear_surface_lasso()
            self.update()
            return

        indices = None
        stats = {}
        try:
            if isinstance(result, dict):
                indices = result.get("indices", None)
                stats = result.get("stats", {}) or {}
        except Exception:
            indices = None
            stats = {}

        if indices is None:
            icon, lbl = self._surface_lasso_tool_strings()
            self.status_info = f"{icon} {lbl}: ?좏깮??硫댁씠 ?놁뒿?덈떎."
            self.clear_surface_lasso()
            self.update()
            return

        try:
            idx_arr = np.asarray(indices, dtype=np.int32).reshape(-1)
        except Exception:
            idx_arr = np.zeros((0,), dtype=np.int32)

        target = str(getattr(self, "_surface_lasso_apply_target", getattr(self, "_surface_paint_target", "outer"))).strip().lower()
        if target not in {"outer", "inner", "migu"}:
            target = "outer"
        target_lbl = {"outer": "?몃㈃", "inner": "?대㈃", "migu": "誘멸뎄"}.get(target, target)
        modifiers = getattr(self, "_surface_lasso_apply_modifiers", Qt.KeyboardModifier.NoModifier)

        target_set = self._get_surface_target_set(obj, target)
        remove = bool(modifiers & Qt.KeyboardModifier.AltModifier)

        if idx_arr.size:
            try:
                chunk = int(getattr(self, "_surface_assign_chunk", 200000))
            except Exception:
                chunk = 200000
            chunk = max(20000, min(chunk, 500000))

            other_sets = self._get_other_surface_sets(obj, target) if not remove else []
            try:
                idx_arr = np.asarray(idx_arr, dtype=np.int32).reshape(-1)
            except Exception:
                idx_arr = np.zeros((0,), dtype=np.int32)

            for start in range(0, int(idx_arr.size), int(chunk)):
                ids = idx_arr[start : start + int(chunk)].tolist()
                if remove:
                    target_set.difference_update(ids)
                else:
                    target_set.update(ids)
                    for s in other_sets:
                        s.difference_update(ids)

        self._emit_surface_assignment_changed(obj)

        try:
            selected_n = int(stats.get("selected", int(idx_arr.size)))
            cand_n = int(stats.get("candidates", 0))
        except Exception:
            selected_n = int(idx_arr.size)
            cand_n = 0
        wand_info = ""
        try:
            wand_n = int(stats.get("wand_selected", 0) or 0)
            wand_c = int(stats.get("wand_candidates", 0) or 0)
            if wand_n and wand_c:
                wand_info = f" / ?꾨뱶 {wand_n:,}/{wand_c:,}"
        except Exception:
            wand_info = ""

        comp_info = ""
        try:
            comp_n = int(stats.get("component_selected", 0) or 0)
            comp_c = int(stats.get("component_candidates", 0) or 0)
            if comp_n and comp_c and comp_c > comp_n:
                comp_info = f" / ?곌껐 {comp_n:,}/{comp_c:,}"
        except Exception:
            comp_info = ""

        trunc_info = ""
        try:
            if bool(stats.get("truncated", False)):
                ms = int(stats.get("max_selected_faces", 0) or 0)
                trunc_info = f" / 理쒕? {ms:,} ?쒗븳" if ms > 0 else " / 理쒕? ?쒗븳"
        except Exception:
            trunc_info = ""

        op = "?쒓굅" if remove else "異붽?"
        icon, lbl = self._surface_lasso_tool_strings()
        msg = f"{icon} {lbl} [{target_lbl}]: {op} {selected_n:,} faces{comp_info}{wand_info}{trunc_info}"
        if cand_n:
            msg += f" (?꾨낫 {cand_n:,})"
        self.status_info = msg

        # Keep tool active, but clear the polygon for the next stroke.
        self.clear_surface_lasso()
        self.update()

    def _finish_surface_magnetic_lasso(self, modifiers, *, seed_pos=None) -> None:
        """?꾩옱 '寃쎄퀎(硫댁쟻+?먯꽍) ?ш?誘? ?대━怨??곸뿭???ы븿?섎뒗 '蹂댁씠?? 硫댁쓣 ??踰덉뿉 吏?뺥빀?덈떎."""
        obj = self.selected_obj
        if obj is None or getattr(obj, "mesh", None) is None:
            self.status_info = "?좑툘 癒쇱? 硫붿돩瑜??좏깮??二쇱꽭??"
            self.clear_surface_magnetic_lasso(clear_cache=False)
            self.update()
            return

        pts = list(getattr(self, "surface_magnetic_points", None) or [])
        if len(pts) < 3:
            self.status_info = "?㎠ 寃쎄퀎(硫댁쟻+?먯꽍): ?곸뿭??3???댁긽 李띿뼱二쇱꽭?? (?고겢由?Enter=?뺤젙)"
            self.update()
            return

        thr = getattr(self, "_surface_magnetic_thread", None)
        try:
            if thr is not None and thr.isRunning():
                self.status_info = "Boundary(area+magnetic) selection is computing..."
                self.update()
                return
        except Exception:
            _log_ignored_exception()

        # Snapshot state at confirm time
        target = str(getattr(self, "_surface_paint_target", "outer")).strip().lower()
        if target not in {"outer", "inner", "migu"}:
            target = "outer"
        self._surface_magnetic_apply_target = target
        self._surface_magnetic_apply_modifiers = modifiers

        # Magic-wand seed: prefer the confirm click position (user intent) if available.
        seed_face_idx = -1
        if seed_pos is not None:
            try:
                px = int(getattr(seed_pos, "x", lambda: -1)())
                py = int(getattr(seed_pos, "y", lambda: -1)())
                if px >= 0 and py >= 0:
                    p_seed = self.pick_point_on_mesh(px, py)
                    if p_seed is not None:
                        res = self.pick_face_at_point(np.asarray(p_seed, dtype=np.float64), return_index=True)
                        if res:
                            seed_face_idx = int(res[0])
            except Exception:
                seed_face_idx = -1

        if seed_face_idx < 0:
            try:
                pts_arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
                if pts_arr.size >= 6:
                    cx = float(np.mean(pts_arr[:, 0]))
                    cy = float(np.mean(pts_arr[:, 1]))
                    self.makeCurrent()
                    vp0 = glGetIntegerv(GL_VIEWPORT)
                    qx0, qy0 = self._gl_window_to_qt_xy(cx, cy, viewport=vp0)
                    p_seed = self.pick_point_on_mesh(int(round(qx0)), int(round(qy0)))
                    if p_seed is not None:
                        res = self.pick_face_at_point(np.asarray(p_seed, dtype=np.float64), return_index=True)
                        if res:
                            seed_face_idx = int(res[0])
            except Exception:
                seed_face_idx = -1

        self.makeCurrent()
        viewport = glGetIntegerv(GL_VIEWPORT)
        mv_raw = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj_raw = glGetDoublev(GL_PROJECTION_MATRIX)
        try:
            mv = np.asarray(mv_raw, dtype=np.float64).reshape(4, 4)
            proj = np.asarray(proj_raw, dtype=np.float64).reshape(4, 4)
        except Exception:
            self.status_info = "?㎠ 寃쎄퀎(硫댁쟻+?먯꽍): 移대찓???곹깭瑜??쎌쓣 ???놁뒿?덈떎."
            self.update()
            return

        try:
            vx, vy, vw, vh = [int(v) for v in viewport[:4]]
        except Exception:
            vx, vy, vw, vh = 0, 0, int(self.width()), int(self.height())
        vw = max(1, vw)
        vh = max(1, vh)

        poly_gl = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        try:
            max_poly = int(getattr(self, "_surface_magnetic_max_poly_points", 800) or 800)
        except Exception:
            max_poly = 800
        poly2 = self._sanitize_polygon_2d(poly_gl, max_points=max(50, int(max_poly)), eps=1e-6)
        if poly2 is None or poly2.shape[0] < 3:
            self.status_info = "?㎠ 寃쎄퀎(硫댁쟻+?먯꽍): ?대━怨ㅼ씠 ?щ컮瑜댁? ?딆뒿?덈떎."
            self.update()
            return
        poly_gl = np.asarray(poly2, dtype=np.float64).reshape(-1, 2)

        # Bounding box in GL coords (clamped to viewport)
        try:
            gl_x0 = int(np.clip(int(np.floor(float(np.min(poly_gl[:, 0])))), vx, vx + vw - 1))
            gl_x1 = int(np.clip(int(np.ceil(float(np.max(poly_gl[:, 0])))), vx, vx + vw - 1))
            gl_y0 = int(np.clip(int(np.floor(float(np.min(poly_gl[:, 1])))), vy, vy + vh - 1))
            gl_y1 = int(np.clip(int(np.ceil(float(np.max(poly_gl[:, 1])))), vy, vy + vh - 1))
        except Exception:
            self.status_info = "?㎠ 寃쎄퀎(硫댁쟻+?먯꽍): ?대━怨ㅼ씠 ?щ컮瑜댁? ?딆뒿?덈떎."
            self.update()
            return

        width = int(max(1, gl_x1 - gl_x0 + 1))
        height = int(max(1, gl_y1 - gl_y0 + 1))

        # Read depth buffer for occlusion filtering (best-effort).
        depth_map = None
        try:
            depth_raw = cast(Any, glReadPixels(gl_x0, gl_y0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT))
            depth_map = np.asarray(depth_raw, dtype=np.float32)
            depth_map = np.squeeze(depth_map)
            if depth_map.ndim != 2:
                depth_map = depth_map.reshape((height, width))
        except Exception:
            depth_map = None

        try:
            tol = float(getattr(self, "_surface_area_depth_tol", 0.01))
        except Exception:
            tol = 0.01
        try:
            max_sel_raw = int(getattr(self, "_surface_area_max_selected_faces", 0))
        except Exception:
            max_sel_raw = 0

        n_faces = 0
        try:
            n_faces = int(getattr(getattr(obj, "mesh", None), "n_faces", 0) or 0)
        except Exception:
            n_faces = 0

        if max_sel_raw <= 0:
            max_sel = int(max(1, n_faces))
        else:
            max_sel = int(max(1, max_sel_raw))
            if n_faces > 0:
                max_sel = min(max_sel, int(n_faces))

        self.status_info = "Boundary(area+magnetic): computing visible faces..."
        self.update()

        wand_enabled = bool(
            modifiers
            & (
                Qt.KeyboardModifier.ShiftModifier
                | Qt.KeyboardModifier.ControlModifier
            )
        )
        try:
            thr2 = _SurfaceLassoSelectThread(
                obj.mesh.vertices,
                obj.mesh.faces,
                np.asarray(getattr(obj, "translation", [0.0, 0.0, 0.0]), dtype=np.float64),
                np.asarray(getattr(obj, "rotation", [0.0, 0.0, 0.0]), dtype=np.float64),
                float(getattr(obj, "scale", 1.0)),
                np.asarray(getattr(getattr(self, "camera", None), "position", np.array([0.0, 0.0, 0.0])), dtype=np.float64),
                mv,
                proj,
                np.asarray(viewport, dtype=np.int32),
                poly_gl,
                (int(gl_x0), int(gl_y0), int(gl_x1), int(gl_y1)),
                (int(gl_x0), int(gl_y0)),
                depth_map,
                face_centroids=getattr(obj, "_face_centroids", None),
                face_normals=getattr(getattr(obj, "mesh", None), "face_normals", None),
                depth_tol=tol,
                max_selected_faces=max_sel,
                wand_seed_face_idx=seed_face_idx if seed_face_idx >= 0 else None,
                wand_enabled=wand_enabled,
                wand_angle_deg=float(getattr(self, "_surface_area_wand_angle_deg", 30.0) or 30.0),
                wand_knn_k=int(getattr(self, "_surface_area_wand_knn_k", 12) or 12),
                wand_time_budget_s=float(getattr(self, "_surface_area_wand_time_budget_s", 2.0) or 2.0),
            )
            thr2.computed.connect(self._on_surface_magnetic_computed)
            thr2.failed.connect(self._on_surface_magnetic_failed)
            self._surface_magnetic_thread = thr2
            thr2.start()
        except Exception as e:
            self._surface_magnetic_thread = None
            self.status_info = f"?좑툘 寃쎄퀎(硫댁쟻+?먯꽍) 怨꾩궛 ?쒖옉 ?ㅽ뙣: {e}"
            self.update()

    def _on_surface_magnetic_failed(self, msg: str) -> None:
        if self.sender() is not getattr(self, "_surface_magnetic_thread", None):
            return
        self._surface_magnetic_thread = None
        self.status_info = f"?좑툘 寃쎄퀎(硫댁쟻+?먯꽍) ?ㅽ뙣: {msg}"
        self.update()

    def _on_surface_magnetic_computed(self, result: object) -> None:
        if self.sender() is not getattr(self, "_surface_magnetic_thread", None):
            return
        self._surface_magnetic_thread = None

        obj = self.selected_obj
        if obj is None or getattr(obj, "mesh", None) is None:
            self.status_info = "?좑툘 寃쎄퀎(硫댁쟻+?먯꽍): ?좏깮 ??곸씠 ?놁뒿?덈떎."
            self.clear_surface_magnetic_lasso(clear_cache=False)
            self.update()
            return

        indices = None
        stats = {}
        try:
            if isinstance(result, dict):
                indices = result.get("indices", None)
                stats = result.get("stats", {}) or {}
        except Exception:
            indices = None
            stats = {}

        if indices is None:
            self.status_info = "?㎠ 寃쎄퀎(硫댁쟻+?먯꽍): ?좏깮??硫댁씠 ?놁뒿?덈떎."
            self.clear_surface_magnetic_lasso(clear_cache=False)
            self.update()
            return

        try:
            idx_arr = np.asarray(indices, dtype=np.int32).reshape(-1)
        except Exception:
            idx_arr = np.zeros((0,), dtype=np.int32)

        target = str(getattr(self, "_surface_magnetic_apply_target", getattr(self, "_surface_paint_target", "outer"))).strip().lower()
        if target not in {"outer", "inner", "migu"}:
            target = "outer"
        target_lbl = {"outer": "?몃㈃", "inner": "?대㈃", "migu": "誘멸뎄"}.get(target, target)
        modifiers = getattr(self, "_surface_magnetic_apply_modifiers", Qt.KeyboardModifier.NoModifier)

        target_set = self._get_surface_target_set(obj, target)
        remove = bool(modifiers & Qt.KeyboardModifier.AltModifier)

        if idx_arr.size:
            try:
                chunk = int(getattr(self, "_surface_assign_chunk", 200000))
            except Exception:
                chunk = 200000
            chunk = max(20000, min(chunk, 500000))

            other_sets = self._get_other_surface_sets(obj, target) if not remove else []
            try:
                idx_arr = np.asarray(idx_arr, dtype=np.int32).reshape(-1)
            except Exception:
                idx_arr = np.zeros((0,), dtype=np.int32)

            for start in range(0, int(idx_arr.size), int(chunk)):
                ids = idx_arr[start : start + int(chunk)].tolist()
                if remove:
                    target_set.difference_update(ids)
                else:
                    target_set.update(ids)
                    for s in other_sets:
                        s.difference_update(ids)

        self._emit_surface_assignment_changed(obj)

        try:
            selected_n = int(stats.get("selected", int(idx_arr.size)))
            cand_n = int(stats.get("candidates", 0))
        except Exception:
            selected_n = int(idx_arr.size)
            cand_n = 0
        wand_info = ""
        try:
            wand_n = int(stats.get("wand_selected", 0) or 0)
            wand_c = int(stats.get("wand_candidates", 0) or 0)
            if wand_n and wand_c:
                wand_info = f" / ?꾨뱶 {wand_n:,}/{wand_c:,}"
        except Exception:
            wand_info = ""

        comp_info = ""
        try:
            comp_n = int(stats.get("component_selected", 0) or 0)
            comp_c = int(stats.get("component_candidates", 0) or 0)
            if comp_n and comp_c and comp_c > comp_n:
                comp_info = f" / ?곌껐 {comp_n:,}/{comp_c:,}"
        except Exception:
            comp_info = ""

        trunc_info = ""
        try:
            if bool(stats.get("truncated", False)):
                ms = int(stats.get("max_selected_faces", 0) or 0)
                trunc_info = f" / 理쒕? {ms:,} ?쒗븳" if ms > 0 else " / 理쒕? ?쒗븳"
        except Exception:
            trunc_info = ""

        op = "?쒓굅" if remove else "異붽?"
        msg = f"?㎠ 寃쎄퀎(硫댁쟻+?먯꽍) [{target_lbl}]: {op} {selected_n:,} faces{comp_info}{wand_info}{trunc_info}"
        if cand_n:
            msg += f" (?꾨낫 {cand_n:,})"
        self.status_info = msg

        # Keep tool active, but clear the polygon for the next stroke.
        self.clear_surface_magnetic_lasso(clear_cache=False)
        self.update()

    def _get_surface_target_set(self, obj: SceneObject, target: str) -> set[int]:
        target = str(target or "").strip().lower()
        if target == "inner":
            if not hasattr(obj, "inner_face_indices") or obj.inner_face_indices is None:
                obj.inner_face_indices = set()
            return obj.inner_face_indices
        if target == "migu":
            if not hasattr(obj, "migu_face_indices") or obj.migu_face_indices is None:
                obj.migu_face_indices = set()
            return obj.migu_face_indices
        if not hasattr(obj, "outer_face_indices") or obj.outer_face_indices is None:
            obj.outer_face_indices = set()
        return obj.outer_face_indices

    def _get_other_surface_sets(self, obj: SceneObject, target: str) -> list[set[int]]:
        t = str(target or "").strip().lower()
        sets: list[set[int]] = []
        for name in ("outer", "inner", "migu"):
            if name == t:
                continue
            try:
                sets.append(self._get_surface_target_set(obj, name))
            except Exception:
                continue
        return sets

    def _get_face_adjacency(self, obj: SceneObject):
        mesh = getattr(obj, "mesh", None)
        if mesh is None or getattr(mesh, "faces", None) is None:
            return None
        n_faces = int(getattr(mesh, "n_faces", 0) or 0)
        cached_n = int(getattr(obj, "_face_adjacency_faces_count", 0) or 0)
        adjacency = getattr(obj, "_face_adjacency", None)
        if adjacency is not None and cached_n == n_faces:
            return adjacency

        try:
            faces = np.asarray(mesh.faces, dtype=np.int32)
            if faces.ndim != 2 or faces.shape[1] < 3 or n_faces <= 0:
                return None
        except Exception:
            return None

        adjacency = [[] for _ in range(n_faces)]
        edge_to_face: dict[tuple[int, int], int] = {}

        for fi in range(n_faces):
            try:
                a = int(faces[fi, 0])
                b = int(faces[fi, 1])
                c = int(faces[fi, 2])
            except Exception:
                continue
            edges = ((a, b), (b, c), (c, a))
            for u, v in edges:
                if u == v:
                    continue
                e = (u, v) if u < v else (v, u)
                other = edge_to_face.get(e)
                if other is None:
                    edge_to_face[e] = fi
                    continue
                if other != fi:
                    adjacency[fi].append(other)
                    adjacency[other].append(fi)

        try:
            obj._face_adjacency = adjacency
            obj._face_adjacency_faces_count = int(n_faces)
        except Exception:
            _log_ignored_exception()

        return adjacency

    def _grow_smooth_patch(self, obj: SceneObject, seed_face_idx: int, *, max_angle_deg: float = 30.0) -> set[int]:
        mesh = getattr(obj, "mesh", None)
        if mesh is None:
            return {int(seed_face_idx)}

        try:
            if getattr(mesh, "face_normals", None) is None:
                mesh.compute_normals(compute_vertex_normals=False)
        except Exception:
            _log_ignored_exception()

        normals = getattr(mesh, "face_normals", None)
        if normals is None:
            return {int(seed_face_idx)}

        n_faces = int(getattr(mesh, "n_faces", 0) or 0)
        seed = int(seed_face_idx)
        if seed < 0 or seed >= n_faces:
            return set()

        # Safety limits to keep the UI responsive on huge meshes.
        try:
            max_faces = int(getattr(self, "_surface_grow_max_faces", 120000))
        except Exception:
            max_faces = 120000
        max_faces = max(1, max_faces)
        try:
            time_budget_s = float(getattr(self, "_surface_grow_time_budget_s", 0.35))
        except Exception:
            time_budget_s = 0.35
        time_budget_s = max(0.05, min(time_budget_s, 3.0))

        cos_thr = float(np.cos(np.radians(float(max_angle_deg))))
        visited: set[int] = {seed}
        stack: list[int] = [seed]

        try:
            fn = np.asarray(normals, dtype=np.float64)
        except Exception:
            return {seed}

        # Prefer edge adjacency on moderate meshes; fall back to centroid-KNN for very large meshes.
        adjacency = None
        try:
            edge_adj_max = int(getattr(self, "_surface_grow_edge_adj_max_faces", 250000))
        except Exception:
            edge_adj_max = 250000
        if n_faces <= max(5000, edge_adj_max):
            adjacency = self._get_face_adjacency(obj)

        centroids = getattr(obj, "_face_centroids", None)
        tree = getattr(obj, "_face_centroid_kdtree", None)
        if adjacency is None:
            if centroids is None or int(getattr(obj, "_face_centroid_faces_count", 0) or 0) != n_faces:
                try:
                    f = np.asarray(getattr(mesh, "faces", None), dtype=np.int32)
                    v = np.asarray(getattr(mesh, "vertices", None), dtype=np.float64)
                    v0 = v[f[:, 0]]
                    v1 = v[f[:, 1]]
                    v2 = v[f[:, 2]]
                    centroids = ((v0 + v1 + v2) / 3.0).astype(np.float32, copy=False)
                except Exception:
                    centroids = None
                    tree = None
                    if centroids is not None and centroids.size != 0:
                        try:
                            from scipy import spatial as _spatial

                            tree = cast(Any, _spatial).cKDTree(centroids)
                        except Exception:
                            tree = None
                try:
                    obj._face_centroids = centroids
                    obj._face_centroid_kdtree = tree
                    obj._face_centroid_faces_count = int(n_faces)
                except Exception:
                    _log_ignored_exception()

        start_t = time.monotonic()
        while stack:
            if len(visited) >= max_faces:
                break
            if time.monotonic() - start_t > time_budget_s:
                break
            fi = stack.pop()
            try:
                n0 = fn[fi, :3]
            except Exception:
                continue

            if adjacency is not None:
                neigh = adjacency[fi]
            elif tree is not None and centroids is not None:
                try:
                    k = int(getattr(self, "_surface_grow_knn", 18))
                except Exception:
                    k = 18
                k = max(6, min(k, 64, n_faces))
                try:
                    _d, idx = tree.query(np.asarray(centroids[int(fi)], dtype=np.float64), k=int(k))
                    neigh = np.atleast_1d(idx).astype(np.int32, copy=False).tolist()
                except Exception:
                    neigh = []
            else:
                neigh = []

            for nb in neigh:
                nb_i = int(nb)
                if nb_i == fi or nb_i in visited or nb_i < 0 or nb_i >= n_faces:
                    continue
                try:
                    n1 = fn[nb_i, :3]
                    if float(np.dot(n0, n1)) >= cos_thr:
                        visited.add(nb_i)
                        stack.append(nb_i)
                except Exception:
                    continue

        return visited

    def _camera_local_for_obj(self, obj: SceneObject) -> np.ndarray | None:
        try:
            cam = getattr(self, "camera", None)
            cam_w = np.asarray(getattr(cam, "position", None), dtype=np.float64).reshape(-1)
            if cam_w.size < 3 or not np.isfinite(cam_w[:3]).all():
                return None

            trans = np.asarray(getattr(obj, "translation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            if trans.size < 3:
                trans = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            rot_deg = np.asarray(getattr(obj, "rotation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            if rot_deg.size < 3:
                rot_deg = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            scale = float(getattr(obj, "scale", 1.0))
            if abs(scale) < 1e-12:
                scale = 1.0

            rx, ry, rz = np.radians(rot_deg[:3])
            cx, sx = float(np.cos(rx)), float(np.sin(rx))
            cy, sy = float(np.cos(ry)), float(np.sin(ry))
            cz, sz = float(np.cos(rz)), float(np.sin(rz))

            rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
            rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
            rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            rot_mat = rot_x @ rot_y @ rot_z

            return (rot_mat.T @ (cam_w[:3] - trans[:3])) / float(scale)
        except Exception:
            return None

    def _grow_smooth_patch_local(
        self,
        obj: SceneObject,
        seed_face_idx: int,
        *,
        max_angle_deg: float = 35.0,
        radius_world: float = 0.0,
        max_faces: int = 12000,
        time_budget_s: float = 0.06,
        front_only: bool = True,
    ) -> set[int]:
        mesh = getattr(obj, "mesh", None)
        if mesh is None:
            return {int(seed_face_idx)}

        try:
            if getattr(mesh, "face_normals", None) is None:
                mesh.compute_normals(compute_vertex_normals=False)
        except Exception:
            _log_ignored_exception()

        normals = getattr(mesh, "face_normals", None)
        if normals is None:
            return {int(seed_face_idx)}

        n_faces = int(getattr(mesh, "n_faces", 0) or 0)
        seed = int(seed_face_idx)
        if seed < 0 or seed >= n_faces:
            return set()

        max_faces_i = max(1, int(max_faces))
        time_budget = float(max(0.01, min(float(time_budget_s), 1.0)))
        cos_thr = float(np.cos(np.radians(float(max_angle_deg))))

        # Build adjacency only for moderate meshes; use centroid KNN for huge meshes.
        adjacency = None
        try:
            edge_adj_max = int(getattr(self, "_surface_grow_edge_adj_max_faces", 250000))
        except Exception:
            edge_adj_max = 250000
        if n_faces <= max(5000, edge_adj_max):
            adjacency = self._get_face_adjacency(obj)

        centroids = getattr(obj, "_face_centroids", None)
        tree = getattr(obj, "_face_centroid_kdtree", None)
        if centroids is None or int(getattr(obj, "_face_centroid_faces_count", 0) or 0) != n_faces:
            try:
                f = np.asarray(getattr(mesh, "faces", None), dtype=np.int32)
                v = np.asarray(getattr(mesh, "vertices", None), dtype=np.float64)
                v0 = v[f[:, 0]]
                v1 = v[f[:, 1]]
                v2 = v[f[:, 2]]
                centroids = ((v0 + v1 + v2) / 3.0).astype(np.float32, copy=False)
            except Exception:
                centroids = None
            tree = None
            if centroids is not None and centroids.size != 0:
                try:
                    from scipy import spatial as _spatial

                    tree = cast(Any, _spatial).cKDTree(centroids)
                except Exception:
                    tree = None
            try:
                obj._face_centroids = centroids
                obj._face_centroid_kdtree = tree
                obj._face_centroid_faces_count = int(n_faces)
            except Exception:
                _log_ignored_exception()

        if adjacency is None and tree is None and centroids is not None and centroids.size != 0:
            try:
                from scipy import spatial as _spatial

                tree = cast(Any, _spatial).cKDTree(centroids)
                try:
                    obj._face_centroid_kdtree = tree
                except Exception:
                    _log_ignored_exception()
            except Exception:
                tree = None

        try:
            fn = np.asarray(normals, dtype=np.float64)
        except Exception:
            return {seed}

        seed_cent = None
        if centroids is not None and centroids.ndim == 2 and centroids.shape[0] == n_faces and centroids.shape[1] >= 3:
            try:
                seed_cent = np.asarray(centroids[seed, :3], dtype=np.float64).reshape(3)
            except Exception:
                seed_cent = None

        obj_scale = float(getattr(obj, "scale", 1.0))
        if abs(obj_scale) < 1e-12:
            obj_scale = 1.0
        r_world = float(max(0.0, radius_world))
        r_local = (r_world / abs(obj_scale)) if r_world > 0.0 else 0.0

        cam_local = self._camera_local_for_obj(obj) if bool(front_only) else None

        def is_front(fi: int) -> bool:
            if cam_local is None or seed_cent is None or centroids is None:
                return True
            try:
                c = np.asarray(centroids[fi, :3], dtype=np.float64).reshape(3)
                v = cam_local[:3].reshape(3) - c
                return float(np.dot(fn[fi, :3], v)) > 1e-12
            except Exception:
                return True

        def in_radius(fi: int) -> bool:
            if r_local <= 0.0 or seed_cent is None or centroids is None:
                return True
            try:
                c = np.asarray(centroids[fi, :3], dtype=np.float64).reshape(3)
                return float(np.linalg.norm(c - seed_cent)) <= float(r_local)
            except Exception:
                return True

        def iter_neighbors(fi: int) -> list[int]:
            if adjacency is not None:
                try:
                    return [int(x) for x in adjacency[int(fi)]]
                except Exception:
                    return []
            if tree is not None and centroids is not None:
                try:
                    k = int(getattr(self, "_surface_brush_knn", 18))
                except Exception:
                    k = 18
                k = max(6, min(k, 64, n_faces))
                try:
                    _d, idx = tree.query(np.asarray(centroids[int(fi), :3], dtype=np.float64), k=int(k))
                    return np.atleast_1d(idx).astype(np.int32, copy=False).tolist()
                except Exception:
                    return []
            return []

        visited: set[int] = {seed}
        stack: list[int] = [seed]
        start_t = time.monotonic()

        while stack:
            if len(visited) >= max_faces_i:
                break
            if time.monotonic() - start_t > time_budget:
                break

            fi = int(stack.pop())
            try:
                n0 = fn[fi, :3]
            except Exception:
                continue

            for nb in iter_neighbors(fi):
                nb_i = int(nb)
                if nb_i < 0 or nb_i >= n_faces or nb_i in visited:
                    continue
                try:
                    if float(np.dot(n0, fn[nb_i, :3])) < cos_thr:
                        continue
                    if not in_radius(nb_i):
                        continue
                    if not is_front(nb_i):
                        continue
                    visited.add(nb_i)
                    stack.append(nb_i)
                    if len(visited) >= max_faces_i:
                        break
                except Exception:
                    continue

        # Fallback: if front-only blocks expansion (e.g., flipped normals), retry once without it.
        if bool(front_only) and cam_local is not None and len(visited) <= 1:
            try:
                return self._grow_smooth_patch_local(
                    obj,
                    seed,
                    max_angle_deg=float(max_angle_deg),
                    radius_world=float(radius_world),
                    max_faces=int(max_faces),
                    time_budget_s=float(time_budget_s),
                    front_only=False,
                )
            except Exception:
                return visited

        return visited

    def _grow_smooth_patch_stepwise(
        self,
        obj: SceneObject,
        seed_face_idx: int,
        *,
        max_angle_deg: float = 30.0,
    ) -> set[int]:
        """
        Stepwise "magic-wand" grow:
        - Each Shift/Ctrl click expands the patch by a small number of BFS layers.
        - Keeps a small state so repeated clicks keep expanding instead of selecting everything at once.
        """
        mesh = getattr(obj, "mesh", None)
        if mesh is None:
            return {int(seed_face_idx)}

        try:
            if getattr(mesh, "face_normals", None) is None:
                mesh.compute_normals(compute_vertex_normals=False)
        except Exception:
            _log_ignored_exception()

        normals = getattr(mesh, "face_normals", None)
        if normals is None:
            return {int(seed_face_idx)}

        n_faces = int(getattr(mesh, "n_faces", 0) or 0)
        if n_faces <= 0:
            return set()

        seed_clicked = int(seed_face_idx)
        if seed_clicked < 0 or seed_clicked >= n_faces:
            return set()

        # Config knobs (tuned for usability).
        try:
            step_hops = int(getattr(self, "_surface_grow_step_hops", 2))
        except Exception:
            step_hops = 2
        step_hops = max(1, min(step_hops, 50))
        try:
            max_hops = int(getattr(self, "_surface_grow_max_hops", 80))
        except Exception:
            max_hops = 80
        max_hops = max(1, min(max_hops, 500))
        try:
            max_faces = int(getattr(self, "_surface_grow_max_faces", 120000))
        except Exception:
            max_faces = 120000
        max_faces = max(1, max_faces)
        try:
            time_budget_s = float(getattr(self, "_surface_grow_time_budget_s", 0.12))
        except Exception:
            time_budget_s = 0.12
        time_budget_s = max(0.02, min(time_budget_s, 2.0))
        try:
            radius_world = float(getattr(self, "_surface_grow_radius_world", 0.0) or 0.0)
        except Exception:
            radius_world = 0.0
        radius_world = max(0.0, radius_world)

        target = str(getattr(self, "_surface_paint_target", "outer")).strip().lower()
        if target not in {"outer", "inner", "migu"}:
            target = "outer"

        state = getattr(self, "_surface_grow_state", None)
        if state is None:
            state = {}
            self._surface_grow_state = state

        # Allow continuing growth by Shift-clicking any face inside the current patch.
        # (More forgiving than requiring the exact original seed.)
        active_obj_id = int(state.get("obj_id", -1) or -1)
        active_target = str(state.get("target", "") or "")
        active_seed = int(state.get("seed", -1) or -1)
        visited: set[int] = set(state.get("visited", set()) or set())

        continue_same = False
        if active_obj_id == id(obj) and active_target == target and visited and seed_clicked in visited:
            continue_same = True

        if not continue_same and (active_obj_id != id(obj) or active_target != target or active_seed != seed_clicked):
            visited = {seed_clicked}
            frontier = [seed_clicked]
            hops_done = 0
            state = {
                "obj_id": id(obj),
                "target": target,
                "seed": seed_clicked,
                "visited": visited,
                "frontier": frontier,
                "hops_done": hops_done,
            }
            self._surface_grow_state = state
        else:
            frontier = list(state.get("frontier", []) or [])
            hops_done = int(state.get("hops_done", 0) or 0)
            if not frontier:
                frontier = [int(state.get("seed", seed_clicked) or seed_clicked)]

        if hops_done >= max_hops or len(visited) >= max_faces:
            return set(visited)

        try:
            fn = np.asarray(normals, dtype=np.float64)
        except Exception:
            return set(visited)

        cos_thr = float(np.cos(np.radians(float(max_angle_deg))))
        start_t = time.monotonic()

        # Prefer edge adjacency if available; fall back to centroid KNN.
        adjacency = None
        try:
            edge_adj_max = int(getattr(self, "_surface_grow_edge_adj_max_faces", 250000))
        except Exception:
            edge_adj_max = 250000
        if n_faces <= max(5000, edge_adj_max):
            adjacency = self._get_face_adjacency(obj)

        centroids = getattr(obj, "_face_centroids", None)
        tree = getattr(obj, "_face_centroid_kdtree", None)
        if adjacency is None:
            if centroids is None or int(getattr(obj, "_face_centroid_faces_count", 0) or 0) != n_faces:
                try:
                    f = np.asarray(getattr(mesh, "faces", None), dtype=np.int32)
                    v = np.asarray(getattr(mesh, "vertices", None), dtype=np.float64)
                    v0 = v[f[:, 0]]
                    v1 = v[f[:, 1]]
                    v2 = v[f[:, 2]]
                    centroids = ((v0 + v1 + v2) / 3.0).astype(np.float32, copy=False)
                    try:
                        from scipy import spatial as _spatial

                        tree = cast(Any, _spatial).cKDTree(centroids)
                    except Exception:
                        tree = None
                except Exception:
                    centroids = None
                    tree = None
                try:
                    obj._face_centroids = centroids
                    obj._face_centroid_kdtree = tree
                    obj._face_centroid_faces_count = int(n_faces)
                except Exception:
                    _log_ignored_exception()

        if adjacency is None and tree is None and centroids is not None and centroids.size != 0:
            try:
                from scipy import spatial as _spatial

                tree = cast(Any, _spatial).cKDTree(centroids)
                try:
                    obj._face_centroid_kdtree = tree
                except Exception:
                    _log_ignored_exception()
            except Exception:
                tree = None

        seed_centroid = None
        if radius_world > 0.0:
            try:
                if centroids is None or len(centroids) != n_faces:
                    f = np.asarray(getattr(mesh, "faces", None), dtype=np.int32)
                    v = np.asarray(getattr(mesh, "vertices", None), dtype=np.float64)
                    v0 = v[f[:, 0]]
                    v1 = v[f[:, 1]]
                    v2 = v[f[:, 2]]
                    centroids = ((v0 + v1 + v2) / 3.0).astype(np.float64, copy=False)
                seed_centroid = np.asarray(centroids[int(state.get("seed", seed_clicked))], dtype=np.float64).reshape(-1)
            except Exception:
                seed_centroid = None
        try:
            obj_scale = float(getattr(obj, "scale", 1.0))
        except Exception:
            obj_scale = 1.0

        def iter_neighbors(fi: int) -> list[int]:
            if adjacency is not None:
                try:
                    return [int(x) for x in (adjacency[int(fi)] or [])]
                except Exception:
                    return []
            if tree is not None and centroids is not None:
                try:
                    k = int(getattr(self, "_surface_grow_knn", 18))
                except Exception:
                    k = 18
                k = max(6, min(k, 64, n_faces))
                try:
                    _d, idx = tree.query(np.asarray(centroids[int(fi)], dtype=np.float64), k=int(k))
                    return np.atleast_1d(idx).astype(np.int32, copy=False).tolist()
                except Exception:
                    return []
            return []

        # Expand by a small number of BFS layers per call.
        for _layer in range(step_hops):
            if hops_done >= max_hops or len(visited) >= max_faces:
                break
            if time.monotonic() - start_t > time_budget_s:
                break
            if not frontier:
                break

            next_frontier: list[int] = []
            for fi in frontier:
                if time.monotonic() - start_t > time_budget_s:
                    break
                if len(visited) >= max_faces:
                    break
                try:
                    n0 = fn[int(fi), :3]
                except Exception:
                    continue

                for nb in iter_neighbors(int(fi)):
                    nb_i = int(nb)
                    if nb_i == int(fi) or nb_i < 0 or nb_i >= n_faces or nb_i in visited:
                        continue
                    try:
                        n1 = fn[nb_i, :3]
                        if float(np.dot(n0, n1)) < cos_thr:
                            continue
                        if seed_centroid is not None and radius_world > 0.0 and centroids is not None:
                            cnb = np.asarray(centroids[nb_i], dtype=np.float64).reshape(-1)
                            if cnb.size >= 3 and seed_centroid.size >= 3:
                                if float(np.linalg.norm((cnb[:3] - seed_centroid[:3]) * obj_scale)) > radius_world:
                                    continue
                        visited.add(nb_i)
                        next_frontier.append(nb_i)
                    except Exception:
                        continue

            frontier = next_frontier
            hops_done += 1

        try:
            state["visited"] = visited
            state["frontier"] = frontier
            state["hops_done"] = int(hops_done)
            state["obj_id"] = id(obj)
            state["target"] = target
        except Exception:
            pass

        return set(visited)

    def _pick_surface_brush_face(self, pos, *, remove: bool, modifiers) -> None:
        obj = self.selected_obj
        if obj is None or obj.mesh is None:
            return

        info = self.pick_point_on_mesh_info(pos.x(), pos.y())
        if info is None:
            return
        point, depth_value, gl_x, gl_y, viewport, modelview, projection = info
        try:
            self._record_surface_paint_point(point, getattr(self, "_surface_paint_target", "outer"))
        except Exception:
            _log_ignored_exception()

        res = self.pick_face_at_point(point, return_index=True)
        if not res:
            return
        idx, _verts = res
        idx = int(idx)

        target = str(getattr(self, "_surface_paint_target", "outer"))
        target_set = self._get_surface_target_set(obj, target)

        # Brush stamps a small local patch so the user sees immediate feedback (not just points).
        try:
            base_r = float(getattr(self, "_surface_brush_radius_world", 0.0) or 0.0)
        except Exception:
            base_r = 0.0
        if base_r <= 0.0:
            try:
                px_r = float(getattr(self, "_surface_brush_radius_px", 48.0) or 0.0)
            except Exception:
                px_r = 0.0
            base_r = float(
                self._world_radius_from_px_at_depth(
                    int(gl_x),
                    int(gl_y),
                    float(depth_value),
                    viewport,
                    modelview,
                    projection,
                    float(px_r),
                )
            )
        if base_r <= 0.0:
            try:
                base_r = float(getattr(self, "grid_spacing", 1.0) or 1.0) * 0.75
            except Exception:
                base_r = 0.75
        try:
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                base_r *= 2.0
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                base_r *= 0.5
        except Exception:
            pass
        base_r = float(max(0.0, base_r))

        try:
            ang = float(getattr(self, "_surface_brush_max_angle_deg", 35.0) or 35.0)
        except Exception:
            ang = 35.0
        try:
            max_faces = int(getattr(self, "_surface_brush_max_faces", 12000) or 12000)
        except Exception:
            max_faces = 12000
        try:
            budget = float(getattr(self, "_surface_brush_time_budget_s", 0.06) or 0.06)
        except Exception:
            budget = 0.06
        try:
            front_only = bool(getattr(self, "_surface_brush_front_only", True))
        except Exception:
            front_only = True

        patch = self._grow_smooth_patch_local(
            obj,
            idx,
            max_angle_deg=float(ang),
            radius_world=float(base_r),
            max_faces=int(max_faces),
            time_budget_s=float(budget),
            front_only=bool(front_only),
        )
        if not patch:
            patch = {idx}

        if remove:
            target_set.difference_update(patch)
        else:
            target_set.update(patch)
            # Keep sets exclusive
            for s in self._get_other_surface_sets(obj, target):
                s.difference_update(patch)

        self._emit_surface_assignment_changed(obj)
        try:
            # Brush edits should not accumulate the click-grow state.
            self._surface_grow_state = {}
        except Exception:
            pass

    def _apply_surface_single_face(self, face_idx: int, modifiers, *, picked_point_world: np.ndarray | None = None) -> None:
        obj = self.selected_obj
        if obj is None or obj.mesh is None:
            return

        target = str(getattr(self, "_surface_paint_target", "outer"))
        target_set = self._get_surface_target_set(obj, target)

        if picked_point_world is not None:
            try:
                self._record_surface_paint_point(picked_point_world, target)
            except Exception:
                _log_ignored_exception()

        idx = int(face_idx)
        remove = bool(modifiers & Qt.KeyboardModifier.AltModifier)
        add = bool(modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier))

        if remove:
            target_set.discard(idx)
        elif add:
            target_set.add(idx)
            for s in self._get_other_surface_sets(obj, target):
                s.discard(idx)
        else:
            if idx in target_set:
                target_set.discard(idx)
            else:
                target_set.add(idx)
                for s in self._get_other_surface_sets(obj, target):
                    s.discard(idx)

        self._emit_surface_assignment_changed(obj)
        try:
            # Any non-grow edit resets the stepwise grow state.
            self._surface_grow_state = {}
        except Exception:
            pass

    def _apply_surface_seed_pick(
        self,
        seed_face_idx: int,
        modifiers,
        *,
        picked_point_world: np.ndarray | None = None,
        radius_world_override: float | None = None,
    ) -> None:
        obj = self.selected_obj
        if obj is None or obj.mesh is None:
            return

        target = str(getattr(self, "_surface_paint_target", "outer"))
        target_set = self._get_surface_target_set(obj, target)

        # Stepwise grow when Shift/Ctrl is held (Photoshop-like magic wand).
        # This prevents a single click from swallowing the whole shell surface.
        max_angle = float(getattr(self, "_surface_grow_max_angle_deg", 30.0) or 30.0)
        if modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier):
            patch = self._grow_smooth_patch_stepwise(obj, int(seed_face_idx), max_angle_deg=max_angle)
        else:
            try:
                r_click = (
                    float(radius_world_override)
                    if radius_world_override is not None
                    else float(getattr(self, "_surface_click_radius_world", 0.0) or 0.0)
                )
            except Exception:
                r_click = 0.0
            if r_click <= 0.0:
                try:
                    r_click = float(getattr(self, "grid_spacing", 1.0) or 1.0) * 1.5
                except Exception:
                    r_click = 1.5
            r_click = float(max(0.0, r_click))
            try:
                max_faces = int(getattr(self, "_surface_click_max_faces", 18000) or 18000)
            except Exception:
                max_faces = 18000
            try:
                budget = float(getattr(self, "_surface_click_time_budget_s", 0.08) or 0.08)
            except Exception:
                budget = 0.08
            try:
                front_only = bool(getattr(self, "_surface_click_front_only", True))
            except Exception:
                front_only = True

            patch = self._grow_smooth_patch_local(
                obj,
                int(seed_face_idx),
                max_angle_deg=float(max_angle),
                radius_world=float(r_click),
                max_faces=int(max_faces),
                time_budget_s=float(budget),
                front_only=bool(front_only),
            )
        if not patch:
            return

        if picked_point_world is not None:
            try:
                self._record_surface_paint_point(picked_point_world, target)
            except Exception:
                _log_ignored_exception()

        if modifiers & Qt.KeyboardModifier.AltModifier:
            target_set.difference_update(patch)
        elif modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier):
            target_set.update(patch)
            for s in self._get_other_surface_sets(obj, target):
                s.difference_update(patch)
        else:
            if int(seed_face_idx) in target_set:
                target_set.difference_update(patch)
            else:
                target_set.update(patch)
                for s in self._get_other_surface_sets(obj, target):
                    s.difference_update(patch)

        self._emit_surface_assignment_changed(obj)

    def pick_face_at_point(self, point: np.ndarray, return_index=False):
        """?뱀젙 3D 醫뚰몴媛 ?ы븿???쇨컖??硫댁쓽 ?뺤젏 3媛쒕? 諛섑솚"""
        obj = self.selected_obj
        if not obj or obj.mesh is None:
            return None
        
        # 硫붿돩 濡쒖뺄 醫뚰몴濡?蹂??(T/R/S ?????
        try:
            trans = np.asarray(getattr(obj, "translation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            if trans.size < 3:
                trans = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            rot_deg = np.asarray(getattr(obj, "rotation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            if rot_deg.size < 3:
                rot_deg = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            scale = float(getattr(obj, "scale", 1.0))
            if abs(scale) < 1e-12:
                scale = 1.0

            rx, ry, rz = np.radians(rot_deg[:3])
            cx, sx = float(np.cos(rx)), float(np.sin(rx))
            cy, sy = float(np.cos(ry)), float(np.sin(ry))
            cz, sz = float(np.cos(rz)), float(np.sin(rz))

            rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
            rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
            rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            rot_mat = rot_x @ rot_y @ rot_z
        except Exception:
            trans = np.asarray(getattr(obj, "translation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            if trans.size < 3:
                trans = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            scale = 1.0
            rot_mat = np.eye(3, dtype=np.float64)

        pt = np.asarray(point, dtype=np.float64).reshape(-1)
        if pt.size < 3:
            return None
        lp = (rot_mat.T @ (pt[:3] - trans[:3])) / float(scale)
        
        vertices = obj.mesh.vertices
        faces = obj.mesh.faces
        
        n_faces = int(getattr(obj.mesh, "n_faces", 0))
        if n_faces <= 0 or faces is None or len(faces) == 0:
            return None

        # Fast candidate search via cached face-centroid KDTree
        centroids = getattr(obj, "_face_centroids", None)
        tree = getattr(obj, "_face_centroid_kdtree", None)
        cached_n = int(getattr(obj, "_face_centroid_faces_count", 0) or 0)
        if centroids is None or cached_n != n_faces:
            try:
                f = np.asarray(faces, dtype=np.int32)
                v = np.asarray(vertices, dtype=np.float64)
                v0 = v[f[:, 0]]
                v1 = v[f[:, 1]]
                v2 = v[f[:, 2]]
                centroids = ((v0 + v1 + v2) / 3.0).astype(np.float32, copy=False)
            except Exception:
                centroids = None

            tree = None
            if centroids is not None and centroids.size != 0:
                try:
                    from scipy import spatial as _spatial

                    tree = cast(Any, _spatial).cKDTree(centroids)
                except Exception:
                    tree = None

            try:
                obj._face_centroids = centroids
                obj._face_centroid_kdtree = tree
                obj._face_centroid_faces_count = int(n_faces)
            except Exception:
                _log_ignored_exception()

        if tree is None and centroids is not None and centroids.size != 0:
            try:
                from scipy import spatial as _spatial

                tree = cast(Any, _spatial).cKDTree(centroids)
                try:
                    obj._face_centroid_kdtree = tree
                except Exception:
                    _log_ignored_exception()
            except Exception:
                tree = None

        cand = None
        try:
            k_pick = int(getattr(self, "_pick_face_knn", 64))
        except Exception:
            k_pick = 64
        k = int(max(12, min(k_pick, 256, n_faces)))
        if tree is not None:
            try:
                _d, idx = tree.query(np.asarray(lp[:3], dtype=np.float64), k=int(k))
                cand = np.atleast_1d(idx).astype(np.int32, copy=False)
            except Exception:
                cand = None

        if cand is None:
            if centroids is None or centroids.size == 0:
                return None
            diff = np.asarray(centroids, dtype=np.float64) - np.asarray(lp[:3], dtype=np.float64)
            dist2 = np.einsum("ij,ij->i", diff, diff)
            cand = np.array([int(np.argmin(dist2))], dtype=np.int32)

        def point_triangle_dist2(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            # From "Real-Time Collision Detection" (Christer Ericson)
            ab = b - a
            ac = c - a
            ap = p - a
            d1 = float(np.dot(ab, ap))
            d2 = float(np.dot(ac, ap))
            if d1 <= 0.0 and d2 <= 0.0:
                return float(np.dot(ap, ap))

            bp = p - b
            d3 = float(np.dot(ab, bp))
            d4 = float(np.dot(ac, bp))
            if d3 >= 0.0 and d4 <= d3:
                return float(np.dot(bp, bp))

            vc = d1 * d4 - d3 * d2
            if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                v = d1 / (d1 - d3)
                proj = a + v * ab
                d = p - proj
                return float(np.dot(d, d))

            cp = p - c
            d5 = float(np.dot(ab, cp))
            d6 = float(np.dot(ac, cp))
            if d6 >= 0.0 and d5 <= d6:
                return float(np.dot(cp, cp))

            vb = d5 * d2 - d1 * d6
            if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                w = d2 / (d2 - d6)
                proj = a + w * ac
                d = p - proj
                return float(np.dot(d, d))

            va = d3 * d6 - d5 * d4
            if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                proj = b + w * (c - b)
                d = p - proj
                return float(np.dot(d, d))

            denom = 1.0 / (va + vb + vc)
            v = vb * denom
            w = vc * denom
            proj = a + ab * v + ac * w
            d = p - proj
            return float(np.dot(d, d))

        p = np.asarray(lp[:3], dtype=np.float64)
        best_face_idx_any = -1
        best_d2_any = float("inf")
        best_face_idx_front = -1
        best_d2_front = float("inf")

        cam_local: np.ndarray | None = None
        try:
            cam_w = np.asarray(getattr(getattr(self, "camera", None), "position", None), dtype=np.float64).reshape(-1)
            if cam_w.size >= 3 and np.isfinite(cam_w[:3]).all():
                cam_local = (rot_mat.T @ (cam_w[:3] - trans[:3])) / float(scale)
        except Exception:
            cam_local = None

        for idx in cand.tolist():
            try:
                fi = int(idx)
                if fi < 0 or fi >= n_faces:
                    continue
                face = faces[fi]
                a = np.asarray(vertices[int(face[0])], dtype=np.float64)
                b = np.asarray(vertices[int(face[1])], dtype=np.float64)
                c = np.asarray(vertices[int(face[2])], dtype=np.float64)
                d2 = point_triangle_dist2(p, a, b, c)

                if d2 < best_d2_any:
                    best_d2_any = d2
                    best_face_idx_any = fi

                is_front = False
                if cam_local is not None:
                    try:
                        cent = (a + b + c) / 3.0
                        n_local = np.cross(b - a, c - a)
                        view_vec = cam_local[:3] - cent
                        is_front = float(np.dot(n_local, view_vec)) > 1e-12
                    except Exception:
                        is_front = False

                if is_front and d2 < best_d2_front:
                    best_d2_front = d2
                    best_face_idx_front = fi
                    if best_d2_front <= 1e-12:
                        break
            except Exception:
                continue
                
        best_face_idx = best_face_idx_front if best_face_idx_front != -1 else best_face_idx_any
        if best_face_idx != -1:
            best_face = faces[best_face_idx]
            v_list = []
            for vi in best_face:
                v = np.asarray(vertices[int(vi)], dtype=np.float64)
                w = (rot_mat @ (v[:3] * float(scale))) + trans[:3]
                v_list.append(w)
            if return_index:
                return best_face_idx, v_list
            return v_list
        return None

    def draw_picked_points(self):
        """李띿? ?먮뱾???묒? 援щ줈 ?쒓컖??(怨〓쪧/移섏닔 痢≪젙)"""
        if not self.picked_points and not getattr(self, "measure_picked_points", None):
            return
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)  # ??긽 ?욎뿉 蹂댁씠寃?
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        def draw_points(points: list[np.ndarray], color: tuple[float, float, float, float]):
            if not points:
                return
            glColor4f(float(color[0]), float(color[1]), float(color[2]), float(color[3]))
            for point in points:
                try:
                    x, y, z = float(point[0]), float(point[1]), float(point[2])
                except Exception:
                    continue
                glPushMatrix()
                glTranslatef(x, y, z)

                # ?묒? 援?(?몃쭖泥대줈 洹쇱궗) - ?ш린 0.08cm
                size = 0.08
                glBegin(GL_TRIANGLE_FAN)
                glVertex3f(0, 0, size)  # ?곷떒
                for j in range(9):
                    angle = 2.0 * np.pi * j / 8
                    glVertex3f(size * np.cos(angle), size * np.sin(angle), 0)
                glEnd()

                glBegin(GL_TRIANGLE_FAN)
                glVertex3f(0, 0, -size)  # ?섎떒
                for j in range(9):
                    angle = 2.0 * np.pi * j / 8
                    glVertex3f(size * np.cos(angle), size * np.sin(angle), 0)
                glEnd()

                glPopMatrix()

        # Curvature picks (magenta)
        draw_points(self.picked_points, (0.9, 0.2, 0.9, 0.9))

        # Measure picks (cyan)
        mp = getattr(self, "measure_picked_points", None) or []
        draw_points(mp, (0.1, 0.85, 0.95, 0.9))

        # If exactly 2 measure points exist, draw a line between them for clarity.
        if len(mp) == 2:
            try:
                glLineWidth(2.5)
                glColor4f(0.1, 0.85, 0.95, 0.85)
                glBegin(GL_LINES)
                glVertex3f(float(mp[0][0]), float(mp[0][1]), float(mp[0][2]))
                glVertex3f(float(mp[1][0]), float(mp[1][1]), float(mp[1][2]))
                glEnd()
                glLineWidth(1.0)
            except Exception:
                try:
                    glEnd()
                except Exception:
                    _log_ignored_exception()
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_surface_paint_points(self):
        """?쒕㈃ 吏??????誘멸뎄) 以?李띿? ???쒖떆"""
        pts = getattr(self, "surface_paint_points", None)
        if not pts:
            return

        try:
            glDisable(GL_LIGHTING)
            glDisable(GL_DEPTH_TEST)  # ??긽 蹂댁씠寃?
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            glPointSize(7.0)
            glBegin(GL_POINTS)
            for p, target in pts:
                try:
                    t = str(target or "").strip().lower()
                except Exception:
                    t = "outer"
                if t == "inner":
                    glColor4f(0.75, 0.35, 1.0, 0.95)
                elif t == "migu":
                    glColor4f(0.25, 0.85, 0.35, 0.95)
                else:
                    glColor4f(0.25, 0.65, 1.0, 0.95)

                try:
                    x, y, z = float(p[0]), float(p[1]), float(p[2])
                except Exception:
                    continue
                glVertex3f(x, y, z)
            glEnd()

        except Exception:
            try:
                glEnd()
            except Exception:
                _log_ignored_exception()
        finally:
            try:
                glDisable(GL_BLEND)
                glEnable(GL_DEPTH_TEST)
                glEnable(GL_LIGHTING)
            except Exception:
                _log_ignored_exception()

    def draw_surface_lasso_overlay(self) -> None:
        """?섎윭??吏???곸뿭) ?꾧뎄???붾㈃ ?ш?誘??ㅺ컖?? ?ㅻ쾭?덉씠"""
        try:
            if self.picking_mode != "paint_surface_area":
                return

            pts = getattr(self, "surface_lasso_points", None) or []
            preview = getattr(self, "surface_lasso_preview", None)
            if not pts and preview is None:
                return

            self.makeCurrent()
            viewport = glGetIntegerv(GL_VIEWPORT)
            try:
                vx, vy, w, h = [int(v) for v in viewport[:4]]
            except Exception:
                vx, vy, w, h = 0, 0, int(self.width()), int(self.height())
            if w <= 1 or h <= 1:
                return

            # Project world points -> viewport-relative pixels (top-left origin)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            screen_pts: list[tuple[float, float]] = []
            for p in pts:
                try:
                    win = gluProject(float(p[0]), float(p[1]), float(p[2]), modelview, projection, viewport)
                    if not win:
                        continue
                    xw, yw = float(win[0]), float(win[1])
                    # viewport origin may not be (0,0)
                    sx = xw - float(vx)
                    sy = float(int(vy + h - 1)) - yw
                    if np.isfinite(sx) and np.isfinite(sy):
                        screen_pts.append((float(sx), float(sy)))
                except Exception:
                    continue

            preview_xy = None
            if preview is not None:
                try:
                    gl_px, gl_py = self._qt_to_gl_window_xy(float(preview.x()), float(preview.y()), viewport=viewport)
                    px = float(int(gl_px) - int(vx))
                    py = float(int(vy + h - 1) - int(gl_py))
                    if np.isfinite(px) and np.isfinite(py):
                        preview_xy = (px, py)
                except Exception:
                    preview_xy = None

            glDisable(GL_LIGHTING)
            glDisable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # 2D ortho in pixel coords (top-left origin)
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0.0, float(w), float(h), 0.0, -1.0, 1.0)

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()

            # Snap-ready state (mouse near the start vertex)
            snap_px = 0
            try:
                snap_px = int(getattr(self, "_surface_area_close_snap_px", 12))
            except Exception:
                snap_px = 12
            snap_ready = False
            if (
                snap_px > 0
                and preview_xy is not None
                and len(screen_pts) >= 3
                and np.isfinite(screen_pts[0][0])
                and np.isfinite(screen_pts[0][1])
            ):
                try:
                    dx = float(preview_xy[0]) - float(screen_pts[0][0])
                    dy = float(preview_xy[1]) - float(screen_pts[0][1])
                    snap_ready = float(np.hypot(dx, dy)) <= float(snap_px)
                except Exception:
                    snap_ready = False

            # Line strip (polygon preview)
            glLineWidth(2.0)
            if snap_ready:
                glColor4f(0.2, 0.95, 0.35, 0.9)
            else:
                glColor4f(0.1, 0.6, 1.0, 0.85)
            glBegin(GL_LINE_STRIP)
            for xy in screen_pts:
                try:
                    x, y = float(xy[0]), float(xy[1])
                except Exception:
                    continue
                glVertex3f(x, y, 0.0)
            if preview_xy is not None:
                try:
                    glVertex3f(float(preview_xy[0]), float(preview_xy[1]), 0.0)
                except Exception:
                    _log_ignored_exception()
            glEnd()

            # Snap ring around the first vertex
            if screen_pts and snap_px > 0:
                try:
                    cx0, cy0 = float(screen_pts[0][0]), float(screen_pts[0][1])
                    seg = 28
                    glLineWidth(1.5)
                    if snap_ready:
                        glColor4f(0.2, 0.95, 0.35, 0.85)
                    else:
                        glColor4f(0.75, 0.75, 0.75, 0.55)
                    glBegin(GL_LINE_LOOP)
                    for j in range(seg):
                        a = 2.0 * float(np.pi) * float(j) / float(seg)
                        glVertex3f(cx0 + float(snap_px) * float(np.cos(a)), cy0 + float(snap_px) * float(np.sin(a)), 0.0)
                    glEnd()

                    # Optional closure hint line
                    if snap_ready and preview_xy is not None:
                        glLineWidth(2.0)
                        glColor4f(0.2, 0.95, 0.35, 0.75)
                        glBegin(GL_LINES)
                        glVertex3f(float(preview_xy[0]), float(preview_xy[1]), 0.0)
                        glVertex3f(cx0, cy0, 0.0)
                        glEnd()
                except Exception:
                    try:
                        glEnd()
                    except Exception:
                        _log_ignored_exception()

            # Vertices (start vertex emphasized)
            if screen_pts:
                try:
                    cx0, cy0 = float(screen_pts[0][0]), float(screen_pts[0][1])
                    glPointSize(9.0)
                    glBegin(GL_POINTS)
                    if snap_ready:
                        glColor4f(0.2, 0.95, 0.35, 0.95)
                    else:
                        glColor4f(1.0, 1.0, 1.0, 0.95)
                    glVertex3f(cx0, cy0, 0.0)
                    glEnd()
                except Exception:
                    try:
                        glEnd()
                    except Exception:
                        _log_ignored_exception()

            if len(screen_pts) > 1:
                glPointSize(6.0)
                glColor4f(0.95, 0.95, 0.95, 0.95)
                glBegin(GL_POINTS)
                for xy in screen_pts[1:]:
                    try:
                        x, y = float(xy[0]), float(xy[1])
                    except Exception:
                        continue
                    glVertex3f(x, y, 0.0)
                glEnd()

            # Restore matrices
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)

        except Exception:
            try:
                glEnd()
            except Exception:
                _log_ignored_exception()
        finally:
            try:
                glDisable(GL_BLEND)
                glEnable(GL_DEPTH_TEST)
                glEnable(GL_LIGHTING)
            except Exception:
                _log_ignored_exception()

    def draw_surface_magnetic_lasso_overlay(self) -> None:
        """寃쎄퀎(硫댁쟻+?먯꽍) ?꾧뎄???붾㈃ ?ш?誘??대━怨? ?ㅻ쾭?덉씠 (?붾뱶 ?ъ씤??湲곕컲)."""
        try:
            if self.picking_mode != "paint_surface_magnetic":
                return

            pts_world = getattr(self, "surface_lasso_points", None) or []
            cursor = getattr(self, "surface_lasso_preview", None)
            if cursor is None:
                cursor = getattr(self, "_surface_magnetic_cursor_qt", None)
            if not pts_world and cursor is None:
                return

            self.makeCurrent()
            viewport = glGetIntegerv(GL_VIEWPORT)
            try:
                vx, vy, w, h = [int(v) for v in viewport[:4]]
            except Exception:
                vx, vy, w, h = 0, 0, int(self.width()), int(self.height())
            if w <= 1 or h <= 1:
                return

            # Project world points -> viewport-relative pixels (top-left origin)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            screen_pts: list[tuple[float, float]] = []
            for p in pts_world:
                try:
                    win = gluProject(float(p[0]), float(p[1]), float(p[2]), modelview, projection, viewport)
                    if not win:
                        continue
                    xw, yw = float(win[0]), float(win[1])
                    sx = xw - float(vx)
                    sy = float(int(vy + h - 1)) - yw
                    if np.isfinite(sx) and np.isfinite(sy):
                        screen_pts.append((float(sx), float(sy)))
                except Exception:
                    continue

            preview = None
            preview_snapped = False
            if cursor is not None:
                try:
                    gl_x, gl_y = self._qt_to_gl_window_xy(float(cursor.x()), float(cursor.y()), viewport=viewport)
                    gl_sx, gl_sy = self._surface_magnetic_snap_gl(int(gl_x), int(gl_y))
                    preview_snapped = (int(gl_sx) != int(gl_x)) or (int(gl_sy) != int(gl_y))
                    px = float(int(gl_sx) - int(vx))
                    py = float(int(vy + h - 1) - int(gl_sy))
                    if np.isfinite(px) and np.isfinite(py):
                        preview = (px, py)
                except Exception:
                    preview = None
                    preview_snapped = False

            # Snap-ready state (mouse near the start vertex)
            close_px = 0
            try:
                close_px = int(getattr(self, "_surface_magnetic_close_snap_px", 12))
            except Exception:
                close_px = 12
            snap_ready = False
            if close_px > 0 and preview is not None and len(screen_pts) >= 3:
                try:
                    dx = float(preview[0]) - float(screen_pts[0][0])
                    dy = float(preview[1]) - float(screen_pts[0][1])
                    snap_ready = float(np.hypot(dx, dy)) <= float(close_px)
                except Exception:
                    snap_ready = False

            glDisable(GL_LIGHTING)
            glDisable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # 2D ortho in GL pixel coords (top-left origin)
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0.0, float(w), float(h), 0.0, -1.0, 1.0)

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()

            # Polyline
            if screen_pts:
                glLineWidth(2.0)
                if snap_ready:
                    glColor4f(0.2, 0.95, 0.35, 0.9)
                else:
                    glColor4f(0.15, 0.75, 1.0, 0.85)
                glBegin(GL_LINE_STRIP)
                for xy in screen_pts:
                    try:
                        glVertex3f(float(xy[0]), float(xy[1]), 0.0)
                    except Exception:
                        continue
                if preview is not None:
                    glVertex3f(float(preview[0]), float(preview[1]), 0.0)
                glEnd()

                # Snap ring around the first vertex (close-to-confirm)
                if close_px > 0:
                    try:
                        cx0, cy0 = float(screen_pts[0][0]), float(screen_pts[0][1])
                        seg = 28
                        glLineWidth(1.5)
                        if snap_ready:
                            glColor4f(0.2, 0.95, 0.35, 0.85)
                        else:
                            glColor4f(0.75, 0.75, 0.75, 0.55)
                        glBegin(GL_LINE_LOOP)
                        for j in range(seg):
                            a = 2.0 * float(np.pi) * float(j) / float(seg)
                            glVertex3f(cx0 + float(close_px) * float(np.cos(a)), cy0 + float(close_px) * float(np.sin(a)), 0.0)
                        glEnd()

                        if snap_ready and preview is not None:
                            glLineWidth(2.0)
                            glColor4f(0.2, 0.95, 0.35, 0.75)
                            glBegin(GL_LINES)
                            glVertex3f(float(preview[0]), float(preview[1]), 0.0)
                            glVertex3f(cx0, cy0, 0.0)
                            glEnd()
                    except Exception:
                        try:
                            glEnd()
                        except Exception:
                            _log_ignored_exception()

                # Vertices (start emphasized)
                try:
                    glPointSize(9.0)
                    if snap_ready:
                        glColor4f(0.2, 0.95, 0.35, 0.95)
                    else:
                        glColor4f(1.0, 1.0, 1.0, 0.95)
                    glBegin(GL_POINTS)
                    glVertex3f(float(screen_pts[0][0]), float(screen_pts[0][1]), 0.0)
                    glEnd()
                except Exception:
                    try:
                        glEnd()
                    except Exception:
                        _log_ignored_exception()

                if len(screen_pts) > 1:
                    glPointSize(6.0)
                    glColor4f(0.95, 0.95, 0.95, 0.95)
                    glBegin(GL_POINTS)
                    for xy in screen_pts[1:]:
                        try:
                            glVertex3f(float(xy[0]), float(xy[1]), 0.0)
                        except Exception:
                            continue
                    glEnd()

            # Cursor preview + snap radius
            if preview is not None:
                try:
                    snap_px = float(getattr(self, "_surface_magnetic_snap_radius_px", 14) or 14.0)
                except Exception:
                    snap_px = 14.0
                snap_px = float(max(0.0, min(snap_px, 400.0)))

                glPointSize(8.0)
                if preview_snapped:
                    glColor4f(0.2, 0.95, 0.35, 0.95)
                else:
                    glColor4f(0.95, 0.95, 0.95, 0.75)
                glBegin(GL_POINTS)
                glVertex3f(float(preview[0]), float(preview[1]), 0.0)
                glEnd()

                if snap_px > 0.0:
                    try:
                        seg = 32
                        glLineWidth(1.5)
                        if preview_snapped:
                            glColor4f(0.2, 0.95, 0.35, 0.6)
                        else:
                            glColor4f(0.85, 0.85, 0.85, 0.35)
                        glBegin(GL_LINE_LOOP)
                        for j in range(seg):
                            a = 2.0 * float(np.pi) * float(j) / float(seg)
                            glVertex3f(
                                float(preview[0]) + snap_px * float(np.cos(a)),
                                float(preview[1]) + snap_px * float(np.sin(a)),
                                0.0,
                            )
                        glEnd()
                    except Exception:
                        try:
                            glEnd()
                        except Exception:
                            _log_ignored_exception()

            # Restore matrices
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)

        except Exception:
            try:
                glEnd()
            except Exception:
                _log_ignored_exception()
        finally:
            try:
                glDisable(GL_BLEND)
                glEnable(GL_DEPTH_TEST)
                glEnable(GL_LIGHTING)
            except Exception:
                _log_ignored_exception()
    
    def draw_fitted_arc(self):
        """?쇳똿???먰샇 ?쒓컖??(?좏깮??媛앹껜??遺李⑸맖)"""
        obj = self.selected_obj
        if not obj or not obj.fitted_arcs:
            # ?꾩떆 ?먰샇??洹몃━湲?(?꾩쭅 媛앹껜??遺李?????寃쎌슦)
            if self.fitted_arc is not None:
                self._draw_single_arc(self.fitted_arc, None)
            return
        
        # 媛앹껜??遺李⑸맂 紐⑤뱺 ?먰샇 洹몃━湲?
        for arc in obj.fitted_arcs:
            self._draw_single_arc(arc, obj)
    
    def _draw_single_arc(self, arc, obj):
        """?⑥씪 ?먰샇 洹몃━湲?(?댁젣 ??긽 ?붾뱶 醫뚰몴 湲곗?)"""
        from src.core.curvature_fitter import CurvatureFitter
        
        glDisable(GL_LIGHTING)
        glColor3f(0.9, 0.2, 0.9)  # 留덉젨?
        glLineWidth(3.0)
        
        # ?먰샇 ?먮뱾 ?앹꽦
        fitter = CurvatureFitter()
        arc_points = fitter.generate_arc_points(arc, 64)
        
        # ??洹몃━湲?
        glBegin(GL_LINE_LOOP)
        for point in arc_points:
            glVertex3fv(point)
        glEnd()
        
        # 以묒떖?먯꽌 ?먯＜源뚯? ??(諛섏?由??쒖떆)
        glColor3f(1.0, 1.0, 0.0)  # ?몃???
        glBegin(GL_LINES)
        glVertex3fv(arc.center)
        glVertex3fv(arc_points[0])
        glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def draw_floor_picks(self):
        """諛붾떏硫?吏?????쒓컖??(??+ ?곌껐??"""
        if not self.floor_picks:
            return
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)  # 硫붿돩 ?욎뿉 ?쒖떆
        
        # ??洹몃━湲?(?뚮????먰삎 留덉빱)
        glColor3f(0.2, 0.4, 1.0)  # ?뚮???
        glPointSize(12.0)
        glBegin(GL_POINTS)
        for point in self.floor_picks:
            glVertex3fv(point)
        glEnd()
        
        # ???ъ씠 ?곌껐??(?몃???
        if len(self.floor_picks) >= 2:
            glColor3f(1.0, 0.9, 0.2)  # ?몃???
            glLineWidth(3.0)
            glBegin(GL_LINE_STRIP)
            for point in self.floor_picks:
                glVertex3fv(point)
            glEnd()
            
            # ??긽 ?쒖옉???앹젏 ?곌껐?섏뿬 ?곸뿭 ?쒖떆
            glBegin(GL_LINES)
            glVertex3fv(self.floor_picks[-1])
            glVertex3fv(self.floor_picks[0])
            glEnd()
            
            glBegin(GL_LINE_STRIP)
            for point in self.floor_picks:
                glVertex3fv(point)
            glEnd()
            
            # 諛섑닾紐??곸뿭 硫??쒖떆 (異⑸텇???먯씠 紐⑥씠硫?
            if len(self.floor_picks) >= 3:
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glColor4f(0.2, 0.8, 0.2, 0.3)  # 珥덈줉??諛섑닾紐?                # ?ㅺ컖??硫?(Triangle Fan)
                glBegin(GL_TRIANGLE_FAN)
                for point in self.floor_picks:
                    glVertex3fv(point)
                glEnd()
        
        # ??踰덊샇 ?쒖떆???묒? 留덉빱 (1, 2, 3)
        glColor3f(1.0, 1.0, 1.0)
        marker_size = 0.3
        for i, point in enumerate(self.floor_picks):
            glPushMatrix()
            glTranslatef(point[0], point[1], point[2] + 0.5)
            # ?レ옄 ????ш린濡?援щ텇 (1=?묒??? 2=以묎컙?? 3=?곗썝)
            size = marker_size * (i + 1)
            glBegin(GL_LINE_LOOP)
            for j in range(16):
                angle = 2.0 * np.pi * j / 16
                glVertex3f(size * np.cos(angle), size * np.sin(angle), 0)
            glEnd()
            glPopMatrix()
        
        glLineWidth(1.0)
        glPointSize(1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def clear_curvature_picks(self):
        """Clear curvature-picked points."""
        self.picked_points = []
        self.fitted_arc = None
        self.update()

    def clear_measure_picks(self) -> None:
        """Clear measure-picked points."""
        try:
            self.measure_picked_points = []
        except Exception:
            self.measure_picked_points = []
        self.update()


# ?뚯뒪?몄슜 ?ㅽ깲?쒖뼹濡??ㅽ뻾
if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    
    viewport = Viewport3D()
    viewport.setWindowTitle("3D Viewport Test")
    viewport.resize(800, 600)
    viewport.show()
    
    sys.exit(app.exec())
