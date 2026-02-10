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
from PyQt6.QtGui import QColor, QFont, QKeyEvent, QMouseEvent, QPainter, QWheelEvent
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
    glColor4fv,
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

            # 단면선이 꺾여 있어도, "외곽선 단면"은 하나의 평면 단면으로 정의되어야 1:1과 왜곡 없는 결과가 나옴.
            # 따라서 폴리라인의 첫/마지막 점을 잇는 직선 기준으로 단면 평면을 정의한다.
            p0 = pts[0]
            p1 = pts[-1]
            d = (p1 - p0).astype(np.float64)
            d[2] = 0.0
            length = float(np.linalg.norm(d))
            if length < 1e-6:
                self.computed.emit({"index": self._index, "profile": []})
                return

            d_unit = d / length
            world_normal = np.array([d_unit[1], -d_unit[0], 0.0], dtype=np.float64)

            local_origin = inv_scale * inv_rot @ (p0 - self._translation)
            local_normal = inv_rot @ world_normal

            contours_local = slicer.slice_with_plane(local_origin.tolist(), local_normal.tolist())
            if not contours_local:
                self.computed.emit({"index": self._index, "profile": []})
                return

            world_contours = []
            for cnt in contours_local:
                if cnt is None or len(cnt) < 2:
                    continue
                world_contours.append((rot_mat @ (cnt * self._scale).T).T + self._translation)

            # "작은 사각형(계단)"처럼 보이는 binning/envelope 대신,
            # 메쉬-평면 교차 폴리라인(단면 외곽)을 그대로 사용한다.
            best_profile = None
            best_score = -1.0

            for cnt in world_contours:
                arr = np.asarray(cnt, dtype=np.float64).reshape(-1, 3)
                if arr.shape[0] < 2:
                    continue

                s = (arr - p0) @ d_unit
                z = arr[:, 2]
                finite = np.isfinite(s) & np.isfinite(z)
                s = s[finite]
                z = z[finite]
                if s.size < 2:
                    continue

                # consecutive duplicates 제거
                if s.size >= 2:
                    ds = np.hypot(np.diff(s), np.diff(z))
                    keep = np.ones((s.size,), dtype=bool)
                    keep[1:] = ds > 1e-6
                    s = s[keep]
                    z = z[keep]
                if s.size < 2:
                    continue

                # 닫힘 보장
                try:
                    if float(np.hypot(s[0] - s[-1], z[0] - z[-1])) > 1e-6:
                        s = np.append(s, s[0])
                        z = np.append(z, z[0])
                except Exception:
                    _log_ignored_exception()

                # 여러 루프가 있으면 "가장 바깥"을 고르기 위해 (s,z) 면적 최대를 선택
                score = float(np.nanmax(s) - np.nanmin(s))
                if s.size >= 4:
                    try:
                        area2 = float(np.dot(s[:-1], z[1:]) - np.dot(z[:-1], s[1:]))
                        score = max(score, abs(area2))
                    except Exception:
                        _log_ignored_exception()

                if score > best_score:
                    best_score = score
                    best_profile = list(zip(s.tolist(), z.tolist()))

            if best_profile is None or len(best_profile) < 2:
                self.computed.emit({"index": self._index, "profile": []})
                return

            self.computed.emit({"index": self._index, "profile": best_profile})
        except Exception as e:
            self.failed.emit(str(e))


class _RoiCutEdgesThread(QThread):
    computed = pyqtSignal(object)  # {"x1": [...], "x2": [...], "y1": [...], "y2": [...]}
    failed = pyqtSignal(str)

    def __init__(
        self,
        mesh: MeshData,
        translation: np.ndarray,
        rotation_deg: np.ndarray,
        scale: float,
        roi_bounds: list[float],
    ):
        super().__init__()
        self._mesh = mesh
        self._translation = np.asarray(translation, dtype=np.float64)
        self._rotation = np.asarray(rotation_deg, dtype=np.float64)
        self._scale = float(scale)
        self._roi_bounds = np.asarray(roi_bounds, dtype=np.float64).reshape(-1)

    def run(self):
        try:
            from scipy.spatial.transform import Rotation as R

            if self._roi_bounds.size < 4:
                self.computed.emit({"x1": [], "x2": [], "y1": [], "y2": []})
                return

            x1, x2, y1, y2 = [float(v) for v in self._roi_bounds[:4]]
            inv_rot = R.from_euler('XYZ', self._rotation, degrees=True).inv().as_matrix()
            inv_scale = 1.0 / self._scale if self._scale != 0 else 1.0
            rot_mat = R.from_euler('XYZ', self._rotation, degrees=True).as_matrix()

            slicer = MeshSlicer(self._mesh.to_trimesh())

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

            # 4개의 경계 평면에서 단면선(경계선) 추출
            edges = {
                "x1": slice_world_plane(np.array([x1, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
                "x2": slice_world_plane(np.array([x2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
                "y1": slice_world_plane(np.array([0.0, y1, 0.0]), np.array([0.0, 1.0, 0.0])),
                "y2": slice_world_plane(np.array([0.0, y2, 0.0]), np.array([0.0, 1.0, 0.0])),
            }
            self.computed.emit(edges)
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
    """
    Trackball 스타일 카메라 (Z-up 좌표계)
    
    조작:
    - 좌클릭 드래그: 회전
    - 우클릭 드래그: 이동 (Pan)
    - 스크롤: 확대/축소
    
    좌표계: X-우, Y-앞, Z-상 (Z-up)
    """
    
    def __init__(self):
        # 카메라 위치 (구면 좌표)
        self.distance = 50.0  # cm
        self.azimuth = 45.0   # 수평 각도 (도) - XY 평면에서
        self.elevation = 30.0 # 수직 각도 (도) - Z축 기준
        
        # 회전 중심 (look-at point)
        self.center = np.array([0.0, 0.0, 0.0])
        
        # Pan 오프셋
        self.pan_offset = np.array([0.0, 0.0, 0.0])
        
        # 줌 제한 - 대폭 확장
        self.min_distance = 0.01
        # 1,000,000cm = 10km까지 확대 가능하게 하여 무한한 느낌 제공
        self.max_distance = 1000000.0
    
    @property
    def position(self) -> np.ndarray:
        """카메라 위치 (직교 좌표) - Z-up 좌표계"""
        az_rad = np.radians(self.azimuth)
        el_rad = np.radians(self.elevation)
        
        # Z-up 좌표계: X-우, Y-앞, Z-상
        x = self.distance * np.cos(el_rad) * np.cos(az_rad)
        y = self.distance * np.cos(el_rad) * np.sin(az_rad)
        z = self.distance * np.sin(el_rad)
        
        return np.array([x, y, z]) + self.center + self.pan_offset
    
    @property
    def up_vector(self) -> np.ndarray:
        """카메라 업 벡터 - Z-up, 단 상면/하면 뷰에서는 Y축 사용"""
        # 상면(elevation ≈ 90°) 또는 하면(elevation ≈ -90°)에서는 
        # 시선 방향이 Z축과 평행하므로 up 벡터를 Y축으로 변경
        if abs(self.elevation) > 85:
            # 상면: Y+ 방향이 화면 위쪽
            return np.array([0.0, 1.0, 0.0])
        return np.array([0.0, 0.0, 1.0])
    
    @property
    def look_at(self) -> np.ndarray:
        """시선 방향 타겟"""
        return self.center + self.pan_offset
    
    def rotate(self, delta_x: float, delta_y: float, sensitivity: float = 0.5):
        """카메라 회전"""
        self.azimuth -= delta_x * sensitivity  # 방향 반전
        self.elevation += delta_y * sensitivity
        
        # 수직 각도 제한 (-89 ~ 89도)
        self.elevation = max(-89.0, min(89.0, self.elevation))
    
    def pan(self, delta_x: float, delta_y: float, sensitivity: float = 0.3):
        """카메라 이동 (Pan) - 마우스 드래그 방향대로 화면이 '따라오게'"""
        # forward(시선)과 up(상면/하면 특수 처리 포함)으로 화면 기준 right/up 벡터 구성
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

        # Pan 속도 = 거리에 비례
        pan_speed = self.distance * sensitivity * 0.005

        # Grab-style: 마우스 방향대로 화면(메쉬)이 움직이게 카메라는 반대로 이동
        # delta_y는 화면 아래로 갈수록 증가(QT 좌표계)
        self.pan_offset -= right * (delta_x * pan_speed)
        self.pan_offset += up * (delta_y * pan_speed)
    
    def zoom(self, delta: float, sensitivity: float = 1.1):
        """카메라 줌"""
        if delta > 0:
            self.distance /= sensitivity
        else:
            self.distance *= sensitivity
        
        self.distance = max(self.min_distance, min(self.max_distance, self.distance))
    
    def reset(self):
        """카메라 초기화"""
        self.distance = 50.0
        self.azimuth = 45.0
        self.elevation = 30.0
        self.center = np.array([0.0, 0.0, 0.0])
        self.pan_offset = np.array([0.0, 0.0, 0.0])
    
    def fit_to_bounds(self, bounds):
        """메쉬 경계에 맞춰 카메라 자동 배치"""
        if bounds is None:
            return
            
        center = (bounds[0] + bounds[1]) / 2
        extents = bounds[1] - bounds[0]
        max_dim = np.max(extents)
        
        self.center = center
        self.pan_offset = np.array([0.0, 0.0, 0.0])
        self.distance = max_dim * 2.0
        self.azimuth = 45.0
        self.elevation = 30.0
        
    def move_relative(self, dx: float, dy: float, dz: float, sensitivity: float = 1.0):
        """카메라 로컬 좌표계 기준 이동 (WASD) - 현재 뷰 기준 직관적 방향"""
        az_rad = np.radians(self.azimuth)
        
        # 1. 오른쪽 벡터 (A/D)
        right = np.array([-np.sin(az_rad), np.cos(az_rad), 0])
        
        # 2. 전진 벡터 (W/S) - 카메라가 바라보는 방향의 수평 투영
        forward_h = np.array([-np.cos(az_rad), -np.sin(az_rad), 0])
        
        # 3. 위쪽 벡터 (Q/E) - 월드 Z
        up_v = np.array([0, 0, 1]) 
        
        # 이동 속도
        move_speed = (self.distance * 0.03 + 2.0) * sensitivity
        
        # 인자 적용 (dx:좌우, dy:상하, dz:전후)
        self.center += right * (dx * move_speed)
        self.center += up_v * (dy * move_speed)
        self.center += forward_h * (dz * move_speed)



    def apply(self):
        """OpenGL에 카메라 변환 적용"""
        try:
            pos = self.position
            target = self.look_at
            up = self.up_vector
            
            gluLookAt(
                pos[0], pos[1], pos[2],
                target[0], target[1], target[2],
                up[0], up[1], up[2]
            )
        except Exception:
            _log_ignored_exception()


class SceneObject:
    """씬 내의 개별 메쉬 객체 관리"""
    def __init__(self, mesh, name="Object"):
        self.mesh = mesh
        self.name = name
        self.visible = True
        self.color = [0.72, 0.72, 0.78]
        
        # 개별 변환 상태
        self.translation = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])
        self.scale = 1.0

        # 정치 고정 상태 (Bake 이후 복귀용)
        self.fixed_state_valid = False
        self.fixed_translation = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.fixed_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.fixed_scale = 1.0
        
        # 피팅된 원호들 (메쉬와 함께 이동)
        self.fitted_arcs = []

        # Saved overlay polylines (e.g., section/cut results) attached to this object.
        # Each item is a dict: {name, kind, points([[x,y,z],...]), visible, color([r,g,b,a]), width}
        self.polyline_layers = []
        
        # 렌더링 리소스
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
        """trimesh 객체 반환 (캐싱)"""
        if self._trimesh is None and self.mesh:
            self._trimesh = self.mesh.to_trimesh()
        return self._trimesh

    def get_world_bounds(self):
        """월드 좌표계에서의 경계 박스 반환"""
        if not self.mesh:
            return np.array([[0,0,0],[0,0,0]])
            
        # 로컬 바운드
        lb = self.mesh.bounds
        
        # 8개의 꼭짓점 생성
        v = np.array([
            [lb[0,0], lb[0,1], lb[0,2]], [lb[1,0], lb[0,1], lb[0,2]],
            [lb[0,0], lb[1,1], lb[0,2]], [lb[1,0], lb[1,1], lb[0,2]],
            [lb[0,0], lb[0,1], lb[1,2]], [lb[1,0], lb[0,1], lb[1,2]],
            [lb[0,0], lb[1,1], lb[1,2]], [lb[1,0], lb[1,1], lb[1,2]]
        ])

        # 월드 변환 적용 (R * (S * V) + T)
        rx, ry, rz = np.radians(self.rotation)
        cx, sx = float(np.cos(rx)), float(np.sin(rx))
        cy, sy = float(np.cos(ry)), float(np.sin(ry))
        cz, sz = float(np.cos(rz)), float(np.sin(rz))

        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
        rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
        rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        # OpenGL 적용 순서(glRotatef X->Y->Z)와 동일한 intrinsic 'xyz' (Rx @ Ry @ Rz)
        rot_mat = rot_x @ rot_y @ rot_z

        world_v = (rot_mat @ (v * float(self.scale)).T).T + self.translation

        return np.array([world_v.min(axis=0), world_v.max(axis=0)])
        
    def cleanup(self):
        if self.vbo_id is not None:
            # OpenGL 컨텍스트가 활성화된 상태에서 호출되어야 함
            try:
                glDeleteBuffers(1, [self.vbo_id])
            except Exception:
                _log_ignored_exception()


class Viewport3D(QOpenGLWidget):
    """
    OpenGL 3D 뷰포트 위젯
    
    기능:
    - 1cm 격자 바닥면
    - XYZ 축 표시
    - CloudCompare 스타일 카메라 조작
    - 메쉬 렌더링
    """
    
    # 시그널
    meshLoaded = pyqtSignal(object)  # 메쉬 로드됨
    meshTransformChanged = pyqtSignal()  # 직접 조작으로 변환됨
    selectionChanged = pyqtSignal(int) # 선택된 객체 인덱스
    floorPointPicked = pyqtSignal(np.ndarray) # 바닥면 점 선택됨
    floorFacePicked = pyqtSignal(list)        # 바닥면(면) 선택됨
    alignToBrushSelected = pyqtSignal()      # 브러시 선택 영역으로 정렬 요청
    floorAlignmentConfirmed = pyqtSignal()   # Enter 키로 정렬 확정 시 발생
    profileUpdated = pyqtSignal(list, list)  # x_profile, y_profile
    lineProfileUpdated = pyqtSignal(list)    # line_profile
    roiSilhouetteExtracted = pyqtSignal(list) # 2D silhouette points
    cutLinesAutoEnded = pyqtSignal()         # 단면선(2개) 입력 모드가 자동 종료됨
    faceSelectionChanged = pyqtSignal(int)   # 선택된 face 개수 변경
    surfaceAssignmentChanged = pyqtSignal(int, int, int)  # outer/inner/migu faces count
    measurePointPicked = pyqtSignal(np.ndarray)  # 치수 측정 점 선택됨 (월드 좌표)
    sliceScanRequested = pyqtSignal(float)   # 슬라이스 스캔 이동량(cm): Ctrl+휠
    sliceCaptureRequested = pyqtSignal(float)  # 현재 슬라이스 촬영 요청(Z cm): C
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 카메라
        self.camera = TrackballCamera()
        
        # 마우스 상태
        self.last_mouse_pos = None
        self.mouse_button = None
        
        # 씬 상태 상향 평준화 (멀티 메쉬)
        self.objects = []
        self.selected_index = -1
        self._mesh_center: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # 렌더링 설정
        self.grid_size = 500.0  # cm (더 크게 확장)
        self.grid_spacing = 1.0  # cm (1.0 = 1cm)
        self.bg_color = [0.96, 0.96, 0.94, 1.0] # #F5F5F0 (Cream/Beige)
        
        # 기즈모 설정
        self.show_gizmo = True
        self.active_gizmo_axis = None
        self.gizmo_size = 10.0
        self.gizmo_radius_factor = 1.15
        self.gizmo_drag_start = None
        
        # 곡률 측정 모드
        self.curvature_pick_mode = False
        self.picked_points = []
        self.fitted_arc = None

        # 치수 측정 모드 (거리/직경 등)
        self.measure_picked_points: list[np.ndarray] = []
        
        # 상태 표시용 텍스트
        self.status_info = ""
        self.surface_runtime_hud_enabled = True
        # Last assist/overlay performance snapshots for runtime HUD.
        self._surface_overlay_last_stats: dict[str, Any] = {}
        self.flat_shading = False # Flat shading 모드 (명암 없이 밝게 보기)
        # 기본 렌더는 "속이 꽉 찬" 느낌을 우선: 불투명 메쉬에서 back-face를 컬링.
        self.solid_shell_render = True
        # X-Ray 렌더링(선택된 객체만): 내부/후면도 함께 보이도록 투명 렌더링
        self.xray_mode = False
        self.xray_alpha = 0.25
        
        # 피킹 모드 ('none', 'curvature', 'floor_3point', 'floor_face', 'floor_brush')
        self.picking_mode = 'none'
        self.brush_selected_faces = set() # 브러시로 선택된 면 인덱스
        self._selection_brush_mode = "replace"  # replace|add|remove
        self._selection_brush_last_pick = 0.0
        # 큰 메쉬에서 클릭이 조금 빗나가면(배경 depth=1.0) 근처 픽셀을 탐색해서 피킹을 보정합니다.
        self._pick_search_radius_px = 8
        self._surface_paint_target = "outer"  # outer|inner|migu
        self._surface_brush_last_pick = 0.0
        # 표면 지정(찍기/브러시) 도구 크기: 화면(px) 기준 기본값(피킹 깊이에서 world로 환산).
        # NOTE: 큰 스캔 메쉬에서도 체감 크기가 너무 작지 않도록 기본값을 올려둡니다. ([ / ]로 조절 가능)
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
        self.floor_picks = []  # 바닥면 지정용 점 리스트
        
        # Undo/Redo 시스템
        self.undo_stack = []
        self.max_undo = 50
        
        # 단면 슬라이싱
        self.slice_enabled = False
        self.slice_z = 0.0
        self.slice_contours = []  # 현재 슬라이스 단면 폴리라인
        
        # 십자선 단면 (Crosshair)
        self.crosshair_enabled = False
        self.crosshair_pos = np.array([0.0, 0.0]) # XY 위치
        self.x_profile = [] # X축 단면 데이터 [(dist, z), ...]
        self.y_profile = [] # Y축 단면 데이터 [(dist, z), ...]
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
        self.active_roi_edge = None # 현재 드래그 중인 모서리 ('left', 'right', 'top', 'bottom')
        self.roi_rect_dragging = False  # 캡쳐 뜨듯이 드래그로 ROI 박스 지정
        self.roi_rect_start = None      # np.array([x,y])
        self._roi_bounds_changed = False
        self._roi_move_dragging = False
        self._roi_move_last_xy = None  # np.ndarray(2,) | None
        self._roi_handle_hit_px = 24
        self.roi_cut_edges: dict[str, list[np.ndarray]] = {"x1": [], "x2": [], "y1": [], "y2": []}  # ROI 잘림 경계선(월드 좌표)
        self.roi_cap_verts: dict[str, np.ndarray | None] = {"x1": None, "x2": None, "y1": None, "y2": None}  # ROI 캡(삼각형) 버텍스
        self.roi_section_world = {"x": [], "y": []}  # ROI로 얻은 단면(바닥 배치)
        # ROI 단면 "채움"(캡) 표시. 기본은 외곽선만 보이도록 OFF.
        self.roi_caps_enabled = False
        self._roi_edges_pending_bounds = None
        self._roi_edges_thread = None
        self._roi_edges_timer = QTimer(self)
        self._roi_edges_timer.setSingleShot(True)
        self._roi_edges_timer.timeout.connect(self._request_roi_edges_compute)

        # Cut guide lines (2 polylines on top view, SVG export용)
        self.cut_lines_enabled = False
        self.cut_lines = [[], []]  # 각 요소는 world 좌표 점 리스트 [np.array([x,y,z]), ...]
        self.cut_line_active = 0   # 0 or 1
        self.cut_line_drawing = False
        self.cut_line_preview = None  # np.array([x,y,z]) - 마지막 점에서 이어지는 프리뷰
        # 각 폴리라인의 "확정/완료" 상태. Enter/우클릭으로 True가 되며, Backspace/Delete로 편집 시 False로 돌아감.
        self._cut_line_final = [False, False]
        # 단면선 입력: '클릭' vs '드래그' 구분(이동/회전 중 점이 찍히는 문제 방지)
        self._cut_line_left_press_pos = None
        self._cut_line_left_dragged = False
        self._cut_line_right_press_pos = None
        self._cut_line_right_dragged = False
        self.cut_section_profiles = [[], []]  # 각 선의 (s,z) 프로파일 [(dist, z), ...]
        self.cut_section_world = [[], []]     # 바닥에 배치된 단면 폴리라인(월드 좌표)
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
        self.floor_penetration_highlight = True
        
        # 드래그 조작용 최적화 변수
        self._drag_depth = 0.0
        self._cached_viewport = None
        self._cached_modelview = None
        self._cached_projection = None
        self._ctrl_drag_active = False
        self._hover_axis = None
        
        # 키보드 조작 타이머 (WASD 연속 이동용)
        self.keys_pressed = set()
        self.move_timer = QTimer(self)
        self.move_timer.timeout.connect(self.process_keyboard_navigation)
        self.move_timer.setInterval(16) # ~60fps (필요 시만 start/stop)
        
        # UI 설정
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # 변환(이동/회전/스케일) 후 단면/ROI 등 파생 데이터를 디바운스 갱신
        self.meshTransformChanged.connect(self._on_mesh_transform_changed)

        # 렌더링은 입력/상태 변경 시에만 update()하도록 유지 (상시 60FPS 렌더링은 대용량 메쉬에서 버벅임 유발)
    
    @property
    def selected_obj(self) -> Optional[SceneObject]:
        if 0 <= self.selected_index < len(self.objects):
            return self.objects[self.selected_index]
        return None
    
    def initializeGL(self):
        """OpenGL 초기화"""
        glClearColor(0.95, 0.95, 0.95, 1.0) # 밝은 배경 (CloudCompare 스타일)
        # 기본 설정
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        
        # 선 부드럽게 (안티앨리어싱)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # 광원 모델 설정 (전역 환경광 낮춤)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.1, 0.1, 0.1, 1.0])
        
        # 광원 설정 (기본값)
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.0, 0.0, 0.0, 1.0]) # 개별 광원 ambient는 0으로
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
        
        # 노멀 정규화 활성화
        glEnable(GL_NORMALIZE)
        
        # 폴리곤 모드
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # 안티앨리어싱 (일부 드라이버에서 불안정할 수 있어 비활성화)
        # glEnable(GL_LINE_SMOOTH)
        # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    
    def resizeGL(self, w: int, h: int):
        """뷰포트 크기 변경"""
        glViewport(0, 0, w, h)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = w / h if h > 0 else 1.0
        # Far plane을 대폭 늘려 광활한 공간 확보 (기존 2000 -> 1,000,000)
        gluPerspective(45.0, aspect, 0.1, 1000000.0)
        
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """그리기"""
        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
        glLoadIdentity()
        
        # 카메라 적용
        self.camera.apply()

        # 바닥 관통(z<0) 하이라이트용 클리핑 평면 갱신 (카메라 이동/회전에 따라 매 프레임 필요)
        if self.floor_penetration_highlight:
            self._update_floor_penetration_clip_plane()
        if self.roi_enabled:
            self._update_roi_clip_planes()
        
        # 0. 광원 위치 업데이트 (밝고 균일한 조명)
        if not self.flat_shading:
            glEnable(GL_LIGHTING)
            
            # 환경광 높임 (전체적으로 밝게)
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.4, 0.4, 0.4, 1.0])
            
            # GL_LIGHT0: 정면 주 조명 (Headlight - 카메라 방향)
            glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 1.0, 0.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.6, 0.6, 0.6, 1.0])
            glLightfv(GL_LIGHT0, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
            
            # GL_LIGHT1: 보조 조명 (뒤쪽에서 - 그림자 완화)
            glEnable(GL_LIGHT1)
            glLightfv(GL_LIGHT1, GL_POSITION, [0.0, 0.0, -1.0, 0.0])
            glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.3, 0.3, 0.3, 1.0])
            glLightfv(GL_LIGHT1, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])
        else:
            glDisable(GL_LIGHTING)
            # Flat shading 시에는 모든 면이 일정 밝기로 보이게
            glColor3f(0.8, 0.8, 0.8)
        
        # 1. 격자 및 축 (Depth buffer에는 기록하지 않음: 메쉬 피킹/깊이 안정화)
        glDepthMask(GL_FALSE)
        self.draw_ground_plane()  # 반투명 바닥
        self.draw_grid()
        self.draw_axes()
        glDepthMask(GL_TRUE)
        
        # 2. 모든 메쉬 객체 렌더링
        sel = int(self.selected_index) if self.selected_index is not None else -1
        xray_enabled = bool(getattr(self, "xray_mode", False))
        for i, obj in enumerate(self.objects):
            if not obj.visible:
                continue

            # X-Ray는 선택된 객체를 루프 뒤에서 별도 렌더링(깊이버퍼 영향 최소화)
            if xray_enabled and i == sel:
                continue

            # ROI 클립은 선택된 객체에만 적용
            if self.roi_enabled and i == sel:
                try:
                    glEnable(GL_CLIP_PLANE1)
                    glEnable(GL_CLIP_PLANE2)
                    glEnable(GL_CLIP_PLANE3)
                    glEnable(GL_CLIP_PLANE4)
                except Exception:
                    _log_ignored_exception("Failed to enable ROI clip planes", level=logging.WARNING)

                self.draw_scene_object(obj, is_selected=True)
                # ROI로 잘린 면을 채워 단면 확인이 쉽게 보이도록 캡(뚜껑) 렌더링
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

        # X-Ray (선택된 객체만, 마지막에 렌더링)
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
            
        # 3. 오버레이 요소 (Depth write off: depth buffer는 메쉬만 유지)
        glDepthMask(GL_FALSE)

        # 3.1 곡률 피팅 요소
        self.draw_picked_points()
        self.draw_fitted_arc()

        # 3.2 표면 지정(찍은 점) 표시
        self.draw_surface_paint_points()
        # 3.3 표면 지정(면적/Area) 올가미 오버레이
        self.draw_surface_lasso_overlay()
        # 3.4 표면 지정(경계/자석) 올가미 오버레이
        self.draw_surface_magnetic_lasso_overlay()

        # 3.5 바닥 정렬 점 표시
        self.draw_floor_picks()

        # 3.6 단면 슬라이스 평면 및 단면선
        if self.slice_enabled:
            self.draw_slice_plane()
            self.draw_slice_contours()

        # 3.7 십자선 단면
        if self.crosshair_enabled:
            self.draw_crosshair()

        # 3.7.25 단면선(2개) 가이드
        if self.cut_lines_enabled or any(len(line) > 0 for line in getattr(self, "cut_lines", [])) or self._has_visible_polyline_layers():
            self.draw_cut_lines()

        # 3.7.5 선형 단면 (Top-view cut line)
        if self.line_section_enabled:
            self.draw_line_section()
             
        # 3.8 2D ROI 크로핑 영역
        if self.roi_enabled:
            self.draw_roi_cut_edges()
            self.draw_roi_box()

        glDepthMask(GL_TRUE)

        # 3.9 ROI 단면(얇은 슬라이스) 바닥 배치
        self.draw_roi_section_plots()
        
        # 4. 회전 기즈모 (선택된 객체에만, 피킹 모드 아닐 때만)
        if self.selected_obj and self.picking_mode == 'none':
            if not self.roi_enabled:
                self.draw_rotation_gizmo(self.selected_obj)
            # 메쉬 치수/중심점 오버레이
            self.draw_mesh_dimensions(self.selected_obj)
            
        # 5. UI 오버레이 (HUD)
        self.draw_orientation_hud()
        self.draw_surface_runtime_hud()

    def _update_floor_penetration_clip_plane(self):
        """월드 바닥(Z=0) 기준으로 '아래쪽'만 남기는 클리핑 평면 정의"""
        try:
            # Plane: z = 0, keep z <= 0  => -z >= 0
            glClipPlane(GL_CLIP_PLANE0, (0.0, 0.0, -1.0, 0.0))
        except Exception:
            # OpenGL 컨텍스트/프로파일에 따라 지원이 안 될 수 있음
            pass

    def _update_roi_clip_planes(self):
        """ROI bounds(x/y)로 선택 메쉬를 크로핑하는 4개 클리핑 평면 정의"""
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
        """반투명 바닥면 그리기 (Z=0, XY 평면) - Z-up 좌표계"""
        # 수평 뷰(정면/측면 등)에서는 바닥면이 선으로 보여 시야를 방해하므로 숨김
        if abs(self.camera.elevation) < 10:
            return
            
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # 양면 렌더링 (아래에서 봐도 보이게)
        glDisable(GL_CULL_FACE)
        
        # 바닥면 크기 (카메라 거리에 비례)
        size = max(self.camera.distance * 3, 200.0)
        
        # 메쉬가 바닥에 닿는지 감지
        touching_floor = False
        if self.selected_obj:
            obj = self.selected_obj
            try:
                # 전체 버텍스 스캔은 대용량 메쉬에서 매우 느림 -> 월드 바운드로 근사
                wb = obj.get_world_bounds()
                touching_floor = float(wb[0][2]) < 0.1
            except Exception:
                touching_floor = False
        
        # 바닥 색상: 닿으면 선명한 초록, 아니면 연한 회색/베이지
        if touching_floor:
            glColor4f(0.1, 0.9, 0.4, 0.4)  # 선명한 초록색
        else:
            glColor4f(0.85, 0.82, 0.78, 0.2)  # 연한 베이지-그레이
        
        # 정점 순서: 반시계 방향 = 위쪽이 앞면 (Z-up)
        glBegin(GL_QUADS)
        glVertex3f(-size, -size, 0)
        glVertex3f(size, -size, 0)
        glVertex3f(size, size, 0)
        glVertex3f(-size, size, 0)
        glEnd()
        
        glDisable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)
    
    def draw_grid(self):
        """무한 격자 바닥면 그리기 (Z=0, XY 평면) - Z-up 좌표계"""
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        
        # 카메라 거리에 따라 기본 간격 결정 (최소 1cm, 최대 10km)
        base_spacing = self.grid_spacing
        levels = [1, 10, 100, 1000] # 1cm, 10cm, 1m, 10m 단위
        
        cam_center = self.camera.look_at
        
        for level in levels:
            spacing = base_spacing * level
            
            # 카메라 거리에 비해 너무 조밀한 격자는 생략하여 성능/가시성 확보
            if spacing < self.camera.distance * 0.01:
                continue
                
            # 카메라 거리에 비해 너무 드문 격자도 드로잉 범위 조절
            view_range = spacing * 100
            # 너무 멀리 있는 격자는 생략
            if view_range < self.camera.distance * 0.5 and level < 1000:
                continue
                
            view_range = min(view_range, 100000.0) # 최대 1km
            half_range = view_range / 2
            
            # 투명도 조절 - 더 진하게
            if level == 1:
                alpha = 0.25
                line_width = 1.0
            elif level == 10:
                alpha = 0.4
                line_width = 1.5
            elif level == 100:
                alpha = 0.5
                line_width = 2.0
            else:  # 1000 (10m)
                alpha = 0.6
                line_width = 2.5
            
            glColor4f(0.5, 0.5, 0.5, alpha)
            glLineWidth(line_width)
            
            snap_x = round(cam_center[0] / spacing) * spacing
            snap_y = round(cam_center[1] / spacing) * spacing
            
            glBegin(GL_LINES)
            steps = 100
            for i in range(-steps // 2, steps // 2 + 1):
                offset = i * spacing
                # X 방향 라인 (Y에 평행)
                x_val = snap_x + offset
                glVertex3f(x_val, snap_y - half_range, 0)
                glVertex3f(x_val, snap_y + half_range, 0)
                
                # Y 방향 라인 (X에 평행)
                y_val = snap_y + offset
                glVertex3f(snap_x - half_range, y_val, 0)
                glVertex3f(snap_x + half_range, y_val, 0)
            glEnd()
            
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def draw_slice_plane(self):
        """현재 슬라이스 높이에 반투명 평면 그리기"""
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # 평면 크기 (그리드 크기와 맞춤)
        s = self.grid_size / 2
        z = self.slice_z
        
        # 반투명 빨간색 평면
        glColor4f(1.0, 0.0, 0.0, 0.15)
        glBegin(GL_QUADS)
        glVertex3f(-s, -s, z)
        glVertex3f(s, -s, z)
        glVertex3f(s, s, z)
        glVertex3f(-s, s, z)
        glEnd()
        
        # 경계선
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
        """추출된 단면선 그리기"""
        if not self.slice_contours:
            return
            
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glColor3f(1.0, 0.0, 1.0)  # 마젠타 색상 (눈에 띄게)
        
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
        """현재 Z 높이에서 단면 재추출"""
        obj = self.selected_obj
        if obj is None or obj.mesh is None:
            self.slice_contours = []
            return

        tm = obj.to_trimesh()
        if tm is None:
            self.slice_contours = []
            return

        slicer = MeshSlicer(cast(Any, tm))
        
        # 객체의 로컬 Z 좌표로 변환 필요 (현재는 월드 Z 기준 슬라이스 구현)
        # TODO: 객체 변환(회전, 이동) 반영 처리
        # 우선 가장 단순하게 월드 Z 기준 (평면 origin을 객체 로컬 좌표로 역변환하여 슬라이스)
        
        # (월드 Z) -> (로컬 좌표)
        # 로직: P_world = R * (S * P_local) + T
        # P_local = (1/S) * R^T * (P_world - T)

        from scipy.spatial.transform import Rotation as R
        inv_rot = R.from_euler('XYZ', obj.rotation, degrees=True).inv().as_matrix()
        inv_scale = 1.0 / obj.scale if obj.scale != 0 else 1.0
        
        # 월드 평면 [0, 0, 1] dot (P - [0, 0, Z_slice]) = 0
        # 로프 좌표에서의 평면 origin과 normal 계산
        world_origin = np.array([0.0, 0.0, float(self.slice_z)], dtype=np.float64)
        local_origin = inv_scale * inv_rot @ (world_origin - obj.translation)

        world_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        local_normal = inv_rot @ world_normal # 회전만 적용 (법선벡터이므로)

        self.slice_contours = slicer.slice_with_plane(local_origin.tolist(), local_normal.tolist())
        
        # 추출된 로컬 좌표 단면을 월드 좌표로 변환하여 저장 (렌더링용)
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
        """십자선 및 메쉬 투영 단면 시각화"""
        glDisable(GL_LIGHTING)
        
        cx, cy = self.crosshair_pos
        s = self.grid_size / 2
        
        # 1. 바닥 십자선 (연한 회색)
        glLineWidth(1.0)
        glColor4f(0.5, 0.5, 0.5, 0.5)
        glBegin(GL_LINES)
        glVertex3f(-s, cy, 0)
        glVertex3f(s, cy, 0)
        glVertex3f(cx, -s, 0)
        glVertex3f(cx, s, 0)
        glEnd()
        
        # 2. 메쉬 투영 단면 (강한 노란색)
        glLineWidth(3.0)
        glColor3f(1.0, 1.0, 0.0)
        
        # X축 프로파일 (Y 고정)
        if self.x_profile:
            glBegin(GL_LINE_STRIP)
            for d, z in self.x_profile:
                # d는 중심(cx)으로부터의 상대 거리일 수 있으므로 주의
                # 여기서는 월드 좌표계 [x, cy, z]로 그리도록 구현되어야 함
                pass # 아래에서 실제 포인트 렌더링
            glEnd()
            
        # 3. 실제 추출된 포인트들 렌더링 (월드 좌표계)
        # X 프로파일: X축 방향으로 가로지르는 선 (Y = cy)
        world_x_profile = getattr(self, '_world_x_profile', None)
        if world_x_profile is not None and len(world_x_profile) > 0:
            glColor3f(1.0, 1.0, 0.0) # Yellow
            glBegin(GL_LINE_STRIP)
            for pt in world_x_profile:
                glVertex3fv(pt)
            glEnd()
            
        # Y 프로파일: Y축 방향으로 가로지르는 선 (X = cx)
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
        """선형 단면(라인) 데이터 초기화"""
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

    def save_current_slice_to_layer(self) -> int:
        """현재 슬라이스만 레이어로 스냅샷 저장."""
        obj = self.selected_obj
        if obj is None:
            return 0
        contours = getattr(self, "slice_contours", None) or []
        if not bool(getattr(self, "slice_enabled", False)) or not contours:
            return 0

        if not hasattr(obj, "polyline_layers") or obj.polyline_layers is None:
            obj.polyline_layers = []

        def to_points_list(points) -> list[list[float]]:
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

        added = 0
        z_val = float(getattr(self, "slice_z", 0.0) or 0.0)
        for i, cnt in enumerate(contours):
            pts = to_points_list(cnt)
            if len(pts) < 2:
                continue
            suffix = f" #{i+1}" if len(contours) > 1 else ""
            name = self._unique_polyline_layer_name(obj, f"Slice-Z {z_val:.2f}cm{suffix}")
            obj.polyline_layers.append(
                {
                    "name": name,
                    "kind": "section_profile",
                    "points": pts,
                    "visible": True,
                    "offset": [0.0, 0.0],
                    "color": [0.75, 0.15, 0.75, 0.9],
                    "width": 2.0,
                }
            )
            added += 1

        if added:
            self.update()
        return int(added)

    def save_current_sections_to_layers(self) -> int:
        """현재 단면/가이드 결과를 스냅샷 레이어로 저장."""
        obj = self.selected_obj
        if obj is None:
            return 0

        if not hasattr(obj, "polyline_layers") or obj.polyline_layers is None:
            obj.polyline_layers = []

        def to_points_list(points) -> list[list[float]]:
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

        added = 0

        # 1) Cut guide lines (top view)
        names = ["단면선-가로", "단면선-세로"]
        colors = [
            (1.0, 0.25, 0.25, 0.95),
            (0.15, 0.55, 1.0, 0.95),
        ]
        for i, line in enumerate(getattr(self, "cut_lines", [[], []]) or []):
            pts = to_points_list(line)
            if len(pts) < 2:
                continue
            name = self._unique_polyline_layer_name(obj, names[i] if i < len(names) else f"단면선-{i+1}")
            obj.polyline_layers.append(
                {
                    "name": name,
                    "kind": "cut_line",
                    "points": pts,
                    "visible": True,
                    "offset": [0.0, 0.0],
                    "color": list(colors[i % len(colors)]),
                    "width": 2.0,
                }
            )
            added += 1

        # 2) Section profiles laid on the floor
        sec_names = ["단면-가로", "단면-세로"]
        for i, line in enumerate(getattr(self, "cut_section_world", [[], []]) or []):
            pts = to_points_list(line)
            if len(pts) < 2:
                continue
            name = self._unique_polyline_layer_name(obj, sec_names[i] if i < len(sec_names) else f"단면-{i+1}")
            obj.polyline_layers.append(
                {
                    "name": name,
                    "kind": "section_profile",
                    "points": pts,
                    "visible": True,
                    "offset": [0.0, 0.0],
                    "color": [0.1, 0.1, 0.1, 0.9],
                    "width": 2.0,
                }
            )
            added += 1

        # 3) ROI section profiles (if any)
        try:
            roi_sec = getattr(self, "roi_section_world", {}) or {}
            for key in ("x", "y"):
                line = roi_sec.get(key, None)
                pts = to_points_list(line)
                if len(pts) < 2:
                    continue
                axis = "가로" if key == "x" else "세로"
                name = self._unique_polyline_layer_name(obj, f"ROI-단면-{axis}")
                obj.polyline_layers.append(
                    {
                        "name": name,
                        "kind": "section_profile",
                        "points": pts,
                        "visible": True,
                        "offset": [0.0, 0.0],
                        "color": [0.1, 0.35, 0.1, 0.9],
                        "width": 2.0,
                    }
                )
                added += 1
        except Exception:
            _log_ignored_exception()

        # 4) Slice contours (if enabled)
        try:
            contours = getattr(self, "slice_contours", None) or []
            if bool(getattr(self, "slice_enabled", False)) and contours:
                z_val = float(getattr(self, "slice_z", 0.0) or 0.0)
                for i, cnt in enumerate(contours):
                    pts = to_points_list(cnt)
                    if len(pts) < 2:
                        continue
                    suffix = f" #{i+1}" if len(contours) > 1 else ""
                    name = self._unique_polyline_layer_name(obj, f"Slice-Z {z_val:.2f}cm{suffix}")
                    obj.polyline_layers.append(
                        {
                            "name": name,
                            "kind": "section_profile",
                            "points": pts,
                            "visible": True,
                            "offset": [0.0, 0.0],
                            "color": [0.75, 0.15, 0.75, 0.9],
                            "width": 2.0,
                        }
                    )
                    added += 1
        except Exception:
            _log_ignored_exception()

        if added:
            self.update()
        return int(added)

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

    def set_cut_lines_enabled(self, enabled: bool):
        self.cut_lines_enabled = bool(enabled)
        if enabled:
            self.picking_mode = 'cut_lines'
            # 프리뷰를 위해 마우스 트래킹 활성화
            self.setMouseTracking(True)
            try:
                self.setFocus()
            except Exception:
                _log_ignored_exception()
            try:
                idx = int(getattr(self, "cut_line_active", 0))
                idx = idx if idx in (0, 1) else 0
                line = self.cut_lines[idx]
                final = getattr(self, "_cut_line_final", [False, False])
                self.cut_line_drawing = bool(line) and not bool(final[idx])
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
        try:
            self._cut_section_pending_indices.clear()
        except Exception:
            _log_ignored_exception()
        self.update()

    def get_cut_lines_world(self):
        """내보내기용: 단면선(2개) 월드 좌표 반환 (순수 python list)"""
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
        """내보내기용: 바닥에 배치된 단면(프로파일) 폴리라인들(단면선/ROI) 월드 좌표 반환"""
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

        # ROI로 생성된 단면(있는 경우)도 함께 내보내기
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
        """메쉬 변환 시, 단면/ROI 등 의존 데이터를 디바운스 갱신."""
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
            if idx == 0:
                # 가로: 메쉬 상단에 배치 (s -> X, z -> Y)
                base_x = min_x
                base_y = max_y + margin
                scale_s = max(1e-6, extent_x) / s_span
            else:
                # 세로: 메쉬 우측에 배치 (z -> X, s -> Y)
                base_x = max_x + margin
                base_y = min_y
                scale_s = max(1e-6, extent_y) / s_span

            flip_s = False
            if idx != 0:
                # 사용자가 단면선을 위->아래(또는 오른쪽->왼쪽)로 그리면,
                # s 축 방향이 뒤집혀 결과가 상/하가 뒤집혀 보일 수 있음.
                # 단면선의 진행 방향이 +Y(세로) / +X(가로)가 되도록 s 축을 거울상으로 뒤집는다.
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
                if idx == 0:
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

        # 대기 중인 최신 요청이 있으면 즉시 다시 시도
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
        # 실패해도 UI는 계속 동작해야 함
        # print(f"Cut section compute failed: {message}")
        pass

    def draw_line_section(self):
        """상면(Top)에서 그은 직선 단면 시각화"""
        if self.line_section_start is None or self.line_section_end is None:
            return

        glDisable(GL_LIGHTING)

        p0 = self.line_section_start
        p1 = self.line_section_end

        # 1) 바닥 평면 위 커팅 라인 (오렌지)
        z = 0.05
        glLineWidth(2.5)
        glColor4f(1.0, 0.55, 0.0, 0.85)
        glBegin(GL_LINES)
        glVertex3f(float(p0[0]), float(p0[1]), z)
        glVertex3f(float(p1[0]), float(p1[1]), z)
        glEnd()

        # 2) 엔드포인트 마커
        # 3) 메쉬 단면선 (라임)
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

    def _cutline_constrain_ortho(self, last_pt: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        """CAD Ortho: 마지막 점 기준으로 X/Y 중 하나만 변화"""
        last_pt = np.asarray(last_pt, dtype=np.float64)
        candidate = np.asarray(candidate, dtype=np.float64).copy()
        dx = float(candidate[0] - last_pt[0])
        dy = float(candidate[1] - last_pt[1])
        if abs(dx) >= abs(dy):
            candidate[1] = last_pt[1]
        else:
            candidate[0] = last_pt[0]
        return candidate

    def _finish_cut_lines_current(self):
        """Enter/우클릭으로 현재 활성 단면선 입력을 마무리."""
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

        # 단면(프로파일) 계산 요청
        try:
            if idx_done in (0, 1) and len(self.cut_lines[idx_done]) >= 2:
                self.schedule_cut_section_update(idx_done, delay_ms=0)
        except Exception:
            _log_ignored_exception()

        # 다른 선이 "미확정"이면 자동 전환 (빈 선 포함)
        try:
            other = 1 - int(idx_done)
            final = getattr(self, "_cut_line_final", [False, False])
            if other in (0, 1) and not bool(final[other]):
                self.cut_line_active = other
                line = self.cut_lines[other]
                self.cut_line_drawing = bool(line) and not bool(final[other])
        except Exception:
            _log_ignored_exception()

        # 둘 다 확정되면 모드 자동 종료
        try:
            final = getattr(self, "_cut_line_final", [False, False])
            if bool(final[0]) and bool(final[1]):
                self.set_cut_lines_enabled(False)
                try:
                    self.cutLinesAutoEnded.emit()
                except Exception:
                    _log_ignored_exception()
        except Exception:
            _log_ignored_exception()

        self.update()

    def draw_cut_lines(self):
        """단면선(2개) 가이드 라인 시각화 (항상 화면 위로)"""
        lines = getattr(self, "cut_lines", [[], []])
        if not lines:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        z = 0.08  # 바닥에서 살짝 띄움
        colors = [
            (1.0, 0.25, 0.25, 0.95),  # red-ish
            (0.15, 0.55, 1.0, 0.95),  # blue-ish
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

        # 프리뷰 세그먼트
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

        # 단면 프로파일(바닥 배치) 렌더링
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
        """현재 선형 단면(라인)으로부터 프로파일 추출"""
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
        # 수직 단면 평면의 법선 (XY에서 라인에 수직)
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

        # 월드 좌표로 변환(렌더링/프로파일용)
        rot_mat = R.from_euler('XYZ', obj.rotation, degrees=True).as_matrix()
        trans = obj.translation
        scale = obj.scale

        world_contours = []
        for cnt in contours_local:
            w_cnt = (rot_mat @ (cnt * scale).T).T + trans
            world_contours.append(w_cnt)
        self.line_section_contours = world_contours

        # 프로파일: (거리, 높이) - 라인 방향으로 투영
        best_profile = []
        best_span = 0.0

        # 라인 세그먼트 범위로 필터링 (약간 여유)
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

            # 그래프는 0부터 시작하도록 shift
            t_sorted = t_sorted - float(t_sorted.min())

            best_profile = list(zip(t_sorted.tolist(), z_sorted.tolist()))
            best_span = span

        self.line_profile = best_profile
        self.lineProfileUpdated.emit(self.line_profile)
        self.update()

    def update_crosshair_profile(self):
        """현재 십자선 위치에서 단면 프로파일 추출"""
        if not self.selected_obj or self.selected_obj.mesh is None:
            self.x_profile = []
            self.y_profile = []
            return
            
        obj = self.selected_obj
        cx, cy = self.crosshair_pos
        
        # 1. 로컬 좌표계로 십자선 위치 변환
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

        # 2. X축 방향 단면 (평면: Y = cy)
        # 월드 상의 평면: Origin=[0, cy, 0], Normal=[0, 1, 0]
        w_orig_x = np.array([0, cy, 0])
        w_norm_x = np.array([0, 1, 0])
        l_orig_x = get_world_to_local(w_orig_x.reshape(1,3))[0]
        l_norm_x = inv_rot @ w_norm_x # 법선은 회전만
        
        # MeshData.section 에러 수정을 위해 to_trimesh() 사용
        tm = obj.to_trimesh()
        if tm is None:
            self.x_profile = []
            self.y_profile = []
            return

        slicer = MeshSlicer(cast(Any, tm))
        contours_x = slicer.slice_with_plane(l_orig_x.tolist(), l_norm_x.tolist())
        
        # 3. Y축 방향 단면 (평면: X = cx)
        w_orig_y = np.array([cx, 0, 0])
        w_norm_y = np.array([1, 0, 0]) # X축에 수직인 평면
        l_orig_y = get_world_to_local(w_orig_y.reshape(1,3))[0]
        l_norm_y = inv_rot @ w_norm_y
        contours_y = slicer.slice_with_plane(l_orig_y.tolist(), l_norm_y.tolist())
        
        # 4. 결과 처리 (그래프용 가공)
        # X 프로파일 (X축 따라 이동 시의 Z값)
        self.x_profile = []
        self._world_x_profile = []
        if contours_x:
            pts_local = np.vstack(contours_x)
            pts_world = get_local_to_world(pts_local)
            # X값 기준으로 정렬
            idx = np.argsort(pts_world[:, 0])
            sorted_pts = pts_world[idx]
            self._world_x_profile = sorted_pts
            # 그래프 데이터: (X좌표, Z좌표)
            self.x_profile = sorted_pts[:, [0, 2]].tolist()
            
        # Y 프로파일 (Y축 따라 이동 시의 Z값)
        self.y_profile = []
        self._world_y_profile = []
        if contours_y:
            pts_local = np.vstack(contours_y)
            pts_world = get_local_to_world(pts_local)
            # Y값 기준으로 정렬
            idx = np.argsort(pts_world[:, 1])
            sorted_pts = pts_world[idx]
            self._world_y_profile = sorted_pts
            # 그래프 데이터: (Y좌표, Z좌표)
            self.y_profile = sorted_pts[:, [1, 2]].tolist()
            
        self.profileUpdated.emit(self.x_profile, self.y_profile)
        self.update()

    def draw_roi_box(self):
        """2D ROI (크로핑 영역) 및 핸들 시각화"""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        x1, x2, y1, y2 = self.roi_bounds
        z = 0.08 # 바닥에서 살짝 띄움 (Z-fight 방지)

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
            glColor4f(0.0, 0.95, 1.0, 0.08)
            glBegin(GL_QUADS)
            glVertex3f(float(x1), float(y1), float(z))
            glVertex3f(float(x2), float(y1), float(z))
            glVertex3f(float(x2), float(y2), float(z))
            glVertex3f(float(x1), float(y2), float(z))
            glEnd()

            glLineWidth(2.0)
            glColor4f(0.0, 0.95, 1.0, 0.6)
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

        # 4방향 화살표 핸들 (간이 삼각형) - 화살표만 보이게
        def draw_arrow(cx, cy, direction):
            size = max(3.0, self.camera.distance * 0.03)

            if direction == 'top': # Y+
                verts = [(cx - size/2, cy, z), (cx + size/2, cy, z), (cx, cy + size, z)]
            elif direction == 'bottom': # Y-
                verts = [(cx - size/2, cy, z), (cx + size/2, cy, z), (cx, cy - size, z)]
            elif direction == 'left': # X-
                verts = [(cx, cy - size/2, z), (cx, cy + size/2, z), (cx - size, cy, z)]
            else: # right (X+)
                verts = [(cx, cy - size/2, z), (cx, cy + size/2, z), (cx + size, cy, z)]

            # Fill
            glBegin(GL_TRIANGLES)
            for vx, vy, vz in verts:
                glVertex3f(float(vx), float(vy), float(vz))
            glEnd()

            # Outline (high contrast)
            glColor4f(0.0, 0.0, 0.0, 0.85)
            glLineWidth(2.0)
            glBegin(GL_LINE_LOOP)
            for vx, vy, vz in verts:
                glVertex3f(float(vx), float(vy), float(vz))
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
        
        # 활성 상태에 따라 색상 변경
        def get_color(edge):
            return [1.0, 0.6, 0.0, 1.0] if self.active_roi_edge == edge else [0.0, 0.95, 1.0, 1.0]

        glColor4fv(get_color('bottom'))
        draw_arrow(mid_x, y1, 'bottom')
        glColor4fv(get_color('top'))
        draw_arrow(mid_x, y2, 'top')
        glColor4fv(get_color('left'))
        draw_arrow(x1, mid_y, 'left')
        glColor4fv(get_color('right'))
        draw_arrow(x2, mid_y, 'right')
        
        glLineWidth(1.0)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_roi_cut_edges(self):
        """ROI 클리핑으로 생기는 잘림 경계선(단면선) 오버레이"""
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
        """ROI로 잘린 단면을 '캡(뚜껑) 면'으로 채워 렌더링.

        NOTE: ROI 클립 플레인(GL_CLIP_PLANE1~4)이 enable 된 상태에서 호출하는 것을 권장.
        """
        caps = getattr(self, "roi_cap_verts", None)
        if not caps:
            return

        glDisable(GL_LIGHTING)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-1.0, -1.0)

        glBegin(GL_TRIANGLES)
        for key, verts in caps.items():
            if verts is None:
                continue

            # 살짝 색을 구분(좌우 x-plane은 붉은톤, 상하 y-plane은 푸른톤)
            if str(key).startswith("x"):
                glColor4f(1.0, 0.88, 0.88, 0.22)
            else:
                glColor4f(0.88, 0.94, 1.0, 0.22)

            try:
                for v in verts:
                    glVertex3f(float(v[0]), float(v[1]), float(v[2]))
            except Exception:
                continue
        glEnd()

        glDisable(GL_POLYGON_OFFSET_FILL)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    @staticmethod
    def _polygon_area2(points_2d: np.ndarray) -> float:
        """2D 폴리곤 signed area*2 (shoelace)."""
        pts = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 3:
            return 0.0
        x = pts[:, 0]
        y = pts[:, 1]
        return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @staticmethod
    def _sanitize_polygon_2d(points_2d: np.ndarray, max_points: int = 800, eps: float = 1e-6) -> Optional[np.ndarray]:
        """triangulation용 2D 폴리라인 정리(중복 제거/닫힘 제거/다운샘플)."""
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

        # 닫힌 루프면 마지막 점 제거
        try:
            if float(np.linalg.norm(pts[0] - pts[-1])) <= eps:
                pts = pts[:-1]
        except Exception:
            _log_ignored_exception()

        if pts.shape[0] < 3:
            return None

        # 연속 중복 제거
        if pts.shape[0] >= 2:
            d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            keep = np.ones((pts.shape[0],), dtype=bool)
            keep[1:] = d > eps
            pts = pts[keep]

        if pts.shape[0] < 3:
            return None

        # 너무 조밀하면 다운샘플
        if int(pts.shape[0]) > int(max_points):
            step = int(np.ceil(float(pts.shape[0]) / float(max_points)))
            step = max(1, step)
            pts = pts[::step]

        if pts.shape[0] < 3:
            return None

        # 다운샘플 후 다시 정리
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
        """단일 루프(홀 없음)용 ear-clipping 삼각분할. 실패 시 None."""
        pts = np.asarray(polygon_2d, dtype=np.float64).reshape(-1, 2)
        n = int(pts.shape[0])
        if n < 3:
            return None

        # CCW로 정렬
        idx_map = list(range(n))  # pts 인덱스 -> 입력 polygon_2d 인덱스
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
        """현재 ROI 잘림 경계선(roi_cut_edges)로 캡(삼각형) 버텍스를 갱신."""
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

            # 가장 큰(면적 기준) 루프 1개만 캡으로 사용
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

            # 루프를 못 만들면(예: 선분 조각만 존재) 전체 점으로 convex hull 캡
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
        """ROI로 생성된 단면을 바닥에 배치해 표시 (세로=우측, 가로=상단)"""
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

        self._roi_edges_thread = _RoiCutEdgesThread(
            obj.mesh,
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

        # 드래그 중 최신 bounds가 대기 중이면 다시 계산
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

        # ROI 캡(잘린 면 채우기) 갱신
        try:
            self._rebuild_roi_caps()
        except Exception:
            _log_ignored_exception()

        # ROI가 '얇아졌을 때' 단면을 바닥에 배치 (회전해서 봐도 단면 확인 가능)
        self.roi_section_world = {"x": [], "y": []}
        try:
            obj = self.selected_obj
            if obj is not None:
                b = np.asarray(obj.get_world_bounds(), dtype=np.float64)
                max_x, max_y = float(b[1][0]), float(b[1][1])
                margin = 5.0

                x1, x2, y1, y2 = [float(v) for v in self.roi_bounds]
                thin_th = 1.0  # cm: 이보다 얇으면 "단면"으로 간주

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

                # 세로 단면 (좌우 폭이 매우 얇을 때) -> 우측에 배치 (Y-Z)
                if abs(x2 - x1) <= thin_th:
                    main = pick_main(self.roi_cut_edges.get("x1", []) or self.roi_cut_edges.get("x2", []))
                    if main is not None and len(main) >= 2:
                        main = np.asarray(main, dtype=np.float64)
                        y_min = float(np.min(main[:, 1]))
                        z_min = float(np.min(main[:, 2]))
                        base_x = max_x + margin
                        base_y = float(b[0][1])
                        pts = []
                        for pt in main:
                            # YZ 단면을 "가로=Y, 세로=Z"로 배치 (높이가 위로 향하도록)
                            pts.append(
                                [
                                    base_x + (float(pt[1]) - y_min),
                                    base_y + (float(pt[2]) - z_min),
                                    0.0,
                                ]
                            )
                        self.roi_section_world["x"] = pts

                # 가로 단면 (상하 폭이 매우 얇을 때) -> 상단에 배치 (X-Z)
                if abs(y2 - y1) <= thin_th:
                    main = pick_main(self.roi_cut_edges.get("y1", []) or self.roi_cut_edges.get("y2", []))
                    if main is not None and len(main) >= 2:
                        main = np.asarray(main, dtype=np.float64)
                        x_min = float(np.min(main[:, 0]))
                        z_min = float(np.min(main[:, 2]))
                        base_y = max_y + margin
                        base_x = float(b[0][0])
                        pts = []
                        for pt in main:
                            # XZ 단면을 "가로=X, 세로=Z"로 배치 (높이가 위로 향하도록)
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
        """지정된 ROI 영역의 메쉬 외곽(실루엣) 추출"""
        if not self.selected_obj or self.selected_obj.mesh is None:
            return
            
        obj = self.selected_obj
        x1, x2, y1, y2 = self.roi_bounds
        
        # 1. 월드 좌표계의 모든 정점 가져오기
        from scipy.spatial.transform import Rotation as R
        rot_mat = R.from_euler('XYZ', obj.rotation, degrees=True).as_matrix()
        world_v = (rot_mat @ (obj.mesh.vertices * obj.scale).T).T + obj.translation
        
        # 2. ROI 영역 내의 점들 필터링
        mask = (world_v[:, 0] >= x1) & (world_v[:, 0] <= x2) & \
               (world_v[:, 1] >= y1) & (world_v[:, 1] <= y2)
        
        inside_v = world_v[mask]
        if len(inside_v) < 3:
            return
            
        # 3. 2D 투영 (XY 평면) 및 Convex Hull 또는 Alpha Shape로 외곽 추출
        # 여기서는 간단히 Convex Hull 사용 (추후 복잡한 형상은 Alpha Shape 필요)
        from scipy.spatial import ConvexHull
        points_2d = inside_v[:, :2]
        try:
            hull = ConvexHull(points_2d)
            silhouette = points_2d[hull.vertices]
            # 다시 2D 리스트 형태로 반환
            self.roiSilhouetteExtracted.emit(silhouette.tolist())
        except Exception:
            _log_ignored_exception()

    
    def draw_axes(self):
        """XYZ 축 그리기 (Z-up 좌표계)"""
        glDisable(GL_LIGHTING)
        
        glPushMatrix()
        
        axis_length = max(self.camera.distance * 100, 1000000.0)
        
        # 축 선 그리기
        glLineWidth(3.5)
        glBegin(GL_LINES)
        
        # X축 (빨강) - 좌우
        glColor3f(0.95, 0.2, 0.2)
        glVertex3f(-axis_length, 0, 0)
        glVertex3f(axis_length, 0, 0)
        
        # Y축 (초록) - 앞뒤 (깊이)
        glColor3f(0.2, 0.85, 0.2)
        glVertex3f(0, -axis_length, 0)
        glVertex3f(0, axis_length, 0)
        
        # Z축 (파랑) - 상하 (수직)
        glColor3f(0.2, 0.2, 0.95)
        glVertex3f(0, 0, -axis_length)
        glVertex3f(0, 0, axis_length)
        
        glEnd()
        glPopMatrix()
        
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)


    
    def _draw_axis_label_marker(self, x, y, z, size, axis):
        """X/Y 축 라벨을 간단한 기하학적 형태로 그리기 (XY 평면에 표시)"""
        glLineWidth(2.5)
        glBegin(GL_LINES)
        
        if axis == 'X':
            # X 모양 (YZ 평면에 표시, X축 끝에서)
            glVertex3f(x, -size, z + size)
            glVertex3f(x, size, z - size)
            glVertex3f(x, -size, z - size)
            glVertex3f(x, size, z + size)
        elif axis == 'Y':
            # Y 모양 (XZ 평면에 표시, Y축 끝에서)
            glVertex3f(-size, y, z + size)
            glVertex3f(0, y, z)
            glVertex3f(size, y, z + size)
            glVertex3f(0, y, z)
            glVertex3f(0, y, z)
            glVertex3f(0, y, z - size)
        
        glEnd()
    
    def _draw_axis_label_marker_z(self, x, y, z, size, axis):
        """Z 축 라벨을 간단한 기하학적 형태로 그리기 (XY 평면에 표시, Z축 끝에서)"""
        glLineWidth(2.5)
        glBegin(GL_LINES)
        
        # Z 모양 (XY 평면에 표시)
        glVertex3f(x - size, y + size, z)
        glVertex3f(x + size, y + size, z)
        glVertex3f(x + size, y + size, z)
        glVertex3f(x - size, y - size, z)
        glVertex3f(x - size, y - size, z)
        glVertex3f(x + size, y - size, z)
        
        glEnd()

        
    def draw_orientation_hud(self):
        """우측 하단에 작은 방향 가이드(HUD) 그리기"""
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # 1. 뷰포트 정보 저장
        viewport = glGetIntegerv(GL_VIEWPORT)
        w, h = viewport[2], viewport[3]
        
        # 2. 현재 뷰 행렬 가져오기 (회전 정보 추출용)
        view_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        
        # 3. HUD용 전용 뷰포트 설정 (우측 하단 80x80)
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
        
        # 4. 뷰 행렬에서 회전만 적용 (이동 제거)
        try:
            rot_matrix = np.array(view_matrix, dtype=np.float64).reshape(4, 4)
            # 이동 성분 제거 (Column-major 기준 12, 13, 14번째 요소가 4행 1,2,3열)
            # numpy array인 경우 r, c 인덱싱 사용
            rot_matrix[3, 0] = 0.0
            rot_matrix[3, 1] = 0.0
            rot_matrix[3, 2] = 0.0
            glLoadMatrixd(rot_matrix)
        except Exception:
            # 행렬 처리에 실패할 경우 HUD 회전 적용 생략 (최소한 크래시는 방지)
            pass
        
        # 5. 축 그리기 (X:Red, Y:Green, Z:Blue)
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
        
        # 5.5 축 라벨 (X, Y, Z) - 각 축 끝에 표시
        label_size = 0.12
        
        # X 라벨
        glColor3f(1.0, 0.2, 0.2)
        glBegin(GL_LINES)
        glVertex3f(1.1 - label_size, label_size, 0)
        glVertex3f(1.1 + label_size, -label_size, 0)
        glVertex3f(1.1 - label_size, -label_size, 0)
        glVertex3f(1.1 + label_size, label_size, 0)
        glEnd()
        
        # Y 라벨
        glColor3f(0.2, 1.0, 0.2)
        glBegin(GL_LINES)
        glVertex3f(-label_size, 1.1 + label_size, 0)
        glVertex3f(0, 1.1, 0)
        glVertex3f(label_size, 1.1 + label_size, 0)
        glVertex3f(0, 1.1, 0)
        glVertex3f(0, 1.1, 0)
        glVertex3f(0, 1.1 - label_size, 0)
        glEnd()
        
        # Z 라벨
        glColor3f(0.2, 0.2, 1.0)
        glBegin(GL_LINES)
        glVertex3f(-label_size, label_size, 1.1)
        glVertex3f(label_size, label_size, 1.1)
        glVertex3f(label_size, label_size, 1.1)
        glVertex3f(-label_size, -label_size, 1.1)
        glVertex3f(-label_size, -label_size, 1.1)
        glVertex3f(label_size, -label_size, 1.1)
        glEnd()
        
        # 6. 복구
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glViewport(0, 0, w, h)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_surface_runtime_hud(self):
        """표면 분리(assist/overlay) 런타임 계측값을 화면 좌측 상단에 표시."""
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
        """개별 메쉬 객체 렌더링"""
        glPushMatrix()

        # 변환 적용
        glTranslatef(*obj.translation)
        glRotatef(obj.rotation[0], 1, 0, 0)
        glRotatef(obj.rotation[1], 0, 1, 0)
        glRotatef(obj.rotation[2], 0, 0, 1)
        glScalef(obj.scale, obj.scale, obj.scale)

        alpha_f = float(alpha)
        if not np.isfinite(alpha_f):
            alpha_f = 1.0
        alpha_f = max(0.0, min(alpha_f, 1.0))
        solid_shell = bool(getattr(self, "solid_shell_render", True)) and alpha_f >= 0.999

        if not depth_write:
            glDepthMask(GL_FALSE)

        if solid_shell:
            # 기본 표시에서는 back-face를 숨겨 "속이 꽉 찬" 형태로 보이게 합니다.
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
        else:
            glDisable(GL_CULL_FACE)
        
        # 메쉬 재질 및 밝기 최적화 (광택 추가로 굴곡 강조)
        if not self.flat_shading:
            glEnable(GL_LIGHTING)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 32.0)
        else:
            glDisable(GL_LIGHTING)
            glDisable(GL_COLOR_MATERIAL)
        
        # 메쉬 색상
        if is_selected:
            col = (0.85, 0.85, 0.95)  # 너무 하얗지 않게 약간 톤다운
        else:
            col = tuple(float(c) for c in (obj.color or [0.72, 0.72, 0.78])[:3])

        if alpha_f < 1.0:
            glColor4f(float(col[0]), float(col[1]), float(col[2]), float(alpha_f))
        else:
            # glColor3f는 alpha를 건드리지 않으므로 이전 draw의 alpha가 남을 수 있습니다.
            # 불투명 메쉬는 alpha=1.0을 명시해 의도치 않은 내부 비침을 막습니다.
            glColor4f(float(col[0]), float(col[1]), float(col[2]), 1.0)
            
        # 브러시로 선택된 면 하이라이트 (임시 오버레이)
        if is_selected and self.picking_mode == 'floor_brush' and self.brush_selected_faces:
            glPushMatrix()
            glDisable(GL_LIGHTING)
            # 메쉬보다 아주 약간 앞에 그리기 (Z-fight 방지)
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

        # 선택된 면 하이라이트 (SelectionPanel)
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

        if obj.vbo_id is not None:
            # VBO 방식 렌더링
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)

            glBindBuffer(GL_ARRAY_BUFFER, obj.vbo_id)
            glVertexPointer(3, GL_FLOAT, 24, ctypes.c_void_p(0))
            glNormalPointer(GL_FLOAT, 24, ctypes.c_void_p(12))

            # 1) 기본 색상 렌더링
            glDrawArrays(GL_TRIANGLES, 0, obj.vertex_count)

            # 2) 바닥 관통(z<0) 영역을 초록색으로 덮어쓰기(클리핑 평면 이용, CPU 스캔 없음)
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
        
        # 바닥 접촉 면 하이라이트는 정치(바닥 정렬) 관련 모드에서만 표시 (대용량 메쉬 성능)
        if is_selected and self.picking_mode in {'floor_3point', 'floor_face', 'floor_brush'}:
            self._draw_floor_contact_faces(obj)

        glDisable(GL_CULL_FACE)
        glPopMatrix()

        if not depth_write:
            glDepthMask(GL_TRUE)
    
    def _draw_floor_contact_faces(self, obj: SceneObject):
        """바닥(Z=0) 근처 면을 초록색으로 하이라이트 (정치 과정 중 표시)"""
        if obj.mesh is None or obj.mesh.faces is None:
            return
        
        faces = obj.mesh.faces
        vertices = obj.mesh.vertices
        
        # 회전 행렬 계산
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler('XYZ', obj.rotation, degrees=True).as_matrix()
        
        total_faces = len(faces)
        
        # 샘플링 (대형 메쉬)
        sample_size = min(80000, total_faces)
        if total_faces > sample_size:
            indices = np.random.choice(total_faces, sample_size, replace=False)
        else:
            indices = np.arange(total_faces)
        
        sample_faces = faces[indices]
        v_indices = sample_faces[:, 0]
        v_points = vertices[v_indices] * obj.scale
        
        # 월드 Z 좌표 계산
        world_z = (r[2, 0] * v_points[:, 0] + 
                   r[2, 1] * v_points[:, 1] + 
                   r[2, 2] * v_points[:, 2]) + obj.translation[2]
        
        # 바닥 근처 감지 (Z < 0.5cm 또는 Z < 0)
        # 정치 모드에서는 바닥 근처(0.5cm 이내)까지 표시
        threshold = 0.5 if self.picking_mode == 'floor_3point' else 0.0
        near_floor_mask = world_z < threshold
        near_floor_indices = indices[np.where(near_floor_mask)[0]]
        
        if len(near_floor_indices) == 0:
            return
        
        # 초록색 채우기
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-4.0, -4.0)
        
        # 색상: 바닥 아래(Z<0)는 진한 초록, 근처(0<Z<0.5)는 연한 초록
        glBegin(GL_TRIANGLES)
        for face_idx in near_floor_indices[:20000]:
            f = faces[face_idx]
            # 이 면의 Z 값 확인
            v0_z = world_z[np.where(indices == face_idx)[0][0]] if face_idx in indices else 0
            # 수평 시점이나 하면 시점(Elevation < 0)에서는 더 투명하게 처리하여 메쉬를 가리지 않게 함
            is_bottom_view = self.camera.elevation < -45
            alpha_penetrate = 0.1 if is_bottom_view else 0.4
            alpha_near = 0.05 if is_bottom_view else 0.2
            
            if v0_z < 0:
                glColor4f(0.0, 1.0, 0.2, alpha_penetrate)  # 진한 초록 (관통)
            else:
                glColor4f(0.5, 1.0, 0.5, alpha_near)  # 연한 초록 (근처)
            for v_idx in f:
                glVertex3fv(vertices[v_idx])
        glEnd()
        
        # 외곽선
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glColor4f(0.0, 0.5, 0.1, 0.5)
        glBegin(GL_TRIANGLES)
        for face_idx in near_floor_indices[:8000]:
            f = faces[face_idx]
            for v_idx in f:
                glVertex3fv(vertices[v_idx])
        glEnd()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glPopAttrib()
    
    def draw_mesh_dimensions(self, obj: SceneObject):
        """메쉬 중심점 십자선 표시 (드래그로 이동 가능)"""
        if obj.mesh is None:
            return

        # 월드 좌표에서 바운딩 박스 계산 (대용량 메쉬에서도 O(1))
        wb = obj.get_world_bounds()
        min_pt = wb[0]
        max_pt = wb[1]
        
        center_x = (min_pt[0] + max_pt[0]) / 2
        center_y = (min_pt[1] + max_pt[1]) / 2
        z = min_pt[2] + 0.1  # 바닥 살짝 위
        
        # 중심점 저장 (드래그용)
        self._mesh_center = np.array([center_x, center_y, z])
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # 작은 십자선 (빨간색)
        glColor3f(1.0, 0.3, 0.3)
        glLineWidth(2.0)
        marker_size = 1.5  # 고정 크기 1.5cm
        glBegin(GL_LINES)
        glVertex3f(center_x - marker_size, center_y, z)
        glVertex3f(center_x + marker_size, center_y, z)
        glVertex3f(center_x, center_y - marker_size, z)
        glVertex3f(center_x, center_y + marker_size, z)
        glEnd()
        
        # 원점 표시 (녹색 작은 원)
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
        """회전 기즈모 그리기"""
        if not self.show_gizmo:
            return
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        glPushMatrix()
        # 선택된 객체의 위치로 이동
        glTranslatef(*obj.translation)
        
        # 기즈모 크기 설정 (객체 스케일 반영)
        size = self.gizmo_size * obj.scale
        
        # 하이라이트용 축 (hover 또는 active)
        highlight_axis = self.active_gizmo_axis or getattr(self, '_hover_axis', None)
        
        # X축
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
        
        # Y축
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
        
        # Z축
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
        """기즈모용 원 그리기"""
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            glVertex3f(x, y, 0)
        glEnd()

    def draw_wireframe(self, obj: SceneObject):
        """와이어프레임 오버레이"""
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
        """새 메쉬를 씬에 추가"""
        if name is None:
            name = f"Object_{len(self.objects) + 1}"
            
        # Optional precomputed caches from the loader thread (performance on huge meshes).
        pre_centroids = getattr(mesh, "_amr_face_centroids", None)
        try:
            pre_centroids_faces = int(getattr(mesh, "_amr_face_centroids_faces_count", 0) or 0)
        except Exception:
            pre_centroids_faces = 0

        # 메쉬 자체를 원점으로 센터링 (로컬 좌표계 생성)
        center = mesh.centroid
        mesh.vertices -= center
        # 캐시 무효화 (vertices 변경)
        try:
            mesh._bounds = None
            mesh._centroid = None
            mesh._surface_area = None
        except Exception:
            _log_ignored_exception()
        # 로딩 시점에는 face normals만 필요 (vertex normals는 필요할 때 계산)
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
        
        # VBO 데이터 생성
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
        
        # 카메라 피팅 (첫 번째 객체인 경우만)
        if len(self.objects) == 1:
            self.update_grid_scale()
        
        self.meshLoaded.emit(mesh)
        self.selectionChanged.emit(self.selected_index)
        self.update()

    def clear_scene(self) -> None:
        """씬의 모든 객체/오버레이를 제거하고 기본 상태로 리셋합니다."""
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
        self.roi_bounds = [-10.0, 10.0, -10.0, 10.0]
        self.roi_cut_edges = {"x1": [], "x2": [], "y1": [], "y2": []}
        self.roi_cap_verts = {"x1": None, "x2": None, "y1": None, "y2": None}
        self.roi_section_world = {"x": [], "y": []}
        self.roi_caps_enabled = False

        self.cut_lines_enabled = False
        self.cut_lines = [[], []]
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
        """선택된 메쉬 크기에 맞춰 격자 스케일 조정"""
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
        """선택된 메쉬 크기에 맞춰 회전 기즈모 반경 조정"""
        obj = self.selected_obj
        if not obj or getattr(obj, "mesh", None) is None:
            return

        try:
            bounds = obj.mesh.bounds
            extents = bounds[1] - bounds[0]
            max_dim = float(np.max(extents))
        except Exception:
            return

        factor = float(getattr(self, "gizmo_radius_factor", 1.15))
        factor = max(1.01, min(3.0, factor))
        self.gizmo_radius_factor = factor
        self.gizmo_size = max_dim * 0.5 * factor
    
    def hit_test_gizmo(self, screen_x, screen_y):
        """기즈모 고리 클릭 검사"""
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
        """객체의 VBO 생성 및 데이터 전송"""
        try:
            if obj is None or obj.mesh is None:
                return

            if obj.mesh.face_normals is None:
                obj.mesh.compute_normals(compute_vertex_normals=False)

            faces = obj.mesh.faces
            v_indices = faces.reshape(-1)
            vertex_count = int(v_indices.size)

            # [vx,vy,vz,nx,ny,nz] float32 interleaved (avoid huge temporaries)
            data = np.empty((vertex_count, 6), dtype=np.float32)
            np.take(obj.mesh.vertices, v_indices, axis=0, out=data[:, :3])

            # face normals repeated 3 times (broadcast assignment, no big temp)
            n_faces = int(faces.shape[0])
            data[:, 3:].reshape((n_faces, 3, 3))[:] = obj.mesh.face_normals[:, None, :]

            obj.vertex_count = vertex_count
            
            if obj.vbo_id is None:
                obj.vbo_id = glGenBuffers(1)
            
            glBindBuffer(GL_ARRAY_BUFFER, obj.vbo_id)
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

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
    
    def fit_view_to_selected_object(self):
        """선택된 객체에 카메라 초점 맞춤"""
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
        정치 확정: 현재 보이는 그대로 메쉬를 고정하고 모든 변환값을 0으로 리셋
        
        - 현재 화면에 보이는 위치 그대로 유지
        - 이동/회전/배율 값이 모두 0으로 리셋
        - 이후 0에서부터 세부 조정 가능
        """
        if not obj:
            return
        
        # 변환이 없으면 스킵
        has_transform = (
            not np.allclose(obj.translation, [0, 0, 0]) or 
            not np.allclose(obj.rotation, [0, 0, 0]) or 
            obj.scale != 1.0
        )
        if not has_transform:
            # 변환이 없어도 "현재 상태"를 고정 상태로 기록
            try:
                obj.fixed_translation = np.asarray(obj.translation, dtype=np.float64).copy()
                obj.fixed_rotation = np.asarray(obj.rotation, dtype=np.float64).copy()
                obj.fixed_scale = float(obj.scale)
                obj.fixed_state_valid = True
            except Exception:
                _log_ignored_exception()
            return
        
        # 1. 회전 행렬 계산
        rx, ry, rz = np.radians(obj.rotation)
        
        cos_x, sin_x = np.cos(rx), np.sin(rx)
        rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        
        cos_z, sin_z = np.cos(rz), np.sin(rz)
        rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        
        # OpenGL 렌더링(glRotate X->Y->Z)과 동일한 합성 회전
        rotation_matrix = rot_x @ rot_y @ rot_z
        
        # 2. 정점 변환 (S -> R -> T 순서, 렌더링과 동일)
        # 스케일
        vertices = obj.mesh.vertices * obj.scale
        
        # 회전
        vertices = (rotation_matrix @ vertices.T).T
        
        # 이동 (월드 좌표에 적용)
        vertices = vertices + obj.translation
        
        # 3. 데이터 업데이트
        obj.mesh.vertices = vertices.astype(np.float32)
        # 캐시 무효화 (vertices 변경)
        try:
            obj.mesh._bounds = None
            obj.mesh._centroid = None
            obj.mesh._surface_area = None
        except Exception:
            _log_ignored_exception()
        
        # 법선 재계산
        obj.mesh.compute_normals(compute_vertex_normals=False, force=True)
        obj._trimesh = None
        
        # 4. 모든 변환값 0으로 리셋 (이제 메쉬 정점 자체가 월드 좌표)
        obj.translation = np.array([0.0, 0.0, 0.0])
        obj.rotation = np.array([0.0, 0.0, 0.0])
        obj.scale = 1.0

        # 4.5 고정 상태 갱신 (실수로 움직여도 복귀 가능)
        try:
            obj.fixed_translation = obj.translation.copy()
            obj.fixed_rotation = obj.rotation.copy()
            obj.fixed_scale = float(obj.scale)
            obj.fixed_state_valid = True
        except Exception:
            _log_ignored_exception()
        
        # 5. VBO 업데이트
        self.update_vbo(obj)
        self.update()
        self.meshTransformChanged.emit()

    def restore_fixed_state(self, obj: SceneObject):
        """정치 확정 이후의 '고정 상태'로 변환값 복귀"""
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
        """현재 선택된 객체의 변환 상태를 스택에 저장"""
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
        """마지막 변환 취소 (Ctrl+Z)"""
        if not self.undo_stack:
            return
            
        state = self.undo_stack.pop()
        obj = state['obj']
        obj.translation = state['translation']
        obj.rotation = state['rotation']
        obj.scale = state['scale']
        
        self.update()
        self.meshTransformChanged.emit()
        self.status_info = "↩️ 변환 취소됨"

    def _begin_ctrl_drag(self, event: QMouseEvent, obj: SceneObject) -> bool:
        """Ctrl+드래그용 깊이/행렬 캐시를 준비합니다."""
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

            # 배경을 클릭한 경우 객체 중심 깊이로 대체
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
        """마우스 버튼 눌림"""
        try:
            self.last_mouse_pos = event.pos()
            self.mouse_button = event.button()
            modifiers = event.modifiers()
            obj_for_ctrl_drag = self.selected_obj
            self._ctrl_drag_active = False

            # Ctrl+우클릭 드래그는 모든 모드에서 "메쉬 이동" 우선.
            # (표면/단면 도구의 우클릭 확정과 충돌하지 않도록 press 시점에서 선점)
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

            # 1. 일반 클릭 (객체 선택 또는 피킹 모드 처리) - 좌클릭만 처리
            if event.button() == Qt.MouseButton.LeftButton:
                # 기즈모 선택 검사 (가장 우선순위) - ROI 모드에서는 숨김/비활성
                if self.picking_mode == 'none' and not getattr(self, "roi_enabled", False):
                    axis = self.hit_test_gizmo(event.pos().x(), event.pos().y())
                    if axis:
                        self.save_undo_state() # 변환 시작 전 상태 저장
                        self.active_gizmo_axis = axis

                        # 캐시 매트릭스 저장 (성능 최적화)
                        self.makeCurrent()
                        self._cached_viewport = glGetIntegerv(GL_VIEWPORT)
                        self._cached_modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
                        self._cached_projection = glGetDoublev(GL_PROJECTION_MATRIX)

                        angle = self._calculate_gizmo_angle(event.pos().x(), event.pos().y())
                        if angle is not None:
                            self.gizmo_drag_start = angle
                            self.update()
                            return

                # 피킹 모드 처리
                if self.picking_mode == 'curvature' and (modifiers & Qt.KeyboardModifier.ShiftModifier):
                    # Shift+클릭으로만 점 찍기
                    point = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                    if point is not None:
                        self.picked_points.append(point)
                        self.update()
                    return

                if self.picking_mode == "measure" and (modifiers & Qt.KeyboardModifier.ShiftModifier):
                    # Shift+클릭으로만 점 찍기
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
                        obj = self.selected_obj
                        if obj is None:
                            return
                        # 로컬 좌표로 변환하여 전달 (작업 도중 객체가 움직여도 점이 붙어있게 함)
                        local_pt = point - obj.translation
                        
                        # 1. 스냅 검사 (첫 번째 점과 가까우면 확정)
                        if len(self.floor_picks) >= 3:
                            first_pt = self.floor_picks[0]
                            dist = np.linalg.norm(local_pt - first_pt)
                            if dist < 0.15: # 스냅 거리 확대 (15cm)
                                self.floorAlignmentConfirmed.emit()
                                return
                                
                        self.floorPointPicked.emit(local_pt)
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
                    # 실제 점 추가는 mouseRelease에서 "클릭"으로 판정됐을 때만 수행
                    self._cut_line_left_press_pos = event.pos()
                    self._cut_line_left_dragged = False
                    return
                 
                elif self.picking_mode == 'crosshair':
                    point = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                    if point is None:
                        # 메쉬 픽이 실패해도(잔존 파손 등) 바닥 평면에서 십자선 이동 가능
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
                                self.update()
                                return
                        else:
                            self.active_roi_edge = str(handle)
                            self._roi_move_dragging = False
                            self._roi_move_last_xy = None
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
                            # 초기에는 최소 크기를 확보해(0.1cm) 이후 드래그로 확장
                            self.roi_bounds = [float(pt[0]), float(pt[0]) + 0.1, float(pt[1]), float(pt[1]) + 0.1]
                            self.schedule_roi_edges_update(0)
                            self.update()
                            return
                    # Otherwise: allow normal camera drag (left=rotate, right=pan)

            # 단면선 모드: 우클릭은 "확정"(click) 용도로 사용 (드래그 시에는 Pan 유지)
            if self.picking_mode == "cut_lines" and event.button() == Qt.MouseButton.RightButton:
                self._cut_line_right_press_pos = event.pos()
                self._cut_line_right_dragged = False
                return

            # 표면 지정(면적/Area): 우클릭은 "확정"(click) 용도로 사용 (드래그 시에는 Pan 유지)
            if self.picking_mode == "paint_surface_area" and event.button() == Qt.MouseButton.RightButton:
                self._surface_area_right_press_pos = event.pos()
                self._surface_area_right_dragged = False
                try:
                    self.surface_lasso_preview = event.pos()
                except Exception:
                    _log_ignored_exception()
                return

            # 표면 지정(경계/자석): 우클릭은 "확정"(click) 용도로 사용 (드래그 시에는 Pan 유지)
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

            # 3. 객체 조작 (Shift/Ctrl + 드래그)
            obj = self.selected_obj
            if (
                obj
                and (not getattr(self, "roi_enabled", False))
                and (modifiers & Qt.KeyboardModifier.ShiftModifier or modifiers & Qt.KeyboardModifier.ControlModifier)
            ):
                 self.save_undo_state() # 변환 시작 전 상태 저장
                 
                 # Ctrl+드래그(이동)를 위한 초기 깊이값 저장 (마우스가 가리키는 지점의 깊이)
                 if modifiers & Qt.KeyboardModifier.ControlModifier:
                     self._ctrl_drag_active = bool(self._begin_ctrl_drag(event, obj))
                 return
            
            # 4. 휠 클릭: 포커스 이동 (Focus move)
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
        """마우스 버튼 놓음"""

        # 단면선(2개): 좌클릭=점 추가(클릭으로만), 우클릭=현재 선 확정
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
                        # Prefer picking on the mesh surface (3D) and project to top XY.
                        picked = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                        if picked is None:
                            pt = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                            if pt is not None:
                                picked = np.array([pt[0], pt[1], 0.0], dtype=np.float64)

                        if picked is not None:
                            idx = int(getattr(self, "cut_line_active", 0))
                            if idx not in (0, 1):
                                idx = 0
                            line = self.cut_lines[idx]

                            # 기존 단면 결과는 편집 시 무효화
                            self.cut_section_profiles[idx] = []
                            self.cut_section_world[idx] = []

                            p = np.array([float(picked[0]), float(picked[1]), 0.0], dtype=np.float64)

                            # Polyline input (ㄱㄴ 모양 등): 좌클릭으로 점을 계속 추가하고,
                            # Enter/우클릭으로 "확정"합니다. (각 세그먼트는 Ortho로 수평/수직 제약)
                            try:
                                final = getattr(self, "_cut_line_final", [False, False])
                                if bool(final[idx]) and len(line) >= 2:
                                    # 이미 확정된 라인은 Backspace/Delete로 편집하거나 지우고 다시 시작.
                                    self.cut_line_drawing = False
                                    self.cut_line_preview = None
                                    self.update()
                                    return
                                final[idx] = False
                            except Exception:
                                _log_ignored_exception()

                            if len(line) == 0:
                                line.append(p)
                            else:
                                last = np.asarray(line[-1], dtype=np.float64)
                                p2 = self._cutline_constrain_ortho(last, p)
                                if float(np.linalg.norm(p2[:2] - last[:2])) <= 1e-6:
                                    self.cut_line_preview = p2
                                    self.cut_line_drawing = True
                                    self.update()
                                    return
                                line.append(p2)

                            self.cut_line_drawing = True
                            self.cut_line_preview = None
                            if len(line) >= 2:
                                self.schedule_cut_section_update(idx, delay_ms=150)

                            self.update()
                except Exception:
                    _log_ignored_exception()
                self._cut_line_left_press_pos = None
                self._cut_line_left_dragged = False

        # 표면 지정(찍기): 드래그가 아니면 릴리즈에서 1회 적용
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

        # 표면 지정(면적/Area): 좌클릭=점 추가(클릭일 때만), 우클릭=확정(클릭일 때만)
        if self.picking_mode == "paint_surface_area":
            if event.button() == Qt.MouseButton.LeftButton:
                try:
                    if (
                        getattr(self, "_surface_area_left_press_pos", None) is not None
                        and not bool(getattr(self, "_surface_area_left_dragged", False))
                    ):
                        thr = getattr(self, "_surface_lasso_thread", None)
                        if thr is not None and bool(getattr(thr, "isRunning", lambda: False)()):
                            self.status_info = "⏳ 둘러서 선택 계산 중…"
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
                                self.status_info = "⚠️ 메쉬 위를 클릭해 점을 찍어 주세요."
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

        # 표면 지정(경계/자석): 좌클릭=점 추가(클릭일 때만), 우클릭=확정(클릭일 때만)
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
                                self.status_info = "⏳ 경계 선택 계산 중…"
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
            # 드래그 스로틀로 인해 마지막 위치가 반영되지 않을 수 있어, 릴리즈 시 1회 확정 업데이트
            self._crosshair_last_update = 0.0
            self.schedule_crosshair_profile_update(0)

        if self.mouse_button == Qt.MouseButton.LeftButton and getattr(self, "roi_enabled", False):
            # ROI 드래그(핸들/사각형) 종료 시 최종 상태로 1회 확정 계산
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
        
        # 캐시 초기화
        self._cached_viewport = None
        self._cached_modelview = None
        self._cached_projection = None
        self._ctrl_drag_active = False
        
        self.update()
    
    def mouseMoveEvent(self, a0: QMouseEvent | None):
        if a0 is None:
            return
        event = a0
        """마우스 이동 (드래그)"""
        try:
            # 단면선 모드: 드래그 중에는 "클릭"으로 오인하지 않도록 플래그 처리
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

            # 표면 지정(찍기): 드래그는 카메라 회전으로 간주하고, 릴리즈에서만 "클릭" 처리
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

            # 표면 지정(면적/Area): 미리보기 + 드래그 판정
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

            # 표면 지정(경계/자석): 커서 프리뷰 + 클릭으로 점 추가(스냅)
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

            # 단면선(2개) 프리뷰: 마우스 트래킹(버튼 없이 이동)에서도 동작
            if self.picking_mode == 'cut_lines' and getattr(self, "cut_line_drawing", False):
                if self.mouse_button is None or self.mouse_button == Qt.MouseButton.LeftButton:
                    picked = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                    if picked is None:
                        pt = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                        if pt is not None:
                            picked = np.array([pt[0], pt[1], 0.0], dtype=np.float64)
                    if picked is not None:
                        p = np.array([float(picked[0]), float(picked[1]), 0.0], dtype=np.float64)
                        try:
                            idx = int(getattr(self, "cut_line_active", 0))
                            idx = idx if idx in (0, 1) else 0
                            line = self.cut_lines[idx]
                            if line:
                                last = np.asarray(line[-1], dtype=np.float64)
                                self.cut_line_preview = self._cutline_constrain_ortho(last, p)
                            else:
                                self.cut_line_preview = p
                        except Exception:
                            self.cut_line_preview = p
                        self.update()
                # 버튼 없이 이동 중이면 카메라 드래그 로직을 타지 않도록 조기 종료
                if self.mouse_button is None:
                    return

            if self.last_mouse_pos is None:
                self.last_mouse_pos = event.pos()
                return

            # 이전 위치 저장 및 현재 위치 갱신 (드래그 계산용)
            prev_pos = self.last_mouse_pos
            dx = event.pos().x() - prev_pos.x()
            dy = event.pos().y() - prev_pos.y()
            self.last_mouse_pos = event.pos()
            
            obj = self.selected_obj
            modifiers = event.modifiers()
            
            # 1. 기즈모 드래그 (좌클릭 + 기즈모 드래그 시작됨)
            if self.gizmo_drag_start is not None and self.active_gizmo_axis and obj and self.mouse_button == Qt.MouseButton.LeftButton:
                angle_info = self._calculate_gizmo_angle(event.pos().x(), event.pos().y())
                if angle_info is not None:
                    current_angle = angle_info
                    delta_angle = np.degrees(current_angle - self.gizmo_drag_start)
                    
                    # "자동차 핸들" 직관성: 마우스의 회전 방향을 메쉬 회전에 1:1 매칭
                    # 카메라 시선과 회전축의 방향성(도트곱)을 통해 visual CW/CCW를 결정
                    view_dir = self.camera.look_at - self.camera.position
                    view_dir /= np.linalg.norm(view_dir)
                    
                    axis_vec = np.zeros(3)
                    if self.active_gizmo_axis == 'X':
                        axis_vec[0] = 1.0
                    elif self.active_gizmo_axis == 'Y':
                        axis_vec[1] = 1.0
                    elif self.active_gizmo_axis == 'Z':
                        axis_vec[2] = 1.0
                    
                    # 시각적 반전 여부 결정 (핸들을 돌리는 방향과 메쉬가 도는 방향 일치)
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
                
            # 2. 기즈모 호버 하이라이트 (버튼 안 눌렸을 때만)
            if event.buttons() == Qt.MouseButton.NoButton:
                axis = self.hit_test_gizmo(event.pos().x(), event.pos().y())
                # 호버 시 하이라이트만 변경, active_gizmo_axis는 클릭 시에만 설정
                if axis != getattr(self, '_hover_axis', None):
                    self._hover_axis = axis
                    self.update()
                return
            
            # 3. 객체 직접 조작 (Ctrl+드래그 = 메쉬 이동, 마우스 커서를 정확히 따라감)
            if (
                (not getattr(self, "roi_enabled", False))
                and (
                    bool(modifiers & Qt.KeyboardModifier.ControlModifier)
                    or bool(getattr(self, "_ctrl_drag_active", False))
                )
                and obj
                and self._cached_viewport is not None
            ):
                # 마우스 프레스 시 캡처된 깊이와 매트릭스 재사용 (성능 향상)
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
                    
                    # 6개 좌표계 정렬 뷰에서는 2차원 이동 강제 (직관성 향상)
                    el = self.camera.elevation
                    az = self.camera.azimuth % 360
                    
                    if abs(el) > 85: # 상면(90) / 하면(-90)
                        delta_world[2] = 0
                    elif abs(el) < 5: # 정면, 후면, 좌, 우
                        # 정면(-90/270), 후면(90) -> Y축 고정
                        if abs(az - 90) < 5 or abs(az - 270) < 5:
                            delta_world[1] = 0
                        # 우측(0/360), 좌측(180) -> X축 고정
                        elif abs(az) < 5 or abs(az - 360) < 5 or abs(az - 180) < 5:
                            delta_world[0] = 0
                            
                    obj.translation += delta_world
                
                self.meshTransformChanged.emit()
                self.update()
                return
            
            elif (not getattr(self, "roi_enabled", False)) and (modifiers & Qt.KeyboardModifier.AltModifier) and obj:
                # 트랙볼 스타일 회전 - 방향 반전
                rot_speed = 0.5
                
                # 화면 수평 드래그 -> Z축 회전 (방향 반전)
                obj.rotation[2] += dx * rot_speed
                
                # 화면 수직 드래그 -> 카메라 방향 기준 피칭 (방향 반전)
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
                # 직관적 회전: 드래그 방향 = 회전 방향 (카메라 기준)
                rot_speed = 0.5
                
                # 좌우 드래그 -> 화면 Y축 기준 회전 (Z축 회전)
                obj.rotation[2] -= dx * rot_speed
                
                # 상하 드래그 -> 화면에서 앞뒤로 굴림 (카메라 방향 고려)
                az_rad = np.radians(self.camera.azimuth)
                obj.rotation[0] += dy * rot_speed * np.cos(az_rad)
                obj.rotation[1] += dy * rot_speed * np.sin(az_rad)
                
                self.meshTransformChanged.emit()
                self.update()
                return


                
            # 0. 브러시 피킹 처리
            if self.mouse_button == Qt.MouseButton.LeftButton and self.picking_mode == 'floor_brush':
                self._pick_brush_face(event.pos())
                self.update()
                return

            # 0.1 선택 브러시 처리 (SelectionPanel)
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

            # 0.2 표면 지정 브러시 처리 (outer/inner/migu)
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

            # 0.4 선형 단면(라인) 드래그 처리
            if self.picking_mode == 'line_section' and self.line_section_dragging and self.mouse_button == Qt.MouseButton.LeftButton:
                pt = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                if pt is not None and self.line_section_start is not None:
                    end = np.array([pt[0], pt[1], 0.0], dtype=float)

                    # Shift: CAD Ortho (수평/수직 고정)
                    if modifiers & Qt.KeyboardModifier.ShiftModifier:
                        start = np.array(self.line_section_start, dtype=float)
                        dx_l = float(end[0] - start[0])
                        dy_l = float(end[1] - start[1])
                        if abs(dx_l) >= abs(dy_l):
                            end[1] = start[1]
                        else:
                            end[0] = start[0]

                    self.line_section_end = end

                    # 연산 비용(슬라이싱) 절약을 위해 약간 스로틀링
                    now = time.monotonic()
                    if now - self._line_section_last_update > 0.08:
                        self._line_section_last_update = now
                        self.schedule_line_profile_update(150)
                    else:
                        self.update()
                return
            
            # 0.5 십자선 드래그 처리
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

            # 0.54 ROI 이동 드래그 (중앙 핸들)
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
                        self.schedule_roi_edges_update(120)
                        self.update()
                return

            # 0.55 ROI 사각형 드래그(캡쳐처럼 지정)
            if getattr(self, "roi_enabled", False) and getattr(self, "roi_rect_dragging", False) and self.mouse_button == Qt.MouseButton.LeftButton:
                pt = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                if pt is not None and self.roi_rect_start is not None:
                    x0, y0 = float(self.roi_rect_start[0]), float(self.roi_rect_start[1])
                    x1, y1 = float(pt[0]), float(pt[1])
                    min_x = min(x0, x1)
                    max_x = max(x0, x1)
                    min_y = min(y0, y1)
                    max_y = max(y0, y1)
                    # 최소 크기 확보 (0.1cm)
                    if max_x - min_x < 0.1:
                        max_x = min_x + 0.1
                    if max_y - min_y < 0.1:
                        max_y = min_y + 0.1
                    self.roi_bounds = [min_x, max_x, min_y, max_y]
                    self._roi_bounds_changed = True
                    self.schedule_roi_edges_update(120)
                    self.update()
                return
             
            # 0.6 ROI 핸들 드래그 처리
            if self.roi_enabled and self.active_roi_edge and self.active_roi_edge != "move" and self.mouse_button == Qt.MouseButton.LeftButton:
                # XY 평면으로 투영하여 마우스 월드 좌표 획득 (z=0으로 가정)
                # pick_point_on_mesh는 메쉬 표면을 찍으므로, 여기서는 단순히 레이-평면 교차 사용
                # 도움을 위해 pick_point_on_mesh 활용 가능하나 바닥일 때 고려
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
                     
                    self._roi_bounds_changed = True
                    self.schedule_roi_edges_update(120)
                    self.update()
                return

            # 4. 일반 카메라 조작 (드래그)
            if self.mouse_button == Qt.MouseButton.LeftButton:
                # 좌클릭: 회전 (기본)
                self.camera.rotate(dx, dy)
                self.update()
            elif self.mouse_button == Qt.MouseButton.RightButton:
                # 우클릭: 이동 (Pan) - 언제나 가능하게 하여 "자유로운" 느낌 부여
                self.camera.pan(dx, dy)
                self.update()
            elif self.mouse_button == Qt.MouseButton.MiddleButton:
                # 휠 클릭 드래그: 회전 (좌클릭이 피킹 등으로 막혔을 때 대안)
                self.camera.rotate(dx, dy)
                self.update()
            
        except Exception:
            _log_ignored_exception("Mouse move error", level=logging.WARNING)
    
    def _calculate_gizmo_angle(self, screen_x, screen_y):
        """기즈모 중심 기준 2D 화면 공간에서의 각도 계산 (가장 직관적인 원형 드래그 방식)"""
        obj = self.selected_obj
        if not obj or not self.active_gizmo_axis:
            return None
        
        try:
            # 캐시된 매트릭스가 있으면 사용 (성능 최적화)
            if self._cached_viewport is not None:
                viewport = self._cached_viewport
                modelview = self._cached_modelview
                projection = self._cached_projection
            else:
                self.makeCurrent()
                viewport = glGetIntegerv(GL_VIEWPORT)
                modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
                projection = glGetDoublev(GL_PROJECTION_MATRIX)
            
            # 기즈모 중심(오브젝트 위치)을 화면으로 투영
            obj_pos = obj.translation
            win_pos = gluProject(obj_pos[0], obj_pos[1], obj_pos[2], modelview, projection, viewport)
            if not win_pos:
                return None

            center_x, center_y = self._gl_window_to_qt_xy(float(win_pos[0]), float(win_pos[1]), viewport=viewport)

            # 중심점에서 마우스 포인터까지의 2D 각도 (atan2)
            # 화면 좌표계는 Y가 아래로 증가하므로 부호 주의
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
        """마우스 휠: 기본 줌, Ctrl+휠은 슬라이스 스캔."""
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
        self.camera.zoom(delta)
        self.update()

    def keyPressEvent(self, a0: QKeyEvent | None):
        if a0 is None:
            return
        event = a0
        """키보드 입력"""
        self.keys_pressed.add(event.key())
        if event.key() in (Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D, Qt.Key.Key_Q, Qt.Key.Key_E):
            if not self.move_timer.isActive():
                self.move_timer.start()

        # 0. 슬라이스 단축키 (표면 분리 작업 중 빠른 단면 스캔/촬영)
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
                    self.status_info = f"단면 스캔 스텝 {sign * step_cm:+.2f}cm (, / .)"
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
                    self.status_info = f"📸 단면 촬영 요청 (Z={z_now:.2f}cm, C)"
                    self.update()
                except Exception:
                    _log_ignored_exception()
                return

        # 0. 단면선(2개) 도구 단축키
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
                    self.cut_line_preview = None
                    idx = int(getattr(self, "cut_line_active", 0))
                    idx = idx if idx in (0, 1) else 0
                    line = self.cut_lines[idx]
                    final = getattr(self, "_cut_line_final", [False, False])
                    self.cut_line_drawing = bool(line) and not bool(final[idx])
                    self.update()
                except Exception:
                    _log_ignored_exception()
                return
            if key == Qt.Key.Key_Escape:
                # 도구는 유지하고, 현재 프리뷰/드로잉만 취소
                self.cut_line_drawing = False
                self.cut_line_preview = None
                self.update()
                return
        
        # 0.1 둘러서 지정(영역) 도구 단축키
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
                self.status_info = "⭕ 작업 취소됨"
                self.update()
                return

        # 0.2 경계(면적+자석) 올가미 도구 단축키
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
                self.status_info = "🧲 작업 취소됨"
                self.update()
                return

        # 1. Enter/Return 키: 바닥 정렬 확정
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.picking_mode in ('floor_3point', 'floor_face', 'floor_brush'):
                self.floorAlignmentConfirmed.emit()
                return

        # 2. ESC: 작업 취소
        if event.key() == Qt.Key.Key_Escape:
            if self.picking_mode != 'none':
                self.picking_mode = 'none'
                self.floor_picks = []
                self._surface_paint_left_press_pos = None
                self._surface_paint_left_dragged = False
                self.status_info = "⭕ 작업 취소됨"
                self.update()
                return
        
        # 3. Ctrl+Z: Undo
        if event.key() == Qt.Key.Key_Z and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.undo()
            return

        # 기즈모나 카메라 뷰 관련 키
        key = event.key()
        if key == Qt.Key.Key_R:
            self.camera.reset()
            self.update()
        elif key == Qt.Key.Key_F:
            self.fit_view_to_selected_object()
        else:
            key_dec = getattr(Qt.Key, "Key_BracketLeft", -1)
            key_inc = getattr(Qt.Key, "Key_BracketRight", -1)
            if key in (key_dec, key_inc):
                # 표면 지정 도구에서는 [ ] 키로 브러시/찍기 크기를 조절합니다.
                if self.picking_mode in {"paint_surface_brush", "paint_surface_face"}:
                    attr = (
                        "_surface_brush_radius_px"
                        if self.picking_mode == "paint_surface_brush"
                        else "_surface_click_radius_px"
                    )
                    label = "브러시" if self.picking_mode == "paint_surface_brush" else "찍기"
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
                    self.status_info = f"🖌️ 표면 {label} 크기: {new:.0f}px ([ / ] 조절)"
                    self.update()
                    return

                # 경계(면적+자석) 도구에서는 [ ] 키로 스냅 반경을 조절합니다.
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
                    self.status_info = f"🧲 자석 반경: {new:.0f}px ([ / ] 조절)"
                    self.update()
                    return

                # 기본: [ ] 키로 기즈모 크기 조절
                if key == key_dec:
                    self.gizmo_radius_factor = max(1.01, float(self.gizmo_radius_factor) * 0.9)
                    self.update_gizmo_size()
                    self.status_info = f"? 기즈모 크기: x{self.gizmo_radius_factor:.2f}"
                    self.update()
                elif key == key_inc:
                    self.gizmo_radius_factor = min(3.0, float(self.gizmo_radius_factor) / 0.9)
                    self.update_gizmo_size()
                    self.status_info = f"? 기즈모 크기: x{self.gizmo_radius_factor:.2f}"
                    self.update()
              
        super().keyPressEvent(event)

    def keyReleaseEvent(self, a0: QKeyEvent | None):
        if a0 is None:
            return
        event = a0
        """키 뗌 처리"""
        if event.key() in self.keys_pressed:
            self.keys_pressed.remove(event.key())
        if not (self.keys_pressed & {Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D, Qt.Key.Key_Q, Qt.Key.Key_E}):
            if self.move_timer.isActive():
                self.move_timer.stop()
        super().keyReleaseEvent(event)

    def process_keyboard_navigation(self):
        """WASD 연속 이동 처리"""
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
        """ROI 핸들(화살표/중앙) 클릭 검사 (screen-space 우선)."""
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

            handles = {
                "bottom": (mid_x, float(y1), z),
                "top": (mid_x, float(y2), z),
                "left": (float(x1), mid_y, z),
                "right": (float(x2), mid_y, z),
                "move": (mid_x, mid_y, z),
            }

            try:
                thr = int(getattr(self, "_roi_handle_hit_px", 24) or 24)
            except Exception:
                thr = 24
            thr = max(8, min(thr, 120))
            thr2 = float(thr * thr)

            best = None
            best_d2 = float("inf")
            for edge, (wx, wy, wz) in handles.items():
                try:
                    win = gluProject(float(wx), float(wy), float(wz), modelview, projection, viewport)
                    if not win:
                        continue
                    qx, qy = self._gl_window_to_qt_xy(float(win[0]), float(win[1]), viewport=viewport)
                    dx = float(pos.x()) - float(qx)
                    dy = float(pos.y()) - float(qy)
                    d2 = float(dx * dx + dy * dy)
                    if d2 <= thr2 and d2 < best_d2:
                        best = str(edge)
                        best_d2 = d2
                except Exception:
                    continue

            if best is not None:
                return best
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

            threshold = float(getattr(self.camera, "distance", 50.0) or 50.0) * 0.05
            if np.hypot(wx - mid_x, wy - float(y1)) < threshold:
                return "bottom"
            if np.hypot(wx - mid_x, wy - float(y2)) < threshold:
                return "top"
            if np.hypot(wx - float(x1), wy - mid_y) < threshold:
                return "left"
            if np.hypot(wx - float(x2), wy - mid_y) < threshold:
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
        """고해상도 오프스크린 렌더링

        `orthographic=True`를 사용하면 정사영(glOrtho)으로 캡처하여 1:1 스케일 도면에 유리합니다.
        """
        self.makeCurrent()
        prev_viewport = None
        try:
            prev_viewport = glGetIntegerv(GL_VIEWPORT)
        except Exception:
            prev_viewport = None

        # 1. FBO 생성
        fbo = QOpenGLFramebufferObject(width, height, QOpenGLFramebufferObject.Attachment.Depth)
        fbo.bind()
        
        # 2. 렌더링 설정 (오프스크린용)
        glViewport(0, 0, width, height)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        aspect = width / height
        if not orthographic:
            gluPerspective(45.0, aspect, 0.1, 1000000.0)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        
        # 3. 그리기 (UI 제외하고 깨끗하게)
        glClearColor(1.0, 1.0, 1.0, 1.0) # 화이트 배경
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
        
        # 광원
        glEnable(GL_LIGHTING)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.4, 0.4, 0.4, 1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.6, 0.6, 0.6, 1.0])
        
        # 메쉬만 렌더링 (그리드나 HUD 제외)
        sel = int(self.selected_index) if self.selected_index is not None else -1
        for i, obj in enumerate(self.objects):
            if not obj.visible:
                continue
            if only_selected and sel >= 0 and i != sel:
                continue
            self.draw_scene_object(obj, is_selected=(i == sel))
        
        # 4. 행렬 캡처 (SVG 투영 정렬용)
        mv = glGetDoublev(GL_MODELVIEW_MATRIX)
        proj = glGetDoublev(GL_PROJECTION_MATRIX)
        vp = glGetIntegerv(GL_VIEWPORT)
        
        glFlush()
        qimage = fbo.toImage()
        
        # 5. 복구
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        fbo.release()

        # 원래 뷰포트 복구
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
        """화면 좌표에서 월드 레이(origin, dir) 계산"""
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
        """화면 좌표에서 Z=z 평면과의 교점(월드 좌표) 계산"""
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
        화면 좌표를 메쉬 표면의 3D 좌표로 변환합니다.

        Returns:
            (pt_world, depth_value, gl_x, gl_y, viewport, modelview, projection) 또는 None
        """
        if not self.objects:
            return None
        obj = self.selected_obj
        if not obj:
            return None

        # OpenGL 뷰포트, 투영, 모델뷰 행렬 가져오기
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

        # 깊이 버퍼에서 깊이 값 읽기
        depth = cast(Any, glReadPixels(int(gl_x), int(gl_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT))
        try:
            depth_value = float(depth[0][0])
        except Exception:
            depth_value = float("nan")

        # 배경을 클릭한 경우: 근처 픽셀 depth를 탐색해서 피킹을 보정
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

        # 화면 좌표를 월드 좌표로 변환
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
        """화면 좌표를 메쉬 표면의 3D 좌표로 변환"""
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
        """같은 depth plane에서 px 거리 -> world 거리로 환산(근사)."""
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
        """브러시로 면 선택"""
        point = self.pick_point_on_mesh(pos.x(), pos.y())
        if point is not None:
            res = self.pick_face_at_point(point, return_index=True)
            if res:
                idx, v = res
                self.brush_selected_faces.add(idx)

    def _pick_selection_brush_face(self, pos):
        """SelectionPanel용 브러시 선택 (obj.selected_faces에 반영)"""
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
        """표면 지정(찍은 점) 표시를 지웁니다."""
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
        """둘러서 지정(영역) 올가미(다각형) 오버레이를 초기화합니다."""
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
        """경계(면적+자석) 올가미 상태를 초기화합니다."""
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
        """경계(면적+자석) 올가미 도구 시작: 캐시 준비 + 기존 선 초기화."""
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
            self.status_info = "⚠️ 메쉬 위를 클릭해 점을 찍어 주세요."
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
            return "🧲", "경계(면적+자석)"
        return "⭕", "둘러서 지정"

    def _finish_surface_lasso(self, modifiers, *, seed_pos=None) -> None:
        """현재 올가미(다각형) 영역에 포함되는 '보이는' 면을 한 번에 지정합니다."""
        obj = self.selected_obj
        if obj is None or getattr(obj, "mesh", None) is None:
            self.status_info = "⚠️ 먼저 메쉬를 선택해 주세요."
            self.clear_surface_lasso()
            self.update()
            return

        pts_world = list(getattr(self, "surface_lasso_points", None) or [])
        if len(pts_world) < 3:
            icon, lbl = self._surface_lasso_tool_strings(
                "boundary" if str(getattr(self, "picking_mode", "none")) == "paint_surface_magnetic" else "area"
            )
            self.status_info = f"{icon} {lbl}: 점을 3개 이상 찍어주세요. (우클릭/Enter=확정)"
            self.update()
            return

        thr = getattr(self, "_surface_lasso_thread", None)
        try:
            if thr is not None and thr.isRunning():
                _icon, lbl = self._surface_lasso_tool_strings()
                self.status_info = f"⏳ {lbl} 선택 계산 중…"
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
            self.status_info = f"{icon} {lbl}: 점 입력이 올바르지 않습니다."
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
            self.status_info = f"{icon} {lbl}: 점 입력이 올바르지 않습니다."
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

        self.status_info = f"{icon} {lbl}: 보이는 면 계산 중…"
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
            self.status_info = f"⚠️ {lbl} 계산 시작 실패: {e}"
            self.update()

    def _on_surface_lasso_failed(self, msg: str) -> None:
        if self.sender() is not getattr(self, "_surface_lasso_thread", None):
            return
        self._surface_lasso_thread = None
        _icon, lbl = self._surface_lasso_tool_strings()
        self.status_info = f"⚠️ {lbl} 실패: {msg}"
        self.update()

    def _on_surface_lasso_computed(self, result: object) -> None:
        if self.sender() is not getattr(self, "_surface_lasso_thread", None):
            return
        self._surface_lasso_thread = None

        obj = self.selected_obj
        if obj is None or getattr(obj, "mesh", None) is None:
            _icon, lbl = self._surface_lasso_tool_strings()
            self.status_info = f"⚠️ {lbl}: 선택 대상이 없습니다."
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
            self.status_info = f"{icon} {lbl}: 선택된 면이 없습니다."
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
        target_lbl = {"outer": "외면", "inner": "내면", "migu": "미구"}.get(target, target)
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
                wand_info = f" / 완드 {wand_n:,}/{wand_c:,}"
        except Exception:
            wand_info = ""

        comp_info = ""
        try:
            comp_n = int(stats.get("component_selected", 0) or 0)
            comp_c = int(stats.get("component_candidates", 0) or 0)
            if comp_n and comp_c and comp_c > comp_n:
                comp_info = f" / 연결 {comp_n:,}/{comp_c:,}"
        except Exception:
            comp_info = ""

        trunc_info = ""
        try:
            if bool(stats.get("truncated", False)):
                ms = int(stats.get("max_selected_faces", 0) or 0)
                trunc_info = f" / 최대 {ms:,} 제한" if ms > 0 else " / 최대 제한"
        except Exception:
            trunc_info = ""

        op = "제거" if remove else "추가"
        icon, lbl = self._surface_lasso_tool_strings()
        msg = f"{icon} {lbl} [{target_lbl}]: {op} {selected_n:,} faces{comp_info}{wand_info}{trunc_info}"
        if cand_n:
            msg += f" (후보 {cand_n:,})"
        self.status_info = msg

        # Keep tool active, but clear the polygon for the next stroke.
        self.clear_surface_lasso()
        self.update()

    def _finish_surface_magnetic_lasso(self, modifiers, *, seed_pos=None) -> None:
        """현재 '경계(면적+자석) 올가미' 폴리곤 영역에 포함되는 '보이는' 면을 한 번에 지정합니다."""
        obj = self.selected_obj
        if obj is None or getattr(obj, "mesh", None) is None:
            self.status_info = "⚠️ 먼저 메쉬를 선택해 주세요."
            self.clear_surface_magnetic_lasso(clear_cache=False)
            self.update()
            return

        pts = list(getattr(self, "surface_magnetic_points", None) or [])
        if len(pts) < 3:
            self.status_info = "🧲 경계(면적+자석): 영역을 3점 이상 찍어주세요. (우클릭/Enter=확정)"
            self.update()
            return

        thr = getattr(self, "_surface_magnetic_thread", None)
        try:
            if thr is not None and thr.isRunning():
                self.status_info = "⏳ 경계(면적+자석) 선택 계산 중…"
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
            self.status_info = "🧲 경계(면적+자석): 카메라 상태를 읽을 수 없습니다."
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
            self.status_info = "🧲 경계(면적+자석): 폴리곤이 올바르지 않습니다."
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
            self.status_info = "🧲 경계(면적+자석): 폴리곤이 올바르지 않습니다."
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

        self.status_info = "🧲 경계(면적+자석): 보이는 면 계산 중…"
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
            self.status_info = f"⚠️ 경계(면적+자석) 계산 시작 실패: {e}"
            self.update()

    def _on_surface_magnetic_failed(self, msg: str) -> None:
        if self.sender() is not getattr(self, "_surface_magnetic_thread", None):
            return
        self._surface_magnetic_thread = None
        self.status_info = f"⚠️ 경계(면적+자석) 실패: {msg}"
        self.update()

    def _on_surface_magnetic_computed(self, result: object) -> None:
        if self.sender() is not getattr(self, "_surface_magnetic_thread", None):
            return
        self._surface_magnetic_thread = None

        obj = self.selected_obj
        if obj is None or getattr(obj, "mesh", None) is None:
            self.status_info = "⚠️ 경계(면적+자석): 선택 대상이 없습니다."
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
            self.status_info = "🧲 경계(면적+자석): 선택된 면이 없습니다."
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
        target_lbl = {"outer": "외면", "inner": "내면", "migu": "미구"}.get(target, target)
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
                wand_info = f" / 완드 {wand_n:,}/{wand_c:,}"
        except Exception:
            wand_info = ""

        comp_info = ""
        try:
            comp_n = int(stats.get("component_selected", 0) or 0)
            comp_c = int(stats.get("component_candidates", 0) or 0)
            if comp_n and comp_c and comp_c > comp_n:
                comp_info = f" / 연결 {comp_n:,}/{comp_c:,}"
        except Exception:
            comp_info = ""

        trunc_info = ""
        try:
            if bool(stats.get("truncated", False)):
                ms = int(stats.get("max_selected_faces", 0) or 0)
                trunc_info = f" / 최대 {ms:,} 제한" if ms > 0 else " / 최대 제한"
        except Exception:
            trunc_info = ""

        op = "제거" if remove else "추가"
        msg = f"🧲 경계(면적+자석) [{target_lbl}]: {op} {selected_n:,} faces{comp_info}{wand_info}{trunc_info}"
        if cand_n:
            msg += f" (후보 {cand_n:,})"
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
        """특정 3D 좌표가 포함된 삼각형 면의 정점 3개를 반환"""
        obj = self.selected_obj
        if not obj or obj.mesh is None:
            return None
        
        # 메쉬 로컬 좌표로 변환 (T/R/S 역변환)
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
        """찍은 점들을 작은 구로 시각화 (곡률/치수 측정)"""
        if not self.picked_points and not getattr(self, "measure_picked_points", None):
            return
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)  # 항상 앞에 보이게
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

                # 작은 구 (편많체로 근사) - 크기 0.08cm
                size = 0.08
                glBegin(GL_TRIANGLE_FAN)
                glVertex3f(0, 0, size)  # 상단
                for j in range(9):
                    angle = 2.0 * np.pi * j / 8
                    glVertex3f(size * np.cos(angle), size * np.sin(angle), 0)
                glEnd()

                glBegin(GL_TRIANGLE_FAN)
                glVertex3f(0, 0, -size)  # 하단
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
        """표면 지정(외/내/미구) 중 찍은 점 표시"""
        pts = getattr(self, "surface_paint_points", None)
        if not pts:
            return

        try:
            glDisable(GL_LIGHTING)
            glDisable(GL_DEPTH_TEST)  # 항상 보이게
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
        """둘러서 지정(영역) 도구의 화면 올가미(다각형) 오버레이"""
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
        """경계(면적+자석) 도구의 화면 올가미(폴리곤) 오버레이 (월드 포인트 기반)."""
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
        """피팅된 원호 시각화 (선택된 객체에 부착됨)"""
        obj = self.selected_obj
        if not obj or not obj.fitted_arcs:
            # 임시 원호도 그리기 (아직 객체에 부착 안 된 경우)
            if self.fitted_arc is not None:
                self._draw_single_arc(self.fitted_arc, None)
            return
        
        # 객체에 부착된 모든 원호 그리기
        for arc in obj.fitted_arcs:
            self._draw_single_arc(arc, obj)
    
    def _draw_single_arc(self, arc, obj):
        """단일 원호 그리기 (이제 항상 월드 좌표 기준)"""
        from src.core.curvature_fitter import CurvatureFitter
        
        glDisable(GL_LIGHTING)
        glColor3f(0.9, 0.2, 0.9)  # 마젠타
        glLineWidth(3.0)
        
        # 원호 점들 생성
        fitter = CurvatureFitter()
        arc_points = fitter.generate_arc_points(arc, 64)
        
        # 원 그리기
        glBegin(GL_LINE_LOOP)
        for point in arc_points:
            glVertex3fv(point)
        glEnd()
        
        # 중심에서 원주까지 선 (반지름 표시)
        glColor3f(1.0, 1.0, 0.0)  # 노란색
        glBegin(GL_LINES)
        glVertex3fv(arc.center)
        glVertex3fv(arc_points[0])
        glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def draw_floor_picks(self):
        """바닥면 지정 점 시각화 (점 + 연결선)"""
        if not self.floor_picks:
            return
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)  # 메쉬 앞에 표시
        
        # 점 그리기 (파란색 원형 마커)
        glColor3f(0.2, 0.4, 1.0)  # 파란색
        glPointSize(12.0)
        glBegin(GL_POINTS)
        for point in self.floor_picks:
            glVertex3fv(point)
        glEnd()
        
        # 점 사이 연결선 (노란색)
        if len(self.floor_picks) >= 2:
            glColor3f(1.0, 0.9, 0.2)  # 노란색
            glLineWidth(3.0)
            glBegin(GL_LINE_STRIP)
            for point in self.floor_picks:
                glVertex3fv(point)
            glEnd()
            
            # 항상 시작점-끝점 연결하여 영역 표시
            glBegin(GL_LINES)
            glVertex3fv(self.floor_picks[-1])
            glVertex3fv(self.floor_picks[0])
            glEnd()
            
            glBegin(GL_LINE_STRIP)
            for point in self.floor_picks:
                glVertex3fv(point)
            glEnd()
            
            # 반투명 영역 면 표시 (충분한 점이 모이면)
            if len(self.floor_picks) >= 3:
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glColor4f(0.2, 0.8, 0.2, 0.3)  # 초록색 반투명
                # 다각형 면 (Triangle Fan)
                glBegin(GL_TRIANGLE_FAN)
                for point in self.floor_picks:
                    glVertex3fv(point)
                glEnd()
        
        # 점 번호 표시용 작은 마커 (1, 2, 3)
        glColor3f(1.0, 1.0, 1.0)
        marker_size = 0.3
        for i, point in enumerate(self.floor_picks):
            glPushMatrix()
            glTranslatef(point[0], point[1], point[2] + 0.5)
            # 숫자 대신 크기로 구분 (1=작은원, 2=중간원, 3=큰원)
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
        """곡률 측정용 점들 초기화"""
        self.picked_points = []
        self.fitted_arc = None
        self.update()

    def clear_measure_picks(self) -> None:
        """치수 측정용 점들 초기화"""
        try:
            self.measure_picked_points = []
        except Exception:
            self.measure_picked_points = []
        self.update()


# 테스트용 스탠드얼론 실행
if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    
    viewport = Viewport3D()
    viewport.setWindowTitle("3D Viewport Test")
    viewport.resize(800, 600)
    viewport.show()
    
    sys.exit(app.exec())
