"""
3D Viewport Widget with OpenGL
Copyright (C) 2026 balguljang2 (lzpxilfe)
Licensed under the GNU General Public License v2.0 (GPL2)
"""

import time
import numpy as np
from typing import Optional, Tuple

from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize, QThread
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QPainter, QColor, QImage
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLFramebufferObject

from OpenGL.GL import *
from OpenGL.GLU import *
import ctypes

from ..core.mesh_loader import MeshData
from ..core.mesh_slicer import MeshSlicer


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

            inv_rot = R.from_euler('xyz', self._rotation, degrees=True).inv().as_matrix()
            inv_scale = 1.0 / self._scale if self._scale != 0 else 1.0

            def world_to_local(pt_world: np.ndarray) -> np.ndarray:
                return inv_scale * (inv_rot @ (pt_world - self._translation))

            rot_mat = R.from_euler('xyz', self._rotation, degrees=True).as_matrix()

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

            contours_x = slicer.slice_with_plane(l_orig_x, l_norm_x)
            contours_y = slicer.slice_with_plane(l_orig_y, l_norm_y)

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

            inv_rot = R.from_euler('xyz', self._rotation, degrees=True).inv().as_matrix()
            inv_scale = 1.0 / self._scale if self._scale != 0 else 1.0
            local_origin = inv_scale * inv_rot @ (world_origin - self._translation)
            local_normal = inv_rot @ world_normal

            slicer = MeshSlicer(self._mesh.to_trimesh())
            contours_local = slicer.slice_with_plane(local_origin, local_normal)

            rot_mat = R.from_euler('xyz', self._rotation, degrees=True).as_matrix()
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
        polyline_world: list,
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

            inv_rot = R.from_euler('xyz', self._rotation, degrees=True).inv().as_matrix()
            inv_scale = 1.0 / self._scale if self._scale != 0 else 1.0
            rot_mat = R.from_euler('xyz', self._rotation, degrees=True).as_matrix()

            slicer = MeshSlicer(self._mesh.to_trimesh())

            profile_pts: list[tuple[float, float]] = []
            s_offset = 0.0

            for si in range(len(pts) - 1):
                p0 = pts[si]
                p1 = pts[si + 1]
                d = p1 - p0
                d[2] = 0.0
                seg_len = float(np.linalg.norm(d))
                if seg_len < 1e-6:
                    continue

                d_unit = d / seg_len
                world_normal = np.array([d_unit[1], -d_unit[0], 0.0], dtype=np.float64)
                world_origin = p0

                local_origin = inv_scale * inv_rot @ (world_origin - self._translation)
                local_normal = inv_rot @ world_normal

                contours_local = slicer.slice_with_plane(local_origin, local_normal)
                if not contours_local:
                    s_offset += seg_len
                    continue

                world_contours = []
                for cnt in contours_local:
                    if cnt is None or len(cnt) < 2:
                        continue
                    world_contours.append((rot_mat @ (cnt * self._scale).T).T + self._translation)

                best = None
                best_span = 0.0
                margin = max(seg_len * 0.02, 0.2)
                t_min = -margin
                t_max = seg_len + margin

                for cnt in world_contours:
                    t = (cnt - p0) @ d_unit
                    mask = (t >= t_min) & (t <= t_max)
                    if int(mask.sum()) < 2:
                        continue
                    t_f = t[mask]
                    span = float(t_f.max() - t_f.min())
                    if span <= best_span:
                        continue
                    best_span = span
                    best = (t_f.copy(), cnt[mask, 2].copy())

                if best is None:
                    s_offset += seg_len
                    continue

                t_f, z_f = best
                order = np.argsort(t_f)
                t_sorted = t_f[order]
                z_sorted = z_f[order]

                # clamp into segment [0, seg_len]
                t_sorted = np.clip(t_sorted, 0.0, seg_len)

                # downsample if too dense
                if len(t_sorted) > 3000:
                    step = int(len(t_sorted) // 3000) + 1
                    t_sorted = t_sorted[::step]
                    z_sorted = z_sorted[::step]

                # append with cumulative distance
                for t_val, z_val in zip(t_sorted.tolist(), z_sorted.tolist()):
                    s_val = s_offset + float(t_val)
                    if profile_pts and abs(profile_pts[-1][0] - s_val) < 1e-6:
                        continue
                    profile_pts.append((s_val, float(z_val)))

                s_offset += seg_len

            self.computed.emit({"index": self._index, "profile": profile_pts})
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
        roi_bounds: list,
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
            inv_rot = R.from_euler('xyz', self._rotation, degrees=True).inv().as_matrix()
            inv_scale = 1.0 / self._scale if self._scale != 0 else 1.0
            rot_mat = R.from_euler('xyz', self._rotation, degrees=True).as_matrix()

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
            pass


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
        
        # 렌더링 리소스
        self.vbo_id = None
        self.vertex_count = 0
        self.selected_faces = set()
        self._trimesh = None # Lazy-loaded trimesh object
        
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
            except:
                pass


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
        
        # 렌더링 설정
        self.grid_size = 500.0  # cm (더 크게 확장)
        self.grid_spacing = 1.0  # cm (1.0 = 1cm)
        self.bg_color = [0.96, 0.96, 0.94, 1.0] # #F5F5F0 (Cream/Beige)
        
        # 기즈모 설정
        self.show_gizmo = True
        self.active_gizmo_axis = None
        self.gizmo_size = 10.0
        self.gizmo_drag_start = None
        
        # 곡률 측정 모드
        self.curvature_pick_mode = False
        self.picked_points = []
        self.fitted_arc = None
        
        # 상태 표시용 텍스트
        self.status_info = ""
        self.flat_shading = False # Flat shading 모드 (명암 없이 밝게 보기)
        
        # 피킹 모드 ('none', 'curvature', 'floor_3point', 'floor_face', 'floor_brush')
        self.picking_mode = 'none'
        self.brush_selected_faces = set() # 브러시로 선택된 면 인덱스
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
        self.roi_cut_edges = {"x1": [], "x2": [], "y1": [], "y2": []}  # ROI 잘림 경계선(월드 좌표)
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
        self.cut_section_profiles = [[], []]  # 각 선의 (s,z) 프로파일 [(dist, z), ...]
        self.cut_section_world = [[], []]     # 바닥에 배치된 단면 폴리라인(월드 좌표)
        self._cut_section_pending_index = None
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
        self._hover_axis = None
        
        # 키보드 조작 타이머 (WASD 연속 이동용)
        self.keys_pressed = set()
        self.move_timer = QTimer(self)
        self.move_timer.timeout.connect(self.process_keyboard_navigation)
        self.move_timer.setInterval(16) # ~60fps (필요 시만 start/stop)
        
        # UI 설정
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

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
    
    def resizeGL(self, width: int, height: int):
        """뷰포트 크기 변경"""
        glViewport(0, 0, width, height)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = width / height if height > 0 else 1.0
        # Far plane을 대폭 늘려 광활한 공간 확보 (기존 2000 -> 1,000,000)
        gluPerspective(45.0, aspect, 0.1, 1000000.0)
        
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """그리기"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
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
        
        # 1. 격자 및 축
        self.draw_ground_plane()  # 반투명 바닥
        self.draw_grid()
        self.draw_axes()
        
        # 2. 모든 메쉬 객체 렌더링
        sel = int(self.selected_index) if self.selected_index is not None else -1
        for i, obj in enumerate(self.objects):
            if not obj.visible:
                continue

            if self.roi_enabled and i == sel:
                try:
                    glEnable(GL_CLIP_PLANE1)
                    glEnable(GL_CLIP_PLANE2)
                    glEnable(GL_CLIP_PLANE3)
                    glEnable(GL_CLIP_PLANE4)
                except Exception:
                    pass

                self.draw_scene_object(obj, is_selected=True)

                try:
                    glDisable(GL_CLIP_PLANE1)
                    glDisable(GL_CLIP_PLANE2)
                    glDisable(GL_CLIP_PLANE3)
                    glDisable(GL_CLIP_PLANE4)
                except Exception:
                    pass
                continue

            self.draw_scene_object(obj, is_selected=(i == sel))
            
        # 3. 곡률 피팅 요소
        self.draw_picked_points()
        self.draw_fitted_arc()
        
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
        if self.cut_lines_enabled or any(len(l) > 0 for l in getattr(self, "cut_lines", [])):
            self.draw_cut_lines()

        # 3.7.5 선형 단면 (Top-view cut line)
        if self.line_section_enabled:
            self.draw_line_section()
             
        # 3.8 2D ROI 크로핑 영역
        if self.roi_enabled:
            self.draw_roi_cut_edges()
            self.draw_roi_box()
        
        # 4. 회전 기즈모 (선택된 객체에만, 피킹 모드 아닐 때만)
        if self.selected_obj and self.picking_mode == 'none':
            self.draw_rotation_gizmo(self.selected_obj)
            # 메쉬 치수/중심점 오버레이
            self.draw_mesh_dimensions(self.selected_obj)
            
        # 5. UI 오버레이 (HUD)
        self.draw_orientation_hud()

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
            pass
    
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
        if not self.selected_obj or self.selected_obj.mesh is None:
            self.slice_contours = []
            return
            
        slicer = MeshSlicer(self.selected_obj.to_trimesh())
        
        # 객체의 로컬 Z 좌표로 변환 필요 (현재는 월드 Z 기준 슬라이스 구현)
        # TODO: 객체 변환(회전, 이동) 반영 처리
        # 우선 가장 단순하게 월드 Z 기준 (평면 origin을 객체 로컬 좌표로 역변환하여 슬라이스)
        
        # (월드 Z) -> (로컬 좌표)
        # 로직: P_world = R * (S * P_local) + T
        # P_local = (1/S) * R^T * (P_world - T)
        
        from scipy.spatial.transform import Rotation as R
        inv_rot = R.from_euler('xyz', self.selected_obj.rotation, degrees=True).inv().as_matrix()
        inv_scale = 1.0 / self.selected_obj.scale if self.selected_obj.scale != 0 else 1.0
        
        # 월드 평면 [0, 0, 1] dot (P - [0, 0, Z_slice]) = 0
        # 로프 좌표에서의 평면 origin과 normal 계산
        world_origin = np.array([0, 0, self.slice_z])
        local_origin = inv_scale * inv_rot @ (world_origin - self.selected_obj.translation)
        
        world_normal = np.array([0,0,1])
        local_normal = inv_rot @ world_normal # 회전만 적용 (법선벡터이므로)
        
        self.slice_contours = slicer.slice_with_plane(local_origin, local_normal)
        
        # 추출된 로컬 좌표 단면을 월드 좌표로 변환하여 저장 (렌더링용)
        rot_mat = R.from_euler('xyz', self.selected_obj.rotation, degrees=True).as_matrix()
        scale = self.selected_obj.scale
        trans = self.selected_obj.translation
        
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
            pass
        self.update()

    def set_cut_lines_enabled(self, enabled: bool):
        self.cut_lines_enabled = bool(enabled)
        if enabled:
            self.picking_mode = 'cut_lines'
            # 프리뷰를 위해 마우스 트래킹 활성화
            self.setMouseTracking(True)
        else:
            if self.picking_mode == 'cut_lines':
                self.picking_mode = 'none'
            self.cut_line_drawing = False
            self.cut_line_preview = None
            self.setMouseTracking(False)
        self.update()

    def clear_cut_line(self, index: int):
        try:
            idx = int(index)
            if idx not in (0, 1):
                return
            self.cut_lines[idx] = []
            self.cut_section_profiles[idx] = []
            self.cut_section_world[idx] = []
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
        self.cut_section_profiles = [[], []]
        self.cut_section_world = [[], []]
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
        return out

    def get_cut_sections_world(self):
        """내보내기용: 바닥에 배치된 단면(프로파일) 폴리라인 2개 월드 좌표 반환"""
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
        return out

    def schedule_cut_section_update(self, index: int, delay_ms: int = 0):
        self._cut_section_pending_index = int(index)
        self._cut_section_timer.start(max(0, int(delay_ms)))

    def _layout_cut_section_world(self, profile: list, index: int):
        obj = self.selected_obj
        if obj is None or not profile:
            return []

        try:
            b = obj.get_world_bounds()
            min_x, min_y = float(b[0][0]), float(b[0][1])
            max_x, max_y = float(b[1][0]), float(b[1][1])
            span_x = max_x - min_x
            span_y = max_y - min_y
            margin = max(5.0, max(span_x, span_y) * 0.05)

            s = np.array([p[0] for p in profile], dtype=np.float64)
            z = np.array([p[1] for p in profile], dtype=np.float64)
            z_min = float(np.nanmin(z)) if len(z) else 0.0

            pts_world = []
            if int(index) == 0:
                # 가로: 메쉬 상단에 배치 (s -> X, z -> Y)
                base_x = min_x
                base_y = max_y + margin
                for si, zi in zip(s.tolist(), z.tolist()):
                    pts_world.append([base_x + float(si), base_y + (float(zi) - z_min), 0.0])
            else:
                # 세로: 메쉬 우측에 배치 (z -> X, s -> Y)
                base_x = max_x + margin
                base_y = min_y
                for si, zi in zip(s.tolist(), z.tolist()):
                    pts_world.append([base_x + (float(zi) - z_min), base_y + float(si), 0.0])
            return pts_world
        except Exception:
            return []

    def _request_cut_section_compute(self):
        if not self.cut_lines_enabled and self.picking_mode != 'cut_lines':
            return

        obj = self.selected_obj
        if obj is None or obj.mesh is None:
            return

        thread = getattr(self, "_cut_section_thread", None)
        if thread is not None and thread.isRunning():
            return

        idx = self._cut_section_pending_index
        self._cut_section_pending_index = None
        if idx is None:
            return
        idx = int(idx)
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
        self._cut_section_thread.start()

    def _on_cut_section_computed(self, result: object):
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
        glPointSize(6.0)
        glBegin(GL_POINTS)
        glVertex3f(float(p0[0]), float(p0[1]), z)
        glVertex3f(float(p1[0]), float(p1[1]), z)
        glEnd()
        glPointSize(1.0)

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
                pass

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

    def _on_crosshair_profile_computed(self, result: dict):
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
            pass
        self.update()

    def _on_crosshair_profile_finished(self):
        thread = getattr(self, "_crosshair_profile_thread", None)
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                pass
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
                pass
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

    def _on_line_profile_computed(self, result: dict):
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
            pass
        self.update()

    def _on_line_profile_finished(self):
        thread = getattr(self, "_line_profile_thread", None)
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                pass
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
        inv_rot = R.from_euler('xyz', obj.rotation, degrees=True).inv().as_matrix()
        inv_scale = 1.0 / obj.scale if obj.scale != 0 else 1.0

        local_origin = inv_scale * inv_rot @ (world_origin - obj.translation)
        local_normal = inv_rot @ world_normal

        slicer = MeshSlicer(obj.to_trimesh())
        contours_local = slicer.slice_with_plane(local_origin, local_normal)

        # 월드 좌표로 변환(렌더링/프로파일용)
        rot_mat = R.from_euler('xyz', obj.rotation, degrees=True).as_matrix()
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
        inv_rot = R.from_euler('xyz', obj.rotation, degrees=True).inv().as_matrix()
        inv_scale = 1.0 / obj.scale if obj.scale != 0 else 1.0
        
        def get_world_to_local(pts_world):
            # P_local = (1/S) * R^T * (P_world - T)
            return inv_scale * (inv_rot @ (pts_world - obj.translation).T).T

        def get_local_to_world(pts_local):
            # P_world = R * (S * P_local) + T
            rot_mat = R.from_euler('xyz', obj.rotation, degrees=True).as_matrix()
            return (rot_mat @ (pts_local * obj.scale).T).T + obj.translation

        # 2. X축 방향 단면 (평면: Y = cy)
        # 월드 상의 평면: Origin=[0, cy, 0], Normal=[0, 1, 0]
        w_orig_x = np.array([0, cy, 0])
        w_norm_x = np.array([0, 1, 0])
        l_orig_x = get_world_to_local(w_orig_x.reshape(1,3))[0]
        l_norm_x = inv_rot @ w_norm_x # 법선은 회전만
        
        # MeshData.section 에러 수정을 위해 to_trimesh() 사용
        slicer = MeshSlicer(obj.to_trimesh())
        contours_x = slicer.slice_with_plane(l_orig_x, l_norm_x)
        
        # 3. Y축 방향 단면 (평면: X = cx)
        w_orig_y = np.array([cx, 0, 0])
        w_norm_y = np.array([1, 0, 0]) # X축에 수직인 평면
        l_orig_y = get_world_to_local(w_orig_y.reshape(1,3))[0]
        l_norm_y = inv_rot @ w_norm_y
        contours_y = slicer.slice_with_plane(l_orig_y, l_norm_y)
        
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

        # 각 모서리 중앙에 화살표 표시
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # 활성 상태에 따라 색상 변경
        def get_color(edge):
            return [1.0, 0.6, 0.0, 1.0] if self.active_roi_edge == edge else [0.0, 0.95, 1.0, 1.0]

        glColor4fv(get_color('bottom')); draw_arrow(mid_x, y1, 'bottom')
        glColor4fv(get_color('top'));    draw_arrow(mid_x, y2, 'top')
        glColor4fv(get_color('left'));   draw_arrow(x1, mid_y, 'left')
        glColor4fv(get_color('right'));  draw_arrow(x2, mid_y, 'right')
        
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
        plane_active = {
            "left": "x1",
            "right": "x2",
            "bottom": "y1",
            "top": "y2",
        }.get(active, None)

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
        self._roi_edges_thread.start()

    def _on_roi_edges_computed(self, edges: object):
        try:
            self.roi_cut_edges = dict(edges)
        except Exception:
            return
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
        rot_mat = R.from_euler('xyz', obj.rotation, degrees=True).as_matrix()
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
        except:
            pass

    
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
        glVertex3f(0, 0, 0); glVertex3f(1, 0, 0)
        # Y: Green
        glColor3f(0.2, 1.0, 0.2)
        glVertex3f(0, 0, 0); glVertex3f(0, 1, 0)
        # Z: Blue
        glColor3f(0.2, 0.2, 1.0)
        glVertex3f(0, 0, 0); glVertex3f(0, 0, 1)
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

    
    def draw_scene_object(self, obj: SceneObject, is_selected: bool = False):
        """개별 메쉬 객체 렌더링"""
        glPushMatrix()
        
        # 변환 적용
        glTranslatef(*obj.translation)
        glRotatef(obj.rotation[0], 1, 0, 0)
        glRotatef(obj.rotation[1], 0, 1, 0)
        glRotatef(obj.rotation[2], 0, 0, 1)
        glScalef(obj.scale, obj.scale, obj.scale)
        
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
            glColor3f(0.85, 0.85, 0.95) # 너무 하얗지 않게 약간 톤다운
        else:
            glColor3f(*obj.color)
            
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
                        glDepthMask(GL_TRUE)
                        glDisable(GL_CLIP_PLANE0)
                except Exception:
                    pass

            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
        
        # 바닥 접촉 면 하이라이트는 정치(바닥 정렬) 관련 모드에서만 표시 (대용량 메쉬 성능)
        if is_selected and self.picking_mode in {'floor_3point', 'floor_face', 'floor_brush'}:
            self._draw_floor_contact_faces(obj)
        
        glPopMatrix()
    
    def _draw_floor_contact_faces(self, obj: SceneObject):
        """바닥(Z=0) 근처 면을 초록색으로 하이라이트 (정치 과정 중 표시)"""
        if obj.mesh is None or obj.mesh.faces is None:
            return
        
        faces = obj.mesh.faces
        vertices = obj.mesh.vertices
        
        # 회전 행렬 계산
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler('xyz', obj.rotation, degrees=True).as_matrix()
        
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
            glLineWidth(5.0); glColor3f(1.0, 0.8, 0.0)
        else: glLineWidth(2.5)
        glPushMatrix(); glRotatef(90, 0, 1, 0); self._draw_gizmo_circle(size); glPopMatrix()
        
        # Y축
        glColor3f(0.2, 1.0, 0.2)
        if highlight_axis == 'Y':
            glLineWidth(5.0); glColor3f(1.0, 0.8, 0.0)
        else: glLineWidth(2.5)
        glPushMatrix(); glRotatef(90, 1, 0, 0); self._draw_gizmo_circle(size); glPopMatrix()
        
        # Z축
        glColor3f(0.2, 0.2, 1.0)
        if highlight_axis == 'Z':
            glLineWidth(5.0); glColor3f(1.0, 0.8, 0.0)
        else: glLineWidth(2.5)
        self._draw_gizmo_circle(size)
        
        glPopMatrix()
        glLineWidth(1.0); glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)

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
            
        # 메쉬 자체를 원점으로 센터링 (로컬 좌표계 생성)
        center = mesh.centroid
        mesh.vertices -= center
        # 캐시 무효화 (vertices 변경)
        try:
            mesh._bounds = None
            mesh._centroid = None
            mesh._surface_area = None
        except Exception:
            pass
        # 로딩 시점에는 face normals만 필요 (vertex normals는 필요할 때 계산)
        mesh.compute_normals(compute_vertex_normals=False)
        
        new_obj = SceneObject(mesh, name)
        self.objects.append(new_obj)
        self.selected_index = len(self.objects) - 1
        
        # VBO 데이터 생성
        self.update_vbo(new_obj)
        
        # 카메라 피팅 (첫 번째 객체인 경우만)
        if len(self.objects) == 1:
            self.update_grid_scale()
        
        self.meshLoaded.emit(mesh)
        self.selectionChanged.emit(self.selected_index)
        self.update()

    def update_grid_scale(self):
        """선택된 메쉬 크기에 맞춰 격자 스케일 조정"""
        obj = self.selected_obj
        if not obj:
            return
            
        bounds = obj.mesh.bounds
        extents = bounds[1] - bounds[0]
        max_dim = np.max(extents)
        
        if max_dim < 50:  self.grid_spacing = 1.0; self.grid_size = 100.0  # 1cm grid for small objects
        elif max_dim < 200: self.grid_spacing = 5.0; self.grid_size = 500.0  # 5cm grid
        else: self.grid_spacing = 10.0; self.grid_size = max_dim * 1.5      # 10cm grid for large objects
            
        self.camera.distance = max_dim * 2
        self.camera.center = obj.translation.copy()
        self.camera.pan_offset = np.array([0.0, 0.0, 0.0])
        self.gizmo_size = max_dim * 0.7
    
    def hit_test_gizmo(self, screen_x, screen_y):
        """기즈모 고리 클릭 검사"""
        obj = self.selected_obj
        if not obj: return None
        
        try:
            self.makeCurrent()
            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            win_y = viewport[3] - screen_y
            
            # Project object center to screen to get a reference point for gizmo size
            obj_screen_pos = gluProject(*obj.translation, modelview, projection, viewport)
            if not obj_screen_pos: return None

            # Calculate a world-space ray from screen coordinates
            near_pt = gluUnProject(screen_x, win_y, 0.0, modelview, projection, viewport)
            far_pt = gluUnProject(screen_x, win_y, 1.0, modelview, projection, viewport)
            if not near_pt or not far_pt: return None
            
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
        except Exception as e:
            # print(f"Gizmo hit test error: {e}")
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
        except Exception as e:
            print(f"VBO creation failed for {obj.name}: {e}")
    
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
        if not obj: return
        
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
                pass
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
            pass
        
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
            pass
        
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
        if not obj: return
        
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
    
    def mousePressEvent(self, event: QMouseEvent):
        """마우스 버튼 눌림"""
        try:
            self.last_mouse_pos = event.pos()
            self.mouse_button = event.button()
            modifiers = event.modifiers()

            # 1. 일반 클릭 (객체 선택 또는 피킹 모드 처리) - 좌클릭만 처리
            if event.button() == Qt.MouseButton.LeftButton:
                # 기즈모 선택 검사 (가장 우선순위)
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
                        
                elif self.picking_mode == 'floor_3point':
                    point = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                    if point is not None:
                        # 로컬 좌표로 변환하여 전달 (작업 도중 객체가 움직여도 점이 붙어있게 함)
                        local_pt = point - self.selected_obj.translation
                        
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

                elif self.picking_mode == 'cut_lines':
                    if event.button() != Qt.MouseButton.LeftButton:
                        return
                    pt = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                    if pt is None:
                        return

                    idx = int(getattr(self, "cut_line_active", 0))
                    if idx not in (0, 1):
                        idx = 0
                    line = self.cut_lines[idx]
                    # 기존 단면 결과는 편집 시 무효화
                    self.cut_section_profiles[idx] = []
                    self.cut_section_world[idx] = []
                    p = np.array([pt[0], pt[1], 0.0], dtype=np.float64)
                    if len(line) == 0:
                        line.append(p)
                    else:
                        last = np.asarray(line[-1], dtype=np.float64)
                        p2 = self._cutline_constrain_ortho(last, p)
                        if float(np.linalg.norm(p2[:2] - last[:2])) > 1e-6:
                            line.append(p2)

                    self.cut_line_drawing = True
                    self.cut_line_preview = None
                    self.update()
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
                    # ROI 핸들 클릭 검사
                    self.active_roi_edge = self._hit_test_roi(event.pos())
                    if self.active_roi_edge:
                        self.update()
                        return

            # 3. 객체 조작 (Shift/Ctrl + 드래그)
            obj = self.selected_obj
            if obj and (modifiers & Qt.KeyboardModifier.ShiftModifier or modifiers & Qt.KeyboardModifier.ControlModifier):
                 self.save_undo_state() # 변환 시작 전 상태 저장
                 
                 # Ctrl+드래그(이동)를 위한 초기 깊이값 저장 (마우스가 가리키는 지점의 깊이)
                 if modifiers & Qt.KeyboardModifier.ControlModifier:
                     self.makeCurrent()
                     self._cached_viewport = glGetIntegerv(GL_VIEWPORT)
                     self._cached_modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
                     self._cached_projection = glGetDoublev(GL_PROJECTION_MATRIX)
                     
                     win_y = self._cached_viewport[3] - event.pos().y()
                     depth = glReadPixels(event.pos().x(), win_y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
                     self._drag_depth = depth[0][0]
                     
                     # 배경을 클릭한 경우 객체 중심의 깊이 사용
                     if self._drag_depth >= 1.0:
                         obj_win_pos = gluProject(*obj.translation, self._cached_modelview, self._cached_projection, self._cached_viewport)
                         if obj_win_pos:
                             self._drag_depth = obj_win_pos[2]
                 return
            
            # 4. 휠 클릭: 포커스 이동 (Focus move)
            if event.button() == Qt.MouseButton.MiddleButton and modifiers == Qt.KeyboardModifier.NoModifier:
                point = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                if point is not None:
                    self.camera.center = point
                    self.camera.pan_offset = np.array([0.0, 0.0, 0.0])
                    self.update()
                    return

        except Exception as e:
            print(f"Mouse press error: {e}")

    def mouseReleaseEvent(self, event: QMouseEvent):
        """마우스 버튼 놓음"""
        if self.mouse_button == Qt.MouseButton.LeftButton and self.picking_mode == 'floor_brush':
            if self.brush_selected_faces:
                self.alignToBrushSelected.emit()
            self.picking_mode = 'none'
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

        self.mouse_button = None
        self.active_gizmo_axis = None
        self.gizmo_drag_start = None
        self.last_mouse_pos = None
        
        # 캐시 초기화
        self._cached_viewport = None
        self._cached_modelview = None
        self._cached_projection = None
        
        self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """마우스 이동 (드래그)"""
        try:
            # 단면선(2개) 프리뷰: 마우스 트래킹(버튼 없이 이동)에서도 동작
            if self.picking_mode == 'cut_lines' and getattr(self, "cut_line_drawing", False):
                if self.mouse_button is None or self.mouse_button == Qt.MouseButton.LeftButton:
                    pt = self.pick_point_on_plane_z(event.pos().x(), event.pos().y(), z=0.0)
                    if pt is not None:
                        p = np.array([pt[0], pt[1], 0.0], dtype=np.float64)
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
                    if self.active_gizmo_axis == 'X': axis_vec[0] = 1.0
                    elif self.active_gizmo_axis == 'Y': axis_vec[1] = 1.0
                    elif self.active_gizmo_axis == 'Z': axis_vec[2] = 1.0
                    
                    # 시각적 반전 여부 결정 (핸들을 돌리는 방향과 메쉬가 도는 방향 일치)
                    dot = np.dot(view_dir, axis_vec)
                    flip = 1.0 if dot > 0 else -1.0
                    
                    if self.active_gizmo_axis == 'X': obj.rotation[0] += delta_angle * flip
                    elif self.active_gizmo_axis == 'Y': obj.rotation[1] += delta_angle * flip
                    elif self.active_gizmo_axis == 'Z': obj.rotation[2] += delta_angle * flip
                        
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
            if (modifiers & Qt.KeyboardModifier.ControlModifier) and obj and self._cached_viewport is not None:
                # 마우스 프레스 시 캡처된 깊이와 매트릭스 재사용 (성능 향상)
                curr_win_y = self._cached_viewport[3] - event.pos().y()
                prev_win_y = self._cached_viewport[3] - prev_pos.y()
                
                curr_world = gluUnProject(event.pos().x(), curr_win_y, self._drag_depth, 
                                          self._cached_modelview, self._cached_projection, self._cached_viewport)
                prev_world = gluUnProject(prev_pos.x(), prev_win_y, self._drag_depth, 
                                          self._cached_modelview, self._cached_projection, self._cached_viewport)
                
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
            
            elif (modifiers & Qt.KeyboardModifier.AltModifier) and obj:
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
            
            elif (modifiers & Qt.KeyboardModifier.ShiftModifier) and obj and self.mouse_button == Qt.MouseButton.LeftButton and self.picking_mode != 'line_section':
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
            
            # 0.6 ROI 핸들 드래그 처리
            if self.roi_enabled and self.active_roi_edge and self.mouse_button == Qt.MouseButton.LeftButton:
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
                    
                    if self.active_roi_edge == 'left':   self.roi_bounds[0] = min(wx, self.roi_bounds[1] - 0.1)
                    elif self.active_roi_edge == 'right':  self.roi_bounds[1] = max(wx, self.roi_bounds[0] + 0.1)
                    elif self.active_roi_edge == 'bottom': self.roi_bounds[2] = min(wy, self.roi_bounds[3] - 0.1)
                    elif self.active_roi_edge == 'top':    self.roi_bounds[3] = max(wy, self.roi_bounds[2] + 0.1)
                    
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
            
        except Exception as e:
            print(f"Mouse move error: {e}")
    
    def _calculate_gizmo_angle(self, screen_x, screen_y):
        """기즈모 중심 기준 2D 화면 공간에서의 각도 계산 (가장 직관적인 원형 드래그 방식)"""
        obj = self.selected_obj
        if not obj or not self.active_gizmo_axis: return None
        
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
            if not win_pos: return None
            
            center_x = win_pos[0]
            center_y = viewport[3] - win_pos[1]
            
            # 중심점에서 마우스 포인터까지의 2D 각도 (atan2)
            # 화면 좌표계는 Y가 아래로 증가하므로 부호 주의
            dx = screen_x - center_x
            dy = screen_y - center_y
            
            angle = np.arctan2(dy, dx)
            return angle
        except Exception as e:
            # print(f"Gizmo angle error: {e}")
            return None
    
    def wheelEvent(self, event: QWheelEvent):
        """마우스 휠 (줌)"""
        delta = event.angleDelta().y()
        self.camera.zoom(delta)
        self.update()
    
    def keyPressEvent(self, event):
        """키보드 입력"""
        self.keys_pressed.add(event.key())
        if event.key() in (Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D, Qt.Key.Key_Q, Qt.Key.Key_E):
            if not self.move_timer.isActive():
                self.move_timer.start()

        # 0. 단면선(2개) 도구 단축키
        if self.picking_mode == 'cut_lines':
            key = event.key()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                idx_done = int(getattr(self, "cut_line_active", 0))
                self.cut_line_drawing = False
                self.cut_line_preview = None
                # 단면(프로파일) 계산 요청
                try:
                    if idx_done in (0, 1) and len(self.cut_lines[idx_done]) >= 2:
                        self.schedule_cut_section_update(idx_done, delay_ms=0)
                except Exception:
                    pass
                # 첫 번째 선을 끝냈고 두 번째가 비어있으면 자동 전환
                try:
                    if idx_done == 0 and not self.cut_lines[1]:
                        self.cut_line_active = 1
                except Exception:
                    pass
                self.update()
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
                    if not line:
                        self.cut_line_drawing = False
                        self.cut_line_preview = None
                    self.update()
                except Exception:
                    pass
                return
            if key == Qt.Key.Key_Tab:
                try:
                    self.cut_line_active = 1 - int(getattr(self, "cut_line_active", 0))
                    self.cut_line_preview = None
                    self.update()
                except Exception:
                    pass
                return
            if key == Qt.Key.Key_Escape:
                # 도구는 유지하고, 현재 프리뷰/드로잉만 취소
                self.cut_line_drawing = False
                self.cut_line_preview = None
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
            
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
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
        if Qt.Key.Key_W in self.keys_pressed: dz += 1
        if Qt.Key.Key_S in self.keys_pressed: dz -= 1
        if Qt.Key.Key_A in self.keys_pressed: dx -= 1
        if Qt.Key.Key_D in self.keys_pressed: dx += 1
        if Qt.Key.Key_Q in self.keys_pressed: dy += 1
        if Qt.Key.Key_E in self.keys_pressed: dy -= 1
        
        if dx != 0 or dy != 0 or dz != 0:
            self.camera.move_relative(dx, dy, dz)
            self.update()
    
    def _hit_test_roi(self, pos):
        """ROI 핸들(화살표) 클릭 검사"""
        ray_origin, ray_dir = self.get_ray(pos.x(), pos.y())
        if ray_origin is None or ray_dir is None:
            return None
        # Z=0 평면상의 좌표 계산
        if abs(ray_dir[2]) < 1e-6: return None
        t = -ray_origin[2] / ray_dir[2]
        hit_pt = ray_origin + t * ray_dir
        wx, wy = hit_pt[0], hit_pt[1]
        
        x1, x2, y1, y2 = self.roi_bounds
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # 카메라 거리 기반 동적 히트 테스트 반경
        threshold = self.camera.distance * 0.05
        
        # 각 핸들 위치와 거리 체크
        if np.hypot(wx - mid_x, wy - y1) < threshold: return 'bottom'
        if np.hypot(wx - mid_x, wy - y2) < threshold: return 'top'
        if np.hypot(wx - x1, wy - mid_y) < threshold: return 'left'
        if np.hypot(wx - x2, wy - mid_y) < threshold: return 'right'
        
        return None

    def capture_high_res_image(self, width: int = 2048, height: int = 2048, *, only_selected: bool = False):
        """고해상도 오프스크린 렌더링"""
        self.makeCurrent()
        
        # 1. FBO 생성
        fbo = QOpenGLFramebufferObject(width, height, QOpenGLFramebufferObject.Attachment.Depth)
        fbo.bind()
        
        # 2. 렌더링 설정 (오프스크린용)
        glViewport(0, 0, width, height)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        aspect = width / height
        gluPerspective(45.0, aspect, 0.1, 1000000.0)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        
        # 3. 그리기 (UI 제외하고 깨끗하게)
        glClearColor(1.0, 1.0, 1.0, 1.0) # 화이트 배경
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        self.camera.apply()
        
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
        glViewport(0, 0, self.width(), self.height())
        self.update()
        
        return qimage, mv, proj, vp

    def get_ray(self, screen_x: int, screen_y: int):
        """화면 좌표에서 월드 레이(origin, dir) 계산"""
        try:
            self.makeCurrent()

            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)

            win_y = viewport[3] - screen_y

            near_pt = gluUnProject(screen_x, win_y, 0.0, modelview, projection, viewport)
            far_pt = gluUnProject(screen_x, win_y, 1.0, modelview, projection, viewport)
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
        
    def pick_point_on_mesh(self, screen_x: int, screen_y: int):
        """화면 좌표를 메쉬 표면의 3D 좌표로 변환"""
        if not self.objects: return None
        obj = self.selected_obj
        if not obj: return None
        
        # OpenGL 뷰포트, 투영, 모델뷰 행렬 가져오기
        self.makeCurrent()
        
        viewport = glGetIntegerv(GL_VIEWPORT)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        # 화면 Y 좌표 반전 (OpenGL은 아래가 0)
        win_y = viewport[3] - screen_y
        
        # 깊이 버퍼에서 깊이 값 읽기
        depth = glReadPixels(screen_x, win_y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
        depth_value = depth[0][0]
        
        # 배경을 클릭한 경우
        if depth_value >= 1.0:
            return None
        
        # 화면 좌표를 월드 좌표로 변환
        world_x, world_y, world_z = gluUnProject(
            screen_x, win_y, depth_value,
            modelview, projection, viewport
        )
        
        return np.array([world_x, world_y, world_z])
    

    def _pick_brush_face(self, pos):
        """브러시로 면 선택"""
        point = self.pick_point_on_mesh(pos.x(), pos.y())
        if point is not None:
            res = self.pick_face_at_point(point, return_index=True)
            if res:
                idx, v = res
                self.brush_selected_faces.add(idx)

    def pick_face_at_point(self, point: np.ndarray, return_index=False):
        """특정 3D 좌표가 포함된 삼각형 면의 정점 3개를 반환"""
        obj = self.selected_obj
        if not obj or obj.mesh is None: return None
        
        # 메쉬 로컬 좌표로 변환
        inv_trans = -obj.translation
        lp = point + inv_trans
        
        vertices = obj.mesh.vertices
        faces = obj.mesh.faces
        
        best_face_idx = -1
        min_dist = float('inf')
        
        # 최적화: 클릭한 지점 주변의 삼각형을 더 잘 찾기 위해
        for idx, face in enumerate(faces):
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            centroid = (v0 + v1 + v2) / 3.0
            dist = np.linalg.norm(lp - centroid)
            if dist < min_dist:
                min_dist = dist
                best_face_idx = idx
            
            if dist < 0.05: # 더 정밀하게
                break
                
        if best_face_idx != -1:
            best_face = faces[best_face_idx]
            v_list = [vertices[best_face[0]] + obj.translation, 
                      vertices[best_face[1]] + obj.translation, 
                      vertices[best_face[2]] + obj.translation]
            if return_index:
                return best_face_idx, v_list
            return v_list
        return None

    def draw_picked_points(self):
        """찍은 점들을 작은 빨간 구로 시각화"""
        if not self.picked_points:
            return
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)  # 항상 앞에 보이게
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # 아주 작은 마젠타 구형 점
        glColor4f(0.9, 0.2, 0.9, 0.9)  # 마젠타
        
        for i, point in enumerate(self.picked_points):
            glPushMatrix()
            glTranslatef(point[0], point[1], point[2])
            
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
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
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


# 테스트용 스탠드얼론 실행
if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    
    viewport = Viewport3D()
    viewport.setWindowTitle("3D Viewport Test")
    viewport.resize(800, 600)
    viewport.show()
    
    sys.exit(app.exec())
