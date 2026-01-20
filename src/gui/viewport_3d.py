"""
3D Viewport Widget with OpenGL
Copyright (C) 2026 balguljang2 (lzpxilfe)
Licensed under the GNU General Public License v2.0 (GPL2)
"""

import numpy as np
from typing import Optional, Tuple

from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QPainter, QColor
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from OpenGL.GL import *
from OpenGL.GLU import *
import ctypes


class TrackballCamera:
    """
    Trackball 스타일 카메라 (CloudCompare 방식)
    
    조작:
    - 좌클릭 드래그: 회전
    - 우클릭 드래그: 이동 (Pan)
    - 스크롤: 확대/축소
    """
    
    def __init__(self):
        # 카메라 위치 (구면 좌표)
        self.distance = 50.0  # cm
        self.azimuth = 45.0   # 수평 각도 (도)
        self.elevation = 30.0 # 수직 각도 (도)
        
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
        """카메라 위치 (직교 좌표)"""
        az_rad = np.radians(self.azimuth)
        el_rad = np.radians(self.elevation)
        
        x = self.distance * np.cos(el_rad) * np.sin(az_rad)
        y = self.distance * np.sin(el_rad)
        z = self.distance * np.cos(el_rad) * np.cos(az_rad)
        
        return np.array([x, y, z]) + self.center + self.pan_offset
    
    @property
    def up_vector(self) -> np.ndarray:
        """카메라 업 벡터"""
        return np.array([0.0, 1.0, 0.0])
    
    @property
    def look_at(self) -> np.ndarray:
        """시선 방향 타겟"""
        return self.center + self.pan_offset
    
    def rotate(self, delta_x: float, delta_y: float, sensitivity: float = 0.5):
        """카메라 회전"""
        self.azimuth += delta_x * sensitivity
        self.elevation += delta_y * sensitivity
        
        # 수직 각도 제한 (-89 ~ 89도)
        self.elevation = max(-89.0, min(89.0, self.elevation))
    
    def pan(self, delta_x: float, delta_y: float, sensitivity: float = 0.3):
        """카메라 이동 (Pan) - 더 자연스럽게"""
        # 카메라의 오른쪽/위쪽 방향 계산
        az_rad = np.radians(self.azimuth)
        el_rad = np.radians(self.elevation)
        
        # 카메라 기준 오른쪽 벡터
        right = np.array([np.cos(az_rad), 0, -np.sin(az_rad)])
        
        # 카메라 기준 위쪽 벡터 (elevation 고려)
        up = np.array([
            -np.sin(el_rad) * np.sin(az_rad),
            np.cos(el_rad),
            -np.sin(el_rad) * np.cos(az_rad)
        ])
        
        # Pan 속도 = 거리에 비례하되 적당한 배율
        pan_speed = self.distance * sensitivity * 0.005
        self.pan_offset += right * (-delta_x * pan_speed)
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
        """카메라 로컬 좌표계 기준 이동 (WASD)"""
        az_rad = np.radians(self.azimuth)
        
        # 1. 오른쪽 벡터 (X)
        right = np.array([np.cos(az_rad), 0, -np.sin(az_rad)])
        
        # 2. 위쪽 벡터 (Y)
        up = np.array([0, 1, 0]) 
        
        # 3. 전진 벡터 (Z) - 수평 방향 전진
        forward = np.array([-np.sin(az_rad), 0, -np.cos(az_rad)])
        
        # 이동 속도를 좀 더 유연하게 (거리 비례 + 기본 속도)
        move_speed = (self.distance * 0.08 + 10.0) * sensitivity
        
        self.center += right * (dx * move_speed)
        self.center += up * (dy * move_speed)
        self.center += forward * (dz * move_speed)

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
        
        # 개별 변환 상태
        self.translation = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])
        self.scale = 1.0
        
        # 렌더링 리소스
        self.vbo_id = None
        self.vertex_count = 0
        self.color = np.array([0.8, 0.8, 0.8])
        self.selected_faces = set()
        
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
        self.grid_size = 200.0  # cm (더 크게)
        self.grid_spacing = 10.0 # cm
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
        
        # 키보드 조작 타이머 (WASD 연속 이동용)
        self.keys_pressed = set()
        self.move_timer = QTimer()
        self.move_timer.timeout.connect(self.process_keyboard_navigation)
        self.move_timer.start(16) # ~60fps
        
        # UI 설정
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    @property
    def selected_obj(self) -> Optional[SceneObject]:
        if 0 <= self.selected_index < len(self.objects):
            return self.objects[self.selected_index]
        return None
    
    def initializeGL(self):
        """OpenGL 초기화"""
        glClearColor(*self.bg_color)
        
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # 광원 설정
        light_pos = [50.0, 100.0, 50.0, 0.0]
        light_ambient = [0.3, 0.3, 0.3, 1.0]
        light_diffuse = [0.8, 0.8, 0.8, 1.0]
        
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        
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
        
        # 1. 격자 및 축
        self.draw_grid()
        self.draw_axes()
        
        # 2. 모든 메쉬 객체 렌더링
        for i, obj in enumerate(self.objects):
            if not obj.visible:
                continue
            self.draw_scene_object(obj, is_selected=(i == self.selected_index))
            
        # 3. 곡률 피팅 요소
        self.draw_picked_points()
        self.draw_fitted_arc()
        
        # 4. 회전 기즈모 (선택된 객체에만)
        if self.selected_obj:
            self.draw_rotation_gizmo(self.selected_obj)
    
    def draw_grid(self):
        """진정한 무한 격자 바닥면 그리기 (카메라 거리에 따라 동적 스케일링)"""
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
                
            view_range = min(view_range, 100000.0) # 최대 1km (안정성을 위해 10km에서 1km로 조정)
            half_range = view_range / 2
            
            # 투명도 조절 (레벨이 높을수록, 혹은 카메라 중심에서 멀수록 연하게)
            # 최소 알파를 0.15로 높여 무한한 격자가 항상 보이도록 함
            alpha = max(0.15, min(0.4, 2.0 / (level ** 0.5)))
            glColor4f(0.7, 0.7, 0.7, alpha)
            
            snap_x = round(cam_center[0] / spacing) * spacing
            snap_z = round(cam_center[2] / spacing) * spacing
            
            glLineWidth(1.0 if level == 1 else 1.5)
            glBegin(GL_LINES)
            steps = 100
            for i in range(-steps // 2, steps // 2 + 1):
                offset = i * spacing
                x_val = snap_x + offset
                glVertex3f(x_val, 0, snap_z - half_range)
                glVertex3f(x_val, 0, snap_z + half_range)
                
                z_val = snap_z + offset
                glVertex3f(snap_x - half_range, 0, z_val)
                glVertex3f(snap_x + half_range, 0, z_val)
            glEnd()
            
        glEnable(GL_LIGHTING)
    
    def draw_axes(self):
        """XYZ 축 그리기 (무한히 뻗어나감)"""
        glDisable(GL_LIGHTING)
        
        # 축 길이를 카메라 거리에 비례하여 대폭 확장 (사실상 끝이 안 보이게)
        # 기본 10km 이상으로 설정하여 무한한 느낌 강화
        axis_length = max(self.camera.distance * 100, 1000000.0)
        
        glLineWidth(3.0)
        glBegin(GL_LINES)
        
        # X축 (빨강)
        glColor3f(0.9, 0.2, 0.2)
        glVertex3f(-axis_length, 0, 0)
        glVertex3f(axis_length, 0, 0)
        
        # Y축 (초록)
        glColor3f(0.2, 0.8, 0.2)
        glVertex3f(0, -axis_length, 0)
        glVertex3f(0, axis_length, 0)
        
        # Z축 (파랑)
        glColor3f(0.2, 0.2, 0.9)
        glVertex3f(0, 0, -axis_length)
        glVertex3f(0, 0, axis_length)
        
        glEnd()
        glLineWidth(1.0)
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
        
        # 메쉬 색상
        if is_selected:
            glColor3f(0.8, 0.8, 1.0) # 선택 시 약간 푸른빛
        else:
            glColor3f(*obj.color)
        
        if obj.vbo_id is not None:
            # VBO 방식 렌더링
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            
            glBindBuffer(GL_ARRAY_BUFFER, obj.vbo_id)
            glVertexPointer(3, GL_FLOAT, 24, ctypes.c_void_p(0))
            glNormalPointer(GL_FLOAT, 24, ctypes.c_void_p(12))
            
            glDrawArrays(GL_TRIANGLES, 0, obj.vertex_count)
            
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
        
        glPopMatrix()

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
        
        # X축
        glColor3f(1.0, 0.2, 0.2)
        if self.active_gizmo_axis == 'X':
            glLineWidth(5.0); glColor3f(1.0, 0.8, 0.0)
        else: glLineWidth(2.5)
        glPushMatrix(); glRotatef(90, 0, 1, 0); self._draw_gizmo_circle(size); glPopMatrix()
        
        # Y축
        glColor3f(0.2, 1.0, 0.2)
        if self.active_gizmo_axis == 'Y':
            glLineWidth(5.0); glColor3f(1.0, 0.8, 0.0)
        else: glLineWidth(2.5)
        glPushMatrix(); glRotatef(90, 1, 0, 0); self._draw_gizmo_circle(size); glPopMatrix()
        
        # Z축
        glColor3f(0.2, 0.2, 1.0)
        if self.active_gizmo_axis == 'Z':
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
        mesh.compute_normals()
        
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
        
        if max_dim < 10:  self.grid_spacing = 1.0; self.grid_size = 20.0
        elif max_dim < 100: self.grid_spacing = 5.0; self.grid_size = 150.0
        else: self.grid_spacing = 10.0; self.grid_size = max_dim * 1.5
            
        self.camera.distance = max_dim * 2
        self.camera.center = obj.translation.copy()
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
            v_indices = obj.mesh.faces.flatten()
            vertices = obj.mesh.vertices[v_indices].astype(np.float32)
            normals = np.repeat(obj.mesh.face_normals, 3, axis=0).astype(np.float32)
            
            data = np.hstack([vertices, normals]).flatten()
            obj.vertex_count = len(vertices)
            
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
            obj = self.objects[index]
            self.selectionChanged.emit(index)
            self.update()
    
    def mousePressEvent(self, event: QMouseEvent):
        """마우스 버튼 눌림"""
        try:
            # Shift+클릭: 곡률 측정용 점 찍기
            if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier and 
                event.button() == Qt.MouseButton.LeftButton and
                self.curvature_pick_mode):
                
                point = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                if point is not None:
                    self.picked_points.append(point)
                    self.update()
                return
            
            # 기즈모 클릭 체크
            if event.button() == Qt.MouseButton.LeftButton:
                axis = self.hit_test_gizmo(event.pos().x(), event.pos().y())
                if axis:
                    self.active_gizmo_axis = axis
                    angle = self._calculate_gizmo_angle(event.pos().x(), event.pos().y())
                    if angle is not None:
                        self.gizmo_drag_start = angle
                        self.mouse_button = Qt.MouseButton.LeftButton  # 중요: 버튼 상태 설정
                        self.last_mouse_pos = event.pos()
                        self.update()
                        return
            
            # 휠 클릭: 포커스 이동 (Focus move)
            if event.button() == Qt.MouseButton.MiddleButton:
                point = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
                if point is not None:
                    self.camera.center = point
                    self.camera.pan_offset = np.array([0.0, 0.0, 0.0])
                    self.update()
            
            self.last_mouse_pos = event.pos()
            self.mouse_button = event.button()
        except Exception as e:
            print(f"Mouse press error: {e}")
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """마우스 버튼 놓음"""
        self.active_gizmo_axis = None
        self.gizmo_drag_start = None
        self.last_mouse_pos = None
        self.mouse_button = None
        self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """마우스 이동 (드래그)"""
        try:
            if self.last_mouse_pos is None:
                self.last_mouse_pos = event.pos()
                return

            dx = event.pos().x() - self.last_mouse_pos.x()
            dy = event.pos().y() - self.last_mouse_pos.y()
            self.last_mouse_pos = event.pos()
            
            obj = self.selected_obj
            modifiers = event.modifiers()
            
            # 1. 기즈모 드래그 (좌클릭 + 기즈모 활성 상태)
            if self.active_gizmo_axis and obj and self.mouse_button == Qt.MouseButton.LeftButton and self.gizmo_drag_start is not None:
                angle_info = self._calculate_gizmo_angle(event.pos().x(), event.pos().y())
                if angle_info is not None:
                    current_angle = angle_info
                    delta_angle = np.degrees(current_angle - self.gizmo_drag_start)
                    
                    if self.active_gizmo_axis == 'X': obj.rotation[0] += delta_angle
                    elif self.active_gizmo_axis == 'Y': obj.rotation[1] -= delta_angle
                    elif self.active_gizmo_axis == 'Z': obj.rotation[2] += delta_angle
                        
                    self.gizmo_drag_start = current_angle
                    self.meshTransformChanged.emit()
                    self.update()
                    return
                
            # 2. 기즈모 호버 하이라이트 (버튼 안 눌렸을 때)
            if event.buttons() == Qt.MouseButtons.NoButton:
                axis = self.hit_test_gizmo(event.pos().x(), event.pos().y())
                if axis != self.active_gizmo_axis:
                    self.active_gizmo_axis = axis
                    self.update()
                return
            
            # 3. 객체 직접 조작 (Ctrl / Alt)
            if (modifiers & Qt.KeyboardModifier.ControlModifier) and obj:
                az_rad = np.radians(self.camera.azimuth)
                move_speed = self.camera.distance * 0.002
                dx_world = -dx * np.cos(az_rad) * move_speed
                dz_world = -dx * np.sin(az_rad) * move_speed
                dy_world = -dy * move_speed
                
                obj.translation[0] += dx_world
                obj.translation[1] += dy_world
                obj.translation[2] += dz_world
                self.meshTransformChanged.emit()
                self.update()
                return
            
            elif (modifiers & Qt.KeyboardModifier.AltModifier) and obj:
                rot_speed = 0.5
                obj.rotation[1] += dx * rot_speed
                obj.rotation[0] += dy * rot_speed
                self.meshTransformChanged.emit()
                self.update()
                return
                
            # 4. 일반 카메라 조작 (언제나 가능해야 함)
            if self.mouse_button == Qt.MouseButton.LeftButton:
                self.camera.rotate(dx, dy)
                self.update()
            elif self.mouse_button == Qt.MouseButton.RightButton:
                self.camera.pan(dx, dy)
                self.update()
            elif self.mouse_button == Qt.MouseButton.MiddleButton:
                # 휠 드래그로도 회전 가능하게 (사용자 편의)
                self.camera.rotate(dx, dy)
                self.update()
            
        except Exception as e:
            print(f"Mouse move error: {e}")
    
    def _calculate_gizmo_angle(self, screen_x, screen_y):
        """기즈모 중심에서 마우스 포인터까지의 각도 계산"""
        obj = self.selected_obj
        if not obj: return None
        try:
            self.makeCurrent()
            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            win_y = viewport[3] - screen_y
            p_screen = gluProject(*obj.translation, modelview, projection, viewport)
            if not p_screen: return None
            dx = screen_x - p_screen[0]
            dy = win_y - p_screen[1]
            return np.arctan2(dy, dx) if abs(dx) > 1e-6 or abs(dy) > 1e-6 else 0.0
        except: return None
    
    def wheelEvent(self, event: QWheelEvent):
        """마우스 휠 (줌)"""
        delta = event.angleDelta().y()
        self.camera.zoom(delta)
        self.update()
    
    def keyPressEvent(self, event):
        """키보드 입력"""
        self.keys_pressed.add(event.key())
        
        key = event.key()
        # R: 카메라 리셋
        if key == Qt.Key.Key_R:
            self.camera.reset()
            self.update()
        # F: 메쉬에 맞춤
        elif key == Qt.Key.Key_F:
            self.fit_view_to_selected_object()
        
        # 숫자 키 조작 (생략 가능, TrackballCamera 내부적으로 이미 처리됨)
        # ...
    
    def keyReleaseEvent(self, event):
        if event.key() in self.keys_pressed:
            self.keys_pressed.remove(event.key())

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
    
    def draw_picked_points(self):
        """찍은 점들을 빨간 구로 시각화"""
        if not self.picked_points:
            return
        
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 0.2, 0.2)  # 빨간색
        
        for point in self.picked_points:
            glPushMatrix()
            glTranslatef(point[0], point[1], point[2])
            
            # 간단한 점 대신 십자 마커
            size = 0.5  # cm
            glLineWidth(3.0)
            glBegin(GL_LINES)
            glVertex3f(-size, 0, 0)
            glVertex3f(size, 0, 0)
            glVertex3f(0, -size, 0)
            glVertex3f(0, size, 0)
            glVertex3f(0, 0, -size)
            glVertex3f(0, 0, size)
            glEnd()
            
            glPopMatrix()
        
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def draw_fitted_arc(self):
        """피팅된 원호 시각화"""
        if self.fitted_arc is None:
            return
        
        from src.core.curvature_fitter import CurvatureFitter
        
        glDisable(GL_LIGHTING)
        glColor3f(0.2, 0.8, 0.2)  # 초록색
        glLineWidth(2.0)
        
        # 원호 점들 생성
        fitter = CurvatureFitter()
        arc_points = fitter.generate_arc_points(self.fitted_arc, 64)
        
        # 원 그리기
        glBegin(GL_LINE_LOOP)
        for point in arc_points:
            glVertex3fv(point)
        glEnd()
        
        # 중심에서 원주까지 선 (반지름 표시)
        glColor3f(1.0, 1.0, 0.0)  # 노란색
        glBegin(GL_LINES)
        glVertex3fv(self.fitted_arc.center)
        glVertex3fv(arc_points[0])
        glEnd()
        
        glLineWidth(1.0)
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
