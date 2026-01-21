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
        """카메라 업 벡터 - Z-up"""
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
        """카메라 이동 (Pan) - Z-up 좌표계, 잡고 끌어오는 방식"""
        az_rad = np.radians(self.azimuth)
        el_rad = np.radians(self.elevation)
        
        # 카메라 기준 오른쪽 벡터 (XY 평면에서)
        right = np.array([-np.sin(az_rad), np.cos(az_rad), 0])
        
        # 카메라 기준 위쪽 벡터 (Z-up 고려)
        up = np.array([
            -np.sin(el_rad) * np.cos(az_rad),
            -np.sin(el_rad) * np.sin(az_rad),
            np.cos(el_rad)
        ])
        
        # Pan 속도 = 거리에 비례하되 적당한 배율
        pan_speed = self.distance * sensitivity * 0.005
        # 잡고 끌어오는 방향으로 (마우스 이동 반대방향으로 카메라 이동)
        self.pan_offset += right * (delta_x * pan_speed)
        self.pan_offset += up * (-delta_y * pan_speed)
    
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
        """카메라 로컬 좌표계 기준 이동 (WASD) - Z-up"""
        az_rad = np.radians(self.azimuth)
        
        # 1. 오른쪽 벡터 (A/D)
        right = np.array([-np.sin(az_rad), np.cos(az_rad), 0])
        
        # 2. 위쪽 벡터 (Q/E)
        up = np.array([0, 0, 1]) 
        
        # 3. 전진 벡터 (W/S) - 카메라가 바라보는 방향
        forward = np.array([-np.cos(az_rad), -np.sin(az_rad), 0])
        
        # 이동 속도
        move_speed = (self.distance * 0.03 + 3.0) * sensitivity
        
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
        
        # 피팅된 원호들 (메쉬와 함께 이동)
        self.fitted_arcs = []
        
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
    floorPointPicked = pyqtSignal(np.ndarray) # 바닥면 점 선택됨
    floorFacePicked = pyqtSignal(list)        # 바닥면 3점(면) 선택됨
    alignToBrushSelected = pyqtSignal()      # 브러시 선택 영역으로 정렬 요청
    floorAlignmentConfirmed = pyqtSignal()   # Enter 키로 정렬 확정 시 발생
    
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
        self.floor_picks = []  # 바닥 3점 정렬용 점 리스트
        
        # Undo/Redo 시스템
        self.undo_stack = []
        self.max_undo = 50
        
        # 키보드 조작 타이머 (WASD 연속 이동용)
        self.keys_pressed = set()
        self.move_timer = QTimer()
        self.move_timer.timeout.connect(self.process_keyboard_navigation)
        self.move_timer.start(16) # ~60fps
        
        # UI 설정
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # 타이머 (for continuous update, e.g., for status info)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~60 FPS
    
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
        for i, obj in enumerate(self.objects):
            if not obj.visible:
                continue
            self.draw_scene_object(obj, is_selected=(i == self.selected_index))
            
        # 3. 곡률 피팅 요소
        self.draw_picked_points()
        self.draw_fitted_arc()
        
        # 3.5 바닥 정렬 점 표시
        self.draw_floor_picks()
        
        # 4. 회전 기즈모 (선택된 객체에만)
        if self.selected_obj:
            self.draw_rotation_gizmo(self.selected_obj)
            # 메쉬 치수/중심점 오버레이
            self.draw_mesh_dimensions(self.selected_obj)
            
        # 5. UI 오버레이 (HUD)
        self.draw_orientation_hud()
    
    def draw_ground_plane(self):
        """반투명 바닥면 그리기 (Z=0, XY 평면) - Z-up 좌표계"""
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
            world_z = obj.mesh.vertices[:, 2] + obj.translation[2]
            near_floor_count = np.sum(world_z < 0.1) # 바닥 아래로 내려갔거나 거의 닿음
            touching_floor = near_floor_count > 10 # 10개 이상의 점만 있어도 닿음으로 표시 (더 민감하게)
        
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
            
            glDrawArrays(GL_TRIANGLES, 0, obj.vertex_count)
            
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
        
        # 바닥 접촉 면 하이라이트 (Z=0 근처 면을 초록색으로)
        if is_selected:
            self._draw_floor_contact_faces(obj)
        
        glPopMatrix()
    
    def _draw_floor_contact_faces(self, obj: SceneObject):
        """바닥(Z=0)을 뚫은 면을 초록색으로 하이라이트 (대형 메쉬 최적화 및 회전 대응)"""
        if obj.mesh is None or obj.mesh.faces is None:
            return
        
        faces = obj.mesh.faces
        vertices = obj.mesh.vertices
        
        # 회전 행렬 계산 (Z축 관통 확인용)
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler('xyz', obj.rotation, degrees=True).as_matrix()
        
        # 정점들의 월드 Z 좌표 근사계산: (R * (S*v))_z + Tz
        # 모든 정점을 변환하면 느리므로 샘플링
        total_faces = len(faces)
        
        # 매번 다른 샘플을 사용하여 움직일 때 누락 없이 보이게 함
        sample_size = 50000 if total_faces > 500000 else total_faces
        indices = np.random.choice(total_faces, sample_size, replace=False)
        
        sample_faces = faces[indices]
        # 각 면의 중심점 또는 정점 하나만 체크 (성능 상)
        v_indices = sample_faces[:, 0]
        v_points = vertices[v_indices] * obj.scale
        
        # 월드 Z 좌표 = R[2,0]*x + R[2,1]*y + R[2,2]*z + Tz
        world_z = (r[2, 0] * v_points[:, 0] + 
                   r[2, 1] * v_points[:, 1] + 
                   r[2, 2] * v_points[:, 2]) + obj.translation[2]
        
        penetrating_mask = world_z < 0
        penetrating_indices = indices[np.where(penetrating_mask)[0]]
        
        if len(penetrating_indices) == 0:
            return
        
        # 초록색 실칠 (Paint the mesh)
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-4.0, -4.0) # 메쉬보다 확실히 앞으로
        
        glColor4f(0.0, 1.0, 0.2, 0.8) # 더 불투명하고 선명한 초록
        
        glBegin(GL_TRIANGLES)
        for face_idx in penetrating_indices[:15000]: # 그리기 성능 제한
            f = faces[face_idx]
            for v_idx in f:
                glVertex3fv(vertices[v_idx]) # 현 매트릭스 스택(로컬) 기준
        glEnd()
        
        # 외곽선 살짝 추가
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glColor4f(0.0, 0.5, 0.1, 0.5)
        glBegin(GL_TRIANGLES)
        for face_idx in penetrating_indices[:5000]:
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
        
        # 월드 좌표에서 바운딩 박스 계산
        vertices = obj.mesh.vertices + obj.translation
        min_pt = vertices.min(axis=0)
        max_pt = vertices.max(axis=0)
        
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
            return
        
        # 1. 회전 행렬 계산
        rx, ry, rz = np.radians(obj.rotation)
        
        cos_x, sin_x = np.cos(rx), np.sin(rx)
        rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        
        cos_z, sin_z = np.cos(rz), np.sin(rz)
        rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        
        rotation_matrix = rot_z @ rot_y @ rot_x
        
        # 2. 정점 변환 (S -> R -> T 순서, 렌더링과 동일)
        # 스케일
        vertices = obj.mesh.vertices * obj.scale
        
        # 회전
        vertices = (rotation_matrix @ vertices.T).T
        
        # 이동 (월드 좌표에 적용)
        vertices = vertices + obj.translation
        
        # 3. 데이터 업데이트
        obj.mesh.vertices = vertices.astype(np.float32)
        
        # 법선 재계산
        obj.mesh.compute_normals()
        
        # 4. 모든 변환값 0으로 리셋 (이제 메쉬 정점 자체가 월드 좌표)
        obj.translation = np.array([0.0, 0.0, 0.0])
        obj.rotation = np.array([0.0, 0.0, 0.0])
        obj.scale = 1.0
        
        # 5. VBO 업데이트
        self.update_vbo(obj)
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
                        # 1. 스냅 검사 (첫 번째 점과 가까우면 확정)
                        if len(self.floor_picks) >= 3:
                            first_pt = self.floor_picks[0]
                            dist = np.linalg.norm(point - first_pt)
                            if dist < 0.15: # 스냅 거리 확대 (15cm)
                                self.floorAlignmentConfirmed.emit()
                                return
                                
                        self.floorPointPicked.emit(point)
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

            # 3. 객체 조작 (Shift/Ctrl + 드래그)
            if obj and (modifiers & Qt.KeyboardModifier.ShiftModifier or modifiers & Qt.KeyboardModifier.ControlModifier):
                 self.save_undo_state() # 변환 시작 전 상태 저장
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
            
        self.mouse_button = None
        self.active_gizmo_axis = None
        self.gizmo_drag_start = None
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
            
            # 1. 기즈모 드래그 (좌클릭 + 기즈모 드래그 시작됨)
            if self.gizmo_drag_start is not None and self.active_gizmo_axis and obj and self.mouse_button == Qt.MouseButton.LeftButton:
                angle_info = self._calculate_gizmo_angle(event.pos().x(), event.pos().y())
                if angle_info is not None:
                    current_angle = angle_info
                    delta_angle = np.degrees(current_angle - self.gizmo_drag_start)
                    
                    if self.active_gizmo_axis == 'X': obj.rotation[0] -= delta_angle
                    elif self.active_gizmo_axis == 'Y': obj.rotation[1] += delta_angle
                    elif self.active_gizmo_axis == 'Z': obj.rotation[2] -= delta_angle
                        
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
            
            # 3. 객체 직접 조작 (Ctrl+드래그 = 메쉬 이동)
            if (modifiers & Qt.KeyboardModifier.ControlModifier) and obj:
                az_rad = np.radians(self.camera.azimuth)
                move_speed = self.camera.distance * 0.002
                
                # 마우스 이동 반대 방향으로 메쉬 이동 (밀고 당기는 느낌)
                right_x = -np.sin(az_rad) * move_speed
                right_y = np.cos(az_rad) * move_speed
                
                obj.translation[0] += dx * right_x
                obj.translation[1] += dx * right_y
                obj.translation[2] -= dy * move_speed  # 위/아래 반전 (Z-up)
                
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
            
            elif (modifiers & Qt.KeyboardModifier.ShiftModifier) and obj and self.mouse_button == Qt.MouseButton.LeftButton:
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
        """바닥 정렬용 3점 시각화 (점 + 연결선)"""
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
            
            # 반투명 영역 면 표시 (3점 이상 시)
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
