"""
3D Viewport Widget with OpenGL
CloudCompare 스타일 카메라 조작이 가능한 3D 뷰포트
"""

import numpy as np
from typing import Optional, Tuple

from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QPainter, QColor
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from OpenGL.GL import *
from OpenGL.GLU import *


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
        
        # 줌 제한
        self.min_distance = 1.0
        self.max_distance = 1000.0
    
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
    
    def pan(self, delta_x: float, delta_y: float, sensitivity: float = 0.1):
        """카메라 이동 (Pan)"""
        # 카메라의 오른쪽/위쪽 방향 계산
        az_rad = np.radians(self.azimuth)
        
        right = np.array([np.cos(az_rad), 0, -np.sin(az_rad)])
        up = np.array([0, 1, 0])
        
        # Pan 적용
        pan_speed = self.distance * sensitivity * 0.01
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
    
    def fit_to_bounds(self, bounds: np.ndarray):
        """바운딩 박스에 맞춰 카메라 조정"""
        center = (bounds[0] + bounds[1]) / 2
        size = np.linalg.norm(bounds[1] - bounds[0])
        
        self.center = center
        self.pan_offset = np.array([0.0, 0.0, 0.0])
        self.distance = size * 1.5
        
    def apply(self):
        """OpenGL에 카메라 변환 적용"""
        pos = self.position
        target = self.look_at
        up = self.up_vector
        
        gluLookAt(
            pos[0], pos[1], pos[2],
            target[0], target[1], target[2],
            up[0], up[1], up[2]
        )


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
    selectionChanged = pyqtSignal(list)  # 선택 변경됨
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 카메라
        self.camera = TrackballCamera()
        
        # 마우스 상태
        self.last_mouse_pos = None
        self.mouse_button = None
        
        # 렌더링 설정
        self.grid_size = 100  # cm
        self.grid_spacing = 1  # cm (1cm 격자)
        # 아이콘 배경색과 유사한 미색 (Cream/Beige)
        self.background_color = (0.96, 0.96, 0.94, 1.0) # #F5F5F0
        
        # 메쉬 데이터
        self.mesh = None
        self.mesh_vbo = None
        self.mesh_color = (0.7, 0.7, 0.8)
        
        # 선택된 면
        self.selected_faces = set()
        
        # 변환
        self.mesh_translation = np.array([0.0, 0.0, 0.0])
        self.mesh_rotation = np.array([0.0, 0.0, 0.0])  # 도
        
        # VBO 데이터
        self.vbo_id = None
        self.vertex_count = 0
        self.using_vbo = False
        
        # 곡률 피팅 모드
        self.curvature_pick_mode = False
        self.picked_points = []  # 3D 점 리스트
        self.fitted_arc = None   # FittedArc 객체
        
        # UI 설정
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def initializeGL(self):
        """OpenGL 초기화"""
        glClearColor(*self.background_color)
        
        # 깊이 테스트
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        # 조명
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        
        # 광원 설정
        light_pos = [50.0, 100.0, 50.0, 0.0]
        light_ambient = [0.3, 0.3, 0.3, 1.0]
        light_diffuse = [0.8, 0.8, 0.8, 1.0]
        
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        
        # 폴리곤 모드
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # 안티앨리어싱
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    
    def resizeGL(self, width: int, height: int):
        """뷰포트 크기 변경"""
        glViewport(0, 0, width, height)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = width / height if height > 0 else 1.0
        gluPerspective(45.0, aspect, 0.1, 2000.0)
        
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """렌더링"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # 카메라 적용
        self.camera.apply()
        
        # 격자 그리기
        self.draw_grid()
        
        # 축 그리기
        self.draw_axes()
        
        # 메쉬 그리기
        if self.mesh is not None:
            self.draw_mesh()
        
        # 곡률 피팅 시각화
        self.draw_picked_points()
        self.draw_fitted_arc()
    
    def draw_grid(self):
        """가변 격자 바닥면 그리기"""
        glDisable(GL_LIGHTING)
        
        # 메쉬 크기에 따라 격자 범위 확장
        size = self.grid_size
        spacing = self.grid_spacing
        
        half_size = size / 2
        
        # 메인 격자
        glColor3f(0.8, 0.8, 0.8)
        glLineWidth(1.0)
        
        glBegin(GL_LINES)
        # 0을 중심으로 spacing 간격으로 그리기
        steps = int(size / spacing)
        for i in range(steps + 1):
            val = -half_size + i * spacing
            # X 방향 선
            glVertex3f(val, 0, -half_size)
            glVertex3f(val, 0, half_size)
            # Z 방향 선
            glVertex3f(-half_size, 0, val)
            glVertex3f(half_size, 0, val)
        glEnd()
        
        # 주요 격자 (10단위)
        major_spacing = spacing * 10
        glColor3f(0.7, 0.7, 0.7)
        glLineWidth(1.5)
        
        glBegin(GL_LINES)
        steps_major = int(size / major_spacing)
        for i in range(steps_major + 1):
            val = -half_size + i * major_spacing
            glVertex3f(val, 0, -half_size)
            glVertex3f(val, 0, half_size)
            glVertex3f(-half_size, 0, val)
            glVertex3f(half_size, 0, val)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_axes(self):
        """XYZ 축 그리기 (원점에서)"""
        glDisable(GL_LIGHTING)
        
        axis_length = 10.0  # 10cm
        
        glLineWidth(3.0)
        glBegin(GL_LINES)
        
        # X축 (빨강)
        glColor3f(1.0, 0.2, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(axis_length, 0, 0)
        
        # Y축 (초록)
        glColor3f(0.2, 1.0, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(0, axis_length, 0)
        
        # Z축 (파랑)
        glColor3f(0.2, 0.2, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, axis_length)
        
        glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def draw_mesh(self):
        """메쉬 렌더링"""
        if self.mesh is None:
            return
        
        glPushMatrix()
        
        # 변환 적용
        glTranslatef(*self.mesh_translation)
        glRotatef(self.mesh_rotation[0], 1, 0, 0)
        glRotatef(self.mesh_rotation[1], 0, 1, 0)
        glRotatef(self.mesh_rotation[2], 0, 0, 1)
        
        # 메쉬 색상
        glColor3f(*self.mesh_color)
        
        if self.using_vbo and self.vbo_id is not None:
            # VBO 방식 렌더링 (매우 빠름)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
            glVertexPointer(3, GL_FLOAT, 24, ctypes.c_void_p(0))
            glNormalPointer(GL_FLOAT, 24, ctypes.c_void_p(12))
            
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
            
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
        else:
            # Immediate Mode (느림, 대체용)
            vertices = self.mesh.vertices
            faces = self.mesh.faces
            normals = self.mesh.face_normals
            
            glBegin(GL_TRIANGLES)
            for i, face in enumerate(faces):
                if normals is not None:
                    glNormal3fv(normals[i])
                
                # 선택된 면 강조
                if i in self.selected_faces:
                    glColor3f(1.0, 0.5, 0.0)
                else:
                    glColor3f(*self.mesh_color)
                    
                for vi in face:
                    glVertex3fv(vertices[vi])
            glEnd()
        
        # 와이어프레임 오버레이 (선택 사항)
        # self.draw_wireframe()
        
        glPopMatrix()
    
    def draw_wireframe(self):
        """와이어프레임 오버레이"""
        if self.mesh is None:
            return
        
        glDisable(GL_LIGHTING)
        glColor3f(0.3, 0.3, 0.3)
        glLineWidth(1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        
        glBegin(GL_TRIANGLES)
        for face in faces:
            for vi in face:
                glVertex3fv(vertices[vi])
        glEnd()
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_LIGHTING)
    
    def load_mesh(self, mesh):
        """메쉬 로드 및 최적화"""
        self.mesh = mesh
        self.mesh.compute_normals()
        
        # 메쉬 중심을 원점으로
        center = mesh.centroid
        self.mesh_translation = -center
        
        # 카메라 및 격자 스케일 조정
        self.update_grid_scale()
        
        # VBO 데이터 생성
        self.update_vbo()
        
        self.meshLoaded.emit(mesh)
        self.update()

    def update_grid_scale(self):
        """메쉬 크기에 맞춰 격자 스케일 조정"""
        if self.mesh is None:
            return
            
        bounds = self.mesh.bounds
        extents = bounds[1] - bounds[0]
        max_dim = np.max(extents)
        
        # 적절한 격자 크기 계산 (메쉬의 2~3배)
        if max_dim < 10:  # 10cm 미만
            self.grid_spacing = 1.0  # 1cm
            self.grid_size = 20.0
        elif max_dim < 100:  # 1m 미만
            self.grid_spacing = 5.0  # 5cm
            self.grid_size = 150.0
        else:  # 1m 이상
            self.grid_spacing = 10.0  # 10cm
            self.grid_size = max_dim * 1.5
            
        # 카메라를 메쉬에 맞춤
        self.camera.distance = max_dim * 2
        self.camera.center = np.array([0.0, 0.0, 0.0])

    def update_vbo(self):
        """VBO 생성 및 데이터 전송"""
        if self.mesh is None:
            return
            
        try:
            # 면 정점 데이터 구성 (pos, normal)
            # trimesh의 faces를 이용해 정점을 나열 (삼각형 하나당 3개 정점)
            v_indices = self.mesh.faces.flatten()
            vertices = self.mesh.vertices[v_indices].astype(np.float32)
            
            # 면 법선을 각 정점에 할당 (평면 셰이딩 방식)
            # trimesh.face_normals는 면당 하나이므로 반복 필요
            normals = np.repeat(self.mesh.face_normals, 3, axis=0).astype(np.float32)
            
            # 데이터 인터리빙 (pos, normal, pos, normal...)
            # 각 정점당 (x,y,z, nx,ny,nz) = 6 floats
            data = np.hstack([vertices, normals]).flatten()
            self.vertex_count = len(vertices)
            
            # VBO 생성
            if self.vbo_id is None:
                self.vbo_id = glGenBuffers(1)
            
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            
            self.using_vbo = True
            
        except Exception as e:
            print(f"VBO creation failed: {e}")
            self.using_vbo = False
    
    def set_mesh_translation(self, x: float, y: float, z: float):
        """메쉬 이동"""
        self.mesh_translation = np.array([x, y, z])
        self.update()
    
    def set_mesh_rotation(self, rx: float, ry: float, rz: float):
        """메쉬 회전 (도)"""
        self.mesh_rotation = np.array([rx, ry, rz])
        self.update()
    
    def mousePressEvent(self, event: QMouseEvent):
        """마우스 버튼 눌림"""
        # Shift+클릭: 곡률 측정용 점 찍기
        if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier and 
            event.button() == Qt.MouseButton.LeftButton and
            self.curvature_pick_mode and self.mesh is not None):
            
            point = self.pick_point_on_mesh(event.pos().x(), event.pos().y())
            if point is not None:
                self.picked_points.append(point)
                self.update()
            return
        
        self.last_mouse_pos = event.pos()
        self.mouse_button = event.button()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """마우스 버튼 놓음"""
        self.last_mouse_pos = None
        self.mouse_button = None
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """마우스 이동 (드래그)"""
        if self.last_mouse_pos is None:
            return
        
        dx = event.pos().x() - self.last_mouse_pos.x()
        dy = event.pos().y() - self.last_mouse_pos.y()
        
        if self.mouse_button == Qt.MouseButton.LeftButton:
            # 좌클릭: 회전
            self.camera.rotate(dx, dy)
        elif self.mouse_button == Qt.MouseButton.RightButton:
            # 우클릭: 이동
            self.camera.pan(dx, dy)
        elif self.mouse_button == Qt.MouseButton.MiddleButton:
            # 가운데 버튼: 회전 (CloudCompare 스타일)
            self.camera.rotate(dx, dy)
        
        self.last_mouse_pos = event.pos()
        self.update()
    
    def wheelEvent(self, event: QWheelEvent):
        """마우스 휠 (줌)"""
        delta = event.angleDelta().y()
        self.camera.zoom(delta)
        self.update()
    
    def keyPressEvent(self, event):
        """키보드 입력"""
        key = event.key()
        
        # R: 카메라 리셋
        if key == Qt.Key.Key_R:
            self.camera.reset()
            self.update()
        
        # F: 메쉬에 맞춤
        elif key == Qt.Key.Key_F and self.mesh is not None:
            self.camera.fit_to_bounds(self.mesh.bounds)
            self.update()
        
        # 숫자 키: 뷰 프리셋 (6방향)
        elif key == Qt.Key.Key_1:  # 정면 (Front)
            self.camera.azimuth = 0
            self.camera.elevation = 0
            self.update()
        elif key == Qt.Key.Key_2:  # 후면 (Back)
            self.camera.azimuth = 180
            self.camera.elevation = 0
            self.update()
        elif key == Qt.Key.Key_3:  # 우측면 (Right)
            self.camera.azimuth = 90
            self.camera.elevation = 0
            self.update()
        elif key == Qt.Key.Key_4:  # 좌측면 (Left)
            self.camera.azimuth = -90
            self.camera.elevation = 0
            self.update()
        elif key == Qt.Key.Key_5:  # 상면 (Top)
            self.camera.azimuth = 0
            self.camera.elevation = 89
            self.update()
        elif key == Qt.Key.Key_6:  # 하면 (Bottom)
            self.camera.azimuth = 0
            self.camera.elevation = -89
            self.update()
    
    def pick_point_on_mesh(self, screen_x: int, screen_y: int):
        """
        화면 좌표를 메쉬 표면의 3D 좌표로 변환 (Ray Casting)
        
        Args:
            screen_x, screen_y: 화면 좌표
            
        Returns:
            3D 점 (numpy array) 또는 None
        """
        if self.mesh is None:
            return None
        
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
