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
        self.background_color = (0.2, 0.2, 0.25, 1.0)
        
        # 메쉬 데이터
        self.mesh = None
        self.mesh_vbo = None
        self.mesh_color = (0.7, 0.7, 0.8)
        
        # 선택된 면
        self.selected_faces = set()
        
        # 변환
        self.mesh_translation = np.array([0.0, 0.0, 0.0])
        self.mesh_rotation = np.array([0.0, 0.0, 0.0])  # 도
        
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
    
    def draw_grid(self):
        """1cm 격자 바닥면 그리기"""
        glDisable(GL_LIGHTING)
        
        half_size = self.grid_size / 2
        
        # 메인 격자 (1cm)
        glColor3f(0.4, 0.4, 0.45)
        glLineWidth(1.0)
        
        glBegin(GL_LINES)
        for i in range(-int(half_size), int(half_size) + 1, self.grid_spacing):
            # X 방향 선
            glVertex3f(i, 0, -half_size)
            glVertex3f(i, 0, half_size)
            # Z 방향 선
            glVertex3f(-half_size, 0, i)
            glVertex3f(half_size, 0, i)
        glEnd()
        
        # 10cm 마다 굵은 선
        glColor3f(0.5, 0.5, 0.55)
        glLineWidth(2.0)
        
        glBegin(GL_LINES)
        for i in range(-int(half_size), int(half_size) + 1, 10):
            # X 방향 선
            glVertex3f(i, 0, -half_size)
            glVertex3f(i, 0, half_size)
            # Z 방향 선
            glVertex3f(-half_size, 0, i)
            glVertex3f(half_size, 0, i)
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
        
        # 삼각형 렌더링
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        normals = self.mesh.face_normals if self.mesh.face_normals is not None else None
        
        glBegin(GL_TRIANGLES)
        for i, face in enumerate(faces):
            # 면 법선
            if normals is not None:
                glNormal3fv(normals[i])
            
            # 선택된 면은 다른 색상
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
        """메쉬 로드"""
        self.mesh = mesh
        self.mesh.compute_normals()
        
        # 메쉬 중심을 원점으로
        center = mesh.centroid
        self.mesh_translation = -center
        
        # 카메라를 메쉬에 맞춤
        bounds = mesh.bounds
        size = np.linalg.norm(bounds[1] - bounds[0])
        self.camera.distance = size * 2
        self.camera.center = np.array([0.0, 0.0, 0.0])
        
        self.meshLoaded.emit(mesh)
        self.update()
    
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


# 테스트용 스탠드얼론 실행
if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    
    viewport = Viewport3D()
    viewport.setWindowTitle("3D Viewport Test")
    viewport.resize(800, 600)
    viewport.show()
    
    sys.exit(app.exec())
