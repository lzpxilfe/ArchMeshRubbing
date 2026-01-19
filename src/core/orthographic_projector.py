"""
Orthographic Projector Module
정사투영 평면도 생성 - 3D 메쉬를 정치 후 수직으로 내려다본 이미지 생성

기와를 펼치지 않고 위에서 내려다 본 평면도를 생성합니다.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import numpy as np
from PIL import Image

from .mesh_loader import MeshData


@dataclass
class ProjectionResult:
    """
    정사투영 결과
    
    Attributes:
        image: 투영된 이미지 (numpy array)
        depth_map: 깊이 맵 (선택)
        scale: 픽셀당 실제 크기 (mm/pixel 등)
        bounds: 투영 영역의 실제 경계 [[min_x, min_y], [max_x, max_y]]
        direction: 투영 방향
    """
    image: np.ndarray
    depth_map: Optional[np.ndarray] = None
    scale: float = 1.0
    bounds: Optional[np.ndarray] = None
    direction: str = 'top'
    unit: str = 'mm'
    
    @property
    def width_real(self) -> float:
        """실제 너비"""
        if self.bounds is not None:
            return self.bounds[1, 0] - self.bounds[0, 0]
        return self.image.shape[1] * self.scale
    
    @property
    def height_real(self) -> float:
        """실제 높이"""
        if self.bounds is not None:
            return self.bounds[1, 1] - self.bounds[0, 1]
        return self.image.shape[0] * self.scale
    
    def to_pil_image(self) -> Image.Image:
        """PIL Image로 변환"""
        if self.image.dtype != np.uint8:
            # 정규화
            img = self.image.astype(np.float64)
            img = (img - img.min()) / (img.max() - img.min() + 1e-10) * 255
            img = img.astype(np.uint8)
        else:
            img = self.image
        
        if len(img.shape) == 2:
            return Image.fromarray(img, mode='L')
        else:
            return Image.fromarray(img)
    
    def save(self, filepath: str, dpi: int = 300) -> None:
        """
        스케일 정보를 포함하여 이미지 저장
        
        Args:
            filepath: 저장 경로
            dpi: 해상도 (dots per inch)
        """
        img = self.to_pil_image()
        
        # DPI 정보 추가
        img.save(filepath, dpi=(dpi, dpi))


class OrthographicProjector:
    """
    정사투영 평면도 생성기
    
    3D 메쉬를 정치(正置)시킨 후 특정 방향에서 수직으로 투영합니다.
    """
    
    def __init__(self, resolution: int = 1024):
        """
        Args:
            resolution: 출력 이미지의 긴 변 해상도
        """
        self.resolution = resolution
    
    def align_mesh(self, mesh: MeshData, 
                   method: str = 'pca',
                   up_axis: str = 'z') -> MeshData:
        """
        메쉬 자동 정렬 (정치)
        
        Args:
            mesh: 입력 메쉬
            method: 정렬 방법 ('pca', 'bbox', 'none')
            up_axis: 상향 축 ('x', 'y', 'z')
            
        Returns:
            정렬된 메쉬
        """
        vertices = mesh.vertices.copy()
        
        # 중심을 원점으로
        centroid = vertices.mean(axis=0)
        vertices -= centroid
        
        if method == 'pca':
            # PCA로 주축 찾기
            cov = np.cov(vertices.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # 고유값 크기 순으로 정렬 (내림차순)
            order = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, order]
            
            # 회전 행렬 구성 (주축을 x, y, z에 정렬)
            R = eigenvectors.T
            
            # 오른손 좌표계 보장
            if np.linalg.det(R) < 0:
                R[2, :] *= -1
            
            vertices = vertices @ R.T
            
        elif method == 'bbox':
            # 바운딩 박스 기반 정렬 (간단히 중심만 이동)
            pass
        
        # 상향 축 조정
        if up_axis == 'x':
            # X를 위로 (Y, Z, X 순서)
            vertices = vertices[:, [1, 2, 0]]
        elif up_axis == 'y':
            # Y를 위로 (X, Z, Y 순서)
            vertices = vertices[:, [0, 2, 1]]
        # z는 기본값
        
        return MeshData(
            vertices=vertices,
            faces=mesh.faces.copy(),
            normals=None,  # 재계산 필요
            face_normals=None,
            uv_coords=mesh.uv_coords.copy() if mesh.uv_coords is not None else None,
            texture=mesh.texture,
            unit=mesh.unit,
            filepath=mesh.filepath
        )
    
    def project(self, mesh: MeshData,
                direction: Literal['top', 'bottom', 'front', 'back', 'left', 'right'] = 'top',
                resolution: Optional[int] = None,
                render_mode: str = 'depth') -> ProjectionResult:
        """
        정사투영 이미지 생성
        
        Args:
            mesh: 입력 메쉬 (정렬된 상태 권장)
            direction: 투영 방향
            resolution: 출력 해상도 (None이면 기본값 사용)
            render_mode: 렌더 모드 ('depth', 'normal', 'silhouette', 'ambient_occlusion')
            
        Returns:
            ProjectionResult: 투영 결과
        """
        resolution = resolution or self.resolution
        
        # 투영 축 결정
        axis_map = {
            'top': (2, 0, 1, False),      # Z축에서 아래로 (XY 평면)
            'bottom': (2, 0, 1, True),    # Z축에서 위로
            'front': (1, 0, 2, False),    # Y축에서 뒤로 (XZ 평면)
            'back': (1, 0, 2, True),      # Y축에서 앞으로
            'left': (0, 1, 2, True),      # X축에서 오른쪽으로 (YZ 평면)
            'right': (0, 1, 2, False),    # X축에서 왼쪽으로
        }
        
        depth_axis, x_axis, y_axis, flip_depth = axis_map[direction]
        
        vertices = mesh.vertices
        
        # 투영 좌표
        x_coords = vertices[:, x_axis]
        y_coords = vertices[:, y_axis]
        z_coords = vertices[:, depth_axis]
        
        if flip_depth:
            z_coords = -z_coords
        
        # 바운딩 박스
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        z_min, z_max = z_coords.min(), z_coords.max()
        
        # 이미지 크기 계산 (종횡비 유지)
        width = x_max - x_min
        height = y_max - y_min
        
        if width > height:
            img_width = resolution
            img_height = int(resolution * height / width)
        else:
            img_height = resolution
            img_width = int(resolution * width / height)
        
        img_height = max(img_height, 1)
        img_width = max(img_width, 1)
        
        # 스케일 (픽셀당 실제 크기)
        scale = max(width / img_width, height / img_height)
        
        # 깊이 버퍼 초기화
        depth_buffer = np.full((img_height, img_width), np.inf)
        
        # 래스터화
        for face in mesh.faces:
            self._rasterize_triangle(
                x_coords[face], y_coords[face], z_coords[face],
                x_min, x_max, y_min, y_max,
                img_width, img_height,
                depth_buffer
            )
        
        # 렌더 모드에 따른 이미지 생성
        if render_mode == 'depth':
            image = self._render_depth(depth_buffer, z_min, z_max)
        elif render_mode == 'silhouette':
            image = self._render_silhouette(depth_buffer)
        elif render_mode == 'normal':
            image = self._render_normal_shading(mesh, depth_buffer, direction)
        else:
            image = self._render_depth(depth_buffer, z_min, z_max)
        
        # 깊이맵 정규화
        depth_map = depth_buffer.copy()
        depth_map[np.isinf(depth_map)] = np.nan
        
        return ProjectionResult(
            image=image,
            depth_map=depth_map,
            scale=scale,
            bounds=np.array([[x_min, y_min], [x_max, y_max]]),
            direction=direction,
            unit=mesh.unit
        )
    
    def _rasterize_triangle(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                            x_min: float, x_max: float, 
                            y_min: float, y_max: float,
                            img_width: int, img_height: int,
                            depth_buffer: np.ndarray) -> None:
        """삼각형 래스터화 (Z-버퍼 알고리즘)"""
        # 화면 좌표로 변환
        sx = ((x - x_min) / (x_max - x_min) * (img_width - 1)).astype(np.float64)
        sy = ((y - y_min) / (y_max - y_min) * (img_height - 1)).astype(np.float64)
        sy = (img_height - 1) - sy  # Y축 뒤집기
        
        # 바운딩 박스
        min_x = max(0, int(np.floor(sx.min())))
        max_x = min(img_width - 1, int(np.ceil(sx.max())))
        min_y = max(0, int(np.floor(sy.min())))
        max_y = min(img_height - 1, int(np.ceil(sy.max())))
        
        # 엣지 함수 계수
        def edge_function(v0, v1, p):
            return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])
        
        v0 = np.array([sx[0], sy[0]])
        v1 = np.array([sx[1], sy[1]])
        v2 = np.array([sx[2], sy[2]])
        
        area = edge_function(v0, v1, v2)
        if abs(area) < 1e-10:
            return
        
        # 삼각형 내부 픽셀 채우기
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                p = np.array([px + 0.5, py + 0.5])
                
                w0 = edge_function(v1, v2, p)
                w1 = edge_function(v2, v0, p)
                w2 = edge_function(v0, v1, p)
                
                # 삼각형 내부 체크
                if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                    # 무게중심 좌표
                    w0 /= area
                    w1 /= area
                    w2 /= area
                    
                    # 깊이 보간
                    depth = w0 * z[0] + w1 * z[1] + w2 * z[2]
                    
                    # Z-버퍼 테스트
                    if depth < depth_buffer[py, px]:
                        depth_buffer[py, px] = depth
    
    def _render_depth(self, depth_buffer: np.ndarray, 
                      z_min: float, z_max: float) -> np.ndarray:
        """깊이맵 렌더링 (탁본 효과)"""
        image = depth_buffer.copy()
        
        # 배경 처리
        mask = np.isinf(image)
        
        # 깊이 정규화 (가까울수록 밝게)
        valid = ~mask
        if valid.any():
            img_min = image[valid].min()
            img_max = image[valid].max()
            
            if img_max - img_min > 1e-10:
                image[valid] = 1.0 - (image[valid] - img_min) / (img_max - img_min)
            else:
                image[valid] = 1.0
        
        image[mask] = 0  # 배경은 검정
        
        # 8비트로 변환
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def _render_silhouette(self, depth_buffer: np.ndarray) -> np.ndarray:
        """실루엣 렌더링"""
        image = np.zeros_like(depth_buffer, dtype=np.uint8)
        image[~np.isinf(depth_buffer)] = 255
        return image
    
    def _render_normal_shading(self, mesh: MeshData, 
                                depth_buffer: np.ndarray,
                                direction: str) -> np.ndarray:
        """노멀 기반 음영 렌더링"""
        # 간단한 깊이 기반 렌더링으로 대체 (노멀 맵은 추후 구현)
        return self._render_depth(depth_buffer, 
                                   mesh.vertices[:, 2].min(), 
                                   mesh.vertices[:, 2].max())
    
    def project_with_texture(self, mesh: MeshData,
                             direction: str = 'top',
                             resolution: Optional[int] = None) -> ProjectionResult:
        """
        텍스처가 있는 메쉬의 정사투영
        
        Args:
            mesh: 입력 메쉬 (텍스처 포함)
            direction: 투영 방향
            resolution: 출력 해상도
            
        Returns:
            ProjectionResult: 컬러 이미지 포함
        """
        if not mesh.has_texture:
            return self.project(mesh, direction, resolution)
        
        # 텍스처 투영은 추후 구현
        # 현재는 깊이맵으로 대체
        return self.project(mesh, direction, resolution)


# 테스트용
if __name__ == '__main__':
    print("Orthographic Projector module loaded successfully")
    print("Use: projector = OrthographicProjector(); result = projector.project(mesh)")
