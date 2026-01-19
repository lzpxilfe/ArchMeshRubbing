"""
Surface Visualizer Module
표면 시각화 - 텍스처 없는 메쉬용 탁본 효과 생성

깊이맵, 노멀맵, 곡률맵 등을 이용해 탁본과 유사한 이미지를 생성합니다.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import numpy as np
from PIL import Image
from scipy import ndimage

from .mesh_loader import MeshData
from .flattener import FlattenedMesh


@dataclass
class RubbingImage:
    """
    탁본 이미지 결과
    
    Attributes:
        image: 탁본 이미지 (numpy array)
        width_real: 실제 너비
        height_real: 실제 높이
        unit: 단위
        dpi: 해상도
    """
    image: np.ndarray
    width_real: float
    height_real: float
    unit: str = 'mm'
    dpi: int = 300
    
    @property
    def width_pixels(self) -> int:
        return self.image.shape[1]
    
    @property
    def height_pixels(self) -> int:
        return self.image.shape[0]
    
    @property
    def pixels_per_unit(self) -> float:
        """단위당 픽셀 수"""
        return self.width_pixels / self.width_real
    
    def to_pil_image(self) -> Image.Image:
        """PIL Image로 변환"""
        if self.image.dtype != np.uint8:
            img = self.image.astype(np.float64)
            img = (img - img.min()) / (img.max() - img.min() + 1e-10) * 255
            img = img.astype(np.uint8)
        else:
            img = self.image
        
        if len(img.shape) == 2:
            return Image.fromarray(img, mode='L')
        else:
            return Image.fromarray(img)
    
    def save(self, filepath: str, include_scale_bar: bool = True) -> None:
        """
        이미지 저장
        
        Args:
            filepath: 저장 경로
            include_scale_bar: 스케일 바 포함 여부
        """
        img = self.to_pil_image()
        
        if include_scale_bar:
            img = self._add_scale_bar(img)
        
        img.save(filepath, dpi=(self.dpi, self.dpi))
    
    def _add_scale_bar(self, img: Image.Image) -> Image.Image:
        """스케일 바 추가"""
        from PIL import ImageDraw, ImageFont
        
        # 이미지 복사
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # 스케일 바 크기 계산 (이미지 너비의 10-20%)
        target_bar_ratio = 0.15
        target_bar_pixels = int(img.width * target_bar_ratio)
        
        # 실제 크기에 맞는 깔끔한 숫자로 조정
        bar_real_size = target_bar_pixels / self.pixels_per_unit
        
        # 깔끔한 숫자로 반올림 (1, 2, 5, 10, 20, 50, 100...)
        nice_values = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        magnitude = 10 ** np.floor(np.log10(bar_real_size))
        normalized = bar_real_size / magnitude
        
        nice_normalized = min(nice_values, key=lambda x: abs(x - normalized))
        nice_bar_size = nice_normalized * magnitude
        
        # 실제 픽셀 크기
        bar_pixels = int(nice_bar_size * self.pixels_per_unit)
        
        # 스케일 바 위치 (오른쪽 하단)
        margin = 20
        bar_height = 10
        bar_x = img.width - margin - bar_pixels
        bar_y = img.height - margin - bar_height - 20
        
        # 배경
        draw.rectangle([bar_x - 5, bar_y - 5, 
                       bar_x + bar_pixels + 5, bar_y + bar_height + 25],
                      fill='white', outline='black')
        
        # 스케일 바
        draw.rectangle([bar_x, bar_y, bar_x + bar_pixels, bar_y + bar_height],
                      fill='black')
        
        # 텍스트
        label = f"{nice_bar_size:.0f} {self.unit}"
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = bar_x + (bar_pixels - text_width) // 2
        draw.text((text_x, bar_y + bar_height + 3), label, fill='black', font=font)
        
        return img


class SurfaceVisualizer:
    """
    표면 시각화 도구
    
    텍스처가 없는 메쉬에서 탁본과 유사한 이미지를 생성합니다.
    """
    
    def __init__(self, default_dpi: int = 300):
        """
        Args:
            default_dpi: 기본 출력 해상도
        """
        self.default_dpi = default_dpi
    
    def generate_rubbing(self, flattened: FlattenedMesh,
                         width_pixels: int = 2000,
                         style: str = 'traditional',
                         light_angle: float = 45.0) -> RubbingImage:
        """
        탁본 효과 이미지 생성
        
        Args:
            flattened: 평면화된 메쉬
            width_pixels: 출력 이미지 너비 (픽셀)
            style: 스타일 ('traditional', 'modern', 'relief')
            light_angle: 조명 각도 (도)
            
        Returns:
            RubbingImage: 탁본 이미지
        """
        # 이미지 크기 계산
        aspect_ratio = flattened.height / flattened.width
        height_pixels = int(width_pixels * aspect_ratio)
        
        # 깊이맵 생성
        depth_map = self._create_depth_map(flattened, width_pixels, height_pixels)
        
        # 스타일에 따른 렌더링
        if style == 'traditional':
            image = self._render_traditional_rubbing(depth_map, light_angle)
        elif style == 'modern':
            image = self._render_modern_rubbing(depth_map, light_angle)
        elif style == 'relief':
            image = self._render_relief(depth_map, light_angle)
        else:
            image = self._render_traditional_rubbing(depth_map, light_angle)
        
        return RubbingImage(
            image=image,
            width_real=flattened.width,
            height_real=flattened.height,
            unit=flattened.original_mesh.unit,
            dpi=self.default_dpi
        )
    
    def _create_depth_map(self, flattened: FlattenedMesh,
                          width: int, height: int) -> np.ndarray:
        """
        평면화된 메쉬에서 깊이맵 생성
        
        로컬 높이 변화를 깊이로 사용합니다.
        """
        # 정규화된 UV 좌표
        normalized = flattened.normalize()
        uv = normalized.uv
        
        # 원본 메쉬의 법선 벡터에서 높이 추정
        original = flattened.original_mesh
        original.compute_normals()
        
        # 법선의 Z 성분을 높이 변화로 사용
        heights = original.normals[:, 2] if original.normals is not None else np.zeros(len(uv))
        
        # 깊이 버퍼 초기화
        depth_buffer = np.zeros((height, width), dtype=np.float64)
        count_buffer = np.zeros((height, width), dtype=np.int32)
        
        # 각 삼각형 래스터화
        for face in flattened.faces:
            self._rasterize_triangle_with_values(
                uv[face], heights[face],
                width, height,
                depth_buffer, count_buffer
            )
        
        # 평균 계산
        mask = count_buffer > 0
        depth_buffer[mask] /= count_buffer[mask]
        
        # 빈 영역 채우기 (주변 값으로 보간)
        if not mask.all():
            depth_buffer = self._fill_holes(depth_buffer, mask)
        
        return depth_buffer
    
    def _rasterize_triangle_with_values(self, uv: np.ndarray, values: np.ndarray,
                                        width: int, height: int,
                                        buffer: np.ndarray, 
                                        count: np.ndarray) -> None:
        """값을 보간하며 삼각형 래스터화"""
        # 픽셀 좌표로 변환
        px = (uv[:, 0] * (width - 1)).astype(np.float64)
        py = ((1 - uv[:, 1]) * (height - 1)).astype(np.float64)  # Y 뒤집기
        
        # 바운딩 박스
        min_x = max(0, int(np.floor(px.min())))
        max_x = min(width - 1, int(np.ceil(px.max())))
        min_y = max(0, int(np.floor(py.min())))
        max_y = min(height - 1, int(np.ceil(py.max())))
        
        # 면적 계산
        v0 = np.array([px[0], py[0]])
        v1 = np.array([px[1], py[1]])
        v2 = np.array([px[2], py[2]])
        
        area = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
        if abs(area) < 1e-10:
            return
        
        # 삼각형 내부 픽셀 채우기
        for iy in range(min_y, max_y + 1):
            for ix in range(min_x, max_x + 1):
                p = np.array([ix + 0.5, iy + 0.5])
                
                # 무게중심 좌표
                w0 = ((v1[0] - p[0]) * (v2[1] - p[1]) - (v2[0] - p[0]) * (v1[1] - p[1])) / area
                w1 = ((v2[0] - p[0]) * (v0[1] - p[1]) - (v0[0] - p[0]) * (v2[1] - p[1])) / area
                w2 = 1 - w0 - w1
                
                # 삼각형 내부 체크
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # 값 보간
                    value = w0 * values[0] + w1 * values[1] + w2 * values[2]
                    buffer[iy, ix] += value
                    count[iy, ix] += 1
    
    def _fill_holes(self, image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """빈 영역을 주변 값으로 채움"""
        from scipy.ndimage import distance_transform_edt, binary_dilation
        
        result = image.copy()
        
        # 유효한 영역 확장
        for _ in range(10):
            # 현재 마스크 확장
            dilated = binary_dilation(valid_mask)
            
            # 새로 추가된 영역
            new_valid = dilated & ~valid_mask
            
            if not new_valid.any():
                break
            
            # 주변 값으로 채우기
            for axis in range(2):
                # 각 방향으로 이동
                for direction in [-1, 1]:
                    shifted = np.roll(result, direction, axis=axis)
                    shifted_valid = np.roll(valid_mask, direction, axis=axis)
                    
                    # 새 영역에 값 할당
                    update_mask = new_valid & shifted_valid
                    result[update_mask] = shifted[update_mask]
            
            valid_mask = dilated
        
        return result
    
    def _render_traditional_rubbing(self, depth_map: np.ndarray,
                                     light_angle: float) -> np.ndarray:
        """
        전통적인 탁본 스타일 렌더링
        
        검은 바탕에 흰색/회색 음영
        """
        # 그라디언트 계산 (기울기)
        grad_y, grad_x = np.gradient(depth_map)
        
        # 조명 방향
        light_rad = np.radians(light_angle)
        light_x = np.cos(light_rad)
        light_y = np.sin(light_rad)
        
        # 램버시안 셰이딩
        # 법선 벡터: (-grad_x, -grad_y, 1) 정규화
        normal_z = np.ones_like(depth_map)
        norm = np.sqrt(grad_x**2 + grad_y**2 + normal_z**2)
        
        nx = -grad_x / norm
        ny = -grad_y / norm
        nz = normal_z / norm
        
        # 조명과 법선의 내적
        intensity = light_x * nx + light_y * ny + 0.5 * nz
        intensity = np.clip(intensity, 0, 1)
        
        # 탁본 효과: 높은 곳이 밝게
        image = (intensity * 255).astype(np.uint8)
        
        return image
    
    def _render_modern_rubbing(self, depth_map: np.ndarray,
                                light_angle: float) -> np.ndarray:
        """
        현대적인 탁본 스타일 (고대비)
        """
        base = self._render_traditional_rubbing(depth_map, light_angle)
        
        # 대비 증가
        enhanced = base.astype(np.float64)
        enhanced = (enhanced - enhanced.mean()) * 1.5 + enhanced.mean()
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _render_relief(self, depth_map: np.ndarray,
                       light_angle: float) -> np.ndarray:
        """
        릴리프(양각) 효과 렌더링
        """
        # Sobel 필터로 엣지 강조
        grad_x = ndimage.sobel(depth_map, axis=1)
        grad_y = ndimage.sobel(depth_map, axis=0)
        
        # 조명 방향에 따른 음영
        light_rad = np.radians(light_angle)
        emboss = np.cos(light_rad) * grad_x + np.sin(light_rad) * grad_y
        
        # 정규화
        emboss = (emboss - emboss.min()) / (emboss.max() - emboss.min() + 1e-10)
        image = (emboss * 255).astype(np.uint8)
        
        return image
    
    def generate_depth_map(self, flattened: FlattenedMesh,
                           width_pixels: int = 2000) -> RubbingImage:
        """
        깊이맵 이미지 생성
        
        Args:
            flattened: 평면화된 메쉬
            width_pixels: 출력 이미지 너비
            
        Returns:
            RubbingImage: 깊이맵 이미지
        """
        aspect_ratio = flattened.height / flattened.width
        height_pixels = int(width_pixels * aspect_ratio)
        
        depth_map = self._create_depth_map(flattened, width_pixels, height_pixels)
        
        # 정규화 및 8비트 변환
        normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-10)
        image = (normalized * 255).astype(np.uint8)
        
        return RubbingImage(
            image=image,
            width_real=flattened.width,
            height_real=flattened.height,
            unit=flattened.original_mesh.unit,
            dpi=self.default_dpi
        )
    
    def generate_curvature_map(self, flattened: FlattenedMesh,
                               width_pixels: int = 2000) -> RubbingImage:
        """
        곡률맵 생성 (문양 강조)
        
        Args:
            flattened: 평면화된 메쉬
            width_pixels: 출력 이미지 너비
            
        Returns:
            RubbingImage: 곡률맵 이미지
        """
        aspect_ratio = flattened.height / flattened.width
        height_pixels = int(width_pixels * aspect_ratio)
        
        depth_map = self._create_depth_map(flattened, width_pixels, height_pixels)
        
        # 라플라시안으로 곡률 계산
        curvature = ndimage.laplace(depth_map)
        
        # 정규화
        curvature = np.abs(curvature)
        curvature = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-10)
        image = (curvature * 255).astype(np.uint8)
        
        return RubbingImage(
            image=image,
            width_real=flattened.width,
            height_real=flattened.height,
            unit=flattened.original_mesh.unit,
            dpi=self.default_dpi
        )


# 테스트용
if __name__ == '__main__':
    print("Surface Visualizer module loaded successfully")
    print("Use: visualizer = SurfaceVisualizer(); rubbing = visualizer.generate_rubbing(flattened)")
