"""
2D 실루엣 추출 및 SVG 내보내기 모듈 (v5 - 디버깅 완료)
메쉬를 6방향에서 투영하여 외곽선을 폴리라인으로 추출합니다.
"""
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


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
                           translation: np.ndarray = None,
                           rotation: np.ndarray = None,
                           scale: float = 1.0) -> tuple:
        """
        메쉬를 투영하여 2D 외곽선(폴리라인)을 추출합니다.
        """
        if view not in self.VIEWS:
            raise ValueError(f"Unknown view: {view}")
        
        view_config = self.VIEWS[view]
        ax0, ax1 = view_config['axes']
        
        # 정점 변환
        vertices = mesh.vertices.copy() * scale
        
        if rotation is not None and np.any(rotation != 0):
            from scipy.spatial.transform import Rotation as R
            r = R.from_euler('xyz', rotation, degrees=True)
            vertices = r.apply(vertices)
        
        if translation is not None:
            vertices = vertices + translation
        
        # 2D 투영 (실제 좌표)
        proj_2d = vertices[:, [ax0, ax1]]
        
        # 바운딩 박스 (실제 좌표)
        min_coords = proj_2d.min(axis=0)
        max_coords = proj_2d.max(axis=0)
        size = max_coords - min_coords
        
        # 여백 5%
        margin = size * 0.05
        min_coords -= margin
        max_coords += margin
        size = max_coords - min_coords
        
        # 그리드 크기 계산
        aspect = size[0] / size[1] if size[1] > 0 else 1.0
        if aspect >= 1:
            grid_w = self.resolution
            grid_h = max(1, int(self.resolution / aspect))
        else:
            grid_h = self.resolution
            grid_w = max(1, int(self.resolution * aspect))
        
        # 좌표 변환 함수: 실제 좌표 → 이미지 좌표
        # 이미지 좌표계: 왼쪽 위가 (0,0), X는 오른쪽, Y는 아래로 증가
        # 실제 좌표계: X는 오른쪽, Y는 위쪽으로 증가
        def real_to_img(pts):
            img_pts = np.zeros_like(pts)
            img_pts[:, 0] = (pts[:, 0] - min_coords[0]) / size[0] * (grid_w - 1)
            # Y축 반전 (실제 Y+가 위쪽, 이미지 Y+가 아래쪽)
            img_pts[:, 1] = (1 - (pts[:, 1] - min_coords[1]) / size[1]) * (grid_h - 1)
            return img_pts
        
        # 역함수: 이미지 좌표 → 실제 좌표
        def img_to_real(pts):
            real_pts = np.zeros_like(pts, dtype=float)
            real_pts[:, 0] = pts[:, 0] / (grid_w - 1) * size[0] + min_coords[0]
            real_pts[:, 1] = (1 - pts[:, 1] / (grid_h - 1)) * size[1] + min_coords[1]
            return real_pts
        
        # 메쉬 래스터화 (PIL 사용)
        img = Image.new('L', (grid_w, grid_h), 0)
        draw = ImageDraw.Draw(img)
        
        faces = mesh.faces
        n_faces = len(faces)
        
        # 대형 메쉬 샘플링
        if n_faces > self.MAX_FACES_FOR_RASTERIZE:
            sample_idx = np.random.choice(n_faces, self.MAX_FACES_FOR_RASTERIZE, replace=False)
            faces = faces[sample_idx]
        
        for face in faces:
            tri_2d = proj_2d[face]
            tri_img = real_to_img(tri_2d)
            tri_img = np.clip(tri_img, 0, [grid_w - 1, grid_h - 1]).astype(int)
            
            polygon = [(int(p[0]), int(p[1])) for p in tri_img]
            draw.polygon(polygon, fill=255, outline=255)
        
        occupancy = np.array(img)
        
        # 외곽선 추출
        contours = []
        if HAS_CV2:
            found_contours, _ = cv2.findContours(occupancy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in found_contours:
                if len(cnt) > 2:
                    img_pts = cnt.reshape(-1, 2).astype(float)
                    real_pts = img_to_real(img_pts)
                    contours.append(real_pts)
        else:
            # scipy fallback
            from scipy import ndimage
            edges = ndimage.sobel(occupancy.astype(float))
            edge_idx = np.column_stack(np.where(np.abs(edges) > 10))
            if len(edge_idx) > 0:
                # (y, x) -> (x, y)
                img_pts = edge_idx[:, ::-1].astype(float)
                real_pts = img_to_real(img_pts)
                contours = [real_pts]
        
        bounds = {
            'min': min_coords,
            'max': max_coords,
            'size': size,
            'grid_size': (grid_w, grid_h),
            'occupancy': occupancy,
            'real_to_img': real_to_img,
            'img_to_real': img_to_real
        }
        
        return contours, bounds
    
    def generate_mesh_with_grid_image(self, bounds: dict, occupancy: np.ndarray,
                                       grid_spacing: float = 1.0) -> Image.Image:
        """
        메쉬 실루엣 + 격자 오버레이 이미지 생성 (곱하기 블렌드)
        """
        grid_w, grid_h = bounds['grid_size']
        size = bounds['size']
        min_coords = bounds['min']
        
        # 1. 메쉬 실루엣 이미지 (연한 회색)
        mesh_img = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
        if occupancy is not None:
            mesh_mask = Image.fromarray(occupancy).convert('L')
            mesh_color = Image.new('RGB', (grid_w, grid_h), (180, 180, 180))
            mesh_img = Image.composite(mesh_color, mesh_img, mesh_mask)
        
        # 2. 격자 이미지 생성
        grid_img = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
        draw = ImageDraw.Draw(grid_img)
        line_color = (200, 200, 200)
        
        # 세로선 (X 방향 격자)
        x_start = np.ceil(min_coords[0] / grid_spacing) * grid_spacing
        x = x_start
        while x <= min_coords[0] + size[0]:
            # 실제 X → 이미지 X (그대로 매핑)
            px = int((x - min_coords[0]) / size[0] * (grid_w - 1))
            if 0 <= px < grid_w:
                draw.line([(px, 0), (px, grid_h - 1)], fill=line_color, width=1)
            x += grid_spacing
        
        # 가로선 (Y 방향 격자)
        y_start = np.ceil(min_coords[1] / grid_spacing) * grid_spacing
        y = y_start
        while y <= min_coords[1] + size[1]:
            # 실제 Y → 이미지 Y (Y축 반전)
            py = int((1 - (y - min_coords[1]) / size[1]) * (grid_h - 1))
            if 0 <= py < grid_h:
                draw.line([(0, py), (grid_w - 1, py)], fill=line_color, width=1)
            y += grid_spacing
        
        # 3. 곱하기 블렌드 (Multiply)
        mesh_arr = np.array(mesh_img, dtype=np.float32) / 255.0
        grid_arr = np.array(grid_img, dtype=np.float32) / 255.0
        result_arr = (mesh_arr * grid_arr * 255).astype(np.uint8)
        
        return Image.fromarray(result_arr)
    
    def export_svg(self, contours: list, bounds: dict,
                   background_image: Image.Image = None,
                   stroke_color: str = "#FF00FF",
                   stroke_width: float = 0.1,  # 0.1cm = 1mm
                   output_path: str = None) -> str:
        """
        실루엣을 SVG로 내보냅니다. (격자+메쉬 배경 + 폴리라인)
        """
        size = bounds['size']
        min_coords = bounds['min']
        
        svg_width = size[0]
        svg_height = size[1]
        
        svg_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" ',
            f'xmlns:xlink="http://www.w3.org/1999/xlink" ',
            f'width="{svg_width:.2f}cm" height="{svg_height:.2f}cm" ',
            f'viewBox="0 0 {svg_width:.4f} {svg_height:.4f}">',
            '<title>Mesh Profile Export</title>',
            '<desc>Scale: 1:1 (1 unit = 1 cm)</desc>',
        ]
        
        # 배경 이미지 (메쉬+격자)
        if background_image is not None:
            buffer = io.BytesIO()
            background_image.save(buffer, format='PNG', optimize=True)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            svg_parts.append(
                f'<image x="0" y="0" width="{svg_width:.4f}" height="{svg_height:.4f}" '
                f'preserveAspectRatio="none" '
                f'xlink:href="data:image/png;base64,{img_base64}" />'
            )
        
        # 외곽선 폴리라인
        if contours:
            svg_parts.append(f'<g id="contours" stroke="{stroke_color}" fill="none" stroke-width="{stroke_width}" stroke-linejoin="round">')
            for contour in contours:
                if len(contour) < 2:
                    continue
                # 실제 좌표 → SVG 좌표
                # SVG viewBox는 실제 단위(cm)와 일치
                # SVG Y축은 아래로 증가하므로 반전 필요
                svg_pts = contour.copy()
                svg_pts[:, 0] = svg_pts[:, 0] - min_coords[0]
                svg_pts[:, 1] = svg_height - (svg_pts[:, 1] - min_coords[1])
                
                points_str = " ".join([f"{p[0]:.3f},{p[1]:.3f}" for p in svg_pts])
                svg_parts.append(f'<polygon points="{points_str}" />')
            svg_parts.append('</g>')
        
        svg_parts.append('</svg>')
        
        svg_content = '\n'.join(svg_parts)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            return output_path
        
        return svg_content
    
    def export_profile(self, mesh, view: str, output_path: str,
                       translation: np.ndarray = None,
                       rotation: np.ndarray = None,
                       scale: float = 1.0,
                       grid_spacing: float = 1.0,
                       include_grid: bool = True) -> str:
        """
        메쉬의 2D 프로파일을 SVG로 내보내는 통합 메서드.
        """
        contours, bounds = self.extract_silhouette(
            mesh, view, translation, rotation, scale
        )
        
        background_image = None
        if include_grid:
            background_image = self.generate_mesh_with_grid_image(
                bounds, bounds['occupancy'], grid_spacing
            )
        
        return self.export_svg(
            contours, bounds, background_image,
            stroke_width=0.1,
            output_path=output_path
        )
