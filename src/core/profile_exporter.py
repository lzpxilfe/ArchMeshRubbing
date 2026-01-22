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
                           scale: float = 1.0,
                           opengl_matrices: tuple = None) -> tuple:
        """
        메쉬를 투영하여 단일 외곽선(폴리라인)을 추출합니다.
        opengl_matrices: (modelview, projection, viewport) - 제공 시 실제 렌더링 화면과 완벽히 일치시킴
        """
        if opengl_matrices:
            mv, proj, vp = opengl_matrices
            # 1. 월드 변환 적용
            vertices = mesh.vertices.copy() * scale
            if rotation is not None:
                from scipy.spatial.transform import Rotation as R
                r = R.from_euler('xyz', rotation, degrees=True)
                vertices = r.apply(vertices)
            if translation is not None:
                vertices += translation
            
            # 2. OpenGL 정사영 (Screen Space)
            # Clip Space = P * M * V
            v_homo = np.hstack([vertices, np.ones((len(vertices), 1))])
            v_clip = v_homo @ (mv @ proj) # (N, 4)
            # NDC
            v_ndc = v_clip[:, :3] / v_clip[:, 3:]
            # Screen
            proj_2d = np.zeros((len(vertices), 2))
            proj_2d[:, 0] = (v_ndc[:, 0] + 1) / 2 * vp[2]
            proj_2d[:, 1] = (v_ndc[:, 1] + 1) / 2 * vp[3]
            # OpenGL Y축 반전 보정 (이미지 좌표계로)
            proj_2d[:, 1] = vp[3] - proj_2d[:, 1]
            
            # 3. 바운딩 박스 (픽셀단위)
            min_px = proj_2d.min(axis=0)
            max_px = proj_2d.max(axis=0)
            
            # 4. 월드 스케일 계산 (SVG cm 단위를 위해)
            # 중심점에서 ±1cm 거리에 있는 점들을 투영해서 픽셀 거리 측정
            center_world = vertices.mean(axis=0)
            p1 = center_world.copy(); p1[0] += 1.0
            def project_pt(p):
                vh = np.append(p, 1.0)
                vc = vh @ (mv @ proj)
                vn = vc[:3] / vc[3]
                return np.array([(vn[0]+1)/2 * vp[2], vp[3] - (vn[1]+1)/2 * vp[3]])
            
            px_per_cm = np.linalg.norm(project_pt(center_world) - project_pt(p1))
            if px_per_cm < 1e-6: px_per_cm = 100.0 # Fallback
            
            # 5. 이미 렌더링된 이미지가 있으므로 래스터화는 외곽선 추출용으로만 사용
            grid_w, grid_h = vp[2], vp[3]
            occupancy = np.zeros((grid_h, grid_w), dtype=np.uint8)
            if HAS_CV2:
                img_v = proj_2d.astype(np.int32)
                for face in mesh.faces:
                    cv2.fillPoly(occupancy, [img_v[face]], 255)
            else:
                img = Image.new('L', (grid_w, grid_h), 0)
                draw = ImageDraw.Draw(img)
                for face in mesh.faces:
                    draw.polygon([(int(p[0]), int(p[1])) for p in proj_2d[face]], fill=255)
                occupancy = np.array(img)
                
            found_contours, _ = cv2.findContours(occupancy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            contours = []
            if found_contours:
                main_cnt = max(found_contours, key=cv2.contourArea)
                epsilon = 0.0002 * cv2.arcLength(main_cnt, True)
                approx = cv2.approxPolyDP(main_cnt, epsilon, True)
                # 픽셀 좌표 contours
                contours.append(approx.reshape(-1, 2).astype(float))
            
                # 6. 바운드 설정 (cm 단위 크기)
            bounds = {
                'min': np.array([0, 0]),
                'max': np.array([vp[2]/px_per_cm, vp[3]/px_per_cm]),
                'size': np.array([vp[2]/px_per_cm, vp[3]/px_per_cm]),
                'px_per_cm': px_per_cm,
                'is_pixels': True, 
                'vp_size': (vp[2], vp[3]),
                'matrices': (mv, proj, vp),
                'vertices_world': vertices # 그리드 범위 계산용
            }
            return contours, bounds

        # Fallback (Orthogonal-like simple projection)
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
        if view == 'back': proj_2d[:, 0] = -proj_2d[:, 0]
        if view == 'left': proj_2d[:, 0] = -proj_2d[:, 0]
        if view == 'bottom': proj_2d[:, 1] = -proj_2d[:, 1]
        
        # 바운딩 박스 (실제 좌표)
        min_coords = proj_2d.min(axis=0)
        max_coords = proj_2d.max(axis=0)
        size = max_coords - min_coords
        margin = size * 0.02
        min_coords -= margin
        max_coords += margin
        size = max_coords - min_coords
        
        grid_res = max(self.resolution, 2048)
        aspect = size[0] / size[1] if size[1] > 0 else 1.0
        if aspect >= 1:
            grid_w, grid_h = grid_res, max(1, int(grid_res / aspect))
        else:
            grid_h, grid_w = grid_res, max(1, int(grid_res * aspect))
        
        def real_to_img(pts):
            img_pts = np.zeros_like(pts)
            img_pts[:, 0] = (pts[:, 0] - min_coords[0]) / size[0] * (grid_w - 1)
            img_pts[:, 1] = (1 - (pts[:, 1] - min_coords[1]) / size[1]) * (grid_h - 1)
            return img_pts

        occupancy = np.zeros((grid_h, grid_w), dtype=np.uint8)
        if HAS_CV2:
            img_v = real_to_img(proj_2d).astype(np.int32)
            for face in mesh.faces: cv2.fillPoly(occupancy, [img_v[face]], 255)
        else:
            img = Image.new('L', (grid_w, grid_h), 0)
            draw = ImageDraw.Draw(img)
            img_v = real_to_img(proj_2d)
            for face in mesh.faces: 
                draw.polygon([(int(p[0]), int(p[1])) for p in img_v[face]], fill=255)
            occupancy = np.array(img)
        
        found_contours, _ = cv2.findContours(occupancy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        contours = []
        if found_contours:
            main_cnt = max(found_contours, key=cv2.contourArea)
            epsilon = 0.0002 * cv2.arcLength(main_cnt, True)
            approx = cv2.approxPolyDP(main_cnt, epsilon, True)
            img_pts = approx.reshape(-1, 2).astype(float)
            
            # 이미지 좌표 -> 실제 좌표 변환
            real_pts = np.zeros_like(img_pts)
            real_pts[:, 0] = img_pts[:, 0] / (grid_w - 1) * size[0] + min_coords[0]
            real_pts[:, 1] = (1 - img_pts[:, 1] / (grid_h - 1)) * size[1] + min_coords[1]
            contours.append(real_pts)
        
        bounds = {
            'min': min_coords, 'max': max_coords, 'size': size,
            'grid_size': (grid_w, grid_h), 'is_pixels': False
        }
        return contours, bounds
    
    def generate_composite_image(self, viewport_image: Image.Image, bounds: dict,
                                 grid_spacing: float = 1.0) -> Image.Image:
        """
        사용자가 실제 본 화면(viewport_image) + 격자 오버레이 이미지 생성 (Multiply)
        """
        img_w, img_h = viewport_image.size
        grid_img = Image.new('RGB', (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(grid_img)
        line_color = (200, 200, 200)

        if bounds.get('is_pixels') and 'matrices' in bounds:
            # 1. 행렬 기반 격자 투영
            mv, proj, vp = bounds['matrices']
            
            # 투류용 헬퍼
            def w_to_i(wx, wy, wz=0):
                vh = np.array([wx, wy, wz, 1.0])
                vc = vh @ (mv @ proj)
                if abs(vc[3]) < 1e-6: return None
                vn = vc[:3] / vc[3]
                return (vn[0]+1)/2 * img_w, img_h - (vn[1]+1)/2 * img_h

            # 가시 범위 계산 (메쉬 주변)
            v_world = bounds['vertices_world']
            w_min, w_max = v_world.min(axis=0), v_world.max(axis=0)
            pad = 20.0 # 20cm 여유
            
            # 세로선 (X축)
            x_start = np.floor((w_min[0] - pad) / grid_spacing) * grid_spacing
            x_end = np.ceil((w_max[0] + pad) / grid_spacing) * grid_spacing
            y_start, y_end = w_min[1] - pad, w_max[1] + pad
            
            x = x_start
            while x <= x_end:
                p1 = w_to_i(x, y_start); p2 = w_to_i(x, y_end)
                if p1 and p2: draw.line([p1, p2], fill=line_color, width=1)
                x += grid_spacing
                
            # 가로선 (Y축)
            y_start_grid = np.floor((w_min[1] - pad) / grid_spacing) * grid_spacing
            y_end_grid = np.ceil((w_max[1] + pad) / grid_spacing) * grid_spacing
            x_start_f, x_end_f = w_min[0] - pad, w_max[0] + pad
            
            y = y_start_grid
            while y <= y_end_grid:
                p1 = w_to_i(x_start_f, y); p2 = w_to_i(x_end_f, y)
                if p1 and p2: draw.line([p1, p2], fill=line_color, width=1)
                y += grid_spacing
        else:
            # 2. 고전적 평면 투영 (정사영 베이스)
            size = bounds['size']
            min_coords = bounds['min']
            
            # 세로선 (X)
            x = np.ceil(min_coords[0] / grid_spacing) * grid_spacing
            while x <= min_coords[0] + size[0]:
                px = int((x - min_coords[0]) / size[0] * (img_w - 1))
                if 0 <= px < img_w: draw.line([(px, 0), (px, img_h - 1)], fill=line_color, width=1)
                x += grid_spacing
            # 가로선 (Y)
            y = np.ceil(min_coords[1] / grid_spacing) * grid_spacing
            while y <= min_coords[1] + size[1]:
                py = int((1 - (y - min_coords[1]) / size[1]) * (img_h - 1))
                if 0 <= py < img_h: draw.line([(0, py), (img_w - 1, py)], fill=line_color, width=1)
                y += grid_spacing
            
        # 3. Multiply Blend
        mesh_arr = np.array(viewport_image.convert('RGB'), dtype=np.float32) / 255.0
        grid_arr = np.array(grid_img, dtype=np.float32) / 255.0
        result_arr = (mesh_arr * grid_arr * 255.0).astype(np.uint8)
        return Image.fromarray(result_arr)
    
    def export_svg(self, contours: list, bounds: dict,
                    background_image: Image.Image = None,
                    stroke_color: str = "#000000", # 기본 검정색
                    stroke_width: float = 0.05,    # 0.5mm
                    output_path: str = None) -> str:
        """
        최종 SVG 생성 (배경 이미지 + 단일 벡터 외곽선)
        """
        size = bounds['size']
        min_coords = bounds['min']
        
        svg_width = size[0]
        svg_height = size[1]
        
        svg_parts = [
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
            '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
            f'<svg xmlns="http://www.w3.org/2000/svg" ',
            f'xmlns:xlink="http://www.w3.org/1999/xlink" ',
            f'width="{svg_width:.2f}cm" height="{svg_height:.2f}cm" ',
            f'viewBox="0 0 {svg_width:.6f} {svg_height:.6f}">',
            f'<!-- Produced by ArchMeshRubbing - Resolution: {background_image.size if background_image else "Vector Only"} -->',
        ]
        
        # 1. 배경 이미지 삽입
        if background_image is not None:
            buffer = io.BytesIO()
            background_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            svg_parts.append(
                f'<image x="0" y="0" width="{svg_width:.6f}" height="{svg_height:.6f}" '
                f'preserveAspectRatio="none" '
                f'xlink:href="data:image/png;base64,{img_base64}" />'
            )
            
        # 2. 외곽선 벡터 삽입
        if contours:
            svg_parts.append(f'<g id="outline" stroke="{stroke_color}" fill="none" stroke-width="{stroke_width}">')
            for contour in contours:
                if len(contour) < 2: continue
                # 좌표 변환 (SVG Y축 반전)
                svg_pts = contour.copy()
                svg_pts[:, 0] -= min_coords[0]
                svg_pts[:, 1] = svg_height - (svg_pts[:, 1] - min_coords[1])
                
                points_str = " ".join([f"{p[0]:.6f},{p[1]:.6f}" for p in svg_pts])
                svg_parts.append(f'<polygon points="{points_str}" />')
            svg_parts.append('</g>')
            
        svg_parts.append('</svg>')
        
        content = '\n'.join(svg_parts)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return output_path
        return content
        
    def export_profile(self, mesh, view: str, output_path: str,
                       translation: np.ndarray = None,
                       rotation: np.ndarray = None,
                       scale: float = 1.0,
                       grid_spacing: float = 1.0,
                       include_grid: bool = True,
                       viewport_image: Image.Image = None) -> str:
        """
        메쉬의 2D 프로파일을 SVG로 내보내는 통합 메서드.
        """
        contours, bounds = self.extract_silhouette(
            mesh, view, translation, rotation, scale
        )
        
        background_image = None
        if include_grid:
            if viewport_image:
                # 캡처된 실제 이미지와 격자 합성
                background_image = self.generate_composite_image(
                    viewport_image, bounds, grid_spacing
                )
            else:
                # (Fallback) 실루엣과 격자 합성 - occupancy가 없으므로 빈 이미지 배경
                dummy_viewport = Image.new('RGB', bounds['grid_size'], (255, 255, 255))
                background_image = self.generate_composite_image(
                    dummy_viewport, bounds, grid_spacing
                )
        
        return self.export_svg(
            contours, bounds, background_image,
            stroke_width=0.05,
            output_path=output_path
        )
