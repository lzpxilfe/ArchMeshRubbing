"""
메쉬 단면 슬라이싱 모듈
평면으로 메쉬를 자르고 단면 폴리라인을 추출합니다.
"""
import numpy as np
from typing import List, Tuple, Optional, Sequence
import trimesh


class MeshSlicer:
    """메쉬를 평면으로 자르는 슬라이서"""
    
    def __init__(self, mesh: trimesh.Trimesh):
        """
        Args:
            mesh: trimesh.Trimesh 객체
        """
        self.mesh = mesh
    
    def slice_at_z(self, z_height: float) -> List[np.ndarray]:
        """
        Z축 높이에서 수평면으로 메쉬를 자릅니다.
        
        Args:
            z_height: 슬라이스 높이 (Z 좌표)
            
        Returns:
            단면 폴리라인 리스트 (각 폴리라인은 Nx3 배열)
        """
        plane_origin = [0.0, 0.0, float(z_height)]
        plane_normal = [0.0, 0.0, 1.0]  # Z+ 방향

        return self.slice_with_plane(plane_origin, plane_normal)

    def slice_with_plane(self, origin: Sequence[float], normal: Sequence[float]) -> List[np.ndarray]:
        """
        임의의 평면으로 메쉬를 자릅니다.
        
        Args:
            origin: 평면 위의 한 점 [x, y, z]
            normal: 평면 법선 벡터 [nx, ny, nz]
            
        Returns:
            단면 폴리라인 리스트 (각 폴리라인은 Nx3 배열)
        """
        try:
            # trimesh.section()으로 단면 추출
            section = self.mesh.section(
                plane_origin=origin,
                plane_normal=normal
            )
            
            if section is None:
                return []
            
            # Path3D를 2D로 변환
            section_2d, to_3d_transform = section.to_planar()
            
            # 폴리라인 추출
            contours = []
            for entity in section_2d.entities:
                points_2d = section_2d.vertices[entity.points]
                # 3D로 다시 변환
                points_3d = trimesh.transformations.transform_points(
                    np.column_stack([points_2d, np.zeros(len(points_2d))]),
                    to_3d_transform
                )
                contours.append(points_3d)
            
            return contours
            
        except Exception:
            # print(f"Slice error: {e}") # 사용자 요청으로 에러 로그 숨김
            return []
    
    def get_z_range(self) -> Tuple[float, float]:
        """메쉬의 Z축 범위를 반환합니다."""
        z_min = self.mesh.vertices[:, 2].min()
        z_max = self.mesh.vertices[:, 2].max()
        return (z_min, z_max)
    
    def slice_multiple_z(self, z_values: List[float]) -> dict:
        """
        여러 Z 높이에서 슬라이스합니다.
        
        Args:
            z_values: Z 높이 리스트
            
        Returns:
            {z_height: contours} 딕셔너리
        """
        results = {}
        for z in z_values:
            results[z] = self.slice_at_z(z)
        return results
    
    def export_slice_svg(self, z_height: float, output_path: str,
                         stroke_color: str = "#FF0000",
                         stroke_width: float = 0.1) -> Optional[str]:
        """
        단면을 SVG로 내보냅니다.
        
        Args:
            z_height: 슬라이스 높이
            output_path: 저장 경로
            stroke_color: 선 색상
            stroke_width: 선 두께 (cm)
            
        Returns:
            저장된 파일 경로 또는 None
        """
        contours = self.slice_at_z(z_height)
        
        if not contours:
            return None
        
        # 바운딩 박스 계산 (XY 평면)
        all_points = np.vstack(contours)
        min_x, min_y = all_points[:, 0].min(), all_points[:, 1].min()
        max_x, max_y = all_points[:, 0].max(), all_points[:, 1].max()
        
        # 여백 5%
        margin_x = (max_x - min_x) * 0.05
        margin_y = (max_y - min_y) * 0.05
        min_x -= margin_x
        min_y -= margin_y
        max_x += margin_x
        max_y += margin_y
        
        width = max_x - min_x
        height = max_y - min_y
        
        # SVG 생성
        svg_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" ',
            f'width="{width:.2f}cm" height="{height:.2f}cm" ',
            f'viewBox="0 0 {width:.4f} {height:.4f}">',
            f'<title>Cross Section at Z={z_height:.2f}</title>',
            '<desc>Scale: 1:1 (1 unit = 1 cm)</desc>',
        ]
        
        # 격자 추가 (1cm 간격)
        svg_parts.append('<g id="grid" stroke="#CCCCCC" stroke-width="0.02">')
        for x in np.arange(np.ceil(min_x), max_x, 1.0):
            px = x - min_x
            svg_parts.append(f'<line x1="{px:.3f}" y1="0" x2="{px:.3f}" y2="{height:.3f}" />')
        for y in np.arange(np.ceil(min_y), max_y, 1.0):
            py = height - (y - min_y)
            svg_parts.append(f'<line x1="0" y1="{py:.3f}" x2="{width:.3f}" y2="{py:.3f}" />')
        svg_parts.append('</g>')
        
        # 단면 폴리라인
        svg_parts.append(f'<g id="section" stroke="{stroke_color}" fill="none" stroke-width="{stroke_width}">')
        for contour in contours:
            if len(contour) < 2:
                continue
            # SVG 좌표로 변환
            svg_pts = contour[:, :2].copy()
            svg_pts[:, 0] -= min_x
            svg_pts[:, 1] = height - (svg_pts[:, 1] - min_y)
            
            points_str = " ".join([f"{p[0]:.3f},{p[1]:.3f}" for p in svg_pts])
            svg_parts.append(f'<polyline points="{points_str}" fill="none" />')
        svg_parts.append('</g>')
        
        svg_parts.append('</svg>')
        
        svg_content = '\n'.join(svg_parts)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        return output_path
