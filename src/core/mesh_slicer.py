"""
메쉬 단면 슬라이싱 모듈
평면으로 메쉬를 자르고 단면 폴리라인을 추출합니다.
"""
import math
import numpy as np
from typing import List, Tuple, Optional, Sequence
import trimesh

from .unit_utils import normalize_unit as _normalize_unit
from .unit_utils import resolve_svg_unit as _resolve_svg_unit


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
    
    def slice_multiple_z(self, z_values: List[float]) -> dict[float, List[np.ndarray]]:
        """
        여러 Z 높이에서 슬라이스합니다.
        
        Args:
            z_values: Z 높이 리스트
            
        Returns:
            {z_height: contours} 딕셔너리
        """
        results: dict[float, List[np.ndarray]] = {}
        for z in z_values:
            results[z] = self.slice_at_z(z)
        return results
    
    def export_slice_svg(
        self,
        z_height: float,
        output_path: str,
        stroke_color: str = "#FF0000",
        stroke_width: float = 0.1,
        unit: Optional[str] = None,
        grid_spacing_cm: float = 1.0,
    ) -> Optional[str]:
        """
        단면을 SVG로 내보냅니다.
        
        Args:
            z_height: 슬라이스 높이
            output_path: 저장 경로
            stroke_color: 선 색상
            stroke_width: 선 두께 (cm)
            unit: SVG 출력 단위 ('mm' | 'cm' | 'm'). None이면 메쉬 단위를 따릅니다.
            grid_spacing_cm: 격자 간격 (cm)
            
        Returns:
            저장된 파일 경로 또는 None
        """
        contours = self.slice_at_z(z_height)
        
        if not contours:
            return None
        
        mesh_unit = None
        try:
            meta = getattr(self.mesh, "metadata", None)
            if isinstance(meta, dict):
                mesh_unit = meta.get("unit")
        except Exception:
            mesh_unit = None

        svg_unit, unit_scale = _resolve_svg_unit(mesh_unit, unit)
        mesh_u = _normalize_unit(mesh_unit)

        # 바운딩 박스 계산 (XY 평면)
        try:
            all_points = np.vstack([c for c in contours if len(c) >= 2])
        except Exception:
            return None
        if all_points.size == 0:
            return None
        min_x, min_y = all_points[:, 0].min(), all_points[:, 1].min()
        max_x, max_y = all_points[:, 0].max(), all_points[:, 1].max()
        
        # 여백 5%
        margin_x = (max_x - min_x) * 0.05
        margin_y = (max_y - min_y) * 0.05
        min_x -= margin_x
        min_y -= margin_y
        max_x += margin_x
        max_y += margin_y
        
        width = float(max_x - min_x)
        height = float(max_y - min_y)
        if not np.isfinite(width) or not np.isfinite(height):
            return None
        width = float(max(width, 1e-9))
        height = float(max(height, 1e-9))

        width_svg = width * float(unit_scale)
        height_svg = height * float(unit_scale)
        z_out = float(z_height) * float(unit_scale)

        def _cm_to_svg_units(value_cm: float) -> float:
            if svg_unit == "mm":
                return float(value_cm) * 10.0
            return float(value_cm)

        # SVG 생성
        svg_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" ',
            f'width="{width_svg:.2f}{svg_unit}" height="{height_svg:.2f}{svg_unit}" ',
            f'viewBox="0 0 {width_svg:.4f} {height_svg:.4f}">',
            f'<title>Cross Section at Z={z_out:.2f}{svg_unit}</title>',
            f'<desc>Scale: 1:1 (units: {svg_unit}, mesh unit: {mesh_u})</desc>',
        ]

        # 격자 추가 (1cm 간격)
        units_per_cm = 10.0
        if mesh_u == "cm":
            units_per_cm = 1.0
        elif mesh_u == "m":
            units_per_cm = 0.01

        step_mesh = float(grid_spacing_cm) * float(units_per_cm)
        if np.isfinite(step_mesh) and step_mesh > 1e-12:
            grid_sw = _cm_to_svg_units(0.02)
            svg_parts.append(f'<g id="grid" stroke="#CCCCCC" stroke-width="{grid_sw:.4f}">')
            start_x = math.ceil(float(min_x) / step_mesh) * step_mesh
            start_y = math.ceil(float(min_y) / step_mesh) * step_mesh
            for x in np.arange(start_x, max_x + 1e-12, step_mesh):
                px = (float(x) - float(min_x)) * float(unit_scale)
                svg_parts.append(
                    f'<line x1="{px:.3f}" y1="0" x2="{px:.3f}" y2="{height_svg:.3f}" />'
                )
            for y in np.arange(start_y, max_y + 1e-12, step_mesh):
                py = (float(max_y) - float(y)) * float(unit_scale)
                svg_parts.append(
                    f'<line x1="0" y1="{py:.3f}" x2="{width_svg:.3f}" y2="{py:.3f}" />'
                )
            svg_parts.append('</g>')

        # 단면 폴리라인
        section_sw = _cm_to_svg_units(float(stroke_width))
        svg_parts.append(
            f'<g id="section" stroke="{stroke_color}" fill="none" stroke-width="{section_sw:.4f}">'
        )
        for contour in contours:
            if len(contour) < 2:
                continue
            # SVG 좌표로 변환
            svg_pts = contour[:, :2].copy()
            svg_pts[:, 0] = (svg_pts[:, 0] - float(min_x)) * float(unit_scale)
            svg_pts[:, 1] = (float(max_y) - svg_pts[:, 1]) * float(unit_scale)

            points_str = " ".join([f"{p[0]:.3f},{p[1]:.3f}" for p in svg_pts])
            svg_parts.append(f'<polyline points="{points_str}" fill="none" />')
        svg_parts.append('</g>')
        
        svg_parts.append('</svg>')
        
        svg_content = '\n'.join(svg_parts)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        return output_path
