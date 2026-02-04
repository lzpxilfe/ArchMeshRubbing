"""
Flattened Mesh → SVG exporter

평면화(Flatten)된 UV 결과를 실측 단위로 SVG로 내보냅니다.

기본은 경계(Outline)만 출력하며, 필요 시 메쉬 와이어프레임 출력도 지원합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .flattener import FlattenedMesh
from .unit_utils import resolve_svg_unit as _resolve_unit


@dataclass(frozen=True)
class SVGExportOptions:
    unit: Optional[str] = None  # 'mm' | 'cm' | 'm' (None이면 mesh.unit 사용)
    margin: float = 0.0
    include_grid: bool = False
    grid_spacing: float = 1.0
    include_outline: bool = True
    include_wireframe: bool = False
    stroke_color: str = "#000000"
    stroke_width: float = 0.05  # in SVG units (e.g. cm)
    grid_color: str = "#CCCCCC"
    grid_stroke_width: float = 0.02


class FlattenedSVGExporter:
    """FlattenedMesh를 실측 SVG로 내보내는 유틸리티."""

    def export(self, flattened: FlattenedMesh, output_path: str | Path,
               options: SVGExportOptions | None = None) -> str:
        options = options or SVGExportOptions()
        output_path = Path(output_path)

        svg_unit, unit_scale = _resolve_unit(flattened.original_mesh.unit, options.unit)

        # UV → 실측 좌표(원본 단위) → SVG 단위로 변환
        uv_real = flattened.uv.astype(np.float64) * float(flattened.scale)
        if uv_real.ndim != 2 or uv_real.shape[0] == 0:
            svg_parts: list[str] = [
                '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
                '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
                '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
                f'<svg xmlns="http://www.w3.org/2000/svg" width="1{svg_unit}" height="1{svg_unit}" viewBox="0 0 1 1">',
                '<!-- Produced by ArchMeshRubbing (Flattened SVG) -->',
                '<!-- Empty UV: nothing to export -->',
                '</svg>',
            ]
            output_path.write_text("\n".join(svg_parts), encoding="utf-8")
            return str(output_path)

        min_uv = uv_real.min(axis=0)
        max_uv = uv_real.max(axis=0)

        # margins (원본 단위)
        margin = float(options.margin)
        min_uv = min_uv - margin
        max_uv = max_uv + margin

        size = (max_uv - min_uv) * unit_scale
        width = float(max(size[0], 1e-6))
        height = float(max(size[1], 1e-6))

        # 좌표를 viewBox(0..width, 0..height)로 옮김. SVG는 y-down.
        def to_svg_xy(points_real: np.ndarray) -> np.ndarray:
            pts = (points_real - min_uv) * unit_scale
            pts[:, 1] = height - pts[:, 1]
            return pts

        svg_parts: list[str] = [
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
            '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
            '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width:.4f}{svg_unit}" height="{height:.4f}{svg_unit}" '
            f'viewBox="0 0 {width:.6f} {height:.6f}">',
            '<!-- Produced by ArchMeshRubbing (Flattened SVG) -->',
        ]

        if options.include_grid and options.grid_spacing > 0:
            svg_parts.append(
                f'<g id="grid" stroke="{options.grid_color}" '
                f'stroke-width="{options.grid_stroke_width}">'
            )
            spacing = float(options.grid_spacing)
            x = 0.0
            while x <= width + 1e-9:
                svg_parts.append(f'<line x1="{x:.6f}" y1="0" x2="{x:.6f}" y2="{height:.6f}" />')
                x += spacing
            y = 0.0
            while y <= height + 1e-9:
                svg_parts.append(f'<line x1="0" y1="{y:.6f}" x2="{width:.6f}" y2="{y:.6f}" />')
                y += spacing
            svg_parts.append('</g>')

        if options.include_wireframe:
            svg_parts.append(
                f'<g id="wireframe" stroke="{options.stroke_color}" fill="none" '
                f'stroke-width="{options.stroke_width}">'
            )
            faces = flattened.faces
            pts = uv_real.copy()
            for face in faces:
                try:
                    if int(np.max(face)) >= int(pts.shape[0]):
                        continue
                    tri = pts[face].copy()
                except Exception:
                    continue
                tri_svg = to_svg_xy(tri)
                p = " ".join([f"{xy[0]:.6f},{xy[1]:.6f}" for xy in tri_svg])
                svg_parts.append(f'<polygon points="{p}" fill="none" />')
            svg_parts.append('</g>')

        if options.include_outline:
            svg_parts.append(
                f'<g id="outline" stroke="{options.stroke_color}" fill="none" '
                f'stroke-width="{options.stroke_width}">'
            )
            loops = flattened.original_mesh.get_boundary_loops()
            if loops:
                for loop in loops:
                    try:
                        if int(np.max(loop)) >= int(uv_real.shape[0]):
                            continue
                        poly = uv_real[loop].copy()
                    except Exception:
                        continue
                    poly_svg = to_svg_xy(poly)
                    p = " ".join([f"{xy[0]:.6f},{xy[1]:.6f}" for xy in poly_svg])
                    svg_parts.append(f'<polyline points="{p}" fill="none" />')
            else:
                # 닫힌 메쉬 등 경계가 없으면 UV convex hull로 대체
                hull = self._convex_hull_2d(uv_real)
                if len(hull) >= 3:
                    poly_svg = to_svg_xy(uv_real[hull].copy())
                    p = " ".join([f"{xy[0]:.6f},{xy[1]:.6f}" for xy in poly_svg])
                    svg_parts.append(f'<polygon points="{p}" fill="none" />')

            svg_parts.append('</g>')

        svg_parts.append('</svg>')

        output_path.write_text("\n".join(svg_parts), encoding="utf-8")
        return str(output_path)

    def _convex_hull_2d(self, points: np.ndarray) -> np.ndarray:
        """
        Andrew monotone chain convex hull.
        Returns indices of hull vertices in CCW order (without repeated start).
        """
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 3:
            return np.arange(len(pts), dtype=np.int32)

        order = np.lexsort((pts[:, 1], pts[:, 0]))
        pts_sorted = pts[order]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: list[int] = []
        for idx, p in zip(order, pts_sorted):
            while len(lower) >= 2 and cross(pts[lower[-2]], pts[lower[-1]], p) <= 0:
                lower.pop()
            lower.append(int(idx))

        upper: list[int] = []
        for idx, p in zip(order[::-1], pts_sorted[::-1]):
            while len(upper) >= 2 and cross(pts[upper[-2]], pts[upper[-1]], p) <= 0:
                upper.pop()
            upper.append(int(idx))

        hull = lower[:-1] + upper[:-1]
        return np.asarray(hull, dtype=np.int32)
