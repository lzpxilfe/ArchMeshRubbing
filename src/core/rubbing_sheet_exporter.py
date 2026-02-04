"""
Rubbing Sheet (Composite) SVG Exporter

한 번의 SVG 출력으로 다음 항목을 함께 내보내기 위한 모듈입니다.
- 수직(Top view) 외곽 실측(벡터)
- 단면(컷) 프로파일(벡터)
- 내/외면 탁본 이미지(전개 후 탁본 스타일 렌더)

미구(ㄴ자 꺾임 등) 전개/출력은 후순위로 두고, 현재 버전은 Z-방향(face normal) 기준으로
내/외면을 자동 분리하여 전개합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import base64
import io
from typing import Optional

import numpy as np
from PIL import Image

from .flattener import ARAPFlattener, FlattenedMesh
from .mesh_loader import MeshData
from .profile_exporter import ProfileExporter
from .surface_visualizer import SurfaceVisualizer


@dataclass(frozen=True)
class SheetExportOptions:
    unit: Optional[str] = None  # 'mm' | 'cm' | 'm' (None이면 mesh.unit 사용)
    dpi: int = 300

    margin_mm: float = 8.0
    gap_mm: float = 6.0

    stroke_width_mm: float = 0.15
    stroke_color: str = "#111111"

    cut_line_colors: tuple[str, str] = ("#ff4040", "#2b8cff")
    cut_line_width_mm: float = 0.15

    section_color: str = "#111111"
    section_width_mm: float = 0.15

    normal_split_threshold: float = 0.15  # dot(Z) threshold for outer/inner split

    flatten_iterations: int = 30
    rubbing_style: str = "traditional"

    include_labels: bool = True
    label_font_size_mm: float = 3.5
    label_gap_mm: float = 1.5


def _resolve_unit(mesh_unit: str, requested: Optional[str]) -> tuple[str, float]:
    unit = (requested or mesh_unit or "mm").lower()

    if unit in {"mm", "millimeter", "millimeters"}:
        return "mm", 1.0
    if unit in {"cm", "centimeter", "centimeters"}:
        return "cm", 1.0
    if unit in {"m", "meter", "meters"}:
        # SVG에서 m는 불편하므로 cm로 변환
        return "cm", 100.0

    return "mm", 1.0


def _mm_to_svg_units(mm: float, svg_unit: str) -> float:
    if svg_unit == "mm":
        return float(mm)
    if svg_unit == "cm":
        return float(mm) / 10.0
    return float(mm)


def _encode_png_data_uri(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _polyline_to_path(points: np.ndarray, close: bool = False) -> str:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return ""
    if not np.isfinite(pts[:, :2]).all():
        return ""
    d_parts = [f"M {pts[0, 0]:.6f} {pts[0, 1]:.6f}"]
    for p in pts[1:]:
        d_parts.append(f"L {p[0]:.6f} {p[1]:.6f}")
    if close:
        d_parts.append("Z")
    return " ".join(d_parts)


def _is_closed(points: np.ndarray, tol: float = 1e-6) -> bool:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return False
    try:
        return float(np.linalg.norm(pts[0, :2] - pts[-1, :2])) <= float(tol)
    except Exception:
        return False


class RubbingSheetExporter:
    def export(
        self,
        mesh: MeshData,
        output_path: str | Path,
        *,
        cut_lines_world: Optional[list] = None,
        cut_profiles_world: Optional[list] = None,
        outer_face_indices: Optional[list[int] | np.ndarray] = None,
        inner_face_indices: Optional[list[int] | np.ndarray] = None,
        options: SheetExportOptions | None = None,
    ) -> str:
        options = options or SheetExportOptions()
        output_path = Path(output_path)

        svg_unit, unit_scale = _resolve_unit(mesh.unit, options.unit)

        # 1) Top-view measurement group (outline + cut guides)
        top_group, top_w, top_h = self._build_top_view_group(
            mesh,
            cut_lines_world=cut_lines_world,
            cut_profiles_world=cut_profiles_world,
            svg_unit=svg_unit,
            unit_scale=unit_scale,
            options=options,
        )

        # 2) Split mesh into outer/inner (prefer user-assigned face indices if provided)
        outer_mesh = None
        inner_mesh = None
        if outer_face_indices is not None:
            try:
                idx = np.asarray(outer_face_indices, dtype=np.int32).reshape(-1)
                if idx.size > 0:
                    outer_mesh = mesh.extract_submesh(idx)
            except Exception:
                outer_mesh = None
        if inner_face_indices is not None:
            try:
                idx = np.asarray(inner_face_indices, dtype=np.int32).reshape(-1)
                if idx.size > 0:
                    inner_mesh = mesh.extract_submesh(idx)
            except Exception:
                inner_mesh = None

        if outer_mesh is None or inner_mesh is None:
            auto_outer, auto_inner = self._split_outer_inner(
                mesh, threshold=float(options.normal_split_threshold)
            )
            if outer_mesh is None:
                outer_mesh = auto_outer
            if inner_mesh is None:
                inner_mesh = auto_inner

        # 3) Flatten + rubbing images
        outer_flat, outer_rub = self._flatten_and_rub(outer_mesh, svg_unit=svg_unit, unit_scale=unit_scale, options=options)
        inner_flat, inner_rub = self._flatten_and_rub(inner_mesh, svg_unit=svg_unit, unit_scale=unit_scale, options=options)

        outer_w = float(outer_rub.width_real) * unit_scale
        outer_h = float(outer_rub.height_real) * unit_scale
        inner_w = float(inner_rub.width_real) * unit_scale
        inner_h = float(inner_rub.height_real) * unit_scale

        margin = _mm_to_svg_units(float(options.margin_mm), svg_unit)
        gap = _mm_to_svg_units(float(options.gap_mm), svg_unit)
        top_row_h = max(outer_h, inner_h, 1e-6)

        sheet_w = max(margin * 2 + outer_w + gap + inner_w, margin * 2 + top_w)
        sheet_h = margin * 3 + top_row_h + top_h

        # Coordinates
        outer_x = margin
        outer_y = margin
        inner_x = margin + outer_w + gap
        inner_y = margin

        top_x = margin + max(0.0, (sheet_w - margin * 2 - top_w) * 0.5)
        top_y = margin * 2 + top_row_h

        label_font = _mm_to_svg_units(float(options.label_font_size_mm), svg_unit)
        stroke_width = _mm_to_svg_units(float(options.stroke_width_mm), svg_unit)

        outer_img_uri = _encode_png_data_uri(outer_rub.to_pil_image())
        inner_img_uri = _encode_png_data_uri(inner_rub.to_pil_image())

        svg_parts: list[str] = [
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
            '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{sheet_w:.4f}{svg_unit}" height="{sheet_h:.4f}{svg_unit}" '
            f'viewBox="0 0 {sheet_w:.6f} {sheet_h:.6f}">',
            "<!-- Produced by ArchMeshRubbing (Composite Sheet SVG) -->",
        ]

        # Outer rubbing
        svg_parts.append(f'<g id="outer_rubbing" transform="translate({outer_x:.6f},{outer_y:.6f})">')
        svg_parts.append(
            f'<image x="0" y="0" width="{outer_w:.6f}" height="{outer_h:.6f}" '
            f'preserveAspectRatio="none" href="{outer_img_uri}" xlink:href="{outer_img_uri}" />'
        )
        if options.include_labels:
            svg_parts.append(
                f'<text x="0" y="{-0.5 * margin:.6f}" font-size="{label_font:.6f}" '
                f'fill="{options.stroke_color}">외면 탁본</text>'
            )
        svg_parts.append("</g>")

        # Inner rubbing
        svg_parts.append(f'<g id="inner_rubbing" transform="translate({inner_x:.6f},{inner_y:.6f})">')
        svg_parts.append(
            f'<image x="0" y="0" width="{inner_w:.6f}" height="{inner_h:.6f}" '
            f'preserveAspectRatio="none" href="{inner_img_uri}" xlink:href="{inner_img_uri}" />'
        )
        if options.include_labels:
            svg_parts.append(
                f'<text x="0" y="{-0.5 * margin:.6f}" font-size="{label_font:.6f}" '
                f'fill="{options.stroke_color}">내면 탁본</text>'
            )
        svg_parts.append("</g>")

        # Top-view measurement
        svg_parts.append(f'<g id="top_measurement" transform="translate({top_x:.6f},{top_y:.6f})">')
        if options.include_labels:
            svg_parts.append(
                f'<text x="0" y="{-0.5 * margin:.6f}" font-size="{label_font:.6f}" '
                f'fill="{options.stroke_color}">수직 외곽/단면 실측</text>'
            )
        svg_parts.append(top_group)
        svg_parts.append("</g>")

        # Optional: simple border
        svg_parts.append(
            f'<rect x="0" y="0" width="{sheet_w:.6f}" height="{sheet_h:.6f}" '
            f'fill="none" stroke="{options.stroke_color}" stroke-width="{stroke_width:.6f}" />'
        )

        svg_parts.append("</svg>")

        output_path.write_text("\n".join(svg_parts), encoding="utf-8")
        return str(output_path)

    def _split_outer_inner(self, mesh: MeshData, *, threshold: float) -> tuple[MeshData, MeshData]:
        mesh.compute_normals(compute_vertex_normals=False, force=True)
        face_normals = mesh.face_normals
        if face_normals is None or face_normals.size == 0 or mesh.faces.size == 0:
            return mesh, mesh

        fn = np.asarray(face_normals, dtype=np.float64)
        if fn.ndim != 2 or fn.shape[1] < 3:
            return mesh, mesh

        dots = fn[:, 2]
        thr = float(abs(threshold))

        outer_idx = np.where(dots >= thr)[0].astype(np.int32)
        inner_idx = np.where(dots <= -thr)[0].astype(np.int32)

        # Fallback: sign split
        if outer_idx.size == 0 or inner_idx.size == 0:
            outer_idx = np.where(dots >= 0.0)[0].astype(np.int32)
            inner_idx = np.where(dots < 0.0)[0].astype(np.int32)

        if outer_idx.size == 0:
            outer_idx = np.arange(mesh.n_faces, dtype=np.int32)
        if inner_idx.size == 0:
            inner_idx = np.arange(mesh.n_faces, dtype=np.int32)

        return mesh.extract_submesh(outer_idx), mesh.extract_submesh(inner_idx)

    def _flatten_and_rub(
        self,
        mesh: MeshData,
        *,
        svg_unit: str,
        unit_scale: float,
        options: SheetExportOptions,
    ) -> tuple[FlattenedMesh, object]:
        flattener = ARAPFlattener(max_iterations=int(options.flatten_iterations))
        flattened = flattener.flatten(mesh, boundary_type="free", initial_method="lscm")

        dpi = int(options.dpi)
        width_real = float(flattened.width)
        unit = (mesh.unit or "mm").lower()
        if unit == "mm":
            width_in = width_real / 25.4
        elif unit == "cm":
            width_in = width_real / 2.54
        elif unit == "m":
            width_in = (width_real * 100.0) / 2.54
        else:
            width_in = width_real / 25.4

        width_pixels = max(800, int(width_in * dpi))
        width_pixels = min(width_pixels, 12000)

        visualizer = SurfaceVisualizer(default_dpi=dpi)
        rubbing = visualizer.generate_rubbing(
            flattened, width_pixels=width_pixels, style=str(options.rubbing_style)
        )
        return flattened, rubbing

    def _build_top_view_group(
        self,
        mesh: MeshData,
        *,
        cut_lines_world: Optional[list],
        cut_profiles_world: Optional[list],
        svg_unit: str,
        unit_scale: float,
        options: SheetExportOptions,
    ) -> tuple[str, float, float]:
        exporter = ProfileExporter(resolution=2048)
        contours, _bounds = exporter.extract_silhouette(mesh, view="top")

        pts_xy: list[np.ndarray] = []
        for c in contours or []:
            arr = np.asarray(c, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
                pts_xy.append(arr[:, :2])

        def add_world_polylines(polys):
            if not polys:
                return
            for line in polys:
                arr = np.asarray(line, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
                    pts_xy.append(arr[:, :2])

        add_world_polylines(cut_lines_world)
        add_world_polylines(cut_profiles_world)

        if not pts_xy:
            w = 1.0 * unit_scale
            h = 1.0 * unit_scale
            return '<g id="top_view"></g>', w, h

        all_pts = np.vstack(pts_xy)
        finite = np.isfinite(all_pts).all(axis=1)
        all_pts = all_pts[finite]
        if all_pts.size == 0:
            w = 1.0 * unit_scale
            h = 1.0 * unit_scale
            return '<g id="top_view"></g>', w, h

        min_x = float(all_pts[:, 0].min())
        max_x = float(all_pts[:, 0].max())
        min_y = float(all_pts[:, 1].min())
        max_y = float(all_pts[:, 1].max())

        pad = max(1.0, 0.02 * max(max_x - min_x, max_y - min_y))
        min_x -= pad
        max_x += pad
        min_y -= pad
        max_y += pad

        w = (max_x - min_x) * unit_scale
        h = (max_y - min_y) * unit_scale
        w = float(max(w, 1e-6))
        h = float(max(h, 1e-6))

        def to_svg_xy(points: np.ndarray) -> np.ndarray:
            pts = np.asarray(points, dtype=np.float64)
            xy = pts[:, :2].copy()
            xy[:, 0] = (xy[:, 0] - min_x) * unit_scale
            xy[:, 1] = (max_y - xy[:, 1]) * unit_scale
            return xy

        outline_sw = _mm_to_svg_units(float(options.stroke_width_mm), svg_unit)
        cut_sw = _mm_to_svg_units(float(options.cut_line_width_mm), svg_unit)
        sec_sw = _mm_to_svg_units(float(options.section_width_mm), svg_unit)

        parts: list[str] = ['<g id="top_view" fill="none" stroke-linejoin="round" stroke-linecap="round">']

        # Outline
        parts.append(
            f'<g id="outline" stroke="{options.stroke_color}" stroke-width="{outline_sw:.6f}">'
        )
        for c in contours or []:
            arr = np.asarray(c, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
                continue
            pts = to_svg_xy(arr)
            d = _polyline_to_path(pts, close=True)
            if d:
                parts.append(f'<path d="{d}" />')
        parts.append("</g>")

        # Cut lines
        if cut_lines_world:
            parts.append(f'<g id="cut_lines" stroke-width="{cut_sw:.6f}">')
            for i, line in enumerate(cut_lines_world):
                arr = np.asarray(line, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
                    continue
                pts = to_svg_xy(arr)
                d = _polyline_to_path(pts, close=False)
                if not d:
                    continue
                color = options.cut_line_colors[i % len(options.cut_line_colors)]
                parts.append(f'<path d="{d}" stroke="{color}" />')
            parts.append("</g>")

        # Section profiles (already laid out on floor in world XY)
        if cut_profiles_world:
            parts.append(f'<g id="sections" stroke="{options.section_color}" stroke-width="{sec_sw:.6f}">')
            for i, line in enumerate(cut_profiles_world):
                arr = np.asarray(line, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
                    continue
                pts = to_svg_xy(arr)
                d = _polyline_to_path(pts, close=_is_closed(pts))
                if d:
                    parts.append(f'<path d="{d}" />')
            parts.append("</g>")

        parts.append("</g>")

        return "\n".join(parts), w, h
