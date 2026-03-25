"""
Rubbing Sheet (Composite) SVG Exporter

한 번의 SVG 출력으로 다음 항목을 함께 내보내기 위한 모듈입니다.
- 수직(Top view) 외곽 실측(벡터)
- 단면(컷) 프로파일(벡터)
- 내/외면 탁본 이미지(전개 후 탁본 스타일 렌더)

기본 철학은 삼각형 조각을 보여주는 것이 아니라,
기록면을 연속 탁본 이미지와 실측 벡터 레이어로 내보내는 것입니다.
미구(ㄴ자 꺾임 등) 전개/출력은 후순위로 두고, 현재 버전은 Z-방향(face normal) 기준으로
내/외면을 자동 분리하여 전개합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import base64
import io
from typing import Any, Optional

import numpy as np
from PIL import Image

from .flattener import FlattenedMesh, flatten_with_method
from .mesh_loader import MeshData
from .profile_exporter import ProfileExporter
from .surface_visualizer import RubbingImage, SurfaceVisualizer
from .unit_utils import resolve_svg_unit as _resolve_unit
from .unit_utils import mm_to_svg_units as _mm_to_svg_units


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
    flatten_method: str = "arap"  # 'arap' | 'lscm' | 'area' | 'cylinder' | 'section' (UI strings also accepted)
    flatten_distortion: float = 0.5  # 0..1 (area→angle), used when flatten_method='area'
    flatten_initial_method: str = "lscm"  # 'lscm' | 'section'
    cylinder_axis: Any = "auto"  # 'auto' | 'x' | 'y' | 'z' | explicit 3D axis vector
    cylinder_radius: Optional[float] = None  # mesh/world units; if None, estimated from geometry
    section_guides: list[dict[str, Any]] | None = None
    section_record_view: str | None = None
    rubbing_style: str = "traditional"
    # Digital rubbing (image-based) options
    rubbing_height_mode: str = "normal_z"  # 'normal_z' | 'axis'
    rubbing_remove_curvature: bool = False
    rubbing_reference_sigma: float | None = None
    rubbing_relief_strength: float = 1.0
    rubbing_preset: str | None = None
    rubbing_image_mode: str = "mesh"
    rubbing_smooth_sigma: float = 0.0
    rubbing_detail_strength: float = 1.0
    rubbing_detail_sigma: float | None = None
    rubbing_texture_detail_scale: float = 1.0
    rubbing_texture_smooth_sigma_extra: float = 0.0
    rubbing_texture_postprocess: str | None = None

    include_labels: bool = True
    label_font_size_mm: float = 3.5
    label_gap_mm: float = 1.5
    single_surface_label: str | None = None


@dataclass(frozen=True)
class _RenderedSide:
    group_id: str
    label: str
    rubbing: RubbingImage


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
    def split_outer_inner(self, mesh: MeshData, *, threshold: float = 0.15) -> tuple[MeshData, MeshData]:
        """Public helper for auto split without manual face selection."""
        return self._split_outer_inner(mesh, threshold=float(threshold))

    def export(
        self,
        mesh: MeshData,
        output_path: str | Path,
        *,
        cut_lines_world: Optional[list[list[list[float]]]] = None,
        cut_profiles_world: Optional[list[list[list[float]]]] = None,
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

        rendered_sides: list[_RenderedSide] = []
        single_surface_label = str(options.single_surface_label or "").strip()
        if single_surface_label:
            single_side = self._render_side(
                mesh,
                group_id="surface_rubbing",
                label=single_surface_label,
                svg_unit=svg_unit,
                unit_scale=unit_scale,
                options=options,
                cut_lines_world=cut_lines_world,
            )
            if single_side is not None:
                rendered_sides.append(single_side)
        else:
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

            # 3) Flatten + rubbing images (skip empty sides gracefully)
            outer_side = self._render_side(
                outer_mesh,
                group_id="outer_rubbing",
                label="외면 탁본",
                svg_unit=svg_unit,
                unit_scale=unit_scale,
                options=options,
                cut_lines_world=cut_lines_world,
            )
            if outer_side is not None:
                rendered_sides.append(outer_side)
            inner_side = self._render_side(
                inner_mesh,
                group_id="inner_rubbing",
                label="내면 탁본",
                svg_unit=svg_unit,
                unit_scale=unit_scale,
                options=options,
                cut_lines_world=cut_lines_world,
            )
            if inner_side is not None:
                rendered_sides.append(inner_side)

        if not rendered_sides:
            raise ValueError("No non-empty surface is available for rubbing sheet export.")

        margin = _mm_to_svg_units(float(options.margin_mm), svg_unit)
        gap = _mm_to_svg_units(float(options.gap_mm), svg_unit)
        side_sizes = [
            (
                float(side.rubbing.width_real) * unit_scale,
                float(side.rubbing.height_real) * unit_scale,
            )
            for side in rendered_sides
        ]
        row_w = float(sum(w for w, _h in side_sizes))
        if len(side_sizes) > 1:
            row_w += gap * float(len(side_sizes) - 1)
        top_row_h = max((h for _w, h in side_sizes), default=1e-6)

        sheet_w = max(margin * 2 + row_w, margin * 2 + top_w)
        sheet_h = margin * 3 + top_row_h + top_h

        # Coordinates
        row_x = margin + max(0.0, (sheet_w - margin * 2 - row_w) * 0.5)
        row_y = margin

        top_x = margin + max(0.0, (sheet_w - margin * 2 - top_w) * 0.5)
        top_y = margin * 2 + top_row_h

        label_font = _mm_to_svg_units(float(options.label_font_size_mm), svg_unit)
        stroke_width = _mm_to_svg_units(float(options.stroke_width_mm), svg_unit)

        svg_parts: list[str] = [
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
            '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{sheet_w:.4f}{svg_unit}" height="{sheet_h:.4f}{svg_unit}" '
            f'viewBox="0 0 {sheet_w:.6f} {sheet_h:.6f}">',
            "<!-- Produced by ArchMeshRubbing (Composite Sheet SVG) -->",
        ]

        # Rubbing groups
        cursor_x = row_x
        for side, (side_w, side_h) in zip(rendered_sides, side_sizes):
            img_uri = _encode_png_data_uri(side.rubbing.to_pil_image())
            svg_parts.append(f'<g id="{side.group_id}" transform="translate({cursor_x:.6f},{row_y:.6f})">')
            svg_parts.append(
                f'<image x="0" y="0" width="{side_w:.6f}" height="{side_h:.6f}" '
                f'preserveAspectRatio="none" href="{img_uri}" xlink:href="{img_uri}" />'
            )
            if options.include_labels:
                svg_parts.append(
                    f'<text x="0" y="{-0.5 * margin:.6f}" font-size="{label_font:.6f}" '
                    f'fill="{options.stroke_color}">{side.label}</text>'
                )
            svg_parts.append("</g>")
            cursor_x += side_w + gap

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

        # Fallback 2: robust surface separator (topology + cylindrical cues).
        if outer_idx.size == 0 or inner_idx.size == 0:
            try:
                from .surface_separator import SurfaceSeparator

                sep = SurfaceSeparator()
                res = sep.auto_detect_surfaces(mesh, method="auto", return_submeshes=True)
                if (
                    res is not None
                    and res.outer_surface is not None
                    and res.inner_surface is not None
                    and res.outer_surface.n_faces > 0
                    and res.inner_surface.n_faces > 0
                ):
                    return res.outer_surface, res.inner_surface
            except Exception:
                pass

        # Final fallback: keep sets disjoint so we never duplicate the whole mesh twice.
        all_idx = np.arange(mesh.n_faces, dtype=np.int32)
        if outer_idx.size == 0 and inner_idx.size > 0:
            outer_idx = np.setdiff1d(all_idx, inner_idx, assume_unique=False).astype(np.int32, copy=False)
        if inner_idx.size == 0 and outer_idx.size > 0:
            inner_idx = np.setdiff1d(all_idx, outer_idx, assume_unique=False).astype(np.int32, copy=False)

        if outer_idx.size == 0:
            outer_idx = all_idx
        if inner_idx.size == 0:
            inner_idx = np.zeros((0,), dtype=np.int32)

        return mesh.extract_submesh(outer_idx), mesh.extract_submesh(inner_idx)

    def _mesh_has_faces(self, mesh: MeshData | None) -> bool:
        if mesh is None:
            return False
        try:
            return int(getattr(mesh, "n_faces", 0) or 0) > 0 and int(getattr(mesh, "n_vertices", 0) or 0) > 0
        except Exception:
            return False

    def _render_side(
        self,
        mesh: MeshData | None,
        *,
        group_id: str,
        label: str,
        svg_unit: str,
        unit_scale: float,
        options: SheetExportOptions,
        cut_lines_world: Optional[list[list[list[float]]]] = None,
    ) -> _RenderedSide | None:
        if not self._mesh_has_faces(mesh):
            return None
        _flattened, rubbing = self._flatten_and_rub(
            mesh,
            svg_unit=svg_unit,
            unit_scale=unit_scale,
            options=options,
            cut_lines_world=cut_lines_world,
        )
        return _RenderedSide(group_id=group_id, label=label, rubbing=rubbing)

    def _flatten_and_rub(
        self,
        mesh: MeshData,
        *,
        svg_unit: str,
        unit_scale: float,
        options: SheetExportOptions,
        cut_lines_world: Optional[list[list[list[float]]]] = None,
    ) -> tuple[FlattenedMesh, RubbingImage]:
        flattened = flatten_with_method(
            mesh,
            method=str(options.flatten_method),
            iterations=int(options.flatten_iterations),
            distortion=float(options.flatten_distortion),
            boundary_type="free",
            initial_method=str(getattr(options, "flatten_initial_method", "lscm")),
            cylinder_axis=getattr(options, "cylinder_axis", "auto"),
            cylinder_radius=options.cylinder_radius,
            cut_lines_world=cut_lines_world,
            section_guides=getattr(options, "section_guides", None),
            section_record_view=getattr(options, "section_record_view", None),
        )

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
            flattened,
            width_pixels=width_pixels,
            style=str(options.rubbing_style),
            height_mode=str(getattr(options, "rubbing_height_mode", "normal_z")),
            remove_curvature=bool(getattr(options, "rubbing_remove_curvature", False)),
            reference_sigma=getattr(options, "rubbing_reference_sigma", None),
            relief_strength=float(getattr(options, "rubbing_relief_strength", 1.0)),
            preset=getattr(options, "rubbing_preset", None),
            image_mode=str(getattr(options, "rubbing_image_mode", "mesh")),
            smooth_sigma=float(getattr(options, "rubbing_smooth_sigma", 0.0)),
            detail_strength=float(getattr(options, "rubbing_detail_strength", 1.0)),
            detail_sigma=getattr(options, "rubbing_detail_sigma", None),
            texture_detail_scale=float(getattr(options, "rubbing_texture_detail_scale", 1.0) or 1.0),
            texture_smooth_sigma_extra=float(getattr(options, "rubbing_texture_smooth_sigma_extra", 0.0) or 0.0),
            texture_postprocess_extra=getattr(options, "rubbing_texture_postprocess", None),
        )
        return flattened, rubbing

    def _build_top_view_group(
        self,
        mesh: MeshData,
        *,
        cut_lines_world: Optional[list[list[list[float]]]],
        cut_profiles_world: Optional[list[list[list[float]]]],
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
        finite = np.all(np.isfinite(all_pts), axis=1)
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
