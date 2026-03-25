"""
Recording-surface review sheet utilities.

This module builds user-facing review images that emphasize a continuous
recording surface rather than a triangle wireframe. The primary output is a
side-by-side sheet with:
- a rubbing-style continuous surface image
- a clean outline confirmation view
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .flattener import FlattenedMesh


@dataclass(frozen=True)
class RecordingSurfaceReviewOptions:
    dpi: int = 300
    width_pixels: int = 1600
    rubbing_preset: str = "자연(이미지)"
    rubbing_detail_scale: float = 1.0
    rubbing_smooth_sigma_extra: float = 0.0
    rubbing_texture_postprocess: str | None = None
    title: str = "기록면 검토 시트"
    panel_title_rubbing: str = "연속 탁본형 기록면"
    panel_title_outline: str = "외곽 확인"
    summary_lines: tuple[str, ...] = ()
    outer_margin: int = 28
    panel_gap: int = 24
    header_gap: int = 18
    panel_title_gap: int = 10
    footer_gap: int = 12
    background_color: tuple[int, int, int] = (255, 255, 255)
    text_color: tuple[int, int, int] = (33, 37, 41)
    accent_color: tuple[int, int, int] = (214, 69, 65)
    outline_color: tuple[int, int, int] = (24, 24, 27)
    annotation_background: tuple[int, int, int] = (255, 255, 255)
    annotation_outline: tuple[int, int, int] = (204, 214, 224)
    show_scale_bar: bool = True
    show_orientation_legend: bool = True
    orientation_label_u: str = "길이축"
    orientation_label_v: str = "단면"


@dataclass(frozen=True)
class RecordingSurfaceReview:
    combined_image: Image.Image
    rubbing_image: Image.Image
    outline_image: Image.Image


def build_recording_surface_summary_lines(
    flattened: FlattenedMesh,
    *,
    record_label: str = "",
    target_label: str = "",
    strategy_suffix: str = "",
    mode_label: str = "",
    tile_class_label: str = "",
    split_scheme_label: str = "",
    record_strategy_label: str = "",
    guide_count: int | None = None,
    mandrel_radius_world: float | None = None,
    extra_lines: Iterable[str] = (),
) -> tuple[str, ...]:
    unit = str(getattr(getattr(flattened, "original_mesh", None), "unit", "") or "unit")

    first_parts: list[str] = []
    if str(record_label or "").strip():
        first_parts.append(f"기록면: {str(record_label).strip()}")
    if str(target_label or "").strip():
        first_parts.append(f"대상: {str(target_label).strip()}{str(strategy_suffix or '')}")
    elif str(strategy_suffix or "").strip():
        first_parts.append(str(strategy_suffix).strip())

    detail_parts: list[str] = []
    if str(mode_label or "").strip():
        detail_parts.append(f"모드: {str(mode_label).strip()}")
    if str(tile_class_label or "").strip():
        detail_parts.append(f"유형: {str(tile_class_label).strip()}")
    if str(split_scheme_label or "").strip():
        detail_parts.append(f"분할: {str(split_scheme_label).strip()}")
    if str(record_strategy_label or "").strip():
        detail_parts.append(f"기록 방식: {str(record_strategy_label).strip()}")
    if guide_count is not None and int(guide_count) > 0:
        detail_parts.append(f"대표 단면 가이드 {int(guide_count)}개")
    if mandrel_radius_world is not None:
        try:
            radius = float(mandrel_radius_world)
        except Exception:
            radius = None
        if radius is not None and np.isfinite(radius) and radius > 0.0:
            detail_parts.append(f"와통 반경 {radius:.3f} {unit}")

    size_line = (
        f"크기: {float(flattened.width):.2f} x {float(flattened.height):.2f} {unit}"
        f" | 왜곡(평균/최대): {float(flattened.mean_distortion):.1%} / {float(flattened.max_distortion):.1%}"
    )

    lines: list[str] = []
    if first_parts:
        lines.append(" | ".join(first_parts))
    if detail_parts:
        lines.append(" | ".join(detail_parts))
    lines.append(size_line)

    for line in extra_lines or ():
        text = str(line or "").strip()
        if text:
            lines.append(text)
    return tuple(lines)


def _line_height(font: ImageFont.ImageFont) -> int:
    try:
        bbox = font.getbbox("Ag")
        return max(12, int(bbox[3] - bbox[1]))
    except Exception:
        return 14


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    return image.convert("RGB")


def _format_real_length(length: float, unit: str) -> str:
    try:
        value = float(length)
    except Exception:
        value = 0.0
    if not np.isfinite(value) or value <= 0.0:
        return f"0 {unit}"
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))} {unit}"
    if value >= 10.0:
        return f"{value:.1f} {unit}"
    return f"{value:.2f} {unit}"


def _nice_scale_length(max_length: float) -> float | None:
    try:
        value = float(max_length)
    except Exception:
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None

    exponent = math.floor(math.log10(value))
    base = 10.0 ** exponent
    for multiplier in (5.0, 2.0, 1.0):
        candidate = multiplier * base
        if candidate <= value:
            return float(candidate)
    return float(base / 2.0) if base > 1.0 else float(value)


def _record_view_label(flattened: FlattenedMesh) -> str:
    meta = dict(getattr(flattened, "meta", {}) or {})
    record_view = str(meta.get("section_record_view", "") or "").strip().lower()
    if record_view == "top":
        return "상면"
    if record_view == "bottom":
        return "하면"
    return ""


def _draw_scale_bar(
    draw: ImageDraw.ImageDraw,
    *,
    flattened: FlattenedMesh,
    image_size: tuple[int, int],
    font: ImageFont.ImageFont,
    background: tuple[int, int, int],
    outline: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> None:
    width_px, height_px = int(image_size[0]), int(image_size[1])
    if width_px <= 0 or height_px <= 0:
        return

    real_width = float(getattr(flattened, "width", 0.0) or 0.0)
    if not np.isfinite(real_width) or real_width <= 0.0:
        return

    max_bar_real = real_width * 0.22
    bar_real = _nice_scale_length(max_bar_real)
    if bar_real is None or bar_real <= 0.0:
        return

    bar_px = int(round((bar_real / real_width) * float(width_px - 1)))
    if bar_px < 24:
        return

    line_h = _line_height(font)
    margin = max(12, int(round(min(width_px, height_px) * 0.03)))
    tick_h = max(6, int(round(line_h * 0.8)))
    bar_y = height_px - margin - tick_h - line_h - 8
    bar_x = margin + 10
    label = _format_real_length(bar_real, str(getattr(flattened.original_mesh, "unit", "") or "unit"))

    box = [
        bar_x - 10,
        bar_y - 8,
        bar_x + bar_px + 78,
        bar_y + tick_h + line_h + 12,
    ]
    draw.rectangle(box, fill=background, outline=outline, width=1)
    draw.line([(bar_x, bar_y), (bar_x + bar_px, bar_y)], fill=text_color, width=3)
    draw.line([(bar_x, bar_y - tick_h), (bar_x, bar_y + tick_h)], fill=text_color, width=2)
    draw.line([(bar_x + bar_px, bar_y - tick_h), (bar_x + bar_px, bar_y + tick_h)], fill=text_color, width=2)
    draw.text((bar_x, bar_y + tick_h + 4), label, fill=text_color, font=font)


def _draw_orientation_legend(
    draw: ImageDraw.ImageDraw,
    *,
    flattened: FlattenedMesh,
    image_size: tuple[int, int],
    options: RecordingSurfaceReviewOptions,
    font: ImageFont.ImageFont,
) -> None:
    width_px, height_px = int(image_size[0]), int(image_size[1])
    if width_px <= 0 or height_px <= 0:
        return

    line_h = _line_height(font)
    margin = max(12, int(round(min(width_px, height_px) * 0.03)))
    record_view = _record_view_label(flattened)

    lines = []
    if record_view:
        lines.append(f"기록면: {record_view}")
    lines.append(f"u -> {str(options.orientation_label_u or '길이축')}")
    lines.append(f"v ^ {str(options.orientation_label_v or '단면')}")

    box_w = max(160, max(len(line) for line in lines) * 7 + 24)
    box_h = (line_h * len(lines)) + 18
    box = [width_px - margin - box_w, margin, width_px - margin, margin + box_h]
    draw.rectangle(box, fill=options.annotation_background, outline=options.annotation_outline, width=1)

    y = margin + 8
    for line in lines:
        draw.text((box[0] + 10, y), line, fill=options.text_color, font=font)
        y += line_h


def _annotate_panel_image(
    image: Image.Image,
    flattened: FlattenedMesh,
    *,
    options: RecordingSurfaceReviewOptions,
) -> Image.Image:
    annotated = _to_rgb(image)
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    if bool(getattr(options, "show_scale_bar", True)):
        _draw_scale_bar(
            draw,
            flattened=flattened,
            image_size=annotated.size,
            font=font,
            background=options.annotation_background,
            outline=options.annotation_outline,
            text_color=options.text_color,
        )
    if bool(getattr(options, "show_orientation_legend", True)):
        _draw_orientation_legend(
            draw,
            flattened=flattened,
            image_size=annotated.size,
            options=options,
            font=font,
        )
    return annotated


def _build_outline_image(
    flattened: FlattenedMesh,
    *,
    size: tuple[int, int],
    background: tuple[int, int, int],
    stroke_color: tuple[int, int, int],
) -> Image.Image:
    width_px = max(1, int(size[0]))
    height_px = max(1, int(size[1]))
    image = Image.new("RGB", (width_px, height_px), color=background)
    draw = ImageDraw.Draw(image)

    pixels = np.asarray(flattened.get_pixel_coordinates(width_px, height_px), dtype=np.int32)
    stroke_w = max(2, int(round(min(width_px, height_px) / 420.0)))

    loops = []
    try:
        loops = list(flattened.original_mesh.get_boundary_loops() or [])
    except Exception:
        loops = []

    drawn = False
    for loop in loops:
        try:
            idx = np.asarray(loop, dtype=np.int32).reshape(-1)
        except Exception:
            continue
        if idx.size < 2 or int(np.max(idx)) >= int(pixels.shape[0]) or int(np.min(idx)) < 0:
            continue
        coords = [(int(p[0]), int(p[1])) for p in pixels[idx]]
        if len(coords) < 2:
            continue
        draw.line(coords + [coords[0]], fill=stroke_color, width=stroke_w)
        drawn = True

    if not drawn:
        try:
            uv_norm = flattened.normalize().uv
            if uv_norm.ndim == 2 and uv_norm.shape[0] >= 3:
                pts = np.asarray(uv_norm[:, :2], dtype=np.float64)
                pts[:, 0] *= float(max(1, width_px - 1))
                pts[:, 1] *= float(max(1, height_px - 1))
                pts[:, 1] = float(max(1, height_px - 1)) - pts[:, 1]
                min_xy = np.min(pts, axis=0)
                max_xy = np.max(pts, axis=0)
                rect = [
                    (int(min_xy[0]), int(min_xy[1])),
                    (int(max_xy[0]), int(min_xy[1])),
                    (int(max_xy[0]), int(max_xy[1])),
                    (int(min_xy[0]), int(max_xy[1])),
                    (int(min_xy[0]), int(min_xy[1])),
                ]
                draw.line(rect, fill=stroke_color, width=stroke_w)
        except Exception:
            pass

    return image


def _overlay_outline(
    base_image: Image.Image,
    flattened: FlattenedMesh,
    *,
    stroke_color: tuple[int, int, int],
) -> Image.Image:
    image = _to_rgb(base_image)
    draw = ImageDraw.Draw(image)
    width_px, height_px = image.size
    pixels = np.asarray(flattened.get_pixel_coordinates(width_px, height_px), dtype=np.int32)
    stroke_w = max(2, int(round(min(width_px, height_px) / 460.0)))

    loops = []
    try:
        loops = list(flattened.original_mesh.get_boundary_loops() or [])
    except Exception:
        loops = []

    for loop in loops:
        try:
            idx = np.asarray(loop, dtype=np.int32).reshape(-1)
        except Exception:
            continue
        if idx.size < 2 or int(np.max(idx)) >= int(pixels.shape[0]) or int(np.min(idx)) < 0:
            continue
        coords = [(int(p[0]), int(p[1])) for p in pixels[idx]]
        if len(coords) < 2:
            continue
        draw.line(coords + [coords[0]], fill=stroke_color, width=stroke_w)
    return image


def render_recording_surface_review(
    flattened: FlattenedMesh,
    *,
    options: RecordingSurfaceReviewOptions | None = None,
    rubbing_image: Image.Image | None = None,
) -> RecordingSurfaceReview:
    options = options or RecordingSurfaceReviewOptions()

    if rubbing_image is None:
        from .surface_visualizer import SurfaceVisualizer

        visualizer = SurfaceVisualizer(default_dpi=int(options.dpi))
        rubbing = visualizer.generate_rubbing(
            flattened,
            width_pixels=int(options.width_pixels),
            preset=str(options.rubbing_preset),
            texture_detail_scale=float(getattr(options, "rubbing_detail_scale", 1.0) or 1.0),
            texture_smooth_sigma_extra=float(getattr(options, "rubbing_smooth_sigma_extra", 0.0) or 0.0),
            texture_postprocess_extra=getattr(options, "rubbing_texture_postprocess", None),
        )
        rubbing_image = rubbing.to_pil_image()

    rubbing_rgb = _to_rgb(rubbing_image)
    rubbing_overlay = _overlay_outline(
        rubbing_rgb,
        flattened,
        stroke_color=options.accent_color,
    )
    outline_image = _build_outline_image(
        flattened,
        size=rubbing_overlay.size,
        background=options.background_color,
        stroke_color=options.outline_color,
    )
    rubbing_overlay = _annotate_panel_image(rubbing_overlay, flattened, options=options)
    outline_image = _annotate_panel_image(outline_image, flattened, options=options)

    font = ImageFont.load_default()
    line_h = _line_height(font)
    summary_lines = tuple(str(line) for line in (options.summary_lines or ()) if str(line).strip())

    outer = int(max(0, options.outer_margin))
    gap = int(max(0, options.panel_gap))
    header_gap = int(max(0, options.header_gap))
    panel_title_gap = int(max(0, options.panel_title_gap))
    footer_gap = int(max(0, options.footer_gap))

    title_h = line_h
    summary_h = line_h * len(summary_lines) if summary_lines else 0
    panel_title_h = line_h
    panel_w = int(rubbing_overlay.size[0])
    panel_h = int(rubbing_overlay.size[1])

    total_w = (outer * 2) + (panel_w * 2) + gap
    total_h = (
        outer
        + title_h
        + (header_gap if summary_h > 0 else 0)
        + summary_h
        + header_gap
        + panel_title_h
        + panel_title_gap
        + panel_h
        + footer_gap
    )

    combined = Image.new("RGB", (max(1, total_w), max(1, total_h)), color=options.background_color)
    draw = ImageDraw.Draw(combined)

    y = outer
    draw.text((outer, y), str(options.title), fill=options.text_color, font=font)
    y += title_h

    if summary_lines:
        y += header_gap
        for line in summary_lines:
            draw.text((outer, y), line, fill=options.text_color, font=font)
            y += line_h

    y += header_gap
    left_x = outer
    right_x = outer + panel_w + gap

    draw.text((left_x, y), str(options.panel_title_rubbing), fill=options.text_color, font=font)
    draw.text((right_x, y), str(options.panel_title_outline), fill=options.text_color, font=font)

    y += panel_title_h + panel_title_gap
    combined.paste(rubbing_overlay, (left_x, y))
    combined.paste(outline_image, (right_x, y))

    return RecordingSurfaceReview(
        combined_image=combined,
        rubbing_image=rubbing_overlay,
        outline_image=outline_image,
    )
