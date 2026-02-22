"""
Image-only cylindrical dewarp utility.

This module intentionally does not use mesh faces/UVs. It remaps a viewport image
as if the horizontal axis were an orthographic projection of a cylinder:
    x = sin(theta),  u = theta
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage


def _normalize_fill_value(fill_value: int | Iterable[int], channels: int) -> np.ndarray:
    if channels <= 1:
        try:
            val = int(fill_value)  # type: ignore[arg-type]
        except Exception:
            val = 255
        return np.asarray([float(np.clip(val, 0, 255))], dtype=np.float64)

    if isinstance(fill_value, Iterable) and not isinstance(fill_value, (str, bytes)):
        vals = [int(v) for v in list(fill_value)]
        if len(vals) < channels:
            vals = vals + [vals[-1] if vals else 255] * (channels - len(vals))
        vals = vals[:channels]
        return np.asarray([float(np.clip(v, 0, 255)) for v in vals], dtype=np.float64)

    try:
        val = int(fill_value)  # type: ignore[arg-type]
    except Exception:
        val = 255
    return np.asarray([float(np.clip(val, 0, 255))] * channels, dtype=np.float64)


def _pil_to_array(image: Image.Image) -> tuple[np.ndarray, str]:
    mode = str(getattr(image, "mode", "") or "").upper()
    if mode in {"RGBA", "LA"}:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float64)
        rgb = rgba[..., :3]
        a = rgba[..., 3:4] / 255.0
        # Composite on white background so transparent viewport edges stay clean.
        out = rgb * a + 255.0 * (1.0 - a)
        return out.astype(np.float64, copy=False), "RGB"
    if mode in {"L", "I;16", "I", "F"}:
        arr = np.asarray(image.convert("L"), dtype=np.float64)
        return arr, "L"
    arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    return arr, "RGB"


def _segment_foreground_mask_from_src(
    src: np.ndarray,
    *,
    white_threshold: float = 6.0,
    min_component_px: int = 256,
) -> np.ndarray:
    if src.ndim == 2:
        diff = np.abs(255.0 - np.asarray(src, dtype=np.float64))
    else:
        rgb = np.asarray(src[..., :3], dtype=np.float64)
        diff = np.max(np.abs(255.0 - rgb), axis=2)

    try:
        thr = float(white_threshold)
    except Exception:
        thr = 6.0
    if not np.isfinite(thr):
        thr = 6.0
    thr = float(np.clip(thr, 1.0, 64.0))

    mask = diff > thr
    if int(mask.sum()) <= 0:
        return mask

    # Keep largest connected component to ignore overlays/noise.
    try:
        labels, n_labels = ndimage.label(mask)
        if int(n_labels) > 1:
            counts = np.bincount(labels.reshape(-1))
            counts[0] = 0
            keep = int(np.argmax(counts))
            if keep > 0 and int(counts[keep]) >= int(max(1, min_component_px)):
                mask = labels == keep
    except Exception:
        pass

    try:
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    except Exception:
        pass

    return np.asarray(mask, dtype=bool)


def trace_foreground_x_range(
    image: Image.Image | np.ndarray,
    *,
    white_threshold: float = 6.0,
    min_component_px: int = 256,
    min_column_fill_ratio: float = 0.005,
    margin_px: int = 2,
) -> tuple[float, float] | None:
    """
    Trace horizontal foreground extent from a mostly-white viewport capture.

    This is a light-weight "SAM-like" auto tracing fallback without model weights:
    1) segment foreground from white background,
    2) keep largest connected component,
    3) estimate stable x-range from filled columns.
    """
    if isinstance(image, Image.Image):
        src, _mode = _pil_to_array(image)
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            src = arr.astype(np.float64, copy=False)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            src = arr[..., :3].astype(np.float64, copy=False)
        else:
            return None

    if src.ndim == 2:
        h, w = int(src.shape[0]), int(src.shape[1])
    else:
        h, w = int(src.shape[0]), int(src.shape[1])
    if h <= 0 or w <= 0:
        return None

    mask = _segment_foreground_mask_from_src(
        src,
        white_threshold=float(white_threshold),
        min_component_px=int(min_component_px),
    )
    if int(mask.sum()) <= 0:
        return None

    col_counts = np.asarray(mask, dtype=np.uint8).sum(axis=0)
    min_fill = max(1, int(np.ceil(float(h) * float(max(0.0, min_column_fill_ratio)))))
    idx = np.flatnonzero(col_counts >= min_fill)
    if idx.size < 2:
        idx = np.flatnonzero(col_counts > 0)
    if idx.size < 2:
        return None

    x0 = int(idx[0])
    x1 = int(idx[-1])
    try:
        pad = int(margin_px)
    except Exception:
        pad = 2
    pad = max(0, min(pad, max(0, w // 8)))
    x0 = max(0, x0 - pad)
    x1 = min(w - 1, x1 + pad)
    if x1 <= x0:
        return None
    return float(x0), float(x1)


def unwrap_cylindrical_view_image(
    image: Image.Image | np.ndarray,
    *,
    visible_angle_deg: float = 170.0,
    strength: float = 1.0,
    seam_shift_pct: float = 0.0,
    center_x: float | None = None,
    source_x_range: tuple[float, float] | None = None,
    output_width: int | None = None,
    output_height: int | None = None,
    fill_value: int | Iterable[int] = 255,
    interpolation_order: int = 1,
    chunk_rows: int = 128,
) -> Image.Image:
    """
    Cylindrical dewarp on a viewport image (image-only fast path).

    Args:
        image: PIL image or HxW/HxWxC numpy image.
        visible_angle_deg: Assumed visible cylinder arc in degrees (0 < angle < 180).
        strength: 0..1 blend between linear image and full cylindrical dewarp.
        seam_shift_pct: Circular seam shift in percent of output width (-50..50).
        center_x: Optional source image center x in pixel coordinates.
        source_x_range: Optional source x range (x_min, x_max) used as dewarp target.
        output_width: Output width (default: source width).
        output_height: Output height (default: source height).
        fill_value: Constant fill for out-of-range sampling.
        interpolation_order: ndimage interpolation order (0=nearest, 1=bilinear).
        chunk_rows: Row chunk size to keep memory bounded on large exports.

    Returns:
        Dewarped PIL.Image (`L` or `RGB`).
    """
    if isinstance(image, Image.Image):
        src, mode = _pil_to_array(image)
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            src = arr.astype(np.float64, copy=False)
            mode = "L"
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            src = arr[..., :3].astype(np.float64, copy=False)
            mode = "RGB"
        else:
            raise ValueError("image must be PIL.Image or numpy HxW/HxWxC array")

    if src.ndim == 2:
        h_src, w_src = int(src.shape[0]), int(src.shape[1])
        channels = 1
    else:
        h_src, w_src, channels = int(src.shape[0]), int(src.shape[1]), int(src.shape[2])
    if h_src <= 0 or w_src <= 0:
        raise ValueError("empty source image")

    out_w = int(output_width) if output_width is not None else int(w_src)
    out_h = int(output_height) if output_height is not None else int(h_src)
    out_w = max(1, out_w)
    out_h = max(1, out_h)

    try:
        s = float(strength)
    except Exception:
        s = 1.0
    if not np.isfinite(s):
        s = 1.0
    s = float(np.clip(s, 0.0, 1.0))

    try:
        angle = float(visible_angle_deg)
    except Exception:
        angle = 170.0
    if not np.isfinite(angle):
        angle = 170.0
    angle = float(np.clip(angle, 10.0, 179.9))
    half_angle = np.deg2rad(angle * 0.5)

    cx = float(w_src - 1) * 0.5
    half_span = float(max(1, w_src - 1)) * 0.5
    if source_x_range is not None:
        try:
            x0 = float(source_x_range[0])
            x1 = float(source_x_range[1])
            if np.isfinite(x0) and np.isfinite(x1):
                if x0 > x1:
                    x0, x1 = x1, x0
                x0 = float(np.clip(x0, 0.0, float(max(0, w_src - 1))))
                x1 = float(np.clip(x1, 0.0, float(max(0, w_src - 1))))
                span = float(x1 - x0)
                if span > 1e-6:
                    cx = 0.5 * (x0 + x1)
                    half_span = 0.5 * span
        except Exception:
            pass
    else:
        if center_x is not None:
            try:
                cx = float(center_x)
            except Exception:
                cx = float(w_src - 1) * 0.5
        if not np.isfinite(cx):
            cx = float(w_src - 1) * 0.5
        cx = float(np.clip(cx, 0.0, float(max(0, w_src - 1))))
        half_span = min(cx, float(max(0, w_src - 1)) - cx)
        if not np.isfinite(half_span) or half_span <= 1e-9:
            half_span = float(max(1, w_src - 1)) * 0.5

    if out_w == 1:
        u = np.asarray([0.0], dtype=np.float64)
    else:
        u = np.linspace(-1.0, 1.0, num=int(out_w), dtype=np.float64)

    # Linear mapping (no dewarp) and cylindrical mapping.
    x_linear = cx + u * half_span
    denom = float(np.sin(half_angle))
    if abs(denom) <= 1e-9:
        x_cyl = x_linear.copy()
    else:
        theta = u * half_angle
        x_cyl = cx + (np.sin(theta) / denom) * half_span
    x_map = (1.0 - s) * x_linear + s * x_cyl

    if out_h == 1:
        y_values = np.asarray([0.0], dtype=np.float64)
    else:
        y_values = np.linspace(0.0, float(max(0, h_src - 1)), num=int(out_h), dtype=np.float64)

    # Clamp interpolation order to a safe range for ndimage.map_coordinates.
    try:
        order = int(interpolation_order)
    except Exception:
        order = 1
    order = max(0, min(order, 3))

    fill = _normalize_fill_value(fill_value, channels)
    rows_per_chunk = max(1, int(chunk_rows))

    if channels == 1:
        out = np.empty((out_h, out_w), dtype=np.float64)
    else:
        out = np.empty((out_h, out_w, channels), dtype=np.float64)

    for y0 in range(0, out_h, rows_per_chunk):
        y1 = min(out_h, y0 + rows_per_chunk)
        yy = np.repeat(y_values[y0:y1, None], out_w, axis=1)
        xx = np.repeat(x_map[None, :], y1 - y0, axis=0)
        coords = np.vstack([yy.reshape(-1), xx.reshape(-1)])

        if channels == 1:
            sampled = ndimage.map_coordinates(
                np.asarray(src, dtype=np.float64),
                coords,
                order=order,
                mode="constant",
                cval=float(fill[0]),
            )
            out[y0:y1, :] = sampled.reshape(y1 - y0, out_w)
        else:
            for c in range(channels):
                sampled = ndimage.map_coordinates(
                    np.asarray(src[..., c], dtype=np.float64),
                    coords,
                    order=order,
                    mode="constant",
                    cval=float(fill[c]),
                )
                out[y0:y1, :, c] = sampled.reshape(y1 - y0, out_w)

    out = np.clip(out, 0.0, 255.0).astype(np.uint8, copy=False)
    try:
        seam = float(seam_shift_pct)
    except Exception:
        seam = 0.0
    if np.isfinite(seam):
        seam = float(np.clip(seam, -50.0, 50.0))
        if abs(seam) > 1e-9 and out_w > 1:
            shift_px = int(np.round((seam / 100.0) * float(out_w)))
            if shift_px != 0:
                out = np.roll(out, shift=int(shift_px), axis=1)

    if mode == "L":
        return Image.fromarray(out, mode="L")
    return Image.fromarray(out, mode="RGB")


def unwrap_cylindrical_band_image(
    image: Image.Image | np.ndarray,
    *,
    visible_angle_deg: float = 170.0,
    strength: float = 1.0,
    seam_shift_pct: float = 0.0,
    source_x_range: tuple[float, float] | None = None,
    output_width: int | None = None,
    output_height: int | None = None,
    fit_height_to_band: bool = True,
    white_threshold: float = 6.0,
    min_component_px: int = 256,
    fill_value: int | Iterable[int] = 255,
    interpolation_order: int = 1,
    chunk_rows: int = 128,
    boundary_smooth_sigma: float = 1.5,
) -> Image.Image:
    """
    Band-oriented unwrap for "tile-like" curved strips in current viewport image.

    It traces top/bottom silhouette boundaries and remaps the band into a straight
    rectangular image. Horizontal mapping still supports cylindrical dewarp.
    """
    if isinstance(image, Image.Image):
        src, mode = _pil_to_array(image)
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            src = arr.astype(np.float64, copy=False)
            mode = "L"
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            src = arr[..., :3].astype(np.float64, copy=False)
            mode = "RGB"
        else:
            raise ValueError("image must be PIL.Image or numpy HxW/HxWxC array")

    if src.ndim == 2:
        h_src, w_src = int(src.shape[0]), int(src.shape[1])
        channels = 1
    else:
        h_src, w_src, channels = int(src.shape[0]), int(src.shape[1]), int(src.shape[2])
    if h_src <= 0 or w_src <= 0:
        raise ValueError("empty source image")

    # Foreground silhouette for top/bottom boundaries.
    mask = _segment_foreground_mask_from_src(
        src,
        white_threshold=float(white_threshold),
        min_component_px=int(min_component_px),
    )
    if int(mask.sum()) <= 0:
        # Fallback to simple cylindrical mapping.
        return unwrap_cylindrical_view_image(
            image,
            visible_angle_deg=float(visible_angle_deg),
            strength=float(strength),
            seam_shift_pct=float(seam_shift_pct),
            source_x_range=source_x_range,
            output_width=output_width,
            output_height=output_height,
            fill_value=fill_value,
            interpolation_order=interpolation_order,
            chunk_rows=chunk_rows,
        )

    has = np.any(mask, axis=0)
    if not bool(np.any(has)):
        return unwrap_cylindrical_view_image(
            image,
            visible_angle_deg=float(visible_angle_deg),
            strength=float(strength),
            seam_shift_pct=float(seam_shift_pct),
            source_x_range=source_x_range,
            output_width=output_width,
            output_height=output_height,
            fill_value=fill_value,
            interpolation_order=interpolation_order,
            chunk_rows=chunk_rows,
        )

    top = np.full((w_src,), np.nan, dtype=np.float64)
    bottom = np.full((w_src,), np.nan, dtype=np.float64)
    top[has] = np.argmax(mask[:, has], axis=0).astype(np.float64)
    bottom[has] = (h_src - 1 - np.argmax(mask[::-1, has], axis=0)).astype(np.float64)

    valid_idx = np.flatnonzero(np.isfinite(top) & np.isfinite(bottom) & (bottom > top))
    if valid_idx.size < 2:
        return unwrap_cylindrical_view_image(
            image,
            visible_angle_deg=float(visible_angle_deg),
            strength=float(strength),
            seam_shift_pct=float(seam_shift_pct),
            source_x_range=source_x_range,
            output_width=output_width,
            output_height=output_height,
            fill_value=fill_value,
            interpolation_order=interpolation_order,
            chunk_rows=chunk_rows,
        )

    x_grid = np.arange(w_src, dtype=np.float64)
    top_filled = np.interp(x_grid, valid_idx.astype(np.float64), top[valid_idx])
    bottom_filled = np.interp(x_grid, valid_idx.astype(np.float64), bottom[valid_idx])

    try:
        sigma = float(boundary_smooth_sigma)
    except Exception:
        sigma = 1.5
    if not np.isfinite(sigma):
        sigma = 1.5
    sigma = float(np.clip(sigma, 0.0, 12.0))
    if sigma > 0.01:
        top_filled = ndimage.gaussian_filter1d(top_filled, sigma=sigma, mode="nearest")
        bottom_filled = ndimage.gaussian_filter1d(bottom_filled, sigma=sigma, mode="nearest")

    # Source x range for sampling span.
    x0 = float(valid_idx[0])
    x1 = float(valid_idx[-1])
    if source_x_range is not None:
        try:
            xa = float(source_x_range[0])
            xb = float(source_x_range[1])
            if np.isfinite(xa) and np.isfinite(xb):
                if xa > xb:
                    xa, xb = xb, xa
                xa = float(np.clip(xa, 0.0, float(max(0, w_src - 1))))
                xb = float(np.clip(xb, 0.0, float(max(0, w_src - 1))))
                if xb - xa >= 1.0:
                    x0, x1 = xa, xb
        except Exception:
            pass
    if x1 <= x0:
        x0, x1 = float(valid_idx[0]), float(valid_idx[-1])

    if output_width is not None:
        out_w = max(1, int(output_width))
    else:
        out_w = max(1, int(np.round(x1 - x0 + 1.0)))

    thickness = np.maximum(1.0, bottom_filled - top_filled)
    if output_height is not None and not bool(fit_height_to_band):
        out_h = max(1, int(output_height))
    else:
        # Robust height from median visible thickness in selected range.
        ix0 = int(np.clip(np.floor(x0), 0, max(0, w_src - 1)))
        ix1 = int(np.clip(np.ceil(x1), 0, max(0, w_src - 1)))
        if ix1 < ix0:
            ix0, ix1 = ix1, ix0
        local = thickness[ix0 : ix1 + 1] if ix1 >= ix0 else thickness
        h_med = float(np.median(local)) if local.size else float(np.median(thickness))
        out_h = max(64, int(np.round(h_med + 1.0)))
        out_h = min(out_h, max(64, int(h_src * 2)))

    try:
        s = float(strength)
    except Exception:
        s = 1.0
    if not np.isfinite(s):
        s = 1.0
    s = float(np.clip(s, 0.0, 1.0))

    try:
        angle = float(visible_angle_deg)
    except Exception:
        angle = 170.0
    if not np.isfinite(angle):
        angle = 170.0
    angle = float(np.clip(angle, 10.0, 179.9))
    half_angle = np.deg2rad(angle * 0.5)

    if out_w == 1:
        u = np.asarray([0.0], dtype=np.float64)
    else:
        u = np.linspace(-1.0, 1.0, num=int(out_w), dtype=np.float64)
    cx = 0.5 * (x0 + x1)
    half_span = max(1.0, 0.5 * (x1 - x0))
    x_linear = cx + u * half_span
    denom = float(np.sin(half_angle))
    if abs(denom) <= 1e-9:
        x_cyl = x_linear.copy()
    else:
        theta = u * half_angle
        x_cyl = cx + (np.sin(theta) / denom) * half_span
    x_map = (1.0 - s) * x_linear + s * x_cyl
    x_map = np.clip(x_map, 0.0, float(max(0, w_src - 1)))

    top_at = np.interp(x_map, x_grid, top_filled)
    bottom_at = np.interp(x_map, x_grid, bottom_filled)
    band_h = np.maximum(1.0, bottom_at - top_at)

    if out_h == 1:
        t_values = np.asarray([0.0], dtype=np.float64)
    else:
        t_values = np.linspace(0.0, 1.0, num=int(out_h), dtype=np.float64)
    y_map = top_at[None, :] + t_values[:, None] * band_h[None, :]

    try:
        order = int(interpolation_order)
    except Exception:
        order = 1
    order = max(0, min(order, 3))
    rows_per_chunk = max(1, int(chunk_rows))
    fill = _normalize_fill_value(fill_value, channels)

    if channels == 1:
        out = np.empty((out_h, out_w), dtype=np.float64)
    else:
        out = np.empty((out_h, out_w, channels), dtype=np.float64)

    x_rows = np.repeat(x_map[None, :], repeats=rows_per_chunk, axis=0)
    for y0 in range(0, out_h, rows_per_chunk):
        y1 = min(out_h, y0 + rows_per_chunk)
        yy = y_map[y0:y1, :]
        xx = x_rows[: y1 - y0, :]
        coords = np.vstack([yy.reshape(-1), xx.reshape(-1)])
        if channels == 1:
            sampled = ndimage.map_coordinates(
                np.asarray(src, dtype=np.float64),
                coords,
                order=order,
                mode="constant",
                cval=float(fill[0]),
            )
            out[y0:y1, :] = sampled.reshape(y1 - y0, out_w)
        else:
            for c in range(channels):
                sampled = ndimage.map_coordinates(
                    np.asarray(src[..., c], dtype=np.float64),
                    coords,
                    order=order,
                    mode="constant",
                    cval=float(fill[c]),
                )
                out[y0:y1, :, c] = sampled.reshape(y1 - y0, out_w)

    out = np.clip(out, 0.0, 255.0).astype(np.uint8, copy=False)
    try:
        seam = float(seam_shift_pct)
    except Exception:
        seam = 0.0
    if np.isfinite(seam):
        seam = float(np.clip(seam, -50.0, 50.0))
        if abs(seam) > 1e-9 and out_w > 1:
            shift_px = int(np.round((seam / 100.0) * float(out_w)))
            if shift_px != 0:
                out = np.roll(out, shift=int(shift_px), axis=1)

    if mode == "L":
        return Image.fromarray(out, mode="L")
    return Image.fromarray(out, mode="RGB")
