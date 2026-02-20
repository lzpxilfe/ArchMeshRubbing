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


def unwrap_cylindrical_view_image(
    image: Image.Image | np.ndarray,
    *,
    visible_angle_deg: float = 170.0,
    strength: float = 1.0,
    center_x: float | None = None,
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
        center_x: Optional source image center x in pixel coordinates.
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

    if center_x is None:
        cx = float(w_src - 1) * 0.5
    else:
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
    if mode == "L":
        return Image.fromarray(out, mode="L")
    return Image.fromarray(out, mode="RGB")

