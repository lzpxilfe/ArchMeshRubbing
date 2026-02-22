import numpy as np
from PIL import Image

from src.core.cylindrical_image_unwrapper import (
    trace_foreground_x_range,
    unwrap_cylindrical_band_image,
    unwrap_cylindrical_view_image,
)


def _gradient_rgb(width: int, height: int) -> Image.Image:
    x = np.linspace(0, 255, num=width, dtype=np.float64)
    row = np.stack([x, x, x], axis=1).astype(np.uint8)
    img = np.repeat(row[None, :, :], repeats=height, axis=0)
    return Image.fromarray(img, mode="RGB")


def test_unwrap_shape_and_mode_are_preserved():
    src = _gradient_rgb(128, 64)
    out = unwrap_cylindrical_view_image(src, visible_angle_deg=170.0, strength=1.0)
    assert out.mode == "RGB"
    assert out.size == src.size


def test_unwrap_strength_zero_is_identity_like():
    src = _gradient_rgb(121, 33)
    out = unwrap_cylindrical_view_image(src, visible_angle_deg=170.0, strength=0.0)
    a = np.asarray(src, dtype=np.int16)
    b = np.asarray(out, dtype=np.int16)
    # Bilinear sampling with identical map should be exact or off-by-1 around edges.
    assert int(np.max(np.abs(a - b))) <= 1


def test_unwrap_changes_horizontal_mapping_with_strength():
    src = _gradient_rgb(201, 21)
    out_linear = unwrap_cylindrical_view_image(src, visible_angle_deg=180.0, strength=0.0)
    out_cyl = unwrap_cylindrical_view_image(src, visible_angle_deg=180.0, strength=1.0)

    row_linear = np.asarray(out_linear, dtype=np.float64)[10, :, 0]
    row_cyl = np.asarray(out_cyl, dtype=np.float64)[10, :, 0]

    # Center should remain almost unchanged.
    center = len(row_linear) // 2
    assert abs(float(row_cyl[center]) - float(row_linear[center])) < 2.0

    # Cylindrical mapping compresses both sides toward the center.
    q1 = len(row_linear) // 4
    q3 = (3 * len(row_linear)) // 4
    assert float(row_cyl[q1]) < float(row_linear[q1]) - 5.0
    assert float(row_cyl[q3]) > float(row_linear[q3]) + 5.0


def test_unwrap_source_x_range_focuses_target_band():
    src = _gradient_rgb(201, 25)
    out = unwrap_cylindrical_view_image(src, strength=0.0, source_x_range=(50.0, 150.0))
    row = np.asarray(out, dtype=np.float64)[12, :, 0]
    # Left edge should map near source x=50 (about 63.75), not near 0.
    assert float(row[0]) > 55.0
    # Right edge should map near source x=150 (about 191.25), not near 255.
    assert float(row[-1]) < 200.0


def test_trace_foreground_x_range_on_white_background():
    h, w = 80, 160
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[20:65, 40:121, :] = 60
    x_rng = trace_foreground_x_range(img, white_threshold=4.0)
    assert x_rng is not None
    x0, x1 = x_rng
    assert 36.0 <= float(x0) <= 44.0
    assert 117.0 <= float(x1) <= 124.0


def test_unwrap_seam_shift_rolls_output_horizontally():
    src = _gradient_rgb(160, 40)
    base = unwrap_cylindrical_view_image(src, strength=0.0, seam_shift_pct=0.0)
    shifted = unwrap_cylindrical_view_image(src, strength=0.0, seam_shift_pct=25.0)
    a = np.asarray(base, dtype=np.uint8)
    b = np.asarray(shifted, dtype=np.uint8)
    expect = np.roll(a, shift=int(round(0.25 * float(a.shape[1]))), axis=1)
    assert int(np.max(np.abs(expect.astype(np.int16) - b.astype(np.int16)))) <= 1


def test_band_unwrap_straightens_curved_strip_to_full_height():
    h, w = 120, 240
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    xs = np.arange(w, dtype=np.float64)
    top = 24.0 + 12.0 * np.sin(np.linspace(-1.0, 1.0, num=w) * np.pi)
    thickness = 30.0 + 4.0 * np.cos(np.linspace(-1.0, 1.0, num=w) * np.pi)
    bottom = top + thickness
    for x in range(w):
        y0 = int(max(0, min(h - 1, round(float(top[x])))))
        y1 = int(max(0, min(h - 1, round(float(bottom[x])))))
        if y1 <= y0:
            continue
        ys = np.arange(y0, y1 + 1, dtype=np.float64)
        t = (ys - float(y0)) / max(1.0, float(y1 - y0))
        vals = np.clip(np.round(255.0 * t), 0.0, 255.0).astype(np.uint8)
        img[ys.astype(np.int32), x, 0] = vals
        img[ys.astype(np.int32), x, 1] = vals
        img[ys.astype(np.int32), x, 2] = vals

    out = unwrap_cylindrical_band_image(
        Image.fromarray(img, mode="RGB"),
        strength=0.0,
        source_x_range=(24.0, 216.0),
        output_width=180,
        fit_height_to_band=True,
        interpolation_order=1,
    )
    arr = np.asarray(out, dtype=np.float64)
    assert arr.shape[0] >= 64
    # After band-straighten, top should be dark and bottom bright across most columns.
    assert float(np.mean(arr[0, :, 0])) < 50.0
    assert float(np.mean(arr[-1, :, 0])) > 205.0
