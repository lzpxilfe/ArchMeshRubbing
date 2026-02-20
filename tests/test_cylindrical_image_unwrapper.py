import numpy as np
from PIL import Image

from src.core.cylindrical_image_unwrapper import unwrap_cylindrical_view_image


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
