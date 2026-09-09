"""The texture must reach the screen the right way up and the right colour."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.gui.mesh_texture import (
    MAX_TEXTURE_SIDE_DEFAULT,
    can_show_texture,
    texture_corner_uv,
    texture_description,
    texture_image,
)


def _mesh(image, uv):
    return SimpleNamespace(texture=image, uv_coords=uv)


def test_a_texture_is_an_image_and_uvs_together() -> None:
    """Either half alone shows nothing: an image with no UVs has nowhere to
    go, and UVs with no image have nothing to carry."""

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    uv = np.zeros((3, 2), dtype=np.float64)
    assert can_show_texture(_mesh(image, uv))
    assert not can_show_texture(_mesh(image, None))
    assert not can_show_texture(_mesh(None, uv))
    assert not can_show_texture(None)


def test_the_pattern_is_not_upside_down() -> None:
    """Image rows run down from the top; UV v runs up from the bottom.  The
    rows are reversed once, here, or every pattern is mirrored top to bottom
    on the artifact and nobody can tell by looking at a pot."""

    image = np.zeros((2, 1, 3), dtype=np.uint8)
    image[0, 0] = (255, 0, 0)  # top row red
    image[1, 0] = (0, 0, 255)  # bottom row blue
    prepared = texture_image(image)
    # First row of the upload buffer is what GL puts at v = 0: the bottom.
    assert tuple(prepared.pixels[0, 0]) == (0, 0, 255)
    assert tuple(prepared.pixels[1, 0]) == (255, 0, 0)


def test_a_float_image_becomes_bytes_instead_of_black() -> None:
    """A 0-1 float image cast straight to uint8 is all zeros - a black pot."""

    image = np.zeros((2, 2, 3), dtype=np.float32)
    image[:, :, 0] = 1.0
    image[:, :, 1] = 0.5
    prepared = texture_image(image)
    assert prepared.pixels.dtype == np.uint8
    assert int(prepared.pixels[..., 0].max()) == 255
    assert 120 <= int(prepared.pixels[..., 1].max()) <= 135


def test_a_float_image_already_in_0_255_is_not_scaled_again() -> None:
    image = np.full((2, 2, 3), 200.0, dtype=np.float64)
    prepared = texture_image(image)
    assert int(prepared.pixels.max()) == 200


def test_every_channel_count_a_loader_hands_back_is_named() -> None:
    for channels, layout in ((1, "luminance"), (2, "luminance_alpha"), (3, "rgb"), (4, "rgba")):
        image = np.zeros((3, 3, channels), dtype=np.uint8)
        assert texture_image(image).layout == layout
    grey = np.zeros((3, 3), dtype=np.uint8)
    assert texture_image(grey).layout == "luminance"


def test_an_image_too_big_for_the_driver_is_shrunk_and_says_so() -> None:
    """Refusing a 8k scan would hide the pattern; shrinking it silently would
    leave nothing to explain a soft picture.  It shrinks, and it reports."""

    image = np.zeros((300, 900, 3), dtype=np.uint8)
    prepared = texture_image(image, max_side=100)
    assert prepared.width <= 100 and prepared.height <= 100
    assert prepared.shrunk_from == (900, 300)
    assert prepared.pixels.flags["C_CONTIGUOUS"]

    kept = texture_image(image, max_side=1000)
    assert kept.shrunk_from is None


def test_the_upload_buffer_is_contiguous_bytes() -> None:
    """GL reads straight out of this buffer: a view with a negative stride
    would upload whatever memory happens to follow it."""

    image = np.zeros((5, 7, 3), dtype=np.uint8)
    prepared = texture_image(image)
    assert prepared.pixels.flags["C_CONTIGUOUS"]
    assert prepared.pixels.dtype == np.uint8
    assert prepared.pixels.shape == (5, 7, 3)
    assert prepared.width == 7 and prepared.height == 5
    assert prepared.channels == 3


def test_an_image_with_no_pixels_or_too_many_channels_is_refused() -> None:
    with pytest.raises(ValueError):
        texture_image(np.zeros((0, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        texture_image(np.zeros((4, 4, 5), dtype=np.uint8))
    with pytest.raises(ValueError):
        texture_image(np.zeros((2, 2, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        texture_image(np.zeros((4, 4, 3), dtype=np.uint8), max_side=0)


def test_the_uvs_are_laid_out_corner_by_corner_like_the_geometry() -> None:
    """The vertex buffer is faces.reshape(-1); the texture coordinates must
    follow the same order or the pattern lands on the wrong faces."""

    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    corners = texture_corner_uv(uv, faces)
    assert corners.shape == (6, 2)
    assert corners.dtype == np.float32
    assert np.allclose(corners[0], (0.0, 0.0))
    assert np.allclose(corners[1], (1.0, 0.0))
    assert np.allclose(corners[5], (0.0, 1.0))
    assert corners.flags["C_CONTIGUOUS"]


def test_uvs_that_do_not_cover_the_faces_are_refused() -> None:
    """Reading past the end would upload other vertices' coordinates - a
    pattern smeared across the mesh, and no error to say why."""

    uv = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(ValueError):
        texture_corner_uv(uv, faces)
    with pytest.raises(ValueError):
        texture_corner_uv(np.zeros((3,), dtype=np.float64), faces)
    with pytest.raises(ValueError):
        texture_corner_uv(uv, np.zeros((0, 3), dtype=np.int64))


def test_a_uv_with_more_than_two_columns_keeps_the_first_two() -> None:
    uv = np.array([[0.25, 0.5, 9.0], [0.75, 0.5, 9.0], [0.5, 1.0, 9.0]], dtype=np.float64)
    corners = texture_corner_uv(uv, np.array([[0, 1, 2]], dtype=np.int64))
    assert corners.shape == (3, 2)
    assert np.allclose(corners[0], (0.25, 0.5))


def test_the_window_can_say_why_a_mesh_shows_no_texture() -> None:
    """'No texture' and 'a texture this program cannot read' are different
    answers, and the archaeologist has to be told which one this is."""

    described = texture_description(_mesh(np.zeros((2, 2, 3), np.uint8), np.zeros((3, 2))))
    assert described["showable"] is True
    assert described["width"] == 2 and described["layout"] == "rgb"

    missing = texture_description(_mesh(None, np.zeros((3, 2))))
    assert missing["showable"] is False
    assert "텍스처" in missing["reason"]

    unreadable = texture_description(_mesh(np.zeros((2, 2, 7), np.uint8), np.zeros((3, 2))))
    assert unreadable["showable"] is False
    assert "channels" in unreadable["reason"]


def test_the_default_limit_is_one_a_gl21_driver_can_be_asked_for() -> None:
    assert MAX_TEXTURE_SIDE_DEFAULT >= 2048
