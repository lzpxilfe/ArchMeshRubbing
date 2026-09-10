"""What a texture must become before OpenGL can show it.

The scanner hands back a colour image and a UV per vertex.  The pattern the
archaeologist has to read - the incised lines, the painted flower, the
glaze - lives in that image and nowhere else in the file, so a viewer that
draws the mesh grey shows the shape and hides the evidence.

Everything here is plain arrays: no GL calls, no widget, no context.  The
viewport does the binding; this decides what bytes get bound, so the
decisions can be tested where there is no screen at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Largest side we will hand OpenGL when the driver does not say otherwise.
#: GL 2.1 guarantees far less, but every driver this program runs on allows
#: at least this, and the real limit is asked for at bind time.
MAX_TEXTURE_SIDE_DEFAULT = 4096

#: The pixel layouts we can bind, by channel count.  The names are the
#: viewport's to translate into GL enums - this module never imports GL.
PIXEL_LAYOUTS: dict[int, str] = {
    1: "luminance",
    2: "luminance_alpha",
    3: "rgb",
    4: "rgba",
}


@dataclass(frozen=True, slots=True)
class TextureImage:
    """An image ready to upload: bytes, size, and how it got that way."""

    pixels: np.ndarray
    width: int
    height: int
    layout: str
    #: The source size, when it had to be shrunk to fit the driver's limit.
    shrunk_from: tuple[int, int] | None = None

    @property
    def channels(self) -> int:
        return int(self.pixels.shape[2])

    def to_dict(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "layout": self.layout,
            "shrunk_from": list(self.shrunk_from) if self.shrunk_from else None,
            "width": self.width,
        }


def can_show_texture(mesh: Any) -> bool:
    """Whether this mesh carries both halves of a texture.

    An image without UVs cannot be laid on the surface, and UVs without an
    image have nothing to lay - either alone is not a texture.
    """

    if mesh is None:
        return False
    image = getattr(mesh, "texture", None)
    uv = getattr(mesh, "uv_coords", None)
    if image is None or uv is None:
        return False
    try:
        return bool(np.asarray(image).size) and bool(np.asarray(uv).size)
    except Exception:
        return False


def texture_image(image: Any, *, max_side: int = MAX_TEXTURE_SIDE_DEFAULT) -> TextureImage:
    """The image as bytes OpenGL will take, bottom row first.

    Two conversions matter and both are silent errors when skipped.  A float
    image (some loaders hand back 0-1) uploaded as bytes is black.  And an
    image uploaded top row first is the pattern upside down: image rows run
    downward from the top, while UV v runs upward from the bottom, which is
    the convention OBJ and glTF both write.  So the rows are reversed here,
    once, where it can be tested.
    """

    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError("a texture image must be 2D or 3D")
    height, width, channels = (int(value) for value in array.shape)
    if height <= 0 or width <= 0:
        raise ValueError("a texture image must have pixels")
    if channels not in PIXEL_LAYOUTS:
        raise ValueError(f"a texture image must have 1-4 channels; got {channels}")
    if int(max_side) < 1:
        raise ValueError("max_side must be at least 1")

    if array.dtype == np.uint8:
        pixels = array
    elif np.issubdtype(array.dtype, np.floating):
        # 0-1 is the common float convention; anything above 1 is already a
        # 0-255 range stored as float, so scale by what is actually there.
        finite = array[np.isfinite(array)]
        top = float(finite.max()) if finite.size else 0.0
        scale = 255.0 if top <= 1.0 else 1.0
        pixels = np.clip(np.nan_to_num(array) * scale, 0.0, 255.0).astype(np.uint8)
    else:
        pixels = np.clip(array, 0, 255).astype(np.uint8)

    shrunk_from: tuple[int, int] | None = None
    limit = int(max_side)
    if width > limit or height > limit:
        shrunk_from = (width, height)
        step = int(max(1, -(-max(width, height) // limit)))
        pixels = pixels[::step, ::step, :]
        height, width = int(pixels.shape[0]), int(pixels.shape[1])

    # Bottom row first, and contiguous: GL reads straight out of this buffer.
    pixels = np.ascontiguousarray(pixels[::-1], dtype=np.uint8)
    return TextureImage(
        pixels=pixels,
        width=width,
        height=height,
        layout=PIXEL_LAYOUTS[channels],
        shrunk_from=shrunk_from,
    )


def texture_corner_uv(uv_coords: Any, faces: Any) -> np.ndarray:
    """One UV per triangle corner, in the order the vertex buffer is built.

    The geometry buffer is `faces.reshape(-1)` - every triangle's three
    corners in turn - so the texture coordinates must be laid out the same
    way or the pattern lands on the wrong faces.
    """

    uv = np.asarray(uv_coords, dtype=np.float32)
    if uv.ndim != 2 or int(uv.shape[1]) < 2:
        raise ValueError("uv coordinates must be (N, 2)")
    index = np.asarray(faces).reshape(-1)
    if index.size == 0:
        raise ValueError("a textured mesh must have faces")
    if int(index.max(initial=0)) >= int(uv.shape[0]):
        raise ValueError("a face names a vertex the uv coordinates do not have")
    return np.ascontiguousarray(uv[index, :2], dtype=np.float32)


def texture_description(mesh: Any, *, max_side: int = MAX_TEXTURE_SIDE_DEFAULT) -> dict[str, Any]:
    """What the window can say about this mesh's texture, or why not."""

    if not can_show_texture(mesh):
        return {"showable": False, "reason": "이 메쉬에는 텍스처와 UV가 함께 있지 않습니다."}
    try:
        prepared = texture_image(mesh.texture, max_side=max_side)
    except ValueError as exc:
        return {"showable": False, "reason": str(exc)}
    described = {"showable": True, **prepared.to_dict()}
    return described


__all__ = [
    "MAX_TEXTURE_SIDE_DEFAULT",
    "PIXEL_LAYOUTS",
    "TextureImage",
    "can_show_texture",
    "texture_corner_uv",
    "texture_description",
    "texture_image",
]
