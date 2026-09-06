"""Painted decoration as a field on the developed wall: the base colour map
sampled onto the same lattice the relief uses, reduced to how much paint
each pixel carries.

Porcelain is not rubbed.  Its decoration is paint - gold over the glaze,
cobalt under it, iron brown - and a measured drawing draws it as lines: a
painted line along its centre, a painted area by its edge.  What the normal
map is to the incised pot, the base colour map is to the painted bowl: the
same atlas carries both, and the same developed lattice, so a painted line
is traced where a stroke would be and carried back onto the elevation the
same way.  This module gives the lines reader the paint field; the reader's
stroke rules do the rest, with the paint standing in for depth.

``chroma`` names how a texel's RGB becomes paint, in 0..1:

- ``red_over_blue/v1``: red minus blue - gold, ochre, iron red over white;
- ``blue_over_red/v1``: blue minus red - cobalt blue over white;
- ``darkness/v1``: one minus the luminance - iron black and brown.

Nothing here is a lightness model; the numbers are what the map says,
clipped, and the threshold is the reader's.  A painted area wider than
``band_um`` is not a line: only a thin ribbon inside its boundary is kept,
so the reader traces its edge and not a wandering centre.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifact_cancellation import CancellationProbe, raise_if_cancelled
from .artifact_texture_relief import (
    MAX_NORMAL_MAP_BYTES,
    MAX_NORMAL_MAP_SIDE,
    TEXTURE_ATLAS_KIND,
    ArtifactTextureReliefError,
    TextureAtlas,
    _sha256_of_file,
    rasterise_developed_texels,
)

TEXTURE_PAINT_SAMPLING = "nearest_texel/v1"
TEXTURE_PAINT_CHROMA_RED_OVER_BLUE = "red_over_blue/v1"
TEXTURE_PAINT_CHROMA_BLUE_OVER_RED = "blue_over_red/v1"
TEXTURE_PAINT_CHROMA_DARKNESS = "darkness/v1"
TEXTURE_PAINT_CHROMAS: tuple[str, ...] = (
    TEXTURE_PAINT_CHROMA_RED_OVER_BLUE,
    TEXTURE_PAINT_CHROMA_BLUE_OVER_RED,
    TEXTURE_PAINT_CHROMA_DARKNESS,
)
DEFAULT_TEXTURE_PAINT_CHROMA = TEXTURE_PAINT_CHROMA_RED_OVER_BLUE
#: A painted area wider than this is drawn by its edge, not its centre.
DEFAULT_TEXTURE_PAINT_BAND_UM = 800
MIN_TEXTURE_PAINT_BAND_UM = 0
MAX_TEXTURE_PAINT_BAND_UM = 50_000
#: The ribbon kept inside a wide area's boundary, in pixels, for the edge
#: to be traced along.
_EDGE_RIBBON_PX = 1.5


class ArtifactTexturePaintError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ColourMap:
    """A base colour map: each texel's RGB as the file has it, 8 bits."""

    rgb: np.ndarray
    sha256: str
    byte_length: int

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.rgb.dtype != np.uint8:
            raise ArtifactTexturePaintError("colour map must be an (h, w, 3) uint8 image")
        if not (isinstance(self.sha256, str) and len(self.sha256) == 64):
            raise ArtifactTexturePaintError("colour map sha256 must be 64 hex characters")

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    def recipe_block(self) -> dict[str, Any]:
        return {
            "byte_length": int(self.byte_length),
            "height": self.height,
            "sha256": self.sha256,
            "width": self.width,
        }


def read_colour_map(path: str | Path) -> ColourMap:
    """Decode a base colour image to RGB bytes, with the file's own hash."""

    from PIL import Image  # noqa: PLC0415

    source = Path(path)
    if not source.is_file():
        raise ArtifactTexturePaintError(f"colour map is not a file: {source}")
    byte_length = source.stat().st_size
    if byte_length > MAX_NORMAL_MAP_BYTES:
        raise ArtifactTexturePaintError("colour map exceeds the size limit")
    with Image.open(source) as image:
        if image.width > MAX_NORMAL_MAP_SIDE or image.height > MAX_NORMAL_MAP_SIDE:
            raise ArtifactTexturePaintError("colour map exceeds the side limit")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return ColourMap(rgb=np.ascontiguousarray(rgb), sha256=_sha256_of_file(source), byte_length=int(byte_length))


def texture_paint_block(
    atlas: TextureAtlas,
    colour_map: ColourMap,
    *,
    chroma: str = DEFAULT_TEXTURE_PAINT_CHROMA,
    band_um: int = DEFAULT_TEXTURE_PAINT_BAND_UM,
) -> dict[str, Any]:
    """The recipe block that names the two files, the chroma and the band."""

    if chroma not in TEXTURE_PAINT_CHROMAS:
        raise ArtifactTexturePaintError(f"chroma must be one of {', '.join(TEXTURE_PAINT_CHROMAS)}")
    if isinstance(band_um, bool) or not isinstance(band_um, (int, np.integer)):
        raise ArtifactTexturePaintError("band_um must be an integer")
    band = int(band_um)
    if not MIN_TEXTURE_PAINT_BAND_UM <= band <= MAX_TEXTURE_PAINT_BAND_UM:
        raise ArtifactTexturePaintError(
            f"band_um must be in the inclusive range {MIN_TEXTURE_PAINT_BAND_UM}..{MAX_TEXTURE_PAINT_BAND_UM}"
        )
    return {
        "atlas": atlas.recipe_block(),
        "band_um": band,
        "chroma": chroma,
        "colour_map": colour_map.recipe_block(),
        "sampling": TEXTURE_PAINT_SAMPLING,
    }


def validate_texture_paint_block(value: object) -> dict[str, Any]:
    """Check the block's shape and ranges without the files at hand."""

    if not isinstance(value, Mapping):
        raise ArtifactTexturePaintError("texture_paint must be an object")
    expected_keys = {"atlas", "band_um", "chroma", "colour_map", "sampling"}
    if set(value) != expected_keys:
        raise ArtifactTexturePaintError(
            f"texture_paint must carry exactly {', '.join(sorted(expected_keys))}"
        )
    if value["sampling"] != TEXTURE_PAINT_SAMPLING:
        raise ArtifactTexturePaintError("texture_paint names a sampling this release does not have")
    if value["chroma"] not in TEXTURE_PAINT_CHROMAS:
        raise ArtifactTexturePaintError("texture_paint names a chroma this release does not have")
    band = value["band_um"]
    if isinstance(band, bool) or not isinstance(band, int):
        raise ArtifactTexturePaintError("texture_paint band_um must be an integer")
    if not MIN_TEXTURE_PAINT_BAND_UM <= band <= MAX_TEXTURE_PAINT_BAND_UM:
        raise ArtifactTexturePaintError("texture_paint band_um is out of range")
    atlas = value["atlas"]
    if not isinstance(atlas, Mapping) or set(atlas) != {
        "byte_length",
        "kind",
        "sha256",
        "triangle_count",
        "vertex_count",
    }:
        raise ArtifactTexturePaintError("texture_paint atlas block is malformed")
    if atlas["kind"] != TEXTURE_ATLAS_KIND:
        raise ArtifactTexturePaintError("texture_paint names an atlas kind this release does not have")
    colour_map = value["colour_map"]
    if not isinstance(colour_map, Mapping) or set(colour_map) != {"byte_length", "height", "sha256", "width"}:
        raise ArtifactTexturePaintError("texture_paint colour_map block is malformed")
    for block in (atlas, colour_map):
        sha = block["sha256"]
        if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
            raise ArtifactTexturePaintError("texture_paint sha256 must be 64 lowercase hex characters")
        for key in block:
            if key in ("kind", "sha256"):
                continue
            number = block[key]
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ArtifactTexturePaintError(f"texture_paint {key} must be a non-negative integer")
    return {
        "atlas": dict(atlas),
        "band_um": int(band),
        "chroma": str(value["chroma"]),
        "colour_map": dict(colour_map),
        "sampling": TEXTURE_PAINT_SAMPLING,
    }


def require_texture_paint_sources(
    block: Mapping[str, Any], atlas: TextureAtlas, colour_map: ColourMap
) -> None:
    """The files at hand must be the files the recipe names."""

    if atlas.sha256 != block["atlas"]["sha256"] or atlas.byte_length != int(block["atlas"]["byte_length"]):
        raise ArtifactTexturePaintError(
            "the texture atlas at hand is not the one the recipe names "
            f"(recipe sha256 {block['atlas']['sha256'][:12]}..., file {atlas.sha256[:12]}...)"
        )
    if atlas.triangle_count != int(block["atlas"]["triangle_count"]) or int(atlas.vertices.shape[0]) != int(
        block["atlas"]["vertex_count"]
    ):
        raise ArtifactTexturePaintError("the texture atlas at hand welds to different counts")
    if colour_map.sha256 != block["colour_map"]["sha256"] or colour_map.byte_length != int(
        block["colour_map"]["byte_length"]
    ):
        raise ArtifactTexturePaintError(
            "the colour map at hand is not the one the recipe names "
            f"(recipe sha256 {block['colour_map']['sha256'][:12]}..., file {colour_map.sha256[:12]}...)"
        )
    if colour_map.width != int(block["colour_map"]["width"]) or colour_map.height != int(
        block["colour_map"]["height"]
    ):
        raise ArtifactTexturePaintError("the colour map at hand has different dimensions")


def paint_of(rgb: np.ndarray, chroma: str) -> np.ndarray:
    """How much paint each RGB carries under the chroma rule, in 0..1."""

    colour = np.asarray(rgb, dtype=np.float64) / 255.0
    if chroma == TEXTURE_PAINT_CHROMA_RED_OVER_BLUE:
        return np.clip(colour[..., 0] - colour[..., 2], 0.0, 1.0)
    if chroma == TEXTURE_PAINT_CHROMA_BLUE_OVER_RED:
        return np.clip(colour[..., 2] - colour[..., 0], 0.0, 1.0)
    if chroma == TEXTURE_PAINT_CHROMA_DARKNESS:
        return np.clip(1.0 - (0.299 * colour[..., 0] + 0.587 * colour[..., 1] + 0.114 * colour[..., 2]), 0.0, 1.0)
    raise ArtifactTexturePaintError(f"chroma must be one of {', '.join(TEXTURE_PAINT_CHROMAS)}")


def texture_paint_field(
    *,
    developed_uv_mm: np.ndarray,
    developed_faces: np.ndarray,
    developed_points_mm: np.ndarray,
    source_face_indices: np.ndarray,
    source_vertex_indices: np.ndarray,
    atlas: TextureAtlas,
    colour_map: ColourMap,
    pixels_per_mm: int,
    margin_pixels: int,
    chroma: str,
    band_um: int,
    threshold: float,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[np.ndarray, int, int, dict[str, Any]]:
    """The paint at every pixel centre of the developed lattice, in 0..1
    where the development covers the pixel and -inf elsewhere, with a wide
    painted area reduced to a ribbon inside its boundary.  ``threshold`` is
    the paint a pixel needs to count as painted when the areas are measured
    - the reader's own threshold, so the two agree."""

    from scipy.ndimage import distance_transform_edt, label, maximum  # noqa: PLC0415

    try:
        lattice = rasterise_developed_texels(
            developed_uv_mm=developed_uv_mm,
            developed_faces=developed_faces,
            developed_points_mm=developed_points_mm,
            source_face_indices=source_face_indices,
            source_vertex_indices=source_vertex_indices,
            atlas=atlas,
            map_width=colour_map.width,
            map_height=colour_map.height,
            pixels_per_mm=pixels_per_mm,
            margin_pixels=margin_pixels,
            with_axes=False,
            cancellation_probe=cancellation_probe,
        )
    except ArtifactTextureReliefError as exc:
        raise ArtifactTexturePaintError(str(exc)) from exc
    covered = lattice.covered
    paint = np.zeros(covered.shape, dtype=np.float64)
    paint[covered] = paint_of(colour_map.rgb[lattice.texel_row[covered], lattice.texel_col[covered]], chroma)
    raise_if_cancelled(cancellation_probe)
    painted = covered & (paint >= float(threshold))
    painted_count = int(np.count_nonzero(painted))
    wide_count = 0
    if painted_count and band_um > 0:
        labels, count = label(painted, structure=np.ones((3, 3), dtype=bool))
        distance = distance_transform_edt(painted)
        widest = np.asarray(maximum(distance, labels, index=np.arange(1, count + 1)), dtype=np.float64)
        is_wide = 2.0 * widest / float(pixels_per_mm) > band_um / 1000.0
        wide_count = int(np.count_nonzero(is_wide))
        if wide_count:
            wide = np.concatenate([[False], is_wide])[labels]
            # Inside a wide area only the ribbon along its boundary keeps its
            # paint, so the reader's ridge falls on the edge.
            paint = np.where(wide & (distance > _EDGE_RIBBON_PX), 0.0, paint)
    field = np.full(covered.shape, -np.inf, dtype=np.float64)
    field[covered] = paint[covered]
    qc: dict[str, Any] = {
        "texture_paint_covered_pixel_count": int(np.count_nonzero(covered)),
        "texture_paint_painted_pixel_count": painted_count,
        "texture_paint_unmatched_corner_count": lattice.unmatched_corners,
        "texture_paint_wide_area_count": wide_count,
    }
    return field, lattice.minimum_u, lattice.minimum_v, qc


__all__ = [
    "DEFAULT_TEXTURE_PAINT_BAND_UM",
    "DEFAULT_TEXTURE_PAINT_CHROMA",
    "TEXTURE_PAINT_CHROMAS",
    "TEXTURE_PAINT_CHROMA_BLUE_OVER_RED",
    "TEXTURE_PAINT_CHROMA_DARKNESS",
    "TEXTURE_PAINT_CHROMA_RED_OVER_BLUE",
    "TEXTURE_PAINT_SAMPLING",
    "ArtifactTexturePaintError",
    "ColourMap",
    "paint_of",
    "read_colour_map",
    "require_texture_paint_sources",
    "texture_paint_block",
    "texture_paint_field",
    "validate_texture_paint_block",
]
