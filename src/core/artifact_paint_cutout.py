"""A painted mark cut out of the base colour and kept as an image: the
plum blossom on a bowl's wall, the ink character on a dish's foot.

Not every painted thing wants tracing.  A brushed character or a dense
motif is often better shown as it is - cut out of the colour map the way a
magic wand cuts it in an image editor, only the painted pixels kept, and
pasted onto the drawing at its own place.  The cutout is a raster on a
view's plane: the colour map is sampled onto the orthographic lattice of
that view, each pixel reduced to how much paint it carries under a chroma
rule, the coverage ramped from ``threshold`` (no ink) to ``full`` (solid
ink), and the painted extent cropped with a margin.  Its tone is the
reading's choice: ink, one colour at the paint's coverage, in a tone that
does not shout over the line work; or the paint's own colour as the map
has it, with the coverage as its alpha - the wand's selection lifted
whole.  The record keeps the receipt - what the raster is, where it sits
in the view, its hash - and the pixels travel beside it, as a rubbing's
do; the sheet pastes them only when they match.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_cancellation import CancellationProbe, raise_if_cancelled
from .artifact_document import (
    ArtifactDocument,
    ArtifactDocumentError,
    DerivedRecord,
    OperationContext,
    RecordLifecycleStatus,
)
from .artifact_outline_extractor import OutlineView, outline_frame
from .artifact_session import ArtifactSession, ArtifactSessionError
from .artifact_texture_paint import (
    DEFAULT_TEXTURE_PAINT_CHROMA,
    ArtifactTexturePaintError,
    ColourMap,
    require_texture_paint_sources,
    texture_paint_and_colour_field,
    texture_paint_block,
    validate_texture_paint_block,
)
from .artifact_texture_relief import ArtifactTextureReliefError, TextureAtlas
from .canonical_json import CanonicalJSONError, canonical_json_bytes

PAINT_CUTOUT_RECORD_TYPE = "measurement.paint_cutout.v1"
PAINT_CUTOUT_OPERATION_KIND = "paint_cutout"
PAINT_CUTOUT_ALGORITHM = "archmeshrubbing.view_paint_cutout"
PAINT_CUTOUT_ALGORITHM_VERSION = "1.0.0"
PAINT_CUTOUT_COORDINATE_SPACE = "view_plane_mm/v1"
PAINT_CUTOUT_PAYLOAD_EXTENSION_KEY = "org.archmeshrubbing:paint-cutout-v1"
PAINT_CUTOUT_RECEIPT_SCHEMA_VERSION = "1.0.0"
PAINT_CUTOUT_GEOMETRY_REF_PREFIX = "urn:archmeshrubbing:paint-cutout:sha256:"
#: Ink: grey 0 with the coverage as alpha.  Colour: the map's own RGB with
#: the coverage as alpha.  The channel count names the format.
PAINT_CUTOUT_PIXEL_FORMAT_INK = "gray8_alpha8_ink_coverage/v1"
PAINT_CUTOUT_PIXEL_FORMAT_COLOUR = "rgb8_alpha8_painted_colour/v1"
PAINT_CUTOUT_PIXEL_FORMATS: dict[int, str] = {2: PAINT_CUTOUT_PIXEL_FORMAT_INK, 4: PAINT_CUTOUT_PIXEL_FORMAT_COLOUR}
PAINT_CUTOUT_TONE_INK = "ink/v1"
PAINT_CUTOUT_TONE_COLOUR = "colour_as_painted/v1"
PAINT_CUTOUT_TONES = (PAINT_CUTOUT_TONE_INK, PAINT_CUTOUT_TONE_COLOUR)
PAINT_CUTOUT_TONE_FORMATS: dict[str, str] = {
    PAINT_CUTOUT_TONE_INK: PAINT_CUTOUT_PIXEL_FORMAT_INK,
    PAINT_CUTOUT_TONE_COLOUR: PAINT_CUTOUT_PIXEL_FORMAT_COLOUR,
}
DEFAULT_PAINT_CUTOUT_TONE = PAINT_CUTOUT_TONE_INK
PAINT_CUTOUT_PAINTER = "far_to_near_facing_faces/v1"
PAINT_CUTOUT_ROW_ORDER = "top_row_first/v1"

DEFAULT_PAINT_CUTOUT_PIXELS_PER_MM = 10
DEFAULT_PAINT_CUTOUT_FACING_COS = 0.35
DEFAULT_PAINT_CUTOUT_THRESHOLD_THOUSANDTHS = 60
DEFAULT_PAINT_CUTOUT_FULL_THOUSANDTHS = 250
DEFAULT_PAINT_CUTOUT_MARGIN_UM = 1_000
MIN_PAINT_CUTOUT_PIXELS_PER_MM = 1
MAX_PAINT_CUTOUT_PIXELS_PER_MM = 50
MAX_PAINT_CUTOUT_PIXELS = 20_000_000
MAX_PAINT_CUTOUT_GRID_INDEX = 10**9


class ArtifactPaintCutoutError(ValueError):
    pass


def _strict_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactPaintCutoutError(f"{name} must be an integer")
    number = int(value)
    if not minimum <= number <= maximum:
        raise ArtifactPaintCutoutError(f"{name} must be in the inclusive range {minimum}..{maximum}")
    return number


def _exact_keys(value: object, keys: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactPaintCutoutError(f"{name} must be an object")
    if set(value) != keys:
        raise ArtifactPaintCutoutError(f"{name} must carry exactly {', '.join(sorted(keys))}")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not (isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)):
        raise ArtifactPaintCutoutError(f"{name} must be 64 lowercase hex characters")
    return value


def _view_name(view: object) -> str:
    if isinstance(view, OutlineView):
        return view.value
    if isinstance(view, str):
        try:
            return OutlineView(view).value
        except ValueError as exc:
            raise ArtifactPaintCutoutError(f"unknown view: {view!r}") from exc
    raise ArtifactPaintCutoutError("view must be an OutlineView or its name")


@dataclass(frozen=True, slots=True)
class PaintCutoutRaster:
    """The cutout's pixels on a view's plane: grey 0 (ink) or the paint's
    own colour, with the alpha the paint's coverage, row 0 the top of the
    view, and where its left and bottom edges lie in the view's
    millimetres."""

    pixels: np.ndarray
    pixels_per_meter: int
    left_um: int
    bottom_um: int
    view: str

    def __post_init__(self) -> None:
        array = np.asarray(self.pixels)
        if (
            array.dtype != np.uint8
            or array.ndim != 3
            or array.shape[2] not in PAINT_CUTOUT_PIXEL_FORMATS
            or array.shape[0] <= 0
            or array.shape[1] <= 0
        ):
            raise ArtifactPaintCutoutError("cutout pixels must be a non-empty HxWx2 (ink) or HxWx4 (colour) uint8 array")
        if array.shape[0] * array.shape[1] > MAX_PAINT_CUTOUT_PIXELS:
            raise ArtifactPaintCutoutError("cutout pixels exceed the safety limit")
        ppm = _strict_int(
            self.pixels_per_meter, name="pixels_per_meter", minimum=1000, maximum=MAX_PAINT_CUTOUT_PIXELS_PER_MM * 1000
        )
        if ppm % 1000 != 0:
            raise ArtifactPaintCutoutError("pixels_per_meter must encode an integer pixels/mm")
        for name, value in (("left_um", self.left_um), ("bottom_um", self.bottom_um)):
            _strict_int(value, name=name, minimum=-MAX_PAINT_CUTOUT_GRID_INDEX, maximum=MAX_PAINT_CUTOUT_GRID_INDEX)
        copied = np.ascontiguousarray(array).copy()
        copied.setflags(write=False)
        object.__setattr__(self, "pixels", copied)
        object.__setattr__(self, "pixels_per_meter", ppm)
        object.__setattr__(self, "view", _view_name(self.view))

    @property
    def channels(self) -> int:
        return int(self.pixels.shape[2])

    @property
    def pixel_format(self) -> str:
        return PAINT_CUTOUT_PIXEL_FORMATS[self.channels]

    @property
    def alpha(self) -> np.ndarray:
        return self.pixels[..., self.channels - 1]

    @property
    def width_pixels(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height_pixels(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width_um(self) -> int:
        return self.width_pixels * 1_000_000 // self.pixels_per_meter

    @property
    def height_um(self) -> int:
        return self.height_pixels * 1_000_000 // self.pixels_per_meter

    @property
    def rectangle_mm(self) -> tuple[float, float, float, float]:
        """(left, bottom, right, top) in the view's millimetres."""

        return (
            self.left_um / 1000.0,
            self.bottom_um / 1000.0,
            (self.left_um + self.width_um) / 1000.0,
            (self.bottom_um + self.height_um) / 1000.0,
        )

    def semantic_header(self) -> dict[str, Any]:
        return {
            "bottom_um": int(self.bottom_um),
            "coordinate_space": PAINT_CUTOUT_COORDINATE_SPACE,
            "height_pixels": self.height_pixels,
            "left_um": int(self.left_um),
            "pixel_format": self.pixel_format,
            "pixels_per_meter": self.pixels_per_meter,
            "row_order": PAINT_CUTOUT_ROW_ORDER,
            "schema_version": PAINT_CUTOUT_RECEIPT_SCHEMA_VERSION,
            "view": self.view,
            "width_pixels": self.width_pixels,
        }

    @property
    def raster_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"archmeshrubbing.paint-cutout-raster\0")
        digest.update(canonical_json_bytes(self.semantic_header()))
        digest.update(b"\0")
        digest.update(self.pixels.tobytes(order="C"))
        return digest.hexdigest()

    @property
    def geometry_ref(self) -> str:
        return f"{PAINT_CUTOUT_GEOMETRY_REF_PREFIX}{self.raster_sha256}"

    def receipt(self) -> dict[str, Any]:
        return {
            **self.semantic_header(),
            "raster_sha256": self.raster_sha256,
            "raw_pixel_byte_length": int(self.pixels.nbytes),
            "raw_pixel_sha256": hashlib.sha256(self.pixels.tobytes(order="C")).hexdigest(),
        }

    def qc_summary(self) -> dict[str, Any]:
        alpha = self.alpha.astype(np.float64)
        inked = alpha > 0
        return {
            "coverage_mean_thousandths": int(round(float(alpha[inked].mean()) / 255.0 * 1000.0)) if inked.any() else 0,
            "height_pixels": self.height_pixels,
            "inked_pixel_count": int(np.count_nonzero(inked)),
            "pixel_format": self.pixel_format,
            "raster_sha256": self.raster_sha256,
            "width_pixels": self.width_pixels,
        }


_RECEIPT_KEYS = frozenset(
    {
        "bottom_um",
        "coordinate_space",
        "height_pixels",
        "left_um",
        "pixel_format",
        "pixels_per_meter",
        "raster_sha256",
        "raw_pixel_byte_length",
        "raw_pixel_sha256",
        "row_order",
        "schema_version",
        "view",
        "width_pixels",
    }
)


def validate_paint_cutout_receipt(value: object) -> dict[str, Any]:
    receipt = _exact_keys(value, _RECEIPT_KEYS, name="paint cutout receipt")
    if receipt["coordinate_space"] != PAINT_CUTOUT_COORDINATE_SPACE:
        raise ArtifactPaintCutoutError("paint cutout receipt names another coordinate space")
    pixel_format = receipt["pixel_format"]
    if pixel_format not in PAINT_CUTOUT_TONE_FORMATS.values() or receipt["row_order"] != PAINT_CUTOUT_ROW_ORDER:
        raise ArtifactPaintCutoutError("paint cutout receipt names a pixel layout this release does not have")
    channels = next(count for count, name in PAINT_CUTOUT_PIXEL_FORMATS.items() if name == pixel_format)
    if receipt["schema_version"] != PAINT_CUTOUT_RECEIPT_SCHEMA_VERSION:
        raise ArtifactPaintCutoutError("paint cutout receipt schema is invalid")
    width = _strict_int(receipt["width_pixels"], name="width_pixels", minimum=1, maximum=MAX_PAINT_CUTOUT_PIXELS)
    height = _strict_int(receipt["height_pixels"], name="height_pixels", minimum=1, maximum=MAX_PAINT_CUTOUT_PIXELS)
    if width * height > MAX_PAINT_CUTOUT_PIXELS:
        raise ArtifactPaintCutoutError("paint cutout receipt exceeds the pixel limit")
    ppm = _strict_int(receipt["pixels_per_meter"], name="pixels_per_meter", minimum=1000, maximum=MAX_PAINT_CUTOUT_PIXELS_PER_MM * 1000)
    if ppm % 1000 != 0:
        raise ArtifactPaintCutoutError("pixels_per_meter must encode an integer pixels/mm")
    if (
        _strict_int(receipt["raw_pixel_byte_length"], name="raw_pixel_byte_length", minimum=2, maximum=4 * MAX_PAINT_CUTOUT_PIXELS)
        != channels * width * height
    ):
        raise ArtifactPaintCutoutError("paint cutout receipt byte length does not match its size and format")
    return {
        "bottom_um": _strict_int(receipt["bottom_um"], name="bottom_um", minimum=-MAX_PAINT_CUTOUT_GRID_INDEX, maximum=MAX_PAINT_CUTOUT_GRID_INDEX),
        "coordinate_space": PAINT_CUTOUT_COORDINATE_SPACE,
        "height_pixels": height,
        "left_um": _strict_int(receipt["left_um"], name="left_um", minimum=-MAX_PAINT_CUTOUT_GRID_INDEX, maximum=MAX_PAINT_CUTOUT_GRID_INDEX),
        "pixel_format": pixel_format,
        "pixels_per_meter": ppm,
        "raster_sha256": _sha256(receipt["raster_sha256"], name="raster_sha256"),
        "raw_pixel_byte_length": channels * width * height,
        "raw_pixel_sha256": _sha256(receipt["raw_pixel_sha256"], name="raw_pixel_sha256"),
        "row_order": PAINT_CUTOUT_ROW_ORDER,
        "schema_version": PAINT_CUTOUT_RECEIPT_SCHEMA_VERSION,
        "view": _view_name(receipt["view"]),
        "width_pixels": width,
    }


def paint_cutout_recipe(
    atlas: TextureAtlas,
    colour_map: ColourMap,
    *,
    view: OutlineView | str,
    source_vertex_count: int,
    source_face_count: int,
    pixels_per_mm: int = DEFAULT_PAINT_CUTOUT_PIXELS_PER_MM,
    facing_cos: float = DEFAULT_PAINT_CUTOUT_FACING_COS,
    chroma: str = DEFAULT_TEXTURE_PAINT_CHROMA,
    threshold_thousandths: int = DEFAULT_PAINT_CUTOUT_THRESHOLD_THOUSANDTHS,
    full_thousandths: int = DEFAULT_PAINT_CUTOUT_FULL_THOUSANDTHS,
    margin_um: int = DEFAULT_PAINT_CUTOUT_MARGIN_UM,
    window_mm: Sequence[float] | None = None,
    tone: str = DEFAULT_PAINT_CUTOUT_TONE,
) -> dict[str, Any]:
    """The recipe: the two files, the view, and every number that decides a pixel.

    ``texture_paint`` is the lines reader's paint block with ``band_um`` 0:
    nothing is reduced to its edge here, the cutout keeps the paint whole.
    ``window_mm`` is (left, bottom, right, top) in the view's millimetres:
    only paint inside it is cut out - the one motif wanted, not every
    painted thing the view shows - or None for the whole view.  ``tone`` is
    ink (one colour, the coverage its strength) or the paint's own colour.
    """

    if tone not in PAINT_CUTOUT_TONES:
        raise ArtifactPaintCutoutError(f"tone must be one of {', '.join(PAINT_CUTOUT_TONES)}; got {tone!r}")
    try:
        paint = texture_paint_block(atlas, colour_map, chroma=chroma, band_um=0)
    except ArtifactTexturePaintError as exc:
        raise ArtifactPaintCutoutError(str(exc)) from exc
    if isinstance(facing_cos, bool) or not isinstance(facing_cos, (int, float)) or not math.isfinite(facing_cos):
        raise ArtifactPaintCutoutError("facing_cos must be a finite number")
    threshold = _strict_int(threshold_thousandths, name="threshold_thousandths", minimum=1, maximum=999)
    full = _strict_int(full_thousandths, name="full_thousandths", minimum=2, maximum=1000)
    if full <= threshold:
        raise ArtifactPaintCutoutError("full_thousandths must be above threshold_thousandths")
    window: dict[str, int] | None = None
    if window_mm is not None:
        values = list(window_mm)
        if len(values) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values
        ):
            raise ArtifactPaintCutoutError("window_mm must be (left, bottom, right, top) finite millimetres")
        left, bottom, right, top = (int(round(float(v) * 1000.0)) for v in values)
        window = _window_block(left, bottom, right, top)
    return {
        "algorithm": PAINT_CUTOUT_ALGORITHM,
        "algorithm_version": PAINT_CUTOUT_ALGORITHM_VERSION,
        "coordinate_space": PAINT_CUTOUT_COORDINATE_SPACE,
        "ink_policy": {
            "full_thousandths": full,
            "margin_um": _strict_int(margin_um, name="margin_um", minimum=0, maximum=100_000),
            "ramp": "linear_coverage_between_thresholds/v1",
            "threshold_thousandths": threshold,
            "tone": tone,
        },
        "kind": PAINT_CUTOUT_OPERATION_KIND,
        "raster_policy": {
            "facing_cos_millionths": _strict_int(
                int(round(float(facing_cos) * 1_000_000)), name="facing_cos", minimum=50_000, maximum=1_000_000
            ),
            "painter": PAINT_CUTOUT_PAINTER,
            "pixels_per_mm": _strict_int(
                pixels_per_mm, name="pixels_per_mm", minimum=MIN_PAINT_CUTOUT_PIXELS_PER_MM, maximum=MAX_PAINT_CUTOUT_PIXELS_PER_MM
            ),
        },
        "source_face_count": _strict_int(source_face_count, name="source_face_count", minimum=1, maximum=10**9),
        "source_vertex_count": _strict_int(source_vertex_count, name="source_vertex_count", minimum=3, maximum=10**9),
        "texture_paint": paint,
        "view": _view_name(view),
        "window": window,
    }


def _window_block(left: int, bottom: int, right: int, top: int) -> dict[str, int]:
    limit = MAX_PAINT_CUTOUT_GRID_INDEX
    block = {
        "bottom_um": _strict_int(bottom, name="window bottom_um", minimum=-limit, maximum=limit),
        "left_um": _strict_int(left, name="window left_um", minimum=-limit, maximum=limit),
        "right_um": _strict_int(right, name="window right_um", minimum=-limit, maximum=limit),
        "top_um": _strict_int(top, name="window top_um", minimum=-limit, maximum=limit),
    }
    if not (block["left_um"] < block["right_um"] and block["bottom_um"] < block["top_um"]):
        raise ArtifactPaintCutoutError("window must have positive width and height")
    return block


_RECIPE_KEYS = frozenset(
    {
        "algorithm",
        "algorithm_version",
        "coordinate_space",
        "ink_policy",
        "kind",
        "raster_policy",
        "source_face_count",
        "source_vertex_count",
        "texture_paint",
        "view",
        "window",
    }
)


def validate_paint_cutout_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the recipe from its own numbers and require the same bytes."""

    block = _exact_keys(recipe, _RECIPE_KEYS, name="paint cutout recipe")
    if block["algorithm"] != PAINT_CUTOUT_ALGORITHM or block["algorithm_version"] != PAINT_CUTOUT_ALGORITHM_VERSION:
        raise ArtifactPaintCutoutError("paint cutout recipe names another algorithm")
    if block["coordinate_space"] != PAINT_CUTOUT_COORDINATE_SPACE or block["kind"] != PAINT_CUTOUT_OPERATION_KIND:
        raise ArtifactPaintCutoutError("paint cutout recipe names another coordinate space or kind")
    ink = _exact_keys(
        block["ink_policy"], frozenset({"full_thousandths", "margin_um", "ramp", "threshold_thousandths", "tone"}), name="ink_policy"
    )
    if ink["ramp"] != "linear_coverage_between_thresholds/v1":
        raise ArtifactPaintCutoutError("paint cutout recipe names a ramp this release does not have")
    if ink["tone"] not in PAINT_CUTOUT_TONES:
        raise ArtifactPaintCutoutError("paint cutout recipe names a tone this release does not have")
    raster = _exact_keys(block["raster_policy"], frozenset({"facing_cos_millionths", "painter", "pixels_per_mm"}), name="raster_policy")
    if raster["painter"] != PAINT_CUTOUT_PAINTER:
        raise ArtifactPaintCutoutError("paint cutout recipe names another painter")
    try:
        paint = validate_texture_paint_block(block["texture_paint"])
    except ArtifactTexturePaintError as exc:
        raise ArtifactPaintCutoutError(str(exc)) from exc
    if int(paint["band_um"]) != 0:
        raise ArtifactPaintCutoutError("a paint cutout keeps the paint whole: band_um must be 0")
    threshold = _strict_int(ink["threshold_thousandths"], name="threshold_thousandths", minimum=1, maximum=999)
    full = _strict_int(ink["full_thousandths"], name="full_thousandths", minimum=2, maximum=1000)
    if full <= threshold:
        raise ArtifactPaintCutoutError("full_thousandths must be above threshold_thousandths")
    raw_window = block["window"]
    window: dict[str, int] | None = None
    if raw_window is not None:
        w = _exact_keys(raw_window, frozenset({"bottom_um", "left_um", "right_um", "top_um"}), name="window")
        window = _window_block(w["left_um"], w["bottom_um"], w["right_um"], w["top_um"])
    rebuilt = {
        "algorithm": PAINT_CUTOUT_ALGORITHM,
        "algorithm_version": PAINT_CUTOUT_ALGORITHM_VERSION,
        "coordinate_space": PAINT_CUTOUT_COORDINATE_SPACE,
        "ink_policy": {
            "full_thousandths": full,
            "margin_um": _strict_int(ink["margin_um"], name="margin_um", minimum=0, maximum=100_000),
            "ramp": "linear_coverage_between_thresholds/v1",
            "threshold_thousandths": threshold,
            "tone": str(ink["tone"]),
        },
        "kind": PAINT_CUTOUT_OPERATION_KIND,
        "raster_policy": {
            "facing_cos_millionths": _strict_int(raster["facing_cos_millionths"], name="facing_cos_millionths", minimum=50_000, maximum=1_000_000),
            "painter": PAINT_CUTOUT_PAINTER,
            "pixels_per_mm": _strict_int(raster["pixels_per_mm"], name="pixels_per_mm", minimum=MIN_PAINT_CUTOUT_PIXELS_PER_MM, maximum=MAX_PAINT_CUTOUT_PIXELS_PER_MM),
        },
        "source_face_count": _strict_int(block["source_face_count"], name="source_face_count", minimum=1, maximum=10**9),
        "source_vertex_count": _strict_int(block["source_vertex_count"], name="source_vertex_count", minimum=3, maximum=10**9),
        "texture_paint": paint,
        "view": _view_name(block["view"]),
        "window": window,
    }
    try:
        if canonical_json_bytes(dict(recipe)) != canonical_json_bytes(rebuilt):
            raise ArtifactPaintCutoutError("paint cutout recipe does not match the production contract")
    except CanonicalJSONError as exc:
        raise ArtifactPaintCutoutError(str(exc)) from exc
    return rebuilt


def extract_paint_cutout(
    canonical_vertices_mm: object,
    faces: object,
    atlas: TextureAtlas,
    colour_map: ColourMap,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[PaintCutoutRaster, dict[str, Any]]:
    """Cut the paint out of the colour map as seen in the recipe's view."""

    validated = validate_paint_cutout_recipe(recipe)
    vertices = np.asarray(canonical_vertices_mm, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ArtifactPaintCutoutError("mesh must be (n, 3) vertices and (m, 3) faces")
    if int(vertices.shape[0]) != validated["source_vertex_count"] or int(triangles.shape[0]) != validated["source_face_count"]:
        raise ArtifactPaintCutoutError("mesh does not match the recipe's vertex and face counts")
    try:
        require_texture_paint_sources(validated["texture_paint"], atlas, colour_map)
    except ArtifactTexturePaintError as exc:
        raise ArtifactPaintCutoutError(str(exc)) from exc
    if triangles.shape != atlas.triangles.shape or not np.array_equal(triangles, atlas.triangles):
        raise ArtifactPaintCutoutError(
            "the mesh is not the texture atlas's geometry: its triangles differ; "
            "open the geometry the atlas welds to (write_atlas_geometry)"
        )
    raise_if_cancelled(cancellation_probe)
    frame = outline_frame(validated["view"])
    origin = np.asarray(frame.origin_world_mm, dtype=np.float64)
    u_axis = np.asarray(frame.u_axis_world, dtype=np.float64)
    v_axis = np.asarray(frame.v_axis_world, dtype=np.float64)
    toward_viewer = np.asarray(frame.normal_world, dtype=np.float64)
    pixels_per_mm = int(validated["raster_policy"]["pixels_per_mm"])
    facing_cos = validated["raster_policy"]["facing_cos_millionths"] / 1_000_000.0
    corners = vertices[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    facing = np.zeros(triangles.shape[0], dtype=np.float64)
    nonzero = lengths > 0.0
    facing[nonzero] = (normals[nonzero] @ toward_viewer) / lengths[nonzero]
    visible = np.flatnonzero(facing >= facing_cos)
    if visible.size == 0:
        raise ArtifactPaintCutoutError("no face of the mesh faces the view; nothing to cut out")
    # Far to near: the nearest wall is painted last and wins its pixels.
    depth = corners[visible].mean(axis=1) @ toward_viewer
    order = visible[np.argsort(depth, kind="stable")]
    uv = np.column_stack([(vertices - origin) @ u_axis, (vertices - origin) @ v_axis])
    used = np.unique(triangles[order])
    compact = np.full(vertices.shape[0], -1, dtype=np.int64)
    compact[used] = np.arange(used.size, dtype=np.int64)
    ink = validated["ink_policy"]
    threshold = ink["threshold_thousandths"] / 1000.0
    full = ink["full_thousandths"] / 1000.0
    in_colour = ink["tone"] == PAINT_CUTOUT_TONE_COLOUR
    try:
        paint, rgb, minimum_u, minimum_v, source_qc = texture_paint_and_colour_field(
            developed_uv_mm=uv[used],
            developed_faces=compact[triangles[order]],
            developed_points_mm=vertices[used],
            source_face_indices=order,
            source_vertex_indices=used,
            atlas=atlas,
            colour_map=colour_map,
            pixels_per_mm=pixels_per_mm,
            margin_pixels=2,
            chroma=str(validated["texture_paint"]["chroma"]),
            band_um=0,
            threshold=threshold,
            with_colour=in_colour,
            cancellation_probe=cancellation_probe,
        )
    except (ArtifactTexturePaintError, ArtifactTextureReliefError) as exc:
        raise ArtifactPaintCutoutError(str(exc)) from exc
    raise_if_cancelled(cancellation_probe)
    covered = np.isfinite(paint)
    coverage = np.where(covered, np.clip((paint - threshold) / (full - threshold), 0.0, 1.0), 0.0)
    window = validated["window"]
    if window is not None:
        # Pixel centres: column c is at (minimum_u + c + 0.5) / ppm mm.
        cols_mm = (minimum_u + np.arange(coverage.shape[1]) + 0.5) / pixels_per_mm
        rows_mm = (minimum_v + np.arange(coverage.shape[0]) + 0.5) / pixels_per_mm
        inside = (
            (cols_mm[None, :] >= window["left_um"] / 1000.0)
            & (cols_mm[None, :] <= window["right_um"] / 1000.0)
            & (rows_mm[:, None] >= window["bottom_um"] / 1000.0)
            & (rows_mm[:, None] <= window["top_um"] / 1000.0)
        )
        coverage = np.where(inside, coverage, 0.0)
    inked = coverage > 0.0
    if not inked.any():
        raise ArtifactPaintCutoutError(
            "nothing is painted in this view above the threshold; lower threshold_thousandths, "
            "widen the window, choose another view, or do not take the cutout"
        )
    rows = np.flatnonzero(inked.any(axis=1))
    cols = np.flatnonzero(inked.any(axis=0))
    margin = int(round(ink["margin_um"] / 1000.0 * pixels_per_mm))
    row0, row1 = max(0, int(rows[0]) - margin), min(coverage.shape[0], int(rows[-1]) + margin + 1)
    col0, col1 = max(0, int(cols[0]) - margin), min(coverage.shape[1], int(cols[-1]) + margin + 1)
    alpha = np.rint(coverage[row0:row1, col0:col1] * 255.0).astype(np.uint8)
    if in_colour and rgb is not None:
        # The paint's own colour where there is paint; clear pixels carry no colour.
        pixels = np.zeros((row1 - row0, col1 - col0, 4), dtype=np.uint8)
        pixels[..., :3] = np.where(alpha[..., None] > 0, rgb[row0:row1, col0:col1], 0)
        pixels[..., 3] = alpha
    else:
        pixels = np.zeros((row1 - row0, col1 - col0, 2), dtype=np.uint8)
        pixels[..., 1] = alpha
    # The lattice's row 0 is the lowest v; the raster's row 0 is the top.
    pixels = np.ascontiguousarray(pixels[::-1])
    raster = PaintCutoutRaster(
        pixels=pixels,
        pixels_per_meter=pixels_per_mm * 1000,
        left_um=int(round((minimum_u + col0) / pixels_per_mm * 1000.0)),
        bottom_um=int(round((minimum_v + row0) / pixels_per_mm * 1000.0)),
        view=validated["view"],
    )
    qc = {
        **raster.qc_summary(),
        "texture_paint_covered_pixel_count": source_qc["texture_paint_covered_pixel_count"],
        "texture_paint_painted_pixel_count": source_qc["texture_paint_painted_pixel_count"],
        "view": validated["view"],
        "visible_face_count": int(visible.size),
    }
    return raster, qc


@dataclass(frozen=True, slots=True)
class PaintCutoutComputation:
    context: OperationContext
    projection_snapshot: Any
    raster: PaintCutoutRaster
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]


def compute_paint_cutout(
    session: ArtifactSession,
    atlas: TextureAtlas,
    colour_map: ColourMap,
    *,
    view: OutlineView | str,
    pixels_per_mm: int = DEFAULT_PAINT_CUTOUT_PIXELS_PER_MM,
    facing_cos: float = DEFAULT_PAINT_CUTOUT_FACING_COS,
    chroma: str = DEFAULT_TEXTURE_PAINT_CHROMA,
    threshold_thousandths: int = DEFAULT_PAINT_CUTOUT_THRESHOLD_THOUSANDTHS,
    full_thousandths: int = DEFAULT_PAINT_CUTOUT_FULL_THOUSANDTHS,
    margin_um: int = DEFAULT_PAINT_CUTOUT_MARGIN_UM,
    window_mm: Sequence[float] | None = None,
    tone: str = DEFAULT_PAINT_CUTOUT_TONE,
    cancellation_probe: CancellationProbe | None = None,
) -> PaintCutoutComputation:
    """Cut the paint out as positioned by the session's active Align."""

    if not isinstance(session, ArtifactSession):
        raise ArtifactPaintCutoutError("session must be an ArtifactSession")
    if not isinstance(atlas, TextureAtlas) or not isinstance(colour_map, ColourMap):
        raise ArtifactPaintCutoutError("atlas and colour_map must be TextureAtlas and ColourMap")
    try:
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactPaintCutoutError(str(exc)) from exc
    vertices = np.asarray(projection.mesh.vertices, dtype=np.float64)
    triangles = np.asarray(projection.mesh.faces, dtype=np.int64)
    recipe = paint_cutout_recipe(
        atlas,
        colour_map,
        view=view,
        source_vertex_count=int(vertices.shape[0]),
        source_face_count=int(triangles.shape[0]),
        pixels_per_mm=pixels_per_mm,
        facing_cos=facing_cos,
        chroma=chroma,
        threshold_thousandths=threshold_thousandths,
        full_thousandths=full_thousandths,
        margin_um=margin_um,
        window_mm=window_mm,
        tone=tone,
    )
    raster, qc = extract_paint_cutout(vertices, triangles, atlas, colour_map, recipe, cancellation_probe=cancellation_probe)
    try:
        context = session.capture_operation(recipe=recipe, selection_hash=raster.raster_sha256)
    except ArtifactSessionError as exc:
        raise ArtifactPaintCutoutError(str(exc)) from exc
    return PaintCutoutComputation(
        context=context, projection_snapshot=projection.snapshot, raster=raster, recipe=recipe, qc=qc
    )


def paint_cutout_computation_matches_active_projection(session: ArtifactSession, computation: PaintCutoutComputation) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(computation, PaintCutoutComputation):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    return current.render_key == computation.projection_snapshot.render_key


def commit_paint_cutout(
    session: ArtifactSession,
    computation: PaintCutoutComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    """Append the cutout's receipt as a record; the pixels travel beside it."""

    if not paint_cutout_computation_matches_active_projection(session, computation):
        raise ArtifactPaintCutoutError("paint cutout computation is stale for the active projection")
    validated_recipe = validate_paint_cutout_recipe(computation.recipe)
    raster = computation.raster
    if computation.context.selection_hash != raster.raster_sha256:
        raise ArtifactPaintCutoutError("paint cutout context selection_hash does not match the raster")
    extensions = {PAINT_CUTOUT_PAYLOAD_EXTENSION_KEY: raster.receipt()}
    try:
        document = session.document.append_record_from_context(
            context=computation.context,
            id=record_id,
            type=PAINT_CUTOUT_RECORD_TYPE,
            geometry_ref=raster.geometry_ref,
            recipe=dict(validated_recipe),
            qc=dict(computation.qc),
            lifecycle_status=RecordLifecycleStatus.READY,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=depends_on_record_ids,
            extensions=extensions,
        )
    except ArtifactDocumentError as exc:
        raise ArtifactPaintCutoutError(str(exc)) from exc
    return session.with_document(document)


def paint_cutout_receipt_from_record(record: DerivedRecord) -> dict[str, Any]:
    """Resolve and re-verify one cutout record's receipt."""

    if not isinstance(record, DerivedRecord):
        raise ArtifactPaintCutoutError("record must be a DerivedRecord")
    if record.type != PAINT_CUTOUT_RECORD_TYPE:
        raise ArtifactPaintCutoutError(f"record is not a paint cutout: {record.type!r}")
    receipt = validate_paint_cutout_receipt(record.extensions.get(PAINT_CUTOUT_PAYLOAD_EXTENSION_KEY))
    if record.geometry_ref != f"{PAINT_CUTOUT_GEOMETRY_REF_PREFIX}{receipt['raster_sha256']}":
        raise ArtifactPaintCutoutError("paint cutout record geometry_ref does not match its receipt")
    recipe = validate_paint_cutout_recipe(record.recipe)
    if receipt["view"] != recipe["view"]:
        raise ArtifactPaintCutoutError("paint cutout receipt and recipe name different views")
    if receipt["pixels_per_meter"] != int(recipe["raster_policy"]["pixels_per_mm"]) * 1000:
        raise ArtifactPaintCutoutError("paint cutout receipt and recipe name different resolutions")
    if receipt["pixel_format"] != PAINT_CUTOUT_TONE_FORMATS[str(recipe["ink_policy"]["tone"])]:
        raise ArtifactPaintCutoutError("paint cutout receipt's pixel format is not its recipe's tone")
    return receipt


def paint_cutout_tone(recipe: Mapping[str, Any]) -> str:
    """The tone a validated cutout recipe asks for: ink or the paint's colour."""

    return str(validate_paint_cutout_recipe(recipe)["ink_policy"]["tone"])


def require_paint_cutout_raster(record: DerivedRecord, raster: object) -> PaintCutoutRaster:
    """The pixels at hand must be the pixels the record's receipt proves."""

    receipt = paint_cutout_receipt_from_record(record)
    if not isinstance(raster, PaintCutoutRaster):
        raise ArtifactPaintCutoutError(f"record {record.id!r} needs its PaintCutoutRaster at hand")
    if raster.raster_sha256 != receipt["raster_sha256"]:
        raise ArtifactPaintCutoutError(
            f"the cutout raster at hand is not the one record {record.id!r} proves "
            f"(receipt {receipt['raster_sha256'][:12]}..., raster {raster.raster_sha256[:12]}...)"
        )
    return raster


def validate_paint_cutout_records(document: ArtifactDocument) -> None:
    """Strictly validate every cutout receipt embedded in a document."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactPaintCutoutError("document must be an ArtifactDocument")
    for record in document.records:
        if record.type == PAINT_CUTOUT_RECORD_TYPE:
            paint_cutout_receipt_from_record(record)


__all__ = [
    "ArtifactPaintCutoutError",
    "DEFAULT_PAINT_CUTOUT_FULL_THOUSANDTHS",
    "DEFAULT_PAINT_CUTOUT_MARGIN_UM",
    "DEFAULT_PAINT_CUTOUT_PIXELS_PER_MM",
    "DEFAULT_PAINT_CUTOUT_THRESHOLD_THOUSANDTHS",
    "DEFAULT_PAINT_CUTOUT_TONE",
    "PAINT_CUTOUT_PAYLOAD_EXTENSION_KEY",
    "PAINT_CUTOUT_PIXEL_FORMAT_COLOUR",
    "PAINT_CUTOUT_PIXEL_FORMAT_INK",
    "PAINT_CUTOUT_RECORD_TYPE",
    "PAINT_CUTOUT_TONES",
    "PAINT_CUTOUT_TONE_COLOUR",
    "PAINT_CUTOUT_TONE_INK",
    "PaintCutoutComputation",
    "PaintCutoutRaster",
    "commit_paint_cutout",
    "compute_paint_cutout",
    "extract_paint_cutout",
    "paint_cutout_computation_matches_active_projection",
    "paint_cutout_receipt_from_record",
    "paint_cutout_recipe",
    "paint_cutout_tone",
    "require_paint_cutout_raster",
    "validate_paint_cutout_receipt",
    "validate_paint_cutout_recipe",
    "validate_paint_cutout_records",
]
