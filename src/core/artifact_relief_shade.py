"""양각 음영: the relief of a vessel's wall as a drafter sees it from the
side, lit from the upper left, kept as a shade the sheet stipples.

A lotus petal carved in relief on a celadon dish is not a line.  The
drafter shows its bulk by stippling: countless small dots, denser where
the surface turns away from the light, none where it faces it, so the
petal stands out of the wall on paper as it does in the hand.  This record
reads the shade those dots follow, and only the shade; how the dots fall
is the sheet's business, at the sheet's scale.

The reading is on one of the four side views.  The wall is rasterised as
the front-most depth at every pixel of the view's lattice; the surface of
revolution the relief stands on - the radius the wall has at every row,
read off the same raster as the median radius the near wall's depth
implies - is taken away, so what is left is the relief alone.  Two
filters follow: a wide one takes out what is slower than any motif (an
oval rim, a scan's slight lean), a narrow one the scan's grain.  The
relief is then lit as a height field, the slope across the view corrected
for the wall's foreshortening round the axis, and where a slope faces
away from the light the shade is that much darker than a flat wall's.
The record keeps the receipt of that raster - one grey channel that is
always black, and the alpha as the darkness - and the pixels travel beside
it, as a rubbing's do.
"""

from __future__ import annotations

import hashlib
import math
import warnings
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
from .artifact_rubbing_extractor import ArtifactRubbingError, _rasterize_depth_field
from .artifact_session import ArtifactSession, ArtifactSessionError
from .canonical_json import CanonicalJSONError, canonical_json_bytes

RELIEF_SHADE_RECORD_TYPE = "measurement.relief_shade.v1"
RELIEF_SHADE_OPERATION_KIND = "relief_shade"
RELIEF_SHADE_ALGORITHM = "archmeshrubbing.view_relief_shade"
RELIEF_SHADE_ALGORITHM_VERSION = "1.0.0"
RELIEF_SHADE_COORDINATE_SPACE = "view_plane_mm/v1"
RELIEF_SHADE_PAYLOAD_EXTENSION_KEY = "org.archmeshrubbing:relief-shade-v1"
RELIEF_SHADE_RECEIPT_SCHEMA_VERSION = "1.0.0"
RELIEF_SHADE_GEOMETRY_REF_PREFIX = "urn:archmeshrubbing:relief-shade:sha256:"
RELIEF_SHADE_PIXEL_FORMAT = "gray8_alpha8_shade/v1"
RELIEF_SHADE_ROW_ORDER = "top_row_first/v1"
RELIEF_SHADE_PAINTER = "front_most_depth/v1"
RELIEF_SHADE_BASE_MODEL = "revolution_row_median/v1"
RELIEF_SHADE_LIGHT_MODEL = "lambert_height_field/v1"
RELIEF_SHADE_FORESHORTENING = "arc_cos_corrected/v1"
#: The base is a surface of revolution about the canonical axis, which the
#: four side views hold as their v axis; a plan view has no such base.
RELIEF_SHADE_VIEWS: tuple[str, ...] = ("front", "back", "left", "right")

DEFAULT_RELIEF_SHADE_PIXELS_PER_MM = 10
DEFAULT_RELIEF_SHADE_FACING_COS = 0.05
DEFAULT_RELIEF_SHADE_MARGIN_UM = 1_000
#: The base radius of a row is read where the wall faces the viewer
#: squarely: within this fraction of the row's silhouette half-width.
DEFAULT_RELIEF_SHADE_INNER_FRACTION_THOUSANDTHS = 600
#: Beyond this fraction of the silhouette the wall is seen edge-on and the
#: depth says nothing about relief.
DEFAULT_RELIEF_SHADE_GRAZING_FRACTION_THOUSANDTHS = 920
DEFAULT_RELIEF_SHADE_ROW_SMOOTHING_UM = 300
#: A pixel further than this from the base is not relief: the far wall
#: seen over a rim that is not level, a stray triangle.
DEFAULT_RELIEF_SHADE_OUTLIER_UM = 10_000
#: What is slower than any motif is not relief either.
DEFAULT_RELIEF_SHADE_SLOW_UM = 8_000
#: What is finer than the scan can carry is its grain.
DEFAULT_RELIEF_SHADE_GRAIN_UM = 400
#: Light from the upper left, raised: (across, up, towards the viewer).
DEFAULT_RELIEF_SHADE_LIGHT_THOUSANDTHS: tuple[int, int, int] = (-1000, 1000, 1200)
DEFAULT_RELIEF_SHADE_GAIN_THOUSANDTHS = 2000
#: A shade fainter than this is not a shade: a mesh facet, a scan's grain.
DEFAULT_RELIEF_SHADE_FLOOR_THOUSANDTHS = 60
#: The hollows: the ground between raised motifs lies in their shadow, and
#: a hand stipples it darker the deeper it lies.  Off by default; a depth
#: in micrometres at which the ground is fully dark, and how dark.
DEFAULT_RELIEF_SHADE_CAVITY_UM = 0
DEFAULT_RELIEF_SHADE_CAVITY_GAIN_THOUSANDTHS = 1000
DEFAULT_RELIEF_SHADE_EDGE_EROSION_PIXELS = 2
RELIEF_SHADE_LAYER_SEPARATION_UM = 50
MIN_RELIEF_SHADE_PIXELS_PER_MM = 1
MAX_RELIEF_SHADE_PIXELS_PER_MM = 50
MAX_RELIEF_SHADE_PIXELS = 20_000_000
MAX_RELIEF_SHADE_GRID_INDEX = 10**9


class ArtifactReliefShadeError(ValueError):
    """A relief shade could not be read, recorded, or trusted."""


def _strict_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactReliefShadeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ArtifactReliefShadeError(f"{name} must be from {minimum} to {maximum}")
    return int(value)


def _exact_keys(value: object, keys: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactReliefShadeError(f"{name} must be a mapping")
    if set(value.keys()) != keys:
        raise ArtifactReliefShadeError(f"{name} keys must be exactly {sorted(keys)}")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ArtifactReliefShadeError(f"{name} must be a lowercase sha256 hex digest")
    return value


def _view_name(view: object) -> str:
    name = view.value if isinstance(view, OutlineView) else view
    if not isinstance(name, str) or name not in RELIEF_SHADE_VIEWS:
        raise ArtifactReliefShadeError(
            f"relief shade view must be one of {', '.join(RELIEF_SHADE_VIEWS)}; got {name!r}"
        )
    return name


@dataclass(frozen=True, slots=True)
class ReliefShadeRaster:
    """The shade on a view's plane: grey 0 with the alpha the darkness,
    row 0 the top of the view, and where its left and bottom edges lie in
    the view's millimetres."""

    pixels: np.ndarray
    pixels_per_meter: int
    left_um: int
    bottom_um: int
    view: str

    def __post_init__(self) -> None:
        array = np.asarray(self.pixels)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ArtifactReliefShadeError("shade pixels must be a non-empty HxWx2 uint8 array")
        if array.shape[0] * array.shape[1] > MAX_RELIEF_SHADE_PIXELS:
            raise ArtifactReliefShadeError("shade pixels exceed the safety limit")
        ppm = _strict_int(
            self.pixels_per_meter, name="pixels_per_meter", minimum=1000, maximum=MAX_RELIEF_SHADE_PIXELS_PER_MM * 1000
        )
        if ppm % 1000 != 0:
            raise ArtifactReliefShadeError("pixels_per_meter must encode an integer pixels/mm")
        for name, value in (("left_um", self.left_um), ("bottom_um", self.bottom_um)):
            _strict_int(value, name=name, minimum=-MAX_RELIEF_SHADE_GRID_INDEX, maximum=MAX_RELIEF_SHADE_GRID_INDEX)
        copied = np.ascontiguousarray(array).copy()
        copied.setflags(write=False)
        object.__setattr__(self, "pixels", copied)
        object.__setattr__(self, "pixels_per_meter", ppm)
        object.__setattr__(self, "view", _view_name(self.view))

    @property
    def darkness(self) -> np.ndarray:
        return self.pixels[..., 1]

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
            "coordinate_space": RELIEF_SHADE_COORDINATE_SPACE,
            "height_pixels": self.height_pixels,
            "left_um": int(self.left_um),
            "pixel_format": RELIEF_SHADE_PIXEL_FORMAT,
            "pixels_per_meter": self.pixels_per_meter,
            "row_order": RELIEF_SHADE_ROW_ORDER,
            "schema_version": RELIEF_SHADE_RECEIPT_SCHEMA_VERSION,
            "view": self.view,
            "width_pixels": self.width_pixels,
        }

    @property
    def raster_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"archmeshrubbing.relief-shade-raster\0")
        digest.update(canonical_json_bytes(self.semantic_header()))
        digest.update(b"\0")
        digest.update(self.pixels.tobytes(order="C"))
        return digest.hexdigest()

    @property
    def geometry_ref(self) -> str:
        return f"{RELIEF_SHADE_GEOMETRY_REF_PREFIX}{self.raster_sha256}"

    def receipt(self) -> dict[str, Any]:
        return {
            **self.semantic_header(),
            "raster_sha256": self.raster_sha256,
            "raw_pixel_byte_length": int(self.pixels.nbytes),
            "raw_pixel_sha256": hashlib.sha256(self.pixels.tobytes(order="C")).hexdigest(),
        }

    def qc_summary(self) -> dict[str, Any]:
        darkness = self.darkness.astype(np.float64)
        shaded = darkness > 0
        return {
            "darkness_mean_thousandths": int(round(float(darkness[shaded].mean()) / 255.0 * 1000.0)) if shaded.any() else 0,
            "height_pixels": self.height_pixels,
            "raster_sha256": self.raster_sha256,
            "shaded_pixel_count": int(np.count_nonzero(shaded)),
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


def validate_relief_shade_receipt(value: object) -> dict[str, Any]:
    receipt = _exact_keys(value, _RECEIPT_KEYS, name="relief shade receipt")
    if receipt["coordinate_space"] != RELIEF_SHADE_COORDINATE_SPACE:
        raise ArtifactReliefShadeError("relief shade receipt names another coordinate space")
    if receipt["pixel_format"] != RELIEF_SHADE_PIXEL_FORMAT or receipt["row_order"] != RELIEF_SHADE_ROW_ORDER:
        raise ArtifactReliefShadeError("relief shade receipt names a pixel layout this release does not have")
    if receipt["schema_version"] != RELIEF_SHADE_RECEIPT_SCHEMA_VERSION:
        raise ArtifactReliefShadeError("relief shade receipt schema is invalid")
    width = _strict_int(receipt["width_pixels"], name="width_pixels", minimum=1, maximum=MAX_RELIEF_SHADE_PIXELS)
    height = _strict_int(receipt["height_pixels"], name="height_pixels", minimum=1, maximum=MAX_RELIEF_SHADE_PIXELS)
    if width * height > MAX_RELIEF_SHADE_PIXELS:
        raise ArtifactReliefShadeError("relief shade receipt exceeds the pixel limit")
    ppm = _strict_int(receipt["pixels_per_meter"], name="pixels_per_meter", minimum=1000, maximum=MAX_RELIEF_SHADE_PIXELS_PER_MM * 1000)
    if ppm % 1000 != 0:
        raise ArtifactReliefShadeError("pixels_per_meter must encode an integer pixels/mm")
    if (
        _strict_int(receipt["raw_pixel_byte_length"], name="raw_pixel_byte_length", minimum=2, maximum=2 * MAX_RELIEF_SHADE_PIXELS)
        != 2 * width * height
    ):
        raise ArtifactReliefShadeError("relief shade receipt byte length does not match its size")
    return {
        "bottom_um": _strict_int(receipt["bottom_um"], name="bottom_um", minimum=-MAX_RELIEF_SHADE_GRID_INDEX, maximum=MAX_RELIEF_SHADE_GRID_INDEX),
        "coordinate_space": RELIEF_SHADE_COORDINATE_SPACE,
        "height_pixels": height,
        "left_um": _strict_int(receipt["left_um"], name="left_um", minimum=-MAX_RELIEF_SHADE_GRID_INDEX, maximum=MAX_RELIEF_SHADE_GRID_INDEX),
        "pixel_format": RELIEF_SHADE_PIXEL_FORMAT,
        "pixels_per_meter": ppm,
        "raster_sha256": _sha256(receipt["raster_sha256"], name="raster_sha256"),
        "raw_pixel_byte_length": 2 * width * height,
        "raw_pixel_sha256": _sha256(receipt["raw_pixel_sha256"], name="raw_pixel_sha256"),
        "row_order": RELIEF_SHADE_ROW_ORDER,
        "schema_version": RELIEF_SHADE_RECEIPT_SCHEMA_VERSION,
        "view": _view_name(receipt["view"]),
        "width_pixels": width,
    }


def _window_block(left: int, bottom: int, right: int, top: int) -> dict[str, int]:
    limit = MAX_RELIEF_SHADE_GRID_INDEX
    block = {
        "bottom_um": _strict_int(bottom, name="window bottom_um", minimum=-limit, maximum=limit),
        "left_um": _strict_int(left, name="window left_um", minimum=-limit, maximum=limit),
        "right_um": _strict_int(right, name="window right_um", minimum=-limit, maximum=limit),
        "top_um": _strict_int(top, name="window top_um", minimum=-limit, maximum=limit),
    }
    if not (block["left_um"] < block["right_um"] and block["bottom_um"] < block["top_um"]):
        raise ArtifactReliefShadeError("window must have positive width and height")
    return block


def _light_block(light: Sequence[int]) -> list[int]:
    values = list(light)
    if len(values) != 3:
        raise ArtifactReliefShadeError("light_thousandths must be (across, up, towards the viewer)")
    out = [_strict_int(v, name="light_thousandths", minimum=-10_000, maximum=10_000) for v in values]
    if out[2] <= 0:
        raise ArtifactReliefShadeError("the light must come from the viewer's side of the wall (towards > 0)")
    return out


def relief_shade_recipe(
    *,
    view: OutlineView | str,
    source_vertex_count: int,
    source_face_count: int,
    pixels_per_mm: int = DEFAULT_RELIEF_SHADE_PIXELS_PER_MM,
    facing_cos: float = DEFAULT_RELIEF_SHADE_FACING_COS,
    margin_um: int = DEFAULT_RELIEF_SHADE_MARGIN_UM,
    inner_fraction_thousandths: int = DEFAULT_RELIEF_SHADE_INNER_FRACTION_THOUSANDTHS,
    grazing_fraction_thousandths: int = DEFAULT_RELIEF_SHADE_GRAZING_FRACTION_THOUSANDTHS,
    row_smoothing_um: int = DEFAULT_RELIEF_SHADE_ROW_SMOOTHING_UM,
    outlier_um: int = DEFAULT_RELIEF_SHADE_OUTLIER_UM,
    slow_um: int = DEFAULT_RELIEF_SHADE_SLOW_UM,
    grain_um: int = DEFAULT_RELIEF_SHADE_GRAIN_UM,
    light_thousandths: Sequence[int] = DEFAULT_RELIEF_SHADE_LIGHT_THOUSANDTHS,
    gain_thousandths: int = DEFAULT_RELIEF_SHADE_GAIN_THOUSANDTHS,
    floor_thousandths: int = DEFAULT_RELIEF_SHADE_FLOOR_THOUSANDTHS,
    cavity_um: int = DEFAULT_RELIEF_SHADE_CAVITY_UM,
    cavity_gain_thousandths: int = DEFAULT_RELIEF_SHADE_CAVITY_GAIN_THOUSANDTHS,
    edge_erosion_pixels: int = DEFAULT_RELIEF_SHADE_EDGE_EROSION_PIXELS,
    window_mm: Sequence[float] | None = None,
) -> dict[str, Any]:
    """The recipe: the view and every number that decides a pixel.

    ``window_mm`` is (left, bottom, right, top) in the view's millimetres:
    only relief inside it is shaded - the band the motif occupies, not the
    lip's underside or the foot's root, which the line work already draws
    - or None for the whole view.
    """

    if isinstance(facing_cos, bool) or not isinstance(facing_cos, (int, float)) or not math.isfinite(facing_cos):
        raise ArtifactReliefShadeError("facing_cos must be a finite number")
    window: dict[str, int] | None = None
    if window_mm is not None:
        values = list(window_mm)
        if len(values) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values
        ):
            raise ArtifactReliefShadeError("window_mm must be (left, bottom, right, top) finite millimetres")
        left, bottom, right, top = (int(round(float(v) * 1000.0)) for v in values)
        window = _window_block(left, bottom, right, top)
    inner = _strict_int(inner_fraction_thousandths, name="inner_fraction_thousandths", minimum=100, maximum=1000)
    grazing = _strict_int(grazing_fraction_thousandths, name="grazing_fraction_thousandths", minimum=100, maximum=1000)
    if grazing < inner:
        raise ArtifactReliefShadeError("grazing_fraction_thousandths must not be below inner_fraction_thousandths")
    slow = _strict_int(slow_um, name="slow_um", minimum=0, maximum=1_000_000)
    grain = _strict_int(grain_um, name="grain_um", minimum=0, maximum=100_000)
    if slow and grain and slow <= grain:
        raise ArtifactReliefShadeError("slow_um must be wider than grain_um")
    return {
        "algorithm": RELIEF_SHADE_ALGORITHM,
        "algorithm_version": RELIEF_SHADE_ALGORITHM_VERSION,
        "base_policy": {
            "grazing_fraction_thousandths": grazing,
            "inner_fraction_thousandths": inner,
            "model": RELIEF_SHADE_BASE_MODEL,
            "outlier_um": _strict_int(outlier_um, name="outlier_um", minimum=1, maximum=1_000_000),
            "row_smoothing_um": _strict_int(row_smoothing_um, name="row_smoothing_um", minimum=0, maximum=100_000),
        },
        "coordinate_space": RELIEF_SHADE_COORDINATE_SPACE,
        "kind": RELIEF_SHADE_OPERATION_KIND,
        "raster_policy": {
            "facing_cos_millionths": _strict_int(
                int(round(float(facing_cos) * 1_000_000)), name="facing_cos", minimum=0, maximum=1_000_000
            ),
            "layer_separation_um": RELIEF_SHADE_LAYER_SEPARATION_UM,
            "margin_um": _strict_int(margin_um, name="margin_um", minimum=0, maximum=100_000),
            "painter": RELIEF_SHADE_PAINTER,
            "pixels_per_mm": _strict_int(
                pixels_per_mm, name="pixels_per_mm", minimum=MIN_RELIEF_SHADE_PIXELS_PER_MM, maximum=MAX_RELIEF_SHADE_PIXELS_PER_MM
            ),
        },
        "relief_policy": {"grain_um": grain, "slow_um": slow},
        "shade_policy": {
            "cavity_gain_thousandths": _strict_int(cavity_gain_thousandths, name="cavity_gain_thousandths", minimum=0, maximum=100_000),
            "cavity_um": _strict_int(cavity_um, name="cavity_um", minimum=0, maximum=100_000),
            "edge_erosion_pixels": _strict_int(edge_erosion_pixels, name="edge_erosion_pixels", minimum=0, maximum=100),
            "floor_thousandths": _strict_int(floor_thousandths, name="floor_thousandths", minimum=0, maximum=999),
            "foreshortening": RELIEF_SHADE_FORESHORTENING,
            "gain_thousandths": _strict_int(gain_thousandths, name="gain_thousandths", minimum=1, maximum=100_000),
            "light_thousandths": _light_block(light_thousandths),
            "model": RELIEF_SHADE_LIGHT_MODEL,
        },
        "source_face_count": _strict_int(source_face_count, name="source_face_count", minimum=1, maximum=10**9),
        "source_vertex_count": _strict_int(source_vertex_count, name="source_vertex_count", minimum=3, maximum=10**9),
        "view": _view_name(view),
        "window": window,
    }


_RECIPE_KEYS = frozenset(
    {
        "algorithm",
        "algorithm_version",
        "base_policy",
        "coordinate_space",
        "kind",
        "raster_policy",
        "relief_policy",
        "shade_policy",
        "source_face_count",
        "source_vertex_count",
        "view",
        "window",
    }
)


def validate_relief_shade_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the recipe from its own numbers and require the same bytes."""

    block = _exact_keys(recipe, _RECIPE_KEYS, name="relief shade recipe")
    if block["algorithm"] != RELIEF_SHADE_ALGORITHM or block["algorithm_version"] != RELIEF_SHADE_ALGORITHM_VERSION:
        raise ArtifactReliefShadeError("relief shade recipe names another algorithm")
    if block["coordinate_space"] != RELIEF_SHADE_COORDINATE_SPACE or block["kind"] != RELIEF_SHADE_OPERATION_KIND:
        raise ArtifactReliefShadeError("relief shade recipe names another coordinate space or kind")
    raster = _exact_keys(
        block["raster_policy"],
        frozenset({"facing_cos_millionths", "layer_separation_um", "margin_um", "painter", "pixels_per_mm"}),
        name="raster_policy",
    )
    if raster["painter"] != RELIEF_SHADE_PAINTER or raster["layer_separation_um"] != RELIEF_SHADE_LAYER_SEPARATION_UM:
        raise ArtifactReliefShadeError("relief shade recipe names another painter")
    base = _exact_keys(
        block["base_policy"],
        frozenset({"grazing_fraction_thousandths", "inner_fraction_thousandths", "model", "outlier_um", "row_smoothing_um"}),
        name="base_policy",
    )
    if base["model"] != RELIEF_SHADE_BASE_MODEL:
        raise ArtifactReliefShadeError("relief shade recipe names a base this release does not have")
    relief = _exact_keys(block["relief_policy"], frozenset({"grain_um", "slow_um"}), name="relief_policy")
    shade = _exact_keys(
        block["shade_policy"],
        frozenset({"cavity_gain_thousandths", "cavity_um", "edge_erosion_pixels", "floor_thousandths", "foreshortening", "gain_thousandths", "light_thousandths", "model"}),
        name="shade_policy",
    )
    if shade["model"] != RELIEF_SHADE_LIGHT_MODEL or shade["foreshortening"] != RELIEF_SHADE_FORESHORTENING:
        raise ArtifactReliefShadeError("relief shade recipe names a light this release does not have")
    raw_window = block["window"]
    window_mm: list[float] | None = None
    if raw_window is not None:
        w = _exact_keys(raw_window, frozenset({"bottom_um", "left_um", "right_um", "top_um"}), name="window")
        checked = _window_block(w["left_um"], w["bottom_um"], w["right_um"], w["top_um"])
        window_mm = [checked["left_um"] / 1000.0, checked["bottom_um"] / 1000.0, checked["right_um"] / 1000.0, checked["top_um"] / 1000.0]
    facing = _strict_int(raster["facing_cos_millionths"], name="facing_cos_millionths", minimum=0, maximum=1_000_000)
    light = shade["light_thousandths"]
    if not isinstance(light, Sequence) or isinstance(light, str):
        raise ArtifactReliefShadeError("light_thousandths must be a list of three integers")
    rebuilt = relief_shade_recipe(
        view=block["view"],
        source_vertex_count=block["source_vertex_count"],
        source_face_count=block["source_face_count"],
        pixels_per_mm=raster["pixels_per_mm"],
        facing_cos=facing / 1_000_000.0,
        margin_um=raster["margin_um"],
        inner_fraction_thousandths=base["inner_fraction_thousandths"],
        grazing_fraction_thousandths=base["grazing_fraction_thousandths"],
        row_smoothing_um=base["row_smoothing_um"],
        outlier_um=base["outlier_um"],
        slow_um=relief["slow_um"],
        grain_um=relief["grain_um"],
        light_thousandths=light,
        gain_thousandths=shade["gain_thousandths"],
        floor_thousandths=shade["floor_thousandths"],
        cavity_um=shade["cavity_um"],
        cavity_gain_thousandths=shade["cavity_gain_thousandths"],
        edge_erosion_pixels=shade["edge_erosion_pixels"],
        window_mm=window_mm,
    )
    try:
        if canonical_json_bytes(dict(recipe)) != canonical_json_bytes(rebuilt):
            raise ArtifactReliefShadeError("relief shade recipe does not match the production contract")
    except CanonicalJSONError as exc:
        raise ArtifactReliefShadeError(str(exc)) from exc
    return rebuilt


def _masked_blur(field: np.ndarray, ok: np.ndarray, sigma_pixels: float) -> np.ndarray:
    """A Gaussian blur that does not let the uncovered pixels pull the
    covered ones towards zero: the blur of the field over the blur of the
    mask."""

    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    if sigma_pixels <= 0.0:
        return np.where(ok, field, 0.0)
    weight = gaussian_filter(ok.astype(np.float64), sigma_pixels)
    smoothed = gaussian_filter(np.where(ok, field, 0.0), sigma_pixels)
    return np.where(weight > 1e-3, smoothed / np.maximum(weight, 1e-3), 0.0)


def extract_relief_shade(
    canonical_vertices_mm: object,
    faces: object,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[ReliefShadeRaster, dict[str, Any]]:
    """Read the relief's shade in the recipe's view."""

    from scipy.ndimage import binary_erosion, gaussian_filter1d  # noqa: PLC0415

    validated = validate_relief_shade_recipe(recipe)
    vertices = np.asarray(canonical_vertices_mm, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ArtifactReliefShadeError("mesh must be (n, 3) vertices and (m, 3) faces")
    if int(vertices.shape[0]) != validated["source_vertex_count"] or int(triangles.shape[0]) != validated["source_face_count"]:
        raise ArtifactReliefShadeError("mesh does not match the recipe's vertex and face counts")
    raise_if_cancelled(cancellation_probe)
    frame = outline_frame(validated["view"])
    origin = np.asarray(frame.origin_world_mm, dtype=np.float64)
    u_axis = np.asarray(frame.u_axis_world, dtype=np.float64)
    v_axis = np.asarray(frame.v_axis_world, dtype=np.float64)
    toward_viewer = np.asarray(frame.normal_world, dtype=np.float64)
    raster_policy = validated["raster_policy"]
    pixels_per_mm = int(raster_policy["pixels_per_mm"])
    facing_cos = raster_policy["facing_cos_millionths"] / 1_000_000.0
    corners = vertices[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    facing = np.zeros(triangles.shape[0], dtype=np.float64)
    nonzero = lengths > 0.0
    facing[nonzero] = (normals[nonzero] @ toward_viewer) / lengths[nonzero]
    visible = np.flatnonzero(facing >= facing_cos)
    if visible.size == 0:
        raise ArtifactReliefShadeError("no face of the mesh faces the view; nothing to shade")
    relative = vertices - origin
    projected = np.column_stack([relative @ u_axis, relative @ v_axis])
    depths = relative @ toward_viewer
    try:
        depth, minimum_u, minimum_v, raster_qc = _rasterize_depth_field(
            projected,
            depths,
            triangles[visible],
            pixels_per_mm=pixels_per_mm,
            margin_pixels=int(round(raster_policy["margin_um"] / 1000.0 * pixels_per_mm)),
            layer_separation_mm=raster_policy["layer_separation_um"] / 1000.0,
            cancellation_probe=cancellation_probe,
        )
    except ArtifactRubbingError as exc:
        raise ArtifactReliefShadeError(str(exc)) from exc
    raise_if_cancelled(cancellation_probe)
    # The depth lattice's row 0 is its lowest v; the shade's row 0 will be the top.
    height, width = depth.shape
    xs = (minimum_u + np.arange(width) + 0.5) / pixels_per_mm
    covered = np.isfinite(depth) & (depth > 0.0)
    if not covered.any():
        raise ArtifactReliefShadeError("the view sees no wall on the viewer's side of the axis; nothing to shade")

    # The base: the radius that explains the near wall's depth at each row,
    # read where the wall faces the viewer squarely.
    base_policy = validated["base_policy"]
    inner = base_policy["inner_fraction_thousandths"] / 1000.0
    grazing = base_policy["grazing_fraction_thousandths"] / 1000.0
    abs_x = np.abs(xs)[None, :]
    silhouette = np.where(covered, abs_x, 0.0).max(axis=1)
    depth_near = np.where(covered, depth, np.nan)
    with np.errstate(invalid="ignore"):
        implied = np.sqrt(depth_near**2 + abs_x**2)
    with warnings.catch_warnings():
        # A row with no squarely seen pixel is all NaN, and is interpolated below.
        warnings.simplefilter("ignore", RuntimeWarning)
        row_radius = np.nanmedian(np.where(abs_x <= inner * silhouette[:, None], implied, np.nan), axis=1)
    known = np.isfinite(row_radius)
    if not known.any():
        raise ArtifactReliefShadeError("no row of the view shows the wall squarely enough to read its base radius")
    rows = np.arange(height, dtype=np.float64)
    row_radius = np.interp(rows, rows[known], row_radius[known])
    smoothing_pixels = base_policy["row_smoothing_um"] / 1000.0 * pixels_per_mm
    if smoothing_pixels > 0.0:
        row_radius = gaussian_filter1d(row_radius, smoothing_pixels)
    base = np.sqrt(np.maximum(row_radius[:, None] ** 2 - abs_x**2, 0.0))
    relief = np.where(covered, depth, 0.0) - base
    ok = covered & (abs_x <= grazing * row_radius[:, None]) & (np.abs(relief) <= base_policy["outlier_um"] / 1000.0)
    window = validated["window"]
    if window is not None:
        cols_mm = xs
        rows_mm = (minimum_v + np.arange(height) + 0.5) / pixels_per_mm
        ok &= (
            (cols_mm[None, :] >= window["left_um"] / 1000.0)
            & (cols_mm[None, :] <= window["right_um"] / 1000.0)
            & (rows_mm[:, None] >= window["bottom_um"] / 1000.0)
            & (rows_mm[:, None] <= window["top_um"] / 1000.0)
        )
    if not ok.any():
        raise ArtifactReliefShadeError(
            "no relief lies where the recipe looks; widen the window, choose another view, or do not take the shade"
        )
    raise_if_cancelled(cancellation_probe)
    relief_policy = validated["relief_policy"]
    grain = _masked_blur(relief, ok, relief_policy["grain_um"] / 1000.0 * pixels_per_mm)
    slow = _masked_blur(relief, ok, relief_policy["slow_um"] / 1000.0 * pixels_per_mm) if relief_policy["slow_um"] else 0.0
    filtered = np.where(ok, grain - slow, 0.0)
    raise_if_cancelled(cancellation_probe)

    # The light on a height field whose normal is (-dh/dx, -dh/dy, 1); the
    # slope across the view is what the wall's turn round the axis makes it.
    shade_policy = validated["shade_policy"]
    gradient_x = np.zeros_like(filtered)
    gradient_y = np.zeros_like(filtered)
    gradient_x[:, 1:-1] = (filtered[:, 2:] - filtered[:, :-2]) * pixels_per_mm / 2.0
    gradient_y[1:-1, :] = (filtered[2:, :] - filtered[:-2, :]) * pixels_per_mm / 2.0
    cos_round = np.sqrt(np.clip(1.0 - (abs_x / np.maximum(row_radius[:, None], 1e-6)) ** 2, 0.0, 1.0))
    gradient_x *= cos_round
    normal_x, normal_y, normal_z = -gradient_x, -gradient_y, np.ones_like(filtered)
    norm = np.sqrt(normal_x**2 + normal_y**2 + 1.0)
    light = np.asarray(shade_policy["light_thousandths"], dtype=np.float64) / 1000.0
    light /= np.linalg.norm(light)
    lit = (normal_x * light[0] + normal_y * light[1] + normal_z * light[2]) / norm
    flat = light[2]
    gain = shade_policy["gain_thousandths"] / 1000.0
    darkness = np.clip((flat - lit) / flat * gain, 0.0, 1.0)
    cavity = shade_policy["cavity_um"] / 1000.0
    if cavity > 0.0:
        # The ground between the motifs, in their shadow: darker the deeper.
        hollow = np.clip(-filtered / cavity, 0.0, 1.0) * (shade_policy["cavity_gain_thousandths"] / 1000.0)
        darkness = np.clip(darkness + hollow, 0.0, 1.0)
    darkness = np.where(darkness >= shade_policy["floor_thousandths"] / 1000.0, darkness, 0.0)
    # The blur that took out the grain leans on nothing past the covered
    # edge, and a slope appears there that the wall does not have: no pixel
    # nearer the edge than twice the grain's width is trusted, and never
    # fewer than the recipe's own erosion.
    trusted = ok
    erosion = max(
        int(shade_policy["edge_erosion_pixels"]),
        int(math.ceil(2.0 * relief_policy["grain_um"] / 1000.0 * pixels_per_mm)),
    )
    if erosion > 0:
        trusted = binary_erosion(ok, iterations=erosion)
    darkness = np.where(trusted, darkness, 0.0)
    shaded = darkness > 0.0
    if not shaded.any():
        raise ArtifactReliefShadeError(
            "the wall is flat to the light where the recipe looks; nothing to shade - raise the gain or do not take the shade"
        )
    raise_if_cancelled(cancellation_probe)
    row_index = np.flatnonzero(shaded.any(axis=1))
    col_index = np.flatnonzero(shaded.any(axis=0))
    margin = int(round(raster_policy["margin_um"] / 1000.0 * pixels_per_mm))
    row0, row1 = max(0, int(row_index[0]) - margin), min(height, int(row_index[-1]) + margin + 1)
    col0, col1 = max(0, int(col_index[0]) - margin), min(width, int(col_index[-1]) + margin + 1)
    alpha = np.rint(darkness[row0:row1, col0:col1] * 255.0).astype(np.uint8)
    pixels = np.zeros((row1 - row0, col1 - col0, 2), dtype=np.uint8)
    pixels[..., 1] = alpha
    # The lattice's row 0 is the lowest v; the raster's row 0 is the top.
    pixels = np.ascontiguousarray(pixels[::-1])
    raster = ReliefShadeRaster(
        pixels=pixels,
        pixels_per_meter=pixels_per_mm * 1000,
        left_um=int(round((minimum_u + col0) / pixels_per_mm * 1000.0)),
        bottom_um=int(round((minimum_v + row0) / pixels_per_mm * 1000.0)),
        view=validated["view"],
    )
    percentiles = np.percentile(filtered[ok], [5, 50, 95])
    qc = {
        **raster.qc_summary(),
        "base_radius_max_um": int(round(float(row_radius.max()) * 1000.0)),
        "base_radius_min_um": int(round(float(row_radius.min()) * 1000.0)),
        "covered_pixel_count": int(raster_qc["covered_pixel_count"]),
        "relief_max_um": int(round(float(filtered[ok].max()) * 1000.0)),
        "relief_min_um": int(round(float(filtered[ok].min()) * 1000.0)),
        "relief_p05_um": int(round(float(percentiles[0]) * 1000.0)),
        "relief_p50_um": int(round(float(percentiles[1]) * 1000.0)),
        "relief_p95_um": int(round(float(percentiles[2]) * 1000.0)),
        "trusted_pixel_count": int(np.count_nonzero(trusted)),
        "view": validated["view"],
        "visible_face_count": int(visible.size),
    }
    return raster, qc


@dataclass(frozen=True, slots=True)
class ReliefShadeComputation:
    context: OperationContext
    projection_snapshot: Any
    raster: ReliefShadeRaster
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]


def compute_relief_shade(
    session: ArtifactSession,
    *,
    view: OutlineView | str,
    cancellation_probe: CancellationProbe | None = None,
    **recipe_options: Any,
) -> ReliefShadeComputation:
    """Read the shade as positioned by the session's active Align."""

    if not isinstance(session, ArtifactSession):
        raise ArtifactReliefShadeError("session must be an ArtifactSession")
    try:
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactReliefShadeError(str(exc)) from exc
    vertices = np.asarray(projection.mesh.vertices, dtype=np.float64)
    triangles = np.asarray(projection.mesh.faces, dtype=np.int64)
    recipe = relief_shade_recipe(
        view=view,
        source_vertex_count=int(vertices.shape[0]),
        source_face_count=int(triangles.shape[0]),
        **recipe_options,
    )
    raster, qc = extract_relief_shade(vertices, triangles, recipe, cancellation_probe=cancellation_probe)
    try:
        context = session.capture_operation(recipe=recipe, selection_hash=raster.raster_sha256)
    except ArtifactSessionError as exc:
        raise ArtifactReliefShadeError(str(exc)) from exc
    return ReliefShadeComputation(
        context=context, projection_snapshot=projection.snapshot, raster=raster, recipe=recipe, qc=qc
    )


def relief_shade_computation_matches_active_projection(session: ArtifactSession, computation: ReliefShadeComputation) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(computation, ReliefShadeComputation):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    return current.render_key == computation.projection_snapshot.render_key


def commit_relief_shade(
    session: ArtifactSession,
    computation: ReliefShadeComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    """Append the shade's receipt as a record; the pixels travel beside it."""

    if not relief_shade_computation_matches_active_projection(session, computation):
        raise ArtifactReliefShadeError("relief shade computation is stale for the active projection")
    validated_recipe = validate_relief_shade_recipe(computation.recipe)
    raster = computation.raster
    if computation.context.selection_hash != raster.raster_sha256:
        raise ArtifactReliefShadeError("relief shade context selection_hash does not match the raster")
    extensions = {RELIEF_SHADE_PAYLOAD_EXTENSION_KEY: raster.receipt()}
    try:
        document = session.document.append_record_from_context(
            context=computation.context,
            id=record_id,
            type=RELIEF_SHADE_RECORD_TYPE,
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
        raise ArtifactReliefShadeError(str(exc)) from exc
    return session.with_document(document)


def relief_shade_receipt_from_record(record: DerivedRecord) -> dict[str, Any]:
    """Resolve and re-verify one shade record's receipt."""

    if not isinstance(record, DerivedRecord):
        raise ArtifactReliefShadeError("record must be a DerivedRecord")
    if record.type != RELIEF_SHADE_RECORD_TYPE:
        raise ArtifactReliefShadeError(f"record is not a relief shade: {record.type!r}")
    receipt = validate_relief_shade_receipt(record.extensions.get(RELIEF_SHADE_PAYLOAD_EXTENSION_KEY))
    if record.geometry_ref != f"{RELIEF_SHADE_GEOMETRY_REF_PREFIX}{receipt['raster_sha256']}":
        raise ArtifactReliefShadeError("relief shade record geometry_ref does not match its receipt")
    recipe = validate_relief_shade_recipe(record.recipe)
    if receipt["view"] != recipe["view"]:
        raise ArtifactReliefShadeError("relief shade receipt and recipe name different views")
    if receipt["pixels_per_meter"] != int(recipe["raster_policy"]["pixels_per_mm"]) * 1000:
        raise ArtifactReliefShadeError("relief shade receipt and recipe name different resolutions")
    return receipt


def require_relief_shade_raster(record: DerivedRecord, raster: object) -> ReliefShadeRaster:
    """The pixels at hand must be the pixels the record's receipt proves."""

    receipt = relief_shade_receipt_from_record(record)
    if not isinstance(raster, ReliefShadeRaster):
        raise ArtifactReliefShadeError(f"record {record.id!r} needs its ReliefShadeRaster at hand")
    if raster.raster_sha256 != receipt["raster_sha256"]:
        raise ArtifactReliefShadeError(
            f"the shade raster at hand is not the one record {record.id!r} proves "
            f"(receipt {receipt['raster_sha256'][:12]}..., raster {raster.raster_sha256[:12]}...)"
        )
    return raster


def validate_relief_shade_records(document: ArtifactDocument) -> None:
    """Strictly validate every shade receipt embedded in a document."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactReliefShadeError("document must be an ArtifactDocument")
    for record in document.records:
        if record.type == RELIEF_SHADE_RECORD_TYPE:
            relief_shade_receipt_from_record(record)


__all__ = [
    "ArtifactReliefShadeError",
    "DEFAULT_RELIEF_SHADE_CAVITY_GAIN_THOUSANDTHS",
    "DEFAULT_RELIEF_SHADE_CAVITY_UM",
    "DEFAULT_RELIEF_SHADE_FLOOR_THOUSANDTHS",
    "DEFAULT_RELIEF_SHADE_GAIN_THOUSANDTHS",
    "DEFAULT_RELIEF_SHADE_GRAIN_UM",
    "DEFAULT_RELIEF_SHADE_LIGHT_THOUSANDTHS",
    "DEFAULT_RELIEF_SHADE_PIXELS_PER_MM",
    "DEFAULT_RELIEF_SHADE_SLOW_UM",
    "RELIEF_SHADE_PAYLOAD_EXTENSION_KEY",
    "RELIEF_SHADE_PIXEL_FORMAT",
    "RELIEF_SHADE_RECORD_TYPE",
    "RELIEF_SHADE_VIEWS",
    "ReliefShadeComputation",
    "ReliefShadeRaster",
    "commit_relief_shade",
    "compute_relief_shade",
    "extract_relief_shade",
    "relief_shade_computation_matches_active_projection",
    "relief_shade_receipt_from_record",
    "relief_shade_recipe",
    "require_relief_shade_raster",
    "validate_relief_shade_receipt",
    "validate_relief_shade_recipe",
    "validate_relief_shade_records",
]
