"""요철 음영: the relief of a vessel's wall as a drafter sees it from the
side, lit from the upper left, kept as a shade the sheet stipples.

A lotus petal carved in relief on a celadon dish is not a line.  The
drafter shows its bulk by stippling: countless small dots, denser where
the surface turns away from the light, none where it faces it, so the
petal stands out of the wall on paper as it does in the hand.  This record
reads the shade those dots follow, and only the shade; how the dots fall
is the sheet's business, at the sheet's scale.

The reading does not ask which way the surface departs from the wall.  A
pot pressed with rows of dots shades under it exactly as a raised petal
does - a slope is a slope to the light - so what is read is 요철, relief
and hollow together, and nothing here says which one an artifact carries.

The reading is on one of the four side views.  The wall is rasterised as
the front-most depth at every pixel of the view's lattice; the surface of
revolution the relief stands on - the radius the wall has at every row,
read off the same raster as the median radius the near wall's depth
implies - is taken away, so what is left is the relief alone.  Two
filters follow: a running median across the view, a few motifs wide,
takes out what runs round the vessel slower than any motif (an oval rim,
a scan's slight lean) without the halo a blur would leave round each
motif; a narrow blur takes out the scan's grain.  The
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
RELIEF_SHADE_ALGORITHM_VERSION = "1.2.0"
RELIEF_SHADE_COORDINATE_SPACE = "view_plane_mm/v1"
RELIEF_SHADE_PAYLOAD_EXTENSION_KEY = "org.archmeshrubbing:relief-shade-v1"
RELIEF_SHADE_RECEIPT_SCHEMA_VERSION = "1.0.0"
RELIEF_SHADE_GEOMETRY_REF_PREFIX = "urn:archmeshrubbing:relief-shade:sha256:"
RELIEF_SHADE_PIXEL_FORMAT = "gray8_alpha8_shade/v1"
RELIEF_SHADE_ROW_ORDER = "top_row_first/v1"
RELIEF_SHADE_PAINTER = "front_most_depth/v1"
RELIEF_SHADE_BASE_MODEL = "revolution_row_median/v1"
#: A plan view's base.  Looking down the axis, the drawing plane is the
#: artifact's own x-y, so a pixel's distance from the centre is its radius
#: on the artifact and a body of revolution has one height on each ring.
#: The base is that ring's median height, and the relief is how far the
#: surface stands above it - which is what a rubbing of a lid's top reads.
RELIEF_SHADE_RING_BASE_MODEL = "revolution_ring_median/v1"
#: How far round the wall a view pixel is, is measured against the
#: silhouette on its own side of the row, not the row's median radius: a
#: warped vessel stands past its median on the wide side, and measuring
#: there against the median cut the whole outer band off as grazing.
RELIEF_SHADE_SILHOUETTE = "each_side_of_row/v1"
RELIEF_SHADE_LIGHT_MODEL = "lambert_height_field/v1"
#: 탁본의 농담.  The other way to turn a relief into ink, and the one a
#: rubbing actually uses: the paper is pressed onto the surface, the dabber
#: inks whatever the paper lies on, and the ink falls away with how far the
#: surface sits below it.  A light says which way a slope faces, so a
#: stamped mark shows one flank; contact says what the paper touched, so
#: the whole mark reads - the ground dark, the impression pale.  The
#: numbers are the rubbing extractor's, measured on the corded and grooved
#: profile: the paper bridges 0.7 mm, the ink is gone 0.12 mm below it,
#: and 70% contact ink leaves a plain wall dark grey with the relief still
#: readable inside it.
RELIEF_SHADE_CONTACT_MODEL = "contact_envelope_ink/v1"
RELIEF_SHADE_SHADE_MODELS: tuple[str, ...] = (RELIEF_SHADE_LIGHT_MODEL, RELIEF_SHADE_CONTACT_MODEL)
DEFAULT_RELIEF_SHADE_PAPER_UM = 700
DEFAULT_RELIEF_SHADE_BLACK_POINT_UM = 120
DEFAULT_RELIEF_SHADE_CONTACT_INK_THOUSANDTHS = 700
RELIEF_SHADE_FORESHORTENING = "arc_cos_corrected/v1"
#: The base is a surface of revolution about the canonical axis, which the
#: four side views hold as their v axis.  A plan view holds it as its
#: normal instead, so its base is read by ring rather than by row, and it
#: is a domain of its own.
RELIEF_SHADE_VIEWS: tuple[str, ...] = ("front", "back", "left", "right")
RELIEF_SHADE_PLAN_VIEWS: tuple[str, ...] = ("top", "bottom")
#: Where the shade is read.  A side view shows the petals a viewer sees,
#: three or four of them, foreshortened towards the silhouette; the axis
#: development unrolls the wall about its axis so every petal round the
#: vessel lies flat on one strip, the way a drafter draws a band of
#: ornament as a 전개도.  On the development u is the arc r(z) * theta
#: from a seam meridian and v the meridian arc length up the wall, both
#: read off the outside's own median profile, so the strip is undistorted
#: and the relief is the radius above that profile.
RELIEF_SHADE_DOMAIN_VIEW = "view/v1"
RELIEF_SHADE_DOMAIN_DEVELOPMENT = "axis_development/v1"
#: A lid is drawn plan over elevation and its top is rubbed there, the
#: knob left out - the drafter's rule for a lid.  The plan domain reads
#: that top: the plan view's own millimetres, the artifact's own profile
#: read ring by ring as the base, and a ring window that takes the knob
#: out and stops before the edge turns down.
RELIEF_SHADE_DOMAIN_PLAN = "axis_plan/v1"
RELIEF_SHADE_DOMAINS: tuple[str, ...] = (
    RELIEF_SHADE_DOMAIN_VIEW,
    RELIEF_SHADE_DOMAIN_DEVELOPMENT,
    RELIEF_SHADE_DOMAIN_PLAN,
)
RELIEF_SHADE_DEVELOPMENT_LABEL = "development"
RELIEF_SHADE_DEVELOPMENT_SPACE = "axis_development_mm/v1"
RELIEF_SHADE_DEVELOPMENT_DIRECTION = "counterclockwise_from_seam/v1"
#: A flaring wall's true development is a fan; a drafter's strip is
#: straight, the way a band of ornament is read, so the strip is the wall
#: seen as a cylinder of one reference radius - the profile's radius at
#: the middle of the window's height - and the pattern keeps its
#: verticals while its widths scale with the wall's own radius over that.
RELIEF_SHADE_DEVELOPMENT_MODEL = "cylindrical_at_window_middle/v1"
#: The seam: where the strip is cut.  The back of the front view - the
#: meridian at +90 degrees - so the front the elevation shows lies whole
#: in the middle of the strip.
DEFAULT_RELIEF_SHADE_SEAM_MILLIDEG = 90_000
DEFAULT_RELIEF_SHADE_PROFILE_BIN_UM = 250
#: A face whose normal leaves the axis by at least this cosine is the
#: outer wall; the rim's top, the floor and the foot's underside are not.
DEFAULT_RELIEF_SHADE_OUTWARD_COS_THOUSANDTHS = 250

DEFAULT_RELIEF_SHADE_PIXELS_PER_MM = 10
DEFAULT_RELIEF_SHADE_FACING_COS = 0.05
DEFAULT_RELIEF_SHADE_MARGIN_UM = 1_000
#: The base radius of a row is read where the wall faces the viewer
#: squarely: within this fraction of the row's silhouette half-width.
DEFAULT_RELIEF_SHADE_INNER_FRACTION_THOUSANDTHS = 600
#: Beyond this fraction of the silhouette the wall is seen edge-on and the
#: depth says nothing about relief.
DEFAULT_RELIEF_SHADE_GRAZING_FRACTION_THOUSANDTHS = 980
#: How far up and down the wall the slow level is blended, so the row-wise
#: median does not print as streaks: a millimetre, well under a motif.
DEFAULT_RELIEF_SHADE_SLOW_BLEND_UM = 1_000
DEFAULT_RELIEF_SHADE_ROW_SMOOTHING_UM = 300
#: A pixel further than this from the base is not relief: the far wall
#: seen over a rim that is not level, a stray triangle.
DEFAULT_RELIEF_SHADE_OUTLIER_UM = 10_000
#: What runs round the vessel slower than any motif is not relief either:
#: an oval rim or a lean varies once round the circumference, a motif in a
#: hand's breadth.  The width sits between the two.
DEFAULT_RELIEF_SHADE_SLOW_UM = 40_000
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
# How many bands a development's rows are reported in.  The strip's rows are
# meridian arc and a sheet that pastes the strip on an elevation needs the
# height each row came from, and how far the wall stood from the axis there:
# sixteen bands keep the error inside a band second order on any profile a
# strip is cut from, as they do for a rubbing's artboard.
RELIEF_SHADE_DEVELOPMENT_PROFILE_BANDS = 16
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


def _view_name(view: object, *, allowed: tuple[str, ...] = RELIEF_SHADE_VIEWS) -> str:
    name = view.value if isinstance(view, OutlineView) else view
    if not isinstance(name, str) or name not in allowed:
        raise ArtifactReliefShadeError(
            f"relief shade view must be one of {', '.join(allowed)}; got {name!r}"
        )
    return name


def _space_label(value: object) -> str:
    """A raster's label: the view it was read in, or ``development``."""

    name = value.value if isinstance(value, OutlineView) else value
    if isinstance(name, str) and name == RELIEF_SHADE_DEVELOPMENT_LABEL:
        return name
    return _view_name(name, allowed=RELIEF_SHADE_VIEWS + RELIEF_SHADE_PLAN_VIEWS)


def _shade_model(name: object) -> str:
    if not isinstance(name, str) or name not in RELIEF_SHADE_SHADE_MODELS:
        raise ArtifactReliefShadeError(
            f"shade_model must be one of {', '.join(RELIEF_SHADE_SHADE_MODELS)}; got {name!r}"
        )
    return name


def _coordinate_space(label: str) -> str:
    return RELIEF_SHADE_DEVELOPMENT_SPACE if label == RELIEF_SHADE_DEVELOPMENT_LABEL else RELIEF_SHADE_COORDINATE_SPACE


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
        object.__setattr__(self, "view", _space_label(self.view))

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

    @property
    def is_development(self) -> bool:
        return self.view == RELIEF_SHADE_DEVELOPMENT_LABEL

    def semantic_header(self) -> dict[str, Any]:
        return {
            "bottom_um": int(self.bottom_um),
            "coordinate_space": _coordinate_space(self.view),
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
    label = _space_label(receipt["view"])
    if receipt["coordinate_space"] != _coordinate_space(label):
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
        "coordinate_space": _coordinate_space(label),
        "height_pixels": height,
        "left_um": _strict_int(receipt["left_um"], name="left_um", minimum=-MAX_RELIEF_SHADE_GRID_INDEX, maximum=MAX_RELIEF_SHADE_GRID_INDEX),
        "pixel_format": RELIEF_SHADE_PIXEL_FORMAT,
        "pixels_per_meter": ppm,
        "raster_sha256": _sha256(receipt["raster_sha256"], name="raster_sha256"),
        "raw_pixel_byte_length": 2 * width * height,
        "raw_pixel_sha256": _sha256(receipt["raw_pixel_sha256"], name="raw_pixel_sha256"),
        "row_order": RELIEF_SHADE_ROW_ORDER,
        "schema_version": RELIEF_SHADE_RECEIPT_SCHEMA_VERSION,
        "view": label,
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
    source_vertex_count: int,
    source_face_count: int,
    view: OutlineView | str | None = None,
    domain: str = RELIEF_SHADE_DOMAIN_VIEW,
    seam_millideg: int = DEFAULT_RELIEF_SHADE_SEAM_MILLIDEG,
    profile_bin_um: int = DEFAULT_RELIEF_SHADE_PROFILE_BIN_UM,
    outward_cos_thousandths: int = DEFAULT_RELIEF_SHADE_OUTWARD_COS_THOUSANDTHS,
    pixels_per_mm: int = DEFAULT_RELIEF_SHADE_PIXELS_PER_MM,
    facing_cos: float = DEFAULT_RELIEF_SHADE_FACING_COS,
    margin_um: int = DEFAULT_RELIEF_SHADE_MARGIN_UM,
    inner_fraction_thousandths: int = DEFAULT_RELIEF_SHADE_INNER_FRACTION_THOUSANDTHS,
    grazing_fraction_thousandths: int = DEFAULT_RELIEF_SHADE_GRAZING_FRACTION_THOUSANDTHS,
    row_smoothing_um: int = DEFAULT_RELIEF_SHADE_ROW_SMOOTHING_UM,
    outlier_um: int = DEFAULT_RELIEF_SHADE_OUTLIER_UM,
    slow_um: int = DEFAULT_RELIEF_SHADE_SLOW_UM,
    slow_blend_um: int = DEFAULT_RELIEF_SHADE_SLOW_BLEND_UM,
    grain_um: int = DEFAULT_RELIEF_SHADE_GRAIN_UM,
    light_thousandths: Sequence[int] = DEFAULT_RELIEF_SHADE_LIGHT_THOUSANDTHS,
    gain_thousandths: int = DEFAULT_RELIEF_SHADE_GAIN_THOUSANDTHS,
    floor_thousandths: int = DEFAULT_RELIEF_SHADE_FLOOR_THOUSANDTHS,
    cavity_um: int = DEFAULT_RELIEF_SHADE_CAVITY_UM,
    cavity_gain_thousandths: int = DEFAULT_RELIEF_SHADE_CAVITY_GAIN_THOUSANDTHS,
    edge_erosion_pixels: int = DEFAULT_RELIEF_SHADE_EDGE_EROSION_PIXELS,
    window_mm: Sequence[float] | None = None,
    ring_window_mm: Sequence[float] | None = None,
    shade_model: str = RELIEF_SHADE_LIGHT_MODEL,
    paper_um: int = DEFAULT_RELIEF_SHADE_PAPER_UM,
    black_point_um: int = DEFAULT_RELIEF_SHADE_BLACK_POINT_UM,
    contact_ink_thousandths: int = DEFAULT_RELIEF_SHADE_CONTACT_INK_THOUSANDTHS,
) -> dict[str, Any]:
    """The recipe: where the shade is read and every number that decides a pixel.

    ``domain`` is a side ``view`` (then ``view`` names it) or the axis
    development (then ``view`` is None and ``seam_millideg`` says where
    the strip is cut, ``profile_bin_um`` how finely the outside's profile
    is read, ``outward_cos_thousandths`` which faces are the outer wall).
    ``window_mm`` is (left, bottom, right, top): in a view, the view's
    millimetres; on the development, left and right along the strip and
    bottom and top as heights on the artifact - only relief inside it is
    shaded, the band the motif occupies, not the lip's underside or the
    foot's root, which the line work already draws - or None for all.
    """

    if domain not in RELIEF_SHADE_DOMAINS:
        raise ArtifactReliefShadeError(f"domain must be one of {', '.join(RELIEF_SHADE_DOMAINS)}; got {domain!r}")
    on_plan = domain == RELIEF_SHADE_DOMAIN_PLAN
    if ring_window_mm is not None and not on_plan:
        raise ArtifactReliefShadeError(
            "a ring window is a band of radius about the axis, which only a plan view has; "
            "use window_mm in a view or on the development"
        )
    plan: dict[str, Any] | None = None
    if on_plan:
        inner_um, outer_um = 0, 0
        if ring_window_mm is not None:
            values = list(ring_window_mm)
            if len(values) != 2 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0.0
                for v in values
            ):
                raise ArtifactReliefShadeError("ring_window_mm must be (inner, outer) finite millimetres, not negative")
            inner_um, outer_um = (int(round(float(v) * 1000.0)) for v in values)
            if outer_um and outer_um <= inner_um:
                raise ArtifactReliefShadeError("ring_window_mm's outer radius must be beyond its inner one")
        plan = {
            "model": RELIEF_SHADE_RING_BASE_MODEL,
            "ring_inner_um": _strict_int(inner_um, name="ring_window inner", minimum=0, maximum=10**9),
            "ring_outer_um": _strict_int(outer_um, name="ring_window outer", minimum=0, maximum=10**9),
        }
    development: dict[str, Any] | None = None
    if domain == RELIEF_SHADE_DOMAIN_DEVELOPMENT:
        if view is not None:
            raise ArtifactReliefShadeError("a development shade is read round the axis, not in a view; leave view None")
        development = {
            "direction": RELIEF_SHADE_DEVELOPMENT_DIRECTION,
            "model": RELIEF_SHADE_DEVELOPMENT_MODEL,
            "outward_cos_thousandths": _strict_int(outward_cos_thousandths, name="outward_cos_thousandths", minimum=50, maximum=1000),
            "profile_bin_um": _strict_int(profile_bin_um, name="profile_bin_um", minimum=10, maximum=10_000),
            "seam_millideg": _strict_int(seam_millideg, name="seam_millideg", minimum=-180_000, maximum=180_000),
        }
    elif view is None:
        raise ArtifactReliefShadeError("a view shade needs its view")
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
    slow_blend = _strict_int(slow_blend_um, name="slow_blend_um", minimum=0, maximum=100_000)
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
            "silhouette": RELIEF_SHADE_SILHOUETTE,
            "outlier_um": _strict_int(outlier_um, name="outlier_um", minimum=1, maximum=1_000_000),
            "row_smoothing_um": _strict_int(row_smoothing_um, name="row_smoothing_um", minimum=0, maximum=100_000),
        },
        "coordinate_space": _coordinate_space(RELIEF_SHADE_DEVELOPMENT_LABEL if development else "front"),
        "development_policy": development,
        "plan_policy": plan,
        "domain": domain,
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
        "relief_policy": {"grain_um": grain, "slow_blend_um": slow_blend, "slow_um": slow},
        "shade_policy": {
            "black_point_um": _strict_int(black_point_um, name="black_point_um", minimum=1, maximum=100_000),
            "contact_ink_thousandths": _strict_int(
                contact_ink_thousandths, name="contact_ink_thousandths", minimum=1, maximum=1000
            ),
            "paper_um": _strict_int(paper_um, name="paper_um", minimum=0, maximum=100_000),
            "cavity_gain_thousandths": _strict_int(cavity_gain_thousandths, name="cavity_gain_thousandths", minimum=0, maximum=100_000),
            "cavity_um": _strict_int(cavity_um, name="cavity_um", minimum=0, maximum=100_000),
            "edge_erosion_pixels": _strict_int(edge_erosion_pixels, name="edge_erosion_pixels", minimum=0, maximum=100),
            "floor_thousandths": _strict_int(floor_thousandths, name="floor_thousandths", minimum=0, maximum=999),
            "foreshortening": RELIEF_SHADE_FORESHORTENING,
            "gain_thousandths": _strict_int(gain_thousandths, name="gain_thousandths", minimum=1, maximum=100_000),
            "light_thousandths": _light_block(light_thousandths),
            "model": _shade_model(shade_model),
        },
        "source_face_count": _strict_int(source_face_count, name="source_face_count", minimum=1, maximum=10**9),
        "source_vertex_count": _strict_int(source_vertex_count, name="source_vertex_count", minimum=3, maximum=10**9),
        "view": None
        if development
        else _view_name(view, allowed=RELIEF_SHADE_PLAN_VIEWS if on_plan else RELIEF_SHADE_VIEWS),
        "window": window,
    }


_RECIPE_KEYS = frozenset(
    {
        "algorithm",
        "algorithm_version",
        "base_policy",
        "coordinate_space",
        "development_policy",
        "domain",
        "kind",
        "plan_policy",
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
    if block["kind"] != RELIEF_SHADE_OPERATION_KIND or block["domain"] not in RELIEF_SHADE_DOMAINS:
        raise ArtifactReliefShadeError("relief shade recipe names another kind or domain")
    plan = block["plan_policy"]
    plan_kwargs: dict[str, Any] = {}
    if plan is not None:
        if block["domain"] != RELIEF_SHADE_DOMAIN_PLAN:
            raise ArtifactReliefShadeError("only a plan shade carries a plan policy")
        checked_plan = _exact_keys(
            plan, frozenset({"model", "ring_inner_um", "ring_outer_um"}), name="plan_policy"
        )
        if checked_plan["model"] != RELIEF_SHADE_RING_BASE_MODEL:
            raise ArtifactReliefShadeError("relief shade recipe names a plan base this release does not have")
        if checked_plan["ring_inner_um"] or checked_plan["ring_outer_um"]:
            plan_kwargs = {
                "ring_window_mm": (
                    checked_plan["ring_inner_um"] / 1000.0,
                    checked_plan["ring_outer_um"] / 1000.0,
                )
            }
    elif block["domain"] == RELIEF_SHADE_DOMAIN_PLAN:
        raise ArtifactReliefShadeError("a plan shade needs its plan policy")
    development = block["development_policy"]
    dev_kwargs: dict[str, Any] = {}
    if development is not None:
        dev = _exact_keys(
            development, frozenset({"direction", "model", "outward_cos_thousandths", "profile_bin_um", "seam_millideg"}), name="development_policy"
        )
        if dev["direction"] != RELIEF_SHADE_DEVELOPMENT_DIRECTION or dev["model"] != RELIEF_SHADE_DEVELOPMENT_MODEL:
            raise ArtifactReliefShadeError("relief shade recipe names a development this release does not have")
        dev_kwargs = {
            "seam_millideg": dev["seam_millideg"],
            "profile_bin_um": dev["profile_bin_um"],
            "outward_cos_thousandths": dev["outward_cos_thousandths"],
        }
    raster = _exact_keys(
        block["raster_policy"],
        frozenset({"facing_cos_millionths", "layer_separation_um", "margin_um", "painter", "pixels_per_mm"}),
        name="raster_policy",
    )
    if raster["painter"] != RELIEF_SHADE_PAINTER or raster["layer_separation_um"] != RELIEF_SHADE_LAYER_SEPARATION_UM:
        raise ArtifactReliefShadeError("relief shade recipe names another painter")
    base = _exact_keys(
        block["base_policy"],
        frozenset({"grazing_fraction_thousandths", "inner_fraction_thousandths", "model", "outlier_um", "row_smoothing_um", "silhouette"}),
        name="base_policy",
    )
    if base["model"] != RELIEF_SHADE_BASE_MODEL or base["silhouette"] != RELIEF_SHADE_SILHOUETTE:
        raise ArtifactReliefShadeError("relief shade recipe names a base this release does not have")
    relief = _exact_keys(block["relief_policy"], frozenset({"grain_um", "slow_blend_um", "slow_um"}), name="relief_policy")
    shade = _exact_keys(
        block["shade_policy"],
        frozenset({"black_point_um", "cavity_gain_thousandths", "cavity_um", "contact_ink_thousandths", "edge_erosion_pixels", "floor_thousandths", "foreshortening", "gain_thousandths", "light_thousandths", "model", "paper_um"}),
        name="shade_policy",
    )
    if shade["model"] not in RELIEF_SHADE_SHADE_MODELS or shade["foreshortening"] != RELIEF_SHADE_FORESHORTENING:
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
        domain=block["domain"],
        **dev_kwargs,
        **plan_kwargs,
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
        slow_blend_um=relief["slow_blend_um"],
        grain_um=relief["grain_um"],
        light_thousandths=light,
        gain_thousandths=shade["gain_thousandths"],
        floor_thousandths=shade["floor_thousandths"],
        shade_model=shade["model"],
        paper_um=shade["paper_um"],
        black_point_um=shade["black_point_um"],
        contact_ink_thousandths=shade["contact_ink_thousandths"],
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


def _masked_blur(field: np.ndarray, ok: np.ndarray, sigma_pixels: float | tuple[float, float]) -> np.ndarray:
    """A Gaussian blur that does not let the uncovered pixels pull the
    covered ones towards zero: the blur of the field over the blur of the
    mask.  ``sigma_pixels`` may be one width or (down the view, across)."""

    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    if max(sigma_pixels) <= 0.0 if isinstance(sigma_pixels, tuple) else sigma_pixels <= 0.0:
        return np.where(ok, field, 0.0)
    weight = gaussian_filter(ok.astype(np.float64), sigma_pixels)
    smoothed = gaussian_filter(np.where(ok, field, 0.0), sigma_pixels)
    return np.where(weight > 1e-3, smoothed / np.maximum(weight, 1e-3), 0.0)


def _blend_rows(field: np.ndarray, row_valid: np.ndarray, sigma_pixels: float) -> np.ndarray:
    """A Gaussian blend of a field down its rows; rows with nothing covered
    do not vote, so a window's edge does not pull the level to zero."""

    from scipy.ndimage import gaussian_filter1d  # noqa: PLC0415

    if sigma_pixels <= 0.0:
        return field
    weight = gaussian_filter1d(row_valid.astype(np.float64), sigma_pixels, mode="nearest")
    blended = gaussian_filter1d(field * row_valid[:, None], sigma_pixels, axis=0, mode="nearest")
    return np.where(weight[:, None] > 1e-3, blended / np.maximum(weight[:, None], 1e-3), field)


def _masked_running_median(field: np.ndarray, ok: np.ndarray, width_pixels: float) -> np.ndarray:
    """The running median across each row over a window ``width_pixels``
    wide; uncovered pixels do not vote, and a window with fewer than a
    quarter of its pixels covered takes the row's median instead."""

    from scipy.ndimage import median_filter, uniform_filter1d  # noqa: PLC0415

    width = max(3, int(round(width_pixels)) | 1)
    # Uncovered pixels are filled with the row's own median so they pull
    # the window towards the wall's level rather than towards zero.
    row_median = np.array([np.median(row[mask]) if mask.any() else 0.0 for row, mask in zip(field, ok, strict=True)])
    filled = np.where(ok, field, row_median[:, None])
    slow = median_filter(filled, size=(1, width), mode="nearest")
    coverage = uniform_filter1d(ok.astype(np.float64), width, axis=1, mode="constant")
    return np.where(coverage >= 0.25, slow, row_median[:, None])


def _view_depth_field(
    vertices: np.ndarray,
    triangles: np.ndarray,
    validated: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None,
) -> tuple[np.ndarray, int, int, dict[str, Any], int]:
    """The front-most depth of the wall on a side view's lattice."""

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
    return depth, minimum_u, minimum_v, raster_qc, int(visible.size)


def _development_depth_field(
    vertices: np.ndarray,
    triangles: np.ndarray,
    validated: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None,
) -> tuple[np.ndarray, int, int, dict[str, Any], int, tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
    """The wall's radius on the axis development's lattice.

    The outside's median profile r(z) gives every vertex its station: u is
    the reference radius times its angle from the seam, v the meridian arc
    up the profile to its height; the depth is its own radius, so what
    stands proud of the profile stands proud on the strip.  An undercut face is behind the
    wall's face at the same station and loses its pixels to it, as it
    would to a viewer.  Returned with the field is the profile's height,
    arc and radius: the first two turn a height window into rows, and all
    three let a sheet say what height a row of the strip came from and how
    far the wall stood from the axis there.
    """

    from .artifact_profile_break import ArtifactProfileBreakError, _facing_profile  # noqa: PLC0415

    policy = validated["development_policy"]
    raster_policy = validated["raster_policy"]
    pixels_per_mm = int(raster_policy["pixels_per_mm"])
    try:
        heights, radii, _spread = _facing_profile(
            vertices, triangles, height_bin_um=int(policy["profile_bin_um"]), inward=False, cancellation_probe=cancellation_probe
        )
    except ArtifactProfileBreakError as exc:
        raise ArtifactReliefShadeError(str(exc)) from exc
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(heights), np.diff(radii)))])
    z = vertices[:, 2]
    radius = np.hypot(vertices[:, 0], vertices[:, 1])
    station = np.interp(z, heights, arc)
    window = validated["window"]
    middle = (
        0.5 * (window["bottom_um"] + window["top_um"]) / 1000.0
        if window is not None
        else 0.5 * float(heights[0] + heights[-1])
    )
    reference_radius = float(np.interp(middle, heights, radii))
    seam = math.radians(policy["seam_millideg"] / 1000.0)
    theta = np.mod(np.arctan2(vertices[:, 1], vertices[:, 0]) - seam, 2.0 * math.pi)
    corners = vertices[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    centroids = corners.mean(axis=1)
    radial = np.einsum("ij,ij->i", normals[:, :2], centroids[:, :2])
    lengths = np.linalg.norm(normals, axis=1) * np.maximum(np.hypot(centroids[:, 0], centroids[:, 1]), 1e-12)
    outward = radial > (policy["outward_cos_thousandths"] / 1000.0) * lengths
    # A face across the seam is not a distorted face: it is the wedge that
    # closes the round, and dropping it leaves the strip one facet short and
    # the motif on the seam cut by that much.  It is unwrapped instead - its
    # low corners lifted a full turn - so it is drawn once, past the seam,
    # where the cut through the cylinder actually puts it.
    corner_theta = theta[triangles]
    across_seam = (corner_theta.max(axis=1) - corner_theta.min(axis=1)) > math.pi
    unwrapped = np.where(
        across_seam[:, None] & (corner_theta < math.pi), corner_theta + 2.0 * math.pi, corner_theta
    )
    visible = np.flatnonzero(outward)
    if visible.size == 0:
        raise ArtifactReliefShadeError("no face of the mesh faces away from the axis; nothing to develop")
    seam_face_count = int(np.count_nonzero(across_seam[visible]))
    # Each drawn face gets its own three corners, so a corner lifted past the
    # seam keeps the height and radius of the vertex it still is.
    corner_index = triangles[visible]
    projected = np.column_stack(
        [
            (reference_radius * unwrapped[visible]).reshape(-1),
            station[corner_index].reshape(-1),
        ]
    )
    radius = radius[corner_index].reshape(-1)
    triangles = np.arange(projected.shape[0], dtype=np.int64).reshape(-1, 3)
    visible = np.arange(triangles.shape[0], dtype=np.int64)
    try:
        depth, minimum_u, minimum_v, raster_qc = _rasterize_depth_field(
            projected,
            radius,
            triangles[visible],
            pixels_per_mm=pixels_per_mm,
            margin_pixels=int(round(raster_policy["margin_um"] / 1000.0 * pixels_per_mm)),
            layer_separation_mm=raster_policy["layer_separation_um"] / 1000.0,
            cancellation_probe=cancellation_probe,
        )
    except ArtifactRubbingError as exc:
        raise ArtifactReliefShadeError(str(exc)) from exc
    raster_qc = {**raster_qc, "seam_face_count": seam_face_count}
    return depth, minimum_u, minimum_v, raster_qc, int(visible.size), (heights, arc, reference_radius, radii)


def extract_relief_shade(
    canonical_vertices_mm: object,
    faces: object,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[ReliefShadeRaster, dict[str, Any]]:
    """Read the relief's shade where the recipe looks: a side view, or the
    wall unrolled about its axis."""

    from scipy.ndimage import binary_erosion, gaussian_filter1d  # noqa: PLC0415

    validated = validate_relief_shade_recipe(recipe)
    vertices = np.asarray(canonical_vertices_mm, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ArtifactReliefShadeError("mesh must be (n, 3) vertices and (m, 3) faces")
    if int(vertices.shape[0]) != validated["source_vertex_count"] or int(triangles.shape[0]) != validated["source_face_count"]:
        raise ArtifactReliefShadeError("mesh does not match the recipe's vertex and face counts")
    raise_if_cancelled(cancellation_probe)
    raster_policy = validated["raster_policy"]
    base_policy = validated["base_policy"]
    pixels_per_mm = int(raster_policy["pixels_per_mm"])
    on_development = validated["domain"] == RELIEF_SHADE_DOMAIN_DEVELOPMENT
    on_plan = validated["domain"] == RELIEF_SHADE_DOMAIN_PLAN
    smoothing_pixels = base_policy["row_smoothing_um"] / 1000.0 * pixels_per_mm
    profile: tuple[np.ndarray, np.ndarray, float, np.ndarray] | None = None
    plan_rings: tuple[np.ndarray, np.ndarray] | None = None
    if on_development:
        depth, minimum_u, minimum_v, raster_qc, visible_count, profile = _development_depth_field(
            vertices, triangles, validated, cancellation_probe=cancellation_probe
        )
    else:
        depth, minimum_u, minimum_v, raster_qc, visible_count = _view_depth_field(
            vertices, triangles, validated, cancellation_probe=cancellation_probe
        )
    raise_if_cancelled(cancellation_probe)
    # The depth lattice's row 0 is its lowest v; the shade's row 0 will be the top.
    height, width = depth.shape
    xs = (minimum_u + np.arange(width) + 0.5) / pixels_per_mm
    rows_mm = (minimum_v + np.arange(height) + 0.5) / pixels_per_mm
    rows = np.arange(height, dtype=np.float64)
    if on_development:
        covered = np.isfinite(depth)
        if not covered.any():
            raise ArtifactReliefShadeError("the development covers no pixel; nothing to shade")
        # The base: the wall's median radius round the vessel at each row.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row_radius = np.nanmedian(np.where(covered, depth, np.nan), axis=1)
        known = np.isfinite(row_radius)
        row_radius = np.interp(rows, rows[known], row_radius[known])
        if smoothing_pixels > 0.0:
            row_radius = gaussian_filter1d(row_radius, smoothing_pixels)
        relief = np.where(covered, depth, 0.0) - row_radius[:, None]
        ok = covered & (np.abs(relief) <= base_policy["outlier_um"] / 1000.0)
        cos_round = None
    elif on_plan:
        covered = np.isfinite(depth)
        if not covered.any():
            raise ArtifactReliefShadeError("the plan view sees no surface; nothing to shade")
        # Looking down the axis, the drawing plane is the artifact's own
        # x-y, so a pixel's distance from the centre is its radius on the
        # artifact.  A body of revolution has one height on each ring, and
        # that ring's median is the base the relief stands on.
        radius = np.hypot(xs[None, :], rows_mm[:, None])
        rings = np.minimum(
            (radius * pixels_per_mm).astype(np.int64),
            int(math.ceil(float(radius.max()) * pixels_per_mm)),
        )
        ring_count = int(rings.max()) + 1
        base_ring = np.full(ring_count, np.nan)
        flat_rings = rings[covered]
        flat_depth = depth[covered]
        order = np.argsort(flat_rings, kind="stable")
        sorted_rings = flat_rings[order]
        sorted_depth = flat_depth[order]
        edges = np.searchsorted(sorted_rings, np.arange(ring_count + 1))
        for ring in range(ring_count):
            lo, hi = int(edges[ring]), int(edges[ring + 1])
            if hi > lo:
                base_ring[ring] = np.median(sorted_depth[lo:hi])
        known = np.isfinite(base_ring)
        if not known.any():
            raise ArtifactReliefShadeError("no ring of the plan view carries a surface to read a base from")
        indices = np.arange(ring_count, dtype=np.float64)
        base_ring = np.interp(indices, indices[known], base_ring[known])
        if smoothing_pixels > 0.0:
            base_ring = gaussian_filter1d(base_ring, smoothing_pixels)
        # A pixel nearer the viewer than its ring stands proud of the wall.
        relief = np.where(covered, base_ring[rings] - depth, 0.0)
        ok = covered & (np.abs(relief) <= base_policy["outlier_um"] / 1000.0)
        plan_policy = validated["plan_policy"]
        inner_mm = plan_policy["ring_inner_um"] / 1000.0
        outer_mm = plan_policy["ring_outer_um"] / 1000.0
        if inner_mm > 0.0:
            ok &= radius >= inner_mm
        if outer_mm > 0.0:
            ok &= radius <= outer_mm
        cos_round = None
        plan_rings = (indices / pixels_per_mm, base_ring)
    else:
        covered = np.isfinite(depth) & (depth > 0.0)
        if not covered.any():
            raise ArtifactReliefShadeError("the view sees no wall on the viewer's side of the axis; nothing to shade")
        # The base: the radius that explains the near wall's depth at each
        # row, read where the wall faces the viewer squarely.
        inner = base_policy["inner_fraction_thousandths"] / 1000.0
        grazing = base_policy["grazing_fraction_thousandths"] / 1000.0
        abs_x = np.abs(xs)[None, :]
        # Each row's silhouette on either side: how far the wall reaches
        # left and right of the axis there.  A warped vessel reaches
        # farther on one side than its median radius says.
        reach_left = np.where(covered & (xs[None, :] < 0.0), -xs[None, :], 0.0).max(axis=1)
        reach_right = np.where(covered & (xs[None, :] > 0.0), xs[None, :], 0.0).max(axis=1)
        silhouette = np.maximum(reach_left, reach_right)
        own_reach = np.where(xs[None, :] < 0.0, reach_left[:, None], reach_right[:, None])
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
        row_radius = np.interp(rows, rows[known], row_radius[known])
        if smoothing_pixels > 0.0:
            row_radius = gaussian_filter1d(row_radius, smoothing_pixels)
        # The relief is radial - how far the wall stands from the base
        # radius at that height - read as the implied radius less the base,
        # not as depth less a base depth: a depth difference blows up at
        # the silhouette, where the base depth falls away steeply, and a
        # radial one does not.
        relief = np.where(covered, implied, row_radius[:, None]) - row_radius[:, None]
        ok = covered & (abs_x <= grazing * own_reach) & (np.abs(relief) <= base_policy["outlier_um"] / 1000.0)
        # The slope across the view is what the wall's turn round the axis
        # makes it, measured to the silhouette on the pixel's own side.
        cos_round = np.sqrt(np.clip(1.0 - (abs_x / np.maximum(own_reach, 1e-6)) ** 2, 0.0, 1.0))
    window = validated["window"]
    if window is not None:
        bottom_mm = window["bottom_um"] / 1000.0
        top_mm = window["top_um"] / 1000.0
        if profile is not None:
            # On the development the window's bottom and top are heights on
            # the artifact; the rows are meridian arc, so they are converted.
            heights, arc, _reference, _radii = profile
            bottom_mm = float(np.interp(bottom_mm, heights, arc))
            top_mm = float(np.interp(top_mm, heights, arc))
        ok &= (
            (xs[None, :] >= window["left_um"] / 1000.0)
            & (xs[None, :] <= window["right_um"] / 1000.0)
            & (rows_mm[:, None] >= bottom_mm)
            & (rows_mm[:, None] <= top_mm)
        )
    if not ok.any():
        raise ArtifactReliefShadeError(
            "no relief lies where the recipe looks; widen the window, choose another view, or do not take the shade"
        )
    raise_if_cancelled(cancellation_probe)
    relief_policy = validated["relief_policy"]
    grain = _masked_blur(relief, ok, relief_policy["grain_um"] / 1000.0 * pixels_per_mm)
    # The slow unevenness runs round the vessel - an oval rim, a lean - so
    # it is taken out across the view only; up and down the wall the base
    # has already taken the profile, and a blur across a lip or a foot's
    # root would smear that step into a band of false relief.
    # A median, not a blur: a blur of a petal leaves a halo of false hollow
    # round it, the median of a window a few petals wide is the level the
    # wall stands at there and the petals stand on.
    slow = _masked_running_median(relief, ok, relief_policy["slow_um"] / 1000.0 * pixels_per_mm) if relief_policy["slow_um"] else 0.0
    if relief_policy["slow_um"] and relief_policy["slow_blend_um"]:
        # Row by row the median steps, and the steps print as streaks
        # across the shade; the wall's slow level does not step between
        # one row and the next, so it is blended a little up and down.
        slow = _blend_rows(np.asarray(slow), ok.any(axis=1), relief_policy["slow_blend_um"] / 1000.0 * pixels_per_mm)
    if on_plan and relief_policy["slow_um"] > 0:
        # A disc has no rows.  The slow level a plan view has to take out is
        # the vessel's warp - one side standing proud of the other - which
        # runs round the axis, not across the view, and a median along x
        # prints as vertical streaks through the middle.  It is read as a
        # wide blur over the disc instead, far wider than any stamp.
        slow = _masked_blur(relief, ok, relief_policy["slow_um"] / 2000.0 * pixels_per_mm)
    filtered = np.where(ok, grain - slow, 0.0)
    raise_if_cancelled(cancellation_probe)

    shade_policy = validated["shade_policy"]
    if shade_policy["model"] == RELIEF_SHADE_CONTACT_MODEL:
        # 탁본: the paper is pressed on and inked where it lies on the
        # surface.  The paper bridges what is narrower than itself, so what
        # it rests on is the relief's own upper envelope over that width;
        # the ink falls away with how far the surface sits below it and is
        # gone at the black point.  A stamped mark then reads whole - the
        # ground dark, the impression pale - where a light would show only
        # the flank that turns from it.
        from scipy.ndimage import maximum_filter  # noqa: PLC0415

        paper_pixels = int(round(shade_policy["paper_um"] / 1000.0 * pixels_per_mm))
        surface = np.where(ok, filtered, -np.inf)
        paper = (
            maximum_filter(surface, size=2 * paper_pixels + 1, mode="nearest")
            if paper_pixels > 0
            else surface
        )
        below = np.where(ok, np.maximum(paper - filtered, 0.0), 0.0)
        ink = shade_policy["contact_ink_thousandths"] / 1000.0
        darkness = np.where(
            ok, ink * np.clip(1.0 - below / (shade_policy["black_point_um"] / 1000.0), 0.0, 1.0), 0.0
        )
    else:
        # The light on a height field whose normal is (-dh/dx, -dh/dy, 1).
        gradient_x = np.zeros_like(filtered)
        gradient_y = np.zeros_like(filtered)
        gradient_x[:, 1:-1] = (filtered[:, 2:] - filtered[:, :-2]) * pixels_per_mm / 2.0
        gradient_y[1:-1, :] = (filtered[2:, :] - filtered[:-2, :]) * pixels_per_mm / 2.0
        if cos_round is not None:
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
        view=relief_shade_label(validated),
    )
    percentiles = np.percentile(filtered[ok], [5, 50, 95])
    qc = {
        **raster.qc_summary(),
        # A plan view's base is a height on each ring, not a radius on each
        # row, so it is reported under its own keys below.
        **(
            {}
            if plan_rings is None
            else {
                "base_height_max_um": int(round(float(plan_rings[1].max()) * 1000.0)),
                "base_height_min_um": int(round(float(plan_rings[1].min()) * 1000.0)),
            }
        ),
        **(
            {}
            if plan_rings is not None
            else {
                "base_radius_max_um": int(round(float(row_radius.max()) * 1000.0)),
                "base_radius_min_um": int(round(float(row_radius.min()) * 1000.0)),
            }
        ),
        "covered_pixel_count": int(raster_qc["covered_pixel_count"]),
        "domain": validated["domain"],
        "relief_max_um": int(round(float(filtered[ok].max()) * 1000.0)),
        "relief_min_um": int(round(float(filtered[ok].min()) * 1000.0)),
        "relief_p05_um": int(round(float(percentiles[0]) * 1000.0)),
        "relief_p50_um": int(round(float(percentiles[1]) * 1000.0)),
        "relief_p95_um": int(round(float(percentiles[2]) * 1000.0)),
        "trusted_pixel_count": int(np.count_nonzero(trusted)),
        "view": validated["view"],
        "visible_face_count": visible_count,
    }
    if plan_rings is not None:
        ring_radius, ring_base = plan_rings
        plan_policy = validated["plan_policy"]
        qc["plan_ring_inner_um"] = int(plan_policy["ring_inner_um"])
        qc["plan_ring_outer_um"] = int(plan_policy["ring_outer_um"])
        # What the relief was read against: the artifact's own profile, the
        # height of each ring, over the band the shade covers.  Evenly
        # spaced so a reader can see the surface it stands on without the
        # raster.
        stations = np.linspace(0.0, float(ring_radius[-1]), RELIEF_SHADE_DEVELOPMENT_PROFILE_BANDS + 1)
        qc["plan_ring_radius_um"] = [int(round(float(value) * 1000.0)) for value in stations]
        qc["plan_ring_base_um"] = [
            int(round(float(value) * 1000.0)) for value in np.interp(stations, ring_radius, ring_base)
        ]
    if profile is not None:
        heights, arc, reference_radius, radii = profile
        qc["development_arc_um"] = int(round(float(arc[-1]) * 1000.0))
        qc["development_reference_radius_um"] = int(round(reference_radius * 1000.0))
        # The circumference is the round the strip was cut from, not the
        # rightmost pixel that happened to be inked: a reader who measures a
        # vessel's girth off this number must get the girth.
        qc["development_circumference_um"] = int(round(2.0 * math.pi * reference_radius * 1000.0))
        qc["development_inked_width_um"] = int(
            round(float(np.nanmax(np.where(covered, xs[None, :], np.nan))) * 1000.0)
        )
        qc["profile_bin_count"] = int(heights.size)
        qc["seam_face_count"] = int(raster_qc.get("seam_face_count", 0))
        qc["seam_millideg"] = int(validated["development_policy"]["seam_millideg"])
        # The strip's rows are meridian arc, so a sheet that lays the strip
        # back on an elevation cannot read a height off them.  These say it:
        # the height each of evenly spaced rows came from, from the raster's
        # bottom edge to its top, and how far the wall stood from the axis
        # there - which is where the drafter's scissors go when the strip is
        # pasted from the centre line out to the edge.
        bottom_v = float(minimum_v + row0) / pixels_per_mm
        top_v = float(minimum_v + row1) / pixels_per_mm
        stations = np.linspace(bottom_v, top_v, RELIEF_SHADE_DEVELOPMENT_PROFILE_BANDS + 1)
        qc["development_height_profile_um"] = [
            int(round(float(value) * 1000.0)) for value in np.interp(stations, arc, heights)
        ]
        qc["development_radius_profile_um"] = [
            int(round(float(value) * 1000.0)) for value in np.interp(stations, arc, radii)
        ]
    return raster, qc


def relief_shade_label(recipe: Mapping[str, Any]) -> str:
    """The label a recipe's raster carries: its view, or ``development``."""

    if recipe.get("domain") == RELIEF_SHADE_DOMAIN_DEVELOPMENT:
        return RELIEF_SHADE_DEVELOPMENT_LABEL
    return str(recipe["view"])


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
    view: OutlineView | str | None = None,
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
    if receipt["view"] != relief_shade_label(recipe):
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
    "RELIEF_SHADE_DEVELOPMENT_LABEL",
    "RELIEF_SHADE_DOMAINS",
    "RELIEF_SHADE_DOMAIN_DEVELOPMENT",
    "RELIEF_SHADE_DOMAIN_VIEW",
    "RELIEF_SHADE_PAYLOAD_EXTENSION_KEY",
    "RELIEF_SHADE_PIXEL_FORMAT",
    "RELIEF_SHADE_RECORD_TYPE",
    "RELIEF_SHADE_CONTACT_MODEL",
    "RELIEF_SHADE_DOMAIN_PLAN",
    "RELIEF_SHADE_PLAN_VIEWS",
    "RELIEF_SHADE_VIEWS",
    "ReliefShadeComputation",
    "ReliefShadeRaster",
    "commit_relief_shade",
    "compute_relief_shade",
    "extract_relief_shade",
    "relief_shade_computation_matches_active_projection",
    "relief_shade_label",
    "relief_shade_receipt_from_record",
    "relief_shade_recipe",
    "require_relief_shade_raster",
    "validate_relief_shade_receipt",
    "validate_relief_shade_recipe",
    "validate_relief_shade_records",
]
