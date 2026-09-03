"""A rubbing on the developed surface: relief drawn on unrolled coordinates.

The six-view Digital Rubbing looks at the artifact from one side, so a curved
wall is foreshortened and the relief on its flanks is seen at a slant.  Paper
is not: it lies on the surface, takes the relief the surface has, and comes
off flat.  This module draws that relief on the coordinates a tile-unwrap
record has already developed, so the raster corresponds to the strip of paper
a rubber pastes beside the drawing.

Depth is the radius about the centre the strip was unrolled on - the measured
rotation axis for a positioned pot, the fitted section centres for a tile.
The quantisation, the local-mean relief operator, and the ink mapping are the
ones the six-view rubbing uses, so the two rasters read the same way.  Nothing
is resampled from a six-view raster: the depth is rasterised on the developed
triangles directly, and the record names the exact development it was drawn
on, by payload hash, so it can be recomputed and checked.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_cancellation import CancellationProbe, raise_if_cancelled
from .artifact_document import (
    ArtifactDocument,
    ArtifactDocumentError,
    DerivedRecord,
    OperationContext,
    RecordFreshness,
    RecordLifecycleStatus,
    canonical_recipe_hash,
)
from .artifact_rubbing_extractor import (
    ArtifactRubbingError,
    DigitalRubbingResourceEstimate,
    MAX_RUBBING_DIMENSION,
    MAX_RUBBING_GRID_INDEX,
    MAX_RUBBING_PIXELS,
    MAX_RUBBING_PIXELS_PER_MM,
    RUBBING_ESTIMATE_FIXED_OVERHEAD_BYTES,
    RUBBING_ESTIMATE_GEOMETRY_MULTIPLIER,
    RUBBING_ESTIMATED_PEAK_BYTES_PER_PIXEL,
    RUBBING_PIXEL_FORMAT,
    _rasterize_depth_field,
    _render_local_relief,
    ga8_raster_summary,
    rubbing_policy_blocks,
)
from .artifact_scene_adapter import ArtifactProjectionSnapshot
from .artifact_session import ArtifactSession, ArtifactSessionError
from .artifact_tile_unwrap_extractor import (
    ArtifactTileUnwrapError,
    TileUnwrapMesh,
    extract_tile_unwrap_development,
)
from .artifact_tile_unwrap_record import (
    ArtifactTileUnwrapRecordError,
    TILE_UNWRAP_RECORD_TYPE,
    tile_unwrap_receipt_from_record,
)
from .canonical_json import canonical_json_bytes, canonical_json_sha256
from .mesh_loader import MeshData


DEVELOPED_RUBBING_RECORD_TYPE = "raster.developed_rubbing.v1"
DEVELOPED_RUBBING_OPERATION_KIND = "developed_rubbing"
DEVELOPED_RUBBING_ALGORITHM = "archmeshrubbing.developed_local_mean_relief"
DEVELOPED_RUBBING_ALGORITHM_VERSION = "1.0.0"
DEVELOPED_RUBBING_RASTER_SCHEMA_VERSION = "1.0.0"
DEVELOPED_RUBBING_COORDINATE_SPACE = "canonical_mm_developed_raster/v1"
DEVELOPED_RUBBING_RASTER_HASH_SCOPE = "header-rfc8785+pixels-ga8-row-major/v1"
DEVELOPED_RUBBING_DEPTH_MEASURE = "radius_about_unrolling_centre/v1"
# A rubbing of a strip is a rectangle of paper: the rubber tapes it on, lifts
# it, and pastes it beside the drawing with straight edges.  The development
# it is drawn on is not rectangular - its boundary follows whole triangles,
# and the facet spacing that quantises that boundary is the arc r * dtheta,
# so a strip came out with its width stepping row by row.  Cropping to the
# largest fully covered rectangle gives back the piece of paper.  A sherd is
# not a strip, though, and cropping one would throw most of it away, so the
# full development stays available.
ARTBOARD_LARGEST_COVERED_RECTANGLE = "largest_covered_rectangle/v1"
ARTBOARD_DEVELOPMENT_BOUNDS = "development_bounds_plus_margin/v1"
ARTBOARD_POLICIES = (
    ARTBOARD_DEVELOPMENT_BOUNDS,
    ARTBOARD_LARGEST_COVERED_RECTANGLE,
)
DEVELOPED_RUBBING_RECEIPT_EXTENSION_KEY = "org.archmeshrubbing:developed-rubbing-v1"
DEVELOPED_RUBBING_RECEIPT_MEDIA_TYPE = (
    "application/vnd.archmeshrubbing.developed-rubbing-receipt+json"
)
DEVELOPED_RUBBING_GEOMETRY_REF_PREFIX = (
    "urn:archmeshrubbing:developed-rubbing-raster:sha256:"
)
MAX_DEVELOPED_RUBBING_RECEIPT_BYTES = 64 * 1024

_ROW_ORDER = "top_to_bottom_v_descending"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactDevelopedRubbingError(ValueError):
    """A rubbing on a developed surface cannot be produced or trusted."""


def _exact_keys(value: object, expected: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactDevelopedRubbingError(f"{name} must be an object")
    keys = set(value.keys())
    if keys != expected:
        missing = sorted(expected - keys)
        unexpected = sorted(keys - expected)
        raise ArtifactDevelopedRubbingError(
            f"{name} keys are invalid (missing={missing}, unexpected={unexpected})"
        )
    return value


def _strict_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactDevelopedRubbingError(f"{name} must be an integer")
    number = int(value)
    if number < minimum or number > maximum:
        raise ArtifactDevelopedRubbingError(
            f"{name} must be in the inclusive range {minimum}..{maximum}"
        )
    return number


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ArtifactDevelopedRubbingError(f"{name} must be a lowercase SHA-256")
    return value


def _record_id(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ArtifactDevelopedRubbingError(f"{name} must be a non-empty record ID")
    return value


def _artboard_policy(value: object) -> str:
    if not isinstance(value, str) or value not in ARTBOARD_POLICIES:
        raise ArtifactDevelopedRubbingError(
            "artboard_policy must be "
            f"{ARTBOARD_LARGEST_COVERED_RECTANGLE!r} or "
            f"{ARTBOARD_DEVELOPMENT_BOUNDS!r}"
        )
    return value


def developed_rubbing_recipe(
    *,
    development_record_id: str,
    development_sha256: str,
    development_recipe_hash: str,
    pixels_per_mm: int,
    margin_um: int,
    reference_radius_um: int,
    depth_quantization_um: int,
    black_point_um: int,
    ink_strength_percent: int,
    relief_polarity: str,
    artboard_policy: str = ARTBOARD_LARGEST_COVERED_RECTANGLE,
) -> dict[str, Any]:
    """Resolve every option before context capture, naming the development."""

    policy = _artboard_policy(artboard_policy)
    record_id = _record_id(development_record_id, name="development record ID")
    unwrap_sha = _sha256(development_sha256, name="development unwrap_sha256")
    recipe_hash = _sha256(development_recipe_hash, name="development recipe_hash")
    try:
        blocks = rubbing_policy_blocks(
            pixels_per_mm=pixels_per_mm,
            margin_um=margin_um,
            reference_radius_um=reference_radius_um,
            depth_quantization_um=depth_quantization_um,
            black_point_um=black_point_um,
            ink_strength_percent=ink_strength_percent,
            relief_polarity=relief_polarity,
        )
    except ArtifactRubbingError as exc:
        raise ArtifactDevelopedRubbingError(str(exc)) from exc
    depth_policy = dict(blocks["depth_policy"])
    depth_policy["measure"] = DEVELOPED_RUBBING_DEPTH_MEASURE
    pixel_policy = blocks["pixel_policy"]
    if (
        policy == ARTBOARD_LARGEST_COVERED_RECTANGLE
        and int(pixel_policy["margin_pixels"]) != 0
    ):
        # A margin is uncovered paper, and the crop exists to remove exactly
        # that; asking for both is asking for two different artboards.
        raise ArtifactDevelopedRubbingError(
            "a rubbing cropped to its covered rectangle cannot also carry a "
            "margin; set margin_um to 0 or use the development-bounds artboard"
        )
    return {
        "algorithm": DEVELOPED_RUBBING_ALGORITHM,
        "algorithm_version": DEVELOPED_RUBBING_ALGORITHM_VERSION,
        "artboard_policy": policy,
        "coordinate_space": DEVELOPED_RUBBING_COORDINATE_SPACE,
        "depth_policy": depth_policy,
        "development": {
            "record_id": record_id,
            "record_type": TILE_UNWRAP_RECORD_TYPE,
            "recipe_hash": recipe_hash,
            "unwrap_sha256": unwrap_sha,
        },
        "kind": DEVELOPED_RUBBING_OPERATION_KIND,
        "pixel_policy": blocks["pixel_policy"],
        "rasterization_policy": {
            "edge_rule": "barycentric_nonnegative_with_fixed_epsilon/v1",
            "multi_layer_policy": "frontmost_and_count_second_depth/v1",
            "projected_zero_area_faces": "drop_and_count",
            "sampling": "none",
            "surface": "developed_uv_triangles_um/v1",
            "z_tie": "geometry_invariant_max_depth",
        },
        "relief_policy": blocks["relief_policy"],
        "resource_limits": blocks["resource_limits"],
    }


def validate_developed_rubbing_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(recipe, Mapping):
        raise ArtifactDevelopedRubbingError("developed rubbing recipe must be an object")
    pixel_policy = recipe.get("pixel_policy")
    depth_policy = recipe.get("depth_policy")
    relief_policy = recipe.get("relief_policy")
    development = recipe.get("development")
    if not all(
        isinstance(value, Mapping)
        for value in (pixel_policy, depth_policy, relief_policy, development)
    ):
        raise ArtifactDevelopedRubbingError(
            "developed rubbing recipe policies are invalid"
        )
    assert isinstance(pixel_policy, Mapping)
    assert isinstance(depth_policy, Mapping)
    assert isinstance(relief_policy, Mapping)
    assert isinstance(development, Mapping)
    expected = developed_rubbing_recipe(
        artboard_policy=recipe.get("artboard_policy"),  # type: ignore[arg-type]
        development_record_id=development.get("record_id"),  # type: ignore[arg-type]
        development_sha256=development.get("unwrap_sha256"),  # type: ignore[arg-type]
        development_recipe_hash=development.get("recipe_hash"),  # type: ignore[arg-type]
        pixels_per_mm=pixel_policy.get("pixels_per_mm"),  # type: ignore[arg-type]
        margin_um=pixel_policy.get("margin_requested_um"),  # type: ignore[arg-type]
        reference_radius_um=relief_policy.get(  # type: ignore[arg-type]
            "reference_radius_requested_um"
        ),
        depth_quantization_um=depth_policy.get("quantization_um"),  # type: ignore[arg-type]
        black_point_um=relief_policy.get("black_point_requested_um"),  # type: ignore[arg-type]
        ink_strength_percent=relief_policy.get("ink_strength_percent"),  # type: ignore[arg-type]
        relief_polarity=relief_policy.get("polarity"),  # type: ignore[arg-type]
    )
    try:
        actual_hash = canonical_recipe_hash(recipe)
    except (ArtifactDocumentError, ValueError, TypeError) as exc:
        raise ArtifactDevelopedRubbingError(str(exc)) from exc
    if actual_hash != canonical_recipe_hash(expected):
        raise ArtifactDevelopedRubbingError(
            "developed rubbing recipe does not match the production contract"
        )
    return expected


@dataclass(frozen=True, slots=True)
class DevelopedRubbingRaster:
    """A grey+alpha raster on the developed (u, v) lattice, bound to its development."""

    pixels: np.ndarray
    pixels_per_meter: int
    minimum_u_pixel_index: int
    minimum_v_pixel_index: int
    development_sha256: str

    def __post_init__(self) -> None:
        array = np.asarray(self.pixels)
        if (
            array.dtype != np.uint8
            or array.ndim != 3
            or array.shape[2] != 2
            or array.shape[0] <= 0
            or array.shape[1] <= 0
        ):
            raise ArtifactDevelopedRubbingError(
                "raster pixels must be a non-empty HxWx2 uint8 array"
            )
        if array.shape[0] * array.shape[1] > MAX_RUBBING_PIXELS:
            raise ArtifactDevelopedRubbingError("raster pixels exceed the safety limit")
        ppm = _strict_int(
            self.pixels_per_meter,
            name="pixels_per_meter",
            minimum=1000,
            maximum=MAX_RUBBING_PIXELS_PER_MM * 1000,
        )
        if ppm % 1000 != 0:
            raise ArtifactDevelopedRubbingError(
                "pixels_per_meter must encode an integer pixels/mm"
            )
        for name, value in (
            ("minimum_u_pixel_index", self.minimum_u_pixel_index),
            ("minimum_v_pixel_index", self.minimum_v_pixel_index),
        ):
            _strict_int(
                value,
                name=name,
                minimum=-MAX_RUBBING_GRID_INDEX,
                maximum=MAX_RUBBING_GRID_INDEX,
            )
        copied = np.ascontiguousarray(array).copy()
        copied.setflags(write=False)
        object.__setattr__(self, "pixels", copied)
        object.__setattr__(self, "pixels_per_meter", ppm)
        object.__setattr__(
            self,
            "development_sha256",
            _sha256(self.development_sha256, name="development_sha256"),
        )

    @property
    def width_pixels(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height_pixels(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def raw_pixel_sha256(self) -> str:
        return hashlib.sha256(self.pixels.tobytes(order="C")).hexdigest()

    def semantic_header(self) -> dict[str, Any]:
        return {
            "coordinate_space": DEVELOPED_RUBBING_COORDINATE_SPACE,
            "development_sha256": self.development_sha256,
            "height_pixels": self.height_pixels,
            "minimum_u_pixel_index": self.minimum_u_pixel_index,
            "minimum_v_pixel_index": self.minimum_v_pixel_index,
            "pixel_format": RUBBING_PIXEL_FORMAT,
            "pixels_per_meter": self.pixels_per_meter,
            "raster_hash_scope": DEVELOPED_RUBBING_RASTER_HASH_SCOPE,
            "row_order": _ROW_ORDER,
            "schema_version": DEVELOPED_RUBBING_RASTER_SCHEMA_VERSION,
            "width_pixels": self.width_pixels,
        }

    @property
    def raster_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"archmeshrubbing.developed-rubbing-raster\0")
        digest.update(canonical_json_bytes(self.semantic_header()))
        digest.update(b"\0")
        digest.update(self.pixels.tobytes(order="C"))
        return digest.hexdigest()

    @property
    def geometry_ref(self) -> str:
        return f"{DEVELOPED_RUBBING_GEOMETRY_REF_PREFIX}{self.raster_sha256}"

    def receipt(self) -> dict[str, Any]:
        header = self.semantic_header()
        return {
            **header,
            "height_mm_exact": {
                "denominator": self.pixels_per_meter,
                "numerator": self.height_pixels * 1000,
            },
            "raw_pixel_byte_length": int(self.pixels.nbytes),
            "raw_pixel_sha256": self.raw_pixel_sha256,
            "raster_sha256": self.raster_sha256,
            "width_mm_exact": {
                "denominator": self.pixels_per_meter,
                "numerator": self.width_pixels * 1000,
            },
        }

    def qc_summary(self) -> dict[str, Any]:
        try:
            summary = ga8_raster_summary(self.pixels)
        except ArtifactRubbingError as exc:
            raise ArtifactDevelopedRubbingError(str(exc)) from exc
        return {
            **summary,
            "development_sha256": self.development_sha256,
            "pixels_per_meter": self.pixels_per_meter,
            "raster_sha256": self.raster_sha256,
            "raw_pixel_sha256": self.raw_pixel_sha256,
        }


def validate_developed_rubbing_receipt(value: object) -> dict[str, Any]:
    receipt = _exact_keys(
        value,
        {
            "coordinate_space",
            "development_sha256",
            "height_mm_exact",
            "height_pixels",
            "minimum_u_pixel_index",
            "minimum_v_pixel_index",
            "pixel_format",
            "pixels_per_meter",
            "raster_hash_scope",
            "raster_sha256",
            "raw_pixel_byte_length",
            "raw_pixel_sha256",
            "row_order",
            "schema_version",
            "width_mm_exact",
            "width_pixels",
        },
        name="developed rubbing receipt",
    )
    literal_fields = {
        "coordinate_space": DEVELOPED_RUBBING_COORDINATE_SPACE,
        "pixel_format": RUBBING_PIXEL_FORMAT,
        "raster_hash_scope": DEVELOPED_RUBBING_RASTER_HASH_SCOPE,
        "row_order": _ROW_ORDER,
        "schema_version": DEVELOPED_RUBBING_RASTER_SCHEMA_VERSION,
    }
    for key, expected in literal_fields.items():
        if receipt[key] != expected:
            raise ArtifactDevelopedRubbingError(
                f"developed rubbing receipt field {key!r} is invalid"
            )
    width = _strict_int(
        receipt["width_pixels"],
        name="width_pixels",
        minimum=1,
        maximum=MAX_RUBBING_DIMENSION,
    )
    height = _strict_int(
        receipt["height_pixels"],
        name="height_pixels",
        minimum=1,
        maximum=MAX_RUBBING_DIMENSION,
    )
    if width * height > MAX_RUBBING_PIXELS:
        raise ArtifactDevelopedRubbingError("receipt raster exceeds the pixel limit")
    ppm = _strict_int(
        receipt["pixels_per_meter"],
        name="pixels_per_meter",
        minimum=1000,
        maximum=MAX_RUBBING_PIXELS_PER_MM * 1000,
    )
    if ppm % 1000 != 0:
        raise ArtifactDevelopedRubbingError(
            "pixels_per_meter must encode integer pixels/mm"
        )
    minimum_u = _strict_int(
        receipt["minimum_u_pixel_index"],
        name="minimum_u_pixel_index",
        minimum=-MAX_RUBBING_GRID_INDEX,
        maximum=MAX_RUBBING_GRID_INDEX,
    )
    minimum_v = _strict_int(
        receipt["minimum_v_pixel_index"],
        name="minimum_v_pixel_index",
        minimum=-MAX_RUBBING_GRID_INDEX,
        maximum=MAX_RUBBING_GRID_INDEX,
    )
    byte_length = _strict_int(
        receipt["raw_pixel_byte_length"],
        name="raw_pixel_byte_length",
        minimum=2,
        maximum=MAX_RUBBING_PIXELS * 2,
    )
    if byte_length != width * height * 2:
        raise ArtifactDevelopedRubbingError(
            "developed rubbing raw pixel byte length is inconsistent"
        )
    development_sha = _sha256(receipt["development_sha256"], name="development_sha256")
    raw_sha = _sha256(receipt["raw_pixel_sha256"], name="raw_pixel_sha256")
    raster_sha = _sha256(receipt["raster_sha256"], name="raster_sha256")

    def exact_dimension(name: str, pixels: int) -> dict[str, int]:
        rational = _exact_keys(
            receipt[name],
            {"denominator", "numerator"},
            name=name,
        )
        denominator = _strict_int(
            rational["denominator"],
            name=f"{name}.denominator",
            minimum=1,
            maximum=MAX_RUBBING_PIXELS_PER_MM * 1000,
        )
        numerator = _strict_int(
            rational["numerator"],
            name=f"{name}.numerator",
            minimum=1,
            maximum=MAX_RUBBING_DIMENSION * 1000,
        )
        if denominator != ppm or numerator != pixels * 1000:
            raise ArtifactDevelopedRubbingError(
                f"{name} is inconsistent with the grid"
            )
        return {"denominator": denominator, "numerator": numerator}

    width_exact = exact_dimension("width_mm_exact", width)
    height_exact = exact_dimension("height_mm_exact", height)
    return {
        **literal_fields,
        "development_sha256": development_sha,
        "height_mm_exact": height_exact,
        "height_pixels": height,
        "minimum_u_pixel_index": minimum_u,
        "minimum_v_pixel_index": minimum_v,
        "pixels_per_meter": ppm,
        "raster_sha256": raster_sha,
        "raw_pixel_byte_length": byte_length,
        "raw_pixel_sha256": raw_sha,
        "width_mm_exact": width_exact,
        "width_pixels": width,
    }


def _validate_qc_against_receipt(
    qc: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    width = int(receipt["width_pixels"])
    height = int(receipt["height_pixels"])
    expected = {
        "development_sha256": receipt["development_sha256"],
        "height_pixels": height,
        "pixel_count": width * height,
        "pixel_format": RUBBING_PIXEL_FORMAT,
        "pixels_per_meter": receipt["pixels_per_meter"],
        "raster_sha256": receipt["raster_sha256"],
        "raw_pixel_sha256": receipt["raw_pixel_sha256"],
        "width_pixels": width,
    }
    for key, value in expected.items():
        if qc.get(key) != value:
            raise ArtifactDevelopedRubbingError(
                f"developed rubbing QC field {key!r} does not match its receipt"
            )
    if qc.get("alpha_binary") is not True:
        raise ArtifactDevelopedRubbingError("developed rubbing QC alpha must be binary")
    covered = qc.get("covered_pixel_count")
    if (
        isinstance(covered, bool)
        or not isinstance(covered, int)
        or covered < 1
        or covered > width * height
    ):
        raise ArtifactDevelopedRubbingError(
            "developed rubbing QC coverage is inconsistent with its receipt"
        )


def _largest_covered_rectangle(covered: np.ndarray) -> tuple[int, int, int, int]:
    """(top, left, height, width) of the biggest all-covered rectangle.

    The largest-rectangle-in-a-histogram scan, row by row.  Ties go to the
    topmost then leftmost rectangle so one raster always yields one crop.
    """

    height = int(covered.shape[0])
    width = int(covered.shape[1])
    best_area = 0
    best = (0, 0, 0, 0)
    runs = np.zeros((width,), dtype=np.int64)
    for row in range(height):
        runs = np.where(covered[row], runs + 1, 0)
        stack: list[tuple[int, int]] = []
        for column in range(width + 1):
            current = int(runs[column]) if column < width else 0
            start = column
            while stack and stack[-1][1] >= current:
                origin, tall = stack.pop()
                area = tall * (column - origin)
                candidate = (row - tall + 1, origin, tall, column - origin)
                if area > best_area or (
                    area == best_area and area > 0 and candidate[:2] < best[:2]
                ):
                    best_area = area
                    best = candidate
                start = origin
            stack.append((start, current))
    if best_area <= 0:
        raise ArtifactDevelopedRubbingError(
            "the development covers no rectangle of pixels to crop to"
        )
    return best


def extract_developed_rubbing(
    unwrap: TileUnwrapMesh,
    radius_mm: object,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[DevelopedRubbingRaster, dict[str, Any]]:
    """Draw the relief of ``radius_mm`` on the developed triangles of ``unwrap``."""

    raise_if_cancelled(cancellation_probe)
    validated = validate_developed_rubbing_recipe(recipe)
    if not isinstance(unwrap, TileUnwrapMesh):
        raise ArtifactDevelopedRubbingError("unwrap must be a TileUnwrapMesh")
    radius = np.asarray(radius_mm, dtype=np.float64)
    if radius.shape != (unwrap.vertex_count,):
        raise ArtifactDevelopedRubbingError(
            "development radius must carry one value per developed vertex"
        )
    if not bool(np.isfinite(radius).all()):
        raise ArtifactDevelopedRubbingError(
            "development radius contains non-finite values"
        )
    development = validated["development"]
    pixel_policy = validated["pixel_policy"]
    depth_policy = validated["depth_policy"]
    relief_policy = validated["relief_policy"]
    assert isinstance(development, Mapping)
    assert isinstance(pixel_policy, Mapping)
    assert isinstance(depth_policy, Mapping)
    assert isinstance(relief_policy, Mapping)
    pixels_per_mm = int(pixel_policy["pixels_per_mm"])
    # The development is exact in micrometres; dividing by 1000 is the only
    # rounding between the record and the lattice, and it is the same for
    # every reader.
    projected = np.asarray(unwrap.uv_um, dtype=np.float64) / 1000.0
    faces = np.asarray(unwrap.faces, dtype=np.int64)
    try:
        depth, minimum_u, minimum_v, raster_qc = _rasterize_depth_field(
            projected,
            radius,
            faces,
            pixels_per_mm=pixels_per_mm,
            margin_pixels=int(pixel_policy["margin_pixels"]),
            layer_separation_mm=float(depth_policy["quantization_um"]) / 1000.0,
            cancellation_probe=cancellation_probe,
        )
        pixels, relief_qc = _render_local_relief(
            depth,
            depth_quantization_um=int(depth_policy["quantization_um"]),
            reference_radius_pixels=int(relief_policy["reference_radius_pixels"]),
            effective_black_point_ticks=int(
                relief_policy["effective_black_point_ticks"]
            ),
            relief_polarity=str(relief_policy["polarity"]),
            minimum_reference_sample_count=int(
                relief_policy["minimum_reference_sample_count"]
            ),
            cancellation_probe=cancellation_probe,
        )
    except ArtifactRubbingError as exc:
        raise ArtifactDevelopedRubbingError(str(exc)) from exc
    raise_if_cancelled(cancellation_probe)

    policy = str(validated["artboard_policy"])
    uncropped_height = int(pixels.shape[0])
    uncropped_width = int(pixels.shape[1])
    # artboard_width_pixels / artboard_height_pixels already report the extent
    # before any crop, so only the coverage inside it is new here.  The four
    # trim counts are always reported, zero when the policy keeps the whole
    # development, so a reader need not branch on the policy to add them up.
    crop: dict[str, object] = {
        "artboard_policy": policy,
        "cropped_bottom_pixels": 0,
        "cropped_left_pixels": 0,
        "cropped_right_pixels": 0,
        "cropped_top_pixels": 0,
        "uncropped_covered_pixel_count": int(
            np.count_nonzero(pixels[:, :, 1] == 255)
        ),
    }
    if policy == ARTBOARD_LARGEST_COVERED_RECTANGLE:
        # The relief is computed on the whole development first and cropped
        # afterwards, so the tone inside the rectangle is the tone it had with
        # its neighbours present: the crop frames the rubbing, it does not
        # change it.
        top, left, cropped_height, cropped_width = _largest_covered_rectangle(
            pixels[:, :, 1] == 255
        )
        raise_if_cancelled(cancellation_probe)
        pixels = np.ascontiguousarray(
            pixels[top : top + cropped_height, left : left + cropped_width]
        )
        # Raster row zero is the largest v, so trimming the bottom rows is
        # what moves the artboard's minimum v.
        minimum_u += left
        minimum_v += uncropped_height - top - cropped_height
        crop.update(
            {
                "cropped_bottom_pixels": uncropped_height - top - cropped_height,
                "cropped_left_pixels": left,
                "cropped_right_pixels": uncropped_width - left - cropped_width,
                "cropped_top_pixels": top,
            }
        )
    raise_if_cancelled(cancellation_probe)
    raster = DevelopedRubbingRaster(
        pixels=pixels,
        pixels_per_meter=pixels_per_mm * 1000,
        minimum_u_pixel_index=minimum_u,
        minimum_v_pixel_index=minimum_v,
        development_sha256=str(development["unwrap_sha256"]),
    )
    raise_if_cancelled(cancellation_probe)
    qc = {
        "all_developed_faces_included": True,
        "depth_measure": DEVELOPED_RUBBING_DEPTH_MEASURE,
        "development_face_count": unwrap.face_count,
        "development_vertex_count": unwrap.vertex_count,
        "radius_max_um_rounded": int(round(float(np.max(radius)) * 1000.0)),
        "radius_min_um_rounded": int(round(float(np.min(radius)) * 1000.0)),
        "sampling_applied": False,
        **crop,
        **raster_qc,
        **relief_qc,
        **raster.qc_summary(),
    }
    raise_if_cancelled(cancellation_probe)
    return raster, qc


def _development_record(
    document: ArtifactDocument,
    record_id: str,
) -> tuple[DerivedRecord, dict[str, Any]]:
    if not isinstance(document, ArtifactDocument):
        raise ArtifactDevelopedRubbingError("document must be an ArtifactDocument")
    record = document.record_index.get(record_id)
    if record is None:
        raise ArtifactDevelopedRubbingError(
            f"development record {record_id!r} does not exist"
        )
    if record.type != TILE_UNWRAP_RECORD_TYPE:
        raise ArtifactDevelopedRubbingError(
            f"development record {record_id!r} is not a tile unwrap record"
        )
    if record.lifecycle_status is not RecordLifecycleStatus.READY:
        raise ArtifactDevelopedRubbingError(
            f"development record {record_id!r} is not READY"
        )
    try:
        freshness = document.record_freshness(record.id)
    except ArtifactDocumentError as exc:
        raise ArtifactDevelopedRubbingError(str(exc)) from exc
    if freshness is not RecordFreshness.FRESH:
        raise ArtifactDevelopedRubbingError(
            f"development record {record_id!r} is not FRESH ({freshness.value})"
        )
    try:
        receipt = tile_unwrap_receipt_from_record(record)
    except ArtifactTileUnwrapRecordError as exc:
        raise ArtifactDevelopedRubbingError(str(exc)) from exc
    return record, receipt


def development_record_for_recipe(
    document: ArtifactDocument,
    recipe: Mapping[str, Any],
) -> tuple[DerivedRecord, dict[str, Any]]:
    """The READY + FRESH development a recipe names, proven to still be it."""

    validated = validate_developed_rubbing_recipe(recipe)
    development = validated["development"]
    assert isinstance(development, Mapping)
    record, receipt = _development_record(document, str(development["record_id"]))
    if receipt["unwrap_sha256"] != development["unwrap_sha256"]:
        raise ArtifactDevelopedRubbingError(
            "recipe names a development the record no longer carries"
        )
    if record.recipe_hash != development["recipe_hash"]:
        raise ArtifactDevelopedRubbingError(
            "recipe names a development recipe the record no longer carries"
        )
    return record, receipt


def developed_rubbing_recipe_for_record(
    document: ArtifactDocument,
    development_record_id: str,
    *,
    pixels_per_mm: int,
    margin_um: int,
    reference_radius_um: int,
    depth_quantization_um: int,
    black_point_um: int,
    ink_strength_percent: int,
    relief_polarity: str,
    artboard_policy: str = ARTBOARD_LARGEST_COVERED_RECTANGLE,
) -> dict[str, Any]:
    """Name a READY + FRESH development by hash and resolve the raster options."""

    record, receipt = _development_record(
        document,
        _record_id(development_record_id, name="development record ID"),
    )
    return developed_rubbing_recipe(
        development_record_id=record.id,
        development_sha256=str(receipt["unwrap_sha256"]),
        development_recipe_hash=record.recipe_hash,
        pixels_per_mm=pixels_per_mm,
        margin_um=margin_um,
        reference_radius_um=reference_radius_um,
        depth_quantization_um=depth_quantization_um,
        black_point_um=black_point_um,
        ink_strength_percent=ink_strength_percent,
        relief_polarity=relief_polarity,
        artboard_policy=artboard_policy,
    )


def estimate_developed_rubbing_resources(
    receipt: Mapping[str, Any],
    recipe: Mapping[str, Any],
    *,
    source_vertex_count: int,
    source_face_count: int,
    source_geometry_bytes: int,
) -> DigitalRubbingResourceEstimate:
    """Bound the raster from the development's exact extent, before any work.

    The development receipt already states the strip's bounds in micrometres,
    so the artboard is known without recomputing the unwrap.  The byte figure
    over-approximates the depth buffers, integral images, masks, the output
    raster, the recomputed development, and the source geometry it is cut
    from; it is an admission estimate, not an allocator promise.
    """

    validated = validate_developed_rubbing_recipe(recipe)
    pixel_policy = validated["pixel_policy"]
    assert isinstance(pixel_policy, Mapping)
    pixels_per_mm = int(pixel_policy["pixels_per_mm"])
    margin_pixels = int(pixel_policy["margin_pixels"])
    bounds = receipt.get("bounds_um")
    if not isinstance(bounds, Mapping):
        raise ArtifactDevelopedRubbingError("development receipt bounds are invalid")
    for name in ("source_vertex_count", "source_face_count", "source_geometry_bytes"):
        _strict_int(locals()[name], name=name, minimum=0, maximum=2**62)
    extents: list[int] = []
    for axis_name in ("u", "v"):
        minimum_um = _strict_int(
            bounds.get(f"minimum_{axis_name}"),
            name=f"bounds_um.minimum_{axis_name}",
            minimum=0,
            maximum=2**52,
        )
        maximum_um = _strict_int(
            bounds.get(f"maximum_{axis_name}"),
            name=f"bounds_um.maximum_{axis_name}",
            minimum=1,
            maximum=2**52,
        )
        content_min = math.floor(float(minimum_um) / 1000.0 * float(pixels_per_mm))
        content_max = math.ceil(float(maximum_um) / 1000.0 * float(pixels_per_mm))
        extents.append(content_max - content_min + 2 * margin_pixels)
    width, height = extents
    if width <= 0 or height <= 0:
        raise ArtifactDevelopedRubbingError("developed artboard has zero extent")
    if width > MAX_RUBBING_DIMENSION or height > MAX_RUBBING_DIMENSION:
        raise ArtifactDevelopedRubbingError(
            "developed artboard exceeds the dimension safety limit"
        )
    pixel_count = width * height
    if pixel_count > MAX_RUBBING_PIXELS:
        raise ArtifactDevelopedRubbingError(
            "developed artboard exceeds the pixel safety limit; choose a lower "
            "physical resolution"
        )
    development_bytes = (
        int(receipt.get("vertex_count", 0)) * (16 + 8 + 8)
        + int(receipt.get("face_count", 0)) * (12 + 8)
    )
    estimated_peak_bytes = (
        pixel_count * RUBBING_ESTIMATED_PEAK_BYTES_PER_PIXEL
        + int(source_geometry_bytes) * RUBBING_ESTIMATE_GEOMETRY_MULTIPLIER
        + development_bytes * RUBBING_ESTIMATE_GEOMETRY_MULTIPLIER
        + RUBBING_ESTIMATE_FIXED_OVERHEAD_BYTES
    )
    try:
        return DigitalRubbingResourceEstimate(
            width_pixels=width,
            height_pixels=height,
            pixel_count=pixel_count,
            vertex_count=max(1, int(source_vertex_count)),
            face_count=max(1, int(source_face_count)),
            estimated_peak_bytes=estimated_peak_bytes,
        )
    except ArtifactRubbingError as exc:
        raise ArtifactDevelopedRubbingError(str(exc)) from exc


def derive_developed_rubbing(
    document: ArtifactDocument,
    mesh: MeshData,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[DevelopedRubbingRaster, dict[str, Any]]:
    """Recompute the named development from canonical-mm triangles, prove it,
    and draw the relief on it.

    The development record stores a receipt, not coordinates.  Recomputing it
    from its own recipe and requiring the receipt and payload hash to match
    is what lets the rubbing claim it was drawn on exactly that development.
    """

    raise_if_cancelled(cancellation_probe)
    validated = validate_developed_rubbing_recipe(recipe)
    record, receipt = development_record_for_recipe(document, validated)
    raise_if_cancelled(cancellation_probe)
    try:
        unwrap, development_qc, radius = extract_tile_unwrap_development(
            mesh,
            record.recipe,
            cancellation_probe=cancellation_probe,
        )
    except ArtifactTileUnwrapError as exc:
        raise ArtifactDevelopedRubbingError(str(exc)) from exc
    selection_sha256 = str(receipt["selection_sha256"])
    if unwrap.receipt(selection_sha256=selection_sha256) != receipt:
        raise ArtifactDevelopedRubbingError(
            "recomputed development does not match its record receipt"
        )
    payload = unwrap.canonical_payload_bytes(selection_sha256=selection_sha256)
    if hashlib.sha256(payload).hexdigest() != receipt["unwrap_sha256"]:
        raise ArtifactDevelopedRubbingError(
            "recomputed development payload does not match its record"
        )
    raise_if_cancelled(cancellation_probe)
    raster, qc = extract_developed_rubbing(
        unwrap,
        radius,
        validated,
        cancellation_probe=cancellation_probe,
    )
    qc = {
        **qc,
        "development_distortion_max_millionths": int(
            development_qc["distortion_max_millionths"]
        ),
        "development_distortion_mean_millionths": int(
            development_qc["distortion_mean_millionths"]
        ),
        "development_record_id": record.id,
    }
    return raster, qc


@dataclass(frozen=True, slots=True)
class DevelopedRubbingComputation:
    context: OperationContext
    projection_snapshot: ArtifactProjectionSnapshot
    raster: DevelopedRubbingRaster
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.context, OperationContext):
            raise ArtifactDevelopedRubbingError("context must be an OperationContext")
        if not isinstance(self.projection_snapshot, ArtifactProjectionSnapshot):
            raise ArtifactDevelopedRubbingError("projection_snapshot is invalid")
        if not isinstance(self.raster, DevelopedRubbingRaster):
            raise ArtifactDevelopedRubbingError(
                "raster must be a DevelopedRubbingRaster"
            )
        validated_recipe = validate_developed_rubbing_recipe(self.recipe)
        if canonical_recipe_hash(validated_recipe) != self.context.recipe_hash:
            raise ArtifactDevelopedRubbingError(
                "developed rubbing recipe does not match captured context"
            )
        development = validated_recipe["development"]
        assert isinstance(development, Mapping)
        if self.raster.development_sha256 != development["unwrap_sha256"]:
            raise ArtifactDevelopedRubbingError(
                "developed rubbing raster was drawn on another development"
            )
        snapshot = self.projection_snapshot
        if (
            snapshot.geometry_revision_id != self.context.geometry_revision_id
            or snapshot.source_metadata_revision_id
            != self.context.source_metadata_revision_id
            or snapshot.align_revision_id != self.context.align_revision_id
            or tuple(self.context.source_asset_ids) != (snapshot.source_asset_id,)
        ):
            raise ArtifactDevelopedRubbingError(
                "developed rubbing projection snapshot does not match captured context"
            )
        object.__setattr__(self, "recipe", MappingProxyType(validated_recipe))
        object.__setattr__(self, "qc", MappingProxyType(dict(self.qc)))

    @property
    def development_record_id(self) -> str:
        development = self.recipe["development"]
        assert isinstance(development, Mapping)
        return str(development["record_id"])

    def recipe_dict(self) -> dict[str, Any]:
        return dict(self.recipe)

    def qc_dict(self) -> dict[str, Any]:
        return dict(self.qc)


def _compute_with_recipe(
    session: ArtifactSession,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None,
) -> DevelopedRubbingComputation:
    try:
        context = session.capture_operation(recipe=recipe)
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactDevelopedRubbingError(str(exc)) from exc
    raise_if_cancelled(cancellation_probe)
    raster, qc = derive_developed_rubbing(
        session.document,
        projection.mesh,
        recipe,
        cancellation_probe=cancellation_probe,
    )
    raise_if_cancelled(cancellation_probe)
    return DevelopedRubbingComputation(
        context=context,
        projection_snapshot=projection.snapshot,
        raster=raster,
        recipe=recipe,
        qc=qc,
    )


def compute_developed_rubbing(
    session: ArtifactSession,
    development_record_id: str,
    *,
    pixels_per_mm: int,
    margin_um: int,
    reference_radius_um: int,
    depth_quantization_um: int,
    black_point_um: int,
    ink_strength_percent: int,
    relief_polarity: str,
    artboard_policy: str = ARTBOARD_LARGEST_COVERED_RECTANGLE,
    cancellation_probe: CancellationProbe | None = None,
) -> DevelopedRubbingComputation:
    if not isinstance(session, ArtifactSession):
        raise ArtifactDevelopedRubbingError("session must be an ArtifactSession")
    raise_if_cancelled(cancellation_probe)
    recipe = developed_rubbing_recipe_for_record(
        session.document,
        development_record_id,
        pixels_per_mm=pixels_per_mm,
        margin_um=margin_um,
        reference_radius_um=reference_radius_um,
        depth_quantization_um=depth_quantization_um,
        black_point_um=black_point_um,
        ink_strength_percent=ink_strength_percent,
        relief_polarity=relief_polarity,
        artboard_policy=artboard_policy,
    )
    return _compute_with_recipe(session, recipe, cancellation_probe=cancellation_probe)


def compute_developed_rubbing_from_recipe(
    session: ArtifactSession,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> DevelopedRubbingComputation:
    if not isinstance(session, ArtifactSession):
        raise ArtifactDevelopedRubbingError("session must be an ArtifactSession")
    validated = validate_developed_rubbing_recipe(recipe)
    return _compute_with_recipe(
        session,
        validated,
        cancellation_probe=cancellation_probe,
    )


def developed_rubbing_computation_matches_active_projection(
    session: ArtifactSession,
    computation: DevelopedRubbingComputation,
) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(
        computation, DevelopedRubbingComputation
    ):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    expected = computation.projection_snapshot
    return (
        current.document_id == expected.document_id
        and current.document_schema_version == expected.document_schema_version
        and current.source_asset_id == expected.source_asset_id
        and current.geometry_revision_id == expected.geometry_revision_id
        and current.source_metadata_revision_id == expected.source_metadata_revision_id
        and current.align_revision_id == expected.align_revision_id
        and current.geometry_sha256 == expected.geometry_sha256
        and current.geometry_hash_scope == expected.geometry_hash_scope
        and current.matrix4x4 == expected.matrix4x4
    )


def require_current_developed_rubbing_computation(
    session: ArtifactSession,
    computation: DevelopedRubbingComputation,
) -> None:
    if not developed_rubbing_computation_matches_active_projection(
        session, computation
    ):
        raise ArtifactDevelopedRubbingError(
            "developed rubbing computation is stale for the active scene projection"
        )


def append_developed_rubbing_record_from_context(
    document: ArtifactDocument,
    *,
    context: OperationContext,
    raster: DevelopedRubbingRaster,
    recipe: Mapping[str, Any],
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
    qc: Mapping[str, Any] | None = None,
) -> ArtifactDocument:
    """Append the raster receipt; the development is always a dependency."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactDevelopedRubbingError("document must be an ArtifactDocument")
    if not isinstance(context, OperationContext):
        raise ArtifactDevelopedRubbingError("context must be an OperationContext")
    if not isinstance(raster, DevelopedRubbingRaster):
        raise ArtifactDevelopedRubbingError("raster must be a DevelopedRubbingRaster")
    validated_recipe = validate_developed_rubbing_recipe(recipe)
    if canonical_recipe_hash(validated_recipe) != context.recipe_hash:
        raise ArtifactDevelopedRubbingError(
            "developed rubbing recipe does not match the operation context"
        )
    development = validated_recipe["development"]
    assert isinstance(development, Mapping)
    receipt = validate_developed_rubbing_receipt(raster.receipt())
    if receipt["development_sha256"] != development["unwrap_sha256"]:
        raise ArtifactDevelopedRubbingError(
            "developed rubbing raster was drawn on another development"
        )
    pixel_policy = validated_recipe["pixel_policy"]
    if not isinstance(pixel_policy, Mapping) or pixel_policy.get(
        "pixels_per_meter"
    ) != receipt["pixels_per_meter"]:
        raise ArtifactDevelopedRubbingError(
            "developed rubbing recipe and receipt pixel grids differ"
        )
    development_id = str(development["record_id"])
    development_record, development_receipt = _development_record(
        document, development_id
    )
    if development_receipt["unwrap_sha256"] != development["unwrap_sha256"]:
        raise ArtifactDevelopedRubbingError(
            "development record no longer carries the development this raster used"
        )
    if development_record.recipe_hash != development["recipe_hash"]:
        raise ArtifactDevelopedRubbingError(
            "development record recipe changed after the raster was drawn"
        )
    dependencies = tuple(str(value) for value in depends_on_record_ids)
    if development_id not in dependencies:
        dependencies = (development_id, *dependencies)
    computed_qc = raster.qc_summary()
    for key, value in dict(qc or {}).items():
        if key in computed_qc and computed_qc[key] != value:
            raise ArtifactDevelopedRubbingError(
                f"caller QC cannot override computed field {key!r}"
            )
        computed_qc[key] = value
    _validate_qc_against_receipt(computed_qc, receipt)
    receipt_bytes = canonical_json_bytes(receipt)
    if len(receipt_bytes) > MAX_DEVELOPED_RUBBING_RECEIPT_BYTES:
        raise ArtifactDevelopedRubbingError(
            "developed rubbing receipt exceeds its size limit"
        )
    extensions = {
        DEVELOPED_RUBBING_RECEIPT_EXTENSION_KEY: {
            "media_type": DEVELOPED_RUBBING_RECEIPT_MEDIA_TYPE,
            "receipt": receipt,
            "receipt_byte_length": len(receipt_bytes),
            "receipt_sha256": canonical_json_sha256(receipt),
            "schema_version": DEVELOPED_RUBBING_RASTER_SCHEMA_VERSION,
        }
    }
    try:
        return document.append_record_from_context(
            context=context,
            id=record_id,
            type=DEVELOPED_RUBBING_RECORD_TYPE,
            geometry_ref=raster.geometry_ref,
            recipe=validated_recipe,
            qc=computed_qc,
            lifecycle_status=RecordLifecycleStatus.READY,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=dependencies,
            extensions=extensions,
        )
    except ArtifactDocumentError as exc:
        raise ArtifactDevelopedRubbingError(str(exc)) from exc


def commit_developed_rubbing(
    session: ArtifactSession,
    computation: DevelopedRubbingComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    """Append the raster receipt at its captured context, even if historical."""

    if not isinstance(session, ArtifactSession):
        raise ArtifactDevelopedRubbingError("session must be an ArtifactSession")
    if not isinstance(computation, DevelopedRubbingComputation):
        raise ArtifactDevelopedRubbingError(
            "computation must be a DevelopedRubbingComputation"
        )
    try:
        document = append_developed_rubbing_record_from_context(
            session.document,
            context=computation.context,
            raster=computation.raster,
            recipe=computation.recipe,
            record_id=record_id,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=depends_on_record_ids,
            qc=computation.qc,
        )
        return session.with_document(document)
    except ArtifactSessionError as exc:
        raise ArtifactDevelopedRubbingError(str(exc)) from exc


def developed_rubbing_receipt_from_record(record: DerivedRecord) -> dict[str, Any]:
    if not isinstance(record, DerivedRecord):
        raise ArtifactDevelopedRubbingError("record must be a DerivedRecord")
    if record.type != DEVELOPED_RUBBING_RECORD_TYPE:
        raise ArtifactDevelopedRubbingError("record is not a developed rubbing record")
    descriptor = _exact_keys(
        record.extensions.get(DEVELOPED_RUBBING_RECEIPT_EXTENSION_KEY),
        {
            "media_type",
            "receipt",
            "receipt_byte_length",
            "receipt_sha256",
            "schema_version",
        },
        name="developed rubbing descriptor",
    )
    if descriptor["media_type"] != DEVELOPED_RUBBING_RECEIPT_MEDIA_TYPE:
        raise ArtifactDevelopedRubbingError(
            "developed rubbing descriptor media type is invalid"
        )
    if descriptor["schema_version"] != DEVELOPED_RUBBING_RASTER_SCHEMA_VERSION:
        raise ArtifactDevelopedRubbingError(
            "developed rubbing descriptor schema is invalid"
        )
    receipt = validate_developed_rubbing_receipt(descriptor["receipt"])
    receipt_bytes = canonical_json_bytes(receipt)
    if descriptor["receipt_byte_length"] != len(receipt_bytes):
        raise ArtifactDevelopedRubbingError(
            "developed rubbing receipt byte length is invalid"
        )
    if descriptor["receipt_sha256"] != canonical_json_sha256(receipt):
        raise ArtifactDevelopedRubbingError(
            "developed rubbing receipt SHA-256 is invalid"
        )
    if record.geometry_ref != (
        f"{DEVELOPED_RUBBING_GEOMETRY_REF_PREFIX}{receipt['raster_sha256']}"
    ):
        raise ArtifactDevelopedRubbingError(
            "developed rubbing geometry_ref does not match receipt"
        )
    recipe = validate_developed_rubbing_recipe(record.recipe)
    development = recipe["development"]
    assert isinstance(development, Mapping)
    if development["unwrap_sha256"] != receipt["development_sha256"]:
        raise ArtifactDevelopedRubbingError(
            "developed rubbing record recipe and receipt name different developments"
        )
    if str(development["record_id"]) not in record.depends_on_record_ids:
        raise ArtifactDevelopedRubbingError(
            "developed rubbing record does not depend on its development"
        )
    pixel_policy = recipe["pixel_policy"]
    if not isinstance(pixel_policy, Mapping) or pixel_policy.get(
        "pixels_per_meter"
    ) != receipt["pixels_per_meter"]:
        raise ArtifactDevelopedRubbingError(
            "developed rubbing record pixel grid does not match receipt"
        )
    record_qc = record.to_dict()["qc"]
    assert isinstance(record_qc, dict)
    _validate_qc_against_receipt(record_qc, receipt)
    return receipt


def validate_developed_rubbing_records(document: ArtifactDocument) -> None:
    """Check every developed rubbing against its receipt and its development."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactDevelopedRubbingError("document must be an ArtifactDocument")
    for record in document.records:
        if record.type != DEVELOPED_RUBBING_RECORD_TYPE:
            continue
        receipt = developed_rubbing_receipt_from_record(record)
        development = record.recipe["development"]
        assert isinstance(development, Mapping)
        development_record = document.record_index.get(str(development["record_id"]))
        if development_record is None:
            raise ArtifactDevelopedRubbingError(
                f"developed rubbing {record.id!r} names a missing development"
            )
        if development_record.type != TILE_UNWRAP_RECORD_TYPE:
            raise ArtifactDevelopedRubbingError(
                f"developed rubbing {record.id!r} names a non-development record"
            )
        try:
            development_receipt = tile_unwrap_receipt_from_record(development_record)
        except ArtifactTileUnwrapRecordError as exc:
            raise ArtifactDevelopedRubbingError(str(exc)) from exc
        if development_receipt["unwrap_sha256"] != receipt["development_sha256"]:
            raise ArtifactDevelopedRubbingError(
                f"developed rubbing {record.id!r} was drawn on a development its "
                "record no longer carries"
            )
        if development_record.recipe_hash != development["recipe_hash"]:
            raise ArtifactDevelopedRubbingError(
                f"developed rubbing {record.id!r} names a development recipe its "
                "record no longer carries"
            )


__all__ = [
    "ARTBOARD_DEVELOPMENT_BOUNDS",
    "ARTBOARD_LARGEST_COVERED_RECTANGLE",
    "ARTBOARD_POLICIES",
    "ArtifactDevelopedRubbingError",
    "DEVELOPED_RUBBING_ALGORITHM",
    "DEVELOPED_RUBBING_ALGORITHM_VERSION",
    "DEVELOPED_RUBBING_COORDINATE_SPACE",
    "DEVELOPED_RUBBING_DEPTH_MEASURE",
    "DEVELOPED_RUBBING_GEOMETRY_REF_PREFIX",
    "DEVELOPED_RUBBING_OPERATION_KIND",
    "DEVELOPED_RUBBING_RASTER_HASH_SCOPE",
    "DEVELOPED_RUBBING_RASTER_SCHEMA_VERSION",
    "DEVELOPED_RUBBING_RECEIPT_EXTENSION_KEY",
    "DEVELOPED_RUBBING_RECEIPT_MEDIA_TYPE",
    "DEVELOPED_RUBBING_RECORD_TYPE",
    "DevelopedRubbingComputation",
    "DevelopedRubbingRaster",
    "MAX_DEVELOPED_RUBBING_RECEIPT_BYTES",
    "append_developed_rubbing_record_from_context",
    "commit_developed_rubbing",
    "compute_developed_rubbing",
    "compute_developed_rubbing_from_recipe",
    "derive_developed_rubbing",
    "developed_rubbing_computation_matches_active_projection",
    "development_record_for_recipe",
    "developed_rubbing_receipt_from_record",
    "developed_rubbing_recipe",
    "developed_rubbing_recipe_for_record",
    "estimate_developed_rubbing_resources",
    "extract_developed_rubbing",
    "require_current_developed_rubbing_computation",
    "validate_developed_rubbing_receipt",
    "validate_developed_rubbing_recipe",
    "validate_developed_rubbing_records",
]
