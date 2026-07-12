"""Authoritative, headless Digital Rubbing in canonical millimetres.

The legacy visualizer is intentionally not used here.  This module projects
all canonical-mm triangles into an explicit six-view frame, resolves the
front-most surface at physical pixel centres, quantizes depth to integer
micrometre ticks, and applies a mask-aware local-mean relief operator.  No
OpenGL frame, screenshot, vertex splat, adaptive resolution, Gaussian backend,
or silent fallback participates in the measured raster.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .alignment_utils import require_affine_matrix4x4
from .artifact_document import OperationContext, canonical_recipe_hash
from .artifact_outline_extractor import OutlineView, outline_frame
from .artifact_scene_adapter import ArtifactProjectionSnapshot
from .artifact_session import ArtifactSession, ArtifactSessionError
from .artifact_vector_extractor import _validated_mesh_arrays
from .artifact_vector_record import PlanarFrame
from .canonical_json import canonical_json_bytes


RUBBING_ALGORITHM = "archmeshrubbing.orthographic_local_mean_relief"
RUBBING_ALGORITHM_VERSION = "1.0.0"
RUBBING_RASTER_SCHEMA_VERSION = "1.0.0"
RUBBING_COORDINATE_SPACE = "canonical_mm_orthographic_raster/v1"
RUBBING_PIXEL_FORMAT = "grayscale-alpha-8/v1"
RUBBING_RASTER_HASH_SCOPE = "header-rfc8785+pixels-ga8-row-major/v1"

DEFAULT_RUBBING_PIXELS_PER_MM = 10
DEFAULT_RUBBING_MARGIN_UM = 2_000
DEFAULT_RUBBING_REFERENCE_RADIUS_UM = 3_000
DEFAULT_RUBBING_DEPTH_QUANTIZATION_UM = 10
DEFAULT_RUBBING_BLACK_POINT_UM = 250
DEFAULT_RUBBING_INK_STRENGTH_PERCENT = 100
DEFAULT_RUBBING_POLARITY = "bidirectional"

MAX_RUBBING_VERTICES = 5_000_000
MAX_RUBBING_FACES = 2_000_000
MAX_RUBBING_PIXELS = 8_000_000
MAX_RUBBING_DIMENSION = 100_000
MAX_RUBBING_PIXELS_PER_MM = 100
MAX_RUBBING_REFERENCE_RADIUS_PIXELS = 512
MAX_RUBBING_TRIANGLE_PIXEL_TESTS = 250_000_000
MAX_RUBBING_GRID_INDEX = 2**48
MAX_RUBBING_DEPTH_TICKS = 2**40
MAX_RUBBING_INTEGRAL_SUM = 2**62
RASTER_ROW_BLOCK_SIZE = 128
RUBBING_ESTIMATED_PEAK_BYTES_PER_PIXEL = 96
RUBBING_ESTIMATE_FIXED_OVERHEAD_BYTES = 32 * 1024 * 1024
# Source/session projection, float64 relative/projected/depth/local arrays,
# face widening, validator temporaries, and result ownership can coexist.  The
# multiplier is deliberately conservative for tiny rasters with millions of
# unreferenced vertices, where a per-pixel estimate alone is misleading.
RUBBING_ESTIMATE_GEOMETRY_MULTIPLIER = 8
# ``ArtifactSceneAdapter.materialize`` keeps the immutable source attributes
# alive while allocating disjoint UV/texture copies for the projection.  Count
# both resident arrays so admission remains conservative even though Digital
# Rubbing itself does not consume texture data.
RUBBING_ESTIMATE_MATERIALIZED_ATTRIBUTE_MULTIPLIER = 2

_SUPPORTED_POLARITIES = frozenset({"raised", "incised", "bidirectional"})


class ArtifactRubbingError(ValueError):
    """An authoritative Digital Rubbing result cannot be produced safely."""


@dataclass(frozen=True, slots=True)
class DigitalRubbingResourceEstimate:
    """Conservative preflight estimate without allocating raster-sized arrays."""

    width_pixels: int
    height_pixels: int
    pixel_count: int
    vertex_count: int
    face_count: int
    estimated_peak_bytes: int

    def __post_init__(self) -> None:
        values = {
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
            "pixel_count": self.pixel_count,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "estimated_peak_bytes": self.estimated_peak_bytes,
        }
        if any(type(value) is not int or value <= 0 for value in values.values()):
            raise ArtifactRubbingError(
                "Digital Rubbing resource estimate values must be positive integers"
            )


def _strict_int(
    value: object,
    *,
    field_name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactRubbingError(f"{field_name} must be an integer")
    number = int(value)
    if number < minimum or number > maximum:
        raise ArtifactRubbingError(
            f"{field_name} must be in the inclusive range {minimum}..{maximum}"
        )
    return number


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ArtifactRubbingError("internal integer scale conversion is invalid")
    return (numerator + denominator - 1) // denominator


def rubbing_materialized_attribute_bytes(
    *,
    uv_coords: np.ndarray | None,
    texture: np.ndarray | None,
) -> int:
    """Return UV/texture storage copied by scene materialization, without copying.

    The application calls this against the immutable source mesh before
    ``materialize()`` so a texture that cannot fit the configured memory budget
    is rejected before a second large allocation is attempted.
    """

    total = 0
    for field_name, value in (("uv_coords", uv_coords), ("texture", texture)):
        if value is None:
            continue
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise ArtifactRubbingError(
                f"{field_name} must be a non-object NumPy array for resource estimation"
            )
        total += int(value.nbytes)
    return total


def _view(value: object) -> OutlineView:
    try:
        return OutlineView(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactRubbingError(
            "view must be one of: top, bottom, front, back, right, left"
        ) from exc


def _polarity(value: object) -> str:
    if not isinstance(value, str) or value not in _SUPPORTED_POLARITIES:
        raise ArtifactRubbingError(
            "relief_polarity must be raised, incised, or bidirectional"
        )
    return value


def rubbing_recipe(
    view: OutlineView | str,
    *,
    pixels_per_mm: int,
    margin_um: int,
    reference_radius_um: int,
    depth_quantization_um: int,
    black_point_um: int,
    ink_strength_percent: int,
    relief_polarity: str,
) -> dict[str, Any]:
    """Resolve every physical/display option before context capture."""

    resolved_view = _view(view)
    ppm = _strict_int(
        pixels_per_mm,
        field_name="pixels_per_mm",
        minimum=1,
        maximum=MAX_RUBBING_PIXELS_PER_MM,
    )
    margin = _strict_int(
        margin_um,
        field_name="margin_um",
        minimum=0,
        maximum=1_000_000_000,
    )
    radius_um = _strict_int(
        reference_radius_um,
        field_name="reference_radius_um",
        minimum=1,
        maximum=1_000_000_000,
    )
    quantization_um = _strict_int(
        depth_quantization_um,
        field_name="depth_quantization_um",
        minimum=1,
        maximum=1_000_000,
    )
    black_um = _strict_int(
        black_point_um,
        field_name="black_point_um",
        minimum=1,
        maximum=1_000_000_000,
    )
    strength = _strict_int(
        ink_strength_percent,
        field_name="ink_strength_percent",
        minimum=1,
        maximum=400,
    )
    polarity = _polarity(relief_polarity)
    margin_pixels = _ceil_div(margin * ppm, 1000)
    reference_radius_pixels = max(1, _ceil_div(radius_um * ppm, 1000))
    if reference_radius_pixels > MAX_RUBBING_REFERENCE_RADIUS_PIXELS:
        raise ArtifactRubbingError(
            "reference radius exceeds the authoritative raster safety limit"
        )
    effective_black_point_um = max(1, _ceil_div(black_um * 100, strength))
    effective_black_point_ticks = max(
        1,
        _ceil_div(effective_black_point_um, quantization_um),
    )
    return {
        "algorithm": RUBBING_ALGORITHM,
        "algorithm_version": RUBBING_ALGORITHM_VERSION,
        "artboard_policy": "global_pixel_lattice_bounds_plus_margin/v1",
        "coordinate_space": RUBBING_COORDINATE_SPACE,
        "depth_policy": {
            "front_surface": "maximum_frame_normal_depth",
            "quantization_rounding": "nearest_ties_to_even/v1",
            "quantization_um": quantization_um,
        },
        "frame": outline_frame(resolved_view).to_dict(),
        "kind": "digital_rubbing",
        "pixel_policy": {
            "margin_pixels": margin_pixels,
            "margin_requested_um": margin,
            "pixel_centres": "half_integer_global_lattice/v1",
            "pixel_format": RUBBING_PIXEL_FORMAT,
            "pixels_per_meter": ppm * 1000,
            "pixels_per_mm": ppm,
            "row_order": "top_to_bottom_v_descending",
        },
        "rasterization_policy": {
            "backfaces": "include",
            "edge_rule": "barycentric_nonnegative_with_fixed_epsilon/v1",
            "multi_layer_policy": "frontmost_and_count_second_depth/v1",
            "projected_zero_area_faces": "drop_and_count",
            "sampling": "none",
            "z_tie": "geometry_invariant_max_depth",
        },
        "relief_policy": {
            "black_point_requested_um": black_um,
            "effective_black_point_ticks": effective_black_point_ticks,
            "effective_black_point_um": effective_black_point_um,
            "ink_strength_percent": strength,
            "minimum_reference_sample_count": 3,
            "polarity": polarity,
            "reference_filter": "masked_square_local_mean_integer_integral/v1",
            "reference_radius_pixels": reference_radius_pixels,
            "reference_radius_requested_um": radius_um,
            "tone_rounding": "nearest_half_up_integer/v1",
        },
        "resource_limits": {
            "max_dimension": MAX_RUBBING_DIMENSION,
            "max_faces": MAX_RUBBING_FACES,
            "max_pixels": MAX_RUBBING_PIXELS,
            "max_triangle_pixel_tests": MAX_RUBBING_TRIANGLE_PIXEL_TESTS,
            "max_vertices": MAX_RUBBING_VERTICES,
        },
        "view": resolved_view.value,
    }


def validate_rubbing_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(recipe, Mapping):
        raise ArtifactRubbingError("Digital Rubbing recipe must be an object")
    pixel_policy = recipe.get("pixel_policy")
    depth_policy = recipe.get("depth_policy")
    relief_policy = recipe.get("relief_policy")
    if not all(
        isinstance(value, Mapping)
        for value in (pixel_policy, depth_policy, relief_policy)
    ):
        raise ArtifactRubbingError("Digital Rubbing recipe policies are invalid")
    assert isinstance(pixel_policy, Mapping)
    assert isinstance(depth_policy, Mapping)
    assert isinstance(relief_policy, Mapping)
    expected = rubbing_recipe(
        _view(recipe.get("view")),
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
    if canonical_recipe_hash(recipe) != canonical_recipe_hash(expected):
        raise ArtifactRubbingError(
            "Digital Rubbing recipe does not match the production contract"
        )
    return expected


@dataclass(frozen=True, slots=True)
class DigitalRubbingRaster:
    pixels: np.ndarray
    frame: PlanarFrame
    view: OutlineView | str
    pixels_per_meter: int
    minimum_u_pixel_index: int
    minimum_v_pixel_index: int

    def __post_init__(self) -> None:
        array = np.asarray(self.pixels)
        if (
            array.dtype != np.uint8
            or array.ndim != 3
            or array.shape[2] != 2
            or array.shape[0] <= 0
            or array.shape[1] <= 0
        ):
            raise ArtifactRubbingError("raster pixels must be a non-empty HxWx2 uint8 array")
        if array.shape[0] * array.shape[1] > MAX_RUBBING_PIXELS:
            raise ArtifactRubbingError("raster pixels exceed the safety limit")
        if not isinstance(self.frame, PlanarFrame):
            raise ArtifactRubbingError("raster frame must be a PlanarFrame")
        resolved_view = _view(self.view)
        if self.frame != outline_frame(resolved_view):
            raise ArtifactRubbingError("raster frame does not match its six-view side")
        ppm = _strict_int(
            self.pixels_per_meter,
            field_name="pixels_per_meter",
            minimum=1000,
            maximum=MAX_RUBBING_PIXELS_PER_MM * 1000,
        )
        if ppm % 1000 != 0:
            raise ArtifactRubbingError("pixels_per_meter must encode an integer pixels/mm")
        for name, value in (
            ("minimum_u_pixel_index", self.minimum_u_pixel_index),
            ("minimum_v_pixel_index", self.minimum_v_pixel_index),
        ):
            _strict_int(
                value,
                field_name=name,
                minimum=-MAX_RUBBING_GRID_INDEX,
                maximum=MAX_RUBBING_GRID_INDEX,
            )
        copied = np.ascontiguousarray(array).copy()
        copied.setflags(write=False)
        object.__setattr__(self, "pixels", copied)
        object.__setattr__(self, "view", resolved_view)
        object.__setattr__(self, "pixels_per_meter", ppm)

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
            "coordinate_space": RUBBING_COORDINATE_SPACE,
            "frame": self.frame.to_dict(),
            "height_pixels": self.height_pixels,
            "minimum_u_pixel_index": self.minimum_u_pixel_index,
            "minimum_v_pixel_index": self.minimum_v_pixel_index,
            "pixel_format": RUBBING_PIXEL_FORMAT,
            "pixels_per_meter": self.pixels_per_meter,
            "raster_hash_scope": RUBBING_RASTER_HASH_SCOPE,
            "row_order": "top_to_bottom_v_descending",
            "schema_version": RUBBING_RASTER_SCHEMA_VERSION,
            "view": OutlineView(self.view).value,
            "width_pixels": self.width_pixels,
        }

    @property
    def raster_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"archmeshrubbing.digital-rubbing-raster\0")
        digest.update(canonical_json_bytes(self.semantic_header()))
        digest.update(b"\0")
        digest.update(self.pixels.tobytes(order="C"))
        return digest.hexdigest()

    @property
    def geometry_ref(self) -> str:
        return (
            "urn:archmeshrubbing:digital-rubbing-raster:sha256:"
            f"{self.raster_sha256}"
        )

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
        gray = self.pixels[:, :, 0]
        alpha = self.pixels[:, :, 1]
        covered = alpha == 255
        covered_count = int(np.count_nonzero(covered))
        if covered_count:
            covered_gray = gray[covered]
            minimum = int(covered_gray.min())
            maximum = int(covered_gray.max())
            ink_sum = int(np.sum(255 - covered_gray, dtype=np.int64))
            inked_count = int(np.count_nonzero(covered_gray < 255))
        else:
            minimum = 255
            maximum = 255
            ink_sum = 0
            inked_count = 0
        if np.any((alpha != 0) & (alpha != 255)):
            raise ArtifactRubbingError("raster alpha mask must be binary")
        return {
            "alpha_binary": True,
            "covered_gray_max": maximum,
            "covered_gray_min": minimum,
            "covered_pixel_count": covered_count,
            "height_pixels": self.height_pixels,
            "ink_sum": ink_sum,
            "inked_pixel_count": inked_count,
            "pixel_count": self.width_pixels * self.height_pixels,
            "pixel_format": RUBBING_PIXEL_FORMAT,
            "pixels_per_meter": self.pixels_per_meter,
            "raster_sha256": self.raster_sha256,
            "raw_pixel_sha256": self.raw_pixel_sha256,
            "width_pixels": self.width_pixels,
        }


@dataclass(frozen=True, slots=True)
class ArtifactRubbingComputation:
    context: OperationContext
    projection_snapshot: ArtifactProjectionSnapshot
    raster: DigitalRubbingRaster
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.context, OperationContext):
            raise ArtifactRubbingError("context must be an OperationContext")
        if not isinstance(self.projection_snapshot, ArtifactProjectionSnapshot):
            raise ArtifactRubbingError("projection_snapshot is invalid")
        if not isinstance(self.raster, DigitalRubbingRaster):
            raise ArtifactRubbingError("raster must be a DigitalRubbingRaster")
        validated_recipe = validate_rubbing_recipe(self.recipe)
        if canonical_recipe_hash(validated_recipe) != self.context.recipe_hash:
            raise ArtifactRubbingError("rubbing recipe does not match captured context")
        snapshot = self.projection_snapshot
        if (
            snapshot.geometry_revision_id != self.context.geometry_revision_id
            or snapshot.source_metadata_revision_id
            != self.context.source_metadata_revision_id
            or snapshot.align_revision_id != self.context.align_revision_id
            or tuple(self.context.source_asset_ids) != (snapshot.source_asset_id,)
        ):
            raise ArtifactRubbingError(
                "rubbing projection snapshot does not match captured context"
            )
        frozen_recipe = MappingProxyType(validated_recipe)
        frozen_qc = MappingProxyType(dict(self.qc))
        object.__setattr__(self, "recipe", frozen_recipe)
        object.__setattr__(self, "qc", frozen_qc)

    def recipe_dict(self) -> dict[str, Any]:
        return dict(self.recipe)

    def qc_dict(self) -> dict[str, Any]:
        return dict(self.qc)


def _rasterize_front_depth(
    vertices: np.ndarray,
    faces: np.ndarray,
    frame: PlanarFrame,
    *,
    pixels_per_mm: int,
    margin_pixels: int,
    layer_separation_mm: float,
) -> tuple[np.ndarray, int, int, dict[str, int]]:
    origin = np.asarray(frame.origin_world_mm, dtype=np.float64)
    u_axis = np.asarray(frame.u_axis_world, dtype=np.float64)
    v_axis = np.asarray(frame.v_axis_world, dtype=np.float64)
    normal = np.asarray(frame.normal_world, dtype=np.float64)
    relative = vertices - origin
    projected = np.column_stack((relative @ u_axis, relative @ v_axis))
    depths = relative @ normal
    referenced = np.unique(faces.reshape(-1))
    scaled = projected[referenced] * float(pixels_per_mm)
    if not np.isfinite(scaled).all() or not np.isfinite(depths[referenced]).all():
        raise ArtifactRubbingError("rubbing projection contains non-finite coordinates")
    if float(np.max(np.abs(scaled))) > MAX_RUBBING_GRID_INDEX:
        raise ArtifactRubbingError("rubbing projection exceeds the pixel-grid safety range")
    content_min_u = math.floor(float(np.min(scaled[:, 0])))
    content_max_u = math.ceil(float(np.max(scaled[:, 0])))
    content_min_v = math.floor(float(np.min(scaled[:, 1])))
    content_max_v = math.ceil(float(np.max(scaled[:, 1])))
    minimum_u = content_min_u - margin_pixels
    maximum_u = content_max_u + margin_pixels
    minimum_v = content_min_v - margin_pixels
    maximum_v = content_max_v + margin_pixels
    width = maximum_u - minimum_u
    height = maximum_v - minimum_v
    if width <= 0 or height <= 0:
        raise ArtifactRubbingError("rubbing artboard has zero physical extent")
    if width > MAX_RUBBING_DIMENSION or height > MAX_RUBBING_DIMENSION:
        raise ArtifactRubbingError("rubbing artboard exceeds the dimension safety limit")
    if width * height > MAX_RUBBING_PIXELS:
        raise ArtifactRubbingError(
            "rubbing artboard exceeds the pixel safety limit; choose a lower physical resolution"
        )

    local = projected * float(pixels_per_mm) - np.array(
        [minimum_u, minimum_v], dtype=np.float64
    )
    depth_buffer = np.full((height, width), -np.inf, dtype=np.float64)
    second_depth_buffer = np.full((height, width), -np.inf, dtype=np.float64)
    projected_zero_area = 0
    triangle_pixel_tests = 0
    epsilon = 1e-12
    for face in faces:
        triangle = local[face]
        triangle_depth = depths[face]
        ax, ay = float(triangle[0, 0]), float(triangle[0, 1])
        bx, by = float(triangle[1, 0]), float(triangle[1, 1])
        cx, cy = float(triangle[2, 0]), float(triangle[2, 1])
        denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if denominator == 0.0:
            projected_zero_area += 1
            continue
        minimum_x = max(0, int(math.floor(min(ax, bx, cx) - 0.5)))
        maximum_x = min(width - 1, int(math.ceil(max(ax, bx, cx) - 0.5)))
        minimum_y = max(0, int(math.floor(min(ay, by, cy) - 0.5)))
        maximum_y = min(height - 1, int(math.ceil(max(ay, by, cy) - 0.5)))
        if minimum_x > maximum_x or minimum_y > maximum_y:
            continue
        bbox_tests = (maximum_x - minimum_x + 1) * (maximum_y - minimum_y + 1)
        triangle_pixel_tests += bbox_tests
        if triangle_pixel_tests > MAX_RUBBING_TRIANGLE_PIXEL_TESTS:
            raise ArtifactRubbingError(
                "rubbing exceeds the triangle-pixel work safety limit"
            )
        xs = np.arange(minimum_x, maximum_x + 1, dtype=np.float64) + 0.5
        for y_start in range(minimum_y, maximum_y + 1, RASTER_ROW_BLOCK_SIZE):
            y_stop = min(maximum_y + 1, y_start + RASTER_ROW_BLOCK_SIZE)
            ys = np.arange(y_start, y_stop, dtype=np.float64)[:, None] + 0.5
            x_grid = xs[None, :]
            w0 = ((by - cy) * (x_grid - cx) + (cx - bx) * (ys - cy)) / denominator
            w1 = ((cy - ay) * (x_grid - cx) + (ax - cx) * (ys - cy)) / denominator
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -epsilon) & (w1 >= -epsilon) & (w2 >= -epsilon)
            if not bool(np.any(inside)):
                continue
            interpolated = (
                w0 * float(triangle_depth[0])
                + w1 * float(triangle_depth[1])
                + w2 * float(triangle_depth[2])
            )
            block = depth_buffer[y_start:y_stop, minimum_x : maximum_x + 1]
            second = second_depth_buffer[
                y_start:y_stop, minimum_x : maximum_x + 1
            ]
            new_front = inside & (interpolated > block + layer_separation_mm)
            if bool(np.any(new_front)):
                second[new_front] = np.maximum(second[new_front], block[new_front])
                block[new_front] = interpolated[new_front]
            new_second = (
                inside
                & (interpolated < block - layer_separation_mm)
                & (interpolated > second + layer_separation_mm)
            )
            if bool(np.any(new_second)):
                second[new_second] = interpolated[new_second]
    covered = np.isfinite(depth_buffer)
    covered_count = int(np.count_nonzero(covered))
    if covered_count == 0:
        raise ArtifactRubbingError("rubbing projection covers no physical pixel centres")
    multiple_layers = np.isfinite(second_depth_buffer)
    multi_layer_pixel_count = int(np.count_nonzero(multiple_layers))
    maximum_second_layer_gap_um = 0
    if multi_layer_pixel_count:
        maximum_second_layer_gap_um = int(
            round(
                float(
                    np.max(
                        depth_buffer[multiple_layers]
                        - second_depth_buffer[multiple_layers]
                    )
                )
                * 1000.0
            )
        )
    return depth_buffer, minimum_u, minimum_v, {
        "artboard_height_pixels": height,
        "artboard_width_pixels": width,
        "covered_pixel_count": covered_count,
        "maximum_second_layer_gap_um_rounded": maximum_second_layer_gap_um,
        "multi_layer_pixel_count": multi_layer_pixel_count,
        "projected_zero_area_face_count": projected_zero_area,
        "projected_nonzero_area_face_count": int(faces.shape[0])
        - projected_zero_area,
        "triangle_pixel_test_count": triangle_pixel_tests,
    }


def _integral_image(values: np.ndarray) -> np.ndarray:
    integral = np.asarray(values, dtype=np.int64).copy()
    np.cumsum(integral, axis=0, dtype=np.int64, out=integral)
    np.cumsum(integral, axis=1, dtype=np.int64, out=integral)
    return np.pad(integral, ((1, 0), (1, 0)), mode="constant")


def _render_local_relief(
    depth_buffer: np.ndarray,
    *,
    depth_quantization_um: int,
    reference_radius_pixels: int,
    effective_black_point_ticks: int,
    relief_polarity: str,
    minimum_reference_sample_count: int,
) -> tuple[np.ndarray, dict[str, int]]:
    covered = np.isfinite(depth_buffer)
    covered_depths = depth_buffer[covered]
    minimum_depth = float(np.min(covered_depths))
    span_mm = float(np.max(covered_depths) - minimum_depth)
    scaled = (depth_buffer[covered] - minimum_depth) * (
        1000.0 / float(depth_quantization_um)
    )
    if not np.isfinite(scaled).all() or float(np.max(scaled)) > MAX_RUBBING_DEPTH_TICKS:
        raise ArtifactRubbingError("rubbing depth span exceeds the quantized safety range")
    height = int(np.size(depth_buffer, axis=0))
    width = int(np.size(depth_buffer, axis=1))
    ticks = np.zeros((height, width), dtype=np.int64)
    ticks[covered] = np.rint(scaled).astype(np.int64)
    maximum_tick = int(ticks[covered].max())
    covered_count = int(np.count_nonzero(covered))
    if maximum_tick * covered_count > MAX_RUBBING_INTEGRAL_SUM:
        raise ArtifactRubbingError("rubbing integer integral would overflow")
    sum_integral = _integral_image(ticks)
    count_integral = _integral_image(covered.astype(np.int64))
    x_indices = np.arange(width, dtype=np.int64)
    x0 = np.maximum(0, x_indices - reference_radius_pixels)
    x1 = np.minimum(width, x_indices + reference_radius_pixels + 1)
    output = np.empty((height, width, 2), dtype=np.uint8)
    output[:, :, 0] = 255
    output[:, :, 1] = 0
    ink_sum = 0
    inked_count = 0
    for row_start in range(0, height, RASTER_ROW_BLOCK_SIZE):
        row_stop = min(height, row_start + RASTER_ROW_BLOCK_SIZE)
        rows = np.arange(row_start, row_stop, dtype=np.int64)
        y0 = np.maximum(0, rows - reference_radius_pixels)
        y1 = np.minimum(height, rows + reference_radius_pixels + 1)
        window_sum = (
            sum_integral[y1[:, None], x1[None, :]]
            - sum_integral[y0[:, None], x1[None, :]]
            - sum_integral[y1[:, None], x0[None, :]]
            + sum_integral[y0[:, None], x0[None, :]]
        )
        window_count = (
            count_integral[y1[:, None], x1[None, :]]
            - count_integral[y0[:, None], x1[None, :]]
            - count_integral[y1[:, None], x0[None, :]]
            + count_integral[y0[:, None], x0[None, :]]
        )
        tick_block = ticks[row_start:row_stop]
        mask_block = covered[row_start:row_stop]
        signed_response = tick_block * window_count - window_sum
        if relief_polarity == "raised":
            response = np.maximum(signed_response, 0)
        elif relief_polarity == "incised":
            response = np.maximum(-signed_response, 0)
        else:
            response = np.abs(signed_response)
        valid_reference = window_count >= minimum_reference_sample_count
        denominator = effective_black_point_ticks * window_count
        usable = mask_block & valid_reference & (denominator > 0)
        response = np.where(usable, np.minimum(response, denominator), 0)
        if int(np.max(denominator, initial=0)) > (2**63 - 1) // 255:
            raise ArtifactRubbingError("rubbing tone mapping would overflow")
        drop = np.zeros(response.shape, dtype=np.int64)
        drop[usable] = (
            response[usable] * 255 + denominator[usable] // 2
        ) // denominator[usable]
        gray = np.asarray(255 - drop, dtype=np.uint8)
        output[row_start:row_stop, :, 0] = gray
        output[row_start:row_stop, :, 1] = np.where(mask_block, 255, 0).astype(
            np.uint8
        )
        ink_sum += int(np.sum(drop[mask_block], dtype=np.int64))
        inked_count += int(np.count_nonzero(drop[mask_block] > 0))
    # Raster work used v-increasing rows. PNG/raster row zero is the top.
    top_down = np.ascontiguousarray(np.flipud(output))
    return top_down, {
        "depth_span_quantized_ticks": maximum_tick,
        "depth_span_unquantized_um_rounded": int(round(span_mm * 1000.0)),
        "ink_sum": ink_sum,
        "inked_pixel_count": inked_count,
    }


def extract_digital_rubbing(
    vertices_world_mm: object,
    faces: object,
    recipe: Mapping[str, Any],
) -> tuple[DigitalRubbingRaster, dict[str, Any]]:
    """Render one deterministic front-surface rubbing from canonical-mm triangles."""

    validated = validate_rubbing_recipe(recipe)
    vertices, face_array = _validated_mesh_arrays(vertices_world_mm, faces)
    if vertices.shape[0] > MAX_RUBBING_VERTICES:
        raise ArtifactRubbingError("rubbing exceeds the vertex safety limit")
    if face_array.shape[0] > MAX_RUBBING_FACES:
        raise ArtifactRubbingError("rubbing exceeds the face safety limit")
    view = _view(validated["view"])
    frame = outline_frame(view)
    pixel_policy = validated["pixel_policy"]
    depth_policy = validated["depth_policy"]
    relief_policy = validated["relief_policy"]
    assert isinstance(pixel_policy, Mapping)
    assert isinstance(depth_policy, Mapping)
    assert isinstance(relief_policy, Mapping)
    pixels_per_mm = int(pixel_policy["pixels_per_mm"])
    depth, minimum_u, minimum_v, raster_qc = _rasterize_front_depth(
        vertices,
        face_array,
        frame,
        pixels_per_mm=pixels_per_mm,
        margin_pixels=int(pixel_policy["margin_pixels"]),
        layer_separation_mm=float(depth_policy["quantization_um"]) / 1000.0,
    )
    pixels, relief_qc = _render_local_relief(
        depth,
        depth_quantization_um=int(depth_policy["quantization_um"]),
        reference_radius_pixels=int(relief_policy["reference_radius_pixels"]),
        effective_black_point_ticks=int(relief_policy["effective_black_point_ticks"]),
        relief_polarity=str(relief_policy["polarity"]),
        minimum_reference_sample_count=int(
            relief_policy["minimum_reference_sample_count"]
        ),
    )
    raster = DigitalRubbingRaster(
        pixels=pixels,
        frame=frame,
        view=view,
        pixels_per_meter=pixels_per_mm * 1000,
        minimum_u_pixel_index=minimum_u,
        minimum_v_pixel_index=minimum_v,
    )
    qc = {
        "all_projected_faces_included": True,
        "input_face_count": int(face_array.shape[0]),
        "input_vertex_count": int(vertices.shape[0]),
        "sampling_applied": False,
        "view": view.value,
        **raster_qc,
        **relief_qc,
        **raster.qc_summary(),
    }
    return raster, qc


def estimate_digital_rubbing_resources(
    vertices_world_mm: object,
    faces: object,
    recipe: Mapping[str, Any],
    *,
    source_to_world_mm_matrix4x4: (
        np.ndarray
        | list[list[float]]
        | tuple[tuple[float, ...], ...]
        | None
    ) = None,
    uv_coords: np.ndarray | None = None,
    texture: np.ndarray | None = None,
) -> DigitalRubbingResourceEstimate:
    """Estimate raster dimensions and peak memory before starting heavy work.

    The byte estimate intentionally over-approximates the current pair of
    float64 depth buffers, integer relief intermediates, integral images,
    masks, output raster, row-block temporaries, and geometry copies.  It is an
    admission-control estimate rather than a promise about allocator RSS.
    """

    validated = validate_rubbing_recipe(recipe)
    vertices, face_array = _validated_mesh_arrays(vertices_world_mm, faces)
    if vertices.shape[0] > MAX_RUBBING_VERTICES:
        raise ArtifactRubbingError("rubbing exceeds the vertex safety limit")
    if face_array.shape[0] > MAX_RUBBING_FACES:
        raise ArtifactRubbingError("rubbing exceeds the face safety limit")

    frame = outline_frame(_view(validated["view"]))
    pixel_policy = validated["pixel_policy"]
    assert isinstance(pixel_policy, Mapping)
    pixels_per_mm = int(pixel_policy["pixels_per_mm"])
    margin_pixels = int(pixel_policy["margin_pixels"])

    referenced = np.unique(face_array.reshape(-1))
    origin = np.asarray(frame.origin_world_mm, dtype=np.float64)
    u_axis = np.asarray(frame.u_axis_world, dtype=np.float64)
    v_axis = np.asarray(frame.v_axis_world, dtype=np.float64)
    # Preflight must not allocate projection arrays for millions of unused
    # vertices. Execution still accounts for all geometry in the conservative
    # byte estimate, while physical artboard bounds need referenced vertices.
    referenced_vertices = vertices[referenced]
    if source_to_world_mm_matrix4x4 is not None:
        try:
            projection_matrix = require_affine_matrix4x4(
                source_to_world_mm_matrix4x4,
                field_name="source_to_world_mm_matrix4x4",
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactRubbingError(str(exc)) from exc
        referenced_vertices = (
            referenced_vertices @ projection_matrix[:3, :3].T
            + projection_matrix[:3, 3]
        )
    relative = referenced_vertices - origin
    scaled = np.column_stack((relative @ u_axis, relative @ v_axis)) * float(
        pixels_per_mm
    )
    if not np.isfinite(scaled).all():
        raise ArtifactRubbingError("rubbing projection contains non-finite coordinates")
    if float(np.max(np.abs(scaled))) > MAX_RUBBING_GRID_INDEX:
        raise ArtifactRubbingError("rubbing projection exceeds the pixel-grid safety range")

    content_min_u = math.floor(float(np.min(scaled[:, 0])))
    content_max_u = math.ceil(float(np.max(scaled[:, 0])))
    content_min_v = math.floor(float(np.min(scaled[:, 1])))
    content_max_v = math.ceil(float(np.max(scaled[:, 1])))
    width = content_max_u - content_min_u + 2 * margin_pixels
    height = content_max_v - content_min_v + 2 * margin_pixels
    if width <= 0 or height <= 0:
        raise ArtifactRubbingError("rubbing artboard has zero physical extent")
    if width > MAX_RUBBING_DIMENSION or height > MAX_RUBBING_DIMENSION:
        raise ArtifactRubbingError("rubbing artboard exceeds the dimension safety limit")
    pixel_count = width * height
    if pixel_count > MAX_RUBBING_PIXELS:
        raise ArtifactRubbingError(
            "rubbing artboard exceeds the pixel safety limit; choose a lower physical resolution"
        )

    geometry_bytes = int(vertices.nbytes + face_array.nbytes)
    materialized_attribute_bytes = rubbing_materialized_attribute_bytes(
        uv_coords=uv_coords,
        texture=texture,
    )
    estimated_peak_bytes = (
        pixel_count * RUBBING_ESTIMATED_PEAK_BYTES_PER_PIXEL
        + geometry_bytes * RUBBING_ESTIMATE_GEOMETRY_MULTIPLIER
        + materialized_attribute_bytes
        * RUBBING_ESTIMATE_MATERIALIZED_ATTRIBUTE_MULTIPLIER
        + RUBBING_ESTIMATE_FIXED_OVERHEAD_BYTES
    )
    return DigitalRubbingResourceEstimate(
        width_pixels=width,
        height_pixels=height,
        pixel_count=pixel_count,
        vertex_count=int(vertices.shape[0]),
        face_count=int(face_array.shape[0]),
        estimated_peak_bytes=estimated_peak_bytes,
    )


def compute_artifact_rubbing(
    session: ArtifactSession,
    view: OutlineView | str,
    *,
    pixels_per_mm: int,
    margin_um: int,
    reference_radius_um: int,
    depth_quantization_um: int,
    black_point_um: int,
    ink_strength_percent: int,
    relief_polarity: str,
) -> ArtifactRubbingComputation:
    if not isinstance(session, ArtifactSession):
        raise ArtifactRubbingError("session must be an ArtifactSession")
    recipe = rubbing_recipe(
        view,
        pixels_per_mm=pixels_per_mm,
        margin_um=margin_um,
        reference_radius_um=reference_radius_um,
        depth_quantization_um=depth_quantization_um,
        black_point_um=black_point_um,
        ink_strength_percent=ink_strength_percent,
        relief_polarity=relief_polarity,
    )
    try:
        context = session.capture_operation(recipe=recipe)
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactRubbingError(str(exc)) from exc
    raster, qc = extract_digital_rubbing(
        projection.mesh.vertices,
        projection.mesh.faces,
        recipe,
    )
    return ArtifactRubbingComputation(
        context=context,
        projection_snapshot=projection.snapshot,
        raster=raster,
        recipe=recipe,
        qc=qc,
    )


def compute_artifact_rubbing_from_recipe(
    session: ArtifactSession,
    recipe: Mapping[str, Any],
) -> ArtifactRubbingComputation:
    validated = validate_rubbing_recipe(recipe)
    pixel = validated["pixel_policy"]
    depth = validated["depth_policy"]
    relief = validated["relief_policy"]
    assert isinstance(pixel, Mapping)
    assert isinstance(depth, Mapping)
    assert isinstance(relief, Mapping)
    return compute_artifact_rubbing(
        session,
        _view(validated["view"]),
        pixels_per_mm=int(pixel["pixels_per_mm"]),
        margin_um=int(pixel["margin_requested_um"]),
        reference_radius_um=int(relief["reference_radius_requested_um"]),
        depth_quantization_um=int(depth["quantization_um"]),
        black_point_um=int(relief["black_point_requested_um"]),
        ink_strength_percent=int(relief["ink_strength_percent"]),
        relief_polarity=str(relief["polarity"]),
    )


def rubbing_computation_matches_active_projection(
    session: ArtifactSession,
    computation: ArtifactRubbingComputation,
) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(
        computation, ArtifactRubbingComputation
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


def require_current_rubbing_computation(
    session: ArtifactSession,
    computation: ArtifactRubbingComputation,
) -> None:
    if not rubbing_computation_matches_active_projection(session, computation):
        raise ArtifactRubbingError(
            "Digital Rubbing computation is stale for the active scene projection"
        )


def commit_artifact_rubbing(
    session: ArtifactSession,
    computation: ArtifactRubbingComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    """Append the raster receipt at its captured context, even if historical."""

    if not isinstance(session, ArtifactSession):
        raise ArtifactRubbingError("session must be an ArtifactSession")
    if not isinstance(computation, ArtifactRubbingComputation):
        raise ArtifactRubbingError(
            "computation must be an ArtifactRubbingComputation"
        )
    from .artifact_rubbing_record import (  # noqa: PLC0415
        ArtifactRubbingRecordError,
        append_rubbing_record_from_context,
    )

    try:
        document = append_rubbing_record_from_context(
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
    except (ArtifactRubbingRecordError, ArtifactSessionError) as exc:
        raise ArtifactRubbingError(str(exc)) from exc


__all__ = [
    "ArtifactRubbingComputation",
    "ArtifactRubbingError",
    "DEFAULT_RUBBING_BLACK_POINT_UM",
    "DEFAULT_RUBBING_DEPTH_QUANTIZATION_UM",
    "DEFAULT_RUBBING_INK_STRENGTH_PERCENT",
    "DEFAULT_RUBBING_MARGIN_UM",
    "DEFAULT_RUBBING_PIXELS_PER_MM",
    "DEFAULT_RUBBING_POLARITY",
    "DEFAULT_RUBBING_REFERENCE_RADIUS_UM",
    "DigitalRubbingRaster",
    "DigitalRubbingResourceEstimate",
    "RUBBING_ALGORITHM",
    "RUBBING_ALGORITHM_VERSION",
    "RUBBING_COORDINATE_SPACE",
    "RUBBING_PIXEL_FORMAT",
    "RUBBING_RASTER_HASH_SCOPE",
    "RUBBING_RASTER_SCHEMA_VERSION",
    "compute_artifact_rubbing",
    "compute_artifact_rubbing_from_recipe",
    "extract_digital_rubbing",
    "estimate_digital_rubbing_resources",
    "commit_artifact_rubbing",
    "require_current_rubbing_computation",
    "rubbing_computation_matches_active_projection",
    "rubbing_recipe",
    "RUBBING_ESTIMATED_PEAK_BYTES_PER_PIXEL",
    "RUBBING_ESTIMATE_FIXED_OVERHEAD_BYTES",
    "RUBBING_ESTIMATE_GEOMETRY_MULTIPLIER",
    "RUBBING_ESTIMATE_MATERIALIZED_ATTRIBUTE_MULTIPLIER",
    "rubbing_materialized_attribute_bytes",
    "validate_rubbing_recipe",
]
