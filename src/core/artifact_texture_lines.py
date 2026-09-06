"""문양 내선: the incised pattern of a wall, read from its normal map and
drawn on the elevation as inner lines.

The rubbing pasted on the axis shows the comb pattern as ink; the
guidelines also draw it, on the elevation, as lines ([K1] 2013 p. 37, the
pattern of the left half-elevation).  Where a scan keeps the incisions in
its normal map and not in its mesh (docs/REAL_DATA_TRIAL.md), this module
reads them from the map: the wall as one orthographic view is a
"development" whose (u, v) are the view's own coordinates, the map's tilt
integrates to a height over that view exactly as it does over an unrolled
strip, and every incision is a valley of that height.  The valley floors,
traced to sub-pixel lines, simplified and smoothed by a stated width, are
the record - `measurement.texture_lines.v1` - in the view's frame, in whole
micrometres, drawn on the figure whose plane is that view's.

Two things a reader should know are stated by the recipe, not decided at
print time: the least curvature a valley needs to count as an incision,
and the width the traced lines were smoothed with.  The record is computed
once under the active alignment and read back verified, like a crease
reading; it is never recomputed while a sheet is drawn.
"""

from __future__ import annotations

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
from .artifact_texture_relief import (
    ArtifactTextureReliefError,
    NormalMap,
    TextureAtlas,
    require_texture_relief_sources,
    rigid_rotation_between,
    texture_relief_block,
    texture_relief_depth_field,
    validate_texture_relief_block,
)
from .canonical_json import (
    CanonicalJSONError,
    canonical_json_bytes,
    canonical_json_sha256,
)
from .drawing_smoothing import smooth_polyline


TEXTURE_LINES_RECORD_TYPE = "measurement.texture_lines.v1"
TEXTURE_LINES_OPERATION_KIND = "texture_lines"
TEXTURE_LINES_ALGORITHM = "archmeshrubbing.texture_normal_map_valleys"
TEXTURE_LINES_ALGORITHM_VERSION = "1.0.0"
TEXTURE_LINES_COORDINATE_SPACE = "canonical_um_planar_view/v1"
TEXTURE_LINES_PAYLOAD_SCHEMA_VERSION = "1.0.0"
TEXTURE_LINES_PAYLOAD_EXTENSION_KEY = "org.archmeshrubbing:texture-lines-v1"
TEXTURE_LINES_PAYLOAD_MEDIA_TYPE = "application/vnd.archmeshrubbing.texture-lines+json"
TEXTURE_LINES_GEOMETRY_REF_PREFIX = "urn:archmeshrubbing:texture-lines:sha256:"
TEXTURE_LINES_VIEWS: tuple[str, ...] = tuple(sorted(view.value for view in OutlineView))
#: How the wall is put on the view's raster: faces facing the viewer at
#: least ``facing_cos`` are painted far to near, so the nearest wall wins
#: every pixel without a depth buffer.
TEXTURE_LINES_PAINTER = "far_to_near_by_centroid/v1"
#: How a valley is found: the Hessian of the smoothed height, its largest
#: eigenvalue the curvature across the valley, the floor where the slope
#: along that direction crosses zero within half a pixel (Steger's line).
TEXTURE_LINES_VALLEY_RULE = "hessian_largest_eigenvalue_zero_crossing/v1"
#: Where the height is integrated and the valleys traced.  ``view`` works on
#: the orthographic view itself, which foreshortens the wall towards the
#: silhouette until a millimetre-wide incision is a pixel wide; ``axis_development``
#: unrolls every face that faces the view about the measured rotation axis -
#: u the arc r x (theta - theta_view), v the axial height - traces there,
#: where the pattern is seen the way the rubbing's paper sees it, and carries
#: each line back through its triangle onto the view.  That is how the
#: drawing is made by hand: the pattern is traced from the rubbing, not
#: from the elevation, and the two agree because they are the same reading.
TEXTURE_LINES_DOMAIN_VIEW = "view"
TEXTURE_LINES_DOMAIN_AXIS = "axis_development"
TEXTURE_LINES_DOMAINS: tuple[str, ...] = (TEXTURE_LINES_DOMAIN_VIEW, TEXTURE_LINES_DOMAIN_AXIS)

DEFAULT_TEXTURE_LINES_PIXELS_PER_MM = 5
#: The base the map's tilt is measured against is the sampled normal
#: smoothed this wide.  Wider than the rubbing's default: an incision a
#: millimetre across loses most of its own tilt to a one-millimetre base,
#: and the height it integrates to comes out a quarter of what was cut.
DEFAULT_TEXTURE_LINES_SMOOTHING_UM = 2_000
DEFAULT_TEXTURE_LINES_FACING_COS = 0.35
DEFAULT_TEXTURE_LINES_SCALE_MM = 0.3
DEFAULT_TEXTURE_LINES_CURVATURE_MIN_PER_MM = 0.3
#: A valley pixel counts only in a run that somewhere reaches this
#: curvature: the lower threshold follows a stroke to its faint ends, the
#: higher one keeps grain from being a stroke (hysteresis, as Canny's).
DEFAULT_TEXTURE_LINES_CURVATURE_SEED_PER_MM = 0.6
#: Chain ends within this distance that point at each other are joined:
#: a stroke the tracer dropped for a pixel or two is one stroke.
DEFAULT_TEXTURE_LINES_LINK_MM = 1.0
DEFAULT_TEXTURE_LINES_LINK_ANGLE_DEG = 35.0
DEFAULT_TEXTURE_LINES_MIN_LENGTH_MM = 1.5
DEFAULT_TEXTURE_LINES_LINE_SMOOTHING_MM = 0.3
#: Which way an incision goes in the integrated height: -1 when the map's
#: normals point out of the wall, so a cut is a valley; +1 when they point
#: in, so the same cut comes out a ridge.  The file does not say which; the
#: QC counts both so the drafter chooses with numbers, as for the encoding.
DEFAULT_TEXTURE_LINES_INCISION_SIGN = -1

MIN_TEXTURE_LINES_PIXELS_PER_MM = 1
MAX_TEXTURE_LINES_PIXELS_PER_MM = 50
MAX_TEXTURE_LINES = 20_000
MAX_TEXTURE_LINE_POINTS = 500_000
#: Pixels of coverage border left out of the valley search: the height is
#: not trusted where the smoothing kernel ran off the wall.
_BORDER_SIGMAS = 2.0


class ArtifactTextureLinesError(ValueError):
    """A pattern-line reading cannot be computed, recorded or read back safely."""


def _strict_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactTextureLinesError(f"{name} must be an integer")
    number = int(value)
    if number < minimum or number > maximum:
        raise ArtifactTextureLinesError(
            f"{name} must be in the inclusive range {minimum}..{maximum}"
        )
    return number


def _exact_keys(value: object, keys: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactTextureLinesError(f"{name} must be an object")
    if set(value) != set(keys):
        raise ArtifactTextureLinesError(
            f"{name} must carry exactly {', '.join(sorted(keys))}"
        )
    return value


def _view_name(view: object) -> str:
    try:
        return OutlineView(view).value
    except (TypeError, ValueError) as exc:
        raise ArtifactTextureLinesError(
            f"view must be one of {', '.join(TEXTURE_LINES_VIEWS)}"
        ) from exc


def _domain(value: object) -> str:
    if value not in TEXTURE_LINES_DOMAINS:
        raise ArtifactTextureLinesError(
            f"domain must be one of {', '.join(TEXTURE_LINES_DOMAINS)}"
        )
    return str(value)


def _incision_sign(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) not in (-1, 1):
        raise ArtifactTextureLinesError("incision_sign must be -1 or 1")
    return int(value)


def _um(value: float, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ArtifactTextureLinesError(f"{name} must be a number")
    if not math.isfinite(float(value)):
        raise ArtifactTextureLinesError(f"{name} must be finite")
    return _strict_int(int(round(float(value) * 1000.0)), name=name, minimum=minimum, maximum=maximum)


def texture_lines_recipe(
    atlas: TextureAtlas,
    normal_map: NormalMap,
    *,
    view: OutlineView | str,
    source_vertex_count: int,
    source_face_count: int,
    pixels_per_mm: int = DEFAULT_TEXTURE_LINES_PIXELS_PER_MM,
    smoothing_um: int = DEFAULT_TEXTURE_LINES_SMOOTHING_UM,
    facing_cos: float = DEFAULT_TEXTURE_LINES_FACING_COS,
    scale_mm: float = DEFAULT_TEXTURE_LINES_SCALE_MM,
    curvature_min_per_mm: float = DEFAULT_TEXTURE_LINES_CURVATURE_MIN_PER_MM,
    min_length_mm: float = DEFAULT_TEXTURE_LINES_MIN_LENGTH_MM,
    line_smoothing_mm: float = DEFAULT_TEXTURE_LINES_LINE_SMOOTHING_MM,
    incision_sign: int = DEFAULT_TEXTURE_LINES_INCISION_SIGN,
    domain: str = TEXTURE_LINES_DOMAIN_VIEW,
    curvature_seed_per_mm: float | None = None,
    link_mm: float = DEFAULT_TEXTURE_LINES_LINK_MM,
) -> dict[str, Any]:
    """The recipe: the two files, the view, and every number that decides a line.

    ``curvature_seed_per_mm`` defaults to twice ``curvature_min_per_mm``.
    """

    try:
        relief = texture_relief_block(atlas, normal_map, smoothing_um=smoothing_um)
    except ArtifactTextureReliefError as exc:
        raise ArtifactTextureLinesError(str(exc)) from exc
    if isinstance(facing_cos, bool) or not isinstance(facing_cos, (int, float)) or not math.isfinite(facing_cos):
        raise ArtifactTextureLinesError("facing_cos must be a finite number")
    facing = int(round(float(facing_cos) * 1_000_000))
    return {
        "algorithm": TEXTURE_LINES_ALGORITHM,
        "algorithm_version": TEXTURE_LINES_ALGORITHM_VERSION,
        "coordinate_space": TEXTURE_LINES_COORDINATE_SPACE,
        "detection_policy": {
            "curvature_min_per_m": _um(
                curvature_min_per_mm, name="curvature_min_per_mm", minimum=1, maximum=1_000_000
            ),
            "curvature_seed_per_m": _um(
                2.0 * float(curvature_min_per_mm) if curvature_seed_per_mm is None else curvature_seed_per_mm,
                name="curvature_seed_per_mm",
                minimum=1,
                maximum=1_000_000,
            ),
            "incision_sign": _incision_sign(incision_sign),
            "line_smoothing_um": _um(
                line_smoothing_mm, name="line_smoothing_mm", minimum=0, maximum=3_000
            ),
            "link_um": _um(link_mm, name="link_mm", minimum=0, maximum=10_000),
            "min_length_um": _um(min_length_mm, name="min_length_mm", minimum=0, maximum=1_000_000),
            "scale_um": _um(scale_mm, name="scale_mm", minimum=50, maximum=5_000),
            "valley": TEXTURE_LINES_VALLEY_RULE,
        },
        "raster_policy": {
            "facing_cos_millionths": _strict_int(
                facing, name="facing_cos", minimum=50_000, maximum=1_000_000
            ),
            "painter": TEXTURE_LINES_PAINTER,
            "pixels_per_mm": _strict_int(
                pixels_per_mm,
                name="pixels_per_mm",
                minimum=MIN_TEXTURE_LINES_PIXELS_PER_MM,
                maximum=MAX_TEXTURE_LINES_PIXELS_PER_MM,
            ),
        },
        "domain": _domain(domain),
        "source_face_count": _strict_int(
            source_face_count, name="source_face_count", minimum=1, maximum=10**9
        ),
        "source_vertex_count": _strict_int(
            source_vertex_count, name="source_vertex_count", minimum=3, maximum=10**9
        ),
        "texture_relief": relief,
        "view": _view_name(view),
    }


_RECIPE_KEYS = frozenset(
    {
        "algorithm",
        "algorithm_version",
        "coordinate_space",
        "detection_policy",
        "domain",
        "raster_policy",
        "source_face_count",
        "source_vertex_count",
        "texture_relief",
        "view",
    }
)


def validate_texture_lines_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the recipe from its own numbers and require the same bytes."""

    block = _exact_keys(recipe, _RECIPE_KEYS, name="texture lines recipe")
    if block["algorithm"] != TEXTURE_LINES_ALGORITHM:
        raise ArtifactTextureLinesError("texture lines recipe names another algorithm")
    if block["algorithm_version"] != TEXTURE_LINES_ALGORITHM_VERSION:
        raise ArtifactTextureLinesError("texture lines recipe names another algorithm version")
    if block["coordinate_space"] != TEXTURE_LINES_COORDINATE_SPACE:
        raise ArtifactTextureLinesError("texture lines recipe names another coordinate space")
    detection = _exact_keys(
        block["detection_policy"],
        frozenset(
            {
                "curvature_min_per_m",
                "curvature_seed_per_m",
                "incision_sign",
                "line_smoothing_um",
                "link_um",
                "min_length_um",
                "scale_um",
                "valley",
            }
        ),
        name="detection_policy",
    )
    if detection["valley"] != TEXTURE_LINES_VALLEY_RULE:
        raise ArtifactTextureLinesError("texture lines recipe names another valley rule")
    raster = _exact_keys(
        block["raster_policy"],
        frozenset({"facing_cos_millionths", "painter", "pixels_per_mm"}),
        name="raster_policy",
    )
    if raster["painter"] != TEXTURE_LINES_PAINTER:
        raise ArtifactTextureLinesError("texture lines recipe names another painter")
    try:
        relief = validate_texture_relief_block(block["texture_relief"])
    except ArtifactTextureReliefError as exc:
        raise ArtifactTextureLinesError(str(exc)) from exc
    rebuilt = {
        "algorithm": TEXTURE_LINES_ALGORITHM,
        "algorithm_version": TEXTURE_LINES_ALGORITHM_VERSION,
        "coordinate_space": TEXTURE_LINES_COORDINATE_SPACE,
        "detection_policy": {
            "curvature_min_per_m": _strict_int(
                detection["curvature_min_per_m"], name="curvature_min_per_m", minimum=1, maximum=1_000_000
            ),
            "curvature_seed_per_m": _strict_int(
                detection["curvature_seed_per_m"], name="curvature_seed_per_m", minimum=1, maximum=1_000_000
            ),
            "incision_sign": _incision_sign(detection["incision_sign"]),
            "line_smoothing_um": _strict_int(
                detection["line_smoothing_um"], name="line_smoothing_um", minimum=0, maximum=3_000
            ),
            "link_um": _strict_int(detection["link_um"], name="link_um", minimum=0, maximum=10_000),
            "min_length_um": _strict_int(
                detection["min_length_um"], name="min_length_um", minimum=0, maximum=1_000_000
            ),
            "scale_um": _strict_int(detection["scale_um"], name="scale_um", minimum=50, maximum=5_000),
            "valley": TEXTURE_LINES_VALLEY_RULE,
        },
        "raster_policy": {
            "facing_cos_millionths": _strict_int(
                raster["facing_cos_millionths"],
                name="facing_cos_millionths",
                minimum=50_000,
                maximum=1_000_000,
            ),
            "painter": TEXTURE_LINES_PAINTER,
            "pixels_per_mm": _strict_int(
                raster["pixels_per_mm"],
                name="pixels_per_mm",
                minimum=MIN_TEXTURE_LINES_PIXELS_PER_MM,
                maximum=MAX_TEXTURE_LINES_PIXELS_PER_MM,
            ),
        },
        "domain": _domain(block["domain"]),
        "source_face_count": _strict_int(
            block["source_face_count"], name="source_face_count", minimum=1, maximum=10**9
        ),
        "source_vertex_count": _strict_int(
            block["source_vertex_count"], name="source_vertex_count", minimum=3, maximum=10**9
        ),
        "texture_relief": relief,
        "view": _view_name(block["view"]),
    }
    if rebuilt["detection_policy"]["curvature_seed_per_m"] < rebuilt["detection_policy"]["curvature_min_per_m"]:
        raise ArtifactTextureLinesError("curvature_seed_per_m must not be below curvature_min_per_m")
    try:
        if canonical_json_bytes(rebuilt) != canonical_json_bytes(dict(block)):
            raise ArtifactTextureLinesError("texture lines recipe is not in canonical form")
    except CanonicalJSONError as exc:
        raise ArtifactTextureLinesError(str(exc)) from exc
    return rebuilt


Polyline = tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class TextureLinesPayload:
    """Every incision one reading traced, in one view's frame, in µm."""

    schema_version: str
    view: str
    polylines: tuple[Polyline, ...]

    def __post_init__(self) -> None:
        if self.schema_version != TEXTURE_LINES_PAYLOAD_SCHEMA_VERSION:
            raise ArtifactTextureLinesError(
                f"unsupported texture lines payload schema: {self.schema_version!r}"
            )
        object.__setattr__(self, "view", _view_name(self.view))
        cleaned: list[Polyline] = []
        limit = 10**9
        for polyline in self.polylines:
            points = tuple(
                (
                    _strict_int(x, name="texture line point", minimum=-limit, maximum=limit),
                    _strict_int(y, name="texture line point", minimum=-limit, maximum=limit),
                )
                for x, y in polyline
            )
            if len(points) < 2:
                raise ArtifactTextureLinesError("a texture line has at least two points")
            if any(a == b for a, b in zip(points, points[1:])):
                raise ArtifactTextureLinesError("a texture line repeats a point")
            cleaned.append(points)
        if not cleaned:
            raise ArtifactTextureLinesError("a texture lines payload holds at least one line")
        if len(cleaned) > MAX_TEXTURE_LINES:
            raise ArtifactTextureLinesError(
                f"a texture lines payload holds at most {MAX_TEXTURE_LINES} lines"
            )
        if sum(len(polyline) for polyline in cleaned) > MAX_TEXTURE_LINE_POINTS:
            raise ArtifactTextureLinesError(
                f"a texture lines payload holds at most {MAX_TEXTURE_LINE_POINTS} points"
            )
        object.__setattr__(self, "polylines", tuple(cleaned))

    @property
    def line_count(self) -> int:
        return len(self.polylines)

    @property
    def point_count(self) -> int:
        return sum(len(polyline) for polyline in self.polylines)

    @property
    def total_length_um(self) -> int:
        total = 0.0
        for polyline in self.polylines:
            points = np.asarray(polyline, dtype=np.float64)
            total += float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        return int(round(total))

    def to_dict(self) -> dict[str, Any]:
        return {
            "polylines": [[list(point) for point in polyline] for polyline in self.polylines],
            "schema_version": self.schema_version,
            "view": self.view,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TextureLinesPayload":
        block = _exact_keys(
            data, frozenset({"polylines", "schema_version", "view"}), name="texture lines payload"
        )
        raw = block["polylines"]
        if not isinstance(raw, (list, tuple)):
            raise ArtifactTextureLinesError("texture lines payload polylines must be an array")
        polylines: list[Polyline] = []
        for polyline in raw:
            if not isinstance(polyline, (list, tuple)):
                raise ArtifactTextureLinesError("a texture line must be an array")
            points: list[tuple[int, int]] = []
            for point in polyline:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ArtifactTextureLinesError("a texture line point must be [x, y]")
                points.append((point[0], point[1]))  # type: ignore[arg-type]
            polylines.append(tuple(points))
        schema_version = block["schema_version"]
        view = block["view"]
        return cls(
            schema_version=schema_version if isinstance(schema_version, str) else "",
            view=view if isinstance(view, str) else "",
            polylines=tuple(polylines),
        )

    def canonical_json_bytes(self) -> bytes:
        try:
            return canonical_json_bytes(self.to_dict())
        except CanonicalJSONError as exc:
            raise ArtifactTextureLinesError(str(exc)) from exc

    @property
    def sha256(self) -> str:
        try:
            return canonical_json_sha256(self.to_dict())
        except CanonicalJSONError as exc:
            raise ArtifactTextureLinesError(str(exc)) from exc

    @property
    def geometry_ref(self) -> str:
        return f"{TEXTURE_LINES_GEOMETRY_REF_PREFIX}{self.sha256}"

    def qc_summary(self) -> dict[str, Any]:
        return {
            "line_count": self.line_count,
            "point_count": self.point_count,
            "total_length_um": self.total_length_um,
            "view": self.view,
        }


def _valley_points(
    height_mm: np.ndarray,
    good: np.ndarray,
    *,
    pixels_per_mm: int,
    scale_mm: float,
    curvature_min_per_mm: float,
    curvature_seed_per_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where the smoothed height has a valley floor: a mask of pixels, and
    at each the sub-pixel offset (du, dv) to the floor.

    With ``curvature_seed_per_mm`` a floor pixel is kept only in a connected
    run that somewhere reaches the seed curvature: the lower threshold
    follows a stroke to its faint ends, the higher keeps the grain out.
    """

    from scipy.ndimage import binary_erosion, gaussian_filter, label  # noqa: PLC0415

    sigma = scale_mm * float(pixels_per_mm)
    field = np.where(good, height_mm, 0.0)
    # Derivatives of the Gaussian-smoothed height, in mm per pixel^n.
    hx = gaussian_filter(field, sigma, order=(0, 1))
    hy = gaussian_filter(field, sigma, order=(1, 0))
    hxx = gaussian_filter(field, sigma, order=(0, 2))
    hyy = gaussian_filter(field, sigma, order=(2, 0))
    hxy = gaussian_filter(field, sigma, order=(1, 1))
    half_trace = 0.5 * (hxx + hyy)
    half_gap = 0.5 * (hxx - hyy)
    root = np.sqrt(half_gap**2 + hxy**2)
    largest = half_trace + root
    # Eigenvector of the largest eigenvalue: across the valley.
    nx = hxy
    ny = largest - hxx
    swap = np.abs(nx) + np.abs(ny) < 1e-12
    nx = np.where(swap, largest - hyy, nx)
    ny = np.where(swap, hxy, ny)
    norm = np.hypot(nx, ny)
    safe = norm > 1e-15
    nx = np.where(safe, nx / np.maximum(norm, 1e-15), 1.0)
    ny = np.where(safe, ny / np.maximum(norm, 1e-15), 0.0)
    curvature_per_mm = largest * float(pixels_per_mm) ** 2
    slope_across = hx * nx + hy * ny
    concave = largest > 0.0
    offset = np.where(concave, -slope_across / np.where(concave, largest, 1.0), 2.0)
    inner = binary_erosion(good, iterations=max(1, int(math.ceil(_BORDER_SIGMAS * sigma))))
    # One pixel across the valley: keep a pixel only where the curvature is
    # no less than at its two neighbours along the across direction, the
    # direction rounded to the nearest of the four pixel axes.
    angle = np.arctan2(ny, nx) % math.pi
    sector = np.rint(angle / (math.pi / 4.0)).astype(np.int64) % 4
    steps = ((0, 1), (1, 1), (1, 0), (1, -1))  # (row, col) for 0, 45, 90, 135 degrees
    ahead = np.full_like(largest, -np.inf)
    behind = np.full_like(largest, -np.inf)
    padded = np.pad(largest, 1, mode="constant", constant_values=-np.inf)
    height_px, width_px = largest.shape
    for index, (dr, dc) in enumerate(steps):
        mask = sector == index
        forward = padded[1 + dr : 1 + dr + height_px, 1 + dc : 1 + dc + width_px]
        backward = padded[1 - dr : 1 - dr + height_px, 1 - dc : 1 - dc + width_px]
        ahead = np.where(mask, forward, ahead)
        behind = np.where(mask, backward, behind)
    crest = (largest >= ahead) & (largest >= behind)
    valley = (
        inner
        & concave
        & crest
        & (curvature_per_mm >= curvature_min_per_mm)
        & (np.abs(offset) <= 1.0)
    )
    if curvature_seed_per_mm is not None and curvature_seed_per_mm > curvature_min_per_mm:
        labels, count = label(valley, structure=np.ones((3, 3), dtype=bool))
        if count:
            seeded = np.zeros(count + 1, dtype=bool)
            seeded[np.unique(labels[valley & (curvature_per_mm >= curvature_seed_per_mm)])] = True
            seeded[0] = False
            valley = seeded[labels]
    offset = np.where(valley, np.clip(offset, -1.0, 1.0), 0.0)
    return valley, offset * nx, offset * ny


_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _trace_chains(valley: np.ndarray) -> list[list[tuple[int, int]]]:
    """Walk the valley pixels into chains: from every end inward, then round
    whatever closed on itself.  At a junction the walk keeps the straightest
    way on; the branches it leaves become chains of their own, so every
    pixel is in exactly one chain."""

    rows, cols = np.nonzero(valley)
    pixels = set(zip(rows.tolist(), cols.tolist()))
    degree: dict[tuple[int, int], int] = {}
    for pixel in pixels:
        r, c = pixel
        degree[pixel] = sum((r + dr, c + dc) in pixels for dr, dc in _NEIGHBOURS)
    visited: set[tuple[int, int]] = set()
    chains: list[list[tuple[int, int]]] = []

    def walk(start: tuple[int, int]) -> None:
        chain = [start]
        visited.add(start)
        current = start
        previous: tuple[int, int] | None = None
        while True:
            r, c = current
            candidates = [
                (r + dr, c + dc)
                for dr, dc in _NEIGHBOURS
                if (r + dr, c + dc) in pixels and (r + dr, c + dc) not in visited
            ]
            if not candidates:
                break
            if previous is not None and len(candidates) > 1:
                # Keep going the way we came: pick the neighbour nearest the
                # continuation of the last step.
                heading = (r - previous[0], c - previous[1])
                candidates.sort(
                    key=lambda p: -((p[0] - r) * heading[0] + (p[1] - c) * heading[1])
                    / (math.hypot(p[0] - r, p[1] - c) * max(math.hypot(*heading), 1e-9))
                )
            else:
                # Prefer 4-neighbours so a diagonal does not skip a pixel.
                candidates.sort(key=lambda p: abs(p[0] - r) + abs(p[1] - c))
            following = candidates[0]
            visited.add(following)
            chain.append(following)
            previous, current = current, following
        chains.append(chain)

    for pixel in sorted(pixels):
        if pixel not in visited and degree[pixel] <= 1:
            walk(pixel)
    for pixel in sorted(pixels):
        if pixel not in visited:
            walk(pixel)
    return chains


def _link_polylines(
    polylines: Sequence[np.ndarray], *, gap_mm: float, angle_deg: float, rounds: int = 4
) -> list[np.ndarray]:
    """Join polylines end to end where two ends lie within ``gap_mm`` and
    point at each other within ``angle_deg``; the pair that points most
    directly joins first, ties to the earlier lines.  Repeated until nothing
    joins, at most ``rounds`` times."""

    from scipy.spatial import cKDTree  # noqa: PLC0415

    lines = [np.asarray(line, dtype=np.float64) for line in polylines]
    if gap_mm <= 0.0 or len(lines) < 2:
        return lines
    cos_min = math.cos(math.radians(angle_deg))
    for _round in range(rounds):
        ends: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for index, line in enumerate(lines):
            if line.shape[0] < 2:
                continue
            head = line[0] - line[min(3, line.shape[0] - 1)]
            tail = line[-1] - line[max(-4, -line.shape[0])]
            ends.append((index, 0, line[0], head / max(float(np.linalg.norm(head)), 1e-12)))
            ends.append((index, 1, line[-1], tail / max(float(np.linalg.norm(tail)), 1e-12)))
        if len(ends) < 2:
            break
        tree = cKDTree(np.array([end[2] for end in ends]))
        scored: list[tuple[float, int, int]] = []
        for a, b in sorted(tree.query_pairs(r=gap_mm)):
            line_a, _end_a, point_a, direction_a = ends[a]
            line_b, _end_b, point_b, direction_b = ends[b]
            if line_a == line_b:
                continue
            gap = point_b - point_a
            distance = float(np.linalg.norm(gap))
            facing = float(direction_a @ -direction_b)
            if distance > 1e-9:
                unit = gap / distance
                if float(direction_a @ unit) < cos_min or float(direction_b @ -unit) < cos_min:
                    continue
            elif facing < cos_min:
                continue
            scored.append((distance - facing, a, b))
        scored.sort()
        taken: set[int] = set()
        merged: set[int] = set()
        joined = 0
        for _score, a, b in scored:
            if a in taken or b in taken:
                continue
            line_a, end_a, _pa, _da = ends[a]
            line_b, end_b, _pb, _db = ends[b]
            if line_a in merged or line_b in merged:
                continue
            first = lines[line_a] if end_a == 1 else lines[line_a][::-1]
            second = lines[line_b] if end_b == 0 else lines[line_b][::-1]
            lines[line_a] = np.vstack([first, second])
            lines[line_b] = np.zeros((0, 2), dtype=np.float64)
            taken.update((a, b))
            merged.update((line_a, line_b))
            joined += 1
        lines = [line for line in lines if line.shape[0] >= 2]
        if joined == 0:
            break
    return lines


class _DevelopedLocator:
    """Find, for a point of the developed plane, the developed triangle it
    lies in and the canonical position that triangle carries it to."""

    def __init__(
        self,
        uv_mm: np.ndarray,
        developed_faces: np.ndarray,
        vertices_mm: np.ndarray,
        face_facing: np.ndarray,
    ) -> None:
        from scipy.spatial import cKDTree  # noqa: PLC0415

        self.uv = np.asarray(uv_mm, dtype=np.float64)
        self.faces = np.asarray(developed_faces, dtype=np.int64)
        self.vertices = np.asarray(vertices_mm, dtype=np.float64)
        self.facing = np.asarray(face_facing, dtype=np.float64)
        self.tree = cKDTree(self.uv[self.faces].mean(axis=1))

    def locate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        count = min(12, self.faces.shape[0])
        _distances, nearest = self.tree.query(points, k=count)
        nearest = np.asarray(nearest, dtype=np.int64).reshape(points.shape[0], -1)
        world = np.zeros((points.shape[0], 3), dtype=np.float64)
        facing = np.zeros(points.shape[0], dtype=np.float64)
        for index, point in enumerate(points):
            chosen = int(nearest[index, 0])
            weights = np.array([1.0 / 3.0] * 3)
            for candidate in nearest[index]:
                a, b, c = self.uv[self.faces[candidate]]
                denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
                if denominator == 0.0:
                    continue
                w0 = ((b[1] - c[1]) * (point[0] - c[0]) + (c[0] - b[0]) * (point[1] - c[1])) / denominator
                w1 = ((c[1] - a[1]) * (point[0] - c[0]) + (a[0] - c[0]) * (point[1] - c[1])) / denominator
                w2 = 1.0 - w0 - w1
                if w0 >= -1e-6 and w1 >= -1e-6 and w2 >= -1e-6:
                    chosen = int(candidate)
                    weights = np.clip(np.array([w0, w1, w2]), 0.0, 1.0)
                    weights /= weights.sum()
                    break
            world[index] = weights @ self.vertices[self.faces[chosen]]
            facing[index] = self.facing[chosen]
        return world, facing


def extract_texture_lines(
    canonical_vertices_mm: object,
    faces: object,
    atlas: TextureAtlas,
    normal_map: NormalMap,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[TextureLinesPayload, dict[str, Any]]:
    """Trace the incisions the normal map holds, as seen in the recipe's view."""

    validated = validate_texture_lines_recipe(recipe)
    vertices = np.asarray(canonical_vertices_mm, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ArtifactTextureLinesError("mesh must be (n, 3) vertices and (m, 3) faces")
    if int(vertices.shape[0]) != validated["source_vertex_count"] or int(triangles.shape[0]) != validated[
        "source_face_count"
    ]:
        raise ArtifactTextureLinesError("mesh does not match the recipe's vertex and face counts")
    try:
        require_texture_relief_sources(validated["texture_relief"], atlas, normal_map)
    except ArtifactTextureReliefError as exc:
        raise ArtifactTextureLinesError(str(exc)) from exc
    if triangles.shape != atlas.triangles.shape or not np.array_equal(triangles, atlas.triangles):
        raise ArtifactTextureLinesError(
            "the mesh is not the texture atlas's geometry: its triangles differ; "
            "open the geometry the atlas welds to (write_atlas_geometry)"
        )
    try:
        rotation = rigid_rotation_between(atlas.vertices, vertices)
    except ArtifactTextureReliefError as exc:
        raise ArtifactTextureLinesError(str(exc)) from exc
    raise_if_cancelled(cancellation_probe)

    frame = outline_frame(validated["view"])
    origin = np.asarray(frame.origin_world_mm, dtype=np.float64)
    u_axis = np.asarray(frame.u_axis_world, dtype=np.float64)
    v_axis = np.asarray(frame.v_axis_world, dtype=np.float64)
    toward_viewer = np.asarray(frame.normal_world, dtype=np.float64)
    raster_policy = validated["raster_policy"]
    detection = validated["detection_policy"]
    pixels_per_mm = int(raster_policy["pixels_per_mm"])
    facing_cos = raster_policy["facing_cos_millionths"] / 1_000_000.0

    corners = vertices[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    facing = np.zeros(triangles.shape[0], dtype=np.float64)
    nonzero = lengths > 0.0
    facing[nonzero] = (normals[nonzero] @ toward_viewer) / lengths[nonzero]
    domain = validated["domain"]
    view_uv = np.column_stack([(vertices - origin) @ u_axis, (vertices - origin) @ v_axis])
    if domain == TEXTURE_LINES_DOMAIN_VIEW:
        visible = np.flatnonzero(facing >= facing_cos)
        if visible.size == 0:
            raise ArtifactTextureLinesError("no face of the mesh faces the view; nothing to read")
        # Far to near: the nearest wall is painted last and wins its pixels.
        depth = corners[visible].mean(axis=1) @ toward_viewer
        order = visible[np.argsort(depth, kind="stable")]
        uv = view_uv
    else:
        if abs(float(toward_viewer[2])) > 1e-9:
            raise ArtifactTextureLinesError(
                "axis_development reads a side view; the top and bottom look along the axis"
            )
        # Every outer-wall face that faces the view at all, unrolled about
        # the axis: u the arc from the view's own meridian at that vertex's
        # radius, v the axial height.  The inner wall of the far side faces
        # the view too, but it is behind the near wall and this is a reading
        # of the outside, so only faces whose normal leaves the axis are
        # taken; among those, faces are painted from the axis outward and
        # the outermost wins its pixels where a chip folds under.
        centroids = corners.mean(axis=1)
        outward = np.einsum("ij,ij->i", normals[:, :2], centroids[:, :2]) > 0.0
        view_angle = math.atan2(float(toward_viewer[1]), float(toward_viewer[0]))
        centroid_turn = (
            np.arctan2(centroids[:, 1], centroids[:, 0]) - view_angle + math.pi
        ) % (2.0 * math.pi) - math.pi
        # And on the near half by position: a facet on the far side that
        # happens to tilt towards the view would unroll a second sheet of
        # paper beyond the silhouette.
        near = np.abs(centroid_turn) <= math.pi / 2.0
        visible = np.flatnonzero((facing > 0.0) & outward & near)
        if visible.size == 0:
            raise ArtifactTextureLinesError(
                "no outer-wall face of the mesh faces the view; nothing to read"
            )
        radius_of = np.linalg.norm(centroids[visible][:, :2], axis=1)
        order = visible[np.argsort(radius_of, kind="stable")]
        theta = np.arctan2(vertices[:, 1], vertices[:, 0])
        turn = (theta - view_angle + math.pi) % (2.0 * math.pi) - math.pi
        uv = np.column_stack([np.hypot(vertices[:, 0], vertices[:, 1]) * turn, vertices[:, 2]])
    # Only the vertices the painted faces use: the raster is sized from the
    # developed coordinates it is given, and the far side of a pot unrolled
    # about its axis would otherwise double the paper for nothing.
    used = np.unique(triangles[order])
    compact = np.full(vertices.shape[0], -1, dtype=np.int64)
    compact[used] = np.arange(used.size, dtype=np.int64)
    developed_faces = compact[triangles[order]]
    try:
        height, minimum_u, minimum_v, relief_qc = texture_relief_depth_field(
            developed_uv_mm=uv[used],
            developed_faces=developed_faces,
            developed_points_mm=vertices[used],
            source_face_indices=order,
            source_vertex_indices=used,
            atlas=atlas,
            normal_map=normal_map,
            source_to_canonical_rotation=rotation,
            pixels_per_mm=pixels_per_mm,
            margin_pixels=2,
            smoothing_um=int(validated["texture_relief"]["smoothing_um"]),
            cancellation_probe=cancellation_probe,
        )
    except ArtifactTextureReliefError as exc:
        raise ArtifactTextureLinesError(str(exc)) from exc
    raise_if_cancelled(cancellation_probe)
    good = np.isfinite(height)
    sign = int(detection["incision_sign"])
    scale_mm = detection["scale_um"] / 1000.0
    curvature_min = detection["curvature_min_per_m"] / 1000.0
    # An incision is a valley of the height when the map's normals point out
    # of the wall and a ridge when they point in; the recipe says which, and
    # both counts are reported so the choice can be checked.
    curvature_seed = detection["curvature_seed_per_m"] / 1000.0
    signed = np.where(good, -sign * height, -np.inf)
    valley, du, dv = _valley_points(
        signed,
        good,
        pixels_per_mm=pixels_per_mm,
        scale_mm=scale_mm,
        curvature_min_per_mm=curvature_min,
        curvature_seed_per_mm=curvature_seed,
    )
    raise_if_cancelled(cancellation_probe)
    opposite, _du, _dv = _valley_points(
        np.where(good, sign * height, -np.inf),
        good,
        pixels_per_mm=pixels_per_mm,
        scale_mm=scale_mm,
        curvature_min_per_mm=curvature_min,
        curvature_seed_per_mm=curvature_seed,
    )
    raise_if_cancelled(cancellation_probe)
    chains = _trace_chains(valley)
    min_length = detection["min_length_um"] / 1000.0
    smoothing = detection["line_smoothing_um"] / 1000.0
    locator = (
        None
        if domain == TEXTURE_LINES_DOMAIN_VIEW
        else _DevelopedLocator(uv[used], developed_faces, vertices[used], facing[order])
    )
    total_raw = len(chains)
    traced: list[np.ndarray] = []
    for chain in chains:
        if len(chain) < 2:
            continue
        rows = np.asarray([r for r, _c in chain], dtype=np.int64)
        cols = np.asarray([c for _r, c in chain], dtype=np.int64)
        # Pixel centre plus the sub-pixel offset to the floor, in developed
        # mm; row 0 is the lowest v, as the relief raster has it.
        traced.append(
            np.column_stack(
                [
                    (minimum_u + cols + 0.5 + du[rows, cols]) / float(pixels_per_mm),
                    (minimum_v + rows + 0.5 + dv[rows, cols]) / float(pixels_per_mm),
                ]
            )
        )
    raise_if_cancelled(cancellation_probe)
    traced = _link_polylines(
        traced, gap_mm=detection["link_um"] / 1000.0, angle_deg=DEFAULT_TEXTURE_LINES_LINK_ANGLE_DEG
    )
    polylines: list[Polyline] = []
    for developed in traced:
        if locator is None:
            runs = [developed]
        else:
            # Back through the triangle each point lies in to its canonical
            # position, then onto the view; a line is cut where the wall
            # turns past the facing threshold, so nothing is drawn round the
            # silhouette onto the far side.
            world, face_facing = locator.locate(developed)
            projected = np.column_stack([(world - origin) @ u_axis, (world - origin) @ v_axis])
            keep = face_facing >= facing_cos
            runs = []
            start = None
            for index, flag in enumerate(keep.tolist() + [False]):
                if flag and start is None:
                    start = index
                elif not flag and start is not None:
                    if index - start >= 2:
                        runs.append(projected[start:index])
                    start = None
        for points in runs:
            length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
            if length < min_length:
                continue
            smoothed = (
                smooth_polyline(points, closed=False, sigma_mm=smoothing)
                if smoothing > 0.0
                else tuple((float(x), float(y)) for x, y in points)
            )
            as_um: list[tuple[int, int]] = []
            for x, y in smoothed:
                point = (int(round(x * 1000.0)), int(round(y * 1000.0)))
                if as_um and as_um[-1] == point:
                    continue
                as_um.append(point)
            if len(as_um) >= 2:
                polylines.append(tuple(as_um))
        raise_if_cancelled(cancellation_probe)
    if not polylines:
        raise ArtifactTextureLinesError(
            "no incision was traced in this view; lower curvature_min_per_mm or "
            "min_length_mm, or do not take the reading"
        )
    if len(polylines) > MAX_TEXTURE_LINES:
        raise ArtifactTextureLinesError(
            f"a texture lines reading holds at most {MAX_TEXTURE_LINES} lines; raise "
            "min_length_mm or curvature_min_per_mm"
        )
    payload = TextureLinesPayload(
        schema_version=TEXTURE_LINES_PAYLOAD_SCHEMA_VERSION,
        view=validated["view"],
        polylines=tuple(polylines),
    )
    qc = {
        **payload.qc_summary(),
        "chain_count_before_filter": total_raw,
        "chain_count_after_link": len(traced),
        "domain": domain,
        "texture_relief_covered_pixel_count": relief_qc["texture_relief_covered_pixel_count"],
        "texture_relief_height_max_um_rounded": relief_qc["texture_relief_height_max_um_rounded"],
        "texture_relief_height_min_um_rounded": relief_qc["texture_relief_height_min_um_rounded"],
        "texture_relief_integration_misfit_millionths": relief_qc[
            "texture_relief_integration_misfit_millionths"
        ],
        "opposite_sign_pixel_count": int(np.count_nonzero(opposite)),
        "valley_pixel_count": int(np.count_nonzero(valley)),
        "visible_face_count": int(visible.size),
    }
    return payload, qc


@dataclass(frozen=True, slots=True)
class TextureLinesComputation:
    context: OperationContext
    projection_snapshot: Any
    payload: TextureLinesPayload
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]


def compute_texture_lines(
    session: ArtifactSession,
    atlas: TextureAtlas,
    normal_map: NormalMap,
    *,
    view: OutlineView | str,
    pixels_per_mm: int = DEFAULT_TEXTURE_LINES_PIXELS_PER_MM,
    smoothing_um: int = DEFAULT_TEXTURE_LINES_SMOOTHING_UM,
    facing_cos: float = DEFAULT_TEXTURE_LINES_FACING_COS,
    scale_mm: float = DEFAULT_TEXTURE_LINES_SCALE_MM,
    curvature_min_per_mm: float = DEFAULT_TEXTURE_LINES_CURVATURE_MIN_PER_MM,
    min_length_mm: float = DEFAULT_TEXTURE_LINES_MIN_LENGTH_MM,
    line_smoothing_mm: float = DEFAULT_TEXTURE_LINES_LINE_SMOOTHING_MM,
    incision_sign: int = DEFAULT_TEXTURE_LINES_INCISION_SIGN,
    domain: str = TEXTURE_LINES_DOMAIN_VIEW,
    curvature_seed_per_mm: float | None = None,
    link_mm: float = DEFAULT_TEXTURE_LINES_LINK_MM,
    cancellation_probe: CancellationProbe | None = None,
) -> TextureLinesComputation:
    """Trace the wall's incisions as positioned by the session's active Align."""

    if not isinstance(session, ArtifactSession):
        raise ArtifactTextureLinesError("session must be an ArtifactSession")
    if _domain(domain) == TEXTURE_LINES_DOMAIN_AXIS:
        from .artifact_axis_alignment import AXIS_ALIGN_RECIPE_KIND  # noqa: PLC0415

        align_id = session.document.active_align_revision_id
        align = (
            session.document.align_revision_index.get(align_id)
            if isinstance(align_id, str)
            else None
        )
        if align is None or align.recipe.get("kind") != AXIS_ALIGN_RECIPE_KIND:
            raise ArtifactTextureLinesError(
                "unrolling about the axis needs an artifact positioned on its "
                "measured rotation axis; the active Align was not made from one"
            )
    if not isinstance(atlas, TextureAtlas) or not isinstance(normal_map, NormalMap):
        raise ArtifactTextureLinesError("atlas and normal_map must be TextureAtlas and NormalMap")
    try:
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactTextureLinesError(str(exc)) from exc
    vertices = np.asarray(projection.mesh.vertices, dtype=np.float64)
    triangles = np.asarray(projection.mesh.faces, dtype=np.int64)
    recipe = texture_lines_recipe(
        atlas,
        normal_map,
        view=view,
        source_vertex_count=int(vertices.shape[0]),
        source_face_count=int(triangles.shape[0]),
        pixels_per_mm=pixels_per_mm,
        smoothing_um=smoothing_um,
        facing_cos=facing_cos,
        scale_mm=scale_mm,
        curvature_min_per_mm=curvature_min_per_mm,
        min_length_mm=min_length_mm,
        line_smoothing_mm=line_smoothing_mm,
        incision_sign=incision_sign,
        domain=domain,
        curvature_seed_per_mm=curvature_seed_per_mm,
        link_mm=link_mm,
    )
    try:
        context = session.capture_operation(recipe=recipe)
    except ArtifactSessionError as exc:
        raise ArtifactTextureLinesError(str(exc)) from exc
    payload, qc = extract_texture_lines(
        vertices, triangles, atlas, normal_map, recipe, cancellation_probe=cancellation_probe
    )
    return TextureLinesComputation(
        context=context,
        projection_snapshot=projection.snapshot,
        payload=payload,
        recipe=recipe,
        qc=qc,
    )


def texture_lines_computation_matches_active_projection(
    session: ArtifactSession, computation: TextureLinesComputation
) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(
        computation, TextureLinesComputation
    ):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    return current.render_key == computation.projection_snapshot.render_key


def commit_texture_lines(
    session: ArtifactSession,
    computation: TextureLinesComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    if not texture_lines_computation_matches_active_projection(session, computation):
        raise ArtifactTextureLinesError(
            "texture lines computation is stale for the active projection"
        )
    validated_recipe = validate_texture_lines_recipe(computation.recipe)
    payload = computation.payload
    payload_bytes = payload.canonical_json_bytes()
    extensions = {
        TEXTURE_LINES_PAYLOAD_EXTENSION_KEY: {
            "byte_length": len(payload_bytes),
            "media_type": TEXTURE_LINES_PAYLOAD_MEDIA_TYPE,
            "payload": payload.to_dict(),
            "schema_version": TEXTURE_LINES_PAYLOAD_SCHEMA_VERSION,
            "sha256": payload.sha256,
        }
    }
    try:
        document = session.document.append_record_from_context(
            context=computation.context,
            id=record_id,
            type=TEXTURE_LINES_RECORD_TYPE,
            geometry_ref=payload.geometry_ref,
            recipe=dict(validated_recipe),
            qc=dict(computation.qc),
            lifecycle_status=RecordLifecycleStatus.READY,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=depends_on_record_ids,
            extensions=extensions,
        )
    except ArtifactDocumentError as exc:
        raise ArtifactTextureLinesError(str(exc)) from exc
    return session.with_document(document)


_DESCRIPTOR_KEYS = frozenset(
    {"byte_length", "media_type", "payload", "schema_version", "sha256"}
)


def texture_lines_payload_from_record(record: DerivedRecord) -> TextureLinesPayload:
    """Resolve and re-verify one record's inline reading."""

    if not isinstance(record, DerivedRecord):
        raise ArtifactTextureLinesError("record must be a DerivedRecord")
    if record.type != TEXTURE_LINES_RECORD_TYPE:
        raise ArtifactTextureLinesError(f"record is not a texture lines reading: {record.type!r}")
    descriptor = _exact_keys(
        record.extensions.get(TEXTURE_LINES_PAYLOAD_EXTENSION_KEY),
        _DESCRIPTOR_KEYS,
        name="texture lines payload descriptor",
    )
    if descriptor["media_type"] != TEXTURE_LINES_PAYLOAD_MEDIA_TYPE:
        raise ArtifactTextureLinesError("texture lines payload media_type is invalid")
    if descriptor["schema_version"] != TEXTURE_LINES_PAYLOAD_SCHEMA_VERSION:
        raise ArtifactTextureLinesError("texture lines payload descriptor schema is invalid")
    raw_payload = descriptor["payload"]
    if not isinstance(raw_payload, Mapping):
        raise ArtifactTextureLinesError("texture lines payload descriptor payload must be an object")
    payload = TextureLinesPayload.from_dict(raw_payload)
    payload_bytes = payload.canonical_json_bytes()
    byte_length = descriptor["byte_length"]
    if type(byte_length) is not int or byte_length != len(payload_bytes):
        raise ArtifactTextureLinesError("texture lines payload byte_length does not match payload")
    if descriptor["sha256"] != payload.sha256:
        raise ArtifactTextureLinesError("texture lines payload SHA-256 does not match payload")
    if record.geometry_ref != payload.geometry_ref:
        raise ArtifactTextureLinesError("texture lines record geometry_ref does not match payload")
    validated = validate_texture_lines_recipe(record.recipe)
    if validated["view"] != payload.view:
        raise ArtifactTextureLinesError("texture lines record recipe and payload name different views")
    thawed_qc = record.to_dict()["qc"]
    assert isinstance(thawed_qc, dict)
    summary = payload.qc_summary()
    if any(thawed_qc.get(key) != value for key, value in summary.items()):
        raise ArtifactTextureLinesError("texture lines record QC does not match its payload")
    return payload


def validate_texture_lines_records(document: ArtifactDocument) -> None:
    """Strictly validate every texture lines record embedded in a document."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactTextureLinesError("document must be an ArtifactDocument")
    for record in document.records:
        if record.type == TEXTURE_LINES_RECORD_TYPE:
            texture_lines_payload_from_record(record)


__all__ = [
    "ArtifactTextureLinesError",
    "DEFAULT_TEXTURE_LINES_CURVATURE_MIN_PER_MM",
    "DEFAULT_TEXTURE_LINES_CURVATURE_SEED_PER_MM",
    "DEFAULT_TEXTURE_LINES_LINK_MM",
    "DEFAULT_TEXTURE_LINES_FACING_COS",
    "DEFAULT_TEXTURE_LINES_INCISION_SIGN",
    "DEFAULT_TEXTURE_LINES_LINE_SMOOTHING_MM",
    "DEFAULT_TEXTURE_LINES_MIN_LENGTH_MM",
    "DEFAULT_TEXTURE_LINES_PIXELS_PER_MM",
    "DEFAULT_TEXTURE_LINES_SCALE_MM",
    "DEFAULT_TEXTURE_LINES_SMOOTHING_UM",
    "TEXTURE_LINES_ALGORITHM",
    "TEXTURE_LINES_ALGORITHM_VERSION",
    "TEXTURE_LINES_DOMAINS",
    "TEXTURE_LINES_DOMAIN_AXIS",
    "TEXTURE_LINES_DOMAIN_VIEW",
    "TEXTURE_LINES_PAYLOAD_EXTENSION_KEY",
    "TEXTURE_LINES_RECORD_TYPE",
    "TextureLinesComputation",
    "TextureLinesPayload",
    "commit_texture_lines",
    "compute_texture_lines",
    "extract_texture_lines",
    "texture_lines_computation_matches_active_projection",
    "texture_lines_payload_from_record",
    "texture_lines_recipe",
    "validate_texture_lines_recipe",
    "validate_texture_lines_records",
]
