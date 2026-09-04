"""홈: the grooves that run right round a pot, found from its own profile.

A groove that circles the body is drawn with three lines, not one.  The
recessed line at the bottom of the groove is a 간선 - a straight line broken a
few times - and the two raised edges either side are solid 직선, because a
groove is one place that goes in and two that stand out.

Nobody should have to paint those by hand.  On an artifact stood on its
measured rotation axis, a groove that goes all the way round is exactly a
place where the wall is set back from the wall around it, at every angle.  So
this module reads the outer wall's radius as a function of height and looks
for the set-back bands; the drawing convention and the geometry describe the
same thing.

Three decisions carry the result.

*The radius at a height is the median across the revolution.*  A groove that
circles the pot moves that median.  A dent on one side does not, and a
one-sided dent is damage, not technique - a different record type says so.

*The wall it is measured against is a local quadratic fit, refitted with the
set-back bins removed.*  A pot wall curves, and a straight local model reads
that curvature as a groove: on the test profile a 24 mm straight window is
wrong by 0.26 mm, which is deeper than the grooves being looked for.  The
refit keeps a groove from dragging down the very baseline it is measured
against.

*A groove has raised ground on both sides.*  The dip beside a single cordon
has it on one side only, and that dip is the cordon's flank, not a groove.
The two edges have to agree, or the candidate is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
from .artifact_session import ArtifactSession, ArtifactSessionError
from .artifact_surface_strip import (
    ArtifactSurfaceStripError,
    select_surface_strip,
    strip_parameters,
)
from .canonical_json import (
    CanonicalJSONError,
    canonical_json_bytes,
    canonical_json_sha256,
)


PROFILE_GROOVE_RECORD_TYPE = "measurement.profile_groove.v1"
PROFILE_GROOVE_OPERATION_KIND = "profile_groove"
PROFILE_GROOVE_ALGORITHM = "archmeshrubbing.axial_profile_groove"
PROFILE_GROOVE_ALGORITHM_VERSION = "1.0.0"
PROFILE_GROOVE_COORDINATE_SPACE = "canonical_mm_axis_profile/v1"
PROFILE_GROOVE_PAYLOAD_SCHEMA_VERSION = "1.0.0"
PROFILE_GROOVE_PAYLOAD_EXTENSION_KEY = "org.archmeshrubbing:profile-groove-v1"
PROFILE_GROOVE_PAYLOAD_MEDIA_TYPE = (
    "application/vnd.archmeshrubbing.profile-groove+json"
)
PROFILE_GROOVE_GEOMETRY_REF_PREFIX = "urn:archmeshrubbing:profile-groove:sha256:"

DEFAULT_GROOVE_HEIGHT_BIN_UM = 250
DEFAULT_GROOVE_MINIMUM_DEPTH_UM = 150
DEFAULT_GROOVE_MAXIMUM_WIDTH_UM = 8_000

# Fixed policy.  These are recorded in the recipe rather than left in code, so
# a stored record keeps determining its own result even if a later release
# would choose differently.
GROOVE_BASELINE_HALF_WINDOW_PERCENT = 150
GROOVE_BASELINE_FIT_DEGREE = 2
GROOVE_MINIMUM_BIN_SAMPLE_COUNT = 4
GROOVE_TROUGH_TIE_PERCENT = 5
GROOVE_EDGE_CLIMB_PERCENT = 5
GROOVE_EDGE_ASYMMETRY_PERCENT = 50

MIN_GROOVE_HEIGHT_BIN_UM = 10
MAX_GROOVE_HEIGHT_BIN_UM = 5_000
MIN_GROOVE_DEPTH_UM = 10
MAX_GROOVE_DEPTH_UM = 50_000
MIN_GROOVE_WIDTH_UM = 100
MAX_GROOVE_WIDTH_UM = 100_000
MAX_GROOVE_PROFILE_BINS = 200_000
MAX_GROOVE_COUNT = 512


class ArtifactProfileGrooveError(ValueError):
    """A groove reading cannot be produced from this artifact safely."""


def _strict_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactProfileGrooveError(f"{name} must be an integer")
    number = int(value)
    if number < minimum or number > maximum:
        raise ArtifactProfileGrooveError(
            f"{name} must be in the inclusive range {minimum}..{maximum}"
        )
    return number


def _exact_keys(
    value: object,
    keys: frozenset[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactProfileGrooveError(f"{name} must be an object")
    if set(value) != set(keys):
        raise ArtifactProfileGrooveError(
            f"{name} must carry exactly {', '.join(sorted(keys))}"
        )
    return value


def profile_groove_recipe(
    *,
    height_bin_um: int = DEFAULT_GROOVE_HEIGHT_BIN_UM,
    minimum_depth_um: int = DEFAULT_GROOVE_MINIMUM_DEPTH_UM,
    maximum_width_um: int = DEFAULT_GROOVE_MAXIMUM_WIDTH_UM,
) -> dict[str, Any]:
    """Resolve the three numbers that decide what counts as a groove."""

    bin_um = _strict_int(
        height_bin_um,
        name="height_bin_um",
        minimum=MIN_GROOVE_HEIGHT_BIN_UM,
        maximum=MAX_GROOVE_HEIGHT_BIN_UM,
    )
    depth_um = _strict_int(
        minimum_depth_um,
        name="minimum_depth_um",
        minimum=MIN_GROOVE_DEPTH_UM,
        maximum=MAX_GROOVE_DEPTH_UM,
    )
    width_um = _strict_int(
        maximum_width_um,
        name="maximum_width_um",
        minimum=MIN_GROOVE_WIDTH_UM,
        maximum=MAX_GROOVE_WIDTH_UM,
    )
    if width_um < 4 * bin_um:
        raise ArtifactProfileGrooveError(
            "maximum_width_um must span at least four height bins, or a groove "
            "cannot be told from a single sample; widen it or use a finer "
            "height_bin_um"
        )
    return {
        "algorithm": PROFILE_GROOVE_ALGORITHM,
        "algorithm_version": PROFILE_GROOVE_ALGORITHM_VERSION,
        "baseline_policy": {
            "fit": "local_least_squares_polynomial/v1",
            "fit_degree": GROOVE_BASELINE_FIT_DEGREE,
            "half_window_percent_of_maximum_width": (
                GROOVE_BASELINE_HALF_WINDOW_PERCENT
            ),
            "refit": "drop_bins_below_minimum_depth/v1",
        },
        "coordinate_space": PROFILE_GROOVE_COORDINATE_SPACE,
        "detection_policy": {
            "edge_asymmetry_percent": GROOVE_EDGE_ASYMMETRY_PERCENT,
            "edge_climb_percent": GROOVE_EDGE_CLIMB_PERCENT,
            "maximum_width_um": width_um,
            "minimum_depth_um": depth_um,
            "trough_tie_percent": GROOVE_TROUGH_TIE_PERCENT,
        },
        "kind": PROFILE_GROOVE_OPERATION_KIND,
        "longitudinal_axis": "z",
        "profile_policy": {
            "height_bin_um": bin_um,
            "minimum_bin_sample_count": GROOVE_MINIMUM_BIN_SAMPLE_COUNT,
            "radius_statistic": "median_across_revolution/v1",
            "surface": "outward_wall_full_revolution/v1",
        },
        "resource_limits": {
            "max_groove_count": MAX_GROOVE_COUNT,
            "max_profile_bins": MAX_GROOVE_PROFILE_BINS,
        },
    }


def validate_profile_groove_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the recipe from its own numbers and require the same bytes."""

    if not isinstance(recipe, Mapping):
        raise ArtifactProfileGrooveError("profile groove recipe must be an object")
    profile_policy = recipe.get("profile_policy")
    detection_policy = recipe.get("detection_policy")
    if not isinstance(profile_policy, Mapping) or not isinstance(
        detection_policy, Mapping
    ):
        raise ArtifactProfileGrooveError("profile groove recipe policies are invalid")
    expected = profile_groove_recipe(
        height_bin_um=profile_policy.get("height_bin_um"),  # type: ignore[arg-type]
        minimum_depth_um=detection_policy.get("minimum_depth_um"),  # type: ignore[arg-type]
        maximum_width_um=detection_policy.get("maximum_width_um"),  # type: ignore[arg-type]
    )
    try:
        same = canonical_json_bytes(dict(recipe)) == canonical_json_bytes(expected)
    except CanonicalJSONError as exc:
        raise ArtifactProfileGrooveError(str(exc)) from exc
    if not same:
        raise ArtifactProfileGrooveError(
            "profile groove recipe does not match the production contract"
        )
    return expected


@dataclass(frozen=True, slots=True)
class ProfileGroove:
    """One groove: where its bottom is, where its two raised edges are."""

    trough_height_um: int
    trough_radius_um: int
    lower_edge_height_um: int
    lower_edge_radius_um: int
    upper_edge_height_um: int
    upper_edge_radius_um: int
    depth_um: int
    revolution_spread_um: int

    def __post_init__(self) -> None:
        if not (
            self.lower_edge_height_um
            < self.trough_height_um
            < self.upper_edge_height_um
        ):
            raise ArtifactProfileGrooveError(
                "a groove's bottom must lie between its two edges"
            )
        if self.depth_um <= 0:
            raise ArtifactProfileGrooveError("a groove must have a positive depth")
        if self.revolution_spread_um < 0:
            raise ArtifactProfileGrooveError(
                "a groove's spread across the revolution cannot be negative"
            )

    @property
    def width_um(self) -> int:
        return self.upper_edge_height_um - self.lower_edge_height_um

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_um": self.depth_um,
            "lower_edge_height_um": self.lower_edge_height_um,
            "lower_edge_radius_um": self.lower_edge_radius_um,
            "revolution_spread_um": self.revolution_spread_um,
            "trough_height_um": self.trough_height_um,
            "trough_radius_um": self.trough_radius_um,
            "upper_edge_height_um": self.upper_edge_height_um,
            "upper_edge_radius_um": self.upper_edge_radius_um,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProfileGroove":
        block = _exact_keys(
            data,
            frozenset(
                {
                    "depth_um",
                    "lower_edge_height_um",
                    "lower_edge_radius_um",
                    "revolution_spread_um",
                    "trough_height_um",
                    "trough_radius_um",
                    "upper_edge_height_um",
                    "upper_edge_radius_um",
                }
            ),
            name="profile groove",
        )
        limit = 10**12
        return cls(
            **{
                key: _strict_int(block[key], name=key, minimum=-limit, maximum=limit)
                for key in block
            }
        )


@dataclass(frozen=True, slots=True)
class ProfileGroovePayload:
    """Every groove one reading found, ordered up the artifact."""

    schema_version: str
    grooves: tuple[ProfileGroove, ...]
    profile_bin_count: int
    profile_minimum_height_um: int
    profile_maximum_height_um: int

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_GROOVE_PAYLOAD_SCHEMA_VERSION:
            raise ArtifactProfileGrooveError(
                f"unsupported profile groove payload schema: {self.schema_version!r}"
            )
        grooves = tuple(self.grooves)
        if any(not isinstance(groove, ProfileGroove) for groove in grooves):
            raise ArtifactProfileGrooveError(
                "profile groove payload holds ProfileGroove values"
            )
        if not grooves:
            raise ArtifactProfileGrooveError(
                "a groove reading with no groove records nothing; relax "
                "minimum_depth_um or maximum_width_um, or do not take the reading"
            )
        if len(grooves) > MAX_GROOVE_COUNT:
            raise ArtifactProfileGrooveError(
                f"a groove reading holds at most {MAX_GROOVE_COUNT} grooves"
            )
        ordered = tuple(sorted(grooves, key=lambda item: item.trough_height_um))
        heights = [groove.trough_height_um for groove in ordered]
        if len(set(heights)) != len(heights):
            raise ArtifactProfileGrooveError(
                "two grooves cannot share one trough height"
            )
        for lower, upper in zip(ordered, ordered[1:]):
            if lower.upper_edge_height_um > upper.lower_edge_height_um:
                raise ArtifactProfileGrooveError(
                    "grooves must not overlap: one groove's upper edge sits "
                    "above the next groove's lower edge"
                )
        object.__setattr__(self, "grooves", ordered)
        if self.profile_minimum_height_um >= self.profile_maximum_height_um:
            raise ArtifactProfileGrooveError(
                "the profile's height range must be non-empty"
            )
        if self.profile_bin_count <= 0:
            raise ArtifactProfileGrooveError("the profile must hold at least one bin")

    def to_dict(self) -> dict[str, Any]:
        return {
            "grooves": [groove.to_dict() for groove in self.grooves],
            "profile_bin_count": self.profile_bin_count,
            "profile_maximum_height_um": self.profile_maximum_height_um,
            "profile_minimum_height_um": self.profile_minimum_height_um,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProfileGroovePayload":
        block = _exact_keys(
            data,
            frozenset(
                {
                    "grooves",
                    "profile_bin_count",
                    "profile_maximum_height_um",
                    "profile_minimum_height_um",
                    "schema_version",
                }
            ),
            name="profile groove payload",
        )
        raw = block["grooves"]
        if not isinstance(raw, (list, tuple)):
            raise ArtifactProfileGrooveError(
                "profile groove payload grooves must be an array"
            )
        schema_version = block["schema_version"]
        limit = 10**12
        return cls(
            schema_version=schema_version if isinstance(schema_version, str) else "",
            grooves=tuple(
                ProfileGroove.from_dict(entry)  # type: ignore[arg-type]
                for entry in raw
            ),
            profile_bin_count=_strict_int(
                block["profile_bin_count"],
                name="profile_bin_count",
                minimum=0,
                maximum=MAX_GROOVE_PROFILE_BINS,
            ),
            profile_minimum_height_um=_strict_int(
                block["profile_minimum_height_um"],
                name="profile_minimum_height_um",
                minimum=-limit,
                maximum=limit,
            ),
            profile_maximum_height_um=_strict_int(
                block["profile_maximum_height_um"],
                name="profile_maximum_height_um",
                minimum=-limit,
                maximum=limit,
            ),
        )

    def canonical_json_bytes(self) -> bytes:
        try:
            return canonical_json_bytes(self.to_dict())
        except CanonicalJSONError as exc:
            raise ArtifactProfileGrooveError(str(exc)) from exc

    @property
    def sha256(self) -> str:
        try:
            return canonical_json_sha256(self.to_dict())
        except CanonicalJSONError as exc:
            raise ArtifactProfileGrooveError(str(exc)) from exc

    @property
    def geometry_ref(self) -> str:
        return f"{PROFILE_GROOVE_GEOMETRY_REF_PREFIX}{self.sha256}"

    def qc_summary(self) -> dict[str, Any]:
        depths = [groove.depth_um for groove in self.grooves]
        widths = [groove.width_um for groove in self.grooves]
        spreads = [groove.revolution_spread_um for groove in self.grooves]
        return {
            "groove_count": len(self.grooves),
            "maximum_depth_um": max(depths),
            "maximum_revolution_spread_um": max(spreads),
            "maximum_width_um": max(widths),
            "minimum_depth_um": min(depths),
            "minimum_width_um": min(widths),
            "payload_sha256": self.sha256,
            "profile_bin_count": self.profile_bin_count,
            "profile_maximum_height_um": self.profile_maximum_height_um,
            "profile_minimum_height_um": self.profile_minimum_height_um,
            "trough_heights_um": [
                groove.trough_height_um for groove in self.grooves
            ],
        }


def _outer_wall_profile(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    height_bin_um: int,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (height mm, median radius mm, spread mm) up the outer wall.

    The outer wall is taken with the same selector a strip is cut with, so the
    inner wall, the rim annulus and the floor cannot reach the profile: a
    reading that mixed the two walls would find grooves in neither.
    """

    raise_if_cancelled(cancellation_probe)
    try:
        selection = select_surface_strip(vertices, faces, strip_parameters())
    except ArtifactSurfaceStripError as exc:
        raise ArtifactProfileGrooveError(
            f"the outer wall could not be told from the inner: {exc}"
        ) from exc
    used = np.unique(np.asarray(faces)[selection.face_indices].reshape(-1))
    points = np.asarray(vertices, dtype=np.float64)[used]
    raise_if_cancelled(cancellation_probe)
    heights = points[:, 2]
    radii = np.hypot(points[:, 0], points[:, 1])
    lowest = float(heights.min())
    highest = float(heights.max())
    bin_mm = float(height_bin_um) / 1000.0
    span = highest - lowest
    if span <= 0.0:
        raise ArtifactProfileGrooveError(
            "the outer wall has no height to read a profile along"
        )
    count = int(math.ceil(span / bin_mm))
    if count > MAX_GROOVE_PROFILE_BINS:
        raise ArtifactProfileGrooveError(
            f"a {span:.1f} mm wall at {bin_mm:.3f} mm bins needs {count} bins, "
            f"past the {MAX_GROOVE_PROFILE_BINS} safety limit; use a coarser "
            "height_bin_um"
        )
    edges = lowest + bin_mm * np.arange(count + 1, dtype=np.float64)
    index = np.clip(np.searchsorted(edges, heights, side="right") - 1, 0, count - 1)
    raise_if_cancelled(cancellation_probe)
    order = np.argsort(index, kind="stable")
    sorted_index = index[order]
    sorted_radii = radii[order]
    starts = np.searchsorted(sorted_index, np.arange(count), side="left")
    stops = np.searchsorted(sorted_index, np.arange(count), side="right")
    centres: list[float] = []
    medians: list[float] = []
    spreads: list[float] = []
    for b in range(count):
        if stops[b] - starts[b] < GROOVE_MINIMUM_BIN_SAMPLE_COUNT:
            continue
        values = sorted_radii[starts[b] : stops[b]]
        low, high = np.percentile(values, (25.0, 75.0))
        centres.append(0.5 * float(edges[b] + edges[b + 1]))
        medians.append(float(np.median(values)))
        spreads.append(float(high - low))
    if len(centres) < 8:
        raise ArtifactProfileGrooveError(
            "the outer wall gave too few height bins to read a profile; use a "
            "coarser height_bin_um or a denser mesh"
        )
    return (
        np.asarray(centres, dtype=np.float64),
        np.asarray(medians, dtype=np.float64),
        np.asarray(spreads, dtype=np.float64),
    )


def _local_polynomial(
    heights: np.ndarray,
    radii: np.ndarray,
    keep: np.ndarray,
    *,
    half_window: int,
) -> np.ndarray:
    """Fit a low polynomial to each centred window and read it at the centre."""

    fitted = np.empty(heights.size, dtype=np.float64)
    needed = GROOVE_BASELINE_FIT_DEGREE + 2
    for i in range(heights.size):
        low = max(0, i - half_window)
        high = min(heights.size, i + half_window + 1)
        take = keep[low:high]
        window_heights = heights[low:high][take]
        window_radii = radii[low:high][take]
        if (
            window_heights.size >= needed
            and float(window_heights.max() - window_heights.min()) > 0.0
        ):
            centred = window_heights - heights[i]
            coefficients = np.polyfit(centred, window_radii, GROOVE_BASELINE_FIT_DEGREE)
            fitted[i] = float(np.polyval(coefficients, 0.0))
        else:
            fitted[i] = float(radii[i])
    return fitted


def _wall_baseline(
    heights: np.ndarray,
    radii: np.ndarray,
    *,
    half_window: int,
    minimum_depth_mm: float,
) -> np.ndarray:
    """The wall a groove is cut into, with the grooves taken back out."""

    first = _local_polynomial(
        heights, radii, np.ones(heights.size, dtype=bool), half_window=half_window
    )
    keep = (radii - first) > -minimum_depth_mm
    if int(np.count_nonzero(keep)) < GROOVE_BASELINE_FIT_DEGREE + 2:
        return first
    return _local_polynomial(heights, radii, keep, half_window=half_window)


def detect_profile_grooves(
    vertices: object,
    faces: object,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> ProfileGroovePayload:
    """Read every circumferential groove the recipe's numbers admit."""

    validated = validate_profile_groove_recipe(recipe)
    profile_policy = validated["profile_policy"]
    detection_policy = validated["detection_policy"]
    bin_um = int(profile_policy["height_bin_um"])
    minimum_depth_mm = float(detection_policy["minimum_depth_um"]) / 1000.0
    maximum_width_mm = float(detection_policy["maximum_width_um"]) / 1000.0
    bin_mm = float(bin_um) / 1000.0

    heights, radii, spreads = _outer_wall_profile(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces),
        height_bin_um=bin_um,
        cancellation_probe=cancellation_probe,
    )
    raise_if_cancelled(cancellation_probe)
    half_window = max(
        GROOVE_BASELINE_FIT_DEGREE + 1,
        int(round(GROOVE_BASELINE_HALF_WINDOW_PERCENT * maximum_width_mm / (100.0 * bin_mm))),
    )
    baseline = _wall_baseline(
        heights,
        radii,
        half_window=half_window,
        minimum_depth_mm=minimum_depth_mm,
    )
    raise_if_cancelled(cancellation_probe)
    residual = radii - baseline
    walk_cap = max(1, int(round(0.5 * maximum_width_mm / bin_mm)))
    set_back = residual <= -minimum_depth_mm

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(residual.size + 1):
        inside = i < residual.size and bool(set_back[i])
        if inside and start is None:
            start = i
        elif not inside and start is not None:
            runs.append((start, i - 1))
            start = None

    grooves: list[ProfileGroove] = []
    for low_index, high_index in runs:
        raise_if_cancelled(cancellation_probe)
        run = residual[low_index : high_index + 1]
        deepest = float(run.min())
        # A groove cut with a tool has a flat bottom, and which bin of a flat
        # bottom is deepest comes down to a hair of drift.  Every bin within a
        # small share of the depth counts as the bottom, and the line goes in
        # the middle of it, where a drafter would put it.
        tie = GROOVE_TROUGH_TIE_PERCENT * -deepest / 100.0
        tied = np.flatnonzero(run <= deepest + tie) + low_index
        trough = int(tied[tied.size // 2])

        # Out of the run the wall resumes.  Keep climbing only while the
        # surface gains appreciably, which is a raised rim; a wall whose
        # residual drifts by a hair is not a rim, so a plain groove's edge is
        # its own mouth.
        climb = GROOVE_EDGE_CLIMB_PERCENT * -float(residual[trough]) / 100.0

        def walk(edge: int, step: int) -> int | None:
            j = edge + step
            if j < 0 or j >= residual.size:
                return None
            while abs(j - trough) < walk_cap:
                following = j + step
                if following < 0 or following >= residual.size:
                    break
                if float(residual[following] - residual[j]) < climb:
                    break
                j = following
            return j

        lower = walk(low_index, -1)
        upper = walk(high_index, +1)
        if lower is None or upper is None:
            # A groove that runs off the end of the wall has no edge there, so
            # there is no second line to draw and no reading to record.
            continue
        rise_lower = float(residual[lower])
        rise_upper = float(residual[upper])
        depth_mm = min(rise_lower, rise_upper) - float(residual[trough])
        if depth_mm < minimum_depth_mm:
            continue
        if float(heights[upper] - heights[lower]) > maximum_width_mm:
            continue
        # A groove has raised ground on both sides.  A dip beside a single
        # cordon has it on one side only: that is the cordon's flank.
        asymmetry = GROOVE_EDGE_ASYMMETRY_PERCENT * depth_mm / 100.0
        if abs(rise_lower - rise_upper) > asymmetry:
            continue
        grooves.append(
            ProfileGroove(
                trough_height_um=int(round(float(heights[trough]) * 1000.0)),
                trough_radius_um=int(round(float(radii[trough]) * 1000.0)),
                lower_edge_height_um=int(round(float(heights[lower]) * 1000.0)),
                lower_edge_radius_um=int(round(float(radii[lower]) * 1000.0)),
                upper_edge_height_um=int(round(float(heights[upper]) * 1000.0)),
                upper_edge_radius_um=int(round(float(radii[upper]) * 1000.0)),
                depth_um=max(1, int(round(depth_mm * 1000.0))),
                revolution_spread_um=int(round(float(spreads[trough]) * 1000.0)),
            )
        )
        if len(grooves) > MAX_GROOVE_COUNT:
            raise ArtifactProfileGrooveError(
                f"more than {MAX_GROOVE_COUNT} grooves were read, which is a "
                "noisy surface rather than a technique; raise minimum_depth_um"
            )
    if not grooves:
        raise ArtifactProfileGrooveError(
            "no groove runs right round this artifact at "
            f"{float(detection_policy['minimum_depth_um']) / 1000.0:.2f} mm deep "
            f"and at most {maximum_width_mm:.1f} mm wide; lower minimum_depth_um "
            "for a shallower groove, or raise maximum_width_um for a broader one"
        )
    return ProfileGroovePayload(
        schema_version=PROFILE_GROOVE_PAYLOAD_SCHEMA_VERSION,
        grooves=tuple(grooves),
        profile_bin_count=int(heights.size),
        profile_minimum_height_um=int(round(float(heights[0]) * 1000.0)),
        profile_maximum_height_um=int(round(float(heights[-1]) * 1000.0)),
    )


@dataclass(frozen=True, slots=True)
class ProfileGrooveComputation:
    context: OperationContext
    projection_snapshot: Any
    payload: ProfileGroovePayload
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]

    def recipe_dict(self) -> dict[str, Any]:
        return dict(self.recipe)

    def qc_dict(self) -> dict[str, Any]:
        return dict(self.qc)


def compute_artifact_profile_grooves(
    session: ArtifactSession,
    *,
    height_bin_um: int = DEFAULT_GROOVE_HEIGHT_BIN_UM,
    minimum_depth_um: int = DEFAULT_GROOVE_MINIMUM_DEPTH_UM,
    maximum_width_um: int = DEFAULT_GROOVE_MAXIMUM_WIDTH_UM,
    cancellation_probe: CancellationProbe | None = None,
) -> ProfileGrooveComputation:
    """Read the grooves of an artifact stood on its measured rotation axis."""

    from .artifact_axis_alignment import AXIS_ALIGN_RECIPE_KIND  # noqa: PLC0415

    if not isinstance(session, ArtifactSession):
        raise ArtifactProfileGrooveError("session must be an ArtifactSession")
    align_id = session.document.active_align_revision_id
    align = (
        session.document.align_revision_index.get(align_id)
        if isinstance(align_id, str)
        else None
    )
    if align is None or align.recipe.get("kind") != AXIS_ALIGN_RECIPE_KIND:
        raise ArtifactProfileGrooveError(
            "a groove that runs right round the artifact only means something "
            "about its rotation axis; the active Align was not made from one"
        )
    try:
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactProfileGrooveError(str(exc)) from exc
    recipe = profile_groove_recipe(
        height_bin_um=height_bin_um,
        minimum_depth_um=minimum_depth_um,
        maximum_width_um=maximum_width_um,
    )
    payload = detect_profile_grooves(
        projection.mesh.vertices,
        projection.mesh.faces,
        recipe,
        cancellation_probe=cancellation_probe,
    )
    try:
        context = session.capture_operation(
            recipe=recipe,
            selection_hash=payload.sha256,
        )
    except ArtifactSessionError as exc:
        raise ArtifactProfileGrooveError(str(exc)) from exc
    return ProfileGrooveComputation(
        context=context,
        projection_snapshot=projection.snapshot,
        payload=payload,
        recipe=recipe,
        qc=payload.qc_summary(),
    )


def profile_groove_computation_matches_active_projection(
    session: ArtifactSession,
    computation: ProfileGrooveComputation,
) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(
        computation, ProfileGrooveComputation
    ):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    return current.render_key == computation.projection_snapshot.render_key


def append_profile_groove_record_from_context(
    document: ArtifactDocument,
    *,
    context: OperationContext,
    payload: ProfileGroovePayload,
    recipe: Mapping[str, Any],
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactDocument:
    """Append one verified groove reading without touching source geometry."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactProfileGrooveError("document must be an ArtifactDocument")
    if not isinstance(context, OperationContext):
        raise ArtifactProfileGrooveError("context must be an OperationContext")
    if not isinstance(payload, ProfileGroovePayload):
        raise ArtifactProfileGrooveError("payload must be a ProfileGroovePayload")
    validated_recipe = validate_profile_groove_recipe(recipe)
    if context.selection_hash != payload.sha256:
        raise ArtifactProfileGrooveError(
            "profile groove context selection_hash does not match the reading"
        )
    payload_bytes = payload.canonical_json_bytes()
    extensions = {
        PROFILE_GROOVE_PAYLOAD_EXTENSION_KEY: {
            "byte_length": len(payload_bytes),
            "media_type": PROFILE_GROOVE_PAYLOAD_MEDIA_TYPE,
            "payload": payload.to_dict(),
            "schema_version": PROFILE_GROOVE_PAYLOAD_SCHEMA_VERSION,
            "sha256": payload.sha256,
        }
    }
    try:
        return document.append_record_from_context(
            context=context,
            id=record_id,
            type=PROFILE_GROOVE_RECORD_TYPE,
            geometry_ref=payload.geometry_ref,
            recipe=dict(validated_recipe),
            qc=payload.qc_summary(),
            lifecycle_status=RecordLifecycleStatus.READY,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=depends_on_record_ids,
            extensions=extensions,
        )
    except ArtifactDocumentError as exc:
        raise ArtifactProfileGrooveError(str(exc)) from exc


def commit_profile_grooves(
    session: ArtifactSession,
    computation: ProfileGrooveComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    if not profile_groove_computation_matches_active_projection(session, computation):
        raise ArtifactProfileGrooveError(
            "profile groove computation is stale for the active projection"
        )
    document = append_profile_groove_record_from_context(
        session.document,
        context=computation.context,
        payload=computation.payload,
        recipe=computation.recipe,
        record_id=record_id,
        created_at=created_at,
        operator=operator,
        depends_on_record_ids=depends_on_record_ids,
    )
    return session.with_document(document)


_DESCRIPTOR_KEYS = frozenset(
    {"byte_length", "media_type", "payload", "schema_version", "sha256"}
)


def profile_groove_payload_from_record(
    record: DerivedRecord,
) -> ProfileGroovePayload:
    """Resolve and re-verify one groove record's inline reading."""

    if not isinstance(record, DerivedRecord):
        raise ArtifactProfileGrooveError("record must be a DerivedRecord")
    if record.type != PROFILE_GROOVE_RECORD_TYPE:
        raise ArtifactProfileGrooveError(
            f"record is not a profile groove reading: {record.type!r}"
        )
    descriptor = _exact_keys(
        record.extensions.get(PROFILE_GROOVE_PAYLOAD_EXTENSION_KEY),
        _DESCRIPTOR_KEYS,
        name="profile groove payload descriptor",
    )
    if descriptor["media_type"] != PROFILE_GROOVE_PAYLOAD_MEDIA_TYPE:
        raise ArtifactProfileGrooveError(
            "profile groove payload media_type is invalid"
        )
    if descriptor["schema_version"] != PROFILE_GROOVE_PAYLOAD_SCHEMA_VERSION:
        raise ArtifactProfileGrooveError(
            "profile groove payload descriptor schema is invalid"
        )
    raw_payload = descriptor["payload"]
    if not isinstance(raw_payload, Mapping):
        raise ArtifactProfileGrooveError(
            "profile groove payload descriptor payload must be an object"
        )
    payload = ProfileGroovePayload.from_dict(raw_payload)
    byte_length = descriptor["byte_length"]
    if type(byte_length) is not int or byte_length != len(
        payload.canonical_json_bytes()
    ):
        raise ArtifactProfileGrooveError(
            "profile groove payload byte_length does not match payload"
        )
    if descriptor["sha256"] != payload.sha256:
        raise ArtifactProfileGrooveError(
            "profile groove payload SHA-256 does not match payload"
        )
    if record.geometry_ref != payload.geometry_ref:
        raise ArtifactProfileGrooveError(
            "profile groove record geometry_ref does not match payload"
        )
    validate_profile_groove_recipe(record.recipe)
    return payload


def validate_profile_groove_records(document: ArtifactDocument) -> None:
    """Strictly validate every groove reading embedded in a document."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactProfileGrooveError("document must be an ArtifactDocument")
    for record in document.records:
        if record.type == PROFILE_GROOVE_RECORD_TYPE:
            profile_groove_payload_from_record(record)


__all__ = [
    "ArtifactProfileGrooveError",
    "DEFAULT_GROOVE_HEIGHT_BIN_UM",
    "DEFAULT_GROOVE_MAXIMUM_WIDTH_UM",
    "DEFAULT_GROOVE_MINIMUM_DEPTH_UM",
    "PROFILE_GROOVE_ALGORITHM",
    "PROFILE_GROOVE_ALGORITHM_VERSION",
    "PROFILE_GROOVE_OPERATION_KIND",
    "PROFILE_GROOVE_PAYLOAD_EXTENSION_KEY",
    "PROFILE_GROOVE_RECORD_TYPE",
    "ProfileGroove",
    "ProfileGrooveComputation",
    "ProfileGroovePayload",
    "append_profile_groove_record_from_context",
    "commit_profile_grooves",
    "compute_artifact_profile_grooves",
    "detect_profile_grooves",
    "profile_groove_computation_matches_active_projection",
    "profile_groove_payload_from_record",
    "profile_groove_recipe",
    "validate_profile_groove_recipe",
    "validate_profile_groove_records",
]
