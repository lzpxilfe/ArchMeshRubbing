"""Where the wall of a vessel of revolution turns a corner: the profile's
breaks, read once and drawn as the lines that run right round.

A wheel-thrown pot's outside is a profile r(z) turned about the axis.  Where
that profile bends sharply - the shoulder's edge, the foot ring's root, the
edge where the foot stands proud of the body - the bend runs round the whole
pot, and a measured drawing draws it on the elevation as one straight
horizontal line from the silhouette in to the axis: the circle seen
edge-on.  It is not a groove (nothing is cut) and not the outline (nothing
ends there); it is an inner line, and it belongs to the artifact's axis, not
to a view.

A break is where the profile's direction, taken over ``span_um`` on either
side, turns by ``angle_min_deg`` or more; the reading keeps the sharpest
point of each bend and reports its height, the radius there and the signed
turn - positive where the corner stands proud of the surface read (a convex
edge), negative where it is a root the surface folds into (concave).  The
rim and the base are the profile's ends and are not breaks.

The inside has a profile too, and a reading names which it takes: the
outward-facing surface, drawn on the elevation from the silhouette in to
the axis; or the inward-facing one - the inner wall where the floor meets
it, a ledge inside a lid's seating, the fold inside a shoulder - which shows
through the cut on the section's side, from the axis out to the inner wall.
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
from .artifact_profile_groove import (
    GROOVE_MINIMUM_BIN_SAMPLE_COUNT,
    MAX_GROOVE_HEIGHT_BIN_UM,
    MAX_GROOVE_PROFILE_BINS,
    MIN_GROOVE_HEIGHT_BIN_UM,
    PROFILE_GROOVE_COORDINATE_SPACE,
)
from .artifact_session import ArtifactSession, ArtifactSessionError
from .canonical_json import CanonicalJSONError, canonical_json_bytes, canonical_json_sha256

PROFILE_BREAK_RECORD_TYPE = "measurement.profile_break.v1"
PROFILE_BREAK_OPERATION_KIND = "profile_break"
PROFILE_BREAK_ALGORITHM = "archmeshrubbing.axial_profile_break"
PROFILE_BREAK_ALGORITHM_VERSION = "1.0.0"
PROFILE_BREAK_PAYLOAD_SCHEMA_VERSION = "1.0.0"
PROFILE_BREAK_PAYLOAD_EXTENSION_KEY = "org.archmeshrubbing:profile-break-v1"
PROFILE_BREAK_PAYLOAD_MEDIA_TYPE = "application/vnd.archmeshrubbing.profile-break+json"
PROFILE_BREAK_GEOMETRY_REF_PREFIX = "urn:archmeshrubbing:profile-break:sha256:"

DEFAULT_BREAK_HEIGHT_BIN_UM = 250
DEFAULT_BREAK_ANGLE_MIN_DEG = 25
DEFAULT_BREAK_SPAN_UM = 1_500
MIN_BREAK_ANGLE_DEG = 5
MAX_BREAK_ANGLE_DEG = 170
MIN_BREAK_SPAN_UM = 100
MAX_BREAK_SPAN_UM = 100_000
MAX_BREAK_COUNT = 512

#: The surfaces a profile can be read up: every face whose normal leaves
#: the axis (the outside, seen on the elevation) or every face whose normal
#: points at it (the inner wall, seen through the cut on the section's side).
PROFILE_BREAK_SURFACE_OUTWARD = "outward_facing_every_shell/v1"
PROFILE_BREAK_SURFACE_INWARD = "inward_facing_every_shell/v1"
PROFILE_BREAK_SURFACES = (PROFILE_BREAK_SURFACE_OUTWARD, PROFILE_BREAK_SURFACE_INWARD)
#: The grazing angle: a face counts for a surface when its normal leans that
#: way by more than this cosine, so a floor, a rim's top or an underside
#: belongs to neither.
PROFILE_BREAK_FACING_COS = 0.25


class ArtifactProfileBreakError(ValueError):
    pass


def _strict_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactProfileBreakError(f"{name} must be an integer")
    number = int(value)
    if not minimum <= number <= maximum:
        raise ArtifactProfileBreakError(
            f"{name} must be in the inclusive range {minimum}..{maximum}"
        )
    return number


def _exact_keys(value: object, keys: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactProfileBreakError(f"{name} must be an object")
    if set(value) != keys:
        raise ArtifactProfileBreakError(f"{name} must carry exactly {', '.join(sorted(keys))}")
    return value


def profile_break_recipe(
    *,
    height_bin_um: int = DEFAULT_BREAK_HEIGHT_BIN_UM,
    angle_min_deg: int = DEFAULT_BREAK_ANGLE_MIN_DEG,
    span_um: int = DEFAULT_BREAK_SPAN_UM,
    surface: str = PROFILE_BREAK_SURFACE_OUTWARD,
) -> dict[str, Any]:
    """Resolve the three numbers that decide what counts as a break, and
    which surface the profile is read up."""

    if surface not in PROFILE_BREAK_SURFACES:
        raise ArtifactProfileBreakError(
            f"surface must be one of {', '.join(PROFILE_BREAK_SURFACES)}; got {surface!r}"
        )
    bin_um = _strict_int(
        height_bin_um, name="height_bin_um", minimum=MIN_GROOVE_HEIGHT_BIN_UM, maximum=MAX_GROOVE_HEIGHT_BIN_UM
    )
    angle = _strict_int(angle_min_deg, name="angle_min_deg", minimum=MIN_BREAK_ANGLE_DEG, maximum=MAX_BREAK_ANGLE_DEG)
    span = _strict_int(span_um, name="span_um", minimum=MIN_BREAK_SPAN_UM, maximum=MAX_BREAK_SPAN_UM)
    if span < 2 * bin_um:
        raise ArtifactProfileBreakError(
            "span_um must reach at least two height bins on either side, or a "
            "direction cannot be taken; widen it or use a finer height_bin_um"
        )
    return {
        "algorithm": PROFILE_BREAK_ALGORITHM,
        "algorithm_version": PROFILE_BREAK_ALGORITHM_VERSION,
        "coordinate_space": PROFILE_GROOVE_COORDINATE_SPACE,
        "detection_policy": {
            "angle_min_deg": angle,
            "direction": "chord_over_span_either_side/v1",
            "keep": "sharpest_within_span/v1",
            "span_um": span,
        },
        "kind": PROFILE_BREAK_OPERATION_KIND,
        "longitudinal_axis": "z",
        "profile_policy": {
            "height_bin_um": bin_um,
            "minimum_bin_sample_count": GROOVE_MINIMUM_BIN_SAMPLE_COUNT,
            "radius_statistic": "median_across_revolution/v1",
            "surface": surface,
        },
        "resource_limits": {
            "max_break_count": MAX_BREAK_COUNT,
            "max_profile_bins": MAX_GROOVE_PROFILE_BINS,
        },
    }


def validate_profile_break_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the recipe from its own numbers and require the same bytes."""

    if not isinstance(recipe, Mapping):
        raise ArtifactProfileBreakError("profile break recipe must be an object")
    profile_policy = recipe.get("profile_policy")
    detection_policy = recipe.get("detection_policy")
    if not isinstance(profile_policy, Mapping) or not isinstance(detection_policy, Mapping):
        raise ArtifactProfileBreakError("profile break recipe policies are invalid")
    expected = profile_break_recipe(
        height_bin_um=profile_policy.get("height_bin_um"),  # type: ignore[arg-type]
        angle_min_deg=detection_policy.get("angle_min_deg"),  # type: ignore[arg-type]
        span_um=detection_policy.get("span_um"),  # type: ignore[arg-type]
        surface=profile_policy.get("surface"),  # type: ignore[arg-type]
    )
    try:
        same = canonical_json_bytes(dict(recipe)) == canonical_json_bytes(expected)
    except CanonicalJSONError as exc:
        raise ArtifactProfileBreakError(str(exc)) from exc
    if not same:
        raise ArtifactProfileBreakError("profile break recipe does not match the production contract")
    return expected


def profile_break_surface(recipe: Mapping[str, Any]) -> str:
    """The surface a validated break recipe reads: outward or inward."""

    return str(validate_profile_break_recipe(recipe)["profile_policy"]["surface"])


@dataclass(frozen=True, slots=True)
class ProfileBreak:
    """One corner of the profile: where it is, how far from the axis, and
    which way and how far the wall turns there.  ``turn_millidegrees`` is
    positive where the corner stands proud of the surface the reading took
    (convex, an edge), negative where the surface folds in (a root)."""

    height_um: int
    radius_um: int
    turn_millidegrees: int
    revolution_spread_um: int

    def __post_init__(self) -> None:
        if self.radius_um <= 0:
            raise ArtifactProfileBreakError("a break lies at a positive radius")
        if self.turn_millidegrees == 0:
            raise ArtifactProfileBreakError("a break turns one way or the other")
        if self.revolution_spread_um < 0:
            raise ArtifactProfileBreakError("a break's spread across the revolution cannot be negative")

    @property
    def convex(self) -> bool:
        return self.turn_millidegrees > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "height_um": self.height_um,
            "radius_um": self.radius_um,
            "revolution_spread_um": self.revolution_spread_um,
            "turn_millidegrees": self.turn_millidegrees,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProfileBreak":
        block = _exact_keys(
            data,
            frozenset({"height_um", "radius_um", "revolution_spread_um", "turn_millidegrees"}),
            name="profile break",
        )
        limit = 10**12
        return cls(
            height_um=_strict_int(block["height_um"], name="height_um", minimum=-limit, maximum=limit),
            radius_um=_strict_int(block["radius_um"], name="radius_um", minimum=-limit, maximum=limit),
            turn_millidegrees=_strict_int(
                block["turn_millidegrees"], name="turn_millidegrees", minimum=-180_000, maximum=180_000
            ),
            revolution_spread_um=_strict_int(
                block["revolution_spread_um"], name="revolution_spread_um", minimum=-limit, maximum=limit
            ),
        )


@dataclass(frozen=True, slots=True)
class ProfileBreakPayload:
    """Every break one reading found, ordered up the artifact."""

    schema_version: str
    breaks: tuple[ProfileBreak, ...]
    profile_bin_count: int
    profile_minimum_height_um: int
    profile_maximum_height_um: int

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_BREAK_PAYLOAD_SCHEMA_VERSION:
            raise ArtifactProfileBreakError(
                f"unsupported profile break payload schema: {self.schema_version!r}"
            )
        breaks = tuple(self.breaks)
        if any(not isinstance(item, ProfileBreak) for item in breaks):
            raise ArtifactProfileBreakError("profile break payload holds ProfileBreak values")
        if not breaks:
            raise ArtifactProfileBreakError(
                "a break reading with no break records nothing; lower angle_min_deg, "
                "or do not take the reading"
            )
        if len(breaks) > MAX_BREAK_COUNT:
            raise ArtifactProfileBreakError(f"a break reading holds at most {MAX_BREAK_COUNT} breaks")
        ordered = tuple(sorted(breaks, key=lambda item: item.height_um))
        heights = [item.height_um for item in ordered]
        if len(set(heights)) != len(heights):
            raise ArtifactProfileBreakError("two breaks cannot share one height")
        object.__setattr__(self, "breaks", ordered)
        if self.profile_minimum_height_um >= self.profile_maximum_height_um:
            raise ArtifactProfileBreakError("the profile's height range must be non-empty")
        if self.profile_bin_count <= 0:
            raise ArtifactProfileBreakError("the profile must hold at least one bin")

    def to_dict(self) -> dict[str, Any]:
        return {
            "breaks": [item.to_dict() for item in self.breaks],
            "profile_bin_count": self.profile_bin_count,
            "profile_maximum_height_um": self.profile_maximum_height_um,
            "profile_minimum_height_um": self.profile_minimum_height_um,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProfileBreakPayload":
        block = _exact_keys(
            data,
            frozenset(
                {
                    "breaks",
                    "profile_bin_count",
                    "profile_maximum_height_um",
                    "profile_minimum_height_um",
                    "schema_version",
                }
            ),
            name="profile break payload",
        )
        raw = block["breaks"]
        if not isinstance(raw, (list, tuple)):
            raise ArtifactProfileBreakError("profile break payload breaks must be an array")
        schema_version = block["schema_version"]
        limit = 10**12
        return cls(
            schema_version=schema_version if isinstance(schema_version, str) else "",
            breaks=tuple(ProfileBreak.from_dict(entry) for entry in raw),  # type: ignore[arg-type]
            profile_bin_count=_strict_int(
                block["profile_bin_count"], name="profile_bin_count", minimum=0, maximum=MAX_GROOVE_PROFILE_BINS
            ),
            profile_minimum_height_um=_strict_int(
                block["profile_minimum_height_um"], name="profile_minimum_height_um", minimum=-limit, maximum=limit
            ),
            profile_maximum_height_um=_strict_int(
                block["profile_maximum_height_um"], name="profile_maximum_height_um", minimum=-limit, maximum=limit
            ),
        )

    def canonical_json_bytes(self) -> bytes:
        try:
            return canonical_json_bytes(self.to_dict())
        except CanonicalJSONError as exc:
            raise ArtifactProfileBreakError(str(exc)) from exc

    @property
    def sha256(self) -> str:
        try:
            return canonical_json_sha256(self.to_dict())
        except CanonicalJSONError as exc:
            raise ArtifactProfileBreakError(str(exc)) from exc

    @property
    def geometry_ref(self) -> str:
        return f"{PROFILE_BREAK_GEOMETRY_REF_PREFIX}{self.sha256}"

    def qc_summary(self) -> dict[str, Any]:
        turns = [item.turn_millidegrees for item in self.breaks]
        return {
            "break_count": len(self.breaks),
            "break_heights_um": [item.height_um for item in self.breaks],
            "concave_break_count": sum(1 for turn in turns if turn < 0),
            "convex_break_count": sum(1 for turn in turns if turn > 0),
            "maximum_revolution_spread_um": max(item.revolution_spread_um for item in self.breaks),
            "maximum_turn_millidegrees": max(abs(turn) for turn in turns),
            "payload_sha256": self.sha256,
            "profile_bin_count": self.profile_bin_count,
            "profile_maximum_height_um": self.profile_maximum_height_um,
            "profile_minimum_height_um": self.profile_minimum_height_um,
        }


def _facing_profile(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    height_bin_um: int,
    inward: bool,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (height mm, median radius mm, spread mm) up one surface of the
    artifact: every face whose normal leaves the axis (the outside) or, for
    ``inward``, every face whose normal points at it (the inner wall), of
    every shell.  A foot ring is often its own shell, and the corner where
    it meets the body is exactly a break, so the profile must not stop at a
    shell's edge; the rim's top, the floor and the underside face neither
    way and reach neither profile."""

    raise_if_cancelled(cancellation_probe)
    side = "inside" if inward else "outside"
    corners = vertices[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    centroids = corners.mean(axis=1)
    radial = np.einsum("ij,ij->i", normals[:, :2], centroids[:, :2])
    lengths = np.linalg.norm(normals, axis=1) * np.maximum(np.hypot(centroids[:, 0], centroids[:, 1]), 1e-12)
    # By more than the grazing angle: a face standing near enough upright,
    # or leaning, that its normal points the way asked.
    if inward:
        facing = np.flatnonzero(radial < -PROFILE_BREAK_FACING_COS * lengths)
    else:
        facing = np.flatnonzero(radial > PROFILE_BREAK_FACING_COS * lengths)
    if facing.size == 0:
        raise ArtifactProfileBreakError(
            f"no face of the mesh faces {'towards' if inward else 'away from'} the axis; nothing to read"
        )
    used = np.unique(faces[facing].reshape(-1))
    points = vertices[used]
    heights = points[:, 2]
    radii = np.hypot(points[:, 0], points[:, 1])
    lowest, highest = float(heights.min()), float(heights.max())
    bin_mm = float(height_bin_um) / 1000.0
    span = highest - lowest
    if span <= 0.0:
        raise ArtifactProfileBreakError(f"the {side} has no height to read a profile along")
    count = int(math.ceil(span / bin_mm))
    if count > MAX_GROOVE_PROFILE_BINS:
        raise ArtifactProfileBreakError(
            f"a {span:.1f} mm wall at {bin_mm:.3f} mm bins needs {count} bins, past the "
            f"{MAX_GROOVE_PROFILE_BINS} safety limit; use a coarser height_bin_um"
        )
    edges = lowest + bin_mm * np.arange(count + 1, dtype=np.float64)
    index = np.clip(np.searchsorted(edges, heights, side="right") - 1, 0, count - 1)
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
        raise ArtifactProfileBreakError(
            f"the {side} gave too few height bins to read a profile; use a coarser "
            "height_bin_um or a denser mesh"
        )
    return (
        np.asarray(centres, dtype=np.float64),
        np.asarray(medians, dtype=np.float64),
        np.asarray(spreads, dtype=np.float64),
    )


def detect_profile_breaks(
    vertices: object,
    faces: object,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> ProfileBreakPayload:
    """Find the corners of one wall's profile under a validated recipe."""

    validated = validate_profile_break_recipe(recipe)
    bin_um = int(validated["profile_policy"]["height_bin_um"])
    inward = validated["profile_policy"]["surface"] == PROFILE_BREAK_SURFACE_INWARD
    angle_min = math.radians(float(validated["detection_policy"]["angle_min_deg"]))
    span_mm = float(validated["detection_policy"]["span_um"]) / 1000.0
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ArtifactProfileBreakError("mesh must be (n, 3) vertices and (m, 3) faces")
    heights, radii, spreads = _facing_profile(
        points, triangles, height_bin_um=bin_um, inward=inward, cancellation_probe=cancellation_probe
    )
    # A left turn going up the (r, z) half-plane is a corner standing proud
    # of the outside; the inside has its material on the other hand, so the
    # same turn is a root there.  The sign is kept relative to the surface.
    proud = -1.0 if inward else 1.0
    raise_if_cancelled(cancellation_probe)
    count = int(heights.size)
    # The direction the wall runs, below and above each sample: the chord
    # from the sample a span below to here, and from here to the sample a
    # span above.  Both chords are asked to reach the full span, so a
    # sample near an end of the profile has no direction and is no break.
    turns = np.zeros(count, dtype=np.float64)
    has_turn = np.zeros(count, dtype=bool)
    for index in range(count):
        below = np.flatnonzero(heights <= heights[index] - span_mm)
        above = np.flatnonzero(heights >= heights[index] + span_mm)
        if below.size == 0 or above.size == 0:
            continue
        low = int(below[-1])
        high = int(above[0])
        before = np.array([radii[index] - radii[low], heights[index] - heights[low]])
        after = np.array([radii[high] - radii[index], heights[high] - heights[index]])
        if float(np.linalg.norm(before)) <= 0.0 or float(np.linalg.norm(after)) <= 0.0:
            continue
        turns[index] = proud * math.atan2(
            float(before[0] * after[1] - before[1] * after[0]), float(before @ after)
        )
        has_turn[index] = True
    raise_if_cancelled(cancellation_probe)
    sharp = has_turn & (np.abs(turns) >= angle_min)
    breaks: list[ProfileBreak] = []
    for index in np.flatnonzero(sharp):
        # The sharpest sample of a bend: no sharper sample within a span.
        window = (np.abs(heights - heights[index]) <= span_mm) & has_turn
        if np.abs(turns[index]) < float(np.abs(turns[window]).max()):
            continue
        if any(abs(item.height_um - int(round(heights[index] * 1000.0))) <= int(round(span_mm * 1000.0)) for item in breaks):
            continue
        breaks.append(
            ProfileBreak(
                height_um=int(round(float(heights[index]) * 1000.0)),
                radius_um=int(round(float(radii[index]) * 1000.0)),
                turn_millidegrees=int(round(math.degrees(float(turns[index])) * 1000.0)),
                revolution_spread_um=int(round(float(spreads[index]) * 1000.0)),
            )
        )
    return ProfileBreakPayload(
        schema_version=PROFILE_BREAK_PAYLOAD_SCHEMA_VERSION,
        breaks=tuple(breaks),
        profile_bin_count=count,
        profile_minimum_height_um=int(round(float(heights.min()) * 1000.0)),
        profile_maximum_height_um=int(round(float(heights.max()) * 1000.0)),
    )


@dataclass(frozen=True, slots=True)
class ProfileBreakComputation:
    context: OperationContext
    projection_snapshot: Any
    payload: ProfileBreakPayload
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]

    def recipe_dict(self) -> dict[str, Any]:
        return dict(self.recipe)

    def qc_dict(self) -> dict[str, Any]:
        return dict(self.qc)


def compute_artifact_profile_breaks(
    session: ArtifactSession,
    *,
    height_bin_um: int = DEFAULT_BREAK_HEIGHT_BIN_UM,
    angle_min_deg: int = DEFAULT_BREAK_ANGLE_MIN_DEG,
    span_um: int = DEFAULT_BREAK_SPAN_UM,
    surface: str = PROFILE_BREAK_SURFACE_OUTWARD,
    cancellation_probe: CancellationProbe | None = None,
) -> ProfileBreakComputation:
    """Read the corners of one surface's profile on an artifact stood on its
    rotation axis."""

    from .artifact_axis_alignment import AXIS_ALIGN_RECIPE_KIND  # noqa: PLC0415

    if not isinstance(session, ArtifactSession):
        raise ArtifactProfileBreakError("session must be an ArtifactSession")
    align_id = session.document.active_align_revision_id
    align = session.document.align_revision_index.get(align_id) if isinstance(align_id, str) else None
    if align is None or align.recipe.get("kind") != AXIS_ALIGN_RECIPE_KIND:
        raise ArtifactProfileBreakError(
            "a corner that runs right round the artifact only means something "
            "about its rotation axis; the active Align was not made from one"
        )
    try:
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactProfileBreakError(str(exc)) from exc
    recipe = profile_break_recipe(
        height_bin_um=height_bin_um, angle_min_deg=angle_min_deg, span_um=span_um, surface=surface
    )
    payload = detect_profile_breaks(
        projection.mesh.vertices, projection.mesh.faces, recipe, cancellation_probe=cancellation_probe
    )
    try:
        context = session.capture_operation(recipe=recipe, selection_hash=payload.sha256)
    except ArtifactSessionError as exc:
        raise ArtifactProfileBreakError(str(exc)) from exc
    return ProfileBreakComputation(
        context=context,
        projection_snapshot=projection.snapshot,
        payload=payload,
        recipe=recipe,
        qc=payload.qc_summary(),
    )


def profile_break_computation_matches_active_projection(
    session: ArtifactSession, computation: ProfileBreakComputation
) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(computation, ProfileBreakComputation):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    return current.render_key == computation.projection_snapshot.render_key


def append_profile_break_record_from_context(
    document: ArtifactDocument,
    *,
    context: OperationContext,
    payload: ProfileBreakPayload,
    recipe: Mapping[str, Any],
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactDocument:
    """Append one verified break reading without touching source geometry."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactProfileBreakError("document must be an ArtifactDocument")
    if not isinstance(context, OperationContext):
        raise ArtifactProfileBreakError("context must be an OperationContext")
    if not isinstance(payload, ProfileBreakPayload):
        raise ArtifactProfileBreakError("payload must be a ProfileBreakPayload")
    validated_recipe = validate_profile_break_recipe(recipe)
    if context.selection_hash != payload.sha256:
        raise ArtifactProfileBreakError("profile break context selection_hash does not match the reading")
    payload_bytes = payload.canonical_json_bytes()
    extensions = {
        PROFILE_BREAK_PAYLOAD_EXTENSION_KEY: {
            "byte_length": len(payload_bytes),
            "media_type": PROFILE_BREAK_PAYLOAD_MEDIA_TYPE,
            "payload": payload.to_dict(),
            "schema_version": PROFILE_BREAK_PAYLOAD_SCHEMA_VERSION,
            "sha256": payload.sha256,
        }
    }
    try:
        return document.append_record_from_context(
            context=context,
            id=record_id,
            type=PROFILE_BREAK_RECORD_TYPE,
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
        raise ArtifactProfileBreakError(str(exc)) from exc


def commit_profile_breaks(
    session: ArtifactSession,
    computation: ProfileBreakComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    if not profile_break_computation_matches_active_projection(session, computation):
        raise ArtifactProfileBreakError("profile break computation is stale for the active projection")
    document = append_profile_break_record_from_context(
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


_DESCRIPTOR_KEYS = frozenset({"byte_length", "media_type", "payload", "schema_version", "sha256"})


def profile_break_payload_from_record(record: DerivedRecord) -> ProfileBreakPayload:
    """Resolve and re-verify one break record's inline reading."""

    if not isinstance(record, DerivedRecord):
        raise ArtifactProfileBreakError("record must be a DerivedRecord")
    if record.type != PROFILE_BREAK_RECORD_TYPE:
        raise ArtifactProfileBreakError(f"record is not a profile break reading: {record.type!r}")
    descriptor = _exact_keys(
        record.extensions.get(PROFILE_BREAK_PAYLOAD_EXTENSION_KEY),
        _DESCRIPTOR_KEYS,
        name="profile break payload descriptor",
    )
    if descriptor["media_type"] != PROFILE_BREAK_PAYLOAD_MEDIA_TYPE:
        raise ArtifactProfileBreakError("profile break payload media_type is invalid")
    if descriptor["schema_version"] != PROFILE_BREAK_PAYLOAD_SCHEMA_VERSION:
        raise ArtifactProfileBreakError("profile break payload descriptor schema is invalid")
    raw_payload = descriptor["payload"]
    if not isinstance(raw_payload, Mapping):
        raise ArtifactProfileBreakError("profile break payload descriptor payload must be an object")
    payload = ProfileBreakPayload.from_dict(raw_payload)
    byte_length = descriptor["byte_length"]
    if type(byte_length) is not int or byte_length != len(payload.canonical_json_bytes()):
        raise ArtifactProfileBreakError("profile break payload byte_length does not match payload")
    if descriptor["sha256"] != payload.sha256:
        raise ArtifactProfileBreakError("profile break payload SHA-256 does not match payload")
    if record.geometry_ref != payload.geometry_ref:
        raise ArtifactProfileBreakError("profile break record geometry_ref does not match payload")
    validate_profile_break_recipe(record.recipe)
    return payload


def validate_profile_break_records(document: ArtifactDocument) -> None:
    """Strictly validate every break reading embedded in a document."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactProfileBreakError("document must be an ArtifactDocument")
    for record in document.records:
        if record.type == PROFILE_BREAK_RECORD_TYPE:
            profile_break_payload_from_record(record)


__all__ = [
    "ArtifactProfileBreakError",
    "DEFAULT_BREAK_ANGLE_MIN_DEG",
    "DEFAULT_BREAK_HEIGHT_BIN_UM",
    "DEFAULT_BREAK_SPAN_UM",
    "PROFILE_BREAK_ALGORITHM",
    "PROFILE_BREAK_PAYLOAD_EXTENSION_KEY",
    "PROFILE_BREAK_PAYLOAD_SCHEMA_VERSION",
    "PROFILE_BREAK_RECORD_TYPE",
    "PROFILE_BREAK_SURFACES",
    "PROFILE_BREAK_SURFACE_INWARD",
    "PROFILE_BREAK_SURFACE_OUTWARD",
    "ProfileBreak",
    "ProfileBreakComputation",
    "ProfileBreakPayload",
    "append_profile_break_record_from_context",
    "commit_profile_breaks",
    "compute_artifact_profile_breaks",
    "detect_profile_breaks",
    "profile_break_computation_matches_active_projection",
    "profile_break_payload_from_record",
    "profile_break_recipe",
    "profile_break_surface",
    "validate_profile_break_recipe",
    "validate_profile_break_records",
]
