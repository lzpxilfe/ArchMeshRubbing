"""Durable condition annotations: loss, restoration, cracking, and wear.

An archaeological drawing has to say which part of the object is the object and
which part is a modern repair.  Without that, a reader cannot tell what the
measurements they are looking at are measurements *of*.

The annotated region is a set of triangle faces, not a polygon drawn on one
picture.  A face set belongs to the artifact: it reprojects into every view, it
survives reopening, and it stays true when the alignment changes.  A polygon
drawn on an elevation belongs to that elevation only, and says nothing about
the same damage seen from above.

The face set is stored as canonical `[start, end_exclusive]` ranges - sorted,
disjoint, and maximally merged - so one region has exactly one encoding and
therefore exactly one hash.  This is the same shape the recording-surface
selection uses in `artifact_tile_unwrap_extractor`, deliberately, so a reader
that can parse one can parse the other.  It is a separate implementation
because a condition region must not inherit a tile-unwrapping limit, and
because the ranges are durable *here*: the tile unwrap keeps only the digest.

Every view's boundary is computed once, at commit time, and kept in the one
record.  The alternative - projecting while drawing - would put a Shapely
computation inside the presentation layer, and the alternative to that - one
record per view - would split a single act of authorship into six receipts.
"""

from __future__ import annotations

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
from .artifact_outline_extractor import (
    DEFAULT_OUTLINE_PRECISION_GRID_MM,
    OUTLINE_LEGACY_ALGORITHM_VERSION,
    OutlineView,
    extract_outline_geometry,
    outline_frame,
)
from .artifact_outline_topology import (
    ArtifactOutlineTopologyError,
    validate_outline_topology,
)
from .artifact_scene_adapter import ArtifactProjectionSnapshot
from .artifact_session import ArtifactSession, ArtifactSessionError
from .artifact_vector_record import (
    ArtifactVectorRecordError,
    VectorGeometryPayload,
    VectorRecordKind,
)
from .canonical_json import (
    CanonicalJSONError,
    canonical_json_bytes,
    canonical_json_sha256,
)


CONDITION_RECORD_TYPE = "annotation.condition.v1"
#: The recipe's `kind`, which names the operation the way every other recipe
#: does (`cutline`, `outline`, `tile_unwrap`, ...).  The state of the object -
#: missing, restored, crack, worn - is the recipe's `condition`.
CONDITION_OPERATION_KIND = "condition_annotation"
CONDITION_ALGORITHM = "archmeshrubbing.condition_region_projection"
CONDITION_ALGORITHM_VERSION = "1.0.0"
CONDITION_PAYLOAD_SCHEMA_VERSION = "1.0.0"
CONDITION_SELECTION_SCHEMA_VERSION = "1.0.0"
CONDITION_PAYLOAD_EXTENSION_KEY = "org.archmeshrubbing:condition-annotation-v1"
CONDITION_PAYLOAD_MEDIA_TYPE = (
    "application/vnd.archmeshrubbing.condition-annotation+json"
)
CONDITION_GEOMETRY_REF_PREFIX = "urn:archmeshrubbing:condition-annotation:sha256:"
CONDITION_SELECTION_KIND = "canonical_face_ranges"

CONDITION_MISSING = "missing"
CONDITION_RESTORED = "restored"
CONDITION_CRACK = "crack"
CONDITION_WORN = "worn"
#: Closed vocabulary, in canonical (sorted) order.
CONDITION_KINDS: tuple[str, ...] = (
    CONDITION_CRACK,
    CONDITION_MISSING,
    CONDITION_RESTORED,
    CONDITION_WORN,
)
#: The six orthographic views a region is projected into, in canonical order.
CONDITION_VIEWS: tuple[str, ...] = tuple(
    sorted(view.value for view in OutlineView)
)

#: Why a view carries no boundary.  Closed, and recorded per view in the
#: payload, so a record states for all six views either a boundary or the
#: reason there is none - never merely "the rest of them".
CONDITION_SKIP_NO_AREA = "no_projected_area"
CONDITION_SKIP_DEGENERATE = "degenerate_projection"
CONDITION_SKIP_REASONS: tuple[str, ...] = (
    CONDITION_SKIP_DEGENERATE,
    CONDITION_SKIP_NO_AREA,
)

MAX_CONDITION_FACES = 250_000
# A set of N faces can never need more than N ranges, so the two limits are the
# same number.  The range limit still has to exist on its own: an untrusted
# document is checked for length before anything is decoded from it.  Run-length
# is not unconditionally smaller - a region that alternates face by face encodes
# larger than the plain index list would - and this is where that ends.
MAX_CONDITION_FACE_RANGES = MAX_CONDITION_FACES
MAX_CONDITION_TOTAL_FACES = 2_000_000
MAX_CONDITION_PAYLOAD_BYTES = 16 * 1024 * 1024


class ArtifactConditionAnnotationError(ValueError):
    """A condition region, its projection, or its durable record is invalid."""


def _exact_keys(
    value: object,
    expected: frozenset[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactConditionAnnotationError(f"{name} must be an object")
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise ArtifactConditionAnnotationError(
            f"{name} is missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise ArtifactConditionAnnotationError(
            f"{name} has unknown fields: {', '.join(unknown)}"
        )
    return value


def _strict_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactConditionAnnotationError(f"{name} must be an integer")
    number = int(value)
    if number < minimum or number > maximum:
        raise ArtifactConditionAnnotationError(
            f"{name} must be in the inclusive range {minimum}..{maximum}"
        )
    return number


def condition_kind(value: object) -> str:
    if not isinstance(value, str) or value not in CONDITION_KINDS:
        raise ArtifactConditionAnnotationError(
            f"condition kind must be one of {', '.join(CONDITION_KINDS)}"
        )
    return value


def _precision_grid(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ArtifactConditionAnnotationError(
            "precision_grid_mm must be a finite number"
        )
    grid = float(value)
    if not np.isfinite(grid) or grid <= 0.0:
        raise ArtifactConditionAnnotationError(
            "precision_grid_mm must be greater than zero"
        )
    return grid


def face_ranges_from_indices(
    indices: object,
    *,
    total_face_count: int,
) -> tuple[tuple[int, int], ...]:
    """Return the one canonical range encoding of a set of face indices.

    Duplicates collapse and order is irrelevant, because the argument names a
    *set*.  Adjacent runs are merged, so a region cannot be encoded two ways
    and hash two ways.
    """

    total = _strict_int(
        total_face_count,
        name="total_face_count",
        minimum=1,
        maximum=MAX_CONDITION_TOTAL_FACES,
    )
    try:
        values = np.asarray(indices, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ArtifactConditionAnnotationError(
            "condition face indices must be integers"
        ) from exc
    values = np.unique(values)
    if values.size == 0:
        raise ArtifactConditionAnnotationError(
            "a condition region needs at least one face"
        )
    if values.size > MAX_CONDITION_FACES:
        raise ArtifactConditionAnnotationError(
            f"a condition region holds at most {MAX_CONDITION_FACES} faces"
        )
    if int(values[0]) < 0 or int(values[-1]) >= total:
        raise ArtifactConditionAnnotationError(
            "condition face index is outside the geometry"
        )
    boundaries = np.flatnonzero(np.diff(values) != 1)
    starts = np.concatenate(([values[0]], values[boundaries + 1]))
    ends = np.concatenate((values[boundaries], [values[-1]])) + 1
    ranges = tuple(
        (int(start), int(end)) for start, end in zip(starts, ends, strict=True)
    )
    if len(ranges) > MAX_CONDITION_FACE_RANGES:
        raise ArtifactConditionAnnotationError(
            f"a condition region holds at most {MAX_CONDITION_FACE_RANGES} ranges"
        )
    return ranges


def validate_face_ranges(
    value: object,
    *,
    total_face_count: int,
) -> tuple[tuple[int, int], ...]:
    """Accept only the canonical encoding, never merely a decodable one."""

    total = _strict_int(
        total_face_count,
        name="total_face_count",
        minimum=1,
        maximum=MAX_CONDITION_TOTAL_FACES,
    )
    if not isinstance(value, (list, tuple)):
        raise ArtifactConditionAnnotationError("face_ranges must be an array")
    if not value:
        raise ArtifactConditionAnnotationError(
            "a condition region needs at least one face"
        )
    if len(value) > MAX_CONDITION_FACE_RANGES:
        raise ArtifactConditionAnnotationError(
            f"a condition region holds at most {MAX_CONDITION_FACE_RANGES} ranges"
        )
    ranges: list[tuple[int, int]] = []
    selected = 0
    previous_end = -1
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ArtifactConditionAnnotationError(
                f"face_ranges[{index}] must be [start, end_exclusive]"
            )
        start = _strict_int(
            item[0],
            name=f"face_ranges[{index}][0]",
            minimum=0,
            maximum=total - 1,
        )
        end = _strict_int(
            item[1],
            name=f"face_ranges[{index}][1]",
            minimum=1,
            maximum=total,
        )
        if start >= end:
            raise ArtifactConditionAnnotationError("face_ranges must be non-empty")
        # `<=` rather than `<`: two ranges that merely touch are two encodings
        # of one region, and only the merged one is canonical.
        if index > 0 and start <= previous_end:
            raise ArtifactConditionAnnotationError(
                "face_ranges must be sorted, disjoint, and maximally merged"
            )
        ranges.append((start, end))
        selected += end - start
        previous_end = end
    if selected > MAX_CONDITION_FACES:
        raise ArtifactConditionAnnotationError(
            f"a condition region holds at most {MAX_CONDITION_FACES} faces"
        )
    return tuple(ranges)


def face_indices_from_ranges(
    ranges: Sequence[Sequence[int]],
    *,
    total_face_count: int,
) -> np.ndarray:
    """Return the face set a canonical encoding stands for."""

    canonical = validate_face_ranges(ranges, total_face_count=total_face_count)
    count = sum(end - start for start, end in canonical)
    indices = np.empty((count,), dtype=np.int64)
    offset = 0
    for start, end in canonical:
        indices[offset : offset + (end - start)] = np.arange(
            start, end, dtype=np.int64
        )
        offset += end - start
    indices.setflags(write=False)
    return indices


def condition_selection(
    *,
    total_face_count: int,
    face_ranges: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Return the durable face-set block, digest included.

    `selection_sha256` is taken over the decoded index set rather than over the
    encoding, so a tool holding a plain list of face indices can reproduce it
    without implementing the range format.  Here the digest is a checksum, not
    an assertion: unlike the recording-surface selection, the set it covers is
    stored beside it.
    """

    canonical = validate_face_ranges(face_ranges, total_face_count=total_face_count)
    indices = face_indices_from_ranges(canonical, total_face_count=total_face_count)
    digest = canonical_json_sha256(
        {
            "faces": [int(index) for index in indices],
            "kind": CONDITION_SELECTION_KIND,
            "schema_version": CONDITION_SELECTION_SCHEMA_VERSION,
            "total_face_count": int(total_face_count),
        }
    )
    return {
        "face_ranges": [[start, end] for start, end in canonical],
        "kind": CONDITION_SELECTION_KIND,
        "schema_version": CONDITION_SELECTION_SCHEMA_VERSION,
        "selected_face_count": int(indices.size),
        "selection_sha256": digest,
        "total_face_count": int(total_face_count),
    }


_SELECTION_KEYS = frozenset(
    {
        "face_ranges",
        "kind",
        "schema_version",
        "selected_face_count",
        "selection_sha256",
        "total_face_count",
    }
)


def validate_condition_selection(value: object) -> dict[str, Any]:
    selection = _exact_keys(value, _SELECTION_KEYS, name="condition selection")
    if selection["kind"] != CONDITION_SELECTION_KIND:
        raise ArtifactConditionAnnotationError(
            "condition selection kind is unsupported"
        )
    if selection["schema_version"] != CONDITION_SELECTION_SCHEMA_VERSION:
        raise ArtifactConditionAnnotationError(
            "condition selection schema is unsupported"
        )
    canonical = condition_selection(
        total_face_count=_strict_int(
            selection["total_face_count"],
            name="condition selection.total_face_count",
            minimum=1,
            maximum=MAX_CONDITION_TOTAL_FACES,
        ),
        face_ranges=selection["face_ranges"],
    )
    if selection["selected_face_count"] != canonical["selected_face_count"]:
        raise ArtifactConditionAnnotationError(
            "condition selection selected_face_count is inconsistent"
        )
    if selection["selection_sha256"] != canonical["selection_sha256"]:
        raise ArtifactConditionAnnotationError(
            "condition selection SHA-256 is inconsistent"
        )
    return canonical


@dataclass(frozen=True, slots=True)
class ConditionViewBoundary:
    """The region's silhouette in one orthographic view."""

    view: str
    outline: VectorGeometryPayload

    def __post_init__(self) -> None:
        if not isinstance(self.view, str) or self.view not in CONDITION_VIEWS:
            raise ArtifactConditionAnnotationError(
                f"condition view must be one of {', '.join(CONDITION_VIEWS)}"
            )
        if not isinstance(self.outline, VectorGeometryPayload):
            raise ArtifactConditionAnnotationError(
                "condition view boundary must be a VectorGeometryPayload"
            )
        if VectorRecordKind(self.outline.kind) is not VectorRecordKind.OUTLINE:
            raise ArtifactConditionAnnotationError(
                "condition view boundary must be an outline payload"
            )
        if self.outline.frame != outline_frame(self.view):
            raise ArtifactConditionAnnotationError(
                f"condition boundary for the {self.view} view is not in that "
                "view's canonical frame"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"outline": self.outline.to_dict(), "view": self.view}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ConditionViewBoundary":
        block = _exact_keys(data, frozenset({"outline", "view"}), name="condition view")
        raw_outline = block["outline"]
        if not isinstance(raw_outline, Mapping):
            raise ArtifactConditionAnnotationError(
                "condition view outline must be an object"
            )
        try:
            outline = VectorGeometryPayload.from_dict(raw_outline)
        except ArtifactVectorRecordError as exc:
            raise ArtifactConditionAnnotationError(
                f"condition view outline is invalid: {exc}"
            ) from exc
        view = block["view"]
        return cls(view=view if isinstance(view, str) else "", outline=outline)

    def topology(self) -> Mapping[str, Any]:
        try:
            return validate_outline_topology(self.outline).to_dict()
        except ArtifactOutlineTopologyError as exc:
            raise ArtifactConditionAnnotationError(
                f"condition boundary in the {self.view} view is not a valid "
                f"outline topology: {exc}"
            ) from exc

    def qc_summary(self) -> dict[str, Any]:
        topology = self.topology()
        return {
            "area_mm2": topology["area_mm2"],
            "bounds_mm": list(topology["bounds_mm"]),
            "component_count": topology["component_count"],
            "hole_count": topology["hole_count"],
            "point_count": sum(len(path.points_mm) for path in self.outline.paths),
            "view": self.view,
        }


@dataclass(frozen=True, slots=True)
class ConditionAnnotationPayload:
    """One painted region: what it is, which faces it covers, how it projects."""

    schema_version: str
    condition: str
    selection: Mapping[str, Any]
    views: tuple[ConditionViewBoundary, ...]
    skipped_views: tuple[Mapping[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != CONDITION_PAYLOAD_SCHEMA_VERSION:
            raise ArtifactConditionAnnotationError(
                f"unsupported condition payload schema: {self.schema_version!r}"
            )
        object.__setattr__(self, "condition", condition_kind(self.condition))
        object.__setattr__(self, "selection", validate_condition_selection(self.selection))
        views = tuple(self.views)
        if any(not isinstance(view, ConditionViewBoundary) for view in views):
            raise ArtifactConditionAnnotationError(
                "condition views must be ConditionViewBoundary values"
            )
        if not views:
            # A region no view can see leaves nothing on any drawing, so there
            # is nothing to record and no way for a reader to check the claim.
            raise ArtifactConditionAnnotationError(
                "a condition region must project into at least one view"
            )
        names = [view.view for view in views]
        if len(set(names)) != len(names):
            raise ArtifactConditionAnnotationError(
                "a condition region holds at most one boundary per view"
            )
        object.__setattr__(
            self, "views", tuple(sorted(views, key=lambda item: item.view))
        )
        skipped: list[dict[str, str]] = []
        for entry in self.skipped_views:
            block = _exact_keys(
                entry,
                frozenset({"reason", "view"}),
                name="condition skipped view",
            )
            view = block["view"]
            reason = block["reason"]
            if not isinstance(view, str) or view not in CONDITION_VIEWS:
                raise ArtifactConditionAnnotationError(
                    f"condition view must be one of {', '.join(CONDITION_VIEWS)}"
                )
            if not isinstance(reason, str) or reason not in CONDITION_SKIP_REASONS:
                raise ArtifactConditionAnnotationError(
                    "condition skip reason must be one of "
                    f"{', '.join(CONDITION_SKIP_REASONS)}"
                )
            skipped.append({"reason": reason, "view": view})
        # Every view is accounted for exactly once.  "The views not listed" is
        # not a statement a reader can check; this is.
        accounted = names + [entry["view"] for entry in skipped]
        if sorted(accounted) != list(CONDITION_VIEWS):
            raise ArtifactConditionAnnotationError(
                "a condition region must account for each of the six views "
                "exactly once, with a boundary or a reason"
            )
        object.__setattr__(
            self,
            "skipped_views",
            tuple(sorted(skipped, key=lambda item: item["view"])),
        )

    @property
    def face_count(self) -> int:
        return int(self.selection["selected_face_count"])

    @property
    def total_face_count(self) -> int:
        return int(self.selection["total_face_count"])

    @property
    def selection_sha256(self) -> str:
        return str(self.selection["selection_sha256"])

    def face_indices(self) -> np.ndarray:
        return face_indices_from_ranges(
            self.selection["face_ranges"],
            total_face_count=self.total_face_count,
        )

    def boundary_for_view(self, view: str) -> ConditionViewBoundary | None:
        for boundary in self.views:
            if boundary.view == view:
                return boundary
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "schema_version": self.schema_version,
            "selection": dict(self.selection),
            "skipped_views": [dict(entry) for entry in self.skipped_views],
            "views": [view.to_dict() for view in self.views],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ConditionAnnotationPayload":
        block = _exact_keys(
            data,
            frozenset(
                {"condition", "schema_version", "selection", "skipped_views", "views"}
            ),
            name="condition payload",
        )
        raw_views = block["views"]
        raw_skipped = block["skipped_views"]
        if not isinstance(raw_views, (list, tuple)) or not isinstance(
            raw_skipped, (list, tuple)
        ):
            raise ArtifactConditionAnnotationError(
                "condition payload views must be an array"
            )
        schema_version = block["schema_version"]
        condition = block["condition"]
        return cls(
            schema_version=schema_version if isinstance(schema_version, str) else "",
            condition=condition if isinstance(condition, str) else "",
            selection=block["selection"],  # type: ignore[arg-type]
            views=tuple(
                ConditionViewBoundary.from_dict(view)  # type: ignore[arg-type]
                for view in raw_views
            ),
            skipped_views=tuple(raw_skipped),  # type: ignore[arg-type]
        )

    def canonical_json_bytes(self) -> bytes:
        try:
            encoded = canonical_json_bytes(self.to_dict())
        except CanonicalJSONError as exc:
            raise ArtifactConditionAnnotationError(str(exc)) from exc
        if len(encoded) > MAX_CONDITION_PAYLOAD_BYTES:
            raise ArtifactConditionAnnotationError(
                "condition payload exceeds the "
                f"{MAX_CONDITION_PAYLOAD_BYTES}-byte inline safety limit"
            )
        return encoded

    @property
    def sha256(self) -> str:
        try:
            return canonical_json_sha256(self.to_dict())
        except CanonicalJSONError as exc:
            raise ArtifactConditionAnnotationError(str(exc)) from exc

    @property
    def geometry_ref(self) -> str:
        return f"{CONDITION_GEOMETRY_REF_PREFIX}{self.sha256}"

    def qc_summary(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "empty_views": [entry["view"] for entry in self.skipped_views],
            "empty_view_reasons": [dict(entry) for entry in self.skipped_views],
            "face_count": self.face_count,
            "face_range_count": len(self.selection["face_ranges"]),
            "payload_sha256": self.sha256,
            "projected_view_count": len(self.views),
            "selection_sha256": self.selection_sha256,
            "total_face_count": self.total_face_count,
            "views": [view.qc_summary() for view in self.views],
        }


def condition_recipe(
    *,
    condition: str,
    precision_grid_mm: float,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the closed recipe one condition region is computed under.

    The whole selection is part of the recipe, not only its digest, because the
    region *is* the operation: a worker handed nothing but this recipe and the
    geometry can carry it out, and two different regions annotated the same
    way are two different computations that must not share a recipe hash.
    """

    return {
        "algorithm": CONDITION_ALGORITHM,
        "algorithm_version": CONDITION_ALGORITHM_VERSION,
        "condition": condition_kind(condition),
        "kind": CONDITION_OPERATION_KIND,
        "precision_grid_mm": _precision_grid(precision_grid_mm),
        "selection": validate_condition_selection(selection),
        "views": list(CONDITION_VIEWS),
    }


_RECIPE_KEYS = frozenset(
    {
        "algorithm",
        "algorithm_version",
        "condition",
        "kind",
        "precision_grid_mm",
        "selection",
        "views",
    }
)


def validate_condition_recipe(value: object) -> dict[str, Any]:
    recipe = _exact_keys(value, _RECIPE_KEYS, name="condition recipe")
    if recipe["algorithm"] != CONDITION_ALGORITHM:
        raise ArtifactConditionAnnotationError("condition algorithm is unsupported")
    if recipe["algorithm_version"] != CONDITION_ALGORITHM_VERSION:
        raise ArtifactConditionAnnotationError(
            "condition algorithm version is unsupported"
        )
    if recipe["kind"] != CONDITION_OPERATION_KIND:
        raise ArtifactConditionAnnotationError("condition recipe kind is unsupported")
    if list(recipe["views"]) != list(CONDITION_VIEWS):
        raise ArtifactConditionAnnotationError(
            "a condition region is projected into all six views"
        )
    return {
        "algorithm": CONDITION_ALGORITHM,
        "algorithm_version": CONDITION_ALGORITHM_VERSION,
        "condition": condition_kind(recipe["condition"]),
        "kind": CONDITION_OPERATION_KIND,
        "precision_grid_mm": _precision_grid(recipe["precision_grid_mm"]),
        "selection": validate_condition_selection(recipe["selection"]),
        "views": list(CONDITION_VIEWS),
    }


def _twice_area(triangles: np.ndarray) -> np.ndarray:
    return (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
        triangles[:, 2, 1] - triangles[:, 0, 1]
    ) - (triangles[:, 1, 1] - triangles[:, 0, 1]) * (
        triangles[:, 2, 0] - triangles[:, 0, 0]
    )


def _view_leaves_area(
    vertices_world_mm: np.ndarray,
    face_subset: np.ndarray,
    view: str,
    *,
    precision_grid_mm: float,
) -> bool:
    """Answer whether this region has any drawable area in this view.

    A region seen exactly edge-on projects to a line, and a region finer than
    the precision grid rounds away.  Neither is a fault: it is what the view
    shows.  This mirrors the outline extractor's own two emptiness tests so the
    caller can skip a view instead of catching an error, which would also
    swallow the failures that *are* faults.  Anything it cannot decide it hands
    to the extractor, whose validation is the authoritative one.
    """

    try:
        frame = outline_frame(view)
        origin = np.asarray(frame.origin_world_mm, dtype=np.float64)
        u_axis = np.asarray(frame.u_axis_world, dtype=np.float64)
        v_axis = np.asarray(frame.v_axis_world, dtype=np.float64)
        relative = vertices_world_mm - origin
        projected = np.column_stack((relative @ u_axis, relative @ v_axis))
        referenced = np.unique(face_subset.reshape(-1))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            referenced_scaled = projected[referenced] / precision_grid_mm
        if not bool(np.isfinite(referenced_scaled).all()):
            return True
        grid_origin = np.asarray(
            (
                np.floor(float(np.min(referenced_scaled[:, 0]))),
                np.floor(float(np.min(referenced_scaled[:, 1]))),
            ),
            dtype=np.float64,
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            lattice = projected / precision_grid_mm - grid_origin
        triangles = lattice[face_subset]
        candidates = triangles[_twice_area(triangles) != 0.0]
        if candidates.shape[0] == 0:
            return False
        return bool(np.any(_twice_area(np.rint(candidates)) != 0.0))
    except (TypeError, ValueError, IndexError):
        return True


def project_condition_region(
    vertices_world_mm: object,
    faces: object,
    face_indices: object,
    *,
    precision_grid_mm: float,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[tuple[ConditionViewBoundary, ...], tuple[dict[str, str], ...]]:
    """Project one face set into every view, and say why any view has none.

    A view is skipped rather than fatal.  A band of surface can be edge-on in
    one direction, or snap at the precision grid into pieces that touch, and
    neither is a reason to refuse an annotation that draws correctly in the
    five other views.  The skip is never silent: the reason is stored in the
    record, under a closed vocabulary, alongside the boundaries.  A fault that
    is really a fault - a limit breached, a broken backend - fails every view
    and is refused below.
    """

    grid = _precision_grid(precision_grid_mm)
    try:
        vertices = np.asarray(vertices_world_mm, dtype=np.float64)
        all_faces = np.asarray(faces, dtype=np.int64)
        selected = np.asarray(face_indices, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ArtifactConditionAnnotationError(
            f"condition region geometry is not an array: {exc}"
        ) from exc
    if all_faces.ndim != 2 or all_faces.shape[1] != 3:
        raise ArtifactConditionAnnotationError("faces must be an (n, 3) array")
    if selected.size == 0:
        raise ArtifactConditionAnnotationError(
            "a condition region needs at least one face"
        )
    if int(selected.min()) < 0 or int(selected.max()) >= all_faces.shape[0]:
        raise ArtifactConditionAnnotationError(
            "condition face index is outside the geometry"
        )
    face_subset = all_faces[selected]

    boundaries: list[ConditionViewBoundary] = []
    skipped: list[dict[str, str]] = []
    # Kept only for the refusal below.  A fault that stops every view - a limit
    # breached, a backend that cannot run - must say what it was, not report six
    # identical reason codes.
    failures: list[str] = []
    for view in CONDITION_VIEWS:
        raise_if_cancelled(cancellation_probe)
        if not _view_leaves_area(
            vertices,
            face_subset,
            view,
            precision_grid_mm=grid,
        ):
            skipped.append({"reason": CONDITION_SKIP_NO_AREA, "view": view})
            continue
        try:
            # A condition region is a painted face subset, not a closed
            # surface, and its records were all written under the plain
            # lattice union; they keep that contract so they recompute as
            # they were.
            result = extract_outline_geometry(
                vertices,
                face_subset,
                view,
                precision_grid_mm=grid,
                algorithm_version=OUTLINE_LEGACY_ALGORITHM_VERSION,
                cancellation_probe=cancellation_probe,
            )
        except ValueError as exc:
            skipped.append({"reason": CONDITION_SKIP_DEGENERATE, "view": view})
            failures.append(f"{view}: {exc}")
            continue
        boundaries.append(ConditionViewBoundary(view=view, outline=result.payload))
    if not boundaries:
        detail = "; ".join(failures) if failures else ", ".join(
            f"{entry['view']}={entry['reason']}" for entry in skipped
        )
        raise ArtifactConditionAnnotationError(
            "this region has no usable boundary in any of the six views, so it "
            f"would leave nothing on any drawing ({detail})"
        )
    return tuple(boundaries), tuple(skipped)


@dataclass(frozen=True, slots=True)
class ConditionAnnotationComputation:
    """A finished condition region, still uncommitted."""

    context: OperationContext
    projection_snapshot: ArtifactProjectionSnapshot
    payload: ConditionAnnotationPayload
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]

    def recipe_dict(self) -> dict[str, Any]:
        return validate_condition_recipe(self.recipe)

    def qc_dict(self) -> dict[str, Any]:
        return self.payload.qc_summary()


def project_condition_from_recipe(
    vertices_world_mm: object,
    faces: object,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> ConditionAnnotationPayload:
    """Carry out one condition recipe against canonical geometry.

    This is the whole computation: the recipe names the region and the grid,
    the geometry supplies the surface, and nothing else is consulted.  A worker
    thread runs exactly this, and so does an offline check.
    """

    validated = validate_condition_recipe(recipe)
    selection = validated["selection"]
    total_face_count = int(np.asarray(faces).shape[0])
    if int(selection["total_face_count"]) != total_face_count:
        raise ArtifactConditionAnnotationError(
            "condition selection was made on a mesh with a different face count"
        )
    views, skipped = project_condition_region(
        vertices_world_mm,
        faces,
        face_indices_from_ranges(
            selection["face_ranges"],
            total_face_count=total_face_count,
        ),
        precision_grid_mm=float(validated["precision_grid_mm"]),
        cancellation_probe=cancellation_probe,
    )
    return ConditionAnnotationPayload(
        schema_version=CONDITION_PAYLOAD_SCHEMA_VERSION,
        condition=str(validated["condition"]),
        selection=selection,
        views=views,
        skipped_views=skipped,
    )


def compute_condition_annotation(
    session: ArtifactSession,
    *,
    condition: str,
    face_indices: object,
    precision_grid_mm: float = DEFAULT_OUTLINE_PRECISION_GRID_MM,
    cancellation_probe: CancellationProbe | None = None,
) -> ConditionAnnotationComputation:
    """Project one painted region under the session's active alignment."""

    raise_if_cancelled(cancellation_probe)
    if not isinstance(session, ArtifactSession):
        raise ArtifactConditionAnnotationError("session must be an ArtifactSession")
    try:
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactConditionAnnotationError(str(exc)) from exc
    total_face_count = int(np.asarray(projection.mesh.faces).shape[0])
    selection = condition_selection(
        total_face_count=total_face_count,
        face_ranges=face_ranges_from_indices(
            face_indices,
            total_face_count=total_face_count,
        ),
    )
    recipe = condition_recipe(
        condition=condition,
        precision_grid_mm=precision_grid_mm,
        selection=selection,
    )
    try:
        context = session.capture_operation(
            recipe=recipe,
            selection_hash=str(selection["selection_sha256"]),
        )
    except ArtifactSessionError as exc:
        raise ArtifactConditionAnnotationError(str(exc)) from exc
    payload = project_condition_from_recipe(
        projection.mesh.vertices,
        projection.mesh.faces,
        recipe,
        cancellation_probe=cancellation_probe,
    )
    return ConditionAnnotationComputation(
        context=context,
        projection_snapshot=projection.snapshot,
        payload=payload,
        recipe=recipe,
        qc=payload.qc_summary(),
    )


def condition_computation_matches_active_projection(
    session: ArtifactSession,
    computation: ConditionAnnotationComputation,
) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(
        computation, ConditionAnnotationComputation
    ):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    return current.render_key == computation.projection_snapshot.render_key


def append_condition_record_from_context(
    document: ArtifactDocument,
    *,
    context: OperationContext,
    payload: ConditionAnnotationPayload,
    recipe: Mapping[str, Any],
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactDocument:
    """Append one verified condition region without touching source geometry."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactConditionAnnotationError("document must be an ArtifactDocument")
    if not isinstance(context, OperationContext):
        raise ArtifactConditionAnnotationError("context must be an OperationContext")
    if not isinstance(payload, ConditionAnnotationPayload):
        raise ArtifactConditionAnnotationError(
            "payload must be a ConditionAnnotationPayload"
        )
    validated_recipe = validate_condition_recipe(recipe)
    if validated_recipe["condition"] != payload.condition:
        raise ArtifactConditionAnnotationError(
            "condition recipe does not name the condition the payload records"
        )
    if validated_recipe["selection"] != payload.selection:
        raise ArtifactConditionAnnotationError(
            "condition recipe does not name the region it was computed for"
        )
    if context.selection_hash != payload.selection_sha256:
        raise ArtifactConditionAnnotationError(
            "condition context selection_hash does not match the region"
        )
    payload_bytes = payload.canonical_json_bytes()
    extensions = {
        CONDITION_PAYLOAD_EXTENSION_KEY: {
            "byte_length": len(payload_bytes),
            "media_type": CONDITION_PAYLOAD_MEDIA_TYPE,
            "payload": payload.to_dict(),
            "schema_version": CONDITION_PAYLOAD_SCHEMA_VERSION,
            "sha256": payload.sha256,
        }
    }
    try:
        return document.append_record_from_context(
            context=context,
            id=record_id,
            type=CONDITION_RECORD_TYPE,
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
        raise ArtifactConditionAnnotationError(str(exc)) from exc


def commit_condition_annotation(
    session: ArtifactSession,
    computation: ConditionAnnotationComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    if not condition_computation_matches_active_projection(session, computation):
        raise ArtifactConditionAnnotationError(
            "condition computation is stale for the active projection"
        )
    document = append_condition_record_from_context(
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


def condition_payload_from_record(
    record: DerivedRecord,
) -> ConditionAnnotationPayload:
    """Resolve and re-verify one condition record's inline region."""

    if not isinstance(record, DerivedRecord):
        raise ArtifactConditionAnnotationError("record must be a DerivedRecord")
    if record.type != CONDITION_RECORD_TYPE:
        raise ArtifactConditionAnnotationError(
            f"record is not a condition annotation: {record.type!r}"
        )
    descriptor = _exact_keys(
        record.extensions.get(CONDITION_PAYLOAD_EXTENSION_KEY),
        _DESCRIPTOR_KEYS,
        name="condition payload descriptor",
    )
    if descriptor["media_type"] != CONDITION_PAYLOAD_MEDIA_TYPE:
        raise ArtifactConditionAnnotationError("condition payload media_type is invalid")
    if descriptor["schema_version"] != CONDITION_PAYLOAD_SCHEMA_VERSION:
        raise ArtifactConditionAnnotationError(
            "condition payload descriptor schema is invalid"
        )
    raw_payload = descriptor["payload"]
    if not isinstance(raw_payload, Mapping):
        raise ArtifactConditionAnnotationError(
            "condition payload descriptor payload must be an object"
        )
    payload = ConditionAnnotationPayload.from_dict(raw_payload)
    payload_bytes = payload.canonical_json_bytes()
    byte_length = descriptor["byte_length"]
    if type(byte_length) is not int or byte_length != len(payload_bytes):
        raise ArtifactConditionAnnotationError(
            "condition payload byte_length does not match payload"
        )
    if descriptor["sha256"] != payload.sha256:
        raise ArtifactConditionAnnotationError(
            "condition payload SHA-256 does not match payload"
        )
    if record.geometry_ref != payload.geometry_ref:
        raise ArtifactConditionAnnotationError(
            "condition record geometry_ref does not match payload"
        )
    recipe = validate_condition_recipe(record.recipe)
    if recipe["condition"] != payload.condition:
        raise ArtifactConditionAnnotationError(
            "condition record recipe does not name the condition it stores"
        )
    if recipe["selection"] != payload.selection:
        raise ArtifactConditionAnnotationError(
            "condition record recipe does not name the region it stores"
        )
    if record.selection_hash != payload.selection_sha256:
        raise ArtifactConditionAnnotationError(
            "condition record selection_hash does not match its region"
        )
    thawed_qc = record.to_dict()["qc"]
    assert isinstance(thawed_qc, dict)
    expected_qc = payload.qc_summary()
    if thawed_qc != expected_qc:
        differing = sorted(
            key
            for key in set(thawed_qc) | set(expected_qc)
            if thawed_qc.get(key) != expected_qc.get(key)
        )
        raise ArtifactConditionAnnotationError(
            f"condition record QC does not match its payload: {', '.join(differing)}"
        )
    return payload


def validate_condition_annotation_records(document: ArtifactDocument) -> None:
    """Strictly validate every condition record embedded in a document."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactConditionAnnotationError("document must be an ArtifactDocument")
    for record in document.records:
        if record.type == CONDITION_RECORD_TYPE:
            condition_payload_from_record(record)


__all__ = [
    "ArtifactConditionAnnotationError",
    "CONDITION_ALGORITHM",
    "CONDITION_ALGORITHM_VERSION",
    "CONDITION_CRACK",
    "CONDITION_GEOMETRY_REF_PREFIX",
    "CONDITION_KINDS",
    "CONDITION_MISSING",
    "CONDITION_OPERATION_KIND",
    "CONDITION_PAYLOAD_EXTENSION_KEY",
    "CONDITION_PAYLOAD_MEDIA_TYPE",
    "CONDITION_PAYLOAD_SCHEMA_VERSION",
    "CONDITION_RECORD_TYPE",
    "CONDITION_RESTORED",
    "CONDITION_SELECTION_KIND",
    "CONDITION_SKIP_DEGENERATE",
    "CONDITION_SKIP_NO_AREA",
    "CONDITION_SKIP_REASONS",
    "CONDITION_VIEWS",
    "CONDITION_WORN",
    "ConditionAnnotationComputation",
    "ConditionAnnotationPayload",
    "ConditionViewBoundary",
    "MAX_CONDITION_FACES",
    "MAX_CONDITION_FACE_RANGES",
    "MAX_CONDITION_PAYLOAD_BYTES",
    "append_condition_record_from_context",
    "commit_condition_annotation",
    "compute_condition_annotation",
    "condition_computation_matches_active_projection",
    "condition_kind",
    "condition_payload_from_record",
    "condition_recipe",
    "condition_selection",
    "face_indices_from_ranges",
    "face_ranges_from_indices",
    "project_condition_from_recipe",
    "project_condition_region",
    "validate_condition_annotation_records",
    "validate_condition_recipe",
    "validate_condition_selection",
    "validate_face_ranges",
]
