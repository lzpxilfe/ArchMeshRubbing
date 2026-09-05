"""Manufacturing technique marks as a durable set of faces.

목리조정흔, 지두흔, 타날흔, 테쌓기흔, 물손질흔, 마연흔, 목판정면, 내박자흔, 깎기 -
the marks a potter's tools and hands left on the wall - are read off the surface by the archaeologist and
drawn on the elevation.  The program does not find them; it records where the
archaeologist painted them, projects that region into the six views the way
a condition region is projected, and draws each kind by its convention.

The record is the condition record's twin: the same canonical run-length
face set, the same per-view boundaries computed at commit time, the same
fail-closed rules.  What differs is the vocabulary and the drawing: a
finger mark (지두흔) is drawn as a U-shaped symbol rather than by its
boundary, because that is how the convention shows one.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_outline_extractor import (
    OUTLINE_CLOSING_ALGORITHM_VERSION,
    OUTLINE_LEGACY_ALGORITHM_VERSION,
)
from .artifact_condition_annotation import (
    CONDITION_SKIP_REASONS,
    CONDITION_VIEWS,
    MAX_CONDITION_PAYLOAD_BYTES,
    ArtifactConditionAnnotationError,
    ConditionViewBoundary,
    condition_selection,
    face_indices_from_ranges,
    face_ranges_from_indices,
    project_condition_region,
    validate_condition_selection,
)
from .artifact_document import (
    ArtifactDocument,
    ArtifactDocumentError,
    DerivedRecord,
    OperationContext,
    RecordLifecycleStatus,
)
from .artifact_outline_extractor import DEFAULT_OUTLINE_PRECISION_GRID_MM
from .artifact_cancellation import CancellationProbe, raise_if_cancelled
from .artifact_scene_adapter import ArtifactProjectionSnapshot
from .artifact_session import ArtifactSession, ArtifactSessionError
from .canonical_json import CanonicalJSONError, canonical_json_bytes, canonical_json_sha256


TECHNIQUE_RECORD_TYPE = "annotation.technique.v1"
TECHNIQUE_OPERATION_KIND = "technique_annotation"
TECHNIQUE_ALGORITHM = "archmeshrubbing.technique_region_projection"
#: 1.0.0 drew the boundary with the plain lattice union, like a condition;
#: 1.1.0 draws it with outline algorithm 1.1.0's grid closing, because a
#: mark painted on a rough wall snaps into slivers that touch and the plain
#: union then refuses every view; 1.2.0 also leaves out of each view the
#: faces seen there at a grazing angle, because a mark that reaches the
#: silhouette projects to a sliver at its edge and the sliver would make the
#: whole view fail; 1.3.0 unions the region exactly and snaps it once as a
#: whole (extract_region_geometry) instead of snapping its thousands of tiny
#: triangles one by one, which on a fine mesh left slivers the closing could
#: not mend; 1.4.0 tidies up after the grazing cut of 1.2.0, keeping inside
#: each painted piece only the largest piece that cut leaves - the cut is
#: per face and a relieved wall makes it ragged, so it strews single faces
#: beside the mark that touch it at a lattice corner and cost the whole view.
#: Every earlier version recomputes as written.
TECHNIQUE_ALGORITHM_VERSION = "1.4.0"
TECHNIQUE_ALGORITHM_VERSIONS = (
    "1.0.0",
    "1.1.0",
    "1.2.0",
    "1.3.0",
    TECHNIQUE_ALGORITHM_VERSION,
)
#: Pinned as literals: a technique record recomputes with the outline
#: algorithm it was written under, not with whichever is current.  1.3.0 on
#: draws with extract_region_geometry, where this is unused.
_OUTLINE_ALGORITHM_FOR_TECHNIQUE: Mapping[str, str] = {
    "1.0.0": OUTLINE_LEGACY_ALGORITHM_VERSION,
    "1.1.0": OUTLINE_CLOSING_ALGORITHM_VERSION,
    "1.2.0": OUTLINE_CLOSING_ALGORITHM_VERSION,
    "1.3.0": OUTLINE_CLOSING_ALGORITHM_VERSION,
    "1.4.0": OUTLINE_CLOSING_ALGORITHM_VERSION,
}
#: cos 81.4 degrees: a face tilted more than that from the view is left out.
TECHNIQUE_GRAZING_COSINE_MIN = 0.15
_GRAZING_FOR_TECHNIQUE: Mapping[str, float] = {
    "1.0.0": 0.0,
    "1.1.0": 0.0,
    "1.2.0": TECHNIQUE_GRAZING_COSINE_MIN,
    "1.3.0": TECHNIQUE_GRAZING_COSINE_MIN,
    "1.4.0": TECHNIQUE_GRAZING_COSINE_MIN,
}
_REGION_UNION_FOR_TECHNIQUE: Mapping[str, bool] = {
    "1.0.0": False,
    "1.1.0": False,
    "1.2.0": False,
    "1.3.0": True,
    "1.4.0": True,
}
_TRIM_ISLANDS_FOR_TECHNIQUE: Mapping[str, bool] = {
    "1.0.0": False,
    "1.1.0": False,
    "1.2.0": False,
    "1.3.0": False,
    "1.4.0": True,
}
#: 1.0.0 held the kind, the face set and the six views.  1.1.0 adds which
#: side of the wall the faces are on (a coil seam or a finger press is
#: usually seen inside, where the wall was not smoothed over) and, when the
#: drafter observed it, the direction the tool moved.  A 1.0.0 payload reads
#: and digests exactly as it was written.
#: 1.2.0 adds four kinds to the vocabulary.  The keys do not change - a
#: 1.2.0 payload has exactly the shape of a 1.1.0 one - but the closed set of
#: values `technique` may take does, and a reader that only knows 1.1.0 must
#: not be handed 마연흔 and left to guess.  So the new kinds are refused in a
#: 1.0.0 or 1.1.0 payload, and every payload already written reads and
#: digests exactly as it was written.
TECHNIQUE_PAYLOAD_SCHEMA_VERSION = "1.2.0"
TECHNIQUE_PAYLOAD_SCHEMA_VERSIONS = ("1.0.0", "1.1.0", TECHNIQUE_PAYLOAD_SCHEMA_VERSION)
#: Which side of the wall a painted face set is on.  Decided from the mesh:
#: a face whose normal points away from the mesh's centroid is exterior, one
#: whose normal points toward it is interior; a region that mixes the two by
#: more than a tenth either way is "mixed", which a sheet treats as exterior
#: and says so.
SURFACE_EXTERIOR = "exterior"
SURFACE_INTERIOR = "interior"
SURFACE_MIXED = "mixed"
SURFACE_SIDES = (SURFACE_EXTERIOR, SURFACE_INTERIOR, SURFACE_MIXED)
_SURFACE_SIDE_MARGIN = 100_000  # millionths: a tenth of the faces
TECHNIQUE_PAYLOAD_EXTENSION_KEY = "org.archmeshrubbing:technique-annotation-v1"
TECHNIQUE_PAYLOAD_MEDIA_TYPE = "application/vnd.archmeshrubbing.technique-annotation+json"
TECHNIQUE_GEOMETRY_REF_PREFIX = "urn:archmeshrubbing:technique-annotation:sha256:"

#: 목판정면: the wall finished with a wooden board held against it.
TECHNIQUE_BOARD_FINISHING = "board_finishing"
#: 마연흔: the leather-hard wall compacted and glossed with a smooth tool.
TECHNIQUE_BURNISHING = "burnishing"
TECHNIQUE_COIL_JOINT = "coil_joint"
TECHNIQUE_FINGER_MARK = "finger_mark"
#: 내박자흔: the anvil held inside the wall while the paddle struck outside.
TECHNIQUE_INTERIOR_ANVIL = "interior_anvil"
TECHNIQUE_PADDLING = "paddling"
#: 깎기: material pared away with a blade, leaving facets and their edges.
TECHNIQUE_PARING = "paring"
TECHNIQUE_WATER_SMOOTHING = "water_smoothing"
TECHNIQUE_WOOD_GRAIN = "wood_grain_smoothing"
#: Closed vocabulary, in canonical (sorted) order.
TECHNIQUE_KINDS: tuple[str, ...] = (
    TECHNIQUE_BOARD_FINISHING,
    TECHNIQUE_BURNISHING,
    TECHNIQUE_COIL_JOINT,
    TECHNIQUE_FINGER_MARK,
    TECHNIQUE_INTERIOR_ANVIL,
    TECHNIQUE_PADDLING,
    TECHNIQUE_PARING,
    TECHNIQUE_WATER_SMOOTHING,
    TECHNIQUE_WOOD_GRAIN,
)
#: The four a 1.1.0 payload cannot name.  Sorted, so the refusal reads the
#: same way every time.
TECHNIQUE_KINDS_SINCE_1_2: tuple[str, ...] = (
    TECHNIQUE_BOARD_FINISHING,
    TECHNIQUE_BURNISHING,
    TECHNIQUE_INTERIOR_ANVIL,
    TECHNIQUE_PARING,
)
#: What the drafter calls each kind.
TECHNIQUE_KIND_LABELS_KO: Mapping[str, str] = {
    TECHNIQUE_BOARD_FINISHING: "목판정면",
    TECHNIQUE_BURNISHING: "마연흔",
    TECHNIQUE_COIL_JOINT: "테쌓기흔",
    TECHNIQUE_FINGER_MARK: "지두흔",
    TECHNIQUE_INTERIOR_ANVIL: "내박자흔",
    TECHNIQUE_PADDLING: "타날흔",
    TECHNIQUE_PARING: "깎기",
    TECHNIQUE_WATER_SMOOTHING: "물손질흔",
    TECHNIQUE_WOOD_GRAIN: "목리조정흔",
}
TECHNIQUE_VIEWS = CONDITION_VIEWS
TECHNIQUE_SKIP_REASONS = CONDITION_SKIP_REASONS


class ArtifactTechniqueAnnotationError(ValueError):
    """A technique region cannot be recorded or read as declared."""


def _exact_keys(value: object, keys: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactTechniqueAnnotationError(f"{name} must be an object")
    if set(value) != keys:
        raise ArtifactTechniqueAnnotationError(
            f"{name} must have exactly the keys {', '.join(sorted(keys))}"
        )
    return value


def technique_kind(value: object) -> str:
    if not isinstance(value, str) or value not in TECHNIQUE_KINDS:
        raise ArtifactTechniqueAnnotationError(
            f"technique must be one of {', '.join(TECHNIQUE_KINDS)}"
        )
    return value


def _precision_grid(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactTechniqueAnnotationError("precision_grid_mm must be a number")
    grid = float(value)
    if not (0.0 < grid <= 10.0) or grid != grid:
        raise ArtifactTechniqueAnnotationError(
            "precision_grid_mm must be greater than 0 and at most 10"
        )
    return grid


def _selection(value: object) -> dict[str, Any]:
    try:
        return validate_condition_selection(value)
    except ArtifactConditionAnnotationError as exc:
        raise ArtifactTechniqueAnnotationError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class TechniqueAnnotationPayload:
    """One painted mark: what technique it is, which faces, how it projects."""

    schema_version: str
    technique: str
    selection: Mapping[str, Any]
    views: tuple[ConditionViewBoundary, ...]
    skipped_views: tuple[Mapping[str, str], ...] = ()
    surface_side: str | None = None
    interior_face_fraction_millionths: int | None = None
    direction_deg: float | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in TECHNIQUE_PAYLOAD_SCHEMA_VERSIONS:
            raise ArtifactTechniqueAnnotationError(
                f"unsupported technique payload schema: {self.schema_version!r}"
            )
        if (
            self.schema_version != TECHNIQUE_PAYLOAD_SCHEMA_VERSION
            and self.technique in TECHNIQUE_KINDS_SINCE_1_2
        ):
            raise ArtifactTechniqueAnnotationError(
                f"technique {self.technique!r} needs a "
                f"{TECHNIQUE_PAYLOAD_SCHEMA_VERSION} payload"
            )
        if self.schema_version == "1.0.0":
            if (
                self.surface_side is not None
                or self.interior_face_fraction_millionths is not None
                or self.direction_deg is not None
            ):
                raise ArtifactTechniqueAnnotationError(
                    "a 1.0.0 technique payload carries no surface side or direction"
                )
        else:
            if self.surface_side not in SURFACE_SIDES:
                raise ArtifactTechniqueAnnotationError(
                    f"technique surface_side must be one of {', '.join(SURFACE_SIDES)}"
                )
            fraction = self.interior_face_fraction_millionths
            if type(fraction) is not int or not (0 <= fraction <= 1_000_000):
                raise ArtifactTechniqueAnnotationError(
                    "technique interior_face_fraction_millionths must be an integer "
                    "between 0 and 1000000"
                )
            if self.direction_deg is not None:
                object.__setattr__(
                    self, "direction_deg", _direction_deg(self.direction_deg)
                )
        object.__setattr__(self, "technique", technique_kind(self.technique))
        object.__setattr__(self, "selection", _selection(self.selection))
        views = tuple(self.views)
        if any(not isinstance(view, ConditionViewBoundary) for view in views):
            raise ArtifactTechniqueAnnotationError(
                "technique views must be ConditionViewBoundary values"
            )
        if not views:
            raise ArtifactTechniqueAnnotationError(
                "a technique region must project into at least one view"
            )
        names = [view.view for view in views]
        if len(set(names)) != len(names):
            raise ArtifactTechniqueAnnotationError(
                "a technique region holds at most one boundary per view"
            )
        object.__setattr__(self, "views", tuple(sorted(views, key=lambda item: item.view)))
        skipped: list[dict[str, str]] = []
        for entry in self.skipped_views:
            block = _exact_keys(
                entry, frozenset({"reason", "view"}), name="technique skipped view"
            )
            view = block["view"]
            reason = block["reason"]
            if not isinstance(view, str) or view not in TECHNIQUE_VIEWS:
                raise ArtifactTechniqueAnnotationError(
                    f"technique view must be one of {', '.join(TECHNIQUE_VIEWS)}"
                )
            if not isinstance(reason, str) or reason not in TECHNIQUE_SKIP_REASONS:
                raise ArtifactTechniqueAnnotationError(
                    "technique skip reason must be one of "
                    f"{', '.join(TECHNIQUE_SKIP_REASONS)}"
                )
            skipped.append({"reason": reason, "view": view})
        accounted = names + [entry["view"] for entry in skipped]
        if sorted(accounted) != list(TECHNIQUE_VIEWS):
            raise ArtifactTechniqueAnnotationError(
                "a technique region must account for each of the six views "
                "exactly once, with a boundary or a reason"
            )
        object.__setattr__(
            self, "skipped_views", tuple(sorted(skipped, key=lambda item: item["view"]))
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
            self.selection["face_ranges"], total_face_count=self.total_face_count
        )

    def boundary_for_view(self, view: str) -> ConditionViewBoundary | None:
        for boundary in self.views:
            if boundary.view == view:
                return boundary
        return None

    def to_dict(self) -> dict[str, Any]:
        block: dict[str, Any] = {
            "schema_version": self.schema_version,
            "selection": dict(self.selection),
            "skipped_views": [dict(entry) for entry in self.skipped_views],
            "technique": self.technique,
            "views": [view.to_dict() for view in self.views],
        }
        if self.schema_version != "1.0.0":
            block["direction_deg"] = self.direction_deg
            block["interior_face_fraction_millionths"] = self.interior_face_fraction_millionths
            block["surface_side"] = self.surface_side
        return block

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TechniqueAnnotationPayload":
        base_keys = frozenset({"schema_version", "selection", "skipped_views", "technique", "views"})
        if isinstance(data, Mapping) and data.get("schema_version") == "1.0.0":
            block = _exact_keys(data, base_keys, name="technique payload")
        else:
            block = _exact_keys(
                data,
                base_keys | {"direction_deg", "interior_face_fraction_millionths", "surface_side"},
                name="technique payload",
            )
        raw_views = block["views"]
        raw_skipped = block["skipped_views"]
        if not isinstance(raw_views, (list, tuple)) or not isinstance(
            raw_skipped, (list, tuple)
        ):
            raise ArtifactTechniqueAnnotationError("technique payload views must be an array")
        try:
            views = tuple(
                ConditionViewBoundary.from_dict(view)  # type: ignore[arg-type]
                for view in raw_views
            )
        except ArtifactConditionAnnotationError as exc:
            raise ArtifactTechniqueAnnotationError(str(exc)) from exc
        schema_version = block["schema_version"]
        technique = block["technique"]
        side = block.get("surface_side")
        fraction = block.get("interior_face_fraction_millionths")
        direction = block.get("direction_deg")
        return cls(
            schema_version=schema_version if isinstance(schema_version, str) else "",
            technique=technique if isinstance(technique, str) else "",
            selection=block["selection"],  # type: ignore[arg-type]
            views=views,
            skipped_views=tuple(raw_skipped),  # type: ignore[arg-type]
            surface_side=side if isinstance(side, str) else None,
            interior_face_fraction_millionths=(
                fraction if type(fraction) is int else None
            ),
            direction_deg=direction if isinstance(direction, (int, float)) else None,
        )

    def canonical_json_bytes(self) -> bytes:
        try:
            encoded = canonical_json_bytes(self.to_dict())
        except CanonicalJSONError as exc:
            raise ArtifactTechniqueAnnotationError(str(exc)) from exc
        if len(encoded) > MAX_CONDITION_PAYLOAD_BYTES:
            raise ArtifactTechniqueAnnotationError(
                f"technique payload exceeds the {MAX_CONDITION_PAYLOAD_BYTES}-byte "
                "inline safety limit"
            )
        return encoded

    @property
    def sha256(self) -> str:
        try:
            return canonical_json_sha256(self.to_dict())
        except CanonicalJSONError as exc:
            raise ArtifactTechniqueAnnotationError(str(exc)) from exc

    @property
    def geometry_ref(self) -> str:
        return f"{TECHNIQUE_GEOMETRY_REF_PREFIX}{self.sha256}"

    def qc_summary(self) -> dict[str, Any]:
        try:
            views = [view.qc_summary() for view in self.views]
        except ArtifactConditionAnnotationError as exc:
            raise ArtifactTechniqueAnnotationError(str(exc)) from exc
        return {
            "empty_view_reasons": [dict(entry) for entry in self.skipped_views],
            "empty_views": [entry["view"] for entry in self.skipped_views],
            "face_count": self.face_count,
            "face_range_count": len(self.selection["face_ranges"]),
            "payload_sha256": self.sha256,
            "projected_view_count": len(self.views),
            "selection_sha256": self.selection_sha256,
            "technique": self.technique,
            "total_face_count": self.total_face_count,
            "views": views,
            **(
                {}
                if self.schema_version == "1.0.0"
                else {
                    "direction_deg": self.direction_deg,
                    "interior_face_fraction_millionths": self.interior_face_fraction_millionths,
                    "surface_side": self.surface_side,
                }
            ),
        }


def _direction_deg(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ArtifactTechniqueAnnotationError(
            "technique direction_deg must be a finite number of degrees or null"
        )
    return float(value) % 180.0


def surface_side_of_faces(
    vertices_world_mm: object, faces: object, face_indices: object
) -> tuple[str, int]:
    """Say which side of the wall a face set is on, and how much of it is inside.

    Exterior faces have normals pointing away from the mesh's centroid and
    interior faces toward it; the fraction is the share of the painted area
    facing in.  Degenerate faces carry no area and no vote.
    """

    vertices = np.asarray(vertices_world_mm, dtype=np.float64)
    all_faces = np.asarray(faces, dtype=np.int64)
    selected = np.asarray(face_indices, dtype=np.int64).reshape(-1)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or all_faces.ndim != 2:
        raise ArtifactTechniqueAnnotationError("surface side needs (n, 3) vertices and faces")
    if selected.size == 0:
        raise ArtifactTechniqueAnnotationError("surface side needs at least one face")
    if selected.min() < 0 or selected.max() >= all_faces.shape[0]:
        raise ArtifactTechniqueAnnotationError("surface side face index is outside the geometry")
    centre = vertices.mean(axis=0)
    corners = vertices[all_faces[selected]]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area2 = np.linalg.norm(normals, axis=1)
    centroids = corners.mean(axis=1)
    facing_in = np.einsum("ij,ij->i", normals, centroids - centre) < 0.0
    total = float(area2.sum())
    if total <= 0.0:
        raise ArtifactTechniqueAnnotationError("surface side needs faces with area")
    fraction = int(round(1_000_000 * float(area2[facing_in].sum()) / total))
    if fraction >= 1_000_000 - _SURFACE_SIDE_MARGIN:
        return SURFACE_INTERIOR, fraction
    if fraction <= _SURFACE_SIDE_MARGIN:
        return SURFACE_EXTERIOR, fraction
    return SURFACE_MIXED, fraction


def technique_recipe(
    *,
    technique: str,
    precision_grid_mm: float,
    selection: Mapping[str, Any],
    algorithm_version: str = TECHNIQUE_ALGORITHM_VERSION,
    direction_deg: float | None = None,
) -> dict[str, Any]:
    """The closed recipe one technique region is computed under.

    The whole selection is in the recipe, as for a condition: the region is
    the operation, and two marks painted the same way are two computations.
    """

    recipe: dict[str, Any] = {
        "algorithm": TECHNIQUE_ALGORITHM,
        "algorithm_version": _technique_algorithm_version(algorithm_version),
        "kind": TECHNIQUE_OPERATION_KIND,
        "precision_grid_mm": _precision_grid(precision_grid_mm),
        "selection": _selection(selection),
        "technique": technique_kind(technique),
        "views": list(TECHNIQUE_VIEWS),
    }
    if direction_deg is not None:
        # Omitted when unknown, so a recipe written before the key existed
        # keeps its bytes and its hash.
        recipe["direction_deg"] = _direction_deg(direction_deg)
    return recipe


_RECIPE_KEYS = frozenset(
    {"algorithm", "algorithm_version", "kind", "precision_grid_mm", "selection", "technique", "views"}
)


def validate_technique_recipe(value: object) -> dict[str, Any]:
    keys = _RECIPE_KEYS
    if isinstance(value, Mapping) and "direction_deg" in value:
        keys = _RECIPE_KEYS | {"direction_deg"}
    recipe = _exact_keys(value, keys, name="technique recipe")
    if recipe["algorithm"] != TECHNIQUE_ALGORITHM:
        raise ArtifactTechniqueAnnotationError("technique algorithm is unsupported")
    _technique_algorithm_version(recipe["algorithm_version"])
    if recipe["kind"] != TECHNIQUE_OPERATION_KIND:
        raise ArtifactTechniqueAnnotationError("technique recipe kind is unsupported")
    if list(recipe["views"]) != list(TECHNIQUE_VIEWS):
        raise ArtifactTechniqueAnnotationError(
            "a technique region is projected into all six views"
        )
    return technique_recipe(
        technique=recipe["technique"],
        precision_grid_mm=recipe["precision_grid_mm"],
        selection=recipe["selection"],
        algorithm_version=str(recipe["algorithm_version"]),
        direction_deg=recipe.get("direction_deg"),
    )


def _technique_algorithm_version(value: object) -> str:
    if not isinstance(value, str) or value not in TECHNIQUE_ALGORITHM_VERSIONS:
        raise ArtifactTechniqueAnnotationError("technique algorithm version is unsupported")
    return value


@dataclass(frozen=True, slots=True)
class TechniqueAnnotationComputation:
    """A finished technique region, still uncommitted."""

    context: OperationContext
    projection_snapshot: ArtifactProjectionSnapshot
    payload: TechniqueAnnotationPayload
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]

    def recipe_dict(self) -> dict[str, Any]:
        return validate_technique_recipe(self.recipe)

    def qc_dict(self) -> dict[str, Any]:
        return self.payload.qc_summary()


def project_technique_from_recipe(
    vertices_world_mm: object,
    faces: object,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> TechniqueAnnotationPayload:
    """Carry out one technique recipe against canonical geometry."""

    validated = validate_technique_recipe(recipe)
    selection = validated["selection"]
    total_face_count = int(np.asarray(faces).shape[0])
    if int(selection["total_face_count"]) != total_face_count:
        raise ArtifactTechniqueAnnotationError(
            "technique selection was made on a mesh with a different face count"
        )
    try:
        views, skipped = project_condition_region(
            vertices_world_mm,
            faces,
            face_indices_from_ranges(
                selection["face_ranges"], total_face_count=total_face_count
            ),
            precision_grid_mm=float(validated["precision_grid_mm"]),
            cancellation_probe=cancellation_probe,
            outline_algorithm_version=_OUTLINE_ALGORITHM_FOR_TECHNIQUE[
                str(validated["algorithm_version"])
            ],
            grazing_cosine_min=_GRAZING_FOR_TECHNIQUE[str(validated["algorithm_version"])],
            region_union=_REGION_UNION_FOR_TECHNIQUE[str(validated["algorithm_version"])],
            trim_cut_islands=_TRIM_ISLANDS_FOR_TECHNIQUE[
                str(validated["algorithm_version"])
            ],
        )
    except ArtifactConditionAnnotationError as exc:
        raise ArtifactTechniqueAnnotationError(str(exc)) from exc
    side, fraction = surface_side_of_faces(
        vertices_world_mm,
        faces,
        face_indices_from_ranges(selection["face_ranges"], total_face_count=total_face_count),
    )
    return TechniqueAnnotationPayload(
        schema_version=TECHNIQUE_PAYLOAD_SCHEMA_VERSION,
        technique=str(validated["technique"]),
        selection=selection,
        views=views,
        skipped_views=skipped,
        surface_side=side,
        interior_face_fraction_millionths=fraction,
        direction_deg=validated.get("direction_deg"),
    )


def technique_selection(
    *, total_face_count: int, face_indices: object
) -> dict[str, Any]:
    """The durable face-set block for a painted mark."""

    try:
        return condition_selection(
            total_face_count=total_face_count,
            face_ranges=face_ranges_from_indices(
                face_indices, total_face_count=total_face_count
            ),
        )
    except ArtifactConditionAnnotationError as exc:
        raise ArtifactTechniqueAnnotationError(str(exc)) from exc


def compute_technique_annotation(
    session: ArtifactSession,
    *,
    technique: str,
    face_indices: object,
    precision_grid_mm: float = DEFAULT_OUTLINE_PRECISION_GRID_MM,
    cancellation_probe: CancellationProbe | None = None,
    direction_deg: float | None = None,
) -> TechniqueAnnotationComputation:
    """Project one painted mark under the session's active alignment."""

    raise_if_cancelled(cancellation_probe)
    if not isinstance(session, ArtifactSession):
        raise ArtifactTechniqueAnnotationError("session must be an ArtifactSession")
    try:
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactTechniqueAnnotationError(str(exc)) from exc
    total_face_count = int(np.asarray(projection.mesh.faces).shape[0])
    selection = technique_selection(
        total_face_count=total_face_count, face_indices=face_indices
    )
    recipe = technique_recipe(
        technique=technique,
        precision_grid_mm=precision_grid_mm,
        selection=selection,
        direction_deg=direction_deg,
    )
    try:
        context = session.capture_operation(
            recipe=recipe, selection_hash=str(selection["selection_sha256"])
        )
    except ArtifactSessionError as exc:
        raise ArtifactTechniqueAnnotationError(str(exc)) from exc
    payload = project_technique_from_recipe(
        projection.mesh.vertices,
        projection.mesh.faces,
        recipe,
        cancellation_probe=cancellation_probe,
    )
    return TechniqueAnnotationComputation(
        context=context,
        projection_snapshot=projection.snapshot,
        payload=payload,
        recipe=recipe,
        qc=payload.qc_summary(),
    )


def technique_computation_matches_active_projection(
    session: ArtifactSession, computation: TechniqueAnnotationComputation
) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(
        computation, TechniqueAnnotationComputation
    ):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    return current.render_key == computation.projection_snapshot.render_key


def append_technique_record_from_context(
    document: ArtifactDocument,
    *,
    context: OperationContext,
    payload: TechniqueAnnotationPayload,
    recipe: Mapping[str, Any],
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactDocument:
    """Append one verified technique region without touching source geometry."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactTechniqueAnnotationError("document must be an ArtifactDocument")
    if not isinstance(context, OperationContext):
        raise ArtifactTechniqueAnnotationError("context must be an OperationContext")
    if not isinstance(payload, TechniqueAnnotationPayload):
        raise ArtifactTechniqueAnnotationError("payload must be a TechniqueAnnotationPayload")
    validated_recipe = validate_technique_recipe(recipe)
    if validated_recipe["technique"] != payload.technique:
        raise ArtifactTechniqueAnnotationError(
            "technique recipe does not name the technique the payload records"
        )
    if validated_recipe["selection"] != payload.selection:
        raise ArtifactTechniqueAnnotationError(
            "technique recipe does not name the region it was computed for"
        )
    if context.selection_hash != payload.selection_sha256:
        raise ArtifactTechniqueAnnotationError(
            "technique context selection_hash does not match the region"
        )
    payload_bytes = payload.canonical_json_bytes()
    extensions = {
        TECHNIQUE_PAYLOAD_EXTENSION_KEY: {
            "byte_length": len(payload_bytes),
            "media_type": TECHNIQUE_PAYLOAD_MEDIA_TYPE,
            "payload": payload.to_dict(),
            "schema_version": TECHNIQUE_PAYLOAD_SCHEMA_VERSION,
            "sha256": payload.sha256,
        }
    }
    try:
        return document.append_record_from_context(
            context=context,
            id=record_id,
            type=TECHNIQUE_RECORD_TYPE,
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
        raise ArtifactTechniqueAnnotationError(str(exc)) from exc


def commit_technique_annotation(
    session: ArtifactSession,
    computation: TechniqueAnnotationComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    if not technique_computation_matches_active_projection(session, computation):
        raise ArtifactTechniqueAnnotationError(
            "technique computation is stale for the active projection"
        )
    document = append_technique_record_from_context(
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


def technique_payload_from_record(record: DerivedRecord) -> TechniqueAnnotationPayload:
    """Resolve and re-verify one technique record's inline region."""

    if not isinstance(record, DerivedRecord):
        raise ArtifactTechniqueAnnotationError("record must be a DerivedRecord")
    if record.type != TECHNIQUE_RECORD_TYPE:
        raise ArtifactTechniqueAnnotationError(
            f"record is not a technique annotation: {record.type!r}"
        )
    descriptor = _exact_keys(
        record.extensions.get(TECHNIQUE_PAYLOAD_EXTENSION_KEY),
        _DESCRIPTOR_KEYS,
        name="technique payload descriptor",
    )
    if descriptor["media_type"] != TECHNIQUE_PAYLOAD_MEDIA_TYPE:
        raise ArtifactTechniqueAnnotationError("technique payload media_type is invalid")
    if descriptor["schema_version"] != TECHNIQUE_PAYLOAD_SCHEMA_VERSION:
        raise ArtifactTechniqueAnnotationError(
            "technique payload descriptor schema is invalid"
        )
    raw_payload = descriptor["payload"]
    if not isinstance(raw_payload, Mapping):
        raise ArtifactTechniqueAnnotationError(
            "technique payload descriptor payload must be an object"
        )
    payload = TechniqueAnnotationPayload.from_dict(raw_payload)
    payload_bytes = payload.canonical_json_bytes()
    byte_length = descriptor["byte_length"]
    if type(byte_length) is not int or byte_length != len(payload_bytes):
        raise ArtifactTechniqueAnnotationError(
            "technique payload byte_length does not match payload"
        )
    if descriptor["sha256"] != payload.sha256:
        raise ArtifactTechniqueAnnotationError("technique payload SHA-256 does not match payload")
    if record.geometry_ref != payload.geometry_ref:
        raise ArtifactTechniqueAnnotationError(
            "technique record geometry_ref does not match payload"
        )
    recipe = validate_technique_recipe(record.recipe)
    if recipe["technique"] != payload.technique:
        raise ArtifactTechniqueAnnotationError(
            "technique record recipe does not name the technique it stores"
        )
    if recipe["selection"] != payload.selection:
        raise ArtifactTechniqueAnnotationError(
            "technique record recipe does not name the region it stores"
        )
    if record.selection_hash != payload.selection_sha256:
        raise ArtifactTechniqueAnnotationError(
            "technique record selection_hash does not match its region"
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
        raise ArtifactTechniqueAnnotationError(
            f"technique record QC does not match its payload: {', '.join(differing)}"
        )
    return payload


def validate_technique_annotation_records(document: ArtifactDocument) -> None:
    """Strictly validate every technique record embedded in a document."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactTechniqueAnnotationError("document must be an ArtifactDocument")
    for record in document.records:
        if record.type == TECHNIQUE_RECORD_TYPE:
            technique_payload_from_record(record)


__all__ = [
    "ArtifactTechniqueAnnotationError",
    "TECHNIQUE_ALGORITHM",
    "TECHNIQUE_ALGORITHM_VERSION",
    "TECHNIQUE_ALGORITHM_VERSIONS",
    "TECHNIQUE_GRAZING_COSINE_MIN",
    "TECHNIQUE_BOARD_FINISHING",
    "TECHNIQUE_BURNISHING",
    "TECHNIQUE_COIL_JOINT",
    "TECHNIQUE_FINGER_MARK",
    "TECHNIQUE_INTERIOR_ANVIL",
    "TECHNIQUE_GEOMETRY_REF_PREFIX",
    "TECHNIQUE_KINDS",
    "TECHNIQUE_KINDS_SINCE_1_2",
    "TECHNIQUE_KIND_LABELS_KO",
    "TECHNIQUE_OPERATION_KIND",
    "TECHNIQUE_PADDLING",
    "TECHNIQUE_PARING",
    "TECHNIQUE_PAYLOAD_EXTENSION_KEY",
    "TECHNIQUE_PAYLOAD_MEDIA_TYPE",
    "TECHNIQUE_PAYLOAD_SCHEMA_VERSION",
    "TECHNIQUE_PAYLOAD_SCHEMA_VERSIONS",
    "SURFACE_EXTERIOR",
    "SURFACE_INTERIOR",
    "SURFACE_MIXED",
    "SURFACE_SIDES",
    "surface_side_of_faces",
    "TECHNIQUE_RECORD_TYPE",
    "TECHNIQUE_VIEWS",
    "TECHNIQUE_WATER_SMOOTHING",
    "TECHNIQUE_WOOD_GRAIN",
    "TechniqueAnnotationComputation",
    "TechniqueAnnotationPayload",
    "append_technique_record_from_context",
    "commit_technique_annotation",
    "compute_technique_annotation",
    "project_technique_from_recipe",
    "technique_computation_matches_active_projection",
    "technique_kind",
    "technique_payload_from_record",
    "technique_recipe",
    "technique_selection",
    "validate_technique_annotation_records",
    "validate_technique_recipe",
]
