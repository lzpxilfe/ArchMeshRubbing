"""능선 record: the ridges a mesh carries, read once and kept with the drawing.

A stone tool's plan shows the ridges between its flake scars as inner lines
([K1] 2013 p. 48).  `artifact_crease_lines` finds them; this module makes
the finding a record - `measurement.crease.v1` - so it is computed under the
document's active alignment, hashed, re-verifiable, and drawn from what was
stored rather than recomputed at print time.

The record holds, for each of the six orthographic views, the polylines
that view can see, in that view's frame, in whole micrometres.  The two
numbers that decide what counts as a ridge - the least dihedral angle and
the least chain length - are in the recipe, so a record keeps determining
its own result if a later release would choose differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_cancellation import CancellationProbe, raise_if_cancelled
from .artifact_crease_lines import (
    CREASE_LINK_ANGLE_DEG,
    CREASE_LINK_ROUNDS,
    CREASE_WELD_MM,
    CREST_RULE_CURVATURE_V2,
    CREST_RULE_TURNING_V1,
    DEFAULT_CREASE_DIHEDRAL_MIN_DEG,
    DEFAULT_CREASE_MIN_LENGTH_MM,
    ArtifactCreaseError,
    creases_seen_from,
    detect_convex_creases,
)
from .artifact_document import (
    ArtifactDocument,
    ArtifactDocumentError,
    DerivedRecord,
    OperationContext,
    RecordLifecycleStatus,
)
from .artifact_outline_extractor import OutlineView
from .artifact_session import ArtifactSession, ArtifactSessionError
from .canonical_json import (
    CanonicalJSONError,
    canonical_json_bytes,
    canonical_json_sha256,
)


CREASE_RECORD_TYPE = "measurement.crease.v1"
CREASE_OPERATION_KIND = "crease"
CREASE_ALGORITHM = "archmeshrubbing.convex_dihedral_crease"
#: 1.0.0 picked the crest of a rounded ridge by each edge's own turning
#: against every neighbour within half the scale, which on a scanned mesh
#: leaves a ridge in fragments; 1.1.0 reads each side's normal over a patch,
#: scores the crest by its height above the chord, lets only edges across
#: the ridge compete, and can join chain ends (``link_um``).  A reading
#: taken under 1.0.0 is recomputed under 1.0.0 (docs/LITHIC_TRIAL.md).
CREASE_LEGACY_ALGORITHM_VERSION = "1.0.0"
CREASE_ALGORITHM_VERSION = "1.1.0"
CREASE_ALGORITHM_VERSIONS = frozenset({CREASE_LEGACY_ALGORITHM_VERSION, CREASE_ALGORITHM_VERSION})
CREASE_COORDINATE_SPACE = "canonical_um_planar_per_view/v1"
CREASE_PAYLOAD_SCHEMA_VERSION = "1.0.0"
CREASE_PAYLOAD_EXTENSION_KEY = "org.archmeshrubbing:crease-v1"
CREASE_PAYLOAD_MEDIA_TYPE = "application/vnd.archmeshrubbing.crease+json"
CREASE_GEOMETRY_REF_PREFIX = "urn:archmeshrubbing:crease:sha256:"
#: The six views, in canonical order.
CREASE_VIEWS: tuple[str, ...] = tuple(sorted(view.value for view in OutlineView))

MIN_CREASE_DIHEDRAL_MILLIDEG = 1_000
MAX_CREASE_DIHEDRAL_MILLIDEG = 179_000
MIN_CREASE_LENGTH_UM = 0
MAX_CREASE_LENGTH_UM = 1_000_000
MAX_CREASE_CHAINS = 4_096
MAX_CREASE_POINTS = 250_000


class ArtifactCreaseRecordError(ValueError):
    """A crease reading cannot be recorded or read back safely."""


def _strict_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactCreaseRecordError(f"{name} must be an integer")
    number = int(value)
    if number < minimum or number > maximum:
        raise ArtifactCreaseRecordError(
            f"{name} must be in the inclusive range {minimum}..{maximum}"
        )
    return number


def _exact_keys(value: object, keys: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactCreaseRecordError(f"{name} must be an object")
    if set(value) != set(keys):
        raise ArtifactCreaseRecordError(
            f"{name} must carry exactly {', '.join(sorted(keys))}"
        )
    return value


def crease_recipe(
    *,
    dihedral_min_deg: float = DEFAULT_CREASE_DIHEDRAL_MIN_DEG,
    min_length_mm: float = DEFAULT_CREASE_MIN_LENGTH_MM,
    scale_mm: float = 0.0,
    link_mm: float = 0.0,
    algorithm_version: str = CREASE_ALGORITHM_VERSION,
) -> dict[str, Any]:
    """Resolve the numbers that decide what counts as a ridge.

    ``scale_mm`` at zero reads the bend between the two faces of an edge;
    above zero it reads the bend between the surface that far to either
    side, for ridges a scan has rounded over.  ``link_mm`` above zero joins
    chain ends that point at each other within that distance; it belongs to
    1.1.0, and the 1.0.0 recipe has no place for it.
    """

    if algorithm_version not in CREASE_ALGORITHM_VERSIONS:
        raise ArtifactCreaseRecordError(
            f"crease algorithm_version must be one of {sorted(CREASE_ALGORITHM_VERSIONS)}"
        )
    try:
        dihedral = float(dihedral_min_deg)
        length = float(min_length_mm)
        scale = float(scale_mm)
        link = float(link_mm)
    except (TypeError, ValueError) as exc:
        raise ArtifactCreaseRecordError("crease thresholds must be numbers") from exc
    if not all(np.isfinite(value) for value in (dihedral, length, scale, link)):
        raise ArtifactCreaseRecordError("crease thresholds must be finite")
    scale_um = _strict_int(
        int(round(scale * 1000.0)),
        name="scale_mm (in micrometres)",
        minimum=0,
        maximum=MAX_CREASE_LENGTH_UM,
    )
    link_um = _strict_int(
        int(round(link * 1000.0)),
        name="link_mm (in micrometres)",
        minimum=0,
        maximum=MAX_CREASE_LENGTH_UM,
    )
    if algorithm_version == CREASE_LEGACY_ALGORITHM_VERSION and link_um != 0:
        raise ArtifactCreaseRecordError(
            f"crease algorithm {CREASE_LEGACY_ALGORITHM_VERSION} does not join chains; "
            "link_mm must be 0"
        )
    dihedral_millideg = _strict_int(
        int(round(dihedral * 1000.0)),
        name="dihedral_min_deg (in millidegrees)",
        minimum=MIN_CREASE_DIHEDRAL_MILLIDEG,
        maximum=MAX_CREASE_DIHEDRAL_MILLIDEG,
    )
    length_um = _strict_int(
        int(round(length * 1000.0)),
        name="min_length_mm (in micrometres)",
        minimum=MIN_CREASE_LENGTH_UM,
        maximum=MAX_CREASE_LENGTH_UM,
    )
    policy: dict[str, Any] = {
        "convexity": "far_corner_below_neighbour_plane/v1",
        "crest": CREST_RULE_TURNING_V1,
        "dihedral_min_millideg": dihedral_millideg,
        "min_length_um": length_um,
        "scale_um": scale_um,
        "weld_um": int(round(CREASE_WELD_MM * 1000.0)),
    }
    if algorithm_version == CREASE_ALGORITHM_VERSION:
        policy["crest"] = CREST_RULE_CURVATURE_V2
        policy["link"] = (
            f"ends_facing_within_{int(round(CREASE_LINK_ANGLE_DEG))}deg_"
            f"{CREASE_LINK_ROUNDS}_rounds/v1"
        )
        policy["link_um"] = link_um
        policy["sample"] = "area_weighted_normals_within_half_scale/v1"
    return {
        "algorithm": CREASE_ALGORITHM,
        "algorithm_version": algorithm_version,
        "chain_order": "longest_first_then_start_point/v1",
        "coordinate_space": CREASE_COORDINATE_SPACE,
        "detection_policy": policy,
        "kind": CREASE_OPERATION_KIND,
        "resource_limits": {
            "max_chains": MAX_CREASE_CHAINS,
            "max_points": MAX_CREASE_POINTS,
        },
        "views": list(CREASE_VIEWS),
        "visibility": "both_faces_toward_viewer/v1",
    }


def validate_crease_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the recipe from its own numbers and require the same bytes."""

    if not isinstance(recipe, Mapping):
        raise ArtifactCreaseRecordError("crease recipe must be an object")
    policy = recipe.get("detection_policy")
    if not isinstance(policy, Mapping):
        raise ArtifactCreaseRecordError("crease recipe detection_policy is invalid")
    version = recipe.get("algorithm_version")
    if not isinstance(version, str) or version not in CREASE_ALGORITHM_VERSIONS:
        raise ArtifactCreaseRecordError(
            f"crease recipe algorithm_version must be one of {sorted(CREASE_ALGORITHM_VERSIONS)}"
        )
    dihedral = policy.get("dihedral_min_millideg")
    length = policy.get("min_length_um")
    scale = policy.get("scale_um")
    link = policy.get("link_um", 0)
    for value in (dihedral, length, scale, link):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ArtifactCreaseRecordError("crease recipe thresholds must be integers")
    assert (
        isinstance(dihedral, int)
        and isinstance(length, int)
        and isinstance(scale, int)
        and isinstance(link, int)
    )
    expected = crease_recipe(
        dihedral_min_deg=dihedral / 1000.0,
        min_length_mm=length / 1000.0,
        scale_mm=scale / 1000.0,
        link_mm=link / 1000.0,
        algorithm_version=version,
    )
    try:
        same = canonical_json_bytes(dict(recipe)) == canonical_json_bytes(expected)
    except CanonicalJSONError as exc:
        raise ArtifactCreaseRecordError(str(exc)) from exc
    if not same:
        raise ArtifactCreaseRecordError(
            "crease recipe does not match the production contract"
        )
    return expected


Polyline = tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class CreaseViewLines:
    """What one view sees of the ridges: polylines in its frame, in µm."""

    view: str
    polylines: tuple[Polyline, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.view, str) or self.view not in CREASE_VIEWS:
            raise ArtifactCreaseRecordError(
                f"crease view must be one of {', '.join(CREASE_VIEWS)}"
            )
        cleaned: list[Polyline] = []
        limit = 10**9
        for polyline in self.polylines:
            points = tuple(
                (
                    _strict_int(x, name="crease point", minimum=-limit, maximum=limit),
                    _strict_int(y, name="crease point", minimum=-limit, maximum=limit),
                )
                for x, y in polyline
            )
            if len(points) < 2:
                raise ArtifactCreaseRecordError("a crease polyline has at least two points")
            if any(a == b for a, b in zip(points, points[1:])):
                raise ArtifactCreaseRecordError("a crease polyline repeats a point")
            cleaned.append(points)
        object.__setattr__(self, "polylines", tuple(cleaned))

    @property
    def point_count(self) -> int:
        return sum(len(polyline) for polyline in self.polylines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "polylines": [[list(point) for point in polyline] for polyline in self.polylines],
            "view": self.view,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CreaseViewLines":
        block = _exact_keys(data, frozenset({"polylines", "view"}), name="crease view")
        raw = block["polylines"]
        if not isinstance(raw, (list, tuple)):
            raise ArtifactCreaseRecordError("crease view polylines must be an array")
        polylines: list[Polyline] = []
        for polyline in raw:
            if not isinstance(polyline, (list, tuple)):
                raise ArtifactCreaseRecordError("a crease polyline must be an array")
            points: list[tuple[int, int]] = []
            for point in polyline:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ArtifactCreaseRecordError("a crease point must be [x, y]")
                points.append((point[0], point[1]))  # type: ignore[arg-type]
            polylines.append(tuple(points))
        view = block["view"]
        return cls(view=view if isinstance(view, str) else "", polylines=tuple(polylines))


@dataclass(frozen=True, slots=True)
class CreasePayload:
    """Every ridge one reading found, and what each view sees of them."""

    schema_version: str
    chain_count: int
    total_length_um: int
    max_dihedral_millideg: int
    views: tuple[CreaseViewLines, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CREASE_PAYLOAD_SCHEMA_VERSION:
            raise ArtifactCreaseRecordError(
                f"unsupported crease payload schema: {self.schema_version!r}"
            )
        _strict_int(self.chain_count, name="chain_count", minimum=1, maximum=MAX_CREASE_CHAINS)
        _strict_int(self.total_length_um, name="total_length_um", minimum=1, maximum=10**12)
        _strict_int(
            self.max_dihedral_millideg,
            name="max_dihedral_millideg",
            minimum=0,
            maximum=180_000,
        )
        views = tuple(self.views)
        if any(not isinstance(view, CreaseViewLines) for view in views):
            raise ArtifactCreaseRecordError("crease views must be CreaseViewLines values")
        if sorted(view.view for view in views) != list(CREASE_VIEWS):
            raise ArtifactCreaseRecordError(
                "a crease payload carries each of the six views exactly once"
            )
        if sum(view.point_count for view in views) > MAX_CREASE_POINTS:
            raise ArtifactCreaseRecordError(
                f"a crease payload holds at most {MAX_CREASE_POINTS} points"
            )
        object.__setattr__(self, "views", tuple(sorted(views, key=lambda item: item.view)))

    def lines_for_view(self, view: str) -> CreaseViewLines | None:
        for lines in self.views:
            if lines.view == view:
                return lines
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_count": self.chain_count,
            "max_dihedral_millideg": self.max_dihedral_millideg,
            "schema_version": self.schema_version,
            "total_length_um": self.total_length_um,
            "views": [view.to_dict() for view in self.views],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CreasePayload":
        block = _exact_keys(
            data,
            frozenset(
                {
                    "chain_count",
                    "max_dihedral_millideg",
                    "schema_version",
                    "total_length_um",
                    "views",
                }
            ),
            name="crease payload",
        )
        raw_views = block["views"]
        if not isinstance(raw_views, (list, tuple)):
            raise ArtifactCreaseRecordError("crease payload views must be an array")
        schema_version = block["schema_version"]
        return cls(
            schema_version=schema_version if isinstance(schema_version, str) else "",
            chain_count=block["chain_count"],  # type: ignore[arg-type]
            total_length_um=block["total_length_um"],  # type: ignore[arg-type]
            max_dihedral_millideg=block["max_dihedral_millideg"],  # type: ignore[arg-type]
            views=tuple(CreaseViewLines.from_dict(view) for view in raw_views),  # type: ignore[arg-type]
        )

    def canonical_json_bytes(self) -> bytes:
        try:
            return canonical_json_bytes(self.to_dict())
        except CanonicalJSONError as exc:
            raise ArtifactCreaseRecordError(str(exc)) from exc

    @property
    def sha256(self) -> str:
        try:
            return canonical_json_sha256(self.to_dict())
        except CanonicalJSONError as exc:
            raise ArtifactCreaseRecordError(str(exc)) from exc

    @property
    def geometry_ref(self) -> str:
        return f"{CREASE_GEOMETRY_REF_PREFIX}{self.sha256}"

    def qc_summary(self) -> dict[str, Any]:
        return {
            "chain_count": self.chain_count,
            "max_dihedral_millideg": self.max_dihedral_millideg,
            "total_length_um": self.total_length_um,
            "view_polyline_counts": {view.view: len(view.polylines) for view in self.views},
        }


def read_creases(
    vertices: object,
    faces: object,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> CreasePayload:
    """Read every ridge the recipe's numbers admit, and what each view sees."""

    validated = validate_crease_recipe(recipe)
    policy = validated["detection_policy"]
    dihedral_min_deg = int(policy["dihedral_min_millideg"]) / 1000.0
    min_length_mm = int(policy["min_length_um"]) / 1000.0
    scale_mm = int(policy["scale_um"]) / 1000.0
    link_mm = int(policy.get("link_um", 0)) / 1000.0
    crest_rule = str(policy["crest"])
    raise_if_cancelled(cancellation_probe)
    try:
        chains = detect_convex_creases(
            vertices,
            faces,
            dihedral_min_deg=dihedral_min_deg,
            min_length_mm=min_length_mm,
            scale_mm=scale_mm,
            link_mm=link_mm,
            crest_rule=crest_rule,
        )
    except ArtifactCreaseError as exc:
        raise ArtifactCreaseRecordError(str(exc)) from exc
    raise_if_cancelled(cancellation_probe)
    if not chains:
        raise ArtifactCreaseRecordError(
            "a crease reading with no ridge records nothing; lower "
            "dihedral_min_deg or min_length_mm, or do not take the reading"
        )
    if len(chains) > MAX_CREASE_CHAINS:
        raise ArtifactCreaseRecordError(
            f"a crease reading holds at most {MAX_CREASE_CHAINS} chains"
        )
    views: list[CreaseViewLines] = []
    for view in CREASE_VIEWS:
        raise_if_cancelled(cancellation_probe)
        polylines: list[Polyline] = []
        for polyline in creases_seen_from(chains, view, min_length_mm=min_length_mm):
            points_um = np.rint(polyline * 1000.0).astype(np.int64)
            points: list[tuple[int, int]] = []
            for x, y in points_um.tolist():
                if points and points[-1] == (x, y):
                    continue
                points.append((int(x), int(y)))
            if len(points) >= 2:
                polylines.append(tuple(points))
        views.append(CreaseViewLines(view=view, polylines=tuple(polylines)))
    return CreasePayload(
        schema_version=CREASE_PAYLOAD_SCHEMA_VERSION,
        chain_count=len(chains),
        total_length_um=int(round(sum(chain.length_mm for chain in chains) * 1000.0)),
        max_dihedral_millideg=int(
            round(max(chain.max_dihedral_deg for chain in chains) * 1000.0)
        ),
        views=tuple(views),
    )


@dataclass(frozen=True, slots=True)
class CreaseComputation:
    context: OperationContext
    projection_snapshot: Any
    payload: CreasePayload
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]


def compute_crease_reading(
    session: ArtifactSession,
    *,
    dihedral_min_deg: float = DEFAULT_CREASE_DIHEDRAL_MIN_DEG,
    min_length_mm: float = DEFAULT_CREASE_MIN_LENGTH_MM,
    scale_mm: float = 0.0,
    link_mm: float = 0.0,
    cancellation_probe: CancellationProbe | None = None,
) -> CreaseComputation:
    """Read the ridges of the artifact as positioned by its active Align."""

    if not isinstance(session, ArtifactSession):
        raise ArtifactCreaseRecordError("session must be an ArtifactSession")
    recipe = crease_recipe(
        dihedral_min_deg=dihedral_min_deg,
        min_length_mm=min_length_mm,
        scale_mm=scale_mm,
        link_mm=link_mm,
    )
    try:
        context = session.capture_operation(recipe=recipe)
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactCreaseRecordError(str(exc)) from exc
    payload = read_creases(
        projection.mesh.vertices,
        projection.mesh.faces,
        recipe,
        cancellation_probe=cancellation_probe,
    )
    return CreaseComputation(
        context=context,
        projection_snapshot=projection.snapshot,
        payload=payload,
        recipe=recipe,
        qc=payload.qc_summary(),
    )


def crease_computation_matches_active_projection(
    session: ArtifactSession, computation: CreaseComputation
) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(
        computation, CreaseComputation
    ):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    return current.render_key == computation.projection_snapshot.render_key


def append_crease_record_from_context(
    document: ArtifactDocument,
    *,
    context: OperationContext,
    payload: CreasePayload,
    recipe: Mapping[str, Any],
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactDocument:
    """Append one verified crease reading without touching source geometry."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactCreaseRecordError("document must be an ArtifactDocument")
    if not isinstance(context, OperationContext):
        raise ArtifactCreaseRecordError("context must be an OperationContext")
    if not isinstance(payload, CreasePayload):
        raise ArtifactCreaseRecordError("payload must be a CreasePayload")
    validated_recipe = validate_crease_recipe(recipe)
    payload_bytes = payload.canonical_json_bytes()
    extensions = {
        CREASE_PAYLOAD_EXTENSION_KEY: {
            "byte_length": len(payload_bytes),
            "media_type": CREASE_PAYLOAD_MEDIA_TYPE,
            "payload": payload.to_dict(),
            "schema_version": CREASE_PAYLOAD_SCHEMA_VERSION,
            "sha256": payload.sha256,
        }
    }
    try:
        return document.append_record_from_context(
            context=context,
            id=record_id,
            type=CREASE_RECORD_TYPE,
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
        raise ArtifactCreaseRecordError(str(exc)) from exc


def commit_crease_reading(
    session: ArtifactSession,
    computation: CreaseComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    if not crease_computation_matches_active_projection(session, computation):
        raise ArtifactCreaseRecordError(
            "crease computation is stale for the active projection"
        )
    document = append_crease_record_from_context(
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


def crease_payload_from_record(record: DerivedRecord) -> CreasePayload:
    """Resolve and re-verify one crease record's inline reading."""

    if not isinstance(record, DerivedRecord):
        raise ArtifactCreaseRecordError("record must be a DerivedRecord")
    if record.type != CREASE_RECORD_TYPE:
        raise ArtifactCreaseRecordError(f"record is not a crease reading: {record.type!r}")
    descriptor = _exact_keys(
        record.extensions.get(CREASE_PAYLOAD_EXTENSION_KEY),
        _DESCRIPTOR_KEYS,
        name="crease payload descriptor",
    )
    if descriptor["media_type"] != CREASE_PAYLOAD_MEDIA_TYPE:
        raise ArtifactCreaseRecordError("crease payload media_type is invalid")
    if descriptor["schema_version"] != CREASE_PAYLOAD_SCHEMA_VERSION:
        raise ArtifactCreaseRecordError("crease payload descriptor schema is invalid")
    raw_payload = descriptor["payload"]
    if not isinstance(raw_payload, Mapping):
        raise ArtifactCreaseRecordError("crease payload descriptor payload must be an object")
    payload = CreasePayload.from_dict(raw_payload)
    payload_bytes = payload.canonical_json_bytes()
    byte_length = descriptor["byte_length"]
    if type(byte_length) is not int or byte_length != len(payload_bytes):
        raise ArtifactCreaseRecordError("crease payload byte_length does not match payload")
    if descriptor["sha256"] != payload.sha256:
        raise ArtifactCreaseRecordError("crease payload SHA-256 does not match payload")
    if record.geometry_ref != payload.geometry_ref:
        raise ArtifactCreaseRecordError("crease record geometry_ref does not match payload")
    validate_crease_recipe(record.recipe)
    thawed_qc = record.to_dict()["qc"]
    assert isinstance(thawed_qc, dict)
    if thawed_qc != payload.qc_summary():
        raise ArtifactCreaseRecordError("crease record QC does not match its payload")
    return payload


def validate_crease_records(document: ArtifactDocument) -> None:
    """Strictly validate every crease record embedded in a document."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactCreaseRecordError("document must be an ArtifactDocument")
    for record in document.records:
        if record.type == CREASE_RECORD_TYPE:
            crease_payload_from_record(record)


__all__ = [
    "ArtifactCreaseRecordError",
    "CREASE_ALGORITHM",
    "CREASE_ALGORITHM_VERSION",
    "CREASE_ALGORITHM_VERSIONS",
    "CREASE_LEGACY_ALGORITHM_VERSION",
    "CREASE_PAYLOAD_EXTENSION_KEY",
    "CREASE_PAYLOAD_MEDIA_TYPE",
    "CREASE_PAYLOAD_SCHEMA_VERSION",
    "CREASE_RECORD_TYPE",
    "CREASE_VIEWS",
    "CreaseComputation",
    "CreasePayload",
    "CreaseViewLines",
    "MAX_CREASE_CHAINS",
    "MAX_CREASE_POINTS",
    "append_crease_record_from_context",
    "commit_crease_reading",
    "compute_crease_reading",
    "crease_computation_matches_active_projection",
    "crease_payload_from_record",
    "crease_recipe",
    "read_creases",
    "validate_crease_recipe",
    "validate_crease_records",
]
