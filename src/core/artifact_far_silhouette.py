"""The silhouette of the half behind the view plane: what shows through the cut.

A half-elevation, half-section figure cuts the vessel on the plane of the
view through its axis and shows the near half's outside on one side and
the wall's cut on the other.  Through that cut the reader sees the inside
of the far half, and its edges - the rim going round the back, the far
foot's floor - end where the far half ends.  On a level vessel those edges
are level lines at the rim's and the floor's heights, and a straight run
of the elevation's edge past the fold draws them.  On a warped vessel the
rim is not level and the line that goes round the back is the far half's
own silhouette: this record measures it, as the outline of exactly the
faces whose centre lies behind the view plane, with the outline extractor
and its fixed grid, so the far edge is measured the way the near one is.

The record is the outline of a face subset and nothing more; which of its
edges the drawing shows, and where each stops short of the cut, is the
sheet's decision (``drawing_sheet.DrawingSheetOptions.outline_reach``).
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
    OutlineView,
    extract_outline_geometry,
    outline_frame,
    outline_recipe,
    validate_outline_record_contract,
)
from .artifact_session import ArtifactSession, ArtifactSessionError
from .artifact_vector_extractor import ArtifactVectorExtractionError
from .artifact_vector_record import (
    VECTOR_COORDINATE_SPACE,
    VECTOR_PAYLOAD_SCHEMA_VERSION,
    ArtifactVectorRecordError,
    VectorGeometryPayload,
    VectorRecordKind,
)
from .canonical_json import CanonicalJSONError, canonical_json_bytes

FAR_SILHOUETTE_RECORD_TYPE = "measurement.far_silhouette.v1"
FAR_SILHOUETTE_OPERATION_KIND = "far_silhouette"
FAR_SILHOUETTE_ALGORITHM = "archmeshrubbing.far_silhouette"
FAR_SILHOUETTE_ALGORITHM_VERSION = "1.0.0"
FAR_SILHOUETTE_PAYLOAD_EXTENSION_KEY = "org.archmeshrubbing:far-silhouette-v1"
FAR_SILHOUETTE_PAYLOAD_MEDIA_TYPE = "application/vnd.archmeshrubbing.far-silhouette+json"
FAR_SILHOUETTE_GEOMETRY_REF_PREFIX = "urn:archmeshrubbing:far-silhouette:sha256:"

#: The faces the silhouette is taken of: those whose centre lies strictly on
#: the side of the view plane away from the viewer.  A face the plane passes
#: through goes with the side its centre is on; one centred exactly on the
#: plane is the near half's.
FAR_SILHOUETTE_SIDE = "behind_view_plane/v1"
FAR_SILHOUETTE_MEMBERSHIP = "face_centroid_strictly_behind_plane/v1"

_DESCRIPTOR_KEYS = frozenset({"byte_length", "media_type", "payload", "schema_version", "sha256"})


class ArtifactFarSilhouetteError(ValueError):
    """Raised when a far silhouette cannot be read, stored, or verified."""


def _outline_view(value: object) -> OutlineView:
    if isinstance(value, OutlineView):
        return value
    try:
        return OutlineView(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactFarSilhouetteError(f"unsupported outline view: {value!r}") from exc


def far_silhouette_recipe(view: OutlineView | str, *, precision_grid_mm: float) -> dict[str, Any]:
    """The outline contract of the far half: the view, the plane the half is
    behind, and the whole outline recipe the faces are put through."""

    resolved = _outline_view(view)
    try:
        outline = outline_recipe(resolved, precision_grid_mm=precision_grid_mm)
    except ArtifactVectorExtractionError as exc:
        raise ArtifactFarSilhouetteError(str(exc)) from exc
    frame = outline_frame(resolved)
    return {
        "algorithm": FAR_SILHOUETTE_ALGORITHM,
        "algorithm_version": FAR_SILHOUETTE_ALGORITHM_VERSION,
        "coordinate_space": VECTOR_COORDINATE_SPACE,
        "face_scope": {
            "membership": FAR_SILHOUETTE_MEMBERSHIP,
            "plane": {
                "normal_world": [float(value) for value in frame.normal_world],
                "origin_world_mm": [float(value) for value in frame.origin_world_mm],
            },
            "side": FAR_SILHOUETTE_SIDE,
        },
        "kind": FAR_SILHOUETTE_OPERATION_KIND,
        "outline": outline,
        "view": resolved.value,
    }


def validate_far_silhouette_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the recipe from its own view and grid and require the same bytes."""

    if not isinstance(recipe, Mapping):
        raise ArtifactFarSilhouetteError("far silhouette recipe must be an object")
    outline = recipe.get("outline")
    if not isinstance(outline, Mapping):
        raise ArtifactFarSilhouetteError("far silhouette recipe carries the outline recipe it ran")
    grid = outline.get("precision_grid_mm")
    if isinstance(grid, bool) or not isinstance(grid, (int, float)):
        raise ArtifactFarSilhouetteError("far silhouette recipe outline.precision_grid_mm must be a number")
    expected = far_silhouette_recipe(recipe.get("view"), precision_grid_mm=float(grid))  # type: ignore[arg-type]
    try:
        same = canonical_json_bytes(dict(recipe)) == canonical_json_bytes(expected)
    except CanonicalJSONError as exc:
        raise ArtifactFarSilhouetteError(str(exc)) from exc
    if not same:
        raise ArtifactFarSilhouetteError("far silhouette recipe does not match the production contract")
    return expected


def far_silhouette_view(recipe: Mapping[str, Any]) -> OutlineView:
    """The view a validated far silhouette recipe was taken in."""

    return OutlineView(validate_far_silhouette_recipe(recipe)["view"])


def far_face_mask(vertices: np.ndarray, faces: np.ndarray, view: OutlineView | str) -> np.ndarray:
    """Which faces lie behind the view plane: centre strictly on the side the
    frame's normal points away from.  The normal points at the viewer."""

    frame = outline_frame(_outline_view(view))
    origin = np.asarray(frame.origin_world_mm, dtype=np.float64)
    normal = np.asarray(frame.normal_world, dtype=np.float64)
    centres = vertices[faces].mean(axis=1)
    return (centres - origin) @ normal < 0.0


def far_silhouette_geometry_ref(payload: VectorGeometryPayload) -> str:
    return f"{FAR_SILHOUETTE_GEOMETRY_REF_PREFIX}{payload.sha256}"


def extract_far_silhouette(
    vertices_world_mm: object,
    faces: object,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[VectorGeometryPayload, dict[str, Any]]:
    """The outline of the faces behind the view plane, and its QC."""

    raise_if_cancelled(cancellation_probe)
    validated = validate_far_silhouette_recipe(recipe)
    view = OutlineView(validated["view"])
    grid = float(validated["outline"]["precision_grid_mm"])
    try:
        vertices = np.asarray(vertices_world_mm, dtype=np.float64)
        face_array = np.asarray(faces, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ArtifactFarSilhouetteError("vertices and faces must be numeric arrays") from exc
    if vertices.ndim != 2 or vertices.shape[1] != 3 or face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ArtifactFarSilhouetteError("vertices must be Nx3 and faces Mx3")
    if face_array.shape[0] == 0:
        raise ArtifactFarSilhouetteError("a far silhouette needs faces")
    if face_array.min() < 0 or face_array.max() >= vertices.shape[0]:
        raise ArtifactFarSilhouetteError("faces reference vertices that do not exist")
    behind = far_face_mask(vertices, face_array, view)
    far_count = int(behind.sum())
    if far_count == 0:
        raise ArtifactFarSilhouetteError(
            f"no face lies behind the {view.value} view's plane, so there is no far half to take "
            "the silhouette of"
        )
    raise_if_cancelled(cancellation_probe)
    try:
        geometry = extract_outline_geometry(
            vertices,
            face_array[behind],
            view,
            precision_grid_mm=grid,
            cancellation_probe=cancellation_probe,
        )
    except ArtifactVectorExtractionError as exc:
        raise ArtifactFarSilhouetteError(f"the far half's silhouette could not be taken: {exc}") from exc
    payload = geometry.payload
    qc = {
        **dict(geometry.qc),
        "far_face_count": far_count,
        "near_face_count": int(face_array.shape[0]) - far_count,
        "payload_sha256": payload.sha256,
    }
    return payload, qc


@dataclass(frozen=True, slots=True)
class FarSilhouetteComputation:
    context: OperationContext
    projection_snapshot: Any
    payload: VectorGeometryPayload
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]

    def recipe_dict(self) -> dict[str, Any]:
        return dict(self.recipe)

    def qc_dict(self) -> dict[str, Any]:
        return dict(self.qc)


def compute_artifact_far_silhouette(
    session: ArtifactSession,
    view: OutlineView | str,
    *,
    precision_grid_mm: float,
    cancellation_probe: CancellationProbe | None = None,
) -> FarSilhouetteComputation:
    """Take the silhouette of the half of the artifact behind one view's plane."""

    if not isinstance(session, ArtifactSession):
        raise ArtifactFarSilhouetteError("session must be an ArtifactSession")
    recipe = far_silhouette_recipe(view, precision_grid_mm=precision_grid_mm)
    try:
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactFarSilhouetteError(str(exc)) from exc
    payload, qc = extract_far_silhouette(
        projection.mesh.vertices, projection.mesh.faces, recipe, cancellation_probe=cancellation_probe
    )
    try:
        context = session.capture_operation(recipe=recipe, selection_hash=payload.sha256)
    except ArtifactSessionError as exc:
        raise ArtifactFarSilhouetteError(str(exc)) from exc
    return FarSilhouetteComputation(
        context=context,
        projection_snapshot=projection.snapshot,
        payload=payload,
        recipe=recipe,
        qc=qc,
    )


def far_silhouette_computation_matches_active_projection(
    session: ArtifactSession, computation: FarSilhouetteComputation
) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(computation, FarSilhouetteComputation):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    return current.render_key == computation.projection_snapshot.render_key


def append_far_silhouette_record_from_context(
    document: ArtifactDocument,
    *,
    context: OperationContext,
    payload: VectorGeometryPayload,
    recipe: Mapping[str, Any],
    qc: Mapping[str, Any],
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactDocument:
    """Append one verified far silhouette without touching source geometry."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactFarSilhouetteError("document must be an ArtifactDocument")
    if not isinstance(context, OperationContext):
        raise ArtifactFarSilhouetteError("context must be an OperationContext")
    if not isinstance(payload, VectorGeometryPayload):
        raise ArtifactFarSilhouetteError("payload must be a VectorGeometryPayload")
    validated_recipe = validate_far_silhouette_recipe(recipe)
    _check_payload_against_recipe(payload, validated_recipe)
    if context.selection_hash != payload.sha256:
        raise ArtifactFarSilhouetteError("far silhouette context selection_hash does not match the payload")
    try:
        payload_bytes = payload.canonical_json_bytes()
    except ArtifactVectorRecordError as exc:
        raise ArtifactFarSilhouetteError(str(exc)) from exc
    extensions = {
        FAR_SILHOUETTE_PAYLOAD_EXTENSION_KEY: {
            "byte_length": len(payload_bytes),
            "media_type": FAR_SILHOUETTE_PAYLOAD_MEDIA_TYPE,
            "payload": payload.to_dict(),
            "schema_version": VECTOR_PAYLOAD_SCHEMA_VERSION,
            "sha256": payload.sha256,
        }
    }
    try:
        return document.append_record_from_context(
            context=context,
            id=record_id,
            type=FAR_SILHOUETTE_RECORD_TYPE,
            geometry_ref=far_silhouette_geometry_ref(payload),
            recipe=dict(validated_recipe),
            qc=dict(qc),
            lifecycle_status=RecordLifecycleStatus.READY,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=depends_on_record_ids,
            extensions=extensions,
        )
    except ArtifactDocumentError as exc:
        raise ArtifactFarSilhouetteError(str(exc)) from exc


def commit_far_silhouette(
    session: ArtifactSession,
    computation: FarSilhouetteComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    if not far_silhouette_computation_matches_active_projection(session, computation):
        raise ArtifactFarSilhouetteError("far silhouette computation is stale for the active projection")
    document = append_far_silhouette_record_from_context(
        session.document,
        context=computation.context,
        payload=computation.payload,
        recipe=computation.recipe,
        qc=computation.qc,
        record_id=record_id,
        created_at=created_at,
        operator=operator,
        depends_on_record_ids=depends_on_record_ids,
    )
    return session.with_document(document)


def _check_payload_against_recipe(payload: VectorGeometryPayload, recipe: Mapping[str, Any]) -> None:
    """The payload is an outline in the recipe's view, on its grid, in the
    extractor's canonical form - re-proved with the outline contract."""

    if VectorRecordKind(payload.kind) is not VectorRecordKind.OUTLINE:
        raise ArtifactFarSilhouetteError("a far silhouette payload is an outline payload")
    view = OutlineView(recipe["view"])
    if payload.frame != outline_frame(view):
        raise ArtifactFarSilhouetteError("far silhouette payload frame does not match its view")
    try:
        validate_outline_record_contract(payload, recipe["outline"])
    except ArtifactVectorExtractionError as exc:
        raise ArtifactFarSilhouetteError(f"far silhouette payload is not a canonical outline: {exc}") from exc


def _exact_keys(value: object, keys: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactFarSilhouetteError(f"{name} must be an object")
    if set(value) != keys:
        raise ArtifactFarSilhouetteError(f"{name} must carry exactly {', '.join(sorted(keys))}")
    return value


def far_silhouette_payload_from_record(record: DerivedRecord) -> VectorGeometryPayload:
    """Resolve and re-verify one far silhouette record's inline outline."""

    if not isinstance(record, DerivedRecord):
        raise ArtifactFarSilhouetteError("record must be a DerivedRecord")
    if record.type != FAR_SILHOUETTE_RECORD_TYPE:
        raise ArtifactFarSilhouetteError(f"record is not a far silhouette: {record.type!r}")
    descriptor = _exact_keys(
        record.extensions.get(FAR_SILHOUETTE_PAYLOAD_EXTENSION_KEY),
        _DESCRIPTOR_KEYS,
        name="far silhouette payload descriptor",
    )
    if descriptor["media_type"] != FAR_SILHOUETTE_PAYLOAD_MEDIA_TYPE:
        raise ArtifactFarSilhouetteError("far silhouette payload media_type is invalid")
    if descriptor["schema_version"] != VECTOR_PAYLOAD_SCHEMA_VERSION:
        raise ArtifactFarSilhouetteError("far silhouette payload descriptor schema is invalid")
    raw_payload = descriptor["payload"]
    if not isinstance(raw_payload, Mapping):
        raise ArtifactFarSilhouetteError("far silhouette payload descriptor payload must be an object")
    try:
        payload = VectorGeometryPayload.from_dict(raw_payload)
        payload_bytes = payload.canonical_json_bytes()
    except ArtifactVectorRecordError as exc:
        raise ArtifactFarSilhouetteError(str(exc)) from exc
    byte_length = descriptor["byte_length"]
    if type(byte_length) is not int or byte_length != len(payload_bytes):
        raise ArtifactFarSilhouetteError("far silhouette payload byte_length does not match payload")
    if descriptor["sha256"] != payload.sha256:
        raise ArtifactFarSilhouetteError("far silhouette payload SHA-256 does not match payload")
    if record.geometry_ref != far_silhouette_geometry_ref(payload):
        raise ArtifactFarSilhouetteError("far silhouette record geometry_ref does not match payload")
    recipe = validate_far_silhouette_recipe(record.recipe)
    _check_payload_against_recipe(payload, recipe)
    return payload


def validate_far_silhouette_records(document: ArtifactDocument) -> None:
    """Strictly validate every far silhouette embedded in a document."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactFarSilhouetteError("document must be an ArtifactDocument")
    for record in document.records:
        if record.type == FAR_SILHOUETTE_RECORD_TYPE:
            far_silhouette_payload_from_record(record)


__all__ = [
    "ArtifactFarSilhouetteError",
    "FAR_SILHOUETTE_ALGORITHM",
    "FAR_SILHOUETTE_ALGORITHM_VERSION",
    "FAR_SILHOUETTE_MEMBERSHIP",
    "FAR_SILHOUETTE_PAYLOAD_EXTENSION_KEY",
    "FAR_SILHOUETTE_RECORD_TYPE",
    "FAR_SILHOUETTE_SIDE",
    "FarSilhouetteComputation",
    "append_far_silhouette_record_from_context",
    "commit_far_silhouette",
    "compute_artifact_far_silhouette",
    "extract_far_silhouette",
    "far_face_mask",
    "far_silhouette_computation_matches_active_projection",
    "far_silhouette_payload_from_record",
    "far_silhouette_recipe",
    "far_silhouette_view",
    "validate_far_silhouette_recipe",
    "validate_far_silhouette_records",
]
