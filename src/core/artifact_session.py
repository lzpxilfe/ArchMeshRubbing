"""Headless command/session boundary for one authoritative artifact.

The session owns an immutable source-space mesh snapshot and an
``ArtifactDocument``.  UI transforms are previews until ``commit_preview``
appends a proper-rigid Align revision.  Every projection is rematerialized
from source geometry, so commit/undo cycles cannot accumulate vertex drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
import uuid

import numpy as np

from .alignment_utils import compose_align_matrices, scene_trs_matrix_about_pivot
from .artifact_document import (
    ARTIFACT_DOCUMENT_SCHEMA_VERSION,
    GEOMETRY_HASH_SCOPE_V1,
    AlignRevision,
    ArtifactDocument,
    ArtifactDocumentError,
    GeometryRevision,
    Handedness,
    MetadataConfirmationStatus,
    OperationContext,
    SourceAsset,
    SourceMetadataRevision,
    source_to_canonical_mm_matrix,
)
from .artifact_scene_adapter import (
    ArtifactProjectionSnapshot,
    ArtifactSceneAdapter,
    ArtifactSceneProjection,
    VerifiedGeometryIdentity,
)
from .artifact_vector_record import (
    ArtifactVectorRecordError,
    VectorGeometryPayload,
    append_vector_record_from_context,
)
from .geometry_identity import mesh_geometry_sha256
from .mesh_loader import MeshData


class ArtifactSessionError(ValueError):
    """The session command would violate the document/source boundary."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4()}"


def _clone_optional_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    copied = np.asarray(value).copy()
    copied.setflags(write=False)
    return copied


def immutable_source_mesh(mesh: MeshData) -> MeshData:
    """Return a disjoint source-space snapshot with read-only array payloads."""

    if not isinstance(mesh, MeshData):
        raise ArtifactSessionError("mesh must be a MeshData")
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int32).copy()
    vertices.setflags(write=False)
    faces.setflags(write=False)
    snapshot = MeshData(
        vertices=vertices,
        faces=faces,
        normals=_clone_optional_array(mesh.normals),
        face_normals=_clone_optional_array(mesh.face_normals),
        uv_coords=_clone_optional_array(mesh.uv_coords),
        texture=_clone_optional_array(mesh.texture),
        unit=str(mesh.unit),
        filepath=Path(mesh.filepath) if mesh.filepath is not None else None,
        source_identity=mesh.source_identity,
        source_format=mesh.source_format,
    )
    # MeshData normalization may replace arrays, so freeze the final arrays.
    snapshot.vertices.setflags(write=False)
    snapshot.faces.setflags(write=False)
    for value in (
        snapshot.normals,
        snapshot.face_normals,
        snapshot.uv_coords,
        snapshot.texture,
    ):
        if value is not None:
            value.setflags(write=False)
    return snapshot


def _media_type(source_format: str | None) -> str:
    return {
        "ply": "model/ply",
        "obj": "model/obj",
        "stl": "model/stl",
        "off": "model/vnd.off",
        "gltf": "model/gltf+json",
        "glb": "model/gltf-binary",
    }.get(str(source_format or "").lower(), "application/octet-stream")


def _matrix4x4_tuple(
    value: np.ndarray,
) -> tuple[tuple[float, float, float, float], ...]:
    matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    return tuple(
        (
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
        )
        for row in matrix
    )


@dataclass(frozen=True, slots=True)
class ArtifactSession:
    document: ArtifactDocument
    source_mesh: MeshData
    verified_geometry: VerifiedGeometryIdentity
    resolved_source_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.document, ArtifactDocument):
            raise ArtifactSessionError("document must be an ArtifactDocument")
        if not isinstance(self.source_mesh, MeshData):
            raise ArtifactSessionError("source_mesh must be a MeshData")
        resolved = str(self.resolved_source_path or "").strip()
        if not resolved:
            raise ArtifactSessionError("resolved_source_path must be non-empty")
        object.__setattr__(self, "resolved_source_path", resolved)
        self._require_unchanged_source_geometry()
        try:
            from .artifact_record_validation import (  # noqa: PLC0415
                validate_known_records,
            )

            validate_known_records(self.document)
            snapshot = ArtifactSceneAdapter(self.document).capture_snapshot(
                self.verified_geometry
            )
            ArtifactSceneAdapter(self.document).validate_source_mesh(
                self.source_mesh,
                snapshot,
            )
        except (ArtifactVectorRecordError, ValueError) as exc:
            raise ArtifactSessionError(str(exc)) from exc

    def _require_unchanged_source_geometry(self) -> None:
        try:
            computed = mesh_geometry_sha256(
                self.source_mesh,
                scope=self.verified_geometry.geometry_hash_scope,
            )
        except ValueError as exc:
            raise ArtifactSessionError(str(exc)) from exc
        if computed != self.verified_geometry.geometry_sha256:
            raise ArtifactSessionError(
                "source geometry SHA-256 no longer matches verified geometry"
            )

    @classmethod
    def create_from_source(
        cls,
        mesh: MeshData,
        *,
        resolved_source_path: str,
        unit: str,
        axes: Mapping[str, str],
        handedness: Handedness | str,
        software_version: str,
        operator: str,
        created_at: str | None = None,
        document_id: str | None = None,
        metadata_revision_id: str | None = None,
        align_revision_id: str | None = None,
    ) -> "ArtifactSession":
        source = immutable_source_mesh(mesh)
        fingerprint = source.source_identity
        if fingerprint is None:
            raise ArtifactSessionError("source mesh has no immutable source fingerprint")
        source_format = str(source.source_format or "").strip().lower().removeprefix(".")
        if not source_format:
            raise ArtifactSessionError(
                "source mesh has no parser format for deterministic reopen"
            )
        resolved_path = str(Path(resolved_source_path).expanduser().resolve(strict=False))
        timestamp = str(created_at or _utc_now())
        geometry_sha256 = mesh_geometry_sha256(source)
        geometry_id = f"geometry:sha256:{geometry_sha256}"
        metadata_id = metadata_revision_id or _new_id("metadata")
        align_id = align_revision_id or _new_id("align")
        source_asset = SourceAsset.from_fingerprint(
            fingerprint,
            asset_ref=f"external:{fingerprint.original_name}",
            media_type=_media_type(source.source_format),
        )
        geometry = GeometryRevision(
            id=geometry_id,
            source_asset_ids=(source_asset.id,),
            geometry_sha256=geometry_sha256,
            geometry_hash_scope=GEOMETRY_HASH_SCOPE_V1,
            import_recipe={
                "format": source_format,
                "loader": "trimesh",
                "maintain_order": True,
                "process": False,
                "sanitizer": "meshdata-v1",
            },
            topology_map_ref=None,
            qc={
                "face_count": int(source.faces.shape[0]),
                "finite_vertices": True,
                "vertex_count": int(source.vertices.shape[0]),
            },
            created_at=timestamp,
            operator=operator,
        )
        metadata = SourceMetadataRevision(
            id=metadata_id,
            parent_id=None,
            geometry_revision_id=geometry.id,
            unit=unit,
            axes=dict(axes),
            handedness=handedness,
            confirmation_status=MetadataConfirmationStatus.CONFIRMED,
            source_to_canonical_mm=_matrix4x4_tuple(
                source_to_canonical_mm_matrix(unit, axes)
            ),
            created_at=timestamp,
            operator=operator,
        )
        align = AlignRevision(
            id=align_id,
            parent_id=None,
            source_metadata_revision_id=metadata.id,
            matrix4x4=_matrix4x4_tuple(np.eye(4, dtype=np.float64)),
            recipe={"kind": "initial_identity"},
            qc={"proper_rigid": True},
            created_at=timestamp,
            operator=operator,
        )
        document = ArtifactDocument(
            schema_version=ARTIFACT_DOCUMENT_SCHEMA_VERSION,
            document_id=document_id or f"urn:uuid:{uuid.uuid4()}",
            software_version=software_version,
            source_assets=(source_asset,),
            geometry_revisions=(geometry,),
            source_metadata_revisions=(metadata,),
            align_revisions=(align,),
            active_source_metadata_revision_id=metadata.id,
            active_align_revision_id=align.id,
            records=(),
        )
        verified = VerifiedGeometryIdentity(
            source_asset_id=source_asset.id,
            geometry_revision_id=geometry.id,
            geometry_sha256=geometry.geometry_sha256,
            geometry_hash_scope=geometry.geometry_hash_scope,
        )
        return cls(
            document=document,
            source_mesh=source,
            verified_geometry=verified,
            resolved_source_path=resolved_path,
        )

    @classmethod
    def bind_loaded_document(
        cls,
        document: ArtifactDocument,
        mesh: MeshData,
        *,
        resolved_source_path: str,
    ) -> "ArtifactSession":
        source = immutable_source_mesh(mesh)
        active_metadata_id = document.active_source_metadata_revision_id
        if active_metadata_id is None:
            raise ArtifactSessionError("loaded document has no active metadata revision")
        metadata = document.source_metadata_revision_index[active_metadata_id]
        geometry = document.geometry_revision_index[metadata.geometry_revision_id]
        if len(geometry.source_asset_ids) != 1:
            raise ArtifactSessionError("M0-3 supports exactly one source asset")
        source_asset = document.source_asset_index[geometry.source_asset_ids[0]]
        fingerprint = source.source_identity
        if fingerprint is None:
            raise ArtifactSessionError(
                "loaded source has no immutable source fingerprint"
            )
        if (
            fingerprint.identity_scope != source_asset.identity_scope
            or fingerprint.sha256 != source_asset.sha256
            or fingerprint.size_bytes != source_asset.size_bytes
        ):
            raise ArtifactSessionError(
                "loaded source bytes do not match the ArtifactDocument source identity"
            )
        saved_format = str(geometry.import_recipe.get("format", "") or "").strip().lower()
        actual_format = str(source.source_format or "").strip().lower().removeprefix(".")
        if not saved_format or actual_format != saved_format:
            raise ArtifactSessionError(
                "loaded source parser format does not match the ArtifactDocument import recipe"
            )
        computed = mesh_geometry_sha256(source, scope=geometry.geometry_hash_scope)
        verified = VerifiedGeometryIdentity(
            source_asset_id=geometry.source_asset_ids[0],
            geometry_revision_id=geometry.id,
            geometry_sha256=computed,
            geometry_hash_scope=geometry.geometry_hash_scope,
        )
        return cls(
            document=document,
            source_mesh=source,
            verified_geometry=verified,
            resolved_source_path=str(resolved_source_path),
        )

    def materialize(self) -> ArtifactSceneProjection:
        self.projection_snapshot()
        return ArtifactSceneAdapter(self.document).materialize(
            self.source_mesh,
            self.verified_geometry,
        )

    def projection_snapshot(self) -> ArtifactProjectionSnapshot:
        """Validate source geometry and return the current immutable binding."""

        self._require_unchanged_source_geometry()
        adapter = ArtifactSceneAdapter(self.document)
        snapshot = adapter.capture_snapshot(self.verified_geometry)
        adapter.validate_source_mesh(self.source_mesh, snapshot)
        return snapshot

    def with_document(self, document: ArtifactDocument) -> "ArtifactSession":
        return ArtifactSession(
            document=document,
            source_mesh=self.source_mesh,
            verified_geometry=self.verified_geometry,
            resolved_source_path=self.resolved_source_path,
        )

    def commit_preview(
        self,
        *,
        translation_mm: np.ndarray | list[float] | tuple[float, ...],
        rotation_deg: np.ndarray | list[float] | tuple[float, ...],
        scale: float,
        pivot_mm: np.ndarray | list[float] | tuple[float, ...] = (0.0, 0.0, 0.0),
        operator: str,
        created_at: str | None = None,
        revision_id: str | None = None,
    ) -> "ArtifactSession":
        if not np.isclose(float(scale), 1.0, rtol=0.0, atol=1e-12):
            raise ArtifactSessionError(
                "native Align preview cannot contain scale; source units belong to metadata"
            )
        parent_id = self.document.active_align_revision_id
        if parent_id is None:
            raise ArtifactSessionError("an active Align revision is required")
        parent = self.document.align_revision_index[parent_id]
        try:
            delta = scene_trs_matrix_about_pivot(
                translation_mm,
                rotation_deg,
                1.0,
                pivot_mm,
            )
            matrix = compose_align_matrices(delta, parent.matrix)
        except (TypeError, ValueError) as exc:
            raise ArtifactSessionError(str(exc)) from exc
        translation = np.asarray(translation_mm, dtype=np.float64).reshape(-1)[:3]
        rotation = np.asarray(rotation_deg, dtype=np.float64).reshape(-1)[:3]
        pivot = np.asarray(pivot_mm, dtype=np.float64).reshape(-1)[:3]
        revision = AlignRevision(
            id=revision_id or _new_id("align"),
            parent_id=parent.id,
            source_metadata_revision_id=parent.source_metadata_revision_id,
            matrix4x4=_matrix4x4_tuple(matrix),
            recipe={
                "convention": "delta @ parent",
                "kind": "manual_scene_trs_delta",
                "rotation_deg": rotation.tolist(),
                "translation_mm": translation.tolist(),
                "pivot_mm": pivot.tolist(),
            },
            qc={"proper_rigid": True},
            created_at=str(created_at or _utc_now()),
            operator=operator,
        )
        return self.with_document(self.document.append_align_revision(revision))

    def activate_align(self, revision_id: str) -> "ArtifactSession":
        try:
            document = self.document.activate_align_revision(revision_id)
        except ArtifactDocumentError as exc:
            raise ArtifactSessionError(str(exc)) from exc
        return self.with_document(document)

    def activate_parent_align(self) -> "ArtifactSession":
        active_id = self.document.active_align_revision_id
        if active_id is None:
            raise ArtifactSessionError("an active Align revision is required")
        parent_id = self.document.align_revision_index[active_id].parent_id
        if parent_id is None:
            raise ArtifactSessionError("the active Align revision has no parent")
        return self.activate_align(parent_id)

    def capture_operation(
        self,
        *,
        recipe: Mapping[str, object],
        selection_hash: str | None = None,
    ) -> OperationContext:
        """Capture the immutable source/geometry/metadata/Align compute context."""

        self.projection_snapshot()
        try:
            return self.document.capture_operation_context(
                recipe=recipe,
                selection_hash=selection_hash,
            )
        except ArtifactDocumentError as exc:
            raise ArtifactSessionError(str(exc)) from exc

    def capture_vector_operation(
        self,
        *,
        recipe: Mapping[str, object],
        selection_hash: str | None = None,
    ) -> OperationContext:
        """Backward-compatible vector-specific name for capture_operation()."""

        return self.capture_operation(
            recipe=recipe,
            selection_hash=selection_hash,
        )

    def commit_vector_record(
        self,
        *,
        context: OperationContext,
        payload: VectorGeometryPayload,
        recipe: Mapping[str, object],
        record_id: str,
        created_at: str,
        operator: str,
        depends_on_record_ids: tuple[str, ...] = (),
        qc: Mapping[str, object] | None = None,
    ) -> "ArtifactSession":
        """Append one verified vector payload without changing source geometry."""

        self.projection_snapshot()
        try:
            document = append_vector_record_from_context(
                self.document,
                context=context,
                payload=payload,
                recipe=recipe,
                record_id=record_id,
                created_at=created_at,
                operator=operator,
                depends_on_record_ids=depends_on_record_ids,
                qc=qc,
            )
        except (ArtifactDocumentError, ArtifactVectorRecordError) as exc:
            raise ArtifactSessionError(str(exc)) from exc
        return self.with_document(document)


__all__ = [
    "ArtifactSession",
    "ArtifactSessionError",
    "immutable_source_mesh",
]
