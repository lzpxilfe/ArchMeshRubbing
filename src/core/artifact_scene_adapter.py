"""Headless materialization boundary between ArtifactDocument and scene geometry.

This module deliberately knows nothing about Qt, OpenGL, cameras, or mutable
``SceneObject`` fields.  It projects one externally verified source-space
``MeshData`` into canonical world millimeters from an immutable document
snapshot.  Materialization always starts from the source-space mesh; it never
centers, bakes back into the source, or chains from an earlier projection.

Geometry hashing remains an importer/verifier responsibility.  The declared
``geometry_hash_scope`` identifies that external contract, but this adapter
does not invent byte framing rules that are absent from the document schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from .alignment_utils import transform_points
from .artifact_document import (
    ArtifactDocument,
    ArtifactDocumentError,
    GeometryRevision,
    SourceAsset,
)
from .mesh_loader import MeshData


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactSceneAdapterError(ValueError):
    """Base error for a rejected document/geometry materialization."""


class SourceBindingError(ArtifactSceneAdapterError):
    """The supplied mesh is not bound to the declared source asset."""


class GeometryBindingError(ArtifactSceneAdapterError):
    """External geometry identity evidence does not match the document."""


class StaleProjectionContextError(ArtifactSceneAdapterError):
    """A staged projection context no longer matches the current document."""


def _required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactSceneAdapterError(f"{field_name} must be a non-empty string")
    return value


def _required_sha256(value: object, *, field_name: str) -> str:
    digest = _required_string(value, field_name=field_name).lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ArtifactSceneAdapterError(
            f"{field_name} must be 64 lowercase hexadecimal characters"
        )
    return digest


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float, float], ...]:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (4, 4) or not np.isfinite(arr).all():
        raise ArtifactSceneAdapterError("projection matrix must be a finite 4x4 matrix")
    arr = arr.copy()
    arr[arr == 0.0] = 0.0
    return tuple(tuple(float(cell) for cell in row) for row in arr)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class VerifiedGeometryIdentity:
    """Identity evidence produced by the geometry importer/verifier.

    Construction of this value is a trust-boundary assertion, not a hashing
    operation.  The adapter compares it strictly with ``GeometryRevision`` and
    uses the declared hash scope, but does not reinterpret or recompute it.
    """

    source_asset_id: str
    geometry_revision_id: str
    geometry_sha256: str
    geometry_hash_scope: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_asset_id",
            _required_string(self.source_asset_id, field_name="source_asset_id"),
        )
        object.__setattr__(
            self,
            "geometry_revision_id",
            _required_string(
                self.geometry_revision_id,
                field_name="geometry_revision_id",
            ),
        )
        object.__setattr__(
            self,
            "geometry_sha256",
            _required_sha256(self.geometry_sha256, field_name="geometry_sha256"),
        )
        object.__setattr__(
            self,
            "geometry_hash_scope",
            _required_string(
                self.geometry_hash_scope,
                field_name="geometry_hash_scope",
            ),
        )


@dataclass(frozen=True, slots=True)
class ArtifactProjectionSnapshot:
    """Immutable identity and matrix snapshot used for one scene projection."""

    document_id: str
    document_schema_version: str
    document_sha256: str
    source_asset_id: str
    geometry_revision_id: str
    source_metadata_revision_id: str
    align_revision_id: str
    geometry_sha256: str
    geometry_hash_scope: str
    matrix4x4: tuple[tuple[float, float, float, float], ...]

    @property
    def matrix(self) -> np.ndarray:
        """Return a mutable float64 copy of the source-to-world-mm matrix."""

        return np.asarray(self.matrix4x4, dtype=np.float64).copy()

    @property
    def render_key(self) -> tuple[object, ...]:
        """Identity of the live geometry projection, excluding document content.

        Appending a DerivedRecord changes ``document_sha256`` without changing
        any coordinates or GPU buffers.  Consumers may reuse an existing mesh
        only when this complete render key remains equal.
        """

        return (
            self.document_id,
            self.document_schema_version,
            self.source_asset_id,
            self.geometry_revision_id,
            self.source_metadata_revision_id,
            self.align_revision_id,
            self.geometry_sha256,
            self.geometry_hash_scope,
            self.matrix4x4,
        )

    def has_same_render_projection(self, other: object) -> bool:
        return (
            isinstance(other, ArtifactProjectionSnapshot)
            and self.render_key == other.render_key
        )


@dataclass(frozen=True, slots=True)
class ArtifactSceneProjection:
    """A fresh world-mm MeshData projection plus its immutable snapshot.

    ``MeshData`` is intentionally a runtime mutable container, so callers must
    treat ``mesh`` as projection-owned.  Every materialization returns a new
    container and disjoint arrays; modifying it cannot change the source mesh.
    """

    mesh: MeshData
    snapshot: ArtifactProjectionSnapshot


class ArtifactSceneAdapter:
    """Pure, Qt/OpenGL-free scene materializer for one ArtifactDocument."""

    def __init__(self, document: ArtifactDocument):
        if not isinstance(document, ArtifactDocument):
            raise ArtifactSceneAdapterError("document must be an ArtifactDocument")
        self._document = document

    @property
    def document(self) -> ArtifactDocument:
        return self._document

    def validate_geometry_identity(
        self,
        verified: VerifiedGeometryIdentity,
    ) -> tuple[GeometryRevision, SourceAsset]:
        """Match external geometry evidence to the active document context."""

        if not isinstance(verified, VerifiedGeometryIdentity):
            raise GeometryBindingError(
                "verified geometry identity must be a VerifiedGeometryIdentity"
            )

        active_align_id = self._document.active_align_revision_id
        if active_align_id is None:
            raise GeometryBindingError("document has no active Align revision")
        align = self._document.align_revision_index.get(active_align_id)
        if align is None:
            raise GeometryBindingError("active Align revision is missing")

        active_metadata_id = self._document.active_source_metadata_revision_id
        if active_metadata_id is None:
            raise GeometryBindingError("document has no active metadata revision")
        metadata = self._document.source_metadata_revision_index.get(active_metadata_id)
        if metadata is None:
            raise GeometryBindingError("active metadata revision is missing")
        if align.source_metadata_revision_id != metadata.id:
            raise GeometryBindingError("active Align and metadata context do not match")

        geometry = self._document.geometry_revision_index.get(
            verified.geometry_revision_id
        )
        if geometry is None:
            raise GeometryBindingError(
                f"geometry revision {verified.geometry_revision_id!r} is missing"
            )
        if geometry.id != metadata.geometry_revision_id:
            raise GeometryBindingError(
                "verified geometry revision is not the active metadata geometry"
            )
        if verified.geometry_sha256 != geometry.geometry_sha256:
            raise GeometryBindingError("verified geometry SHA-256 does not match document")
        if verified.geometry_hash_scope != geometry.geometry_hash_scope:
            raise GeometryBindingError("verified geometry hash scope does not match document")

        # This adapter currently accepts exactly one primary source-space mesh.
        # Composite/sidecar materialization requires a future multi-asset adapter.
        if tuple(geometry.source_asset_ids) != (verified.source_asset_id,):
            raise GeometryBindingError(
                "one-mesh adapter requires exactly the verified source asset"
            )
        source_asset = self._document.source_asset_index.get(
            verified.source_asset_id
        )
        if source_asset is None:
            raise GeometryBindingError(
                f"source asset {verified.source_asset_id!r} is missing"
            )
        return geometry, source_asset

    def capture_snapshot(
        self,
        verified: VerifiedGeometryIdentity,
    ) -> ArtifactProjectionSnapshot:
        """Capture the exact active source/metadata/Align projection context."""

        geometry, source_asset = self.validate_geometry_identity(verified)
        active_align_id = self._document.active_align_revision_id
        active_metadata_id = self._document.active_source_metadata_revision_id
        assert active_align_id is not None
        assert active_metadata_id is not None

        try:
            matrix = self._document.active_canonical_matrix()
        except ArtifactDocumentError as exc:
            raise GeometryBindingError(
                f"active document cannot materialize canonical geometry: {exc}"
            ) from exc

        return ArtifactProjectionSnapshot(
            document_id=self._document.document_id,
            document_schema_version=self._document.schema_version,
            document_sha256=self._document.canonical_sha256,
            source_asset_id=source_asset.id,
            geometry_revision_id=geometry.id,
            source_metadata_revision_id=active_metadata_id,
            align_revision_id=active_align_id,
            geometry_sha256=geometry.geometry_sha256,
            geometry_hash_scope=geometry.geometry_hash_scope,
            matrix4x4=_matrix_tuple(matrix),
        )

    def require_current_snapshot(
        self,
        expected: ArtifactProjectionSnapshot,
        verified: VerifiedGeometryIdentity,
    ) -> ArtifactProjectionSnapshot:
        """Reject a staged snapshot if any document/context field changed."""

        if not isinstance(expected, ArtifactProjectionSnapshot):
            raise StaleProjectionContextError(
                "expected snapshot must be an ArtifactProjectionSnapshot"
            )
        current = self.capture_snapshot(verified)
        if expected != current:
            raise StaleProjectionContextError(
                "staged projection context is stale for the current document"
            )
        return current

    def validate_source_mesh(
        self,
        source_mesh: MeshData,
        snapshot: ArtifactProjectionSnapshot,
    ) -> None:
        """Validate source identity and basic source-space geometry invariants."""

        if not isinstance(source_mesh, MeshData):
            raise SourceBindingError("source_mesh must be a MeshData")
        if not isinstance(snapshot, ArtifactProjectionSnapshot):
            raise SourceBindingError("snapshot must be an ArtifactProjectionSnapshot")

        source_asset = self._document.source_asset_index.get(snapshot.source_asset_id)
        if source_asset is None:
            raise SourceBindingError("snapshot source asset is missing from document")
        identity = source_mesh.source_identity
        if identity is None:
            raise SourceBindingError("source mesh has no verified source fingerprint")
        if identity.id != source_asset.id or identity.sha256 != source_asset.sha256:
            raise SourceBindingError("source mesh SHA-256 does not match source asset")
        if identity.size_bytes != source_asset.size_bytes:
            raise SourceBindingError("source mesh byte length does not match source asset")
        if identity.identity_scope != source_asset.identity_scope:
            raise SourceBindingError("source mesh identity scope does not match source asset")

        vertices = np.asarray(source_mesh.vertices)
        faces = np.asarray(source_mesh.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            raise SourceBindingError("source mesh vertices must have shape (N, 3), N > 0")
        if not np.isfinite(vertices).all():
            raise SourceBindingError("source mesh vertices must be finite")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise SourceBindingError("source mesh faces must have shape (M, 3)")
        if faces.size > 0:
            if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
                raise SourceBindingError("source mesh contains invalid face indices")

    def materialize(
        self,
        source_mesh: MeshData,
        verified: VerifiedGeometryIdentity,
        *,
        expected_snapshot: ArtifactProjectionSnapshot | None = None,
    ) -> ArtifactSceneProjection:
        """Return a fresh, uncentered canonical-world-mm projection.

        If ``expected_snapshot`` is supplied, all document and revision fields
        must still match.  This is the staging/swap guard for a future GUI
        adapter: a late projection is never attached to a newer context.
        """

        snapshot = (
            self.capture_snapshot(verified)
            if expected_snapshot is None
            else self.require_current_snapshot(expected_snapshot, verified)
        )
        self.validate_source_mesh(source_mesh, snapshot)

        source_vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
        world_vertices = transform_points(source_vertices, snapshot.matrix)
        projected_faces = np.asarray(source_mesh.faces, dtype=np.int32).copy()
        # A signed-axis metadata mapping may reflect source coordinates. Keep
        # face row/ID order stable while reversing each triangle so culling and
        # recomputed outward normals remain consistent in canonical space.
        if float(np.linalg.det(snapshot.matrix[:3, :3])) < 0.0 and projected_faces.size:
            projected_faces = projected_faces[:, [0, 2, 1]]
        world_mesh = MeshData(
            vertices=world_vertices,
            faces=projected_faces,
            normals=None,
            face_normals=None,
            uv_coords=(
                np.asarray(source_mesh.uv_coords).copy()
                if source_mesh.uv_coords is not None
                else None
            ),
            texture=(
                np.asarray(source_mesh.texture).copy()
                if source_mesh.texture is not None
                else None
            ),
            unit="mm",
            filepath=source_mesh.filepath,
            source_identity=source_mesh.source_identity,
            source_format=source_mesh.source_format,
            source_import_recipe=source_mesh.source_import_recipe,
            source_admission_receipt=source_mesh.source_admission_receipt,
            source_resources=source_mesh.source_resources,
        )
        return ArtifactSceneProjection(mesh=world_mesh, snapshot=snapshot)


__all__ = [
    "ArtifactProjectionSnapshot",
    "ArtifactSceneAdapter",
    "ArtifactSceneAdapterError",
    "ArtifactSceneProjection",
    "GeometryBindingError",
    "SourceBindingError",
    "StaleProjectionContextError",
    "VerifiedGeometryIdentity",
]
