"""Deterministic identity for imported triangle geometry.

The source SHA-256 identifies the exact input file bytes.  This module
identifies the decoded geometry after the declared import recipe has produced
finite positions and triangle indices.  The framing is versioned so a future
change cannot silently reuse the same digest meaning.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np

from .artifact_document import GEOMETRY_HASH_SCOPE_V1
from .mesh_loader import MeshData


_DOMAIN_V1 = b"archmeshrubbing.geometry\x00"


class GeometryIdentityError(ValueError):
    """Geometry cannot be represented by the declared hash scope."""


def canonical_geometry_sha256(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    scope: str = GEOMETRY_HASH_SCOPE_V1,
) -> str:
    """Hash positions and triangles using the V1 canonical byte framing.

    Framing is ``domain || scope || N || positions || M || triangles`` where
    counts are unsigned little-endian 64-bit integers, positions are C-order
    little-endian float64 and triangles are C-order little-endian int32.
    Signed zero is normalized to positive zero.  Vertex and face order, and
    triangle winding, remain identity-bearing.
    """

    if scope != GEOMETRY_HASH_SCOPE_V1:
        raise GeometryIdentityError(f"unsupported geometry hash scope: {scope!r}")

    positions = np.asarray(vertices)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise GeometryIdentityError("vertices must have shape (N, 3)")
    try:
        positions64 = np.array(positions, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeometryIdentityError(f"vertices cannot be converted to float64: {exc}") from exc
    if not np.isfinite(positions64).all():
        raise GeometryIdentityError("vertices must contain only finite values")
    positions64[positions64 == 0.0] = 0.0
    positions_le = np.ascontiguousarray(positions64.astype("<f8", copy=False))

    triangles = np.asarray(faces)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise GeometryIdentityError("faces must have shape (M, 3)")
    if not np.issubdtype(triangles.dtype, np.integer):
        raise GeometryIdentityError("faces must contain integer indices")
    try:
        triangles64 = np.array(triangles, dtype=np.int64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeometryIdentityError(f"faces cannot be converted to integers: {exc}") from exc
    if triangles64.size:
        if int(triangles64.min()) < 0:
            raise GeometryIdentityError("faces cannot contain negative indices")
        if int(triangles64.max()) >= int(positions64.shape[0]):
            raise GeometryIdentityError("faces reference a missing vertex")
        if int(triangles64.max()) > np.iinfo(np.int32).max:
            raise GeometryIdentityError("faces exceed the int32 index range")
    triangles_le = np.ascontiguousarray(triangles64.astype("<i4", copy=False))

    digest = hashlib.sha256()
    digest.update(_DOMAIN_V1)
    digest.update(scope.encode("ascii"))
    digest.update(b"\x00")
    digest.update(struct.pack("<Q", int(positions_le.shape[0])))
    digest.update(positions_le.tobytes(order="C"))
    digest.update(struct.pack("<Q", int(triangles_le.shape[0])))
    digest.update(triangles_le.tobytes(order="C"))
    return digest.hexdigest()


def mesh_geometry_sha256(
    mesh: MeshData,
    *,
    scope: str = GEOMETRY_HASH_SCOPE_V1,
) -> str:
    """Return the canonical geometry digest for a validated ``MeshData``."""

    if not isinstance(mesh, MeshData):
        raise GeometryIdentityError("mesh must be a MeshData")
    return canonical_geometry_sha256(mesh.vertices, mesh.faces, scope=scope)
