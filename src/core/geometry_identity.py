"""Deterministic identity for imported triangle geometry.

The source SHA-256 identifies the exact input file bytes.  This module
identifies the decoded geometry after the declared import recipe has produced
finite positions and triangle indices.  The framing is versioned so a future
change cannot silently reuse the same digest meaning.
"""

from __future__ import annotations

import numpy as np

from .artifact_document import GEOMETRY_HASH_SCOPE_V1
from .mesh_admission import MeshAdmissionError, admitted_geometry_sha256
from .mesh_loader import MeshData


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

    try:
        return admitted_geometry_sha256(vertices, faces)
    except (MeshAdmissionError, TypeError, ValueError, OverflowError) as exc:
        raise GeometryIdentityError(str(exc)) from exc


def mesh_geometry_sha256(
    mesh: MeshData,
    *,
    scope: str = GEOMETRY_HASH_SCOPE_V1,
) -> str:
    """Return the canonical geometry digest for a validated ``MeshData``."""

    if not isinstance(mesh, MeshData):
        raise GeometryIdentityError("mesh must be a MeshData")
    return canonical_geometry_sha256(mesh.vertices, mesh.faces, scope=scope)
