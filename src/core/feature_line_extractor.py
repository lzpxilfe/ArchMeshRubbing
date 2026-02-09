"""
Feature line (sharp edge) extraction for archaeological drawing.

This module detects "sharp" edges by dihedral angle between adjacent faces.
It is used to export ridge/crease line layers to SVG for post-processing
in vector editors like Illustrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh_loader import MeshData


@dataclass(frozen=True)
class SharpEdgeResult:
    """Sharp edge list with adjacency metadata.

    Attributes:
        edges: (E, 2) vertex indices (undirected, sorted per row)
        face_pairs: (E, 2) adjacent face indices. For boundary edges, face_pairs[:, 1] is -1.
        dihedral_deg: (E,) dihedral angles in degrees. Boundary edges use 180.0.
    """

    edges: np.ndarray
    face_pairs: np.ndarray
    dihedral_deg: np.ndarray


def _compute_face_normals(mesh: MeshData) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if faces.ndim != 2 or faces.shape[1] < 3 or faces.size == 0 or vertices.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    n = n / norms
    return n


def extract_sharp_edges(
    mesh: MeshData,
    *,
    angle_deg: float = 60.0,
    include_boundary: bool = False,
    min_edge_length: float = 0.0,
) -> SharpEdgeResult:
    """Extract sharp edges from a triangle mesh.

    Args:
        mesh: MeshData (assumed triangles).
        angle_deg: Dihedral threshold in degrees. Higher -> fewer, sharper edges.
        include_boundary: If True, boundary edges are also returned.
        min_edge_length: Minimum edge length (in mesh units) to keep. Helps suppress noise.

    Returns:
        SharpEdgeResult
    """

    # Fast cache on the mesh instance (safe in this app: mesh is effectively immutable after load).
    try:
        cache = getattr(mesh, "_sharp_edge_cache", None)
        cache_meta = getattr(mesh, "_sharp_edge_cache_meta", None)
    except Exception:
        cache = None
        cache_meta = None

    faces = np.asarray(mesh.faces, dtype=np.int32)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    n_faces = int(faces.shape[0]) if faces.ndim == 2 else 0
    n_vertices = int(vertices.shape[0]) if vertices.ndim == 2 else 0
    meta = (n_vertices, n_faces, int(id(vertices)), int(id(faces)))
    key = (float(angle_deg), bool(include_boundary), float(min_edge_length))

    if isinstance(cache, dict) and cache_meta == meta:
        try:
            hit = cache.get(key)
            if isinstance(hit, SharpEdgeResult):
                return hit
        except Exception:
            pass

    if n_faces == 0 or n_vertices == 0:
        result = SharpEdgeResult(
            edges=np.zeros((0, 2), dtype=np.int32),
            face_pairs=np.zeros((0, 2), dtype=np.int32),
            dihedral_deg=np.zeros((0,), dtype=np.float64),
        )
        try:
            setattr(mesh, "_sharp_edge_cache", {key: result})
            setattr(mesh, "_sharp_edge_cache_meta", meta)
        except Exception:
            pass
        return result

    # Face normals: reuse if present; otherwise compute.
    fn = getattr(mesh, "face_normals", None)
    fn_arr = None
    try:
        fn_arr = np.asarray(fn, dtype=np.float64) if fn is not None else None
    except Exception:
        fn_arr = None
    if fn_arr is None or fn_arr.shape != (n_faces, 3):
        fn_arr = _compute_face_normals(mesh)

    # Build undirected edge list (M*3, 2) and owning face ids.
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = np.vstack([e01, e12, e20]).astype(np.int32, copy=False)
    edges.sort(axis=1)
    face_ids = np.repeat(np.arange(n_faces, dtype=np.int32), 3)

    # Sort edges to group identical ones.
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges_s = edges[order]
    face_s = face_ids[order]

    # Group boundaries.
    is_new = np.empty((edges_s.shape[0],), dtype=bool)
    is_new[0] = True
    is_new[1:] = np.any(edges_s[1:] != edges_s[:-1], axis=1)
    starts = np.flatnonzero(is_new).astype(np.int32, copy=False)
    counts = np.diff(np.append(starts, edges_s.shape[0])).astype(np.int32, copy=False)

    unique_edges = edges_s[starts]
    face0 = face_s[starts]
    face1 = np.full((starts.shape[0],), -1, dtype=np.int32)

    # Common case: manifold edges shared by exactly 2 faces.
    two_mask = counts == 2
    if np.any(two_mask):
        s2 = starts[two_mask]
        face1[two_mask] = face_s[s2 + 1]

    # Non-manifold edges (count > 2): pick a face pair with the maximum dihedral angle.
    nm_mask = counts > 2
    if np.any(nm_mask):
        nm_idx = np.flatnonzero(nm_mask)
        for i in nm_idx.tolist():
            s = int(starts[i])
            c = int(counts[i])
            f = face_s[s : s + c]
            if f.size < 2:
                continue
            n = fn_arr[f]
            # Find pair with minimum dot (maximum angle).
            dots = n @ n.T
            np.fill_diagonal(dots, 1.0)
            ij = int(np.argmin(dots))
            a = int(ij // dots.shape[1])
            b = int(ij % dots.shape[1])
            face0[i] = int(f[a])
            face1[i] = int(f[b])

    # Compute dihedral angles for edges that have 2 faces.
    dihedral = np.zeros((unique_edges.shape[0],), dtype=np.float64)
    has_two = face1 >= 0
    if np.any(has_two):
        n0 = fn_arr[face0[has_two]]
        n1 = fn_arr[face1[has_two]]
        dot = np.einsum("ij,ij->i", n0, n1)
        dot = np.clip(dot, -1.0, 1.0)
        dihedral[has_two] = np.degrees(np.arccos(dot))

    # Boundary edges (one face).
    if include_boundary:
        dihedral[~has_two] = 180.0

    # Angle filter.
    ang = float(angle_deg)
    if not np.isfinite(ang):
        ang = 60.0
    keep = (dihedral >= ang) & (has_two | bool(include_boundary))

    # Length filter (optional).
    min_len = float(min_edge_length)
    if min_len > 0 and np.any(keep):
        e_keep = unique_edges[keep]
        p0 = vertices[e_keep[:, 0]]
        p1 = vertices[e_keep[:, 1]]
        lens = np.linalg.norm(p1 - p0, axis=1)
        keep_idx = lens >= min_len
        if not bool(np.all(keep_idx)):
            # Apply keep_idx back to the global keep mask.
            keep_positions = np.flatnonzero(keep)
            keep = np.zeros_like(keep, dtype=bool)
            keep[keep_positions[keep_idx]] = True

    result = SharpEdgeResult(
        edges=unique_edges[keep].astype(np.int32, copy=False),
        face_pairs=np.stack([face0[keep], face1[keep]], axis=1).astype(np.int32, copy=False),
        dihedral_deg=dihedral[keep].astype(np.float64, copy=False),
    )

    try:
        if not isinstance(cache, dict) or cache_meta != meta:
            cache = {}
        cache[key] = result
        setattr(mesh, "_sharp_edge_cache", cache)
        setattr(mesh, "_sharp_edge_cache_meta", meta)
    except Exception:
        pass

    return result

