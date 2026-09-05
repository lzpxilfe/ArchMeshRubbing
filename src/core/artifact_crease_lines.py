"""능선: the convex creases of a surface, read from the mesh's own edges.

A flaked stone tool is drawn with inner lines on its plan: the ridges where
one flake scar meets the next ([K1] 2013 p. 48, 내선).  They are where the
surface bends outward sharply - two faces meeting at an edge with a large
dihedral angle, convex side out - and a scan carries them as exactly that.
This module finds those edges, strings them into chains, and says which of
them a given orthographic view can see.

It is geometry only.  What is a ridge and what is noise is a threshold, and
the threshold that suits a real scan - where a ridge is rounded over a
millimetre and the mesh is noisy - has not been measured; see
docs/LITHIC_TRIAL.md.  The defaults suit a mesh whose creases are edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .artifact_outline_extractor import OutlineView, outline_frame


DEFAULT_CREASE_DIHEDRAL_MIN_DEG = 25.0
DEFAULT_CREASE_MIN_LENGTH_MM = 2.0
#: Vertices closer than this are one vertex: a scan often stores a shared
#: corner once per face.
CREASE_WELD_MM = 1e-3


class ArtifactCreaseError(ValueError):
    """A crease reading cannot be produced from this mesh safely."""


@dataclass(frozen=True, slots=True)
class CreaseChain:
    """One run of crease edges, end to end, with the faces either side.

    ``points_mm`` are the chain's vertices in order; ``dihedral_deg`` has one
    angle per edge; ``left_normals`` and ``right_normals`` the unit normals
    of the two faces at each edge, so a view can tell whether it sees the
    edge or the stone hides it.
    """

    points_mm: np.ndarray
    dihedral_deg: np.ndarray
    left_normals: np.ndarray
    right_normals: np.ndarray

    @property
    def length_mm(self) -> float:
        return float(np.linalg.norm(np.diff(self.points_mm, axis=0), axis=1).sum())

    @property
    def max_dihedral_deg(self) -> float:
        return float(self.dihedral_deg.max())

    @property
    def closed(self) -> bool:
        return bool(np.allclose(self.points_mm[0], self.points_mm[-1]))


def _welded(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = np.round(vertices / CREASE_WELD_MM).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    welded_points = np.zeros((unique.shape[0], 3), dtype=np.float64)
    np.add.at(welded_points, inverse, vertices)
    counts = np.bincount(inverse, minlength=unique.shape[0]).astype(np.float64)
    welded_points /= counts[:, None]
    return welded_points, inverse[faces]


def _crest_edges_at_scale(
    points: np.ndarray,
    corners: np.ndarray,
    normals: np.ndarray,
    edge_low: np.ndarray,
    edge_high: np.ndarray,
    face_one: np.ndarray,
    face_two: np.ndarray,
    local_dihedral: np.ndarray,
    *,
    scale_mm: float,
    dihedral_min_deg: float,
) -> np.ndarray:
    """Which shared edges are the crest of a ridge rounded over ``scale_mm``.

    For each edge, the surface is sampled ``scale_mm`` away on either side -
    out from the edge's midpoint toward each face's centroid, then snapped
    to the nearest face - and the bend is the angle between the normals
    there, convex when the far sample lies below the near sample's plane.
    Every edge across a rounded ridge bends that much at that scale, so of
    the edges that qualify only those whose own two faces bend most among
    the qualifying edges within half the scale are kept: the crest.
    """

    from scipy.spatial import cKDTree  # noqa: PLC0415

    centroids = corners.mean(axis=1)
    tree = cKDTree(centroids)
    mid = 0.5 * (points[edge_low] + points[edge_high])
    along_edge = points[edge_high] - points[edge_low]
    along_edge /= np.maximum(np.linalg.norm(along_edge, axis=1), 1e-12)[:, None]
    samples = []
    for face in (face_one, face_two):
        # Straight out from the edge across the face - perpendicular to the
        # edge, in the face's own plane, toward its centroid - so the sample
        # lies on this side of the ridge and on this surface.  A sliver's
        # centroid lies almost along the edge, so it gives the side only.
        own = normals[face]
        direction = np.cross(own, along_edge)
        toward = np.einsum("ij,ij->i", centroids[face] - mid, direction)
        direction *= np.where(toward < 0.0, -1.0, 1.0)[:, None]
        length = np.linalg.norm(direction, axis=1)
        direction = direction / np.maximum(length, 1e-12)[:, None]
        target = mid + direction * scale_mm
        _distances, nearest = tree.query(target, k=12)
        nearest = np.asarray(nearest).reshape(target.shape[0], -1)
        chosen = nearest[:, 0].copy()
        for row in range(target.shape[0]):
            for candidate in nearest[row]:
                # Not the back of the stone, and not still on the ridge.
                if float(normals[candidate] @ own[row]) <= 0.0:
                    continue
                if float((centroids[candidate] - mid[row]) @ direction[row]) < 0.5 * scale_mm:
                    continue
                chosen[row] = candidate
                break
        samples.append(chosen)
    near_one, near_two = samples
    n1, n2 = normals[near_one], normals[near_two]
    cosine = np.clip(np.einsum("ij,ij->i", n1, n2), -1.0, 1.0)
    bend = np.degrees(np.arccos(cosine))
    below = np.einsum("ij,ij->i", centroids[near_two] - centroids[near_one], n1)
    candidate = (bend >= dihedral_min_deg) & (below < 0.0)
    keep = np.zeros(candidate.shape[0], dtype=bool)
    indices = np.flatnonzero(candidate)
    if indices.size == 0:
        return keep
    # How sharply the surface turns across each edge, per millimetre: the
    # two faces' bend over how far their centroids stand from the edge on
    # either side.  A bend alone favours big triangles, and the straight
    # distance between centroids runs along a sliver; this favours the
    # crest whatever the mesh happens to do there.
    span = np.zeros(mid.shape[0], dtype=np.float64)
    for face in (face_one, face_two):
        offset = centroids[face] - points[edge_low]
        span += np.linalg.norm(np.cross(offset, along_edge), axis=1)
    turning = local_dihedral / np.maximum(span, 1e-6)
    crest_tree = cKDTree(mid[indices])
    neighbours = crest_tree.query_ball_point(mid[indices], r=0.5 * scale_mm)
    for position, index in enumerate(indices):
        rivals = indices[np.asarray(neighbours[position], dtype=np.int64)]
        keep[index] = turning[index] >= 0.9 * float(turning[rivals].max())
    return keep


def detect_convex_creases(
    vertices: object,
    faces: object,
    *,
    dihedral_min_deg: float = DEFAULT_CREASE_DIHEDRAL_MIN_DEG,
    min_length_mm: float = DEFAULT_CREASE_MIN_LENGTH_MM,
    scale_mm: float = 0.0,
) -> tuple[CreaseChain, ...]:
    """Every chain of convex edges bent by at least ``dihedral_min_deg``.

    An edge is convex when the far corner of one face lies below the plane
    of the other, taking the faces' own winding as outward.  Edges shared by
    other than two faces are not creases: an open boundary or a tangle is
    something else.  Chains shorter than ``min_length_mm`` are dropped.

    With ``scale_mm`` at zero the bend is the angle between the two faces
    that share the edge, which is right for a mesh whose ridges are edges.
    A scanned ridge is rounded over a millimetre or two, and no single edge
    of it bends much; ``scale_mm`` then measures the bend between the
    surface ``scale_mm`` to either side of the edge instead, and keeps, of
    the edges that bend enough at that scale, only the ones that bend most
    among their neighbours within half that distance - the crest.  The
    scale that suits a real scan has not been measured.
    """

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ArtifactCreaseError("crease detection needs (n, 3) vertices and (m, 3) faces")
    if not (0.0 < float(dihedral_min_deg) < 180.0):
        raise ArtifactCreaseError("dihedral_min_deg must lie strictly between 0 and 180")
    if float(min_length_mm) < 0.0:
        raise ArtifactCreaseError("min_length_mm cannot be negative")
    if not np.isfinite(float(scale_mm)) or float(scale_mm) < 0.0:
        raise ArtifactCreaseError("scale_mm must be zero or a positive length")
    if triangles.size == 0:
        return ()
    points, triangles = _welded(points, triangles)

    corners = points[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0.0):
        raise ArtifactCreaseError("faces contain a degenerate triangle")
    normals /= lengths[:, None]

    # Every edge once per face, with the corner opposite it.
    edge_a = np.concatenate([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    edge_b = np.concatenate([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    opposite = np.concatenate([triangles[:, 2], triangles[:, 0], triangles[:, 1]])
    face_of = np.tile(np.arange(triangles.shape[0]), 3)
    low = np.minimum(edge_a, edge_b)
    high = np.maximum(edge_a, edge_b)
    order = np.lexsort((high, low))
    low, high, opposite, face_of = low[order], high[order], opposite[order], face_of[order]
    keys = low.astype(np.int64) * (points.shape[0] + 1) + high
    _unique, starts, counts = np.unique(keys, return_index=True, return_counts=True)
    shared = starts[counts == 2]

    first, second = shared, shared + 1
    n1, n2 = normals[face_of[first]], normals[face_of[second]]
    cosine = np.clip(np.einsum("ij,ij->i", n1, n2), -1.0, 1.0)
    dihedral = np.degrees(np.arccos(cosine))
    # Convex: the second face's far corner lies below the first face's plane.
    below = np.einsum(
        "ij,ij->i", points[opposite[second]] - points[low[first]], n1
    )
    if float(scale_mm) > 0.0:
        keep = _crest_edges_at_scale(
            points,
            corners,
            normals,
            low[first],
            high[first],
            face_of[first],
            face_of[second],
            dihedral,
            scale_mm=float(scale_mm),
            dihedral_min_deg=float(dihedral_min_deg),
        )
    else:
        keep = (dihedral >= float(dihedral_min_deg)) & (below < 0.0)
    crease_edges = np.stack([low[first][keep], high[first][keep]], axis=1)
    crease_dihedral = dihedral[keep]
    crease_n1, crease_n2 = n1[keep], n2[keep]
    if crease_edges.shape[0] == 0:
        return ()

    # String edges into chains: walk from every vertex that is not simply
    # passed through, then pick up whatever closed loops remain.
    incident: dict[int, list[int]] = {}
    for index, (a, b) in enumerate(crease_edges.tolist()):
        incident.setdefault(a, []).append(index)
        incident.setdefault(b, []).append(index)
    used = np.zeros(crease_edges.shape[0], dtype=bool)
    chains: list[CreaseChain] = []

    def walk(start_vertex: int, start_edge: int) -> None:
        vertex = start_vertex
        edge = start_edge
        run_vertices = [vertex]
        run_edges = []
        while True:
            used[edge] = True
            run_edges.append(edge)
            a, b = crease_edges[edge]
            vertex = int(b if a == vertex else a)
            run_vertices.append(vertex)
            following = [e for e in incident[vertex] if not used[e]]
            if len(incident[vertex]) != 2 or not following:
                break
            edge = following[0]
        chains.append(
            CreaseChain(
                points_mm=points[np.asarray(run_vertices)],
                dihedral_deg=crease_dihedral[np.asarray(run_edges)],
                left_normals=crease_n1[np.asarray(run_edges)],
                right_normals=crease_n2[np.asarray(run_edges)],
            )
        )

    for vertex in sorted(incident):
        if len(incident[vertex]) == 2:
            continue
        for edge in incident[vertex]:
            if not used[edge]:
                walk(vertex, edge)
    for edge in range(crease_edges.shape[0]):
        if not used[edge]:
            walk(int(crease_edges[edge][0]), edge)

    kept = tuple(chain for chain in chains if chain.length_mm >= float(min_length_mm))
    return tuple(sorted(kept, key=lambda chain: (-chain.length_mm, chain.points_mm[0].tolist())))


def creases_seen_from(
    chains: Sequence[CreaseChain],
    view: OutlineView | str,
    *,
    min_length_mm: float = DEFAULT_CREASE_MIN_LENGTH_MM,
) -> list[np.ndarray]:
    """The chains' visible runs, projected into one orthographic view.

    An edge is seen when both its faces turn toward the viewer; a run of
    seen edges becomes one polyline in the view's frame, in millimetres.
    Runs shorter than ``min_length_mm`` are dropped.
    """

    frame = outline_frame(OutlineView(view) if isinstance(view, str) else view)
    origin = np.asarray(frame.origin_world_mm, dtype=np.float64)
    u_axis = np.asarray(frame.u_axis_world, dtype=np.float64)
    v_axis = np.asarray(frame.v_axis_world, dtype=np.float64)
    # A view's frame normal points out of the page at the viewer: +Z for the
    # plan seen from above.
    toward_viewer = np.asarray(frame.normal_world, dtype=np.float64)
    polylines: list[np.ndarray] = []
    for chain in chains:
        seen = (chain.left_normals @ toward_viewer > 0.0) & (
            chain.right_normals @ toward_viewer > 0.0
        )
        start = None
        for index in range(len(seen) + 1):
            if index < len(seen) and seen[index]:
                if start is None:
                    start = index
                continue
            if start is not None:
                run = chain.points_mm[start : index + 1] - origin
                projected = np.column_stack([run @ u_axis, run @ v_axis])
                if float(np.linalg.norm(np.diff(projected, axis=0), axis=1).sum()) >= float(
                    min_length_mm
                ):
                    polylines.append(projected)
                start = None
    return polylines


def crease_summary(chains: Sequence[CreaseChain]) -> dict[str, Any]:
    """The numbers a reading would record."""

    return {
        "chain_count": len(chains),
        "total_length_mm": round(sum(chain.length_mm for chain in chains), 6),
        "max_dihedral_deg": round(
            max((chain.max_dihedral_deg for chain in chains), default=0.0), 6
        ),
    }


__all__ = [
    "DEFAULT_CREASE_DIHEDRAL_MIN_DEG",
    "DEFAULT_CREASE_MIN_LENGTH_MM",
    "ArtifactCreaseError",
    "CreaseChain",
    "crease_summary",
    "creases_seen_from",
    "detect_convex_creases",
]
