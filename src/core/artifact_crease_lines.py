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

#: How the crest of a rounded ridge is picked from the edges that bend enough
#: at the scale.  The first rule compared every qualifying edge within half
#: the scale by its own two faces' turning per millimetre; on a scanned mesh
#: the edges along one ridge suppress each other and the ridge comes out as
#: fragments a few edges long, more of them the finer the mesh
#: (docs/LITHIC_TRIAL.md).  The second reads each side's normal as the
#: area-weighted mean of the faces within half the scale of the sample point,
#: scores an edge by how high it stands above the chord between the two
#: samples, and lets only edges lying across the ridge - offset from this one
#: at more than 60 degrees to its direction - compete.  A record names its
#: rule, so a reading taken under the first is recomputed under the first.
CREST_RULE_TURNING_V1 = "turning_per_mm_maximum_within_half_scale/v1"
CREST_RULE_CURVATURE_V2 = "largest_curvature_maximum_zero_crossing/v2"
CREST_RULES = frozenset({CREST_RULE_TURNING_V1, CREST_RULE_CURVATURE_V2})
DEFAULT_CREST_RULE = CREST_RULE_CURVATURE_V2
#: Two chain ends are joined when they point at each other within this angle.
CREASE_LINK_ANGLE_DEG = 45.0
#: Joining is repeated until nothing joins, at most this many times.
CREASE_LINK_ROUNDS = 4


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
    """Which shared edges are the crest of a ridge rounded over ``scale_mm``,
    by the first rule (``CREST_RULE_TURNING_V1``).

    For each edge, the surface is sampled ``scale_mm`` away on either side -
    out from the edge's midpoint toward each face's centroid, then snapped
    to the nearest face - and the bend is the angle between the normals
    there, convex when the far sample lies below the near sample's plane.
    Every edge across a rounded ridge bends that much at that scale, so of
    the edges that qualify only those whose own two faces bend most among
    the qualifying edges within half the scale are kept: the crest.  Kept
    as it was for the records that name it; see ``_ridge_chains_at_scale``
    for what a scanned mesh needs.
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


#: Neighbours consulted per vertex when the surface is read at a scale.
_RIDGE_NEIGHBOURS = 48


def _ridge_chains_at_scale(
    points: np.ndarray,
    triangles: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    *,
    scale_mm: float,
    dihedral_min_deg: float,
) -> list[CreaseChain]:
    """The ridges of a surface read at ``scale_mm``, as lines through the
    triangles (``CREST_RULE_CURVATURE_V2``).

    At every vertex the normal is the area-weighted mean of the faces within
    half the scale, and the shape operator is fitted to how that normal
    changes toward the vertices within the scale, giving the largest
    principal curvature and its direction; convex is positive with the
    faces' own winding as outward.  A ridge is where that curvature is a
    maximum along its own direction - the derivative changes sign from
    positive to negative - and the line is found where it crosses the
    mesh's edges, interpolated, so a ridge is one polyline with a point on
    every edge it crosses, whatever the facets under it do.  An edge
    crossing counts when the bend it implies over twice the scale,
    ``curvature x 2 x scale``, reaches ``dihedral_min_deg``.  Chains break
    only where a triangle is crossed three times.
    """

    from scipy.spatial import cKDTree  # noqa: PLC0415

    count = points.shape[0]
    # Vertex normals: the area-weighted mean of the faces within half the
    # scale of the vertex, restricted to faces that face the same way as
    # the vertex's own faces (a thin edge sees its other side otherwise).
    own = np.zeros((count, 3), dtype=np.float64)
    for corner in range(3):
        np.add.at(own, triangles[:, corner], face_normals * face_areas[:, None])
    own /= np.maximum(np.linalg.norm(own, axis=1), 1e-12)[:, None]
    centroids = points[triangles].mean(axis=1)
    face_tree = cKDTree(centroids)
    distances, nearest = face_tree.query(
        points, k=_RIDGE_NEIGHBOURS, distance_upper_bound=0.5 * scale_mm
    )
    distances = np.asarray(distances).reshape(count, -1)
    nearest = np.asarray(nearest).reshape(count, -1)
    found = np.isfinite(distances)
    nearest = np.where(found, nearest, 0)
    patch = face_normals[nearest]
    facing = np.einsum("ijk,ik->ij", patch, own) > 0.0
    weight = face_areas[nearest] * (found & facing)
    normals = np.einsum("ij,ijk->ik", weight, patch)
    bare = weight.sum(axis=1) <= 0.0
    normals[bare] = own[bare]
    normals /= np.maximum(np.linalg.norm(normals, axis=1), 1e-12)[:, None]

    # A tangent basis at every vertex.
    helper = np.where(np.abs(normals[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    t1 = np.cross(normals, helper)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1), 1e-12)[:, None]
    t2 = np.cross(normals, t1)

    # The shape operator, fitted to the vertices within the scale: the
    # change of normal against the change of position, both in the tangent
    # plane, in the least-squares sense.
    vertex_tree = cKDTree(points)
    distances, nearest = vertex_tree.query(
        points, k=_RIDGE_NEIGHBOURS, distance_upper_bound=scale_mm
    )
    distances = np.asarray(distances).reshape(count, -1)
    nearest = np.asarray(nearest).reshape(count, -1)
    found = np.isfinite(distances) & (distances > 0.0)
    nearest = np.where(found, nearest, 0)
    dp = points[nearest] - points[:, None, :]
    dn = normals[nearest] - normals[:, None, :]
    u = np.stack([np.einsum("ijk,ik->ij", dp, t1), np.einsum("ijk,ik->ij", dp, t2)], axis=2)
    w = np.stack([np.einsum("ijk,ik->ij", dn, t1), np.einsum("ijk,ik->ij", dn, t2)], axis=2)
    mask = found.astype(np.float64)[:, :, None]
    u *= mask
    w *= mask
    uu = np.einsum("ijk,ijl->ikl", u, u)  # (n, 2, 2)
    wu = np.einsum("ijk,ijl->ikl", w, u)  # (n, 2, 2)
    # Regularise so a vertex with too few neighbours gives zero, not noise.
    uu += 1e-9 * np.eye(2)[None, :, :]
    shape = wu @ np.linalg.inv(uu)
    shape = 0.5 * (shape + np.transpose(shape, (0, 2, 1)))
    eigenvalues, eigenvectors = np.linalg.eigh(shape)
    kappa = eigenvalues[:, 1]  # the largest, signed: convex positive
    direction_2d = eigenvectors[:, :, 1]
    direction = direction_2d[:, :1] * t1 + direction_2d[:, 1:] * t2
    enough = found.sum(axis=1) >= 6
    kappa = np.where(enough, kappa, 0.0)

    # The derivative of the largest curvature along its own direction,
    # from a plane fitted to the curvature over the same neighbours.
    dk = (kappa[nearest] - kappa[:, None]) * mask[:, :, 0]
    uk = np.einsum("ijk,ij->ik", u, dk)
    gradient_2d = np.einsum("ikl,il->ik", np.linalg.inv(uu), uk)
    slope = np.einsum("ik,ik->i", gradient_2d, direction_2d)

    # Every mesh edge once, with its two vertices.
    edge_a = np.concatenate([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    edge_b = np.concatenate([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    low = np.minimum(edge_a, edge_b)
    high = np.maximum(edge_a, edge_b)
    keys = low.astype(np.int64) * (count + 1) + high
    unique_keys, first = np.unique(keys, return_index=True)
    low, high = low[first], high[first]
    # Orient the far vertex's direction with the near one's, and read the
    # slope at both along the near one's direction.
    agree = np.einsum("ij,ij->i", direction[low], direction[high])
    sign = np.where(agree < 0.0, -1.0, 1.0)
    slope_low = slope[low]
    slope_high = slope[high] * sign
    travel = np.einsum("ij,ij->i", points[high] - points[low], direction[low])
    travel_sign = np.where(travel < 0.0, -1.0, 1.0)
    crossing = (
        (kappa[low] > 0.0)
        & (kappa[high] > 0.0)
        & (np.abs(agree) >= 0.5)
        & (slope_low * slope_high < 0.0)
        & ((slope_low - slope_high) * travel_sign > 0.0)
    )
    t = np.zeros(low.shape[0], dtype=np.float64)
    denominator = slope_low - slope_high
    ok = crossing & (np.abs(denominator) > 1e-12)
    t[ok] = slope_low[ok] / denominator[ok]
    kappa_at = kappa[low] + t * (kappa[high] - kappa[low])
    # The bend the curvature implies over twice the scale; a fitted
    # curvature at a sharp edge can imply more than a half turn, and a
    # surface cannot bend more than that within itself.
    bend = np.minimum(np.degrees(kappa_at * 2.0 * scale_mm), 180.0)
    crossing &= ok & (bend >= dihedral_min_deg)
    if not crossing.any():
        return []
    crossing_point = points[low] + t[:, None] * (points[high] - points[low])

    # Edge index per triangle side, to find the crossings each triangle has.
    edge_index = np.searchsorted(unique_keys, keys).reshape(3, -1).T  # (m, 3)
    crossed = crossing[edge_index]
    two = crossed.sum(axis=1) == 2
    if not two.any():
        return []
    pairs = np.array([np.flatnonzero(row) for row in crossed[two]])  # (s, 2) local sides
    faces_two = edge_index[two]
    segment_a = faces_two[np.arange(faces_two.shape[0]), pairs[:, 0]]
    segment_b = faces_two[np.arange(faces_two.shape[0]), pairs[:, 1]]

    # String the segments into chains through their shared crossings.
    incident: dict[int, list[int]] = {}
    for index, (a, b) in enumerate(zip(segment_a.tolist(), segment_b.tolist())):
        incident.setdefault(a, []).append(index)
        incident.setdefault(b, []).append(index)
    used = np.zeros(segment_a.shape[0], dtype=bool)
    chains: list[CreaseChain] = []

    def walk(start_crossing: int, start_segment: int) -> None:
        crossing_id = start_crossing
        segment = start_segment
        run_crossings = [crossing_id]
        run_segments = []
        while True:
            used[segment] = True
            run_segments.append(segment)
            a, b = int(segment_a[segment]), int(segment_b[segment])
            crossing_id = b if a == crossing_id else a
            run_crossings.append(crossing_id)
            following = [s for s in incident[crossing_id] if not used[s]]
            if len(incident[crossing_id]) != 2 or not following:
                break
            segment = following[0]
        ids = np.asarray(run_crossings)
        seg = np.asarray(run_segments)
        # One bend per segment, the mean of its two crossings; the faces
        # either side are the two vertices of the segment's first crossing.
        first_crossing = np.where(segment_a[seg] == ids[:-1], segment_a[seg], segment_b[seg])
        chains.append(
            CreaseChain(
                points_mm=crossing_point[ids],
                dihedral_deg=0.5 * (bend[ids[:-1]] + bend[ids[1:]]),
                left_normals=normals[low[first_crossing]],
                right_normals=normals[high[first_crossing]],
            )
        )

    for crossing_id in sorted(incident):
        if len(incident[crossing_id]) == 2:
            continue
        for segment in incident[crossing_id]:
            if not used[segment]:
                walk(crossing_id, segment)
    for segment in range(segment_a.shape[0]):
        if not used[segment]:
            walk(int(segment_a[segment]), segment)
    return chains


def link_chains(
    chains: Sequence[CreaseChain],
    *,
    gap_mm: float,
    angle_deg: float = CREASE_LINK_ANGLE_DEG,
    rounds: int = CREASE_LINK_ROUNDS,
) -> tuple[CreaseChain, ...]:
    """Join chains end to end where two ends point at each other.

    Two ends join when they lie within ``gap_mm`` and each end's outward
    direction points at the other within ``angle_deg``; a shared vertex is
    a gap of zero.  Where several ends meet, the pair that points at each
    other most directly joins and the rest stay separate, so a ridge runs
    through a junction and the branch stays its own line.  A gap is bridged
    by a straight segment that borrows the dihedral and normals of the edge
    it continues.  Repeated until nothing joins, at most ``rounds`` times.
    Deterministic: ties go to the earlier chains.
    """

    from scipy.spatial import cKDTree  # noqa: PLC0415

    if float(gap_mm) < 0.0 or not np.isfinite(float(gap_mm)):
        raise ArtifactCreaseError("gap_mm must be zero or a positive length")
    if not (0.0 < float(angle_deg) < 90.0):
        raise ArtifactCreaseError("angle_deg must lie strictly between 0 and 90")
    polylines = [np.asarray(chain.points_mm, dtype=np.float64) for chain in chains]
    dihedrals = [np.asarray(chain.dihedral_deg, dtype=np.float64) for chain in chains]
    lefts = [np.asarray(chain.left_normals, dtype=np.float64) for chain in chains]
    rights = [np.asarray(chain.right_normals, dtype=np.float64) for chain in chains]
    cos_min = float(np.cos(np.radians(float(angle_deg))))
    for _round in range(max(int(rounds), 0)):
        ends: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for index, polyline in enumerate(polylines):
            if polyline.shape[0] < 2:
                continue
            # The outward direction over the last few points, so one noisy
            # edge does not decide where the chain is heading.
            head = polyline[0] - polyline[min(3, polyline.shape[0] - 1)]
            tail = polyline[-1] - polyline[max(-4, -polyline.shape[0])]
            ends.append((index, 0, polyline[0], head / max(float(np.linalg.norm(head)), 1e-12)))
            ends.append((index, 1, polyline[-1], tail / max(float(np.linalg.norm(tail)), 1e-12)))
        if len(ends) < 2:
            break
        tree = cKDTree(np.array([end[2] for end in ends]))
        scored: list[tuple[float, int, int]] = []
        for a, b in sorted(tree.query_pairs(r=float(gap_mm))):
            chain_a, _end_a, point_a, direction_a = ends[a]
            chain_b, _end_b, point_b, direction_b = ends[b]
            if chain_a == chain_b:
                continue
            gap = point_b - point_a
            distance = float(np.linalg.norm(gap))
            facing = float(direction_a @ -direction_b)
            if distance > 1e-9:
                unit = gap / distance
                if float(direction_a @ unit) < cos_min or float(direction_b @ -unit) < cos_min:
                    continue
            elif facing < cos_min:
                continue
            scored.append((distance - facing, a, b))
        scored.sort()
        taken: set[int] = set()
        merged: set[int] = set()
        for _score, a, b in scored:
            if a in taken or b in taken:
                continue
            chain_a, end_a, _point_a, _direction_a = ends[a]
            chain_b, end_b, _point_b, _direction_b = ends[b]
            if chain_a in merged or chain_b in merged:
                continue
            first, first_d, first_l, first_r = (
                polylines[chain_a],
                dihedrals[chain_a],
                lefts[chain_a],
                rights[chain_a],
            )
            second, second_d, second_l, second_r = (
                polylines[chain_b],
                dihedrals[chain_b],
                lefts[chain_b],
                rights[chain_b],
            )
            # Orient so the first chain's tail meets the second chain's head.
            if end_a == 0:
                first, first_d, first_l, first_r = (
                    first[::-1],
                    first_d[::-1],
                    first_l[::-1],
                    first_r[::-1],
                )
            if end_b == 1:
                second, second_d, second_l, second_r = (
                    second[::-1],
                    second_d[::-1],
                    second_l[::-1],
                    second_r[::-1],
                )
            if np.allclose(first[-1], second[0]):
                joined = np.vstack([first, second[1:]])
                joined_d = np.concatenate([first_d, second_d])
                joined_l = np.vstack([first_l, second_l])
                joined_r = np.vstack([first_r, second_r])
            else:
                joined = np.vstack([first, second])
                bridge_d = np.array([min(float(first_d[-1]), float(second_d[0]))])
                joined_d = np.concatenate([first_d, bridge_d, second_d])
                joined_l = np.vstack([first_l, first_l[-1:], second_l])
                joined_r = np.vstack([first_r, first_r[-1:], second_r])
            polylines.append(joined)
            dihedrals.append(joined_d)
            lefts.append(joined_l)
            rights.append(joined_r)
            merged.update((chain_a, chain_b))
            taken.update((a, b))
        if not merged:
            break
        survivors = [index for index in range(len(polylines)) if index not in merged]
        polylines = [polylines[index] for index in survivors]
        dihedrals = [dihedrals[index] for index in survivors]
        lefts = [lefts[index] for index in survivors]
        rights = [rights[index] for index in survivors]
    return tuple(
        CreaseChain(points_mm=p, dihedral_deg=d, left_normals=left, right_normals=right)
        for p, d, left, right in zip(polylines, dihedrals, lefts, rights)
    )


def detect_convex_creases(
    vertices: object,
    faces: object,
    *,
    dihedral_min_deg: float = DEFAULT_CREASE_DIHEDRAL_MIN_DEG,
    min_length_mm: float = DEFAULT_CREASE_MIN_LENGTH_MM,
    scale_mm: float = 0.0,
    link_mm: float = 0.0,
    crest_rule: str = DEFAULT_CREST_RULE,
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
    the edges that bend enough at that scale, only the crest, picked by
    ``crest_rule``.  On a scanned mesh the crest comes out in pieces - the
    kept edges do not all touch, and a ridge forks where the noise does -
    and ``link_mm`` above zero joins chain ends that point at each other
    within that distance before the short ones are dropped (``link_chains``).
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
    if not np.isfinite(float(link_mm)) or float(link_mm) < 0.0:
        raise ArtifactCreaseError("link_mm must be zero or a positive length")
    if crest_rule not in CREST_RULES:
        raise ArtifactCreaseError(f"crest_rule must be one of {sorted(CREST_RULES)}")
    if triangles.size == 0:
        return ()
    points, triangles = _welded(points, triangles)

    corners = points[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0.0):
        raise ArtifactCreaseError("faces contain a degenerate triangle")
    normals /= lengths[:, None]

    if float(scale_mm) > 0.0 and crest_rule == CREST_RULE_CURVATURE_V2:
        chains = _ridge_chains_at_scale(
            points,
            triangles,
            normals,
            0.5 * lengths,
            scale_mm=float(scale_mm),
            dihedral_min_deg=float(dihedral_min_deg),
        )
    else:
        chains = _edge_chains(
            points,
            triangles,
            corners,
            normals,
            scale_mm=float(scale_mm),
            dihedral_min_deg=float(dihedral_min_deg),
        )
    if not chains:
        return ()
    ordered: Sequence[CreaseChain] = sorted(
        chains, key=lambda chain: (-chain.length_mm, chain.points_mm[0].tolist())
    )
    if float(link_mm) > 0.0:
        ordered = link_chains(ordered, gap_mm=float(link_mm))
    kept = tuple(chain for chain in ordered if chain.length_mm >= float(min_length_mm))
    return tuple(sorted(kept, key=lambda chain: (-chain.length_mm, chain.points_mm[0].tolist())))


def _edge_chains(
    points: np.ndarray,
    triangles: np.ndarray,
    corners: np.ndarray,
    normals: np.ndarray,
    *,
    scale_mm: float,
    dihedral_min_deg: float,
) -> list[CreaseChain]:
    """Chains of the mesh's own edges: the edge reading, or the first crest
    rule when ``scale_mm`` is above zero."""

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
    if scale_mm > 0.0:
        keep = _crest_edges_at_scale(
            points,
            corners,
            normals,
            low[first],
            high[first],
            face_of[first],
            face_of[second],
            dihedral,
            scale_mm=scale_mm,
            dihedral_min_deg=dihedral_min_deg,
        )
    else:
        keep = (dihedral >= dihedral_min_deg) & (below < 0.0)
    crease_edges = np.stack([low[first][keep], high[first][keep]], axis=1)
    crease_dihedral = dihedral[keep]
    crease_n1, crease_n2 = n1[keep], n2[keep]
    if crease_edges.shape[0] == 0:
        return []

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
    return chains


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
    "CREASE_LINK_ANGLE_DEG",
    "CREASE_LINK_ROUNDS",
    "CREST_RULES",
    "CREST_RULE_CURVATURE_V2",
    "CREST_RULE_TURNING_V1",
    "DEFAULT_CREASE_DIHEDRAL_MIN_DEG",
    "DEFAULT_CREASE_MIN_LENGTH_MM",
    "DEFAULT_CREST_RULE",
    "ArtifactCreaseError",
    "CreaseChain",
    "crease_summary",
    "creases_seen_from",
    "detect_convex_creases",
    "link_chains",
]
