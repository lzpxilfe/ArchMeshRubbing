"""What a real scan carries that a generated solid does not.

Everything the program was tested against until now was a closed, clean,
axis-aligned solid.  A museum scan is none of those.  The 국립중앙박물관 scan of
빗살무늬토기 신수22891 - 391,456 faces, published under 공공누리 1유형 - is one
body of 391,432 faces and one loose crumb of 24, has 11 boundary edges and
28 non-manifold edges, and stands on Y rather than Z.  See
docs/REAL_DATA_TRIAL.md for the numbers and where the file came from.

These functions put those things onto a generated mesh, one at a time, so a
test can ask what the program does when it meets each.  They are
deterministic - hashed, never a random state - so a test that fails fails
the same way twice.
"""

from __future__ import annotations

import math

import numpy as np


def _hash01(*values: int) -> float:
    """A settled number in [0, 1) for a tuple of integers."""

    h = 0x9E3779B9
    for value in values:
        h ^= (int(value) * 0x85EBCA6B) & 0xFFFFFFFF
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return (h & 0xFFFFFFFF) / 4294967296.0


def add_loose_crumb(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    size_mm: float = 1.5,
    gap_mm: float = 3.0,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Put a small detached patch of triangles beside the artifact.

    A scanner leaves them: a fleck of the mount, a piece of the turntable, a
    shadow that reconstructed as its own island.  The crumb here is a closed
    tetrahedron, so it is a solid in its own right and cannot be mistaken for
    a hole in the artifact.  It sits ``gap_mm`` outside the bounding box on
    +X, at a hashed height, which is where a real one usually turns up: near
    the object but not touching it.
    """

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    low, high = points.min(axis=0), points.max(axis=0)
    height = float(low[2] + (high[2] - low[2]) * (0.3 + 0.4 * _hash01(seed, 7)))
    origin = np.array(
        [float(high[0]) + float(gap_mm), float(low[1] + high[1]) / 2.0, height]
    )
    s = float(size_mm)
    crumb = origin + np.array(
        [
            [0.0, 0.0, 0.0],
            [s, 0.0, 0.0],
            [s * 0.5, s * 0.87, 0.0],
            [s * 0.5, s * 0.29, s * 0.82],
        ]
    )
    base = points.shape[0]
    crumb_faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64
    ) + base
    return (
        np.vstack([points, crumb]),
        np.vstack([triangles, crumb_faces]),
    )


def punch_hole(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    centre_mm: tuple[float, float, float],
    radius_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Take a bite out of the surface, leaving an open boundary.

    A scan has them wherever the scanner could not see: under a rim, inside a
    narrow mouth, where the object sat on its mount.  The vertices are left
    alone so the numbering a selection refers to does not shift.
    """

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    centre = np.asarray(centre_mm, dtype=np.float64)
    centroids = points[triangles].mean(axis=1)
    keep = np.linalg.norm(centroids - centre, axis=1) > float(radius_mm)
    if not keep.any():
        raise ValueError("the hole would take every face")
    return points, triangles[keep]


def bite_the_rim(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    from_angle_deg: float,
    to_angle_deg: float,
    depth_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Break a piece out of the top edge, the way a pot is usually found.

    Faces above ``depth_mm`` below the top, between the two angles about the
    Z axis, are taken away.  The edge left behind runs along whatever
    triangles happened to be there, which is what makes it look broken rather
    than cut.
    """

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    centroids = points[triangles].mean(axis=1)
    top = float(points[:, 2].max())
    angle = np.degrees(np.arctan2(centroids[:, 1], centroids[:, 0]))
    span = (angle - float(from_angle_deg)) % 360.0
    inside = span <= (float(to_angle_deg) - float(from_angle_deg)) % 360.0
    bitten = inside & (centroids[:, 2] > top - float(depth_mm))
    if bitten.all():
        raise ValueError("the bite would take every face")
    return points, triangles[~bitten]


def bridge_the_wall(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    z_mm: float,
    from_angle_deg: float,
    to_angle_deg: float,
    band_mm: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Put a break across the wall, with both fracture faces scanned.

    The museum's 빗살무늬토기 is mended from sherds, and at one join the scanner
    meshed the inside of the wall: between z -250 and -240 mm the gap between
    the outer and inner surfaces, 7 to 11 mm everywhere else, is filled with
    vertices 1 to 2 mm apart, and the scan's 11 boundary and 28 non-manifold
    edges all sit there.  A section through it closes into two loops - the
    wall looks cut through at that height - and the drawing shows two bars
    across the wall where the drafter expects one line.

    This makes that: over the angle span the wall's skin between the mesh
    ring just below ``z_mm`` and the ring just above it is taken away, and
    each of those rings is stitched across the wall thickness, so the lower
    part of the wall is closed by one fracture face and the upper part by
    the other.  The sides of the window are left open, as the real join's
    edges are.  No vertex is added or moved.
    """

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    radius = np.hypot(points[:, 0], points[:, 1])
    angle = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    span = (angle - float(from_angle_deg)) % 360.0
    width = (float(to_angle_deg) - float(from_angle_deg)) % 360.0
    in_span = span <= width
    near = np.abs(points[:, 2] - float(z_mm)) <= float(band_mm)
    candidates = np.flatnonzero(in_span & near & (radius > 1e-9))
    if candidates.size < 4:
        raise ValueError("no wall within the band to bridge")
    # The outer surface is the larger radius in the band, the inner the
    # smaller; on each, the ring just below z_mm and the ring just above.
    mid_radius = (radius[candidates].max() + radius[candidates].min()) / 2.0
    rings: dict[tuple[str, str], np.ndarray] = {}
    for side, on_side in (
        ("outer", radius[candidates] > mid_radius),
        ("inner", radius[candidates] < mid_radius),
    ):
        on = candidates[on_side]
        heights = np.unique(np.round(points[on, 2], 6))
        below = heights[heights <= float(z_mm)]
        above = heights[heights > float(z_mm)]
        if below.size == 0 or above.size == 0:
            raise ValueError(f"the band holds no ring on both sides of z on the {side} wall")
        for level, height in (("low", below.max()), ("high", above.min())):
            ring = on[np.abs(points[on, 2] - height) < 1e-6]
            rings[(side, level)] = ring[np.argsort(span[ring])]

    def stitch(outer: np.ndarray, inner: np.ndarray) -> list[list[int]]:
        # Walk both rings by angle, always advancing the one that lags, so
        # the face between them is a fan of triangles with no crossing.
        added: list[list[int]] = []
        i = j = 0
        while i < outer.size - 1 or j < inner.size - 1:
            advance_outer = j >= inner.size - 1 or (
                i < outer.size - 1 and span[outer[i + 1]] <= span[inner[j + 1]]
            )
            if advance_outer:
                added.append([int(outer[i]), int(outer[i + 1]), int(inner[j])])
                i += 1
            else:
                added.append([int(outer[i]), int(inner[j + 1]), int(inner[j])])
                j += 1
        return added

    added = stitch(rings[("outer", "low")], rings[("inner", "low")]) + stitch(
        rings[("outer", "high")], rings[("inner", "high")]
    )
    # The skin between the two rings, on both surfaces, within the span.
    centroids = points[triangles].mean(axis=1)
    centroid_span = (np.degrees(np.arctan2(centroids[:, 1], centroids[:, 0])) - float(from_angle_deg)) % 360.0
    centroid_radius = np.hypot(centroids[:, 0], centroids[:, 1])
    window = np.zeros(triangles.shape[0], dtype=bool)
    for side, on_side in (
        ("outer", centroid_radius > mid_radius),
        ("inner", centroid_radius < mid_radius),
    ):
        low = float(points[rings[(side, "low")][0], 2])
        high = float(points[rings[(side, "high")][0], 2])
        window |= (
            on_side
            & (centroid_span <= width)
            & (centroids[:, 2] > low)
            & (centroids[:, 2] < high)
            & (np.abs(centroids[:, 2] - float(z_mm)) <= float(band_mm))
        )
    if not window.any():
        raise ValueError("no skin between the rings to take away")
    return points, np.vstack([triangles[~window], np.asarray(added, dtype=np.int64)])


def stand_it_wrong(
    vertices: np.ndarray,
    *,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Put the artifact where a scan file actually puts it.

    A published scan is in the scanner's frame, not the artifact's: the
    museum's 빗살무늬토기 stands on Y with its centroid nowhere near the origin.
    Standing it up is what 정치 is for, and a test that starts on +Z never
    exercises it.
    """

    points = np.asarray(vertices, dtype=np.float64)
    a, b, c = (math.radians(float(d)) for d in (roll_deg, pitch_deg, yaw_deg))
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]]
    )
    ry = np.array(
        [[math.cos(b), 0.0, math.sin(b)], [0.0, 1.0, 0.0], [-math.sin(b), 0.0, math.cos(b)]]
    )
    rz = np.array(
        [[math.cos(c), -math.sin(c), 0.0], [math.sin(c), math.cos(c), 0.0], [0.0, 0.0, 1.0]]
    )
    return points @ (rz @ ry @ rx).T + np.asarray(offset_mm, dtype=np.float64)


def warp(
    vertices: np.ndarray,
    *,
    oval_mm: float = 0.0,
    lean_mm: float = 0.0,
    sag_mm: float = 0.0,
) -> np.ndarray:
    """Take the artifact off being a surface of revolution.

    A thrown pot dries and fires out of true.  ``oval_mm`` makes the section
    an ellipse rather than a circle, ``lean_mm`` slides the axis sideways from
    bottom to top, and ``sag_mm`` swells one side of the wall.  All three are
    the artifact's own shape, not noise, so they scale with height rather than
    wobbling: no fit of a rotation axis makes them go away.
    """

    points = np.asarray(vertices, dtype=np.float64).copy()
    low, high = float(points[:, 2].min()), float(points[:, 2].max())
    span = max(high - low, 1e-9)
    t = (points[:, 2] - low) / span
    if oval_mm:
        radius = np.hypot(points[:, 0], points[:, 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            angle = np.arctan2(points[:, 1], points[:, 0])
        scale = np.where(radius > 1e-9, 1.0 + float(oval_mm) * np.cos(2.0 * angle) / np.maximum(radius, 1e-9), 1.0)
        points[:, 0] *= scale
        points[:, 1] *= scale
    if lean_mm:
        points[:, 0] += float(lean_mm) * t
    if sag_mm:
        points[:, 1] += float(sag_mm) * np.sin(math.pi * t) * (points[:, 1] > 0.0)
    return points


def mesh_report(vertices: np.ndarray, faces: np.ndarray) -> dict[str, int]:
    """The numbers a drafter would want before drawing anything.

    The same numbers docs/REAL_DATA_TRIAL.md records for the museum scan, so
    a synthetic mesh can be compared with a real one directly.
    """

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    _unique, inverse = np.unique(np.round(points, 3), axis=0, return_inverse=True)
    welded = np.asarray(inverse).reshape(-1)[triangles]
    edges = np.sort(
        np.stack([welded[:, [0, 1]], welded[:, [1, 2]], welded[:, [2, 0]]]).reshape(-1, 2),
        axis=1,
    )
    _rows, counts = np.unique(edges, axis=0, return_counts=True)
    rows = np.repeat(np.arange(welded.shape[0], dtype=np.int64), 3)
    incidence = coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, welded.reshape(-1))),
        shape=(welded.shape[0], int(_unique.shape[0])),
    ).tocsr()
    pieces, labels = connected_components(incidence @ incidence.T, directed=False)
    sizes = np.bincount(labels)
    return {
        "face_count": int(triangles.shape[0]),
        "welded_vertex_count": int(_unique.shape[0]),
        "boundary_edge_count": int((counts == 1).sum()),
        "nonmanifold_edge_count": int((counts > 2).sum()),
        "connected_piece_count": int(pieces),
        "smallest_piece_faces": int(sizes.min()),
        "largest_piece_faces": int(sizes.max()),
    }


__all__ = [
    "add_loose_crumb",
    "bite_the_rim",
    "bridge_the_wall",
    "mesh_report",
    "punch_hole",
    "stand_it_wrong",
    "warp",
]
