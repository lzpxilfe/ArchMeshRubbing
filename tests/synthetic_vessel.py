"""A hollow vessel of revolution for tests that need a pot, not a tile.

Shared by the drawing and unwrap tests.  The profile has a bounded slope, so
no band of the wall is edge-on in the front view, and the mesh is spun by half
a segment so the x-z plane a section is cut on passes between vertex columns.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.core.artifact_session import ArtifactSession
from src.core.artifact_surface_measurement import (
    ArtifactSurfaceMeasurementComputation,
    commit_artifact_surface_measurement,
    extract_surface_measurement,
    resolve_surface_anchor_from_ray,
    surface_diameter_recipe,
    surface_measurement_selection_hash,
)
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


HEIGHT_MM = 90.0
FLOOR_MM = 10.0
WALL_MM = 7.0
RIM_ID = "record:circle-rim"
FLOOR_ID = "record:circle-floor"


def outer_radius(z_mm: float) -> float:
    t = z_mm / HEIGHT_MM
    return 25.0 + 22.0 * t + 9.0 * math.sin(math.pi * t)


Relief = Callable[[float, float], float]


def hollow_vessel(
    *,
    segments: int = 24,
    rings: int = 10,
    relief: Relief | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return (vertices, faces, rim circle points, floor circle points).

    ``relief(angle_rad, z_mm)`` is a radial offset in millimetres added to
    the outer wall only: a stamped or corded surface a rubbing has to show.
    """

    phase = math.pi / segments
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    def ring(radius: float, z_mm: float, *, textured: bool = False) -> int:
        start = len(vertices)
        for segment in range(segments):
            angle = phase + 2.0 * math.pi * segment / segments
            r = radius
            if textured and relief is not None:
                r += float(relief(angle, z_mm))
            vertices.append([r * math.cos(angle), r * math.sin(angle), z_mm])
        return start

    def band(lower: int, upper: int, *, inward: bool = False) -> None:
        # Wound so the normal points out of the clay: away from the axis on the
        # outer wall, towards it on the inner wall.  A scan is oriented the
        # same way, and it is what lets the outside be told from the inside.
        for segment in range(segments):
            following = (segment + 1) % segments
            first = [lower + segment, lower + following, upper + following]
            second = [lower + segment, upper + following, upper + segment]
            if inward:
                first.reverse()
                second.reverse()
            faces.append(first)
            faces.append(second)

    outer = [
        ring(
            outer_radius(HEIGHT_MM * index / rings),
            HEIGHT_MM * index / rings,
            textured=True,
        )
        for index in range(rings + 1)
    ]
    for index in range(rings):
        band(outer[index], outer[index + 1])
    inner_heights = [
        FLOOR_MM + (HEIGHT_MM - FLOOR_MM) * index / rings for index in range(rings + 1)
    ]
    inner = [ring(outer_radius(z) - WALL_MM, z) for z in inner_heights]
    for index in range(rings):
        band(inner[index], inner[index + 1], inward=True)
    band(outer[rings], inner[rings])

    base_center = len(vertices)
    vertices.append([0.0, 0.0, 0.0])
    for segment in range(segments):
        faces.append(
            [base_center, outer[0] + (segment + 1) % segments, outer[0] + segment]
        )
    floor_center = len(vertices)
    vertices.append([0.0, 0.0, FLOOR_MM])
    for segment in range(segments):
        faces.append(
            [floor_center, inner[0] + segment, inner[0] + (segment + 1) % segments]
        )

    rim_radius = outer_radius(HEIGHT_MM) - WALL_MM / 2.0
    floor_radius = (outer_radius(FLOOR_MM) - WALL_MM) * 0.6
    quarters = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
    rim_points = [
        np.array([rim_radius * math.cos(a), rim_radius * math.sin(a), HEIGHT_MM])
        for a in quarters
    ]
    floor_points = [
        np.array([floor_radius * math.cos(a), floor_radius * math.sin(a), FLOOR_MM])
        for a in quarters
    ]
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
        rim_points,
        floor_points,
    )


def outer_wall_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Faces whose normal points away from the axis: the outside of the wall."""

    tri = vertices[faces.astype(np.int64)]
    centroids = tri.mean(axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    radial = centroids.copy()
    radial[:, 2] = 0.0
    radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
    return (np.einsum("ij,ij->i", normals, radial) > 0.5) & (
        np.abs(normals[:, 2]) < 0.8
    )


def meridional_strip_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    center_angle_rad: float,
    width_mm: float,
) -> np.ndarray:
    """Outer-wall faces within a strip of constant width about one meridian."""

    tri = vertices[faces.astype(np.int64)]
    centroids = tri.mean(axis=1)
    theta = np.arctan2(centroids[:, 1], centroids[:, 0])
    radius = np.hypot(centroids[:, 0], centroids[:, 1])
    offset = np.angle(np.exp(1j * (theta - center_angle_rad)))
    within = np.abs(offset) <= (width_mm / 2.0) / np.maximum(radius, 1e-9)
    return np.flatnonzero(outer_wall_faces(vertices, faces) & within)


def _anchor(vertices: np.ndarray, faces: np.ndarray, point: np.ndarray) -> Any:
    return resolve_surface_anchor_from_ray(
        vertices,
        faces,
        source_faces=faces,
        ray_origin_world_mm=point + np.asarray([0.0, 0.0, 1.0]),
        ray_direction_world=[0.0, 0.0, -1.0],
        depth_point_world_mm=point,
        pixel_footprint_um=10,
    )


def _commit_circle(
    session: ArtifactSession,
    vertices: np.ndarray,
    faces: np.ndarray,
    points: list[np.ndarray],
    *,
    record_id: str,
    created_at: str,
) -> ArtifactSession:
    recipe = surface_diameter_recipe(
        [_anchor(vertices, faces, point) for point in points],
        source_vertex_count=int(vertices.shape[0]),
        source_face_count=int(faces.shape[0]),
    )
    receipt, qc = extract_surface_measurement(vertices, faces, recipe)
    context = session.capture_operation(
        recipe=recipe,
        selection_hash=surface_measurement_selection_hash(recipe),
    )
    return commit_artifact_surface_measurement(
        session,
        ArtifactSurfaceMeasurementComputation(
            context=context,
            projection_snapshot=session.projection_snapshot(),
            receipt=receipt,
            recipe=recipe,
            qc=qc,
        ),
        record_id=record_id,
        created_at=created_at,
        operator="tester",
    )


def positioned_vessel_session(
    *,
    segments: int = 24,
    rings: int = 10,
    document_id: str = "artifact:vessel",
    relief: Relief | None = None,
) -> tuple[ArtifactSession, np.ndarray, np.ndarray]:
    """A vessel stood on its axis by two measured circles, plus its arrays."""

    vertices, faces, rim_points, floor_points = hollow_vessel(
        segments=segments, rings=rings, relief=relief
    )
    mesh = MeshData(
        vertices=vertices,
        faces=faces,
        unit="mm",
        filepath=Path("/source/vessel.ply"),
        source_identity=SourceFingerprint(
            sha256="7" * 64,
            size_bytes=8192,
            mtime_ns=1,
            original_name="vessel.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/vessel.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="vessel-test",
        operator="tester",
        created_at="2026-09-03T00:00:00Z",
        document_id=document_id,
        metadata_revision_id=f"metadata:{document_id}",
        align_revision_id=f"align:{document_id}",
    )
    session = _commit_circle(
        session, vertices, faces, floor_points,
        record_id=FLOOR_ID, created_at="2026-09-03T00:00:01Z",
    )
    session = _commit_circle(
        session, vertices, faces, rim_points,
        record_id=RIM_ID, created_at="2026-09-03T00:00:02Z",
    )
    session = session.commit_axis_alignment(
        top_record_id=RIM_ID,
        bottom_record_id=FLOOR_ID,
        operator="tester",
        created_at="2026-09-03T00:00:03Z",
        revision_id="align:axis",
    )
    return session, vertices, faces


# ---------------------------------------------------------------------------
# A fired wall's micro-relief, for tests that need a rough surface: finger
# drag as value noise at two scales, and sand temper as round grains and pits.
# Deterministic, and shaped so a 0.19-0.25 mm mesh can carry it.


def _hash01(ix: int, iy: int, seed: int) -> float:
    h = (ix * 73856093) ^ (iy * 19349663) ^ (seed * 83492791)
    h &= 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    h ^= h >> 15
    return h / 4294967296.0


def value_noise(x_mm: float, y_mm: float, *, cell_mm: float, seed: int) -> float:
    """Smooth deterministic noise in [-1, 1] with features about cell_mm across."""

    gx, gy = x_mm / cell_mm, y_mm / cell_mm
    ix, iy = math.floor(gx), math.floor(gy)
    fx, fy = gx - ix, gy - iy
    sx, sy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a = _hash01(ix, iy, seed)
    b = _hash01(ix + 1, iy, seed)
    c = _hash01(ix, iy + 1, seed)
    d = _hash01(ix + 1, iy + 1, seed)
    return 2.0 * ((a * (1 - sx) + b * sx) * (1 - sy) + (c * (1 - sx) + d * sx) * sy) - 1.0


def _rotated(x_mm: float, y_mm: float, degrees: float) -> tuple[float, float]:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return c * x_mm - s * y_mm, s * x_mm + c * y_mm


def temper(x_mm: float, y_mm: float, *, cell_mm: float, seed: int) -> float:
    """Sand temper: at most one round grain or pit per cell, at a hashed spot."""

    gx, gy = x_mm / cell_mm, y_mm / cell_mm
    ix, iy = math.floor(gx), math.floor(gy)
    total = 0.0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx, cy = ix + dx, iy + dy
            if _hash01(cx, cy, seed + 5) > 0.7:
                continue
            px = cx + _hash01(cx, cy, seed)
            py = cy + _hash01(cx, cy, seed + 1)
            radius = (0.14 + 0.16 * _hash01(cx, cy, seed + 2)) / cell_mm
            d2 = ((gx - px) ** 2 + (gy - py) ** 2) / (radius * radius)
            if d2 < 4.0:
                sign = -1.0 if _hash01(cx, cy, seed + 3) < 0.55 else 1.0
                height = 0.012 + 0.016 * _hash01(cx, cy, seed + 4)
                total += sign * height * math.exp(-d2)
    return total


def grain(angle_rad: float, z_mm: float, radius_mm: float = 40.0) -> float:
    """Micro-relief of a fired wall, in millimetres: drag, grains and pits."""

    arc = angle_rad * radius_mm
    cx, cy = _rotated(arc, z_mm, 31.0)
    fx, fy = _rotated(arc, z_mm, -58.0)
    coarse = 0.025 * value_noise(cx, cy, cell_mm=0.6, seed=3)
    fine = 0.010 * value_noise(fx, fy, cell_mm=0.25, seed=11)
    return coarse + fine + temper(arc, z_mm, cell_mm=0.45, seed=101)


def incision(offset_mm: float, *, half_width_mm: float, depth_mm: float) -> float:
    """A drawn line has a rounded bed, not a slot: a cosine trough."""

    if abs(offset_mm) >= half_width_mm:
        return 0.0
    return -depth_mm * 0.5 * (1.0 + math.cos(math.pi * offset_mm / half_width_mm))


def grained_surface(angle_rad: float, z_mm: float) -> float:
    """Three incised lines, a corded band and one more line, on a grained wall."""

    relief = grain(angle_rad, z_mm)
    for centre in (26.0, 30.0, 34.0):
        relief += incision(z_mm - centre, half_width_mm=1.1, depth_mm=0.2)
    if 38.0 < z_mm < 52.0:
        # The band fades in and out over a millimetre: a paddle's edge is
        # not a cliff, and a cliff on a fine mesh is a face the unwrap refuses.
        edge = min(z_mm - 38.0, 52.0 - z_mm, 1.0)
        window = 0.5 * (1.0 - math.cos(math.pi * edge))
        relief += window * 0.25 * max(
            0.0, math.sin(2.0 * math.pi * (z_mm + 6.0 * angle_rad) / 4.0)
        )
    relief += incision(z_mm - 58.0, half_width_mm=1.1, depth_mm=0.18)
    return relief
