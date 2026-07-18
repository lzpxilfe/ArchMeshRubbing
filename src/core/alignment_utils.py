"""
Geometry helpers for floor/plane alignment workflows.
"""

from __future__ import annotations

import numpy as np


MATRIX_ATOL = 1e-9


def _as_vec3(value: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError("Expected at least 3 values for a 3D vector.")
    return arr[:3]


def _finite_vec3(
    value: np.ndarray | list[float] | tuple[float, ...],
    *,
    field_name: str,
) -> np.ndarray:
    arr = _as_vec3(value).astype(np.float64, copy=False)
    if not np.isfinite(arr).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return arr


def scene_rotation_matrix(
    rotation_deg: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Return the legacy viewport's one authoritative Euler rotation matrix.

    The fixed-function renderer calls ``glRotatef(X)``, then ``Y``, then ``Z``.
    OpenGL post-multiplies the current matrix, so column-vector geometry is
    transformed by ``Rx @ Ry @ Rz``.  In SciPy terminology this is intrinsic
    uppercase ``"XYZ"``; lowercase ``"xyz"`` is a different rotation order.
    Euler angles are a UI adapter only.  Durable alignment uses a 4x4 matrix.
    """

    rx, ry, rz = np.deg2rad(
        _finite_vec3(rotation_deg, field_name="rotation_deg")
    )
    cx, sx = float(np.cos(rx)), float(np.sin(rx))
    cy, sy = float(np.cos(ry)), float(np.sin(ry))
    cz, sz = float(np.cos(rz)), float(np.sin(rz))

    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float64,
    )
    rot_y = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float64,
    )
    rot_z = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rot_x @ rot_y @ rot_z


def scene_trs_matrix(
    translation: np.ndarray | list[float] | tuple[float, ...],
    rotation_deg: np.ndarray | list[float] | tuple[float, ...],
    scale: float,
) -> np.ndarray:
    """Build the legacy scene transform as ``T @ Rx @ Ry @ Rz @ S``.

    Matrices are float64, row-major when serialized, and applied to column
    vectors.  Only a finite positive uniform scale is accepted; scientific
    alignment revisions themselves are rigid and reject scale separately.
    """

    translation_vec = _finite_vec3(translation, field_name="translation")
    scale_value = float(scale)
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError("scale must be a finite positive value")

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scene_rotation_matrix(rotation_deg) * scale_value
    matrix[:3, 3] = translation_vec
    return matrix


def scene_trs_matrix_about_pivot(
    translation: np.ndarray | list[float] | tuple[float, ...],
    rotation_deg: np.ndarray | list[float] | tuple[float, ...],
    scale: float,
    pivot: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Build ``T(delta) @ T(pivot) @ R @ S @ T(-pivot)``.

    Canonical ArtifactDocument geometry retains its scientific origin instead
    of being destructively centered.  This explicit runtime preview pivot
    preserves object-centered interaction while keeping durable vertices and
    Align matrices unmodified until commit.
    """

    pivot_vec = _finite_vec3(pivot, field_name="pivot")
    delta = scene_trs_matrix(translation, rotation_deg, scale)
    to_pivot = np.eye(4, dtype=np.float64)
    to_pivot[:3, 3] = pivot_vec
    from_pivot = np.eye(4, dtype=np.float64)
    from_pivot[:3, 3] = -pivot_vec
    # delta already contains translation. Insert the pivot around its linear
    # component without applying translation twice.
    translation_only = np.eye(4, dtype=np.float64)
    translation_only[:3, 3] = delta[:3, 3]
    linear = delta.copy()
    linear[:3, 3] = 0.0
    return translation_only @ to_pivot @ linear @ from_pivot


def require_affine_matrix4x4(
    matrix: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
    *,
    field_name: str = "matrix4x4",
) -> np.ndarray:
    """Return a finite invertible affine matrix or raise ``ValueError``."""

    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{field_name} must have shape (4, 4)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{field_name} must contain only finite values")
    if not np.allclose(arr[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=MATRIX_ATOL):
        raise ValueError(f"{field_name} must be an affine column-vector matrix")
    determinant = float(np.linalg.det(arr[:3, :3]))
    if not np.isfinite(determinant) or abs(determinant) <= MATRIX_ATOL:
        raise ValueError(f"{field_name} must have an invertible linear component")
    return arr.copy()


def require_rigid_matrix4x4(
    matrix: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
    *,
    field_name: str = "matrix4x4",
) -> np.ndarray:
    """Return a proper rigid transform, rejecting scale, shear and reflection."""

    arr = require_affine_matrix4x4(matrix, field_name=field_name)
    rotation = arr[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float64),
        rtol=0.0,
        atol=MATRIX_ATOL,
    ):
        raise ValueError(f"{field_name} must be rigid (scale/shear are not allowed)")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, rtol=0.0, atol=MATRIX_ATOL):
        raise ValueError(f"{field_name} must be a proper rotation with determinant +1")
    return arr


def compose_align_matrices(
    delta_matrix: np.ndarray | list[list[float]],
    parent_matrix: np.ndarray | list[list[float]],
) -> np.ndarray:
    """Compose an Align delta as ``A_new = delta @ parent``."""

    delta = require_rigid_matrix4x4(delta_matrix, field_name="delta_matrix")
    parent = require_rigid_matrix4x4(parent_matrix, field_name="parent_matrix")
    return require_rigid_matrix4x4(
        delta @ parent,
        field_name="composed_align_matrix",
    )


def transform_points(
    points: np.ndarray | list[list[float]],
    matrix: np.ndarray | list[list[float]],
) -> np.ndarray:
    """Apply a column-vector affine matrix to an ``(..., 3)`` point array."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 0 or pts.shape[-1] != 3:
        raise ValueError("points must have shape (..., 3)")
    if not np.isfinite(pts).all():
        raise ValueError("points must contain only finite values")
    affine = require_affine_matrix4x4(matrix)
    original_shape = pts.shape
    flat = pts.reshape(-1, 3)
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = flat @ affine[:3, :3].T + affine[:3, 3]
    if not np.isfinite(transformed).all():
        raise ValueError("transformed points must contain only finite values")
    return transformed.reshape(original_shape)


def transform_directions(
    directions: np.ndarray | list[list[float]],
    matrix: np.ndarray | list[list[float]],
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Transform direction vectors without applying translation."""

    vectors = np.asarray(directions, dtype=np.float64)
    if vectors.ndim == 0 or vectors.shape[-1] != 3:
        raise ValueError("directions must have shape (..., 3)")
    if not np.isfinite(vectors).all():
        raise ValueError("directions must contain only finite values")
    affine = require_affine_matrix4x4(matrix)
    original_shape = vectors.shape
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = vectors.reshape(-1, 3) @ affine[:3, :3].T
    if not np.isfinite(transformed).all():
        raise ValueError("transformed directions must contain only finite values")
    if normalize:
        lengths = np.linalg.norm(transformed, axis=1)
        if np.any(lengths <= MATRIX_ATOL) or not np.isfinite(lengths).all():
            raise ValueError("cannot normalize a zero-length transformed direction")
        transformed = transformed / lengths[:, None]
    return transformed.reshape(original_shape)


def transform_plane_world_to_local(
    origin_world: np.ndarray | list[float] | tuple[float, ...],
    normal_world: np.ndarray | list[float] | tuple[float, ...],
    local_to_world: np.ndarray | list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a world-space plane into local coordinates exactly."""

    affine = require_affine_matrix4x4(local_to_world, field_name="local_to_world")
    inverse = np.linalg.inv(affine)
    origin_local = transform_points(
        _finite_vec3(origin_world, field_name="origin_world").reshape(1, 3),
        inverse,
    )[0]
    # n_world . (L*x_local + t - origin_world) = 0, so local normal is L^T*n.
    normal = affine[:3, :3].T @ _finite_vec3(normal_world, field_name="normal_world")
    normal_local = normalize_vector(normal)
    if normal_local is None:
        raise ValueError("normal_world produces a degenerate local plane normal")
    return origin_local, normal_local


def bounds_corners(bounds: np.ndarray | list[list[float]]) -> np.ndarray:
    """Return the eight corners of a finite ``[[min], [max]]`` bounds array."""

    arr = np.asarray(bounds, dtype=np.float64)
    if arr.shape != (2, 3) or not np.isfinite(arr).all():
        raise ValueError("bounds must be a finite array with shape (2, 3)")
    if np.any(arr[0] > arr[1]):
        raise ValueError("bounds minimum must not exceed maximum")
    low, high = arr
    return np.asarray(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=np.float64,
    )


def transform_bounds(
    bounds: np.ndarray | list[list[float]],
    matrix: np.ndarray | list[list[float]],
) -> np.ndarray:
    """Transform AABB corners and return the enclosing world-space AABB."""

    corners = transform_points(bounds_corners(bounds), matrix)
    return np.asarray([corners.min(axis=0), corners.max(axis=0)], dtype=np.float64)


def normalize_vector(
    value: np.ndarray | list[float] | tuple[float, ...],
    *,
    eps: float = 1e-12,
) -> np.ndarray | None:
    """Return normalized 3D vector or None when magnitude is near zero."""
    vec = _as_vec3(value)
    nrm = float(np.linalg.norm(vec))
    if (not np.isfinite(nrm)) or nrm <= float(eps):
        return None
    return vec / nrm


def _rotation_matrix_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_n = normalize_vector(axis)
    if axis_n is None:
        return np.eye(3, dtype=np.float64)

    x, y, z = float(axis_n[0]), float(axis_n[1]), float(axis_n[2])
    c = float(np.cos(float(angle_rad)))
    s = float(np.sin(float(angle_rad)))
    cc = 1.0 - c
    return np.array(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ],
        dtype=np.float64,
    )


def rotation_matrix_align_vectors(
    source: np.ndarray | list[float] | tuple[float, ...],
    target: np.ndarray | list[float] | tuple[float, ...],
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Rotation matrix R such that R @ source ~= target.

    Handles anti-parallel vectors (180-degree case) robustly.
    """
    src = normalize_vector(source, eps=eps)
    dst = normalize_vector(target, eps=eps)
    if src is None or dst is None:
        return np.eye(3, dtype=np.float64)

    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot >= 1.0 - eps:
        return np.eye(3, dtype=np.float64)

    if dot <= -1.0 + eps:
        # Build a stable orthogonal axis for 180-degree rotation.
        axis = np.cross(src, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(axis)) <= eps:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(axis)) <= eps:
            axis = np.cross(src, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        return _rotation_matrix_axis_angle(axis, np.pi)

    axis = np.cross(src, dst)
    axis_n = float(np.linalg.norm(axis))
    if axis_n <= eps:
        return np.eye(3, dtype=np.float64)

    axis /= axis_n
    angle = float(np.arccos(dot))
    return _rotation_matrix_axis_angle(axis, angle)


def fit_plane_normal(
    points: np.ndarray,
    *,
    robust: bool = True,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Fit a plane normal from points.

    Returns:
        (normal, centroid) where normal is unit-length.
        None if the input is degenerate.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < 3:
        return None

    work = pts

    # Robust first-pass candidate from deterministic triplet sampling.
    if robust and work.shape[0] >= 5:
        n_pts = int(work.shape[0])
        rng = np.random.default_rng(0)
        max_samples = int(min(256, max(24, n_pts * 3)))

        best_ref_p = None
        best_ref_n = None
        best_score = float("inf")

        for _ in range(max_samples):
            try:
                i, j, k = rng.choice(n_pts, size=3, replace=False).tolist()
            except Exception:
                continue
            a = work[int(i)]
            b = work[int(j)]
            c = work[int(k)]
            cand_n = normalize_vector(np.cross(b - a, c - a), eps=eps)
            if cand_n is None:
                continue

            dist = np.abs((work - a) @ cand_n)
            score = float(np.median(dist))
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_ref_p = a
                best_ref_n = cand_n

        if best_ref_p is not None and best_ref_n is not None:
            dist = np.abs((work - best_ref_p) @ best_ref_n)
            med = float(np.median(dist))
            mad = float(np.median(np.abs(dist - med)))
            if np.isfinite(mad) and mad > eps:
                threshold = med + 3.5 * mad
            else:
                threshold = float(np.percentile(dist, 85.0))
            keep = dist <= max(threshold, eps)
            if int(np.count_nonzero(keep)) >= 3:
                work = work[keep]

    centroid = np.mean(work, axis=0)
    centered = work - centroid
    _u, s, vh = np.linalg.svd(centered, full_matrices=False)
    if s.size < 3:
        return None
    if float(s[1]) <= eps:
        # Nearly collinear picks: cannot define a stable plane.
        return None

    normal = normalize_vector(vh[2, :], eps=eps)
    if normal is None:
        return None
    return normal, centroid


def orient_plane_normal_toward(
    normal: np.ndarray | list[float] | tuple[float, ...],
    plane_point: np.ndarray | list[float] | tuple[float, ...],
    toward_point: np.ndarray | list[float] | tuple[float, ...],
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Orient normal so it points toward `toward_point` from `plane_point`.
    """
    n = normalize_vector(normal, eps=eps)
    if n is None:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    p = _as_vec3(plane_point)
    t = _as_vec3(toward_point)
    if float(np.dot(t - p, n)) < 0.0:
        return -n
    return n


def compute_floor_contact_shift(
    z_values: np.ndarray,
    *,
    tolerance: float = 0.02,
    max_auto_shift: float = 0.2,
) -> float:
    """
    Compute additional +Z shift to resolve minor penetration below Z=0.

    The shift is clamped so large corrections do not unexpectedly "float" the mesh.
    Units are the mesh world units (cm in this project).
    """
    z = np.asarray(z_values, dtype=np.float64).reshape(-1)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0.0

    min_z = float(np.min(z))
    if min_z >= -float(tolerance):
        return 0.0

    shift = -min_z
    if shift > float(max_auto_shift):
        return 0.0
    return float(shift)


def compute_minimax_center_shift(z_values: np.ndarray) -> float:
    """
    Return minimax center shift for 1D values.

    This is the translation `t` that minimizes `max_i |z_i - t|`.
    """
    z = np.asarray(z_values, dtype=np.float64).reshape(-1)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0.0
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if (not np.isfinite(z_min)) or (not np.isfinite(z_max)):
        return 0.0
    return float(0.5 * (z_min + z_max))


def compute_nonpenetration_lift(
    z_values: np.ndarray,
    *,
    floor_z: float = 0.0,
    eps: float = 1e-12,
) -> float:
    """
    Return additional +Z lift required to keep all values on/above `floor_z`.
    """
    z = np.asarray(z_values, dtype=np.float64).reshape(-1)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0.0

    min_z = float(np.min(z))
    if (not np.isfinite(min_z)):
        return 0.0

    needed = float(floor_z) - min_z
    if needed <= float(eps):
        return 0.0
    return float(needed)
