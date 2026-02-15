"""
Geometry helpers for floor/plane alignment workflows.
"""

from __future__ import annotations

import numpy as np


def _as_vec3(value: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError("Expected at least 3 values for a 3D vector.")
    return arr[:3]


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
