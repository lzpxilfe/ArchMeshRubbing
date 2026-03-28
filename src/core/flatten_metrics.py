"""Flatten metric utilities.

Centralized distortion and quality helpers for flatten results.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .mesh_loader import MeshData


def compute_face_distortion(mesh: MeshData, uv: np.ndarray) -> np.ndarray:
    """Compute per-face distortion [0, 1] on unfolded triangles.

    Distortion is a blend of area distortion and average stretch distortion.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    uv_arr = np.asarray(uv, dtype=np.float64)

    if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] < 3:
        return np.zeros((0,), dtype=np.float64)

    f = faces[:, :3].astype(np.int32, copy=False)
    n_faces = int(f.shape[0])
    distortions = np.ones((n_faces,), dtype=np.float64)

    try:
        v0 = vertices[f[:, 0]]
        v1 = vertices[f[:, 1]]
        v2 = vertices[f[:, 2]]
        uv2 = uv_arr[:, :2]
        t0 = uv2[f[:, 0]]
        t1 = uv2[f[:, 1]]
        t2 = uv2[f[:, 2]]
    except Exception:
        return distortions

    e1_3d = v1 - v0
    e2_3d = v2 - v0
    area_3d = 0.5 * np.linalg.norm(np.cross(e1_3d, e2_3d), axis=1)

    e1_2d = t1 - t0
    e2_2d = t2 - t0
    cross2 = e1_2d[:, 0] * e2_2d[:, 1] - e1_2d[:, 1] * e2_2d[:, 0]
    area_2d = 0.5 * np.abs(cross2)

    valid_area = np.isfinite(area_3d) & np.isfinite(area_2d) & (area_3d > 1e-10) & (area_2d > 1e-10)
    if not np.any(valid_area):
        return distortions

    a3 = area_3d[valid_area]
    a2 = area_2d[valid_area]
    area_ratio = np.minimum(a2 / a3, a3 / a2)
    area_distortion = 1.0 - area_ratio

    len_e1_3d = np.linalg.norm(e1_3d, axis=1)
    len_e2_3d = np.linalg.norm(e2_3d, axis=1)
    len_e1_2d = np.linalg.norm(e1_2d, axis=1)
    len_e2_2d = np.linalg.norm(e2_2d, axis=1)

    stretch_distortion = np.ones((n_faces,), dtype=np.float64)
    valid_len = (
        (len_e1_3d > 1e-10)
        & (len_e2_3d > 1e-10)
        & (len_e1_2d > 1e-10)
        & (len_e2_2d > 1e-10)
        & np.isfinite(len_e1_3d)
        & np.isfinite(len_e2_3d)
        & np.isfinite(len_e1_2d)
        & np.isfinite(len_e2_2d)
    )
    if np.any(valid_len):
        r1 = len_e1_2d[valid_len] / len_e1_3d[valid_len]
        r2 = len_e2_2d[valid_len] / len_e2_3d[valid_len]
        ratio1 = np.minimum(r1, 1.0 / r1)
        ratio2 = np.minimum(r2, 1.0 / r2)
        stretch_distortion[valid_len] = 1.0 - 0.5 * (ratio1 + ratio2)

    distortions[valid_area] = 0.5 * (area_distortion + stretch_distortion[valid_area])
    return np.clip(distortions, 0.0, 1.0)


def distortion_summary(distortion: np.ndarray | None) -> dict[str, Any]:
    """Summarize face-wise distortion values.

    Returns keys: count, mean, median, max, p95.
    """
    arr = np.asarray(distortion if distortion is not None else np.array([], dtype=np.float64), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {
            "count": 0,
            "mean": 1.0,
            "median": 1.0,
            "max": 1.0,
            "p95": 1.0,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.quantile(arr, 0.95)),
    }
