"""Flatten metric utilities.

Centralized distortion and quality helpers for flatten results.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .mesh_loader import MeshData


def compute_face_distortion(mesh: MeshData, uv: np.ndarray) -> np.ndarray:
    """Compute per-face distortion [0, 1] on unfolded triangles.

    The score is the worst normalized error among all three edge lengths,
    triangle area, and the two singular values of the local 3D-to-2D
    Jacobian.  Looking at only the two edges incident to vertex zero misses a
    large class of shear errors where the opposite edge changes dramatically.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    uv_arr = np.asarray(uv, dtype=np.float64)

    if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] < 3:
        return np.zeros((0,), dtype=np.float64)

    f = faces[:, :3].astype(np.int32, copy=False)
    n_faces = int(f.shape[0])
    distortions = np.ones((n_faces,), dtype=np.float64)

    if uv_arr.ndim != 2 or uv_arr.shape[0] != vertices.shape[0] or uv_arr.shape[1] < 2:
        return distortions
    epsilon = 1e-10
    chunk_faces = 200_000
    for start in range(0, n_faces, chunk_faces):
        end = min(n_faces, start + chunk_faces)
        face_chunk = f[start:end]
        try:
            v0 = vertices[face_chunk[:, 0]]
            v1 = vertices[face_chunk[:, 1]]
            v2 = vertices[face_chunk[:, 2]]
            t0 = uv_arr[face_chunk[:, 0], :2]
            t1 = uv_arr[face_chunk[:, 1], :2]
            t2 = uv_arr[face_chunk[:, 2], :2]
        except Exception:
            continue

        source_e1 = v1 - v0
        source_e2 = v2 - v0
        source_e3 = v2 - v1
        target_e1 = t1 - t0
        target_e2 = t2 - t0
        target_e3 = t2 - t1
        source_lengths = np.stack(
            (
                np.linalg.norm(source_e1, axis=1),
                np.linalg.norm(source_e2, axis=1),
                np.linalg.norm(source_e3, axis=1),
            ),
            axis=1,
        )
        target_lengths = np.stack(
            (
                np.linalg.norm(target_e1, axis=1),
                np.linalg.norm(target_e2, axis=1),
                np.linalg.norm(target_e3, axis=1),
            ),
            axis=1,
        )
        source_twice_area = np.linalg.norm(
            np.cross(source_e1, source_e2), axis=1
        )
        target_signed_twice_area = (
            target_e1[:, 0] * target_e2[:, 1]
            - target_e1[:, 1] * target_e2[:, 0]
        )
        target_twice_area = np.abs(target_signed_twice_area)
        valid = (
            np.isfinite(source_lengths).all(axis=1)
            & np.isfinite(target_lengths).all(axis=1)
            & np.isfinite(source_twice_area)
            & np.isfinite(target_twice_area)
            & (np.min(source_lengths, axis=1) > epsilon)
            & (np.min(target_lengths, axis=1) > epsilon)
            & (source_twice_area > epsilon)
            & (target_twice_area > epsilon)
        )
        if not np.any(valid):
            continue

        source_valid = source_lengths[valid]
        target_valid = target_lengths[valid]
        stretch = target_valid / source_valid
        edge_similarity = np.minimum(stretch, 1.0 / stretch)
        edge_distortion = 1.0 - np.min(edge_similarity, axis=1)

        area_stretch = target_twice_area[valid] / source_twice_area[valid]
        area_distortion = 1.0 - np.minimum(area_stretch, 1.0 / area_stretch)

        e1 = source_e1[valid]
        e2 = source_e2[valid]
        du1 = target_e1[valid]
        du2 = target_e2[valid]
        basis_x = source_valid[:, 0]
        e1_unit = e1 / basis_x[:, None]
        second_x = np.sum(e2 * e1_unit, axis=1)
        second_y_sq = np.maximum(
            np.sum(e2 * e2, axis=1) - second_x * second_x,
            0.0,
        )
        second_y = np.sqrt(second_y_sq)
        jacobian_valid = second_y > epsilon
        jacobian_distortion = np.ones_like(edge_distortion)
        if np.any(jacobian_valid):
            a = basis_x[jacobian_valid]
            b = second_x[jacobian_valid]
            c = second_y[jacobian_valid]
            u1 = du1[jacobian_valid]
            u2 = du2[jacobian_valid]
            j00 = u1[:, 0] / a
            j10 = u1[:, 1] / a
            j01 = (u2[:, 0] - j00 * b) / c
            j11 = (u2[:, 1] - j10 * b) / c
            trace = j00 * j00 + j01 * j01 + j10 * j10 + j11 * j11
            determinant = j00 * j11 - j01 * j10
            discriminant = np.sqrt(
                np.maximum(trace * trace - 4.0 * determinant * determinant, 0.0)
            )
            sigma_max = np.sqrt(np.maximum(0.5 * (trace + discriminant), 0.0))
            sigma_min = np.sqrt(np.maximum(0.5 * (trace - discriminant), 0.0))
            singular_valid = (
                np.isfinite(sigma_max)
                & np.isfinite(sigma_min)
                & (sigma_max > epsilon)
                & (sigma_min > epsilon)
            )
            local = np.ones_like(sigma_max)
            if np.any(singular_valid):
                max_error = 1.0 - np.minimum(
                    sigma_max[singular_valid],
                    1.0 / sigma_max[singular_valid],
                )
                min_error = 1.0 - np.minimum(
                    sigma_min[singular_valid],
                    1.0 / sigma_min[singular_valid],
                )
                local[singular_valid] = np.maximum(max_error, min_error)
            jacobian_distortion[jacobian_valid] = local

        scores = np.maximum.reduce(
            (edge_distortion, area_distortion, jacobian_distortion)
        )
        valid_indices = np.flatnonzero(valid) + start
        distortions[valid_indices] = scores
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
