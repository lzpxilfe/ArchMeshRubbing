"""Cylindrical flattening models."""

from __future__ import annotations

from typing import Any

import numpy as np

from .flatten_utils import (
    _angles_to_min_range,
    _axis_unit_vector,
    _coerce_explicit_axis_vector,
    _cylinder_axis_score,
    _normalize_cylinder_axis_choice,
    _pca_axes_3d,
    _robust_circle_fit_2d,
    _seam_hint_from_cut_lines,
)
from .mesh_loader import MeshData


def cylindrical_parameterization(
    mesh: MeshData,
    *,
    axis: str | Any = "auto",
    radius: float | None = None,
    cut_lines_world: list[list[list[float]]] | None = None,
    seam_hint: float | None = None,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Simple cylindrical unwrapping (developable approximation)."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] == 0 or vertices.shape[1] < 3:
        return np.zeros((0, 2), dtype=np.float64)

    axis_choice = _normalize_cylinder_axis_choice(axis)
    axis_source = axis_choice
    axis_score = None
    if axis_choice == "auto":
        candidates: list[tuple[str, np.ndarray]] = []
        try:
            pca = _pca_axes_3d(vertices)
            for i in range(3):
                candidates.append((f"pca{i}", pca[:, i].copy()))
        except Exception:
            candidates = []
        for ax in ("x", "y", "z"):
            candidates.append((ax, _axis_unit_vector(ax)))

        best_lbl = None
        best_vec = None
        best = float("inf")
        for lbl, vec in candidates:
            score = _cylinder_axis_score(vertices, vec)
            if np.isfinite(score) and score < best:
                best = float(score)
                best_lbl = str(lbl)
                best_vec = np.asarray(vec, dtype=np.float64).reshape(3)
        if best_vec is None:
            try:
                pca = _pca_axes_3d(vertices)
                best_lbl = "pca0"
                best_vec = np.asarray(pca[:, 0], dtype=np.float64).reshape(3)
                best = _cylinder_axis_score(vertices, best_vec)
            except Exception:
                best_lbl = "z"
                best_vec = _axis_unit_vector("z")
                best = _cylinder_axis_score(vertices, best_vec)
        axis_source = str(best_lbl or "auto")
        axis_score = None if not np.isfinite(best) else float(best)
        a = np.asarray(best_vec, dtype=np.float64).reshape(3)
    else:
        explicit = _coerce_explicit_axis_vector(axis)
        a = explicit if explicit is not None else _axis_unit_vector(axis)

    try:
        a = np.asarray(a, dtype=np.float64).reshape(3)
        n_a = float(np.linalg.norm(a))
        if not np.isfinite(n_a) or n_a < 1e-12:
            a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            a = a / n_a
    except Exception:
        a = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    temp = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(a[0])) < 0.9
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    b1 = np.cross(a, temp)
    b1_norm = float(np.linalg.norm(b1))
    if not np.isfinite(b1_norm) or b1_norm < 1e-12:
        b1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        b1_norm = float(np.linalg.norm(b1))
    b1 /= b1_norm
    b2 = np.cross(a, b1)

    t = vertices @ a
    x0 = vertices @ b1
    y0 = vertices @ b2

    cx = None
    cy = None
    try:
        n = int(x0.size)
        max_fit = int(getattr(mesh, "_cylinder_fit_max_points", 50000) or 50000)
        max_fit = max(2000, int(max_fit))
        if n > max_fit and n > 0:
            idx = np.linspace(0, n - 1, num=max_fit, dtype=np.int64)
            fit = _robust_circle_fit_2d(x0[idx], y0[idx])
        else:
            fit = _robust_circle_fit_2d(x0, y0)
        if fit is not None:
            c2, _r2 = fit
            cx = float(c2[0])
            cy = float(c2[1])
    except Exception:
        cx = None
        cy = None

    if cx is None or cy is None or not np.isfinite(cx) or not np.isfinite(cy):
        try:
            cx = float(np.nanmean(x0))
            cy = float(np.nanmean(y0))
        except Exception:
            cx = 0.0
            cy = 0.0

    center = (float(cx) * b1) + (float(cy) * b2)
    x = x0 - float(cx)
    y = y0 - float(cy)
    theta = np.arctan2(y, x)

    seam_from_lines = _seam_hint_from_cut_lines(
        cut_lines_world,
        axis=a,
        b1=b1,
        b2=b2,
        center=center,
    )
    seam_pref = seam_hint if seam_hint is not None else seam_from_lines
    theta_wrapped, seam_angle, span = _angles_to_min_range(theta, seam_hint=seam_pref)

    r = np.hypot(x, y)
    if radius is None:
        r_f = r[np.isfinite(r)]
        radius_val = float(np.median(r_f)) if r_f.size else 1.0
    else:
        try:
            radius_val = float(radius)
        except Exception:
            radius_val = 0.0
        if not np.isfinite(radius_val) or abs(radius_val) < 1e-12:
            r_f = r[np.isfinite(r)]
            radius_val = float(np.median(r_f)) if r_f.size else 1.0

    radius_val = abs(float(radius_val))
    if radius_val < 1e-6:
        radius_val = 1.0

    u = theta_wrapped * radius_val
    v = t.astype(np.float64, copy=False)

    uv = np.stack([u, v], axis=1)
    if not np.isfinite(uv).all():
        uv = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
    if bool(return_meta):
        meta: dict[str, Any] = {
            "cylinder_axis_choice": str(axis_choice),
            "cylinder_axis_source": str(axis_source),
            "cylinder_axis_score": None if axis_score is None else float(axis_score),
            "cylinder_axis": np.asarray(a, dtype=np.float64).reshape(3),
            "cylinder_center": np.asarray(center, dtype=np.float64).reshape(3),
            "cylinder_radius": float(radius_val),
            "cylinder_seam_angle": float(seam_angle),
            "cylinder_span": float(span),
        }
        return uv, meta
    return uv
