"""
Helpers for fitting circular section profiles in tile workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(slots=True)
class CircleFit2DResult:
    center_xy: tuple[float, float] | None = None
    radius: float | None = None
    radius_iqr: float = 0.0
    rmse: float = 0.0
    arc_span_deg: float = 0.0
    used_points: int = 0
    confidence: float = 0.0

    def is_defined(self) -> bool:
        return self.center_xy is not None and self.radius is not None


def _fit_circle_linear(points_xy: np.ndarray) -> tuple[float, float, float] | None:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = (x * x) + (y * y)
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except Exception:
        return None
    cx = float(sol[0])
    cy = float(sol[1])
    c0 = float(sol[2])
    radius_sq = c0 + (cx * cx) + (cy * cy)
    if not math.isfinite(radius_sq) or radius_sq <= 1e-12:
        return None
    radius = math.sqrt(radius_sq)
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(radius)):
        return None
    return (cx, cy, radius)


def _arc_span_deg(points_xy: np.ndarray, center_xy: tuple[float, float]) -> float:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return 0.0
    cx, cy = center_xy
    angles = np.mod(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx), 2.0 * np.pi)
    angles = np.sort(angles[np.isfinite(angles)])
    if angles.size < 2:
        return 0.0
    wrapped = np.concatenate([angles, angles[:1] + (2.0 * np.pi)])
    gaps = np.diff(wrapped)
    max_gap = float(np.max(gaps)) if gaps.size > 0 else (2.0 * np.pi)
    span = max(0.0, min(2.0 * np.pi, (2.0 * np.pi) - max_gap))
    return float(np.degrees(span))


def fit_circle_2d(points_xy: np.ndarray, *, min_points: int = 8) -> CircleFit2DResult:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.size <= 0:
        return CircleFit2DResult()
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        return CircleFit2DResult(used_points=int(pts.shape[0]))

    fit = _fit_circle_linear(pts)
    if fit is None:
        return CircleFit2DResult(used_points=int(pts.shape[0]))

    work = pts
    for _ in range(2):
        cx, cy, radius = fit
        radii = np.linalg.norm(work - np.array([cx, cy], dtype=np.float64), axis=1)
        radii = radii[np.isfinite(radii)]
        if radii.size < max(3, min_points // 2):
            break
        median_r = float(np.median(radii))
        residuals = np.abs(radii - median_r)
        mad = float(np.median(residuals)) if residuals.size > 0 else 0.0
        q85 = float(np.quantile(residuals, 0.85)) if residuals.size > 0 else 0.0
        thresh = max(mad * 3.0, q85, max(median_r, 1e-6) * 0.015)
        mask = residuals <= thresh
        if int(np.count_nonzero(mask)) < max(6, min_points):
            break
        candidate = _fit_circle_linear(work[mask])
        if candidate is None:
            break
        work = work[mask]
        fit = candidate

    cx, cy, _radius = fit
    radii = np.linalg.norm(work - np.array([cx, cy], dtype=np.float64), axis=1)
    radii = radii[np.isfinite(radii)]
    if radii.size < 3:
        return CircleFit2DResult(used_points=int(work.shape[0]))

    radius = float(np.median(radii))
    q25, q75 = np.quantile(radii, [0.25, 0.75])
    radius_iqr = float(max(0.0, q75 - q25))
    rmse = float(np.sqrt(np.mean((radii - radius) ** 2))) if radii.size > 0 else 0.0
    arc_span = _arc_span_deg(work, (cx, cy))

    rel_rmse = rmse / max(abs(radius), 1e-6)
    rel_iqr = radius_iqr / max(abs(radius), 1e-6)
    span_score = float(np.clip(arc_span / 180.0, 0.0, 1.0))
    rmse_score = float(np.clip(1.0 - (rel_rmse * 10.0), 0.0, 1.0))
    iqr_score = float(np.clip(1.0 - (rel_iqr * 8.0), 0.0, 1.0))
    count_score = float(np.clip((float(work.shape[0]) - 6.0) / 36.0, 0.0, 1.0))
    confidence = float(
        np.clip(
            (span_score * 0.45) + (rmse_score * 0.25) + (iqr_score * 0.15) + (count_score * 0.15),
            0.0,
            1.0,
        )
    )

    return CircleFit2DResult(
        center_xy=(float(cx), float(cy)),
        radius=radius,
        radius_iqr=radius_iqr,
        rmse=rmse,
        arc_span_deg=arc_span,
        used_points=int(work.shape[0]),
        confidence=confidence,
    )
