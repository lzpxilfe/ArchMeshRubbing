"""Smoothing a drawn line without moving what it was measured from.

An outline snapped to a half-millimetre grid, or a section cut through a
scanner's triangles, follows every step of the grid and every facet of the
mesh: on the page the line jitters.  A drafter's pen does not.  Smoothing
here is a presentation choice made on the sheet, declared on the sheet, and
applied to a copy of the record's points at draw time; the record keeps
its measured coordinates and its hash.

The operation is the plain one: resample the line at even steps along its
length, average each sample with its neighbours under a Gaussian of the
stated width, then drop the samples a straight segment already covers.  A
closed ring is averaged around its whole circumference; an open line is
mirrored through each end so the ends stay exactly where they were.  A
ring shorter than a few widths is left alone - averaging would shrink it
towards its centre, which is not smoothing but erasing.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


#: Widths past this stop being a pen's smoothness and start redrawing the
#: form; the composer refuses them.
MAX_LINE_SMOOTHING_MM = 3.0
#: A closed ring shorter than this many widths is not smoothed at all.
MIN_RING_LENGTH_IN_SIGMAS = 8.0
#: Tolerance for dropping resampled points that lie on a straight segment.
SIMPLIFY_TOLERANCE_MM = 0.02


def _resample_step_mm(sigma_mm: float) -> float:
    return min(0.25, max(0.02, sigma_mm / 5.0))


def _cumulative_length(points: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _resample(points: np.ndarray, *, closed: bool, step_mm: float) -> np.ndarray:
    """Even samples along the polyline; a closed ring gets no repeated end."""

    loop = np.vstack([points, points[:1]]) if closed else points
    lengths = _cumulative_length(loop)
    total = float(lengths[-1])
    if total <= 0.0:
        return points.copy()
    count = max(int(math.ceil(total / step_mm)), 4 if closed else 2)
    if closed:
        targets = np.arange(count, dtype=np.float64) * (total / count)
    else:
        targets = np.linspace(0.0, total, count + 1)
    xs = np.interp(targets, lengths, loop[:, 0])
    ys = np.interp(targets, lengths, loop[:, 1])
    return np.column_stack([xs, ys])


def _gaussian_kernel(sigma_samples: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma_samples)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    return kernel / kernel.sum()


def _blur(samples: np.ndarray, *, closed: bool, sigma_samples: float) -> np.ndarray:
    kernel = _gaussian_kernel(sigma_samples)
    radius = kernel.size // 2
    if closed:
        padded = np.concatenate([samples[-radius:], samples, samples[:radius]])
    else:
        # Mirror the line through each end point: the average at the end is
        # the end itself, and a straight run stays straight.
        head = 2.0 * samples[:1] - samples[1 : radius + 1][::-1]
        tail = 2.0 * samples[-1:] - samples[-radius - 1 : -1][::-1]
        padded = np.concatenate([head, samples, tail])
    out = np.column_stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)]
    )
    if not closed:
        out[0] = samples[0]
        out[-1] = samples[-1]
    return out


def _simplify(points: np.ndarray, *, tolerance_mm: float) -> np.ndarray:
    """Douglas-Peucker on an open polyline, iteratively."""

    count = points.shape[0]
    if count <= 2:
        return points
    keep = np.zeros(count, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, count - 1)]
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
        segment = points[end] - points[start]
        length = float(np.hypot(segment[0], segment[1]))
        inner = points[start + 1 : end]
        if length <= 1e-12:
            distances = np.linalg.norm(inner - points[start], axis=1)
        else:
            distances = np.abs(
                (inner[:, 0] - points[start, 0]) * segment[1]
                - (inner[:, 1] - points[start, 1]) * segment[0]
            ) / length
        farthest = int(np.argmax(distances))
        if float(distances[farthest]) > tolerance_mm:
            split = start + 1 + farthest
            keep[split] = True
            stack.append((start, split))
            stack.append((split, end))
    return points[keep]


def smooth_polyline(
    points_mm: Sequence[Sequence[float]],
    *,
    closed: bool,
    sigma_mm: float,
    tolerance_mm: float = SIMPLIFY_TOLERANCE_MM,
) -> tuple[tuple[float, float], ...]:
    """Return the line smoothed with a Gaussian of ``sigma_mm`` along its length.

    The result is a new polyline; the input is not touched.  Zero or a
    negative width returns the input points unchanged.  A closed ring shorter
    than ``MIN_RING_LENGTH_IN_SIGMAS`` widths is returned unchanged too.
    """

    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_mm must be an N x 2 array")
    if not np.isfinite(points).all():
        raise ValueError("points_mm must be finite")
    original = tuple((float(x), float(y)) for x, y in points)
    if sigma_mm <= 0.0 or points.shape[0] < 3:
        return original
    loop = np.vstack([points, points[:1]]) if closed else points
    total = float(_cumulative_length(loop)[-1])
    if total <= 0.0:
        return original
    if closed and total < MIN_RING_LENGTH_IN_SIGMAS * sigma_mm:
        return original
    step = _resample_step_mm(sigma_mm)
    samples = _resample(points, closed=closed, step_mm=step)
    blurred = _blur(samples, closed=closed, sigma_samples=sigma_mm / step)
    if closed:
        # Simplify as an open line from the first sample round to itself,
        # then drop the repeated end.
        ring = np.vstack([blurred, blurred[:1]])
        simplified = _simplify(ring, tolerance_mm=tolerance_mm)[:-1]
    else:
        simplified = _simplify(blurred, tolerance_mm=tolerance_mm)
    return tuple((float(x), float(y)) for x, y in simplified)


__all__ = [
    "MAX_LINE_SMOOTHING_MM",
    "MIN_RING_LENGTH_IN_SIGMAS",
    "SIMPLIFY_TOLERANCE_MM",
    "smooth_polyline",
]
