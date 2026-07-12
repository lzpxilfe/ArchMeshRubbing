"""Pure coordinate rebasing helpers for precision-safe viewport rendering.

Scientific geometry remains in absolute float64 world millimetres.  These
helpers create transient render coordinates before data crosses a float32 GPU
boundary; none of the returned origins or matrices are durable authority.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]
_AFFINE_ATOL = 1e-12


def _finite_vec3(value: ArrayLike, *, field_name: str) -> Float64Array:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite 3-vector") from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{field_name} must be a finite 3-vector")
    return vector.copy()


def _finite_points(value: ArrayLike, *, field_name: str) -> Float64Array:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must have shape (..., 3) and be finite"
        ) from exc
    if points.ndim == 0 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{field_name} must have shape (..., 3) and be finite")
    return points


def _finite_bounds(value: ArrayLike) -> Float64Array:
    try:
        bounds = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("bounds must be a finite [[min], [max]] array") from exc
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("bounds must be a finite [[min], [max]] array")
    if np.any(bounds[0] > bounds[1]):
        raise ValueError("bounds minimum must not exceed bounds maximum")
    return bounds


def _finite_affine(value: ArrayLike, *, field_name: str) -> Float64Array:
    try:
        matrix = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite affine 4x4 matrix") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{field_name} must be a finite affine 4x4 matrix")
    if not np.allclose(
        matrix[3],
        [0.0, 0.0, 0.0, 1.0],
        rtol=0.0,
        atol=_AFFINE_ATOL,
    ):
        raise ValueError(f"{field_name} must be an affine 4x4 matrix")
    return matrix


def render_origin_from_bounds(bounds: ArrayLike) -> Float64Array:
    """Return the finite float64 midpoint of canonical world bounds.

    Halving before addition avoids overflow for large, same-sign finite bounds.
    """

    validated = _finite_bounds(bounds)
    midpoint = validated[0] * 0.5 + validated[1] * 0.5
    if not np.isfinite(midpoint).all():
        raise ValueError("bounds midpoint is not finite")
    return midpoint


def world_to_render_points(
    points_world: ArrayLike,
    render_origin_world: ArrayLike,
) -> Float64Array:
    """Translate absolute world points into one transient render frame."""

    points = _finite_points(points_world, field_name="points_world")
    origin = _finite_vec3(render_origin_world, field_name="render_origin_world")
    with np.errstate(over="ignore", invalid="ignore"):
        relative = points - origin
    if not np.isfinite(relative).all():
        raise ValueError("world-to-render translation produced non-finite coordinates")
    return np.asarray(relative, dtype=np.float64)


def render_to_world_points(
    points_render: ArrayLike,
    render_origin_world: ArrayLike,
) -> Float64Array:
    """Restore transient render points to absolute float64 world coordinates."""

    points = _finite_points(points_render, field_name="points_render")
    origin = _finite_vec3(render_origin_world, field_name="render_origin_world")
    with np.errstate(over="ignore", invalid="ignore"):
        absolute = points + origin
    if not np.isfinite(absolute).all():
        raise ValueError("render-to-world translation produced non-finite coordinates")
    return np.asarray(absolute, dtype=np.float64)


def encode_relative_float32(
    vertices_local: ArrayLike,
    vbo_origin_local: ArrayLike,
) -> Float32Array:
    """Encode ``float32(vertices_local - vbo_origin_local)`` safely.

    Subtraction is deliberately performed in float64 so a large absolute world
    offset is removed before the precision-losing GPU conversion.
    """

    relative = world_to_render_points(vertices_local, vbo_origin_local)
    with np.errstate(over="ignore", invalid="ignore"):
        encoded = relative.astype(np.float32)
    if not np.isfinite(encoded).all():
        raise ValueError("relative coordinates cannot be represented as finite float32")
    return encoded


def rebase_affine_for_render(
    local_to_world: ArrayLike,
    vbo_origin_local: ArrayLike,
    render_origin_world: ArrayLike,
) -> Float64Array:
    """Map VBO-relative positions into a frame-relative render coordinate system.

    For ``q = float32(v - O)``, world affine ``M=(L,t)``, and frame origin
    ``R``, the returned affine has ``linear=L`` and
    ``translation=L@O + t - R``.  Thus it represents
    ``T(-R) @ M @ T(O)`` without constructing cancellation-prone translations.
    """

    matrix = _finite_affine(local_to_world, field_name="local_to_world")
    vbo_origin = _finite_vec3(vbo_origin_local, field_name="vbo_origin_local")
    render_origin = _finite_vec3(
        render_origin_world,
        field_name="render_origin_world",
    )
    rebased = matrix.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        rebased[:3, 3] = matrix[:3, :3] @ vbo_origin + matrix[:3, 3] - render_origin
    if not np.isfinite(rebased).all():
        raise ValueError("affine rebasing produced a non-finite matrix")
    return rebased


def rebase_world_plane_for_render(
    normal_world: ArrayLike,
    offset_world: float,
    render_origin_world: ArrayLike,
) -> tuple[Float64Array, float]:
    """Translate a world plane ``n.p + d = 0`` into render coordinates.

    With ``p = r + R``, the render-space equation is
    ``n.r + (d + n.R) = 0``.
    """

    normal = _finite_vec3(normal_world, field_name="normal_world")
    if not np.any(normal != 0.0):
        raise ValueError("normal_world must not be the zero vector")
    try:
        offset = float(offset_world)
    except (TypeError, ValueError) as exc:
        raise ValueError("offset_world must be finite") from exc
    if not np.isfinite(offset):
        raise ValueError("offset_world must be finite")
    render_origin = _finite_vec3(
        render_origin_world,
        field_name="render_origin_world",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        render_offset = offset + float(np.dot(normal, render_origin))
    if not np.isfinite(render_offset):
        raise ValueError("plane rebasing produced a non-finite offset")
    return normal, render_offset


def absolute_modelview_from_render(
    render_modelview: ArrayLike,
    render_origin_world: ArrayLike,
) -> Float64Array:
    """Return a modelview that accepts absolute-world points.

    A render modelview consumes ``r = p - R``.  The equivalent absolute-world
    matrix has the same linear part and translation ``t - L@R``.
    """

    matrix = _finite_affine(render_modelview, field_name="render_modelview")
    render_origin = _finite_vec3(
        render_origin_world,
        field_name="render_origin_world",
    )
    absolute = matrix.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        absolute[:3, 3] = matrix[:3, 3] - matrix[:3, :3] @ render_origin
    if not np.isfinite(absolute).all():
        raise ValueError("absolute modelview conversion produced a non-finite matrix")
    return absolute


__all__ = [
    "absolute_modelview_from_render",
    "encode_relative_float32",
    "rebase_affine_for_render",
    "rebase_world_plane_for_render",
    "render_origin_from_bounds",
    "render_to_world_points",
    "world_to_render_points",
]
