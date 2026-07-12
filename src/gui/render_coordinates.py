"""Pure coordinate rebasing helpers for precision-safe viewport rendering.

Scientific geometry remains in absolute float64 world millimetres.  These
helpers create transient render coordinates before data crosses a float32 GPU
boundary; none of the returned origins or matrices are durable authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray


Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]
_AFFINE_ATOL = 1e-12
_HOMOGENEOUS_EPS = 1e-15


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


def _finite_matrix4(value: ArrayLike, *, field_name: str) -> Float64Array:
    try:
        matrix = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite 4x4 matrix") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{field_name} must be a finite 4x4 matrix")
    return matrix


def _integer(value: object, *, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{field_name} must be an integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class RenderFrameSnapshot:
    """One immutable render-relative camera/depth coordinate contract."""

    frame_serial: int
    projection_generation: int
    viewport: tuple[int, int, int, int]
    modelview_render: Float64Array = field(repr=False, compare=False)
    projection: Float64Array = field(repr=False, compare=False)
    render_origin_world_mm: Float64Array = field(repr=False, compare=False)
    view_projection: Float64Array = field(init=False, repr=False, compare=False)
    inverse_view_projection: Float64Array = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        frame_serial = _integer(self.frame_serial, field_name="frame_serial")
        generation = _integer(
            self.projection_generation,
            field_name="projection_generation",
        )
        if frame_serial < 0 or generation < 0:
            raise ValueError("render frame counters must be non-negative integers")

        try:
            raw_viewport = tuple(self.viewport)
        except TypeError as exc:
            raise ValueError("viewport must contain four integers") from exc
        if len(raw_viewport) != 4:
            raise ValueError("viewport must contain x, y, positive width, positive height")
        viewport = tuple(
            _integer(value, field_name="viewport value")
            for value in raw_viewport
        )
        if viewport[2] <= 0 or viewport[3] <= 0:
            raise ValueError("viewport must contain x, y, positive width, positive height")

        modelview = _finite_affine(
            self.modelview_render,
            field_name="modelview_render",
        )
        projection = _finite_matrix4(self.projection, field_name="projection")
        origin = _finite_vec3(
            self.render_origin_world_mm,
            field_name="render_origin_world_mm",
        )
        with np.errstate(over="ignore", invalid="ignore"):
            view_projection = projection @ modelview
        if not np.isfinite(view_projection).all():
            raise ValueError("render view-projection matrix is not finite")
        try:
            inverse = np.linalg.inv(view_projection)
        except np.linalg.LinAlgError as exc:
            raise ValueError("render view-projection matrix must be invertible") from exc
        if not np.isfinite(inverse).all():
            raise ValueError("inverse render view-projection matrix is not finite")

        for array in (modelview, projection, origin, view_projection, inverse):
            array.setflags(write=False)
        object.__setattr__(self, "frame_serial", frame_serial)
        object.__setattr__(self, "projection_generation", generation)
        object.__setattr__(self, "viewport", viewport)
        object.__setattr__(self, "modelview_render", modelview)
        object.__setattr__(self, "projection", projection)
        object.__setattr__(self, "render_origin_world_mm", origin)
        object.__setattr__(self, "view_projection", view_projection)
        object.__setattr__(self, "inverse_view_projection", inverse)


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


def project_world_to_window(
    frame: RenderFrameSnapshot,
    point_world: ArrayLike,
) -> Float64Array:
    """Project one absolute world point through an immutable render frame."""

    if not isinstance(frame, RenderFrameSnapshot):
        raise TypeError("frame must be a RenderFrameSnapshot")
    point = _finite_vec3(point_world, field_name="point_world")
    point_render = world_to_render_points(
        point,
        frame.render_origin_world_mm,
    ).reshape(3)
    homogeneous = np.array(
        [point_render[0], point_render[1], point_render[2], 1.0],
        dtype=np.float64,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        clip = frame.view_projection @ homogeneous
    w = float(clip[3])
    if not np.isfinite(clip).all() or abs(w) <= _HOMOGENEOUS_EPS:
        raise ValueError("world projection produced an invalid homogeneous point")
    ndc = clip[:3] / w
    if not np.isfinite(ndc).all():
        raise ValueError("world projection produced non-finite device coordinates")
    vx, vy, width, height = frame.viewport
    window = np.array(
        [
            float(vx) + (float(ndc[0]) + 1.0) * float(width) * 0.5,
            float(vy) + (float(ndc[1]) + 1.0) * float(height) * 0.5,
            (float(ndc[2]) + 1.0) * 0.5,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(window).all():
        raise ValueError("world projection produced a non-finite window point")
    return window


def unproject_window_to_world(
    frame: RenderFrameSnapshot,
    point_window: ArrayLike,
) -> Float64Array:
    """Unproject one window/depth sample to absolute float64 world millimetres."""

    if not isinstance(frame, RenderFrameSnapshot):
        raise TypeError("frame must be a RenderFrameSnapshot")
    window = _finite_vec3(point_window, field_name="point_window")
    depth = float(window[2])
    if depth < 0.0 or depth > 1.0:
        raise ValueError("window depth must be within [0, 1]")
    vx, vy, width, height = frame.viewport
    ndc = np.array(
        [
            ((float(window[0]) - float(vx)) / float(width)) * 2.0 - 1.0,
            ((float(window[1]) - float(vy)) / float(height)) * 2.0 - 1.0,
            depth * 2.0 - 1.0,
            1.0,
        ],
        dtype=np.float64,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        render_h = frame.inverse_view_projection @ ndc
    w = float(render_h[3])
    if not np.isfinite(render_h).all() or abs(w) <= _HOMOGENEOUS_EPS:
        raise ValueError("window unprojection produced an invalid homogeneous point")
    point_render = render_h[:3] / w
    if not np.isfinite(point_render).all():
        raise ValueError("window unprojection produced a non-finite render point")
    return render_to_world_points(
        point_render,
        frame.render_origin_world_mm,
    ).reshape(3)


def world_ray_from_window(
    frame: RenderFrameSnapshot,
    window_x: float,
    window_y: float,
) -> tuple[Float64Array, Float64Array]:
    """Return one finite absolute-world ray from an immutable render frame."""

    try:
        x = float(window_x)
        y = float(window_y)
    except (TypeError, ValueError) as exc:
        raise ValueError("window coordinates must be finite") from exc
    if not np.isfinite([x, y]).all():
        raise ValueError("window coordinates must be finite")
    near = unproject_window_to_world(frame, [x, y, 0.0])
    far = unproject_window_to_world(frame, [x, y, 1.0])
    direction = far - near
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= _HOMOGENEOUS_EPS:
        raise ValueError("window ray direction is invalid")
    direction = direction / norm
    if not np.isfinite(direction).all():
        raise ValueError("window ray direction is not finite")
    return near, np.asarray(direction, dtype=np.float64)


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
    "RenderFrameSnapshot",
    "absolute_modelview_from_render",
    "encode_relative_float32",
    "project_world_to_window",
    "rebase_affine_for_render",
    "rebase_world_plane_for_render",
    "render_origin_from_bounds",
    "render_to_world_points",
    "unproject_window_to_world",
    "world_to_render_points",
    "world_ray_from_window",
]
