from __future__ import annotations

import numpy as np
import pytest

from src.gui.render_coordinates import (
    RenderFrameSnapshot,
    absolute_modelview_from_render,
    compute_clip_range,
    encode_relative_float32,
    perspective_depth_resolution_mm,
    project_world_to_window,
    rebase_affine_for_render,
    rebase_world_plane_for_render,
    render_origin_from_bounds,
    render_to_world_points,
    unproject_window_to_world,
    world_to_render_points,
    world_ray_from_window,
)


def _apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    return values @ matrix[:3, :3].T + matrix[:3, 3]


def test_large_world_offset_is_removed_before_float32_encoding() -> None:
    base = 1_000_000_000.0
    offsets = np.asarray([0.0, 0.125, 1.0, 3.0], dtype=np.float64)
    vertices = np.column_stack(
        [
            base + offsets,
            np.full(offsets.shape, base + 20.0),
            np.full(offsets.shape, base - 30.0),
        ]
    )
    original = vertices.copy()

    encoded = encode_relative_float32(
        vertices,
        [base, base + 20.0, base - 30.0],
    )

    assert encoded.dtype == np.float32
    np.testing.assert_array_equal(encoded[:, 0], offsets.astype(np.float32))
    np.testing.assert_array_equal(encoded[:, 1:], 0.0)
    np.testing.assert_array_equal(vertices, original)
    assert np.unique(vertices[:, 0].astype(np.float32)).size == 1


def test_bounds_origin_and_world_render_round_trip() -> None:
    bounds = np.asarray(
        [
            [1_000_000_000.0, -2_000_000_004.0, 9.0],
            [1_000_000_008.0, -1_999_999_996.0, 15.0],
        ],
        dtype=np.float64,
    )
    origin = render_origin_from_bounds(bounds)
    points = np.asarray(
        [
            [1_000_000_000.125, -2_000_000_000.0, 10.0],
            [1_000_000_003.0, -1_999_999_999.5, 14.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(origin, [1_000_000_004.0, -2_000_000_000.0, 12.0])
    relative = world_to_render_points(points, origin)
    restored = render_to_world_points(relative, origin)

    np.testing.assert_array_equal(restored, points)
    assert relative.dtype == np.float64


def test_rebased_affine_matches_world_transform_with_nonzero_pivot() -> None:
    linear = np.asarray(
        [
            [0.0, -1.25, 0.2],
            [0.75, 0.1, 0.0],
            [0.0, 0.3, 1.5],
        ],
        dtype=np.float64,
    )
    pivot = np.asarray([1_000_000_003.0, -2_000_000_005.0, 41.0])
    preview_translation = np.asarray([4.0, -7.0, 2.5])
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = linear
    # T(preview) @ T(pivot) @ L @ T(-pivot): the pivot is already reflected
    # in this arbitrary affine's translation before render rebasing begins.
    matrix[:3, 3] = preview_translation + pivot - linear @ pivot

    vbo_origin = np.asarray([1_000_000_000.0, -2_000_000_000.0, 40.0])
    render_origin = np.asarray([999_999_980.0, -1_999_999_990.0, 35.0])
    vertices_world = vbo_origin + np.asarray(
        [[0.0, 0.0, 0.0], [0.125, 1.0, 3.0], [-2.0, 4.0, -1.0]],
        dtype=np.float64,
    )
    q = encode_relative_float32(vertices_world, vbo_origin).astype(np.float64)

    render_matrix = rebase_affine_for_render(
        matrix,
        vbo_origin,
        render_origin,
    )

    np.testing.assert_array_equal(render_matrix[:3, :3], linear)
    np.testing.assert_allclose(
        _apply_affine(render_matrix, q),
        _apply_affine(matrix, vertices_world) - render_origin,
        rtol=0.0,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        render_matrix[:3, 3],
        linear @ vbo_origin + matrix[:3, 3] - render_origin,
        rtol=0.0,
        atol=0.0,
    )


def test_rebased_plane_preserves_world_plane_sign_and_zero_set() -> None:
    normal = np.asarray([2.0, -3.0, 0.5], dtype=np.float64)
    offset = -17.0
    render_origin = np.asarray([1_000_000_000.0, -2_000_000_000.0, 30.0])
    points_world = np.asarray(
        [
            [render_origin[0] + 1.0, render_origin[1], render_origin[2]],
            [render_origin[0] - 2.0, render_origin[1] + 4.0, render_origin[2] - 1.0],
            [8.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    render_normal, render_offset = rebase_world_plane_for_render(
        normal,
        offset,
        render_origin,
    )
    points_render = world_to_render_points(points_world, render_origin)
    world_values = points_world @ normal + offset
    render_values = points_render @ render_normal + render_offset

    np.testing.assert_allclose(render_values, world_values, rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(np.signbit(render_values), np.signbit(world_values))
    assert world_values[-1] == pytest.approx(0.0)
    assert render_values[-1] == pytest.approx(0.0, abs=1e-6)


def test_absolute_modelview_is_equivalent_for_world_points() -> None:
    render_modelview = np.asarray(
        [
            [0.0, 1.0, 0.0, -4.0],
            [-1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, -8.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    render_origin = np.asarray([1_000_000_000.0, -2_000_000_000.0, 30.0])
    points_world = render_origin + np.asarray(
        [[0.0, 0.0, 0.0], [0.125, 1.0, 3.0], [-2.0, 4.0, -1.0]],
        dtype=np.float64,
    )

    absolute_modelview = absolute_modelview_from_render(
        render_modelview,
        render_origin,
    )

    np.testing.assert_allclose(
        _apply_affine(absolute_modelview, points_world),
        _apply_affine(render_modelview, points_world - render_origin),
        rtol=0.0,
        atol=1e-7,
    )


def _perspective_matrix(
    *,
    fov_y_deg: float = 45.0,
    aspect: float = 1.25,
    near: float = 1.0,
    far: float = 100.0,
) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fov_y_deg) * 0.5)
    return np.asarray(
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


def _orthographic_matrix(
    *,
    left: float = -100.0,
    right: float = 100.0,
    bottom: float = -80.0,
    top: float = 80.0,
    near: float = 0.05,
    far: float = 2_100.0,
) -> np.ndarray:
    return np.asarray(
        [
            [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
            [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
            [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _render_frame(
    origin: np.ndarray,
    *,
    serial: int = 1,
    modelview: np.ndarray | None = None,
    projection: np.ndarray | None = None,
) -> RenderFrameSnapshot:
    return RenderFrameSnapshot(
        frame_serial=serial,
        projection_generation=7,
        viewport=(11, 17, 1000, 800),
        modelview_render=(
            np.eye(4, dtype=np.float64) if modelview is None else modelview
        ),
        projection=(
            _perspective_matrix() if projection is None else projection
        ),
        render_origin_world_mm=np.asarray(origin, dtype=np.float64),
    )


def test_render_frame_projection_is_invariant_to_large_world_offset() -> None:
    zero = _render_frame(np.zeros(3), serial=1)
    large_origin = np.asarray(
        [1_000_000_003.0, -1_000_000_007.0, 500_000_011.0],
        dtype=np.float64,
    )
    large = _render_frame(large_origin, serial=2)
    local_points = np.asarray(
        [[0.125, 1.0, -10.0], [1.0, -0.5, -25.0], [3.0, 0.25, -50.0]],
        dtype=np.float64,
    )

    for local in local_points:
        window_zero = project_world_to_window(zero, local)
        window_large = project_world_to_window(large, large_origin + local)
        np.testing.assert_allclose(window_large, window_zero, rtol=0.0, atol=1e-11)


def test_nonidentity_render_frame_round_trips_large_world_points() -> None:
    angle_x = np.radians(-11.0)
    angle_y = np.radians(23.0)
    rotate_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle_x), -np.sin(angle_x)],
            [0.0, np.sin(angle_x), np.cos(angle_x)],
        ],
        dtype=np.float64,
    )
    rotate_y = np.asarray(
        [
            [np.cos(angle_y), 0.0, np.sin(angle_y)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle_y), 0.0, np.cos(angle_y)],
        ],
        dtype=np.float64,
    )
    modelview = np.eye(4, dtype=np.float64)
    modelview[:3, :3] = rotate_x @ rotate_y
    modelview[:3, 3] = [2.5, -1.25, -30.0]
    origin = np.asarray(
        [1_000_000_003.0, -1_000_000_007.0, 500_000_011.0],
        dtype=np.float64,
    )
    frame = _render_frame(
        origin,
        modelview=modelview,
        projection=_perspective_matrix(near=0.1, far=1_000.0),
    )
    points_world = origin + np.asarray(
        [[0.125, 1.0, -10.0], [3.0, -0.5, -25.0], [-2.0, 4.0, -50.0]],
        dtype=np.float64,
    )

    for point_world in points_world:
        window = project_world_to_window(frame, point_world)
        restored = unproject_window_to_world(frame, window)
        np.testing.assert_allclose(restored, point_world, rtol=0.0, atol=2e-7)


def _quantize_normalized_depth_24(depth: float) -> float:
    levels = (1 << 24) - 1
    return float(np.rint(float(depth) * levels) / levels)


@pytest.mark.parametrize(
    ("projection", "point_render", "maximum_error_mm"),
    [
        (
            _perspective_matrix(near=0.05, far=2_100.0),
            np.asarray([3.0, -2.0, -250.0], dtype=np.float64),
            0.01,
        ),
        (
            _orthographic_matrix(near=0.05, far=2_100.0),
            np.asarray([1.0, 0.5, -1_000.0], dtype=np.float64),
            0.001,
        ),
    ],
    ids=("perspective", "canonical-orthographic"),
)
def test_24bit_depth_reconstruction_stays_within_explicit_clip_budget(
    projection: np.ndarray,
    point_render: np.ndarray,
    maximum_error_mm: float,
) -> None:
    """Check representative clip configurations, not a global depth guarantee."""

    origin = np.asarray(
        [1_000_000_003.0, -1_000_000_007.0, 500_000_011.0],
        dtype=np.float64,
    )
    frame = _render_frame(origin, projection=projection)
    point_world = origin + point_render
    window = project_world_to_window(frame, point_world)
    sampled_window = window.copy()
    sampled_window[2] = _quantize_normalized_depth_24(float(window[2]))

    reconstructed = unproject_window_to_world(frame, sampled_window)
    error_mm = float(np.linalg.norm(reconstructed - point_world))

    assert error_mm > 0.0
    assert error_mm < maximum_error_mm


def test_reference_perspective_float32_depth_stays_below_two_microns() -> None:
    origin = np.asarray(
        [1_000_000_003.0, -1_000_000_007.0, 500_000_011.0],
        dtype=np.float64,
    )
    frame = _render_frame(origin)
    point_world = origin + [0.125, 1.0, -10.0]
    window = project_world_to_window(frame, point_world)
    sampled_window = window.copy()
    sampled_window[2] = float(np.float32(sampled_window[2]))

    reconstructed = unproject_window_to_world(frame, sampled_window)

    np.testing.assert_allclose(reconstructed[:2], point_world[:2], rtol=0.0, atol=1e-6)
    assert float(np.linalg.norm(reconstructed - point_world)) < 0.002


def test_render_frame_ray_and_snapshot_origin_are_finite_and_immutable() -> None:
    origin = np.asarray(
        [1_000_000_003.0, -1_000_000_007.0, 500_000_011.0],
        dtype=np.float64,
    )
    frame = _render_frame(origin)
    window = project_world_to_window(frame, origin + [0.0, 0.0, -10.0])
    ray_origin, ray_direction = world_ray_from_window(frame, window[0], window[1])

    assert np.isfinite(ray_origin).all()
    np.testing.assert_allclose(ray_direction, [0.0, 0.0, -1.0], rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(frame.render_origin_world_mm, origin)
    assert not frame.render_origin_world_mm.flags.writeable
    assert not frame.modelview_render.flags.writeable
    assert not frame.projection.flags.writeable
    with pytest.raises(ValueError):
        frame.render_origin_world_mm[0] = 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_serial": -1},
        {"frame_serial": 1.0},
        {"projection_generation": True},
        {"viewport": (0, 0, 0, 100)},
        {"viewport": (0, 0, 100.5, 100)},
        {"projection": np.zeros((4, 4))},
        {"render_origin_world_mm": [0.0, np.nan, 0.0]},
    ],
)
def test_render_frame_rejects_invalid_contract(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "frame_serial": 1,
        "projection_generation": 0,
        "viewport": (0, 0, 100, 100),
        "modelview_render": np.eye(4),
        "projection": _perspective_matrix(),
        "render_origin_world_mm": np.zeros(3),
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        RenderFrameSnapshot(**values)  # type: ignore[arg-type]


def test_render_frame_projection_rejects_nonfinite_and_invalid_depth() -> None:
    frame = _render_frame(np.zeros(3))
    with pytest.raises(ValueError):
        project_world_to_window(frame, [np.nan, 0.0, -10.0])
    with pytest.raises(ValueError, match="depth"):
        unproject_window_to_world(frame, [10.0, 10.0, 1.1])
    with pytest.raises(ValueError):
        world_ray_from_window(frame, np.inf, 10.0)


@pytest.mark.parametrize(
    "bounds",
    [
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, np.nan, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
    ],
)
def test_render_origin_rejects_invalid_bounds(bounds: object) -> None:
    with pytest.raises(ValueError):
        render_origin_from_bounds(bounds)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        (world_to_render_points, ([[1.0, 2.0]], [0.0, 0.0, 0.0])),
        (world_to_render_points, ([[1.0, 2.0, np.inf]], [0.0, 0.0, 0.0])),
        (render_to_world_points, ([[1.0, 2.0, 3.0]], [0.0, np.nan, 0.0])),
        (encode_relative_float32, ([[1.0, 2.0, 3.0]], [0.0, 0.0])),
    ],
)
def test_point_operations_reject_invalid_inputs(
    operation: object, args: tuple[object, object]
) -> None:
    with pytest.raises(ValueError):
        operation(*args)  # type: ignore[operator]


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(3),
        np.full((4, 4), np.nan),
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        ),
    ],
)
def test_matrix_operations_reject_non_affine_inputs(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError):
        rebase_affine_for_render(matrix, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        absolute_modelview_from_render(matrix, [0.0, 0.0, 0.0])


def test_plane_and_rebasing_reject_invalid_vectors_and_scalars() -> None:
    identity = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError, match="zero vector"):
        rebase_world_plane_for_render([0.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="offset_world"):
        rebase_world_plane_for_render([0.0, 0.0, 1.0], np.inf, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="vbo_origin_local"):
        rebase_affine_for_render(identity, [0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="render_origin_world"):
        absolute_modelview_from_render(identity, [0.0, 0.0, np.nan])


def test_relative_encoding_rejects_float32_overflow() -> None:
    maximum = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="world-to-render"):
        encode_relative_float32(
            [[maximum, 0.0, 0.0]],
            [-maximum, 0.0, 0.0],
        )


# --- perspective depth resolution -------------------------------------------------
#
# Depth picking unprojects a 24-bit depth sample back into world millimetres, so
# the clip range decides whether a surface anchor can be resolved at all.  These
# cases pin the real artifact framings the application produces.


_VIEWPORT_PIXELS = 900
_FOV_Y_DEG = 45.0


def _pixel_footprint_mm(depth_mm: float) -> float:
    """World millimetres one pixel spans at ``depth_mm``, for a 900 px view."""

    span = 2.0 * depth_mm * np.tan(np.radians(_FOV_Y_DEG) * 0.5)
    return float(span / _VIEWPORT_PIXELS)


def _pick_tolerance_mm(depth_mm: float, coordinate_grid_um: float = 1.0) -> float:
    """Mirror ``artifact_surface_measurement._pick_depth_tolerance_um``.

    The measurement worker accepts a framebuffer depth sample only when it
    agrees with the CPU ray/triangle hit to within
    ``max(50 um, 2 * pixel_footprint + 4 * grid)``.  A clip range that cannot
    resolve depth to inside that window makes every native surface anchor fail.
    """

    footprint_um = _pixel_footprint_mm(depth_mm) * 1000.0
    return max(50.0, 2.0 * footprint_um + 4.0 * coordinate_grid_um) / 1000.0


@pytest.mark.parametrize(
    ("label", "largest_dimension_mm"),
    [
        ("100 mm potsherd", 100.0),
        ("200 mm roof tile", 200.0),
        ("450 mm storage jar", 450.0),
    ],
)
def test_fit_to_object_framing_resolves_depth_inside_the_pick_tolerance(
    label: str,
    largest_dimension_mm: float,
) -> None:
    # ``fit`` places the camera at twice the largest dimension while the
    # bounding-sphere radius is the half-diagonal, so ``distance <= 4 * radius``
    # for every real artifact.  That is exactly the regime where the previous
    # clip policy collapsed the near plane to 1e-5 mm.
    radius = largest_dimension_mm * (3.0**0.5) / 2.0
    distance = largest_dimension_mm * 2.0
    assert distance <= radius * 4.0, label

    near, far = compute_clip_range(
        view_distance_mm=distance,
        scene_radius_mm=radius,
        camera_distance_mm=distance,
        horizon_factor=10.0,
    )
    assert 0.0 < near < far

    # Across the whole depth span the artifact occupies, one depth tick must
    # stay well inside the tolerance the surface-anchor gate applies.
    for depth in (distance - radius, distance, distance + radius):
        resolution = perspective_depth_resolution_mm(
            clip_near_mm=near, clip_far_mm=far, depth_mm=depth
        )
        tolerance = _pick_tolerance_mm(depth)
        assert resolution < tolerance / 4.0, (label, depth, resolution, tolerance)

        # The collapsed near plane the previous code produced in this framing
        # blows straight through the same gate, which is the regression.
        collapsed = perspective_depth_resolution_mm(
            clip_near_mm=1e-5, clip_far_mm=far, depth_mm=depth
        )
        assert collapsed > tolerance * 100.0, (label, depth, collapsed, tolerance)


def test_clip_near_never_collapses_across_framing_ratios() -> None:
    radius = 86.6
    for ratio in (1.0, 1.5, 2.0, 3.0, 4.0, 12.7, 50.0):
        distance = radius * ratio
        near, far = compute_clip_range(
            view_distance_mm=distance,
            scene_radius_mm=radius,
            camera_distance_mm=distance,
            horizon_factor=10.0,
        )
        # A thousandth of the camera distance, never the 1e-5 mm collapse.
        assert near == pytest.approx(max(1e-4, distance * 1e-3))
        assert far > distance + radius
        resolution = perspective_depth_resolution_mm(
            clip_near_mm=near, clip_far_mm=far, depth_mm=distance
        )
        # Depth resolution and pixel footprint both scale with camera
        # distance, so the ratio between them stays bounded at every zoom.
        assert resolution < _pick_tolerance_mm(distance) / 4.0, (ratio, resolution)


def test_perspective_depth_resolution_matches_closed_form() -> None:
    near, far, depth = 0.2, 2200.0, 200.0
    steps = float((1 << 24) - 1)
    expected = (depth * depth * (far - near)) / (near * far * steps)
    assert perspective_depth_resolution_mm(
        clip_near_mm=near, clip_far_mm=far, depth_mm=depth
    ) == pytest.approx(expected)

    # A collapsed near plane is what the regression guards against: the same
    # framing with the old 1e-5 mm near plane cannot resolve a millimetre.
    collapsed = perspective_depth_resolution_mm(
        clip_near_mm=1e-5, clip_far_mm=far, depth_mm=depth
    )
    assert collapsed > 100.0


def test_clip_range_rejects_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_clip_range(
            view_distance_mm=float("nan"),
            scene_radius_mm=1.0,
            camera_distance_mm=1.0,
            horizon_factor=10.0,
        )
    with pytest.raises(ValueError, match="0 < near < far"):
        perspective_depth_resolution_mm(
            clip_near_mm=0.0, clip_far_mm=10.0, depth_mm=1.0
        )
