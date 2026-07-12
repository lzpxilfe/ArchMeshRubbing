from __future__ import annotations

import numpy as np
import pytest

from src.gui.render_coordinates import (
    absolute_modelview_from_render,
    encode_relative_float32,
    rebase_affine_for_render,
    rebase_world_plane_for_render,
    render_origin_from_bounds,
    render_to_world_points,
    world_to_render_points,
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
