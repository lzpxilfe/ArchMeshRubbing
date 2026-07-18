import unittest

import numpy as np

from src.core.alignment_utils import (
    compose_align_matrices,
    compute_floor_contact_shift,
    compute_minimax_center_shift,
    compute_nonpenetration_lift,
    fit_plane_normal,
    orient_plane_normal_toward,
    require_affine_matrix4x4,
    require_rigid_matrix4x4,
    rotation_matrix_align_vectors,
    scene_trs_matrix,
    scene_trs_matrix_about_pivot,
    transform_bounds,
    transform_directions,
    transform_plane_world_to_local,
    transform_points,
)


class TestAlignmentUtils(unittest.TestCase):
    def test_scene_trs_matches_fixed_opengl_golden_matrix(self):
        matrix = scene_trs_matrix(
            [10.0, 20.0, 30.0],
            [90.0, 90.0, 0.0],
            2.0,
        )
        expected = np.array(
            [
                [0.0, 0.0, 2.0, 10.0],
                [2.0, 0.0, 0.0, 20.0],
                [0.0, 2.0, 0.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        point = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

        np.testing.assert_allclose(matrix, expected, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            transform_points(point, matrix),
            [[10.0, 22.0, 30.0]],
            rtol=0.0,
            atol=1e-12,
        )

    def test_scene_trs_multi_axis_golden_and_inverse_roundtrip(self):
        matrix = scene_trs_matrix([10.0, -5.0, 2.0], [30.0, 45.0, 60.0], 2.0)
        point = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)

        transformed = transform_points(point, matrix)

        np.testing.assert_allclose(
            transformed,
            [[12.500257725523, -4.760461016789, 9.049207925823]],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            transform_points(transformed, np.linalg.inv(matrix)),
            point,
            rtol=0.0,
            atol=1e-12,
        )

    def test_transform_points_rejects_finite_input_that_overflows(self):
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 0] = 1e200

        with self.assertRaisesRegex(ValueError, "transformed points"):
            transform_points([[1e200, 0.0, 0.0]], matrix)

        with self.assertRaisesRegex(ValueError, "transformed directions"):
            transform_directions([[1e200, 0.0, 0.0]], matrix)

    def test_scene_trs_about_pivot_keeps_object_centered_without_recentering(self):
        matrix = scene_trs_matrix_about_pivot(
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 90.0],
            1.0,
            [100.0, 0.0, 0.0],
        )

        np.testing.assert_allclose(
            transform_points([[110.0, 0.0, 0.0]], matrix),
            [[105.0, 10.0, 0.0]],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            transform_points([[100.0, 0.0, 0.0]], matrix),
            [[105.0, 0.0, 0.0]],
            rtol=0.0,
            atol=1e-12,
        )

    def test_transform_bounds_and_world_plane_use_same_matrix(self):
        matrix = scene_trs_matrix([10.0, 20.0, 30.0], [90.0, 90.0, 0.0], 2.0)
        bounds = np.array([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]])
        transformed_bounds = transform_bounds(bounds, matrix)
        corners = np.array(
            [[x, y, z] for x in (-1.0, 4.0) for y in (-2.0, 5.0) for z in (-3.0, 6.0)]
        )
        transformed_corners = transform_points(corners, matrix)
        np.testing.assert_allclose(
            transformed_bounds,
            [transformed_corners.min(axis=0), transformed_corners.max(axis=0)],
            rtol=0.0,
            atol=1e-12,
        )

        world_origin = transform_points([[0.0, 0.0, 1.5]], matrix)[0]
        origin_local, normal_local = transform_plane_world_to_local(
            world_origin,
            [0.0, 1.0, 0.0],
            matrix,
        )
        np.testing.assert_allclose(origin_local, [0.0, 0.0, 1.5], rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(normal_local)), 1.0, places=12)

    def test_rigid_validation_and_delta_parent_composition(self):
        parent = scene_trs_matrix([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], 1.0)
        delta = scene_trs_matrix([-2.0, 0.5, 1.0], [0.0, 15.0, 0.0], 1.0)
        composed = compose_align_matrices(delta, parent)
        point = np.array([[0.25, -0.5, 2.0]])
        np.testing.assert_allclose(
            transform_points(point, composed),
            transform_points(transform_points(point, parent), delta),
            rtol=0.0,
            atol=1e-12,
        )

        invalid_matrices = []
        scaled = np.eye(4)
        scaled[0, 0] = 2.0
        invalid_matrices.append(scaled)
        sheared = np.eye(4)
        sheared[0, 1] = 0.1
        invalid_matrices.append(sheared)
        reflected = np.eye(4)
        reflected[0, 0] = -1.0
        invalid_matrices.append(reflected)
        perspective = np.eye(4)
        perspective[3, 0] = 0.1
        invalid_matrices.append(perspective)
        nonfinite = np.eye(4)
        nonfinite[0, 0] = np.inf
        invalid_matrices.append(nonfinite)
        for invalid in invalid_matrices:
            with self.subTest(matrix=invalid), self.assertRaises(ValueError):
                require_rigid_matrix4x4(invalid)

        metadata_scale = np.diag([10.0, 10.0, 10.0, 1.0])
        np.testing.assert_allclose(require_affine_matrix4x4(metadata_scale), metadata_scale)

    def test_rotation_matrix_align_vectors_antiparallel(self):
        src = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        dst = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        rot = rotation_matrix_align_vectors(src, dst)
        out = rot @ src

        np.testing.assert_allclose(out, dst, atol=1e-8, rtol=0.0)

    def test_rotation_matrix_align_vectors_general(self):
        src = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        dst = np.array([-4.0, 5.0, 2.0], dtype=np.float64)
        src_u = src / np.linalg.norm(src)
        dst_u = dst / np.linalg.norm(dst)

        rot = rotation_matrix_align_vectors(src_u, dst_u)
        out = rot @ src_u

        np.testing.assert_allclose(out, dst_u, atol=1e-8, rtol=0.0)

    def test_fit_plane_normal_with_outlier(self):
        # Plane: z = 0.5 * x + 1.0
        xs = np.linspace(-4.0, 4.0, 9)
        ys = np.linspace(-3.0, 3.0, 7)
        pts = []
        for x in xs:
            for y in ys:
                z = 0.5 * x + 1.0
                pts.append([x, y, z])
        pts = np.asarray(pts, dtype=np.float64)

        # Add a strong outlier.
        pts = np.vstack([pts, np.array([[100.0, 100.0, -500.0]], dtype=np.float64)])

        fit = fit_plane_normal(pts, robust=True)
        self.assertIsNotNone(fit)
        normal, centroid = fit  # type: ignore[misc]

        expected = np.array([-0.5, 0.0, 1.0], dtype=np.float64)
        expected /= np.linalg.norm(expected)

        # Allow sign ambiguity.
        align = abs(float(np.dot(normal, expected)))
        self.assertGreater(align, 0.999)
        self.assertTrue(np.isfinite(centroid).all())

    def test_orient_plane_normal_toward(self):
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        plane_point = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        toward = np.array([0.0, 0.0, 10.0], dtype=np.float64)

        out = orient_plane_normal_toward(normal, plane_point, toward)
        np.testing.assert_allclose(out, np.array([0.0, 0.0, 1.0]), atol=1e-10, rtol=0.0)

    def test_compute_floor_contact_shift_clamped(self):
        z = np.array([-0.05, -0.01, 0.2], dtype=np.float64)
        self.assertAlmostEqual(compute_floor_contact_shift(z, tolerance=0.02, max_auto_shift=0.2), 0.05)

        z_large = np.array([-3.0, -2.5, 0.1], dtype=np.float64)
        self.assertAlmostEqual(compute_floor_contact_shift(z_large, tolerance=0.02, max_auto_shift=0.2), 0.0)

    def test_compute_minimax_center_shift(self):
        z = np.array([-1.0, 2.0, 3.0], dtype=np.float64)
        self.assertAlmostEqual(compute_minimax_center_shift(z), 1.0)

        z_empty = np.array([], dtype=np.float64)
        self.assertAlmostEqual(compute_minimax_center_shift(z_empty), 0.0)

    def test_compute_nonpenetration_lift(self):
        z = np.array([-0.25, 0.1, 0.4], dtype=np.float64)
        self.assertAlmostEqual(compute_nonpenetration_lift(z, floor_z=0.0), 0.25)

        z_ok = np.array([0.0, 0.2, 1.5], dtype=np.float64)
        self.assertAlmostEqual(compute_nonpenetration_lift(z_ok, floor_z=0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
