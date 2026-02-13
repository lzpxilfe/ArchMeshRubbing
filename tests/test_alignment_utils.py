import unittest

import numpy as np

from src.core.alignment_utils import (
    compute_floor_contact_shift,
    fit_plane_normal,
    orient_plane_normal_toward,
    rotation_matrix_align_vectors,
)


class TestAlignmentUtils(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

