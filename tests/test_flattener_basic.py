import unittest

import numpy as np

from src.core.flattener import ARAPFlattener
from src.core.mesh_loader import MeshData


class TestFlattenerBasic(unittest.TestCase):
    def test_arap_flatten_square_mesh_returns_finite_uv(self):
        # Simple planar square made of two triangles.
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        mesh = MeshData(vertices=vertices, faces=faces, unit="cm")

        flattener = ARAPFlattener(max_iterations=5, tolerance=1e-8)
        out = flattener.flatten(mesh, boundary_type="free", initial_method="lscm")

        uv = np.asarray(out.uv, dtype=np.float64)
        self.assertEqual(uv.shape, (4, 2))
        self.assertTrue(np.isfinite(uv).all())

        # Ensure the result isn't completely degenerate.
        tri = uv[faces][:, :, :2]
        e1 = tri[:, 1, :] - tri[:, 0, :]
        e2 = tri[:, 2, :] - tri[:, 0, :]
        area2 = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        self.assertTrue(np.all(area2 > 1e-10))

        dist = np.asarray(out.distortion_per_face, dtype=np.float64)
        self.assertEqual(dist.shape, (2,))
        self.assertTrue(np.isfinite(dist).all())
        self.assertTrue(np.all((dist >= 0.0) & (dist <= 1.0)))

