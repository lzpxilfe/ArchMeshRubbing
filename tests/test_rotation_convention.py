import unittest

import numpy as np

from src.core.mesh_loader import MeshData
from src.core.profile_exporter import ProfileExporter


def _opengl_rotation_matrix_xyz_deg(rotation_deg: np.ndarray) -> np.ndarray:
    """Match OpenGL fixed-function order: glRotate(X)->glRotate(Y)->glRotate(Z)."""
    rx, ry, rz = [float(v) for v in np.asarray(rotation_deg, dtype=np.float64).reshape(-1)[:3]]
    rx, ry, rz = np.deg2rad([rx, ry, rz])

    cx, sx = float(np.cos(rx)), float(np.sin(rx))
    cy, sy = float(np.cos(ry)), float(np.sin(ry))
    cz, sz = float(np.cos(rz)), float(np.sin(rz))

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_x @ rot_y @ rot_z


class TestRotationConvention(unittest.TestCase):
    def test_profile_exporter_world_bounds_rotation_matches_opengl(self):
        vertices = np.array(
            [
                [-0.3, -0.1, 0.0],
                [0.4, 0.2, 0.1],
                [0.0, 0.6, 0.2],
                [0.2, -0.4, -0.1],
            ],
            dtype=np.float64,
        )
        mesh = MeshData(vertices=vertices, faces=np.zeros((0, 3), dtype=np.int32), unit="cm")

        translation = np.array([1.0, -2.0, 3.0], dtype=np.float64)
        rotation = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        scale = 1.7

        exporter = ProfileExporter(resolution=64)
        mv = np.eye(4, dtype=np.float64)
        proj = np.eye(4, dtype=np.float64)
        vp = np.array([0, 0, 64, 64], dtype=np.int32)

        _, bounds = exporter.extract_silhouette(
            mesh,
            view="top",
            translation=translation,
            rotation=rotation,
            scale=scale,
            opengl_matrices=(mv, proj, vp),
            viewport_image=None,
        )

        world_bounds = np.asarray(bounds.get("world_bounds"), dtype=np.float64)
        self.assertEqual(world_bounds.shape, (2, 3))

        lb = np.asarray(mesh.bounds, dtype=np.float64)
        corners = np.array(
            [
                [lb[0, 0], lb[0, 1], lb[0, 2]],
                [lb[1, 0], lb[0, 1], lb[0, 2]],
                [lb[0, 0], lb[1, 1], lb[0, 2]],
                [lb[1, 0], lb[1, 1], lb[0, 2]],
                [lb[0, 0], lb[0, 1], lb[1, 2]],
                [lb[1, 0], lb[0, 1], lb[1, 2]],
                [lb[0, 0], lb[1, 1], lb[1, 2]],
                [lb[1, 0], lb[1, 1], lb[1, 2]],
            ],
            dtype=np.float64,
        )
        corners = corners * float(scale)
        rot = _opengl_rotation_matrix_xyz_deg(rotation)
        corners = (rot @ corners.T).T
        corners = corners + translation

        expected_min = corners.min(axis=0)
        expected_max = corners.max(axis=0)

        np.testing.assert_allclose(world_bounds[0], expected_min, rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(world_bounds[1], expected_max, rtol=0.0, atol=1e-10)

