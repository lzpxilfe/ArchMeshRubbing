import unittest

import numpy as np

from src.core.flattener import flatten_with_method
from src.core.mesh_loader import MeshData


def _make_half_cylinder_patch(
    *,
    radius: float = 1.0,
    length: float = 2.0,
    n_theta: int = 24,
    n_len: int = 8,
) -> MeshData:
    thetas = np.linspace(-0.5 * np.pi, 0.5 * np.pi, int(n_theta) + 1, dtype=np.float64)
    ys = np.linspace(0.0, float(length), int(n_len) + 1, dtype=np.float64)

    vertices: list[list[float]] = []
    for y in ys:
        for th in thetas:
            vertices.append(
                [
                    float(radius) * float(np.cos(th)),
                    float(y),
                    float(radius) * float(np.sin(th)),
                ]
            )
    v = np.asarray(vertices, dtype=np.float64)

    def idx(iy: int, it: int) -> int:
        return int(iy) * (int(n_theta) + 1) + int(it)

    faces: list[list[int]] = []
    for iy in range(int(n_len)):
        for it in range(int(n_theta)):
            a = idx(iy, it)
            b = idx(iy, it + 1)
            c = idx(iy + 1, it + 1)
            d = idx(iy + 1, it)
            faces.append([a, b, c])
            faces.append([a, c, d])
    f = np.asarray(faces, dtype=np.int32)
    return MeshData(vertices=v, faces=f, unit="cm")


class TestFlattenerSizeGuard(unittest.TestCase):
    def test_size_guard_warns_for_large_radius(self):
        mesh = _make_half_cylinder_patch()
        out = flatten_with_method(
            mesh,
            method="cylinder",
            cylinder_axis="y",
            cylinder_radius=6.0,
        )
        meta = dict(getattr(out, "meta", {}) or {})
        self.assertTrue(bool(meta.get("flatten_size_warning", False)))
        self.assertFalse(bool(meta.get("flatten_size_guard_applied", False)))
        self.assertGreater(float(meta.get("flatten_size_dim_ratio_before", 0.0)), 6.0)
        self.assertAlmostEqual(float(meta.get("flatten_size_guard_scale", 1.0)), 1.0, places=9)

    def test_size_guard_applies_for_extreme_radius(self):
        mesh = _make_half_cylinder_patch()
        out = flatten_with_method(
            mesh,
            method="cylinder",
            cylinder_axis="y",
            cylinder_radius=600.0,
        )
        meta = dict(getattr(out, "meta", {}) or {})
        self.assertTrue(bool(meta.get("flatten_size_warning", False)))
        self.assertTrue(bool(meta.get("flatten_size_guard_applied", False)))
        self.assertLess(float(meta.get("flatten_size_dim_ratio_after", 9999.0)), 13.0)
        self.assertLess(float(out.width), float(np.max(mesh.extents)) * 13.0)
        self.assertTrue(np.isfinite(np.asarray(out.uv, dtype=np.float64)).all())

    def test_normal_lscm_does_not_warn(self):
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        mesh = MeshData(vertices=vertices, faces=faces, unit="cm")
        out = flatten_with_method(mesh, method="lscm", iterations=0)
        meta = dict(getattr(out, "meta", {}) or {})
        self.assertFalse(bool(meta.get("flatten_size_warning", False)))
        self.assertFalse(bool(meta.get("flatten_size_guard_applied", False)))


if __name__ == "__main__":
    unittest.main()
