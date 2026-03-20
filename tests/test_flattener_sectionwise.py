import unittest

import numpy as np

from src.core.flattener import flatten_with_method
from src.core.mesh_loader import MeshData


def _make_variable_radius_u_patch(
    *,
    radius_base: float = 30.0,
    radius_amp: float = 8.0,
    length: float = 120.0,
    theta0: float = -0.5 * np.pi,
    theta1: float = 0.5 * np.pi,
    n_theta: int = 30,
    n_len: int = 24,
) -> tuple[MeshData, np.ndarray, float]:
    ys = np.linspace(0.0, float(length), int(n_len) + 1, dtype=np.float64)
    thetas = np.linspace(float(theta0), float(theta1), int(n_theta) + 1, dtype=np.float64)

    vertices: list[list[float]] = []
    radii: list[float] = []
    for y in ys:
        r = float(radius_base) + float(radius_amp) * float(np.sin(np.pi * float(y) / float(length)))
        radii.append(r)
        for th in thetas:
            vertices.append(
                [
                    r * float(np.cos(th)),
                    float(y),
                    r * float(np.sin(th)),
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

    mesh = MeshData(vertices=v, faces=np.asarray(faces, dtype=np.int32), unit="mm")
    theta_span = float(theta1 - theta0)
    return mesh, np.asarray(radii, dtype=np.float64), theta_span


class TestFlattenerSectionwise(unittest.TestCase):
    def test_sectionwise_unwrap_tracks_variable_cross_section_width(self):
        mesh, row_radii, theta_span = _make_variable_radius_u_patch()
        n_rows = int(row_radii.size)
        n_cols = 31

        out_cyl = flatten_with_method(mesh, method="cylinder", cylinder_axis="y")
        out_section = flatten_with_method(mesh, method="section", cylinder_axis="y")

        uv_cyl = np.asarray(out_cyl.uv, dtype=np.float64)
        uv_section = np.asarray(out_section.uv, dtype=np.float64)

        self.assertEqual(uv_cyl.shape, (n_rows * n_cols, 2))
        self.assertEqual(uv_section.shape, (n_rows * n_cols, 2))
        self.assertTrue(np.isfinite(uv_section).all())

        def row_spans(uv: np.ndarray) -> np.ndarray:
            spans = []
            for iy in range(n_rows):
                row = uv[iy * n_cols : (iy + 1) * n_cols, 0]
                spans.append(float(np.max(row) - np.min(row)))
            return np.asarray(spans, dtype=np.float64)

        expected = theta_span * row_radii
        spans_cyl = row_spans(uv_cyl)
        spans_section = row_spans(uv_section)

        err_cyl = float(np.mean(np.abs(spans_cyl - expected) / np.maximum(expected, 1e-9)))
        err_section = float(np.mean(np.abs(spans_section - expected) / np.maximum(expected, 1e-9)))

        self.assertLess(err_section, 0.08)
        self.assertLess(err_section, err_cyl * 0.5)

        meta = dict(getattr(out_section, "meta", {}) or {})
        self.assertEqual(str(meta.get("flatten_method")), "section")
        self.assertTrue(bool(meta.get("sectionwise", False)))
        self.assertGreaterEqual(int(meta.get("section_count", 0)), 12)

    def test_arap_accepts_section_initialization(self):
        mesh, _row_radii, _theta_span = _make_variable_radius_u_patch()
        out = flatten_with_method(
            mesh,
            method="arap",
            iterations=3,
            initial_method="section",
            cylinder_axis="y",
        )

        uv = np.asarray(out.uv, dtype=np.float64)
        self.assertEqual(uv.shape[0], mesh.n_vertices)
        self.assertTrue(np.isfinite(uv).all())

        meta = dict(getattr(out, "meta", {}) or {})
        self.assertEqual(str(meta.get("flatten_method")), "arap")
        self.assertEqual(str(meta.get("initial_method")), "section")
        self.assertTrue(bool(meta.get("sectionwise", False)))


if __name__ == "__main__":
    unittest.main()
