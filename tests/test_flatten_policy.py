import unittest

import numpy as np

from src.core.flatten_policy import fallback_chain_for_context, recommend_flatten_mode
from src.core.mesh_loader import MeshData
from src.core.tile_form_model import SectionObservation, TileInterpretationState


def _make_variable_radius_u_patch(
    *,
    radius_base: float = 30.0,
    radius_amp: float = 8.0,
    length: float = 120.0,
    theta0: float = -0.5 * np.pi,
    theta1: float = 0.5 * np.pi,
    n_theta: int = 30,
    n_len: int = 24,
) -> MeshData:
    ys = np.linspace(0.0, float(length), int(n_len) + 1, dtype=np.float64)
    thetas = np.linspace(float(theta0), float(theta1), int(n_theta) + 1, dtype=np.float64)

    vertices: list[list[float]] = []
    for y in ys:
        r = float(radius_base) + float(radius_amp) * float(np.sin(np.pi * float(y) / float(length)))
        for th in thetas:
            vertices.append([r * float(np.cos(th)), float(y), r * float(np.sin(th))])
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

    return MeshData(vertices=v, faces=np.asarray(faces, dtype=np.int32), unit="mm")


class TestFlattenPolicy(unittest.TestCase):
    def test_tile_like_mesh_prefers_sectionwise(self):
        mesh = _make_variable_radius_u_patch()
        state = TileInterpretationState(
            section_observations=[
                SectionObservation(station=10.0, accepted=True, profile_point_count=64),
                SectionObservation(station=60.0, accepted=True, profile_point_count=64),
                SectionObservation(station=110.0, accepted=True, profile_point_count=64),
            ]
        )

        rec = recommend_flatten_mode(mesh, state, None)

        self.assertTrue(rec.enabled)
        self.assertEqual(rec.method, "section")
        self.assertEqual(rec.ui_label, "기와 추천 펼침")
        self.assertGreaterEqual(rec.confidence, 0.65)
        self.assertIn("section", rec.fallback_chain)

    def test_non_tile_mesh_does_not_force_sectionwise(self):
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

        rec = recommend_flatten_mode(mesh, None, None)

        self.assertFalse(rec.enabled)
        self.assertEqual(rec.method, "arap")

    def test_section_fallback_chain_order(self):
        self.assertEqual(
            fallback_chain_for_context("section"),
            ["section", "area", "cylinder", "arap"],
        )


if __name__ == "__main__":
    unittest.main()
