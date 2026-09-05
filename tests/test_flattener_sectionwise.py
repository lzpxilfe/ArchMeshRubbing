import unittest

import numpy as np

from src.core.artifact_cancellation import ArtifactComputationCancelledError
from src.core.flattener import flatten_with_method
from src.core.flatten_models_sectionwise import (
    _bounded_row_shift,
    sectionwise_cylindrical_parameterization,
    sectionwise_quality_gate,
)
from src.core.flatten_utils import _robust_circle_fit_2d
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
    def test_section_fitting_polls_cooperative_cancellation(self):
        mesh, _row_radii, _theta_span = _make_variable_radius_u_patch()
        poll_count = 0

        def cancellation_probe() -> bool:
            nonlocal poll_count
            poll_count += 1
            return poll_count >= 4

        with self.assertRaises(ArtifactComputationCancelledError):
            sectionwise_cylindrical_parameterization(
                mesh,
                axis="y",
                n_sections=24,
                return_meta=True,
                cancellation_probe=cancellation_probe,
            )

        self.assertGreaterEqual(poll_count, 4)

    def test_row_shift_search_polls_each_candidate_for_cancellation(self):
        values = np.linspace(1.0, 2.0, 256, dtype=np.float64)
        poll_count = 0

        def cancellation_probe() -> bool:
            nonlocal poll_count
            poll_count += 1
            return poll_count >= 3

        with self.assertRaises(ArtifactComputationCancelledError):
            _bounded_row_shift(
                du=values * 0.2,
                dv=values * 0.5,
                source_length=values,
                alpha=np.ones_like(values),
                cancellation_probe=cancellation_probe,
            )

        self.assertEqual(poll_count, 3)

    def test_circle_fit_is_stable_at_large_survey_offsets(self):
        angles = np.linspace(-0.7, 0.9, 96, dtype=np.float64)
        expected_center = np.array([10_000_000.25, -20_000_000.5])
        expected_radius = 68.125
        x = expected_center[0] + expected_radius * np.cos(angles)
        y = expected_center[1] + expected_radius * np.sin(angles)

        fitted = _robust_circle_fit_2d(x, y)

        self.assertIsNotNone(fitted)
        assert fitted is not None
        center, radius = fitted
        np.testing.assert_allclose(center, expected_center, atol=1e-6, rtol=0.0)
        self.assertAlmostEqual(radius, expected_radius, places=6)

    def test_relief_on_the_measured_axis_is_reported_not_refused(self):
        """A corded tile's back is steep everywhere, and that is the point.

        Under a fitted centre the three distortion numbers say how well the
        centre was fitted, so all three refuse.  On the measured axis nothing
        is fitted and they say how steeply the wall stands - a cord 0.35 mm
        proud at a 3 mm pitch carries about a quarter more area than the
        cylinder - so all three are reported and none refuses.  Every other
        rule here is untouched.
        """

        meta = {
            "section_fit_valid_count": 12,
            "section_count": 12,
            "section_spacing": 1.0,
            "section_centerline_length": 10.0,
            "section_mean_span": float(np.deg2rad(30.0)),
        }
        steep = {"max": 0.30, "p95": 0.20, "mean": 0.09, "median": 0.05}

        needs_fallback, reason = sectionwise_quality_gate(
            meta, distortion_summary=steep
        )
        self.assertTrue(needs_fallback)
        self.assertEqual(reason, "section_distortion_max")

        on_axis = dict(meta, section_center_policy="axis_origin")
        needs_fallback, reason = sectionwise_quality_gate(
            on_axis, distortion_summary=steep
        )
        self.assertFalse(needs_fallback)
        self.assertEqual(reason, "")

        # A degenerate trace is not relief, and the axis does not excuse it.
        collapsed = dict(on_axis, section_centerline_length=0.0)
        needs_fallback, reason = sectionwise_quality_gate(
            collapsed, distortion_summary=steep
        )
        self.assertTrue(needs_fallback)
        self.assertEqual(reason, "section_trace_degenerate")

        sparse = dict(on_axis, section_fit_valid_count=3)
        needs_fallback, reason = sectionwise_quality_gate(
            sparse, distortion_summary=steep
        )
        self.assertTrue(needs_fallback)
        self.assertEqual(reason, "section_fit_too_sparse")

    def test_sectionwise_quality_gate_uses_radian_arc_span(self):
        meta = {
            "section_fit_valid_count": 12,
            "section_count": 12,
            "section_spacing": 1.0,
            "section_centerline_length": 10.0,
            "section_mean_span": float(np.deg2rad(30.0)),
        }

        needs_fallback, reason = sectionwise_quality_gate(meta)
        self.assertFalse(needs_fallback)
        self.assertEqual(reason, "")

        meta["section_mean_span"] = float(np.deg2rad(10.0))
        needs_fallback, reason = sectionwise_quality_gate(meta)
        self.assertTrue(needs_fallback)
        self.assertEqual(reason, "section_arc_span_too_small")

    def test_sparse_sectionwise_input_reports_policy_fallback(self):
        mesh, _row_radii, _theta_span = _make_variable_radius_u_patch(n_theta=4, n_len=2)

        out = flatten_with_method(mesh, method="section", cylinder_axis="y")
        meta = dict(getattr(out, "meta", {}) or {})

        self.assertEqual(str(meta.get("flatten_method")), "area")
        self.assertEqual(str(meta.get("requested_flatten_method")), "section")
        self.assertEqual(str(meta.get("fallback_from")), "section")
        self.assertEqual(str(meta.get("fallback_reason")), "too_few_points")
        self.assertEqual(str(meta.get("fallback_used_method")), "area")

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
        mean_span_rad = float(meta.get("section_mean_span_rad", 0.0))
        mean_span_deg = float(meta.get("section_mean_span_deg", 0.0))
        self.assertAlmostEqual(float(meta.get("section_mean_span", 0.0)), mean_span_rad)
        self.assertAlmostEqual(mean_span_deg, float(np.rad2deg(mean_span_rad)))

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

    def test_sectionwise_uses_section_guides_and_bottom_record_flip(self):
        mesh, row_radii, _theta_span = _make_variable_radius_u_patch(n_theta=18, n_len=12)
        ys = np.linspace(0.0, 120.0, int(row_radii.size), dtype=np.float64)

        guide_rows = [1, 3, 6, 9, 11]
        guides = [
            {
                "station": float(ys[idx]),
                "radius_world": float(row_radii[idx]),
                "confidence": 0.9,
            }
            for idx in guide_rows
        ]

        out_top = flatten_with_method(
            mesh,
            method="section",
            cylinder_axis="y",
            section_guides=guides,
            section_record_view="top",
        )
        out_bottom = flatten_with_method(
            mesh,
            method="section",
            cylinder_axis="y",
            section_guides=guides,
            section_record_view="bottom",
        )

        uv_top = np.asarray(out_top.uv, dtype=np.float64)
        uv_bottom = np.asarray(out_bottom.uv, dtype=np.float64)
        self.assertEqual(uv_top.shape, uv_bottom.shape)
        self.assertTrue(np.isfinite(uv_top).all())
        self.assertTrue(np.isfinite(uv_bottom).all())

        meta_top = dict(getattr(out_top, "meta", {}) or {})
        meta_bottom = dict(getattr(out_bottom, "meta", {}) or {})

        self.assertEqual(int(meta_top.get("section_guided_count", 0)), len(guides))
        self.assertEqual(int(meta_top.get("section_guided_radius_count", 0)), len(guides))
        self.assertEqual(str(meta_top.get("section_record_view", "")), "top")
        self.assertFalse(bool(meta_top.get("section_u_flipped", False)))

        self.assertEqual(int(meta_bottom.get("section_guided_count", 0)), len(guides))
        self.assertEqual(str(meta_bottom.get("section_record_view", "")), "bottom")
        self.assertTrue(bool(meta_bottom.get("section_u_flipped", False)))

        mirror_sum = uv_top[:, 0] + uv_bottom[:, 0]
        self.assertLess(float(np.std(mirror_sum)), 1e-5)


if __name__ == "__main__":
    unittest.main()
