import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.core.flattener import FlattenedMesh, flatten_with_method
from src.core.mesh_loader import MeshData, MeshLoader
from src.core.recording_surface_review import (
    RecordingSurfaceReviewOptions,
    build_recording_surface_summary_lines,
    render_recording_surface_review,
)


class TestRecordingSurfaceReview(unittest.TestCase):
    FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

    def _make_variable_radius_u_patch(
        self,
        *,
        radius_base: float = 30.0,
        radius_amp: float = 6.0,
        length: float = 100.0,
        theta0: float = -0.5 * np.pi,
        theta1: float = 0.5 * np.pi,
        n_theta: int = 18,
        n_len: int = 10,
    ) -> tuple[MeshData, np.ndarray]:
        ys = np.linspace(0.0, float(length), int(n_len) + 1, dtype=np.float64)
        thetas = np.linspace(float(theta0), float(theta1), int(n_theta) + 1, dtype=np.float64)

        vertices: list[list[float]] = []
        radii: list[float] = []
        for y in ys:
            radius = float(radius_base) + float(radius_amp) * float(
                np.sin(np.pi * float(y) / float(length))
            )
            radii.append(radius)
            for theta in thetas:
                vertices.append(
                    [
                        radius * float(np.cos(theta)),
                        float(y),
                        radius * float(np.sin(theta)),
                    ]
                )

        def idx(row: int, col: int) -> int:
            return int(row) * (int(n_theta) + 1) + int(col)

        faces: list[list[int]] = []
        for row in range(int(n_len)):
            for col in range(int(n_theta)):
                a = idx(row, col)
                b = idx(row, col + 1)
                c = idx(row + 1, col + 1)
                d = idx(row + 1, col)
                faces.append([a, b, c])
                faces.append([a, c, d])

        mesh = MeshData(
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int32),
            unit="mm",
        )
        return mesh, np.asarray(radii, dtype=np.float64)

    def _make_flattened_square(self) -> FlattenedMesh:
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        mesh = MeshData(vertices=vertices, faces=faces, unit="cm")
        uv = np.asarray(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return FlattenedMesh(
            uv=uv,
            faces=faces,
            original_mesh=mesh,
            distortion_per_face=np.asarray([0.01, 0.02], dtype=np.float64),
            scale=1.0,
            meta={"flatten_method": "section"},
        )

    def test_render_review_sheet_builds_combined_image(self):
        flattened = self._make_flattened_square()
        rubbing_image = Image.new("RGB", (320, 160), color=(180, 180, 180))

        review = render_recording_surface_review(
            flattened,
            options=RecordingSurfaceReviewOptions(
                title="기록면 검토 시트",
                summary_lines=("기록면: 상면", "모드: 기와 해석 기반"),
            ),
            rubbing_image=rubbing_image,
        )

        self.assertEqual(review.rubbing_image.size, (320, 160))
        self.assertEqual(review.outline_image.size, (320, 160))
        self.assertGreater(review.combined_image.size[0], 640)
        self.assertGreater(review.combined_image.size[1], 160)

        outline_arr = np.asarray(review.outline_image, dtype=np.uint8)
        self.assertTrue(np.any(outline_arr < 250))

        combined_arr = np.asarray(review.combined_image, dtype=np.uint8)
        self.assertTrue(np.any(combined_arr < 250))

    def test_build_summary_lines_include_tile_context(self):
        flattened = self._make_flattened_square()
        lines = build_recording_surface_summary_lines(
            flattened,
            record_label="상면 기록면",
            target_label="현재 선택",
            strategy_suffix=" (기와 해석 기반)",
            mode_label="기와 해석 기반",
            tile_class_label="수키와",
            split_scheme_label="4분할",
            record_strategy_label="표준 시점 가시면 자동 준비",
            guide_count=4,
            mandrel_radius_world=21.5,
            extra_lines=("외곽과 연속 기록면을 함께 검토합니다.",),
        )

        self.assertGreaterEqual(len(lines), 4)
        self.assertIn("기록면: 상면 기록면", lines[0])
        self.assertIn("대상: 현재 선택", lines[0])
        self.assertIn("유형: 수키와", lines[1])
        self.assertIn("분할: 4분할", lines[1])
        self.assertIn("대표 단면 가이드 4개", lines[1])
        self.assertIn("와통 반경 21.500 cm", lines[1])
        self.assertIn("왜곡(평균/최대)", lines[2])
        self.assertEqual(lines[-1], "외곽과 연속 기록면을 함께 검토합니다.")

    def test_render_review_sheet_draws_scale_and_orientation_annotations(self):
        flattened = self._make_flattened_square()
        flattened.meta = {"section_record_view": "top"}
        rubbing_image = Image.new("RGB", (360, 180), color=(180, 180, 180))

        review = render_recording_surface_review(
            flattened,
            options=RecordingSurfaceReviewOptions(
                title="주석 포함 검토 시트",
                summary_lines=("기록면: 상면",),
            ),
            rubbing_image=rubbing_image,
        )

        rubbing_arr = np.asarray(review.rubbing_image, dtype=np.uint8)
        top_right_crop = rubbing_arr[12:70, 220:340]
        bottom_left_crop = rubbing_arr[110:172, 18:150]

        self.assertTrue(np.any(top_right_crop < 150))
        self.assertTrue(np.any(bottom_left_crop < 150))

    def test_render_review_sheet_passes_texture_adjustments_to_visualizer(self):
        flattened = self._make_flattened_square()
        captured = {}

        class _FakeRubbing:
            def to_pil_image(self):
                return Image.new("L", (320, 160), color=180)

        class _FakeVisualizer:
            def __init__(self, default_dpi=300):
                captured["dpi"] = int(default_dpi)

            def generate_rubbing(self, flattened_arg, **kwargs):
                captured["flattened"] = flattened_arg
                captured.update(kwargs)
                return _FakeRubbing()

        with patch("src.core.surface_visualizer.SurfaceVisualizer", _FakeVisualizer):
            review = render_recording_surface_review(
                flattened,
                options=RecordingSurfaceReviewOptions(
                    dpi=450,
                    width_pixels=900,
                    rubbing_preset="자연(이미지)+CLAHE",
                    rubbing_detail_scale=1.35,
                    rubbing_smooth_sigma_extra=0.4,
                    rubbing_texture_postprocess="unsharp",
                    rubbing_light_angle=0.0,
                    rubbing_light_elevation=19.0,
                    title="기록면 검토 시트",
                ),
            )

        self.assertEqual(captured.get("dpi"), 450)
        self.assertEqual(captured.get("flattened"), flattened)
        self.assertEqual(captured.get("width_pixels"), 900)
        self.assertEqual(captured.get("preset"), "자연(이미지)+CLAHE")
        self.assertAlmostEqual(float(captured.get("texture_detail_scale", 0.0)), 1.35, places=6)
        self.assertAlmostEqual(float(captured.get("texture_smooth_sigma_extra", 0.0)), 0.4, places=6)
        self.assertEqual(captured.get("texture_postprocess_extra"), "unsharp")
        self.assertAlmostEqual(float(captured.get("light_angle", -1.0)), 0.0, places=6)
        self.assertAlmostEqual(float(captured.get("light_elevation", 0.0)), 19.0, places=6)
        self.assertEqual(review.rubbing_image.size, (320, 160))

    def test_render_review_sheet_leaves_light_unset_for_preset_defaults(self):
        flattened = self._make_flattened_square()
        captured = {}

        class _FakeRubbing:
            def to_pil_image(self):
                return Image.new("L", (320, 160), color=180)

        class _FakeVisualizer:
            def __init__(self, default_dpi=300):
                captured["dpi"] = int(default_dpi)

            def generate_rubbing(self, flattened_arg, **kwargs):
                captured["flattened"] = flattened_arg
                captured.update(kwargs)
                return _FakeRubbing()

        with patch("src.core.surface_visualizer.SurfaceVisualizer", _FakeVisualizer):
            render_recording_surface_review(
                flattened,
                options=RecordingSurfaceReviewOptions(
                    dpi=300,
                    width_pixels=640,
                    rubbing_preset="부드러움",
                    title="기록면 검토 시트",
                ),
            )

        self.assertIsNone(captured.get("light_angle"))
        self.assertIsNone(captured.get("light_elevation"))

    def test_render_review_sheet_for_tile_guided_section_unwrap(self):
        mesh, row_radii = self._make_variable_radius_u_patch()
        stations = np.linspace(0.0, 100.0, int(row_radii.size), dtype=np.float64)
        guide_rows = [1, 4, 7, 9]
        guides = [
            {
                "station": float(stations[idx]),
                "radius_world": float(row_radii[idx]),
                "confidence": 0.95,
            }
            for idx in guide_rows
        ]

        flattened = flatten_with_method(
            mesh,
            method="section",
            cylinder_axis="y",
            section_guides=guides,
            section_record_view="top",
        )

        review = render_recording_surface_review(
            flattened,
            options=RecordingSurfaceReviewOptions(
                width_pixels=720,
                title="기와 기록면 검토 시트",
                summary_lines=(
                    "기록면: 상면",
                    f"대표 단면 가이드: {len(guides)}개",
                ),
            ),
        )

        self.assertGreater(review.rubbing_image.size[0], 400)
        self.assertGreater(review.rubbing_image.size[1], 120)
        self.assertEqual(review.rubbing_image.size, review.outline_image.size)
        self.assertGreater(review.combined_image.size[0], review.rubbing_image.size[0] * 2)

        meta = dict(getattr(flattened, "meta", {}) or {})
        self.assertEqual(str(meta.get("flatten_method")), "section")
        self.assertEqual(str(meta.get("section_record_view", "")), "top")
        self.assertEqual(int(meta.get("section_guided_count", 0)), len(guides))

        outline_arr = np.asarray(review.outline_image, dtype=np.uint8)
        rubbing_arr = np.asarray(review.rubbing_image, dtype=np.uint8)
        self.assertTrue(np.any(outline_arr < 250))
        self.assertTrue(np.any(rubbing_arr < 250))

    def test_render_review_sheet_from_obj_tile_fixture(self):
        loader = MeshLoader(default_unit="mm")
        mesh = loader.load(self.FIXTURE_DIR / "tile_u_patch.obj")

        guides = [
            {"station": 0.0, "radius_world": 20.0, "confidence": 0.9},
            {"station": 50.0, "radius_world": 24.0, "confidence": 0.95},
            {"station": 100.0, "radius_world": 20.0, "confidence": 0.9},
        ]

        flattened = flatten_with_method(
            mesh,
            method="section",
            cylinder_axis="y",
            section_guides=guides,
            section_record_view="top",
        )
        review = render_recording_surface_review(
            flattened,
            options=RecordingSurfaceReviewOptions(
                width_pixels=640,
                title="기와 OBJ 기록면 검토 시트",
                summary_lines=(
                    "fixture: tile_u_patch.obj",
                    "기록면: 상면",
                ),
            ),
        )

        self.assertEqual(mesh.n_vertices, 15)
        self.assertEqual(mesh.n_faces, 16)
        self.assertTrue(np.isfinite(np.asarray(flattened.uv, dtype=np.float64)).all())
        self.assertGreater(review.combined_image.size[0], 1000)
        self.assertGreater(review.combined_image.size[1], 150)

        meta = dict(getattr(flattened, "meta", {}) or {})
        self.assertEqual(str(meta.get("flatten_method")), "section")
        self.assertEqual(int(meta.get("section_guided_count", 0)), len(guides))
        self.assertEqual(str(meta.get("section_record_view", "")), "top")


if __name__ == "__main__":
    unittest.main()
