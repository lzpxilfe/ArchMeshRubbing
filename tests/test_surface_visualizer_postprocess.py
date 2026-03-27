import unittest

import numpy as np

from src.core.flattener import FlattenedMesh
from src.core.mesh_loader import MeshData
from src.core.surface_visualizer import SurfaceVisualizer


class TestSurfaceVisualizerPostprocess(unittest.TestCase):
    def _make_flattened_mesh(self) -> FlattenedMesh:
        mesh = MeshData(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.2],
                    [1.0, 1.0, 0.4],
                    [0.0, 1.0, 0.1],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray(
                [
                    [0, 1, 2],
                    [0, 2, 3],
                ],
                dtype=np.int32,
            ),
            unit="cm",
        )
        return FlattenedMesh(
            uv=np.asarray(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray(
                [
                    [0, 1, 2],
                    [0, 2, 3],
                ],
                dtype=np.int32,
            ),
            original_mesh=mesh,
            distortion_per_face=np.asarray([0.01, 0.02], dtype=np.float64),
        )

    def _make_textured_flattened_mesh(self) -> FlattenedMesh:
        mesh = MeshData(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.2],
                    [1.0, 1.0, 0.4],
                    [0.0, 1.0, 0.1],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray(
                [
                    [0, 1, 2],
                    [0, 2, 3],
                ],
                dtype=np.int32,
            ),
            uv_coords=np.asarray(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            texture=np.asarray(
                [
                    [[0, 0, 0], [255, 255, 255]],
                    [[255, 255, 255], [0, 0, 0]],
                ],
                dtype=np.uint8,
            ),
            unit="cm",
        )
        return FlattenedMesh(
            uv=np.asarray(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray(
                [
                    [0, 1, 2],
                    [0, 2, 3],
                ],
                dtype=np.int32,
            ),
            original_mesh=mesh,
            distortion_per_face=np.asarray([0.01, 0.02], dtype=np.float64),
        )

    def test_preset_with_image_postprocess(self):
        flattened = self._make_flattened_mesh()
        visualizer = SurfaceVisualizer()
        img = visualizer.generate_rubbing(
            flattened,
            width_pixels=80,
            preset="자연(이미지)+CLAHE",
            height_mode="normal_z",
        )
        self.assertEqual(img.image.dtype, np.uint8)
        self.assertEqual(img.image.shape, (img.height_pixels, img.width_pixels))
        self.assertEqual(img.image.ndim, 2)

    def test_local_contrast_preset_outputs_uint8(self):
        flattened = self._make_flattened_mesh()
        visualizer = SurfaceVisualizer()
        img = visualizer.generate_rubbing(
            flattened,
            width_pixels=80,
            preset="로컬 대비(텍스처)",
            height_mode="normal_z",
        )
        self.assertEqual(img.image.dtype, np.uint8)
        self.assertGreaterEqual(int(img.image.min()), 0)
        self.assertLessEqual(int(img.image.max()), 255)

    def test_texture_source_uses_mesh_texture(self):
        flattened = self._make_textured_flattened_mesh()
        visualizer = SurfaceVisualizer()
        img = visualizer.generate_rubbing(
            flattened,
            width_pixels=64,
            texture_source="texture",
            texture_postprocess="local_contrast",
            height_mode="normal_z",
        )
        self.assertEqual(img.image.dtype, np.uint8)
        self.assertEqual(img.image.ndim, 2)
        self.assertGreater(int(img.image.max()), int(img.image.min()))

    def test_hybrid_texture_preset_outputs_variation(self):
        flattened = self._make_textured_flattened_mesh()
        visualizer = SurfaceVisualizer()
        img = visualizer.generate_rubbing(
            flattened,
            width_pixels=64,
            preset="하이브리드(형상+텍스처)",
            height_mode="normal_z",
        )
        self.assertEqual(img.image.dtype, np.uint8)
        self.assertEqual(img.image.ndim, 2)
        self.assertGreater(int(np.std(img.image)), 0)

    def test_light_elevation_control_changes_render(self):
        flattened = self._make_flattened_mesh()
        visualizer = SurfaceVisualizer()
        low = visualizer.generate_rubbing(
            flattened,
            width_pixels=96,
            style="traditional",
            light_angle=35.0,
            light_elevation=12.0,
            height_mode="normal_z",
        )
        high = visualizer.generate_rubbing(
            flattened,
            width_pixels=96,
            style="traditional",
            light_angle=35.0,
            light_elevation=62.0,
            height_mode="normal_z",
        )
        self.assertFalse(np.array_equal(low.image, high.image))

    def test_normalize_postprocess_steps(self):
        steps = SurfaceVisualizer._normalize_postprocess_steps(" CLAHE, local_contrast , denoise ")
        self.assertEqual(tuple(steps), ("clahe", "local_contrast", "denoise"))

        steps = SurfaceVisualizer._normalize_postprocess_steps(["unsharp", "  ", None])
        self.assertEqual(tuple(steps), ("unsharp",))

    def test_merge_postprocess_steps_deduplicates_and_keeps_order(self):
        steps = SurfaceVisualizer._merge_postprocess_steps(
            "clahe,local_contrast",
            ["local_contrast", "unsharp"],
        )
        self.assertEqual(tuple(steps), ("clahe", "local_contrast", "unsharp"))

    def test_as_float01_image_preserves_shape_for_flat_float_input(self):
        image = np.full((6, 4), 0.25, dtype=np.float64)
        out = SurfaceVisualizer._as_float01_image(image)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(out.ndim, 2)
        self.assertTrue(np.allclose(out, 0.0))

    def test_as_float01_image_uses_uint8_range_directly(self):
        image = np.asarray([[0, 127, 255]], dtype=np.uint8)
        out = SurfaceVisualizer._as_float01_image(image)
        expected = image.astype(np.float64) / 255.0
        self.assertTrue(np.allclose(out, expected))


if __name__ == "__main__":
    unittest.main()
