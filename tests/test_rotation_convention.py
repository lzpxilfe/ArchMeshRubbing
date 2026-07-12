import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app_interactive import MainWindow, SliceComputeThread
from src.core.mesh_loader import MeshData
from src.core.profile_exporter import ProfileExporter
from src.gui.viewport_3d import SceneObject, TrackballCamera, Viewport3D


_GOLDEN_TRANSLATION = np.array([10.0, 20.0, 30.0], dtype=np.float64)
_GOLDEN_ROTATION_DEG = np.array([90.0, 90.0, 0.0], dtype=np.float64)
_GOLDEN_SCALE = 2.0
_GOLDEN_LOCAL_POINT = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
_GOLDEN_WORLD_POINT = np.array([[10.0, 22.0, 30.0]], dtype=np.float64)


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
    @staticmethod
    def _golden_point_mesh() -> MeshData:
        return MeshData(
            vertices=_GOLDEN_LOCAL_POINT.copy(),
            faces=np.zeros((0, 3), dtype=np.int32),
            unit="cm",
        )

    def test_scene_object_world_bounds_match_fixed_trs_golden(self):
        obj = SceneObject(self._golden_point_mesh(), name="golden")
        obj.translation = _GOLDEN_TRANSLATION.copy()
        obj.rotation = _GOLDEN_ROTATION_DEG.copy()
        obj.scale = _GOLDEN_SCALE

        bounds = np.asarray(obj.get_world_bounds(), dtype=np.float64)

        np.testing.assert_allclose(
            bounds,
            np.vstack([_GOLDEN_WORLD_POINT, _GOLDEN_WORLD_POINT]),
            rtol=0.0,
            atol=1e-10,
        )

    def test_camera_applies_only_render_relative_eye_and_target(self):
        camera = TrackballCamera()
        camera.center = np.array(
            [1_000_000_000.0, -2_000_000_000.0, 500_000_000.0],
            dtype=np.float64,
        )
        camera.pan_offset = np.array([0.125, 1.0, -0.5], dtype=np.float64)
        camera.distance = 10.0
        camera.azimuth = 0.0
        camera.elevation = 0.0
        render_origin = camera.center.copy()

        with patch("src.gui.viewport_3d.gluLookAt") as look_at:
            camera.apply(render_origin)

        args = look_at.call_args.args
        np.testing.assert_allclose(args[:3], [10.125, 1.0, -0.5], atol=1e-12)
        np.testing.assert_allclose(args[3:6], [0.125, 1.0, -0.5], atol=1e-12)
        np.testing.assert_array_equal(args[6:9], [0.0, 0.0, 1.0])

    def test_scene_object_render_matrix_preserves_pivoted_absolute_world_points(self):
        base = np.array(
            [1_000_000_000.0, -1_000_000_000.0, 500_000_000.0],
            dtype=np.float64,
        )
        mesh = MeshData(
            vertices=base
            + np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.125, 0.0]],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        original_vertices = mesh.vertices.copy()
        obj = SceneObject(mesh, "render-origin golden")
        obj.translation = np.array([4.0, -3.0, 2.0], dtype=np.float64)
        obj.rotation = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        obj.scale = 1.25
        obj._amr_preview_pivot_mm = mesh.centroid.copy()
        obj._amr_vbo_origin_local_mm = np.array(
            [base[0] + 0.5, base[1] + 0.0625, base[2]],
            dtype=np.float64,
        )
        scene_origin = np.asarray(obj.get_world_bounds(), dtype=np.float64).mean(axis=0)

        relative = (
            mesh.vertices - obj._amr_vbo_origin_local_mm
        ).astype(np.float32)
        render_matrix = obj.render_model_matrix(scene_origin)
        rendered = (
            relative.astype(np.float64) @ render_matrix[:3, :3].T
            + render_matrix[:3, 3]
        )
        reconstructed = rendered + scene_origin
        expected = (
            mesh.vertices @ obj.local_to_world_matrix()[:3, :3].T
            + obj.local_to_world_matrix()[:3, 3]
        )

        np.testing.assert_allclose(reconstructed, expected, rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(mesh.vertices, original_vertices)

    def test_fit_view_uses_absolute_world_bounds_at_large_survey_offset(self):
        base = np.array(
            [1_000_000_000.0, -1_000_000_000.0, 500_000_000.0],
            dtype=np.float64,
        )
        mesh = MeshData(
            vertices=base
            + np.array(
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 2.0, 1.0]],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        obj = SceneObject(mesh, "large-coordinate fit")
        camera = TrackballCamera()
        viewport_like = SimpleNamespace(
            selected_obj=obj,
            camera=camera,
            update=Mock(),
        )

        Viewport3D.fit_view_to_selected_object(viewport_like)

        bounds = np.asarray(obj.get_world_bounds(), dtype=np.float64)
        np.testing.assert_array_equal(camera.center, bounds.mean(axis=0))
        np.testing.assert_array_equal(camera.pan_offset, np.zeros(3))
        self.assertEqual(camera.distance, 8.0)
        viewport_like.update.assert_called_once_with()

    def test_main_window_world_mesh_builder_matches_fixed_trs_golden(self):
        source = self._golden_point_mesh()

        transformed = MainWindow._build_world_mesh_from_transform(
            source,
            translation=_GOLDEN_TRANSLATION,
            rotation=_GOLDEN_ROTATION_DEG,
            scale=_GOLDEN_SCALE,
        )

        np.testing.assert_allclose(
            transformed.vertices,
            _GOLDEN_WORLD_POINT,
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_array_equal(source.vertices, _GOLDEN_LOCAL_POINT)

    def test_slice_thread_plane_and_contour_match_fixed_trs_golden(self):
        captured: dict[str, np.ndarray] = {}

        class CapturingMeshSlicer:
            def __init__(self, _mesh):
                pass

            def slice_with_plane(self, origin, normal):
                captured["origin"] = np.asarray(origin, dtype=np.float64).copy()
                captured["normal"] = np.asarray(normal, dtype=np.float64).copy()
                return [_GOLDEN_LOCAL_POINT.copy()]

        results: list[tuple[float, object]] = []
        failures: list[tuple[float, str]] = []
        thread = SliceComputeThread(
            self._golden_point_mesh(),
            _GOLDEN_TRANSLATION,
            _GOLDEN_ROTATION_DEG,
            _GOLDEN_SCALE,
            30.0,
        )
        thread.computed.connect(lambda z, contours: results.append((float(z), contours)))
        thread.failed.connect(lambda z, message: failures.append((float(z), str(message))))

        with patch("src.core.mesh_slicer.MeshSlicer", CapturingMeshSlicer):
            thread.run()

        self.assertEqual(failures, [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 30.0)
        np.testing.assert_allclose(
            captured["origin"],
            [-10.0, 0.0, -5.0],
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            captured["normal"],
            [0.0, 1.0, 0.0],
            rtol=0.0,
            atol=1e-10,
        )
        contours = results[0][1]
        self.assertIsInstance(contours, list)
        self.assertEqual(len(contours), 1)
        np.testing.assert_allclose(
            np.asarray(contours[0], dtype=np.float64),
            _GOLDEN_WORLD_POINT,
            rtol=0.0,
            atol=1e-10,
        )

    def test_profile_exporter_world_bounds_match_fixed_trs_golden(self):
        exporter = ProfileExporter(resolution=64)
        identity = np.eye(4, dtype=np.float64)
        viewport = np.array([0, 0, 64, 64], dtype=np.int32)

        _, bounds = exporter.extract_silhouette(
            self._golden_point_mesh(),
            view="top",
            translation=_GOLDEN_TRANSLATION,
            rotation=_GOLDEN_ROTATION_DEG,
            scale=_GOLDEN_SCALE,
            opengl_matrices=(identity, identity, viewport),
            viewport_image=None,
        )

        np.testing.assert_allclose(
            np.asarray(bounds["world_bounds"], dtype=np.float64),
            np.vstack([_GOLDEN_WORLD_POINT, _GOLDEN_WORLD_POINT]),
            rtol=0.0,
            atol=1e-10,
        )

    def test_legacy_bake_uses_fixed_trs_golden_and_marks_save_as_unsafe(self):
        obj = SceneObject(self._golden_point_mesh(), name="golden")
        obj.translation = _GOLDEN_TRANSLATION.copy()
        obj.rotation = _GOLDEN_ROTATION_DEG.copy()
        obj.scale = _GOLDEN_SCALE
        viewport_like = SimpleNamespace(
            update_vbo=Mock(),
            update=Mock(),
            _emit_mesh_transform_changed=Mock(),
        )

        Viewport3D.bake_object_transform(viewport_like, obj)

        np.testing.assert_allclose(
            obj.mesh.vertices,
            _GOLDEN_WORLD_POINT,
            rtol=0.0,
            atol=1e-6,
        )
        self.assertEqual(obj.mesh.vertices.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(obj.translation, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(obj.rotation, [0.0, 0.0, 0.0])
        self.assertEqual(obj.scale, 1.0)
        self.assertTrue(obj._amr_has_unpersisted_bake)
        self.assertEqual(obj._amr_alignment_status, "legacy_baked_unverifiable")

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
