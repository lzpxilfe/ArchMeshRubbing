from __future__ import annotations

import ast
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from OpenGL.GL import GL_FALSE

from src.core.artifact_scene_adapter import ArtifactProjectionSnapshot
from src.core.mesh_loader import MeshData
from src.gui.render_coordinates import RenderFrameSnapshot
from src.gui.viewport_3d import SceneObject, Viewport3D, _SurfaceLassoSelectThread


_VIEWPORT_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "gui" / "viewport_3d.py"
)


def _viewport_method_nodes() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(_VIEWPORT_SOURCE.read_text(encoding="utf-8-sig"))
    viewport = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Viewport3D"
    )
    return {
        node.name: node
        for node in viewport.body
        if isinstance(node, ast.FunctionDef)
    }


def _direct_call_names(node: ast.FunctionDef) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name):
            names.append(func.id)
        elif isinstance(func, ast.Attribute):
            names.append(func.attr)
    return names


def _fake_viewport(origin: np.ndarray) -> SimpleNamespace:
    fake = SimpleNamespace(
        _scene_render_origin_world_mm=lambda: origin.copy(),
    )
    fake._world_to_render_point = MethodType(Viewport3D._world_to_render_point, fake)
    fake._submit_world_vertex = MethodType(Viewport3D._submit_world_vertex, fake)
    return fake


def _identity_frame(
    origin: np.ndarray,
    *,
    serial: int,
    projection: np.ndarray | None = None,
) -> RenderFrameSnapshot:
    return RenderFrameSnapshot(
        frame_serial=serial,
        projection_generation=0,
        viewport=(0, 0, 100, 100),
        modelview_render=np.eye(4, dtype=np.float64),
        projection=(
            np.eye(4, dtype=np.float64)
            if projection is None
            else np.asarray(projection, dtype=np.float64)
        ),
        render_origin_world_mm=np.asarray(origin, dtype=np.float64),
    )


def _artifact_projection_snapshot() -> ArtifactProjectionSnapshot:
    return ArtifactProjectionSnapshot(
        document_id="artifact:surface-pick",
        document_schema_version="1.0.0",
        document_sha256="1" * 64,
        source_asset_id="source:surface-pick",
        geometry_revision_id="geometry:surface-pick",
        source_metadata_revision_id="metadata:surface-pick",
        align_revision_id="align:surface-pick",
        geometry_sha256="2" * 64,
        geometry_hash_scope="source_mesh",
        matrix4x4=tuple(tuple(float(value) for value in row) for row in np.eye(4)),
    )


def _surface_pick_viewport(
    *,
    xray_mode: bool = False,
    roi_enabled: bool = False,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    obj = SimpleNamespace(
        _amr_artifact_projection_snapshot=_artifact_projection_snapshot(),
        visible=True,
        translation=np.asarray(translation, dtype=np.float64),
        rotation=np.asarray(rotation, dtype=np.float64),
        scale=float(scale),
    )
    fake = SimpleNamespace(
        objects=[obj],
        selected_obj=obj,
        xray_mode=bool(xray_mode),
        roi_enabled=bool(roi_enabled),
        pick_point_on_mesh_info=Mock(),
        _world_radius_from_px_at_depth=Mock(return_value=0.00125),
    )
    return fake, obj


def _mouse_event(
    x: float,
    y: float,
    *,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    buttons: Qt.MouseButton = Qt.MouseButton.LeftButton,
) -> SimpleNamespace:
    point = SimpleNamespace(x=lambda: x, y=lambda: y)
    return SimpleNamespace(
        pos=lambda: point,
        position=lambda: point,
        modifiers=lambda: modifiers,
        buttons=lambda: buttons,
    )


def test_gpu_world_boundaries_subtract_origin_before_float_calls() -> None:
    origin = np.array(
        [1_000_000_000.0, -1_000_000_000.0, 500_000_000.0],
        dtype=np.float64,
    )
    points = origin + np.array(
        [[0.0, 0.125, 0.0], [1.0, 1.0, 3.0]],
        dtype=np.float64,
    )
    before = points.copy()
    fake = _fake_viewport(origin)
    submitted: list[tuple[float, float, float]] = []
    translated: list[tuple[float, float, float]] = []

    with (
        patch(
            "src.gui.viewport_3d.glVertex3f",
            side_effect=lambda x, y, z: submitted.append((x, y, z)),
        ),
        patch(
            "src.gui.viewport_3d.glTranslatef",
            side_effect=lambda x, y, z: translated.append((x, y, z)),
        ),
    ):
        for point in points:
            Viewport3D._submit_world_vertex(fake, point)
            Viewport3D._translate_to_world_point(fake, point)

    expected = np.array([[0.0, 0.125, 0.0], [1.0, 1.0, 3.0]])
    np.testing.assert_array_equal(np.asarray(submitted), expected)
    np.testing.assert_array_equal(np.asarray(translated), expected)
    np.testing.assert_array_equal(np.asarray(submitted, dtype=np.float32), expected)
    np.testing.assert_array_equal(np.asarray(translated, dtype=np.float32), expected)
    np.testing.assert_array_equal(points, before)
    assert float(np.max(np.abs(submitted))) <= 3.0


def test_world_projection_and_unprojection_use_explicit_snapshot_frame() -> None:
    frame_origin = np.array(
        [1_000_000_000.0, -1_000_000_000.0, 500_000_000.0],
        dtype=np.float64,
    )
    current_origin = np.array(
        [-4_000_000_000.0, 3_000_000_000.0, 2_000_000_000.0],
        dtype=np.float64,
    )
    fake = _fake_viewport(current_origin)
    point = frame_origin + [0.125, 1.0, 0.5]
    modelview = np.eye(4)
    projection = np.eye(4)
    viewport = np.array([0, 0, 100, 100], dtype=np.int32)
    frame = RenderFrameSnapshot(
        frame_serial=1,
        projection_generation=0,
        viewport=tuple(int(value) for value in viewport),
        modelview_render=modelview,
        projection=projection,
        render_origin_world_mm=frame_origin,
    )
    current_frame = RenderFrameSnapshot(
        frame_serial=2,
        projection_generation=0,
        viewport=tuple(int(value) for value in viewport),
        modelview_render=modelview,
        projection=projection,
        render_origin_world_mm=current_origin,
    )
    fake._current_render_frame_snapshot = lambda: current_frame

    window = Viewport3D._project_world_point(fake, point, frame)
    restored = Viewport3D._unproject_world_point(
        fake,
        float(window[0]),
        float(window[1]),
        float(window[2]),
        frame,
    )

    np.testing.assert_array_equal(window, [56.25, 100.0, 0.75])
    np.testing.assert_array_equal(restored, point)


def test_depth_pick_unprojects_with_the_frame_that_produced_the_depth() -> None:
    depth_origin = np.array(
        [1_000_000_000.0, -1_000_000_000.0, 500_000_000.0],
        dtype=np.float64,
    )
    later_origin = np.array(
        [-4_000_000_000.0, 3_000_000_000.0, 2_000_000_000.0],
        dtype=np.float64,
    )
    depth_frame = _identity_frame(depth_origin, serial=10)
    later_frame = _identity_frame(later_origin, serial=11)
    expected = depth_origin + [0.0, 0.0, 0.5]
    obj = SimpleNamespace(
        get_world_bounds=lambda: np.asarray(
            [expected - 1.0, expected + 1.0],
            dtype=np.float64,
        )
    )
    live_frame = {"value": depth_frame}
    call_order: list[str] = []

    def current_frame() -> RenderFrameSnapshot:
        call_order.append("frame")
        return live_frame["value"]

    def read_depth_and_advance_frame(*_args: object) -> np.ndarray:
        call_order.append("depth")
        live_frame["value"] = later_frame
        return np.asarray([[0.75]], dtype=np.float32)

    fake = SimpleNamespace(
        objects=[obj],
        selected_obj=obj,
        makeCurrent=lambda: None,
        _current_render_frame_snapshot=current_frame,
        _qt_to_gl_window_xy=lambda *_args, **_kwargs: (50, 50),
    )
    fake._unproject_world_point = MethodType(
        Viewport3D._unproject_world_point,
        fake,
    )

    with patch(
        "src.gui.viewport_3d.glReadPixels",
        side_effect=read_depth_and_advance_frame,
    ) as read_pixels:
        info = Viewport3D.pick_point_on_mesh_info(
            fake,
            50,
            49,
            allow_depth_search=False,
        )

    assert info is not None
    point, depth, gl_x, gl_y, viewport, returned_frame = info
    np.testing.assert_array_equal(point, expected)
    assert depth == 0.75
    assert (gl_x, gl_y) == (50, 50)
    np.testing.assert_array_equal(viewport, [0, 0, 100, 100])
    assert returned_frame is depth_frame
    assert call_order == ["frame", "depth"]
    read_pixels.assert_called_once()


def test_surface_anchor_observation_freezes_exact_frame_without_depth_search() -> None:
    frame = _identity_frame(
        np.array([1_000_000_000.0, -2_000_000_000.0, 500_000_000.0]),
        serial=71,
    )
    fake, obj = _surface_pick_viewport()
    depth_point = frame.render_origin_world_mm + [0.125, 0.25, 0.5]
    fake.pick_point_on_mesh_info.return_value = (
        np.asarray(depth_point, dtype=np.float64),
        0.625,
        43,
        57,
        np.asarray(frame.viewport, dtype=np.int32),
        frame,
    )
    ray_origin = frame.render_origin_world_mm + [0.125, 0.25, 10.0]
    ray_direction = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)

    with patch(
        "src.gui.viewport_3d.world_ray_from_window",
        return_value=(ray_origin, ray_direction),
    ) as world_ray:
        observation = Viewport3D.capture_surface_anchor_observation(fake, 42, 41)

    assert observation is not None
    assert observation.projection_snapshot is obj._amr_artifact_projection_snapshot
    assert observation.frame_serial == 71
    assert observation.projection_generation == frame.projection_generation
    assert observation.depth_point_world_mm == tuple(depth_point)
    assert observation.ray_origin_world_mm == tuple(ray_origin)
    assert observation.ray_direction_world == (0.0, 0.0, -1.0)
    assert observation.pixel_footprint_um == 2
    assert observation.depth_search_offset_px == (0, 0)
    fake.pick_point_on_mesh_info.assert_called_once_with(
        42,
        41,
        allow_depth_search=False,
        search_radius_px=0,
    )
    fake._world_radius_from_px_at_depth.assert_called_once_with(
        43,
        57,
        0.625,
        frame,
        1.0,
    )
    world_ray.assert_called_once_with(frame, 43, 57)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"xray_mode": True}, "X-ray"),
        ({"roi_enabled": True}, "ROI clipping"),
        ({"translation": (0.001, 0.0, 0.0)}, "translation preview"),
        ({"rotation": (0.0, 0.001, 0.0)}, "rotation preview"),
        ({"scale": 1.000001}, "scale preview"),
    ],
)
def test_surface_anchor_observation_fails_closed_before_depth_read(
    override: dict[str, object],
    message: str,
) -> None:
    fake, _obj = _surface_pick_viewport(**override)

    with pytest.raises(ValueError, match=message):
        Viewport3D.capture_surface_anchor_observation(fake, 12, 34)

    fake.pick_point_on_mesh_info.assert_not_called()
    fake._world_radius_from_px_at_depth.assert_not_called()


def test_ctrl_drag_keeps_press_frame_after_a_new_live_frame_is_published() -> None:
    press_origin = np.array(
        [1_000_000_000.0, -1_000_000_000.0, 500_000_000.0],
        dtype=np.float64,
    )
    later_origin = np.array(
        [-4_000_000_000.0, 3_000_000_000.0, 2_000_000_000.0],
        dtype=np.float64,
    )
    press_frame = _identity_frame(press_origin, serial=20)
    later_projection = np.eye(4, dtype=np.float64)
    later_projection[0, 0] = 2.0
    later_frame = _identity_frame(
        later_origin,
        serial=21,
        projection=later_projection,
    )
    live_frame = {"value": press_frame}
    obj = SimpleNamespace(
        translation=np.zeros(3, dtype=np.float64),
        world_pivot=lambda: press_origin.copy(),
    )
    transform_events: list[float] = []
    fake = SimpleNamespace(
        makeCurrent=lambda: None,
        _current_render_frame_snapshot=lambda: live_frame["value"],
        _qt_to_gl_window_xy=lambda x, y, **_kwargs: (
            int(round(float(x))),
            int(round(99.0 - float(y))),
        ),
        mouse_button=Qt.MouseButton.LeftButton,
        picking_mode="none",
        last_mouse_pos=_mouse_event(50.0, 50.0).pos(),
        last_mouse_posf=(50.0, 50.0),
        selected_obj=obj,
        gizmo_drag_start=None,
        active_gizmo_axis=None,
        roi_enabled=False,
        _ctrl_drag_active=True,
        update=lambda: None,
        _emit_mesh_transform_changed=lambda *, suspend_tape_sec: (
            transform_events.append(float(suspend_tape_sec))
        ),
    )
    fake._project_world_point = MethodType(Viewport3D._project_world_point, fake)
    fake._unproject_world_point = MethodType(
        Viewport3D._unproject_world_point,
        fake,
    )

    with patch(
        "src.gui.viewport_3d.glReadPixels",
        return_value=np.asarray([[0.5]], dtype=np.float32),
    ):
        assert Viewport3D._begin_ctrl_drag(
            fake,
            _mouse_event(50.0, 50.0),
            obj,
        )

    assert fake._cached_render_frame is press_frame
    live_frame["value"] = later_frame
    fake._amr_render_frame_snapshot = later_frame
    unproject_frames: list[RenderFrameSnapshot | None] = []

    def tracked_unproject(
        window_x: float,
        window_y: float,
        depth: float,
        frame: RenderFrameSnapshot | None = None,
    ) -> np.ndarray:
        unproject_frames.append(frame)
        return Viewport3D._unproject_world_point(
            fake,
            window_x,
            window_y,
            depth,
            frame,
        )

    fake._unproject_world_point = tracked_unproject
    Viewport3D.mouseMoveEvent(
        fake,
        _mouse_event(
            60.0,
            50.0,
            modifiers=Qt.KeyboardModifier.ControlModifier,
        ),
    )

    assert unproject_frames == [press_frame, press_frame]
    assert fake._cached_render_frame is press_frame
    np.testing.assert_allclose(
        obj.translation,
        [0.2, 0.0, 0.0],
        rtol=0.0,
        atol=1e-7,
    )
    assert transform_events == [0.24]


def test_render_frame_capture_binds_matrices_viewport_origin_and_generation() -> None:
    origin = np.array(
        [1_000_000_000.0, -1_000_000_000.0, 500_000_000.0],
        dtype=np.float64,
    )
    fake = SimpleNamespace(
        _amr_render_frame_serial=4,
        _projection_generation=9,
        _amr_render_frame_snapshot=None,
        _amr_render_frame_depth_signature=None,
        _scene_render_origin_world_mm=lambda: origin.copy(),
        objects=[],
        selected_index=-1,
    )
    fake._depth_state_signature_for_frame = MethodType(
        Viewport3D._depth_state_signature_for_frame,
        fake,
    )
    modelview = np.eye(4, dtype=np.float64)
    modelview[:3, 3] = [0.125, 1.0, -10.0]
    projection = np.eye(4, dtype=np.float64)
    raw_matrices = [modelview.T.copy(), projection.T.copy()]

    with (
        patch("src.gui.viewport_3d.glGetIntegerv", return_value=[3, 5, 800, 600]),
        patch("src.gui.viewport_3d.glGetDoublev", side_effect=raw_matrices),
    ):
        frame = Viewport3D._capture_render_frame_snapshot(fake)

    assert frame.frame_serial == 5
    assert frame.projection_generation == 9
    assert frame.viewport == (3, 5, 800, 600)
    np.testing.assert_array_equal(frame.modelview_render, modelview)
    np.testing.assert_array_equal(frame.projection, projection)
    np.testing.assert_array_equal(frame.render_origin_world_mm, origin)
    assert Viewport3D._current_render_frame_snapshot(fake) is frame

    fake._projection_generation = 10
    assert Viewport3D._current_render_frame_snapshot(fake) is None


def test_worker_camera_position_is_recovered_from_captured_modelview() -> None:
    origin = np.asarray(
        [1_000_000_000.0, -1_000_000_000.0, 500_000_000.0],
        dtype=np.float64,
    )
    modelview = np.eye(4, dtype=np.float64)
    modelview[:3, :3] = np.asarray(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    modelview[:3, 3] = [4.0, -2.0, -10.0]
    frame = RenderFrameSnapshot(
        frame_serial=6,
        projection_generation=0,
        viewport=(0, 0, 100, 100),
        modelview_render=modelview,
        projection=np.eye(4),
        render_origin_world_mm=origin,
    )

    expected = origin + np.linalg.inv(modelview)[:3, 3]

    np.testing.assert_array_equal(
        Viewport3D._camera_world_position_for_frame(frame),
        expected,
    )


def test_world_overlay_methods_cannot_bypass_render_origin_boundary() -> None:
    methods = _viewport_method_nodes()
    fully_world_vertex_methods = {
        "draw_axes",
        "draw_crosshair",
        "draw_line_section",
        "_draw_cutline_rubber_ribbon",
        "draw_cut_lines",
        "draw_roi_box",
        "draw_roi_cut_edges",
        "draw_roi_caps",
        "draw_mesh_dimensions",
        "draw_surface_paint_points",
        "_draw_single_arc",
    }
    forbidden_vertices = {"glVertex3f", "glVertex3fv"}
    for name in fully_world_vertex_methods:
        calls = set(_direct_call_names(methods[name]))
        assert not calls.intersection(forbidden_vertices), (name, calls)

    for name in {"draw_picked_points", "draw_floor_picks"}:
        calls = set(_direct_call_names(methods[name]))
        assert "glTranslatef" not in calls
        assert "glVertex3fv" not in calls
        assert "_translate_to_world_point" in calls

    gizmo_calls = set(_direct_call_names(methods["draw_rotation_gizmo"]))
    assert "glTranslatef" not in gizmo_calls
    assert "_translate_to_world_point" in gizmo_calls

    for name in {
        "draw_surface_lasso_overlay",
        "draw_surface_magnetic_lasso_overlay",
    }:
        calls = set(_direct_call_names(methods[name]))
        assert "gluProject" not in calls
        assert "_project_world_point" in calls

    paint_calls = set(_direct_call_names(methods["paintGL"]))
    assert "_draw_in_absolute_world" not in paint_calls
    first_paint_attributes = {
        target.attr
        for statement in methods["paintGL"].body[:3]
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Attribute)
    }
    assert "_amr_render_frame_snapshot" in first_paint_attributes
    assert "_cached_render_frame" not in first_paint_attributes
    paint_attribute_stores = {
        child.attr
        for child in ast.walk(methods["paintGL"])
        if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Store)
    }
    assert "_cached_render_frame" not in paint_attribute_stores

    for method in methods.values():
        calls = set(_direct_call_names(method))
        assert "gluProject" not in calls, method.name
        assert "gluUnProject" not in calls, method.name

    for callback in (
        "_on_surface_lasso_computed",
        "_on_surface_magnetic_computed",
        "_on_visible_face_select_computed",
    ):
        calls = set(_direct_call_names(methods[callback]))
        assert "_surface_worker_current_target" in calls, callback


def test_paint_publishes_depth_frame_after_mesh_and_before_overlays() -> None:
    paint = _viewport_method_nodes()["paintGL"]
    call_lines: dict[str, list[int]] = {}
    for node in ast.walk(paint):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else None
        )
        if name is not None:
            call_lines.setdefault(name, []).append(node.lineno)

    capture_lines = call_lines.get("_capture_render_frame_snapshot", [])
    assert len(capture_lines) == 1
    capture_line = capture_lines[0]
    assert max(call_lines["draw_scene_object"]) < capture_line
    for overlay in (
        "draw_axes",
        "draw_picked_points",
        "draw_surface_lasso_overlay",
        "draw_cut_lines",
        "draw_native_vector_preview",
        "draw_rotation_gizmo",
        "draw_orientation_hud",
    ):
        assert capture_line < min(call_lines[overlay]), overlay


def test_resize_invalidates_render_and_drag_frames_before_projection_change() -> None:
    fake = SimpleNamespace(
        _amr_render_frame_snapshot=object(),
        _cached_render_frame=object(),
        _apply_main_projection=lambda _w, _h: None,
    )
    with patch("src.gui.viewport_3d.glViewport") as viewport:
        Viewport3D.resizeGL(fake, 800, 600)

    assert fake._amr_render_frame_snapshot is None
    assert fake._cached_render_frame is None
    viewport.assert_called_once_with(0, 0, 800, 600)


def test_transform_change_invalidates_live_frame_but_preserves_drag_frame() -> None:
    drag_frame = object()
    emitted: list[bool] = []
    fake = SimpleNamespace(
        _amr_render_frame_snapshot=object(),
        _cached_render_frame=drag_frame,
        _cutline_tape_suspend_until=0.0,
        meshTransformChanged=SimpleNamespace(emit=lambda: emitted.append(True)),
    )

    Viewport3D._emit_mesh_transform_changed(fake, suspend_tape_sec=0.2)

    assert fake._amr_render_frame_snapshot is None
    assert fake._cached_render_frame is drag_frame
    assert emitted == [True]


def test_external_transform_signal_slot_invalidates_live_frame_only() -> None:
    drag_frame = object()
    fake = SimpleNamespace(
        _amr_render_frame_snapshot=object(),
        _cached_render_frame=drag_frame,
        _clear_cutline_tape_cache=lambda: None,
        _invalidate_surface_magnetic_cache=lambda: None,
        clear_surface_lasso=lambda: None,
        clear_surface_magnetic_lasso=lambda **_kwargs: None,
        clear_surface_paint_points=lambda: None,
        crosshair_enabled=False,
        line_section_enabled=False,
        roi_enabled=False,
        cut_lines=[[], []],
        _cut_line_final=[False, False],
    )

    Viewport3D._on_mesh_transform_changed(fake)

    assert fake._amr_render_frame_snapshot is None
    assert fake._cached_render_frame is drag_frame


def test_selection_change_invalidates_frames_and_detaches_surface_workers() -> None:
    cancelled: list[str] = []
    selected: list[int] = []
    fake = SimpleNamespace(
        objects=[object(), object()],
        selected_index=0,
        _amr_render_frame_snapshot=object(),
        _cached_render_frame=object(),
        _cached_viewport=object(),
        _cached_modelview=object(),
        _cached_projection=object(),
        _ctrl_drag_active=True,
        mouse_button=Qt.MouseButton.LeftButton,
        active_gizmo_axis="X",
        gizmo_drag_start=1.25,
        _gizmo_drag_screen_angle=0.5,
        _cancel_surface_lasso_thread=lambda: cancelled.append("area"),
        _cancel_surface_magnetic_thread=lambda: cancelled.append("magnetic"),
        _cancel_visible_face_select_thread=lambda: cancelled.append("visible"),
        _invalidate_surface_magnetic_cache=lambda: cancelled.append("cache"),
        surface_lasso_points=[np.ones(3)],
        surface_lasso_face_indices=[0],
        surface_lasso_preview=object(),
        surface_magnetic_points=[(1, 1)],
        surface_paint_points=[(np.ones(3), "outer")],
        _surface_grow_state={"old": object()},
        update=lambda: None,
        selectionChanged=SimpleNamespace(emit=lambda index: selected.append(index)),
        _active_polyline_layer_obj_index=-1,
        _active_polyline_layer_index=-1,
    )
    fake.clear_surface_lasso = MethodType(Viewport3D.clear_surface_lasso, fake)
    fake.clear_surface_magnetic_lasso = MethodType(
        Viewport3D.clear_surface_magnetic_lasso,
        fake,
    )
    fake.clear_surface_paint_points = MethodType(
        Viewport3D.clear_surface_paint_points,
        fake,
    )
    fake._reset_selection_authority_transients = MethodType(
        Viewport3D._reset_selection_authority_transients,
        fake,
    )

    Viewport3D.select_object(fake, 1)

    assert fake.selected_index == 1
    assert fake._amr_render_frame_snapshot is None
    assert fake._cached_render_frame is None
    assert fake._cached_viewport is None
    assert fake._cached_modelview is None
    assert fake._cached_projection is None
    assert not fake._ctrl_drag_active
    assert fake.mouse_button is None
    assert fake.active_gizmo_axis is None
    assert fake.gizmo_drag_start is None
    assert fake._gizmo_drag_screen_angle is None
    assert cancelled == [
        "area",
        "magnetic",
        "visible",
        "cache",
        "area",
        "magnetic",
    ]
    assert fake.surface_lasso_points == []
    assert fake.surface_lasso_face_indices == []
    assert fake.surface_lasso_preview is None
    assert fake.surface_magnetic_points == []
    assert fake.surface_paint_points == []
    assert selected == [1]


def test_adding_second_object_clears_first_object_boundary_authority() -> None:
    mesh_a = MeshData(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    mesh_b = MeshData(
        vertices=np.asarray(
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    old = SceneObject(mesh_a, "artifact A")
    fake = SimpleNamespace(
        objects=[old],
        selected_index=0,
        _amr_render_frame_snapshot=object(),
        _amr_render_frame_depth_signature=("A",),
        _cached_render_frame=object(),
        _cached_viewport=object(),
        _cached_modelview=object(),
        _cached_projection=object(),
        _ctrl_drag_active=False,
        mouse_button=Qt.MouseButton.LeftButton,
        active_gizmo_axis="Z",
        gizmo_drag_start=0.75,
        _gizmo_drag_screen_angle=0.25,
        surface_lasso_points=[np.ones(3)],
        surface_lasso_face_indices=[0],
        surface_lasso_preview=object(),
        surface_magnetic_points=[(1, 1)],
        surface_paint_points=[(np.ones(3), "outer")],
        _surface_grow_state={},
        _cancel_surface_lasso_thread=lambda: None,
        _cancel_surface_magnetic_thread=lambda: None,
        _cancel_visible_face_select_thread=lambda: None,
        _invalidate_surface_magnetic_cache=lambda: None,
        update_vbo=lambda _obj: True,
        update_grid_scale=lambda: None,
        camera=SimpleNamespace(fit_to_bounds=lambda _bounds: None, pan_offset=None),
        meshLoaded=SimpleNamespace(emit=lambda _mesh: None),
        selectionChanged=SimpleNamespace(emit=lambda _index: None),
        update=lambda: None,
    )
    fake.clear_surface_lasso = MethodType(Viewport3D.clear_surface_lasso, fake)
    fake.clear_surface_magnetic_lasso = MethodType(
        Viewport3D.clear_surface_magnetic_lasso,
        fake,
    )
    fake.clear_surface_paint_points = MethodType(
        Viewport3D.clear_surface_paint_points,
        fake,
    )
    fake._reset_selection_authority_transients = MethodType(
        Viewport3D._reset_selection_authority_transients,
        fake,
    )

    Viewport3D.add_mesh_object(
        fake,
        mesh_b,
        name="artifact B",
        center_at_origin=False,
    )

    assert fake.selected_index == 1
    assert len(fake.objects) == 2
    assert fake.surface_lasso_points == []
    assert fake.surface_lasso_face_indices == []
    assert fake.surface_lasso_preview is None
    assert fake.surface_magnetic_points == []
    assert fake.surface_paint_points == []
    assert fake.mouse_button is None
    assert fake.active_gizmo_axis is None
    assert fake.gizmo_drag_start is None


def test_surface_worker_context_rejects_other_object_transform_or_frame() -> None:
    mesh = MeshData(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    obj = SceneObject(mesh, "worker target")
    other = SceneObject(mesh, "other target")
    frame = _identity_frame(np.zeros(3), serial=30)
    fake = SimpleNamespace(
        objects=[obj, other],
        selected_index=0,
        selected_obj=obj,
        _projection_generation=0,
        _amr_render_frame_snapshot=frame,
        _amr_render_frame_depth_signature=None,
        _scene_render_origin_world_mm=lambda: np.zeros(3),
        xray_mode=False,
        solid_shell_render=True,
        roi_enabled=False,
        roi_caps_enabled=False,
        roi_bounds=(-1.0, 1.0, -1.0, 1.0),
        roi_cap_verts={},
    )
    fake._depth_state_signature_for_frame = MethodType(
        Viewport3D._depth_state_signature_for_frame,
        fake,
    )
    fake._current_render_frame_snapshot = MethodType(
        Viewport3D._current_render_frame_snapshot,
        fake,
    )
    fake._surface_magnetic_cache_signature = MethodType(
        Viewport3D._surface_magnetic_cache_signature,
        fake,
    )
    fake._amr_render_frame_depth_signature = fake._depth_state_signature_for_frame(
        frame
    )
    worker = SimpleNamespace(
        _local_to_world_matrix=obj.local_to_world_matrix().copy(),
    )
    Viewport3D._bind_surface_worker_context(fake, worker, obj, frame)

    assert Viewport3D._surface_worker_current_target(fake, worker) is obj

    fake.selected_obj = other
    assert Viewport3D._surface_worker_current_target(fake, worker) is None
    fake.selected_obj = obj

    obj.translation[0] = 1.0
    assert Viewport3D._surface_worker_current_target(fake, worker) is None
    obj.translation[0] = 0.0

    other.visible = False
    assert Viewport3D._surface_worker_current_target(fake, worker) is None
    other.visible = True

    obj._amr_geometry_draw_revision += 1
    assert Viewport3D._surface_worker_current_target(fake, worker) is None
    obj._amr_geometry_draw_revision -= 1

    changed_projection = np.eye(4, dtype=np.float64)
    changed_projection[0, 0] = 2.0
    next_frame = _identity_frame(
        np.zeros(3),
        serial=31,
        projection=changed_projection,
    )
    fake._amr_render_frame_snapshot = next_frame
    fake._amr_render_frame_depth_signature = fake._depth_state_signature_for_frame(
        next_frame
    )
    assert Viewport3D._surface_worker_current_target(fake, worker) is None


def test_magnetic_cache_signature_is_bound_to_snapshot_projection() -> None:
    mesh = MeshData(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    obj = SceneObject(mesh, "magnetic target")
    first = _identity_frame(np.zeros(3), serial=40)
    projection = np.eye(4, dtype=np.float64)
    projection[0, 0] = 0.5
    second = _identity_frame(np.zeros(3), serial=41, projection=projection)
    fake = SimpleNamespace(
        objects=[obj],
        selected_index=0,
        selected_obj=obj,
        _projection_generation=0,
        _amr_render_frame_snapshot=first,
        _amr_render_frame_depth_signature=None,
        _scene_render_origin_world_mm=lambda: np.zeros(3),
        xray_mode=False,
        solid_shell_render=True,
        roi_enabled=False,
        roi_caps_enabled=False,
        roi_bounds=(-1.0, 1.0, -1.0, 1.0),
        roi_cap_verts={},
    )
    fake._depth_state_signature_for_frame = MethodType(
        Viewport3D._depth_state_signature_for_frame,
        fake,
    )
    fake._current_render_frame_snapshot = MethodType(
        Viewport3D._current_render_frame_snapshot,
        fake,
    )
    fake._amr_render_frame_depth_signature = fake._depth_state_signature_for_frame(
        first
    )

    first_sig = Viewport3D._surface_magnetic_cache_signature(fake, first)
    fake._amr_render_frame_snapshot = second
    fake._amr_render_frame_depth_signature = fake._depth_state_signature_for_frame(
        second
    )
    second_sig = Viewport3D._surface_magnetic_cache_signature(fake, second)

    assert first_sig is not None
    assert second_sig is not None
    assert first_sig != second_sig


def test_mutable_depth_state_cannot_relabel_an_old_frame_cache() -> None:
    mesh = MeshData(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    obj = SceneObject(mesh, "depth authority")
    old_frame = _identity_frame(np.zeros(3), serial=45)
    fake = SimpleNamespace(
        objects=[obj],
        selected_index=0,
        selected_obj=obj,
        _projection_generation=0,
        _amr_render_frame_snapshot=old_frame,
        _amr_render_frame_depth_signature=None,
        _scene_render_origin_world_mm=lambda: np.zeros(3),
        xray_mode=False,
        solid_shell_render=True,
        roi_enabled=False,
        roi_caps_enabled=False,
        roi_bounds=(-1.0, 1.0, -1.0, 1.0),
        roi_cap_verts={},
    )
    fake._depth_state_signature_for_frame = MethodType(
        Viewport3D._depth_state_signature_for_frame,
        fake,
    )
    fake._current_render_frame_snapshot = MethodType(
        Viewport3D._current_render_frame_snapshot,
        fake,
    )
    fake._amr_render_frame_depth_signature = fake._depth_state_signature_for_frame(
        old_frame
    )

    assert fake._current_render_frame_snapshot() is old_frame
    fake.xray_mode = True

    assert fake._current_render_frame_snapshot() is None
    assert Viewport3D._surface_magnetic_cache_signature(fake, old_frame) is None

    new_frame = _identity_frame(np.zeros(3), serial=46)
    fake._amr_render_frame_snapshot = new_frame
    fake._amr_render_frame_depth_signature = fake._depth_state_signature_for_frame(
        new_frame
    )
    assert fake._current_render_frame_snapshot() is new_frame


def test_magnetic_cache_reads_the_snapshot_viewport_and_rejects_stale_snap() -> None:
    frame = RenderFrameSnapshot(
        frame_serial=50,
        projection_generation=0,
        viewport=(7, 9, 4, 3),
        modelview_render=np.eye(4),
        projection=np.eye(4),
        render_origin_world_mm=np.zeros(3),
    )
    obj = SimpleNamespace(mesh=object())
    fake = SimpleNamespace(
        selected_obj=obj,
        makeCurrent=lambda: None,
        _invalidate_surface_magnetic_cache=lambda: None,
        _surface_magnetic_cache_sig=("old",),
        _surface_magnetic_cache_signature=lambda: ("new",),
        _surface_magnetic_dist=np.zeros((3, 4), dtype=np.float32),
        _surface_magnetic_nn_x=np.zeros((3, 4), dtype=np.int32),
        _surface_magnetic_nn_y=np.zeros((3, 4), dtype=np.int32),
        _surface_magnetic_cache_viewport=(7, 9, 4, 3),
    )
    depth = np.asarray(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=np.float32,
    )

    with patch("src.gui.viewport_3d.glReadPixels", return_value=depth) as read:
        Viewport3D._compute_surface_magnetic_cache(fake, frame)

    read.assert_called_once()
    assert read.call_args.args[:4] == (7, 9, 4, 3)
    assert Viewport3D._surface_magnetic_snap_gl(fake, 8, 10) == (8, 10)


def test_failed_magnetic_cache_refresh_cannot_publish_stale_arrays() -> None:
    frame = _identity_frame(np.zeros(3), serial=51)
    fake = SimpleNamespace(
        _current_render_frame_snapshot=lambda: frame,
        _surface_magnetic_cache_signature=lambda _frame=None: ("new",),
        _compute_surface_magnetic_cache=lambda _frame=None: False,
        _surface_magnetic_cache_sig=("old",),
        _surface_magnetic_dist=np.zeros((2, 2), dtype=np.float32),
        _surface_magnetic_nn_x=np.zeros((2, 2), dtype=np.int32),
        _surface_magnetic_nn_y=np.zeros((2, 2), dtype=np.int32),
        _surface_magnetic_cache_viewport=(0, 0, 2, 2),
    )
    fake._invalidate_surface_magnetic_cache = MethodType(
        Viewport3D._invalidate_surface_magnetic_cache,
        fake,
    )

    assert not Viewport3D._ensure_surface_magnetic_cache(fake)
    assert fake._surface_magnetic_cache_sig is None
    assert fake._surface_magnetic_dist is None
    assert fake._surface_magnetic_nn_x is None
    assert fake._surface_magnetic_nn_y is None
    assert fake._surface_magnetic_cache_viewport is None


def test_selection_and_floor_diagnostics_never_write_depth() -> None:
    mesh = MeshData(
        vertices=np.asarray(
            [[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    obj = SceneObject(mesh, "depth-neutral diagnostics")
    obj.selected_faces = {0}
    fake = SimpleNamespace(
        picking_mode="select_face",
        brush_selected_faces=set(),
        mouse_button=None,
        camera=SimpleNamespace(elevation=0.0),
    )
    depth_masks: list[object] = []
    gl_noops = (
        "glPushAttrib",
        "glPopAttrib",
        "glPushMatrix",
        "glPopMatrix",
        "glDisable",
        "glEnable",
        "glBlendFunc",
        "glPolygonOffset",
        "glColor3f",
        "glColor4f",
        "glBegin",
        "glEnd",
        "glVertex3fv",
    )
    patches = [patch(f"src.gui.viewport_3d.{name}") for name in gl_noops]
    for item in patches:
        item.start()
    depth_patch = patch(
        "src.gui.viewport_3d.glDepthMask",
        side_effect=lambda value: depth_masks.append(value),
    )
    depth_patch.start()
    try:
        Viewport3D._draw_mesh_selection_highlights(
            fake,
            obj,
            local_vbo_origin=np.zeros(3),
            is_selected=True,
        )
        fake.picking_mode = "floor_face"
        Viewport3D._draw_floor_contact_faces(
            fake,
            obj,
            local_vbo_origin=np.zeros(3),
        )
    finally:
        depth_patch.stop()
        for item in reversed(patches):
            item.stop()

    assert depth_masks == [GL_FALSE, GL_FALSE]


def test_gizmo_and_dimensions_use_absolute_world_pivot_and_center() -> None:
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
    obj = SceneObject(mesh, "large-coordinate overlay")
    obj._amr_preview_pivot_mm = mesh.centroid.copy()
    pivot_calls: list[np.ndarray] = []
    dimension_vertices: list[tuple[float, float, float]] = []
    gizmo_fake = SimpleNamespace(
        show_gizmo=True,
        gizmo_size=2.0,
        active_gizmo_axis=None,
        _hover_axis=None,
        _translate_to_world_point=lambda point: pivot_calls.append(
            np.asarray(point, dtype=np.float64).copy()
        ),
        _draw_gizmo_circle=lambda _size: None,
    )
    dimensions_fake = SimpleNamespace(
        _mesh_center=np.zeros(3),
        _submit_world_xyz=lambda x, y, z: dimension_vertices.append((x, y, z)),
    )
    gl_noops = (
        "glDisable",
        "glEnable",
        "glPushMatrix",
        "glPopMatrix",
        "glColor3f",
        "glLineWidth",
        "glRotatef",
        "glBegin",
        "glEnd",
    )
    patches = [patch(f"src.gui.viewport_3d.{name}") for name in gl_noops]
    mocks = [item.start() for item in patches]
    try:
        Viewport3D.draw_rotation_gizmo(gizmo_fake, obj)
        Viewport3D.draw_mesh_dimensions(dimensions_fake, obj)
    finally:
        for item in reversed(patches):
            item.stop()
        del mocks

    np.testing.assert_array_equal(pivot_calls, [obj.world_pivot()])
    bounds = np.asarray(obj.get_world_bounds(), dtype=np.float64)
    center = bounds.mean(axis=0)
    assert len(dimension_vertices) == 20
    assert all(abs(x - center[0]) <= 1.5 for x, _, _ in dimension_vertices)
    assert all(abs(y - center[1]) <= 1.5 for _, y, _ in dimension_vertices)
    np.testing.assert_array_equal(
        dimensions_fake._mesh_center,
        [center[0], center[1], bounds[0, 2] + 0.1],
    )


def test_cutline_and_roi_caps_reach_gl_as_small_relative_vertices() -> None:
    origin = np.array([1_000_000_000.0, -1_000_000_000.0, 0.0])
    fake = _fake_viewport(origin)
    fake._submit_world_xyz = MethodType(Viewport3D._submit_world_xyz, fake)
    fake.cut_lines = [
        [
            origin + [0.0, 0.0, 0.0],
            origin + [0.125, 1.0, 0.0],
        ],
        [],
    ]
    fake.cut_lines_enabled = True
    fake.picking_mode = "none"
    fake._cut_line_final = [False, False]
    fake.cut_line_active = 0
    fake.cut_line_drawing = False
    fake.cut_line_preview = None
    fake.cut_section_world = [[], []]
    fake.selected_obj = None
    fake.selected_index = -1
    fake._is_mesh_transform_interacting = lambda: False
    fake.roi_cap_verts = {
        "x1": np.asarray(
            [
                origin + [0.0, 0.0, 0.0],
                origin + [0.125, 0.0, 1.0],
                origin + [0.0, 1.0, 0.0],
            ]
        )
    }
    submitted: list[tuple[float, float, float]] = []
    gl_noops = (
        "glDisable",
        "glEnable",
        "glColor4f",
        "glLineWidth",
        "glPointSize",
        "glBegin",
        "glEnd",
        "glBlendFunc",
        "glPolygonOffset",
    )
    patches = [patch(f"src.gui.viewport_3d.{name}") for name in gl_noops]
    for item in patches:
        item.start()
    vertex_patch = patch(
        "src.gui.viewport_3d.glVertex3f",
        side_effect=lambda x, y, z: submitted.append((x, y, z)),
    )
    vertex_patch.start()
    try:
        Viewport3D.draw_cut_lines(fake)
        Viewport3D.draw_roi_caps(fake)
    finally:
        vertex_patch.stop()
        for item in reversed(patches):
            item.stop()

    expected = np.asarray(
        [
            [0.0, 0.0, 0.08],
            [0.125, 1.0, 0.08],
            [0.0, 0.0, 0.0],
            [0.125, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(np.asarray(submitted, dtype=np.float32), expected)
    assert float(np.max(np.abs(submitted))) <= 1.0


def test_lasso_worker_captures_authoritative_pivoted_object_matrix() -> None:
    base = np.array([1_000_000_000.0, -1_000_000_000.0, 0.0])
    mesh = MeshData(
        vertices=base
        + np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
    )
    obj = SceneObject(mesh, "pivoted lasso")
    obj._amr_preview_pivot_mm = mesh.centroid.copy()
    obj.translation = np.array([3.0, -2.0, 1.0])
    obj.rotation = np.array([10.0, 20.0, 30.0])
    obj.scale = 1.25
    matrix = obj.local_to_world_matrix()

    worker = _SurfaceLassoSelectThread(
        mesh.vertices,
        mesh.faces,
        matrix,
        np.array([0.0, 0.0, 10.0]),
        np.eye(4),
        np.eye(4),
        np.array([0, 0, 100, 100]),
        np.array([[0.0, 0.0], [99.0, 0.0], [99.0, 99.0]]),
        (0, 0, 99, 99),
        (0, 0),
        None,
        render_origin_world_mm=base,
    )

    np.testing.assert_array_equal(worker._local_to_world_matrix, matrix)
    np.testing.assert_array_equal(worker._render_origin_world_mm, base)


def test_area_and_magnetic_lasso_project_ndarray_results_into_hud() -> None:
    origin = np.array([1_000_000_000.0, -1_000_000_000.0, 0.0])
    frame = RenderFrameSnapshot(
        frame_serial=1,
        projection_generation=0,
        viewport=(0, 0, 100, 100),
        modelview_render=np.eye(4),
        projection=np.eye(4),
        render_origin_world_mm=origin,
    )
    fake = SimpleNamespace(
        picking_mode="paint_surface_area",
        surface_lasso_points=[origin.copy()],
        surface_lasso_preview=None,
        makeCurrent=lambda: None,
        _current_render_frame_snapshot=lambda: frame,
        width=lambda: 100,
        height=lambda: 100,
        _surface_area_close_snap_px=12,
        _surface_magnetic_close_snap_px=12,
        _surface_magnetic_snap_radius_px=14,
        _surface_magnetic_cursor_qt=None,
    )
    fake._project_world_point = MethodType(Viewport3D._project_world_point, fake)
    submitted: list[tuple[float, float, float]] = []
    gl_noops = (
        "glDisable",
        "glEnable",
        "glBlendFunc",
        "glMatrixMode",
        "glPushMatrix",
        "glPopMatrix",
        "glLoadIdentity",
        "glOrtho",
        "glLineWidth",
        "glColor4f",
        "glBegin",
        "glEnd",
        "glPointSize",
    )
    patches = [patch(f"src.gui.viewport_3d.{name}") for name in gl_noops]
    for item in patches:
        item.start()
    vertex_patch = patch(
        "src.gui.viewport_3d.glVertex3f",
        side_effect=lambda x, y, z: submitted.append((x, y, z)),
    )
    vertex_patch.start()
    try:
        Viewport3D.draw_surface_lasso_overlay(fake)
        fake.picking_mode = "paint_surface_magnetic"
        Viewport3D.draw_surface_magnetic_lasso_overlay(fake)
    finally:
        vertex_patch.stop()
        for item in reversed(patches):
            item.stop()

    assert submitted
    assert (50.0, 49.0, 0.0) in submitted
    assert sum(point == (50.0, 49.0, 0.0) for point in submitted) >= 2
