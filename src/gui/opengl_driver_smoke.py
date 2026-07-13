"""Real-driver precision smoke for the native QOpenGLWidget viewport.

This module is deliberately a standalone process, not an ordinary pytest.
Qt's ``offscreen`` platform cannot create the QOpenGLWidget context used by
the application.  Run it under a native window platform (for example
``cocoa`` on macOS or ``xcb`` inside Xvfb on Linux).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray


REPORT_SCHEMA = "archmeshrubbing.opengl_driver_smoke"
REPORT_SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
PROBE_WIDGET_SIZE = (768, 768)
PROBE_BASE_WORLD_MM = np.array(
    [1_000_000_000.0, -2_000_000_000.0, 3_000_000_000.0],
    dtype=np.float64,
)
PROBE_OFFSETS_MM = np.array(
    [
        [-0.500, -0.375, 0.000],
        [-0.125, -0.375, 0.000],
        [-0.125, 0.375, 0.000],
        [-0.500, 0.375, 0.000],
        [0.125, -0.375, 0.125],
        [0.500, -0.375, 0.125],
        [0.500, 0.375, 0.125],
        [0.125, 0.375, 0.125],
    ],
    dtype=np.float64,
)
PROBE_FACES = np.array(
    [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
    dtype=np.int32,
)
PROBE_STEP_HEIGHT_MM = 0.125
PROBE_GAP_WIDTH_MM = 0.250

for _constant_array in (PROBE_BASE_WORLD_MM, PROBE_OFFSETS_MM, PROBE_FACES):
    _constant_array.setflags(write=False)


class DriverSmokeFailure(RuntimeError):
    """A fail-closed actual-context or render precision failure."""


class _CheckRecorder:
    def __init__(self, report: dict[str, Any]) -> None:
        self._checks = report.setdefault("checks", [])

    def require(
        self,
        check_id: str,
        condition: object,
        detail: dict[str, Any] | None = None,
    ) -> None:
        ok = bool(condition)
        entry: dict[str, Any] = {"id": str(check_id), "ok": ok}
        if detail:
            entry.update(_json_value(detail))
        self._checks.append(entry)
        # Native driver failures can terminate the process below Python's
        # exception boundary.  Emit each completed checkpoint immediately so
        # CI still records the last proven boundary before a hard crash.
        try:
            print(
                "OPENGL_SMOKE_CHECK "
                + json.dumps(entry, allow_nan=False, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass
        if not ok:
            raise DriverSmokeFailure(f"actual OpenGL check failed: {check_id}")


def probe_geometry() -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Return a fresh >=1e9 mm mesh with a 0.125 mm height step."""

    vertices = PROBE_BASE_WORLD_MM.reshape(1, 3) + PROBE_OFFSETS_MM
    return np.asarray(vertices, dtype=np.float64), PROBE_FACES.copy()


def connected_component_sizes(mask: object) -> list[int]:
    """Return four-neighbour component sizes for a small boolean image."""

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("component mask must be a 2D array")
    height, width = array.shape
    visited = np.zeros_like(array, dtype=bool)
    sizes: list[int] = []
    for y0, x0 in np.argwhere(array):
        y = int(y0)
        x = int(x0)
        if visited[y, x]:
            continue
        visited[y, x] = True
        stack = [(y, x)]
        size = 0
        while stack:
            cy, cx = stack.pop()
            size += 1
            if cy > 0 and array[cy - 1, cx] and not visited[cy - 1, cx]:
                visited[cy - 1, cx] = True
                stack.append((cy - 1, cx))
            if cy + 1 < height and array[cy + 1, cx] and not visited[cy + 1, cx]:
                visited[cy + 1, cx] = True
                stack.append((cy + 1, cx))
            if cx > 0 and array[cy, cx - 1] and not visited[cy, cx - 1]:
                visited[cy, cx - 1] = True
                stack.append((cy, cx - 1))
            if cx + 1 < width and array[cy, cx + 1] and not visited[cy, cx + 1]:
                visited[cy, cx + 1] = True
                stack.append((cy, cx + 1))
        sizes.append(size)
    return sorted(sizes, reverse=True)


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically create one non-overwriting JSON evidence report."""

    payload = json.dumps(
        _json_value(report),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"report already exists: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-filesystem hard link is an atomic no-replace publication:
        # concurrent writers cannot overwrite whichever report wins first.
        os.link(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _source_provenance() -> dict[str, Any]:
    commit = str(os.environ.get("GITHUB_SHA") or "").strip().lower()
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit
    ):
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip().lower()
        except (OSError, subprocess.SubprocessError):
            commit = "unknown"

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        source_tree = "dirty" if status.strip() else "clean"
    except (OSError, subprocess.SubprocessError):
        source_tree = "unknown"

    lock_path = ROOT / "requirements" / "runtime-py312.lock"
    try:
        lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    except OSError:
        lock_sha256 = "unknown"

    distributions: dict[str, str] = {}
    for name in ("numpy", "PyOpenGL", "PyQt6", "PyQt6-Qt6"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = "missing"

    github: dict[str, str] = {}
    for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_WORKFLOW"):
        value = str(os.environ.get(key) or "").strip()
        if value:
            github[key.removeprefix("GITHUB_").lower()] = value

    return {
        "commit": commit,
        "source_tree": source_tree,
        "runtime_lock_sha256": lock_sha256,
        "distributions": distributions,
        "github": github,
    }


def _gl_array(raw: object, dtype: np.dtype[Any], expected: int) -> np.ndarray:
    if isinstance(raw, (bytes, bytearray, memoryview)):
        array = np.frombuffer(raw, dtype=dtype)
    else:
        source = np.asarray(raw)
        if source.dtype == dtype:
            array = source.reshape(-1)
        elif int(source.nbytes) == int(expected) * int(dtype.itemsize):
            # glGetBufferSubData may expose an untyped uint8 ndarray even
            # though the buffer contains packed float32 values.
            array = np.frombuffer(np.ascontiguousarray(source).tobytes(), dtype=dtype)
        else:
            array = np.asarray(raw, dtype=dtype).reshape(-1)
    if int(array.size) != int(expected):
        raise DriverSmokeFailure(
            f"OpenGL read returned {array.size} values; expected {expected}"
        )
    return array.copy()


def _gl_string(gl: Any, name: object) -> str:
    value = gl.glGetString(name)
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _pump_events(app: Any, *, attempts: int = 1, delay_s: float = 0.0) -> None:
    for _ in range(max(1, int(attempts))):
        app.processEvents()
        if delay_s > 0.0:
            time.sleep(float(delay_s))


def _wait_for_context(app: Any, viewport: Any, timeout_s: float) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        app.processEvents()
        context = viewport.context()
        if bool(viewport.isValid()) and context is not None and bool(context.isValid()):
            return
        time.sleep(0.01)
    raise DriverSmokeFailure(
        "QOpenGLWidget did not create a valid native context before timeout"
    )


def _probe_qt_fbo(viewport: Any, gl: Any, recorder: _CheckRecorder) -> dict[str, Any]:
    from PyQt6.QtOpenGL import QOpenGLFramebufferObject

    fbo = None
    viewport.makeCurrent()
    try:
        fbo = QOpenGLFramebufferObject(
            64,
            64,
            QOpenGLFramebufferObject.Attachment.Depth,
        )
        recorder.require("qt_fbo.valid", fbo.isValid())
        recorder.require("qt_fbo.bound", fbo.bind())
        status = int(gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER))
        recorder.require(
            "qt_fbo.complete",
            status == int(gl.GL_FRAMEBUFFER_COMPLETE),
            {"status": status},
        )
        gl.glViewport(0, 0, 64, 64)
        gl.glDisable(gl.GL_DITHER)
        gl.glClearColor(32.0 / 255.0, 64.0 / 255.0, 128.0 / 255.0, 1.0)
        gl.glClearDepth(0.375)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glClear(int(gl.GL_COLOR_BUFFER_BIT) | int(gl.GL_DEPTH_BUFFER_BIT))
        gl.glFinish()
        color = _gl_array(
            gl.glReadPixels(32, 32, 1, 1, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE),
            np.dtype(np.uint8),
            4,
        )
        depth = float(
            _gl_array(
                gl.glReadPixels(
                    32,
                    32,
                    1,
                    1,
                    gl.GL_DEPTH_COMPONENT,
                    gl.GL_FLOAT,
                ),
                np.dtype(np.float32),
                1,
            )[0]
        )
        recorder.require(
            "qt_fbo.color_readback",
            bool(np.array_equal(color, np.array([32, 64, 128, 255], dtype=np.uint8))),
            {"rgba": color},
        )
        recorder.require(
            "qt_fbo.depth_readback",
            abs(depth - 0.375) <= 1e-6,
            {"depth": depth},
        )
        return {"rgba": color.tolist(), "depth": depth, "status": status}
    finally:
        if fbo is not None:
            try:
                fbo.release()
            except Exception:
                pass
        gl.glClearDepth(1.0)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        viewport.doneCurrent()


def _read_vbo_payload(
    viewport: Any,
    obj: Any,
    gl: Any,
    recorder: _CheckRecorder,
) -> dict[str, Any]:
    vertex_count = int(obj.vertex_count)
    vbo_id = int(obj.vbo_id or 0)
    recorder.require("scene.vbo_id", vbo_id > 0, {"vbo_id": vbo_id})
    viewport.makeCurrent()
    try:
        recorder.require("scene.vbo_driver_object", gl.glIsBuffer(vbo_id))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_id)
        byte_count = vertex_count * 6 * np.dtype(np.float32).itemsize
        raw = gl.glGetBufferSubData(gl.GL_ARRAY_BUFFER, 0, byte_count)
        payload = _gl_array(raw, np.dtype(np.float32), vertex_count * 6).reshape(
            vertex_count,
            6,
        )
    finally:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        viewport.doneCurrent()

    source_vertices = np.asarray(obj.mesh.vertices, dtype=np.float64)
    source_faces = np.asarray(obj.mesh.faces, dtype=np.int32)
    origin = np.asarray(obj._amr_vbo_origin_local_mm, dtype=np.float64)
    expected = (source_vertices - origin.reshape(1, 3))[
        source_faces.reshape(-1)
    ]
    error = float(
        np.max(np.abs(payload[:, :3].astype(np.float64) - expected))
    )
    recorder.require(
        "scene.vbo_relative_payload",
        error <= 1e-6 and float(np.max(np.abs(payload[:, :3]))) <= 1.0,
        {
            "max_error_mm": error,
            "max_abs_position_mm": float(np.max(np.abs(payload[:, :3]))),
        },
    )
    return {
        "id": vbo_id,
        "vertex_count": vertex_count,
        "origin_world_mm": origin,
        "position_min_mm": payload[:, :3].min(axis=0),
        "position_max_mm": payload[:, :3].max(axis=0),
        "max_payload_error_mm": error,
    }


def _projected_pixel(frame: Any, point_world: np.ndarray) -> tuple[int, int]:
    from src.gui.render_coordinates import project_world_to_window

    window = project_world_to_window(frame, point_world)
    vx, vy, width, height = (int(value) for value in frame.viewport)
    x = int(np.clip(int(round(float(window[0]))), vx, vx + width - 1))
    y = int(np.clip(int(round(float(window[1]))), vy, vy + height - 1))
    return x, y


def _pixel_from_image(
    image: np.ndarray,
    frame: Any,
    point: tuple[int, int],
) -> np.ndarray:
    vx, vy, _width, _height = (int(value) for value in frame.viewport)
    return np.asarray(image[int(point[1] - vy), int(point[0] - vx)])


def _green_near(
    color: np.ndarray,
    frame: Any,
    point: tuple[int, int],
    radius: int = 4,
) -> bool:
    vx, vy, width, height = (int(value) for value in frame.viewport)
    x = int(point[0] - vx)
    y = int(point[1] - vy)
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    patch = color[y0:y1, x0:x1, :3].astype(np.int16)
    if patch.size == 0:
        return False
    red = patch[:, :, 0]
    green = patch[:, :, 1]
    blue = patch[:, :, 2]
    return bool(np.any((green >= 120) & (green > red * 3 // 2) & (green > blue * 3 // 2)))


def _render_mode(
    *,
    app: Any,
    viewport: Any,
    gl: Any,
    recorder: _CheckRecorder,
    mode: str,
) -> dict[str, Any]:
    from src.gui.render_coordinates import (
        unproject_window_to_world,
        world_ray_from_window,
    )

    previous_frame = viewport._current_render_frame_snapshot()
    previous_frame_serial = (
        int(previous_frame.frame_serial) if previous_frame is not None else -1
    )
    # Camera fields are set directly by this diagnostic.  Revoke the old frame
    # first so a failed/no-op repaint cannot relabel previous pixels as the new
    # projection mode.
    viewport._amr_render_frame_snapshot = None
    viewport._amr_render_frame_depth_signature = None

    if mode == "perspective":
        viewport._canonical_view_key = None
        viewport._front_back_ortho_enabled = False
        viewport._ortho_frame_override = None
        viewport.camera.azimuth = 90.0
        viewport.camera.elevation = 60.0
        tolerance_mm = 0.02
    elif mode == "top_orthographic":
        viewport._canonical_view_key = "top"
        viewport._front_back_ortho_enabled = True
        viewport._ortho_frame_override = (0.75, 0.75)
        viewport.camera.azimuth = 0.0
        viewport.camera.elevation = 90.0
        tolerance_mm = 0.002
    else:
        raise ValueError(f"unknown render mode: {mode}")

    viewport.camera.center = PROBE_BASE_WORLD_MM + np.array(
        [0.0, 0.0, PROBE_STEP_HEIGHT_MM * 0.5],
        dtype=np.float64,
    )
    viewport.camera.pan_offset = np.zeros(3, dtype=np.float64)
    viewport.camera.distance = 8.0
    viewport.update()
    viewport.repaint()
    _pump_events(app, attempts=8, delay_s=0.005)

    frame = viewport._current_render_frame_snapshot()
    recorder.require(
        f"{mode}.fresh_frame_published",
        frame is not None
        and int(frame.frame_serial) > int(previous_frame_serial),
        {
            "previous_frame_serial": previous_frame_serial,
            "frame_serial": int(frame.frame_serial) if frame is not None else None,
        },
    )
    if frame is None:
        raise DriverSmokeFailure("render frame was not published")

    projection = np.asarray(frame.projection, dtype=np.float64)
    is_orthographic = bool(
        abs(float(projection[3, 3]) - 1.0) <= 1e-12
        and abs(float(projection[3, 2])) <= 1e-12
    )
    expected_orthographic = mode == "top_orthographic"
    recorder.require(
        f"{mode}.projection_kind",
        is_orthographic == expected_orthographic,
        {
            "expected": "orthographic" if expected_orthographic else "perspective",
            "projection_3_2": float(projection[3, 2]),
            "projection_3_3": float(projection[3, 3]),
        },
    )

    vx, vy, width, height = (int(value) for value in frame.viewport)
    recorder.require(
        f"{mode}.viewport_positive",
        width > 0 and height > 0,
        {"viewport": frame.viewport},
    )

    viewport.makeCurrent()
    try:
        actual_viewport = tuple(
            int(value) for value in np.asarray(gl.glGetIntegerv(gl.GL_VIEWPORT)).reshape(-1)[:4]
        )
        depth = _gl_array(
            gl.glReadPixels(
                vx,
                vy,
                width,
                height,
                gl.GL_DEPTH_COMPONENT,
                gl.GL_FLOAT,
            ),
            np.dtype(np.float32),
            width * height,
        ).reshape(height, width)
        color = _gl_array(
            gl.glReadPixels(
                vx,
                vy,
                width,
                height,
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
            ),
            np.dtype(np.uint8),
            width * height * 4,
        ).reshape(height, width, 4)
        gl_error = int(gl.glGetError())
    finally:
        viewport.doneCurrent()

    recorder.require(
        f"{mode}.snapshot_viewport_matches_driver",
        actual_viewport == tuple(frame.viewport),
        {"driver": actual_viewport, "snapshot": frame.viewport},
    )
    recorder.require(
        f"{mode}.gl_error",
        gl_error == int(gl.GL_NO_ERROR),
        {"gl_error": gl_error},
    )

    depth_mask = np.isfinite(depth) & (depth < np.float32(1.0 - 1e-6))
    component_sizes = connected_component_sizes(depth_mask)
    recorder.require(
        f"{mode}.two_visual_components",
        len(component_sizes) == 2 and min(component_sizes) >= 16,
        {"component_sizes_px": component_sizes},
    )

    sample_world = (
        PROBE_BASE_WORLD_MM + np.array([-0.3125, 0.0, 0.0]),
        PROBE_BASE_WORLD_MM + np.array([0.3125, 0.0, 0.125]),
    )
    plane_z = (
        float(PROBE_BASE_WORLD_MM[2]),
        float(PROBE_BASE_WORLD_MM[2] + PROBE_STEP_HEIGHT_MM),
    )
    sample_pixels = tuple(_projected_pixel(frame, point) for point in sample_world)
    sample_depths = tuple(
        float(_pixel_from_image(depth, frame, point)) for point in sample_pixels
    )
    sample_colors = tuple(
        _pixel_from_image(color, frame, point).astype(np.uint8)
        for point in sample_pixels
    )
    recorder.require(
        f"{mode}.plate_depth_pixels",
        all(np.isfinite(value) and 0.0 < value < 1.0 for value in sample_depths),
        {"depths": sample_depths, "pixels": sample_pixels},
    )
    recorder.require(
        f"{mode}.plate_color_pixels",
        all(int(np.min(value[:3])) >= 120 for value in sample_colors),
        {"rgba": [value.tolist() for value in sample_colors]},
    )

    gap_world = PROBE_BASE_WORLD_MM + np.array([0.0, 0.300, 0.0625])
    gap_pixel = _projected_pixel(frame, gap_world)
    gap_depth = float(_pixel_from_image(depth, frame, gap_pixel))
    gap_color = _pixel_from_image(color, frame, gap_pixel).astype(np.uint8)
    recorder.require(
        f"{mode}.gap_background_depth",
        np.isfinite(gap_depth) and gap_depth >= 1.0 - 1e-6,
        {"depth": gap_depth, "pixel": gap_pixel},
    )
    recorder.require(
        f"{mode}.gap_background_color",
        int(np.max(gap_color[:3])) <= 5,
        {"rgba": gap_color.tolist()},
    )

    overlay_world = PROBE_BASE_WORLD_MM + np.array([0.0, 0.0, 0.250])
    overlay_pixel = _projected_pixel(frame, overlay_world)
    recorder.require(
        f"{mode}.relative_overlay_pixel",
        _green_near(color, frame, overlay_pixel),
        {"pixel": overlay_pixel},
    )

    picked_points: list[np.ndarray] = []
    restored_points: list[np.ndarray] = []
    pick_errors: list[float] = []
    for index, (pixel, expected_z) in enumerate(zip(sample_pixels, plane_z, strict=True)):
        gl_x, gl_y = pixel
        scale_x = float(width) / float(max(1, viewport.width()))
        scale_y = float(height) / float(max(1, viewport.height()))
        qt_x = (float(gl_x) - float(vx)) / scale_x
        qt_y = (float(vy + height - 1) - float(gl_y)) / scale_y
        pick = viewport.pick_point_on_mesh_info(
            int(round(qt_x)),
            int(round(qt_y)),
            allow_depth_search=False,
        )
        recorder.require(f"{mode}.pick_{index}_present", pick is not None)
        if pick is None:
            raise DriverSmokeFailure("mesh pick unexpectedly returned None")
        picked = np.asarray(pick[0], dtype=np.float64).reshape(3)
        pick_gl_x = int(pick[2])
        pick_gl_y = int(pick[3])
        pick_frame = pick[5]
        recorder.require(
            f"{mode}.pick_{index}_same_frame",
            int(pick_frame.frame_serial) == int(frame.frame_serial),
            {
                "pick_frame_serial": int(pick_frame.frame_serial),
                "render_frame_serial": int(frame.frame_serial),
            },
        )
        restored = unproject_window_to_world(
            frame,
            [float(gl_x), float(gl_y), sample_depths[index]],
        )
        ray_origin, ray_direction = world_ray_from_window(
            frame,
            float(pick_gl_x),
            float(pick_gl_y),
        )
        denom = float(ray_direction[2])
        recorder.require(
            f"{mode}.pick_{index}_ray_not_parallel",
            abs(denom) > 1e-12,
            {"ray_z": denom},
        )
        ray_t = (float(expected_z) - float(ray_origin[2])) / denom
        oracle = ray_origin + ray_t * ray_direction
        error = float(np.linalg.norm(picked - oracle))
        z_error = abs(float(picked[2]) - float(expected_z))
        recorder.require(
            f"{mode}.pick_{index}_plane_accuracy",
            error <= tolerance_mm and z_error <= tolerance_mm,
            {
                "error_mm": error,
                "z_error_mm": z_error,
                "tolerance_mm": tolerance_mm,
            },
        )
        picked_points.append(picked)
        restored_points.append(np.asarray(restored, dtype=np.float64))
        pick_errors.append(error)

    pick_delta = float(picked_points[1][2] - picked_points[0][2])
    restored_delta = float(restored_points[1][2] - restored_points[0][2])
    delta_tolerance = 0.004 if mode == "perspective" else 0.002
    recorder.require(
        f"{mode}.submillimeter_depth_delta",
        abs(pick_delta - PROBE_STEP_HEIGHT_MM) <= delta_tolerance
        and abs(restored_delta - PROBE_STEP_HEIGHT_MM) <= delta_tolerance,
        {
            "expected_mm": PROBE_STEP_HEIGHT_MM,
            "pick_delta_mm": pick_delta,
            "restored_delta_mm": restored_delta,
            "tolerance_mm": delta_tolerance,
        },
    )

    return {
        "mode": mode,
        "projection_kind": "orthographic" if is_orthographic else "perspective",
        "frame_serial": int(frame.frame_serial),
        "viewport": frame.viewport,
        "render_origin_world_mm": frame.render_origin_world_mm,
        "component_sizes_px": component_sizes,
        "sample_pixels": sample_pixels,
        "sample_depths": sample_depths,
        "sample_rgba": [value.tolist() for value in sample_colors],
        "gap_pixel": gap_pixel,
        "overlay_pixel": overlay_pixel,
        "picked_world_mm": picked_points,
        "restored_world_mm": restored_points,
        "max_pick_error_mm": max(pick_errors),
        "pick_height_delta_mm": pick_delta,
        "restored_height_delta_mm": restored_delta,
    }


def run_driver_smoke(
    report: dict[str, Any],
    *,
    qt_platform: str | None = None,
    context_timeout_s: float = 8.0,
) -> None:
    """Run the actual native widget/VBO/FBO/depth pipeline or fail closed."""

    if qt_platform:
        os.environ["QT_QPA_PLATFORM"] = str(qt_platform)

    from OpenGL import GL
    from PyQt6.QtCore import QCoreApplication, QEvent, Qt
    from PyQt6.QtGui import QGuiApplication, QOpenGLContext
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    from PyQt6.QtWidgets import QApplication

    from src.core.mesh_loader import MeshData
    from src.gui.opengl_context import (
        OPENGL_MINIMUM_DEPTH_BITS,
        OPENGL_MINIMUM_VERSION,
        install_compatibility_surface_format,
    )
    from src.gui.viewport_3d import Viewport3D

    class _PrecisionSmokeViewport(Viewport3D):
        def draw_ground_plane(self) -> None:
            return

        def draw_grid(self) -> None:
            return

        def draw_axes(self) -> None:
            return

        def draw_rotation_gizmo(self, _obj: object) -> None:
            return

        def draw_mesh_dimensions(self, _obj: object) -> None:
            return

        def draw_orientation_hud(self) -> None:
            return

        def draw_surface_runtime_hud(self) -> None:
            return

    recorder = _CheckRecorder(report)
    viewport = None
    app = None
    caught: Exception | None = None
    cleanup_errors: list[str] = []
    requested_format = None
    try:
        recorder.require(
            "process.fresh_qapplication",
            QGuiApplication.instance() is None,
        )
        requested_format = install_compatibility_surface_format()
        app_args = ["archmeshrubbing-opengl-driver-smoke"]
        if qt_platform:
            app_args.extend(["-platform", str(qt_platform)])
        app = QApplication(app_args)
        actual_platform = str(QGuiApplication.platformName() or "unknown")
        recorder.require(
            "qt.native_platform",
            actual_platform not in {"offscreen", "minimal", "unknown"}
            and (not qt_platform or actual_platform == str(qt_platform)),
            {"actual": actual_platform, "requested": qt_platform},
        )

        frame_swaps: list[float] = []
        viewport = _PrecisionSmokeViewport()
        recorder.require(
            "qt.depth_preserving_update_behavior",
            viewport.updateBehavior()
            == QOpenGLWidget.UpdateBehavior.PartialUpdate,
            {"update_behavior": viewport.updateBehavior().name},
        )
        viewport.frameSwapped.connect(lambda: frame_swaps.append(time.monotonic()))
        viewport.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        viewport.resize(*PROBE_WIDGET_SIZE)
        viewport.show()
        _wait_for_context(app, viewport, context_timeout_s)
        context = viewport.context()
        recorder.require(
            "qt.widget_context_valid",
            bool(viewport.isValid()) and context is not None and context.isValid(),
        )
        if context is None:
            raise DriverSmokeFailure("QOpenGLWidget context is missing")

        viewport.makeCurrent()
        try:
            recorder.require(
                "qt.widget_context_current",
                QOpenGLContext.currentContext() == context,
            )
            vendor = _gl_string(GL, GL.GL_VENDOR)
            renderer = _gl_string(GL, GL.GL_RENDERER)
            version = _gl_string(GL, GL.GL_VERSION)
            recorder.require(
                "driver.identity",
                bool(vendor and renderer and version),
                {"vendor": vendor, "renderer": renderer, "version": version},
            )
            actual_format = context.format()
            actual_version = (
                int(actual_format.majorVersion()),
                int(actual_format.minorVersion()),
            )
            recorder.require(
                "driver.compatibility_version",
                actual_version >= OPENGL_MINIMUM_VERSION
                and actual_format.renderableType()
                == requested_format.renderableType(),
                {
                    "actual_version": actual_version,
                    "actual_profile": actual_format.profile().name,
                    "renderable_type": actual_format.renderableType().name,
                },
            )
            depth_bits = int(
                np.asarray(GL.glGetIntegerv(GL.GL_DEPTH_BITS)).reshape(-1)[0]
            )
            recorder.require(
                "driver.depth_bits",
                depth_bits >= OPENGL_MINIMUM_DEPTH_BITS,
                {"depth_bits": depth_bits},
            )
            default_fbo = int(viewport.defaultFramebufferObject())
            bound_fbo = int(
                np.asarray(GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)).reshape(-1)[0]
            )
            default_status_raw = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            default_status = (
                int(default_status_raw) if default_status_raw is not None else -1
            )
            recorder.require(
                "qt.widget_default_fbo",
                default_fbo > 0
                and bound_fbo == default_fbo
                and default_status == int(GL.GL_FRAMEBUFFER_COMPLETE),
                {
                    "default_fbo": default_fbo,
                    "bound_fbo": bound_fbo,
                    "status": default_status,
                },
            )
        finally:
            viewport.doneCurrent()

        report["context"] = {
            "qt_platform": actual_platform,
            "qt_requested_platform": qt_platform,
            "vendor": vendor,
            "renderer": renderer,
            "version": version,
            "surface_version": actual_version,
            "surface_profile": actual_format.profile().name,
            "depth_bits": depth_bits,
            "default_fbo": default_fbo,
            "device_pixel_ratio": float(viewport.devicePixelRatioF()),
            "software_renderer": any(
                token in renderer.casefold()
                for token in ("llvmpipe", "softpipe", "software", "swiftshader")
            ),
        }
        report["qt_fbo_probe"] = _probe_qt_fbo(viewport, GL, recorder)

        source_vertices, faces = probe_geometry()
        source_snapshot = source_vertices.copy()
        mesh = MeshData(vertices=source_vertices.copy(), faces=faces.copy(), unit="mm")
        viewport.add_mesh_object(
            mesh,
            "large-coordinate-submillimeter-step",
            center_at_origin=False,
        )
        recorder.require(
            "scene.source_vertices_unchanged_after_upload",
            np.array_equal(mesh.vertices, source_snapshot),
        )
        recorder.require(
            "scene.absolute_coordinate_scale",
            float(np.max(np.abs(mesh.vertices))) >= 1_000_000_000.0,
            {"max_abs_world_mm": float(np.max(np.abs(mesh.vertices)))},
        )
        obj = viewport.selected_obj
        recorder.require("scene.selected_object", obj is not None)
        if obj is None:
            raise DriverSmokeFailure("probe mesh was not selected")
        report["vbo"] = _read_vbo_payload(viewport, obj, GL, recorder)

        viewport.flat_shading = True
        viewport.show_gizmo = False
        viewport.surface_runtime_hud_enabled = False
        viewport.show_surface_assignment_overlay = False
        viewport.floor_penetration_highlight = False
        viewport.native_vector_preview_world = [
            (
                PROBE_BASE_WORLD_MM.reshape(1, 3)
                + np.array(
                    [[0.0, -0.125, 0.250], [0.0, 0.125, 0.250]],
                    dtype=np.float64,
                ),
                False,
            )
        ]
        viewport.makeCurrent()
        try:
            GL.glClearDepth(1.0)
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        finally:
            viewport.doneCurrent()

        report["scene"] = {
            "base_world_mm": PROBE_BASE_WORLD_MM,
            "max_abs_world_mm": float(np.max(np.abs(mesh.vertices))),
            "step_height_mm": PROBE_STEP_HEIGHT_MM,
            "gap_width_mm": PROBE_GAP_WIDTH_MM,
            "overlay_length_mm": 0.250,
        }
        report["render_modes"] = [
            _render_mode(
                app=app,
                viewport=viewport,
                gl=GL,
                recorder=recorder,
                mode="perspective",
            ),
            _render_mode(
                app=app,
                viewport=viewport,
                gl=GL,
                recorder=recorder,
                mode="top_orthographic",
            ),
        ]
        recorder.require(
            "scene.source_vertices_unchanged_after_render",
            np.array_equal(mesh.vertices, source_snapshot),
        )
        # WA_DontShowOnScreen keeps this diagnostic invisible on a developer's
        # desktop, so Qt may not emit the compositor-level frameSwapped signal.
        # The published frame serial plus actual widget FBO readback above are
        # the blocking paintGL evidence.
        report["context"]["frame_swap_count"] = len(frame_swaps)
    except Exception as exc:
        caught = exc
    finally:
        if viewport is not None:
            try:
                if bool(viewport.isValid()):
                    viewport.makeCurrent()
                    try:
                        viewport.clear_scene()
                    finally:
                        viewport.doneCurrent()
            except Exception as exc:
                cleanup_errors.append(f"scene cleanup: {type(exc).__name__}: {exc}")
            try:
                viewport.close()
                viewport.deleteLater()
            except Exception as exc:
                cleanup_errors.append(f"widget cleanup: {type(exc).__name__}: {exc}")
        if app is not None:
            try:
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                _pump_events(app, attempts=3)
                app.quit()
            except Exception as exc:
                cleanup_errors.append(f"application cleanup: {type(exc).__name__}: {exc}")

    report["cleanup_errors"] = cleanup_errors
    if caught is not None:
        raise caught
    if cleanup_errors:
        raise DriverSmokeFailure("; ".join(cleanup_errors))


def _new_report() -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "ok": False,
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
        },
        "tested_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00",
            "Z",
        ),
        "source": _source_provenance(),
        "checks": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the real QOpenGLWidget large-coordinate precision smoke."
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--qt-platform",
        help="Native Qt platform to require (for example xcb, cocoa, or windows).",
    )
    parser.add_argument("--context-timeout", type=float, default=8.0)
    args = parser.parse_args(argv)

    report = _new_report()
    try:
        run_driver_smoke(
            report,
            qt_platform=args.qt_platform,
            context_timeout_s=max(1.0, float(args.context_timeout)),
        )
        report["ok"] = True
    except Exception as exc:
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    rendered = json.dumps(
        _json_value(report),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    print(rendered)
    if args.report is not None:
        try:
            write_report(args.report, report)
        except Exception as exc:
            print(
                f"failed to write OpenGL smoke report: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2
    return 0 if bool(report.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
