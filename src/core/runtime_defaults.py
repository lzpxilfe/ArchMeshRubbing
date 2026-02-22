"""
Runtime defaults for CLI/GUI processing.

Values can be overridden via environment variables to avoid hardcoded tuning
in multiple entrypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


ENV_EXPORT_DPI = "ARCHMESHRUBBING_EXPORT_DPI"
ENV_RENDER_RESOLUTION = "ARCHMESHRUBBING_RENDER_RESOLUTION"
ENV_ARAP_MAX_ITERATIONS = "ARCHMESHRUBBING_ARAP_MAX_ITERATIONS"
ENV_GUI_MIN_RESOLUTION = "ARCHMESHRUBBING_GUI_MIN_RESOLUTION"
ENV_GUI_MAX_RESOLUTION = "ARCHMESHRUBBING_GUI_MAX_RESOLUTION"


@dataclass(frozen=True)
class RuntimeDefaults:
    export_dpi: int
    render_resolution: int
    arap_max_iterations: int
    gui_min_resolution: int
    gui_max_resolution: int


def _read_int_env(
    env_name: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default

    if min_value is not None and value < min_value:
        return default
    if max_value is not None and value > max_value:
        return default
    return value


def load_runtime_defaults() -> RuntimeDefaults:
    default_gui_min = 500
    default_gui_max = 8000

    gui_min_resolution = _read_int_env(ENV_GUI_MIN_RESOLUTION, default_gui_min, min_value=16)
    gui_max_resolution = _read_int_env(
        ENV_GUI_MAX_RESOLUTION,
        default_gui_max,
        min_value=gui_min_resolution,
    )

    if gui_min_resolution > gui_max_resolution:
        gui_min_resolution = default_gui_min
        gui_max_resolution = default_gui_max

    render_resolution = _read_int_env(
        ENV_RENDER_RESOLUTION,
        2000,
        min_value=gui_min_resolution,
        max_value=gui_max_resolution,
    )

    return RuntimeDefaults(
        export_dpi=_read_int_env(ENV_EXPORT_DPI, 300, min_value=72, max_value=2400),
        render_resolution=render_resolution,
        arap_max_iterations=_read_int_env(ENV_ARAP_MAX_ITERATIONS, 30, min_value=1, max_value=500),
        gui_min_resolution=gui_min_resolution,
        gui_max_resolution=gui_max_resolution,
    )


DEFAULTS = load_runtime_defaults()
