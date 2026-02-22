from src.core.runtime_defaults import (
    ENV_ARAP_MAX_ITERATIONS,
    ENV_EXPORT_DPI,
    ENV_GUI_MAX_RESOLUTION,
    ENV_GUI_MIN_RESOLUTION,
    ENV_RENDER_RESOLUTION,
    load_runtime_defaults,
)


def _clear_runtime_env(monkeypatch):
    for key in (
        ENV_EXPORT_DPI,
        ENV_RENDER_RESOLUTION,
        ENV_ARAP_MAX_ITERATIONS,
        ENV_GUI_MIN_RESOLUTION,
        ENV_GUI_MAX_RESOLUTION,
    ):
        monkeypatch.delenv(key, raising=False)


def test_runtime_defaults_without_env(monkeypatch):
    _clear_runtime_env(monkeypatch)
    defaults = load_runtime_defaults()

    assert defaults.export_dpi == 300
    assert defaults.render_resolution == 2000
    assert defaults.arap_max_iterations == 30
    assert defaults.gui_min_resolution == 500
    assert defaults.gui_max_resolution == 8000


def test_runtime_defaults_with_valid_env(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv(ENV_EXPORT_DPI, "450")
    monkeypatch.setenv(ENV_RENDER_RESOLUTION, "4096")
    monkeypatch.setenv(ENV_ARAP_MAX_ITERATIONS, "80")
    monkeypatch.setenv(ENV_GUI_MIN_RESOLUTION, "256")
    monkeypatch.setenv(ENV_GUI_MAX_RESOLUTION, "8192")

    defaults = load_runtime_defaults()

    assert defaults.export_dpi == 450
    assert defaults.render_resolution == 4096
    assert defaults.arap_max_iterations == 80
    assert defaults.gui_min_resolution == 256
    assert defaults.gui_max_resolution == 8192


def test_runtime_defaults_invalid_values_fallback(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv(ENV_EXPORT_DPI, "abc")
    monkeypatch.setenv(ENV_RENDER_RESOLUTION, "0")
    monkeypatch.setenv(ENV_ARAP_MAX_ITERATIONS, "-1")
    monkeypatch.setenv(ENV_GUI_MIN_RESOLUTION, "10000")
    monkeypatch.setenv(ENV_GUI_MAX_RESOLUTION, "10")

    defaults = load_runtime_defaults()

    assert defaults.export_dpi == 300
    assert defaults.render_resolution == 2000
    assert defaults.arap_max_iterations == 30
    assert defaults.gui_min_resolution == 500
    assert defaults.gui_max_resolution == 8000
