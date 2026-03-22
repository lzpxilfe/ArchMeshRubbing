import math

import numpy as np

from src.core.tile_profile_fitting import fit_circle_2d


def test_fit_circle_2d_recovers_arc_radius():
    theta = np.linspace(-1.1, 1.1, 64, dtype=np.float64)
    center = np.array([2.5, -1.2], dtype=np.float64)
    radius = 7.4
    pts = np.column_stack(
        [
            center[0] + (np.cos(theta) * radius),
            center[1] + (np.sin(theta) * radius),
        ]
    )

    result = fit_circle_2d(pts)

    assert result.is_defined()
    assert result.center_xy is not None
    assert abs(float(result.center_xy[0]) - float(center[0])) < 0.1
    assert abs(float(result.center_xy[1]) - float(center[1])) < 0.1
    assert result.radius is not None
    assert abs(float(result.radius) - radius) < 0.1
    assert result.arc_span_deg > 100.0
    assert result.confidence > 0.5


def test_fit_circle_2d_tolerates_small_noise():
    theta = np.linspace(-0.9, 0.9, 48, dtype=np.float64)
    center = np.array([0.0, 0.0], dtype=np.float64)
    radius = 5.0
    pts = np.column_stack([np.cos(theta) * radius, np.sin(theta) * radius])
    pts += np.column_stack(
        [
            np.sin(theta * 3.0) * 0.01,
            np.cos(theta * 2.0) * 0.01,
        ]
    )

    result = fit_circle_2d(pts)

    assert result.is_defined()
    assert result.radius is not None
    assert math.isfinite(float(result.radius))
    assert abs(float(result.radius) - radius) < 0.05
    assert result.rmse < 0.05
