"""선 평활: a jittering measured line drawn the way a pen would draw it,
declared on the sheet, with the record's coordinates untouched."""

from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from src.core.drawing_sheet import (
    INTERPRETATION_LABEL,
    DrawingSheetError,
    Interpretation,
    compose_drawing_sheet,
    validate_drawing_sheet_bytes,
)
from src.core.drawing_smoothing import (
    MAX_LINE_SMOOTHING_MM,
    MIN_RING_LENGTH_IN_SIGMAS,
    smooth_polyline,
)


def _jittered_circle(radius: float, *, count: int = 720, jitter: float = 0.25, seed: int = 3):
    """A circle traced on a grid: each point pushed in or out by up to ``jitter``."""

    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    radii = radius + rng.uniform(-jitter, jitter, size=count)
    return [(float(r * math.cos(a)), float(r * math.sin(a))) for r, a in zip(radii, angles)]


def _turning_energy(points, *, closed: bool) -> float:
    """Sum of squared turning angles: what a jitter adds and a pen removes."""

    pts = np.asarray(points, dtype=np.float64)
    if closed:
        pts = np.vstack([pts, pts[:2]])
    segments = np.diff(pts, axis=0)
    headings = np.arctan2(segments[:, 1], segments[:, 0])
    turns = np.diff(headings)
    turns = (turns + math.pi) % (2.0 * math.pi) - math.pi
    return float(np.sum(turns**2))


def _polygon_area(points) -> float:
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_a_jittered_ring_comes_out_round_and_the_same_size() -> None:
    """Grid jitter of a quarter millimetre on a 50 mm radius: the smoothed
    ring turns evenly, keeps its area, and its points are the circle's."""

    jagged = _jittered_circle(50.0)
    smooth = smooth_polyline(jagged, closed=True, sigma_mm=1.0)

    assert len(smooth) >= 3
    assert _turning_energy(smooth, closed=True) < _turning_energy(jagged, closed=True) / 100.0
    # Area within a tenth of a percent: smoothing does not shrink the form.
    assert abs(_polygon_area(smooth) - math.pi * 50.0**2) < 0.001 * math.pi * 50.0**2
    # The jitter of a quarter millimetre is cut to well under half.
    radii = np.hypot(*np.asarray(smooth).T)
    assert float(np.abs(radii - 50.0).max()) < 0.15
    assert float(np.abs(radii - 50.0).mean()) < 0.05
    # And far fewer points: the pen's line is a few hundred segments, not
    # one per grid step.
    assert len(smooth) < len(jagged)


def test_an_open_line_keeps_its_ends_and_a_straight_run_stays_straight() -> None:
    rng = np.random.default_rng(7)
    xs = np.linspace(0.0, 40.0, 401)
    jagged = [(float(x), float(rng.uniform(-0.2, 0.2))) for x in xs]
    smooth = smooth_polyline(jagged, closed=False, sigma_mm=1.0)

    assert smooth[0] == jagged[0]
    assert smooth[-1] == jagged[-1]
    assert max(abs(y) for _x, y in smooth) < 0.1
    # A line already straight is the same line, endpoints and all, after
    # simplification: two points.
    straight = [(float(x), 0.5 * float(x)) for x in xs]
    assert smooth_polyline(straight, closed=False, sigma_mm=1.0) == (straight[0], straight[-1])


def test_a_small_ring_and_a_zero_width_are_left_alone() -> None:
    small = _jittered_circle(1.0, count=36, jitter=0.02)
    assert smooth_polyline(small, closed=True, sigma_mm=1.0) == tuple(small)
    assert 2.0 * math.pi * 1.0 < MIN_RING_LENGTH_IN_SIGMAS * 1.0
    jagged = _jittered_circle(50.0)
    assert smooth_polyline(jagged, closed=True, sigma_mm=0.0) == tuple(jagged)
    with pytest.raises(ValueError):
        smooth_polyline([(0.0, float("nan"))], closed=False, sigma_mm=1.0)


def test_the_sheet_declares_the_width_and_its_bytes_move_only_then() -> None:
    from test_drawing_sheet import CUTLINE_ID, OUTLINE_ID, _options, _session

    document = _session().document
    plain = compose_drawing_sheet(document, [OUTLINE_ID, CUTLINE_ID], options=_options())
    unsmoothed = compose_drawing_sheet(
        document,
        [OUTLINE_ID, CUTLINE_ID],
        options=_options(interpretation=Interpretation(line_smoothing_mm=0.0)),
    )
    assert unsmoothed.svg_bytes == plain.svg_bytes
    assert "interpretation" not in json.loads(plain.sidecar_bytes)

    smoothed = compose_drawing_sheet(
        document,
        [OUTLINE_ID, CUTLINE_ID],
        options=_options(interpretation=Interpretation(line_smoothing_mm=0.5)),
    )
    validate_drawing_sheet_bytes(smoothed.svg_bytes, smoothed.sidecar_bytes)
    assert smoothed.svg_bytes != plain.svg_bytes
    sidecar = json.loads(smoothed.sidecar_bytes)
    assert sidecar["interpretation"] == {
        "groove_edge_emphasis": 0.0,
        "line_smoothing_mm": 0.5,
        "note": "",
        "straight_far_edges": False,
        "stroke_straightening_deg": 0.0,
    }
    (row,) = [row for row in sidecar["title_block"] if row["label"] == INTERPRETATION_LABEL]
    assert row["value"] == "선 평활 0.5 mm"
    assert row["value"] in smoothed.svg_bytes.decode("utf-8")
    # The records are what they were: the sidecar names the same payload
    # hashes, because the smoothing is the sheet's, not the measurement's.
    plain_hashes = {f["record_id"]: f["vector_payload_sha256"] for f in json.loads(plain.sidecar_bytes)["figures"]}
    smooth_hashes = {f["record_id"]: f["vector_payload_sha256"] for f in sidecar["figures"]}
    assert smooth_hashes == plain_hashes

    # The validator holds the line: a sheet whose sidecar says it smoothed
    # but whose title block does not is refused.
    forged = json.loads(smoothed.sidecar_bytes)
    forged["title_block"] = [row for row in forged["title_block"] if row["label"] != INTERPRETATION_LABEL]
    from test_drawing_sheet import _resigned_sidecar

    with pytest.raises(DrawingSheetError, match="does not say so"):
        validate_drawing_sheet_bytes(smoothed.svg_bytes, _resigned_sidecar(forged, smoothed.svg_bytes))

    with pytest.raises(DrawingSheetError, match="at most"):
        Interpretation(line_smoothing_mm=MAX_LINE_SMOOTHING_MM + 0.1)
    with pytest.raises(DrawingSheetError, match="line_smoothing_mm"):
        Interpretation(line_smoothing_mm=-0.5)
    assert replace(Interpretation(), line_smoothing_mm=1.0).title_row() == (
        INTERPRETATION_LABEL,
        "선 평활 1 mm",
    )
