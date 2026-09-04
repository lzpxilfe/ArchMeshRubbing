from __future__ import annotations

import math

import pytest
from shapely.geometry import LineString, Polygon

from src.core.artifact_vector_record import VectorPath
from src.core.drawing_marks import (
    DrawingMarkError,
    MarkStyle,
    PARALLEL_LINES,
    PRESS_OVALS,
    SEAM_LINE,
    STROKE_PATCHES,
    TECHNIQUE_MARK_STYLES,
    generate_marks,
    mark_style_for_line_kind,
    region_polygons,
)
from src.core.drawing_style import (
    DrawingStyleError,
    LINE_KINDS,
    TECHNIQUE_COIL_JOINT,
    TECHNIQUE_FINGER_MARK,
    TECHNIQUE_LINE_KINDS,
    TECHNIQUE_PADDLING,
    TECHNIQUE_WATER_SMOOTHING,
    TECHNIQUE_WOOD_GRAIN,
)


def _band(width: float = 60.0, height: float = 12.0) -> Polygon:
    return Polygon([(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)])


def _inside(strokes, polygon: Polygon, *, tolerance_mm: float = 1e-3) -> None:
    grown = polygon.buffer(tolerance_mm)
    for stroke in strokes:
        assert grown.contains(LineString(stroke.points_mm)), stroke


# --- the style table ------------------------------------------------------------


def test_every_technique_line_kind_has_a_mark_style_and_nothing_else_does() -> None:
    assert set(TECHNIQUE_MARK_STYLES) == set(TECHNIQUE_LINE_KINDS.values())
    for kind in TECHNIQUE_LINE_KINDS.values():
        assert kind in LINE_KINDS
        assert mark_style_for_line_kind(kind) is TECHNIQUE_MARK_STYLES[kind]
    with pytest.raises(DrawingStyleError, match="not drawn as a technique mark"):
        mark_style_for_line_kind("outline_visible")


def test_the_shapes_follow_the_sources() -> None:
    """The textbooks draw each mark a particular way; the table says which."""

    assert TECHNIQUE_MARK_STYLES[TECHNIQUE_FINGER_MARK].representation == PRESS_OVALS
    # 1-2 cm per press, [K1] p.35.
    assert 10.0 <= TECHNIQUE_MARK_STYLES[TECHNIQUE_FINGER_MARK].length_mm <= 20.0
    assert TECHNIQUE_MARK_STYLES[TECHNIQUE_COIL_JOINT].representation == SEAM_LINE
    assert TECHNIQUE_MARK_STYLES[TECHNIQUE_WOOD_GRAIN].representation == STROKE_PATCHES
    assert TECHNIQUE_MARK_STYLES[TECHNIQUE_WATER_SMOOTHING].representation == PARALLEL_LINES
    assert TECHNIQUE_MARK_STYLES[TECHNIQUE_PADDLING].representation == PARALLEL_LINES
    # A wet hand on a turning pot leaves horizontal lines.
    assert TECHNIQUE_MARK_STYLES[TECHNIQUE_WATER_SMOOTHING].angle_deg == 0.0
    # Only the stroke kinds take a direction.
    assert TECHNIQUE_MARK_STYLES[TECHNIQUE_WOOD_GRAIN].directional
    assert not TECHNIQUE_MARK_STYLES[TECHNIQUE_FINGER_MARK].directional
    assert not TECHNIQUE_MARK_STYLES[TECHNIQUE_COIL_JOINT].directional


def test_a_mark_style_refuses_nonsense() -> None:
    with pytest.raises(DrawingMarkError, match="representation"):
        MarkStyle(representation="blob")
    with pytest.raises(DrawingMarkError, match="spacing_mm"):
        MarkStyle(representation=PARALLEL_LINES, spacing_mm=0.0)
    with pytest.raises(DrawingMarkError, match="jitter"):
        MarkStyle(representation=PARALLEL_LINES, jitter=1.5)
    with pytest.raises(DrawingMarkError, match="finite"):
        MarkStyle(representation=PARALLEL_LINES, angle_deg=float("nan"))
    # Angles are directions on paper, so 200° is 20°.
    assert MarkStyle(representation=PARALLEL_LINES, angle_deg=200.0).angle_deg == 20.0
    rotated = TECHNIQUE_MARK_STYLES[TECHNIQUE_WOOD_GRAIN].with_angle(30.0)
    assert rotated.angle_deg == 30.0
    assert rotated.to_dict() == {**TECHNIQUE_MARK_STYLES[TECHNIQUE_WOOD_GRAIN].to_dict(), "angle_deg": 30.0}


# --- the generators --------------------------------------------------------------


def test_strokes_are_deterministic_for_a_seed_and_differ_between_seeds() -> None:
    band = _band()
    for kind, style in TECHNIQUE_MARK_STYLES.items():
        first = generate_marks([band], style, seed=f"{kind}:a")
        again = generate_marks([band], style, seed=f"{kind}:a")
        other = generate_marks([band], style, seed=f"{kind}:b")
        assert first == again, kind
        assert first, kind
        if style.jitter > 0.0:
            assert first != other, kind


def test_every_stroke_stays_inside_the_painted_region() -> None:
    # A region with a hole, so clipping is exercised on both ring kinds.
    region = _band(60.0, 30.0).difference(Polygon([(20, 10), (40, 10), (40, 20), (20, 20)]))
    for kind in (TECHNIQUE_WOOD_GRAIN, TECHNIQUE_WATER_SMOOTHING, TECHNIQUE_PADDLING):
        strokes = generate_marks([region], TECHNIQUE_MARK_STYLES[kind], seed=kind)
        assert strokes
        _inside(strokes, region, tolerance_mm=1e-3)
        for stroke in strokes:
            assert not stroke.closed
            VectorPath(id="x", role="technique_stroke", closed=False, points_mm=stroke.points_mm)


def test_a_finger_press_is_one_oval_and_a_row_is_a_row_of_them() -> None:
    style = TECHNIQUE_MARK_STYLES[TECHNIQUE_FINGER_MARK]
    one = generate_marks([_band(14.0, 11.0)], style, seed="one")
    assert len(one) == 1
    assert one[0].closed
    VectorPath(id="o", role="technique_stroke", closed=True, points_mm=one[0].points_mm)
    # The oval fills the press: about the region's size, not a dot in it.
    xs = [x for x, _ in one[0].points_mm]
    ys = [y for _, y in one[0].points_mm]
    assert 0.6 * 14.0 <= max(xs) - min(xs) <= 14.0
    assert 0.5 * 11.0 <= max(ys) - min(ys) <= 11.0

    row = generate_marks([_band(60.0, 12.0)], style, seed="row")
    # 60 mm at 15 mm a press is four presses in a row ([K2] 도면 2: 열).
    assert len(row) == 4
    assert all(stroke.closed for stroke in row)
    centres = sorted(sum(x for x, _ in stroke.points_mm) / len(stroke.points_mm) for stroke in row)
    gaps = [b - a for a, b in zip(centres, centres[1:])]
    assert all(12.0 <= gap <= 18.0 for gap in gaps)

    # Two separate presses are two ovals whatever the seed.
    two = generate_marks([_band(12.0, 10.0), _band(12.0, 10.0)], style, seed="two")
    assert len(two) == 2


def test_a_coil_seam_is_one_wavy_line_along_the_region() -> None:
    style = TECHNIQUE_MARK_STYLES[TECHNIQUE_COIL_JOINT]
    strokes = generate_marks([_band(80.0, 6.0)], style, seed="seam")
    assert len(strokes) == 1
    seam = strokes[0]
    assert not seam.closed
    xs = [x for x, _ in seam.points_mm]
    ys = [y for _, y in seam.points_mm]
    # Runs the length of the band, along its middle, wandering a little.
    assert max(xs) - min(xs) >= 0.9 * 80.0
    assert all(abs(y - 3.0) <= style.waviness_mm + 1e-6 for y in ys)
    assert max(ys) - min(ys) > 0.0
    _inside(strokes, _band(80.0, 6.0), tolerance_mm=1e-3)

    # A tall region has a vertical seam: the line follows the long axis.
    tall = generate_marks([_band(6.0, 80.0)], style, seed="tall")
    assert len(tall) == 1
    ys = [y for _, y in tall[0].points_mm]
    assert max(ys) - min(ys) >= 0.9 * 80.0


def test_wood_grain_strokes_come_in_clusters_along_the_given_direction() -> None:
    style = TECHNIQUE_MARK_STYLES[TECHNIQUE_WOOD_GRAIN]
    band = _band(40.0, 40.0)
    vertical = generate_marks([band], style, seed="v")
    horizontal = generate_marks([band], style.with_angle(0.0), seed="v")
    assert vertical and horizontal

    def mean_direction(strokes) -> float:
        # A stroke has no front or back, so average the doubled angles: 0°
        # and 180° are then the same direction, as they are on paper.
        sx = sy = 0.0
        for stroke in strokes:
            (x0, y0), (x1, y1) = stroke.points_mm[0], stroke.points_mm[-1]
            theta = 2.0 * math.atan2(y1 - y0, x1 - x0)
            sx += math.cos(theta)
            sy += math.sin(theta)
        return (math.degrees(math.atan2(sy, sx)) / 2.0) % 180.0

    assert abs(mean_direction(vertical) - 90.0) < 8.0
    assert min(mean_direction(horizontal), 180.0 - mean_direction(horizontal)) < 8.0
    # Lines within a cluster are fine and close; clusters are more than one line.
    assert len(vertical) > 20
    lengths = [
        math.hypot(s.points_mm[-1][0] - s.points_mm[0][0], s.points_mm[-1][1] - s.points_mm[0][1])
        for s in vertical
    ]
    assert max(lengths) <= style.length_mm + 1e-6


def test_water_smoothing_lines_are_horizontal_and_broken() -> None:
    style = TECHNIQUE_MARK_STYLES[TECHNIQUE_WATER_SMOOTHING]
    band = _band(60.0, 10.0)
    strokes = generate_marks([band], style, seed="water")
    assert strokes
    for stroke in strokes:
        ys = [y for _, y in stroke.points_mm]
        assert max(ys) - min(ys) <= 2.0 * style.waviness_mm + 1e-6
    # Breaks: no single run spans the whole band.
    spans = [max(x for x, _ in s.points_mm) - min(x for x, _ in s.points_mm) for s in strokes]
    assert max(spans) < 60.0
    # About one line per spacing across the band's height.
    rows = {round(sum(y for _, y in s.points_mm) / len(s.points_mm) / style.spacing_mm) for s in strokes}
    assert 6 <= len(rows) <= 14


def test_a_lattice_paddle_crosses_two_families() -> None:
    plain = TECHNIQUE_MARK_STYLES[TECHNIQUE_PADDLING]
    crossed = MarkStyle(**{**plain.to_dict(), "crossed": True})
    band = _band(30.0, 30.0)
    single = generate_marks([band], plain, seed="p")
    double = generate_marks([band], crossed, seed="p")
    assert len(double) > 1.6 * len(single)


def test_an_empty_or_absurd_region_is_handled() -> None:
    style = TECHNIQUE_MARK_STYLES[TECHNIQUE_WOOD_GRAIN]
    assert generate_marks([], style, seed="none") == []
    with pytest.raises(DrawingMarkError, match="more than"):
        generate_marks([_band(2000.0, 2000.0)], style, seed="huge")


def test_region_polygons_rebuilds_the_region_from_outline_paths() -> None:
    exterior = VectorPath(
        id="e",
        role="exterior",
        closed=True,
        points_mm=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
    )
    hole = VectorPath(
        id="h",
        role="hole",
        closed=True,
        points_mm=((4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)),
    )
    polygons = region_polygons([exterior, hole])
    assert len(polygons) == 1
    assert polygons[0].area == pytest.approx(100.0 - 4.0)
    with pytest.raises(DrawingMarkError, match="role"):
        region_polygons([VectorPath(id="s", role="section", closed=True, points_mm=exterior.points_mm)])
