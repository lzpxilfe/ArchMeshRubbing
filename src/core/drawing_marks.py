"""Strokes that draw a potter's technique marks inside a painted region.

A technique record (`annotation.technique.v1`) says where a mark is: a face
set on the mesh and its silhouette in each view.  It does not say what goes
on the paper there, and the silhouette itself is the wrong answer - a report
drawing never draws the *extent* of a smoothing mark, it draws the marks.
This module draws them the way the measured-drawing textbooks do [K1][K2]:

- 손누름·지두흔: every press is a small closed oval, and presses sit in rows
  along a coil seam ([K2] 도면 2, 도면 3; [K1] p.35 gives 1-2 cm for a 지두흔).
  A press whose lower rim runs out into the wall is drawn as an inverted U
  instead, and the drafter chooses which of the two the mark reads as.
- 테쌓기흔: the seam itself, a thin slightly wavy line across the wall
  ([K2] 도면 2, 도면 4).  It is read on the inner wall - the outside was
  smoothed over - so it goes on the section half of a mirrored figure
  whatever faces were painted.
- 목리조정흔: clusters of fine parallel strokes with a direction, one cluster
  per pass of the wooden tool ([K2] 도면 6, 도면 7; the text asks for 군집 and
  for the direction to be shown).
- 물손질흔(회전물손질): fine parallel lines following the rotation.
- 타날흔: nothing is drawn on the elevation.  [K1] p.37 has the paddle
  pattern recorded by a rubbing, and [K3] p.11 shows that rubbing pasted on
  the drawing with the elevation line left visible; the record says where
  the paddling is, the rubbing strip on the axis says what it looks like.

The numbers - spacings, lengths, jitter - are provisional; the shapes are the
sources'.  Every stroke is deterministic: the same region under the same seed
gives the same strokes, so a sheet can be re-rendered byte for byte, and the
sidecar records the style that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import random
from typing import Any, Mapping, Sequence

from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.geometry.base import BaseGeometry

from .drawing_style import (
    DrawingStyleError,
    TECHNIQUE_BOARD_FINISHING,
    TECHNIQUE_BURNISHING,
    TECHNIQUE_COIL_JOINT,
    TECHNIQUE_FINGER_MARK,
    TECHNIQUE_INTERIOR_ANVIL,
    TECHNIQUE_PADDLING,
    TECHNIQUE_PARING,
    TECHNIQUE_WATER_SMOOTHING,
    TECHNIQUE_WOOD_GRAIN,
)


class DrawingMarkError(ValueError):
    """Raised when a mark cannot be drawn from what it was given."""


#: How a mark is put on paper.
PRESS_OVALS = "press_ovals"
#: The same press drawn as an inverted U - the rim of the depression, open
#: below - rather than as a closed ring.  A finger press is not a hole with an
#: edge all round it: the wall runs on out of it at the bottom, and a drafter
#: who sees it that way draws the part they can see.
PRESS_ARCS = "press_arcs"
SEAM_LINE = "seam_line"
STROKE_PATCHES = "stroke_patches"
PARALLEL_LINES = "parallel_lines"
#: The mark is left to the rubbing: no stroke goes on the line drawing.
RUBBING = "rubbing"
REPRESENTATIONS = (
    PRESS_OVALS,
    PRESS_ARCS,
    SEAM_LINE,
    STROKE_PATCHES,
    PARALLEL_LINES,
    RUBBING,
)

#: Which wall a mark's convention reads it on, where the sources settle it.
#: A coil seam is on both walls of the pot and readable on neither from
#: outside: the potter smoothed the outside over, and it is the inner wall,
#: seen through the cut, that carries the wavy line ([K2] 도면 2, 도면 4).
#: So the seam goes on the section half whatever faces were painted - the
#: painted region says where round the pot the seam runs, not which wall it
#: was read on.  A kind absent here is placed by its record's own faces.
#: An anvil (내박자) is held against the inside of the wall while the paddle
#: strikes the outside, so its mark is on the inner wall by definition - it
#: cannot be anywhere else, whatever faces the drafter happened to paint.
MARK_INTERIOR = "interior"
MARK_EXTERIOR = "exterior"
MARK_OBSERVED_SIDES: Mapping[str, str] = {
    TECHNIQUE_COIL_JOINT: MARK_INTERIOR,
    TECHNIQUE_INTERIOR_ANVIL: MARK_INTERIOR,
}

#: A region that would need more strokes than this is asking for a texture,
#: not a drawing; the caller should paint a smaller region or widen the
#: spacing rather than have the sheet swallow it.
MAX_MARK_STROKES = 20_000
#: Points along an oval and along a wavy line, per period.
_OVAL_STEPS = 24
#: An arc is half a turn, so it gets half the oval's points.
_ARC_STEPS = _OVAL_STEPS // 2
_WAVE_STEPS_PER_PERIOD = 8
#: Coordinates are rounded so the strokes do not carry floating noise into
#: the SVG bytes.
_DECIMALS = 4


@dataclass(frozen=True, slots=True)
class MarkStyle:
    """How one technique kind is drawn, in the figure's millimetres.

    A mark is a real thing of a real size - a fingertip, a tool's width - so
    its measures are the artifact's millimetres and scale with the figure,
    not paper millimetres that would stay put when the sheet is reduced.
    """

    representation: str
    angle_deg: float = 0.0
    """Stroke direction on the paper, degrees counter-clockwise from +x."""
    spacing_mm: float = 0.6
    """Distance between the lines of a cluster or of a parallel family."""
    length_mm: float = 10.0
    """Length of one cluster along its direction, or the long axis of an oval."""
    width_mm: float = 3.5
    """Width of one cluster across its direction."""
    gap_mm: float = 1.0
    """Space left between clusters, or the length of a break in a line."""
    waviness_mm: float = 0.0
    """Amplitude of a line's wander; 0 draws it straight."""
    wave_period_mm: float = 6.0
    break_every_mm: float = 0.0
    """Break a line about this often; 0 leaves it whole."""
    crossed: bool = False
    """Add a second family at right angles (a lattice paddle)."""
    jitter: float = 0.15
    """Relative irregularity, 0 to 1: a drafter's hand, not a machine's."""

    def __post_init__(self) -> None:
        if self.representation not in REPRESENTATIONS:
            raise DrawingMarkError(
                f"mark representation must be one of {', '.join(REPRESENTATIONS)}"
            )
        for name in (
            "spacing_mm",
            "length_mm",
            "width_mm",
            "gap_mm",
            "wave_period_mm",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise DrawingMarkError(f"mark {name} must be a positive number")
            object.__setattr__(self, name, float(value))
        for name in ("angle_deg", "waviness_mm", "break_every_mm", "jitter"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise DrawingMarkError(f"mark {name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if self.waviness_mm < 0.0 or self.break_every_mm < 0.0:
            raise DrawingMarkError("mark waviness_mm and break_every_mm cannot be negative")
        if not (0.0 <= self.jitter <= 1.0):
            raise DrawingMarkError("mark jitter must be between 0 and 1")
        if not isinstance(self.crossed, bool):
            raise DrawingMarkError("mark crossed must be a boolean")
        object.__setattr__(self, "angle_deg", float(self.angle_deg) % 180.0)

    def with_angle(self, angle_deg: float) -> "MarkStyle":
        return MarkStyle(
            representation=self.representation,
            angle_deg=angle_deg,
            spacing_mm=self.spacing_mm,
            length_mm=self.length_mm,
            width_mm=self.width_mm,
            gap_mm=self.gap_mm,
            waviness_mm=self.waviness_mm,
            wave_period_mm=self.wave_period_mm,
            break_every_mm=self.break_every_mm,
            crossed=self.crossed,
            jitter=self.jitter,
        )

    def with_representation(self, representation: str) -> "MarkStyle":
        return replace(self, representation=representation)

    @property
    def directional(self) -> bool:
        """Whether the drafter's direction changes what is drawn."""

        return self.representation in (STROKE_PATCHES, PARALLEL_LINES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "angle_deg": self.angle_deg,
            "break_every_mm": self.break_every_mm,
            "crossed": self.crossed,
            "gap_mm": self.gap_mm,
            "jitter": self.jitter,
            "length_mm": self.length_mm,
            "representation": self.representation,
            "spacing_mm": self.spacing_mm,
            "wave_period_mm": self.wave_period_mm,
            "waviness_mm": self.waviness_mm,
            "width_mm": self.width_mm,
        }


#: How each technique kind is drawn.  Keys are line kinds, so the sheet goes
#: from a record's kind to its line kind and from there to its strokes.  The
#: shapes follow [K1][K2]; the numbers are provisional.
TECHNIQUE_MARK_STYLES: Mapping[str, MarkStyle] = {
    # The seam runs round the pot, so it is drawn along the region's long
    # axis, wandering a little as a hand-built seam does.
    TECHNIQUE_COIL_JOINT: MarkStyle(
        representation=SEAM_LINE, waviness_mm=0.15, wave_period_mm=8.0, jitter=0.2
    ),
    # 1-2 cm per press ([K1] p.35); a long painted region is a row of them.
    TECHNIQUE_FINGER_MARK: MarkStyle(
        representation=PRESS_OVALS, length_mm=15.0, jitter=0.12
    ),
    # A paddle's pattern is the rubbing's to show ([K1] p.37, [K3] p.11);
    # drawing lines for it would put a guess where the rubbing puts a fact.
    TECHNIQUE_PADDLING: MarkStyle(representation=RUBBING),
    # 회전물손질: lines follow the rotation, so horizontal on an upright pot,
    # a little wavy and broken as a wet hand leaves them.
    TECHNIQUE_WATER_SMOOTHING: MarkStyle(
        representation=PARALLEL_LINES,
        angle_deg=0.0,
        spacing_mm=1.2,
        waviness_mm=0.12,
        wave_period_mm=5.0,
        break_every_mm=7.0,
        gap_mm=0.8,
        jitter=0.2,
    ),
    # One cluster per pass of the tool: fine parallel lines, 종방향 unless the
    # drafter says otherwise, laid in clusters with gaps between them.
    TECHNIQUE_WOOD_GRAIN: MarkStyle(
        representation=STROKE_PATCHES,
        angle_deg=90.0,
        spacing_mm=0.45,
        length_mm=12.0,
        width_mm=4.0,
        gap_mm=1.5,
        jitter=0.2,
    ),
    # The four 정면 kinds.  Their shapes follow what the tool does to the
    # wall, not a figure in a source: no page of [K1][K2][K3] in hand shows
    # 마연흔, 목판정면, 내박자흔 or 깎기 drawn, so these are the provisional
    # preset's own and docs/DRAWING_CONVENTIONS.md lists them as owing a
    # citation.  What is not provisional is the vocabulary: an archaeologist
    # who reads one of these off the wall can now record it.
    #
    # 목판정면: a board held flat against a turning wall leaves broad, even,
    # nearly straight striations - wider apart than a wet hand's and without
    # its wobble.
    TECHNIQUE_BOARD_FINISHING: MarkStyle(
        representation=PARALLEL_LINES,
        angle_deg=0.0,
        spacing_mm=2.2,
        waviness_mm=0.04,
        wave_period_mm=12.0,
        break_every_mm=18.0,
        gap_mm=1.2,
        jitter=0.1,
    ),
    # 마연: a pebble worked over leather-hard clay in short overlapping
    # passes.  Fine, close, short strokes in patches, laid obliquely because
    # a burnishing hand does not follow the wheel.
    TECHNIQUE_BURNISHING: MarkStyle(
        representation=STROKE_PATCHES,
        angle_deg=60.0,
        spacing_mm=0.3,
        length_mm=7.0,
        width_mm=3.0,
        gap_mm=1.0,
        jitter=0.25,
    ),
    # 내박자: the anvil's face pressed into the inner wall, one rounded
    # depression per blow, in the rows the paddling followed outside.  Drawn
    # as the depression's rim, closed, because that is what is seen from
    # inside - and on the section half, since it is an inner-wall mark.
    TECHNIQUE_INTERIOR_ANVIL: MarkStyle(
        representation=PRESS_OVALS,
        length_mm=22.0,
        width_mm=4.5,
        jitter=0.1,
    ),
    # 깎기: a blade takes a facet off the wall and leaves its edge.  Long,
    # straight, well separated strokes - the edges between facets, not a
    # texture - steeply set, as paring near a foot ring runs.
    TECHNIQUE_PARING: MarkStyle(
        representation=STROKE_PATCHES,
        angle_deg=75.0,
        spacing_mm=1.6,
        length_mm=16.0,
        width_mm=5.0,
        gap_mm=2.5,
        jitter=0.15,
    ),
}


#: The readings a kind may be drawn with, beyond the one it defaults to.
#: A press is the only mark the sources leave a choice on: [K2] 도면 3 has
#: rows of closed presses, and a press whose lower rim runs out into the wall
#: is drawn as the arc alone.  Both are the same observation; which one is on
#: the paper is the drafter's.  Every other kind has one drawing.
TECHNIQUE_MARK_ALTERNATIVES: Mapping[str, tuple[str, ...]] = {
    TECHNIQUE_FINGER_MARK: (PRESS_OVALS, PRESS_ARCS),
}


@dataclass(frozen=True, slots=True)
class MarkStroke:
    """One stroke on the paper: an open polyline or a closed ring."""

    closed: bool
    points_mm: tuple[tuple[float, float], ...]


def _rng(seed: str) -> random.Random:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _round(x: float, y: float) -> tuple[float, float]:
    return (round(float(x), _DECIMALS) + 0.0, round(float(y), _DECIMALS) + 0.0)


def region_polygons(paths: Sequence[Any]) -> list[Polygon]:
    """Build the painted region from an outline payload's paths.

    The outline's `exterior` rings are the region's parts and its `hole` rings
    are cut out of whichever part contains them.  An invalid ring is repaired
    with a zero buffer rather than refused: the region came out of the
    projection code already validated, and this only guards the arithmetic.
    """

    exteriors: list[Polygon] = []
    holes: list[Polygon] = []
    for path in paths:
        points = [tuple(point) for point in path.points_mm]
        if len(points) < 3:
            continue
        ring = Polygon(points)
        if not ring.is_valid:
            ring = ring.buffer(0)
        if ring.is_empty:
            continue
        if path.role == "hole":
            holes.append(ring)
        elif path.role == "exterior":
            exteriors.append(ring)
        else:
            raise DrawingMarkError(f"region path role {path.role!r} is not exterior or hole")
    polygons: list[Polygon] = []
    for exterior in exteriors:
        shape: BaseGeometry = exterior
        for hole in holes:
            if exterior.contains(hole.representative_point()):
                shape = shape.difference(hole)
        for part in getattr(shape, "geoms", [shape]):
            if isinstance(part, Polygon) and not part.is_empty and part.area > 0.0:
                polygons.append(part)
    return polygons


def _long_axis(polygon: Polygon) -> tuple[tuple[float, float], float, float, float]:
    """Centre, long length, short length and angle (radians) of the region."""

    box = polygon.minimum_rotated_rectangle
    coords = list(box.exterior.coords)[:4]
    if len(coords) < 4:
        minx, miny, maxx, maxy = polygon.bounds
        return ((minx + maxx) / 2.0, (miny + maxy) / 2.0), maxx - minx, maxy - miny, 0.0
    edges = []
    for index in range(4):
        (x0, y0), (x1, y1) = coords[index], coords[(index + 1) % 4]
        edges.append((math.hypot(x1 - x0, y1 - y0), math.atan2(y1 - y0, x1 - x0)))
    long_edge = max(edges[:2], key=lambda item: item[0])
    short_edge = min(edges[:2], key=lambda item: item[0])
    centre = box.centroid
    angle = long_edge[1] % math.pi
    return (float(centre.x), float(centre.y)), long_edge[0], short_edge[0], angle


def _clip(line: LineString, polygon: Polygon, *, min_length_mm: float) -> list[LineString]:
    clipped = line.intersection(polygon)
    pieces: list[LineString] = []
    if clipped.is_empty:
        return pieces
    geoms = clipped.geoms if isinstance(clipped, MultiLineString) else getattr(
        clipped, "geoms", [clipped]
    )
    for piece in geoms:
        if isinstance(piece, LineString) and piece.length >= min_length_mm:
            pieces.append(piece)
    return pieces


def _stroke(points: Sequence[tuple[float, float]], *, closed: bool) -> MarkStroke | None:
    rounded: list[tuple[float, float]] = []
    for x, y in points:
        point = _round(x, y)
        if rounded and rounded[-1] == point:
            continue
        rounded.append(point)
    if closed and len(rounded) >= 2 and rounded[0] == rounded[-1]:
        rounded.pop()
    if len(rounded) < (3 if closed else 2):
        return None
    if not closed and rounded[0] == rounded[-1]:
        return None
    return MarkStroke(closed=closed, points_mm=tuple(rounded))


def _wavy(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    amplitude_mm: float,
    period_mm: float,
    phase: float,
) -> list[tuple[float, float]]:
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length <= 0.0:
        return [start]
    if amplitude_mm <= 0.0:
        return [start, end]
    ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
    nx, ny = -uy, ux
    steps = max(2, int(length / period_mm * _WAVE_STEPS_PER_PERIOD))
    points: list[tuple[float, float]] = []
    for index in range(steps + 1):
        t = index / steps
        s = t * length
        offset = amplitude_mm * math.sin(2.0 * math.pi * s / period_mm + phase)
        points.append((start[0] + ux * s + nx * offset, start[1] + uy * s + ny * offset))
    return points


def _check_budget(count: int) -> None:
    if count > MAX_MARK_STROKES:
        raise DrawingMarkError(
            f"a technique mark would need more than {MAX_MARK_STROKES} strokes; "
            "paint a smaller region or widen the mark's spacing"
        )


@dataclass(frozen=True, slots=True)
class _Press:
    """One press placed in the region: where it sits and how big it is."""

    centre: tuple[float, float]
    long_mm: float
    short_mm: float
    axis: float
    """The region's own long-axis angle, before this press was tilted."""
    tilt: float
    phase_a: float
    phase_b: float


def _presses(polygon: Polygon, style: MarkStyle, rng: random.Random) -> list[_Press]:
    """Lay a row of presses along the painted region's long axis.

    Shared by the two ways a press is drawn, so a mark keeps its places and
    its sizes when the drafter changes how it reads; the random draws happen
    in the same order either way.
    """

    centre, long_mm, short_mm, angle = _long_axis(polygon)
    if long_mm <= 0.0 or short_mm <= 0.0:
        return []
    count = max(1, int(round(long_mm / style.length_mm))) if long_mm > 1.5 * style.length_mm else 1
    pitch = long_mm / count
    ux, uy = math.cos(angle), math.sin(angle)
    a = min(pitch, style.length_mm) * 0.45
    b = min(max(short_mm * 0.45, a * 0.45), a)
    presses: list[_Press] = []
    for index in range(count):
        offset = (index + 0.5 - count / 2.0) * pitch
        # A pressed patch is not a drawn ellipse: two slow undulations of the
        # radius make it read as a fingertip rather than a compass.
        phase_a, phase_b = rng.random() * math.tau, rng.random() * math.tau
        tilt = angle + (rng.random() - 0.5) * style.jitter * math.pi / 3.0
        presses.append(
            _Press(
                centre=(centre[0] + ux * offset, centre[1] + uy * offset),
                long_mm=a,
                short_mm=b,
                axis=angle,
                tilt=tilt,
                phase_a=phase_a,
                phase_b=phase_b,
            )
        )
    return presses


def _press_wobble(theta: float, press: _Press, style: MarkStyle) -> float:
    return 1.0 + style.jitter * (
        0.5 * math.sin(2.0 * theta + press.phase_a)
        + 0.5 * math.sin(3.0 * theta + press.phase_b)
    )


def _press_ovals(polygon: Polygon, style: MarkStyle, rng: random.Random) -> list[MarkStroke]:
    strokes: list[MarkStroke] = []
    for press in _presses(polygon, style, rng):
        cx, cy = press.centre
        a, b = press.long_mm, press.short_mm
        cos_t, sin_t = math.cos(press.tilt), math.sin(press.tilt)
        points: list[tuple[float, float]] = []
        for step in range(_OVAL_STEPS):
            theta = math.tau * step / _OVAL_STEPS
            wobble = _press_wobble(theta, press, style)
            x0, y0 = a * wobble * math.cos(theta), b * wobble * math.sin(theta)
            points.append((cx + x0 * cos_t - y0 * sin_t, cy + x0 * sin_t + y0 * cos_t))
        stroke = _stroke(points, closed=True)
        if stroke is not None:
            strokes.append(stroke)
    return strokes


def _press_arcs(polygon: Polygon, style: MarkStyle, rng: random.Random) -> list[MarkStroke]:
    """The same presses drawn as inverted Us instead of closed rings.

    Half a turn of the same wobbly ellipse: the stroke runs from one end of
    the press, over the crown, to the other, and the wall below is left
    unbroken.  The crown is turned to the paper's +y whatever way the seam
    runs, so a row of presses reads the same way up across the sheet.
    """

    strokes: list[MarkStroke] = []
    for press in _presses(polygon, style, rng):
        cx, cy = press.centre
        cos_t, sin_t = math.cos(press.tilt), math.sin(press.tilt)
        a = press.long_mm
        # The crown lies along the press's short axis.  Which of its two
        # directions is decided from the region's own axis rather than from
        # this press's tilt, so every arc in a row bulges the same way: read
        # off the tilt, a row running down the sheet would alternate as the
        # jitter took the tilt either side of the vertical.
        b = press.short_mm if math.cos(press.axis) >= 0.0 else -press.short_mm
        points: list[tuple[float, float]] = []
        for step in range(_ARC_STEPS + 1):
            theta = math.pi * step / _ARC_STEPS
            wobble = _press_wobble(theta, press, style)
            x0, y0 = a * wobble * math.cos(theta), b * wobble * math.sin(theta)
            points.append((cx + x0 * cos_t - y0 * sin_t, cy + x0 * sin_t + y0 * cos_t))
        stroke = _stroke(points, closed=False)
        if stroke is not None:
            strokes.append(stroke)
    return strokes


def _seam_lines(polygon: Polygon, style: MarkStyle, rng: random.Random) -> list[MarkStroke]:
    centre, long_mm, _short_mm, angle = _long_axis(polygon)
    if long_mm <= 0.0:
        return []
    ux, uy = math.cos(angle), math.sin(angle)
    nx, ny = -uy, ux
    step_mm = 1.0
    steps = max(8, int(long_mm / step_mm))
    reach = long_mm * 2.0
    runs: list[list[tuple[float, float]]] = [[]]
    for index in range(steps + 1):
        s = (index / steps - 0.5) * long_mm
        px, py = centre[0] + ux * s, centre[1] + uy * s
        probe = LineString([(px - nx * reach, py - ny * reach), (px + nx * reach, py + ny * reach)])
        pieces = _clip(probe, polygon, min_length_mm=0.0)
        if not pieces:
            if runs[-1]:
                runs.append([])
            continue
        piece = max(pieces, key=lambda item: item.length)
        mid = piece.interpolate(0.5, normalized=True)
        runs[-1].append((float(mid.x), float(mid.y)))
    strokes: list[MarkStroke] = []
    phase = rng.random() * math.tau
    for run in runs:
        if len(run) < 2:
            continue
        points: list[tuple[float, float]] = []
        travelled = 0.0
        for index, (x, y) in enumerate(run):
            if index:
                travelled += math.hypot(x - run[index - 1][0], y - run[index - 1][1])
            offset = style.waviness_mm * math.sin(
                2.0 * math.pi * travelled / style.wave_period_mm + phase
            )
            points.append((x + nx * offset, y + ny * offset))
        stroke = _stroke(points, closed=False)
        if stroke is not None:
            strokes.append(stroke)
    return strokes


def _broken(
    piece: LineString, style: MarkStyle, rng: random.Random
) -> list[list[tuple[float, float]]]:
    """Cut one clipped line into runs with gaps, if the style asks for breaks."""

    if style.break_every_mm <= 0.0 or piece.length <= style.break_every_mm:
        return [[(float(x), float(y)) for x, y in piece.coords]]
    runs: list[list[tuple[float, float]]] = []
    position = 0.0
    while position < piece.length:
        run_length = style.break_every_mm * (1.0 + (rng.random() - 0.5) * style.jitter)
        end = min(piece.length, position + run_length)
        if end - position > style.gap_mm:
            samples = max(2, int((end - position) / 1.0))
            runs.append(
                [
                    (float(point.x), float(point.y))
                    for point in (
                        piece.interpolate(position + (end - position) * k / samples)
                        for k in range(samples + 1)
                    )
                ]
            )
        position = end + style.gap_mm
    return runs


def _parallel_lines(
    polygon: Polygon, style: MarkStyle, rng: random.Random
) -> list[MarkStroke]:
    strokes: list[MarkStroke] = []
    angles = [math.radians(style.angle_deg)]
    if style.crossed:
        angles.append(math.radians(style.angle_deg + 90.0))
    minx, miny, maxx, maxy = polygon.bounds
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    half_diagonal = math.hypot(maxx - minx, maxy - miny) / 2.0 + style.spacing_mm
    for angle in angles:
        ux, uy = math.cos(angle), math.sin(angle)
        nx, ny = -uy, ux
        count = int(2.0 * half_diagonal / style.spacing_mm) + 1
        phase = rng.random() * math.tau
        for index in range(count + 1):
            _check_budget(len(strokes))
            offset = -half_diagonal + index * style.spacing_mm
            offset += (rng.random() - 0.5) * style.spacing_mm * style.jitter
            start = (cx + nx * offset - ux * half_diagonal, cy + ny * offset - uy * half_diagonal)
            end = (cx + nx * offset + ux * half_diagonal, cy + ny * offset + uy * half_diagonal)
            line = LineString(
                _wavy(
                    start,
                    end,
                    amplitude_mm=style.waviness_mm,
                    period_mm=style.wave_period_mm,
                    phase=phase + index * 0.7,
                )
            )
            for piece in _clip(line, polygon, min_length_mm=style.spacing_mm * 0.5):
                for run in _broken(piece, style, rng):
                    stroke = _stroke(run, closed=False)
                    if stroke is not None:
                        strokes.append(stroke)
    return strokes


def _stroke_patches(
    polygon: Polygon, style: MarkStyle, rng: random.Random
) -> list[MarkStroke]:
    strokes: list[MarkStroke] = []
    angle = math.radians(style.angle_deg)
    ux, uy = math.cos(angle), math.sin(angle)
    nx, ny = -uy, ux
    minx, miny, maxx, maxy = polygon.bounds
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    half_diagonal = math.hypot(maxx - minx, maxy - miny) / 2.0
    along = style.length_mm + style.gap_mm
    across = style.width_mm + style.gap_mm
    rows = int(2.0 * half_diagonal / across) + 2
    columns = int(2.0 * half_diagonal / along) + 2
    lines_per_patch = max(2, int(style.width_mm / style.spacing_mm) + 1)
    for row in range(rows):
        v = -half_diagonal + row * across
        stagger = 0.5 * along if row % 2 else 0.0
        for column in range(columns):
            u = -half_diagonal + column * along + stagger
            jx = (rng.random() - 0.5) * style.gap_mm * style.jitter * 2.0
            jy = (rng.random() - 0.5) * style.gap_mm * style.jitter * 2.0
            px = cx + ux * u + nx * v + jx
            py = cy + uy * u + ny * v + jy
            tilt = angle + (rng.random() - 0.5) * style.jitter * math.radians(30.0)
            tx, ty = math.cos(tilt), math.sin(tilt)
            sx, sy = -ty, tx
            # Skip a cluster whose centre is outside: a stroke starts on the
            # region, it does not reach in from beside it.
            if not polygon.intersects(
                LineString(
                    [
                        (px - tx * style.length_mm / 2.0, py - ty * style.length_mm / 2.0),
                        (px + tx * style.length_mm / 2.0, py + ty * style.length_mm / 2.0),
                    ]
                )
            ):
                continue
            _check_budget(len(strokes))
            for line_index in range(lines_per_patch):
                w = (line_index - (lines_per_patch - 1) / 2.0) * style.spacing_mm
                head = 0.5 * style.length_mm * (1.0 - rng.random() * style.jitter * 0.5)
                tail = 0.5 * style.length_mm * (1.0 - rng.random() * style.jitter * 0.5)
                start = (px + sx * w - tx * tail, py + sy * w - ty * tail)
                end = (px + sx * w + tx * head, py + sy * w + ty * head)
                for piece in _clip(
                    LineString([start, end]), polygon, min_length_mm=style.spacing_mm
                ):
                    stroke = _stroke(
                        [(float(x), float(y)) for x, y in piece.coords], closed=False
                    )
                    if stroke is not None:
                        strokes.append(stroke)
    return strokes


def _nothing(polygon: Polygon, style: MarkStyle, rng: random.Random) -> list[MarkStroke]:
    return []


_GENERATORS = {
    RUBBING: _nothing,
    PRESS_OVALS: _press_ovals,
    PRESS_ARCS: _press_arcs,
    SEAM_LINE: _seam_lines,
    PARALLEL_LINES: _parallel_lines,
    STROKE_PATCHES: _stroke_patches,
}


def generate_marks(
    polygons: Sequence[Polygon], style: MarkStyle, *, seed: str
) -> list[MarkStroke]:
    """Draw one technique mark's strokes inside its region, deterministically.

    `seed` names the mark (a record id and a view); the same seed and region
    always give the same strokes, in the same order.
    """

    if not isinstance(style, MarkStyle):
        raise DrawingMarkError("style must be a MarkStyle")
    rng = _rng(str(seed))
    generator = _GENERATORS[style.representation]
    strokes: list[MarkStroke] = []
    for polygon in polygons:
        if not isinstance(polygon, Polygon) or polygon.is_empty:
            continue
        strokes.extend(generator(polygon, style, rng))
        _check_budget(len(strokes))
    return strokes


def mark_style_for_line_kind(
    line_kind: str, *, representation: str | None = None
) -> MarkStyle:
    """The style a kind is drawn with, or one of the ways it may be drawn.

    `representation` is the drafter's choice between the readings a kind
    allows.  It is refused for a kind with nothing to choose: how a coil seam
    or a paddle goes on paper is the convention's, not a preference, and a
    sheet that could ask for any representation on any kind would let a
    drawing say something the sources do not.
    """

    style = TECHNIQUE_MARK_STYLES.get(str(line_kind))
    if style is None:
        raise DrawingStyleError(f"line kind {line_kind!r} is not drawn as a technique mark")
    if representation is None or representation == style.representation:
        return style
    allowed = TECHNIQUE_MARK_ALTERNATIVES.get(str(line_kind), ())
    if representation not in allowed:
        offer = ", ".join(allowed) if allowed else "nothing else"
        raise DrawingMarkError(
            f"line kind {line_kind!r} is not drawn as {representation!r}; "
            f"it is drawn as {style.representation!r} and may be drawn as {offer}"
        )
    return style.with_representation(representation)


def observed_side_for_line_kind(line_kind: str) -> str | None:
    """The wall this kind's convention reads the mark on, or None."""

    return MARK_OBSERVED_SIDES.get(str(line_kind))


__all__ = [
    "DrawingMarkError",
    "MARK_EXTERIOR",
    "MARK_INTERIOR",
    "MARK_OBSERVED_SIDES",
    "MAX_MARK_STROKES",
    "MarkStroke",
    "MarkStyle",
    "PARALLEL_LINES",
    "PRESS_ARCS",
    "PRESS_OVALS",
    "REPRESENTATIONS",
    "RUBBING",
    "SEAM_LINE",
    "STROKE_PATCHES",
    "TECHNIQUE_MARK_ALTERNATIVES",
    "TECHNIQUE_MARK_STYLES",
    "generate_marks",
    "mark_style_for_line_kind",
    "observed_side_for_line_kind",
    "region_polygons",
]
