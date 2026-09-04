"""Deterministic SVG primitives shared by every drawing this project emits.

A 1:1 vector export and a composed sheet must place the same measured line in
the same place, so they use one implementation of the coordinate mapping and
one number format.  Two implementations would drift, and the drift would be
invisible until someone measured a printed sheet against a printed export.

Everything here is presentation.  A `Placement` maps a record's canonical
millimetre coordinates onto paper, and nothing in this module can change what
those coordinates are.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import AbstractSet, Mapping, Protocol, Sequence

import numpy as np

from .drawing_style import (
    DrawingStylePreset,
    LINE_KINDS,
    layer_id,
)


SVG_NAMESPACE = "http://www.w3.org/2000/svg"

# Coordinates and lengths are written with at most 12 decimals.  Anything that
# needs more precision than that is not a drawing, it is a measurement, and the
# record it came from is where that precision lives.
SVG_DECIMALS = 12
SVG_TOKEN_TOLERANCE_MM = 1e-9


class _PathLike(Protocol):
    """Anything shaped like a measured path.

    Structural rather than nominal: the vector record layer owns `VectorPath`
    and this module must not import it, or presentation would depend on the
    record model it is supposed to stay clear of.
    """

    @property
    def id(self) -> str: ...

    @property
    def role(self) -> str: ...

    @property
    def closed(self) -> bool: ...

    @property
    def points_mm(self) -> Sequence[Sequence[float]]: ...


class SVGRenderError(ValueError):
    """A value cannot be written into a deterministic SVG."""


def finite_number(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SVGRenderError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise SVGRenderError(f"{field_name} must be a finite number")
    if minimum is not None and number < minimum:
        raise SVGRenderError(f"{field_name} must be at least {minimum}")
    if strictly_positive and number <= 0.0:
        raise SVGRenderError(f"{field_name} must be greater than zero")
    return 0.0 if number == 0.0 else number


def number_token(value: object, *, field_name: str) -> str:
    """Return the fixed precision token shared by SVG size, viewBox and paths."""

    number = finite_number(value, field_name=field_name)
    token = f"{number:.{SVG_DECIMALS}f}".rstrip("0").rstrip(".")
    if token in {"", "-0"}:
        return "0"
    # The check is that the token can carry the value at all, not that it
    # carries every bit of it.  Twelve decimals of a millimetre is a picometre,
    # so the rounding itself is never information; a tolerance set at exactly
    # half the last decimal, though, refuses honest coordinates whose binary
    # representation lands a fraction over the boundary - a cut line through a
    # measured pot did, and took the whole sheet down with it.  A nanometre is
    # three orders below the finest scanner and a thousand times the rounding
    # step, so it separates "unwritable" from "rounded".
    if not math.isclose(float(token), number, rel_tol=0.0, abs_tol=SVG_TOKEN_TOLERANCE_MM):
        raise SVGRenderError(
            f"{field_name} exceeds the {SVG_DECIMALS}-decimal SVG precision contract"
        )
    return token


def xml_attribute(value: object) -> str:
    from xml.sax.saxutils import escape as xml_escape  # noqa: PLC0415

    return xml_escape(str(value), {'"': "&quot;", "'": "&apos;"})


@dataclass(frozen=True, slots=True)
class Placement:
    """Where one record's content sits on paper, and at what reduction.

    `content_bounds_mm` is the record's own extent in canonical millimetres.
    `origin_mm` is where that extent's top-left corner lands on the page.  The
    y axis flips, because a drawing counts downwards from the top of the sheet
    while the artifact's canonical frame counts upwards.
    """

    content_bounds_mm: tuple[float, float, float, float]
    origin_mm: tuple[float, float] = (0.0, 0.0)
    scale_denominator: float = 1.0

    def __post_init__(self) -> None:
        bounds = tuple(
            finite_number(item, field_name="content_bounds_mm")
            for item in self.content_bounds_mm
        )
        if len(bounds) != 4:
            raise SVGRenderError("content_bounds_mm must have four values")
        if bounds[0] > bounds[2] or bounds[1] > bounds[3]:
            raise SVGRenderError("content_bounds_mm must be ordered min then max")
        object.__setattr__(self, "content_bounds_mm", bounds)
        origin = tuple(
            finite_number(item, field_name="origin_mm") for item in self.origin_mm
        )
        if len(origin) != 2:
            raise SVGRenderError("origin_mm must have two values")
        object.__setattr__(self, "origin_mm", origin)
        object.__setattr__(
            self,
            "scale_denominator",
            finite_number(
                self.scale_denominator,
                field_name="scale_denominator",
                strictly_positive=True,
            ),
        )

    @property
    def width_mm(self) -> float:
        """Paper width the placed content occupies."""

        minimum_x, _minimum_y, maximum_x, _maximum_y = self.content_bounds_mm
        return (maximum_x - minimum_x) / self.scale_denominator

    @property
    def height_mm(self) -> float:
        """Paper height the placed content occupies."""

        _minimum_x, minimum_y, _maximum_x, maximum_y = self.content_bounds_mm
        return (maximum_y - minimum_y) / self.scale_denominator

    def paper_xy(self, point: Sequence[float]) -> tuple[float, float]:
        """Map one canonical millimetre point onto the page."""

        minimum_x, _minimum_y, _maximum_x, maximum_y = self.content_bounds_mm
        origin_x, origin_y = self.origin_mm
        x = origin_x + (float(point[0]) - minimum_x) / self.scale_denominator
        y = origin_y + (maximum_y - float(point[1])) / self.scale_denominator
        return x, y


def center_axis_line(
    frame: Mapping[str, Sequence[float]],
    *,
    axis_world: Sequence[float] = (0.0, 0.0, 1.0),
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the rotation axis as an infinite line in a record's own plane.

    The result is `(base, direction)` in the frame's millimetre coordinates,
    with `direction` a unit vector oriented towards +v - and towards +u when
    the axis lies along the frame's v = 0 line - so that "which side" is a
    fixed question and not one whose answer depends on how a record happened
    to be built.

    Returns `None` when the axis is perpendicular to the plane, which is the
    top and bottom views: there it projects to a point, and there is no line.
    """

    try:
        origin = np.asarray(frame["origin_world_mm"], dtype=np.float64)
        u_axis = np.asarray(frame["u_axis_world"], dtype=np.float64)
        v_axis = np.asarray(frame["v_axis_world"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise SVGRenderError(f"frame is not a planar frame: {exc}") from exc
    axis = np.asarray(axis_world, dtype=np.float64)
    if origin.shape != (3,) or u_axis.shape != (3,) or v_axis.shape != (3,):
        raise SVGRenderError("frame axes must be three-component vectors")

    direction = (float(np.dot(axis, u_axis)), float(np.dot(axis, v_axis)))
    length = math.hypot(*direction)
    # A degenerate projection is the perpendicular case, not an error.
    if length <= 1e-12:
        return None
    unit = (direction[0] / length, direction[1] / length)
    if unit[1] < 0.0 or (unit[1] == 0.0 and unit[0] < 0.0):
        unit = (-unit[0], -unit[1])
    base = (float(np.dot(-origin, u_axis)), float(np.dot(-origin, v_axis)))
    return base, unit


def axis_profile_chord(
    frame: Mapping[str, Sequence[float]],
    *,
    height_mm: float,
    radius_mm: float,
    axis_world: Sequence[float] = (0.0, 0.0, 1.0),
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the full-width chord at one height on a body of revolution.

    A feature that runs right round the artifact - a groove, a cordon - crosses
    an elevation as a straight line from one side of the silhouette to the
    other.  Given the height it sits at and the radius the wall has there, this
    is that line in the record's own plane.

    Returns `None` unless the drawing plane *contains* the axis.  On a plane
    the axis merely leans into, the chord would be a foreshortened projection
    of the feature rather than the feature, and a drawing that showed it would
    be claiming a width the artifact does not have there.
    """

    try:
        origin = np.asarray(frame["origin_world_mm"], dtype=np.float64)
        u_axis = np.asarray(frame["u_axis_world"], dtype=np.float64)
        v_axis = np.asarray(frame["v_axis_world"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise SVGRenderError(f"frame is not a planar frame: {exc}") from exc
    if origin.shape != (3,) or u_axis.shape != (3,) or v_axis.shape != (3,):
        raise SVGRenderError("frame axes must be three-component vectors")
    axis = np.asarray(axis_world, dtype=np.float64)
    if axis.shape != (3,):
        raise SVGRenderError("axis_world must be a three-component vector")
    if not (
        math.isfinite(float(height_mm)) and math.isfinite(float(radius_mm))
    ):
        raise SVGRenderError("height_mm and radius_mm must be finite")
    if float(radius_mm) < 0.0:
        raise SVGRenderError("radius_mm must not be negative")

    direction = np.array(
        (float(np.dot(axis, u_axis)), float(np.dot(axis, v_axis))),
        dtype=np.float64,
    )
    # The axis lies in the plane only when all of its length survives the
    # projection onto the plane's two in-plane directions.
    if abs(float(np.hypot(*direction)) - 1.0) > 1e-9:
        return None
    perpendicular = np.array((-direction[1], direction[0]), dtype=np.float64)
    base = np.array(
        (float(np.dot(-origin, u_axis)), float(np.dot(-origin, v_axis))),
        dtype=np.float64,
    )
    centre = base + float(height_mm) * direction
    left = centre - float(radius_mm) * perpendicular
    right = centre + float(radius_mm) * perpendicular
    return (
        (float(left[0]), float(left[1])),
        (float(right[0]), float(right[1])),
    )


def broken_chord(
    start: Sequence[float],
    end: Sequence[float],
    *,
    break_count: int,
    break_mm: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Split one straight line into collinear pieces with evenly spaced gaps.

    간선: the recessed line of a groove is a straight line broken a few times.
    The count is what a drafter decides, not a repeating texture, so the breaks
    are cut into the geometry instead of being asked of a dash pattern - a dash
    pattern fixes how often a line breaks, never how many times.

    Falls back to the unbroken line when the gaps would not fit, so a short
    line is drawn whole rather than as a row of dots.
    """

    if break_count < 0:
        raise SVGRenderError("break_count must not be negative")
    if not math.isfinite(float(break_mm)) or float(break_mm) < 0.0:
        raise SVGRenderError("break_mm must be a non-negative length")
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    whole = [((x0, y0), (x1, y1))]
    length = math.hypot(x1 - x0, y1 - y0)
    gaps = float(break_mm) * break_count
    if break_count == 0 or length <= 0.0 or gaps >= length:
        return whole
    piece = (length - gaps) / (break_count + 1)
    if piece <= 0.0:
        return whole
    unit = ((x1 - x0) / length, (y1 - y0) / length)
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    travelled = 0.0
    for index in range(break_count + 1):
        head = travelled
        travelled += piece
        segments.append(
            (
                (x0 + unit[0] * head, y0 + unit[1] * head),
                (x0 + unit[0] * travelled, y0 + unit[1] * travelled),
            )
        )
        travelled += float(break_mm)
    return segments


FINGER_MARK_ARC_STEPS = 8


def finger_mark_symbol(
    bounds_mm: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    """The U a fingertip leaves, sized to the mark's extent, open upward.

    지두흔 is drawn as a U rather than by its boundary: the boundary of a
    pressed patch is a blob that says nothing, and the U is the sign the
    convention uses for it.  The symbol fills the region's bounding box - the
    two arms on its sides, the bend along its bottom - so it sits where the
    mark is and is as large as the mark.  Which way the U opens is the
    drafter's convention; upward is a provisional choice, recorded as such.
    """

    left, bottom, right, top = (float(value) for value in bounds_mm)
    if not all(math.isfinite(value) for value in (left, bottom, right, top)):
        raise SVGRenderError("finger mark bounds must be finite")
    if right <= left or top <= bottom:
        raise SVGRenderError("finger mark bounds must have positive width and height")
    width = right - left
    height = top - bottom
    # The bend is a half-ellipse across the bottom, as deep as half the width
    # or the whole height, whichever is smaller, so a tall mark keeps straight
    # arms and a flat one is all bend.
    bend = min(width / 2.0, height)
    centre_x = left + width / 2.0
    bend_top = bottom + bend
    points: list[tuple[float, float]] = [(left, top), (left, bend_top)]
    for step in range(1, FINGER_MARK_ARC_STEPS):
        angle = math.pi + math.pi * step / FINGER_MARK_ARC_STEPS
        points.append(
            (centre_x + (width / 2.0) * math.cos(angle), bend_top + bend * math.sin(angle))
        )
    points.append((right, bend_top))
    points.append((right, top))
    return tuple(points)


def center_axis_segment(
    frame: Mapping[str, Sequence[float]],
    bounds: Sequence[float],
    *,
    axis_world: Sequence[float] = (0.0, 0.0, 1.0),
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the artifact's rotation axis as a segment in a record's frame.

    A positioned artifact's axis is `axis_world` through the world origin.  This
    projects that line into the record's own plane and clips it to the drawn
    content, so an elevation gets the centre line at the place the artifact
    actually turns about.

    Returns `None` when the axis is perpendicular to the plane, which is the
    top and bottom views: there the axis projects to a point, and a drawing that
    put a line through it would be asserting something untrue.
    """

    line = center_axis_line(frame, axis_world=axis_world)
    if line is None:
        return None
    base, direction = line

    minimum_u, minimum_v, maximum_u, maximum_v = (float(value) for value in bounds)
    # Liang-Barsky against the content rectangle, on the infinite line.
    low, high = -math.inf, math.inf
    for delta, base_value, low_edge, high_edge in (
        (direction[0], base[0], minimum_u, maximum_u),
        (direction[1], base[1], minimum_v, maximum_v),
    ):
        if abs(delta) <= 1e-12:
            if base_value < low_edge or base_value > high_edge:
                return None
            continue
        first = (low_edge - base_value) / delta
        second = (high_edge - base_value) / delta
        low = max(low, min(first, second))
        high = min(high, max(first, second))
    if not (low < high):
        return None
    return (
        (base[0] + direction[0] * low, base[1] + direction[1] * low),
        (base[0] + direction[0] * high, base[1] + direction[1] * high),
    )


def half_plane_side(
    point: Sequence[float],
    *,
    base: Sequence[float],
    direction: Sequence[float],
) -> float:
    """Return where a point falls relative to an oriented line.

    Negative is the paper-left side and positive the paper-right side, because
    `center_axis_line` orients the direction towards +v and paper x follows +u.
    Exact zero is on the line and belongs to neither half.
    """

    return (float(point[0]) - float(base[0])) * float(direction[1]) - (
        float(point[1]) - float(base[1])
    ) * float(direction[0])


def _crossing_point(
    first: Sequence[float],
    second: Sequence[float],
    first_side: float,
    second_side: float,
) -> tuple[float, float]:
    ratio = first_side / (first_side - second_side)
    return (
        float(first[0]) + ratio * (float(second[0]) - float(first[0])),
        float(first[1]) + ratio * (float(second[1]) - float(first[1])),
    )


def clip_closed_ring(
    points: Sequence[Sequence[float]],
    *,
    base: Sequence[float],
    direction: Sequence[float],
    keep_negative: bool,
    label: str = "path",
) -> list[tuple[float, float]] | None:
    """Return one half of a closed ring, closed along the cutting line.

    A ring that stays wholly on one side is kept or dropped whole.  A ring that
    crosses the line exactly twice - the case a vessel's elevation outline is,
    entering at the rim and leaving at the base - is cut and closed along the
    line itself.

    More than two crossings is refused rather than drawn.  Sutherland-Hodgman
    would answer with a single ring joined by an edge lying on the cutting
    line, which prints as a boundary the artifact does not have; a shape that
    genuinely falls into several pieces at the axis needs a decision this
    module is not entitled to make.
    """

    ring = [(float(point[0]), float(point[1])) for point in points]
    if len(ring) < 3:
        raise SVGRenderError(f"{label}: a closed ring needs at least three points")
    sides = [half_plane_side(point, base=base, direction=direction) for point in ring]
    keep = (lambda value: value < 0.0) if keep_negative else (lambda value: value > 0.0)

    crossings = 0
    for index, side in enumerate(sides):
        following = sides[(index + 1) % len(ring)]
        if (side < 0.0 < following) or (following < 0.0 < side):
            crossings += 1
    if crossings > 2:
        raise SVGRenderError(
            f"{label}: the cutting line divides this closed path into more than "
            "two pieces, so which half is the drawing is not a question this "
            "layout can answer"
        )
    if crossings == 0:
        # Points exactly on the line decide nothing; the rest of the ring does.
        return ring if any(keep(side) for side in sides) else None

    clipped: list[tuple[float, float]] = []
    for index, point in enumerate(ring):
        side = sides[index]
        following_index = (index + 1) % len(ring)
        following = sides[following_index]
        if keep(side) or side == 0.0:
            clipped.append(point)
        if (side < 0.0 < following) or (following < 0.0 < side):
            clipped.append(_crossing_point(point, ring[following_index], side, following))
    deduplicated: list[tuple[float, float]] = []
    for point in clipped:
        if not deduplicated or deduplicated[-1] != point:
            deduplicated.append(point)
    if len(deduplicated) > 1 and deduplicated[0] == deduplicated[-1]:
        deduplicated.pop()
    return deduplicated if len(deduplicated) >= 3 else None


def clip_open_path(
    points: Sequence[Sequence[float]],
    *,
    base: Sequence[float],
    direction: Sequence[float],
    keep_negative: bool,
) -> list[list[tuple[float, float]]]:
    """Return the parts of an open polyline that lie on the kept side."""

    chain = [(float(point[0]), float(point[1])) for point in points]
    if len(chain) < 2:
        return []
    sides = [half_plane_side(point, base=base, direction=direction) for point in chain]
    keep = (lambda value: value < 0.0) if keep_negative else (lambda value: value > 0.0)

    pieces: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for index in range(len(chain) - 1):
        side, following = sides[index], sides[index + 1]
        if keep(side) or side == 0.0:
            if not current or current[-1] != chain[index]:
                current.append(chain[index])
        if (side < 0.0 < following) or (following < 0.0 < side):
            crossing = _crossing_point(chain[index], chain[index + 1], side, following)
            if not current or current[-1] != crossing:
                current.append(crossing)
            if keep(side):
                pieces.append(current)
                current = []
            else:
                current = [crossing]
    last_side = sides[-1]
    if keep(last_side) or last_side == 0.0:
        if not current or current[-1] != chain[-1]:
            current.append(chain[-1])
    if current:
        pieces.append(current)
    return [piece for piece in pieces if len(piece) >= 2]


def split_ring_off_line(
    points: Sequence[Sequence[float]],
    *,
    base: Sequence[float],
    direction: Sequence[float],
) -> list[list[tuple[float, float]]] | None:
    """Return a ring's edges with the ones lying on the line removed.

    A ring that was cut at a line is closed along that line, and the closing
    chord is not a boundary of anything: it is where the drawing was folded.
    Stroking it prints an edge the object does not have.  This returns the
    remaining open chains, or `None` when no edge lies on the line and the
    ring is therefore whole.
    """

    ring = [(float(point[0]), float(point[1])) for point in points]
    if len(ring) < 3:
        return None
    extent = max(
        max(point[0] for point in ring) - min(point[0] for point in ring),
        max(point[1] for point in ring) - min(point[1] for point in ring),
        1.0,
    )
    tolerance = 1e-9 * extent
    on_line = [
        abs(half_plane_side(point, base=base, direction=direction)) <= tolerance
        for point in ring
    ]
    edges = [
        (index, (index + 1) % len(ring))
        for index in range(len(ring))
    ]
    dropped = [
        index
        for index, (first, second) in enumerate(edges)
        if on_line[first] and on_line[second]
    ]
    if not dropped:
        return None

    chains: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    # Start after a dropped edge so a chain is never split across the wrap.
    start = (dropped[-1] + 1) % len(edges)
    for step in range(len(edges)):
        index = (start + step) % len(edges)
        first, second = edges[index]
        if index in dropped:
            if len(current) >= 2:
                chains.append(current)
            current = []
            continue
        if not current:
            current = [ring[first]]
        current.append(ring[second])
    if len(current) >= 2:
        chains.append(current)
    return chains


def path_element(
    *,
    path_id: str,
    role: str,
    closed: bool,
    points_mm: Sequence[Sequence[float]],
    placement: Placement,
    fill: str | None = None,
    stroke: str | None = None,
) -> str:
    """Return one `<path>` element for a measured path."""

    commands: list[str] = []
    for index, point in enumerate(points_mm):
        x, y = placement.paper_xy(point)
        command = "M" if index == 0 else "L"
        commands.append(
            f"{command} {number_token(x, field_name='path.x')} "
            f"{number_token(y, field_name='path.y')}"
        )
    if closed:
        commands.append("Z")
    fill_attribute = "" if fill is None else f' fill="{fill}"'
    stroke_attribute = "" if stroke is None else f' stroke="{stroke}"'
    return (
        f'<path id="{xml_attribute(path_id)}" '
        f'data-role="{xml_attribute(role)}"{fill_attribute}{stroke_attribute} '
        f'd="{" ".join(commands)}"/>'
    )


def hatch_pattern_id(kind: str) -> str:
    return f"hatch-{kind.replace('_', '-')}"


def hatch_pattern_elements(
    kinds: Sequence[str],
    *,
    preset: DrawingStylePreset,
    color: str,
    indent: str,
) -> list[str]:
    """Return the `<pattern>` elements for every hatched line kind."""

    hatch = preset.hatch
    spacing = number_token(hatch.spacing_mm, field_name="hatch.spacing_mm")
    half = number_token(hatch.spacing_mm / 2.0, field_name="hatch.spacing_mm")
    stroke = number_token(hatch.stroke_width_mm, field_name="hatch.stroke_width_mm")
    angle = number_token(hatch.angle_deg, field_name="hatch.angle_deg")
    lines: list[str] = []
    for kind in kinds:
        # The line sits half a tile in, so the whole stroke stays inside the
        # tile; drawn on the edge it would be clipped to half its weight.
        lines.extend(
            (
                f'{indent}<pattern id="{hatch_pattern_id(kind)}" '
                'patternUnits="userSpaceOnUse" '
                f'width="{spacing}" height="{spacing}" '
                f'patternTransform="rotate({angle})">',
                f'{indent}  <path d="M {half} 0 L {half} {spacing}" '
                f'stroke="{color}" stroke-width="{stroke}" fill="none"/>',
                f"{indent}</pattern>",
            )
        )
    return lines


def hatched_kinds(paths_by_kind: Mapping[str, Sequence["_PathLike"]], *, preset: DrawingStylePreset) -> list[str]:
    """Return the line kinds that will actually be hatched, in drawing order."""

    return [
        kind
        for kind in LINE_KINDS
        if kind in paths_by_kind
        and preset.style(kind).hatch
        and any(path.closed for path in paths_by_kind[kind])
    ]


def layer_elements(
    paths_by_kind: Mapping[str, Sequence["_PathLike"]],
    *,
    preset: DrawingStylePreset,
    placement: Placement,
    hatched: Sequence[str],
    indent: str,
    fill_only_ids: AbstractSet[str] = frozenset(),
) -> list[str]:
    """Return one `<g>` per line kind, in the vocabulary's own order.

    The order is fixed by the vocabulary rather than by the record, so two
    renders of the same drawing cannot disagree.  A kind with no paths is
    omitted: an empty layer is noise in the layer panel.
    """

    lines: list[str] = []
    for kind in LINE_KINDS:
        paths = paths_by_kind.get(kind)
        if not paths:
            continue
        style = preset.style(kind)
        attributes = (
            f'stroke-width="'
            f'{number_token(style.stroke_width_mm, field_name="stroke_width_mm")}"'
        )
        if style.dash_pattern_mm:
            dashes = ",".join(
                number_token(length, field_name="dash_pattern_mm")
                for length in style.dash_pattern_mm
            )
            attributes += f' stroke-dasharray="{dashes}"'
        lines.append(f'{indent}<g id="{layer_id(kind)}" {attributes}>')
        fill = f"url(#{hatch_pattern_id(kind)})" if kind in hatched else None
        for path in paths:
            # An open path has no interior, so filling it would shade the area
            # under its implicit closing chord.
            # A fill-only path carries an area whose boundary is drawn
            # elsewhere, or not at all: stroking it would print the edge its
            # own layer deliberately left out.
            fill_only = path.id in fill_only_ids
            lines.append(
                indent
                + "  "
                + path_element(
                    path_id=path.id,
                    role=path.role,
                    closed=path.closed,
                    points_mm=path.points_mm,
                    placement=placement,
                    fill=fill if (fill and path.closed) else None,
                    stroke="none" if fill_only else None,
                )
            )
        lines.append(f"{indent}</g>")
    return lines




__all__ = [
    "Placement",
    "SVGRenderError",
    "SVG_DECIMALS",
    "SVG_NAMESPACE",
    "center_axis_line",
    "center_axis_segment",
    "clip_closed_ring",
    "clip_open_path",
    "half_plane_side",
    "finite_number",
    "axis_profile_chord",
    "broken_chord",
    "hatch_pattern_elements",
    "hatch_pattern_id",
    "hatched_kinds",
    "layer_elements",
    "number_token",
    "path_element",
    "split_ring_off_line",
    "xml_attribute",
]
