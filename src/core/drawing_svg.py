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
from typing import Mapping, Sequence

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
    if not math.isclose(float(token), number, rel_tol=0.0, abs_tol=5e-13):
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


def path_element(
    *,
    path_id: str,
    role: str,
    closed: bool,
    points_mm: Sequence[Sequence[float]],
    placement: Placement,
    fill: str | None = None,
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
    return (
        f'<path id="{xml_attribute(path_id)}" '
        f'data-role="{xml_attribute(role)}"{fill_attribute} '
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
                )
            )
        lines.append(f"{indent}</g>")
    return lines


class _PathLike:
    """Structural note: anything with id, role, closed and points_mm."""

    id: str
    role: str
    closed: bool
    points_mm: Sequence[Sequence[float]]


__all__ = [
    "Placement",
    "SVGRenderError",
    "SVG_DECIMALS",
    "SVG_NAMESPACE",
    "finite_number",
    "hatch_pattern_elements",
    "hatch_pattern_id",
    "hatched_kinds",
    "layer_elements",
    "number_token",
    "path_element",
    "xml_attribute",
]
