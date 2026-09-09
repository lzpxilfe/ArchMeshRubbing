"""The room the artifact is turned in.

A flat grey field tells you nothing about where the artifact is.  Turn a
bowl in it and the rim reads as an ellipse with no ground under it; tip it
and you cannot see that it tipped.  What a person needs to judge a form by
eye is what a studio gives them: a floor to stand it on, a light that comes
from one side, and a horizon that says which way is up.

This module is the backdrop as numbers - a vertical gradient, a floor grid
at real millimetre spacing, a height ruler standing on it, and the ground
shadow's ellipse - so it can be tested without a graphics context and drawn
by whatever draws it.  It holds no OpenGL and no Qt: the viewport asks it
what to draw, and draws that.

The room has all three axes on purpose.  The floor gives x and y; a rim, a
neck and a foot are told apart by height, so z gets a ruler of its own at
the same step, standing at the back corner of the artifact's footprint.
And the three axes cross at the world origin, drawn there and nowhere else:
positioning an artifact is putting its own axis onto that point, so a room
that hides it hides the very thing the eye is judging.

Nothing here touches a measurement.  The grid is drawn at the artifact's own
millimetres so a glance gives a sense of size, and it is drawn under the
artifact, never over it: a backdrop that could be mistaken for a scale bar
would be a measurement the drawing did not take.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

Colour = tuple[float, float, float]

#: The gradient the artifact turns against: a cool light above, a warmer
#: and slightly darker floor below, so a silhouette reads against both.
SKY_COLOUR: Colour = (0.93, 0.94, 0.96)
HORIZON_COLOUR: Colour = (0.86, 0.87, 0.89)
FLOOR_COLOUR: Colour = (0.78, 0.77, 0.75)
#: How far up the window the horizon sits.  Low, so a standing vessel has
#: room above it and the floor stays a floor rather than a wall.
HORIZON_FRACTION = 1.0 / 3.0

#: The floor grid: a fine line every step, a heavier one every tenth.
GRID_COLOUR: Colour = (0.68, 0.68, 0.70)
GRID_MAJOR_COLOUR: Colour = (0.55, 0.55, 0.58)
GRID_MAJOR_EVERY = 10

#: The steps a grid may take, in millimetres.  A grid is chosen so that the
#: artifact spans a readable number of squares, whatever its size.
GRID_STEPS_MM: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0)
#: How many fine squares the artifact should span, about.
GRID_TARGET_SQUARES = 12
#: How far the floor reaches past the artifact, as a fraction of its width.
GRID_MARGIN_FRACTION = 1.5

#: The ground shadow: an ellipse under the artifact, darkest at its centre.
SHADOW_COLOUR: Colour = (0.36, 0.35, 0.34)
SHADOW_OPACITY = 0.28
#: How much wider than the artifact the shadow spreads.
SHADOW_SPREAD = 1.15

#: Where the light comes from, in the viewer's frame - x to the right, y
#: up, z toward the viewer.  Over the recorder's left shoulder, which is
#: where a measured drawing is lit from, and it stays there as the artifact
#: is turned: a light fixed to the artifact would leave half the turntable
#: unreadable.  The fill comes from the other side, low, and weak.
KEY_LIGHT: tuple[float, float, float] = (-0.55, 0.62, 0.56)
FILL_LIGHT: tuple[float, float, float] = (0.65, -0.25, 0.35)
KEY_STRENGTH = 0.78
FILL_STRENGTH = 0.22
AMBIENT = 0.30


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 0.0:
        return (0.0, 0.0, 1.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def grid_step_mm(extent_mm: float) -> float:
    """The grid step for an artifact this wide, in millimetres.

    Chosen so the artifact spans about ``GRID_TARGET_SQUARES`` squares: a
    30 mm sherd gets a fine grid and a 900 mm jar a coarse one, and in both
    the squares stay countable by eye.
    """

    if not math.isfinite(extent_mm) or extent_mm <= 0.0:
        return GRID_STEPS_MM[0]
    wanted = extent_mm / GRID_TARGET_SQUARES
    for step in GRID_STEPS_MM:
        if step >= wanted:
            return step
    return GRID_STEPS_MM[-1]


@dataclass(frozen=True, slots=True)
class FloorGrid:
    """The floor under the artifact: where it is, and how it is ruled."""

    step_mm: float
    half_extent_mm: float
    floor_z_mm: float

    @property
    def line_count(self) -> int:
        """How many lines the grid has in one direction."""

        return 2 * int(self.half_extent_mm // self.step_mm) + 1

    def offsets_mm(self) -> list[float]:
        """The grid lines' distances from the centre, in millimetres."""

        reach = int(self.half_extent_mm // self.step_mm)
        return [index * self.step_mm for index in range(-reach, reach + 1)]

    def is_major(self, offset_mm: float) -> bool:
        """Whether this line is one of the heavier ones."""

        index = int(round(offset_mm / self.step_mm))
        return index % GRID_MAJOR_EVERY == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "floor_z_mm": self.floor_z_mm,
            "half_extent_mm": self.half_extent_mm,
            "line_count": self.line_count,
            "major_every": GRID_MAJOR_EVERY,
            "step_mm": self.step_mm,
        }


def floor_grid(bounds_mm: tuple[float, float, float, float, float, float]) -> FloorGrid:
    """The floor for an artifact with these world bounds.

    ``bounds_mm`` is ``(min x, min y, min z, max x, max y, max z)``.  The
    floor sits at the artifact's lowest point, so a vessel stood on its foot
    stands on the floor and a tipped one visibly does not.
    """

    min_x, min_y, min_z, max_x, max_y, max_z = (float(value) for value in bounds_mm)
    if not all(math.isfinite(value) for value in (min_x, min_y, min_z, max_x, max_y, max_z)):
        raise ValueError("bounds must be finite")
    width = max(max_x - min_x, max_y - min_y, 1e-6)
    step = grid_step_mm(width)
    half = max(width * GRID_MARGIN_FRACTION, step * 2.0)
    return FloorGrid(step_mm=step, half_extent_mm=half, floor_z_mm=min_z)


@dataclass(frozen=True, slots=True)
class GroundShadow:
    """The artifact's shadow on the floor: centre, radii, and how dark."""

    centre_mm: tuple[float, float]
    radius_x_mm: float
    radius_y_mm: float
    floor_z_mm: float
    opacity: float = SHADOW_OPACITY

    def to_dict(self) -> dict[str, Any]:
        return {
            "centre_mm": list(self.centre_mm),
            "floor_z_mm": self.floor_z_mm,
            "opacity": self.opacity,
            "radius_x_mm": self.radius_x_mm,
            "radius_y_mm": self.radius_y_mm,
        }


def ground_shadow(bounds_mm: tuple[float, float, float, float, float, float]) -> GroundShadow:
    """Where the artifact darkens the floor it stands on."""

    min_x, min_y, min_z, max_x, max_y, _max_z = (float(value) for value in bounds_mm)
    return GroundShadow(
        centre_mm=(0.5 * (min_x + max_x), 0.5 * (min_y + max_y)),
        radius_x_mm=max(0.5 * (max_x - min_x) * SHADOW_SPREAD, 1e-6),
        radius_y_mm=max(0.5 * (max_y - min_y) * SHADOW_SPREAD, 1e-6),
        floor_z_mm=min_z,
    )


def background_colour(height_fraction: float) -> Colour:
    """The backdrop's colour at this height up the window, 0 at the bottom.

    Two ramps meeting at the horizon, which sits a third of the way up: a
    silhouette reads against the light above and the darker floor below,
    wherever the artifact is turned to.
    """

    fraction = min(1.0, max(0.0, float(height_fraction)))
    horizon = HORIZON_FRACTION
    if fraction <= horizon:
        blend = fraction / horizon
        low, high = FLOOR_COLOUR, HORIZON_COLOUR
    else:
        blend = (fraction - horizon) / (1.0 - horizon)
        low, high = HORIZON_COLOUR, SKY_COLOUR
    return (
        low[0] + (high[0] - low[0]) * blend,
        low[1] + (high[1] - low[1]) * blend,
        low[2] + (high[2] - low[2]) * blend,
    )


def surface_shade(normal: tuple[float, float, float]) -> float:
    """How bright a surface with this normal is, from 0 to 1.

    The normal is in the viewer's frame - x right, y up, z toward the
    viewer - because the lights are the room's, not the artifact's.  A key
    light over the left shoulder, a weak fill from the other side to keep
    the shaded half from going flat black, and enough ambient that a surface
    turned right away is still a surface.
    """

    unit_normal = _unit(normal)
    key = sum(a * b for a, b in zip(unit_normal, _unit(KEY_LIGHT)))
    fill = sum(a * b for a, b in zip(unit_normal, _unit(FILL_LIGHT)))
    lit = AMBIENT + KEY_STRENGTH * max(0.0, key) + FILL_STRENGTH * max(0.0, fill)
    return min(1.0, max(0.0, lit))


#: The three axes at the world origin, in the order x, y, z.  Muted enough
#: to sit under a drawing's own lines, distinct enough to tell apart.
ORIGIN_AXIS_COLOURS: tuple[Colour, Colour, Colour] = (
    (0.76, 0.30, 0.28),
    (0.30, 0.58, 0.32),
    (0.28, 0.42, 0.74),
)
#: How many grid squares each arm of the triad reaches from the origin.
ORIGIN_ARM_STEPS = 2.0


@dataclass(frozen=True, slots=True)
class OriginAxes:
    """The three axes crossing at the world origin.

    Positioning an artifact (정치) is putting its own axis onto this origin:
    the rotation axis is z through it, the section plane passes through it,
    and the drawing's centre line is its trace.  A room that does not show
    where it is cannot show whether the artifact is standing on it - which
    is the one thing the eye is being asked to judge while positioning.
    """

    arm_mm: float
    step_mm: float

    def segments(self) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], Colour]]:
        """Each axis as (name, from, to, colour), from the origin outward.

        The negative half is a stub a quarter as long, so the cross reads as
        three directions from one point rather than six equal rays.
        """

        out: list[tuple[str, tuple[float, float, float], tuple[float, float, float], Colour]] = []
        for index, name in enumerate("xyz"):
            forward = [0.0, 0.0, 0.0]
            forward[index] = self.arm_mm
            backward = [0.0, 0.0, 0.0]
            backward[index] = -0.25 * self.arm_mm
            colour = ORIGIN_AXIS_COLOURS[index]
            out.append((name, tuple(backward), tuple(forward), colour))  # type: ignore[arg-type]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"arm_mm": self.arm_mm, "step_mm": self.step_mm}


def origin_axes(bounds_mm: tuple[float, float, float, float, float, float]) -> OriginAxes:
    """The triad at (0, 0, 0), sized to the room's own grid."""

    grid = floor_grid(bounds_mm)
    return OriginAxes(arm_mm=grid.step_mm * ORIGIN_ARM_STEPS, step_mm=grid.step_mm)


def stands_on_origin(
    bounds_mm: tuple[float, float, float, float, float, float],
    *,
    tolerance_mm: float = 1.0,
) -> bool:
    """Whether the artifact's footprint is centred on the origin and on the
    floor through it - what positioning is trying to achieve.

    This is a reading of the room, not a measurement: it says what the eye
    would say from the picture, and the Align record says what is true.
    """

    min_x, min_y, min_z, max_x, max_y, _max_z = (float(value) for value in bounds_mm)
    centre_x = 0.5 * (min_x + max_x)
    centre_y = 0.5 * (min_y + max_y)
    return (
        abs(centre_x) <= tolerance_mm
        and abs(centre_y) <= tolerance_mm
        and abs(min_z) <= tolerance_mm
    )


@dataclass(frozen=True, slots=True)
class HeightRuler:
    """The room's z: a vertical line beside the artifact, ticked in mm."""

    base_mm: tuple[float, float]
    floor_z_mm: float
    top_z_mm: float
    step_mm: float

    def ticks_mm(self) -> list[float]:
        """The heights above the floor a tick is drawn at."""

        if self.step_mm <= 0.0:
            return []
        count = int((self.top_z_mm - self.floor_z_mm) // self.step_mm)
        return [index * self.step_mm for index in range(count + 1)]

    def is_major(self, height_mm: float) -> bool:
        index = int(round(height_mm / self.step_mm))
        return index % GRID_MAJOR_EVERY == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_mm": list(self.base_mm),
            "floor_z_mm": self.floor_z_mm,
            "major_every": GRID_MAJOR_EVERY,
            "step_mm": self.step_mm,
            "tick_count": len(self.ticks_mm()),
            "top_z_mm": self.top_z_mm,
        }


def height_ruler(bounds_mm: tuple[float, float, float, float, float, float]) -> HeightRuler:
    """The vertical companion to the floor grid.

    The floor says how wide the artifact is and nothing about how tall.  A
    rim is a rim and a neck is a neck by their height, so the room needs a z
    as much as an x and a y: one line standing on the floor at the back
    corner of the artifact's footprint, ticked at the floor's own step so
    the two read as one ruler bent upright.  It stops at the artifact's top,
    not above it, so it never suggests a size the artifact does not have.
    """

    min_x, min_y, min_z, max_x, max_y, max_z = (float(value) for value in bounds_mm)
    if not all(math.isfinite(value) for value in (min_x, min_y, min_z, max_x, max_y, max_z)):
        raise ValueError("bounds must be finite")
    grid = floor_grid(bounds_mm)
    step = grid.step_mm
    return HeightRuler(
        # Behind and to the left of the footprint, one square clear of it, so
        # the ruler never crosses the artifact it stands beside.
        base_mm=(min_x - step, max_y + step),
        floor_z_mm=min_z,
        top_z_mm=max(max_z, min_z),
        step_mm=step,
    )


def backdrop_bands() -> tuple[tuple[float, Colour], ...]:
    """The gradient as stops, bottom to top, for whatever draws it.

    Two bands meeting at the horizon.  A renderer that interpolates between
    consecutive stops reproduces `background_colour` exactly, so the window
    and any offscreen picture of it are the same room.
    """

    return ((0.0, FLOOR_COLOUR), (HORIZON_FRACTION, HORIZON_COLOUR), (1.0, SKY_COLOUR))


def shadow_outline(shadow: GroundShadow, segments: int = 48) -> list[tuple[float, float, float]]:
    """The shadow's rim in world millimetres, for a fan or a polygon."""

    if segments < 3:
        raise ValueError("a shadow needs at least three points")
    centre_x, centre_y = shadow.centre_mm
    return [
        (
            centre_x + shadow.radius_x_mm * math.cos(2.0 * math.pi * index / segments),
            centre_y + shadow.radius_y_mm * math.sin(2.0 * math.pi * index / segments),
            shadow.floor_z_mm,
        )
        for index in range(segments)
    ]


def backdrop_description(bounds_mm: tuple[float, float, float, float, float, float]) -> dict[str, Any]:
    """Everything the viewport needs to draw the room, as plain numbers."""

    grid = floor_grid(bounds_mm)
    return {
        "grid": grid.to_dict(),
        "height_ruler": height_ruler(bounds_mm).to_dict(),
        "origin_axes": origin_axes(bounds_mm).to_dict(),
        "origin_axis_colours": [list(colour) for colour in ORIGIN_AXIS_COLOURS],
        "stands_on_origin": stands_on_origin(bounds_mm),
        "grid_colour": list(GRID_COLOUR),
        "grid_major_colour": list(GRID_MAJOR_COLOUR),
        "shadow": ground_shadow(bounds_mm).to_dict(),
        "shadow_colour": list(SHADOW_COLOUR),
        "sky_colour": list(SKY_COLOUR),
        "horizon_colour": list(HORIZON_COLOUR),
        "floor_colour": list(FLOOR_COLOUR),
    }


__all__ = [
    "AMBIENT",
    "FILL_LIGHT",
    "FILL_STRENGTH",
    "FLOOR_COLOUR",
    "GRID_COLOUR",
    "GRID_MAJOR_COLOUR",
    "GRID_MAJOR_EVERY",
    "GRID_STEPS_MM",
    "HORIZON_COLOUR",
    "HORIZON_FRACTION",
    "KEY_LIGHT",
    "KEY_STRENGTH",
    "ORIGIN_ARM_STEPS",
    "ORIGIN_AXIS_COLOURS",
    "SHADOW_COLOUR",
    "SHADOW_OPACITY",
    "SKY_COLOUR",
    "FloorGrid",
    "GroundShadow",
    "HeightRuler",
    "OriginAxes",
    "backdrop_bands",
    "backdrop_description",
    "background_colour",
    "floor_grid",
    "grid_step_mm",
    "ground_shadow",
    "height_ruler",
    "origin_axes",
    "shadow_outline",
    "stands_on_origin",
    "surface_shade",
]
