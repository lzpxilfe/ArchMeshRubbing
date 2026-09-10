"""The room the artifact is turned in is numbers, and the numbers hold.

A backdrop is not a measurement, so nothing here checks a millimetre of the
artifact.  What these tests hold is the three promises the backdrop makes to
the eye: the ground is where the artifact stands, the grid is countable at
any size, and no light ever turns a surface into a hole.
"""

from __future__ import annotations

import math

from itertools import pairwise

import pytest

from src.gui.studio_backdrop import (
    GRID_MAJOR_EVERY,
    KEY_LIGHT,
    GRID_STEPS_MM,
    GRID_TARGET_SQUARES,
    backdrop_description,
    background_colour,
    floor_grid,
    grid_step_mm,
    ground_shadow,
    origin_axes,
    stands_on_origin,
    surface_shade,
)

#: A sherd, a bowl, a storage jar: the range a single studio has to hold.
SIZES_MM = (12.0, 30.0, 95.0, 240.0, 900.0)


@pytest.mark.parametrize("extent_mm", SIZES_MM)
def test_the_grid_stays_countable_at_every_size(extent_mm: float) -> None:
    """A sherd gets a fine grid and a jar a coarse one, and in both the
    artifact spans a number of squares a person can count at a glance."""

    step = grid_step_mm(extent_mm)
    assert step in GRID_STEPS_MM
    squares = extent_mm / step
    assert 1.0 <= squares <= 2.0 * GRID_TARGET_SQUARES, f"{extent_mm} mm spans {squares:.1f} squares"


def test_a_grid_step_is_asked_for_in_millimetres_and_never_zero() -> None:
    assert grid_step_mm(0.0) == GRID_STEPS_MM[0]
    assert grid_step_mm(-5.0) == GRID_STEPS_MM[0]
    assert grid_step_mm(float("nan")) == GRID_STEPS_MM[0]
    assert grid_step_mm(1e9) == GRID_STEPS_MM[-1]


def test_the_floor_is_where_the_artifact_stands() -> None:
    """A vessel on its foot stands on the floor; lift it and the floor
    follows it up, because the floor is the artifact's lowest point."""

    standing = floor_grid((-40.0, -40.0, 0.0, 40.0, 40.0, 120.0))
    assert standing.floor_z_mm == 0.0
    lifted = floor_grid((-40.0, -40.0, 17.5, 40.0, 40.0, 137.5))
    assert lifted.floor_z_mm == 17.5
    assert lifted.step_mm == standing.step_mm, "raising it does not change the ruling"


def test_the_floor_reaches_past_the_artifact_and_rules_evenly() -> None:
    grid = floor_grid((-40.0, -40.0, 0.0, 40.0, 40.0, 120.0))
    assert grid.half_extent_mm > 80.0, "the floor reaches past the artifact"
    offsets = grid.offsets_mm()
    assert len(offsets) == grid.line_count
    assert offsets[0] == pytest.approx(-offsets[-1]), "the ruling is centred"
    gaps = {round(b - a, 9) for a, b in pairwise(offsets)}
    assert gaps == {round(grid.step_mm, 9)}, "every square is the same square"
    assert grid.is_major(0.0)
    assert grid.is_major(grid.step_mm * GRID_MAJOR_EVERY)
    assert not grid.is_major(grid.step_mm)


def test_a_floor_cannot_be_asked_for_at_an_unmeasured_place() -> None:
    with pytest.raises(ValueError, match="finite"):
        floor_grid((0.0, 0.0, 0.0, float("nan"), 1.0, 1.0))


def test_the_shadow_sits_under_the_artifact_and_spreads_past_it() -> None:
    bounds = (-30.0, -20.0, 4.0, 50.0, 20.0, 120.0)
    shadow = ground_shadow(bounds)
    assert shadow.centre_mm == pytest.approx((10.0, 0.0)), "under the artifact, not under the origin"
    assert shadow.floor_z_mm == 4.0, "the shadow lies on the floor the artifact stands on"
    assert shadow.radius_x_mm > 40.0 and shadow.radius_y_mm > 20.0
    assert 0.0 < shadow.opacity < 1.0, "a shadow darkens the floor, it does not delete it"


def test_the_backdrop_is_continuous_from_floor_to_sky() -> None:
    """No band, no seam: a silhouette crossing the horizon does not cross a
    step in the background that could be read as an edge of the artifact."""

    heights = [index / 400.0 for index in range(401)]
    colours = [background_colour(height) for height in heights]
    steps = [
        max(abs(a - b) for a, b in zip(first, second, strict=True))
        for first, second in pairwise(colours)
    ]
    assert max(steps) < 0.01, "the gradient has no visible step in it"
    assert colours[0] != colours[-1], "the floor end and the sky end differ"
    assert sum(colours[-1]) > sum(colours[0]), "the sky is the lighter end"
    assert background_colour(-2.0) == colours[0], "off the bottom is still the floor"
    assert background_colour(3.0) == colours[-1], "off the top is still the sky"


@pytest.mark.parametrize(
    "normal",
    [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        KEY_LIGHT,
        (0.3, -0.4, 0.87),
    ],
)
def test_no_surface_is_ever_a_hole(normal) -> None:
    """A surface turned away from every light is still a surface: the
    ambient floor keeps it readable, and no surface blows out to white."""

    shade = surface_shade(normal)
    assert 0.2 <= shade <= 1.0, f"{normal} shaded to {shade:.3f}"


def test_the_lit_side_is_the_side_the_light_is_on() -> None:
    lit = surface_shade(KEY_LIGHT)
    away = surface_shade(tuple(-value for value in KEY_LIGHT))
    assert lit > away + 0.3, "the light comes from one side and it shows"


def test_the_light_is_over_the_viewer_s_left_shoulder() -> None:
    """The key light is in the viewer's frame, and it is above: a light from
    below would read the artifact's form upside down."""

    assert KEY_LIGHT[0] < 0.0, "from the left"
    assert KEY_LIGHT[1] > 0.0, "from above"
    assert KEY_LIGHT[2] > 0.0, "from in front of the artifact"
    facing = surface_shade((0.0, 0.0, 1.0))
    above = surface_shade((0.0, 1.0, 0.0))
    below = surface_shade((0.0, -1.0, 0.0))
    assert above > below, "the top of a form catches more light than its underside"
    assert facing > below


def test_a_degenerate_normal_does_not_break_the_shading() -> None:
    assert 0.0 <= surface_shade((0.0, 0.0, 0.0)) <= 1.0


def test_the_viewport_is_handed_plain_numbers_it_can_draw() -> None:
    """The backdrop holds no graphics context: everything the viewport needs
    is numbers, so it can be tested where there is no screen at all."""

    description = backdrop_description((-40.0, -40.0, 0.0, 40.0, 40.0, 120.0))
    assert set(description) == {
        "floor_colour",
        "grid",
        "grid_colour",
        "grid_major_colour",
        "horizon_colour",
        "origin_axes",
        "origin_axis_colours",
        "shadow",
        "shadow_colour",
        "sky_colour",
        "stands_on_origin",
    }
    assert description["grid"]["step_mm"] == grid_step_mm(80.0)
    assert description["shadow"]["floor_z_mm"] == 0.0
    for key in ("sky_colour", "horizon_colour", "floor_colour", "grid_colour", "shadow_colour"):
        channels = description[key]
        assert len(channels) == 3
        assert all(0.0 <= channel <= 1.0 for channel in channels), key
    assert all(math.isfinite(value) for value in description["grid"].values())


def test_the_three_axes_cross_at_the_origin() -> None:
    """Positioning puts the artifact's own axis on (0, 0, 0), so the room
    draws the cross there and nowhere else."""

    axes = origin_axes((-40.0, -40.0, 0.0, 40.0, 40.0, 120.0))
    segments = axes.segments()
    assert [name for name, _from, _to, _colour in segments] == ["x", "y", "z"]
    for index, (_name, start, stop, colour) in enumerate(segments):
        # Each arm runs along its own axis and through the origin.
        for other in range(3):
            if other != index:
                assert start[other] == 0.0 and stop[other] == 0.0
        assert start[index] < 0.0 < stop[index], "the cross passes through the origin"
        assert stop[index] == pytest.approx(axes.arm_mm)
        assert len(colour) == 3 and all(0.0 <= channel <= 1.0 for channel in colour)
    assert len({colour for _n, _s, _e, colour in segments}) == 3, "x, y and z are told apart"
    assert axes.arm_mm == axes.step_mm * 2.0, "the cross is measured in the room's own squares"


def test_the_room_says_whether_the_artifact_stands_on_the_origin() -> None:
    """That is what positioning is for: the eye should be able to see it,
    and the reading agrees with what the picture shows."""

    on_it = (-40.0, -40.0, 0.0, 40.0, 40.0, 120.0)
    beside_it = (10.0, -40.0, 0.0, 90.0, 40.0, 120.0)
    above_it = (-40.0, -40.0, 6.0, 40.0, 40.0, 126.0)
    assert stands_on_origin(on_it)
    assert not stands_on_origin(beside_it), "off centre is not positioned"
    assert not stands_on_origin(above_it), "floating above the floor is not positioned"
    assert stands_on_origin((-40.0, -40.0, 0.4, 40.0, 40.0, 120.0)), "0.4 mm is within a scan's own noise"
