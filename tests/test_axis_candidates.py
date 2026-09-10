"""정치 도우미: 어느 축으로, 어느 높이에 원을 잡을지 프로그램이 먼저 말한다.

The archaeologist has had to guess where the two circles go, and the guess
fails late: a height whose rim is not level leaves a third of the ring with
no surface to place an anchor on, and the failure arrives as an empty
sequence deep in the fit.  What is held here is that the mesh is asked
first, and answers with numbers - how round, how much of the way round,
how far apart - and that nothing here decides anything: the record is still
the anchors the archaeologist accepts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.artifact_axis_candidates import (
    ArtifactAxisCandidateError,
    propose_axis_circle_pairs,
    propose_axis_circles,
    propose_up_axis,
    usable_axis_circles,
)


def _cone(*, height: float = 90.0, rings: int = 60, segments: int = 96, tilt: float = 0.0) -> np.ndarray:
    """A wall of revolution about +Z, optionally leant over."""

    points = []
    for ring in range(rings + 1):
        z = height * ring / rings
        radius = 25.0 + 0.35 * z
        for step in range(segments):
            angle = 2.0 * math.pi * step / segments
            points.append((radius * math.cos(angle), radius * math.sin(angle), z))
    cloud = np.asarray(points, dtype=np.float64)
    if tilt:
        c, s = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
        cloud = cloud @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]).T
    return cloud


def test_the_mesh_says_which_way_is_up() -> None:
    """A vessel is round about one axis and not the other two, so the axis
    whose bands wobble least is the one it was turned on.  The answer names
    one of the file's own axes, which is what an archaeologist has to map
    onto the artifact."""

    scored = propose_up_axis(_cone())
    assert scored[0][0] == "+Z"
    # Round about Z by construction; the other two are not round at all.
    assert scored[0][1] < 5.0
    assert scored[1][1] > 10.0 * max(scored[0][1], 1e-6) or not math.isfinite(scored[1][1])


def test_where_the_artifact_is_round_comes_back_with_its_numbers() -> None:
    cloud = _cone()
    candidates = propose_axis_circles(cloud, step_mm=5.0)
    assert len(candidates) > 10
    for candidate in candidates:
        # A cone of revolution is round everywhere, all the way round.
        assert candidate.covered_sectors == candidate.sector_count
        assert candidate.roundness_um < 500
        assert abs(candidate.radius_mm - (25.0 + 0.35 * candidate.height_mm)) < 0.5
    assert usable_axis_circles(candidates) == candidates


def test_a_height_whose_rim_is_not_level_says_how_much_of_it_is_there() -> None:
    """The rim of the round-bottomed pot was not level, and the ring of
    anchors could not be placed at the height chosen.  That is what this
    reading is for: the wedges with no surface are counted rather than
    hidden, and the candidate is left out of the proposal by a stated rule
    instead of failing later."""

    cloud = _cone()
    # Shear the top away on one side, the way a broken or tilted rim goes.
    keep = ~((cloud[:, 2] > 80.0) & (cloud[:, 0] > 0.0))
    ragged = cloud[keep]
    candidates = propose_axis_circles(ragged, step_mm=5.0)
    high = [c for c in candidates if c.height_mm > 82.0]
    assert high, "the height is still reported"
    assert all(c.covered_sectors < c.sector_count for c in high)
    assert all(c.covered_fraction_thousandths < 750 for c in high)
    # And it is not offered: half a ring is no place to put anchors.
    assert all(c.height_mm <= 82.0 for c in usable_axis_circles(candidates))


def test_a_pair_is_offered_with_the_gate_the_alignment_will_apply() -> None:
    """Two circles a few millimetres apart cannot fix a direction - their
    own fit error is larger than the line between them - and the axis
    alignment refuses such a pair.  The proposal says so before the
    archaeologist measures anything, and offers the long baseline first."""

    candidates = propose_axis_circles(_cone(), step_mm=5.0)
    pairs = propose_axis_circle_pairs(candidates, limit=4)
    assert pairs and pairs[0].centre_line_usable
    assert pairs[0].separation_mm > 40.0
    # A body of revolution standing upright leans nowhere.
    assert pairs[0].lean_deg < 1.0

    close = [c for c in candidates if 40.0 <= c.height_mm <= 47.0]
    assert len(close) == 2
    (only,) = propose_axis_circle_pairs(close, limit=1)
    assert only.separation_mm < 0.25 * max(c.radius_mm for c in close)
    assert not only.centre_line_usable


def test_a_leaning_mesh_is_read_along_the_up_it_was_given() -> None:
    """The mesh is not positioned yet, so the up direction is the caller's -
    the manual pre-orientation, or an axis this module proposed.  Read along
    the wrong one, the same vessel is not round anywhere."""

    tilt = math.radians(25.0)
    leaning = _cone(tilt=25.0)
    # The cone was turned about +X, so the axis it was turned on now lies here.
    upright = propose_axis_circles(leaning, up=(0.0, -math.sin(tilt), math.cos(tilt)), step_mm=5.0)
    askew = propose_axis_circles(leaning, up=(0.0, 0.0, 1.0), step_mm=5.0)

    offered_upright = usable_axis_circles(upright)
    offered_askew = usable_axis_circles(askew)
    assert offered_upright == upright, "read along its own axis, every height is round"
    # What matters is what the archaeologist is offered, not what a wedge or
    # two of stray surface happens to fit: the wrong up leaves fewer heights
    # standing, and not one of them is as round as the worst of the right up.
    assert len(offered_askew) < len(offered_upright)
    assert max(c.roundness_um for c in offered_upright) < min(
        c.roundness_um for c in offered_askew
    )


def test_the_reading_refuses_what_it_cannot_read() -> None:
    with pytest.raises(ArtifactAxisCandidateError, match="vertices must be"):
        propose_axis_circles(np.zeros((2, 3)))
    with pytest.raises(ArtifactAxisCandidateError, match="must be finite"):
        propose_axis_circles(np.full((10, 3), np.nan))
    with pytest.raises(ArtifactAxisCandidateError, match="band_mm must be"):
        propose_axis_circles(_cone(), band_mm=0.0)
    with pytest.raises(ArtifactAxisCandidateError, match="sector_count must be"):
        propose_axis_circles(_cone(), sector_count=3)
    with pytest.raises(ArtifactAxisCandidateError, match="must not be the zero vector"):
        propose_axis_circles(_cone(), up=(0.0, 0.0, 0.0))


def test_the_panel_says_where_to_put_the_circles_in_the_drafters_words() -> None:
    """The reading has to arrive as something an archaeologist can act on:
    which way is up, which two heights, and whether the pair will carry a
    centre line.  It is a proposal and says so - the record is still the
    diameters the archaeologist measures."""

    from app_interactive import MainWindow

    summary, message, table = MainWindow._axis_candidate_report(_cone())

    assert summary.startswith("+Z 축")
    assert "위쪽으로 가장 그럴듯한 축: +Z" in message
    assert "좋은 짝부터:" in message
    assert "원 중심선을 쓸 수 있습니다" in message
    assert "제안일 뿐입니다" in message
    # Every height that was read is in the table, offered or not.
    assert len(table.splitlines()) > 10
    assert table.splitlines()[0].startswith("높이 mm")


def test_a_mesh_with_no_round_height_is_told_so_rather_than_offered_one() -> None:
    """A slab is not a body of revolution.  Saying so is the reading's job;
    guessing a height would send the archaeologist to measure a diameter that
    does not exist."""

    from app_interactive import MainWindow

    rng = np.random.default_rng(20260910)
    slab = np.column_stack(
        [
            rng.uniform(-40.0, 40.0, 4000),
            rng.uniform(-25.0, 25.0, 4000),
            rng.uniform(0.0, 12.0, 4000),
        ]
    )
    summary, message, _table = MainWindow._axis_candidate_report(slab)
    assert "쓸 만한 높이 없음" in summary or "짝 없음" in summary
    assert "회전체가 아니거나" in message or "짝을 이룰 만큼" in message
