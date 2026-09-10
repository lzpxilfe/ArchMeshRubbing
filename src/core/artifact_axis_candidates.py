"""정치를 도와주는 것: 어느 축으로, 어느 높이에 원을 잡을지 먼저 말한다.

Standing an artifact on its rotation axis needs two measured circles, and
the program has had nothing to say about where they should go.  The
archaeologist picks points on the surface, the fit comes back, and only
then - sometimes only at the axis alignment, sometimes only at the drawing
- does it turn out that the height chosen was a poor one: a third of the
circumference missing because the rim is not level, or a band so short
that the two centres are closer than their own fit error.

Nothing here is a record and nothing here decides.  These are readings of
the mesh that answer the two questions a drafter asks first - *which way is
up* and *where is this thing round* - with numbers the archaeologist can
weigh: how far the surface strays from a circle at that height, how much of
the way round it is there at all, and, for a pair, whether it clears the
gates ``build_axis_alignment`` will apply.  What is measured, and what
becomes a record, is still the anchors the archaeologist accepts.

The circle fit is Kåsa's - the algebraic fit that minimises the residual of
x² + y² - 2ax - 2by - c - and its residual is reported as the RMS distance
from the fitted circle, in micrometres.  It is used here to *rank* heights,
not to measure the artifact: the diameter that goes on the drawing is the
surface measurement record's, fitted from anchors resolved on the mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


class ArtifactAxisCandidateError(RuntimeError):
    """Raised when the mesh cannot be read for circle candidates."""


#: How many wedges the circumference is cut into when asking how much of the
#: way round the surface is there.  Twelve is the ring of anchors a diameter
#: record is usually fitted from, so an empty wedge here is an anchor the
#: archaeologist would not be able to place.
DEFAULT_SECTOR_COUNT = 12
#: The slab a candidate is read from, either side of its height.  Thinner
#: than this and a coarse scan leaves wedges empty for no reason but its own
#: triangle size; thicker and a flaring wall's radius is read as a smear.
DEFAULT_BAND_MM = 1.0
#: How far apart the heights that are tried are.
DEFAULT_STEP_MM = 2.0
#: A wedge with fewer points than this is not surface, it is a stray.
MINIMUM_SECTOR_POINTS = 3
#: A circle has three numbers in it, so three points always fit one exactly
#: and their residual is zero however little they look like a ring.  Four is
#: the fewest that can disagree with a circle, and so the fewest a wobble can
#: honestly be read from.
MINIMUM_RING_SECTORS = 4
#: A candidate the program will not put forward: too little of the way round
#: is there to place a ring of anchors on.
MINIMUM_COVERED_FRACTION_THOUSANDTHS = 750
#: Nor will it put forward a wobble this far from a circle - beyond it the
#: height is not a circle of the artifact but a handle, a spout, or a break.
MAXIMUM_ROUNDNESS_UM = 3_000

UP_AXES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("+X", (1.0, 0.0, 0.0)),
    ("+Y", (0.0, 1.0, 0.0)),
    ("+Z", (0.0, 0.0, 1.0)),
)


@dataclass(frozen=True, slots=True)
class AxisCircleCandidate:
    """One height the artifact might have a circle at, and how good it is."""

    height_mm: float
    radius_mm: float
    #: The RMS distance of the ring's outermost points from the fitted
    #: circle.  Small is round.
    roundness_um: int
    #: Where that circle's centre sits across the axis, from the slab's own
    #: fit - two candidates whose centres disagree are not one axis.
    centre_mm: tuple[float, float]
    covered_sectors: int
    sector_count: int
    point_count: int

    @property
    def covered_fraction_thousandths(self) -> int:
        return int(round(1000.0 * self.covered_sectors / self.sector_count))


@dataclass(frozen=True, slots=True)
class AxisCirclePair:
    """Two candidates offered as the top and bottom of one axis."""

    top: AxisCircleCandidate
    bottom: AxisCircleCandidate
    separation_mm: float
    #: How far the line through the two centres leans from the up direction
    #: the candidates were read along.  A big lean is not wrong - the mesh
    #: may simply be lying over - but it is worth seeing before accepting.
    lean_deg: float
    #: What ``build_axis_alignment`` would say about the centre line: whether
    #: the two centres are far enough apart to fix a direction.
    centre_line_usable: bool


def _up_vector(up: Sequence[float]) -> np.ndarray:
    vector = np.asarray(up, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ArtifactAxisCandidateError("up must be three finite numbers")
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise ArtifactAxisCandidateError("up must not be the zero vector")
    return vector / length


def _in_plane_axes(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two directions across the up axis, chosen the same way every time."""

    seed = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(up, seed)
    first /= np.linalg.norm(first)
    return first, np.cross(up, first)


def _fit_circle(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Kåsa's circle through these points: (centre, radius, RMS residual)."""

    design = np.column_stack([2.0 * points, np.ones(points.shape[0])])
    target = points[:, 0] ** 2 + points[:, 1] ** 2
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    centre = solution[:2]
    radius_squared = float(solution[2]) + float(centre @ centre)
    if not math.isfinite(radius_squared) or radius_squared <= 0.0:
        raise ArtifactAxisCandidateError("the points do not lie on a circle")
    radius = math.sqrt(radius_squared)
    residual = np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1]) - radius
    return centre, radius, float(np.sqrt(np.mean(residual**2)))


def propose_up_axis(
    canonical_vertices_mm: object,
    *,
    band_count: int = 20,
    sector_count: int = DEFAULT_SECTOR_COUNT,
) -> tuple[tuple[str, float, int], ...]:
    """Which way is up, read off the mesh: the three axes, roundest first.

    A vessel is round about one axis and not about the other two, so the
    axis whose bands are roundest is the one it was turned on.  Each entry
    is (the axis's name, the median wobble of its bands as a fraction of
    their radius in thousandths, how many bands could be read at all).  The
    file's own axes are what is tried - the answer names one of them, and
    the archaeologist maps it onto the artifact.
    """

    vertices = np.asarray(canonical_vertices_mm, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        raise ArtifactAxisCandidateError("vertices must be (n, 3) with at least three points")
    if not np.all(np.isfinite(vertices)):
        raise ArtifactAxisCandidateError("vertices must be finite")
    scored: list[tuple[str, float, int]] = []
    for name, direction in UP_AXES:
        up = _up_vector(direction)
        first, second = _in_plane_axes(up)
        along = vertices @ up
        across = np.column_stack([vertices @ first, vertices @ second])
        edges = np.linspace(float(along.min()), float(along.max()), band_count + 1)
        wobbles: list[float] = []
        for index in range(band_count):
            inside = (along >= edges[index]) & (along < edges[index + 1])
            if int(inside.sum()) < sector_count * MINIMUM_SECTOR_POINTS:
                continue
            read = _ring_from_slab(across[inside], sector_count=sector_count)
            if read is None or read[0].shape[0] < sector_count:
                continue
            _ring, _centre, radius, residual = read
            if radius <= 0.0:
                continue
            wobbles.append(residual / radius)
        scored.append(
            (name, 1000.0 * float(np.median(wobbles)) if wobbles else math.inf, len(wobbles))
        )
    return tuple(sorted(scored, key=lambda entry: entry[1]))


def _outermost_by_sector(
    across: np.ndarray, *, sector_count: int, centre: np.ndarray
) -> np.ndarray | None:
    """The furthest point in each wedge round ``centre`` that has one.

    An empty wedge is left out rather than guessed at, so the number of
    points that come back says how much of the way round the surface is
    there.  None when too few wedges carry surface to read a wobble from.
    """

    offset = across - centre
    angle = np.arctan2(offset[:, 1], offset[:, 0])
    wedge = np.floor((angle + math.pi) / (2.0 * math.pi) * sector_count).astype(np.int64)
    wedge = np.clip(wedge, 0, sector_count - 1)
    radius = np.hypot(offset[:, 0], offset[:, 1])
    picked: list[int] = []
    for index in range(sector_count):
        inside = np.flatnonzero(wedge == index)
        if inside.size < MINIMUM_SECTOR_POINTS:
            continue
        picked.append(int(inside[np.argmax(radius[inside])]))
    if len(picked) < MINIMUM_RING_SECTORS:
        return None
    return across[picked]


def _ring_from_slab(
    across: np.ndarray, *, sector_count: int
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """The surface round the axis in one slab: (ring, centre, radius, RMS).

    The wedges have to be counted about the artifact's own centre, not about
    the mean of whatever surface survives at that height.  Half a rim, read
    about its own mean, spreads across nearly every wedge and reports itself
    as a whole circle - which is exactly the failure this reading exists to
    prevent.  So the slab is read twice: once about the mean to find a
    circle, then again about that circle's centre, which is where the
    counting is honest.
    """

    seed = _outermost_by_sector(across, sector_count=sector_count, centre=across.mean(axis=0))
    if seed is None:
        return None
    try:
        seed_centre, _radius, _residual = _fit_circle(seed)
    except (ArtifactAxisCandidateError, np.linalg.LinAlgError):
        return None
    ring = _outermost_by_sector(across, sector_count=sector_count, centre=seed_centre)
    if ring is None:
        return None
    try:
        centre, radius, residual = _fit_circle(ring)
    except (ArtifactAxisCandidateError, np.linalg.LinAlgError):
        return None
    if not math.isfinite(residual) or not math.isfinite(radius):
        return None
    return ring, centre, radius, residual


def propose_axis_circles(
    canonical_vertices_mm: object,
    *,
    up: Sequence[float] = (0.0, 0.0, 1.0),
    band_mm: float = DEFAULT_BAND_MM,
    step_mm: float = DEFAULT_STEP_MM,
    sector_count: int = DEFAULT_SECTOR_COUNT,
) -> tuple[AxisCircleCandidate, ...]:
    """Where along the up axis this artifact is round, and how round.

    Each height is read from a slab ``band_mm`` either side of it: the
    furthest point in each of ``sector_count`` wedges round the centre is
    taken as the surface, a circle is fitted to those points, and the fit's
    residual is the wobble.  A height where some wedges are empty still
    carries a candidate, saying how many of them are there - that is the rim
    which is not level, and counting it is the whole point; leaving it out of
    the proposal is ``usable_axis_circles``'s job, by a stated rule.
    """

    vertices = np.asarray(canonical_vertices_mm, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        raise ArtifactAxisCandidateError("vertices must be (n, 3) with at least three points")
    if not np.all(np.isfinite(vertices)):
        raise ArtifactAxisCandidateError("vertices must be finite")
    if not math.isfinite(band_mm) or band_mm <= 0.0:
        raise ArtifactAxisCandidateError("band_mm must be a positive length")
    if not math.isfinite(step_mm) or step_mm <= 0.0:
        raise ArtifactAxisCandidateError("step_mm must be a positive length")
    if sector_count < 4:
        raise ArtifactAxisCandidateError("sector_count must be at least four")
    axis = _up_vector(up)
    first, second = _in_plane_axes(axis)
    along = vertices @ axis
    across = np.column_stack([vertices @ first, vertices @ second])
    low, high = float(along.min()), float(along.max())
    candidates: list[AxisCircleCandidate] = []
    height = low + band_mm
    while height <= high - band_mm + 1e-9:
        inside = np.abs(along - height) <= band_mm
        count = int(inside.sum())
        if count >= MINIMUM_RING_SECTORS * MINIMUM_SECTOR_POINTS:
            # The wedges that carry surface, and the furthest point in each.
            # A height with empty wedges is still worth reporting - that is
            # exactly the rim which is not level, and saying so is the whole
            # point - so the circle is fitted to the wedges there are.
            read = _ring_from_slab(across[inside], sector_count=sector_count)
            if read is not None:
                ring, centre, radius, residual = read
                candidates.append(
                    AxisCircleCandidate(
                        height_mm=round(height, 4),
                        radius_mm=round(radius, 4),
                        roundness_um=int(round(residual * 1000.0)),
                        centre_mm=(round(float(centre[0]), 4), round(float(centre[1]), 4)),
                        covered_sectors=int(ring.shape[0]),
                        sector_count=sector_count,
                        point_count=count,
                    )
                )
        height += step_mm
    return tuple(candidates)


def usable_axis_circles(
    candidates: Sequence[AxisCircleCandidate],
    *,
    maximum_roundness_um: int = MAXIMUM_ROUNDNESS_UM,
) -> tuple[AxisCircleCandidate, ...]:
    """The candidates worth offering: round enough, and there all the way
    round.  What is left out is left out for one of two stated reasons, and
    the caller still has the whole list to show beside them."""

    return tuple(
        candidate
        for candidate in candidates
        if candidate.roundness_um <= maximum_roundness_um
        and candidate.covered_fraction_thousandths >= MINIMUM_COVERED_FRACTION_THOUSANDTHS
    )


def propose_axis_circle_pairs(
    candidates: Sequence[AxisCircleCandidate],
    *,
    up: Sequence[float] = (0.0, 0.0, 1.0),
    limit: int = 5,
) -> tuple[AxisCirclePair, ...]:
    """Pairs of candidates offered as one axis, best first.

    A pair is judged on what the axis is actually fitted from: the two
    circles' own roundness, and how far apart they stand.  The separation
    gates are the ones ``build_axis_alignment`` applies, so a pair marked
    unusable for a centre line is one that module would refuse - and the
    archaeologist can then take the common normal or stand it on its foot
    instead, which is a choice the geometry does not make.
    """

    from .artifact_axis_alignment import (  # noqa: PLC0415
        MINIMUM_CENTER_SEPARATION_MM,
        MINIMUM_SEPARATION_TO_RADIUS_RATIO,
    )

    axis = _up_vector(up)
    usable = usable_axis_circles(candidates)
    pairs: list[AxisCirclePair] = []
    for lower in usable:
        for upper in usable:
            if upper.height_mm <= lower.height_mm:
                continue
            separation = upper.height_mm - lower.height_mm
            across = math.hypot(
                upper.centre_mm[0] - lower.centre_mm[0],
                upper.centre_mm[1] - lower.centre_mm[1],
            )
            span = math.hypot(separation, across)
            widest = max(upper.radius_mm, lower.radius_mm)
            pairs.append(
                AxisCirclePair(
                    top=upper,
                    bottom=lower,
                    separation_mm=round(span, 4),
                    lean_deg=round(math.degrees(math.atan2(across, separation)), 3),
                    centre_line_usable=(
                        span >= MINIMUM_CENTER_SEPARATION_MM
                        and span >= MINIMUM_SEPARATION_TO_RADIUS_RATIO * widest
                    ),
                )
            )
    # Round circles far apart first.  The direction is fitted from both
    # centres, so what it is worth is roughly the wobble divided by the
    # baseline - a long baseline of two wobbling circles is no better than a
    # short one.  Where that ties, and on a turned pot it very nearly always
    # does, the longer baseline wins: the wobble that is measured is not the
    # only error the circles carry, and every error left un-measured shrinks
    # the same way when the two circles stand further apart.
    pairs.sort(
        key=lambda pair: (
            not pair.centre_line_usable,
            max(pair.top.roundness_um, pair.bottom.roundness_um) / max(pair.separation_mm, 1e-6),
            -pair.separation_mm,
        )
    )
    _ = axis  # the lean is already measured along it
    # Offer choices, not the same choice three times.  The heights next to
    # the best pair's are very nearly as good and would fill the list without
    # telling the archaeologist anything, so a height that is already spoken
    # for does not come back in another pair.
    offered: list[AxisCirclePair] = []
    spoken_for: set[float] = set()
    for pair in pairs:
        if len(offered) >= max(int(limit), 0):
            break
        if pair.top.height_mm in spoken_for or pair.bottom.height_mm in spoken_for:
            continue
        offered.append(pair)
        spoken_for.add(pair.top.height_mm)
        spoken_for.add(pair.bottom.height_mm)
    return tuple(offered)


__all__ = [
    "ArtifactAxisCandidateError",
    "AxisCircleCandidate",
    "AxisCirclePair",
    "DEFAULT_BAND_MM",
    "DEFAULT_SECTOR_COUNT",
    "DEFAULT_STEP_MM",
    "MAXIMUM_ROUNDNESS_UM",
    "MINIMUM_COVERED_FRACTION_THOUSANDTHS",
    "propose_axis_circle_pairs",
    "propose_axis_circles",
    "propose_up_axis",
    "usable_axis_circles",
]
