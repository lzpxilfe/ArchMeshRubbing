"""깨진 자리는 단면이 말한다 — 입면 선은 닫는다.

A pot broken at the rim is still drawn as a vessel: on a 좌 반입면 · 우
반단면 figure the elevation half is a silhouette closed against the centre
line, and a silhouette has no free end.  What it shows at the break is the
edge that is left, drawn round; leaving its line stopped in mid air makes
the half read as an unfinished drawing rather than as a broken pot.

The cut is where the break belongs.  The section half draws the same wall
seen through, and there the wall really does end at the break: its closed
ring becomes two wall lines with nothing across their ends, which is what
"이 조각은 이어진다" looks like on a cut.  The hatched face goes on - a
hatch needs a closed face, and the fold's own unstroked copy keeps it, so
the shading still says the wall is solid.

A figure standing on its own - a sherd of roof tile drawn as itself, with
no section folded against it - is opened as before: there is no centre line
for it to close against.
"""

from __future__ import annotations

import json
from typing import Any

from src.core.artifact_vector_record import VectorPath
from src.core.drawing_sheet import _reach_past_fold, validate_drawing_sheet_bytes
from test_drawing_mirror import (
    ELEVATION_ID,
    SVG_NS,
    _figure,
    _find,
    _mirrored_sheet,
)


def _paths(svg_bytes: bytes, layer: str) -> list[Any]:
    group = _find(_figure(svg_bytes), f"{SVG_NS}g[@id='{layer}']")
    assert group is not None, f"the mirrored figure draws {layer}"
    return [element for element in group.iter() if "d" in element.attrib]


def _points(element: Any) -> list[tuple[float, float]]:
    numbers = [
        float(token)
        for token in element.attrib["d"]
        .replace("M", " ")
        .replace("L", " ")
        .replace("Z", " ")
        .split()
    ]
    return list(zip(numbers[0::2], numbers[1::2], strict=True))


def test_the_break_opens_the_cut_and_leaves_the_elevation_closed() -> None:
    whole = _paths(_mirrored_sheet().svg_bytes, "layer-section-cut")
    assert any(path.attrib["d"].rstrip().endswith("Z") for path in whole), (
        "a wall measured to its end is a closed ring"
    )

    bundle = _mirrored_sheet(sherd_breaks=((ELEVATION_ID, "top"),))
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    broken = _paths(bundle.svg_bytes, "layer-section-cut")

    stroked = [path for path in broken if path.attrib.get("stroke") != "none"]
    filled = [path for path in broken if path.attrib.get("stroke") == "none"]
    assert not any(path.attrib["d"].rstrip().endswith("Z") for path in stroked)
    assert all(":sherd" in path.attrib["id"] for path in stroked)
    # Two lines - the outer wall and the inner - and nothing joining them.
    assert len(stroked) == 2
    # The hatched face is still there: a wall drawn as two hairlines would
    # say the piece is hollow, and the break says nothing about the clay.
    assert filled and all(path.attrib["d"].rstrip().endswith("Z") for path in filled)

    # The elevation half is untouched by the break: same lines, same bytes,
    # closed against the centre line the way a silhouette is.
    outline = _paths(bundle.svg_bytes, "layer-outline-visible")
    assert outline, "the elevation half is still drawn"
    assert not any(":sherd" in path.attrib["id"] for path in outline)
    assert [path.attrib["d"] for path in outline] == [
        path.attrib["d"]
        for path in _paths(_mirrored_sheet().svg_bytes, "layer-outline-visible")
    ]
    # And the cut now stops short of where the elevation still reaches: that
    # gap is the trim, and it is the whole of what the break says here.
    highest_seen = min(y for path in outline for _x, y in _points(path))
    highest_cut = min(y for path in stroked for _x, y in _points(path))
    assert 0.01 < highest_cut - highest_seen < 4.0


def test_the_sheet_counts_the_lines_the_break_cut() -> None:
    foot = json.loads(
        _mirrored_sheet(sherd_breaks=((ELEVATION_ID, "bottom"),)).sidecar_bytes
    )
    bundle = _mirrored_sheet(sherd_breaks=((ELEVATION_ID, "top"),))
    sidecar = json.loads(bundle.sidecar_bytes)

    (entry,) = sidecar["sherd_breaks"]
    assert entry["record_id"] == ELEVATION_ID and entry["side"] == "top"
    # One ring cut - the section's - since the elevation's own line is left
    # closed; the foot is the same, on the fixture's floor.
    assert entry["cut_path_count"] == 1
    (bottom,) = foot["sherd_breaks"]
    assert bottom["cut_path_count"] == 1


def test_the_scissors_touch_the_cut_and_not_the_rim_running_past_the_fold() -> None:
    """An elevation edge that runs on past the fold is the rim seen going
    round, not the wall ending, so a break leaves it whole - and does not
    measure itself from it either.  Both go wrong the same way: the run
    stands at the elevation's height, above the cut's own top, so trimming
    to it would cut that one line and leave the cut closed."""

    whole = _mirrored_sheet(outline_reach="section")
    broken = _mirrored_sheet(outline_reach="section", sherd_breaks=((ELEVATION_ID, "top"),))
    validate_drawing_sheet_bytes(broken.svg_bytes, broken.sidecar_bytes)

    def runs(bundle) -> dict[str, str]:
        return {
            element.attrib["id"]: element.attrib["d"]
            for element in _figure(bundle.svg_bytes).iter()
            if ":past-axis:" in element.attrib.get("id", "")
        }

    past = runs(whole)
    assert past, "the fixture's rim does run past the fold"
    # The break changed nothing about it: same lines, same points.
    assert runs(broken) == past
    # And the cut is what was opened.
    assert any(
        ":sherd" in path.attrib["id"]
        for path in _paths(broken.svg_bytes, "layer-section-cut")
    )
    (entry,) = json.loads(broken.sidecar_bytes)["sherd_breaks"]
    assert entry["cut_path_count"] == 1


def test_the_rim_runs_right_across_even_where_the_cut_does_not_rise_to_it() -> None:
    """The elevation is the silhouette of the whole vessel and a break is
    never level, so the highest surviving rim stands above the one meridian
    the cut was taken on.  A ray level with that rim passes clear over every
    line the cut has - and the edge is there all the same, so it runs right
    round to the artifact's own edge and stops a millimetre short of the
    cut, instead of not being drawn at all."""

    # The axis is the vertical x = 0; the cut is a wall from x 30 to 40,
    # its top at y 20, and the rim's edge crosses the axis at y 10.
    wall = VectorPath(
        id="cut",
        role="cut",
        closed=True,
        points_mm=((30.0, 20.0), (40.0, 20.0), (40.0, 60.0), (30.0, 60.0)),
    )
    run = _reach_past_fold(
        (0.0, 10.0), (1.0, 0.0), [wall],
        base=(0.0, 0.0), direction=(0.0, 1.0), across=(1.0, 0.0), gap_mm=1.0,
    )
    assert run is not None
    start, end = run
    assert start == (0.0, 10.0)
    # Right across to a millimetre short of the cut's own outer edge, and
    # level: the rim is a circle seen edge on, not a line into the wall.
    assert abs(end[0] - 39.0) < 1e-9 and abs(end[1] - 10.0) < 1e-9
    # Where a section line does lie in the way, it still stops at that.
    lower = _reach_past_fold(
        (0.0, 30.0), (1.0, 0.0), [wall],
        base=(0.0, 0.0), direction=(0.0, 1.0), across=(1.0, 0.0), gap_mm=1.0,
    )
    assert lower is not None and abs(lower[1][0] - 29.0) < 1e-9


def test_a_figure_with_no_break_keeps_its_bytes() -> None:
    assert _mirrored_sheet().svg_bytes == _mirrored_sheet(sherd_breaks=()).svg_bytes
