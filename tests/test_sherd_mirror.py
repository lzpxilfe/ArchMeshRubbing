"""깨진 자리는 유물의 사실이지 반쪽의 사정이 아니다.

A pot broken at the rim is drawn as a piece of a pot: the lines stop short
of the break and nothing closes them there.  On a 좌 반입면 · 우 반단면
figure the same wall is drawn twice - once seen and once cut - so a break
that stops the elevation's lines and leaves the cut running right over the
top says two things about one artifact, and one of them is a claim nobody
measured.

What is held here is that naming the elevation's broken side breaks its
section half too: the cut's closed ring becomes the two wall lines with
nothing across their ends, which is what "이 조각은 이어진다" looks like on
a cut.  The hatched fill goes with the ring - a hatch needs a closed face,
and the face is not closed where the wall was not measured to an end.
"""

from __future__ import annotations

import json
from typing import Any

from src.core.drawing_sheet import validate_drawing_sheet_bytes
from test_drawing_mirror import (
    ELEVATION_ID,
    SVG_NS,
    _figure,
    _find,
    _mirrored_sheet,
)


def _cut_paths(svg_bytes: bytes) -> list[Any]:
    group = _find(_figure(svg_bytes), f"{SVG_NS}g[@id='layer-section-cut']")
    assert group is not None, "the mirrored figure draws the wall's cut"
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


def test_the_break_opens_the_cut_as_well_as_the_outline() -> None:
    whole = _cut_paths(_mirrored_sheet().svg_bytes)
    assert any(path.attrib["d"].rstrip().endswith("Z") for path in whole), (
        "a wall measured to its end is a closed ring"
    )

    bundle = _mirrored_sheet(sherd_breaks=((ELEVATION_ID, "top"),))
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    broken = _cut_paths(bundle.svg_bytes)

    stroked = [path for path in broken if path.attrib.get("stroke") != "none"]
    filled = [path for path in broken if path.attrib.get("stroke") == "none"]
    assert not any(path.attrib["d"].rstrip().endswith("Z") for path in stroked)
    assert all(":sherd" in path.attrib["id"] for path in stroked)
    # Two lines - the outer wall and the inner - and nothing joining them.
    assert len(stroked) == 2
    # The hatched face is still there: a wall drawn as two hairlines would
    # say the piece is hollow, and the break says nothing about the clay.
    assert filled and all(path.attrib["d"].rstrip().endswith("Z") for path in filled)
    broken = stroked
    # Seen and cut stop at the same break: the outer wall's two drawings end
    # level.  The inner wall's own end is lower because this fixture's rim is
    # bevelled, and that is the artifact, not the trim.
    outline = _find(_figure(bundle.svg_bytes), f"{SVG_NS}g[@id='layer-outline-visible']")
    assert outline is not None
    highest_seen = min(
        y
        for element in outline.iter()
        if "d" in element.attrib
        for _x, y in _points(element)
    )
    highest_cut = min(y for path in broken for _x, y in _points(path))
    assert abs(highest_cut - highest_seen) < 0.01


def test_the_sheet_counts_the_lines_the_break_cut_on_both_halves() -> None:
    foot = json.loads(
        _mirrored_sheet(sherd_breaks=((ELEVATION_ID, "bottom"),)).sidecar_bytes
    )
    bundle = _mirrored_sheet(sherd_breaks=((ELEVATION_ID, "top"),))
    sidecar = json.loads(bundle.sidecar_bytes)

    (entry,) = sidecar["sherd_breaks"]
    assert entry["record_id"] == ELEVATION_ID and entry["side"] == "top"
    # The rim is where this fixture's outline and its cut both reach, so two
    # lines are cut there; the foot sits on the fixture's floor, which the
    # cut reaches too, so both count the outline and the cut.
    assert entry["cut_path_count"] == 2
    (bottom,) = foot["sherd_breaks"]
    assert bottom["cut_path_count"] == 2


def test_a_figure_with_no_break_keeps_its_bytes() -> None:
    assert _mirrored_sheet().svg_bytes == _mirrored_sheet(sherd_breaks=()).svg_bytes
