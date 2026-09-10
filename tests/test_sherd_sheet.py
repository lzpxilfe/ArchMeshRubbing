"""A piece of something is drawn as a piece of something.

Most of what a site yields is broken: half a tile, a sherd of a pot.  The
place where it stopped being whole is not an edge the maker made, and a
drawing that closes the figure there says the artifact ended there - a claim
nobody measured.  The archaeologist's rule is to stop the lines short of the
break and draw nothing across it, and to say on the sheet that this is a
fragment.

Which side is broken is the archaeologist's word: no reading tells a break
from a cut end for certain.  The sheet then draws it, records it, and the
offline validator holds the page and the sidecar to each other.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any

from itertools import pairwise

import pytest

from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_vector_extractor import (
    commit_vector_computation,
    compute_artifact_cutline,
)
from src.core.artifact_vector_record import PlanarFrame
from src.core.canonical_json import canonical_json_bytes
from src.core.drawing_sheet import (
    SECTION_MARK_OVERRUN_PAPER_MM,
    SHERD_LABEL,
    SHERD_TRIM_PAPER_MM,
    DrawingSheetError,
    DrawingSheetOptions,
    TitleBlock,
    compose_drawing_sheet,
    sherd_title_value,
    validate_drawing_sheet_bytes,
)
from src.core.drawing_sheet_spec import plate_spec, plate_spec_options
from synthetic_tile import AMKIWA_SHAPE, tile_session

SVG_NS = "{http://www.w3.org/2000/svg}"
PLAN_ID = "record:sherd-plan"
SECTION_ID = "record:sherd-section"
LEVEL_ID = "record:sherd-level"
SCALE = 4.0
#: Where the cut was taken along the piece, as a share of its length.
CUT_SHARE = 0.35


def _sherd_document():
    """Half an 암키와, drawn in plan: the break runs across the length."""

    session, _vertices, _faces = tile_session(
        AMKIWA_SHAPE,
        axial_step_mm=6.0,
        angular_step_mm=6.0,
        relief=False,
        broken_at_share=0.55,
        document_id="artifact:sherd",
    )
    outline = compute_artifact_outline(session, "top", precision_grid_mm=0.5)
    session = commit_vector_computation(
        session,
        outline,
        record_id=PLAN_ID,
        created_at="2026-09-09T00:00:00Z",
        operator="tester",
    )
    low = _vertices.min(axis=0)
    high = _vertices.max(axis=0)
    station = float(low[1] + (high[1] - low[1]) * CUT_SHARE)
    section = compute_artifact_cutline(
        session,
        PlanarFrame(
            origin_world_mm=(0.0, station, 0.0),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 0.0, 1.0),
            normal_world=(0.0, -1.0, 0.0),
        ),
    )
    session = commit_vector_computation(
        session,
        section,
        record_id=SECTION_ID,
        created_at="2026-09-09T00:00:01Z",
        operator="tester",
    )
    # A cut level with the plan: parallel to it, so it leaves no trace there.
    level = compute_artifact_cutline(
        session,
        PlanarFrame(
            origin_world_mm=(0.0, 0.0, float(low[2] + (high[2] - low[2]) * 0.5)),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 1.0, 0.0),
            normal_world=(0.0, 0.0, 1.0),
        ),
    )
    return commit_vector_computation(
        session,
        level,
        record_id=LEVEL_ID,
        created_at="2026-09-09T00:00:02Z",
        operator="tester",
    ).document


def _options(**overrides) -> DrawingSheetOptions:
    settings: dict[str, Any] = {
        "title_block": TitleBlock(artifact_label="암키와 편 001"),
        "scale_denominator": SCALE,
    }
    settings.update(overrides)
    return DrawingSheetOptions(**settings)


def _figure(svg_bytes: bytes) -> Any:
    root = ET.fromstring(svg_bytes)
    figures = root.find(f"{SVG_NS}g[@id='sheet-figures']")
    assert figures is not None
    return figures[0]


def _points(path: Any) -> list[tuple[float, float]]:
    tokens = path.attrib["d"].replace("M", " ").replace("L", " ").replace("Z", " ")
    numbers = [float(token) for token in tokens.split()]
    return list(zip(numbers[0::2], numbers[1::2], strict=True))


def _outline_paths(svg_bytes: bytes) -> list[Any]:
    figure = _figure(svg_bytes)
    layer = figure.find(f"{SVG_NS}g[@id='layer-outline-visible']")
    assert layer is not None, "the plan draws an outline"
    return [path for path in layer if path.attrib.get("stroke") != "none"]


@pytest.fixture(scope="module")
def document():
    return _sherd_document()


def test_the_drawing_stops_short_of_the_break_and_nothing_closes_it(document) -> None:
    """The measured thing: where the ink ends on the broken side.

    Whole, the plan's outline runs to the end of the piece.  Named as a
    break, the same outline stops a paper millimetre and a half short of it,
    and no line runs across - which is what tells a reader the tile went on.
    """

    whole = compose_drawing_sheet(document, [PLAN_ID], options=_options())
    whole_points = [point for path in _outline_paths(whole.svg_bytes) for point in _points(path)]
    whole_top = min(y for _x, y in whole_points)  # paper y grows downward

    broken = compose_drawing_sheet(
        document, [PLAN_ID], options=_options(sherd_breaks=((PLAN_ID, "top"),))
    )
    broken_paths = _outline_paths(broken.svg_bytes)
    broken_points = [point for path in broken_paths for point in _points(path)]
    broken_top = min(y for _x, y in broken_points)

    assert broken_top - whole_top == pytest.approx(SHERD_TRIM_PAPER_MM, abs=0.02), (
        "the lines stop a paper millimetre and a half short of the break"
    )
    # Nothing is drawn along the break: no path has two points on that line
    # far enough apart to read as an edge.
    on_the_line = [point for point in broken_points if abs(point[1] - broken_top) < 1e-6]
    assert on_the_line, "the lines do end level, on the trim line"
    for path in broken_paths:
        points = _points(path)
        for (x1, y1), (x2, y2) in pairwise(points):
            if abs(y1 - broken_top) < 1e-6 and abs(y2 - broken_top) < 1e-6:
                span = abs(x2 - x1)
                assert span < 1.0, f"a {span:.1f} mm segment runs along the break"
    # The rest of the piece is untouched: the same lowest edge as before.
    assert max(y for _x, y in broken_points) == pytest.approx(
        max(y for _x, y in whole_points), abs=1e-9
    )
    # A cut ring is drawn open, so no path on that figure closes.
    for path in broken_paths:
        assert not path.attrib["d"].strip().endswith("Z"), "a broken figure is not closed"


def test_the_sheet_says_it_is_a_fragment_and_the_validator_holds_it_to_that(document) -> None:
    bundle = compose_drawing_sheet(
        document,
        [PLAN_ID],
        options=_options(sherd_breaks=((PLAN_ID, "top"), (PLAN_ID, "left"))),
    )
    sidecar = json.loads(bundle.sidecar_bytes)
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)

    assert [entry["side"] for entry in sidecar["sherd_breaks"]] == ["left", "top"]
    assert all(entry["record_id"] == PLAN_ID for entry in sidecar["sherd_breaks"])
    assert all(entry["cut_path_count"] >= 1 for entry in sidecar["sherd_breaks"])
    assert all(
        entry["trim_paper_mm"] == SHERD_TRIM_PAPER_MM for entry in sidecar["sherd_breaks"]
    )
    value = sherd_title_value([(PLAN_ID, "top"), (PLAN_ID, "left")])
    assert value == "좌·상 2곳"
    assert {"label": SHERD_LABEL, "value": value} in sidecar["title_block"]
    assert value in bundle.svg_bytes.decode("utf-8")

    # The row and the key stand or fall together.
    without_row = dict(sidecar)
    without_row["title_block"] = [
        row for row in sidecar["title_block"] if row["label"] != SHERD_LABEL
    ]
    with pytest.raises(DrawingSheetError, match="does not say so"):
        validate_drawing_sheet_bytes(bundle.svg_bytes, canonical_json_bytes(without_row))
    without_key = {key: value for key, value in sidecar.items() if key != "sherd_breaks"}
    with pytest.raises(DrawingSheetError, match="does not carry"):
        validate_drawing_sheet_bytes(bundle.svg_bytes, canonical_json_bytes(without_key))


def test_a_whole_artifact_says_nothing_and_keeps_its_bytes(document) -> None:
    """An empty tuple changes nothing at all."""

    first = compose_drawing_sheet(document, [PLAN_ID], options=_options())
    again = compose_drawing_sheet(document, [PLAN_ID], options=_options(sherd_breaks=()))
    assert first.svg_bytes == again.svg_bytes
    assert first.sidecar_bytes == again.sidecar_bytes
    sidecar = json.loads(first.sidecar_bytes)
    assert "sherd_breaks" not in sidecar
    assert not any(row["label"] == SHERD_LABEL for row in sidecar["title_block"])


def test_a_break_that_names_nothing_is_refused(document) -> None:
    """A choice that named nothing would be silently lost, so it is refused."""

    with pytest.raises(DrawingSheetError, match="does not draw"):
        compose_drawing_sheet(
            document, [PLAN_ID], options=_options(sherd_breaks=(("record:absent", "top"),))
        )
    with pytest.raises(DrawingSheetError, match="sides must be one of"):
        _options(sherd_breaks=((PLAN_ID, "sideways"),))
    with pytest.raises(DrawingSheetError, match="twice"):
        _options(sherd_breaks=((PLAN_ID, "top"), (PLAN_ID, "top")))
    with pytest.raises(DrawingSheetError, match="pairs"):
        _options(sherd_breaks=((PLAN_ID,),))


def test_the_break_travels_in_the_plate_specification(document) -> None:
    """The plate is made again from its document and its specification."""

    options = _options(sherd_breaks=((PLAN_ID, "top"),))
    spec = plate_spec([PLAN_ID], options)
    assert spec["sherd_breaks"] == [[PLAN_ID, "top"]]
    records, restored = plate_spec_options(spec)
    assert list(records) == [PLAN_ID]
    assert restored.sherd_breaks == ((PLAN_ID, "top"),)
    assert plate_spec(records, restored) == spec
    first = compose_drawing_sheet(document, [PLAN_ID], options=options)
    again = compose_drawing_sheet(document, list(records), options=restored)
    assert first.svg_bytes == again.svg_bytes


def test_the_plate_says_where_the_section_was_taken(document) -> None:
    """A cut at a place the archaeologist chose, said on the figure.

    The mark is the cut plane's trace on the plan: a straight line at the
    cut's own station, running past the artifact at both ends so it cannot
    be read as an edge, with A and A′ at the ends to match the section.
    """

    bundle = compose_drawing_sheet(
        document,
        [PLAN_ID, SECTION_ID],
        options=_options(section_marks=((SECTION_ID, PLAN_ID),)),
    )
    sidecar = json.loads(bundle.sidecar_bytes)
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)

    entry = sidecar["section_marks"][0]
    assert entry == {
        "figure_record_id": PLAN_ID,
        "from_mm": entry["from_mm"],
        "letters": ["A", "A′"],
        "section_record_id": SECTION_ID,
        "to_mm": entry["to_mm"],
    }
    # The mark lies at the cut's own station, level across the plan.
    assert entry["from_mm"][1] == pytest.approx(entry["to_mm"][1], abs=1e-6)

    figure = _figure(bundle.svg_bytes)
    layer = figure.find(f"{SVG_NS}g[@id='layer-section-mark']")
    assert layer is not None, "the mark has a layer of its own"
    points = [point for path in layer for point in _points(path)]
    assert len(points) == 2
    # It runs past the artifact at both ends.
    plan_points = [point for path in _outline_paths(bundle.svg_bytes) for point in _points(path)]
    left = min(x for x, _y in plan_points)
    right = max(x for x, _y in plan_points)
    assert min(x for x, _y in points) < left - SECTION_MARK_OVERRUN_PAPER_MM + 0.5
    assert max(x for x, _y in points) > right + SECTION_MARK_OVERRUN_PAPER_MM - 0.5

    letters = [
        element.text
        for element in figure.iter()
        if element.tag.endswith("}text") and element.text in {"A", "A′"}
    ]
    assert sorted(letters) == ["A", "A′"]


def test_a_cut_that_does_not_cross_the_figure_is_refused(document) -> None:
    """A plan and a cut parallel to it have no trace, and asking for one is
    refused rather than drawn as nothing."""

    with pytest.raises(DrawingSheetError, match="does not cross"):
        compose_drawing_sheet(
            document,
            [PLAN_ID, LEVEL_ID],
            options=_options(section_marks=((LEVEL_ID, PLAN_ID),)),
        )
    with pytest.raises(DrawingSheetError, match="does not draw"):
        compose_drawing_sheet(
            document,
            [PLAN_ID, SECTION_ID],
            options=_options(section_marks=((SECTION_ID, "record:absent"),)),
        )
    with pytest.raises(DrawingSheetError, match="on itself"):
        _options(section_marks=((SECTION_ID, SECTION_ID),))


def test_the_mark_travels_in_the_plate_specification(document) -> None:
    options = _options(section_marks=((SECTION_ID, PLAN_ID),))
    spec = plate_spec([PLAN_ID, SECTION_ID], options)
    assert spec["section_marks"] == [[SECTION_ID, PLAN_ID]]
    records, restored = plate_spec_options(spec)
    assert restored.section_marks == ((SECTION_ID, PLAN_ID),)
    first = compose_drawing_sheet(document, [PLAN_ID, SECTION_ID], options=options)
    again = compose_drawing_sheet(document, list(records), options=restored)
    assert first.svg_bytes == again.svg_bytes
