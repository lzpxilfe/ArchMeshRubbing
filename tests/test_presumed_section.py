"""잰 면이 아닌 자리: the cut broken where the surface under it was supplied.

Most scans reach a drafter watertight - the holes are closed before the file
is handed over - and a repair that sews a joint adds a band of its own.  Both
are surface, both are cut by the section plane, and drawn solid both say the
wall is there and this thick.  The archaeologist's rule is that they are
drawn dashed: 모르는 두께가 있는거잖아.

What is held here is that the break is a fact about the mesh and not a
flourish: the stretch is the section of the invented triangles alone, so it
lies on the cut, and where it does the solid line stops and the dashed one
begins.  The hatching stays, because the shading says where the wall is and
the dashes say how much of it was measured; those are different statements
and the drawing makes both.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.core.artifact_vector_record import VectorPath
from src.core.drawing_style import SECTION_CUT, SECTION_PRESUMED, get_preset
from src.core.drawing_sheet import (
    PRESUMED_SECTION_LABEL,
    DrawingSheetError,
    _presumed_section_runs,
    compose_drawing_sheet,
    presumed_section_title_value,
    validate_drawing_sheet_bytes,
)
from test_drawing_mirror import (
    ELEVATION_ID,
    SECTION_ID,
    SVG_NS,
    _figure,
    _find,
    _mirrored_sheet,
    _options,
    _positioned,
)


#: Two points of the fixture's own outer wall, on the section's side of the
#: fold.  A real stretch is the cut of the invented triangles; here it is
#: taken straight off the record, which is the same thing said by hand.
WALL_STRETCH = ((44.62, 35.0), (46.36, 44.0))


def _ids(root: Any, layer: str) -> list[str]:
    group = _find(root, f"{SVG_NS}g[@id='layer-{layer}']")
    if group is None:
        return []
    return [
        element.attrib["id"]
        for element in group.iter()
        if "d" in element.attrib and "id" in element.attrib
    ]


def _sheet(**overrides: Any):
    return _mirrored_sheet(
        presumed_section=((ELEVATION_ID, WALL_STRETCH),), **overrides
    )


def test_the_cut_is_solid_up_to_the_supplied_surface_and_dashed_across_it() -> None:
    bundle = _sheet()
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    figure = _figure(bundle.svg_bytes)

    dashed = _ids(figure, "section-presumed")
    assert dashed, "the stretch is drawn in its own layer"
    assert all(":presumed" in path_id for path_id in dashed)
    # And the same stretch is not also drawn solid: a dashed line over a
    # solid one is a solid line.
    solid = _ids(figure, "section-cut")
    assert solid, "the rest of the cut is still drawn"
    assert not set(dashed) & set(solid)
    # The cut face is still shaded: the hatch comes from the closed ring,
    # which the fold kept as an unstroked copy beside the broken line.
    assert any(path_id.endswith(":fill") for path_id in solid)
    fills = [
        element
        for element in _find(figure, f"{SVG_NS}g[@id='layer-section-cut']").iter()
        if element.attrib.get("id", "").endswith(":fill")
    ]
    assert fills and all(
        element.attrib.get("stroke") == "none" for element in fills
    ), "the hatched ring prints no boundary of its own"


def test_the_sheet_says_on_its_face_which_of_the_cut_was_not_measured() -> None:
    bundle = _sheet()
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))

    block = sidecar["presumed_section"]
    assert block["line_kind"] == SECTION_PRESUMED
    assert block["source"] == "surface_not_measured"
    (entry,) = block["entries"]
    assert entry["elevation_record_id"] == ELEVATION_ID
    assert entry["point_count"] == 2
    (row,) = [
        row
        for row in sidecar["title_block"]
        if row["label"] == PRESUMED_SECTION_LABEL
    ]
    assert row["value"] == presumed_section_title_value([(ELEVATION_ID, WALL_STRETCH)])
    assert "1곳" in row["value"]

    plain = _mirrored_sheet()
    quiet = json.loads(plain.sidecar_bytes.decode("utf-8"))
    assert "presumed_section" not in quiet
    assert not [
        row for row in quiet["title_block"] if row["label"] == PRESUMED_SECTION_LABEL
    ]


def test_a_sheet_drawn_without_the_option_is_the_same_bytes_as_before() -> None:
    assert _mirrored_sheet().svg_bytes == _mirrored_sheet(presumed_section=()).svg_bytes
    assert _sheet().svg_bytes != _mirrored_sheet().svg_bytes


def test_a_sheet_that_breaks_its_cut_must_say_so_in_its_title_block() -> None:
    from src.core.canonical_json import canonical_json_bytes  # noqa: PLC0415

    bundle = _sheet()
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    sidecar["title_block"] = [
        row
        for row in sidecar["title_block"]
        if row["label"] != PRESUMED_SECTION_LABEL
    ]
    with pytest.raises(
        DrawingSheetError, match="does not match the digest|does not say so"
    ):
        validate_drawing_sheet_bytes(bundle.svg_bytes, canonical_json_bytes(sidecar))


def test_a_stretch_that_lies_on_no_part_of_the_cut_is_refused() -> None:
    """Fail closed: a stretch taken on another plane draws nothing at all,
    and a drawing that quietly dropped it would claim the whole cut was
    measured."""

    with pytest.raises(DrawingSheetError, match="lies on no part of the cut"):
        compose_drawing_sheet(
            _positioned().document,
            [ELEVATION_ID],
            options=_options(
                mirror_sections=((ELEVATION_ID, SECTION_ID),),
                outline_reach="axis",
                presumed_section=((ELEVATION_ID, ((900.0, 900.0), (901.0, 901.0))),),
            ),
        )


def test_the_options_refuse_a_stretch_that_names_nothing_drawable() -> None:
    base = {"mirror_sections": ((ELEVATION_ID, SECTION_ID),)}
    with pytest.raises(DrawingSheetError, match="not the elevation half"):
        _options(**base, presumed_section=((SECTION_ID, WALL_STRETCH),))
    with pytest.raises(DrawingSheetError, match="at least two points"):
        _options(**base, presumed_section=((ELEVATION_ID, (WALL_STRETCH[0],)),))
    with pytest.raises(DrawingSheetError, match="must be finite numbers"):
        _options(
            **base,
            presumed_section=((ELEVATION_ID, ((0.0, 0.0), (float("nan"), 1.0))),),
        )
    with pytest.raises(DrawingSheetError, match=r"\(u_mm, v_mm\) pairs"):
        _options(**base, presumed_section=((ELEVATION_ID, ((0.0, 0.0), (1.0,))),))


def test_a_ring_that_has_to_be_broken_keeps_an_unstroked_copy_for_its_hatch() -> None:
    """The one case the fold does not already cover: a closed cut, whole on
    one side of the axis, with a presumed stretch on it."""

    square = VectorPath(
        id="cutline:path:0000",
        role="section",
        closed=True,
        points_mm=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
    )
    out, fill_only, broken = _presumed_section_runs(
        {SECTION_CUT: [square]},
        set(),
        presumed=[((10.0, 0.0), (10.0, 10.0))],
        tolerance_mm=0.05,
        preset=get_preset("provisional/v1"),
    )

    assert broken == 1
    assert fill_only == {"cutline:path:0000:fill"}
    fills = [path for path in out[SECTION_CUT] if path.id in fill_only]
    assert len(fills) == 1 and fills[0].closed
    assert fills[0].points_mm == square.points_mm
    (dashed,) = out[SECTION_PRESUMED]
    assert not dashed.closed
    assert dashed.points_mm == ((10.0, 0.0), (10.0, 10.0))
    # The three sides that were measured come back as one open chain, and
    # together with the dashed side they are the ring once round.
    solid = [path for path in out[SECTION_CUT] if path.id not in fill_only]
    assert [path.points_mm for path in solid] == [
        ((0.0, 0.0), (10.0, 0.0)),
        ((10.0, 10.0), (0.0, 10.0), (0.0, 0.0)),
    ]


def test_a_cut_that_lies_nowhere_near_the_stretch_is_left_whole() -> None:
    square = VectorPath(
        id="cutline:path:0000",
        role="section",
        closed=True,
        points_mm=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
    )
    out, fill_only, broken = _presumed_section_runs(
        {SECTION_CUT: [square]},
        set(),
        presumed=[((40.0, 0.0), (40.0, 10.0))],
        tolerance_mm=0.05,
        preset=get_preset("provisional/v1"),
    )

    assert broken == 0
    assert fill_only == set()
    assert out[SECTION_CUT] == [square]
    assert SECTION_PRESUMED not in out
