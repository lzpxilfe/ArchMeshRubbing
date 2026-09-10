"""전개 띠를 입면에 붙이기: 가운데 선부터 유물의 끝까지.

A strip taken right round a pot is longer than any elevation of it, and a
drafter does not squeeze it to fit.  The paper goes down with one edge on
the centre line and the scissors follow the vessel's own edge, so the motif
stands at true size over the width the artifact has when you look at it.

What is held here is that the paste is measured, not placed by eye: every
dot sits at the height its row was read from and no dot stands past the
wall's radius there, both read off the record's own profile of the
development.  A record made before that profile was recorded is refused
rather than pasted at a guessed height.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any

import pytest

from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_relief_shade import (
    commit_relief_shade,
    compute_relief_shade,
)
from src.core.artifact_vector_extractor import commit_vector_computation
from src.core.drawing_sheet import (
    DrawingSheetError,
    DrawingSheetOptions,
    SheetPage,
    TitleBlock,
    compose_drawing_sheet,
    validate_drawing_sheet_bytes,
)
from synthetic_vessel import positioned_vessel_session
from test_relief_shade import _petal

STAMP = "2026-09-10T00:00:00Z"


@pytest.fixture(scope="module")
def pasted():
    session, _vertices, _faces = positioned_vessel_session(
        segments=96, rings=45, relief=_petal, document_id="artifact:pasted"
    )
    session = commit_vector_computation(
        session,
        compute_artifact_outline(session, "front", precision_grid_mm=0.5),
        record_id="record:front",
        created_at="2026-09-10T00:01:00Z",
        operator="tester",
    )
    computation = compute_relief_shade(session, domain="axis_development/v1")
    session = commit_relief_shade(
        session, computation, record_id="record:strip", created_at=STAMP, operator="tester"
    )
    return session, computation.raster


def _sheet(pasted, **overrides: Any):
    session, raster = pasted
    settings: dict[str, Any] = {
        "title_block": TitleBlock(artifact_label="전개 붙이기 시험 호"),
        "page": SheetPage(size="A4", orientation="portrait"),
        "scale_denominator": 2.0,
    }
    settings.update(overrides)
    return compose_drawing_sheet(
        session.document,
        ["record:front"],
        options=DrawingSheetOptions(**settings),
        rasters={"record:strip": raster} if overrides.get("relief_developments_on_axis") else None,
    )


def _dots(svg_bytes: bytes) -> list[tuple[float, float]]:
    root = ET.fromstring(svg_bytes)
    return [
        (float(element.attrib["cx"]), float(element.attrib["cy"]))
        for element in root.iter()
        if element.tag.endswith("circle")
    ]


def test_the_strip_is_laid_from_the_centre_line_and_cut_at_the_wall(pasted) -> None:
    session, raster = pasted
    shade_record = session.document.record_index["record:strip"]
    bundle = _sheet(pasted, relief_developments_on_axis=(("record:strip", "record:front"),))
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    dots = _dots(bundle.svg_bytes)
    assert dots, "the strip's ink reaches the elevation"

    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    (drawn,) = sidecar["relief_stipples"]["drawn"]
    assert drawn["record_id"] == "record:strip"
    assert drawn["figure_record_id"] == "record:front"
    assert drawn["half"] == "elevation"
    assert int(drawn["dot_count"]) == len(dots)
    # The strip is 2πr round and the elevation half covers a quarter of that
    # arc at most, so most of the round is cut off - that is the scissors,
    # and the sheet counts what they took.
    assert int(drawn["dropped_section_side_count"]) > 0

    figure = sidecar["figures"][0]
    origin_x, origin_y = figure["origin_mm"]
    # Nothing the paste added stands outside the figure the lines drew: the
    # ink is on the pot, not beside it.
    assert all(origin_x - 1e-6 <= x <= origin_x + float(figure["width_mm"]) + 1e-6 for x, _y in dots)
    assert all(
        origin_y - 1e-6 <= y <= origin_y + float(figure["height_mm"]) + 1e-6 for _x, y in dots
    )

    scale = 2.0
    profile = shade_record.qc["development_height_profile_um"]
    radii = shade_record.qc["development_radius_profile_um"]
    # Vertically the paste is the artifact's own heights, not the strip's arc:
    # the ink spans no more than the band the rows were read from.
    span = max(y for _x, y in dots) - min(y for _x, y in dots)
    assert span <= (profile[-1] - profile[0]) / 1000.0 / scale + 1e-6
    # Across, the scissors: no dot stands further from the centre line than
    # the wall's own radius there.
    reach = max(x for x, _y in dots) - min(x for x, _y in dots)
    assert reach <= max(radii) / 1000.0 / scale + 1e-6


def test_a_sheet_that_pastes_nothing_keeps_its_bytes(pasted) -> None:
    assert _sheet(pasted).svg_bytes == _sheet(pasted, relief_developments_on_axis=()).svg_bytes
    assert "relief_stipples" not in json.loads(_sheet(pasted).sidecar_bytes.decode("utf-8"))


def test_the_paste_is_refused_where_it_would_be_a_guess(pasted) -> None:
    session, raster = pasted
    options = DrawingSheetOptions(
        title_block=TitleBlock(artifact_label="전개 붙이기 시험 호"),
        page=SheetPage(size="A4", orientation="portrait"),
        scale_denominator=2.0,
        relief_developments_on_axis=(("record:strip", "record:front"),),
    )
    # A strip pasted inside a figure is not also a figure of its own.
    with pytest.raises(DrawingSheetError, match="must not also be a figure of its own"):
        compose_drawing_sheet(
            session.document,
            ["record:front", "record:strip"],
            options=options,
            rasters={"record:strip": raster},
        )
    # A shade read in a view has no meridian to lay along the axis.
    view_shade = compute_relief_shade(session, view="front")
    with_view = commit_relief_shade(
        session, view_shade, record_id="record:view", created_at=STAMP, operator="tester"
    )
    with pytest.raises(DrawingSheetError, match="not on the axis development"):
        compose_drawing_sheet(
            with_view.document,
            ["record:front"],
            options=DrawingSheetOptions(
                title_block=TitleBlock(artifact_label="전개 붙이기 시험 호"),
                page=SheetPage(size="A4", orientation="portrait"),
                scale_denominator=2.0,
                relief_developments_on_axis=(("record:view", "record:front"),),
            ),
            rasters={"record:view": view_shade.raster},
        )
    # The same shade cannot be both stippled in place and pasted on the axis.
    with pytest.raises(DrawingSheetError, match="not both"):
        DrawingSheetOptions(
            title_block=TitleBlock(artifact_label="전개 붙이기 시험 호"),
            scale_denominator=2.0,
            relief_stipples=(("record:strip", "record:front"),),
            relief_developments_on_axis=(("record:strip", "record:front"),),
        )
