"""석기: what the pipeline built for pots and tiles does with a stone tool.

The guidelines lay a stone tool flat in the direction it was used and draw
the plan with a long section and a cross section - or all six projections
in third-angle ([K2] 2014, 석기 실측 방법; [K1] 2013 pp. 45-49).  Nothing
here turns on an axis, so nothing here needs an Align.  These tests run
the existing six-view outline, the section cut and the sheet on a flaked
tool and pin what they give: a start, not a lithic drawing - the inner
lines that show the flaking are not drawn yet, and docs/LITHIC_TRIAL.md
says so.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_vector_extractor import (
    commit_vector_computation,
    compute_artifact_cutline,
)
from src.core.artifact_vector_record import PlanarFrame
from src.core.drawing_sheet import (
    DrawingSheetOptions,
    SheetPage,
    TitleBlock,
    compose_drawing_sheet,
    validate_drawing_sheet_bytes,
)
from scan_defects import mesh_report
from synthetic_lithic import (
    BIFACE_SHAPE,
    flaked_tool,
    lithic_session,
    plan_area_mm2,
    plan_radius,
)

PLAN_ID = "record:biface-plan"
LONG_ID = "record:biface-long-section"
CROSS_ID = "record:biface-cross-section"
STAMP = "2026-09-05T00:00:01Z"

#: The long section runs the tool's length in the x-z plane; the cross
#: section cuts it at the widest point in the y-z plane.  Both are offset
#: a fraction of a millimetre so no mesh vertex lies on the plane.
LONG_SECTION = PlanarFrame(
    origin_world_mm=(0.0, 0.13, 0.0),
    u_axis_world=(1.0, 0.0, 0.0),
    v_axis_world=(0.0, 0.0, 1.0),
    normal_world=(0.0, -1.0, 0.0),
)
CROSS_SECTION = PlanarFrame(
    origin_world_mm=(0.13, 0.0, 0.0),
    u_axis_world=(0.0, 1.0, 0.0),
    v_axis_world=(0.0, 0.0, 1.0),
    normal_world=(1.0, 0.0, 0.0),
)


@pytest.fixture(scope="module")
def tool():
    return lithic_session()


def test_the_generated_tool_is_a_closed_solid_wound_outward() -> None:
    vertices, faces = flaked_tool()
    report = mesh_report(vertices, faces.astype(np.int64))
    assert report["boundary_edge_count"] == 0
    assert report["nonmanifold_edge_count"] == 0
    assert report["connected_piece_count"] == 1
    triangles = vertices[faces.astype(np.int64)]
    volume = float(
        np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
    ) / 6.0
    assert volume > 0.0
    # Hand-sized: 84 mm long, 50 wide, 24 thick.
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    assert extent[0] == pytest.approx(2.0 * BIFACE_SHAPE.half_length_mm, abs=0.1)
    # The pinch toward the tip puts the widest point a little behind the
    # middle, so the width lies between the pinched middle and the ellipse.
    assert 2.0 * plan_radius(BIFACE_SHAPE, math.pi / 2.0) <= extent[1] <= 2.0 * BIFACE_SHAPE.half_width_mm
    assert 15.0 < extent[2] < 30.0


def test_all_six_projections_draw_without_an_axis(tool) -> None:
    """[K2] asks for six views in third-angle where it can be had; here it
    can, and the plan is the plan to a tenth of a percent."""

    session, _vertices, _faces = tool
    areas = {}
    for view in ("top", "bottom", "front", "back", "left", "right"):
        outline = compute_artifact_outline(session, view, precision_grid_mm=0.01)
        assert outline.qc["component_count"] == 1
        assert outline.qc["hole_count"] == 0
        areas[view] = float(outline.qc["outline_area_mm2"])
    assert areas["top"] == pytest.approx(plan_area_mm2(BIFACE_SHAPE), rel=0.002)
    # The same plan from either side, to the grid's own noise.
    assert areas["top"] == pytest.approx(areas["bottom"], rel=1e-5)
    # The side views are the tool's thickness: far smaller than the plan,
    # and the end views smaller again.
    assert areas["front"] < areas["top"] / 1.5
    assert areas["left"] < areas["front"]


def test_the_two_sections_are_each_one_closed_ring(tool) -> None:
    session, _vertices, _faces = tool
    for frame in (LONG_SECTION, CROSS_SECTION):
        section = compute_artifact_cutline(session, frame)
        assert [path.closed for path in section.payload.paths] == [True]
    long = compute_artifact_cutline(session, LONG_SECTION)
    points = np.asarray(long.payload.paths[0].points_mm)
    # The bulb of percussion swells the ventral face near the platform end.
    platform = points[points[:, 0] < -BIFACE_SHAPE.half_length_mm * 0.4]
    tip = points[points[:, 0] > BIFACE_SHAPE.half_length_mm * 0.4]
    assert float(platform[:, 1].min()) < float(tip[:, 1].min()) - 1.0


def _bounds(payload) -> tuple[float, float, float, float]:
    points = np.vstack([np.asarray(path.points_mm) for path in payload.paths])
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def test_the_lithic_layout_puts_each_section_where_its_axis_says_and_in_register(tool) -> None:
    """[K1] p. 45: the cross section under the plan, the long section beside
    it, each aligned with the plan on the axis they share.

    This tool lies along X, so the long section (x-z) shares the plan's
    horizontal axis and goes under it, and the cross section, drawn with the
    length up the page (z across, y up), shares the plan's vertical axis and
    goes to its right.  Stand the tool along Y as the guidelines do and the
    two swap places; the rule is the register, not the name.
    """

    from src.core.drawing_sheet import DrawingSheetError
    from src.core.drawing_svg import Placement

    session, _vertices, _faces = tool
    plan = compute_artifact_outline(session, "top", precision_grid_mm=0.01)
    session = commit_vector_computation(
        session, plan, record_id=PLAN_ID, created_at=STAMP, operator="tester"
    )
    long = compute_artifact_cutline(session, LONG_SECTION)
    session = commit_vector_computation(
        session, long, record_id=LONG_ID, created_at="2026-09-05T00:00:02Z", operator="tester"
    )
    upright_cross = compute_artifact_cutline(
        session,
        PlanarFrame(
            origin_world_mm=(0.13, 0.0, 0.0),
            u_axis_world=(0.0, 0.0, 1.0),
            v_axis_world=(0.0, 1.0, 0.0),
            normal_world=(-1.0, 0.0, 0.0),
        ),
    )
    session = commit_vector_computation(
        session,
        upright_cross,
        record_id=CROSS_ID,
        created_at="2026-09-05T00:00:03Z",
        operator="tester",
    )
    flat_cross = compute_artifact_cutline(session, CROSS_SECTION)
    session = commit_vector_computation(
        session,
        flat_cross,
        record_id="record:biface-cross-flat",
        created_at="2026-09-05T00:00:04Z",
        operator="tester",
    )
    document = session.document

    def options(**overrides) -> DrawingSheetOptions:
        settings = {
            "title_block": TitleBlock(artifact_label="합성 뗀석기", rows=(("작성", "tester"),)),
            "page": SheetPage(size="A4", orientation="portrait"),
            "scale_denominator": 1.0,
        }
        settings.update(overrides)
        return DrawingSheetOptions(**settings)

    bundle = compose_drawing_sheet(
        document,
        [CROSS_ID, PLAN_ID, LONG_ID],
        options=options(plan_with_sections=(PLAN_ID, CROSS_ID, LONG_ID)),
    )
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    sidecar = json.loads(bundle.sidecar_bytes)
    assert sidecar["layout"] == {
        "below": LONG_ID,
        "kind": "plan_with_sections/v1",
        "plan": PLAN_ID,
        "right": CROSS_ID,
    }
    figures = {entry["record_id"]: entry for entry in sidecar["figures"]}
    assert [entry["record_id"] for entry in sidecar["figures"]] == [PLAN_ID, LONG_ID, CROSS_ID]

    def placement(record_id: str, payload) -> Placement:
        return Placement(
            content_bounds_mm=_bounds(payload),
            origin_mm=tuple(figures[record_id]["origin_mm"]),
            scale_denominator=1.0,
        )

    plan_at = placement(PLAN_ID, plan.payload)
    below_at = placement(LONG_ID, long.payload)
    right_at = placement(CROSS_ID, upright_cross.payload)
    # Under the plan, and the same x on the page for the same x of the tool.
    assert below_at.origin_mm[1] >= plan_at.origin_mm[1] + plan_at.height_mm
    for x in (-30.0, 0.0, 25.0):
        assert below_at.paper_xy((x, 0.0))[0] == pytest.approx(plan_at.paper_xy((x, 0.0))[0])
    # Beside the plan, and the same y on the page for the same y of the tool.
    assert right_at.origin_mm[0] >= plan_at.origin_mm[0] + plan_at.width_mm
    for y in (-20.0, 0.0, 15.0):
        assert right_at.paper_xy((0.0, y))[1] == pytest.approx(plan_at.paper_xy((0.0, y))[1])

    # A section that shares neither axis has no place under or beside the
    # plan, and the sheet must be exactly the three records.
    with pytest.raises(DrawingSheetError, match="no axis"):
        compose_drawing_sheet(
            document,
            [PLAN_ID, LONG_ID, "record:biface-cross-flat"],
            options=options(plan_with_sections=(PLAN_ID, LONG_ID, "record:biface-cross-flat")),
        )
    with pytest.raises(DrawingSheetError, match="exactly its plan"):
        compose_drawing_sheet(
            document,
            [PLAN_ID, LONG_ID],
            options=options(plan_with_sections=(PLAN_ID, CROSS_ID, LONG_ID)),
        )
    # Without the option the row layout is what it was.
    row = json.loads(
        compose_drawing_sheet(document, [PLAN_ID, LONG_ID, CROSS_ID], options=options()).sidecar_bytes
    )
    assert "layout" not in row


def test_a_plan_with_its_two_sections_composes_a_valid_sheet(tool) -> None:
    """The three drawings the guidelines ask for, on one A4 page.

    Laid out in the order given, left to right: the composer has no notion
    yet of the lithic convention that puts the cross section under the
    plan and the long section beside it.
    """

    session, _vertices, _faces = tool
    plan = compute_artifact_outline(session, "top", precision_grid_mm=0.01)
    session = commit_vector_computation(
        session, plan, record_id=PLAN_ID, created_at=STAMP, operator="tester"
    )
    session = commit_vector_computation(
        session,
        compute_artifact_cutline(session, LONG_SECTION),
        record_id=LONG_ID,
        created_at="2026-09-05T00:00:02Z",
        operator="tester",
    )
    session = commit_vector_computation(
        session,
        compute_artifact_cutline(session, CROSS_SECTION),
        record_id=CROSS_ID,
        created_at="2026-09-05T00:00:03Z",
        operator="tester",
    )
    bundle = compose_drawing_sheet(
        session.document,
        [PLAN_ID, LONG_ID, CROSS_ID],
        options=DrawingSheetOptions(
            title_block=TitleBlock(artifact_label="합성 뗀석기", rows=(("작성", "tester"),)),
            page=SheetPage(size="A4", orientation="portrait"),
            scale_denominator=1.0,
        ),
    )
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    sidecar = json.loads(bundle.sidecar_bytes)
    assert [figure["record_id"] for figure in sidecar["figures"]] == [PLAN_ID, LONG_ID, CROSS_ID]
    assert "section_loops" not in sidecar
