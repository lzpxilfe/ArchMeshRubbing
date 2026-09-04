"""홈: the grooves a pot carries right round it, read from its own profile.

A groove is drawn with three lines - the recessed one broken, the two raised
ones solid - so the reading has to produce three heights, not one, and it has
to tell a groove from the two things that look like one: a dent on a single
side, and the flank of a raised cordon.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pytest

from src.core.artifact_document import RecordFreshness
from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_profile_groove import (
    ArtifactProfileGrooveError,
    PROFILE_GROOVE_RECORD_TYPE,
    ProfileGroove,
    commit_profile_grooves,
    compute_artifact_profile_grooves,
    detect_profile_grooves,
    profile_groove_payload_from_record,
    profile_groove_recipe,
    validate_profile_groove_recipe,
)
from src.core.artifact_record_validation import validate_known_records
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_extractor import (
    commit_vector_computation,
    compute_artifact_cutline,
)
from src.core.artifact_vector_record import (
    VECTOR_COORDINATE_SPACE,
    VECTOR_PAYLOAD_SCHEMA_VERSION,
    PlanarFrame,
    VectorGeometryPayload,
    VectorPath,
    VectorRecordKind,
)
from src.core.drawing_sheet import (
    DrawingSheetOptions,
    DrawingSheetError,
    SheetPage,
    TitleBlock,
    compose_drawing_sheet,
    validate_drawing_sheet_bytes,
)
from src.core.drawing_style import (
    GROOVE_TROUGH_BREAK_COUNT,
    TECHNIQUE_GROOVE_EDGE,
)
from src.core.project_file import load_artifact_project, save_artifact_project
from synthetic_vessel import hollow_vessel, positioned_vessel_session


# The synthetic builder measures height from its own base; positioning moves
# the origin to the measured floor, ten millimetres lower.
FLOOR_OFFSET_MM = 10.0
STAMP = "2026-09-04T00:00:00Z"


def one_groove(angle_rad: float, z_mm: float) -> float:
    """A single 침선 two millimetres wide and 0.4 mm deep, right round."""

    return -0.4 if abs(z_mm - 30.0) < 1.0 else 0.0


def three_grooves(angle_rad: float, z_mm: float) -> float:
    centres = (26.0, 30.0, 34.0)
    return -0.45 if min(abs(z_mm - c) for c in centres) < 0.7 else 0.0


def one_sided_dent(angle_rad: float, z_mm: float) -> float:
    """Damage, not technique: it never goes right round."""

    around = abs(((angle_rad + math.pi) % (2.0 * math.pi)) - math.pi)
    return -1.0 if abs(z_mm - 30.0) < 1.0 and around < 0.4 else 0.0


def raised_cordon(angle_rad: float, z_mm: float) -> float:
    return 0.6 if abs(z_mm - 45.0) < 1.5 else 0.0


def cordon_flanked_groove(angle_rad: float, z_mm: float) -> float:
    if 1.0 <= abs(z_mm - 30.0) < 2.0:
        return 0.3
    return -0.4 if abs(z_mm - 30.0) < 1.0 else 0.0


def _read(relief, **overrides: Any):
    vertices, faces, _rim, _floor = hollow_vessel(
        segments=96, rings=360, relief=relief
    )
    return detect_profile_grooves(
        vertices, faces, profile_groove_recipe(**overrides)
    )


def test_a_groove_that_runs_right_round_is_found_with_its_two_edges() -> None:
    payload = _read(one_groove)

    assert len(payload.grooves) == 1
    groove = payload.grooves[0]
    # The synthetic groove is centred on its own z = 30 and is 2 mm wide.
    assert abs(groove.trough_height_um / 1000.0 - 30.0) < 0.4
    assert groove.lower_edge_height_um < groove.trough_height_um
    assert groove.upper_edge_height_um > groove.trough_height_um
    assert abs(groove.depth_um - 400) <= 10
    assert abs(groove.width_um - 2_000) <= 500
    # It runs right round, so the radius barely varies with angle.
    assert groove.revolution_spread_um < 50


def test_the_pots_own_shape_is_not_a_groove() -> None:
    """This profile flares and bulges; none of that is technique."""

    with pytest.raises(ArtifactProfileGrooveError) as raised:
        _read(None)
    message = str(raised.value)
    # The refusal names the two numbers a user would reach for.
    assert "minimum_depth_um" in message
    assert "maximum_width_um" in message


def test_a_dent_on_one_side_is_not_a_groove() -> None:
    """The radius is read as a median across the revolution, so a dent that
    only touches one side never moves it far enough to count."""

    with pytest.raises(ArtifactProfileGrooveError, match="no groove"):
        _read(one_sided_dent)


def test_a_cordon_is_not_a_groove() -> None:
    """A raised band has a dip beside it, but that dip is its flank: the
    ground on one side stands far higher than the ground on the other."""

    with pytest.raises(ArtifactProfileGrooveError, match="no groove"):
        _read(raised_cordon)


def test_a_groove_between_two_cordons_is_measured_crest_to_trough() -> None:
    """Both edges stand up equally here, so it is a groove, and its depth is
    what a drafter sees: from the rims down to the bottom."""

    payload = _read(cordon_flanked_groove)

    assert len(payload.grooves) == 1
    groove = payload.grooves[0]
    assert abs(groove.depth_um - 700) <= 20
    assert groove.lower_edge_radius_um > groove.trough_radius_um
    assert groove.upper_edge_radius_um > groove.trough_radius_um


def test_grooves_come_back_in_order_and_do_not_overlap() -> None:
    payload = _read(three_grooves)

    heights = [groove.trough_height_um for groove in payload.grooves]
    assert len(heights) == 3
    assert heights == sorted(heights)
    for lower, upper in zip(payload.grooves, payload.grooves[1:]):
        assert lower.upper_edge_height_um <= upper.lower_edge_height_um
    assert payload.qc_summary()["groove_count"] == 3


def test_a_shallow_groove_waits_until_the_gate_is_lowered() -> None:
    shallow = lambda angle, z: -0.1 if abs(z - 30.0) < 1.0 else 0.0  # noqa: E731

    with pytest.raises(ArtifactProfileGrooveError, match="no groove"):
        _read(shallow)
    payload = _read(shallow, minimum_depth_um=70)
    assert len(payload.grooves) == 1
    assert abs(payload.grooves[0].depth_um - 100) <= 15


def test_the_recipe_is_a_closed_integer_contract() -> None:
    recipe = profile_groove_recipe()
    assert validate_profile_groove_recipe(recipe) == recipe

    tampered = dict(recipe)
    detection = dict(tampered["detection_policy"])
    detection["edge_asymmetry_percent"] = 90
    tampered["detection_policy"] = detection
    with pytest.raises(ArtifactProfileGrooveError, match="production contract"):
        validate_profile_groove_recipe(tampered)

    with pytest.raises(ArtifactProfileGrooveError, match="height_bin_um"):
        profile_groove_recipe(height_bin_um=0)
    # A window narrower than four bins cannot hold a groove and its two edges.
    with pytest.raises(ArtifactProfileGrooveError, match="four height bins"):
        profile_groove_recipe(height_bin_um=250, maximum_width_um=500)


def test_a_groove_cannot_be_read_off_a_hand_dragged_artifact() -> None:
    """"Right round" is a claim about the rotation axis, so it needs one."""

    session, _vertices, _faces = positioned_vessel_session(
        segments=96, rings=360, relief=one_groove
    )
    assert compute_artifact_profile_grooves(session).payload.grooves

    dragged = session.activate_parent_align()
    with pytest.raises(ArtifactProfileGrooveError, match="rotation axis"):
        compute_artifact_profile_grooves(dragged)


def test_the_record_keeps_the_reading_across_a_save() -> None:
    session, _vertices, _faces = positioned_vessel_session(
        segments=96, rings=360, relief=three_grooves
    )
    computation = compute_artifact_profile_grooves(session)
    session = commit_profile_grooves(
        session,
        computation,
        record_id="record:groove:body",
        created_at=STAMP,
        operator="tester",
    )
    record = session.document.record_index["record:groove:body"]
    assert record.type == PROFILE_GROOVE_RECORD_TYPE
    assert session.document.record_freshness(record.id) is RecordFreshness.FRESH
    validate_known_records(session.document)

    # Positioning put the origin at the measured floor, so the heights the
    # record carries are canonical ones, not the builder's.
    heights = [
        groove.trough_height_um / 1000.0 for groove in computation.payload.grooves
    ]
    assert all(
        abs(height - (centre - FLOOR_OFFSET_MM)) < 0.5
        for height, centre in zip(heights, (26.0, 30.0, 34.0))
    )

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "pot.amr"
        save_artifact_project(path, session.document)
        loaded = load_artifact_project(path)
        validate_known_records(loaded)
        reopened = profile_groove_payload_from_record(
            loaded.record_index["record:groove:body"]
        )
        assert reopened.to_dict() == computation.payload.to_dict()
        assert reopened.sha256 == computation.payload.sha256


def test_a_groove_must_sit_between_its_edges() -> None:
    with pytest.raises(ArtifactProfileGrooveError, match="between its two edges"):
        ProfileGroove(
            trough_height_um=1_000,
            trough_radius_um=40_000,
            lower_edge_height_um=2_000,
            lower_edge_radius_um=40_500,
            upper_edge_height_um=3_000,
            upper_edge_radius_um=40_500,
            depth_um=500,
            revolution_spread_um=0,
        )


def _sheet_session() -> ArtifactSession:
    session, _vertices, _faces = positioned_vessel_session(
        segments=96, rings=360, relief=three_grooves
    )
    outline = compute_artifact_outline(session, "front", precision_grid_mm=0.01)
    session = commit_vector_computation(
        session,
        outline,
        record_id="record:elevation:front",
        created_at=STAMP,
        operator="tester",
    )
    cutline = compute_artifact_cutline(
        session,
        PlanarFrame(
            origin_world_mm=(0.0, 0.0, 0.0),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 0.0, 1.0),
            normal_world=(0.0, -1.0, 0.0),
        ),
    )
    session = commit_vector_computation(
        session,
        cutline,
        record_id="record:section:front",
        created_at=STAMP,
        operator="tester",
    )
    computation = compute_artifact_profile_grooves(session)
    return commit_profile_grooves(
        session,
        computation,
        record_id="record:groove:body",
        created_at=STAMP,
        operator="tester",
    )


@pytest.fixture(scope="module")
def sheet_session() -> ArtifactSession:
    return _sheet_session()


def _options(**overrides: Any) -> DrawingSheetOptions:
    base: dict[str, Any] = {
        "title_block": TitleBlock(artifact_label="침선 토기", rows=()),
        "page": SheetPage(size="A4", orientation="portrait"),
        "scale_denominator": 1.0,
    }
    base.update(overrides)
    return DrawingSheetOptions(**base)


def test_a_groove_draws_one_broken_line_and_two_solid_ones(
    sheet_session: ArtifactSession,
) -> None:
    bundle = compose_drawing_sheet(
        sheet_session.document,
        ["record:elevation:front"],
        options=_options(groove_records=("record:groove:body",)),
    )
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    svg = bundle.svg_bytes.decode("utf-8")

    edge_layer = svg.split('id="layer-technique-groove-edge"')[1].split("</g>")[0]
    trough_layer = svg.split('id="layer-technique-groove-trough"')[1].split("</g>")[0]
    # Three grooves: two solid edges each, and a bottom broken into pieces.
    # A full elevation shows both halves, each broken about the centre.
    assert edge_layer.count("<path") == 6
    assert trough_layer.count("<path") == 3 * 2 * (GROOVE_TROUGH_BREAK_COUNT + 1)
    # The bottom line is broken by geometry, never by a dash pattern: a dash
    # pattern fixes how often a line breaks and not how many times.
    assert "stroke-dasharray" not in trough_layer


def test_a_half_elevation_keeps_the_breaks_it_was_meant_to_have(
    sheet_session: ArtifactSession,
) -> None:
    """The breaks belong to the line a drafter draws.

    A 좌 반입면 draws half the pot's width, so breaking the whole chord would
    leave half of them on the half that gets clipped away.  Each half is broken
    on its own, and the drawn half carries its full count either way.
    """

    bundle = compose_drawing_sheet(
        sheet_session.document,
        ["record:elevation:front"],
        options=_options(
            groove_records=("record:groove:body",),
            mirror_sections=(
                ("record:elevation:front", "record:section:front"),
            ),
        ),
    )
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    svg = bundle.svg_bytes.decode("utf-8")
    trough_layer = svg.split('id="layer-technique-groove-trough"')[1].split("</g>")[0]

    assert trough_layer.count("<path") == 3 * (GROOVE_TROUGH_BREAK_COUNT + 1)


def test_a_groove_line_spans_the_wall_at_its_own_height(
    sheet_session: ArtifactSession,
) -> None:
    from src.core.artifact_vector_export import profile_groove_vector_paths
    from src.core.artifact_vector_record import vector_payload_from_record

    payload = vector_payload_from_record(
        sheet_session.document.record_index["record:elevation:front"]
    )
    grooves = profile_groove_payload_from_record(
        sheet_session.document.record_index["record:groove:body"]
    ).grooves
    paths = profile_groove_vector_paths(
        payload, grooves, record_id="record:groove:body"
    )

    for groove, edges in zip(
        grooves,
        [paths[TECHNIQUE_GROOVE_EDGE][i : i + 2] for i in range(0, 6, 2)],
    ):
        for path in edges:
            points = np.asarray(path.points_mm, dtype=np.float64)
            # The frame's v is the axis, so a line sits at its own height and
            # reaches the wall's radius on both sides of the axis.
            assert np.allclose(points[:, 1], groove.lower_edge_height_um / 1000.0) or (
                np.allclose(points[:, 1], groove.upper_edge_height_um / 1000.0)
            )
            assert abs(points[:, 0].min() + points[:, 0].max()) < 1e-9


def test_a_plan_view_shows_a_groove_as_a_circle_so_gets_no_line(
    sheet_session: ArtifactSession,
) -> None:
    """Seen from above, a groove that runs right round is a circle.

    A plane the axis only leans into is refused for the same reason: the line
    there would be a foreshortened projection of the groove, claiming a width
    the artifact does not have at that height.
    """

    from src.core.artifact_outline_extractor import outline_frame
    from src.core.artifact_vector_export import profile_groove_vector_paths
    from src.core.drawing_svg import axis_profile_chord

    plan = outline_frame("top").to_dict()
    assert axis_profile_chord(plan, height_mm=20.0, radius_mm=40.0) is None

    tilted = {
        "origin_world_mm": (0.0, 0.0, 0.0),
        "u_axis_world": (1.0, 0.0, 0.0),
        "v_axis_world": (0.0, math.sin(0.3), math.cos(0.3)),
    }
    assert axis_profile_chord(tilted, height_mm=20.0, radius_mm=40.0) is None

    # An elevation gets the line, so the refusal above is about the plane and
    # not about the reading.
    elevation = outline_frame("front").to_dict()
    assert axis_profile_chord(elevation, height_mm=20.0, radius_mm=40.0) is not None

    payload = VectorGeometryPayload(
        schema_version=VECTOR_PAYLOAD_SCHEMA_VERSION,
        kind=VectorRecordKind.OUTLINE,
        coordinate_space=VECTOR_COORDINATE_SPACE,
        frame=outline_frame("top"),
        paths=(
            VectorPath(
                id="outline:component:0000:exterior",
                role="exterior",
                closed=True,
                points_mm=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
            ),
        ),
    )
    grooves = profile_groove_payload_from_record(
        sheet_session.document.record_index["record:groove:body"]
    ).grooves
    assert profile_groove_vector_paths(payload, grooves, record_id="x") == {}


def test_a_sheet_without_groove_records_is_what_it_always_was(
    sheet_session: ArtifactSession,
) -> None:
    plain = compose_drawing_sheet(
        sheet_session.document, ["record:elevation:front"], options=_options()
    )
    assert b"technique-groove" not in plain.svg_bytes
    assert b'"groove"' not in plain.sidecar_bytes

    drawn = compose_drawing_sheet(
        sheet_session.document,
        ["record:elevation:front"],
        options=_options(groove_records=("record:groove:body",)),
    )
    assert drawn.svg_bytes != plain.svg_bytes


def test_a_groove_record_must_be_ready_and_fresh_to_be_drawn(
    sheet_session: ArtifactSession,
) -> None:
    with pytest.raises(DrawingSheetError, match="does not exist"):
        compose_drawing_sheet(
            sheet_session.document,
            ["record:elevation:front"],
            options=_options(groove_records=("record:groove:missing",)),
        )
    with pytest.raises(DrawingSheetError, match="not a groove reading"):
        compose_drawing_sheet(
            sheet_session.document,
            ["record:elevation:front"],
            options=_options(groove_records=("record:elevation:front",)),
        )
