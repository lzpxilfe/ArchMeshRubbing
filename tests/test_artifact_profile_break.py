"""단면 꺾임: the corners of a vessel's profile, read once and drawn as the
lines that run right round - the foot ring's root, the edge where the foot
stands proud, a shoulder's edge - each one straight line at its height on
any figure whose plane holds the axis."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_profile_break import (
    PROFILE_BREAK_RECORD_TYPE,
    PROFILE_BREAK_SURFACE_INWARD,
    PROFILE_BREAK_SURFACE_OUTWARD,
    ArtifactProfileBreakError,
    ProfileBreak,
    ProfileBreakPayload,
    commit_profile_breaks,
    compute_artifact_profile_breaks,
    profile_break_payload_from_record,
    profile_break_recipe,
    profile_break_surface,
    validate_profile_break_recipe,
)
from src.core.artifact_record_validation import validate_known_records
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_extractor import commit_vector_computation, compute_artifact_cutline
from src.core.artifact_vector_record import PlanarFrame
from src.core.drawing_sheet import DrawingSheetError, DrawingSheetOptions, SheetPage, TitleBlock, compose_drawing_sheet, validate_drawing_sheet_bytes
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.project_file import load_artifact_project, save_artifact_project
from src.core.source_identity import SourceFingerprint
from synthetic_vessel import _commit_circle

STAMP = "2026-09-06T00:00:00Z"
#: The profile, (radius, height) up the outside: a foot ring standing proud
#: under a body that swells to a sharp shoulder edge and closes to the rim.
FOOT_EDGE_Z = 3.0
FOOT_ROOT_Z = 8.0
SHOULDER_Z = 60.0
PROFILE = [
    (30.0, 0.0),  # foot bottom
    (33.0, FOOT_EDGE_Z),  # the foot's outer edge, a convex corner
    (33.0, FOOT_ROOT_Z),  # up the foot's outer face
    (42.0, FOOT_ROOT_Z + 0.5),  # the root: the body steps out over the foot (concave)
    (48.0, 20.0),
    (54.0, 40.0),
    (58.0, SHOULDER_Z),  # the shoulder's edge, a convex corner
    (50.0, 75.0),
    (46.0, 90.0),  # rim
]


def _footed_vessel(segments: int = 48) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """A hollow vessel of revolution with the profile above and a 4 mm wall."""

    phase = math.pi / segments
    # The outside, then the inside 4 mm in, joined at the rim and the floor.
    outside = [(r, z) for r, z in PROFILE]
    inside_corners = [(r - 4.0, z) for r, z in reversed(PROFILE) if z >= FOOT_ROOT_Z + 0.5]
    # The inside's straight runs are subdivided so the inner wall has rings
    # enough to read a profile up; no corner is added.
    inside: list[tuple[float, float]] = []
    for (r0, z0), (r1, z1) in zip(inside_corners, inside_corners[1:]):
        for t in np.linspace(0.0, 1.0, 4, endpoint=False):
            inside.append((r0 + (r1 - r0) * float(t), z0 + (z1 - z0) * float(t)))
    inside.append(inside_corners[-1])
    rings = outside + inside
    vertices: list[list[float]] = []
    for r, z in rings:
        for k in range(segments):
            a = phase + 2.0 * math.pi * k / segments
            vertices.append([r * math.cos(a), r * math.sin(a), z])
    faces: list[list[int]] = []
    for ring in range(len(rings) - 1):
        lower, upper = ring * segments, (ring + 1) * segments
        for k in range(segments):
            n = (k + 1) % segments
            faces.append([lower + k, lower + n, upper + n])
            faces.append([lower + k, upper + n, upper + k])
    # Cap the foot bottom and the inner floor with fans.
    bottom = len(vertices)
    vertices.append([0.0, 0.0, 0.0])
    for k in range(segments):
        faces.append([bottom, (k + 1) % segments, k])
    floor_ring = len(rings) - 1
    floor_centre = len(vertices)
    vertices.append([0.0, 0.0, FOOT_ROOT_Z + 0.5])
    for k in range(segments):
        faces.append([floor_centre, floor_ring * segments + k, floor_ring * segments + (k + 1) % segments])
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    rim = [v[(len(outside) - 1) * segments + k] for k in (0, segments // 3, 2 * segments // 3)]
    floor = [v[floor_ring * segments + k] for k in (0, segments // 3, 2 * segments // 3)]
    return v, f, rim, floor


@pytest.fixture(scope="module")
def footed() -> ArtifactSession:
    vertices, faces, rim, floor = _footed_vessel()
    mesh = MeshData(
        vertices=vertices, faces=faces, unit="mm", filepath=Path("/source/footed.ply"),
        source_identity=SourceFingerprint(sha256="9" * 64, size_bytes=4096, mtime_ns=1, original_name="footed.ply", format="ply"),
        source_format="ply", source_import_recipe=current_mesh_import_recipe("ply"),
    )
    session = ArtifactSession.create_from_source(
        mesh, resolved_source_path="/source/footed.ply", unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"}, handedness="right",
        software_version="break-test", operator="tester", created_at=STAMP,
        document_id="artifact:footed", metadata_revision_id="metadata:footed", align_revision_id="align:footed",
    )
    session = _commit_circle(session, vertices, faces, floor, record_id="record:floor", created_at="2026-09-06T00:00:01Z")
    session = _commit_circle(session, vertices, faces, rim, record_id="record:rim", created_at="2026-09-06T00:00:02Z")
    session = session.commit_axis_alignment(
        top_record_id="record:rim", bottom_record_id="record:floor", operator="tester",
        created_at="2026-09-06T00:00:03Z", revision_id="align:axis",
    )
    session = commit_vector_computation(
        session, compute_artifact_outline(session, "front", precision_grid_mm=0.5),
        record_id="record:front", created_at="2026-09-06T00:01:00Z", operator="tester",
    )
    return commit_vector_computation(
        session,
        compute_artifact_cutline(session, PlanarFrame(origin_world_mm=(0.0, 0.0, 0.0), u_axis_world=(1.0, 0.0, 0.0), v_axis_world=(0.0, 0.0, 1.0), normal_world=(0.0, -1.0, 0.0))),
        record_id="record:section", created_at="2026-09-06T00:02:00Z", operator="tester",
    )


def _floor_offset(session: ArtifactSession) -> float:
    """The canonical frame stands the floor circle at height zero."""

    return FOOT_ROOT_Z + 0.5


def test_the_foot_and_the_shoulder_come_out_as_the_corners_they_are(footed) -> None:
    """Three corners: the foot's outer edge (convex), the foot's root where
    the body steps out over it (concave), the shoulder's edge (convex) - at
    their heights, at their radii, and nothing on the smooth wall between."""

    computation = compute_artifact_profile_breaks(footed, angle_min_deg=25, span_um=2_000)
    payload = computation.payload
    offset = _floor_offset(footed)
    found = [(item.height_um / 1000.0 + offset, item.radius_um / 1000.0, item.convex) for item in payload.breaks]
    expected = [(FOOT_EDGE_Z, 33.0, True), (FOOT_ROOT_Z, 33.0, False), (SHOULDER_Z, 58.0, True)]
    assert len(found) == 3, found
    for (height, radius, convex), (want_z, want_r, want_convex) in zip(found, expected):
        assert abs(height - want_z) < 1.0, (found, expected)
        assert abs(radius - want_r) < 1.5, (found, expected)
        assert convex is want_convex
    qc = computation.qc
    assert qc["break_count"] == 3 and qc["convex_break_count"] == 2 and qc["concave_break_count"] == 1
    assert validate_profile_break_recipe(computation.recipe) == computation.recipe
    # The recipe is its numbers and nothing else.
    forged = json.loads(json.dumps(computation.recipe))
    forged["detection_policy"]["angle_min_deg"] = 30
    assert validate_profile_break_recipe(forged) != computation.recipe
    forged["detection_policy"]["keep"] = "first/v1"
    with pytest.raises(ArtifactProfileBreakError, match="production contract"):
        validate_profile_break_recipe(forged)
    with pytest.raises(ArtifactProfileBreakError, match="two height bins"):
        profile_break_recipe(height_bin_um=1000, span_um=1500)
    # Too strict an angle finds nothing, and says which number to lower.
    with pytest.raises(ArtifactProfileBreakError, match="lower angle_min_deg"):
        compute_artifact_profile_breaks(footed, angle_min_deg=160)
    # A payload is ordered up the wall, and a turn of nothing is no break.
    with pytest.raises(ArtifactProfileBreakError, match="turns one way"):
        ProfileBreak(height_um=1, radius_um=1000, turn_millidegrees=0, revolution_spread_um=0)
    with pytest.raises(ArtifactProfileBreakError, match="share one height"):
        ProfileBreakPayload(
            schema_version="1.0.0",
            breaks=(
                ProfileBreak(height_um=5, radius_um=1000, turn_millidegrees=30_000, revolution_spread_um=0),
                ProfileBreak(height_um=5, radius_um=1200, turn_millidegrees=-30_000, revolution_spread_um=0),
            ),
            profile_bin_count=10, profile_minimum_height_um=0, profile_maximum_height_um=100,
        )


def test_the_reading_is_a_record_that_reopens_and_draws_as_lines_round_the_pot(footed) -> None:
    """Committed, saved and reopened the reading is the same; on the sheet
    each corner is one straight inner line at its height across the
    elevation, cut at the axis on a mirrored figure, and the sidecar names
    the reading.  A record a plan view cannot show draws nothing there."""

    computation = compute_artifact_profile_breaks(footed, angle_min_deg=25, span_um=2_000)
    session = commit_profile_breaks(footed, computation, record_id="record:breaks", created_at=STAMP, operator="tester")
    record = session.document.record_index["record:breaks"]
    assert record.type == PROFILE_BREAK_RECORD_TYPE
    assert profile_break_payload_from_record(record) == computation.payload
    path = Path(tempfile.mkdtemp()) / "footed.amr"
    save_artifact_project(path, session.document)
    reopened = load_artifact_project(path)
    validate_known_records(reopened)
    assert profile_break_payload_from_record(reopened.record_index["record:breaks"]) == computation.payload

    options = DrawingSheetOptions(
        title_block=TitleBlock(artifact_label="굽 달린 시험 호"),
        page=SheetPage(size="A4", orientation="portrait"), scale_denominator=2.0,
        mirror_sections=(("record:front", "record:section"),),
        break_records=("record:breaks",),
    )
    bundle = compose_drawing_sheet(session.document, ["record:front"], options=options)
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    root = ET.fromstring(bundle.svg_bytes)
    lines = [el for el in root.iter() if "profile-break:record:breaks:" in el.attrib.get("id", "")]
    assert len(lines) == 3
    for el in lines:
        tokens = el.attrib["d"].replace("M", " ").replace("L", " ").split()
        ys = [float(tokens[i]) for i in range(1, len(tokens), 2)]
        xs = [float(tokens[i]) for i in range(0, len(tokens), 2)]
        assert max(ys) - min(ys) < 1e-6, "a corner runs level round the pot"
        assert max(xs) - min(xs) > 5.0
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    assert sidecar["profile_breaks"]["records"][0]["break_count"] == 3
    assert sidecar["profile_breaks"]["records"][0]["surface"] == PROFILE_BREAK_SURFACE_OUTWARD
    assert sidecar["profile_breaks"]["drawn"] == [
        {
            "break_count": "3", "broken_count": "0", "chosen_count": "0", "figure_record_id": "record:front",
            "half": "elevation", "omitted_count": "0", "record_id": "record:breaks", "solid_count": "3",
            "surface": PROFILE_BREAK_SURFACE_OUTWARD,
        }
    ]
    assert [(line["index"], line["style"], line["source"]) for line in sidecar["profile_breaks"]["records"][0]["lines"]] == [
        (0, "solid", "rule"), (1, "solid", "rule"), (2, "solid", "rule"),
    ]
    assert sidecar["profile_breaks"]["not_drawn"] == []
    # Without the option the sheet has no such block.
    plain = compose_drawing_sheet(session.document, ["record:front"], options=DrawingSheetOptions(
        title_block=TitleBlock(artifact_label="굽 달린 시험 호"), page=SheetPage(size="A4", orientation="portrait"),
        scale_denominator=2.0, mirror_sections=(("record:front", "record:section"),)))
    assert "profile_breaks" not in json.loads(plain.sidecar_bytes.decode("utf-8"))
    with pytest.raises(DrawingSheetError, match="not a profile break reading"):
        compose_drawing_sheet(session.document, ["record:front"], options=DrawingSheetOptions(
            title_block=TitleBlock(artifact_label="굽 달린 시험 호"), page=SheetPage(size="A4", orientation="portrait"),
            scale_denominator=2.0, break_records=("record:front",)))


def _line_xs(root: ET.Element, id_part: str) -> list[list[float]]:
    out = []
    for el in root.iter():
        if id_part in el.attrib.get("id", "") and "d" in el.attrib:
            tokens = el.attrib["d"].replace("M", " ").replace("L", " ").split()
            out.append([float(tokens[i]) for i in range(0, len(tokens), 2)])
    return out


def test_the_inside_has_corners_too_and_they_show_through_the_cut(footed) -> None:
    """Read up the inner wall, the footed vessel has one corner: the fold
    inside the shoulder, a root seen from within.  On a mirrored figure it
    is drawn on the section's half, from the axis out to the inner wall,
    while the outside's corners stay on the elevation's half; an elevation
    on its own cannot show it and says so; a section on its own draws it
    across, wall to wall, and leaves the outside's corners to its cut."""

    inside = compute_artifact_profile_breaks(
        footed, angle_min_deg=25, span_um=2_000, surface=PROFILE_BREAK_SURFACE_INWARD
    )
    offset = _floor_offset(footed)
    found = [(item.height_um / 1000.0 + offset, item.radius_um / 1000.0, item.convex) for item in inside.payload.breaks]
    assert len(found) == 1, found
    assert abs(found[0][0] - SHOULDER_Z) < 1.0 and abs(found[0][1] - 54.0) < 1.5, found
    assert found[0][2] is False, "the fold inside the shoulder is a root, not an edge"
    assert profile_break_surface(inside.recipe) == PROFILE_BREAK_SURFACE_INWARD
    with pytest.raises(ArtifactProfileBreakError, match="surface must be"):
        profile_break_recipe(surface="both/v1")

    outside = compute_artifact_profile_breaks(footed, angle_min_deg=25, span_um=2_000)
    session = commit_profile_breaks(footed, outside, record_id="record:out", created_at=STAMP, operator="tester")
    session = commit_profile_breaks(session, inside, record_id="record:in", created_at=STAMP, operator="tester")
    title = TitleBlock(artifact_label="굽 달린 시험 호")
    page = SheetPage(size="A4", orientation="portrait")

    bundle = compose_drawing_sheet(
        session.document, ["record:front"],
        options=DrawingSheetOptions(
            title_block=title, page=page, scale_denominator=2.0,
            mirror_sections=(("record:front", "record:section"),), break_records=("record:out", "record:in"),
        ),
    )
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    root = ET.fromstring(bundle.svg_bytes)
    axis_x = _line_xs(root, "mirror:center-axis")[0][0]
    outer = _line_xs(root, "profile-break:record:out:")
    inner = _line_xs(root, "profile-break:record:in:")
    assert len(outer) == 3 and len(inner) == 1
    for xs in outer:
        assert max(xs) <= axis_x + 1e-6, "the outside's corners stop at the fold by default"
    xs = inner[0]
    assert min(xs) >= axis_x - 1e-6, "the inside's corner starts at the axis"
    # An inner line does not touch the cut: it stops a paper millimetre
    # short of the inner wall (54 mm at 1:2 is 27 mm on paper).
    assert abs((max(xs) - min(xs)) - (54.0 / 2.0 - 1.0)) < 0.6, "and stops short of the inner wall"
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    assert [(entry["record_id"], entry["half"], entry["solid_count"], entry["broken_count"]) for entry in sidecar["profile_breaks"]["drawn"]] == [
        ("record:in", "section", "1", "0"), ("record:out", "elevation", "3", "0"),
    ]
    assert sidecar["profile_breaks"]["not_drawn"] == []
    assert sidecar["profile_breaks"]["reach"] == "axis" and sidecar["profile_breaks"]["reach_gap_paper_mm"] == 1.0
    assert sidecar["profile_breaks"]["solid_min_deg"] == 30

    # A corner that turns less than the threshold is drawn broken - once
    # under it, twice under half of it - with a paper millimetre of gap.
    # The fold inside the shoulder turns about 39 degrees: once under 60,
    # twice under 180.
    broken = compose_drawing_sheet(
        session.document, ["record:front"],
        options=DrawingSheetOptions(
            title_block=title, page=page, scale_denominator=2.0, break_solid_min_deg=60,
            mirror_sections=(("record:front", "record:section"),), break_records=("record:out", "record:in"),
        ),
    )
    root = ET.fromstring(broken.svg_bytes)
    pieces = _line_xs(root, "profile-break:record:in:000:")
    assert len(pieces) == 2, pieces
    ends = sorted((min(xs), max(xs)) for xs in pieces)
    assert abs((ends[1][0] - ends[0][1]) - 1.0) < 1e-6, "one paper millimetre of gap"
    assert ends[0][0] >= axis_x - 1e-6 and abs(ends[1][1] - (axis_x + 54.0 / 2.0 - 1.0)) < 0.6
    sidecar = json.loads(broken.sidecar_bytes.decode("utf-8"))
    assert [(entry["record_id"], entry["solid_count"], entry["broken_count"]) for entry in sidecar["profile_breaks"]["drawn"]] == [
        ("record:in", "0", "1"), ("record:out", "0", "3"),
    ]
    twice = compose_drawing_sheet(
        session.document, ["record:front"],
        options=DrawingSheetOptions(
            title_block=title, page=page, scale_denominator=2.0, break_solid_min_deg=180,
            mirror_sections=(("record:front", "record:section"),), break_records=("record:in",),
        ),
    )
    assert len(_line_xs(ET.fromstring(twice.svg_bytes), "profile-break:record:in:000:")) == 3
    with pytest.raises(DrawingSheetError, match="break_solid_min_deg must be"):
        DrawingSheetOptions(title_block=title, page=page, scale_denominator=2.0, break_solid_min_deg=200)
    # The outline's rim edge, though, runs on past the fold in the outline's
    # weight: across the cavity to a paper millimetre short of the inner wall
    # at the rim (42 mm at 1:2, less the gap); the base's edge lies under the
    # solid foot and stops at the fold.
    past = [xs for xs in _line_xs(root, ":past-axis:") if True]
    assert len(past) == 1 and min(past[0]) >= axis_x - 1e-6, past
    assert abs((max(past[0]) - axis_x) - (42.0 / 2.0 - 1.0)) < 1.0, past
    figure = sidecar["mirrored_figures"][0]
    assert figure["outline_reach"] == "section" and figure["outline_past_axis_count"] == "1"

    # Asked, the outside's corner lines run on too: the shoulder's across
    # the cavity to the inner wall (54 mm, less the gap); the foot's two lie
    # where the section is solid and stop at the fold.  Asked the other way,
    # the outline's edges stop at the fold.
    reaching = compose_drawing_sheet(
        session.document, ["record:front"],
        options=DrawingSheetOptions(
            title_block=title, page=page, scale_denominator=2.0, break_reach="section", outline_reach="axis",
            mirror_sections=(("record:front", "record:section"),), break_records=("record:out", "record:in"),
        ),
    )
    root = ET.fromstring(reaching.svg_bytes)
    outer = _line_xs(root, "profile-break:record:out:")
    past = [xs for xs in outer if max(xs) > axis_x + 1e-6]
    assert len(outer) == 4 and len(past) == 1 and min(past[0]) >= axis_x - 1e-6
    assert abs((max(past[0]) - axis_x) - (54.0 / 2.0 - 1.0)) < 1.0, past
    assert [xs for xs in _line_xs(root, ":past-axis:") if "profile-break" not in str(xs)] == past
    sidecar = json.loads(reaching.sidecar_bytes.decode("utf-8"))
    assert sidecar["profile_breaks"]["reach"] == "section"
    assert sidecar["mirrored_figures"][0]["outline_reach"] == "axis" and sidecar["mirrored_figures"][0]["outline_past_axis_count"] == "0"
    with pytest.raises(DrawingSheetError, match="break_reach must be"):
        DrawingSheetOptions(title_block=title, page=page, scale_denominator=2.0, break_reach="rim")
    with pytest.raises(DrawingSheetError, match="outline_reach must be"):
        DrawingSheetOptions(title_block=title, page=page, scale_denominator=2.0, outline_reach="rim")

    plain = compose_drawing_sheet(
        session.document, ["record:front"],
        options=DrawingSheetOptions(title_block=title, page=page, scale_denominator=2.0, break_records=("record:out", "record:in")),
    )
    root = ET.fromstring(plain.svg_bytes)
    # Three corners, each drawn as two halves from the axis outward.
    assert len(_line_xs(root, "profile-break:record:out:")) == 6
    assert _line_xs(root, "profile-break:record:in:") == []
    sidecar = json.loads(plain.sidecar_bytes.decode("utf-8"))
    assert sidecar["profile_breaks"]["not_drawn"] == [
        {
            "figure_record_id": "record:front", "reason": "interior_needs_section_half",
            "record_id": "record:in", "surface": PROFILE_BREAK_SURFACE_INWARD,
        }
    ]

    section = compose_drawing_sheet(
        session.document, ["record:section"],
        options=DrawingSheetOptions(title_block=title, page=page, scale_denominator=2.0, break_records=("record:out", "record:in")),
    )
    root = ET.fromstring(section.svg_bytes)
    assert _line_xs(root, "profile-break:record:out:") == []
    inner = _line_xs(root, "profile-break:record:in:")
    # Wall to wall as two halves from the axis, each stopping a paper
    # millimetre short of its wall.
    assert len(inner) == 2 and all(abs((max(xs) - min(xs)) - (54.0 / 2.0 - 1.0)) < 0.6 for xs in inner), inner
    sidecar = json.loads(section.sidecar_bytes.decode("utf-8"))
    assert [entry["reason"] for entry in sidecar["profile_breaks"]["not_drawn"]] == ["exterior_needs_elevation"]


def test_the_archaeologist_has_the_last_word_on_each_corner(footed) -> None:
    """The rule proposes a corner's line from its turn; a hand that has
    looked at the vessel may see it otherwise, and says so corner by
    corner - solid, broken once or twice, or left out - and the sidecar
    tells the rule's lines from the chosen ones.  A choice that names
    nothing is refused rather than lost."""

    outside = compute_artifact_profile_breaks(footed, angle_min_deg=25, span_um=2_000)
    session = commit_profile_breaks(footed, outside, record_id="record:out", created_at=STAMP, operator="tester")
    title = TitleBlock(artifact_label="굽 달린 시험 호")
    page = SheetPage(size="A4", orientation="portrait")

    def sheet(**kwargs):
        return compose_drawing_sheet(
            session.document, ["record:front"],
            options=DrawingSheetOptions(
                title_block=title, page=page, scale_denominator=2.0,
                mirror_sections=(("record:front", "record:section"),), break_records=("record:out",), **kwargs,
            ),
        )

    # All three corners turn past 30 degrees: solid by the rule.
    by_rule = sheet()
    root = ET.fromstring(by_rule.svg_bytes)
    assert len(_line_xs(root, "profile-break:record:out:")) == 3
    # The foot's edge broken twice, its root left out, the shoulder as the rule has it.
    chosen = sheet(break_styles=(("record:out", 0, "broken_twice"), ("record:out", 1, "omit")))
    validate_drawing_sheet_bytes(chosen.svg_bytes, chosen.sidecar_bytes)
    root = ET.fromstring(chosen.svg_bytes)
    assert len(_line_xs(root, "profile-break:record:out:000:")) == 3, "three pieces: broken twice"
    assert _line_xs(root, "profile-break:record:out:001:") == [], "left out"
    assert len(_line_xs(root, "profile-break:record:out:002:")) == 1, "solid, by the rule"
    sidecar = json.loads(chosen.sidecar_bytes.decode("utf-8"))
    drawn = sidecar["profile_breaks"]["drawn"][0]
    assert (drawn["solid_count"], drawn["broken_count"], drawn["omitted_count"], drawn["chosen_count"]) == ("1", "1", "1", "2")
    assert [(line["index"], line["style"], line["source"]) for line in sidecar["profile_breaks"]["records"][0]["lines"]] == [
        (0, "broken_twice", "choice"), (1, "omit", "choice"), (2, "solid", "rule"),
    ]
    assert sidecar["profile_breaks"]["not_drawn"] == []
    # The same choice, said with the same words, is the same sheet; no
    # choice at all is the rule's sheet.
    assert sheet(break_styles=(("record:out", 1, "omit"), ("record:out", 0, "broken_twice"))).svg_bytes == chosen.svg_bytes
    assert sheet(break_styles=()).svg_bytes == by_rule.svg_bytes
    # Every corner left out: the record is drawn nowhere, and the sidecar says why.
    none = sheet(break_styles=tuple(("record:out", index, "omit") for index in range(3)))
    assert _line_xs(ET.fromstring(none.svg_bytes), "profile-break:record:out:") == []
    assert [entry["reason"] for entry in json.loads(none.sidecar_bytes.decode("utf-8"))["profile_breaks"]["not_drawn"]] == ["every_corner_omitted"]

    with pytest.raises(DrawingSheetError, match="which has 3 corners"):
        sheet(break_styles=(("record:out", 3, "solid"),))
    with pytest.raises(DrawingSheetError, match="not in break_records"):
        sheet(break_styles=(("record:in", 0, "solid"),))
    with pytest.raises(DrawingSheetError, match="styles must be one of"):
        sheet(break_styles=(("record:out", 0, "dotted"),))
    with pytest.raises(DrawingSheetError, match="twice"):
        sheet(break_styles=(("record:out", 0, "solid"), ("record:out", 0, "omit")))
    with pytest.raises(DrawingSheetError, match="corner indices must be integers"):
        sheet(break_styles=(("record:out", True, "solid"),))
    with pytest.raises(DrawingSheetError, match="triples"):
        sheet(break_styles=(("record:out", 0),))
