"""문양 내선: the incisions a normal map holds, traced as lines in a view and
drawn on the elevation.

The vessel is the one test_texture_relief builds - a smooth mesh, a normal
map with three ring grooves cut at known heights - so the answer is known:
in the front view the grooves are three horizontal lines across the wall,
at the grooves' heights, and nothing else.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.artifact_record_validation import validate_known_records
from src.core.artifact_texture_lines import (
    DEFAULT_TEXTURE_LINES_SMOOTHING_UM,
    TEXTURE_LINES_ALGORITHM,
    TEXTURE_LINES_RECORD_TYPE,
    ArtifactTextureLinesError,
    TextureLinesPayload,
    commit_texture_lines,
    compute_texture_lines,
    extract_texture_lines,
    texture_lines_payload_from_record,
    texture_lines_recipe,
    validate_texture_lines_recipe,
)
from src.core.artifact_texture_relief import read_normal_map, read_obj_texture_atlas
from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_vector_extractor import commit_vector_computation, compute_artifact_cutline
from src.core.artifact_vector_record import PlanarFrame
from src.core.drawing_sheet import (
    DrawingSheetError,
    DrawingSheetOptions,
    TitleBlock,
    compose_drawing_sheet,
    validate_drawing_sheet_bytes,
)
from src.core.project_file import load_artifact_project, save_artifact_project
from synthetic_vessel import FLOOR_MM, positioned_vessel_session
from test_texture_relief import GROOVE_HEIGHTS_MM, _write_normal_map, _write_textured_obj

STAMP = "2026-09-06T00:00:00Z"
#: The vessel stands with its measured floor at the origin, so a groove cut
#: at z on the wall is at z - FLOOR_MM in the canonical frame.
CANONICAL_GROOVE_V_MM = tuple(height - FLOOR_MM for height in GROOVE_HEIGHTS_MM)


@pytest.fixture(scope="module")
def grooved():
    session, vertices, faces = positioned_vessel_session(
        segments=96, rings=40, document_id="artifact:texture-lines"
    )
    directory = tempfile.mkdtemp()
    obj_path = Path(directory) / "vessel.obj"
    map_path = Path(directory) / "vessel_nor.png"
    _write_textured_obj(obj_path, vertices, np.asarray(faces, dtype=np.int64))
    _write_normal_map(map_path)
    atlas = read_obj_texture_atlas(obj_path)
    normal_map = read_normal_map(map_path)
    computation = compute_texture_lines(session, atlas, normal_map, view="front")
    session = commit_texture_lines(
        session, computation, record_id="record:pattern:front", created_at=STAMP, operator="tester"
    )
    return session, atlas, normal_map, computation


def _lines_mm(payload: TextureLinesPayload) -> list[np.ndarray]:
    return [np.asarray(polyline, dtype=np.float64) / 1000.0 for polyline in payload.polylines]


def test_the_three_grooves_come_out_as_three_lines_where_they_were_cut(grooved) -> None:
    """Each groove is one line across the visible wall at its own height,
    and there is no fourth: the smooth wall between them traces nothing."""

    _session, _atlas, _normal_map, computation = grooved
    lines = _lines_mm(computation.payload)
    assert len(lines) == 3
    heights = sorted(float(np.median(line[:, 1])) for line in lines)
    for found, cut in zip(heights, sorted(CANONICAL_GROOVE_V_MM)):
        assert abs(found - cut) < 0.15
    for line in lines:
        # Level to within a tenth of a millimetre along its whole length...
        assert float(line[:, 1].max() - line[:, 1].min()) < 0.1
        # ...and long: most of the chord the facing threshold lets through,
        # which at these radii is more than the wall's own radius.
        assert float(line[:, 0].max() - line[:, 0].min()) > 50.0
        # Symmetric about the axis, as the wall is.
        assert abs(float(line[:, 0].max() + line[:, 0].min())) < 1.0
    qc = computation.qc
    assert qc["line_count"] == 3
    assert qc["view"] == "front"
    assert qc["visible_face_count"] > 0
    # This map's normals point out of the wall, so a cut is a valley: the
    # valley count is the three grooves and the ridge count is next to nothing.
    assert qc["valley_pixel_count"] > 5 * qc["opposite_sign_pixel_count"]
    # The height the map integrates to is the depth that was cut, within a
    # tenth: the wider base leaves the groove its own tilt.
    assert -350 < qc["texture_relief_height_min_um_rounded"] < -200


def test_unrolled_about_the_axis_the_lines_reach_the_silhouette(grooved) -> None:
    """Traced on the view the wall foreshortens towards its edges; unrolled
    about the axis every stroke is seen face on, as the rubbing's paper sees
    it, and each line comes back through its triangle onto the elevation.
    Same grooves, same heights, longer lines - out to where the wall turns
    past the facing threshold."""

    session, atlas, normal_map, on_view = grooved
    unrolled = compute_texture_lines(
        session, atlas, normal_map, view="front", domain="axis_development"
    )
    assert unrolled.recipe["domain"] == "axis_development"
    assert unrolled.qc["domain"] == "axis_development"
    lines = _lines_mm(unrolled.payload)
    assert len(lines) == 3
    by_height = {round(float(np.median(line[:, 1]))): line for line in lines}
    view_by_height = {round(float(np.median(line[:, 1]))): line for line in _lines_mm(on_view.payload)}
    assert set(by_height) == set(view_by_height) == {round(v) for v in CANONICAL_GROOVE_V_MM}
    for height, line in by_height.items():
        assert float(line[:, 1].max() - line[:, 1].min()) < 0.1
        seen = view_by_height[height]
        # The same groove, and no shorter than the view saw it.
        assert abs(float(np.median(line[:, 1])) - float(np.median(seen[:, 1]))) < 0.1
        assert float(line[:, 0].max()) >= float(seen[:, 0].max()) - 0.5
        assert float(line[:, 0].min()) <= float(seen[:, 0].min()) + 0.5
    # It is a reading of the outside about the axis: a view along the axis
    # has no meridian to unroll from, and a session not positioned on its
    # axis cannot unroll at all.
    with pytest.raises(ArtifactTextureLinesError, match="side view"):
        compute_texture_lines(session, atlas, normal_map, view="top", domain="axis_development")
    with pytest.raises(ArtifactTextureLinesError, match="domain must be one of"):
        compute_texture_lines(session, atlas, normal_map, view="front", domain="paper")


def test_the_recipe_names_every_number_and_rebuilds_byte_for_byte(grooved) -> None:
    _session, atlas, normal_map, computation = grooved
    recipe = computation.recipe
    assert recipe["algorithm"] == TEXTURE_LINES_ALGORITHM
    assert recipe["view"] == "front"
    assert recipe["texture_relief"]["smoothing_um"] == DEFAULT_TEXTURE_LINES_SMOOTHING_UM
    assert recipe["texture_relief"]["atlas"]["sha256"] == atlas.sha256
    assert recipe["texture_relief"]["normal_map"]["sha256"] == normal_map.sha256
    assert validate_texture_lines_recipe(recipe) == recipe
    # The seed curvature defaults to twice the floor, and the link gap to
    # a millimetre; both are the recipe's numbers.
    assert recipe["detection_policy"]["curvature_seed_per_m"] == 2 * recipe["detection_policy"]["curvature_min_per_m"]
    assert recipe["detection_policy"]["link_um"] == 1000
    forged = json.loads(json.dumps(recipe))
    forged["detection_policy"]["curvature_min_per_m"] = 200
    assert validate_texture_lines_recipe(forged) != recipe
    forged["detection_policy"]["curvature_min_per_m"] = 700
    with pytest.raises(ArtifactTextureLinesError, match="must not be below"):
        validate_texture_lines_recipe(forged)
    forged["detection_policy"]["curvature_min_per_m"] = 200
    forged["detection_policy"]["extra"] = 1
    with pytest.raises(ArtifactTextureLinesError, match="exactly"):
        validate_texture_lines_recipe(forged)
    with pytest.raises(ArtifactTextureLinesError, match="view must be one of"):
        texture_lines_recipe(
            atlas, normal_map, view="sideways", source_vertex_count=10, source_face_count=10
        )
    with pytest.raises(ArtifactTextureLinesError, match="scale_mm"):
        texture_lines_recipe(
            atlas, normal_map, view="front", source_vertex_count=10, source_face_count=10, scale_mm=0.01
        )


def test_a_reading_that_traces_nothing_is_refused_and_the_files_must_match(grooved) -> None:
    session, atlas, normal_map, _computation = grooved
    with pytest.raises(ArtifactTextureLinesError, match="no incision was traced"):
        compute_texture_lines(session, atlas, normal_map, view="front", curvature_min_per_mm=500.0)
    # The recipe names the files by hash; a mesh that is not the atlas's
    # geometry is refused before anything is read.
    projection = session.materialize()
    vertices = np.asarray(projection.mesh.vertices, dtype=np.float64)
    faces = np.asarray(projection.mesh.faces, dtype=np.int64)
    recipe = texture_lines_recipe(
        atlas,
        normal_map,
        view="front",
        source_vertex_count=int(vertices.shape[0]),
        source_face_count=int(faces.shape[0]),
    )
    with pytest.raises(ArtifactTextureLinesError, match="not the texture atlas's geometry"):
        extract_texture_lines(vertices, faces[::-1], atlas, normal_map, recipe)
    with pytest.raises(ArtifactTextureLinesError, match="not a rigid motion"):
        extract_texture_lines(vertices * 1.01, faces, atlas, normal_map, recipe)


def test_the_record_reopens_verified_and_a_tampered_line_is_caught(grooved) -> None:
    session, _atlas, _normal_map, computation = grooved
    record = session.document.record_index["record:pattern:front"]
    assert record.type == TEXTURE_LINES_RECORD_TYPE
    payload = texture_lines_payload_from_record(record)
    assert payload == computation.payload
    validate_known_records(session.document)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "vessel.amr"
        save_artifact_project(path, session.document)
        reopened = load_artifact_project(path)
    again = texture_lines_payload_from_record(reopened.record_index["record:pattern:front"])
    assert again == payload

    tampered = record.to_dict()
    lines = tampered["extensions"]["org.archmeshrubbing:texture-lines-v1"]["payload"]["polylines"]
    lines[0][0][1] += 1000
    from src.core.artifact_document import DerivedRecord

    with pytest.raises(ArtifactTextureLinesError, match="SHA-256"):
        texture_lines_payload_from_record(DerivedRecord.from_dict(tampered))


def test_the_lines_go_on_the_elevation_and_on_no_other_figure(grooved) -> None:
    """The reading is of the front; the front outline gets it as 내선, on the
    elevation half of a mirrored figure, and a left outline gets nothing."""

    session, _atlas, _normal_map, computation = grooved
    for view in ("front", "left"):
        outline = compute_artifact_outline(session, view, precision_grid_mm=0.05)
        session = commit_vector_computation(
            session, outline, record_id=f"record:outline:{view}", created_at=STAMP, operator="tester"
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
        session, cutline, record_id="record:section:front", created_at=STAMP, operator="tester"
    )
    options = DrawingSheetOptions(
        title_block=TitleBlock(artifact_label="시험 토기", rows=()),
        texture_line_records=("record:pattern:front",),
        mirror_sections=(("record:outline:front", "record:section:front"),),
    )
    bundle = compose_drawing_sheet(
        session.document, ["record:outline:front", "record:outline:left"], options=options
    )
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    svg = bundle.svg_bytes.decode("utf-8")
    sidecar = json.loads(bundle.sidecar_bytes)
    block = sidecar["texture_lines"]
    assert block["records"] == [
        {
            "line_count": 3,
            "payload_sha256": computation.payload.sha256,
            "recipe_hash": session.document.record_index["record:pattern:front"].recipe_hash,
            "record_id": "record:pattern:front",
            "view": "front",
        }
    ]
    assert block["drawn"] == [
        {
            "figure_record_id": "record:outline:front",
            "line_kind": "outline_hole",
            "polyline_count": "3",
            "record_id": "record:pattern:front",
            "view": "front",
        }
    ]
    # The lines are on the page, on the elevation half only: every drawn
    # point is left of the axis, so the section half stays a section.
    pieces = [piece for piece in svg.split('id="mirror:left:texture-line:')[1:]]
    assert len(pieces) == 3
    for piece in pieces:
        d = piece.split(' d="')[1].split('"')[0]
        xs = [float(token) for token in d.replace("M", " ").replace("L", " ").split()[0::2]]
        assert max(xs) <= 12.0 + 1e-6 + 60.0  # left of the axis at figure x + half width
    assert 'id="mirror:right:texture-line:' not in svg

    # Without the option, the sheet is byte for byte the one it always was.
    plain = compose_drawing_sheet(
        session.document,
        ["record:outline:front", "record:outline:left"],
        options=DrawingSheetOptions(
            title_block=TitleBlock(artifact_label="시험 토기", rows=()),
            mirror_sections=(("record:outline:front", "record:section:front"),),
        ),
    )
    assert "texture_lines" not in json.loads(plain.sidecar_bytes)
    assert "texture-line:" not in plain.svg_bytes.decode("utf-8")

    with pytest.raises(DrawingSheetError, match="not a texture lines reading"):
        compose_drawing_sheet(
            session.document,
            ["record:outline:front"],
            options=DrawingSheetOptions(
                title_block=TitleBlock(artifact_label="시험 토기", rows=()),
                texture_line_records=("record:outline:front",),
            ),
        )


def test_the_stroke_rule_reads_the_grooves_as_the_paper_would(grooved) -> None:
    """The other reading: a sheet of paper wider than the incision lies
    across it and never reaches its floor, so under the paper the incision
    is one ribbon, and the stroke is that ribbon's centre line.  Same three
    grooves, same heights, on the view and unrolled; the recipe carries the
    paper's numbers and none of the curvature's, and the payload names each
    line's pattern (schema 1.1.0) - three lone rings are loose, not a
    pattern."""

    from src.core.artifact_texture_lines import (
        TEXTURE_LINES_PATTERN_RULE,
        TEXTURE_LINES_PATTERNS_SCHEMA_VERSION,
        TEXTURE_LINES_STROKE_RULE,
    )

    session, atlas, normal_map, _valley = grooved
    for domain in ("view", "axis_development"):
        strokes = compute_texture_lines(
            session, atlas, normal_map, view="front", domain=domain, rule=TEXTURE_LINES_STROKE_RULE
        )
        lines = _lines_mm(strokes.payload)
        assert len(lines) == 3
        heights = sorted(float(np.median(line[:, 1])) for line in lines)
        for found, cut in zip(heights, sorted(CANONICAL_GROOVE_V_MM)):
            assert abs(found - cut) < 0.15
        for line in lines:
            assert float(line[:, 1].max() - line[:, 1].min()) < 0.25
            assert float(line[:, 0].max() - line[:, 0].min()) > 50.0
        detection = strokes.recipe["detection_policy"]
        assert detection["valley"] == TEXTURE_LINES_STROKE_RULE
        assert detection["pattern"] == TEXTURE_LINES_PATTERN_RULE
        assert set(detection) == {
            "close_um", "depth_um", "incision_sign", "line_smoothing_um", "link_um",
            "min_length_um", "orientation_um", "pattern", "pattern_gap_um", "spur_um",
            "straightness_min_percent", "valley", "window_um",
        }
        assert detection["window_um"] == 2500 and detection["depth_um"] == 80
        assert validate_texture_lines_recipe(strokes.recipe) == strokes.recipe
        payload = strokes.payload
        assert payload.schema_version == TEXTURE_LINES_PATTERNS_SCHEMA_VERSION
        assert payload.pattern_of == (-1, -1, -1)
        assert payload.patterns == () and payload.bands == ()
        assert strokes.qc["pattern_count"] == 0 and strokes.qc["loose_line_count"] == 3
        assert strokes.qc["seam_line_count"] == 0 and strokes.qc["band_count"] == 0
        assert strokes.qc["closed_or_wandering_chain_count"] == 0
        assert strokes.qc["stroke_pixel_count"] > strokes.qc["skeleton_pixel_count"] > 0
    # A stroke recipe with a curvature key, or a valley recipe with a paper
    # key, is not the recipe it claims to be.
    forged = json.loads(json.dumps(strokes.recipe))
    forged["detection_policy"]["scale_um"] = 300
    with pytest.raises(ArtifactTextureLinesError, match="exactly"):
        validate_texture_lines_recipe(forged)
    with pytest.raises(ArtifactTextureLinesError, match="rule must be one of"):
        texture_lines_recipe(
            atlas, normal_map, view="front", source_vertex_count=10, source_face_count=10, rule="paper"
        )
    # Nothing under a paper that must be a metre deep: refused, and it says
    # which number to lower.
    with pytest.raises(ArtifactTextureLinesError, match="no stroke was traced"):
        compute_texture_lines(
            session, atlas, normal_map, view="front", rule=TEXTURE_LINES_STROKE_RULE, depth_mm=9.0
        )
    # The record reopens with its patterns.
    session = commit_texture_lines(
        session, strokes, record_id="record:strokes:front", created_at=STAMP, operator="tester"
    )
    reopened = texture_lines_payload_from_record(session.document.record_index["record:strokes:front"])
    assert reopened == strokes.payload
    directory = tempfile.mkdtemp()
    path = Path(directory) / "strokes.amr"
    save_artifact_project(path, session.document)
    again = load_artifact_project(path)
    validate_known_records(again)
    assert texture_lines_payload_from_record(again.record_index["record:strokes:front"]) == strokes.payload


def test_strokes_are_grouped_into_patterns_by_direction_and_neighbourhood() -> None:
    """The eye's grouping, as a rule: strokes that run the same way within
    a few millimetres of one another are one pattern; a stroke across them
    is not, and neither is a stroke off by itself.  Patterns are numbered
    from the top of the wall down."""

    from src.core.artifact_texture_lines import _group_strokes, _is_open_stroke

    def stroke(x: float, y: float, angle_deg: float, length: float = 3.0) -> np.ndarray:
        direction = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
        return np.array([x, y]) + np.outer(np.linspace(-0.5, 0.5, 7) * length, direction)

    upper = [stroke(2.0 * i, 40.0, 60.0) for i in range(6)]  # a row leaning right, high up
    lower = [stroke(2.0 * i, 20.0, 120.0) for i in range(6)]  # a row leaning left, lower
    across = [stroke(6.0, 20.5, 30.0)]  # one stroke across the lower row
    alone = [stroke(40.0, 30.0, 60.0)]  # far from everything
    pattern_of, patterns = _group_strokes(upper + lower + across + alone)
    assert pattern_of == [0] * 6 + [1] * 6 + [-1, -1]
    assert [pattern["line_count"] for pattern in patterns] == [6, 6]
    assert patterns[0]["direction_deg"] == 60 and patterns[1]["direction_deg"] == 120
    # Patterns lying at one height form one band; the bands run down the
    # wall, and a pattern belongs to exactly one.
    from src.core.artifact_texture_lines import _bands_of, _is_seam

    bands = _bands_of(patterns, [(38_000, 42_000), (18_000, 22_000)], gap_um=3_000)
    assert bands == [
        {"directions_deg": [60], "height_um_max": 42_000, "height_um_min": 38_000, "patterns": [0]},
        {"directions_deg": [120], "height_um_max": 22_000, "height_um_min": 18_000, "patterns": [1]},
    ]
    merged = _bands_of(patterns, [(38_000, 42_000), (36_000, 40_000)], gap_um=3_000)
    assert merged == [
        {"directions_deg": [60, 120], "height_um_max": 42_000, "height_um_min": 36_000, "patterns": [0, 1]}
    ]
    # A long loose line that wanders is a seam or a crack, not a stroke;
    # a long straight one is a stroke, and a short wanderer is just loose.
    wander = np.array([[0.0, 0.0], [2.0, 5.0], [4.0, 0.0], [6.0, 5.0], [8.0, 0.0], [10.0, 5.0], [12.0, 0.0]])
    assert _is_seam(wander, min_length_mm=12.0, straightness_max=0.6)
    assert not _is_seam(np.array([[0.0, 0.0], [8.0, 0.0], [16.0, 0.0]]), min_length_mm=12.0, straightness_max=0.6)
    assert not _is_seam(wander[:3], min_length_mm=12.0, straightness_max=0.6)
    # A ring is never a stroke, and a wanderer is not one of this pattern.
    ring = [(0, 0), (0, 1), (1, 2), (2, 2), (3, 1), (3, 0), (2, -1), (1, -1), (0, 0)]
    assert not _is_open_stroke(ring, straightness_min=0.0)
    hook = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 2), (2, 1)]
    assert _is_open_stroke(hook, straightness_min=0.0)
    assert not _is_open_stroke(hook, straightness_min=0.8)
    straight = [(0, i) for i in range(9)]
    assert _is_open_stroke(straight, straightness_min=0.95)


def test_each_pattern_is_its_own_group_on_the_sheet_and_can_be_struck_out(grooved) -> None:
    """The program reads, the archaeologist decides: on the sheet every
    pattern of a reading is one `<g>` inside the 내선 layer, the loose
    strokes another, so a reviewer can strike one out as a whole; and a
    pattern struck out at composition is left off and named in the sidecar."""

    from src.core.artifact_texture_lines import TEXTURE_LINES_STROKE_RULE

    session, atlas, normal_map, _valley = grooved
    strokes = compute_texture_lines(
        session, atlas, normal_map, view="front", rule=TEXTURE_LINES_STROKE_RULE
    )
    session = commit_texture_lines(
        session, strokes, record_id="record:strokes:front", created_at=STAMP, operator="tester"
    )
    outline = compute_artifact_outline(session, "front", precision_grid_mm=0.05)
    session = commit_vector_computation(
        session, outline, record_id="record:outline:front", created_at=STAMP, operator="tester"
    )
    options = DrawingSheetOptions(
        title_block=TitleBlock(artifact_label="시험 토기", rows=()),
        texture_line_records=("record:strokes:front",),
    )
    bundle = compose_drawing_sheet(session.document, ["record:outline:front"], options=options)
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    svg = bundle.svg_bytes.decode("utf-8")
    sidecar = json.loads(bundle.sidecar_bytes)
    # Three loose lines in one group, each still its own path.
    assert svg.count('<g id="texture-pattern:record:strokes:front:loose">') == 1
    assert svg.count('id="texture-line:record:strokes:front:loose:') == 3
    assert svg.index('<g id="texture-pattern:') < svg.index('id="texture-line:record:strokes:front:loose:00000"')
    assert sidecar["texture_lines"]["records"][0]["pattern_count"] == 0
    assert sidecar["texture_lines"]["drawn"][0]["hidden_patterns"] == ""
    assert sidecar["texture_lines"]["drawn"][0]["drawn_polyline_count"] == "3"
    assert "hidden_patterns" not in sidecar["texture_lines"]

    struck = compose_drawing_sheet(
        session.document,
        ["record:outline:front"],
        options=DrawingSheetOptions(
            title_block=TitleBlock(artifact_label="시험 토기", rows=()),
            texture_line_records=("record:strokes:front",),
            texture_line_hidden_patterns=(("record:strokes:front", -1),),
        ),
    )
    validate_drawing_sheet_bytes(struck.svg_bytes, struck.sidecar_bytes)
    assert "texture-line:" not in struck.svg_bytes.decode("utf-8")
    struck_sidecar = json.loads(struck.sidecar_bytes)
    assert struck_sidecar["texture_lines"]["hidden_patterns"] == [
        {"pattern": -1, "record_id": "record:strokes:front"}
    ]
    assert struck_sidecar["texture_lines"]["drawn"][0]["hidden_patterns"] == "-1"
    assert struck_sidecar["texture_lines"]["drawn"][0]["drawn_polyline_count"] == "0"
    with pytest.raises(DrawingSheetError, match="not among texture_line_records"):
        DrawingSheetOptions(
            title_block=TitleBlock(artifact_label="시험 토기", rows=()),
            texture_line_hidden_patterns=(("record:strokes:front", 0),),
        )
    # -2 names the seams; below that nothing is named.
    DrawingSheetOptions(
        title_block=TitleBlock(artifact_label="시험 토기", rows=()),
        texture_line_records=("record:strokes:front",),
        texture_line_hidden_patterns=(("record:strokes:front", -2),),
    )
    with pytest.raises(DrawingSheetError, match="pattern index"):
        DrawingSheetOptions(
            title_block=TitleBlock(artifact_label="시험 토기", rows=()),
            texture_line_records=("record:strokes:front",),
            texture_line_hidden_patterns=(("record:strokes:front", -3),),
        )
    assert sidecar["texture_lines"]["drawn"][0]["seam_line_count"] == "0"
    assert sidecar["texture_lines"]["drawn"][0]["band_count"] == "0"
    assert sidecar["texture_lines"]["records"][0]["band_count"] == 0
    assert sidecar["texture_lines"]["drawn"][0]["straightened_polyline_count"] == "0"


def test_a_pattern_stroke_is_drawn_as_the_clean_segment_it_is_when_asked() -> None:
    """The draftsman draws a combed row as parallel strokes, not as a
    tracing of each.  With stroke_straightening_deg set, a straight stroke
    of a pattern is drawn as one segment of its own length through its own
    middle, turned to its neighbours' median direction; a curved stroke, a
    loose stroke and a seam are left as traced; and the sheet says so."""

    import math

    from src.core.artifact_texture_lines import TextureLinesPayload
    from src.core.drawing_sheet import (
        MAX_STROKE_STRAIGHTENING_DEG,
        Interpretation,
        _fitted_segment,
        _straightened_strokes,
    )

    # A jittery stroke: 6 mm long at 45 degrees, wobbling 0.1 mm.
    def wobbly(x0: float, y0: float, angle_deg: float, length_mm: float, wobble: float):
        direction = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
        normal = np.array([-direction[1], direction[0]])
        return [
            tuple(int(round(v * 1000.0)) for v in np.array([x0, y0]) + direction * t + normal * wobble * math.sin(t * 7.0))
            for t in np.linspace(0.0, length_mm, 12)
        ]

    # Six strokes a millimetre apart, so each has the whole row as neighbours.
    lines = [wobbly(1.0 * i, 0.0, 45.0 + (i % 3 - 1) * 4.0, 6.0, 0.1) for i in range(6)]
    hook = wobbly(20.0, 0.0, 45.0, 6.0, 0.0)
    hook = hook + [(hook[-1][0] + 400, hook[-1][1] - 300)]  # a hook off the end
    curve = [(30_000 + int(2_000 * math.cos(t)), int(2_000 * math.sin(t))) for t in np.linspace(0.0, 2.0, 15)]
    # One stroke the tracer dropped for half a millimetre: two fragments
    # on one line at 45 degrees, 3 mm each, 0.5 mm apart along it.
    fragments = [wobbly(40.0, 0.0, 45.0, 3.0, 0.0), wobbly(40.0 + 3.5 / math.sqrt(2.0), 3.5 / math.sqrt(2.0), 45.0, 3.0, 0.0)]
    payload = TextureLinesPayload(
        schema_version="1.1.0",
        view="front",
        polylines=tuple(tuple(line) for line in lines) + (tuple(hook), tuple(curve)) + tuple(tuple(f) for f in fragments),
        pattern_of=(0,) * 10,
        patterns=({"direction_deg": 45, "line_count": 10},),
        bands=({"directions_deg": [45], "height_um_max": 10_000, "height_um_min": -1_000, "patterns": [0]},),
    )
    segments = _straightened_strokes(payload, angle_deg=15.0)
    assert set(segments) == set(range(7)) | {8, 9}  # the strokes and the fragments; not the curve
    from src.core.drawing_sheet import _joined_strokes

    joined, absorbed = _joined_strokes(payload, segments)
    # The six strokes a millimetre apart stay six; the fragments become one
    # stroke from end to end, 6.5 mm long, and the second is absorbed.
    assert absorbed == {9} and set(joined) == set(range(7)) | {8}
    (x0, y0), (x1, y1) = joined[8]
    assert abs(math.hypot(x1 - x0, y1 - y0) - 6.5) < 0.1
    assert abs(math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0 - 45.0) < 0.5
    segments = {index: segments[index] for index in range(7)}
    angles = []
    for index in range(6):
        (x0, y0), (x1, y1) = segments[index]
        angles.append(math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0)
        assert abs(math.hypot(x1 - x0, y1 - y0) - 6.0) < 0.3
    # All six turned to one direction, the row's median.
    assert max(angles) - min(angles) < 1e-6 and abs(angles[0] - 45.0) < 1.0
    # The hook was cut off the end before the fit.
    (x0, y0), (x1, y1) = segments[6]
    assert abs(math.hypot(x1 - x0, y1 - y0) - 6.0) < 0.5
    assert _fitted_segment(np.asarray(curve, dtype=np.float64) / 1000.0, tolerance_mm=0.3) is None
    # Declared on the sheet, bounded, and off by default.
    assert Interpretation(stroke_straightening_deg=15.0).title_row()[1] == "획 직선화 15°"
    assert Interpretation().to_dict()["stroke_straightening_deg"] == 0.0
    with pytest.raises(DrawingSheetError, match="stroke_straightening_deg"):
        Interpretation(stroke_straightening_deg=MAX_STROKE_STRAIGHTENING_DEG + 1.0)


def test_the_ridge_rule_reads_two_strokes_where_the_threshold_reads_one(grooved) -> None:
    """In a combed row a millimetre apart the ribbons under the paper touch,
    and the centre line of what is deeper than a threshold runs between the
    two strokes as one.  The ridge rule takes the stroke to be the line of
    greatest depth across the run, so two grooves a millimetre apart are
    two lines at their own rows; on the grooved vessel it reads the same
    three rings; and its recipe carries no closing gap."""

    from src.core.artifact_texture_lines import (
        TEXTURE_LINES_RIDGE_RULE,
        TEXTURE_LINES_STROKE_RULE,
        _ridge_points,
        _stroke_points,
    )

    # 20 x 20 mm of flat wall at 10 px/mm with two horizontal grooves 1 mm
    # apart, 0.3 mm deep, each 0.3 mm wide (sigma), running across the middle.
    rows = np.arange(200, dtype=np.float64)[:, None]
    cols = np.arange(200, dtype=np.float64)[None, :]
    profile = np.exp(-((rows - 95.0) ** 2) / 18.0) + np.exp(-((rows - 105.0) ** 2) / 18.0)
    signed = -0.3 * profile * ((cols >= 30) & (cols <= 170))
    good = np.ones_like(signed, dtype=bool)
    paper = dict(pixels_per_mm=10, window_mm=2.5, depth_mm=0.08, spur_mm=0.8)
    from scipy.ndimage import label

    ridge, ridge_qc = _ridge_points(signed, good, orientation_mm=1.5, **paper)
    centre, centre_qc = _stroke_points(signed, good, close_mm=0.5, **paper)
    eight = np.ones((3, 3), dtype=bool)
    ridge_labels, ridge_count = label(ridge, structure=eight)
    centre_labels, centre_count = label(centre, structure=eight)
    assert ridge_count == 2 and centre_count == 1
    for index in (1, 2):
        at = np.flatnonzero((ridge_labels == index).any(axis=1))
        assert at.size <= 3  # one row, give or take the staircase
        assert abs(float(at.mean()) - 95.0) < 1.5 or abs(float(at.mean()) - 105.0) < 1.5
        assert (ridge_labels == index).any(axis=0).sum() > 100
    # The centre line of the merged ribbon lies between the grooves.
    between = np.flatnonzero((centre_labels == 1).any(axis=1))
    assert 97.0 < float(between.mean()) < 103.0
    assert ridge_qc["stroke_pixel_count"] == centre_qc["stroke_pixel_count"] > 0
    assert 0 < ridge_qc["skeleton_pixel_count"] < ridge_qc["stroke_pixel_count"]

    session, atlas, normal_map, _valley = grooved
    strokes = compute_texture_lines(
        session, atlas, normal_map, view="front", rule=TEXTURE_LINES_RIDGE_RULE
    )
    lines = _lines_mm(strokes.payload)
    assert len(lines) == 3
    heights = sorted(float(np.median(line[:, 1])) for line in lines)
    for found, cut in zip(heights, sorted(CANONICAL_GROOVE_V_MM)):
        assert abs(found - cut) < 0.15
    for line in lines:
        assert float(line[:, 1].max() - line[:, 1].min()) < 0.25
        assert float(line[:, 0].max() - line[:, 0].min()) > 50.0
    detection = strokes.recipe["detection_policy"]
    assert detection["valley"] == TEXTURE_LINES_RIDGE_RULE
    assert "close_um" not in detection and detection["orientation_um"] == 2500
    assert validate_texture_lines_recipe(strokes.recipe) == strokes.recipe
    assert strokes.payload.pattern_of == (-1, -1, -1)
    assert strokes.qc["orientation_dropped_chain_count"] == 0
    # A ridge recipe with a closing gap, or a centre-line recipe without
    # one, is not the recipe it claims to be.
    forged = json.loads(json.dumps(strokes.recipe))
    forged["detection_policy"]["close_um"] = 500
    with pytest.raises(ArtifactTextureLinesError, match="exactly"):
        validate_texture_lines_recipe(forged)
    forged["detection_policy"]["valley"] = TEXTURE_LINES_STROKE_RULE
    assert validate_texture_lines_recipe(forged) == forged
    del forged["detection_policy"]["close_um"]
    with pytest.raises(ArtifactTextureLinesError, match="exactly"):
        validate_texture_lines_recipe(forged)
    session = commit_texture_lines(
        session, strokes, record_id="record:ridge:front", created_at=STAMP, operator="tester"
    )
    assert texture_lines_payload_from_record(session.document.record_index["record:ridge:front"]) == strokes.payload


def test_a_painted_line_is_drawn_along_its_centre_and_a_painted_band_by_its_edges(grooved) -> None:
    """Porcelain is not rubbed: its decoration is paint, and the base colour
    map is to the painted bowl what the normal map is to the incised pot.
    On the grooved vessel's atlas a colour map paints a thin gold ring at
    one height and a wide gold band at another.  Read with the ridge rule
    from the colour map, the ring is one line at its height and the band
    two lines at its edges; the grooves, which the colour map does not
    show, are not read.  The recipe carries the paint block and no relief
    block, and refuses the wrong files."""

    from PIL import Image

    from src.core.artifact_texture_lines import TEXTURE_LINES_RIDGE_RULE
    from src.core.artifact_texture_paint import (
        TEXTURE_PAINT_CHROMA_BLUE_OVER_RED,
        ColourMap,
        paint_of,
        read_colour_map,
    )
    from synthetic_vessel import HEIGHT_MM
    from test_texture_relief import MAP_SIDE

    session, atlas, normal_map, _valley = grooved
    directory = tempfile.mkdtemp()
    map_path = Path(directory) / "vessel_bc.png"
    rgb = np.full((MAP_SIDE, MAP_SIDE, 3), 245, dtype=np.uint8)
    z = HEIGHT_MM * (1.0 - (np.arange(MAP_SIDE) + 0.5) / MAP_SIDE)  # row 0 is the top
    gold = np.array([200, 160, 60], dtype=np.uint8)
    line_height, band_low, band_high = 40.0, 18.0, 22.0
    rgb[np.abs(z - line_height) <= 0.15] = gold  # a 0.3 mm line
    rgb[(z >= band_low) & (z <= band_high)] = gold  # a 4 mm band
    Image.fromarray(rgb, mode="RGB").save(map_path)
    colour_map = read_colour_map(map_path)
    assert isinstance(colour_map, ColourMap) and colour_map.width == MAP_SIDE
    painted = compute_texture_lines(
        session,
        atlas,
        None,
        colour_map=colour_map,
        view="front",
        domain="axis_development",
        rule=TEXTURE_LINES_RIDGE_RULE,
        depth_mm=0.2,
        orientation_mm=0.0,
        straightness_min_percent=0,
        min_length_mm=5.0,
        band_mm=0.8,
    )
    lines = _lines_mm(painted.payload)
    heights = sorted(round(float(np.median(line[:, 1])) - 0.0, 1) for line in lines)
    expected = sorted(v - FLOOR_MM for v in (line_height, band_low, band_high))
    assert len(lines) == 3, heights
    for found, cut in zip(heights, expected):
        assert abs(found - cut) < 0.4, (heights, expected)
    for line in lines:
        assert float(line[:, 1].max() - line[:, 1].min()) < 0.5
        assert float(line[:, 0].max() - line[:, 0].min()) > 40.0
    recipe = painted.recipe
    assert "texture_paint" in recipe and "texture_relief" not in recipe
    assert recipe["texture_paint"]["chroma"] == "red_over_blue/v1"
    assert recipe["texture_paint"]["band_um"] == 800
    assert recipe["texture_paint"]["colour_map"]["sha256"] == colour_map.sha256
    assert validate_texture_lines_recipe(recipe) == recipe
    qc = painted.qc
    assert qc["texture_paint_wide_area_count"] == 1
    assert qc["texture_paint_painted_pixel_count"] > 0
    assert "texture_relief_covered_pixel_count" not in qc
    # The chroma is what the map says: gold is red over blue, not the reverse.
    assert float(paint_of(gold[None, None, :], "red_over_blue/v1")[0, 0]) > 0.5
    assert float(paint_of(gold[None, None, :], TEXTURE_PAINT_CHROMA_BLUE_OVER_RED)[0, 0]) == 0.0
    # Both blocks, or the wrong map, are refused.
    forged = json.loads(json.dumps(recipe))
    forged["texture_relief"] = _valley.recipe["texture_relief"]
    with pytest.raises(ArtifactTextureLinesError, match="exactly"):
        validate_texture_lines_recipe(forged)
    forged = json.loads(json.dumps(recipe))
    forged["texture_paint"]["chroma"] = "gold/v1"
    with pytest.raises(ArtifactTextureLinesError, match="chroma"):
        validate_texture_lines_recipe(forged)
    with pytest.raises(ArtifactTextureLinesError, match="none was given"):
        extract_texture_lines(
            np.asarray(session.materialize().mesh.vertices), np.asarray(session.materialize().mesh.faces),
            atlas, normal_map, recipe,
        )
    other = ColourMap(rgb=colour_map.rgb, sha256="0" * 64, byte_length=colour_map.byte_length)
    with pytest.raises(ArtifactTextureLinesError, match="not the one the recipe names"):
        extract_texture_lines(
            np.asarray(session.materialize().mesh.vertices), np.asarray(session.materialize().mesh.faces),
            atlas, None, recipe, colour_map=other,
        )
    with pytest.raises(ArtifactTextureLinesError, match="normal map or a colour map"):
        texture_lines_recipe(atlas, None, view="front", source_vertex_count=10, source_face_count=10)
    # The record reopens as any texture lines record does.
    session = commit_texture_lines(
        session, painted, record_id="record:paint:front", created_at=STAMP, operator="tester"
    )
    assert texture_lines_payload_from_record(session.document.record_index["record:paint:front"]) == painted.payload
    path = Path(directory) / "painted.amr"
    save_artifact_project(path, session.document)
    validate_known_records(load_artifact_project(path))
