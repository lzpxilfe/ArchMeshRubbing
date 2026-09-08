"""What the scan did not measure, said as such: presumed lines and the
staircase fold.

A bottle scanned from outside has no wall thickness.  The section is a
profile line; the inside is known only down the neck; the floor's height
comes from a ruler.  The archaeologist's rule: draw to where it is known,
continue the measured inner wall a little as a dashed line, put a dashed
floor where the ruler found it, and say so on the sheet.  And when a
motif sits on the axis, step the fold round it - in a staircase where
the wall narrows above the motif.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_vector_extractor import commit_vector_computation, compute_artifact_cutline
from src.core.artifact_vector_record import PlanarFrame
from src.core.drawing_sheet import (
    PRESUMED_LABEL,
    DrawingSheetError,
    _dashes,
    compose_drawing_sheet,
    presumed_title_value,
    validate_drawing_sheet_bytes,
)
from test_drawing_mirror import (
    ELEVATION_ID,
    SECTION_ID,
    SVG_NS,
    _axis_x,
    _figure,
    _find,
    _mirrored_sheet,
    _options,
    _points,
    _rim_v,
)

FRONT = PlanarFrame(
    origin_world_mm=(0.0, 0.0, 0.0), u_axis_world=(1.0, 0.0, 0.0), v_axis_world=(0.0, 0.0, 1.0), normal_world=(0.0, -1.0, 0.0)
)


def _paths(root: Any, id_part: str) -> list[list[tuple[float, float]]]:
    return [_points(el) for el in root.iter() if id_part in el.attrib.get("id", "") and "d" in el.attrib]


def test_a_dashed_line_is_dashes_with_gaps_and_a_cut_last_dash() -> None:
    pieces = _dashes((0.0, 0.0), (10.0, 0.0), dash_mm=3.0, gap_mm=1.0)
    assert [(round(a[0], 6), round(b[0], 6)) for a, b in pieces] == [(0.0, 3.0), (4.0, 7.0), (8.0, 10.0)]
    assert _dashes((0.0, 0.0), (1.0, 0.0), dash_mm=3.0, gap_mm=1.0) == [((0.0, 0.0), (1.0, 0.0))]
    assert _dashes((0.0, 0.0), (0.5, 0.0), dash_mm=3.0, gap_mm=1.0) == [], "under a quarter dash is nothing"
    assert _dashes((0.0, 0.0), (0.0, 0.0), dash_mm=3.0, gap_mm=1.0) == []


def test_a_presumed_floor_is_a_dashed_level_line_from_the_axis_and_the_sheet_says_so() -> None:
    """On the hollow vessel a floor at 30 mm runs from the axis across the
    cavity, dashed, to a paper millimetre short of the inner wall; the
    title block carries the 추정 row and the sidecar the entry."""

    floor = 30.0
    bundle = _mirrored_sheet(presumed_lines=((ELEVATION_ID, "floor", floor, 0.0),))
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    figure = _figure(bundle.svg_bytes)
    axis_x = _axis_x(figure)
    dashes = _paths(figure, "presumed:floor:00:")
    assert len(dashes) >= 4, "a dashed line is several pieces"
    ys = {round(y, 4) for dash in dashes for _x, y in dash}
    assert len(ys) == 1, "the floor is level"
    xs = sorted(x for dash in dashes for x, _y in dash)
    assert abs(xs[0] - axis_x) < 1e-6, "it starts on the axis"
    # The inner wall at 30 mm is 37.7 mm from the axis on the vessel; at
    # 1:2 that is 18.9 paper mm, and the line stops a paper millimetre short.
    assert 14.0 < xs[-1] - axis_x < 18.0, xs[-1] - axis_x
    # Dashes are in the cut's layer, with gaps between them.
    layer = _find(figure, f"{SVG_NS}g[@id='layer-section-cut']")
    assert any("presumed:floor" in el.attrib.get("id", "") for el in layer.iter())
    starts = sorted(dash[0][0] for dash in dashes)
    assert all(later - earlier > 0.9 for earlier, later in zip(starts, starts[1:]))
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    (entry,) = sidecar["presumed_lines"]["entries"]
    assert entry["kind"] == "floor" and entry["height_um"] == 30000 and entry["length_um"] == 0
    assert entry["piece_count"] == len(dashes) and 14000 < entry["reach_um"] < 40000
    assert sidecar["presumed_lines"]["source"] == "archaeologist"
    (row,) = [row for row in sidecar["title_block"] if row["label"] == PRESUMED_LABEL]
    assert row["value"] == presumed_title_value([(ELEVATION_ID, "floor", 30.0, 0.0)]) == "바닥 30 mm"
    # A shorter floor is the length asked for.
    short = _mirrored_sheet(presumed_lines=((ELEVATION_ID, "floor", floor, 10.0),))
    (short_xs) = sorted(x for dash in _paths(_figure(short.svg_bytes), "presumed:floor:00:") for x, _y in dash)
    assert abs((short_xs[-1] - short_xs[0]) - 5.0) < 0.1, "10 mm at 1:2 is 5 paper mm"
    # The sheet without presumed lines has no row and no block.
    plain = json.loads(_mirrored_sheet().sidecar_bytes.decode("utf-8"))
    assert "presumed_lines" not in plain and not [row for row in plain["title_block"] if row["label"] == PRESUMED_LABEL]


def test_a_sheet_that_draws_presumed_lines_must_say_so_in_its_title_block() -> None:
    bundle = _mirrored_sheet(presumed_lines=((ELEVATION_ID, "floor", 30.0, 0.0),))
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    sidecar["title_block"] = [row for row in sidecar["title_block"] if row["label"] != PRESUMED_LABEL]
    from src.core.canonical_json import canonical_json_bytes  # noqa: PLC0415

    forged = canonical_json_bytes(sidecar)
    with pytest.raises(DrawingSheetError, match="does not match the digest|does not say so"):
        validate_drawing_sheet_bytes(bundle.svg_bytes, forged)


def test_the_options_refuse_a_presumed_line_that_names_nothing_drawable() -> None:
    with pytest.raises(DrawingSheetError, match="not the elevation half"):
        _options(mirror_sections=((ELEVATION_ID, SECTION_ID),), presumed_lines=((SECTION_ID, "floor", 30.0, 0.0),))
    with pytest.raises(DrawingSheetError, match="kind must be one of"):
        _options(mirror_sections=((ELEVATION_ID, SECTION_ID),), presumed_lines=((ELEVATION_ID, "roof", 30.0, 0.0),))
    with pytest.raises(DrawingSheetError, match="needs a positive length"):
        _options(mirror_sections=((ELEVATION_ID, SECTION_ID),), presumed_lines=((ELEVATION_ID, "wall_on", 30.0, 0.0),))
    with pytest.raises(DrawingSheetError, match="must be a finite number"):
        _options(mirror_sections=((ELEVATION_ID, SECTION_ID),), presumed_lines=((ELEVATION_ID, "floor", float("nan"), 0.0),))
    with pytest.raises(DrawingSheetError, match="length_mm must be from 0"):
        _options(mirror_sections=((ELEVATION_ID, SECTION_ID),), presumed_lines=((ELEVATION_ID, "floor", 30.0, 900.0),))


def _open_shell_session():
    """The mirror vessel without its inner wall and floor: an outside, a
    base, and the rim's lip - a bottle scanned from outside, whose cut is
    one open profile line ending inside the lip."""

    from pathlib import Path  # noqa: PLC0415

    from src.core.artifact_session import ArtifactSession  # noqa: PLC0415
    from src.core.mesh_import_recipe import current_mesh_import_recipe  # noqa: PLC0415
    from src.core.mesh_loader import MeshData  # noqa: PLC0415
    from src.core.source_identity import SourceFingerprint  # noqa: PLC0415
    from test_drawing_mirror import FLOOR_ID, RIM_ID, WALL_MM, _commit_circle, _outer_radius, _vessel  # noqa: PLC0415

    vertices, faces, rim_points, floor_points = _vessel()
    centres = vertices[faces].mean(axis=1)
    radius = np.hypot(centres[:, 0], centres[:, 1])
    outer_here = np.array([_outer_radius(float(z)) for z in centres[:, 2]])
    # Inner-wall faces and the floor lie well inside the outer radius; the
    # lip band at the rim and the base disc at zero stay.
    inner = (radius < outer_here - 0.5 * WALL_MM) & (centres[:, 2] > 0.5) & (centres[:, 2] < 89.0)
    kept = faces[~inner]
    mesh = MeshData(
        vertices=vertices, faces=kept, unit="mm", filepath=Path("/source/shell.ply"),
        source_identity=SourceFingerprint(sha256="b" * 64, size_bytes=8192, mtime_ns=1, original_name="shell.ply", format="ply"),
        source_format="ply", source_import_recipe=current_mesh_import_recipe("ply"),
    )
    session = ArtifactSession.create_from_source(
        mesh, resolved_source_path="/source/shell.ply", unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"}, handedness="right",
        software_version="presumed-test", operator="tester", created_at="2026-09-09T00:00:00Z",
        document_id="artifact:shell", metadata_revision_id="metadata:shell", align_revision_id="align:shell",
    )
    # The floor is gone with the inside; the lower circle sits on the base disc.
    base_points = [np.array([p[0], p[1], 0.0]) for p in floor_points]
    session = _commit_circle(session, vertices, kept, base_points, record_id=FLOOR_ID, created_at="2026-09-09T00:00:01Z")
    session = _commit_circle(session, vertices, kept, rim_points, record_id=RIM_ID, created_at="2026-09-09T00:00:02Z")
    session = session.commit_axis_alignment(
        top_record_id=RIM_ID, bottom_record_id=FLOOR_ID, operator="tester", created_at="2026-09-09T00:00:03Z", revision_id="align:axis"
    )
    session = commit_vector_computation(
        session, compute_artifact_outline(session, "front", precision_grid_mm=0.5),
        record_id=ELEVATION_ID, created_at="2026-09-09T00:01:00Z", operator="tester",
    )
    cut = compute_artifact_cutline(session, FRONT)
    assert not any(path.closed for path in cut.payload.paths), "an open shell cuts to open lines"
    return commit_vector_computation(session, cut, record_id=SECTION_ID, created_at="2026-09-09T00:02:00Z", operator="tester")


def test_the_measured_inner_wall_goes_on_as_a_dashed_line_from_its_open_end() -> None:
    """The open shell's cut ends inside the lip.  wall_on continues that end
    downward, after one gap, in the end's own direction, for the length
    the archaeologist gives; the floor line still runs from the axis to the
    outer wall, which is the only wall there is."""

    session = _open_shell_session()
    rim = _rim_v()
    bundle = compose_drawing_sheet(
        session.document, [ELEVATION_ID],
        options=_options(
            mirror_sections=((ELEVATION_ID, SECTION_ID),), outline_reach="axis",
            presumed_lines=((ELEVATION_ID, "wall_on", rim - 2.0, 12.0), (ELEVATION_ID, "floor", 8.0, 0.0)),
        ),
    )
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    figure = _figure(bundle.svg_bytes)
    axis_x = _axis_x(figure)
    on = _paths(figure, "presumed:wall_on:00:")
    assert len(on) >= 3
    points = [p for dash in on for p in dash]
    assert all(x > axis_x + 1.0 for x, _y in points), "it goes on inside the section's side"
    # This shell's lip is a flat band, so the measured wall ends running
    # level toward the axis, and the presumed line goes on that way.
    assert max(y for _x, y in points) - min(y for _x, y in points) < 0.05, "level, like the lip it continues"
    left = min(x for x, _y in points)
    right = max(x for x, _y in points)
    assert 5.0 < right - left <= 6.05, "12 mm at 1:2 is 6 paper mm, less nothing"
    floor = _paths(figure, "presumed:floor:01:")
    assert floor and abs(min(x for dash in floor for x, _y in dash) - axis_x) < 1e-6
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    kinds = [(entry["kind"], entry["piece_count"] > 0, "not_drawn" in entry) for entry in sidecar["presumed_lines"]["entries"]]
    assert kinds == [("wall_on", True, False), ("floor", True, False)]
    assert sidecar["presumed_lines"]["entries"][0]["from_um"][1] > (rim - 12.0) * 1000
    (row,) = [row for row in sidecar["title_block"] if row["label"] == PRESUMED_LABEL]
    assert row["value"] == f"안벽 {rim - 2.0:g} mm에서 12 mm 더 · 바닥 8 mm"


def test_two_steps_that_meet_are_one_staircase() -> None:
    """Two jogs sharing a height draw the centre line stepping from the
    lower reach straight to the higher, not back to the axis between."""

    rim = _rim_v()
    jogs = ((ELEVATION_ID, rim - 20.0, rim - 10.0, 12.0), (ELEVATION_ID, rim - 10.0, rim - 2.0, 6.0))
    bundle = _mirrored_sheet(mirror_jogs=jogs)
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    figure = _figure(bundle.svg_bytes)
    axis_x = _axis_x(figure)
    axis = _points(_find(figure, f"{SVG_NS}g[@id='layer-center-axis']/{SVG_NS}path"))
    offsets = [round(x - axis_x, 2) for x, _y in axis]
    # The line is drawn from the bottom up: bottom, out to 6 (12 mm at
    # 1:2), up, in to 3 (6 mm) without touching the axis, up, back, top.
    assert offsets == [0.0, 0.0, 6.0, 6.0, 3.0, 3.0, 0.0, 0.0], offsets
    separate = _mirrored_sheet(mirror_jogs=((ELEVATION_ID, rim - 20.0, rim - 12.0, 12.0), (ELEVATION_ID, rim - 10.0, rim - 2.0, 6.0)))
    apart = [round(x - _axis_x(_figure(separate.svg_bytes)), 2) for x, _y in _points(_find(_figure(separate.svg_bytes), f"{SVG_NS}g[@id='layer-center-axis']/{SVG_NS}path"))]
    assert apart == [0.0, 0.0, 6.0, 6.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0], apart
