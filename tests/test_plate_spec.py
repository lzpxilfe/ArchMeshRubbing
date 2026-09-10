"""A plate carries its own specification and can be made again from it.

Everything the composer was told - the records, the scale, the fold and
its steps, the presumed lines, the lines struck out - is one closed JSON
document.  It round-trips exactly, it rides in the sheet's sidecar, the
offline validator re-proves it, and the command line makes the plate
again from a project and the spec alone.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_extractor import commit_vector_computation
from src.core.canonical_json import canonical_json_bytes
from src.core.drawing_sheet import (
    DrawingSheetError,
    DrawingSheetOptions,
    Interpretation,
    SheetPage,
    TitleBlock,
    compose_drawing_sheet,
    validate_drawing_sheet_bytes,
)
from src.core.drawing_sheet_spec import (
    PLATE_SPEC_FORMAT,
    PLATE_SPEC_SCHEMA_VERSION,
    REACH_TO_WALL,
    PlateSpecError,
    plate_spec,
    plate_spec_options,
    read_plate_spec,
    write_plate_spec,
)
from src.core.drawing_style import user_preset
from src.core.mesh_loader import MeshLoader
from src.core.project_file import save_artifact_session_project
from test_drawing_mirror import ELEVATION_ID, SECTION_ID, _options, _positioned

REPO_ROOT = Path(__file__).resolve().parent.parent
TRIANGLE_PLY = b"""ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
20 0 5
0 0 30
3 0 1 2
"""


def _rich_options() -> DrawingSheetOptions:
    """Most of what a bottle plate asks for, a user preset included."""

    return _options(
        mirror_sections=((ELEVATION_ID, SECTION_ID),),
        mirror_jogs=((ELEVATION_ID, 10.0, 30.0, 12.0), (ELEVATION_ID, 30.0, 60.0, math.inf)),
        presumed_lines=((ELEVATION_ID, "floor", 8.0, 0.0), (ELEVATION_ID, "wall_on", 80.0, 12.0)),
        outline_reach="axis",
        line_cap="butt",
        center_axis_style="dash_dot",
        scale_denominator=2.0,
        page=SheetPage(size="A4", orientation="landscape", margin_mm=10.0),
        style_preset=user_preset({"outline_visible": 0.4}),
        interpretation=Interpretation(line_smoothing_mm=1.5, straight_far_edges=False, note="시험"),
        title_block=TitleBlock(artifact_label="명세 시험", rows=(("자료", "합성"),)),
    )


def test_a_spec_round_trips_exactly_and_names_the_wall_step_by_its_word() -> None:
    options = _rich_options()
    spec = plate_spec([ELEVATION_ID], options)
    assert spec["format"] == PLATE_SPEC_FORMAT and spec["schema_version"] == PLATE_SPEC_SCHEMA_VERSION
    assert spec["mirror_jogs"][1][3] == REACH_TO_WALL, "an infinite reach is the word, not a float"
    assert isinstance(spec["style_preset"], dict), "a user preset travels in full"
    ids, again = plate_spec_options(spec)
    assert ids == [ELEVATION_ID]
    assert again.mirror_jogs[1][3] == math.inf
    assert plate_spec(ids, again) == spec, "reading and writing again is the same bytes"
    assert json.loads(canonical_json_bytes(spec)) == spec, "the spec is plain JSON"
    # A default option left out reads as its default; written back it is present.
    minimal = {"format": PLATE_SPEC_FORMAT, "schema_version": PLATE_SPEC_SCHEMA_VERSION, "records": [ELEVATION_ID], "title_block": {"artifact_label": "x"}}
    ids, defaults = plate_spec_options(minimal)
    assert defaults.scale_denominator == 1.0 and defaults.mirror_sections == ()
    assert plate_spec(ids, defaults)["scale_denominator"] == 1.0


def test_a_spec_refuses_what_the_composer_would_not_accept_and_what_it_cannot_read() -> None:
    good = plate_spec([ELEVATION_ID], _rich_options())
    for change, message in (
        ({"colour": "red"}, "unknown keys"),
        ({"format": "other"}, "format must be"),
        ({"schema_version": "9.0.0"}, "schema_version must be"),
        ({"records": []}, "at least one record"),
        ({"scale_denominator": True}, "must be a finite number"),
        ({"scale_denominator": "2"}, "must be a finite number"),
        ({"show_center_axis": 1}, "true or false"),
        ({"break_solid_min_deg": 30.5}, "must be an integer"),
        ({"style_preset": "nobody/v9"}, "preset"),
        ({"mirror_jogs": [[ELEVATION_ID, REACH_TO_WALL, 30.0, 12.0]]}, "along_from_mm must be a finite number"),
        ({"mirror_jogs": [[ELEVATION_ID, 10.0, 30.0, -1.0]]}, "not a plate the composer accepts"),
        ({"page": {"size": "A4", "sides": 2}}, "page must be an object"),
        ({"interpretation": {"note": "x", "mood": "y"}}, "interpretation must be an object"),
        ({"title_block": {"artifact_label": ""}}, "not a plate the composer accepts"),
    ):
        with pytest.raises(PlateSpecError, match=message):
            plate_spec_options({**good, **change})
    with pytest.raises(PlateSpecError, match="must be an object"):
        plate_spec_options([good])


def test_the_sheet_carries_its_spec_and_the_validator_re_proves_it() -> None:
    document = _positioned().document
    options = _options(mirror_sections=((ELEVATION_ID, SECTION_ID),), outline_reach="axis", scale_denominator=2.0)
    bundle = compose_drawing_sheet(document, [ELEVATION_ID], options=options)
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    assert sidecar["plate_spec"] == plate_spec([ELEVATION_ID], options)
    # The spec makes the same sheet again.
    ids, again = plate_spec_options(sidecar["plate_spec"])
    twice = compose_drawing_sheet(document, ids, options=again)
    assert twice.svg_bytes == bundle.svg_bytes and twice.sidecar_bytes == bundle.sidecar_bytes
    # A spec that is not canonical, or describes another sheet, is refused.
    for tamper, message in (
        ({"scale_denominator": 3.0}, "does not describe this sheet"),
        ({"records": [ELEVATION_ID, SECTION_ID]}, "does not describe this sheet"),
        ({"gutter_mm": None}, "not in its canonical form"),
        ({"colour": "red"}, "malformed"),
    ):
        forged = dict(sidecar)
        forged["plate_spec"] = {key: value for key, value in {**sidecar["plate_spec"], **tamper}.items() if value is not None or key in ("plan_over_elevation", "plan_with_sections")}
        with pytest.raises(DrawingSheetError, match=f"does not match the digest|{message}"):
            validate_drawing_sheet_bytes(bundle.svg_bytes, canonical_json_bytes(forged))


def test_a_spec_file_is_written_readably_and_read_back_strictly(tmp_path: Path) -> None:
    spec = plate_spec([ELEVATION_ID], _rich_options())
    path = write_plate_spec(tmp_path / "plate.json", spec)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("{\n  ") and "명세 시험" in text, "indented, and Korean as Korean"
    assert read_plate_spec(path) == spec
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(PlateSpecError, match="not valid JSON"):
        read_plate_spec(tmp_path / "bad.json")
    (tmp_path / "loose.json").write_text(json.dumps({**spec, "extra": 1}), encoding="utf-8")
    with pytest.raises(PlateSpecError, match="unknown keys"):
        read_plate_spec(tmp_path / "loose.json")
    with pytest.raises(PlateSpecError):
        write_plate_spec(tmp_path / "never.json", {**spec, "extra": 1})
    assert not (tmp_path / "never.json").exists()


def _embedded_project(tmp_path: Path) -> tuple[Path, str]:
    """A saved project with one outline record, its source embedded."""

    source = tmp_path / "triangle.ply"
    source.write_bytes(TRIANGLE_PLY)
    mesh = MeshLoader(default_unit="mm").load(source, unit="mm", compute_face_normals=False)
    session = ArtifactSession.create_from_source(
        mesh, resolved_source_path=str(source), unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"}, handedness="right",
        software_version="plate-spec-test", operator="tester", created_at="2026-09-09T00:00:00Z",
        document_id="artifact:plate-spec", metadata_revision_id="metadata:plate-spec", align_revision_id="align:plate-spec",
    )
    session = commit_vector_computation(
        session, compute_artifact_outline(session, "front", precision_grid_mm=0.5),
        record_id="record:plan", created_at="2026-09-09T00:01:00Z", operator="tester",
    )
    project = tmp_path / "triangle.amr"
    save_artifact_session_project(project, session)
    return project, "record:plan"


def test_the_command_line_makes_the_plate_again_from_the_project_and_the_spec(tmp_path: Path) -> None:
    from src.application.plate_from_spec import PlateFromSpecError, compose_plate_from_project  # noqa: PLC0415

    project, record_id = _embedded_project(tmp_path)
    spec = plate_spec(
        [record_id],
        DrawingSheetOptions(title_block=TitleBlock(artifact_label="삼각형"), scale_denominator=1.0),
    )
    spec_path = write_plate_spec(tmp_path / "plate.json", spec)
    bundle = compose_plate_from_project(project, spec)
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    assert json.loads(bundle.sidecar_bytes.decode("utf-8"))["plate_spec"] == spec

    out = tmp_path / "out" / "plate.svg"
    run = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "--plate", str(project), str(spec_path), str(out)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600, check=False,
    )
    assert run.returncode == 0, run.stderr
    assert out.read_bytes() == bundle.svg_bytes, "the command line and the library make the same plate"
    sidecar = out.with_suffix(".provenance.json")
    validate_drawing_sheet_bytes(out.read_bytes(), sidecar.read_bytes())
    # A record the project does not have, and pixels the command line cannot recompute, are refused.
    with pytest.raises(PlateFromSpecError):
        compose_plate_from_project(project, {**spec, "records": ["record:absent"]})
    with pytest.raises(PlateFromSpecError, match="cannot recompute"):
        compose_plate_from_project(project, {**spec, "paint_cutouts": [["record:cutout", record_id, "in_place"]]})
    failed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "--plate", str(project), str(tmp_path / "missing.json"), str(out)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600, check=False,
    )
    assert failed.returncode == 1 and "plate failed" in failed.stderr
