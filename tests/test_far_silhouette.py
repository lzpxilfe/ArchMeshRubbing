"""The far half's silhouette: what shows through the cut on a warped vessel.

On a level vessel the rim seen going round the back through the cut is a
level line, and a straight run of the elevation's edge past the fold draws
it.  On a vessel whose rim is not level - the celadon dish warped in the
kiln - that line slopes, and only the far half's measured silhouette says
how.  These tests stand a footed vessel up, tilt its rim so the back is
6 mm higher than the front, and check that the record measures the far
half alone, reopens as it was, and draws on the section's side as the
sloping edge it is.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.core.artifact_far_silhouette import (
    FAR_SILHOUETTE_RECORD_TYPE,
    ArtifactFarSilhouetteError,
    commit_far_silhouette,
    compute_artifact_far_silhouette,
    far_face_mask,
    far_silhouette_payload_from_record,
    far_silhouette_recipe,
    validate_far_silhouette_recipe,
)
from src.core.artifact_outline_extractor import compute_artifact_outline, outline_frame
from src.core.artifact_record_validation import validate_known_records
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_extractor import commit_vector_computation, compute_artifact_cutline
from src.core.artifact_vector_record import PlanarFrame, VectorGeometryPayload
from src.core.drawing_sheet import (
    DrawingSheetError,
    DrawingSheetOptions,
    SheetPage,
    TitleBlock,
    compose_drawing_sheet,
    validate_drawing_sheet_bytes,
)
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.project_file import load_artifact_project, save_artifact_project
from src.core.source_identity import SourceFingerprint
from synthetic_vessel import _commit_circle
from test_artifact_profile_break import _footed_vessel

STAMP = "2026-09-08T00:00:00Z"
RIM_Z = 90.0
TILT_MM = 6.0
RIM_R = 46.0


def _tilted_vessel() -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """The footed vessel with its rim sheared: the back (+y) 6 mm higher
    than the front, blended in over the top 10 mm so the wall stays whole."""

    vertices, faces, rim, floor = _footed_vessel()
    v = vertices.copy()
    high = v[:, 2] > RIM_Z - 10.0
    angle = np.arctan2(v[high, 1], v[high, 0])
    v[high, 2] += TILT_MM * np.sin(angle) * (v[high, 2] - (RIM_Z - 10.0)) / 10.0
    rim_points = [v[int(np.flatnonzero((vertices == p).all(axis=1))[0])] for p in rim]
    return v, faces, rim_points, floor


@pytest.fixture(scope="module")
def tilted() -> ArtifactSession:
    vertices, faces, rim, floor = _tilted_vessel()
    mesh = MeshData(
        vertices=vertices, faces=faces, unit="mm", filepath=Path("/source/tilted.ply"),
        source_identity=SourceFingerprint(sha256="a" * 64, size_bytes=4096, mtime_ns=1, original_name="tilted.ply", format="ply"),
        source_format="ply", source_import_recipe=current_mesh_import_recipe("ply"),
    )
    session = ArtifactSession.create_from_source(
        mesh, resolved_source_path="/source/tilted.ply", unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"}, handedness="right",
        software_version="far-test", operator="tester", created_at=STAMP,
        document_id="artifact:tilted", metadata_revision_id="metadata:tilted", align_revision_id="align:tilted",
    )
    session = _commit_circle(session, vertices, faces, floor, record_id="record:floor", created_at="2026-09-08T00:00:01Z")
    session = _commit_circle(session, vertices, faces, rim, record_id="record:rim", created_at="2026-09-08T00:00:02Z")
    session = session.commit_axis_alignment(
        top_record_id="record:rim", bottom_record_id="record:floor", operator="tester",
        created_at="2026-09-08T00:00:03Z", revision_id="align:axis",
    )
    session = commit_vector_computation(
        session, compute_artifact_outline(session, "front", precision_grid_mm=0.5),
        record_id="record:front", created_at="2026-09-08T00:01:00Z", operator="tester",
    )
    return commit_vector_computation(
        session,
        compute_artifact_cutline(session, PlanarFrame(origin_world_mm=(0.0, 0.0, 0.0), u_axis_world=(1.0, 0.0, 0.0), v_axis_world=(0.0, 0.0, 1.0), normal_world=(0.0, -1.0, 0.0))),
        record_id="record:section", created_at="2026-09-08T00:02:00Z", operator="tester",
    )


def _top_at(payload: VectorGeometryPayload, u: float) -> float:
    """The highest point of the payload's rings on the vertical line at u."""

    best = -math.inf
    for path in payload.paths:
        points = list(path.points_mm) + [path.points_mm[0]]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if (x0 - u) * (x1 - u) > 0.0 or x0 == x1:
                continue
            best = max(best, y0 + (y1 - y0) * (u - x0) / (x1 - x0))
    return best


def _floor_offset(session: ArtifactSession) -> float:
    """The canonical frame stands the floor circle at height zero."""

    return 8.5


def test_the_far_half_is_the_half_behind_the_view_and_its_rim_is_the_back_rim(tilted) -> None:
    """The mask takes the faces whose centre is behind the front view's
    plane, and the silhouette of those faces alone tops out at the back
    rim's height over the axis, where the near half would sit 12 mm lower;
    committed, saved and reopened it is the same record."""

    projection = tilted.materialize()
    vertices = np.asarray(projection.mesh.vertices, dtype=np.float64)
    faces = np.asarray(projection.mesh.faces, dtype=np.int64)
    behind = far_face_mask(vertices, faces, "front")
    centres = vertices[faces].mean(axis=1)
    assert behind.any() and (~behind).any()
    assert (centres[behind, 1] > 0.0).all() and (centres[~behind, 1] <= 0.0).all(), "behind the front view is +y"

    computation = compute_artifact_far_silhouette(tilted, "front", precision_grid_mm=0.5)
    payload = computation.payload
    assert payload.frame == outline_frame("front")
    assert computation.qc["far_face_count"] + computation.qc["near_face_count"] == int(faces.shape[0])
    assert computation.qc["far_face_count"] == int(behind.sum())
    offset = _floor_offset(tilted)
    # Over the axis the far silhouette is the back rim: 6 mm above the
    # nominal rim.  A third of the way out (60 degrees round the back) it
    # is 5.2 mm above; the whole outline says the same, because the back
    # is the higher side, and the near half alone would be 12 mm lower.
    assert abs(_top_at(payload, 0.0) - (RIM_Z + TILT_MM - offset)) < 0.8
    u = RIM_R * math.cos(math.radians(60.0))
    assert abs(_top_at(payload, u) - (RIM_Z + TILT_MM * math.sin(math.radians(60.0)) - offset)) < 0.8
    whole = compute_artifact_outline(tilted, "front", precision_grid_mm=0.5).payload
    assert abs(_top_at(whole, 0.0) - _top_at(payload, 0.0)) < 0.6

    session = commit_far_silhouette(tilted, computation, record_id="record:far", created_at=STAMP, operator="tester")
    record = session.document.record_index["record:far"]
    assert record.type == FAR_SILHOUETTE_RECORD_TYPE
    assert far_silhouette_payload_from_record(record) == payload
    assert record.recipe["face_scope"]["side"] == "behind_view_plane/v1"
    assert record.recipe["outline"]["precision_grid_mm"] == 0.5
    path = Path(tempfile.mkdtemp()) / "tilted.amr"
    save_artifact_project(path, session.document)
    reopened = load_artifact_project(path)
    validate_known_records(reopened)
    assert far_silhouette_payload_from_record(reopened.record_index["record:far"]) == payload


def test_a_far_silhouette_that_does_not_add_up_is_refused(tilted) -> None:
    recipe = far_silhouette_recipe("front", precision_grid_mm=0.5)
    assert validate_far_silhouette_recipe(recipe) == recipe
    with pytest.raises(ArtifactFarSilhouetteError, match="unsupported outline view"):
        far_silhouette_recipe("rim", precision_grid_mm=0.5)
    tampered = dict(recipe)
    tampered["face_scope"] = {**recipe["face_scope"], "side": "in_front_of_view_plane/v1"}
    with pytest.raises(ArtifactFarSilhouetteError, match="does not match the production contract"):
        validate_far_silhouette_recipe(tampered)
    computation = compute_artifact_far_silhouette(tilted, "front", precision_grid_mm=0.5)
    session = commit_far_silhouette(tilted, computation, record_id="record:far", created_at=STAMP, operator="tester")
    record = session.document.record_index["record:far"]
    descriptor = dict(record.extensions["org.archmeshrubbing:far-silhouette-v1"])
    descriptor["sha256"] = "0" * 64
    from dataclasses import replace  # noqa: PLC0415

    forged = replace(record, extensions={"org.archmeshrubbing:far-silhouette-v1": descriptor})
    with pytest.raises(ArtifactFarSilhouetteError, match="SHA-256 does not match"):
        far_silhouette_payload_from_record(forged)
    with pytest.raises(ArtifactFarSilhouetteError, match="not a far silhouette"):
        far_silhouette_payload_from_record(session.document.record_index["record:front"])


def _paths(root: ET.Element, id_part: str) -> list[list[tuple[float, float]]]:
    out = []
    for el in root.iter():
        if id_part in el.attrib.get("id", "") and "d" in el.attrib:
            tokens = el.attrib["d"].replace("M", " ").replace("L", " ").split()
            out.append([(float(tokens[i]), float(tokens[i + 1])) for i in range(0, len(tokens), 2)])
    return out


def _axis_x(root: ET.Element) -> float:
    (axis,) = _paths(root, "mirror:center-axis")
    return axis[0][0]


def test_the_rim_goes_round_the_back_the_way_it_was_measured(tilted) -> None:
    """With outline_reach='far' the section's side shows the far half's own
    rim edge: it leaves the axis at the back rim's height and comes down
    to the rim's tip, stopping the gap short of the cut, where the straight
    run would have stayed level.  Under the solid foot nothing shows.  The
    sidecar names the silhouette and counts the edges."""

    computation = compute_artifact_far_silhouette(tilted, "front", precision_grid_mm=0.5)
    session = commit_far_silhouette(tilted, computation, record_id="record:far", created_at=STAMP, operator="tester")
    title = TitleBlock(artifact_label="구연이 기운 시험 호")
    page = SheetPage(size="A4", orientation="portrait")
    far = compose_drawing_sheet(
        session.document, ["record:front"],
        options=DrawingSheetOptions(
            title_block=title, page=page, scale_denominator=2.0,
            mirror_sections=(("record:front", "record:section"),),
            far_silhouettes=(("record:front", "record:far"),), outline_reach="far",
        ),
    )
    validate_drawing_sheet_bytes(far.svg_bytes, far.sidecar_bytes)
    root = ET.fromstring(far.svg_bytes)
    edges = _paths(root, ":far-edge:record:far:")
    assert len(edges) == 1, [len(edge) for edge in edges]
    (edge,) = edges
    axis_x = _axis_x(root)
    assert abs(edge[0][0] - axis_x) < 0.3, "the edge leaves the axis"
    assert edge[-1][0] - axis_x > RIM_R / 2.0 - 4.0, "and reaches nearly to the rim's tip"
    assert edge[-1][0] - axis_x < RIM_R / 2.0 - 0.5, "stopping short of the cut"
    # Paper y grows downward: the far end is lower on the page by about
    # the tilt at 1:2, less what the last gap-length hides.
    drop = edge[-1][1] - edge[0][1]
    assert TILT_MM / 2.0 - 1.5 < drop < TILT_MM / 2.0 + 0.3, drop
    assert all(later[1] >= earlier[1] - 0.05 for earlier, later in zip(edge, edge[1:])), "it comes down all the way"
    assert not _paths(root, ":past-axis:"), "no straight run beside it"
    figure = json.loads(far.sidecar_bytes.decode("utf-8"))["mirrored_figures"][0]
    assert figure["outline_reach"] == "far" and figure["far_edge_count"] == "1"
    assert figure["far_silhouette_record_id"] == "record:far"
    assert figure["far_silhouette_recipe_hash"] == session.document.record_index["record:far"].recipe_hash
    assert figure["outline_past_axis_count"] == "0"

    straight = compose_drawing_sheet(
        session.document, ["record:front"],
        options=DrawingSheetOptions(
            title_block=title, page=page, scale_denominator=2.0,
            mirror_sections=(("record:front", "record:section"),), outline_reach="section",
        ),
    )
    # A straight run, where there is one, is a two-point line in the near
    # edge's own direction; it is not the measured far edge, and the sheet
    # does not pretend it is.
    straight_root = ET.fromstring(straight.svg_bytes)
    assert not _paths(straight_root, ":far-edge:")
    assert all(len(run) == 2 for run in _paths(straight_root, ":past-axis:"))
    assert "far_silhouette_record_id" not in json.loads(straight.sidecar_bytes.decode("utf-8"))["mirrored_figures"][0]

    # Asked for straight far edges, the same edge is the one segment between
    # its measured ends, the title block says so, and the sidecar names it.
    from src.core.drawing_sheet import INTERPRETATION_LABEL, Interpretation  # noqa: PLC0415

    ruled = compose_drawing_sheet(
        session.document, ["record:front"],
        options=DrawingSheetOptions(
            title_block=title, page=page, scale_denominator=2.0,
            mirror_sections=(("record:front", "record:section"),),
            far_silhouettes=(("record:front", "record:far"),), outline_reach="far",
            interpretation=Interpretation(straight_far_edges=True),
        ),
    )
    validate_drawing_sheet_bytes(ruled.svg_bytes, ruled.sidecar_bytes)
    (segment,) = _paths(ET.fromstring(ruled.svg_bytes), ":far-edge:record:far:")
    assert len(segment) == 2 and len(edge) > 2
    assert all(abs(a - b) < 1e-6 for a, b in zip(segment[0], edge[0])) and all(abs(a - b) < 1e-6 for a, b in zip(segment[-1], edge[-1]))
    ruled_sidecar = json.loads(ruled.sidecar_bytes.decode("utf-8"))
    assert ruled_sidecar["mirrored_figures"][0]["far_edges"] == "straight" and figure["far_edges"] == "measured"
    assert ruled_sidecar["interpretation"]["straight_far_edges"] is True
    (row,) = [row for row in ruled_sidecar["title_block"] if row["label"] == INTERPRETATION_LABEL]
    assert row["value"] == "뒷선 직선"
    with pytest.raises(DrawingSheetError, match="straight_far_edges must be a boolean"):
        Interpretation(straight_far_edges=1)  # type: ignore[arg-type]

    # The options refuse a silhouette that would not be drawn, a reach
    # with nothing to draw it from, and a silhouette listed as a figure.
    with pytest.raises(DrawingSheetError, match="drawn only with outline_reach='far'"):
        DrawingSheetOptions(
            title_block=title, page=page, scale_denominator=2.0,
            mirror_sections=(("record:front", "record:section"),), far_silhouettes=(("record:front", "record:far"),),
        )
    with pytest.raises(DrawingSheetError, match="needs a far silhouette for every mirrored figure"):
        DrawingSheetOptions(
            title_block=title, page=page, scale_denominator=2.0,
            mirror_sections=(("record:front", "record:section"),), outline_reach="far",
        )
    with pytest.raises(DrawingSheetError, match="not the elevation half"):
        DrawingSheetOptions(
            title_block=title, page=page, scale_denominator=2.0, outline_reach="far",
            mirror_sections=(("record:front", "record:section"),), far_silhouettes=(("record:section", "record:far"),),
        )
    with pytest.raises(DrawingSheetError, match="must not also be a figure of its own"):
        compose_drawing_sheet(
            session.document, ["record:front", "record:far"],
            options=DrawingSheetOptions(
                title_block=title, page=page, scale_denominator=2.0, outline_reach="far",
                mirror_sections=(("record:front", "record:section"),), far_silhouettes=(("record:front", "record:far"),),
            ),
        )
    with pytest.raises(DrawingSheetError, match="is not a far silhouette"):
        compose_drawing_sheet(
            session.document, ["record:front"],
            options=DrawingSheetOptions(
                title_block=title, page=page, scale_denominator=2.0, outline_reach="far",
                mirror_sections=(("record:front", "record:section"),), far_silhouettes=(("record:front", "record:rim"),),
            ),
        )
    with pytest.raises(DrawingSheetError, match="break_reach must be one of section, axis"):
        DrawingSheetOptions(title_block=title, page=page, scale_denominator=2.0, break_reach="far")
