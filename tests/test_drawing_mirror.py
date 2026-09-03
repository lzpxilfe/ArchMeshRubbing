"""좌 반입면 · 우 반단면: one figure whose halves are elevation and section."""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.core.artifact_condition_annotation import (
    commit_condition_annotation,
    compute_condition_annotation,
)
from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_session import ArtifactSession
from src.core.artifact_surface_measurement import (
    ArtifactSurfaceMeasurementComputation,
    commit_artifact_surface_measurement,
    extract_surface_measurement,
    resolve_surface_anchor_from_ray,
    surface_diameter_recipe,
    surface_measurement_selection_hash,
)
from src.core.artifact_vector_extractor import (
    commit_vector_computation,
    compute_artifact_cutline,
)
from src.core.artifact_vector_record import PlanarFrame
from src.core.drawing_sheet import (
    DrawingSheetError,
    DrawingSheetOptions,
    SheetPage,
    TitleBlock,
    compose_drawing_sheet,
    validate_drawing_sheet_bytes,
)
from src.core.drawing_svg import (
    SVGRenderError,
    center_axis_line,
    clip_closed_ring,
    clip_open_path,
    half_plane_side,
    split_ring_off_line,
)
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


SVG_NS = "{http://www.w3.org/2000/svg}"

ELEVATION_ID = "record:elevation-front"
SECTION_ID = "record:section-front"
RIM_ID = "record:circle-rim"
FLOOR_ID = "record:circle-floor"

SEGMENTS = 24
RINGS = 10
HEIGHT_MM = 90.0
FLOOR_MM = 10.0
WALL_MM = 7.0
# Half a segment of spin, so the x-z plane a section is cut on passes between
# vertex columns instead of along them.  A scanned mesh is never that regular;
# a synthetic one has to be nudged or every cut lands on an on-plane edge.
PHASE = math.pi / SEGMENTS


def _outer_radius(z_mm: float) -> float:
    t = z_mm / HEIGHT_MM
    return 25.0 + 22.0 * t + 9.0 * math.sin(math.pi * t)


def _vessel() -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """A hollow vessel of revolution with a flat rim and a flat floor."""

    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    def ring(radius: float, z_mm: float) -> int:
        start = len(vertices)
        for segment in range(SEGMENTS):
            angle = PHASE + 2.0 * math.pi * segment / SEGMENTS
            vertices.append(
                [radius * math.cos(angle), radius * math.sin(angle), z_mm]
            )
        return start

    def band(lower: int, upper: int) -> None:
        for segment in range(SEGMENTS):
            following = (segment + 1) % SEGMENTS
            faces.append([lower + segment, lower + following, upper + following])
            faces.append([lower + segment, upper + following, upper + segment])

    outer = [
        ring(_outer_radius(HEIGHT_MM * index / RINGS), HEIGHT_MM * index / RINGS)
        for index in range(RINGS + 1)
    ]
    for index in range(RINGS):
        band(outer[index], outer[index + 1])

    inner_heights = [
        FLOOR_MM + (HEIGHT_MM - FLOOR_MM) * index / RINGS for index in range(RINGS + 1)
    ]
    inner = [ring(_outer_radius(z) - WALL_MM, z) for z in inner_heights]
    for index in range(RINGS):
        band(inner[index], inner[index + 1])
    band(outer[RINGS], inner[RINGS])

    base_center = len(vertices)
    vertices.append([0.0, 0.0, 0.0])
    for segment in range(SEGMENTS):
        faces.append(
            [
                base_center,
                outer[0] + (segment + 1) % SEGMENTS,
                outer[0] + segment,
            ]
        )
    floor_center = len(vertices)
    vertices.append([0.0, 0.0, FLOOR_MM])
    for segment in range(SEGMENTS):
        faces.append(
            [
                floor_center,
                inner[0] + segment,
                inner[0] + (segment + 1) % SEGMENTS,
            ]
        )

    rim_radius = _outer_radius(HEIGHT_MM) - WALL_MM / 2.0
    floor_radius = (_outer_radius(FLOOR_MM) - WALL_MM) * 0.6
    quarters = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
    rim_points = [
        np.array(
            [rim_radius * math.cos(a), rim_radius * math.sin(a), HEIGHT_MM],
            dtype=np.float64,
        )
        for a in quarters
    ]
    floor_points = [
        np.array(
            [floor_radius * math.cos(a), floor_radius * math.sin(a), FLOOR_MM],
            dtype=np.float64,
        )
        for a in quarters
    ]
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
        rim_points,
        floor_points,
    )


def _anchor(vertices: np.ndarray, faces: np.ndarray, point: np.ndarray) -> Any:
    return resolve_surface_anchor_from_ray(
        vertices,
        faces,
        source_faces=faces,
        ray_origin_world_mm=point + np.asarray([0.0, 0.0, 1.0]),
        ray_direction_world=[0.0, 0.0, -1.0],
        depth_point_world_mm=point,
        pixel_footprint_um=10,
    )


def _commit_circle(
    session: ArtifactSession,
    vertices: np.ndarray,
    faces: np.ndarray,
    points: list[np.ndarray],
    *,
    record_id: str,
    created_at: str,
) -> ArtifactSession:
    recipe = surface_diameter_recipe(
        [_anchor(vertices, faces, point) for point in points],
        source_vertex_count=int(vertices.shape[0]),
        source_face_count=int(faces.shape[0]),
    )
    receipt, qc = extract_surface_measurement(vertices, faces, recipe)
    context = session.capture_operation(
        recipe=recipe,
        selection_hash=surface_measurement_selection_hash(recipe),
    )
    return commit_artifact_surface_measurement(
        session,
        ArtifactSurfaceMeasurementComputation(
            context=context,
            projection_snapshot=session.projection_snapshot(),
            receipt=receipt,
            recipe=recipe,
            qc=qc,
        ),
        record_id=record_id,
        created_at=created_at,
        operator="tester",
    )


_SECTION_PLANE = PlanarFrame(
    origin_world_mm=(0.0, 0.0, 0.0),
    u_axis_world=(1.0, 0.0, 0.0),
    v_axis_world=(0.0, 0.0, 1.0),
    normal_world=(0.0, -1.0, 0.0),
)


@lru_cache(maxsize=1)
def _positioned() -> ArtifactSession:
    """A vessel stood upright by two measured circles, with both drawings."""

    vertices, faces, rim_points, floor_points = _vessel()
    mesh = MeshData(
        vertices=vertices,
        faces=faces,
        unit="mm",
        filepath=Path("/source/vessel.ply"),
        source_identity=SourceFingerprint(
            sha256="7" * 64,
            size_bytes=8192,
            mtime_ns=1,
            original_name="vessel.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/vessel.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="mirror-test",
        operator="tester",
        created_at="2026-09-03T00:00:00Z",
        document_id="artifact:vessel",
        metadata_revision_id="metadata:vessel",
        align_revision_id="align:vessel",
    )
    session = _commit_circle(
        session, vertices, faces, floor_points,
        record_id=FLOOR_ID, created_at="2026-09-03T00:00:01Z",
    )
    session = _commit_circle(
        session, vertices, faces, rim_points,
        record_id=RIM_ID, created_at="2026-09-03T00:00:02Z",
    )
    session = session.commit_axis_alignment(
        top_record_id=RIM_ID,
        bottom_record_id=FLOOR_ID,
        operator="tester",
        created_at="2026-09-03T00:00:03Z",
        revision_id="align:axis",
    )
    session = commit_vector_computation(
        session,
        compute_artifact_outline(session, "front", precision_grid_mm=0.5),
        record_id=ELEVATION_ID,
        created_at="2026-09-03T00:01:00Z",
        operator="tester",
    )
    return commit_vector_computation(
        session,
        compute_artifact_cutline(session, _SECTION_PLANE),
        record_id=SECTION_ID,
        created_at="2026-09-03T00:02:00Z",
        operator="tester",
    )


def _options(**overrides: Any) -> DrawingSheetOptions:
    settings: dict[str, Any] = {
        "title_block": TitleBlock(artifact_label="합성 시험 유물 · 완형 호"),
        "scale_denominator": 2.0,
        "page": SheetPage(size="A4", orientation="portrait"),
    }
    settings.update(overrides)
    return DrawingSheetOptions(**settings)


def _mirrored_sheet(**overrides: Any):
    return compose_drawing_sheet(
        _positioned().document,
        [ELEVATION_ID],
        options=_options(
            mirror_sections=((ELEVATION_ID, SECTION_ID),),
            **overrides,
        ),
    )


# --- the clipping primitives --------------------------------------------------


def test_a_half_plane_cut_keeps_the_side_it_was_asked_for() -> None:
    square = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]
    left = clip_closed_ring(
        square, base=(0.0, 0.0), direction=(0.0, 1.0), keep_negative=True
    )
    right = clip_closed_ring(
        square, base=(0.0, 0.0), direction=(0.0, 1.0), keep_negative=False
    )

    assert left is not None and right is not None
    assert all(point[0] <= 0.0 for point in left)
    assert all(point[0] >= 0.0 for point in right)
    # Together the halves are the whole square again.
    assert _ring_area(left) + _ring_area(right) == pytest.approx(16.0)


def _ring_area(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        total += point[0] * following[1] - following[0] * point[1]
    return abs(total) / 2.0


def test_a_ring_wholly_on_one_side_is_kept_or_dropped_whole() -> None:
    offset = [(1.0, -2.0), (3.0, -2.0), (3.0, 2.0), (1.0, 2.0)]

    assert (
        clip_closed_ring(
            offset, base=(0.0, 0.0), direction=(0.0, 1.0), keep_negative=True
        )
        is None
    )
    kept = clip_closed_ring(
        offset, base=(0.0, 0.0), direction=(0.0, 1.0), keep_negative=False
    )
    assert kept == [(1.0, -2.0), (3.0, -2.0), (3.0, 2.0), (1.0, 2.0)]


def test_a_ring_the_line_breaks_into_pieces_is_refused_not_bridged() -> None:
    """Sutherland-Hodgman would answer with an edge along the cutting line."""

    dumbbell = [
        (-3.0, -3.0), (3.0, -3.0), (3.0, -1.0), (-1.0, -1.0),
        (-1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (-3.0, 3.0),
    ]

    with pytest.raises(SVGRenderError, match="more than two pieces"):
        clip_closed_ring(
            dumbbell, base=(0.0, 0.0), direction=(0.0, 1.0), keep_negative=False
        )


def test_an_open_path_keeps_every_part_on_the_kept_side() -> None:
    zigzag = [(-3.0, 0.0), (3.0, 0.0), (3.0, 5.0), (-3.0, 5.0), (-3.0, 8.0), (3.0, 8.0)]

    pieces = clip_open_path(
        zigzag, base=(0.0, 0.0), direction=(0.0, 1.0), keep_negative=False
    )

    assert len(pieces) == 2
    assert all(point[0] >= 0.0 for piece in pieces for point in piece)


def test_the_fold_edge_is_the_one_the_drawing_drops() -> None:
    square = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]
    right = clip_closed_ring(
        square, base=(0.0, 0.0), direction=(0.0, 1.0), keep_negative=False
    )
    assert right is not None

    chains = split_ring_off_line(right, base=(0.0, 0.0), direction=(0.0, 1.0))
    assert chains is not None and len(chains) == 1
    assert chains[0] == [(0.0, -2.0), (2.0, -2.0), (2.0, 2.0), (0.0, 2.0)]
    # A ring the line never touched keeps every edge it has.
    assert split_ring_off_line(square, base=(9.0, 0.0), direction=(0.0, 1.0)) is None


def test_the_axis_line_is_oriented_the_same_way_whatever_the_frame() -> None:
    front = center_axis_line(
        {
            "origin_world_mm": (0.0, 0.0, 0.0),
            "u_axis_world": (1.0, 0.0, 0.0),
            "v_axis_world": (0.0, 0.0, 1.0),
        }
    )
    back = center_axis_line(
        {
            "origin_world_mm": (0.0, 0.0, 0.0),
            "u_axis_world": (-1.0, 0.0, 0.0),
            "v_axis_world": (0.0, 0.0, -1.0),
        }
    )
    assert front is not None and back is not None
    # Both frames see +Z; the direction is normalised towards +v either way, so
    # "which side" does not depend on how a record happened to be built.
    assert front[1] == (0.0, 1.0)
    assert back[1] == (0.0, 1.0)
    assert half_plane_side((-5.0, 0.0), base=front[0], direction=front[1]) < 0.0

    assert (
        center_axis_line(
            {
                "origin_world_mm": (0.0, 0.0, 0.0),
                "u_axis_world": (1.0, 0.0, 0.0),
                "v_axis_world": (0.0, 1.0, 0.0),
            }
        )
        is None
    )


# --- the mirrored figure ------------------------------------------------------


def test_one_figure_carries_the_elevation_left_and_the_section_right() -> None:
    bundle = _mirrored_sheet()
    root = ET.fromstring(bundle.svg_bytes)
    figures = _find(root, f"{SVG_NS}g[@id='sheet-figures']")
    assert len(figures) == 1, "the two halves are one figure, not two drawings"

    figure = figures[0]
    assert figure.attrib["data-record-id"] == ELEVATION_ID
    assert figure.attrib["data-mirror-section-record-id"] == SECTION_ID

    layers = {layer.attrib["id"]: layer for layer in figure}
    assert set(layers) == {
        "layer-section-cut",
        "layer-outline-visible",
        "layer-center-axis",
    }
    for path in layers["layer-outline-visible"]:
        assert path.attrib["id"].startswith("mirror:left:")
    for path in layers["layer-section-cut"]:
        assert path.attrib["id"].startswith("mirror:right:")


def test_each_half_stays_on_its_own_side_of_the_axis() -> None:
    figure = _figure(_mirrored_sheet().svg_bytes)
    axis_x = _axis_x(figure)

    left = [
        point
        for path in _find(figure, f"{SVG_NS}g[@id='layer-outline-visible']")
        for point in _points(path)
    ]
    right = [
        point
        for path in _find(figure, f"{SVG_NS}g[@id='layer-section-cut']")
        for point in _points(path)
    ]
    assert left and right
    assert max(point[0] for point in left) == pytest.approx(axis_x, abs=1e-9)
    assert min(point[0] for point in right) == pytest.approx(axis_x, abs=1e-9)


def _find(parent: Any, path: str) -> Any:
    """Find one element, or fail the test where the element was expected."""

    found = parent.find(path)
    assert found is not None, f"expected {path!r} in {parent.tag}"
    return found


def _figure(svg_bytes: bytes) -> Any:
    return _find(ET.fromstring(svg_bytes), f"{SVG_NS}g[@id='sheet-figures']")[0]


def _axis_x(figure: Any) -> float:
    return _points(
        _find(figure, f"{SVG_NS}g[@id='layer-center-axis']/{SVG_NS}path")
    )[0][0]


def _points(path: Any) -> list[tuple[float, float]]:
    tokens = path.attrib["d"].replace("M", " ").replace("L", " ").replace("Z", " ")
    numbers = [float(token) for token in tokens.split()]
    return list(zip(numbers[0::2], numbers[1::2], strict=True))


def test_the_fold_is_a_centre_line_and_not_a_drawn_edge() -> None:
    """The chord that closes each half lies on the axis and is not the object."""

    figure = _figure(_mirrored_sheet().svg_bytes)
    axis_x = _axis_x(figure)

    for layer_id in ("layer-outline-visible", "layer-section-cut"):
        for path in _find(figure, f"{SVG_NS}g[@id='{layer_id}']"):
            if path.attrib.get("stroke") == "none":
                continue
            points = _points(path)
            assert "Z" not in path.attrib["d"], (
                "a half closed along the axis would stroke a boundary the "
                "artifact does not have"
            )
            for first, second in zip(points, points[1:], strict=False):
                on_axis = (
                    abs(first[0] - axis_x) < 1e-9 and abs(second[0] - axis_x) < 1e-9
                )
                assert not on_axis, "the fold edge is still being stroked"


def test_the_cut_face_is_still_shaded_without_a_stroke_along_the_axis() -> None:
    figure = _figure(_mirrored_sheet().svg_bytes)
    section = _find(figure, f"{SVG_NS}g[@id='layer-section-cut']")

    fills = [path for path in section if path.attrib.get("stroke") == "none"]
    assert len(fills) == 1
    assert fills[0].attrib["fill"] == "url(#hatch-section-cut)"
    assert fills[0].attrib["d"].endswith("Z")
    assert fills[0].attrib["id"].endswith(":fill")


def test_a_mirrored_figure_draws_its_own_axis_without_being_asked() -> None:
    """The axis is the seam of this convention, not an optional annotation."""

    figure = _figure(_mirrored_sheet(show_center_axis=False).svg_bytes)

    assert figure.find(f"{SVG_NS}g[@id='layer-center-axis']") is not None


def test_a_condition_region_is_cut_to_the_elevation_half() -> None:
    session = _positioned()
    computation = compute_condition_annotation(
        session,
        condition="missing",
        face_indices=list(range(0, 40)),
        precision_grid_mm=0.5,
    )
    session = commit_condition_annotation(
        session,
        computation,
        record_id="record:condition-1",
        created_at="2026-09-03T00:03:00Z",
        operator="tester",
    )

    bundle = compose_drawing_sheet(
        session.document,
        [ELEVATION_ID],
        options=_options(
            mirror_sections=((ELEVATION_ID, SECTION_ID),),
            condition_records=("record:condition-1",),
        ),
    )
    figure = _figure(bundle.svg_bytes)
    layer = figure.find(f"{SVG_NS}g[@id='layer-condition-missing']")
    assert layer is not None, "a condition on the elevation belongs on its half"

    axis_x = _axis_x(figure)
    for path in layer:
        assert path.attrib["id"].startswith("mirror:left:condition:")
        assert all(point[0] <= axis_x + 1e-9 for point in _points(path))


def test_the_sidecar_names_both_records_and_which_side_each_took() -> None:
    bundle = _mirrored_sheet()
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))

    assert sidecar["mirrored_figures"] == [
        {
            "elevation_record_id": ELEVATION_ID,
            "elevation_side": "left",
            "section_record_id": SECTION_ID,
            "section_recipe_hash": _positioned()
            .document.record_index[SECTION_ID]
            .recipe_hash,
            "section_side": "right",
        }
    ]
    # The section is inside the figure, so it is not listed as one.
    assert [figure["record_id"] for figure in sidecar["figures"]] == [ELEVATION_ID]
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)


def test_a_sheet_without_mirror_sections_is_the_sheet_it_always_was() -> None:
    document = _positioned().document
    plain = compose_drawing_sheet(document, [ELEVATION_ID], options=_options())
    explicit = compose_drawing_sheet(
        document, [ELEVATION_ID], options=_options(mirror_sections=())
    )

    assert plain.svg_bytes == explicit.svg_bytes
    assert b"mirror" not in plain.svg_bytes
    assert "mirrored_figures" not in json.loads(plain.sidecar_bytes.decode("utf-8"))


# --- what it refuses ----------------------------------------------------------


def test_a_mirrored_figure_needs_an_artifact_stood_on_its_axis() -> None:
    """Half a pot is only meaningful about the axis the pot turns on."""

    # Back to the Align the import made, and fresh drawings under it: the
    # records are drawable, but nothing has measured where the axis is.
    dragged = _positioned().activate_parent_align()
    dragged = commit_vector_computation(
        dragged,
        compute_artifact_outline(dragged, "front", precision_grid_mm=0.5),
        record_id="record:dragged-elevation",
        created_at="2026-09-03T00:05:00Z",
        operator="tester",
    )
    dragged = commit_vector_computation(
        dragged,
        compute_artifact_cutline(dragged, _SECTION_PLANE),
        record_id="record:dragged-section",
        created_at="2026-09-03T00:06:00Z",
        operator="tester",
    )

    with pytest.raises(DrawingSheetError, match="rotation axis"):
        compose_drawing_sheet(
            dragged.document,
            ["record:dragged-elevation"],
            options=_options(
                mirror_sections=(
                    ("record:dragged-elevation", "record:dragged-section"),
                )
            ),
        )


def test_both_halves_must_be_in_one_plane() -> None:
    session = _positioned()
    turned = commit_vector_computation(
        session,
        compute_artifact_cutline(
            session,
            PlanarFrame(
                origin_world_mm=(0.0, 0.0, 0.0),
                u_axis_world=(0.0, 1.0, 0.0),
                v_axis_world=(0.0, 0.0, 1.0),
                normal_world=(1.0, 0.0, 0.0),
            ),
        ),
        record_id="record:section-side",
        created_at="2026-09-03T00:04:00Z",
        operator="tester",
    )

    with pytest.raises(DrawingSheetError, match="not in the same"):
        compose_drawing_sheet(
            turned.document,
            [ELEVATION_ID],
            options=_options(
                mirror_sections=((ELEVATION_ID, "record:section-side"),)
            ),
        )


def test_the_two_halves_must_be_an_outline_and_a_cutline() -> None:
    document = _positioned().document
    with pytest.raises(DrawingSheetError, match="not an outline"):
        compose_drawing_sheet(
            document,
            [SECTION_ID],
            options=_options(mirror_sections=((SECTION_ID, ELEVATION_ID),)),
        )

    session = commit_vector_computation(
        _positioned(),
        compute_artifact_outline(_positioned(), "right", precision_grid_mm=0.5),
        record_id="record:elevation-right",
        created_at="2026-09-03T00:07:00Z",
        operator="tester",
    )
    with pytest.raises(DrawingSheetError, match="not a cutline"):
        compose_drawing_sheet(
            session.document,
            [ELEVATION_ID],
            options=_options(
                mirror_sections=((ELEVATION_ID, "record:elevation-right"),)
            ),
        )


def test_the_section_half_is_not_also_a_figure_of_its_own() -> None:
    with pytest.raises(DrawingSheetError, match="must not also be a figure"):
        compose_drawing_sheet(
            _positioned().document,
            [ELEVATION_ID, SECTION_ID],
            options=_options(mirror_sections=((ELEVATION_ID, SECTION_ID),)),
        )


def test_the_elevation_half_must_be_one_of_the_sheets_records() -> None:
    with pytest.raises(DrawingSheetError, match="must be one of the sheet"):
        compose_drawing_sheet(
            _positioned().document,
            [ELEVATION_ID],
            options=_options(
                mirror_sections=(("record:absent", SECTION_ID),)
            ),
        )


def test_a_record_cannot_be_both_halves_or_two_figures_halves() -> None:
    with pytest.raises(DrawingSheetError, match="both halves"):
        _options(mirror_sections=((ELEVATION_ID, ELEVATION_ID),))
    with pytest.raises(DrawingSheetError, match="at most one mirrored figure"):
        _options(
            mirror_sections=(
                (ELEVATION_ID, SECTION_ID),
                ("record:other", SECTION_ID),
            )
        )
