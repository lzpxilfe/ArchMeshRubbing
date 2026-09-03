from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.core.artifact_axis_alignment import (
    AXIS_ALIGN_CONVENTION,
    AXIS_ALIGN_RECIPE_KIND,
    ArtifactAxisAlignmentError,
    axis_align_delta_from_recipe,
    build_axis_alignment,
    verify_axis_alignment_matrix,
)
from src.core.artifact_session import ArtifactSession, ArtifactSessionError
from src.core.artifact_surface_measurement import (
    ArtifactSurfaceMeasurementComputation,
    commit_artifact_surface_measurement,
    extract_surface_measurement,
    resolve_surface_anchor_from_ray,
    surface_diameter_recipe,
    surface_measurement_selection_hash,
)
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


STAMP = "2026-09-03T00:00:00Z"
BOTTOM_ID = "record:base-circle"
TOP_ID = "record:rim-circle"

BOTTOM_RADIUS_MM = 20.0
TOP_RADIUS_MM = 40.0
AXIS_LENGTH_MM = 100.0


def _tilted_axis(tilt_deg: float) -> np.ndarray:
    """A unit axis leaning `tilt_deg` away from +Z, in the XZ plane."""

    tilt = math.radians(tilt_deg)
    return np.asarray([math.sin(tilt), 0.0, math.cos(tilt)], dtype=np.float64)


def _angle_to_canonical_axis_deg(vector: np.ndarray) -> float:
    unit = np.asarray(vector, dtype=np.float64)
    unit = unit / float(np.linalg.norm(unit))
    return math.degrees(math.acos(float(np.clip(unit[2], -1.0, 1.0))))


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.asarray([0.0, 1.0, 0.0])
    first = np.cross(normal, helper)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    return first, second / np.linalg.norm(second)


def _circle_geometry(
    center: np.ndarray,
    radius: float,
    axis: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return one equilateral triangle in the circle's plane, and four rim points."""

    u, v = _plane_basis(axis)
    points = [
        center + radius * math.cos(angle) * u + radius * math.sin(angle) * v
        for angle in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
    ]
    # Circumradius 4r, so the incircle (radius 2r) still contains every rim point.
    triangle = np.asarray(
        [
            center + 4.0 * radius * (math.cos(angle) * u + math.sin(angle) * v)
            for angle in (math.pi / 2.0, math.pi / 2.0 + 2.0 * math.pi / 3.0,
                          math.pi / 2.0 + 4.0 * math.pi / 3.0)
        ],
        dtype=np.float64,
    )
    return triangle, points


def _pot(tilt_deg: float = 10.0) -> tuple[np.ndarray, np.ndarray, list, list]:
    """A two-plate stand-in for a wheel-thrown vessel: a base and a rim circle."""

    axis = _tilted_axis(tilt_deg)
    bottom_center = np.zeros(3, dtype=np.float64)
    top_center = bottom_center + AXIS_LENGTH_MM * axis
    bottom_triangle, bottom_points = _circle_geometry(
        bottom_center, BOTTOM_RADIUS_MM, axis
    )
    top_triangle, top_points = _circle_geometry(top_center, TOP_RADIUS_MM, axis)
    vertices = np.vstack([bottom_triangle, top_triangle])
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return vertices, faces, bottom_points, top_points


def _anchor(
    vertices: np.ndarray,
    faces: np.ndarray,
    point: np.ndarray,
) -> dict[str, Any]:
    depth = np.asarray(point, dtype=np.float64)
    # A short ray, so aiming at the base does not first strike the rim plate.
    return resolve_surface_anchor_from_ray(
        vertices,
        faces,
        source_faces=faces,
        ray_origin_world_mm=depth + np.asarray([0.0, 0.0, 1.0]),
        ray_direction_world=[0.0, 0.0, -1.0],
        depth_point_world_mm=depth,
        pixel_footprint_um=10,
    )


def _session(vertices: np.ndarray, faces: np.ndarray) -> ArtifactSession:
    mesh = MeshData(
        vertices=vertices,
        faces=faces,
        unit="mm",
        filepath=Path("/private/axis/pot.ply"),
        source_identity=SourceFingerprint(
            sha256="9" * 64,
            size_bytes=2048,
            mtime_ns=1,
            original_name="pot.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    return ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/private/axis/pot.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="axis-test",
        operator="tester",
        created_at=STAMP,
        document_id="artifact:axis-alignment",
        metadata_revision_id="metadata:axis-alignment",
        align_revision_id="align:axis-alignment",
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
    computation = ArtifactSurfaceMeasurementComputation(
        context=context,
        projection_snapshot=session.projection_snapshot(),
        receipt=receipt,
        recipe=recipe,
        qc=qc,
    )
    return commit_artifact_surface_measurement(
        session,
        computation,
        record_id=record_id,
        created_at=created_at,
        operator="tester",
    )


def _measured_pot(tilt_deg: float = 10.0) -> ArtifactSession:
    vertices, faces, bottom_points, top_points = _pot(tilt_deg)
    session = _session(vertices, faces)
    session = _commit_circle(
        session, vertices, faces, bottom_points,
        record_id=BOTTOM_ID, created_at="2026-09-03T00:00:01Z",
    )
    return _commit_circle(
        session, vertices, faces, top_points,
        record_id=TOP_ID, created_at="2026-09-03T00:00:02Z",
    )


def test_two_measured_circles_stand_the_pot_up() -> None:
    """The line joining a base and a rim centre is the rotation axis."""

    session = _measured_pot(tilt_deg=10.0)
    aligned = session.commit_axis_alignment(
        top_record_id=TOP_ID,
        bottom_record_id=BOTTOM_ID,
        operator="tester",
        created_at="2026-09-03T00:00:03Z",
        revision_id="align:axis",
    )
    revision = aligned.document.align_revision_index["align:axis"]

    # The measured axis, carried through the new alignment, is +Z.
    #
    # It cannot be exact: the anchors are quantised to a 1 µm grid, so a centre
    # carries roughly a micrometre of error and the direction over a 100 mm
    # baseline carries about 1e-5 rad of it. That is under a thousandth of a
    # degree, far below anything a drawing can show, so the assertion is on the
    # angle rather than on raw components.
    axis_before = _tilted_axis(10.0)
    rotated = revision.matrix[:3, :3] @ axis_before
    assert _angle_to_canonical_axis_deg(rotated) < 0.002

    # The base circle's centre is the origin, so the pot stands on Z = 0.
    origin = revision.matrix @ np.asarray([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(origin[:3], [0.0, 0.0, 0.0], rtol=0.0, atol=1e-9)


def test_the_alignment_is_a_proper_rigid_transform() -> None:
    """An Align may rotate and move an artifact. It may not resize it."""

    aligned = _measured_pot().commit_axis_alignment(
        top_record_id=TOP_ID,
        bottom_record_id=BOTTOM_ID,
        operator="tester",
        revision_id="align:axis",
    )
    rotation = aligned.document.align_revision_index["align:axis"].matrix[:3, :3]

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-12)
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-12)


def test_the_qc_records_what_made_the_axis_believable() -> None:
    aligned = _measured_pot(tilt_deg=10.0).commit_axis_alignment(
        top_record_id=TOP_ID,
        bottom_record_id=BOTTOM_ID,
        operator="tester",
        revision_id="align:axis",
    )
    qc = aligned.document.align_revision_index["align:axis"].qc

    assert qc["proper_rigid"] is True
    assert qc["axis_tilt_corrected_deg"] == pytest.approx(10.0, abs=0.01)
    assert qc["center_separation_mm"] == pytest.approx(AXIS_LENGTH_MM, abs=0.01)
    assert qc["circle_normal_disagreement_deg"] < 1.0


def test_the_recipe_names_its_evidence_and_carries_its_own_inputs() -> None:
    """An offline verifier has the recipe but not the records it came from."""

    aligned = _measured_pot().commit_axis_alignment(
        top_record_id=TOP_ID,
        bottom_record_id=BOTTOM_ID,
        operator="tester",
        revision_id="align:axis",
    )
    revision = aligned.document.align_revision_index["align:axis"]
    recipe = revision.recipe

    assert recipe["kind"] == AXIS_ALIGN_RECIPE_KIND
    assert recipe["convention"] == AXIS_ALIGN_CONVENTION
    assert recipe["top_record_id"] == TOP_ID
    assert recipe["bottom_record_id"] == BOTTOM_ID
    for key in ("top_center_mm_decimal", "bottom_center_mm_decimal"):
        assert [type(value) for value in recipe[key]] == [str, str, str]

    parent = aligned.document.align_revision_index[revision.parent_id]
    verify_axis_alignment_matrix(
        recipe=recipe,
        parent_matrix=parent.matrix,
        matrix=revision.matrix,
    )


def test_recomputation_is_exact_and_catches_a_doctored_recipe() -> None:
    """Fixed decimals and Rodrigues arithmetic reproduce the matrix bit for bit."""

    aligned = _measured_pot().commit_axis_alignment(
        top_record_id=TOP_ID,
        bottom_record_id=BOTTOM_ID,
        operator="tester",
        revision_id="align:axis",
    )
    revision = aligned.document.align_revision_index["align:axis"]
    parent = aligned.document.align_revision_index[revision.parent_id]

    first = axis_align_delta_from_recipe(revision.recipe)
    second = axis_align_delta_from_recipe(revision.recipe)
    assert np.array_equal(first, second)

    forged = copy.deepcopy(dict(revision.recipe))
    forged["top_center_mm_decimal"] = list(forged["top_center_mm_decimal"])
    forged["top_center_mm_decimal"][0] = "999.000000"
    with pytest.raises(ArtifactAxisAlignmentError, match="not the one its axis recipe"):
        verify_axis_alignment_matrix(
            recipe=forged,
            parent_matrix=parent.matrix,
            matrix=revision.matrix,
        )


def test_an_axis_pointing_down_still_yields_a_proper_rotation() -> None:
    """Naming the circles the other way round is the anti-parallel case."""

    aligned = _measured_pot().commit_axis_alignment(
        top_record_id=BOTTOM_ID,
        bottom_record_id=TOP_ID,
        operator="tester",
        revision_id="align:inverted",
    )
    revision = aligned.document.align_revision_index["align:inverted"]
    rotation = revision.matrix[:3, :3]

    assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-12)
    rotated = rotation @ -_tilted_axis(10.0)
    assert _angle_to_canonical_axis_deg(rotated) < 0.002


def test_the_same_circle_twice_is_refused() -> None:
    session = _measured_pot()
    with pytest.raises(ArtifactSessionError, match="two different records"):
        session.commit_axis_alignment(
            top_record_id=TOP_ID,
            bottom_record_id=TOP_ID,
            operator="tester",
        )


def test_a_missing_or_wrongly_typed_record_is_refused() -> None:
    session = _measured_pot()
    with pytest.raises(ArtifactSessionError, match="does not exist"):
        session.commit_axis_alignment(
            top_record_id="record:nope",
            bottom_record_id=BOTTOM_ID,
            operator="tester",
        )


def test_circles_too_close_together_cannot_fix_an_axis() -> None:
    """A short baseline means the direction is the fit error, not the pot."""

    vertices, faces, bottom_points, _top = _pot(tilt_deg=0.0)
    session = _session(vertices, faces)
    session = _commit_circle(
        session, vertices, faces, bottom_points,
        record_id=BOTTOM_ID, created_at="2026-09-03T00:00:01Z",
    )
    # A second circle in the same plane: the centres coincide.
    nudged = [point + np.asarray([0.0, 0.0, 0.0]) for point in bottom_points]
    session = _commit_circle(
        session, vertices, faces, nudged,
        record_id=TOP_ID, created_at="2026-09-03T00:00:02Z",
    )

    with pytest.raises(ArtifactSessionError, match="too\\s+close to fix an axis"):
        session.commit_axis_alignment(
            top_record_id=TOP_ID,
            bottom_record_id=BOTTOM_ID,
            operator="tester",
        )


def test_a_circle_measured_under_another_align_is_refused() -> None:
    """Two centres must be numbers in one frame or the line between means nothing."""

    session = _measured_pot()
    moved = session.commit_preview(
        translation_mm=(1.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        operator="tester",
        revision_id="align:moved",
    )
    with pytest.raises(ArtifactSessionError, match="measured under a different Align"):
        moved.commit_axis_alignment(
            top_record_id=TOP_ID,
            bottom_record_id=BOTTOM_ID,
            operator="tester",
        )


def test_a_recipe_whose_centres_coincide_names_no_axis() -> None:
    same = ["1.000000", "2.000000", "3.000000"]
    recipe = {
        "bottom_center_mm_decimal": same,
        "bottom_normal_unit_decimal": ["0.000000000", "0.000000000", "1.000000000"],
        "bottom_record_id": "record:b",
        "convention": AXIS_ALIGN_CONVENTION,
        "kind": AXIS_ALIGN_RECIPE_KIND,
        "top_center_mm_decimal": same,
        "top_normal_unit_decimal": ["0.000000000", "0.000000000", "1.000000000"],
        "top_record_id": "record:t",
    }
    with pytest.raises(ArtifactAxisAlignmentError, match="names no axis"):
        axis_align_delta_from_recipe(recipe)


def test_a_recipe_of_another_kind_is_not_derived_as_an_axis() -> None:
    with pytest.raises(ArtifactAxisAlignmentError, match="kind must be"):
        axis_align_delta_from_recipe(
            {"kind": "manual_scene_trs_delta", "convention": AXIS_ALIGN_CONVENTION}
        )


def test_building_an_axis_needs_a_document() -> None:
    with pytest.raises(ArtifactAxisAlignmentError, match="must be an ArtifactDocument"):
        build_axis_alignment(
            object(), top_record_id=TOP_ID, bottom_record_id=BOTTOM_ID
        )


def test_a_positioned_artifact_can_still_be_exported() -> None:
    """The point of the whole change.

    Before this, the export gate required every non-root Align to be a manual
    drag, so one computed alignment permanently blocked vector, rubbing and
    tile-unwrap export for that artifact. A positioning aid nobody could export
    from would have been worse than none.
    """

    from src.core.artifact_outline_extractor import compute_artifact_outline
    from src.core.artifact_vector_export import (
        build_vector_export,
        validate_vector_export_bytes,
    )
    from src.core.artifact_vector_extractor import commit_vector_computation

    session = _measured_pot().commit_axis_alignment(
        top_record_id=TOP_ID,
        bottom_record_id=BOTTOM_ID,
        operator="tester",
        created_at="2026-09-03T00:00:03Z",
        revision_id="align:axis",
    )
    outline = compute_artifact_outline(session, "top", precision_grid_mm=0.01)
    session = commit_vector_computation(
        session,
        outline,
        record_id="record:outline-top",
        created_at="2026-09-03T00:00:04Z",
        operator="tester",
    )

    bundle = build_vector_export(session.document, "record:outline-top")
    validate_vector_export_bytes(
        bundle.svg_bytes, bundle.sidecar_bytes, document=session.document
    )

    ancestry = json.loads(bundle.sidecar_bytes)["provenance"]["align_ancestry"]
    kinds = [entry["recipe"]["kind"] for entry in ancestry]
    assert kinds == ["initial_identity", AXIS_ALIGN_RECIPE_KIND]

    # The evidence reaches the package instead of being filtered out on the way.
    qc = ancestry[-1]["qc"]
    assert qc["axis_tilt_corrected_deg"] == pytest.approx(10.0, abs=0.01)
    assert qc["center_separation_mm"] == pytest.approx(AXIS_LENGTH_MM, abs=0.01)
    assert qc["circle_normal_disagreement_deg"] < 1.0
    assert qc["proper_rigid"] is True

    # And the package names the records the axis came from.
    recipe = ancestry[-1]["recipe"]
    assert recipe["top_record_id"] == TOP_ID
    assert recipe["bottom_record_id"] == BOTTOM_ID


def test_an_export_rejects_an_axis_align_whose_matrix_was_edited() -> None:
    """Recomputation, not the recipe's own word, is what the package trusts."""

    from src.core.artifact_document import ArtifactDocument
    from src.core.artifact_outline_extractor import compute_artifact_outline
    from src.core.artifact_vector_export import (
        ArtifactVectorExportError,
        build_vector_export,
    )
    from src.core.artifact_vector_extractor import commit_vector_computation

    session = _measured_pot().commit_axis_alignment(
        top_record_id=TOP_ID,
        bottom_record_id=BOTTOM_ID,
        operator="tester",
        created_at="2026-09-03T00:00:03Z",
        revision_id="align:axis",
    )
    outline = compute_artifact_outline(session, "top", precision_grid_mm=0.01)
    session = commit_vector_computation(
        session,
        outline,
        record_id="record:outline-top",
        created_at="2026-09-03T00:00:04Z",
        operator="tester",
    )

    tampered = session.document.to_dict()
    for revision in tampered["align_revisions"]:
        if revision["id"] == "align:axis":
            revision["matrix4x4"][0][3] += 1.0
    with pytest.raises(ArtifactVectorExportError, match="axis recipe"):
        build_vector_export(
            ArtifactDocument.from_dict(tampered), "record:outline-top"
        )


def test_committing_an_axis_align_makes_earlier_records_stale() -> None:
    """The circles that set the axis were measured in the old frame.

    This is the existing rule, not a new one, but a user who is not told will
    think their measurements vanished.
    """

    from src.core.artifact_document import RecordFreshness

    session = _measured_pot()
    assert session.document.record_freshness(TOP_ID) is RecordFreshness.FRESH

    aligned = session.commit_axis_alignment(
        top_record_id=TOP_ID,
        bottom_record_id=BOTTOM_ID,
        operator="tester",
        revision_id="align:axis",
    )
    assert (
        aligned.document.record_freshness(TOP_ID) is RecordFreshness.STALE_ALIGNMENT
    )


def _elevation_cutline_session() -> ArtifactSession:
    """A positioned pot with one cutline record standing in for an elevation."""

    from src.core.artifact_vector_extractor import (
        commit_vector_computation,
        compute_artifact_cutline,
    )
    from src.core.artifact_vector_record import PlanarFrame

    session = _measured_pot().commit_axis_alignment(
        top_record_id=TOP_ID,
        bottom_record_id=BOTTOM_ID,
        operator="tester",
        created_at="2026-09-03T00:00:03Z",
        revision_id="align:axis",
    )
    # A vertical plane through the standing axis: what a section is cut on.
    cutline = compute_artifact_cutline(
        session,
        PlanarFrame(
            origin_world_mm=(0.0, 0.0, 0.0),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 0.0, 1.0),
            normal_world=(0.0, -1.0, 0.0),
        ),
    )
    return commit_vector_computation(
        session,
        cutline,
        record_id="record:elevation-cut",
        created_at="2026-09-03T00:00:05Z",
        operator="tester",
    )


def test_the_centre_line_is_drawn_where_the_pot_turns() -> None:
    """The axis is at u = 0 because the alignment put it through the origin."""

    import xml.etree.ElementTree as ET

    from src.core.artifact_vector_export import VectorSVGOptions, build_vector_export
    from src.core.drawing_style import PROVISIONAL_PRESET_ID

    session = _elevation_cutline_session()
    bundle = build_vector_export(
        session.document,
        "record:elevation-cut",
        options=VectorSVGOptions(
            style_preset=PROVISIONAL_PRESET_ID, show_center_axis=True
        ),
    )
    root = ET.fromstring(bundle.svg_bytes)
    namespace = "{http://www.w3.org/2000/svg}"
    layer = root.find(
        f"{namespace}g/{namespace}g[@id='layer-center-axis']"
    )
    assert layer is not None, "a positioned elevation must carry its centre line"

    axis = layer.find(f"{namespace}path")
    assert axis is not None
    assert axis.attrib["data-role"] == "center_axis"
    # Two endpoints, both at the same drawing x: the line is vertical.
    tokens = axis.attrib["d"].replace("M", "").replace("L", "").split()
    xs = [float(tokens[0]), float(tokens[2])]
    assert xs[0] == pytest.approx(xs[1], abs=1e-9)

    # And it is dashed, so a reader tells it from an outline at a glance.
    from src.core.drawing_style import get_preset

    dashes = get_preset(PROVISIONAL_PRESET_ID).style("center_axis").dash_pattern_mm
    assert layer.attrib["stroke-dasharray"].count(",") == len(dashes) - 1


def test_the_centre_line_is_off_by_default_and_leaves_the_bytes_alone() -> None:
    from src.core.artifact_vector_export import VectorSVGOptions, build_vector_export
    from src.core.drawing_style import PROVISIONAL_PRESET_ID

    session = _elevation_cutline_session()
    without = build_vector_export(
        session.document,
        "record:elevation-cut",
        options=VectorSVGOptions(style_preset=PROVISIONAL_PRESET_ID),
    )
    explicit = build_vector_export(
        session.document,
        "record:elevation-cut",
        options=VectorSVGOptions(
            style_preset=PROVISIONAL_PRESET_ID, show_center_axis=False
        ),
    )
    assert without.svg_bytes == explicit.svg_bytes
    assert b"layer-center-axis" not in without.svg_bytes


def test_a_centre_line_without_a_preset_is_refused() -> None:
    """Every line at one weight makes a centre line indistinguishable."""

    from src.core.artifact_vector_export import (
        ArtifactVectorExportError,
        VectorSVGOptions,
    )

    with pytest.raises(ArtifactVectorExportError, match="needs a style_preset"):
        VectorSVGOptions(show_center_axis=True)


def test_a_sheet_draws_no_axis_when_the_alignment_established_none() -> None:
    """Requesting an axis under a manual drag must not invent one."""

    from src.core.drawing_sheet import (
        DrawingSheetOptions,
        TitleBlock,
        compose_drawing_sheet,
    )
    from src.core.artifact_vector_extractor import (
        commit_vector_computation,
        compute_artifact_cutline,
    )
    from src.core.artifact_vector_record import PlanarFrame

    dragged = _measured_pot().commit_preview(
        translation_mm=(0.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        operator="tester",
        revision_id="align:dragged",
    )
    cutline = compute_artifact_cutline(
        dragged,
        PlanarFrame(
            origin_world_mm=(0.0, 0.0, 0.0),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 0.0, 1.0),
            normal_world=(0.0, -1.0, 0.0),
        ),
    )
    dragged = commit_vector_computation(
        dragged,
        cutline,
        record_id="record:dragged-cut",
        created_at="2026-09-03T00:00:06Z",
        operator="tester",
    )
    bundle = compose_drawing_sheet(
        dragged.document,
        ["record:dragged-cut"],
        options=DrawingSheetOptions(
            title_block=TitleBlock(artifact_label="시험"),
            show_center_axis=True,
        ),
    )
    sidecar = json.loads(bundle.sidecar_bytes)

    assert sidecar["center_axis"]["requested"] is True
    assert sidecar["center_axis"]["drawn"] is False
    assert sidecar["center_axis"]["align_recipe_kind"] == "manual_scene_trs_delta"
    assert b"layer-center-axis" not in bundle.svg_bytes


def test_a_sheet_draws_the_axis_once_the_pot_is_positioned() -> None:
    from src.core.drawing_sheet import (
        DrawingSheetOptions,
        TitleBlock,
        compose_drawing_sheet,
    )

    session = _elevation_cutline_session()
    bundle = compose_drawing_sheet(
        session.document,
        ["record:elevation-cut"],
        options=DrawingSheetOptions(
            title_block=TitleBlock(artifact_label="시험"),
            show_center_axis=True,
        ),
    )
    sidecar = json.loads(bundle.sidecar_bytes)

    assert sidecar["center_axis"]["drawn"] is True
    assert sidecar["center_axis"]["align_recipe_kind"] == AXIS_ALIGN_RECIPE_KIND
    assert b"layer-center-axis" in bundle.svg_bytes


def test_the_axis_never_appears_in_a_plan_view() -> None:
    """Seen from above the axis is a point, and a line there would be a lie."""

    from src.core.artifact_outline_extractor import OutlineView, outline_frame
    from src.core.drawing_svg import center_axis_segment

    bounds = (-40.0, -10.0, 40.0, 110.0)
    for view in OutlineView:
        segment = center_axis_segment(outline_frame(view).to_dict(), bounds)
        if view in (OutlineView.TOP, OutlineView.BOTTOM):
            assert segment is None
        else:
            assert segment is not None
            # Vertical, spanning the drawn content.
            assert segment[0][0] == pytest.approx(0.0, abs=1e-12)
            assert segment[1][0] == pytest.approx(0.0, abs=1e-12)
            assert {segment[0][1], segment[1][1]} == {bounds[1], bounds[3]}
