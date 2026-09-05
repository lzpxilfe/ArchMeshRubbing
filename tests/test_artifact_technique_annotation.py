from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import pytest

from src.core.artifact_condition_annotation import CONDITION_RECORD_TYPE
from src.core.artifact_record_validation import (
    ArtifactKnownRecordError,
    validate_known_records,
)
from src.core.artifact_session import ArtifactSession
from src.core.artifact_technique_annotation import (
    ArtifactTechniqueAnnotationError,
    TECHNIQUE_KIND_LABELS_KO,
    TECHNIQUE_KINDS,
    TECHNIQUE_PAYLOAD_EXTENSION_KEY,
    TECHNIQUE_RECORD_TYPE,
    commit_technique_annotation,
    compute_technique_annotation,
    technique_payload_from_record,
    technique_recipe,
    technique_selection,
    validate_technique_recipe,
)
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.project_file import load_artifact_project, save_artifact_project
from src.core.source_identity import SourceFingerprint


STAMP = "2026-09-04T00:00:00Z"
COMMITTED_AT = "2026-09-04T00:00:01Z"
RECORD_ID = "record:technique-1"


def _session(half_extent_mm: float = 10.0) -> ArtifactSession:
    """A tetrahedron: four faces, none of them axis-parallel."""

    h = float(half_extent_mm)
    mesh = MeshData(
        vertices=np.array(
            [[h, h, h], [-h, -h, h], [-h, h, -h], [h, -h, -h]],
            dtype=np.float64,
        ),
        faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32),
        unit="mm",
        source_identity=SourceFingerprint(
            sha256="e" * 64,
            size_bytes=64,
            mtime_ns=1,
            original_name="technique.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    return ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/technique.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="technique-test",
        operator="tester",
        created_at=STAMP,
        document_id="artifact:technique-test",
        metadata_revision_id="metadata:technique-test",
        align_revision_id="align:technique-test",
    )


def _committed(technique: str = "finger_mark", faces: object = (0, 2)) -> ArtifactSession:
    session = _session()
    computation = compute_technique_annotation(
        session,
        technique=technique,
        face_indices=faces,
    )
    return commit_technique_annotation(
        session,
        computation,
        record_id=RECORD_ID,
        created_at=COMMITTED_AT,
        operator="tester",
    )


# --- the vocabulary -----------------------------------------------------------


def test_the_technique_vocabulary_is_closed_and_named() -> None:
    assert TECHNIQUE_KINDS == (
        "board_finishing",
        "burnishing",
        "coil_joint",
        "finger_mark",
        "interior_anvil",
        "paddling",
        "paring",
        "water_smoothing",
        "wood_grain_smoothing",
    )
    assert set(TECHNIQUE_KIND_LABELS_KO) == set(TECHNIQUE_KINDS)
    assert TECHNIQUE_KIND_LABELS_KO["finger_mark"] == "지두흔"
    assert TECHNIQUE_KIND_LABELS_KO["coil_joint"] == "테쌓기흔"
    assert TECHNIQUE_KIND_LABELS_KO["burnishing"] == "마연흔"
    assert TECHNIQUE_KIND_LABELS_KO["interior_anvil"] == "내박자흔"

    selection = technique_selection(total_face_count=4, face_indices=(0,))
    with pytest.raises(ArtifactTechniqueAnnotationError, match="technique must be one of"):
        technique_recipe(technique="slipping", precision_grid_mm=0.01, selection=selection)
    # A condition kind is not a technique, whatever the two have in common.
    with pytest.raises(ArtifactTechniqueAnnotationError, match="technique must be one of"):
        technique_recipe(technique="missing", precision_grid_mm=0.01, selection=selection)


def test_the_recipe_is_the_whole_mark() -> None:
    selection = technique_selection(total_face_count=4, face_indices=(2, 0, 0))
    recipe = validate_technique_recipe(
        technique_recipe(technique="paddling", precision_grid_mm=0.05, selection=selection)
    )
    assert recipe["kind"] == "technique_annotation"
    assert recipe["technique"] == "paddling"
    assert recipe["selection"]["face_ranges"] == [[0, 1], [2, 3]]
    assert recipe["selection"]["selected_face_count"] == 2
    with pytest.raises(ArtifactTechniqueAnnotationError):
        technique_selection(total_face_count=4, face_indices=(0, 4))
    with pytest.raises(ArtifactTechniqueAnnotationError):
        technique_selection(total_face_count=4, face_indices=())


def test_a_mark_is_projected_with_the_closing_and_an_old_recipe_without() -> None:
    """1.1.0 closes the lattice union; a 1.0.0 recipe recomputes as written."""

    from src.core.artifact_technique_annotation import TECHNIQUE_ALGORITHM_VERSIONS

    selection = technique_selection(total_face_count=4, face_indices=(0,))
    current = technique_recipe(technique="paddling", precision_grid_mm=0.05, selection=selection)
    assert current["algorithm_version"] == "1.3.0"
    old = technique_recipe(
        technique="paddling", precision_grid_mm=0.05, selection=selection, algorithm_version="1.0.0"
    )
    assert validate_technique_recipe(old)["algorithm_version"] == "1.0.0"
    assert TECHNIQUE_ALGORITHM_VERSIONS == ("1.0.0", "1.1.0", "1.2.0", "1.3.0")
    with pytest.raises(ArtifactTechniqueAnnotationError, match="algorithm version"):
        technique_recipe(
            technique="paddling", precision_grid_mm=0.05, selection=selection, algorithm_version="2.0.0"
        )
    # Both compute on the same face; the closing changes nothing on a clean
    # triangle, so the boundaries agree - the contract differs, not the shape.
    session = _session()
    from src.core.artifact_technique_annotation import project_technique_from_recipe

    mesh = session.materialize().mesh
    new_payload = project_technique_from_recipe(mesh.vertices, mesh.faces, current)
    old_payload = project_technique_from_recipe(mesh.vertices, mesh.faces, old)
    assert [v.view for v in new_payload.views] == [v.view for v in old_payload.views]


# --- the record -----------------------------------------------------------------


def test_a_committed_mark_carries_the_region_and_validates() -> None:
    session = _committed()
    record = session.document.record_index[RECORD_ID]

    assert record.type == TECHNIQUE_RECORD_TYPE
    assert record.type != CONDITION_RECORD_TYPE
    assert record.selection_hash == record.recipe["selection"]["selection_sha256"]
    validate_known_records(session.document)

    payload = technique_payload_from_record(record)
    assert list(payload.face_indices()) == [0, 2]
    assert payload.technique == "finger_mark"
    assert payload.face_count == 2
    # A tetrahedron face is seen from every direction it faces; the mark
    # projects into some views and accounts for the rest.
    assert payload.views
    qc = payload.qc_summary()
    assert qc["face_count"] == 2
    assert qc["projected_view_count"] == len(payload.views)


def test_the_face_set_survives_saving_and_reopening() -> None:
    document = _committed("coil_joint").document

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "technique.amr"
        save_artifact_project(path, document)
        loaded = load_artifact_project(path)

    reopened = technique_payload_from_record(loaded.record_index[RECORD_ID])
    assert list(reopened.face_indices()) == [0, 2]
    assert reopened.technique == "coil_joint"
    assert reopened == technique_payload_from_record(document.record_index[RECORD_ID])
    assert loaded.canonical_sha256 == document.canonical_sha256


def test_a_tampered_mark_is_refused_on_read_and_in_the_registry() -> None:
    document = _committed().document
    record = document.record_index[RECORD_ID]
    descriptor = dict(record.extensions[TECHNIQUE_PAYLOAD_EXTENSION_KEY])
    payload = dict(descriptor["payload"])
    selection = dict(payload["selection"])
    selection["face_ranges"] = [[1, 2], [3, 4]]
    payload["selection"] = selection
    descriptor["payload"] = payload
    tampered = replace(record, extensions={TECHNIQUE_PAYLOAD_EXTENSION_KEY: descriptor})

    with pytest.raises(ArtifactTechniqueAnnotationError, match="SHA-256"):
        technique_payload_from_record(tampered)

    tampered_document = replace(document, records=(tampered,))
    with pytest.raises(ArtifactKnownRecordError, match="SHA-256"):
        validate_known_records(tampered_document)


def test_a_relabelled_mark_is_refused() -> None:
    """The kind is inside the digest: a finger mark cannot be re-read as paddling."""

    document = _committed().document
    record = document.record_index[RECORD_ID]
    descriptor = dict(record.extensions[TECHNIQUE_PAYLOAD_EXTENSION_KEY])
    payload = dict(descriptor["payload"])
    payload["technique"] = "paddling"
    descriptor["payload"] = payload
    tampered = replace(record, extensions={TECHNIQUE_PAYLOAD_EXTENSION_KEY: descriptor})

    with pytest.raises(ArtifactTechniqueAnnotationError):
        technique_payload_from_record(tampered)


def test_technique_records_do_not_change_the_completion_gate() -> None:
    """A mark is information added to a survey, not a condition of one."""

    from src.application.artifact_workflow_progress import (  # noqa: PLC0415
        derive_artifact_workflow_progress,
    )

    session = _session()
    annotated = commit_technique_annotation(
        session,
        compute_technique_annotation(session, technique="paddling", face_indices=[1]),
        record_id=RECORD_ID,
        created_at=COMMITTED_AT,
        operator="tester",
    )
    before = derive_artifact_workflow_progress(session, align_ready=True)
    after = derive_artifact_workflow_progress(annotated, align_ready=True)
    assert after == before


# --- which side of the wall, and which way the tool moved ----------------------


def test_the_record_says_which_side_of_the_wall_a_mark_is_on() -> None:
    """A coil seam or a finger press is usually seen inside the pot, where the
    wall was not smoothed over; the record decides from the mesh, not the
    drafter, so a sheet can put the mark on the section half."""

    from synthetic_vessel import hollow_vessel

    from src.core.artifact_technique_annotation import (
        SURFACE_EXTERIOR,
        SURFACE_INTERIOR,
        SURFACE_MIXED,
        surface_side_of_faces,
    )

    vertices, faces, _rim, _floor = hollow_vessel(segments=24, rings=10)
    outer_count = 10 * 24 * 2
    outer = list(range(0, 48))
    inner = list(range(outer_count, outer_count + 48))
    assert surface_side_of_faces(vertices, faces, outer) == (SURFACE_EXTERIOR, 0)
    assert surface_side_of_faces(vertices, faces, inner) == (SURFACE_INTERIOR, 1_000_000)
    side, fraction = surface_side_of_faces(vertices, faces, outer + inner)
    assert side == SURFACE_MIXED
    assert 300_000 < fraction < 700_000
    with pytest.raises(ArtifactTechniqueAnnotationError, match="outside the geometry"):
        surface_side_of_faces(vertices, faces, [len(faces)])
    with pytest.raises(ArtifactTechniqueAnnotationError, match="at least one face"):
        surface_side_of_faces(vertices, faces, [])

    # The committed payload carries it, digests it, and reports it in QC.
    session = _committed("coil_joint")
    payload = technique_payload_from_record(session.document.record_index[RECORD_ID])
    assert payload.schema_version == "1.2.0"
    assert payload.surface_side == SURFACE_EXTERIOR
    assert payload.interior_face_fraction_millionths == 0
    assert payload.direction_deg is None
    qc = payload.qc_summary()
    assert qc["surface_side"] == SURFACE_EXTERIOR
    assert qc["direction_deg"] is None


def test_a_direction_the_drafter_observed_travels_in_the_recipe_and_payload() -> None:
    selection = technique_selection(total_face_count=4, face_indices=(0,))
    plain = technique_recipe(technique="wood_grain_smoothing", precision_grid_mm=0.05, selection=selection)
    assert "direction_deg" not in plain
    turned = technique_recipe(
        technique="wood_grain_smoothing", precision_grid_mm=0.05, selection=selection, direction_deg=210.0
    )
    # Degrees on the paper: 210 is the same direction as 30.
    assert turned["direction_deg"] == 30.0
    assert validate_technique_recipe(turned)["direction_deg"] == 30.0
    with pytest.raises(ArtifactTechniqueAnnotationError, match="direction_deg"):
        technique_recipe(
            technique="wood_grain_smoothing",
            precision_grid_mm=0.05,
            selection=selection,
            direction_deg=float("nan"),
        )

    session = _session()
    computation = compute_technique_annotation(
        session, technique="wood_grain_smoothing", face_indices=[0], direction_deg=45.0
    )
    committed = commit_technique_annotation(
        session, computation, record_id=RECORD_ID, created_at=COMMITTED_AT, operator="tester"
    )
    payload = technique_payload_from_record(committed.document.record_index[RECORD_ID])
    assert payload.direction_deg == 45.0
    assert committed.document.record_index[RECORD_ID].recipe["direction_deg"] == 45.0


def test_the_four_surface_finishing_kinds_need_a_1_2_0_payload() -> None:
    """A new kind is a new closed value set, so it needs its own version.

    The keys do not change - a 1.2.0 payload has the shape of a 1.1.0 one -
    but a reader that knows only 1.1.0 would be handed 마연흔 and left to
    guess what it is.  So the four are refused in an older payload, and every
    payload already written reads and digests as it was written.
    """

    from src.core.artifact_technique_annotation import (
        TECHNIQUE_KINDS_SINCE_1_2,
        TechniqueAnnotationPayload,
    )

    assert TECHNIQUE_KINDS_SINCE_1_2 == (
        "board_finishing",
        "burnishing",
        "interior_anvil",
        "paring",
    )
    assert set(TECHNIQUE_KINDS_SINCE_1_2) <= set(TECHNIQUE_KINDS)

    current = technique_payload_from_record(
        _committed("burnishing").document.record_index[RECORD_ID]
    )
    assert current.schema_version == "1.2.0"
    assert current.technique == "burnishing"

    for kind in TECHNIQUE_KINDS_SINCE_1_2:
        with pytest.raises(ArtifactTechniqueAnnotationError, match="needs a 1.2.0 payload"):
            TechniqueAnnotationPayload(
                schema_version="1.1.0",
                technique=kind,
                selection=current.selection,
                views=current.views,
                skipped_views=current.skipped_views,
                surface_side=current.surface_side,
                interior_face_fraction_millionths=(
                    current.interior_face_fraction_millionths
                ),
            )
        with pytest.raises(ArtifactTechniqueAnnotationError, match="needs a 1.2.0 payload"):
            TechniqueAnnotationPayload(
                schema_version="1.0.0",
                technique=kind,
                selection=current.selection,
                views=current.views,
                skipped_views=current.skipped_views,
            )

    # A kind the older versions always knew still writes in an older payload.
    older = TechniqueAnnotationPayload(
        schema_version="1.1.0",
        technique="coil_joint",
        selection=current.selection,
        views=current.views,
        skipped_views=current.skipped_views,
        surface_side=current.surface_side,
        interior_face_fraction_millionths=current.interior_face_fraction_millionths,
    )
    assert older.schema_version == "1.1.0"


def test_a_1_0_0_payload_still_reads_and_digests_as_it_was_written() -> None:
    from src.core.artifact_technique_annotation import TechniqueAnnotationPayload

    current = technique_payload_from_record(_committed().document.record_index[RECORD_ID])
    old = TechniqueAnnotationPayload(
        schema_version="1.0.0",
        technique=current.technique,
        selection=current.selection,
        views=current.views,
        skipped_views=current.skipped_views,
    )
    encoded = old.to_dict()
    assert set(encoded) == {"schema_version", "selection", "skipped_views", "technique", "views"}
    assert TechniqueAnnotationPayload.from_dict(encoded) == old
    assert old.sha256 != current.sha256
    assert "surface_side" not in old.qc_summary()
    # The new keys are not optional on the new version, and forbidden on the old.
    with pytest.raises(ArtifactTechniqueAnnotationError, match="surface_side"):
        TechniqueAnnotationPayload(
            schema_version="1.1.0",
            technique=current.technique,
            selection=current.selection,
            views=current.views,
            skipped_views=current.skipped_views,
        )
    with pytest.raises(ArtifactTechniqueAnnotationError, match="1.0.0"):
        TechniqueAnnotationPayload(
            schema_version="1.0.0",
            technique=current.technique,
            selection=current.selection,
            views=current.views,
            skipped_views=current.skipped_views,
            surface_side="exterior",
        )


def test_faces_seen_edge_on_are_left_out_of_that_view() -> None:
    """A band that reaches the silhouette is drawn where it faces the viewer;
    the edge-on faces at the silhouette are not a sliver anyone would draw."""

    from synthetic_vessel import hollow_vessel

    from src.core.artifact_technique_annotation import project_technique_from_recipe

    segments = 72
    vertices, faces, _rim, _floor = hollow_vessel(segments=segments, rings=10)
    # Outer band faces of one ring run right round the pot.  The front view
    # (along y) sees the faces near angle 0 and 180 edge-on.
    ring = 5
    all_round = [(ring * segments + seg) * 2 + k for seg in range(segments) for k in (0, 1)]
    selection = technique_selection(total_face_count=len(faces), face_indices=all_round)
    current = technique_recipe(technique="coil_joint", precision_grid_mm=0.5, selection=selection)
    payload = project_technique_from_recipe(vertices, faces, current)
    front = payload.boundary_for_view("front")
    assert front is not None
    xs = [x for path in front.outline.paths for x, _ in path.points_mm]
    radius = max(abs(float(v[0])) for v in vertices)
    # Not the full silhouette width: the grazing edge was left out.
    assert max(xs) - min(xs) < 2.0 * radius - 1.0

    # A set made only of edge-on faces has nothing to show in that view, and
    # says so with its own reason rather than failing the record.
    edge_on = [
        (ring * segments + seg) * 2 + k
        for seg in (0, segments - 1, segments // 2 - 1, segments // 2)
        for k in (0, 1)
    ]
    selection = technique_selection(total_face_count=len(faces), face_indices=edge_on)
    recipe = technique_recipe(technique="coil_joint", precision_grid_mm=0.5, selection=selection)
    payload = project_technique_from_recipe(vertices, faces, recipe)
    reasons = {entry["view"]: entry["reason"] for entry in payload.skipped_views}
    assert reasons.get("front") == "grazing_view"
    assert reasons.get("back") == "grazing_view"
    assert payload.boundary_for_view("left") is not None


def test_a_region_unioned_whole_matches_the_outline_union_within_a_cell() -> None:
    """extract_region_geometry is the outline's lattice union done once on the
    whole region: on a smooth band the two agree to a cell's worth of area."""

    from synthetic_vessel import hollow_vessel

    from src.core.artifact_outline_extractor import (
        REGION_ALGORITHM,
        extract_outline_geometry,
        extract_region_geometry,
    )

    segments = 72
    vertices, faces, _rim, _floor = hollow_vessel(segments=segments, rings=10)
    ring = 5
    front_facing = [(ring * segments + seg) * 2 + k for seg in range(48, 62) for k in (0, 1)]
    subset = faces[front_facing]
    payload, qc = extract_region_geometry(vertices, subset, "front", precision_grid_mm=0.5)
    outline = extract_outline_geometry(
        vertices, subset, "front", precision_grid_mm=0.5, algorithm_version="1.1.0"
    )
    assert qc["algorithm"] == REGION_ALGORITHM
    assert qc["grid_closing_radius_cells"] == 1
    assert payload.frame == outline.payload.frame
    region_area = float(qc["outline_area_mm2"])
    outline_area = float(outline.qc["outline_area_mm2"])
    assert abs(region_area - outline_area) <= 0.05 * outline_area
    for path in payload.paths:
        assert path.role in ("exterior", "hole")
