"""펼 수 있는 기록면: what a development can carry of a surface that undercuts.

A lip is planted at a known place on the fixture tile by pushing one column of
the sheet round the drum past its neighbour, which turns the faces between
them over.  The painted sheet is then refused by the development, the helper
has to take out that place and nothing else, and what it keeps has to pass the
development's own gates.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from src.application.artifact_measurements import ArtifactMeasurementController
from src.application.artifact_workbench import ArtifactWorkbench, RecordBindingTransition
from src.core.artifact_document import ArtifactDocument
from src.core.artifact_mandrel import FACING_AWAY_FROM_AXIS
from src.core.artifact_record_validation import (
    ArtifactKnownRecordError,
    validate_known_records,
)
from src.core.artifact_session import ArtifactSession
from src.core.artifact_tile_unwrap_extractor import (
    SECTION_CENTER_CANONICAL_AXIS,
    SECTION_CENTER_FIT_PER_SECTION,
    ArtifactTileUnwrapError,
    extract_tile_unwrap,
    selection_face_indices,
    tile_unwrap_recipe,
)
from src.core.artifact_unrollable_surface import (
    UNROLLABLE_EXTENSION_KEY,
    UNROLLABLE_RECORD_TYPE,
    ArtifactUnrollableSurfaceError,
    UnrollableSelectionComputation,
    unrollable_recording_surface,
    unrollable_selection_for_request,
    unrollable_selection_receipt_from_record,
    verify_unrollable_selection_record_against_mesh,
)
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint

from synthetic_tile import AMKIWA_SHAPE, hollow_tile


LIP_HEIGHT_MM = 170.0
LIP_HALF_HEIGHT_MM = 8.0
LIP_SHIFT_MM = 3.0
STAMP = "2026-09-11T00:00:00Z"


def _push_a_column(
    vertices: np.ndarray, faces: np.ndarray, sheet: np.ndarray, height_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Push one column of a sheet round the drum past its neighbour.

    Returns the moved vertices and the lip's centre.  The column keeps its
    radius, so the sheet stays on its drum; only the faces between the moved
    column and the one it was pushed past turn over, the way the wall of an
    undercut does.
    """

    vertices = np.array(vertices, dtype=np.float64, copy=True)
    sheet_vertices = np.unique(faces[sheet].reshape(-1))
    angle = np.arctan2(vertices[sheet_vertices, 1], vertices[sheet_vertices, 0])
    middle = float(np.median(angle))
    height = vertices[sheet_vertices, 2]
    near = np.abs(height - height_mm) <= LIP_HALF_HEIGHT_MM
    # The one column of vertices closest to the sheet's middle meridian.
    distance = np.abs(angle - middle)
    column_angle = angle[near][int(np.argmin(distance[near]))]
    column = sheet_vertices[near & (np.abs(angle - column_angle) < 1e-6)]
    radius = np.hypot(vertices[column, 0], vertices[column, 1])
    moved = angle[np.isin(sheet_vertices, column)] - LIP_SHIFT_MM / radius
    vertices[column, 0] = radius * np.cos(moved)
    vertices[column, 1] = radius * np.sin(moved)
    return vertices, vertices[column].mean(axis=0)


class _StandingTile:
    """The fixture 암키와 standing on its drum, its two sheets told apart."""

    def __init__(self, *, relief: bool = True) -> None:
        vertices, faces = hollow_tile(AMKIWA_SHAPE, relief=relief, on_canonical_axis=True)
        radius = np.hypot(vertices[:, 0], vertices[:, 1])
        split = (AMKIWA_SHAPE.inner_radius_mm + AMKIWA_SHAPE.outer_radius_mm) / 2.0
        self.vertices = vertices
        self.faces = faces
        self.inner = np.flatnonzero((radius[faces] < split).all(axis=1))
        self.outer = np.flatnonzero((radius[faces] > split).all(axis=1))

    def mesh(self) -> MeshData:
        return MeshData(vertices=self.vertices, faces=self.faces, unit="mm")

    def with_a_lip(self) -> tuple["_StandingTile", np.ndarray]:
        """The tile with one column of its outer sheet pushed into a lip.

        Returns the tile and the lip's centre.
        """

        lipped = _StandingTile.__new__(_StandingTile)
        lipped.faces = self.faces
        lipped.inner = self.inner
        lipped.outer = self.outer
        lipped.vertices, centre = _push_a_column(
            self.vertices, self.faces, self.outer, LIP_HEIGHT_MM
        )
        return lipped, centre


@pytest.fixture(scope="module")
def tile() -> _StandingTile:
    return _StandingTile()


@pytest.fixture(scope="module")
def lipped(tile: _StandingTile) -> tuple[_StandingTile, np.ndarray]:
    return tile.with_a_lip()


def _develops(tile: _StandingTile, faces: np.ndarray, view: str) -> None:
    extract_tile_unwrap(
        tile.mesh(),
        tile_unwrap_recipe(
            longitudinal_axis="z",
            record_view=view,
            total_face_count=int(tile.faces.shape[0]),
            selected_face_indices=faces,
            n_sections=32,
            section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
        ),
    )


@pytest.mark.parametrize("sheet, view", [("outer", "top"), ("inner", "bottom")])
def test_a_sheet_that_unrolls_loses_nothing(tile, sheet: str, view: str) -> None:
    requested = getattr(tile, sheet)
    surface = unrollable_recording_surface(
        tile.mesh(), requested, longitudinal_axis="z", record_view=view
    )
    assert surface.excluded_face_indices.size == 0
    assert np.array_equal(surface.face_indices, np.sort(requested))
    assert surface.places == ()
    assert surface.rounds == 1
    assert surface.surface_faces_axis is (sheet == "inner")
    _develops(tile, surface.face_indices, view)


def test_the_lip_is_what_the_development_refuses(lipped) -> None:
    tile, _centre = lipped
    with pytest.raises(ArtifactTileUnwrapError, match="foldover|overlap"):
        _develops(tile, tile.outer, "top")


def test_the_lip_is_left_out_and_the_rest_unrolls(lipped) -> None:
    tile, centre = lipped
    surface = unrollable_recording_surface(
        tile.mesh(), tile.outer, longitudinal_axis="z", record_view="top"
    )
    assert surface.excluded_face_indices.size > 0
    assert surface.folded_face_count > 0
    # One place, where the lip was planted, and a sliver of the sheet.
    assert len(surface.places) >= 1
    nearest = min(
        math.dist(place.centre_mm, tuple(centre)) for place in surface.places
    )
    assert nearest < 2.0 * LIP_HALF_HEIGHT_MM
    for place in surface.places:
        assert math.dist(place.centre_mm, tuple(centre)) < 4.0 * LIP_HALF_HEIGHT_MM
    assert surface.excluded_area_share < 0.01
    assert not surface.surface_faces_axis
    _develops(tile, surface.face_indices, "top")


def test_the_answer_does_not_depend_on_the_run(lipped) -> None:
    tile, _centre = lipped
    first = unrollable_recording_surface(
        tile.mesh(), tile.outer, longitudinal_axis="z", record_view="top"
    )
    second = unrollable_recording_surface(
        tile.mesh(), tile.outer[::-1].copy(), longitudinal_axis="z", record_view="top"
    )
    assert np.array_equal(first.face_indices, second.face_indices)
    assert first.summary() == second.summary()


def test_it_serves_only_a_development_about_a_measured_axis(tile) -> None:
    with pytest.raises(ArtifactUnrollableSurfaceError, match="measured axis"):
        unrollable_recording_surface(
            tile.mesh(),
            tile.outer,
            longitudinal_axis="z",
            record_view="top",
            section_center_policy=SECTION_CENTER_FIT_PER_SECTION,
        )


def test_a_closed_shell_has_nowhere_to_open(tile) -> None:
    everything = np.arange(tile.faces.shape[0])
    with pytest.raises(ArtifactUnrollableSurfaceError):
        unrollable_recording_surface(
            tile.mesh(), everything, longitudinal_axis="z", record_view="top"
        )


# ---------------------------------------------------------------------------
# The record: surface.unrollable_selection.v1
# ---------------------------------------------------------------------------


class _Ids:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, prefix: str) -> str:
        self.count += 1
        return f"{prefix}:unrollable-test-{self.count}"


def _publisher(workbench: ArtifactWorkbench):
    def publish(transition: object) -> None:
        assert isinstance(transition, RecordBindingTransition)
        workbench.finalize_record_binding(workbench.activate_record_binding(transition))

    return publish


def _controller(session: ArtifactSession) -> tuple[ArtifactWorkbench, ArtifactMeasurementController]:
    ids = _Ids()
    workbench = ArtifactWorkbench(session=session, id_factory=ids)
    return workbench, ArtifactMeasurementController(workbench, id_factory=ids)


def _stand_on_its_drum(tile: _StandingTile, name: str) -> ArtifactSession:
    """The tile as a document, stood on the 와통 measured on its inner sheet -
    the only way a development about the axis is allowed to begin."""

    mesh = MeshData(
        vertices=tile.vertices,
        faces=tile.faces,
        unit="mm",
        filepath=Path(f"/source/{name}.ply"),
        source_identity=SourceFingerprint(
            sha256="5" * 64,
            size_bytes=int(tile.vertices.size),
            mtime_ns=1,
            original_name=f"{name}.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path=f"/source/{name}.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="unrollable-test",
        operator="tester",
        created_at=STAMP,
        document_id=f"artifact:{name}",
        metadata_revision_id=f"metadata:{name}",
        align_revision_id=f"align:{name}-initial",
    ).commit_preview(
        translation_mm=(0.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="tester",
        created_at=STAMP,
        revision_id=f"align:{name}-as-scanned",
    )
    workbench, controller = _controller(session)
    item = controller.begin_mandrel_cylinder(
        selected_face_indices=tile.inner.tolist(), created_at=STAMP, operator="tester"
    )
    publication = controller.publish_result(
        item, controller.execute(item), _publisher(workbench)
    )
    measured = workbench.snapshot.session
    assert isinstance(measured, ArtifactSession)
    return measured.commit_mandrel_axis_alignment(
        mandrel_record_id=publication.record_id,
        operator="tester",
        created_at=STAMP,
        revision_id=f"align:{name}-on-the-drum",
    )


@pytest.fixture(scope="module")
def standing_lipped(lipped) -> tuple[ArtifactSession, _StandingTile]:
    tile, _centre = lipped
    return _stand_on_its_drum(tile, "lipped"), tile


@pytest.fixture(scope="module")
def standing_clean(tile) -> ArtifactSession:
    return _stand_on_its_drum(tile, "clean")


def _record_exclusion(session: ArtifactSession, faces: np.ndarray, *, view: str = "top"):
    workbench, controller = _controller(session)
    item = controller.begin_unrollable_selection(
        selected_face_indices=faces.tolist(),
        longitudinal_axis="z",
        record_view=view,
        section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
        created_at=STAMP,
        operator="tester",
    )
    result = controller.execute(item)
    publication = controller.publish_result(item, result, _publisher(workbench))
    computation = result.computation
    assert isinstance(computation, UnrollableSelectionComputation)
    return workbench, controller, publication.record_id, computation


def test_the_record_says_what_was_left_out_and_where(standing_lipped) -> None:
    session, tile = standing_lipped
    workbench, _controller_, record_id, computation = _record_exclusion(session, tile.outer)
    current = workbench.snapshot.session
    assert isinstance(current, ArtifactSession)
    record = current.document.record_index[record_id]
    assert record.type == UNROLLABLE_RECORD_TYPE
    receipt = unrollable_selection_receipt_from_record(record)
    helper = unrollable_recording_surface(
        current.materialize().mesh, tile.outer, longitudinal_axis="z", record_view="top"
    )

    assert receipt["counts"]["excluded_face_count"] == int(helper.excluded_face_indices.size) > 0
    assert receipt["counts"]["kept_face_count"] == helper.face_count
    assert receipt["place_count"] == len(helper.places) >= 1
    assert np.array_equal(computation.kept_face_indices, helper.face_indices)
    # The request stays whole in the recipe; what was left out is named face
    # for face, and its share of the area is the record's QC.
    assert record.recipe["selection"]["selected_face_count"] == int(np.unique(tile.outer).size)
    assert np.array_equal(
        selection_face_indices(receipt["excluded_selection"]), helper.excluded_face_indices
    )
    assert record.qc["excluded_area_share"] == pytest.approx(helper.excluded_area_share, abs=1e-6)
    assert record.qc["excluded_area_share"] < 0.01
    assert record.qc["surface_facing"] == FACING_AWAY_FROM_AXIS


def test_a_sheet_that_unrolls_is_recorded_whole(standing_clean, tile) -> None:
    workbench, _controller_, record_id, computation = _record_exclusion(
        standing_clean, tile.inner, view="bottom"
    )
    current = workbench.snapshot.session
    assert isinstance(current, ArtifactSession)
    record = current.document.record_index[record_id]
    receipt = unrollable_selection_receipt_from_record(record)
    assert receipt["excluded_selection"] is None
    assert receipt["place_count"] == 0 and receipt["places"] == []
    assert record.qc["excluded_face_count"] == 0 and record.qc["excluded_area_share"] == 0.0
    assert np.array_equal(computation.kept_face_indices, np.unique(tile.inner))


def test_the_development_of_what_was_kept_depends_on_the_record(standing_lipped) -> None:
    session, tile = standing_lipped
    workbench, controller, record_id, computation = _record_exclusion(session, tile.outer)
    current = workbench.snapshot.session
    assert isinstance(current, ArtifactSession)
    kept = computation.kept_face_indices
    request = {
        "total_face_count": int(tile.faces.shape[0]),
        "longitudinal_axis": "z",
        "record_view": "top",
        "section_center_policy": SECTION_CENTER_CANONICAL_AXIS,
    }
    assert (
        unrollable_selection_for_request(current.document, selected_face_indices=kept, **request)
        == record_id
    )
    # The painted request is not what was kept, and the kept faces made into
    # another development were not tested for it.
    assert (
        unrollable_selection_for_request(
            current.document, selected_face_indices=tile.outer, **request
        )
        is None
    )
    assert (
        unrollable_selection_for_request(
            current.document, selected_face_indices=kept, n_sections=24, **request
        )
        is None
    )

    item = controller.begin_tile_unwrap(
        longitudinal_axis="z",
        record_view="top",
        selected_face_indices=kept.tolist(),
        section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
        created_at=STAMP,
        operator="tester",
        depends_on_record_ids=(record_id,),
    )
    publication = controller.publish_result(item, controller.execute(item), _publisher(workbench))
    developed = workbench.snapshot.session
    assert isinstance(developed, ArtifactSession)
    development = developed.document.record_index[publication.record_id]
    assert tuple(development.depends_on_record_ids) == (record_id,)
    validate_known_records(developed.document)


def test_a_development_cannot_claim_faces_it_did_not_develop(standing_lipped) -> None:
    session, tile = standing_lipped
    workbench, controller, record_id, _computation = _record_exclusion(session, tile.outer)
    # The inner sheet develops, but it is not what the exclusion kept.
    item = controller.begin_tile_unwrap(
        longitudinal_axis="z",
        record_view="bottom",
        selected_face_indices=tile.inner.tolist(),
        section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
        created_at=STAMP,
        operator="tester",
        depends_on_record_ids=(record_id,),
    )
    result = controller.execute(item)
    with pytest.raises(ValueError) as refused:
        controller.publish_result(item, result, _publisher(workbench))
    assert "does not develop the faces it kept" in str(refused.value)


def test_the_record_survives_a_round_trip_and_a_tampered_one_is_refused(
    standing_lipped,
) -> None:
    session, tile = standing_lipped
    workbench, _controller_, record_id, _computation = _record_exclusion(session, tile.outer)
    current = workbench.snapshot.session
    assert isinstance(current, ArtifactSession)
    document = current.document
    reopened = ArtifactDocument.from_dict(document.to_dict())
    validate_known_records(reopened)
    assert unrollable_selection_receipt_from_record(
        reopened.record_index[record_id]
    ) == unrollable_selection_receipt_from_record(document.record_index[record_id])

    for tamper in ("excluded", "kept", "count"):
        data = copy.deepcopy(document.to_dict())
        record = next(item for item in data["records"] if item["id"] == record_id)
        receipt = record["extensions"][UNROLLABLE_EXTENSION_KEY]["receipt"]
        if tamper == "excluded":
            receipt["excluded_selection"]["face_ranges"][0][1] += 1
        elif tamper == "kept":
            receipt["kept_selection_sha256"] = "0" * 64
        else:
            receipt["counts"]["folded_face_count"] += 1
        with pytest.raises((ArtifactKnownRecordError, ArtifactUnrollableSurfaceError)):
            validate_known_records(ArtifactDocument.from_dict(data))


def test_the_claim_is_rechecked_by_running_the_request_again(standing_lipped) -> None:
    session, tile = standing_lipped
    workbench, _controller_, record_id, _computation = _record_exclusion(session, tile.outer)
    current = workbench.snapshot.session
    assert isinstance(current, ArtifactSession)
    record = current.document.record_index[record_id]
    canonical = current.materialize().mesh
    verify_unrollable_selection_record_against_mesh(record, canonical)

    # A second lip on the same sheet is a place the record does not name.
    vertices, _centre = _push_a_column(canonical.vertices, tile.faces, tile.outer, 100.0)
    with pytest.raises(ArtifactUnrollableSurfaceError, match="another"):
        verify_unrollable_selection_record_against_mesh(
            record, MeshData(vertices=vertices, faces=tile.faces, unit="mm")
        )
