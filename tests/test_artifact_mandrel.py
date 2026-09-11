"""와통 기준 축: the drum a roof tile was formed on, measured and stood upon.

The fixture tile is moved into an arbitrary scanner pose by a rotation and a
translation the test keeps, so nothing hands the canonical pose to the
development: it has to come out of the measurement, and the test can say how
far from the truth it landed.

Two tiles are used.  One without relief is a true cylinder, so the fit has to
find its drum to the micrometre - that tests the code.  One with the fixture's
cloth relief on the 내면 tests what a real surface does: on a 76-degree arc the
centre and the radius trade along the arc's bisector, so a pattern a fifth of a
millimetre deep moves the best cylinder by a few tenths.  That bound is taken
from the relief's own depth rather than guessed.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from src.application.artifact_measurements import (
    ArtifactMeasurementController,
    ArtifactMeasurementError,
)
from src.application.artifact_workbench import ArtifactWorkbench, RecordBindingTransition
from src.core.artifact_axis_alignment import (
    AXIS_ALIGN_RECIPE_KIND,
    AXIS_SOURCE_CIRCLE_NORMALS,
    verify_axis_alignment_matrix,
)
from src.core.artifact_document import ArtifactDocument
from src.core.artifact_mandrel import (
    FACING_AWAY_FROM_AXIS,
    FACING_TOWARD_AXIS,
    MANDREL_EXTENSION_KEY,
    MANDREL_RECORD_TYPE,
    ArtifactMandrelError,
    mandrel_receipt_from_record,
    mandrel_recipe,
    verify_mandrel_record_against_mesh,
)
from src.core.artifact_record_validation import (
    ArtifactKnownRecordError,
    validate_known_records,
)
from src.core.artifact_session import ArtifactSession, ArtifactSessionError
from src.core.artifact_tile_unwrap_record import TILE_UNWRAP_RECORD_TYPE
from src.core.artifact_tile_unwrap_extractor import (
    SECTION_CENTER_CANONICAL_AXIS,
    tile_unwrap_recipe,
    tile_unwrap_selection,
)
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint

from synthetic_tile import AMKIWA_SHAPE, hollow_tile


STAMP = "2026-09-11T00:00:00Z"
SCAN_ROTATION_AXIS = (1.0, 2.0, 3.0)
SCAN_ROTATION_DEG = 37.0
SCAN_TRANSLATION_MM = np.array([40.0, -120.0, 550.0])


def _rotation(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    unit = unit / float(np.linalg.norm(unit))
    angle = math.radians(degrees)
    cross = np.array(
        [[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


class _Ids:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, prefix: str) -> str:
        self.count += 1
        return f"{prefix}:mandrel-test-{self.count}"


def _publisher(workbench: ArtifactWorkbench):
    def publish(transition: object) -> None:
        assert isinstance(transition, RecordBindingTransition)
        workbench.finalize_record_binding(workbench.activate_record_binding(transition))

    return publish


class _ScanPosedTile:
    """A 암키와 lying where a scanner left it, with the truth kept aside."""

    def __init__(self, *, relief: bool) -> None:
        canonical, faces = hollow_tile(AMKIWA_SHAPE, relief=relief, on_canonical_axis=True)
        radius = np.hypot(canonical[:, 0], canonical[:, 1])
        split = (AMKIWA_SHAPE.inner_radius_mm + AMKIWA_SHAPE.outer_radius_mm) / 2.0
        self.canonical = canonical
        self.faces = faces
        self.canonical_radius = radius
        self.inner = np.flatnonzero((radius[faces] < split).all(axis=1))
        self.outer = np.flatnonzero((radius[faces] > split).all(axis=1))
        self.rotation = _rotation(SCAN_ROTATION_AXIS, SCAN_ROTATION_DEG)
        self.scanned = canonical @ self.rotation.T + SCAN_TRANSLATION_MM
        mesh = MeshData(
            vertices=self.scanned,
            faces=faces,
            unit="mm",
            filepath=Path("/source/scanned-amkiwa.ply"),
            source_identity=SourceFingerprint(
                sha256="7" * 64,
                size_bytes=int(self.scanned.size),
                mtime_ns=1,
                original_name="scanned-amkiwa.ply",
                format="ply",
            ),
            source_format="ply",
            source_import_recipe=current_mesh_import_recipe("ply"),
        )
        self.session = ArtifactSession.create_from_source(
            mesh,
            resolved_source_path="/source/scanned-amkiwa.ply",
            unit="mm",
            axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
            handedness="right",
            software_version="mandrel-test",
            operator="tester",
            created_at=STAMP,
            document_id=f"artifact:scanned-amkiwa-{'relief' if relief else 'smooth'}",
            metadata_revision_id="metadata:scanned-amkiwa",
            align_revision_id="align:scanned-amkiwa-initial",
        ).commit_preview(
            translation_mm=(0.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            pivot_mm=(0.0, 0.0, 0.0),
            operator="tester",
            created_at=STAMP,
            revision_id="align:scanned-amkiwa-as-scanned",
        )

    def truth_direction(self) -> np.ndarray:
        return self.rotation @ np.array([0.0, 0.0, 1.0])

    def measure(self, faces: np.ndarray, session: ArtifactSession | None = None):
        ids = _Ids()
        workbench = ArtifactWorkbench(session=session or self.session, id_factory=ids)
        controller = ArtifactMeasurementController(workbench, id_factory=ids)
        item = controller.begin_mandrel_cylinder(
            selected_face_indices=faces.tolist(),
            created_at=STAMP,
            operator="tester",
        )
        result = controller.execute(item)
        publication = controller.publish_result(item, result, _publisher(workbench))
        measured = workbench.snapshot.session
        assert isinstance(measured, ArtifactSession)
        return measured, publication.record_id


@pytest.fixture(scope="module")
def tile() -> _ScanPosedTile:
    return _ScanPosedTile(relief=True)


@pytest.fixture(scope="module")
def smooth_tile() -> _ScanPosedTile:
    return _ScanPosedTile(relief=False)


@pytest.fixture(scope="module")
def measured_inner(tile: _ScanPosedTile) -> tuple[ArtifactSession, str]:
    return tile.measure(tile.inner)


@pytest.fixture(scope="module")
def measured_smooth(smooth_tile: _ScanPosedTile) -> tuple[ArtifactSession, str]:
    return smooth_tile.measure(smooth_tile.inner)


def _axis(receipt: dict) -> tuple[np.ndarray, np.ndarray, float]:
    point = np.array([float(value) for value in receipt["axis"]["point_mm_decimal"]])
    unit = np.array([float(value) for value in receipt["axis"]["unit_decimal"]])
    return point, unit / float(np.linalg.norm(unit)), float(receipt["radius_mm_decimal"])


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    return math.degrees(math.acos(min(1.0, abs(float(first @ second)))))


def _line_distance(point: np.ndarray, unit: np.ndarray, target: np.ndarray) -> float:
    offset = target - point
    return float(np.linalg.norm(offset - (offset @ unit) * unit))


def test_a_true_cylinder_gives_up_its_drum_to_the_micrometre(
    smooth_tile, measured_smooth
) -> None:
    session, record_id = measured_smooth
    record = session.document.record_index[record_id]
    assert record.type == MANDREL_RECORD_TYPE
    receipt = mandrel_receipt_from_record(record)
    point, unit, radius = _axis(receipt)

    assert _angle_deg(unit, smooth_tile.truth_direction()) < 1e-4
    assert _line_distance(point, unit, SCAN_TRANSLATION_MM) < 0.002
    assert radius == pytest.approx(AMKIWA_SHAPE.inner_radius_mm, abs=0.002)
    assert float(receipt["quality"]["radial_rms_residual_mm_decimal"]) < 0.001
    assert receipt["surface_facing"] == FACING_TOWARD_AXIS
    assert float(receipt["quality"]["arc_span_deg_decimal"]) == pytest.approx(
        AMKIWA_SHAPE.span_deg, abs=0.1
    )
    assert float(receipt["quality"]["length_along_axis_mm_decimal"]) == pytest.approx(
        AMKIWA_SHAPE.length_mm, abs=0.01
    )


def test_relief_moves_the_drum_no_further_than_its_own_depth_allows(
    tile, measured_inner
) -> None:
    session, record_id = measured_inner
    receipt = mandrel_receipt_from_record(session.document.record_index[record_id])
    point, unit, radius = _axis(receipt)
    used = np.unique(tile.faces[tile.inner].reshape(-1))
    depth = float(tile.canonical_radius[used].max() - tile.canonical_radius[used].min())

    # The direction is fixed by the length, which relief does not shorten.
    assert _angle_deg(unit, tile.truth_direction()) < 0.01
    # Centre and radius trade along the arc's bisector, by a few depths.
    assert _line_distance(point, unit, SCAN_TRANSLATION_MM) < 3.0 * depth
    assert abs(radius - float(tile.canonical_radius[used].mean())) < 3.0 * depth
    assert receipt["surface_facing"] == FACING_TOWARD_AXIS


def test_the_outer_face_shares_the_drum_and_faces_away(tile, measured_inner) -> None:
    inner_session, inner_id = measured_inner
    _inner_point, inner_unit, inner_radius = _axis(
        mandrel_receipt_from_record(inner_session.document.record_index[inner_id])
    )
    session, record_id = tile.measure(tile.outer)
    receipt = mandrel_receipt_from_record(session.document.record_index[record_id])
    _point, unit, radius = _axis(receipt)

    assert receipt["surface_facing"] == FACING_AWAY_FROM_AXIS
    assert _angle_deg(unit, inner_unit) < 0.01
    assert radius - inner_radius == pytest.approx(AMKIWA_SHAPE.thickness_mm, abs=0.3)


def test_standing_on_the_drum_puts_its_axis_on_z(smooth_tile, measured_smooth) -> None:
    tile = smooth_tile
    session, record_id = measured_smooth
    aligned = session.commit_mandrel_axis_alignment(
        mandrel_record_id=record_id,
        operator="tester",
        created_at=STAMP,
        revision_id="align:on-the-drum",
    )
    revision = aligned.document.align_revision_index[aligned.document.active_align_revision_id]
    assert revision.recipe["kind"] == AXIS_ALIGN_RECIPE_KIND
    assert revision.recipe["axis_source"] == AXIS_SOURCE_CIRCLE_NORMALS
    assert revision.recipe["top_record_id"] == revision.recipe["bottom_record_id"] == record_id
    parent = aligned.document.align_revision_index[revision.parent_id]
    verify_axis_alignment_matrix(
        recipe=revision.recipe,
        parent_matrix=parent.matrix,
        matrix=revision.matrix,
    )

    # Standing on its drum, every vertex is as far from +Z as it was from the
    # tile's own axis before the scanner moved it.
    stood = np.asarray(aligned.materialize().mesh.vertices, dtype=np.float64)
    radius_about_z = np.hypot(stood[:, 0], stood[:, 1])
    assert float(np.abs(radius_about_z - tile.canonical_radius).max()) < 0.002


def test_the_standing_tile_unrolls_about_its_drum(tile, measured_inner) -> None:
    session, record_id = measured_inner
    aligned = session.commit_mandrel_axis_alignment(
        mandrel_record_id=record_id, operator="tester", created_at=STAMP
    )
    ids = _Ids()
    workbench = ArtifactWorkbench(session=aligned, id_factory=ids)
    controller = ArtifactMeasurementController(workbench, id_factory=ids)
    for faces, view in ((tile.inner, "bottom"), (tile.outer, "top")):
        item = controller.begin_tile_unwrap(
            longitudinal_axis="z",
            record_view=view,
            selected_face_indices=faces.tolist(),
            n_sections=32,
            section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
            created_at=STAMP,
            operator="tester",
        )
        result = controller.execute(item)
        controller.publish_result(item, result, _publisher(workbench))
    final = workbench.snapshot.session
    assert isinstance(final, ArtifactSession)
    document = final.document
    developments = [record for record in document.records if record.type == TILE_UNWRAP_RECORD_TYPE]
    assert len(developments) == 2
    mandrel = document.record_index[record_id]
    # Measured on the surface it then unrolled: one face set, one digest.
    assert any(record.selection_hash == mandrel.selection_hash for record in developments)
    validate_known_records(document)


def test_the_selection_is_written_the_way_a_development_writes_it(tile) -> None:
    total = int(tile.faces.shape[0])
    development = tile_unwrap_recipe(
        longitudinal_axis="z",
        record_view="bottom",
        total_face_count=total,
        selected_face_indices=tile.inner,
    )
    assert tile_unwrap_selection(
        total_face_count=total, selected_face_indices=tile.inner
    ) == development["selection"]
    recipe = mandrel_recipe(total_face_count=total, selected_face_indices=tile.inner)
    assert recipe["selection"] == development["selection"]


def test_a_selection_across_both_faces_is_refused(tile) -> None:
    both = np.concatenate([tile.inner, tile.outer])
    with pytest.raises(ArtifactMeasurementError, match="both faces of the wall"):
        tile.measure(both)


def test_a_strip_too_narrow_to_fix_the_radius_is_refused(tile) -> None:
    centres = tile.canonical[tile.faces[tile.inner]].mean(axis=1)
    angle = np.degrees(np.arctan2(centres[:, 0], centres[:, 1]))
    angle -= float(np.median(angle))
    narrow = tile.inner[np.abs(angle) < 4.0]
    with pytest.raises(ArtifactMeasurementError, match="round its drum"):
        tile.measure(narrow)


def test_a_handful_of_faces_is_not_a_drum(tile) -> None:
    with pytest.raises(ArtifactMeasurementError, match="at least"):
        tile.measure(tile.inner[:10])


def test_a_drum_measured_under_an_earlier_align_is_refused(tile, measured_inner) -> None:
    session, record_id = measured_inner
    moved = session.commit_preview(
        translation_mm=(1.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="tester",
        created_at=STAMP,
        revision_id="align:nudged",
    )
    with pytest.raises(ArtifactSessionError, match="different Align"):
        moved.commit_mandrel_axis_alignment(
            mandrel_record_id=record_id, operator="tester", created_at=STAMP
        )


def test_only_a_drum_record_can_stand_a_tile(measured_inner) -> None:
    session, _record_id = measured_inner
    with pytest.raises(ArtifactSessionError, match="does not exist"):
        session.commit_mandrel_axis_alignment(
            mandrel_record_id="record:nowhere", operator="tester", created_at=STAMP
        )


def test_the_record_survives_a_document_round_trip(measured_inner) -> None:
    session, record_id = measured_inner
    reopened = ArtifactDocument.from_dict(session.document.to_dict())
    validate_known_records(reopened)
    assert mandrel_receipt_from_record(
        reopened.record_index[record_id]
    ) == mandrel_receipt_from_record(session.document.record_index[record_id])


@pytest.mark.parametrize(
    "tamper",
    ["radius", "facing", "section_off_axis"],
)
def test_a_tampered_receipt_is_refused(measured_inner, tamper: str) -> None:
    session, record_id = measured_inner
    document = copy.deepcopy(session.document.to_dict())
    record = next(item for item in document["records"] if item["id"] == record_id)
    receipt = record["extensions"][MANDREL_EXTENSION_KEY]["receipt"]
    if tamper == "radius":
        receipt["radius_mm_decimal"] = "999.000000"
    elif tamper == "facing":
        receipt["surface_facing"] = FACING_AWAY_FROM_AXIS
    else:
        receipt["sections"]["top"]["center_mm_decimal"][0] = "0.000000"
    with pytest.raises((ArtifactKnownRecordError, ArtifactMandrelError)):
        validate_known_records(ArtifactDocument.from_dict(document))


def test_the_claim_is_rechecked_against_the_surface(tile, measured_inner) -> None:
    session, record_id = measured_inner
    record = session.document.record_index[record_id]
    projection = session.materialize()
    verify_mandrel_record_against_mesh(record, projection.mesh.vertices, projection.mesh.faces)

    # The same stored drum does not fit a surface half a millimetre proud of it.
    vertices = np.asarray(projection.mesh.vertices, dtype=np.float64).copy()
    point, unit, _radius = _axis(mandrel_receipt_from_record(record))
    offset = vertices - point
    radial = offset - np.outer(offset @ unit, unit)
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    with pytest.raises(ArtifactMandrelError, match="receipt claims"):
        verify_mandrel_record_against_mesh(
            record, vertices + 0.5 * radial, projection.mesh.faces
        )
