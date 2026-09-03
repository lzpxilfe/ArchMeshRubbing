"""전개 탁본: the relief of a pot strip, drawn on its own unrolled coordinates.

A paper strip on a corded pot takes the cords the way the surface has them.
The six-view rubbing looks at the strip from the side; this one draws the
same relief on the development the tile-unwrap record already proved, so
the raster is the strip a rubber would paste beside the drawing.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pytest

from src.application.artifact_exports import ArtifactExportController
from src.application.artifact_measurements import (
    ArtifactMeasurementController,
    ArtifactMeasurementError,
    MeasurementOperationKind,
)
from src.application.artifact_workbench import (
    ArtifactWorkbench,
    RecordBindingTransition,
)
from src.core.artifact_developed_rubbing import (
    DEVELOPED_RUBBING_RECORD_TYPE,
    ArtifactDevelopedRubbingError,
    DevelopedRubbingComputation,
    DevelopedRubbingRaster,
    commit_developed_rubbing,
    compute_developed_rubbing,
    compute_developed_rubbing_from_recipe,
    developed_rubbing_receipt_from_record,
    developed_rubbing_recipe,
    estimate_developed_rubbing_resources,
    validate_developed_rubbing_receipt,
    validate_developed_rubbing_recipe,
)
from src.core.artifact_document import RecordFreshness
from src.core.artifact_outline_extractor import outline_frame
from src.core.artifact_record_validation import (
    ArtifactKnownRecordError,
    validate_known_records,
)
from src.core.artifact_rubbing_export import (
    ArtifactRubbingExportError,
    build_rubbing_export,
    export_rubbing_package,
    validate_rubbing_export_package,
)
from src.core.artifact_rubbing_extractor import DigitalRubbingRaster
from src.core.artifact_session import ArtifactSession
from src.core.drawing_sheet import (
    DrawingSheetOptions,
    TitleBlock,
    compose_drawing_sheet,
    validate_drawing_sheet_bytes,
)
from src.core.artifact_tile_unwrap_extractor import (
    SECTION_CENTER_CANONICAL_AXIS,
    STATION_MERIDIAN_ARC,
    commit_artifact_tile_unwrap,
    compute_artifact_tile_unwrap,
)
from src.core.artifact_tile_unwrap_record import tile_unwrap_receipt_from_record
from src.core.project_file import load_artifact_project, save_artifact_project
from synthetic_vessel import (
    RIM_ID,
    meridional_strip_faces,
    positioned_vessel_session,
)


UNWRAP_ID = "record:unwrap:strip"
STAMP = "2026-09-03T00:00:10Z"
OPTIONS: dict[str, Any] = {
    "pixels_per_mm": 10,
    "margin_um": 1_000,
    "reference_radius_um": 3_000,
    "depth_quantization_um": 10,
    "black_point_um": 250,
    "ink_strength_percent": 100,
    "relief_polarity": "bidirectional",
}


def cords(angle_rad: float, z_mm: float) -> float:
    """승문: raised cords 0.25 mm high on a 4 mm pitch, running a little diagonal,
    with one shallow incised line across the belly."""

    ridge = 0.25 * max(0.0, math.sin(2.0 * math.pi * (z_mm + 6.0 * angle_rad) / 4.0))
    groove = -0.2 if abs(z_mm - 45.0) < 1.0 else 0.0
    return ridge + groove


def _strip_session(relief=None, *, segments: int = 96, rings: int = 96) -> ArtifactSession:
    session, vertices, faces = positioned_vessel_session(
        segments=segments, rings=rings, relief=relief
    )
    selected = meridional_strip_faces(
        vertices, faces, center_angle_rad=math.pi / 2.0, width_mm=20.0
    )
    unwrap = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="z",
        record_view="top",
        selected_face_indices=selected,
        n_sections=12,
        section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
        station_policy=STATION_MERIDIAN_ARC,
    )
    return commit_artifact_tile_unwrap(
        session, unwrap, record_id=UNWRAP_ID, created_at=STAMP, operator="tester"
    )


@pytest.fixture(scope="module")
def corded() -> ArtifactSession:
    return _strip_session(cords)


@pytest.fixture(scope="module")
def plain() -> ArtifactSession:
    return _strip_session(None)


def _rubbing(session: ArtifactSession, **overrides: Any):
    return compute_developed_rubbing(session, UNWRAP_ID, **{**OPTIONS, **overrides})


def _drop_in_the_middle(raster: DevelopedRubbingRaster) -> np.ndarray:
    """Ink (255 - grey) on covered pixels of the strip's interior.

    Along the rim, the base, and the stepped sides of a painted strip the
    reference window is one-sided, and the profile slope leaks into it as
    tone.  The six-view rubbing has the same edge; neither is the relief.
    """

    gray = raster.pixels[:, :, 0].astype(np.int64)
    alpha = raster.pixels[:, :, 1]
    rows = slice(80, raster.height_pixels - 80)
    quarter = raster.width_pixels // 4
    columns = slice(quarter, raster.width_pixels - quarter)
    return np.where(alpha[rows, columns] == 255, 255 - gray[rows, columns], -1)


def test_the_raster_is_the_strip_at_one_to_one(corded: ArtifactSession) -> None:
    receipt = tile_unwrap_receipt_from_record(corded.document.record_index[UNWRAP_ID])
    computation = _rubbing(corded)
    raster = computation.raster

    # 10 px/mm, 1 mm margin each side; the development's exact bounds decide.
    width_mm = receipt["bounds_um"]["maximum_u"] / 1000.0
    height_mm = receipt["bounds_um"]["maximum_v"] / 1000.0
    assert raster.width_pixels == math.ceil(width_mm * 10.0) + 20
    assert raster.height_pixels == math.ceil(height_mm * 10.0) + 20
    assert 94.0 < height_mm < 95.5  # the meridian, not the 90 mm axis height
    assert raster.receipt()["height_mm_exact"] == {
        "denominator": 10_000,
        "numerator": raster.height_pixels * 1000,
    }
    qc = computation.qc_dict()
    assert qc["development_record_id"] == UNWRAP_ID
    assert qc["development_sha256"] == receipt["unwrap_sha256"]
    assert qc["multi_layer_pixel_count"] == 0
    assert qc["projected_zero_area_face_count"] == 0
    assert qc["radius_min_um_rounded"] == 25_000
    assert 47_500 < qc["radius_max_um_rounded"] < 48_500


def test_cords_ink_and_a_plain_wall_stays_light(
    corded: ArtifactSession, plain: ArtifactSession
) -> None:
    corded_drop = _drop_in_the_middle(_rubbing(corded).raster)
    plain_drop = _drop_in_the_middle(_rubbing(plain).raster)

    corded_ink = corded_drop[corded_drop >= 0]
    plain_ink = plain_drop[plain_drop >= 0]
    # The profile's own curvature leaves a faint tone (a few dozen levels at
    # most); the cords go well past half black.
    assert plain_ink.max() < 60
    assert corded_ink.max() > 120
    assert corded_ink.mean() > 3.0 * plain_ink.mean()


def test_cords_repeat_down_the_strip_at_their_pitch(corded: ArtifactSession) -> None:
    raster = _rubbing(corded, relief_polarity="raised").raster
    drop = _drop_in_the_middle(raster)
    profile = np.where(drop >= 0, drop, 0).sum(axis=1).astype(np.float64)
    profile -= profile.mean()
    spectrum = np.abs(np.fft.rfft(profile))
    frequencies = np.fft.rfftfreq(profile.size, d=0.1)  # rows are 0.1 mm
    peak_mm = 1.0 / frequencies[1:][int(np.argmax(spectrum[1:]))]
    assert 3.5 < peak_mm < 4.5


def test_raised_polarity_inks_the_crests_more_than_incised_does(
    corded: ArtifactSession,
) -> None:
    raised = _rubbing(corded, relief_polarity="raised").qc_dict()["ink_sum"]
    incised = _rubbing(corded, relief_polarity="incised").qc_dict()["ink_sum"]
    both = _rubbing(corded).qc_dict()["ink_sum"]
    assert raised > incised
    assert both == raised + incised


def test_the_recipe_reproduces_the_raster_exactly(corded: ArtifactSession) -> None:
    first = _rubbing(corded)
    second = compute_developed_rubbing_from_recipe(corded, first.recipe_dict())
    assert second.raster.raster_sha256 == first.raster.raster_sha256
    assert second.raster.pixels.tobytes() == first.raster.pixels.tobytes()
    assert validate_developed_rubbing_recipe(first.recipe_dict()) == first.recipe_dict()


def test_the_record_names_its_development_by_hash_and_depends_on_it(
    corded: ArtifactSession,
) -> None:
    computation = _rubbing(corded)
    session = commit_developed_rubbing(
        corded,
        computation,
        record_id="record:developed:strip",
        created_at="2026-09-03T00:00:11Z",
        operator="tester",
    )
    record = session.document.record_index["record:developed:strip"]
    unwrap_receipt = tile_unwrap_receipt_from_record(
        session.document.record_index[UNWRAP_ID]
    )

    assert record.type == DEVELOPED_RUBBING_RECORD_TYPE
    assert record.depends_on_record_ids == (UNWRAP_ID,)
    assert record.recipe["development"]["unwrap_sha256"] == unwrap_receipt["unwrap_sha256"]
    assert session.document.record_freshness(record.id) is RecordFreshness.FRESH
    validate_known_records(session.document)
    receipt = developed_rubbing_receipt_from_record(record)
    assert receipt == computation.raster.receipt()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pot.amr"
        save_artifact_project(path, session.document)
        loaded = load_artifact_project(path)
    validate_known_records(loaded)
    assert developed_rubbing_receipt_from_record(loaded.record_index[record.id]) == receipt
    # The dependency is what makes a later re-alignment mark the rubbing stale
    # together with its development.
    dragged = session.activate_parent_align()
    assert dragged.document.record_freshness(UNWRAP_ID) is not RecordFreshness.FRESH
    assert dragged.document.record_freshness(record.id) is not RecordFreshness.FRESH


def test_a_development_the_record_no_longer_carries_is_refused(
    corded: ArtifactSession, plain: ArtifactSession
) -> None:
    corded_receipt = tile_unwrap_receipt_from_record(
        corded.document.record_index[UNWRAP_ID]
    )
    plain_record = plain.document.record_index[UNWRAP_ID]
    assert corded_receipt["unwrap_sha256"] != tile_unwrap_receipt_from_record(
        plain_record
    )["unwrap_sha256"]
    forged = developed_rubbing_recipe(
        development_record_id=UNWRAP_ID,
        development_sha256=corded_receipt["unwrap_sha256"],
        development_recipe_hash=plain_record.recipe_hash,
        **OPTIONS,
    )
    with pytest.raises(ArtifactDevelopedRubbingError, match="no longer carries"):
        compute_developed_rubbing_from_recipe(plain, forged)


def test_only_a_ready_fresh_tile_unwrap_can_be_developed(
    corded: ArtifactSession,
) -> None:
    with pytest.raises(ArtifactDevelopedRubbingError, match="not a tile unwrap"):
        compute_developed_rubbing(corded, RIM_ID, **OPTIONS)
    with pytest.raises(ArtifactDevelopedRubbingError, match="does not exist"):
        compute_developed_rubbing(corded, "record:unwrap:nowhere", **OPTIONS)
    dragged = corded.activate_parent_align()
    with pytest.raises(ArtifactDevelopedRubbingError, match="not FRESH"):
        compute_developed_rubbing(dragged, UNWRAP_ID, **OPTIONS)


def test_receipt_and_recipe_validation_fail_closed(corded: ArtifactSession) -> None:
    computation = _rubbing(corded)
    receipt = computation.raster.receipt()
    assert validate_developed_rubbing_receipt(receipt) == receipt

    wider = {**receipt, "width_pixels": receipt["width_pixels"] + 1}
    with pytest.raises(ArtifactDevelopedRubbingError, match="byte length"):
        validate_developed_rubbing_receipt(wider)
    other = {**receipt, "development_sha256": "0" * 64}
    assert validate_developed_rubbing_receipt(other)["development_sha256"] == "0" * 64

    # A raster that claims another development cannot ride on this recipe.
    elsewhere = DevelopedRubbingRaster(
        pixels=computation.raster.pixels,
        pixels_per_meter=computation.raster.pixels_per_meter,
        minimum_u_pixel_index=computation.raster.minimum_u_pixel_index,
        minimum_v_pixel_index=computation.raster.minimum_v_pixel_index,
        development_sha256="0" * 64,
    )
    with pytest.raises(ArtifactDevelopedRubbingError, match="another development"):
        DevelopedRubbingComputation(
            context=computation.context,
            projection_snapshot=computation.projection_snapshot,
            raster=elsewhere,
            recipe=computation.recipe,
            qc=computation.qc,
        )

    recipe = computation.recipe_dict()
    recipe["pixel_policy"] = {**recipe["pixel_policy"], "pixels_per_mm": 11}
    with pytest.raises(ArtifactDevelopedRubbingError, match="production contract"):
        validate_developed_rubbing_recipe(recipe)


def test_a_tampered_document_does_not_validate(corded: ArtifactSession) -> None:
    computation = _rubbing(corded)
    session = commit_developed_rubbing(
        corded,
        computation,
        record_id="record:developed:strip",
        created_at="2026-09-03T00:00:11Z",
        operator="tester",
    )
    payload = session.document.to_dict()
    for record in payload["records"]:
        if record["id"] == "record:developed:strip":
            record["depends_on_record_ids"] = []
    from src.core.artifact_document import ArtifactDocument  # noqa: PLC0415

    tampered = ArtifactDocument.from_dict(payload)
    with pytest.raises(ArtifactKnownRecordError, match="depend on its development"):
        validate_known_records(tampered)


def test_the_estimate_knows_the_artboard_before_any_work(corded: ArtifactSession) -> None:
    computation = _rubbing(corded)
    receipt = tile_unwrap_receipt_from_record(corded.document.record_index[UNWRAP_ID])
    estimate = estimate_developed_rubbing_resources(
        receipt,
        computation.recipe_dict(),
        source_vertex_count=int(corded.source_mesh.vertices.shape[0]),
        source_face_count=int(corded.source_mesh.faces.shape[0]),
        source_geometry_bytes=int(
            corded.source_mesh.vertices.nbytes + corded.source_mesh.faces.nbytes
        ),
    )
    assert (estimate.width_pixels, estimate.height_pixels) == (
        computation.raster.width_pixels,
        computation.raster.height_pixels,
    )
    assert estimate.estimated_peak_bytes > estimate.pixel_count * 64


def test_the_same_package_carries_a_developed_rubbing(corded: ArtifactSession) -> None:
    computation = _rubbing(corded)
    session = commit_developed_rubbing(
        corded,
        computation,
        record_id="record:developed:package",
        created_at="2026-09-03T00:00:11Z",
        operator="tester",
    )
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "strip.amr-rubbing"
        try:
            published = export_rubbing_package(
                destination,
                session.document,
                "record:developed:package",
                computation.raster,
            )
        except ArtifactRubbingExportError as exc:
            # A host that cannot fsync a directory still publishes the package
            # atomically and says so by raising with committed set; that is the
            # writer's contract, not a failure to write.
            if not exc.committed:
                raise
            published = destination
        assert published == destination
        assert destination.is_dir()
        bundle = validate_rubbing_export_package(destination, document=session.document)
        relocated = validate_rubbing_export_package(destination)
        sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))

    assert bundle.raster_sha256 == relocated.raster_sha256 == computation.raster.raster_sha256
    assert (bundle.width_pixels, bundle.height_pixels) == (
        computation.raster.width_pixels,
        computation.raster.height_pixels,
    )
    assert sidecar["schema_version"] == "1.2.0"
    assert sidecar["recipe"]["kind"] == "developed_rubbing"
    assert sidecar["raster_receipt"]["coordinate_space"] == "canonical_mm_developed_raster/v1"
    assert sidecar["provenance"]["record"]["type"] == DEVELOPED_RUBBING_RECORD_TYPE
    assert sidecar["presentation"]["physical_scale"] == "1:1_planar_sampling"
    assert sidecar["qc"]["raster"]["development_sha256"] == (
        computation.raster.development_sha256
    )
    # The development travels in the dependency closure, so the reader knows
    # which strip the paper was on.
    closure_ids = {entry["id"] for entry in sidecar["provenance"]["dependency_closure"]}
    assert UNWRAP_ID in closure_ids

    # A six-view raster cannot be passed off as this record, and a stale
    # development blocks the export the way it blocks the drawing.
    with pytest.raises(ArtifactRubbingExportError, match="coordinate space"):
        build_rubbing_export(
            session.document,
            "record:developed:package",
            DigitalRubbingRaster(
                pixels=computation.raster.pixels,
                frame=outline_frame("top"),
                view="top",
                pixels_per_meter=computation.raster.pixels_per_meter,
                minimum_u_pixel_index=0,
                minimum_v_pixel_index=0,
            ),
        )
    dragged = session.activate_parent_align()
    with pytest.raises(ArtifactRubbingExportError, match="FRESH"):
        build_rubbing_export(
            dragged.document, "record:developed:package", computation.raster
        )


def test_the_export_controller_reproduces_the_developed_raster(
    corded: ArtifactSession, tmp_path: Path
) -> None:
    computation = _rubbing(corded)
    session = commit_developed_rubbing(
        corded,
        computation,
        record_id="record:developed:export",
        created_at="2026-09-03T00:00:11Z",
        operator="tester",
    )
    controller = ArtifactExportController(ArtifactWorkbench(session=session))
    destination = tmp_path / "strip.amr-rubbing"
    work_item = controller.begin_rubbing(destination, "record:developed:export")
    result = controller.execute(work_item)
    controller.publish_result(work_item, result)

    bundle = validate_rubbing_export_package(destination, document=session.document)
    assert bundle.raster_sha256 == computation.raster.raster_sha256
    assert bundle.pixels_per_meter == 10_000


def _publish(workbench: ArtifactWorkbench):
    def publish(transition) -> None:
        assert isinstance(transition, RecordBindingTransition)
        activation = workbench.activate_record_binding(transition)
        workbench.finalize_record_binding(activation)

    return publish


def test_the_controller_draws_and_publishes_a_developed_rubbing(
    corded: ArtifactSession,
) -> None:
    workbench = ArtifactWorkbench(session=corded)
    controller = ArtifactMeasurementController(workbench)
    work_item = controller.begin_developed_rubbing(
        UNWRAP_ID,
        **OPTIONS,
        record_id="record:developed:controller",
        created_at=STAMP,
        operator="tester",
    )
    assert work_item.kind is MeasurementOperationKind.DEVELOPED_RUBBING
    assert work_item.depends_on_record_ids == (UNWRAP_ID,)
    result = controller.execute(work_item)
    assert controller.rubbing_resource_estimate(work_item) is not None
    publication = controller.publish_result(work_item, result, _publish(workbench))

    published = workbench.snapshot.session
    assert isinstance(published, ArtifactSession)
    record = published.document.record_index[publication.record_id]
    assert record.type == DEVELOPED_RUBBING_RECORD_TYPE
    assert record.depends_on_record_ids == (UNWRAP_ID,)
    validate_known_records(published.document)


def test_the_controller_refuses_a_development_that_is_not_there(
    corded: ArtifactSession,
) -> None:
    controller = ArtifactMeasurementController(ArtifactWorkbench(session=corded))
    with pytest.raises(ArtifactMeasurementError, match="does not exist"):
        controller.begin_developed_rubbing("record:unwrap:nowhere", **OPTIONS)
    with pytest.raises(ArtifactMeasurementError, match="not a tile unwrap"):
        controller.begin_developed_rubbing(RIM_ID, **OPTIONS)


def test_the_strip_goes_onto_a_drawing_sheet_at_the_sheet_scale(
    corded: ArtifactSession,
) -> None:
    """What a rubber does with the paper: paste it beside the drawing."""

    computation = _rubbing(corded)
    session = commit_developed_rubbing(
        corded,
        computation,
        record_id="record:developed:sheet",
        created_at="2026-09-03T00:00:11Z",
        operator="tester",
    )
    options = DrawingSheetOptions(
        title_block=TitleBlock(
            artifact_label="시험 토기 001",
            rows=(("작성", "tester"), ("일자", "2026-09-03")),
        ),
        scale_denominator=2.0,
    )

    bundle = compose_drawing_sheet(
        session.document,
        ["record:developed:sheet"],
        options=options,
        rasters={"record:developed:sheet": computation.raster},
    )
    root = ET.fromstring(bundle.svg_bytes.decode("utf-8"))
    images = list(root.iter("{http://www.w3.org/2000/svg}image"))
    assert len(images) == 1
    raster = computation.raster
    assert float(images[0].attrib["width"]) == pytest.approx(
        raster.width_pixels * 1000.0 / raster.pixels_per_meter / 2.0, abs=1e-6
    )
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    figure = sidecar["figures"][0]
    assert figure["record_type"] == DEVELOPED_RUBBING_RECORD_TYPE
    assert figure["raster_sha256"] == raster.raster_sha256
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
