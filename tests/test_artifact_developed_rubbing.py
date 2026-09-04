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
    ARTBOARD_DEVELOPMENT_BOUNDS,
    ARTBOARD_HEIGHT_PROFILE_BANDS,
    ARTBOARD_LARGEST_COVERED_RECTANGLE,
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
from src.core.artifact_rubbing_extractor import (
    MAX_RUBBING_PAPER_TONE_PERCENT,
    RECESS_TONE_RETAINED_PERCENT,
    RELIEF_MODEL_CONTACT,
    DigitalRubbingRaster,
)
from src.core.artifact_session import ArtifactSession
from src.core.drawing_sheet import (
    DrawingSheetError,
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
from src.core.artifact_surface_strip import (
    select_positioned_surface_strip,
    strip_parameters,
)
from synthetic_vessel import (
    RIM_ID,
    positioned_vessel_session,
)


UNWRAP_ID = "record:unwrap:strip"
STRIP_WIDTH_UM = 20_000
STAMP = "2026-09-03T00:00:10Z"
OPTIONS: dict[str, Any] = {
    "pixels_per_mm": 10,
    # A rubbing cropped to its covered rectangle carries no margin: the crop
    # exists to remove exactly that.
    "margin_um": 0,
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
    session, _vertices, _faces = positioned_vessel_session(
        segments=segments, rings=rings, relief=relief
    )
    selected = select_positioned_surface_strip(
        session,
        strip_parameters(
            reference_angle_microdegrees=90_000_000, width_um=STRIP_WIDTH_UM
        ),
    ).face_indices
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


def test_the_rubbing_is_a_rectangle_of_the_width_that_was_asked_for(
    corded: ArtifactSession,
) -> None:
    """A rubbing is a piece of paper: straight edges, one width, top to bottom.

    The development it is drawn on is not rectangular - its boundary follows
    whole triangles, and the spacing that quantises that boundary is the arc
    r x dtheta, which is wider where the body swells - so an uncropped strip
    stepped in width row by row and read as stacked trapezoids.
    """

    raster = _rubbing(corded).raster
    covered = raster.pixels[:, :, 1] == 255

    assert bool(covered.all())
    widths = {int(row.sum()) for row in covered}
    assert widths == {raster.width_pixels}
    # 10 px/mm on a 20 mm strip: the lattice may add one pixel, never a
    # millimetre.
    assert abs(raster.width_pixels / 10.0 - STRIP_WIDTH_UM / 1000.0) <= 0.2
    # The height is the meridian the paper covers, not the 90 mm axis height.
    assert 94.0 < raster.height_pixels / 10.0 < 95.5


def test_the_record_says_which_development_and_how_it_was_framed(
    corded: ArtifactSession,
) -> None:
    receipt = tile_unwrap_receipt_from_record(corded.document.record_index[UNWRAP_ID])
    computation = _rubbing(corded)
    raster = computation.raster

    assert raster.receipt()["height_mm_exact"] == {
        "denominator": 10_000,
        "numerator": raster.height_pixels * 1000,
    }
    qc = computation.qc_dict()
    assert qc["development_record_id"] == UNWRAP_ID
    assert qc["development_sha256"] == receipt["unwrap_sha256"]
    assert qc["artboard_policy"] == ARTBOARD_LARGEST_COVERED_RECTANGLE
    assert qc["multi_layer_pixel_count"] == 0
    assert qc["projected_zero_area_face_count"] == 0
    assert qc["radius_min_um_rounded"] == 25_000
    assert 47_500 < qc["radius_max_um_rounded"] < 48_500
    # The crop is reported, not silent: the ragged margin it removed is the
    # difference between the artboard and the rectangle.
    assert qc["artboard_width_pixels"] > raster.width_pixels
    assert qc["cropped_left_pixels"] + qc["cropped_right_pixels"] == (
        qc["artboard_width_pixels"] - raster.width_pixels
    )
    assert qc["cropped_top_pixels"] + qc["cropped_bottom_pixels"] == (
        qc["artboard_height_pixels"] - raster.height_pixels
    )
    assert qc["uncropped_covered_pixel_count"] > qc["covered_pixel_count"]


def test_a_sherd_keeps_its_whole_development_when_asked(
    corded: ArtifactSession,
) -> None:
    """Cropping a strip gives back the paper; cropping a sherd would throw
    most of it away, so the full development stays available."""

    computation = _rubbing(
        corded,
        artboard_policy=ARTBOARD_DEVELOPMENT_BOUNDS,
        margin_um=1_000,
    )
    whole = computation.raster
    rectangle = _rubbing(corded).raster

    assert whole.width_pixels > rectangle.width_pixels
    assert not bool((whole.pixels[:, :, 1] == 255).all())
    # The four trim counts are reported whichever policy ran, so a reader can
    # add them up without branching; keeping the whole development trims none.
    qc = computation.qc_dict()
    assert qc["artboard_policy"] == ARTBOARD_DEVELOPMENT_BOUNDS
    for side in ("top", "bottom", "left", "right"):
        assert qc[f"cropped_{side}_pixels"] == 0
    assert qc["uncropped_covered_pixel_count"] == qc["covered_pixel_count"]

    with pytest.raises(ArtifactDevelopedRubbingError, match="cannot also carry a margin"):
        _rubbing(corded, margin_um=1_000)


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
    # The estimate is admission control, so it bounds the artboard before the
    # crop: the crop only ever takes pixels away.
    qc = computation.qc_dict()
    assert (estimate.width_pixels, estimate.height_pixels) == (
        qc["artboard_width_pixels"],
        qc["artboard_height_pixels"],
    )
    assert estimate.width_pixels >= computation.raster.width_pixels
    assert estimate.height_pixels >= computation.raster.height_pixels
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
    assert sidecar["schema_version"] == "1.3.0"
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


def test_the_dabber_leaves_ink_on_the_whole_sheet(corded: ArtifactSession) -> None:
    """A rubbing is tapped, not traced.

    An oiled cotton dabber loaded onto paper laid over the surface leaves ink
    everywhere it touches.  The parts that stand out take it densely, but the
    parts that sit back take a little too, so the sheet is a light wash with
    the pattern dark on it - not black marks on white paper.
    """

    dry = _rubbing(corded, relief_polarity="raised").raster
    washed = _rubbing(
        corded, relief_polarity="raised", paper_tone_percent=20
    ).raster

    dry_grey = dry.pixels[:, :, 0]
    washed_grey = washed.pixels[:, :, 0]
    # The dry rubbing leaves plain wall as bare paper; the dabbed one does not.
    assert int(dry_grey.max()) == 255
    assert int(washed_grey.max()) < 255
    # The lightest place on the sheet is the bottom of the deepest recess, and
    # even there the wash keeps its retained share.
    wash = (255 * 20 + 50) // 100
    assert int(washed_grey.max()) == 255 - wash + (
        wash * (100 - RECESS_TONE_RETAINED_PERCENT) + 50
    ) // 100
    assert int(washed_grey.min()) == 0
    assert int(np.median(washed_grey)) < int(np.median(dry_grey))
    # The wash only ever adds ink.  Nowhere on the sheet does dabbing leave a
    # place lighter than the same place without it, so the relief the dry
    # rubbing found is still there, on a ground instead of on bare paper.
    assert bool((washed_grey <= dry_grey).all())
    assert bool((dry_grey[dry_grey == 0] == washed_grey[dry_grey == 0]).all())


def test_a_recess_comes_out_lighter_than_the_sheet_but_not_white(
    corded: ArtifactSession,
) -> None:
    """The incised line across the belly is the test: the dabber reaches into
    it less than onto the flat, so it reads as a lighter line on the wash, and
    never as bare paper."""

    raster = _rubbing(
        corded, relief_polarity="raised", paper_tone_percent=20
    ).raster
    grey = raster.pixels[:, :, 0].astype(np.int64)
    interior = grey[80 : raster.height_pixels - 80, :]
    lightest = int(interior.max())

    # Nothing reaches bare paper, and the lightest place is lighter than the
    # tone a flat surface takes - that lightest place is the groove.
    assert lightest < 255
    flat_tone = (255 * 20 + 50) // 100
    assert lightest > 255 - flat_tone
    # It never gives up more than the retained share of the wash.
    floor = 255 - flat_tone
    assert lightest <= 255 - (flat_tone * RECESS_TONE_RETAINED_PERCENT) // 100 + 1
    assert floor <= 255


def test_a_recipe_written_before_the_wash_reproduces_byte_for_byte(
    corded: ArtifactSession,
) -> None:
    """The wash keys are absent when it is off, so an older recipe rebuilds to
    the same bytes and recomputes to the same raster."""

    plain = _rubbing(corded, relief_polarity="raised")
    recipe = plain.recipe_dict()
    relief = recipe["relief_policy"]

    assert "paper_tone_percent" not in relief
    assert "paper_tone_level" not in relief
    assert "recess_tone_retained_percent" not in relief
    again = compute_developed_rubbing_from_recipe(corded, recipe)
    assert again.raster.raster_sha256 == plain.raster.raster_sha256

    washed = _rubbing(corded, relief_polarity="raised", paper_tone_percent=20)
    washed_relief = washed.recipe_dict()["relief_policy"]
    assert washed_relief["paper_tone_percent"] == 20
    assert washed_relief["paper_tone_level"] == 51
    assert washed_relief["recess_tone_retained_percent"] == (
        RECESS_TONE_RETAINED_PERCENT
    )
    assert washed.raster.raster_sha256 != plain.raster.raster_sha256
    replayed = compute_developed_rubbing_from_recipe(corded, washed.recipe_dict())
    assert replayed.raster.raster_sha256 == washed.raster.raster_sha256


def test_the_wash_is_refused_outside_its_range(corded: ArtifactSession) -> None:
    for value in (-1, MAX_RUBBING_PAPER_TONE_PERCENT + 1):
        with pytest.raises(
            ArtifactDevelopedRubbingError, match="paper_tone_percent"
        ):
            _rubbing(corded, paper_tone_percent=value)


def test_the_record_says_what_heights_its_artboard_was_taken_from(
    corded: ArtifactSession,
) -> None:
    """A strip is pasted back at the height it came from, so the record has
    to say what that height is, row by row up the artboard."""

    qc = _rubbing(corded).qc_dict()
    profile = qc["artboard_height_profile_um"]

    assert qc["artboard_base_height_um"] == profile[0]
    assert qc["artboard_top_height_um"] == profile[-1]
    assert profile == sorted(profile)
    assert len(profile) == ARTBOARD_HEIGHT_PROFILE_BANDS + 1
    # Positioning puts the origin on the measured floor, ten millimetres up
    # the outer wall, so the strip's bottom row sits just above -10 mm and its
    # top row just below the 80 mm rim; the paper between them is longer than
    # that span because the wall bulges.
    assert -10_500 < profile[0] < -9_000
    assert 79_000 < profile[-1] < 80_000
    raster = _rubbing(corded).raster
    paper_um = raster.height_pixels * 1_000_000 // raster.pixels_per_meter
    assert paper_um > profile[-1] - profile[0]


def _axis_sheet_session(corded: ArtifactSession) -> tuple[ArtifactSession, Any]:
    from src.core.artifact_outline_extractor import compute_artifact_outline
    from src.core.artifact_vector_extractor import (
        commit_vector_computation,
        compute_artifact_cutline,
    )
    from src.core.artifact_vector_record import PlanarFrame

    session = corded
    outline = compute_artifact_outline(session, "front", precision_grid_mm=0.02)
    session = commit_vector_computation(
        session, outline, record_id="record:elevation:front",
        created_at=STAMP, operator="tester",
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
        session, cutline, record_id="record:section:front",
        created_at=STAMP, operator="tester",
    )
    computation = _rubbing(session, relief_polarity="raised", paper_tone_percent=20)
    session = commit_developed_rubbing(
        session, computation, record_id="record:rubbing:axis",
        created_at=STAMP, operator="tester",
    )
    return session, computation


@pytest.fixture(scope="module")
def axis_sheet(corded: ArtifactSession) -> tuple[ArtifactSession, Any]:
    return _axis_sheet_session(corded)


def _axis_options(**overrides: Any) -> DrawingSheetOptions:
    base: dict[str, Any] = {
        "title_block": TitleBlock(artifact_label="시험 토기", rows=()),
        "mirror_sections": (("record:elevation:front", "record:section:front"),),
        "rubbings_on_axis": (("record:rubbing:axis", "record:elevation:front"),),
    }
    base.update(overrides)
    return DrawingSheetOptions(**base)


def test_the_strip_is_pasted_flush_against_the_centre_line(
    axis_sheet: tuple[ArtifactSession, Any],
) -> None:
    """The pottery convention: one edge of the rectangle exactly on the axis,
    on the elevation side, each band at the height it was taken from."""

    session, computation = axis_sheet
    bundle = compose_drawing_sheet(
        session.document,
        ["record:elevation:front"],
        options=_axis_options(),
        rasters={"record:rubbing:axis": computation.raster},
    )
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    svg = bundle.svg_bytes.decode("utf-8")
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))

    # One figure carries everything: the elevation, the section, the paper.
    assert len(sidecar["figures"]) == 1
    pasted = sidecar["figures"][0]["rubbing_on_axis"]
    assert pasted["record_id"] == "record:rubbing:axis"
    assert pasted["raster_sha256"] == computation.raster.raster_sha256
    assert pasted["fit"] == "paper"
    assert pasted["side"] == "elevation"
    left, bottom, right, top = pasted["rectangle_mm"]
    # The axis of the front frame is u = 0; the paper ends on it and extends
    # towards the elevation side, at its own width and its own length.
    assert right == 0.0
    assert left == pytest.approx(-computation.raster.width_pixels / 10.0)
    qc = computation.qc_dict()
    assert bottom == pytest.approx(qc["artboard_base_height_um"] / 1000.0)
    assert top - bottom == pytest.approx(computation.raster.height_pixels / 10.0)
    assert top > qc["artboard_top_height_um"] / 1000.0
    # Pixels are embedded once and pasted whole.
    assert svg.count("data:image/png;base64,") == 1
    assert svg.count('<use xlink:href="#rubbing-on-axis-0000-pixels"/>') == 1
    # The centre line is drawn back over the paper.
    assert svg.index("rubbing-on-axis-0000-band-00") < svg.index(
        'id="layer-center-axis"'
    )
    assert sidecar["rubbings_on_axis"] == [
        {
            "figure_record_id": "record:elevation:front",
            "rubbing_record_id": "record:rubbing:axis",
        }
    ]
    # The pasted paper says what it is, in the title block and under itself.
    assert {"label": "탁본", "value": "3D 메쉬에서 계산 · 종이 탁본 아님"} in sidecar[
        "title_block"
    ]
    caption = sidecar["figures"][0]["caption"]
    assert caption.startswith("전산 탁본 · 높이 모델 · 창 ")
    assert caption.endswith(" · 먹 100% · 기저 20%")
    assert 'id="rubbing-caption-0000"' in svg
    assert caption in svg


def test_the_height_fit_pastes_each_band_where_it_was_taken(
    axis_sheet: tuple[ArtifactSession, Any],
) -> None:
    """The alternative for a reader who wants the rubbing's grooves level with
    the elevation's lines: bands, each at its own height, off by default."""

    session, computation = axis_sheet
    bundle = compose_drawing_sheet(
        session.document,
        ["record:elevation:front"],
        options=_axis_options(rubbing_on_axis_fit="axis_height"),
        rasters={"record:rubbing:axis": computation.raster},
    )
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    pasted = json.loads(bundle.sidecar_bytes.decode("utf-8"))["figures"][0][
        "rubbing_on_axis"
    ]
    left, bottom, right, top = pasted["rectangle_mm"]
    qc = computation.qc_dict()
    assert pasted["fit"] == "axis_height"
    assert len(pasted["band_heights_mm"]) == ARTBOARD_HEIGHT_PROFILE_BANDS + 1
    assert top == pytest.approx(qc["artboard_top_height_um"] / 1000.0)
    assert top - bottom < computation.raster.height_pixels / 10.0
    assert bundle.svg_bytes.decode("utf-8").count("<use xlink:href=") == (
        ARTBOARD_HEIGHT_PROFILE_BANDS
    )


def test_a_pasted_rubbing_is_not_also_a_figure(
    axis_sheet: tuple[ArtifactSession, Any],
) -> None:
    session, computation = axis_sheet
    with pytest.raises(DrawingSheetError, match="must not also be a figure"):
        compose_drawing_sheet(
            session.document,
            ["record:elevation:front", "record:rubbing:axis"],
            options=_axis_options(),
            rasters={"record:rubbing:axis": computation.raster},
        )
    with pytest.raises(DrawingSheetError, match="recomputed raster must be given"):
        compose_drawing_sheet(
            session.document,
            ["record:elevation:front"],
            options=_axis_options(),
        )
    with pytest.raises(DrawingSheetError, match="rubbing_on_axis_fit"):
        _axis_options(rubbing_on_axis_fit="stretch")


def test_a_rubbing_computed_before_heights_were_recorded_is_refused(
    axis_sheet: tuple[ArtifactSession, Any],
) -> None:
    """An older record has no way to say where it goes; the sheet says so
    rather than guessing a height."""

    from dataclasses import replace

    session, computation = axis_sheet
    record = session.document.record_index["record:rubbing:axis"]
    stale_qc = {
        key: value
        for key, value in record.qc.items()
        if not key.startswith("artboard_") or key.endswith("_pixels")
    }
    older = replace(record, qc=stale_qc)
    document = replace(
        session.document,
        records=tuple(
            older if item.id == record.id else item for item in session.document.records
        ),
    )
    with pytest.raises(DrawingSheetError, match="compute the rubbing again"):
        compose_drawing_sheet(
            document,
            ["record:elevation:front"],
            options=_axis_options(),
            rasters={"record:rubbing:axis": computation.raster},
        )


CONTACT: dict[str, Any] = {
    "relief_model": RELIEF_MODEL_CONTACT,
    "reference_radius_um": 700,
    "black_point_um": 120,
    "contact_ink_percent": 70,
    "relief_polarity": "raised",
    "paper_tone_percent": 0,
}


def test_the_contact_model_inks_a_plain_wall_evenly(plain: ArtifactSession) -> None:
    """Paper pressed onto a plain wall lies on all of it, so all of it takes
    the contact tone - no wash, no shading from the wall's own curvature."""

    raster = _rubbing(plain, **CONTACT).raster
    interior = raster.pixels[80:-80, 40:-40, 0].astype(np.int64)
    contact_level = (255 * 70 + 50) // 100
    # The darkest tone anywhere is the contact tone, and everything is within
    # a facet of it.  This synthetic wall is a 96-gon, so between two vertex
    # rings the paper bridges a chord 21 um below the true circle, which the
    # contact model reads faithfully as a hair less ink; a real scan is not
    # faceted.  What must not appear is a gradient: no tone farther from the
    # contact tone than that sag is worth.
    assert int(interior.min()) == 255 - contact_level
    assert int(interior.max()) - int(interior.min()) <= 20
    assert int(np.percentile(interior, 99)) - int(np.percentile(interior, 1)) <= 20


def test_the_contact_model_leaves_an_incised_line_white(corded: ArtifactSession) -> None:
    """The paper bridges a groove, so the groove's floor is where the ink is
    not - a light line on a dark ground, which is what a rubbing of a groove
    looks like, and the reverse of what the local-mean shading gave it."""

    contact = _rubbing(corded, **CONTACT).raster
    shaded = _rubbing(corded, relief_polarity="raised").raster
    grey_contact = contact.pixels[:, :, 0].astype(np.int64)
    grey_shaded = shaded.pixels[:, :, 0].astype(np.int64)
    # The groove is cut at z = 45 mm on the builder's scale, 35 mm canonical;
    # find the raster rows lightest under the contact model in the middle
    # third of the strip, and check they are the same rows the shading model
    # darkened around.
    middle = slice(contact.width_pixels // 3, 2 * contact.width_pixels // 3)
    contact_rows = grey_contact[:, middle].mean(axis=1)
    shaded_rows = grey_shaded[:, middle].mean(axis=1)
    lightest = int(np.argmax(contact_rows[80:-80])) + 80
    wall = int(np.median(contact_rows))
    assert contact_rows[lightest] > wall + 60
    # The shading model reads the wall beside the groove as raised: dark.
    assert shaded_rows[lightest - 12 : lightest + 12].min() < np.median(shaded_rows) - 20


def test_the_contact_model_inks_a_cord_on_its_ridge_only(corded: ArtifactSession) -> None:
    raster = _rubbing(corded, **CONTACT).raster
    grey = raster.pixels[:, :, 0].astype(np.int64)
    band = grey[100:300, 40:-40]
    contact_level = (255 * 70 + 50) // 100
    # Ridges lie on the paper and take the contact tone (a facet's sag over
    # it); the valleys between them do not reach the paper and go to white.
    assert int(band.min()) == 255 - contact_level
    assert int(np.percentile(band, 10)) <= 255 - contact_level + 20
    assert int(np.percentile(band, 90)) >= 200
    assert int(band.max()) == 255


def test_the_contact_model_is_its_own_recipe_and_older_recipes_stay_shading(
    corded: ArtifactSession,
) -> None:
    shaded = _rubbing(corded, relief_polarity="raised")
    relief = shaded.recipe_dict()["relief_policy"]
    assert "model" not in relief
    assert "contact_ink_percent" not in relief

    contact = _rubbing(corded, **CONTACT)
    relief = contact.recipe_dict()["relief_policy"]
    assert relief["model"] == RELIEF_MODEL_CONTACT
    assert relief["contact_ink_percent"] == 70
    assert relief["contact_ink_level"] == (255 * 70 + 50) // 100
    assert relief["envelope_filter"] == "masked_square_local_max/v1"
    replayed = compute_developed_rubbing_from_recipe(corded, contact.recipe_dict())
    assert replayed.raster.raster_sha256 == contact.raster.raster_sha256
    assert replayed.raster.raster_sha256 != shaded.raster.raster_sha256

    with pytest.raises(ArtifactDevelopedRubbingError, match="one side"):
        _rubbing(corded, **{**CONTACT, "relief_polarity": "bidirectional"})
