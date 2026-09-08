"""Make a plate again from a project and its specification, without the GUI.

The plate's sidecar carries a `plate_spec` block: everything the composer
was told.  Given the project the records live in and that block (or a
spec file the archaeologist edited), this composes the same plate - or the
plate with one number changed - from the command line.  Line work needs
only the document; a rubbing pasted on the sheet needs its pixels, which
a record does not store, so they are recomputed from the record's recipe
against the embedded source, exactly as the GUI's export does, and refused
if they do not reproduce the record's receipt.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.application.artifact_measurements import DEFAULT_RUBBING_MEMORY_BUDGET_BYTES
from src.core.artifact_developed_rubbing import (
    DEVELOPED_RUBBING_RECORD_TYPE,
    ArtifactDevelopedRubbingError,
    compute_developed_rubbing_from_recipe,
    development_record_for_recipe,
    developed_rubbing_receipt_from_record,
    estimate_developed_rubbing_resources,
    require_current_developed_rubbing_computation,
)
from src.core.artifact_rubbing_extractor import (
    ArtifactRubbingError,
    compute_artifact_rubbing_from_recipe,
    estimate_digital_rubbing_resources,
    require_current_rubbing_computation,
)
from src.core.artifact_rubbing_record import RUBBING_RECORD_TYPE, rubbing_receipt_from_record
from src.core.artifact_session import ArtifactSession, ArtifactSessionError
from src.core.drawing_sheet import DrawingSheetBundle, DrawingSheetError, compose_drawing_sheet
from src.core.drawing_sheet_spec import PlateSpecError, plate_spec_options
from src.core.project_file import (
    EmbeddedSourceRequiredError,
    ProjectFormatError,
    load_artifact_project,
    load_artifact_session_project,
)


class PlateFromSpecError(RuntimeError):
    """The plate could not be made from this project and spec."""


def _within_budget(estimated_peak_bytes: int, record_id: str) -> None:
    """The same memory budget the workbench and the GUI hold themselves to.

    A plate made from the command line reads the same records on the same
    machine; it may not spend what the application refuses to spend.
    """

    if estimated_peak_bytes > DEFAULT_RUBBING_MEMORY_BUDGET_BYTES:
        raise PlateFromSpecError(
            f"recomputing rubbing {record_id!r} is estimated to peak at "
            f"{estimated_peak_bytes} bytes, past the "
            f"{DEFAULT_RUBBING_MEMORY_BUDGET_BYTES}-byte budget"
        )


def _recompute_rubbing(session: ArtifactSession, record: Any) -> Any:
    """The record's pixels again from its recipe, or a refusal."""

    mesh = session.source_mesh
    try:
        if record.type == DEVELOPED_RUBBING_RECORD_TYPE:
            _development, development_receipt = development_record_for_recipe(session.document, record.recipe)
            _within_budget(
                estimate_developed_rubbing_resources(
                    development_receipt,
                    record.recipe,
                    source_vertex_count=int(mesh.vertices.shape[0]),
                    source_face_count=int(mesh.faces.shape[0]),
                    source_geometry_bytes=int(mesh.vertices.nbytes + mesh.faces.nbytes),
                ).estimated_peak_bytes,
                record.id,
            )
            developed = compute_developed_rubbing_from_recipe(session, record.recipe)
            require_current_developed_rubbing_computation(session, developed)
            if developed.raster.receipt() != developed_rubbing_receipt_from_record(record):
                raise PlateFromSpecError(
                    f"recomputed developed rubbing {record.id!r} does not match its record receipt"
                )
            return developed.raster
        if record.type == RUBBING_RECORD_TYPE:
            snapshot = session.projection_snapshot()
            _within_budget(
                estimate_digital_rubbing_resources(
                    mesh.vertices,
                    mesh.faces,
                    record.recipe,
                    source_to_world_mm_matrix4x4=snapshot.matrix4x4,
                    uv_coords=mesh.uv_coords,
                    texture=mesh.texture,
                ).estimated_peak_bytes,
                record.id,
            )
            computation = compute_artifact_rubbing_from_recipe(session, record.recipe)
            require_current_rubbing_computation(session, computation)
            if computation.raster.receipt() != rubbing_receipt_from_record(record):
                raise PlateFromSpecError(
                    f"recomputed Digital Rubbing {record.id!r} does not match its record receipt"
                )
            return computation.raster
    except (ArtifactDevelopedRubbingError, ArtifactRubbingError, ArtifactSessionError) as exc:
        raise PlateFromSpecError(f"rubbing {record.id!r} could not be recomputed: {exc}") from exc
    raise PlateFromSpecError(f"record {record.id!r} is not a rubbing")


def compose_plate_from_project(project_path: str | Path, spec: Mapping[str, Any]) -> DrawingSheetBundle:
    """Compose the plate ``spec`` describes from the records in ``project_path``."""

    try:
        record_ids, options = plate_spec_options(spec)
    except PlateSpecError as exc:
        raise PlateFromSpecError(str(exc)) from exc
    if options.paint_cutouts or options.relief_stipples:
        # Their pixels come from the source's colour map and a shade the GUI
        # or a script computed; neither is in the project, and pasting a
        # raster that cannot be re-proved is not something this path does.
        names = sorted({entry[0] for entry in options.paint_cutouts} | {entry[0] for entry in options.relief_stipples})
        raise PlateFromSpecError(
            "this plate pastes pixels the command line cannot recompute from the project "
            f"({', '.join(names)}); make it where those records were computed"
        )
    try:
        document = load_artifact_project(project_path)
    except ProjectFormatError as exc:
        raise PlateFromSpecError(f"project could not be opened: {exc}") from exc
    rubbing_ids = [
        record_id
        for record_id in [*record_ids, *(rubbing_id for rubbing_id, _elevation in options.rubbings_on_axis)]
        if getattr(document.record_index.get(record_id), "type", None) in {RUBBING_RECORD_TYPE, DEVELOPED_RUBBING_RECORD_TYPE}
    ]
    rasters: dict[str, Any] = {}
    if rubbing_ids:
        try:
            session = load_artifact_session_project(project_path)
        except EmbeddedSourceRequiredError as exc:
            raise PlateFromSpecError(
                "the plate pastes a rubbing, whose pixels are recomputed from the source; "
                f"this project does not embed its source: {exc}"
            ) from exc
        except ProjectFormatError as exc:
            raise PlateFromSpecError(f"project source could not be materialised: {exc}") from exc
        for record_id in rubbing_ids:
            rasters[record_id] = _recompute_rubbing(session, session.document.record_index[record_id])
        document = session.document
    try:
        return compose_drawing_sheet(document, record_ids, options=options, rasters=rasters)
    except DrawingSheetError as exc:
        raise PlateFromSpecError(str(exc)) from exc


def write_plate_files(svg_bytes: bytes, sidecar_bytes: bytes, svg_path: str | Path) -> tuple[Path, Path]:
    """Write a plate's two files so that a failure leaves neither half-written.

    The drawing and its sidecar are one statement - the sidecar carries the
    digest of those exact SVG bytes and the manifest hash of the document
    they came from - so a plate whose sidecar is the previous run's is a
    drawing that lies about where it came from.  Both are written to hidden
    temporary files beside their destinations, flushed to the platter, and
    only then renamed; if the second write fails the first is not renamed,
    and whatever was there before is untouched.
    """

    out_svg = Path(svg_path)
    if out_svg.suffix.lower() != ".svg":
        raise PlateFromSpecError("a plate is written to an .svg path")
    sidecar = out_svg.with_suffix(".provenance.json")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, payload in ((out_svg, svg_bytes), (sidecar, sidecar_bytes)):
            handle, raw = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
            )
            temporary = Path(raw)
            staged.append((temporary, destination))
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, destination in staged:
            os.replace(temporary, destination)
    except OSError:
        for temporary, _destination in staged:
            temporary.unlink(missing_ok=True)
        raise
    return out_svg, sidecar


def write_plate(bundle: DrawingSheetBundle, svg_path: str | Path) -> tuple[Path, Path]:
    """Write the SVG and, beside it, its ``.provenance.json`` sidecar."""

    return write_plate_files(bundle.svg_bytes, bundle.sidecar_bytes, svg_path)


__all__ = [
    "PlateFromSpecError",
    "compose_plate_from_project",
    "write_plate",
    "write_plate_files",
]
