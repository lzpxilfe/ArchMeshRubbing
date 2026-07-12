"""Self-contained deterministic 1:1 PNG packages for Digital Rubbing records."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid

from .artifact_document import (
    ArtifactDocument,
    ArtifactDocumentError,
    DerivedRecord,
    RecordFreshness,
    RecordLifecycleStatus,
    canonical_recipe_hash,
)
from .artifact_rubbing_extractor import (
    ArtifactRubbingError,
    DigitalRubbingRaster,
    validate_rubbing_recipe,
)
from .artifact_rubbing_record import (
    ArtifactRubbingRecordError,
    RUBBING_RECORD_TYPE,
    rubbing_receipt_from_record,
    validate_rubbing_receipt,
)
from .artifact_vector_export import (
    ArtifactVectorExportError,
    build_public_export_provenance,
    fsync_export_directory,
    publish_export_directory_noreplace,
    read_bounded_export_file,
    validate_public_export_provenance,
    write_new_export_file,
)
from .canonical_json import (
    CanonicalJSONError,
    canonical_json_bytes,
    canonical_json_sha256,
)
from .canonical_png import (
    CanonicalPNGError,
    MAX_CANONICAL_PNG_BYTES,
    decode_canonical_ga8_png,
    encode_canonical_ga8_png,
)


RUBBING_EXPORT_FORMAT = "archmeshrubbing_rubbing_export"
RUBBING_EXPORT_SCHEMA_VERSION = "1.0.0"
RUBBING_EXPORT_DIRECTORY_SUFFIX = ".amr-rubbing"
RUBBING_EXPORT_PNG_NAME = "artifact.png"
RUBBING_EXPORT_SIDECAR_NAME = "artifact.amr-rubbing.json"
RUBBING_EXPORT_PNG_MEDIA_TYPE = "image/png"
RUBBING_EXPORT_SIDECAR_MEDIA_TYPE = (
    "application/vnd.archmeshrubbing.rubbing-export+json"
)
RUBBING_PNG_METADATA_FORMAT = "archmeshrubbing_rubbing_png_metadata"
RUBBING_PNG_METADATA_SCHEMA_VERSION = "1.0.0"
MAX_RUBBING_EXPORT_SIDECAR_BYTES = 16 * 1024 * 1024


class ArtifactRubbingExportError(ValueError):
    """A Digital Rubbing package violates scale, integrity, or provenance."""


@dataclass(frozen=True, slots=True)
class RubbingExportBundle:
    png_bytes: bytes
    sidecar_bytes: bytes
    png_sha256: str
    sidecar_sha256: str
    raster_sha256: str
    raw_pixel_sha256: str
    width_pixels: int
    height_pixels: int
    pixels_per_meter: int


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _exact_keys(value: object, expected: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactRubbingExportError(f"{name} must be an object")
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise ArtifactRubbingExportError(
            f"{name} is missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise ArtifactRubbingExportError(
            f"{name} has unknown fields: {', '.join(unknown)}"
        )
    return value


def _strict_sidecar_bytes(sidecar_bytes: bytes) -> dict[str, Any]:
    if (
        not isinstance(sidecar_bytes, bytes)
        or not sidecar_bytes
        or len(sidecar_bytes) > MAX_RUBBING_EXPORT_SIDECAR_BYTES
    ):
        raise ArtifactRubbingExportError("rubbing sidecar size is outside the limit")

    def reject_constant(value: str) -> None:
        raise ArtifactRubbingExportError(f"sidecar contains invalid constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactRubbingExportError(
                    f"sidecar contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        decoded = json.loads(
            sidecar_bytes.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactRubbingExportError("sidecar is not strict UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ArtifactRubbingExportError("sidecar root must be an object")
    try:
        expected = canonical_json_bytes(decoded)
    except CanonicalJSONError as exc:
        raise ArtifactRubbingExportError(str(exc)) from exc
    if expected != sidecar_bytes:
        raise ArtifactRubbingExportError("sidecar is not RFC 8785 canonical JSON")
    return decoded


def _require_exportable_record(
    document: ArtifactDocument,
    record_id: str,
) -> tuple[DerivedRecord, dict[str, Any], dict[str, Any]]:
    if not isinstance(document, ArtifactDocument):
        raise ArtifactRubbingExportError("document must be an ArtifactDocument")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ArtifactRubbingExportError("record_id must be a non-empty string")
    record = document.record_index.get(record_id)
    if record is None or record.type != RUBBING_RECORD_TYPE:
        raise ArtifactRubbingExportError("Digital Rubbing record does not exist")
    if record.lifecycle_status is not RecordLifecycleStatus.READY:
        raise ArtifactRubbingExportError("only READY Digital Rubbing records may export")
    try:
        freshness = document.record_freshness(record.id)
    except ArtifactDocumentError as exc:
        raise ArtifactRubbingExportError(str(exc)) from exc
    if freshness is not RecordFreshness.FRESH:
        raise ArtifactRubbingExportError(
            f"only FRESH Digital Rubbing records may export (got {freshness.value})"
        )
    try:
        receipt = rubbing_receipt_from_record(record)
    except ArtifactRubbingRecordError as exc:
        raise ArtifactRubbingExportError(str(exc)) from exc
    record_qc = record.to_dict()["qc"]
    if not isinstance(record_qc, dict):
        raise ArtifactRubbingExportError("record QC must be an object")
    return record, receipt, record_qc


def _require_matching_raster(
    raster: DigitalRubbingRaster,
    receipt: Mapping[str, Any],
) -> None:
    if not isinstance(raster, DigitalRubbingRaster):
        raise ArtifactRubbingExportError("raster must be a DigitalRubbingRaster")
    if raster.receipt() != dict(receipt):
        raise ArtifactRubbingExportError(
            "recomputed raster does not match the Digital Rubbing record receipt"
        )


def _public_provenance(
    document: ArtifactDocument,
    record: DerivedRecord,
) -> dict[str, Any]:
    try:
        return build_public_export_provenance(document, record)
    except ArtifactVectorExportError as exc:
        raise ArtifactRubbingExportError(str(exc)) from exc


def _validated_public_provenance(value: object) -> Mapping[str, Any]:
    try:
        return validate_public_export_provenance(value)
    except ArtifactVectorExportError as exc:
        raise ArtifactRubbingExportError(str(exc)) from exc


def _presentation(receipt: Mapping[str, Any]) -> dict[str, Any]:
    width = receipt["width_pixels"]
    height = receipt["height_pixels"]
    minimum_u = receipt["minimum_u_pixel_index"]
    minimum_v = receipt["minimum_v_pixel_index"]
    assert isinstance(width, int)
    assert isinstance(height, int)
    assert isinstance(minimum_u, int)
    assert isinstance(minimum_v, int)
    return {
        "artboard_pixel_bounds_uv": [
            minimum_u,
            minimum_v,
            minimum_u + width,
            minimum_v + height,
        ],
        "height_mm_exact": receipt["height_mm_exact"],
        "height_pixels": height,
        "physical_scale": "1:1_planar_sampling",
        "pixel_pitch_mm_exact": {
            "denominator": receipt["pixels_per_meter"],
            "numerator": 1000,
        },
        "pixels_per_meter": receipt["pixels_per_meter"],
        "unit": "mm",
        "width_mm_exact": receipt["width_mm_exact"],
        "width_pixels": width,
    }


def _privacy() -> dict[str, Any]:
    return {
        "absolute_source_path_embedded": False,
        "annotations_embedded_in_primary_png": False,
        "external_resources": False,
        "review_labels_embedded_in_primary_png": False,
    }


def _claims_sha256(sidecar: Mapping[str, Any]) -> str:
    claims = {
        key: sidecar[key]
        for key in (
            "format",
            "presentation",
            "privacy",
            "provenance",
            "qc",
            "raster_receipt",
            "recipe",
            "schema_version",
        )
    }
    try:
        return canonical_json_sha256(claims)
    except CanonicalJSONError as exc:
        raise ArtifactRubbingExportError(str(exc)) from exc


def _png_metadata(
    *,
    provenance: Mapping[str, Any],
    receipt: Mapping[str, Any],
    claims_sha256: str,
) -> dict[str, Any]:
    document = provenance["document"]
    record = provenance["record"]
    assert isinstance(document, Mapping)
    assert isinstance(record, Mapping)
    return {
        "document_id": document["document_id"],
        "document_manifest_sha256": document["manifest_sha256"],
        "format": RUBBING_PNG_METADATA_FORMAT,
        "geometry_ref": record["geometry_ref"],
        "height_pixels": receipt["height_pixels"],
        "physical_scale": "1:1_planar_sampling",
        "pixels_per_meter": receipt["pixels_per_meter"],
        "raster_sha256": receipt["raster_sha256"],
        "raw_pixel_sha256": receipt["raw_pixel_sha256"],
        "recipe_hash": record["recipe_hash"],
        "record_id": record["id"],
        "record_type": record["type"],
        "schema_version": RUBBING_PNG_METADATA_SCHEMA_VERSION,
        "sidecar": RUBBING_EXPORT_SIDECAR_NAME,
        "sidecar_claims_sha256": claims_sha256,
        "width_pixels": receipt["width_pixels"],
    }


def build_rubbing_export(
    document: ArtifactDocument,
    record_id: str,
    raster: DigitalRubbingRaster,
) -> RubbingExportBundle:
    record, receipt, record_qc = _require_exportable_record(document, record_id)
    _require_matching_raster(raster, receipt)
    provenance = _public_provenance(document, record)
    sidecar: dict[str, Any] = {
        "format": RUBBING_EXPORT_FORMAT,
        "presentation": _presentation(receipt),
        "privacy": _privacy(),
        "provenance": provenance,
        "qc": {
            "export_gate": {
                "raster_recomputed_and_verified": True,
                "record_freshness": RecordFreshness.FRESH.value,
                "record_lifecycle_status": RecordLifecycleStatus.READY.value,
            },
            "raster": raster.qc_summary(),
            "record": record_qc,
            "scale": {
                "physical_scale": "1:1_planar_sampling",
                "png_phys_matches_sidecar": True,
                "unit": "mm",
            },
        },
        "raster_receipt": receipt,
        "recipe": record.to_dict()["recipe"],
        "schema_version": RUBBING_EXPORT_SCHEMA_VERSION,
    }
    claims_sha = _claims_sha256(sidecar)
    metadata = _png_metadata(
        provenance=provenance,
        receipt=receipt,
        claims_sha256=claims_sha,
    )
    try:
        png_bytes = encode_canonical_ga8_png(
            raster.pixels,
            pixels_per_meter=raster.pixels_per_meter,
            metadata=metadata,
        )
    except CanonicalPNGError as exc:
        raise ArtifactRubbingExportError(str(exc)) from exc
    png_sha = _sha256_bytes(png_bytes)
    sidecar["artifact"] = {
        "file": RUBBING_EXPORT_PNG_NAME,
        "media_type": RUBBING_EXPORT_PNG_MEDIA_TYPE,
        "sha256": png_sha,
        "size_bytes": len(png_bytes),
    }
    try:
        sidecar_bytes = canonical_json_bytes(sidecar)
    except CanonicalJSONError as exc:
        raise ArtifactRubbingExportError(str(exc)) from exc
    if len(sidecar_bytes) > MAX_RUBBING_EXPORT_SIDECAR_BYTES:
        raise ArtifactRubbingExportError("rubbing export sidecar exceeds its limit")
    return validate_rubbing_export_bytes(
        png_bytes,
        sidecar_bytes,
        document=document,
    )


def _validate_qc(
    value: object,
    *,
    raster: DigitalRubbingRaster,
) -> Mapping[str, Any]:
    qc = _exact_keys(
        value,
        {"export_gate", "raster", "record", "scale"},
        name="qc",
    )
    if qc["export_gate"] != {
        "raster_recomputed_and_verified": True,
        "record_freshness": "fresh",
        "record_lifecycle_status": "ready",
    }:
        raise ArtifactRubbingExportError("rubbing export gate is invalid")
    if qc["raster"] != raster.qc_summary():
        raise ArtifactRubbingExportError("sidecar raster QC does not match PNG pixels")
    record_qc = qc["record"]
    if not isinstance(record_qc, Mapping):
        raise ArtifactRubbingExportError("sidecar record QC must be an object")
    for key, expected in raster.qc_summary().items():
        if record_qc.get(key) != expected:
            raise ArtifactRubbingExportError(
                f"sidecar record QC field {key!r} does not match PNG pixels"
            )
    if qc["scale"] != {
        "physical_scale": "1:1_planar_sampling",
        "png_phys_matches_sidecar": True,
        "unit": "mm",
    }:
        raise ArtifactRubbingExportError("rubbing scale QC is invalid")
    return qc


def validate_rubbing_export_bytes(
    png_bytes: bytes,
    sidecar_bytes: bytes,
    *,
    document: ArtifactDocument | None = None,
) -> RubbingExportBundle:
    sidecar = _strict_sidecar_bytes(sidecar_bytes)
    root = _exact_keys(
        sidecar,
        {
            "artifact",
            "format",
            "presentation",
            "privacy",
            "provenance",
            "qc",
            "raster_receipt",
            "recipe",
            "schema_version",
        },
        name="rubbing export sidecar",
    )
    if root["format"] != RUBBING_EXPORT_FORMAT:
        raise ArtifactRubbingExportError("rubbing export format is invalid")
    if root["schema_version"] != RUBBING_EXPORT_SCHEMA_VERSION:
        raise ArtifactRubbingExportError("rubbing export schema is invalid")
    artifact = _exact_keys(
        root["artifact"],
        {"file", "media_type", "sha256", "size_bytes"},
        name="artifact",
    )
    if (
        artifact["file"] != RUBBING_EXPORT_PNG_NAME
        or artifact["media_type"] != RUBBING_EXPORT_PNG_MEDIA_TYPE
        or artifact["size_bytes"] != len(png_bytes)
        or artifact["sha256"] != _sha256_bytes(png_bytes)
    ):
        raise ArtifactRubbingExportError("PNG artifact descriptor does not match bytes")
    try:
        receipt = validate_rubbing_receipt(root["raster_receipt"])
        recipe = validate_rubbing_recipe(root["recipe"])
        pixels, ppm, metadata = decode_canonical_ga8_png(png_bytes)
        raster = DigitalRubbingRaster(
            pixels=pixels,
            frame=outline_frame_from_receipt(receipt),
            view=receipt["view"],
            pixels_per_meter=ppm,
            minimum_u_pixel_index=int(receipt["minimum_u_pixel_index"]),
            minimum_v_pixel_index=int(receipt["minimum_v_pixel_index"]),
        )
    except (
        ArtifactRubbingError,
        ArtifactRubbingRecordError,
        CanonicalPNGError,
    ) as exc:
        raise ArtifactRubbingExportError(str(exc)) from exc
    if raster.receipt() != receipt:
        raise ArtifactRubbingExportError("PNG pixels do not match raster receipt")
    if ppm != receipt["pixels_per_meter"]:
        raise ArtifactRubbingExportError("PNG pHYs does not match raster receipt")
    provenance = _validated_public_provenance(root["provenance"])
    record_provenance = provenance["record"]
    assert isinstance(record_provenance, Mapping)
    if (
        record_provenance["type"] != RUBBING_RECORD_TYPE
        or record_provenance["geometry_ref"] != raster.geometry_ref
        or record_provenance["recipe_hash"] != canonical_recipe_hash(recipe)
    ):
        raise ArtifactRubbingExportError(
            "rubbing record provenance does not match raster/recipe"
        )
    presentation = _exact_keys(
        root["presentation"],
        {
            "artboard_pixel_bounds_uv",
            "height_mm_exact",
            "height_pixels",
            "physical_scale",
            "pixel_pitch_mm_exact",
            "pixels_per_meter",
            "unit",
            "width_mm_exact",
            "width_pixels",
        },
        name="presentation",
    )
    if dict(presentation) != _presentation(receipt):
        raise ArtifactRubbingExportError("rubbing presentation does not match receipt")
    if root["privacy"] != _privacy():
        raise ArtifactRubbingExportError("rubbing privacy declaration is invalid")
    _validate_qc(root["qc"], raster=raster)
    claims_sha = _claims_sha256(root)
    expected_metadata = _png_metadata(
        provenance=provenance,
        receipt=receipt,
        claims_sha256=claims_sha,
    )
    if metadata != expected_metadata:
        raise ArtifactRubbingExportError("PNG metadata does not bind the sidecar claims")

    if document is not None:
        record_id = record_provenance["id"]
        if not isinstance(record_id, str):
            raise ArtifactRubbingExportError("provenance record ID is invalid")
        record, document_receipt, record_qc = _require_exportable_record(
            document,
            record_id,
        )
        if document_receipt != receipt:
            raise ArtifactRubbingExportError("export receipt does not match document")
        if provenance != _public_provenance(document, record):
            raise ArtifactRubbingExportError("export provenance does not match document")
        if recipe != record.to_dict()["recipe"]:
            raise ArtifactRubbingExportError("export recipe does not match document")
        qc = root["qc"]
        assert isinstance(qc, Mapping)
        if qc["record"] != record_qc:
            raise ArtifactRubbingExportError("export QC does not match document")

    return RubbingExportBundle(
        png_bytes=png_bytes,
        sidecar_bytes=sidecar_bytes,
        png_sha256=_sha256_bytes(png_bytes),
        sidecar_sha256=_sha256_bytes(sidecar_bytes),
        raster_sha256=raster.raster_sha256,
        raw_pixel_sha256=raster.raw_pixel_sha256,
        width_pixels=raster.width_pixels,
        height_pixels=raster.height_pixels,
        pixels_per_meter=raster.pixels_per_meter,
    )


def outline_frame_from_receipt(receipt: Mapping[str, Any]):
    """Parse a validated receipt frame without trusting caller object identity."""

    from .artifact_vector_record import PlanarFrame  # noqa: PLC0415

    frame = receipt["frame"]
    if not isinstance(frame, Mapping):
        raise ArtifactRubbingExportError("raster receipt frame is invalid")
    return PlanarFrame.from_dict(frame)


def validate_rubbing_export_package(
    directory: str | os.PathLike[str],
    *,
    document: ArtifactDocument | None = None,
) -> RubbingExportBundle:
    path = Path(directory)
    if path.is_symlink() or not path.is_dir():
        raise ArtifactRubbingExportError("rubbing export package must be a real directory")
    entries = sorted(path.iterdir(), key=lambda item: item.name)
    if [entry.name for entry in entries] != sorted(
        [RUBBING_EXPORT_PNG_NAME, RUBBING_EXPORT_SIDECAR_NAME]
    ):
        raise ArtifactRubbingExportError(
            "rubbing export package must contain exactly two files"
        )
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ArtifactRubbingExportError("rubbing export members must be regular files")
    try:
        png_bytes = read_bounded_export_file(
            path / RUBBING_EXPORT_PNG_NAME,
            limit=MAX_CANONICAL_PNG_BYTES,
            label="PNG",
        )
        sidecar_bytes = read_bounded_export_file(
            path / RUBBING_EXPORT_SIDECAR_NAME,
            limit=MAX_RUBBING_EXPORT_SIDECAR_BYTES,
            label="sidecar",
        )
    except ArtifactVectorExportError as exc:
        raise ArtifactRubbingExportError(str(exc)) from exc
    return validate_rubbing_export_bytes(
        png_bytes,
        sidecar_bytes,
        document=document,
    )


def export_rubbing_package(
    directory: str | os.PathLike[str],
    document: ArtifactDocument,
    record_id: str,
    raster: DigitalRubbingRaster,
) -> Path:
    destination = Path(directory).expanduser()
    if not destination.name.endswith(RUBBING_EXPORT_DIRECTORY_SUFFIX):
        raise ArtifactRubbingExportError(
            f"export directory must end with {RUBBING_EXPORT_DIRECTORY_SUFFIX}"
        )
    if destination.exists() or destination.is_symlink():
        raise ArtifactRubbingExportError("export destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_rubbing_export(document, record_id, raster)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.mkdir(mode=0o777)
        write_new_export_file(temporary / RUBBING_EXPORT_PNG_NAME, bundle.png_bytes)
        write_new_export_file(
            temporary / RUBBING_EXPORT_SIDECAR_NAME,
            bundle.sidecar_bytes,
        )
        fsync_export_directory(temporary)
        validate_rubbing_export_package(temporary, document=document)
        publish_export_directory_noreplace(temporary, destination)
        fsync_export_directory(destination.parent)
    except ArtifactVectorExportError as exc:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise ArtifactRubbingExportError(str(exc)) from exc
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    return destination


__all__ = [
    "ArtifactRubbingExportError",
    "MAX_RUBBING_EXPORT_SIDECAR_BYTES",
    "RUBBING_EXPORT_DIRECTORY_SUFFIX",
    "RUBBING_EXPORT_FORMAT",
    "RUBBING_EXPORT_PNG_MEDIA_TYPE",
    "RUBBING_EXPORT_PNG_NAME",
    "RUBBING_EXPORT_SCHEMA_VERSION",
    "RUBBING_EXPORT_SIDECAR_MEDIA_TYPE",
    "RUBBING_EXPORT_SIDECAR_NAME",
    "RubbingExportBundle",
    "build_rubbing_export",
    "export_rubbing_package",
    "validate_rubbing_export_bytes",
    "validate_rubbing_export_package",
]
