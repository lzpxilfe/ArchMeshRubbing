"""Immutable Digital Rubbing record receipts for ArtifactDocument.

The project manifest stores a bounded, content-addressed raster receipt rather
than base64 image bytes.  The authoritative pixels are reproducible from the
verified source, Align revision, and complete recipe; export packages carry
the actual PNG for offline use.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .artifact_document import (
    ArtifactDocument,
    ArtifactDocumentError,
    DerivedRecord,
    OperationContext,
    RecordLifecycleStatus,
)
from .artifact_outline_extractor import OutlineView, outline_frame
from .artifact_rubbing_extractor import (
    ArtifactRubbingError,
    DigitalRubbingRaster,
    RUBBING_COORDINATE_SPACE,
    RUBBING_PIXEL_FORMAT,
    RUBBING_RASTER_HASH_SCOPE,
    RUBBING_RASTER_SCHEMA_VERSION,
    validate_rubbing_recipe,
)
from .artifact_vector_record import PlanarFrame
from .canonical_json import canonical_json_bytes, canonical_json_sha256


RUBBING_RECORD_TYPE = "raster.digital_rubbing.v1"
RUBBING_RECEIPT_EXTENSION_KEY = "org.archmeshrubbing:digital-rubbing-v1"
RUBBING_RECEIPT_MEDIA_TYPE = (
    "application/vnd.archmeshrubbing.digital-rubbing-receipt+json"
)
RUBBING_GEOMETRY_REF_PREFIX = (
    "urn:archmeshrubbing:digital-rubbing-raster:sha256:"
)
MAX_RUBBING_RECEIPT_BYTES = 64 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactRubbingRecordError(ValueError):
    """A Digital Rubbing record or receipt violates its durable contract."""


def _exact_keys(value: object, expected: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactRubbingRecordError(f"{name} must be an object")
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise ArtifactRubbingRecordError(
            f"{name} is missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise ArtifactRubbingRecordError(
            f"{name} has unknown fields: {', '.join(unknown)}"
        )
    return value


def _strict_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactRubbingRecordError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ArtifactRubbingRecordError(
            f"{name} must be in the inclusive range {minimum}..{maximum}"
        )
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ArtifactRubbingRecordError(f"{name} must be a lowercase SHA-256")
    return value


def validate_rubbing_receipt(value: object) -> dict[str, Any]:
    receipt = _exact_keys(
        value,
        {
            "coordinate_space",
            "frame",
            "height_mm_exact",
            "height_pixels",
            "minimum_u_pixel_index",
            "minimum_v_pixel_index",
            "pixel_format",
            "pixels_per_meter",
            "raster_hash_scope",
            "raster_sha256",
            "raw_pixel_byte_length",
            "raw_pixel_sha256",
            "row_order",
            "schema_version",
            "view",
            "width_mm_exact",
            "width_pixels",
        },
        name="Digital Rubbing receipt",
    )
    if receipt["schema_version"] != RUBBING_RASTER_SCHEMA_VERSION:
        raise ArtifactRubbingRecordError("rubbing receipt schema is unsupported")
    if receipt["coordinate_space"] != RUBBING_COORDINATE_SPACE:
        raise ArtifactRubbingRecordError("rubbing receipt coordinate space is invalid")
    if receipt["pixel_format"] != RUBBING_PIXEL_FORMAT:
        raise ArtifactRubbingRecordError("rubbing receipt pixel format is invalid")
    if receipt["raster_hash_scope"] != RUBBING_RASTER_HASH_SCOPE:
        raise ArtifactRubbingRecordError("rubbing receipt hash scope is invalid")
    if receipt["row_order"] != "top_to_bottom_v_descending":
        raise ArtifactRubbingRecordError("rubbing receipt row order is invalid")
    try:
        view = OutlineView(receipt["view"])
    except (TypeError, ValueError) as exc:
        raise ArtifactRubbingRecordError("rubbing receipt view is invalid") from exc
    raw_frame = receipt["frame"]
    if not isinstance(raw_frame, Mapping):
        raise ArtifactRubbingRecordError("rubbing receipt frame must be an object")
    try:
        frame = PlanarFrame.from_dict(raw_frame)
    except ValueError as exc:
        raise ArtifactRubbingRecordError(str(exc)) from exc
    if frame != outline_frame(view):
        raise ArtifactRubbingRecordError("rubbing receipt frame does not match its view")
    width = _strict_int(
        receipt["width_pixels"], name="width_pixels", minimum=1, maximum=100_000
    )
    height = _strict_int(
        receipt["height_pixels"], name="height_pixels", minimum=1, maximum=100_000
    )
    ppm = _strict_int(
        receipt["pixels_per_meter"],
        name="pixels_per_meter",
        minimum=1000,
        maximum=100_000,
    )
    if ppm % 1000 != 0:
        raise ArtifactRubbingRecordError("pixels_per_meter must encode integer pixels/mm")
    minimum_u = _strict_int(
        receipt["minimum_u_pixel_index"],
        name="minimum_u_pixel_index",
        minimum=-(2**48),
        maximum=2**48,
    )
    minimum_v = _strict_int(
        receipt["minimum_v_pixel_index"],
        name="minimum_v_pixel_index",
        minimum=-(2**48),
        maximum=2**48,
    )
    byte_length = _strict_int(
        receipt["raw_pixel_byte_length"],
        name="raw_pixel_byte_length",
        minimum=2,
        maximum=16_000_000,
    )
    if byte_length != width * height * 2:
        raise ArtifactRubbingRecordError("rubbing raw pixel byte length is inconsistent")
    raw_sha = _sha256(receipt["raw_pixel_sha256"], name="raw_pixel_sha256")
    raster_sha = _sha256(receipt["raster_sha256"], name="raster_sha256")

    def exact_dimension(name: str, pixels: int) -> dict[str, int]:
        rational = _exact_keys(
            receipt[name],
            {"denominator", "numerator"},
            name=name,
        )
        denominator = _strict_int(
            rational["denominator"],
            name=f"{name}.denominator",
            minimum=1,
            maximum=100_000,
        )
        numerator = _strict_int(
            rational["numerator"],
            name=f"{name}.numerator",
            minimum=1,
            maximum=100_000_000_000,
        )
        if denominator != ppm or numerator != pixels * 1000:
            raise ArtifactRubbingRecordError(f"{name} is inconsistent with the grid")
        return {"denominator": denominator, "numerator": numerator}

    width_exact = exact_dimension("width_mm_exact", width)
    height_exact = exact_dimension("height_mm_exact", height)
    return {
        "coordinate_space": RUBBING_COORDINATE_SPACE,
        "frame": frame.to_dict(),
        "height_mm_exact": height_exact,
        "height_pixels": height,
        "minimum_u_pixel_index": minimum_u,
        "minimum_v_pixel_index": minimum_v,
        "pixel_format": RUBBING_PIXEL_FORMAT,
        "pixels_per_meter": ppm,
        "raster_hash_scope": RUBBING_RASTER_HASH_SCOPE,
        "raster_sha256": raster_sha,
        "raw_pixel_byte_length": byte_length,
        "raw_pixel_sha256": raw_sha,
        "row_order": "top_to_bottom_v_descending",
        "schema_version": RUBBING_RASTER_SCHEMA_VERSION,
        "view": view.value,
        "width_mm_exact": width_exact,
        "width_pixels": width,
    }


def _validate_qc_against_receipt(
    qc: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    if not isinstance(qc, Mapping):
        raise ArtifactRubbingRecordError("rubbing record QC must be an object")
    expected = {
        "height_pixels": receipt["height_pixels"],
        "pixel_count": int(receipt["width_pixels"]) * int(receipt["height_pixels"]),
        "pixel_format": receipt["pixel_format"],
        "pixels_per_meter": receipt["pixels_per_meter"],
        "raster_sha256": receipt["raster_sha256"],
        "raw_pixel_sha256": receipt["raw_pixel_sha256"],
        "width_pixels": receipt["width_pixels"],
    }
    for key, value in expected.items():
        if qc.get(key) != value:
            raise ArtifactRubbingRecordError(
                f"rubbing record QC field {key!r} does not match its receipt"
            )
    pixel_count = int(expected["pixel_count"])
    covered = _strict_int(
        qc.get("covered_pixel_count"),
        name="qc.covered_pixel_count",
        minimum=1,
        maximum=pixel_count,
    )
    _strict_int(
        qc.get("inked_pixel_count"),
        name="qc.inked_pixel_count",
        minimum=0,
        maximum=covered,
    )
    _strict_int(
        qc.get("ink_sum"),
        name="qc.ink_sum",
        minimum=0,
        maximum=255 * covered,
    )
    minimum_gray = _strict_int(
        qc.get("covered_gray_min"),
        name="qc.covered_gray_min",
        minimum=0,
        maximum=255,
    )
    maximum_gray = _strict_int(
        qc.get("covered_gray_max"),
        name="qc.covered_gray_max",
        minimum=0,
        maximum=255,
    )
    if minimum_gray > maximum_gray:
        raise ArtifactRubbingRecordError("rubbing gray QC bounds are reversed")
    if qc.get("alpha_binary") is not True:
        raise ArtifactRubbingRecordError("rubbing alpha mask QC is invalid")


def append_rubbing_record_from_context(
    document: ArtifactDocument,
    *,
    context: OperationContext,
    raster: DigitalRubbingRaster,
    recipe: Mapping[str, Any],
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
    qc: Mapping[str, Any] | None = None,
) -> ArtifactDocument:
    if not isinstance(document, ArtifactDocument):
        raise ArtifactRubbingRecordError("document must be an ArtifactDocument")
    if not isinstance(context, OperationContext):
        raise ArtifactRubbingRecordError("context must be an OperationContext")
    if not isinstance(raster, DigitalRubbingRaster):
        raise ArtifactRubbingRecordError("raster must be a DigitalRubbingRaster")
    try:
        validated_recipe = validate_rubbing_recipe(recipe)
    except ArtifactRubbingError as exc:
        raise ArtifactRubbingRecordError(str(exc)) from exc
    receipt = validate_rubbing_receipt(raster.receipt())
    recipe_frame = validated_recipe["frame"]
    if recipe_frame != receipt["frame"]:
        raise ArtifactRubbingRecordError("rubbing recipe and receipt frames differ")
    pixel_policy = validated_recipe["pixel_policy"]
    if not isinstance(pixel_policy, Mapping) or pixel_policy.get(
        "pixels_per_meter"
    ) != receipt["pixels_per_meter"]:
        raise ArtifactRubbingRecordError("rubbing recipe and receipt pixel grids differ")
    computed_qc = raster.qc_summary()
    for key, value in dict(qc or {}).items():
        if key in computed_qc and computed_qc[key] != value:
            raise ArtifactRubbingRecordError(
                f"caller QC cannot override computed field {key!r}"
            )
        computed_qc[key] = value
    _validate_qc_against_receipt(computed_qc, receipt)
    receipt_bytes = canonical_json_bytes(receipt)
    if len(receipt_bytes) > MAX_RUBBING_RECEIPT_BYTES:
        raise ArtifactRubbingRecordError("rubbing receipt exceeds its size limit")
    extensions = {
        RUBBING_RECEIPT_EXTENSION_KEY: {
            "media_type": RUBBING_RECEIPT_MEDIA_TYPE,
            "receipt": receipt,
            "receipt_byte_length": len(receipt_bytes),
            "receipt_sha256": canonical_json_sha256(receipt),
            "schema_version": RUBBING_RASTER_SCHEMA_VERSION,
        }
    }
    try:
        return document.append_record_from_context(
            context=context,
            id=record_id,
            type=RUBBING_RECORD_TYPE,
            geometry_ref=raster.geometry_ref,
            recipe=validated_recipe,
            qc=computed_qc,
            lifecycle_status=RecordLifecycleStatus.READY,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=depends_on_record_ids,
            extensions=extensions,
        )
    except ArtifactDocumentError as exc:
        raise ArtifactRubbingRecordError(str(exc)) from exc


def rubbing_receipt_from_record(record: DerivedRecord) -> dict[str, Any]:
    if not isinstance(record, DerivedRecord):
        raise ArtifactRubbingRecordError("record must be a DerivedRecord")
    if record.type != RUBBING_RECORD_TYPE:
        raise ArtifactRubbingRecordError("record is not a Digital Rubbing record")
    descriptor = _exact_keys(
        record.extensions.get(RUBBING_RECEIPT_EXTENSION_KEY),
        {
            "media_type",
            "receipt",
            "receipt_byte_length",
            "receipt_sha256",
            "schema_version",
        },
        name="Digital Rubbing descriptor",
    )
    if descriptor["media_type"] != RUBBING_RECEIPT_MEDIA_TYPE:
        raise ArtifactRubbingRecordError("rubbing descriptor media type is invalid")
    if descriptor["schema_version"] != RUBBING_RASTER_SCHEMA_VERSION:
        raise ArtifactRubbingRecordError("rubbing descriptor schema is invalid")
    receipt = validate_rubbing_receipt(descriptor["receipt"])
    receipt_bytes = canonical_json_bytes(receipt)
    if descriptor["receipt_byte_length"] != len(receipt_bytes):
        raise ArtifactRubbingRecordError("rubbing receipt byte length is invalid")
    if descriptor["receipt_sha256"] != canonical_json_sha256(receipt):
        raise ArtifactRubbingRecordError("rubbing receipt SHA-256 is invalid")
    if record.geometry_ref != f"{RUBBING_GEOMETRY_REF_PREFIX}{receipt['raster_sha256']}":
        raise ArtifactRubbingRecordError("rubbing geometry_ref does not match receipt")
    try:
        recipe = validate_rubbing_recipe(record.recipe)
    except ArtifactRubbingError as exc:
        raise ArtifactRubbingRecordError(str(exc)) from exc
    if recipe["frame"] != receipt["frame"]:
        raise ArtifactRubbingRecordError("rubbing record frame does not match receipt")
    pixel_policy = recipe["pixel_policy"]
    if not isinstance(pixel_policy, Mapping) or pixel_policy.get(
        "pixels_per_meter"
    ) != receipt["pixels_per_meter"]:
        raise ArtifactRubbingRecordError("rubbing record pixel grid does not match receipt")
    record_qc = record.to_dict()["qc"]
    assert isinstance(record_qc, dict)
    _validate_qc_against_receipt(record_qc, receipt)
    return receipt


def validate_rubbing_records(document: ArtifactDocument) -> None:
    if not isinstance(document, ArtifactDocument):
        raise ArtifactRubbingRecordError("document must be an ArtifactDocument")
    for record in document.records:
        if record.type == RUBBING_RECORD_TYPE:
            rubbing_receipt_from_record(record)


__all__ = [
    "ArtifactRubbingRecordError",
    "MAX_RUBBING_RECEIPT_BYTES",
    "RUBBING_GEOMETRY_REF_PREFIX",
    "RUBBING_RECEIPT_EXTENSION_KEY",
    "RUBBING_RECEIPT_MEDIA_TYPE",
    "RUBBING_RECORD_TYPE",
    "append_rubbing_record_from_context",
    "rubbing_receipt_from_record",
    "validate_rubbing_receipt",
    "validate_rubbing_records",
]
