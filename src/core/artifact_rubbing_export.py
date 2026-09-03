"""Self-contained deterministic 1:1 PNG packages for Digital Rubbing records."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
from threading import RLock
from typing import Any, Mapping, TypeAlias
import uuid

from .artifact_developed_rubbing import (
    ArtifactDevelopedRubbingError,
    DEVELOPED_RUBBING_COORDINATE_SPACE,
    DEVELOPED_RUBBING_RECORD_TYPE,
    DevelopedRubbingRaster,
    developed_rubbing_receipt_from_record,
    validate_developed_rubbing_receipt,
    validate_developed_rubbing_recipe,
)
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
    validate_current_public_export_provenance,
    validate_legacy_public_export_provenance,
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
# 1.2.0 admits a raster drawn on a developed surface next to the six-view
# raster; the receipt's coordinate space says which one a package carries.
_CURRENT_RUBBING_EXPORT_SCHEMA_VERSION = "1.2.0"
RUBBING_EXPORT_SCHEMA_VERSION = _CURRENT_RUBBING_EXPORT_SCHEMA_VERSION
SUPPORTED_RUBBING_EXPORT_SCHEMA_VERSIONS = frozenset({"1.0.0", "1.1.0", "1.2.0"})
_RUBBING_RECORD_TYPES = frozenset({RUBBING_RECORD_TYPE, DEVELOPED_RUBBING_RECORD_TYPE})
RubbingRaster: TypeAlias = DigitalRubbingRaster | DevelopedRubbingRaster
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
MAX_IGNORABLE_OS_METADATA_BYTES = 1024 * 1024
_MAX_STAGING_DIRECTORY_ATTEMPTS = 16
_RUBBING_STAGING_PREFIX = ".amrr-stage-"
_RUBBING_QUARANTINE_PREFIX = ".amrr-discard-"
_UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")
_IGNORABLE_OS_METADATA_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})
_STAGING_OWNERS_LOCK = RLock()
_STAGING_OWNERS: dict[str, _OwnedStagingDirectory] = {}
_PREPARED_PUBLICATIONS: dict[object, PreparedRubbingPublication] = {}
_quarantine_export_directory_noreplace = publish_export_directory_noreplace


class ArtifactRubbingExportError(ValueError):
    """A Digital Rubbing package violates scale, integrity, or provenance."""

    def __init__(self, message: str, *, committed: bool = False) -> None:
        super().__init__(message)
        self.committed = bool(committed)


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


@dataclass(frozen=True, slots=True)
class _OwnedStagingDirectory:
    path: Path
    destination: Path
    device: int
    inode: int
    parent_device: int
    parent_inode: int
    staging_directory_fsync_confirmed: bool = False


@dataclass(frozen=True, slots=True)
class _ExportEntryFingerprint:
    name: str
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True, eq=False)
class PreparedRubbingPublication:
    """Opaque exact authority to publish one validated rubbing staging inode."""

    staging_directory: Path
    destination: Path
    _owned: _OwnedStagingDirectory = dataclass_field(repr=False)
    _fingerprint: tuple[_ExportEntryFingerprint, ...] = dataclass_field(repr=False)
    _staging_directory_fsync_confirmed: bool = dataclass_field(repr=False)
    _nonce: object = dataclass_field(repr=False, compare=False)


def _staging_registry_key(path: Path) -> str:
    return os.path.abspath(os.fspath(path))


def _register_rubbing_staging(staging: _OwnedStagingDirectory) -> None:
    with _STAGING_OWNERS_LOCK:
        _STAGING_OWNERS[_staging_registry_key(staging.path)] = staging


def _forget_rubbing_staging(path: Path) -> None:
    with _STAGING_OWNERS_LOCK:
        key = _staging_registry_key(path)
        _STAGING_OWNERS.pop(key, None)
        for nonce, prepared in tuple(_PREPARED_PUBLICATIONS.items()):
            if _staging_registry_key(prepared.staging_directory) == key:
                _PREPARED_PUBLICATIONS.pop(nonce, None)


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
    if record is None or record.type not in _RUBBING_RECORD_TYPES:
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
        if record.type == DEVELOPED_RUBBING_RECORD_TYPE:
            receipt = developed_rubbing_receipt_from_record(record)
        else:
            receipt = rubbing_receipt_from_record(record)
    except (ArtifactRubbingRecordError, ArtifactDevelopedRubbingError) as exc:
        raise ArtifactRubbingExportError(str(exc)) from exc
    record_qc = record.to_dict()["qc"]
    if not isinstance(record_qc, dict):
        raise ArtifactRubbingExportError("record QC must be an object")
    return record, receipt, record_qc


def _receipt_is_developed(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("coordinate_space") == DEVELOPED_RUBBING_COORDINATE_SPACE


def _require_matching_raster(
    raster: RubbingRaster,
    receipt: Mapping[str, Any],
) -> None:
    if not isinstance(raster, (DigitalRubbingRaster, DevelopedRubbingRaster)):
        raise ArtifactRubbingExportError(
            "raster must be a DigitalRubbingRaster or a DevelopedRubbingRaster"
        )
    if isinstance(raster, DevelopedRubbingRaster) != _receipt_is_developed(receipt):
        raise ArtifactRubbingExportError(
            "raster coordinate space does not match the Digital Rubbing record"
        )
    if raster.receipt() != dict(receipt):
        raise ArtifactRubbingExportError(
            "recomputed raster does not match the Digital Rubbing record receipt"
        )


def _public_provenance(
    document: ArtifactDocument,
    record: DerivedRecord,
    *,
    include_current_contract: bool = True,
) -> dict[str, Any]:
    try:
        return build_public_export_provenance(
            document,
            record,
            include_current_contract=include_current_contract,
        )
    except ArtifactVectorExportError as exc:
        raise ArtifactRubbingExportError(str(exc)) from exc


def _validated_public_provenance(value: object) -> Mapping[str, Any]:
    try:
        return validate_legacy_public_export_provenance(value)
    except ArtifactVectorExportError as exc:
        raise ArtifactRubbingExportError(str(exc)) from exc


def _validated_current_public_provenance(value: object) -> Mapping[str, Any]:
    try:
        return validate_current_public_export_provenance(value)
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
    raster: RubbingRaster,
) -> RubbingExportBundle:
    record, receipt, record_qc = _require_exportable_record(document, record_id)
    _require_matching_raster(raster, receipt)
    provenance = _public_provenance(
        document,
        record,
        include_current_contract=(
            RUBBING_EXPORT_SCHEMA_VERSION
            == _CURRENT_RUBBING_EXPORT_SCHEMA_VERSION
        ),
    )
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
    raster: RubbingRaster,
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
    schema_version = root["schema_version"]
    if (
        not isinstance(schema_version, str)
        or schema_version not in SUPPORTED_RUBBING_EXPORT_SCHEMA_VERSIONS
    ):
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
    raw_receipt = root["raster_receipt"]
    developed = isinstance(raw_receipt, Mapping) and _receipt_is_developed(raw_receipt)
    if developed and schema_version != "1.2.0":
        raise ArtifactRubbingExportError(
            "a rubbing on a developed surface needs rubbing export schema 1.2.0"
        )
    raster: RubbingRaster
    try:
        pixels, ppm, metadata = decode_canonical_ga8_png(png_bytes)
        if developed:
            receipt = validate_developed_rubbing_receipt(raw_receipt)
            recipe = validate_developed_rubbing_recipe(root["recipe"])
            raster = DevelopedRubbingRaster(
                pixels=pixels,
                pixels_per_meter=ppm,
                minimum_u_pixel_index=int(receipt["minimum_u_pixel_index"]),
                minimum_v_pixel_index=int(receipt["minimum_v_pixel_index"]),
                development_sha256=str(receipt["development_sha256"]),
            )
            expected_record_type = DEVELOPED_RUBBING_RECORD_TYPE
        else:
            receipt = validate_rubbing_receipt(raw_receipt)
            recipe = validate_rubbing_recipe(root["recipe"])
            raster = DigitalRubbingRaster(
                pixels=pixels,
                frame=outline_frame_from_receipt(receipt),
                view=receipt["view"],
                pixels_per_meter=ppm,
                minimum_u_pixel_index=int(receipt["minimum_u_pixel_index"]),
                minimum_v_pixel_index=int(receipt["minimum_v_pixel_index"]),
            )
            expected_record_type = RUBBING_RECORD_TYPE
    except (
        ArtifactRubbingError,
        ArtifactRubbingRecordError,
        ArtifactDevelopedRubbingError,
        CanonicalPNGError,
    ) as exc:
        raise ArtifactRubbingExportError(str(exc)) from exc
    if raster.receipt() != receipt:
        raise ArtifactRubbingExportError("PNG pixels do not match raster receipt")
    if ppm != receipt["pixels_per_meter"]:
        raise ArtifactRubbingExportError("PNG pHYs does not match raster receipt")
    provenance = (
        _validated_current_public_provenance(root["provenance"])
        if schema_version == _CURRENT_RUBBING_EXPORT_SCHEMA_VERSION
        else _validated_public_provenance(root["provenance"])
    )
    geometry_provenance = provenance["geometry_revision"]
    assert isinstance(geometry_provenance, Mapping)
    geometry_qc = geometry_provenance["qc"]
    assert isinstance(geometry_qc, Mapping)
    if (
        schema_version == "1.0.0"
        and "import_admission" in geometry_qc
    ):
        raise ArtifactRubbingExportError(
            "rubbing export 1.0.0 cannot contain mesh admission provenance"
        )
    record_provenance = provenance["record"]
    assert isinstance(record_provenance, Mapping)
    if (
        record_provenance["type"] != expected_record_type
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
        if provenance != _public_provenance(
            document,
            record,
            include_current_contract=(
                schema_version == _CURRENT_RUBBING_EXPORT_SCHEMA_VERSION
            ),
        ):
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
    normative_entries = [
        entry for entry in entries if entry.name not in _IGNORABLE_OS_METADATA_NAMES
    ]
    ignored_entries = [
        entry for entry in entries if entry.name in _IGNORABLE_OS_METADATA_NAMES
    ]
    if [entry.name for entry in normative_entries] != sorted(
        [RUBBING_EXPORT_PNG_NAME, RUBBING_EXPORT_SIDECAR_NAME]
    ):
        raise ArtifactRubbingExportError(
            "rubbing export package must contain exactly two normative files"
        )
    if any(
        entry.is_symlink() or not entry.is_file() for entry in normative_entries
    ):
        raise ArtifactRubbingExportError("rubbing export members must be regular files")
    for entry in ignored_entries:
        if (
            entry.is_symlink()
            or not entry.is_file()
            or entry.stat().st_size > MAX_IGNORABLE_OS_METADATA_BYTES
        ):
            raise ArtifactRubbingExportError("OS metadata entry is unsafe")
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


def _validate_rubbing_destination(directory: str | os.PathLike[str]) -> Path:
    destination = Path(
        os.path.abspath(os.fspath(Path(directory).expanduser()))
    )
    if not destination.name.endswith(RUBBING_EXPORT_DIRECTORY_SUFFIX):
        raise ArtifactRubbingExportError(
            f"export directory must end with {RUBBING_EXPORT_DIRECTORY_SUFFIX}"
        )
    return destination


def _absolute_staging_path(directory: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(Path(directory).expanduser())))


def _uuid_hex() -> str:
    token = uuid.uuid4().hex.lower()
    if _UUID_HEX_RE.fullmatch(token) is None:
        raise ArtifactRubbingExportError("UUID provider returned an invalid staging token")
    return token


def _real_directory_identity(path: Path, *, label: str) -> os.stat_result:
    try:
        identity = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactRubbingExportError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISDIR(identity.st_mode):
        raise ArtifactRubbingExportError(f"{label} must be a real directory")
    return identity


def _path_exists_without_following(path: Path) -> bool:
    try:
        path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ArtifactRubbingExportError(f"cannot inspect export path: {exc}") from exc
    return True


def _fingerprint_entry(path: Path, *, name: str) -> _ExportEntryFingerprint:
    try:
        identity = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactRubbingExportError(
            f"cannot fingerprint rubbing export member {name!r}: {exc}"
        ) from exc
    return _ExportEntryFingerprint(
        name=name,
        device=identity.st_dev,
        inode=identity.st_ino,
        mode=identity.st_mode,
        size=identity.st_size,
        mtime_ns=identity.st_mtime_ns,
        ctime_ns=identity.st_ctime_ns,
    )


def _capture_rubbing_package_fingerprint(
    staging: Path,
) -> tuple[_ExportEntryFingerprint, ...]:
    directory = _fingerprint_entry(staging, name=".")
    if not stat.S_ISDIR(directory.mode):
        raise ArtifactRubbingExportError(
            "rubbing export staging path is not a real directory"
        )
    try:
        entries = sorted(staging.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise ArtifactRubbingExportError(
            f"cannot enumerate rubbing export staging directory: {exc}"
        ) from exc
    return (directory,) + tuple(
        _fingerprint_entry(entry, name=entry.name) for entry in entries
    )


def _owned_destination_is_visible(staging: _OwnedStagingDirectory) -> bool:
    try:
        current = staging.destination.stat(follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISDIR(current.st_mode)
        and current.st_dev == staging.device
        and current.st_ino == staging.inode
    )


def _raise_if_owned_destination_is_visible(
    staging: _OwnedStagingDirectory,
) -> None:
    if _owned_destination_is_visible(staging):
        raise ArtifactRubbingExportError(
            "rubbing export staging inode is already visible at the destination; "
            "publication occurred outside the authorized commit",
            committed=True,
        )


def _require_current_parent(staging: _OwnedStagingDirectory) -> None:
    parent = _real_directory_identity(
        staging.destination.parent,
        label="rubbing export destination parent",
    )
    if (parent.st_dev, parent.st_ino) != (
        staging.parent_device,
        staging.parent_inode,
    ):
        raise ArtifactRubbingExportError(
            "rubbing export destination parent was replaced"
        )


def _require_owned_staging_identity(staging: _OwnedStagingDirectory) -> None:
    try:
        current = staging.path.stat(follow_symlinks=False)
    except FileNotFoundError:
        _raise_if_owned_destination_is_visible(staging)
        raise ArtifactRubbingExportError(
            "owned rubbing export staging directory is missing"
        ) from None
    except OSError as exc:
        raise ArtifactRubbingExportError(
            f"cannot inspect rubbing export staging directory: {exc}"
        ) from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != (staging.device, staging.inode)
    ):
        raise ArtifactRubbingExportError(
            "rubbing export staging directory was replaced"
        )


def _invalidate_rubbing_prepared_locked(staging: _OwnedStagingDirectory) -> None:
    for nonce, prepared in tuple(_PREPARED_PUBLICATIONS.items()):
        if prepared._owned is staging:
            _PREPARED_PUBLICATIONS.pop(nonce, None)


def _rename_rubbing_noreplace(source: Path, destination: Path) -> None:
    try:
        publish_export_directory_noreplace(source, destination)
    except ArtifactVectorExportError as exc:
        raise ArtifactRubbingExportError(str(exc), committed=exc.committed) from exc


def _quarantine_rubbing_noreplace(source: Path, destination: Path) -> None:
    try:
        _quarantine_export_directory_noreplace(source, destination)
    except ArtifactVectorExportError as exc:
        raise ArtifactRubbingExportError(str(exc), committed=exc.committed) from exc


def _create_owned_rubbing_staging_directory(
    destination: Path,
) -> _OwnedStagingDirectory:
    parent = _real_directory_identity(
        destination.parent,
        label="rubbing export destination parent",
    )
    for _attempt in range(_MAX_STAGING_DIRECTORY_ATTEMPTS):
        candidate = destination.parent / f"{_RUBBING_STAGING_PREFIX}{_uuid_hex()}"
        try:
            candidate.mkdir(mode=0o777)
        except FileExistsError:
            # A collision belongs to another process. Never reuse or delete it.
            continue
        except OSError as exc:
            raise ArtifactRubbingExportError(
                f"cannot create rubbing export staging directory: {exc}"
            ) from exc
        try:
            identity = candidate.stat(follow_symlinks=False)
        except OSError as exc:
            raise ArtifactRubbingExportError(
                f"cannot inspect rubbing export staging directory: {exc}"
            ) from exc
        if not stat.S_ISDIR(identity.st_mode):
            raise ArtifactRubbingExportError(
                "rubbing export staging path is not a real directory"
            )
        return _OwnedStagingDirectory(
            path=candidate,
            destination=destination,
            device=identity.st_dev,
            inode=identity.st_ino,
            parent_device=parent.st_dev,
            parent_inode=parent.st_ino,
        )
    raise ArtifactRubbingExportError(
        "cannot reserve rubbing export staging directory after 16 attempts"
    )


def _empty_rubbing_directory_fd(directory_fd: int) -> None:
    """Remove entries only through a verified directory descriptor."""

    with os.scandir(directory_fd) as iterator:
        names = sorted(entry.name for entry in iterator)
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    nofollow = int(getattr(os, "O_NOFOLLOW", 0))
    for name in names:
        try:
            identity = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(identity.st_mode):
            os.unlink(name, dir_fd=directory_fd)
            continue
        child_fd = os.open(
            name,
            flags | nofollow,
            dir_fd=directory_fd,
        )
        try:
            opened_identity = os.fstat(child_fd)
            if not os.path.samestat(identity, opened_identity):
                raise ArtifactRubbingExportError(
                    "rubbing export child directory changed during cleanup"
                )
            _empty_rubbing_directory_fd(child_fd)
            current_name = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if not os.path.samestat(opened_identity, current_name):
                raise ArtifactRubbingExportError(
                    "rubbing export child directory was replaced during cleanup"
                )
            os.rmdir(name, dir_fd=directory_fd)
        finally:
            os.close(child_fd)


def _descriptor_cleanup_available() -> bool:
    required_dir_fd = (os.open, os.stat, os.unlink, os.rmdir)
    return (
        all(function in os.supports_dir_fd for function in required_dir_fd)
        and os.scandir in os.supports_fd
    )


def _windows_cleanup_fallback_required() -> bool:
    return os.name == "nt"


def _discard_owned_rubbing_staging_directory(
    staging: _OwnedStagingDirectory,
) -> bool:
    """Quarantine by rename before inspecting or recursively deleting a name."""

    _require_current_parent(staging)
    quarantine: Path | None = None
    for _attempt in range(_MAX_STAGING_DIRECTORY_ATTEMPTS):
        candidate = staging.path.parent / (
            f"{_RUBBING_QUARANTINE_PREFIX}{_uuid_hex()}"
        )
        try:
            _quarantine_rubbing_noreplace(staging.path, candidate)
        except ArtifactRubbingExportError as exc:
            if _path_exists_without_following(staging.path):
                if "already exists" in str(exc):
                    continue
                raise ArtifactRubbingExportError(
                    f"cannot quarantine rubbing export staging directory: {exc}"
                ) from exc
            _raise_if_owned_destination_is_visible(staging)
            return False
        quarantine = candidate
        break
    if quarantine is None:
        raise ArtifactRubbingExportError(
            "cannot reserve rubbing export discard quarantine after 16 attempts"
        )

    if not _descriptor_cleanup_available():
        if _windows_cleanup_fallback_required():
            # Python exposes no descriptor-relative directory deletion on
            # Windows. The fixed-length random quarantine removes the public
            # staging-name race; verify the inode immediately before the
            # best-available recursive cleanup.
            try:
                quarantined = quarantine.stat(follow_symlinks=False)
            except OSError:
                return False
            if (
                stat.S_ISDIR(quarantined.st_mode)
                and (quarantined.st_dev, quarantined.st_ino)
                == (staging.device, staging.inode)
            ):
                try:
                    shutil.rmtree(quarantine)
                except OSError as exc:
                    raise ArtifactRubbingExportError(
                        "owned rubbing export was quarantined, but Windows cleanup "
                        f"is not proven: {exc}"
                    ) from exc
                return not _path_exists_without_following(quarantine)
        try:
            _quarantine_rubbing_noreplace(quarantine, staging.path)
        except ArtifactRubbingExportError:
            pass
        return False

    parent_descriptor: int | None = None
    quarantine_descriptor: int | None = None
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    nofollow = int(getattr(os, "O_NOFOLLOW", 0))
    try:
        parent_descriptor = os.open(quarantine.parent, flags)
        quarantine_descriptor = os.open(
            quarantine.name,
            flags | nofollow,
            dir_fd=parent_descriptor,
        )
        quarantined = os.fstat(quarantine_descriptor)
    except OSError:
        if quarantine_descriptor is not None:
            os.close(quarantine_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        return False
    if (
        not stat.S_ISDIR(quarantined.st_mode)
        or (quarantined.st_dev, quarantined.st_ino)
        != (staging.device, staging.inode)
    ):
        os.close(quarantine_descriptor)
        os.close(parent_descriptor)
        try:
            _rename_rubbing_noreplace(quarantine, staging.path)
        except ArtifactRubbingExportError:
            pass
        return False

    try:
        _empty_rubbing_directory_fd(quarantine_descriptor)
        current_name = os.stat(
            quarantine.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not os.path.samestat(quarantined, current_name):
            try:
                _quarantine_rubbing_noreplace(quarantine, staging.path)
            except ArtifactRubbingExportError:
                pass
            return False
        os.rmdir(quarantine.name, dir_fd=parent_descriptor)
    except (NotImplementedError, OSError, TypeError) as exc:
        raise ArtifactRubbingExportError(
            "owned rubbing export was quarantined, but cleanup is not proven: "
            f"{exc}"
        ) from exc
    finally:
        os.close(quarantine_descriptor)
        os.close(parent_descriptor)
    if _path_exists_without_following(quarantine):
        raise ArtifactRubbingExportError(
            "owned rubbing export quarantine still exists; cleanup is not proven"
        )
    return True


def _stage_rubbing_package_owned(
    directory: str | os.PathLike[str],
    document: ArtifactDocument,
    record_id: str,
    raster: RubbingRaster,
) -> _OwnedStagingDirectory:
    destination = _validate_rubbing_destination(directory)
    if destination.exists() or destination.is_symlink():
        raise ArtifactRubbingExportError("export destination already exists")

    # Reject invalid or stale records before creating the destination parent.
    bundle = build_rubbing_export(document, record_id, raster)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ArtifactRubbingExportError(
            f"cannot create rubbing export parent directory: {exc}"
        ) from exc

    staging = _create_owned_rubbing_staging_directory(destination)
    _register_rubbing_staging(staging)
    try:
        write_new_export_file(
            staging.path / RUBBING_EXPORT_PNG_NAME,
            bundle.png_bytes,
        )
        write_new_export_file(
            staging.path / RUBBING_EXPORT_SIDECAR_NAME,
            bundle.sidecar_bytes,
        )
        staging = replace(
            staging,
            staging_directory_fsync_confirmed=fsync_export_directory(staging.path),
        )
        with _STAGING_OWNERS_LOCK:
            _STAGING_OWNERS[_staging_registry_key(staging.path)] = staging
        validate_rubbing_export_package(staging.path, document=document)
    except ArtifactVectorExportError as exc:
        converted: BaseException = ArtifactRubbingExportError(str(exc))
        try:
            discarded = _discard_owned_rubbing_staging_directory(staging)
        except Exception as cleanup_exc:
            raise ArtifactRubbingExportError(
                "rubbing export staging failed and cleanup is not proven"
            ) from cleanup_exc
        finally:
            _forget_rubbing_staging(staging.path)
        if not discarded:
            raise ArtifactRubbingExportError(
                "rubbing export staging failed and cleanup is not proven"
            ) from converted
        raise converted from exc
    except Exception as exc:
        try:
            discarded = _discard_owned_rubbing_staging_directory(staging)
        except Exception as cleanup_exc:
            raise ArtifactRubbingExportError(
                "rubbing export staging failed and cleanup is not proven"
            ) from cleanup_exc
        finally:
            _forget_rubbing_staging(staging.path)
        if not discarded:
            raise ArtifactRubbingExportError(
                "rubbing export staging failed and cleanup is not proven"
            ) from exc
        raise
    return staging


def stage_rubbing_package(
    directory: str | os.PathLike[str],
    document: ArtifactDocument,
    record_id: str,
    raster: RubbingRaster,
) -> Path:
    """Create and verify a same-parent package without publishing it.

    Ownership of the returned directory transfers to the caller. A later
    publication failure leaves that directory intact; the compatibility wrapper
    cleans only staging directories that it created itself.
    """

    staging = _stage_rubbing_package_owned(
        directory,
        document,
        record_id,
        raster,
    )
    return staging.path


def discard_staged_rubbing_package(
    staging_directory: str | os.PathLike[str],
    directory: str | os.PathLike[str],
) -> bool:
    """Delete only a rubbing staging directory still owned by this process."""

    destination = _validate_rubbing_destination(directory)
    staging = _absolute_staging_path(staging_directory)
    if (
        staging.parent != destination.parent
        or not staging.name.startswith(_RUBBING_STAGING_PREFIX)
    ):
        return False
    key = _staging_registry_key(staging)
    with _STAGING_OWNERS_LOCK:
        owned = _STAGING_OWNERS.get(key)
        if owned is None or owned.destination != destination:
            return False
        try:
            discarded = _discard_owned_rubbing_staging_directory(owned)
        finally:
            _STAGING_OWNERS.pop(key, None)
            _invalidate_rubbing_prepared_locked(owned)
        return discarded


def prepare_staged_rubbing_publication(
    staging_directory: str | os.PathLike[str],
    directory: str | os.PathLike[str],
    *,
    document: ArtifactDocument | None = None,
) -> PreparedRubbingPublication:
    """Fully validate one owned staging inode and mint an exact capability."""

    destination = _validate_rubbing_destination(directory)
    staging = _absolute_staging_path(staging_directory)
    key = _staging_registry_key(staging)
    with _STAGING_OWNERS_LOCK:
        owned = _STAGING_OWNERS.get(key)
        if owned is None:
            raise ArtifactRubbingExportError(
                "rubbing export staging directory was not created by this process"
            )
        if owned.destination != destination:
            raise ArtifactRubbingExportError(
                "rubbing export staging authority is bound to a different destination"
            )

    _require_current_parent(owned)
    _require_owned_staging_identity(owned)
    if _path_exists_without_following(destination):
        _raise_if_owned_destination_is_visible(owned)
        raise ArtifactRubbingExportError("export destination already exists")
    before = _capture_rubbing_package_fingerprint(staging)
    validate_rubbing_export_package(staging, document=document)
    after = _capture_rubbing_package_fingerprint(staging)
    if before != after:
        raise ArtifactRubbingExportError(
            "rubbing export staging package changed while being validated"
        )

    nonce = object()
    prepared = PreparedRubbingPublication(
        staging_directory=staging,
        destination=destination,
        _owned=owned,
        _fingerprint=after,
        _staging_directory_fsync_confirmed=(
            owned.staging_directory_fsync_confirmed
        ),
        _nonce=nonce,
    )
    with _STAGING_OWNERS_LOCK:
        if _STAGING_OWNERS.get(key) is not owned:
            raise ArtifactRubbingExportError(
                "rubbing export staging authority changed while being validated"
            )
        _require_current_parent(owned)
        _require_owned_staging_identity(owned)
        if _capture_rubbing_package_fingerprint(staging) != after:
            raise ArtifactRubbingExportError(
                "rubbing export staging package changed after validation"
            )
        _PREPARED_PUBLICATIONS[nonce] = prepared
    return prepared


def discard_prepared_rubbing_package(
    prepared: PreparedRubbingPublication,
) -> bool:
    """Discard only the inode authorized by the exact prepared capability."""

    if not isinstance(prepared, PreparedRubbingPublication):
        raise ArtifactRubbingExportError(
            "prepared publication must be a PreparedRubbingPublication"
        )
    with _STAGING_OWNERS_LOCK:
        if _PREPARED_PUBLICATIONS.get(prepared._nonce) is not prepared:
            _raise_if_owned_destination_is_visible(prepared._owned)
            return False
    return discard_staged_rubbing_package(
        prepared.staging_directory,
        prepared.destination,
    )


def publish_prepared_rubbing_package(
    prepared: PreparedRubbingPublication,
) -> Path:
    """Fast final commit for an exact, fully validated rubbing capability."""

    if not isinstance(prepared, PreparedRubbingPublication):
        raise ArtifactRubbingExportError(
            "prepared publication must be a PreparedRubbingPublication"
        )
    owned = prepared._owned
    key = _staging_registry_key(prepared.staging_directory)
    with _STAGING_OWNERS_LOCK:
        if _PREPARED_PUBLICATIONS.get(prepared._nonce) is not prepared:
            _raise_if_owned_destination_is_visible(owned)
            raise ArtifactRubbingExportError(
                "prepared rubbing publication capability is invalid or consumed"
            )
        if _STAGING_OWNERS.get(key) is not owned:
            _raise_if_owned_destination_is_visible(owned)
            raise ArtifactRubbingExportError(
                "rubbing export staging authority is no longer current"
            )
        _require_current_parent(owned)
        _require_owned_staging_identity(owned)
        if _path_exists_without_following(prepared.destination):
            _raise_if_owned_destination_is_visible(owned)
            raise ArtifactRubbingExportError("export destination already exists")
        if (
            _capture_rubbing_package_fingerprint(prepared.staging_directory)
            != prepared._fingerprint
        ):
            raise ArtifactRubbingExportError(
                "rubbing export staging package changed after preparation"
            )
        _rename_rubbing_noreplace(
            prepared.staging_directory,
            prepared.destination,
        )
        try:
            published_identity = prepared.destination.stat(follow_symlinks=False)
        except OSError as exc:
            raise ArtifactRubbingExportError(
                "rubbing export was renamed, but destination identity could not be "
                f"verified: {exc}",
                committed=True,
            ) from exc
        if (
            not stat.S_ISDIR(published_identity.st_mode)
            or (published_identity.st_dev, published_identity.st_ino)
            != (owned.device, owned.inode)
        ):
            raise ArtifactRubbingExportError(
                "rubbing export was renamed, but destination inode is not the "
                "authorized staging inode",
                committed=True,
            )
        _STAGING_OWNERS.pop(key, None)
        _invalidate_rubbing_prepared_locked(owned)
    try:
        parent_fsync_confirmed = fsync_export_directory(
            prepared.destination.parent
        )
    except OSError as exc:
        raise ArtifactRubbingExportError(
            "rubbing export was atomically published, but directory fsync failed; "
            f"crash durability is uncertain: {exc}",
            committed=True,
        ) from exc
    if (
        not prepared._staging_directory_fsync_confirmed
        or not parent_fsync_confirmed
    ):
        raise ArtifactRubbingExportError(
            "rubbing export was atomically published, but directory fsync is "
            "unsupported; crash durability is uncertain",
            committed=True,
        )
    return prepared.destination


def publish_staged_rubbing_package(
    staging_directory: str | os.PathLike[str],
    directory: str | os.PathLike[str],
    *,
    document: ArtifactDocument | None = None,
) -> Path:
    """Compatibility wrapper: prepare fully, then commit the exact capability."""

    prepared = prepare_staged_rubbing_publication(
        staging_directory,
        directory,
        document=document,
    )
    return publish_prepared_rubbing_package(prepared)


def export_rubbing_package(
    directory: str | os.PathLike[str],
    document: ArtifactDocument,
    record_id: str,
    raster: RubbingRaster,
) -> Path:
    """Stage and atomically publish a new ``*.amr-rubbing`` package."""

    staging = stage_rubbing_package(
        directory,
        document,
        record_id,
        raster,
    )
    try:
        return publish_staged_rubbing_package(
            staging,
            directory,
            document=document,
        )
    except Exception as exc:
        if isinstance(exc, ArtifactRubbingExportError) and exc.committed:
            raise
        try:
            discarded = discard_staged_rubbing_package(staging, directory)
        except ArtifactRubbingExportError as cleanup_exc:
            if cleanup_exc.committed:
                raise
            raise ArtifactRubbingExportError(
                "rubbing export failed and staging cleanup is not proven"
            ) from cleanup_exc
        if not discarded:
            raise ArtifactRubbingExportError(
                "rubbing export failed and staging cleanup is not proven"
            ) from exc
        raise


__all__ = [
    "ArtifactRubbingExportError",
    "MAX_RUBBING_EXPORT_SIDECAR_BYTES",
    "MAX_IGNORABLE_OS_METADATA_BYTES",
    "RUBBING_EXPORT_DIRECTORY_SUFFIX",
    "RUBBING_EXPORT_FORMAT",
    "RUBBING_EXPORT_PNG_MEDIA_TYPE",
    "RUBBING_EXPORT_PNG_NAME",
    "RUBBING_EXPORT_SCHEMA_VERSION",
    "SUPPORTED_RUBBING_EXPORT_SCHEMA_VERSIONS",
    "RUBBING_EXPORT_SIDECAR_MEDIA_TYPE",
    "RUBBING_EXPORT_SIDECAR_NAME",
    "RubbingExportBundle",
    "PreparedRubbingPublication",
    "build_rubbing_export",
    "discard_staged_rubbing_package",
    "discard_prepared_rubbing_package",
    "export_rubbing_package",
    "publish_staged_rubbing_package",
    "prepare_staged_rubbing_publication",
    "publish_prepared_rubbing_package",
    "stage_rubbing_package",
    "validate_rubbing_export_bytes",
    "validate_rubbing_export_package",
]
