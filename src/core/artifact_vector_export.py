"""Deterministic, self-contained 1:1 SVG exports for artifact vector records.

The SVG is a presentation derivative of a verified canonical-mm vector
payload.  A JSON sidecar is normative for provenance and integrity.  Export
is deliberately fail-closed: only READY, FRESH records whose embedded payload
and computed QC verify may leave the authoritative document boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from threading import RLock
from typing import Any, Mapping
import uuid
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape as xml_escape

import numpy as np

from .canonical_json import CanonicalJSONError, canonical_json_sha256
from .artifact_document import (
    ARTIFACT_DOCUMENT_SCHEMA_VERSION,
    PRIMARY_SOURCE_ASSET_ROLE,
    AlignRevision,
    ArtifactDocument,
    ArtifactDocumentError,
    DerivedRecord,
    GeometryRevision,
    RecordFreshness,
    RecordLifecycleStatus,
    SourceMetadataRevision,
    canonical_recipe_hash,
)
from .artifact_vector_record import (
    ArtifactVectorRecordError,
    VectorGeometryPayload,
    VectorRecordKind,
    validate_vector_recipe,
    vector_payload_from_record,
)
from .mesh_import_recipe import (
    MeshImportRecipeError,
    validate_mesh_import_recipe,
)
from .source_identity import PRIMARY_FILE_IDENTITY_SCOPE


VECTOR_EXPORT_FORMAT = "archmeshrubbing_vector_export"
VECTOR_EXPORT_SCHEMA_VERSION = "1.0.0"
VECTOR_EXPORT_DIRECTORY_SUFFIX = ".amr-vector"
VECTOR_EXPORT_SVG_NAME = "artifact.svg"
VECTOR_EXPORT_SIDECAR_NAME = "artifact.amr-vector.json"
VECTOR_EXPORT_SVG_MEDIA_TYPE = "image/svg+xml"
VECTOR_EXPORT_SIDECAR_MEDIA_TYPE = (
    "application/vnd.archmeshrubbing.vector-export+json"
)
VECTOR_SVG_METADATA_FORMAT = "archmeshrubbing_svg_metadata"
VECTOR_SVG_METADATA_SCHEMA_VERSION = "1.0.0"
MAX_VECTOR_EXPORT_SVG_BYTES = 64 * 1024 * 1024
MAX_VECTOR_EXPORT_SIDECAR_BYTES = 24 * 1024 * 1024
MAX_IGNORABLE_OS_METADATA_BYTES = 1024 * 1024
_MAX_STAGING_DIRECTORY_ATTEMPTS = 16
_VECTOR_STAGING_PREFIX = ".amrv-stage-"
_VECTOR_QUARANTINE_PREFIX = ".amrv-discard-"
_UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")
_STAGING_OWNERS_LOCK = RLock()
_STAGING_OWNERS: dict[str, _OwnedStagingDirectory] = {}
_PREPARED_PUBLICATIONS: dict[object, PreparedVectorPublication] = {}

_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_SECONDS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_PUBLIC_ALIGN_RECIPE_KEYS = frozenset(
    {"convention", "kind", "pivot_mm", "rotation_deg", "translation_mm"}
)
_PUBLIC_ALIGN_QC_KEYS = frozenset({"proper_rigid", "rigid"})
_PUBLIC_GEOMETRY_RECIPE_KEYS = frozenset(
    {
        "dependency_policy",
        "force",
        "format",
        "loader",
        "loader_version",
        "maintain_order",
        "parser_runtime_sha256",
        "process",
        "recipe_id",
        "recipe_version",
        "runtime_lock_sha256",
        "sanitizer",
        "scene_merge",
    }
)
_PUBLIC_GEOMETRY_QC_KEYS = frozenset(
    {"face_count", "finite_vertices", "vertex_count"}
)
_IGNORABLE_OS_METADATA_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


class ArtifactVectorExportError(ValueError):
    """A vector export cannot prove its physical scale or provenance."""

    def __init__(self, message: str, *, committed: bool = False) -> None:
        super().__init__(message)
        self.committed = bool(committed)


def _finite_number(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ArtifactVectorExportError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ArtifactVectorExportError(f"{field_name} must be a finite number")
    if minimum is not None and number < minimum:
        raise ArtifactVectorExportError(f"{field_name} must be at least {minimum}")
    if strictly_positive and number <= 0.0:
        raise ArtifactVectorExportError(f"{field_name} must be greater than zero")
    return 0.0 if number == 0.0 else number


def _required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactVectorExportError(f"{field_name} must be a non-empty string")
    return value


def _exact_keys(
    value: object,
    expected: set[str],
    *,
    model_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactVectorExportError(f"{model_name} must be an object")
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise ArtifactVectorExportError(
            f"{model_name} is missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise ArtifactVectorExportError(
            f"{model_name} has unknown fields: {', '.join(unknown)}"
        )
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ArtifactVectorExportError(
            f"vector export sidecar is not strict JSON: {exc}"
        ) from exc


def _strict_json_bytes(value: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_constant(token: str) -> None:
        raise ArtifactVectorExportError(f"{label} contains non-finite number {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ArtifactVectorExportError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = item
        return result

    try:
        decoded = value.decode("utf-8")
        result = json.loads(
            decoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise ArtifactVectorExportError(f"{label} is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactVectorExportError(f"{label} is invalid JSON: {exc.msg}") from exc
    except ArtifactVectorExportError:
        raise
    except (RecursionError, ValueError) as exc:
        raise ArtifactVectorExportError(f"{label} cannot be parsed safely") from exc
    if not isinstance(result, Mapping):
        raise ArtifactVectorExportError(f"{label} root must be an object")
    return result


def _number_token(value: object, *, field_name: str) -> str:
    """Return the fixed precision token shared by SVG size and viewBox."""

    number = _finite_number(value, field_name=field_name)
    token = f"{number:.12f}".rstrip("0").rstrip(".")
    if token in {"", "-0"}:
        return "0"
    if not math.isclose(float(token), number, rel_tol=0.0, abs_tol=5e-13):
        raise ArtifactVectorExportError(
            f"{field_name} exceeds the 12-decimal SVG precision contract"
        )
    return token


def _xml_attribute(value: object) -> str:
    return xml_escape(str(value), {'"': "&quot;", "'": "&apos;"})


@dataclass(frozen=True, slots=True)
class VectorSVGOptions:
    margin_mm: float = 5.0
    stroke_width_mm: float = 0.2
    stroke_color: str = "#111111"
    title: str = "ArchMeshRubbing measured vector"

    def __post_init__(self) -> None:
        margin = _finite_number(
            self.margin_mm,
            field_name="margin_mm",
            minimum=0.0,
        )
        stroke_width = _finite_number(
            self.stroke_width_mm,
            field_name="stroke_width_mm",
            strictly_positive=True,
        )
        color = _required_string(self.stroke_color, field_name="stroke_color")
        if _HEX_COLOR_RE.fullmatch(color) is None:
            raise ArtifactVectorExportError(
                "stroke_color must be a six-digit hexadecimal color"
            )
        title = _required_string(self.title, field_name="title").strip()
        if len(title) > 512:
            raise ArtifactVectorExportError("title must not exceed 512 characters")
        if margin < stroke_width / 2.0:
            raise ArtifactVectorExportError(
                "margin_mm must be at least half of stroke_width_mm to prevent clipping"
            )
        object.__setattr__(self, "margin_mm", margin)
        object.__setattr__(self, "stroke_width_mm", stroke_width)
        object.__setattr__(self, "stroke_color", color.lower())
        object.__setattr__(self, "title", title)


@dataclass(frozen=True, slots=True)
class VectorExportBundle:
    svg_bytes: bytes
    sidecar_bytes: bytes
    svg_sha256: str
    sidecar_sha256: str
    vector_payload_sha256: str
    width_mm: float
    height_mm: float


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
class PreparedVectorPublication:
    """Opaque authority to publish one fully validated staging inode once.

    Equality is intentionally identity-only.  The module registry must contain
    this exact instance; constructing or copying a look-alike never grants
    publication authority.
    """

    staging_directory: Path
    destination: Path
    _owned: _OwnedStagingDirectory = dataclass_field(repr=False)
    _fingerprint: tuple[_ExportEntryFingerprint, ...] = dataclass_field(repr=False)
    _staging_directory_fsync_confirmed: bool = dataclass_field(repr=False)
    _nonce: object = dataclass_field(repr=False, compare=False)


def _staging_registry_key(path: Path) -> str:
    return os.path.abspath(os.fspath(path))


def _register_vector_staging(staging: _OwnedStagingDirectory) -> None:
    with _STAGING_OWNERS_LOCK:
        _STAGING_OWNERS[_staging_registry_key(staging.path)] = staging


def _forget_vector_staging(path: Path) -> None:
    with _STAGING_OWNERS_LOCK:
        key = _staging_registry_key(path)
        _STAGING_OWNERS.pop(key, None)
        for nonce, prepared in tuple(_PREPARED_PUBLICATIONS.items()):
            if _staging_registry_key(prepared.staging_directory) == key:
                _PREPARED_PUBLICATIONS.pop(nonce, None)


def _payload_bounds(payload: VectorGeometryPayload) -> tuple[float, float, float, float]:
    points = np.vstack(
        [np.asarray(path.points_mm, dtype=np.float64) for path in payload.paths]
    )
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return (
        float(minimum[0]),
        float(minimum[1]),
        float(maximum[0]),
        float(maximum[1]),
    )


def _dimensions(
    bounds: tuple[float, float, float, float],
    margin_mm: float,
) -> tuple[float, float]:
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    width = (maximum_x - minimum_x) + 2.0 * margin_mm
    height = (maximum_y - minimum_y) + 2.0 * margin_mm
    _finite_number(width, field_name="width_mm", strictly_positive=True)
    _finite_number(height, field_name="height_mm", strictly_positive=True)
    return width, height


def _verified_record_qc(
    record: DerivedRecord,
    payload: VectorGeometryPayload,
) -> dict[str, Any]:
    expected = payload.qc_summary()
    record_qc = record.to_dict()["qc"]
    if not isinstance(record_qc, dict):
        raise ArtifactVectorExportError("record QC must be an object")
    for key, expected_value in expected.items():
        if record_qc.get(key) != expected_value:
            raise ArtifactVectorExportError(
                f"record QC field {key!r} does not match the vector payload"
            )
    return record_qc


def _require_exportable_record(
    document: ArtifactDocument,
    record_id: str,
) -> tuple[DerivedRecord, VectorGeometryPayload, dict[str, Any]]:
    if not isinstance(document, ArtifactDocument):
        raise ArtifactVectorExportError("document must be an ArtifactDocument")
    record_key = _required_string(record_id, field_name="record_id")
    record = document.record_index.get(record_key)
    if record is None:
        raise ArtifactVectorExportError(f"record {record_key!r} does not exist")
    if record.lifecycle_status is not RecordLifecycleStatus.READY:
        raise ArtifactVectorExportError("only READY vector records may be exported")
    try:
        freshness = document.record_freshness(record.id)
    except ArtifactDocumentError as exc:
        raise ArtifactVectorExportError(str(exc)) from exc
    if freshness is not RecordFreshness.FRESH:
        raise ArtifactVectorExportError(
            f"only FRESH vector records may be exported (got {freshness.value})"
        )
    try:
        payload = vector_payload_from_record(record)
    except ArtifactVectorRecordError as exc:
        raise ArtifactVectorExportError(str(exc)) from exc
    return record, payload, _verified_record_qc(record, payload)


def _source_asset_provenance(document: ArtifactDocument, record: DerivedRecord) -> list[dict[str, Any]]:
    geometry = document.geometry_revision_index[record.geometry_revision_id]
    result: list[dict[str, Any]] = []
    for source_id in geometry.source_asset_ids:
        asset = document.source_asset_index[source_id]
        result.append(
            {
                "id": asset.id,
                "identity_scope": asset.identity_scope,
                "media_type": asset.media_type,
                "original_name": asset.original_name,
                "role": asset.role,
                "sha256": asset.sha256,
                "size_bytes": asset.size_bytes,
            }
        )
    return result


def _public_mapping(value: Mapping[str, Any], allowed_keys: frozenset[str]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in sorted(value)
        if key in allowed_keys
    }


def _public_mesh_import_recipe(
    value: Mapping[str, Any],
    *,
    require_current_runtime: bool,
) -> dict[str, Any]:
    """Return the complete path-free executable parser contract.

    Strict recipes contain runtime identity and dependency-policy fields which
    are necessary to reproduce the source geometry.  Unknown document-only
    extensions remain private, while every recognized recipe field is kept and
    the resulting closed contract is validated before it enters provenance.
    """

    public = _public_mapping(value, _PUBLIC_GEOMETRY_RECIPE_KEYS)
    try:
        validate_mesh_import_recipe(
            public,
            allow_legacy=True,
            require_current_runtime=require_current_runtime,
        )
    except MeshImportRecipeError as exc:
        raise ArtifactVectorExportError(
            f"invalid public mesh import recipe: {exc}"
        ) from exc
    return public


def _public_align_revision(document: ArtifactDocument, record: DerivedRecord) -> dict[str, Any]:
    align = document.align_revision_index[record.align_revision_id]
    data = align.to_dict()
    return {
        "created_at": data["created_at"],
        "id": data["id"],
        "matrix4x4": data["matrix4x4"],
        "operator": data["operator"],
        "parent_id": data["parent_id"],
        "qc": _public_mapping(data["qc"], _PUBLIC_ALIGN_QC_KEYS),
        "recipe": _public_mapping(data["recipe"], _PUBLIC_ALIGN_RECIPE_KEYS),
        "source_metadata_revision_id": data["source_metadata_revision_id"],
    }


def _public_metadata_revision(document: ArtifactDocument, record: DerivedRecord) -> dict[str, Any]:
    align = document.align_revision_index[record.align_revision_id]
    metadata = document.source_metadata_revision_index[align.source_metadata_revision_id]
    data = metadata.to_dict()
    return {
        "axes": data["axes"],
        "confirmation_status": data["confirmation_status"],
        "created_at": data["created_at"],
        "geometry_revision_id": data["geometry_revision_id"],
        "handedness": data["handedness"],
        "id": data["id"],
        "operator": data["operator"],
        "parent_id": data["parent_id"],
        "source_to_canonical_mm": data["source_to_canonical_mm"],
        "unit": data["unit"],
    }


def _public_geometry_revision(document: ArtifactDocument, record: DerivedRecord) -> dict[str, Any]:
    geometry = document.geometry_revision_index[record.geometry_revision_id]
    data = geometry.to_dict()
    return {
        "created_at": data["created_at"],
        "geometry_hash_scope": data["geometry_hash_scope"],
        "geometry_sha256": data["geometry_sha256"],
        "id": data["id"],
        "import_recipe": _public_mesh_import_recipe(
            data["import_recipe"],
            require_current_runtime=True,
        ),
        "operator": data["operator"],
        "qc": _public_mapping(data["qc"], _PUBLIC_GEOMETRY_QC_KEYS),
        "source_asset_ids": data["source_asset_ids"],
    }


def _record_receipt(document: ArtifactDocument, record: DerivedRecord) -> dict[str, Any]:
    return {
        "align_revision_id": record.align_revision_id,
        "depends_on_record_ids": list(record.depends_on_record_ids),
        "freshness": document.record_freshness(record.id).value,
        "geometry_ref": record.geometry_ref,
        "geometry_revision_id": record.geometry_revision_id,
        "id": record.id,
        "lifecycle_status": RecordLifecycleStatus(record.lifecycle_status).value,
        "recipe_hash": record.recipe_hash,
        "type": record.type,
    }


def _dependency_closure(document: ArtifactDocument, record: DerivedRecord) -> list[dict[str, Any]]:
    records = document.record_index
    pending = list(record.depends_on_record_ids)
    seen: set[str] = set()
    while pending:
        dependency_id = pending.pop()
        if dependency_id in seen:
            continue
        dependency = records.get(dependency_id)
        if dependency is None:
            raise ArtifactVectorExportError(
                f"record dependency {dependency_id!r} is missing"
            )
        seen.add(dependency_id)
        pending.extend(dependency.depends_on_record_ids)
    return [_record_receipt(document, records[record_id]) for record_id in sorted(seen)]


def _provenance(document: ArtifactDocument, record: DerivedRecord) -> dict[str, Any]:
    return {
        "align_revision": _public_align_revision(document, record),
        "dependency_closure": _dependency_closure(document, record),
        "document": {
            "active_align_revision_id": document.active_align_revision_id,
            "active_source_metadata_revision_id": (
                document.active_source_metadata_revision_id
            ),
            "document_id": document.document_id,
            "manifest_sha256": _sha256_bytes(document.canonical_json_bytes()),
            "schema_version": document.schema_version,
            "software_version": document.software_version,
        },
        "geometry_revision": _public_geometry_revision(document, record),
        "record": {
            "align_revision_id": record.align_revision_id,
            "created_at": record.created_at,
            "depends_on_record_ids": list(record.depends_on_record_ids),
            "geometry_revision_id": record.geometry_revision_id,
            "geometry_ref": record.geometry_ref,
            "id": record.id,
            "lifecycle_status": RecordLifecycleStatus(record.lifecycle_status).value,
            "operator": record.operator,
            "recipe_hash": record.recipe_hash,
            "selection_hash": record.selection_hash,
            "type": record.type,
        },
        "source_assets": _source_asset_provenance(document, record),
        "source_metadata_revision": _public_metadata_revision(document, record),
    }


def build_public_export_provenance(
    document: ArtifactDocument,
    record: DerivedRecord,
) -> dict[str, Any]:
    """Build the shared, path-sanitized provenance shape for public exports."""

    return _provenance(document, record)


def _svg_metadata(
    *,
    provenance: Mapping[str, Any],
    payload_sha256: str,
    sidecar_claims_sha256: str,
    width_mm: float,
    height_mm: float,
) -> dict[str, Any]:
    document = provenance["document"]
    record = provenance["record"]
    assert isinstance(document, Mapping)
    assert isinstance(record, Mapping)
    return {
        "document_id": document["document_id"],
        "document_manifest_sha256": document["manifest_sha256"],
        "format": VECTOR_SVG_METADATA_FORMAT,
        "geometry_ref": record["geometry_ref"],
        "height_mm": height_mm,
        "physical_scale": "1:1",
        "recipe_hash": record["recipe_hash"],
        "record_id": record["id"],
        "record_type": record["type"],
        "schema_version": VECTOR_SVG_METADATA_SCHEMA_VERSION,
        "sidecar": VECTOR_EXPORT_SIDECAR_NAME,
        "sidecar_claims_sha256": sidecar_claims_sha256,
        "unit": "mm",
        "vector_payload_sha256": payload_sha256,
        "width_mm": width_mm,
    }


def _render_svg(
    payload: VectorGeometryPayload,
    *,
    options: VectorSVGOptions,
    provenance: Mapping[str, Any],
    sidecar_claims_sha256: str,
) -> tuple[bytes, tuple[float, float, float, float], float, float]:
    bounds = _payload_bounds(payload)
    width_mm, height_mm = _dimensions(bounds, options.margin_mm)
    width_token = _number_token(width_mm, field_name="width_mm")
    height_token = _number_token(height_mm, field_name="height_mm")
    stroke_token = _number_token(
        options.stroke_width_mm,
        field_name="stroke_width_mm",
    )
    metadata = _svg_metadata(
        provenance=provenance,
        payload_sha256=payload.sha256,
        sidecar_claims_sha256=sidecar_claims_sha256,
        width_mm=width_mm,
        height_mm=height_mm,
    )
    metadata_text = _canonical_json_bytes(metadata).decode("utf-8").rstrip("\n")
    minimum_x, _minimum_y, _maximum_x, maximum_y = bounds
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="{_SVG_NAMESPACE}" version="1.1" '
            f'width="{width_token}mm" height="{height_token}mm" '
            f'viewBox="0 0 {width_token} {height_token}">'
        ),
        f"  <title>{xml_escape(options.title)}</title>",
        (
            '  <metadata id="archmeshrubbing-provenance">'
            f"{xml_escape(metadata_text)}</metadata>"
        ),
        (
            '  <g id="measured-vectors" fill="none" '
            f'stroke="{options.stroke_color}" stroke-width="{stroke_token}" '
            'stroke-linecap="round" stroke-linejoin="round">'
        ),
    ]
    for path in payload.paths:
        commands: list[str] = []
        for index, point in enumerate(path.points_mm):
            x = float(point[0]) - minimum_x + options.margin_mm
            y = maximum_y - float(point[1]) + options.margin_mm
            command = "M" if index == 0 else "L"
            commands.append(
                f"{command} {_number_token(x, field_name='path.x')} "
                f"{_number_token(y, field_name='path.y')}"
            )
        if path.closed:
            commands.append("Z")
        lines.append(
            f'    <path id="{_xml_attribute(path.id)}" '
            f'data-role="{_xml_attribute(path.role)}" d="{' '.join(commands)}"/>'
        )
    lines.extend(("  </g>", "</svg>"))
    svg_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    if len(svg_bytes) > MAX_VECTOR_EXPORT_SVG_BYTES:
        raise ArtifactVectorExportError("SVG exceeds the export safety limit")
    return svg_bytes, bounds, width_mm, height_mm


def _presentation(
    *,
    bounds: tuple[float, float, float, float],
    width_mm: float,
    height_mm: float,
    options: VectorSVGOptions,
) -> dict[str, Any]:
    return {
        "content_bounds_mm": list(bounds),
        "height_mm": height_mm,
        "margin_mm": options.margin_mm,
        "physical_scale": "1:1",
        "stroke_color": options.stroke_color,
        "stroke_width_mm": options.stroke_width_mm,
        "title": options.title,
        "unit": "mm",
        "view_box": [
            "0",
            "0",
            _number_token(width_mm, field_name="width_mm"),
            _number_token(height_mm, field_name="height_mm"),
        ],
        "width_mm": width_mm,
    }


def _sidecar_claims_sha256(sidecar: Mapping[str, Any]) -> str:
    """Bind every normative sidecar claim into the canonical SVG metadata.

    The artifact descriptor is excluded because it contains the SVG hash and
    would create a circular digest.  Provenance, recipe, QC, presentation, and
    the full vector payload are all covered.
    """

    claims = {
        key: sidecar[key]
        for key in (
            "format",
            "presentation",
            "provenance",
            "qc",
            "recipe",
            "schema_version",
            "vector_payload",
            "vector_payload_sha256",
        )
    }
    try:
        return canonical_json_sha256(claims)
    except CanonicalJSONError as exc:
        raise ArtifactVectorExportError(str(exc)) from exc


def build_vector_export(
    document: ArtifactDocument,
    record_id: str,
    *,
    options: VectorSVGOptions | None = None,
) -> VectorExportBundle:
    """Build and internally verify a canonical SVG + provenance sidecar."""

    render_options = VectorSVGOptions() if options is None else options
    if not isinstance(render_options, VectorSVGOptions):
        raise ArtifactVectorExportError("options must be VectorSVGOptions")
    record, payload, record_qc = _require_exportable_record(document, record_id)
    provenance = _provenance(document, record)
    bounds = _payload_bounds(payload)
    width_mm, height_mm = _dimensions(bounds, render_options.margin_mm)
    sidecar: dict[str, Any] = {
        "format": VECTOR_EXPORT_FORMAT,
        "presentation": _presentation(
            bounds=bounds,
            width_mm=width_mm,
            height_mm=height_mm,
            options=render_options,
        ),
        "provenance": provenance,
        "qc": {
            "export_gate": {
                "payload_verified": True,
                "record_freshness": RecordFreshness.FRESH.value,
                "record_lifecycle_status": RecordLifecycleStatus.READY.value,
            },
            "payload": payload.qc_summary(),
            "record": record_qc,
            "scale": {
                "physical_scale": "1:1",
                "unit": "mm",
                "viewbox_matches_physical_dimensions": True,
            },
        },
        "recipe": record.to_dict()["recipe"],
        "schema_version": VECTOR_EXPORT_SCHEMA_VERSION,
        "vector_payload": payload.to_dict(),
        "vector_payload_sha256": payload.sha256,
    }
    claims_sha256 = _sidecar_claims_sha256(sidecar)
    svg_bytes, rendered_bounds, rendered_width, rendered_height = _render_svg(
        payload,
        options=render_options,
        provenance=provenance,
        sidecar_claims_sha256=claims_sha256,
    )
    if (
        rendered_bounds != bounds
        or rendered_width != width_mm
        or rendered_height != height_mm
    ):
        raise ArtifactVectorExportError("SVG renderer changed the measured bounds")
    svg_sha256 = _sha256_bytes(svg_bytes)
    sidecar["artifact"] = {
        "file": VECTOR_EXPORT_SVG_NAME,
        "media_type": VECTOR_EXPORT_SVG_MEDIA_TYPE,
        "sha256": svg_sha256,
        "size_bytes": len(svg_bytes),
    }
    sidecar_bytes = _canonical_json_bytes(sidecar)
    if len(sidecar_bytes) > MAX_VECTOR_EXPORT_SIDECAR_BYTES:
        raise ArtifactVectorExportError("sidecar exceeds the export safety limit")
    bundle = validate_vector_export_bytes(
        svg_bytes,
        sidecar_bytes,
        document=document,
    )
    return bundle


def _validated_sidecar_payload(
    sidecar: Mapping[str, Any],
) -> VectorGeometryPayload:
    raw_payload = sidecar["vector_payload"]
    if not isinstance(raw_payload, Mapping):
        raise ArtifactVectorExportError("vector_payload must be an object")
    try:
        payload = VectorGeometryPayload.from_dict(raw_payload)
    except ArtifactVectorRecordError as exc:
        raise ArtifactVectorExportError(str(exc)) from exc
    digest = sidecar["vector_payload_sha256"]
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ArtifactVectorExportError("vector_payload_sha256 is invalid")
    if digest != payload.sha256:
        raise ArtifactVectorExportError(
            "vector payload semantic SHA-256 does not match the sidecar"
        )
    return payload


def _options_from_presentation(value: object) -> tuple[VectorSVGOptions, Mapping[str, Any]]:
    presentation = _exact_keys(
        value,
        {
            "content_bounds_mm",
            "height_mm",
            "margin_mm",
            "physical_scale",
            "stroke_color",
            "stroke_width_mm",
            "title",
            "unit",
            "view_box",
            "width_mm",
        },
        model_name="presentation",
    )
    if presentation["physical_scale"] != "1:1" or presentation["unit"] != "mm":
        raise ArtifactVectorExportError("presentation must declare 1:1 millimetres")
    options = VectorSVGOptions(
        margin_mm=_finite_number(
            presentation["margin_mm"], field_name="presentation.margin_mm", minimum=0.0
        ),
        stroke_width_mm=_finite_number(
            presentation["stroke_width_mm"],
            field_name="presentation.stroke_width_mm",
            strictly_positive=True,
        ),
        stroke_color=_required_string(
            presentation["stroke_color"], field_name="presentation.stroke_color"
        ),
        title=_required_string(presentation["title"], field_name="presentation.title"),
    )
    return options, presentation


def _validate_dependency_closure(
    value: object,
    *,
    root_record: Mapping[str, Any],
    expected_align_id: str,
    expected_geometry_id: str,
) -> None:
    if not isinstance(value, list):
        raise ArtifactVectorExportError("provenance.dependency_closure must be an array")
    receipts: dict[str, Mapping[str, Any]] = {}
    for index, raw_receipt in enumerate(value):
        receipt = _exact_keys(
            raw_receipt,
            {
                "align_revision_id",
                "depends_on_record_ids",
                "freshness",
                "geometry_ref",
                "geometry_revision_id",
                "id",
                "lifecycle_status",
                "recipe_hash",
                "type",
            },
            model_name=f"provenance.dependency_closure[{index}]",
        )
        for key in (
            "align_revision_id",
            "geometry_ref",
            "geometry_revision_id",
            "id",
            "type",
        ):
            _required_string(
                receipt[key],
                field_name=f"provenance.dependency_closure[{index}].{key}",
            )
        receipt_id = str(receipt["id"])
        if receipt_id == root_record["id"] or receipt_id in receipts:
            raise ArtifactVectorExportError("dependency closure contains a duplicate/root ID")
        recipe_hash = receipt["recipe_hash"]
        if not isinstance(recipe_hash, str) or _SHA256_RE.fullmatch(recipe_hash) is None:
            raise ArtifactVectorExportError("dependency receipt recipe_hash is invalid")
        dependencies = receipt["depends_on_record_ids"]
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) or not item.strip() for item in dependencies
        ):
            raise ArtifactVectorExportError("dependency receipt dependency IDs are invalid")
        if dependencies != sorted(set(dependencies)):
            raise ArtifactVectorExportError(
                "dependency receipt dependency IDs must be unique and sorted"
            )
        if (
            receipt["align_revision_id"] != expected_align_id
            or receipt["geometry_revision_id"] != expected_geometry_id
        ):
            raise ArtifactVectorExportError(
                "dependency receipt coordinate context does not match the root record"
            )
        if (
            receipt["lifecycle_status"] != RecordLifecycleStatus.READY.value
            or receipt["freshness"] != RecordFreshness.FRESH.value
        ):
            raise ArtifactVectorExportError("dependency receipt is not READY + FRESH")
        receipts[receipt_id] = receipt

    root_id = str(root_record["id"])
    graph: dict[str, list[str]] = {
        root_id: list(root_record["depends_on_record_ids"]),
        **{
            record_id: list(receipt["depends_on_record_ids"])
            for record_id, receipt in receipts.items()
        },
    }
    state: dict[str, int] = {}
    reachable: set[str] = set()

    def visit(record_id: str) -> None:
        color = state.get(record_id, 0)
        if color == 1:
            raise ArtifactVectorExportError("dependency closure contains a cycle")
        if color == 2:
            return
        if record_id not in graph:
            raise ArtifactVectorExportError(
                f"dependency closure is missing receipt {record_id!r}"
            )
        state[record_id] = 1
        if record_id != root_id:
            reachable.add(record_id)
        for dependency_id in graph[record_id]:
            visit(dependency_id)
        state[record_id] = 2

    visit(root_id)
    if reachable != set(receipts):
        raise ArtifactVectorExportError("dependency closure contains unreachable receipts")


def _validate_provenance_shape(value: object) -> Mapping[str, Any]:
    provenance = _exact_keys(
        value,
        {
            "align_revision",
            "dependency_closure",
            "document",
            "geometry_revision",
            "record",
            "source_assets",
            "source_metadata_revision",
        },
        model_name="provenance",
    )
    document = _exact_keys(
        provenance["document"],
        {
            "active_align_revision_id",
            "active_source_metadata_revision_id",
            "document_id",
            "manifest_sha256",
            "schema_version",
            "software_version",
        },
        model_name="provenance.document",
    )
    manifest_sha = document["manifest_sha256"]
    if not isinstance(manifest_sha, str) or _SHA256_RE.fullmatch(manifest_sha) is None:
        raise ArtifactVectorExportError("document manifest_sha256 is invalid")
    for key in (
        "active_align_revision_id",
        "active_source_metadata_revision_id",
        "document_id",
        "schema_version",
        "software_version",
    ):
        _required_string(document[key], field_name=f"provenance.document.{key}")
    if document["schema_version"] != ARTIFACT_DOCUMENT_SCHEMA_VERSION:
        raise ArtifactVectorExportError("provenance document schema is unsupported")
    record = _exact_keys(
        provenance["record"],
        {
            "align_revision_id",
            "created_at",
            "depends_on_record_ids",
            "geometry_revision_id",
            "geometry_ref",
            "id",
            "lifecycle_status",
            "operator",
            "recipe_hash",
            "selection_hash",
            "type",
        },
        model_name="provenance.record",
    )
    recipe_hash = record["recipe_hash"]
    if not isinstance(recipe_hash, str) or _SHA256_RE.fullmatch(recipe_hash) is None:
        raise ArtifactVectorExportError("record recipe_hash is invalid")
    for key in (
        "align_revision_id",
        "geometry_revision_id",
        "geometry_ref",
        "id",
        "operator",
        "type",
    ):
        _required_string(record[key], field_name=f"provenance.record.{key}")
    created_at = record["created_at"]
    if not isinstance(created_at, str) or _UTC_SECONDS_RE.fullmatch(created_at) is None:
        raise ArtifactVectorExportError("provenance record created_at is invalid")
    selection_hash = record["selection_hash"]
    if selection_hash is not None and (
        not isinstance(selection_hash, str)
        or _SHA256_RE.fullmatch(selection_hash) is None
    ):
        raise ArtifactVectorExportError("provenance record selection_hash is invalid")
    root_dependencies = record["depends_on_record_ids"]
    if not isinstance(root_dependencies, list) or any(
        not isinstance(item, str) or not item.strip() for item in root_dependencies
    ):
        raise ArtifactVectorExportError(
            "provenance record depends_on_record_ids is invalid"
        )
    if root_dependencies != sorted(set(root_dependencies)):
        raise ArtifactVectorExportError(
            "provenance record dependency IDs must be unique and sorted"
        )
    try:
        align = AlignRevision.from_dict(
            dict(_exact_keys(
                provenance["align_revision"],
                {
                    "id",
                    "parent_id",
                    "source_metadata_revision_id",
                    "matrix4x4",
                    "recipe",
                    "qc",
                    "created_at",
                    "operator",
                },
                model_name="provenance.align_revision",
            ))
            | {"extensions": {}}
        )
        metadata = SourceMetadataRevision.from_dict(
            dict(_exact_keys(
                provenance["source_metadata_revision"],
                {
                    "id",
                    "parent_id",
                    "geometry_revision_id",
                    "unit",
                    "axes",
                    "handedness",
                    "confirmation_status",
                    "source_to_canonical_mm",
                    "created_at",
                    "operator",
                },
                model_name="provenance.source_metadata_revision",
            ))
            | {"extensions": {}}
        )
        geometry = GeometryRevision.from_dict(
            dict(_exact_keys(
                provenance["geometry_revision"],
                {
                    "id",
                    "source_asset_ids",
                    "geometry_sha256",
                    "geometry_hash_scope",
                    "import_recipe",
                    "qc",
                    "created_at",
                    "operator",
                },
                model_name="provenance.geometry_revision",
            ))
            | {"topology_map_ref": None, "extensions": {}}
        )
    except ArtifactDocumentError as exc:
        raise ArtifactVectorExportError(f"invalid revision provenance: {exc}") from exc
    public_import_recipe = _public_mesh_import_recipe(
        geometry.import_recipe,
        require_current_runtime=False,
    )
    if dict(geometry.import_recipe) != public_import_recipe:
        raise ArtifactVectorExportError(
            "provenance geometry import_recipe contains non-public fields"
        )
    try:
        metadata.require_confirmed_matrix()
    except ArtifactDocumentError as exc:
        raise ArtifactVectorExportError(
            f"1:1 export requires confirmed source metadata: {exc}"
        ) from exc
    if record.get("id") is None:
        raise ArtifactVectorExportError("provenance record id is invalid")
    if record.get("lifecycle_status") != RecordLifecycleStatus.READY.value:
        raise ArtifactVectorExportError("provenance record is not READY")
    if record.get("geometry_ref") is None:
        raise ArtifactVectorExportError("provenance record geometry_ref is invalid")
    if align.id != record["align_revision_id"]:
        raise ArtifactVectorExportError("record and Align provenance do not match")
    if align.source_metadata_revision_id != metadata.id:
        raise ArtifactVectorExportError("Align and metadata provenance do not match")
    if metadata.geometry_revision_id != geometry.id:
        raise ArtifactVectorExportError("metadata and geometry provenance do not match")
    if geometry.id != record["geometry_revision_id"]:
        raise ArtifactVectorExportError("record and geometry provenance do not match")
    if document["active_align_revision_id"] != align.id:
        raise ArtifactVectorExportError(
            "export-time active Align does not match the record"
        )
    if document["active_source_metadata_revision_id"] != metadata.id:
        raise ArtifactVectorExportError(
            "export-time active metadata does not match the record"
        )
    _validate_dependency_closure(
        provenance["dependency_closure"],
        root_record=record,
        expected_align_id=align.id,
        expected_geometry_id=geometry.id,
    )
    assets = provenance["source_assets"]
    if not isinstance(assets, list) or not assets:
        raise ArtifactVectorExportError("provenance.source_assets must be non-empty")
    for index, asset in enumerate(assets):
        item = _exact_keys(
            asset,
            {
                "id",
                "identity_scope",
                "media_type",
                "original_name",
                "role",
                "sha256",
                "size_bytes",
            },
            model_name=f"provenance.source_assets[{index}]",
        )
        digest = item["sha256"]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ArtifactVectorExportError("source asset SHA-256 is invalid")
        size = item["size_bytes"]
        if type(size) is not int or size < 0:
            raise ArtifactVectorExportError("source asset size_bytes is invalid")
        if item["id"] != f"sha256:{digest}":
            raise ArtifactVectorExportError("source asset ID does not match its SHA-256")
        for key in (
            "id",
            "identity_scope",
            "media_type",
            "original_name",
            "role",
        ):
            _required_string(
                item[key],
                field_name=f"provenance.source_assets[{index}].{key}",
            )
        if item["role"] != PRIMARY_SOURCE_ASSET_ROLE:
            raise ArtifactVectorExportError("source asset role is unsupported")
        if item["identity_scope"] != PRIMARY_FILE_IDENTITY_SCOPE:
            raise ArtifactVectorExportError("source asset identity scope is unsupported")
    if tuple(geometry.source_asset_ids) != tuple(
        sorted(str(asset["id"]) for asset in assets)
    ):
        raise ArtifactVectorExportError("geometry and source asset provenance do not match")
    return provenance


def validate_public_export_provenance(value: object) -> Mapping[str, Any]:
    """Validate the shared public provenance shape without an artifact type."""

    return _validate_provenance_shape(value)


def _validate_qc(
    value: object,
    *,
    payload: VectorGeometryPayload,
) -> Mapping[str, Any]:
    qc = _exact_keys(
        value,
        {"export_gate", "payload", "record", "scale"},
        model_name="qc",
    )
    gate = _exact_keys(
        qc["export_gate"],
        {"payload_verified", "record_freshness", "record_lifecycle_status"},
        model_name="qc.export_gate",
    )
    if gate != {
        "payload_verified": True,
        "record_freshness": "fresh",
        "record_lifecycle_status": "ready",
    }:
        raise ArtifactVectorExportError("export gate does not prove READY + FRESH payload")
    payload_qc = qc["payload"]
    if payload_qc != payload.qc_summary():
        raise ArtifactVectorExportError("sidecar payload QC does not match vector payload")
    record_qc = qc["record"]
    if not isinstance(record_qc, Mapping):
        raise ArtifactVectorExportError("qc.record must be an object")
    for key, expected in payload.qc_summary().items():
        if record_qc.get(key) != expected:
            raise ArtifactVectorExportError(
                f"sidecar record QC field {key!r} does not match payload"
            )
    scale = _exact_keys(
        qc["scale"],
        {"physical_scale", "unit", "viewbox_matches_physical_dimensions"},
        model_name="qc.scale",
    )
    if scale != {
        "physical_scale": "1:1",
        "unit": "mm",
        "viewbox_matches_physical_dimensions": True,
    }:
        raise ArtifactVectorExportError("scale QC is invalid")
    return qc


def _validate_svg_xml(
    svg_bytes: bytes,
    *,
    expected_metadata: Mapping[str, Any],
) -> None:
    upper = svg_bytes.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ArtifactVectorExportError("SVG must not contain a DTD or entity declaration")
    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError as exc:
        raise ArtifactVectorExportError(f"SVG XML is invalid: {exc}") from exc
    if root.tag != f"{{{_SVG_NAMESPACE}}}svg":
        raise ArtifactVectorExportError("SVG root namespace is invalid")
    forbidden_tags = {
        f"{{{_SVG_NAMESPACE}}}script",
        f"{{{_SVG_NAMESPACE}}}image",
        f"{{{_SVG_NAMESPACE}}}foreignObject",
        f"{{{_SVG_NAMESPACE}}}use",
    }
    for element in root.iter():
        if element.tag in forbidden_tags:
            raise ArtifactVectorExportError("SVG contains a forbidden external/active element")
        for attribute in element.attrib:
            local = attribute.rsplit("}", 1)[-1].lower()
            if local == "href" or local.startswith("on") or local == "style":
                raise ArtifactVectorExportError("SVG contains a forbidden attribute")
    metadata_nodes = root.findall(f"{{{_SVG_NAMESPACE}}}metadata")
    if len(metadata_nodes) != 1 or metadata_nodes[0].text is None:
        raise ArtifactVectorExportError("SVG must contain exactly one provenance metadata node")
    embedded = _strict_json_bytes(
        metadata_nodes[0].text.encode("utf-8"),
        label="SVG metadata",
    )
    if embedded != expected_metadata:
        raise ArtifactVectorExportError("SVG metadata does not match the sidecar")


def validate_vector_export_bytes(
    svg_bytes: bytes,
    sidecar_bytes: bytes,
    *,
    document: ArtifactDocument | None = None,
) -> VectorExportBundle:
    """Validate package bytes without network, original mesh, or GUI state."""

    if not isinstance(svg_bytes, bytes) or not isinstance(sidecar_bytes, bytes):
        raise ArtifactVectorExportError("SVG and sidecar payloads must be bytes")
    if not svg_bytes or len(svg_bytes) > MAX_VECTOR_EXPORT_SVG_BYTES:
        raise ArtifactVectorExportError("SVG byte length is outside the safety limit")
    if not sidecar_bytes or len(sidecar_bytes) > MAX_VECTOR_EXPORT_SIDECAR_BYTES:
        raise ArtifactVectorExportError("sidecar byte length is outside the safety limit")
    sidecar = _strict_json_bytes(sidecar_bytes, label="vector export sidecar")
    _exact_keys(
        sidecar,
        {
            "artifact",
            "format",
            "presentation",
            "provenance",
            "qc",
            "recipe",
            "schema_version",
            "vector_payload",
            "vector_payload_sha256",
        },
        model_name="vector export sidecar",
    )
    if sidecar["format"] != VECTOR_EXPORT_FORMAT:
        raise ArtifactVectorExportError("vector export format is invalid")
    if sidecar["schema_version"] != VECTOR_EXPORT_SCHEMA_VERSION:
        raise ArtifactVectorExportError("vector export schema version is invalid")
    artifact = _exact_keys(
        sidecar["artifact"],
        {"file", "media_type", "sha256", "size_bytes"},
        model_name="artifact",
    )
    if artifact["file"] != VECTOR_EXPORT_SVG_NAME:
        raise ArtifactVectorExportError("artifact file name is not canonical")
    if artifact["media_type"] != VECTOR_EXPORT_SVG_MEDIA_TYPE:
        raise ArtifactVectorExportError("artifact media type is invalid")
    if type(artifact["size_bytes"]) is not int or artifact["size_bytes"] != len(svg_bytes):
        raise ArtifactVectorExportError("SVG byte length does not match the sidecar")
    svg_sha256 = _sha256_bytes(svg_bytes)
    if artifact["sha256"] != svg_sha256:
        raise ArtifactVectorExportError("SVG SHA-256 does not match the sidecar")

    payload = _validated_sidecar_payload(sidecar)
    provenance = _validate_provenance_shape(sidecar["provenance"])
    record_provenance = provenance["record"]
    assert isinstance(record_provenance, Mapping)
    if record_provenance["geometry_ref"] != payload.geometry_ref:
        raise ArtifactVectorExportError(
            "record geometry_ref does not match the vector payload"
        )
    if record_provenance["type"] != VectorRecordKind(payload.kind).record_type:
        raise ArtifactVectorExportError("record type does not match the vector payload")
    recipe = sidecar["recipe"]
    if not isinstance(recipe, Mapping):
        raise ArtifactVectorExportError("recipe must be an object")
    try:
        validate_vector_recipe(
            recipe,
            expected_kind=VectorRecordKind(payload.kind),
        )
    except ArtifactVectorRecordError as exc:
        raise ArtifactVectorExportError(str(exc)) from exc
    try:
        recipe_hash = canonical_recipe_hash(recipe)
    except ArtifactDocumentError as exc:
        raise ArtifactVectorExportError(str(exc)) from exc
    if record_provenance["recipe_hash"] != recipe_hash:
        raise ArtifactVectorExportError("recipe hash does not match the sidecar recipe")
    _validate_qc(sidecar["qc"], payload=payload)
    claims_sha256 = _sidecar_claims_sha256(sidecar)

    options, presentation = _options_from_presentation(sidecar["presentation"])
    expected_svg, expected_bounds, expected_width, expected_height = _render_svg(
        payload,
        options=options,
        provenance=provenance,
        sidecar_claims_sha256=claims_sha256,
    )
    if presentation["content_bounds_mm"] != list(expected_bounds):
        raise ArtifactVectorExportError("presentation bounds do not match vector payload")
    width = _finite_number(presentation["width_mm"], field_name="presentation.width_mm")
    height = _finite_number(presentation["height_mm"], field_name="presentation.height_mm")
    if width != expected_width or height != expected_height:
        raise ArtifactVectorExportError("presentation dimensions do not match vector payload")
    expected_view_box = [
        "0",
        "0",
        _number_token(width, field_name="presentation.width_mm"),
        _number_token(height, field_name="presentation.height_mm"),
    ]
    if presentation["view_box"] != expected_view_box:
        raise ArtifactVectorExportError(
            "SVG physical dimensions and viewBox do not share exact tokens"
        )
    metadata = _svg_metadata(
        provenance=provenance,
        payload_sha256=payload.sha256,
        sidecar_claims_sha256=claims_sha256,
        width_mm=width,
        height_mm=height,
    )
    if svg_bytes != expected_svg:
        raise ArtifactVectorExportError("SVG bytes are not the canonical payload derivative")
    _validate_svg_xml(svg_bytes, expected_metadata=metadata)

    if document is not None:
        record_id = record_provenance["id"]
        if not isinstance(record_id, str):
            raise ArtifactVectorExportError("provenance record id is invalid")
        record, document_payload, record_qc = _require_exportable_record(
            document,
            record_id,
        )
        if document_payload.sha256 != payload.sha256:
            raise ArtifactVectorExportError("export payload does not match the document record")
        if provenance != _provenance(document, record):
            raise ArtifactVectorExportError("export provenance does not match the document")
        if dict(recipe) != record.to_dict()["recipe"]:
            raise ArtifactVectorExportError("export recipe does not match the document record")
        qc = sidecar["qc"]
        assert isinstance(qc, Mapping)
        if qc["record"] != record_qc:
            raise ArtifactVectorExportError("export QC does not match the document record")

    return VectorExportBundle(
        svg_bytes=svg_bytes,
        sidecar_bytes=sidecar_bytes,
        svg_sha256=svg_sha256,
        sidecar_sha256=_sha256_bytes(sidecar_bytes),
        vector_payload_sha256=payload.sha256,
        width_mm=width,
        height_mm=height,
    )


def validate_vector_export_package(
    directory: str | os.PathLike[str],
    *,
    document: ArtifactDocument | None = None,
) -> VectorExportBundle:
    """Validate an exact two-file, relocatable vector export directory."""

    path = Path(directory)
    if path.is_symlink() or not path.is_dir():
        raise ArtifactVectorExportError("vector export package must be a real directory")
    entries = sorted(path.iterdir(), key=lambda item: item.name)
    normative_entries = [
        entry for entry in entries if entry.name not in _IGNORABLE_OS_METADATA_NAMES
    ]
    ignored_entries = [
        entry for entry in entries if entry.name in _IGNORABLE_OS_METADATA_NAMES
    ]
    if [entry.name for entry in normative_entries] != sorted(
        [VECTOR_EXPORT_SIDECAR_NAME, VECTOR_EXPORT_SVG_NAME]
    ):
        raise ArtifactVectorExportError(
            "vector export package must contain exactly two normative files"
        )
    for entry in normative_entries:
        if entry.is_symlink() or not entry.is_file():
            raise ArtifactVectorExportError("vector export members must be regular files")
    for entry in ignored_entries:
        if (
            entry.is_symlink()
            or not entry.is_file()
            or entry.stat().st_size > MAX_IGNORABLE_OS_METADATA_BYTES
        ):
            raise ArtifactVectorExportError("OS metadata entry is unsafe")
    return validate_vector_export_bytes(
        _read_bounded_file(
            path / VECTOR_EXPORT_SVG_NAME,
            limit=MAX_VECTOR_EXPORT_SVG_BYTES,
            label="SVG",
        ),
        _read_bounded_file(
            path / VECTOR_EXPORT_SIDECAR_NAME,
            limit=MAX_VECTOR_EXPORT_SIDECAR_BYTES,
            label="sidecar",
        ),
        document=document,
    )


def _read_bounded_file(path: Path, *, limit: int, label: str) -> bytes:
    try:
        declared_size = path.stat().st_size
    except OSError as exc:
        raise ArtifactVectorExportError(f"cannot inspect {label} file: {exc}") from exc
    if declared_size <= 0 or declared_size > limit:
        raise ArtifactVectorExportError(f"{label} byte length is outside the safety limit")
    try:
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
    except OSError as exc:
        raise ArtifactVectorExportError(f"cannot read {label} file: {exc}") from exc
    if len(payload) != declared_size or len(payload) > limit:
        raise ArtifactVectorExportError(
            f"{label} file changed while being read or exceeds the safety limit"
        )
    return payload


def read_bounded_export_file(path: Path, *, limit: int, label: str) -> bytes:
    return _read_bounded_file(path, limit=limit, label=label)


def _fsync_parent(path: Path) -> bool:
    descriptor: int | None = None
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    try:
        descriptor = os.open(path, flags)
        os.fsync(descriptor)
    except (AttributeError, NotImplementedError):
        return False
    except OSError as exc:
        unsupported_errnos = {
            errno.EACCES,
            errno.EBADF,
            errno.EINVAL,
            getattr(errno, "ENOSYS", errno.EINVAL),
            errno.EPERM,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if exc.errno in unsupported_errnos:
            return False
        raise
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    return True


def _write_new_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_new_export_file(path: Path, payload: bytes) -> None:
    _write_new_file(path, payload)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory while refusing an existing destination."""

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    result: int
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise ArtifactVectorExportError(
                "atomic non-overwriting directory publish is unavailable"
            )
        # AT_FDCWD=-100, RENAME_NOREPLACE=1.
        result = int(renameat2(-100, source_bytes, -100, destination_bytes, 1))
    elif sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise ArtifactVectorExportError(
                "atomic non-overwriting directory publish is unavailable"
            )
        # Darwin RENAME_EXCL=0x00000004.
        result = int(renamex_np(source_bytes, destination_bytes, 0x00000004))
    elif os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise ArtifactVectorExportError("export destination already exists") from exc
        except OSError as exc:
            raise ArtifactVectorExportError(
                f"cannot atomically publish vector export: {exc}"
            ) from exc
        return
    else:
        raise ArtifactVectorExportError(
            "atomic non-overwriting directory publish is unsupported on this platform"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ArtifactVectorExportError("export destination already exists")
    raise ArtifactVectorExportError(
        f"cannot atomically publish vector export: {os.strerror(error_number)}"
    )


# Cleanup uses a distinct reference so publication-race instrumentation cannot
# accidentally intercept the quarantine move as a second publication attempt.
_quarantine_directory_noreplace = _rename_directory_noreplace


def publish_export_directory_noreplace(source: Path, destination: Path) -> None:
    _rename_directory_noreplace(source, destination)


def fsync_export_directory(path: Path) -> bool:
    return _fsync_parent(path)


def _validate_vector_destination(directory: str | os.PathLike[str]) -> Path:
    destination = Path(
        os.path.abspath(os.fspath(Path(directory).expanduser()))
    )
    if not destination.name.endswith(VECTOR_EXPORT_DIRECTORY_SUFFIX):
        raise ArtifactVectorExportError(
            f"export directory must end with {VECTOR_EXPORT_DIRECTORY_SUFFIX}"
        )
    return destination


def _absolute_staging_path(directory: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(Path(directory).expanduser())))


def _uuid_hex() -> str:
    token = uuid.uuid4().hex.lower()
    if _UUID_HEX_RE.fullmatch(token) is None:
        raise ArtifactVectorExportError("UUID provider returned an invalid staging token")
    return token


def _real_directory_identity(path: Path, *, label: str) -> os.stat_result:
    try:
        identity = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactVectorExportError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISDIR(identity.st_mode):
        raise ArtifactVectorExportError(f"{label} must be a real directory")
    return identity


def _path_exists_without_following(path: Path) -> bool:
    try:
        path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ArtifactVectorExportError(f"cannot inspect export path: {exc}") from exc
    return True


def _fingerprint_entry(path: Path, *, name: str) -> _ExportEntryFingerprint:
    try:
        identity = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactVectorExportError(
            f"cannot fingerprint vector export member {name!r}: {exc}"
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


def _capture_vector_package_fingerprint(
    staging: Path,
) -> tuple[_ExportEntryFingerprint, ...]:
    directory = _fingerprint_entry(staging, name=".")
    if not stat.S_ISDIR(directory.mode):
        raise ArtifactVectorExportError(
            "vector export staging path is not a real directory"
        )
    try:
        entries = sorted(staging.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise ArtifactVectorExportError(
            f"cannot enumerate vector export staging directory: {exc}"
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
        raise ArtifactVectorExportError(
            "vector export staging inode is already visible at the destination; "
            "publication occurred outside the authorized commit",
            committed=True,
        )


def _require_current_parent(staging: _OwnedStagingDirectory) -> None:
    parent = _real_directory_identity(
        staging.destination.parent,
        label="vector export destination parent",
    )
    if (parent.st_dev, parent.st_ino) != (
        staging.parent_device,
        staging.parent_inode,
    ):
        raise ArtifactVectorExportError(
            "vector export destination parent was replaced"
        )


def _require_owned_staging_identity(staging: _OwnedStagingDirectory) -> None:
    try:
        current = staging.path.stat(follow_symlinks=False)
    except FileNotFoundError:
        _raise_if_owned_destination_is_visible(staging)
        raise ArtifactVectorExportError(
            "owned vector export staging directory is missing"
        ) from None
    except OSError as exc:
        raise ArtifactVectorExportError(
            f"cannot inspect vector export staging directory: {exc}"
        ) from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != (staging.device, staging.inode)
    ):
        raise ArtifactVectorExportError(
            "vector export staging directory was replaced"
        )


def _invalidate_vector_prepared_locked(staging: _OwnedStagingDirectory) -> None:
    for nonce, prepared in tuple(_PREPARED_PUBLICATIONS.items()):
        if prepared._owned is staging:
            _PREPARED_PUBLICATIONS.pop(nonce, None)


def _create_owned_vector_staging_directory(
    destination: Path,
) -> _OwnedStagingDirectory:
    parent = _real_directory_identity(
        destination.parent,
        label="vector export destination parent",
    )
    for _attempt in range(_MAX_STAGING_DIRECTORY_ATTEMPTS):
        candidate = destination.parent / f"{_VECTOR_STAGING_PREFIX}{_uuid_hex()}"
        try:
            # Respect the user's umask instead of forcing a private 0700
            # directory. Shared-lab exports remain readable while restrictive
            # umasks still win.
            candidate.mkdir(mode=0o777)
        except FileExistsError:
            # A colliding path belongs to somebody else. Never inspect, reuse,
            # or remove it; reserve a fresh UUID-derived name instead.
            continue
        except OSError as exc:
            raise ArtifactVectorExportError(
                f"cannot create vector export staging directory: {exc}"
            ) from exc
        try:
            identity = candidate.stat(follow_symlinks=False)
        except OSError as exc:
            raise ArtifactVectorExportError(
                f"cannot inspect vector export staging directory: {exc}"
            ) from exc
        if not stat.S_ISDIR(identity.st_mode):
            raise ArtifactVectorExportError(
                "vector export staging path is not a real directory"
            )
        return _OwnedStagingDirectory(
            path=candidate,
            destination=destination,
            device=identity.st_dev,
            inode=identity.st_ino,
            parent_device=parent.st_dev,
            parent_inode=parent.st_ino,
        )
    raise ArtifactVectorExportError(
        "cannot reserve vector export staging directory after 16 attempts"
    )


def _empty_vector_directory_fd(directory_fd: int) -> None:
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
                raise ArtifactVectorExportError(
                    "vector export child directory changed during cleanup"
                )
            _empty_vector_directory_fd(child_fd)
            current_name = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if not os.path.samestat(opened_identity, current_name):
                raise ArtifactVectorExportError(
                    "vector export child directory was replaced during cleanup"
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


def _discard_owned_vector_staging_directory(
    staging: _OwnedStagingDirectory,
) -> bool:
    """Quarantine by rename before inspecting or recursively deleting a name."""

    _require_current_parent(staging)
    quarantine: Path | None = None
    for _attempt in range(_MAX_STAGING_DIRECTORY_ATTEMPTS):
        candidate = staging.path.parent / (
            f"{_VECTOR_QUARANTINE_PREFIX}{_uuid_hex()}"
        )
        try:
            _quarantine_directory_noreplace(staging.path, candidate)
        except ArtifactVectorExportError as exc:
            if _path_exists_without_following(staging.path):
                if "already exists" in str(exc):
                    continue
                raise ArtifactVectorExportError(
                    f"cannot quarantine vector export staging directory: {exc}"
                ) from exc
            _raise_if_owned_destination_is_visible(staging)
            return False
        quarantine = candidate
        break
    if quarantine is None:
        raise ArtifactVectorExportError(
            "cannot reserve vector export discard quarantine after 16 attempts"
        )

    if not _descriptor_cleanup_available():
        if _windows_cleanup_fallback_required():
            # Windows has no descriptor-relative directory deletion in the
            # Python standard library. The unpredictable same-parent
            # quarantine still prevents deletion through the caller-visible
            # staging name; verify the inode once more immediately before the
            # best-available recursive removal.
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
                    raise ArtifactVectorExportError(
                        "owned vector export was quarantined, but Windows cleanup "
                        f"is not proven: {exc}"
                    ) from exc
                return not _path_exists_without_following(quarantine)
        try:
            _quarantine_directory_noreplace(quarantine, staging.path)
        except ArtifactVectorExportError:
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
        # A foreign replacement was atomically moved out of the public staging
        # name. Restore it without overwriting any concurrent claimant.
        try:
            _rename_directory_noreplace(quarantine, staging.path)
        except ArtifactVectorExportError:
            pass
        return False

    try:
        _empty_vector_directory_fd(quarantine_descriptor)
        current_name = os.stat(
            quarantine.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not os.path.samestat(quarantined, current_name):
            try:
                _quarantine_directory_noreplace(quarantine, staging.path)
            except ArtifactVectorExportError:
                pass
            return False
        os.rmdir(quarantine.name, dir_fd=parent_descriptor)
    except (NotImplementedError, OSError, TypeError) as exc:
        raise ArtifactVectorExportError(
            "owned vector export was quarantined, but cleanup is not proven: "
            f"{exc}"
        ) from exc
    finally:
        os.close(quarantine_descriptor)
        os.close(parent_descriptor)
    if _path_exists_without_following(quarantine):
        raise ArtifactVectorExportError(
            "owned vector export quarantine still exists; cleanup is not proven"
        )
    return True


def _stage_vector_package_owned(
    directory: str | os.PathLike[str],
    document: ArtifactDocument,
    record_id: str,
    *,
    options: VectorSVGOptions | None = None,
) -> _OwnedStagingDirectory:
    destination = _validate_vector_destination(directory)
    if destination.exists() or destination.is_symlink():
        raise ArtifactVectorExportError("export destination already exists")

    # Validate the authoritative record and build deterministic bytes before
    # creating even the destination parent. Invalid work must have no filesystem
    # side effect.
    bundle = build_vector_export(document, record_id, options=options)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ArtifactVectorExportError(
            f"cannot create vector export parent directory: {exc}"
        ) from exc

    staging = _create_owned_vector_staging_directory(destination)
    _register_vector_staging(staging)
    try:
        _write_new_file(staging.path / VECTOR_EXPORT_SVG_NAME, bundle.svg_bytes)
        _write_new_file(
            staging.path / VECTOR_EXPORT_SIDECAR_NAME,
            bundle.sidecar_bytes,
        )
        staging = replace(
            staging,
            staging_directory_fsync_confirmed=_fsync_parent(staging.path),
        )
        with _STAGING_OWNERS_LOCK:
            _STAGING_OWNERS[_staging_registry_key(staging.path)] = staging
        validate_vector_export_package(staging.path, document=document)
    except Exception as exc:
        try:
            discarded = _discard_owned_vector_staging_directory(staging)
        except Exception as cleanup_exc:
            raise ArtifactVectorExportError(
                "vector export staging failed and cleanup is not proven"
            ) from cleanup_exc
        finally:
            _forget_vector_staging(staging.path)
        if not discarded:
            raise ArtifactVectorExportError(
                "vector export staging failed and cleanup is not proven"
            ) from exc
        raise
    return staging


def stage_vector_package(
    directory: str | os.PathLike[str],
    document: ArtifactDocument,
    record_id: str,
    *,
    options: VectorSVGOptions | None = None,
) -> Path:
    """Create and verify a same-parent package without publishing it.

    Ownership of the returned staging directory transfers to the caller. A
    later publication failure intentionally leaves it available to that caller;
    the compatibility wrapper cleans only staging directories it created.
    """

    staging = _stage_vector_package_owned(
        directory,
        document,
        record_id,
        options=options,
    )
    return staging.path


def discard_staged_vector_package(
    staging_directory: str | os.PathLike[str],
    directory: str | os.PathLike[str],
) -> bool:
    """Delete only a staging directory created by this process and API call.

    ``False`` means cleanup could not be proven. Missing owned paths are not
    treated as success, and a staging inode already visible at the destination
    raises a committed-effect error.
    """

    destination = _validate_vector_destination(directory)
    staging = _absolute_staging_path(staging_directory)
    if (
        staging.parent != destination.parent
        or not staging.name.startswith(_VECTOR_STAGING_PREFIX)
    ):
        return False
    key = _staging_registry_key(staging)
    with _STAGING_OWNERS_LOCK:
        owned = _STAGING_OWNERS.get(key)
        if owned is None or owned.destination != destination:
            return False
        try:
            discarded = _discard_owned_vector_staging_directory(owned)
        finally:
            _STAGING_OWNERS.pop(key, None)
            _invalidate_vector_prepared_locked(owned)
        return discarded


def prepare_staged_vector_publication(
    staging_directory: str | os.PathLike[str],
    directory: str | os.PathLike[str],
    *,
    document: ArtifactDocument | None = None,
) -> PreparedVectorPublication:
    """Fully validate one owned staging inode and mint an exact capability.

    Call this outside the application's final authority lock. Publication via
    :func:`publish_prepared_vector_package` then performs only identity and
    stat-fingerprint rechecks plus the no-replace rename.
    """

    destination = _validate_vector_destination(directory)
    staging = _absolute_staging_path(staging_directory)
    key = _staging_registry_key(staging)
    with _STAGING_OWNERS_LOCK:
        owned = _STAGING_OWNERS.get(key)
        if owned is None:
            raise ArtifactVectorExportError(
                "vector export staging directory was not created by this process"
            )
        if owned.destination != destination:
            raise ArtifactVectorExportError(
                "vector export staging authority is bound to a different destination"
            )

    _require_current_parent(owned)
    _require_owned_staging_identity(owned)
    if _path_exists_without_following(destination):
        _raise_if_owned_destination_is_visible(owned)
        raise ArtifactVectorExportError("export destination already exists")
    before = _capture_vector_package_fingerprint(staging)
    validate_vector_export_package(staging, document=document)
    after = _capture_vector_package_fingerprint(staging)
    if before != after:
        raise ArtifactVectorExportError(
            "vector export staging package changed while being validated"
        )

    nonce = object()
    prepared = PreparedVectorPublication(
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
            raise ArtifactVectorExportError(
                "vector export staging authority changed while being validated"
            )
        _require_current_parent(owned)
        _require_owned_staging_identity(owned)
        if _capture_vector_package_fingerprint(staging) != after:
            raise ArtifactVectorExportError(
                "vector export staging package changed after validation"
            )
        _PREPARED_PUBLICATIONS[nonce] = prepared
    return prepared


def discard_prepared_vector_package(
    prepared: PreparedVectorPublication,
) -> bool:
    """Discard only the inode authorized by the exact prepared capability."""

    if not isinstance(prepared, PreparedVectorPublication):
        raise ArtifactVectorExportError(
            "prepared publication must be a PreparedVectorPublication"
        )
    with _STAGING_OWNERS_LOCK:
        if _PREPARED_PUBLICATIONS.get(prepared._nonce) is not prepared:
            _raise_if_owned_destination_is_visible(prepared._owned)
            return False
    return discard_staged_vector_package(
        prepared.staging_directory,
        prepared.destination,
    )


def publish_prepared_vector_package(
    prepared: PreparedVectorPublication,
) -> Path:
    """Fast final commit for an exact, fully validated vector capability."""

    if not isinstance(prepared, PreparedVectorPublication):
        raise ArtifactVectorExportError(
            "prepared publication must be a PreparedVectorPublication"
        )
    owned = prepared._owned
    key = _staging_registry_key(prepared.staging_directory)
    with _STAGING_OWNERS_LOCK:
        if _PREPARED_PUBLICATIONS.get(prepared._nonce) is not prepared:
            _raise_if_owned_destination_is_visible(owned)
            raise ArtifactVectorExportError(
                "prepared vector publication capability is invalid or consumed"
            )
        if _STAGING_OWNERS.get(key) is not owned:
            _raise_if_owned_destination_is_visible(owned)
            raise ArtifactVectorExportError(
                "vector export staging authority is no longer current"
            )
        _require_current_parent(owned)
        _require_owned_staging_identity(owned)
        if _path_exists_without_following(prepared.destination):
            _raise_if_owned_destination_is_visible(owned)
            raise ArtifactVectorExportError("export destination already exists")
        if (
            _capture_vector_package_fingerprint(prepared.staging_directory)
            != prepared._fingerprint
        ):
            raise ArtifactVectorExportError(
                "vector export staging package changed after preparation"
            )
        _rename_directory_noreplace(
            prepared.staging_directory,
            prepared.destination,
        )
        try:
            published_identity = prepared.destination.stat(follow_symlinks=False)
        except OSError as exc:
            raise ArtifactVectorExportError(
                "vector export was renamed, but destination identity could not be "
                f"verified: {exc}",
                committed=True,
            ) from exc
        if (
            not stat.S_ISDIR(published_identity.st_mode)
            or (published_identity.st_dev, published_identity.st_ino)
            != (owned.device, owned.inode)
        ):
            raise ArtifactVectorExportError(
                "vector export was renamed, but destination inode is not the "
                "authorized staging inode",
                committed=True,
            )
        _STAGING_OWNERS.pop(key, None)
        _invalidate_vector_prepared_locked(owned)
    try:
        parent_fsync_confirmed = _fsync_parent(prepared.destination.parent)
    except OSError as exc:
        raise ArtifactVectorExportError(
            "vector export was atomically published, but directory fsync failed; "
            f"crash durability is uncertain: {exc}",
            committed=True,
        ) from exc
    if (
        not prepared._staging_directory_fsync_confirmed
        or not parent_fsync_confirmed
    ):
        raise ArtifactVectorExportError(
            "vector export was atomically published, but directory fsync is "
            "unsupported; crash durability is uncertain",
            committed=True,
        )
    return prepared.destination


def publish_staged_vector_package(
    staging_directory: str | os.PathLike[str],
    directory: str | os.PathLike[str],
    *,
    document: ArtifactDocument | None = None,
) -> Path:
    """Compatibility wrapper: prepare fully, then commit the exact capability."""

    prepared = prepare_staged_vector_publication(
        staging_directory,
        directory,
        document=document,
    )
    return publish_prepared_vector_package(prepared)


def export_vector_package(
    directory: str | os.PathLike[str],
    document: ArtifactDocument,
    record_id: str,
    *,
    options: VectorSVGOptions | None = None,
) -> Path:
    """Stage and atomically publish a new ``*.amr-vector`` package."""

    staging = stage_vector_package(
        directory,
        document,
        record_id,
        options=options,
    )
    try:
        return publish_staged_vector_package(
            staging,
            directory,
            document=document,
        )
    except Exception as exc:
        if isinstance(exc, ArtifactVectorExportError) and exc.committed:
            raise
        try:
            discarded = discard_staged_vector_package(staging, directory)
        except ArtifactVectorExportError as cleanup_exc:
            if cleanup_exc.committed:
                raise
            raise ArtifactVectorExportError(
                "vector export failed and staging cleanup is not proven"
            ) from cleanup_exc
        if not discarded:
            raise ArtifactVectorExportError(
                "vector export failed and staging cleanup is not proven"
            ) from exc
        raise


__all__ = [
    "ArtifactVectorExportError",
    "MAX_VECTOR_EXPORT_SIDECAR_BYTES",
    "MAX_VECTOR_EXPORT_SVG_BYTES",
    "VECTOR_EXPORT_DIRECTORY_SUFFIX",
    "VECTOR_EXPORT_FORMAT",
    "VECTOR_EXPORT_SCHEMA_VERSION",
    "VECTOR_EXPORT_SIDECAR_MEDIA_TYPE",
    "VECTOR_EXPORT_SIDECAR_NAME",
    "VECTOR_EXPORT_SVG_MEDIA_TYPE",
    "VECTOR_EXPORT_SVG_NAME",
    "VectorExportBundle",
    "VectorSVGOptions",
    "PreparedVectorPublication",
    "build_vector_export",
    "build_public_export_provenance",
    "discard_staged_vector_package",
    "discard_prepared_vector_package",
    "export_vector_package",
    "fsync_export_directory",
    "publish_staged_vector_package",
    "prepare_staged_vector_publication",
    "publish_prepared_vector_package",
    "publish_export_directory_noreplace",
    "read_bounded_export_file",
    "stage_vector_package",
    "validate_vector_export_bytes",
    "validate_vector_export_package",
    "validate_public_export_provenance",
    "write_new_export_file",
]
