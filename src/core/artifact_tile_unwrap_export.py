"""Portable 1:1 export packages for authoritative roof-tile unwrapping."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from threading import RLock
from typing import Any, Mapping
import uuid
from xml.sax.saxutils import escape

import numpy as np

from .artifact_document import (
    ArtifactDocument,
    DerivedRecord,
    RecordFreshness,
    RecordLifecycleStatus,
    canonical_recipe_hash,
)
from .artifact_record_validation import validate_known_records
from .artifact_tile_unwrap_extractor import (
    MAX_TILE_UNWRAP_PAYLOAD_BYTES,
    SECTION_CENTER_FIT_PER_SECTION,
    ArtifactTileUnwrapError,
    TileUnwrapMesh,
    recompute_tile_unwrap_payload_qc,
    selection_face_indices,
    validate_tile_unwrap_recipe,
)
from .artifact_tile_unwrap_record import (
    TILE_UNWRAP_DISTORTION_FACE_MAX_MILLIONTHS,
    TILE_UNWRAP_DISTORTION_MEAN_MAX_MILLIONTHS,
    TILE_UNWRAP_DISTORTION_P95_MAX_MILLIONTHS,
    TILE_UNWRAP_RECORD_TYPE,
    ArtifactTileUnwrapRecordError,
    tile_unwrap_receipt_from_record,
    validate_tile_unwrap_qc,
    validate_tile_unwrap_receipt,
)
from .artifact_vector_export import (
    ArtifactVectorExportError,
    build_public_export_provenance,
    fsync_export_directory,
    publish_export_directory_noreplace,
    read_bounded_export_file,
    validate_current_public_export_provenance,
    write_new_export_file,
)
from .canonical_json import (
    CanonicalJSONError,
    canonical_json_bytes,
    canonical_json_sha256,
)


TILE_UNWRAP_EXPORT_FORMAT = "archmeshrubbing_tile_unwrap_export"
TILE_UNWRAP_EXPORT_SCHEMA_VERSION = "1.5.0"
TILE_UNWRAP_LEGACY_EXPORT_SCHEMA_VERSION = "1.1.0"
#: The 1.2 sidecar predates the section centre and station policies, so it
#: can carry a 1.1 or 1.2 recipe but never a 1.3 one.
TILE_UNWRAP_1_2_EXPORT_SCHEMA_VERSION = "1.2.0"
#: The 1.3 sidecar bounds every record's per-face distortion at 25%; a
#: record unrolled about the measured axis may report more, and only the
#: 1.4 sidecar can carry it.
TILE_UNWRAP_1_3_EXPORT_SCHEMA_VERSION = "1.3.0"
#: The 1.4 sidecar lifted only the per-face maximum for an axis-centred
#: record and still bounds its mean at 7.5% and its 95th percentile at 15%.
#: A tile whose back is meshed finely enough to carry its cord passes both
#: of those, so the 1.5 sidecar reports all three there and bounds none;
#: under a fitted centre every bound is exactly what it was.
TILE_UNWRAP_1_4_EXPORT_SCHEMA_VERSION = "1.4.0"
TILE_UNWRAP_EXPORT_DIRECTORY_SUFFIX = ".amr-unwrap"
TILE_UNWRAP_EXPORT_PAYLOAD_NAME = "artifact.amr-unwrap.bin"
TILE_UNWRAP_EXPORT_OBJ_NAME = "artifact.obj"
TILE_UNWRAP_EXPORT_SVG_NAME = "artifact.svg"
TILE_UNWRAP_EXPORT_SIDECAR_NAME = "artifact.amr-unwrap.json"

TILE_UNWRAP_PAYLOAD_MEDIA_TYPE = "application/vnd.archmeshrubbing.tile-unwrap-payload"
TILE_UNWRAP_OBJ_MEDIA_TYPE = "model/obj"
TILE_UNWRAP_SVG_MEDIA_TYPE = "image/svg+xml"
TILE_UNWRAP_SIDECAR_MEDIA_TYPE = (
    "application/vnd.archmeshrubbing.tile-unwrap-export+json"
)

MAX_TILE_UNWRAP_OBJ_BYTES = 256 * 1024 * 1024
MAX_TILE_UNWRAP_SVG_BYTES = 64 * 1024 * 1024
MAX_TILE_UNWRAP_SIDECAR_BYTES = 16 * 1024 * 1024

_PACKAGE_NAMES = frozenset(
    {
        TILE_UNWRAP_EXPORT_PAYLOAD_NAME,
        TILE_UNWRAP_EXPORT_OBJ_NAME,
        TILE_UNWRAP_EXPORT_SVG_NAME,
        TILE_UNWRAP_EXPORT_SIDECAR_NAME,
    }
)
_TILE_UNWRAP_STAGING_PREFIX = ".amru-stage-"
_TILE_UNWRAP_QUARANTINE_PREFIX = ".amru-discard-"
_STAGING_OWNERS_LOCK = RLock()
_STAGING_OWNERS: dict[str, "_OwnedTileUnwrapStaging"] = {}
_PREPARED_PUBLICATIONS: dict[object, "PreparedTileUnwrapPublication"] = {}


class ArtifactTileUnwrapExportError(ValueError):
    """An unwrap export or package violates its closed public contract."""

    def __init__(self, message: str, *, committed: bool = False) -> None:
        super().__init__(message)
        self.committed = bool(committed)


@dataclass(frozen=True, slots=True)
class TileUnwrapExportBundle:
    payload_bytes: bytes
    obj_bytes: bytes
    svg_bytes: bytes
    sidecar_bytes: bytes


@dataclass(frozen=True, slots=True)
class TileUnwrapExportPublication:
    destination: Path
    durability_confirmed: bool


@dataclass(frozen=True, slots=True)
class _OwnedTileUnwrapStaging:
    path: Path
    destination: Path
    device: int
    inode: int
    parent_device: int
    parent_inode: int
    staging_directory_fsync_confirmed: bool


@dataclass(frozen=True, slots=True)
class _TileUnwrapEntryFingerprint:
    name: str
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True, eq=False)
class PreparedTileUnwrapPublication:
    """Opaque exact authority to publish one validated unwrap staging inode."""

    staging_directory: Path
    destination: Path
    _owned: _OwnedTileUnwrapStaging = dataclass_field(repr=False)
    _fingerprint: tuple[_TileUnwrapEntryFingerprint, ...] = dataclass_field(
        repr=False
    )
    _staging_directory_fsync_confirmed: bool = dataclass_field(repr=False)
    _nonce: object = dataclass_field(repr=False, compare=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact_keys(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactTileUnwrapExportError(f"{name} must be an object")
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise ArtifactTileUnwrapExportError(
            f"{name} is missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise ArtifactTileUnwrapExportError(
            f"{name} has unknown fields: {', '.join(unknown)}"
        )
    return value


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactTileUnwrapExportError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ArtifactTileUnwrapExportError(
            f"{name} must be in the inclusive range {minimum}..{maximum}"
        )
    return value


def _artifact_descriptor(
    *, name: str, media_type: str, payload: bytes
) -> dict[str, Any]:
    return {
        "byte_length": len(payload),
        "media_type": media_type,
        "name": name,
        "sha256": _sha256_bytes(payload),
    }


def _require_exportable_record(
    document: ArtifactDocument,
    record_id: str,
) -> DerivedRecord:
    if not isinstance(document, ArtifactDocument):
        raise ArtifactTileUnwrapExportError("document must be an ArtifactDocument")
    validate_known_records(document)
    record = document.record_index.get(str(record_id))
    if record is None:
        raise ArtifactTileUnwrapExportError(f"record {record_id!r} does not exist")
    if record.type != TILE_UNWRAP_RECORD_TYPE:
        raise ArtifactTileUnwrapExportError("record is not a tile unwrap record")
    if record.lifecycle_status is not RecordLifecycleStatus.READY:
        raise ArtifactTileUnwrapExportError("only READY tile unwrap records may export")
    if document.record_freshness(record.id) is not RecordFreshness.FRESH:
        raise ArtifactTileUnwrapExportError("only FRESH tile unwrap records may export")
    return record


def _require_matching_unwrap(
    record: DerivedRecord,
    unwrap: TileUnwrapMesh,
) -> dict[str, Any]:
    if not isinstance(unwrap, TileUnwrapMesh):
        raise ArtifactTileUnwrapExportError("unwrap must be a TileUnwrapMesh")
    receipt = tile_unwrap_receipt_from_record(record)
    computed = unwrap.receipt(selection_sha256=str(receipt["selection_sha256"]))
    if computed != receipt:
        raise ArtifactTileUnwrapExportError(
            "recomputed tile unwrap does not match the durable receipt"
        )
    payload = unwrap.canonical_payload_bytes(
        selection_sha256=str(receipt["selection_sha256"])
    )
    if _sha256_bytes(payload) != receipt["unwrap_sha256"]:
        raise ArtifactTileUnwrapExportError(
            "canonical tile unwrap payload does not match its receipt"
        )
    return receipt


def _millimetre_token(value_um: int) -> str:
    sign = "-" if value_um < 0 else ""
    whole, fractional = divmod(abs(int(value_um)), 1000)
    if fractional == 0:
        return f"{sign}{whole}"
    return f"{sign}{whole}.{fractional:03d}".rstrip("0")


def _boundary_loops(unwrap: TileUnwrapMesh) -> tuple[tuple[int, ...], ...]:
    faces = np.asarray(unwrap.faces, dtype=np.int32)
    occurrences: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a_raw, b_raw, c_raw in faces:
        a, b, c = int(a_raw), int(b_raw), int(c_raw)
        for start, end in ((a, b), (b, c), (c, a)):
            key = (min(start, end), max(start, end))
            occurrences.setdefault(key, []).append((start, end))
    if any(len(items) > 2 for items in occurrences.values()):
        raise ArtifactTileUnwrapExportError(
            "tile unwrap has a non-manifold edge and cannot form a boundary"
        )
    boundary = {items[0] for items in occurrences.values() if len(items) == 1}
    if not boundary:
        raise ArtifactTileUnwrapExportError("tile unwrap has no open boundary")
    outgoing: dict[int, int] = {}
    incoming: dict[int, int] = {}
    for start, end in boundary:
        if start in outgoing or end in incoming:
            raise ArtifactTileUnwrapExportError(
                "tile unwrap boundary is branched or inconsistently oriented"
            )
        outgoing[start] = end
        incoming[end] = start
    if set(outgoing) != set(incoming):
        raise ArtifactTileUnwrapExportError("tile unwrap boundary is not closed")
    uv = np.asarray(unwrap.uv_um, dtype=np.int64)

    def vertex_key(index: int) -> tuple[int, int, int]:
        return (int(uv[index, 0]), int(uv[index, 1]), index)

    unused = set(boundary)
    loops: list[tuple[int, ...]] = []
    while unused:
        start = min((edge[0] for edge in unused), key=vertex_key)
        current = start
        loop: list[int] = []
        while True:
            loop.append(current)
            next_vertex = outgoing.get(current)
            if next_vertex is None or (current, next_vertex) not in unused:
                raise ArtifactTileUnwrapExportError(
                    "tile unwrap boundary traversal is inconsistent"
                )
            unused.remove((current, next_vertex))
            current = next_vertex
            if current == start:
                break
            if len(loop) > len(boundary):
                raise ArtifactTileUnwrapExportError(
                    "tile unwrap boundary traversal did not close"
                )
        if len(loop) < 3:
            raise ArtifactTileUnwrapExportError(
                "tile unwrap boundary loop has fewer than three vertices"
            )
        loops.append(tuple(loop))

    def loop_key(loop: tuple[int, ...]) -> tuple[Any, ...]:
        points = uv[np.asarray(loop, dtype=np.int64)].astype(np.float64)
        shifted = np.roll(points, -1, axis=0)
        twice_area = float(
            np.sum(points[:, 0] * shifted[:, 1] - points[:, 1] * shifted[:, 0])
        )
        return (
            -abs(twice_area),
            tuple(vertex_key(index) for index in loop),
        )

    return tuple(sorted(loops, key=loop_key))


def _render_obj(unwrap: TileUnwrapMesh, receipt: Mapping[str, Any]) -> bytes:
    lines = [
        "# ArchMeshRubbing authoritative tile unwrap",
        "# unit: millimetre; physical scale: 1:1",
        f"# unwrap_sha256: {receipt['unwrap_sha256']}",
        "o artifact_tile_unwrap",
    ]
    for u, v in np.asarray(unwrap.uv_um, dtype=np.int64):
        lines.append(f"v {_millimetre_token(int(u))} {_millimetre_token(int(v))} 0")
    for a, b, c in np.asarray(unwrap.faces, dtype=np.int32):
        lines.append(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}")
    encoded = ("\n".join(lines) + "\n").encode("ascii")
    if len(encoded) > MAX_TILE_UNWRAP_OBJ_BYTES:
        raise ArtifactTileUnwrapExportError("tile unwrap OBJ exceeds size limit")
    return encoded


def _svg_metadata(
    receipt: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    sidecar_claims_sha256: str,
) -> dict[str, Any]:
    record = provenance["record"]
    document = provenance["document"]
    assert isinstance(record, Mapping)
    assert isinstance(document, Mapping)
    return {
        "coordinate_quantum_um": receipt["coordinate_quantum_um"],
        "document_id": document["document_id"],
        "document_manifest_sha256": document["manifest_sha256"],
        "format": "archmeshrubbing_tile_unwrap_svg_metadata",
        "geometry_ref": record["geometry_ref"],
        "physical_scale": "1:1",
        "recipe_hash": record["recipe_hash"],
        "record_id": record["id"],
        "record_type": record["type"],
        "schema_version": "1.0.0",
        "selection_sha256": receipt["selection_sha256"],
        "sidecar": TILE_UNWRAP_EXPORT_SIDECAR_NAME,
        "sidecar_claims_sha256": sidecar_claims_sha256,
        "unit": "mm",
        "unwrap_sha256": receipt["unwrap_sha256"],
    }


def _render_svg(
    unwrap: TileUnwrapMesh,
    receipt: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    sidecar_claims_sha256: str,
) -> tuple[bytes, int]:
    loops = _boundary_loops(unwrap)
    width_um = int(receipt["width_mm_exact"]["numerator"])
    height_um = int(receipt["height_mm_exact"]["numerator"])
    width = _millimetre_token(width_um)
    height = _millimetre_token(height_um)
    metadata_text = canonical_json_bytes(
        _svg_metadata(
            receipt,
            provenance,
            sidecar_claims_sha256=sidecar_claims_sha256,
        )
    ).decode("utf-8")
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}">'
        ),
        f'  <metadata id="archmeshrubbing">{escape(metadata_text)}</metadata>',
        (
            f'  <g id="tile-boundary" transform="translate(0 {height}) scale(1 -1)" '
            'fill="none" stroke="#000000" stroke-width="0.2" '
            'vector-effect="non-scaling-stroke">'
        ),
    ]
    uv = np.asarray(unwrap.uv_um, dtype=np.int64)
    for index, loop in enumerate(loops):
        commands: list[str] = []
        for point_index, vertex_index in enumerate(loop):
            u, v = uv[vertex_index]
            prefix = "M" if point_index == 0 else "L"
            commands.append(
                f"{prefix} {_millimetre_token(int(u))} {_millimetre_token(int(v))}"
            )
        commands.append("Z")
        lines.append(f'    <path id="boundary-{index}" d="{" ".join(commands)}"/>')
    lines.extend(("  </g>", "</svg>"))
    encoded = ("\n".join(lines) + "\n").encode("utf-8")
    if len(encoded) > MAX_TILE_UNWRAP_SVG_BYTES:
        raise ArtifactTileUnwrapExportError("tile unwrap SVG exceeds size limit")
    return encoded, len(loops)


def _privacy_claims() -> dict[str, bool]:
    return {
        "absolute_paths_embedded": False,
        "canonical_source_geometry_embedded": False,
        "operator_annotations_embedded": False,
    }


def _sidecar_claims(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "format",
            "geometry",
            "presentation",
            "privacy",
            "provenance",
            "qc",
            "recipe",
            "schema_version",
        )
    }


def build_tile_unwrap_export(
    document: ArtifactDocument,
    record_id: str,
    unwrap: TileUnwrapMesh,
) -> TileUnwrapExportBundle:
    record = _require_exportable_record(document, record_id)
    receipt = _require_matching_unwrap(record, unwrap)
    try:
        provenance = build_public_export_provenance(document, record)
    except ArtifactVectorExportError as exc:
        raise ArtifactTileUnwrapExportError(str(exc)) from exc
    payload_bytes = unwrap.canonical_payload_bytes(
        selection_sha256=str(receipt["selection_sha256"])
    )
    obj_bytes = _render_obj(unwrap, receipt)
    loop_count = len(_boundary_loops(unwrap))
    record_qc = record.to_dict()["qc"]
    assert isinstance(record_qc, dict)
    claims_source = {
        "format": TILE_UNWRAP_EXPORT_FORMAT,
        "geometry": receipt,
        "presentation": {
            "boundary_definition": "single_incident_face_edges",
            "boundary_loop_count": loop_count,
            "height_mm_exact": receipt["height_mm_exact"],
            "physical_scale": "1:1",
            "unit": "mm",
            "width_mm_exact": receipt["width_mm_exact"],
            "y_axis": "up",
        },
        "privacy": _privacy_claims(),
        "provenance": provenance,
        "qc": {
            "export_gate": {
                "payload_verified": True,
                "record_freshness": RecordFreshness.FRESH.value,
                "record_lifecycle_status": RecordLifecycleStatus.READY.value,
            },
            "record": record_qc,
        },
        "recipe": record.to_dict()["recipe"],
        "schema_version": TILE_UNWRAP_EXPORT_SCHEMA_VERSION,
    }
    claims_sha256 = canonical_json_sha256(_sidecar_claims(claims_source))
    svg_bytes, rendered_loop_count = _render_svg(
        unwrap,
        receipt,
        provenance,
        sidecar_claims_sha256=claims_sha256,
    )
    if rendered_loop_count != loop_count:  # pragma: no cover - deterministic helper
        raise ArtifactTileUnwrapExportError("tile unwrap boundary count changed")
    sidecar = {
        **claims_source,
        "artifacts": {
            "canonical_payload": _artifact_descriptor(
                name=TILE_UNWRAP_EXPORT_PAYLOAD_NAME,
                media_type=TILE_UNWRAP_PAYLOAD_MEDIA_TYPE,
                payload=payload_bytes,
            ),
            "flat_mesh": _artifact_descriptor(
                name=TILE_UNWRAP_EXPORT_OBJ_NAME,
                media_type=TILE_UNWRAP_OBJ_MEDIA_TYPE,
                payload=obj_bytes,
            ),
            "outline": _artifact_descriptor(
                name=TILE_UNWRAP_EXPORT_SVG_NAME,
                media_type=TILE_UNWRAP_SVG_MEDIA_TYPE,
                payload=svg_bytes,
            ),
        },
        "claims_sha256": claims_sha256,
    }
    try:
        sidecar_bytes = canonical_json_bytes(sidecar) + b"\n"
    except CanonicalJSONError as exc:
        raise ArtifactTileUnwrapExportError(str(exc)) from exc
    if len(sidecar_bytes) > MAX_TILE_UNWRAP_SIDECAR_BYTES:
        raise ArtifactTileUnwrapExportError("tile unwrap sidecar exceeds size limit")
    bundle = TileUnwrapExportBundle(
        payload_bytes=payload_bytes,
        obj_bytes=obj_bytes,
        svg_bytes=svg_bytes,
        sidecar_bytes=sidecar_bytes,
    )
    validate_tile_unwrap_export_bytes(
        payload_bytes,
        obj_bytes,
        svg_bytes,
        sidecar_bytes,
        document=document,
    )
    return bundle


def _strict_sidecar_bytes(sidecar_bytes: bytes) -> dict[str, Any]:
    if (
        not isinstance(sidecar_bytes, bytes)
        or len(sidecar_bytes) < 3
        or len(sidecar_bytes) > MAX_TILE_UNWRAP_SIDECAR_BYTES
        or not sidecar_bytes.endswith(b"\n")
    ):
        raise ArtifactTileUnwrapExportError("tile unwrap sidecar bytes are invalid")
    try:
        value = json.loads(sidecar_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactTileUnwrapExportError(
            "tile unwrap sidecar is invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ArtifactTileUnwrapExportError("tile unwrap sidecar must be an object")
    try:
        if canonical_json_bytes(value) + b"\n" != sidecar_bytes:
            raise ArtifactTileUnwrapExportError(
                "tile unwrap sidecar is not canonical RFC 8785 JSON"
            )
    except CanonicalJSONError as exc:
        raise ArtifactTileUnwrapExportError(str(exc)) from exc
    return value


def _validate_artifact_descriptor(
    value: object,
    *,
    expected_name: str,
    expected_media_type: str,
    payload: bytes,
    label: str,
) -> None:
    descriptor = _exact_keys(
        value,
        {"byte_length", "media_type", "name", "sha256"},
        name=f"artifacts.{label}",
    )
    expected = _artifact_descriptor(
        name=expected_name,
        media_type=expected_media_type,
        payload=payload,
    )
    if dict(descriptor) != expected:
        raise ArtifactTileUnwrapExportError(
            f"tile unwrap {label} descriptor does not match bytes"
        )


def validate_tile_unwrap_export_bytes(
    payload_bytes: bytes,
    obj_bytes: bytes,
    svg_bytes: bytes,
    sidecar_bytes: bytes,
    *,
    document: ArtifactDocument | None = None,
) -> dict[str, Any]:
    if len(payload_bytes) > MAX_TILE_UNWRAP_PAYLOAD_BYTES:
        raise ArtifactTileUnwrapExportError("tile unwrap payload exceeds size limit")
    if not obj_bytes or len(obj_bytes) > MAX_TILE_UNWRAP_OBJ_BYTES:
        raise ArtifactTileUnwrapExportError("tile unwrap OBJ byte length is invalid")
    if not svg_bytes or len(svg_bytes) > MAX_TILE_UNWRAP_SVG_BYTES:
        raise ArtifactTileUnwrapExportError("tile unwrap SVG byte length is invalid")
    sidecar = _strict_sidecar_bytes(sidecar_bytes)
    root = _exact_keys(
        sidecar,
        {
            "artifacts",
            "claims_sha256",
            "format",
            "geometry",
            "presentation",
            "privacy",
            "provenance",
            "qc",
            "recipe",
            "schema_version",
        },
        name="tile unwrap sidecar",
    )
    if root["format"] != TILE_UNWRAP_EXPORT_FORMAT:
        raise ArtifactTileUnwrapExportError("tile unwrap export format is invalid")
    sidecar_schema_version = root["schema_version"]
    if sidecar_schema_version not in {
        TILE_UNWRAP_EXPORT_SCHEMA_VERSION,
        TILE_UNWRAP_1_4_EXPORT_SCHEMA_VERSION,
        TILE_UNWRAP_1_3_EXPORT_SCHEMA_VERSION,
        TILE_UNWRAP_1_2_EXPORT_SCHEMA_VERSION,
        TILE_UNWRAP_LEGACY_EXPORT_SCHEMA_VERSION,
    }:
        raise ArtifactTileUnwrapExportError("tile unwrap export schema is invalid")
    claims_sha256 = root["claims_sha256"]
    if (
        not isinstance(claims_sha256, str)
        or len(claims_sha256) != 64
        or any(character not in "0123456789abcdef" for character in claims_sha256)
        or claims_sha256 != canonical_json_sha256(_sidecar_claims(root))
    ):
        raise ArtifactTileUnwrapExportError(
            "tile unwrap sidecar claims SHA-256 is invalid"
        )
    try:
        receipt = validate_tile_unwrap_receipt(root["geometry"])
        recipe = validate_tile_unwrap_recipe(root["recipe"])
        provenance = validate_current_public_export_provenance(root["provenance"])
    except (
        ArtifactTileUnwrapRecordError,
        ArtifactTileUnwrapError,
        ArtifactVectorExportError,
    ) as exc:
        raise ArtifactTileUnwrapExportError(str(exc)) from exc
    if sidecar_schema_version == TILE_UNWRAP_LEGACY_EXPORT_SCHEMA_VERSION and (
        recipe.get("algorithm_version") != "1.1.0"
        or recipe.get("schema_version") != "1.1.0"
    ):
        raise ArtifactTileUnwrapExportError(
            "legacy tile unwrap export requires a legacy 1.1 recipe"
        )
    if sidecar_schema_version == TILE_UNWRAP_1_2_EXPORT_SCHEMA_VERSION and (
        recipe.get("schema_version") not in {"1.1.0", "1.2.0"}
    ):
        raise ArtifactTileUnwrapExportError(
            "a 1.2 tile unwrap export cannot carry a recipe newer than 1.2"
        )
    selection = recipe["selection"]
    assert isinstance(selection, Mapping)
    if receipt["selection_sha256"] != selection["selection_sha256"]:
        raise ArtifactTileUnwrapExportError(
            "tile unwrap receipt selection differs from recipe"
        )
    if receipt["source_face_count"] != selection["selected_face_count"]:
        raise ArtifactTileUnwrapExportError(
            "tile unwrap receipt selection count differs from recipe"
        )
    expected_source_face_indices = selection_face_indices(selection)
    try:
        unwrap, _header = TileUnwrapMesh.from_canonical_payload_bytes(
            payload_bytes,
            expected_selection_sha256=str(receipt["selection_sha256"]),
        )
    except ArtifactTileUnwrapError as exc:
        raise ArtifactTileUnwrapExportError(str(exc)) from exc
    if _sha256_bytes(payload_bytes) != receipt["unwrap_sha256"]:
        raise ArtifactTileUnwrapExportError("tile unwrap payload SHA-256 is invalid")
    if unwrap.receipt(selection_sha256=str(receipt["selection_sha256"])) != receipt:
        raise ArtifactTileUnwrapExportError(
            "tile unwrap payload does not reproduce its receipt"
        )
    if not np.array_equal(
        np.asarray(unwrap.source_face_indices, dtype=np.int64),
        expected_source_face_indices,
    ):
        raise ArtifactTileUnwrapExportError(
            "tile unwrap payload source faces differ from recipe selection"
        )
    artifacts = _exact_keys(
        root["artifacts"],
        {"canonical_payload", "flat_mesh", "outline"},
        name="artifacts",
    )
    _validate_artifact_descriptor(
        artifacts["canonical_payload"],
        expected_name=TILE_UNWRAP_EXPORT_PAYLOAD_NAME,
        expected_media_type=TILE_UNWRAP_PAYLOAD_MEDIA_TYPE,
        payload=payload_bytes,
        label="canonical_payload",
    )
    _validate_artifact_descriptor(
        artifacts["flat_mesh"],
        expected_name=TILE_UNWRAP_EXPORT_OBJ_NAME,
        expected_media_type=TILE_UNWRAP_OBJ_MEDIA_TYPE,
        payload=obj_bytes,
        label="flat_mesh",
    )
    _validate_artifact_descriptor(
        artifacts["outline"],
        expected_name=TILE_UNWRAP_EXPORT_SVG_NAME,
        expected_media_type=TILE_UNWRAP_SVG_MEDIA_TYPE,
        payload=svg_bytes,
        label="outline",
    )
    expected_obj = _render_obj(unwrap, receipt)
    expected_svg, loop_count = _render_svg(
        unwrap,
        receipt,
        provenance,
        sidecar_claims_sha256=claims_sha256,
    )
    if obj_bytes != expected_obj:
        raise ArtifactTileUnwrapExportError("tile unwrap OBJ is not canonical")
    if svg_bytes != expected_svg:
        raise ArtifactTileUnwrapExportError("tile unwrap SVG is not canonical")
    presentation = _exact_keys(
        root["presentation"],
        {
            "boundary_definition",
            "boundary_loop_count",
            "height_mm_exact",
            "physical_scale",
            "unit",
            "width_mm_exact",
            "y_axis",
        },
        name="presentation",
    )
    if dict(presentation) != {
        "boundary_definition": "single_incident_face_edges",
        "boundary_loop_count": loop_count,
        "height_mm_exact": receipt["height_mm_exact"],
        "physical_scale": "1:1",
        "unit": "mm",
        "width_mm_exact": receipt["width_mm_exact"],
        "y_axis": "up",
    }:
        raise ArtifactTileUnwrapExportError("tile unwrap presentation is invalid")
    if root["privacy"] != _privacy_claims():
        raise ArtifactTileUnwrapExportError("tile unwrap privacy claims are invalid")
    qc = _exact_keys(root["qc"], {"export_gate", "record"}, name="qc")
    if qc["export_gate"] != {
        "payload_verified": True,
        "record_freshness": RecordFreshness.FRESH.value,
        "record_lifecycle_status": RecordLifecycleStatus.READY.value,
    }:
        raise ArtifactTileUnwrapExportError("tile unwrap export gate is invalid")
    if not isinstance(qc["record"], Mapping):
        raise ArtifactTileUnwrapExportError("tile unwrap record QC must be an object")
    try:
        validate_tile_unwrap_qc(
            qc["record"],
            receipt,
            section_center_policy=str(
                recipe.get("section_center_policy", SECTION_CENTER_FIT_PER_SECTION)
            ),
        )
    except ArtifactTileUnwrapRecordError as exc:
        raise ArtifactTileUnwrapExportError(str(exc)) from exc
    if sidecar_schema_version not in {
        TILE_UNWRAP_EXPORT_SCHEMA_VERSION,
        TILE_UNWRAP_1_4_EXPORT_SCHEMA_VERSION,
    } and (
        int(qc["record"]["distortion_max_millionths"])
        > TILE_UNWRAP_DISTORTION_FACE_MAX_MILLIONTHS
    ):
        # Every sidecar before 1.4 bounds the per-face maximum in its schema,
        # whatever the centre policy; the runtime holds the same line.
        raise ArtifactTileUnwrapExportError(
            "a tile unwrap export before 1.4 cannot carry a face over the "
            "distortion gate"
        )
    if sidecar_schema_version != TILE_UNWRAP_EXPORT_SCHEMA_VERSION and (
        int(qc["record"]["distortion_mean_millionths"])
        > TILE_UNWRAP_DISTORTION_MEAN_MAX_MILLIONTHS
        or int(qc["record"]["distortion_p95_millionths"])
        > TILE_UNWRAP_DISTORTION_P95_MAX_MILLIONTHS
    ):
        # And every sidecar before 1.5 bounds the mean and the 95th percentile
        # whatever the centre policy, 1.4 included: it lifted the per-face
        # maximum alone.  Only a 1.5 sidecar can carry an axis-centred record
        # whose relief pushes those two past their gates.
        raise ArtifactTileUnwrapExportError(
            "a tile unwrap export before 1.5 cannot carry a mean or 95th "
            "percentile over the distortion gate"
        )
    try:
        recomputed_payload_qc = recompute_tile_unwrap_payload_qc(unwrap)
    except ArtifactTileUnwrapError as exc:
        raise ArtifactTileUnwrapExportError(
            f"tile unwrap payload topology QC failed: {exc}"
        ) from exc
    record_qc = qc["record"]
    assert isinstance(record_qc, Mapping)
    for key, expected in recomputed_payload_qc.items():
        if record_qc.get(key) != expected:
            raise ArtifactTileUnwrapExportError(
                f"tile unwrap payload recomputed QC differs at {key!r}"
            )
    provenance_record = provenance["record"]
    assert isinstance(provenance_record, Mapping)
    if provenance_record["type"] != TILE_UNWRAP_RECORD_TYPE:
        raise ArtifactTileUnwrapExportError("tile unwrap provenance type is invalid")
    if provenance_record["geometry_ref"] != (
        f"urn:archmeshrubbing:tile-unwrap:sha256:{receipt['unwrap_sha256']}"
    ):
        raise ArtifactTileUnwrapExportError(
            "tile unwrap provenance geometry_ref is invalid"
        )
    if provenance_record["selection_hash"] != receipt["selection_sha256"]:
        raise ArtifactTileUnwrapExportError(
            "tile unwrap provenance selection is invalid"
        )
    if provenance_record["recipe_hash"] != canonical_recipe_hash(recipe):
        raise ArtifactTileUnwrapExportError("tile unwrap recipe hash is invalid")
    if document is not None:
        record_id = str(provenance_record["id"])
        record = _require_exportable_record(document, record_id)
        if tile_unwrap_receipt_from_record(record) != receipt:
            raise ArtifactTileUnwrapExportError(
                "tile unwrap package receipt differs from document"
            )
        if record.to_dict()["recipe"] != recipe:
            raise ArtifactTileUnwrapExportError(
                "tile unwrap package recipe differs from document"
            )
        if record.to_dict()["qc"] != dict(qc["record"]):
            raise ArtifactTileUnwrapExportError(
                "tile unwrap package QC differs from document"
            )
        try:
            expected_provenance = build_public_export_provenance(document, record)
        except ArtifactVectorExportError as exc:
            raise ArtifactTileUnwrapExportError(str(exc)) from exc
        if dict(provenance) != expected_provenance:
            raise ArtifactTileUnwrapExportError(
                "tile unwrap package provenance differs from document"
            )
    return sidecar


def validate_tile_unwrap_export_package(
    directory: str | os.PathLike[str],
    *,
    document: ArtifactDocument | None = None,
) -> dict[str, Any]:
    path = Path(directory)
    try:
        identity = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactTileUnwrapExportError(
            f"cannot inspect tile unwrap package: {exc}"
        ) from exc
    if not stat.S_ISDIR(identity.st_mode):
        raise ArtifactTileUnwrapExportError(
            "tile unwrap package is not a real directory"
        )
    try:
        entries = {entry.name: entry for entry in path.iterdir()}
    except OSError as exc:
        raise ArtifactTileUnwrapExportError(
            f"cannot enumerate tile unwrap package: {exc}"
        ) from exc
    if set(entries) != _PACKAGE_NAMES:
        raise ArtifactTileUnwrapExportError(
            "tile unwrap package entries do not match the closed contract"
        )
    limits = {
        TILE_UNWRAP_EXPORT_PAYLOAD_NAME: MAX_TILE_UNWRAP_PAYLOAD_BYTES,
        TILE_UNWRAP_EXPORT_OBJ_NAME: MAX_TILE_UNWRAP_OBJ_BYTES,
        TILE_UNWRAP_EXPORT_SVG_NAME: MAX_TILE_UNWRAP_SVG_BYTES,
        TILE_UNWRAP_EXPORT_SIDECAR_NAME: MAX_TILE_UNWRAP_SIDECAR_BYTES,
    }
    payloads: dict[str, bytes] = {}
    for name, entry in entries.items():
        entry_stat = entry.stat(follow_symlinks=False)
        if not stat.S_ISREG(entry_stat.st_mode):
            raise ArtifactTileUnwrapExportError(
                f"tile unwrap package entry {name!r} is not a regular file"
            )
        try:
            payloads[name] = read_bounded_export_file(
                entry, limit=limits[name], label=name
            )
        except ArtifactVectorExportError as exc:
            raise ArtifactTileUnwrapExportError(str(exc)) from exc
    return validate_tile_unwrap_export_bytes(
        payloads[TILE_UNWRAP_EXPORT_PAYLOAD_NAME],
        payloads[TILE_UNWRAP_EXPORT_OBJ_NAME],
        payloads[TILE_UNWRAP_EXPORT_SVG_NAME],
        payloads[TILE_UNWRAP_EXPORT_SIDECAR_NAME],
        document=document,
    )


def _destination_path(directory: str | os.PathLike[str]) -> Path:
    destination = Path(os.path.abspath(os.fspath(Path(directory).expanduser())))
    if not destination.name.endswith(TILE_UNWRAP_EXPORT_DIRECTORY_SUFFIX):
        raise ArtifactTileUnwrapExportError(
            f"export directory must end with {TILE_UNWRAP_EXPORT_DIRECTORY_SUFFIX}"
        )
    if destination.exists() or destination.is_symlink():
        raise ArtifactTileUnwrapExportError("export destination already exists")
    if not destination.parent.is_dir():
        raise ArtifactTileUnwrapExportError("export parent directory does not exist")
    return destination


def _staging_registry_key(path: Path) -> str:
    return os.path.abspath(os.fspath(path))


def _path_exists_without_following(path: Path) -> bool:
    try:
        path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def _require_current_parent(staging: _OwnedTileUnwrapStaging) -> None:
    try:
        current = staging.path.parent.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactTileUnwrapExportError(
            f"cannot inspect tile unwrap export parent: {exc}"
        ) from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or current.st_dev != staging.parent_device
        or current.st_ino != staging.parent_inode
    ):
        raise ArtifactTileUnwrapExportError(
            "tile unwrap export parent identity changed"
        )


def _require_owned_staging_identity(staging: _OwnedTileUnwrapStaging) -> None:
    try:
        current = staging.path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactTileUnwrapExportError(
            f"cannot inspect tile unwrap export staging: {exc}"
        ) from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or current.st_dev != staging.device
        or current.st_ino != staging.inode
    ):
        raise ArtifactTileUnwrapExportError(
            "tile unwrap export staging identity changed; path was preserved"
        )


def _capture_tile_unwrap_package_fingerprint(
    directory: Path,
) -> tuple[_TileUnwrapEntryFingerprint, ...]:
    try:
        entries = {entry.name: entry for entry in directory.iterdir()}
    except OSError as exc:
        raise ArtifactTileUnwrapExportError(
            f"cannot inspect tile unwrap staging package: {exc}"
        ) from exc
    if set(entries) != _PACKAGE_NAMES:
        raise ArtifactTileUnwrapExportError(
            "tile unwrap staging package has unexpected entries"
        )
    fingerprints: list[_TileUnwrapEntryFingerprint] = []
    for name in sorted(entries):
        try:
            identity = entries[name].stat(follow_symlinks=False)
        except OSError as exc:
            raise ArtifactTileUnwrapExportError(
                f"cannot inspect tile unwrap staging entry {name!r}: {exc}"
            ) from exc
        if not stat.S_ISREG(identity.st_mode):
            raise ArtifactTileUnwrapExportError(
                f"tile unwrap staging entry {name!r} is not a regular file"
            )
        fingerprints.append(
            _TileUnwrapEntryFingerprint(
                name=name,
                device=identity.st_dev,
                inode=identity.st_ino,
                mode=identity.st_mode,
                size=identity.st_size,
                mtime_ns=identity.st_mtime_ns,
                ctime_ns=identity.st_ctime_ns,
            )
        )
    return tuple(fingerprints)


def _invalidate_tile_unwrap_prepared_locked(
    staging: _OwnedTileUnwrapStaging,
) -> None:
    for nonce, prepared in tuple(_PREPARED_PUBLICATIONS.items()):
        if prepared._owned is staging:
            _PREPARED_PUBLICATIONS.pop(nonce, None)


def _raise_if_owned_destination_is_visible(
    staging: _OwnedTileUnwrapStaging,
) -> None:
    try:
        current = staging.destination.stat(follow_symlinks=False)
    except OSError:
        return
    if (
        stat.S_ISDIR(current.st_mode)
        and current.st_dev == staging.device
        and current.st_ino == staging.inode
    ):
        raise ArtifactTileUnwrapExportError(
            "tile unwrap staging is already visible at its destination",
            committed=True,
        )


def _cleanup_owned_stage(path: Path, *, device: int, inode: int) -> None:
    try:
        current = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ArtifactTileUnwrapExportError(
            f"cannot inspect tile unwrap staging for cleanup: {exc}"
        ) from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or current.st_dev != device
        or current.st_ino != inode
    ):
        raise ArtifactTileUnwrapExportError(
            "tile unwrap staging cleanup was not proven; path was preserved"
        )
    entries = list(path.iterdir())
    if any(entry.name not in _PACKAGE_NAMES for entry in entries):
        raise ArtifactTileUnwrapExportError(
            "tile unwrap staging cleanup found foreign entries; path was preserved"
        )
    for entry in entries:
        entry_stat = entry.stat(follow_symlinks=False)
        if not stat.S_ISREG(entry_stat.st_mode):
            raise ArtifactTileUnwrapExportError(
                "tile unwrap staging cleanup found a non-file; path was preserved"
            )
    for entry in entries:
        entry.unlink()
    path.rmdir()


def stage_tile_unwrap_package(
    directory: str | os.PathLike[str],
    document: ArtifactDocument,
    record_id: str,
    unwrap: TileUnwrapMesh,
) -> Path:
    """Build and verify a hidden same-parent package without publishing it."""

    destination = _destination_path(directory)
    bundle = build_tile_unwrap_export(document, record_id, unwrap)
    stage = Path(
        tempfile.mkdtemp(
            prefix=_TILE_UNWRAP_STAGING_PREFIX,
            dir=destination.parent,
        )
    )
    identity = stage.stat(follow_symlinks=False)
    parent_identity = destination.parent.stat(follow_symlinks=False)
    try:
        for name, payload in (
            (TILE_UNWRAP_EXPORT_PAYLOAD_NAME, bundle.payload_bytes),
            (TILE_UNWRAP_EXPORT_OBJ_NAME, bundle.obj_bytes),
            (TILE_UNWRAP_EXPORT_SVG_NAME, bundle.svg_bytes),
            (TILE_UNWRAP_EXPORT_SIDECAR_NAME, bundle.sidecar_bytes),
        ):
            try:
                write_new_export_file(stage / name, payload)
            except (OSError, ArtifactVectorExportError) as exc:
                raise ArtifactTileUnwrapExportError(
                    f"cannot write tile unwrap package: {exc}"
                ) from exc
        validate_tile_unwrap_export_package(stage, document=document)
        staging_fsync_confirmed = fsync_export_directory(stage)
        current = stage.stat(follow_symlinks=False)
        if current.st_dev != identity.st_dev or current.st_ino != identity.st_ino:
            raise ArtifactTileUnwrapExportError("tile unwrap staging identity changed")
        owned = _OwnedTileUnwrapStaging(
            path=stage,
            destination=destination,
            device=identity.st_dev,
            inode=identity.st_ino,
            parent_device=parent_identity.st_dev,
            parent_inode=parent_identity.st_ino,
            staging_directory_fsync_confirmed=staging_fsync_confirmed,
        )
        key = _staging_registry_key(stage)
        with _STAGING_OWNERS_LOCK:
            if key in _STAGING_OWNERS:
                raise ArtifactTileUnwrapExportError(
                    "tile unwrap staging registry collision"
                )
            _STAGING_OWNERS[key] = owned
        return stage
    except Exception as exc:
        try:
            _cleanup_owned_stage(
                stage,
                device=identity.st_dev,
                inode=identity.st_ino,
            )
        except ArtifactTileUnwrapExportError as cleanup_exc:
            raise cleanup_exc from exc
        if isinstance(exc, ArtifactTileUnwrapExportError):
            raise
        raise ArtifactTileUnwrapExportError(str(exc)) from exc


def prepare_staged_tile_unwrap_publication(
    staging_directory: str | os.PathLike[str],
    directory: str | os.PathLike[str],
    *,
    document: ArtifactDocument | None = None,
) -> PreparedTileUnwrapPublication:
    stage = Path(os.path.abspath(os.fspath(Path(staging_directory).expanduser())))
    destination = Path(os.path.abspath(os.fspath(Path(directory).expanduser())))
    key = _staging_registry_key(stage)
    with _STAGING_OWNERS_LOCK:
        owned = _STAGING_OWNERS.get(key)
        if owned is None or owned.destination != destination:
            raise ArtifactTileUnwrapExportError(
                "tile unwrap staging is not owned for this destination"
            )
        _require_current_parent(owned)
        _require_owned_staging_identity(owned)
        if _path_exists_without_following(destination):
            _raise_if_owned_destination_is_visible(owned)
            raise ArtifactTileUnwrapExportError("export destination already exists")
        validate_tile_unwrap_export_package(stage, document=document)
        fingerprint = _capture_tile_unwrap_package_fingerprint(stage)
        nonce = object()
        prepared = PreparedTileUnwrapPublication(
            staging_directory=stage,
            destination=destination,
            _owned=owned,
            _fingerprint=fingerprint,
            _staging_directory_fsync_confirmed=(
                owned.staging_directory_fsync_confirmed
            ),
            _nonce=nonce,
        )
        _PREPARED_PUBLICATIONS[nonce] = prepared
        return prepared


def discard_staged_tile_unwrap_package(
    staging_directory: str | os.PathLike[str],
    directory: str | os.PathLike[str],
) -> bool:
    stage = Path(os.path.abspath(os.fspath(Path(staging_directory).expanduser())))
    destination = Path(os.path.abspath(os.fspath(Path(directory).expanduser())))
    key = _staging_registry_key(stage)
    with _STAGING_OWNERS_LOCK:
        owned = _STAGING_OWNERS.get(key)
        if owned is None or owned.destination != destination:
            return False
        _require_current_parent(owned)
        try:
            _require_owned_staging_identity(owned)
        except ArtifactTileUnwrapExportError:
            _raise_if_owned_destination_is_visible(owned)
            raise
        quarantine = destination.parent / (
            f"{_TILE_UNWRAP_QUARANTINE_PREFIX}{uuid.uuid4().hex}"
        )
        try:
            publish_export_directory_noreplace(stage, quarantine)
        except ArtifactVectorExportError as exc:
            _raise_if_owned_destination_is_visible(owned)
            raise ArtifactTileUnwrapExportError(
                f"cannot quarantine tile unwrap staging: {exc}"
            ) from exc
        try:
            quarantined = quarantine.stat(follow_symlinks=False)
        except OSError as exc:
            raise ArtifactTileUnwrapExportError(
                f"cannot verify quarantined tile unwrap staging: {exc}"
            ) from exc
        if (
            not stat.S_ISDIR(quarantined.st_mode)
            or quarantined.st_dev != owned.device
            or quarantined.st_ino != owned.inode
        ):
            raise ArtifactTileUnwrapExportError(
                "quarantined tile unwrap staging identity changed; path was preserved"
            )
        _STAGING_OWNERS.pop(key, None)
        _invalidate_tile_unwrap_prepared_locked(owned)
    _cleanup_owned_stage(quarantine, device=owned.device, inode=owned.inode)
    return True


def discard_prepared_tile_unwrap_package(
    prepared: PreparedTileUnwrapPublication,
) -> bool:
    if not isinstance(prepared, PreparedTileUnwrapPublication):
        raise ArtifactTileUnwrapExportError(
            "prepared publication must be a PreparedTileUnwrapPublication"
        )
    with _STAGING_OWNERS_LOCK:
        if _PREPARED_PUBLICATIONS.get(prepared._nonce) is not prepared:
            _raise_if_owned_destination_is_visible(prepared._owned)
            raise ArtifactTileUnwrapExportError(
                "prepared tile unwrap publication capability is invalid or consumed"
            )
    return discard_staged_tile_unwrap_package(
        prepared.staging_directory,
        prepared.destination,
    )


def publish_prepared_tile_unwrap_package(
    prepared: PreparedTileUnwrapPublication,
) -> Path:
    """Fast final commit for an exact, fully validated unwrap capability."""

    if not isinstance(prepared, PreparedTileUnwrapPublication):
        raise ArtifactTileUnwrapExportError(
            "prepared publication must be a PreparedTileUnwrapPublication"
        )
    owned = prepared._owned
    key = _staging_registry_key(prepared.staging_directory)
    with _STAGING_OWNERS_LOCK:
        if _PREPARED_PUBLICATIONS.get(prepared._nonce) is not prepared:
            _raise_if_owned_destination_is_visible(owned)
            raise ArtifactTileUnwrapExportError(
                "prepared tile unwrap publication capability is invalid or consumed"
            )
        if _STAGING_OWNERS.get(key) is not owned:
            _raise_if_owned_destination_is_visible(owned)
            raise ArtifactTileUnwrapExportError(
                "tile unwrap staging authority is no longer current"
            )
        _require_current_parent(owned)
        _require_owned_staging_identity(owned)
        if _path_exists_without_following(prepared.destination):
            _raise_if_owned_destination_is_visible(owned)
            raise ArtifactTileUnwrapExportError("export destination already exists")
        if (
            _capture_tile_unwrap_package_fingerprint(prepared.staging_directory)
            != prepared._fingerprint
        ):
            raise ArtifactTileUnwrapExportError(
                "tile unwrap staging package changed after preparation"
            )
        try:
            publish_export_directory_noreplace(
                prepared.staging_directory,
                prepared.destination,
            )
        except ArtifactVectorExportError as exc:
            raise ArtifactTileUnwrapExportError(str(exc)) from exc
        try:
            published_identity = prepared.destination.stat(follow_symlinks=False)
        except OSError as exc:
            raise ArtifactTileUnwrapExportError(
                "tile unwrap export was renamed, but destination identity could "
                f"not be verified: {exc}",
                committed=True,
            ) from exc
        if (
            not stat.S_ISDIR(published_identity.st_mode)
            or published_identity.st_dev != owned.device
            or published_identity.st_ino != owned.inode
        ):
            raise ArtifactTileUnwrapExportError(
                "tile unwrap export destination is not the authorized staging inode",
                committed=True,
            )
        _STAGING_OWNERS.pop(key, None)
        _invalidate_tile_unwrap_prepared_locked(owned)
    try:
        parent_fsync_confirmed = fsync_export_directory(
            prepared.destination.parent
        )
    except OSError as exc:
        raise ArtifactTileUnwrapExportError(
            "tile unwrap export was published, but directory fsync failed; "
            f"crash durability is uncertain: {exc}",
            committed=True,
        ) from exc
    if (
        not prepared._staging_directory_fsync_confirmed
        or not parent_fsync_confirmed
    ):
        raise ArtifactTileUnwrapExportError(
            "tile unwrap export was published, but directory fsync is unsupported; "
            "crash durability is uncertain",
            committed=True,
        )
    return prepared.destination


def export_tile_unwrap_package(
    directory: str | os.PathLike[str],
    document: ArtifactDocument,
    record_id: str,
    unwrap: TileUnwrapMesh,
) -> TileUnwrapExportPublication:
    """Build, verify, and atomically publish one no-overwrite directory."""

    stage = stage_tile_unwrap_package(directory, document, record_id, unwrap)
    prepared: PreparedTileUnwrapPublication | None = None
    try:
        prepared = prepare_staged_tile_unwrap_publication(
            stage,
            directory,
            document=document,
        )
        destination = publish_prepared_tile_unwrap_package(prepared)
        return TileUnwrapExportPublication(
            destination=destination,
            durability_confirmed=True,
        )
    except ArtifactTileUnwrapExportError as exc:
        if exc.committed:
            destination = Path(
                os.path.abspath(os.fspath(Path(directory).expanduser()))
            )
            validate_tile_unwrap_export_package(destination, document=document)
            return TileUnwrapExportPublication(
                destination=destination,
                durability_confirmed=False,
            )
        if not exc.committed:
            try:
                if prepared is not None:
                    discard_prepared_tile_unwrap_package(prepared)
                else:
                    discard_staged_tile_unwrap_package(stage, directory)
            except ArtifactTileUnwrapExportError as cleanup_exc:
                raise cleanup_exc from exc
        raise


__all__ = [
    "ArtifactTileUnwrapExportError",
    "MAX_TILE_UNWRAP_OBJ_BYTES",
    "MAX_TILE_UNWRAP_SIDECAR_BYTES",
    "MAX_TILE_UNWRAP_SVG_BYTES",
    "TILE_UNWRAP_EXPORT_DIRECTORY_SUFFIX",
    "TILE_UNWRAP_EXPORT_FORMAT",
    "TILE_UNWRAP_EXPORT_OBJ_NAME",
    "TILE_UNWRAP_EXPORT_PAYLOAD_NAME",
    "TILE_UNWRAP_EXPORT_SCHEMA_VERSION",
    "TILE_UNWRAP_1_2_EXPORT_SCHEMA_VERSION",
    "TILE_UNWRAP_1_3_EXPORT_SCHEMA_VERSION",
    "TILE_UNWRAP_1_4_EXPORT_SCHEMA_VERSION",
    "TILE_UNWRAP_LEGACY_EXPORT_SCHEMA_VERSION",
    "TILE_UNWRAP_EXPORT_SIDECAR_NAME",
    "TILE_UNWRAP_EXPORT_SVG_NAME",
    "TileUnwrapExportBundle",
    "TileUnwrapExportPublication",
    "PreparedTileUnwrapPublication",
    "build_tile_unwrap_export",
    "discard_prepared_tile_unwrap_package",
    "discard_staged_tile_unwrap_package",
    "export_tile_unwrap_package",
    "prepare_staged_tile_unwrap_publication",
    "publish_prepared_tile_unwrap_package",
    "stage_tile_unwrap_package",
    "validate_tile_unwrap_export_bytes",
    "validate_tile_unwrap_export_package",
]
