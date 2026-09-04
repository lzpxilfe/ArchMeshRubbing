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
from typing import AbstractSet, Any, Mapping, Sequence
import uuid
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape as xml_escape

import numpy as np

from .alignment_utils import (
    MATRIX_ATOL,
    compose_align_matrices,
    scene_trs_matrix_about_pivot,
)
from .artifact_axis_alignment import (
    AXIS_ALIGN_RECIPE_KIND,
    ArtifactAxisAlignmentError,
    verify_axis_alignment_matrix,
)
from .artifact_outline_extractor import (
    OUTLINE_ALGORITHM_VERSION,
    OUTLINE_ALGORITHM_VERSIONS,
    OUTLINE_GRID_CLOSING_RADIUS_CELLS,
    OUTLINE_LEGACY_ALGORITHM_VERSION,
    REVIEWED_OUTLINE_BACKENDS,
)
from .drawing_style import (
    CENTER_AXIS,
    GROOVE_TROUGH_BREAK_COUNT,
    GROOVE_TROUGH_BREAK_MM,
    TECHNIQUE_GROOVE_EDGE,
    TECHNIQUE_GROOVE_TROUGH,
    DrawingStyleError,
    DrawingStylePreset,
    preset_claim as drawing_style_preset_claim,
    preset_from_claim as drawing_style_preset_from_claim,
    resolve_preset as resolve_drawing_style_preset,
    line_kind_for_record_role,
)
from .drawing_svg import (
    Placement,
    SVGRenderError,
    axis_profile_chord,
    broken_chord,
    center_axis_segment,
    hatch_pattern_elements,
    hatched_kinds,
    layer_elements,
    path_element,
)
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
    VectorPath,
    VectorRecordKind,
    validate_vector_payload_recipe_contract,
    validate_vector_recipe,
    vector_payload_from_record,
)
from .mesh_import_recipe import (
    MeshImportRecipeError,
    RUNTIME_POLICY_RECORD_ONLY,
    validate_mesh_import_recipe,
)
from .mesh_admission import (
    MeshAdmissionError,
    decoded_admission_from_counts,
    validate_mesh_admission_receipt,
)
from .source_identity import PRIMARY_FILE_IDENTITY_SCOPE


VECTOR_EXPORT_FORMAT = "archmeshrubbing_vector_export"
_CURRENT_VECTOR_EXPORT_SCHEMA_VERSION = "1.4.0"
VECTOR_EXPORT_SCHEMA_VERSION = _CURRENT_VECTOR_EXPORT_SCHEMA_VERSION
#: 1.1.0 introduced the current provenance contract (import admission, axis
#: Align); 1.2.0 is 1.1.0 plus outline algorithm 1.1.0 - the grid closing -
#: and its four QC keys; 1.3.0 is 1.2.0 plus a user drawing style preset's
#: definition in the presentation claim; 1.4.0 is 1.3.0 with the five
#: technique line kinds in that definition's closed key set.  All four carry
#: the current contract; 1.0.0 is legacy.
_CURRENT_CONTRACT_VECTOR_EXPORT_SCHEMA_VERSIONS = frozenset(
    {"1.1.0", "1.2.0", "1.3.0", "1.4.0"}
)
#: The sidecars that can carry an outline computed with the grid closing.
_GRID_CLOSING_VECTOR_EXPORT_SCHEMA_VERSIONS = frozenset({"1.2.0", "1.3.0", "1.4.0"})
SUPPORTED_VECTOR_EXPORT_SCHEMA_VERSIONS = frozenset(
    {"1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0"}
)
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
_AXIS_ALIGN_RECIPE_KEYS = frozenset(
    {
        "bottom_center_mm_decimal",
        "bottom_normal_unit_decimal",
        "bottom_record_id",
        "convention",
        "kind",
        "top_center_mm_decimal",
        "top_normal_unit_decimal",
        "top_record_id",
    }
)
_MANUAL_ALIGN_RECIPE_KIND = "manual_scene_trs_delta"
_NON_ROOT_ALIGN_RECIPE_KINDS = frozenset(
    {_MANUAL_ALIGN_RECIPE_KIND, AXIS_ALIGN_RECIPE_KIND}
)
# The union of every recipe key any supported kind uses.  `_public_mapping`
# filters against this before validation, so a key missing here is dropped in
# silence rather than rejected: the recipe would then fail its own exact-key
# check with a confusing "missing field" instead of an unsupported-kind error.
_PUBLIC_ALIGN_RECIPE_KEYS = frozenset(
    {"convention", "kind", "pivot_mm", "rotation_deg", "translation_mm"}
) | _AXIS_ALIGN_RECIPE_KEYS
# Likewise for QC.  A computed alignment records what made it believable, and
# that evidence has to survive into the package rather than being filtered out
# on the way.
_AXIS_ALIGN_QC_KEYS = frozenset(
    {
        "axis_tilt_corrected_deg",
        "center_separation_mm",
        "circle_normal_disagreement_deg",
        "proper_rigid",
    }
)
_PUBLIC_ALIGN_QC_KEYS = frozenset({"proper_rigid", "rigid"}) | _AXIS_ALIGN_QC_KEYS
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
        "resolver_profile",
        "runtime_lock_sha256",
        "sanitizer",
        "scene_merge",
        "source_manifest",
    }
)
_PUBLIC_GEOMETRY_QC_KEYS = frozenset(
    {"face_count", "finite_vertices", "import_admission", "vertex_count"}
)
_LEGACY_PUBLIC_GEOMETRY_QC_KEYS = frozenset(
    {"face_count", "finite_vertices", "vertex_count"}
)
_VECTOR_PAYLOAD_QC_KEYS = frozenset(
    {
        "bounds_mm",
        "closed_path_count",
        "coordinate_space",
        "finite",
        "path_count",
        "payload_sha256",
        "point_count",
        "total_length_mm",
        "total_length_rounding_decimal_places",
        "unit",
    }
)
_CUTLINE_RECORD_QC_KEYS = frozenset(
    {
        "candidate_face_count",
        "classification_tolerance_mm",
        "collinear_point_removal_count",
        "coplanar_face_count",
        "duplicate_segment_count",
        "input_face_count",
        "input_vertex_count",
        "intersected_face_count",
        "max_endpoint_snap_mm",
        "max_plane_residual_mm",
        "non_manifold_junction_count",
        "on_plane_edge_face_count",
        "point_tangent_count",
        "raw_segment_count",
        "stitch_tolerance_mm",
        "unique_segment_count",
    }
)
_OUTLINE_RECORD_QC_KEYS = frozenset(
    {
        "all_projected_faces_included",
        "backend_geos_version",
        "backend_shapely_version",
        "component_count",
        "face_chunk_count",
        "fixed_grid_triangle_count",
        "grid_area_delta_mm2",
        "grid_collapsed_triangle_count",
        "grid_component_merge_count",
        "grid_component_split_count",
        "grid_origin_index_uv",
        "grid_snap_axis_upper_bound_mm",
        "grid_snap_error_contract",
        "grid_snap_radial_upper_bound_squared_mm2",
        "hole_count",
        "input_face_count",
        "input_vertex_count",
        "outline_area_mm2",
        "outline_collinear_point_removal_count",
        "outline_perimeter_mm",
        "outline_topology",
        "output_grid_residual_max_mm",
        "precision_grid_mm",
        "projected_degenerate_triangle_count",
        "projected_non_degenerate_triangle_count",
        "sampling_applied",
        "topology_valid",
        "unsnapped_area_mm2",
        "unsnapped_comparison_status",
        "unsnapped_component_count",
        "view",
    }
)
#: Present exactly when the outline was computed with the grid closing
#: (outline algorithm 1.1.0), which only a 1.2.0 sidecar can carry.
_OUTLINE_CLOSING_QC_KEYS = frozenset(
    {
        "grid_closing_area_delta_mm2",
        "grid_closing_component_merge_count",
        "grid_closing_hole_fill_count",
        "grid_closing_radius_cells",
    }
)
_PRODUCTION_CUTLINE_ALGORITHM = "archmeshrubbing.triangle_plane_cutline"
_PRODUCTION_OUTLINE_ALGORITHM = "archmeshrubbing.projected_triangle_union"
_PRODUCTION_VECTOR_ALGORITHM_VERSION = "1.0.0"
_PRODUCTION_ALGORITHM_VERSIONS: Mapping[str, frozenset[str]] = {
    _PRODUCTION_CUTLINE_ALGORITHM: frozenset({_PRODUCTION_VECTOR_ALGORITHM_VERSION}),
    _PRODUCTION_OUTLINE_ALGORITHM: frozenset(OUTLINE_ALGORITHM_VERSIONS),
}
_SNAP_ERROR_CONTRACTS: Mapping[str, tuple[str, float]] = {
    OUTLINE_LEGACY_ALGORITHM_VERSION: ("axis<=grid/2;radial<=grid/sqrt(2)", 0.5),
    OUTLINE_ALGORITHM_VERSION: (
        "axis<=1.5*grid;radial<=1.5*grid*sqrt(2)",
        OUTLINE_GRID_CLOSING_RADIUS_CELLS + 0.5,
    ),
}
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
    expected: AbstractSet[str],
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
    style_preset: str | DrawingStylePreset | None = None
    """Drawing style preset id or preset, or `None` for the single-weight drawing.

    `None` is the default deliberately.  A drawing exported before presets
    existed must keep rendering to the same bytes, or every package already
    written would stop verifying against its own sidecar.
    """
    show_center_axis: bool = False
    """Draw the artifact's rotation axis as a centre line.

    Off by default, and meaningful only where a preset separates line kinds.
    The caller decides whether the active Align actually established an axis;
    drawing one otherwise would put a claim on the page that nothing backs.
    """

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
        if not isinstance(self.show_center_axis, bool):
            raise ArtifactVectorExportError("show_center_axis must be a boolean")
        if self.show_center_axis and self.style_preset is None:
            raise ArtifactVectorExportError(
                "show_center_axis needs a style_preset; without one every line is "
                "drawn at the same weight and a centre line would be "
                "indistinguishable from the outline"
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
        if self.style_preset is not None:
            try:
                preset = resolve_drawing_style_preset(self.style_preset)
            except DrawingStyleError as exc:
                raise ArtifactVectorExportError(str(exc)) from exc
            # A layer's own weight replaces the single stroke width, so the
            # clipping margin has to clear the widest line the preset draws.
            widest = max(
                style.stroke_width_mm for style in preset.lines.values()
            )
            if margin < widest / 2.0:
                raise ArtifactVectorExportError(
                    f"margin_mm must be at least half of the widest stroke in "
                    f"preset {preset.preset_id!r} ({widest} mm) to prevent clipping"
                )
            # A registered preset stays a name; a user preset is kept whole,
            # since nothing else could resolve it again.
            object.__setattr__(
                self, "style_preset", preset if preset.is_user else preset.preset_id
            )


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


def _closed_public_mapping(
    value: object,
    allowed_keys: frozenset[str],
    *,
    model_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactVectorExportError(f"{model_name} must be an object")
    unknown = sorted(set(value) - set(allowed_keys))
    if unknown:
        raise ArtifactVectorExportError(
            f"{model_name} has unknown fields: {', '.join(unknown)}"
        )
    return value


def _public_mesh_import_recipe(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete path-free executable parser contract.

    Strict recipes contain runtime identity and dependency-policy fields which
    are necessary to reproduce the source geometry.  Unknown document-only
    extensions remain private, while every recognized recipe field is kept and
    the resulting closed contract is validated before it enters provenance.

    The recipe describes the past import, not this export, so it is validated
    under `record_only`.  Requiring the current runtime here would have made a
    project that reopened correctly under a newer parser still refuse to export
    from itself, which is the same archive defect one step later.
    """

    public = _public_mapping(value, _PUBLIC_GEOMETRY_RECIPE_KEYS)
    try:
        validate_mesh_import_recipe(
            public,
            allow_legacy=True,
            runtime_policy=RUNTIME_POLICY_RECORD_ONLY,
        )
    except MeshImportRecipeError as exc:
        raise ArtifactVectorExportError(
            f"invalid public mesh import recipe: {exc}"
        ) from exc
    return public


def _public_align_value(align: AlignRevision) -> dict[str, Any]:
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


def _public_align_revision(
    document: ArtifactDocument,
    record: DerivedRecord,
) -> dict[str, Any]:
    return _public_align_value(
        document.align_revision_index[record.align_revision_id]
    )


def _public_align_ancestry(
    document: ArtifactDocument,
    record: DerivedRecord,
) -> list[dict[str, Any]]:
    """Return the exact root-to-active Align chain for offline recomputation."""

    revisions = document.align_revision_index
    current_id: str | None = record.align_revision_id
    reverse_chain: list[AlignRevision] = []
    seen: set[str] = set()
    while current_id is not None:
        if current_id in seen:
            raise ArtifactVectorExportError("Align ancestry contains a cycle")
        revision = revisions.get(current_id)
        if revision is None:
            raise ArtifactVectorExportError(
                f"Align ancestry is missing revision {current_id!r}"
            )
        seen.add(current_id)
        reverse_chain.append(revision)
        if len(reverse_chain) > 4096:
            raise ArtifactVectorExportError(
                "Align ancestry exceeds the 4096-revision safety limit"
            )
        current_id = revision.parent_id
    return [_public_align_value(item) for item in reversed(reverse_chain)]


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


def _public_geometry_revision(
    document: ArtifactDocument,
    record: DerivedRecord,
    *,
    include_current_contract: bool,
) -> dict[str, Any]:
    geometry = document.geometry_revision_index[record.geometry_revision_id]
    data = geometry.to_dict()
    return {
        "created_at": data["created_at"],
        "geometry_hash_scope": data["geometry_hash_scope"],
        "geometry_sha256": data["geometry_sha256"],
        "id": data["id"],
        "import_recipe": _public_mesh_import_recipe(data["import_recipe"]),
        "operator": data["operator"],
        "qc": _public_mapping(
            data["qc"],
            (
                _PUBLIC_GEOMETRY_QC_KEYS
                if include_current_contract
                else _LEGACY_PUBLIC_GEOMETRY_QC_KEYS
            ),
        ),
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


def _provenance(
    document: ArtifactDocument,
    record: DerivedRecord,
    *,
    include_current_contract: bool,
) -> dict[str, Any]:
    provenance = {
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
        "geometry_revision": _public_geometry_revision(
            document,
            record,
            include_current_contract=include_current_contract,
        ),
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
    if include_current_contract:
        provenance["align_ancestry"] = _public_align_ancestry(document, record)
    return provenance


def build_public_export_provenance(
    document: ArtifactDocument,
    record: DerivedRecord,
    *,
    include_current_contract: bool = True,
) -> dict[str, Any]:
    """Build the shared, path-sanitized provenance shape for public exports."""

    return _provenance(
        document,
        record,
        include_current_contract=include_current_contract,
    )


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


def _placement(
    bounds: tuple[float, float, float, float],
    options: VectorSVGOptions,
) -> Placement:
    """Return the 1:1 placement of a record inside its own margined page."""

    return Placement(
        content_bounds_mm=bounds,
        origin_mm=(options.margin_mm, options.margin_mm),
        scale_denominator=1.0,
    )


def _path_element(
    path: VectorPath,
    *,
    bounds: tuple[float, float, float, float],
    options: VectorSVGOptions,
    fill: str | None = None,
) -> str:
    """Return one `<path>` element, in millimetres of the drawing's own frame."""

    try:
        return path_element(
            path_id=path.id,
            role=path.role,
            closed=path.closed,
            points_mm=path.points_mm,
            placement=_placement(bounds, options),
            fill=fill,
        )
    except SVGRenderError as exc:
        raise ArtifactVectorExportError(str(exc)) from exc


def _paths_by_line_kind(
    payload: VectorGeometryPayload,
    *,
    center_axis: bool = False,
) -> dict[str, list[VectorPath]]:
    by_kind: dict[str, list[VectorPath]] = {}
    for path in payload.paths:
        try:
            kind = line_kind_for_record_role(path.role)
        except DrawingStyleError as exc:
            raise ArtifactVectorExportError(str(exc)) from exc
        by_kind.setdefault(kind, []).append(path)
    if center_axis:
        axis_path = center_axis_vector_path(payload)
        if axis_path is not None:
            by_kind.setdefault(CENTER_AXIS, []).append(axis_path)
    return by_kind


def center_axis_vector_path(payload: VectorGeometryPayload) -> VectorPath | None:
    """Return the rotation axis drawn across this record's own content.

    The axis is not a measurement and has no record of its own: it is where the
    active Align put the artifact.  Deciding *whether* an alignment established
    an axis belongs to the caller; this only draws the line once that is known.
    """

    try:
        segment = center_axis_segment(payload.frame.to_dict(), _payload_bounds(payload))
    except SVGRenderError as exc:
        raise ArtifactVectorExportError(str(exc)) from exc
    if segment is None:
        return None
    return VectorPath(
        id="center-axis",
        role=CENTER_AXIS,
        closed=False,
        points_mm=segment,
    )


def profile_groove_vector_paths(
    payload: VectorGeometryPayload,
    grooves: Sequence[Any],
    *,
    record_id: str,
) -> dict[str, list[VectorPath]]:
    """Draw one groove reading across a record that shows the artifact's side.

    Each groove becomes three lines: the two raised edges as solid 직선, and
    the recessed bottom as a 간선, a straight line broken a few times.  That is
    what a groove is - one place that goes in and two that stand out - so the
    drawing carries the same count the surface does.

    Returns nothing at all for a plane the axis does not lie in.  A plan view
    sees a circumferential groove as a circle, not a line, and a foreshortened
    plane would give it a width the artifact does not have there.
    """

    frame = payload.frame.to_dict()
    by_kind: dict[str, list[VectorPath]] = {}
    for index, groove in enumerate(grooves):
        for role, height_um, radius_um, broken in (
            (
                TECHNIQUE_GROOVE_EDGE,
                groove.lower_edge_height_um,
                groove.lower_edge_radius_um,
                False,
            ),
            (
                TECHNIQUE_GROOVE_TROUGH,
                groove.trough_height_um,
                groove.trough_radius_um,
                True,
            ),
            (
                TECHNIQUE_GROOVE_EDGE,
                groove.upper_edge_height_um,
                groove.upper_edge_radius_um,
                False,
            ),
        ):
            try:
                chord = axis_profile_chord(
                    frame,
                    height_mm=float(height_um) / 1000.0,
                    radius_mm=float(radius_um) / 1000.0,
                )
            except SVGRenderError as exc:
                raise ArtifactVectorExportError(str(exc)) from exc
            if chord is None:
                return {}
            start, end = chord
            if broken:
                # Break each half of the chord about the axis rather than the
                # chord as a whole.  A 좌 반입면 draws one half, and it has to
                # carry the breaks a drafter would put in the line they drew;
                # breaking the whole chord would leave half of them on the
                # half that gets clipped away.  On a full elevation the two
                # halves then break symmetrically about the centre.
                centre = (
                    0.5 * (start[0] + end[0]),
                    0.5 * (start[1] + end[1]),
                )
                try:
                    pieces = broken_chord(
                        centre,
                        start,
                        break_count=GROOVE_TROUGH_BREAK_COUNT,
                        break_mm=GROOVE_TROUGH_BREAK_MM,
                    ) + broken_chord(
                        centre,
                        end,
                        break_count=GROOVE_TROUGH_BREAK_COUNT,
                        break_mm=GROOVE_TROUGH_BREAK_MM,
                    )
                except SVGRenderError as exc:
                    raise ArtifactVectorExportError(str(exc)) from exc
            else:
                pieces = [(start, end)]
            for piece_index, (head, tail) in enumerate(pieces):
                marker = "trough" if broken else f"edge{height_um}"
                by_kind.setdefault(role, []).append(
                    VectorPath(
                        id=f"groove:{record_id}:{index}:{marker}:{piece_index}",
                        role=role,
                        closed=False,
                        points_mm=(head, tail),
                    )
                )
    return by_kind


def _styled_layers(
    payload: VectorGeometryPayload,
    *,
    bounds: tuple[float, float, float, float],
    options: VectorSVGOptions,
) -> list[str]:
    """Return the drawing body as one group per line kind."""

    assert options.style_preset is not None
    preset = resolve_drawing_style_preset(options.style_preset)
    by_kind = _paths_by_line_kind(payload, center_axis=options.show_center_axis)
    hatched = hatched_kinds(by_kind, preset=preset)

    try:
        lines: list[str] = []
        if hatched:
            lines.append("  <defs>")
            lines.extend(
                hatch_pattern_elements(
                    hatched,
                    preset=preset,
                    color=options.stroke_color,
                    indent="    ",
                )
            )
            lines.append("  </defs>")
        lines.append(
            '  <g id="measured-vectors" fill="none" '
            f'stroke="{options.stroke_color}" '
            'stroke-linecap="round" stroke-linejoin="round">'
        )
        lines.extend(
            layer_elements(
                by_kind,
                preset=preset,
                placement=_placement(bounds, options),
                hatched=hatched,
                indent="    ",
            )
        )
        lines.append("  </g>")
    except SVGRenderError as exc:
        raise ArtifactVectorExportError(str(exc)) from exc
    return lines


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
    ]
    if options.style_preset is None:
        lines.append(
            '  <g id="measured-vectors" fill="none" '
            f'stroke="{options.stroke_color}" stroke-width="{stroke_token}" '
            'stroke-linecap="round" stroke-linejoin="round">'
        )
        for path in payload.paths:
            lines.append(
                f"    {_path_element(path, bounds=bounds, options=options)}"
            )
        lines.append("  </g>")
    else:
        lines.extend(_styled_layers(payload, bounds=bounds, options=options))
    lines.append("</svg>")
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
    presentation = {
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
    if options.style_preset is not None:
        # Recorded only for a styled drawing, so an unstyled sidecar keeps the
        # exact key set and exact bytes it had before presets existed.
        preset = resolve_drawing_style_preset(options.style_preset)
        presentation["style_preset"] = drawing_style_preset_claim(preset)
        presentation["show_center_axis"] = options.show_center_axis
    return presentation


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
    provenance = _provenance(
        document,
        record,
        include_current_contract=(
            VECTOR_EXPORT_SCHEMA_VERSION == _CURRENT_VECTOR_EXPORT_SCHEMA_VERSION
        ),
    )
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


def _validated_style_preset(
    value: object, *, schema_version: str
) -> "str | DrawingStylePreset":
    """Return the preset a styled sidecar claims, proving it still holds.

    The sidecar records the preset's canonical digest, so a preset whose values
    were edited after the drawing was made is caught here rather than silently
    re-rendering the drawing with different line weights.  A user preset
    carries its full definition over the whole line-kind vocabulary, which
    only the current sidecar's closed key set can hold: an older one lacks
    the kinds added since.
    """

    if (
        isinstance(value, Mapping)
        and "definition" in value
        and schema_version != _CURRENT_VECTOR_EXPORT_SCHEMA_VERSION
    ):
        raise ArtifactVectorExportError(
            f"a vector export before {_CURRENT_VECTOR_EXPORT_SCHEMA_VERSION} cannot "
            "carry a user drawing style preset"
        )
    try:
        preset = drawing_style_preset_from_claim(value)
    except DrawingStyleError as exc:
        raise ArtifactVectorExportError(str(exc)) from exc
    return preset if preset.is_user else preset.preset_id


def _options_from_presentation(
    value: object, *, schema_version: str = _CURRENT_VECTOR_EXPORT_SCHEMA_VERSION
) -> tuple[VectorSVGOptions, Mapping[str, Any]]:
    base_keys = {
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
    }
    # Two closed contracts, not one contract with an optional field: a drawing
    # is either styled or it is not, and the presence of the key says which.
    styled = isinstance(value, Mapping) and "style_preset" in value
    presentation = _exact_keys(
        value,
        (base_keys | {"show_center_axis", "style_preset"}) if styled else base_keys,
        model_name="presentation",
    )
    if presentation["physical_scale"] != "1:1" or presentation["unit"] != "mm":
        raise ArtifactVectorExportError("presentation must declare 1:1 millimetres")
    style_preset = (
        _validated_style_preset(
            presentation["style_preset"], schema_version=schema_version
        )
        if styled
        else None
    )
    show_center_axis = bool(presentation["show_center_axis"]) if styled else False
    if styled and not isinstance(presentation["show_center_axis"], bool):
        raise ArtifactVectorExportError(
            "presentation.show_center_axis must be a boolean"
        )
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
        style_preset=style_preset,
        show_center_axis=show_center_axis,
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


def _validated_public_align_revision(
    value: object,
    *,
    model_name: str,
) -> tuple[Mapping[str, Any], AlignRevision]:
    align_value = _exact_keys(
        value,
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
        model_name=model_name,
    )
    _closed_public_mapping(
        align_value["recipe"],
        _PUBLIC_ALIGN_RECIPE_KEYS,
        model_name=f"{model_name}.recipe",
    )
    _closed_public_mapping(
        align_value["qc"],
        _PUBLIC_ALIGN_QC_KEYS,
        model_name=f"{model_name}.qc",
    )
    try:
        align = AlignRevision.from_dict(dict(align_value) | {"extensions": {}})
    except ArtifactDocumentError as exc:
        raise ArtifactVectorExportError(
            f"invalid {model_name}: {exc}"
        ) from exc
    return align_value, align


def _validate_provenance_shape(
    value: object,
    *,
    require_current_contract: bool,
) -> Mapping[str, Any]:
    expected_keys = {
        "align_revision",
        "dependency_closure",
        "document",
        "geometry_revision",
        "record",
        "source_assets",
        "source_metadata_revision",
    }
    if require_current_contract:
        expected_keys.add("align_ancestry")
    provenance = _exact_keys(
        value,
        expected_keys,
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
    _align_value, align = _validated_public_align_revision(
        provenance["align_revision"],
        model_name="provenance.align_revision",
    )
    geometry_value = _exact_keys(
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
    )
    _closed_public_mapping(
        geometry_value["qc"],
        _PUBLIC_GEOMETRY_QC_KEYS,
        model_name="provenance.geometry_revision.qc",
    )
    try:
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
            dict(geometry_value) | {"topology_map_ref": None, "extensions": {}}
        )
    except ArtifactDocumentError as exc:
        raise ArtifactVectorExportError(f"invalid revision provenance: {exc}") from exc
    public_import_recipe = _public_mesh_import_recipe(geometry.import_recipe)
    if dict(geometry.import_recipe) != public_import_recipe:
        raise ArtifactVectorExportError(
            "provenance geometry import_recipe contains non-public fields"
        )
    admission = geometry.qc.get("import_admission")
    if admission is not None:
        try:
            admission_receipt = validate_mesh_admission_receipt(admission)
        except MeshAdmissionError as exc:
            raise ArtifactVectorExportError(
                f"invalid provenance mesh admission receipt: {exc}"
            ) from exc
        accepted = admission_receipt["accepted"]
        if accepted["geometry_sha256"] != geometry.geometry_sha256:
            raise ArtifactVectorExportError(
                "provenance mesh admission does not match geometry SHA-256"
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
    try:
        import_execution = validate_mesh_import_recipe(
            geometry.import_recipe,
            allow_legacy=True,
            runtime_policy=RUNTIME_POLICY_RECORD_ONLY,
        )
    except MeshImportRecipeError as exc:
        raise ArtifactVectorExportError(
            f"invalid provenance mesh import recipe: {exc}"
        ) from exc
    if import_execution.source_manifest is not None:
        if len(assets) != 1:
            raise ArtifactVectorExportError(
                "closed source manifest currently requires one primary source asset"
            )
        primary = import_execution.source_manifest.primary_entry
        asset = assets[0]
        if (
            asset["id"] != primary.content_id
            or asset["sha256"] != primary.sha256
            or asset["size_bytes"] != primary.size_bytes
        ):
            raise ArtifactVectorExportError(
                "source manifest primary entry does not match source asset provenance"
            )
    return provenance


def validate_public_export_provenance(value: object) -> Mapping[str, Any]:
    """Validate either supported shared public provenance contract."""

    if isinstance(value, Mapping) and "align_ancestry" in value:
        return validate_current_public_export_provenance(value)
    return _validate_provenance_shape(value, require_current_contract=False)


def validate_legacy_public_export_provenance(
    value: object,
) -> Mapping[str, Any]:
    """Validate the immutable, exact public provenance shape used by 1.0."""

    return _validate_provenance_shape(value, require_current_contract=False)


def validate_current_public_export_provenance(
    value: object,
) -> Mapping[str, Any]:
    """Validate the closed provenance contract used by current 1.1 exports."""

    provenance = _validate_provenance_shape(value, require_current_contract=True)
    _validate_current_vector_provenance(provenance)
    return provenance


def _strict_nonnegative_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ArtifactVectorExportError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _strict_vec3(value: object, *, field_name: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ArtifactVectorExportError(f"{field_name} must contain three numbers")
    for index, item in enumerate(value):
        _finite_number(item, field_name=f"{field_name}[{index}]")


def _validate_current_vector_provenance(
    provenance: Mapping[str, Any],
) -> None:
    active_align = provenance["align_revision"]
    assert isinstance(active_align, Mapping)
    ancestry = provenance["align_ancestry"]
    if (
        not isinstance(ancestry, list)
        or not ancestry
        or len(ancestry) > 4096
    ):
        raise ArtifactVectorExportError(
            "provenance.align_ancestry must contain 1 to 4096 revisions"
        )

    parsed_chain: list[AlignRevision] = []
    public_chain: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(ancestry):
        model_name = f"provenance.align_ancestry[{index}]"
        public, revision = _validated_public_align_revision(
            item,
            model_name=model_name,
        )
        if revision.id in seen_ids:
            raise ArtifactVectorExportError(
                "provenance Align ancestry revision IDs must be unique"
            )
        seen_ids.add(revision.id)
        recipe_value = public["recipe"]
        assert isinstance(recipe_value, Mapping)
        recipe_kind = recipe_value.get("kind")
        # QC keys depend on the recipe: a manual drag establishes nothing but
        # rigidity, while a computed alignment carries the evidence that made it
        # believable.  Validating QC against a fixed key set would have silently
        # dropped that evidence on its way into the package.
        _validate_align_qc(public["qc"], recipe_kind, model_name=f"{model_name}.qc")
        if index == 0:
            _exact_keys(
                recipe_value,
                {"kind"},
                model_name=f"{model_name}.recipe",
            )
            if recipe_kind != "initial_identity":
                raise ArtifactVectorExportError(
                    "provenance Align ancestry must start at initial_identity"
                )
            if revision.parent_id is not None:
                raise ArtifactVectorExportError(
                    "initial Align ancestry revision must not have a parent"
                )
            if not np.array_equal(revision.matrix, np.eye(4, dtype=np.float64)):
                raise ArtifactVectorExportError(
                    "initial Align ancestry matrix must be identity"
                )
        else:
            if recipe_kind not in _NON_ROOT_ALIGN_RECIPE_KINDS:
                supported = ", ".join(sorted(_NON_ROOT_ALIGN_RECIPE_KINDS))
                raise ArtifactVectorExportError(
                    f"non-root Align ancestry revisions must be one of: {supported}"
                )
            parent = parsed_chain[-1]
            if revision.parent_id != parent.id:
                raise ArtifactVectorExportError(
                    "provenance Align ancestry parent chain is not exact"
                )
            if (
                revision.source_metadata_revision_id
                != parent.source_metadata_revision_id
            ):
                raise ArtifactVectorExportError(
                    "provenance Align ancestry changes source metadata"
                )
            # Every non-root revision is re-derived from its own recipe and
            # compared against the matrix it stores.  The recipe is never taken
            # on trust, whichever kind it is.
            if recipe_kind == AXIS_ALIGN_RECIPE_KIND:
                axis = _exact_keys(
                    recipe_value,
                    _AXIS_ALIGN_RECIPE_KEYS,
                    model_name=f"{model_name}.recipe",
                )
                try:
                    verify_axis_alignment_matrix(
                        recipe=axis,
                        parent_matrix=parent.matrix,
                        matrix=revision.matrix,
                    )
                except ArtifactAxisAlignmentError as exc:
                    raise ArtifactVectorExportError(
                        f"cannot verify {model_name}: {exc}"
                    ) from exc
            else:
                manual = _exact_keys(
                    recipe_value,
                    {
                        "convention",
                        "kind",
                        "pivot_mm",
                        "rotation_deg",
                        "translation_mm",
                    },
                    model_name=f"{model_name}.recipe",
                )
                if manual["convention"] != "delta @ parent":
                    raise ArtifactVectorExportError(
                        "manual Align ancestry convention is invalid"
                    )
                for key in ("pivot_mm", "rotation_deg", "translation_mm"):
                    _strict_vec3(
                        manual[key],
                        field_name=f"{model_name}.recipe.{key}",
                    )
                try:
                    delta = scene_trs_matrix_about_pivot(
                        manual["translation_mm"],
                        manual["rotation_deg"],
                        1.0,
                        manual["pivot_mm"],
                    )
                    recomputed = compose_align_matrices(delta, parent.matrix)
                except (TypeError, ValueError) as exc:
                    raise ArtifactVectorExportError(
                        f"cannot recompute {model_name}: {exc}"
                    ) from exc
                if not np.allclose(
                    revision.matrix,
                    recomputed,
                    rtol=0.0,
                    atol=MATRIX_ATOL,
                ):
                    raise ArtifactVectorExportError(
                        "provenance Align ancestry matrix does not match its delta recipe"
                    )
        public_chain.append(public)
        parsed_chain.append(revision)

    if dict(active_align) != dict(public_chain[-1]):
        raise ArtifactVectorExportError(
            "provenance.align_revision must exactly match the last Align ancestry entry"
        )
    active_revision = parsed_chain[-1]
    document = provenance["document"]
    record = provenance["record"]
    metadata = provenance["source_metadata_revision"]
    assert isinstance(document, Mapping)
    assert isinstance(record, Mapping)
    assert isinstance(metadata, Mapping)
    if (
        document["active_align_revision_id"] != active_revision.id
        or record["align_revision_id"] != active_revision.id
    ):
        raise ArtifactVectorExportError(
            "active/record Align IDs do not match the Align ancestry tip"
        )
    if active_revision.source_metadata_revision_id != metadata["id"]:
        raise ArtifactVectorExportError(
            "Align ancestry tip does not match source metadata provenance"
        )

    geometry = provenance["geometry_revision"]
    assert isinstance(geometry, Mapping)
    geometry_qc = _exact_keys(
        geometry["qc"],
        {"face_count", "finite_vertices", "import_admission", "vertex_count"},
        model_name="provenance.geometry_revision.qc",
    )
    face_count = _strict_nonnegative_int(
        geometry_qc["face_count"],
        field_name="provenance.geometry_revision.qc.face_count",
    )
    vertex_count = _strict_nonnegative_int(
        geometry_qc["vertex_count"],
        field_name="provenance.geometry_revision.qc.vertex_count",
    )
    if geometry_qc["finite_vertices"] is not True:
        raise ArtifactVectorExportError(
            "provenance.geometry_revision.qc.finite_vertices must be true"
        )
    try:
        admission = validate_mesh_admission_receipt(geometry_qc["import_admission"])
    except MeshAdmissionError as exc:
        raise ArtifactVectorExportError(
            f"invalid provenance mesh admission receipt: {exc}"
        ) from exc
    accepted = admission["accepted"]
    if face_count != accepted["triangle_count"]:
        raise ArtifactVectorExportError(
            "geometry face_count does not match mesh admission"
        )
    if vertex_count != accepted["vertex_count"]:
        raise ArtifactVectorExportError(
            "geometry vertex_count does not match mesh admission"
        )
    decoded = admission["decoded"]
    try:
        decoded_admission_from_counts(
            vertex_count=decoded["vertex_count"],
            triangle_count=decoded["triangle_count"],
            array_bytes=decoded["array_bytes"],
        )
    except MeshAdmissionError as exc:
        raise ArtifactVectorExportError(
            f"invalid provenance decoded mesh admission: {exc}"
        ) from exc
    decoded_canonical_bytes = (
        24 * decoded["vertex_count"] + 12 * decoded["triangle_count"]
    )
    accepted_canonical_bytes = (
        24 * accepted["vertex_count"] + 12 * accepted["triangle_count"]
    )
    if decoded["array_bytes"] < max(
        decoded_canonical_bytes,
        accepted_canonical_bytes,
    ):
        raise ArtifactVectorExportError(
            "mesh admission decoded bytes are below canonical decoded/accepted arrays"
        )

    try:
        import_execution = validate_mesh_import_recipe(
            geometry["import_recipe"],
            allow_legacy=True,
            runtime_policy=RUNTIME_POLICY_RECORD_ONLY,
        )
    except MeshImportRecipeError as exc:
        raise ArtifactVectorExportError(
            f"invalid provenance mesh import recipe: {exc}"
        ) from exc
    if admission["source_format"] != import_execution.source_format:
        raise ArtifactVectorExportError(
            "mesh admission source_format does not match the import recipe"
        )
    assets = provenance["source_assets"]
    assert isinstance(assets, list)
    if len(assets) != 1:
        raise ArtifactVectorExportError(
            "current export provenance requires exactly one primary source asset"
        )
    primary_asset = assets[0]
    assert isinstance(primary_asset, Mapping)
    if admission["source_size_bytes"] != primary_asset["size_bytes"]:
        raise ArtifactVectorExportError(
            "mesh admission source_size_bytes does not match the primary source asset"
        )


def _validate_outline_topology_qc(value: object) -> None:
    topology = _exact_keys(
        value,
        {
            "area_mm2",
            "bounds_mm",
            "component_areas_mm2",
            "component_count",
            "component_exterior_path_ids",
            "component_hole_counts",
            "exterior_count",
            "geometry_type",
            "hole_assignments",
            "hole_count",
            "ring_count",
            "topology_valid",
            "validity_reason",
        },
        model_name="qc.record.outline_topology",
    )
    _finite_number(
        topology["area_mm2"],
        field_name="qc.record.outline_topology.area_mm2",
        minimum=0.0,
    )
    bounds = topology["bounds_mm"]
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise ArtifactVectorExportError(
            "qc.record.outline_topology.bounds_mm must contain four numbers"
        )
    for index, item in enumerate(bounds):
        _finite_number(
            item,
            field_name=f"qc.record.outline_topology.bounds_mm[{index}]",
        )
    component_areas = topology["component_areas_mm2"]
    if not isinstance(component_areas, list):
        raise ArtifactVectorExportError(
            "qc.record.outline_topology.component_areas_mm2 must be an array"
        )
    for index, item in enumerate(component_areas):
        _finite_number(
            item,
            field_name=(
                f"qc.record.outline_topology.component_areas_mm2[{index}]"
            ),
            minimum=0.0,
        )
    for key in (
        "component_count",
        "exterior_count",
        "hole_count",
        "ring_count",
    ):
        _strict_nonnegative_int(
            topology[key],
            field_name=f"qc.record.outline_topology.{key}",
        )
    for key in ("component_exterior_path_ids", "component_hole_counts"):
        if not isinstance(topology[key], list):
            raise ArtifactVectorExportError(
                f"qc.record.outline_topology.{key} must be an array"
            )
    for index, item in enumerate(topology["component_exterior_path_ids"]):
        _required_string(
            item,
            field_name=(
                "qc.record.outline_topology.component_exterior_path_ids"
                f"[{index}]"
            ),
        )
    for index, item in enumerate(topology["component_hole_counts"]):
        _strict_nonnegative_int(
            item,
            field_name=(
                f"qc.record.outline_topology.component_hole_counts[{index}]"
            ),
        )
    assignments = topology["hole_assignments"]
    if not isinstance(assignments, list):
        raise ArtifactVectorExportError(
            "qc.record.outline_topology.hole_assignments must be an array"
        )
    for index, item in enumerate(assignments):
        assignment = _exact_keys(
            item,
            {"exterior_path_id", "hole_path_id"},
            model_name=f"qc.record.outline_topology.hole_assignments[{index}]",
        )
        for key in ("exterior_path_id", "hole_path_id"):
            _required_string(
                assignment[key],
                field_name=(
                    f"qc.record.outline_topology.hole_assignments[{index}].{key}"
                ),
            )
    if topology["geometry_type"] not in {"Polygon", "MultiPolygon"}:
        raise ArtifactVectorExportError(
            "qc.record.outline_topology.geometry_type is invalid"
        )
    if topology["topology_valid"] is not True:
        raise ArtifactVectorExportError(
            "qc.record.outline_topology.topology_valid must be true"
        )
    _required_string(
        topology["validity_reason"],
        field_name="qc.record.outline_topology.validity_reason",
    )


def _validate_align_qc(
    value: object,
    recipe_kind: object,
    *,
    model_name: str,
) -> None:
    """Close the Align QC against the key set its recipe kind is entitled to."""

    expected = (
        _AXIS_ALIGN_QC_KEYS
        if recipe_kind == AXIS_ALIGN_RECIPE_KIND
        else frozenset({"proper_rigid"})
    )
    qc = _exact_keys(value, expected, model_name=model_name)
    if qc["proper_rigid"] is not True:
        raise ArtifactVectorExportError(f"{model_name}.proper_rigid must be true")
    if recipe_kind != AXIS_ALIGN_RECIPE_KIND:
        return
    for key in ("axis_tilt_corrected_deg", "circle_normal_disagreement_deg"):
        angle = _finite_number(qc[key], field_name=f"{model_name}.{key}", minimum=0.0)
        if angle > 180.0:
            raise ArtifactVectorExportError(f"{model_name}.{key} must be at most 180")
    _finite_number(
        qc["center_separation_mm"],
        field_name=f"{model_name}.center_separation_mm",
        strictly_positive=True,
    )


def _require_reviewed_outline_backend(value: Mapping[str, Any]) -> None:
    """Require the recorded outline backend to be one this project reviewed.

    An outline record names the exact Shapely and GEOS it was computed with.
    Both are checked as a pair, because the guarantee belongs to the
    combination: Shapely is a binding over a specific GEOS, and it is GEOS that
    decides the fixed-precision union.
    """

    shapely_version = value.get("backend_shapely_version")
    geos_version = value.get("backend_geos_version")
    if shapely_version is None and geos_version is None:
        return
    if (shapely_version, geos_version) not in REVIEWED_OUTLINE_BACKENDS:
        reviewed = ", ".join(
            f"Shapely {pair[0]}/GEOS {pair[1]}"
            for pair in sorted(REVIEWED_OUTLINE_BACKENDS)
        )
        raise ArtifactVectorExportError(
            "outline record names an unreviewed geometry backend: "
            f"Shapely {shapely_version!r} with GEOS {geos_version!r}; "
            f"reviewed backends are {reviewed}"
        )


def _validate_current_record_qc(
    value: object,
    *,
    payload: VectorGeometryPayload,
    recipe: Mapping[str, Any],
    provenance: Mapping[str, Any],
    schema_version: str = _CURRENT_VECTOR_EXPORT_SCHEMA_VERSION,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactVectorExportError("qc.record must be an object")
    kind = VectorRecordKind(payload.kind)
    algorithm = recipe.get("algorithm")
    algorithm_version = recipe.get("algorithm_version")
    production_algorithm = (
        _PRODUCTION_CUTLINE_ALGORITHM
        if kind is VectorRecordKind.CUTLINE
        else _PRODUCTION_OUTLINE_ALGORITHM
    )
    known_algorithms = {
        _PRODUCTION_CUTLINE_ALGORITHM: VectorRecordKind.CUTLINE,
        _PRODUCTION_OUTLINE_ALGORITHM: VectorRecordKind.OUTLINE,
    }
    if algorithm in known_algorithms:
        if known_algorithms[algorithm] is not kind:
            raise ArtifactVectorExportError(
                "production vector algorithm does not match the payload kind"
            )
        if algorithm_version not in _PRODUCTION_ALGORITHM_VERSIONS[str(algorithm)]:
            raise ArtifactVectorExportError(
                "production vector algorithm version is unsupported"
            )
    is_production = algorithm == production_algorithm
    # The grid closing arrived with outline 1.1.0 and the 1.2.0 sidecar; an
    # earlier sidecar cannot carry it, and a closed outline cannot omit it.
    closing = (
        kind is VectorRecordKind.OUTLINE
        and is_production
        and algorithm_version == OUTLINE_ALGORITHM_VERSION
    )
    if closing and schema_version not in _GRID_CLOSING_VECTOR_EXPORT_SCHEMA_VERSIONS:
        raise ArtifactVectorExportError(
            "a vector export before 1.2.0 cannot carry an outline computed with "
            "grid closing"
        )
    optional_keys = (
        _CUTLINE_RECORD_QC_KEYS
        if kind is VectorRecordKind.CUTLINE
        else _OUTLINE_RECORD_QC_KEYS | (_OUTLINE_CLOSING_QC_KEYS if closing else frozenset())
    )
    always_kind_keys = (
        frozenset()
        if kind is VectorRecordKind.CUTLINE
        else frozenset({"outline_topology"})
    )
    allowed = _VECTOR_PAYLOAD_QC_KEYS | (
        optional_keys if is_production else always_kind_keys
    )
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ArtifactVectorExportError(
            f"qc.record has unknown {kind.value} fields: {', '.join(unknown)}"
        )
    required_kind_keys = optional_keys if is_production else always_kind_keys
    missing = sorted(set(required_kind_keys) - set(value))
    if missing:
        raise ArtifactVectorExportError(
            f"qc.record is missing {kind.value} fields: {', '.join(missing)}"
        )

    count_keys = {
        "candidate_face_count",
        "collinear_point_removal_count",
        "component_count",
        "coplanar_face_count",
        "duplicate_segment_count",
        "face_chunk_count",
        "fixed_grid_triangle_count",
        "grid_collapsed_triangle_count",
        "hole_count",
        "input_face_count",
        "input_vertex_count",
        "intersected_face_count",
        "non_manifold_junction_count",
        "on_plane_edge_face_count",
        "outline_collinear_point_removal_count",
        "point_tangent_count",
        "projected_degenerate_triangle_count",
        "projected_non_degenerate_triangle_count",
        "raw_segment_count",
        "unique_segment_count",
    }
    for key in sorted(count_keys & set(value)):
        _strict_nonnegative_int(value[key], field_name=f"qc.record.{key}")
    nullable_count_keys = {
        "grid_component_merge_count",
        "grid_component_split_count",
        "unsnapped_component_count",
    }
    for key in sorted(nullable_count_keys & set(value)):
        if value[key] is not None:
            _strict_nonnegative_int(value[key], field_name=f"qc.record.{key}")
    nonnegative_number_keys = {
        "classification_tolerance_mm",
        "grid_snap_axis_upper_bound_mm",
        "grid_snap_radial_upper_bound_squared_mm2",
        "max_endpoint_snap_mm",
        "max_plane_residual_mm",
        "outline_area_mm2",
        "outline_perimeter_mm",
        "output_grid_residual_max_mm",
        "precision_grid_mm",
        "stitch_tolerance_mm",
        "unsnapped_area_mm2",
    }
    for key in sorted(nonnegative_number_keys & set(value)):
        if value[key] is not None:
            _finite_number(
                value[key],
                field_name=f"qc.record.{key}",
                minimum=0.0,
            )
    if "grid_area_delta_mm2" in value and value["grid_area_delta_mm2"] is not None:
        _finite_number(
            value["grid_area_delta_mm2"],
            field_name="qc.record.grid_area_delta_mm2",
        )
    # The recorded pair must be one this project reviewed, not necessarily the
    # one it computes with today.  Demanding the current pin here meant that
    # upgrading Shapely rejected every outline package written before the
    # upgrade, including packages this same code had produced.
    _require_reviewed_outline_backend(value)
    if "all_projected_faces_included" in value:
        if value["all_projected_faces_included"] is not True:
            raise ArtifactVectorExportError(
                "qc.record.all_projected_faces_included must be true"
            )
    if "sampling_applied" in value and value["sampling_applied"] is not False:
        raise ArtifactVectorExportError("qc.record.sampling_applied must be false")
    if "topology_valid" in value and value["topology_valid"] is not True:
        raise ArtifactVectorExportError("qc.record.topology_valid must be true")
    if "grid_snap_error_contract" in value and value["grid_snap_error_contract"] != (
        _SNAP_ERROR_CONTRACTS[
            str(algorithm_version)
            if closing
            else OUTLINE_LEGACY_ALGORITHM_VERSION
        ][0]
    ):
        raise ArtifactVectorExportError(
            "qc.record.grid_snap_error_contract is invalid"
        )
    for key in sorted(_OUTLINE_CLOSING_QC_KEYS & set(value)):
        if key == "grid_closing_radius_cells":
            if value[key] != OUTLINE_GRID_CLOSING_RADIUS_CELLS:
                raise ArtifactVectorExportError(
                    "qc.record.grid_closing_radius_cells is not the production radius"
                )
        elif key == "grid_closing_area_delta_mm2":
            _finite_number(value[key], field_name=f"qc.record.{key}")
        else:
            _strict_nonnegative_int(value[key], field_name=f"qc.record.{key}")
    if "grid_origin_index_uv" in value:
        grid_origin = value["grid_origin_index_uv"]
        if (
            not isinstance(grid_origin, list)
            or len(grid_origin) != 2
            or any(type(item) is not int for item in grid_origin)
        ):
            raise ArtifactVectorExportError(
                "qc.record.grid_origin_index_uv must contain two integers"
            )
    if "view" in value and value["view"] not in {
        "back",
        "bottom",
        "front",
        "left",
        "right",
        "top",
    }:
        raise ArtifactVectorExportError("qc.record.view is invalid")
    if "unsnapped_comparison_status" in value:
        status = value["unsnapped_comparison_status"]
        if status not in {"available", "unavailable_geos_union_failure"}:
            raise ArtifactVectorExportError(
                "qc.record.unsnapped_comparison_status is invalid"
            )
        nullable_fields = {
            "grid_area_delta_mm2",
            "grid_component_merge_count",
            "grid_component_split_count",
            "unsnapped_area_mm2",
            "unsnapped_component_count",
        }
        for key in nullable_fields & set(value):
            if status == "available" and value[key] is None:
                raise ArtifactVectorExportError(
                    f"qc.record.{key} must be available when raw comparison succeeds"
                )
            if status != "available" and value[key] is not None:
                raise ArtifactVectorExportError(
                    f"qc.record.{key} must be null when raw comparison is unavailable"
                )
    if "outline_topology" in value:
        _validate_outline_topology_qc(value["outline_topology"])
        from .artifact_outline_topology import (  # noqa: PLC0415
            ArtifactOutlineTopologyError,
            validate_outline_topology,
        )

        try:
            expected_topology = validate_outline_topology(payload).to_dict()
        except ArtifactOutlineTopologyError as exc:
            raise ArtifactVectorExportError(
                f"outline payload topology is invalid: {exc}"
            ) from exc
        if value["outline_topology"] != expected_topology:
            raise ArtifactVectorExportError(
                "qc.record.outline_topology does not match the vector payload"
            )
    if is_production:
        geometry = provenance["geometry_revision"]
        assert isinstance(geometry, Mapping)
        geometry_qc = geometry["qc"]
        assert isinstance(geometry_qc, Mapping)
        try:
            admission = validate_mesh_admission_receipt(
                geometry_qc["import_admission"]
            )
        except (KeyError, MeshAdmissionError) as exc:
            raise ArtifactVectorExportError(
                f"invalid production QC mesh admission authority: {exc}"
            ) from exc
        accepted = admission["accepted"]
        if value["input_face_count"] != accepted["triangle_count"]:
            raise ArtifactVectorExportError(
                "qc.record.input_face_count does not match mesh admission"
            )
        if value["input_vertex_count"] != accepted["vertex_count"]:
            raise ArtifactVectorExportError(
                "qc.record.input_vertex_count does not match mesh admission"
            )
        if kind is VectorRecordKind.CUTLINE:
            for key in (
                "classification_tolerance_mm",
                "max_endpoint_snap_mm",
                "max_plane_residual_mm",
                "stitch_tolerance_mm",
            ):
                _finite_number(
                    value[key],
                    field_name=f"qc.record.{key}",
                    minimum=0.0,
                )
            if (
                value["classification_tolerance_mm"]
                != recipe["classification_tolerance_mm"]
            ):
                raise ArtifactVectorExportError(
                    "Cutline QC classification tolerance does not match recipe"
                )
            if value["stitch_tolerance_mm"] != recipe["stitch_tolerance_mm"]:
                raise ArtifactVectorExportError(
                    "Cutline QC stitch tolerance does not match recipe"
                )
            if (
                value["max_plane_residual_mm"]
                > value["classification_tolerance_mm"]
            ):
                raise ArtifactVectorExportError(
                    "Cutline QC plane residual exceeds classification tolerance"
                )
            if value["max_endpoint_snap_mm"] > value["stitch_tolerance_mm"]:
                raise ArtifactVectorExportError(
                    "Cutline QC endpoint snap exceeds stitch tolerance"
                )
        else:
            for key in (
                "grid_snap_axis_upper_bound_mm",
                "grid_snap_radial_upper_bound_squared_mm2",
                "output_grid_residual_max_mm",
                "precision_grid_mm",
            ):
                _finite_number(
                    value[key],
                    field_name=f"qc.record.{key}",
                    minimum=0.0,
                )
            precision_grid = recipe["precision_grid_mm"]
            if value["precision_grid_mm"] != precision_grid:
                raise ArtifactVectorExportError(
                    "Outline QC precision grid does not match recipe"
                )
            if value["view"] != recipe["view"]:
                raise ArtifactVectorExportError(
                    "Outline QC view does not match recipe"
                )
            backend = recipe["backend"]
            if not isinstance(backend, Mapping):
                raise ArtifactVectorExportError(
                    "Outline recipe backend must be an object"
                )
            if value["backend_geos_version"] != backend["geos_version"]:
                raise ArtifactVectorExportError(
                    "Outline QC GEOS version does not match recipe"
                )
            if value["backend_shapely_version"] != backend["shapely_version"]:
                raise ArtifactVectorExportError(
                    "Outline QC Shapely version does not match recipe"
                )
            snap_cells = _SNAP_ERROR_CONTRACTS[
                str(algorithm_version) if closing else OUTLINE_LEGACY_ALGORITHM_VERSION
            ][1]
            if value["grid_snap_axis_upper_bound_mm"] != precision_grid * snap_cells:
                raise ArtifactVectorExportError(
                    "Outline QC axis snap bound does not match precision grid"
                )
            if (
                value["grid_snap_radial_upper_bound_squared_mm2"]
                != 2.0 * (precision_grid * snap_cells) ** 2
            ):
                raise ArtifactVectorExportError(
                    "Outline QC radial snap bound does not match precision grid"
                )
            if value["output_grid_residual_max_mm"] > precision_grid / 2.0:
                raise ArtifactVectorExportError(
                    "Outline QC output grid residual exceeds its grid contract"
                )
    return value


def _validate_qc(
    value: object,
    *,
    payload: VectorGeometryPayload,
    recipe: Mapping[str, Any],
    provenance: Mapping[str, Any],
    schema_version: str,
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
    if schema_version in _CURRENT_CONTRACT_VECTOR_EXPORT_SCHEMA_VERSIONS:
        _validate_current_record_qc(
            record_qc,
            payload=payload,
            recipe=recipe,
            provenance=provenance,
            schema_version=schema_version,
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
    schema_version = sidecar["schema_version"]
    if (
        not isinstance(schema_version, str)
        or schema_version not in SUPPORTED_VECTOR_EXPORT_SCHEMA_VERSIONS
    ):
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
    provenance = _validate_provenance_shape(
        sidecar["provenance"],
        require_current_contract=(
            schema_version in _CURRENT_CONTRACT_VECTOR_EXPORT_SCHEMA_VERSIONS
        ),
    )
    if schema_version in _CURRENT_CONTRACT_VECTOR_EXPORT_SCHEMA_VERSIONS:
        _validate_current_vector_provenance(provenance)
    else:
        legacy_geometry = provenance["geometry_revision"]
        assert isinstance(legacy_geometry, Mapping)
        legacy_geometry_qc = legacy_geometry["qc"]
        assert isinstance(legacy_geometry_qc, Mapping)
        if "import_admission" in legacy_geometry_qc:
            raise ArtifactVectorExportError(
                "legacy vector export schema 1.0.0 cannot contain import_admission"
            )
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
        if schema_version in _CURRENT_CONTRACT_VECTOR_EXPORT_SCHEMA_VERSIONS:
            validate_vector_payload_recipe_contract(payload, recipe)
    except ArtifactVectorRecordError as exc:
        raise ArtifactVectorExportError(str(exc)) from exc
    try:
        recipe_hash = canonical_recipe_hash(recipe)
    except ArtifactDocumentError as exc:
        raise ArtifactVectorExportError(str(exc)) from exc
    if record_provenance["recipe_hash"] != recipe_hash:
        raise ArtifactVectorExportError("recipe hash does not match the sidecar recipe")
    _validate_qc(
        sidecar["qc"],
        payload=payload,
        recipe=recipe,
        provenance=provenance,
        schema_version=schema_version,
    )
    claims_sha256 = _sidecar_claims_sha256(sidecar)

    options, presentation = _options_from_presentation(
        sidecar["presentation"], schema_version=schema_version
    )
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
        if provenance != _provenance(
            document,
            record,
            include_current_contract=(
                schema_version in _CURRENT_CONTRACT_VECTOR_EXPORT_SCHEMA_VERSIONS
            ),
        ):
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
    "SUPPORTED_VECTOR_EXPORT_SCHEMA_VERSIONS",
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
    "validate_current_public_export_provenance",
    "validate_legacy_public_export_provenance",
    "validate_public_export_provenance",
    "write_new_export_file",
]
