"""Deterministic, receipt-ready roof-tile surface unwrapping.

The legacy flattening UI is intentionally permissive: it can guess an axis,
smooth coordinates, and fall back to another algorithm.  An archaeological
measurement record needs a narrower contract.  This module therefore accepts
an explicit canonical axis, persists the exact source-face selection, rejects
fallbacks, and quantizes the final 2D coordinates to an integer micrometre
grid before hashing them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_cancellation import (
    CancellationProbe,
    poll_cancellation,
    raise_if_cancelled,
)
from .artifact_document import OperationContext, canonical_recipe_hash
from .artifact_scene_adapter import ArtifactProjectionSnapshot
from .artifact_session import ArtifactSession, ArtifactSessionError
from .canonical_json import (
    CanonicalJSONError,
    canonical_json_bytes,
    canonical_json_sha256,
)
from .flatten_metrics import compute_face_distortion, distortion_summary
from .flatten_models_sectionwise import (
    sectionwise_cylindrical_parameterization,
    sectionwise_quality_gate,
)
from .mesh_loader import MeshData


TILE_UNWRAP_ALGORITHM = "archmeshrubbing.sectionwise_tile_unwrap"
TILE_UNWRAP_ALGORITHM_VERSION = "1.2.0"
TILE_UNWRAP_RECIPE_SCHEMA_VERSION = "1.2.0"
TILE_UNWRAP_OUTPUT_SCHEMA_VERSION = "1.1.0"
TILE_UNWRAP_COORDINATE_SPACE = "canonical_mm_tile_unwrap/v1"
TILE_UNWRAP_HASH_SCOPE = (
    "amr-tile-unwrap-v1:length-prefixed-header+uv-i64le+faces-i32le+"
    "source-vertices-i64le+source-faces-i64le"
)
TILE_UNWRAP_SELECTION_SCHEMA_VERSION = "1.0.0"
TILE_UNWRAP_COORDINATE_QUANTUM_UM = 1
TILE_UNWRAP_GEOMETRY_REF_PREFIX = "urn:archmeshrubbing:tile-unwrap:sha256:"

MIN_TILE_UNWRAP_SECTIONS = 12
MAX_TILE_UNWRAP_SECTIONS = 96
MAX_TILE_UNWRAP_VERTICES = 2_000_000
MAX_TILE_UNWRAP_FACES = 2_000_000
MAX_TILE_UNWRAP_SELECTION_RANGES = 250_000
MAX_TILE_UNWRAP_COORDINATE_UM = 2**52
MAX_TILE_UNWRAP_PAYLOAD_BYTES = 128 * 1024 * 1024
MAX_TILE_UNWRAP_QC_FACES = 250_000
MAX_TILE_UNWRAP_OVERLAP_CANDIDATES = 1_000_000
MAX_TILE_UNWRAP_GRID_ASSIGNMENTS = 4_000_000
MIN_TILE_UNWRAP_SEAM_ANGLE_MICRODEGREES = -180_000_000
MAX_TILE_UNWRAP_SEAM_ANGLE_MICRODEGREES_EXCLUSIVE = 180_000_000

_LEGACY_TILE_UNWRAP_ALGORITHM_VERSION = "1.1.0"
_LEGACY_TILE_UNWRAP_RECIPE_SCHEMA_VERSION = "1.1.0"
_AUTO_SEAM_POLICY = "minimum_angular_range_auto"
_FIXED_SEAM_POLICY = "fixed_angle_microdegrees"

_TILE_UNWRAP_PAYLOAD_MAGIC = b"AMR-TILE-UNWRAP\x00v1\x00"
_TILE_UNWRAP_COMPONENT_LABELS = (
    b"uv_um_i64le",
    b"faces_i32le",
    b"source_vertex_indices_i64le",
    b"source_face_indices_i64le",
)


class ArtifactTileUnwrapError(ValueError):
    """A tile-unwrapping input or output violates the measurement contract."""


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactTileUnwrapError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or result > maximum:
        raise ArtifactTileUnwrapError(
            f"{name} must be in the inclusive range {minimum}..{maximum}"
        )
    return result


def _exact_keys(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactTileUnwrapError(f"{name} must be an object")
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise ArtifactTileUnwrapError(f"{name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ArtifactTileUnwrapError(
            f"{name} has unknown fields: {', '.join(unknown)}"
        )
    return value


def _axis(value: object) -> str:
    axis = str(value or "").strip().lower()
    if axis not in {"x", "y", "z"}:
        raise ArtifactTileUnwrapError(
            "longitudinal_axis must be explicit canonical 'x', 'y', or 'z'"
        )
    return axis


def _record_view(value: object) -> str:
    view = str(value or "").strip().lower()
    if view not in {"top", "bottom"}:
        raise ArtifactTileUnwrapError("record_view must be 'top' or 'bottom'")
    return view


def _seam_angle_microdegrees(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactTileUnwrapError(
            "seam_angle_microdegrees must be null or an integer"
        )
    result = int(value)
    if not (
        MIN_TILE_UNWRAP_SEAM_ANGLE_MICRODEGREES
        <= result
        < MAX_TILE_UNWRAP_SEAM_ANGLE_MICRODEGREES_EXCLUSIVE
    ):
        raise ArtifactTileUnwrapError(
            "seam_angle_microdegrees must be in the half-open range "
            "[-180000000, 180000000)"
        )
    return result


def _selection_value(
    *,
    total_face_count: int,
    face_ranges: Sequence[Sequence[int]],
) -> dict[str, Any]:
    total = _strict_int(
        total_face_count,
        name="selection.total_face_count",
        minimum=1,
        maximum=MAX_TILE_UNWRAP_FACES,
    )
    raw_ranges = tuple(face_ranges)
    if not raw_ranges or len(raw_ranges) > MAX_TILE_UNWRAP_SELECTION_RANGES:
        raise ArtifactTileUnwrapError(
            "selection.face_ranges must contain a bounded non-empty range list"
        )
    canonical_ranges: list[list[int]] = []
    selected_count = 0
    previous_end = 0
    for index, raw in enumerate(raw_ranges):
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ArtifactTileUnwrapError(
                f"selection.face_ranges[{index}] must be [start, end_exclusive]"
            )
        start = _strict_int(
            raw[0],
            name=f"selection.face_ranges[{index}][0]",
            minimum=0,
            maximum=total - 1,
        )
        end = _strict_int(
            raw[1],
            name=f"selection.face_ranges[{index}][1]",
            minimum=1,
            maximum=total,
        )
        if start >= end:
            raise ArtifactTileUnwrapError("selection ranges must be non-empty")
        if index > 0 and start <= previous_end:
            raise ArtifactTileUnwrapError(
                "selection ranges must be sorted, disjoint, and maximally merged"
            )
        canonical_ranges.append([start, end])
        selected_count += end - start
        if selected_count > MAX_TILE_UNWRAP_QC_FACES:
            raise ArtifactTileUnwrapError(
                "recording-surface selection exceeds the 250000-face QC limit"
            )
        previous_end = end
    selection_core = {
        "face_ranges": canonical_ranges,
        "kind": "canonical_face_ranges",
        "schema_version": TILE_UNWRAP_SELECTION_SCHEMA_VERSION,
        "selected_face_count": selected_count,
        "total_face_count": total,
    }
    return {
        **selection_core,
        "selection_sha256": canonical_json_sha256(selection_core),
    }


def _indices_to_ranges(
    indices: np.ndarray, *, total_face_count: int
) -> list[list[int]]:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ArtifactTileUnwrapError("at least one recording-surface face is required")
    values = np.unique(values)
    if values.size > MAX_TILE_UNWRAP_QC_FACES:
        raise ArtifactTileUnwrapError(
            "recording-surface selection exceeds the 250000-face QC limit"
        )
    if int(values[0]) < 0 or int(values[-1]) >= total_face_count:
        raise ArtifactTileUnwrapError("recording-surface face index is out of range")
    ranges: list[list[int]] = []
    start = int(values[0])
    previous = start
    for raw in values[1:]:
        current = int(raw)
        if current != previous + 1:
            ranges.append([start, previous + 1])
            start = current
        previous = current
    ranges.append([start, previous + 1])
    return ranges


def selection_face_indices(selection: Mapping[str, Any]) -> np.ndarray:
    validated = validate_tile_unwrap_selection(selection)
    indices = np.empty((int(validated["selected_face_count"]),), dtype=np.int64)
    offset = 0
    for start, end in validated["face_ranges"]:
        count = int(end) - int(start)
        indices[offset : offset + count] = np.arange(start, end, dtype=np.int64)
        offset += count
    indices.setflags(write=False)
    return indices


def validate_tile_unwrap_selection(value: object) -> dict[str, Any]:
    selection = _exact_keys(
        value,
        {
            "face_ranges",
            "kind",
            "schema_version",
            "selected_face_count",
            "selection_sha256",
            "total_face_count",
        },
        name="tile unwrap selection",
    )
    if selection["kind"] != "canonical_face_ranges":
        raise ArtifactTileUnwrapError("tile unwrap selection kind is unsupported")
    if selection["schema_version"] != TILE_UNWRAP_SELECTION_SCHEMA_VERSION:
        raise ArtifactTileUnwrapError("tile unwrap selection schema is unsupported")
    raw_ranges = selection["face_ranges"]
    if not isinstance(raw_ranges, (list, tuple)):
        raise ArtifactTileUnwrapError("selection.face_ranges must be an array")
    canonical = _selection_value(
        total_face_count=_strict_int(
            selection["total_face_count"],
            name="selection.total_face_count",
            minimum=1,
            maximum=MAX_TILE_UNWRAP_FACES,
        ),
        face_ranges=raw_ranges,
    )
    if selection["selected_face_count"] != canonical["selected_face_count"]:
        raise ArtifactTileUnwrapError("selection selected_face_count is inconsistent")
    if selection["selection_sha256"] != canonical["selection_sha256"]:
        raise ArtifactTileUnwrapError("selection SHA-256 is inconsistent")
    return canonical


def tile_unwrap_recipe(
    *,
    longitudinal_axis: str,
    record_view: str,
    total_face_count: int,
    selected_face_indices: Sequence[int] | np.ndarray | None = None,
    n_sections: int = 32,
    seam_angle_microdegrees: int | None = None,
) -> dict[str, Any]:
    total = _strict_int(
        total_face_count,
        name="total_face_count",
        minimum=1,
        maximum=MAX_TILE_UNWRAP_FACES,
    )
    if selected_face_indices is None:
        ranges = [[0, total]]
    else:
        ranges = _indices_to_ranges(
            np.asarray(selected_face_indices, dtype=np.int64),
            total_face_count=total,
        )
    selection = _selection_value(total_face_count=total, face_ranges=ranges)
    seam_angle = _seam_angle_microdegrees(seam_angle_microdegrees)
    return {
        "algorithm": TILE_UNWRAP_ALGORITHM,
        "algorithm_version": TILE_UNWRAP_ALGORITHM_VERSION,
        "coordinate_quantum_um": TILE_UNWRAP_COORDINATE_QUANTUM_UM,
        "coordinate_space": TILE_UNWRAP_COORDINATE_SPACE,
        "fallback_policy": "reject",
        "kind": "tile_unwrap",
        "longitudinal_axis": _axis(longitudinal_axis),
        "n_sections": _strict_int(
            n_sections,
            name="n_sections",
            minimum=MIN_TILE_UNWRAP_SECTIONS,
            maximum=MAX_TILE_UNWRAP_SECTIONS,
        ),
        "record_view": _record_view(record_view),
        "schema_version": TILE_UNWRAP_RECIPE_SCHEMA_VERSION,
        "seam_angle_microdegrees": seam_angle,
        "seam_policy": (
            _AUTO_SEAM_POLICY if seam_angle is None else _FIXED_SEAM_POLICY
        ),
        "selection": selection,
        "smoothing_iterations": 0,
    }


def validate_tile_unwrap_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(recipe, Mapping):
        raise ArtifactTileUnwrapError("tile unwrap recipe must be an object")
    algorithm_version = recipe.get("algorithm_version")
    schema_version = recipe.get("schema_version")
    if (
        algorithm_version == _LEGACY_TILE_UNWRAP_ALGORITHM_VERSION
        and schema_version == _LEGACY_TILE_UNWRAP_RECIPE_SCHEMA_VERSION
    ):
        return _validate_legacy_tile_unwrap_recipe(recipe)
    if (
        algorithm_version != TILE_UNWRAP_ALGORITHM_VERSION
        or schema_version != TILE_UNWRAP_RECIPE_SCHEMA_VERSION
    ):
        raise ArtifactTileUnwrapError(
            "tile unwrap recipe algorithm/schema version is unsupported"
        )

    value = _exact_keys(
        recipe,
        {
            "algorithm",
            "algorithm_version",
            "coordinate_quantum_um",
            "coordinate_space",
            "fallback_policy",
            "kind",
            "longitudinal_axis",
            "n_sections",
            "record_view",
            "schema_version",
            "seam_angle_microdegrees",
            "seam_policy",
            "selection",
            "smoothing_iterations",
        },
        name="tile unwrap recipe",
    )
    expected_literals = {
        "algorithm": TILE_UNWRAP_ALGORITHM,
        "algorithm_version": TILE_UNWRAP_ALGORITHM_VERSION,
        "coordinate_quantum_um": TILE_UNWRAP_COORDINATE_QUANTUM_UM,
        "coordinate_space": TILE_UNWRAP_COORDINATE_SPACE,
        "fallback_policy": "reject",
        "kind": "tile_unwrap",
        "schema_version": TILE_UNWRAP_RECIPE_SCHEMA_VERSION,
        "smoothing_iterations": 0,
    }
    for key, expected in expected_literals.items():
        if value[key] != expected:
            raise ArtifactTileUnwrapError(
                f"tile unwrap recipe field {key!r} is invalid"
            )
    seam_angle = _seam_angle_microdegrees(value["seam_angle_microdegrees"])
    expected_seam_policy = (
        _AUTO_SEAM_POLICY if seam_angle is None else _FIXED_SEAM_POLICY
    )
    if value["seam_policy"] != expected_seam_policy:
        raise ArtifactTileUnwrapError(
            "tile unwrap recipe seam policy and angle are inconsistent"
        )
    return {
        **expected_literals,
        "longitudinal_axis": _axis(value["longitudinal_axis"]),
        "n_sections": _strict_int(
            value["n_sections"],
            name="n_sections",
            minimum=MIN_TILE_UNWRAP_SECTIONS,
            maximum=MAX_TILE_UNWRAP_SECTIONS,
        ),
        "record_view": _record_view(value["record_view"]),
        "seam_angle_microdegrees": seam_angle,
        "seam_policy": expected_seam_policy,
        "selection": validate_tile_unwrap_selection(value["selection"]),
    }


def _validate_legacy_tile_unwrap_recipe(
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable 1.1 recipe shape without upgrading its hash."""

    value = _exact_keys(
        recipe,
        {
            "algorithm",
            "algorithm_version",
            "coordinate_quantum_um",
            "coordinate_space",
            "fallback_policy",
            "kind",
            "longitudinal_axis",
            "n_sections",
            "record_view",
            "schema_version",
            "seam_policy",
            "selection",
            "smoothing_iterations",
        },
        name="legacy tile unwrap recipe",
    )
    expected_literals = {
        "algorithm": TILE_UNWRAP_ALGORITHM,
        "algorithm_version": _LEGACY_TILE_UNWRAP_ALGORITHM_VERSION,
        "coordinate_quantum_um": TILE_UNWRAP_COORDINATE_QUANTUM_UM,
        "coordinate_space": TILE_UNWRAP_COORDINATE_SPACE,
        "fallback_policy": "reject",
        "kind": "tile_unwrap",
        "schema_version": _LEGACY_TILE_UNWRAP_RECIPE_SCHEMA_VERSION,
        "seam_policy": _AUTO_SEAM_POLICY,
        "smoothing_iterations": 0,
    }
    for key, expected in expected_literals.items():
        if value[key] != expected:
            raise ArtifactTileUnwrapError(
                f"legacy tile unwrap recipe field {key!r} is invalid"
            )
    return {
        **expected_literals,
        "longitudinal_axis": _axis(value["longitudinal_axis"]),
        "n_sections": _strict_int(
            value["n_sections"],
            name="n_sections",
            minimum=MIN_TILE_UNWRAP_SECTIONS,
            maximum=MAX_TILE_UNWRAP_SECTIONS,
        ),
        "record_view": _record_view(value["record_view"]),
        "selection": validate_tile_unwrap_selection(value["selection"]),
    }


def _little_endian_bytes(array: np.ndarray, dtype: str) -> bytes:
    contiguous = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    return contiguous.tobytes(order="C")


def _framed_payload(
    header: Mapping[str, Any], components: Sequence[tuple[bytes, bytes]]
) -> bytes:
    try:
        header_bytes = canonical_json_bytes(header)
    except CanonicalJSONError as exc:
        raise ArtifactTileUnwrapError(str(exc)) from exc
    chunks = [_TILE_UNWRAP_PAYLOAD_MAGIC]
    for label, payload in ((b"header", header_bytes), *components):
        chunks.extend(
            (
                len(label).to_bytes(2, "big"),
                label,
                len(payload).to_bytes(8, "big"),
                payload,
            )
        )
    framed = b"".join(chunks)
    if len(framed) > MAX_TILE_UNWRAP_PAYLOAD_BYTES:
        raise ArtifactTileUnwrapError("tile unwrap payload exceeds its safety limit")
    return framed


def _framed_hash(
    header: Mapping[str, Any], components: Sequence[tuple[bytes, bytes]]
) -> str:
    return hashlib.sha256(_framed_payload(header, components)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rounded_millionths(value: float) -> int:
    if not np.isfinite(value):
        raise ArtifactTileUnwrapError("distortion metric is not finite")
    return int(np.rint(float(value) * 1_000_000.0))


@dataclass(frozen=True, slots=True)
class TileUnwrapMesh:
    """Quantized 2D mesh with exact correspondence to canonical source rows."""

    uv_um: np.ndarray
    faces: np.ndarray
    source_vertex_indices: np.ndarray
    source_face_indices: np.ndarray
    axis: str
    record_view: str

    def __post_init__(self) -> None:
        uv = np.asarray(self.uv_um)
        faces = np.asarray(self.faces)
        source_vertices = np.asarray(self.source_vertex_indices)
        source_faces = np.asarray(self.source_face_indices)
        if uv.dtype.kind not in {"i", "u"} or uv.ndim != 2 or uv.shape[1] != 2:
            raise ArtifactTileUnwrapError(
                "tile unwrap UVs must be an integer (N, 2) array"
            )
        if faces.dtype.kind not in {"i", "u"} or faces.ndim != 2 or faces.shape[1] != 3:
            raise ArtifactTileUnwrapError(
                "tile unwrap faces must be an integer (M, 3) array"
            )
        if source_vertices.dtype.kind not in {"i", "u"} or source_vertices.ndim != 1:
            raise ArtifactTileUnwrapError(
                "source vertex indices must be an integer vector"
            )
        if source_faces.dtype.kind not in {"i", "u"} or source_faces.ndim != 1:
            raise ArtifactTileUnwrapError(
                "source face indices must be an integer vector"
            )
        if uv.shape[0] < 3 or uv.shape[0] > MAX_TILE_UNWRAP_VERTICES:
            raise ArtifactTileUnwrapError(
                "tile unwrap vertex count is outside safety limits"
            )
        if faces.shape[0] < 1 or faces.shape[0] > MAX_TILE_UNWRAP_FACES:
            raise ArtifactTileUnwrapError(
                "tile unwrap face count is outside safety limits"
            )
        if (
            source_vertices.shape[0] != uv.shape[0]
            or source_faces.shape[0] != faces.shape[0]
        ):
            raise ArtifactTileUnwrapError(
                "tile unwrap correspondence counts are inconsistent"
            )
        if np.any(uv < 0) or np.any(uv > MAX_TILE_UNWRAP_COORDINATE_UM):
            raise ArtifactTileUnwrapError(
                "tile unwrap coordinate exceeds the exact grid limit"
            )
        if np.any(faces < 0) or np.any(faces > np.iinfo(np.int32).max):
            raise ArtifactTileUnwrapError("tile unwrap face index is invalid")
        if np.any(source_vertices < 0) or np.any(
            source_vertices > np.iinfo(np.int64).max
        ):
            raise ArtifactTileUnwrapError("source vertex index is invalid")
        if np.any(source_faces < 0) or np.any(
            source_faces > np.iinfo(np.int64).max
        ):
            raise ArtifactTileUnwrapError("source face index is invalid")
        uv_i64 = np.asarray(uv, dtype=np.int64)
        faces_i32 = np.asarray(faces, dtype=np.int32)
        source_vertices_i64 = np.asarray(source_vertices, dtype=np.int64)
        source_faces_i64 = np.asarray(source_faces, dtype=np.int64)
        if np.any(faces_i32 < 0) or np.any(faces_i32 >= uv_i64.shape[0]):
            raise ArtifactTileUnwrapError("tile unwrap face index is invalid")
        if np.any(source_vertices_i64 < 0) or np.any(np.diff(source_vertices_i64) <= 0):
            raise ArtifactTileUnwrapError(
                "source vertex indices must be sorted and unique"
            )
        if np.any(source_faces_i64 < 0) or np.any(np.diff(source_faces_i64) <= 0):
            raise ArtifactTileUnwrapError(
                "source face indices must be sorted and unique"
            )
        for array in (uv_i64, faces_i32, source_vertices_i64, source_faces_i64):
            array.setflags(write=False)
        object.__setattr__(self, "uv_um", uv_i64)
        object.__setattr__(self, "faces", faces_i32)
        object.__setattr__(self, "source_vertex_indices", source_vertices_i64)
        object.__setattr__(self, "source_face_indices", source_faces_i64)
        object.__setattr__(self, "axis", _axis(self.axis))
        object.__setattr__(self, "record_view", _record_view(self.record_view))

    @property
    def vertex_count(self) -> int:
        return int(self.uv_um.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def _binary_components(self) -> tuple[tuple[bytes, bytes], ...]:
        return (
            (b"uv_um_i64le", _little_endian_bytes(self.uv_um, "<i8")),
            (b"faces_i32le", _little_endian_bytes(self.faces, "<i4")),
            (
                b"source_vertex_indices_i64le",
                _little_endian_bytes(self.source_vertex_indices, "<i8"),
            ),
            (
                b"source_face_indices_i64le",
                _little_endian_bytes(self.source_face_indices, "<i8"),
            ),
        )

    def _payload_header(self, *, selection_sha256: str) -> dict[str, Any]:
        if (
            not isinstance(selection_sha256, str)
            or len(selection_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in selection_sha256
            )
        ):
            raise ArtifactTileUnwrapError("selection_sha256 must be a SHA-256")
        return {
            "axis": self.axis,
            "coordinate_quantum_um": TILE_UNWRAP_COORDINATE_QUANTUM_UM,
            "coordinate_space": TILE_UNWRAP_COORDINATE_SPACE,
            "face_count": self.face_count,
            "hash_scope": TILE_UNWRAP_HASH_SCOPE,
            "record_view": self.record_view,
            "schema_version": TILE_UNWRAP_OUTPUT_SCHEMA_VERSION,
            "selection_sha256": selection_sha256,
            "vertex_count": self.vertex_count,
        }

    def canonical_payload_bytes(self, *, selection_sha256: str) -> bytes:
        """Return the exact portable payload whose SHA-256 is geometry_ref."""

        return _framed_payload(
            self._payload_header(selection_sha256=selection_sha256),
            self._binary_components(),
        )

    @classmethod
    def from_canonical_payload_bytes(
        cls,
        payload: bytes,
        *,
        expected_selection_sha256: str | None = None,
    ) -> tuple["TileUnwrapMesh", dict[str, Any]]:
        """Parse the closed binary contract without trusting container metadata."""

        if not isinstance(payload, bytes):
            raise ArtifactTileUnwrapError("tile unwrap payload must be bytes")
        if (
            len(payload) <= len(_TILE_UNWRAP_PAYLOAD_MAGIC)
            or len(payload) > MAX_TILE_UNWRAP_PAYLOAD_BYTES
        ):
            raise ArtifactTileUnwrapError("tile unwrap payload byte length is invalid")
        view = memoryview(payload)
        offset = 0

        def read_exact(length: int, *, name: str) -> bytes:
            nonlocal offset
            if length < 0 or offset + length > len(view):
                raise ArtifactTileUnwrapError(f"tile unwrap payload truncates {name}")
            value = bytes(view[offset : offset + length])
            offset += length
            return value

        if read_exact(len(_TILE_UNWRAP_PAYLOAD_MAGIC), name="magic") != (
            _TILE_UNWRAP_PAYLOAD_MAGIC
        ):
            raise ArtifactTileUnwrapError("tile unwrap payload magic is invalid")
        parts: dict[bytes, bytes] = {}
        expected_labels = (b"header", *_TILE_UNWRAP_COMPONENT_LABELS)
        for expected_label in expected_labels:
            label_length = int.from_bytes(read_exact(2, name="label length"), "big")
            label = read_exact(label_length, name="label")
            if label != expected_label:
                raise ArtifactTileUnwrapError(
                    "tile unwrap payload component order is invalid"
                )
            component_length = int.from_bytes(
                read_exact(8, name="component length"), "big"
            )
            parts[label] = read_exact(
                component_length,
                name=label.decode("ascii"),
            )
        if offset != len(view):
            raise ArtifactTileUnwrapError("tile unwrap payload has trailing bytes")
        header_bytes = parts[b"header"]
        try:
            header_raw = json.loads(header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactTileUnwrapError(
                "tile unwrap payload header is invalid JSON"
            ) from exc
        if not isinstance(header_raw, Mapping):
            raise ArtifactTileUnwrapError(
                "tile unwrap payload header must be an object"
            )
        try:
            if canonical_json_bytes(header_raw) != header_bytes:
                raise ArtifactTileUnwrapError(
                    "tile unwrap payload header is not canonical RFC 8785 JSON"
                )
        except CanonicalJSONError as exc:
            raise ArtifactTileUnwrapError(str(exc)) from exc
        header = _exact_keys(
            header_raw,
            {
                "axis",
                "coordinate_quantum_um",
                "coordinate_space",
                "face_count",
                "hash_scope",
                "record_view",
                "schema_version",
                "selection_sha256",
                "vertex_count",
            },
            name="tile unwrap payload header",
        )
        if header["coordinate_quantum_um"] != TILE_UNWRAP_COORDINATE_QUANTUM_UM:
            raise ArtifactTileUnwrapError("tile unwrap payload quantum is invalid")
        if header["coordinate_space"] != TILE_UNWRAP_COORDINATE_SPACE:
            raise ArtifactTileUnwrapError(
                "tile unwrap payload coordinate space is invalid"
            )
        if header["hash_scope"] != TILE_UNWRAP_HASH_SCOPE:
            raise ArtifactTileUnwrapError("tile unwrap payload hash scope is invalid")
        if header["schema_version"] != TILE_UNWRAP_OUTPUT_SCHEMA_VERSION:
            raise ArtifactTileUnwrapError("tile unwrap payload schema is invalid")
        selection_sha = str(header["selection_sha256"])
        if len(selection_sha) != 64 or any(
            character not in "0123456789abcdef" for character in selection_sha
        ):
            raise ArtifactTileUnwrapError(
                "tile unwrap payload selection SHA-256 is invalid"
            )
        if (
            expected_selection_sha256 is not None
            and selection_sha != expected_selection_sha256
        ):
            raise ArtifactTileUnwrapError(
                "tile unwrap payload selection does not match"
            )
        vertex_count = _strict_int(
            header["vertex_count"],
            name="payload.vertex_count",
            minimum=3,
            maximum=MAX_TILE_UNWRAP_VERTICES,
        )
        face_count = _strict_int(
            header["face_count"],
            name="payload.face_count",
            minimum=1,
            maximum=MAX_TILE_UNWRAP_FACES,
        )
        expected_lengths = {
            b"uv_um_i64le": vertex_count * 2 * 8,
            b"faces_i32le": face_count * 3 * 4,
            b"source_vertex_indices_i64le": vertex_count * 8,
            b"source_face_indices_i64le": face_count * 8,
        }
        for label, expected_length in expected_lengths.items():
            if len(parts[label]) != expected_length:
                raise ArtifactTileUnwrapError(
                    f"tile unwrap payload {label.decode('ascii')} length is invalid"
                )
        uv = np.frombuffer(parts[b"uv_um_i64le"], dtype="<i8").reshape(-1, 2).copy()
        faces = np.frombuffer(parts[b"faces_i32le"], dtype="<i4").reshape(-1, 3).copy()
        source_vertices = np.frombuffer(
            parts[b"source_vertex_indices_i64le"], dtype="<i8"
        ).copy()
        source_faces = np.frombuffer(
            parts[b"source_face_indices_i64le"], dtype="<i8"
        ).copy()
        unwrap = cls(
            uv_um=uv,
            faces=faces,
            source_vertex_indices=source_vertices,
            source_face_indices=source_faces,
            axis=str(header["axis"]),
            record_view=str(header["record_view"]),
        )
        normalized_header = unwrap._payload_header(selection_sha256=selection_sha)
        if dict(header) != normalized_header:
            raise ArtifactTileUnwrapError(
                "tile unwrap payload header counts are inconsistent"
            )
        if unwrap.canonical_payload_bytes(selection_sha256=selection_sha) != payload:
            raise ArtifactTileUnwrapError("tile unwrap payload is not canonical")
        return unwrap, normalized_header

    def receipt(self, *, selection_sha256: str) -> dict[str, Any]:
        components = self._binary_components()
        bounds_min = np.min(self.uv_um, axis=0)
        bounds_max = np.max(self.uv_um, axis=0)
        header = self._payload_header(selection_sha256=selection_sha256)
        component_hashes = {
            label.decode("ascii"): _sha256_bytes(payload)
            for label, payload in components
        }
        return {
            **header,
            "bounds_um": {
                "maximum_u": int(bounds_max[0]),
                "maximum_v": int(bounds_max[1]),
                "minimum_u": int(bounds_min[0]),
                "minimum_v": int(bounds_min[1]),
            },
            "component_sha256": component_hashes,
            "height_mm_exact": {
                "denominator": 1000,
                "numerator": int(bounds_max[1] - bounds_min[1]),
            },
            "source_face_count": int(self.source_face_indices.shape[0]),
            "source_vertex_count": int(self.source_vertex_indices.shape[0]),
            "unwrap_sha256": _framed_hash(header, components),
            "width_mm_exact": {
                "denominator": 1000,
                "numerator": int(bounds_max[0] - bounds_min[0]),
            },
        }

    @property
    def geometry_ref(self) -> str:
        # The selection digest is part of the record recipe and receipt.  This
        # property is intentionally unavailable without that durable context.
        raise AttributeError("use geometry_ref_for_selection(selection_sha256)")

    def geometry_ref_for_selection(self, selection_sha256: str) -> str:
        receipt = self.receipt(selection_sha256=selection_sha256)
        return f"{TILE_UNWRAP_GEOMETRY_REF_PREFIX}{receipt['unwrap_sha256']}"


def _submesh_for_selection(
    mesh: MeshData,
    source_face_indices: np.ndarray,
) -> tuple[MeshData, np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        raise ArtifactTileUnwrapError("canonical mesh has invalid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] < 1:
        raise ArtifactTileUnwrapError("canonical mesh has invalid faces")
    selected_faces = faces[source_face_indices]
    source_vertex_indices = np.unique(selected_faces.reshape(-1)).astype(np.int64)
    if source_vertex_indices.size > MAX_TILE_UNWRAP_VERTICES:
        raise ArtifactTileUnwrapError("recording surface has too many vertices")
    local_faces = np.searchsorted(source_vertex_indices, selected_faces).astype(
        np.int32
    )
    selected_vertices = vertices[source_vertex_indices]
    return (
        MeshData(
            vertices=selected_vertices,
            faces=local_faces,
            unit="mm",
        ),
        source_vertex_indices,
        np.asarray(source_face_indices, dtype=np.int64),
    )


def _surface_topology_qc(
    faces: np.ndarray,
    *,
    vertex_count: int,
    cancellation_probe: CancellationProbe | None,
) -> dict[str, int]:
    """Audit one recording surface before its geometry can become READY."""

    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ArtifactTileUnwrapError("recording surface faces must be triangles")
    face_count = int(triangles.shape[0])
    if face_count < 1:
        raise ArtifactTileUnwrapError("recording surface has no faces")
    if face_count > MAX_TILE_UNWRAP_QC_FACES:
        raise ArtifactTileUnwrapError(
            "recording surface exceeds the topology/overlap QC face limit; "
            "select a smaller contiguous surface"
        )

    parent = np.arange(face_count, dtype=np.int32)

    def find(index: int) -> int:
        root = index
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[index]) != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            if first_root < second_root:
                parent[second_root] = first_root
            else:
                parent[first_root] = second_root

    # key -> [first face, first start, first end, count, second start, second end]
    edge_state: dict[tuple[int, int], list[int]] = {}
    canonical_faces: set[tuple[int, int, int]] = set()
    duplicate_faces = 0
    for face_index, raw_face in enumerate(triangles):
        poll_cancellation(cancellation_probe, face_index)
        a, b, c = (int(raw_face[0]), int(raw_face[1]), int(raw_face[2]))
        if min(a, b, c) < 0 or max(a, b, c) >= vertex_count:
            raise ArtifactTileUnwrapError(
                "recording surface contains an invalid vertex index"
            )
        ordered_face = sorted((a, b, c))
        canonical_face = (ordered_face[0], ordered_face[1], ordered_face[2])
        if canonical_face in canonical_faces:
            duplicate_faces += 1
        else:
            canonical_faces.add(canonical_face)
        for start, end in ((a, b), (b, c), (c, a)):
            key = (min(start, end), max(start, end))
            state = edge_state.get(key)
            if state is None:
                edge_state[key] = [face_index, start, end, 1, -1, -1]
            else:
                union(face_index, state[0])
                if state[3] == 1:
                    state[4] = start
                    state[5] = end
                state[3] += 1

    nonmanifold_edges = 0
    inconsistent_edges = 0
    boundary_adjacency: dict[int, list[int]] = {}
    for state in edge_state.values():
        count = state[3]
        if count > 2:
            nonmanifold_edges += 1
        elif count == 2:
            first_direction = (state[1], state[2])
            second_direction = (state[4], state[5])
            if second_direction == first_direction:
                inconsistent_edges += 1
        elif count == 1:
            start, end = state[1], state[2]
            boundary_adjacency.setdefault(start, []).append(end)
            boundary_adjacency.setdefault(end, []).append(start)

    if not boundary_adjacency:
        boundary_loop_count = 0
    elif any(len(neighbors) != 2 for neighbors in boundary_adjacency.values()):
        boundary_loop_count = -1
    else:
        remaining = set(boundary_adjacency)
        boundary_loop_count = 0
        while remaining:
            boundary_loop_count += 1
            stack = [min(remaining)]
            while stack:
                vertex = stack.pop()
                if vertex not in remaining:
                    continue
                remaining.remove(vertex)
                stack.extend(boundary_adjacency[vertex])

    component_count = len({find(index) for index in range(face_count)})
    return {
        "boundary_loop_count": boundary_loop_count,
        "connected_component_count": component_count,
        "duplicate_face_count": duplicate_faces,
        "inconsistent_oriented_edge_count": inconsistent_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
    }


def _positive_area_triangle_overlap(
    first: np.ndarray,
    second: np.ndarray,
) -> bool:
    """Exact separating-axis test on integer-micrometre triangles."""

    first_points = tuple((int(point[0]), int(point[1])) for point in first)
    second_points = tuple((int(point[0]), int(point[1])) for point in second)
    for triangle in (first_points, second_points):
        for index in range(3):
            start = triangle[index]
            end = triangle[(index + 1) % 3]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            axis_x, axis_y = -dy, dx
            first_projection = tuple(
                point[0] * axis_x + point[1] * axis_y
                for point in first_points
            )
            second_projection = tuple(
                point[0] * axis_x + point[1] * axis_y
                for point in second_points
            )
            overlap = min(max(first_projection), max(second_projection)) - max(
                min(first_projection), min(second_projection)
            )
            if overlap <= 0:
                return False
    return True


def _uv_overlap_pair_count(
    uv_um: np.ndarray,
    faces: np.ndarray,
    *,
    cancellation_probe: CancellationProbe | None,
) -> int:
    """Detect positive-area global overlap using a deterministic uniform grid."""

    uv = np.asarray(uv_um, dtype=np.int64)
    triangles = uv[np.asarray(faces, dtype=np.int32)]
    face_count = int(triangles.shape[0])
    if face_count > MAX_TILE_UNWRAP_QC_FACES:
        raise ArtifactTileUnwrapError(
            "tile unwrap exceeds the global-overlap QC face limit"
        )
    minimum = np.min(triangles, axis=1)
    maximum = np.max(triangles, axis=1)
    global_minimum = np.min(minimum, axis=0)
    global_maximum = np.max(maximum, axis=0)
    span_x = int(global_maximum[0] - global_minimum[0]) + 1
    span_y = int(global_maximum[1] - global_minimum[1]) + 1
    if span_x <= 1 or span_y <= 1:
        raise ArtifactTileUnwrapError("tile unwrap has a collapsed global extent")

    target_cells = max(1, math.ceil(face_count / 8))
    aspect = float(span_x) / float(span_y)
    cells_x = max(1, math.ceil(math.sqrt(target_cells * aspect)))
    cells_y = max(1, math.ceil(target_cells / cells_x))
    cell_width = max(1, math.ceil(span_x / cells_x))
    cell_height = max(1, math.ceil(span_y / cells_y))

    ranges: list[tuple[int, int, int, int]] = []
    cells: dict[tuple[int, int], list[int]] = {}
    assignments = 0
    for face_index in range(face_count):
        poll_cancellation(cancellation_probe, face_index)
        min_x = int((int(minimum[face_index, 0]) - int(global_minimum[0])) // cell_width)
        max_x = int((int(maximum[face_index, 0]) - int(global_minimum[0])) // cell_width)
        min_y = int((int(minimum[face_index, 1]) - int(global_minimum[1])) // cell_height)
        max_y = int((int(maximum[face_index, 1]) - int(global_minimum[1])) // cell_height)
        value = (min_x, max_x, min_y, max_y)
        ranges.append(value)
        face_assignments = (max_x - min_x + 1) * (max_y - min_y + 1)
        assignments += face_assignments
        if assignments > MAX_TILE_UNWRAP_GRID_ASSIGNMENTS:
            raise ArtifactTileUnwrapError(
                "tile unwrap overlap grid exceeds its bounded assignment budget"
            )
        for cell_x in range(min_x, max_x + 1):
            for cell_y in range(min_y, max_y + 1):
                cells.setdefault((cell_x, cell_y), []).append(face_index)

    # ``examined_pairs`` budgets the *expensive* exact overlap test.  It is a
    # runtime guard only: it never reaches the recipe, the QC block or the
    # receipt, so where it counts cannot move any recorded hash.  Counting
    # candidates before the cheap bounding-box rejection made the budget scale
    # with grid occupancy rather than with real work, which tripped at roughly
    # 55,000 faces and put the documented 250,000-face limit out of reach.
    candidate_pairs = 0
    examined_pairs = 0
    for cell_index, cell in enumerate(sorted(cells)):
        poll_cancellation(cancellation_probe, cell_index)
        members = cells[cell]
        for left_offset, left in enumerate(members):
            left_range = ranges[left]
            for right in members[left_offset + 1 :]:
                right_range = ranges[right]
                canonical_cell = (
                    max(left_range[0], right_range[0]),
                    max(left_range[2], right_range[2]),
                )
                if canonical_cell != cell:
                    continue
                candidate_pairs += 1
                poll_cancellation(cancellation_probe, candidate_pairs)
                if (
                    min(
                        int(maximum[left, 0]),
                        int(maximum[right, 0]),
                    )
                    <= max(
                        int(minimum[left, 0]),
                        int(minimum[right, 0]),
                    )
                    or min(
                        int(maximum[left, 1]),
                        int(maximum[right, 1]),
                    )
                    <= max(
                        int(minimum[left, 1]),
                        int(minimum[right, 1]),
                    )
                ):
                    continue
                examined_pairs += 1
                if examined_pairs > MAX_TILE_UNWRAP_OVERLAP_CANDIDATES:
                    raise ArtifactTileUnwrapError(
                        "tile unwrap overlap QC exceeds its examined-pair budget"
                    )
                if _positive_area_triangle_overlap(
                    triangles[left], triangles[right]
                ):
                    return 1
    return 0


def _orientation_qc(
    uv_um: np.ndarray,
    faces: np.ndarray,
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> dict[str, int]:
    """Classify quantized triangle orientation without float cancellation."""

    uv = np.asarray(uv_um, dtype=np.int64)
    triangles = uv[np.asarray(faces, dtype=np.int32)]
    degenerate = 0
    positive = 0
    negative = 0
    for face_index, triangle in enumerate(triangles):
        poll_cancellation(cancellation_probe, face_index)
        ax = int(triangle[1, 0]) - int(triangle[0, 0])
        ay = int(triangle[1, 1]) - int(triangle[0, 1])
        bx = int(triangle[2, 0]) - int(triangle[0, 0])
        by = int(triangle[2, 1]) - int(triangle[0, 1])
        signed_twice_area = ax * by - ay * bx
        if signed_twice_area > 0:
            positive += 1
        elif signed_twice_area < 0:
            negative += 1
        else:
            degenerate += 1
    foldovers = min(positive, negative)
    return {
        "degenerate_uv_face_count": degenerate,
        "foldover_face_count": foldovers,
        "negative_orientation_face_count": negative,
        "positive_orientation_face_count": positive,
    }


def recompute_tile_unwrap_payload_qc(
    unwrap: TileUnwrapMesh,
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> dict[str, int]:
    """Recompute every topology/orientation claim available from payload bytes."""

    if not isinstance(unwrap, TileUnwrapMesh):
        raise ArtifactTileUnwrapError("unwrap must be a TileUnwrapMesh")
    topology = _surface_topology_qc(
        unwrap.faces,
        vertex_count=unwrap.vertex_count,
        cancellation_probe=cancellation_probe,
    )
    orientation = _orientation_qc(
        unwrap.uv_um,
        unwrap.faces,
        cancellation_probe=cancellation_probe,
    )
    overlap = _uv_overlap_pair_count(
        unwrap.uv_um,
        unwrap.faces,
        cancellation_probe=cancellation_probe,
    )
    return {
        **topology,
        **orientation,
        "uv_overlap_pair_count": overlap,
    }


def extract_tile_unwrap(
    mesh: MeshData,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[TileUnwrapMesh, dict[str, Any]]:
    """Compute one authoritative unwrap or fail without substituting a method."""

    validated = validate_tile_unwrap_recipe(recipe)
    selection = validated["selection"]
    assert isinstance(selection, Mapping)
    source_face_indices = selection_face_indices(selection)
    if int(selection["total_face_count"]) != int(np.asarray(mesh.faces).shape[0]):
        raise ArtifactTileUnwrapError(
            "tile unwrap selection does not match canonical mesh face count"
        )
    raise_if_cancelled(cancellation_probe)
    submesh, source_vertex_indices, source_face_indices = _submesh_for_selection(
        mesh,
        source_face_indices,
    )
    topology = _surface_topology_qc(
        submesh.faces,
        vertex_count=submesh.n_vertices,
        cancellation_probe=cancellation_probe,
    )
    if topology["connected_component_count"] != 1:
        raise ArtifactTileUnwrapError(
            "recording surface must be one edge-connected component"
        )
    if topology["duplicate_face_count"] != 0:
        raise ArtifactTileUnwrapError("recording surface contains duplicate faces")
    if topology["nonmanifold_edge_count"] != 0:
        raise ArtifactTileUnwrapError("recording surface contains non-manifold edges")
    if topology["inconsistent_oriented_edge_count"] != 0:
        raise ArtifactTileUnwrapError(
            "recording surface has inconsistent triangle orientation"
        )
    if topology["boundary_loop_count"] < 1:
        raise ArtifactTileUnwrapError(
            "recording surface must have a closed, non-branched open boundary"
        )
    raise_if_cancelled(cancellation_probe)
    result = sectionwise_cylindrical_parameterization(
        submesh,
        axis=validated["longitudinal_axis"],
        n_sections=int(validated["n_sections"]),
        record_view=str(validated["record_view"]),
        seam_angle_microdegrees=validated.get("seam_angle_microdegrees"),
        return_meta=True,
        cancellation_probe=cancellation_probe,
    )
    if not isinstance(result, tuple):  # pragma: no cover - return_meta contract
        raise ArtifactTileUnwrapError("sectionwise unwrap returned no quality metadata")
    uv, meta = result
    if bool(meta.get("sectionwise_fallback", False)):
        reason = str(meta.get("sectionwise_reason", "sectionwise_internal_fallback"))
        raise ArtifactTileUnwrapError(
            f"authoritative tile unwrap rejected algorithm fallback: {reason}"
        )
    # Stations are drawn from mesh quantiles, so ties collapse and the achieved
    # count can fall short of the requested one.  The record contract requires
    # the two to agree, so report it here -- with the count the mesh can
    # actually support -- instead of discarding the finished computation at
    # commit time behind a message that does not say what to change.
    requested_sections = int(validated["n_sections"])
    achieved_sections = int(meta.get("section_count", 0))
    if achieved_sections != requested_sections:
        raise ArtifactTileUnwrapError(
            f"tile unwrap requested {requested_sections} sections but this "
            f"recording surface supports {achieved_sections}; set n_sections to "
            f"{achieved_sections} or select a longer surface"
        )
    uv_mm = np.asarray(uv, dtype=np.float64)
    if uv_mm.shape != (submesh.n_vertices, 2) or not np.isfinite(uv_mm).all():
        raise ArtifactTileUnwrapError("sectionwise unwrap returned invalid coordinates")
    uv_mm = uv_mm - np.min(uv_mm, axis=0, keepdims=True)
    scaled = uv_mm * (1000.0 / TILE_UNWRAP_COORDINATE_QUANTUM_UM)
    if not np.isfinite(scaled).all() or np.any(scaled > MAX_TILE_UNWRAP_COORDINATE_UM):
        raise ArtifactTileUnwrapError("tile unwrap exceeds the exact coordinate grid")
    uv_um = np.rint(scaled).astype(np.int64)
    quantized_uv_mm = uv_um.astype(np.float64) / 1000.0
    face_distortion = compute_face_distortion(submesh, quantized_uv_mm)
    summary = distortion_summary(face_distortion)
    needs_fallback, reason = sectionwise_quality_gate(meta, distortion_summary=summary)
    if needs_fallback:
        raise ArtifactTileUnwrapError(
            f"authoritative tile unwrap failed its quality gate: {reason}"
        )
    unwrap = TileUnwrapMesh(
        uv_um=uv_um,
        faces=np.asarray(submesh.faces, dtype=np.int32),
        source_vertex_indices=source_vertex_indices,
        source_face_indices=source_face_indices,
        axis=str(validated["longitudinal_axis"]),
        record_view=str(validated["record_view"]),
    )
    orientation = _orientation_qc(
        unwrap.uv_um,
        unwrap.faces,
        cancellation_probe=cancellation_probe,
    )
    if orientation["degenerate_uv_face_count"] > 0:
        raise ArtifactTileUnwrapError(
            "authoritative tile unwrap contains faces collapsed by the 1 um grid"
        )
    if orientation["foldover_face_count"] > 0:
        raise ArtifactTileUnwrapError(
            "authoritative tile unwrap contains orientation foldovers"
        )
    uv_overlap_pair_count = _uv_overlap_pair_count(
        unwrap.uv_um,
        unwrap.faces,
        cancellation_probe=cancellation_probe,
    )
    if uv_overlap_pair_count > 0:
        raise ArtifactTileUnwrapError(
            "authoritative tile unwrap contains positive-area global UV overlap"
        )
    receipt = unwrap.receipt(selection_sha256=str(selection["selection_sha256"]))
    qc = {
        **orientation,
        **topology,
        "distortion_max_millionths": _rounded_millionths(float(summary["max"])),
        "distortion_mean_millionths": _rounded_millionths(float(summary["mean"])),
        "distortion_median_millionths": _rounded_millionths(float(summary["median"])),
        "distortion_p95_millionths": _rounded_millionths(float(summary["p95"])),
        "face_count": unwrap.face_count,
        "height_um": int(receipt["height_mm_exact"]["numerator"]),
        "section_centerline_length_um": int(
            np.rint(float(meta.get("section_centerline_length", 0.0)) * 1000.0)
        ),
        "section_count": int(meta.get("section_count", 0)),
        "section_fit_valid_count": int(meta.get("section_fit_valid_count", 0)),
        "section_mean_radius_um": int(
            np.rint(float(meta.get("section_mean_radius", 0.0)) * 1000.0)
        ),
        "section_mean_span_microdegrees": int(
            np.rint(float(meta.get("section_mean_span_deg", 0.0)) * 1_000_000.0)
        ),
        "section_row_shift_applied": bool(
            meta.get("section_row_shift_applied", False)
        ),
        "section_row_shift_max_um": int(
            np.rint(float(meta.get("section_row_shift_max_world", 0.0)) * 1000.0)
        ),
        "section_row_shift_station_count": int(
            meta.get("section_row_shift_station_count", 0)
        ),
        "selected_face_count": int(selection["selected_face_count"]),
        "selection_sha256": str(selection["selection_sha256"]),
        "unwrap_sha256": str(receipt["unwrap_sha256"]),
        "uv_overlap_pair_count": uv_overlap_pair_count,
        "vertex_count": unwrap.vertex_count,
        "width_um": int(receipt["width_mm_exact"]["numerator"]),
    }
    raise_if_cancelled(cancellation_probe)
    return unwrap, qc


@dataclass(frozen=True, slots=True)
class ArtifactTileUnwrapComputation:
    context: OperationContext
    projection_snapshot: ArtifactProjectionSnapshot
    unwrap: TileUnwrapMesh
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.context, OperationContext):
            raise ArtifactTileUnwrapError("context must be an OperationContext")
        if not isinstance(self.projection_snapshot, ArtifactProjectionSnapshot):
            raise ArtifactTileUnwrapError("projection_snapshot is invalid")
        if not isinstance(self.unwrap, TileUnwrapMesh):
            raise ArtifactTileUnwrapError("unwrap must be a TileUnwrapMesh")
        validated_recipe = validate_tile_unwrap_recipe(self.recipe)
        if canonical_recipe_hash(validated_recipe) != self.context.recipe_hash:
            raise ArtifactTileUnwrapError("tile unwrap recipe does not match context")
        selection = validated_recipe["selection"]
        assert isinstance(selection, Mapping)
        if self.context.selection_hash != selection["selection_sha256"]:
            raise ArtifactTileUnwrapError(
                "tile unwrap selection does not match context"
            )
        snapshot = self.projection_snapshot
        if (
            snapshot.geometry_revision_id != self.context.geometry_revision_id
            or snapshot.source_metadata_revision_id
            != self.context.source_metadata_revision_id
            or snapshot.align_revision_id != self.context.align_revision_id
            or tuple(self.context.source_asset_ids) != (snapshot.source_asset_id,)
        ):
            raise ArtifactTileUnwrapError(
                "tile unwrap projection snapshot does not match context"
            )
        object.__setattr__(self, "recipe", MappingProxyType(validated_recipe))
        object.__setattr__(self, "qc", MappingProxyType(dict(self.qc)))

    def recipe_dict(self) -> dict[str, Any]:
        return dict(self.recipe)

    def qc_dict(self) -> dict[str, Any]:
        return dict(self.qc)


def compute_artifact_tile_unwrap(
    session: ArtifactSession,
    *,
    longitudinal_axis: str,
    record_view: str,
    selected_face_indices: Sequence[int] | np.ndarray | None = None,
    n_sections: int = 32,
    seam_angle_microdegrees: int | None = None,
    cancellation_probe: CancellationProbe | None = None,
) -> ArtifactTileUnwrapComputation:
    if not isinstance(session, ArtifactSession):
        raise ArtifactTileUnwrapError("session must be an ArtifactSession")
    face_count = int(np.asarray(session.source_mesh.faces).shape[0])
    recipe = tile_unwrap_recipe(
        longitudinal_axis=longitudinal_axis,
        record_view=record_view,
        total_face_count=face_count,
        selected_face_indices=selected_face_indices,
        n_sections=n_sections,
        seam_angle_microdegrees=seam_angle_microdegrees,
    )
    return _compute_artifact_tile_unwrap_with_validated_recipe(
        session,
        recipe,
        cancellation_probe=cancellation_probe,
    )


def _compute_artifact_tile_unwrap_with_validated_recipe(
    session: ArtifactSession,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None,
) -> ArtifactTileUnwrapComputation:
    """Compute without silently upgrading a persisted recipe version."""

    selection = recipe["selection"]
    assert isinstance(selection, Mapping)
    try:
        context = session.capture_operation(
            recipe=recipe,
            selection_hash=str(selection["selection_sha256"]),
        )
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactTileUnwrapError(str(exc)) from exc
    unwrap, qc = extract_tile_unwrap(
        projection.mesh,
        recipe,
        cancellation_probe=cancellation_probe,
    )
    return ArtifactTileUnwrapComputation(
        context=context,
        projection_snapshot=projection.snapshot,
        unwrap=unwrap,
        recipe=recipe,
        qc=qc,
    )


def compute_artifact_tile_unwrap_from_recipe(
    session: ArtifactSession,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> ArtifactTileUnwrapComputation:
    if not isinstance(session, ArtifactSession):
        raise ArtifactTileUnwrapError("session must be an ArtifactSession")
    validated = validate_tile_unwrap_recipe(recipe)
    return _compute_artifact_tile_unwrap_with_validated_recipe(
        session,
        validated,
        cancellation_probe=cancellation_probe,
    )


def tile_unwrap_computation_matches_active_projection(
    session: ArtifactSession,
    computation: ArtifactTileUnwrapComputation,
) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(
        computation, ArtifactTileUnwrapComputation
    ):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    expected = computation.projection_snapshot
    return current.has_same_render_projection(expected)


def require_current_tile_unwrap_computation(
    session: ArtifactSession,
    computation: ArtifactTileUnwrapComputation,
) -> None:
    if not tile_unwrap_computation_matches_active_projection(session, computation):
        raise ArtifactTileUnwrapError(
            "tile unwrap computation is stale for the active scene projection"
        )


def commit_artifact_tile_unwrap(
    session: ArtifactSession,
    computation: ArtifactTileUnwrapComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    """Append an unwrap receipt at the computation's captured context."""

    if not isinstance(session, ArtifactSession):
        raise ArtifactTileUnwrapError("session must be an ArtifactSession")
    if not isinstance(computation, ArtifactTileUnwrapComputation):
        raise ArtifactTileUnwrapError(
            "computation must be an ArtifactTileUnwrapComputation"
        )
    from .artifact_tile_unwrap_record import (  # noqa: PLC0415
        ArtifactTileUnwrapRecordError,
        append_tile_unwrap_record_from_context,
    )

    try:
        document = append_tile_unwrap_record_from_context(
            session.document,
            context=computation.context,
            unwrap=computation.unwrap,
            recipe=computation.recipe,
            record_id=record_id,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=depends_on_record_ids,
            qc=computation.qc,
        )
        return session.with_document(document)
    except (ArtifactTileUnwrapRecordError, ArtifactSessionError) as exc:
        raise ArtifactTileUnwrapError(str(exc)) from exc


__all__ = [
    "ArtifactTileUnwrapComputation",
    "ArtifactTileUnwrapError",
    "MAX_TILE_UNWRAP_COORDINATE_UM",
    "MAX_TILE_UNWRAP_FACES",
    "MAX_TILE_UNWRAP_PAYLOAD_BYTES",
    "MAX_TILE_UNWRAP_SELECTION_RANGES",
    "MAX_TILE_UNWRAP_VERTICES",
    "TILE_UNWRAP_ALGORITHM",
    "TILE_UNWRAP_ALGORITHM_VERSION",
    "TILE_UNWRAP_COORDINATE_QUANTUM_UM",
    "TILE_UNWRAP_COORDINATE_SPACE",
    "TILE_UNWRAP_GEOMETRY_REF_PREFIX",
    "TILE_UNWRAP_HASH_SCOPE",
    "TILE_UNWRAP_OUTPUT_SCHEMA_VERSION",
    "TILE_UNWRAP_RECIPE_SCHEMA_VERSION",
    "TILE_UNWRAP_SELECTION_SCHEMA_VERSION",
    "TileUnwrapMesh",
    "commit_artifact_tile_unwrap",
    "compute_artifact_tile_unwrap",
    "compute_artifact_tile_unwrap_from_recipe",
    "extract_tile_unwrap",
    "require_current_tile_unwrap_computation",
    "recompute_tile_unwrap_payload_qc",
    "selection_face_indices",
    "tile_unwrap_computation_matches_active_projection",
    "tile_unwrap_recipe",
    "validate_tile_unwrap_recipe",
    "validate_tile_unwrap_selection",
]
