"""Bounded, reproducible admission for authoritative scanned meshes.

The parser is deliberately not the authority for deciding whether a decoded
mesh is safe to enter the measurement workflow.  This module applies one
fixed profile before expensive downstream copies are made and produces a
path-free receipt which can be persisted with the geometry revision.

The limits describe the first public Windows workbench profile.  They are
fixed scientific/workflow parameters rather than environment tuning knobs so
the same source and locked parser cannot be accepted under one installation
and silently acquire different provenance under another.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
import os
import re
import struct
from typing import Any, BinaryIO, Mapping, Sequence

import numpy as np

from .source_manifest import (
    MAX_SOURCE_MANIFEST_ENTRIES,
    SourceManifestError,
    resolve_logical_reference,
)


MESH_ADMISSION_PROFILE = "archmeshrubbing.authoritative_mesh_admission"
MESH_ADMISSION_SCHEMA_VERSION = "1.0.0"
MESH_ADMISSION_STATUS = "accepted"

# These limits are shared by the authoritative measurement kernels.  A larger
# scan must be deliberately segmented or a future versioned profile must raise
# the limits after its Windows memory envelope has been measured.
MAX_MESH_SOURCE_BYTES = 4 * 1024 * 1024 * 1024
# Trimesh's locked OBJ/OFF/ASCII PLY/STL loaders materialize the complete text
# payload (and commonly decoded copies) before returning arrays.  Keep those
# parser inputs within a separately measured Windows envelope; binary formats
# retain the larger primary-source cap above.
MAX_MESH_TEXT_SOURCE_BYTES = 256 * 1024 * 1024
MAX_MESH_VERTICES = 5_000_000
MAX_MESH_TRIANGLES = 2_000_000
MAX_MESH_DECODED_ARRAY_BYTES = 2 * 1024 * 1024 * 1024
MAX_MESH_ESTIMATED_PEAK_BYTES = 3 * 1024 * 1024 * 1024
MAX_MESH_TEXTURE_BYTES = 512 * 1024 * 1024
MAX_MESH_DEPENDENCY_BYTES = 512 * 1024 * 1024
MAX_MESH_DEPENDENCY_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_MESH_HEADER_BYTES = 1024 * 1024
MAX_MESH_GLTF_JSON_BYTES = 16 * 1024 * 1024
# Trimesh preserves every binary PLY element/property in ``_ply_raw`` until
# Scene materialization completes.  Bound data outside XYZ positions and the
# selected face-index list before the parser can retain or copy it.
MAX_MESH_PLY_AUXILIARY_BYTES = 128 * 1024 * 1024
MAX_MESH_PLY_VERTEX_PROPERTIES = 16
MAX_MESH_PLY_FACE_PROPERTIES = 8
WINDOWS_RUNTIME_MEMORY_RESERVE_BYTES = 512 * 1024 * 1024

_ARRAY_CHECK_CHUNK = 250_000
_GEOMETRY_HASH_DOMAIN = b"archmeshrubbing.geometry\x00"
_GEOMETRY_HASH_SCOPE = "positions-f64le+triangles-i32le/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DECLARATION_KINDS = frozenset(
    {
        "ascii_stl_stream",
        "binary_stl_header",
        "glb_json_chunk",
        "gltf_json",
        "not_available",
        "obj_stream",
        "off_header",
        "ply_ascii_header",
        "ply_binary_header",
    }
)
_MAX_MESH_TEXT_LINE_BYTES = 1024 * 1024
_PLY_SCALAR_FORMATS = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}
_PLY_INTEGER_FORMATS = frozenset(
    key for key, value in _PLY_SCALAR_FORMATS.items() if value not in {"f", "d"}
)
_GLTF_COMPONENT_BYTES = {
    5120: 1,
    5121: 1,
    5122: 2,
    5123: 2,
    5125: 4,
    5126: 4,
}
_GLTF_TYPE_SHAPES = {
    "SCALAR": (1, 1),
    "VEC2": (1, 2),
    "VEC3": (1, 3),
    "VEC4": (1, 4),
    "MAT2": (2, 2),
    "MAT3": (3, 3),
    "MAT4": (4, 4),
}


class MeshAdmissionError(ValueError):
    """A source cannot safely enter the authoritative measurement workflow."""


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise MeshAdmissionError(f"{name} must be an integer")
    number = int(value)
    if number < minimum or number > maximum:
        raise MeshAdmissionError(
            f"{name} must be in the inclusive range {minimum}..{maximum}"
        )
    return number


def _optional_count(value: object, *, name: str, maximum: int) -> int | None:
    if value is None:
        return None
    return _strict_int(value, name=name, minimum=0, maximum=maximum)


def _source_format(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MeshAdmissionError("source_format must be a non-empty string")
    normalized = value.strip().lower().removeprefix(".")
    if normalized not in {"obj", "ply", "stl", "off", "gltf", "glb"}:
        raise MeshAdmissionError(f"unsupported source_format: {normalized!r}")
    return normalized


def _exact_mapping(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MeshAdmissionError(f"{name} must be an object")
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise MeshAdmissionError(f"{name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise MeshAdmissionError(f"{name} has unknown fields: {', '.join(unknown)}")
    return value


def _raise_count_limit(*, name: str, observed: int, maximum: int) -> None:
    if observed > maximum:
        raise MeshAdmissionError(
            f"mesh admission rejected {name}: {observed:,} exceeds the "
            f"Windows workflow limit of {maximum:,}"
        )


def _require_text_source_budget(size_bytes: int, *, label: str) -> None:
    if size_bytes > MAX_MESH_TEXT_SOURCE_BYTES:
        raise MeshAdmissionError(
            f"mesh admission rejected {label}: {size_bytes:,} bytes exceeds the "
            f"Windows text-parser limit of {MAX_MESH_TEXT_SOURCE_BYTES:,} bytes"
        )


@dataclass(frozen=True, slots=True)
class MeshSourcePreflight:
    source_format: str
    source_size_bytes: int
    declaration_kind: str = "not_available"
    declared_vertex_count: int | None = None
    declared_face_element_count: int | None = None
    declared_triangle_count: int | None = None
    declared_parser_bytes: int | None = None
    # Runtime-only exact bindings for non-data glTF buffer URIs.  Paths never
    # enter the durable admission receipt; the closed source manifest records
    # their content identities separately.
    external_buffer_byte_lengths: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_format", _source_format(self.source_format))
        object.__setattr__(
            self,
            "source_size_bytes",
            _strict_int(
                self.source_size_bytes,
                name="source_size_bytes",
                minimum=1,
                maximum=MAX_MESH_SOURCE_BYTES,
            ),
        )
        kind = str(self.declaration_kind or "").strip()
        if kind not in _DECLARATION_KINDS:
            raise MeshAdmissionError(
                f"unsupported mesh declaration kind: {kind!r}"
            )
        object.__setattr__(self, "declaration_kind", kind)
        if self.source_format in {"obj", "off"} or kind in {
            "ascii_stl_stream",
            "ply_ascii_header",
        }:
            _require_text_source_budget(
                self.source_size_bytes,
                label=f"{self.source_format.upper()} text source",
            )
        object.__setattr__(
            self,
            "declared_vertex_count",
            _optional_count(
                self.declared_vertex_count,
                name="declared_vertex_count",
                maximum=MAX_MESH_VERTICES,
            ),
        )
        object.__setattr__(
            self,
            "declared_face_element_count",
            _optional_count(
                self.declared_face_element_count,
                name="declared_face_element_count",
                maximum=MAX_MESH_TRIANGLES,
            ),
        )
        object.__setattr__(
            self,
            "declared_triangle_count",
            _optional_count(
                self.declared_triangle_count,
                name="declared_triangle_count",
                maximum=MAX_MESH_TRIANGLES,
            ),
        )
        parser_bytes = self.declared_parser_bytes
        if parser_bytes is not None:
            parser_bytes = _strict_int(
                parser_bytes,
                name="declared_parser_bytes",
                minimum=1,
                maximum=MAX_MESH_DECODED_ARRAY_BYTES,
            )
            if kind not in {
                "glb_json_chunk",
                "gltf_json",
                "ply_ascii_header",
                "ply_binary_header",
            }:
                raise MeshAdmissionError(
                    "declared parser bytes are unsupported for this source kind"
                )
        if kind in {
            "glb_json_chunk",
            "gltf_json",
            "ply_ascii_header",
            "ply_binary_header",
        } and parser_bytes is None:
            raise MeshAdmissionError(
                "declared PLY/glTF source must bind its parser footprint"
            )
        object.__setattr__(self, "declared_parser_bytes", parser_bytes)
        try:
            raw_bindings = tuple(self.external_buffer_byte_lengths)
        except TypeError as exc:
            raise MeshAdmissionError(
                "external glTF buffer bindings must be iterable"
            ) from exc
        normalized_bindings: dict[str, int] = {}
        for raw_binding in raw_bindings:
            if not isinstance(raw_binding, (tuple, list)) or len(raw_binding) != 2:
                raise MeshAdmissionError(
                    "external glTF buffer binding must contain URI and byte length"
                )
            raw_uri, raw_length = raw_binding
            if not isinstance(raw_uri, str) or not raw_uri:
                raise MeshAdmissionError(
                    "external glTF buffer URI must be a non-empty string"
                )
            try:
                logical_path = resolve_logical_reference("", raw_uri)
            except SourceManifestError as exc:
                raise MeshAdmissionError(
                    f"external glTF buffer URI is unsafe: {raw_uri!r}"
                ) from exc
            byte_length = _strict_int(
                raw_length,
                name=f"external glTF buffer {logical_path!r} byteLength",
                minimum=1,
                maximum=MAX_MESH_DEPENDENCY_BYTES,
            )
            previous = normalized_bindings.get(logical_path)
            if previous is not None and previous != byte_length:
                raise MeshAdmissionError(
                    "one external glTF buffer URI has conflicting byte lengths"
                )
            normalized_bindings[logical_path] = byte_length
        if len(normalized_bindings) > MAX_SOURCE_MANIFEST_ENTRIES - 1:
            raise MeshAdmissionError(
                "external glTF buffers exceed the portable dependency entry budget"
            )
        if sum(normalized_bindings.values()) > MAX_MESH_DEPENDENCY_TOTAL_BYTES:
            raise MeshAdmissionError(
                "external glTF buffers exceed the fixed dependency byte budget"
            )
        if normalized_bindings and self.source_format not in {"gltf", "glb"}:
            raise MeshAdmissionError(
                "external glTF buffer bindings require a glTF or GLB source"
            )
        object.__setattr__(
            self,
            "external_buffer_byte_lengths",
            tuple(sorted(normalized_bindings.items())),
        )


@dataclass(frozen=True, slots=True)
class _PlyProperty:
    name: str
    scalar_type: str | None = None
    list_count_type: str | None = None
    list_item_type: str | None = None


@dataclass(frozen=True, slots=True)
class _PlyElement:
    name: str
    count: int
    properties: tuple[_PlyProperty, ...]


@dataclass(frozen=True, slots=True)
class _PlyHeader:
    encoding: str
    header_bytes: int
    elements: tuple[_PlyElement, ...]


@dataclass(frozen=True, slots=True)
class _PlyLayout:
    vertex: _PlyElement
    face: _PlyElement
    vertex_indices: _PlyProperty
    texcoord: _PlyProperty | None


@dataclass(frozen=True, slots=True)
class DecodedMeshAdmission:
    vertex_count: int
    triangle_count: int
    array_bytes: int
    estimated_peak_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vertex_count",
            _strict_int(
                self.vertex_count,
                name="decoded.vertex_count",
                minimum=3,
                maximum=MAX_MESH_VERTICES,
            ),
        )
        object.__setattr__(
            self,
            "triangle_count",
            _strict_int(
                self.triangle_count,
                name="decoded.triangle_count",
                minimum=1,
                maximum=MAX_MESH_TRIANGLES,
            ),
        )
        object.__setattr__(
            self,
            "array_bytes",
            _strict_int(
                self.array_bytes,
                name="decoded.array_bytes",
                minimum=1,
                maximum=MAX_MESH_DECODED_ARRAY_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "estimated_peak_bytes",
            _strict_int(
                self.estimated_peak_bytes,
                name="decoded.estimated_peak_bytes",
                minimum=self.array_bytes,
                maximum=MAX_MESH_ESTIMATED_PEAK_BYTES,
            ),
        )


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MeshAdmissionError(
            f"{name} must contain 64 lowercase hexadecimal characters"
        )
    return value


def _parse_nonnegative_ascii_int(value: str, *, name: str) -> int:
    token = value.strip()
    if not token.isascii() or not token.isdecimal():
        raise MeshAdmissionError(f"{name} must be a non-negative decimal integer")
    return int(token)


def _parse_ply_header(prefix: bytes) -> _PlyHeader:
    marker = re.search(
        br"(?:^|\r\n|\n|\r)end_header(?:\r\n|\n|\r)",
        prefix,
    )
    if marker is None:
        if len(prefix) > MAX_MESH_HEADER_BYTES:
            raise MeshAdmissionError(
                f"PLY header exceeds the {MAX_MESH_HEADER_BYTES:,}-byte safety limit"
            )
        raise MeshAdmissionError("PLY source has no bounded end_header marker")
    header_bytes = int(marker.end())
    try:
        lines = prefix[:header_bytes].decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise MeshAdmissionError("PLY header must be ASCII") from exc
    if not lines or lines[0].strip() != "ply":
        raise MeshAdmissionError("PLY source is missing the ply magic header")

    encoding = ""
    elements: list[_PlyElement] = []
    current_name: str | None = None
    current_count = 0
    current_properties: list[_PlyProperty] = []

    def finish_element() -> None:
        nonlocal current_name, current_count, current_properties
        if current_name is None:
            return
        elements.append(
            _PlyElement(
                name=current_name,
                count=current_count,
                properties=tuple(current_properties),
            )
        )
        current_name = None
        current_count = 0
        current_properties = []

    for line in lines[1:]:
        tokens = line.strip().split()
        if not tokens or tokens[0] in {"comment", "obj_info"}:
            continue
        if tokens[0] == "format":
            if len(tokens) != 3 or tokens[2] != "1.0":
                raise MeshAdmissionError("PLY source must use format version 1.0")
            if encoding:
                raise MeshAdmissionError("PLY source has multiple format declarations")
            encoding = tokens[1]
            if encoding not in {
                "ascii",
                "binary_little_endian",
                "binary_big_endian",
            }:
                raise MeshAdmissionError("PLY source encoding is unsupported")
            continue
        if tokens[0] == "element":
            if len(tokens) != 3:
                raise MeshAdmissionError("PLY element declaration is malformed")
            finish_element()
            current_name = tokens[1]
            current_count = _parse_nonnegative_ascii_int(
                tokens[2], name=f"PLY element {tokens[1]} count"
            )
            if current_count > MAX_MESH_VERTICES + MAX_MESH_TRIANGLES:
                raise MeshAdmissionError(
                    f"PLY element {tokens[1]} exceeds the bounded row budget"
                )
            continue
        if tokens[0] == "property":
            if current_name is None:
                raise MeshAdmissionError("PLY property appears before an element")
            if len(tokens) == 3:
                scalar_type = tokens[1].lower()
                if scalar_type not in _PLY_SCALAR_FORMATS:
                    raise MeshAdmissionError(
                        f"PLY scalar type is unsupported: {scalar_type!r}"
                    )
                current_properties.append(
                    _PlyProperty(name=tokens[2], scalar_type=scalar_type)
                )
                continue
            if len(tokens) == 5 and tokens[1] == "list":
                count_type = tokens[2].lower()
                item_type = tokens[3].lower()
                if (
                    count_type not in _PLY_SCALAR_FORMATS
                    or item_type not in _PLY_SCALAR_FORMATS
                ):
                    raise MeshAdmissionError("PLY list scalar type is unsupported")
                current_properties.append(
                    _PlyProperty(
                        name=tokens[4],
                        list_count_type=count_type,
                        list_item_type=item_type,
                    )
                )
                continue
            raise MeshAdmissionError("PLY property declaration is malformed")
        if tokens[0] == "end_header":
            if len(tokens) != 1:
                raise MeshAdmissionError("PLY end_header declaration is malformed")
            break
        raise MeshAdmissionError(
            f"PLY header directive is unsupported: {tokens[0]!r}"
        )
    finish_element()
    if not encoding:
        raise MeshAdmissionError("PLY source is missing its format declaration")
    if not elements:
        raise MeshAdmissionError("PLY source has no element declarations")
    return _PlyHeader(
        encoding=encoding,
        header_bytes=header_bytes,
        elements=tuple(elements),
    )


def _ply_supported_layout(header: _PlyHeader) -> _PlyLayout:
    """Accept only the PLY structures measured against the locked parser.

    Trimesh retains all declared PLY elements in ``metadata['_ply_raw']``.
    Rejecting arbitrary elements and list properties here prevents a source
    from hiding parser-owned arrays outside the admitted mesh footprint.
    """

    if tuple(element.name for element in header.elements) != ("vertex", "face"):
        raise MeshAdmissionError(
            "authoritative PLY must contain exactly vertex then face elements"
        )
    vertex, face = header.elements
    if vertex.count < 3:
        raise MeshAdmissionError("PLY must declare at least three vertices")
    if face.count < 1:
        raise MeshAdmissionError("PLY must declare at least one face")
    if len(vertex.properties) > MAX_MESH_PLY_VERTEX_PROPERTIES:
        raise MeshAdmissionError(
            "PLY vertex properties exceed the fixed parser profile"
        )
    if len(face.properties) > MAX_MESH_PLY_FACE_PROPERTIES:
        raise MeshAdmissionError("PLY face properties exceed the fixed parser profile")

    for element in (vertex, face):
        names = [item.name for item in element.properties]
        if len(names) != len(set(names)):
            raise MeshAdmissionError(
                f"PLY element {element.name!r} contains duplicate property names"
            )
    vertex_properties = {item.name: item for item in vertex.properties}
    if not {"x", "y", "z"}.issubset(vertex_properties):
        raise MeshAdmissionError("PLY vertex element must declare scalar x, y, and z")
    if any(item.scalar_type is None for item in vertex.properties):
        raise MeshAdmissionError("PLY vertex list properties are not admitted")

    list_properties = [
        item for item in face.properties if item.list_count_type is not None
    ]
    index_properties = [
        item
        for item in list_properties
        if item.name in {"vertex_index", "vertex_indices"}
    ]
    if len(index_properties) != 1:
        raise MeshAdmissionError(
            "PLY face element must declare exactly one named vertex-index list"
        )
    index_property = index_properties[0]
    assert index_property.list_count_type is not None
    assert index_property.list_item_type is not None
    if (
        index_property.list_count_type not in _PLY_INTEGER_FORMATS
        or index_property.list_item_type not in _PLY_INTEGER_FORMATS
    ):
        raise MeshAdmissionError("PLY face vertex-index list must use integer types")

    texcoord_properties = [item for item in list_properties if item.name == "texcoord"]
    if len(texcoord_properties) > 1:
        raise MeshAdmissionError("PLY face element contains duplicate texcoord lists")
    texcoord = texcoord_properties[0] if texcoord_properties else None
    admitted_lists = {index_property.name}
    if texcoord is not None:
        assert texcoord.list_count_type is not None
        assert texcoord.list_item_type is not None
        if texcoord.list_count_type not in _PLY_INTEGER_FORMATS:
            raise MeshAdmissionError("PLY texcoord list count must use an integer type")
        if _PLY_SCALAR_FORMATS[texcoord.list_item_type] not in {"f", "d"}:
            raise MeshAdmissionError("PLY texcoord list items must use floating-point types")
        admitted_lists.add(texcoord.name)
    if any(item.name not in admitted_lists for item in list_properties):
        raise MeshAdmissionError(
            "PLY face list properties are limited to vertex indices and texcoord"
        )
    return _PlyLayout(
        vertex=vertex,
        face=face,
        vertex_indices=index_property,
        texcoord=texcoord,
    )


def _ply_declaration(header: _PlyHeader) -> tuple[int, int]:
    layout = _ply_supported_layout(header)
    return layout.vertex.count, layout.face.count


def _off_declaration(prefix: bytes) -> tuple[int, int] | None:
    try:
        text = prefix.decode("ascii")
    except UnicodeDecodeError:
        return None
    logical_lines: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            logical_lines.append(line.split())
        if len(logical_lines) >= 2:
            break
    if not logical_lines or logical_lines[0][0] != "OFF":
        return None
    count_tokens = logical_lines[0][1:]
    if len(count_tokens) < 2:
        if len(logical_lines) < 2:
            return None
        count_tokens = logical_lines[1]
    if len(count_tokens) < 2:
        return None
    return (
        _parse_nonnegative_ascii_int(count_tokens[0], name="OFF vertex count"),
        _parse_nonnegative_ascii_int(count_tokens[1], name="OFF face count"),
    )


def _stl_declaration(prefix: bytes, source_size_bytes: int) -> tuple[int, int] | None:
    if len(prefix) < 84:
        return None
    triangle_count = int(struct.unpack_from("<I", prefix, 80)[0])
    expected_size = 84 + 50 * triangle_count
    if expected_size != source_size_bytes:
        return None
    return triangle_count * 3, triangle_count


def _gltf_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MeshAdmissionError(f"glTF JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise MeshAdmissionError(f"glTF JSON contains non-finite number {value!r}")


def _parse_gltf_json(payload: bytes) -> Mapping[str, Any]:
    if len(payload) < 1:
        raise MeshAdmissionError("glTF JSON is empty")
    if len(payload) > MAX_MESH_GLTF_JSON_BYTES:
        raise MeshAdmissionError(
            "glTF JSON exceeds the fixed 16 MiB preflight limit"
        )
    try:
        text = payload.rstrip(b" \t\r\n\x00").decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_gltf_json_object,
            parse_constant=_reject_json_constant,
        )
    except MeshAdmissionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise MeshAdmissionError("glTF JSON is malformed or too deeply nested") from exc
    if not isinstance(value, Mapping):
        raise MeshAdmissionError("glTF JSON root must be an object")
    asset = value.get("asset")
    if not isinstance(asset, Mapping) or str(asset.get("version", "")) != "2.0":
        raise MeshAdmissionError("authoritative glTF must declare asset.version 2.0")
    return value


def _read_exact(source_stream: BinaryIO, size: int, *, label: str) -> bytes:
    payload = source_stream.read(size)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise MeshAdmissionError("mesh source stream must return bytes")
    result = bytes(payload)
    if len(result) != size:
        raise MeshAdmissionError(f"{label} is truncated")
    return result


def _gltf_document_from_stream(
    source_stream: BinaryIO,
    *,
    source_format: str,
    source_size_bytes: int,
) -> tuple[Mapping[str, Any], str]:
    source_stream.seek(0)
    if source_format == "gltf":
        if source_size_bytes > MAX_MESH_GLTF_JSON_BYTES:
            raise MeshAdmissionError(
                "glTF JSON exceeds the fixed 16 MiB preflight limit; use "
                "external buffers or GLB instead of embedding large data URIs"
            )
        payload = _read_exact(
            source_stream,
            source_size_bytes,
            label="glTF JSON",
        )
        return _parse_gltf_json(payload), "gltf_json"

    header = _read_exact(source_stream, 12, label="GLB header")
    magic, version, declared_length = struct.unpack("<4sII", header)
    if magic != b"glTF" or version != 2:
        raise MeshAdmissionError("authoritative GLB must use glTF binary version 2")
    if int(declared_length) != source_size_bytes:
        raise MeshAdmissionError("GLB declared byte length differs from the source")
    chunk_header = _read_exact(source_stream, 8, label="GLB JSON chunk header")
    json_length, chunk_type = struct.unpack("<I4s", chunk_header)
    if chunk_type != b"JSON":
        raise MeshAdmissionError("GLB first chunk must contain JSON")
    if json_length < 1 or json_length > MAX_MESH_GLTF_JSON_BYTES:
        raise MeshAdmissionError(
            "GLB JSON chunk exceeds the fixed 16 MiB preflight limit"
        )
    if 20 + int(json_length) > source_size_bytes:
        raise MeshAdmissionError("GLB JSON chunk exceeds the declared file length")
    if int(json_length) % 4 != 0:
        raise MeshAdmissionError("GLB JSON chunk length must be 4-byte aligned")
    payload = _read_exact(
        source_stream,
        int(json_length),
        label="GLB JSON chunk",
    )
    document = _parse_gltf_json(payload)

    buffers = document.get("buffers", [])
    if not isinstance(buffers, list):
        raise MeshAdmissionError("glTF buffers must be an array")
    embedded_buffer_length: int | None = None
    for index, raw_buffer in enumerate(buffers):
        if not isinstance(raw_buffer, Mapping):
            raise MeshAdmissionError(f"glTF buffer[{index}] must be an object")
        if "uri" not in raw_buffer:
            if index != 0 or embedded_buffer_length is not None:
                raise MeshAdmissionError(
                    "GLB may bind only buffer[0] to its BIN chunk"
                )
            embedded_buffer_length = _strict_int(
                raw_buffer.get("byteLength"),
                name="glTF buffer[0].byteLength",
                minimum=1,
                maximum=MAX_MESH_SOURCE_BYTES,
            )

    position = 20 + int(json_length)
    has_bin_chunk = position < source_size_bytes
    if not has_bin_chunk:
        if embedded_buffer_length is not None:
            raise MeshAdmissionError("GLB embedded buffer has no BIN chunk")
        return document, "glb_json_chunk"

    if source_size_bytes - position < 8:
        raise MeshAdmissionError("GLB BIN chunk header is truncated")
    bin_header = _read_exact(source_stream, 8, label="GLB BIN chunk header")
    bin_length, bin_type = struct.unpack("<I4s", bin_header)
    if bin_type != b"BIN\x00":
        raise MeshAdmissionError("GLB second chunk must contain BIN data")
    if int(bin_length) % 4 != 0:
        raise MeshAdmissionError("GLB BIN chunk length must be 4-byte aligned")
    bin_start = position + 8
    bin_end = bin_start + int(bin_length)
    if bin_end != source_size_bytes:
        raise MeshAdmissionError(
            "GLB must contain exactly one bounded BIN chunk after JSON"
        )
    if embedded_buffer_length is None:
        raise MeshAdmissionError(
            "GLB contains a BIN chunk but buffer[0] declares an external URI"
        )
    padded_buffer_length = (embedded_buffer_length + 3) & ~3
    if int(bin_length) != padded_buffer_length:
        raise MeshAdmissionError(
            "GLB BIN chunk length differs from buffer[0].byteLength"
        )
    padding_length = padded_buffer_length - embedded_buffer_length
    if padding_length:
        source_stream.seek(bin_start + embedded_buffer_length)
        padding = _read_exact(
            source_stream,
            padding_length,
            label="GLB BIN padding",
        )
        if padding != b"\x00" * padding_length:
            raise MeshAdmissionError("GLB BIN padding must contain only zero bytes")
    return document, "glb_json_chunk"


def _gltf_data_uri_length(uri: str, *, buffer_index: int) -> int:
    header, separator, encoded = uri.partition(",")
    if (
        not separator
        or not header.startswith("data:")
        or not header.endswith(";base64")
        or uri.count("base64,") != 1
    ):
        raise MeshAdmissionError(f"glTF buffer[{buffer_index}] data URI is malformed")
    parameters = header[5:].split(";")
    if not parameters or parameters[-1] != "base64":
        raise MeshAdmissionError(
            f"glTF buffer[{buffer_index}] data URI must use canonical base64"
        )
    try:
        encoded_bytes = encoded.encode("ascii", errors="strict")
        decoded = base64.b64decode(encoded_bytes, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise MeshAdmissionError(
            f"glTF buffer[{buffer_index}] data URI has invalid base64"
        ) from exc
    return len(decoded)


def _gltf_external_buffer_bindings(
    document: Mapping[str, Any],
    *,
    source_format: str,
) -> tuple[tuple[str, int], ...]:
    buffers = document.get("buffers", [])
    if not isinstance(buffers, list):
        raise MeshAdmissionError("glTF buffers must be an array")
    bindings: dict[str, int] = {}
    for index, raw_buffer in enumerate(buffers):
        if not isinstance(raw_buffer, Mapping):
            raise MeshAdmissionError(f"glTF buffer[{index}] must be an object")
        declared_length = _strict_int(
            raw_buffer.get("byteLength"),
            name=f"glTF buffer[{index}].byteLength",
            minimum=1,
            maximum=MAX_MESH_SOURCE_BYTES,
        )
        raw_uri = raw_buffer.get("uri")
        if raw_uri is None:
            if source_format != "glb" or index != 0:
                raise MeshAdmissionError(
                    f"glTF buffer[{index}] without a URI is valid only as GLB buffer[0]"
                )
            continue
        if not isinstance(raw_uri, str) or not raw_uri:
            raise MeshAdmissionError(f"glTF buffer[{index}].uri must be non-empty text")
        # The locked Trimesh parser searches for the literal lower-case token
        # ``base64,`` anywhere in a URI before it consults the closed resolver.
        # Accept only the exact data-URI shape that therefore has identical
        # preflight and parser semantics.  In particular, a relative filename
        # containing that token must never be hashed as a dependency while the
        # parser silently decodes different bytes from the filename itself.
        if raw_uri.startswith("data:"):
            observed_length = _gltf_data_uri_length(raw_uri, buffer_index=index)
            if observed_length != declared_length:
                raise MeshAdmissionError(
                    f"glTF buffer[{index}] data URI length differs from byteLength"
                )
            continue
        if raw_uri[:5].lower() == "data:" or "base64," in raw_uri:
            raise MeshAdmissionError(
                f"glTF buffer[{index}].uri has ambiguous non-canonical base64 syntax"
            )
        try:
            logical_path = resolve_logical_reference("", raw_uri)
        except SourceManifestError as exc:
            raise MeshAdmissionError(
                f"glTF buffer[{index}] external URI is unsafe"
            ) from exc
        if declared_length > MAX_MESH_DEPENDENCY_BYTES:
            raise MeshAdmissionError(
                f"glTF buffer[{index}] exceeds the fixed dependency byte budget"
            )
        previous = bindings.get(logical_path)
        if previous is not None and previous != declared_length:
            raise MeshAdmissionError(
                "one external glTF buffer URI has conflicting byteLength declarations"
            )
        bindings[logical_path] = declared_length
    if len(bindings) > MAX_SOURCE_MANIFEST_ENTRIES - 1:
        raise MeshAdmissionError(
            "external glTF buffers exceed the portable dependency entry budget"
        )
    if sum(bindings.values()) > MAX_MESH_DEPENDENCY_TOTAL_BYTES:
        raise MeshAdmissionError(
            "external glTF buffers exceed the fixed dependency byte budget"
        )
    return tuple(sorted(bindings.items()))


def _gltf_parser_footprint(
    document: Mapping[str, Any],
    *,
    source_size_bytes: int,
    vertex_count: int,
    triangle_count: int,
) -> int:
    parser_bytes = _strict_int(
        source_size_bytes,
        name="glTF source_size_bytes",
        minimum=1,
        maximum=MAX_MESH_SOURCE_BYTES,
    )

    def add_parser_bytes(value: int, *, label: str) -> None:
        nonlocal parser_bytes
        if value < 0 or value > MAX_MESH_DECODED_ARRAY_BYTES - parser_bytes:
            raise MeshAdmissionError(
                f"glTF {label} exceed the fixed parser byte budget"
            )
        parser_bytes += value

    for index, raw_buffer in enumerate(_gltf_array(document, "buffers")):
        item = _gltf_object(raw_buffer, name=f"glTF buffer[{index}]")
        add_parser_bytes(
            _strict_int(
                item.get("byteLength"),
                name=f"glTF buffer[{index}].byteLength",
                minimum=1,
                maximum=MAX_MESH_SOURCE_BYTES,
            ),
            label="declared buffers",
        )

    # Trimesh 4.11.5 eagerly creates a bytes slice for every declared
    # bufferView, including unused and overlapping views.  Count every slice,
    # rather than unique source ranges, so duplicate declarations cannot turn
    # a small buffer into multi-gigabyte parser allocations.
    for index, raw_view in enumerate(_gltf_array(document, "bufferViews")):
        item = _gltf_object(raw_view, name=f"glTF bufferView[{index}]")
        add_parser_bytes(
            _strict_int(
                item.get("byteLength"),
                name=f"glTF bufferView[{index}].byteLength",
                minimum=1,
                maximum=MAX_MESH_SOURCE_BYTES,
            ),
            label="bufferView slices",
        )

    # The locked parser also materializes every accessor, even when no mesh
    # primitive references it: contiguous accessors retain a sliced bytes
    # object, strided accessors copy into a contiguous ndarray, and accessors
    # without a bufferView allocate a zero array.  Include their complete
    # declared footprint alongside the view slices above.
    max_accessor_count = max(MAX_MESH_VERTICES, 3 * MAX_MESH_TRIANGLES)
    for index, raw_accessor in enumerate(_gltf_array(document, "accessors")):
        item = _gltf_object(raw_accessor, name=f"glTF accessor[{index}]")
        component_type = _strict_int(
            item.get("componentType"),
            name=f"glTF accessor[{index}].componentType",
            minimum=min(_GLTF_COMPONENT_BYTES),
            maximum=max(_GLTF_COMPONENT_BYTES),
        )
        element_bytes = _gltf_accessor_element_bytes(
            component_type,
            str(item.get("type", "")),
        )
        count = _strict_int(
            item.get("count"),
            name=f"glTF accessor[{index}].count",
            minimum=1,
            maximum=max_accessor_count,
        )
        add_parser_bytes(
            count * element_bytes,
            label="accessor arrays",
        )

    add_parser_bytes(
        24 * vertex_count + 12 * triangle_count,
        label="canonical arrays",
    )
    admission = decoded_admission_from_counts(
        vertex_count=vertex_count,
        triangle_count=triangle_count,
        array_bytes=parser_bytes,
    )
    require_windows_runtime_capacity(admission.estimated_peak_bytes)
    return parser_bytes


def _gltf_array(document: Mapping[str, Any], name: str) -> list[Any]:
    value = document.get(name, [])
    if not isinstance(value, list):
        raise MeshAdmissionError(f"glTF {name} must be an array")
    if len(value) > MAX_MESH_VERTICES:
        raise MeshAdmissionError(f"glTF {name} exceeds the bounded entry budget")
    return value


def _gltf_object(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MeshAdmissionError(f"{name} must be an object")
    return value


def _gltf_index(value: object, *, name: str, length: int) -> int:
    if length < 1:
        raise MeshAdmissionError(f"{name} references an empty array")
    return _strict_int(value, name=name, minimum=0, maximum=length - 1)


def _gltf_accessor_element_bytes(component_type: int, accessor_type: str) -> int:
    component_bytes = _GLTF_COMPONENT_BYTES.get(component_type)
    shape = _GLTF_TYPE_SHAPES.get(accessor_type)
    if component_bytes is None or shape is None:
        raise MeshAdmissionError("glTF accessor componentType or type is unsupported")
    columns, rows = shape
    if columns == 1:
        return rows * component_bytes
    column_bytes = rows * component_bytes
    padded_column_bytes = ((column_bytes + 3) // 4) * 4
    return columns * padded_column_bytes


def _gltf_mesh_declaration(
    document: Mapping[str, Any],
) -> tuple[int, int, int]:
    extensions_used = document.get("extensionsUsed", [])
    if not isinstance(extensions_used, list) or not all(
        isinstance(item, str) for item in extensions_used
    ):
        raise MeshAdmissionError("glTF extensionsUsed must be an array of strings")
    if "EXT_mesh_gpu_instancing" in extensions_used:
        raise MeshAdmissionError(
            "glTF GPU instancing is not admitted by the v1 Windows profile"
        )

    buffers = _gltf_array(document, "buffers")
    buffer_lengths: list[int] = []
    for index, raw_buffer in enumerate(buffers):
        item = _gltf_object(raw_buffer, name=f"glTF buffer[{index}]")
        buffer_lengths.append(
            _strict_int(
                item.get("byteLength"),
                name=f"glTF buffer[{index}].byteLength",
                minimum=1,
                maximum=MAX_MESH_SOURCE_BYTES,
            )
        )

    buffer_views = _gltf_array(document, "bufferViews")
    view_state: list[tuple[int, int | None]] = []
    for index, raw_view in enumerate(buffer_views):
        item = _gltf_object(raw_view, name=f"glTF bufferView[{index}]")
        buffer_index = _gltf_index(
            item.get("buffer"),
            name=f"glTF bufferView[{index}].buffer",
            length=len(buffer_lengths),
        )
        byte_offset = _strict_int(
            item.get("byteOffset", 0),
            name=f"glTF bufferView[{index}].byteOffset",
            minimum=0,
            maximum=buffer_lengths[buffer_index],
        )
        byte_length = _strict_int(
            item.get("byteLength"),
            name=f"glTF bufferView[{index}].byteLength",
            minimum=1,
            maximum=buffer_lengths[buffer_index],
        )
        if byte_offset + byte_length > buffer_lengths[buffer_index]:
            raise MeshAdmissionError(
                f"glTF bufferView[{index}] exceeds its declared buffer"
            )
        stride_value = item.get("byteStride")
        stride = None
        if stride_value is not None:
            stride = _strict_int(
                stride_value,
                name=f"glTF bufferView[{index}].byteStride",
                minimum=4,
                maximum=252,
            )
        view_state.append((byte_length, stride))

    accessors = _gltf_array(document, "accessors")
    if not accessors:
        raise MeshAdmissionError("glTF source has no accessors")
    # Trimesh 4.11.5 creates its accessor table only inside the top-level
    # ``if \"bufferViews\" in header`` branch.  Although glTF permits an
    # accessor without a bufferView to be initialized with zeroes, omitting the
    # array key makes the locked parser reach an unbound ``access`` local.  Keep
    # the Windows parser profile deterministic by rejecting that shape before
    # parser execution; an explicit empty array still exercises zero-backed
    # accessor allocation safely.
    if "bufferViews" not in document:
        raise MeshAdmissionError(
            "glTF accessors require a top-level bufferViews array for the "
            "locked parser"
        )
    accessor_state: list[dict[str, int | str]] = []
    total_accessor_bytes = 0
    max_accessor_count = max(MAX_MESH_VERTICES, 3 * MAX_MESH_TRIANGLES)
    for index, raw_accessor in enumerate(accessors):
        item = _gltf_object(raw_accessor, name=f"glTF accessor[{index}]")
        component_type = _strict_int(
            item.get("componentType"),
            name=f"glTF accessor[{index}].componentType",
            minimum=min(_GLTF_COMPONENT_BYTES),
            maximum=max(_GLTF_COMPONENT_BYTES),
        )
        accessor_type = str(item.get("type", ""))
        element_bytes = _gltf_accessor_element_bytes(
            component_type,
            accessor_type,
        )
        count = _strict_int(
            item.get("count"),
            name=f"glTF accessor[{index}].count",
            minimum=1,
            maximum=max_accessor_count,
        )
        accessor_bytes = count * element_bytes
        total_accessor_bytes += accessor_bytes
        if total_accessor_bytes > MAX_MESH_DECODED_ARRAY_BYTES:
            raise MeshAdmissionError(
                "glTF declared accessor arrays exceed the fixed decoded-byte budget"
            )

        view_value = item.get("bufferView")
        if view_value is not None:
            view_index = _gltf_index(
                view_value,
                name=f"glTF accessor[{index}].bufferView",
                length=len(view_state),
            )
            view_length, view_stride = view_state[view_index]
            byte_offset = _strict_int(
                item.get("byteOffset", 0),
                name=f"glTF accessor[{index}].byteOffset",
                minimum=0,
                maximum=view_length,
            )
            stride = element_bytes if view_stride is None else view_stride
            if stride < element_bytes:
                raise MeshAdmissionError(
                    f"glTF accessor[{index}] byteStride is smaller than its element"
                )
            required = byte_offset + element_bytes + (count - 1) * stride
            if required > view_length:
                raise MeshAdmissionError(
                    f"glTF accessor[{index}] exceeds its declared bufferView"
                )
        elif "byteOffset" in item:
            raise MeshAdmissionError(
                f"glTF accessor[{index}] cannot define byteOffset without bufferView"
            )

        sparse_value = item.get("sparse")
        if sparse_value is not None:
            sparse = _gltf_object(
                sparse_value,
                name=f"glTF accessor[{index}].sparse",
            )
            sparse_count = _strict_int(
                sparse.get("count"),
                name=f"glTF accessor[{index}].sparse.count",
                minimum=1,
                maximum=count,
            )
            indices = _gltf_object(
                sparse.get("indices"),
                name=f"glTF accessor[{index}].sparse.indices",
            )
            sparse_index_type = _strict_int(
                indices.get("componentType"),
                name=f"glTF accessor[{index}].sparse.indices.componentType",
                minimum=5121,
                maximum=5125,
            )
            if sparse_index_type not in {5121, 5123, 5125}:
                raise MeshAdmissionError("glTF sparse indices must be unsigned integers")
            sparse_index_view = _gltf_index(
                indices.get("bufferView"),
                name=f"glTF accessor[{index}].sparse.indices.bufferView",
                length=len(view_state),
            )
            sparse_index_offset = _strict_int(
                indices.get("byteOffset", 0),
                name=f"glTF accessor[{index}].sparse.indices.byteOffset",
                minimum=0,
                maximum=view_state[sparse_index_view][0],
            )
            if (
                sparse_index_offset
                + sparse_count * _GLTF_COMPONENT_BYTES[sparse_index_type]
                > view_state[sparse_index_view][0]
            ):
                raise MeshAdmissionError("glTF sparse index data exceeds its bufferView")
            values = _gltf_object(
                sparse.get("values"),
                name=f"glTF accessor[{index}].sparse.values",
            )
            sparse_value_view = _gltf_index(
                values.get("bufferView"),
                name=f"glTF accessor[{index}].sparse.values.bufferView",
                length=len(view_state),
            )
            sparse_value_offset = _strict_int(
                values.get("byteOffset", 0),
                name=f"glTF accessor[{index}].sparse.values.byteOffset",
                minimum=0,
                maximum=view_state[sparse_value_view][0],
            )
            if (
                sparse_value_offset + sparse_count * element_bytes
                > view_state[sparse_value_view][0]
            ):
                raise MeshAdmissionError("glTF sparse value data exceeds its bufferView")

        accessor_state.append(
            {
                "component_type": component_type,
                "count": count,
                "element_bytes": element_bytes,
                "type": accessor_type,
            }
        )

    meshes = _gltf_array(document, "meshes")
    if not meshes:
        raise MeshAdmissionError("glTF source has no meshes")
    footprints: list[tuple[int, int]] = []
    for mesh_index, raw_mesh in enumerate(meshes):
        mesh = _gltf_object(raw_mesh, name=f"glTF mesh[{mesh_index}]")
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list) or not primitives:
            raise MeshAdmissionError(f"glTF mesh[{mesh_index}] has no primitives")
        mesh_vertices = 0
        mesh_triangles = 0
        for primitive_index, raw_primitive in enumerate(primitives):
            primitive = _gltf_object(
                raw_primitive,
                name=f"glTF mesh[{mesh_index}].primitive[{primitive_index}]",
            )
            attributes = _gltf_object(
                primitive.get("attributes"),
                name=(
                    f"glTF mesh[{mesh_index}].primitive[{primitive_index}].attributes"
                ),
            )
            position_index = _gltf_index(
                attributes.get("POSITION"),
                name=(
                    f"glTF mesh[{mesh_index}].primitive[{primitive_index}].POSITION"
                ),
                length=len(accessor_state),
            )
            position = accessor_state[position_index]
            if position["type"] != "VEC3":
                raise MeshAdmissionError("glTF POSITION accessor must use VEC3")
            position_count = int(position["count"])
            for attribute_name, accessor_value in attributes.items():
                attribute_index = _gltf_index(
                    accessor_value,
                    name=(
                        f"glTF mesh[{mesh_index}].primitive[{primitive_index}]"
                        f".attribute[{attribute_name}]"
                    ),
                    length=len(accessor_state),
                )
                if int(accessor_state[attribute_index]["count"]) != position_count:
                    raise MeshAdmissionError(
                        "glTF primitive attribute accessors must have equal counts"
                    )
            mode = _strict_int(
                primitive.get("mode", 4),
                name=f"glTF mesh[{mesh_index}].primitive[{primitive_index}].mode",
                minimum=0,
                maximum=6,
            )
            if mode not in {4, 5, 6}:
                continue
            index_value = primitive.get("indices")
            primitive_count = position_count
            if index_value is not None:
                index_accessor = accessor_state[
                    _gltf_index(
                        index_value,
                        name=(
                            f"glTF mesh[{mesh_index}].primitive[{primitive_index}]"
                            ".indices"
                        ),
                        length=len(accessor_state),
                    )
                ]
                if index_accessor["type"] != "SCALAR" or int(
                    index_accessor["component_type"]
                ) not in {5121, 5123, 5125}:
                    raise MeshAdmissionError(
                        "glTF triangle indices must use an unsigned SCALAR accessor"
                    )
                primitive_count = int(index_accessor["count"])
            if mode == 4:
                if primitive_count % 3 != 0:
                    raise MeshAdmissionError(
                        "glTF TRIANGLES index/vertex count must be divisible by 3"
                    )
                triangles = primitive_count // 3
            else:
                triangles = max(0, primitive_count - 2)
            if triangles < 1:
                raise MeshAdmissionError("glTF triangle primitive has no triangles")
            mesh_vertices += position_count
            mesh_triangles += triangles
            _raise_count_limit(
                name="declared glTF mesh vertices",
                observed=mesh_vertices,
                maximum=MAX_MESH_VERTICES,
            )
            _raise_count_limit(
                name="declared glTF mesh triangles",
                observed=mesh_triangles,
                maximum=MAX_MESH_TRIANGLES,
            )
        footprints.append((mesh_vertices, mesh_triangles))

    node_references = [0] * len(footprints)
    for node_index, raw_node in enumerate(_gltf_array(document, "nodes")):
        node = _gltf_object(raw_node, name=f"glTF node[{node_index}]")
        extensions = node.get("extensions")
        if isinstance(extensions, Mapping) and "EXT_mesh_gpu_instancing" in extensions:
            raise MeshAdmissionError(
                "glTF GPU instancing is not admitted by the v1 Windows profile"
            )
        mesh_value = node.get("mesh")
        if mesh_value is not None:
            node_references[
                _gltf_index(
                    mesh_value,
                    name=f"glTF node[{node_index}].mesh",
                    length=len(footprints),
                )
            ] += 1

    declared_vertices = 0
    declared_triangles = 0
    for mesh_index, (vertices, triangles) in enumerate(footprints):
        multiplicity = max(1, node_references[mesh_index])
        declared_vertices += vertices * multiplicity
        declared_triangles += triangles * multiplicity
        _raise_count_limit(
            name="declared glTF scene vertices",
            observed=declared_vertices,
            maximum=MAX_MESH_VERTICES,
        )
        _raise_count_limit(
            name="declared glTF scene triangles",
            observed=declared_triangles,
            maximum=MAX_MESH_TRIANGLES,
        )
    if declared_vertices < 3 or declared_triangles < 1:
        raise MeshAdmissionError("glTF source contains no authoritative triangle mesh")

    declared_payload_bytes = max(total_accessor_bytes, sum(buffer_lengths), 1)
    declared = decoded_admission_from_counts(
        vertex_count=declared_vertices,
        triangle_count=declared_triangles,
        array_bytes=declared_payload_bytes,
    )
    require_windows_runtime_capacity(declared.estimated_peak_bytes)
    return declared_vertices, declared_triangles, declared_triangles


def _bounded_binary_lines(source_stream: BinaryIO):
    while True:
        line = source_stream.readline(_MAX_MESH_TEXT_LINE_BYTES + 1)
        if not isinstance(line, (bytes, bytearray, memoryview)):
            raise MeshAdmissionError("mesh source stream must return bytes")
        line_bytes = bytes(line)
        if not line_bytes:
            return
        if len(line_bytes) > _MAX_MESH_TEXT_LINE_BYTES:
            raise MeshAdmissionError(
                "mesh source contains a text line above the 1 MiB safety limit"
            )
        yield line_bytes


def _obj_stream_declaration(
    source_stream: BinaryIO,
) -> tuple[int, int, int]:
    vertices = 0
    face_elements = 0
    triangles = 0
    for raw_line in _bounded_binary_lines(source_stream):
        content = raw_line.split(b"#", 1)[0].strip()
        if not content:
            continue
        tokens = content.split()
        if tokens[0] == b"v":
            vertices += 1
            _raise_count_limit(
                name="declared vertices",
                observed=vertices,
                maximum=MAX_MESH_VERTICES,
            )
        elif tokens[0] == b"f":
            vertex_references = len(tokens) - 1
            if vertex_references < 3:
                raise MeshAdmissionError(
                    "OBJ face must contain at least three vertex references"
                )
            face_elements += 1
            triangles += vertex_references - 2
            _raise_count_limit(
                name="declared triangulated faces",
                observed=triangles,
                maximum=MAX_MESH_TRIANGLES,
            )
    return vertices, face_elements, triangles


def _ascii_stl_stream_declaration(
    source_stream: BinaryIO,
) -> tuple[int, int, int]:
    facets = 0
    vertex_rows = 0
    for raw_line in _bounded_binary_lines(source_stream):
        tokens = raw_line.strip().lower().split()
        if not tokens:
            continue
        if tokens[0] == b"facet":
            facets += 1
            _raise_count_limit(
                name="declared triangulated faces",
                observed=facets,
                maximum=MAX_MESH_TRIANGLES,
            )
        elif tokens[0] == b"vertex":
            vertex_rows += 1
            _raise_count_limit(
                name="declared vertices",
                observed=vertex_rows,
                maximum=MAX_MESH_VERTICES,
            )
    conservative_vertices = max(vertex_rows, facets * 3)
    _raise_count_limit(
        name="declared vertices",
        observed=conservative_vertices,
        maximum=MAX_MESH_VERTICES,
    )
    return conservative_vertices, facets, facets


def _off_stream_triangle_count(
    source_stream: BinaryIO,
    *,
    vertex_count: int,
    face_count: int,
) -> int:
    logical_lines = (
        content
        for raw_line in _bounded_binary_lines(source_stream)
        if (content := raw_line.split(b"#", 1)[0].strip())
    )
    try:
        first = next(logical_lines).split()
    except StopIteration as exc:
        raise MeshAdmissionError("OFF source is empty") from exc
    if not first or first[0] != b"OFF":
        raise MeshAdmissionError("OFF source is missing its magic header")
    if len(first) < 3:
        try:
            next(logical_lines)
        except StopIteration as exc:
            raise MeshAdmissionError("OFF source is missing its count row") from exc

    for _ in range(vertex_count):
        try:
            next(logical_lines)
        except StopIteration as exc:
            raise MeshAdmissionError(
                "OFF source ended before all declared vertices"
            ) from exc

    triangles = 0
    for _ in range(face_count):
        try:
            tokens = next(logical_lines).split()
        except StopIteration as exc:
            raise MeshAdmissionError(
                "OFF source ended before all declared faces"
            ) from exc
        if not tokens:
            raise MeshAdmissionError("OFF face row is empty")
        try:
            polygon_vertices = int(tokens[0])
        except (TypeError, ValueError) as exc:
            raise MeshAdmissionError("OFF face size is not an integer") from exc
        if polygon_vertices < 3 or len(tokens) < polygon_vertices + 1:
            raise MeshAdmissionError("OFF face row is malformed")
        triangles += polygon_vertices - 2
        _raise_count_limit(
            name="declared triangulated faces",
            observed=triangles,
            maximum=MAX_MESH_TRIANGLES,
        )
    return triangles


def _ply_scalar_bytes(data_type: str) -> int:
    return struct.calcsize(_PLY_SCALAR_FORMATS[data_type])


def _ply_parser_footprint(
    *,
    layout: _PlyLayout,
    source_size_bytes: int,
    typed_payload_bytes: int,
    auxiliary_bytes: int,
    triangle_count: int,
    index_item_count: int,
) -> int:
    if auxiliary_bytes > MAX_MESH_PLY_AUXILIARY_BYTES:
        raise MeshAdmissionError(
            f"PLY auxiliary properties require {auxiliary_bytes:,} bytes, exceeding "
            f"the fixed limit of {MAX_MESH_PLY_AUXILIARY_BYTES:,} bytes"
        )
    output_vertices = layout.vertex.count
    list_property_count = 1
    if layout.texcoord is not None:
        # Trimesh may split positions at per-face UV seams.  The number of
        # resulting vertices is bounded by the number of referenced corners.
        output_vertices = max(output_vertices, index_item_count)
        list_property_count += 1
    _raise_count_limit(
        name="declared PLY parser vertices",
        observed=output_vertices,
        maximum=MAX_MESH_VERTICES,
    )
    canonical_array_bytes = 24 * output_vertices + 12 * triangle_count
    list_row_overhead = 64 * layout.face.count * list_property_count
    parser_bytes = (
        source_size_bytes
        + typed_payload_bytes
        + canonical_array_bytes
        + list_row_overhead
    )
    admission = decoded_admission_from_counts(
        vertex_count=output_vertices,
        triangle_count=triangle_count,
        array_bytes=parser_bytes,
    )
    require_windows_runtime_capacity(admission.estimated_peak_bytes)
    return parser_bytes


def _ply_ascii_triangle_count(
    source_stream: BinaryIO,
    *,
    header: _PlyHeader,
    source_size_bytes: int,
) -> tuple[int, int]:
    layout = _ply_supported_layout(header)
    source_stream.seek(header.header_bytes)
    lines = iter(_bounded_binary_lines(source_stream))
    triangles = 0
    index_item_count = 0
    typed_payload_bytes = 0
    auxiliary_bytes = 0
    for element in header.elements:
        for _ in range(element.count):
            try:
                tokens = next(lines).split()
            except StopIteration as exc:
                raise MeshAdmissionError(
                    f"PLY source ended inside element {element.name!r}"
                ) from exc
            cursor = 0
            list_counts: dict[str, int] = {}
            for prop in element.properties:
                if prop.scalar_type is not None:
                    if cursor >= len(tokens):
                        raise MeshAdmissionError("PLY ASCII scalar row is truncated")
                    cursor += 1
                    scalar_bytes = _ply_scalar_bytes(prop.scalar_type)
                    typed_payload_bytes += scalar_bytes
                    if element.name == "face" or prop.name not in {"x", "y", "z"}:
                        auxiliary_bytes += scalar_bytes
                else:
                    if cursor >= len(tokens):
                        raise MeshAdmissionError("PLY ASCII row is truncated")
                    try:
                        list_count = int(tokens[cursor])
                    except (TypeError, ValueError) as exc:
                        raise MeshAdmissionError(
                            "PLY ASCII list count is not an integer"
                        ) from exc
                    if list_count < 0:
                        raise MeshAdmissionError("PLY list count cannot be negative")
                    maximum_count = (
                        2 * MAX_MESH_VERTICES
                        if prop is layout.texcoord
                        else MAX_MESH_VERTICES
                    )
                    if list_count > maximum_count:
                        raise MeshAdmissionError("PLY list exceeds the bounded item budget")
                    cursor += 1
                    if cursor + list_count > len(tokens):
                        raise MeshAdmissionError("PLY ASCII list row is truncated")
                    assert prop.list_count_type is not None
                    assert prop.list_item_type is not None
                    count_bytes = _ply_scalar_bytes(prop.list_count_type)
                    item_bytes = list_count * _ply_scalar_bytes(prop.list_item_type)
                    typed_payload_bytes += count_bytes + item_bytes
                    if prop is layout.texcoord:
                        auxiliary_bytes += count_bytes + item_bytes
                    list_counts[prop.name] = list_count
                    cursor += list_count
            if cursor != len(tokens):
                raise MeshAdmissionError("PLY ASCII row has undeclared values")
            if element is layout.face:
                polygon_vertices = list_counts.get(layout.vertex_indices.name)
                if polygon_vertices is None or polygon_vertices < 3:
                    raise MeshAdmissionError("PLY face must contain at least 3 vertices")
                if layout.texcoord is not None and list_counts.get("texcoord") != (
                    2 * polygon_vertices
                ):
                    raise MeshAdmissionError(
                        "PLY face texcoord count must be twice its vertex-index count"
                    )
                index_item_count += polygon_vertices
                triangles += polygon_vertices - 2
                _raise_count_limit(
                    name="declared triangulated faces",
                    observed=triangles,
                    maximum=MAX_MESH_TRIANGLES,
                )
    if any(raw_line.strip() for raw_line in lines):
        raise MeshAdmissionError("PLY source contains rows after declared elements")
    parser_bytes = _ply_parser_footprint(
        layout=layout,
        source_size_bytes=source_size_bytes,
        typed_payload_bytes=typed_payload_bytes,
        auxiliary_bytes=auxiliary_bytes,
        triangle_count=triangles,
        index_item_count=index_item_count,
    )
    return triangles, parser_bytes


def _ply_binary_triangle_count(
    source_stream: BinaryIO,
    *,
    header: _PlyHeader,
    source_size_bytes: int,
) -> tuple[int, int]:
    layout = _ply_supported_layout(header)
    endian = "<" if header.encoding == "binary_little_endian" else ">"
    source_stream.seek(header.header_bytes)
    offset = header.header_bytes
    triangles = 0
    index_item_count = 0
    typed_payload_bytes = 0
    auxiliary_bytes = 0
    fixed_list_counts: dict[tuple[str, str], int] = {}
    for element in header.elements:
        if all(prop.scalar_type is not None for prop in element.properties):
            row_size = sum(
                _ply_scalar_bytes(prop.scalar_type)
                for prop in element.properties
                if prop.scalar_type is not None
            )
            skip_element_bytes = row_size * element.count
            typed_payload_bytes += skip_element_bytes
            if element is layout.vertex:
                primary_row_bytes = sum(
                    _ply_scalar_bytes(prop.scalar_type)
                    for prop in element.properties
                    if prop.scalar_type is not None and prop.name in {"x", "y", "z"}
                )
                auxiliary_bytes += (row_size - primary_row_bytes) * element.count
            offset += skip_element_bytes
            if offset > source_size_bytes:
                raise MeshAdmissionError(
                    f"PLY binary element {element.name!r} is truncated"
                )
            source_stream.seek(skip_element_bytes, 1)
            continue
        for _ in range(element.count):
            list_counts: dict[str, int] = {}
            for prop in element.properties:
                if prop.scalar_type is not None:
                    scalar_size = _ply_scalar_bytes(prop.scalar_type)
                    typed_payload_bytes += scalar_size
                    if element is layout.face:
                        auxiliary_bytes += scalar_size
                    offset += scalar_size
                    if offset > source_size_bytes:
                        raise MeshAdmissionError("PLY binary scalar row is truncated")
                    source_stream.seek(scalar_size, 1)
                    continue

                assert prop.list_count_type is not None
                assert prop.list_item_type is not None
                count_format = _PLY_SCALAR_FORMATS[prop.list_count_type]
                if count_format in {"f", "d"}:
                    raise MeshAdmissionError("PLY list count type must be an integer")
                count_size = struct.calcsize(count_format)
                count_bytes = source_stream.read(count_size)
                if not isinstance(count_bytes, (bytes, bytearray, memoryview)):
                    raise MeshAdmissionError("mesh source stream must return bytes")
                if len(count_bytes) != count_size:
                    raise MeshAdmissionError("PLY binary list count is truncated")
                offset += count_size
                list_count = int(struct.unpack(endian + count_format, count_bytes)[0])
                if list_count < 0:
                    raise MeshAdmissionError("PLY list count cannot be negative")
                maximum_count = (
                    2 * MAX_MESH_VERTICES
                    if prop is layout.texcoord
                    else MAX_MESH_VERTICES
                )
                if list_count > maximum_count:
                    raise MeshAdmissionError("PLY list exceeds the bounded item budget")
                list_key = (element.name, prop.name)
                expected_count = fixed_list_counts.setdefault(list_key, list_count)
                if list_count != expected_count:
                    raise MeshAdmissionError(
                        "binary PLY list lengths must remain constant for the "
                        "locked parser"
                    )
                item_size = _ply_scalar_bytes(prop.list_item_type)
                skip_bytes = list_count * item_size
                typed_payload_bytes += count_size + skip_bytes
                if prop is layout.texcoord:
                    auxiliary_bytes += count_size + skip_bytes
                offset += skip_bytes
                if offset > source_size_bytes:
                    raise MeshAdmissionError("PLY binary list row is truncated")
                source_stream.seek(skip_bytes, 1)
                list_counts[prop.name] = list_count
            if element is layout.face:
                polygon_vertices = list_counts.get(layout.vertex_indices.name)
                if polygon_vertices is None or polygon_vertices < 3:
                    raise MeshAdmissionError("PLY face must contain at least 3 vertices")
                if layout.texcoord is not None and list_counts.get("texcoord") != (
                    2 * polygon_vertices
                ):
                    raise MeshAdmissionError(
                        "PLY face texcoord count must be twice its vertex-index count"
                    )
                index_item_count += polygon_vertices
                triangles += polygon_vertices - 2
                _raise_count_limit(
                    name="declared triangulated faces",
                    observed=triangles,
                    maximum=MAX_MESH_TRIANGLES,
                )
    if offset != source_size_bytes:
        raise MeshAdmissionError("PLY binary payload length differs from its declarations")
    parser_bytes = _ply_parser_footprint(
        layout=layout,
        source_size_bytes=source_size_bytes,
        typed_payload_bytes=typed_payload_bytes,
        auxiliary_bytes=auxiliary_bytes,
        triangle_count=triangles,
        index_item_count=index_item_count,
    )
    return triangles, parser_bytes


def _ply_stream_triangle_count(
    source_stream: BinaryIO,
    *,
    prefix: bytes,
    source_size_bytes: int,
) -> tuple[int, int]:
    header = _parse_ply_header(prefix)
    if header.encoding == "ascii":
        return _ply_ascii_triangle_count(
            source_stream,
            header=header,
            source_size_bytes=source_size_bytes,
        )
    return _ply_binary_triangle_count(
        source_stream,
        header=header,
        source_size_bytes=source_size_bytes,
    )


def preflight_mesh_source(
    source_stream: BinaryIO,
    *,
    source_format: str,
    source_size_bytes: int,
) -> MeshSourcePreflight:
    """Inspect bounded declarations on the exact descriptor used by the parser."""

    normalized_format = _source_format(source_format)
    size = _strict_int(
        source_size_bytes,
        name="source_size_bytes",
        minimum=1,
        maximum=MAX_MESH_SOURCE_BYTES,
    )
    try:
        source_stream.seek(0)
        prefix_limit = (
            MAX_MESH_GLTF_JSON_BYTES + 1
            if normalized_format == "gltf"
            else MAX_MESH_HEADER_BYTES + 1
        )
        prefix = source_stream.read(min(size, prefix_limit))
        if not isinstance(prefix, (bytes, bytearray, memoryview)):
            raise MeshAdmissionError("mesh source stream must return bytes")
        prefix_bytes = bytes(prefix)
    except (AttributeError, OSError) as exc:
        raise MeshAdmissionError("mesh source stream must be seekable") from exc
    finally:
        try:
            source_stream.seek(0)
        except (AttributeError, OSError):
            pass

    declaration: tuple[int, int] | None = None
    declaration_kind = "not_available"
    declared_triangles: int | None = None
    declared_parser_bytes: int | None = None
    external_buffer_byte_lengths: tuple[tuple[str, int], ...] = ()
    try:
        if normalized_format == "ply":
            ply_header = _parse_ply_header(prefix_bytes)
            declaration = _ply_declaration(ply_header)
            if ply_header.encoding == "ascii":
                declaration_kind = "ply_ascii_header"
                _require_text_source_budget(size, label="PLY ASCII source")
            else:
                declaration_kind = "ply_binary_header"
            _raise_count_limit(
                name="declared vertices",
                observed=declaration[0],
                maximum=MAX_MESH_VERTICES,
            )
            _raise_count_limit(
                name="declared face elements",
                observed=declaration[1],
                maximum=MAX_MESH_TRIANGLES,
            )
            declared_triangles, declared_parser_bytes = _ply_stream_triangle_count(
                source_stream,
                prefix=prefix_bytes,
                source_size_bytes=size,
            )
        elif normalized_format == "off":
            _require_text_source_budget(size, label="OFF text source")
            declaration = _off_declaration(prefix_bytes)
            if declaration is None:
                raise MeshAdmissionError("OFF source has no bounded count header")
            declaration_kind = "off_header"
            _raise_count_limit(
                name="declared vertices",
                observed=declaration[0],
                maximum=MAX_MESH_VERTICES,
            )
            _raise_count_limit(
                name="declared face elements",
                observed=declaration[1],
                maximum=MAX_MESH_TRIANGLES,
            )
            source_stream.seek(0)
            declared_triangles = _off_stream_triangle_count(
                source_stream,
                vertex_count=declaration[0],
                face_count=declaration[1],
            )
        elif normalized_format == "stl":
            declaration = _stl_declaration(prefix_bytes, size)
            if declaration is not None:
                declaration_kind = "binary_stl_header"
                _raise_count_limit(
                    name="declared vertices",
                    observed=declaration[0],
                    maximum=MAX_MESH_VERTICES,
                )
                _raise_count_limit(
                    name="declared face elements",
                    observed=declaration[1],
                    maximum=MAX_MESH_TRIANGLES,
                )
                declared_triangles = declaration[1]
            else:
                declaration_kind = "ascii_stl_stream"
                _require_text_source_budget(size, label="STL ASCII source")
                source_stream.seek(0)
                vertices, faces, declared_triangles = (
                    _ascii_stl_stream_declaration(source_stream)
                )
                declaration = (vertices, faces)
        elif normalized_format == "obj":
            declaration_kind = "obj_stream"
            _require_text_source_budget(size, label="OBJ text source")
            source_stream.seek(0)
            vertices, faces, declared_triangles = _obj_stream_declaration(
                source_stream
            )
            declaration = (vertices, faces)
        elif normalized_format in {"gltf", "glb"}:
            document, declaration_kind = _gltf_document_from_stream(
                source_stream,
                source_format=normalized_format,
                source_size_bytes=size,
            )
            vertices, faces, declared_triangles = _gltf_mesh_declaration(document)
            declaration = (vertices, faces)
            external_buffer_byte_lengths = _gltf_external_buffer_bindings(
                document,
                source_format=normalized_format,
            )
            declared_parser_bytes = _gltf_parser_footprint(
                document,
                source_size_bytes=size,
                vertex_count=vertices,
                triangle_count=declared_triangles,
            )
    finally:
        try:
            source_stream.seek(0)
        except (AttributeError, OSError):
            pass

    declared_vertices: int | None = None
    declared_faces: int | None = None
    if declaration is not None:
        declared_vertices, declared_faces = declaration
        _raise_count_limit(
            name="declared vertices",
            observed=declared_vertices,
            maximum=MAX_MESH_VERTICES,
        )
        _raise_count_limit(
            name="declared face elements",
            observed=declared_faces,
            maximum=MAX_MESH_TRIANGLES,
        )
    if declared_triangles is not None:
        _raise_count_limit(
            name="declared triangulated faces",
            observed=declared_triangles,
            maximum=MAX_MESH_TRIANGLES,
        )

    return MeshSourcePreflight(
        source_format=normalized_format,
        source_size_bytes=size,
        declaration_kind=declaration_kind,
        declared_vertex_count=declared_vertices,
        declared_face_element_count=declared_faces,
        declared_triangle_count=declared_triangles,
        declared_parser_bytes=declared_parser_bytes,
        external_buffer_byte_lengths=external_buffer_byte_lengths,
    )


def estimated_texture_bytes(value: object) -> int:
    """Return a conservative decoded byte estimate without forcing image decode."""

    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    size = getattr(value, "size", None)
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        return 0
    try:
        width = int(size[0])
        height = int(size[1])
    except (TypeError, ValueError, OverflowError):
        return 0
    if width < 0 or height < 0:
        raise MeshAdmissionError("decoded texture dimensions must be non-negative")
    try:
        bands = tuple(getattr(value, "getbands")())
    except Exception:
        bands = ()
    band_count = max(1, len(bands))
    mode = str(getattr(value, "mode", "") or "").upper()
    if mode.startswith("I;16"):
        bytes_per_band = 2
    elif mode in {"I", "F"}:
        bytes_per_band = 4
    else:
        bytes_per_band = 1
    return width * height * band_count * bytes_per_band


def require_texture_budget(value: object) -> int:
    estimate = estimated_texture_bytes(value)
    if estimate > MAX_MESH_TEXTURE_BYTES:
        raise MeshAdmissionError(
            f"mesh admission rejected decoded texture: {estimate:,} bytes exceeds "
            f"the Windows workflow limit of {MAX_MESH_TEXTURE_BYTES:,} bytes"
        )
    return estimate


def decoded_admission_from_counts(
    *,
    vertex_count: int,
    triangle_count: int,
    array_bytes: int,
) -> DecodedMeshAdmission:
    """Apply the aggregate Windows Open envelope to already inspected arrays."""

    vertices = _strict_int(
        vertex_count,
        name="decoded.vertex_count",
        minimum=3,
        maximum=MAX_MESH_VERTICES,
    )
    triangles = _strict_int(
        triangle_count,
        name="decoded.triangle_count",
        minimum=1,
        maximum=MAX_MESH_TRIANGLES,
    )
    payload_bytes = _strict_int(
        array_bytes,
        name="decoded.array_bytes",
        minimum=1,
        maximum=MAX_MESH_DECODED_ARRAY_BYTES,
    )
    estimated_peak_bytes = 3 * payload_bytes + 32 * vertices + 48 * triangles
    if estimated_peak_bytes > MAX_MESH_ESTIMATED_PEAK_BYTES:
        raise MeshAdmissionError(
            f"mesh admission rejected estimated Open peak: "
            f"{estimated_peak_bytes:,} bytes exceeds the Windows workflow "
            f"limit of {MAX_MESH_ESTIMATED_PEAK_BYTES:,} bytes"
        )
    return DecodedMeshAdmission(
        vertex_count=vertices,
        triangle_count=triangles,
        array_bytes=payload_bytes,
        estimated_peak_bytes=estimated_peak_bytes,
    )


def _windows_available_memory_bytes() -> tuple[int, int]:
    import ctypes  # noqa: PLC0415

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        raise OSError("ctypes.WinDLL is unavailable")
    kernel32 = win_dll("kernel32", use_last_error=True)
    function = kernel32.GlobalMemoryStatusEx
    function.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
    function.restype = ctypes.c_int
    if not function(ctypes.byref(status)):
        get_last_error = getattr(ctypes, "get_last_error", lambda: 0)
        error = int(get_last_error())
        raise OSError(error, "GlobalMemoryStatusEx failed")
    return int(status.ullAvailPhys), int(status.ullAvailPageFile)


def require_windows_runtime_capacity(estimated_peak_bytes: int) -> None:
    """Apply a non-durable capability gate on the current Windows machine."""

    peak = _strict_int(
        estimated_peak_bytes,
        name="estimated_peak_bytes",
        minimum=1,
        maximum=MAX_MESH_ESTIMATED_PEAK_BYTES,
    )
    if os.name != "nt":
        return
    try:
        available_physical, available_page_file = _windows_available_memory_bytes()
    except OSError as exc:
        raise MeshAdmissionError(
            "Windows runtime memory capacity could not be verified"
        ) from exc
    required = peak + WINDOWS_RUNTIME_MEMORY_RESERVE_BYTES
    if available_physical < required or available_page_file < required:
        raise MeshAdmissionError(
            "mesh satisfies the durable admission profile but this Windows PC "
            f"does not have the required free memory: need {required:,} bytes "
            f"including reserve, physical={available_physical:,}, "
            f"commit={available_page_file:,}"
        )


def inspect_decoded_mesh(
    vertices: object,
    faces: object,
    *,
    optional_arrays: Sequence[object] = (),
) -> DecodedMeshAdmission:
    """Validate decoded arrays without making whole-mesh conversion copies."""

    vertex_array = np.asarray(vertices)
    face_array = np.asarray(faces)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise MeshAdmissionError("decoded vertices must have shape (N, 3)")
    if not np.issubdtype(vertex_array.dtype, np.number) or np.issubdtype(
        vertex_array.dtype, np.complexfloating
    ):
        raise MeshAdmissionError("decoded vertices must use a real numeric dtype")
    vertex_count = int(vertex_array.shape[0])
    _raise_count_limit(
        name="decoded vertices",
        observed=vertex_count,
        maximum=MAX_MESH_VERTICES,
    )
    if vertex_count < 3:
        raise MeshAdmissionError("decoded mesh must contain at least three vertices")

    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise MeshAdmissionError("decoded faces must contain only triangles")
    if not np.issubdtype(face_array.dtype, np.integer):
        raise MeshAdmissionError("decoded triangle indices must be integers")
    triangle_count = int(face_array.shape[0])
    _raise_count_limit(
        name="decoded triangles",
        observed=triangle_count,
        maximum=MAX_MESH_TRIANGLES,
    )
    if triangle_count < 1:
        raise MeshAdmissionError("decoded mesh must contain at least one triangle")

    for start in range(0, vertex_count, _ARRAY_CHECK_CHUNK):
        chunk = vertex_array[start : start + _ARRAY_CHECK_CHUNK]
        try:
            finite = bool(np.isfinite(chunk).all())
        except TypeError as exc:
            raise MeshAdmissionError("decoded vertices cannot be tested for finiteness") from exc
        if not finite:
            raise MeshAdmissionError("decoded mesh contains non-finite vertex coordinates")

    for start in range(0, triangle_count, _ARRAY_CHECK_CHUNK):
        chunk = face_array[start : start + _ARRAY_CHECK_CHUNK]
        if int(np.min(chunk)) < 0 or int(np.max(chunk)) >= vertex_count:
            raise MeshAdmissionError("decoded mesh contains an invalid triangle index")
        if int(np.max(chunk)) > np.iinfo(np.int32).max:
            raise MeshAdmissionError("decoded triangle index exceeds the int32 range")

    # Admission covers both parser-owned arrays and the canonical MeshData
    # conversion that follows (float64 positions, int32 triangle indices).
    # A compact float32 parser result must not understate the Windows Open peak.
    array_bytes = max(
        int(vertex_array.nbytes),
        vertex_count * 3 * np.dtype(np.float64).itemsize,
    ) + max(
        int(face_array.nbytes),
        triangle_count * 3 * np.dtype(np.int32).itemsize,
    )
    for value in optional_arrays:
        if value is None:
            continue
        try:
            optional = np.asarray(value)
        except Exception as exc:
            raise MeshAdmissionError("decoded optional mesh array is unreadable") from exc
        current_bytes = int(optional.nbytes)
        if optional.ndim == 2 and optional.shape[1:] == (2,) and np.issubdtype(
            optional.dtype,
            np.number,
        ):
            current_bytes = max(
                current_bytes,
                int(optional.shape[0]) * 2 * np.dtype(np.float64).itemsize,
            )
        array_bytes += current_bytes
        if array_bytes > MAX_MESH_DECODED_ARRAY_BYTES:
            break
    if array_bytes > MAX_MESH_DECODED_ARRAY_BYTES:
        raise MeshAdmissionError(
            f"mesh admission rejected decoded arrays: {array_bytes:,} bytes exceeds "
            f"the Windows workflow limit of {MAX_MESH_DECODED_ARRAY_BYTES:,} bytes"
        )

    # Loader arrays, immutable source snapshot, canonical projection, CPU/GPU
    # normals/centroids/VBO can coexist briefly during native Open.
    return decoded_admission_from_counts(
        vertex_count=vertex_count,
        triangle_count=triangle_count,
        array_bytes=array_bytes,
    )


def admitted_geometry_sha256(vertices: object, faces: object) -> str:
    """Bind an admission receipt to exact accepted positions and triangles."""

    vertex_array = np.asarray(vertices)
    face_array = np.asarray(faces)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise MeshAdmissionError("accepted vertices must have shape (N, 3)")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise MeshAdmissionError("accepted faces must have shape (M, 3)")
    if not np.issubdtype(vertex_array.dtype, np.number) or np.issubdtype(
        vertex_array.dtype, np.complexfloating
    ):
        raise MeshAdmissionError("accepted vertices must use a real numeric dtype")
    if not np.issubdtype(face_array.dtype, np.integer):
        raise MeshAdmissionError("accepted faces must contain integer indices")
    digest = hashlib.sha256()
    digest.update(_GEOMETRY_HASH_DOMAIN)
    digest.update(_GEOMETRY_HASH_SCOPE.encode("ascii"))
    digest.update(b"\x00")
    digest.update(struct.pack("<Q", int(vertex_array.shape[0])))
    for start in range(0, int(vertex_array.shape[0]), _ARRAY_CHECK_CHUNK):
        chunk = np.array(
            vertex_array[start : start + _ARRAY_CHECK_CHUNK],
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if not np.isfinite(chunk).all():
            raise MeshAdmissionError("accepted mesh contains non-finite vertices")
        chunk[chunk == 0.0] = 0.0
        digest.update(np.ascontiguousarray(chunk.astype("<f8", copy=False)).tobytes())
    digest.update(struct.pack("<Q", int(face_array.shape[0])))
    for start in range(0, int(face_array.shape[0]), _ARRAY_CHECK_CHUNK):
        chunk = np.asarray(
            face_array[start : start + _ARRAY_CHECK_CHUNK],
            dtype=np.int64,
        )
        if chunk.size and (
            int(np.min(chunk)) < 0
            or int(np.max(chunk)) >= int(vertex_array.shape[0])
            or int(np.max(chunk)) > np.iinfo(np.int32).max
        ):
            raise MeshAdmissionError("accepted mesh contains invalid triangle indices")
        digest.update(np.ascontiguousarray(chunk.astype("<i4", copy=False)).tobytes())
    return digest.hexdigest()


def build_mesh_admission_receipt(
    preflight: MeshSourcePreflight,
    decoded: DecodedMeshAdmission,
    *,
    accepted_vertex_count: int,
    accepted_triangle_count: int,
    accepted_geometry_sha256: str,
) -> dict[str, Any]:
    """Create the exact path-free QC receipt persisted with new geometry."""

    if not isinstance(preflight, MeshSourcePreflight):
        raise MeshAdmissionError("preflight must be a MeshSourcePreflight")
    if not isinstance(decoded, DecodedMeshAdmission):
        raise MeshAdmissionError("decoded must be a DecodedMeshAdmission")
    accepted_vertices = _strict_int(
        accepted_vertex_count,
        name="accepted.vertex_count",
        minimum=3,
        maximum=MAX_MESH_VERTICES,
    )
    accepted_triangles = _strict_int(
        accepted_triangle_count,
        name="accepted.triangle_count",
        minimum=1,
        maximum=MAX_MESH_TRIANGLES,
    )
    if accepted_vertices != decoded.vertex_count:
        raise MeshAdmissionError("mesh sanitizer must preserve decoded vertex order")
    if accepted_triangles > decoded.triangle_count:
        raise MeshAdmissionError("mesh sanitizer cannot add triangles")
    geometry_sha256 = _sha256(
        accepted_geometry_sha256,
        name="accepted.geometry_sha256",
    )

    receipt = {
        "accepted": {
            "geometry_sha256": geometry_sha256,
            "removed_degenerate_triangle_count": (
                decoded.triangle_count - accepted_triangles
            ),
            "triangle_count": accepted_triangles,
            "vertex_count": accepted_vertices,
        },
        "checks": {
            "finite_vertices": True,
            "valid_triangle_indices": True,
        },
        "declaration": {
            "face_element_count": preflight.declared_face_element_count,
            "kind": preflight.declaration_kind,
            "parser_bytes": preflight.declared_parser_bytes,
            "triangle_count": preflight.declared_triangle_count,
            "vertex_count": preflight.declared_vertex_count,
        },
        "decoded": {
            "array_bytes": decoded.array_bytes,
            "estimated_peak_bytes": decoded.estimated_peak_bytes,
            "triangle_count": decoded.triangle_count,
            "vertex_count": decoded.vertex_count,
        },
        "limits": {
            "max_decoded_array_bytes": MAX_MESH_DECODED_ARRAY_BYTES,
            "max_dependency_bytes": MAX_MESH_DEPENDENCY_BYTES,
            "max_dependency_total_bytes": MAX_MESH_DEPENDENCY_TOTAL_BYTES,
            "max_estimated_peak_bytes": MAX_MESH_ESTIMATED_PEAK_BYTES,
            "max_gltf_json_bytes": MAX_MESH_GLTF_JSON_BYTES,
            "max_header_bytes": MAX_MESH_HEADER_BYTES,
            "max_ply_auxiliary_bytes": MAX_MESH_PLY_AUXILIARY_BYTES,
            "max_ply_face_properties": MAX_MESH_PLY_FACE_PROPERTIES,
            "max_ply_vertex_properties": MAX_MESH_PLY_VERTEX_PROPERTIES,
            "max_source_bytes": MAX_MESH_SOURCE_BYTES,
            "max_text_source_bytes": MAX_MESH_TEXT_SOURCE_BYTES,
            "max_text_line_bytes": _MAX_MESH_TEXT_LINE_BYTES,
            "max_texture_bytes": MAX_MESH_TEXTURE_BYTES,
            "max_triangles": MAX_MESH_TRIANGLES,
            "max_vertices": MAX_MESH_VERTICES,
        },
        "profile": MESH_ADMISSION_PROFILE,
        "schema_version": MESH_ADMISSION_SCHEMA_VERSION,
        "source_format": preflight.source_format,
        "source_size_bytes": preflight.source_size_bytes,
        "status": MESH_ADMISSION_STATUS,
    }
    return validate_mesh_admission_receipt(receipt)


def mesh_admission_receipt_for_arrays(
    vertices: object,
    faces: object,
    *,
    source_format: str,
    source_size_bytes: int,
    optional_arrays: Sequence[object] = (),
) -> dict[str, Any]:
    """Build a no-declaration receipt for an already decoded source snapshot."""

    preflight = MeshSourcePreflight(
        source_format=source_format,
        source_size_bytes=source_size_bytes,
    )
    decoded = inspect_decoded_mesh(
        vertices,
        faces,
        optional_arrays=optional_arrays,
    )
    return build_mesh_admission_receipt(
        preflight,
        decoded,
        accepted_vertex_count=decoded.vertex_count,
        accepted_triangle_count=decoded.triangle_count,
        accepted_geometry_sha256=admitted_geometry_sha256(vertices, faces),
    )


def validate_mesh_admission_receipt(value: object) -> dict[str, Any]:
    """Validate and canonicalize one persisted mesh-admission receipt."""

    root = _exact_mapping(
        value,
        {
            "accepted",
            "checks",
            "declaration",
            "decoded",
            "limits",
            "profile",
            "schema_version",
            "source_format",
            "source_size_bytes",
            "status",
        },
        name="mesh admission receipt",
    )
    if root["profile"] != MESH_ADMISSION_PROFILE:
        raise MeshAdmissionError("mesh admission profile is unsupported")
    if root["schema_version"] != MESH_ADMISSION_SCHEMA_VERSION:
        raise MeshAdmissionError("mesh admission schema version is unsupported")
    if root["status"] != MESH_ADMISSION_STATUS:
        raise MeshAdmissionError("mesh admission status must be accepted")
    source_format = _source_format(root["source_format"])
    source_size = _strict_int(
        root["source_size_bytes"],
        name="source_size_bytes",
        minimum=1,
        maximum=MAX_MESH_SOURCE_BYTES,
    )

    limits = _exact_mapping(
        root["limits"],
        {
            "max_decoded_array_bytes",
            "max_dependency_bytes",
            "max_dependency_total_bytes",
            "max_estimated_peak_bytes",
            "max_gltf_json_bytes",
            "max_header_bytes",
            "max_ply_auxiliary_bytes",
            "max_ply_face_properties",
            "max_ply_vertex_properties",
            "max_source_bytes",
            "max_text_source_bytes",
            "max_text_line_bytes",
            "max_texture_bytes",
            "max_triangles",
            "max_vertices",
        },
        name="mesh admission limits",
    )
    expected_limits = {
        "max_decoded_array_bytes": MAX_MESH_DECODED_ARRAY_BYTES,
        "max_dependency_bytes": MAX_MESH_DEPENDENCY_BYTES,
        "max_dependency_total_bytes": MAX_MESH_DEPENDENCY_TOTAL_BYTES,
        "max_estimated_peak_bytes": MAX_MESH_ESTIMATED_PEAK_BYTES,
        "max_gltf_json_bytes": MAX_MESH_GLTF_JSON_BYTES,
        "max_header_bytes": MAX_MESH_HEADER_BYTES,
        "max_ply_auxiliary_bytes": MAX_MESH_PLY_AUXILIARY_BYTES,
        "max_ply_face_properties": MAX_MESH_PLY_FACE_PROPERTIES,
        "max_ply_vertex_properties": MAX_MESH_PLY_VERTEX_PROPERTIES,
        "max_source_bytes": MAX_MESH_SOURCE_BYTES,
        "max_text_source_bytes": MAX_MESH_TEXT_SOURCE_BYTES,
        "max_text_line_bytes": _MAX_MESH_TEXT_LINE_BYTES,
        "max_texture_bytes": MAX_MESH_TEXTURE_BYTES,
        "max_triangles": MAX_MESH_TRIANGLES,
        "max_vertices": MAX_MESH_VERTICES,
    }
    if dict(limits) != expected_limits:
        raise MeshAdmissionError("mesh admission limits do not match the v1 profile")

    declaration = _exact_mapping(
        root["declaration"],
        {
            "face_element_count",
            "kind",
            "parser_bytes",
            "triangle_count",
            "vertex_count",
        },
        name="mesh admission declaration",
    )
    declaration_kind = str(declaration["kind"] or "")
    if declaration_kind not in _DECLARATION_KINDS:
        raise MeshAdmissionError("mesh admission declaration kind is unsupported")
    declared_vertices = _optional_count(
        declaration["vertex_count"],
        name="declaration.vertex_count",
        maximum=MAX_MESH_VERTICES,
    )
    declared_faces = _optional_count(
        declaration["face_element_count"],
        name="declaration.face_element_count",
        maximum=MAX_MESH_TRIANGLES,
    )
    declared_triangles = _optional_count(
        declaration["triangle_count"],
        name="declaration.triangle_count",
        maximum=MAX_MESH_TRIANGLES,
    )
    declared_parser_bytes = _optional_count(
        declaration["parser_bytes"],
        name="declaration.parser_bytes",
        maximum=MAX_MESH_DECODED_ARRAY_BYTES,
    )
    if declaration_kind == "not_available" and (
        declared_vertices is not None
        or declared_faces is not None
        or declared_triangles is not None
        or declared_parser_bytes is not None
    ):
        raise MeshAdmissionError(
            "not_available mesh declaration cannot contain counts"
        )
    if declaration_kind != "not_available" and (
        declared_vertices is None
        or declared_faces is None
        or declared_triangles is None
    ):
        raise MeshAdmissionError(
            "declared mesh source must contain vertex, face, and triangle counts"
        )
    if declaration_kind in {
        "glb_json_chunk",
        "gltf_json",
        "ply_ascii_header",
        "ply_binary_header",
    }:
        if declared_parser_bytes is None or declared_parser_bytes < 1:
            raise MeshAdmissionError(
                "declared PLY/glTF source must contain a parser byte footprint"
            )
        assert declared_vertices is not None
        assert declared_triangles is not None
        minimum_parser_bytes = (
            source_size + 24 * declared_vertices + 12 * declared_triangles
        )
        if declared_parser_bytes < minimum_parser_bytes:
            raise MeshAdmissionError(
                "declared parser byte footprint is below its source and canonical arrays"
            )
        decoded_admission_from_counts(
            vertex_count=declared_vertices,
            triangle_count=declared_triangles,
            array_bytes=declared_parser_bytes,
        )
    elif declared_parser_bytes is not None:
        raise MeshAdmissionError(
            "non-PLY mesh declaration cannot contain a parser byte footprint"
        )
    compatible_kinds = {
        "obj": {"not_available", "obj_stream"},
        "off": {"not_available", "off_header"},
        "ply": {"not_available", "ply_ascii_header", "ply_binary_header"},
        "stl": {"not_available", "ascii_stl_stream", "binary_stl_header"},
        "gltf": {"not_available", "gltf_json"},
        "glb": {"not_available", "glb_json_chunk"},
    }
    if declaration_kind not in compatible_kinds[source_format]:
        raise MeshAdmissionError(
            "mesh admission declaration kind does not match source_format"
        )
    if source_format in {"obj", "off"} or declaration_kind in {
        "ascii_stl_stream",
        "ply_ascii_header",
    }:
        _require_text_source_budget(
            source_size,
            label=f"{source_format.upper()} text source",
        )

    decoded_value = _exact_mapping(
        root["decoded"],
        {
            "array_bytes",
            "estimated_peak_bytes",
            "triangle_count",
            "vertex_count",
        },
        name="mesh admission decoded",
    )
    decoded = DecodedMeshAdmission(
        vertex_count=decoded_value["vertex_count"],
        triangle_count=decoded_value["triangle_count"],
        array_bytes=decoded_value["array_bytes"],
        estimated_peak_bytes=decoded_value["estimated_peak_bytes"],
    )
    expected_peak_bytes = (
        3 * decoded.array_bytes
        + 32 * decoded.vertex_count
        + 48 * decoded.triangle_count
    )
    if decoded.estimated_peak_bytes != expected_peak_bytes:
        raise MeshAdmissionError(
            "decoded estimated_peak_bytes does not match the v1 admission formula"
        )
    if declaration_kind != "not_available":
        assert declared_vertices is not None
        assert declared_faces is not None
        assert declared_triangles is not None
        if declared_vertices < 3 or declared_faces < 1 or declared_triangles < 1:
            raise MeshAdmissionError(
                "declared mesh source must contain an authoritative triangle mesh"
            )
        if declared_faces > declared_triangles:
            raise MeshAdmissionError(
                "declared face elements exceed declared triangulated faces"
            )
        if declared_triangles != decoded.triangle_count:
            raise MeshAdmissionError(
                "declared triangle count differs from decoded input"
            )

        # The locked OBJ and PLY parsers may split source positions at material
        # or per-face texture seams.  They must never discard declared source
        # positions.  PLY splitting is bounded by source corners; Trimesh may
        # instead retain one complete OBJ position table per material group, of
        # which there can be no more than source face elements.  Other admitted
        # formats preserve the declared vertex count.
        if declaration_kind in {
            "obj_stream",
            "ply_ascii_header",
            "ply_binary_header",
        }:
            if decoded.vertex_count < declared_vertices:
                raise MeshAdmissionError(
                    "decoded vertex count is below the declared source positions"
                )
            if declaration_kind.startswith("ply_"):
                source_corner_count = declared_triangles + 2 * declared_faces
                maximum_split_vertices = max(
                    declared_vertices,
                    source_corner_count,
                )
            else:
                maximum_split_vertices = declared_vertices * declared_faces
            if decoded.vertex_count > maximum_split_vertices:
                raise MeshAdmissionError(
                    "decoded vertex count exceeds the declared seam-split bound"
                )
        elif decoded.vertex_count != declared_vertices:
            raise MeshAdmissionError(
                "declared vertex count differs from decoded input"
            )

        if declaration_kind in {
            "ascii_stl_stream",
            "binary_stl_header",
            "glb_json_chunk",
            "gltf_json",
        } and declared_faces != declared_triangles:
            raise MeshAdmissionError(
                "declared triangle primitives must have one face element per triangle"
            )
        if declaration_kind in {"ascii_stl_stream", "binary_stl_header"} and (
            declared_vertices != 3 * declared_triangles
        ):
            raise MeshAdmissionError(
                "declared STL vertices must contain three rows per triangle"
            )

        if declared_parser_bytes is not None:
            minimum_decoded_parser_bytes = (
                source_size
                + 24 * decoded.vertex_count
                + 12 * decoded.triangle_count
            )
            if declared_parser_bytes < minimum_decoded_parser_bytes:
                raise MeshAdmissionError(
                    "declared parser byte footprint is below the source and "
                    "decoded canonical arrays"
                )
    accepted_value = _exact_mapping(
        root["accepted"],
        {
            "geometry_sha256",
            "removed_degenerate_triangle_count",
            "triangle_count",
            "vertex_count",
        },
        name="mesh admission accepted",
    )
    accepted_vertices = _strict_int(
        accepted_value["vertex_count"],
        name="accepted.vertex_count",
        minimum=3,
        maximum=MAX_MESH_VERTICES,
    )
    accepted_triangles = _strict_int(
        accepted_value["triangle_count"],
        name="accepted.triangle_count",
        minimum=1,
        maximum=MAX_MESH_TRIANGLES,
    )
    removed = _strict_int(
        accepted_value["removed_degenerate_triangle_count"],
        name="accepted.removed_degenerate_triangle_count",
        minimum=0,
        maximum=decoded.triangle_count,
    )
    if accepted_vertices != decoded.vertex_count:
        raise MeshAdmissionError("accepted vertex count differs from decoded input")
    if accepted_triangles + removed != decoded.triangle_count:
        raise MeshAdmissionError("accepted triangle accounting is inconsistent")
    geometry_sha256 = _sha256(
        accepted_value["geometry_sha256"],
        name="accepted.geometry_sha256",
    )

    checks = _exact_mapping(
        root["checks"],
        {"finite_vertices", "valid_triangle_indices"},
        name="mesh admission checks",
    )
    if checks["finite_vertices"] is not True:
        raise MeshAdmissionError("mesh admission requires finite vertices")
    if checks["valid_triangle_indices"] is not True:
        raise MeshAdmissionError("mesh admission requires valid triangle indices")

    return {
        "accepted": {
            "geometry_sha256": geometry_sha256,
            "removed_degenerate_triangle_count": removed,
            "triangle_count": accepted_triangles,
            "vertex_count": accepted_vertices,
        },
        "checks": {
            "finite_vertices": True,
            "valid_triangle_indices": True,
        },
        "declaration": {
            "face_element_count": declared_faces,
            "kind": declaration_kind,
            "parser_bytes": declared_parser_bytes,
            "triangle_count": declared_triangles,
            "vertex_count": declared_vertices,
        },
        "decoded": {
            "array_bytes": decoded.array_bytes,
            "estimated_peak_bytes": decoded.estimated_peak_bytes,
            "triangle_count": decoded.triangle_count,
            "vertex_count": decoded.vertex_count,
        },
        "limits": expected_limits,
        "profile": MESH_ADMISSION_PROFILE,
        "schema_version": MESH_ADMISSION_SCHEMA_VERSION,
        "source_format": source_format,
        "source_size_bytes": source_size,
        "status": MESH_ADMISSION_STATUS,
    }


def require_mesh_matches_admission_receipt(
    value: object,
    vertices: object,
    faces: object,
    *,
    optional_arrays: Sequence[object] = (),
    source_format: str | None = None,
    source_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Reject geometry mutation after the parser issued its admission receipt."""

    receipt = validate_mesh_admission_receipt(value)
    vertex_array = np.asarray(vertices)
    face_array = np.asarray(faces)
    if vertex_array.ndim != 2 or vertex_array.shape[1:] != (3,):
        raise MeshAdmissionError("current mesh vertices must have shape (N, 3)")
    if face_array.ndim != 2 or face_array.shape[1:] != (3,):
        raise MeshAdmissionError("current mesh faces must have shape (M, 3)")
    accepted = receipt["accepted"]
    if int(vertex_array.shape[0]) != int(accepted["vertex_count"]):
        raise MeshAdmissionError(
            "current mesh vertex count differs from its admission receipt"
        )
    if int(face_array.shape[0]) != int(accepted["triangle_count"]):
        raise MeshAdmissionError(
            "current mesh triangle count differs from its admission receipt"
        )
    current_array_bytes = int(vertex_array.nbytes) + int(face_array.nbytes)
    for value_array in optional_arrays:
        if value_array is None:
            continue
        try:
            current_array_bytes += int(np.asarray(value_array).nbytes)
        except Exception as exc:
            raise MeshAdmissionError(
                "current optional mesh array is unreadable"
            ) from exc
    if int(receipt["decoded"]["array_bytes"]) < current_array_bytes:
        raise MeshAdmissionError(
            "current mesh arrays exceed the decoded bytes in its admission receipt"
        )
    if source_format is not None:
        normalized_format = _source_format(source_format)
        if receipt["source_format"] != normalized_format:
            raise MeshAdmissionError(
                "current mesh source format differs from its admission receipt"
            )
    if source_size_bytes is not None:
        current_source_size = _strict_int(
            source_size_bytes,
            name="current source_size_bytes",
            minimum=1,
            maximum=MAX_MESH_SOURCE_BYTES,
        )
        if int(receipt["source_size_bytes"]) != current_source_size:
            raise MeshAdmissionError(
                "current mesh source size differs from its admission receipt"
            )
    observed_sha256 = admitted_geometry_sha256(vertex_array, face_array)
    if observed_sha256 != accepted["geometry_sha256"]:
        raise MeshAdmissionError(
            "current mesh geometry differs from its admission receipt "
            "(SHA-256 mismatch)"
        )
    return receipt


__all__ = [
    "admitted_geometry_sha256",
    "DecodedMeshAdmission",
    "decoded_admission_from_counts",
    "estimated_texture_bytes",
    "MAX_MESH_DECODED_ARRAY_BYTES",
    "MAX_MESH_DEPENDENCY_BYTES",
    "MAX_MESH_DEPENDENCY_TOTAL_BYTES",
    "MAX_MESH_ESTIMATED_PEAK_BYTES",
    "MAX_MESH_GLTF_JSON_BYTES",
    "MAX_MESH_HEADER_BYTES",
    "MAX_MESH_PLY_AUXILIARY_BYTES",
    "MAX_MESH_PLY_FACE_PROPERTIES",
    "MAX_MESH_PLY_VERTEX_PROPERTIES",
    "MAX_MESH_SOURCE_BYTES",
    "MAX_MESH_TEXT_SOURCE_BYTES",
    "MAX_MESH_TEXTURE_BYTES",
    "MAX_MESH_TRIANGLES",
    "MAX_MESH_VERTICES",
    "MESH_ADMISSION_PROFILE",
    "MESH_ADMISSION_SCHEMA_VERSION",
    "MESH_ADMISSION_STATUS",
    "MeshAdmissionError",
    "MeshSourcePreflight",
    "build_mesh_admission_receipt",
    "inspect_decoded_mesh",
    "mesh_admission_receipt_for_arrays",
    "preflight_mesh_source",
    "require_mesh_matches_admission_receipt",
    "require_texture_budget",
    "require_windows_runtime_capacity",
    "validate_mesh_admission_receipt",
    "WINDOWS_RUNTIME_MEMORY_RESERVE_BYTES",
]
