from __future__ import annotations

from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct
from types import SimpleNamespace
from typing import BinaryIO
from unittest import mock

import numpy as np
import pytest
import trimesh

import src.core.mesh_admission as mesh_admission
import src.core.mesh_loader as mesh_loader
import src.core.source_identity as source_identity
from src.core.artifact_session import ArtifactSession, ArtifactSessionError
from src.core.mesh_admission import (
    MAX_MESH_DEPENDENCY_BYTES,
    MAX_MESH_DEPENDENCY_TOTAL_BYTES,
    MAX_MESH_SOURCE_BYTES,
    MAX_MESH_TRIANGLES,
    MAX_MESH_VERTICES,
    WINDOWS_RUNTIME_MEMORY_RESERVE_BYTES,
    MeshAdmissionError,
    decoded_admission_from_counts,
    inspect_decoded_mesh,
    mesh_admission_receipt_for_arrays,
    preflight_mesh_source,
    require_mesh_matches_admission_receipt,
    require_windows_runtime_capacity,
    validate_mesh_admission_receipt,
)
from src.core.mesh_loader import MeshLoader
from src.core.project_file import (
    load_artifact_session_project,
    save_artifact_session_project,
)
from src.core.source_identity import (
    SourceChangedError,
    SourceFingerprint,
    SourceSizeLimitError,
    open_fingerprinted_file,
)
from src.core.source_manifest import (
    DEPENDENCY_RESOURCE_ROLE,
    PRIMARY_RESOURCE_ROLE,
    SourceManifest,
    SourceManifestEntry,
    SourceManifestError,
    fixed_media_type,
)


TRIANGLE_PLY = b"""ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
0 1 0
3 0 1 2
"""

QUAD_OBJ = b"""v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"""

QUAD_OFF = b"""OFF
4 1 0
0 0 0
1 0 0
1 1 0
0 1 0
4 0 1 2 3
"""

QUAD_ASCII_PLY = b"""ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
1 1 0
0 1 0
4 0 1 2 3
"""


def _zero_initialized_triangle_gltf(*, accessor_count: int = 3) -> bytes:
    document = {
        "asset": {"version": "2.0"},
        "bufferViews": [],
        "accessors": [
            {
                "componentType": 5126,
                "count": accessor_count,
                "type": "VEC3",
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "mode": 4,
                    }
                ]
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    return json.dumps(document, separators=(",", ":")).encode("utf-8")

TRIANGLE_ASCII_STL = b"""solid triangle
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
endsolid triangle
"""


def _triangle_arrays() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        np.array([[0, 1, 2]], dtype=np.int32),
    )


def _manifest_entry(
    logical_path: str,
    *,
    role: str,
    size_bytes: int,
    digest: str = "0" * 64,
) -> SourceManifestEntry:
    return SourceManifestEntry(
        logical_path=logical_path,
        media_type=fixed_media_type(logical_path),
        role=role,
        sha256=digest,
        size_bytes=size_bytes,
    )


class _TrackingHash:
    def __init__(self) -> None:
        self._delegate = hashlib.new("sha256")
        self.update_sizes: list[int] = []

    def update(self, data: bytes) -> None:
        self.update_sizes.append(len(data))
        self._delegate.update(data)

    def hexdigest(self) -> str:
        return self._delegate.hexdigest()


def test_four_gib_source_cap_rejects_open_descriptor_before_hashing_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "oversized.ply"
    source.write_bytes(b"x")
    baseline = source_identity._stat_snapshot(source.stat())
    oversized = replace(baseline, size_bytes=MAX_MESH_SOURCE_BYTES + 1)
    tracker = _TrackingHash()

    assert MAX_MESH_SOURCE_BYTES == 1 << 32
    assert source.stat().st_size == 1
    with mock.patch.object(
        source_identity,
        "_stat_snapshot",
        return_value=oversized,
    ), mock.patch.object(
        source_identity.hashlib,
        "sha256",
        return_value=tracker,
    ):
        with pytest.raises(SourceSizeLimitError, match="safety limit"):
            with open_fingerprinted_file(
                source,
                max_size_bytes=MAX_MESH_SOURCE_BYTES,
            ):
                pytest.fail("an oversized descriptor must never be yielded")

    assert tracker.update_sizes == []


def test_source_growth_cannot_read_past_the_hashing_cap(tmp_path: Path) -> None:
    source = tmp_path / "growing.ply"
    source.write_bytes(b"mesh")

    class _GrowingHash(_TrackingHash):
        def __init__(self) -> None:
            super().__init__()
            self.grew = False

        def update(self, data: bytes) -> None:
            super().update(data)
            if not self.grew:
                with source.open("ab", buffering=0) as writer:
                    writer.write(b"x")
                self.grew = True

    tracker = _GrowingHash()
    with mock.patch.object(
        source_identity.hashlib,
        "sha256",
        return_value=tracker,
    ):
        with pytest.raises(SourceSizeLimitError, match="grew beyond"):
            with open_fingerprinted_file(
                source,
                chunk_size=4,
                max_size_bytes=4,
            ):
                pytest.fail("a source that grows past the cap must not be yielded")

    assert tracker.update_sizes == [4]
    assert source.stat().st_size == 5


def test_mesh_loader_applies_source_cap_before_invoking_parser(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    source.write_bytes(b"x")
    limit_error = SourceSizeLimitError("synthetic descriptor exceeds source cap")

    with mock.patch.object(
        mesh_loader,
        "open_fingerprinted_file",
        side_effect=limit_error,
    ) as open_source, mock.patch.object(
        mesh_loader,
        "_load_authoritative_trimesh",
    ) as parse_mesh:
        with pytest.raises(SourceSizeLimitError, match="source cap"):
            MeshLoader().load(source, compute_face_normals=False)

    assert open_source.call_args.kwargs["max_size_bytes"] == MAX_MESH_SOURCE_BYTES
    parse_mesh.assert_not_called()


def test_path_parser_cannot_read_bytes_appended_after_fingerprint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.ply"
    source.write_bytes(TRIANGLE_PLY)
    observed: list[bytes] = []
    vertices, faces = _triangle_arrays()

    def _growing_parser(stream: BinaryIO, **_kwargs: object) -> trimesh.Trimesh:
        with source.open("ab", buffering=0) as writer:
            writer.write(b"unverified-tail")
        stream.seek(0)
        observed.append(stream.read())
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    with mock.patch.object(
        mesh_loader,
        "_load_authoritative_trimesh",
        side_effect=_growing_parser,
    ):
        with pytest.raises(SourceChangedError):
            MeshLoader().load(source, compute_face_normals=False)

    assert observed == [TRIANGLE_PLY]


def test_path_parser_reads_snapshot_when_source_is_rewritten_same_length(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.ply"
    source.write_bytes(TRIANGLE_PLY)
    rewritten = TRIANGLE_PLY.replace(b"1 0 0", b"9 0 0", 1)
    assert len(rewritten) == len(TRIANGLE_PLY)
    observed: list[bytes] = []
    vertices, faces = _triangle_arrays()
    real_preflight = mesh_loader.preflight_mesh_source

    def _rewriting_preflight(
        stream: BinaryIO,
        *,
        source_format: str,
        source_size_bytes: int,
    ) -> mesh_admission.MeshSourcePreflight:
        source.write_bytes(rewritten)
        return real_preflight(
            stream,
            source_format=source_format,
            source_size_bytes=source_size_bytes,
        )

    def _observing_parser(stream: BinaryIO, **_kwargs: object) -> trimesh.Trimesh:
        stream.seek(0)
        observed.append(stream.read())
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    with mock.patch.object(
        mesh_loader,
        "preflight_mesh_source",
        side_effect=_rewriting_preflight,
    ), mock.patch.object(
        mesh_loader,
        "_load_authoritative_trimesh",
        side_effect=_observing_parser,
    ):
        with pytest.raises(SourceChangedError):
            MeshLoader().load(source, compute_face_normals=False)

    assert observed == [TRIANGLE_PLY]


def test_same_length_rewrite_before_snapshot_is_rejected_before_parser(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.ply"
    source.write_bytes(TRIANGLE_PLY)
    rewritten = TRIANGLE_PLY.replace(b"1 0 0", b"9 0 0", 1)
    real_spool = mesh_loader._spool_verified_source

    def _rewriting_spool(
        source_stream: BinaryIO,
        target_stream: BinaryIO,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
        mismatch_error: type[Exception],
    ) -> str:
        source.write_bytes(rewritten)
        return real_spool(
            source_stream,
            target_stream,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            mismatch_error=mismatch_error,
        )

    with mock.patch.object(
        mesh_loader,
        "_spool_verified_source",
        side_effect=_rewriting_spool,
    ), mock.patch.object(
        mesh_loader,
        "_load_authoritative_trimesh",
    ) as parser:
        with pytest.raises(SourceChangedError, match="SHA-256 mismatch"):
            MeshLoader().load(source, compute_face_normals=False)

    parser.assert_not_called()


def test_declared_and_decoded_count_caps_need_no_large_allocation() -> None:
    oversized_header = f"""ply
format ascii 1.0
element vertex {MAX_MESH_VERTICES + 1}
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
""".encode("ascii")
    with pytest.raises(MeshAdmissionError, match="declared vertices"):
        preflight_mesh_source(
            BytesIO(oversized_header),
            source_format="ply",
            source_size_bytes=len(oversized_header),
        )

    with pytest.raises(MeshAdmissionError, match="decoded.vertex_count"):
        decoded_admission_from_counts(
            vertex_count=MAX_MESH_VERTICES + 1,
            triangle_count=1,
            array_bytes=1,
        )
    with pytest.raises(MeshAdmissionError, match="decoded.triangle_count"):
        decoded_admission_from_counts(
            vertex_count=3,
            triangle_count=MAX_MESH_TRIANGLES + 1,
            array_bytes=1,
        )


@pytest.mark.parametrize(
    ("source_bytes", "source_format", "expected_kind"),
    [
        (QUAD_OBJ, "obj", "obj_stream"),
        (QUAD_OFF, "off", "off_header"),
        (QUAD_ASCII_PLY, "ply", "ply_ascii_header"),
        (TRIANGLE_ASCII_STL, "stl", "ascii_stl_stream"),
    ],
)
def test_text_preflight_records_polygon_triangulation(
    source_bytes: bytes,
    source_format: str,
    expected_kind: str,
) -> None:
    preflight = preflight_mesh_source(
        BytesIO(source_bytes),
        source_format=source_format,
        source_size_bytes=len(source_bytes),
    )

    assert preflight.declaration_kind == expected_kind
    expected_vertices = 3 if source_format == "stl" else 4
    expected_triangles = 1 if source_format == "stl" else 2
    assert preflight.declared_vertex_count == expected_vertices
    assert preflight.declared_face_element_count == 1
    assert preflight.declared_triangle_count == expected_triangles


def test_binary_ply_preflight_counts_polygon_without_decoding_arrays() -> None:
    header = b"""ply
format binary_little_endian 1.0
element vertex 4
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
"""
    vertices = struct.pack(
        "<12f",
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    face = struct.pack("<B4i", 4, 0, 1, 2, 3)
    payload = header + vertices + face

    preflight = preflight_mesh_source(
        BytesIO(payload),
        source_format="ply",
        source_size_bytes=len(payload),
    )

    assert preflight.declared_triangle_count == 2
    assert preflight.declaration_kind == "ply_binary_header"
    assert preflight.declared_parser_bytes is not None
    assert preflight.declared_parser_bytes > len(payload)


def test_binary_ply_rejects_variable_list_lengths_before_parser(
    tmp_path: Path,
) -> None:
    header = b"""ply
format binary_little_endian 1.0
element vertex 6
property float x
property float y
property float z
element face 3
property list uchar int vertex_indices
end_header
"""
    vertices = struct.pack(
        "<18f",
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        2,
        1,
        0,
        2,
        0,
        0,
    )
    # Counts 4/3/5 have the same aggregate item count as three quads, so the
    # locked parser's fixed-first-row dtype would accept the total byte length
    # while silently reading the later row boundaries incorrectly.
    faces = b"".join(
        (
            struct.pack("<B4i", 4, 0, 1, 2, 3),
            struct.pack("<B3i", 3, 0, 2, 3),
            struct.pack("<B5i", 5, 0, 1, 4, 5, 3),
        )
    )
    source_path = tmp_path / "variable-lists.ply"
    source_path.write_bytes(header + vertices + faces)

    with mock.patch(
        "src.core.mesh_loader._load_authoritative_trimesh"
    ) as parser:
        with pytest.raises(MeshAdmissionError, match="list lengths must remain constant"):
            MeshLoader().load(source_path)
    parser.assert_not_called()


def test_gltf_duplicate_buffer_views_exceed_parser_budget_before_parser(
    tmp_path: Path,
) -> None:
    buffer_size = 1024 * 1024
    duplicate_view_count = 2048
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": buffer_size, "uri": "geometry.bin"}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
            *(
                {
                    "buffer": 0,
                    "byteOffset": 1,
                    "byteLength": buffer_size - 1,
                }
                for _ in range(duplicate_view_count)
            ),
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR",
            },
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "mode": 4,
                    }
                ]
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    source_path = tmp_path / "duplicate-views.gltf"
    source_path.write_text(
        json.dumps(document, separators=(",", ":")),
        encoding="utf-8",
    )
    (tmp_path / "geometry.bin").write_bytes(b"\0" * buffer_size)

    with mock.patch(
        "src.core.mesh_loader._load_authoritative_trimesh"
    ) as parser:
        with pytest.raises(MeshAdmissionError, match="bufferView slices"):
            MeshLoader().load(source_path)
    parser.assert_not_called()


def test_gltf_accessors_without_buffer_views_key_rejected_before_parser(
    tmp_path: Path,
) -> None:
    document = json.loads(_zero_initialized_triangle_gltf())
    del document["bufferViews"]
    source_path = tmp_path / "missing-buffer-views.gltf"
    source_path.write_text(
        json.dumps(document, separators=(",", ":")),
        encoding="utf-8",
    )

    with mock.patch(
        "src.core.mesh_loader._load_authoritative_trimesh"
    ) as parser:
        with pytest.raises(
            MeshAdmissionError,
            match="accessors require a top-level bufferViews array",
        ):
            MeshLoader().load(source_path)
    parser.assert_not_called()


def test_binary_ply_rejects_parser_retained_arbitrary_element() -> None:
    header = b"""ply
format binary_little_endian 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
element junk 4
property uchar payload
end_header
"""
    payload = (
        header
        + struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
        + struct.pack("<B3i", 3, 0, 1, 2)
        + b"junk"
    )

    with pytest.raises(MeshAdmissionError, match="exactly vertex then face"):
        preflight_mesh_source(
            BytesIO(payload),
            source_format="ply",
            source_size_bytes=len(payload),
        )


def test_binary_ply_rejects_trailing_undeclared_payload() -> None:
    header = b"""ply
format binary_little_endian 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
"""
    payload = (
        header
        + struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
        + struct.pack("<B3i", 3, 0, 1, 2)
        + b"undeclared"
    )

    with pytest.raises(MeshAdmissionError, match="payload length"):
        preflight_mesh_source(
            BytesIO(payload),
            source_format="ply",
            source_size_bytes=len(payload),
        )


def test_binary_ply_rejects_auxiliary_property_budget_before_parser() -> None:
    header = b"""ply
format binary_little_endian 1.0
element vertex 3
property float x
property float y
property float z
property uchar red
element face 1
property list uchar int vertex_indices
end_header
"""
    payload = (
        header
        + b"".join(
            struct.pack("<3fB", *row)
            for row in (
                (0.0, 0.0, 0.0, 1),
                (1.0, 0.0, 0.0, 2),
                (0.0, 1.0, 0.0, 3),
            )
        )
        + struct.pack("<B3i", 3, 0, 1, 2)
    )

    with mock.patch.object(mesh_admission, "MAX_MESH_PLY_AUXILIARY_BYTES", 2):
        with pytest.raises(MeshAdmissionError, match="auxiliary properties"):
            preflight_mesh_source(
                BytesIO(payload),
                source_format="ply",
                source_size_bytes=len(payload),
            )


@pytest.mark.parametrize(
    ("source_bytes", "source_format"),
    [
        (QUAD_OBJ, "obj"),
        (QUAD_OFF, "off"),
        (QUAD_ASCII_PLY, "ply"),
        (TRIANGLE_ASCII_STL, "stl"),
    ],
)
def test_text_parser_source_cap_rejects_before_decode(
    source_bytes: bytes,
    source_format: str,
) -> None:
    with mock.patch.object(
        mesh_admission,
        "MAX_MESH_TEXT_SOURCE_BYTES",
        len(source_bytes) - 1,
    ):
        with pytest.raises(MeshAdmissionError, match="text-parser limit"):
            preflight_mesh_source(
                BytesIO(source_bytes),
                source_format=source_format,
                source_size_bytes=len(source_bytes),
            )


def test_gltf_json_preflight_counts_triangle_primitives() -> None:
    payload = _zero_initialized_triangle_gltf()

    preflight = preflight_mesh_source(
        BytesIO(payload),
        source_format="gltf",
        source_size_bytes=len(payload),
    )

    assert preflight.declaration_kind == "gltf_json"
    assert preflight.declared_vertex_count == 3
    assert preflight.declared_face_element_count == 1
    assert preflight.declared_triangle_count == 1
    assert preflight.declared_parser_bytes is not None
    assert preflight.declared_parser_bytes > len(payload)


def test_glb_bin_chunk_cannot_exceed_declared_buffer_before_parser(
    tmp_path: Path,
) -> None:
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 36}],
        "bufferViews": [{"buffer": 0, "byteLength": 36}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
            }
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_chunk = b"\x00" * (1024 * 1024)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    payload = b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total_length),
            struct.pack("<I4s", len(json_chunk), b"JSON"),
            json_chunk,
            struct.pack("<I4s", len(bin_chunk), b"BIN\x00"),
            bin_chunk,
        )
    )
    source = tmp_path / "oversized-bin.glb"
    source.write_bytes(payload)

    with mock.patch.object(mesh_loader, "_load_authoritative_trimesh") as parser:
        with pytest.raises(MeshAdmissionError, match="BIN chunk length differs"):
            MeshLoader().load(source, compute_face_normals=False)

    parser.assert_not_called()


def test_gltf_accessor_amplification_is_rejected_before_parser(
    tmp_path: Path,
) -> None:
    source = tmp_path / "amplified.gltf"
    source.write_bytes(_zero_initialized_triangle_gltf(accessor_count=1_000_000_002))

    with mock.patch.object(
        mesh_loader,
        "_load_authoritative_trimesh",
    ) as parser:
        with pytest.raises(MeshAdmissionError, match=r"accessor\[0\]\.count"):
            MeshLoader().load(source, compute_face_normals=False)

    parser.assert_not_called()


@pytest.mark.parametrize(
    ("source_bytes", "source_format"),
    [
        (QUAD_OBJ, "obj"),
        (QUAD_OFF, "off"),
        (QUAD_ASCII_PLY, "ply"),
    ],
)
def test_polygon_amplification_is_rejected_before_parser(
    source_bytes: bytes,
    source_format: str,
) -> None:
    with mock.patch.object(mesh_admission, "MAX_MESH_TRIANGLES", 1):
        with pytest.raises(MeshAdmissionError, match="triangulated faces"):
            preflight_mesh_source(
                BytesIO(source_bytes),
                source_format=source_format,
                source_size_bytes=len(source_bytes),
            )


def test_obj_polygon_limit_stops_mesh_loader_before_trimesh(tmp_path: Path) -> None:
    source = tmp_path / "quad.obj"
    source.write_bytes(QUAD_OBJ)

    with mock.patch.object(
        mesh_admission,
        "MAX_MESH_TRIANGLES",
        1,
    ), mock.patch.object(
        mesh_loader,
        "_load_authoritative_trimesh",
    ) as parser:
        with pytest.raises(MeshAdmissionError, match="triangulated faces"):
            MeshLoader().load(source, compute_face_normals=False)

    parser.assert_not_called()


def test_decoded_mesh_requires_finite_vertices_and_valid_indices() -> None:
    vertices, faces = _triangle_arrays()
    non_finite = vertices.copy()
    non_finite[1, 2] = np.nan
    with pytest.raises(MeshAdmissionError, match="non-finite"):
        inspect_decoded_mesh(non_finite, faces)

    out_of_range = faces.copy()
    out_of_range[0, 2] = len(vertices)
    with pytest.raises(MeshAdmissionError, match="invalid triangle index"):
        inspect_decoded_mesh(vertices, out_of_range)

    negative = faces.copy()
    negative[0, 0] = -1
    with pytest.raises(MeshAdmissionError, match="invalid triangle index"):
        inspect_decoded_mesh(vertices, negative)


def test_receipt_binds_counts_and_exact_geometry_hash() -> None:
    vertices, faces = _triangle_arrays()
    receipt = mesh_admission_receipt_for_arrays(
        vertices,
        faces,
        source_format="ply",
        source_size_bytes=len(TRIANGLE_PLY),
    )
    json_round_trip = json.loads(json.dumps(receipt))

    assert validate_mesh_admission_receipt(json_round_trip) == receipt
    assert require_mesh_matches_admission_receipt(receipt, vertices, faces) == receipt

    changed_vertices = vertices.copy()
    changed_vertices[1, 0] += 0.125
    with pytest.raises(MeshAdmissionError, match="geometry differs"):
        require_mesh_matches_admission_receipt(receipt, changed_vertices, faces)

    changed_faces = np.vstack((faces, faces))
    with pytest.raises(MeshAdmissionError, match="triangle count differs"):
        require_mesh_matches_admission_receipt(receipt, vertices, changed_faces)


def test_receipt_cannot_underreport_current_array_bytes_or_source_identity() -> None:
    vertices, faces = _triangle_arrays()
    receipt = mesh_admission_receipt_for_arrays(
        vertices,
        faces,
        source_format="ply",
        source_size_bytes=len(TRIANGLE_PLY),
    )
    forged = json.loads(json.dumps(receipt))
    forged["decoded"]["array_bytes"] = 1
    forged["decoded"]["estimated_peak_bytes"] = (
        3
        + 32 * forged["decoded"]["vertex_count"]
        + 48 * forged["decoded"]["triangle_count"]
    )

    with pytest.raises(MeshAdmissionError, match="arrays exceed"):
        require_mesh_matches_admission_receipt(forged, vertices, faces)
    with pytest.raises(MeshAdmissionError, match="source format differs"):
        require_mesh_matches_admission_receipt(
            receipt,
            vertices,
            faces,
            source_format="obj",
        )
    with pytest.raises(MeshAdmissionError, match="source size differs"):
        require_mesh_matches_admission_receipt(
            receipt,
            vertices,
            faces,
            source_size_bytes=len(TRIANGLE_PLY) + 1,
        )


def test_receipt_matches_public_json_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    vertices, faces = _triangle_arrays()
    preflight = preflight_mesh_source(
        BytesIO(TRIANGLE_PLY),
        source_format="ply",
        source_size_bytes=len(TRIANGLE_PLY),
    )
    decoded = inspect_decoded_mesh(vertices, faces)
    receipt = mesh_admission.build_mesh_admission_receipt(
        preflight,
        decoded,
        accepted_vertex_count=3,
        accepted_triangle_count=1,
        accepted_geometry_sha256=mesh_admission.admitted_geometry_sha256(
            vertices,
            faces,
        ),
    )
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "mesh_admission_receipt-1.0.0.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    assert list(jsonschema.Draft202012Validator(schema).iter_errors(receipt)) == []

    forged = json.loads(json.dumps(receipt))
    forged["declaration"]["parser_bytes"] = len(TRIANGLE_PLY)
    with pytest.raises(MeshAdmissionError, match="below its source"):
        validate_mesh_admission_receipt(forged)

    wrong_kind = json.loads(json.dumps(receipt))
    wrong_kind["source_format"] = "stl"
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(wrong_kind))

    false_no_declaration = json.loads(json.dumps(receipt))
    false_no_declaration["declaration"]["kind"] = "not_available"
    false_no_declaration["declaration"]["parser_bytes"] = None
    assert list(
        jsonschema.Draft202012Validator(schema).iter_errors(false_no_declaration)
    )


def test_receipt_rejects_declared_counts_that_contradict_decoded_input() -> None:
    vertices, faces = _triangle_arrays()
    preflight = preflight_mesh_source(
        BytesIO(TRIANGLE_PLY),
        source_format="ply",
        source_size_bytes=len(TRIANGLE_PLY),
    )
    decoded = inspect_decoded_mesh(vertices, faces)
    receipt = mesh_admission.build_mesh_admission_receipt(
        preflight,
        decoded,
        accepted_vertex_count=3,
        accepted_triangle_count=1,
        accepted_geometry_sha256=mesh_admission.admitted_geometry_sha256(
            vertices,
            faces,
        ),
    )

    wrong_triangles = json.loads(json.dumps(receipt))
    declaration = wrong_triangles["declaration"]
    declaration["face_element_count"] = 2
    declaration["triangle_count"] = 2
    declaration["vertex_count"] = 4
    declaration["parser_bytes"] = len(TRIANGLE_PLY) + 24 * 4 + 12 * 2
    with pytest.raises(MeshAdmissionError, match="declared triangle count"):
        validate_mesh_admission_receipt(wrong_triangles)

    too_many_source_vertices = json.loads(json.dumps(receipt))
    declaration = too_many_source_vertices["declaration"]
    declaration["vertex_count"] = 4
    declaration["parser_bytes"] = len(TRIANGLE_PLY) + 24 * 4 + 12
    with pytest.raises(MeshAdmissionError, match="below the declared source positions"):
        validate_mesh_admission_receipt(too_many_source_vertices)


def test_receipt_parser_footprint_covers_ply_seam_split_decoded_arrays() -> None:
    seam_ply = b"""ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
property list uchar float texcoord
end_header
0 0 0
1 0 0
1 1 0
0 1 0
3 0 1 2 6 0 0 1 0 1 1
3 0 2 3 6 .5 .5 1 1 0 1
"""
    preflight = preflight_mesh_source(
        BytesIO(seam_ply),
        source_format="ply",
        source_size_bytes=len(seam_ply),
    )
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [4, 2, 3]], dtype=np.int32)
    decoded = inspect_decoded_mesh(vertices, faces)
    receipt = mesh_admission.build_mesh_admission_receipt(
        preflight,
        decoded,
        accepted_vertex_count=5,
        accepted_triangle_count=2,
        accepted_geometry_sha256=mesh_admission.admitted_geometry_sha256(
            vertices,
            faces,
        ),
    )

    forged = json.loads(json.dumps(receipt))
    forged["declaration"]["parser_bytes"] = len(seam_ply) + 24 * 4 + 12 * 2
    with pytest.raises(MeshAdmissionError, match="decoded canonical arrays"):
        validate_mesh_admission_receipt(forged)


def test_sidecar_budgets_reject_metadata_before_replay_and_bound_capture(
    tmp_path: Path,
) -> None:
    primary = _manifest_entry(
        "artifact.obj",
        role=PRIMARY_RESOURCE_ROLE,
        size_bytes=1,
    )
    oversized_dependency = _manifest_entry(
        "texture.png",
        role=DEPENDENCY_RESOURCE_ROLE,
        size_bytes=MAX_MESH_DEPENDENCY_BYTES + 1,
    )
    oversized_manifest = SourceManifest(
        primary_logical_path=primary.logical_path,
        entries=(primary, oversized_dependency),
    )
    replay_loader = mock.Mock(return_value=(b"", "unused"))
    replay = mesh_loader._ClosedManifestResolver(oversized_manifest, replay_loader)

    with pytest.raises(FileNotFoundError, match="per-resource byte budget"):
        replay.get(oversized_dependency.logical_path)
    replay_loader.assert_not_called()

    small_dependency = _manifest_entry(
        "material.mtl",
        role=DEPENDENCY_RESOURCE_ROLE,
        size_bytes=1,
        digest=hashlib.sha256(b"x").hexdigest(),
    )
    small_manifest = SourceManifest(
        primary_logical_path=primary.logical_path,
        entries=(primary, small_dependency),
    )
    aggregate_loader = mock.Mock(return_value=(b"x", "unused"))
    aggregate_replay = mesh_loader._ClosedManifestResolver(
        small_manifest,
        aggregate_loader,
    )
    aggregate_replay._state.total_dependency_bytes = (
        MAX_MESH_DEPENDENCY_TOTAL_BYTES
    )

    with pytest.raises(FileNotFoundError, match="decoded-byte budget"):
        aggregate_replay.get(small_dependency.logical_path)
    aggregate_loader.assert_not_called()

    capture = mesh_loader._CapturingDirectoryResolver(tmp_path, "artifact.obj")
    capture._state.total_dependency_bytes = MAX_MESH_DEPENDENCY_TOTAL_BYTES
    fingerprint = SourceFingerprint(
        sha256=hashlib.sha256(b"x").hexdigest(),
        size_bytes=1,
        mtime_ns=0,
        original_name="material.mtl",
        format="mtl",
    )
    with mock.patch.object(
        mesh_loader,
        "_read_resolved_resource",
        return_value=(b"x", str(tmp_path / "material.mtl"), fingerprint),
    ) as read_resource:
        with pytest.raises(FileNotFoundError, match="decoded-byte budget"):
            capture.get("material.mtl")

    read_resource.assert_not_called()
    assert capture._state.total_dependency_bytes == MAX_MESH_DEPENDENCY_TOTAL_BYTES
    assert capture._state.payloads == {}
    assert capture._state.resources == {}


def test_sidecar_payload_read_uses_verified_length_plus_one_sentinel(
    tmp_path: Path,
) -> None:
    dependency = tmp_path / "texture.bin"
    dependency.write_bytes(b"abcdef")
    fingerprint = SourceFingerprint(
        sha256=hashlib.sha256(b"abcde").hexdigest(),
        size_bytes=5,
        mtime_ns=dependency.stat().st_mtime_ns,
        original_name=dependency.name,
        format="bin",
    )
    raw_stream = dependency.open("rb")

    class _TrackingStream:
        def __init__(self) -> None:
            self.read_sizes: list[int] = []

        def fileno(self) -> int:
            return raw_stream.fileno()

        def read(self, size: int = -1) -> bytes:
            self.read_sizes.append(size)
            return raw_stream.read(size)

    tracked = _TrackingStream()
    opened = mock.MagicMock()
    opened.__enter__.return_value = (tracked, fingerprint)
    opened.__exit__.return_value = False
    try:
        with mock.patch.object(
            mesh_loader,
            "open_fingerprinted_file",
            return_value=opened,
        ):
            with pytest.raises(SourceManifestError, match="size changed"):
                mesh_loader._read_resolved_resource(
                    tmp_path,
                    dependency.name,
                    max_size_bytes=5,
                )
    finally:
        raw_stream.close()

    assert tracked.read_sizes == [6]


def test_scene_aggregate_is_rejected_before_concatenate(tmp_path: Path) -> None:
    source = tmp_path / "scene.ply"
    source.write_bytes(TRIANGLE_PLY)
    vertices, faces = _triangle_arrays()
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
        geom_name="first",
        node_name="first",
    )
    scene.add_geometry(
        trimesh.Trimesh(vertices=vertices + 2.0, faces=faces, process=False),
        geom_name="second",
        node_name="second",
    )

    with mock.patch.object(
        mesh_admission,
        "MAX_MESH_VERTICES",
        5,
    ), mock.patch.object(
        mesh_loader,
        "_load_authoritative_trimesh",
        return_value=scene,
    ), mock.patch.object(
        mesh_loader.trimesh.util,
        "concatenate",
    ) as concatenate:
        with pytest.raises(MeshAdmissionError, match="decoded.vertex_count"):
            MeshLoader().load(source, compute_face_normals=False)

    concatenate.assert_not_called()


def test_actual_multigeometry_loader_cannot_merge_before_admission(
    tmp_path: Path,
) -> None:
    vertices, faces = _triangle_arrays()
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
        geom_name="first",
        node_name="first",
    )
    scene.add_geometry(
        trimesh.Trimesh(vertices=vertices + 2.0, faces=faces, process=False),
        geom_name="second",
        node_name="second",
    )
    source = tmp_path / "two-components.glb"
    glb_bytes = scene.export(file_type="glb")
    assert isinstance(glb_bytes, bytes)
    source.write_bytes(glb_bytes)

    with mock.patch.object(
        mesh_admission,
        "MAX_MESH_VERTICES",
        5,
    ), mock.patch.object(
        mesh_loader.trimesh.util,
        "concatenate",
    ) as concatenate:
        with pytest.raises(
            MeshAdmissionError,
            match="declared glTF scene vertices",
        ):
            MeshLoader().load(source, compute_face_normals=False)

    concatenate.assert_not_called()


def test_scene_admission_counts_repeated_graph_instances() -> None:
    vertices, faces = _triangle_arrays()
    component = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    scene = trimesh.Scene()
    scene.add_geometry(component, geom_name="shared", node_name="first")
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = 10.0
    scene.graph.update(
        frame_to="second",
        matrix=transform,
        geometry="shared",
    )

    with mock.patch.object(
        mesh_admission,
        "MAX_MESH_VERTICES",
        5,
    ), mock.patch.object(
        mesh_loader.trimesh.util,
        "concatenate",
    ) as concatenate:
        with pytest.raises(MeshAdmissionError, match="decoded.vertex_count"):
            mesh_loader._materialize_admitted_scene(scene)

    concatenate.assert_not_called()


def test_mixed_scene_never_copies_unadmitted_point_geometry() -> None:
    component = trimesh.Trimesh(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        process=False,
    )
    points = trimesh.points.PointCloud(
        np.asarray([[100.0, 100.0, 100.0]], dtype=np.float64)
    )
    scene = trimesh.Scene()
    scene.add_geometry(component, geom_name="surface", node_name="surface-node")
    scene.add_geometry(points, geom_name="side-points", node_name="points-node")

    with mock.patch.object(
        points,
        "copy",
        side_effect=AssertionError("unadmitted PointCloud was copied"),
    ) as point_copy:
        materialized = mesh_loader._materialize_admitted_scene(scene)

    point_copy.assert_not_called()
    assert isinstance(materialized, trimesh.Trimesh)
    np.testing.assert_array_equal(materialized.faces, component.faces)
    np.testing.assert_allclose(materialized.vertices, component.vertices)


def test_windows_runtime_gate_uses_mocked_physical_and_commit_capacity() -> None:
    peak = 64 * 1024 * 1024
    required = peak + WINDOWS_RUNTIME_MEMORY_RESERVE_BYTES
    windows = SimpleNamespace(name="nt")

    with mock.patch.object(mesh_admission, "os", windows), mock.patch.object(
        mesh_admission,
        "_windows_available_memory_bytes",
        side_effect=[
            (required, required),
            (required - 1, required),
            (required, required - 1),
        ],
    ) as available:
        require_windows_runtime_capacity(peak)
        with pytest.raises(MeshAdmissionError, match="required free memory"):
            require_windows_runtime_capacity(peak)
        with pytest.raises(MeshAdmissionError, match="required free memory"):
            require_windows_runtime_capacity(peak)

    assert available.call_count == 3

    with mock.patch.object(mesh_admission, "os", windows), mock.patch.object(
        mesh_admission,
        "_windows_available_memory_bytes",
        side_effect=OSError("GlobalMemoryStatusEx failed"),
    ):
        with pytest.raises(MeshAdmissionError, match="could not be verified"):
            require_windows_runtime_capacity(peak)

    with mock.patch.object(
        mesh_admission,
        "os",
        SimpleNamespace(name="posix"),
    ), mock.patch.object(
        mesh_admission,
        "_windows_available_memory_bytes",
    ) as unavailable_outside_windows:
        require_windows_runtime_capacity(peak)

    unavailable_outside_windows.assert_not_called()


def test_admission_receipt_survives_embedded_project_round_trip(
    tmp_path: Path,
) -> None:
    source = tmp_path / "artifact.ply"
    source.write_bytes(TRIANGLE_PLY)
    mesh = MeshLoader(default_unit="mm").load(
        source,
        unit="mm",
        compute_face_normals=False,
    )
    receipt = dict(mesh.source_admission_receipt or {})
    assert receipt
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path=str(source),
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="test",
        operator="mesh-admission-test",
        created_at="2026-07-18T00:00:00Z",
        document_id="artifact:mesh-admission-test",
        metadata_revision_id="metadata:mesh-admission-test",
        align_revision_id="align:mesh-admission-test",
    )
    project = tmp_path / "receipt-round-trip.amr"

    save_artifact_session_project(project, session)
    restored = load_artifact_session_project(project)
    durable_receipt = restored.document.geometry_revisions[0].qc["import_admission"]

    assert durable_receipt == receipt
    assert restored.source_mesh.source_admission_receipt == receipt
    require_mesh_matches_admission_receipt(
        durable_receipt,
        restored.source_mesh.vertices,
        restored.source_mesh.faces,
    )


def test_receipt_validator_rejects_forged_decoded_vertex_count(
    tmp_path: Path,
) -> None:
    source = tmp_path / "artifact.ply"
    source.write_bytes(TRIANGLE_PLY)
    mesh = MeshLoader(default_unit="mm").load(
        source,
        unit="mm",
        compute_face_normals=False,
    )
    forged = json.loads(json.dumps(mesh.source_admission_receipt))
    forged["decoded"]["vertex_count"] = 4
    forged["accepted"]["vertex_count"] = 4
    forged["decoded"]["estimated_peak_bytes"] = (
        3 * forged["decoded"]["array_bytes"]
        + 32 * forged["decoded"]["vertex_count"]
        + 48 * forged["decoded"]["triangle_count"]
    )
    with pytest.raises(MeshAdmissionError, match="declared seam-split bound"):
        validate_mesh_admission_receipt(forged)


def test_session_rejects_receipt_that_underreports_decoded_arrays(
    tmp_path: Path,
) -> None:
    source = tmp_path / "artifact.ply"
    source.write_bytes(TRIANGLE_PLY)
    mesh = MeshLoader(default_unit="mm").load(
        source,
        unit="mm",
        compute_face_normals=False,
    )
    forged = json.loads(json.dumps(mesh.source_admission_receipt))
    forged["decoded"]["array_bytes"] = 1
    forged["decoded"]["estimated_peak_bytes"] = (
        3
        + 32 * forged["decoded"]["vertex_count"]
        + 48 * forged["decoded"]["triangle_count"]
    )
    mesh.source_admission_receipt = validate_mesh_admission_receipt(forged)

    with pytest.raises(ArtifactSessionError, match="arrays exceed"):
        ArtifactSession.create_from_source(
            mesh,
            resolved_source_path=str(source),
            unit="mm",
            axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
            handedness="right",
            software_version="test",
            operator="mesh-admission-test",
            created_at="2026-07-18T00:00:00Z",
        )


def test_receipt_rejects_tampered_peak_estimate() -> None:
    vertices, faces = _triangle_arrays()
    receipt = mesh_admission_receipt_for_arrays(
        vertices,
        faces,
        source_format="ply",
        source_size_bytes=len(TRIANGLE_PLY),
    )
    receipt["decoded"]["estimated_peak_bytes"] += 1

    with pytest.raises(MeshAdmissionError, match="admission formula"):
        validate_mesh_admission_receipt(receipt)
