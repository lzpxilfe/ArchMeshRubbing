from __future__ import annotations

from io import BytesIO
import hashlib
import json
import struct

import numpy as np
import pytest

from src.core.mesh_loader import ExternalMeshDependencyError, MeshLoader


OBJ_WITH_MATERIAL = b"""mtllib material.mtl
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
usemtl painted
f 1/1 2/2 3/3
"""

PLY_WITH_TEXTURE = b"""ply
format ascii 1.0
comment TextureFile texture.png
element vertex 3
property float x
property float y
property float z
property float s
property float t
element face 1
property list uchar int vertex_indices
end_header
0 0 0 0 0
1 0 0 1 0
0 1 0 0 1
3 0 1 2
"""


def _triangle_glb(
    *,
    external_image_uri: str | None = None,
    external_buffer_uri: str | None = None,
) -> bytes:
    document: dict[str, object] = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 42}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": 36,
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": 36,
                "byteLength": 6,
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "max": [1, 1, 0],
                "min": [0, 0, 0],
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR",
                "max": [2],
                "min": [0],
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
    if external_buffer_uri is not None:
        buffers = document["buffers"]
        assert isinstance(buffers, list)
        buffers[0]["uri"] = external_buffer_uri
    if external_image_uri is not None:
        document["images"] = [{"uri": external_image_uri}]
        document["textures"] = [{"source": 0}]
        document["materials"] = [
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
        ]
        meshes = document["meshes"]
        assert isinstance(meshes, list)
        meshes[0]["primitives"][0]["material"] = 0

    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    binary_chunk = struct.pack(
        "<9f3H",
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0,
        1,
        2,
    )
    binary_chunk += b"\0" * (-len(binary_chunk) % 4)

    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total_length),
            struct.pack("<I4s", len(json_chunk), b"JSON"),
            json_chunk,
            struct.pack("<I4s", len(binary_chunk), b"BIN\0"),
            binary_chunk,
        )
    )


def _load_verified(payload: bytes, source_format: str):
    return MeshLoader().load_verified_stream(
        BytesIO(payload),
        unit="mm",
        source_format=source_format,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
        original_name=f"scan.{source_format}",
    )


@pytest.mark.parametrize(
    ("payload", "source_format"),
    [
        pytest.param(OBJ_WITH_MATERIAL, "obj", id="obj-mtllib"),
        pytest.param(PLY_WITH_TEXTURE, "ply", id="ply-texture-file"),
        pytest.param(
            _triangle_glb(external_image_uri="texture.png"),
            "glb",
            id="glb-external-image",
        ),
        pytest.param(
            _triangle_glb(external_buffer_uri="geometry.bin"),
            "glb",
            id="glb-external-buffer",
        ),
    ],
)
def test_authoritative_import_denies_external_mesh_dependencies(
    payload: bytes,
    source_format: str,
) -> None:
    with pytest.raises(
        ExternalMeshDependencyError,
        match="dependency_policy=deny_external",
    ):
        _load_verified(payload, source_format)


def test_authoritative_import_loads_self_contained_glb() -> None:
    mesh = _load_verified(_triangle_glb(), "glb")

    np.testing.assert_allclose(
        mesh.vertices,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    np.testing.assert_array_equal(mesh.faces, np.array([[0, 1, 2]], dtype=np.int32))
    assert mesh.source_format == "glb"
    assert mesh.source_import_recipe is not None
    assert mesh.source_import_recipe["dependency_policy"] == "deny_external"


def test_file_info_uses_the_same_external_dependency_gate(tmp_path) -> None:
    source_path = tmp_path / "scan.obj"
    source_path.write_bytes(OBJ_WITH_MATERIAL)
    (tmp_path / "material.mtl").write_text(
        "newmtl painted\nmap_Kd private-texture.png\n",
        encoding="utf-8",
    )

    info = MeshLoader().get_file_info(source_path)

    assert "dependency_policy=deny_external" in str(info["error"])
    assert "n_vertices" not in info
