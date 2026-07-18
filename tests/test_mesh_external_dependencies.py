from __future__ import annotations

import base64
from contextlib import contextmanager
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import struct
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from src.core.mesh_admission import MeshAdmissionError
from src.core.mesh_loader import (
    ExternalMeshDependencyError,
    MeshLoader,
    _ClosedManifestResolver,
)
from src.core.source_manifest import (
    DEPENDENCY_RESOURCE_ROLE,
    PRIMARY_RESOURCE_ROLE,
    SourceManifest,
    SourceManifestEntry,
    SourceManifestError,
)
from src.core.source_identity import SourceFingerprint, open_fingerprinted_file


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

    chunks = [
        struct.pack("<I4s", len(json_chunk), b"JSON"),
        json_chunk,
    ]
    if external_buffer_uri is None:
        chunks.extend(
            (
                struct.pack("<I4s", len(binary_chunk), b"BIN\0"),
                binary_chunk,
            )
        )
    total_length = 12 + sum(len(chunk) for chunk in chunks)
    return b"".join((struct.pack("<4sII", b"glTF", 2, total_length), *chunks))


def _triangle_gltf_document(buffer_uri: str) -> tuple[dict[str, object], bytes]:
    binary = struct.pack(
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
    document: dict[str, object] = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary), "uri": buffer_uri}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963},
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
    return document, binary


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


def test_path_import_captures_obj_material_and_texture_closure(tmp_path) -> None:
    source_path = tmp_path / "scan.obj"
    source_path.write_bytes(OBJ_WITH_MATERIAL)
    (tmp_path / "material.mtl").write_text(
        "newmtl painted\nmap_Kd texture.png\n",
        encoding="utf-8",
    )
    Image.new("RGB", (2, 2), color=(12, 34, 56)).save(tmp_path / "texture.png")

    mesh = MeshLoader().load(source_path)

    assert mesh.source_import_recipe is not None
    assert mesh.source_import_recipe["recipe_version"] == "2.0.0"
    assert mesh.source_import_recipe["dependency_policy"] == "closed_manifest"
    manifest = mesh.source_import_recipe["source_manifest"]
    assert isinstance(manifest, dict)
    assert [entry["logical_path"] for entry in manifest["entries"]] == [
        "material.mtl",
        "scan.obj",
        "texture.png",
    ]
    assert [resource.entry.logical_path for resource in mesh.source_resources] == [
        "material.mtl",
        "scan.obj",
        "texture.png",
    ]
    assert mesh.uv_coords is not None
    assert mesh.texture is not None


def test_closed_path_replay_rejects_changed_dependency(tmp_path) -> None:
    source_path = tmp_path / "scan.obj"
    source_path.write_bytes(OBJ_WITH_MATERIAL)
    material_path = tmp_path / "material.mtl"
    material_path.write_text("newmtl painted\nmap_Kd texture.png\n", encoding="utf-8")
    Image.new("RGB", (2, 2), color=(12, 34, 56)).save(tmp_path / "texture.png")
    first = MeshLoader().load(source_path)
    assert first.source_import_recipe is not None

    material_path.write_text("newmtl painted\nKd 1 0 0\n", encoding="utf-8")

    with pytest.raises(
        ExternalMeshDependencyError,
        match="closed source manifest",
    ):
        MeshLoader().load(
            source_path,
            import_recipe=first.source_import_recipe,
            capture_dependencies=False,
        )


def test_path_import_captures_ply_texture_file(tmp_path) -> None:
    source_path = tmp_path / "scan.ply"
    source_path.write_bytes(PLY_WITH_TEXTURE)
    Image.new("RGB", (2, 2), color=(90, 80, 70)).save(tmp_path / "texture.png")

    mesh = MeshLoader().load(source_path)

    assert mesh.source_import_recipe is not None
    assert mesh.source_import_recipe["dependency_policy"] == "closed_manifest"
    manifest = mesh.source_import_recipe["source_manifest"]
    assert isinstance(manifest, dict)
    assert [entry["logical_path"] for entry in manifest["entries"]] == [
        "scan.ply",
        "texture.png",
    ]
    assert mesh.texture is not None


def test_path_import_captures_gltf_external_buffer(tmp_path) -> None:
    document, binary = _triangle_gltf_document("geometry.bin")
    source_path = tmp_path / "scan.gltf"
    source_path.write_text(json.dumps(document), encoding="utf-8")
    (tmp_path / "geometry.bin").write_bytes(binary)

    mesh = MeshLoader().load(source_path)

    np.testing.assert_allclose(
        mesh.vertices,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert mesh.source_import_recipe is not None
    assert mesh.source_import_recipe["dependency_policy"] == "closed_manifest"
    assert [resource.entry.logical_path for resource in mesh.source_resources] == [
        "geometry.bin",
        "scan.gltf",
    ]


@pytest.mark.parametrize("sidecar_delta", (-1, 1))
def test_path_import_rejects_external_buffer_length_before_parser(
    tmp_path,
    sidecar_delta: int,
) -> None:
    document, binary = _triangle_gltf_document("geometry.bin")
    source_path = tmp_path / "scan.gltf"
    source_path.write_text(json.dumps(document), encoding="utf-8")
    sidecar = binary[:-1] if sidecar_delta < 0 else binary + b"\0"
    (tmp_path / "geometry.bin").write_bytes(sidecar)

    with mock.patch(
        "src.core.mesh_loader._load_authoritative_trimesh"
    ) as parser:
        with pytest.raises(
            ExternalMeshDependencyError,
            match="do not match",
        ):
            MeshLoader().load(source_path)
    parser.assert_not_called()


def test_path_import_rejects_same_length_dependency_rewrite_before_parser(
    tmp_path,
) -> None:
    document, original = _triangle_gltf_document("geometry.bin")
    source_path = tmp_path / "scan.gltf"
    source_path.write_text(json.dumps(document), encoding="utf-8")
    dependency_path = tmp_path / "geometry.bin"
    dependency_path.write_bytes(original)
    rewritten = bytes([original[0] ^ 0xFF]) + original[1:]
    assert len(rewritten) == len(original)

    @contextmanager
    def race_after_fingerprint(path, **kwargs):
        requested_path = Path(path).resolve()
        if requested_path != dependency_path.resolve():
            with open_fingerprinted_file(path, **kwargs) as opened:
                yield opened
            return

        with dependency_path.open("rb") as descriptor:
            stable_stat = os.fstat(descriptor.fileno())
            fingerprint = SourceFingerprint(
                sha256=hashlib.sha256(original).hexdigest(),
                size_bytes=len(original),
                mtime_ns=stable_stat.st_mtime_ns,
                original_name=dependency_path.name,
                format="bin",
            )

            class RewrittenReadStream:
                def fileno(self) -> int:
                    return descriptor.fileno()

                def read(self, size: int = -1) -> bytes:
                    if size < 0:
                        return rewritten
                    return rewritten[:size]

            yield RewrittenReadStream(), fingerprint

    with mock.patch(
        "src.core.mesh_loader.open_fingerprinted_file",
        side_effect=race_after_fingerprint,
    ), mock.patch(
        "src.core.mesh_loader._load_authoritative_trimesh"
    ) as parser:
        with pytest.raises(
            ExternalMeshDependencyError,
            match="external glTF buffer bytes do not match",
        ):
            MeshLoader().load(source_path)
    parser.assert_not_called()


def test_path_import_keeps_data_uri_gltf_self_contained(tmp_path) -> None:
    _placeholder, binary = _triangle_gltf_document("unused.bin")
    uri = "data:application/octet-stream;base64," + base64.b64encode(
        binary
    ).decode("ascii")
    document, _binary = _triangle_gltf_document(uri)
    source_path = tmp_path / "scan.gltf"
    source_path.write_text(json.dumps(document), encoding="utf-8")

    mesh = MeshLoader().load(source_path)

    assert mesh.source_import_recipe is not None
    assert mesh.source_import_recipe["dependency_policy"] == "deny_external"
    assert [resource.entry.logical_path for resource in mesh.source_resources] == [
        "scan.gltf"
    ]


def test_path_import_rejects_data_uri_length_before_parser(tmp_path) -> None:
    _placeholder, binary = _triangle_gltf_document("unused.bin")
    uri = "data:application/octet-stream;base64," + base64.b64encode(
        binary
    ).decode("ascii")
    document, _binary = _triangle_gltf_document(uri)
    buffers = document["buffers"]
    assert isinstance(buffers, list)
    buffers[0]["byteLength"] = len(binary) + 1
    source_path = tmp_path / "scan.gltf"
    source_path.write_text(json.dumps(document), encoding="utf-8")

    with mock.patch(
        "src.core.mesh_loader._load_authoritative_trimesh"
    ) as parser:
        with pytest.raises(MeshAdmissionError, match="data URI length"):
            MeshLoader().load(source_path)
    parser.assert_not_called()


def test_path_import_rejects_parser_ambiguous_base64_uri_before_parser(
    tmp_path,
) -> None:
    _placeholder, binary = _triangle_gltf_document("unused.bin")
    uri = "geometry.bin;base64," + base64.b64encode(binary).decode("ascii")
    document, _binary = _triangle_gltf_document(uri)
    source_path = tmp_path / "scan.gltf"
    source_path.write_text(json.dumps(document), encoding="utf-8")
    # This is a valid Windows filename.  Without the admission rejection the
    # manifest would bind these bytes, while the locked parser treats the URI
    # suffix as embedded base64 and decodes ``binary`` instead.
    (tmp_path / uri).write_bytes(b"\0" * len(binary))

    with mock.patch(
        "src.core.mesh_loader._load_authoritative_trimesh"
    ) as parser:
        with pytest.raises(MeshAdmissionError, match="ambiguous non-canonical"):
            MeshLoader().load(source_path)
    parser.assert_not_called()


@pytest.mark.parametrize(
    "uri",
    (
        "DATA:application/octet-stream;base64,AAAA",
        "data:application/octet-stream;BASE64,AAAA",
    ),
)
def test_path_import_rejects_noncanonical_data_uri_before_parser(
    tmp_path,
    uri: str,
) -> None:
    document, _binary = _triangle_gltf_document(uri)
    source_path = tmp_path / "scan.gltf"
    source_path.write_text(json.dumps(document), encoding="utf-8")

    with mock.patch(
        "src.core.mesh_loader._load_authoritative_trimesh"
    ) as parser:
        with pytest.raises(MeshAdmissionError, match="(ambiguous|malformed)"):
            MeshLoader().load(source_path)
    parser.assert_not_called()


def test_path_import_rejects_remote_gltf_uri(tmp_path) -> None:
    document, _binary = _triangle_gltf_document(
        "https://example.invalid/private.bin"
    )
    source_path = tmp_path / "scan.gltf"
    source_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        ExternalMeshDependencyError,
        match="relative-contained-v1",
    ):
        MeshLoader().load(source_path)


def test_path_import_captures_glb_external_buffer(tmp_path) -> None:
    _document, binary = _triangle_gltf_document("geometry.bin")
    source_path = tmp_path / "scan.glb"
    source_path.write_bytes(_triangle_glb(external_buffer_uri="geometry.bin"))
    (tmp_path / "geometry.bin").write_bytes(binary)

    mesh = MeshLoader().load(source_path)

    assert mesh.source_import_recipe is not None
    assert mesh.source_import_recipe["dependency_policy"] == "closed_manifest"
    assert [resource.entry.logical_path for resource in mesh.source_resources] == [
        "geometry.bin",
        "scan.glb",
    ]


def test_path_import_captures_glb_external_image(tmp_path) -> None:
    source_path = tmp_path / "scan.glb"
    source_path.write_bytes(_triangle_glb(external_image_uri="texture.png"))
    Image.new("RGB", (2, 2), color=(21, 43, 65)).save(tmp_path / "texture.png")

    mesh = MeshLoader().load(source_path)

    assert mesh.source_import_recipe is not None
    assert mesh.source_import_recipe["dependency_policy"] == "closed_manifest"
    assert [resource.entry.logical_path for resource in mesh.source_resources] == [
        "scan.glb",
        "texture.png",
    ]


def test_contained_capture_rejects_parent_traversal_even_when_target_exists(
    tmp_path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "scan.obj"
    source_path.write_bytes(OBJ_WITH_MATERIAL.replace(b"material.mtl", b"../secret.mtl"))
    (tmp_path / "secret.mtl").write_text("newmtl painted\n", encoding="utf-8")

    with pytest.raises(
        ExternalMeshDependencyError,
        match="relative-contained-v1",
    ):
        MeshLoader().load(source_path)


def test_contained_capture_rejects_symlink_escape(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "scan.obj"
    source_path.write_bytes(OBJ_WITH_MATERIAL)
    outside = tmp_path / "material.mtl"
    outside.write_text("newmtl painted\n", encoding="utf-8")
    try:
        (source_dir / "material.mtl").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(
        ExternalMeshDependencyError,
        match="relative-contained-v1",
    ):
        MeshLoader().load(source_path)


def test_file_info_uses_contained_capture_and_reports_missing_sidecar(tmp_path) -> None:
    source_path = tmp_path / "scan.obj"
    source_path.write_bytes(OBJ_WITH_MATERIAL)
    (tmp_path / "material.mtl").write_text(
        "newmtl painted\nmap_Kd private-texture.png\n",
        encoding="utf-8",
    )

    info = MeshLoader().get_file_info(source_path)

    assert "resolver_profile=relative-contained-v1" in str(info["error"])
    assert "n_vertices" not in info


def test_closed_resolver_containment_probe_is_an_exact_dependency_request() -> None:
    dependency_payload = b"newmtl painted\n"
    manifest = SourceManifest(
        primary_logical_path="scan.obj",
        entries=(
            SourceManifestEntry(
                logical_path="scan.obj",
                media_type="model/obj",
                role=PRIMARY_RESOURCE_ROLE,
                sha256="a" * 64,
                size_bytes=10,
            ),
            SourceManifestEntry(
                logical_path="material.mtl",
                media_type="model/mtl",
                role=DEPENDENCY_RESOURCE_ROLE,
                sha256=hashlib.sha256(dependency_payload).hexdigest(),
                size_bytes=len(dependency_payload),
            ),
        ),
    )

    resolver = _ClosedManifestResolver(
        manifest,
        lambda _entry: (dependency_payload, "archive.amr!/material"),
    )
    assert "material.mtl" in resolver
    resolver.validate_after_load()
    assert [item.entry.logical_path for item in resolver.resources] == [
        "material.mtl"
    ]

    undeclared = _ClosedManifestResolver(
        manifest,
        lambda _entry: (dependency_payload, "archive.amr!/material"),
    )
    assert "optional.mtl" not in undeclared
    with pytest.raises(ExternalMeshDependencyError, match="exactly match"):
        undeclared.validate_after_load()

    def broken_loader(_entry: SourceManifestEntry) -> tuple[bytes, str]:
        raise RuntimeError("archive member became unavailable")

    broken = _ClosedManifestResolver(manifest, broken_loader)
    with pytest.raises(FileNotFoundError, match="became unavailable"):
        broken.get("material.mtl")
    with pytest.raises(ExternalMeshDependencyError, match="exactly match"):
        broken.validate_after_load()

    unsafe_namespace = _ClosedManifestResolver(
        manifest,
        lambda _entry: (dependency_payload, "archive.amr!/material"),
    )
    with pytest.raises(SourceManifestError, match="escapes"):
        unsafe_namespace.namespaced("../../outside")
    with pytest.raises(ExternalMeshDependencyError, match="exactly match"):
        unsafe_namespace.validate_after_load()
