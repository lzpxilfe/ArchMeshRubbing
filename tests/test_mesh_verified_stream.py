from __future__ import annotations

from io import BytesIO
import hashlib
from pathlib import Path
from typing import BinaryIO, cast
from unittest.mock import patch

import numpy as np
import pytest

from src.core.mesh_loader import MeshLoader
from src.core.mesh_import_recipe import current_mesh_import_recipe


PLY_BYTES = b"""ply
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


class _ChunkTrackingStream(BytesIO):
    def __init__(self, payload: bytes, *, max_return_size: int) -> None:
        super().__init__(payload)
        self.max_return_size = max_return_size
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0:
            raise AssertionError("whole-stream reads are forbidden")
        return super().read(min(size, self.max_return_size))


class _EndlessStream:
    def __init__(self) -> None:
        self.read_calls = 0

    def read(self, size: int = -1) -> bytes:
        assert 0 < size <= 1024 * 1024
        self.read_calls += 1
        if self.read_calls > 3:
            raise AssertionError("oversized stream was read after crossing the expected bound")
        return b"x"


def _load(payload: bytes = PLY_BYTES, *, original_name: str = "scan.ply"):
    return MeshLoader(default_unit="mm").load_verified_stream(
        BytesIO(payload),
        unit="mm",
        source_format="ply",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
        original_name=original_name,
    )


def test_load_verified_stream_loads_ply_and_records_observed_identity() -> None:
    mesh = _load()

    np.testing.assert_allclose(
        mesh.vertices,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    np.testing.assert_array_equal(mesh.faces, np.array([[0, 1, 2]], dtype=np.int32))
    assert mesh.face_normals is not None
    np.testing.assert_allclose(mesh.face_normals, np.array([[0.0, 0.0, 1.0]]))
    assert mesh.normals is None
    assert mesh.unit == "mm"
    assert mesh.filepath == Path("scan.ply")
    assert mesh.source_format == "ply"
    assert mesh.source_import_recipe == current_mesh_import_recipe("ply")
    assert mesh.source_identity is not None
    assert mesh.source_identity.sha256 == hashlib.sha256(PLY_BYTES).hexdigest()
    assert mesh.source_identity.size_bytes == len(PLY_BYTES)
    assert mesh.source_identity.mtime_ns == 0
    assert mesh.source_identity.original_name == "scan.ply"
    assert mesh.source_identity.format == "ply"


@pytest.mark.parametrize(
    ("expected_sha256", "expected_size_bytes", "message"),
    [
        ("0" * 64, len(PLY_BYTES), "SHA-256 mismatch"),
        (hashlib.sha256(PLY_BYTES).hexdigest(), len(PLY_BYTES) + 1, "size mismatch"),
    ],
)
def test_load_verified_stream_rejects_mismatch_before_parsing(
    expected_sha256: str,
    expected_size_bytes: int,
    message: str,
) -> None:
    with patch("src.core.mesh_loader.trimesh.load") as trimesh_load:
        with pytest.raises(ValueError, match=message):
            MeshLoader().load_verified_stream(
                BytesIO(PLY_BYTES),
                unit="mm",
                source_format="ply",
                expected_sha256=expected_sha256,
                expected_size_bytes=expected_size_bytes,
                original_name="scan.ply",
            )

    trimesh_load.assert_not_called()


def test_load_verified_stream_stops_when_input_crosses_expected_size() -> None:
    source = _EndlessStream()

    with patch("src.core.mesh_loader.trimesh.load") as trimesh_load:
        with pytest.raises(ValueError, match="observed at least 3"):
            MeshLoader().load_verified_stream(
                cast(BinaryIO, source),
                unit="mm",
                source_format="ply",
                expected_sha256=hashlib.sha256(b"xx").hexdigest(),
                expected_size_bytes=2,
                original_name="unbounded.ply",
            )

    assert source.read_calls == 3
    trimesh_load.assert_not_called()


def test_load_verified_stream_uses_bounded_reads_and_never_reads_whole_stream() -> None:
    source = _ChunkTrackingStream(PLY_BYTES, max_return_size=17)

    mesh = MeshLoader().load_verified_stream(
        source,
        unit="mm",
        source_format="ply",
        expected_sha256=hashlib.sha256(PLY_BYTES).hexdigest(),
        expected_size_bytes=len(PLY_BYTES),
        original_name="chunked.ply",
    )

    assert mesh.n_faces == 1
    assert len(source.read_sizes) > 2
    assert all(0 < size <= 1024 * 1024 for size in source.read_sizes)


def test_load_verified_stream_rejects_unsupported_format_before_reading() -> None:
    source = _ChunkTrackingStream(PLY_BYTES, max_return_size=17)

    with pytest.raises(ValueError, match="Unsupported format"):
        MeshLoader().load_verified_stream(
            source,
            unit="mm",
            source_format="xyz",
            expected_sha256=hashlib.sha256(PLY_BYTES).hexdigest(),
            expected_size_bytes=len(PLY_BYTES),
            original_name="scan.xyz",
        )

    assert source.read_sizes == []


def test_identical_verified_bytes_have_path_independent_geometry() -> None:
    first = _load(original_name="first/location/scan.ply")
    relocated = _load(original_name="renamed-copy.data")

    np.testing.assert_array_equal(first.vertices, relocated.vertices)
    np.testing.assert_array_equal(first.faces, relocated.faces)
    np.testing.assert_array_equal(first.face_normals, relocated.face_normals)
    assert first.filepath == Path("first/location/scan.ply")
    assert relocated.filepath == Path("renamed-copy.data")
    assert first.source_identity is not None
    assert relocated.source_identity is not None
    assert first.source_identity.content_matches(relocated.source_identity)
    assert first.source_identity.original_name != relocated.source_identity.original_name
