from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import jsonschema
import pytest

from src.portable_archive import (
    PORTABLE_ARCHIVE_COMMENT,
    PORTABLE_ARCHIVE_ROOT,
    PortableArchiveError,
    build_portable_archive,
    extract_portable_archive,
    verify_portable_archive,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATE_EPOCH = 1_752_364_800
SOURCE_COMMIT = "a" * 40


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "ArchMeshRubbing.exe").write_bytes(b"frozen executable\n")
    internal = path / "_internal"
    internal.mkdir()
    (internal / "runtime.bin").write_bytes(bytes(range(256)) * 8)
    (internal / "한글 자료.txt").write_text("문화유산 기록\n", encoding="utf-8")
    evidence = path / "release-evidence"
    evidence.mkdir()
    (evidence / "release-evidence.json").write_bytes(
        _canonical_json(
            {
                "created": "2026-07-13T00:00:00Z",
                "evidence_files": [],
                "payload_sha256": "b" * 64,
                "schema_version": "1.0.0",
                "source_commit": SOURCE_COMMIT,
            }
        )
    )
    return path


def _build(payload: Path, output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True)
    archive = output / "ArchMeshRubbing-Windows-x64-portable.zip"
    manifest = output / "ArchMeshRubbing-Windows-x64-portable.zip.manifest.json"
    with patch("src.portable_archive.verify_release_evidence"):
        build_portable_archive(
            payload,
            archive,
            manifest,
            source_date_epoch=SOURCE_DATE_EPOCH,
        )
    return archive, manifest


def test_portable_archive_is_deterministic_schema_valid_and_extracts_atomically(
    tmp_path: Path,
) -> None:
    payload_a = _payload(tmp_path / "payload-a")
    payload_b = _payload(tmp_path / "payload-b")
    archive_a, manifest_a = _build(payload_a, tmp_path / "output-a")
    archive_b, manifest_b = _build(payload_b, tmp_path / "output-b")

    assert archive_a.read_bytes() == archive_b.read_bytes()
    assert manifest_a.read_bytes() == manifest_b.read_bytes()
    result = verify_portable_archive(archive_a, manifest_a)
    assert result.file_count == 4
    assert result.source_commit == SOURCE_COMMIT
    assert result.archive_sha256 == hashlib.sha256(archive_a.read_bytes()).hexdigest()

    manifest_value = json.loads(manifest_a.read_text(encoding="utf-8"))
    schema = json.loads(
        (ROOT / "schemas" / "portable_archive_manifest-1.0.0.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(manifest_value)
    with ZipFile(archive_a) as archive:
        assert archive.comment == PORTABLE_ARCHIVE_COMMENT
        assert archive.namelist() == [
            f"{PORTABLE_ARCHIVE_ROOT}/{entry['path']}"
            for entry in manifest_value["entries"]
        ]
        assert all(info.compress_type == ZIP_DEFLATED for info in archive.infolist())
        assert len({info.date_time for info in archive.infolist()}) == 1

    destination = tmp_path / "한글 경로" / "유물 기록 도구"
    extracted = extract_portable_archive(archive_a, manifest_a, destination)
    assert extracted == result
    assert (destination / "ArchMeshRubbing.exe").read_bytes() == b"frozen executable\n"
    assert (destination / "_internal" / "한글 자료.txt").read_text(
        encoding="utf-8"
    ) == "문화유산 기록\n"
    with pytest.raises(PortableArchiveError, match="refusing to reuse"):
        extract_portable_archive(archive_a, manifest_a, destination)


def test_portable_archive_refuses_overwrite_and_payload_outputs(tmp_path: Path) -> None:
    payload = _payload(tmp_path / "payload")
    output = tmp_path / "output"
    archive, manifest = _build(payload, output)
    with (
        patch("src.portable_archive.verify_release_evidence"),
        pytest.raises(PortableArchiveError, match="refusing to overwrite"),
    ):
        build_portable_archive(
            payload,
            archive,
            manifest,
            source_date_epoch=SOURCE_DATE_EPOCH,
        )

    with (
        patch("src.portable_archive.verify_release_evidence"),
        pytest.raises(PortableArchiveError, match="outside the payload root"),
    ):
        build_portable_archive(
            payload,
            payload / "archive.zip",
            output / "other-manifest.json",
            source_date_epoch=SOURCE_DATE_EPOCH,
        )


@pytest.mark.parametrize(
    "relative_paths",
    [
        ("A.txt", "a.TXT"),
        ("CON.txt",),
        ("trailing-dot.",),
    ],
)
def test_portable_archive_rejects_windows_unsafe_payload_names(
    tmp_path: Path,
    relative_paths: tuple[str, ...],
) -> None:
    payload = _payload(tmp_path / "payload")
    for relative in relative_paths:
        (payload / relative).write_bytes(b"unsafe\n")
    if len(relative_paths) == 2 and (payload / relative_paths[0]).samefile(
        payload / relative_paths[1]
    ):
        pytest.skip("test filesystem is case-insensitive")
    with (
        patch("src.portable_archive.verify_release_evidence"),
        pytest.raises(PortableArchiveError),
    ):
        build_portable_archive(
            payload,
            tmp_path / "archive.zip",
            tmp_path / "manifest.json",
            source_date_epoch=SOURCE_DATE_EPOCH,
        )


def test_portable_archive_rejects_symlink_payload(tmp_path: Path) -> None:
    payload = _payload(tmp_path / "payload")
    link = payload / "linked-runtime.bin"
    try:
        link.symlink_to(payload / "_internal" / "runtime.bin")
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are unavailable")
    with (
        patch("src.portable_archive.verify_release_evidence"),
        pytest.raises(PortableArchiveError, match="symbolic link"),
    ):
        build_portable_archive(
            payload,
            tmp_path / "archive.zip",
            tmp_path / "manifest.json",
            source_date_epoch=SOURCE_DATE_EPOCH,
        )


def test_portable_archive_detects_archive_and_manifest_tampering(tmp_path: Path) -> None:
    payload = _payload(tmp_path / "payload")
    archive, manifest = _build(payload, tmp_path / "output")

    archive_bytes = bytearray(archive.read_bytes())
    archive_bytes[len(archive_bytes) // 2] ^= 0x01
    archive.write_bytes(archive_bytes)
    with pytest.raises(PortableArchiveError, match="hash or size"):
        verify_portable_archive(archive, manifest)

    archive, manifest = _build(payload, tmp_path / "second-output")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["entries"][0]["path"] = "../escape.exe"
    manifest.write_bytes(_canonical_json(value))
    with pytest.raises(PortableArchiveError, match="portable manifest path"):
        verify_portable_archive(archive, manifest)


def test_portable_archive_requires_verified_release_evidence(tmp_path: Path) -> None:
    payload = _payload(tmp_path / "payload")
    with pytest.raises(PortableArchiveError, match="release evidence failed verification"):
        build_portable_archive(
            payload,
            tmp_path / "archive.zip",
            tmp_path / "manifest.json",
            source_date_epoch=SOURCE_DATE_EPOCH,
        )
    assert not (tmp_path / "archive.zip").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_windows_package_workflow_is_portable_offline_and_installer_independent() -> None:
    workflow = (ROOT / ".github" / "workflows" / "package-smoke.yml").read_text(
        encoding="utf-8"
    )
    assert "tools/build_portable_archive.py build" in workflow
    assert "tools/build_portable_archive.py verify" in workflow
    assert "tools/build_portable_archive.py extract" in workflow
    assert "문화유산 기록\\ArchMeshRubbing" in workflow
    assert "New-NetFirewallRule" in workflow
    assert "--self-test-report" in workflow
    assert "--opengl-driver-smoke-report" in workflow
    assert "WaitForExit(120000)" in workflow
    assert "actions/upload-artifact" not in workflow
    assert "ISCC" not in workflow
    assert "Inno Setup" not in workflow
    assert not (ROOT / "installer" / "ArchMeshRubbing.iss").exists()
