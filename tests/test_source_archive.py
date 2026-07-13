from __future__ import annotations

import base64
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile

import pytest

from src.core.canonical_json import canonical_json_bytes
from src.source_archive import (
    SOURCE_ARCHIVE_INTERNAL_MANIFEST,
    SourceArchiveError,
    build_source_archive,
    verify_source_archive,
)


ROOT = Path(__file__).resolve().parents[1]


def _git(
    repository: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        input=input_bytes,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Source Archive Test")
    _git(repository, "config", "user.email", "source-archive@example.invalid")
    (repository / "LICENSE").write_text("GPL source license\n", encoding="utf-8")
    (repository / "src").mkdir()
    (repository / "src" / "main.py").write_text(
        'print("committed source")\n', encoding="utf-8"
    )
    (repository / "tools").mkdir()
    script = repository / "tools" / "run.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (repository / "문서").mkdir()
    (repository / "문서" / "설명.txt").write_text("기록 도구\n", encoding="utf-8")
    _git(repository, "add", "--all")
    _git(repository, "update-index", "--chmod=+x", "tools/run.sh")
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_AUTHOR_DATE": "2026-07-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2026-07-01T00:00:00Z",
        }
    )
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "fixture"],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
    )
    commit = _git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    return repository, commit


def _paths(directory: Path) -> tuple[Path, Path]:
    return (
        directory / "ArchMeshRubbing-source.zip",
        directory / "ArchMeshRubbing-source.json",
    )


def test_source_archive_is_deterministic_commit_exact_and_offline(
    tmp_path: Path,
) -> None:
    repository, commit = _repository(tmp_path)
    first_archive, first_sidecar = _paths(tmp_path / "first")
    second_archive, second_sidecar = _paths(tmp_path / "second")

    first = build_source_archive(
        repository,
        first_archive,
        first_sidecar,
        commit=commit,
    )
    (repository / "src" / "main.py").write_text(
        'print("uncommitted and excluded")\n', encoding="utf-8"
    )
    second = build_source_archive(
        repository,
        second_archive,
        second_sidecar,
        commit=commit,
    )

    assert first == second
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert first_sidecar.read_bytes() == second_sidecar.read_bytes()
    assert first.file_count == 4
    assert first.source_commit == commit
    with ZipFile(first_archive) as archive:
        names = archive.namelist()
        assert names[0].endswith(f"/{SOURCE_ARCHIVE_INTERNAL_MANIFEST}")
        assert names[1:] == [
            f"{first.root_directory}/LICENSE",
            f"{first.root_directory}/src/main.py",
            f"{first.root_directory}/tools/run.sh",
            f"{first.root_directory}/문서/설명.txt",
        ]
        manifest = json.loads(archive.read(names[0]))
        assert manifest["source_commit"] == commit
        assert manifest["source_tree"] == (
            _git(repository, "rev-parse", f"{commit}^{{tree}}")
            .decode("ascii")
            .strip()
        )
        assert base64.b64decode(
            manifest["commit_object"]["payload"],
            validate=True,
        ) == _git(repository, "cat-file", "commit", commit)
        assert manifest["license"]["expression"] == "GPL-2.0-only"
        assert manifest["files"][2]["mode"] == "100755"
        assert archive.read(f"{first.root_directory}/src/main.py") == (
            b'print("committed source")\n'
        )

    shutil.rmtree(repository)
    assert verify_source_archive(first_archive, first_sidecar) == first


def test_source_archive_rejects_overwrite_and_every_independent_tamper(
    tmp_path: Path,
) -> None:
    repository, commit = _repository(tmp_path)
    archive_path, sidecar_path = _paths(tmp_path / "output")
    build_source_archive(repository, archive_path, sidecar_path, commit=commit)

    with pytest.raises(SourceArchiveError, match="refusing to overwrite"):
        build_source_archive(repository, archive_path, sidecar_path, commit=commit)

    sidecar = json.loads(sidecar_path.read_bytes())
    sidecar["source_commit"] = "0" * 40
    tampered_sidecar = tmp_path / "tampered.json"
    tampered_sidecar.write_bytes(canonical_json_bytes(sidecar))
    with pytest.raises(SourceArchiveError):
        verify_source_archive(archive_path, tampered_sidecar)

    damaged = bytearray(archive_path.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    archive_path.write_bytes(damaged)
    with pytest.raises(SourceArchiveError, match="hash or size"):
        verify_source_archive(archive_path, sidecar_path)


def test_source_archive_rejects_non_regular_git_tree_entries(tmp_path: Path) -> None:
    repository, _commit = _repository(tmp_path)
    blob = _git(repository, "hash-object", "-w", "--stdin", input_bytes=b"LICENSE")
    oid = blob.decode("ascii").strip()
    _git(repository, "update-index", "--add", "--cacheinfo", f"120000,{oid},link")
    _git(repository, "commit", "--quiet", "-m", "add symbolic link")
    commit = _git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    archive_path, sidecar_path = _paths(tmp_path / "unsupported")

    with pytest.raises(SourceArchiveError, match="not a regular tracked file"):
        build_source_archive(
            repository,
            archive_path,
            sidecar_path,
            commit=commit,
        )


def test_generated_source_contract_matches_public_json_schema(tmp_path: Path) -> None:
    jsonschema = importlib.import_module("jsonschema")
    schema = json.loads(
        (ROOT / "schemas" / "source_archive-1.0.0.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    manifest_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": schema["$defs"],
        "$ref": "#/$defs/sourceManifest",
    }
    jsonschema.Draft202012Validator.check_schema(manifest_schema)

    repository, commit = _repository(tmp_path)
    archive_path, sidecar_path = _paths(tmp_path / "schema")
    build_source_archive(repository, archive_path, sidecar_path, commit=commit)
    sidecar = json.loads(sidecar_path.read_bytes())
    with ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(archive.namelist()[0]))

    sidecar_validator = jsonschema.Draft202012Validator(schema)
    manifest_validator = jsonschema.Draft202012Validator(manifest_schema)
    assert list(sidecar_validator.iter_errors(sidecar)) == []
    assert list(manifest_validator.iter_errors(manifest)) == []

    manifest["files"][0]["mode"] = "120000"
    assert list(manifest_validator.iter_errors(manifest))
