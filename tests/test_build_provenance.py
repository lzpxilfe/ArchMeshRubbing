from __future__ import annotations

from collections.abc import Callable
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

import pytest

from src.build_provenance import (
    BuildProvenanceError,
    generate_build_provenance,
    github_actions_invocation,
    verify_build_provenance,
)
from src.core.canonical_json import canonical_json_bytes
from src.portable_archive import build_portable_archive, extract_portable_archive
from src.release_evidence import generate_release_evidence
from src.source_archive import build_source_archive


ROOT = Path(__file__).resolve().parents[1]


def _remove_read_only(
    function: Callable[..., object],
    path: str,
    _error: BaseException,
) -> None:
    os.chmod(path, os.stat(path, follow_symlinks=False).st_mode | stat.S_IWRITE)
    function(path)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_repository(root: Path) -> tuple[Path, str, int]:
    repository = root / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Build Provenance Test")
    _git(repository, "config", "user.email", "provenance@example.invalid")
    (repository / "LICENSE").write_text("GPL source license\n", encoding="utf-8")
    (repository / "main.py").write_text('print("source")\n', encoding="utf-8")
    _git(repository, "add", "--all")
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
    commit = _git(repository, "rev-parse", "HEAD")
    epoch = int(_git(repository, "show", "-s", "--format=%ct", commit), 10)
    return repository, commit, epoch


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _payload(root: Path, repository: Path, commit: str) -> Path:
    payload = root / "payload"
    project_license = (repository / "LICENSE").read_bytes()
    _write(payload / "_internal" / "LICENSE", project_license)
    runtime_lock = _write(
        payload / "_internal" / "requirements" / "runtime-py312.lock",
        b"Alpha==1.0\n",
    )
    wheel_lock = _write(
        payload
        / "_internal"
        / "requirements"
        / "windows-py312-x64-hashed.lock",
        (
            "--require-hashes\n"
            "--only-binary=:all:\n"
            f"Alpha==1.0 --hash=sha256:{'1' * 64}\n"
        ).encode("utf-8"),
    )
    policy = canonical_json_bytes({"packages": {}, "schema_version": "1.0.0"})
    _write(
        payload / "_internal" / "requirements" / "runtime-license-policy.json",
        policy,
    )
    public_policy = canonical_json_bytes(
        {
            "artifact_scope": "windows-x64-pyinstaller-portable",
            "decision": "blocked",
            "legal_advice": False,
            "project_license": {
                "expression": "GPL-2.0-only",
                "path": "LICENSE",
                "sha256": hashlib.sha256(project_license).hexdigest(),
            },
            "reason_code": "synthetic-license-review-pending",
            "reviewed_on": "2026-07-01",
            "rights_holder_authorization": "not-recorded",
            "runtime_license_observations": [
                {
                    "canonical_name": "alpha",
                    "evidence_field": "License-Expression",
                    "license_expression": "MIT",
                    "version": "1.0",
                }
            ],
            "schema_version": "1.0.0",
            "sources": [
                {
                    "label": "Synthetic source",
                    "url": "https://example.invalid/license-policy",
                }
            ],
        }
    )
    _write(
        payload / "_internal" / "requirements" / "public-release-policy.json",
        public_policy,
    )
    build_info = {
        "channel": "ci-smoke",
        "commit": commit,
        "dependency_lock_sha256": hashlib.sha256(runtime_lock.read_bytes()).hexdigest(),
        "schema_version": "1.2.0",
        "source_tree": "clean",
        "version": "0.1.0",
        "windows_wheel_lock_sha256": hashlib.sha256(
            wheel_lock.read_bytes()
        ).hexdigest(),
    }
    _write(
        payload / "_internal" / "resources" / "build_info.json",
        canonical_json_bytes(build_info),
    )
    _write(payload / "ArchMeshRubbing.exe", b"synthetic executable\n")
    _write(
        payload / "_internal" / "alpha-1.0.dist-info" / "METADATA",
        (
            "Metadata-Version: 2.4\n"
            "Name: Alpha\n"
            "Version: 1.0\n"
            "License-Expression: MIT\n"
            "License-File: LICENSE\n\n"
        ).encode("utf-8"),
    )
    _write(
        payload / "_internal" / "alpha-1.0.dist-info" / "licenses" / "LICENSE",
        b"Alpha license text\n",
    )
    source = payload / "source"
    build_source_archive(
        repository,
        source / "ArchMeshRubbing-source.zip",
        source / "ArchMeshRubbing-source.json",
        commit=commit,
    )
    generate_release_evidence(
        payload,
        payload / "release-evidence",
        created_at="2026-07-01T00:00:00Z",
    )
    return payload


def _artifact_chain(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, str, int, Path]:
    repository, commit, epoch = _source_repository(tmp_path)
    payload = _payload(tmp_path, repository, commit)
    output = tmp_path / "output"
    output.mkdir()
    archive = output / "ArchMeshRubbing-0.1.0-Windows-x64-portable.zip"
    manifest = output / f"{archive.name}.manifest.json"
    build_portable_archive(
        payload,
        archive,
        manifest,
        source_date_epoch=epoch,
    )
    provenance = output / f"{archive.name}.provenance.json"
    return payload, archive, manifest, provenance, commit, epoch, repository


def _environment(commit: str) -> dict[str, str]:
    return {
        "GITHUB_ACTIONS": "true",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_JOB": "portable-package-smoke",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_REF_PROTECTED": "true",
        "GITHUB_REPOSITORY": "lzpxilfe/ArchMeshRubbing",
        "GITHUB_REPOSITORY_ID": "123456789",
        "GITHUB_REPOSITORY_OWNER_ID": "9876543",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "29270074037",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_SHA": commit,
        "GITHUB_WORKFLOW_REF": (
            "lzpxilfe/ArchMeshRubbing/.github/workflows/"
            "package-smoke.yml@refs/heads/main"
        ),
        "GITHUB_WORKFLOW_SHA": commit,
        "RUNNER_ARCH": "X64",
        "RUNNER_ENVIRONMENT": "github-hosted",
        "RUNNER_NAME": "GitHub Actions 1000000000",
        "RUNNER_OS": "Windows",
    }


def test_build_provenance_is_deterministic_schema_valid_and_offline(
    tmp_path: Path,
) -> None:
    payload, archive, manifest, provenance, commit, _epoch, repository = (
        _artifact_chain(tmp_path)
    )
    invocation = github_actions_invocation(_environment(commit))
    first = generate_build_provenance(
        archive,
        manifest,
        payload,
        provenance,
        invocation=invocation,
    )
    second_path = provenance.with_name("second.provenance.json")
    second = generate_build_provenance(
        archive,
        manifest,
        payload,
        second_path,
        invocation=invocation,
    )

    assert first == second
    assert provenance.read_bytes() == second_path.read_bytes()
    assert first.source_commit == commit
    assert first.run_id == "29270074037"
    assert verify_build_provenance(provenance, archive, manifest, payload) == first
    value = json.loads(provenance.read_bytes())
    assert value["authentication"] == {"kind": "none", "signature_present": False}
    assert value["builder"]["repository_id"] == "123456789"

    jsonschema = importlib.import_module("jsonschema")
    schema = json.loads(
        (ROOT / "schemas" / "build_provenance-1.0.0.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(value)) == []

    with pytest.raises(BuildProvenanceError, match="refusing to overwrite"):
        generate_build_provenance(
            archive,
            manifest,
            payload,
            provenance,
            invocation=invocation,
        )

    shutil.rmtree(repository, onexc=_remove_read_only)
    assert verify_build_provenance(provenance, archive, manifest, payload) == first


def test_build_provenance_verifies_extracted_korean_path_and_detects_payload_drift(
    tmp_path: Path,
) -> None:
    payload, archive, manifest, provenance, commit, _epoch, _repository = (
        _artifact_chain(tmp_path)
    )
    result = generate_build_provenance(
        archive,
        manifest,
        payload,
        provenance,
        invocation=github_actions_invocation(_environment(commit)),
    )
    extracted = tmp_path / "문화유산 기록" / "ArchMeshRubbing"
    extract_portable_archive(archive, manifest, extracted)
    assert verify_build_provenance(provenance, archive, manifest, extracted) == result

    (extracted / "ArchMeshRubbing.exe").write_bytes(b"tampered executable\n")
    with pytest.raises(BuildProvenanceError, match="failed verification"):
        verify_build_provenance(provenance, archive, manifest, extracted)


def test_build_provenance_rejects_tampered_builder_and_mismatched_invocation(
    tmp_path: Path,
) -> None:
    payload, archive, manifest, provenance, commit, _epoch, _repository = (
        _artifact_chain(tmp_path)
    )
    generate_build_provenance(
        archive,
        manifest,
        payload,
        provenance,
        invocation=github_actions_invocation(_environment(commit)),
    )
    original = json.loads(provenance.read_bytes())
    value = json.loads(provenance.read_bytes())
    value["builder"]["run_url"] = "https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/1"
    provenance.write_bytes(canonical_json_bytes(value))
    with pytest.raises(BuildProvenanceError, match="run URL"):
        verify_build_provenance(provenance, archive, manifest, payload)

    false_unsigned = provenance.with_name("false-unsigned.provenance.json")
    original["authentication"]["signature_present"] = 0
    false_unsigned.write_bytes(canonical_json_bytes(original))
    with pytest.raises(BuildProvenanceError, match="unsigned status"):
        verify_build_provenance(false_unsigned, archive, manifest, payload)

    different = _environment("b" * 40)
    with pytest.raises(BuildProvenanceError, match="does not match artifact source"):
        generate_build_provenance(
            archive,
            manifest,
            payload,
            provenance.with_name("mismatched.provenance.json"),
            invocation=github_actions_invocation(different),
        )


def test_github_actions_invocation_rejects_wrong_runner_and_missing_identity() -> None:
    environment = _environment("a" * 40)
    environment["RUNNER_ENVIRONMENT"] = "self-hosted"
    with pytest.raises(BuildProvenanceError, match="builder identity"):
        github_actions_invocation(environment)

    environment = _environment("a" * 40)
    del environment["GITHUB_REPOSITORY_ID"]
    with pytest.raises(BuildProvenanceError, match="GITHUB_REPOSITORY_ID"):
        github_actions_invocation(environment)
