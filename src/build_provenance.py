"""Canonical unsigned provenance for one verified Windows portable build.

The record binds the portable ZIP, its canonical manifest, the exact source
archive, release evidence, and the GitHub Actions invocation/runner identity.
It is deliberately marked unsigned: verification proves internal integrity
and cross-file consistency, not that GitHub or the project authenticated it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
import uuid

from src.core.canonical_json import CanonicalJSONError, canonical_json_bytes
from src.portable_archive import PortableArchiveError, verify_portable_archive
from src.release_evidence import (
    EVIDENCE_DIRECTORY_NAME,
    EVIDENCE_SCHEMA_VERSION,
    ReleaseEvidenceError,
    verify_release_evidence,
)
from src.source_archive import (
    SOURCE_ARCHIVE_DIRECTORY,
    SOURCE_ARCHIVE_FILENAME,
    SOURCE_ARCHIVE_SIDECAR_FILENAME,
    SourceArchiveError,
    verify_source_archive,
)


BUILD_PROVENANCE_FORMAT = "org.archmeshrubbing.build-provenance"
BUILD_PROVENANCE_SCHEMA_VERSION = "1.0.0"
BUILD_PROVENANCE_REPOSITORY = "lzpxilfe/ArchMeshRubbing"
BUILD_PROVENANCE_SERVER_URL = "https://github.com"
BUILD_PROVENANCE_WORKFLOW_PATH = ".github/workflows/package-smoke.yml"
BUILD_PROVENANCE_JOB = "portable-package-smoke"
BUILD_PROVENANCE_MAX_BYTES = 256 * 1024

_OBJECT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_POSITIVE_ID_RE = re.compile(r"^[1-9][0-9]{0,31}$")
_REF_RE = re.compile(r"^refs/(?:heads|tags|pull)/[^\x00-\x1f\x7f]{1,1000}$")
_SAFE_TEXT_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")
_EVENTS = frozenset({"pull_request", "push", "workflow_dispatch"})


class BuildProvenanceError(RuntimeError):
    """The provenance record or one of its bound artifacts is invalid."""


@dataclass(frozen=True, slots=True)
class BuildInvocation:
    """Validated source identity and builder claims captured for one run."""

    source_commit: str
    builder: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class BuildProvenanceResult:
    """Stable summary returned after complete offline verification."""

    provenance_sha256: str
    provenance_size: int
    portable_archive_sha256: str
    source_archive_sha256: str
    source_commit: str
    source_tree: str
    run_id: str
    run_attempt: int

    def detail(self) -> str:
        return (
            f"provenance={self.provenance_sha256}, "
            f"portable={self.portable_archive_sha256}, "
            f"source={self.source_archive_sha256}, commit={self.source_commit}, "
            f"tree={self.source_tree}, run={self.run_id}/{self.run_attempt}"
        )


def _canonical_bytes(value: object) -> bytes:
    try:
        return canonical_json_bytes(value)
    except CanonicalJSONError as exc:
        raise BuildProvenanceError("build provenance is not canonical JSON") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise BuildProvenanceError(f"could not hash bound file: {path}") from exc
    return digest.hexdigest(), size


def _exact_mapping(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BuildProvenanceError(f"{name} must be an object")
    observed = set(value)
    if observed != expected:
        raise BuildProvenanceError(
            f"{name} fields are invalid; missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )
    return value


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BuildProvenanceError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise BuildProvenanceError(f"{name} is outside the supported range")
    return value


def _object_id(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _OBJECT_RE.fullmatch(value) is None:
        raise BuildProvenanceError(f"{name} must be a full lowercase Git object ID")
    return value


def _positive_id(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _POSITIVE_ID_RE.fullmatch(value) is None:
        raise BuildProvenanceError(f"{name} must be a positive decimal identifier")
    return value


def _safe_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SAFE_TEXT_RE.fullmatch(value) is None:
        raise BuildProvenanceError(f"{name} contains unsupported text")
    return value


def _ref(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _REF_RE.fullmatch(value) is None:
        raise BuildProvenanceError(f"{name} is not a supported GitHub ref")
    return value


def _regular_file(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise BuildProvenanceError(f"{label} must not be a symbolic link")
    resolved = expanded.resolve(strict=False)
    if not resolved.is_file() or resolved.is_symlink():
        raise BuildProvenanceError(f"{label} must be a regular file")
    return resolved


def _payload_root(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise BuildProvenanceError("portable payload must not be a symbolic link")
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as exc:
        raise BuildProvenanceError("portable payload does not exist") from exc
    if not resolved.is_dir():
        raise BuildProvenanceError("portable payload must be a directory")
    return resolved


def _read_canonical_json(
    path: Path,
    *,
    label: str,
    limit: int = BUILD_PROVENANCE_MAX_BYTES,
) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BuildProvenanceError(f"could not read {label}") from exc
    if not raw or len(raw) > limit:
        raise BuildProvenanceError(f"{label} byte length is invalid")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BuildProvenanceError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict) or _canonical_bytes(value) != raw:
        raise BuildProvenanceError(f"{label} is not canonical RFC 8785 JSON")
    return value, raw


def _validate_builder(value: object) -> dict[str, object]:
    builder = _exact_mapping(
        value,
        {
            "checkout_sha",
            "event_name",
            "job",
            "provider",
            "ref",
            "ref_protected",
            "repository",
            "repository_id",
            "repository_owner_id",
            "run_attempt",
            "run_id",
            "run_url",
            "runner_arch",
            "runner_environment",
            "runner_name",
            "runner_os",
            "server_url",
            "workflow_ref",
            "workflow_sha",
        },
        name="build provenance builder",
    )
    if (
        builder["provider"] != "github-actions"
        or builder["server_url"] != BUILD_PROVENANCE_SERVER_URL
        or builder["repository"] != BUILD_PROVENANCE_REPOSITORY
        or builder["job"] != BUILD_PROVENANCE_JOB
        or builder["runner_environment"] != "github-hosted"
        or builder["runner_os"] != "Windows"
        or builder["runner_arch"] != "X64"
    ):
        raise BuildProvenanceError("build provenance builder identity is unsupported")
    checkout_sha = _object_id(builder["checkout_sha"], name="builder checkout SHA")
    workflow_sha = _object_id(builder["workflow_sha"], name="workflow SHA")
    repository_id = _positive_id(builder["repository_id"], name="repository ID")
    owner_id = _positive_id(
        builder["repository_owner_id"],
        name="repository owner ID",
    )
    run_id = _positive_id(builder["run_id"], name="workflow run ID")
    run_attempt = _strict_int(
        builder["run_attempt"],
        name="workflow run attempt",
        minimum=1,
        maximum=1_000_000,
    )
    event_name = builder["event_name"]
    if not isinstance(event_name, str) or event_name not in _EVENTS:
        raise BuildProvenanceError("build provenance event is unsupported")
    ref = _ref(builder["ref"], name="build provenance ref")
    ref_protected = builder["ref_protected"]
    if not isinstance(ref_protected, bool):
        raise BuildProvenanceError("ref_protected must be a boolean")
    runner_name = _safe_text(builder["runner_name"], name="runner name")
    workflow_ref = _safe_text(builder["workflow_ref"], name="workflow ref")
    expected_prefix = (
        f"{BUILD_PROVENANCE_REPOSITORY}/{BUILD_PROVENANCE_WORKFLOW_PATH}@"
    )
    if not workflow_ref.startswith(expected_prefix):
        raise BuildProvenanceError("workflow ref names a different workflow")
    _ref(workflow_ref[len(expected_prefix) :], name="workflow ref revision")
    expected_url = (
        f"{BUILD_PROVENANCE_SERVER_URL}/{BUILD_PROVENANCE_REPOSITORY}/"
        f"actions/runs/{run_id}"
    )
    if builder["run_url"] != expected_url:
        raise BuildProvenanceError("workflow run URL is inconsistent")
    return {
        "checkout_sha": checkout_sha,
        "event_name": str(event_name),
        "job": BUILD_PROVENANCE_JOB,
        "provider": "github-actions",
        "ref": ref,
        "ref_protected": ref_protected,
        "repository": BUILD_PROVENANCE_REPOSITORY,
        "repository_id": repository_id,
        "repository_owner_id": owner_id,
        "run_attempt": run_attempt,
        "run_id": run_id,
        "run_url": expected_url,
        "runner_arch": "X64",
        "runner_environment": "github-hosted",
        "runner_name": runner_name,
        "runner_os": "Windows",
        "server_url": BUILD_PROVENANCE_SERVER_URL,
        "workflow_ref": workflow_ref,
        "workflow_sha": workflow_sha,
    }


def github_actions_invocation(
    environment: Mapping[str, str],
) -> BuildInvocation:
    """Read only documented GitHub/runner variables into a closed identity."""

    def required(name: str) -> str:
        value = environment.get(name)
        if value is None or not value:
            raise BuildProvenanceError(f"GitHub Actions variable is missing: {name}")
        return value

    if required("GITHUB_ACTIONS") != "true":
        raise BuildProvenanceError("provenance generation requires GitHub Actions")
    protected_text = required("GITHUB_REF_PROTECTED")
    if protected_text not in {"true", "false"}:
        raise BuildProvenanceError("GITHUB_REF_PROTECTED must be true or false")
    run_id = required("GITHUB_RUN_ID")
    try:
        run_attempt: object = int(required("GITHUB_RUN_ATTEMPT"), 10)
    except ValueError as exc:
        raise BuildProvenanceError("GITHUB_RUN_ATTEMPT must be an integer") from exc
    server_url = required("GITHUB_SERVER_URL")
    repository = required("GITHUB_REPOSITORY")
    builder = _validate_builder(
        {
            "checkout_sha": required("GITHUB_SHA"),
            "event_name": required("GITHUB_EVENT_NAME"),
            "job": required("GITHUB_JOB"),
            "provider": "github-actions",
            "ref": required("GITHUB_REF"),
            "ref_protected": protected_text == "true",
            "repository": repository,
            "repository_id": required("GITHUB_REPOSITORY_ID"),
            "repository_owner_id": required("GITHUB_REPOSITORY_OWNER_ID"),
            "run_attempt": run_attempt,
            "run_id": run_id,
            "run_url": f"{server_url}/{repository}/actions/runs/{run_id}",
            "runner_arch": required("RUNNER_ARCH"),
            "runner_environment": required("RUNNER_ENVIRONMENT"),
            "runner_name": required("RUNNER_NAME"),
            "runner_os": required("RUNNER_OS"),
            "server_url": server_url,
            "workflow_ref": required("GITHUB_WORKFLOW_REF"),
            "workflow_sha": required("GITHUB_WORKFLOW_SHA"),
        }
    )
    return BuildInvocation(
        source_commit=str(builder["checkout_sha"]),
        builder=builder,
    )


def _verify_payload_entries(payload_root: Path, manifest: Mapping[str, Any]) -> None:
    entries = manifest.get("entries")
    if not isinstance(entries, list):  # already guarded by portable verifier
        raise BuildProvenanceError("portable manifest has no entry list")
    expected: dict[str, Mapping[str, Any]] = {}
    for value in entries:
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
            raise BuildProvenanceError("portable manifest entry is invalid")
        expected[str(value["path"])] = value
    observed: set[str] = set()
    for path in payload_root.rglob("*"):
        if path.is_symlink():
            raise BuildProvenanceError("portable payload contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise BuildProvenanceError("portable payload contains a non-regular file")
        relative = path.relative_to(payload_root).as_posix()
        record = expected.get(relative)
        if record is None:
            raise BuildProvenanceError(f"portable payload has an extra file: {relative}")
        digest, size = _sha256_file(path)
        if record.get("sha256") != digest or record.get("size") != size:
            raise BuildProvenanceError(
                f"portable payload differs from archive manifest: {relative}"
            )
        observed.add(relative)
    missing = set(expected) - observed
    if missing:
        raise BuildProvenanceError(
            f"portable payload is missing archived files: {sorted(missing)}"
        )


def _path_descriptor(path: str, raw: bytes) -> dict[str, object]:
    return {"path": path, "sha256": _sha256_bytes(raw), "size": len(raw)}


def _artifact_snapshot(
    portable_archive: Path,
    portable_manifest: Path,
    payload: Path,
) -> tuple[dict[str, object], str, str, Path, Path, Path]:
    archive_file = _regular_file(portable_archive, label="portable archive")
    manifest_file = _regular_file(portable_manifest, label="portable manifest")
    root = _payload_root(payload)
    try:
        portable_result = verify_portable_archive(archive_file, manifest_file)
        evidence_result = verify_release_evidence(root)
    except (PortableArchiveError, ReleaseEvidenceError) as exc:
        raise BuildProvenanceError(f"bound portable payload failed verification: {exc}") from exc
    portable_value, portable_raw = _read_canonical_json(
        manifest_file,
        label="portable manifest",
        limit=64 * 1024 * 1024,
    )
    _verify_payload_entries(root, portable_value)

    source_archive = root / SOURCE_ARCHIVE_DIRECTORY / SOURCE_ARCHIVE_FILENAME
    source_sidecar = root / SOURCE_ARCHIVE_DIRECTORY / SOURCE_ARCHIVE_SIDECAR_FILENAME
    try:
        source_result = verify_source_archive(source_archive, source_sidecar)
    except SourceArchiveError as exc:
        raise BuildProvenanceError(f"bound source archive failed verification: {exc}") from exc
    source_sidecar_file = _regular_file(source_sidecar, label="source archive sidecar")
    _source_value, source_sidecar_raw = _read_canonical_json(
        source_sidecar_file,
        label="source archive sidecar",
        limit=32 * 1024 * 1024,
    )

    evidence_path = f"{EVIDENCE_DIRECTORY_NAME}/release-evidence.json"
    evidence_file = _regular_file(root / evidence_path, label="release evidence index")
    evidence_value, evidence_raw = _read_canonical_json(
        evidence_file,
        label="release evidence index",
    )
    if (
        evidence_value.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence_value.get("payload_sha256") != evidence_result.payload_sha256
    ):
        raise BuildProvenanceError("release evidence index summary is inconsistent")
    commits = {
        portable_result.source_commit,
        source_result.source_commit,
        evidence_value.get("source_commit"),
    }
    if len(commits) != 1 or not all(isinstance(value, str) for value in commits):
        raise BuildProvenanceError("portable, source, and release evidence commits differ")

    artifacts: dict[str, object] = {
        "portable": {
            "archive": {
                "file": archive_file.name,
                "sha256": portable_result.archive_sha256,
                "size": portable_result.archive_size,
            },
            "file_count": portable_result.file_count,
            "manifest": {
                "file": manifest_file.name,
                "sha256": _sha256_bytes(portable_raw),
                "size": len(portable_raw),
            },
            "payload_sha256": portable_result.payload_sha256,
            "payload_size": portable_result.payload_size,
        },
        "release_evidence": {
            "index": _path_descriptor(evidence_path, evidence_raw),
            "payload_sha256": evidence_result.payload_sha256,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
        },
        "source": {
            "archive": {
                "path": f"{SOURCE_ARCHIVE_DIRECTORY}/{SOURCE_ARCHIVE_FILENAME}",
                "sha256": source_result.archive_sha256,
                "size": source_result.archive_size,
            },
            "file_count": source_result.file_count,
            "sidecar": _path_descriptor(
                f"{SOURCE_ARCHIVE_DIRECTORY}/{SOURCE_ARCHIVE_SIDECAR_FILENAME}",
                source_sidecar_raw,
            ),
            "source_sha256": source_result.source_sha256,
            "source_size": source_result.source_size,
        },
    }
    return (
        artifacts,
        source_result.source_commit,
        source_result.source_tree,
        archive_file,
        manifest_file,
        root,
    )


def _validate_provenance(value: object) -> dict[str, object]:
    record = _exact_mapping(
        value,
        {
            "artifacts",
            "authentication",
            "builder",
            "format",
            "schema_version",
            "source_commit",
            "source_tree",
        },
        name="build provenance",
    )
    if (
        record["format"] != BUILD_PROVENANCE_FORMAT
        or record["schema_version"] != BUILD_PROVENANCE_SCHEMA_VERSION
    ):
        raise BuildProvenanceError("build provenance identity is unsupported")
    authentication = _exact_mapping(
        record["authentication"],
        {"kind", "signature_present"},
        name="build provenance authentication",
    )
    if (
        authentication["kind"] != "none"
        or authentication["signature_present"] is not False
    ):
        raise BuildProvenanceError("build provenance must declare its unsigned status")
    artifacts = _exact_mapping(
        record["artifacts"],
        {"portable", "release_evidence", "source"},
        name="build provenance artifacts",
    )
    builder = _validate_builder(record["builder"])
    commit = _object_id(record["source_commit"], name="provenance source commit")
    tree = _object_id(record["source_tree"], name="provenance source tree")
    if len(commit) != len(tree) or builder["checkout_sha"] != commit:
        raise BuildProvenanceError("build provenance source identity is inconsistent")
    return {
        "artifacts": dict(artifacts),
        "authentication": {"kind": "none", "signature_present": False},
        "builder": builder,
        "format": BUILD_PROVENANCE_FORMAT,
        "schema_version": BUILD_PROVENANCE_SCHEMA_VERSION,
        "source_commit": commit,
        "source_tree": tree,
    }


def _output_path(path: Path, *, payload_root: Path) -> Path:
    expanded = path.expanduser()
    if expanded.exists() or expanded.is_symlink():
        raise BuildProvenanceError(
            f"refusing to overwrite existing build provenance: {expanded}"
        )
    resolved = expanded.resolve(strict=False)
    if resolved.exists() or resolved.is_symlink():
        raise BuildProvenanceError(
            f"refusing to overwrite existing build provenance: {resolved}"
        )
    try:
        resolved.relative_to(payload_root)
    except ValueError:
        pass
    else:
        raise BuildProvenanceError("build provenance must be outside the payload")
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BuildProvenanceError("could not create provenance parent directory") from exc
    return resolved


def _unlink_owned(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        current = path.stat(follow_symlinks=False)
        if (int(current.st_dev), int(current.st_ino)) == identity:
            path.unlink()
    except (FileNotFoundError, OSError):
        return


def generate_build_provenance(
    portable_archive: Path,
    portable_manifest: Path,
    payload: Path,
    output_path: Path,
    *,
    invocation: BuildInvocation,
) -> BuildProvenanceResult:
    """Generate one no-overwrite unsigned record after verifying every input."""

    artifacts, commit, tree, archive_file, manifest_file, root = _artifact_snapshot(
        portable_archive,
        portable_manifest,
        payload,
    )
    supplied_commit = _object_id(
        invocation.source_commit,
        name="invocation source commit",
    )
    builder = _validate_builder(invocation.builder)
    if supplied_commit != commit or builder["checkout_sha"] != commit:
        raise BuildProvenanceError("builder invocation does not match artifact source")
    output = _output_path(output_path, payload_root=root)
    if output in {archive_file, manifest_file}:
        raise BuildProvenanceError("build provenance path collides with an input")
    record = {
        "artifacts": artifacts,
        "authentication": {"kind": "none", "signature_present": False},
        "builder": builder,
        "format": BUILD_PROVENANCE_FORMAT,
        "schema_version": BUILD_PROVENANCE_SCHEMA_VERSION,
        "source_commit": commit,
        "source_tree": tree,
    }
    encoded = _canonical_bytes(record)
    if len(encoded) > BUILD_PROVENANCE_MAX_BYTES:
        raise BuildProvenanceError("build provenance exceeds the byte budget")
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    identity: tuple[int, int] | None = None
    try:
        temporary.write_bytes(encoded)
        stat = temporary.stat()
        identity = int(stat.st_dev), int(stat.st_ino)
        os.link(temporary, output)
    except FileExistsError as exc:
        raise BuildProvenanceError(
            f"refusing to overwrite concurrently created provenance: {output}"
        ) from exc
    except OSError as exc:
        raise BuildProvenanceError(f"could not publish build provenance: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    try:
        return verify_build_provenance(output, archive_file, manifest_file, root)
    except BuildProvenanceError:
        _unlink_owned(output, identity)
        raise


def verify_build_provenance(
    provenance_path: Path,
    portable_archive: Path,
    portable_manifest: Path,
    payload: Path,
) -> BuildProvenanceResult:
    """Verify the unsigned record and all artifacts without network access."""

    provenance = _regular_file(provenance_path, label="build provenance")
    value, raw = _read_canonical_json(provenance, label="build provenance")
    record = _validate_provenance(value)
    artifacts, commit, tree, _archive, _manifest, _root = _artifact_snapshot(
        portable_archive,
        portable_manifest,
        payload,
    )
    if (
        _canonical_bytes(record["artifacts"]) != _canonical_bytes(artifacts)
        or record["source_commit"] != commit
        or record["source_tree"] != tree
    ):
        raise BuildProvenanceError("build provenance does not match bound artifacts")
    builder = record["builder"]
    if not isinstance(builder, Mapping):  # validated above
        raise BuildProvenanceError("build provenance builder is invalid")
    portable = artifacts["portable"]
    source = artifacts["source"]
    if not isinstance(portable, Mapping) or not isinstance(source, Mapping):
        raise BuildProvenanceError("build provenance artifact summary is invalid")
    portable_descriptor = portable["archive"]
    source_descriptor = source["archive"]
    if not isinstance(portable_descriptor, Mapping) or not isinstance(
        source_descriptor, Mapping
    ):
        raise BuildProvenanceError("build provenance archive descriptor is invalid")
    return BuildProvenanceResult(
        provenance_sha256=_sha256_bytes(raw),
        provenance_size=len(raw),
        portable_archive_sha256=str(portable_descriptor["sha256"]),
        source_archive_sha256=str(source_descriptor["sha256"]),
        source_commit=commit,
        source_tree=tree,
        run_id=str(builder["run_id"]),
        run_attempt=int(builder["run_attempt"]),
    )


__all__ = [
    "BUILD_PROVENANCE_FORMAT",
    "BUILD_PROVENANCE_SCHEMA_VERSION",
    "BuildInvocation",
    "BuildProvenanceError",
    "BuildProvenanceResult",
    "generate_build_provenance",
    "github_actions_invocation",
    "verify_build_provenance",
]
