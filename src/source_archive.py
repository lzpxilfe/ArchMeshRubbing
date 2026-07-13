"""Deterministic, offline-verifiable corresponding-source archives.

Release archives are built from immutable Git objects rather than the live
working tree.  The ZIP contains every tracked regular blob for one exact
commit plus a canonical internal manifest.  An external canonical sidecar
binds the ZIP bytes to the commit, tree, source file set, and project license.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, TypedDict
import unicodedata
import uuid
from zipfile import BadZipFile, ZIP_STORED, ZipFile, ZipInfo

from src.core.canonical_json import CanonicalJSONError, canonical_json_bytes


SOURCE_ARCHIVE_FORMAT = "org.archmeshrubbing.source-archive"
SOURCE_MANIFEST_FORMAT = "org.archmeshrubbing.source-manifest"
SOURCE_ARCHIVE_SCHEMA_VERSION = "1.0.0"
SOURCE_ARCHIVE_DIRECTORY = "source"
SOURCE_ARCHIVE_FILENAME = "ArchMeshRubbing-source.zip"
SOURCE_ARCHIVE_SIDECAR_FILENAME = "ArchMeshRubbing-source.json"
SOURCE_ARCHIVE_INTERNAL_MANIFEST = "SOURCE-MANIFEST.json"
SOURCE_ARCHIVE_REPOSITORY = "https://github.com/lzpxilfe/ArchMeshRubbing"
SOURCE_ARCHIVE_LICENSE_EXPRESSION = "GPL-2.0-only"
SOURCE_ARCHIVE_COMMENT = b"ArchMeshRubbing corresponding source v1"

SOURCE_ARCHIVE_MAX_FILES = 20_000
SOURCE_ARCHIVE_MAX_FILE_BYTES = 64 * 1024 * 1024
SOURCE_ARCHIVE_MAX_TOTAL_BYTES = 256 * 1024 * 1024
SOURCE_ARCHIVE_MAX_MANIFEST_BYTES = 32 * 1024 * 1024
SOURCE_ARCHIVE_MAX_ARCHIVE_BYTES = 320 * 1024 * 1024
SOURCE_ARCHIVE_MAX_COMMIT_OBJECT_BYTES = 1024 * 1024
SOURCE_ARCHIVE_MAX_PATH_CHARACTERS = 4096

_OBJECT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DEVICE_RE = re.compile(
    r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$",
    flags=re.IGNORECASE,
)
_SOURCE_MODES = frozenset({"100644", "100755"})


class SourceArchiveError(RuntimeError):
    """A source tree, archive, or manifest violates the release contract."""


class _SourceRecord(TypedDict):
    git_blob_oid: str
    mode: str
    path: str
    sha256: str
    size: int


@dataclass(slots=True)
class _TreeDirectory:
    directories: dict[str, "_TreeDirectory"] = field(default_factory=dict)
    files: dict[str, _SourceRecord] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceArchiveResult:
    archive_sha256: str
    archive_size: int
    file_count: int
    source_sha256: str
    source_size: int
    source_commit: str
    source_tree: str
    root_directory: str

    def detail(self) -> str:
        return (
            f"archive={self.archive_sha256}, source={self.source_sha256}, "
            f"files={self.file_count}, bytes={self.source_size}, "
            f"commit={self.source_commit}, tree={self.source_tree}"
        )


def _canonical_bytes(value: object) -> bytes:
    try:
        return canonical_json_bytes(value)
    except CanonicalJSONError as exc:
        raise SourceArchiveError("source manifest is not canonical JSON") from exc


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
        raise SourceArchiveError(f"could not read source archive file: {path}") from exc
    return digest.hexdigest(), size


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourceArchiveError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise SourceArchiveError(
            f"{name} must be in the inclusive range {minimum}..{maximum}"
        )
    return value


def _exact_mapping(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceArchiveError(f"{name} must be an object")
    observed = set(value)
    if observed != expected:
        raise SourceArchiveError(
            f"{name} fields are invalid; missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )
    return value


def _object_id(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _OBJECT_RE.fullmatch(value) is None:
        raise SourceArchiveError(f"{name} must be a full lowercase Git object ID")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SourceArchiveError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_component(component: str, *, label: str) -> None:
    if (
        not component
        or component in {".", ".."}
        or component.endswith((" ", "."))
        or ":" in component
        or "\x00" in component
        or any(ord(character) < 32 for character in component)
        or _WINDOWS_DEVICE_RE.fullmatch(component) is not None
    ):
        raise SourceArchiveError(f"{label} is not a portable Windows path")
    if unicodedata.normalize("NFC", component) != component:
        raise SourceArchiveError(f"{label} must use NFC Unicode normalization")


def _relative_path(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > SOURCE_ARCHIVE_MAX_PATH_CHARACTERS
        or "\\" in value
    ):
        raise SourceArchiveError(f"{label} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise SourceArchiveError(f"{label} is not canonical")
    for component in path.parts:
        _validate_component(component, label=label)
    return value


def _root_directory(value: object) -> str:
    if not isinstance(value, str) or "/" in value or "\\" in value:
        raise SourceArchiveError("source archive root directory is invalid")
    _validate_component(value, label="source archive root directory")
    return value


def _zip_datetime(source_date_epoch: int) -> tuple[int, int, int, int, int, int]:
    if isinstance(source_date_epoch, bool) or not isinstance(source_date_epoch, int):
        raise SourceArchiveError("source_date_epoch must be an integer")
    try:
        value = datetime.fromtimestamp(source_date_epoch, tz=timezone.utc)
    except (OSError, OverflowError, ValueError) as exc:
        raise SourceArchiveError("source_date_epoch is outside the ZIP range") from exc
    if value.year < 1980 or value.year > 2107:
        raise SourceArchiveError("source_date_epoch is outside the ZIP range")
    return (
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second // 2 * 2,
    )


def _run_git(repository: Path, *arguments: str, text: bool = True) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceArchiveError(
            f"Git could not read immutable source objects: {' '.join(arguments)}"
        ) from exc
    return completed.stdout


def _resolve_repository(repository: Path) -> Path:
    try:
        root = repository.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SourceArchiveError("source repository does not exist") from exc
    if not root.is_dir():
        raise SourceArchiveError("source repository must be a directory")
    observed = str(_run_git(root, "rev-parse", "--show-toplevel")).strip()
    try:
        top = Path(observed).resolve(strict=True)
    except OSError as exc:
        raise SourceArchiveError("Git returned an unreadable repository root") from exc
    if top != root:
        raise SourceArchiveError("source repository must be the Git top-level directory")
    return root


def _resolve_commit(repository: Path, commit: str | None) -> str:
    reference = "HEAD" if commit is None else commit
    if commit is not None:
        _object_id(commit, name="source commit")
    resolved = str(
        _run_git(repository, "rev-parse", "--verify", f"{reference}^{{commit}}")
    ).strip()
    resolved = _object_id(resolved, name="resolved source commit")
    if commit is not None and resolved != commit:
        raise SourceArchiveError("source commit did not resolve exactly")
    return resolved


def _commit_tree_and_epoch(repository: Path, commit: str) -> tuple[str, int]:
    tree = str(_run_git(repository, "rev-parse", f"{commit}^{{tree}}"))
    tree = _object_id(tree.strip(), name="source tree")
    epoch_text = str(_run_git(repository, "show", "-s", "--format=%ct", commit)).strip()
    try:
        epoch = int(epoch_text, 10)
    except ValueError as exc:
        raise SourceArchiveError("source commit timestamp is invalid") from exc
    _zip_datetime(epoch)
    return tree, epoch


def _git_object_digest(payload: bytes, *, kind: str, oid_length: int) -> str:
    if kind not in {"blob", "commit", "tree"}:
        raise SourceArchiveError("unsupported Git object kind")
    framed = f"{kind} {len(payload)}\0".encode("ascii") + payload
    if oid_length == 40:
        return hashlib.sha1(framed, usedforsecurity=False).hexdigest()
    if oid_length == 64:
        return hashlib.sha256(framed).hexdigest()
    raise SourceArchiveError("unsupported Git object hash length")


def _git_blob_digest(payload: bytes, oid_length: int) -> str:
    return _git_object_digest(payload, kind="blob", oid_length=oid_length)


def _read_commit_object(repository: Path, commit: str) -> bytes:
    payload = _run_git(repository, "cat-file", "commit", commit, text=False)
    if not isinstance(payload, bytes):  # pragma: no cover - text=False contract
        raise SourceArchiveError("Git commit reader returned text")
    if not payload or len(payload) > SOURCE_ARCHIVE_MAX_COMMIT_OBJECT_BYTES:
        raise SourceArchiveError("Git commit object byte length is invalid")
    if _git_object_digest(payload, kind="commit", oid_length=len(commit)) != commit:
        raise SourceArchiveError("Git commit object bytes do not match commit ID")
    return payload


def _commit_tree_header(payload: bytes) -> str:
    first_line = payload.split(b"\n", 1)[0]
    if not first_line.startswith(b"tree "):
        raise SourceArchiveError("Git commit object has no leading tree header")
    try:
        tree = first_line[5:].decode("ascii")
    except UnicodeDecodeError as exc:
        raise SourceArchiveError("Git commit tree header is not ASCII") from exc
    return _object_id(tree, name="Git commit tree header")


def _read_blob(repository: Path, oid: str) -> bytes:
    payload = _run_git(repository, "cat-file", "blob", oid, text=False)
    if not isinstance(payload, bytes):  # pragma: no cover - text=False contract
        raise SourceArchiveError("Git blob reader returned text")
    if len(payload) > SOURCE_ARCHIVE_MAX_FILE_BYTES:
        raise SourceArchiveError(f"source blob exceeds the per-file budget: {oid}")
    if _git_blob_digest(payload, len(oid)) != oid:
        raise SourceArchiveError(f"Git blob bytes do not match object ID: {oid}")
    return payload


def _tree_entries(repository: Path, commit: str) -> list[tuple[str, str, str]]:
    raw = _run_git(repository, "ls-tree", "-rz", "--full-tree", commit, text=False)
    if not isinstance(raw, bytes):  # pragma: no cover - text=False contract
        raise SourceArchiveError("Git tree reader returned text")
    entries: list[tuple[str, str, str]] = []
    seen_casefold: set[str] = set()
    for encoded in raw.split(b"\x00"):
        if not encoded:
            continue
        try:
            header, path_bytes = encoded.split(b"\t", 1)
            mode_bytes, object_type, oid_bytes = header.split(b" ", 2)
            mode = mode_bytes.decode("ascii")
            kind = object_type.decode("ascii")
            oid = oid_bytes.decode("ascii")
            path = path_bytes.decode("utf-8", errors="strict")
        except (ValueError, UnicodeDecodeError) as exc:
            raise SourceArchiveError("Git tree contains an invalid entry") from exc
        if kind != "blob" or mode not in _SOURCE_MODES:
            raise SourceArchiveError(
                f"source tree entry is not a regular tracked file: {path} ({mode} {kind})"
            )
        _object_id(oid, name=f"Git blob ID for {path}")
        canonical_path = _relative_path(path, label="source file path")
        folded = canonical_path.casefold()
        if folded in seen_casefold:
            raise SourceArchiveError(
                f"source tree has a case-insensitive path collision: {canonical_path}"
            )
        seen_casefold.add(folded)
        entries.append((canonical_path, mode, oid))
    entries.sort(key=lambda item: item[0].encode("utf-8"))
    if not entries or len(entries) > SOURCE_ARCHIVE_MAX_FILES:
        raise SourceArchiveError("source tree file count is outside the safety budget")
    return entries


def _collect_source(
    repository: Path,
    entries: list[tuple[str, str, str]],
) -> tuple[list[_SourceRecord], dict[str, bytes]]:
    records: list[_SourceRecord] = []
    payloads: dict[str, bytes] = {}
    total = 0
    for path, mode, oid in entries:
        payload = _read_blob(repository, oid)
        total += len(payload)
        if total > SOURCE_ARCHIVE_MAX_TOTAL_BYTES:
            raise SourceArchiveError("source tree exceeds the total byte budget")
        payloads[path] = payload
        records.append(
            {
                "git_blob_oid": oid,
                "mode": mode,
                "path": path,
                "sha256": _sha256_bytes(payload),
                "size": len(payload),
            }
        )
    if "LICENSE" not in payloads:
        raise SourceArchiveError("source tree has no tracked LICENSE file")
    return records, payloads


def _source_descriptor(records: list[_SourceRecord]) -> dict[str, object]:
    return {
        "file_count": len(records),
        "sha256": _sha256_bytes(_canonical_bytes(records)),
        "size": sum(record["size"] for record in records),
    }


def _reconstructed_tree_oid(records: list[_SourceRecord], *, oid_length: int) -> str:
    root = _TreeDirectory()
    for record in records:
        parts = PurePosixPath(record["path"]).parts
        current = root
        for component in parts[:-1]:
            if component in current.files:
                raise SourceArchiveError("source tree path changes a file into a directory")
            current = current.directories.setdefault(component, _TreeDirectory())
        basename = parts[-1]
        if basename in current.directories or basename in current.files:
            raise SourceArchiveError("source tree contains a duplicate path")
        if len(record["git_blob_oid"]) != oid_length:
            raise SourceArchiveError("source tree mixes Git object hash algorithms")
        current.files[basename] = record

    def digest(directory: _TreeDirectory) -> str:
        entries: list[tuple[bytes, bool, bytes]] = []
        for name, record in directory.files.items():
            encoded = name.encode("utf-8")
            payload = (
                record["mode"].encode("ascii")
                + b" "
                + encoded
                + b"\x00"
                + bytes.fromhex(record["git_blob_oid"])
            )
            entries.append((encoded, False, payload))
        for name, child in directory.directories.items():
            encoded = name.encode("utf-8")
            child_oid = digest(child)
            payload = b"40000 " + encoded + b"\x00" + bytes.fromhex(child_oid)
            entries.append((encoded, True, payload))
        entries.sort(key=lambda item: item[0] + (b"/" if item[1] else b"\x00"))
        tree_payload = b"".join(item[2] for item in entries)
        return _git_object_digest(
            tree_payload,
            kind="tree",
            oid_length=oid_length,
        )

    return digest(root)


def _source_manifest(
    *,
    commit: str,
    tree: str,
    source_date_epoch: int,
    root_directory: str,
    records: list[_SourceRecord],
    commit_object: bytes,
) -> dict[str, object]:
    license_record = next(record for record in records if record["path"] == "LICENSE")
    return {
        "commit_object": {
            "encoding": "base64",
            "payload": base64.b64encode(commit_object).decode("ascii"),
            "sha256": _sha256_bytes(commit_object),
            "size": len(commit_object),
        },
        "files": records,
        "format": SOURCE_MANIFEST_FORMAT,
        "license": {
            "expression": SOURCE_ARCHIVE_LICENSE_EXPRESSION,
            "path": "LICENSE",
            "sha256": license_record["sha256"],
        },
        "repository": SOURCE_ARCHIVE_REPOSITORY,
        "root_directory": root_directory,
        "schema_version": SOURCE_ARCHIVE_SCHEMA_VERSION,
        "source": _source_descriptor(records),
        "source_commit": commit,
        "source_date_epoch": source_date_epoch,
        "source_tree": tree,
    }


def _zip_info(
    name: str,
    *,
    mode: str,
    date_time: tuple[int, int, int, int, int, int],
) -> ZipInfo:
    info = ZipInfo(name, date_time=date_time)
    info.compress_type = ZIP_STORED
    info.create_system = 3
    info.external_attr = int(mode, 8) << 16
    info.extra = b""
    info.comment = b""
    return info


def _write_archive(
    path: Path,
    *,
    root_directory: str,
    date_time: tuple[int, int, int, int, int, int],
    manifest_bytes: bytes,
    records: list[_SourceRecord],
    payloads: Mapping[str, bytes],
) -> None:
    try:
        with ZipFile(
            path,
            mode="x",
            compression=ZIP_STORED,
            allowZip64=True,
            strict_timestamps=True,
        ) as archive:
            archive.comment = SOURCE_ARCHIVE_COMMENT
            manifest_name = f"{root_directory}/{SOURCE_ARCHIVE_INTERNAL_MANIFEST}"
            archive.writestr(
                _zip_info(manifest_name, mode="100644", date_time=date_time),
                manifest_bytes,
            )
            for record in records:
                relative = record["path"]
                payload = payloads[relative]
                if (
                    len(payload) != record["size"]
                    or _sha256_bytes(payload) != record["sha256"]
                ):
                    raise SourceArchiveError(
                        f"source bytes changed while archiving: {relative}"
                    )
                archive.writestr(
                    _zip_info(
                        f"{root_directory}/{relative}",
                        mode=record["mode"],
                        date_time=date_time,
                    ),
                    payload,
                )
    except (OSError, BadZipFile) as exc:
        raise SourceArchiveError(f"could not write source archive: {exc}") from exc


def _output_path(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.exists() or expanded.is_symlink():
        raise SourceArchiveError(f"refusing to overwrite existing {label}: {expanded}")
    resolved = expanded.resolve(strict=False)
    if resolved.exists() or resolved.is_symlink():
        raise SourceArchiveError(f"refusing to overwrite existing {label}: {resolved}")
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SourceArchiveError(f"could not create {label} parent") from exc
    if not resolved.parent.is_dir():
        raise SourceArchiveError(f"{label} parent is not a directory")
    return resolved


def _publish_no_replace(staging: Path, destination: Path) -> tuple[int, int]:
    identity = staging.stat()
    try:
        os.link(staging, destination)
    except FileExistsError as exc:
        raise SourceArchiveError(
            f"refusing to overwrite concurrently created output: {destination}"
        ) from exc
    except OSError as exc:
        raise SourceArchiveError(f"could not publish output: {destination}") from exc
    return int(identity.st_dev), int(identity.st_ino)


def _unlink_owned(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        current = path.stat(follow_symlinks=False)
        if (int(current.st_dev), int(current.st_ino)) == identity:
            path.unlink()
    except (FileNotFoundError, OSError):
        return


def build_source_archive(
    repository: Path,
    archive_path: Path,
    sidecar_path: Path,
    *,
    commit: str | None = None,
) -> SourceArchiveResult:
    """Build a no-overwrite ZIP directly from one immutable Git commit."""

    root = _resolve_repository(repository)
    resolved_commit = _resolve_commit(root, commit)
    tree, source_date_epoch = _commit_tree_and_epoch(root, resolved_commit)
    root_directory = _root_directory(
        f"ArchMeshRubbing-source-{resolved_commit[:12]}"
    )
    entries = _tree_entries(root, resolved_commit)
    records, payloads = _collect_source(root, entries)
    if _reconstructed_tree_oid(records, oid_length=len(tree)) != tree:
        raise SourceArchiveError("source file records do not reconstruct the Git tree")
    commit_object = _read_commit_object(root, resolved_commit)
    if _commit_tree_header(commit_object) != tree:
        raise SourceArchiveError("Git commit object does not reference the source tree")
    source_manifest = _source_manifest(
        commit=resolved_commit,
        tree=tree,
        source_date_epoch=source_date_epoch,
        root_directory=root_directory,
        records=records,
        commit_object=commit_object,
    )
    manifest_bytes = _canonical_bytes(source_manifest)
    if len(manifest_bytes) > SOURCE_ARCHIVE_MAX_MANIFEST_BYTES:
        raise SourceArchiveError("source manifest exceeds the byte budget")

    archive_final = _output_path(archive_path, label="source archive")
    sidecar_final = _output_path(sidecar_path, label="source sidecar")
    if archive_final == sidecar_final:
        raise SourceArchiveError("source archive and sidecar paths must differ")
    date_time = _zip_datetime(source_date_epoch)
    archive_temporary = (
        archive_final.parent / f".{archive_final.name}.{uuid.uuid4().hex}.tmp"
    )
    sidecar_temporary = (
        sidecar_final.parent / f".{sidecar_final.name}.{uuid.uuid4().hex}.tmp"
    )
    archive_identity: tuple[int, int] | None = None
    sidecar_identity: tuple[int, int] | None = None
    try:
        _write_archive(
            archive_temporary,
            root_directory=root_directory,
            date_time=date_time,
            manifest_bytes=manifest_bytes,
            records=records,
            payloads=payloads,
        )
        archive_sha256, archive_size = _sha256_file(archive_temporary)
        sidecar = {
            "archive": {
                "file": archive_final.name,
                "sha256": archive_sha256,
                "size": archive_size,
            },
            "format": SOURCE_ARCHIVE_FORMAT,
            "manifest": {
                "path": f"{root_directory}/{SOURCE_ARCHIVE_INTERNAL_MANIFEST}",
                "sha256": _sha256_bytes(manifest_bytes),
                "size": len(manifest_bytes),
            },
            "root_directory": root_directory,
            "schema_version": SOURCE_ARCHIVE_SCHEMA_VERSION,
            "source": _source_descriptor(records),
            "source_commit": resolved_commit,
            "source_date_epoch": source_date_epoch,
            "source_tree": tree,
        }
        sidecar_temporary.write_bytes(_canonical_bytes(sidecar))
        archive_identity = _publish_no_replace(archive_temporary, archive_final)
        sidecar_identity = _publish_no_replace(sidecar_temporary, sidecar_final)
    except OSError as exc:
        raise SourceArchiveError(f"could not build source archive: {exc}") from exc
    finally:
        if archive_identity is not None and sidecar_identity is None:
            _unlink_owned(archive_final, archive_identity)
        archive_temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)

    try:
        return verify_source_archive(archive_final, sidecar_final)
    except SourceArchiveError:
        _unlink_owned(archive_final, archive_identity)
        _unlink_owned(sidecar_final, sidecar_identity)
        raise


def _read_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SourceArchiveError(f"could not read {label}") from exc
    if not raw or len(raw) > SOURCE_ARCHIVE_MAX_MANIFEST_BYTES:
        raise SourceArchiveError(f"{label} byte length is invalid")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceArchiveError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict) or _canonical_bytes(value) != raw:
        raise SourceArchiveError(f"{label} is not canonical RFC 8785 JSON")
    return value


def _validate_records(value: object) -> list[_SourceRecord]:
    if not isinstance(value, list) or not value or len(value) > SOURCE_ARCHIVE_MAX_FILES:
        raise SourceArchiveError("source manifest file list is invalid")
    records: list[_SourceRecord] = []
    seen_casefold: set[str] = set()
    total = 0
    for raw in value:
        record = _exact_mapping(
            raw,
            {"git_blob_oid", "mode", "path", "sha256", "size"},
            name="source file record",
        )
        path = _relative_path(record["path"], label="source file path")
        folded = path.casefold()
        if folded in seen_casefold:
            raise SourceArchiveError(f"source manifest path collision: {path}")
        seen_casefold.add(folded)
        mode = record["mode"]
        if mode not in _SOURCE_MODES:
            raise SourceArchiveError(f"source file mode is invalid: {path}")
        oid = _object_id(record["git_blob_oid"], name=f"Git blob ID for {path}")
        digest = _sha256(record["sha256"], name=f"source SHA-256 for {path}")
        size = _strict_int(
            record["size"],
            name=f"source size for {path}",
            minimum=0,
            maximum=SOURCE_ARCHIVE_MAX_FILE_BYTES,
        )
        total += size
        if total > SOURCE_ARCHIVE_MAX_TOTAL_BYTES:
            raise SourceArchiveError("source manifest exceeds the total byte budget")
        records.append(
            {
                "git_blob_oid": oid,
                "mode": str(mode),
                "path": path,
                "sha256": digest,
                "size": size,
            }
        )
    if records != sorted(records, key=lambda item: item["path"].encode("utf-8")):
        raise SourceArchiveError("source manifest files are not canonically sorted")
    if "LICENSE" not in {record["path"] for record in records}:
        raise SourceArchiveError("source manifest has no LICENSE file")
    return records


def _validate_source_manifest(value: object) -> dict[str, Any]:
    manifest = _exact_mapping(
        value,
        {
            "commit_object",
            "files",
            "format",
            "license",
            "repository",
            "root_directory",
            "schema_version",
            "source",
            "source_commit",
            "source_date_epoch",
            "source_tree",
        },
        name="source manifest",
    )
    if (
        manifest["format"] != SOURCE_MANIFEST_FORMAT
        or manifest["schema_version"] != SOURCE_ARCHIVE_SCHEMA_VERSION
        or manifest["repository"] != SOURCE_ARCHIVE_REPOSITORY
    ):
        raise SourceArchiveError("source manifest identity is unsupported")
    root = _root_directory(manifest["root_directory"])
    commit = _object_id(manifest["source_commit"], name="source commit")
    tree = _object_id(manifest["source_tree"], name="source tree")
    if len(commit) != len(tree):
        raise SourceArchiveError("source commit and tree use different hash algorithms")
    if root != f"ArchMeshRubbing-source-{commit[:12]}":
        raise SourceArchiveError("source archive root does not match source commit")
    epoch = _strict_int(
        manifest["source_date_epoch"],
        name="source_date_epoch",
        minimum=315532800,
        maximum=4354819199,
    )
    _zip_datetime(epoch)
    records = _validate_records(manifest["files"])
    commit_descriptor = _exact_mapping(
        manifest["commit_object"],
        {"encoding", "payload", "sha256", "size"},
        name="Git commit object descriptor",
    )
    if commit_descriptor["encoding"] != "base64" or not isinstance(
        commit_descriptor["payload"], str
    ):
        raise SourceArchiveError("Git commit object encoding is invalid")
    try:
        commit_object = base64.b64decode(
            commit_descriptor["payload"].encode("ascii"),
            validate=True,
        )
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise SourceArchiveError("Git commit object base64 is invalid") from exc
    commit_size = _strict_int(
        commit_descriptor["size"],
        name="Git commit object size",
        minimum=1,
        maximum=SOURCE_ARCHIVE_MAX_COMMIT_OBJECT_BYTES,
    )
    if (
        len(commit_object) != commit_size
        or _sha256(commit_descriptor["sha256"], name="Git commit object SHA-256")
        != _sha256_bytes(commit_object)
        or _git_object_digest(commit_object, kind="commit", oid_length=len(commit))
        != commit
        or _commit_tree_header(commit_object) != tree
    ):
        raise SourceArchiveError("Git commit object descriptor is inconsistent")
    if _reconstructed_tree_oid(records, oid_length=len(tree)) != tree:
        raise SourceArchiveError("source records do not reconstruct the declared Git tree")
    source = _exact_mapping(
        manifest["source"],
        {"file_count", "sha256", "size"},
        name="source descriptor",
    )
    if dict(source) != _source_descriptor(records):
        raise SourceArchiveError("source descriptor does not match file records")
    license_value = _exact_mapping(
        manifest["license"],
        {"expression", "path", "sha256"},
        name="source license descriptor",
    )
    license_record = next(record for record in records if record["path"] == "LICENSE")
    if dict(license_value) != {
        "expression": SOURCE_ARCHIVE_LICENSE_EXPRESSION,
        "path": "LICENSE",
        "sha256": license_record["sha256"],
    }:
        raise SourceArchiveError("source license descriptor is invalid")
    return {
        "commit_object": {
            "encoding": "base64",
            "payload": str(commit_descriptor["payload"]),
            "sha256": str(commit_descriptor["sha256"]),
            "size": commit_size,
        },
        "files": records,
        "format": SOURCE_MANIFEST_FORMAT,
        "license": dict(license_value),
        "repository": SOURCE_ARCHIVE_REPOSITORY,
        "root_directory": root,
        "schema_version": SOURCE_ARCHIVE_SCHEMA_VERSION,
        "source": dict(source),
        "source_commit": commit,
        "source_date_epoch": epoch,
        "source_tree": tree,
    }


def _validate_sidecar(
    value: object,
    *,
    archive_name: str,
) -> dict[str, Any]:
    sidecar = _exact_mapping(
        value,
        {
            "archive",
            "format",
            "manifest",
            "root_directory",
            "schema_version",
            "source",
            "source_commit",
            "source_date_epoch",
            "source_tree",
        },
        name="source archive sidecar",
    )
    if (
        sidecar["format"] != SOURCE_ARCHIVE_FORMAT
        or sidecar["schema_version"] != SOURCE_ARCHIVE_SCHEMA_VERSION
    ):
        raise SourceArchiveError("source archive sidecar identity is unsupported")
    root = _root_directory(sidecar["root_directory"])
    commit = _object_id(sidecar["source_commit"], name="sidecar source commit")
    tree = _object_id(sidecar["source_tree"], name="sidecar source tree")
    if len(commit) != len(tree):
        raise SourceArchiveError("sidecar commit and tree use different hash algorithms")
    if root != f"ArchMeshRubbing-source-{commit[:12]}":
        raise SourceArchiveError("source sidecar root does not match source commit")
    epoch = _strict_int(
        sidecar["source_date_epoch"],
        name="sidecar source_date_epoch",
        minimum=315532800,
        maximum=4354819199,
    )
    _zip_datetime(epoch)
    archive = _exact_mapping(
        sidecar["archive"],
        {"file", "sha256", "size"},
        name="source archive descriptor",
    )
    if archive["file"] != archive_name:
        raise SourceArchiveError("source sidecar names a different archive")
    archive_descriptor = {
        "file": archive_name,
        "sha256": _sha256(archive["sha256"], name="source archive SHA-256"),
        "size": _strict_int(
            archive["size"],
            name="source archive size",
            minimum=1,
            maximum=SOURCE_ARCHIVE_MAX_ARCHIVE_BYTES,
        ),
    }
    manifest_descriptor = _exact_mapping(
        sidecar["manifest"],
        {"path", "sha256", "size"},
        name="internal source manifest descriptor",
    )
    expected_manifest_path = f"{root}/{SOURCE_ARCHIVE_INTERNAL_MANIFEST}"
    if manifest_descriptor["path"] != expected_manifest_path:
        raise SourceArchiveError("internal source manifest path is invalid")
    manifest_result = {
        "path": expected_manifest_path,
        "sha256": _sha256(
            manifest_descriptor["sha256"],
            name="internal source manifest SHA-256",
        ),
        "size": _strict_int(
            manifest_descriptor["size"],
            name="internal source manifest size",
            minimum=1,
            maximum=SOURCE_ARCHIVE_MAX_MANIFEST_BYTES,
        ),
    }
    source = _exact_mapping(
        sidecar["source"],
        {"file_count", "sha256", "size"},
        name="sidecar source descriptor",
    )
    source_result = {
        "file_count": _strict_int(
            source["file_count"],
            name="sidecar source file count",
            minimum=1,
            maximum=SOURCE_ARCHIVE_MAX_FILES,
        ),
        "sha256": _sha256(source["sha256"], name="sidecar source SHA-256"),
        "size": _strict_int(
            source["size"],
            name="sidecar source size",
            minimum=1,
            maximum=SOURCE_ARCHIVE_MAX_TOTAL_BYTES,
        ),
    }
    return {
        "archive": archive_descriptor,
        "format": SOURCE_ARCHIVE_FORMAT,
        "manifest": manifest_result,
        "root_directory": root,
        "schema_version": SOURCE_ARCHIVE_SCHEMA_VERSION,
        "source": source_result,
        "source_commit": commit,
        "source_date_epoch": epoch,
        "source_tree": tree,
    }


def _read_zip_member(archive: ZipFile, info: ZipInfo, *, limit: int) -> bytes:
    if info.file_size > limit:
        raise SourceArchiveError(f"source archive member exceeds its limit: {info.filename}")
    payload = bytearray()
    try:
        with archive.open(info, "r") as stream:
            while chunk := stream.read(1024 * 1024):
                payload.extend(chunk)
                if len(payload) > limit:
                    raise SourceArchiveError(
                        f"source archive member exceeds its limit: {info.filename}"
                    )
    except (OSError, BadZipFile, RuntimeError) as exc:
        raise SourceArchiveError(
            f"could not read source archive member: {info.filename}"
        ) from exc
    return bytes(payload)


def verify_source_archive(
    archive_path: Path,
    sidecar_path: Path,
) -> SourceArchiveResult:
    """Verify exact archive bytes, ZIP metadata, manifest, and every Git blob."""

    archive_input = archive_path.expanduser()
    sidecar_input = sidecar_path.expanduser()
    if archive_input.is_symlink():
        raise SourceArchiveError("source archive must not be a symbolic link")
    if sidecar_input.is_symlink():
        raise SourceArchiveError("source sidecar must not be a symbolic link")
    archive_file = archive_input.resolve(strict=False)
    sidecar_file = sidecar_input.resolve(strict=False)
    if not archive_file.is_file() or archive_file.is_symlink():
        raise SourceArchiveError("source archive must be a regular file")
    if not sidecar_file.is_file() or sidecar_file.is_symlink():
        raise SourceArchiveError("source sidecar must be a regular file")
    try:
        archive_stat = archive_file.stat(follow_symlinks=False)
    except OSError as exc:
        raise SourceArchiveError("source archive cannot be inspected") from exc
    if archive_stat.st_size < 1 or archive_stat.st_size > SOURCE_ARCHIVE_MAX_ARCHIVE_BYTES:
        raise SourceArchiveError("source archive byte length is outside the safety budget")
    sidecar_raw = _read_canonical_json(sidecar_file, label="source archive sidecar")
    sidecar = _validate_sidecar(sidecar_raw, archive_name=archive_file.name)
    archive_sha256, archive_size = _sha256_file(archive_file)
    if sidecar["archive"] != {
        "file": archive_file.name,
        "sha256": archive_sha256,
        "size": archive_size,
    }:
        raise SourceArchiveError("source archive hash or size does not match sidecar")
    date_time = _zip_datetime(int(sidecar["source_date_epoch"]))
    root = str(sidecar["root_directory"])
    internal_name = f"{root}/{SOURCE_ARCHIVE_INTERNAL_MANIFEST}"
    try:
        with ZipFile(archive_file, mode="r", allowZip64=True) as archive:
            if archive.comment != SOURCE_ARCHIVE_COMMENT:
                raise SourceArchiveError("source archive comment is invalid")
            infos = archive.infolist()
            if not infos or infos[0].filename != internal_name:
                raise SourceArchiveError("source archive internal manifest is not first")
            manifest_info = infos[0]
            manifest_bytes = _read_zip_member(
                archive,
                manifest_info,
                limit=SOURCE_ARCHIVE_MAX_MANIFEST_BYTES,
            )
            if (
                len(manifest_bytes) != sidecar["manifest"]["size"]
                or _sha256_bytes(manifest_bytes) != sidecar["manifest"]["sha256"]
            ):
                raise SourceArchiveError("internal source manifest descriptor mismatch")
            try:
                manifest_raw = json.loads(manifest_bytes.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SourceArchiveError("internal source manifest is invalid JSON") from exc
            if _canonical_bytes(manifest_raw) != manifest_bytes:
                raise SourceArchiveError("internal source manifest is not canonical JSON")
            manifest = _validate_source_manifest(manifest_raw)
            for key in (
                "root_directory",
                "source_commit",
                "source_date_epoch",
                "source_tree",
                "source",
            ):
                if manifest[key] != sidecar[key]:
                    raise SourceArchiveError(
                        f"source sidecar and internal manifest differ at {key}"
                    )
            records = manifest["files"]
            expected_names = [internal_name] + [
                f"{root}/{record['path']}" for record in records
            ]
            if [info.filename for info in infos] != expected_names:
                raise SourceArchiveError("source archive members differ from manifest")
            expected_metadata = [(manifest_info, "100644")]
            expected_metadata.extend(
                (info, record["mode"])
                for info, record in zip(infos[1:], records, strict=True)
            )
            for info, mode in expected_metadata:
                expected_flag_bits = 0x800 if not info.filename.isascii() else 0
                if (
                    info.is_dir()
                    or info.compress_type != ZIP_STORED
                    or info.date_time != date_time
                    or info.create_system != 3
                    or info.external_attr != int(mode, 8) << 16
                    or info.flag_bits != expected_flag_bits
                    or info.extra
                    or info.comment
                ):
                    raise SourceArchiveError(
                        f"source archive member metadata is invalid: {info.filename}"
                    )
            for info, record in zip(infos[1:], records, strict=True):
                if info.file_size != record["size"]:
                    raise SourceArchiveError(
                        f"source archive member size is invalid: {record['path']}"
                    )
                payload = _read_zip_member(
                    archive,
                    info,
                    limit=SOURCE_ARCHIVE_MAX_FILE_BYTES,
                )
                if (
                    len(payload) != record["size"]
                    or _sha256_bytes(payload) != record["sha256"]
                    or _git_blob_digest(payload, len(record["git_blob_oid"]))
                    != record["git_blob_oid"]
                ):
                    raise SourceArchiveError(
                        f"source archive member does not match Git blob: {record['path']}"
                    )
    except BadZipFile as exc:
        raise SourceArchiveError("source archive is not a valid ZIP") from exc
    source = sidecar["source"]
    return SourceArchiveResult(
        archive_sha256=archive_sha256,
        archive_size=archive_size,
        file_count=int(source["file_count"]),
        source_sha256=str(source["sha256"]),
        source_size=int(source["size"]),
        source_commit=str(sidecar["source_commit"]),
        source_tree=str(sidecar["source_tree"]),
        root_directory=root,
    )


__all__ = [
    "SOURCE_ARCHIVE_DIRECTORY",
    "SOURCE_ARCHIVE_FILENAME",
    "SOURCE_ARCHIVE_FORMAT",
    "SOURCE_ARCHIVE_INTERNAL_MANIFEST",
    "SOURCE_ARCHIVE_SCHEMA_VERSION",
    "SOURCE_ARCHIVE_SIDECAR_FILENAME",
    "SOURCE_MANIFEST_FORMAT",
    "SourceArchiveError",
    "SourceArchiveResult",
    "build_source_archive",
    "verify_source_archive",
]
