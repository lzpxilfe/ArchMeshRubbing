"""Deterministic, fail-closed Windows portable archive support.

The public Windows distribution does not need an installer compiler.  This
module packages the already verified PyInstaller payload as one ZIP plus a
canonical sidecar manifest, then validates every member before extraction.
The archive deliberately contains one root directory and no directory,
symlink, device-name, or path-traversal entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, TypedDict
import unicodedata
import uuid
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from src.release_evidence import (
    EVIDENCE_DIRECTORY_NAME,
    ReleaseEvidenceError,
    verify_release_evidence,
)


PORTABLE_ARCHIVE_FORMAT = "org.archmeshrubbing.portable-archive"
PORTABLE_ARCHIVE_SCHEMA_VERSION = "1.0.0"
PORTABLE_ARCHIVE_ROOT = "ArchMeshRubbing"
PORTABLE_ARCHIVE_COMMENT = b"ArchMeshRubbing portable archive v1"
PORTABLE_ARCHIVE_COMPRESSION_LEVEL = 9
PORTABLE_ARCHIVE_EXTERNAL_ATTR = 0o600 << 16
PORTABLE_ARCHIVE_MAX_ENTRIES = 100_000
PORTABLE_ARCHIVE_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024 * 1024
PORTABLE_ARCHIVE_MAX_MANIFEST_BYTES = 64 * 1024 * 1024

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_WINDOWS_DEVICE_RE = re.compile(
    r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$",
    flags=re.IGNORECASE,
)


class PortableArchiveError(RuntimeError):
    """The portable archive is unsafe, inconsistent, or non-canonical."""


class _FileRecord(TypedDict):
    path: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class PortableArchiveResult:
    archive_sha256: str
    archive_size: int
    file_count: int
    payload_sha256: str
    payload_size: int
    source_commit: str

    def detail(self) -> str:
        return (
            f"archive={self.archive_sha256}, payload={self.payload_sha256}, "
            f"files={self.file_count}, bytes={self.payload_size}, "
            f"source={self.source_commit}"
        )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise PortableArchiveError("portable manifest is not canonical JSON") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise PortableArchiveError(f"could not hash file: {path}") from exc
    return digest.hexdigest(), size


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
        raise PortableArchiveError(f"{label} is not a portable Windows path")
    if unicodedata.normalize("NFC", component) != component:
        raise PortableArchiveError(f"{label} must use NFC Unicode normalization")


def _validate_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PortableArchiveError(f"{label} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise PortableArchiveError(f"{label} is not canonical")
    for component in path.parts:
        _validate_component(component, label=label)
    return value


def _validate_root_directory(value: object) -> str:
    if not isinstance(value, str) or "/" in value or "\\" in value:
        raise PortableArchiveError("portable root directory is invalid")
    _validate_component(value, label="portable root directory")
    if value != PORTABLE_ARCHIVE_ROOT:
        raise PortableArchiveError(
            f"portable root directory must be {PORTABLE_ARCHIVE_ROOT!r}"
        )
    return value


def _zip_datetime(source_date_epoch: int) -> tuple[int, int, int, int, int, int]:
    if isinstance(source_date_epoch, bool) or not isinstance(source_date_epoch, int):
        raise PortableArchiveError("source_date_epoch must be an integer")
    try:
        value = datetime.fromtimestamp(source_date_epoch, tz=timezone.utc)
    except (OSError, OverflowError, ValueError) as exc:
        raise PortableArchiveError("source_date_epoch is outside the ZIP range") from exc
    if value.year < 1980 or value.year > 2107:
        raise PortableArchiveError("source_date_epoch is outside the ZIP range")
    return (value.year, value.month, value.day, value.hour, value.minute, value.second // 2 * 2)


def _collect_payload_files(payload_root: Path) -> list[_FileRecord]:
    records: list[_FileRecord] = []
    seen_casefold: set[str] = set()
    try:
        paths = list(payload_root.rglob("*"))
    except OSError as exc:
        raise PortableArchiveError("could not enumerate portable payload") from exc
    for path in paths:
        relative = path.relative_to(payload_root)
        if path.is_symlink():
            raise PortableArchiveError(
                f"portable payload contains a symbolic link: {relative.as_posix()}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise PortableArchiveError(
                f"portable payload contains an unsupported entry: {relative.as_posix()}"
            )
        name = _validate_relative_path(relative.as_posix(), label="payload file path")
        folded = name.casefold()
        if folded in seen_casefold:
            raise PortableArchiveError(
                f"portable payload has a case-insensitive collision: {name}"
            )
        seen_casefold.add(folded)
        digest, size = _sha256_file(path)
        records.append({"path": name, "sha256": digest, "size": size})
    records.sort(key=lambda record: str(record["path"]))
    if not records:
        raise PortableArchiveError("portable payload has no files")
    if len(records) > PORTABLE_ARCHIVE_MAX_ENTRIES:
        raise PortableArchiveError("portable payload exceeds the entry budget")
    total = sum(int(record["size"]) for record in records)
    if total > PORTABLE_ARCHIVE_MAX_PAYLOAD_BYTES:
        raise PortableArchiveError("portable payload exceeds the byte budget")
    return records


def _payload_sha256(records: list[_FileRecord]) -> str:
    return _sha256_bytes(_canonical_json_bytes(records))


def _release_evidence_descriptor(payload_root: Path) -> dict[str, object]:
    try:
        verify_release_evidence(payload_root)
    except ReleaseEvidenceError as exc:
        raise PortableArchiveError(
            f"portable payload release evidence failed verification: {exc}"
        ) from exc
    relative = f"{EVIDENCE_DIRECTORY_NAME}/release-evidence.json"
    path = payload_root / PurePosixPath(relative)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PortableArchiveError("portable payload has no release evidence index") from exc
    if len(raw) > PORTABLE_ARCHIVE_MAX_MANIFEST_BYTES:
        raise PortableArchiveError("release evidence index exceeds the byte budget")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortableArchiveError("release evidence index is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or _canonical_json_bytes(value) != raw:
        raise PortableArchiveError("release evidence index is not canonical JSON")
    source_commit = value.get("source_commit")
    if not isinstance(source_commit, str) or _COMMIT_RE.fullmatch(source_commit) is None:
        raise PortableArchiveError("release evidence source commit is invalid")
    return {
        "path": relative,
        "sha256": _sha256_bytes(raw),
        "source_commit": source_commit,
    }


def _validate_output_path(path: Path, *, payload_root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if resolved.exists():
        raise PortableArchiveError(f"refusing to overwrite existing {label}: {resolved}")
    try:
        resolved.relative_to(payload_root)
    except ValueError:
        pass
    else:
        raise PortableArchiveError(f"{label} must be outside the payload root")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_archive(
    payload_root: Path,
    archive_path: Path,
    *,
    records: list[_FileRecord],
    root_directory: str,
    date_time: tuple[int, int, int, int, int, int],
) -> None:
    with ZipFile(
        archive_path,
        mode="x",
        compression=ZIP_DEFLATED,
        compresslevel=PORTABLE_ARCHIVE_COMPRESSION_LEVEL,
        allowZip64=True,
        strict_timestamps=True,
    ) as archive:
        archive.comment = PORTABLE_ARCHIVE_COMMENT
        for record in records:
            relative = str(record["path"])
            source = payload_root / PurePosixPath(relative)
            try:
                payload = source.read_bytes()
            except OSError as exc:
                raise PortableArchiveError(f"could not read payload file: {relative}") from exc
            if len(payload) != record["size"] or _sha256_bytes(payload) != record["sha256"]:
                raise PortableArchiveError(f"payload changed while archiving: {relative}")
            info = ZipInfo(f"{root_directory}/{relative}", date_time=date_time)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = PORTABLE_ARCHIVE_EXTERNAL_ATTR
            info.extra = b""
            info.comment = b""
            archive.writestr(
                info,
                payload,
                compress_type=ZIP_DEFLATED,
                compresslevel=PORTABLE_ARCHIVE_COMPRESSION_LEVEL,
            )


def _publish_file_no_replace(staging: Path, destination: Path) -> None:
    """Publish one same-filesystem file without a check-then-replace race."""

    try:
        os.link(staging, destination)
    except FileExistsError as exc:
        raise PortableArchiveError(
            f"refusing to overwrite concurrently created output: {destination}"
        ) from exc
    except OSError as exc:
        raise PortableArchiveError(f"could not publish output: {destination}") from exc


def _file_identity(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise PortableArchiveError(f"could not identify published output: {path}") from exc
    return int(stat.st_dev), int(stat.st_ino)


def _unlink_if_identity(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        current = path.stat()
        if (int(current.st_dev), int(current.st_ino)) == identity:
            path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def build_portable_archive(
    payload_root: Path,
    archive_path: Path,
    manifest_path: Path,
    *,
    source_date_epoch: int,
    root_directory: str = PORTABLE_ARCHIVE_ROOT,
) -> PortableArchiveResult:
    """Build a deterministic ZIP and canonical no-overwrite sidecar manifest."""

    try:
        root = payload_root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise PortableArchiveError("portable payload root does not exist") from exc
    if not root.is_dir():
        raise PortableArchiveError("portable payload root is not a directory")
    root_name = _validate_root_directory(root_directory)
    date_time = _zip_datetime(source_date_epoch)
    archive_final = _validate_output_path(
        archive_path, payload_root=root, label="portable archive"
    )
    manifest_final = _validate_output_path(
        manifest_path, payload_root=root, label="portable manifest"
    )
    if archive_final == manifest_final:
        raise PortableArchiveError("portable archive and manifest paths must differ")

    release_evidence = _release_evidence_descriptor(root)
    records = _collect_payload_files(root)
    payload_digest = _payload_sha256(records)
    payload_size = sum(int(record["size"]) for record in records)

    archive_temporary = archive_final.parent / f".{archive_final.name}.{uuid.uuid4().hex}.tmp"
    manifest_temporary = manifest_final.parent / f".{manifest_final.name}.{uuid.uuid4().hex}.tmp"
    archive_identity: tuple[int, int] | None = None
    manifest_identity: tuple[int, int] | None = None
    try:
        _write_archive(
            root,
            archive_temporary,
            records=records,
            root_directory=root_name,
            date_time=date_time,
        )
        archive_digest, archive_size = _sha256_file(archive_temporary)
        manifest = {
            "archive": {
                "file": archive_final.name,
                "sha256": archive_digest,
                "size": archive_size,
            },
            "compression": {
                "algorithm": "deflate",
                "level": PORTABLE_ARCHIVE_COMPRESSION_LEVEL,
            },
            "entries": records,
            "format": PORTABLE_ARCHIVE_FORMAT,
            "payload": {
                "file_count": len(records),
                "sha256": payload_digest,
                "size": payload_size,
            },
            "release_evidence": release_evidence,
            "root_directory": root_name,
            "schema_version": PORTABLE_ARCHIVE_SCHEMA_VERSION,
            "source_date_epoch": source_date_epoch,
        }
        manifest_temporary.write_bytes(_canonical_json_bytes(manifest))
        archive_identity = _file_identity(archive_temporary)
        _publish_file_no_replace(archive_temporary, archive_final)
        manifest_staging_identity = _file_identity(manifest_temporary)
        _publish_file_no_replace(manifest_temporary, manifest_final)
        manifest_identity = manifest_staging_identity
    except (OSError, BadZipFile) as exc:
        raise PortableArchiveError(f"could not build portable archive: {exc}") from exc
    finally:
        if archive_identity is not None and manifest_identity is None:
            _unlink_if_identity(archive_final, archive_identity)
        archive_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)

    try:
        return verify_portable_archive(archive_final, manifest_final)
    except PortableArchiveError:
        _unlink_if_identity(archive_final, archive_identity)
        _unlink_if_identity(manifest_final, manifest_identity)
        raise


def _read_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PortableArchiveError("could not read portable manifest") from exc
    if not raw or len(raw) > PORTABLE_ARCHIVE_MAX_MANIFEST_BYTES:
        raise PortableArchiveError("portable manifest exceeds the byte budget")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortableArchiveError("portable manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or _canonical_json_bytes(value) != raw:
        raise PortableArchiveError("portable manifest is not canonical JSON")
    return value, raw


def _validated_manifest(path: Path, *, archive_name: str) -> dict[str, Any]:
    manifest, _raw = _read_manifest(path)
    expected_fields = {
        "archive",
        "compression",
        "entries",
        "format",
        "payload",
        "release_evidence",
        "root_directory",
        "schema_version",
        "source_date_epoch",
    }
    if set(manifest) != expected_fields:
        raise PortableArchiveError("portable manifest fields are invalid")
    if (
        manifest["format"] != PORTABLE_ARCHIVE_FORMAT
        or manifest["schema_version"] != PORTABLE_ARCHIVE_SCHEMA_VERSION
    ):
        raise PortableArchiveError("portable manifest schema is unsupported")
    root_directory = _validate_root_directory(manifest["root_directory"])
    _zip_datetime(manifest["source_date_epoch"])

    archive = manifest["archive"]
    if not isinstance(archive, dict) or set(archive) != {"file", "sha256", "size"}:
        raise PortableArchiveError("portable archive descriptor is invalid")
    if archive["file"] != archive_name:
        raise PortableArchiveError("portable manifest names a different archive")
    if not isinstance(archive["sha256"], str) or _HASH_RE.fullmatch(archive["sha256"]) is None:
        raise PortableArchiveError("portable archive digest is invalid")
    if isinstance(archive["size"], bool) or not isinstance(archive["size"], int) or archive["size"] <= 0:
        raise PortableArchiveError("portable archive size is invalid")

    compression = manifest["compression"]
    if compression != {"algorithm": "deflate", "level": PORTABLE_ARCHIVE_COMPRESSION_LEVEL}:
        raise PortableArchiveError("portable archive compression is unsupported")

    entries = manifest["entries"]
    if not isinstance(entries, list) or not entries or len(entries) > PORTABLE_ARCHIVE_MAX_ENTRIES:
        raise PortableArchiveError("portable manifest entry list is invalid")
    validated_entries: list[_FileRecord] = []
    seen_casefold: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size"}:
            raise PortableArchiveError("portable manifest entry is invalid")
        name = _validate_relative_path(entry["path"], label="portable manifest path")
        folded = name.casefold()
        if folded in seen_casefold:
            raise PortableArchiveError(f"portable manifest path collision: {name}")
        seen_casefold.add(folded)
        digest = entry["sha256"]
        size = entry["size"]
        if not isinstance(digest, str) or _HASH_RE.fullmatch(digest) is None:
            raise PortableArchiveError(f"portable manifest digest is invalid: {name}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise PortableArchiveError(f"portable manifest size is invalid: {name}")
        validated_entries.append({"path": name, "sha256": digest, "size": size})
    if validated_entries != sorted(validated_entries, key=lambda entry: str(entry["path"])):
        raise PortableArchiveError("portable manifest entries are not sorted")

    payload = manifest["payload"]
    total_size = sum(int(entry["size"]) for entry in validated_entries)
    if total_size > PORTABLE_ARCHIVE_MAX_PAYLOAD_BYTES:
        raise PortableArchiveError("portable manifest exceeds the byte budget")
    expected_payload = {
        "file_count": len(validated_entries),
        "sha256": _payload_sha256(validated_entries),
        "size": total_size,
    }
    if payload != expected_payload:
        raise PortableArchiveError("portable payload descriptor is invalid")

    evidence = manifest["release_evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256", "source_commit"}:
        raise PortableArchiveError("portable release evidence descriptor is invalid")
    evidence_path = _validate_relative_path(evidence["path"], label="release evidence path")
    if evidence_path != f"{EVIDENCE_DIRECTORY_NAME}/release-evidence.json":
        raise PortableArchiveError("portable release evidence path is invalid")
    if not isinstance(evidence["sha256"], str) or _HASH_RE.fullmatch(evidence["sha256"]) is None:
        raise PortableArchiveError("portable release evidence digest is invalid")
    if not isinstance(evidence["source_commit"], str) or _COMMIT_RE.fullmatch(evidence["source_commit"]) is None:
        raise PortableArchiveError("portable source commit is invalid")
    entry_by_path = {str(entry["path"]): entry for entry in validated_entries}
    launcher_entry = entry_by_path.get("ArchMeshRubbing.exe")
    if launcher_entry is None or int(launcher_entry["size"]) <= 0:
        raise PortableArchiveError(
            "portable archive does not contain the Windows launcher"
        )
    evidence_entry = entry_by_path.get(evidence_path)
    if evidence_entry is None or evidence_entry["sha256"] != evidence["sha256"]:
        raise PortableArchiveError("portable archive does not bind its release evidence")
    manifest["root_directory"] = root_directory
    manifest["entries"] = validated_entries
    return manifest


def _hash_zip_member(
    archive: ZipFile,
    info: ZipInfo,
    *,
    capture: bool,
) -> tuple[str, int, bytes | None]:
    digest = hashlib.sha256()
    size = 0
    captured: bytearray | None = bytearray() if capture else None
    try:
        with archive.open(info, "r") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
                if captured is not None:
                    captured.extend(chunk)
    except (OSError, BadZipFile, RuntimeError) as exc:
        raise PortableArchiveError(f"could not read portable member: {info.filename}") from exc
    return digest.hexdigest(), size, bytes(captured) if captured is not None else None


def verify_portable_archive(archive_path: Path, manifest_path: Path) -> PortableArchiveResult:
    """Verify archive bytes, metadata, paths, release evidence, and every file."""

    archive_file = archive_path.expanduser().resolve(strict=False)
    manifest_file = manifest_path.expanduser().resolve(strict=False)
    if not archive_file.is_file() or not manifest_file.is_file():
        raise PortableArchiveError("portable archive and manifest must both exist")
    manifest = _validated_manifest(manifest_file, archive_name=archive_file.name)
    archive_digest, archive_size = _sha256_file(archive_file)
    if manifest["archive"] != {
        "file": archive_file.name,
        "sha256": archive_digest,
        "size": archive_size,
    }:
        raise PortableArchiveError("portable archive hash or size does not match manifest")

    records = manifest["entries"]
    root_directory = str(manifest["root_directory"])
    expected_names = [f"{root_directory}/{entry['path']}" for entry in records]
    date_time = _zip_datetime(manifest["source_date_epoch"])
    evidence_bytes: bytes | None = None
    try:
        with ZipFile(archive_file, mode="r", allowZip64=True) as archive:
            if archive.comment != PORTABLE_ARCHIVE_COMMENT:
                raise PortableArchiveError("portable archive comment is invalid")
            infos = archive.infolist()
            if [info.filename for info in infos] != expected_names:
                raise PortableArchiveError("portable archive members do not match manifest")
            for info, record in zip(infos, records, strict=True):
                expected_flag_bits = 0x800 if not info.filename.isascii() else 0
                if (
                    info.is_dir()
                    or info.compress_type != ZIP_DEFLATED
                    or info.date_time != date_time
                    or info.create_system != 0
                    or info.external_attr != PORTABLE_ARCHIVE_EXTERNAL_ATTR
                    or info.flag_bits != expected_flag_bits
                    or info.extra
                    or info.comment
                ):
                    raise PortableArchiveError(
                        f"portable archive member metadata is invalid: {info.filename}"
                    )
                if info.file_size != record["size"]:
                    raise PortableArchiveError(
                        f"portable archive member size is invalid: {info.filename}"
                    )
                is_evidence = record["path"] == manifest["release_evidence"]["path"]
                digest, size, captured = _hash_zip_member(
                    archive,
                    info,
                    capture=is_evidence,
                )
                if digest != record["sha256"] or size != record["size"]:
                    raise PortableArchiveError(
                        f"portable archive member hash mismatch: {info.filename}"
                    )
                if is_evidence:
                    evidence_bytes = captured
    except BadZipFile as exc:
        raise PortableArchiveError("portable archive is not a valid ZIP") from exc

    if evidence_bytes is None:
        raise PortableArchiveError("portable archive release evidence is unreadable")
    try:
        evidence = json.loads(evidence_bytes.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortableArchiveError("archived release evidence is invalid") from exc
    if (
        not isinstance(evidence, dict)
        or _canonical_json_bytes(evidence) != evidence_bytes
        or evidence.get("source_commit") != manifest["release_evidence"]["source_commit"]
    ):
        raise PortableArchiveError("archived release evidence source identity is invalid")

    payload = manifest["payload"]
    return PortableArchiveResult(
        archive_sha256=archive_digest,
        archive_size=archive_size,
        file_count=int(payload["file_count"]),
        payload_sha256=str(payload["sha256"]),
        payload_size=int(payload["size"]),
        source_commit=str(manifest["release_evidence"]["source_commit"]),
    )


def extract_portable_archive(
    archive_path: Path,
    manifest_path: Path,
    destination: Path,
) -> PortableArchiveResult:
    """Verify fully, then extract atomically while stripping the root directory."""

    result = verify_portable_archive(archive_path, manifest_path)
    archive_file = archive_path.expanduser().resolve(strict=True)
    manifest_file = manifest_path.expanduser().resolve(strict=True)
    manifest = _validated_manifest(manifest_file, archive_name=archive_file.name)
    archive_digest, archive_size = _sha256_file(archive_file)
    if (
        manifest["archive"]["sha256"] != result.archive_sha256
        or manifest["archive"]["size"] != result.archive_size
        or archive_digest != result.archive_sha256
        or archive_size != result.archive_size
        or manifest["payload"]["sha256"] != result.payload_sha256
        or manifest["release_evidence"]["source_commit"] != result.source_commit
    ):
        raise PortableArchiveError("portable inputs changed after verification")
    final = destination.expanduser().resolve(strict=False)
    if final.exists():
        raise PortableArchiveError(f"refusing to reuse extraction destination: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.{uuid.uuid4().hex}.tmp"
    if staging.exists():
        raise PortableArchiveError("portable extraction staging path already exists")
    staging.mkdir()
    root_directory = str(manifest["root_directory"])
    try:
        with ZipFile(archive_file, mode="r", allowZip64=True) as archive:
            infos = {info.filename: info for info in archive.infolist()}
            for record in manifest["entries"]:
                relative = str(record["path"])
                info = infos[f"{root_directory}/{relative}"]
                target = staging.joinpath(*PurePosixPath(relative).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with archive.open(info, "r") as source, target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if digest.hexdigest() != record["sha256"] or size != record["size"]:
                    raise PortableArchiveError(f"extracted payload mismatch: {relative}")
        extracted_records = _collect_payload_files(staging)
        if extracted_records != manifest["entries"]:
            raise PortableArchiveError("extracted payload contains unexpected bytes")
        os.replace(staging, final)
    except (OSError, BadZipFile, KeyError) as exc:
        raise PortableArchiveError(f"could not extract portable archive: {exc}") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return result


__all__ = [
    "PORTABLE_ARCHIVE_FORMAT",
    "PORTABLE_ARCHIVE_ROOT",
    "PORTABLE_ARCHIVE_SCHEMA_VERSION",
    "PortableArchiveError",
    "PortableArchiveResult",
    "build_portable_archive",
    "extract_portable_archive",
    "verify_portable_archive",
]
