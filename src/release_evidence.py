"""Deterministic, fail-closed evidence for a frozen Windows payload.

The SPDX document inventories bundled Python distributions.  A separate
SHA-256 manifest inventories every payload file because SPDX packages marked
``filesAnalyzed=false`` must not claim file-level analysis.  The evidence
directory is intentionally excluded from the payload manifest and is instead
bound by ``release-evidence.json`` to avoid a self-referential hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import compat32
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any


EVIDENCE_SCHEMA_VERSION = "1.0.0"
BUILD_MANIFEST_SCHEMA_VERSION = "1.2.0"
EVIDENCE_DIRECTORY_NAME = "release-evidence"
EVIDENCE_FILES = (
    "payload-manifest.json",
    "sbom.spdx.json",
    "third-party-notices.json",
    "THIRD_PARTY_NOTICES.md",
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_CHANNEL_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,31}$")
_INSTALLER_MANAGED_RE = re.compile(r"^unins[0-9]{3}\.(?:dat|exe|msg)$")
_PIN_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)==(?P<version>[^;\s]+)$"
)
_HASHED_PIN_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)==(?P<version>[^;\s]+)"
    r"\s+--hash=sha256:(?P<sha256>[0-9a-f]{64})$"
)


class ReleaseEvidenceError(RuntimeError):
    """Raised when release evidence cannot be generated or verified safely."""


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Stable summary returned after a complete evidence verification."""

    file_count: int
    package_count: int
    payload_sha256: str
    total_size: int

    def detail(self) -> str:
        return (
            f"payload={self.payload_sha256}, files={self.file_count}, "
            f"bytes={self.total_size}, runtime_packages={self.package_count}"
        )


@dataclass(frozen=True, slots=True)
class BuildContext:
    version: str
    channel: str
    commit: str
    build_manifest_path: str
    build_manifest_sha256: str
    runtime_lock_path: str
    runtime_lock_sha256: str
    runtime_pins: dict[str, tuple[str, str]]
    wheel_lock_path: str
    wheel_lock_sha256: str
    wheel_pins: dict[str, tuple[str, str, str]]
    license_policy_path: str
    license_policy_sha256: str
    license_policy: dict[str, Any]


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one evidence document without platform-dependent whitespace."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ReleaseEvidenceError(f"could not hash payload file: {path}") from exc
    return digest.hexdigest()


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _validate_relative_path(value: str, *, label: str) -> str:
    if not value or "\\" in value:
        raise ReleaseEvidenceError(f"{label} is not a portable relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseEvidenceError(f"{label} is not a safe relative path")
    normalized = path.as_posix()
    if normalized != value:
        raise ReleaseEvidenceError(f"{label} is not normalized")
    return value


def _read_limited(path: Path, *, label: str, limit: int = 8 * 1024 * 1024) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ReleaseEvidenceError(f"{label} is missing or unreadable") from exc
    if size <= 0 or size > limit:
        raise ReleaseEvidenceError(f"{label} size is invalid")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ReleaseEvidenceError(f"{label} is unreadable") from exc


def _load_canonical_json(
    path: Path,
    *,
    label: str,
    allow_trailing_newline: bool = False,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_limited(path, label=label)
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseEvidenceError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseEvidenceError(f"{label} must be a JSON object")
    canonical = canonical_json_bytes(value)
    accepted = {canonical}
    if allow_trailing_newline:
        accepted.add(canonical + b"\n")
    if raw not in accepted:
        raise ReleaseEvidenceError(f"{label} is not canonical JSON")
    return value, raw


def parse_exact_lock(path: Path) -> tuple[dict[str, tuple[str, str]], bytes]:
    """Parse the source/frozen runtime lock and reject non-exact entries."""

    raw = _read_limited(path, label="runtime dependency lock", limit=256 * 1024)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ReleaseEvidenceError("runtime dependency lock is not UTF-8") from exc
    pins: dict[str, tuple[str, str]] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN_RE.fullmatch(line)
        if match is None:
            raise ReleaseEvidenceError(
                f"runtime dependency lock line {line_number} is not an exact pin"
            )
        name = match.group("name")
        key = _canonical_name(name)
        if key in pins:
            raise ReleaseEvidenceError(f"runtime dependency lock repeats {name}")
        pins[key] = (name, match.group("version"))
    if not pins:
        raise ReleaseEvidenceError("runtime dependency lock has no pins")
    return pins, raw


def parse_hashed_lock(
    path: Path,
) -> tuple[dict[str, tuple[str, str, str]], bytes]:
    """Parse the flattened Windows wheel lock used by hash-checking pip."""

    raw = _read_limited(path, label="Windows wheel lock", limit=512 * 1024)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ReleaseEvidenceError("Windows wheel lock is not UTF-8") from exc
    options: set[str] = set()
    pins: dict[str, tuple[str, str, str]] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line in {"--require-hashes", "--only-binary=:all:"}:
            if line in options:
                raise ReleaseEvidenceError(f"Windows wheel lock repeats {line}")
            options.add(line)
            continue
        match = _HASHED_PIN_RE.fullmatch(line)
        if match is None:
            raise ReleaseEvidenceError(
                f"Windows wheel lock line {line_number} is not one exact hashed pin"
            )
        name = match.group("name")
        key = _canonical_name(name)
        if key in pins:
            raise ReleaseEvidenceError(f"Windows wheel lock repeats {name}")
        pins[key] = (name, match.group("version"), match.group("sha256"))
    required_options = {"--require-hashes", "--only-binary=:all:"}
    if options != required_options:
        raise ReleaseEvidenceError(
            "Windows wheel lock must enable --require-hashes and --only-binary=:all:"
        )
    if not pins:
        raise ReleaseEvidenceError("Windows wheel lock has no pins")
    return pins, raw


def _payload_relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ReleaseEvidenceError("payload path escaped its root") from exc


def _find_unique_file(root: Path, suffix: tuple[str, ...], *, label: str) -> Path:
    matches: list[Path] = []
    for path in root.rglob(suffix[-1]):
        if path.is_symlink():
            raise ReleaseEvidenceError(f"payload contains a symbolic link: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if tuple(relative.parts[-len(suffix) :]) == suffix:
            matches.append(path)
    if len(matches) != 1:
        raise ReleaseEvidenceError(
            f"payload must contain exactly one {label}; found {len(matches)}"
        )
    return matches[0]


def _load_license_policy(path: Path) -> tuple[dict[str, Any], bytes]:
    value, raw = _load_canonical_json(
        path,
        label="runtime license policy",
        allow_trailing_newline=True,
    )
    if set(value) != {"packages", "schema_version"}:
        raise ReleaseEvidenceError("runtime license policy fields are invalid")
    if value["schema_version"] != "1.0.0" or not isinstance(value["packages"], dict):
        raise ReleaseEvidenceError("runtime license policy schema is unsupported")
    for key, package in value["packages"].items():
        if not isinstance(key, str) or _canonical_name(key) != key:
            raise ReleaseEvidenceError("runtime license policy package key is invalid")
        if not isinstance(package, dict) or set(package) != {
            "fallback_license_files",
            "version",
        }:
            raise ReleaseEvidenceError(f"runtime license policy for {key} is invalid")
        if not isinstance(package["version"], str) or not package["version"]:
            raise ReleaseEvidenceError(f"runtime license policy version for {key} is invalid")
        fallbacks = package["fallback_license_files"]
        if not isinstance(fallbacks, list) or not fallbacks:
            raise ReleaseEvidenceError(f"runtime license policy for {key} has no fallback")
        for fallback in fallbacks:
            if not isinstance(fallback, dict) or set(fallback) != {
                "path",
                "sha256",
                "source_archive",
                "source_archive_sha256",
                "source_path",
            }:
                raise ReleaseEvidenceError(f"runtime license fallback for {key} is invalid")
            _validate_relative_path(str(fallback["path"]), label="fallback license path")
            _validate_relative_path(
                str(fallback["source_path"]), label="fallback source path"
            )
            if not isinstance(fallback["source_archive"], str) or not fallback[
                "source_archive"
            ]:
                raise ReleaseEvidenceError("fallback source archive is invalid")
            for field in ("sha256", "source_archive_sha256"):
                if not isinstance(fallback[field], str) or _HASH_RE.fullmatch(
                    fallback[field]
                ) is None:
                    raise ReleaseEvidenceError(f"fallback {field} is invalid")
    return value, raw


def _load_build_context(payload_root: Path) -> BuildContext:
    manifest_path = _find_unique_file(
        payload_root,
        ("resources", "build_info.json"),
        label="frozen build manifest",
    )
    runtime_lock_path = _find_unique_file(
        payload_root,
        ("requirements", "runtime-py312.lock"),
        label="runtime dependency lock",
    )
    wheel_lock_path = _find_unique_file(
        payload_root,
        ("requirements", "windows-py312-x64-hashed.lock"),
        label="Windows wheel lock",
    )
    policy_path = _find_unique_file(
        payload_root,
        ("requirements", "runtime-license-policy.json"),
        label="runtime license policy",
    )
    manifest, manifest_raw = _load_canonical_json(
        manifest_path, label="frozen build manifest"
    )
    if set(manifest) != {
        "channel",
        "commit",
        "dependency_lock_sha256",
        "schema_version",
        "source_tree",
        "version",
        "windows_wheel_lock_sha256",
    }:
        raise ReleaseEvidenceError("frozen build manifest fields are invalid")
    for field in ("dependency_lock_sha256", "windows_wheel_lock_sha256"):
        if not isinstance(manifest[field], str) or _HASH_RE.fullmatch(manifest[field]) is None:
            raise ReleaseEvidenceError(f"frozen build manifest {field} is invalid")
    if not isinstance(manifest["commit"], str) or _COMMIT_RE.fullmatch(
        manifest["commit"]
    ) is None:
        raise ReleaseEvidenceError("frozen build manifest commit is invalid")
    for field in ("channel", "version"):
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise ReleaseEvidenceError(f"frozen build manifest {field} is invalid")
    if _CHANNEL_RE.fullmatch(manifest["channel"]) is None:
        raise ReleaseEvidenceError("frozen build manifest channel is invalid")
    if manifest["schema_version"] != BUILD_MANIFEST_SCHEMA_VERSION:
        raise ReleaseEvidenceError("frozen build manifest schema is unsupported")
    if manifest["source_tree"] not in {"clean", "dirty", "unknown"}:
        raise ReleaseEvidenceError("frozen build manifest source tree state is invalid")

    runtime_pins, runtime_raw = parse_exact_lock(runtime_lock_path)
    wheel_pins, wheel_raw = parse_hashed_lock(wheel_lock_path)
    runtime_sha256 = _sha256_bytes(runtime_raw)
    wheel_sha256 = _sha256_bytes(wheel_raw)
    if manifest["dependency_lock_sha256"] != runtime_sha256:
        raise ReleaseEvidenceError("runtime dependency lock does not match build manifest")
    if manifest["windows_wheel_lock_sha256"] != wheel_sha256:
        raise ReleaseEvidenceError("Windows wheel lock does not match build manifest")
    for key, (name, version) in runtime_pins.items():
        wheel_pin = wheel_pins.get(key)
        if wheel_pin is None or wheel_pin[1] != version:
            raise ReleaseEvidenceError(
                f"Windows wheel lock does not match runtime pin {name}=={version}"
            )

    policy, policy_raw = _load_license_policy(policy_path)
    return BuildContext(
        version=manifest["version"],
        channel=manifest["channel"],
        commit=manifest["commit"],
        build_manifest_path=_payload_relative(payload_root, manifest_path),
        build_manifest_sha256=_sha256_bytes(manifest_raw),
        runtime_lock_path=_payload_relative(payload_root, runtime_lock_path),
        runtime_lock_sha256=runtime_sha256,
        runtime_pins=runtime_pins,
        wheel_lock_path=_payload_relative(payload_root, wheel_lock_path),
        wheel_lock_sha256=wheel_sha256,
        wheel_pins=wheel_pins,
        license_policy_path=_payload_relative(payload_root, policy_path),
        license_policy_sha256=_sha256_bytes(policy_raw),
        license_policy=policy,
    )


def _payload_file_records(
    payload_root: Path,
    *,
    allow_installer_managed: bool = False,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_casefold: set[str] = set()
    for path in payload_root.rglob("*"):
        relative = path.relative_to(payload_root)
        if relative.parts and relative.parts[0] == EVIDENCE_DIRECTORY_NAME:
            continue
        if (
            allow_installer_managed
            and len(relative.parts) == 1
            and _INSTALLER_MANAGED_RE.fullmatch(relative.name) is not None
        ):
            continue
        if path.is_symlink():
            raise ReleaseEvidenceError(
                f"payload contains a symbolic link: {relative.as_posix()}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ReleaseEvidenceError(
                f"payload contains an unsupported filesystem entry: {relative.as_posix()}"
            )
        name = _validate_relative_path(relative.as_posix(), label="payload file path")
        folded = name.casefold()
        if folded in seen_casefold:
            raise ReleaseEvidenceError(f"payload has a case-insensitive path collision: {name}")
        seen_casefold.add(folded)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ReleaseEvidenceError(f"payload file is unreadable: {name}") from exc
        records.append({"path": name, "sha256": _sha256_file(path), "size": size})
    records.sort(key=lambda item: str(item["path"]))
    if not records:
        raise ReleaseEvidenceError("payload has no files")
    return records


def _is_license_file(dist_info: Path, path: Path) -> bool:
    relative = path.relative_to(dist_info)
    parts = tuple(part.lower() for part in relative.parts)
    basename = parts[-1]
    return "licenses" in parts or basename.startswith(("license", "copying", "notice"))


def _decode_license_text(path: Path) -> tuple[bytes, str]:
    raw = _read_limited(path, label=f"license evidence {path}")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ReleaseEvidenceError(f"license evidence is not UTF-8: {path}") from exc
    return raw, text.replace("\r\n", "\n").replace("\r", "\n")


def _metadata_homepage(message: Any) -> str | None:
    candidates: list[str] = []
    for value in message.get_all("Project-URL", []):
        raw = str(value)
        if "," in raw:
            _label, url = raw.split(",", 1)
            candidates.append(url.strip())
    homepage = message.get("Home-page")
    if homepage:
        candidates.append(str(homepage).strip())
    return next(
        (value for value in candidates if value.startswith(("https://", "http://"))),
        None,
    )


def _collect_runtime_packages(
    payload_root: Path,
    context: BuildContext,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    metadata_by_name: dict[str, tuple[Path, Any, bytes]] = {}
    for metadata_path in payload_root.rglob("METADATA"):
        if EVIDENCE_DIRECTORY_NAME in metadata_path.relative_to(payload_root).parts:
            continue
        if metadata_path.is_symlink() or metadata_path.parent.is_symlink():
            raise ReleaseEvidenceError("distribution metadata must not be a symbolic link")
        if not metadata_path.is_file() or not metadata_path.parent.name.endswith(".dist-info"):
            continue
        raw = _read_limited(metadata_path, label="distribution METADATA")
        try:
            message = BytesParser(policy=compat32).parsebytes(raw)
        except Exception as exc:
            raise ReleaseEvidenceError(
                f"could not parse distribution metadata: {metadata_path}"
            ) from exc
        name = message.get("Name")
        version = message.get("Version")
        if not name or not version:
            raise ReleaseEvidenceError("distribution metadata omits Name or Version")
        key = _canonical_name(str(name))
        if key in metadata_by_name:
            raise ReleaseEvidenceError(f"payload repeats distribution metadata for {name}")
        metadata_by_name[key] = (metadata_path, message, raw)

    expected = set(context.runtime_pins)
    if set(metadata_by_name) != expected:
        missing = sorted(expected - set(metadata_by_name))
        unexpected = sorted(set(metadata_by_name) - expected)
        raise ReleaseEvidenceError(
            "payload distribution metadata differs from runtime lock; "
            f"missing={missing}, unexpected={unexpected}"
        )

    policy_packages = context.license_policy["packages"]
    used_policy: set[str] = set()
    packages: list[dict[str, object]] = []
    license_texts: dict[str, str] = {}
    for key in sorted(expected):
        expected_name, expected_version = context.runtime_pins[key]
        metadata_path, message, metadata_raw = metadata_by_name[key]
        metadata_name = str(message.get("Name"))
        metadata_version = str(message.get("Version"))
        if _canonical_name(metadata_name) != key or metadata_version != expected_version:
            raise ReleaseEvidenceError(
                f"distribution metadata does not match {expected_name}=={expected_version}"
            )
        wheel_name, wheel_version, wheel_sha256 = context.wheel_pins[key]
        if wheel_version != expected_version:
            raise ReleaseEvidenceError(f"wheel version mismatch for {expected_name}")

        evidence: list[dict[str, object]] = []
        for path in sorted(metadata_path.parent.rglob("*")):
            if path.is_symlink():
                raise ReleaseEvidenceError("license evidence must not be a symbolic link")
            if not path.is_file() or not _is_license_file(metadata_path.parent, path):
                continue
            raw, text = _decode_license_text(path)
            relative = _payload_relative(payload_root, path)
            evidence.append(
                {
                    "origin": "wheel-dist-info",
                    "path": relative,
                    "sha256": _sha256_bytes(raw),
                    "size": len(raw),
                }
            )
            license_texts[relative] = text

        if not evidence:
            fallback_policy = policy_packages.get(key)
            if not isinstance(fallback_policy, dict):
                raise ReleaseEvidenceError(
                    f"{expected_name} has no bundled license text or reviewed fallback"
                )
            if fallback_policy["version"] != expected_version:
                raise ReleaseEvidenceError(
                    f"license fallback version mismatch for {expected_name}"
                )
            used_policy.add(key)
            for fallback in fallback_policy["fallback_license_files"]:
                policy_relative = str(fallback["path"])
                path = _find_unique_file(
                    payload_root,
                    tuple(Path(policy_relative).parts),
                    label=f"reviewed license fallback {policy_relative}",
                )
                raw, text = _decode_license_text(path)
                if _sha256_bytes(raw) != fallback["sha256"]:
                    raise ReleaseEvidenceError(
                        f"reviewed license fallback hash mismatch: {policy_relative}"
                    )
                payload_relative = _payload_relative(payload_root, path)
                evidence.append(
                    {
                        "origin": "reviewed-source-fallback",
                        "path": payload_relative,
                        "policy_path": policy_relative,
                        "sha256": fallback["sha256"],
                        "size": len(raw),
                        "source_archive": fallback["source_archive"],
                        "source_archive_sha256": fallback[
                            "source_archive_sha256"
                        ],
                        "source_path": fallback["source_path"],
                    }
                )
                license_texts[payload_relative] = text

        expression = message.get("License-Expression")
        license_expression = str(expression).strip() if expression else None
        if license_expression is not None and (
            not license_expression or len(license_expression) > 1024
        ):
            raise ReleaseEvidenceError(
                f"License-Expression metadata is invalid for {expected_name}"
            )
        classifiers = sorted(
            str(value)
            for value in message.get_all("Classifier", [])
            if str(value).startswith("License ::")
        )
        package: dict[str, object] = {
            "canonical_name": key,
            "display_name": metadata_name,
            "homepage": _metadata_homepage(message),
            "legacy_license_present": bool(message.get("License")),
            "license_classifiers": classifiers,
            "license_evidence": evidence,
            "license_expression": license_expression,
            "metadata_path": _payload_relative(payload_root, metadata_path),
            "metadata_sha256": _sha256_bytes(metadata_raw),
            "version": metadata_version,
            "wheel_name": wheel_name,
            "wheel_sha256": wheel_sha256,
        }
        packages.append(package)

    stale_policy = sorted(set(policy_packages) - used_policy)
    if stale_policy:
        raise ReleaseEvidenceError(
            "runtime license policy contains unused fallbacks: " + ", ".join(stale_policy)
        )
    return packages, license_texts


def _normalize_created_at(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseEvidenceError("SPDX creation time is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReleaseEvidenceError("SPDX creation time is not ISO 8601") from exc
    if parsed.tzinfo is None:
        raise ReleaseEvidenceError("SPDX creation time must include a timezone")
    normalized = parsed.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")


def _payload_manifest(
    context: BuildContext,
    files: list[dict[str, object]],
) -> dict[str, object]:
    total_size = sum(int(item["size"]) for item in files)
    payload_sha256 = _sha256_bytes(canonical_json_bytes(files))
    return {
        "application": {"name": "ArchMeshRubbing", "version": context.version},
        "build": {
            "build_manifest_path": context.build_manifest_path,
            "build_manifest_sha256": context.build_manifest_sha256,
            "channel": context.channel,
            "license_policy_path": context.license_policy_path,
            "license_policy_sha256": context.license_policy_sha256,
            "runtime_lock_path": context.runtime_lock_path,
            "runtime_lock_sha256": context.runtime_lock_sha256,
            "source_commit": context.commit,
            "windows_wheel_lock_path": context.wheel_lock_path,
            "windows_wheel_lock_sha256": context.wheel_lock_sha256,
        },
        "evidence_directory_excluded": EVIDENCE_DIRECTORY_NAME,
        "installed_files_excluded": [
            "uninsNNN.dat",
            "uninsNNN.exe",
            "uninsNNN.msg",
        ],
        "file_count": len(files),
        "files": files,
        "payload_sha256": payload_sha256,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "total_size": total_size,
    }


def _spdx_id(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9.-]+", "-", name).strip("-.")
    if not token:
        raise ReleaseEvidenceError("could not construct an SPDX identifier")
    return f"SPDXRef-Package-{token}"


def _spdx_document(
    context: BuildContext,
    *,
    created_at: str,
    payload_sha256: str,
    packages: list[dict[str, object]],
) -> dict[str, object]:
    root_id = _spdx_id("ArchMeshRubbing")
    spdx_packages: list[dict[str, object]] = [
        {
            "SPDXID": root_id,
            "comment": (
                "File-level SHA-256 evidence is in payload-manifest.json. "
                "Public binary distribution remains blocked pending license "
                "compatibility review."
            ),
            "copyrightText": "NOASSERTION",
            "downloadLocation": (
                "git+https://github.com/lzpxilfe/ArchMeshRubbing.git@"
                f"{context.commit}"
            ),
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "GPL-2.0-only",
            "name": "ArchMeshRubbing",
            "supplier": "NOASSERTION",
            "versionInfo": context.version,
        }
    ]
    relationships: list[dict[str, str]] = [
        {
            "relatedSpdxElement": root_id,
            "relationshipType": "DESCRIBES",
            "spdxElementId": "SPDXRef-DOCUMENT",
        }
    ]
    for package in packages:
        canonical_name = str(package["canonical_name"])
        package_id = _spdx_id(canonical_name)
        item: dict[str, object] = {
            "SPDXID": package_id,
            "checksums": [
                {
                    "algorithm": "SHA256",
                    "checksumValue": package["wheel_sha256"],
                }
            ],
            "comment": (
                f"Bundled metadata: {package['metadata_path']}; full license "
                "evidence is reproduced in THIRD_PARTY_NOTICES.md."
            ),
            "copyrightText": "NOASSERTION",
            "downloadLocation": (
                f"https://pypi.org/project/{package['display_name']}/"
                f"{package['version']}/"
            ),
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceLocator": (
                        f"pkg:pypi/{canonical_name}@{package['version']}"
                    ),
                    "referenceType": "purl",
                }
            ],
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": package["license_expression"] or "NOASSERTION",
            "name": package["display_name"],
            "supplier": "NOASSERTION",
            "versionInfo": package["version"],
        }
        if package["homepage"] is not None:
            item["homepage"] = package["homepage"]
        spdx_packages.append(item)
        relationships.append(
            {
                "relatedSpdxElement": package_id,
                "relationshipType": "CONTAINS",
                "spdxElementId": root_id,
            }
        )
    return {
        "SPDXID": "SPDXRef-DOCUMENT",
        "creationInfo": {
            "comment": (
                "The deterministic timestamp is the source commit time supplied "
                "by the build workflow."
            ),
            "created": created_at,
            "creators": ["Tool: ArchMeshRubbing release-evidence/1.0.0"],
        },
        "dataLicense": "CC0-1.0",
        "documentDescribes": [root_id],
        "documentNamespace": (
            "https://github.com/lzpxilfe/ArchMeshRubbing/evidence/"
            f"{context.commit}/{payload_sha256}"
        ),
        "name": f"ArchMeshRubbing-{context.version}-Windows-x64",
        "packages": spdx_packages,
        "relationships": relationships,
        "spdxVersion": "SPDX-2.3",
    }


def _notices_document(
    context: BuildContext,
    *,
    created_at: str,
    packages: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "created": created_at,
        "license_policy_sha256": context.license_policy_sha256,
        "packages": packages,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "source_commit": context.commit,
        "windows_wheel_lock_sha256": context.wheel_lock_sha256,
    }


def _notices_markdown(
    context: BuildContext,
    *,
    packages: list[dict[str, object]],
    license_texts: dict[str, str],
) -> bytes:
    lines = [
        "# ArchMeshRubbing third-party notices",
        "",
        f"Source commit: `{context.commit}`  ",
        f"Windows wheel lock SHA-256: `{context.wheel_lock_sha256}`",
        "",
        "This file is mechanically generated from the metadata and license texts",
        "inside the frozen payload. It records evidence; it is not legal advice.",
        "Public binary distribution remains disabled until the repository and",
        "runtime license compatibility decision is complete.",
        "",
    ]
    for package in packages:
        lines.extend(
            [
                f"## {package['display_name']} {package['version']}",
                "",
                f"- Wheel SHA-256: `{package['wheel_sha256']}`",
                f"- Bundled metadata: `{package['metadata_path']}`",
                "- License-Expression: "
                + (
                    f"`{package['license_expression']}`"
                    if package["license_expression"]
                    else "not supplied by wheel metadata"
                ),
                "- Legacy license field present: "
                + ("yes" if package["legacy_license_present"] else "no"),
            ]
        )
        classifiers = package["license_classifiers"]
        if classifiers:
            lines.append("- License classifiers: " + "; ".join(classifiers))
        lines.append("")
        for evidence in package["license_evidence"]:
            path = str(evidence["path"])
            lines.extend(
                [
                    f"### `{path}`",
                    "",
                    f"SHA-256: `{evidence['sha256']}`; origin: "
                    f"`{evidence['origin']}`.",
                ]
            )
            if evidence["origin"] == "reviewed-source-fallback":
                lines.extend(
                    [
                        "",
                        f"Source archive: `{evidence['source_archive']}` "
                        f"(`{evidence['source_archive_sha256']}`), path "
                        f"`{evidence['source_path']}`.",
                    ]
                )
            lines.append("")
            text = license_texts[path]
            lines.extend("    " + line for line in text.split("\n"))
            lines.append("")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
    except FileExistsError as exc:
        raise ReleaseEvidenceError(f"refusing to overwrite evidence file: {path}") from exc
    except OSError as exc:
        raise ReleaseEvidenceError(f"could not write evidence file: {path}") from exc


def _expected_output_directory(payload_root: Path, output_dir: Path) -> tuple[Path, Path]:
    try:
        root = payload_root.resolve(strict=True)
    except OSError as exc:
        raise ReleaseEvidenceError("payload root does not exist") from exc
    if not root.is_dir() or payload_root.is_symlink():
        raise ReleaseEvidenceError("payload root must be a real directory")
    output = output_dir.resolve(strict=False)
    if output != root / EVIDENCE_DIRECTORY_NAME:
        raise ReleaseEvidenceError(
            f"evidence must be written to payload/{EVIDENCE_DIRECTORY_NAME}"
        )
    return root, output


def generate_release_evidence(
    payload_root: Path,
    output_dir: Path,
    *,
    created_at: str,
) -> VerificationResult:
    """Generate and immediately re-verify all evidence documents."""

    root, output = _expected_output_directory(payload_root, output_dir)
    if output.exists():
        raise ReleaseEvidenceError("release evidence directory already exists")
    normalized_created = _normalize_created_at(created_at)
    context = _load_build_context(root)
    files = _payload_file_records(root)
    manifest = _payload_manifest(context, files)
    packages, license_texts = _collect_runtime_packages(root, context)
    spdx = _spdx_document(
        context,
        created_at=normalized_created,
        payload_sha256=str(manifest["payload_sha256"]),
        packages=packages,
    )
    notices = _notices_document(
        context,
        created_at=normalized_created,
        packages=packages,
    )
    documents = {
        "payload-manifest.json": canonical_json_bytes(manifest),
        "sbom.spdx.json": canonical_json_bytes(spdx),
        "third-party-notices.json": canonical_json_bytes(notices),
        "THIRD_PARTY_NOTICES.md": _notices_markdown(
            context,
            packages=packages,
            license_texts=license_texts,
        ),
    }
    try:
        output.mkdir()
        for name in EVIDENCE_FILES:
            _write_exclusive(output / name, documents[name])
        index_records = [
            {
                "path": name,
                "sha256": _sha256_bytes(documents[name]),
                "size": len(documents[name]),
            }
            for name in EVIDENCE_FILES
        ]
        index = {
            "created": normalized_created,
            "evidence_files": index_records,
            "payload_sha256": manifest["payload_sha256"],
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "source_commit": context.commit,
        }
        _write_exclusive(output / "release-evidence.json", canonical_json_bytes(index))
        return verify_release_evidence(root, output)
    except Exception:
        if output.exists():
            shutil.rmtree(output)
        raise


def _verify_evidence_directory(output: Path) -> tuple[dict[str, Any], str]:
    if output.is_symlink() or not output.is_dir():
        raise ReleaseEvidenceError("release evidence directory is missing or linked")
    observed: set[str] = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ReleaseEvidenceError("release evidence contains a symbolic link")
        if path.is_dir():
            if path != output:
                raise ReleaseEvidenceError("release evidence contains a subdirectory")
            continue
        observed.add(path.relative_to(output).as_posix())
    expected = set(EVIDENCE_FILES) | {"release-evidence.json"}
    if observed != expected:
        raise ReleaseEvidenceError(
            f"release evidence file set differs; missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )
    index, _raw = _load_canonical_json(
        output / "release-evidence.json", label="release evidence index"
    )
    if set(index) != {
        "created",
        "evidence_files",
        "payload_sha256",
        "schema_version",
        "source_commit",
    } or index["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise ReleaseEvidenceError("release evidence index fields are invalid")
    created = _normalize_created_at(index["created"])
    if index["created"] != created:
        raise ReleaseEvidenceError("release evidence creation time is not normalized")
    records = index["evidence_files"]
    if not isinstance(records, list) or [item.get("path") for item in records] != list(
        EVIDENCE_FILES
    ):
        raise ReleaseEvidenceError("release evidence index ordering is invalid")
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size"}:
            raise ReleaseEvidenceError("release evidence index record is invalid")
        path = output / str(record["path"])
        raw = _read_limited(path, label=f"release evidence {record['path']}", limit=64 * 1024 * 1024)
        if record["size"] != len(raw) or record["sha256"] != _sha256_bytes(raw):
            raise ReleaseEvidenceError(
                f"release evidence hash or size mismatch: {record['path']}"
            )
    return index, created


def verify_release_evidence(
    payload_root: Path,
    evidence_dir: Path | None = None,
) -> VerificationResult:
    """Verify identity, payload bytes, SBOM, and notice derivation from scratch."""

    requested = evidence_dir or payload_root / EVIDENCE_DIRECTORY_NAME
    root, output = _expected_output_directory(payload_root, requested)
    index, created_at = _verify_evidence_directory(output)
    context = _load_build_context(root)
    if index["source_commit"] != context.commit:
        raise ReleaseEvidenceError("release evidence source commit does not match payload")

    manifest, _manifest_raw = _load_canonical_json(
        output / "payload-manifest.json", label="payload manifest"
    )
    files = _payload_file_records(root, allow_installer_managed=True)
    expected_manifest = _payload_manifest(context, files)
    if manifest != expected_manifest:
        raise ReleaseEvidenceError("payload manifest does not match actual payload bytes")
    if index["payload_sha256"] != manifest["payload_sha256"]:
        raise ReleaseEvidenceError("release evidence index does not bind the payload")

    packages, license_texts = _collect_runtime_packages(root, context)
    notices, _notices_raw = _load_canonical_json(
        output / "third-party-notices.json", label="third-party notices data"
    )
    expected_notices = _notices_document(
        context,
        created_at=created_at,
        packages=packages,
    )
    if notices != expected_notices:
        raise ReleaseEvidenceError("third-party notices do not match payload metadata")

    spdx, _spdx_raw = _load_canonical_json(
        output / "sbom.spdx.json", label="SPDX SBOM"
    )
    expected_spdx = _spdx_document(
        context,
        created_at=created_at,
        payload_sha256=str(manifest["payload_sha256"]),
        packages=packages,
    )
    if spdx != expected_spdx:
        raise ReleaseEvidenceError("SPDX SBOM does not match payload metadata")

    expected_markdown = _notices_markdown(
        context,
        packages=packages,
        license_texts=license_texts,
    )
    try:
        actual_markdown = (output / "THIRD_PARTY_NOTICES.md").read_bytes()
    except OSError as exc:
        raise ReleaseEvidenceError("third-party notice text is unreadable") from exc
    if actual_markdown != expected_markdown:
        raise ReleaseEvidenceError("third-party notice text does not match license evidence")

    return VerificationResult(
        file_count=int(manifest["file_count"]),
        package_count=len(packages),
        payload_sha256=str(manifest["payload_sha256"]),
        total_size=int(manifest["total_size"]),
    )
