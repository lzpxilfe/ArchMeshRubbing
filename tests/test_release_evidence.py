from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from src.release_evidence import (
    EVIDENCE_DIRECTORY_NAME,
    EVIDENCE_FILES,
    ReleaseEvidenceError,
    canonical_json_bytes,
    generate_release_evidence,
    parse_exact_lock,
    parse_hashed_lock,
    verify_release_evidence,
)
from tools.build_native import exact_lock_pins


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _synthetic_payload(root: Path) -> Path:
    payload = root / "payload"
    runtime_lock = _write(
        payload / "_internal" / "requirements" / "runtime-py312.lock",
        b"Alpha==1.0\nBeta==2.0\n",
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
            f"Beta==2.0 --hash=sha256:{'2' * 64}\n"
        ).encode("utf-8"),
    )
    fallback = b"Beta reviewed license text\n"
    fallback_path = "third_party_licenses/Beta-2.0-LICENSE.txt"
    _write(payload / fallback_path, fallback)
    policy = {
        "packages": {
            "beta": {
                "fallback_license_files": [
                    {
                        "path": fallback_path,
                        "sha256": hashlib.sha256(fallback).hexdigest(),
                        "source_archive": "beta-2.0.tar.gz",
                        "source_archive_sha256": "3" * 64,
                        "source_path": "beta-2.0/LICENSE",
                    }
                ],
                "version": "2.0",
            }
        },
        "schema_version": "1.0.0",
    }
    _write(
        payload / "_internal" / "requirements" / "runtime-license-policy.json",
        canonical_json_bytes(policy),
    )
    manifest = {
        "channel": "ci-smoke",
        "commit": "a" * 40,
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
        canonical_json_bytes(manifest),
    )
    _write(payload / "ArchMeshRubbing.exe", b"synthetic executable\n")
    _write(
        payload / "_internal" / "alpha-1.0.dist-info" / "METADATA",
        (
            "Metadata-Version: 2.4\n"
            "Name: Alpha\n"
            "Version: 1.0\n"
            "License-Expression: MIT\n"
            "Project-URL: Homepage, https://example.invalid/alpha\n"
            "License-File: LICENSE\n\n"
        ).encode("utf-8"),
    )
    _write(
        payload / "_internal" / "alpha-1.0.dist-info" / "licenses" / "LICENSE",
        b"Alpha license text\n",
    )
    _write(
        payload / "_internal" / "beta-2.0.dist-info" / "METADATA",
        (
            "Metadata-Version: 2.1\n"
            "Name: Beta\n"
            "Version: 2.0\n"
            "Classifier: License :: OSI Approved :: BSD License\n\n"
        ).encode("utf-8"),
    )
    return payload


def test_release_evidence_is_deterministic_and_derived_from_payload() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        payload_a = _synthetic_payload(base / "a")
        payload_b = _synthetic_payload(base / "b")
        created = "2026-07-13T09:10:11+09:00"
        result_a = generate_release_evidence(
            payload_a,
            payload_a / EVIDENCE_DIRECTORY_NAME,
            created_at=created,
        )
        result_b = generate_release_evidence(
            payload_b,
            payload_b / EVIDENCE_DIRECTORY_NAME,
            created_at=created,
        )

        assert result_a == result_b
        assert result_a.package_count == 2
        assert verify_release_evidence(payload_a) == result_a
        (payload_a / "unins000.exe").write_bytes(b"installer-managed\n")
        (payload_a / "unins000.dat").write_bytes(b"installer-managed\n")
        assert verify_release_evidence(payload_a) == result_a
        evidence_a = payload_a / EVIDENCE_DIRECTORY_NAME
        evidence_b = payload_b / EVIDENCE_DIRECTORY_NAME
        for name in (*EVIDENCE_FILES, "release-evidence.json"):
            assert (evidence_a / name).read_bytes() == (evidence_b / name).read_bytes()

        spdx = json.loads((evidence_a / "sbom.spdx.json").read_text("utf-8"))
        assert spdx["spdxVersion"] == "SPDX-2.3"
        assert spdx["creationInfo"]["created"] == "2026-07-13T00:10:11Z"
        assert len(spdx["packages"]) == 3
        notices = json.loads(
            (evidence_a / "third-party-notices.json").read_text("utf-8")
        )
        beta = next(item for item in notices["packages"] if item["canonical_name"] == "beta")
        assert beta["license_evidence"][0]["origin"] == "reviewed-source-fallback"
        markdown = (evidence_a / "THIRD_PARTY_NOTICES.md").read_text("utf-8")
        assert "Alpha license text" in markdown
        assert "Beta reviewed license text" in markdown


def test_release_evidence_detects_payload_and_evidence_tampering() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        payload = _synthetic_payload(Path(temporary))
        generate_release_evidence(
            payload,
            payload / EVIDENCE_DIRECTORY_NAME,
            created_at="2026-07-13T00:10:11Z",
        )
        (payload / "ArchMeshRubbing.exe").write_bytes(b"tampered\n")
        with pytest.raises(ReleaseEvidenceError, match="payload manifest"):
            verify_release_evidence(payload)

    with tempfile.TemporaryDirectory() as temporary:
        payload = _synthetic_payload(Path(temporary))
        generate_release_evidence(
            payload,
            payload / EVIDENCE_DIRECTORY_NAME,
            created_at="2026-07-13T00:10:11Z",
        )
        notice = payload / EVIDENCE_DIRECTORY_NAME / "THIRD_PARTY_NOTICES.md"
        notice.write_bytes(notice.read_bytes() + b"tampered\n")
        with pytest.raises(ReleaseEvidenceError, match="hash or size mismatch"):
            verify_release_evidence(payload)


def test_release_evidence_rejects_missing_license_and_unexpected_metadata() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        payload = _synthetic_payload(Path(temporary))
        (payload / "third_party_licenses" / "Beta-2.0-LICENSE.txt").unlink()
        with pytest.raises(ReleaseEvidenceError, match="fallback is missing"):
            generate_release_evidence(
                payload,
                payload / EVIDENCE_DIRECTORY_NAME,
                created_at="2026-07-13T00:10:11Z",
            )
        assert not (payload / EVIDENCE_DIRECTORY_NAME).exists()

    with tempfile.TemporaryDirectory() as temporary:
        payload = _synthetic_payload(Path(temporary))
        _write(
            payload / "_internal" / "gamma-3.0.dist-info" / "METADATA",
            b"Metadata-Version: 2.1\nName: Gamma\nVersion: 3.0\n\n",
        )
        with pytest.raises(ReleaseEvidenceError, match="differs from runtime lock"):
            generate_release_evidence(
                payload,
                payload / EVIDENCE_DIRECTORY_NAME,
                created_at="2026-07-13T00:10:11Z",
            )


def test_committed_windows_lock_is_flattened_exact_and_hash_checked() -> None:
    runtime, _runtime_raw = parse_exact_lock(
        ROOT / "requirements" / "runtime-py312.lock"
    )
    wheels, _wheel_raw = parse_hashed_lock(
        ROOT / "requirements" / "windows-py312-x64-hashed.lock"
    )
    assert set(runtime) <= set(wheels)
    assert len(runtime) == 10
    assert set(wheels) == {
        "altgraph",
        "numpy",
        "packaging",
        "pefile",
        "pillow",
        "pyinstaller",
        "pyinstaller-hooks-contrib",
        "pyopengl",
        "pyqt6",
        "pyqt6-qt6",
        "pyqt6-sip",
        "pywin32-ctypes",
        "rfc8785",
        "scipy",
        "setuptools",
        "shapely",
        "trimesh",
    }
    assert all(len(pin[2]) == 64 for pin in wheels.values())

    build_pins = exact_lock_pins(
        ROOT / "requirements" / "build-py312.lock",
        platform_name="win32",
    )
    assert set(wheels) == set(build_pins)
    assert {
        key: (name, version) for key, (name, version, _sha256) in wheels.items()
    } == build_pins

    policy = json.loads(
        (ROOT / "requirements" / "runtime-license-policy.json").read_text("utf-8")
    )
    fallback = policy["packages"]["pyopengl"]["fallback_license_files"][0]
    fallback_bytes = (ROOT / fallback["path"]).read_bytes()
    assert hashlib.sha256(fallback_bytes).hexdigest() == fallback["sha256"]
    assert fallback["source_archive_sha256"] == (
        "c4a02d6866b54eb119c8e9b3fb04fa835a95ab802dd96607ab4cdb0012df8335"
    )


def test_hashed_lock_rejects_missing_security_options_and_unhashed_lines() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        lock = Path(temporary) / "windows.lock"
        lock.write_text(
            f"Alpha==1.0 --hash=sha256:{'1' * 64}\n",
            encoding="utf-8",
        )
        with pytest.raises(ReleaseEvidenceError, match="must enable"):
            parse_hashed_lock(lock)

        lock.write_text(
            "--require-hashes\n"
            "--only-binary=:all:\n"
            "Alpha==1.0\n",
            encoding="utf-8",
        )
        with pytest.raises(ReleaseEvidenceError, match="not one exact hashed pin"):
            parse_hashed_lock(lock)
