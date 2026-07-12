from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import pytest

from src import __version__
from src import build_info
from tools.generate_build_info import (
    build_manifest,
    detect_source_tree,
    write_manifest,
)


def test_generated_manifest_binds_version_commit_channel_and_runtime_lock() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        lock = root / "runtime.lock"
        lock.write_bytes(b"numpy==2.4.3\n")
        manifest = build_manifest(
            channel="ci-smoke",
            commit="a" * 40,
            lock_path=lock,
            source_tree="dirty",
        )
        assert manifest == {
            "channel": "ci-smoke",
            "commit": "a" * 40,
            "dependency_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            "schema_version": "1.1.0",
            "source_tree": "dirty",
            "version": __version__,
        }

        output = root / "resources" / "build_info.json"
        write_manifest(output, manifest)
        assert json.loads(output.read_text(encoding="utf-8")) == manifest


def test_frozen_build_metadata_requires_and_validates_embedded_manifest() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        lock = root / "runtime-py312.lock"
        lock.write_bytes(b"pins\n")
        digest = hashlib.sha256(lock.read_bytes()).hexdigest()
        manifest_path = root / "build_info.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "channel": "ci-smoke",
                    "commit": "b" * 40,
                    "dependency_lock_sha256": digest,
                    "schema_version": "1.1.0",
                    "source_tree": "clean",
                    "version": __version__,
                }
            ),
            encoding="utf-8",
        )

        def resource_path(*parts: str) -> Path:
            if parts == build_info.FROZEN_BUILD_MANIFEST_PARTS:
                return manifest_path
            raise AssertionError(parts)

        with (
            patch.object(
                build_info,
                "runtime_lock",
                return_value=(lock, {}, digest),
            ),
            patch.object(build_info, "resource_path", side_effect=resource_path),
            patch.object(build_info.sys, "frozen", True, create=True),
        ):
            metadata = build_info.build_metadata()
            assert build_info._check_build_identity().startswith("channel=ci-smoke")
        assert metadata["manifest_present"] is True
        assert metadata["commit"] == "b" * 40
        assert metadata["source_tree"] == "clean"


def test_invalid_build_manifest_inputs_and_lock_binding_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        lock = root / "runtime.lock"
        lock.write_bytes(b"pins\n")
        with pytest.raises(ValueError, match="channel"):
            build_manifest(channel="Release Candidate", commit="a" * 40, lock_path=lock)
        with pytest.raises(ValueError, match="commit"):
            build_manifest(channel="stable", commit="not-a-hash", lock_path=lock)
        with pytest.raises(ValueError, match="source_tree"):
            build_manifest(
                channel="stable",
                commit="a" * 40,
                lock_path=lock,
                source_tree="maybe",
            )

        manifest_path = root / "build_info.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "channel": "stable",
                    "commit": "c" * 40,
                    "dependency_lock_sha256": "0" * 64,
                    "schema_version": "1.1.0",
                    "source_tree": "clean",
                    "version": __version__,
                }
            ),
            encoding="utf-8",
        )
        digest = hashlib.sha256(lock.read_bytes()).hexdigest()
        with (
            patch.object(
                build_info,
                "runtime_lock",
                return_value=(lock, {}, digest),
            ),
            patch.object(
                build_info,
                "resource_path",
                return_value=manifest_path,
            ),
            pytest.raises(RuntimeError, match="lock hash"),
        ):
            build_info.build_metadata()


def test_source_tree_detection_reports_clean_dirty_and_unknown() -> None:
    clean = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    dirty = subprocess.CompletedProcess([], 0, stdout=" M src/a.py\n", stderr="")
    with patch(
        "tools.generate_build_info.subprocess.run",
        return_value=clean,
    ):
        assert detect_source_tree(Path("/repo")) == "clean"
    with patch(
        "tools.generate_build_info.subprocess.run",
        return_value=dirty,
    ):
        assert detect_source_tree(Path("/repo")) == "dirty"
    with patch(
        "tools.generate_build_info.subprocess.run",
        side_effect=OSError("git unavailable"),
    ):
        assert detect_source_tree(Path("/repo")) == "unknown"
