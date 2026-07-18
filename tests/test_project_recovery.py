from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import textwrap
from unittest.mock import Mock, patch

import pytest

import src.core.project_recovery as project_recovery
from src.core.artifact_session import ArtifactSession
from src.core.mesh_loader import MeshLoader
from src.core.project_file import (
    load_artifact_session_project,
    save_artifact_project,
    save_artifact_session_project,
)
from src.core.project_recovery import (
    ProjectRecoveryError,
    discover_interrupted_project_saves,
    recover_interrupted_project_save,
)


_PLY_BYTES = textwrap.dedent(
    """\
    ply
    format ascii 1.0
    comment interrupted project recovery fixture
    element vertex 4
    property float x
    property float y
    property float z
    element face 4
    property list uchar int vertex_indices
    end_header
    0 0 0
    1 0 0
    0 1 0
    0 0 1
    3 0 2 1
    3 0 1 3
    3 1 2 3
    3 2 0 3
    """
).encode("ascii")


def _session(directory: Path) -> tuple[ArtifactSession, Path]:
    source = directory / "source" / "유물.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(_PLY_BYTES)
    mesh = MeshLoader(default_unit="mm").load(source, unit="mm")
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path=str(source),
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="project-recovery-test/1",
        operator="project-recovery-test",
        created_at="2026-07-14T00:00:00Z",
        document_id="artifact:project-recovery-test",
        metadata_revision_id="metadata:project-recovery-test",
        align_revision_id="align:project-recovery-test",
    )
    return session, source


def _valid_candidate(
    directory: Path,
    *,
    intended_name: str = "현장기록.amr",
    token: str = "recover1",
) -> tuple[ArtifactSession, Path, Path]:
    session, source = _session(directory)
    seed = directory / "seed.amr"
    save_artifact_session_project(seed, session)
    candidate = directory / f".{intended_name}.{token}.tmp"
    shutil.copyfile(seed, candidate)
    seed.unlink()
    source.unlink()
    return session, candidate, directory / intended_name


def test_discovery_is_folder_scoped_strict_bounded_and_newest_first(
    tmp_path: Path,
) -> None:
    older = tmp_path / ".older.amr.abc_def1.tmp"
    newer = tmp_path / ".새 기록.AMR.1234abcd.tmp"
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

    (tmp_path / ".wrong.txt.1234abcd.tmp").write_bytes(b"foreign")
    (tmp_path / ".long.amr.123456789.tmp").write_bytes(b"foreign")
    (tmp_path / ".uppercase-token.amr.ABCDEFGH.tmp").write_bytes(b"foreign")
    (tmp_path / ".directory.amr.abcdefgh.tmp").mkdir()
    try:
        (tmp_path / ".link.amr.abcdefgh.tmp").symlink_to(older)
    except OSError:
        pass

    found = discover_interrupted_project_saves(tmp_path)

    assert [Path(item.candidate_path).name for item in found] == [
        newer.name,
        older.name,
    ]
    assert Path(found[0].intended_destination) == tmp_path / "새 기록.AMR"
    assert found[0].size_bytes == len(b"newer")


def test_discovery_fails_closed_instead_of_truncating_candidate_overflow(
    tmp_path: Path,
) -> None:
    (tmp_path / ".one.amr.abcdefgh.tmp").write_bytes(b"one")
    (tmp_path / ".two.amr.1234abcd.tmp").write_bytes(b"two")

    with patch("src.core.project_recovery.MAX_INTERRUPTED_SAVE_CANDIDATES", 1):
        with pytest.raises(ProjectRecoveryError) as raised:
            discover_interrupted_project_saves(tmp_path)

    assert raised.value.stage == "discovery"
    assert "too many" in str(raised.value)


def test_discovery_uses_path_stat_instead_of_windows_direntry_identity_cache(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / ".field.amr.abcdefgh.tmp"
    candidate_path.write_bytes(b"candidate")
    entry = Mock()
    entry.name = candidate_path.name
    scan = Mock()
    scan.__enter__ = Mock(return_value=iter((entry,)))
    scan.__exit__ = Mock(return_value=False)

    with patch("src.core.project_recovery.os.scandir", return_value=scan):
        found = discover_interrupted_project_saves(tmp_path)

    assert len(found) == 1
    assert found[0].candidate_path == str(candidate_path)
    assert found[0].inode == candidate_path.stat(follow_symlinks=False).st_ino
    entry.stat.assert_not_called()


@pytest.mark.parametrize(
    ("source", "destination", "expected_source", "expected_destination"),
    (
        (
            r"C:\문화유산\깊은 경로\.복구본.amr.abcdefgh.tmp",
            r"C:\문화유산\깊은 경로\복구본.amr",
            r"\\?\C:\문화유산\깊은 경로\.복구본.amr.abcdefgh.tmp",
            r"\\?\C:\문화유산\깊은 경로\복구본.amr",
        ),
        (
            r"\\조사실-서버\공유 자료\유물\.복구본.amr.abcdefgh.tmp",
            r"\\조사실-서버\공유 자료\유물\복구본.amr",
            r"\\?\UNC\조사실-서버\공유 자료\유물\.복구본.amr.abcdefgh.tmp",
            r"\\?\UNC\조사실-서버\공유 자료\유물\복구본.amr",
        ),
    ),
)
def test_windows_recovery_publish_uses_unicode_write_through_create_new(
    source: str,
    destination: str,
    expected_source: str,
    expected_destination: str,
) -> None:
    calls: list[tuple[str, str, int]] = []

    def move_file_ex(source_path: str, destination_path: str, flags: int) -> int:
        calls.append((source_path, destination_path, flags))
        return 1

    backend = project_recovery._publish_file_noreplace(
        Path(source),
        Path(destination),
        platform_name="nt",
        move_file_ex=move_file_ex,
        get_last_error=lambda: 0,
    )

    assert backend == project_recovery.WINDOWS_RECOVERY_PUBLISH_BACKEND
    assert calls == [
        (
            expected_source,
            expected_destination,
            project_recovery.MOVEFILE_WRITE_THROUGH,
        )
    ]


@pytest.mark.parametrize("winerror", (80, 183))
def test_windows_recovery_publish_maps_destination_race_to_file_exists(
    winerror: int,
) -> None:
    with pytest.raises(FileExistsError) as raised:
        project_recovery._windows_publish_file_noreplace(
            r"C:\유물\.복구.amr.abcdefgh.tmp",
            r"C:\유물\복구.amr",
            move_file_ex=lambda _source, _destination, _flags: 0,
            get_last_error=lambda: winerror,
        )

    assert raised.value.errno == winerror
    assert "already exists" in str(raised.value)


def test_windows_recovery_publish_failure_never_falls_back_to_rename() -> None:
    with patch.object(project_recovery.os, "rename") as fallback:
        with pytest.raises(OSError) as raised:
            project_recovery._publish_file_noreplace(
                Path(r"C:\유물\.복구.amr.abcdefgh.tmp"),
                Path(r"C:\유물\복구.amr"),
                platform_name="nt",
                move_file_ex=lambda _source, _destination, _flags: 0,
                get_last_error=lambda: 5,
            )

    assert raised.value.errno == 5
    assert "write-through create-new" in str(raised.value)
    fallback.assert_not_called()


def test_recovery_materializes_embedded_source_and_never_touches_existing_files(
    tmp_path: Path,
) -> None:
    session, candidate_path, intended = _valid_candidate(tmp_path)
    candidate_bytes = candidate_path.read_bytes()
    intended.write_bytes(b"existing project stays byte-for-byte")
    existing_bytes = intended.read_bytes()
    candidate = discover_interrupted_project_saves(tmp_path)[0]
    recovered = tmp_path / "복구본.amr"

    result = recover_interrupted_project_save(candidate, recovered)

    assert Path(result.destination) == recovered
    assert result.candidate_path == str(candidate_path)
    assert result.document_id == session.document.document_id
    assert result.project_sha256 == hashlib.sha256(candidate_bytes).hexdigest()
    assert result.size_bytes == len(candidate_bytes)
    assert result.durability_warning is None
    assert (
        result.publish_backend
        == project_recovery.project_recovery_publish_backend_identifier()
    )
    assert candidate_path.read_bytes() == candidate_bytes
    assert intended.read_bytes() == existing_bytes
    restored = load_artifact_session_project(recovered)
    assert restored.document.canonical_json_bytes() == session.document.canonical_json_bytes()
    assert list(tmp_path.glob(f".{recovered.name}.*.tmp")) == []


@pytest.mark.parametrize(
    ("payload", "expected_stage"),
    ((b"not an AMR archive", "validation"), (b"", "candidate_identity")),
)
def test_invalid_or_incomplete_candidate_fails_without_publishing(
    tmp_path: Path,
    payload: bytes,
    expected_stage: str,
) -> None:
    candidate_path = tmp_path / ".record.amr.abcdefgh.tmp"
    candidate_path.write_bytes(payload)
    candidate = discover_interrupted_project_saves(tmp_path)[0]
    recovered = tmp_path / "recovered.amr"

    with pytest.raises(ProjectRecoveryError) as raised:
        recover_interrupted_project_save(candidate, recovered)

    assert raised.value.stage == expected_stage
    assert candidate_path.read_bytes() == payload
    assert not recovered.exists()
    assert list(tmp_path.glob(f".{recovered.name}.*.tmp")) == []


def test_manifest_only_artifact_is_not_misrepresented_as_offline_recovery(
    tmp_path: Path,
) -> None:
    session, _source = _session(tmp_path)
    seed = tmp_path / "manifest-only.amr"
    save_artifact_project(seed, session.document)
    candidate_path = tmp_path / ".field.amr.abcdefgh.tmp"
    shutil.copyfile(seed, candidate_path)
    candidate = discover_interrupted_project_saves(tmp_path)[0]
    recovered = tmp_path / "recovered.amr"

    with pytest.raises(ProjectRecoveryError) as raised:
        recover_interrupted_project_save(candidate, recovered)

    assert raised.value.stage == "validation"
    assert candidate_path.exists()
    assert not recovered.exists()


def test_candidate_identity_is_pinned_between_discovery_and_copy(tmp_path: Path) -> None:
    _session_value, candidate_path, _intended = _valid_candidate(tmp_path)
    candidate_bytes = candidate_path.read_bytes()
    candidate = discover_interrupted_project_saves(tmp_path)[0]
    replacement = tmp_path / "replacement.tmp"
    replacement.write_bytes(candidate_bytes)
    os.replace(replacement, candidate_path)
    recovered = tmp_path / "recovered.amr"

    with pytest.raises(ProjectRecoveryError) as raised:
        recover_interrupted_project_save(candidate, recovered)

    assert raised.value.stage == "candidate_identity"
    assert candidate_path.read_bytes() == candidate_bytes
    assert not recovered.exists()


def test_validated_staging_path_must_keep_the_copied_inode(tmp_path: Path) -> None:
    session, candidate_path, _intended = _valid_candidate(tmp_path)
    candidate = discover_interrupted_project_saves(tmp_path)[0]
    replacement = tmp_path / "foreign-valid.amr"
    replacement.write_bytes(candidate_path.read_bytes())
    recovered = tmp_path / "recovered.amr"

    def replace_staging(path: Path) -> ArtifactSession:
        os.replace(replacement, Path(path))
        return session

    with patch(
        "src.core.project_recovery.load_artifact_session_project",
        side_effect=replace_staging,
    ):
        with pytest.raises(ProjectRecoveryError) as raised:
            recover_interrupted_project_save(candidate, recovered)

    assert raised.value.stage == "validation"
    assert candidate_path.exists()
    assert not recovered.exists()
    assert list(tmp_path.glob(f".{recovered.name}.*.tmp")) == []


def test_existing_or_racing_destination_is_never_overwritten(tmp_path: Path) -> None:
    _session_value, candidate_path, _intended = _valid_candidate(tmp_path)
    candidate = discover_interrupted_project_saves(tmp_path)[0]
    recovered = tmp_path / "recovered.amr"
    recovered.write_bytes(b"existing winner")

    with pytest.raises(ProjectRecoveryError) as raised:
        recover_interrupted_project_save(candidate, recovered)

    assert raised.value.stage == "prepare"
    assert recovered.read_bytes() == b"existing winner"
    assert candidate_path.exists()

    recovered.unlink()

    def racing_publish(_stage: Path, destination: Path) -> None:
        Path(destination).write_bytes(b"concurrent winner")
        raise FileExistsError(str(destination))

    with patch(
        "src.core.project_recovery._publish_file_noreplace",
        side_effect=racing_publish,
    ):
        with pytest.raises(ProjectRecoveryError) as raced:
            recover_interrupted_project_save(candidate, recovered)

    assert raced.value.stage == "publish"
    assert recovered.read_bytes() == b"concurrent winner"
    assert candidate_path.exists()
    assert list(tmp_path.glob(f".{recovered.name}.*.tmp")) == []


def test_windows_publish_failure_preserves_candidate_and_cleans_owned_staging(
    tmp_path: Path,
) -> None:
    _session_value, candidate_path, _intended = _valid_candidate(tmp_path)
    candidate_bytes = candidate_path.read_bytes()
    candidate = discover_interrupted_project_saves(tmp_path)[0]
    recovered = tmp_path / "recovered.amr"

    with patch(
        "src.core.project_recovery._publish_file_noreplace",
        side_effect=OSError(5, "injected Win32 publication failure"),
    ):
        with pytest.raises(ProjectRecoveryError) as raised:
            recover_interrupted_project_save(candidate, recovered)

    assert raised.value.stage == "publish"
    assert not raised.value.committed
    assert candidate_path.read_bytes() == candidate_bytes
    assert not recovered.exists()
    assert list(tmp_path.glob(f".{recovered.name}.*.tmp")) == []


def test_directory_fsync_failure_returns_committed_durability_warning(
    tmp_path: Path,
) -> None:
    session, candidate_path, _intended = _valid_candidate(tmp_path)
    candidate = discover_interrupted_project_saves(tmp_path)[0]
    recovered = tmp_path / "recovered.amr"

    def publish_as_posix(stage: Path, destination: Path) -> str:
        os.rename(stage, destination)
        return project_recovery.POSIX_RECOVERY_PUBLISH_BACKEND

    with (
        patch(
            "src.core.project_recovery._publish_file_noreplace",
            side_effect=publish_as_posix,
        ),
        patch(
            "src.core.project_recovery._best_effort_fsync_directory",
            side_effect=OSError("injected directory sync failure"),
        ),
    ):
        result = recover_interrupted_project_save(candidate, recovered)

    assert result.durability_warning is not None
    assert "durability is uncertain" in result.durability_warning
    assert result.publish_backend == project_recovery.POSIX_RECOVERY_PUBLISH_BACKEND
    assert candidate_path.exists()
    assert load_artifact_session_project(recovered).document == session.document


def test_windows_write_through_recovery_skips_posix_directory_fsync(
    tmp_path: Path,
) -> None:
    _session_value, candidate_path, _intended = _valid_candidate(tmp_path)
    candidate = discover_interrupted_project_saves(tmp_path)[0]
    recovered = tmp_path / "윈도우-복구본.amr"

    def publish_as_windows(stage: Path, destination: Path) -> str:
        os.rename(stage, destination)
        return project_recovery.WINDOWS_RECOVERY_PUBLISH_BACKEND

    with (
        patch(
            "src.core.project_recovery._publish_file_noreplace",
            side_effect=publish_as_windows,
        ),
        patch(
            "src.core.project_recovery._best_effort_fsync_directory"
        ) as directory_fsync,
    ):
        result = recover_interrupted_project_save(candidate, recovered)

    assert result.durability_warning is None
    assert result.publish_backend == project_recovery.WINDOWS_RECOVERY_PUBLISH_BACKEND
    directory_fsync.assert_not_called()
    assert candidate_path.exists()
    assert load_artifact_session_project(recovered)
