"""Fail-closed recovery for interrupted native project saves.

Transactional project saves use a same-directory temporary file named
``.<destination>.XXXXXXXX.tmp``.  A normal failure removes that file, but an
abrupt process or machine stop can leave it behind.  This module provides an
explicit, folder-scoped recovery path without startup scanning, automatic
deletion, or destination replacement.

Discovery trusts only the writer's exact basename shape.  Recovery then pins
the candidate's filesystem identity, copies it through an already-open regular
file descriptor, fsyncs a new staging file, and reopens that copy through the
production embedded-session loader.  Only a fully self-contained native AMR
is published with the platform's same-filesystem atomic no-replace rename.
The interrupted candidate is never modified or removed.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

from .project_file import (
    MAX_PROJECT_FILE_BYTES,
    ProjectFormatError,
    _best_effort_fsync_directory,
    load_artifact_session_project,
)


MAX_INTERRUPTED_SAVE_CANDIDATES = 64
_TEMP_NAME_RE = re.compile(
    r"^\.(?P<destination>.+\.(?i:amr))\.(?P<token>[a-z0-9_]{8})\.tmp$",
)
_COPY_CHUNK_BYTES = 1024 * 1024


class ProjectRecoveryError(RuntimeError):
    """An interrupted-save candidate could not be safely recovered."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        committed: bool = False,
    ) -> None:
        super().__init__(message)
        self.stage = str(stage)
        self.committed = bool(committed)


@dataclass(frozen=True, slots=True)
class InterruptedProjectSave:
    """One regular file that matches the native writer's temporary name."""

    candidate_path: str
    intended_destination: str
    size_bytes: int
    modified_time_ns: int
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class ProjectRecoveryResult:
    """Receipt for one verified, create-new recovered project."""

    destination: str
    candidate_path: str
    document_id: str
    project_sha256: str
    size_bytes: int
    durability_warning: str | None = None


def _absolute_path(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _candidate_destination(directory: Path, name: str) -> Path | None:
    match = _TEMP_NAME_RE.fullmatch(name)
    if match is None:
        return None
    destination_name = match.group("destination")
    if Path(destination_name).name != destination_name:
        return None
    return directory / destination_name


def discover_interrupted_project_saves(
    directory: str | os.PathLike[str],
) -> tuple[InterruptedProjectSave, ...]:
    """List bounded, regular interrupted-save candidates in one chosen folder.

    This is deliberately a cheap structural discovery.  A listed candidate is
    not declared valid until :func:`recover_interrupted_project_save` copies and
    fully materializes it through the production native project loader.
    """

    folder = _absolute_path(directory)
    try:
        folder_identity = folder.stat(follow_symlinks=False)
    except OSError as exc:
        raise ProjectRecoveryError(
            "discovery",
            f"Recovery folder cannot be inspected: {exc}",
        ) from exc
    if not stat.S_ISDIR(folder_identity.st_mode):
        raise ProjectRecoveryError(
            "discovery",
            "Recovery location must be a directory",
        )

    candidates: list[InterruptedProjectSave] = []
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                intended = _candidate_destination(folder, entry.name)
                if intended is None:
                    continue
                candidate_path = folder / entry.name
                try:
                    # Windows DirEntry.stat() may expose zero placeholder
                    # st_dev/st_ino values from directory-enumeration metadata,
                    # while a later os.stat() returns the real file identity.
                    # Use the same path-based syscall as every recovery fence.
                    identity = candidate_path.stat(follow_symlinks=False)
                except OSError:
                    # A racing or unreadable entry is not safe to offer.
                    continue
                if not stat.S_ISREG(identity.st_mode):
                    continue
                candidates.append(
                    InterruptedProjectSave(
                        candidate_path=str(candidate_path),
                        intended_destination=str(intended),
                        size_bytes=int(identity.st_size),
                        modified_time_ns=int(identity.st_mtime_ns),
                        device=int(identity.st_dev),
                        inode=int(identity.st_ino),
                    )
                )
                if len(candidates) > MAX_INTERRUPTED_SAVE_CANDIDATES:
                    raise ProjectRecoveryError(
                        "discovery",
                        "Recovery folder contains too many interrupted-save candidates",
                    )
    except ProjectRecoveryError:
        raise
    except OSError as exc:
        raise ProjectRecoveryError(
            "discovery",
            f"Recovery folder cannot be enumerated: {exc}",
        ) from exc

    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                -candidate.modified_time_ns,
                candidate.candidate_path,
            ),
        )
    )


def _identity_tuple(identity: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(identity.st_dev),
        int(identity.st_ino),
        int(identity.st_size),
        int(identity.st_mtime_ns),
    )


def _expected_identity(candidate: InterruptedProjectSave) -> tuple[int, int, int, int]:
    return (
        int(candidate.device),
        int(candidate.inode),
        int(candidate.size_bytes),
        int(candidate.modified_time_ns),
    )


def _validated_candidate_paths(
    candidate: InterruptedProjectSave,
) -> tuple[Path, Path, os.stat_result]:
    if not isinstance(candidate, InterruptedProjectSave):
        raise TypeError("candidate must be an InterruptedProjectSave")
    candidate_path = _absolute_path(candidate.candidate_path)
    intended = _absolute_path(candidate.intended_destination)
    derived = _candidate_destination(candidate_path.parent, candidate_path.name)
    if derived is None or _absolute_path(derived) != intended:
        raise ProjectRecoveryError(
            "candidate_identity",
            "Interrupted-save filename no longer matches its intended .amr destination",
        )
    try:
        identity = candidate_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ProjectRecoveryError(
            "candidate_identity",
            f"Interrupted-save candidate cannot be inspected: {exc}",
        ) from exc
    if not stat.S_ISREG(identity.st_mode):
        raise ProjectRecoveryError(
            "candidate_identity",
            "Interrupted-save candidate must remain a regular, non-symlink file",
        )
    if _identity_tuple(identity) != _expected_identity(candidate):
        raise ProjectRecoveryError(
            "candidate_identity",
            "Interrupted-save candidate changed after discovery",
        )
    if identity.st_size <= 0 or identity.st_size > MAX_PROJECT_FILE_BYTES:
        raise ProjectRecoveryError(
            "candidate_identity",
            "Interrupted-save candidate is empty or exceeds the project size limit",
        )
    return candidate_path, intended, identity


def _unlink_if_owned(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        current = path.stat(follow_symlinks=False)
        if (int(current.st_dev), int(current.st_ino)) == identity:
            path.unlink()
    except (FileNotFoundError, OSError):
        return


def _publish_file_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename one same-filesystem file without replacement."""

    if os.name == "nt":
        # On Windows Python's rename fails when the destination exists.
        os.rename(source, destination)
        return

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    result: int
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(
                getattr(errno, "ENOTSUP", errno.EINVAL),
                "atomic no-replace rename is unavailable",
                str(destination),
            )
        # AT_FDCWD=-100, RENAME_NOREPLACE=1.
        result = int(renameat2(-100, source_bytes, -100, destination_bytes, 1))
    elif sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise OSError(
                getattr(errno, "ENOTSUP", errno.EINVAL),
                "atomic no-replace rename is unavailable",
                str(destination),
            )
        # Darwin RENAME_EXCL=0x00000004.
        result = int(renamex_np(source_bytes, destination_bytes, 0x00000004))
    else:
        raise OSError(
            getattr(errno, "ENOTSUP", errno.EINVAL),
            "atomic no-replace rename is unsupported on this platform",
            str(destination),
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def recover_interrupted_project_save(
    candidate: InterruptedProjectSave,
    destination: str | os.PathLike[str],
) -> ProjectRecoveryResult:
    """Copy, fully validate, and create a recovered native ``.amr``.

    ``destination`` is create-new: an existing path is never replaced.  The
    candidate remains untouched on both success and failure.
    """

    candidate_path, _intended, discovered_identity = _validated_candidate_paths(
        candidate
    )
    output = _absolute_path(destination)
    if output.suffix.lower() != ".amr":
        raise ProjectRecoveryError(
            "prepare",
            "Recovered project destination must use the .amr extension",
        )
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProjectRecoveryError(
            "prepare",
            f"Recovered project folder cannot be created: {exc}",
        ) from exc
    if output.exists() or output.is_symlink():
        raise ProjectRecoveryError(
            "prepare",
            "Recovered project destination already exists; overwrite is not allowed",
        )

    source_fd: int | None = None
    stage_fd: int | None = None
    stage_path: Path | None = None
    published_identity: tuple[int, int] | None = None
    published = False
    stage = "candidate_open"
    digest = hashlib.sha256()
    copied = 0
    copied_stage_identity: tuple[int, int, int, int] | None = None
    try:
        open_flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
        open_flags |= int(getattr(os, "O_NOFOLLOW", 0))
        source_fd = os.open(candidate_path, open_flags)
        opened_identity = os.fstat(source_fd)
        if (
            not stat.S_ISREG(opened_identity.st_mode)
            or _identity_tuple(opened_identity) != _identity_tuple(discovered_identity)
        ):
            raise ProjectRecoveryError(
                "candidate_identity",
                "Interrupted-save candidate changed while it was opened",
            )

        stage = "stage_create"
        stage_fd, raw_stage_path = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=str(output.parent),
        )
        stage_path = Path(raw_stage_path)

        stage = "copy"
        source = os.fdopen(source_fd, "rb")
        source_fd = None
        try:
            target = os.fdopen(stage_fd, "wb")
            stage_fd = None
        except Exception:
            source.close()
            raise
        with source, target:
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > MAX_PROJECT_FILE_BYTES:
                    raise ProjectRecoveryError(
                        "copy",
                        "Interrupted-save candidate exceeds the project size limit",
                    )
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
            final_open_identity = os.fstat(source.fileno())
            staged_identity = os.fstat(target.fileno())
            if not stat.S_ISREG(staged_identity.st_mode):
                raise ProjectRecoveryError(
                    "copy",
                    "Recovery staging descriptor is not a regular file",
                )
            copied_stage_identity = _identity_tuple(staged_identity)

        if copied != candidate.size_bytes:
            raise ProjectRecoveryError(
                "candidate_identity",
                "Interrupted-save candidate size changed during recovery copy",
            )
        if _identity_tuple(final_open_identity) != _expected_identity(candidate):
            raise ProjectRecoveryError(
                "candidate_identity",
                "Interrupted-save candidate changed during recovery copy",
            )
        try:
            final_path_identity = candidate_path.stat(follow_symlinks=False)
        except OSError as exc:
            raise ProjectRecoveryError(
                "candidate_identity",
                f"Interrupted-save candidate disappeared during recovery: {exc}",
            ) from exc
        if _identity_tuple(final_path_identity) != _expected_identity(candidate):
            raise ProjectRecoveryError(
                "candidate_identity",
                "Interrupted-save candidate path changed during recovery copy",
            )

        stage = "validation"
        session = load_artifact_session_project(stage_path)

        stage_identity = stage_path.stat(follow_symlinks=False)
        if (
            copied_stage_identity is None
            or not stat.S_ISREG(stage_identity.st_mode)
            or _identity_tuple(stage_identity) != copied_stage_identity
            or stage_identity.st_size != copied
        ):
            raise ProjectRecoveryError(
                "validation",
                "Validated recovery staging file changed before publication",
            )
        published_identity = (int(stage_identity.st_dev), int(stage_identity.st_ino))

        stage = "publish"
        try:
            _publish_file_noreplace(stage_path, output)
        except FileExistsError as exc:
            raise ProjectRecoveryError(
                "publish",
                "Recovered project destination appeared concurrently; overwrite refused",
            ) from exc
        published = True
        output_identity = output.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(output_identity.st_mode)
            or (int(output_identity.st_dev), int(output_identity.st_ino))
            != published_identity
        ):
            raise ProjectRecoveryError(
                "publish",
                "Published recovery path does not reference the validated staging file",
            )

        durability_warning: str | None = None
        stage = "directory_fsync"
        try:
            _best_effort_fsync_directory(output.parent)
        except OSError as exc:
            durability_warning = (
                "Recovered project was published, but directory fsync failed; "
                f"crash durability is uncertain: {exc}"
            )
        return ProjectRecoveryResult(
            destination=str(output),
            candidate_path=str(candidate_path),
            document_id=session.document.document_id,
            project_sha256=digest.hexdigest(),
            size_bytes=copied,
            durability_warning=durability_warning,
        )
    except ProjectRecoveryError:
        if published and stage != "directory_fsync":
            _unlink_if_owned(output, published_identity)
        raise
    except (ProjectFormatError, OSError, RuntimeError, ValueError) as exc:
        if published:
            _unlink_if_owned(output, published_identity)
        raise ProjectRecoveryError(
            stage,
            f"Interrupted project save recovery failed during {stage}: {exc}",
            committed=False,
        ) from exc
    finally:
        if source_fd is not None:
            try:
                os.close(source_fd)
            except OSError:
                pass
        if stage_fd is not None:
            try:
                os.close(stage_fd)
            except OSError:
                pass
        if stage_path is not None:
            try:
                stage_path.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = [
    "InterruptedProjectSave",
    "MAX_INTERRUPTED_SAVE_CANDIDATES",
    "ProjectRecoveryError",
    "ProjectRecoveryResult",
    "discover_interrupted_project_saves",
    "recover_interrupted_project_save",
]
