"""Generate immutable metadata embedded in a native ArchMeshRubbing build."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import __version__  # noqa: E402
from src.build_info import BUILD_INFO_SCHEMA_VERSION  # noqa: E402


_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_CHANNEL_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,31}$")


def detect_source_tree(root: Path = ROOT) -> str:
    """Return an honest Git worktree state without making it a build blocker."""

    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return "dirty" if completed.stdout.strip() else "clean"


def build_manifest(
    *,
    channel: str,
    commit: str,
    lock_path: Path,
    wheel_lock_path: Path,
    source_tree: str = "unknown",
) -> dict[str, str]:
    if _CHANNEL_RE.fullmatch(channel) is None:
        raise ValueError("channel must be a stable lowercase identifier")
    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("commit must be a lowercase 40- or 64-character Git hash")
    if source_tree not in {"clean", "dirty", "unknown"}:
        raise ValueError("source_tree must be clean, dirty, or unknown")
    lock_bytes = lock_path.read_bytes()
    if not lock_bytes:
        raise ValueError("runtime lock must not be empty")
    wheel_lock_bytes = wheel_lock_path.read_bytes()
    if not wheel_lock_bytes:
        raise ValueError("Windows wheel lock must not be empty")
    return {
        "channel": channel,
        "commit": commit,
        "dependency_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "schema_version": BUILD_INFO_SCHEMA_VERSION,
        "source_tree": source_tree,
        "version": __version__,
        "windows_wheel_lock_sha256": hashlib.sha256(wheel_lock_bytes).hexdigest(),
    }


def write_manifest(path: Path, manifest: dict[str, str]) -> None:
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "requirements" / "runtime-py312.lock",
    )
    parser.add_argument(
        "--wheel-lock",
        type=Path,
        default=ROOT / "requirements" / "windows-py312-x64-hashed.lock",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-tree",
        choices=("auto", "clean", "dirty", "unknown"),
        default="auto",
        help="source worktree state; auto inspects the repository",
    )
    args = parser.parse_args()
    source_tree = (
        detect_source_tree(ROOT)
        if args.source_tree == "auto"
        else str(args.source_tree)
    )
    manifest = build_manifest(
        channel=args.channel,
        commit=args.commit,
        lock_path=args.lock,
        wheel_lock_path=args.wheel_lock,
        source_tree=source_tree,
    )
    write_manifest(args.output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
