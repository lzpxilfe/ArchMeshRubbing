"""
ArchMeshRubbing project file I/O (.amr)

The project format is a zip container with a JSON manifest.
This keeps the file compact (compresses large index lists) and allows
future extension (e.g., embedding meshes) without breaking compatibility.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import zipfile


PROJECT_FORMAT = "archmeshrubbing_project"
PROJECT_VERSION = 1
MANIFEST_NAME = "project.json"


class ProjectFormatError(RuntimeError):
    pass


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def save_project(path: str | Path, state: dict[str, Any], *, meta: dict[str, Any] | None = None) -> str:
    """
    Save a project file.

    Args:
        path: destination path (usually ends with .amr)
        state: app/scene state (JSON-serializable)
        meta: optional metadata (e.g., app version)
    """
    out_path = Path(path)
    doc: dict[str, Any] = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "saved_at": _utc_now_iso(),
        "meta": dict(meta or {}),
        "state": state,
    }

    data = json.dumps(doc, ensure_ascii=False, indent=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, data.encode("utf-8"))
    return str(out_path)


def load_project(path: str | Path) -> dict[str, Any]:
    """
    Load a project file.

    Returns:
        Parsed project document (including keys: format/version/meta/state)
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    raw: str
    if zipfile.is_zipfile(in_path):
        with zipfile.ZipFile(in_path, "r") as zf:
            try:
                raw_bytes = zf.read(MANIFEST_NAME)
            except KeyError as e:
                raise ProjectFormatError(f"Missing {MANIFEST_NAME} in project file") from e
        raw = raw_bytes.decode("utf-8", errors="replace")
    else:
        # Developer-friendly fallback: allow plain JSON for debugging.
        raw = in_path.read_text(encoding="utf-8", errors="replace")

    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ProjectFormatError(f"Invalid JSON: {e}") from e

    if not isinstance(doc, dict):
        raise ProjectFormatError("Invalid project document (expected JSON object)")

    fmt = str(doc.get("format", "")).strip()
    ver = doc.get("version", None)
    if fmt != PROJECT_FORMAT:
        raise ProjectFormatError(f"Unsupported project format: {fmt!r}")
    if ver != PROJECT_VERSION:
        raise ProjectFormatError(f"Unsupported project version: {ver!r}")

    state = doc.get("state", None)
    if not isinstance(state, dict):
        raise ProjectFormatError("Invalid project document: missing 'state' object")

    meta = doc.get("meta", {})
    if meta is None:
        doc["meta"] = {}
    elif not isinstance(meta, dict):
        doc["meta"] = {"_raw": meta}

    return doc
