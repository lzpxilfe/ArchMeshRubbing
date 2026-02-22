"""
Logging helpers.

This project is primarily a GUI app, so logs should go to a stable file location
by default. The goal is to avoid "silent failures" while keeping the UI
responsive and uncluttered.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

_LOG_ONCE_KEYS: set[str] = set()
_LOG_ONCE_LOCK = threading.Lock()


def default_log_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
        return Path(base) / "ArchMeshRubbing" / "logs"

    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home) / "archmeshrubbing" / "logs"

    return Path.home() / ".local" / "state" / "archmeshrubbing" / "logs"


def _parse_log_level(level: str | int) -> int:
    if isinstance(level, int):
        return int(level)
    value = str(level).strip().upper()
    if not value:
        return logging.INFO
    return int(getattr(logging, value, logging.INFO))


def setup_logging(
    *,
    log_level: str | int = "INFO",
    log_dir: Optional[str | Path] = None,
    filename: str = "archmeshrubbing.log",
) -> Optional[Path]:
    """
    Configure root logging to a UTF-8 file.

    This is idempotent: if a FileHandler is already attached, it won't add
    another one.
    """
    root = logging.getLogger()

    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                return Path(handler.baseFilename)
            except Exception:
                return None

    level = _parse_log_level(os.environ.get("ARCHMESHRUBBING_LOG_LEVEL") or log_level)
    root.setLevel(level)

    resolved_dir = Path(log_dir) if log_dir is not None else default_log_dir()
    try:
        resolved_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    log_path = resolved_dir / filename

    fmt = "%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    try:
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(file_handler)
    except Exception:
        return None

    logging.captureWarnings(True)
    root.info("Logging initialized: %s (level=%s)", log_path, logging.getLevelName(level))
    return log_path


def format_exception_message(prefix: str, message: str, *, log_path: Optional[Path]) -> str:
    if log_path is None:
        return f"{prefix}\n\n{message}"
    return f"{prefix}\n\n{message}\n\n(로그 파일: {log_path})"


def log_once(
    logger: logging.Logger,
    key: str,
    level: int,
    msg: str,
    *args,
    exc_info: bool | BaseException | None = None,
) -> bool:
    """
    Logs at most once per process for the given key.

    Useful for suppressing repeated warnings inside render loops while still
    capturing at least one traceback.
    """
    k = str(key)
    with _LOG_ONCE_LOCK:
        if k in _LOG_ONCE_KEYS:
            return False
        _LOG_ONCE_KEYS.add(k)

    logger.log(level, msg, *args, exc_info=exc_info)
    return True
