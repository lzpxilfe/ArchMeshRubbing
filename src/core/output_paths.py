"""
Output path helpers for common exports.

Centralizes naming conventions so CLI/GUI stay in sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

RUBBING_SUFFIX = ".rubbing.png"
PROJECTION_SUFFIX = ".projection.png"
INNER_SUFFIX = ".inner.ply"
OUTER_SUFFIX = ".outer.ply"


def _as_path(value: PathLike) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _resolve_output_path(input_path: PathLike, output_path: Optional[PathLike], suffix: str) -> Path:
    if output_path:
        return _as_path(output_path)
    return _as_path(input_path).with_suffix(suffix)


def rubbing_output_path(input_path: PathLike, output_path: Optional[PathLike] = None) -> Path:
    return _resolve_output_path(input_path, output_path, RUBBING_SUFFIX)


def projection_output_path(input_path: PathLike, output_path: Optional[PathLike] = None) -> Path:
    return _resolve_output_path(input_path, output_path, PROJECTION_SUFFIX)


def inner_surface_path(input_path: PathLike, output_path: Optional[PathLike] = None) -> Path:
    return _resolve_output_path(input_path, output_path, INNER_SUFFIX)


def outer_surface_path(input_path: PathLike, output_path: Optional[PathLike] = None) -> Path:
    return _resolve_output_path(input_path, output_path, OUTER_SUFFIX)
