"""Pure, unit-aware formatting for the mesh information panels."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Literal


SurfaceAreaDisplayState = Literal["exact", "estimate", "unavailable"]
_SUPPORTED_MESH_UNITS = frozenset({"mm", "cm", "m"})


@dataclass(frozen=True, slots=True)
class MeshDisplayText:
    """Ready-to-display values with explicit unit and area confidence."""

    size_text: str
    area_text: str
    unit: str | None
    area_state: SurfaceAreaDisplayState


def format_mesh_display_values(
    *,
    extents: Iterable[Any],
    surface_area: Any,
    unit: object,
    surface_area_state: SurfaceAreaDisplayState,
    decimal_places: int = 1,
) -> MeshDisplayText:
    """Format mesh dimensions and area without guessing a physical unit.

    The numeric coordinates are already expressed in ``MeshData.unit``.  New
    authoritative work uses millimetres as its canonical unit, but this
    formatter deliberately respects an explicitly confirmed ``cm`` or ``m``
    source instead of hard-coding ``cm``.  Missing/unknown units and invalid
    values fail closed so a raw coordinate cannot be presented as a physical
    measurement.
    """

    if (
        isinstance(decimal_places, bool)
        or not isinstance(decimal_places, int)
        or not 0 <= decimal_places <= 6
    ):
        raise ValueError("decimal_places must be an integer from 0 through 6")
    if surface_area_state not in {"exact", "estimate", "unavailable"}:
        raise ValueError(f"unsupported surface_area_state: {surface_area_state!r}")

    normalized_unit = _normalized_unit(unit)
    if normalized_unit is None:
        return MeshDisplayText(
            size_text="사용 불가 (단위 미확인)",
            area_text="계산 불가 (단위 미확인)",
            unit=None,
            area_state="unavailable",
        )

    dimension_values = _finite_dimensions(extents)
    if dimension_values is None:
        size_text = "사용 불가"
    else:
        size_text = (
            " × ".join(
                _format_number(value, decimal_places) for value in dimension_values
            )
            + f" {normalized_unit}"
        )

    area_number = _finite_nonnegative(surface_area)
    if surface_area_state == "unavailable" or area_number is None:
        area_text = "계산 불가"
        output_state: SurfaceAreaDisplayState = "unavailable"
    else:
        value_text = _format_number(area_number, decimal_places)
        if surface_area_state == "estimate":
            area_text = f"약 {value_text} {normalized_unit}² (추정)"
            output_state = "estimate"
        else:
            area_text = f"{value_text} {normalized_unit}²"
            output_state = "exact"

    return MeshDisplayText(
        size_text=size_text,
        area_text=area_text,
        unit=normalized_unit,
        area_state=output_state,
    )


def _normalized_unit(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in _SUPPORTED_MESH_UNITS else None


def _finite_dimensions(values: Iterable[Any]) -> tuple[float, float, float] | None:
    try:
        dimensions = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(dimensions) != 3:
        return None
    if any(not math.isfinite(value) or value < 0.0 for value in dimensions):
        return None
    return dimensions


def _finite_nonnegative(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def _format_number(value: float, decimal_places: int) -> str:
    # Avoid displaying a negative zero introduced by upstream floating-point
    # transforms while retaining normal rounding for measured coordinates.
    if value == 0.0:
        value = 0.0
    return f"{value:,.{decimal_places}f}"
