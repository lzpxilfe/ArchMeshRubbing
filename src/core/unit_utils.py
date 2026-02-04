"""
Unit helpers (MeshData units ↔ SVG units)

This project typically treats mesh coordinates as being in `mesh.unit` (mm/cm/m).
SVG exports commonly use `mm` or `cm` units. These helpers centralize the
conversion rules so different exporters stay consistent.
"""

from __future__ import annotations

from typing import Optional


def normalize_unit(unit: Optional[str]) -> str:
    u = str(unit or "").strip().lower()
    if u in {"mm", "millimeter", "millimeters"}:
        return "mm"
    if u in {"cm", "centimeter", "centimeters"}:
        return "cm"
    if u in {"m", "meter", "meters"}:
        return "m"
    return "mm"


def resolve_svg_unit(mesh_unit: Optional[str], requested: Optional[str]) -> tuple[str, float]:
    """
    Returns:
        (svg_unit, unit_scale)

    `unit_scale` is a multiplier applied to values expressed in mesh units to get SVG units.
    """
    unit = normalize_unit(requested or mesh_unit or "mm")

    if unit == "mm":
        return "mm", 1.0
    if unit == "cm":
        return "cm", 1.0
    if unit == "m":
        # SVG에서 m는 불편하므로 cm로 변환 (1m = 100cm)
        return "cm", 100.0
    return "mm", 1.0


def mm_to_svg_units(mm: float, svg_unit: str) -> float:
    u = normalize_unit(svg_unit)
    if u == "mm":
        return float(mm)
    if u == "cm":
        return float(mm) / 10.0
    return float(mm)

