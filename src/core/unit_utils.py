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
    SVG exports in this project use `mm` or `cm` (meters are exported as `cm`).
    """
    mesh_u = normalize_unit(mesh_unit)
    req_u = normalize_unit(requested) if requested is not None else None

    # Default: follow mesh units when possible (meters become centimeters).
    if req_u is None:
        if mesh_u == "m":
            return "cm", 100.0
        return mesh_u, 1.0

    # Requested `mm`
    if req_u == "mm":
        if mesh_u == "mm":
            return "mm", 1.0
        if mesh_u == "cm":
            return "mm", 10.0
        if mesh_u == "m":
            return "mm", 1000.0
        return "mm", 1.0

    # Requested `cm` (or `m`, which is exported as `cm`)
    if mesh_u == "mm":
        return "cm", 0.1
    if mesh_u == "cm":
        return "cm", 1.0
    if mesh_u == "m":
        return "cm", 100.0
    return "cm", 0.1


def mm_to_svg_units(mm: float, svg_unit: str) -> float:
    u = normalize_unit(svg_unit)
    if u == "mm":
        return float(mm)
    if u == "cm":
        return float(mm) / 10.0
    return float(mm)
