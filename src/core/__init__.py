"""
Core processing modules for ArchMeshRubbing.

This package uses lazy attribute loading so lightweight entry points such as
CLI help text can import `src.core.*` modules without immediately requiring
the full numeric/GUI stack.
"""

from __future__ import annotations

from importlib import import_module

_LAZY_IMPORTS = {
    "MeshLoader": ".mesh_loader",
    "MeshData": ".mesh_loader",
    "ARAPFlattener": ".flattener",
    "FlattenedMesh": ".flattener",
    "OrthographicProjector": ".orthographic_projector",
    "ProjectionResult": ".orthographic_projector",
    "SurfaceVisualizer": ".surface_visualizer",
    "RubbingImage": ".surface_visualizer",
    "SurfaceSeparator": ".surface_separator",
    "SeparatedSurfaces": ".surface_separator",
    "RegionSelector": ".region_selector",
    "SelectionResult": ".region_selector",
    "FlattenedSVGExporter": ".flattened_svg_exporter",
    "SVGExportOptions": ".flattened_svg_exporter",
    "RubbingSheetExporter": ".rubbing_sheet_exporter",
    "SheetExportOptions": ".rubbing_sheet_exporter",
    "RecordingSurfaceReview": ".recording_surface_review",
    "RecordingSurfaceReviewOptions": ".recording_surface_review",
    "build_recording_surface_summary_lines": ".recording_surface_review",
    "render_recording_surface_review": ".recording_surface_review",
    "TileClass": ".tile_form_model",
    "SplitScheme": ".tile_form_model",
    "AxisSource": ".tile_form_model",
    "AxisHint": ".tile_form_model",
    "SectionObservation": ".tile_form_model",
    "MandrelFitResult": ".tile_form_model",
    "TileInterpretationSlot": ".tile_form_model",
    "TileInterpretationState": ".tile_form_model",
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
