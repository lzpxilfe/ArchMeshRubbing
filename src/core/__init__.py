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
    "FlattenResultMeta": ".flattener",
    "FlattenedMesh": ".flattener",
    "flatten_with_method": ".flattener",
    "flatten_with_recommendation": ".flattener",
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
    "CircleFit2DResult": ".tile_profile_fitting",
    "fit_circle_2d": ".tile_profile_fitting",
    "FlattenAlternative": ".flatten_policy",
    "FlattenRecommendation": ".flatten_policy",
    "build_alternatives": ".flatten_policy",
    "explain_recommendation": ".flatten_policy",
    "fallback_chain_for_context": ".flatten_policy",
    "get_recommended_method": ".flatten_policy",
    "recommend_flatten_mode": ".flatten_policy",
    "SyntheticTileSpec": ".tile_synthetic",
    "SyntheticTileGroundTruth": ".tile_synthetic",
    "SyntheticTileArtifact": ".tile_synthetic",
    "TileEvaluationReport": ".tile_synthetic",
    "SyntheticBenchmarkCaseResult": ".tile_synthetic",
    "SyntheticBenchmarkSuiteReport": ".tile_synthetic",
    "default_synthetic_tile_spec": ".tile_synthetic",
    "synthetic_tile_spec_from_preset": ".tile_synthetic",
    "generate_synthetic_tile": ".tile_synthetic",
    "evaluate_tile_interpretation": ".tile_synthetic",
    "render_synthetic_tile_review_sheet": ".tile_synthetic",
    "save_synthetic_tile_bundle": ".tile_synthetic",
    "save_synthetic_benchmark_suite": ".tile_synthetic",
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
