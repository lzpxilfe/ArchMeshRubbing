"""
Core processing modules for ArchMeshRubbing
"""

from .mesh_loader import MeshLoader, MeshData
from .flattener import ARAPFlattener, FlattenedMesh
from .orthographic_projector import OrthographicProjector, ProjectionResult
from .surface_visualizer import SurfaceVisualizer, RubbingImage
from .surface_separator import SurfaceSeparator, SeparatedSurfaces
from .region_selector import RegionSelector, SelectionResult
from .flattened_svg_exporter import FlattenedSVGExporter, SVGExportOptions
from .rubbing_sheet_exporter import RubbingSheetExporter, SheetExportOptions

__all__ = [
    # Mesh loading
    'MeshLoader',
    'MeshData',
    # Flattening
    'ARAPFlattener',
    'FlattenedMesh',
    # Orthographic projection
    'OrthographicProjector',
    'ProjectionResult',
    # Surface visualization
    'SurfaceVisualizer',
    'RubbingImage',
    # Surface separation
    'SurfaceSeparator',
    'SeparatedSurfaces',
    # Region selection
    'RegionSelector',
    'SelectionResult',
    # Flattened SVG export
    'FlattenedSVGExporter',
    'SVGExportOptions',
    # Composite sheet SVG export
    'RubbingSheetExporter',
    'SheetExportOptions',
]
