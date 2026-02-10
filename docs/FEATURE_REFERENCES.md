# Feature Reference Map

This file answers: "Which public references influenced each feature?"

Reference IDs are defined in:
- `docs/REFERENCES.md`

## Legend

- `paper`: academic/public method reference
- `library`: implemented mainly through third-party library APIs
- `heuristic`: project-specific engineering logic (not a direct paper implementation)

## Feature -> Reference -> Code

| Feature | Core approach | Reference IDs | Type | Main code |
|---|---|---|---|---|
| Mesh flattening (ARAP) | ARAP optimization with cotangent edge weights | `[R1]`, `[R4]` | paper + heuristic | `src/core/flattener.py` |
| Mesh flattening (LSCM) | Least-squares conformal parameterization | `[R2]` | paper | `src/core/flattener.py` |
| Mesh flattening (area mode) | Tutte/LSCM blend + global area scale match | `[R3]`, `[R2]` | paper + heuristic | `src/core/flattener.py` |
| Cylindrical unwrap | PCA/world-axis candidates + circle-fit center + seam unwrap | `[R8]` | paper + heuristic | `src/core/flattener.py` |
| Flatten size stabilization | Outlier ratio guard (warning/correction metadata) | - | heuristic | `src/core/flattener.py` |
| Outer/Inner split (views mode) | centroid-binned depth visibility + topology propagation | `[R7]`, `[R5]` | paper + heuristic | `src/core/surface_separator.py` |
| Outer/Inner split (cylinder mode) | radial-thickness separation from cylinder fit | `[R8]` | paper + heuristic | `src/core/surface_separator.py` |
| Outer/Inner split (normals mode) | face-normal dot(reference direction) classification | `[R9]` | heuristic (common practice) | `src/core/surface_separator.py` |
| Seed-assist split (manual assist) | user seeds + auto prediction remap + conservative axis rule | - | heuristic | `src/core/surface_separator.py` |
| Migu inference | boundary strip / unknown bridge inference from labeled sides | - | heuristic | `src/core/surface_separator.py` |
| Real-time section slicing | plane/mesh intersection through trimesh section pipeline | `[L3]` | library + heuristic | `src/core/mesh_slicer.py`, `app_interactive.py` |
| Silhouette extraction (2D drawing) | occupancy raster + contour extraction (OpenCV/SciPy fallback) | `[L5]`, `[L2]`, `[R6]` | library + paper + heuristic | `src/core/profile_exporter.py` |
| Feature line extraction | dihedral-angle sharp edge detection | `[R9]` | heuristic (common practice) | `src/core/feature_line_extractor.py` |
| Rubbing image rendering | depth/value map + Sobel gradient shading + Gaussian filtering | `[L2]` | library + heuristic | `src/core/surface_visualizer.py` |
| Orthographic projection | orthographic projection/depth/silhouette style raster pipeline | `[R7]` | concept + heuristic | `src/core/orthographic_projector.py` |
| Unified SVG sheet export | top outline + section vectors + rubbing images composition | - | heuristic | `src/core/rubbing_sheet_exporter.py` |

## Practical interpretation

- If a feature row includes `heuristic`, that part is tuned for robust field data and GUI responsiveness.
- If a feature row includes only `library`, the behavior follows that library's implementation details and version.
