# References

ArchMeshRubbing is an open-source tool for producing measurement-friendly 2D outputs (SVG/images) from archaeological 3D scan meshes.
Below are public references for the main algorithms and techniques used in the project.

## Mesh Parameterization / Flattening

- **ARAP (As-Rigid-As-Possible) surface modeling / parameterization**
  - Olga Sorkine, Marc Alexa, “As-Rigid-As-Possible Surface Modeling”, *Symposium on Geometry Processing (SGP)*, 2007.
- **LSCM (Least Squares Conformal Maps)**
  - Bruno Lévy, Sylvain Petitjean, Nicolas Ray, Jérome Maillot, “Least Squares Conformal Maps for Automatic Texture Atlas Generation”, *SIGGRAPH*, 2002.

## Polyline Simplification

- **Ramer–Douglas–Peucker (RDP) algorithm**
  - Urs Ramer, “An iterative procedure for the polygonal approximation of plane curves”, 1972.
  - David Douglas, Thomas Peucker, “Algorithms for the reduction of the number of points required to represent a digitized line or its caricature”, 1973.

## Rendering / Projection

- **Z-buffer / depth buffer visibility**
  - Conceptually equivalent to classic Z-buffer visibility in rasterization; used for “which faces are visible from top/bottom” heuristics in `src/core/surface_separator.py`.

## Geometry Features

- **Sharp edge / feature line detection**
  - Common practice: classify edges using dihedral angles between adjacent face normals (no single canonical paper; the implementation here follows the standard definition).

## Notes

- This document is for attribution and learning; it is *not* a dependency list.
- If you contribute a new algorithm, please add a short reference here (paper/book/blog) when possible.

