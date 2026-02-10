# References

ArchMeshRubbing implements production-oriented variants of known methods.
This document lists the bibliographic/official references.

For the feature-by-feature mapping, see:
- `docs/FEATURE_REFERENCES.md`

## Papers / Algorithms

- `[R1]` ARAP (As-Rigid-As-Possible)
  - Olga Sorkine, Marc Alexa, “As-Rigid-As-Possible Surface Modeling”, *SGP*, 2007.

- `[R2]` LSCM (Least Squares Conformal Maps)
  - Bruno Levy, Sylvain Petitjean, Nicolas Ray, Jérome Maillot, “Least Squares Conformal Maps for Automatic Texture Atlas Generation”, *SIGGRAPH*, 2002.

- `[R3]` Tutte embedding
  - W. T. Tutte, “How to Draw a Graph”, *Proceedings of the London Mathematical Society*, 1963.

- `[R4]` Cotangent/discrete differential operators (practical background)
  - Mark Meyer, Mathieu Desbrun, Peter Schroder, Alan H. Barr, “Discrete Differential-Geometry Operators for Triangulated 2-Manifolds”, 2002.

- `[R5]` Dijkstra shortest path
  - E. W. Dijkstra, “A Note on Two Problems in Connexion with Graphs”, 1959.

- `[R6]` Ramer-Douglas-Peucker polyline simplification
  - Urs Ramer, “An iterative procedure for the polygonal approximation of plane curves”, 1972.
  - David Douglas, Thomas Peucker, “Algorithms for the reduction of the number of points required to represent a digitized line or its caricature”, 1973.

- `[R7]` Z-buffer visibility concept
  - Edwin Catmull, “A Subdivision Algorithm for Computer Display of Curved Surfaces”, 1974. (classical depth-buffer visibility context)

- `[R8]` Circle fit (algebraic least-squares baseline)
  - I. Kasa, “A curve fitting procedure and its error analysis”, *IEEE Transactions on Instrumentation and Measurement*, 1976.

- `[R9]` Dihedral-angle sharp edge criterion
  - Standard mesh-processing practice: classify feature edges by dihedral angle between adjacent face normals.

## Libraries / Official Docs

- `[L1]` NumPy - https://numpy.org/
- `[L2]` SciPy - https://scipy.org/
- `[L3]` trimesh - https://trimsh.org/
- `[L4]` Pillow - https://python-pillow.org/
- `[L5]` OpenCV - https://docs.opencv.org/
- `[L6]` PyQt6 - https://doc.qt.io/qtforpython-6/
- `[L7]` PyOpenGL - https://pyopengl.sourceforge.net/

## Notes

- Not every behavior is a direct paper reproduction; some parts are engineering heuristics for stability/performance.
- When adding a new major algorithm, include both:
  - a bibliography entry here (`[R*]` or `[L*]`)
  - a feature mapping entry in `docs/FEATURE_REFERENCES.md`

