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

- `[R10]` Polynomial Texture Maps (PTM)
  - Tom Malzbender, Dan Gelb, Hans Wolters, “Polynomial Texture Maps,” *ACM SIGGRAPH 2001 Conference Proceedings*, 2001, pp. 519-528.
  - Technical Report version: HP Labs, *HPL-2001-33R1 Polynomial Texture Maps*.
  - https://shiftleft.com/mirrors/www.hpl.hp.com/techreports/2001/HPL-2001-33R1.html

- `[R11]` Reflectance transformation / PTM origin
  - T. Malzbender et al., “Enhancement of Shape Perception by Surface Reflectance Transformation,” *HP Labs Technical Report HPL-2000-38R1*, 2000.
  - https://shiftleft.com/mirrors/www.hpl.hp.com/techreports/2000/HPL-2000-38R1.html

- `[R12]` Bilateral filtering
  - C. Tomasi, R. Manduchi, “Bilateral Filtering for Gray and Color Images,” *ICCV*, 1998.
  - https://doi.org/10.1109/ICCV.1998.710815

- `[R13]` CLAHE
  - Karel J. Zuiderveld, “Contrast Limited Adaptive Histogram Equalization,” *Graphics Gems IV*, 1994.
  - https://doi.org/10.1016/b978-0-12-336156-1.50061-6

- `[R14]` Archaeological PTM / RTI deployment
  - Graeme Earl, Kirk Martinez, Tom Malzbender, “Archaeological applications of polynomial texture mapping: analysis, conservation and representation,” *Journal of Archaeological Science*, 2010.
  - https://doi.org/10.1016/j.jas.2010.03.009

- `[R15]` RTI workflows and implementation
  - Ted Kinsman, “An Easy to Build Reflectance Transformation Imaging (RTI) System,” *Journal of Biocommunication*, 2016.
  - https://doi.org/10.5210/jbc.v40i1.6625

## Libraries / Official Docs

- `[L1]` NumPy - https://numpy.org/
- `[L2]` SciPy - https://scipy.org/
- `[L3]` trimesh - https://trimsh.org/
- `[L4]` Pillow - https://python-pillow.org/
- `[L5]` OpenCV - https://docs.opencv.org/
- `[L6]` PyQt6 - https://doc.qt.io/qtforpython-6/
- `[L7]` PyOpenGL - https://pyopengl.sourceforge.net/
- `[L8]` OpenCV CLAHE API - https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html
- `[L9]` OpenCV Bilateral Filter API - https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
- `[L10]` OpenCV filtering tutorials (smoothing, denoise, bilateral) - https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

## Notes

- Not every behavior is a direct paper reproduction; some parts are engineering heuristics for stability/performance.
- When adding a new major algorithm, include both:
  - a bibliography entry here (`[R*]` or `[L*]`)
  - a feature mapping entry in `docs/FEATURE_REFERENCES.md`
