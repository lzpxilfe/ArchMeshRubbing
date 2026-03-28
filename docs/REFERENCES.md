# References

ArchMeshRubbing은 논문 구현체 그 자체라기보다,
알고리즘 레퍼런스와 연구도구용 휴리스틱을 결합한 프로젝트입니다.

이 문서는 다음 두 범주의 참고 자료를 정리합니다.

- `알고리즘/수치기하 레퍼런스`
- `고고학 기록 시각화/판독 레퍼런스`

기능별 매핑은 아래 문서를 참고하세요.

- [`docs/FEATURE_REFERENCES.md`](FEATURE_REFERENCES.md)

---

## R-series: Geometry, parameterization, fitting

- `[R1]` Olga Sorkine, Marc Alexa, [*As-Rigid-As-Possible Surface Modeling*](https://diglib.eg.org/handle/10.2312/SGP.SGP07.109-116), SGP 2007.
- `[R2]` Bruno Lévy, Sylvain Petitjean, Nicolas Ray, Jérome Maillot, [*Least Squares Conformal Maps for Automatic Texture Atlas Generation*](https://brunolevy.github.io/papers/LSCM_SIGGRAPH_2002.pdf), SIGGRAPH 2002.
- `[R3]` W. T. Tutte, [*How to Draw a Graph*](https://academic.oup.com/plms/article/s3-13/1/743/1531546), Proceedings of the London Mathematical Society, 1963.
- `[R4]` Mark Meyer, Mathieu Desbrun, Peter Schröder, Alan H. Barr, [*Discrete Differential-Geometry Operators for Triangulated 2-Manifolds*](https://authors.library.caltech.edu/records/0rsjd-50h08), 2002.
- `[R5]` E. W. Dijkstra, *A Note on Two Problems in Connexion with Graphs*, Numerische Mathematik, 1959.
- `[R6]` I. Kasa, [*A Curve Fitting Procedure and its Error Analysis*](https://ieeexplore.ieee.org/document/6312298), IEEE Transactions on Instrumentation and Measurement, 1976.
- `[R7]` Edwin Catmull, *A Subdivision Algorithm for Computer Display of Curved Surfaces*, 1974.
- `[R8]` Urs Ramer, *An iterative procedure for the polygonal approximation of plane curves*, 1972.
- `[R9]` David Douglas, Thomas Peucker, *Algorithms for the reduction of the number of points required to represent a digitized line or its caricature*, 1973.

---

## A-series: Archaeology / surface reading / imaging inspiration

- `[A1]` Tom Malzbender, Dan Gelb, Hans Wolters, [*Polynomial Texture Maps*](https://shiftleft.com/mirrors/www.hpl.hp.com/techreports/2001/HPL-2001-33R1.pdf), HP Labs / SIGGRAPH 2001.
- `[A2]` Smithsonian Museum Conservation Institute, [*Reflectance Transformation Imaging*](https://mci.si.edu/reflectance-transformation-imaging).
- `[A3]` Nicola Dellepiane, Mauro Callieri, Matteo Pittaluga, Roberto Scopigno, [*Archaeological applications of polynomial texture mapping: analysis, conservation and representation*](https://www.sciencedirect.com/science/article/pii/S0305440310001093), Journal of Archaeological Science, 2011.
- `[A4]` Historic England, [*Multi-light Imaging - Highlight-Reflectance Transformation Imaging (H-RTI) for Cultural Heritage*](https://historicengland.org.uk/images-books/publications/multi-light-imaging-heritage-applications/), 2018.

---

## L-series: Libraries / official docs

- `[L1]` [NumPy](https://numpy.org/)
- `[L2]` [SciPy](https://scipy.org/)
- `[L3]` [trimesh](https://trimsh.org/)
- `[L4]` [Pillow](https://python-pillow.org/)
- `[L5]` [OpenCV](https://docs.opencv.org/)
- `[L6]` [PyQt6 / Qt for Python](https://doc.qt.io/qtforpython-6/)
- `[L7]` [PyOpenGL](https://pyopengl.sourceforge.net/)

---

## Notes

- 모든 기능이 특정 논문의 직접 재현은 아닙니다.
- `sectionwise flatten`, `tile recommendation policy`, `flatten size guard`, `rubbing-like contrast control` 등은 프로젝트 목적에 맞게 조합한 휴리스틱이 포함됩니다.
- 고고학 판독용 시각화는 `A-series` 문헌의 문제의식에서 영감을 받되, 현재 구현은 실사용 중심의 단순화된 렌더/강조 파이프라인입니다.
