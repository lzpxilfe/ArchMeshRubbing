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
- `[R10]` Franklin C. Crow, [*Summed-Area Tables for Texture Mapping*](https://dl.acm.org/doi/10.1145/800031.808600), SIGGRAPH 1984. 디지털 탁본의 정수 integral image(누적합) 창 평균에 해당합니다.
- `[R11]` Pierre D. Wellner, [*Adaptive Thresholding for the DigitalDesk*](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-322.html), Xerox EuroPARC Technical Report EPC-93-110, 1993. 국소 평균을 기준선으로 삼아 국소 대비를 남기는 방식의 공개 레퍼런스입니다.

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
- `[L8]` [Shapely](https://shapely.readthedocs.io/) / [GEOS](https://libgeos.org/) - 외곽선 다각형 합집합과 ring/hole topology 판정의 계산 backend입니다. 두 버전은 `src/core/artifact_outline_extractor.py`의 `REQUIRED_SHAPELY_VERSION`, `REQUIRED_GEOS_VERSION`으로 고정되어 record에 기록됩니다.

---

## Notes

- 모든 기능이 특정 논문의 직접 재현은 아닙니다.
- `sectionwise flatten`, `tile recommendation policy`, `flatten size guard`, `rubbing-like contrast control` 등은 프로젝트 목적에 맞게 조합한 휴리스틱이 포함됩니다.
- 고고학 판독용 시각화는 `A-series` 문헌의 문제의식에서 영감을 받되, 현재 구현은 실사용 중심의 단순화된 렌더/강조 파이프라인입니다.
- `A-series`(PTM/RTI)는 문제의식 참고이며 구현 근거가 아닙니다. 현재 저장소에는 다중 광원 촬영, 픽셀별 반사 모델 적합, `.ptm`/`.rti` 입출력이 없습니다. 이 격차는 [`docs/COMPETITIVE_GAP_ANALYSIS.md`](COMPETITIVE_GAP_ANALYSIS.md)의 RTI 행과 같은 내용입니다.
