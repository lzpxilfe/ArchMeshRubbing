# References

ArchMeshRubbing은 논문 구현체 그 자체라기보다,
알고리즘 레퍼런스와 연구도구용 휴리스틱을 결합한 프로젝트입니다.

이 문서는 다음 두 범주의 참고 자료를 정리합니다.

- `알고리즘/수치기하 레퍼런스`
- `고고학 기록 시각화/판독 레퍼런스`
- `한국 실측 도면 관례 교재`

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

## K-series: Korean measured-drawing conventions (공개 교육 교재)

도면 관례의 수치와 표현 방식은 이 계열에서만 옮긴다. 상용 프로그램 화면이나 기억에서 옮기지 않는다([`docs/DRAWING_CONVENTIONS.md`](DRAWING_CONVENTIONS.md)의 클린룸 원칙).

- `[K1]` 한국문화재조사연구기관협회(현 한국문화유산협회), 2013, 『매장문화재전문교육 - 유물 실측의 이해』 (2013. 11. 12 - 11. 15 교육 교재). 김동숙(성림문화재연구원) "유물 실측의 기본 원리", "토기 종류에 따른 실측법", "석기 및 금속류 실측법". 유물제도 펜 굵기(그림 27, p.25: 단면 0.6, 평면·입면 0.4, 결실부 0.1; 강조 실선 0.3, 허선·세부 0.1), 정면 수법 표현하기(p.19), 손누름흔·지두흔·무문 타날흔·내박자흔 구분(p.35: 지두흔 1-2 cm), 타날흔 관찰(p.37: 타날은 탁본으로 기록), 실선·허선 넣기(p.19: 허선 간격 3 mm 내외).
- `[K2]` 한국문화재조사연구기관협회(현 한국문화유산협회), 2014, 『매장문화재조사 전문교육 - 유물실측의 이해 - 충청·호남권역 교육』 (2014. 7. 21 - 7. 25 교육 교재). 나건주(금강문화유산연구원) "선사시대 토기 실측법". 테쌓기흔적·손누름자국·목리조정흔의 도면 표현(도면 2 - 도면 7, pp.17-22): 손누름자국은 점토띠 경계를 따라 횡으로 열을 지은 작은 타원, 테쌓기 경계는 가는 횡선, 목리조정흔은 방향을 가진 가는 평행선 군집, 마연은 가늘고 긴 렌즈상 단위.
- `[K3]` 한국문화유산협회, 2015, 『KAAH 2015년도 매장문화재조사 전문교육 - 유물실측의 이해 - 중부권역』 (2015. 7. 20 - 7. 24 교육 교재). 원저단경호 제작방법과 타날문 종류(pp.9-12: 격자문·평행선문·승문, 타날 방향), 평기와 속성 용어(pp.34-42: 타날판 단판·중판·장판, 타날 방향, 하단 내면 물손질·깎기), 반파된 자기 도면 복원 실측(pp.45-49: 도면 복원한 유물은 중심선을 점선으로).

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
