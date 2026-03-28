# 🏺 ArchMeshRubbing

> `Archaeology-first mesh recording tool`
>
> 3D 메쉬를 일반 CG 자산처럼 다루지 않고, **기록면(recording surface)** 과 **판독 가능한 산출물** 중심으로 다루는 고고학 연구용 데스크톱 도구입니다.

ArchMeshRubbing은 길쭉한 기와형 메쉬를 불러와,
`정위치 → 기록면 선택 → 장축 확인/수정 → 펼치기 → rubbing 시각화 → PNG/SVG export`
흐름으로 빠르게 첫 결과를 얻는 것을 목표로 합니다.

---

## ✨ 왜 이 도구인가요?

일반적인 3D 툴은 UV, seam, material 같은 CG 용어와 작업 흐름에 익숙할 때 강합니다.

ArchMeshRubbing은 반대로, 고고학 연구자가 익숙한 질문에서 출발합니다.

- `이 메쉬를 기록용으로 제대로 놓았는가?`
- `어느 면이 실제 판독 대상 기록면인가?`
- `장축 방향이 맞는가?`
- `곡률을 고려해 펼쳤을 때 문양/흔적이 읽히는가?`
- `논문/보고서에 바로 넣을 PNG/SVG를 뽑을 수 있는가?`

---

## 🧭 핵심 사용자 흐름

기본 흐름은 아래 6단계를 기준으로 설계되어 있습니다.

1. `파일 불러오기`
2. `정위치`
3. `기록면/관심영역 선택`
4. `장축 확인/수정`
5. `기와 추천 펼침(sectionwise 기본 추천)`
6. `rubbing 시각화 후 PNG/SVG export`

처음 쓰는 사용자도 **5분 안에 첫 결과**를 얻는 것이 목표입니다.

---

## 🪄 이번 구현에서 강화된 핵심 기능

### 1. 기와형 메쉬용 기본 추천 펼침

- 길이 방향이 뚜렷하고 곡면 단면이 반복되는 메쉬에서는 기본 추천을 `sectionwise flatten`으로 설정
- UI에서는 내부 알고리즘명이 아니라 **`기와 추천 펼침`** 으로 표기
- 추천 이유를 함께 표시
  - 예: `기와형/장축 반복 단면 패턴이 높아 기와 추천 펼침을 기본으로 권장`
- 사용자는 필요할 때만 아래 대안으로 전환 가능
  - `저왜곡 펼침`
  - `기록면 기반 펼침`
  - `곡면 추적 펼침`
  - `각도 보존 펼침`

### 2. 장축 기반 펼치기 + 수동 보정 흐름

- 장축 자동 추정
- 사용자의 수동 축 보정 반영
- 대표 단면/와통 반경 힌트 반영
- sectionwise 품질이 부족하면 fallback 경로 제공
  - `sectionwise → area → cylinder → arap`

### 3. rubbing-like 판독 시각화

예쁜 렌더보다 **문양/흔적 판독성**을 우선합니다.

- `normal` 기반 시각화
- `curvature` 기반 시각화
- `height/depth` 계열 시각화
- `contrast`, `strength` 조절 가능
- 논문/보고서용 PNG 저장 가능

### 4. 연구 산출물 중심 export

- `flattened 좌표`
- `기록면 전개 SVG`
- `rubbing PNG`
- `기록면 검토 시트`
- `6방향 실측 도면 패키지`

---

## 🧱 현재 flatten 모드 정리

| 사용자용 이름 | 내부 키 | 언제 추천하나 | 특징 |
|---|---|---|---|
| `기와 추천 펼침` | `section` | 기와형/장축 반복 단면이 뚜렷할 때 | 길이 방향 해석성과 단면 반복성을 우선 |
| `저왜곡 펼침` | `arap` | 일반 목적 기본 대안 | 형태 왜곡을 상대적으로 억제 |
| `기록면 기반 펼침` | `area` | 면적 안정성과 기록면 해석을 우선할 때 | 비교적 안정적인 전개 |
| `곡면 추적 펼침` | `cylinder` | 원통성/곡면 흐름이 분명할 때 | 축을 따라 펼침 |
| `각도 보존 펼침` | `lscm` | 각도 보존이 더 중요할 때 | conformal 계열 대안 |

---

## 🧠 제품 구조와 코어 구조

최근 리팩터로 flatten 코어를 책임별로 분리했습니다.

### Core layout

- [`src/core/flattener.py`](src/core/flattener.py): 공개 API, 오케스트레이션, 호환 계층
- [`src/core/flatten_policy.py`](src/core/flatten_policy.py): 기와 추천 정책, 대안 모드, fallback 체인
- [`src/core/flatten_models_arap.py`](src/core/flatten_models_arap.py): ARAP/LSCM/Tutte 기반 일반 flatten
- [`src/core/flatten_models_cylindrical.py`](src/core/flatten_models_cylindrical.py): cylindrical unwrap
- [`src/core/flatten_models_sectionwise.py`](src/core/flatten_models_sectionwise.py): sectionwise flatten
- [`src/core/flatten_utils.py`](src/core/flatten_utils.py): axis/seam/size guard/smoothing/mesh sanitize
- [`src/core/flatten_metrics.py`](src/core/flatten_metrics.py): distortion summary, 품질 지표
- [`src/core/flatten_types.py`](src/core/flatten_types.py): `FlattenedMesh`, `FlattenResultMeta`

### 왜 이렇게 나눴나요?

- `flattener.py` 단일 파일 비대화 해소
- 알고리즘별 테스트 가능성 개선
- 정책 계층과 엔진 계층 분리
- 사용자 언어와 내부 기술 언어 분리

---

## 🖼️ 주요 산출물

### 기록면 전개

- OBJ/PLY/STL/GLTF 메쉬에서 기록면을 펼친 2D 결과 생성
- 기와형 메쉬는 기본적으로 `기와 추천 펼침` 우선

### 디지털 탁본(rubbing-like)

- 표면 미세 요철을 읽기 쉽도록 대비 강화
- 포토리얼리스틱 렌더가 아니라 **판독 보조 이미지**에 집중

### 실측/검토 패키지

- 전개 SVG
- rubbing PNG
- review sheet
- 6방향 도면 패키지

---

## ⚡ Quick Start

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-optional.txt
python app_interactive.py
```

### Windows

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-optional.txt
python app_interactive.py
```

또는:

```bash
python main.py --gui
```

---

## 🖥️ CLI 예시

```bash
python main.py --help
python main.py mesh.obj
python main.py --open-project sample.amr
python main.py --info mesh.obj
python main.py --flatten mesh.obj unwrap.png
python main.py --review mesh.obj review.png
python main.py --generate-synthetic sugkiwa_quarter 7 synthetic_tile.obj
python main.py --benchmark-synthetic ./benchmarks 1,2,3
python main.py --project mesh.obj planview.png
```

`--flatten`, `--review`는 빠른 전체 경로용입니다.
상면/하면 기록면을 유도형으로 준비하려면 GUI 사용을 권장합니다.

---

## 📦 지원 포맷

- `OBJ`
- `PLY`
- `STL`
- `OFF`
- `glTF (.gltf)`
- `glTF Binary (.glb)`

---

## 📚 구현 참고 레퍼런스

이번 기능 구현은 “논문을 그대로 복제”했다기보다, **기하 처리 알고리즘 + 고고학 기록 시각화 관점 + 실사용 안정화 휴리스틱**을 조합한 형태입니다.

대표 참고 레퍼런스는 아래와 같습니다.

### Geometry / Flattening

- Olga Sorkine, Marc Alexa, [*As-Rigid-As-Possible Surface Modeling*](https://diglib.eg.org/handle/10.2312/SGP.SGP07.109-116)
- Bruno Lévy et al., [*Least Squares Conformal Maps for Automatic Texture Atlas Generation*](https://brunolevy.github.io/papers/LSCM_SIGGRAPH_2002.pdf)
- W. T. Tutte, [*How to Draw a Graph*](https://academic.oup.com/plms/article/s3-13/1/743/1531546)
- Mark Meyer et al., [*Discrete Differential-Geometry Operators for Triangulated 2-Manifolds*](https://authors.library.caltech.edu/records/0rsjd-50h08)
- I. Kasa, [*A Curve Fitting Procedure and its Error Analysis*](https://ieeexplore.ieee.org/document/6312298)

### Archaeology / Surface-reading inspiration

- Tom Malzbender et al., [*Polynomial Texture Maps*](https://shiftleft.com/mirrors/www.hpl.hp.com/techreports/2001/HPL-2001-33R1.pdf)
- Smithsonian MCI, [*Reflectance Transformation Imaging*](https://mci.si.edu/reflectance-transformation-imaging)
- Nicola Dellepiane et al., [*Archaeological applications of Polynomial Texture Mapping: analysis, conservation and representation*](https://www.sciencedirect.com/science/article/pii/S0305440310001093)

자세한 목록은 아래 문서를 참고하세요.

- [docs/REFERENCES.md](docs/REFERENCES.md)
- [docs/FEATURE_REFERENCES.md](docs/FEATURE_REFERENCES.md)

---

## 🧭 프로젝트 철학

이 프로젝트는 메쉬를 단순한 “렌더링 대상”으로 보지 않습니다.

- `정위치`는 도면 기준을 위한 단계
- `기록면 선택`은 연구 표면을 지정하는 단계
- `펼치기`는 해석 가능한 좌표계를 만드는 단계
- `rubbing`은 문양과 흔적을 읽기 위한 판독 보조 단계
- `export`는 연구 산출물을 만드는 단계

즉, 내부 계산은 메쉬를 사용하더라도,
사용자가 최종적으로 다루는 것은 **기록 가능한 2D 결과**여야 한다는 관점을 따릅니다.

---

## 📌 현재 상태

현재 버전은 특히 아래에 집중하고 있습니다.

- `기와형 메쉬 기본 추천 펼침(sectionwise)`
- `장축/단면/와통 정보를 활용하는 유도형 기록면 전개`
- `rubbing-like 시각화`
- `기록면 전개 SVG + PNG export`
- `4축 기본 UI 정리`
- `synthetic benchmark 기반 회귀 검증 기반`

---

## 📄 License

GNU General Public License v2.0 (GPLv2)
