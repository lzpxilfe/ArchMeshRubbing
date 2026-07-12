# 🏺 ArchMeshRubbing

> `Archaeology-first mesh recording tool`
>
> 3D 메쉬를 일반 CG 자산처럼 다루지 않고, **기록면(recording surface)** 과 **판독 가능한 산출물** 중심으로 다루는 고고학 연구용 데스크톱 도구입니다.

ArchMeshRubbing은 스캔한 문화유산 3D 메쉬를 원본 보존형 연구 자료로 불러와,
`Open → 단위·축 확인 → Align → Cutline/Outline → Digital Rubbing → 1:1 SVG/PNG export`
흐름으로 기록하고 다시 검증하는 오프라인 오픈소스 워크벤치를 목표로 합니다. 기와형 메쉬의 기록면 전개 기능은 이 기반 위에 남아 있는 전문 워크플로우입니다.

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

새 native 흐름은 아래 6단계를 기준으로 설계되어 있습니다.

1. `원본 파일 불러오기와 SHA-256 확인`
2. `단위·축 확인 및 정위치(Align revision) 확정`
3. `Top/Front/Right Cutline 기록`
4. `6면 Outline 기록`
5. `6면 Digital Rubbing 계산·기록`
6. `READY + FRESH 기록에서 1:1 SVG/PNG package export`

Open 직후 만들어지는 `recipe.kind="initial_identity"` Align은 canonical materialization을 위한 기준점이지 연구자의 정위치 확정이 아닙니다. 사용자가 변화량이 0인 경우까지 포함해 첫 Align을 명시적으로 확정하기 전에는 workflow가 `ALIGN_REQUIRED`에 머물며 Cutline/Outline/Digital Rubbing과 vector/rubbing export가 비활성화됩니다. 첫 확정은 immutable child Align revision을 남기고 `MEASUREMENT_READY`로 전환하며, parent activation으로 초기 기준점에 돌아가면 다시 측정이 잠깁니다.

처음 쓰는 사용자도 **5분 안에 첫 결과**를 얻는 것이 목표입니다.

---

## 🪄 이번 구현에서 강화된 핵심 기능

### 0. 원본 보존형 ArtifactDocument + 검증 Cutline/Outline/Digital Rubbing

- 원본 file SHA-256, decode geometry SHA-256, 확인된 단위·축, immutable Align revision을 분리해 저장
- Open → Align commit → Cutline record가 항상 source-space 원본에서 canonical millimeter로 다시 계산됨
- Top/Front/Right 단면을 명시적 right-handed plane frame으로 기록
- 화면용 단면 tape나 world XY 투영을 SVG 원본으로 사용하지 않음
- Cutline payload·recipe·QC를 RFC 8785 semantic hash와 함께 `.amr`에 보존
- Top/Bottom/Front/Back/Right/Left 6면 Outline을 전체 삼각형의 고정 mm 격자 투영 합집합으로 계산
- Outline의 오목부·구멍·분리 성분을 모두 보존하고 self-intersection·hole 소유권·component 중첩을 저장/재로딩 때 재검증
- Shapely 2.1.2 + GEOS 3.13.1, precision grid, grid collapse/merge receipt를 recipe·QC에 고정
- `READY + FRESH` 기록만 `*.amr-vector/`의 1:1 `artifact.svg` + provenance sidecar로 내보냄
- Digital Rubbing은 6면 canonical frame, 정수 pixels/mm·µm recipe, front-depth raster와 QC를 `raster.digital_rubbing.v1` record로 보존
- canonical GA8 PNG는 고정 chunk/DEFLATE bytes와 exact `pHYs`를 사용하며, `*.amr-rubbing/`에 provenance sidecar와 함께 저장
- vector/rubbing package는 원본 mesh와 GUI가 없어도 이동 후 별도 프로세스에서 offline 검증 가능
- Qt/OpenGL과 분리된 `ArtifactWorkbench`가 ticketed Open, 명시적 Align readiness, `state_version`/`authority_epoch` 기반 publication을 소유
- native DerivedRecord worker는 시작 session과 projection을, viewport의 cut-section/ROI worker는 worker identity와 projection generation을 확인하여 늦은 결과가 현재 문서·overlay·새 worker를 덮지 못하게 함
- scene publication의 rollback·scene 복원·finalize 자체가 불확실하면 fatal authority 상태로 전환해 검증된 Open 전까지 저장·실측·내보내기를 차단

Native 문서에서는 기존 screenshot/OpenCV/convex-hull 2D 도면과 `SurfaceVisualizer`/flatten 기반 PNG·SVG를 측정 산출물로 내보내는 우회 경로를 차단합니다. 검증된 Cutline/Outline record는 `.amr-vector`, Digital Rubbing record는 `.amr-rubbing`으로 내보냅니다. 로컬 차단 테스트와 3-OS CI matrix 구성은 완료됐지만 원격 Windows·macOS·Linux matrix 통과 및 설치형 바이너리 배포는 아직 확인 전입니다.

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

### 3. Legacy rubbing-like 판독 시각화

예쁜 렌더보다 **문양/흔적 판독성**을 우선하는 기존 검토 경로입니다. 이 결과는 연구 검토 이미지이며, 위 `raster.digital_rubbing.v1`의 재현 가능한 1:1 PNG와 구분합니다.

- `normal` 기반 시각화
- `curvature` 기반 시각화
- `height/depth` 계열 시각화
- `contrast`, `strength` 조절 가능
- 논문/보고서용 PNG 저장 가능

### 4. 연구 산출물 중심 export

- `*.amr-vector/`: 검증 Cutline/Outline 1:1 SVG + provenance
- `*.amr-rubbing/`: 검증 Digital Rubbing 1:1 PNG + provenance
- `flattened 좌표`
- `기록면 전개 SVG` (legacy 검토용)
- `rubbing PNG` (legacy 검토용)
- `기록면 검토 시트` (legacy 검토용)
- `6방향 도면 패키지` (legacy 검토용)

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

새 측정 경로는 GUI와 분리된 headless 신뢰 코어를 사용합니다.

### Application workflow shell

- [`src/application/artifact_workbench.py`](src/application/artifact_workbench.py): 한 artifact의 session/project path, 단일 pending Open, workflow readiness와 projection publication authority
- [`src/application/artifact_measurements.py`](src/application/artifact_measurements.py): Cutline/Outline/Digital Rubbing의 immutable work item, Workbench 공유 예약·메모리 admission, 취소, exact result capability와 same-Align rebase
- Open/new import/project reopen과 Align commit/parent activation은 ticket과 compare-and-swap 검증을 거쳐 `prepare → activate → finalize`되며, scene swap 실패는 이전 authority로 rollback
- Cutline/Outline/Digital Rubbing은 GUI에서 파라미터만 캡처하고 단일 worker에서 계산합니다. worker는 문서를 변경하지 않으며, 완료 결과는 예약된 record ID와 일회성 result capability를 검증한 뒤 현재의 같은 Align session에 rebase합니다. 재개방한 기록은 READY + FRESH 목록에서 명시적으로 선택하고, 일시적인 Open/scene 충돌로 게시하지 못한 결과는 새 ID를 만들지 않고 재시도합니다.

### Artifact trust core

- [`src/core/artifact_document.py`](src/core/artifact_document.py): source·metadata·Align·DerivedRecord revision graph
- [`src/core/artifact_session.py`](src/core/artifact_session.py): 검증 source와 immutable document의 materialization 경계
- [`src/core/artifact_vector_extractor.py`](src/core/artifact_vector_extractor.py): canonical-mm Cutline
- [`src/core/artifact_outline_extractor.py`](src/core/artifact_outline_extractor.py): fixed-grid 6면 Outline
- [`src/core/artifact_rubbing_extractor.py`](src/core/artifact_rubbing_extractor.py): deterministic 6면 Digital Rubbing raster
- [`src/core/artifact_vector_export.py`](src/core/artifact_vector_export.py): `.amr-vector` 1:1 SVG package
- [`src/core/artifact_rubbing_export.py`](src/core/artifact_rubbing_export.py): `.amr-rubbing` canonical PNG package
- [`src/core/project_file.py`](src/core/project_file.py): strict AMR v2 저장·로딩과 원자 교체

기존 기와 기록면 기능도 flatten 코어를 책임별로 분리해 유지합니다.

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

### 디지털 탁본

- native 경로는 원본과 활성 Align에서 6면 raster를 다시 계산하고 recipe·QC·raster hash를 기록
- `artifact.png`는 1:1 planar sampling과 exact pixels-per-meter를 선언
- 포토리얼리스틱 렌더가 아니라 **재현 가능한 판독 보조 이미지**에 집중

### 실측/검토 패키지

- 전개 SVG
- rubbing PNG
- review sheet
- 6방향 도면 패키지

위 전개·review sheet·기존 6방향 도면은 현재 **검토용 legacy 산출물**입니다. 1:1 측정 산출물로 검증되는 새 경로는 ArtifactDocument record에서 생성한 `.amr-vector` SVG와 `.amr-rubbing` PNG입니다.

---

## ⚡ Quick Start

현재 Quick Start는 source checkout 실행 절차입니다. 서명된 Windows·macOS·Linux 설치 파일은 아직 제공하지 않으며, 첫 공개 안정판 전에 OS별 frozen build·설치·실행 검증을 별도 게이트로 통과해야 합니다.

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-optional.txt
python app_interactive.py
```

### Windows

```bat
py -3.12 -m venv .venv
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

## 🧪 개발 품질 게이트

개발·CI 환경은 런타임 의존성과 함께 `requirements-dev.txt`를 설치합니다.

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m ruff check .
python -c "import subprocess,sys; raise SystemExit(subprocess.call([sys.executable,'-m','pyright','--pythonpath',sys.executable,'-p','pyright-m0.json']))"
python -m pytest -q
```

`pyright-m0.json`은 현재 M0 신뢰 커널 범위의 차단 게이트입니다. 전체 트리 타입 검사는 아직 부채를 보고하는 단계이며, 통과를 뜻하지 않습니다. 재현 환경, 게이트 범위, GUI 스모크 테스트의 한계는 [docs/QUALITY_GATES.md](docs/QUALITY_GATES.md)에 기록합니다.

프로젝트 저장 형식, 원자적 저장 보장, 원본 SHA-256의 범위와 v1 migration 정책은 [docs/PROJECT_FORMAT.md](docs/PROJECT_FORMAT.md)를 참고하세요.

구조 개편 방향과 보존/교체 경계는 [docs/ARCHITECTURE_DECISION.md](docs/ARCHITECTURE_DECISION.md)에 기록합니다.

### 로컬 native smoke build

Python 3.12의 깨끗한 환경에서만 unsigned 로컬 앱을 만듭니다.

```bash
python -m pip install -r requirements/build-py312.lock
python tools/build_native.py
```

이 명령은 기존 산출물을 기본적으로 덮어쓰지 않고, 빌드 뒤 실제 frozen executable의 offline self-test를 실행합니다. 현재 공개 바이너리는 만들지 않습니다. 저장소의 GPLv2-only 표기와 bundled PyQt6의 GPL-3.0-only 라이선스 결정을 먼저 해결해야 하며, 서명·notarization·실제 3-OS 원격 결과도 남아 있습니다. 자세한 절차와 차단 게이트는 [docs/NATIVE_PACKAGING.md](docs/NATIVE_PACKAGING.md)를 참고하세요.

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

원본 hash와 재현 가능한 geometry identity를 함께 보존하는 native 작업 문서의 권위 원본은 다음 포맷을 지원합니다.

- `OBJ`
- `PLY`
- `STL`
- `OFF`
- `glTF Binary (.glb)`

외부 `.bin` 파일을 참조할 수 있는 `glTF (.gltf)`는 parser 호환성만 유지하며, sidecar까지 하나의 source identity로 묶는 manifest가 구현되기 전에는 권위 원본으로 열지 않습니다. 현재는 self-contained `.glb`로 변환해 사용하세요.

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

- 원본 hash·명시적 단위·immutable Align revision 기반 `ArtifactDocument`
- canonical-mm Cutline과 6면 fixed-grid Outline
- recipe·QC·receipt가 있는 6면 Digital Rubbing과 1:1 `.amr-rubbing` export
- `.amr-vector`/`.amr-rubbing`의 이동 가능한 offline 검증
- 기와형 메쉬 기본 추천 펼침과 synthetic benchmark는 legacy 전문 기능으로 유지
- Python 3.12 macOS arm64 frozen 앱의 10-check offline self-test 통과
- Open/Align authority와 two-phase scene publication을 Qt/OpenGL-free `ArtifactWorkbench`로 이식
- Cutline/Outline/Digital Rubbing command와 worker 수명주기를 Qt/OpenGL-free application shell로 이식
- 다음 단계: export 작업의 최종 authority 재검증, record-only scene rebind, 실제 원격 3-OS CI, 라이선스 결정, GPU/대용량 유물 pilot 진행

---

## 📄 License

GNU General Public License v2.0 (GPLv2)

현재 문구는 `or later`를 포함하지 않습니다. PyQt6를 포함한 공개 바이너리 배포는 [native packaging license gate](docs/NATIVE_PACKAGING.md#라이선스)가 해결될 때까지 보류합니다.

## Citation

이 저장소가 연구, 수업, 현장 업무에 도움이 되었다면 GitHub의 **Cite this repository** 버튼으로 인용해 주세요.

[![Cite this repository](https://img.shields.io/badge/Cite_this-repository-2ea44f?logo=github)](https://github.com/lzpxilfe/ArchMeshRubbing)
[![Star this repository](https://img.shields.io/github/stars/lzpxilfe/ArchMeshRubbing?style=social)](https://github.com/lzpxilfe/ArchMeshRubbing)

인용 메타데이터는 [CITATION.cff](CITATION.cff)에 보관합니다.
