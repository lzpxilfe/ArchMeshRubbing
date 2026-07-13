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

Align 확정 뒤에도 모든 기능이 한꺼번에 열리지는 않습니다. 현재 활성 Align의 고유한 `READY + FRESH` 기록을 기준으로 `Cutline 3/3 → Outline 6/6 → Digital Rubbing 6/6` 순서로 다음 단계가 열리고, 완료 버튼은 초록색으로 바뀝니다. application command도 같은 gate를 강제하며, 각 Outline은 Cutline 3면을, 각 Digital Rubbing은 dependency-valid Outline 6면을 직접 참조합니다. 이 선행 record coverage가 없으면 READY 기록도 완료 증거로 세지 않습니다. 같은 방향을 여러 번 기록해도 한 면으로 계산합니다. Align을 바꾸면 기존 기록은 삭제하지 않고 stale로 제외하며, 이전 Align을 다시 활성화하거나 프로젝트를 재열면 문서의 record graph에서 진행도를 그대로 복원합니다.

처음 쓰는 사용자도 **5분 안에 첫 결과**를 얻는 것이 목표입니다.

---

## 🪄 이번 구현에서 강화된 핵심 기능

### 0. 원본 보존형 ArtifactDocument + 검증 Cutline/Outline/Digital Rubbing

- 원본 file SHA-256, decode geometry SHA-256, 확인된 단위·축, immutable Align revision을 분리해 저장
- 새 문서는 원본의 절대 경로를 canonical manifest에 넣지 않고 `external:<original_name>` locator만 저장함. 실제 OS 경로는 현재 session에만 유지하므로 같은 원본·recipe의 문서와 SVG/PNG hash가 drive/root 위치에 따라 달라지지 않음
- native `.amr` 저장은 `ArtifactDocument`와 함께 검증된 주 원본 file bytes를 SHA-256 content-addressed blob으로 포함함. 외부 원본을 삭제하거나 프로젝트를 다른 컴퓨터로 옮겨도 `.amr` 하나에서 saved parser·단위·Align·geometry hash를 다시 검증해 열 수 있고, 열린 archive를 Save/Save As로 다시 저장할 수 있음
- 기존 manifest-only `.amr`는 계속 읽되 원본 선택이 필요함. 아직 dependency manifest가 없으므로 OBJ MTL·텍스처, PLY `TextureFile`, glTF/GLB 외부 buffer/image 요청은 자동 파일 탐색을 허용하지 않고 Open 단계에서 즉시 거부함
- 신규 import는 Trimesh·NumPy·Pillow parser subset digest, 정확한 parser flag, scene merge/sanitizer, `dependency_policy=deny_external`을 닫힌 `mesh-import-recipe 1.0`으로 기록하고, reopen·embedded materialization·export provenance까지 같은 receipt를 실행·비교함
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
- native DerivedRecord worker는 시작 session과 projection을, viewport의 cut-section/ROI/surface-selection worker는 worker identity·target mesh/TRS·render frame을 확인하여 늦은 결과가 현재 문서·overlay·다른 유물·새 worker를 덮지 못하게 함
- Cutline·Outline·Digital Rubbing worker는 공통 취소 Event를 계산 내부의 deterministic chunk 경계까지 전달하며, 사용자는 진행 창에서 강제 스레드 종료 없이 취소를 요청할 수 있음
- DerivedRecord 추가는 같은 render projection의 문서 binding만 compare-and-swap하며 live mesh·VBO·카메라·선택·preview cache를 다시 만들지 않음
- 대좌표 장면은 CPU·문서의 절대 float64 world-mm 좌표를 유지하면서, 객체별 VBO origin을 float64에서 먼저 빼 relative `GL_FLOAT`로 업로드하고 live scene의 안정적인 render origin에 camera·model transform을 rebase함
- 두 origin은 viewport 전용 transient 상태이며 ArtifactDocument·record·QC·hash·export에 기록하지 않음. mesh·cutline·ROI·pick·gizmo 등 활성 world overlay를 render-relative로 제출하고 CPU face 계산은 absolute float64를 유지함
- depth pick·screen projection·Ctrl drag는 해당 depth buffer를 그린 modelview·projection·viewport·scene origin과 visibility·ROI·X-ray·object TRS/geometry revision을 하나의 read-only frame authority로 묶어 다른 시점의 상태가 섞이지 않게 함
- 앱 시작 전에 OpenGL 2.1 compatibility·24-bit depth surface 계약을 명시하고, paint 뒤 depth readback/pick에 필요한 widget FBO attachment는 `PartialUpdate`로 보존함. 별도 native-process smoke가 실제 QOpenGLWidget context/FBO/VBO/pixel/depth/pick 경로를 검증함
- SVG/PNG export worker는 보이지 않는 same-parent staging package를 완전 검증해 exact inode/fingerprint capability를 만들고, GUI callback이 현재 Workbench의 source·Align·exact `READY + FRESH` record를 다시 확인한 뒤 빠른 재확인·rename으로만 공개
- export 중 같은 Align에 무관한 record가 추가돼도 안전하게 게시하지만 Align/Open 완료로 권위가 바뀌면 destination을 만들지 않고 자신이 소유한 staging만 정리함
- scene publication의 rollback·scene 복원·finalize 자체가 불확실하면 fatal authority 상태로 전환해 검증된 Open 전까지 저장·실측·내보내기를 차단

Native 문서에서는 기존 screenshot/OpenCV/convex-hull 2D 도면과 `SurfaceVisualizer`/flatten 기반 PNG·SVG를 측정 산출물로 내보내는 우회 경로를 차단합니다. 검증된 Cutline/Outline record는 `.amr-vector`, Digital Rubbing record는 `.amr-rubbing`으로 내보냅니다. Source checkout의 Python 3.12 품질 게이트와 Windows·macOS·Linux persistence matrix는 code commit `166103dcf0ea`의 [GitHub Actions run 29182584810](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29182584810)에서 모두 통과했습니다. 이어 commit `e4bf6dcac4b1`의 [Frozen package smoke run 29213279508](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29213279508)에서 Ubuntu·Windows·macOS frozen build와 실행 파일 self-test가 모두 통과했습니다. 공개 installer·서명·frozen native-GL 검증은 아직 별도 과제입니다.

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
- [`src/application/artifact_workflow_progress.py`](src/application/artifact_workflow_progress.py): 검증된 session의 `READY + FRESH` view coverage에서 Cutline 3면·Outline 6면·Digital Rubbing 6면 진행도와 순차 gate를 재구성
- [`src/application/artifact_measurements.py`](src/application/artifact_measurements.py): Cutline/Outline/Digital Rubbing의 immutable work item, Workbench 공유 예약·메모리 admission, cooperative cancellation, exact result capability와 same-Align rebase
- [`src/application/artifact_exports.py`](src/application/artifact_exports.py): vector/rubbing export의 exact capability, worker staging, 안전한 정리와 final-authority publication
- Open/new import/project reopen과 Align commit/parent activation은 ticket과 compare-and-swap 검증을 거쳐 `prepare → activate → finalize`되며, scene swap 실패는 이전 authority로 rollback
- Cutline/Outline/Digital Rubbing은 GUI에서 파라미터만 캡처하고 단일 worker에서 계산합니다. worker는 문서를 변경하지 않으며, 완료 결과는 예약된 record ID와 일회성 result capability를 검증한 뒤 현재의 같은 Align session에 rebase합니다. record append publication은 GPU 장면을 교체하지 않습니다. 재개방한 기록은 READY + FRESH 목록에서 명시적으로 선택하고, 일시적인 Open/scene 충돌로 게시하지 못한 측정 결과는 새 ID를 만들지 않고 재시도합니다.
- vector/rubbing export는 비싼 생성·탁본 재계산을 worker에서 staging까지만 수행합니다. 최종 no-replace rename은 현재 권위를 다시 검증한 뒤 실행하며, 목적지 경합에서는 기존 승자를 보존합니다. rename 뒤 directory `fsync`가 실패하거나 Windows처럼 지원 여부를 확인할 수 없으면 저장 완료와 crash durability 미확정을 구분해 경고합니다.

### Artifact trust core

- [`src/core/artifact_cancellation.py`](src/core/artifact_cancellation.py): GUI와 무관한 cooperative cancellation probe·고유 종료 신호
- [`src/core/artifact_document.py`](src/core/artifact_document.py): source·metadata·Align·DerivedRecord revision graph
- [`src/core/artifact_session.py`](src/core/artifact_session.py): 검증 source와 immutable document의 materialization 경계
- [`src/core/artifact_vector_extractor.py`](src/core/artifact_vector_extractor.py): canonical-mm Cutline
- [`src/core/artifact_outline_extractor.py`](src/core/artifact_outline_extractor.py): fixed-grid 6면 Outline
- [`src/core/artifact_rubbing_extractor.py`](src/core/artifact_rubbing_extractor.py): deterministic 6면 Digital Rubbing raster
- [`src/core/artifact_vector_export.py`](src/core/artifact_vector_export.py): `.amr-vector` 1:1 SVG package
- [`src/core/artifact_rubbing_export.py`](src/core/artifact_rubbing_export.py): `.amr-rubbing` canonical PNG package
- [`src/core/project_file.py`](src/core/project_file.py): strict AMR v2 저장·로딩과 원자 교체

### Renderer precision boundary

- [`src/gui/render_coordinates.py`](src/gui/render_coordinates.py): absolute float64 연구 좌표를 변경하지 않고 객체별 VBO origin과 scene render origin으로 GPU 표시 좌표만 rebasing하는 Qt/OpenGL-free 수학 경계
- [`src/gui/opengl_context.py`](src/gui/opengl_context.py): 앱과 실제 driver smoke가 공유하는 OpenGL 2.1 compatibility·24-bit depth 요청 계약
- [`src/gui/opengl_driver_smoke.py`](src/gui/opengl_driver_smoke.py): native QPA의 숨겨진 실제 `Viewport3D` widget FBO에서 survey-scale VBO·pixel·depth·pick을 readback하고 source-state/environment JSON을 남기는 독립 프로세스 게이트. compositor의 최종 on-screen presentation은 별도 범위
- main mesh VBO, camera/model transform, native vector preview, ground/grid와 활성 cutline·ROI·pick·gizmo 등 world overlay를 render-relative 제출로 이식
- 한 frame의 modelview·projection·viewport·scene origin을 묶은 `RenderFrameSnapshot`과 그 프레임의 visibility·ROI·X-ray·object TRS/geometry depth signature로 depth unprojection, screen projection, ray, Ctrl drag의 좌표·픽셀 계약을 일치시킴
- 라쏘/가시 면 worker 결과는 시작 객체·mesh·TRS·depth authority와 완료 시점을 다시 비교하며, magnetic depth-edge cache는 잡힌 frame authority와 함께만 재사용. 객체 전환 시 미완성 gesture/polygon도 종료
- mocked OpenGL·pure float64 게이트와 별도로, 로컬 macOS Apple M4와 Linux CI의 Xvfb+xcb+Mesa llvmpipe 실제 OpenGL context에서 `>= 1e9 mm` 장면의 0.25 mm gap·0.125 mm 높이차를 원근/정사영 모두 검증함. Linux actual-GL 61/61 결과는 commit `166103dcf0ea`의 run `29182584810`에 결합됨

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

CI/persistence의 일반 `QT_QPA_PLATFORM=offscreen` suite는 실제 QOpenGLWidget context를 생성하거나 검증하지 않습니다. macOS에서 native driver smoke를 별도 프로세스로 실행하려면 다음 명령을 사용합니다. `--report`는 기존 파일을 덮어쓰지 않으므로 매 실행마다 존재하지 않는 새 경로를 지정하세요. Linux의 Xvfb+Mesa 명령과 판정 범위는 [docs/QUALITY_GATES.md](docs/QUALITY_GATES.md)에 있습니다.

```bash
python -m src.gui.opengl_driver_smoke \
  --qt-platform cocoa \
  --report build/opengl-driver-smoke.json
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

이 명령은 기존 산출물을 기본적으로 덮어쓰지 않고, 빌드 뒤 실제 frozen executable의 offline self-test를 실행합니다. 현재 공개 바이너리는 만들지 않습니다. 3-OS frozen build/self-test는 원격 CI에서 통과했지만, 저장소의 GPLv2-only 표기와 bundled PyQt6의 GPL-3.0-only 라이선스 결정, 서명·notarization·installer·frozen native-GL 검증이 남아 있습니다. 자세한 절차와 차단 게이트는 [docs/NATIVE_PACKAGING.md](docs/NATIVE_PACKAGING.md)를 참고하세요.

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
합성 기와 생성·평가·review sheet 흐름은 [합성 벤치마크 가이드](docs/SYNTHETIC_BENCHMARKS.md)를 참고하세요.

---

## 📦 지원 포맷

원본 hash와 재현 가능한 geometry identity를 함께 보존하는 native 작업 문서의 권위 원본은 다음 포맷을 지원합니다.

- sidecar를 참조하지 않는 `OBJ`
- `TextureFile`을 참조하지 않는 `PLY`
- `STL`
- `OFF`
- 외부 URI가 없는 self-contained `glTF Binary (.glb)`

외부 `.bin` 파일을 참조할 수 있는 `glTF (.gltf)`는 parser 호환성 self-test만 유지하며, sidecar까지 하나의 source identity로 묶는 manifest가 구현되기 전에는 권위 원본으로 열지 않습니다. `.glb`도 외부 image/buffer URI가 있으면 거부됩니다. 현재는 모든 buffer와 image를 포함한 self-contained `.glb`로 변환해 사용하세요.

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
- 주 원본 bytes를 포함한 content-addressed `.amr` 저장, 원본 삭제 뒤 독립 프로세스 reopen, archive-to-archive 재저장
- versioned closed mesh import recipe, parser-runtime subset identity, 외부 sidecar deny gate와 동일 receipt의 독립 프로세스 재실행
- 기와형 메쉬 기본 추천 펼침과 synthetic benchmark는 legacy 전문 기능으로 유지
- code commit `898a8bfc144f` 기준 Python 3.12 macOS arm64 frozen 앱의 10-check offline self-test 통과
- commit `e4bf6dcac4b1`의 원격 CI에서 Ubuntu·Windows·macOS frozen build와 각 실행 파일 self-test 모두 통과
- Open/Align authority와 two-phase scene publication을 Qt/OpenGL-free `ArtifactWorkbench`로 이식
- Cutline/Outline/Digital Rubbing command와 worker 수명주기를 Qt/OpenGL-free application shell로 이식
- DerivedRecord의 VBO-free binding rebind와 SVG/PNG worker staging → final-authority publication 이식
- dependency-valid `READY + FRESH` record graph와 application command에서 Cutline 3/3 → Outline 6/6 → Digital Rubbing 6/6 순차 gate·초록 완료 표시·재열기/Align 복원 구현
- Cutline 면·경로, Outline fixed-grid/union/topology, Digital Rubbing raster/relief 내부의 안전 경계까지 사용자 cooperative cancellation 연결
- 대좌표 render-origin 이식: relative VBO·camera/model rebasing·world overlay 제출과 frame-bound depth picking/drag 계약 구현
- 실제 OpenGL driver smoke 구현: code commit `f25b424d6936`의 clean source tree, Python 3.12.13/macOS Apple M4에서 61개 context/FBO/VBO/pixel/depth/pick 조건 통과, 0.125 mm 높이차를 원근 `0.124783 mm`, 정사영 `0.124998 mm`로 복원. report는 commit/tree 상태·runtime lock·dependency version·UTC 시각을 기록함
- source checkout CI 검증: commit `166103dcf0ea`의 run `29182584810`에서 quality `572 passed`, macOS/Ubuntu persistence 각 `501 passed`, Windows persistence `498 passed + 3 platform-specific skips`, Linux llvmpipe actual-GL `61/61` 통과
- 다음 단계: content-addressed sidecar dependency manifest와 bundle resolver, Windows·macOS native QPA와 3-OS frozen actual-GL 확대, 라이선스 결정, 대표 GPU·대용량 실제 유물 pilot 진행

---

## 📄 License

GNU General Public License v2.0 (GPLv2)

현재 문구는 `or later`를 포함하지 않습니다. PyQt6를 포함한 공개 바이너리 배포는 [native packaging license gate](docs/NATIVE_PACKAGING.md#라이선스)가 해결될 때까지 보류합니다.

## Citation

이 저장소가 연구, 수업, 현장 업무에 도움이 되었다면 GitHub의 **Cite this repository** 버튼으로 인용해 주세요.

[![Cite this repository](https://img.shields.io/badge/Cite_this-repository-2ea44f?logo=github)](https://github.com/lzpxilfe/ArchMeshRubbing)
[![Star this repository](https://img.shields.io/github/stars/lzpxilfe/ArchMeshRubbing?style=social)](https://github.com/lzpxilfe/ArchMeshRubbing)

인용 메타데이터는 [CITATION.cff](CITATION.cff)에 보관합니다.
