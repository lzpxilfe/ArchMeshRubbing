# ArchMeshRubbing

> `Archaeology-first mesh recording tool`
>
> 3D 메쉬를 일반 CG 자산처럼 다루지 않고, **기록면(recording surface)** 과 **판독 가능한 산출물** 중심으로 다루는 고고학 연구용 데스크톱 도구입니다.

ArchMeshRubbing은 스캔한 문화유산 3D 메쉬를 원본 보존형 연구 자료로 불러와,
`Open → 단위·축 확인 → Align → Cutline/Outline·기와 기록면 전개 → Digital Rubbing → 1:1 export`
흐름으로 기록하고 다시 검증하는 오프라인 오픈소스 워크벤치를 목표로 합니다. 기와형 메쉬의 기록면 전개도 이제 같은 원본·Align·record 신뢰 경계 안에서 계산할 수 있습니다.

첫 공개 안정판의 필수 데스크톱 대상은 **Windows**입니다. macOS·Linux용 source 호환 코드는 보존하지만, 이번 안정판의 완료 조건과 패키지 CI에는 포함하지 않습니다.

공개 경쟁 기능과 현재 격차를 과장 없이 추적하는 기준은 [`docs/COMPETITIVE_GAP_ANALYSIS.md`](docs/COMPETITIVE_GAP_ANALYSIS.md)에 기록합니다.

---

## 왜 이 도구인가요?

일반적인 3D 툴은 UV, seam, material 같은 CG 용어와 작업 흐름에 익숙할 때 강합니다.

ArchMeshRubbing은 반대로, 고고학 연구자가 익숙한 질문에서 출발합니다.

- `이 메쉬를 기록용으로 제대로 놓았는가?`
- `어느 면이 실제 판독 대상 기록면인가?`
- `장축 방향이 맞는가?`
- `곡률을 고려해 펼쳤을 때 문양/흔적이 읽히는가?`
- `논문/보고서에 바로 넣을 PNG/SVG를 뽑을 수 있는가?`

---

## 핵심 사용자 흐름

새 native 흐름은 아래 6단계를 기준으로 설계되어 있습니다.

1. `원본 파일 불러오기와 SHA-256 확인`
2. `단위·축 확인 및 정위치(Align revision) 확정`
3. `Top/Front/Right Cutline 기록`
4. `6면 Outline 기록`
5. `6면 Digital Rubbing 계산·기록`
6. `READY + FRESH 기록에서 1:1 SVG/PNG package export`

Open 직후 만들어지는 `recipe.kind="initial_identity"` Align은 canonical materialization을 위한 기준점이지 연구자의 정위치 확정이 아닙니다. 사용자가 변화량이 0인 경우까지 포함해 첫 Align을 명시적으로 확정하기 전에는 workflow가 `ALIGN_REQUIRED`에 머물며 Cutline/Outline/Digital Rubbing/기와 전개와 vector/rubbing/tile-unwrap export가 비활성화됩니다. 첫 확정은 immutable child Align revision을 남기고 `MEASUREMENT_READY`로 전환하며, parent activation으로 초기 기준점에 돌아가면 다시 측정이 잠깁니다.

Align 확정 뒤에도 모든 기능이 한꺼번에 열리지는 않습니다. 현재 활성 Align의 고유한 `READY + FRESH` 기록을 기준으로 `Cutline 3/3 → Outline 6/6 → Digital Rubbing 6/6` 순서로 다음 단계가 열리고, 완료 버튼은 초록색으로 바뀝니다. application command도 같은 gate를 강제하며, 각 Outline은 Cutline 3면을, 각 Digital Rubbing은 dependency-valid Outline 6면을 직접 참조합니다. 이 선행 record coverage가 없으면 READY 기록도 완료 증거로 세지 않습니다. 같은 방향을 여러 번 기록해도 한 면으로 계산합니다. Align을 바꾸면 기존 기록은 삭제하지 않고 stale로 제외하며, 이전 Align을 다시 활성화하거나 프로젝트를 재열면 문서의 record graph에서 진행도를 그대로 복원합니다.

처음 쓰는 사용자도 **5분 안에 첫 결과**를 얻는 것이 목표입니다.

---

## 이번 구현에서 강화된 핵심 기능

### 0. 원본 보존형 ArtifactDocument + 검증 Cutline/Outline/Digital Rubbing

- 원본 file SHA-256, decode geometry SHA-256, 확인된 단위·축, immutable Align revision을 분리해 저장
- 새 문서는 원본의 절대 경로를 canonical 문서에 넣지 않고 `external:<original_name>` locator만 저장함. 실제 OS 경로는 현재 session에만 유지하고, linked resource는 정규화된 상대 POSIX 논리 경로만 기록하므로 같은 source closure·recipe의 문서와 SVG/PNG hash가 drive/root 위치에 따라 달라지지 않음
- native `.amr` 저장은 `ArtifactDocument`와 함께 검증된 주 원본 및 파서가 실제로 읽은 MTL·텍스처·buffer bytes를 SHA-256 content-addressed blob으로 포함함. 외부 파일을 삭제하거나 프로젝트를 다른 컴퓨터로 옮겨도 `.amr` 하나에서 saved parser·단위·Align·geometry hash와 전체 dependency hash를 다시 검증해 열 수 있고, 열린 archive를 Save/Save As로 다시 저장할 수 있음
- `파일 → 중단된 프로젝트 저장 복구…`는 사용자가 고른 폴더에서 writer의 exact `.<destination>.XXXXXXXX.tmp` 이름을 가진 regular file만 후보로 제시함. 후보 inode·크기·수정시각을 고정해 별도 staging으로 복사하고 내장 원본·문서·Align을 production loader로 완전 물질화한 뒤, 존재하지 않는 새 `.amr`에 no-overwrite 게시함. 후보·기존 프로젝트는 성공해도 자동 삭제하거나 변경하지 않으며 현재 장면도 사용자의 별도 확인 없이 교체하지 않음
- 기존 manifest-only `.amr`와 self-contained `mesh-import-recipe 1.0`은 계속 읽음. 새 import에서 외부 resource를 읽지 않으면 v1 `deny_external`, 실제 상대 resource를 읽으면 `mesh-import-recipe 2.0`의 `closed_manifest`와 `relative-contained-v1` resolver로 확정함
- v2 resolver는 manifest에 선언된 exact logical path·SHA-256·크기의 byte stream만 재생함. HTTP/file URI, 절대·drive·UNC 경로, source root 탈출, symlink 탈출, 누락·추가·변경·미사용 dependency는 fail closed로 거부함
- Open → Align commit → Cutline record가 항상 source-space 원본에서 canonical millimeter로 다시 계산됨
- Top/Front/Right 단면을 명시적 right-handed plane frame으로 기록
- 화면용 단면 tape나 world XY 투영을 SVG 원본으로 사용하지 않음
- Cutline payload·recipe·QC를 RFC 8785 semantic hash와 함께 `.amr`에 보존
- Top/Bottom/Front/Back/Right/Left 6면 Outline을 전체 삼각형의 고정 mm 격자 투영 합집합으로 계산
- Outline의 오목부·구멍·분리 성분을 모두 보존하고 self-intersection·hole 소유권·component 중첩을 저장/재로딩 때 재검증
- Shapely 2.1.2 + GEOS 3.13.1, precision grid, grid collapse/merge receipt를 recipe·QC에 고정
- `READY + FRESH` 기록만 `*.amr-vector/`의 1:1 `artifact.svg` + provenance sidecar로 내보내며, 공개 provenance에 주 원본과 dependency의 논리 경로·SHA-256·크기를 포함함
- Digital Rubbing은 6면 canonical frame, 정수 pixels/mm·µm recipe, front-depth raster와 QC를 `raster.digital_rubbing.v1` record로 보존
- canonical GA8 PNG는 고정 chunk/DEFLATE bytes와 exact `pHYs`를 사용하며, `*.amr-rubbing/`에 provenance sidecar와 함께 저장
- 기와 기록면 전개는 명시적 canonical 장축(`X/Y/Z`), Top/Bottom 기록면, 원본 face-range selection을 recipe로 고정하고 sectionwise 알고리즘의 자동 fallback·1 µm 격자 collapse·orientation foldover·품질 기준 초과를 정식 record에서 거부
- 통과한 결과는 `surface.tile_unwrap.v1` receipt로 보존하며 canonical binary 전체 SHA-256, 원본 vertex/face correspondence, exact µm bounds, section fit와 distortion QC를 서로 대조
- `*.amr-unwrap/`은 canonical binary, 평면 OBJ, 실제 mm `width`/`height`/`viewBox`를 갖는 1:1 경계 SVG, 공개 provenance sidecar를 한 묶음으로 no-overwrite 게시
- vector/rubbing/tile-unwrap package는 원본 mesh와 GUI가 없어도 이동 후 별도 프로세스에서 offline 검증 가능
- Qt/OpenGL과 분리된 `ArtifactWorkbench`가 ticketed Open, 명시적 Align readiness, `state_version`/`authority_epoch` 기반 publication을 소유
- native DerivedRecord worker는 시작 session과 projection을, viewport의 cut-section/ROI/surface-selection worker는 worker identity·target mesh/TRS·render frame을 확인하여 늦은 결과가 현재 문서·overlay·다른 유물·새 worker를 덮지 못하게 함
- Cutline·Outline·Digital Rubbing·기와 전개 worker는 공통 취소 Event를 계산 내부의 deterministic chunk 경계까지 전달하며, 사용자는 진행 창에서 강제 스레드 종료 없이 취소를 요청할 수 있음
- DerivedRecord 추가는 같은 render projection의 문서 binding만 compare-and-swap하며 live mesh·VBO·카메라·선택·preview cache를 다시 만들지 않음
- 대좌표 장면은 CPU·문서의 절대 float64 world-mm 좌표를 유지하면서, 객체별 VBO origin을 float64에서 먼저 빼 relative `GL_FLOAT`로 업로드하고 live scene의 안정적인 render origin에 camera·model transform을 rebase함
- 두 origin은 viewport 전용 transient 상태이며 ArtifactDocument·record·QC·hash·export에 기록하지 않음. mesh·cutline·ROI·pick·gizmo 등 활성 world overlay를 render-relative로 제출하고 CPU face 계산은 absolute float64를 유지함
- depth pick·screen projection·Ctrl drag는 해당 depth buffer를 그린 modelview·projection·viewport·scene origin과 visibility·ROI·X-ray·object TRS/geometry revision을 하나의 read-only frame authority로 묶어 다른 시점의 상태가 섞이지 않게 함
- 앱 시작 전에 OpenGL 2.1 compatibility·24-bit depth surface 계약을 명시하고, paint 뒤 depth readback/pick에 필요한 widget FBO attachment는 `PartialUpdate`로 보존함. 별도 native-process smoke가 실제 QOpenGLWidget context/FBO/VBO/pixel/depth/pick 경로를 검증함
- SVG/PNG export worker는 보이지 않는 same-parent staging package를 완전 검증해 exact inode/fingerprint capability를 만들고, GUI callback이 현재 Workbench의 source·Align·exact `READY + FRESH` record를 다시 확인한 뒤 빠른 재확인·rename으로만 공개
- export 중 같은 Align에 무관한 record가 추가돼도 안전하게 게시하지만 Align/Open 완료로 권위가 바뀌면 destination을 만들지 않고 자신이 소유한 staging만 정리함
- scene publication의 rollback·scene 복원·finalize 자체가 불확실하면 fatal authority 상태로 전환해 검증된 Open 전까지 저장·실측·내보내기를 차단

Native 문서에서는 기존 screenshot/OpenCV/convex-hull 2D 도면과 임의 `SurfaceVisualizer`/flatten PNG·SVG를 측정 산출물로 내보내는 우회 경로를 차단합니다. 검증된 Cutline/Outline record는 `.amr-vector`, Digital Rubbing record는 `.amr-rubbing`, 엄격한 기와 전개 record는 `.amr-unwrap`으로 내보냅니다. 과거 세 OS 결과는 이식성의 역사적 증거로만 보존하고 현재 완료 판정은 Windows만 사용합니다. 패키지 gate는 상용 installer compiler 대신 검증형 portable ZIP을 만들고, 한글 경로에 안전하게 추출한 실행 파일을 네트워크 차단 상태에서 complete workflow와 native `qwindows` actual-frame까지 다시 실행하도록 구성합니다. CI 산출물은 업로드하지 않으며 공개 배포·서명을 뜻하지 않습니다.

### 1. 정식 기와 기록면 전개

- Align이 확정된 canonical millimeter geometry에서만 계산
- 자동 장축 추정 대신 기록자가 `X/Y/Z` 장축을 명시해 해석을 recipe에 남김
- 선택 기록면을 정렬·병합된 원본 face 범위와 selection SHA-256으로 보존
- sectionwise 계산이 cylinder/area 등으로 fallback하면 정식 결과로 위장하지 않고 실패
- 결과 좌표를 1 µm 정수 격자로 고정하고 모든 삼각형의 collapse·방향 뒤집힘과 mean/p95 distortion gate를 검사
- application shell의 `begin_tile_unwrap()`가 immutable work item, 취소, stale Align 차단, same-Align record publication을 기존 실측과 같은 방식으로 처리
- 메인 `4축 작업 흐름`의 전용 바로가기에서 native desktop 패널을 열고, 전체/현재 face 선택, canonical `X/Y/Z` 장축, Top/Bottom 기록면, section 수를 명시해 계산·취소·record 재선택·QC 미리보기·1:1 export까지 같은 Workbench 권위로 실행
- 재열기 뒤 선택한 `READY + FRESH` record는 저장 recipe로 전개 좌표를 다시 계산해 receipt와 대조한 뒤에만 미리보기와 export를 허용하며, 기존 자유 flatten UI의 결과는 계속 legacy 검토용으로 구분

### 2. 기와형 메쉬용 기본 추천 펼침

- 길이 방향이 뚜렷하고 곡면 단면이 반복되는 메쉬에서는 기본 추천을 `sectionwise flatten`으로 설정
- UI에서는 내부 알고리즘명이 아니라 **`기와 추천 펼침`** 으로 표기
- 추천 이유를 함께 표시
  - 예: `기와형/장축 반복 단면 패턴이 높아 기와 추천 펼침을 기본으로 권장`
- 사용자는 필요할 때만 아래 대안으로 전환 가능
  - `저왜곡 펼침`
  - `기록면 기반 펼침`
  - `곡면 추적 펼침`
  - `각도 보존 펼침`

### 3. 장축 기반 펼치기 + 수동 보정 흐름

- 장축 자동 추정
- 사용자의 수동 축 보정 반영
- 대표 단면/와통 반경 힌트 반영
- sectionwise 품질이 부족하면 fallback 경로 제공
  - `sectionwise → area → cylinder → arap`

### 4. Legacy rubbing-like 판독 시각화

예쁜 렌더보다 **문양/흔적 판독성**을 우선하는 기존 검토 경로입니다. 이 결과는 연구 검토 이미지이며, 위 `raster.digital_rubbing.v1`의 재현 가능한 1:1 PNG와 구분합니다.

- `normal` 기반 시각화
- `curvature` 기반 시각화
- `height/depth` 계열 시각화
- `contrast`, `strength` 조절 가능
- 논문/보고서용 PNG 저장 가능

### 5. 연구 산출물 중심 export

- `*.amr-vector/`: 검증 Cutline/Outline 1:1 SVG + provenance
- `*.amr-rubbing/`: 검증 Digital Rubbing 1:1 PNG + provenance
- `*.amr-unwrap/`: 검증 기와 기록면 canonical binary + 1:1 SVG + flat OBJ + provenance
- `flattened 좌표`
- `기록면 전개 SVG` (legacy 검토용)
- `rubbing PNG` (legacy 검토용)
- `기록면 검토 시트` (legacy 검토용)
- `6방향 도면 패키지` (legacy 검토용)

---

## 현재 flatten 모드 정리

| 사용자용 이름 | 내부 키 | 언제 추천하나 | 특징 |
|---|---|---|---|
| `기와 추천 펼침` | `section` | 기와형/장축 반복 단면이 뚜렷할 때 | 길이 방향 해석성과 단면 반복성을 우선 |
| `저왜곡 펼침` | `arap` | 일반 목적 기본 대안 | 형태 왜곡을 상대적으로 억제 |
| `기록면 기반 펼침` | `area` | 면적 안정성과 기록면 해석을 우선할 때 | 비교적 안정적인 전개 |
| `곡면 추적 펼침` | `cylinder` | 원통성/곡면 흐름이 분명할 때 | 축을 따라 펼침 |
| `각도 보존 펼침` | `lscm` | 각도 보존이 더 중요할 때 | conformal 계열 대안 |

---

## 제품 구조와 코어 구조

새 측정 경로는 GUI와 분리된 headless 신뢰 코어를 사용합니다.

### Application workflow shell

- [`src/application/artifact_workbench.py`](src/application/artifact_workbench.py): 한 artifact의 session/project path, 단일 pending Open, workflow readiness와 projection publication authority
- [`src/application/artifact_workflow_progress.py`](src/application/artifact_workflow_progress.py): 검증된 session의 `READY + FRESH` view coverage에서 Cutline 3면·Outline 6면·Digital Rubbing 6면 진행도와 순차 gate를 재구성
- [`src/application/artifact_measurements.py`](src/application/artifact_measurements.py): Cutline/Outline/Digital Rubbing/기와 전개의 immutable work item, Workbench 공유 예약·메모리 admission, cooperative cancellation, exact result capability와 same-Align rebase
- [`src/application/artifact_exports.py`](src/application/artifact_exports.py): vector/rubbing/tile-unwrap export의 exact capability, worker staging, 안전한 정리와 final-authority publication
- Open/new import/project reopen과 Align commit/parent activation은 ticket과 compare-and-swap 검증을 거쳐 `prepare → activate → finalize`되며, scene swap 실패는 이전 authority로 rollback
- Align commit/parent activation은 GUI에서 현재 session·scene binding·preview 값만 캡처합니다. 원본 geometry 재해시, candidate session 구성과 canonical materialization은 닫힘이 잠긴 worker에서 준비하고, 완료 시 같은 객체·mesh·binding·preview와 Workbench session/state/epoch/project path인지 다시 확인합니다. 하나라도 바뀌면 결과를 폐기하며, OpenGL context가 필요한 VBO 준비와 two-phase scene publication만 GUI thread에서 수행합니다.
- Cutline/Outline/Digital Rubbing/기와 전개는 application shell에서 파라미터와 가벼운 scene guard만 캡처하고 단일 worker에서 계산합니다. canonical scene materialization·vertex/face 일치 검사와 Digital Rubbing의 해상도별 전체 peak-memory 추정도 controller 실행 수명주기 안의 worker preflight에서 수행하므로 GUI event loop를 막지 않습니다. Rubbing 시작 시에는 원본 geometry·UV·texture 복사를 포함한 보수적 최소 예약을 먼저 잡고, 전체 추정이 예산을 넘으면 계산 전에 fail closed합니다. worker는 문서를 변경하지 않으며, 완료 결과는 예약된 record ID와 일회성 result capability를 검증한 뒤 현재의 같은 Align session에 rebase합니다. record append publication은 GPU 장면을 교체하지 않습니다. 기와의 ‘현재 선택 면’은 exact face-range recipe로 고정하며, 성공적으로 게시될 때 사용자가 선택을 바꾸지 않은 경우에만 소비합니다. 재개방한 기록은 READY + FRESH 목록에서 명시적으로 선택하고, 일시적인 Open/scene 충돌로 게시하지 못한 측정 결과는 새 ID를 만들지 않고 재시도합니다.
- vector/rubbing/tile-unwrap export는 비싼 생성·record recipe 재계산을 worker에서 staging까지만 수행합니다. 최종 no-replace rename은 현재 권위를 다시 검증한 뒤 실행하며, 목적지 경합에서는 기존 승자를 보존합니다. rename 뒤 directory `fsync`가 실패하거나 Windows처럼 지원 여부를 확인할 수 없으면 저장 완료와 crash durability 미확정을 구분해 경고합니다.
- native Save/Save As는 immutable session snapshot과 경량 scene guard만 GUI에서 캡처합니다. Git build metadata 조회, 원본 closure 재해시, ZIP64 작성·fsync, production reader 재개방, source/Align 재물질화는 잠긴 진행 대화상자의 worker에서 수행합니다. 완료 시 session·state version·authority epoch·기존 project path가 모두 그대로일 때만 결과 경로를 현재 프로젝트로 채택합니다. 중간에 권위가 바뀌면 파일은 유효한 과거 snapshot으로 명시하되 현재 문서가 저장됐다고 표시하지 않습니다.
- 중단 저장 복구는 자동 시작 스캔이나 임시본 정리를 하지 않습니다. 사용자가 범위를 폴더 하나로 지정하고 후보·새 목적지를 각각 확인한 뒤 잠긴 worker에서 descriptor copy, file `fsync`, embedded-session materialization과 atomic create-new publish를 수행합니다. 성공 뒤에도 복구본 열기는 별도 확인이며, 실패·목적지 경합·후보 identity 변경에서는 live scene과 모든 기존 경로를 유지합니다.
- 창 종료는 active authoritative 측정·내보내기의 record/publication 권위를 먼저 회수하고 강제 thread termination 없이 최대 30초 join을 기다립니다. native Align 준비나 project save처럼 내부 해시·materialization·파일 호출을 선점할 수 없는 task도 같은 bounded join을 통과해야 합니다. worker 종료와 export staging의 안전한 terminal 상태를 증명하지 못하면 창을 닫지 않으며, 검증된 join 뒤에만 task signal과 identity를 제거해 늦은 결과 게시를 차단합니다.

### Artifact trust core

- [`src/core/artifact_cancellation.py`](src/core/artifact_cancellation.py): GUI와 무관한 cooperative cancellation probe·고유 종료 신호
- [`src/core/artifact_document.py`](src/core/artifact_document.py): source·metadata·Align·DerivedRecord revision graph
- [`src/core/artifact_session.py`](src/core/artifact_session.py): 검증 source와 immutable document의 materialization 경계
- [`src/core/project_recovery.py`](src/core/project_recovery.py): 비정상 종료 save-temp의 bounded discovery, identity-pinned copy, 완전 offline materialization과 no-overwrite 복구 게시
- [`src/core/artifact_vector_extractor.py`](src/core/artifact_vector_extractor.py): canonical-mm Cutline
- [`src/core/artifact_outline_extractor.py`](src/core/artifact_outline_extractor.py): fixed-grid 6면 Outline
- [`src/core/artifact_rubbing_extractor.py`](src/core/artifact_rubbing_extractor.py): deterministic 6면 Digital Rubbing raster
- [`src/core/artifact_tile_unwrap_extractor.py`](src/core/artifact_tile_unwrap_extractor.py): 명시적 장축·face selection·1 µm 격자의 authoritative sectionwise 기와 전개
- [`src/core/artifact_tile_unwrap_record.py`](src/core/artifact_tile_unwrap_record.py): `surface.tile_unwrap.v1` receipt·recipe·QC 검증
- [`src/core/artifact_vector_export.py`](src/core/artifact_vector_export.py): `.amr-vector` 1:1 SVG package
- [`src/core/artifact_rubbing_export.py`](src/core/artifact_rubbing_export.py): `.amr-rubbing` canonical PNG package
- [`src/core/artifact_tile_unwrap_export.py`](src/core/artifact_tile_unwrap_export.py): `.amr-unwrap` binary/OBJ/SVG/provenance package와 offline verifier
- [`src/core/artifact_verification.py`](src/core/artifact_verification.py): `.amr`, `.amr-vector`, `.amr-rubbing`, `.amr-unwrap`를 자동 판별하고 설치본에서도 같은 검증 영수증을 만드는 통합 offline verifier
- [`src/core/project_file.py`](src/core/project_file.py): strict AMR v2 저장·로딩과 원자 교체

### Renderer precision boundary

- [`src/gui/render_coordinates.py`](src/gui/render_coordinates.py): absolute float64 연구 좌표를 변경하지 않고 객체별 VBO origin과 scene render origin으로 GPU 표시 좌표만 rebasing하는 Qt/OpenGL-free 수학 경계
- [`src/gui/opengl_context.py`](src/gui/opengl_context.py): 앱과 실제 driver smoke가 공유하는 OpenGL 2.1 compatibility·24-bit depth 요청 계약
- [`src/gui/opengl_driver_smoke.py`](src/gui/opengl_driver_smoke.py): native QPA의 실제 `Viewport3D` widget FBO에서 survey-scale VBO·pixel·depth·pick을 readback하고 source-state/environment JSON을 남기는 독립 프로세스 게이트. Windows에서는 768×768 비활성 도구창, 그 밖의 역사적 probe에서는 숨김 위젯을 사용하며 compositor의 최종 presentation은 별도 범위
- main mesh VBO, camera/model transform, native vector preview, ground/grid와 활성 cutline·ROI·pick·gizmo 등 world overlay를 render-relative 제출로 이식
- 한 frame의 modelview·projection·viewport·scene origin을 묶은 `RenderFrameSnapshot`과 그 프레임의 visibility·ROI·X-ray·object TRS/geometry depth signature로 depth unprojection, screen projection, ray, Ctrl drag의 좌표·픽셀 계약을 일치시킴
- 라쏘/가시 면 worker 결과는 시작 객체·mesh·TRS·depth authority와 완료 시점을 다시 비교하며, magnetic depth-edge cache는 잡힌 frame authority와 함께만 재사용. 객체 전환 시 미완성 gesture/polygon도 종료
- mocked OpenGL·pure float64 게이트와 별도로, 현재 Windows source와 frozen 실행 파일의 qwindows+llvmpipe 실제 OpenGL context에서 `>= 1e9 mm` 장면의 0.25 mm gap·0.125 mm 높이차를 원근/정사영 모두 검증함. macOS·Linux 결과는 과거 이식성 기록이며 첫 안정판 지원 판정에는 사용하지 않음

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

## 주요 산출물

### 기록면 전개

- OBJ/PLY/STL/GLTF 메쉬에서 기록면을 펼친 2D 결과 생성
- 기와형 메쉬는 기본적으로 `기와 추천 펼침` 우선
- 정식 기와 경로는 `surface.tile_unwrap.v1` record와 `.amr-unwrap` package로 exact selection·단위·왜곡·해시를 보존

### 디지털 탁본

- native 경로는 원본과 활성 Align에서 6면 raster를 다시 계산하고 recipe·QC·raster hash를 기록
- `artifact.png`는 1:1 planar sampling과 exact pixels-per-meter를 선언
- 포토리얼리스틱 렌더가 아니라 **재현 가능한 판독 보조 이미지**에 집중

### 실측/검토 패키지

- 전개 SVG
- rubbing PNG
- review sheet
- 6방향 도면 패키지

자유 flatten 전개·review sheet·기존 6방향 도면은 현재 **검토용 legacy 산출물**입니다. 1:1 측정 산출물로 검증되는 경로는 ArtifactDocument record에서 생성한 `.amr-vector` SVG, `.amr-rubbing` PNG, 그리고 엄격한 sectionwise 계약을 통과한 `.amr-unwrap` 기와 전개입니다.

---

## Quick Start

현재 Quick Start는 Windows source checkout 실행 절차입니다. 다운로드 가능한 서명 바이너리는 아직 제공하지 않습니다. Windows wheel hash lock, 실제 payload 기반 SPDX/NOTICE, exact Git commit의 corresponding-source ZIP, compiler 비종속 portable ZIP과 portable/source/evidence/GitHub Actions run·runner identity를 묶는 canonical unsigned provenance 생성·검증 코드는 저장소에 포함합니다. 첫 공개 안정판 전에는 프로젝트/GUI 라이선스 결정, provenance와 배포물의 Authenticode 또는 검토된 대체 서명·신뢰 정책, 대표 하드웨어·실물 pilot이 남아 있습니다.

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

## 개발 품질 게이트

개발·CI 환경은 런타임 의존성과 함께 `requirements-dev.txt`를 설치합니다.

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m ruff check .
python -c "import subprocess,sys; raise SystemExit(subprocess.call([sys.executable,'-m','pyright','--pythonpath',sys.executable,'-p','pyright-m0.json']))"
python -m pytest -q
```

일반 `QT_QPA_PLATFORM=offscreen` suite는 실제 QOpenGLWidget context를 생성하거나 검증하지 않습니다. CI는 빠른 Linux quality job과 제품 대상인 Windows workflow job으로 나뉩니다. Windows job은 offscreen workflow 뒤 `qwindows` + Qt의 bundled `opengl32sw.dll`을 강제하고 Qt·PyOpenGL dispatch를 같은 DLL로 묶어 실제 768×768 widget FBO/VBO/pixel/depth/pick smoke를 실행합니다. Windows frozen executable도 같은 report CLI를 통과합니다. 이는 Windows software OpenGL 증거이며 대표 하드웨어 GPU나 compositor 최종 presentation 인증은 아닙니다. 자세한 판정 범위는 [docs/QUALITY_GATES.md](docs/QUALITY_GATES.md)에 기록합니다.

`pyright-m0.json`은 현재 M0 신뢰 커널 범위의 차단 게이트입니다. 전체 트리 타입 검사는 아직 부채를 보고하는 단계이며, 통과를 뜻하지 않습니다. 재현 환경, 게이트 범위, GUI 스모크 테스트의 한계는 [docs/QUALITY_GATES.md](docs/QUALITY_GATES.md)에 기록합니다.

프로젝트 저장 형식, 원자적 저장 보장, 원본 SHA-256의 범위와 v1 migration 정책은 [docs/PROJECT_FORMAT.md](docs/PROJECT_FORMAT.md)를 참고하세요.

구조 개편 방향과 보존/교체 경계는 [docs/ARCHITECTURE_DECISION.md](docs/ARCHITECTURE_DECISION.md)에 기록합니다.

### 로컬 native smoke build

Python 3.12의 깨끗한 환경에서만 unsigned 로컬 앱을 만듭니다.

```bash
python -m pip install --require-hashes --only-binary=:all: -r requirements/windows-py312-x64-hashed.lock
python tools/build_native.py
```

이 명령은 기존 산출물을 기본적으로 덮어쓰지 않고, 빌드 뒤 실제 frozen executable의 offline self-test를 실행합니다. Windows build는 payload 전체 SHA-256 manifest, SPDX 2.3 SBOM·제3자 NOTICE와 정확한 Git commit/tree/blob을 담은 corresponding-source ZIP을 생성·재검증합니다. 현재 공개 바이너리는 만들지 않습니다. 패키지 CI는 Windows 하나만 차단 대상으로 사용하며 portable ZIP의 전체 entry를 검증한 뒤, 외부 unsigned provenance에 portable/source/evidence와 workflow·run·hosted runner identity를 결합합니다. 이어 한글 경로 원자적 추출, 같은 evidence·source archive·provenance, outbound 차단 complete workflow, native-QPA software OpenGL, 삭제를 검사합니다. 이 provenance는 내부 무결성 기록이지 서명된 출처 인증이 아닙니다. 라이선스 결정, 서명·신뢰 anchor와 대표 Windows GPU·실물 pilot은 여전히 남아 있습니다. 자세한 절차와 차단 게이트는 [docs/NATIVE_PACKAGING.md](docs/NATIVE_PACKAGING.md)를 참고하세요.

---

## CLI 예시

```bash
python main.py --help
python main.py mesh.obj
python main.py --open-project sample.amr
python main.py --verify-artifact sample.amr --report project-verification.json
python main.py --verify-artifact measured.amr-vector --against-project sample.amr --report vector-verification.json
python main.py --info mesh.obj
python main.py --flatten mesh.obj unwrap.png
python main.py --review mesh.obj review.png
python main.py --generate-synthetic sugkiwa_quarter 7 synthetic_tile.obj
python main.py --benchmark-synthetic ./benchmarks 1,2,3
python main.py --project mesh.obj planview.png
```

`--verify-artifact`는 네트워크, 계정, 라이선스 서버, GUI 없이 받은 자료를 검증한다. `.amr`는 내장 원본을 saved parser로 다시 열고 source/geometry/metadata/Align을 실제로 재물질화해야 성공한다. 세 export package는 exact member bytes, 1:1 scale, recipe, QC, provenance를 각각의 기존 엄격한 validator로 검사한다. `--against-project`를 함께 주면 그 `.amr`도 완전히 재개방한 뒤 export의 `READY + FRESH` record와 document SHA-256까지 일치해야 한다.

결과는 versioned closed JSON인 [`schemas/offline_verification_report-1.0.0.schema.json`](schemas/offline_verification_report-1.0.0.schema.json) 계약을 따른다. 성공 receipt에는 절대 입력 경로와 실행 시각을 기록하지 않으므로 검증에 성공한 같은 자료와 같은 authority mode는 같은 JSON 값을 만든다. `--report`는 기존 파일을 덮어쓰지 않는다. 검증 성공은 종료 코드 `0`, 자료 검증 실패는 `1`, 잘못된 옵션이나 report 저장 실패는 `2`다. `--report`를 생략하면 source/console 실행에서는 JSON 한 줄을 표준출력으로 보낸다.

`--flatten`, `--review`는 빠른 전체 경로용입니다.
상면/하면 기록면을 유도형으로 준비하려면 GUI 사용을 권장합니다.
합성 기와 생성·평가·review sheet 흐름은 [합성 벤치마크 가이드](docs/SYNTHETIC_BENCHMARKS.md)를 참고하세요.

---

## 지원 포맷

원본 hash와 재현 가능한 geometry identity를 함께 보존하는 native 작업 문서의 권위 원본은 다음 포맷을 지원합니다.

- `OBJ`와 상대 경로의 MTL·텍스처
- `PLY`와 상대 경로의 `TextureFile`
- `STL`
- `OFF`
- self-contained 또는 상대 로컬 buffer/image를 사용하는 `glTF (.gltf)`·`glTF Binary (.glb)`

외부 resource가 있으면 파서가 실제로 읽은 파일만 source manifest에 캡처하고 `.amr`에 함께 저장합니다. 원격 URI, 절대 경로, source root 밖으로 나가는 `..`·symlink는 허용하지 않습니다. 현재 한 manifest/source bundle은 최대 61개 엔트리, embedded source 전체는 최대 16 GiB입니다. geometry·UV·texture bytes의 보존과 오프라인 재현은 검증하지만, 여러 material/PBR 조합의 화면 렌더링 충실도는 아직 실제 스캔 pilot과 함께 확대해야 합니다.

---

## 구현 참고 레퍼런스

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

## 프로젝트 철학

이 프로젝트는 메쉬를 단순한 “렌더링 대상”으로 보지 않습니다.

- `정위치`는 도면 기준을 위한 단계
- `기록면 선택`은 연구 표면을 지정하는 단계
- `펼치기`는 해석 가능한 좌표계를 만드는 단계
- `rubbing`은 문양과 흔적을 읽기 위한 판독 보조 단계
- `export`는 연구 산출물을 만드는 단계

즉, 내부 계산은 메쉬를 사용하더라도,
사용자가 최종적으로 다루는 것은 **기록 가능한 2D 결과**여야 한다는 관점을 따릅니다.

---

## 현재 상태

현재 버전은 특히 아래에 집중하고 있습니다.

- 원본 hash·명시적 단위·immutable Align revision 기반 `ArtifactDocument`
- canonical-mm Cutline과 6면 fixed-grid Outline
- recipe·QC·receipt가 있는 6면 Digital Rubbing과 1:1 `.amr-rubbing` export
- 전체/선택 face·명시적 장축·Top/Bottom·section/QC를 한 화면에서 기록하고 재열어 검증하는 native 기와 전개와 1:1 `.amr-unwrap` export
- `.amr`, `.amr-vector`, `.amr-rubbing`, `.amr-unwrap`의 자동 판별·이동 가능한 offline 검증과 exact project 결합 receipt
- 주 원본과 실제 parser dependency bytes를 포함한 content-addressed `.amr` 저장, 원본 디렉터리 삭제 뒤 독립 프로세스 reopen·relocation·archive-to-archive 재저장
- self-contained v1 `deny_external`과 multi-file v2 `closed_manifest` mesh import recipe, parser-runtime subset identity, 경로 탈출·변조·미선언 resource deny gate와 동일 receipt의 독립 프로세스 재실행
- 기와형 메쉬 기본 추천 펼침과 synthetic benchmark는 legacy 전문 기능으로 유지
- Open/Align authority와 two-phase scene publication을 Qt/OpenGL-free `ArtifactWorkbench`로 이식
- Cutline/Outline/Digital Rubbing command와 worker 수명주기를 Qt/OpenGL-free application shell로 이식
- DerivedRecord의 VBO-free binding rebind와 SVG/PNG worker staging → final-authority publication 이식
- dependency-valid `READY + FRESH` record graph와 application command에서 Cutline 3/3 → Outline 6/6 → Digital Rubbing 6/6 순차 gate·초록 완료 표시·재열기/Align 복원 구현
- packaged self-test가 실제 application authority를 통해 Open → explicit Align → Cutline 3/3 → Outline 6/6 → Digital Rubbing 6/6을 수행하고, 외부 PLY 삭제 뒤 embedded `.amr`를 재열어 이동된 1:1 SVG/PNG package의 원본 SHA-256·recipe·QC와 exact-project 결합을 통합 verifier로 offline 재검증
- `--opengl-driver-smoke-report`가 source와 frozen Windows 실행 파일에서 native `qwindows` context를 열고 Qt·PyOpenGL을 bundled `opengl32sw.dll` 하나에 결합한 뒤, 768×768 FBO에서 `>= 1e9 mm` 장면의 relative VBO, color/depth readback, 0.125 mm depth pick을 원근·정사영으로 검증
- Windows x64/CPython 3.12 build wheel 17개를 exact SHA-256으로 잠그고 sdist를 거부하며, frozen/portable payload의 모든 파일 hash와 runtime 10개의 SPDX 2.3 SBOM·라이선스 원문 NOTICE를 실행 파일 self-test에서 재검증
- live worktree가 아니라 exact Git commit의 object database에서 100644/100755 regular blob 전체를 읽어 결정적 corresponding-source ZIP을 만들고, commit/tree/blob ID·SHA-256·GPL-2.0-only LICENSE·portable path를 frozen/portable 14번째 offline self-test에서 재검증
- portable ZIP·manifest, exact-source ZIP·sidecar, release-evidence index와 GitHub repository/workflow SHA/run attempt/Windows X64 hosted-runner 변수를 외부 canonical provenance 하나에 결합하고, 실제 payload 전체와 한글 추출본을 network 없이 재검증. `authentication=none`을 closed contract로 강제해 서명된 attestation으로 오인하지 않음
- Cutline 면·경로, Outline fixed-grid/union/topology, Digital Rubbing raster/relief 내부의 안전 경계까지 사용자 cooperative cancellation 연결
- 대좌표 render-origin 이식: relative VBO·camera/model rebasing·world overlay 제출과 frame-bound depth picking/drag 계약 구현
- 버튼 아이콘은 이모지·운영체제 폰트 대신 직접 그린 16×16 픽셀 그리드와 정수배 고해상도 변형을 사용해 플랫폼별 모양 차이를 제거
- 2026-07-14 통합 verifier 기능 기준 commit `4a21666e7f7b`: [source CI run 29280873586](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29280873586)에서 full pytest `763 passed, 128 subtests`, Ruff, M0 Pyright `0 errors`, Windows workflow `661 passed, 5 skipped, 118 subtests`와 qwindows+llvmpipe actual-frame `66/66` 통과
- 같은 기능 commit의 [portable package run 29280874076](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29280874076)에서 frozen·한글 경로 portable 실행 파일의 14-check offline self-test, outbound deny 상태의 public verification receipt, qwindows+llvmpipe actual-frame `66/66`, 추출본 삭제·방화벽 규칙 정리까지 통과
- 2026-07-14 중단 저장 복구 기준 commit `546d106c6ccf`: [source CI run 29282751462](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29282751462)에서 full pytest `777 passed, 128 subtests`, Ruff, M0 Pyright `0 errors`, Windows workflow `675 passed, 5 skipped, 118 subtests`와 qwindows+llvmpipe actual-frame `66/66` 통과. Windows `DirEntry`의 placeholder file identity까지 회귀 테스트로 고정함
- 같은 복구 commit의 [portable package run 29282751606](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29282751606)에서 frozen·한글 경로 portable 실행 파일이 `recovery=verified-create-new`를 포함한 14-check offline self-test, outbound deny, public verification receipt, qwindows+llvmpipe actual-frame `66/66`, archive/provenance 재검증과 추출본·방화벽 규칙 정리를 통과
- installer compiler 없이 표준 라이브러리만으로 deterministic portable ZIP과 canonical sidecar를 만들고, 경로 탈출·Windows 예약명·대소문자 충돌·symlink·변조를 fail-closed 검증한 뒤 기존 destination을 덮어쓰지 않는 원자적 추출 구현
- 다음 단계: 라이선스 결정, unsigned provenance와 portable/source 배포물을 인증할 서명·신뢰 anchor 정책, 대표 Windows GPU·대용량 실제 유물·저메모리·완전 격리 offline pilot. macOS·Linux 배포 확대는 첫 안정판 이후 별도 범위

---

## License

GNU General Public License v2.0 (GPLv2)

현재 문구는 `or later`를 포함하지 않습니다. PyQt6를 포함한 공개 바이너리 배포는 [native packaging license gate](docs/NATIVE_PACKAGING.md#라이선스)가 해결될 때까지 보류합니다.

## Citation

이 저장소가 연구, 수업, 현장 업무에 도움이 되었다면 GitHub의 **Cite this repository** 버튼으로 인용해 주세요.

[![Cite this repository](https://img.shields.io/badge/Cite_this-repository-2ea44f?logo=github)](https://github.com/lzpxilfe/ArchMeshRubbing)
[![Star this repository](https://img.shields.io/github/stars/lzpxilfe/ArchMeshRubbing?style=social)](https://github.com/lzpxilfe/ArchMeshRubbing)

인용 메타데이터는 [CITATION.cff](CITATION.cff)에 보관합니다.
