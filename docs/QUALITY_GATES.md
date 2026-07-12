# Quality Gates

이 문서는 ArchMeshRubbing의 현재 개발 기준선과 CI가 실제로 보장하는 범위를 기록한다. 장기적으로는 전체 코드베이스에 대한 타입 게이트와 Windows·macOS·Linux 전체 플랫폼 테스트를 목표로 하지만, 현재 게이트를 그 수준으로 과장하지 않는다.

## 재현 환경

CI의 기준 Python은 3.12이다. `requirements.txt`는 `requirements/runtime-py312.lock`을 포함하여 source와 frozen build가 같은 exact runtime resolution을 사용한다. 빌드 toolchain은 `requirements/build-py312.lock`, 검증 도구는 `requirements-dev.txt`에 고정한다. 권위 Outline overlay 조합은 `shapely==2.1.2`와 그 wheel의 GEOS `3.13.1`로 recipe와 runtime gate에 고정한다. lock은 아직 OS별 wheel hash lock은 아니다.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

## 차단 게이트

Pull request에서 다음 세 검사가 모두 통과해야 한다.

```bash
python -m ruff check .
python -c "import subprocess,sys; raise SystemExit(subprocess.call([sys.executable,'-m','pyright','--pythonpath',sys.executable,'-p','pyright-m0.json']))"
python -m pytest -q
```

- Ruff는 전체 트리를 검사한다.
- pytest는 pytest 함수와 `unittest.TestCase`를 모두 수집한다. 별도의 `unittest discover`는 하위 호환성 확인용이며 CI의 권위 수집기가 아니다.
- `pyright-m0.json`은 persistence·source·unit·matrix 경계에 더해 M0-6의 `artifact_document`, `geometry_identity`, `artifact_scene_adapter`, `artifact_session`, Qt/OpenGL-free `artifact_workbench`·`artifact_measurements`·`artifact_exports`, known-record registry, RFC 8785 canonical JSON, vector record/export, Cutline, fixed-grid Outline/topology, Digital Rubbing record/extractor, canonical GA8 PNG와 offline rubbing export 및 해당 테스트를 포함하는 M0 신뢰 커널 범위다. 독립 프로세스 왕복·offline vector/rubbing package 테스트도 목록에 포함한다. wrapper 명령은 활성 Python interpreter를 Pyright에 명시하므로 Windows·macOS·Linux에서 같은 방식으로 dependency를 해석한다.

M0-6 native artifact 신뢰 경계를 빠르게 재검증할 때는 다음 focused suite를 사용한다. 이 명령은 full pytest를 대체하지 않는다.

```bash
python -m pytest -q \
  tests/test_project_file.py \
  tests/test_build_info.py \
  tests/test_build_manifest.py \
  tests/test_build_native.py \
  tests/test_source_identity.py \
  tests/test_alignment_utils.py \
  tests/test_artifact_document.py \
  tests/test_geometry_identity.py \
  tests/test_artifact_scene_adapter.py \
  tests/test_artifact_session.py \
  tests/test_artifact_workbench.py \
  tests/test_artifact_exports.py \
  tests/test_artifact_measurements.py \
  tests/test_artifact_new_process_roundtrip.py \
  tests/test_canonical_json.py \
  tests/test_artifact_vector_record.py \
  tests/test_artifact_vector_export.py \
  tests/test_artifact_vector_extractor.py \
  tests/test_artifact_outline_extractor.py \
  tests/test_artifact_outline_topology.py \
  tests/test_artifact_rubbing_extractor.py \
  tests/test_artifact_rubbing_export.py \
  tests/test_canonical_png.py \
  tests/test_vector_schemas.py \
  tests/test_rubbing_schemas.py \
  tests/test_rotation_convention.py \
  tests/test_app_gui_launcher.py \
  tests/test_gui_smoke.py
```

이 suite의 핵심 증거는 다음과 같다.

- fixed OpenGL 순서 `T @ Rx @ Ry @ Rz @ S`의 hard-coded golden point
- point·bounds·plane·inverse 변환의 동일 행렬 규약
- Align의 scale·shear·reflection·perspective·non-finite 거부
- canonical millimeter metadata, immutable append/activate, deterministic serialization
- Align/metadata 전환에 따른 record freshness와 background operation context 고정
- versioned canonical JSON golden fixture와 Draft 2020-12 `ArtifactDocument 1.0` schema 검증
- raw source `(identity_scope, SHA-256, size)`와 saved parser format 재검증
- versioned geometry framing, signed-zero normalization, order/winding-sensitive geometry SHA-256
- source를 mutate·recenter하지 않는 deterministic world-mm projection과 stale snapshot 거부
- native 한-artifact session의 preview/Align commit/parent activation, source-of-truth scene binding과 unported mutation/save fail-closed
- `initial_identity` baseline의 `ALIGN_REQUIRED`, 변화량이 0인 첫 explicit child Align의 `MEASUREMENT_READY`, parent activation 시 downstream 재차단
- 단일 pending Open ticket, cancelled/superseded worker callback 거부, replacement 실패 시 기존 session·project path 보존
- `state_version`/`authority_epoch` compare-and-swap, transition 종류별 허용 변경과 expected record ID 집합 검증, 동시 candidate의 first-wins
- same-render DerivedRecord append의 `RecordBindingTransition`, exact snapshot capability 검증, live mesh/VBO/선택/cache 보존과 binding CAS rollback
- tentative authority를 observer에 노출하지 않는 prepare/activate/finalize, 정상 rollback과 rollback·scene 복원·finalize 불확실 시 fatal save/measure/export 차단
- scene 교체 뒤 cut-section/ROI worker identity·projection generation·selected object fencing과 오래된 finished callback의 새 worker 보호
- 서로 다른 PID에서 source relocation·saved-parser reopen 후 source/geometry hash, Align matrix와 world vertex가 같은 durable artifact round-trip
- RFC 8785 cross-language number golden과 vector payload/recipe semantic SHA-256
- canonical-mm Cutline의 exact box·multi-component·Front/Right/oblique frame, face order/winding, ambiguity fail-closed
- canonical-mm 6-view Outline의 fixed-grid projected-triangle union, concavity·hole·island 보존, face order/winding/duplicate 안정성
- translated integer lattice의 `1e9 mm` survey offset, grid collapse/merge receipt, chunked balanced union과 resource-limit fail-closed
- Outline ring simple/area, hole ownership/contact, hole/exterior overlap·nesting, production recipe/frame/grid/ID 재검증
- native Outline widget/record/record-derived overlay와 screenshot/OpenCV legacy export 우회 차단
- `READY + FRESH + verified payload`만 허용하는 1:1 SVG, full sidecar-claim binding, confirmed unit/active Align/dependency closure
- SVG/sidecar tamper, XML active content/DTD, duplicate/pathological JSON, privacy allowlist, size cap, concurrent no-replace publish
- package relocation 후 원본 문서·mesh·GUI 없이 별도 PID에서 수행하는 offline 검증
- versioned vector payload/export Draft 2020-12 schemas
- six-view canonical-mm front-depth raster, fixed integer µm quantization/local-mean tone mapping, coverage alpha와 multi-layer QC
- face order·winding·duplicate, hole, large survey offset, resource limit 및 Align 전환 후 late Digital Rubbing 결과 거부
- canonical GA8 PNG의 고정 chunk/DEFLATE bytes, exact `pHYs`, RFC 8785 iTXt metadata, pixel/CRC/chunk/scale tamper 거부
- `READY + FRESH + recomputed raster`만 허용하는 `.amr-rubbing` 1:1 PNG package, relocation 후 독립 PID offline 검증, no-replace publish와 privacy allowlist
- vector/rubbing worker의 hidden same-parent staging·전체 검증·prepared inode/fingerprint capability, 빠른 final Workbench record-authority fence, same-Align append 허용, Align/Open stale 정리와 pending Open GUI 취소 정책
- 고정 길이 staging UUID 충돌·quarantine/foreign inode 보존, exact result/prepared capability 위조·사전 목적지 이동·destination race 차단, post-rename 실제/미지원 directory-fsync `committed` 내구성 경고
- Qt-free Cutline/Outline/Digital Rubbing work item, exact result capability, same-Align 병렬 rebase, Workbench 공유 record reservation, Align/Open stale·취소·rollback 방어와 pending Open 게시 재시도
- Digital Rubbing 누적 peak-memory admission, UV/texture materialize 복사비의 무복사 사전 차단, controller별 한도 우회 방어, 실행 exactly-once, 취소 worker 종료 전 slot 보존
- 재개방 프로젝트의 READY + FRESH vector/rubbing 명시 선택, background recipe 재계산과 완료 시 document/Align/record 재검증, active raster 예산 중첩 차단, 일시 게시 실패 재시도 queue와 보류 중 저장 차단
- versioned Digital Rubbing receipt/export Draft 2020-12 schemas

## 현재 기준선

2026-07-12 M0-6 로컬 검증 결과:

| 검사 | 결과 |
|---|---:|
| Python 3.12.13 `python -m pytest -q` | 463 passed, 113 subtests passed |
| 3-OS persistence-smoke 명시 suite의 로컬 실행 | 392 passed, 113 subtests passed |
| `python -m ruff check .` | passed |
| M0 Pyright wrapper command | 0 errors |
| ArtifactDocument + vector/rubbing payload/export Draft 2020-12 schemas + golden | passed |
| Python 3.12.13 macOS arm64 frozen self-test | 10/10 passed (unsigned, `source_tree=dirty`, `native-self-test-local-smoke-8bd26e3a4733-darwin-3.json`) |

## 아직 차단하지 않는 검사

전체 트리 Pyright는 아직 통과하지 않는다. CI에서는 이 결과를 `continue-on-error`로 보고하여 부채가 보이게 하되, M0 범위를 넘는 기존 오류 때문에 모든 변경을 막지는 않는다. 신뢰 커널 전환이 진행될 때마다 차단 범위를 넓힌다. 독립 프로세스 테스트의 worker program은 Python 문자열이므로 Pyright가 문자열 내부를 분석하지는 않지만, 차단 pytest가 두 문자열을 각각 새 interpreter에서 실제 실행한다.

Windows·macOS·Linux persistence matrix에서는 프로젝트 저장, source/geometry identity, ArtifactDocument·scene adapter·session·application workbench, ticketed Open과 explicit Align gate, RFC 8785/vector record/export/Cutline/Outline/topology/schema, Digital Rubbing record/extractor/canonical PNG/export/schema, 독립 프로세스 source 및 relocated vector/rubbing-package 왕복, matrix golden, GUI 런처, MainWindow 생성, native source-of-truth binding, native Cutline/Outline/Rubbing command, session/version/epoch 및 projection-generation late-result 방어, legacy export 우회와 unported operation/save fail-closed, source mismatch ordering, scene-swap rollback과 fatal authority fallback 스모크를 실행하도록 설정한다. Linux quality job은 별도로 전체 테스트를 실행한다. CI는 job 환경에서, GUI 스모크 테스트는 모듈 로딩 시 `QT_QPA_PLATFORM=offscreen`을 설정한다. 이 스모크는 CPU/document/scene transaction과 widget wiring을 검증하지만 GPU/OpenGL 프레임의 실제 렌더링이나 시각적 정확성은 보장하지 않는다. 원격 matrix가 실제로 통과하기 전에는 3개 OS 완료로 표현하지 않는다.

별도 `package-smoke.yml`은 세 OS에서 exact Python 3.12 build lock, immutable build manifest, PyInstaller spec과 frozen executable의 file-report self-test를 실행하도록 구성한다. 이 검사는 실제 `MainWindow`/QOpenGLWidget/OpenGL import, 6개 mesh parser, PNG codec과 canonical document/vector/rubbing을 포함하지만 offscreen 실제 GL context/render는 보장하지 않는다. 라이선스 게이트가 해결되기 전에는 artifact upload와 release 단계를 두지 않는다. 로컬 macOS arm64 결과만 확인됐으며 원격 3-OS 결과는 아직 없다.

AMR v2 `payload_type="artifact_document"` 1.0의 strict 저장·production-loader staged reopen·checksum·원자 교체와 독립 프로세스 source rebind/materialization은 현재 차단 게이트다. `tests/test_artifact_new_process_roundtrip.py`는 프로세스 A와 B의 PID가 다름을 확인하고, PLY를 다른 경로와 `.raw-scan` suffix로 옮긴 뒤 저장된 parser/unit으로 다시 decode한다. 새로 계산한 source SHA-256·크기와 geometry SHA-256, active Align ID·matrix, parser/unit, world vertices가 같아야 통과한다.

Native GUI의 한-artifact Open/Align commit/save/load는 `ArtifactWorkbench.snapshot.session.document`를 source of truth로 사용하며 MainWindow의 session field는 이행 중 compatibility mirror다. Open과 Align은 ticket/CAS/two-phase publication을 사용한다. Cutline/Outline/Digital Rubbing은 Qt-free `ArtifactMeasurementController`가 recipe/context와 record ID를 Workbench 단위로 예약하고 worker computation만 받은 뒤 current same-Align session에 rebase한다. append-only record publication은 live SceneObject의 document binding만 CAS하고 mesh/VBO를 재생성하지 않는다. `ArtifactExportController`는 vector/rubbing package의 생성·전체 검증·prepared capability 발급까지 worker에서 수행하고 GUI dispatcher의 final Workbench fence에서는 빠른 identity/fingerprint 확인과 rename만 실행한다. command handler는 단일 `TaskThread`를 사용하며 늦은 finished signal이 새 worker/dialog를 지우지 못한다. 재개방 기록은 자동 최신 fallback 없이 명시적으로 선택하고 request token으로 늦은 preview가 최신 선택을 지우지 못하게 한다. rollback 가능한 측정 게시 실패는 exact result를 재시도 queue에 보존한다. save 전 projection snapshot과 geometry를 재검증하며 active/보류 실측과 아직 `DerivedRecord`로 이식되지 않은 선택·기록면·평가 결과는 누락 저장하지 않고 fail closed한다. authority rollback·scene 복원·finalize가 불확실한 fatal 상태에서는 검증된 Open 전까지 저장·실측·내보내기를 모두 막고 task-local 오류가 재열기 배너를 덮지 못한다. 별도의 legacy destructive bake 이후 Save 성공은 native Align 복원 증거가 아니다. legacy runtime은 그런 vertex mutation을 `_amr_has_unpersisted_bake`로 표시하고 snapshot 저장을 차단한다.

### 대좌표 GPU 정밀도: 후속 비차단 게이트

현재 M0-3 게이트는 absolute float64 world-mm CPU 좌표를 검증한다. viewport의 float32 VBO가 `>= 1e9 mm` survey offset에서 mm-scale feature를 보존하는지는 아직 차단하지 않는다. 후속 render-origin 게이트는 CPU에서 transient origin을 뺀 relative float32 upload, picking 시 absolute 좌표 복원, document/geometry hash 불변, 동일 world export를 함께 검사해야 한다. render origin은 metadata·Align·record·QC·hash·export 권위 값에 저장해서는 안 된다.

## 게이트 변경 원칙

- 검사를 삭제하거나 `continue-on-error`로 바꾸는 것은 별도 근거와 리뷰가 필요하다.
- fallback, sampling, 단위 추정, 원본 불일치를 성공으로 숨기는 테스트를 추가하지 않는다.
- 플랫폼 매트릭스가 원격 CI에서 실제로 통과하기 전에는 “3개 OS 검증 완료”라고 표현하지 않는다.
