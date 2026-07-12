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

Pull request에서 다음 code-quality 검사와 별도의 actual-OpenGL job이 모두 통과해야 한다.

```bash
python -m ruff check .
python -c "import subprocess,sys; raise SystemExit(subprocess.call([sys.executable,'-m','pyright','--pythonpath',sys.executable,'-p','pyright-m0.json']))"
python -m pytest -q
```

- Ruff는 전체 트리를 검사한다.
- pytest는 pytest 함수와 `unittest.TestCase`를 모두 수집한다. 별도의 `unittest discover`는 하위 호환성 확인용이며 CI의 권위 수집기가 아니다.
- `pyright-m0.json`은 persistence·source·unit·matrix 경계에 더해 M0-6의 `artifact_document`, `geometry_identity`, `artifact_scene_adapter`, `artifact_session`, core cooperative cancellation, Qt/OpenGL-free `artifact_workbench`·`artifact_workflow_progress`·`artifact_measurements`·`artifact_exports`, `src/gui/opengl_context.py`의 명시적 surface 계약, actual-driver CLI와 actual context를 열지 않는 helper/support tests, Qt/OpenGL-free render-coordinate algebra인 `src/gui/render_coordinates.py`, known-record registry, RFC 8785 canonical JSON, vector record/export, Cutline, fixed-grid Outline/topology, Digital Rubbing record/extractor, canonical GA8 PNG와 offline rubbing export 및 해당 테스트를 포함하는 M0 신뢰 커널 범위다. 독립 프로세스 왕복·offline vector/rubbing package 테스트도 목록에 포함한다. wrapper 명령은 활성 Python interpreter를 Pyright에 명시하므로 Windows·macOS·Linux에서 같은 방식으로 dependency를 해석한다.

`opengl-driver-smoke` job은 일반 pytest와 분리한다. Ubuntu 24.04에서 24-bit Xvfb 화면, native `xcb`, Mesa llvmpipe를 사용하고 `continue-on-error`나 context 실패 skip 없이 다음 명령에 해당하는 검사를 실행한다.

```bash
xvfb-run -a \
  -s "-screen 0 1280x1024x24 +extension GLX +render -noreset" \
  python -m src.gui.opengl_driver_smoke \
  --qt-platform xcb \
  --report build/opengl-driver-smoke.json
```

이 job은 실제 OpenGL context/FBO/VBO/pixel/depth readback을 사용하지만 Mesa software rasterizer 검증이다. 대표 하드웨어 GPU 인증이라고 표현하지 않는다. code commit `166103dcf0ea`의 [GitHub Actions run 29182584810](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29182584810)에서 xcb + llvmpipe actual-GL report의 61개 조건이 통과했다.

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
  tests/test_artifact_workflow_progress.py \
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
  tests/test_render_coordinates.py \
  tests/test_viewport_render_origin.py \
  tests/test_opengl_driver_smoke.py \
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
- `>= 1e9 mm` offset의 0.125/1 mm feature에서 float64 origin subtraction 뒤 relative float32 VBO encoding, pivot-aware model rebasing과 relative camera algebra
- viewport 전용 VBO/scene origin이 document canonical bytes·SHA-256·record-binding publication에 들어가지 않고 live mesh·VBO와 함께 보존되는 경계
- native vector preview와 cutline·ROI·pick·gizmo 등 활성 world overlay의 render-relative GL 제출, absolute float64 CPU face 계산의 분리
- modelview·projection·viewport·scene origin과 visibility·ROI·X-ray·all-object TRS/geometry revision을 하나의 read-only frame authority로 게시한 project/unproject/ray, depth pick·Ctrl drag 수명주기와 repaint 전 mutable-state race/stale frame 거부
- native 한-artifact session의 preview/Align commit/parent activation, source-of-truth scene binding과 unported mutation/save fail-closed
- `initial_identity` baseline의 `ALIGN_REQUIRED`, 변화량이 0인 첫 explicit child Align의 `MEASUREMENT_READY`, parent activation 시 downstream 재차단
- 검증된 `ArtifactSession`의 unique `READY + FRESH` view-set에서 Cutline 3/3 → Outline 6/6 → Digital Rubbing 6/6 순차 gate·초록 완료 상태를 재구성하고, DRAFT/FAILED/blocked/unknown/noncanonical 제외, malformed known record의 session 경계 거부와 reopen·Align stale/restore를 검증
- 단일 pending Open ticket, cancelled/superseded worker callback 거부, replacement 실패 시 기존 session·project path 보존
- `state_version`/`authority_epoch` compare-and-swap, transition 종류별 허용 변경과 expected record ID 집합 검증, 동시 candidate의 first-wins
- same-render DerivedRecord append의 `RecordBindingTransition`, exact snapshot capability 검증, live mesh/VBO/선택/cache 보존과 binding CAS rollback
- tentative authority를 observer에 노출하지 않는 prepare/activate/finalize, 정상 rollback과 rollback·scene 복원·finalize 불확실 시 fatal save/measure/export 차단
- scene 교체 뒤 cut-section/ROI worker identity·projection generation·selected object fencing, surface/visible-face worker의 target mesh·TRS·render-frame fencing과 오래된 finished callback의 다른 유물·새 worker 보호
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
- core 고유 `RuntimeError` 취소 신호, 세 extractor 내부의 bounded polling과 대형 NumPy 단계·최종 결과 fence, false-probe payload/raster/QC 동일성, controller의 `FAILED > STALE > CANCELLED` 경합 우선순위·`CANCELLING → CANCELLED` slot 보존, GUI one-shot 취소 요청·대기 창 수명·무경고 종료
- Digital Rubbing 누적 peak-memory admission, UV/texture materialize 복사비의 무복사 사전 차단, controller별 한도 우회 방어, 실행 exactly-once, 취소 worker 종료 전 slot 보존
- 재개방 프로젝트의 READY + FRESH vector/rubbing 명시 선택, background recipe 재계산과 완료 시 document/Align/record 재검증, active raster 예산 중첩 차단, 일시 게시 실패 재시도 queue와 보류 중 저장 차단
- versioned Digital Rubbing receipt/export Draft 2020-12 schemas

## 현재 기준선

2026-07-12 M0-6 검증 결과:

| 검사 | 결과 |
|---|---:|
| GitHub Actions Python 3.12.13 quality `python -m pytest -q` | 572 passed, 117 subtests passed |
| macOS / Ubuntu persistence smoke | 각각 501 passed, 117 subtests passed |
| Windows persistence smoke | 498 passed, 3 platform-specific skips, 117 subtests passed |
| GitHub Actions `python -m ruff check .` | passed |
| GitHub Actions M0 Pyright wrapper command | 0 errors |
| Ubuntu 24.04 xcb + Mesa llvmpipe actual OpenGL | 61/61 passed at clean source commit `166103dcf0ea`, run `29182584810` |
| Python 3.12.13 macOS arm64 actual OpenGL driver smoke | 61/61 passed at clean code commit `f25b424d6936`, Apple M4, perspective + top orthographic |
| ArtifactDocument + vector/rubbing payload/export Draft 2020-12 schemas + golden | passed |
| Python 3.12.13 macOS arm64 frozen self-test | 10/10 passed at code commit `898a8bfc144f` (unsigned, `source_tree=clean`, `native-self-test-local-smoke-898a8bfc144f-darwin.json`) |
| GitHub Actions 3-OS frozen build + executable self-test | Ubuntu, Windows, macOS 모두 passed at commit `e4bf6dcac4b1`, run `29213279508` |

## 아직 차단하지 않는 검사

전체 트리 Pyright는 아직 통과하지 않는다. CI에서는 이 결과를 `continue-on-error`로 보고하여 부채가 보이게 하되, M0 범위를 넘는 기존 오류 때문에 모든 변경을 막지는 않는다. 신뢰 커널 전환이 진행될 때마다 차단 범위를 넓힌다. 독립 프로세스 테스트의 worker program은 Python 문자열이므로 Pyright가 문자열 내부를 분석하지는 않지만, 차단 pytest가 두 문자열을 각각 새 interpreter에서 실제 실행한다.

Windows·macOS·Linux persistence matrix에서는 프로젝트 저장, source/geometry identity, ArtifactDocument·scene adapter·session·application workbench와 record-derived workflow progress, ticketed Open과 explicit Align gate, RFC 8785/vector record/export/Cutline/Outline/topology/schema, Digital Rubbing record/extractor/canonical PNG/export/schema, 독립 프로세스 source 및 relocated vector/rubbing-package 왕복, render-coordinate algebra·relative VBO/native preview smoke, matrix golden, GUI 런처, MainWindow 생성, native source-of-truth binding, native Cutline/Outline/Rubbing command, 3/6/6 순차 gate와 reopen·Align 진행도 복원, session/version/epoch 및 projection-generation late-result 방어, legacy export 우회와 unported operation/save fail-closed, source mismatch ordering, scene-swap rollback과 fatal authority fallback 스모크를 실행한다. Linux quality job은 별도로 전체 테스트를 실행한다. commit `166103dcf0ea`의 run `29182584810`에서 세 persistence job과 quality job이 모두 통과했다. 이 matrix의 GUI 스모크는 `QT_QPA_PLATFORM=offscreen`을 사용하므로 CPU/document/scene transaction과 widget wiring만 검증하고 실제 OpenGL frame을 증명하지 않는다. 실제 source viewport frame은 별도 Linux Xvfb+xcb+llvmpipe job이 담당한다.

별도 `package-smoke.yml`은 `main` push, pull request, 수동 실행에서 세 OS의 exact Python 3.12 build lock, immutable build manifest, PyInstaller spec과 frozen executable의 file-report self-test를 실행하도록 구성한다. `main` push는 패키지 입력 파일이 바뀐 경우에만 실행한다. 이 검사는 실제 `MainWindow`/QOpenGLWidget/OpenGL import, 6개 mesh parser, PNG codec과 canonical document/vector/rubbing을 포함하지만 offscreen 실제 GL context/render는 보장하지 않는다. 라이선스 게이트가 해결되기 전에는 artifact upload와 release 단계를 두지 않는다. commit `e4bf6dcac4b1`의 [run `29213279508`](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29213279508)에서 Ubuntu, Windows, macOS frozen build와 각 OS 실행 파일의 self-test/report 검증이 모두 첫 시도에 통과했다. 같은 commit의 [CI run `29213279510`](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29213279510)도 quality, 3-OS persistence와 Linux llvmpipe actual-GL을 모두 통과했다. 이 결과는 installer, 서명/notarization, frozen native-QPA actual-GL 또는 완전 차단망 기기 검증을 대신하지 않는다.

AMR v2 `payload_type="artifact_document"` 1.0의 strict 저장·production-loader staged reopen·checksum·원자 교체와 독립 프로세스 source rebind/materialization은 현재 차단 게이트다. `tests/test_artifact_new_process_roundtrip.py`는 프로세스 A와 B의 PID가 다름을 확인하고, PLY를 다른 경로와 `.raw-scan` suffix로 옮긴 뒤 저장된 parser/unit으로 다시 decode한다. 새로 계산한 source SHA-256·크기와 geometry SHA-256, active Align ID·matrix, parser/unit, world vertices가 같아야 통과한다.

Native GUI의 한-artifact Open/Align commit/save/load는 `ArtifactWorkbench.snapshot.session.document`를 source of truth로 사용하며 MainWindow의 session field는 이행 중 compatibility mirror다. Open과 Align은 ticket/CAS/two-phase publication을 사용한다. Cutline/Outline/Digital Rubbing은 Qt-free `ArtifactMeasurementController`가 recipe/context와 record ID를 Workbench 단위로 예약하고 worker computation만 받은 뒤 current same-Align session에 rebase한다. append-only record publication은 live SceneObject의 document binding만 CAS하고 mesh/VBO를 재생성하지 않는다. `ArtifactExportController`는 vector/rubbing package의 생성·전체 검증·prepared capability 발급까지 worker에서 수행하고 GUI dispatcher의 final Workbench fence에서는 빠른 identity/fingerprint 확인과 rename만 실행한다. command handler는 단일 `TaskThread`를 사용하며 늦은 finished signal이 새 worker/dialog를 지우지 못한다. 재개방 기록은 자동 최신 fallback 없이 명시적으로 선택하고 request token으로 늦은 preview가 최신 선택을 지우지 못하게 한다. rollback 가능한 측정 게시 실패는 exact result를 재시도 queue에 보존한다. save 전 projection snapshot과 geometry를 재검증하며 active/보류 실측과 아직 `DerivedRecord`로 이식되지 않은 선택·기록면·평가 결과는 누락 저장하지 않고 fail closed한다. authority rollback·scene 복원·finalize가 불확실한 fatal 상태에서는 검증된 Open 전까지 저장·실측·내보내기를 모두 막고 task-local 오류가 재열기 배너를 덮지 못한다. 별도의 legacy destructive bake 이후 Save 성공은 native Align 복원 증거가 아니다. legacy runtime은 그런 vertex mutation을 `_amr_has_unpersisted_bake`로 표시하고 snapshot 저장을 차단한다.

Native measurement 취소는 `QThread.terminate()`를 사용하지 않는다. 사용자 요청은 먼저 record 게시 권위를 회수하고, Cutline face/path, Outline polygon/union/topology, Digital Rubbing face/row/integral/relief의 다음 안전 경계에서 worker가 고유 core 취소 예외로 종료한다. 현재 실행 중인 단일 NumPy·GEOS 호출, worker 시작 전 Rubbing resource estimate와 scene materialize/source 재검증은 선점할 수 없으므로 “즉시 종료”로 표현하지 않는다. 앱 종료 중 실행 worker를 취소하고 종료 완료를 기다리는 수명주기와 preflight의 worker 이관은 후속 차단 항목이다.

### 대좌표 render-origin: 실제 source driver 게이트 추가

현재 차단 suite는 absolute float64 world-mm document/mesh 불변, `>= 1e9 mm` offset의 mm/sub-mm feature, CPU float64 subtraction 뒤 객체별 relative float32 VBO payload, 안정적인 scene origin에 대한 camera/model affine rebasing, 활성 world overlay의 relative 제출, absolute float64 CPU face 계산과 render origin 비직렬화를 검증한다. 또한 exact modelview·projection·viewport·scene origin과 depth-affecting scene signature를 같이 게시한 frame authority의 project/unproject/ray, depth pick·Ctrl drag 수명주기, 변환·resize·scene rollback·repaint 전 상태 변경 후 stale frame 거부를 검증한다. pure coordinate helper와 해당 테스트는 M0 Pyright에, mocked OpenGL upload·overlay·interaction 테스트는 명시 3-OS persistence suite에 포함한다.

`src/gui/opengl_driver_smoke.py`는 OpenGL 2.1 compatibility·24-bit depth를 QApplication 전에 요청하고 native QPA에서 `WA_DontShowOnScreen`인 실제 `Viewport3D` widget FBO를 렌더·readback한다. compositor의 최종 on-screen presentation을 검증하는 테스트는 아니다. `[1e9, -2e9, 3e9] mm` 기준점에 0.25 mm 간격으로 분리된 두 판과 0.125 mm 높이차, 0.25 mm native vector overlay를 만든다. 실제 production `add_mesh_object → update_vbo → paintGL → RenderFrameSnapshot → glReadPixels → pick_point_on_mesh_info`를 통과하며 다음을 fail closed로 확인한다.

- GL vendor/renderer/version, current widget context, depth-preserving `PartialUpdate`, complete default FBO, depth bits
- 별도 Qt FBO의 정확한 RGBA/depth clear-readback
- driver VBO object와 작은 relative float32 payload; 원본 absolute float64 vertex 불변
- 원근/상면 정사영 각각의 두 depth component, plate pixel, 빈 gap, relative overlay pixel
- 보정 검색을 끈 실제 depth pick, 같은 frame serial, 해석적 ray-plane oracle와 0.125 mm 높이차

2026-07-12 로컬 Python 3.12.13/macOS arm64 Apple M4(`OpenGL 2.1 Metal - 90.5`)에서는 code commit `f25b424d6936e6e8832a81c7a6683cb58515e546`의 clean source tree와 결합된 `opengl-driver-smoke-f25b424-darwin-arm64.json`이 61개 조건을 통과했다. 원근 pick 높이차는 `0.124783 mm`, 최대 ray-plane 오차는 `0.001213 mm`; 정사영 높이차는 `0.124998 mm`, 최대 오차는 `0.00000191 mm`였다. Linux CI에서는 clean source commit `166103dcf0ea`와 결합된 xcb + llvmpipe report가 61/61을 통과했고 원근 높이차 `0.124599 mm`·최대 오차 `0.002970 mm`, 정사영 높이차 `0.124998 mm`·최대 오차 `0.00000381 mm`를 기록했다. JSON은 tested commit/tree 상태, runtime-lock SHA-256, dependency version과 UTC 시각을 포함하고, CI report는 14일 artifact로 보존한다. 이 두 결과는 Linux software rasterizer와 한 macOS 장치의 증거이며 Windows, Intel Mac, frozen executable, 대표 하드웨어 GPU/driver 또는 compositor presentation을 대신하지 않는다. render origin은 metadata·Align·record·QC·hash·export 권위 값에 저장하지 않는다.

## 게이트 변경 원칙

- 검사를 삭제하거나 `continue-on-error`로 바꾸는 것은 별도 근거와 리뷰가 필요하다.
- fallback, sampling, 단위 추정, 원본 불일치를 성공으로 숨기는 테스트를 추가하지 않는다.
- 플랫폼 매트릭스가 원격 CI에서 실제로 통과하기 전에는 “3개 OS 검증 완료”라고 표현하지 않는다.
