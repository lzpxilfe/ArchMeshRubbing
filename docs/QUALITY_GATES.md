# Quality Gates

이 문서는 ArchMeshRubbing의 현재 개발 기준선과 CI가 실제로 보장하는 범위를 기록한다. 첫 공개 안정판의 제품 대상은 Windows 하나다. Linux quality job은 빠른 정적 검사와 전체 pytest 수집을 위한 구현 환경일 뿐 Linux 배포 지원 약속이 아니며, macOS·Linux 패키징은 현재 완료 조건에서 제외한다.

## 재현 환경

CI의 기준 Python은 3.12이다. `requirements.txt`는 `requirements/runtime-py312.lock`을 포함하여 source와 frozen build가 같은 exact runtime resolution을 사용한다. 일반 개발 toolchain은 `requirements/build-py312.lock`, 검증 도구는 `requirements-dev.txt`에 고정한다. 제품 대상 Windows frozen job은 `requirements/windows-py312-x64-hashed.lock`의 정확한 wheel 17개만 `--require-hashes --only-binary=:all:`로 설치한다. 권위 Outline overlay 조합은 `shapely==2.1.2`와 Windows wheel의 GEOS `3.13.1`로 recipe와 runtime gate에 고정한다. 이 wheel lock은 Windows x64/CPython 3.12 전용이며 다른 OS 배포 증거가 아니다.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

## 차단 게이트

Pull request에서 다음 code-quality 검사, Windows offline workflow smoke와 Windows native-QPA software OpenGL smoke가 모두 통과해야 한다.

```bash
python -m ruff check .
python -c "import subprocess,sys; raise SystemExit(subprocess.call([sys.executable,'-m','pyright','--pythonpath',sys.executable,'-p','pyright-m0.json']))"
python -m pytest -q
```

- Ruff는 전체 트리를 검사한다.
- pytest는 pytest 함수와 `unittest.TestCase`를 모두 수집한다. 별도의 `unittest discover`는 하위 호환성 확인용이며 CI의 권위 수집기가 아니다.
- `pyright-m0.json`은 persistence·source identity·source manifest/bundle·unit·matrix 경계에 더해 M0-6의 `artifact_document`, `geometry_identity`, `artifact_scene_adapter`, `artifact_session`, core cooperative cancellation, Qt/OpenGL-free `artifact_workbench`·`artifact_workflow_progress`·`artifact_measurements`·`artifact_exports`·complete workflow self-test, `src/gui/opengl_context.py`의 명시적 surface 계약, actual-driver CLI와 actual context를 열지 않는 helper/support tests, Qt/OpenGL-free render-coordinate algebra인 `src/gui/render_coordinates.py`, known-record registry, RFC 8785 canonical JSON, vector record/export, Cutline, fixed-grid Outline/topology, Digital Rubbing record/extractor, canonical GA8 PNG, authoritative tile unwrap record/export 및 해당 테스트를 포함하는 M0 신뢰 커널 범위다. 독립 프로세스 source-closure 왕복과 offline vector/rubbing/tile-unwrap package 테스트도 목록에 포함한다. wrapper 명령은 활성 Python interpreter를 Pyright에 명시해 Windows CI의 dependency를 정확히 해석한다.

과거 `opengl-driver-smoke`는 Ubuntu 24.04의 24-bit Xvfb, native `xcb`, Mesa llvmpipe로 다음 명령에 해당하는 검사를 통과했다.

```bash
xvfb-run -a \
  -s "-screen 0 1280x1024x24 +extension GLX +render -noreset" \
  python -m src.gui.opengl_driver_smoke \
  --qt-platform xcb \
  --report build/opengl-driver-smoke.json
```

이 결과는 실제 OpenGL context/FBO/VBO/pixel/depth readback을 사용했지만 Linux Mesa software rasterizer의 과거 증거다. 현재 Windows 안정판 차단 job이 아니며 대표 하드웨어 GPU 인증이라고 표현하지 않는다. code commit `166103dcf0ea`의 [GitHub Actions run 29182584810](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29182584810)에서 61개 조건이 통과했다.

M0-6 native artifact 신뢰 경계를 빠르게 재검증할 때는 다음 focused suite를 사용한다. 이 명령은 full pytest를 대체하지 않는다.

```bash
python -m pytest -q \
  tests/test_project_file.py \
  tests/test_build_info.py \
  tests/test_build_manifest.py \
  tests/test_build_provenance.py \
  tests/test_release_evidence.py \
  tests/test_build_native.py \
  tests/test_source_archive.py \
  tests/test_source_identity.py \
  tests/test_source_manifest.py \
  tests/test_mesh_import_recipe.py \
  tests/test_mesh_external_dependencies.py \
  tests/test_mesh_verified_stream.py \
  tests/test_alignment_utils.py \
  tests/test_artifact_document.py \
  tests/test_geometry_identity.py \
  tests/test_artifact_scene_adapter.py \
  tests/test_artifact_session.py \
  tests/test_artifact_workbench.py \
  tests/test_artifact_workflow_progress.py \
  tests/test_artifact_workflow_self_test.py \
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
  tests/test_artifact_tile_unwrap.py \
  tests/test_artifact_tile_unwrap_export.py \
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
- exact-key mesh import recipe, Trimesh·NumPy·Pillow parser-subset digest와 전체 lock provenance 분리, GUI ticket부터 embedded reopen까지 동일 receipt 실행
- self-contained v1 `deny_external`과 OBJ→MTL→texture, PLY `TextureFile`, glTF/GLB external buffer의 v2 `closed_manifest`; logical-path 정규화, remote/absolute/traversal/symlink 탈출·미선언/미사용/변조 resource 거부
- content-addressed source bundle v1/v2 schema, 동일 content alias, checksum을 다시 쓴 변조 package 거부, 외부 source closure 삭제 뒤 `.amr` relocation·reopen·archive-to-archive resave
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
- 서로 다른 PID에서 self-contained PLY와 textured OBJ source closure를 삭제·relocation한 뒤 saved-parser reopen하여 primary/dependency/texture/geometry hash, Align matrix와 world vertex가 같은 durable artifact round-trip
- RFC 8785 cross-language number golden과 vector payload/recipe semantic SHA-256
- canonical-mm Cutline의 exact box·multi-component·Front/Right/oblique frame, face order/winding, ambiguity fail-closed
- canonical-mm 6-view Outline의 fixed-grid projected-triangle union, concavity·hole·island 보존, face order/winding/duplicate 안정성
- translated integer lattice의 `1e9 mm` survey offset, grid collapse/merge receipt, chunked balanced union과 resource-limit fail-closed
- Outline ring simple/area, hole ownership/contact, hole/exterior overlap·nesting, production recipe/frame/grid/ID 재검증
- native Outline widget/record/record-derived overlay와 screenshot/OpenCV legacy export 우회 차단
- `READY + FRESH + verified payload`만 허용하는 1:1 SVG, full sidecar-claim binding, confirmed unit/active Align/dependency closure
- SVG/sidecar tamper, XML active content/DTD, duplicate/pathological JSON, privacy allowlist, size cap, concurrent no-replace publish
- package relocation 후 원본 문서·mesh·GUI 없이 별도 PID에서 수행하는 offline 검증
- `.amr` embedded source의 saved-parser 재물질화와 `.amr-vector`/`.amr-rubbing`/`.amr-unwrap` 자동 판별을 하나로 묶는 `--verify-artifact`; exact project 결합 성공·불일치 실패, manifest-only AMR 실패, symlink·mixed marker·tamper 거부, 절대 경로 없는 결정적 closed JSON receipt
- versioned vector payload/export Draft 2020-12 schemas
- six-view canonical-mm front-depth raster, fixed integer µm quantization/local-mean tone mapping, coverage alpha와 multi-layer QC
- face order·winding·duplicate, hole, large survey offset, resource limit 및 Align 전환 후 late Digital Rubbing 결과 거부
- canonical GA8 PNG의 고정 chunk/DEFLATE bytes, exact `pHYs`, RFC 8785 iTXt metadata, pixel/CRC/chunk/scale tamper 거부
- `READY + FRESH + recomputed raster`만 허용하는 `.amr-rubbing` 1:1 PNG package, relocation 후 독립 PID offline 검증, no-replace publish와 privacy allowlist
- 명시적 canonical 장축과 source face-range selection을 고정하고 자동 fallback, 1 µm grid collapse, foldover, section-fit/mean/p95 distortion gate 실패를 READY record로 승격하지 않는 `surface.tile_unwrap.v1`
- 동일 recipe 재계산과 canonical binary round-trip, source vertex/face correspondence, exact µm bounds·component hash·전체 payload SHA-256, Top/Bottom 구분과 Align stale 보존
- `.amr-unwrap` canonical binary/flat OBJ/physical-mm SVG/provenance sidecar의 byte 결정성, four-member tamper 거부, offline 검증, hidden staging·exact prepared inode/fingerprint capability·no-overwrite final-authority publish와 destination race 정리
- versioned tile unwrap receipt/export Draft 2020-12 schema와 axis·fallback·privacy closed contract
- exact Git commit/tree의 regular blob만 허용하는 결정적 corresponding-source ZIP, 내부 manifest·외부 sidecar·Git blob ID·SHA-256·라이선스 결합과 저장소 삭제 뒤 offline 검증
- portable/source/evidence exact hash와 실제 payload file set, GitHub repository/workflow/run attempt/Windows X64 hosted-runner identity를 결합하는 canonical unsigned provenance, 한글 추출본 offline 재검증, closed `authentication=none` 계약과 변조·source mismatch 거부
- vector/rubbing/tile-unwrap worker의 hidden same-parent staging·전체 검증·prepared inode/fingerprint capability, 빠른 final Workbench record-authority fence, same-Align append 허용, Align/Open stale 정리와 pending Open GUI 취소 정책
- 고정 길이 staging UUID 충돌·quarantine/foreign inode 보존, exact result/prepared capability 위조·사전 목적지 이동·destination race 차단, post-rename 실제/미지원 directory-fsync `committed` 내구성 경고
- Qt-free Cutline/Outline/Digital Rubbing/기와 전개 work item, exact result capability, same-Align 병렬 rebase, Workbench 공유 record reservation, Align/Open stale·취소·rollback 방어와 pending Open 게시 재시도
- core 고유 `RuntimeError` 취소 신호, 세 extractor 내부의 bounded polling과 대형 NumPy 단계·최종 결과 fence, false-probe payload/raster/QC 동일성, controller의 `FAILED > STALE > CANCELLED` 경합 우선순위·`CANCELLING → CANCELLED` slot 보존, GUI one-shot 취소 요청·대기 창 수명·무경고 종료
- Digital Rubbing 누적 peak-memory admission, UV/texture materialize 복사비의 무복사 사전 차단, controller별 한도 우회 방어, 실행 exactly-once, 취소 worker 종료 전 slot 보존
- 재개방 프로젝트의 READY + FRESH vector/rubbing/tile-unwrap 명시 선택, background recipe 재계산과 완료 시 document/Align/record 재검증, active raster 예산 중첩 차단, 일시 게시 실패 재시도 queue와 보류 중 저장 차단
- native Align commit/parent activation handler의 GUI-thread source geometry hash·canonical materialization 부재, 잠긴 non-cancelable worker, 완료 시 exact object/mesh/binding/preview와 session/state/epoch/project-path fence, stale 결과의 publication 전 폐기, OpenGL VBO/scene publication의 GUI-context 유지
- native Save handler의 GUI-thread canonical materialization·project writer·Git subprocess 부재, worker preflight 실패 시 writer 미호출, committed directory-fsync 경고 보존, Save As 경로의 Workbench CAS 채택, 저장 중 authority/project-path 변경 또는 final CAS 경합 시 과거 snapshot 경고와 현재 Save As 상태 보존, non-cancelable 진행 dialog의 worker 종료 전 close/Escape 차단과 종료 시 bounded join
- versioned Digital Rubbing receipt/export Draft 2020-12 schemas

## 검증 기준선과 역사적 기록

아래 표는 2026-07-12 M0-6 당시의 역사적 이식성 결과다. 현재 제품 완료 판정에는 Windows만 사용하며, 최신 Windows 기준선은 표 아래에 기록한다.

| 검사 | 결과 |
|---|---:|
| GitHub Actions Python 3.12.13 quality `python -m pytest -q` | 572 passed, 117 subtests passed |
| macOS / Ubuntu persistence smoke | 각각 501 passed, 117 subtests passed |
| Windows persistence smoke | 498 passed, 3 platform-specific skips, 117 subtests passed |
| GitHub Actions `python -m ruff check .` | passed |
| GitHub Actions M0 Pyright wrapper command | 0 errors |
| Ubuntu 24.04 xcb + Mesa llvmpipe actual OpenGL | 61/61 passed at clean source commit `166103dcf0ea`, run `29182584810` |
| Python 3.12.13 macOS arm64 actual OpenGL driver smoke | 61/61 passed at clean code commit `f25b424d6936`, Apple M4, perspective + top orthographic |
| ArtifactDocument + vector/rubbing/tile-unwrap payload/export + offline verification receipt Draft 2020-12 schemas + golden | passed |
| Python 3.12.13 macOS arm64 frozen self-test | 10/10 passed at code commit `898a8bfc144f` (unsigned, `source_tree=clean`, `native-self-test-local-smoke-898a8bfc144f-darwin.json`) |
| GitHub Actions 3-OS frozen build + executable self-test | Ubuntu, Windows, macOS 모두 passed at commit `e4bf6dcac4b1`, run `29213279508` |

2026-07-13 source-closure 기준 commit `6898f98d2fb3`은 Python 3.12 원격 CI에서 full pytest `660 passed, 128 subtests passed`, Ruff, M0 Pyright, Windows persistence와 Windows frozen executable self-test를 통과했다. 이후 첫 안정판의 플랫폼 완료 판정은 Windows job만 사용한다.

현재 Windows 코드 기준 commit `d4c7d94037be`는 [source CI run 29279156637](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29279156637)에서 full pytest `753 passed, 128 subtests passed`, Ruff, M0 Pyright `0 errors`, Windows workflow `651 passed, 5 skipped, 118 subtests passed`와 qwindows+llvmpipe actual-frame `66/66`을 통과했다. [portable package run 29279156712](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29279156712)도 frozen·한글 경로 portable 실행 파일의 14-check complete-workflow self-test, actual-frame `66/66`, 추출본·방화벽 규칙 정리를 모두 통과했다. complete-workflow check는 Open→Align→3/6/6→completed AMR offline reopen→1:1 SVG/PNG 재현을 한 번에 검증한다. 과거 commit `19558f324deb`의 installer run은 역사적 내부 검증이며 현재 배포 gate는 compiler 비종속 portable ZIP으로 교체했다.

## 아직 차단하지 않는 검사

전체 트리 Pyright는 아직 통과하지 않는다. CI에서는 이 결과를 `continue-on-error`로 보고하여 부채가 보이게 하되, M0 범위를 넘는 기존 오류 때문에 모든 변경을 막지는 않는다. 신뢰 커널 전환이 진행될 때마다 차단 범위를 넓힌다. 독립 프로세스 테스트의 worker program은 Python 문자열이므로 Pyright가 문자열 내부를 분석하지는 않지만, 차단 pytest가 두 문자열을 각각 새 interpreter에서 실제 실행한다.

Windows workflow smoke에서는 프로젝트 저장, source/geometry identity와 versioned source manifest/bundle, ArtifactDocument·scene adapter·session·application workbench와 record-derived workflow progress, ticketed Open과 explicit Align gate, RFC 8785/vector record/export/Cutline/Outline/topology/schema, Digital Rubbing record/extractor/canonical PNG/export/schema, authoritative tile-unwrap record/desktop panel/staged export, 독립 프로세스 source closure 및 relocated vector/rubbing/tile-unwrap package 왕복, render-coordinate algebra·relative VBO/native preview smoke, matrix golden, GUI 런처, MainWindow 생성, native source-of-truth binding, native Cutline/Outline/Rubbing/기와 전개 command, 3/6/6 순차 gate와 completed `.amr` offline reopen, session/version/epoch 및 late-result 방어, legacy export 우회와 fail-closed 경계를 실행한다. Linux quality job은 별도로 전체 테스트와 Ruff·M0 Pyright를 빠르게 실행한다. Windows GUI 스모크는 `QT_QPA_PLATFORM=offscreen`이므로 CPU/document/scene transaction과 widget wiring을 검증하지만 실제 OpenGL frame을 증명하지 않는다.

같은 Windows job의 다음 단계는 `QT_QPA_PLATFORM=windows`, `QT_OPENGL=software`로 실제 `QOpenGLWidget` context를 열고 `src.gui.opengl_driver_smoke` report를 검증한다. Qt wheel에 포함된 `opengl32sw.dll`을 강제하고 PyOpenGL의 GL/WGL dispatch도 같은 DLL에 결합한다. 이렇게 해야 Qt의 software context와 시스템 `opengl32.dll`을 섞어 호출하지 않는다. qwindows/context/768×768 FBO/VBO/pixel/depth/pick 경계를 실행하며, 이는 software OpenGL 실제-frame 증거이지 대표 GPU 또는 compositor 최종 presentation 인증은 아니다.

별도 `package-smoke.yml`은 `main` push, pull request, 수동 실행에서 Windows x64/CPython 3.12의 hash-locked binary wheel set, immutable build manifest, PyInstaller spec과 frozen executable의 file-report self-test를 실행한다. PyInstaller 뒤 exact Git commit/tree/blob의 corresponding-source ZIP을 payload에 넣고, 실제 payload 전체 파일의 path/size/SHA-256 manifest, runtime 10개를 exact wheel SHA-256에 묶은 SPDX 2.3 SBOM, wheel 메타데이터와 라이선스 원문에서 만든 machine/human NOTICE를 생성한다. evidence index가 이 네 문서를 묶고 `release_evidence`와 `source_archive`를 포함한 14-check가 frozen 및 portable payload에서 모두 다시 계산된다. complete-workflow check는 작은 PLY를 실제 application authority로 열고 3/6/6 record를 만든 뒤 completed `.amr`를 외부 원본 없이 재열어 1:1 SVG/PNG를 재현·이동하고, 통합 verifier로 AMR materialization과 두 export의 exact-project 결합까지 검증한다. 이어 같은 frozen executable의 `--opengl-driver-smoke-report`를 native qwindows/software OpenGL로 실행한다. 그 뒤 표준 라이브러리로 portable ZIP과 canonical sidecar를 만들고 모든 entry·release evidence·source archive를 검증하며, 외부 unsigned provenance에 이 artifact들과 GitHub workflow/run/runner identity를 결합한다. `문화유산 기록` 한글 경로 추출본도 같은 provenance와 일치해야 한다. 추출 실행 파일은 outbound 차단 상태의 14-check, public `--verify-artifact ... --report ...` exit/receipt/privacy 경계와 actual-frame을 다시 통과하고 디렉터리 삭제 뒤 잔여물이 없어야 한다. provenance는 `authentication=none`인 무결성 기록이며 서명된 출처 인증이 아니다. 라이선스·서명 게이트가 해결되기 전에는 artifact upload와 release 단계를 두지 않는다.

과거 [진단 run 29254942224](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29254942224)은 비 ASCII installer 경로에서 software OpenGL context 생성에 실패했다. 이 실패를 성공으로 재분류하지 않고, 현재 portable gate가 한글 payload·report 경로의 offline workflow와 software OpenGL을 모두 차단 조건으로 직접 재검증하도록 승격했다. 대표 하드웨어 GPU/driver의 비 ASCII 경로는 여전히 실제 pilot 범위다.

AMR v2 `payload_type="artifact_document"` 1.0의 strict 저장·content-addressed source closure embedding·production-loader staged reopen·checksum·원자 교체와 독립 프로세스 materialization은 현재 차단 게이트다. `tests/test_artifact_new_process_roundtrip.py`는 프로세스 A와 B의 PID가 다름을 확인하고, 프로세스 A가 `.amr`를 저장한 뒤 외부 PLY 또는 textured OBJ의 OBJ·MTL·PNG 전체를 삭제하고 package를 relocation한다. 프로세스 B는 `.amr`의 embedded source closure만 saved parser/unit으로 decode하며, 새로 계산한 primary/dependency SHA-256·크기와 texture/geometry SHA-256, active Align ID·matrix, parser/unit, world vertices가 같아야 통과한다.

Native GUI의 한-artifact Open/Align commit/save/load는 `ArtifactWorkbench.snapshot.session.document`를 source of truth로 사용하며 MainWindow의 session field는 이행 중 compatibility mirror다. Open과 Align은 ticket/CAS/two-phase publication을 사용한다. Align commit/parent activation handler는 GUI에서 가벼운 scene/preview guard만 캡처하고 source hash·candidate session·canonical materialization을 잠긴 worker에서 준비한다. callback은 exact object/mesh/binding/preview와 Workbench session/state/epoch/project path를 다시 확인한 뒤에만 GUI OpenGL context에서 VBO와 scene을 게시한다. Cutline/Outline/Digital Rubbing/기와 전개는 Qt-free `ArtifactMeasurementController`가 recipe/context와 record ID를 Workbench 단위로 예약하고 worker computation만 받은 뒤 current same-Align session에 rebase한다. append-only record publication은 live SceneObject의 document binding만 CAS하고 mesh/VBO를 재생성하지 않는다. `ArtifactExportController`는 vector/rubbing/tile-unwrap package의 생성·record recipe 재계산·전체 검증·prepared capability 발급까지 worker에서 수행하고 GUI dispatcher의 final Workbench fence에서는 빠른 identity/fingerprint 확인과 rename만 실행한다. command handler는 단일 `TaskThread`를 사용하며 늦은 finished signal이 새 worker/dialog를 지우지 못한다. 재개방 기록은 자동 최신 fallback 없이 명시적으로 선택하고 request token으로 늦은 preview가 최신 선택을 지우지 못하게 한다. rollback 가능한 측정 게시 실패는 exact result를 재시도 queue에 보존한다. native Save는 immutable session을 캡처한 뒤 projection/geometry 비교와 source closure 재해시·ZIP/fsync·production reopen/materialization을 잠긴 worker dialog에서 수행한다. 완료 시 exact session/state/epoch/기존 project path가 유지될 때만 현재 경로와 migration flag를 갱신한다. active/보류 실측과 아직 `DerivedRecord`로 이식되지 않은 선택·기록면·평가 결과는 누락 저장하지 않고 fail closed한다. authority rollback·scene 복원·finalize가 불확실한 fatal 상태에서는 검증된 Open 전까지 저장·실측·내보내기를 모두 막고 task-local 오류가 재열기 배너를 덮지 못한다. 별도의 legacy destructive bake 이후 Save 성공은 native Align 복원 증거가 아니다. legacy runtime은 그런 vertex mutation을 `_amr_has_unpersisted_bake`로 표시하고 snapshot 저장을 차단한다.

Native measurement 취소는 `QThread.terminate()`를 사용하지 않는다. 사용자 요청은 먼저 record 게시 권위를 회수하고, Cutline face/path, Outline polygon/union/topology, Digital Rubbing face/row/integral/relief, 기와 section/face 검사의 다음 안전 경계에서 worker가 고유 core 취소 예외로 종료한다. vector/rubbing/tile-unwrap export도 같은 취소 UI에서 publication 권위를 먼저 회수하고 worker가 이미 만든 owned staging을 quarantine 경로로 안전하게 정리한다. 앱 종료는 active authoritative task에 이 취소를 one-shot으로 전달하고 최대 30초 join을 기다리며, 완료를 증명하지 못하면 창을 닫지 않는다. join 성공 뒤 task signal과 identity를 제거하므로 늦은 완료 콜백은 게시될 수 없다. 이 종료/timeout/late-callback 경계는 GUI 차단 테스트에 포함한다. native Align commit/parent activation에서 GUI thread source hash·materialization이 0회인지와 preview/authority가 바뀐 worker 결과가 게시 전에 폐기되는지, native Cutline/Outline/Digital Rubbing/기와 전개 handler가 GUI thread에서 canonical scene materialization을 하지 않는지, worker preflight 실패가 terminal 상태와 record 예약 해제로 끝나는지, Rubbing 전체 resource estimate 도중 취소가 worker 종료 전까지 memory slot을 유지하는지를 차단 테스트로 검증한다. 현재 실행 중인 단일 NumPy·GEOS 호출, Align source hash/materialization과 measurement preflight 내부의 단일 materialization·resource-estimate 호출은 선점할 수 없으므로 “즉시 종료”로 표현하지 않는다. native heavy GUI-thread preflight 제거는 완료했으며 legacy mesh/profile/slice worker 통합은 후속 항목이다.

### 대좌표 render-origin: 실제 source driver 게이트 추가

현재 차단 suite는 absolute float64 world-mm document/mesh 불변, `>= 1e9 mm` offset의 mm/sub-mm feature, CPU float64 subtraction 뒤 객체별 relative float32 VBO payload, 안정적인 scene origin에 대한 camera/model affine rebasing, 활성 world overlay의 relative 제출, absolute float64 CPU face 계산과 render origin 비직렬화를 검증한다. 또한 exact modelview·projection·viewport·scene origin과 depth-affecting scene signature를 같이 게시한 frame authority의 project/unproject/ray, depth pick·Ctrl drag 수명주기, 변환·resize·scene rollback·repaint 전 상태 변경 후 stale frame 거부를 검증한다. pure coordinate helper와 해당 테스트는 M0 Pyright에, mocked OpenGL upload·overlay·interaction 테스트는 Windows workflow smoke에 포함한다.

`src/gui/opengl_driver_smoke.py`는 OpenGL 2.1 compatibility·24-bit depth를 QApplication 전에 요청하고 native QPA의 실제 `Viewport3D` widget FBO를 렌더·readback한다. Windows는 768×768 고정 크기의 비활성 native tool window를 노출하고, 자동 desktop이 paint event를 합치더라도 `makeCurrent()`가 결합한 widget default FBO에서 production `paintGL()`을 명시적으로 실행한다. 역사적 macOS/Linux probe는 `WA_DontShowOnScreen`을 유지한다. compositor의 최종 on-screen presentation을 검증하는 테스트는 아니다. `[1e9, -2e9, 3e9] mm` 기준점에 0.25 mm 간격으로 분리된 두 판과 0.125 mm 높이차, 0.25 mm native vector overlay를 만든다. 실제 production `add_mesh_object → update_vbo → paintGL → RenderFrameSnapshot → glReadPixels → pick_point_on_mesh_info`를 통과하며 다음을 fail closed로 확인한다.

- GL vendor/renderer/version, current widget context, depth-preserving `PartialUpdate`, complete default FBO, depth bits
- 별도 Qt FBO의 정확한 RGBA/depth clear-readback
- driver VBO object와 작은 relative float32 payload; 원본 absolute float64 vertex 불변
- 원근/상면 정사영 각각의 두 depth component, plate pixel, 빈 gap, relative overlay pixel
- 보정 검색을 끈 실제 depth pick, 같은 frame serial, 해석적 ray-plane oracle와 0.125 mm 높이차

2026-07-13 Windows 대상 commit `b12d4874a4a8`의 [source CI run 29251668123](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29251668123)은 qwindows + bundled llvmpipe에서 66/66 조건을 통과했다. 768×768 FBO에서 원근 pick 높이차 `0.124599 mm`·최대 ray-plane 오차 `0.002970 mm`, 정사영 높이차 `0.124998 mm`·최대 오차 `0.00000381 mm`를 기록했다. 같은 commit의 [frozen package run 29251668029](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29251668029)도 동일 gate를 통과했다. JSON은 tested commit/tree 상태, runtime-lock SHA-256, dependency version과 UTC 시각을 포함한다. 과거 macOS Apple M4와 Linux xcb 결과는 이식성의 역사적 증거일 뿐 현재 지원 판정에 사용하지 않는다. Windows software rasterizer 결과도 대표 하드웨어 GPU/driver 또는 compositor presentation을 대신하지 않는다. render origin은 metadata·Align·record·QC·hash·export 권위 값에 저장하지 않는다.

## 게이트 변경 원칙

- 검사를 삭제하거나 `continue-on-error`로 바꾸는 것은 별도 근거와 리뷰가 필요하다.
- fallback, sampling, 단위 추정, 원본 불일치를 성공으로 숨기는 테스트를 추가하지 않는다.
- Windows source·frozen·한글 경로 portable actual-frame 통과는 software-renderer와 archive 역학의 완료 증거로만 표현한다. 라이선스·서명·대표 GPU·실물 대용량 pilot 전에는 “Windows 안정판 검증 완료”라고 표현하지 않으며, macOS·Linux 과거 통과 기록을 현재 배포 지원으로 표현하지 않는다.
