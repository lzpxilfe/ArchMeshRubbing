# Quality Gates

이 문서는 ArchMeshRubbing의 현재 개발 기준선과 CI가 실제로 보장하는 범위를 기록한다. 제품·문서·배포·차단 CI 대상은 Windows x64 하나이며, 지원 운영체제는 Windows 10 version 1809 이상 x64와 Windows 11 x64다. 최소 버전은 고정한 Qt 6.11의 [공식 지원 플랫폼](https://doc.qt.io/qt-6.11/supported-platforms.html)을 따른다. Windows ARM64·32-bit·Server 최종 사용자 환경, 비 Windows 운영체제와 호환 계층, installer·MSIX·Store 배포는 지원 대상이 아니다. 코드에 남은 비 Windows backend는 내부 회귀 시험용이고 제품 완료 근거가 아니다.

GitHub-hosted `windows-latest` x64 runner는 자동 차단 게이트일 뿐 소비자용 Windows 10/11 PC의 하드웨어·driver 증거가 아니다. 공개 전 지원 대상 운영체제 각각의 실제 PC 파일럿을 별도로 통과해야 한다.

## 재현 환경

CI의 기준 Python은 3.12이다. `requirements.txt`는 `requirements/runtime-py312.lock`을 포함하여 source와 frozen build가 같은 direct runtime 버전을 사용한다. 일반 개발 toolchain은 `requirements/build-py312.lock`, 검증 도구의 direct dependency는 `requirements-dev.txt`에 고정하지만 이 두 파일은 transitive wheel 재현 증거가 아니다. 제품 대상 Windows frozen job만 `requirements/windows-py312-x64-hashed.lock`의 정확한 transitive wheel 17개를 `--require-hashes --only-binary=:all:`로 설치한다. 권위 Outline overlay 조합은 `shapely==2.1.2`와 Windows wheel의 GEOS `3.13.1`로 recipe와 runtime gate에 고정한다. 이 wheel lock은 Windows x64/CPython 3.12 전용이며 다른 OS 배포 증거가 아니다.

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

## 차단 게이트

Pull request에서 다음 code-quality 검사, Windows offline workflow smoke와 Windows native-QPA software OpenGL smoke가 모두 통과해야 한다.

```bat
python -m ruff check .
python -c "import subprocess,sys; raise SystemExit(subprocess.call([sys.executable,'-m','pyright','--pythonpath',sys.executable,'-p','pyright-m0.json']))"
python -m pytest -q
```

- Ruff는 전체 트리를 검사한다.
- pytest는 pytest 함수와 `unittest.TestCase`를 모두 수집한다. 별도의 `unittest discover`는 하위 호환성 확인용이며 CI의 권위 수집기가 아니다.
- `pyright-m0.json`은 persistence·source identity·source manifest/bundle·unit·matrix 경계에 더해 M0-6의 `artifact_document`, `geometry_identity`, `artifact_scene_adapter`, `artifact_session`, core cooperative cancellation, Qt/OpenGL-free `artifact_workbench`·`artifact_workflow_progress`·`artifact_measurements`·`artifact_exports`·`artifact_survey_exports`·complete workflow self-test, `src/gui/opengl_context.py`의 명시적 surface 계약, actual-driver CLI와 actual context를 열지 않는 helper/support tests, Qt/OpenGL-free render-coordinate algebra인 `src/gui/render_coordinates.py`, known-record registry, RFC 8785 canonical JSON, vector record/export, Cutline, fixed-grid Outline/topology, Digital Rubbing record/extractor, 1 µm geometry metrics receipt, triangle+barycentric 표면 거리·원 맞춤 지름 receipt, canonical GA8 PNG, atomic survey export, authoritative tile unwrap record/export, fail-closed public-release policy와 release evidence 및 해당 테스트를 포함하는 M0 신뢰 커널 범위다. 독립 프로세스 source-closure 왕복과 offline vector/rubbing/survey/tile-unwrap package 테스트도 목록에 포함한다. wrapper 명령은 활성 Python interpreter를 Pyright에 명시해 Windows CI의 dependency를 정확히 해석한다.

M0-6 native artifact 신뢰 경계를 빠르게 재검증할 때는 다음 focused suite를 사용한다. 이 명령은 full pytest를 대체하지 않는다.

```bat
python -m pytest -q ^
  tests/test_project_file.py ^
  tests/test_project_recovery.py ^
  tests/test_build_info.py ^
  tests/test_main_cli.py ^
  tests/test_build_manifest.py ^
  tests/test_build_provenance.py ^
  tests/test_release_evidence.py ^
  tests/test_build_native.py ^
  tests/test_source_archive.py ^
  tests/test_portable_archive.py ^
  tests/test_source_identity.py ^
  tests/test_source_manifest.py ^
  tests/test_mesh_import_recipe.py ^
  tests/test_mesh_external_dependencies.py ^
  tests/test_mesh_verified_stream.py ^
  tests/test_mesh_admission.py ^
  tests/test_alignment_utils.py ^
  tests/test_artifact_document.py ^
  tests/test_artifact_geometry_metrics.py ^
  tests/test_artifact_surface_measurement.py ^
  tests/test_geometry_identity.py ^
  tests/test_artifact_scene_adapter.py ^
  tests/test_artifact_session.py ^
  tests/test_artifact_workbench.py ^
  tests/test_artifact_workflow_progress.py ^
  tests/test_artifact_workflow_self_test.py ^
  tests/test_artifact_exports.py ^
  tests/test_artifact_measurements.py ^
  tests/test_artifact_new_process_roundtrip.py ^
  tests/test_canonical_json.py ^
  tests/test_artifact_vector_record.py ^
  tests/test_artifact_vector_export.py ^
  tests/test_artifact_vector_extractor.py ^
  tests/test_artifact_outline_extractor.py ^
  tests/test_artifact_outline_topology.py ^
  tests/test_artifact_rubbing_extractor.py ^
  tests/test_artifact_rubbing_export.py ^
  tests/test_artifact_survey_export.py ^
  tests/test_artifact_verification.py ^
  tests/test_artifact_tile_unwrap.py ^
  tests/test_artifact_tile_unwrap_export.py ^
  tests/test_flatten_metrics.py ^
  tests/test_flattener_sectionwise.py ^
  tests/test_field_pilot.py ^
  tests/test_canonical_png.py ^
  tests/test_vector_schemas.py ^
  tests/test_rubbing_schemas.py ^
  tests/test_render_coordinates.py ^
  tests/test_viewport_render_origin.py ^
  tests/test_opengl_driver_smoke.py ^
  tests/test_rotation_convention.py ^
  tests/test_app_gui_launcher.py ^
  tests/test_gui_smoke.py
```

이 suite의 핵심 증거는 다음과 같다.

- fixed OpenGL 순서 `T @ Rx @ Ry @ Rz @ S`의 hard-coded golden point
- point·bounds·plane·inverse 변환의 동일 행렬 규약
- Align의 scale·shear·reflection·perspective·non-finite 거부
- canonical millimeter metadata, immutable append/activate, deterministic serialization
- 1 µm ties-to-even geometry metrics의 고정 표면적, exact-rational 체적, open/non-manifold/orientation/multi-component fail-closed, receipt schema·known-record reopen 검증
- source face row와 10억 분율 barycentric anchor, 전체 CPU ray/triangle hit, large-offset exact squared chord distance, PCA 평면·정규화 대수 Kasa 원의 rank/eigengap/condition/RMS·maximum residual, receipt schema·과거 Align reopen·tamper·취소 검증
- Align/metadata 전환에 따른 record freshness와 background operation context 고정
- versioned canonical JSON golden fixture와 Draft 2020-12 `ArtifactDocument 1.0` schema 검증
- raw source `(identity_scope, SHA-256, size)`와 saved parser format 재검증
- exact-key mesh import recipe, Trimesh·NumPy·Pillow parser-subset digest와 전체 lock provenance 분리, GUI ticket부터 embedded reopen까지 동일 receipt 실행
- self-contained v1 `deny_external`과 OBJ→MTL→texture, PLY `TextureFile`, glTF/GLB external buffer의 v2 `closed_manifest`; logical-path 정규화, remote/absolute/traversal/symlink 탈출·미선언/미사용/변조 resource 거부, exact lower-case `data:...;base64,`만 허용하고 모호한 `base64,` 외부 URI를 거부하며 glTF data URI 및 외부 buffer actual byte length를 declared `byteLength`에 parser 전 결속
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
- canonical document SHA-256 + 정규화 project path + `confirmed | uncertain` 내구성으로 되는 saved-snapshot checkpoint와 `durability_uncertain` save status, 새 import dirty/fully verified reopen clean, Align·record immutable 변경 dirty 파생
- complete-workflow self-test의 실제 Workbench `dirty → saved → dirty → saved` 전이와 exact report marker `checkpoint=dirty>saved>dirty>saved`; frozen/portable Windows job은 marker가 없거나 다르면 실패
- legacy payload와 embedded `ArtifactSession` project writer가 모두 production-validated same-parent staging을 Windows `MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH)`로 commit하고, 긴 한글 drive path·UNC extended path, 정확한 flag, API 실패 시 기존 목적지/temporary 정리와 no-fallback을 검증한다. package report marker `project_commit=windows-movefileex-write-through`가 frozen/portable 양쪽에서 정확한 token으로 없으면 실패한다. 비 Windows 저장 backend 테스트는 내부 회귀 시험일 뿐 제품 완료 판정이 아니다.
- `state_version`/`authority_epoch` compare-and-swap, transition 종류별 허용 변경과 expected record ID 집합 검증, 동시 candidate의 first-wins
- same-render DerivedRecord append의 `RecordBindingTransition`, exact snapshot capability 검증, live mesh/VBO/선택/cache 보존과 binding CAS rollback
- tentative authority를 observer에 노출하지 않는 prepare/activate/finalize, 정상 rollback과 rollback·scene 복원·finalize 불확실 시 fatal save/measure/export 차단
- scene 교체 뒤 cut-section/ROI worker identity·projection generation·selected object fencing, surface/visible-face worker의 target mesh·TRS·render-frame fencing과 오래된 finished callback의 다른 유물·새 worker 보호
- 서로 다른 PID에서 self-contained PLY와 textured OBJ source closure를 삭제·relocation한 뒤 saved-parser reopen하여 primary/dependency/texture/geometry hash, Align matrix와 world vertex가 같은 durable artifact round-trip
- RFC 8785 cross-language number golden과 vector payload/recipe semantic SHA-256
- canonical-mm Cutline의 exact box·multi-component·Front/Right/oblique frame, face order/winding, ambiguity fail-closed
- canonical-mm 6-view Outline의 fixed-grid projected-triangle union, concavity·hole·island 보존, face order/winding/duplicate 안정성; algorithm 1.1.0의 한 격자 closing은 격자가 만든 sliver hole·pinch만 없애고 채운 수·합친 조각 수·면적 변화를 QC로 남기며, 1.0.0 record는 closing 없이 그대로 재계산
- translated integer lattice의 `1e9 mm` survey offset, grid collapse/merge receipt, chunked balanced union과 resource-limit fail-closed
- Outline ring simple/area, hole ownership/contact, hole/exterior overlap·nesting, production recipe/frame/grid/ID 재검증
- native Outline widget/record/record-derived overlay와 screenshot/OpenCV legacy export 우회 차단
- `READY + FRESH + verified payload`만 허용하는 1:1 SVG, full sidecar-claim binding, confirmed unit/active Align/dependency closure
- SVG/sidecar tamper, XML active content/DTD, duplicate/pathological JSON, privacy allowlist, size cap, concurrent no-replace publish
- package relocation 후 원본 문서·mesh·GUI 없이 별도 PID에서 수행하는 offline 검증
- `.amr` embedded source의 saved-parser 재물질화와 `.amr-vector`/`.amr-rubbing`/`.amr-survey`/`.amr-unwrap` 자동 판별을 하나로 묶는 `--verify-artifact`; exact project 결합 성공·불일치 실패, manifest-only AMR 실패, symlink·mixed marker·tamper 거부, 절대 경로 없는 결정적 closed JSON receipt
- 실제 complete-workflow `.amr`/`.amr-survey`를 exact-project로 다시 여는 field-pilot contract, project document/survey aggregate hash에 묶인 Windows 10 build 17763+/Windows 11 client Workstation·native AMD64·64-bit AMD64 process·CPython 3.12·비호환 계층, 닫힌 OpenGL v2 성공 root·필수 check ID·두 mode·동일 runtime self-claim·24시간 receipt, 닫힌 10항목 human review·정량 scale 교차 판정, canonical self-hash/no-overwrite publication, 절대 경로·hostname·사용자명 비수집과 `authentication=none`/single-pilot scope. 테스트의 합성 review/driver fixture는 계약 검증일 뿐 실제 현장 pilot 증거가 아님
- versioned vector payload와 current vector-export 1.3 Draft 2020-12 schema, byte-preserved 1.0·1.1·1.2 schema 및 네 버전 offline runtime 검증; 1.1부터 Align/geometry/payload QC를 exact-key로 닫고 production Cutline/Outline record QC 전체와 payload에서 재계산한 Outline topology를 검증하며, 1.2는 outline algorithm 1.1.0(격자 closing)의 recipe·QC 키를 더하고 그 이전 sidecar는 closing으로 계산한 outline을 담지 못하며, 1.3은 사용자 선 굵기 preset의 정의를 더하고 그 이전 sidecar는 사용자 preset으로 그린 도면을 담지 못한다
- six-view canonical-mm front-depth raster, fixed integer µm quantization/local-mean tone mapping, coverage alpha와 multi-layer QC
- face order·winding·duplicate, hole, large survey offset, resource limit 및 Align 전환 후 late Digital Rubbing 결과 거부
- canonical GA8 PNG의 고정 chunk/DEFLATE bytes, exact `pHYs`, RFC 8785 iTXt metadata, pixel/CRC/chunk/scale tamper 거부
- `READY + FRESH + recomputed raster`만 허용하는 `.amr-rubbing` 1:1 PNG package, relocation 후 독립 PID offline 검증, no-replace publish와 privacy allowlist
- dependency-valid Cutline 3/3·Outline 6/6·Rubbing 6/6만 허용하는 `.amr-survey`, 15개 자식 재검증·canonical aggregate hash·exact project 결합, hidden tree fingerprint capability, 취소/Align/Open/destination race에서 all-or-nothing no-replace 게시와 소유 staging 정리
- hash 전 4 GiB regular primary source cap, parser 전 SHA-256 재검증 spool snapshot, OBJ/OFF/ASCII PLY·STL의 256 MiB whole-text-parser cap, OBJ/PLY/STL/OFF polygon triangulation count, binary PLY의 exact vertex→face 구조·고정 list-row 길이·16/8 property·128 MiB auxiliary data·payload EOF·parser footprint, glTF/GLB 최대 16 MiB JSON·GLB BIN chunk/buffer 길이 binding과 모든 declared buffer/중복·겹침 bufferView slice/미사용 accessor/primitive/node-instance 선언 검증, decoded vertex/triangle/array/texture 한도, Scene graph instance 선합산, sidecar fingerprint와 재읽기 payload의 SHA-256 일치, sidecar 파일별·총량 한도와 Windows physical/commit 여유를 적용하는 authoritative mesh admission
- admission receipt의 exact fixed-limit set, PLY/glTF declared parser bytes, peak 공식, decoded→sanitized triangle accounting, accepted canonical geometry SHA-256, 실제 source format·byte length·accepted array byte 하한, `.amr` 재개방과 공개 provenance 재검증
- 명시적 canonical 장축과 최대 250,000-face source selection, 자동 경계 또는 `[-180°, 180°)`의 0.000001° 고정 seam을 고정하고 자동 fallback, 1 µm grid collapse, topology·duplicate/non-manifold/orientation, 전역 positive-area UV overlap, 3-edge·area·Jacobian singular-value distortion gate(mean 7.5%·p95 15%는 두 단면 중심 정책 모두, 면 최대 25%는 `fit_per_section`에서만; 회전축 위의 `canonical_axis_origin`은 면 최대를 보고만 한다) 실패를 READY record로 승격하지 않는 `surface.tile_unwrap.v1`
- 동일 recipe 재계산과 canonical binary round-trip, source vertex/face correspondence, exact µm bounds·component hash·전체 payload SHA-256, Top/Bottom·자동/고정 seam 구분과 Align stale 보존. current 1.3 recipe/1.4 export와 byte-preserved 1.1·1.2·1.3 recipe/export를 모두 버전 그대로 재계산·offline 검증
- `.amr-unwrap` canonical binary/flat OBJ/physical-mm SVG/provenance sidecar의 byte 결정성, four-member tamper 거부, offline 검증, hidden staging·exact prepared inode/fingerprint capability·no-overwrite final-authority publish와 destination race 정리
- versioned tile unwrap receipt 1.1/current export 1.4 Draft 2020-12 schema, byte-preserved export 1.1·1.2·1.3과 axis·seam·fallback·privacy closed contract; 1.4 이전 sidecar는 면 최대 distortion 25%를 넘는 record를 담지 못한다
- 회전축 정치 문서에서만 자르고, 법선의 바깥 방향과 두 겹의 반지름 대소가 어긋나면(뒤집힌 메쉬) 안쪽 벽을 내주는 대신 거부하며, 방향이 뒤섞이거나 복제된 면이 있거나 여러 조각으로 갈린 띠는 각 조각 크기를 알리고 멈추는 외면 띠 선택
- `READY + FRESH` 전개 record를 payload 해시·recipe 해시로 이름 짓고, 재계산한 전개가 receipt·payload 해시와 다르거나 전개가 STALE이면 그리지 않는 `raster.developed_rubbing.v1`; 전개 record를 항상 dependency로 갖고, 같은 recipe에서 raster SHA-256이 같으며, Digital Rubbing과 같은 raster/pixel/dimension 상한과 admission 예산을 공유
- `.amr-rubbing` sidecar 1.3.0: `recipe.kind`가 receipt 종류(여섯 뷰 / 전개)와 provenance record type을 강제하고, 전개 receipt는 1.2.0 이상, 종이 기저 농담은 1.3.0에만 허용되며, 1.0.0·1.1.0·1.2.0 schema bytes는 그대로 보존
- exact Git commit/tree의 regular blob만 허용하는 결정적 corresponding-source ZIP, 내부 manifest·외부 sidecar·Git blob ID·SHA-256·라이선스 결합과 저장소 삭제 뒤 offline 검증
- portable/source/evidence exact hash와 실제 payload file set, GitHub repository/workflow/run attempt/Windows X64 hosted-runner identity를 결합하는 canonical unsigned provenance, 한글 추출본 offline 재검증, closed `authentication=none` 계약과 변조·source mismatch 거부
- vector/rubbing/survey/tile-unwrap worker의 hidden same-parent staging·전체 검증·prepared inode/fingerprint capability, 빠른 final Workbench record-authority fence, same-Align append 허용, Align/Open stale 정리와 pending Open GUI 취소 정책
- 고정 길이 staging UUID 충돌·quarantine/foreign inode 보존, exact result/prepared capability 위조·사전 목적지 이동·destination race 차단, post-rename 실제/미지원 directory-fsync `committed` 내구성 경고
- Qt-free Cutline/Outline/Digital Rubbing/기와 전개/검증 제원·표면 anchor 거리/지름 work item, exact result capability, same-Align 병렬 rebase, Workbench 공유 record reservation, Align/Open stale·취소·rollback 방어와 pending Open 게시 재시도
- core 고유 `RuntimeError` 취소 신호, extractor의 명시적 Python loop/chunk polling과 최종 결과 fence, 기와 section circle/seam fitting·row-shift section/grid/refinement·export recipe 재계산 probe, false-probe payload/raster/QC 동일성, controller의 `FAILED > STALE > CANCELLED` 경합 우선순위·`CANCELLING → CANCELLED` slot 보존, GUI one-shot 취소 요청·대기 창 수명·무경고 종료. 현재 실행 중인 단일 NumPy·GEOS·선형대수 호출은 반환 전 선점 취소 대상이 아님
- Digital Rubbing 누적 peak-memory admission, UV/texture materialize 복사비의 무복사 사전 차단, controller별 한도 우회 방어, 실행 exactly-once, 취소 worker 종료 전 slot 보존
- 재개방 프로젝트의 READY + FRESH vector/rubbing/tile-unwrap 명시 선택, background recipe 재계산과 완료 시 document/Align/record 재검증, active raster 예산 중첩 차단, 일시 게시 실패 재시도 queue와 보류 중 저장 차단
- native Align commit/parent activation handler의 GUI-thread source geometry hash·canonical materialization 부재, 잠긴 non-cancelable worker, 완료 시 exact object/mesh/binding/preview와 session/state/epoch/project-path fence, stale 결과의 publication 전 폐기, OpenGL VBO/scene publication의 GUI-context 유지
- native Save handler의 GUI-thread canonical materialization·project writer·Git subprocess 부재, worker preflight 실패 시 writer 미호출, 같은 경로 Save/Save As exact snapshot CAS와 confirmed checkpoint 채택, 저장 중 authority/project-path 변경 또는 final CAS 경합 시 과거 snapshot 경고·dirty 유지·현재 Save As 상태 보존, 내부 비 Windows writer의 committed directory-fsync 경고에 대한 durability-uncertain checkpoint 회귀 시험, non-cancelable 진행 dialog의 worker 종료 전 close/Escape 차단과 종료 시 bounded join
- Windows native Close/새 source Open/Project Open/drag-and-drop의 `Save / Discard / Cancel`, Save 선택 시 exact durable checkpoint 후에만 비동기 후속 동작 재개, Save As 취소·writer/CAS 실패·stale·durability-uncertain에서 현재 문서/창 보존, document와 GUI-only transient를 함께 반영한 title `*`·status label, 실제 원본 loader/Project inspector QThread의 request·ticket 회수와 bounded join·late callback 차단
- interrupted save-temp의 exact basename·regular-file·64개 bound discovery, candidate device/inode/size/mtime 고정, embedded source 삭제 뒤 production materialization, manifest-only·깨진·교체된 후보 거부, 기존/racing 목적지 no-overwrite, 후보·기존 intended destination 보존, directory-fsync 내구성 경고와 GUI의 별도 Open 확인
- versioned Digital Rubbing receipt/export Draft 2020-12 schemas

## Windows 검증 기준선

아래 표는 2026-07-12 M0-6 당시의 Windows 기준선이다. 최신 Windows source·portable 기준선은 표 아래에 기록한다.

| 검사 | 결과 |
|---|---:|
| GitHub Actions Python 3.12.13 quality `python -m pytest -q` | 572 passed, 117 subtests passed |
| Windows persistence smoke | 498 passed, 3 platform-specific skips, 117 subtests passed |
| GitHub Actions `python -m ruff check .` | passed |
| GitHub Actions M0 Pyright wrapper command | 0 errors |
| ArtifactDocument + vector/rubbing/survey/tile-unwrap payload/export + offline/field-pilot receipt Draft 2020-12 schemas + golden | passed |

2026-07-13 source-closure 기준 commit `6898f98d2fb3`은 Python 3.12 원격 CI에서 full pytest `660 passed, 128 subtests passed`, Ruff, M0 Pyright, Windows persistence와 Windows frozen executable self-test를 통과했다. 제품 완료 판정에는 Windows job만 사용한다.

통합 verifier 기능 기준 commit `4a21666e7f7b`는 [source CI run 29280873586](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29280873586)에서 full pytest `763 passed, 128 subtests passed`, Ruff, M0 Pyright `0 errors`, Windows workflow `661 passed, 5 skipped, 118 subtests passed`와 qwindows+llvmpipe actual-frame `66/66`을 통과했다. [portable package run 29280874076](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29280874076)도 frozen·한글 경로 portable 실행 파일의 14-check complete-workflow self-test, outbound deny 상태의 public `--verify-artifact --report` exit/receipt/privacy gate, actual-frame `66/66`, 추출본·방화벽 규칙 정리를 모두 통과했다. complete-workflow check는 Open→Align→3/6/6→completed AMR offline reopen→1:1 SVG/PNG 재현과 exact-project 결합을 한 번에 검증한다. 과거 commit `19558f324deb`의 installer run은 역사적 내부 검증이며 현재 배포 gate는 compiler 비종속 portable ZIP으로 교체했다.

중단 저장 복구 기준 commit `546d106c6ccf`는 [source CI run 29282751462](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29282751462)에서 full pytest `777 passed, 128 subtests passed`, Ruff, M0 Pyright `0 errors`, Windows workflow `675 passed, 5 skipped, 118 subtests passed`와 qwindows+llvmpipe actual-frame `66/66`을 통과했다. 이 Windows run은 directory enumeration의 placeholder `st_dev/st_ino`를 실제 path stat identity와 혼용하지 않는 회귀 경계, candidate/staging inode 변경, invalid·manifest-only package, 기존·racing 목적지와 directory-fsync 경고를 직접 실행한다. [portable package run 29282751606](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29282751606)은 frozen 및 outbound-denied 한글 경로 portable 실행 파일에서 `recovery=verified-create-new`가 포함된 14-check complete-workflow, public verifier privacy gate와 actual-frame `66/66`, 추출본·방화벽 규칙 정리를 통과했다. 해당 run의 corresponding-source archive SHA-256은 `40f4f1919537851d1ba5d935074b93d7504b4b201fce1f6c2a8a9c8c48271873`, release-evidence payload는 `0582d30600cf1bda09f1e47ab781abaeca68c9e678c017397d7e8d1f355bc1d7`, portable archive는 `6f0ae53df756c3974c5973e516158ecae2ea60017e8412a6d908fadb641ba444`, portable payload는 `24f09da6e806c9aafed8663c99398181ef0a755958d66e85ad3213d38f778ec6`, unsigned provenance는 `9181ed72dbcf03cc1b9732976898e61da201f95d02abb04c44d0cb23ea55d6f7`이다.

## 아직 차단하지 않는 검사

전체 트리 Pyright는 아직 통과하지 않는다. CI에서는 이 결과를 `continue-on-error`로 보고하여 부채가 보이게 하되, M0 범위를 넘는 기존 오류 때문에 모든 변경을 막지는 않는다. 신뢰 커널 전환이 진행될 때마다 차단 범위를 넓힌다. 독립 프로세스 테스트의 worker program은 Python 문자열이므로 Pyright가 문자열 내부를 분석하지는 않지만, 차단 pytest가 두 문자열을 각각 새 interpreter에서 실제 실행한다.

Windows workflow smoke에서는 프로젝트 저장, exact saved-snapshot checkpoint와 dirty/clean/durability-uncertain 파생, Close/Open/Project Open/drag-and-drop의 `Save / Discard / Cancel` 문서 보존 gate, 중단 save-temp의 identity-pinned `MoveFileExW` write-through create-new 복구와 목적지 경합·실패 시 no-fallback 정리, source/geometry identity와 versioned source manifest/bundle, ArtifactDocument·scene adapter·session·application workbench와 record-derived workflow progress, ticketed Open과 explicit Align gate, RFC 8785/vector record/export/Cutline/Outline/topology/schema, Digital Rubbing record/extractor/canonical PNG/export/schema, atomic survey export, field-pilot review/report/verification schema와 CLI routing, authoritative tile-unwrap record/desktop panel/staged export, 독립 프로세스 source closure 및 relocated vector/rubbing/survey/tile-unwrap package 왕복, render-coordinate algebra·relative VBO/native preview smoke, matrix golden, GUI 런처, MainWindow 생성, native source-of-truth binding, native Cutline/Outline/Rubbing/기와 전개 command, 3/6/6 순차 gate와 completed `.amr` offline reopen, session/version/epoch 및 late-result 방어, legacy export 우회와 fail-closed 경계를 실행한다. 별도 Windows quality job도 전체 테스트와 Ruff·M0 Pyright를 실행한다. Windows GUI 스모크는 `QT_QPA_PLATFORM=offscreen`이므로 CPU/document/scene transaction과 widget wiring을 검증하지만 실제 OpenGL frame을 증명하지 않는다.

같은 Windows job의 다음 단계는 `QT_QPA_PLATFORM=windows`, `QT_OPENGL=software`로 실제 `QOpenGLWidget` context를 열고 `src.gui.opengl_driver_smoke` v2 report를 검증한다. Qt wheel에 포함된 `opengl32sw.dll`을 강제하고 PyOpenGL의 GL/WGL dispatch도 같은 DLL에 결합한다. 이렇게 해야 Qt의 software context와 시스템 `opengl32.dll`을 섞어 호출하지 않는다. qwindows/context/768×768 FBO/VBO/pixel/depth/pick 경계를 실행하며, 이는 software OpenGL 실제-frame 회귀 증거이지 대표 GPU 또는 compositor 최종 presentation 인증은 아니다. GitHub Windows Server에서도 이 일반 smoke는 계속 실행하지만, v2에 저장된 Server product type 때문에 실제 client machine claim과 일치하지 않으며 field-pilot `verified` 근거로 승격되지 않는다.

별도 `package-smoke.yml`은 `main` push, pull request, 수동 실행에서 Windows x64/CPython 3.12의 hash-locked binary wheel set, immutable build manifest, PyInstaller spec과 frozen executable의 file-report self-test를 실행한다. native build 진입점은 `IsWow64Process2`의 native AMD64 host, canonical Git top-level, repository override 환경 부재, requested commit과 HEAD 일치, HEAD의 모든 regular blob path·mode와 stage-0 index 일치, Git EOL clean representation으로 계산한 live worktree blob ID 일치를 manifest 생성 전과 PyInstaller 직후에 강제한다. 이 검사는 index의 `assume-unchanged`/`skip-worktree` hint를 신뢰하지 않으며 외부 content filter와 frozen input 안의 ignored 주입도 거부한다. PyInstaller 뒤 exact Git commit/tree/blob의 corresponding-source ZIP을 payload에 넣고, 실제 payload 전체 파일의 path/size/SHA-256 manifest, runtime 10개를 exact wheel SHA-256에 묶은 SPDX 2.3 SBOM, wheel 메타데이터와 라이선스 원문에서 만든 machine/human NOTICE를 생성한다. evidence index가 이 네 문서를 묶고 `release_evidence`와 `source_archive`를 포함한 14-check가 frozen 및 portable payload에서 모두 다시 계산된다. complete-workflow check는 새 import·Save·record append·same-path Save의 checkpoint 전이를 직접 실행하고 report의 `checkpoint=dirty>saved>dirty>saved`, 실제 Windows writer backend `project_commit=windows-movefileex-write-through`, create-new 복구 backend `recovery_commit=windows-movefileex-write-through-noreplace`를 쉼표로 분리한 정확한 token으로 강제한다. 이어 작은 PLY를 실제 application authority로 열고 3/6/6 record를 만든 뒤 completed `.amr`와 같은 bytes의 interrupted temp를 발견·create-new 복구하고, 외부 원본 없이 복구본을 재열어 Cutline/Outline 1:1 SVG 9개와 Digital Rubbing 1:1 PNG 6개를 모두 재현·이동한다. 이어 같은 15개 record를 한 `.amr-survey`로 원자 게시·이동하고, 통합 verifier로 AMR materialization, 원본 SHA-256·recipe·QC·scale·aggregate hash와 각 export의 exact-project 결합을 검증한다. 별도의 비틀린 원통형 기와 fixture도 record → 외부 원본 삭제 → `.amr` reopen → 현재 1.2 recipe 재계산 → `.amr-unwrap` relocation과 동일 payload SHA-256, 13개 station·최대 6,364 µm row-shift를 검증하며 legacy 자동 seam 1.1 recipe/package도 별도 호환 테스트로 고정한다. 같은 pair를 field-pilot builder에 넣어 artifact는 통과하지만 실제 human/driver evidence가 없으므로 반드시 `pilot=artifact-pass-human-driver-pending`으로 남는지도 검사한다. frozen과 한글 경로 portable report는 이 receipt와 `exports=vector 9/9>rubbing 6/6>unwrap 1/1`, `unwrap=record 1/1>reopen 1/1>export 1/1>hash-match>row-shift `, `survey=verified-atomic-15`가 없으면 실패한다. 원래 project와 temp가 유지되는지도 함께 확인한다. 이어 같은 frozen executable의 `--opengl-driver-smoke-report`를 native qwindows/software OpenGL로 실행한다. 그 뒤 표준 라이브러리로 portable ZIP과 canonical sidecar를 만들고 모든 entry·release evidence·source archive를 검증하며, 외부 unsigned provenance에 이 artifact들과 GitHub workflow/run/runner identity를 결합한다. `문화유산 기록` 한글 경로 추출본도 같은 provenance와 일치해야 한다. 추출 실행 파일은 outbound 차단 상태의 14-check, public `--verify-artifact ... --report ...` exit/receipt/privacy 경계와 actual-frame을 다시 통과하고 디렉터리 삭제 뒤 잔여물이 없어야 한다. provenance는 `authentication=none`인 무결성 기록이며 서명된 출처 인증이 아니다. 라이선스·서명 게이트가 해결되기 전에는 artifact upload와 release 단계를 두지 않는다. 이 패키지 게이트와 현재 완료 증거는 Windows x64만 대상으로 하며, 권리·서명·대표 GPU·실제 유물/고고학자 pilot 완료를 뜻하지 않는다.

이 기와 증거의 비틀린 원통형 fixture는 합성 데이터이며 실물 기와 pilot의 대체 증거가 아니다.

과거 [진단 run 29254942224](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29254942224)은 비 ASCII installer 경로에서 software OpenGL context 생성에 실패했다. 이 실패를 성공으로 재분류하지 않고, 현재 portable gate가 한글 payload·report 경로의 offline workflow와 software OpenGL을 모두 차단 조건으로 직접 재검증하도록 승격했다. 대표 하드웨어 GPU/driver의 비 ASCII 경로는 여전히 실제 pilot 범위다.

AMR v2 `payload_type="artifact_document"` 1.0의 strict 저장·content-addressed source closure embedding·production-loader staged reopen·checksum·원자 교체와 독립 프로세스 materialization은 현재 차단 게이트다. `tests/test_artifact_new_process_roundtrip.py`는 프로세스 A와 B의 PID가 다름을 확인하고, 프로세스 A가 `.amr`를 저장한 뒤 외부 PLY 또는 textured OBJ의 OBJ·MTL·PNG 전체를 삭제하고 package를 relocation한다. 프로세스 B는 `.amr`의 embedded source closure만 saved parser/unit으로 decode하며, 새로 계산한 primary/dependency SHA-256·크기와 texture/geometry SHA-256, active Align ID·matrix, parser/unit, world vertices가 같아야 통과한다.

Native GUI의 한-artifact Open/Align commit/save/load는 `ArtifactWorkbench.snapshot.session.document`를 source of truth로 사용하며 MainWindow의 session field는 이행 중 compatibility mirror다. Open과 Align은 ticket/CAS/two-phase publication을 사용한다. Align commit/parent activation handler는 GUI에서 가벼운 scene/preview guard만 캡처하고 source hash·candidate session·canonical materialization을 잠긴 worker에서 준비한다. callback은 exact object/mesh/binding/preview와 Workbench session/state/epoch/project path를 다시 확인한 뒤에만 GUI OpenGL context에서 VBO와 scene을 게시한다. Cutline/Outline/Digital Rubbing/기와 전개는 Qt-free `ArtifactMeasurementController`가 recipe/context와 record ID를 Workbench 단위로 예약하고 worker computation만 받은 뒤 current same-Align session에 rebase한다. append-only record publication은 live SceneObject의 document binding만 CAS하고 mesh/VBO를 재생성하지 않는다. `ArtifactExportController`는 vector/rubbing/tile-unwrap package를, `ArtifactSurveyExportController`는 완료 15개 record 묶음을 worker에서 생성·재계산·전체 검증하고 prepared capability까지 발급한다. GUI dispatcher의 final Workbench fence에서는 빠른 identity/fingerprint 확인과 rename만 실행한다. command handler는 단일 `TaskThread`를 사용하며 늦은 finished signal이 새 worker/dialog를 지우지 못한다. 재개방 기록은 자동 최신 fallback 없이 명시적으로 선택하고 request token으로 늦은 preview가 최신 선택을 지우지 못하게 한다. rollback 가능한 측정 게시 실패는 exact result를 재시도 queue에 보존한다. native Save는 immutable session을 캡처한 뒤 projection/geometry 비교와 source closure 재해시·ZIP/fsync·production reopen/materialization을 잠긴 worker dialog에서 수행한다. 완료 시 exact session/state/epoch/기존 project path가 유지될 때만 현재 경로와 canonical document SHA-256/path/durability checkpoint, migration flag를 갱신한다. stale·실패·취소·durability-uncertain은 dirty를 해제하지 않으며 Close/Open/Project Open/drag-and-drop의 Save 후속 명령을 허가하지 않는다. active/보류 실측과 아직 `DerivedRecord`로 이식되지 않은 선택·기록면·평가 결과는 누락 저장하지 않고 fail closed한다. authority rollback·scene 복원·finalize가 불확실한 fatal 상태에서는 검증된 Open 전까지 저장·실측·내보내기를 모두 막고 task-local 오류가 재열기 배너를 덮지 못한다. 별도의 legacy destructive bake 이후 Save 성공은 native Align 복원 증거가 아니다. legacy runtime은 그런 vertex mutation을 `_amr_has_unpersisted_bake`로 표시하고 snapshot 저장을 차단한다.

Native measurement 취소는 `QThread.terminate()`를 사용하지 않는다. 사용자 요청은 먼저 record 게시 권위를 회수하고, Cutline face/path, Outline polygon/union/topology, Digital Rubbing face/row/integral/relief, 기와 section circle/seam fitting·row-shift section/grid/refinement·face 검사의 다음 명시적 안전 경계에서 worker가 고유 core 취소 예외로 종료한다. tile-unwrap export의 recipe 재계산도 같은 live Event probe를 받는다. vector/rubbing/survey/tile-unwrap export는 공통 취소 UI에서 publication 권위를 먼저 회수하고 worker가 이미 만든 owned staging을 quarantine 경로로 안전하게 정리한다. 앱 종료는 active authoritative task에 이 취소를 one-shot으로 전달하고 최대 30초 join을 기다리며, 완료를 증명하지 못하면 창을 닫지 않는다. join 성공 뒤 task signal과 identity를 제거하므로 늦은 완료 콜백은 게시될 수 없다. 이 종료/timeout/late-callback 경계는 GUI 차단 테스트에 포함한다. native Align commit/parent activation에서 GUI thread source hash·materialization이 0회인지와 preview/authority가 바뀐 worker 결과가 게시 전에 폐기되는지, native Cutline/Outline/Digital Rubbing/기와 전개 handler가 GUI thread에서 canonical scene materialization을 하지 않는지, worker preflight 실패가 terminal 상태와 record 예약 해제로 끝나는지, Rubbing 전체 resource estimate 도중 취소가 worker 종료 전까지 memory slot을 유지하는지를 차단 테스트로 검증한다. 현재 실행 중인 단일 NumPy·GEOS·선형대수 호출, Align source hash/materialization과 measurement preflight 내부의 단일 materialization·resource-estimate 호출은 호출이 반환할 때까지 선점할 수 없으므로 “즉시 종료”로 표현하지 않는다. native heavy GUI-thread preflight 제거는 완료했으며 legacy mesh/profile/slice worker 통합은 후속 항목이다.

### 대좌표 render-origin: 실제 source driver 게이트 추가

현재 차단 suite는 absolute float64 world-mm document/mesh 불변, `>= 1e9 mm` offset의 mm/sub-mm feature, CPU float64 subtraction 뒤 객체별 relative float32 VBO payload, 안정적인 scene origin에 대한 camera/model affine rebasing, 활성 world overlay의 relative 제출, absolute float64 CPU face 계산과 render origin 비직렬화를 검증한다. 또한 exact modelview·projection·viewport·scene origin과 depth-affecting scene signature를 같이 게시한 frame authority의 project/unproject/ray, depth pick·Ctrl drag 수명주기, 변환·resize·scene rollback·repaint 전 상태 변경 후 stale frame 거부를 검증한다. pure coordinate helper와 해당 테스트는 M0 Pyright에, mocked OpenGL upload·overlay·interaction 테스트는 Windows workflow smoke에 포함한다.

`src/gui/opengl_driver_smoke.py`는 OpenGL 2.1 compatibility·24-bit depth를 QApplication 전에 요청하고 native QPA의 실제 `Viewport3D` widget FBO를 렌더·readback한다. Windows에서는 768×768 고정 크기의 비활성 native tool window를 노출하고, 자동 desktop이 paint event를 합치더라도 `makeCurrent()`가 결합한 widget default FBO에서 production `paintGL()`을 명시적으로 실행한다. compositor의 최종 on-screen presentation을 검증하는 테스트는 아니다. `[1e9, -2e9, 3e9] mm` 기준점에 0.25 mm 간격으로 분리된 두 판과 0.125 mm 높이차, 0.25 mm native vector overlay를 만든다. 실제 production `add_mesh_object → update_vbo → paintGL → RenderFrameSnapshot → glReadPixels → pick_point_on_mesh_info`를 통과하며 다음을 fail closed로 확인한다.

- GL vendor/renderer/version, current widget context, depth-preserving `PartialUpdate`, complete default FBO, depth bits
- 별도 Qt FBO의 정확한 RGBA/depth clear-readback
- driver VBO object와 작은 relative float32 payload; 원본 absolute float64 vertex 불변
- 원근/상면 정사영 각각의 두 depth component, plate pixel, 빈 gap, relative overlay pixel
- 보정 검색을 끈 실제 depth pick, 같은 frame serial, 해석적 ray-plane oracle와 0.125 mm 높이차

2026-07-13 Windows 대상 commit `b12d4874a4a8`의 [source CI run 29251668123](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29251668123)은 qwindows + bundled llvmpipe에서 66/66 조건을 통과했다. 768×768 FBO에서 원근 pick 높이차 `0.124599 mm`·최대 ray-plane 오차 `0.002970 mm`, 정사영 높이차 `0.124998 mm`·최대 오차 `0.00000381 mm`를 기록했다. 같은 commit의 [frozen package run 29251668029](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29251668029)도 동일 gate를 통과했다. JSON은 tested commit/tree 상태, runtime-lock SHA-256, dependency version과 UTC 시각을 포함한다. Windows software rasterizer 결과도 대표 하드웨어 GPU/driver 또는 compositor presentation을 대신하지 않는다. render origin은 metadata·Align·record·QC·hash·export 권위 값에 저장하지 않는다.

## 게이트 변경 원칙

- 검사를 삭제하거나 `continue-on-error`로 바꾸는 것은 별도 근거와 리뷰가 필요하다.
- fallback, sampling, 단위 추정, 원본 불일치를 성공으로 숨기는 테스트를 추가하지 않는다.
- Windows source·frozen·한글 경로 portable actual-frame 통과는 software-renderer와 archive 역학의 완료 증거로만 표현한다. 라이선스·서명·대표 GPU·실물 대용량 pilot 전에는 “Windows 안정판 검증 완료”라고 표현하지 않는다. 비 Windows 결과는 제품 지원·배포 판정에 사용하지 않는다.
