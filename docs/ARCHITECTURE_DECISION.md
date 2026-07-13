# Architecture Decision: 신뢰 코어 보존, 애플리케이션 셸 점진 교체

- 상태: Accepted
- 결정일: 2026-07-12
- 적용 범위: 첫 공개 안정판까지의 구조 개편

## 결정

ArchMeshRubbing을 전면 재작성하지 않는다. 검증 가능한 headless 측정 코어와 포맷 계약은 유지하고, 현재의 거대한 PyQt GUI·OpenGL 상태 계층과 legacy export 경로를 새 애플리케이션 셸로 단계적으로 교체한다.

이 방식은 흔히 strangler migration 또는 branch-by-abstraction으로 부른다. 사용자가 쓰는 한 단계씩 새 경로로 옮기고, 같은 기능의 legacy 진입점을 차단한 뒤 제거한다. 새 셸이 완성될 때까지 기존 프로그램을 실행 가능한 비교 기준과 전문 기와 워크플로우로 남긴다.

## 근거

코드베이스에는 버릴 부분과 지킬 부분이 명확히 함께 존재한다.

- `app_interactive.py`는 약 16,800줄, `src/gui/viewport_3d.py`는 약 15,600줄이며 각각 수백 개 메서드와 많은 광범위 예외 처리를 포함한다. UI, 상태, 렌더링, 작업 실행, 저장 정책이 서로 강하게 얽혀 있어 이 두 파일을 계속 확장하는 비용은 높다.
- 반면 primary source identity, versioned parser source closure, 명시적 단위, immutable Align revision, canonical JSON/PNG, Cutline, Outline, Digital Rubbing, offline export는 GUI와 분리된 코어와 버전 스키마를 가진다.
- Windows 대상 commit `b12d4874a4a8`의 source CI에서 전체 `670 passed, 128 subtests passed`, M0 Pyright `0 errors`, Ruff와 Windows workflow·native actual-frame gate가 통과해 현재 계약을 보호한다.
- 같은 commit의 unsigned Windows frozen 앱은 당시 offline complete workflow와 qwindows+llvmpipe actual-frame gate를 통과했다. 이후 installer 실험도 내부 설치·제거를 통과했지만 상용 compiler 정책 때문에 현재 배포 계약에서 제거했다. 두 결과 모두 서명·대표 하드웨어 GPU 또는 공개 배포 증거는 아니다.

전면 재작성은 이 검증 자산과 오래 축적된 기와 처리 알고리즘까지 동시에 다시 만들게 한다. 반대로 기존 GUI에 기능을 계속 덧붙이면 mutable scene 상태와 권위 기록이 다시 섞인다. 따라서 코어는 보존하고 셸은 교체하는 경계가 가장 안전하다.

## 보존할 것과 교체할 것

| 보존·강화 | 점진 교체·격리 |
|---|---|
| `ArtifactDocument`, `ArtifactSession`, `ArtifactWorkbench`, measurement/export controller, source manifest/bundle과 source/geometry identity | `app_interactive.py`의 단일 `MainWindow` 오케스트레이션 |
| canonical-mm Align/Cutline/Outline/Rubbing 계산 | `viewport_3d.py`의 데이터 소유·도구 상태·렌더링 혼합 |
| RFC 8785 JSON, canonical PNG, versioned schemas | mutable `legacy_ui_state`와 암묵적 작업 완료 상태 |
| `.amr`, `.amr-vector`, `.amr-rubbing` 검증·원자 저장 | screenshot/OpenCV/convex-hull을 실측 산출물로 쓰는 우회 경로 |
| 테스트 fixture, golden hash, offline validator | destructive bake와 저장 가능한 원본 변형 |
| 기와형 flatten 알고리즘과 검토용 legacy 기능 | 플랫폼별 수동 빌드·바로가기·상용 installer compiler 의존 |

Legacy 기능은 즉시 삭제하지 않는다. 연구 검토용이면 명확히 `legacy review`로 표시하고, 1:1 측정 결과처럼 보이지 않게 분리한다. 새 포맷으로 안전하게 승격할 수 없는 상태는 추정 변환하지 않고 fail closed한다.

Source 경계도 같은 원칙을 따른다. `SourceAsset`은 고고학적 권위 원본인 primary mesh identity를 유지하고, MTL·texture·buffer는 `GeometryRevision.import_recipe`의 versioned parser input closure로 기록한다. 새 import가 외부 resource를 읽지 않으면 v1 `deny_external`, source root 안의 상대 resource를 실제로 읽으면 v2 `closed_manifest`로 확정한다. `.amr`은 두 profile 모두 content-addressed bytes로 운반하고 reopen 때 filesystem을 탐색하지 않는다. host path, remote URI, root/symlink 탈출은 신뢰 코어 밖의 편의 기능으로도 fallback하지 않는다.

## 새 셸의 경계

애플리케이션 셸은 최소한 아래 계층을 분리해야 한다.

1. `Application state`: 현재 document/session과 project path, 단일 pending Open, 활성 revision, 작업 가능 여부만 소유한다.
2. `Commands`: Open, Align commit, Cutline/Outline/Rubbing compute, export를 명시적 입력·결과로 실행한다.
3. `Viewport adapter`: document에서 파생된 표시용 scene만 만들며 권위 geometry를 소유하거나 bake하지 않는다.
4. `Workflow panels`: command를 호출하고 record 상태를 표시하되 계산·저장을 직접 수행하지 않는다.
5. `Persistence/export`: GUI와 독립된 코어 API만 사용한다.

Viewport adapter의 대좌표 표시 경계는 권위 geometry를 recenter하지 않고 두 transient origin으로 분리한다. 객체별 `O`는 relative float32 VBO encoding에만, scene별 `R`은 camera/model render frame에만 사용한다. 두 값은 document/session/record/export authority 밖에 있으며 same-render record append에서도 scene/VBO와 함께 유지된다. main mesh·native vector preview·cutline·ROI·pick·gizmo 등 활성 world overlay는 CPU float64에서 `R`을 뺀 뒤 GPU에 제출한다. depth pick·screen projection·Ctrl drag은 해당 depth frame의 modelview·projection·viewport·`R`과 visibility·ROI·X-ray·object TRS/geometry revision을 read-only frame authority로 함께 게시한다. 상태가 먼저 바뀌고 repaint가 뒤따라오는 구간은 stale frame으로 fail closed한다. 실제 `Viewport3D` source와 Windows frozen executable의 survey-scale VBO/pixel/depth/pick은 qwindows+bundled llvmpipe에서 검증했다. 대표 Windows GPU와 scene-swap 프레임 원자성 증거는 남아 있으므로 renderer 교체 전체가 완료된 것으로 판정하지 않는다. 과거 macOS/Linux 결과는 현재 지원 판정에 사용하지 않는다.

GUI 버튼의 초록색 완료 표시는 위젯 내부 boolean이 아니라 `READY + FRESH` record에서 파생해야 한다. Align이 바뀌면 후속 기록을 물리적으로 삭제하기보다 immutable history에 남기고 stale로 판정하여 현재 산출물로 사용하지 못하게 한다. 이 결정은 `src/application/artifact_workflow_progress.py`의 Qt/OpenGL-free 모델과 Cutline 3/3 → Outline 6/6 → Digital Rubbing 6/6 gate로 구현했다. application command도 같은 순서를 강제하며 Outline record는 Cutline 3면, Digital Rubbing record는 dependency-valid Outline 6면을 직접 의존해야 완료 증거가 된다. 프로젝트 재열기와 이전 Align 재활성화도 별도 UI 상태 없이 같은 record graph에서 진행도를 복원한다.

현재 `src/application/artifact_workbench.py`는 Qt·OpenGL 없이 session/project authority, ticketed Open, `state_version`/`authority_epoch` compare-and-swap, 명시적 Align readiness와 two-phase projection publication을 소유한다. Open/new import/project reopen과 Align commit/parent activation은 이 경계를 사용한다. Open이 만드는 `initial_identity`는 materialization baseline일 뿐 사용자 확정이 아니며, 변화량이 0인 첫 Align도 immutable child revision으로 남겨야 측정 단계가 열린다.

Align commit/parent activation의 GUI command는 현재 session·scene object·projection binding·preview TRS/pivot와 Workbench version만 작은 capture로 고정한다. candidate `ArtifactSession` 생성에 포함되는 source geometry 재해시와 canonical projection materialization은 잠긴 `TaskThread`에서 수행한다. 완료 callback은 object/selection/mesh/binding/preview, GUI project flags와 Workbench session/state version/authority epoch를 모두 다시 비교하고, 달라진 결과는 scene publication 전에 폐기한다. GPU VBO 준비와 `prepare → activate → scene swap → finalize` publication은 OpenGL context를 소유한 GUI thread에 남긴다. 따라서 CPU 준비로 event loop를 막지 않으면서 GPU 자원 수명과 authority transaction을 섞지 않는다.

DerivedRecord session update는 기존 source/metadata/Align/record를 바꿀 수 없고 추가할 record ID 집합을 명시해야 한다. 같은 render projection이면 별도 `RecordBindingTransition`이 live object의 document binding만 compare-and-swap하므로 mesh materialization, VBO upload, scene swap과 transient UI 상태 초기화를 반복하지 않는다. projection scene swap 실패는 이전 권위로 rollback하며, rollback·scene 복원·finalize 자체의 성공을 증명할 수 없으면 fatal authority 상태로 전환해 검증된 Open 전까지 저장·실측·내보내기를 차단한다. viewport의 transient cut-section/ROI callback은 current worker identity, projection generation과 selected object로 fence한다. surface/visible-face worker는 여기에 target mesh, object TRS와 render-frame 행렬까지 다시 비교하여 이전 scene·다른 유물·이전 시점의 face ID가 현재 선택을 건드리지 못하게 한다. `ArtifactMeasurementController`는 Cutline/Outline/Digital Rubbing recipe와 projection context, 예약 record ID를 immutable work item으로 캡처한다. GUI handler는 projection binding·TRS·transient mutation 같은 가벼운 guard만 동기적으로 확인하고, source materialization과 vertex/face exact comparison은 controller의 worker preflight로 넘긴다. Rubbing begin은 geometry·UV·texture 복사를 포함하는 보수적 최소 메모리만 예약하며, worker가 해상도별 전체 peak estimate를 계산한 뒤 공유 admission lock 아래에서 예약을 원자적으로 확장한다. preflight 실패·취소도 computation과 같은 terminal 상태 머신을 지나므로 record와 memory 예약이 남지 않는다. 같은 Workbench의 여러 controller는 operation/record reservation과 Rubbing admission을 공유하며 모든 active owner 중 가장 엄격한 메모리·동시 작업 한도를 지킨다. worker는 computation만 반환하고 exact result capability 검증 뒤 현재 same-Align session에 rebase한다. Align/Open finalize는 진행 중 결과를 영구 stale 처리하지만 pending Open 자체는 기존 권위를 바꾸지 않으므로 결과 capability를 재시도 가능하게 보존한다. 취소 중인 Rubbing은 worker가 실제 종료될 때까지 메모리 예약을 유지한다.

`ArtifactExportController`는 1:1 vector/rubbing/tile-unwrap package를 별도 authority effect로 취급한다. worker는 목적지와 같은 parent에 숨김 staging을 만들고 전체 package를 검증한 뒤 destination·parent·device/inode·entry fingerprint에 묶인 일회성 prepared capability를 반환한다. final dispatcher는 captured source·render projection·exact `READY + FRESH` record를 현재 Workbench에서 다시 확인한 뒤 빠른 fingerprint 재확인과 no-replace rename만 실행한다. 같은 Align의 append-only record 변경은 허용하지만 Align/Open 완료는 stale 처리한다. staging 삭제는 먼저 고유 quarantine으로 원자 이동하고 소유 inode를 확인한 뒤 수행하며, 목적지 경합의 승자나 교체된 foreign path는 보존한다. rename 뒤 실제 또는 미지원 directory `fsync`는 published-but-durability-uncertain 결과로 전달한다.

Native project Save는 GUI가 immutable `ArtifactSession`, project path와 Workbench state/authority version을 캡처한 뒤 기존 fail-closed project writer 전체를 worker에서 실행한다. 따라서 live/source geometry exact comparison, Git metadata subprocess, source closure 재해시, ZIP64 streaming, file/directory fsync, staged package production-reader 재개방과 source/Align materialization이 GUI event loop를 점유하지 않는다. writer의 atomic replace가 끝난 뒤 authority가 바뀌었으면 이미 기록된 파일을 과거 snapshot으로 명시하고 현재 Save As 경로·migration flag를 채택하지 않는다. 완료 경로는 Workbench의 `adopt_saved_project_path` compare-and-swap만 갱신하며 path-only state version을 전진시키되 document/render authority epoch는 유지한다. 정상 UI에서는 application-modal 진행 대화상자가 다른 명령을 막으며, programmatic authority 경합도 이 완료 fence가 숨기지 않는다. directory fsync 이후의 committed-but-uncertain 결과는 실패로 오인하지 않고 기존 내구성 경고를 유지한다.

비정상 종료 save-temp 복구는 project writer의 은닉 파일을 자동 소유물로 간주하지 않는다. 사용자가 고른 단일 폴더에서 exact writer basename을 가진 regular file만 bounded discovery하고, 발견한 device/inode/size/mtime capability가 유지된 descriptor를 새 목적지 parent staging으로 복사한다. staging이 production embedded-session loader에서 source/geometry/Align까지 완전히 재현된 뒤에만 OS별 atomic no-replace rename으로 게시한다. 후보·기존 intended destination은 어떤 결과에서도 변경하지 않고, 새 파일을 live Workbench 권위로 채택하는 동작도 별도 Project Open 확인으로 분리한다.

측정 worker 취소는 강제 thread termination이 아니라 공통 Event와 core 고유 취소 예외를 사용한다. Event는 게시 권위를 즉시 회수하고 세 extractor의 deterministic chunk 경계에서 확인된다. 취소되지 않은 실행의 recipe·payload·raster·QC에는 probe가 관여하지 않는다. 단일 NumPy/GEOS 호출, worker preflight 내부의 단일 materialization·resource-estimate 호출, Align candidate의 source hash/materialization과 project writer 내부의 단일 file/ZIP/fsync 단계는 비선점 구간으로 남지만 GUI thread에서는 실행하지 않는다. 창 종료는 현재 authoritative 측정·vector/rubbing/tile-unwrap export `TaskThread`의 one-shot cancel hook으로 record/publication 권위를 먼저 회수하고 최대 30초 동안 cooperative join을 기다린다. 취소 probe가 없는 Align 준비와 project save도 같은 bounded join에서 실제 종료를 증명해야 한다. join을 증명하지 못하면 창 teardown을 거부하고 task·dialog 소유권을 유지한다. join 뒤에는 signal을 끊고 task identity를 지운 뒤에만 창을 해제하므로 늦은 result가 닫히는 문서에 게시되지 않는다. native Align·측정·Save의 heavy GUI-thread 작업 제거는 완료했으며 legacy mesh/profile/slice worker의 통합 수명주기는 후속 셸 이행으로 관리한다.

## 이행 순서

1. 현재 신뢰 코어와 포맷을 고정하고 golden/스키마/Windows CI를 차단 게이트로 유지한다.
2. application shell package를 만들고 Open → 단위 확인 → Align을 먼저 이식한다. 이 단계는 완료됐으며 기존 GUI field는 이행 중 compatibility mirror로만 남긴다.
3. Cutline, 6-view Outline, Digital Rubbing을 command 단위로 옮긴다. immutable work item, worker computation, same-Align rebase와 VBO-free exact record publication까지 이식 완료했다.
4. vector/rubbing/tile-unwrap export를 worker staging과 final Workbench authority publication으로 분리한다. 이 단계는 이식 완료했으며 실제 대형 package에서 lock hold 시간과 취소 지연을 후속 측정한다.
5. 각 단계가 새 record/export로 완전히 연결되면 대응 legacy 측정 진입점을 제거한다.
6. 실제 유물 pilot과 대용량/GPU precision 검증 후 legacy 기와 검토 기능의 유지·플러그인화·제거를 결정한다.
7. Windows wheel hash lock, 실제 payload manifest, SPDX/NOTICE와 검증형 portable ZIP을 유지한다. ZIP은 한글 경로에 추출해 outbound 차단 complete workflow·actual-frame·삭제를 다시 통과해야 한다. 라이선스 결론, 서명 정책, 상위 build provenance와 대표 Windows pilot이 해결된 뒤에만 공개 바이너리를 만든다.

## 완료 기준

- native workflow에서 source vertex를 직접 변경하는 코드가 없다.
- 모든 1:1 SVG/PNG가 `READY + FRESH` record와 provenance에서만 생성된다.
- GUI 없이 동일 document와 source로 같은 canonical hash를 재현한다.
- Windows frozen CI와 Windows native-QPA/실제 OpenGL smoke가 통과한다. (commit `b12d4874a4a8`에서 충족)
- Windows portable ZIP의 모든 entry와 release evidence를 검증하고, 비 ASCII 경로 추출본에서 outbound 차단 complete workflow·actual-frame·삭제가 통과한다.
- 대표 유물군 pilot에서 단위·정렬·선 정밀도·탁본 판독 결과를 고고학자가 검수한다.
- 공개 배포 전 프로젝트와 GUI toolkit의 라이선스가 호환되고 제3자 고지가 완성된다.

## 전면 재작성을 다시 검토할 조건

GUI toolkit을 완전히 바꾸는 결정이 나더라도 신뢰 코어까지 재작성하지 않는 것이 기본이다. 다만 아래 중 하나가 실제 증거로 확인되면 해당 코어 모듈의 제한적 재작성을 별도 ADR로 검토한다.

- 고정된 포맷 계약이 실제 유물 데이터를 표현하지 못한다.
- 독립 구현과의 golden 비교에서 좌표·단위·topology가 반복적으로 불일치한다.
- 성능 목표가 profiling과 prototype으로도 현재 알고리즘 구조에서 달성 불가능하다.
- 보존한 코드의 권리 관계가 공개 배포를 허용하지 않는다.
