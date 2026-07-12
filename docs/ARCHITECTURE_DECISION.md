# Architecture Decision: 신뢰 코어 보존, 애플리케이션 셸 점진 교체

- 상태: Accepted
- 결정일: 2026-07-12
- 적용 범위: 첫 공개 안정판까지의 구조 개편

## 결정

ArchMeshRubbing을 전면 재작성하지 않는다. 검증 가능한 headless 측정 코어와 포맷 계약은 유지하고, 현재의 거대한 PyQt GUI·OpenGL 상태 계층과 legacy export 경로를 새 애플리케이션 셸로 단계적으로 교체한다.

이 방식은 흔히 strangler migration 또는 branch-by-abstraction으로 부른다. 사용자가 쓰는 한 단계씩 새 경로로 옮기고, 같은 기능의 legacy 진입점을 차단한 뒤 제거한다. 새 셸이 완성될 때까지 기존 프로그램을 실행 가능한 비교 기준과 전문 기와 워크플로우로 남긴다.

## 근거

코드베이스에는 버릴 부분과 지킬 부분이 명확히 함께 존재한다.

- `app_interactive.py`는 약 15,300줄, `src/gui/viewport_3d.py`는 약 15,100줄이며 각각 수백 개 메서드와 많은 광범위 예외 처리를 포함한다. UI, 상태, 렌더링, 작업 실행, 저장 정책이 서로 강하게 얽혀 있어 이 두 파일을 계속 확장하는 비용은 높다.
- 반면 source identity, 명시적 단위, immutable Align revision, canonical JSON/PNG, Cutline, Outline, Digital Rubbing, offline export는 GUI와 분리된 코어와 버전 스키마를 가진다.
- Python 3.12 기준 전체 `362 passed, 113 subtests passed`, M0 Pyright `0 errors`, Ruff 통과가 현재 계약을 보호한다.
- Python 3.12 macOS arm64 unsigned frozen 앱은 실제 GUI import/생성, 6개 mesh parser, PNG codec과 canonical document/vector/rubbing을 포함한 offline self-test 10개를 모두 통과했다. 이 로컬 증거는 dirty source tree의 개발 스모크이며 공개 배포 증거가 아니다.

전면 재작성은 이 검증 자산과 오래 축적된 기와 처리 알고리즘까지 동시에 다시 만들게 한다. 반대로 기존 GUI에 기능을 계속 덧붙이면 mutable scene 상태와 권위 기록이 다시 섞인다. 따라서 코어는 보존하고 셸은 교체하는 경계가 가장 안전하다.

## 보존할 것과 교체할 것

| 보존·강화 | 점진 교체·격리 |
|---|---|
| `ArtifactDocument`, `ArtifactSession`, `ArtifactWorkbench`, source/geometry identity | `app_interactive.py`의 단일 `MainWindow` 오케스트레이션 |
| canonical-mm Align/Cutline/Outline/Rubbing 계산 | `viewport_3d.py`의 데이터 소유·도구 상태·렌더링 혼합 |
| RFC 8785 JSON, canonical PNG, versioned schemas | mutable `legacy_ui_state`와 암묵적 작업 완료 상태 |
| `.amr`, `.amr-vector`, `.amr-rubbing` 검증·원자 저장 | screenshot/OpenCV/convex-hull을 실측 산출물로 쓰는 우회 경로 |
| 테스트 fixture, golden hash, offline validator | destructive bake와 저장 가능한 원본 변형 |
| 기와형 flatten 알고리즘과 검토용 legacy 기능 | 플랫폼별 수동 빌드·바로가기 스크립트 |

Legacy 기능은 즉시 삭제하지 않는다. 연구 검토용이면 명확히 `legacy review`로 표시하고, 1:1 측정 결과처럼 보이지 않게 분리한다. 새 포맷으로 안전하게 승격할 수 없는 상태는 추정 변환하지 않고 fail closed한다.

## 새 셸의 경계

애플리케이션 셸은 최소한 아래 계층을 분리해야 한다.

1. `Application state`: 현재 document/session과 project path, 단일 pending Open, 활성 revision, 작업 가능 여부만 소유한다.
2. `Commands`: Open, Align commit, Cutline/Outline/Rubbing compute, export를 명시적 입력·결과로 실행한다.
3. `Viewport adapter`: document에서 파생된 표시용 scene만 만들며 권위 geometry를 소유하거나 bake하지 않는다.
4. `Workflow panels`: command를 호출하고 record 상태를 표시하되 계산·저장을 직접 수행하지 않는다.
5. `Persistence/export`: GUI와 독립된 코어 API만 사용한다.

GUI 버튼의 초록색 완료 표시는 위젯 내부 boolean이 아니라 `READY + FRESH` record에서 파생해야 한다. Align이 바뀌면 후속 기록을 물리적으로 삭제하기보다 immutable history에 남기고 stale로 판정하여 현재 산출물로 사용하지 못하게 한다.

현재 `src/application/artifact_workbench.py`는 Qt·OpenGL 없이 session/project authority, ticketed Open, `state_version`/`authority_epoch` compare-and-swap, 명시적 Align readiness와 two-phase projection publication을 소유한다. Open/new import/project reopen과 Align commit/parent activation은 이 경계를 사용한다. Open이 만드는 `initial_identity`는 materialization baseline일 뿐 사용자 확정이 아니며, 변화량이 0인 첫 Align도 immutable child revision으로 남겨야 측정 단계가 열린다.

DerivedRecord session update는 기존 source/metadata/Align/record를 바꿀 수 없고 추가할 record ID 집합을 명시해야 한다. scene swap 실패는 이전 권위로 rollback하며, rollback·scene 복원·finalize 자체의 성공을 증명할 수 없으면 fatal authority 상태로 전환해 검증된 Open 전까지 저장·실측·내보내기를 차단한다. viewport의 transient cut-section/ROI callback은 current worker identity, projection generation과 selected object로 fence하여 이전 scene의 결과가 새 overlay나 worker authority를 건드리지 못하게 한다. Cutline/Outline/Digital Rubbing 결과 publication은 이 검증을 사용하지만 command 입력과 계산의 소유권은 아직 MainWindow/viewport에 남아 있어 다음 이행 단계에서 옮긴다.

## 이행 순서

1. 현재 신뢰 코어와 포맷을 고정하고 golden/스키마/3-OS CI를 차단 게이트로 유지한다.
2. application shell package를 만들고 Open → 단위 확인 → Align을 먼저 이식한다. 이 단계는 완료됐으며 기존 GUI field는 이행 중 compatibility mirror로만 남긴다.
3. Cutline, 6-view Outline, Digital Rubbing을 command 단위로 옮긴다. 현재 결과 publication은 보호되지만 command 입력·계산 소유권 이식은 진행 중이다.
4. 각 단계가 새 record/export로 완전히 연결되면 대응 legacy 측정 진입점을 제거한다.
5. 실제 유물 pilot과 대용량/GPU precision 검증 후 legacy 기와 검토 기능의 유지·플러그인화·제거를 결정한다.
6. 라이선스, 서명, 3-OS frozen smoke가 해결된 뒤에만 공개 바이너리를 만든다.

## 완료 기준

- native workflow에서 source vertex를 직접 변경하는 코드가 없다.
- 모든 1:1 SVG/PNG가 `READY + FRESH` record와 provenance에서만 생성된다.
- GUI 없이 동일 document와 source로 같은 canonical hash를 재현한다.
- Windows·macOS·Linux frozen CI와 실제 OpenGL smoke가 통과한다.
- 대표 유물군 pilot에서 단위·정렬·선 정밀도·탁본 판독 결과를 고고학자가 검수한다.
- 공개 배포 전 프로젝트와 GUI toolkit의 라이선스가 호환되고 제3자 고지가 완성된다.

## 전면 재작성을 다시 검토할 조건

GUI toolkit을 완전히 바꾸는 결정이 나더라도 신뢰 코어까지 재작성하지 않는 것이 기본이다. 다만 아래 중 하나가 실제 증거로 확인되면 해당 코어 모듈의 제한적 재작성을 별도 ADR로 검토한다.

- 고정된 포맷 계약이 실제 유물 데이터를 표현하지 못한다.
- 독립 구현과의 golden 비교에서 좌표·단위·topology가 반복적으로 불일치한다.
- 성능 목표가 profiling과 prototype으로도 현재 알고리즘 구조에서 달성 불가능하다.
- 보존한 코드의 권리 관계가 공개 배포를 허용하지 않는다.
