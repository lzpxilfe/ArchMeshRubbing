# 현장 파일럿 증거 절차

이 절차는 한 유물, 한 Windows 64-bit 컴퓨터, 한 ArchMeshRubbing 빌드에서 수행한 실측을 재검토 가능한 JSON 기록으로 묶는다. 결과는 대표 유물군 전체의 성능, 제품 출시 승인, 서명된 출처 증명이 아니다. 실제 고고학자의 판정 없이 `verified`가 될 수 없다.

## 준비물

- 완료된 self-contained `PROJECT.amr`
- 같은 프로젝트의 `Cutline 3/3 → Outline 6/6 → Digital Rubbing 6/6`을 담은 `SURVEY.amr-survey`
- 해당 Windows 빌드에서 생성한 native OpenGL driver-smoke report
- 고고학자가 닫힌 항목을 직접 판정한 review JSON
- 1:1 출력물을 확인할 자 또는 캘리퍼스와, 필요하면 Illustrator 등 실제 사용 프로그램

프로젝트와 survey는 GUI에서 완료한 뒤 같은 폴더에 복사해도 된다. 검증은 네트워크, 계정, 라이선스 서버를 사용하지 않는다. offline 항목을 `pass`로 판정하려면 현장과 같은 차단 환경에서 실제 작업을 수행한다.

## 1. Windows 그래픽 경로 기록

현재 파일럿에 사용할 실행 파일에서 다음 명령을 실행한다.

```bat
ArchMeshRubbing.exe --opengl-driver-smoke-report windows-opengl.json
```

source checkout이라면 `python main.py`로 같은 옵션을 실행할 수 있다. report는 native `qwindows` context, 24-bit 이상 depth buffer, 두 투영 모드, 실제 pixel/depth/pick 검사와 정리 성공을 포함해야 한다. 파일럿 CLI는 OpenGL report의 runtime lock SHA-256을 현재 실행 빌드와 대조하고, 양쪽 commit이 알려진 경우 commit도 같아야 통과시킨다.

## 2. 검토 양식 생성

기존 파일을 덮어쓰지 않는 template을 만든다.

```bat
ArchMeshRubbing.exe --field-pilot-review-template archaeologist-review.json
```

template은 의도적으로 모든 항목이 `not_tested`이고 artifact 결합 hash도 `null`이므로 그대로는 통과할 수 없다. 먼저 다음 두 검증 receipt에서 `evidence.document_sha256`와 `evidence.artifact_set_sha256`를 확인한다.

```bat
ArchMeshRubbing.exe --verify-artifact PROJECT.amr --report project-verification.json
ArchMeshRubbing.exe --verify-artifact SURVEY.amr-survey --against-project PROJECT.amr --report survey-verification.json
```

두 값을 review의 `project_document_sha256`, `survey_artifact_set_sha256`에 각각 옮긴다. 둘은 함께 입력하거나 함께 `null`이어야 하며, report를 만들 때 실제 project/survey와 하나라도 다르면 human review가 `fail`이 된다. 이 결합은 다른 유물의 검토 양식을 실수로 재사용하는 일을 막기 위한 것이며 서명은 아니다.

그다음 고고학자가 다음 열 항목을 각각 `pass`, `fail`, `not_tested` 중 하나로 판정한다.

| 항목 | 확인 내용 |
|---|---|
| `source_unit` | 원본 단위와 축 해석이 등록 자료·실물과 맞는가 |
| `align_grounding` | 정렬과 기준면 안착이 실측 목적에 맞는가 |
| `cutline_fidelity` | Top/Front/Right 단면이 메쉬와 실제 형상을 충실히 따르는가 |
| `outline_fidelity` | 6면 외곽, 오목부, 구멍, 분리 성분이 충실한가 |
| `rubbing_legibility` | 탁본이 문자·문양·제작 흔적 판독에 실제 도움이 되는가 |
| `physical_scale_1_1` | 출력 또는 외부 프로그램 왕복 뒤 물리 길이가 허용오차 안인가 |
| `original_source_preserved` | 외부 원본 없이 `.amr`를 재열고 원본 hash를 확인했는가 |
| `offline_operation` | 차단 환경에서 계정·서버 없이 흐름을 완료했는가 |
| `workflow_stability` | 작업 중 충돌·데이터 소실·권위 불일치가 없었는가 |
| `workflow_usability` | 연구자가 흐름을 이해하고 결과를 완성할 수 있었는가 |

`physical_scale_1_1`이 `pass` 또는 `fail`이면 `scale_expected_mm`, `scale_observed_mm`, `scale_tolerance_mm`를 모두 숫자로 기록해야 한다. 관측 오차가 허용오차 이내면 상태는 반드시 `pass`, 밖이면 반드시 `fail`이어야 한다. `not_tested`이면 세 값은 모두 `null`이어야 한다.

`workflow_usability`이 `pass` 또는 `fail`이면 `workflow_elapsed_minutes`를 양수로 기록한다. 실제 검토가 끝나면 `artifact_label`, 가명 또는 기관 내부 식별자인 `reviewer_id`, UTC whole-second 형식의 `reviewed_at_utc`를 placeholder가 아닌 값으로 바꾼다. 검토 시각은 최종 report 생성 시각보다 미래일 수 없다. 두 artifact hash, 모든 열 항목과 시간이 닫혀야 human review가 `pass`가 된다.

## 3. 단일 파일럿 보고서 만들기

```bat
ArchMeshRubbing.exe --field-pilot PROJECT.amr SURVEY.amr-survey ^
  --review archaeologist-review.json ^
  --opengl-report windows-opengl.json ^
  --report field-pilot.json
```

명령은 다음을 한 번에 수행한다.

- `.amr`의 embedded source, 단위·축, geometry, Align, record를 production loader로 재물질화
- `.amr-survey`의 15개 자식 package, 9개 1:1 SVG, 6개 1:1 PNG, aggregate hash를 exact project와 재검증
- OpenGL report의 Windows native context와 현재 빌드 결합 확인
- human review의 닫힌 필드와 정량 scale 판정 확인
- OS·architecture·RAM·peak working set과 project/survey 검증 시간을 기록
- canonical RFC 8785 JSON, semantic `pilot_sha256`, `authentication=none`을 갖는 report를 no-overwrite 원자 게시

종료 코드는 `verified`일 때 `0`, 유효하지만 `failed` 또는 `incomplete`인 report를 게시했을 때 `1`, 옵션·입력 review·출력 게시 오류일 때 `2`다. `--review`나 `--opengl-report`를 생략해도 근거 부족을 숨기지 않는 `incomplete` report가 만들어지고 종료 코드 `1`을 반환한다.

`verified`에는 다음 네 조건이 모두 필요하다.

- project와 exact-project survey 검증 `pass`
- human review 열 항목 전부 `pass`
- Windows native OpenGL report `pass`
- report를 만든 프로세스가 Windows 64-bit

하나라도 명시적으로 실패하면 `failed`, 아직 시험하지 않은 근거가 있으면 `incomplete`다.

## 4. 받은 보고서 확인

```bat
ArchMeshRubbing.exe --verify-field-pilot field-pilot.json --report verification.json
```

이 명령은 canonical bytes, closed schema, 내부 교차 주장과 `pilot_sha256` 일치를 확인한다. report에 기록된 pilot outcome이 `failed`나 `incomplete`여도 report 자체가 일관되면 verification receipt의 `ok`는 `true`다. 이는 과거 입력 파일을 다시 여는 검증이 아니며, self-hash는 누구나 다시 계산할 수 있으므로 서명이나 작성자 인증이 아니다.

구조 계약은 다음 세 파일에 고정한다.

- [`field_pilot_review-1.0.0.schema.json`](../schemas/field_pilot_review-1.0.0.schema.json)
- [`field_pilot_report-1.0.0.schema.json`](../schemas/field_pilot_report-1.0.0.schema.json)
- [`field_pilot_verification-1.0.0.schema.json`](../schemas/field_pilot_verification-1.0.0.schema.json)

## 개인정보와 공개 범위

자동 수집 필드는 hostname, OS 사용자명, 절대 입력 경로를 저장하지 않는다. 입력 파일의 basename, 유물 label, reviewer ID, 자유 입력 `notes`, GPU vendor/renderer는 저장한다. 특히 `notes`는 프로그램이 내용을 익명화하지 않으므로 공개 전에 개인정보·소장 위치·미공개 유물 정보를 직접 제거해야 한다.

report의 scope는 항상 `single_artifact_single_machine`, release claim은 항상 `single_pilot_only_not_release_approval`, authentication은 항상 `none`이다. 대표 암키와·수키와·토기·금속·거친 부식면, 대용량 mesh, 저메모리 PC와 실제 연구자 여러 명에 대한 반복 결과가 따로 쌓이기 전에는 경쟁 제품 전반보다 우월하다는 근거로 사용하지 않는다.
