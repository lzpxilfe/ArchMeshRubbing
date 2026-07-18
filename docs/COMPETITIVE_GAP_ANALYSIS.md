# 공개 기능 대비표와 우위 기준

이 문서는 2026-07-14에 확인한 [ARIA Pro 공개 소개 화면](https://aras-archaeology.kr/ARIA_pro_v1.html)과 ArchMeshRubbing 저장소를 기능 단위로 비교한 내부 제품 기준이다. 공개 UI에 표시된 이름과 동작만 비교했으며, 상대 제품의 비공개 코드·계정·파일·네트워크를 조사하거나 우회하지 않는다. 따라서 ARIA 열의 내용은 구현 품질을 검증한 결론이 아니라 공개 화면에서 확인되는 기능 또는 주장이다.

ArchMeshRubbing의 제품 비교 범위는 Windows 10 version 1809 이상 x64와 Windows 11 x64의 source 실행 및 검증형 portable ZIP이다. 비 Windows 지원이나 installer·MSIX·Store 배포는 비교 우위로 주장하지 않는다.

## “우월하다”의 정의

체크박스 수가 아니라 아래 증거를 모두 만족해야 우위로 기록한다.

1. 같은 원본·단위·Align·recipe에서 같은 결과를 다시 만들 수 있다.
2. 결과 파일만 받은 제3자가 단위·원본/geometry hash·selection·QC를 offline 검증할 수 있다.
3. 합성 정답과 공개 가능한 실물 scan pilot에서 오차·실패율·처리시간을 수치로 비교한다.
4. 계정·구독·license server·인터넷 연결이 없어도 핵심 기록 작업을 끝낼 수 있다.
5. 실패나 fallback을 성공으로 숨기지 않고, 원본과 이전 revision을 삭제하지 않는다.
6. 연구자가 처음 결과를 얻는 시간, 키보드/마우스 동선, 오류 복구까지 함께 평가한다.

## 2026-07-14 기준 기능 대조

| 영역 | ARIA Pro 공개 화면 | ArchMeshRubbing 현재 증거 | 판정과 다음 조건 |
|---|---|---|---|
| 기와 정렬·전개 | 기와 정렬, 원통 투영 이미지, 영역 선택 곡률 전개 UI가 표시됨 | native desktop의 전체/선택 face·X/Y/Z 장축·Top/Bottom·section/QC 패널, variable-radius sectionwise, `surface.tile_unwrap.v1`, 1 µm 좌표·왜곡/foldover gate, 권위 재검증 뒤 `.amr-unwrap` binary/OBJ/SVG/provenance | 계산·신뢰·desktop 연결은 차별화. 실물 암·수키와 pilot 전에는 종합 우위라고 표시하지 않음 |
| 단위·정렬 이력 | 두 점 거리 scale 보정과 축 정렬 UI가 표시됨 | source bytes와 decode geometry의 별도 hash, 확인 단위/축, immutable proper-rigid Align revision, stale history 복원 | ArchMeshRubbing 우위. 실제 사용자 오류율 비교는 필요 |
| 1:1 결과 근거 | SVG drawing/OBJ export 메뉴가 표시됨 | exact-mm SVG/PNG/unwrap, recipe/QC/dependency closure, canonical payload hash, offline verifier, 완료 3/6/6을 15개 자식과 aggregate manifest로 한 번에 게시하는 `.amr-survey` | ArchMeshRubbing 우위. Illustrator 왕복 실물 출력 pilot 필요 |
| 절단·단면 | 직선/교차/경사 절단, clipping, quadrant, 단면선 추출 UI가 표시됨 | 검증 Cutline과 단면/clip 기반 legacy 도구가 있으나 조각 생성·통합 편집 UX는 부족 | ARIA UI가 앞섬. non-destructive fragment record와 recall clip 필요 |
| 조각 관리·복원 | 절단 뒤 자동 분리, piece/line list, pottery restoration UI가 표시됨 | surface separation과 기록면 선택 코어는 있으나 native fragment document workflow가 없음 | ARIA 공개 기능 우세. 원본 보존형 fragment revision이 P1 |
| RTI | 광원 방향·preset·specular/ambient·고대비·확대 UI가 표시됨 | 동등한 native RTI 모듈 없음 | 명확한 격차. 공개 RTI 원리 기반 독립 구현과 reproducible light recipe가 P1 |
| MSII | multi-scale integral invariant, radius/scale/sample, 여러 curvature mode와 preset UI가 표시됨 | curvature/feature 도구는 있으나 검증 MSII record가 없음 | 명확한 격차. 논문·GigaMesh 공개 자료 기반 독립 구현과 benchmark가 P1 |
| Digital Rubbing | contrast, smoothing, offset, noise, inverse, cylindrical image extraction UI가 표시됨 | canonical six-view Digital Rubbing receipt, exact pixels/mm, deterministic GA8 PNG, provenance와 offline validation | 연구 재현성은 ArchMeshRubbing 우위. 판독 품질 blind test와 펼친 면 위 rubbing 결합은 필요 |
| 측정·체적 | 거리 측정과 volume 분석 메뉴가 표시됨 | `measurement.geometry_metrics.v1`의 고정 표면적·topology-gated exact-rational 체적, source triangle+10억 분율 barycentric anchor의 Euclidean chord 거리와 best-fit planar circle 지름을 Align-bound record로 저장·재계산함 | 결과 의미·입력 anchor·reopen 검증 계약은 ArchMeshRubbing 우위. 실제 연구자 반복 측정 오차와 대용량 Windows UX pilot은 필요 |
| project/history | project save/load UI가 표시됨 | content-addressed self-contained `.amr`, parser dependency closure, 외부 원본 삭제 뒤 독립 프로세스 reopen, immutable record graph | ArchMeshRubbing 우위 |
| offline·라이선스 | 체험 제한과 PRO/구독 안내가 표시됨 | 오픈소스 지향, 계정·license server 없는 offline architecture, outbound 차단 Windows package gate | ArchMeshRubbing 방향 우위. 실제 공개 license 결정과 서명 배포 전에는 완료 아님 |
| 언어 | 한국어·영어·중국어·일본어 선택 UI가 표시됨 | 주 UI가 한국어 중심 | ARIA 우세. 문자열 catalog와 번역 검증이 P2 |
| 접근성·온보딩 | 기능이 한 화면에 통합되어 있으나 공개 체험 제한이 존재 | 엄격한 native workflow·진행 gate·기와 신뢰 경로의 desktop binding과 10항목 archaeologist review/elapsed-time field-pilot 계약 구현 | 측정 절차는 준비됨. 실제 5분 첫 결과 usability pilot 증거가 없어 아직 우위 주장 불가 |
| 시각 체계 | 공개 화면에 emoji 기반 label이 다수 보임 | 33개 handcrafted 16×16 pixel icon, deterministic cache와 frozen startup test | ArchMeshRubbing 차별화. 고대비·확대·키보드 접근성 검증은 필요 |
| Windows 전달 | 웹 소개에서 설치형 제품으로 안내됨 | hash-locked build, SPDX/NOTICE/evidence, compiler-independent portable ZIP, 한글 경로·firewall-offline·actual qwindows OpenGL CI | 공급망 검증은 ArchMeshRubbing 우위. 서명·대표 GPU·실물 pilot은 미완료 |

## 실행 순서

### P0: 기와 작업을 실제 연구자에게 닿게 만들기

- 완료: 메인 작업 흐름 바로가기에서 열 수 있는 native desktop `기와 전개` command/panel을 연결했다.
- 완료: 기록면 전체/현재 face selection, 장축 `X/Y/Z`, Top/Bottom, section 수와 QC를 한 화면에서 검토한다.
- 완료: 선택한 `READY + FRESH` record를 recipe로 재계산하고 exact receipt를 확인한 뒤에만 `.amr-unwrap`으로 내보내며 기존 자유 flatten export와 구분한다.
- 완료: Cutline 3/3·Outline 6/6·Digital Rubbing 6/6을 한 버튼으로 `.amr-survey`에 원자 게시하고 이동 후 exact project와 offline 재검증한다.
- 완료: 한 project/survey, 현재 Windows build의 native OpenGL report, 정량 scale과 10항목 고고학자 review를 canonical 단일 파일럿 report로 묶고 근거가 빠지면 `incomplete`로 남기는 공개 CLI·schema를 마련했다.
- 완료: 전체 active geometry의 표면적과 topology-gated 체적을 `measurement.geometry_metrics.v1`로 원자 게시한다. 열린 메쉬·비다양체·방향 불일치·다중 조각은 convex-hull 값으로 숨기지 않고 체적을 unavailable로 남긴다.
- 완료: exact frame의 depth ray를 전체 CPU triangle과 교차해 source face+barycentric anchor로 고정하고, 3D Euclidean chord 거리와 3~64점 PCA 평면·정규화 대수 Kasa 원 지름을 immutable record로 게시·재열어 검증한다. pick QC는 총 스캔 불확도가 아니라 depth 재부착·pixel footprint·edge 근접·fit residual 범위임을 명시한다.
- 실제 상·하면을 자동 분류하는 기록면 classifier와 명시적 seam 편집을 구현한다. 현재 Top/Bottom은 사용자가 고른 동일 face selection의 방향 해석이며 자동 표면 판정이 아니다.
- 공개 가능한 암키와·수키와 scan set을 마련해 폭/길이/단면 arc 오차, foldover, 처리시간을 기록한다.
- SVG를 Illustrator/Inkscape로 열어 physical size를 실물 자와 대조한다.

### P1: 공개 화면에서 확인된 기능 격차 닫기

- non-destructive cutting/fragment revision, recall clip, piece list
- triangle/barycentric anchor 기반 distance·diameter record와 pick uncertainty QC
- RTI light recipe와 deterministic image export
- MSII scale/radius recipe, 공개 합성 benchmark와 feature map record
- 펼친 기와 좌표에서 Digital Rubbing/문양선을 재투영하는 결합 workflow

### P2: 사용성에서 앞서기

- 한국어/영어 우선 string catalog 뒤 중국어/일본어 확장
- keyboard-only, 고대비, 확대, 색각 안전성과 icon tooltip 검증
- 5분 tutorial dataset, 실패 복구 안내, 공개 demo video
- 실물 연구자 blind test 결과와 benchmark를 release evidence에 포함

## 금지선

- 상대 서비스의 로그인·구독·접근 제한을 우회하지 않는다.
- 화면 이름이나 외형을 그대로 복제하지 않는다.
- 논문·공개 문서·독립 측정 문제를 바탕으로 자체 recipe, schema, 알고리즘과 UI를 설계한다.
- 검증하지 않은 AI·정확도·1:1·offline 주장을 README나 release 문구에 넣지 않는다.
