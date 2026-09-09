# Feature Reference Map

이 문서는 “어떤 기능이 어떤 공개 레퍼런스와 연결되는가?”를 정리합니다.

참조 ID는 아래 문서에 정의되어 있습니다.

- [`docs/REFERENCES.md`](REFERENCES.md)

---

## Legend

- `paper`: 논문/공개 알고리즘에서 직접 아이디어를 가져온 경우
- `library`: 라이브러리 구현/공식 문서 의존성이 큰 경우
- `heuristic`: 프로젝트 목적에 맞춘 엔지니어링 규칙/보정 로직
- `inspiration`: 고고학 기록/판독 관점에서 문제 설정에 영향을 준 경우

상태 열은 [ARCHITECTURE_DECISION.md](ARCHITECTURE_DECISION.md)가 요구하는 구분입니다.

- `native`: 현재 출하되는 권위 경로. `.amr` record와 1:1 export를 만든다.
- `legacy`: 연구 검토용으로만 남아 있고, ArtifactDocument session이 열려 있으면 파일 출력이 차단된다.

---

## 출하되는 권위 경로 (native)

| Feature | Core approach | Reference IDs | Type | Status | Main code |
|---|---|---|---|---|---|
| 디지털 탁본 | 6면 정사영 front-depth raster + 정수 µm 양자화 + masked square local-mean relief (summed-area table) | `[R10]`, `[R11]` | paper + engineering contract | native | `src/core/artifact_rubbing_extractor.py` (`RUBBING_ALGORITHM = archmeshrubbing.orthographic_local_mean_relief`) |
| 도판에 붙인 탁본 | 탁본 record를 벡터 도형과 같은 목록·같은 축척으로 배치; receipt로 대조한 raster를 canonical GA8 PNG data URI로 박아 도판이 자기 완결적이고 바이트가 결정적 | - | engineering contract | native | `src/core/drawing_sheet.py` |
| 회전축 기준 외면 띠 | 정치한 축을 기준으로 (자오선 각도, 면을 따라 잰 폭, 높이 범위)에서 면 집합을 자름; 법선의 바깥 방향과 두 겹의 반지름 대소를 함께 보고 외면을 가리며, 뒤집힌·감김이 뒤섞인·조각난 메쉬는 조각 크기를 알리고 거부 | `[L8]` | engineering contract | native | `src/core/artifact_surface_strip.py` |
| 전개 탁본 | `tile_unwrap` record가 증명한 전개 좌표 (u, v) 위에 회전 중심 반지름을 깊이로 삼아 같은 local-mean relief를 그림; 전개 payload 해시로 record에 묶여 재계산·검증되고 같은 `.amr-rubbing` 패키지(sidecar 1.4.0)로 나감 | `[R10]`, `[R11]` | paper + engineering contract | native | `src/core/artifact_developed_rubbing.py` (`raster.developed_rubbing.v1`), `src/core/artifact_rubbing_export.py` |
| 문양 내선 | 한 뷰의 외벽을 뷰 좌표로 전개해 법선 지도의 높이를 적분하고, 헤시안 골선(곡률 문턱·영교차·비최대 억제)을 사슬로 추적해 평활한 폴리라인 record로; 도판은 뷰 frame이 같은 입면에 내선으로 그림. 시문선의 부호(골/마루)와 곡률 문턱은 recipe의 것 | `[K1]` p. 37 | paper + engineering contract | native | `src/core/artifact_texture_lines.py` (`measurement.texture_lines.v1`), `src/core/drawing_sheet.py` (`texture_line_records`) |
| 법선 지도 기복 탁본 | 메쉬가 아니라 텍스처에 구워진 잔 요철: OBJ 모서리 텍스처 좌표로 전개 픽셀마다 물체 공간 법선 지도를 읽고, 평활 법선을 뺀 기울기를 전개축으로 돌려 Frankot-Chellappa FFT 적분한 높이를 같은 relief 렌더러에 넣음; 지도의 축 반전(`encoding`)은 적분 부적합으로 재어 실측자가 고르고 recipe에 남음 | - | engineering contract | native | `src/core/artifact_texture_relief.py`, `src/core/artifact_developed_rubbing.py` (`archmeshrubbing.developed_texture_normal_relief`), `schemas/rubbing_export-1.4.0.schema.json` |
| 접촉 모델 탁본 | 주변 평균으로 기울기를 뺀 면의 국소 상한 포락선(정수 sliding max)에 얼마나 가까운가로 먹을 정함; 닿는 면은 접촉 농담으로 고르게, 침선은 희게, 승문은 마루만 | - | engineering contract | native | `src/core/artifact_rubbing_extractor.py` (`_render_contact_relief`, `relief_model=contact_envelope/v1`) |
| 종이 기저 농담 | 솜방망이는 종이 전체에 먹을 남기므로 주변과 높이가 같은 면이 `paper_tone_percent`만큼 먹고 마루는 검정까지, 골은 그 절반까지만 옅어짐; 0이면 세 열쇠가 recipe에서 빠져 이전 recipe가 바이트까지 재현됨 | - | engineering contract | native | `src/core/artifact_rubbing_extractor.py` (`_render_local_relief`), `schemas/rubbing_export-1.3.0.schema.json` |
| 한 바퀴 도는 홈 | 정치한 축 기준 외면 반지름의 둘레 중앙값을 높이로 읽고, 국소 2차 적합(들어간 칸을 빼고 재적합)에서 들어간 띠를 찾음; 양쪽 능선이 홈 깊이의 절반 넘게 어긋나면 돌대 사면으로 보고 거부 | `[L8]` | engineering contract | native | `src/core/artifact_profile_groove.py` (`measurement.profile_groove.v1`) |
| 간선·직선 | 홈 하나를 골 간선 1줄 + 능선 직선 2줄로 그림; 끊김 횟수는 파선 패턴이 아니라 기하로, 중심축 좌우 각각 넣어 반입면도 제 횟수를 가짐 | - | engineering contract | native | `src/core/drawing_svg.py` (`axis_profile_chord`, `broken_chord`), `src/core/drawing_style.py` |
| 6면 외곽선 | 모든 투영 삼각형의 fixed-grid 다각형 합집합 (rasterisation·convex hull 없음) | `[L8]` | library + engineering contract | native | `src/core/artifact_outline_extractor.py` (`OUTLINE_ALGORITHM = archmeshrubbing.projected_triangle_union`) |
| 회전축 정치 | 지름 record 2개의 원 중심을 잇는 축을 +Z로 보내는 proper rigid Align + recipe 자체 재계산 + 동축성 QC. 중심 간격이 큰 반지름의 25% 미만인 납작한 유물(접시·뚜껑)은 중심선이 맞춤 오차에 묻히므로 두 원의 평면 법선의 평균을 축으로 잡고 recipe `axis_source`(`center_line/v1`·`circle_plane_normals/v1`)에 적는다; 키가 없는 옛 recipe는 중심선이다. 가마에서 뒤틀려 둥글지도 수평이지도 않은 그릇은 실측자가 `standing_on_foot/v1`을 골라 평평한 바닥에 놓은 자세로 세운다 — 굽 원의 anchor 셋이 바닥에 닿는 점이고 축은 그 평면의 법선; 프로그램이 스스로 고르지 않는다 | `[R6]` | paper + engineering contract | native | `src/core/artifact_axis_alignment.py` (`rotation_axis_from_circle_records/v1`) |
| 단면선 | 명시 평면과 canonical-mm 삼각형의 정확 교차 + endpoint 스티칭, 모호한 경우 fail closed | - | engineering contract | native | `src/core/artifact_vector_extractor.py` |
| 외곽선 topology 검증 | ring simple/area, hole ownership, 성분 비중첩 증명 | `[L8]` | library | native | `src/core/artifact_outline_topology.py` |
| 토기 외면 띠 전개 | 정치로 잰 회전축을 단면 중심으로 고정하고 세로축을 자오선 길이로 잰 sectionwise 전개 (원 맞춤 없음) + 폭별 왜곡 QC | `[R6]` | paper + engineering contract | native | `src/core/flatten_models_sectionwise.py` (`section_center="axis_origin"`, `station="meridian"`), `src/core/artifact_tile_unwrap_extractor.py` (recipe 1.3.0) |
| 정식 기와 전개 record | explicit canonical axis + exact face selection + 1 µm quantization + no-fallback quality gate | `[R4]`, `[R6]` | paper + heuristic | native | `src/core/artifact_tile_unwrap_extractor.py`, `src/core/artifact_tile_unwrap_record.py` |
| sectionwise 펼침 | longitudinal axis + repeated cross-sections + local circular fits | `[R4]`, `[R6]` | paper + heuristic | native | `src/core/flatten_models_sectionwise.py` |
| distortion summary | per-face distortion aggregation for QC gate | `[R1]`, `[R4]` | paper + heuristic | native | `src/core/flatten_metrics.py` |
| 원 맞춤 지름 | PCA best-fit plane 위 정규화 대수 Kasa 원 | `[R6]` | paper | native | `src/core/artifact_surface_measurement.py` (`SURFACE_DIAMETER_FIT_POLICY`) |
| 표면 거리 | source triangle + 10억 분율 barycentric anchor의 3D Euclidean chord (측지 거리 아님) | - | engineering contract | native | `src/core/artifact_surface_measurement.py` |
| 표면적·체적 | 1 µm 격자 양자화 표면적 + topology-gated exact-rational 체적 (convex hull fallback 없음) | - | engineering contract | native | `src/core/artifact_geometry_metrics.py` |
| 도면 선 종류 표현 | 닫힌 선 종류 어휘 + 종이 mm preset(해시 봉인) + role→선 종류 매핑 + 레이어드 SVG·단면 해칭 | `[K1]` | paper + engineering contract | native | `src/core/drawing_style.py`, `src/core/artifact_vector_export.py` |
| 좌 반입면 · 우 반단면 | 회전축 기준 반평면 클리핑으로 입면 record와 단면 record를 한 도형에 합침 + 접힌 변은 획을 긋지 않음 | - | engineering contract | native | `src/core/drawing_svg.py`, `src/core/drawing_sheet.py` |
| 뒷면 실루엣과 뒷구연선 | 뷰 평면 뒤에 중심이 놓인 면들만의 외곽선을 외곽선 추출기의 고정 격자로 뜬 record(`measurement.far_silhouette.v1`) + 미러 도형의 `outline_reach="far"`가 단면선에서 간격 넘게 떨어진 그 가장자리(기운 구연의 뒷선, 오목한 저부 밑의 뒤쪽 굽 바닥)를 그대로 그림. 실측자의 스케치에서 온 잠정 관례 | - | engineering contract + heuristic | native | `src/core/artifact_far_silhouette.py`, `src/core/drawing_sheet.py` |
| 잰 데까지만: 추정 점선과 계단 꺾음 | 안쪽이 스캔되지 않은 병의 단면 쪽에 실측자의 값으로 점선을 더함(`presumed_lines`: 안벽 연장 `wall_on`은 끝에서 읽은 두께로 바깥벽을 따라, 자로 잰 바닥 `floor`는 굽 안 밑면을 두께만큼 들어 올려; 제목란 `추정` 행·sidecar 필수) + 높이가 맞닿은 `mirror_jogs`를 계단 하나로 긋고(도달 `math.inf`는 벽까지: 단면을 빼고 외형선이 모서리, 가운데선은 벽에서 끊김) 따 붙인 채색은 먹 픽셀 기준으로 계단 안이면 됨. 실측자의 지시에서 온 잠정 관례 | - | engineering contract | native | `src/core/drawing_sheet.py` |
| 도판 패널 (여섯 탭) | 도판이 취하는 결정 전부를 창에서 다룸: 도면(기록·배치·제목란) · 종이(용지·축척·preset·선) · 반쪽(입면 쪽·외형선·미러 짝·계단·추정선) · 내선(능선·문양·홈·꺾임과 그 판단) · 문양(채색 따 붙이기·양각 점묘·탁본) · 해석. 명세를 읽고 쓰며, 편집하지 않는 항목은 실어 나름 | - | engineering contract | native | `src/gui/plate_panel.py` |
| 도판 명세 `plate_spec` | `DrawingSheetOptions` 전체(record 목록·축척·페이지·preset·계단·추정선·따 붙이기·뒷선·꺾임 선택·해석)를 닫힌 JSON 하나로 쓰고 읽음(왕복 동일, 모르는 키 거부, 벽까지 도달은 `wall`), 도판 sidecar에 동봉해 검증기가 재파싱, `main.py --plate PROJECT.amr SPEC.json OUT.svg`로 프로젝트에서 도판 재제작(탁본은 recipe 재계산, 채색·점묘 픽셀은 거부) | - | engineering contract | native | `src/core/drawing_sheet_spec.py`, `src/application/plate_from_spec.py` |
| 유물 상태 표기 | 3D 면 집합의 정규 run-length 저장 + 커밋 시점 6뷰 투영 경계 + 전 뷰 공백이면 fail closed | `[L8]` | library + engineering contract | native | `src/core/artifact_condition_annotation.py` (`annotation.condition.v1`) |
| 제작 기법 흔적 표기 | 상태 표기와 같은 면 집합 record에 닫힌 기법 어휘(테쌓기흔·지두흔·타날흔·물손질흔·목리조정흔); 도판에는 [K1][K2]의 표현대로 그린다: 지두흔은 누른 자리마다 타원 또는 실측자가 고른 뒤집힌 U, 테쌓기흔은 이음선이되 내면에서 읽으므로 언제나 단면 반쪽, 목리조정흔은 방향 있는 평행선 군집, 물손질흔은 평행선, 타날흔은 입면에 긋지 않고 탁본에 맡김. 획은 seed로 결정적 | `[K1]`, `[K2]`, `[K3]`, `[L8]` | library + engineering contract | native | `src/core/artifact_technique_annotation.py` (`annotation.technique.v1`), `src/core/drawing_marks.py` |
| 실측 도판 | 여러 record를 ISO 용지에 1:N 배치 + 파생 축척 행 + 축척바·제목란 + 넘치면 fail closed | - | engineering contract | native | `src/core/drawing_sheet.py`, `src/core/drawing_svg.py` |
| 단면 빗금 선택 | 잘린 면을 채울지는 실측이 아니라 실측자의 선택; preset의 `section_cut` 빗금만 끈 사용자 preset이 되어 정의 전체가 도면 provenance에 실림 | - | engineering contract | native | `src/core/drawing_style.py` (`user_preset(hatch_cut_faces=...)`) |
| 해석 정도와 고지 | 홈의 골은 실측 그대로 두고 두 능선만 자기 기복의 정한 몫만큼 밖으로 그림; 무엇이든 정하면 제목란에 `해석` 행이 찍히고, sidecar에 블록이 있는데 그 행이 없으면 오프라인 검증기가 거부 | - | engineering contract | native | `src/core/drawing_sheet.py` (`Interpretation`), `src/core/artifact_vector_export.py` |
| 탁본 농담 | 먹의 진하기를 연·중·진 한 칸으로; 접촉 모델은 접촉 먹 농담, 높이 모델은 먹 농도와 종이 기저 농담으로 — 같은 칸이 두 모델에서 뜻이 달라 모델별 표 | - | engineering contract | native | `src/core/artifact_rubbing_extractor.py` (`RUBBING_TONE_SETTINGS`) |
| 1:1 vector export | exact-mm SVG + canonical provenance + 재렌더 바이트 비교 | - | engineering contract | native | `src/core/artifact_vector_export.py` |
| 1:1 탁본 export | 결정적 GA8 PNG + `pHYs` 물리 크기 + sidecar 결합 | - | engineering contract | native | `src/core/artifact_rubbing_export.py`, `src/core/canonical_png.py` |
| 검증형 기와 전개 export | content-addressed canonical binary + flat OBJ + physical-mm SVG + public provenance | - | engineering contract | native | `src/core/artifact_tile_unwrap_export.py` |
| 완료 실측 묶음 | 3/6/6 record를 자식 패키지 15개와 aggregate manifest로 원자 게시 | - | engineering contract | native | `src/core/artifact_survey_export.py` |
| 오프라인 검증 | `.amr`과 네 export 종류의 hash·단위·Align·record·QC 재검증 | - | engineering contract | native | `src/core/artifact_verification.py` |
| 깊이 픽 좌표 | float64 world → render origin 재기준 → 24-bit depth unproject → CPU ray/triangle | - | engineering contract | native | `src/gui/render_coordinates.py`, `src/core/artifact_surface_measurement.py` |
| 단면 위치 표시 | 절단면과 도형 평면이 만나는 직선을 그 도형 위에 일점쇄선으로 긋고 양끝에 A-A′; 유물 밖으로 종이 3 mm 더 나가고, 나란한 평면은 거부 | - | engineering contract (잠정) | native | `src/core/drawing_sheet.py` (`section_marks`) |
| 파편의 깨진 자리 | 실측자가 댄 쪽에서 도형 자신의 선을 종이 1.5 mm 앞에 멈추고 가로질러 아무것도 긋지 않음; 제목란 `파편` 행과 sidecar 블록이 함께 서고, 이름만 대고 자른 것이 없으면 거부 | - | engineering contract (잠정, 실측자 지시) | native | `src/core/drawing_sheet.py` (`sherd_breaks`) |
| 판독 다섯 가지 | 도판이 그리는 꺾임·홈·능선·뒷면 실루엣·양각 음영을 한 가지 호출로 읽고 실측자 이름으로 기록; 판독의 뜻과 거부는 전부 core의 것이고 이 층은 규칙을 더하지 않는다 | - | engineering contract | native | `src/application/artifact_readings.py`, `src/gui/readings_panel.py` |
| 스튜디오 배경 | 유물을 굴려 보는 방 — 바닥·하늘 그러데이션, 실제 밀리미터 격자, 원점에서 교차하는 세 축, 바닥 그림자, 실측자 왼쪽 어깨 위의 광원. **화면에만 있고 record·export·도판에는 들어가지 않는다** (세로 높이자는 실측자 판단으로 뺐다, 2026-09-09) | - | engineering contract | native | `src/gui/studio_backdrop.py`, `src/gui/viewport_3d.py` |
| 패널은 눌리지 않는다 | 세부 패널을 도크에 스크롤로 넣어, 도크가 짧아도 패널이 제 높이를 지키고 도크가 스크롤한다. 여섯 기준 시점 버튼은 최소 높이를 못박아 눌리지 않는다. 세부 패널을 다 열어도 창 최소 높이 1607 px → 419 px | - | engineering contract | native | `app_interactive.py` (`_scrolled`) |

---

## 검토용 legacy 경로

아래 행의 산출물은 ArtifactDocument session이 열려 있으면 파일로 내보낼 수 없습니다(`app_interactive.py`의 `_reject_native_legacy_*`). 1:1 측정 결과가 아니라 화면 검토용입니다.

| Feature | Core approach | Reference IDs | Type | Status | Main code |
|---|---|---|---|---|---|
| ARAP 기반 일반 펼침 | ARAP optimization + cotangent weights | `[R1]`, `[R4]` | paper + heuristic | legacy | `src/core/flatten_models_arap.py` |
| LSCM 기반 초기 전개 | least-squares conformal parameterization | `[R2]` | paper | legacy | `src/core/flatten_models_arap.py` |
| 면적 기반 전개 | Tutte/LSCM blend + global scale normalization | `[R2]`, `[R3]` | paper + heuristic | legacy | `src/core/flattener.py`, `src/core/flatten_models_arap.py` |
| 원통 추적 전개 | axis candidate scoring + circle-fit center + seam unwrap | `[R6]` | paper + heuristic | legacy | `src/core/flatten_models_cylindrical.py` |
| sectionwise 펼침 (legacy flattener 경유) | longitudinal axis + repeated cross-sections + local circular fits | `[R4]`, `[R6]` | paper + heuristic | legacy | `src/core/flatten_models_sectionwise.py`, `src/core/flattener.py` |
| 기와 추천 정책 | tile confidence + 장축비 + 단면 반복성 + 단면/와통 힌트 반영 | - | heuristic | legacy | `src/core/flatten_policy.py` |
| sectionwise fallback | distortion/section quality gate 후 대체 mode 연결 | - | heuristic | legacy | `src/core/flatten_models_sectionwise.py`, `src/core/flattener.py` |
| flatten size stabilization | pathological scale ratio guard + metadata | - | heuristic | legacy | `src/core/flatten_utils.py` |
| distortion summary | per-face distortion aggregation for UI/guard | `[R1]`, `[R4]` | paper + heuristic | legacy | `src/core/flatten_metrics.py` |
| 실시간 단면 분석 | plane/mesh intersection through trimesh-based slicing | `[L3]` | library + heuristic | legacy | `src/core/mesh_slicer.py`, `app_interactive.py` |
| 와통/반경 추정 | 2D circle fitting on sampled profiles | `[R6]` | paper + heuristic | legacy | `src/core/tile_profile_fitting.py`, `app_interactive.py` |
| 기록면 선택/분리 | visibility/depth/topology propagation + labeling | `[R5]`, `[R7]` | paper + heuristic | legacy | `src/core/surface_separator.py` |
| rubbing-like 판독 시각화 | 단일 mesh의 normal/curvature/height 파생 강조 + contrast/strength 조절 (RTI/PTM 구현 아님) | `[A1]`, `[A2]`, `[A3]`, `[L2]` | inspiration + library + heuristic | legacy | `src/core/surface_visualizer.py` |
| 기록면 검토 시트 | flattened output + preview composition | `[A2]`, `[A4]` | inspiration + heuristic | legacy | `src/core/recording_surface_review.py`, `src/core/rubbing_sheet_exporter.py` |
| Legacy SVG 산출물 | permissive flattened geometry export for review workflows | - | heuristic | legacy | `src/core/flattened_svg_exporter.py` |

---

## Practical interpretation

- `heuristic`가 포함된 항목은 현장 데이터 안정성과 GUI 반응성을 위해 조정된 부분이 있습니다.
- `inspiration`은 직접 같은 알고리즘을 구현했다기보다, 판독 중심의 문제 설정과 UI 방향에 영향을 준 경우입니다.
- 특히 `[A1]`-`[A4]`(PTM/RTI)는 문제의식 참고이며, 다중 광원 촬영·픽셀별 반사 모델 적합·`.ptm`/`.rti` 입출력은 native·legacy 어느 경로에도 없습니다. RTI 격차 판정은 [`docs/COMPETITIVE_GAP_ANALYSIS.md`](COMPETITIVE_GAP_ANALYSIS.md)를 따릅니다.
- `sectionwise`, `tile recommendation`, `digital rubbing`은 ArchMeshRubbing의 제품 정체성에 맞게 조합된 레이어입니다.
