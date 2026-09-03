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
| 6면 외곽선 | 모든 투영 삼각형의 fixed-grid 다각형 합집합 (rasterisation·convex hull 없음) | `[L8]` | library + engineering contract | native | `src/core/artifact_outline_extractor.py` (`OUTLINE_ALGORITHM = archmeshrubbing.projected_triangle_union`) |
| 단면선 | 명시 평면과 canonical-mm 삼각형의 정확 교차 + endpoint 스티칭, 모호한 경우 fail closed | - | engineering contract | native | `src/core/artifact_vector_extractor.py` |
| 외곽선 topology 검증 | ring simple/area, hole ownership, 성분 비중첩 증명 | `[L8]` | library | native | `src/core/artifact_outline_topology.py` |
| 정식 기와 전개 record | explicit canonical axis + exact face selection + 1 µm quantization + no-fallback quality gate | `[R4]`, `[R6]` | paper + heuristic | native | `src/core/artifact_tile_unwrap_extractor.py`, `src/core/artifact_tile_unwrap_record.py` |
| sectionwise 펼침 | longitudinal axis + repeated cross-sections + local circular fits | `[R4]`, `[R6]` | paper + heuristic | native | `src/core/flatten_models_sectionwise.py` |
| distortion summary | per-face distortion aggregation for QC gate | `[R1]`, `[R4]` | paper + heuristic | native | `src/core/flatten_metrics.py` |
| 원 맞춤 지름 | PCA best-fit plane 위 정규화 대수 Kasa 원 | `[R6]` | paper | native | `src/core/artifact_surface_measurement.py` (`SURFACE_DIAMETER_FIT_POLICY`) |
| 표면 거리 | source triangle + 10억 분율 barycentric anchor의 3D Euclidean chord (측지 거리 아님) | - | engineering contract | native | `src/core/artifact_surface_measurement.py` |
| 표면적·체적 | 1 µm 격자 양자화 표면적 + topology-gated exact-rational 체적 (convex hull fallback 없음) | - | engineering contract | native | `src/core/artifact_geometry_metrics.py` |
| 도면 선 종류 표현 | 닫힌 선 종류 어휘 + 종이 mm preset(해시 봉인) + role→선 종류 매핑 + 레이어드 SVG·단면 해칭 | - | engineering contract | native | `src/core/drawing_style.py`, `src/core/artifact_vector_export.py` |
| 실측 도판 | 여러 record를 ISO 용지에 1:N 배치 + 파생 축척 행 + 축척바·제목란 + 넘치면 fail closed | - | engineering contract | native | `src/core/drawing_sheet.py`, `src/core/drawing_svg.py` |
| 1:1 vector export | exact-mm SVG + canonical provenance + 재렌더 바이트 비교 | - | engineering contract | native | `src/core/artifact_vector_export.py` |
| 1:1 탁본 export | 결정적 GA8 PNG + `pHYs` 물리 크기 + sidecar 결합 | - | engineering contract | native | `src/core/artifact_rubbing_export.py`, `src/core/canonical_png.py` |
| 검증형 기와 전개 export | content-addressed canonical binary + flat OBJ + physical-mm SVG + public provenance | - | engineering contract | native | `src/core/artifact_tile_unwrap_export.py` |
| 완료 실측 묶음 | 3/6/6 record를 자식 패키지 15개와 aggregate manifest로 원자 게시 | - | engineering contract | native | `src/core/artifact_survey_export.py` |
| 오프라인 검증 | `.amr`과 네 export 종류의 hash·단위·Align·record·QC 재검증 | - | engineering contract | native | `src/core/artifact_verification.py` |
| 깊이 픽 좌표 | float64 world → render origin 재기준 → 24-bit depth unproject → CPU ray/triangle | - | engineering contract | native | `src/gui/render_coordinates.py`, `src/core/artifact_surface_measurement.py` |

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
