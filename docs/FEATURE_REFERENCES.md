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

---

## Feature → Reference → Code

| Feature | Core approach | Reference IDs | Type | Main code |
|---|---|---|---|---|
| ARAP 기반 일반 펼침 | ARAP optimization + cotangent weights | `[R1]`, `[R4]` | paper + heuristic | `src/core/flatten_models_arap.py` |
| LSCM 기반 초기 전개 | least-squares conformal parameterization | `[R2]` | paper | `src/core/flatten_models_arap.py` |
| 면적 기반 전개 | Tutte/LSCM blend + global scale normalization | `[R2]`, `[R3]` | paper + heuristic | `src/core/flattener.py`, `src/core/flatten_models_arap.py` |
| 원통 추적 전개 | axis candidate scoring + circle-fit center + seam unwrap | `[R6]` | paper + heuristic | `src/core/flatten_models_cylindrical.py` |
| sectionwise 펼침 | longitudinal axis + repeated cross-sections + local circular fits | `[R4]`, `[R6]` | paper + heuristic | `src/core/flatten_models_sectionwise.py` |
| 기와 추천 정책 | tile confidence + 장축비 + 단면 반복성 + 단면/와통 힌트 반영 | - | heuristic | `src/core/flatten_policy.py` |
| sectionwise fallback | distortion/section quality gate 후 대체 mode 연결 | - | heuristic | `src/core/flatten_models_sectionwise.py`, `src/core/flattener.py` |
| flatten size stabilization | pathological scale ratio guard + metadata | - | heuristic | `src/core/flatten_utils.py` |
| distortion summary | per-face distortion aggregation for UI/guard | `[R1]`, `[R4]` | paper + heuristic | `src/core/flatten_metrics.py` |
| 실시간 단면 분석 | plane/mesh intersection through trimesh-based slicing | `[L3]` | library + heuristic | `src/core/mesh_slicer.py`, `app_interactive.py` |
| 와통/반경 추정 | 2D circle fitting on sampled profiles | `[R6]` | paper + heuristic | `src/core/tile_profile_fitting.py`, `app_interactive.py` |
| 기록면 선택/분리 | visibility/depth/topology propagation + labeling | `[R5]`, `[R7]` | paper + heuristic | `src/core/surface_separator.py` |
| rubbing-like 판독 시각화 | normal/curvature/height derived enhancement + contrast/strength control | `[A1]`, `[A2]`, `[A3]`, `[L2]` | inspiration + library + heuristic | `src/core/surface_visualizer.py` |
| 기록면 검토 시트 | flattened output + preview composition | `[A2]`, `[A4]` | inspiration + heuristic | `src/core/recording_surface_review.py`, `src/core/rubbing_sheet_exporter.py` |
| SVG 산출물 | flattened geometry export for report/drawing workflows | - | heuristic | `src/core/flattened_svg_exporter.py` |

---

## Practical interpretation

- `heuristic`가 포함된 항목은 현장 데이터 안정성과 GUI 반응성을 위해 조정된 부분이 있습니다.
- `inspiration`은 직접 같은 알고리즘을 구현했다기보다, 판독 중심의 문제 설정과 UI 방향에 영향을 준 경우입니다.
- `sectionwise`, `tile recommendation`, `digital rubbing`은 ArchMeshRubbing의 제품 정체성에 맞게 조합된 레이어입니다.
