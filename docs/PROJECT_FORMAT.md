# AMR Project Format

이 문서는 ArchMeshRubbing 프로젝트 파일의 현재 저장 계약을 설명한다. AMR v2의 우선순위는 기능 수보다 **기존 연구 상태를 손상시키지 않는 저장**, **원본 바이트 식별**, **검증할 수 없는 상태의 명시적 표시**다.

## 컨테이너

`.amr`는 ZIP 컨테이너이며 다음 두 파일을 반드시 포함한다.

| 멤버 | 역할 |
|---|---|
| `project.json` | 버전 envelope와 선택된 `legacy_ui_state` 또는 `artifact_document` payload |
| `checksums.json` | `project.json` 및 향후 멤버의 SHA-256 목록 |

AMR 파일은 확장자와 관계없이 일반 JSON으로 fallback하지 않는다. 개발·마이그레이션용 일반 JSON은 명시적인 `.json` 파일만 허용한다. 따라서 잘린 ZIP을 JSON으로 오인해 실행하지 않는다.

### `project.json` envelope

```json
{
  "format": "archmeshrubbing_project",
  "version": 2,
  "payload_type": "artifact_document",
  "payload_schema_version": "1.0.0",
  "saved_at": "2026-07-11T12:00:00Z",
  "meta": {},
  "state": {
    "schema_version": "1.0.0",
    "document_id": "artifact:example",
    "software_version": "0.1.0",
    "source_assets": [],
    "geometry_revisions": [],
    "source_metadata_revisions": [],
    "align_revisions": [],
    "active_source_metadata_revision_id": null,
    "active_align_revision_id": null,
    "records": [],
    "extensions": {}
  }
}
```

AMR v2 container는 payload 종류와 payload schema를 분리한다.

- `payload_type="artifact_document"`, `payload_schema_version="1.0.0"`은 `state`에 `ArtifactDocument 1.0`의 전체 manifest를 저장한다. `save_artifact_project()`와 `load_artifact_project()`가 이 경계의 권위 API다. envelope 버전, payload schema, 내부 `state.schema_version`, graph reference와 checksum이 모두 일치해야 실행 가능한 문서로 반환한다.
- `payload_type="legacy_ui_state"`는 기존 GUI snapshot의 별도 호환 payload다. `save_project()`와 `load_project()`가 처리한다.

두 API는 payload 종류를 암묵적으로 교차 로드하거나 migration하지 않는다. legacy 상태에서 단위·geometry identity·Align revision을 추정해 `ArtifactDocument`로 승격하지 않으며, artifact payload를 legacy mutable UI state로 강등하지도 않는다.

### `checksums.json`

```json
{
  "algorithm": "sha256",
  "files": {
    "project.json": "<64 lowercase hexadecimal characters>"
  }
}
```

체크섬은 파일 손상이나 변조를 탐지하기 위한 무결성 값이다. 공개키 서명이나 신뢰 기관의 인증이 아니므로 제작자·진본성·작성 시점을 증명한다고 표현해서는 안 된다.

## 외부 원본 식별

파일에서 import한 각 mesh는 기존 `path`와 `source_scale_factor`를 유지하면서 `source`를 추가한다.

```json
{
  "mesh": {
    "path": "/data/artifact.ply",
    "source_scale_factor": 1.0,
    "source": {
      "binding_status": "captured_at_import",
      "parse_format": "ply",
      "identity": {
        "schema_version": 1,
        "id": "sha256:<digest>",
        "kind": "external_file",
        "identity_scope": "primary_file_bytes",
        "sha256": "<digest>",
        "size_bytes": 123456,
        "mtime_ns": 1780000000000000000,
        "original_name": "artifact.ply",
        "format": "ply"
      }
    }
  }
}
```

- 파일 내용 identity는 `sha256 + size_bytes`이며, artifact source binding은 여기에 `identity_scope`까지 같은지 확인한다. 현재 권위 tuple은 `(primary_file_bytes, sha256, size_bytes)`다.
- `path`, `mtime_ns`, `original_name`, `format`은 탐색·표시용 hint다.
- 경로와 이름이 달라도 SHA-256과 크기가 같으면 `relocated=true`인 verified 결과로 열 수 있다.
- `parse_format`은 파일명이 바뀐 동일 원본도 처음과 같은 parser로 해석하기 위한 import recipe다. 후보 파일의 suffix와 별도로 보존한다.
- 같은 크기라도 SHA-256이 다르면 mismatch이며 face ID, cutline, 기록면 데이터 등을 replacement mesh에 적용하지 않는다.
- hash와 mesh parser는 동일한 열린 file descriptor를 사용한다. 경로를 다시 열어 다른 파일의 geometry에 이전 hash를 붙이지 않는다.
- 저장 시 디스크 파일을 다시 hash하지 않는다. import 당시 geometry와 함께 보관한 immutable identity만 직렬화한다.

Artifact payload를 다시 열 때 parser 선택도 검증 대상이다. `GeometryRevision.import_recipe.format`에 최초 parser format을 저장하고, resolved source의 현재 suffix보다 이 값을 우선해 `MeshLoader.load(..., source_format=saved_format)`로 decode한다. loader는 같은 열린 descriptor에서 raw byte fingerprint와 geometry를 얻는다. 이후 `ArtifactSession.bind_loaded_document()`는 다음을 모두 만족할 때만 문서와 mesh를 결합한다.

1. 새로 계산한 `identity_scope`, `sha256`, `size_bytes`가 `SourceAsset`과 일치한다.
2. 실제 사용한 `MeshData.source_format`이 저장된 `import_recipe.format`과 일치한다.
3. 저장된 `geometry_hash_scope`로 decode 결과를 다시 hash한 값이 `GeometryRevision.geometry_sha256`와 일치한다.

저장된 parser format이 없거나 다른 parser로 열렸거나 source/geometry digest가 다르면 materialization 전에 실패한다. 따라서 이름과 확장자가 바뀐 동일 파일은 복원할 수 있지만, 단지 suffix가 같다는 이유로 다른 바이트나 다른 decode 결과를 받아들이지 않는다.

`binding_status`의 현재 값:

| 값 | 의미 |
|---|---|
| `captured_at_import` | raw primary file을 hash한 뒤 동일 descriptor에서 geometry를 import했다. |
| `legacy_unverified` | v1 기록과 현재 source의 관계를 사후에 증명할 수 없다. 현재 hash가 생겨도 기존 기록을 소급해 verified로 바꾸지 않는다. |
| `generated_ephemeral` | 외부 원본 경로가 없는 런타임 생성 mesh다. 현재 v2만으로 새 프로세스에서 복원할 수 있다는 뜻이 아니다. |

### M0-1 identity 범위 제한

`identity_scope=primary_file_bytes`는 **주 파일 하나의 바이트만** 포함한다.

- OBJ의 MTL·텍스처
- glTF의 외부 `.bin`·이미지
- 기타 sidecar 또는 linked asset

위 파일은 아직 identity와 portable package에 포함되지 않는다. 따라서 이 단계의 `verified`는 “주 파일 바이트가 일치한다”는 뜻이며, 모든 렌더링 의존 asset이나 전체 geometry package가 완전하다는 뜻이 아니다. embedded asset과 sidecar manifest는 후속 버전 범위다.

## ArtifactDocument 1.0 도메인 계약

`src/core/artifact_document.py`는 Qt·OpenGL에 의존하지 않는 immutable 문서 모델을 제공한다. 스키마 버전은 `1.0.0`이며, 대략적인 구조는 다음과 같다.

```text
ArtifactDocument
├── SourceAsset[]
├── GeometryRevision[]
├── SourceMetadataRevision[]
├── AlignRevision[]
├── active_source_metadata_revision_id
├── active_align_revision_id
└── DerivedRecord[]
```

이 모델은 AMR v2 container 버전과 별개의 payload schema다. 현재 writer/reader는 `payload_type="artifact_document"`, `payload_schema_version="1.0.0"`으로 이 manifest를 durable하게 저장하고 복원한다. 다만 manifest round-trip만으로 외부 source의 재검증이나 GPU 장면의 시각적 재현까지 증명되는 것은 아니다.

### 권위 경계

| 모델 | 권위 내용 |
|---|---|
| `SourceAsset` | raw asset의 SHA-256, 바이트 크기, media type, 원래 이름, `asset_ref`, `role=primary_mesh`, identity 범위 |
| `GeometryRevision` | 결정적 import 결과의 별도 geometry hash·hash scope, source asset ID, import recipe, topology-map reference, QC |
| `SourceMetadataRevision` | geometry의 단위·축 매핑·handedness와 `source_to_canonical_mm`; parent revision |
| `AlignRevision` | confirmed metadata 위에 적용할 proper rigid 4×4 행렬, parent revision, recipe, QC |
| `DerivedRecord` | geometry/Align revision에 묶인 cutline·outline·rubbing 등의 payload reference, recipe/hash, selection hash, dependency, QC, lifecycle |

SourceAsset hash와 GeometryRevision hash는 같은 개념이 아니다. 전자는 정확한 입력 바이트, 후자는 decode·triangulation·sanitize 등 import recipe를 거친 geometry identity다. 1.0의 geometry hash scope는 `positions-f64le+triangles-i32le/v1`로 명시하여 바이트 인코딩과 topology 범위를 hash와 함께 고정한다. 대형 vertex·face·point 배열은 manifest JSON에 직접 넣지 않고 reference로 연결한다.

#### geometry hash V1 framing

`canonical_geometry_sha256()`의 입력 framing은 다음 byte sequence로 고정한다.

```text
"archmeshrubbing.geometry\0"
|| ASCII("positions-f64le+triangles-i32le/v1") || "\0"
|| uint64le(vertex_count)
|| C-order little-endian float64[vertex_count, 3] positions
|| uint64le(face_count)
|| C-order little-endian int32[face_count, 3] triangles
```

- vertex는 finite float64로 canonicalize하고 `-0.0`을 `+0.0`으로 정규화한다.
- face는 음수가 아니고 존재하는 vertex를 참조하는 정수여야 하며 int32 범위를 넘을 수 없다.
- vertex 순서, face 순서, triangle winding은 identity에 포함된다. 이를 재정렬하면 같은 표면처럼 보여도 다른 geometry revision이다.
- count와 domain separator를 포함하므로 서로 다른 배열 경계를 같은 연속 bytes로 오인하지 않는다.
- 이 framing은 raw source 파일을 hash하지 않는다. 저장된 import recipe로 decode한 source-space `MeshData.vertices`와 `MeshData.faces`를 hash한다.

다른 framing을 같은 scope 이름으로 사용해서는 안 된다. framing 변경은 새 `geometry_hash_scope` 버전과 migration 정책이 필요하다.

### canonical millimeter와 행렬 규약

모든 학술적 계산의 canonical 단위는 millimeter다.

```text
p_world_mm = AlignRevision.matrix4x4
           @ SourceMetadataRevision.source_to_canonical_mm
           @ p_geometry_source
```

- 행렬은 float64 4×4, JSON에서 row-major로 기록하고 column vector에 적용한다.
- confirmed metadata는 `mm | cm | m`, 축 매핑, handedness를 명시해야 한다. `source_to_canonical_mm`은 해당 단위 scale과 signed axis mapping만 정확히 포함하며 translation을 포함하지 않는다.
- unconfirmed metadata는 canonical materialization과 파생 작업 context 생성을 차단한다.
- Align은 proper rigid transform만 허용한다. scale, shear, reflection, perspective, NaN/Infinity를 거부하며 단위 보정을 Align에 숨기지 않는다.
- world-space delta로 새 Align을 만들 때는 `A_new = delta @ A_parent`로 합성한다. Euler로 다시 분해해 저장하지 않는다.
- 새 source에 생성되는 `recipe.kind="initial_identity"` Align은 canonical materialization용 baseline이며 사용자 정위치 확정이 아니다. 첫 명시적 확정은 변화량이 0이어도 non-initial immutable child revision을 append한다.

기존 viewport의 Euler adapter는 다음 한 규약으로 고정한다.

```text
M_scene = T @ Rx @ Ry @ Rz @ S
```

OpenGL의 `glTranslate → glRotate(X) → glRotate(Y) → glRotate(Z) → glScale` 호출과 같은 순서다. SciPy 표기로는 intrinsic uppercase `"XYZ"`에 해당하며 lowercase `"xyz"`는 다른 회전이다. `scene_trs_matrix()`가 이 legacy adapter의 단일 권위며, `S`는 기존 scene 표시 호환을 위한 값일 뿐 학술적 Align revision의 일부가 아니다.

### append, activate, operation context

모든 revision과 record는 immutable value다. 수정은 기존 항목을 덮어쓰지 않고 새 revision/record를 append한 다음 explicit active pointer를 옮긴다.

- metadata revision을 activate하면 active Align은 해제된다.
- Align revision을 activate하면 해당 Align과 그것이 참조하는 metadata pointer가 하나의 context로 함께 복원된다.
- 새 Align은 이전 record를 삭제하거나 재기록하지 않는다.
- background 작업은 시작 시 `OperationContext`와 `ArtifactSession` identity에 source asset, geometry, metadata, Align, recipe hash, optional selection hash를 캡처한다. application commit은 captured session identity와 `state_version`/`authority_epoch`를 compare-and-swap 검증하므로 authority가 바뀐 늦은 결과를 새 Align에 붙이지 않는다.
- derived-record session update는 기존 source/metadata/Align/record를 제거하거나 다시 쓸 수 없고, 추가할 `expected_new_record_ids`의 정렬된 집합과 실제 ID 차집합이 정확히 같아야 한다. 같은 Align에서 다른 record가 먼저 추가된 경우에도 오래된 document로 그 기록을 덮지 않는다.
- core field의 unknown key와 duplicate JSON key, NaN/Infinity를 거부한다. 확장 데이터는 `org.example:key` 형태의 namespaced `extensions`에만 둔다.

canonical JSON은 collection ID, mapping key를 결정적으로 정렬하고 UTF-8·LF·compact separator를 사용한다. 같은 문서는 입력 순서와 관계없이 같은 bytes와 SHA-256을 만들어야 한다. 외부 계약은 `schemas/artifact_document-1.0.0.schema.json`, canonical byte golden은 `tests/fixtures/projects/artifact_document_1_0_0.json`에 버전별로 고정한다.

### DerivedRecord freshness

`lifecycle_status` (`draft | ready | failed`)는 저장된 작업 상태이고 freshness는 active revision graph에서 매번 파생한다.

| freshness | 의미 |
|---|---|
| `fresh` | record의 metadata/Align context가 active context와 같고 모든 dependency가 fresh다. |
| `stale_alignment` | record가 active Align과 다른 Align을 참조한다. |
| `stale_metadata` | record의 Align이 active metadata와 다른 metadata를 참조한다. |
| `missing_dependency` | dependency ID가 문서에 없다. |
| `blocked_dependency` | dependency가 stale이거나 lifecycle `failed`다. |
| `invalid` | schema·reference·geometry 검증 실패를 표현하는 경계 상태다. 현재 core parser는 invalid document를 실행 가능한 문서로 반환하지 않는다. |

이전 Align을 다시 activate하면 그 Align에 묶인 유효한 record는 다시 `fresh`가 될 수 있다. dependency cycle은 거부하며, 서로 다른 Align의 record를 암묵적으로 연결하는 것도 거부한다. cross-revision dependency에는 명시적 transform record가 필요하다.

### M0-3 이후 한 artifact native workflow

M0-3에서 시작한 durable core와 현재 native GUI/application 경계는 다음과 같다.

- `ArtifactDocument 1.0`을 AMR v2의 `artifact_document` payload로 strict serialize/load하고 `project.json` checksum과 원자적 staged validation에 포함한다.
- `ArtifactSession`이 원본 source-space mesh의 immutable snapshot, source byte identity, geometry identity와 document를 하나의 검증된 context로 묶는다.
- headless `ArtifactSceneAdapter`가 항상 immutable source에서 시작해 active `Align @ SourceMetadata`를 적용한 새 float64 world-mm `MeshData`를 만든다. source vertex를 mutate하거나 centroid로 recenter하지 않고, document/revision/hash/matrix snapshot이 바뀐 늦은 결과를 거부한다.
- native application은 정확히 한 artifact를 다루며 `ArtifactWorkbench.snapshot.session.document`를 source of truth로 둔다. MainWindow의 session field는 이행 중 compatibility mirror다. 사용자 Open은 단위·축·handedness 확인 후 ticketed load로 들어가고, `initial_identity` baseline에서는 `ALIGN_REQUIRED`에 머문다. 이동·회전은 preview일 뿐이며 첫 정위치 확정은 변화량이 0이어도 proper-rigid child Align revision을 append한 뒤 immutable source에서 장면을 다시 materialize한다. parent activation으로 baseline에 돌아가면 측정·내보내기가 다시 잠긴다. scale은 metadata 영역이므로 native Align preview에서 차단한다.
- artifact project reopen은 외부 source를 saved parser/unit으로 CPU staging에서 다시 검증한다. `ArtifactWorkbench`는 한 pending Open ticket과 `state_version`/`authority_epoch`를 검증하고, candidate projection을 준비한 뒤 scene notification 동안에만 tentative authority로 활성화한다. scene swap 성공 후 finalize하고, 실패하면 이전 session·scene·project path로 rollback한다. observer는 finalize 전 candidate를 보지 않는다.
- rollback·scene 복원·finalize 자체가 실패해 application authority와 live scene의 일치를 증명할 수 없으면 fatal authority 상태로 전환한다. 이 상태에서는 ordinary Save target을 해제하고 저장·실측·내보내기를 모두 거부하며, 검증된 새 Open만 정상 authority를 회복한다.
- artifact save는 active document만 쓰기 전에 정확히 한 projection, current snapshot, identity preview, source에서 재현한 vertices/faces 일치, destructive bake 부재를 확인한다. `ALIGN_REQUIRED` document 자체는 보존할 수 있지만 Cutline/Outline/Digital Rubbing 계산과 vector/rubbing export는 명시적 Align 전까지 차단한다. 아직 `DerivedRecord`로 승격되지 않은 cutline·선택·기록면·평가 등의 결과가 하나라도 있으면 누락한 채 저장하지 않고 fail closed한다.
- Cutline/Outline/Digital Rubbing은 application layer가 canonical recipe, projection context, exact record ID와 result capability를 소유한다. worker는 session을 commit하지 않고 computation만 반환하며, 완료 시 captured document가 current document의 immutable ancestor이고 active source/metadata/Align/matrix가 같을 때만 current session에 rebase하여 expected record ID 하나를 publish한다. Align/Open finalize 뒤 늦은 결과는 되살아나지 않는다. pending Open이나 rollback 가능한 scene 준비 실패는 계산 결과와 예약 ID를 보존해 명시적으로 재시도하며, 그동안 저장과 새 실측을 차단한다. Rubbing은 Workbench 공유 누적 peak-memory budget, 대형 UV/texture 복사 비용의 사전 산정과 실행 exactly-once를 적용한다.
- `Open → Align commit → save → independent-process load → source rebind → materialize` 왕복을 별도 프로세스에서 검증한다.

`tests/test_artifact_new_process_roundtrip.py`의 차단 게이트는 다음 순서를 실제로 수행한다.

1. 프로세스 A가 PLY source를 같은 descriptor에서 hash·parse하고 cm metadata와 비자명한 pivot Align revision을 만든 뒤 artifact payload로 저장한다.
2. source를 다른 경로로 옮기고 suffix를 `.raw-scan`으로 바꾼다.
3. 다른 PID의 프로세스 B가 artifact payload를 strict load하고 저장된 parser format과 metadata unit으로 source를 다시 연다.
4. 프로세스 B가 raw source SHA-256·크기, decode geometry SHA-256을 새로 계산하고 검증된 session에 bind한 뒤 world-mm geometry를 materialize한다.
5. 두 프로세스의 source SHA-256·크기, geometry SHA-256, active Align ID·float64 matrix, parser format·unit, world vertex를 비교한다.

이 게이트는 canonical CPU projection과 durable payload의 왕복 증거다. `load_artifact_project()` 자체가 외부 파일을 탐색하거나 parser를 실행하는 것은 아니며, source resolution과 saved-parser reopen은 session/GUI orchestration이 수행한다. native 한-artifact workflow가 document source of truth와 scene-swap rollback을 사용하더라도 실제 GPU driver가 그린 프레임의 정밀도·시각적 동일성까지 이 테스트가 증명하지는 않는다.

## M0-4 canonical vector record

Cutline과 Outline의 측정 권위는 화면 스크린샷, OpenGL camera, `cut_section_world` 배치용 tape, ROI convex hull에 두지 않는다. `VectorGeometryPayload 1.0`이 단일 권위이며 machine-readable 계약은 `schemas/vector_payload-1.0.0.schema.json`에 있다.

```text
VectorGeometryPayload
├── kind: cutline | outline
├── coordinate_space: canonical_mm_planar/v1
├── frame
│   ├── origin_world_mm
│   ├── u_axis_world
│   ├── v_axis_world
│   └── normal_world
└── paths[]
    ├── id
    ├── role
    ├── closed
    └── points_mm[][2]
```

- frame은 finite, unit-length, mutually orthogonal, right-handed이며 `u × v = normal`이어야 한다.
- Cutline path role은 `section`만 허용하고 open/closed path를 모두 보존한다.
- Outline은 closed path만 허용하고 `exterior | hole` role을 구분한다. exterior는 CCW, hole은 CW로 canonicalize한다.
- closed path는 사전식 최소점을 시작점으로 두고, open path는 전체 forward/reverse tuple 중 작은 방향을 택한다. path collection도 role·closure·bounds·coordinates·ID로 정렬한다.
- finite 좌표, path/point 수, payload byte 수에 상한을 두며 빈 결과, 열린 경로의 같은 시작/끝, 면적 0인 폐곡선을 거부한다.

payload semantic bytes와 recipe semantic bytes는 **RFC 8785 JSON Canonicalization Scheme**으로 만든다. 따라서 `0.0`, `-0.0`, exponent 표기처럼 언어별 JSON printer가 다를 수 있는 값을 CPython 문자열 형식에 묶지 않는다. semantic SHA-256은 JCS bytes에 적용하며 cross-language number golden은 `tests/test_canonical_json.py`에 고정한다. RFC 8785의 I-JSON 범위를 벗어난 non-finite number와 unsafe integer는 거부한다.

ArtifactDocument 1.0에는 새 first-class payload field를 추가하지 않는다. 알려진 record type은 다음 bounded namespaced extension을 사용한다.

```json
{
  "type": "vector.cutline.v1",
  "geometry_ref": "urn:archmeshrubbing:vector-payload:sha256:<digest>",
  "extensions": {
    "org.archmeshrubbing:vector-payload-v1": {
      "byte_length": 1234,
      "media_type": "application/vnd.archmeshrubbing.vector+json",
      "payload": {},
      "schema_version": "1.0.0",
      "sha256": "<RFC-8785 semantic digest>"
    }
  }
}
```

`vector.cutline.v1`과 `vector.outline.v1`만 이 계약의 알려진 type이다. load/save/export 때 extension 존재, byte length, media type, schema, semantic hash, geometry ref, kind/type, recipe의 `kind + algorithm + algorithm_version`, payload에서 다시 계산한 모든 QC를 재검증한다. 현재 inline payload는 최대 16 MiB, 4,096 paths, 250,000 points다. 향후 ZIP payload member로 이동하더라도 같은 JCS payload digest와 geometry ref를 유지한다.

### Cutline v1 계산 정책

`artifact_vector_extractor.py`는 `ArtifactSession.materialize()`의 fresh canonical-mm triangle을 명시적 `PlanarFrame`과 교차한다. Trimesh section entity ordering이나 전역 tolerance를 권위 값으로 사용하지 않는다.

- signed plane distance classification과 endpoint stitch tolerance는 서로 다른 absolute millimeter recipe 값이다.
- 완전 coplanar face와 두 vertex가 평면 위인 edge는 v1에서 의미가 모호하므로 plane offset을 요구하며 실패한다.
- point tangent는 line result에서 제외하고 QC count에 남긴다. 그것만 존재하면 빈 성공이 아니라 typed failure다.
- graph degree가 3 이상인 branch, coincident segment, tolerance에서 붕괴하는 segment, tolerance보다 큰 stitch cluster, plane residual 초과를 거부한다.
- degree 1/2의 open mesh와 서로 분리된 여러 component는 손실 없이 여러 path로 저장한다.
- triangulation diagonal에서 생긴 tolerance 내 collinear point만 제거하고 smoothing, rasterization, sampling은 하지 않는다.
- 계산 시작 시 OperationContext와 projection snapshot을 고정한다. Align이 바뀐 늦은 결과는 새 Align에 바꿔 붙이지 않으며 current scene overlay로도 표시하지 않는다.

GUI의 native Top/Front/Right 명령은 각각 XY/Z, XZ/Y, YZ/X의 right-handed frame을 사용한다. 계산 결과를 DerivedRecord로 commit한 뒤 초록 overlay도 저장 payload에서 world 좌표로 다시 만든다. 기존 화면용 line layer 자동 저장과 ROI convex-hull Outline은 native 문서에서 차단한다.

### Outline v1 계산 정책

`artifact_outline_extractor.py`는 canonical world-mm mesh의 모든 face를 한 canonical view frame에 직교 투영하고, 사용자가 확정한 `precision_grid_mm`에서 polygon union을 수행한다. 연속 실수 공간의 무한 정밀도 외곽을 주장하지 않으며, **선언된 고정 격자에서의 결정적 외곽**을 산출한다.

| view | u | v | normal |
|---|---|---|---|
| top | +X | +Y | +Z |
| bottom | +X | -Y | -Z |
| front | +X | +Z | -Y |
| back | -X | +Z | +Y |
| right | +Y | +Z | +X |
| left | -Y | +Z | -X |

- backface, winding, depth, camera visibility로 face를 거르지 않는다. 3D-valid face를 모두 투영하고 정확히 2D area 0인 edge-on face만 QC count와 함께 제외한다.
- projected mm 좌표를 `grid`로 나눈 뒤 referenced minimum의 정수 index를 빼서 작은 local lattice로 만든다. GEOS에는 `grid_size=1.0`인 정수 격자를 전달하고 결과를 global index와 `grid mm`로 복원한다. 따라서 `1e9 mm` survey offset에서도 작은 유물의 overlay 계산을 절대좌표 polygon 연산에 맡기지 않는다.
- Shapely `set_precision(..., mode="valid_output")`으로 각 triangle을 격자에 맞춘 뒤 `union_all`한다. 한 격자보다 좁은 triangle, gap, island, hole은 사라지거나 합쳐질 수 있으며 `grid_collapsed_triangle_count`, component merge/split, unsnapped area comparison과 `axis<=grid/2; radial<=grid/sqrt(2)` 계약을 QC에 남긴다.
- 25,000 face씩 먼저 union하여 입력 Shapely 객체를 한꺼번에 보유하지 않고, chunk result를 고정 balanced-pairwise 순서로 합친다. 중간 polygon·coordinate 수, 입력 vertex·face 수, lattice index, 최종 path·point·payload byte 수를 제한하며 초과 시 sampling이나 decimation으로 성공시키지 않는다.
- 최종 Polygon/MultiPolygon의 exterior, hole, disconnected island를 모두 보존한다. Shapely의 ring 순서와 orientation은 신뢰하지 않고 정수 lattice에서 exact-collinear point만 제거한 뒤 exterior CCW, hole CW, 최소 시작점, component/hole 순서와 production ID를 자체 canonicalize한다.
- `artifact_outline_topology.py`는 ring simple/nonzero, hole의 단일 exterior 소유, boundary contact, hole 간 overlap/touch, exterior 간 overlap/nesting/touch, 최종 Polygon/MultiPolygon validity를 repair 없이 검증한다. 이 검사는 append와 load 모두에서 payload-derived QC로 다시 수행된다.
- production algorithm을 주장하는 record는 recipe 전체, six-view frame, grid 좌표, collinear 제거, component/hole ID까지 재검증한다. `vector.outline.v1` 자체는 열린 생태계를 위해 다른 algorithm recipe도 허용하지만 동일한 공통 topology 검증은 반드시 통과해야 한다.
- 권위 계산 backend는 현재 `shapely==2.1.2`와 GEOS `3.13.1` 조합으로 고정한다. 다른 조합은 새 algorithm version과 3-OS golden 없이 같은 recipe를 주장할 수 없다.

GUI는 한 번에 한 view의 `vector.outline.v1` record를 만든다. 여섯 view는 서로 다른 `PlanarFrame`을 가지므로 여섯 개의 독립 record다. 저장된 `READY + FRESH` payload를 다시 읽어 초록 exterior/hole/island overlay와 1:1 SVG를 만들며, 기존 ROI convex hull과 screenshot/OpenCV 단일/6-view export는 native 문서에서 방어적으로 차단한다. 여섯 record를 하나의 새 bundle로 묶는 multi-view package는 별도 포맷 과제다.

## 1:1 vector export package

`*.amr-vector`는 새로 만드는 non-overwriting directory package다.

| normative member | 역할 |
|---|---|
| `artifact.svg` | canonical-mm payload에서 다시 만든 1:1 presentation derivative |
| `artifact.amr-vector.json` | payload, recipe, QC, source/document/revision provenance, dependency closure, artifact hash |

Finder/Explorer가 추가하는 `.DS_Store`, `Thumbs.db`, `desktop.ini`는 1 MiB 이하의 일반 파일일 때만 무시한다. 그 외 추가 member, symlink, oversized member는 거부한다. writer는 같은 parent의 임시 directory에 두 파일을 쓰고 flush/fsync/자체 검증한 뒤 Linux `renameat2(RENAME_NOREPLACE)`, macOS `renamex_np(RENAME_EXCL)`, Windows non-replacing rename으로 publish한다. 목적지가 경합 중 생겨도 덮어쓰지 않는다. directory mode는 강제 0700이 아니라 사용자의 umask를 따른다.

1:1 규칙은 다음과 같다.

- payload와 SVG user coordinate는 모두 millimeter다.
- SVG `width="Wmm"`, `height="Hmm"`, `viewBox="0 0 W H"`는 같은 12-decimal canonical number token을 사용한다.
- scale transform, external image/use/script/style/event attribute, DTD/entity를 허용하지 않는다.
- margin은 stroke width의 절반 이상이어야 하므로 zero-extent open Cutline과 두꺼운 stroke도 artboard 밖으로 잘리지 않는다.
- plane frame을 보존한 2D points를 그리므로 Front/Right/oblique section을 world XY로 다시 투영하지 않는다.

sidecar는 source file SHA-256/size/scope, geometry hash, confirmed unit/axis matrix, Align matrix, export-time active IDs, record recipe/hash/QC와 transitive dependency receipts를 포함한다. extension, asset ref, topology-map ref, 내부 import path처럼 로컬 경로·site note가 들어갈 수 있는 필드는 public provenance allowlist에서 제외한다.

sidecar의 normative claim 전체(artifact descriptor 제외)는 RFC 8785 SHA-256으로 묶어 SVG metadata에 넣고, sidecar는 SVG exact-byte SHA-256을 가진다. 이 비순환 결합은 한 파일만 바뀐 손상을 검출한다. 원본 문서를 함께 줄 때 validator는 document manifest/record와도 대조한다. 문서 없이 relocation한 package도 독립 프로세스에서 payload/claim/SVG 구조와 hash를 offline 검증할 수 있다.

이 hash들은 무결성 값이지 디지털 서명이 아니다. 누군가 sidecar와 SVG를 모두 다시 만들고 모든 hash를 갱신하는 공격에 대한 제작자 진위 증명은 별도의 서명 규약이 필요하다. 기존 screenshot/flatten PNG·DPI export는 1:1 측정 증거로 승격하지 않으며 review image로만 취급한다. 아래 canonical Digital Rubbing PNG만 별도의 raster 계약을 가진다.

## M0-6 Digital Rubbing record와 1:1 PNG package

권위 Digital Rubbing은 OpenGL framebuffer, 카메라 screenshot, legacy `SurfaceVisualizer`, flattened review image를 입력으로 사용하지 않는다. `artifact_rubbing_extractor.py`가 검증된 source와 활성 Align을 canonical world millimeter로 materialize한 뒤 고정된 orthographic frame에서 CPU raster를 계산한다.

`ArtifactDocument 1.0` manifest에는 PNG bytes를 base64로 넣지 않고 다음 bounded receipt를 namespaced extension으로 저장한다. machine-readable 계약은 `schemas/rubbing_receipt-1.0.0.schema.json`이다.

```text
DerivedRecord(type=raster.digital_rubbing.v1)
├── geometry_ref: urn:archmeshrubbing:digital-rubbing-raster:sha256:<digest>
├── recipe: view + frame + physical pixel/depth/relief/resource policies
├── qc: input/raster/depth/ink/multi-layer counts and hashes
└── extensions[org.archmeshrubbing:digital-rubbing-v1]
    ├── media_type
    ├── schema_version
    ├── receipt_byte_length
    ├── receipt_sha256
    └── receipt
        ├── canonical PlanarFrame + six-view side
        ├── global pixel-lattice origin and dimensions
        ├── exact rational width/height in mm
        ├── pixels_per_meter + GA8 row order
        └── raw pixel SHA-256 + semantic raster SHA-256
```

프로젝트를 다시 열 때 known-record registry는 descriptor, receipt length/hash, frame/view, exact dimensions, pixel byte length, geometry ref, recipe와 record QC를 다시 검증한다. 실제 pixel blob은 `.amr`에 들어 있지 않으므로 preview/export에는 검증된 원본 source가 필요하며, record recipe로 raster를 다시 계산해 receipt와 비교한다. self-contained offline 배포물은 아래 `.amr-rubbing` package다.

### Raster 계산 정책

- view frame은 Outline과 같은 Top/Bottom/Front/Back/Right/Left right-handed frame이다.
- pixel density는 정수 `pixels_per_mm`이며 global half-integer pixel-centre lattice와 정수 micrometer 입력을 recipe에 고정한다. 요청한 margin과 reference radius는 pixel 수로 완전히 resolve해 함께 기록한다.
- 모든 valid face를 투영하고 각 pixel에서 frame normal 방향의 최대 depth를 front surface로 택한다. winding/backface로 face를 제거하지 않는다.
- 한 depth quantum 이상 떨어진 두 번째 layer가 있으면 `multi_layer_pixel_count`로 기록한다. v1은 한 view의 frontmost 판독 영상이며, 폐곡면 내부 전체나 재질 투과를 표현하는 X-ray/AO가 아니다.
- depth는 정수 µm tick으로 양자화하고 masked square local-mean integral image에서 raised/incised/bidirectional relief를 계산한다. tone mapping도 정수 규칙을 사용한다.
- coverage는 binary alpha인 GA8(`grayscale-alpha-8/v1`)이고 구멍/비투영 영역은 alpha 0이다. 화면 배경색은 권위 pixel에 합성하지 않는다.
- vertex/face/pixel/dimension/reference-radius/triangle-pixel-test 상한을 넘으면 해상도 축소, sampling 또는 다른 알고리즘으로 조용히 전환하지 않고 실패한다.
- face order·winding·duplicate, hole, large absolute survey offset과 늦은 Align 결과를 차단 테스트로 검증한다. barycentric edge rule은 recipe에 명시하지만, 실제 원격 3-OS golden이 통과하기 전에는 플랫폼 간 exact raster bytes를 완료로 주장하지 않는다.

GUI 계산은 worker에서 시작 당시 immutable `ArtifactSession`만 사용한다. 완료 callback은 projection snapshot뿐 아니라 시작 당시 session object identity도 검사한다. 같은 Align에서 다른 record가 중간에 추가된 경우에도 오래된 document를 publish해 그 기록을 잃을 수 있으므로 late result를 폐기한다. candidate publication은 `prepare_session_commit()`의 session/version/epoch compare-and-swap과 expected record ID 집합 검증을 다시 통과해야 한다. 미리보기는 receipt와 맞는 최초 계산 raster만 표시하며 export 권위로 재사용하지 않는다.

### Canonical PNG와 `.amr-rubbing`

`*.amr-rubbing`은 새로 만드는 non-overwriting directory package이며 정확히 두 regular file만 허용한다.

| normative member | 역할 |
|---|---|
| `artifact.png` | canonical GA8 1:1 planar raster derivative |
| `artifact.amr-rubbing.json` | recipe, raster receipt, QC, source/document/revision provenance와 PNG exact-byte hash |

PNG writer는 `IHDR → sRGB → pHYs → iTXt → IDAT → IEND` 순서, row filter 0, 직접 작성한 stored DEFLATE block을 사용한다. 따라서 Pillow encoder나 zlib compression heuristic에 exact bytes를 맡기지 않는다. `pHYs`의 X/Y pixels-per-meter는 receipt와 같고, width/height와 pixel pitch는 sidecar에서 exact rational millimeter로 반복 선언한다. primary PNG에는 scale bar, 글꼴, review label, 절대 source path나 외부 resource를 넣지 않는다.

sidecar의 artifact descriptor를 제외한 normative claim 전체를 RFC 8785 SHA-256으로 묶어 PNG iTXt metadata에 넣고, sidecar artifact descriptor는 PNG exact-byte SHA-256과 byte length를 가진다. validator는 PNG CRC/Adler/chunk/metadata/pixel hash/semantic raster hash, receipt, recipe, public provenance, scale와 privacy 선언을 모두 대조한다. 원본 document가 없어도 이동한 package를 별도 PID에서 offline 검증할 수 있고, document를 함께 주면 READY + FRESH record와 manifest까지 비교한다. machine-readable sidecar 계약은 `schemas/rubbing_export-1.0.0.schema.json`이다.

writer는 vector package와 같은 same-parent staging, file/directory `fsync`, self-validation, OS별 atomic no-replace publish를 사용한다. 기존 목적지, 추가 member와 symlink는 거부한다. 이 package의 hash도 무결성 값이며 제작자 전자서명은 아니다.

## 원자적 저장 절차

`save_project()`와 `save_artifact_project()`는 다음 순서로 동작한다. artifact API는 쓰기 전에 public representation을 `ArtifactDocument.from_dict()`로 다시 검증하고 canonical bytes가 원본 문서와 같은지도 확인한다.

1. strict JSON 직렬화: NaN/Infinity, non-string object key를 거부한다.
2. 목적 파일과 같은 디렉터리에 고유 임시 파일을 만든다.
3. ZIP을 닫고 file buffer를 flush한 뒤 `fsync`한다.
4. 선택된 payload 종류의 production loader로 임시 파일을 다시 열어 checksum·schema·payload를 검증한다. artifact payload는 artifact loader로 다시 연다.
5. 검증된 임시 파일을 `os.replace`로 한 번에 목적 경로에 교체한다.
6. 지원되는 파일시스템에서는 디렉터리도 `fsync`한다.

write, file `fsync`, 재검증, replace 중 어느 단계에서 실패해도 기존 목적 파일의 바이트는 유지되고 해당 임시 파일은 정리된다. 플랫폼·파일시스템이 directory `fsync`를 지원하지 않는 오류는 명시된 errno에 한해 무시한다. replace 후 실제 I/O 오류가 발생하면 파일은 이미 교체됐지만 crash durability가 불확실한 `committed=true` typed error로 보고한다. 디렉터리 생성 자체와 crash 후 남을 수 있는 고아 임시 파일의 recovery UI는 후속 recovery slice 범위다.

## 엄격한 로딩과 안전 한도

reader는 다음을 거부한다.

- invalid UTF-8, duplicate JSON key, NaN/Infinity
- duplicate·encrypted·위험한 경로의 ZIP member
- 지원하지 않는 압축 방식
- checksum 또는 CRC 불일치
- 64개 초과 member
- 64 MiB 초과 `project.json`
- 256 MiB 초과 단일 member
- 512 MiB 초과 총 비압축 크기
- 500:1 초과 압축 비율
- 약 520 MiB 초과 물리 컨테이너 또는 8 MiB 초과 ZIP central directory

reader는 `ZipFile`을 만들기 전에 EOCD/ZIP64 entry count와 central-directory 크기를 먼저 검사한다. 따라서 수백만 개의 빈 member를 사용해 64-member 한도 검사 전에 대량의 `ZipInfo`를 할당시키는 입력도 거부한다.

지원 버전보다 새로운 container major 또는 payload major는 실행 가능한 state로 반환하지 않는다. envelope의 제한된 scalar 정보만 가진 typed read-only inspection error를 제공한다.

## v1 migration

v1 import는 입력 파일을 수정하지 않는 순수·결정적·멱등 변환이다.

- 없는 source hash, 단위, 좌표계, Align matrix를 추정해 만들지 않는다.
- 외부 mesh에는 `identity=null`, `binding_status=legacy_unverified`를 추가한다.
- 각 object의 alignment는 `legacy_unverifiable`로 강등하고 기존 `fixed_state_valid`를 비활성화한다. 재정렬 전에는 신뢰 가능한 Align으로 표시하지 않는다.
- runtime `_migration.requires_save_as=true` marker를 반환한다.
- GUI의 첫 저장은 원본 v1과 다른 이름의 v2 파일만 허용한다.
- durable v2 파일에는 runtime `_migration` marker를 쓰지 않지만 `legacy_unverified` binding은 보존한다.

## 아직 보장하지 않는 것

- 두 개 이상의 artifact 또는 legacy object를 섞은 hybrid scene의 native document authority
- embedded mesh·sidecar를 포함한 portable project
- legacy `legacy_ui_state`에서 `artifact_document`로 단위·geometry·Align을 추정하지 않는 보존적 migration
- 여섯 개의 독립 Outline record를 원자적으로 계산·commit하고 한 bundle로 배포하는 multi-view package
- 실제 GPU driver frame의 scene-swap 원자성 및 시각적 동일성
- 큰 survey 좌표에서도 millimeter 이하 표시 정밀도를 보존하는 GPU render-origin
- autosave와 crash recovery discovery
- 전자서명 또는 provenance authority 인증

현재 native GUI는 source와 새 scene object를 기존 live scene과 분리해 load·검증하고, 준비된 projection을 검증한 다음 `ArtifactWorkbench` authority와 scene을 two-phase로 교체한다. missing, parse failure, hash mismatch, VBO 준비 또는 swap 실패 시 staging만 폐기하고 이전 scene·session·저장 경로를 복원한다. scene 교체는 projection generation을 증가시키고 이전 cut-section/ROI worker authority를 분리한다. 해당 callback은 current worker identity, generation과 selected object가 모두 일치할 때만 overlay를 갱신하며, 오래된 finished callback은 새 worker pointer를 지울 수 없다. rollback·scene 복원·finalize가 불확실하면 fatal state가 모든 저장·실측·내보내기를 차단한다. offscreen smoke는 이 transaction, worker fencing과 rollback 순서를 검증하지만 실제 GPU driver가 표시한 프레임 사이에 부분 장면이 전혀 노출되지 않는지까지 측정하지 않는다.

### 대좌표와 render-origin 후속 과제

문서, geometry identity, Align, 파생 기록, QC와 export 좌표는 origin을 보존한 **절대 float64 world millimeter**를 계속 사용한다. 현재 viewport처럼 이 값을 곧바로 float32 VBO로 내리면 UTM 계열 또는 `1e9 mm` 수준의 큰 offset에서 작은 유물의 millimeter/sub-millimeter 차이가 양자화될 수 있다. M0-3 headless 왕복 게이트는 CPU float64 좌표를 비교하므로 이 GPU 정밀도를 증명하지 않는다.

후속 renderer는 frame마다 transient `render_origin_mm`을 고르고 CPU float64에서 `p_relative = p_world_mm - render_origin_mm`을 계산한 뒤 relative 좌표만 float32 VBO에 올리는 방식을 검토한다. camera transform, picking과 측정 결과에서 origin을 정확히 복원해야 하며 `render_origin_mm`은 다음 권위 데이터에 들어가면 안 된다.

- `SourceMetadataRevision` 또는 `AlignRevision`
- source/geometry hash framing
- 저장된 record·QC·selection
- SVG/3D export의 world 좌표

향후 게이트는 `>= 1e9 mm` offset과 mm-scale feature를 함께 사용해 relative VBO upload와 pick reconstruction, 변하지 않은 document/geometry hash, 동일한 world export 좌표를 검증해야 한다.

### Legacy destructive bake 임시 안전 조건

현재 `bake_object_transform()`과 일부 floor/brush Align 경로는 표시 TRS를 `MeshData.vertices`에 직접 굽고 TRS를 identity로 되돌린다. `legacy_ui_state` writer는 이 변경된 vertex payload나 bake 전 행렬을 저장하지 않고 원본 source reference와 현재 identity TRS만 저장한다. 새로 발생한 destructive bake는 `_amr_has_unpersisted_bake`와 `legacy_baked_unverifiable`로 표시하며, `_collect_project_state()`는 조용한 데이터 유실 대신 저장을 거부한다. 이 marker가 없던 과거 파일이나 외부 변경에는 baked Align 복원 보장이 없다.

M0-3 native 한-artifact session은 immutable Align commit과 non-destructive scene materialization을 사용하지만 별도의 legacy payload/GUI 경로에는 기존 bake가 남아 있다. 다음을 임시 안전 조건으로 삼는다.

- destructive bake marker가 있는 scene은 현재 legacy AMR Save/Save As를 차단한다.
- 과거 `legacy_ui_state` 파일의 baked 상태를 재현 가능한 Align 기록으로 판정하지 않는다.
- 원본 source를 별도로 보존하고, bake 전 상태와 행렬을 잃을 수 있는 작업을 release-grade workflow로 선전하지 않는다.
- native artifact 왕복 게이트의 성공을 destructive legacy bake 경로의 저장 보장으로 확대 해석하지 않는다. 두 payload mode의 권위와 보장 범위는 분리한다.
