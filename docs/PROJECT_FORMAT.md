# AMR Project Format

이 문서는 ArchMeshRubbing 프로젝트 파일의 현재 저장 계약을 설명한다. AMR v2의 우선순위는 기능 수보다 **기존 연구 상태를 손상시키지 않는 저장**, **원본 바이트 식별**, **검증할 수 없는 상태의 명시적 표시**다.

이 포맷의 제품 구현과 차단 검증 대상은 Windows 10 version 1809 이상 x64와 Windows 11 x64다. 최소 버전은 고정한 Qt 6.11의 [공식 지원 플랫폼](https://doc.qt.io/qt-6.11/supported-platforms.html)을 따른다. 문서 안의 portable logical path와 코드의 내부 비 Windows backend는 파일 내용의 결정성과 회귀 시험을 위한 구현 세부사항이며, 비 Windows 애플리케이션 지원·배포·호환성을 약속하지 않는다.

## 컨테이너

`.amr`는 ZIP 컨테이너다. 모든 v2 문서는 앞의 두 파일을 포함하고, native portable artifact session은 뒤의 두 항목까지 포함한다.

| 멤버 | 역할 |
|---|---|
| `project.json` | 버전 envelope와 선택된 `legacy_ui_state` 또는 `artifact_document` payload |
| `checksums.json` | `project.json` 및 향후 멤버의 SHA-256 목록 |
| `sources/index.json` | ArtifactDocument와 정확히 결합된 canonical source inventory |
| `sources/blobs/sha256/<digest>` | 검증된 주 원본과 parser dependency bytes. SHA-256 이름을 사용하고 `ZIP_STORED`로 기록 |

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

### Portable artifact source bundle

`save_artifact_session_project()`는 정확히 한 `primary_mesh` SourceAsset과 그 geometry import가 실제로 소비한 parser dependency closure를 `sources/index.json`과 content-addressed blob으로 저장한다. index는 RFC 8785 canonical JSON이며 `document_id`, canonical document SHA-256, primary SourceAsset ID, logical path, role, media type, blob SHA-256·크기를 닫힌 스키마로 결합한다.

- `source_bundle 1.0`은 외부 resource를 읽지 않은 기존 self-contained 문서의 주 파일 하나를 운반한다.
- `source_bundle 2.0`은 `mesh_import_recipe 2.0` manifest의 주 파일과 MTL·texture·buffer 같은 `import_dependency`를 운반한다. 동일 bytes를 여러 logical path가 가리키면 index entry는 각각 남기되 content-addressed blob은 하나만 저장한다.
- `ArtifactDocument.SourceAsset`은 계속 고고학적 권위 원본인 주 메쉬 하나를 식별한다. Dependency는 별도 SourceAsset으로 가장하지 않고 `GeometryRevision.import_recipe.source_manifest`의 parser input closure로 기록한다. v2 index의 dependency `source_asset_id` 필드는 하위 호환 필드명이며 값은 `sha256:<digest>` content ID다.
- 여러 GeometryRevision이 있으면 저장 가능한 역사적 revision을 재현할 수 있도록 dependency manifest의 합집합을 운반한다. active geometry가 v2이면 그 주 logical path를, active가 v1이면 보존된 v2 manifest 중 결정적으로 선택한 주 logical path를 사용한다.

저장 과정은 다음 순서로 fail-closed 동작한다.

1. 외부 원본·dependency 또는 이미 열린 `.amr`의 source member를 검증된 descriptor stream으로 연다.
2. 기대 SHA-256·크기를 확인하면서 같은 parent의 임시 ZIP으로 복사한다.
3. 임시 package의 central directory, member 규칙, 전체 checksum과 source index를 production reader로 다시 검증한다.
4. embedded source closure를 저장된 전체 closed import recipe/unit과 manifest-only resolver로 실제 decode하고 document에 bind·materialize하여 source/dependency/geometry/Align projection과 parser receipt를 확인한다.
5. source archive descriptor를 닫은 뒤에만 목적지를 commit한다. Windows 제품 경로는 같은 폴더 staging을 긴 Unicode/UNC path로 변환해 [`MoveFileExW`](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw)의 `MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH`로 교체하며, 이 Win32 호출이 실패하면 기존 목적지를 보존하고 더 약한 rename으로 fallback하지 않는다. 코드에 남은 비 Windows `os.replace`와 parent directory `fsync` 경로는 내부 회귀 시험용이다.

native session 저장 대상은 `.amr` 확장자만 허용한다. 대상 경로가 외부 source resource와 같은 path, symlink 또는 hardlink inode라면 임시 파일 생성 전과 교체 직전에 거부한다. 이미 embedded source에서 열린 session은 같은 `.amr` 위 저장과 Save As를 모두 지원한다. 기존 source bundle이 없는 manifest-only artifact 문서는 계속 읽지만, session materialization에는 외부 주 원본을 다시 선택해야 한다.

한 source manifest와 v2 bundle index는 현재 최대 61개 entry이며, embedded source 전체 합계는 최대 16 GiB다. 일반 member의 기존 256 MiB/총 512 MiB 제한과 분리한다. source blob은 압축 폭탄을 피하고 streaming hash를 가능하게 하기 위해 `ZIP_STORED`만 허용한다. 이 한계보다 큰 multi-file scan은 아직 authoritative import/save 대상이 아니다.

### Runtime saved-snapshot checkpoint

저장 checkpoint는 `.amr` container member가 아니라 Windows native application state다. Workbench는 다음 세 값을 한 묶음으로 유지한다.

- 저장 당시 immutable `ArtifactDocument` canonical SHA-256
- 정규화한 project path
- `confirmed` 또는 `uncertain` 내구성 상태

현재 document SHA-256와 project path가 checkpoint와 exact match하고 내구성이 `confirmed`일 때만 clean이다. checkpoint가 없거나 hash/path가 다르거나 내구성이 `uncertain`이면 unsaved changes로 취급하고 `durability_uncertain` save status를 표시한다. 이 상태는 별도 serialized `dirty` flag로 저장하지 않으며, 창 제목의 `*`와 상태 표시줄도 같은 Workbench snapshot에서 파생한다.

- 새 source import는 저장 checkpoint가 없는 dirty 문서로 게시한다.
- embedded source closure, parser receipt, document/geometry/Align을 production reopen 경계에서 모두 검증한 project는 재열린 exact path/document의 confirmed checkpoint로 시작한다.
- immutable Align revision이나 DerivedRecord를 append/activate해 canonical document SHA-256가 바뀌면 기존 checkpoint는 이력으로 남지만 현재 문서는 dirty이다.
- 같은 경로 Save와 Save As는 모두 캡처한 session/state version/authority epoch/기존 project path의 exact compare-and-swap이 성공한 뒤에만 checkpoint를 갱신한다. 과거 snapshot 파일이 성공적으로 쓰였더라도 stale CAS이면 현재 문서는 clean이 아니다.
- Windows의 `MoveFileExW` write-through 호출이 실패하면 pre-commit 실패이므로 checkpoint를 갱신하지 않는다. 성공하면 Microsoft가 문서화한 “move가 disk에서 완료될 때까지 반환하지 않음” 경계로 `confirmed`를 만들 수 있다. 내부 비 Windows backend에서 `os.replace` 뒤 directory `fsync`가 실제 실패한 committed 결과는 회귀 시험상 파일 게시 성공과 crash durability 미확정을 구분하되 clean으로 승격하지 않는다. 이 backend는 제품 지원 계약이 아니다.

Windows native Close, 새 source Open, Project Open, drag-and-drop은 dirty 문서에서 공통 `Save / Discard / Cancel` gate를 통과한다. `Save`를 고르면 원래의 닫기/열기 명령은 비동기 저장이 exact current snapshot의 confirmed checkpoint를 만든 뒤에만 재개된다. Save As 대화상자 취소, writer/CAS 실패, stale 완료, durability-uncertain 결과는 후속 명령을 허가하지 않고 현재 문서와 창을 보존한다. `Discard`만 명시적으로 checkpoint 불일치를 무시하고 교체를 진행하며 `Cancel`은 아무 것도 바꾸지 않는다.

Packaged complete-workflow self-test는 새 import `dirty` → 첫 exact Save `saved` → record append `dirty` → 동일 경로 exact Save `saved`를 실제 Workbench에서 통과하고 report detail에 `checkpoint=dirty>saved>dirty>saved`와 `project_commit=windows-movefileex-write-through`를 남겨야 한다. frozen과 한글 경로 portable Windows gate는 쉼표로 구분한 두 marker token이 정확히 없으면 실패한다. 이 marker는 runtime checkpoint 전이와 실제 Windows commit backend의 패키지 회귀 증거이지 `.amr` member가 아니다. 현재 완료 판정과 frozen/portable 증거는 Windows에만 적용한다. 이 저장 계약은 코드 권리·공개 배포 적합성, 대표 하드웨어, 실제 유물/고고학자 파일럿 완료를 증명하지 않는다.

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
- 새 ArtifactDocument의 `SourceAsset.asset_ref`는 `external:<original_name>`인 상대 locator다. 현재 세션의 native 절대경로는 `ArtifactSession.resolved_source_path`에만 유지하므로 drive/root가 달라도 같은 source·recipe·revision은 같은 canonical document와 export hash를 만든다. 기존 `external:<absolute-path>` 문서도 계속 읽으며, 상대 locator가 프로젝트 옆에서 해결되지 않으면 source picker로 검증 파일을 다시 지정한다.
- 경로와 이름이 달라도 SHA-256과 크기가 같으면 `relocated=true`인 verified 결과로 열 수 있다.
- `parse_format`은 legacy UI state의 parser hint다. Native ArtifactDocument에서는 아래의 닫힌 `GeometryRevision.import_recipe` 전체가 권위 실행 계약이며 후보 파일의 suffix보다 우선한다.
- 같은 크기라도 SHA-256이 다르면 mismatch이며 face ID, cutline, 기록면 데이터 등을 replacement mesh에 적용하지 않는다.
- hash와 mesh parser는 동일한 열린 file descriptor를 사용한다. 경로를 다시 열어 다른 파일의 geometry에 이전 hash를 붙이지 않는다.
- Windows의 path `stat`과 descriptor `fstat`은 `ctime` 의미가 다르므로 혼합 비교에서는 device/inode/size/mtime만 비교한다. 열린 descriptor의 전후 비교에서는 change time까지 유지해 같은 크기·mtime의 소비 중 변경을 계속 거부한다.
- 저장 시 디스크 파일을 다시 hash하지 않는다. import 당시 geometry와 함께 보관한 immutable identity만 직렬화한다.

Artifact payload를 다시 열 때 parser 선택과 실행 계약도 검증 대상이다. 신규 문서는 외부 resource 소비 여부에 따라 `schemas/mesh_import_recipe-1.0.0.schema.json` 또는 `schemas/mesh_import_recipe-2.0.0.schema.json`의 exact-key 계약을 `GeometryRevision.import_recipe`에 저장한다.

```text
recipe_id/version
format, loader, loader_version
parser_runtime_sha256, runtime_lock_sha256
force=mesh, process=false, maintain_order=true
scene_merge=trimesh.util.concatenate/v1
sanitizer=meshdata-v1
dependency_policy=deny_external
```

이 self-contained v1 recipe로 parse를 시작한 뒤 parser가 상대 resource를 실제로 읽으면 최종 receipt는 아래 필드를 추가한 v2로 확정된다.

```text
recipe_version=2.0.0
dependency_policy=closed_manifest
resolver_profile=relative-contained-v1
source_manifest=(primary_mesh + import_dependency logical path/media type/SHA-256/size)
```

`source_manifest 1.0`은 primary 하나와 parser 또는 parser 전 admission이 성공한 import 동안 실제로 검증한 dependency만 정렬된 상대 POSIX logical path, 고정 media type, SHA-256, 크기로 기록한다. glTF/GLB가 선언한 외부 buffer는 parser 진입 전에 전부 exact-length/hash 검증하므로 이 closure에 포함된다. host 절대 경로는 durable recipe에 들어가지 않는다. replay resolver는 선언된 stream만 정확히 한 번 이상 제공하고, HTTP/file URI, 절대·drive·UNC 경로, source root를 벗어나는 `..`와 symlink, 누락·미선언·변조·미사용 entry를 거부한다. Windows식 `\` reference는 root 안에서만 POSIX path로 정규화한다.

새 GeometryRevision의 `qc.import_admission`은 `schemas/mesh_admission_receipt-1.0.0.schema.json` 계약을 따른다. 열린 regular-file descriptor의 source size를 hash 전에 최대 4 GiB로 제한하고, hash 뒤에는 같은 identity를 다시 만족한 bounded spool snapshot만 preflight/parser에 전달한다. source 전체를 materialize하는 OBJ·OFF·ASCII PLY/STL parser 입력은 최대 256 MiB이며 binary PLY/STL·GLB는 4 GiB primary cap을 유지한다. OBJ·PLY·STL·OFF는 parser 전에 vertex·face element·triangulated face 수를 bounded scan한다. PLY는 정확히 vertex→face element만 허용하고 vertex/face property를 16/8개, auxiliary typed data를 128 MiB로 제한하며 list type·per-face texcoord count·payload EOF와 보수적인 source+typed arrays+canonical arrays+list-row parser footprint를 확인한다. 잠긴 binary PLY parser가 첫 list row로 element dtype을 고정하므로 모든 binary list property는 element 전체에서 같은 길이만 허용한다. glTF/GLB는 최대 16 MiB JSON, GLB JSON `buffer[0].byteLength`와 실제 BIN chunk 길이, 모든 declared buffer·중복/겹침을 포함한 각 bufferView slice·미사용 accessor array·sparse·primitive·node instance 증폭을 검증한다. `data:` buffer는 parser와 같은 lower-case `data:...;base64,` 형식만 허용해 bounded decode 후 declared length와 맞추며, `base64,`를 포함한 모호한 외부 URI는 parser 전에 거부한다. 상대 외부 buffer는 파일별 512 MiB·합계 1 GiB 안에서 parser 전에 actual bytes의 exact length/hash를 검증·cache하고, fingerprint 뒤 재읽은 payload도 같은 SHA-256이어야 manifest와 parser cache에 들어간다. GLB source, 모든 declared buffer·bufferView slice·accessor와 scene-instance canonical arrays를 합친 parser footprint도 같은 2 GiB/3 GiB envelope를 통과해야 한다. decoded profile은 최대 5,000,000 vertices, 2,000,000 triangles, 2 GiB arrays, 3 GiB 추정 native Open peak이며 texture는 512 MiB다. receipt는 선언 수, PLY/glTF parser bytes, decoded/accepted 수, sanitizer가 제거한 degenerate triangle 수, accepted geometry SHA-256과 exact limit set을 보존하고 reopen 시 source format·byte length와 UV/texture를 포함한 현재 accepted array byte 하한까지 대조한다. 현재 Windows physical/commit 여유 검사는 실행 PC에만 적용하는 비내구성 gate라 문서 값에 넣지 않는다. parser subprocess/Job Object crash·memory 격리는 아직 미완료이므로 이 admission을 임의 비신뢰 파일에 대한 보안 sandbox로 해석하지 않는다.

`parser_runtime_sha256`은 정렬된 exact `numpy`, `pillow`, `trimesh` pin 문자열의 SHA-256이며 실제 geometry decode의 실행 gate다. `runtime_lock_sha256`은 전체 frozen 환경 provenance로 보존하지만 Qt 같은 무관한 pin 변경만으로 과거 geometry를 읽지 못하게 하지 않도록 parser 실행 gate로 사용하지 않는다. 알 수 없는 key, flag drift, loader/parser-subset version 불일치, 비정규화 format은 parse 전에 거부한다. 기존 배포가 만든 정확한 5-field recipe와 공식 2-field fixture만 명시적 legacy profile로 실행하며 임의 JSON을 parser option으로 해석하거나 immutable GeometryRevision을 자동 migration하지 않는다.

Resolved source의 suffix 대신 검증된 recipe의 `format`을 사용하고, recipe 전체를 `ArtifactLoadTicket → MeshLoadThread → MeshLoader`로 그대로 전달한다. loader는 같은 열린 descriptor에서 raw byte fingerprint와 geometry를 얻는다. 이후 `ArtifactSession.bind_loaded_document()`는 다음을 모두 만족할 때만 문서와 mesh를 결합한다.

1. 새로 계산한 `identity_scope`, `sha256`, `size_bytes`가 `SourceAsset`과 일치한다.
2. 실제 사용한 `MeshData.source_format`이 저장된 `import_recipe.format`과 일치한다.
3. 실제 parser가 남긴 `MeshData.source_import_recipe`가 저장된 mapping과 key/value까지 정확히 일치한다.
4. v2이면 runtime source resource가 manifest의 모든 logical path·SHA-256·크기를 정확히 제공하고, replay resolver가 같은 dependency closure를 소비한다.
5. durable import admission이 있으면 runtime receipt의 exact limit·declaration·decoded/accepted accounting, source format·byte length·accepted array byte 하한과 canonical geometry SHA-256이 GeometryRevision 및 현재 source snapshot과 일치한다.
6. 저장된 `geometry_hash_scope`로 decode 결과를 다시 hash한 값이 `GeometryRevision.geometry_sha256`와 일치한다.

저장된 recipe가 없거나 지원 profile이 아니거나 다른 parser/receipt로 열렸거나 source/geometry digest가 다르면 materialization 전에 실패한다. 따라서 이름과 확장자가 바뀐 동일 파일은 복원할 수 있지만, 단지 suffix가 같다는 이유로 다른 바이트나 다른 decode 결과를 받아들이지 않는다.

`binding_status`의 현재 값:

| 값 | 의미 |
|---|---|
| `captured_at_import` | raw primary file을 hash한 뒤 동일 descriptor에서 geometry를 import했다. |
| `legacy_unverified` | v1 기록과 현재 source의 관계를 사후에 증명할 수 없다. 현재 hash가 생겨도 기존 기록을 소급해 verified로 바꾸지 않는다. |
| `generated_ephemeral` | 외부 원본 경로가 없는 런타임 생성 mesh다. 현재 v2만으로 새 프로세스에서 복원할 수 있다는 뜻이 아니다. |

### M0-1 primary identity와 parser source closure

`identity_scope=primary_file_bytes`와 `SourceAsset`은 의도적으로 **주 파일 하나의 바이트만** 식별한다. 다음 linked resource는 주 유물 SourceAsset이 아니라 geometry를 재현하는 parser input이므로 v2 `source_manifest`의 `import_dependency`로 별도 식별한다.

- OBJ의 MTL·텍스처
- glTF의 외부 `.bin`·이미지
- 기타 sidecar 또는 linked asset

새 path import는 filesystem 전체를 parser에 노출하지 않는 recording resolver를 사용한다. source root 안의 상대 resource를 읽을 때 같은 stream을 hash·parse하고 최종 v2 manifest로 확정한다. 저장·재열기에는 filesystem 탐색 없이 manifest-only resolver와 content-addressed bundle만 사용한다. self-contained 입력은 기존 v1 `deny_external`로 남는다. 두 profile 모두 parser가 오류를 삼켜도 resolver가 관찰한 금지·누락·미사용 요청을 Open 단계에서 실패시킨다.

따라서 `verified`는 “주 파일 hash 하나가 모든 파일을 대표한다”는 뜻이 아니다. v1은 self-contained 주 파일과 closed recipe, v2는 주 SourceAsset identity + manifest의 전체 resource identity + 같은 recipe decode 결과를 함께 검증했다는 뜻이다.

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
| `SourceAsset` | raw asset의 SHA-256, 바이트 크기, media type, 원래 이름, non-authoritative `asset_ref`, `role=primary_mesh`, identity 범위 |
| `GeometryRevision` | 결정적 import 결과의 별도 geometry hash·hash scope, source asset ID, import recipe, topology-map reference, QC |
| `SourceMetadataRevision` | geometry의 단위·축 매핑·handedness와 `source_to_canonical_mm`; parent revision |
| `AlignRevision` | confirmed metadata 위에 적용할 proper rigid 4×4 행렬, parent revision, recipe, QC |
| `DerivedRecord` | geometry/Align revision에 묶인 cutline·outline·rubbing·tile unwrap·geometry metrics 등의 payload reference, recipe/hash, selection hash, dependency, QC, lifecycle |

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

Native workflow의 단계 완료는 freshness만으로 충분하지 않다. Cutline은 Top/Front/Right canonical frame을 각각 하나 이상 가져야 하고, 완료 증거로 인정되는 각 Outline record는 그 세 Cutline view의 `READY + FRESH` record ID를 직접 dependency로 가져야 한다. 각 Digital Rubbing record도 dependency coverage가 유효한 6면 Outline record ID를 직접 참조해야 한다. application command가 이 dependency를 자동 캡처하며, 외부 도구가 순서를 건너뛰어 만든 READY record는 저장 가능하더라도 native 진행도나 다음 단계 gate의 증거로 사용하지 않는다.

### M0-3 이후 한 artifact native workflow

M0-3에서 시작한 durable core와 현재 native GUI/application 경계는 다음과 같다.

- `ArtifactDocument 1.0`을 AMR v2의 `artifact_document` payload로 strict serialize/load하고 `project.json` checksum과 원자적 staged validation에 포함한다.
- `ArtifactSession`이 원본 source-space mesh의 immutable snapshot, source byte identity, geometry identity와 document를 하나의 검증된 context로 묶는다.
- headless `ArtifactSceneAdapter`가 항상 immutable source에서 시작해 active `Align @ SourceMetadata`를 적용한 새 float64 world-mm `MeshData`를 만든다. source vertex를 mutate하거나 centroid로 recenter하지 않고, document/revision/hash/matrix snapshot이 바뀐 늦은 결과를 거부한다.
- native application은 정확히 한 artifact를 다루며 `ArtifactWorkbench.snapshot.session.document`를 source of truth로 둔다. MainWindow의 session field는 이행 중 compatibility mirror다. 사용자 Open은 단위·축·handedness 확인 후 ticketed load로 들어가고, `initial_identity` baseline에서는 `ALIGN_REQUIRED`에 머문다. 이동·회전은 preview일 뿐이며 첫 정위치 확정은 변화량이 0이어도 proper-rigid child Align revision을 append한 뒤 immutable source에서 장면을 다시 materialize한다. parent activation으로 baseline에 돌아가면 측정·내보내기가 다시 잠긴다. scale은 metadata 영역이므로 native Align preview에서 차단한다.
- Align commit과 parent activation은 GUI에서 immutable session, exact scene binding, preview TRS/pivot와 Workbench version만 캡처한다. source geometry hash 검증, candidate session 생성과 canonical materialization은 application-modal locked worker에서 수행한다. 완료 시 object/mesh/binding/preview, session/state version/authority epoch와 project path가 capture와 모두 같아야만 GUI thread의 VBO 준비·two-phase scene publication으로 넘어간다. 변경된 late result는 document나 scene을 수정하지 않고 폐기한다.
- artifact project reopen은 embedded source가 있으면 별도 picker 없이 background worker에서 package와 source bytes를 검증하고 saved parser/unit으로 CPU staging한다. manifest-only 문서만 외부 source resolver를 사용한다. corrupt embedded package를 external/legacy 경로로 fallback하지 않는다. `ArtifactWorkbench`는 한 pending Open ticket과 `state_version`/`authority_epoch`를 검증하고, candidate projection을 준비한 뒤 scene notification 동안에만 tentative authority로 활성화한다. scene swap 성공 후 finalize하고, 실패하면 이전 session·scene·project path/checkpoint로 rollback한다. observer는 finalize 전 candidate를 보지 않으며, fully verified reopen finalize만 현재 document/path의 confirmed checkpoint를 만든다.
- rollback·scene 복원·finalize 자체가 실패해 application authority와 live scene의 일치를 증명할 수 없으면 fatal authority 상태로 전환한다. 이 상태에서는 ordinary Save target을 해제하고 저장·실측·내보내기를 모두 거부하며, 검증된 새 Open만 정상 authority를 회복한다.
- artifact save는 active document만 쓰기 전에 정확히 한 projection, current snapshot, identity preview, source에서 재현한 vertices/faces 일치, destructive bake 부재를 확인한다. desktop은 immutable session과 Workbench state/authority version, 기존 project path를 캡처하고 이 geometry 비교부터 source closure 재검증·ZIP64/fsync·staged package 재개방/materialization까지 worker에서 실행한다. atomic writer 완료 뒤 캡처 권위가 달라졌으면 만들어진 파일은 과거 snapshot으로 보고하며 현재 project path/checkpoint와 Save As/migration 상태를 갱신하지 않는다. 경로/checkpoint 채택은 Workbench의 exact session/state/epoch CAS이며 성공 시 state version만 전진한다. `ALIGN_REQUIRED` document 자체는 보존할 수 있지만 Cutline/Outline/Digital Rubbing/기와 전개/검증 제원 계산과 vector/rubbing/survey/tile-unwrap export는 명시적 Align 전까지 차단한다. 아직 `DerivedRecord`로 승격되지 않은 cutline·선택·기록면·평가 등의 결과가 하나라도 있으면 누락한 채 저장하지 않고 fail closed한다.
- Cutline/Outline/Digital Rubbing/기와 전개/검증 제원은 application layer가 canonical recipe, projection context, exact record ID와 result capability를 소유한다. GUI handler는 projection binding·TRS·transient mutation만 확인하고, canonical source materialization과 live vertex/face exact comparison은 controller 실행 경계의 worker preflight에서 수행한다. worker는 session을 commit하지 않고 computation만 반환하며, 완료 시 captured document가 current document의 immutable ancestor이고 active source/metadata/Align/matrix가 같을 때만 current session에 rebase하여 expected record ID 하나를 publish한다. DerivedRecord append는 `RecordBindingTransition`으로 live object의 immutable document snapshot만 CAS하고 기존 mesh/VBO를 보존한다. 일반 scene selection은 유지하되, 기와 ‘현재 선택 면’ recipe와 live selection이 게시 시점에도 정확히 같으면 record로 소비된 선택만 비운다. Align/Open finalize 뒤 늦은 결과는 되살아나지 않는다. pending Open이나 rollback 가능한 binding 준비 실패는 계산 결과와 예약 ID를 보존해 명시적으로 재시도하며, 그동안 저장과 새 실측을 차단한다. Rubbing begin은 geometry·UV·texture 복사를 포함한 최소 admission만 예약하고, worker가 해상도별 전체 peak-memory estimate를 계산해 공유 budget 안에서 원자 확장한다. preflight 실패·취소는 같은 terminal 상태 머신에서 예약을 해제하며 실행은 exactly-once다.
- vector/rubbing/survey/tile-unwrap export는 exact work item/result capability를 별도로 예약한다. worker는 비싼 SVG 생성, Rubbing 또는 tile-unwrap recipe 재계산·receipt 비교, package 전체 검증을 수행하고 destination·parent·staging inode·member fingerprint에 묶인 prepared capability까지 만든다. survey export는 dependency-valid 3/6/6의 exact record 15개를 canonical 순서로 캡처하고 6개 raster를 다시 계산한 뒤 부모 tree 전체를 fingerprint한다. final dispatcher는 current source session, render projection과 캡처한 모든 `READY + FRESH` record를 Workbench lock에서 다시 확인한 뒤 빠른 fingerprint 재확인과 atomic no-replace rename만 실행한다. 같은 Align의 append-only record 추가는 허용하고 Align/Open 완료는 destination을 만들지 않은 채 stale 처리한다. pending Open은 core에서 재시도 가능한 stage로 남지만 현재 GUI는 안전하게 정리하고 Open 완료 후 재실행을 안내한다.
- `Open → Align commit → save → independent-process load → source rebind → materialize` 왕복을 별도 프로세스에서 검증한다.

`tests/test_artifact_new_process_roundtrip.py`의 차단 게이트는 self-contained PLY와 MTL·PNG를 사용하는 textured OBJ에 대해 다음 순서를 실제로 수행한다.

1. 프로세스 A가 source closure를 같은 descriptor/resolver stream에서 hash·parse하고 cm metadata와 비자명한 pivot Align revision을 만든 뒤 embedded artifact package로 저장한다.
2. 외부 PLY 또는 OBJ·MTL·PNG가 있던 capture directory를 삭제하고 `.amr`를 다른 directory로 옮긴다.
3. 다른 PID의 프로세스 B가 relocated `.amr`만 strict load하고 content-addressed source blob을 저장된 전체 parser/runtime receipt, source manifest와 metadata unit으로 다시 연다.
4. 프로세스 B가 embedded raw primary/dependency SHA-256·크기, decode geometry SHA-256을 새로 계산하고 검증된 session에 bind한 뒤 world-mm geometry를 materialize한다.
5. 두 프로세스의 source closure, geometry SHA-256, active Align ID·float64 matrix, 전체 import recipe·unit, world vertex와 texture array hash를 비교한다.

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
- Shapely `set_precision(..., mode="valid_output")`으로 각 triangle을 격자에 맞춘 뒤 `union_all`한다. 한 격자보다 좁은 triangle, gap, island, hole은 사라지거나 합쳐질 수 있으며 `grid_collapsed_triangle_count`, component merge/split, unsnapped area comparison을 QC에 남긴다.
- **outline algorithm 1.1.0**은 이 lattice union을 한 격자 반경의 morphological closing(`buffer(+1, mitre)` → `buffer(-1, mitre)` → `set_precision(1)`)으로 닫는다. 실루엣 근처의 투영 triangle은 격자보다 얇아서 개별 snapping이 그 사이에 한 격자 폭의 sliver hole과 외곽선이 한 점에서 맞닿는 pinch를 남기는데, 이것은 닫힌 표면에 없는, 격자의 인공물이다. closing은 두 격자보다 좁은 feature만 제거하며 `grid_closing_radius_cells`·`grid_closing_hole_fill_count`·`grid_closing_component_merge_count`·`grid_closing_area_delta_mm2`를 QC에, `grid_closing`을 recipe에 남긴다. snap 계약은 `axis<=1.5*grid; radial<=1.5*grid*sqrt(2)`(union 반 격자 + closing 반경 + 재snap 반 격자)다. **1.0.0 record**(closing 없음, `axis<=grid/2; radial<=grid/sqrt(2)`)는 recipe의 `algorithm_version`대로 예전 그대로 재계산되며, 상태 표기(`annotation.condition.v1`)의 뷰별 경계는 1.0.0 union을 그대로 쓴다.
- 25,000 face씩 먼저 union하여 입력 Shapely 객체를 한꺼번에 보유하지 않고, chunk result를 고정 balanced-pairwise 순서로 합친다. 중간 polygon·coordinate 수, 입력 vertex·face 수, lattice index, 최종 path·point·payload byte 수를 제한하며 초과 시 sampling이나 decimation으로 성공시키지 않는다.
- 최종 Polygon/MultiPolygon의 exterior, hole, disconnected island를 모두 보존한다. Shapely의 ring 순서와 orientation은 신뢰하지 않고 정수 lattice에서 exact-collinear point만 제거한 뒤 exterior CCW, hole CW, 최소 시작점, component/hole 순서와 production ID를 자체 canonicalize한다.
- `artifact_outline_topology.py`는 ring simple/nonzero, hole의 단일 exterior 소유, boundary contact, hole 간 overlap/touch, exterior 간 overlap/nesting/touch, 최종 Polygon/MultiPolygon validity를 repair 없이 검증한다. 이 검사는 append와 load 모두에서 payload-derived QC로 다시 수행된다.
- production algorithm을 주장하는 record는 recipe 전체, six-view frame, grid 좌표, collinear 제거, component/hole ID까지 재검증한다. `vector.outline.v1` 자체는 열린 생태계를 위해 다른 algorithm recipe도 허용하지만 동일한 공통 topology 검증은 반드시 통과해야 한다.
- 권위 계산 backend는 현재 `shapely==2.1.2`와 GEOS `3.13.1` 조합으로 고정한다. 다른 조합은 새 algorithm version과 Windows golden 없이 같은 recipe를 주장할 수 없다.

GUI는 한 번에 한 view의 `vector.outline.v1` record를 만든다. 여섯 view는 서로 다른 `PlanarFrame`을 가지므로 여섯 개의 독립 record다. 저장된 `READY + FRESH` payload를 다시 읽어 초록 exterior/hole/island overlay와 1:1 SVG를 만들며, 기존 ROI convex hull과 screenshot/OpenCV 단일/6-view export는 native 문서에서 방어적으로 차단한다. 계산·record commit은 view별로 유지하지만 완료된 3/6/6 배포는 아래 `.amr-survey`가 한 번의 부모-directory 원자 게시로 묶는다.

## 1:1 vector export package

`*.amr-vector`는 새로 만드는 non-overwriting directory package다.

| normative member | 역할 |
|---|---|
| `artifact.svg` | canonical-mm payload에서 다시 만든 1:1 presentation derivative |
| `artifact.amr-vector.json` | payload, recipe, QC, source/document/revision provenance, dependency closure, artifact hash |

Explorer가 추가하는 `Thumbs.db`, `desktop.ini`는 1 MiB 이하의 일반 파일일 때만 무시한다. 그 외 추가 member, symlink, oversized member는 거부한다. Windows writer는 같은 parent의 임시 directory에 두 파일을 쓰고 flush/fsync/자체 검증한 뒤 non-replacing rename으로 publish한다. 목적지가 경합 중 생겨도 덮어쓰지 않는다. staging 이름은 목적지 이름 길이와 무관한 고정 길이 UUID component로 배타 생성하며 충돌한 foreign directory를 검사·재사용·삭제하지 않는다. 안전한 discard는 staging을 먼저 고유 quarantine 이름으로 원자 이동하고 inode를 재확인한 뒤 표준 라이브러리의 best-available cleanup을 사용한다. 소유권을 증명할 수 없으면 foreign path를 보존하고 실패한다. 내부 비 Windows backend의 native no-replace·descriptor-relative cleanup·directory `fsync`는 회귀 시험용이며 제품 지원 계약이 아니다. final rename 뒤 실제 오류뿐 아니라 미지원 directory `fsync`도 package가 이미 공개된 `committed=true` durability-uncertain 상태로 전달한다.

1:1 규칙은 다음과 같다.

- payload와 SVG user coordinate는 모두 millimeter다.
- SVG `width="Wmm"`, `height="Hmm"`, `viewBox="0 0 W H"`는 같은 12-decimal canonical number token을 사용한다.
- scale transform, external image/use/script/style/event attribute, DTD/entity를 허용하지 않는다.
- margin은 stroke width의 절반 이상이어야 하므로 zero-extent open Cutline과 두꺼운 stroke도 artboard 밖으로 잘리지 않는다.
- plane frame을 보존한 2D points를 그리므로 Front/Right/oblique section을 world XY로 다시 투영하지 않는다.

sidecar는 primary source file SHA-256/size/scope, v2 source manifest dependency의 logical path/SHA-256/size, geometry hash, confirmed unit/axis matrix, Align matrix, export-time active IDs, record recipe/hash/QC와 transitive record dependency receipts를 포함한다. extension, asset ref, topology-map ref, host locator처럼 로컬 경로·site note가 들어갈 수 있는 필드는 public provenance allowlist에서 제외한다.

현재 writer는 `schemas/vector_export-1.3.0.schema.json`을 사용한다. 1.3은 1.2에 사용자 선 굵기 preset의 정의(`presentation.style_preset.definition`)를 더한 것이고, 그 이전 sidecar는 사용자 preset으로 그린 도면을 담을 수 없다. 1.2는 1.1에 outline algorithm 1.1.0(격자 closing)의 recipe 키 `grid_closing`과 QC 키 넷을 더한 것이고, 1.1.0 이하 sidecar는 closing으로 계산한 outline을 담을 수 없다. 1.0.0·1.1.0·1.2.0 schema bytes는 그대로다. 1.1은 Align recipe/QC, geometry QC와 mesh-admission receipt, root payload QC를 exact-key로 닫고, record QC도 payload kind와 algorithm에 맞춘다. production Cutline/Outline은 해당 extractor QC 전부를 요구하며 custom algorithm은 공통 payload QC만 허용한다(Outline은 payload에서 재계산한 `outline_topology`도 필수). `vector_export-1.0.0.schema.json`은 바이트 불변 legacy 검증용으로 보존하며 reader는 admission receipt가 없던 유효 1.0 package를 계속 offline 검증한다.

sidecar의 normative claim 전체(artifact descriptor 제외)는 RFC 8785 SHA-256으로 묶어 SVG metadata에 넣고, sidecar는 SVG exact-byte SHA-256을 가진다. 이 비순환 결합은 한 파일만 바뀐 손상을 검출한다. 원본 문서를 함께 줄 때 validator는 document manifest/record와도 대조한다. 문서 없이 relocation한 package도 독립 프로세스에서 payload/claim/SVG 구조와 hash를 offline 검증할 수 있다.

이 hash들은 무결성 값이지 디지털 서명이 아니다. 누군가 sidecar와 SVG를 모두 다시 만들고 모든 hash를 갱신하는 공격에 대한 제작자 진위 증명은 별도의 서명 규약이 필요하다. 기존 screenshot/flatten PNG·DPI export는 1:1 측정 증거로 승격하지 않으며 review image로만 취급한다. 아래 canonical Digital Rubbing PNG만 별도의 raster 계약을 가진다.

## 검증 제원 `measurement.geometry_metrics.v1`

native 제원 버튼은 mutable scene의 cache, object scale 또는 Trimesh 표시값을 저장하지 않는다. 활성 source/metadata/Align을 fresh canonical world millimeter로 materialize하고, `artifact_geometry_metrics.py`가 전체 triangle mesh를 명시적인 1 µm 격자에 ties-to-even으로 양자화한 뒤 receipt를 만든다. machine-readable 계약은 `schemas/geometry_metrics_receipt-1.0.0.schema.json`이다.

```text
DerivedRecord(type=measurement.geometry_metrics.v1)
├── geometry_ref: urn:archmeshrubbing:geometry-metrics-receipt:sha256:<JCS digest>
├── recipe
│   ├── coordinate_space: canonical_aligned_mm/v1
│   ├── coordinate_grid_um: 1
│   ├── rounding_mode: round_ties_to_even
│   └── volume_policy: single_closed_consistently_oriented_edge_manifold_component/v1
└── receipt
    ├── integer grid bounds와 최대 양자화 변위
    ├── surface_area.decimal_mm2: 고정 6자리
    ├── topology: boundary/non-manifold/orientation/duplicate/degenerate/component QC
    └── volume
        ├── decimal_mm3: 고정 9자리 presentation
        └── exact_rational_mm3 + signed_six_grid_units3
```

- 표면적은 양자화된 모든 triangle의 면적을 고정 순서로 합산하며 빈 결과나 양의 면적이 없는 결과를 거부한다.
- 체적은 boundary edge 0, non-manifold edge 0, orientation mismatch 0, duplicate/degenerate face 0인 단일 연결 component에서만 계산한다. 이 v1 조건은 full vertex-manifold 증명을 주장하지 않으므로 정책 이름에 `edge_manifold`를 명시한다.
- 유효한 체적은 grid 정수 좌표에서 계산한 signed six-volume을 exact rational mm³로 약분해 보존한다. 9자리 decimal은 이 유리수에서 재계산하며 독립 권위 값이 아니다.
- 열린 scan, 서로 분리된 조각, 방향이 뒤집힌 일부 face, 비다양체에서는 표면적과 topology QC는 기록하되 체적 field를 `unavailable_topology`로 닫는다. convex-hull 근사값이나 자동 repair 결과를 체적으로 대체하지 않는다.
- receipt, byte length, RFC 8785 hash, geometry ref, recipe grid와 QC 복제값을 project load/save마다 known-record registry에서 다시 검증한다. 현재 독립 package export는 없으며 self-contained `.amr`가 이 record의 전달 단위다.

## 표면 anchor 거리·원 맞춤 지름

native 문서의 point 측정은 화면에서 얻은 float 좌표만 보존하지 않는다. exact `RenderFrameSnapshot`의 depth unproject와 ray를 사용하고, 전체 projected triangle을 bounded CPU ray/triangle 교차로 검사한 뒤 framebuffer depth point와 가장 가까운 hit를 source topology의 face row·vertex 순서에 다시 매핑한다. centroid-nearest face 추정은 권위 pick에 사용하지 않는다. machine-readable 계약은 `schemas/surface_measurement_receipt-1.0.0.schema.json`이다.

```text
DerivedRecord(type=measurement.surface_distance.v1 | measurement.circle_diameter.v1)
├── selection_hash: source face+barycentric anchor 배열의 JCS digest
├── recipe
│   ├── coordinate_space: canonical_aligned_mm/v1
│   ├── coordinate_grid_um: 1, rounding_mode: round_ties_to_even
│   ├── pick_method: frame_depth_unproject+cpu_ray_triangle/v1
│   └── anchor: source face_index + face_vertex_indices
│       + barycentric_numerators / 1,000,000,000
└── receipt
    ├── anchor별 resolved point·depth residual·pixel footprint·edge 상태
    ├── distance: exact squared-distance fraction + 6자리 mm
    ├── diameter: PCA plane + normalized algebraic Kasa circle, center·normal·condition·residual + 6자리 mm
    └── quality: pass | review와 닫힌 review reason
```

- `measurement.surface_distance.v1`은 정확히 두 anchor 사이의 3차원 Euclidean chord다. mesh 표면을 따라가는 geodesic 거리, 화면 투영 거리 또는 axis-aligned 폭이 아니다.
- `measurement.circle_diameter.v1`은 3~64개 anchor의 PCA best-fit plane 위에서 좌표를 정규화한 뒤 대수 Kasa 방식으로 맞춘 원의 지름이다. 기하학적 radial least-squares가 아니며, 유물 전체의 maximum diameter, bounding-box 폭 또는 구 지름도 아니다. collinear/duplicate 입력, rank 부족, 평면 고유값 분리가 불충분한 입력, 비양수 반지름과 정책을 넘는 condition은 숫자를 만들지 않고 실패한다.
- source face row와 10억 분율 barycentric weight가 durable anchor다. render origin, framebuffer depth와 ray는 source triangle을 찾기 위한 transient 관측값이며 document 권위로 저장하지 않는다. project reopen은 record가 참조한 과거 metadata·Align matrix로 source vertices를 다시 materialize하고 anchor point, 결과, receipt hash와 QC를 재계산한다.
- pick QC는 depth→CPU hit residual, 한 pixel의 world footprint, triangle edge 근접도와 screen search offset만 보고한다. 지름은 여기에 plane/radial RMS·maximum residual과 fit condition을 더한다. 이는 화면 선택·재부착의 이산화와 원 맞춤 품질에 대한 검사이며 scanner calibration, mesh reconstruction, 표면 결손, 연구자 point 선택을 합친 총 측정불확도로 해석하면 안 된다. `review`도 값의 통계적 신뢰구간이나 현장 적합 판정이 아니다.
- 과거 float world-point 기반 Shift+클릭 결과는 계속 검토용 legacy UI로만 취급하며 이 두 record type으로 자동 승격하지 않는다. 현재 전달 단위는 known-record 검증을 거치는 self-contained `.amr`이고 독립 측정 export는 없다.

## Authoritative tile unwrap record와 `.amr-unwrap`

자유 flatten UI는 축 추정, smoothing, fallback을 허용하므로 검토에는 유용하지만 측정 record의 입력 계약으로 사용하지 않는다. `artifact_tile_unwrap_extractor.py`는 확정 Align의 canonical-mm mesh에서 별도의 엄격한 recipe를 실행한다.

```text
DerivedRecord(type=surface.tile_unwrap.v1)
├── geometry_ref: urn:archmeshrubbing:tile-unwrap:sha256:<payload digest>
├── selection_hash: canonical source-face range selection digest
├── recipe
│   ├── explicit longitudinal_axis: x | y | z
│   ├── record_view: top | bottom
│   ├── n_sections: 12..96
│   ├── seam_policy: minimum_angular_range_auto | fixed_angle_microdegrees
│   ├── seam_angle_microdegrees: null | integer [-180000000, 180000000)
│   ├── coordinate_quantum_um: 1
│   └── fallback_policy: reject
├── qc: section fit + integer distortion millionths + foldover/collapse counts
└── extensions[org.archmeshrubbing:tile-unwrap-v1]
    └── bounded receipt + receipt SHA-256
```

- selection은 정렬·중복 제거·최대 병합한 `[start, end_exclusive]` face range로 recipe에 남긴다. topology와 전역 UV 겹침을 완전 검사하기 위해 한 기록면은 최대 250,000 faces로 제한하며, output은 local vertex/face와 canonical source vertex/face row의 대응을 모두 보존한다.
- sectionwise 1.2 계산은 단면별 중심·반경 뒤 굽힘/비틀림에서 빠진 longitudinal U shift를 실제 3D edge 길이에 맞춰 결정적으로 보정한다. 자동 seam은 각 단면의 최소 angular range 경계를 사용하고, 고정 seam은 canonical 장축에서 파생한 결정적 단면 기준축에 대한 0.000001° 정수 각도를 모든 section에 적용한다. cylinder fallback, 희박한 section fit, mean/p95 distortion gate 초과는 READY record로 만들지 않는다. 면 최대 distortion 25%는 `section_center_policy`가 `fit_per_section`일 때만 게이트다. `canonical_axis_origin`(회전축 위의 토기 띠)에서는 면 하나의 왜곡이 중심 오차가 아니라 벽의 요철이므로 `distortion_max_millionths`로 보고만 한다.
- UV는 1 µm 정수 격자로 양자화한다. 모든 삼각형의 세 edge, 면적비와 local 3D→2D Jacobian singular value를 평가하며, 격자 붕괴·orientation foldover, 다중 edge-connected component, 중복 face, non-manifold/inconsistent edge, branched boundary 또는 positive-area 전역 UV 겹침이 하나라도 있으면 실패한다.
- canonical binary는 RFC 8785 header와 `uv int64le`, `faces int32le`, source vertex/face indices를 domain-separated length-prefix framing으로 묶는다. 파일 전체 SHA-256이 receipt의 `unwrap_sha256` 및 `geometry_ref`와 같다.
- Top과 Bottom은 같은 face selection의 U 방향을 구분해 recipe와 payload hash가 달라지는 독립 해석 결과다. 자동/고정 seam도 recipe hash와 재계산 입력에 포함되며, 고정 경계가 선택 표면을 갈라 foldover·전역 UV 겹침을 만들면 결과를 게시하지 않는다. 현재 runtime은 실제 기와 상·하면을 자동 분류하지 않으므로 기록자가 올바른 단일 기록면 faces를 선택해야 하며, 펼친 좌표 위 texture·Digital Rubbing 재투영도 이 계약에 포함되지 않는다.
- 동일 recipe 재계산은 receipt 전체와 binary bytes가 같아야 한다. Align을 바꾼 과거 record는 삭제하지 않고 `stale_alignment`로 남긴다.

`*.amr-unwrap`은 정확히 네 regular file을 갖는 non-overwriting directory package다.

| normative member | 역할 |
|---|---|
| `artifact.amr-unwrap.bin` | 독립 parser가 다시 읽을 수 있는 authoritative quantized mesh와 source-row correspondence |
| `artifact.obj` | millimeter 단위의 평면 삼각 mesh derivative |
| `artifact.svg` | single-incident-face boundary로 만든 실제 mm `width`/`height`/`viewBox` 1:1 derivative |
| `artifact.amr-unwrap.json` | receipt, recipe, QC, source/document/revision provenance와 세 artifact의 exact-byte hash |

sidecar의 artifact descriptor를 제외한 claim을 RFC 8785 SHA-256으로 묶어 SVG metadata에도 저장하므로 한 member의 독립 손상을 검출한다. validator는 binary를 parse해 receipt를 다시 만들고 OBJ/SVG exact bytes를 재렌더하며, sidecar claim·artifact hash·privacy·READY/FRESH provenance를 대조한다. 원본 document 없이도 package 내부 무결성과 physical scale을 offline 검증할 수 있고, document를 함께 주면 exact record와 manifest까지 비교한다.

writer는 같은 parent의 숨은 staging directory에서 네 파일을 모두 쓰고 자체 검증한 뒤 atomic no-replace rename한다. 기존 destination을 덮지 않으며 destination race의 승자를 보존한다. 현재 tile package writer는 소유 staging의 device/inode, closed member set과 regular-file 상태를 확인할 수 있을 때만 실패 정리를 수행한다. 이 hash도 제작자 서명은 아니다. desktop은 선택한 `READY + FRESH` record의 recipe/receipt를 worker에서 재계산·검증하고 prepared capability를 받은 뒤 current Workbench 권위를 다시 확인해 게시한다. 이 재계산은 live 취소 Event를 section circle/seam fitting과 row-shift section/grid/refinement의 명시적 경계까지 전달하지만, 현재 실행 중인 단일 NumPy·선형대수 호출은 반환 전에 선점하지 않는다. 사용자 취소나 앱 종료는 publication 권위를 먼저 회수하고 worker join 동안 owned staging 정리를 기다린다.

현재 machine-readable 계약은 record receipt의 `schemas/tile_unwrap_receipt-1.1.0.schema.json`과 export sidecar의 `schemas/tile_unwrap_export-1.4.0.schema.json`이다. export schema는 자동 seam 1.1 recipe, 자동/고정 seam 1.2 recipe, 단면 중심·station 정책을 가진 1.3 recipe를 닫힌 형태로 구분하며, current writer는 1.4 sidecar를 만든다. 1.4가 1.3과 다른 것은 한 가지 상한이다: recipe의 `section_center_policy`가 `canonical_axis_origin`이면 `qc.record.distortion_max_millionths`가 25%를 넘을 수 있고, `fit_per_section`이면 1.3처럼 25%가 상한이다. 공개된 `tile_unwrap_export-1.1.0.schema.json`·`1.2.0`·`1.3.0`의 bytes와 recipe hash는 바꾸지 않고 runtime이 기존 패키지와 프로젝트를 계속 offline 검증·재계산한다. 1.4 이전 sidecar는 정책과 무관하게 25%를 넘는 record를 담지 못한다. 두 세대 모두 axis 추정값, fallback 허용값, 사설 경로와 계약 밖 필드를 거부하고 topology·전역 UV overlap·row-shift QC를 고정한다. `section_row_shift_station_count == section_count` 같은 cross-field 의미 제약은 runtime known-record validator가 최종 강제한다. 공개 릴리스 전에 사용된 실험적 1.0 schema 파일은 byte-exact 회귀와 계약 이력 검토를 위해서만 보존한다. 현재 runtime은 1.0 record/package와 이를 포함한 프로젝트를 읽지 않으며 자동 migration도 제공하지 않는다.

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
- face order·winding·duplicate, hole, large absolute survey offset과 늦은 Align 결과를 Windows 차단 테스트로 검증한다. barycentric edge rule은 recipe에 명시하며 canonical raster/export golden을 Windows x64/CPython 3.12 기준선에 고정한다.

GUI 계산은 worker에서 시작 당시 immutable `ArtifactSession`만 사용한다. 완료 callback은 exact result capability와 source/render projection을 검사한다. 같은 Align에서 다른 record가 중간에 추가되면 captured computation을 current immutable descendant document에 rebase하므로 기존 기록을 잃지 않으며, Align/Open authority가 바뀐 late result만 폐기한다. candidate publication은 `prepare_record_commit()`의 session/version/epoch CAS, append-only ancestor와 expected record ID 집합 검증을 다시 통과한다. record append는 SceneObject binding만 교체하고 VBO를 다시 만들지 않는다. 미리보기는 receipt와 맞는 최초 계산 raster만 표시하며 export 권위로 재사용하지 않는다. 재개방 기록의 background preview가 진행 중일 때 같은-Align record가 추가돼도 pending selection을 보존하고 완료 시 exact record/receipt를 다시 확인한다.

### Canonical PNG와 `.amr-rubbing`

`*.amr-rubbing`은 새로 만드는 non-overwriting directory package이며 정확히 두 regular file만 허용한다.

| normative member | 역할 |
|---|---|
| `artifact.png` | canonical GA8 1:1 planar raster derivative |
| `artifact.amr-rubbing.json` | recipe, raster receipt, QC, source/document/revision provenance와 PNG exact-byte hash |

PNG writer는 `IHDR → sRGB → pHYs → iTXt → IDAT → IEND` 순서, row filter 0, 직접 작성한 stored DEFLATE block을 사용한다. 따라서 Pillow encoder나 zlib compression heuristic에 exact bytes를 맡기지 않는다. `pHYs`의 X/Y pixels-per-meter는 receipt와 같고, width/height와 pixel pitch는 sidecar에서 exact rational millimeter로 반복 선언한다. primary PNG에는 scale bar, 글꼴, review label, 절대 source path나 외부 resource를 넣지 않는다.

sidecar의 artifact descriptor를 제외한 normative claim 전체를 RFC 8785 SHA-256으로 묶어 PNG iTXt metadata에 넣고, sidecar artifact descriptor는 PNG exact-byte SHA-256과 byte length를 가진다. validator는 PNG CRC/Adler/chunk/metadata/pixel hash/semantic raster hash, receipt, recipe, public provenance, scale와 privacy 선언을 모두 대조하며 closed provenance mapping의 계약 밖 key도 거부한다. 원본 document가 없어도 이동한 package를 별도 PID에서 offline 검증할 수 있고, document를 함께 주면 READY + FRESH record와 manifest까지 비교한다. mesh admission provenance를 포함하고 여섯 뷰 receipt와 전개 receipt를 모두 허용하며 종이 기저 농담까지 담는 현재 machine-readable sidecar 계약은 `schemas/rubbing_export-1.3.0.schema.json`이며, 공개된 1.0.0·1.1.0·1.2.0 schema bytes와 offline validator 호환성은 보존한다.

writer는 vector package와 같은 same-parent staging, prepared inode/fingerprint capability, file/directory `fsync`, self-validation, OS별 atomic no-replace publish를 사용한다. staging 이름은 배타적으로 예약하고 충돌한 foreign directory를 재사용·삭제하지 않는다. application cleanup도 등록한 device/inode가 그대로일 때만 수행한다. 기존 목적지, 추가 member와 symlink는 거부하며 `.DS_Store`, `Thumbs.db`, `desktop.ini`만 vector와 같은 1 MiB 제한으로 무시한다. final rename 뒤 실제 또는 미지원 directory `fsync`는 destination이 이미 게시된 `committed=true` 오류이며 GUI는 저장 완료와 crash durability 미확정을 구분한다. 이 package의 hash도 무결성 값이며 제작자 전자서명은 아니다.

### 전개 좌표 위의 탁본 `raster.developed_rubbing.v1`

여섯 뷰 탁본이 유물을 옆에서 본 것이라면, 전개 탁본은 `surface.tile_unwrap.v1` record가 증명한 전개 좌표 (u, v) 위에 같은 relief를 그린 것이다. 깊이는 전개 중심(정치한 회전축, 또는 단면마다 맞춘 중심)에 대한 반지름이고, 양자화·local-mean relief·먹 매핑은 Digital Rubbing과 같은 정책 블록을 공유한다. 배경은 [`docs/POTTERY_STRIP_UNWRAP.md`](POTTERY_STRIP_UNWRAP.md)에 있다.

```text
DerivedRecord(type=raster.developed_rubbing.v1)
├── geometry_ref: urn:archmeshrubbing:developed-rubbing-raster:sha256:<digest>
├── depends_on_record_ids: [<tile unwrap record>, ...]   (전개 record는 항상 포함)
├── recipe
│   ├── development: { record_id, record_type, recipe_hash, unwrap_sha256 }
│   ├── depth_policy.measure: radius_about_unrolling_centre/v1
│   └── pixel/relief/resource policies: Digital Rubbing과 동일
├── qc: raster/ink counts, development face/vertex/distortion, radius span
└── extensions[org.archmeshrubbing:developed-rubbing-v1]
    └── receipt: development_sha256 + lattice origin/dimensions + exact mm + hashes
```

다시 열 때는 receipt·recipe·QC의 정합성에 더해 recipe가 이름 지은 전개 record가 문서에 있고, 그 receipt의 `unwrap_sha256`과 recipe 해시가 탁본 recipe가 적은 값과 같은지 확인한다. raster를 재계산할 때는 전개 record의 recipe로 전개를 다시 계산해 receipt와 payload 해시가 맞아야만 탁본을 그린다. 전개가 STALE이면 탁본도 STALE이다.

내보내기는 같은 `.amr-rubbing` package를 쓴다. sidecar schema 1.2.0부터 `raster_receipt`에 여섯 뷰 receipt(`rubbing_receipt-1.0.0`)와 전개 receipt(`developed_rubbing_receipt-1.0.0`) 중 하나를 허용하고, `recipe.kind`가 `developed_rubbing`이면 receipt는 전개 receipt, provenance record type은 `raster.developed_rubbing.v1`이어야 한다. 1.0.0·1.1.0 package는 그대로 검증되며, 전개 탁본은 1.2.0 이상에, 종이 기저 농담을 깐 탁본은 1.3.0에만 들어간다 — 1.2.0의 여섯 뷰 recipe 정의는 고정된 1.0.0에서 오므로 그 열쇠들을 담지 못한다. writer는 전개를 재계산해 receipt·payload 해시를 맞춘 뒤에야 raster를 다시 그리고 package를 만든다.

## 완료 3/6/6 원자 묶음 `.amr-survey`

`*.amr-survey`는 새 알고리즘 결과를 만드는 포맷이 아니라, 한 active Align 아래에서 dependency-valid `READY + FRESH`인 Cutline 3면, Outline 6면, Digital Rubbing 6면의 기존 권위 package를 한 번에 전달하는 non-overwriting directory다. GUI의 `완료 실측 15개 원자 묶음 내보내기`는 세 단계가 모두 완료된 경우에만 활성화된다.

부모 directory에는 정확히 다음 16개 normative entry가 있다.

- `cutline-{top,front,right}.amr-vector/` 3개
- `outline-{top,bottom,front,back,right,left}.amr-vector/` 6개
- `rubbing-{top,bottom,front,back,right,left}.amr-rubbing/` 6개
- `survey.amr-survey.json` canonical aggregate manifest 1개

각 자식은 기존 vector/rubbing validator를 그대로 통과해야 한다. 부모 manifest는 15개 entry를 위 canonical 순서로 고정하고 step/view/record ID, child directory, primary·sidecar 파일 이름·크기·SHA-256, physical scale을 기록한다. `artifact_set_sha256`은 이 entry 배열의 RFC 8785 SHA-256이다. 공통 authority는 document ID/canonical SHA-256, source metadata·geometry·active Align revision, 모든 source asset SHA-256·크기를 결합하며 `qc.coverage_complete=true`, `vector_count=9`, `rubbing_count=6`, `artifact_count=15`를 요구한다. machine-readable closed contract는 `schemas/survey_export-1.0.0.schema.json`이다.

writer는 같은 parent의 숨은 staging root에 15개 자식을 생성하며, Rubbing 6개는 각 record recipe로 다시 계산해 durable receipt와 비교한다. 모든 자식과 canonical manifest를 재검증하고 전체 tree의 closed entry set, regular-file 상태, device/inode/size/time fingerprint와 parent/destination identity를 일회성 prepared capability에 묶는다. GUI callback은 캡처한 source/render projection과 15개 exact record가 모두 현재 `READY + FRESH`인지 Workbench lock에서 한 번에 확인한 뒤 부모 directory를 atomic no-replace rename한다. 따라서 외부에서 보이는 결과는 15개가 모두 있는 package 하나이거나 아무것도 없는 상태다. 취소·stale Align·Open 전환·목적지 경합에서는 소유가 증명된 staging만 격리·정리하며 foreign path를 삭제하지 않는다. 이 원자성은 **완료된 결과의 배포**에 대한 것이고, 15개 record의 계산과 commit 자체가 하나의 트랜잭션이라는 뜻은 아니다.

이 manifest hash도 전자서명이 아니다. 독립 validator는 이동된 package를 원본 mesh 없이 검증할 수 있고, exact `.amr`를 함께 주면 15개 record와 document authority까지 모두 대조한다.

## 통합 offline verification receipt

설치된 실행 파일과 source checkout은 같은 공개 명령을 사용한다.

```text
ArchMeshRubbing.exe --verify-artifact PATH [--against-project PROJECT.amr] [--report REPORT.json]
```

`src/core/artifact_verification.py`는 입력 이름이나 디렉터리 suffix를 신뢰하지 않고 exact sidecar marker로 `.amr-vector`, `.amr-rubbing`, `.amr-survey`, `.amr-unwrap`를 판별한다. 심볼릭 링크, 두 종류 marker가 섞인 디렉터리, 알 수 없는 regular file은 format validator에 들어가기 전에 거부한다. 판별 뒤에는 위 각 절의 기존 validator를 그대로 호출하므로 통합 명령이 scale·recipe·QC·provenance 규칙을 별도로 느슨하게 재구현하지 않는다.

`.amr` 성공은 manifest와 checksum만 읽었다는 뜻이 아니다. `load_artifact_session_project()`가 embedded source closure를 saved parser/import recipe로 다시 읽고 source byte SHA-256, geometry SHA-256, metadata/Align matrix, known records를 검증하고 canonical scene을 materialize해야 한다. 따라서 호환 목적으로 읽을 수 있는 manifest-only artifact project는 이 완전 offline receipt에서는 실패한다.

export package는 기본적으로 package 내부 public provenance에 대해 self-contained 검증한다. `--against-project`를 주면 해당 `.amr`를 먼저 위와 같은 완전한 방식으로 재개방하고, export validator에 그 exact `ArtifactDocument`를 넘긴다. record ID, `READY + FRESH`, payload/receipt, recipe, QC, dependency closure, source/document/revision provenance 중 하나라도 다르면 성공 receipt가 나오지 않는다. project 입력 자체에는 `--against-project`를 적용하지 않는다.

성공 receipt는 artifact 종류별 exact-byte hash와 validated presentation, recipe, QC, provenance를 담는다. project receipt는 document/source/geometry hash, active metadata·Align과 canonical matrix, materialized vertex/face count, record 상태 집계를 담는다. 입력의 basename만 기록하며 절대 input/project path와 실행 시각은 넣지 않는다. 동일 bytes와 동일 authority mode의 report JSON 값은 결정적이다. machine-readable closed 계약은 `schemas/offline_verification_report-1.0.0.schema.json`이다.

`--report`는 `write_json_report()`의 create-new 경계를 사용해 기존 영수증을 덮어쓰지 않는다. 성공은 exit `0`, 검증 실패는 `1`, 명령 형식이나 report 저장 실패는 `2`다. `--report`가 없을 때만 compact sorted JSON 한 줄을 stdout으로 보낸다. 이 receipt의 SHA-256들은 무결성·재현 증거이지 제작자 인증 서명이나 신뢰 anchor가 아니다.

## 원자적 저장 절차

`save_project()`와 `save_artifact_project()`는 다음 순서로 동작한다. artifact API는 쓰기 전에 public representation을 `ArtifactDocument.from_dict()`로 다시 검증하고 canonical bytes가 원본 문서와 같은지도 확인한다.

1. strict JSON 직렬화: NaN/Infinity, non-string object key를 거부한다.
2. 목적 파일과 같은 디렉터리에 고유 임시 파일을 만든다.
3. ZIP을 닫고 file buffer를 flush한 뒤 `fsync`한다.
4. 선택된 payload 종류의 production loader로 임시 파일을 다시 열어 checksum·schema·payload를 검증한다. artifact payload는 artifact loader로 다시 연다.
5. Windows에서는 검증된 임시 파일과 목적지를 extended Unicode path로 바꾼 뒤 `MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)`로 한 번에 commit한다.
6. 내부 비 Windows backend는 회귀 시험에서만 `os.replace`와 parent directory `fsync`를 실행한다. 제품 저장 계약은 5번의 Windows 경로다.

write, file `fsync`, 재검증, Windows write-through move 중 어느 단계에서 실패해도 기존 목적 파일의 바이트는 유지되고 해당 임시 파일은 정리된다. Win32 commit 실패에는 `committed=false`를 보고하고 다른 rename으로 fallback하지 않는다. 내부 비 Windows backend는 directory `fsync` 미지원 오류와 replace 뒤 실제 I/O 오류를 구분하는 회귀 시험을 유지하지만 제품 gate와 지원 계약에는 포함되지 않는다. 프로세스·전원 중단으로 `finally`가 실행되지 못한 경우에만 같은 parent에 임시본이 남을 수 있으며, 아래 수동 복구 경계가 이를 다룬다.

## 비정상 종료 저장 임시본 복구

복구는 시작 시 background scan이나 자동 정리로 실행하지 않는다. 사용자가 GUI에서 폴더 하나를 명시적으로 고르면 `discover_interrupted_project_saves()`가 그 폴더의 regular, non-symlink entry 중 writer의 exact `.<destination>.XXXXXXXX.tmp` 이름만 최대 64개까지 최신순으로 제시한다. `XXXXXXXX`는 Python `mkstemp()`가 만드는 8자의 소문자·숫자·underscore token이고, filename에서 도출한 intended destination은 `.amr`여야 한다. discovery는 후보의 유효성을 주장하지 않으며 device/inode, 크기와 수정시각을 고정할 뿐이다.

사용자가 후보와 **존재하지 않는 새 `.amr` 목적지**를 각각 확인한 뒤에만 다음 복구가 worker에서 실행된다.

1. 후보 filename과 intended destination 결합, regular-file identity·크기·수정시각을 다시 확인한다.
2. `O_NOFOLLOW`를 사용할 수 있는 플랫폼에서는 이를 포함해 후보 descriptor를 열고, 발견 시 identity와 같은 descriptor인지 확인한다.
3. 새 목적지 parent의 `.<new-destination>.XXXXXXXX.tmp` staging으로 descriptor bytes를 bounded streaming copy하고 file `fsync`한다. copy 전후 descriptor와 candidate path identity가 같아야 한다.
4. staging을 `load_artifact_session_project()`로 완전 재개방한다. 즉 checksum·source index만 읽는 것이 아니라 embedded source closure를 saved parser로 decode하고 source/geometry SHA-256, 단위, Align과 canonical projection까지 물질화해야 한다. manifest-only artifact, legacy payload, 깨진 ZIP과 불완전 source closure는 실패한다.
5. Windows는 검증된 staging과 새 목적지를 extended Unicode/UNC path로 바꿔 `MoveFileExW(MOVEFILE_WRITE_THROUGH)`로 게시한다. replace flag를 주지 않으므로 목적지가 이미 있거나 경합 중 생기면 `ERROR_FILE_EXISTS | ERROR_ALREADY_EXISTS`를 create-new 실패로 변환하고, 다른 Win32 오류에서도 기존 승자를 보존한 채 더 약한 rename으로 fallback하지 않는다.
6. 내부 비 Windows backend는 native atomic no-replace rename과 directory `fsync`를 회귀 시험할 때만 사용하며 제품 복구 경로로 문서화하지 않는다.
7. published path가 검증 staging과 같은 inode인지 확인한다. Windows 성공은 write-through publication backend를 receipt에 남긴다. 내부 backend receipt는 제품 지원·배포 증거로 사용하지 않는다.

복구 성공·실패와 무관하게 발견한 중단 임시본과 원래 intended destination은 수정·이동·삭제하지 않는다. publication 후 현재 GUI scene도 자동으로 교체하지 않으며 사용자가 복구본 열기를 다시 확인해야 한다. 복구 도중 다시 중단되면 새 목적지 parent에 같은 writer-compatible temp가 남을 수 있어 같은 절차로 재검증할 수 있다.

## 엄격한 로딩과 안전 한도

reader는 다음을 거부한다.

- invalid UTF-8, duplicate JSON key, NaN/Infinity
- duplicate·encrypted·위험한 경로의 ZIP member
- 지원하지 않는 압축 방식
- checksum 또는 CRC 불일치
- 64개 초과 member
- 64 MiB 초과 `project.json`
- 256 MiB 초과 일반 단일 member 또는 일반 member 총 512 MiB
- 16 GiB 초과 source blob 또는 content-addressed source blob 총합
- 500:1 초과 압축 비율
- 약 16.5 GiB 초과 물리 컨테이너 또는 8 MiB 초과 ZIP central directory

reader는 `ZipFile`을 만들기 전에 EOCD/ZIP64와 bounded central-directory record를 직접 대조한다. EOCD entry count를 작게 위조해도 실제 record가 64개를 넘으면 `ZipInfo`를 만들기 전에 거부한다.

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
- 61개 manifest entry를 넘는 대규모 source closure의 순차 descriptor streaming과 확장된 package member budget
- 여러 material/PBR 조합을 원본 scanner와 동일하게 재현하는 viewport rendering fidelity
- legacy `legacy_ui_state`에서 `artifact_document`로 단위·geometry·Align을 추정하지 않는 보존적 migration
- 3/6/6 record를 하나의 computation/commit 트랜잭션으로 만드는 기능. 완료된 15개 결과의 배포 원자성은 `.amr-survey`로 보장하지만 각 view 계산·record append는 독립 작업이다.
- 실제 GPU driver frame의 scene-swap 원자성 및 시각적 동일성
- Windows native-QPA와 frozen executable에서 검증한 `>= 1e9 mm` 장면의 millimeter 이하 visual/depth-picking 정밀도
- autosave, 시작 시 자동 crash-recovery scan·자동 후보 삭제
- 전자서명 또는 provenance authority 인증

현재 native GUI는 source와 새 scene object를 기존 live scene과 분리해 load·검증하고, 준비된 projection을 검증한 다음 `ArtifactWorkbench` authority와 scene을 two-phase로 교체한다. missing, parse failure, hash mismatch, VBO 준비 또는 swap 실패 시 staging만 폐기하고 이전 scene·session·저장 경로를 복원한다. scene 교체는 projection generation을 증가시키고 이전 cut-section/ROI worker authority를 분리한다. 해당 callback은 current worker identity, generation과 selected object가 모두 일치할 때만 overlay를 갱신하며, 오래된 finished callback은 새 worker pointer를 지울 수 없다. surface/visible-face callback은 target object·mesh·TRS·render-frame 계약을 추가로 확인해 A 유물의 face ID를 B 유물에 적용하지 않는다. rollback·scene 복원·finalize가 불확실하면 fatal state가 모든 저장·실측·내보내기를 차단한다. offscreen smoke는 이 transaction, worker fencing과 rollback 순서를 검증하지만 실제 GPU driver가 표시한 프레임 사이에 부분 장면이 전혀 노출되지 않는지까지 측정하지 않는다.

### 대좌표와 transient render-origin 경계

문서, geometry identity, Align, 파생 기록, QC와 export 좌표는 origin을 보존한 **절대 float64 world millimeter**를 계속 사용한다. viewport는 이를 수정하거나 recenter하지 않는다.

표시 경계에서는 객체별 local VBO origin `O`를 mesh bounds의 midpoint에서 정하고 `q = float32(v - O)`를 CPU float64 연산 뒤 업로드한다. live scene에는 안정적인 world origin `R`을 두며, local-to-world affine `M`은 `T(-R) @ M @ T(O)`로 rebase하고 camera eye/target에서도 `R`을 뺀다. `O`와 `R`은 scene/VBO 수명에만 존재하는 private runtime 상태다.

native vector preview·ground/grid와 cutline·ROI·pick·gizmo 등 활성 world overlay는 world point에서 `R`을 CPU float64로 빼고 제출한다. local marker/HUD primitive는 각 local/pixel 좌표를 유지한다. CPU face centroid와 face 계산은 absolute float64를 유지하며, 고해상도 capture가 반환하는 modelview는 기존 absolute-world 소비자와 호환되도록 복원한다.

depth pick·screen projection·ray·Ctrl drag은 해당 depth buffer를 그린 modelview·projection·viewport·`R`을 `RenderFrameSnapshot`으로 고정한다. 같은 paint depth pass의 visibility·selected object·ROI bounds/caps·X-ray·solid-shell·all-object TRS·VBO geometry revision을 별도 transient depth signature로 원자적으로 함께 게시한다. resize·scene swap·projection generation 변경·object transform 뒤와 state 변경 후 repaint 전의 live snapshot은 폐기하고, 드래그는 press-time snapshot을 release/reset까지 유지한다. 선택 객체가 바뀌면 worker·gizmo/ROI gesture·미완성 surface polygon을 종료한다. 이 계약만으로 실제 GPU depth 정밀도가 증명되는 것은 아니며, 별도 native-process smoke가 실제 widget FBO readback과 pick을 검증한다.

두 transient origin `O`와 `R`은 다음 권위 데이터에 들어가면 안 된다.

- `SourceMetadataRevision` 또는 `AlignRevision`
- source/geometry hash framing
- 저장된 record·QC·selection
- SVG/3D export의 world 좌표

현재 게이트는 pure coordinate algebra, mocked relative VBO/overlay submission, frame-bound project/unproject/depth-pick 수명주기, absolute float64 face 계산, source/scene materialization 불변과 document canonical hash 비직렬화를 검증한다. 별도 `src.gui.opengl_driver_smoke`는 native QPA의 실제 `Viewport3D` widget FBO에서 `[1e9,-2e9,3e9] mm` 장면, relative VBO, 두 depth component, gap 예상 지점의 background와 overlay 예상 위치의 green pixel, 0.125 mm depth-pick 복원을 원근·정사영으로 검증한다. Windows 대상 commit `b12d4874a4a8`의 source CI와 frozen executable은 qwindows+bundled llvmpipe gate를 통과했다. 대표 Windows 하드웨어 GPU와 compositor 최종 presentation은 아직 별도 게이트다.

### Legacy destructive bake 임시 안전 조건

현재 `bake_object_transform()`과 일부 floor/brush Align 경로는 표시 TRS를 `MeshData.vertices`에 직접 굽고 TRS를 identity로 되돌린다. `legacy_ui_state` writer는 이 변경된 vertex payload나 bake 전 행렬을 저장하지 않고 원본 source reference와 현재 identity TRS만 저장한다. 새로 발생한 destructive bake는 `_amr_has_unpersisted_bake`와 `legacy_baked_unverifiable`로 표시하며, `_collect_project_state()`는 조용한 데이터 유실 대신 저장을 거부한다. 이 marker가 없던 과거 파일이나 외부 변경에는 baked Align 복원 보장이 없다.

M0-3 native 한-artifact session은 immutable Align commit과 non-destructive scene materialization을 사용하지만 별도의 legacy payload/GUI 경로에는 기존 bake가 남아 있다. 다음을 임시 안전 조건으로 삼는다.

- destructive bake marker가 있는 scene은 현재 legacy AMR Save/Save As를 차단한다.
- 과거 `legacy_ui_state` 파일의 baked 상태를 재현 가능한 Align 기록으로 판정하지 않는다.
- 원본 source를 별도로 보존하고, bake 전 상태와 행렬을 잃을 수 있는 작업을 release-grade workflow로 선전하지 않는다.
- native artifact 왕복 게이트의 성공을 destructive legacy bake 경로의 저장 보장으로 확대 해석하지 않는다. 두 payload mode의 권위와 보장 범위는 분리한다.
