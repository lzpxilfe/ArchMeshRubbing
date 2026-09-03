# Native Packaging and Release Gates

이 문서는 Windows frozen 앱과 검증형 portable ZIP을 만드는 절차를 설명한다. 지원 대상은 Windows 10 version 1809 이상 x64와 Windows 11 x64이며, 최소 버전은 고정한 Qt 6.11의 [공식 지원 플랫폼](https://doc.qt.io/qt-6.11/supported-platforms.html)을 따른다. Windows ARM64·32-bit·Server 최종 사용자 환경, 비 Windows 운영체제와 호환 계층, installer·MSIX·Store 패키지는 지원하거나 배포하지 않는다. CI는 exact Git commit의 corresponding source를 ZIP에 포함하고 한글 경로에 검증·추출해 실제 실행한 뒤 삭제하지만, 서명, artifact 업로드 또는 공개 배포는 아직 수행하지 않는다.

## 지원 빌드 기준

- CPython 3.12
- `requirements/runtime-py312.lock`: 정확한 런타임 버전
- `requirements/build-py312.lock`: 정확한 PyInstaller toolchain 버전
- `requirements/windows-py312-x64-hashed.lock`: Windows x64/CPython 3.12 런타임·빌드 wheel 17개와 각 SHA-256. `--require-hashes --only-binary=:all:`로 설치한다.
- `requirements/runtime-license-policy.json`, `third_party_licenses/`: wheel에 라이선스 원문이 없는 예외의 검토된 출처·원문 SHA-256
- `requirements/public-release-policy.json`, `src/public_release_policy.py`: 프로젝트 `LICENSE` exact SHA-256, PyQt6 wheel `License-Expression`, rights-holder authorization 상태와 공개 바이너리 결정을 묶는 fail-closed 계약
- `tools/build_native.py`: `IsWow64Process2`로 native AMD64 host와 64-bit AMD64 CPython 3.12를 확인하고 exact checkout을 고정하는 유일한 제품 build 진입점
- `ArchMeshRubbing.spec`: Windows x64·64-bit CPython 3.12가 아니면 분석 전에 거부하는 onedir spec
- `src/portable_archive.py`, `tools/build_portable_archive.py`: 고정 root `ArchMeshRubbing`, 비어 있지 않은 `ArchMeshRubbing.exe`, 전체 payload를 강제하는 ZIP 생성·sidecar 검증·안전한 원자적 추출
- `src/source_archive.py`, `tools/build_source_archive.py`: Git object 기반 corresponding-source ZIP/sidecar 생성과 Git 없이 offline 검증
- `src/build_provenance.py`, `tools/generate_build_provenance.py`: portable/source/evidence와 GitHub Actions invocation·runner identity를 묶는 외부 unsigned provenance 생성·offline 검증
- `schemas/portable_archive_manifest-1.0.0.schema.json`: portable sidecar의 machine-readable 계약
- `schemas/source_archive-1.0.0.schema.json`: source ZIP 외부 sidecar와 내부 manifest의 machine-readable 계약
- `schemas/build_provenance-1.0.0.schema.json`: unsigned build-provenance record의 closed machine-readable 계약
- `schemas/offline_verification_report-1.0.0.schema.json`: 받은 `.amr`와 네 export package의 통합 offline verification receipt 계약
- `schemas/surface_measurement_receipt-1.0.0.schema.json`: source triangle+barycentric 거리·원 맞춤 지름 receipt의 closed 계약
- `schemas/field_pilot_review-1.0.0.schema.json`, `field_pilot_report-1.1.0.schema.json`, `field_pilot_verification-1.0.0.schema.json`: 지원 Windows client build/product type·native AMD64·64-bit AMD64 process·CPython 3.12·비호환 계층까지 닫힌 단일 유물·단일 machine의 현장 검토 증거와 재검증 receipt 계약
- `build/generated/build_info.json`: version, channel, commit, runtime lock과 Windows wheel lock SHA-256

source checkout과 frozen build는 Windows x64의 CPython 3.12만 지원한다. `requirements/runtime-py312.lock`은 공통 dependency 입력과 내부 회귀 시험에 쓰이지만 비 Windows 실행 호환성을 약속하지 않는다. 제품 package job은 평탄화된 Windows hash lock만 설치하며 sdist와 검토되지 않은 wheel을 거부한다.

## 안전한 로컬 빌드

깨끗한 native AMD64 Windows의 64-bit CPython 3.12 환경과 clean Git checkout에서 실행한다. 빌드 도구는 `IsWow64Process2`로 x64-on-ARM64 emulation을 거부하고, canonical `git --show-toplevel`이 실제 build root와 정확히 같은지 확인한다. `GIT_DIR`, `GIT_WORK_TREE`, alternate object/index/config 같은 repository override 환경과 replace object를 허용하지 않으며 요청한 `--commit`은 현재 `HEAD`와 정확히 같아야 한다.

clean 판정은 `git status`의 index hint만 믿지 않는다. HEAD의 모든 regular blob path·blob ID·executable mode를 stage-0 index와 대조하고, 열린 worktree의 regular-file identity를 고정한 사이 `git hash-object --stdin-paths`의 Git canonical clean representation을 HEAD blob과 전수 비교한다. 따라서 `assume-unchanged`와 `skip-worktree`로 숨긴 변경도 실패한다. Git의 정상 `text`/`eol` clean 변환은 허용해 Windows CRLF checkout이 HEAD LF blob과 일치할 수 있지만, 외부 `filter`, `ident`, `working-tree-encoding` content transform은 재현 가능한 bytes를 가릴 수 있어 거부한다. 일반 untracked path와 frozen input root 안의 ignored 주입 파일도 거부한다. 이 전체 검사를 manifest·work directory·dist를 만들기 전과 PyInstaller 직후에 똑같이 통과해야만 corresponding source와 release evidence를 발급한다.

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
set PYTHONDONTWRITEBYTECODE=1
python -m pip install --require-hashes --only-binary=:all: ^
  -r requirements/windows-py312-x64-hashed.lock
python tools/build_native.py
```

`PYTHONDONTWRITEBYTECODE=1`은 build 전 정책 검사와 PyInstaller child까지 source-tree bytecode를 새로 만들지 않게 한다. 이미 `src` 아래에 ignored `__pycache__`/`.pyc` 또는 다른 ignored build input이 있으면 그것을 HEAD source로 오인하지 않고 출력 전에 중단하므로, 별도의 새 checkout에서 빌드한다. 기존 `build/ArchMeshRubbing` 또는 `dist/ArchMeshRubbing*`가 있으면 명령은 기본적으로 중단한다. 검토한 생성물만 명시적으로 교체할 때 `--replace-existing`를 사용한다. PyInstaller cache 삭제도 `--clean-cache`를 지정해야만 수행한다. `--commit`은 다른 commit을 대신 포장하는 선택지가 아니라 CI가 checkout identity를 명시적으로 재확인하는 입력이다.

release evidence를 생성·검증한 onedir payload는 다음처럼 portable ZIP으로 묶는다. 출력 ZIP과 manifest는 기존 파일을 덮어쓰지 않는다.

```powershell
$epoch = [int64]((& git show -s --format=%ct HEAD).Trim())
python tools/build_portable_archive.py build `
  --payload dist\ArchMeshRubbing `
  --archive build\ArchMeshRubbing-Windows-x64-portable.zip `
  --manifest build\ArchMeshRubbing-Windows-x64-portable.zip.manifest.json `
  --source-date-epoch $epoch
python tools/build_portable_archive.py verify `
  --archive build\ArchMeshRubbing-Windows-x64-portable.zip `
  --manifest build\ArchMeshRubbing-Windows-x64-portable.zip.manifest.json
```

`build_and_shortcut.py`는 과거 호환 wrapper다. 폴더를 자동 삭제하거나 바탕화면 바로가기를 만들지 않는다.

## Frozen self-test

빌드 도구는 실제 사용자 실행 파일에 `--self-test-report`를 전달하고 결과 JSON의 `ok=true`를 확인한다.

현재 self-test는 다음을 검사한다.

1. embedded build manifest와 runtime/wheel lock hash
2. 정확한 runtime distribution 버전과 Shapely/GEOS 조합
3. 아이콘, runtime/wheel lock, license evidence policy, fail-closed public-release policy, build에 고정된 전체 JSON schema set
4. frozen/portable payload의 전체 파일 SHA-256 manifest, SPDX 2.3 SBOM, 제3자 NOTICE를 실제 bytes에서 재계산
5. exact Git commit/tree/blob·SHA-256·LICENSE에 결합된 corresponding-source ZIP과 sidecar
6. offscreen Qt application
7. 실제 `MainWindow`, 완료 실측 15개 원자 묶음 버튼, native 기와 전개의 전체/선택 face·X/Y/Z·Top/Bottom·section·record/export controls, `QOpenGLWidget`, OpenGL.GL/GLU import와 생성
8. OBJ, PLY, STL, OFF, glTF, GLB를 closed import recipe와 외부 dependency deny resolver로 여는 production parser 경로
9. Pillow PNG encode/decode
10. canonical `ArtifactDocument` round-trip golden
11. canonical Cutline golden
12. canonical Digital Rubbing golden
13. 실제 PLY → 단위/Align session → embedded `.amr` 저장 → 외부 원본 삭제 → source/geometry/Align/world vertex 재검증
14. 실제 application authority의 Open → explicit Align → Cutline 3/3 → Outline 6/6 → Digital Rubbing 6/6 → 검증 제원 1/1 → 표면 거리 1/1 → 원 맞춤 지름 1/1의 18개 record → completed `.amr` offline reopen → 이동된 1:1 SVG/PNG와 원자적 `.amr-survey`의 원본 SHA-256·recipe·QC·aggregate hash 재검증. 제원 record는 1 µm grid에서 cube 정답 24 mm²/8 mm³와 exact rational volume을, 두 anchor의 Euclidean chord와 네 anchor의 PCA 평면·정규화 대수 Kasa 원은 각각 1.000000 mm를 reopen 뒤 다시 확인한다. 별도의 합성 비틀린 원통형 기와 fixture는 외부 원본 삭제 뒤 `.amr` reopen·현재 1.2 recipe 재계산·이동된 `.amr-unwrap` 검증과 동일 payload SHA-256, 13개 station·최대 6,364 µm row-shift를 강제한다. 기존 자동 seam 1.1 recipe/export도 bytes와 hash를 올리지 않고 별도 검증한다. 이는 실물 기와 pilot의 대체 증거가 아니다. 이 check는 통합 verifier로 `.amr` embedded source materialization receipt를 만들고 개별 산출물과 15개 survey 묶음을 exact project에 다시 결합한다. 이어 같은 pair로 field-pilot report를 만들되 실제 human/OpenGL 입력을 만들어내지 않고 `artifact-pass-human-driver-pending`인 fail-closed 상태를 요구한다.

Offscreen `QOpenGLWidget` 생성은 module/plugin 누락을 잡지만 실제 frame 정확성을 증명하지 않는다. Windows CI는 이어서 `QT_QPA_PLATFORM=windows`, `QT_OPENGL=software`로 native `qwindows`와 bundled `opengl32sw.dll`을 사용해 `src.gui.opengl_driver_smoke`를 실행한다. PyOpenGL의 GL/WGL dispatch도 같은 DLL에 결합해 Qt software context와 시스템 `opengl32.dll`이 섞이지 않게 한다. report는 768×768 실제 widget FBO, VBO, pixel/depth/pick과 두 투영 모드를 모두 검사한다.

```bat
set QT_QPA_PLATFORM=windows
set QT_OPENGL=software
python -m src.gui.opengl_driver_smoke ^
  --qt-platform windows ^
  --report build/opengl-driver-smoke.json
```

2026-07-14 통합 verifier 기능 기준 commit `4a21666e7f7b`의 [source CI run 29280873586](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29280873586)은 qwindows+llvmpipe actual-frame 66/66을, [frozen package run 29280874076](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29280874076)은 frozen·한글 경로 14-check complete workflow, outbound deny 상태 public verification receipt, actual-frame 66/66과 cleanup gate를 통과했다. 이 자동 runner 결과는 실제 지원 대상 Windows 10/11 PC의 하드웨어·driver 파일럿을 대신하지 않는다.

현재 OpenGL smoke v2는 이 일반 Windows Server CI 경로를 차단하지 않고 OS build/product type·native/process architecture·Python·호환 계층 self-claim을 report에 함께 저장한다. field-pilot만 성공 root와 전체 필수 check/mode 구조를 닫아 확인하고, 지원 client에서 24시간 안에 수집한 동일 runtime claim과 exact match할 때 통과시킨다. 따라서 CI Server receipt는 package 회귀 증거로는 유효하지만 Windows 10/11 현장 pilot으로 승격되지 않는다.

## Windows portable CI

`.github/workflows/package-smoke.yml`은 `main` push, pull request, 수동 실행에서 Windows 한 환경에 대해 다음을 요구한다.

- Python 3.12와 hash-locked Windows wheel 17개만 설치하고 dependency closure 검사
- public-release policy와 실제 `LICENSE`/설치된 PyQt6 metadata를 대조하고 `blocked` 상태 강제
- 실제 CPython 3.12 구현체와 `IsWow64Process2` native AMD64 host, canonical Git top-level, override 없는 repository, `HEAD == requested commit`, 전수 HEAD blob/mode/live canonical bytes를 출력 생성 전 확인하고 PyInstaller 뒤 재확인
- commit/runtime lock/wheel lock에 결합된 manifest와 PyInstaller onedir 생성
- live worktree가 아닌 exact commit의 Git object에서 corresponding-source ZIP과 canonical sidecar 생성
- payload 전체 파일 manifest, SPDX 2.3 JSON SBOM, machine/human NOTICE 생성·재검증
- Windows frozen executable의 file-based 14-check와 native-QPA software OpenGL 검사
- commit epoch를 고정 timestamp로 사용해 portable ZIP과 canonical sidecar 생성
- ZIP exact hash/size, 모든 entry path/size/SHA-256, release-evidence index와 source commit 재검증
- portable ZIP·manifest, source ZIP·sidecar, release-evidence index, checkout/workflow SHA와 GitHub run·hosted Windows X64 runner identity를 외부 canonical unsigned provenance에 결합·재검증
- `문화유산 기록\ArchMeshRubbing` 한글 경로에 검증 후 원자적으로 추출
- 추출본 release evidence, corresponding-source archive와 provenance를 다시 계산하고 executable에 outbound deny firewall rule과 실패 proxy 적용
- 한글 경로의 실행 파일과 report 경로에서 14-check complete workflow와 native `qwindows` actual-frame 재실행
- outbound deny 상태의 한글 경로에서 public `--verify-artifact ... --report ...` 명령을 실행하고 exit-code 분리, closed failure receipt, 절대 경로 비노출을 검사
- 실행 뒤 ZIP을 다시 검증하고 추출 디렉터리와 firewall rule 제거

각 실행 파일 단계는 120초의 명시적 상한을 가진다. timeout을 성공으로 바꾸거나 GUI hang을 숨기지 않는다. workflow에는 ZIP·실행 파일 artifact upload나 release 단계가 없다. 라이선스 게이트가 해결되기 전에는 CI 밖으로 바이너리를 게시하지 않는다.

### Corresponding-source 계약

`source/ArchMeshRubbing-source.zip`은 live worktree 복사본이 아니다. 생성기는 지정한 full commit의 Git object database에서 tree와 blob을 직접 읽고 100644/100755 regular file만 받는다. symlink·submodule·비 UTF-8/비 NFC·Windows 예약명·대소문자 충돌을 거부하고, 각 파일의 Git blob ID·SHA-256·크기·mode를 `SOURCE-MANIFEST.json`에 기록한다. 외부 `ArchMeshRubbing-source.json`은 ZIP exact hash/size와 source commit/tree/file-set hash를 결합한다.

ZIP은 압축 library heuristic에 의존하지 않는 `ZIP_STORED`, commit epoch, 고정 metadata와 정렬된 경로를 사용한다. verifier는 Git이나 원래 저장소 없이도 보존된 raw commit object의 ID와 tree header를 검증하고, file record 전체에서 중첩 Git tree ID를 재구성한 뒤, GPL-2.0-only `LICENSE`, 모든 blob ID와 SHA-256을 다시 계산한다. release evidence와 portable manifest가 이 두 source 파일의 bytes도 함께 hash하므로 실행 payload와 함께 이동한 뒤에도 독립 검증할 수 있다. clean build가 아니면 frozen self-test는 이 archive를 정확한 corresponding source라고 인정하지 않는다.

```powershell
python tools/build_source_archive.py build `
  --repository . `
  --commit ((& git rev-parse HEAD).Trim()) `
  --archive dist\ArchMeshRubbing\source\ArchMeshRubbing-source.zip `
  --sidecar dist\ArchMeshRubbing\source\ArchMeshRubbing-source.json
python tools/build_source_archive.py verify `
  --archive dist\ArchMeshRubbing\source\ArchMeshRubbing-source.zip `
  --sidecar dist\ArchMeshRubbing\source\ArchMeshRubbing-source.json
```

### Unsigned build-provenance 계약

`<portable>.provenance.json`은 portable ZIP 바깥에 둔다. 그래야 provenance가 portable ZIP hash를 가리키면서 자기 자신을 다시 ZIP에 넣는 순환 참조를 만들지 않는다. 생성기는 먼저 portable ZIP·manifest를 전부 검증하고 실제 `dist/ArchMeshRubbing`의 모든 file path/size/SHA-256가 manifest와 같은지 비교한다. 이어 release evidence를 payload bytes에서 재생성 검증하고, source ZIP의 raw commit object와 재구성 tree/blob을 검증한 뒤 다음을 하나의 canonical JSON에 결합한다.

- portable ZIP·manifest exact SHA-256/size와 payload file-set hash
- source ZIP·sidecar exact SHA-256/size, source commit/tree와 source file-set hash
- release-evidence index SHA-256와 evidence payload hash
- GitHub repository·owner numeric ID, checkout SHA, workflow ref/SHA, event/ref/protection, run ID/attempt/URL, job ID
- GitHub-hosted Windows X64 runner environment·name

generator는 GitHub가 제공하는 [`GITHUB_*`와 `RUNNER_*` default environment variable](https://docs.github.com/en/actions/reference/workflows-and-actions/variables)만 읽고, repository/workflow/job/Windows X64 hosted-runner 경계를 벗어나면 실패한다. `run_id + run_attempt + job`은 실행 문맥을 특정하지만 공식 문서의 주의대로 `RUNNER_NAME` 자체는 고유하다고 가정하지 않는다.

이 파일은 의도적으로 `{"kind":"none","signature_present":false}`를 강제한다. 따라서 현재 verifier가 증명하는 것은 네 파일군의 상호 무결성과 기록된 실행 문맥의 형식·일관성이다. 누구든 파일 전체를 다시 쓸 수 있으므로 GitHub나 프로젝트가 그 기록을 인증했다는 의미는 아니다. workflow는 `id-token`, `attestations: write`, `actions/attest`, artifact upload를 사용하지 않는다. 공개 배포 전 별도 서명·신뢰 anchor 정책이 이 provenance와 portable/source checksum을 인증해야 한다.

```powershell
python tools/generate_build_provenance.py generate `
  --archive $env:AMR_PORTABLE_ARCHIVE `
  --manifest $env:AMR_PORTABLE_MANIFEST `
  --payload dist\ArchMeshRubbing `
  --output "$env:AMR_PORTABLE_ARCHIVE.provenance.json"
python tools/generate_build_provenance.py verify `
  --provenance "$env:AMR_PORTABLE_ARCHIVE.provenance.json" `
  --archive $env:AMR_PORTABLE_ARCHIVE `
  --manifest $env:AMR_PORTABLE_MANIFEST `
  --payload dist\ArchMeshRubbing
```

### Portable archive 계약

ZIP은 `ArchMeshRubbing/` 루트 하나 아래 regular file만 담는다. 생성기와 검증기는 다음을 강제한다.

- payload file을 POSIX 경로로 정렬하고 같은 payload·commit epoch·Python/zlib 환경에서 exact bytes 재생성
- 모든 timestamp를 commit epoch의 UTC 2초 단위 ZIP 시각으로 고정하고 DEFLATE level 9 사용
- 절대경로, `..`, 역슬래시, control character, 비 NFC 이름, 끝의 점/공백, Windows device name, 대소문자 충돌, symlink 거부
- ZIP directory entry, 암호화, extra/comment member metadata, 중복·누락·추가 파일 거부
- sidecar가 ZIP SHA-256/크기, payload 전체 entry SHA-256/크기, payload tree hash, release-evidence index hash와 source commit을 결합
- 검증 완료 전에 destination을 만들지 않고, 임시 sibling에 전수 재해시한 뒤 destination으로 원자적 publish
- 기존 ZIP, sidecar, destination을 덮어쓰지 않음

sidecar 자체는 RFC 독립적인 sorted compact UTF-8 JSON exact bytes이며, `portable_archive_manifest-1.0.0.schema.json`으로 외부 구현이 구조를 검증할 수 있다. archive hash를 archive 안에 넣는 자기참조를 피하기 위해 sidecar는 ZIP과 함께 배포하는 별도 파일이다.

### 공급망 evidence 형식

`release-evidence/`에는 다음 다섯 파일만 허용한다.

- `payload-manifest.json`: evidence 디렉터리만 제외한 모든 앱 payload의 정규화 경로·크기·SHA-256. portable payload에는 installer가 추가한 예외 파일이 없으므로 root 파일도 빠짐없이 검사하며 public-release policy path/hash/decision도 build descriptor에 기록한다.
- `sbom.spdx.json`: `filesAnalyzed=false`인 앱과 실제 포함 runtime distribution 10개, 각 Windows wheel SHA-256, `CONTAINS` 관계를 기록한 SPDX 2.3 JSON
- `third-party-notices.json`: wheel METADATA와 license evidence path/hash, public-release policy path/hash/decision의 machine-readable 결합
- `THIRD_PARTY_NOTICES.md`: 위 license evidence 원문 전체. PyOpenGL 3.1.10 wheel에 빠진 본문은 같은 버전의 PyPI sdist 경로와 archive/file SHA-256을 정책에 고정해 보완한다.
- `release-evidence.json`: 앞 네 문서의 path·size·SHA-256과 payload root hash를 묶는 index

evidence는 ZIP 생성 전, frozen 실행 전, 한글 경로 추출 뒤에 모두 실제 bytes에서 다시 계산한다. SBOM의 `licenseDeclared`는 wheel이 표준 `License-Expression`을 직접 제공할 때만 옮기고 legacy 자유 형식/분류자를 임의 SPDX 식으로 추정하지 않는다. 이 파일들은 검토 가능한 사실 기록이지 라이선스 호환성에 대한 법률 결론이 아니다.

형식 기준은 [pip secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/), [Python Core Metadata](https://packaging.python.org/en/latest/specifications/core-metadata/), [SPDX 2.3 document creation](https://spdx.github.io/spdx-spec/v2.3/document-creation-information/)과 [package information](https://spdx.github.io/spdx-spec/v2.3/package-information/)이다.

## 공개 배포 게이트

### 라이선스

저장소 source는 `Apache-2.0`이다. bundled `PyQt6 6.11.0` wheel 메타데이터는 `GPL-3.0-only`이므로, PyQt6를 함께 담아 배포하는 **결합물(frozen 실행 파일, portable ZIP)** 은 `GPL-3.0-only` 조건으로 전달된다. Apache-2.0은 GPLv3과 호환되므로(GPLv2와는 비호환) 이 결합은 허용되며, source를 따로 받는 쪽은 Apache-2.0 조건만 지키면 된다. 즉 `src/core`의 포맷·전개·탁본·검증 코드는 Qt 없이 어디서든 재사용할 수 있고, 특허 허여와 함께 제공된다. GUI 바이너리만 GPL-3.0으로 전달된다.

- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [GNU GPL version 3](https://www.gnu.org/licenses/gpl-3.0.html)
- [GNU GPL FAQ: Apache 2.0과 GPLv3 호환성](https://www.gnu.org/licenses/gpl-faq.en.html#apache2)
- [PyPI PyQt6 6.11.0 metadata](https://pypi.org/project/PyQt6/6.11.0/)
- [Riverbank PyQt licensing](https://www.riverbankcomputing.com/software/pyqt/intro/)

`requirements/public-release-policy.json`은 이 상태를 canonical JSON으로 기록한다. schema 1.1.0은 다음을 함께 강제한다.

- 프로젝트 표현식 `Apache-2.0`과 현재 `LICENSE` exact SHA-256
- 결합물 표현식 `GPL-3.0-only`와 `third_party_licenses/GPL-3.0.txt` exact SHA-256
- `PyQt6==6.11.0`의 실제 wheel `License-Expression=GPL-3.0-only`
- `rights_holder_authorization`이 `not-recorded`가 아닐 것, `decision=allowed`
- 검토 날짜와 위 공식 출처 URL

`decision=allowed`는 선언만으로 통과하지 않는다. loader는 프로젝트 라이선스와 관측된 모든 runtime 라이선스에서 결합물 표현식을 **직접 계산**해(`derive_combined_work_expression`) 문서가 선언한 값과 일치할 때만 정책을 받아들이고, 표에 없는 표현식이나 `GPL-2.0-only` + GPL-3.0 같은 비호환 조합은 거부한다. 따라서 JSON 한 단어를 바꿔 배포를 열거나 닫을 수 없다. 이 파일은 PyInstaller payload에 포함되고 payload manifest, SPDX comment, machine/human NOTICE, portable manifest와 provenance hash chain에 들어간다. frozen self-test와 Windows package workflow도 실제 payload bytes에서 다시 검증한다.

```bat
python tools/check_public_release_policy.py status
python tools/check_public_release_policy.py assert-allowed
python tools/check_public_release_policy.py require-public
```

세 명령 모두 현재 정책과 설치된 runtime이 일치하면 성공한다. `assert-blocked`는 정책이 다시 차단 상태로 돌아갔을 때만 성공하므로 회귀 확인용으로 남겨 둔다. 이는 법률 자문이 아니라 재현 가능한 사실 기록이다.

과거 CI가 사용한 Inno Setup 6.7.1의 `Non-commercial use only` compiler 경로는 제거했다. portable ZIP은 Python 표준 라이브러리만 사용하므로 installer compiler 구매·계정·서버가 빌드 전제에 남지 않는다. 과거 installer run은 역사적 내부 검증일 뿐 현재 배포 계약이 아니다.

### 배포 신뢰

공개 릴리스에는 추가로 아래가 필요하다.

- Windows Authenticode 또는 명시적으로 검토한 대체 서명/검증 정책
- 생성된 SPDX/NOTICE의 라이선스 호환성·고지 내용에 대한 최종 사람 검토
- 대표 Windows 하드웨어 GPU/driver와 compositor pilot
- large mesh, low-memory, 완전 격리된 offline machine pilot
- unsigned provenance와 공개 ZIP·sidecar·source archive를 인증하는 checksum/signature·trust-anchor 게시 규칙

hash-locked frozen payload, 검증형 portable ZIP, exact corresponding-source archive, source/evidence/runner를 묶는 unsigned provenance, 한글 경로 offline/actual-frame/삭제 gate의 코드 경계는 구현됐다. 그래도 프로젝트/GUI 라이선스 결론, provenance·배포물의 서명과 신뢰 anchor, 지원 대상 Windows 10/11 x64의 대표 하드웨어·실물 대용량 pilot과 공개 릴리스는 아직 차단한다. 비 Windows 배포는 현재 비목표다.
