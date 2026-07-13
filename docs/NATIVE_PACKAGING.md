# Native Packaging and Release Gates

이 문서는 Windows frozen 앱과 검증형 portable ZIP을 만드는 절차를 설명한다. 첫 안정판의 패키지 경로는 설치 프로그램 compiler, 계정, 라이선스 서버를 요구하지 않는다. CI는 ZIP을 한글 경로에 검증·추출해 실제 실행한 뒤 삭제하지만, 서명, artifact 업로드 또는 공개 배포는 아직 수행하지 않는다.

## 지원 빌드 기준

- CPython 3.12
- `requirements/runtime-py312.lock`: 정확한 런타임 버전
- `requirements/build-py312.lock`: 정확한 PyInstaller toolchain 버전
- `requirements/windows-py312-x64-hashed.lock`: Windows x64/CPython 3.12 런타임·빌드 wheel 17개와 각 SHA-256. `--require-hashes --only-binary=:all:`로 설치한다.
- `requirements/runtime-license-policy.json`, `third_party_licenses/`: wheel에 라이선스 원문이 없는 예외의 검토된 출처·원문 SHA-256
- `ArchMeshRubbing.spec`: 첫 안정판 대상인 Windows onedir build. Linux onedir와 macOS `.app` 분기는 source 호환용이며 현재 릴리스 게이트가 아니다.
- `src/portable_archive.py`, `tools/build_portable_archive.py`: ZIP 생성, sidecar 검증, 안전한 원자적 추출
- `schemas/portable_archive_manifest-1.0.0.schema.json`: portable sidecar의 machine-readable 계약
- `build/generated/build_info.json`: version, channel, commit, runtime lock과 Windows wheel lock SHA-256

일반 source 실행은 플랫폼 중립 version lock을 사용한다. 제품 대상 Windows frozen job은 평탄화된 hash lock만 설치하며 sdist와 검토되지 않은 wheel을 거부한다. 현재 lock은 Windows x64/CPython 3.12 한 대상의 증거이며 다른 OS 지원을 뜻하지 않는다.

## 안전한 로컬 빌드

깨끗한 Windows Python 3.12 환경에서 실행한다.

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --require-hashes --only-binary=:all: ^
  -r requirements/windows-py312-x64-hashed.lock
python tools/build_native.py
```

기존 `build/ArchMeshRubbing` 또는 `dist/ArchMeshRubbing*`가 있으면 명령은 기본적으로 중단한다. 검토한 생성물만 명시적으로 교체할 때 `--replace-existing`를 사용한다. PyInstaller cache 삭제도 `--clean-cache`를 지정해야만 수행한다.

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
3. 아이콘, runtime/wheel lock, license policy, 11개 JSON schema
4. frozen/portable payload의 전체 파일 SHA-256 manifest, SPDX 2.3 SBOM, 제3자 NOTICE를 실제 bytes에서 재계산
5. offscreen Qt application
6. 실제 `MainWindow`, `QOpenGLWidget`, OpenGL.GL/GLU import와 생성
7. OBJ, PLY, STL, OFF, glTF, GLB를 closed import recipe와 외부 dependency deny resolver로 여는 production parser 경로
8. Pillow PNG encode/decode
9. canonical `ArtifactDocument` round-trip golden
10. canonical Cutline golden
11. canonical Digital Rubbing golden
12. 실제 PLY → 단위/Align session → embedded `.amr` 저장 → 외부 원본 삭제 → source/geometry/Align/world vertex 재검증
13. 실제 application authority의 Open → explicit Align → Cutline 3/3 → Outline 6/6 → Digital Rubbing 6/6 → completed `.amr` offline reopen → 이동된 1:1 SVG/PNG의 원본 SHA-256·recipe·QC 재검증

Offscreen `QOpenGLWidget` 생성은 module/plugin 누락을 잡지만 실제 frame 정확성을 증명하지 않는다. Windows CI는 이어서 `QT_QPA_PLATFORM=windows`, `QT_OPENGL=software`로 native `qwindows`와 bundled `opengl32sw.dll`을 사용해 `src.gui.opengl_driver_smoke`를 실행한다. PyOpenGL의 GL/WGL dispatch도 같은 DLL에 결합해 Qt software context와 시스템 `opengl32.dll`이 섞이지 않게 한다. report는 768×768 실제 widget FBO, VBO, pixel/depth/pick과 두 투영 모드를 모두 검사한다.

```bat
set QT_QPA_PLATFORM=windows
set QT_OPENGL=software
python -m src.gui.opengl_driver_smoke ^
  --qt-platform windows ^
  --report build/opengl-driver-smoke.json
```

2026-07-13 Windows 대상 commit `b12d4874a4a8`의 [source CI run 29251668123](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29251668123)은 qwindows+llvmpipe actual-frame 66/66을, [frozen package run 29251668029](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29251668029)은 당시 complete-workflow self-test와 frozen actual-frame gate를 통과했다. 과거 macOS/Linux source 결과는 이식성 기록일 뿐 첫 안정판 지원 판정에 사용하지 않는다.

## Windows portable CI

`.github/workflows/package-smoke.yml`은 `main` push, pull request, 수동 실행에서 Windows 한 환경에 대해 다음을 요구한다.

- Python 3.12와 hash-locked Windows wheel 17개만 설치하고 dependency closure 검사
- commit/runtime lock/wheel lock에 결합된 manifest와 PyInstaller onedir 생성
- payload 전체 파일 manifest, SPDX 2.3 JSON SBOM, machine/human NOTICE 생성·재검증
- Windows frozen executable의 file-based 13-check와 native-QPA software OpenGL 검사
- commit epoch를 고정 timestamp로 사용해 portable ZIP과 canonical sidecar 생성
- ZIP exact hash/size, 모든 entry path/size/SHA-256, release-evidence index와 source commit 재검증
- `문화유산 기록\ArchMeshRubbing` 한글 경로에 검증 후 원자적으로 추출
- 추출본 release evidence를 다시 계산하고 executable에 outbound deny firewall rule과 실패 proxy 적용
- 한글 경로의 실행 파일과 report 경로에서 13-check complete workflow와 native `qwindows` actual-frame 재실행
- 실행 뒤 ZIP을 다시 검증하고 추출 디렉터리와 firewall rule 제거

각 실행 파일 단계는 120초의 명시적 상한을 가진다. timeout을 성공으로 바꾸거나 GUI hang을 숨기지 않는다. workflow에는 ZIP·실행 파일 artifact upload나 release 단계가 없다. 라이선스 게이트가 해결되기 전에는 CI 밖으로 바이너리를 게시하지 않는다.

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

- `payload-manifest.json`: evidence 디렉터리만 제외한 모든 앱 payload의 정규화 경로·크기·SHA-256. portable payload에는 installer가 추가한 예외 파일이 없으므로 root 파일도 빠짐없이 검사한다.
- `sbom.spdx.json`: `filesAnalyzed=false`인 앱과 실제 포함 runtime distribution 10개, 각 Windows wheel SHA-256, `CONTAINS` 관계를 기록한 SPDX 2.3 JSON
- `third-party-notices.json`: wheel METADATA와 license evidence path/hash의 machine-readable 결합
- `THIRD_PARTY_NOTICES.md`: 위 license evidence 원문 전체. PyOpenGL 3.1.10 wheel에 빠진 본문은 같은 버전의 PyPI sdist 경로와 archive/file SHA-256을 정책에 고정해 보완한다.
- `release-evidence.json`: 앞 네 문서의 path·size·SHA-256과 payload root hash를 묶는 index

evidence는 ZIP 생성 전, frozen 실행 전, 한글 경로 추출 뒤에 모두 실제 bytes에서 다시 계산한다. SBOM의 `licenseDeclared`는 wheel이 표준 `License-Expression`을 직접 제공할 때만 옮기고 legacy 자유 형식/분류자를 임의 SPDX 식으로 추정하지 않는다. 이 파일들은 검토 가능한 사실 기록이지 라이선스 호환성에 대한 법률 결론이 아니다.

형식 기준은 [pip secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/), [Python Core Metadata](https://packaging.python.org/en/latest/specifications/core-metadata/), [SPDX 2.3 document creation](https://spdx.github.io/spdx-spec/v2.3/document-creation-information/)과 [package information](https://spdx.github.io/spdx-spec/v2.3/package-information/)이다.

## 공개 배포 차단 게이트

### 라이선스

현재 저장소 파일은 `GPLv2`로 표기되고 `or later`를 명시하지 않는다. bundled `PyQt6 6.11.0` 메타데이터는 `GPL-3.0-only`다. GNU는 GPLv2-only와 GPLv3의 결합이 호환되지 않는다고 설명하고 Riverbank는 무료 PyQt가 GPLv3이라고 명시한다.

- [GNU GPL compatibility FAQ](https://www.gnu.org/licenses/gpl-faq.html.en#v2v3Compatibility)
- [Riverbank PyQt licensing](https://www.riverbankcomputing.com/software/pyqt/intro/)
- [Qt open-source licensing](https://doc.qt.io/qt-6/licensing.html)

따라서 모든 권리자의 동의에 따른 재허가, 적절한 상용 라이선스와 추가 허가, 또는 GUI 경계 교체 같은 전략을 결정하기 전에는 PyQt6 포함 바이너리를 공개하지 않는다. 이는 법률 자문이 아니라 보수적인 릴리스 게이트다.

과거 CI가 사용한 Inno Setup 6.7.1의 `Non-commercial use only` compiler 경로는 제거했다. portable ZIP은 Python 표준 라이브러리만 사용하므로 installer compiler 구매·계정·서버가 빌드 전제에 남지 않는다. 과거 installer run은 역사적 내부 검증일 뿐 현재 배포 계약이 아니다.

### 배포 신뢰

공개 릴리스에는 추가로 아래가 필요하다.

- Windows Authenticode 또는 명시적으로 검토한 대체 서명/검증 정책
- 생성된 SPDX/NOTICE의 라이선스 호환성·고지 내용에 대한 최종 사람 검토
- 대표 Windows 하드웨어 GPU/driver와 compositor pilot
- large mesh, low-memory, 완전 격리된 offline machine pilot
- clean source archive digest와 runner identity까지 포함하는 상위 build provenance
- 공개 ZIP·sidecar·source archive를 묶는 checksum/signature 게시 규칙

hash-locked frozen payload, 검증형 portable ZIP, 한글 경로 offline/actual-frame/삭제 gate의 코드 경계는 구현됐다. 그래도 프로젝트/GUI 라이선스 결론, 서명, source archive·runner provenance, 대표 하드웨어·실물 대용량 pilot과 공개 릴리스는 아직 차단한다. macOS·Linux 배포는 첫 Windows 안정판 이후 별도 범위다.
