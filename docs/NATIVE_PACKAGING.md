# Native Packaging and Release Gates

이 문서는 로컬 unsigned 앱과 CI 전용 unsigned Windows installer를 만드는 절차를 설명한다. CI는 installer를 격리 설치·검증·제거하지만 서명, artifact 업로드 또는 공개 배포는 수행하지 않는다.

## 지원 빌드 기준

- CPython 3.12
- `requirements/runtime-py312.lock`: 정확한 런타임 버전
- `requirements/build-py312.lock`: 정확한 PyInstaller toolchain 버전
- `requirements/windows-py312-x64-hashed.lock`: Windows x64/CPython 3.12의 런타임·빌드 wheel 17개와 각 SHA-256. `--require-hashes --only-binary=:all:`로 설치한다.
- `requirements/runtime-license-policy.json`, `third_party_licenses/`: wheel에 라이선스 원문이 없는 예외의 검토된 출처·원문 SHA-256
- `ArchMeshRubbing.spec`: 첫 안정판 대상인 Windows onedir build. 기존 Linux onedir와 macOS `.app` 분기는 source 호환용으로 남지만 현재 릴리스 게이트가 아니다.
- `installer/ArchMeshRubbing.iss`: stable AppId를 가진 per-user Windows installer 정의. CI에서만 unsigned 검증 산출물을 만들며 파일 연결·자동 실행·바탕화면 바로가기를 추가하지 않는다.
- `build/generated/build_info.json`: version, channel, commit, runtime lock과 Windows wheel lock SHA-256

일반 source 실행은 플랫폼 중립 version lock을 사용한다. 제품 대상 Windows frozen job은 별도의 평탄화된 hash lock만 설치하며 sdist와 해시가 검토되지 않은 wheel을 거부한다. 현재 lock은 Windows x64/CPython 3.12 한 대상의 증거이며 다른 OS 지원을 뜻하지 않는다.

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

`build_and_shortcut.py`는 과거 호환 wrapper다. 더 이상 폴더를 자동 삭제하거나 Windows 바탕화면 바로가기를 만들지 않는다.

## Frozen self-test

빌드 도구는 실제 사용자 실행 파일에 `--self-test-report`를 전달하고 결과 JSON의 `ok=true`를 확인한다.

현재 self-test는 다음을 검사한다.

1. embedded build manifest와 runtime/wheel lock hash
2. 정확한 runtime distribution 버전과 Shapely/GEOS 조합
3. 아이콘, runtime/wheel lock, license policy, 10개 JSON schema
4. frozen/설치 payload의 전체 파일 SHA-256 manifest, SPDX 2.3 SBOM, 제3자 NOTICE를 실제 bytes에서 재계산
5. offscreen Qt application
6. 실제 `MainWindow`, `QOpenGLWidget`, OpenGL.GL/GLU import와 생성
7. OBJ, PLY, STL, OFF, glTF, GLB를 closed import recipe와 외부 dependency deny resolver로 여는 production parser 경로
8. Pillow PNG encode/decode
9. canonical `ArtifactDocument` round-trip golden
10. canonical Cutline golden
11. canonical Digital Rubbing golden
12. 실제 PLY → 단위/Align session → embedded `.amr` 저장 → 외부 원본 삭제 → source/geometry/Align/world vertex 재검증
13. 실제 application authority의 Open → explicit Align → Cutline 3/3 → Outline 6/6 → Digital Rubbing 6/6 → completed `.amr` offline reopen → 이동된 1:1 SVG/PNG의 원본 SHA-256·recipe·QC 재검증

Offscreen에서 `QOpenGLWidget`을 생성하는 검사는 module/plugin 누락을 잡지만 실제 context와 frame 정확성을 증명하지 않는다. Windows CI는 이어서 `QT_QPA_PLATFORM=windows`, `QT_OPENGL=software`로 native `qwindows`와 bundled `opengl32sw.dll`을 사용해 `src.gui.opengl_driver_smoke`를 실행한다. PyOpenGL의 GL/WGL dispatch도 같은 DLL에 결합해 Qt software context와 시스템 `opengl32.dll`이 섞이지 않게 한다. frozen 실행 파일도 `--opengl-driver-smoke-report PATH`로 같은 source module을 실행한다. report는 768×768 실제 widget FBO, VBO, pixel/depth/pick과 두 투영 모드를 모두 검사한다.

앱과 driver smoke는 QApplication 생성 전에 같은 OpenGL 2.1 compatibility·24-bit depth surface format을 요청한다. Windows source smoke 예시는 다음과 같다. report는 기존 파일을 덮어쓰지 않으므로 존재하지 않는 새 경로를 사용한다.

```bat
set QT_QPA_PLATFORM=windows
set QT_OPENGL=software
python -m src.gui.opengl_driver_smoke ^
  --qt-platform windows ^
  --report build/opengl-driver-smoke.json
```

2026-07-13 Windows 대상 commit `b12d4874a4a8`의 [source CI run 29251668123](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29251668123)은 qwindows+llvmpipe actual-frame 66/66을, [frozen package run 29251668029](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29251668029)은 12-check offline self-test와 frozen actual-frame gate를 통과했다. report는 commit/tree 상태·runtime lock·dependency version·UTC 시각을 포함한다. 과거 macOS/Linux source 결과는 이식성 기록일 뿐 첫 안정판 지원 판정에 사용하지 않는다.

## Windows CI

`.github/workflows/package-smoke.yml`은 `main` push, pull request, 수동 실행에서 Windows 한 환경에 대해 다음을 수행한다. `main` push는 패키지 입력 파일이 바뀐 경우에만 실행한다.

- Python 3.12 설정
- hash-locked Windows wheel 17개를 binary-only로 설치하고 dependency closure 검사
- commit/runtime lock/wheel lock에 결합된 manifest 생성
- PyInstaller build
- frozen payload 전체 파일 manifest, SPDX 2.3 JSON SBOM, machine-readable notice와 전체 원문 `THIRD_PARTY_NOTICES.md` 생성·재검증
- Windows frozen executable의 file-based 13-check self-test
- report의 전체 check 성공 확인
- source와 frozen executable 각각의 Windows native-QPA software OpenGL report 및 전체 check 성공 확인
- Inno Setup 6 compiler version·SHA-256, source commit과 installer SHA-256·크기·unsigned 상태 기록
- 격리된 per-user 경로에 무인 설치하고 frozen onedir의 모든 payload를 byte-for-byte SHA-256 비교
- 설치된 실행 파일에 outbound deny firewall rule과 실패 proxy를 적용한 13-check complete workflow와 payload evidence 재검증
- 설치본 native `qwindows` software OpenGL actual-frame 검사, 무인 제거, 설치 디렉터리·Start Menu shortcut 잔여물 검사

`package-smoke.yml`에는 frozen executable·installer artifact upload나 release 단계가 없다. 현재 제품 판정은 Windows job 하나만 사용한다. commit `19558f324deb`의 [run `29255341573`](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29255341573)은 Inno Setup 6.7.1로 만든 62,312,936-byte unsigned installer의 SHA-256과 compiler SHA-256을 기록하고, 465개 payload 전수 비교, outbound 차단 12-check workflow, 설치본 actual-frame 66/66, 제거를 통과했다. 이는 내부 패키지 역학의 증거이며 공개 설치 파일을 제공한다는 뜻은 아니다. 과거 세 OS 결과는 역사적 이식성 기록으로만 남긴다.

격리된 ASCII per-user 경로는 위 전체 gate를 통과했다. 별도 [비 ASCII 경로 진단 run `29254942224`](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29254942224)은 설치·payload 비교·outbound 차단·12-check offline workflow까지 통과했지만 bundled software OpenGL context 생성에 실패했다. 따라서 비 ASCII 설치 경로의 native rendering은 아직 지원 완료로 판정하지 않고 실제 Windows pilot 항목으로 남긴다.

`package-smoke.yml`의 13-check self-test는 offscreen이지만 바로 뒤의 별도 frozen driver smoke는 native `qwindows`에서 실제 OpenGL context와 framebuffer를 사용한다. 다만 Qt의 Mesa software DLL을 강제하므로 대표 Windows 하드웨어 GPU/driver 또는 compositor 최종 표시 인증으로 표현해서는 안 된다.

### 공급망 evidence 형식

`release-evidence/`는 설치 파일에도 그대로 들어가며 다음 다섯 파일만 허용한다.

- `payload-manifest.json`: evidence 디렉터리와 Inno Setup이 설치 시 만드는 root `uninsNNN.*`만 제외한 모든 앱 payload의 정규화 경로·크기·SHA-256
- `sbom.spdx.json`: `filesAnalyzed=false`인 앱과 실제 포함 runtime distribution 10개, 각 Windows wheel SHA-256, `CONTAINS` 관계를 기록한 SPDX 2.3 JSON
- `third-party-notices.json`: wheel METADATA와 license evidence path/hash의 machine-readable 결합
- `THIRD_PARTY_NOTICES.md`: 위 license evidence 원문 전체. PyOpenGL 3.1.10 wheel에 빠진 본문은 같은 버전의 PyPI sdist 경로와 archive/file SHA-256을 정책에 고정해 보완한다.
- `release-evidence.json`: 앞 네 문서의 path·size·SHA-256과 payload root hash를 묶는 index

생성 직후, frozen 실행 전, 설치 뒤에 모두 실제 bytes에서 다시 계산한다. Inno Setup uninstaller는 pre-installer PyInstaller payload가 아니므로 앱 payload manifest 범위 밖이며, 현재 CI의 별도 설치 파일 전수 비교와 향후 Authenticode가 그 경계를 맡는다. SBOM의 `licenseDeclared`는 wheel이 표준 `License-Expression`을 직접 제공할 때만 옮기고, legacy 자유 형식/분류자를 임의 SPDX 식으로 추정하지 않는다. 이 파일들은 검토 가능한 사실 기록이지 라이선스 호환성에 대한 법률 결론이 아니다.

형식과 설치 정책의 기준은 [pip secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/), [Python Core Metadata](https://packaging.python.org/en/latest/specifications/core-metadata/), [SPDX 2.3 document creation](https://spdx.github.io/spdx-spec/v2.3/document-creation-information/)과 [package information](https://spdx.github.io/spdx-spec/v2.3/package-information/)이다.

## 공개 배포 차단 게이트

### 라이선스

현재 저장소 파일은 `GPLv2`로 표기되고 `or later`를 명시하지 않는다. bundled `PyQt6 6.11.0` 메타데이터는 `GPL-3.0-only`다. GNU는 GPLv2-only와 GPLv3의 결합이 호환되지 않는다고 설명하고, Riverbank는 무료 PyQt가 GPLv3이라고 명시한다.

- [GNU GPL compatibility FAQ](https://www.gnu.org/licenses/gpl-faq.html.en#v2v3Compatibility)
- [Riverbank PyQt licensing](https://www.riverbankcomputing.com/software/pyqt/intro/)
- [Qt open-source licensing](https://doc.qt.io/qt-6/licensing.html)

따라서 모든 권리자의 동의에 따른 재허가, 적절한 상용 라이선스와 추가 허가, 또는 GUI 경계 교체 같은 전략을 결정하기 전에는 PyQt6 포함 바이너리를 공개하지 않는다. 이는 법률 자문이 아니라 보수적인 릴리스 게이트다.

CI runner의 Inno Setup 6.7.1 compiler는 `Non-commercial use only`를 출력한다. 공식 안내도 상업적 맥락에서 이 도구로 이익을 얻거나 CI에서 compiler를 호출한다면 commercial license 구매를 기대한다고 명시한다. 현재 CI 산출물은 업로드하지 않는다. 향후 공개·상업 배포 전에는 사용 맥락을 확인해 Inno license를 확보하거나 호환되는 installer toolchain으로 교체한다.

- [Inno Setup 6.7 revision history](https://jrsoftware.org/files/is6-whatsnew.htm)
- [Inno Setup commercial license policy](https://jrsoftware.org/isorder.php)

### 배포 신뢰

공개 릴리스에는 추가로 아래가 필요하다.

- Windows Authenticode
- 생성된 SPDX/NOTICE의 라이선스 호환성·고지 내용에 대한 최종 사람 검토
- 파일 연결을 추가할 경우 해당 Windows 설치/제거 smoke
- 대표 Windows 하드웨어 GPU/driver와 compositor pilot
- large mesh, low-memory, non-ASCII path, offline machine pilot
- clean source tree 또는 source archive digest를 포함하는 build provenance
- 공개 산출물 하나를 선택하는 패키징 규칙과 checksum/signature

Windows installer의 내부 설치·실행·제거 역학과 hash lock·payload manifest·SBOM/NOTICE 생성 경계는 구현됐다. 그래도 라이선스 결론, 서명, source archive/runner까지 포함한 상위 provenance, 대표 Windows pilot과 공개 릴리스는 아직 차단한다. macOS·Linux 배포는 첫 Windows 안정판 이후 별도 범위로 둔다.
