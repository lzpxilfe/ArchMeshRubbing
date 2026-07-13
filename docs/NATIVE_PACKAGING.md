# Native Packaging and Release Gates

이 문서는 로컬 unsigned 앱과 CI smoke artifact를 만드는 절차를 설명한다. 현재 절차는 설치, 서명, notarization, 업로드 또는 공개 배포를 수행하지 않는다.

## 지원 빌드 기준

- CPython 3.12
- `requirements/runtime-py312.lock`: 정확한 런타임 버전
- `requirements/build-py312.lock`: 정확한 PyInstaller toolchain 버전
- `ArchMeshRubbing.spec`: 첫 안정판 대상인 Windows onedir build. 기존 Linux onedir와 macOS `.app` 분기는 source 호환용으로 남지만 현재 릴리스 게이트가 아니다.
- `build/generated/build_info.json`: version, channel, commit, runtime lock SHA-256

lock은 버전을 고정하지만 아직 OS별 wheel SHA-256까지 고정한 공급망 lock은 아니다. 공개 릴리스 전에는 wheelhouse/hash lock 또는 동등한 provenance가 추가로 필요하다.

## 안전한 로컬 빌드

깨끗한 Windows Python 3.12 환경에서 실행한다.

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements/build-py312.lock
python tools/build_native.py
```

기존 `build/ArchMeshRubbing` 또는 `dist/ArchMeshRubbing*`가 있으면 명령은 기본적으로 중단한다. 검토한 생성물만 명시적으로 교체할 때 `--replace-existing`를 사용한다. PyInstaller cache 삭제도 `--clean-cache`를 지정해야만 수행한다.

`build_and_shortcut.py`는 과거 호환 wrapper다. 더 이상 폴더를 자동 삭제하거나 Windows 바탕화면 바로가기를 만들지 않는다.

## Frozen self-test

빌드 도구는 실제 사용자 실행 파일에 `--self-test-report`를 전달하고 결과 JSON의 `ok=true`를 확인한다.

현재 self-test는 다음을 검사한다.

1. embedded build manifest와 runtime lock hash
2. 정확한 runtime distribution 버전과 Shapely/GEOS 조합
3. 아이콘, lock, 10개 JSON schema
4. offscreen Qt application
5. 실제 `MainWindow`, `QOpenGLWidget`, OpenGL.GL/GLU import와 생성
6. OBJ, PLY, STL, OFF, glTF, GLB를 closed import recipe와 외부 dependency deny resolver로 여는 production parser 경로
7. Pillow PNG encode/decode
8. canonical `ArtifactDocument` round-trip golden
9. canonical Cutline golden
10. canonical Digital Rubbing golden
11. 실제 PLY → 단위/Align session → embedded `.amr` 저장 → 외부 원본 삭제 → source/geometry/Align/world vertex 재검증
12. 실제 application authority의 Open → explicit Align → Cutline 3/3 → Outline 6/6 → Digital Rubbing 6/6 → completed `.amr` offline reopen → 이동된 1:1 SVG/PNG의 원본 SHA-256·recipe·QC 재검증

Offscreen에서 `QOpenGLWidget`을 생성하는 검사는 module/plugin 누락을 잡지만 실제 context와 frame 정확성을 증명하지 않는다. Windows CI는 이어서 `QT_QPA_PLATFORM=windows`, `QT_OPENGL=software`로 native `qwindows`와 bundled `opengl32sw.dll`을 사용해 `src.gui.opengl_driver_smoke`를 실행한다. frozen 실행 파일도 `--opengl-driver-smoke-report PATH`로 같은 source module을 실행한다. report는 실제 context/FBO/VBO/pixel/depth/pick과 두 투영 모드를 모두 검사한다.

앱과 driver smoke는 QApplication 생성 전에 같은 OpenGL 2.1 compatibility·24-bit depth surface format을 요청한다. macOS 로컬 실행 예시는 다음과 같다. report는 기존 파일을 덮어쓰지 않으므로 존재하지 않는 새 경로를 사용한다.

```bash
python -m src.gui.opengl_driver_smoke \
  --qt-platform cocoa \
  --report build/opengl-driver-smoke.json
```

2026-07-12 code commit `f25b424d6936e6e8832a81c7a6683cb58515e546`의 clean source tree, Python 3.12.13/macOS arm64 Apple M4에서 `opengl-driver-smoke-f25b424-darwin-arm64.json`의 61개 실제 GL 조건이 통과했다. 이어 clean source commit `166103dcf0ea`의 [GitHub Actions run 29182584810](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29182584810)에서도 Ubuntu 24.04 xcb + Mesa llvmpipe 61/61이 통과했다. report는 commit/tree 상태·runtime lock·dependency version·UTC 시각을 포함한다. 두 결과 모두 source checkout 증거이며 frozen executable actual-GL 검증은 아니다.

2026-07-12 로컬 증거 `native-self-test-local-smoke-898a8bfc144f-darwin.json`은 code commit `898a8bfc144f`의 clean source tree, Python 3.12.13, macOS arm64에서 10/10 통과했다. 이 unsigned build는 Windows, Linux, Intel Mac, universal2 또는 공개 배포 준비를 증명하지 않는다.

## Windows CI

`.github/workflows/package-smoke.yml`은 `main` push, pull request, 수동 실행에서 Windows 한 환경에 대해 다음을 수행한다. `main` push는 패키지 입력 파일이 바뀐 경우에만 실행한다.

- Python 3.12 설정
- exact build lock 설치
- commit/lock에 결합된 manifest 생성
- PyInstaller build
- Windows frozen executable의 file-based 12-check self-test
- report의 전체 check 성공 확인
- source와 frozen executable 각각의 Windows native-QPA software OpenGL report 및 전체 check 성공 확인

`package-smoke.yml`에는 frozen executable artifact upload나 release 단계가 없다. 과거 commit `e4bf6dcac4b1`의 [run `29213279508`](https://github.com/lzpxilfe/ArchMeshRubbing/actions/runs/29213279508)에서 세 OS가 통과했지만, 현재 제품 판정은 Windows job 하나만 사용한다. 이는 unsigned CI smoke 증거이며 공개 설치 파일을 제공한다는 뜻은 아니다.

`package-smoke.yml`의 12-check self-test는 offscreen이지만 바로 뒤의 별도 frozen driver smoke는 native `qwindows`에서 실제 OpenGL context와 framebuffer를 사용한다. 다만 Qt의 Mesa software DLL을 강제하므로 대표 Windows 하드웨어 GPU/driver 또는 compositor 최종 표시 인증으로 표현해서는 안 된다.

## 공개 배포 차단 게이트

### 라이선스

현재 저장소 파일은 `GPLv2`로 표기되고 `or later`를 명시하지 않는다. bundled `PyQt6 6.11.0` 메타데이터는 `GPL-3.0-only`다. GNU는 GPLv2-only와 GPLv3의 결합이 호환되지 않는다고 설명하고, Riverbank는 무료 PyQt가 GPLv3이라고 명시한다.

- [GNU GPL compatibility FAQ](https://www.gnu.org/licenses/gpl-faq.html.en#v2v3Compatibility)
- [Riverbank PyQt licensing](https://www.riverbankcomputing.com/software/pyqt/intro/)
- [Qt open-source licensing](https://doc.qt.io/qt-6/licensing.html)

따라서 모든 권리자의 동의에 따른 재허가, 적절한 상용 라이선스와 추가 허가, 또는 GUI 경계 교체 같은 전략을 결정하기 전에는 PyQt6 포함 바이너리를 공개하지 않는다. 이는 법률 자문이 아니라 보수적인 릴리스 게이트다.

### 배포 신뢰

공개 릴리스에는 추가로 아래가 필요하다.

- Windows Authenticode
- 실제 포함 파일 기준 SBOM과 제3자 NOTICE/license bundle
- Windows 설치/제거/파일 연결 smoke
- 대표 Windows 하드웨어 GPU/driver와 compositor pilot
- large mesh, low-memory, non-ASCII path, offline machine pilot
- clean source tree 또는 source archive digest를 포함하는 build provenance
- 공개 산출물 하나를 선택하는 패키징 규칙과 checksum/signature

macOS·Linux 배포, signing과 installer는 첫 Windows 안정판 이후 별도 범위로 둔다.
