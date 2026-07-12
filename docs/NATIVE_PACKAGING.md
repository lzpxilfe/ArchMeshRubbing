# Native Packaging and Release Gates

이 문서는 로컬 unsigned 앱과 CI smoke artifact를 만드는 절차를 설명한다. 현재 절차는 설치, 서명, notarization, 업로드 또는 공개 배포를 수행하지 않는다.

## 지원 빌드 기준

- CPython 3.12
- `requirements/runtime-py312.lock`: 정확한 런타임 버전
- `requirements/build-py312.lock`: 정확한 PyInstaller toolchain 버전
- `ArchMeshRubbing.spec`: Windows·Linux onedir와 macOS onedir/`.app`의 단일 spec
- `build/generated/build_info.json`: version, channel, commit, runtime lock SHA-256

lock은 버전을 고정하지만 아직 OS별 wheel SHA-256까지 고정한 공급망 lock은 아니다. 공개 릴리스 전에는 wheelhouse/hash lock 또는 동등한 provenance가 추가로 필요하다.

## 안전한 로컬 빌드

깨끗한 Python 3.12 환경에서 실행한다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/build-py312.lock
python tools/build_native.py
```

Windows에서는 활성화 명령만 `.venv\Scripts\activate`로 바꾼다. 기존 `build/ArchMeshRubbing` 또는 `dist/ArchMeshRubbing*`가 있으면 명령은 기본적으로 중단한다. 검토한 생성물만 명시적으로 교체할 때 `--replace-existing`를 사용한다. PyInstaller cache 삭제도 `--clean-cache`를 지정해야만 수행한다.

`build_and_shortcut.py`는 과거 호환 wrapper다. 더 이상 폴더를 자동 삭제하거나 Windows 바탕화면 바로가기를 만들지 않는다.

## Frozen self-test

빌드 도구는 실제 사용자 실행 파일에 `--self-test-report`를 전달하고 결과 JSON의 `ok=true`를 확인한다.

현재 self-test는 다음을 검사한다.

1. embedded build manifest와 runtime lock hash
2. 정확한 runtime distribution 버전과 Shapely/GEOS 조합
3. 아이콘, lock, 5개 JSON schema
4. offscreen Qt application
5. 실제 `MainWindow`, `QOpenGLWidget`, OpenGL.GL/GLU import와 생성
6. OBJ, PLY, STL, OFF, glTF, GLB in-memory parser
7. Pillow PNG encode/decode
8. canonical `ArtifactDocument` round-trip golden
9. canonical Cutline golden
10. canonical Digital Rubbing golden

Offscreen에서 `QOpenGLWidget`을 생성하는 검사는 module/plugin 누락을 잡지만 실제 GPU context와 frame 정확성을 증명하지 않는다. Xvfb+Mesa 또는 실제 데스크톱 세션에서의 GL context/render smoke가 별도 릴리스 게이트다.

2026-07-12 로컬 증거 `native-self-test-local-smoke-429abafebeb6-darwin-5.json`은 Python 3.12.13, macOS arm64에서 10/10 통과했다. 이 unsigned build manifest는 `source_tree=dirty`를 명시하며 Windows, Linux, Intel Mac, universal2 또는 공개 배포 준비를 증명하지 않는다.

## 3-OS CI

`.github/workflows/package-smoke.yml`은 pull request와 수동 실행에서 Ubuntu, Windows, macOS 각각 다음을 수행하도록 구성한다.

- Python 3.12 설정
- exact build lock 설치
- commit/lock에 결합된 manifest 생성
- PyInstaller build
- OS별 frozen executable의 file-based self-test
- report의 전체 check 성공 확인

워크플로우에는 artifact upload나 release 단계가 없다. 원격 workflow가 실제로 성공하기 전에는 3개 OS가 검증됐다고 표현하지 않는다.

## 공개 배포 차단 게이트

### 라이선스

현재 저장소 파일은 `GPLv2`로 표기되고 `or later`를 명시하지 않는다. bundled `PyQt6 6.11.0` 메타데이터는 `GPL-3.0-only`다. GNU는 GPLv2-only와 GPLv3의 결합이 호환되지 않는다고 설명하고, Riverbank는 무료 PyQt가 GPLv3이라고 명시한다.

- [GNU GPL compatibility FAQ](https://www.gnu.org/licenses/gpl-faq.html.en#v2v3Compatibility)
- [Riverbank PyQt licensing](https://www.riverbankcomputing.com/software/pyqt/intro/)
- [Qt open-source licensing](https://doc.qt.io/qt-6/licensing.html)

따라서 모든 권리자의 동의에 따른 재허가, 적절한 상용 라이선스와 추가 허가, 또는 GUI 경계 교체 같은 전략을 결정하기 전에는 PyQt6 포함 바이너리를 공개하지 않는다. 이는 법률 자문이 아니라 보수적인 릴리스 게이트다.

### 배포 신뢰

공개 릴리스에는 추가로 아래가 필요하다.

- Windows Authenticode와 macOS Developer ID/notarization
- 실제 포함 파일 기준 SBOM과 제3자 NOTICE/license bundle
- OS·architecture별 설치/제거/파일 연결 smoke
- 실제 OpenGL context와 대표 GPU/driver smoke
- large mesh, low-memory, non-ASCII path, offline machine pilot
- clean source tree 또는 source archive digest를 포함하는 build provenance
- 공개 산출물 하나를 선택하는 패키징 규칙과 checksum/signature

현재 macOS 로컬 앱은 ad-hoc signing만 있으므로 Gatekeeper 배포 준비가 된 앱이 아니다.
