# Windows 설치·실행·빌드 안내

ArchMeshRubbing을 직접 설치해 쓰고, 실행 파일로 만들어 다른 PC에 옮기고,
결과를 검증하는 전체 절차입니다. 프로그램이 무엇을 하는 물건인지는
[README](../README.md)에 있습니다.

## 지원 환경과 배포 상태

| 항목 | 현재 계약 |
|---|---|
| 운영체제 | Windows 10 version 1809(build 17763) 이상 x64, Windows 11 x64 |
| 실행 환경 | native AMD64 PC, 64-bit CPython 3.12 |
| 그래픽 | OpenGL 2.1 compatibility profile, 24-bit 이상 depth buffer |
| 현재 설치 방식 | 저장소를 받은 뒤 source로 실행, 또는 [직접 만든 unsigned 실행 파일](#실행-파일exe-만들기) |
| 공개 바이너리 | 서명된 installer 또는 다운로드용 portable ZIP을 아직 제공하지 않음 |
| 네트워크 | 의존성 설치에는 인터넷이 필요하지만 핵심 기록·저장·검증은 계정과 서버 없이 offline 실행 |

Windows ARM64, x64-on-ARM64 에뮬레이션, 32-bit Windows, Windows Server, macOS, Linux, WSL, Wine/Proton은 지원하지 않습니다. installer, MSIX, Microsoft Store 패키지도 현재 목표가 아닙니다.

저장소 source는 `Apache-2.0`이고, PyQt6를 포함한 바이너리는 결합물로서 `GPL-3.0` 조건으로 전달됩니다. 라이선스상 공개 배포를 막는 요인은 없으며, 남은 것은 서명과 대표 하드웨어 파일럿입니다. 자세한 내용은 [native packaging 정책](NATIVE_PACKAGING.md)을 참고하세요.


## 설치하기 (Windows)

### 어느 쪽으로 설치할까

두 가지 길이 있고, **처음이라면 A**입니다.

| | A. source로 실행 | B. 실행 파일(EXE)로 만들어 쓰기 |
|---|---|---|
| 누구에게 | 직접 써 볼 사람, 고쳐 볼 사람 | Python을 깔 수 없는 PC에 옮겨 쓸 사람 |
| 필요한 것 | Python 3.12 + Git | 빌드용 PC에 Python 3.12 + Git (쓰는 PC에는 아무것도) |
| 걸리는 시간 | 명령 몇 줄 + 패키지 내려받는 시간 | 그보다 한참 오래 (PyInstaller + 자체 검사), 그 뒤로는 복사만 |
| 결과 | 저장소 폴더에서 `main.py --gui` | `ArchMeshRubbing.exe`가 든 폴더 하나 |
| 절차 | [바로 아래](#1-준비물) | [실행 파일(EXE) 만들기](#실행-파일exe-만들기) |

내려받아 바로 쓸 수 있는 **서명된 설치 프로그램은 아직 없습니다.** B로 만든 실행 파일도 서명되지 않은 로컬 빌드라, 처음 실행할 때 Windows가 경고를 냅니다([그 경고 넘기기](#처음-실행할-때-windows가-막을-때)).

### 1. 준비물

- Windows 10 version 1809(build 17763) 이상 x64 또는 Windows 11 x64
- [CPython 3.12 x64](https://www.python.org/downloads/windows/) — 설치할 때 **Add python.exe to PATH**와 **py launcher**를 함께 체크하세요
- [Git for Windows](https://git-scm.com/download/win)
- 최초 Python package 설치를 위한 인터넷 연결 (그 뒤로는 필요 없습니다)
- 디스크 여유 — PyQt6·numpy·scipy가 들어가므로 가상환경이 꽤 큽니다. 실행 파일까지 만든다면 그만큼 더 필요합니다

Python 3.11, 3.13 등 다른 버전을 대신 사용하지 마세요. PowerShell에서 다음 명령으로 3.12 x64가 보이는지 먼저 확인할 수 있습니다.

```powershell
py -0p
py -3.12 -c "import platform,struct; print(platform.python_version(), platform.machine(), struct.calcsize('P') * 8)"
```

마지막 숫자가 `64`여야 합니다.

### 2. source 설치

일반 PowerShell을 열고 다음 명령을 그대로 실행합니다. 가상환경을 activate하지 않고 그 안의 Python을 직접 호출하므로 PowerShell ExecutionPolicy와 다른 Python의 `pip`가 섞이는 문제를 피할 수 있습니다.

```powershell
git clone https://github.com/lzpxilfe/ArchMeshRubbing.git
Set-Location .\ArchMeshRubbing

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Git을 쓰지 않는다면 GitHub의 `Code → Download ZIP`으로 source를 받은 뒤 압축을 풀고, PowerShell에서 그 폴더로 이동해 `py -3.12 -m venv .venv`부터 실행하면 됩니다.

`requirements-optional.txt`는 legacy 실험용입니다. 일반 실행과 검증형 기능 시험에는 설치하지 않아도 됩니다.

### 3. 설치 확인

```powershell
.\.venv\Scripts\python.exe main.py --version

$report = Join-Path $env:TEMP ("ArchMeshRubbing-self-test-{0}.json" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
.\.venv\Scripts\python.exe main.py --self-test-report $report
Write-Host "Self-test report: $report"
$result = Get-Content $report -Raw | ConvertFrom-Json
$result.ok
$result.checks | Where-Object { -not $_.ok }
```

두 명령의 종료 코드가 `0`이고 `$result.ok`가 `True`이며 실패 check가 출력되지 않으면 고정 Python package, Qt offscreen shell, parser, 프로젝트 왕복, 3/6/6 실측, 측정과 기와 전개를 포함한 통합 self-test가 통과한 것입니다. 실행 중 잠시 출력이 없을 수 있습니다. 이 검사는 실제 Windows 화면 frame까지 증명하지 않습니다. 지원 Windows runtime은 GUI 시작 때 다시 강제되며 native frame은 [OpenGL 진단](#창이-검게-보이거나-opengl-오류가-남)으로 별도 확인할 수 있습니다. report는 기존 파일을 덮어쓰지 않으므로 예시는 매번 새 시각 이름을 만듭니다.

### 4. 앱 실행

```powershell
.\.venv\Scripts\python.exe main.py --gui
```

특정 파일이나 프로젝트를 바로 열 수도 있습니다.

```powershell
.\.venv\Scripts\python.exe main.py --open-mesh "D:\scans\roof-tile.ply"
.\.venv\Scripts\python.exe main.py --open-project "D:\results\roof-tile.amr"
```

이후 다시 실행할 때는 저장소 폴더에서 마지막 `--gui` 명령만 사용하면 됩니다.


## 실행 파일(EXE) 만들기

Python이 깔려 있지 않은 PC — 발굴 현장 노트북, 기관의 공용 PC — 에서도 쓰려면 실행 파일로 만들어 폴더째 옮기면 됩니다. **쓰는 쪽 PC에는 아무것도 설치하지 않아도 됩니다.**

### 무엇이 만들어지나

설치 프로그램(`setup.exe`)이 아니라 **폴더 하나**가 만들어집니다. PyInstaller의 onedir 방식이라 `.exe` 하나만 떼어 내면 실행되지 않습니다 — 폴더 전체가 한 벌입니다.

```text
dist\ArchMeshRubbing\
├─ ArchMeshRubbing.exe          ← 이걸 실행합니다
├─ _internal\                    ← Python 런타임, Qt, 라이브러리 (건드리지 마세요)
├─ source\
│  ├─ ArchMeshRubbing-source.zip   ← 이 실행 파일에 대응하는 정확한 소스
│  └─ ArchMeshRubbing-source.json  ← 그 ZIP의 hash와 commit
└─ release-evidence\             ← 무엇으로 만들어졌는지의 기록
```

`source\` 폴더가 함께 들어가는 데는 이유가 있습니다. 이 실행 파일은 PyQt6를 품고 있어서 **결합물로서 GPL-3.0 조건으로 전달**되고, GPL-3.0은 받는 사람에게 대응하는 소스를 줄 것을 요구합니다. 그래서 빌드가 그 소스를 — live worktree의 복사본이 아니라 **정확히 그 commit의 Git object에서** — 만들어 실행 파일 옆에 넣습니다. **남에게 전달할 때 이 폴더를 빼지 마세요.** 소스만 따로 받는 쪽에는 Apache-2.0만 적용됩니다. 라이선스 판단 자체는 이 문서가 대신하지 않으니, 기관 배포라면 [native packaging 문서](NATIVE_PACKAGING.md)의 정책 절을 함께 보세요.

### 1. 빌드 전용 clone

**source 실행에 쓰던 폴더에서 빌드하지 마세요.** 빌드는 Git worktree가 완전히 깨끗할 것을 요구하는데, 앱을 돌린 폴더에는 `__pycache__`, 시험용 `.amr`, 로그 같은 것이 남아 있어 거의 반드시 거부됩니다. 새로 clone합니다.

```powershell
git clone https://github.com/lzpxilfe/ArchMeshRubbing.git ArchMeshRubbing-build
Set-Location .\ArchMeshRubbing-build
```

### 2. 정확히 고정된 의존성 설치

```powershell
py -3.12 -m venv .venv
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python.exe -m pip install --require-hashes --only-binary=:all: -r requirements\windows-py312-x64-hashed.lock
.\.venv\Scripts\python.exe -m pip check
```

- `PYTHONDONTWRITEBYTECODE = "1"` — Python이 `.pyc` 캐시를 만들지 않게 합니다. 만들면 그것이 untracked 파일이 되어 다음 단계에서 빌드가 거부됩니다.
- `--require-hashes` — 잠금 파일에 적힌 hash와 맞는 wheel만 받습니다. 같은 commit에서 누가 빌드하든 같은 것이 들어갑니다.
- 이 잠금 파일은 실행용(numpy, PyQt6 …)과 빌드용(PyInstaller …)을 모두 담고 있어 한 번에 끝납니다.

### 3. 빌드

```powershell
.\.venv\Scripts\python.exe tools\build_native.py
```

PyInstaller가 패키징한 뒤 만들어진 실행 파일로 자체 검사까지 돌리므로 source 설치보다 한참 오래 걸립니다. 중간에 출력이 없어도 기다리세요. 끝나면 이렇게 나옵니다.

```text
Local unsigned artifact: ...\dist\ArchMeshRubbing\ArchMeshRubbing.exe
Embedded build manifest: ...\build\generated\build_info.json
Verified corresponding source: ...\dist\ArchMeshRubbing\source\ArchMeshRubbing-source.zip
Verified release evidence: ...\dist\ArchMeshRubbing\release-evidence
Frozen self-test passed: ...
No artifact was signed, installed, uploaded, or published.
```

`Frozen self-test passed`가 중요합니다. 빌드 도구가 **만들어진 실행 파일을 직접 돌려** 프로젝트 왕복·3/6/6 실측·측정·기와 전개까지 통과하는지 확인한 뒤에야 성공이라고 말합니다.

쓸 수 있는 선택지:

| 옵션 | 언제 |
|---|---|
| `--replace-existing` | 이미 있는 `build`/`dist` 결과를 지우고 다시 만들 때. 기본은 덮어쓰지 않고 멈춥니다 |
| `--commit <hash>` | HEAD가 아닌 특정 commit으로 찍을 때 (그 commit이 실제로 checkout돼 있어야 합니다) |
| `--channel <이름>` | 빌드 기록에 남길 이름. 기본 `local-smoke` |
| `--clean-cache` | PyInstaller 캐시를 비우고 시작할 때 |
| `--skip-self-test` | 위의 자체 검사를 건너뛸 때. **권장하지 않습니다** |

### 4. 만들어진 것 확인

```powershell
.\dist\ArchMeshRubbing\ArchMeshRubbing.exe --version
.\dist\ArchMeshRubbing\ArchMeshRubbing.exe --gui
```

GUI-subsystem 실행 파일이라 PowerShell이 종료를 기다리지 않습니다. CLI로 쓰며 종료 코드를 봐야 한다면 이렇게 합니다.

```powershell
$p = Start-Process -FilePath .\dist\ArchMeshRubbing\ArchMeshRubbing.exe `
  -ArgumentList "--version" -Wait -PassThru
$p.ExitCode
```

### 5. 다른 PC로 옮기기

1. `dist\ArchMeshRubbing` **폴더 전체**를 복사합니다. `.exe` 하나만 떼면 실행되지 않습니다.
2. 옮긴 PC에 Python도 Git도 필요 없습니다. Windows 10 1809 이상 x64 또는 Windows 11 x64면 됩니다.
3. `ArchMeshRubbing.exe`를 실행합니다.
4. 남에게 전달한다면 `source\` 폴더를 빼지 마세요 — 위에 적은 GPL-3.0 조건이 거기에 걸려 있습니다.

#### 처음 실행할 때 Windows가 막을 때

서명하지 않은 실행 파일이라 **"Windows의 PC 보호"** (SmartScreen) 파란 창이 뜰 수 있습니다. 서명 인증서를 아직 붙이지 않아서이지, 파일이 손상됐다는 뜻은 아닙니다. 자기가 빌드한 것이 맞다면 `추가 정보` → `실행`으로 넘어갑니다.

인터넷으로 받은 ZIP을 푼 경우에는 파일마다 차단 표시가 붙어 있을 수 있습니다. 폴더째 풀기 전에 ZIP의 속성에서 `차단 해제`를 체크하거나, 푼 뒤 PowerShell에서 지웁니다.

```powershell
Get-ChildItem -Recurse .\ArchMeshRubbing | Unblock-File
```

**출처를 모르는 빌드에는 이렇게 하지 마세요.** 서명이 없다는 것은 누가 만들었는지 파일 스스로 증명하지 못한다는 뜻이고, 그래서 이 저장소도 공개 바이너리를 아직 올리지 않습니다.

### 6. portable ZIP (선택)

폴더째 옮기는 대신 검증 가능한 ZIP 하나로 만들 수도 있습니다. 받는 쪽이 내용물이 온전한지 스스로 확인할 수 있습니다.

```powershell
$epoch = [int64]((& git show -s --format=%ct HEAD).Trim())
.\.venv\Scripts\python.exe tools\build_portable_archive.py build `
  --payload dist\ArchMeshRubbing `
  --archive build\ArchMeshRubbing-Windows-x64-portable.zip `
  --manifest build\ArchMeshRubbing-Windows-x64-portable.zip.manifest.json `
  --source-date-epoch $epoch

.\.venv\Scripts\python.exe tools\build_portable_archive.py verify `
  --archive build\ArchMeshRubbing-Windows-x64-portable.zip `
  --manifest build\ArchMeshRubbing-Windows-x64-portable.zip.manifest.json
```

`--source-date-epoch`에 commit 시각을 넣으므로 같은 commit에서 만든 ZIP은 bytes까지 같습니다. ZIP과 manifest를 함께 전달하면 받는 쪽이 `verify`로 다시 검사할 수 있습니다.

### 빌드가 멈출 때

빌드 도구는 애매하면 진행하지 않고 이유를 말하고 멈춥니다. 자주 나오는 것들입니다.

| 멈춘 이유 | 뜻과 해결 |
|---|---|
| `require a clean Git worktree; untracked paths exist` | 추적되지 않는 파일이 있습니다. `git status`로 확인해 지우거나, 빌드 전용 clone에서 다시 하세요. `.pyc`가 원인이면 `PYTHONDONTWRITEBYTECODE` 설정을 빠뜨린 것입니다 |
| `tracked worktree content does not match HEAD` | 추적 파일을 고쳤습니다. commit하거나 되돌리세요 |
| `environment does not match the exact lock` | 의존성이 잠금 파일과 다릅니다. 메시지가 알려 주는 `pip install -r requirements\build-py312.lock`을 그대로 실행하거나, 위 2단계를 새 venv에서 다시 하세요 |
| `outputs already exist; refusing to overwrite` | 지난 빌드 결과가 남아 있습니다. `build`/`dist`를 지우거나 `--replace-existing`을 붙이세요 |
| `require CPython 3.12` | 3.11이나 3.13으로 venv를 만들었습니다. `py -3.12 -m venv`로 다시 만드세요 |
| `supported only on native AMD64 Windows` | ARM64 PC이거나 32-bit Python입니다. 지원 대상이 아닙니다 |
| `packaged self-test reported failure` | 실행 파일은 만들어졌지만 자체 검사에서 떨어졌습니다. 메시지가 가리키는 report JSON의 실패 check를 보세요. **이 상태의 실행 파일을 배포하지 마세요** |

### 하지 않는 것

이 절차는 **설치 프로그램을 만들지 않고, 서명하지 않고, 업로드하지 않고, 시작 메뉴에 등록하지 않습니다.** MSIX와 Microsoft Store 패키지도 현재 목표가 아닙니다. 만들어진 것은 로컬에서 쓰고 직접 전달하는 unsigned 빌드입니다. 자세한 정책과 검증 항목은 [native packaging 문서](NATIVE_PACKAGING.md)에 있습니다.


## 지원 파일

| 형식 | 비고 |
|---|---|
| `.obj` | 상대 경로의 MTL과 texture를 함께 캡처 가능 |
| `.ply` | ASCII/binary 및 상대 `TextureFile` 처리 |
| `.stl` | ASCII/binary |
| `.off` | text mesh |
| `.gltf` | self-contained 또는 원본 폴더 아래 상대 buffer/image |
| `.glb` | glTF Binary |

HTTP/file URI, 절대 resource 경로, 원본 폴더 밖으로 나가는 `..`, symlink 탈출은 허용하지 않습니다. OBJ나 glTF처럼 부속 파일이 있는 자료는 파일 하나만 떼지 말고 원래의 상대 폴더 구조 전체를 복사하세요.

UV와 texture bytes의 프로젝트 보존·오프라인 재현은 검증하지만 여러 material/PBR 조합의 화면 렌더링 충실도는 아직 현장 검증 전입니다. 현재 authoritative SVG, PNG와 기와 전개는 geometry 중심 산출물입니다.

현재 import 상한은 주 원본 4 GiB, text parser 입력 256 MiB, 5,000,000 vertices, 2,000,000 triangles입니다. 기와 전개의 선택 기록면은 최대 250,000 faces입니다. 첫 시험은 원본을 보존한 채 충분히 작은 decimated 복사본으로 시작하는 편이 좋습니다. parser는 아직 별도 보안 process sandbox가 아니므로 출처와 내용을 신뢰할 수 있는 스캔만 여세요.


## 내 스캔 파일로 첫 실물 테스트

### 시험 전에 준비할 것

1. 유일한 원본이 아닌 **복사본**을 준비합니다.
2. 스캐너 또는 export 설정에서 실제 단위가 `mm`, `cm`, `m` 중 무엇인지 확인합니다.
3. 축 방향과 실제로 알고 있는 길이 한 곳을 적어둡니다. 나중에 1:1 scale을 대조할 기준입니다.
4. OBJ·glTF·texture 자료는 부속 파일과 상대 폴더 구조를 함께 복사합니다.
5. 처음에는 한 유물, 한 연결 mesh, 가능한 한 작은 시험본으로 시작합니다.

### 기본 실측 한 바퀴

1. `4축 작업 흐름 → 메쉬 열기`에서 복사한 파일을 엽니다.
2. `원본 단위·좌표축 확인`에서 단위와 signed 축 매핑을 실제 scan 설정대로 선택하고 확인란을 체크합니다. 모르면 추정하지 말고 scanner/export 설정을 먼저 확인하세요.
3. 화면에서 형상과 크기를 확인합니다. 선택·측정 도구가 꺼진 기본 카메라 모드의 조작은 `좌클릭 드래그=회전`, `우클릭 드래그=이동`, `휠=확대·축소`입니다. `1~6`은 정면·후면·우측·좌측·상면·하면, `F`는 메쉬 맞춤, `R`은 뷰 초기화입니다.
4. 상단 정위치 툴바에서 이동·회전 값을 조절합니다. native 문서에서는 scale로 단위를 보정하지 않습니다. 현재 `바닥면 맞춤`, 3점·면·브러시 자동 바닥 정렬은 검증 Align revision으로 아직 이식되지 않았으므로 수동 이동·회전을 사용합니다.
5. 자세가 맞으면 변화량이 `0`이어도 `정치 확정`을 누릅니다. 이때 immutable Align revision이 생기고 검증 실측 버튼이 열립니다.
6. `Ctrl+S`로 `roof-tile.amr` 같은 프로젝트를 먼저 저장합니다. 창 제목의 `*`는 현재 문서에 저장되지 않은 변경이 있다는 뜻입니다.
7. `검증된 실측 · 기와 전개 열기`를 누릅니다.
8. Cutline에서 Top, Front, Right를 각각 선택하고 필요한 mm 평면 위치를 정해 `단면 계산 · 기록`을 실행합니다.
9. Cutline이 `3/3`이 되면 Outline의 6면을 각각 `외곽 계산 · 기록`합니다. 외곽 정밀도 격자보다 좁은 특징은 합쳐질 수 있으므로 결과와 QC를 확인합니다.
10. Outline이 `6/6`이 되면 Digital Rubbing의 6면을 각각 선택해 `탁본 계산 · 기록`을 실행합니다. 실제 크기가 큰 유물에서 `px/mm`를 지나치게 높이면 raster가 매우 커질 수 있으므로 기본값부터 시험하세요.
11. 완료 버튼이 초록색이고 진행도가 `3/3 · 6/6 · 6/6`인지 확인합니다. 아직 export하지 않습니다.
12. `제원측정 도구 열기`에서 표면적·체적, 두 점 거리, 선택점 원 맞춤 지름을 필요에 따라 기록합니다.
13. 기와 전개도 시험한다면 아래 절차로 전개 record까지 먼저 만듭니다. 모든 record가 준비된 뒤 [저장과 export 순서](#저장과-export-순서)를 따릅니다.

거리는 표면을 따라가는 geodesic이 아니라 두 surface anchor 사이의 **3D 직선 거리**입니다. 지름은 유물의 최대 외경이 아니라 3~64개 선택점을 best-fit한 평면 원의 지름입니다. 체적은 단일 연결·폐쇄·일관된 winding의 edge-manifold mesh에서만 제공하며, 열린 mesh나 비다양체·다중 조각에서는 근사값으로 위장하지 않고 unavailable로 남깁니다.

### 기와 기록면 전개 시험

기와 전개는 Align 확정 직후부터 별도로 시험할 수 있습니다. 요철이 살아 있는 실물 기와는 **측정한 축 기준 전개**로 펴야 합니다. 단면마다 원을 맞추는 기본 전개는 타날문·포목흔의 요철을 중심 오차로 읽어 왜곡 게이트에서 거부하기 때문입니다. 그래서 먼저 와통 축으로 세웁니다.

1. 기와를 만든 원통(와통)에 닿았던 **내면(오목면)**을 `표면 보정 도구`의 클릭·브러시·올가미로 칠해 선택하고 `와통 축 측정 · 정치 (현재 선택 면)`을 누릅니다. 선택면 전체로 원통을 맞춰 `measurement.mandrel_cylinder.v1` 기록(반지름, 잔차 rms, 둘레 각, 축 길이)을 남기고 그 축을 +Z로 세우는 Align을 만듭니다. 기와에는 구연·저부가 없고, 호의 단면 원 두 개로는 호가 좁아 중심이 미끄러져 축이 서지 않습니다.
2. 외면 또는 내면 중 실제로 기록할 **한쪽 open surface patch**를 칠해 고릅니다. 두꺼운 폐합 mesh 전체는 전개되지 않습니다.
3. `기록 영역`을 `현재 선택 면`으로, 길이축을 `Z`로 두고 `측정한 축 기준 전개 · 토기 외면 띠, 와통으로 세운 기와`를 켭니다. 이 항목은 회전축이나 와통 축으로 정치한 문서에서만 켜집니다.
4. 풍화된 기와편은 패임·박리·돌출의 옆면이 전개에서 뒤집히거나 겹쳐 거부됩니다. `펼 수 있는 면만 남기기 · 언더컷 제외`를 누르면 전개가 할 계산을 그대로 돌려, 뒤집히는 면과 겹칠 때 종이가 닿지 못하는 뒤쪽 면을 빼고 나머지를 선택으로 둡니다. 종이 탁본이 그 자리를 건너뛰고 비워 두는 것과 같습니다. 이 결과는 `surface.unrollable_selection.v1` 기록(칠한 면, 뺀 면 하나하나, 게이트별 수, 뺀 넓이의 몫, 자리)으로 남고 뺀 면의 수·넓이·자리가 상태 표시줄에 나옵니다. 남은 선택으로 전개하면 전개가 이 기록에 이어져, 전개와 그 위 탁본의 export가 무엇을 비웠는지 함께 싣습니다.
5. `상면/하면`은 표면 자동 분류가 아니라 같은 선택면의 펼침 방향 해석입니다. 올바른 기록면 선택은 사용자가 확인해야 합니다.
6. 펼침 경계는 먼저 `자동 경계`, 단면 수는 기본 `32`로 시험하세요. 허용 범위는 `12~96`입니다.
7. 경계를 연구자가 고정해야 하면 `고정 각도`를 선택해 `[-180°, 180°)` 범위의 값을 지정합니다.
8. `기와 전개 계산 · 기록` 후 section fit, 왜곡, collapse, foldover, overlap QC를 확인합니다. 실패를 억지로 export하지 말고 기록면 선택·장축·seam을 다시 확인합니다.
9. READY + FRESH 결과가 만들어졌는지 확인합니다. 모든 기록을 마친 뒤 아래 공통 순서에서 저장하고 export합니다.

요철이 거의 없는 기와(게임용으로 정리한 저폴리곤 자산 등)는 1·3·4 없이 기본 전개로도 통과할 수 있습니다.

정식 전개 record가 통과하려면 선택 patch는 하나의 edge-connected component이고, 최소 하나의 닫힌 비분기 경계 고리를 가진 open surface이며, triangle orientation이 일관돼야 합니다. duplicate face, non-manifold edge와 폐합 shell 전체는 거부됩니다. 선택면 내부를 가르는 고정 seam은 foldover나 overlap을 만들면 거부될 수 있습니다. 토기 외면의 띠는 손으로 칠하지 않아도 됩니다. `회전축 기준 외면 띠`에 기준 자오선 각도, 띠 폭, 높이 범위를 넣고 `외면 띠 선택`을 누르면 정치한 축을 기준으로 외면만 잘라 현재 선택 면으로 둡니다. 안팎은 면 법선의 방향과 두 겹의 반지름 대소를 함께 보고 가리며, 뒤집힌 메쉬는 안쪽 벽을 내주는 대신 거부합니다.

펼친 좌표 위에 요철을 직접 그리는 전개 탁본은 있습니다. READY + FRESH 전개 기록을 고른 뒤 `선택한 전개 위에 탁본 계산 · 기록`을 누르면 탁본 항목의 해상도·기준 반경·검정 기준 깊이·먹 농도·극성으로 전개 위의 탁본 raster를 만들고, 탁본 기록 목록에서 골라 같은 1:1 PNG 패키지로 내보냅니다. 탁본 모델은 둥근 원판으로 눌러 붙인 종이(창 1.5 mm, 검정 0.15 mm)로 열리고, 표면 극성 `양각`(기본: 먹 45%, 기저 없음)이면 종이가 닿는 곳에 먹이 앉는 보통 탁본으로 보고서 도판의 탁본처럼 파인 곳·기공·균열이 흰 종이로 남습니다. `음각`(먹 85%, 기저 20%)이면 고른 바탕 위에 파인 곳이 검게 섭니다. 어느 쪽이든 종이가 파단면으로 꺾이는 가장자리는 진해집니다. 실제 종이 탁본의 자글자글한 먹 알갱이는 종이 섬유라 3D 스캔에 없으므로 그리지 않습니다. 배경과 실측 수치는 [`POTTERY_STRIP_UNWRAP.md`](POTTERY_STRIP_UNWRAP.md)에 있습니다. 원본 texture를 펼친 좌표 위에 재투영하는 기능은 아직 없습니다.

### 저장과 export 순서

모든 export는 **생성 당시의 전체 프로젝트 hash**에 결박됩니다. 따라서 record를 모두 만든 다음 프로젝트를 저장하고, 그 exact 상태에서 export해야 합니다.

1. 원하는 Cutline, Outline, Rubbing, 제원과 기와 record를 모두 마칩니다.
2. `Ctrl+S`로 최신 문서를 `.amr`에 저장합니다.
3. 필요한 Cutline/Outline record를 선택해 `선택한 검증 벡터 1:1 SVG 내보내기`로 `.amr-vector`를 만듭니다.
4. 필요한 Rubbing record를 선택해 `선택한 검증 탁본 1:1 PNG 패키지 내보내기`로 `.amr-rubbing`을 만듭니다.
5. 진행도가 `3/3 · 6/6 · 6/6`이면 `완료 실측 15개 원자 묶음 내보내기`로 `.amr-survey`를 만듭니다.
6. 기와 전개를 기록했다면 READY + FRESH 결과를 선택해 `선택한 검증 전개 1:1 OBJ · SVG 패키지 내보내기`로 `.amr-unwrap`을 만듭니다.
7. 이후 Align이나 record를 추가·변경했다면 `Ctrl+S` 후 필요한 package를 새 이름으로 다시 export합니다. 이전 package는 생성 당시 프로젝트의 증거이므로 최신 `.amr`와 `--against-project` exact-match하지 않습니다.
8. 앱을 닫았다가 저장한 프로젝트를 재개방합니다.

### 프로젝트 독립성 확인

1. 저장한 `.amr`를 다른 시험 폴더에 복사합니다.
2. 앱을 완전히 닫습니다.
3. 원본 **복사본** 폴더의 이름을 바꾸거나 다른 곳으로 옮깁니다. 유일한 원본은 삭제하지 마세요.
4. 복사한 `.amr`를 `프로젝트 열기`로 엽니다.
5. mesh, 단위, Align, record 목록과 완료 진행도가 복원되는지 확인합니다. READY + FRESH 기록을 다시 선택해 미리보기가 recipe에서 재계산되는지도 확인합니다.

신규 `.amr`는 주 원본과 parser가 실제 사용한 허용 dependency를 content-addressed blob으로 포함합니다. 이 시험이 통과하면 source 경로에 의존하지 않는 프로젝트 왕복을 확인한 것입니다.


## 결과 파일과 오프라인 검증

| 경로 | 내용 |
|---|---|
| `artifact.amr` | 내장 원본, dependency, metadata, Align과 모든 record가 있는 프로젝트 파일 |
| `*.amr-vector/` | `artifact.svg`와 vector provenance JSON |
| `*.amr-rubbing/` | `artifact.png`와 rubbing provenance JSON |
| `*.amr-survey/` | 9개 vector package, 6개 rubbing package와 aggregate manifest |
| `*.amr-unwrap/` | canonical binary, 평면 OBJ, 실제 mm 1:1 SVG와 provenance JSON |

`.amr-*` 결과는 이름에 확장자가 붙은 **폴더 package**입니다. 검증 export와 JSON report는 기존 목적지를 덮어쓰지 않으므로 재시험할 때는 새 이름을 사용하세요.

source 실행에서는 다음처럼 검증합니다.

```powershell
.\.venv\Scripts\python.exe main.py --verify-artifact "D:\results\roof-tile.amr" `
  --report "D:\results\project-verification.json"

.\.venv\Scripts\python.exe main.py --verify-artifact "D:\results\roof-tile.amr-unwrap" `
  --against-project "D:\results\roof-tile.amr" `
  --report "D:\results\unwrap-verification.json"

.\.venv\Scripts\python.exe main.py --verify-artifact "D:\results\roof-tile.amr-survey" `
  --against-project "D:\results\roof-tile.amr" `
  --report "D:\results\survey-verification.json"
```

종료 코드 `0`과 report의 `ok: true`가 성공입니다. `1`은 자료 검증 실패, `2`는 잘못된 옵션 또는 report 저장 실패입니다. 위 source 명령은 PowerShell이 process 종료를 직접 기다리므로 검증용으로 권장합니다. 로컬 GUI-subsystem EXE를 CLI에 사용할 때는 `Start-Process -Wait -PassThru`로 종료를 기다리고 `.ExitCode`를 별도로 확인해야 합니다.

마지막으로 SVG를 Illustrator 또는 Inkscape에서 열고, 알고 있는 길이를 재거나 100% scale로 출력해 자·캘리퍼스로 확인하세요. 자동 검증은 파일 내부의 mm 계약을 확인하지만 실제 printer 설정과 외부 프로그램의 import 동작까지 대신 증명하지 않습니다.


## 첫 시험 합격 체크리스트

- [ ] 설치 self-test의 `ok`가 `true`다.
- [ ] mesh가 열리고 vertex/triangle 수와 실제 단위·크기가 예상과 맞다.
- [ ] 명시적 `정치 확정` 뒤 검증 실측과 기와 전개가 활성화된다.
- [ ] `.amr` 저장, 종료, 재열기 뒤 Align과 record 진행도가 복원된다.
- [ ] Cutline/Outline/Rubbing 또는 기와 전개 결과가 READY + FRESH다.
- [ ] export package를 `--against-project`로 검증했을 때 `ok`가 `true`다.
- [ ] 외부 프로그램에서 알려진 길이와 1:1 scale이 맞다.
- [ ] 원본 파일은 수정되거나 삭제되지 않았다.

시험 결과를 공유할 때는 Windows version/build, GPU, RAM, 원본 format, 실제 단위, vertex/triangle 수, 단계별 성공 여부와 처리시간, 오류 문구, 필요한 화면 캡처를 함께 주세요. 절대 경로, 소장 위치, 미공개 유물 정보와 개인정보는 제거하세요. 정식 10항목 파일럿 절차는 [FIELD_PILOT.md](FIELD_PILOT.md)에 있습니다.


## 문제 해결

### `py -3.12`를 찾지 못함

`py -0p`로 설치된 Python을 확인하고 CPython 3.12 x64를 설치하세요. ARM64 Python, 32-bit Python, 3.11/3.13은 GUI 지원 계약에 포함되지 않습니다.

### package import 실패 또는 module 없음

저장소 폴더에서 가상환경 Python을 직접 사용했는지 확인합니다.

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import PyQt6,OpenGL,numpy,trimesh,shapely; print('imports ok')"
```

### 창이 검게 보이거나 OpenGL 오류가 남

그래픽 driver를 갱신한 뒤 software OpenGL 경로를 별도 PowerShell에서 시험합니다.

```powershell
$env:QT_OPENGL = "software"
$report = Join-Path $env:TEMP ("ArchMeshRubbing-opengl-{0}.json" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
.\.venv\Scripts\python.exe main.py --opengl-driver-smoke-report $report
.\.venv\Scripts\python.exe main.py --gui
```

원래 환경으로 되돌리려면 `Remove-Item Env:QT_OPENGL`을 실행합니다.

### 다음 단계 버튼이 비활성화됨

- Open 뒤 `정치 확정`을 명시적으로 한 번 실행했는지 확인합니다.
- Outline 전 Cutline Top/Front/Right `3/3`, Rubbing 전 Outline 6면 `6/6`이 READY + FRESH인지 확인합니다.
- Align을 바꿨다면 이전 record는 stale이므로 현재 Align에서 다시 기록해야 합니다.

### 기와 전개가 QC에서 거부됨

전체 폐합 mesh 대신 한쪽 기록면 patch를 선택하고, 장축과 seam을 다시 확인하세요. 단면 수는 먼저 `32`, 허용 범위는 `12~96`을 사용합니다. QC failure나 fallback은 정식 결과로 게시되지 않는 것이 정상입니다.

### 로그 확인

```powershell
Get-Content "$env:LOCALAPPDATA\ArchMeshRubbing\logs\archmeshrubbing.log" -Tail 200
```

앱의 `도움말 → 디버그 정보 복사`도 함께 사용하면 실행 환경과 module 위치를 확인하기 쉽습니다.


## CLI 보조 도구

```powershell
.\.venv\Scripts\python.exe main.py --help
.\.venv\Scripts\python.exe main.py --info "D:\scans\roof-tile.ply"
.\.venv\Scripts\python.exe main.py --flatten "D:\scans\roof-tile.ply" "D:\results\quick-rubbing.tiff"
.\.venv\Scripts\python.exe main.py --review "D:\scans\roof-tile.ply" "D:\results\review.png"
.\.venv\Scripts\python.exe main.py --project "D:\scans\roof-tile.ply" "D:\results\planview.png"
.\.venv\Scripts\python.exe main.py --separate "D:\scans\roof-tile.ply"
.\.venv\Scripts\python.exe main.py --generate-synthetic sugkiwa_quarter 7 "D:\results\synthetic-tile.obj"
.\.venv\Scripts\python.exe main.py --benchmark-synthetic "D:\results\benchmarks" 1,2,3
```

`--flatten`, `--review`, `--project`, `--separate`는 빠른 legacy 검토 경로이며 ArtifactDocument의 검증 export가 아닙니다. 합성 기와의 생성물과 평가 방법은 [SYNTHETIC_BENCHMARKS.md](SYNTHETIC_BENCHMARKS.md)를 참고하세요.


## 개발과 품질 확인

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -c "import subprocess,sys; raise SystemExit(subprocess.call([sys.executable,'-m','pyright','--pythonpath',sys.executable,'-p','pyright-m0.json']))"
.\.venv\Scripts\python.exe -m pytest -q
```

차단 품질 게이트는 Windows에서 full pytest, Ruff, M0 trust-kernel Pyright와 native `qwindows` software OpenGL frame을 검사합니다. 전체 tree Pyright는 기존 타입 부채를 계속 보고하지만 아직 차단 게이트는 아닙니다. CI와 local portable build는 실제 Windows hardware GPU, compositor presentation 또는 고고학자 판정을 대신하지 않습니다.

- [Windows CI workflow](https://github.com/lzpxilfe/ArchMeshRubbing/actions/workflows/ci.yml)
- [Windows portable package smoke](https://github.com/lzpxilfe/ArchMeshRubbing/actions/workflows/package-smoke.yml)
