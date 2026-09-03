# Drawing Conventions

이 문서는 실측 도면의 선 종류 어휘와, 각 선 종류에 어떤 굵기·파선을 쓸지 정하는 preset을 다룬다. 구현은 [`src/core/drawing_style.py`](../src/core/drawing_style.py)에 있다.

---

## 왜 별도 계층인가

측정 결과와 표현은 다른 것이다. path의 좌표는 record가 소유하고 hash로 봉인되며 preset이 건드릴 수 없다. preset이 정하는 것은 선 굵기, 파선 간격, 채움뿐이고, 이 값들은 1:1 출력 기준의 **종이 밀리미터**다.

그래서 preset을 바꿔도 측정값은 절대 움직이지 않는다. 반대로 preset이 바뀌면 도면의 바이트는 바뀌므로, 도면은 자기가 그려질 때 쓴 preset의 canonical 해시를 sidecar에 기록한다. 나중에 preset 값을 고치면 예전 도면의 검증이 통과하지 않고 실패한다. 조용히 다른 굵기로 다시 그려지는 일은 없다.

---

## 선 종류 어휘

닫힌 집합이다. `LINE_KINDS`의 순서가 그리는 순서이자 SVG 레이어 순서이며, 이 순서 덕분에 같은 record는 언제 그려도 같은 바이트가 된다.

| 선 종류 | 뜻 | 지금 만드는 곳 |
|---|---|---|
| `section_cut` | 단면 절단선. 닫힌 경우 내부를 해칭한다 | `vector.cutline.v1`의 `section` role |
| `outline_visible` | 보이는 외형선 | `vector.outline.v1`의 `exterior` role |
| `outline_hole` | 유물 내부의 구멍 경계 | `vector.outline.v1`의 `hole` role |
| `center_axis` | 회전축·대칭축의 일점쇄선 | 아직 생산자 없음 |

새 선 종류는 **그것을 실제로 만들어내는 것이 생긴 뒤에** 추가한다. 만들 수 없는 종류를 먼저 이름 붙이면 모든 도면에 빈 레이어가 생기고, preset이 아무도 지키지 않는 관례를 서술하게 된다.

record의 role 이름(`section`, `exterior`, `hole`)은 payload 해시의 일부라 절대 바뀌지 않는다. role에서 선 종류로 가는 매핑은 표현이므로 `RECORD_ROLE_LINE_KINDS`에 있다.

---

## preset과 출처

`preset.source_id`가 `null`이면 **잠정값**이다. 공개된 지침이 뒷받침하지 않는, 그저 1:1 출력에서 읽히도록 고른 값이라는 뜻이다. 이 사실은 preset의 `provisional` 필드로 도면 sidecar에까지 그대로 실린다. 도면을 받은 사람이 그 값이 표준인지 아닌지 확인할 수 있어야 하기 때문이다.

지금 배포되는 preset은 `provisional/v1` 하나이고, 모든 값이 잠정값이다.

### 출처 있는 preset 추가하기

1. 공개 원문에서 수치를 옮겨 적는다. 예를 들어 국립문화유산연구원의 유물 실측 도면 작성 지침, 발굴조사 보고서 작성 기준, 한국고고학회의 도면 작성 기준, 또는 공개된 논문. **기억이나 상용 프로그램 화면에서 옮기지 않는다.**
2. `docs/REFERENCES.md`에 출처를 등재하고 ID를 받는다.
3. `src/core/drawing_style.py`의 `_PRESETS`에 새 항목을 추가한다. `preset_id`는 `기관-지침-연도/v1`처럼 무엇을 따르는지 알 수 있게 짓고, `source_id`에 2단계의 ID를 넣는다.
4. 커밋 메시지에 어느 원문의 어느 항목에서 각 수치를 옮겼는지 적는다.

기존 preset의 값은 고치지 않고 새 preset을 추가한다. 값을 고치면 그 preset으로 만든 기존 도면이 전부 검증에 실패한다.

### 클린룸 원칙

이 프로젝트는 상용 실측 프로그램의 코드나 디컴파일 결과를 보지 않는다. 도면 관례는 공개 지침·논문·공개 표준에서만 가져온다. 상용 제품 고유의 기능 명칭, 아이콘, 색 프로필 이름은 사용하지 않는다. 도면 관례는 특정 회사의 것이 아니라 학계가 공유하는 것이므로, 공개 원문을 인용하는 한 이 경계는 지켜진다.

---

## 현재 잠정값

`provisional/v1`. 모든 값이 종이 밀리미터, 1:1 기준이다.

| 선 종류 | 굵기 (mm) | 파선 (mm) | 채움 |
|---|---|---|---|
| `section_cut` | 0.35 | - | 해칭 |
| `outline_visible` | 0.25 | - | - |
| `outline_hole` | 0.25 | - | - |
| `center_axis` | 0.13 | 4 - 1 - 1 - 1 (일점쇄선) | - |

해칭: 45°, 간격 1.0 mm, 선 굵기 0.13 mm.

출처 ID: 없음. 위 4단계에 따라 채워야 한다.

---

## 사용법

```python
from src.core.artifact_vector_export import VectorSVGOptions, build_vector_export

bundle = build_vector_export(
    document,
    "record:outline-top",
    options=VectorSVGOptions(style_preset="provisional/v1"),
)
```

`style_preset`을 주지 않으면 preset 이전과 **바이트 단위로 같은** 단일 굵기 도면이 나온다. 이 기본값은 의도적이다. preset이 생기기 전에 만들어진 패키지는 자기 SVG를 검증 시점에 다시 렌더해 바이트를 대조하므로, 기본 렌더링이 바뀌면 그 패키지들이 한꺼번에 검증 불가가 된다.

preset을 주면 SVG는 선 종류별 `<g id="layer-...">` 레이어로 나뉘고, Illustrator나 Inkscape에서 레이어로 분리되어 열린다.

---

## 아직 없는 것

축척바, 제목란, 1:N 축척, 결실·복원·균열 같은 상태 표기는 이 계층에 없다. 그것들은 여러 record를 한 장에 배치하는 도판(sheet) 계층의 일이며, 선 종류 어휘가 자리 잡은 다음에 만든다.
