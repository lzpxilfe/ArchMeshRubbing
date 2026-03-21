# Synthetic Benchmark Guide / 합성 벤치마크 가이드

## Why This Exists / 왜 필요한가

실제 기와 스캔이 충분하지 않은 단계에서도, 알고리즘과 워크플로우를 검증할 수 있는 기준 세트가 필요합니다.

Even before you have enough real roof-tile scans, you still need a baseline dataset for checking algorithms and workflows.

ArchMeshRubbing의 synthetic benchmark는 다음을 함께 제공합니다.

ArchMeshRubbing synthetic benchmarks provide all of the following together:

- fabrication-aware synthetic tile meshes
- ground-truth tile interpretation state
- evaluation reports against that ground truth
- recording-surface review sheets for visual regression

## What Gets Generated / 생성되는 것

단일 synthetic bundle은 보통 아래 파일을 만듭니다.

A single synthetic bundle usually creates these files:

- `synthetic_*.obj`
- `synthetic_*.truth.json`
- `synthetic_*.interpretation.json`
- `synthetic_*.evaluation.json`
- `synthetic_*.review.png`
- `synthetic_*.bundle.json`

benchmark suite는 여기에 더해 전체 요약 파일도 만듭니다.

A benchmark suite also writes summary files:

- `synthetic_benchmark_summary.json`
- `synthetic_benchmark_summary.csv`

## CLI Workflow / CLI 사용법

### Single Case / 단일 케이스

```bash
python main.py --generate-synthetic sugkiwa_quarter 7 synthetic_tile.obj
```

이 명령은 synthetic mesh와 정답, 평가, review sheet를 한 묶음으로 저장합니다.

This command saves the synthetic mesh, ground truth, evaluation, and review sheet as one bundle.

### Benchmark Suite / 벤치마크 묶음

```bash
python main.py --benchmark-synthetic ./benchmarks 1,2,3
```

이 명령은 preset별로 여러 seed를 돌려 benchmark suite를 만들고, 각 case의 review sheet와 요약 CSV/JSON을 저장합니다.

This command runs multiple seeds across presets and saves per-case review sheets plus CSV/JSON summaries.

## GUI Workflow / GUI 사용법

기와 해석 패널의 `🧪 합성 데이터 / 정답 평가` 그룹에서 다음을 사용할 수 있습니다.

Inside the tile interpretation panel, open `🧪 합성 데이터 / 정답 평가`:

1. `합성 기와 생성`
   현재 preset과 seed로 synthetic tile 1개를 장면에 추가합니다.
   Generates one synthetic tile and inserts it into the scene.

2. `정답 평가 실행`
   현재 해석 상태를 synthetic ground truth와 비교합니다.
   Compares the current interpretation against the synthetic ground truth.

3. `정답 가설 적용`
   정답 상태를 현재 해석 상태로 복원합니다.
   Restores the ground-truth interpretation as the active state.

4. `합성 벤치마크 묶음 저장`
   현재 synthetic object의 mesh/truth/evaluation/review를 저장합니다.
   Saves the current synthetic object's mesh/truth/evaluation/review bundle.

5. `합성 benchmark suite 저장`
   여러 preset × seed 조합을 한 번에 생성해 suite를 저장합니다.
   Generates and saves a full benchmark suite across multiple presets and seeds.

## Pass/Fail Threshold / 합격 기준점수

synthetic benchmark suite는 `pass threshold`를 기준으로 통과/실패를 요약합니다.

The synthetic benchmark suite uses a pass threshold to summarize which cases passed or failed.

- score가 threshold 이상이면 `pass`
- score가 threshold 미만이면 `fail`

- If a score is greater than or equal to the threshold, the case passes.
- If it is below the threshold, the case fails.

이 기준은 알고리즘 변경 후 regression check를 할 때 유용합니다.

This is useful as a regression gate after algorithm changes.

## Review Sheets / 검토 시트

synthetic review sheet는 단순 wireframe이 아니라 `연속 기록면` 관점의 시각 회귀 자료입니다.

The synthetic review sheet is not a wireframe dump. It is a visual regression artifact centered on the idea of a continuous recording surface.

보통 다음 정보를 함께 담습니다.

It usually includes:

- continuous rubbing-style unwrap
- outline confirmation image
- scale bar and orientation legend
- tile class / split scheme / record surface info
- evaluation-related context

## Recommended Use / 권장 사용 방식

1. synthetic suite를 먼저 만든다.
2. 평균 점수와 실패 케이스를 본다.
3. 실패 케이스의 `*.review.png`를 먼저 확인한다.
4. seam, section guides, top/bottom orientation을 조정한다.
5. 실제 기와 메쉬가 들어오면 synthetic suite와 함께 비교한다.

1. Generate a synthetic suite first.
2. Inspect average score and failing cases.
3. Open the `*.review.png` sheets for failures first.
4. Tune seams, section guides, and top/bottom orientation.
5. Compare the same workflow against real roof-tile meshes when they become available.
