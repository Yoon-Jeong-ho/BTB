from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
OBSERVED = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'
ANALYSIS = UNIT_ROOT / 'analysis.md'
STABLE = '''# 03 DeepSpeed ZeRO 분석

## 해석 프레임
- ZeRO는 data parallel 중복 상태를 shard해 per-rank memory를 줄인다.
- stage가 올라갈수록 memory는 줄지만 communication/checkpoint complexity는 커진다.
- 이 단위의 숫자는 실제 DeepSpeed 실행이 아니라 memory accounting intuition이다.
'''

def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding='utf-8'))

def ensure() -> None:
    missing = [p for p in [SCRATCH, FRAMEWORK] if not p.exists()]
    if missing:
        names = ', '.join(str(p.relative_to(UNIT_ROOT)) for p in missing)
        raise SystemExit(f'필수 metrics 파일이 없습니다: {names}. 먼저 scratch_lab.py와 framework_lab.py를 실행하세요.')

def run() -> None:
    ensure()
    scratch = load(SCRATCH)
    framework = load(FRAMEWORK)
    report = f'''# 03 DeepSpeed ZeRO 실행 관측

## 관측 결과
- DP baseline MB: `{scratch['dp_baseline_mb']}`
- ZeRO-1 MB: `{scratch['zero_stage_1_mb']}`
- ZeRO-2 MB: `{scratch['zero_stage_2_mb']}`
- ZeRO-3 MB: `{scratch['zero_stage_3_mb']}`
- stage 3 saving ratio: `{scratch['stage_3_memory_saving_ratio']}`
- best memory stage: `{framework['best_memory_stage']}`

## 한국어 해석
- optimizer state를 먼저 나누고, 이후 gradient와 parameter까지 shard되며 per-rank memory가 줄어든다.
- 하지만 stage가 높을수록 communication과 checkpoint 복구 복잡도를 함께 봐야 한다.
'''
    OBSERVED.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED.write_text(report, encoding='utf-8')
    ANALYSIS.write_text(STABLE, encoding='utf-8')
    print(report)

if __name__ == '__main__':
    run()
