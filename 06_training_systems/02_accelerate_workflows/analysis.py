from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
OBSERVED = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'
ANALYSIS = UNIT_ROOT / 'analysis.md'
STABLE = '''# 02 Accelerate Workflows 분석

## 해석 프레임
- Accelerate는 학습 루프를 대체하지 않고 실행 환경 적응을 단순화한다.
- `prepare()`가 감추는 wrapper와 여전히 남는 backend complexity를 분리해 읽는다.
- distributed_type과 num_processes는 편의 설정이 아니라 실행 계약이다.
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
    report = f'''# 02 Accelerate Workflows 실행 관측

## 관측 결과
- distributed_type: `{scratch['distributed_type']}`
- num_processes: `{scratch['num_processes']}`
- mixed_precision: `{scratch['mixed_precision']}`
- prepared_object_count: `{framework['prepared_object_count']}`
- remaining_complexity_count: `{framework['remaining_complexity_count']}`

## 한국어 해석
- Accelerate는 device/backward/launcher boilerplate 일부를 줄인다.
- 하지만 effective batch, checkpointing, metric gathering 같은 해석 책임은 남는다.
- 따라서 이 단위는 편의 API 암기가 아니라 abstraction boundary 읽기가 핵심이다.
'''
    OBSERVED.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED.write_text(report, encoding='utf-8')
    ANALYSIS.write_text(STABLE, encoding='utf-8')
    print(report)

if __name__ == '__main__':
    run()
