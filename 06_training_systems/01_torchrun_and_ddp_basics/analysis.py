from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
OBSERVED = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'
ANALYSIS = UNIT_ROOT / 'analysis.md'

STABLE = '''# 01 Torchrun and DDP Basics 분석

## 해석 프레임
- DDP는 model parallel이 아니라 같은 모델 복사본들의 gradient를 평균내는 data parallel 방식이다.
- `rank`, `local_rank`, `world_size`는 코드가 여러 프로세스에서 실행될 때 각 프로세스의 위치를 설명한다.
- 이 단위의 toy metrics는 실제 multi-GPU 통신이 아니라 all-reduce의 산술 의미를 먼저 보여준다.

## 확인 질문
- rank별 gradient가 다른 이유는 무엇인가?
- averaged gradient가 parameter update에 어떻게 반영되는가?
- local rank는 device assignment와 어떻게 연결되는가?
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
    report = f'''# 01 Torchrun and DDP Basics 실행 관측

## 관측 결과
- world_size: `{scratch['world_size']}`
- local_world_size: `{scratch['local_world_size']}`
- averaged_gradient: `{scratch['averaged_gradient']}`
- max_gradient_deviation: `{scratch['max_gradient_deviation']}`
- parameter_after: `{framework['parameter_after']}`

## 한국어 해석
- rank별 gradient는 서로 다르지만, DDP-style 평균 이후 모든 rank는 같은 update를 적용한다고 읽는다.
- `rank_to_local_rank`는 전체 프로세스 번호와 한 노드 안 위치가 다르다는 점을 보여 준다.
- 이 toy run은 실제 통신을 수행하지 않지만, all-reduce mean이 어떤 숫자 계약을 만드는지는 분명하게 보여 준다.
'''
    OBSERVED.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED.write_text(report, encoding='utf-8')
    ANALYSIS.write_text(STABLE, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    run()
