from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'
THEORY_BACKLINK = '[THEORY.md](./THEORY.md)'
SCRATCH_REQUIRED_KEYS = (
    'rnn_order_cosine_gap',
    'lstm_order_cosine_gap',
    'gru_order_cosine_gap',
    'rnn_long_range_signal',
    'lstm_long_range_signal',
    'gru_long_range_signal',
    'teacher_forcing_loss',
    'free_running_loss',
    'teacher_forcing_gap',
    'figure_path',
)
FRAMEWORK_REQUIRED_KEYS = (
    'device',
    'hidden_shapes',
    'rnn_order_cosine_gap',
    'lstm_order_cosine_gap',
    'gru_order_cosine_gap',
    'rnn_long_range_signal',
    'lstm_long_range_signal',
    'gru_long_range_signal',
    'teacher_forcing_loss',
    'free_running_loss',
    'teacher_forcing_gap',
    'decoder_logits_shape',
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding='utf-8'))


def _ensure_metrics_exist() -> None:
    missing = [path for path in (SCRATCH, FRAMEWORK) if not path.exists()]
    if not missing:
        return

    missing_list = ', '.join(str(path.relative_to(UNIT_ROOT)) for path in missing)
    raise SystemExit(
        '필수 metrics 파일이 없습니다: '
        f'{missing_list}. 먼저 scratch_lab.py와 framework_lab.py를 실행하세요.'
    )


def _ensure_required_keys(metrics: dict[str, object], *, name: str, required_keys: tuple[str, ...]) -> None:
    missing_keys = [key for key in required_keys if key not in metrics]
    if not missing_keys:
        return
    raise SystemExit(
        'metrics schema validation failed: '
        f'{name} metrics missing keys: {", ".join(missing_keys)}'
    )


def _ensure_stable_analysis_ready() -> None:
    if not ANALYSIS_PATH.exists():
        raise SystemExit('stable analysis.md가 없습니다. 먼저 추적된 분석 문서를 복구하세요.')
    stable_analysis = ANALYSIS_PATH.read_text(encoding='utf-8')
    if THEORY_BACKLINK not in stable_analysis:
        raise SystemExit('stable analysis.md에 THEORY 링크가 없습니다. 분석 기준 문서를 먼저 고치세요.')


def run() -> None:
    _ensure_metrics_exist()
    _ensure_stable_analysis_ready()

    scratch = _load_json(SCRATCH)
    framework = _load_json(FRAMEWORK)
    _ensure_required_keys(scratch, name='scratch', required_keys=SCRATCH_REQUIRED_KEYS)
    _ensure_required_keys(framework, name='framework', required_keys=FRAMEWORK_REQUIRED_KEYS)

    observed_report = f'''# 03 Sequence Models 실행 관측

## 관측 결과
- scratch RNN order cosine gap: `{scratch['rnn_order_cosine_gap']}`
- scratch LSTM order cosine gap: `{scratch['lstm_order_cosine_gap']}`
- scratch GRU order cosine gap: `{scratch['gru_order_cosine_gap']}`
- scratch long-range signal (RNN / LSTM / GRU): `{scratch['rnn_long_range_signal']}` / `{scratch['lstm_long_range_signal']}` / `{scratch['gru_long_range_signal']}`
- scratch teacher forcing gap: `{scratch['teacher_forcing_gap']}`
- scratch figure: `{scratch['figure_path']}`
- framework device: `{framework['device']}`
- framework hidden shapes: `{framework['hidden_shapes']}`
- framework long-range signal (RNN / LSTM / GRU): `{framework['rnn_long_range_signal']}` / `{framework['lstm_long_range_signal']}` / `{framework['gru_long_range_signal']}`
- framework teacher forcing gap: `{framework['teacher_forcing_gap']}`
- framework decoder logits shape: `{framework['decoder_logits_shape']}`

## 한국어 해석
- 순서를 뒤집은 두 시퀀스에서 cosine gap이 0이 아니라는 사실은, recurrent hidden state가 **bag-of-tokens가 아니라 순서 누적 요약**이라는 점을 바로 보여 준다.
- scratch와 framework 모두에서 `RNN < LSTM/GRU` 순서로 long-range signal이 남는 이유는, vanilla RNN은 같은 변환을 반복 곱하며 초반 정보를 빠르게 희석시키는 반면 gate가 있는 셀은 유지 비율을 따로 배울 수 있기 때문이다.
- scratch의 `lstm_forget_gate_mean` / `gru_update_gate_mean`이 높게 남는 것은 "지금은 덮어쓰지 말고 유지하자" 쪽으로 gate가 기울었다는 직관적 신호다.
- framework decoder의 teacher forcing gap `{framework['teacher_forcing_gap']}` 은 학습 중에는 정답 이전 토큰을 보며 안정적으로 맞췄지만, 추론에서는 자기 예측을 다시 입력으로 넣으면서 오류가 누적될 수 있음을 보여 준다.
- 즉 이 단위의 핵심은 **순서 민감성 → hidden bottleneck → gating 보강 → teacher forcing mismatch** 를 하나의 흐름으로 묶어, 다음 attention 단위에서 왜 과거 위치를 직접 참조하려 하는지 준비하는 데 있다.

## 다음 실험 메모
- filler step 수를 늘리면 vanilla RNN signal이 얼마나 더 빨리 약해지는지 다시 본다.
- decoder 학습 epoch를 늘리거나 줄여 teacher forcing gap이 얼마나 변하는지 관찰한다.
- 이후 attention unit에서는 "같은 문제를 hidden state 하나가 아니라 직접 위치 참조로 풀면 무엇이 달라지는가"를 이어서 읽는다.

## 이론 다시 연결하기
- 핵심 개념 복습: [THEORY.md](../../THEORY.md)
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
