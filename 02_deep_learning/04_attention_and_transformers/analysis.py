from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 04 Attention and Transformers 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 attention과 transformer를 해석하는 **안정적인 프레임**만 남겨, 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- attention row 합이 1이라는 것은, 각 query 출력이 value들의 **가중합**이라는 뜻이다. 그래서 attention은 sequence mixing으로 읽는 것이 가장 자연스럽다.
- multi-head의 핵심은 head 수 자체가 아니라, **서로 다른 mixing 관점**이 병렬로 놓인다는 점이다.
- encoder self-attention은 전체 문맥을 볼 수 있지만, decoder self-attention은 causal mask 때문에 미래를 보지 못한다. encoder-decoder 구조는 여기에 cross-attention이 더해져 입력 memory를 읽는다.
- transformer는 recurrent hidden chain 대신 직접 참조를 허용해 long-range information path를 줄이지만, attention matrix 길이 비용은 여전히 남는다.

## 확인 질문
- attention output을 왜 “토큰 선택”이 아니라 value mixing 결과라고 말할 수 있는가?
- multi-head에서 서로 다른 top key가 나온다면, 그것은 어떤 관점 차이를 시사하는가?
- encoder와 decoder의 가장 큰 차이를 mask 관점에서 어떻게 설명할 수 있는가?
- recurrent bottleneck relief와 길이 제곱 비용 trade-off를 동시에 어떻게 요약할 수 있는가?

## 관련 이론
- [THEORY.md](./THEORY.md): sequence mixing, multi-head intuition, encoder/decoder distinction, recurrent bottleneck relief를 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
'''


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


def run() -> None:
    _ensure_metrics_exist()
    scratch = _load_json(SCRATCH)
    framework = _load_json(FRAMEWORK)

    focus_query = scratch['encoder_decoder']['focus_query_token']
    top_key_sets = scratch['multi_head']['per_query_top_keys']
    focus_top_keys = next(
        (item['top_keys'] for item in top_key_sets if item['query_token'] == focus_query),
        [],
    )

    observed_report = f'''# 04 Attention and Transformers 실행 관측

## 관측 결과
- 시퀀스 길이: `{scratch.get("sequence_length", 0)}`
- scratch max row-sum error: `{scratch.get("max_row_sum_error", 0.0)}`
- scratch head 수: `{scratch.get("multi_head", {}).get("head_count", 0)}`
- encoder 미래 접근 질량: `{scratch.get("encoder_decoder", {}).get("encoder_future_access_mass", 0.0)}`
- decoder 미래 접근 질량: `{scratch.get("encoder_decoder", {}).get("decoder_future_access_mass", 0.0)}`
- framework device: `{framework.get("device", "unknown")}`
- encoder hidden shape: `{framework.get("encoder_hidden_shape", [])}`
- decoder hidden shape: `{framework.get("decoder_hidden_shape", [])}`
- decoder block output shape: `{framework.get("decoder_block_output_shape", [])}`
- decoder future attention max: `{framework.get("decoder_future_attention_max", 0.0)}`
- encoder future attention mean: `{framework.get("encoder_future_attention_mean", 0.0)}`
- per-head difference mean: `{framework.get("per_head_difference_mean", 0.0)}`

## 한국어 해석
- scratch 실험에서 row 합 오차가 `{scratch.get("max_row_sum_error", 0.0)}` 라는 것은 attention output을 value들의 mixing 비율로 읽어도 무리가 없다는 뜻이다.
- focus query `{focus_query}` 는 head마다 `{focus_top_keys}` 같은 top key를 보여, multi-head가 서로 다른 mixing 관점을 만들 수 있음을 드러낸다.
- encoder 규칙에서는 미래 위치 질량이 `{scratch.get("encoder_decoder", {}).get("encoder_future_access_mass", 0.0)}` 남아 있지만, decoder 규칙에서는 `{scratch.get("encoder_decoder", {}).get("decoder_future_access_mass", 0.0)}` 로 막혔다. 즉 causal mask가 “미래를 못 보게 하는 decoder”의 핵심 규칙이다.
- framework 실험에서 encoder / decoder / decoder block shape가 각각 `{framework.get("encoder_hidden_shape", [])}`, `{framework.get("decoder_hidden_shape", [])}`, `{framework.get("decoder_block_output_shape", [])}` 로 유지되었다. transformer block은 shape를 유지한 채 표현만 갱신한다.
- recurrent path는 `{framework.get("recurrent_relief", {}).get("recurrent_steps", 0)}` step을 따라가야 하지만, attention mixing은 `{framework.get("recurrent_relief", {}).get("attention_parallel_rounds", 0)}` round에 관찰된다. 이 차이가 recurrent bottleneck relief의 직관이다.
- cross-attention 사용 여부가 `{framework.get("cross_attention_used", False)}` 인 것은 encoder-decoder 계열이 입력 memory를 따로 읽는다는 사실을 toy block에서 보여 준다.

## 이론 다시 연결하기
- 안정적인 해석 프레임은 `analysis.md`에 유지한다.
- 핵심 개념 복습: [THEORY.md](./THEORY.md)
'''

    OBSERVED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OBSERVED_REPORT.write_text(observed_report, encoding='utf-8')
    ANALYSIS_PATH.write_text(STABLE_ANALYSIS, encoding='utf-8')
    print(observed_report)


if __name__ == '__main__':
    run()
