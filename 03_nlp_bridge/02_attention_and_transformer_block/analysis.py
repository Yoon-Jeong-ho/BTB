from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 02 Attention and Transformer Block 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 attention과 transformer block을 읽는 안정적인 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- attention weight의 각 row 합이 1이라는 것은, query 위치 출력이 value들의 가중합이라는 뜻이다.
- self-attention은 각 토큰 표현을 다른 토큰 정보와 섞어 새 hidden state를 만든다. 그래서 output은 "원래 토큰 하나"가 아니라 sequence mixing 결과다.
- padding mask는 `[PAD]` 열을 가려서 빈 토큰을 참고하지 못하게 하고, causal mask는 미래 열을 가려서 아직 보이면 안 되는 정보를 차단한다.
- transformer block은 residual connection과 feed-forward를 거치면서 shape는 `(batch, seq, dim)`으로 유지하지만, 내부 좌표는 계속 갱신된다.

## 확인 질문
- attention output을 value들의 가중합이라고 말할 수 있는 근거는 무엇인가?
- padding mask와 causal mask는 각각 어떤 종류의 잘못된 정보 유입을 막는가?
- transformer block이 shape를 유지한다는 사실과, 표현이 달라진다는 사실은 어떻게 동시에 성립하는가?

## 관련 이론
- [THEORY.md](./THEORY.md): self-attention, mask, transformer block 핵심 개념을 다시 확인한다.
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

    strongest_links = scratch.get('strongest_links', [])
    first_link = strongest_links[0] if strongest_links else {}
    focus_query = scratch.get('focus_query_token', '이 query')
    focus_weights = scratch.get('focus_query_weights', {})
    focus_top_key = max(focus_weights, key=focus_weights.get) if isinstance(focus_weights, dict) and focus_weights else '알 수 없는 key'

    observed_report = f'''# 02 Attention and Transformer Block 실행 관측

## 관측 결과
- 시퀀스 길이: `{scratch.get("sequence_length", 0)}`
- scratch hidden dim: `{scratch.get("hidden_dim", 0)}`
- framework input ids shape: `{framework.get("input_ids_shape", [])}`
- embedded shape: `{framework.get("embedded_shape", [])}`
- attention output shape: `{framework.get("attention_output_shape", [])}`
- attention weights shape: `{framework.get("attention_weights_shape", [])}`
- transformer block output shape: `{framework.get("transformer_block_output_shape", [])}`
- pad key attention max: `{framework.get("pad_key_attention_max", 0.0)}`
- future attention max: `{framework.get("future_attention_max", 0.0)}`

## 한국어 해석
- 첫 번째 query 토큰 `{first_link.get("query_token", "알 수 없음")}` 은 `{first_link.get("top_key_token", "알 수 없음")}` 쪽으로 가장 큰 weight `{first_link.get("top_weight", 0.0)}` 를 줬다. 즉 attention은 "누구를 참고할지"를 row별로 정한다.
- `{focus_query}` query의 최고 weight가 `{focus_top_key}` 로 모였다는 것은, 그 위치 출력이 해당 key/value 정보와 강하게 섞였다는 뜻이다.
- scratch 실험에서 row 합이 `{scratch.get("row_sums", [])}` 로 1에 가깝게 유지된 것은 attention output이 value들의 확률적 가중합처럼 해석될 수 있음을 보여 준다.
- framework 실험에서 `(batch, seq, dim)` = `{framework.get("embedded_shape", [])}` 가 attention 뒤에도 `{framework.get("attention_output_shape", [])}` 로 유지됐다. transformer block 뒤 shape도 `{framework.get("transformer_block_output_shape", [])}` 이라서, block은 길이와 차원은 유지한 채 표현만 업데이트한다.
- `pad_key_attention_max = {framework.get("pad_key_attention_max", 0.0)}` 와 `future_attention_max = {framework.get("future_attention_max", 0.0)}` 는 mask가 빈 토큰과 미래 토큰을 거의 보지 않도록 막았다는 신호다.

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
