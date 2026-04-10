from __future__ import annotations

import json
from pathlib import Path
from typing import Any

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 01 Tokenization and Embeddings 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 숫자가 조금 바뀌어도 유지되는 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- tokenizer는 문자열을 모델 vocabulary 안의 단위로 바꾸는 단계다. whitespace 단어 수와 subword token 수는 다를 수 있다.
- token id는 lookup key일 뿐이며, embedding table을 통과한 뒤에야 `(batch, seq, dim)` 표현이 생긴다.
- `[PAD]`는 길이 맞춤용 토큰이라서 실제 내용이 아니다. pooling이나 attention 전에는 반드시 mask로 구분해야 한다.
- `[UNK]`가 늘수록 표면 문자열 정보가 한 덩어리로 뭉개져, 모델이 세밀한 차이를 읽기 어려워진다.

## 확인 질문
- 왜 같은 한국어 문장도 whitespace 기준보다 subword 기준에서 더 긴 sequence가 될 수 있는가?
- `[UNK]`가 등장한 위치는 어떤 종류의 정보 손실을 뜻하는가?
- embedding lookup 이후 shape가 어떻게 바뀌고, padding mask가 없으면 어떤 계산이 왜곡되는가?

## 관련 이론
- [THEORY.md](./THEORY.md): tokenization, subword, embedding lookup, padding mask 핵심 개념을 다시 확인한다.
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


def _pick_first_example(examples: list[dict[str, Any]]) -> tuple[str, str, str]:
    if not examples:
        return (
            '실행 예문이 비어 있었다.',
            '이번 실행에서는 분석할 첫 예문이 없어, 토큰 길이 비교를 일반 원칙으로만 설명한다.',
            '빈 예문 목록에서도 tokenizer는 보통 whitespace 단위보다 더 세밀한 subword budget을 만든다.',
        )

    first_example = examples[0]
    sentence = str(first_example.get('sentence', '이름 없는 예문'))
    word_count = int(first_example.get('word_count', 0))
    subword_count = int(first_example.get('subword_count', 0))
    return (
        sentence,
        f'첫 번째 예문 `{sentence}` 은 whitespace로는 `{word_count}`개 어절이지만, 모델이 읽는 subword는 `{subword_count}`개였다.',
        '즉 tokenizer는 사람이 띄어쓴 단어 수와 다른 길이 budget을 만든다.',
    )


def _pick_unknown_interpretation(examples: list[dict[str, Any]], unknown_total: int) -> str:
    for example in examples:
        if int(example.get('unknown_token_count', 0)) > 0:
            sentence = str(example.get('sentence', '이 예문'))
            return (
                f'- `{sentence}` 에서는 vocabulary에 없는 부분이 `[UNK]`로 떨어졌다. '
                '이 경우 세부 표면형 정보가 한 토큰으로 뭉개져, 희귀 표현을 구분하기가 어려워진다.'
            )

    if unknown_total > 0:
        return (
            '- `[UNK]` 총량은 보고되었지만 어떤 예문에서 생겼는지 세부 정보가 비어 있었다. '
            '그래도 의미는 같다. vocabulary 밖 표현이 한 덩어리로 압축되면 표면형 정보가 줄어든다.'
        )

    return (
        '- 이번 실행에서는 `[UNK]`가 관측되지 않았다. 즉 toy vocab이 현재 예문을 모두 덮었거나, '
        '희귀 표현이 입력에 없었다는 뜻이다. 그래도 실제 NLP 파이프라인에서는 OOV 사례를 계속 경계해야 한다.'
    )


def run() -> None:
    _ensure_metrics_exist()
    scratch = _load_json(SCRATCH)
    framework = _load_json(FRAMEWORK)

    raw_examples = scratch.get('examples', [])
    examples = [example for example in raw_examples if isinstance(example, dict)]
    unknown_total = int(scratch.get('unknown_token_total', 0))

    first_sentence, first_example_line, first_example_reason = _pick_first_example(examples)
    unknown_line = _pick_unknown_interpretation(examples, unknown_total)
    non_pad_counts = framework.get('non_pad_counts', [])

    observed_report = f'''# 01 Tokenization and Embeddings 실행 관측

## 관측 결과
- vocab size: `{scratch.get("vocab_size", 0)}`
- 총 whitespace 단어 수: `{scratch.get("total_word_count", 0)}`
- 총 subword 수: `{scratch.get("total_subword_count", 0)}`
- subword expansion ratio: `{scratch.get("subword_expansion_ratio", 0.0)}`
- `[UNK]` 총 개수: `{unknown_total}`
- input ids shape: `{framework.get("input_ids_shape", [])}`
- embedded shape: `{framework.get("embedded_shape", [])}`
- padding mask shape: `{framework.get("padding_mask_shape", [])}`
- pooled shape: `{framework.get("pooled_shape", [])}`

## 한국어 해석
- {first_example_line} {first_example_reason}
{unknown_line}
- PyTorch embedding lookup은 `(batch, seq) = {framework.get("input_ids_shape", [])}` 정수 텐서를 `(batch, seq, dim) = {framework.get("embedded_shape", [])}` 실수 텐서로 바꿨다. 여기서 비로소 각 token 위치에 dense vector가 놓인다.
- padding mask shape가 `{framework.get("padding_mask_shape", [])}` 인 이유는 각 배치의 각 위치가 실제 토큰인지 `[PAD]`인지 표시해야 하기 때문이다.
- non-pad counts가 `{non_pad_counts}` 로 다르다는 것은, 길이가 다른 문장을 같은 batch로 묶을 때 mask 없이는 평균이나 attention 계산이 섞여 버린다는 뜻이다.

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
