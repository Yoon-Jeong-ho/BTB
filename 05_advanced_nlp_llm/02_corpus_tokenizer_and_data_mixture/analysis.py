from __future__ import annotations

import json
from pathlib import Path

UNIT_ROOT = Path(__file__).resolve().parent
SCRATCH = UNIT_ROOT / 'artifacts' / 'scratch-manual' / 'metrics.json'
FRAMEWORK = UNIT_ROOT / 'artifacts' / 'framework-manual' / 'metrics.json'
ANALYSIS_PATH = UNIT_ROOT / 'analysis.md'
OBSERVED_REPORT = UNIT_ROOT / 'artifacts' / 'analysis-manual' / 'latest_report.md'

STABLE_ANALYSIS = '''# 02 Corpus, Tokenizer, and Data Mixture 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 toy corpus 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 corpus quality, tokenizer tradeoff, dedup/contamination, mixture budget을 해석하는 **안정적인 프레임**만 남긴다.
- 따라서 `analysis.py`를 반복 실행해도 이 파일은 같은 내용으로 유지되고, 관측 보고서만 최신 값으로 갱신된다.

## 해석 프레임
- corpus quality는 문서 수가 아니라 **사용 가능한 학습 신호**다. exact/near duplicate와 contamination을 제거하면 raw 문서보다 trainable 문서가 줄어든다.
- tokenizer는 compression과 fairness를 동시에 바꾼다. toy aggressive subword가 더 많은 token을 만들면 같은 context window에서 담을 수 있는 의미 단위가 줄어든다.
- dedup은 train corpus 내부 반복 노출을 줄이고, contamination check는 evaluation overlap 때문에 성능 추정이 부풀려지는 것을 막는다. 둘은 함께 필요하지만 같은 검사가 아니다.
- mixture는 문서 수보다 token budget 기준으로 읽어야 한다. 언어/도메인별 평균 token 길이가 다르면 같은 문서 수라도 gradient 신호 배분이 달라진다.

## 확인 질문
- raw corpus에서 실제 trainable corpus까지 어떤 문서가 빠졌고, 그 이유는 dedup인가 contamination인가?
- tokenizer를 더 잘게 만들면 multilingual mixture에서 어떤 언어가 더 많은 token budget을 쓰게 되는가?
- domain balance를 문서 수가 아니라 token share로 보면 어떤 slice가 과대표/과소대표되는가?
- 작은 고품질 도메인 slice를 oversample할 때 contamination guard를 어디에 둬야 하는가?

## 관련 이론
- [THEORY.md](./THEORY.md): corpus quality, tokenizer tradeoff, dedup/contamination, multilingual mixture, token budget을 다시 확인한다.
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

    scratch_tokenizers = scratch.get('tokenizers', {})
    framework_tokenizers = framework.get('tokenizer_stats', {})
    observed_report = f'''# 02 Corpus, Tokenizer, and Data Mixture 실행 관측

## 관측 결과
- raw document count: `{scratch.get('raw_document_count', 0)}`
- trainable document count: `{scratch.get('trainable_document_count', 0)}`
- scratch dedup removed documents: `{scratch.get('dedup_removed_documents', 0)}`
- scratch contamination blocked documents: `{scratch.get('contamination_blocked_documents', 0)}`
- framework device: `{framework.get('device', 'unknown')}`
- framework accepted document count: `{framework.get('accepted_document_count', 0)}`
- framework removed exact duplicates: `{framework.get('removed_exact_duplicates', 0)}`
- framework removed near duplicates: `{framework.get('removed_near_duplicates', 0)}`
- framework contamination blocked: `{framework.get('contamination_blocked', 0)}`
- language token share: `{framework.get('language_token_share', {})}`
- domain token share: `{framework.get('domain_token_share', {})}`
- context window: `{framework.get('token_budget', {}).get('context_window', 0)}`

## 한국어 해석
- raw 문서 `{scratch.get('raw_document_count', 0)}`개 중 trainable 문서는 `{scratch.get('trainable_document_count', 0)}`개다. 줄어든 문서는 내부 중복 제거와 평가 오염 차단 때문에 빠졌다.
- scratch toy tokenizer에서 whitespace 평균 token 수는 `{scratch_tokenizers.get('toy_whitespace', {}).get('avg_tokens_per_doc', 0)}`이고 aggressive subword 평균은 `{scratch_tokenizers.get('toy_aggressive_subword', {}).get('avg_tokens_per_doc', 0)}`이다. 더 잘게 쪼개면 같은 문서가 더 많은 context budget을 먹는다.
- framework 관측의 언어별 token share `{framework.get('language_token_share', {})}`는 문서 수 균형만으로 multilingual mixture를 판단하면 안 된다는 점을 보여 준다.
- domain token share `{framework.get('domain_token_share', {})}`는 작은 domain slice도 tokenizer와 문서 길이에 따라 실제 token budget에서 다른 무게를 가질 수 있음을 보여 준다.
- unigram-like 평균 token 수 `{framework_tokenizers.get('toy_unigram_like', {}).get('avg_tokens_per_doc', 0)}`와 aggressive 평균 token 수 `{framework_tokenizers.get('toy_aggressive_subword', {}).get('avg_tokens_per_doc', 0)}`의 차이는 tokenizer tradeoff를 sequence length 직관으로 연결한다.

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
