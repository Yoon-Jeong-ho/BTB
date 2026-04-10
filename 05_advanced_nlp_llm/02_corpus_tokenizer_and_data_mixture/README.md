# 02 Corpus, Tokenizer, and Data Mixture

> Status: runnable
> 이 단위는 CPU에서 바로 실행되는 toy corpus/tokenizer 통계 실습이다. 실제 LLM 학습은 하지 않고, corpus quality, dedup/contamination, tokenizer tradeoff, multilingual mixture, token budget 직관만 결정적으로 관찰한다.

## 왜 이 단위를 배우는가
LLM pretraining은 objective만으로 결정되지 않는다. 모델이 보는 실제 신호는 **corpus를 고르고**, **tokenizer로 자르고**, **도메인/언어 slice를 어떤 비율로 섞는지**에 따라 달라진다. 이 단위는 큰 모델을 학습하지 않고도 데이터 준비 단계가 모델 행동과 비용을 어떻게 바꾸는지 숫자로 확인하게 만든다.

## 이번 단위에서 남길 것
- `scratch_lab.py`: toy corpus를 직접 순회하며 중복, contamination, tokenizer별 token 수, mixture token share, SVG를 만든다.
- `framework_lab.py`: 가벼운 tokenizer/statistics 클래스로 같은 문제를 pipeline처럼 재구성한다.
- `analysis.py`: 두 metrics가 없으면 명확히 실패하고, 있으면 실행별 관측 보고서를 만든다.
- `analysis.md`: 반복 실행해도 안정적으로 유지되는 해석 프레임이다.
- `reflection.md`: 한국어 우선 회고 질문이다.

## 실행 방법
```bash
python 05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture/scratch_lab.py
python 05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture/framework_lab.py
python 05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture/analysis.py
```

생성물은 다음 위치에 남는다.
- `artifacts/scratch-manual/metrics.json`
- `artifacts/scratch-manual/corpus_quality_overview.svg`
- `artifacts/framework-manual/metrics.json`
- `artifacts/analysis-manual/latest_report.md`

## 실행 결과 예시
아래 숫자는 이 단위의 deterministic toy corpus에서 실제로 재현되는 관측 형태다.

```text
$ python 05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture/scratch_lab.py
{
  "raw_document_count": 11,
  "dedup_removed_documents": 2,
  "contamination_blocked_documents": 2,
  "trainable_document_count": 7,
  "figure_path": "artifacts/scratch-manual/corpus_quality_overview.svg",
  "token_budget_demo": {
    "context_window": 64,
    "aggressive_token_inflation_vs_whitespace": 1.867
  }
}

$ python 05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture/framework_lab.py
{
  "device": "cpu",
  "tokenizer_name": "toy_unigram_like",
  "removed_exact_duplicates": 1,
  "removed_near_duplicates": 1,
  "contamination_blocked": 2,
  "batch_preview": ["news", "docs", "academic", "code_docs"],
  "token_budget": {"context_window": 64}
}

$ python 05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture/analysis.py
# 02 Corpus, Tokenizer, and Data Mixture 실행 관측
...
```

## 관찰 포인트
1. **corpus quality**: raw 문서 수가 그대로 trainable 신호가 되지 않는다. exact duplicate, near duplicate, benchmark contamination이 빠지면 실제 token budget이 줄어든다.
2. **tokenizer tradeoff**: `toy_unigram_like`는 압축률이 좋고, `toy_aggressive_subword`는 다국어/전문용어를 더 잘게 쪼개 sequence length를 늘린다.
3. **dedup vs contamination**: dedup은 train 내부 반복을 줄이고, contamination은 평가 overlap을 막는다. 둘은 목적이 다르다.
4. **domain balance**: 문서 수가 아니라 token share를 보면 news, docs, academic, dialogue, code_docs가 실제로 얼마나 budget을 쓰는지 드러난다.
5. **multilingual mixture**: 한국어/영어/일본어가 같은 문서 수로 들어가도 tokenizer fragmentation 때문에 token share는 달라진다.
6. **token-budget intuition**: context window 64라는 작은 예산에서도 tokenizer가 잘게 자르면 한 번에 담을 수 있는 문서 수가 줄어든다.

## 다음 단위와의 연결
다음 단위 `05_advanced_nlp_llm/03_domain_adaptive_pretraining`에서는 특정 도메인 데이터를 추가로 먹일 때 어떤 개선과 망각 위험이 생기는지 본다. 이 단위에서 만든 corpus/tokenizer/mixture 관점은 DAPT 전에 **무엇을 얼마나 어떤 token 형태로 더 넣을지** 결정하는 기준이 된다.
