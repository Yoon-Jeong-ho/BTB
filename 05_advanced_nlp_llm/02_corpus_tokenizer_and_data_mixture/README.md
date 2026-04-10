# 02 Corpus, Tokenizer, and Data Mixture

> Status: outlined
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 모습** 이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
LLM은 objective만으로 만들어지지 않는다. **무슨 문서를 넣었는가**, **그 문서를 어떤 단위로 자르는가**, **도메인/언어를 어떤 비율로 섞는가** 가 실제 학습 신호의 질과 비용을 함께 바꾼다. 같은 pretraining objective라도 corpus 품질이 낮거나 tokenizer가 특정 언어를 과도하게 잘게 쪼개면, 모델은 더 많은 step을 써도 원하는 분포 감각을 얻지 못할 수 있다. 이 단위는 data pipeline을 단순 수집 단계가 아니라 **모델 행동을 설계하는 일부** 로 읽게 만드는 출발점이다.

## 이번 단위에서 남길 것
- 학습 목표와 후속 실습 방향을 정리한 `README.md`
- corpus quality, tokenizer trade-off, dedup/contamination, multilingual mixture를 묶은 `THEORY.md`
- 선행 개념 체크리스트 `PREREQS.md`
- outlined 단계 메타데이터를 담은 `lesson.yaml`
- 후속 실습 산출물이 들어갈 자리만 먼저 만든 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`에 대한 명시적 빈자리

## 실습 흐름
1. corpus를 언어, 도메인, 문서 길이, boilerplate 비율, 라이선스/출처 메타데이터 관점으로 나눠 보며 "크기"와 "쓸모 있는 신호"를 구분한다.
2. exact duplicate, near-duplicate, benchmark overlap, split leakage를 한 프레임에서 보면서 dedup과 contamination check가 왜 별도 단계로 필요한지 정리한다.
3. 같은 한국어·영어 혼합 문서 묶음을 여러 tokenizer 설정으로 바라보며 vocabulary size, subword granularity, sequence length 증가율이 어떻게 달라지는지 비교한다.
4. tokenizer choice가 문맥 길이 예산, 희귀 전문용어 표현, 다국어 공정성에 어떤 trade-off를 남기는지 observation point 중심으로 적는다.
5. 일반 웹 문서, 전문 도메인 문서, 대화 데이터, 코드/문서형 자료를 어떤 기준으로 혼합할지 고민하며 mixture ratio를 문서 수가 아니라 **token budget과 품질 가중치** 관점으로 읽는다.
6. 마지막에는 "같은 base LM이라도 특정 도메인 비중을 더 올리면 무엇이 바뀌는가?"를 질문으로 남기며 `05_advanced_nlp_llm/03_domain_adaptive_pretraining`으로 넘어간다.

## 이 단위에서 특히 볼 질문
- corpus size가 커지기만 하면 좋은가, 아니면 품질/중복/노이즈가 더 먼저 병목이 되는가?
- dedup을 강하게 걸수록 항상 좋은가, 아니면 반복적으로 등장해야 하는 합법적 패턴까지 잃을 수 있는가?
- tokenizer vocabulary를 크게 잡으면 무엇이 좋아지고, 무엇이 더 불안정해질 수 있는가?
- 한국어처럼 형태 변화가 잦은 언어와 영어를 함께 다룰 때 shared tokenizer는 어떤 편향을 만들 수 있는가?
- mixture 비율은 문서 수, sample 수, token 수, training step 중 무엇을 기준으로 읽어야 하는가?
- contamination은 exact match만 지우면 끝나는가, 아니면 near-duplicate / benchmark paraphrase / split leakage까지 봐야 하는가?

## 실행 결과 예시
아래는 **아직 완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture/scratch_lab.py
{
  "corpus_stats": {
    "documents": 120000,
    "language_share": {"ko": 0.56, "en": 0.31, "ja": 0.08, "other": 0.05},
    "domain_share": {"general_web": 0.42, "news": 0.18, "academic": 0.14, "forum": 0.11, "code_docs": 0.15},
    "near_duplicate_rate": 0.07,
    "benchmark_overlap_hits": 3
  },
  "tokenizer_comparison": [
    {"name": "bpe_32k", "avg_chars_per_token": 2.9, "avg_tokens_per_doc": 312},
    {"name": "unigram_32k", "avg_chars_per_token": 3.2, "avg_tokens_per_doc": 287}
  ],
  "mixture_plan": {
    "general_web_ko": 0.30,
    "general_web_en": 0.18,
    "domain_academic": 0.22,
    "dialogue": 0.15,
    "code_docs": 0.15
  }
}

$ python 05_advanced_nlp_llm/02_corpus_tokenizer_and_data_mixture/framework_lab.py
{
  "batch_shape": {
    "input_ids": [8, 2048],
    "attention_mask": [8, 2048],
    "domain_ids": [8]
  },
  "tokenizer": "sentencepiece_unigram_32k",
  "batch_mixture_share": {"ko_general": 0.50, "en_general": 0.25, "academic": 0.15, "code_docs": 0.10},
  "dropped_due_to_dedup": 2,
  "contamination_flags": [0, 0, 1, 0, 0, 0, 0, 0]
}
```

핵심은 숫자 자체보다도 **중복/오염 통계**, **tokenizer별 sequence length 차이**, **mixture 비율이 실제 batch 구성과 token budget에 어떻게 반영되는지** 를 읽는 것이다.

## 다음 단위와의 연결
이 단위에서 corpus·tokenizer·mixture 설계를 먼저 분리해서 보면, 다음 단위 `05_advanced_nlp_llm/03_domain_adaptive_pretraining`에서 "base model에 특정 도메인 데이터를 계속 더 먹이면 왜 좋아지거나 망가질 수 있는가"를 더 선명하게 읽을 수 있다. 다시 말해, 이 단위는 DAPT를 시작하기 전에 **무엇을 얼마나 어떤 형태로 더 넣을 것인가** 를 설계하는 데이터 관점의 준비 단계다.
