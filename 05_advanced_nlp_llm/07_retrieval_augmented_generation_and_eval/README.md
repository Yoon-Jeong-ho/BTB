# 07 Retrieval-Augmented Generation and Eval

> Status: runnable
>
> 이 단위는 CPU에서 바로 실행되는 deterministic toy RAG/eval 실습이다. 실제 LLM, 외부 vector DB, 네트워크 호출 없이 retriever-reader / retriever-generator split, retrieval grounding, context injection, citation/evidence expectation, failure mode, eval harness metrics를 작은 데이터로 관찰한다.

## 왜 이 단위를 배우는가
LLM이 모든 사실을 파라미터 안에 완벽히 기억한다고 가정하면 최신 문서, 사내 정책, 긴 근거 문맥, 출처 추적이 필요한 질문에서 곧 한계가 드러난다. retrieval-augmented generation(RAG)은 단순히 “벡터 검색을 붙이는 기능”이 아니라 **무엇을 찾고, 어떻게 문맥으로 주입하고, 어떤 evidence에 근거해 답했다는 기대를 어떻게 평가할 것인가**를 설계하는 문제다.

이 runnable 단위에서는 retriever-reader가 evidence span을 읽거나 abstain하는 방식과 retriever-generator가 retrieved context를 조건으로 더 자유롭게 합성하는 방식을 비교한다. 또한 citation이 붙었다는 사실과 retrieval grounding이 확보됐다는 사실이 왜 다른지, unsupported claim과 stale source를 어떻게 failure mode로 기록하는지, offline retrieval metrics와 online product metrics를 왜 나눠야 하는지까지 확인한다.

## 이번 단위에서 남길 것
- `scratch_lab.py`: 손으로 만든 toy corpus, top-k retrieval, claim-level grounding eval, `rag_grounding_eval.svg` 생성
- `framework_lab.py`: deterministic lightweight retrieval/generation simulation으로 batch shape, retrieval metrics, answer metrics, online proxy 기록
- `analysis.py`: metrics가 없으면 actionable Korean error로 실패하고, 실행별 관측은 `artifacts/analysis-manual/latest_report.md`에 기록
- `analysis.md`: 실행할 때마다 안정적으로 유지되는 해석 프레임
- `reflection.md`: Korean-first learner prompts

## 실습 흐름
1. closed-book 답변 대신 외부 evidence를 찾아야 하는 질문을 고른다.
2. query → chunk → top-k retrieval 흐름에서 relevant chunk가 상위에 들어오는지 본다.
3. 같은 retrieved context를 retriever-reader와 retriever-generator 관점으로 나눠 읽는다.
4. context injection 단계에서 metadata, source freshness, citation tag, prompt order가 답변을 어떻게 제한하는지 확인한다.
5. citation이 있는 답변도 주요 claim이 실제 evidence로 support되는지 unsupported claim count로 평가한다.
6. eval harness를 retrieval metrics(recall@k, MRR, nDCG), answer metrics(groundedness, citation precision, unsupported claim rate), online proxy metrics(acceptance, correction, citation click)로 분리한다.

## 실행 방법
```bash
python 05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/scratch_lab.py
python 05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/framework_lab.py
python 05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/analysis.py
```

## 실행 결과 예시
아래 예시는 이 저장소의 deterministic toy data로 실제 생성되는 metrics 구조를 축약한 것이다.

```text
$ python 05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/scratch_lab.py
{
  "retrieval_metrics": {"recall_at_1": 1.0, "recall_at_3": 1.0, "mrr": 1.0},
  "grounding_eval": {
    "groundedness": 0.875,
    "citation_precision": 1.0,
    "unsupported_claim_rate": 0.125,
    "grounding_expectation": "claim-level evidence, not citation count"
  },
  "figure_path": "artifacts/scratch-manual/rag_grounding_eval.svg"
}

$ python 05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/framework_lab.py
{
  "device": "cpu",
  "simulation": "deterministic_lightweight_rag",
  "retrieval_metrics": {"recall_at_1": 0.666667, "recall_at_3": 1.0, "mrr": 0.833333, "ndcg_at_3": 0.876977},
  "answer_metrics": {"groundedness": 0.875, "citation_precision": 1.0, "unsupported_claim_rate": 0.125}
}

$ python 05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/analysis.py
# 07 Retrieval-Augmented Generation and Eval 실행 관측
- scratch retrieval metrics와 framework answer metrics를 읽고 latest_report.md를 갱신한다.
```

생성 파일:
- `artifacts/scratch-manual/metrics.json`
- `artifacts/scratch-manual/rag_grounding_eval.svg`
- `artifacts/framework-manual/metrics.json`
- `artifacts/analysis-manual/latest_report.md`

## 이 단위에서 특히 볼 질문
- retrieval은 언제 parametric memory를 실제로 보완하고, 언제 irrelevant context만 늘려 성능을 깎는가?
- retriever-reader와 retriever-generator는 같은 retrieved chunk를 받아도 무엇을 더 강하게 통제하고 무엇을 더 자유롭게 두는가?
- citation이 달렸다는 사실과 answer가 truly grounded됐다는 사실은 왜 같은 말이 아닌가?
- context injection 선택은 unsupported claim, missing evidence, stale source 같은 failure mode를 어떻게 만든다?
- retrieval metrics가 좋아도 final answer quality나 online user correction rate가 나쁠 수 있는 이유는 무엇인가?
- evidence가 부족하거나 문서가 충돌할 때 answer / abstain / ask-back 중 무엇을 선택해야 하는가?

## 다음 단위와의 연결
RAG는 hallucination을 자동으로 없애지 않는다. retrieval은 evidence를 줄 기회를 만들 뿐이고, 어떤 evidence를 믿고 어떻게 말할지는 여전히 model behavior 문제다. 그래서 다음 단위 `05_advanced_nlp_llm/08_alignment_safety_and_model_behavior`의 refusal, uncertainty expression, trust calibration을 이해하려면 이 단위의 retrieval grounding과 eval harness를 먼저 분리해 읽어야 한다.
