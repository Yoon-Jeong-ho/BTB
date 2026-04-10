# 07 Retrieval-Augmented Generation and Eval

> Status: outlined
>
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 구조** 이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
LLM이 모든 사실을 파라미터 안에 완벽히 기억할 수 있다고 가정하면, 최신 문서·사내 지식·긴 근거 문맥·출처 추적이 필요한 문제에서 곧바로 한계에 부딪힌다. 검색-증강 생성(retrieval-augmented generation, RAG)은 이런 한계를 단순히 "벡터 검색을 붙이는 기능"이 아니라, **무엇을 찾고, 무엇을 읽히고, 무엇을 근거로 답했다는 기대를 어떻게 검증할 것인가** 로 재구성하는 단위다. 이 단위는 retriever-reader / retriever-generator 구조를 한 프레임에서 비교하고, citation이 붙은 답이 왜 자동으로 grounded answer가 되지는 않는지, 그리고 offline/online 평가를 어떻게 분리해 봐야 하는지까지 함께 정리하게 만든다.

## 이번 단위에서 남길 것
- outlined 상태의 안내 문서 `README.md`
- retriever-reader / retriever-generator 직관, grounding 기대, 평가 관점을 정리한 `THEORY.md`
- 선행 개념과 빠른 자기 점검을 담은 `PREREQS.md`
- 단위 목표와 핵심 질문을 고정한 `lesson.yaml`
- 이후 산출물이 들어갈 자리 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 출력 계약
  - query별 retrieved chunk / rerank / citation 매핑 요약
  - retriever-reader vs retriever-generator 비교 메모
  - unsupported claim / missing evidence / stale source 사례 정리
  - offline retrieval metric과 online user metric을 나눠 본 평가 harness 초안

## 실습 흐름
현재는 outline 문서만 정리된 상태이며, 아래 흐름은 이후 runnable 승격 때 구현할 실습 순서다.
1. 먼저 closed-book 답변과 retrieval-augmented 답변을 나란히 두고, parametric memory만으로 충분한 질문과 외부 문서를 꼭 붙여야 하는 질문을 구분한다.
2. 검색 단계에서는 query → chunk → top-k retrieval → optional rerank(선택적 재정렬) 흐름을 보며, retriever가 실제로 어떤 근거 후보를 상위에 올리는지 관찰한다.
3. 같은 retrieval 결과를 두 구조로 읽는다. retriever-reader 관점에서는 **찾은 문서에서 근거 span을 읽어 답을 고정하는 방식**, retriever-generator 관점에서는 **찾은 문서를 조건으로 자유 생성하되 grounding 제약을 유지하는 방식** 을 비교한다.
4. context injection 단계에서는 chunk 길이, 문서 경계, metadata, citation tag, prompt 배치를 바꾸며 어떤 정보는 살아남고 어떤 정보는 묻히는지 본다.
5. 답변 단계에서는 citation이 붙었는지뿐 아니라, 각 주장(claim)이 실제 retrieved evidence와 연결되는지, evidence가 부족할 때 보수적 답변 / 추가 확인 요청 / 불확실성 표현이 나오는지 본다.
6. 마지막에는 evaluation harness를 둘로 나눈다. retriever 품질(recall@k, MRR, evidence coverage)과 answer 품질(groundedness, citation precision, answer correctness, latency, online acceptance)을 분리해서 보고, 다음 단위 `05_advanced_nlp_llm/08_alignment_safety_and_model_behavior`로 연결한다.

## 이 단위에서 특히 볼 질문
- retrieval은 언제 parametric LM을 실제로 보완하고, 언제 irrelevant context만 늘려 성능을 깎는가?
- retriever-reader와 retriever-generator는 같은 retrieved chunk를 받아도 무엇을 더 강하게 통제하고 무엇을 더 자유롭게 두는가?
- 답변에 citation이 달렸다는 사실과 답변이 truly grounded됐다는 사실은 왜 같은 말이 아닌가?
- chunking, top-k, reranking, prompt placement 같은 context injection 선택은 어떤 실패 모드를 만들 수 있는가?
- retrieval recall이 높아도 final answer quality가 낮을 수 있는 이유는 무엇인가?
- offline metric이 좋아 보여도 online user satisfaction, trust, correction rate와 어긋날 수 있는 이유는 무엇인가?
- evidence가 부족하거나 문서끼리 충돌할 때, 좋은 RAG system은 answer / abstain / ask-back 중 무엇을 어떻게 선택해야 하는가?

## 실행 결과 예시
아래는 **완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/scratch_lab.py
{
  "status": "sample",
  "query": "2026년 내부 정책 개정 후 출장비 상한은 얼마인가?",
  "retrieval_trace": {
    "retriever": "dense_topk_then_rerank",
    "retrieved_chunks": [
      {"chunk_id": "policy_12", "score": 0.88, "source": "travel_policy_v3.md"},
      {"chunk_id": "policy_03", "score": 0.81, "source": "faq_expenses.md"}
    ],
    "evidence_coverage": "partial",
    "missing_signal": ["effective_date_not_explicit"]
  },
  "answer": {
    "mode": "retriever_generator",
    "text": "상한은 1일 18만 원으로 보이지만, 시행일 확인이 추가로 필요합니다.",
    "citations": ["travel_policy_v3.md#chunk12"],
    "unsupported_claims": 0,
    "abstain_flag": false
  },
  "eval": {
    "groundedness": 0.83,
    "citation_precision": 1.0,
    "answer_correctness": "manual_check_needed"
  }
}

$ python 05_advanced_nlp_llm/07_retrieval_augmented_generation_and_eval/framework_lab.py
{
  "status": "sample",
  "batch_shape": {
    "query_embeddings": [8, 768],
    "doc_embeddings": [128, 768],
    "topk_indices": [8, 5],
    "prompt_tokens": [8, 1024]
  },
  "retrieval_metrics": {
    "recall_at_5": 0.76,
    "mrr": 0.61,
    "ndcg_at_10": 0.69
  },
  "answer_metrics": {
    "faithfulness": 0.72,
    "citation_coverage": 0.67,
    "latency_ms_p50": 410
  },
  "online_watch": {
    "accept_rate": 0.58,
    "citation_click_rate": 0.21,
    "user_correction_rate": 0.14
  }
}
```

핵심은 숫자 자체보다도 **retrieval trace가 실제 answer를 어떻게 제한하거나 도와주는지**, **citation이 claim-level grounding을 어디까지 보장하는지**, **retriever metric과 user-facing quality metric을 왜 분리해서 읽어야 하는지** 를 읽는 것이다.

## 다음 단위와의 연결
이 단위에서 RAG를 "문서를 붙인 생성"이 아니라 **evidence-conditioned behavior와 평가 설계 문제** 로 정리해 두면, 다음 단위 `05_advanced_nlp_llm/08_alignment_safety_and_model_behavior`에서 왜 refusal, uncertainty expression, 안전한 citation, trust calibration 같은 사용자 행동 문제가 다시 중요해지는지를 더 잘 볼 수 있다. 즉 retrieval은 hallucination을 줄일 기회를 주지만, 어떤 근거를 믿고 어떻게 말할지는 여전히 model behavior 문제로 이어진다.
