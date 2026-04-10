# 07 Retrieval-Augmented Generation and Eval 선행 개념

## 꼭 알고 오면 좋은 것
- embedding similarity, top-k retrieval, reranking 같은 검색 파이프라인의 아주 기본 직관
- question answering / reading comprehension에서 "질문 + 근거 문맥 + 답" 구조를 읽는 감각
- prompt, context window, system/user instruction처럼 생성 입력을 구성하는 기본 용어
- chunking, document metadata, source freshness가 retrieval quality에 영향을 준다는 점
- answer correctness와 groundedness / citation adequacy를 별개로 봐야 한다는 점
- offline metric과 online product metric이 서로 다른 역할을 한다는 점

## 빠른 자기 점검
- parametric memory만으로 충분한 질문과 외부 retrieval이 꼭 필요한 질문을 예시로 구분할 수 있는가?
- retriever-reader와 retriever-generator가 evidence를 사용하는 방식 차이를 한두 문장으로 설명할 수 있는가?
- citation이 있다고 해서 answer 전체가 grounded된 것은 아니라는 점을 설명할 수 있는가?
- recall@k가 높아도 final answer quality가 낮을 수 있는 이유를 말할 수 있는가?
- evidence가 부족할 때 좋은 시스템이 추측 대신 uncertainty / abstain을 택해야 하는 이유를 이해하는가?

## 먼저 다시 보면 좋은 단위
- [04_nlp/03_machine_reading_comprehension](../../04_nlp/03_machine_reading_comprehension/README.md) — 질문과 근거 문맥을 함께 읽는 기본 QA framing 복습
- [05_advanced_nlp_llm/04_instruction_tuning_and_sft](../04_instruction_tuning_and_sft/README.md) — generator가 instruction과 role framing에 어떻게 반응하는지 복습
- [05_advanced_nlp_llm/06_rlhf_and_reasoning_rl](../06_rlhf_and_reasoning_rl/README.md) — 답변 행동을 objective로 밀던 단계와 retrieval grounding을 붙이는 단계를 구분
- [05_advanced_nlp_llm/06_rlhf_and_reasoning_rl](../06_rlhf_and_reasoning_rl/README.md) — retrieval metric을 modality가 달라도 어떻게 해석하는지 감각 연결
