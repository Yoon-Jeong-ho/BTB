# 07 Retrieval-Augmented Generation and Eval 선행 개념

이 단위는 CPU toy RAG/eval 실습이지만, 아래 감각이 있으면 retriever-reader / retriever-generator split과 retrieval grounding을 더 선명하게 볼 수 있다.

## 꼭 알고 오면 좋은 것
- embedding similarity, keyword overlap, top-k retrieval, reranking의 기본 직관
- question answering에서 “질문 + 근거 문맥 + 답”을 함께 읽는 감각
- prompt, context window, system/user instruction, context injection의 기본 용어
- chunking, document metadata, source freshness가 retrieval quality에 영향을 준다는 점
- citation과 claim-level evidence support가 다를 수 있다는 점
- answer correctness, groundedness, citation precision, unsupported claim rate를 분리해 봐야 한다는 점
- offline metrics와 online user/product metrics가 서로 다른 역할을 한다는 점

## 빠른 자기 점검
- parametric memory만으로 답하기 어려운 질문을 하나 만들 수 있는가?
- retriever-reader가 왜 abstain하기 쉬운 구조인지 설명할 수 있는가?
- retriever-generator가 왜 fluent하지만 unsupported claim에 취약한지 설명할 수 있는가?
- citation이 있어도 retrieval grounding이 부족할 수 있는 예를 말할 수 있는가?
- recall@k가 높아도 final answer quality가 낮을 수 있는 이유를 설명할 수 있는가?
- evidence가 부족할 때 answer / abstain / ask-back 중 무엇을 택할지 기준을 세울 수 있는가?

## 먼저 다시 보면 좋은 단위
- [04_nlp/03_machine_reading_comprehension](../../04_nlp/03_machine_reading_comprehension/README.md) — 질문과 근거 문맥을 함께 읽는 기본 QA framing
- [05_advanced_nlp_llm/04_instruction_tuning_and_sft](../04_instruction_tuning_and_sft/README.md) — generator가 instruction과 role framing에 반응하는 방식
- [05_advanced_nlp_llm/06_rlhf_and_reasoning_rl](../06_rlhf_and_reasoning_rl/README.md) — 답변 행동을 reward/eval signal로 다루는 관점
