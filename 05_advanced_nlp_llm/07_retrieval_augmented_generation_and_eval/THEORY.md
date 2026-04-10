# 07 Retrieval-Augmented Generation and Eval 이론 노트

## 핵심 개념

### 1. RAG intuition: parametric memory와 external evidence를 나눠 본다
- language model은 파라미터 안에 많은 패턴과 사실을 압축해 두지만, 최신 문서·사내 위키·긴 정책 문서처럼 **자주 바뀌거나 출처가 중요한 정보** 는 파라미터만으로 다루기 어렵다.
- retrieval-augmented generation(RAG)은 모델에게 답을 "더 잘 외우게" 하기보다, **답하기 전에 관련 문서를 먼저 찾아 읽히는 구조** 를 붙이는 접근이다.
- 직관적으로는 두 가지 메모리를 분리해서 본다.
  - **parametric memory**: 모델이 이미 가중치 안에 담아 둔 일반 지식과 언어 패턴
  - **non-parametric / external memory**: 지금 질의에 맞춰 동적으로 불러오는 문서, chunk, 테이블, 정책 텍스트
- 이때 핵심 질문은 단순하다.
  - 무엇을 찾을 것인가?
  - 찾은 것 중 무엇을 읽힐 것인가?
  - 읽힌 근거가 실제 답변을 얼마나 제한하는가?
  - 그 답변이 정말 근거에 anchored됐는지를 어떻게 확인할 것인가?

### 2. retriever-reader vs retriever-generator: 같은 retrieval이어도 answer control이 다르다
- RAG를 볼 때 흔히 retriever와 generator만 떠올리지만, 교육적으로는 **retriever-reader** 와 **retriever-generator** 를 구분해 보는 것이 좋다.

#### retriever-reader intuition
- retriever-reader 구조에서는 retriever가 후보 문서를 찾고, reader가 그 안에서 정답 span 또는 evidence-rich answer를 읽는다는 감각이 강하다.
- 전형적인 질문은 다음과 같다.
  - 어떤 chunk가 답이 있을 가능성이 높은가?
  - reader가 그 chunk 안에서 어느 문장을 핵심 근거로 삼는가?
- 이 구조는 **근거 위치를 더 강하게 고정** 하기 쉽고, extractive QA나 evidence extraction에 가까운 태도를 갖는다.
- 장점은 answer가 문서와 더 직접 연결되기 쉽다는 점이고, 단점은 답변 표현의 유연성이 상대적으로 적을 수 있다는 점이다.

#### retriever-generator intuition
- retriever-generator 구조에서는 retriever가 문서를 가져오고, generator가 그 문서를 조건으로 자연어 답변을 생성한다.
- 이 방식은 요약, synthesis, multi-document answer처럼 더 자유로운 생성에 잘 맞는다.
- 하지만 생성 자유도가 커질수록 다음 위험도 커진다.
  - 문서에 없는 연결 문장을 모델이 그럴듯하게 메우기
  - 일부 retrieved evidence만 쓰고 나머지는 parametric memory로 채우기
  - citation은 붙었지만 실제 claim은 citation 바깥에서 만들어 내기
- 따라서 retriever-generator는 단순히 "더 강력한 reader"가 아니라, **grounding discipline을 별도로 설계해야 하는 생성 시스템** 으로 보는 편이 안전하다.

### 3. retrieval grounding과 citation: 출처가 보인다고 자동으로 grounded answer는 아니다
- grounding은 answer의 주장(claim)이 실제 evidence와 연결돼 있음을 뜻한다.
- citation은 그 연결을 사용자에게 보이기 위한 한 방식일 뿐이다.
- 그래서 다음은 서로 다른 질문이다.
  - citation이 있는가?
  - citation이 실제로 relevant evidence를 가리키는가?
  - answer의 모든 주요 claim이 citation으로 덮이는가?
  - evidence가 answer를 truly support하는가, 아니면 loosely related한가?
- 좋은 RAG answer에서 기대하는 것은 단순히 링크 첨부가 아니라 다음과 같다.
  - **claim-evidence alignment**: 주요 주장마다 대응 evidence가 있다.
  - **coverage**: 핵심 주장 중 citation 바깥 claim이 남지 않는다.
  - **specificity**: 너무 넓은 문서 전체가 아니라 관련 chunk/section이 연결된다.
  - **honest uncertainty**: evidence가 부족하면 "확실하지 않다"고 말하거나 추가 확인을 요청한다.
- 즉 citation은 grounding의 proxy일 수는 있어도, grounding 그 자체는 아니다.

### 4. context injection intuition: 무엇을 어떻게 넣느냐가 answer shape를 바꾼다
- retrieval 이후 성능은 종종 검색기보다도 **찾아온 문서를 prompt/context로 어떻게 주입하는가** 에 크게 좌우된다.
- context injection에서 자주 보는 선택은 다음과 같다.
  - chunk 길이와 분할 기준(문단/슬라이딩 윈도우/문서 구조 기반)
  - top-k 개수와 rerank 유무
  - source metadata 포함 여부(title, date, document type)
  - system prompt 앞/뒤 어디에 evidence를 넣는가
  - citation tag를 inline으로 넣을지, answer 뒤에 묶을지
- 이 선택은 단순 formatting이 아니라 실제 failure mode를 만든다.
  - chunk가 너무 짧으면 answer에 필요한 맥락이 잘린다.
  - chunk가 너무 길면 irrelevant text가 늘어나 핵심 evidence가 묻힌다.
  - 오래된 문서와 최신 문서가 섞이면 generator가 잘못된 source를 채택할 수 있다.
  - retrieved document 안의 명령문/형식 노이즈가 prompt처럼 작동해 answer를 오염시킬 수 있다.
- 그래서 context injection은 "검색 결과 붙여 넣기"가 아니라, **어떤 evidence가 모델 attention 안에서 실제로 살아남는가를 설계하는 단계** 로 보는 편이 맞다.

### 5. 대표 failure modes: retrieval miss만이 문제가 아니다
- RAG failure는 대체로 두 층으로 나뉜다.

#### retrieval-side failure
- relevant 문서를 top-k에 못 올림
- 관련은 있지만 너무 넓거나 너무 좁은 chunk만 가져옴
- duplicate / near-duplicate 문서가 상위를 점령해 diversity가 무너짐
- stale source가 최신 source보다 더 높은 score를 받음
- keyword bias나 embedding mismatch 때문에 질문 의도와 다른 문서를 올림

#### answer-side failure
- retrieved evidence는 있었지만 generator가 무시함
- 문서 일부만 읽고 부족한 연결고리를 parametric memory로 채움
- 여러 문서를 섞으며 contradiction을 정리하지 못함
- citation은 붙였지만 실제 claim은 citation이 support하지 않음
- evidence가 부족한데도 confident answer를 생성함
- 충분한 근거가 없을 때 abstain 대신 추측을 함

#### pipeline interaction failure
- retriever는 recall이 높지만 generator input budget 때문에 핵심 chunk가 잘림
- reranker가 answerable chunk보다 surface similarity가 높은 chunk를 선호함
- evaluation dataset은 짧은 factoid QA라 좋아 보이지만 실제 long-form enterprise query에서는 무너짐
- grounding prompt가 강하면 verbosity와 latency가 커지고, 약하면 unsupported claim이 늘어남

### 6. evaluation harness: retriever와 answer를 분리해서 본다
- RAG 평가에서 가장 흔한 실수는 **final answer quality 하나만 보고 retrieval을 다 이해했다고 생각하는 것** 이다.
- 실제로는 적어도 세 층을 나눠 보는 편이 좋다.

#### A. retrieval quality
- relevant document가 top-k 안에 들어왔는가?
- ranking이 적절했는가?
- 대표 지표
  - Recall@k
  - Precision@k
  - MRR
  - nDCG
  - evidence coverage / document hit rate
- retrieval 지표는 "찾아올 기회"를 측정한다. 하지만 answer correctness를 곧바로 보장하지는 않는다.

#### B. answer quality / grounding quality
- answer가 맞는가?
- answer가 retrieved evidence에 grounded돼 있는가?
- citation이 claim-level로 충분한가?
- evidence 부족 시 uncertainty / abstention이 적절한가?
- 대표 관찰 포인트
  - exact match / F1 / task success
  - groundedness / faithfulness
  - citation precision / citation coverage
  - unsupported claim count
  - contradiction rate
- 여기서는 fluency보다 **evidence support** 를 분리해서 보는 것이 중요하다.

#### C. system/product quality
- 실제 사용자가 답을 받아들이는가?
- citation을 클릭하거나 source를 검토하는가?
- latency와 cost가 허용 범위 안에 있는가?
- 필요할 때만 retrieval을 호출하는가?
- online에서 흔히 보는 신호는 다음과 같다.
  - accept rate / answer completion rate
  - follow-up correction rate
  - citation click/open rate
  - escalation rate
  - p50 / p95 latency
  - query abandonment rate
- offline metric이 좋아도 online metric이 반드시 따라오지는 않는다. 긴 답변이 judge에는 좋아 보여도 사용자에게는 느리고 번거로울 수 있기 때문이다.

### 7. offline vs online metrics: 둘은 서로를 대체하지 않는다
- offline evaluation은 빠르게 iteration하기 좋다.
  - 정답 셋, annotated evidence, held-out queries로 retrieval과 grounding을 반복 측정할 수 있다.
- 하지만 offline만으로 놓치기 쉬운 것은 다음과 같다.
  - 실제 사용자의 query phrasing 변화
  - trust / confidence calibration
  - citation을 사용자들이 실제로 읽는지 여부
  - latency와 UX cost
  - long-tail failure와 multi-turn clarification behavior
- online 평가가 필요하다고 해서 offline이 불필요한 것도 아니다.
- 더 실용적인 해석은 다음과 같다.
  - offline은 **실험과 regression guardrail**
  - online은 **사용자 가치와 운영 리스크 확인**
- 좋은 evaluation harness는 이 둘을 연결한다. 예를 들어 offline groundedness가 떨어지는 케이스가 online correction rate 상승과 이어지는지 보는 식이다.

## Common Confusion
- RAG를 "hallucination을 자동으로 없애는 장치"로 보는 실수
- citation이 있으면 answer는 반드시 grounded됐다고 믿는 실수
- citation 개수가 많아질수록 groundedness도 자동으로 높아진다고 생각하는 실수
- retrieval recall만 높이면 end-to-end quality도 자동으로 오른다고 생각하는 실수
- retriever-reader와 retriever-generator 차이를 단지 모델 종류 차이로만 보는 실수
- 더 많은 chunk를 넣을수록 answer가 항상 좋아진다고 생각하는 실수
- evidence가 부족해도 모델이 그럴듯하게 메우면 usability가 더 좋다고 단정하는 실수
- offline QA benchmark 점수가 좋으면 실제 제품 query에서도 그대로 유지된다고 믿는 실수

## 이 단위에서 무엇을 관찰할 것인가
- query 유형에 따라 retrieval이 실제로 도움이 되는 지점과 오히려 noise가 되는 지점은 어디인가?
- 같은 retrieved set을 reader-style로 읽을 때와 generator-style로 읽을 때 unsupported claim 패턴은 어떻게 달라지는가?
- citation precision과 citation coverage 중 어느 쪽이 더 쉽게 무너지는가?
- chunking / reranking / prompt placement를 바꾸면 answer correctness보다 먼저 흔들리는 것은 무엇인가?
- evidence conflict가 있을 때 model은 summary, abstention, ask-back 중 어떤 행동을 택하는가?
- offline retrieval metric, groundedness metric, online correction rate 사이에는 어떤 gap이 남는가?
