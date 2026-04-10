# 05 Open-Ended Research Tracks 이론 노트

## 핵심 개념

### 1. open-ended research는 자유 탐색이 아니라 범위 설계 문제다
- frontier 연구에서 "정답이 아직 없다"는 말은 무엇이든 해도 된다는 뜻이 아니다.
- 오히려 정답이 없기 때문에, 이번 연구 트랙이 **어떤 질문을 다루고 어떤 질문은 아직 다루지 않는지**를 더 명확히 적어야 한다.
- 좋은 research scope는 보통 다음 요소를 함께 가진다.
  - north-star question: 장기적으로 풀고 싶은 큰 질문
  - this-track focus: 이번 트랙에서 실제로 좁혀 볼 작은 질문
  - out-of-scope: 흥미롭지만 이번에는 건드리지 않을 질문
  - fixed constraints: benchmark, budget, tool/schema, data version 같은 고정 조건
- 이 구분이 없으면 연구는 확장만 되고 누적되지 않는다.
- open-ended research를 운영한다는 것은 결국 **질문 자체를 실험 가능한 크기로 자르는 기술**에 가깝다.

### 2. hypothesis framing은 아이디어 수집이 아니라 keep/kill 판정을 가능하게 만드는 일이다
- exploratory research에서 흔한 착각은 "아이디어를 많이 적어 두면 연구가 넓어진다"는 믿음이다.
- 실제로는 hypothesis가 많아질수록 더 중요한 것은 각 hypothesis가 **무슨 관찰이 나오면 계속 밀고, 무슨 관찰이 나오면 접을지**를 미리 적는 일이다.
- 좋은 hypothesis framing에는 최소한 다음이 들어간다.
  - claim: 무엇이 좋아질 것이라고 보는가?
  - mechanism guess: 왜 그 변화가 일어날 것이라고 보는가?
  - observable evidence: 어떤 metric / slice / qualitative sign을 볼 것인가?
  - kill criterion: 어떤 결과가 나오면 이 가설을 현재 버전으로는 접을 것인가?
  - non-goal: 이번 hypothesis가 설명하지 못하는 것은 무엇인가?
- 이때 hypothesis는 거대한 설명 이론일 필요가 없다.
- 오히려 frontier 연구에서는 "planner brief를 더 짧게 제한하면 retry drift가 줄어든다"처럼 **작고 falsifiable한 문장**이 더 운영 가능하다.
- keep/kill 기준이 없는 hypothesis는 연구 아이디어라기보다 future wish list에 가깝다.

### 3. iteration boundary를 먼저 적어야 탐색이 wandering이 되지 않는다
- open-ended research의 가장 흔한 실패는 iteration마다 바뀌는 것이 너무 많아, 결과를 해석할 수 없게 되는 것이다.
- 그래서 각 iteration에는 boundary가 필요하다.
  - 무엇을 바꾸는가?
  - 무엇을 고정하는가?
  - 몇 번까지 retry하는가?
  - variance 안의 작은 변화는 improvement로 볼 것인가?
  - benchmark drift나 protocol mismatch가 보이면 중단하는가?
- 이 경계가 중요한 이유는 두 가지다.
  - 첫째, 작은 학습을 누적할 수 있다.
  - 둘째, 나중에 archive를 읽는 사람이 "왜 여기서 멈췄는가"를 이해할 수 있다.
- iteration boundary가 없으면 이후 해석은 대개 retrospective narrative가 된다. 즉, 결과를 본 뒤 그럴듯한 설명을 붙이는 방식으로 흘러가기 쉽다.
- 반대로 boundary가 분명하면, exploratory research라도 **각 시도가 같은 언어로 비교 가능**해진다.

### 4. exploratory goal에서도 evidence standard는 느슨해지지 않아야 한다
- frontier research에서 자주 듣는 말 중 하나는 "아직 exploratory 단계니까 정량 평가가 약해도 된다"는 것이다.
- 하지만 exploratory라는 이유로 evidence standard까지 흐리면, 남는 것은 인상 깊은 사례 모음과 기억 의존적 회고뿐이다.
- exploratory evidence standard는 보통 다음 층을 함께 가진다.
  - baseline-relative signal: 기존 방식과 비교했을 때 어떤 변화가 있었는가?
  - slice observation: 특정 failure slice에서만 개선/악화가 있었는가?
  - qualitative example: 숫자로 다 안 보이는 흥미로운 패턴이 있는가?
  - negative result: 기대와 반대로 나온 결과는 무엇인가?
  - inconclusive reason: 왜 아직 결론을 못 내리는가?
- 여기서 negative result와 inconclusive result를 구분하는 것이 중요하다.
  - negative result: 관찰은 충분했고, hypothesis가 현재 형태로는 지지되지 않음
  - inconclusive result: 관찰 체계나 실험 경계가 약해서 아직 결론을 내릴 수 없음
- 이 구분이 있어야 team이 다음 행동을 다르게 고를 수 있다.
  - negative면 archive 후 종료가 맞을 수 있고,
  - inconclusive면 measurement, benchmark, artifact contract를 다시 설계해야 할 수 있다.
- exploratory research에서도 "무엇을 아직 모르는가"를 명시하는 문장이 evidence의 일부다.

### 5. stopping rule은 열정의 반대가 아니라 연구 자원의 보호 장치다
- 많은 연구팀이 stopping rule을 "일찍 포기하는 장치"처럼 오해한다.
- 실제로는 그 반대다. stopping rule은 반복 탐색이 의미를 잃는 순간을 식별해서, 시간·예산·집중력을 더 중요한 질문으로 돌리게 해 준다.
- stopping rule은 보통 몇 가지 유형으로 나눠 적는다.
  - success stop: 목표 signal을 충분히 달성했을 때
  - no-signal stop: 반복해도 효과가 variance band 안에 머물 때
  - trust stop: benchmark drift, contamination, protocol mismatch로 결과 해석 신뢰가 무너질 때
  - scope stop: 질문이 더 이상 이 트랙 크기로 쪼개지지 않을 때
  - budget stop: 허용한 iteration / compute / annotation budget을 넘겼을 때
- 특히 exploratory track에서는 trust stop이 중요하다.
- signal이 조금 좋아 보여도 benchmark나 evidence contract가 흔들리면, 더 돌리는 것이 아니라 **먼저 측정 체계를 고쳐야** 한다.
- 즉 멈춤은 실패라기보다, 현재 트랙의 해석 가능성을 지키는 운영 판단이다.

### 6. archive discipline이 없으면 연구는 재시작 가능한 자산이 되지 않는다
- frontier research는 종종 "이번에는 안 됐지만 감은 있었다"는 식의 기억에 의존해 끝난다.
- 하지만 좋은 연구 트랙은 성공한 hypothesis만 남기지 않고, 다음도 함께 archive한다.
  - 왜 이 질문을 열었는가?
  - 어떤 baseline / benchmark / version 위에서 보았는가?
  - 무엇을 시도했고 무엇을 고정했는가?
  - 무엇이 negative result였고 무엇이 inconclusive였는가?
  - 왜 stop / pause / escalate / archive를 선택했는가?
  - 다시 열려면 어떤 조건이 충족되어야 하는가?
- archive discipline의 목적은 문서량을 늘리는 것이 아니라, **같은 실패를 비싼 방식으로 다시 반복하지 않게 하는 것**이다.
- 또한 negative result archive는 팀의 판단력을 지켜 준다. 기록이 없으면 사람들은 몇 주 뒤 같은 아이디어를 "새로운 intuition"으로 다시 들고오기 쉽다.
- 열린 연구일수록 archive는 나중에 보는 사람을 위한 배려가 아니라, 현재 팀의 학습 속도를 지키는 핵심 장치다.

### 7. 자주 생기는 혼동은 scope, evidence, stop을 서로 다른 문제로 취급하는 데서 나온다
- open-ended research 운영에서 흔한 혼동은 다음과 같다.
  - "재미있다"와 "이번 트랙에 맞다"를 같은 판단으로 섞는 경우
  - 좋은 qualitative 사례 몇 개를 바로 hypothesis 지지 evidence로 과대해석하는 경우
  - no-signal과 inconclusive를 구분하지 않고 둘 다 실패라고 처리하는 경우
  - stopping rule을 미리 적지 않고, 나중에 결과가 아쉬우면 탐색을 연장하는 경우
  - archive를 결과 발표 문서쯤으로 생각해 negative result를 의도적으로 약하게 적는 경우
- 이 혼동을 줄이려면 scope, evidence, stop, archive를 한 문서 흐름으로 봐야 한다.
- 어떤 질문을 이 트랙에 넣었는지가 evidence standard를 바꾸고,
- 어떤 evidence standard를 택했는지가 stopping rule을 바꾸며,
- 어떤 stopping rule을 썼는지가 archive note의 의미를 바꾼다.
- 결국 open-ended research는 자유로운 상상력과 엄격한 기록 습관을 동시에 요구한다.

## 직관 / 운영 프레임
- 아주 거칠게 쓰면, exploratory research track의 운영 품질은 다음처럼 생각할 수 있다.
  - `track_quality ≈ scope_clarity × hypothesis_testability × evidence_discipline × archive_reusability`
- 어느 하나가 낮으면 breakthrough처럼 보이는 결과가 나와도 다음 사람이 이어받기 어렵다.
- 특히 archive_reusability가 낮으면 연구는 매번 새로 시작하는 프로젝트처럼 느껴지고, 팀의 장기 누적 학습이 끊긴다.

## 자주 헷갈리는 지점
- open-ended research를 범위가 없는 자유 탐색으로 이해하는 실수
- hypothesis를 큰 비전 문장으로만 적고 keep/kill 기준을 빼먹는 실수
- exploratory phase라는 이유로 baseline 비교나 negative result 기록을 느슨하게 만드는 실수
- stopping rule을 열정 부족으로 오해해 뒤늦게만 적용하는 실수
- archive를 성공 사례 모음집처럼 써서 실패/보류 이유를 숨기는 실수
- no-signal과 measurement failure를 같은 실패로 취급해 다음 행동을 잘못 고르는 실수

## 이 단위에서 무엇을 관찰할 것인가
- 지금 적은 research question이 실제 iteration 단위로 쪼개졌는가, 아니면 여전히 너무 큰 구호에 머무는가?
- 각 hypothesis마다 claim, boundary, evidence field, kill criterion이 같이 적혀 있는가?
- exploratory result를 읽을 때 baseline-relative signal과 qualitative observation이 함께 남는가?
- negative result와 inconclusive result가 다른 후속 행동으로 연결되게 기록되는가?
- stop / pause / escalate / archive 결정이 결과를 본 뒤 즉흥적으로 바뀌지 않고, 미리 정한 rule과 연결되는가?
- archive note만 읽어도 다음 사람이 왜 여기서 멈췄고 언제 다시 열 수 있는지 이해할 수 있는가?
