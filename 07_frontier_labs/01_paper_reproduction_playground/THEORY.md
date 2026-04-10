# 01 Paper Reproduction Playground 이론 노트

## 핵심 개념

### 1. paper reproduction은 논문 전체 복제가 아니라 claim 단위 실험 설계다
- 초심자가 흔히 빠지는 함정은 paper reproduction을 "저자가 한 모든 실험을 다시 돌리는 일"로 이해하는 것이다.
- 하지만 실제 연구 운영에서는 먼저 **어떤 claim을 다시 확인하고 싶은가**를 고르는 편이 훨씬 중요하다.
- 예를 들어 논문에는 여러 층의 claim이 섞여 있다.
  - 핵심 성능 claim: 기존 baseline보다 metric이 얼마나 오른다.
  - 메커니즘 claim: 특정 모듈/손실/데이터 설계가 왜 효과가 있다고 해석하는가.
  - ablation claim: 어떤 구성 요소를 빼면 성능이 얼마나 줄어드는가.
  - 운영 claim: compute 효율, memory 절감, latency 개선이 실제로 나타나는가.
- reproduction playground의 목표는 이 claim들을 모두 한 번에 재현하는 것이 아니라, **가장 중요한 claim을 작은 실험 계약으로 다시 세우는 것**이다.
- 그래서 좋은 reproduction 질문은 "이 논문을 다 따라 했는가?"보다 "이 논문의 어느 주장까지를, 어떤 evidence로, 어떤 제약 아래 다시 확인했는가?"에 가깝다.

### 2. scope control: 무엇을 고정하고 무엇을 버릴지 먼저 정한다
- reproduction이 무너지는 가장 흔한 이유는 구현 능력 부족보다도 **scope가 처음부터 너무 넓기 때문**이다.
- scope control에서는 다음을 먼저 고정한다.
  - 어떤 claim을 대상으로 삼는가?
  - 어떤 dataset / split / task를 쓸 것인가?
  - full-scale 재현인지, trend만 보는 reduced reproduction인지?
  - baseline은 무엇이고, 어떤 metric으로 비교할 것인가?
  - 시간/compute budget은 얼마인가?
- 이때 중요한 태도는 "축소" 그 자체가 아니라 **정직한 축소**다.
- 예를 들어 full paper reproduction이 불가능하다면 다음처럼 잘라야 한다.
  - 전체 데이터 대신 subset으로 trend만 본다.
  - full model 대신 smaller proxy model로 상대적 경향만 본다.
  - full convergence 대신 fixed budget 안에서 early trend를 비교한다.
- 단, 이렇게 줄였다면 claim도 같이 줄여야 한다.
  - "paper와 같은 절대 수치를 재현했다"가 아니라
  - "같은 방향의 relative trend가 subset/proxy setting에서도 유지되는지 봤다"처럼 표현 범위를 낮춰야 한다.
- scope control은 결국 **무리한 약속을 막고, 해석 가능한 reproduction을 남기기 위한 방어 장치**다.

### 3. claim/evidence replication mindset: 숫자를 맞추는 것보다 증거 구조를 다시 세운다
- reproduction을 숫자 맞추기 게임으로만 보면, 수치가 조금만 어긋나도 "실패"처럼 느껴지기 쉽다.
- 그러나 연구적으로는 **왜 비슷했는지 / 왜 달랐는지 설명 가능한 evidence 구조**를 남기는 편이 더 중요할 때가 많다.
- claim/evidence mindset에서는 각 claim마다 최소한 다음을 적는다.
  - claim 자체: 무엇이 더 좋다고 말하는가?
  - evidence type: main metric, ablation delta, runtime, failure case, qualitative output 중 무엇으로 확인하는가?
  - comparison target: 어떤 baseline / prior setting과 비교하는가?
  - acceptance rule: 어느 정도 차이를 "같은 경향"으로 볼 것인가?
  - missing evidence: 지금 setting에서는 무엇을 아직 확인하지 못했는가?
- 이 관점이 중요한 이유는 두 가지다.
  - 첫째, 숫자가 비슷해도 evidence가 약하면 과한 결론을 막을 수 있다.
  - 둘째, 숫자가 다르더라도 원인을 좁힐 단서가 남는다.
- 따라서 좋은 reproduction artifact는 "0.7점 부족했다"에서 멈추지 않고, **variance, preprocessing, seed, hardware, evaluator 차이 중 무엇이 mismatch 후보인지** 를 함께 남긴다.

### 4. baseline vs reported result 비교는 같은 경기장에서 해야 한다
- reproduction에서 가장 위험한 실수 중 하나는 **같은 표에 있는 숫자들을 곧바로 같은 조건의 결과라고 믿는 것**이다.
- 실제로는 논문 숫자와 내 숫자 사이에 다음 차이가 숨어 있을 수 있다.
  - dataset version / split 차이
  - preprocessing / tokenization / augmentation 차이
  - metric 계산 방식 차이
  - seed 개수와 averaging 방식 차이
  - hardware / batch / precision / training length 차이
  - early stopping / checkpoint selection 기준 차이
- 그래서 baseline 비교는 항상 세 층으로 나눠 보는 편이 좋다.

#### A. reported baseline vs reported proposed method
- 논문 안에서 저자가 보고한 표를 그대로 읽는 단계다.
- 여기서는 논문이 어떤 margin을 주장하는지 파악한다.

#### B. reproduced baseline vs reproduced method
- 내가 같은 pipeline 아래에서 다시 돌린 baseline과 method를 비교하는 단계다.
- 이 비교가 가장 중요하다. 왜냐하면 같은 환경에서 얻은 차이만이 **내 reproduction setting 안에서 해석 가능한 delta** 이기 때문이다.

#### C. reported vs reproduced gap
- 논문 숫자와 내 숫자의 절대 차이를 보는 단계다.
- 이 차이는 재밌지만, reproduction core는 아니다. 핵심은 그 차이가 **protocol mismatch인지, variance인지, 실제 구현 실패인지**를 구분하는 것이다.
- 좋은 reproduction 보고는 보통 이렇게 말한다.
  - reported margin은 +1.8pt였다.
  - reproduced setting에서는 baseline 재실행 결과가 먼저 1.2pt 낮았다.
  - 같은 setting 안에서 reproduced margin은 +1.0pt였다.
  - 따라서 방향은 유지되지만 절대 수치는 preprocessing / seed variance 차이의 영향을 받는다.
- 즉 비교의 핵심은 **paper 표를 그대로 베끼는 것**이 아니라, **같은 protocol에서 baseline과 method를 다시 맞붙이는 것**이다.

### 5. reproduction pitfalls: 실패는 대부분 숨은 변수에서 나온다
- reproduction을 어렵게 만드는 것은 대개 model code 몇 줄보다도 **문서에 덜 드러난 운영 변수**다.
- 자주 만나는 함정은 다음과 같다.

#### hidden preprocessing
- normalization, token filtering, prompt template, data cleaning 규칙이 appendix나 code에만 숨어 있을 수 있다.
- 작은 preprocessing 차이가 metric gap보다 더 큰 효과를 낼 수 있다.

#### seed / variance blindness
- paper는 여러 seed 평균인데, reproduction은 한 번만 돌려 놓고 절대 숫자를 바로 비교하기 쉽다.
- reproduced gap이 실제 improvement보다 작거나 비슷한 크기의 variance일 수도 있다.

#### evaluation mismatch
- metric 이름이 같아 보여도 micro/macro averaging, exact match vs normalized EM, post-processing 규칙이 다를 수 있다.
- evaluation script가 다르면 같은 prediction도 다른 점수가 나온다.

#### budget mismatch
- 논문은 대규모 compute에서 충분히 수렴시켰는데, reproduction은 짧은 budget에서 early trend만 본 상황일 수 있다.
- 이때 절대 숫자를 직접 비교하면 해석이 과장된다.

#### undocumented tricks
- checkpoint averaging, gradient clipping, warmup, mixed precision setting, selective data filtering처럼 본문에 약하게 적힌 설정이 실제로는 중요할 수 있다.

#### confirmation bias
- 논문이 좋아 보이기 때문에 비슷한 결과만 찾고, mismatch 증거는 부차적으로 취급하기 쉽다.
- reproduction note는 성공 보고서가 아니라 **관찰 로그**여야 한다.

### 6. 무엇을 기록해야 다음 단위로 이어지는가
- reproduction playground는 단발성 실험이 아니라 이후 capstone과 open-ended research의 템플릿이 된다.
- 그래서 최소한 다음은 artifact로 남겨야 한다.
  - paper citation / claim ID / experiment 목표
  - dataset / split / preprocessing / metric 정의
  - baseline 정의와 재실행 여부
  - seed / hardware / runtime / budget 메모
  - reproduced result와 mismatch hypothesis
  - 다음 실험 제안: 더 좁힐 변수, 다시 확인할 코드 경로, capstone에 재사용할 구성
- 이 기록이 있어야 다음 단위에서 "어떤 모델을 만들 것인가"보다 먼저 **어떤 실험 계약과 비교선을 들고 갈 것인가**를 말할 수 있다.

## 자주 헷갈리는 지점
- reproduction을 full paper clone과 동일시하는 실수
- scope를 줄였는데도 원래 paper claim 전체를 그대로 주장하는 실수
- 논문 표의 baseline 숫자를 그대로 가져와 내 method run과 직접 비교하는 실수
- absolute metric gap만 보고 variance, seed, evaluator 차이를 무시하는 실수
- 코드가 돌아가면 reproduction이 끝났다고 생각하는 실수
- mismatch를 실패로만 보고, 다음 가설을 세울 evidence로 사용하지 않는 실수

## 이 단위에서 무엇을 관찰할 것인가
- 지금 고른 claim은 full reproduction 없이도 검증 가능한 작은 실험으로 내려올 수 있는가?
- baseline과 proposed method를 같은 protocol 아래 다시 비교했는가?
- reported gap과 reproduced gap 중 무엇이 protocol 차이의 영향을 더 크게 받고 있는가?
- mismatch가 났을 때 가장 먼저 떠오르는 후보가 구현 버그인지, preprocessing 차이인지, variance인지 분리되는가?
- artifact를 읽는 다른 사람이 다음 run을 바로 설계할 수 있을 정도로 로그와 메타데이터가 남아 있는가?
- 이 reproduction note가 다음 capstone 설계의 reusable template로 기능할 수 있는가?
