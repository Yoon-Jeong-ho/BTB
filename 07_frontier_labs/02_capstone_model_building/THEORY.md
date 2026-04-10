# 02 Capstone Model Building 이론 노트

## 핵심 개념

### 1. capstone은 큰 아이디어 발표가 아니라 끝낼 수 있는 프로젝트 계약이다
- capstone에서 가장 흔한 실수는 "좋아 보이는 문제를 넓게 잡고 구현하면서 줄이자"는 태도다.
- 하지만 실제로는 반대여야 한다. 먼저 **무엇을 왜 만들고, 무엇은 이번 범위에서 하지 않을지**를 명확히 적어야 한다.
- 좋은 capstone scope는 보통 다음 요소를 한 문장 안에 담는다.
  - 누구를 위한 문제인가?
  - 입력과 출력은 무엇인가?
  - baseline 대비 무엇을 개선하려는가?
  - 성공과 실패를 어떤 기준으로 판단할 것인가?
- scope를 잘 자른 capstone은 작아 보일 수 있지만, 그 대신 결과 해석이 가능하다.
- 반대로 scope가 너무 넓으면 학습 데이터도 흔들리고, 모델 선택 기준도 흔들리고, eval도 자꾸 바뀌어서 마지막에 무엇을 배웠는지 설명하기 어려워진다.
- 따라서 capstone의 출발점은 "뭘 만들까?"보다 **"어떤 프로젝트 계약이면 끝낼 수 있고, 배울 수 있고, 다음 개선으로 이어질까?"** 다.

### 2. milestone decomposition은 작업 목록 분할이 아니라 불확실성 분해다
- milestone을 흔히 TODO 묶음으로 적지만, 좋은 milestone은 **프로젝트의 가장 큰 불확실성을 순서대로 줄이는 관문**이다.
- 예를 들어 capstone을 다음처럼 분해할 수 있다.
  - M0: 문제 정의, dataset/eval contract, baseline freeze
  - M1: 최소 reproducible pipeline 확보
  - M2: 개선 실험과 ablation 수행
  - M3: failure analysis, 보고서, 다음 제안 정리
- 여기서 중요한 것은 milestone마다 종료 조건이 있어야 한다는 점이다.
  - M0 종료 조건: split, metric, baseline이 문서로 고정되었다.
  - M1 종료 조건: 학습/평가 파이프라인이 같은 데이터 계약 아래 반복 가능하다.
  - M2 종료 조건: 개선 실험이 baseline과 같은 protocol에서 비교되었다.
  - M3 종료 조건: failure slice와 next action이 보고서로 남았다.
- milestone이 없으면 프로젝트는 늘 "아직 조금만 더" 상태가 된다.
- milestone이 너무 구현 중심이면 중요한 질문이 뒤로 밀린다. 예를 들어 모델 코드를 다 짰는데 eval split이 바뀌면, 진행한 양과 무관하게 의사결정은 다시 뒤로 돌아간다.
- 그래서 milestone decomposition의 핵심은 **코드 양이 아니라, 어떤 질문을 언제 닫을지 정하는 것**이다.

### 3. dataset / model / eval framing은 함께 설계되어야 한다
- capstone은 dataset만 정하거나 model만 고르는 것으로는 성립하지 않는다. 반드시 **dataset / model / eval 세 계약이 함께 묶여야** 한다.

#### dataset framing
- 어떤 데이터 소스를 쓰는가?
- split은 어떻게 고정되는가?
- leakage, duplication, label noise, coverage imbalance는 어디서 생길 수 있는가?
- offline dataset만 볼지, 추가 수집/정제가 필요한가?
- dataset framing이 약하면 나중에 baseline과 개선 모델의 비교 자체가 흔들린다.

#### model framing
- 가장 먼저 비교할 baseline은 무엇인가?
- 더 큰 모델을 쓰기 전에, 작은 baseline이 이미 문제를 충분히 해결하는지 확인했는가?
- fine-tuning, frozen encoder, retrieval augmentation, reranking 같은 선택지 중 이번 capstone의 주력 경로는 무엇인가?
- compute / memory / latency budget은 어느 정도인가?
- model framing의 목적은 최신 모델 이름을 고르는 것이 아니라, **어떤 비교선 위에서 개선을 주장할지 정하는 것**이다.

#### eval framing
- 주 metric은 무엇인가?
- 보조 metric과 slice metric은 무엇인가?
- 사람이 직접 확인해야 하는 qualitative failure bucket은 무엇인가?
- success 기준은 몇 점 향상인가, 아니면 특정 failure 유형 감소인가?
- eval framing이 없으면 실험은 돌아가도 프로젝트는 끝나지 않는다.

- 결국 dataset / model / eval 중 하나가 비면 이런 일이 생긴다.
  - dataset 계약이 약하면 결과가 leakage인지 개선인지 구분이 안 된다.
  - model 계약이 약하면 baseline이 약해 개선 해석이 과장된다.
  - eval 계약이 약하면 숫자는 나오지만 어떤 실패가 줄었는지 설명이 안 된다.

### 4. capstone 보고서는 사후 장식이 아니라 실험 운영 문서다
- 많은 초심자가 보고서를 "마지막에 결과를 예쁘게 정리하는 문서"로 생각한다.
- 그러나 capstone에서는 보고서 구조를 먼저 잡아 두는 편이 훨씬 유리하다.
- 보고서의 기본 구조는 보통 다음 질문을 순서대로 닫아야 한다.
  - 문제와 scope는 무엇이었는가?
  - baseline과 비교 조건은 무엇이었는가?
  - dataset과 eval protocol은 어떻게 고정했는가?
  - 어떤 milestone을 어떤 순서로 밟았는가?
  - 결과는 무엇이었고, 어디서 실패했는가?
  - 다음 개선은 무엇을 먼저 시도해야 하는가?
- 즉 보고서는 성과 발표 자료가 아니라, **의사결정과 관찰의 압축본**이다.
- 실험을 돌리기 전에 보고서 섹션을 써 두면, run을 할 때도 무엇을 로그로 남겨야 하는지가 선명해진다.
- 그래서 좋은 capstone은 최종 코드보다 먼저 **report outline, result table shape, failure note template**이 정리되어 있다.

### 5. failure analysis는 결과가 안 좋을 때만 쓰는 부록이 아니다
- frontier 프로젝트에서는 "성공한 run만 남기고 실패는 버린다"는 태도가 가장 큰 손실이 된다.
- failure analysis는 단순히 안 된 이유를 적는 메모가 아니라, **프로젝트가 다음 실험으로 이어지게 하는 핵심 산출물**이다.
- 최소한 다음은 처음부터 설계되어 있어야 한다.
  - 어떤 slice에서 실패를 볼 것인가? (길이, 카테고리, 난도, 언어, modality 등)
  - qualitative failure bucket은 어떻게 나눌 것인가?
  - baseline failure와 improved model failure를 같은 방식으로 비교할 것인가?
  - failure를 보고 난 뒤 어떤 가설과 다음 행동으로 연결할 것인가?
- 이 설계가 없으면 결과가 낮게 나왔을 때 "더 큰 모델을 써 보자" 같은 무의미한 반응으로 흐르기 쉽다.
- 반대로 failure analysis가 설계되어 있으면, 성능 향상이 작더라도 **어떤 사용자 slice에서 의미 있는 변화가 있었는지** 읽을 수 있다.

### 6. capstone의 관찰 포인트는 숫자 자체보다 계약의 흔들림이다
- capstone을 운영하면서 특히 봐야 하는 것은 단일 metric보다도 **프로젝트 계약이 어디서 흔들리는지**다.
- 대표적인 관찰 포인트는 다음과 같다.
  - scope creep: 중간에 목표가 늘어나지 않는가?
  - baseline weakness: baseline이 너무 약해서 improvement가 과장되지 않는가?
  - data leakage: validation/test에 train 정보가 새지 않는가?
  - metric gaming: metric은 올랐는데 실제 qualitative quality는 나빠지지 않는가?
  - budget mismatch: 설계한 milestone에 비해 compute/time이 비현실적이지 않은가?
  - report drift: 실험은 많아지는데 보고서 구조는 오히려 흐려지지 않는가?
- 즉 capstone의 난점은 "실험을 많이 돌리는 것"보다 **프로젝트 언어를 일정하게 유지하는 것**에 있다.

## 자주 헷갈리는 지점
- 문제 정의보다 모델 선택을 먼저 해 버리는 실수
- baseline이 약한데도 improvement 숫자를 크게 해석하는 실수
- dataset split과 eval protocol을 늦게 확정해 모든 결과를 다시 해석해야 하는 실수
- milestone을 코드량 기준으로 세우고, 실제 의사결정 종료 기준은 비워 두는 실수
- 보고서를 성공한 결과만 남기는 발표 자료로 착각하는 실수
- failure analysis를 "결과가 망했을 때만 쓰는 섹션"으로 미루는 실수
- scope를 줄였는데도 원래 거대한 문제 전체를 해결한다고 말해 버리는 실수

## 이 단위에서 무엇을 관찰할 것인가
- 지금 적은 problem statement가 실제 데이터와 평가 기준까지 이어지는가?
- baseline이 충분히 정직하고 재현 가능한 비교선으로 잡혀 있는가?
- milestone마다 종료 조건과 required artifact가 명시되어 있는가?
- 결과가 좋아도 나빠도 같은 report outline 안에서 해석할 수 있는가?
- failure slice와 qualitative bucket이 처음부터 정의되어 있는가?
- 다음 단위의 agentic loop가 들어왔을 때 자동 triage 기준으로 쓸 수 있는 문장이 이미 준비되어 있는가?
