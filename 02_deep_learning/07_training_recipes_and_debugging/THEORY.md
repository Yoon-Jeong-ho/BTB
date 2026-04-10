# 07 Training Recipes and Debugging 이론 노트

## 핵심 개념

### 1. training recipe는 부가 옵션이 아니라 수렴 계약이다
- architecture가 같아도 optimizer, learning rate, batch size, weight decay, scheduler, logging 규칙이 달라지면 결과는 크게 달라진다.
- 그래서 training recipe는 "나중에 만지는 미세 조정"이 아니라, **이 실험을 어떤 방식으로 학습시킬 것인가를 정의하는 계약**에 가깝다.
- 작은 모델에서 이 계약을 읽을 수 있어야 큰 모델에서도 loss spike, 과적합, 불안정 학습을 덜 블랙박스처럼 보게 된다.

### 2. learning rate basics
- learning rate는 한 step에서 파라미터를 얼마나 크게 움직일지 정한다.
- **너무 크면** loss가 요동치거나 발산(divergence)하고, activation/gradient가 커지며 NaN으로 이어질 수 있다.
- **너무 작으면** 학습이 지나치게 느리고, train loss조차 충분히 내려가지 않아 underfit처럼 보일 수 있다.
- 중요한 점은 learning rate가 단지 "속도"가 아니라 **안정성, 도달 가능한 minima, 로그의 모양**까지 함께 바꾼다는 것이다.
- later transformer/LLM 계열에서는 warmup 없이 큰 learning rate를 바로 쓰면 초반 loss spike가 잘 생기므로, base LR과 warmup 길이를 함께 생각해야 한다.

### 3. batch size basics
- batch size는 gradient 추정에 몇 개 샘플을 한 번에 쓸지 정한다.
- 작은 batch는 gradient noise가 커서 들쭉날쭉할 수 있지만, 때로는 더 강한 regularization처럼 작동하기도 한다.
- 큰 batch는 step당 추정이 안정적이고 throughput이 좋아질 수 있지만, 메모리 요구량이 커지고 generalization이 항상 좋아지는 것은 아니다.
- 실제 운영에서는 per-device batch size만이 아니라 **effective batch size = batch × grad accumulation × data parallel world size** 를 함께 봐야 한다.
- 따라서 batch size를 바꾸는 것은 단순한 메모리 조정이 아니라, optimization regime 자체를 바꾸는 결정이다.

### 4. weight decay basics
- weight decay는 파라미터 크기를 과도하게 키우지 않도록 누르는 regularization 항이다.
- 직관적으로는 "훈련 데이터를 너무 빡빡하게 외우지 않게 하는 장치"에 가깝다.
- 너무 약하면 train loss는 잘 내려가도 validation 성능이 흔들리며 overfit가 빨리 나타날 수 있다.
- 너무 강하면 필요한 표현력까지 눌러 버려 underfit처럼 보일 수 있다.
- 현대 deep learning에서는 특히 **AdamW의 decoupled weight decay** 관점이 중요하다. learning rate와 weight decay는 모두 성능에 영향을 주지만, 서로 완전히 같은 역할은 아니다.

### 5. scheduler basics
- scheduler는 학습 과정에서 learning rate를 어떻게 바꿀지 정하는 규칙이다.
- 초반에는 warmup으로 너무 급한 업데이트를 막고, 중후반에는 decay로 step 크기를 줄이며 더 안정적으로 수렴하게 만드는 경우가 많다.
- 대표적으로 step decay, cosine decay, linear decay, plateau-based decay 같은 형태가 있다.
- scheduler는 "좋은 base LR 없이도 마법처럼 해결해 주는 도구"가 아니라, **base LR을 시간축에 맞게 조절하는 운영 장치**로 봐야 한다.
- 그래서 total steps, warmup ratio, eval interval과 함께 읽어야 의미가 생긴다.

### 6. overfit / underfit monitoring
- train loss와 validation loss/metric은 서로 다른 질문에 답한다.
  - train loss: 모델이 훈련 데이터 패턴을 얼마나 잘 맞추는가?
  - validation loss/metric: 그 패턴이 보지 않은 데이터에도 유지되는가?
- 자주 보는 패턴은 다음과 같다.
  - **underfit**: train loss와 validation loss가 둘 다 높고 잘 안 내려간다.
  - **overfit**: train loss는 계속 내려가는데 validation loss는 멈추거나 다시 올라간다.
  - **optimization failure**: 둘 다 들쭉날쭉하고 step마다 큰 폭으로 흔들리며 때로는 바로 발산한다.
- loss curve만 보지 말고 gradient norm, activation range, learning-rate schedule, batch 구성 변화도 함께 봐야 한다.
- 좋은 모니터링은 "성능이 좋다/나쁘다"를 넘어, **왜 그런 모양이 나왔는지 추론 가능한 로그**를 남기는 것이다.

### 7. NaN / divergence / data bug debugging
- 딥러닝 디버깅의 첫 단계는 "무엇이 깨졌는가"보다 **어느 계층에서 처음 이상해졌는가** 를 찾는 것이다.
- NaN의 흔한 원인:
  - 입력 텐서에 이미 NaN/Inf가 들어 있음
  - 과도한 learning rate 또는 exploding gradient
  - mixed precision overflow
  - 잘못된 normalization / scaling
  - target range 오류, label index 오류, loss 함수와 target 형식 불일치
- divergence의 흔한 원인:
  - learning rate 과대
  - warmup 부재
  - gradient clipping 부재
  - accumulation / loss scaling 설정 오류
  - scheduler step 타이밍 오류
- data bug의 흔한 원인:
  - 입력-정답 misalignment
  - 잘못된 shuffle / batch collation
  - mask / padding / ignore_index 설정 오류
  - 이미지/토큰 정규화 불일치
  - train/validation split leakage
- 디버깅 순서는 대체로 이렇다.
  1. seed를 고정하고 재현 가능한 최소 케이스를 만든다.
  2. first bad step / batch를 찾는다.
  3. 입력, target, loss 구성 요소, gradient norm의 범위를 본다.
  4. mixed precision, augmentation, scheduler 같은 복잡도를 잠시 꺼 본다.
  5. tiny subset / single batch로 줄여 "이론상 외울 수 있는가"를 확인한다.
  6. 마지막으로 config 차이와 data pipeline 차이를 ablation으로 비교한다.

### 8. ablation / sanity-check habits
- 좋은 실험 습관은 큰 sweep보다 먼저 **작은 sanity check** 로 시작한다.
- 대표적인 sanity check는 다음과 같다.
  - single batch를 거의 완전히 overfit할 수 있는가?
  - tiny subset 몇 개 샘플에서 loss가 꾸준히 내려가는가?
  - random label로 바꾸면 성능이 무너지는가?
  - baseline/frozen/random guess와 비교했을 때 의미 있는 개선이 있는가?
- ablation의 핵심은 "한 번에 하나만 바꾸기"다.
- learning rate, scheduler, batch size, weight decay를 동시에 바꾸면 어떤 변화가 원인인지 읽기 어렵다.
- 결국 좋은 ablation은 최고 점수 표가 아니라, **무엇이 정말 효과를 냈는지 설명 가능한 비교표**를 만드는 일이다.

### 9. 왜 이 단위가 later LLM / system work를 지탱하는가
- instruction tuning, DPO/ORPO, RLHF, long-context fine-tuning에서도 loss spike, warmup, effective batch, data-quality bug는 계속 문제를 만든다.
- distributed training에서는 DDP/FSDP/ZeRO 자체도 중요하지만, 그 위에서 **학습이 제대로 되고 있는지 읽는 운영 감각**이 먼저 필요하다.
- 예를 들어 grad accumulation을 늘렸을 때 이것이 메모리 대응인지, optimization regime 변경인지 읽을 수 있어야 한다.
- checkpoint recovery나 failure monitoring도 결국 "정상 학습 로그"와 "이상 학습 로그"를 구분할 수 있어야 의미가 생긴다.
- 따라서 이 단위는 작은 딥러닝 실험용 디버깅 습관을, 나중에 LLM 및 시스템 단위 runbook으로 확장하는 기초 공사다.

## 자주 헷갈리는 지점
- learning rate를 단순히 "클수록 빠르다"로만 이해하는 실수
- batch size 증대를 throughput 개선으로만 보고 optimization 변화는 무시하는 실수
- weight decay와 scheduler를 둘 다 "정규화 같은 것"으로 뭉뚱그리는 실수
- validation 성능 저하를 무조건 overfit로만 보고, data bug나 metric 계산 오류 가능성은 안 보는 실수
- NaN이 나면 무조건 모델이 깊어서 그렇다고 생각하는 실수
- ablation이라고 해 놓고 여러 하이퍼파라미터를 동시에 바꿔 원인 추적이 불가능해지는 실수

## 이 단위에서 무엇을 관찰할 것인가
- learning rate를 올리거나 내렸을 때 loss curve의 기울기와 진동 패턴이 어떻게 달라지는가?
- batch size / effective batch 변화가 gradient noise, 메모리, validation metric에 어떤 흔적을 남기는가?
- weight decay와 scheduler를 조절했을 때 train-vs-validation 간격이 어떻게 바뀌는가?
- first bad batch를 찾고 나면 문제가 숫자 불안정인지 데이터 버그인지 얼마나 빨리 좁혀지는가?
- single-batch overfit, tiny-subset replay, random-label test 같은 sanity check가 실패 분류에 실제로 도움이 되는가?
- 나중에 LLM fine-tuning이나 distributed training 로그를 볼 때, 여기서 익힌 질문을 그대로 재사용할 수 있는가?
