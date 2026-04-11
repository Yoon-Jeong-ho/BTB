# 08 Alignment, Safety, and Model Behavior 학습자 회고

## 실행 전 예측
1. alignment vs capability가 같은 축이 아니라면, capability-only assistant가 어떤 harmful request에서 실패할지 예측해 보라.
2. refusal이 harmlessness로 보이는 경우와 over-refusal로 usefulness를 해치는 경우를 각각 하나씩 적어 보라.
3. robustness probe에서 paraphrase, formatting noise, jailbreak-style phrasing 중 어떤 입력 변화가 가장 위험할지 예상해 보라.

## 실행 후 관찰
1. `scratch_lab.py`의 behavior_slices를 보고 benign answer rate, harmful refusal rate, over-refusal rate가 서로 어떤 긴장을 갖는지 설명하라.
2. `alignment_behavior_slices.svg`에서 helpfulness와 harmlessness가 동시에 좋아 보이더라도 어떤 slice analysis를 추가로 확인해야 하는가?
3. `framework_lab.py`에서 capability-only assistant와 aligned assistant의 behavior_contract_score 차이를 alignment vs capability 관점으로 해석하라.
4. borderline request에서 safe alternative가 왜 단순 refusal보다 더 좋은 harmlessness 행동일 수 있는지 적어 보라.

## failure mode 질문
1. over-refusal이 늘면 제품 관점에서 어떤 user trust 문제가 생기는가?
2. jailbreak-style phrasing에 안전 행동이 흔들린다면 robustness eval을 어떤 variant로 확장할 것인가?
3. behavioral eval 평균 점수가 좋아도 benign / harmful / borderline slice 중 하나가 나쁘면 어떤 결정을 보류해야 하는가?
4. policy vs system-level safety 관점에서 model policy가 거절을 잘해도 system guardrail이 없으면 어떤 tool permission gating 실패가 남는가?

## 다음 단위로 넘길 메모
- 이 단위가 해결한 것: model behavior를 capability와 분리하고 refusal, over-refusal, harmlessness, robustness를 slice별로 읽는다.
- 아직 남은 것: toy eval은 실제 threat model, traffic distribution, human review process를 대체하지 않는다.
- 다음 benchmark/dataset 설계에서는 behavioral eval slice analysis를 실제 rubric과 regression set으로 확장한다.
