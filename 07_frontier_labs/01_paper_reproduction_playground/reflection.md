# 01 Paper Reproduction Playground 회고 질문

이 reflection은 Korean-first로 작성한다. 답변할 때는 toy metrics 숫자를 그대로 베끼기보다 claim/evidence matrix, baseline/reported/reproduced comparison, scope control, variance, mismatch hypothesis, artifact hygiene가 서로 어떻게 연결되는지 설명한다.

## 학습자 프롬프트
1. 이번 실습의 `C1_adapter_efficiency`는 full paper reproduction이 아니라 reduced claim이다. 어떤 결론은 말할 수 있고, 어떤 결론은 말하면 안 되는가?
2. baseline, reported, reproduced 숫자를 한 표에 놓을 때 primary comparison을 reproduced baseline vs reproduced method로 둬야 하는 이유를 설명하라.
3. reproduced accuracy가 reported accuracy보다 낮게 나온 상황에서 preprocessing_alignment, seed_variance, budget_mismatch 중 무엇을 먼저 확인할지 순서를 정하고 이유를 적어라.
4. claim/evidence matrix에서 acceptance rule이 빠지면 어떤 과잉 해석이 생길 수 있는가?
5. variance summary의 seed std가 reported margin과 비슷한 크기라면 reproduction decision을 어떻게 낮춰 말해야 하는가?
6. artifact hygiene checklist에서 scope boundary와 mismatch hypotheses가 없으면 다음 사람이 어떤 작업을 다시 해야 하는가?
7. 이 단위의 experiment card schema를 다음 capstone 모델 빌딩 단위의 baseline/eval contract로 어떻게 재사용할지 제안하라.
8. “숫자가 paper와 비슷했다”보다 더 강한 reproduction evidence가 되려면 어떤 로그가 추가로 필요할지 적어라.

## 제출 체크리스트
- claim/evidence matrix를 언급했다.
- baseline, reported, reproduced를 분리했다.
- scope control 때문에 줄어든 claim 범위를 설명했다.
- variance와 mismatch hypothesis를 최소 두 가지 이상 다뤘다.
- artifact hygiene가 다음 실험 handoff에 왜 필요한지 설명했다.
