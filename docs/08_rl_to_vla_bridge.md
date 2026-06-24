# 08 RL to VLA Bridge

이 문서는 RLHF에서 쓰는 reward/policy 언어와 VLA에서 필요한 sequential decision-making 언어를 구분하기 위한 다리다. `05_advanced_nlp_llm/06_rlhf_and_reasoning_rl`을 읽었다고 해서 바로 robot/control RL을 이해한 것은 아니다. VLA에서는 token 품질뿐 아니라 상태, 행동, trajectory, 안전 제약이 함께 등장한다.

## MDP intuition

MDP는 sequential decision problem을 다음 요소로 나누어 본다.

- **state / observation**: 현재 보이는 정보다. VLA에서는 이미지, 언어 지시, 로봇 상태가 섞일 수 있다.
- **action**: 선택할 수 있는 행동이다. discrete action token일 수도 있고 continuous control일 수도 있다.
- **transition**: action 뒤 state가 어떻게 바뀌는지다.
- **reward**: 어느 행동/trajectory가 좋은지 평가하는 신호다.
- **policy**: observation을 보고 action을 고르는 규칙이다.

BTB의 VLA entry unit은 실제 로봇 transition 전체를 다루지 않고, observation+instruction을 action token과 safety gate로 바꾸는 최소 실험만 다룬다.

## Trajectory / return / credit assignment

- **trajectory**: observation, action, reward가 시간 순서로 이어진 기록이다.
- **return**: trajectory 전체에서 얻은 reward의 누적값이다.
- **credit assignment**: 긴 trajectory에서 어떤 action이 성공/실패에 기여했는지 나누는 문제다.

LLM RLHF의 rollout도 sequence지만, VLA trajectory는 환경 상태가 action 때문에 실제로 변한다는 점이 더 어렵다.

## Behavior cloning vs RL vs offline RL

- **behavior cloning(BC)**: 전문가 demonstration의 observation→action 쌍을 supervised learning처럼 따라 한다.
- **online RL**: policy가 환경에서 action을 해 보고 reward를 받아 업데이트한다.
- **offline RL**: 이미 모인 trajectory dataset으로 policy를 학습한다. 새 action을 마음대로 시도하지 못하므로 out-of-distribution action을 조심해야 한다.

처음 VLA를 배울 때는 BC로 action token mapping을 이해하고, 그다음 safety/failure/trajectory 평가로 확장하는 편이 안전하다.

## Action space design

VLA에서 action을 어떻게 표현할지에 따라 학습 문제가 달라진다.

- discrete token: `pick_red`, `move_left`, `stop`처럼 분류 문제로 시작하기 쉽다.
- parameterized action: `move(x, y)`처럼 행동 종류와 좌표를 함께 예측한다.
- continuous control: 속도, torque, gripper 값을 직접 낸다.

BTB의 entry unit은 discrete action token으로 시작한다. 이 선택은 이해를 쉽게 하지만 실제 robot trajectory의 세밀한 제어를 생략한다.

## RLHF reward와 robot/control reward는 다르다

RLHF에서는 reward model이 답변 품질을 점수화한다. VLA에서는 action이 환경을 바꾸고, 안전 위반은 단일 점수보다 hard constraint로 다뤄야 할 수 있다.

- RLHF reward: helpfulness, harmlessness, preference win-rate 중심
- VLA reward/safety: task success, collision, forbidden zone, intervention count, recovery 가능성 중심

따라서 “reward가 높다”는 말만으로 VLA 정책이 안전하다고 말하면 안 된다.

## `10_vla`에 들어가기 전 체크리스트

- observation과 state의 차이를 설명할 수 있다.
- action token이 실제 continuous control을 단순화한 표현임을 안다.
- behavior cloning, RL, offline RL의 데이터 사용 방식 차이를 말할 수 있다.
- action accuracy와 safety gate accuracy를 별도 지표로 봐야 하는 이유를 설명할 수 있다.
- ambiguous instruction, unsafe action, observation noise가 왜 별도 failure probe인지 안다.

## 최소 실험 아이디어

- 같은 장면에 안전한 action과 위험한 action 후보를 둘 다 만든다.
- model이 정답 action을 맞혔더라도 safety gate가 틀리면 실패로 표시한다.
- ambiguous instruction 예시를 추가해 “행동하지 않기/확인 질문”이 더 좋은 경우를 기록한다.
