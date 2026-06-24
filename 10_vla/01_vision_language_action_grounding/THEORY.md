# VLA Grounding Theory

VLA(Vision-Language-Action)는 이미지나 센서 관측, 자연어 지시, 행동 출력을 한 흐름으로 묶는다. VQA가 질문에 대한 text answer를 내는 문제라면, VLA는 상태를 바꾸는 `action token`이나 trajectory를 내야 한다.

## 최소 구성

1. **Vision state**: 장면에 무엇이 있는지, 위험물이 있는지, 목표 위치가 어디인지 나타내는 관측이다.
2. **Language instruction**: “빨간 블록을 집어라”, “장애물 앞에서 멈춰라” 같은 목표 조건이다.
3. **Policy head**: 관측과 지시를 합친 표현에서 action logits를 만든다.
4. **Safety gate**: action이 맞더라도 위험 상태에서는 실행을 막거나 stop action을 우선하게 만드는 별도 판정이다.

## action token이 중요한 이유

captioning의 출력은 문장이고, VQA의 출력은 답변이다. 반면 VLA의 출력은 실제 환경에 영향을 주는 행동이다. 그래서 `pick_red_block`, `push_blue_cube`, `stop_before_hazard` 같은 action token은 자연어 label처럼 보이지만, 평가에서는 success rate, trajectory error, intervention count, safety violation과 연결된다.

## safety gate를 분리해서 보는 이유

action accuracy가 높아도 위험한 장면에서 계속 움직이면 실제 시스템은 실패한다. 따라서 VLA에서는 “목표 action을 맞췄는가”와 “실행해도 안전한가”를 분리해서 로그로 남긴다. 이 단위의 toy 실험도 `action_accuracy`와 `safety_gate_accuracy`를 따로 기록한다.

## 실제 VLA로 확장할 때 필요한 것

- demonstration trajectory와 action space 정의
- behavior cloning 또는 offline RL baseline
- policy가 실패한 장면의 qualitative replay
- success rate, trajectory error, intervention count, safety violation
- simulation과 실제 장비 사이의 domain gap 기록

이 단위는 그 전에 “멀티모달 이해 결과가 어떻게 행동 선택 문제로 바뀌는가”를 최소 수치로 확인하는 역할을 한다.
