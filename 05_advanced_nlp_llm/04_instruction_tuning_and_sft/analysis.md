# 04 Instruction Tuning and SFT 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 toy SFT 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 instruction format, supervised fine-tuning, input-output template, role framing, imitation vs helpfulness tradeoff를 해석하는 **안정적인 프레임**만 남긴다.
- 따라서 `analysis.py`를 반복 실행해도 이 파일은 같은 내용으로 유지되고, 관측 보고서만 최신 값으로 갱신된다.

## 해석 프레임
- SFT는 base LM objective를 마법처럼 교체하기보다, serialized instruction example 위에서 assistant response target을 강조하는 supervised imitation 단계다.
- input-output template는 role boundary, generation 시작점, EOS/stop marker, loss target 위치를 결정하는 학습 신호다.
- system/user/assistant role framing은 모델 밖 메타데이터가 아니라 token sequence 안의 conditioning signal이다.
- assistant-only loss mask는 prompt 복창보다 응답 생성에 학습 신호를 집중시킨다.
- imitation score가 높아도 helpfulness가 자동으로 충분해지지는 않으므로, preference optimization으로 이어질 문제를 남겨야 한다.

## 확인 질문
- plain instruction format과 chat template는 같은 예시를 어떤 다른 token boundary로 보여 주는가?
- loss mask에서 prompt tokens를 ignored label로 둔다는 것은 어떤 학습 신호를 제거한다는 뜻인가?
- system message가 있을 때 role framing score가 올라간다면, 이것을 어떤 제품 행동으로 연결할 수 있는가?
- SFT training curve에서 format imitation은 빠르게 좋아지지만 helpfulness proxy는 느리게 움직이는 이유는 무엇인가?
- 다음 preference optimization 단계에서 SFT가 남긴 imitation bias를 어떻게 다시 평가할 것인가?

## 관련 이론
- [THEORY.md](./THEORY.md): instruction format, supervised fine-tuning, input-output template, role framing, imitation/helpfulness tradeoff를 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
