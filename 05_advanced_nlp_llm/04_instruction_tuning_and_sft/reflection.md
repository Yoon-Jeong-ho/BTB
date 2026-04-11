# 04 Instruction Tuning and SFT 회고

- 학습자 입장에서 **instruction format**을 단순한 프롬프트 꾸밈이 아니라 input-output template로 보려면 어떤 boundary를 먼저 표시해야 하는가?
- 같은 요청을 plain instruction-response template와 system/user/assistant chat template로 직렬화했을 때, 내가 보기에는 어느 쪽이 role framing을 더 명확히 드러내는가?
- assistant-only loss mask가 prompt/system/user 토큰을 무시하고 assistant response에만 supervision을 주는 이유를 내 말로 설명해 보라.
- system message가 “한국어로 간결하게 답하라” 같은 제약을 줄 때, 모델은 이것을 기억이 아니라 어떤 conditioning signal로 활용하는가?
- supervised fine-tuning으로 format imitation은 좋아졌지만 helpfulness가 충분하지 않은 사례를 하나 만들어 보라.
- reference answer를 그대로 모방하는 모델이 canned response나 과한 안전 문구를 배울 수 있다면, 내 데이터에서는 어떤 example을 줄이거나 바꾸겠는가?
- 다음 preference optimization 단계로 넘어가기 전, SFT 결과에서 무엇을 metrics로 남기고 무엇을 사람 평가 질문으로 남길 것인가?
