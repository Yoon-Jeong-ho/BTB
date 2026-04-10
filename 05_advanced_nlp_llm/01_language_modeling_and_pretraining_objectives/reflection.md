# 01 Language Modeling and Pretraining Objectives 학습자 회고

- 내가 causal LM을 “다음 토큰 prediction”이라고 설명할 때, 입력과 정답을 정확히 어떻게 그릴 수 있는가?
- masked LM에서 loss-mask density가 낮은데도 왜 유용한 objective가 될 수 있는지 내 말로 다시 설명해 보라.
- span corruption에서 sentinel token `<extra_id_0>`와 `<extra_id_1>`은 어떤 bookkeeping 역할을 하는가?
- 같은 context window에서도 causal LM, masked LM, span corruption의 visible context가 다르다는 사실이 가장 선명하게 느껴진 순간은 언제였는가?
- future token을 못 보는 규칙과 양쪽 문맥을 보는 규칙이 이후 model behavior intuition을 어떻게 다르게 만들까?
- 다음 단위에서 tokenizer/data mixture를 볼 때, 나는 어떤 objective별 질문을 먼저 던질 것인가?
