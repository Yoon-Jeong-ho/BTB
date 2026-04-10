# 03 Sequence Models: RNN, LSTM, GRU

> Status: outlined
> 이 단위는 현재 문서/메타데이터만 정리된 outlined 단계다. 아래 실습 흐름과 출력 예시는 **후속 applied 단계에서 구현될 예상 모습** 이며, 아직 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`는 없다.

## 왜 이 단위를 배우는가
순서가 있는 데이터에서는 **무슨 항목이 있었는가** 만큼이나 **어떤 순서로 들어왔는가** 가 중요하다. 이 단위는 hidden state가 시간축 정보를 어떻게 압축하는지, vanilla RNN이 왜 긴 문맥에서 흔들리는지, 그리고 LSTM/GRU의 gate가 어떤 문제를 완화하려고 나왔는지를 한 흐름으로 묶는다. 이후 transformer를 볼 때도 "왜 recurrence를 벗어나려 했는가"를 감각적으로 이해하게 만드는 중간 다리 역할을 한다.

## 이번 단위에서 남길 것
- 학습 목표와 후속 실습 방향을 정리한 `README.md`
- hidden state, gating, teacher forcing를 연결한 `THEORY.md`
- 선행 개념 체크리스트 `PREREQS.md`
- outlined 단계 메타데이터를 담은 `lesson.yaml`
- 후속 실습 산출물이 들어갈 자리만 먼저 만든 `artifacts/.gitkeep`
- 후속 applied 단계에서 채울 예정인 `scratch_lab.py`, `framework_lab.py`, `analysis.md`, `reflection.md`에 대한 명시적 빈자리

## 실습 흐름
1. 작은 시퀀스를 시간축으로 읽으며 `h_t = f(x_t, h_{t-1})` 형태의 recurrent update가 왜 순서를 보존하는지 본다.
2. 같은 token 집합이라도 순서를 바꾸면 hidden state trajectory와 final state가 달라지는지 관찰한다.
3. vanilla RNN에서 멀리 떨어진 정보가 왜 점점 약해지는지 repeated multiplication과 gradient 전달 관점으로 연결한다.
4. LSTM의 forget/input/output gate와 cell state, GRU의 update/reset gate를 비교하면서 "무엇을 유지하고 무엇을 덮어쓸지"의 직관을 잡는다.
5. sequence modeling에서 입력과 정답을 한 칸씩 shift해 teacher forcing을 거는 이유를 next-token prediction 관점에서 정리한다.
6. 마지막에는 "모든 과거를 hidden state 하나에 눌러 담는 구조가 왜 병목인가"를 질문으로 남기며 `02_deep_learning/04_attention_and_transformers`로 넘어간다.

## 이 단위에서 특히 볼 질문
- hidden state는 과거 정보를 어떻게 요약하고, 무엇을 잃어버리기 쉬운가?
- 순서만 바뀐 두 시퀀스가 왜 다른 state trajectory를 만들 수 있는가?
- vanilla RNN은 왜 long-range dependency에서 gradient가 약해지거나 폭주하기 쉬운가?
- LSTM/GRU의 gate는 각각 어떤 실패를 완화하려고 도입되었는가?
- teacher forcing은 학습을 안정화하지만 inference에서 어떤 mismatch를 남기는가?
- transformer는 왜 recurrence 대신 attention으로 과거 위치를 직접 참조하려 했는가?

## 실행 결과 예시
아래는 **아직 완료된 실행 결과가 아니라**, 후속 applied 단계에서 기대하는 출력 형태 예시다.

```text
# expected output / sample shape only
$ python 02_deep_learning/03_sequence_models_rnn_lstm_gru/scratch_lab.py
{
  "sequence_pairs": [
    {"input": ["A", "B", "C"], "target": ["B", "C", "<eos>"]},
    {"input": ["A", "C", "B"], "target": ["C", "B", "<eos>"]}
  ],
  "final_hidden_cosine_gap": 0.41,
  "vanilla_rnn_long_range_signal": 0.07,
  "lstm_memory_retention": 0.63,
  "gru_update_gate_mean": 0.54
}

$ python 02_deep_learning/03_sequence_models_rnn_lstm_gru/framework_lab.py
{
  "teacher_forcing_loss": 1.84,
  "free_running_loss": 2.31,
  "logit_shape": [2, 5, 12],
  "hidden_shape": {
    "rnn": [1, 2, 16],
    "lstm_h": [1, 2, 16],
    "lstm_c": [1, 2, 16],
    "gru": [1, 2, 16]
  }
}
```

핵심은 숫자 자체보다도 **순서 변화에 따른 hidden state 차이**, **vanilla RNN과 gated unit의 long-range signal 차이**, **teacher forcing 유무에 따른 손실 간격** 을 읽는 것이다.

## 다음 단위와의 연결
이 단위에서 recurrent hidden state의 병목과 gated memory의 장단점을 체감하면, 다음 단위 `02_deep_learning/04_attention_and_transformers`에서 self-attention이 왜 각 위치를 직접 보게 만드는지 더 자연스럽게 받아들일 수 있다. 다시 말해, 이 단위는 transformer를 배우기 전에 "왜 RNN만으로는 답답했는가"를 몸으로 확인하는 준비 단계다.
