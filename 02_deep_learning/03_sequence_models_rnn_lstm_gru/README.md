# 03 Sequence Models: RNN, LSTM, GRU

> Status: runnable
> CPU-safe toy 실습으로 **hidden state 직관, 순서 민감성, vanilla RNN vs LSTM/GRU, teacher forcing gap** 을 직접 관찰하는 단위다.

## 왜 이 단위를 배우는가
순서가 있는 데이터에서는 **무슨 항목이 있었는가** 만큼이나 **어떤 순서로 들어왔는가** 가 중요하다. 이 단위는 hidden state가 시간축 정보를 어떻게 압축하는지, vanilla RNN이 왜 긴 문맥에서 흔들리는지, 그리고 LSTM/GRU의 gate가 어떤 문제를 완화하려고 나왔는지를 한 흐름으로 묶는다. 이후 transformer를 볼 때도 "왜 recurrence를 벗어나려 했는가"를 감각적으로 이해하게 만드는 중간 다리 역할을 한다.

## 이번 단위에서 남길 것
- scratch 계산 결과 `artifacts/scratch-manual/metrics.json`
- scratch figure `artifacts/scratch-manual/hidden_state_diagnostics.svg`
- framework 관측 결과 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 반복 실행에도 안정적으로 유지할 `analysis.md`
- 학습자가 직접 채우는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 손으로 고정한 recurrent update를 굴려, 같은 token 집합이어도 순서가 달라지면 hidden trajectory가 달라진다는 사실을 본다.
2. 같은 scratch 실험에서 첫 토큰만 다른 long-context pair를 비교해, vanilla RNN보다 LSTM/GRU가 초반 신호를 더 오래 붙잡는 방향을 확인한다.
3. 작은 decoder transition table로 teacher forcing과 free running loss를 비교해, 학습/추론 간 mismatch가 어떻게 생기는지 본다.
4. `framework_lab.py`에서 tiny PyTorch `RNN/LSTM/GRU` 와 GRU decoder를 CPU에서 다시 돌려 같은 질문을 프레임워크 관점으로 재현한다.
5. `analysis.py`로 실행별 숫자를 한국어 문장으로 정리하고, 안정적인 해석 문서와 실행별 리포트를 분리한다.

## 실행 방법
```bash
python 02_deep_learning/03_sequence_models_rnn_lstm_gru/scratch_lab.py
python 02_deep_learning/03_sequence_models_rnn_lstm_gru/framework_lab.py
python 02_deep_learning/03_sequence_models_rnn_lstm_gru/analysis.py
```

## 실행 결과 예시
실제 실행 후에는 JSON metrics, SVG figure, 관측 리포트가 모두 `artifacts/` 아래에 남는다.

```text
$ python 02_deep_learning/03_sequence_models_rnn_lstm_gru/scratch_lab.py
{
  "rnn_order_cosine_gap": 0.306428,
  "lstm_order_cosine_gap": 0.012555,
  "gru_order_cosine_gap": 0.013862,
  "rnn_long_range_signal": 0.056409,
  "lstm_long_range_signal": 0.640622,
  "gru_long_range_signal": 0.1205,
  "teacher_forcing_loss": 0.260032,
  "free_running_loss": 1.869828,
  "teacher_forcing_gap": 1.609796,
  "figure_path": "artifacts/scratch-manual/hidden_state_diagnostics.svg"
}

$ python 02_deep_learning/03_sequence_models_rnn_lstm_gru/framework_lab.py
{
  "device": "cpu",
  "hidden_shapes": {
    "rnn": [1, 2, 3],
    "lstm_h": [1, 2, 3],
    "lstm_c": [1, 2, 3],
    "gru": [1, 2, 3]
  },
  "rnn_long_range_signal": 0.056409,
  "lstm_long_range_signal": 0.640622,
  "gru_long_range_signal": 0.1205,
  "teacher_forcing_loss": 0.196735,
  "free_running_loss": 0.481993,
  "teacher_forcing_gap": 0.285259,
  "decoder_logits_shape": [2, 4, 6]
}
```

`hidden_state_diagnostics.svg` 를 열어 보면 **vanilla RNN hidden trajectory**, **RNN/LSTM/GRU long-range retention bar**, **teacher forcing vs free running loss bar** 가 한 화면에 모여 있어 순서 민감성과 gating 직관을 빠르게 훑을 수 있다.

## 이 단위에서 특히 볼 질문
- hidden state는 과거 정보를 어떻게 요약하고, 무엇을 잃어버리기 쉬운가?
- 같은 token 집합이라도 순서를 바꾸면 왜 final state가 달라지는가?
- vanilla RNN보다 LSTM/GRU가 long-range signal을 더 많이 남기는 이유를 gate 관점에서 어떻게 설명할 수 있는가?
- teacher forcing은 학습을 안정화하지만 inference에서 어떤 mismatch를 남기는가?
- transformer는 왜 recurrence 대신 attention으로 과거 위치를 직접 참조하려 했는가?

## 무엇을 읽고 다음 단계로 넘어가면 좋은가
1. [PREREQS.md](./PREREQS.md) — recurrent shape, BOS/EOS, cross entropy 감각을 먼저 점검한다.
2. [THEORY.md](./THEORY.md) — hidden state, vanishing gradient, gating, teacher forcing를 한 번에 정리한다.
3. `scratch_lab.py` 출력과 `hidden_state_diagnostics.svg` — 손계산 기반 직관을 먼저 본다.
4. `framework_lab.py` 출력 — 같은 질문을 tiny PyTorch component가 어떻게 재현하는지 확인한다.
5. `analysis.py`와 `analysis.md` — 숫자를 해석 문장으로 바꾸는 연습을 한다.

## 다음 단위와의 연결
이 단위에서 recurrent hidden state의 병목과 gated memory의 장단점을 체감하면, 다음 단위 `02_deep_learning/04_attention_and_transformers`에서 self-attention이 왜 각 위치를 직접 보게 만드는지 더 자연스럽게 받아들일 수 있다. 다시 말해, 이 단위는 transformer를 배우기 전에 "왜 RNN만으로는 답답했는가"를 몸으로 확인하는 준비 단계다.
