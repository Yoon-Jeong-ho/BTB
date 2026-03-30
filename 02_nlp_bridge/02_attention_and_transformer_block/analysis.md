# 02 Attention and Transformer Block 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 attention과 transformer block을 읽는 안정적인 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- attention weight의 각 row 합이 1이라는 것은, query 위치 출력이 value들의 가중합이라는 뜻이다.
- self-attention은 각 토큰 표현을 다른 토큰 정보와 섞어 새 hidden state를 만든다. 그래서 output은 "원래 토큰 하나"가 아니라 sequence mixing 결과다.
- padding mask는 `[PAD]` 열을 가려서 빈 토큰을 참고하지 못하게 하고, causal mask는 미래 열을 가려서 아직 보이면 안 되는 정보를 차단한다.
- transformer block은 residual connection과 feed-forward를 거치면서 shape는 `(batch, seq, dim)`으로 유지하지만, 내부 좌표는 계속 갱신된다.

## 확인 질문
- attention output을 value들의 가중합이라고 말할 수 있는 근거는 무엇인가?
- padding mask와 causal mask는 각각 어떤 종류의 잘못된 정보 유입을 막는가?
- transformer block이 shape를 유지한다는 사실과, 표현이 달라진다는 사실은 어떻게 동시에 성립하는가?

## 관련 이론
- [THEORY.md](./THEORY.md): self-attention, mask, transformer block 핵심 개념을 다시 확인한다.
