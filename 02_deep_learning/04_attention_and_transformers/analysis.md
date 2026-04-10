# 04 Attention and Transformers 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 attention과 transformer를 해석하는 **안정적인 프레임**만 남겨, 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- attention row 합이 1이라는 것은, 각 query 출력이 value들의 **가중합**이라는 뜻이다. 그래서 attention은 sequence mixing으로 읽는 것이 가장 자연스럽다.
- multi-head의 핵심은 head 수 자체가 아니라, **서로 다른 mixing 관점**이 병렬로 놓인다는 점이다.
- encoder self-attention은 전체 문맥을 볼 수 있지만, decoder self-attention은 causal mask 때문에 미래를 보지 못한다. encoder-decoder 구조는 여기에 cross-attention이 더해져 입력 memory를 읽는다.
- transformer는 recurrent hidden chain 대신 직접 참조를 허용해 long-range information path를 줄이지만, attention matrix 길이 비용은 여전히 남는다.

## 확인 질문
- attention output을 왜 “토큰 선택”이 아니라 value mixing 결과라고 말할 수 있는가?
- multi-head에서 서로 다른 top key가 나온다면, 그것은 어떤 관점 차이를 시사하는가?
- encoder와 decoder의 가장 큰 차이를 mask 관점에서 어떻게 설명할 수 있는가?
- recurrent bottleneck relief와 길이 제곱 비용 trade-off를 동시에 어떻게 요약할 수 있는가?

## 관련 이론
- [THEORY.md](./THEORY.md): sequence mixing, multi-head intuition, encoder/decoder distinction, recurrent bottleneck relief를 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
