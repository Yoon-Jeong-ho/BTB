# 01 Tokenization and Embeddings 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 숫자가 조금 바뀌어도 유지되는 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 줄인다.

## 해석 프레임
- tokenizer는 문자열을 모델 vocabulary 안의 단위로 바꾸는 단계다. whitespace 단어 수와 subword token 수는 다를 수 있다.
- token id는 lookup key일 뿐이며, embedding table을 통과한 뒤에야 `(batch, seq, dim)` 표현이 생긴다.
- `[PAD]`는 길이 맞춤용 토큰이라서 실제 내용이 아니다. pooling이나 attention 전에는 반드시 mask로 구분해야 한다.
- `[UNK]`가 늘수록 표면 문자열 정보가 한 덩어리로 뭉개져, 모델이 세밀한 차이를 읽기 어려워진다.

## 확인 질문
- 왜 같은 한국어 문장도 whitespace 기준보다 subword 기준에서 더 긴 sequence가 될 수 있는가?
- `[UNK]`가 등장한 위치는 어떤 종류의 정보 손실을 뜻하는가?
- embedding lookup 이후 shape가 어떻게 바뀌고, padding mask가 없으면 어떤 계산이 왜곡되는가?

## 관련 이론
- [THEORY.md](./THEORY.md): tokenization, subword, embedding lookup, padding mask 핵심 개념을 다시 확인한다.
