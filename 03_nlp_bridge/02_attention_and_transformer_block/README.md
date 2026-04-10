# 02 Attention and Transformer Block

## 왜 이 단위를 배우는가
`01_tokenization_and_embeddings`에서 문장이 token id와 embedding tensor로 바뀌는 흐름을 봤다면, 이제는 **각 토큰이 서로를 얼마나 참고해서 표현을 섞는지**를 봐야 한다. 이 단위는 attention weight, sequence mixing, mask, transformer block의 shape 보존을 작은 한국어 예제로 연결한다.

## 이번 단위에서 남길 것
- scratch 실험으로 만든 `artifacts/scratch-manual/metrics.json`
- framework 실험으로 만든 `artifacts/framework-manual/metrics.json`
- 실행별 관측치를 적는 `artifacts/analysis-manual/latest_report.md`
- 안정적인 해석 프레임을 담은 `analysis.md`
- 학습자가 직접 적는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 손으로 만든 query/key/value 벡터로 attention score, softmax weight, weighted sum이 어떻게 생기는지 계산한다.
2. `framework_lab.py`에서 PyTorch `MultiheadAttention`과 간단한 transformer-block-style 연산으로 `(batch, seq, dim)` shape가 어떻게 유지되는지 확인한다.
3. `analysis.py`로 attention이 왜 "토큰 혼합"인지, padding/causal mask가 무엇을 막는지, residual + FFN이 어떤 역할을 하는지 한국어로 정리한다.

## 이번 단위에서 특히 볼 질문
- 한 토큰의 attention weight가 다른 토큰으로 얼마나 분산되는가?
- attention output이 왜 "원래 토큰 표현의 복사"가 아니라 가중합된 혼합 표현인가?
- padding mask와 causal mask가 없으면 어떤 위치가 잘못 섞이는가?
- transformer block은 shape를 유지하면서 내부 표현을 어떻게 바꾸는가?

## 다음 단위와의 연결
이 감각이 있으면 `03_nlp`에서 BERT류 self-attention, decoder causal mask, encoder block hidden state 업데이트를 더 이상 마법처럼 보지 않게 된다.
