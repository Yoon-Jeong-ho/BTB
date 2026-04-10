# 02 Corpus, Tokenizer, and Data Mixture 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라질 수 있는 toy corpus 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 corpus quality, tokenizer tradeoff, dedup/contamination, mixture budget을 해석하는 **안정적인 프레임**만 남긴다.
- 따라서 `analysis.py`를 반복 실행해도 이 파일은 같은 내용으로 유지되고, 관측 보고서만 최신 값으로 갱신된다.

## 해석 프레임
- corpus quality는 문서 수가 아니라 **사용 가능한 학습 신호**다. exact/near duplicate와 contamination을 제거하면 raw 문서보다 trainable 문서가 줄어든다.
- tokenizer는 compression과 fairness를 동시에 바꾼다. toy aggressive subword가 더 많은 token을 만들면 같은 context window에서 담을 수 있는 의미 단위가 줄어든다.
- dedup은 train corpus 내부 반복 노출을 줄이고, contamination check는 evaluation overlap 때문에 성능 추정이 부풀려지는 것을 막는다. 둘은 함께 필요하지만 같은 검사가 아니다.
- mixture는 문서 수보다 token budget 기준으로 읽어야 한다. 언어/도메인별 평균 token 길이가 다르면 같은 문서 수라도 gradient 신호 배분이 달라진다.

## 확인 질문
- raw corpus에서 실제 trainable corpus까지 어떤 문서가 빠졌고, 그 이유는 dedup인가 contamination인가?
- tokenizer를 더 잘게 만들면 multilingual mixture에서 어떤 언어가 더 많은 token budget을 쓰게 되는가?
- domain balance를 문서 수가 아니라 token share로 보면 어떤 slice가 과대표/과소대표되는가?
- 작은 고품질 도메인 slice를 oversample할 때 contamination guard를 어디에 둬야 하는가?

## 관련 이론
- [THEORY.md](./THEORY.md): corpus quality, tokenizer tradeoff, dedup/contamination, multilingual mixture, token budget을 다시 확인한다.
- 실행별 최신 관측은 `artifacts/analysis-manual/latest_report.md`를 본다.
