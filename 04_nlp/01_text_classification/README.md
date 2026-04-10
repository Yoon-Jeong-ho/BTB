# 01 Text Classification

## 왜 이 단위를 배우는가
`02_nlp_bridge`에서 tokenization, embedding, attention을 미리 손으로 만져봤다면 이제는 **문장을 실제 라벨 문제로 연결하는 첫 applied NLP 단계**가 필요하다. 이 단위는 작은 한국어 예제로 `bag-of-words baseline -> tiny PyTorch classifier -> 분석 보고서` 흐름을 한 번에 돌려 보게 해서, 이후 감성 분류/토픽 분류/리뷰 분류 실습이 덜 막막하게 느껴지도록 설계했다.

## 이번 단위에서 남길 것
- scratch 실험으로 만든 `artifacts/scratch-manual/metrics.json`
- scratch figure `artifacts/scratch-manual/token_signal.svg`
- framework 실험으로 만든 `artifacts/framework-manual/metrics.json`
- 실행별 관측치를 적는 `artifacts/analysis-manual/latest_report.md`
- 안정적인 해석 프레임을 담은 `analysis.md`
- 학습자가 직접 적는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 whitespace token 기준의 tiny bag-of-words / multinomial Naive Bayes baseline을 돌린다.
2. 같은 toy 데이터셋을 `framework_lab.py`에서 PyTorch embedding-average classifier로 다시 학습한다.
3. `analysis.py`로 baseline과 framework가 각각 무엇을 잘하고, 어디서 헷갈리는지 한국어 문장으로 다시 정리한다.

## 이 단위에서 특히 볼 질문
- 텍스트 분류에서 입력 문장을 숫자로 바꾸는 가장 단순한 방법은 무엇인가?
- token count 기반 baseline은 왜 아직도 강한 출발점인가?
- accuracy와 macro F1을 함께 봐야 하는 이유는 무엇인가?
- tiny neural classifier가 baseline보다 좋아 보일 때, 그 차이는 representation 때문인가 data leakage 때문인가?

## 실행 방법
```bash
python 03_nlp/01_text_classification/scratch_lab.py
python 03_nlp/01_text_classification/framework_lab.py
python 03_nlp/01_text_classification/analysis.py
```

## 실행 결과 예시
아래 숫자는 이 toy unit에서 기대하는 출력 형식의 예시다. 실제 값은 seed나 학습 경로에 따라 조금 달라질 수 있지만, **metrics.json + svg + analysis report** 조합은 유지된다.

```json
{
  "train_size": 8,
  "eval_size": 4,
  "vocab_size": 31,
  "eval_accuracy": 0.75,
  "eval_macro_f1": 0.733333,
  "figure_path": "artifacts/scratch-manual/token_signal.svg"
}
```

```json
{
  "train_size": 8,
  "eval_size": 4,
  "num_classes": 2,
  "vocab_size": 33,
  "eval_accuracy": 0.75,
  "eval_macro_f1": 0.733333
}
```

## 무엇을 읽고 다음 단계로 넘어가면 좋은가
1. [PREREQS.md](./PREREQS.md) — bag-of-words, 확률, tensor shape가 익숙한지 점검한다.
2. [THEORY.md](./THEORY.md) — text classification의 최소 개념 세트를 먼저 읽는다.
3. `scratch_lab.py` 출력과 `token_signal.svg` — token count 기반 분류 감각을 먼저 잡는다.
4. `framework_lab.py` 출력 — 같은 문제를 neural classifier가 어떻게 읽는지 확인한다.
5. `analysis.py`와 `analysis.md` — 숫자를 해석 문장으로 바꾸는 연습을 한다.

## 다음 단위와의 연결
이 감각이 있으면 이후 실제 데이터셋(NSMC, IMDb, YNAT 등)으로 넘어갈 때도 바로 거대한 pretrained model부터 붙잡지 않고, **baseline을 먼저 세우고 비교 질문을 만든 뒤 모델을 해석하는 습관**을 유지할 수 있다.
