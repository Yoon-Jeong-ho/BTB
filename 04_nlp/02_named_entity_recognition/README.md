# 02 Named Entity Recognition

## 왜 이 단위를 배우는가
`03_nlp/01_text_classification`에서 문장 전체를 하나의 라벨로 보내 봤다면, 이제는 **문장 안의 어떤 span이 사람/기관/장소인지 토큰 단위로 찍는 applied NLP 감각**이 필요하다. 이 단위는 작은 한국어 예제로 `BIO alignment 확인 -> token baseline -> tiny PyTorch sequence labeler -> 분석 보고서` 흐름을 직접 돌리게 해서, 이후 KLUE-NER나 CoNLL-2003으로 넘어갈 때도 바로 거대한 모델부터 잡지 않고 **경계(boundary)와 alignment를 먼저 읽는 습관**을 만들도록 설계했다.

## 이번 단위에서 남길 것
- scratch 실험으로 만든 `artifacts/scratch-manual/metrics.json`
- scratch figure `artifacts/scratch-manual/label_distribution.svg`
- framework 실험으로 만든 `artifacts/framework-manual/metrics.json`
- 실행별 관측치를 적는 `artifacts/analysis-manual/latest_report.md`
- 안정적인 해석 프레임을 담은 `analysis.md`
- 학습자가 직접 적는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 toy word-piece tokenizer와 BIO alignment를 직접 확인한다.
2. 같은 aligned token 시퀀스를 이용해 token별 다수결 baseline을 만든 뒤 token accuracy / entity precision / entity recall / entity F1을 계산한다.
3. `framework_lab.py`에서 tiny PyTorch biGRU sequence labeler를 CPU에서 학습시키고, 같은 지표를 다시 비교한다.
4. `analysis.py`로 boundary error, alignment 이슈, scratch 대비 framework 차이를 한국어 문장으로 정리한다.

## 이 단위에서 특히 볼 질문
- word-level gold label을 subword token에 맞출 때 왜 `B-` 와 `I-` 규칙이 중요할까?
- token accuracy가 높아도 entity-level F1이 낮아질 수 있는 이유는 무엇일까?
- sequence labeling에서는 왜 span 경계를 틀리는 순간 여러 token이 연쇄적으로 무너질까?
- tiny neural model이 baseline보다 나아 보일 때, 그 차이는 context를 읽었기 때문인가 alignment leakage 때문인가?

## 실행 방법
```bash
python 03_nlp/02_named_entity_recognition/scratch_lab.py
python 03_nlp/02_named_entity_recognition/framework_lab.py
python 03_nlp/02_named_entity_recognition/analysis.py
```

## 실행 결과 예시
아래 숫자는 이 toy NER unit에서 기대하는 출력 형식의 예시다. 실제 값은 seed와 학습 경로에 따라 조금 달라질 수 있지만, **metrics.json + svg + analysis report** 조합은 유지된다.

```json
{
  "train_size": 8,
  "eval_size": 4,
  "aligned_train_tokens": 75,
  "aligned_eval_tokens": 38,
  "token_accuracy": 0.857143,
  "entity_precision": 0.8,
  "entity_recall": 0.8,
  "entity_f1": 0.8,
  "figure_path": "artifacts/scratch-manual/label_distribution.svg"
}
```

```json
{
  "train_size": 8,
  "eval_size": 4,
  "vocab_size": 63,
  "num_labels": 7,
  "token_accuracy": 0.892857,
  "entity_f1": 0.857143,
  "train_input_shape": [8, 10]
}
```

## 무엇을 읽고 다음 단계로 넘어가면 좋은가
1. [PREREQS.md](./PREREQS.md) — BIO tag, tokenization, sequence shape를 빠르게 점검한다.
2. [THEORY.md](./THEORY.md) — NER에서 alignment와 entity-level metric이 왜 중요한지 먼저 읽는다.
3. `scratch_lab.py` 출력과 `label_distribution.svg` — alignment와 label 분포를 눈으로 확인한다.
4. `framework_lab.py` 출력 — 같은 문제를 tiny sequence labeler가 어떻게 다시 읽는지 확인한다.
5. `analysis.py`와 `analysis.md` — 숫자를 boundary 해석 문장으로 바꾸는 연습을 한다.

## 다음 단위와의 연결
이 감각이 있으면 KLUE-NER, CoNLL-2003, 또는 문서형 IE 실습으로 넘어갈 때도 먼저 **alignment sanity check -> token/entity metric 비교 -> boundary error 해석** 순서를 밟게 된다. 그 습관이 있어야 transformer token classification도 덜 블랙박스처럼 보인다.
