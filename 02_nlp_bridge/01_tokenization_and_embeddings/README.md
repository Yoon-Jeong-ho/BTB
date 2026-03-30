# 01 Tokenization and Embeddings

## 왜 이 단위를 배우는가
`03_nlp`로 들어가면 tokenizer, input ids, attention mask, embedding layer가 너무 빨리 등장한다. 이 단위는 **문장이 어떻게 잘게 쪼개져 정수 id가 되고, 그 id가 다시 dense vector로 바뀌는지**를 작은 한국어 예제로 먼저 눈에 보이게 만든다.

## 이번 단위에서 남길 것
- scratch 실험으로 만든 `artifacts/scratch-manual/metrics.json`
- framework 실험으로 만든 `artifacts/framework-manual/metrics.json`
- 실행별 관측치를 적는 `artifacts/analysis-manual/latest_report.md`
- 안정적인 해석 프레임을 담은 `analysis.md`
- 학습자가 직접 채우는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 toy vocab으로 한국어 문장을 subword-ish하게 분해하고 token id로 바꾼다.
2. `framework_lab.py`에서 PyTorch `Embedding`과 padding mask shape를 확인한다.
3. `analysis.py`로 길이 증가, `[UNK]`, embedding lookup, padding mask 의미를 한국어 문장으로 정리한다.

## 다음 단위와의 연결
이 감각이 있어야 `03_nlp`의 텍스트 분류/NER/MRC에서 tokenizer 길이 budget, attention mask, pretrained embedding 입력 형식을 자연스럽게 읽을 수 있다.
