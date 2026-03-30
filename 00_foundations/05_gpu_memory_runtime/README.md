# 05 GPU Memory Runtime

## 왜 이 단위를 배우는가
GPU 메모리 감각이 없으면 batch size, mixed precision, gradient accumulation, OOM(out of memory) 원인을 설명하기 어렵다. 이 단위는 **무엇이 메모리를 차지하는지**와 **training/inference가 왜 다르게 보이는지**를 숫자로 확인하게 만든다.

## 이번 단위에서 남길 것
- scratch 계산 결과 `artifacts/scratch-manual/metrics.json`
- framework 관측 결과 `artifacts/framework-manual/metrics.json`
- 관측치를 한국어로 해석한 `analysis.md`
- 학습자가 직접 적는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 shape와 dtype만으로 필요한 바이트 수를 먼저 계산한다.
2. `framework_lab.py`에서 CPU 또는 CUDA 환경에서 inference/training runtime 차이를 관측한다.
3. `analysis.py`로 수치가 의미하는 바를 한국어 문장으로 정리한다.

## 다음 단위와의 연결
이 감각은 이후 tokenizer, attention, multimodal unit에서 batch size와 context length를 설계할 때 직접 쓰인다.
