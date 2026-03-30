# 01 Tensor Shapes

## 왜 이 단위를 배우는가
텐서 shape를 읽지 못하면 backpropagation, attention, GPU 메모리 분석까지 모두 흐려진다. 이 단위는 `shape를 먼저 읽고, 그 다음 연산을 읽는 습관`을 만드는 출발점이다.

## 이번 단위에서 남길 것
- scratch 실험으로 만든 `artifacts/scratch-manual/metrics.json`
- framework 실험으로 만든 `artifacts/framework-manual/metrics.json`
- 결과를 한국어로 해석한 `analysis.md`
- 학습자가 직접 채우는 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 NumPy로 shape와 matmul을 직접 확인한다.
2. `framework_lab.py`에서 PyTorch `Linear` layer 출력 shape를 확인한다.
3. `analysis.py`로 관측치를 한국어 해설 문서로 정리한다.

## 다음 단위와의 연결
이후 activation, loss, attention, GPU runtime 단위에서 모두 shape 해석이 기본 전제가 된다.
