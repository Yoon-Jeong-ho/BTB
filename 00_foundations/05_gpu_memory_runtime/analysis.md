# 05 GPU Memory Runtime 분석

## 이 문서를 어떻게 읽을까
- 실행할 때마다 달라지는 관측치는 `artifacts/analysis-manual/latest_report.md`에 기록한다.
- 이 문서는 숫자가 바뀌어도 유지되는 해석 프레임만 남겨, 반복 실행 시 불필요한 diff를 만들지 않도록 한다.

## 해석 프레임
- 같은 shape라도 dtype(fp32 / fp16 / bf16)가 바뀌면 메모리 budget이 바로 달라진다.
- training은 activation 보관, gradient 계산, 추가 상태 때문에 inference보다 더 무거워지는 방향으로 읽는다.
- CUDA가 없더라도 output/gradient 바이트와 runtime 차이를 proxy로 보면 training cost를 설명할 수 있다.

## 확인 질문
- batch size를 키우면 어떤 항목이 함께 증가하는가?
- training과 inference를 비교할 때 parameter만 보면 왜 부족한가?
- 이번 실행에서 관측한 구체적 숫자는 `artifacts/analysis-manual/latest_report.md`에 어떻게 정리되었는가?

## 관련 이론
- [THEORY.md](./THEORY.md): VRAM 구성 요소, mixed precision, training vs inference 차이를 다시 확인한다.
