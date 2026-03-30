# 05 GPU Memory Runtime 이론 노트

## VRAM을 차지하는 주요 항목
- **parameter**: 모델 가중치 자체
- **activation**: forward 중간 결과. backward를 위해 training에서 오래 살아남는다.
- **gradient**: parameter마다 쌓이는 미분 값
- **optimizer state**: Adam 같은 optimizer가 따로 유지하는 통계량

## 왜 training이 inference보다 무거운가
- inference는 보통 `no_grad`로 실행되어 activation을 오래 붙잡지 않는다.
- training은 backward를 위해 activation을 저장하고, gradient와 optimizer state까지 추가로 필요하다.
- 따라서 같은 모델과 batch라도 training 쪽이 메모리와 시간 모두 더 비싸게 측정되는 경우가 많다.

## 관측할 때 볼 항목
- `dtype`: fp32 / fp16 / bf16에 따라 같은 shape라도 바이트 수가 달라진다.
- `device`: cuda면 allocator 관측이 가능하고, cpu면 runtime과 gradient 크기 같은 proxy를 본다.
- `max_memory_allocated`: 실제 tensor allocation의 peak
- `max_memory_reserved`: CUDA allocator가 잡아둔 reserve 크기

## Common Confusion
- “모델이 작으니 OOM이 안 난다”라고 단정하는 실수
- parameter만 세고 activation/gradient/optimizer state를 빼먹는 실수
- training과 inference 숫자를 같은 기준으로 비교하지 않는 실수
