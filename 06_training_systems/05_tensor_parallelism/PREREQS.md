# 05 Tensor Parallelism 선행 개념

## 꼭 알고 오면 좋은 것
- tensor shape, matrix multiplication, hidden dimension이 무엇을 의미하는지에 대한 기본 감각
- Transformer의 attention head, QKV projection, feed-forward expansion이 어떤 큰 행렬곱으로 구현되는지에 대한 이해
- DDP와 sharding(FSDP/ZeRO류)이 각각 무엇을 복제하고 무엇을 나누는지에 대한 큰 그림
- all-reduce, all-gather, reduce-scatter 같은 collective communication이 왜 필요한지에 대한 아주 기본적인 감각
- GPU 메모리 병목과 interconnect bandwidth/latency가 step time에 영향을 준다는 운영 직관
- "레이어를 나눈다"와 "레이어 안의 텐서를 나눈다"를 다른 질문으로 볼 준비

## 먼저 다시 보면 좋은 단위
- [00_foundations/01_tensor_shapes](../../00_foundations/01_tensor_shapes/README.md) — 텐서 차원, reshape, matmul shape 감각 복습
- [00_foundations/05_gpu_memory_runtime](../../00_foundations/05_gpu_memory_runtime/README.md) — 메모리 병목과 장치 runtime 관찰 감각 복습
- [02_deep_learning/04_attention_and_transformers](../../02_deep_learning/04_attention_and_transformers/README.md) — attention/MLP 내부 차원 구조 복습
- [06_training_systems/01_torchrun_and_ddp_basics](../01_torchrun_and_ddp_basics/README.md) — rank/world size와 distributed collective 기본 그림 복습
- [06_training_systems/03_deepspeed_zero](../03_deepspeed_zero/README.md) — 상태 sharding과 communication trade-off 기준선 복습

## 빠른 자기 점검
- 큰 linear layer를 볼 때 input dimension, output dimension, intermediate dimension을 텐서 shape로 설명할 수 있는가?
- attention head를 여러 GPU에 나눈다는 말이 "batch를 나눈다"와 어떻게 다른지 설명할 수 있는가?
- row parallel과 column parallel이 모두 partial result를 만들지만, 왜 필요한 collective 시점이 서로 다를 수 있는지 직관적으로 말할 수 있는가?
- ZeRO/FSDP처럼 상태를 나누는 접근과 tensor parallel처럼 active compute를 나누는 접근의 차이를 구분할 수 있는가?
- 메모리를 줄이기 위해 tensor parallel을 도입했는데도 latency가 늘 수 있다는 점을 받아들일 준비가 되어 있는가?
