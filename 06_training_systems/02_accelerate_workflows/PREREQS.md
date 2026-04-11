# 02 Accelerate Workflows 선행 개념

## 꼭 알고 오면 좋은 것
- `06_training_systems/01_torchrun_and_ddp_basics`의 rank/world-size 감각
- PyTorch training loop의 model / optimizer / dataloader / scheduler 구조
- device placement와 mixed precision의 목적

## 빠른 자기 점검
- baseline training loop에서 device 이동이 어디에 들어가는지 찾을 수 있는가?
- distributed launch와 model code가 왜 서로 영향을 주는지 설명할 수 있는가?
- Accelerate가 감추는 것과 여전히 남는 것을 구분할 준비가 되었는가?
