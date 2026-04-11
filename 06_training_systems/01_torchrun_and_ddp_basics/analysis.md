# 01 Torchrun and DDP Basics 분석

## 해석 프레임
- DDP는 model parallel이 아니라 같은 모델 복사본들의 gradient를 평균내는 data parallel 방식이다.
- `rank`, `local_rank`, `world_size`는 코드가 여러 프로세스에서 실행될 때 각 프로세스의 위치를 설명한다.
- 이 단위의 toy metrics는 실제 multi-GPU 통신이 아니라 all-reduce의 산술 의미를 먼저 보여준다.

## 확인 질문
- rank별 gradient가 다른 이유는 무엇인가?
- averaged gradient가 parameter update에 어떻게 반영되는가?
- local rank는 device assignment와 어떻게 연결되는가?
