# 01 Torchrun and DDP Basics

> Status: runnable

## 왜 이 단위를 배우는가
분산 학습을 처음 보면 `GPU가 여러 개`라는 사실보다 먼저 **같은 코드가 여러 프로세스에서 동시에 실행된다**는 점이 낯설다. `torchrun`은 이 여러 프로세스가 몇 개인지, 각 프로세스의 rank가 무엇인지, 같은 장비 안 local rank가 무엇인지 알려 주는 실행 계약을 만든다. 먼저 CPU-safe simulation으로 직관을 잡고, 이후 선택 실습에서 실제 `torch.distributed` 2-process all-reduce를 확인한다.

## 이번 단위에서 남길 것
- scratch 실행 결과 `artifacts/scratch-manual/metrics.json`
- rank별 gradient SVG `artifacts/scratch-manual/rank_gradients.svg`
- framework 관측 결과 `artifacts/framework-manual/metrics.json`
- 실행별 관측 리포트 `artifacts/analysis-manual/latest_report.md`
- 안정적인 해석 문서 `analysis.md`
- 학습자 회고 `reflection.md`

## 실습 흐름
1. `scratch_lab.py`에서 4개 rank와 2개 local rank를 손으로 구성한다.
2. rank별 local gradient를 평균내며 DDP all-reduce가 왜 같은 update를 만들려고 하는지 본다.
3. `framework_lab.py`에서 같은 계산을 tiny tensor/numeric flow로 다시 확인한다.
4. `analysis.py`로 rank mapping, gradient average, sync 후 parameter update를 한국어로 해석한다.
5. 선택: 아래 명령으로 실제 2-process `gloo` process group과 all-reduce를 실행한다.

```bash
BTB_DEVICE=cpu python -m torch.distributed.run --standalone --nproc-per-node=2 \
  06_training_systems/01_torchrun_and_ddp_basics/torchrun_lab.py
```

성공 시 `artifacts/torchrun-manual/metrics.json`에 관측된 rank, world size, all-reduce 평균이 남는다. 이 경로는 PyTorch distributed가 없는 환경에서는 선택 항목이며, 기본 scratch/framework 실습은 계속 CPU-safe다.

## 이 단위에서 특히 볼 질문
- global rank와 local rank는 왜 둘 다 필요한가?
- world size가 커지면 batch와 gradient 평균을 어떻게 해석해야 하는가?
- DDP는 모델을 쪼개는가, 아니면 같은 모델 복사본들의 gradient를 맞추는가?
- rank별 gradient가 다를 때 all-reduce 이후 무엇이 같아지는가?

## 실행 결과 예시
```text
$ python 06_training_systems/01_torchrun_and_ddp_basics/scratch_lab.py
{
  "world_size": 4,
  "local_world_size": 2,
  "averaged_gradient": 1.025,
  "max_gradient_deviation": 0.225,
  "figure_path": "artifacts/scratch-manual/rank_gradients.svg"
}

$ python 06_training_systems/01_torchrun_and_ddp_basics/framework_lab.py
{
  "backend": "cpu-simulated-ddp",
  "world_size": 4,
  "parameter_before": 2.0,
  "parameter_after": 1.8975,
  "all_ranks_share_update": true
}
```

## 다음 단위와의 연결
이 단위에서 rank/world-size/all-reduce 계약을 잡아 두면, `02_accelerate_workflows`에서 Accelerate가 무엇을 숨겨 주는지, `03_deepspeed_zero`에서 무엇을 새로 쪼개는지 더 분명하게 보인다.
